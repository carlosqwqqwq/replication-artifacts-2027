import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .direct_case import ADDRESS_MODEL, MEMORY_BASE_REGISTER, OBSERVATION_CONTRACT, OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC, TestCase, _memory_region_layout, expected_trap_from_dataflow, testcase_uses_fp_instruction, testcase_uses_vector_instruction
from .execution_environment import require_execution_plane
from ._util import canonical_digest, is_riscv_elf, is_sha256_digest, sha256_file
from .spec_definedness import enabled_extensions, isa_base_extensions, profile_requires_fp_state


DEFAULT_CC = "riscv64-linux-gnu-gcc"
DEFAULT_NM = "riscv64-linux-gnu-nm"
BUILD_TIMEOUT_SEC = 60
TEXT_ADDRESS = "0x10000"


def _frame_contract_digest(frame: dict) -> str:
    return canonical_digest({
        key: value for key, value in frame.items()
        if key not in {"testcase_id", "testcase_code_sha256", "frame_contract_sha256"}
    })

# binutils 2.42 (Ubuntu 24.04 / gcc 13.3.0, Docker x86_64-lab 2026-08-13 全量
# 探测 94 个 catalog suffix token) 不认识的 prefixed 扩展。Raw testcase 指令
# 以 .word/.2byte 原始字节发射，march 只影响 harness prologue 的汇编，因此
# 从工具链身份中剔除这些扩展只改变构建身份，不改变语义身份（isa_profile
# 单独记录在 build_contract）。zicbo 家族在下方展开为 zicbom/zicbop/zicboz
# 后再参与裁剪；zvl* 需要 v/zve 前置，profile 含 v 时工具链接受，不裁剪。
_TOOLCHAIN_UNKNOWN_SUFFIXES = frozenset(
    {
        "zabha",
        "zabhlrsc",
        "zaamo",
        "zacas",
        "zalasr",
        "zalrsc",
        "zbp",
        "zcmop",
        "zcmp",
        "zcmt",
        "zfbfmin",
        "zibi",
        "zicfiss",
        "zimop",
        "sdext",
        "smrnmi",
        "ssctr",
        "svinval_h",
        "zvabd",
        "zvdot4a",
        "zvfbdota32f",
        "zvfbfmin",
        "zvfbfwma",
        "zvfofp4min",
        "zvfofp8min",
        "zvfqwbdota8f",
        "zvfqwdota8f",
        "zvfwbdota16bf",
        "zvfwdota16bf",
        "zvqwbdota8i",
        "zvqwdota8i",
        "zvzip",
        "xventana",
    }
)


@dataclass(frozen=True)
class BuiltElf:
    path: str
    source_path: str
    sha256: str | None
    map: dict
    command: tuple[str, ...]
    exit_code: int | None
    timeout: bool
    missing_toolchain: bool
    stdout: str
    stderr: str
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return (
            type(self.exit_code) is int
            and self.exit_code == 0
            and self.timeout is False
            and self.missing_toolchain is False
            and is_sha256_digest(self.sha256)
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["command"] = list(self.command)
        data["ok"] = self.ok
        return data


def symbol_offsets_from_elf(
    elf_path: Path,
    labels: tuple[str, ...],
    *,
    allow_before_start: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    if (
        not isinstance(labels, (list, tuple))
        or any(type(label) is not str or not label for label in labels)
    ):
        return {}, {}
    try:
        proc = subprocess.run(
            [DEFAULT_NM, "-n", str(elf_path)],
            timeout=BUILD_TIMEOUT_SEC,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}, {}
    if proc.returncode != 0:
        return {}, {}
    output = proc.stdout.decode(errors="replace") if isinstance(proc.stdout, bytes) else proc.stdout
    if not isinstance(output, str):
        return {}, {}
    symbols: dict[str, int] = {}
    requested = {"_start", *labels}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] in requested:
            try:
                address = int(fields[0], 16)
            except (TypeError, ValueError):
                return {}, {}
            if not 0 <= address <= ((1 << 64) - 1):
                return {}, {}
            previous = symbols.get(fields[2])
            if previous is not None and previous != address:
                return {}, {}
            symbols[fields[2]] = address
    start = symbols.get("_start")
    if start is None or (not allow_before_start and any(
        label in symbols and symbols[label] < start for label in labels
    )):
        return {}, {}
    offsets = {label: hex(symbols[label] - start) for label in labels if label in symbols}
    addresses = {label: hex(address) for label, address in symbols.items()}
    return offsets, addresses


def _memory_bytes(testcase: TestCase) -> bytes:
    return b"".join(region.data for region in testcase.initial_memory_regions)


def _byte_lines(data: bytes) -> list[str]:
    indent = "  "
    if not data:
        return [f"{indent}.byte 0"]
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        values = ", ".join(f"0x{value:02x}" for value in chunk)
        lines.append(f"{indent}.byte {values}")
    return lines


def _stack_return_relocation(
    testcase: TestCase, memory: bytes, memory_ranges: list[tuple[int, int, int]],
) -> tuple[int, int, int, int] | None:
    risk = next((item for item in testcase.instruction_meta if "risk" in item.tags), None)
    form = str(getattr(risk, "mnemonic", "")).replace("_", ".")
    if form not in {"cm.popret", "cm.popretz"}:
        return None
    expected_ids = testcase.dataflow_meta.get("expected_executed_instruction_ids", ())
    if not isinstance(expected_ids, (list, tuple)) or len(expected_ids) < 2:
        return None
    target = next(
        (item for item in testcase.instruction_meta if item.instruction_id == expected_ids[1]),
        None,
    )
    try:
        from .rvgen.boundaries import stack_shape, stack_slot_offsets
        xlen = _xlen(testcase.isa_profile)
        fields = dict(risk.operand_fields)
        registers, adjustment = stack_shape(
            {"c_rlist": fields["c_rlist"], "c_spimm": fields["c_spimm"]},
            xlen=xlen,
        )
        slot = next(
            offset + testcase.initial_gpr[2] - start + stack_slot_offsets(
                registers, adjustment, xlen=xlen
            )[1]
            for start, end, offset in memory_ranges
            if start <= testcase.initial_gpr[2] < end
        )
        width = xlen // 8
        value = int.from_bytes(memory[slot : slot + width], "little")
        target_offset = int(target.byte_offset)
    except (AttributeError, IndexError, KeyError, StopIteration, TypeError, ValueError):
        return None
    if (
        target is None or not 0 <= target_offset < len(testcase.code_bytes)
        or slot < 0 or slot + width > len(memory)
    ):
        return None
    return slot, width, value, target_offset


def _logical_memory_registers(testcase: TestCase) -> set[int]:
    registers = {MEMORY_BASE_REGISTER}
    semantic_point = testcase.dataflow_meta.get("semantic_point")
    effect_kind = semantic_point.get("effect_kind") if isinstance(semantic_point, dict) else None
    if effect_kind == "stack-transfer":
        registers.add(2)
    elif effect_kind in {"memory-load", "memory-store", "atomic-rmw", "cache-block"}:
        risk = next((item for item in testcase.instruction_meta if "risk" in item.tags), None)
        if risk is not None and isinstance(risk.rs1, int) and risk.rs1 >= 0:
            registers.add(int(risk.rs1))
        properties = semantic_point.get("properties")
        if isinstance(properties, dict):
            register = properties.get("operand.rs1")
            if isinstance(register, int) and register >= 0:
                registers.add(register)
            elif isinstance(register, str) and register.isdigit():
                registers.add(int(register))
    # A fixed-input RVEMI rewrite may change only the risk instruction's base
    # register (for example x9 -> x20); bind the concrete risk rs1 as well and
    # do not include observer/consumer stores whose rs1 is computed at runtime.
    from .spec_definedness import instruction_effect_class

    for item in testcase.instruction_meta:
        if "risk" not in item.tags or not isinstance(item.rs1, int):
            continue
        if instruction_effect_class(item.mnemonic) in {"load", "store", "atomic"}:
            registers.add(int(item.rs1))
    return registers


def _xlen(isa_profile: str) -> int:
    return 32 if str(isa_profile).lower().startswith("rv32") else 64


def _gpr_count(isa_profile: str) -> int:
    return 16 if "e" in enabled_extensions(isa_profile) else 32


def _observer_registers(isa_profile: str) -> tuple[int, int, int, int]:
    if _gpr_count(isa_profile) == 16:
        return 13, 14, 15, 5
    return 29, 30, 31, 17


def _trap_observer_registers(isa_profile: str) -> dict[str, int]:
    if _gpr_count(isa_profile) == 16:
        return {"mcause": 13, "mepc": 14, "mtval": 15}
    return {"mcause": 31, "mepc": 30, "mtval": 28}


def _saved_register_offsets(isa_profile: str) -> dict[int, int]:
    _, memory_register, output_register, _ = _observer_registers(isa_profile)
    if _gpr_count(isa_profile) == 16:
        return {
            memory_register: -8,
            output_register: -4,
        }
    if _xlen(isa_profile) == 64:
        return {
            29: -24,
            memory_register: -16,
            output_register: -8,
        }
    return {
        29: -12,
        memory_register: -8,
        output_register: -4,
    }


def _saved_register_base(isa_profile: str) -> str:
    return "x29" if _gpr_count(isa_profile) == 32 else "x2"


def _store_u64_lines(*, base: str, offset: int, source: str, xlen: int) -> list[str]:
    if xlen == 64:
        return [f"  sd {source}, {offset}({base})"]
    return [
        f"  sw {source}, {offset}({base})",
        f"  sw x0, {offset + 4}({base})",
    ]


def _store_gpr_lines(*, isa_profile: str, xlen: int) -> list[str]:
    gpr_count = _gpr_count(isa_profile)
    temp_register, _, output_register, _ = _observer_registers(isa_profile)
    saved_offsets = _saved_register_offsets(isa_profile)
    saved_base = _saved_register_base(isa_profile)
    load_register = 28 if saved_base == "x29" and temp_register == 29 else temp_register
    load_mnemonic = "ld" if xlen == 64 else "lw"
    lines = []
    for index in range(32):
        if index >= gpr_count:
            source = "x0"
        elif index in saved_offsets:
            lines.append(
                f"  {load_mnemonic} x{load_register}, {saved_offsets[index]}({saved_base})"
            )
            source = f"x{load_register}"
        else:
            source = f"x{index}"
        lines.extend(
            _store_u64_lines(
                base=f"x{output_register}",
                offset=16 + index * 8,
                source=source,
                xlen=xlen,
            )
        )
    return lines


def _requires_instruction_memory_support(testcase: TestCase) -> bool:
    return any(set("rwx") <= set(region.permissions) for region in testcase.initial_memory_regions)


_SPECIALIZED_FRAME_STATE_DOMAINS = frozenset(
    {"csr", "privileged", "reservation", "vector", "vector-memory", "trap"}
)
_CODE_MEMORY_STATE_DOMAINS = frozenset({"instruction-memory", "lifecycle"})
_FRAME_STATE_DOMAINS = frozenset(
    {
        "scalar",
        "gpr",
        "memory",
        "control-flow",
        "csr",
        "reservation",
        "vector",
        "vector-memory",
        "privileged",
        "trap",
        "instruction-memory",
        "lifecycle",
    }
)
_STATEFUL_FRAME_TAGS = frozenset({
    "csr-state", "vector-state", "vector-csr-state", "vector-config-state",
    "reservation-state", "privileged-state",
})
_STATEFUL_FRAME_MNEMONICS = frozenset({
    "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci",
    "lr.w", "lr.d", "sc.w", "sc.d",
})


def _frame_parts(testcase: TestCase) -> tuple[dict, dict | None, dict | None]:
    dataflow_meta = getattr(testcase, "dataflow_meta", {})
    if not isinstance(dataflow_meta, dict):
        raise ValueError("observer-gap:frame-dataflow")
    semantic_point = dataflow_meta.get("semantic_point")
    realized_facts = dataflow_meta.get("realized_facts")
    if semantic_point is not None and not isinstance(semantic_point, dict):
        raise ValueError("observer-gap:frame-semantic-point")
    if realized_facts is not None and not isinstance(realized_facts, dict):
        raise ValueError("observer-gap:frame-realized-facts")
    return dataflow_meta, semantic_point, realized_facts


def _frame_fact(
    semantic_point: dict | None,
    realized_facts: dict | None,
    key: str,
) -> object | None:
    semantic_value = semantic_point.get(key) if semantic_point else None
    return semantic_value if semantic_value is not None else (
        realized_facts.get(key) if realized_facts else None
    )


def _instruction_items(testcase: TestCase) -> tuple[object, ...]:
    return tuple(testcase.instruction_meta)


def _risk_items(testcase: TestCase) -> tuple[object, ...]:
    return tuple(item for item in _instruction_items(testcase) if "risk" in item.tags)


def _risk_pc_contract(testcase: TestCase) -> dict[str, object]:
    risk_items = _risk_items(testcase)
    if type(getattr(testcase, "code_bytes", None)) is not bytes:
        raise ValueError("observer-gap:risk-pc-binding")
    if len(risk_items) != 1:
        raise ValueError("observer-gap:risk-pc-binding")
    offsets: list[int] = []
    instruction_ids: list[str] = []
    code_size = len(getattr(testcase, "code_bytes", b""))
    for item in risk_items:
        instruction_id = getattr(item, "instruction_id", None)
        if type(instruction_id) is not str or not instruction_id.strip():
            raise ValueError("observer-gap:risk-pc-binding")
        pc_offset = getattr(item, "pc_offset", None)
        byte_offset = getattr(item, "byte_offset", None)
        byte_length = getattr(item, "byte_length", None)
        if (
            type(pc_offset) is not int
            or type(byte_offset) is not int
            or type(byte_length) is not int
            or pc_offset != byte_offset
            or pc_offset < 0
            or byte_length <= 0
            or pc_offset + byte_length > code_size
        ):
            raise ValueError("observer-gap:risk-pc-binding")
        offsets.append(pc_offset)
        instruction_ids.append(instruction_id)
    if len(offsets) != len(set(offsets)):
        raise ValueError("observer-gap:risk-pc-binding")
    return {
        "risk_instruction_ids": instruction_ids,
        "risk_pc_offsets": offsets,
        "binding": "instruction-meta.pc_offset",
        "execution_proof": "static-only" if offsets else "unbound",
    }


def _observer_register_contract(
    testcase: TestCase,
    check_interference: bool = True,
) -> dict[str, object]:
    from .riscv_encoding import form_for_mnemonic, implicit_source_roles, register_domain

    temp_register, memory_register, output_register, syscall_register = _observer_registers(
        testcase.isa_profile
    )
    implicit_sp_risk = any(
        "rs1" in implicit_source_roles(form)
        for item in _risk_items(testcase)
        if (form := form_for_mnemonic(str(getattr(item, "mnemonic", "")))) is not None
    )
    pre_risk_reserved = {MEMORY_BASE_REGISTER, memory_register}
    if not implicit_sp_risk:
        pre_risk_reserved.add(2)
    if _gpr_count(testcase.isa_profile) == 32:
        pre_risk_reserved.add(29)
    pre_risk_reserved = tuple(sorted(pre_risk_reserved))
    risk_registers = sorted({
        int(register)
        for item in _risk_items(testcase)
        for role in ("rd", "rs1", "rs2", "rs3")
        for register in (getattr(item, role, None),)
        if type(register) is int
        and (
            (form := form_for_mnemonic(str(getattr(item, "mnemonic", "")))) is None
            or register_domain(form, role) == "gpr"
        )
    })
    if check_interference:
        interfering = sorted(set(risk_registers) & set(pre_risk_reserved))
        if interfering:
            joined = ",".join(f"x{register}" for register in interfering)
            raise ValueError("observer-gap:observer-register-interference:" + joined)
    saved = _saved_register_offsets(testcase.isa_profile)
    return {
        "pre_risk_reserved": list(pre_risk_reserved),
        "risk_registers": risk_registers,
        "saved_after_risk": sorted(saved),
        "observer_temporaries_after_snapshot": [
            temp_register,
            memory_register,
            output_register,
        ],
        "syscall_register": syscall_register,
        "non_interference": (
            "risk operands exclude pre-risk observer registers"
            if check_interference
            else "trap row: risk instruction is not executed, register interference not applicable"
        ),
    }


def _observer_memory_contract(testcase: TestCase) -> dict[str, object]:
    regions = getattr(testcase, "initial_memory_regions", ())
    if not isinstance(regions, (list, tuple)):
        raise ValueError("observer-gap:observer-memory-layout")
    cursor = 0
    region_ids: list[str] = []
    seen_ids: set[str] = set()
    for region in regions:
        region_id = getattr(region, "region_id", None)
        address = getattr(region, "address", None)
        data = getattr(region, "data", None)
        if (
            type(region_id) is not str
            or not region_id.strip()
            or region_id in seen_ids
            or type(address) is not int
            or type(data) is not bytes
            or address != cursor
        ):
            raise ValueError("observer-gap:observer-memory-layout")
        cursor += len(data)
        seen_ids.add(region_id)
        region_ids.append(region_id)
    return {
        "source_region_ids": region_ids,
        "logical_address_start": 0,
        "logical_size": cursor,
        "flattened_layout": True,
        "result_buffer_disjoint": True,
        "non_interference": "result_buffer and stack are outside test_memory",
    }


def _code_memory_contract(testcase: TestCase) -> dict[str, object]:
    dataflow_meta, semantic_point, realized_facts = _frame_parts(testcase)
    state_domain = dataflow_meta.get("state_domain", "")
    if state_domain is not None and type(state_domain) is not str:
        raise ValueError("observer-gap:frame-state-domain")
    state_domain = state_domain or ""
    effect_kind = _frame_fact(semantic_point, realized_facts, "effect_kind")
    lane = _frame_fact(semantic_point, realized_facts, "lane")
    # legality-expected-trap 行（例如 rv64i 无 zifencei 时的 fence.i）只观察
    # 陷阱交付，程序不会 patch 指令内存：冻结的 effect_kind 虽为
    # instruction-memory，也不需要 rwx 与外部 lifecycle observer。
    trap_lane = expected_trap_from_dataflow(dataflow_meta)
    required = (
        state_domain in _CODE_MEMORY_STATE_DOMAINS
        or effect_kind == "instruction-memory"
    ) and not trap_lane
    permissions = "rwx" if required else "rw"
    if required and not _requires_instruction_memory_support(testcase):
        raise ValueError("target-unavailable:code-memory-permissions")
    return {
        "required": required,
        "permissions": permissions,
        "mprotect_setup": required,
        "lifecycle_observer": "external-required" if required else None,
        "status": "static-setup-only" if required else "ordinary-frame",
    }


def _direct_elf_frame_contract(testcase: TestCase) -> dict[str, object]:
    dataflow_meta, semantic_point, realized_facts = _frame_parts(testcase)
    classification_gap = dataflow_meta.get("candidate_trap_classification_gap")
    if classification_gap:
        raise ValueError(
            "candidate-trap-classification-gap:" + str(classification_gap)[:500]
        )
    observer_gap = dataflow_meta.get("candidate_observer_contract_gap")
    if observer_gap:
        raise ValueError("candidate-observer-contract-gap:" + str(observer_gap)[:500])
    instruction_items = _instruction_items(testcase)
    stateful_instruction = any(
        _STATEFUL_FRAME_TAGS.intersection(tuple(getattr(item, "tags", ())))
        or str(getattr(item, "mnemonic", "")) in _STATEFUL_FRAME_MNEMONICS
        for item in instruction_items
    )
    if stateful_instruction and "state_domain" not in dataflow_meta:
        raise ValueError("observer-gap:frame-state-domain")
    state_domain = dataflow_meta.get("state_domain", "")
    if state_domain is not None and type(state_domain) is not str:
        raise ValueError("observer-gap:frame-state-domain")
    state_domain = state_domain or ""
    if state_domain and state_domain not in _FRAME_STATE_DOMAINS:
        raise ValueError("observer-gap:direct-elf-state-domain")
    disposition = realized_facts.get("disposition") if isinstance(realized_facts, dict) else None
    effect_kind = _frame_fact(semantic_point, realized_facts, "effect_kind")
    lane = _frame_fact(semantic_point, realized_facts, "lane")
    raw_expected = dataflow_meta.get("expected_executed_instruction_ids")
    if raw_expected is None:
        expected_executed = frozenset()
    elif (
        not isinstance(raw_expected, (list, tuple))
        or any(type(item) is not str or not item.strip() for item in raw_expected)
        or len(raw_expected) != len(set(raw_expected))
        or not set(raw_expected) <= {
            str(getattr(item, "instruction_id", ""))
            for item in instruction_items
        }
    ):
        raise ValueError("observer-gap:expected-executed-instruction-ids")
    else:
        expected_executed = frozenset(raw_expected)
    second_fetch_shape = (
        isinstance(semantic_point, dict)
        and semantic_point.get("effect_kind") == "instruction-memory"
        and semantic_point.get("lane") == "normal-defined"
        and isinstance(semantic_point.get("properties"), dict)
        and semantic_point["properties"].get("instruction_memory.second_fetch") is True
    ) or {"fetch0", "fetchobs", "fetchret"}.issubset(expected_executed)
    expected_trap = expected_trap_from_dataflow(dataflow_meta)
    if state_domain == "trap" and expected_trap:
        return _trap_frame_contract(testcase)
    csr_observer_embedded = any(
        "csr-state" in tuple(getattr(item, "tags", ()))
        and "observable" in tuple(getattr(item, "tags", ()))
        and "risk" not in tuple(getattr(item, "tags", ()))
        for item in instruction_items
    )

    def ordinary_frame() -> dict[str, object]:
        return {
            "risk_pc_contract": _risk_pc_contract(testcase),
            "observer_register_contract": _observer_register_contract(testcase),
            "observer_memory_contract": _observer_memory_contract(testcase),
            "code_memory_contract": _code_memory_contract(testcase),
        }

    if state_domain == "csr" and csr_observer_embedded:
        # materialize 为 csr 行固定内嵌 csr-state-observer（csrrs + sd @
        # offset 32），csr 回读因此通过 test-memory 快照字段化，frame 退化为
        # 普通观察帧；无该内嵌的 csr 行继续走下方 fail-closed 分支。
        frame = ordinary_frame()
        frame["csr_observer"] = "embedded-csr-state-observer"
        frame["expected_fields"] = ("csr.warl.readback", "csr.reserved-fields")
        return frame
    if state_domain in {"vector", "vector-memory"} and expected_trap:
        # A legality row is observed purely by the delivered trap (cause/epc/
        # tval or host signal); the vector state itself is never read back on
        # the trap path.  The ELF frame is the minimal trap row, so the
        # vector state observer stays with the backend (qemu gdbstub F-17).
        return _trap_frame_contract(testcase)
    if state_domain in {"vector", "vector-memory"}:
        properties = semantic_point.get("properties") if isinstance(semantic_point, dict) else None
        vector_scalar_row = (
            isinstance(properties, dict)
            and str(properties.get("effect_operation", "")).startswith("vector-to-scalar:")
            and any(
                (tags := tuple(getattr(item, "tags", ())))
                and "observable" in tags
                and "risk" not in tags
                and any(tag in tags for tag in ("vector-state", "vector-csr-state"))
                for item in instruction_items
            )
        )
        if not (_vector_observer_embedded(testcase) or vector_scalar_row):
            raise ValueError("observer-gap:direct-elf-specialized-frame:" + state_domain)
        # F-17 (2026-08-12): the qemu gdbstub exposes vstart/vl/vtype/vlenb
        # and v0..v31 (riscv-vector.xml/riscv-csr.xml), so vector state
        # observation is provided by that backend and the ELF frame only
        # carries the ordinary GPR/memory observer.  Rows whose materialized
        # program observes the vector-scalar result through GPR stores (for
        # example vmv.x.s or vsetvl writing a scalar GPR) also carry the
        # ordinary observer frame; the backend vector observer is added when
        # the generated program embeds a vector state observation instruction
        # (vector-csr-state tag).  libriscv has no vector register observation
        # and stays fail-closed at the backend gate.
        frame = ordinary_frame()
        if _vector_observer_embedded(testcase):
            frame["vector_observer"] = "qemu-gdbstub-f17"
            frame["expected_fields"] = ("vstart", "vl", "vtype", "vlenb", "v0..v31")
        return frame
    if state_domain == "reservation":
        # SC 的 rd 写回经普通 GPR observer 观察，capsule 内存快照足以派生
        # reservation.result/valid 与 memory.events；probe 在 exit_checkpoint
        # 之后读回即可，不需要 GDB 直读 reservation 寄存器。仅当冻结的
        # dataflow 声明了 reservation_shape 时才放行，否则保持 fail-closed。
        if (
            type(dataflow_meta.get("reservation_shape")) is not str
            or not dataflow_meta["reservation_shape"].strip()
        ):
            raise ValueError("observer-gap:direct-elf-specialized-frame:reservation")
        frame = ordinary_frame()
        frame["reservation_observer"] = "sc-rd-writeback-plus-capsule-memory-snapshot"
        frame["expected_fields"] = (
            "reservation.valid",
            "reservation.address",
            "reservation.ordering",
            "reservation.result",
            "memory.events",
        )
        return frame
    if state_domain in _SPECIALIZED_FRAME_STATE_DOMAINS:
        raise ValueError("observer-gap:direct-elf-specialized-frame:" + state_domain)
    if state_domain in _CODE_MEMORY_STATE_DOMAINS or effect_kind == "instruction-memory":
        # legality-expected-trap 行（rv64i 无 zifencei 的 fence.i）没有
        # second-fetch 程序形状，观察只是陷阱交付，走最小 trap 帧。
        if expected_trap:
            return _trap_frame_contract(testcase)
        # T-7 fence.i second-fetch（store -> fence.i -> second-fetch）：
        # 生成侧 semantic point 冻结了 second_fetch 形状；观察由 capsule
        # memory 快照（fetch 结果 store）+ executed-PC 通道（patch 地址在
        # fence.i 之后被取指执行）共同承担，普通观察帧即可认证。其他
        # instruction-memory/lifecycle 形状保持 fail-closed。
        if lane == "normal-defined" and second_fetch_shape:
            frame = ordinary_frame()
            frame["fence_i_observer"] = "executed-pc-plus-capsule-memory-snapshot"
            frame["expected_fields"] = (
                "memory.code-before",
                "memory.code-after",
                "fetch.second",
                "translation.path_identity",
                "executed_pc",
            )
            return frame
        # A normal observer frame cannot certify second fetch/TB/path state.
        raise ValueError("observer-gap:direct-elf-lifecycle-frame")
    # cache 操作（zicbo）与 fence 的 normal-defined realization 是
    # state-environment-pending：没有具体值需要比对，普通观察帧就够用。
    # 之前把它当 gap 直接拒绝，等于让这些 form 永远进不了模拟器。
    return ordinary_frame()


def _vector_observer_embedded(testcase: TestCase) -> bool:
    """vector/vector-memory 域 ELF frame 是否内嵌矢量状态观察指令。"""

    return any(
        "observable" in tuple(getattr(item, "tags", ()))
        and "risk" not in tuple(getattr(item, "tags", ()))
        and any(
            tag in tuple(getattr(item, "tags", ()))
            for tag in ("vector-state", "vector-csr-state", "vector-config-state")
        )
        for item in _instruction_items(testcase)
    )


def _trap_frame_contract(testcase: TestCase) -> dict[str, object]:
    """Minimal frame contract for legality/expected-trap rows.

    The program traps on the risk instruction, then records the mailbox after
    the trap handler has captured ``mcause/mepc/mtval`` in ignored GPR slots.
    """
    return {
        "risk_pc_contract": _risk_pc_contract(testcase),
        "observer_register_contract": _observer_register_contract(
            testcase, check_interference=False
        ),
        "observer_memory_contract": _observer_memory_contract(testcase),
        "code_memory_contract": _code_memory_contract(testcase),
        "observer_contract": "expected-trap-outcome-observation",
        "trap_observer": "expected-trap-delivery",
        "trap_observer_registers": _trap_observer_registers(testcase.isa_profile),
        "frame_kind": "trap-row",
    }


def _compressed_enabled(isa_profile: str) -> bool:
    # C can be implied by a Zc extension in the suffix (including accepted
    # historical G aliases such as ``rv64g_zcmp``), so inspect the canonical
    # extension closure instead of only the base spelling.
    return "c" in enabled_extensions(isa_profile)


def _toolchain_profile(isa_profile: str) -> tuple[str, str, tuple[str, ...]]:
    from .spec_definedness import is_canonical_isa_profile

    if not is_canonical_isa_profile(isa_profile):
        raise ValueError(f"invalid RISC-V ISA profile: {isa_profile!r}")
    base = isa_base_extensions(isa_profile)
    enabled = enabled_extensions(isa_profile)
    xlen = "ilp32" if str(isa_profile).lower().startswith("rv32") else "lp64"
    if xlen == "ilp32" and "e" in base:
        mabi = "ilp32e"
    elif "q" in enabled:
        # The locked GCC toolchain accepts Q instructions in -march but does
        # not provide an ilp32q/lp64q ABI.  The raw testcase bytes do not
        # depend on the ABI's generated FP calls, so D is the usable ABI.
        mabi = f"{xlen}d"
    elif "d" in enabled:
        mabi = f"{xlen}d"
    elif "f" in enabled:
        mabi = f"{xlen}f"
    else:
        mabi = xlen
    head, *suffixes = isa_profile.split("_")
    # gcc 的 base 序不接受 p（canonical order）与 s（supervisor 名单字母）；
    # raw testcase 以 .word 发射，两者只在 prologue 身份里出现，剔除即可。
    toolchain_head = "".join(character for character in head if character not in "ps")
    dropped = tuple(character for character in head[4:] if character in "ps")
    expanded = tuple(
        item
        for suffix in suffixes
        for item in (("zicbom", "zicbop", "zicboz") if suffix.lower() == "zicbo" else (suffix,))
    )
    dropped += tuple(suffix for suffix in expanded if suffix.lower() in _TOOLCHAIN_UNKNOWN_SUFFIXES)
    kept = tuple(suffix for suffix in expanded if suffix.lower() not in _TOOLCHAIN_UNKNOWN_SUFFIXES)
    return "_".join((toolchain_head, *kept)), mabi, dropped


def assembly_source(
    testcase: TestCase,
    *,
    prologue_lines: tuple[str, ...] = (),
    bare_metal: bool = False,
) -> str:
    from .spec_definedness import is_canonical_isa_profile

    if not is_canonical_isa_profile(getattr(testcase, "isa_profile", "")):
        raise ValueError(
            f"invalid RISC-V ISA profile: {getattr(testcase, 'isa_profile', None)!r}"
        )
    if _gpr_count(testcase.isa_profile) == 16 and not bare_metal:
        raise ValueError("RV32E Linux-user harness is unsupported")
    dataflow_meta = getattr(testcase, "dataflow_meta", None) or {}
    if str(dataflow_meta.get("state_domain", "")) == "privileged":
        raise ValueError("privileged direct ELF requires an attached privileged harness")
    realization = dataflow_meta.get("generation_realization", {})
    user_mode = isinstance(realization, dict) and realization.get("privilege_class") in {"user", "u"}
    frame_contract = _direct_elf_frame_contract(testcase)
    if any(not isinstance(line, str) or not line for line in prologue_lines):
        raise ValueError("prologue_lines must contain non-empty strings")
    # The capsule transport always runs this FP prologue, so frame labels
    # (entry_checkpoint/test_start/test_end/exit_checkpoint) shift by four
    # instructions relative to _start.  Keep the linux-syscall frame at the
    # same offsets by padding with nops when the profile carries FP state:
    # the audit compares offsets, and nops cannot change U-mode semantics.
    fp_frame_padding = (
        ("  nop",) * 4
        if profile_requires_fp_state(getattr(testcase, "isa_profile", ""))
        and not any("mstatus" in line for line in prologue_lines)
        else ()
    )
    expected_permissions = str(frame_contract["code_memory_contract"]["permissions"])
    if any(
        set(region.permissions) != set(expected_permissions)
        for region in testcase.initial_memory_regions
    ):
        raise ValueError(
            "direct ELF binder cannot realize requested memory permissions"
        )
    memory = _memory_bytes(testcase)
    observation_size = OBSERVATION_HEADER_SIZE + len(memory)
    xlen = _xlen(testcase.isa_profile)
    temp_register, memory_register, output_register, syscall_register = _observer_registers(testcase.isa_profile)
    save_mnemonic = "sd" if xlen == 64 else "sw"
    saved_offsets = _saved_register_offsets(testcase.isa_profile)
    saved_base = _saved_register_base(testcase.isa_profile)
    stack_save_lines = [
        f"  {save_mnemonic} x{memory_register}, {saved_offsets[memory_register]}({saved_base})",
        f"  {save_mnemonic} x{output_register}, {saved_offsets[output_register]}({saved_base})",
    ]
    restore_sp_lines = ["  mv x2, x29"] if _gpr_count(testcase.isa_profile) == 32 else []
    gpr_count = _gpr_count(testcase.isa_profile)
    harness_sp_register = 29 if gpr_count == 32 else None
    bare_fp_setup_lines = ()
    uses_fp = testcase_uses_fp_instruction(testcase)
    uses_vector = testcase_uses_vector_instruction(testcase)
    if bare_metal and (uses_fp or uses_vector):
        # Bare capsules start with extension state Off. Initialize only state
        # used by this testcase, preserving the other mstatus fields.
        # Spill x31 so its logical address binding survives setup.
        fp_scratch = 31 if gpr_count == 32 else temp_register
        fp_stack_slot = min(saved_offsets.values()) - xlen // 8
        spill_store = "sd" if xlen == 64 else "sw"
        spill_load = "ld" if xlen == 64 else "lw"
        state_mask = (0x6000 if uses_fp else 0) | (0x600 if uses_vector else 0)
        state_initial = (0x2000 if uses_fp else 0) | (0x200 if uses_vector else 0)
        bare_fp_setup_lines = (
            f"  {spill_store} x{fp_scratch}, {fp_stack_slot}({saved_base})",
            f"  li x{fp_scratch}, 0x{state_mask:x}",
            f"  csrc mstatus, x{fp_scratch}",
            f"  li x{fp_scratch}, 0x{state_initial:x}",
            f"  csrs mstatus, x{fp_scratch}",
            f"  {spill_load} x{fp_scratch}, {fp_stack_slot}({saved_base})",
        )
    mask = (1 << xlen) - 1
    hex_width = 8 if xlen == 32 else 16
    logical_memory_registers = _logical_memory_registers(testcase)
    semantic_point = testcase.dataflow_meta.get("semantic_point")
    risk = next((item for item in testcase.instruction_meta if "risk" in item.tags), None)
    logical_code_registers = set()
    if (isinstance(semantic_point, dict)
            and semantic_point.get("effect_kind") == "control-jump"
            and risk is not None and isinstance(risk.rs1, int) and risk.rs1 > 0):
        logical_code_registers.add(int(risk.rs1))
    valid_code_offsets = {int(item.byte_offset) for item in testcase.instruction_meta}
    memory_ranges = []
    memory_offset = 0
    for region in testcase.initial_memory_regions:
        start = int(region.address)
        size = len(region.data)
        memory_ranges.append((start, start + size, memory_offset))
        memory_offset += size
    relocation = _stack_return_relocation(testcase, memory, memory_ranges)
    memory_data_lines = _byte_lines(memory if memory else bytes(8))
    memory_restore_lines: list[str] = []
    if relocation is not None:
        slot, width, value, target_offset = relocation
        prefix, suffix = memory[:slot], memory[slot + width :]
        memory_data_lines = (
            (_byte_lines(prefix) if prefix else [])
            + [f"  .{'dword' if width == 8 else 'word'} test_start + {target_offset}"]
            + (_byte_lines(suffix) if suffix else [])
        )
        store_mnemonic = "sd" if width == 8 else "sw"
        memory_restore_lines = [
            f"  la x{memory_register}, test_memory",
            f"  li x{temp_register}, 0x{value:x}",
            f"  {store_mnemonic} x{temp_register}, {slot}(x{memory_register})",
        ]
    initial_register_lines = []
    if harness_sp_register is not None:
        initial_register_lines.append(
            f"  li x29, 0x{testcase.initial_gpr[29]:0{hex_width}x}"
        )
    initial_register_lines.append("  la x2, stack_top")
    if harness_sp_register is not None:
        initial_register_lines.append(
            f"  {save_mnemonic} x29, {saved_offsets[29]}(x2)"
        )
        initial_register_lines.append(f"  mv x{harness_sp_register}, x2")
    for index, value in enumerate(testcase.initial_gpr):
        if index >= gpr_count or index == 0 or index == harness_sp_register:
            continue
        if index == 2 and harness_sp_register is None:
            continue
        resolved = value & mask
        code_binding = (
            resolved
            if index in logical_code_registers and resolved in valid_code_offsets
            or index in logical_code_registers
            and resolved & 1
            and (resolved & ~1) in valid_code_offsets
            else None
        )
        logical_binding = (
            next(
                (
                    offset + resolved - start
                    for start, end, offset in memory_ranges
                    if start <= resolved < end
                ),
                None,
            )
            if index in logical_memory_registers
            else None
        )
        if logical_binding is None and index in logical_memory_registers and risk is not None and risk.rs1 == index:
            immediate = int(risk.immediate or 0)
            effective = (resolved + immediate) & mask
            logical_binding = next(
                (
                    offset + effective - start - immediate
                    for start, end, offset in memory_ranges
                    if start <= effective < end
                ),
                None,
            )
        if code_binding is not None:
            initial_register_lines.append(f"  la x{index}, test_start")
            if code_binding:
                if -2048 <= code_binding <= 2047:
                    initial_register_lines.append(f"  addi x{index}, x{index}, {code_binding}")
                else:
                    offset_register = memory_register if index != memory_register else output_register
                    initial_register_lines.extend((
                        f"  li x{offset_register}, {code_binding}",
                        f"  add x{index}, x{index}, x{offset_register}",
                    ))
        elif logical_binding is not None and index != MEMORY_BASE_REGISTER:
            initial_register_lines.append(f"  la x{index}, test_memory")
            if logical_binding:
                if -2048 <= logical_binding <= 2047:
                    initial_register_lines.append(f"  addi x{index}, x{index}, {logical_binding}")
                else:
                    initial_register_lines.extend((
                        f"  li x{memory_register}, {logical_binding}",
                        f"  add x{index}, x{index}, x{memory_register}",
                    ))
        elif resolved:
            initial_register_lines.append(f"  li x{index}, 0x{resolved:0{hex_width}x}")
        else:
            initial_register_lines.append(f"  li x{index}, 0")
    trap_setup_lines = ()
    if bare_metal and frame_contract.get("frame_kind") == "trap-row":
        trap_registers = _trap_observer_registers(testcase.isa_profile)
        trap_vector_register = trap_registers["mcause"]
        if gpr_count == 16:
            trap_setup_lines = (
                f"  sw x{trap_vector_register}, -12(x2)",
                f"  la x{trap_vector_register}, bare_metal_trap_exit",
                f"  csrw mtvec, x{trap_vector_register}",
                f"  lw x{trap_vector_register}, -12(x2)",
            )
        else:
            trap_setup_lines = (
                f"  la x{trap_vector_register}, bare_metal_trap_exit",
                f"  csrw mtvec, x{trap_vector_register}",
            )
            trap_setup_lines += (
                f"  li x{trap_vector_register}, "
                f"0x{testcase.initial_gpr[trap_vector_register]:0{hex_width}x}",
            )
    user_mode_lines = ()
    # Linux-user emulators already enter at U-mode.  M-mode CSR writes and
    # mret here trap before the risk instruction; only bare-metal capsules
    # need to lower privilege explicitly.
    if bare_metal and user_mode and frame_contract.get("frame_kind") == "trap-row":
        trap_vector_register = _trap_observer_registers(testcase.isa_profile)["mcause"]
        if gpr_count == 16:
            user_mode_lines = (
                f"  sw x{trap_vector_register}, -12(x2)",
                f"  li x{trap_vector_register}, 0x1800",
                f"  csrrc x0, mstatus, x{trap_vector_register}",
                f"  la x{trap_vector_register}, test_start",
                f"  csrw mepc, x{trap_vector_register}",
                f"  lw x{trap_vector_register}, -12(x2)",
            )
        else:
            user_mode_lines = (
                f"  li x{trap_vector_register}, 0x1800",
                f"  csrrc x0, mstatus, x{trap_vector_register}",
                f"  la x{trap_vector_register}, test_start",
                f"  csrw mepc, x{trap_vector_register}",
            )
            user_mode_lines += (
                f"  li x{trap_vector_register}, "
                f"0x{testcase.initial_gpr[trap_vector_register]:0{hex_width}x}",
            )
        user_mode_lines += ("  mret",)
    if MEMORY_BASE_REGISTER < gpr_count:
        initial_register_lines.append(f"  la x{MEMORY_BASE_REGISTER}, test_memory")
    instruction_memory_setup_lines: list[str] = []
    if _requires_instruction_memory_support(testcase) and not bare_metal:
        text_span = max(4096, ((len(testcase.code_bytes) + 4095) // 4096) * 4096)
        memory_span = max(4096, ((len(memory) + 4095) // 4096) * 4096)
        instruction_memory_setup_lines = [
            "  la x10, _start",
            f"  li x11, {text_span}",
            "  li x12, 7",
            f"  li x{syscall_register}, 226",
            "  ecall",
            "  la x10, test_memory",
            f"  li x11, {memory_span}",
            "  li x12, 7",
            f"  li x{syscall_register}, 226",
            "  ecall",
        ]
    memory_copy_lines: list[str] = []
    if memory:
        offset = 0
        word_bytes = 8 if xlen == 64 else 4
        load_mnemonic = "ld" if xlen == 64 else "lw"
        store_mnemonic = "sd" if xlen == 64 else "sw"
        memory_copy_lines.append(f"  la x{memory_register}, test_memory")
        memory_copy_lines.append(
            f"  addi x{output_register}, x{output_register}, {OBSERVATION_HEADER_SIZE}"
        )
        while offset + word_bytes <= len(memory):
            memory_copy_lines.extend(
                (
                    f"  {load_mnemonic} x{temp_register}, 0(x{memory_register})",
                    f"  {store_mnemonic} x{temp_register}, 0(x{output_register})",
                    f"  addi x{memory_register}, x{memory_register}, {word_bytes}",
                    f"  addi x{output_register}, x{output_register}, {word_bytes}",
                )
            )
            offset += word_bytes
        while offset < len(memory):
            memory_copy_lines.extend(
                (
                    f"  lbu x{temp_register}, 0(x{memory_register})",
                    f"  sb x{temp_register}, 0(x{output_register})",
                    f"  addi x{memory_register}, x{memory_register}, 1",
                    f"  addi x{output_register}, x{output_register}, 1",
                )
            )
            offset += 1
    lines = [
        "  .option rvc" if _compressed_enabled(testcase.isa_profile) else "  .option norvc",
        "  .section .text",
        "  .globl _start",
        "  .globl test_0",
        "  .set test_0, _start",
        "  .globl test_entry",
        "  .set test_entry, _start",
        "_start:",
        *instruction_memory_setup_lines,
        *fp_frame_padding,
        *prologue_lines,
        *initial_register_lines,
        *trap_setup_lines,
        *bare_fp_setup_lines,
        *user_mode_lines,
        "  j test_start",
        "  .balign 4096",
        "entry_checkpoint:",
        "test_start:",
    ]
    instruction_chunks = (
        [
            (
                item.byte_length,
                testcase.code_bytes[item.byte_offset : item.byte_offset + item.byte_length],
            )
            for item in sorted(testcase.instruction_meta, key=lambda item: item.byte_offset)
        ]
        if testcase.instruction_meta
        else [(len(testcase.code_bytes), testcase.code_bytes)]
    )
    for width, chunk in instruction_chunks:
        if width == 2:
            lines.append(f"  .2byte 0x{int.from_bytes(chunk, 'little'):04x}")
        elif width == 4:
            lines.append(f"  .word 0x{int.from_bytes(chunk, 'little'):08x}")
        else:
            lines.extend(_byte_lines(chunk))
    finish_lines = [
        "  li x10, 1",
        "  la x11, tohost",
        f"  {save_mnemonic} x10, 0(x11)",
        "  ebreak",
    ] if bare_metal else [
        "  li x10, 1",
        "  la x11, result_buffer",
        f"  li x12, {observation_size}",
        f"  li x{syscall_register}, 64",
        "  ecall",
        "  li x10, 0",
        f"  li x{syscall_register}, 93",
        "  ecall",
        "  ebreak",
    ]
    trap_registers = _trap_observer_registers(testcase.isa_profile)
    trap_handler_lines = (
        (
            "  .balign 4",
            "bare_metal_trap_exit:",
            f"  csrr x{trap_registers['mcause']}, mcause",
            f"  csrr x{trap_registers['mepc']}, mepc",
            f"  csrr x{trap_registers['mtval']}, mtval",
            "  j exit_checkpoint",
        )
        if trap_setup_lines else ()
    )
    lines.extend(
        [
            "test_end:",
            "  .option push",
            "  .option norvc",
            "exit_checkpoint:",
            *stack_save_lines,
            f"  la x{output_register}, result_buffer",
            *_store_gpr_lines(isa_profile=testcase.isa_profile, xlen=xlen),
            *restore_sp_lines,
            f"  li x{memory_register}, {len(memory)}",
            *_store_u64_lines(base=f"x{output_register}", offset=16 + 32 * 8, source=f"x{memory_register}", xlen=xlen),
            *memory_restore_lines,
            *memory_copy_lines,
            f"  la x{output_register}, result_buffer",
            f"  la x{memory_register}, exit_checkpoint",
            *_store_u64_lines(base=f"x{output_register}", offset=8, source=f"x{memory_register}", xlen=xlen),
            *finish_lines,
            *trap_handler_lines,
            "  .option pop",
            "",
            "  .section .data",
            "  .balign 8",
            "  .globl obs_buf",
            "obs_buf:",
            "result_buffer:",
            *_byte_lines(OBSERVATION_MAGIC),
            f"  .zero {observation_size - len(OBSERVATION_MAGIC)}",
            "  .balign 4096" if _requires_instruction_memory_support(testcase) else "  .balign 256",
            "  .globl test_memory_start",
            "test_memory_start:",
            "test_memory:",
            *memory_data_lines,
            "  .globl test_memory_end",
            "test_memory_end:",
            "",
            "  .section .tohost,\"aw\",@progbits",
            "  .balign 8",
            "  .local tohost",
            "tohost:",
            "  .dword 0",
            "",
            "  .section .bss",
            "  .balign 16",
            "stack_area:",
            "  .zero 4096",
            "stack_top:",
            "",
        ]
    )
    return "\n".join(lines)


def _runtime_symbol_map_is_valid(addresses: dict[str, str], *, code_size: int) -> bool:
    names = (
        "_start",
        "entry_checkpoint",
        "test_start",
        "test_end",
        "exit_checkpoint",
        "result_buffer",
        "test_memory",
    )
    if type(code_size) is not int or code_size <= 0:
        return False
    if not isinstance(addresses, dict) or not all(name in addresses for name in names):
        return False
    try:
        values = {name: int(addresses[name], 0) for name in names}
    except (TypeError, ValueError):
        return False
    return (
        all(type(value) is int and 0 <= value <= (1 << 64) - 1 for value in values.values())
        and all(value % 2 == 0 for value in values.values())
        and values["_start"] <= values["entry_checkpoint"] == values["test_start"]
        <= values["test_end"] == values["exit_checkpoint"]
        and values["test_end"] - values["test_start"] == code_size
        and values["result_buffer"] != values["test_memory"]
    )


def build_elf(
    testcase: TestCase,
    output_path: Path,
    *,
    prologue_lines: tuple[str, ...] = (),
    bare_metal: bool = False,
    timeout_seconds: float | None = None,
) -> BuiltElf:
    require_execution_plane()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_path.with_suffix(".S")
    frame_contract = _direct_elf_frame_contract(testcase)
    source_path.write_text(
        assembly_source(testcase, prologue_lines=prologue_lines, bare_metal=bare_metal),
        encoding="utf-8",
    )
    toolchain_march, toolchain_mabi, toolchain_dropped = _toolchain_profile(testcase.isa_profile)
    if (
        frame_contract.get("frame_kind") == "trap-row"
        and "zicsr" not in toolchain_march.split("_")
    ):
        toolchain_march += "_zicsr"
    text_address = "0x80000000" if bare_metal else TEXT_ADDRESS
    command = (
        DEFAULT_CC,
        *(
            ("-Wl,-N",)
            if bare_metal or _requires_instruction_memory_support(testcase)
            else ()
        ),
        f"-march={toolchain_march}",
        f"-mabi={toolchain_mabi}",
        "-nostdlib",
        "-static",
        "-save-temps=obj",
        "-Wl,--build-id=none", "-Wl,--no-relax",
        "-Wl,-e,_start",
        f"-Wl,-Ttext={text_address}",
        str(source_path),
        "-o",
        str(output_path),
    )
    memory_size = len(_memory_bytes(testcase))
    temp_register, memory_register, output_register, _ = _observer_registers(testcase.isa_profile)
    result_map = {
        "testcase_id": testcase.testcase_id,
        "isa_profile": testcase.isa_profile,
        "compressed_enabled": _compressed_enabled(testcase.isa_profile),
        "text_address": text_address,
        "testcase_address_model": {
            "kind": ADDRESS_MODEL,
            "code_address": "test-section logical byte offset",
            "memory_region_address": "region logical byte offset",
            "runtime_binding": "direct_elf binds test_start, exit_checkpoint, and test_memory symbols",
        },
        "entry_checkpoint": "entry_checkpoint",
        "test_start": "test_start",
        "test_end": "test_end",
        "exit_checkpoint": "exit_checkpoint",
        "observation_contract": OBSERVATION_CONTRACT,
        "observer_saved_registers": sorted(_saved_register_offsets(testcase.isa_profile)),
        "observer_temp_registers_after_snapshot": [temp_register, memory_register, output_register],
        **frame_contract,
        "build_contract": {
            "isa_profile": testcase.isa_profile,
            "toolchain_march": toolchain_march,
            "toolchain_mabi": toolchain_mabi,
            "toolchain_dropped_suffixes": list(toolchain_dropped),
            "compressed_enabled": _compressed_enabled(testcase.isa_profile),
            "prologue_lines": list(prologue_lines),
            "bare_metal": bare_metal,
            "mixed_instruction_lengths": bool(
                testcase.instruction_meta
                and any(item.byte_length != 4 for item in testcase.instruction_meta)
            ),
        },
        "memory_size": memory_size,
        "observation_size": OBSERVATION_HEADER_SIZE + memory_size,
        "testcase_code_size": len(testcase.code_bytes),
        "testcase_code_sha256": hashlib.sha256(testcase.code_bytes).hexdigest(),
        "initial_gpr_sha256": canonical_digest({"gpr": list(testcase.initial_gpr)}),
        "memory_layout": _memory_region_layout(testcase.initial_memory_regions),
    }
    start = time.monotonic()
    build_timeout = BUILD_TIMEOUT_SEC
    if timeout_seconds is not None:
        build_timeout = max(0.001, min(float(timeout_seconds), BUILD_TIMEOUT_SEC))

    def finish(
        exit_code: int | None,
        timeout: bool,
        missing_toolchain: bool,
        stdout: str,
        stderr: str,
    ) -> BuiltElf:
        return BuiltElf(
            path=str(output_path),
            source_path=str(source_path),
            sha256=sha256_file(output_path) if output_path.exists() and exit_code == 0 else None,
            map=result_map,
            command=command,
            exit_code=exit_code,
            timeout=timeout,
            missing_toolchain=missing_toolchain,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        proc = subprocess.run(
            list(command),
            cwd=str(output_path.parent),
            timeout=build_timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        result = finish(
            proc.returncode,
            False,
            False,
            proc.stdout,
            proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        result = finish(
            None,
            True,
            False,
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        )
    except (FileNotFoundError, OSError) as exc:
        result = finish(
            None,
            False,
            True,
            "",
            (
                f"{exc}\n"
                "missing local RV64 cross-build toolchain; run real direct campaigns "
                "from framework/container/x86_64-lab or an equivalent Linux x86_64 lab. "
                "native-rv64 reference is remote via framework.native.rv64_runner."
            ),
        )

    if result.ok:
        try:
            if not is_riscv_elf(output_path) or output_path.read_bytes()[4] != (
                1 if _xlen(testcase.isa_profile) == 32 else 2
            ):
                return replace(
                    result,
                    sha256=None,
                    stderr=result.stderr + "\ncompiler did not produce a RISC-V ELF",
                )
        except (OSError, ValueError):
            return replace(
                result,
                sha256=None,
                stderr=result.stderr + "\ncompiler output ELF identity is unavailable",
            )
        runtime_labels = (
            "entry_checkpoint", "test_start", "test_end", "exit_checkpoint",
            "result_buffer", "test_memory",
        )
        try:
            runtime_symbol_offsets, runtime_symbol_addresses = symbol_offsets_from_elf(
                output_path,
                runtime_labels,
            )
        except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return replace(
                result,
                sha256=None,
                timeout=isinstance(exc, subprocess.TimeoutExpired),
                missing_toolchain=not isinstance(exc, subprocess.TimeoutExpired),
                stderr=result.stderr + f"\nnm unavailable: {exc}",
            )
        required_symbols = {"_start", *runtime_labels}
        if not required_symbols <= set(runtime_symbol_addresses):
            return replace(
                result,
                sha256=None,
                stderr=result.stderr + "\nnm returned incomplete runtime symbol map",
            )
        if not _runtime_symbol_map_is_valid(
            runtime_symbol_addresses, code_size=len(testcase.code_bytes)
        ):
            return replace(
                result,
                sha256=None,
                stderr=result.stderr + "\nnm returned invalid runtime symbol map",
            )
        result_map = {
            **result.map,
            "runtime_symbol_offsets": runtime_symbol_offsets,
            "runtime_symbol_addresses": runtime_symbol_addresses,
            "runtime_text_base": runtime_symbol_addresses.get("_start"),
        }
        result_map["frame_contract_sha256"] = _frame_contract_digest(result_map)
        result = replace(result, map=result_map)
    return result

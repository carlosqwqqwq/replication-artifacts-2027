"""RVGEN 单指令路线的统一 campaign 入口。"""

from collections.abc import Mapping
from dataclasses import replace
import os
from pathlib import Path
import random
import re
import shutil
import time

from framework.direct_case import TestCase, expected_trap_from_dataflow
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from framework.framework_coverage import coverage_observation_identity
from framework.adapters.contracts import is_execution_backend
from framework.execution_environment import ExecutionEnvironmentError
from framework.campaign_route import CampaignRoute
from framework.program_campaign import run_program_campaign
from framework.program_runner import backend_runner
from framework.riscv_semantic_references import (
    rvgen_vendor_reference_observation, sail_reference_observation,
)
from framework.riscv_csrs import (
    OFFICIAL_CSRS, OFFICIAL_RV32_ONLY_CSRS, csr_required_extensions,
)
from framework.riscv_catalog import OFFICIAL_ALL_CATALOG_FORMS
from framework.rvgen import generate_form
from framework.rvemi.program import _register_index
from framework.supply.model_candidate_supply import request_case_rewrite
from framework.rvgen.single_program import (
    _candidate_observer_gap, apply_single_hint, apply_single_profile, build_single_bare_program,
    build_single_program,
    build_single_profile, enumerate_single_rewrites, program_from_testcase,
    run_single_backend, single_profile_actions, single_rewrite_realizes,
)
from framework.rvgen.ledger import identify_testcase
from framework.spec_definedness import (
    enabled_extensions, instruction_effect_class, instruction_spec_for_generation,
    mabi_matches_isa,
)


_FORM_CASES: dict[tuple[str, object], tuple[TestCase, ...]] = {}
_CATALOG_ROOT_FORMS: dict[str, tuple[str, ...]] = {}
_CSR_BY_NAME = {
    name.strip('"').lower(): address
    for address, name in {**OFFICIAL_CSRS, **OFFICIAL_RV32_ONLY_CSRS}.items()
}


def _apply_single_hint_route(program, hint, _output_path, _candidates):
    return apply_single_hint(program, hint)


def _single_profile_realizes(_program, profile, _hint):
    return bool(getattr(profile, "dynamic_witness", False))


SINGLE_ROUTE = CampaignRoute(
    "single", enumerate_single_rewrites, _apply_single_hint_route,
    single_rewrite_realizes, build_single_profile, single_profile_actions,
    apply_single_profile, _single_profile_realizes,
)


def _rv32_qemu_binary(path: str | None) -> str | None:
    candidates = []
    if isinstance(path, str) and path.strip():
        original = Path(path)
        if "qemu-riscv64" in original.name:
            candidates.append(
                original.with_name(original.name.replace("qemu-riscv64", "qemu-riscv32"))
            )
        elif "qemu-riscv32" in original.name:
            candidates.append(original)
    if system := shutil.which("qemu-riscv32"):
        candidates.insert(0, Path(system))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def _target_runtime_identity(target_config: Mapping[str, object]) -> tuple[str | None, str | None]:
    """Return the identity of the executable selected for this Target run."""
    coverage = target_config.get("coverage_config")
    if not isinstance(coverage, Mapping) or not coverage.get("enabled", True):
        return (
            target_config.get("identity_digest")
            if isinstance(target_config.get("identity_digest"), str) else None,
            target_config.get("binary_sha256")
            if isinstance(target_config.get("binary_sha256"), str) else None,
        )
    binary_value = coverage.get("binary_path")
    identity_binary_value = coverage.get("identity_binary_path")
    if not isinstance(binary_value, str) or not isinstance(identity_binary_value, str):
        return (
            target_config.get("identity_digest")
            if isinstance(target_config.get("identity_digest"), str) else None,
            target_config.get("binary_sha256")
            if isinstance(target_config.get("binary_sha256"), str) else None,
        )
    try:
        runtime_identity = coverage_observation_identity(coverage)
    except (OSError, TypeError, ValueError):
        # Coverage setup falls back to the ordinary target binary at execution.
        # Keep the binding on that declared identity when the instrumented
        # binary or its sidecar cannot be verified.
        return (
            target_config.get("identity_digest")
            if isinstance(target_config.get("identity_digest"), str) else None,
            target_config.get("binary_sha256")
            if isinstance(target_config.get("binary_sha256"), str) else None,
        )
    return runtime_identity.identity_digest, runtime_identity.binary_sha256


def _rewrite_hint(value: object) -> object | None:
    if not isinstance(value, Mapping) or "hint" not in value:
        return value
    hint = value.get("hint")
    return hint if isinstance(hint, Mapping) else None


def _testcase_state(testcase: TestCase) -> str:
    meta = testcase.dataflow_meta
    facts = meta.get("realized_facts", {})
    state = facts.get("state_domain") or meta.get("state_domain") or "gpr"
    schema = meta.get("generation_owner", {}).get("schema", {})
    if state == "scalar" and isinstance(schema, Mapping):
        state = "fpr" if schema.get("writeback") == "fpr" else "gpr"
    return str(state)


def _mnemonic(value: object) -> str:
    return str(value).strip().split(None, 1)[0].lower().replace("_", ".")


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if len(text) > 1 and text.startswith(("x", "f", "v")) \
            and text[1:].split()[0].split("(", 1)[0].isdigit():
        return int(text[1:].split()[0].split("(", 1)[0])
    if re.fullmatch(r"[+-]?0[0-7]+", text):
        return int(text, 8)
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(text, 16)
        except ValueError:
            return None


def _sequence_register_value(value: str) -> int | None:
    index = _register_index(value)
    return index if index is not None else (
        int(value[1:]) if len(value) > 1 and value[0] in "fv" and value[1:].isdigit()
        else None
    )


def _hex_bytes(value: object) -> tuple[bytes, ...]:
    text = f"{value:x}" if isinstance(value, int) else str(value).strip().lower()
    bare = text.removeprefix("0x")
    if not bare:
        return ()
    bare = bare if len(bare) % 2 == 0 else f"0{bare}"
    try:
        raw = bytes.fromhex(bare)
    except ValueError:
        return ()
    values = [raw]
    try:
        values.append(int(bare, 16).to_bytes(len(raw), "little"))
    except OverflowError:
        pass
    return tuple(dict.fromkeys(values))


def _raw_bytes(value: object, width: int | None = None) -> bytes | None:
    text = str(value).strip().lower()
    if not re.fullmatch(r"0x[0-9a-f]+", text):
        return None
    digits = text[2:]
    size = width or ((len(digits) + 1) // 2)
    try:
        return int(digits, 16).to_bytes(size, "little")
    except (OverflowError, ValueError):
        return None


def _memory_match(
    regions: Mapping[str, bytes], value: object, *, region: object = None,
    offset: object = None,
) -> bool:
    data = regions.get(str(region)) if region is not None else None
    pool = (data,) if data is not None else tuple(regions.values())
    needles = _hex_bytes(value)
    if not needles:
        return False
    start = _int_value(offset) if offset is not None else None
    if offset is not None and start is None:
        return False
    if start is not None and start < 0:
        return False
    return any(
        any(
            (start is None and needle in blob)
            or (start is not None and blob[start:start + len(needle)] == needle)
            for needle in needles
        )
        for blob in pool
    )


def _code_match(testcase: TestCase, key: str, value: object) -> bool:
    width = {"raw16_hex": 2, "raw32_hex": 4, "raw_word_hex": 4}.get(key)
    needle = _raw_bytes(value, width) if key != "raw16_hex" else None
    needles = (needle,) if needle is not None else tuple(
        item for item in _hex_bytes(value) if len(item) == width
    )
    risk_bytes = tuple(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length]
        for item in testcase.instruction_meta if "risk" in item.tags
    )
    return bool(needles) and any(needle == blob for needle in needles for blob in risk_bytes)


def _risk_operand(item: object, role: str) -> int | None:
    value = getattr(item, role, None)
    if value is not None:
        return int(value)
    return next(
        (_int_value(raw) for name, raw in item.operand_fields if name == role), None
    )


def _encoded_risk_operand(testcase: TestCase, item: object, role: str) -> int | None:
    value = _risk_operand(item, role)
    if value is not None or role != "rs2" or getattr(item, "byte_length", 0) != 4:
        return value
    word = int.from_bytes(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length],
        "little",
    )
    return (word >> 20) & 0x1F


def _state_value_hex_matches(
    testcase: TestCase, risk: tuple[object, ...], key: str, value: object,
    regions: Mapping[str, bytes], state: Mapping[str, object],
) -> bool:
    if key in {"raw16_hex", "raw32_hex", "raw_word_hex"}:
        return _code_match(testcase, key, value)
    if key.startswith("memory_") or key in {
        "qnan_single_hex", "finite_single_hex", "qnan_double_hex",
        "finite_double_hex", "nonboxed_single_hex", "second_single_hex",
        "left_single_hex", "right_single_hex", "numerator_single_hex",
        "denominator_single_hex",
    }:
        offset = state.get("memory_offset")
        if key.startswith("memory_values_"):
            offsets = state.get("memory_offsets", ())
            if not isinstance(offsets, (list, tuple)):
                return False
            return isinstance(value, (list, tuple)) and bool(value) and len(offsets) == len(value) and all(
                _memory_match(
                    regions, item, region=state.get("memory_region"),
                    offset=offsets[index] if index < len(offsets) else None,
                )
                for index, item in enumerate(value)
            )
        return _memory_match(
            regions, value, region=state.get("memory_region"), offset=offset,
        )
    if key == "fcsr_value_hex":
        try:
            expected = int(str(value).strip().lower().removeprefix("0x"), 16)
        except ValueError:
            return False
        return any(
            _mnemonic(item.mnemonic).startswith("csr")
            and _risk_operand(item, "csr") == _CSR_BY_NAME.get("fcsr")
            and (register := _risk_operand(item, "rs1")) is not None
            and testcase.initial_gpr[register] == expected
            for item in risk
        )
    if key.endswith("_value_hex"):
        role = key.removesuffix("_value_hex")
        expected = _int_value(value)
        return expected is not None and any(
            (register := _risk_operand(item, role)) is not None
            and register < len(testcase.initial_gpr)
            and testcase.initial_gpr[register] == expected
            for item in risk
        )
    return False


def _source_state_matches(
    testcase: TestCase, risk: tuple[object, ...], state: Mapping[str, object],
    value: object, regions: Mapping[str, bytes],
) -> bool:
    source = str(state.get("source", "")).lower()
    source_match = re.fullmatch(r"(?:gpr\.)?x(\d+)", source)
    if source_match:
        expected = _int_value(value)
        register = int(source_match.group(1))
        bound = any(
            _risk_operand(item, role) == register
            and getattr(item, f"{role}_domain", "gpr") == "gpr"
            for item in risk for role in ("rs1", "rs2", "rs3")
        )
        return bound and register < len(testcase.initial_gpr) and expected is not None \
            and testcase.initial_gpr[register] == expected
    if source.startswith("test-memory["):
        offset = source.removeprefix("test-memory[").rstrip("]")
        return _memory_match(
            regions, value, region="test-memory", offset=offset,
        )
    if source.startswith("f") and source[1:].isdigit():
        register = int(source[1:])
        for index, role in enumerate(("rs1", "rs2", "rs3")):
            if any(
                _risk_operand(item, role) == register
                and getattr(item, f"{role}_domain", "gpr") == "fpr"
                for item in risk
            ):
                return _memory_match(
                    regions, value, region=state.get("memory_region"),
                    offset=(16, 32, 48)[index],
                )
    return _memory_match(regions, value) if not source else False


def _risk_field(testcase: TestCase, item: object, field: str) -> int | None:
    fields = dict(getattr(item, "operand_fields", ()))
    if field in fields:
        return fields[field]
    if field in {"rd", "rs1", "rs2", "rs3"}:
        return _risk_operand(item, field)
    if field in {"immediate", "imm12"}:
        return getattr(item, "immediate", None)
    if field in {"funct7", "funct3", "opcode"}:
        word = int.from_bytes(
            testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length],
            "little",
        )
        return {
            "funct7": (word >> 25) & 0x7F,
            "funct3": (word >> 12) & 0x7,
            "opcode": word & 0x7F,
        }[field]
    if field == "imm":
        for name, width in (("imm12s", 12), ("bimm12", 13)):
            if name in fields:
                return int(fields[name]) & ((1 << width) - 1)
        word = int.from_bytes(
            testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length],
            "little",
        )
        return word >> 20
    match = re.fullmatch(r"(.+)\[(\d+)\]", field)
    if match and (value := _risk_field(testcase, item, match.group(1))) is not None:
        return (value >> int(match.group(2))) & 1
    return None


def _reserved_fields_match(
    testcase: TestCase, risk: tuple[object, ...], value: object,
) -> bool:
    for raw in str(value).split(","):
        match = re.fullmatch(
            r"\s*([A-Za-z_][A-Za-z0-9_.]*)(?:\[(\d+)\])?\s*=\s*"
            r"(x\d+|[-+]?(?:0x)?[0-9A-Fa-f]+)\s*", raw,
        )
        if match is None:
            return False
        field = match.group(1)
        if match.group(2) is not None:
            field += f"[{match.group(2)}]"
        expected = _int_value(match.group(3))
        if expected is None or not any(
            _risk_field(testcase, item, field) == expected for item in risk
        ):
            return False
    return True


def _load_slots(testcase: TestCase, state: Mapping[str, object]) -> tuple[bytes, ...]:
    region_id = state.get("memory_region")
    region = next(
        (item for item in testcase.initial_memory_regions if item.region_id == region_id),
        None,
    )
    if region is None and len(testcase.initial_memory_regions) == 1:
        region = testcase.initial_memory_regions[0]
    if region is None:
        return ()
    slots = []
    for item in testcase.instruction_meta:
        if "producer" not in item.tags or instruction_effect_class(item.mnemonic) != "load":
            continue
        width = {"b": 1, "h": 2, "w": 4, "d": 8, "q": 16}.get(
            _mnemonic(item.mnemonic)[-1:],
        )
        base = _risk_operand(item, "rs1")
        if width is None or base is None or item.immediate is None:
            continue
        offset = testcase.initial_gpr[base] + item.immediate - region.address
        if 0 <= offset <= len(region.data) - width:
            slots.append(region.data[offset:offset + width])
    return tuple(slots)


def _slot_matches(slot: bytes, value: object) -> bool:
    return any(len(needle) == len(slot) and needle == slot for needle in _hex_bytes(value))


def _input_state_matches(
    testcase: TestCase, state: Mapping[str, object],
) -> bool:
    slots = _load_slots(testcase, state)
    pairs = (
        ("qnan_single_hex", "finite_single_hex"),
        ("qnan_double_hex", "finite_double_hex"),
        ("numerator_single_hex", "denominator_single_hex"),
        ("left_single_hex", "right_single_hex"),
        ("nonboxed_single_hex", "second_single_hex"),
    )
    checked = set()
    for left, right in pairs:
        if left not in state or right not in state:
            continue
        checked |= {left, right}
        if len(slots) < 2 or not _slot_matches(slots[0], state[left]) \
                or not _slot_matches(slots[1], state[right]):
            return False
    for key in set(state) - checked:
        if isinstance(key, str) and key.endswith(("_single_hex", "_double_hex")) and not any(
            _slot_matches(slot, state[key]) for slot in slots
        ):
            return False
    return True


def _sequence_rounding_mode(sequence: object, risk_mnemonic: str) -> int | None:
    modes = {"rne": 0, "rtz": 1, "rdn": 2, "rup": 3, "rmm": 4, "dyn": 7}
    for raw in sequence if isinstance(sequence, (list, tuple)) else ():
        if _mnemonic(raw) != risk_mnemonic:
            continue
        match = re.search(r"\b(rne|rtz|rdn|rup|rmm|dyn)\b", raw.lower())
        return modes[match.group(1)] if match else None
    return None


def _sequence_encoding_matches(
    testcase: TestCase, sequence: object, risk: tuple[object, ...],
) -> bool:
    for raw in sequence if isinstance(sequence, (list, tuple)) else ():
        if not re.search(r"\b(?:encoding|raw(?:16|32)?|word)\b", raw.lower()):
            continue
        match = re.search(r"\b0x([0-9a-fA-F]+)\b", raw)
        if match is None:
            return False
        encoded = _raw_bytes(match.group(0))
        if encoded is None or not any(
            encoded == testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length]
            for item in risk
        ):
            return False
    return True


def _sequence_immediate_matches(
    testcase: TestCase, sequence: object, risk: tuple[object, ...],
) -> bool:
    for raw in sequence if isinstance(sequence, (list, tuple)) else ():
        if re.search(r"\b(?:imm|nzuimm)\s*=", raw.lower()) and not re.search(
            r"\b(?:imm|nzuimm)\s*=\s*[+-]?(?:0x[0-9a-fA-F]+|\d+)", raw,
        ):
            return False
        match = re.search(r"\b(?:imm|nzuimm)\s*=\s*"
                          r"([+-]?(?:0x[0-9a-fA-F]+|\d+))", raw)
        if match is None:
            continue
        expected = _int_value(match.group(1))
        if expected is None:
            return False
        actual = tuple(_risk_field(testcase, item, "immediate") for item in risk)
        if _mnemonic(raw) == "c.lui":
            if not any(value in {expected, expected << 12} for value in actual):
                return False
        elif expected not in actual:
            return False
    return True


def _sequence_risk_operands_match(
    testcase: TestCase, sequence: object, risk: tuple[object, ...],
) -> bool:
    for raw in sequence if isinstance(sequence, (list, tuple)) else ():
        text = str(raw).lower()
        mnemonic = _mnemonic(raw)
        candidates = tuple(
            item for item in risk
            if _mnemonic(item.mnemonic) == mnemonic
        )
        if candidates and any("csr-state" in item.tags for item in candidates):
            continue
        if candidates and not any(
            "risk" in item.tags or "producer" in item.tags or "observable" in item.tags
            for item in candidates
        ):
            continue
        if not candidates:
            continue
        assignments = tuple(
            (role, int(value))
            for role, value in re.findall(
                r"\b(rd|rs[123]|vd|vs[123])\s*=\s*[xfv](\d+)\b", text,
            )
        )
        if assignments:
            if not any(
                all(_risk_operand(item, role) == value for role, value in assignments)
                for item in candidates
            ):
                return False
            continue
        parts = text.split(maxsplit=1)
        operand_text = re.split(
            r"\||\b(?:with|at|on|reserved|encoding)\b",
            parts[1] if len(parts) > 1 else "", 1,
        )[0]
        if "(" in operand_text and instruction_effect_class(candidates[0].mnemonic) not in {
            "load", "store", "atomic",
        } and not mnemonic.startswith("cbo."):
            operand_text = operand_text.split("(", 1)[0]
        registers = tuple(
            index for value in re.findall(
                r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9]*\b", operand_text,
            ) if (index := _sequence_register_value(value)) is not None
        )
        if not registers:
            if len(parts) > 1 and re.fullmatch(r"[A-Za-z_][\w.-]*", parts[1].strip()):
                return False
            continue
        for item in candidates:
            effect = instruction_effect_class(item.mnemonic)
            if mnemonic.startswith("cbo.") or mnemonic in {"c.jalr", "c.srli"}:
                roles = ("rs1",)
            elif mnemonic.startswith("fli."):
                roles = ("rd",)
            elif mnemonic.startswith("lr."):
                roles = ("rd", "rs1")
            elif mnemonic.startswith("csr"):
                roles = ("rd",) if mnemonic.endswith(("wi", "si", "ci")) else ("rd", "rs1")
            elif mnemonic.startswith("v"):
                roles = ("vd", "vs1", "vs2", "vs3")[:len(registers)]
            elif mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}:
                roles = ("rs1", "rs2")
            elif mnemonic in {"beqz", "bnez"}:
                roles = ("rs1",)
            elif effect == "load":
                roles = ("rd", "rs1")
            elif effect == "store":
                roles = ("rs2", "rs1")
            elif effect == "atomic":
                roles = ("rd", "rs1", "rs2")
            else:
                roles = ("rd", "rs1", "rs2", "rs3")[:len(registers)]
            if len(registers) != len(roles) or any(
                _risk_operand(item, role) != value for role, value in zip(roles, registers)
            ):
                continue
            prefix = operand_text
            numbers = re.findall(
                r"(?<![A-Za-z0-9_])-?(?:0x[0-9a-f]+|\d+)(?![A-Za-z0-9_])", prefix,
            )
            actual_immediate = _risk_field(testcase, item, "immediate")
            if effect in {"load", "store", "atomic"} or mnemonic.startswith("cbo."):
                offset = re.search(
                    r"(?<![A-Za-z0-9_])-?(?:0x[0-9a-f]+|\d+)\s*\(", prefix,
                )
                if offset and actual_immediate is not None \
                        and actual_immediate != _int_value(offset.group(0).split("(", 1)[0]):
                    continue
            elif numbers and not mnemonic.startswith("fli.") \
                    and actual_immediate is not None \
                    and actual_immediate != _int_value(numbers[-1]):
                continue
            break
        else:
            return False
    return True


def _case_matches_recipe(testcase: TestCase, generation: Mapping[str, object]) -> bool:
    state = generation.get("initial_state", {})
    sequence = generation.get("sequence", ())
    raw_encoding = isinstance(state, Mapping) and any(
        key in state for key in ("raw16_hex", "raw32_hex", "raw_word_hex")
    )
    ignored = {"expected", "store", "trap", "observation", "returned"}
    if raw_encoding:
        ignored |= {"op", "reserved", "encoding", "illegal"}
    sequence_mnemonics = tuple(
        _mnemonic(item) for item in sequence
        if str(item).strip() and _mnemonic(item) not in ignored
    )
    if any(
        mnemonic not in {"op", "mop.r.n", "c.mop.n"}
        and instruction_spec_for_generation(mnemonic) is None
        for mnemonic in sequence_mnemonics
    ):
        return False
    if not sequence_mnemonics and not raw_encoding:
        return False
    actual = tuple(_mnemonic(item.mnemonic) for item in testcase.instruction_meta)
    risk_mnemonics = {_mnemonic(item.mnemonic) for item in testcase.instruction_meta if "risk" in item.tags}
    risk = tuple(item for item in testcase.instruction_meta if "risk" in item.tags)
    if not raw_encoding and not any(
        _mnemonic(item) in risk_mnemonics for item in sequence
    ):
        return False

    meta = testcase.dataflow_meta
    boundary_labels = set()
    for row in meta.get("risk_contracts", ()) if isinstance(meta, Mapping) else ():
        if isinstance(row, Mapping) and row.get("boundary_class") not in (None, ""):
            boundary_labels.add(str(row["boundary_class"]))
    for key in ("vector_boundary", "memory_boundary"):
        value = meta.get("generation_realization", {}).get(key) if isinstance(meta, Mapping) else None
        if value not in (None, ""):
            boundary_labels.add(str(value))
    semantic = meta.get("semantic_point", {}) if isinstance(meta, Mapping) else {}
    if isinstance(semantic, Mapping) and semantic.get("primary_boundary_class") not in (None, ""):
        boundary_labels.add(str(semantic["primary_boundary_class"]))

    sequence_text = " ".join(map(str, sequence))
    for role, raw in re.findall(r"\b(rd|rs[123])\s*=\s*[xfv](\d+)\b", sequence_text):
        expected = int(raw)
        if not any(
            _risk_operand(item, role) == expected
            for item in testcase.instruction_meta if "risk" in item.tags
        ):
            return False
    def ordered(wanted):
        position = 0
        for item in actual:
            if position < len(wanted) and item == wanted[position]:
                position += 1
        return position == len(wanted)

    sequence_requires_scaffold = len(sequence_mnemonics) > 1 or (
        isinstance(state, Mapping) and any(
            isinstance(key, str) and key.startswith("expected") for key in state
        )
    )
    if sequence_requires_scaffold and not raw_encoding and tuple(sequence_mnemonics) != actual:
        risk_sequence = tuple(item for item in sequence_mnemonics if item in risk_mnemonics)
        declared_actual = tuple(item for item in sequence_mnemonics if item in actual)
        state_has_auxiliary = isinstance(state, Mapping) and any(
            isinstance(key, str) and (
                key.startswith("memory_") or key.endswith("_hex")
                or re.fullmatch(r"[xfv]\d+|rd|rs[123]|boundary", key) is not None
                or key in {
                    "source", "source_value", "rounding_mode", "fflags",
                    "literal_index", "csr", "csr_address", "csr_extension",
                    "rm", "imm12", "immediate",
                }
            )
            for key in state
        )
        expected_output = isinstance(state, Mapping) and any(
            key in {"expected_result", "expected_single", "expected_upper_word"}
            for key in state
        )
        if expected_output or not (
            state_has_auxiliary and ordered(risk_sequence) and ordered(declared_actual)
        ):
            return False
    encoding_sequence = tuple(
        raw for raw in sequence
        if not raw_encoding or re.search(r"\b0x[0-9a-fA-F]+\b", str(raw))
    )
    if not _sequence_encoding_matches(testcase, encoding_sequence, risk):
        return False
    if not _sequence_immediate_matches(testcase, sequence, risk):
        return False
    if not _sequence_risk_operands_match(
        testcase, sequence,
        tuple(testcase.instruction_meta) if sequence_requires_scaffold else risk,
    ):
        return False
    if (
        sequence_mode := _sequence_rounding_mode(sequence, next(iter(risk_mnemonics), ""))
    ) is not None and not any(
        _risk_operand(item, "rm") == sequence_mode for item in risk
    ):
        return False
    if not isinstance(state, Mapping):
        return False
    if "entry" in state and state["entry"] not in (None, "", "_start"):
        return False
    if "memory_offset" in state and (
        (offset := _int_value(state["memory_offset"])) is None or offset < 0
    ):
        return False
    if "memory_offsets" in state and (
        not isinstance(state["memory_offsets"], (list, tuple))
        or any(
            (offset := _int_value(item)) is None or offset < 0
            for item in state["memory_offsets"]
        )
    ):
        return False
    if "memory_offsets" in state and "memory_values_hex" not in state:
        return False
    regions = {item.region_id: item.data for item in testcase.initial_memory_regions}
    region = state.get("memory_region")
    if region not in (None, "") and (not isinstance(region, str) or region not in regions):
        return False
    if not _input_state_matches(testcase, state):
        return False
    ignored = {
        "expected",
        "expected_result", "expected_single",
        "memory_offset", "memory_offsets", "memory_region",
        "source",
    }
    known = ignored | {
        "address", "alias", "base_profile", "boundary", "csr", "csr_address",
        "csr_extension", "entry", "expected", "expected_result", "expected_single",
        "expected_upper_word", "fflags", "fixed_rs2", "imm12", "immediate",
        "instruction", "literal_index", "memory_offset", "memory_offsets",
        "memory_region", "privilege", "rd", "rd_initial", "rd_pair", "rd_rs1",
        "reserved_field", "reserved_fields", "reserved_frm", "rm", "root_source_bits",
        "rounding_mode", "rs1", "rs1_initial", "rs1_rs2", "rs1_value", "rs2",
        "rs2_initial", "rs2_pair", "rs3", "rs3_initial", "shamt", "shamtd", "shamtw",
        "source", "source_bits", "source_value",
    }
    for key, value in state.items():
        if not isinstance(key, str):
            return False
        if key in ignored:
            continue
        if key == "boundary":
            if not isinstance(value, str) or value not in boundary_labels:
                return False
            continue
        if key == "instruction":
            expected = _mnemonic(value)
            if not any(
                _mnemonic(item.mnemonic) == expected
                or expected.startswith("c.mop.")
                and _mnemonic(item.mnemonic) == "c.mop.n"
                and _risk_field(testcase, item, "c_mop_t") == _int_value(expected.rsplit(".", 1)[-1])
                for item in risk
            ):
                return False
            continue
        if key == "alias":
            alias = _mnemonic(value)
            risk_names = {_mnemonic(item.mnemonic) for item in risk}
            if (
                not isinstance(value, str)
                or instruction_spec_for_generation(value) is None
                or alias not in risk_names
                and alias.replace("h.", ".") not in risk_names
            ):
                return False
            continue
        if key == "privilege":
            actual = meta.get("generation_realization", {}).get("privilege_class")
            if not isinstance(value, str) or str(actual).lower() != value.strip().lower():
                return False
            continue
        if key == "expected_upper_word":
            expected = _int_value(value)
            if expected is None or not 0 <= expected <= 0xFFFFFFFF or not any(
                len(slot) >= 8 and int.from_bytes(slot[:8], "little") >> 32 == expected
                for slot in _load_slots(testcase, state)
            ):
                return False
            continue
        if key in {"reserved_field", "reserved_fields"}:
            if not _reserved_fields_match(testcase, risk, value):
                return False
            continue
        if key == "base_profile":
            if not isinstance(value, str) or value.lower() != testcase.isa_profile.lower():
                return False
            continue
        if key == "csr_extension":
            if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9]*", value.lower()) is None \
                    or value.lower() in enabled_extensions(testcase.isa_profile):
                return False
            if not any(
                value.lower() in csr_required_extensions(_risk_operand(item, "csr"))
                for item in risk if _risk_operand(item, "csr") is not None
            ):
                return False
            continue
        if key == "fflags":
            expected = _int_value(value)
            fcsr = _int_value(state.get("fcsr_value_hex")) if "fcsr_value_hex" in state else None
            observed = next(
                (
                    _risk_operand(item, "zimm5")
                    for item in testcase.instruction_meta
                    if _mnemonic(item.mnemonic) == "csrrwi"
                    and _risk_operand(item, "csr") == _CSR_BY_NAME.get("fflags")
                ),
                None,
            )
            if expected is None or not 0 <= expected <= 0x1F \
                    or fcsr is not None and fcsr & 0x1F != expected \
                    or fcsr is None and observed is not None and observed != expected \
                    or fcsr is None and observed is None:
                return False
            continue
        if key == "reserved_frm":
            expected = _int_value(value)
            fcsr = _int_value(state.get("fcsr_value_hex")) if "fcsr_value_hex" in state else None
            if expected is None or expected not in {5, 6} \
                    or fcsr is None and not any(
                        _risk_operand(item, "rm") == expected for item in risk
                    ) \
                    or fcsr is not None and (fcsr >> 5) & 0x7 != expected:
                return False
            continue
        if key == "imm12":
            expected = _int_value(value)
            if expected is None or not -2048 <= expected <= 2047 or not any(
                _risk_field(testcase, item, "imm12") == expected for item in risk
            ):
                return False
            continue
        if key == "address":
            expected = _int_value(value)
            if expected is None or not any(
                (instruction_effect_class(item.mnemonic) in {"load", "store", "atomic"}
                 or _mnemonic(item.mnemonic).startswith("cbo."))
                and item.rs1 is not None
                and (testcase.initial_gpr[item.rs1] + int(item.immediate or 0))
                & ((1 << (32 if testcase.isa_profile.startswith("rv32") else 64)) - 1)
                == expected
                for item in risk
            ):
                if expected != testcase.code_address or any(
                    instruction_effect_class(item.mnemonic) in {"load", "store", "atomic"}
                    or _mnemonic(item.mnemonic).startswith("cbo.") for item in risk
                ):
                    return False
            continue
        if key not in {"csr", "csr_address"} and not (
            key in {"rd", "rs1", "rs2", "rs3"} and _int_value(value) is not None
            and _int_value(value) > 31
        ) and any(
            key in dict(getattr(item, "operand_fields", ())) for item in risk
        ):
            expected = _int_value(value)
            expected_domain = {
                "x": "gpr", "f": "fpr", "v": "vpr",
            }.get(value.strip().lower()[:1]) if isinstance(value, str) else None
            if expected is None or not any(
                _risk_field(testcase, item, key) == expected
                and (expected_domain is None
                     or getattr(item, f"{key}_domain", "gpr") == expected_domain)
                for item in risk
            ):
                return False
            continue
        if key not in known and not key.endswith("_hex") \
                and not re.fullmatch(r"[xfv]\d+", key):
            return False
        if key.startswith("x") and key[1:].isdigit():
            expected = _int_value(value)
            register = int(key[1:])
            if register >= len(testcase.initial_gpr) or expected is None \
                    or testcase.initial_gpr[register] != expected:
                return False
            continue
        if re.fullmatch(r"[fv]\d+", key):
            return False
        if key.endswith("_hex"):
            if not _state_value_hex_matches(testcase, risk, key, value, regions, state):
                return False
            continue
        if key in {"csr", "csr_address"}:
            expected = _int_value(value) if key == "csr_address" else _CSR_BY_NAME.get(
                str(value).strip('"').lower(), _int_value(value)
            )
            if expected is None or not any(
                _risk_operand(item, "csr") == expected for item in risk
            ):
                return False
            continue
        if key in {"source_value", "source_bits"}:
            if not _source_state_matches(testcase, risk, state, value, regions):
                return False
            continue
        if key == "rounding_mode":
            expected = {"rne": 0, "rtz": 1, "rdn": 2, "rup": 3, "rmm": 4, "dyn": 7}.get(
                str(value).lower(), _int_value(value)
            )
            if expected is None:
                return False
            if expected == 1 and any(
                _mnemonic(item.mnemonic) in {"fcvtmod.w.d", "fcvtmod.w.s"}
                for item in risk
            ):
                continue
            if expected != 7 and any(
                _risk_operand(item, "rm") == 7 for item in risk
            ) and any(
                _mnemonic(item.mnemonic) in {"csrrwi", "csrrw"}
                and _risk_operand(item, "csr") == _CSR_BY_NAME.get("frm")
                and (
                    _risk_operand(item, "zimm5") == expected
                    or _risk_operand(item, "rs1") == expected
                )
                for item in testcase.instruction_meta
            ):
                continue
            if not any(_risk_operand(item, "rm") == expected for item in risk):
                return False
            continue
        if key in {"rd_initial", "rs1_initial", "rs2_initial", "rs3_initial"}:
            role = key.removesuffix("_initial")
            expected = _int_value(value)
            if expected is None or not any(
                (register := _risk_operand(item, role)) is not None
                and testcase.initial_gpr[register] == expected
                for item in risk
            ):
                return False
            continue
        if key == "literal_index":
            expected = _int_value(value)
            if expected is None or not any(_risk_operand(item, "rs1") == expected for item in risk):
                return False
            continue
        if key == "fixed_rs2":
            expected = _int_value(value)
            encoded = tuple(
                _encoded_risk_operand(testcase, item, "rs2") for item in risk
            )
            if expected is None or not any(
                item == expected
                for item in encoded
            ):
                return False
            continue
        if key in {"rd_pair", "rs2_pair"}:
            match = re.fullmatch(
                r"x(\d+)/x(\d+)\s*=\s*([^/]+)/([^/]+)", str(value),
            )
            if match is None:
                return False
            left, right = (int(match.group(1)), int(match.group(2)))
            left_value, right_value = (_int_value(match.group(3)), _int_value(match.group(4)))
            role = "rd" if key == "rd_pair" else "rs2"
            if right != left + 1 or not 0 <= left < 31 or left_value is None or right_value is None or not any(
                _risk_operand(item, role) == left for item in risk
            ) or left >= len(testcase.initial_gpr) or right >= len(testcase.initial_gpr) \
                    or testcase.initial_gpr[left] != left_value \
                    or testcase.initial_gpr[right] != right_value:
                return False
            continue
        if key in {"rs1", "rs2", "rs3"} and _int_value(value) is not None \
                and _int_value(value) > 31:
            expected = _int_value(value)
            index = {"rs1": 0, "rs2": 1, "rs3": 2}[key]
            if not any(
                _risk_operand(item, key) is not None
                and _memory_match(
                    regions, value, region=state.get("memory_region"),
                    offset=(16, 32, 48)[index],
                )
                for item in risk
            ):
                return False
            continue
        if key in {"rd", "rs1", "rs2", "rs3", "rm", "immediate", "shamt", "shamtw", "shamtd"}:
            expected = _int_value(value)
            if not any(
                expected is not None and (
                    _risk_operand(item, key) == expected
                    or key == "immediate" and item.immediate == expected
                    or key in {"shamt", "shamtw", "shamtd"} and item.immediate == expected
                )
                for item in risk
            ):
                return False
            continue
        if key == "rs1_value":
            register = next((item.rs1 for item in risk if item.rs1 is not None), None)
            expected = _int_value(value)
            if register is None or expected is None or testcase.initial_gpr[register] != expected:
                return False
            continue
        if key in {"root_source_bits"}:
            needles = _hex_bytes(value)
            slots = _load_slots(testcase, state)
            if not needles or not slots or not any(
                len(needle) == len(slot) for needle in needles for slot in slots
            ):
                return False
            continue
        if key in {"rd_rs1", "rd_rs2", "rs1_rs2"}:
            if key == "rd_rs1" and isinstance(value, str) and "=" in value:
                register_text, expected_text = value.split("=", 1)
                register = _int_value(register_text)
                expected = _int_value(expected_text)
                if register is None or expected is None or register >= len(testcase.initial_gpr):
                    return False
                if not any(
                    _risk_operand(item, "rd") == register
                    and _risk_operand(item, "rs1") == register
                    for item in risk
                ) or testcase.initial_gpr[register] != expected:
                    return False
                continue
            left, right = key.split("_")
            if not any(
                getattr(item, left) is not None and getattr(item, left) == getattr(item, right)
                for item in risk
            ):
                return False
    return True


def _specialize_memory_state(testcase: TestCase, state: Mapping[str, object]) -> TestCase | None:
    region_id = state.get("memory_region")
    region = next((item for item in testcase.initial_memory_regions if item.region_id == region_id), None)
    if region is None and region_id in (None, "") and len(testcase.initial_memory_regions) == 1:
        region = testcase.initial_memory_regions[0]
        region_id = region.region_id
    if region is None:
        return None
    values = state.get("memory_values_hex")
    if values is None:
        values = (state.get("memory_value_hex"),)
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) for item in values):
        return None
    offsets = state.get("memory_offsets")
    if offsets is None:
        offsets = (state.get("memory_offset", 0),)
    if not isinstance(offsets, (list, tuple)) or len(offsets) != len(values):
        return None
    data = bytearray(region.data)
    for raw, offset in zip(values, offsets):
        try:
            text = raw.removeprefix("0x")
            text = text if len(text) % 2 == 0 else f"0{text}"
            offset, payload = _int_value(offset), bytes.fromhex(text)
        except (TypeError, ValueError):
            return None
        if offset is None or offset < 0 or offset + len(payload) > len(data):
            return None
        data[offset:offset + len(payload)] = payload
    regions = tuple(
        replace(item, data=bytes(data)) if item.region_id == region_id else item
        for item in testcase.initial_memory_regions
    )
    return identify_testcase(replace(testcase, initial_memory_regions=regions))


def _specialized_boundary_matches(testcase: TestCase, state: Mapping[str, object]) -> bool:
    boundary = state.get("boundary")
    if boundary in (None, ""):
        return True
    contracts = testcase.dataflow_meta.get("risk_contracts", ())
    if len(contracts) != 1 or not isinstance(contracts[0], Mapping):
        return False
    form_name = testcase.generation_rule_id.partition(":")[0]
    from framework.riscv_encoding import form_for_mnemonic
    from framework.rvgen.admission import admit_testcase
    from framework.rvgen.effects import generation_schema_for_form
    from framework.rvgen.intents import boundary_intents
    from framework.spec_definedness import SemanticRealization
    form = form_for_mnemonic(form_name)
    realization = testcase.dataflow_meta.get("generation_realization")
    if form is None or not isinstance(realization, Mapping):
        return False
    try:
        realization = SemanticRealization.from_dict(realization)
        intents = boundary_intents(
            form, generation_schema_for_form(form), realization,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return any(
        intent.lane == contracts[0].get("lane")
        and intent.boundary_class == boundary
        and admit_testcase(testcase, intent).admitted
        for intent in intents
    )


def _lane_accepts_profile(lane_isa: str, case_profile: object) -> bool:
    # Catalog 每条指令只声明自己的扩展组合（mul -> rv64im，c_add -> rv64ic，
    # csrrw -> rv64i_zicsr）。字面相等会让 rv64imc/rv64imafd 这类加宽 lane
    # 得到空池，也让 rv64i 丢掉 CSR、压缩与乘除边界。改判 lane 的扩展闭包
    # 是否覆盖 case 的扩展闭包，xlen 必须一致。
    if not isinstance(case_profile, str) or not case_profile:
        return False
    if lane_isa[:4] != case_profile[:4]:
        return False
    try:
        return set(enabled_extensions(case_profile)) <= set(enabled_extensions(lane_isa))
    except ValueError:
        return False


def _catalog_root_forms(isa: str) -> tuple[str, ...]:
    # 生成池是冻结 catalog 的全集。ISA/profile 和目标能力只决定后续
    # 是否能执行；不能在单指令生成阶段先把 form 丢掉。这样 RV32、P、V、
    # vendor 和特权编码都会得到自己的可审计 generation-gap，而不是静默
    # 消失在根池之外。
    cached = _CATALOG_ROOT_FORMS.get(isa)
    if cached is not None:
        return cached
    result = tuple(form.mnemonic for form in OFFICIAL_ALL_CATALOG_FORMS)
    _CATALOG_ROOT_FORMS[isa] = result
    return result


def _catalog_root_case(
    isa: str, case_index: int, *, root_seed: int | None = None,
) -> TestCase | None:
    # Keep root selection reproducible and enumerate forms cyclically.  The
    # first catalog-sized window therefore visits every form once; boundary
    # classes then advance cyclically and the concrete witness is seeded.
    forms = _catalog_root_forms(isa)
    if not forms:
        return None
    if root_seed is None:
        root_seed = case_index
    if type(root_seed) is not int:
        raise ValueError("generation-gap:single-root-seed-invalid")
    rng = random.Random(
        f"rvgen-single-root-v1:{isa}:{root_seed}:{case_index}"
    )
    start = (case_index + root_seed) % len(forms)
    ordered_forms = (*forms[start:], *forms[:start])
    for form in ordered_forms:
        key = (form, isa)
        form_cases = _FORM_CASES.get(key)
        if form_cases is None:
            try:
                form_cases = tuple(generate_form(form))
            except (IndexError, KeyError, OverflowError, RuntimeError,
                    TypeError, UnicodeError, ValueError):
                form_cases = ()
            _FORM_CASES[key] = form_cases
        # ``generate_form`` already returns admitted TestCase objects.  Do not
        # apply a second lane/extension whitelist here: the case carries its
        # own XLEN, extensions, state and observation context.
        cases = tuple(form_cases)
        if cases:
            # Keep every materialized profile in the pool.  RV32E and other
            # narrow profiles may later produce a target or harness gap, but
            # dropping them here silently hides a catalog form and makes the
            # open single-instruction route look smaller than it is.
            # Prefer the configured XLEN when that form has a realization; a
            # rv64 run should not accidentally spend most of its budget on
            # rv32 rows simply because ``generate_form`` lists them first.
            xlen_prefix = str(isa).lower()[:4]
            preferred_cases = tuple(
                item for item in cases
                if str(item.isa_profile).lower().startswith(xlen_prefix)
            )
            if preferred_cases:
                cases = preferred_cases
            # The Direct observer reserves GPRs for its pre-risk frame. A
            # concrete catalog witness that uses one of them cannot reach EMI;
            # skip that witness deterministically and let the binding record
            # any form skipped before a compatible witness is selected.
            cases = tuple(item for item in cases if _candidate_observer_gap(item) is None)
            if not cases:
                continue
            by_boundary: dict[str, list[TestCase]] = {}
            for item in cases:
                contracts = item.dataflow_meta.get("risk_contracts", ())
                boundary = next(
                    (
                        str(contract.get("boundary_class"))
                        for contract in contracts
                        if isinstance(contract, Mapping)
                        and isinstance(contract.get("boundary_class"), str)
                        and contract.get("boundary_class")
                    ),
                    "unclassified",
                )
                by_boundary.setdefault(boundary, []).append(item)
            # RV32E needs a different harness because the shared observer uses
            # x29/x30.  Keep it in the catalog record; the execution layer must
            # report a named harness/target gap rather than silently replacing
            # it with a different profile.
            boundary_classes = tuple(sorted(by_boundary))
            # Rotate the semantic boundary on every root while the form index
            # still walks the complete catalog. Waiting an entire catalog
            # cycle before changing this axis made short timeboxed runs look
            # like a single-boundary sampler.
            boundary = boundary_classes[(case_index + root_seed) % len(boundary_classes)]
            selected = by_boundary[boundary]
            return selected[rng.randrange(len(selected))]
    return None


def testcase_from_recipe(
    recipe: Mapping[str, object], *, case_index: int = 0,
    root_pool: bool = False, root_seed: int | None = None,
) -> TestCase:
    if type(case_index) is not int or case_index < 0:
        raise ValueError("generation-gap:single-recipe-case-index-invalid")
    if not isinstance(recipe, Mapping):
        raise ValueError("generation-gap:single-recipe-requires-rvgen-form")
    generation = recipe.get("generation")
    source = generation.get("source") if isinstance(generation, Mapping) else None
    if not isinstance(generation, Mapping) or not isinstance(source, str) \
            or not source.startswith("framework/rvgen:") \
            or generation.get("direct") != f"generate_form({source.split(':', 1)[1]})":
        raise ValueError("generation-gap:single-recipe-requires-rvgen-form")
    form = source.split(":", 1)[1]
    isa = generation.get("isa")
    sequence = generation.get("sequence")
    if (
        not form or not isinstance(isa, str) or not isa.strip()
        or generation.get("route") != "single"
        or recipe.get("route") not in (None, "", "single")
        or not isinstance(sequence, (list, tuple)) or not sequence
        or any(not isinstance(item, str) or not item.strip() for item in sequence)
        or not isinstance(generation.get("mabi"), str) or not generation["mabi"].strip()
        or not isinstance(generation.get("initial_state"), Mapping) or not generation["initial_state"]
    ):
        raise ValueError("generation-gap:single-recipe-requires-rvgen-form")
    mabi = generation["mabi"].lower()
    if not mabi_matches_isa(isa, mabi):
        raise ValueError("generation-gap:single-recipe-isa-mabi-mismatch")
    expected_identity = {}
    for name in ("testcase_id", "base_program_id", "input_id"):
        values = [
            container[name] for container in (recipe, generation)
            if name in container
        ]
        if values and (
            any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != 1
        ):
            raise ValueError("generation-gap:single-recipe-identity-invalid")
        if values:
            expected_identity[name] = values[0]
    if root_pool and not expected_identity:
        selected = _catalog_root_case(
            isa, case_index, root_seed=root_seed,
        )
        if selected is not None:
            return selected
        raise ValueError("generation-gap:single-catalog-root-space-empty")
    key = (form, isa)
    cases = _FORM_CASES.get(key)
    if cases is None:
        try:
            cases = tuple(generate_form(form))
        except (IndexError, KeyError, OverflowError, RuntimeError,
                TypeError, UnicodeError, ValueError):
            cases = ()
        _FORM_CASES[key] = cases
    cases = tuple(item for item in cases if _lane_accepts_profile(isa, item.isa_profile))
    matches = tuple(item for item in cases if _case_matches_recipe(item, generation))
    matches = tuple(
        item for item in matches
        if all(getattr(item, name) == value for name, value in expected_identity.items())
    )
    if not matches and isinstance(generation.get("initial_state"), Mapping):
        specialized = tuple(
            candidate for item in cases
            if (candidate := _specialize_memory_state(item, generation["initial_state"])) is not None
            and _specialized_boundary_matches(candidate, generation["initial_state"])
            and _case_matches_recipe(candidate, generation)
        )
        matches = tuple(
            item for item in specialized
            if all(getattr(item, name) == value for name, value in expected_identity.items())
        )
    if not matches:
        raise ValueError(f"generation-gap:RVGEN form has no case for {recipe.get('root_key')}")
    if len(matches) > 1:
        semantics = {
            (row.get("lane"), row.get("boundary_class"))
            for item in matches
            for row in item.dataflow_meta.get("risk_contracts", ())
            if row.get("instruction_id") == next(
                risk.instruction_id for risk in item.instruction_meta if "risk" in risk.tags
            )
        }
        if len(semantics) != 1:
            raise ValueError(f"generation-gap:RVGEN form match is ambiguous for {recipe.get('root_key')}")
        matches = tuple(sorted(matches, key=lambda item: item.testcase_id))
    expected = {
        key: value for key, value in generation["initial_state"].items()
        if key in {"expected_result", "expected_single"}
    }
    selected = matches[case_index % len(matches)]
    if not expected:
        return selected
    return replace(
        selected,
        dataflow_meta={**selected.dataflow_meta, "recipe_expected": expected},
    )


def _transport_checked(value: object, *, strict: bool = True) -> object:
    rows = value if isinstance(value, (list, tuple)) else (value,)
    for item in rows:
        outcome = item.get("outcome") if isinstance(item, Mapping) else getattr(item, "outcome", None)
        if outcome in {"runner-contract-gap", "unavailable"}:
            reason = item.get("contract_error") if isinstance(item, Mapping) else getattr(item, "contract_error", None)
            backend = item.get("backend") if isinstance(item, Mapping) else getattr(item, "backend", None)
            extra_state = item.get("extra_state") if isinstance(item, Mapping) else getattr(item, "extra_state", None)
            if (
                backend == "rvgen-semantic"
                and isinstance(extra_state, Mapping)
                and extra_state.get("semantic_machine") == "reference-gap"
            ):
                continue
            if not strict:
                # Target-side contract loss is diagnostic.  The adapter may
                # still have emitted executed PCs, an RVOBS1 frame, or a
                # source-coverage trace; keep that observation so coverage
                # and target-attempt accounting can consume the evidence.
                continue
            if (
                backend == "sail-riscv"
                and isinstance(reason, str)
                and reason.startswith((
                    "sail-config-unknown-extension:",
                    "sail-config-extension-disabled:",
                    "unsupported-sail-",
                ))
            ):
                continue
            raise ExecutionEnvironmentError(
                f"single runner transport gap: {reason or outcome}"
            )
    return value


def run_single_campaign(
    testcase: object,
    *,
    work_dir: str | Path,
    root_binding: Mapping[str, object] | None = None,
    reference_backend: str | None = None,
    target_backend: str | None = None,
    target_id: str | None = None,
    target_binary_path: str | None = None,
    rewrite_hint: object | None = None,
    steps: int = FRAMEWORK_CANONICAL_STEPS,
    seed: int = 1,
    beta: float = 1.0,
    mcmc: bool = True,
    target_jit: bool | None = None,
    max_seconds: float | None = None,
    target_timeout_seconds: float | None = None,
    target_replay_budget_seconds: float | None = None,
    reference_path_reward_version: str = "path-v2",
    coverage_config: Mapping[str, object] | None = None,
    small_model: str | None = None,
    small_model_enabled: bool | None = None,
    emi_enabled: bool = True,
    target_configs: Mapping[str, Mapping[str, object]] | None = None,
    open_capability: bool = True,
    chain_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """从 RVGEN TestCase 启动根改写 → reference-driven RVEMI/MH → Target job 发布/回放。"""

    if not isinstance(testcase, TestCase):
        raise TypeError("single campaign requires an RVGEN TestCase")
    if small_model_enabled is None:
        small_model_enabled = True
    if type(small_model_enabled) is not bool or type(emi_enabled) is not bool:
        raise ValueError("invalid framework module toggles")
    if not emi_enabled and mcmc:
        raise ValueError("EMI-off runs require MCMC to be off")
    target_queue_only_requested = os.environ.get("RQ1_TARGET_QUEUE_ONLY", "0") == "1"
    campaign_deadline = (
        time.monotonic() + float(max_seconds)
        if max_seconds is not None else None
    )

    def remaining_budget() -> float | None:
        if campaign_deadline is None:
            return None
        return max(0.0, campaign_deadline - time.monotonic())

    def model_budget() -> float | None:
        remaining = remaining_budget()
        if remaining is None:
            return None
        try:
            configured = float(os.environ.get("RQ1_SMALL_MODEL_BUDGET", "120"))
        except (TypeError, ValueError):
            configured = 120.0
        return max(0.0, min(remaining * 0.5, configured))
    rewrite_hint = _rewrite_hint(rewrite_hint)
    if root_binding is not None and not isinstance(root_binding, Mapping):
        raise ValueError("root binding is invalid")
    supplied_binding = dict(root_binding or {})
    if "route" in supplied_binding and supplied_binding["route"] != "single":
        raise ValueError("root binding is invalid")
    original = program_from_testcase(testcase)
    if isinstance(coverage_config, Mapping):
        original = replace(
            original,
            run_params={**original.run_params, "framework_coverage": dict(coverage_config)},
        )
    declared_observer_fields = supplied_binding.get("observer_fields")
    if isinstance(declared_observer_fields, (list, tuple)):
        original = replace(
            original,
            run_params={
                **original.run_params,
                "observer_fields": list(declared_observer_fields),
            },
        )
    # The single-instruction route has the same optional root-level small-model
    # step as Program-Full.  The model only selects one candidate from the
    # trusted local rewrite pool; it never invents instruction bytes.  Model
    # transport or availability is not a generation gate: a failed request
    # leaves the valid RVGEN root available for EMI/MCMC and target execution.
    single_candidate_pool = tuple(enumerate_single_rewrites(original))
    small_model_name = str(
        small_model or os.environ.get("RQ1_SMALL_MODEL", "")
    ).strip()
    small_model_request_mode = os.environ.get(
        "RQ1_SMALL_MODEL_REQUEST_MODE", "ollama",
    ).strip().lower()
    small_model_request_name = (
        os.environ.get("RQ1_SMALL_MODEL_OPENAI_MODEL", "").strip()
        if small_model_request_mode == "openai-compatible" else
        small_model_name.removeprefix("ollama/").removeprefix("openai-compatible/")
    )
    small_model_record: dict[str, object] = {
        "enabled": small_model_enabled,
        "status": (
            "disabled" if not small_model_enabled else
            "replayed" if rewrite_hint is not None else
            "pending" if small_model_name else "disabled"
        ),
        "model": small_model_name,
        "request_mode": small_model_request_mode,
        "request_model": small_model_request_name,
        "candidate_count": len(single_candidate_pool),
    }
    if rewrite_hint is None and small_model_enabled and small_model_name:
        try:
            model_response = request_case_rewrite(
                single_candidate_pool, model=small_model_name,
                timeout_seconds=model_budget(),
                cache_key="rvgen-single:" + str(getattr(testcase, "sha256", "")),
                cache_dir=os.environ.get(
                    "RQ1_SMALL_MODEL_CACHE_DIR", "/tmp/rq1-small-model-cache-v1",
                ),
            )
            rewrite_hint = model_response
            small_model_record.update(
                status="response" if model_response is not None else "rejected",
            )
        except (OSError, RuntimeError, ValueError) as error:
            # Keep the failure in the binding for audit, but run the trusted
            # root unchanged so an unavailable model cannot erase coverage.
            rewrite_hint = None
            small_model_record.update(
                status="gap", reason=f"{type(error).__name__}: {str(error)[:500]}",
            )
    reference_backend = reference_backend or (
        "sail-riscv"
        if str(getattr(testcase, "isa_profile", "")).lower().startswith("rv32")
        else "native-rv64"
    )
    if reference_backend == "qemu":
        reference_backend = (
            "qemu-riscv32"
            if str(getattr(testcase, "isa_profile", "")).lower().startswith("rv32")
            else "qemu-riscv64"
        )
    bound_target = supplied_binding.get("target")
    bound_target = bound_target if bound_target == "unavailable" or is_execution_backend(bound_target) else "unavailable"
    authoritative = {
        "route": "single",
        "target": target_backend or bound_target,
        "reference_backend": reference_backend, "harness": "rvgen-single",
    }
    if any(
        name in supplied_binding and supplied_binding[name] != value
        for name, value in authoritative.items()
    ):
        raise ValueError("root binding is invalid")
    state = _testcase_state(testcase)
    binding = {
        "root_key": str(getattr(testcase, "base_program_id")),
        "lineage_key": str(getattr(testcase, "testcase_id")),
        "route": "single", "method": "S1" if mcmc else "S0", "rule": "R3",
        "state": state, "observer": "RVOBS1", "reference_backend": reference_backend,
        "harness": "rvgen-single",
        "target": target_backend or "unavailable", "sequence_mode": "dynamic",
        **supplied_binding,
    }
    if target_backend is not None:
        binding["target_id"] = str(
            target_id or supplied_binding.get("target_id") or target_backend
        )
    # Direct 的在线链只从 reference 取得 MCMC 反馈；Target 仍作为覆盖率和
    # observation 侧链执行。binding 层也会设置这个标记，这里保留显式赋值
    # 作为 Direct API 的防护，避免旧 recipe/兼容调用把 Target 结果误当成搜索状态。
    binding["mcmc_feedback_source"] = "reference"
    binding["small_model"] = dict(small_model_record)
    binding["module_toggles"] = {
        "small_model": small_model_enabled,
        "emi": emi_enabled,
        "mcmc": mcmc,
    }
    binding["method"] = (
        "S5" if rewrite_hint is not None and mcmc else
        "S4" if rewrite_hint is not None else
        "S1" if mcmc else "S0"
    )
    root_form = str(getattr(testcase, "generation_rule_id")).partition(":")[0]
    for name, value in (
        ("form", root_form),
        ("generation_rule_id", str(getattr(testcase, "generation_rule_id"))),
    ):
        if name in binding and binding[name] != value:
            raise ValueError("single RVGEN identity does not match root binding")
        binding.setdefault(name, value)
    testcase_is_trap = expected_trap_from_dataflow(testcase.dataflow_meta)
    if testcase.dataflow_meta.get("candidate_mutated"):
        if testcase_is_trap:
            binding["expected_trap"] = True
        else:
            binding.pop("expected_trap", None)
    elif testcase_is_trap:
        if "expected_trap" in binding and binding["expected_trap"] is not True:
            raise ValueError("single RVGEN trap identity does not match root binding")
        binding["expected_trap"] = True
    for name, value in (
        ("generation_testcase_id", testcase.testcase_id),
        ("generation_base_program_id", testcase.base_program_id),
        ("generation_input_id", testcase.input_id),
    ):
        if name in binding and binding[name] != value:
            raise ValueError("single RVGEN identity does not match root binding")
        binding.setdefault(name, value)
    target_runner = (
        None if target_backend is None or target_configs or target_queue_only_requested else
        lambda artifact, *, timeout_seconds=None: _transport_checked(
            run_single_backend(
                artifact, target_backend, target_binary_path, rvvm_jit=target_jit,
                timeout_seconds=timeout_seconds,
            ),
            strict=False,
        )
    )
    reference_spec = {
        "backend": reference_backend,
        "identity_resolution": "runtime",
        "expected": dict(
            binding.get("reference_expected", {})
            if isinstance(binding.get("reference_expected"), Mapping) else
            {"status": "recorded" if reference_backend == "native-rv64" else "normal"}
        ),
    }
    if isinstance(binding.get("reference_fallback_policy"), Mapping):
        reference_spec["fallback"] = dict(binding["reference_fallback_policy"])
    reference_recipe = {
        "generation": {
            "isa": testcase.isa_profile,
            "mabi": "ilp32" if testcase.isa_profile.lower().startswith("rv32") else "lp64d",
            "sequence": [str(getattr(testcase, "generation_rule_id", ""))],
            "initial_state": {"expected": "trap"} if testcase_is_trap else {},
        },
        "observer": {
            "fields": list(binding.get("observer_fields", ()))
            if isinstance(binding.get("observer_fields"), (list, tuple)) else (),
        },
    }
    reference_backend_runner = (
        backend_runner(
            reference_backend, None, reference_spec, reference_recipe,
            coverage_enabled=False,
        )
        if reference_backend == "native-rv64" else None
    )

    def reference_runner(artifact):
        if reference_backend_runner is not None:
            return _transport_checked(reference_backend_runner(artifact))
        if reference_backend == "sail-riscv":
            current = TestCase.from_dict(dict(artifact.run_params["single_case"]))
            return _transport_checked((sail_reference_observation(
                current, profile_id=current.isa_profile, input_id=current.input_id,
            ),))
        if reference_backend == "rvgen-semantic":
            current = TestCase.from_dict(dict(artifact.run_params["single_case"]))
            return _transport_checked((rvgen_vendor_reference_observation(
                current, artifact.executable_path,
            ),))
        return _transport_checked(run_single_backend(artifact, reference_backend))

    target_replay_runners = {}
    target_bindings = {}
    target_job_configs = {}
    if target_configs:
        testcase_isa = str(getattr(testcase, "isa_profile", "") or "").lower()
        for target_id, target_config in target_configs.items():
            backend = target_config.get("backend")
            binary_path = target_config.get("binary_path")
            target_coverage = target_config.get("coverage_config")
            target_timeout = target_config.get("target_timeout_seconds")
            if not isinstance(target_id, str) or not isinstance(backend, str):
                raise ValueError("target matrix config is invalid")
            if testcase_isa.startswith("rv32") and backend == "qemu-riscv64":
                backend = "qemu-riscv32"
                binary_path = _rv32_qemu_binary(
                    binary_path if isinstance(binary_path, str) else None,
                )
            target_binding = {
                **binding,
                "target": backend,
                "target_id": target_id,
                "target_identity_resolution": "runtime",
                **({"target_backend_override": True}
                   if backend != binding.get("target") else {}),
            }
            runtime_identity_digest, runtime_binary_sha256 = (
                (None, None)
                if target_queue_only_requested else
                _target_runtime_identity(target_config)
            )
            for name in ("target_identity_digest", "target_binary_sha256"):
                target_binding.pop(name, None)
                value = (
                    runtime_identity_digest if name == "target_identity_digest"
                    else runtime_binary_sha256
                )
                if value is None:
                    value = target_config.get(
                        "identity_digest"
                        if name == "target_identity_digest" else "binary_sha256"
                    )
                if isinstance(value, str):
                    target_binding[name] = value
            target_bindings[target_id] = target_binding
            identity_section = target_config.get("identity_section")
            if not isinstance(identity_section, Mapping):
                identity_section = {
                    "backend": backend,
                    "binary_path": binary_path,
                    "expected": {
                        "identity_digest": target_binding.get("target_identity_digest"),
                        "binary_sha256": target_binding.get("target_binary_sha256"),
                    },
                }
            target_recipe = {
                **reference_recipe,
                "target": {
                    **dict(binding),
                    **dict(identity_section),
                    "id": target_id,
                    "backend": backend,
                },
            }
            if not target_queue_only_requested:
                target_job_configs[target_id] = {
                    "backend": backend,
                    "binary_path": binary_path,
                    "identity_section": dict(identity_section),
                    "session_command": target_config.get("session_command"),
                    "coverage_config": (
                        dict(target_coverage) if isinstance(target_coverage, Mapping) else None
                    ),
                    "target_timeout_seconds": target_timeout,
                    "target_binding": dict(target_binding),
                    "target_recipe": target_recipe,
                }

                def run_target(
                    artifact, *, timeout_seconds=None,
                    campaign_deadline_monotonic=None,
                    _backend=backend, _binary_path=binary_path,
                    _coverage=target_coverage, _target_timeout=target_timeout,
                ):
                    budget = _target_timeout
                    if timeout_seconds is not None:
                        budget = float(timeout_seconds) if budget is None else min(
                            float(budget), float(timeout_seconds),
                        )
                    if campaign_deadline_monotonic is not None:
                        remaining = max(
                            0.0, campaign_deadline_monotonic - time.monotonic(),
                        )
                        budget = remaining if budget is None else min(float(budget), remaining)
                    if isinstance(_coverage, Mapping):
                        artifact = replace(
                            artifact,
                            run_params={
                                **dict(artifact.run_params),
                                "framework_coverage": dict(_coverage),
                            },
                        )
                    return _transport_checked(run_single_backend(
                        artifact, _backend,
                        _binary_path if isinstance(_binary_path, str) else None,
                        rvvm_jit=(target_jit if _backend == "rvvm-riscv64" else None),
                        timeout_seconds=budget,
                        campaign_deadline_monotonic=campaign_deadline_monotonic,
                    ), strict=False)

                target_replay_runners[target_id] = run_target

    if target_runner is not None and not target_replay_runners:
        selected_target_id = str(
            binding.get("target_id") or target_backend or "target"
        )
        identity_section = {
            "backend": target_backend,
            "binary_path": target_binary_path,
            "identity_resolution": "runtime",
            "expected": {
                "status": "normal",
                "backend": target_backend,
            },
        }
        target_recipe = {
            **reference_recipe,
            "target": {
                **dict(binding),
                **identity_section,
                "id": selected_target_id,
                "backend": target_backend,
            },
        }
        target_job_configs[selected_target_id] = {
            "backend": target_backend,
            "binary_path": target_binary_path,
            "identity_section": identity_section,
            "session_command": (
                coverage_config.get("session_command")
                if isinstance(coverage_config, Mapping) else None
            ),
            "coverage_config": (
                dict(coverage_config) if isinstance(coverage_config, Mapping) else None
            ),
            "target_timeout_seconds": target_timeout_seconds,
            "target_binding": {
                **binding,
                "target": target_backend,
                "target_id": selected_target_id,
            },
            "target_recipe": target_recipe,
        }

    return run_program_campaign(
        original, build_single_program, reference_runner, target_runner,
        work_dir=work_dir, steps=steps, seed=seed, beta=beta, mcmc=mcmc,
        rewrite_hint=rewrite_hint, root_binding=binding, route=SINGLE_ROUTE,
        max_seconds=remaining_budget(),
        target_timeout_seconds=target_timeout_seconds,
        target_replay_budget_seconds=target_replay_budget_seconds,
        reference_path_reward_version=reference_path_reward_version,
        # Always materialize the capsule projection from the same source.  A
        # Linux-user observation Target still needs this companion when its case
        # is replayed against Unicorn/Renode/RAX/RVVM.
        bare_builder=build_single_bare_program,
        candidate_pool=single_candidate_pool,
        target_replay_runners=target_replay_runners,
        target_bindings=target_bindings,
        target_job_configs=target_job_configs,
        target_ids=(
            tuple(target_configs)
            if target_configs and target_queue_only_requested else
            (str(target_id or target_backend),)
            if target_queue_only_requested and target_backend else None
        ),
        open_capability=open_capability,
        emi_enabled=emi_enabled,
        chain_checkpoint_path=chain_checkpoint_path,
    )


__all__ = ["run_single_campaign", "testcase_from_recipe"]

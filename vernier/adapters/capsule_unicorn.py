import argparse
import json
import os
import struct
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.adapters.contracts import (
    PATH_IDENTITY_WITNESS_CONTRACT,
    backend_spec,
    profile_requires_fp_state,
)
from framework.capsule_csr_facts import ADDRESS_FAULT_MCAUSE as _ADDRESS_FAULT_CAUSES
from framework.execution_environment import load_target_binary_manifest
from framework.adapters.runner import _empty_fp_state, _memory_fault_state, _observation_frame, canonical_trap_state, is_zbb_word, observed_state_fields
from framework.direct_case import OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC
from framework._util import _riscv_elf_identity, bounded_small_int, canonical_digest, elf_load_segments, git_head, git_worktree_clean, is_riscv_elf, path_through_stop, pc_int, pc_path, pc_path_in_ranges, runner_contract_gap, sha256_file


PAGE_SIZE = 4096
_CONFIGURATION_IDENTITY_CONTRACT = "unicorn-riscv64-execution-config-v2"
_SIGNAL_BY_MCAUSE = {
    0: "SIGBUS", 1: "SIGSEGV", 2: "SIGILL", 3: "SIGTRAP",
    4: "SIGBUS", 5: "SIGSEGV", 6: "SIGBUS", 7: "SIGSEGV",
}


def _elf_segment_protections(path: Path) -> tuple[tuple[int, int, int], ...]:
    data = path.read_bytes()
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    return tuple(
        (fields[3], fields[1], fields[6])
        for index in range(phnum)
        if (fields := struct.unpack_from("<IIQQQQQQ", data, phoff + index * phentsize))[0] == 1
        and fields[6]
    )


def _elf_page_protections(
    protections: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    pages: dict[int, int] = {}
    for address, flags, size in protections:
        begin = address & ~(PAGE_SIZE - 1)
        end = (address + size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        for page in range(begin, end, PAGE_SIZE):
            pages[page] = pages.get(page, 0) | flags
    return tuple(sorted(pages.items()))


def _path_identity_witness(
    pcs: tuple[int, ...],
) -> dict[str, object]:
    """Return the real instruction-level path witness, or an explicit gap."""
    if any(pc_int(pc) & 1 for pc in pcs):
        pcs = ()
    sequence, digest = pc_path(pcs)
    if not sequence:
        return {
            "contract": PATH_IDENTITY_WITNESS_CONTRACT,
            "status": "unavailable",
            "executed_pc": {"observed": False, "tested_pc_seen": False, "count": 0},
            "jit_path": {"observed": False, "evidence": "unicorn-no-tcg-trace"},
        }
    return {
        "contract": PATH_IDENTITY_WITNESS_CONTRACT,
        "status": "observed",
        # The shared adapter gate consumes the canonical top-level digest;
        # keep it identical to the executed-PC witness rather than making
        # callers know Unicorn's nested layout.
        "digest": digest,
        "executed_pc": {
            "observed": True,
            "tested_pc_seen": False,
            "count": len(sequence),
            "digest": digest,
            "evidence": "UC_HOOK_CODE",
        },
        "jit_path": {
            "observed": False,
            "evidence": "unicorn-api-no-tcg-trace",
        },
        # Adapter execution gates require the structured PC sequence for all
        # non-legacy adapters.  The digest alone is not replayable evidence.
        "executed_pcs": [hex(pc) for pc in pcs],
    }


def _incomplete_observation_error(
    frame: bytes, *, checkpoint_written: bool,
) -> str | None:
    return (
        "capsule ebreak before finalized RVOBS1 observation"
        if frame.startswith(OBSERVATION_MAGIC)
        and checkpoint_written
        and (len(frame) < 16 or int.from_bytes(frame[8:16], "little") == 0)
        else None
    )


def _configuration_identity(
    profile: str,
    entry: int,
    loads: tuple[tuple[int, bytes, int], ...],
    mailbox: int,
    observation_size: int,
    test_memory: int,
    memory_size: int,
    capsule_sha256: str,
    *,
    unicorn_source_commit: str | None = None,
    unicorn_python_version: str = "unknown",
    unicorn_module_sha256: str | None = None,
    target_identity: dict[str, object] | None = None,
    target_identity_status: str = "missing",
) -> dict[str, object]:
    profile = str(profile or "").strip().lower()
    payload = {
        "arch": "riscv64",
        "mode": "UC_MODE_RISCV64",
        "unicorn_source_commit": unicorn_source_commit,
        "unicorn_python_version": unicorn_python_version,
        "unicorn_module_sha256": unicorn_module_sha256,
        "target_identity": target_identity,
        "target_identity_status": target_identity_status,
        "target_identity_digest": (
            target_identity.get("identity_digest") if target_identity else None
        ),
        "target_binary_sha256": (
            target_identity.get("binary_sha256") if target_identity else None
        ),
        "profile": profile,
        "entry": entry,
        "load_segments": [
            (address, len(data), segment_size)
            for address, data, segment_size in loads
        ],
        "mailbox": mailbox,
        "observation_size": observation_size,
        "test_memory": test_memory,
        "memory_size": memory_size,
        "capsule_sha256": capsule_sha256,
    }
    identity_payload = {"contract": _CONFIGURATION_IDENTITY_CONTRACT, **payload}
    return {**identity_payload, "identity_digest": canonical_digest(identity_payload)}


def _target_identity() -> tuple[object | None, str, str | None]:
    manifest_path = os.environ.get("RV_UNICORN_TARGET_IDENTITY_PATH")
    if not manifest_path:
        return None, "missing", None
    try:
        installed = load_target_binary_manifest(
            Path(manifest_path),
            expected_backend="unicorn-riscv64",
            require_source_verified=True,
        )
    except Exception as exc:  # noqa: BLE001 - identity is a named runner gap
        return None, "invalid", f"unicorn-target-identity-invalid: {exc}"
    library = Path(os.environ.get("LIBUNICORN_PATH", "")) / "libunicorn.so.2"
    if not library.is_file() or sha256_file(library) != installed.identity.binary_sha256:
        return None, "mismatch", "unicorn-loaded-library-does-not-match-target-identity"
    return installed.identity, "verified", None


def _read_mstatus_fs(uc, riscv_const) -> tuple[int | None, str, str | None]:
    """Read mstatus.FS (bits 13-14) through Unicorn's CSR register channel.

    Unicorn maps RISC-V CSRs onto named registers (UC_RISCV_REG_MSTATUS);
    reg_read routes the CSR number through riscv_csrrw, so the value is the
    real CSR state.  A read failure is a named gap (fs-observation-
    unavailable); placeholder zeros are never fabricated into the compare.
    """
    try:
        mstatus = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_MSTATUS))
    except Exception as exc:  # noqa: BLE001 - Unicorn may raise UcError here
        return (
            None,
            "gap",
            f"fs-observation-unavailable: {type(exc).__name__}: {exc}",
        )
    return (mstatus >> 13) & 0x3, "observed", None


def _final_fp_state(uc, riscv_const, *, require_fp_state: bool) -> dict[str, object]:
    if not require_fp_state:
        return _empty_fp_state()
    try:
        mstatus_fs, mstatus_fs_observer, mstatus_fs_gap = _read_mstatus_fs(
            uc, riscv_const
        )
        # Unicorn exposes stale/undefined CSR values while FS=Off.  Do not
        # interpret those values as architectural FP state; report the
        # explicit observer gap and leave the state fields absent.
        if mstatus_fs in (None, 0):
            return {
                "fpr_rawbits": None,
                "fflags": None,
                "frm": None,
                "fp_observer": "not-observed",
                "fp_observer_gap": (
                    "unicorn-fp-observer-disabled: mstatus.FS=0"
                    if mstatus_fs == 0 else mstatus_fs_gap
                ),
                "mstatus_fs": mstatus_fs,
                "mstatus_fs_observer": mstatus_fs_observer,
                "mstatus_fs_gap": mstatus_fs_gap,
            }
        return {
            "fpr_rawbits": [
                pc_int(uc.reg_read(getattr(riscv_const, f"UC_RISCV_REG_F{index}")))
                for index in range(32)
            ],
            "fflags": bounded_small_int(
                uc.reg_read(riscv_const.UC_RISCV_REG_FFLAGS), 0x1F, "fflags"
            ),
            "frm": bounded_small_int(
                uc.reg_read(riscv_const.UC_RISCV_REG_FRM), 0x7, "frm"
            ),
            # Observed-state marker: real FPR/FFLAGS/FRM evidence is only
            # admitted to the strict compare when the observer claims it.
            "fp_observer": "observed",
            # mstatus.FS is read separately: an FP profile must expose the
            # real FS bits or an explicit gap, never a fabricated zero.
            "mstatus_fs": mstatus_fs,
            "mstatus_fs_observer": mstatus_fs_observer,
            "mstatus_fs_gap": mstatus_fs_gap,
        }
    except Exception as exc:  # noqa: BLE001 - FP observer is optional per route
        return {
            "fpr_rawbits": None,
            "fflags": None,
            "frm": None,
            "fp_observer": "not-observed",
            "fp_observer_gap": f"unicorn-fp-observer-unavailable: {exc}",
            "adapter_execution_error": f"final-fp-state-unavailable: {exc}",
            "mstatus_fs": mstatus_fs,
            "mstatus_fs_observer": mstatus_fs_observer,
            "mstatus_fs_gap": mstatus_fs_gap,
        }


def _guest_trap_state(uc, riscv_const) -> dict[str, object]:
    """Read the guest exception bridge fields (cause/epc/tval) from the CPU.

    Unicorn maps RISC-V privileged CSRs onto named registers.  Reading them
    separates a real guest exception delivery from a host-side abort; any
    read failure is returned as an explicit gap instead of guessed zeros.

    Unicorn never populates these CSRs on guest exceptions: its exception
    dispatch (qemu/accel/tcg/cpu-exec.c cpu_handle_exception) runs before
    riscv_cpu_do_interrupt and stops the CPU, so mcause/mepc/mtval are never
    written.  Delivered traps are instead observed through UC_HOOK_INTR
    (see _run_capsule); this helper only reports host-side aborts, where the
    all-zero CSR reads stay a named gap.
    """
    try:
        cause = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_MCAUSE))
        epc = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_MEPC))
        tval = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_MTVAL))
    except Exception as exc:  # noqa: BLE001 - backend register reads may raise UcError
        return {
            "guest_trap": "not-observed",
            "guest_trap_error": (
                "guest-trap-csr-observer-gap (mcause/mepc/mtval): "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    if cause == 0 and epc == 0 and tval == 0:
        return {"guest_trap": "not-observed", "guest_trap_error": "csr-unpopulated"}
    return {
        "guest_trap": "delivered",
        "guest_cause": cause,
        "guest_epc": epc,
        "guest_tval": tval,
        "trap_observer": "unicorn-csr",
        "guest_tval_observer": "unicorn-csr",
    }


def _run_capsule(
    elf: Path,
    mailbox: int,
    observation_size: int,
    *,
    test_memory: int | None = None,
    memory_size: int | None = None,
) -> tuple[bytes, tuple[int, ...], dict[str, object], dict[str, object], dict[str, object]]:
    if (test_memory is None) != (memory_size is None):
        raise ValueError("test_memory and memory_size must be provided together")
    if not elf.is_file():
        raise RuntimeError(f"missing capsule ELF: {elf}")
    if observation_size < OBSERVATION_HEADER_SIZE:
        raise ValueError("Unicorn RVOBS1 frame is too small")
    if not is_riscv_elf(elf) and not (
        observation_size < OBSERVATION_HEADER_SIZE and _riscv_elf_identity(elf)
    ):
        raise ValueError("expected RISC-V ELF")
    import unicorn
    from unicorn import (
        UC_ARCH_RISCV,
        UC_HOOK_CODE,
        UC_HOOK_INTR,
        UC_HOOK_MEM_INVALID,
        UC_HOOK_MEM_WRITE,
        UC_MODE_RISCV64,
        UC_PROT_ALL,
        UC_PROT_EXEC,
        UC_PROT_NONE,
        UC_PROT_READ,
        UC_PROT_WRITE,
        Uc,
    )
    from unicorn import riscv_const

    entry, loads = elf_load_segments(elf)
    protections = _elf_segment_protections(elf)
    executable_loads = tuple(
        (address, size) for address, flags, size in protections if flags & 1
    )
    uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)
    mapped: list[tuple[int, int]] = []
    for address, _data, size in sorted(loads):
        begin = address & ~(PAGE_SIZE - 1)
        end = (address + size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        if mapped and begin <= mapped[-1][1]:
            mapped[-1] = (mapped[-1][0], max(mapped[-1][1], end))
        else:
            mapped.append((begin, end))
    for begin, end in mapped:
        uc.mem_map(begin, end - begin, UC_PROT_ALL)
    for address, data, segment_memory_size in loads:
        uc.mem_write(address, data)
        if segment_memory_size < len(data):
            raise ValueError("ELF segment memory size is smaller than file size")
    for page, flags in _elf_page_protections(protections):
        permissions = (UC_PROT_READ if flags & 4 else UC_PROT_NONE) | (UC_PROT_WRITE if flags & 2 else 0) | (UC_PROT_EXEC if flags & 1 else 0)
        uc.mem_protect(page, PAGE_SIZE, permissions)
    pcs: list[int] = []
    decoded_last: dict[str, object] | None = None
    decoded_instruction_count = 0
    trace_records_total = 0
    trace_truncated = False
    halted = False
    profile = str(
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
    ).strip().lower()
    if not profile:
        raise RuntimeError("RV_TESTCASE_ISA_PROFILE is required for Unicorn state interpretation")
    # Do not preflight extension support here.  Unicorn must receive the
    # generated case and return its own illegal-instruction/unsupported-ISA
    # outcome; the campaign records that outcome and moves on.
    require_fp_state = (
        profile_requires_fp_state(profile)
        and os.environ.get("RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "0") == "1"
    )
    if require_fp_state:
        for index in range(32):
            uc.reg_write(getattr(riscv_const, f"UC_RISCV_REG_F{index}"), 0)
        uc.reg_write(riscv_const.UC_RISCV_REG_FFLAGS, 0)
        uc.reg_write(riscv_const.UC_RISCV_REG_FRM, 0)
    capsule_sha256 = sha256_file(elf)
    unicorn_python_version = str(getattr(unicorn, "__version__", "unknown"))
    unicorn_module_path = getattr(unicorn, "__file__", None)
    unicorn_source_commit = os.environ.get("RV_UNICORN_SOURCE_COMMIT")
    unicorn_source_clean = bool(unicorn_source_commit)
    if unicorn_module_path and not unicorn_source_commit:
        unicorn_source_root = Path(unicorn_module_path).resolve().parent
        unicorn_source_commit = git_head(unicorn_source_root)
        unicorn_source_clean = (
            unicorn_source_commit is None or git_worktree_clean(unicorn_source_root)
        )
        if not unicorn_source_clean:
            unicorn_source_commit = None
    unicorn_module_sha256 = (
        sha256_file(Path(unicorn_module_path))
        if unicorn_module_path
        else None
    )
    target_identity, target_identity_status, target_identity_error = _target_identity()
    if (
        target_identity is not None
        and target_identity.source_commit != unicorn_source_commit
    ):
        target_identity = None
        target_identity_status = "mismatch"
        target_identity_error = "unicorn-source-commit-does-not-match-target-identity"
    target_identity_dict = (
        target_identity.to_dict() if target_identity is not None else None
    )
    configuration_identity = _configuration_identity(
        profile,
        entry,
        loads,
        mailbox,
        observation_size,
        test_memory,
        memory_size,
        capsule_sha256,
        unicorn_source_commit=unicorn_source_commit,
        unicorn_python_version=unicorn_python_version,
        unicorn_module_sha256=unicorn_module_sha256,
        target_identity=target_identity_dict,
        target_identity_status=target_identity_status,
    )
    if not unicorn_source_clean:
        configuration_identity["identity_digest"] = None

    mailbox_error: str | None = None
    checkpoint_written = False

    def finalized_mailbox(machine) -> bytes | None:
        try:
            frame = bytes(machine.mem_read(mailbox, observation_size))
        except unicorn.UcError as exc:
            nonlocal mailbox_error
            mailbox_error = f"mailbox read failed: {type(exc).__name__}: {exc}"
            return None
        checkpoint = int.from_bytes(frame[8:16], "little") if len(frame) >= 16 else 0
        if checkpoint == 0 or checkpoint & 1 or _observation_frame(frame) != frame:
            mailbox_error = _incomplete_observation_error(
                frame, checkpoint_written=checkpoint_written,
            ) or "capsule did not expose a complete RVOBS1 frame"
            return None
        return frame

    def code_hook(machine, address: int, size: int, _user) -> None:
        nonlocal halted, mailbox_error, trace_records_total, decoded_last, decoded_instruction_count
        trace_records_total += 1
        pcs.append(pc_int(address))
        size = size or 4
        raw = machine.mem_read(address, size)
        word = int.from_bytes(raw, "little")
        previous_decoded = decoded_last
        decoded_instruction_count += 1
        decoded_last = {"pc": pc_int(address), "word": f"{word:08x}", "size": size}
        # The capsule's exit sentinel is an ebreak placed after the exit
        # checkpoint.  An ebreak in the program is a guest trap, not the
        # harness sentinel, so it must not be swallowed into normal completion.
        is_ebreak = (
            (size == 4 and word == 0x00100073)
            or (size == 2 and word == 0x9002)
        )
        is_completed_ecall = size == 4 and word == 0x00000073 and checkpoint_written
        if not is_ebreak and not is_completed_ecall:
            return
        frame = finalized_mailbox(machine)
        if frame is not None:
            pcs.pop()
            decoded_instruction_count -= 1
            # The terminal ebreak is an adapter sentinel, not part of the
            # guest path reported to the common observation contract.
            trace_records_total -= 1
            decoded_last = previous_decoded
            halted = True
            machine.emu_stop()

    uc.hook_add(UC_HOOK_CODE, code_hook)

    def checkpoint_write_hook(_machine, _access, address, size, _value, _user) -> None:
        nonlocal checkpoint_written
        checkpoint_written |= address <= mailbox + 8 < address + size

    uc.hook_add(UC_HOOK_MEM_WRITE, checkpoint_write_hook)

    trap_recorded: dict[str, object] | None = None

    def trap_hook(machine, intno, _user) -> None:
        """UC_HOOK_INTR: 同步 guest trap 事件通道。

        Unicorn 的 cpu_handle_exception 在 riscv_cpu_do_interrupt 之前把
        CPU 异常分发给 UC_HOOK_INTR（qemu/accel/tcg/cpu-exec.c），intno
        即 QEMU 异常编号，与 mcause 编号一致（target/riscv/cpu_bits.h）；
        已在容器内实测（unicorn 2.1.3）：illegal 携带 intno=2、ecall 携带
        intno=8，均与 mcause 编号一致；trap 指令地址取最后一条已解码指令
        （code hook 在指令执行前记录）。地址类异常（misaligned/page-fault
        等）的 tval 按规范从 MBADADDR 读取；其余 cause 的 mtval 语义可选，
        当前 hook 无法观察时保留具名 gap；
        但实测本构建中 misaligned 访问被静默执行、未映射访问先抛 UcError，
        该 MBADADDR 分支尚未被触发，保留为规范正确的路径。ebreak 不触发
        本通道，而是以 UcError（UC_ERR_INSN_INVALID）中止，由调用方的
        UcError 分支按已解码的 0x00100073 推导 cause=3。

        注册 hook 后异常不再自动报错，因此无论是否记录都必须停止，否则
        Unicorn 会跳过 trap 指令继续执行，所有运行时 trap 都交给观察层。
        """
        nonlocal trap_recorded, halted
        if intno == 8 and checkpoint_written:  # Post-checkpoint ecall ends the capsule.
            if finalized_mailbox(machine) is not None:
                halted = True
                machine.emu_stop()
            return
        try:
            trap_pc = decoded_last["pc"] if decoded_last is not None else pc_int(
                machine.reg_read(riscv_const.UC_RISCV_REG_PC)
            )
        except Exception as exc:  # noqa: BLE001 - preserve cause with a named PC gap
            trap_recorded = {"error": f"trap-pc-unavailable: {type(exc).__name__}: {exc}"}
            machine.emu_stop()
            return
        zbb_word = (decoded_last.get("word") if decoded_last is not None else None)
        zbb_unsupported = (
            intno == 2 and zbb_word is not None and is_zbb_word(int(zbb_word, 16)))
        zbb_state = (
            {"zbb_unsupported": zbb_unsupported} if zbb_word is not None else {}
        )
        try:
            cause = pc_int(intno)
            if cause in _ADDRESS_FAULT_CAUSES:
                raw_tval = pc_int(
                    machine.reg_read(riscv_const.UC_RISCV_REG_MBADADDR)
                )
                tval, tval_observer = raw_tval, "unicorn-intr-csr"
            else:
                tval = None
                tval_observer = "csr-unpopulated"
        except Exception as exc:  # noqa: BLE001 - keep cause/EPC when mtval is unavailable
            trap_recorded = {
                "cause": intno,
                "epc": trap_pc,
                "tval": None,
                **zbb_state,
                "trap_observer": "unicorn-intr-hook",
                "tval_observer": "unavailable",
                "tval_error": f"{type(exc).__name__}: {exc}",
            }
        else:
            trap_recorded = {"cause": cause, "epc": trap_pc, "tval": tval,
                             **zbb_state,
                             "trap_observer": "unicorn-intr-hook",
                             "tval_observer": tval_observer}
        machine.emu_stop()

    uc.hook_add(UC_HOOK_INTR, trap_hook)
    memory_fault: dict[str, object] = {}

    def invalid_memory_hook(machine, access: int, address: int, size: int, value: int, _user) -> bool:
        if not memory_fault:
            fault_pc = decoded_last["pc"] if decoded_last is not None else pc_int(
                machine.reg_read(riscv_const.UC_RISCV_REG_PC)
            )
            if access in {
                getattr(unicorn, "UC_MEM_FETCH_UNMAPPED", -1),
                getattr(unicorn, "UC_MEM_FETCH_PROT", -1),
            }:
                fault_pc = pc_int(machine.reg_read(riscv_const.UC_RISCV_REG_PC))
            memory_fault.update(
                address=pc_int(address),
                access=int(access),
                size=int(size),
                value=int(value),
                pc=fault_pc,
            )
        return False

    uc.hook_add(UC_HOOK_MEM_INVALID, invalid_memory_hook)

    details: dict[str, object] = {
        "source": "unicorn-api",
        "unicorn_source_commit": unicorn_source_commit,
        "unicorn_source_clean": unicorn_source_clean,
        "unicorn_python_version": unicorn_python_version,
        "unicorn_module_sha256": unicorn_module_sha256,
        "target_identity_status": target_identity_status,
        "target_identity": target_identity_dict,
        "target_identity_digest": (
            target_identity.identity_digest if target_identity is not None else None
        ),
        "target_binary_sha256": (
            target_identity.binary_sha256 if target_identity is not None else None
        ),
        "target_source_commit": (
            target_identity.source_commit if target_identity is not None else None
        ),
        "target_identity_path": os.environ.get("RV_UNICORN_TARGET_IDENTITY_PATH"),
        "dependency_family": backend_spec("unicorn-riscv64").dependency_family,
        "capsule_sha256": capsule_sha256,
        "mailbox_address": hex(mailbox),
        "configuration_identity": configuration_identity,
    }
    if target_identity_error is not None:
        details["target_identity_error"] = target_identity_error

    def exception_observation(
        fault_pc: int,
        guest_trap: dict[str, object],
        execution_status: str,
        execution_error: str | None,
    ) -> tuple[bytes, dict[str, object], dict[str, object], dict[str, object]]:
        state = _final_fp_state(uc, riscv_const, require_fp_state=require_fp_state)
        state.update(canonical_trap_state(guest_trap))
        if memory_fault:
            state.update(_memory_fault_state(memory_fault, "unicorn"))
        state["observer_fields"] = observed_state_fields(state, guest_trap)
        if execution_error is not None:
            state["adapter_execution_error"] = execution_error
        path_pcs = tuple(pcs) if (
            guest_trap.get("guest_trap") == "delivered" or memory_fault
        ) else ()
        if path_pcs and not pc_path_in_ranges(path_pcs, executable_loads):
            path_pcs = ()
            execution_error = execution_error or (
                "unicorn execution trace leaves executable ELF"
            )
        details.update(
            execution_status=execution_status,
            decoded_instruction_count=decoded_instruction_count,
            trace_records_total=trace_records_total,
            trace_truncated=trace_truncated,
            trace_available=bool(path_pcs),
            trace_mechanism="UC_HOOK_CODE",
            path_identity=_path_identity_witness(path_pcs),
            **guest_trap,
        )
        if execution_error is not None:
            details["execution_error"] = execution_error
        if memory_fault:
            details["memory_fault"] = memory_fault
        observation_state = {
            # A UcError is not by itself a guest exception.  Unicorn also
            # raises it for host-side unmapped accesses and other adapter
            # failures; only a delivered trap has architectural exception
            # evidence.  Keep the latter distinguishable so the shared
            # parser cannot promote a host abort into a guest observation.
            "signal": _SIGNAL_BY_MCAUSE.get(guest_trap.get("guest_cause"))
            if guest_trap.get("guest_trap") == "delivered" else None,
            "fault_pc": fault_pc,
            **guest_trap,
            "extra_state": state,
        }
        if memory_fault:
            observation_state.update(
                fault_address=memory_fault["address"],
                memory_access=memory_fault["access"],
                memory_size=memory_fault["size"],
                memory_fault_observer="unicorn-memory-hook",
            )
        elif (
            guest_trap.get("guest_trap") == "delivered"
            and guest_trap.get("guest_cause") in _ADDRESS_FAULT_CAUSES
            and type(guest_trap.get("guest_tval")) is int
            and guest_trap.get("guest_tval_observer") == "unicorn-intr-csr"
        ):
            observation_state["fault_address"] = guest_trap["guest_tval"]
            observation_state["fault_address_observer"] = "unicorn-intr-csr"
        # A host-side exception snapshot is not a guest-written RVOBS1
        # mailbox.  Keep trap/diagnostic state on the structured stderr
        # channel; returning a synthetic frame would make the common parser
        # report a complete architectural observation.
        return b"", details, state, observation_state

    try:
        uc.emu_start(entry, 0, count=0)
    except unicorn.UcError as exc:
        # 实测（unicorn 2.1.3）：ebreak 不触发 UC_HOOK_INTR 也不填充 CSR，
        # 而是以 UcError（UC_ERR_INSN_INVALID）中止；cause(=3) 与 epc 由
        # 跟踪到的 32 位 ebreak 指令确定。其他 UcError（如未映射读写）是
        # 宿主侧错误，维持 CSR 具名 gap。
        guest_trap = _guest_trap_state(uc, riscv_const)
        last = decoded_last
        last_word = int(last["word"], 16) if last is not None else None
        last_size = int(last.get("size", 4)) if last is not None else 0
        invalid_instruction = getattr(exc, "errno", None) == getattr(
            unicorn, "UC_ERR_INSN_INVALID", object()
        ) or "UC_ERR_INSN_INVALID" in str(exc)
        decoded_ebreak = last_word is not None and (last_size, last_word) in {
            (4, 0x00100073),
            (2, 0x9002),
        }
        faulting_pc = last["pc"] if last is not None else None
        faulting_word = last_word
        faulting_size = last_size
        if invalid_instruction and not decoded_ebreak:
            try:
                faulting_pc = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_PC))
                raw = bytes(uc.mem_read(faulting_pc, 4))
                faulting_word = int.from_bytes(raw, "little")
                faulting_size = 2 if len(raw) >= 2 and int.from_bytes(raw[:2], "little") & 0x3 != 0x3 else 4
            except Exception:
                pass
        zbb_unsupported = (
            (guest_trap.get("guest_trap") == "delivered"
             and guest_trap.get("guest_cause") == 2
             or invalid_instruction)
            and faulting_pc is not None
            and faulting_size == 4
            and faulting_word is not None
            and is_zbb_word(faulting_word)
        )
        if mailbox_error is not None:
            guest_trap = {
                "guest_trap": "not-observed",
                "guest_trap_error": mailbox_error,
            }
        elif decoded_ebreak:
            guest_trap = {
                "guest_trap": "delivered",
                "guest_cause": 3,
                "guest_epc": faulting_pc,
                # Canonical ebreak has architecturally defined mtval=0.
                "guest_tval": 0,
                "trap_observer": "unicorn-decoded-ebreak",
                "guest_tval_observer": "unicorn-decoded-ebreak",
            }
        elif zbb_unsupported and faulting_pc is not None:
            guest_trap = {
                "guest_trap": "delivered",
                "guest_cause": 2,
                "guest_epc": faulting_pc,
                "guest_tval": None,
                "guest_zbb_unsupported": True,
                "trap_observer": "unicorn-decoded-zbb",
                "guest_tval_observer": "csr-unpopulated",
            }
        elif invalid_instruction and faulting_pc is not None:
            guest_trap = {
                "guest_trap": "delivered",
                "guest_cause": 2,
                "guest_epc": faulting_pc,
                # For an illegal instruction, mtval is the offending
                # instruction encoding.  The decoder already captured that
                # word before Unicorn raised UcError, so this is an observed
                # instruction witness rather than a fabricated CSR value.
                "guest_tval": faulting_word,
                "guest_zbb_unsupported": False,
                "trap_observer": "unicorn-decoded-illegal",
                "guest_tval_observer": "unicorn-decoded-illegal",
            }
        elif last_word is not None:
            guest_trap = dict(guest_trap)
            guest_trap["guest_zbb_unsupported"] = False
        fault_pc = (
            guest_trap.get("guest_epc")
            if guest_trap.get("guest_trap") == "delivered"
            else memory_fault.get("pc")
            if memory_fault
            else decoded_last["pc"]
            if decoded_last is not None
            else pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_PC))
        )
        stdout, details, state, observation_state = exception_observation(
            fault_pc, guest_trap, "uc-exception", mailbox_error or str(exc)
        )
        return (
            stdout,
            tuple(pcs) if details["path_identity"].get("status") == "observed" else (),
            details,
            state,
            observation_state,
        )
    if trap_recorded is not None:
        # 非 ebreak 的同步 guest trap（illegal/misaligned/page-fault 等）
        # 经 UC_HOOK_INTR 事件通道携带真实 cause/epc/tval 可观测。
        if "error" in trap_recorded:
            guest_trap = {
                "guest_trap": "not-observed",
                "guest_trap_error": trap_recorded["error"],
            }
            fault_pc = pc_int(uc.reg_read(riscv_const.UC_RISCV_REG_PC))
        else:
            epc = int(trap_recorded["epc"])
            trap_tval = trap_recorded["tval"]
            trap_tval_observer = trap_recorded.get("tval_observer")
            if (
                trap_recorded.get("cause") == 2
                and trap_tval is None
                and decoded_last is not None
            ):
                # Unicorn delivers illegal-instruction traps through the
                # interrupt hook without populating mtval.  The code hook has
                # already captured the offending instruction, which is the
                # architectural mtval witness for this trap.
                trap_tval = int(str(decoded_last["word"]), 16)
                trap_tval_observer = "unicorn-intr-decoded-illegal"
            guest_trap = {
                "guest_trap": "delivered",
                "guest_cause": trap_recorded["cause"],
                "guest_epc": epc,
                "guest_tval": trap_tval,
                "trap_observer": trap_recorded.get("trap_observer", "unicorn-intr-hook"),
                "guest_tval_observer": trap_tval_observer,
                **(
                    {"guest_zbb_unsupported": trap_recorded["zbb_unsupported"]}
                    if "zbb_unsupported" in trap_recorded
                    else {}
                ),
            }
            fault_pc = epc
        stdout, details, state, observation_state = exception_observation(
            fault_pc, guest_trap, "uc-guest-trap", trap_recorded.get("error")
        )
        return (
            stdout,
            tuple(pcs) if details["path_identity"].get("status") == "observed" else (),
            details,
            state,
            observation_state,
        )
    if not halted:
        raise RuntimeError("capsule did not reach ebreak within instruction/time budget")
    stdout = bytes(uc.mem_read(mailbox, observation_size))
    checkpoint_pc = int.from_bytes(stdout[8:16], "little") if len(stdout) >= 16 else 0
    if checkpoint_pc and checkpoint_pc in pcs:
        pcs = path_through_stop(pcs, checkpoint_pc)
        if not pc_path_in_ranges(pcs, executable_loads):
            details["trace_gap"] = "unicorn execution trace leaves executable ELF"
            pcs = []
    else:
        details["trace_gap"] = "unicorn execution trace misses RVOBS1 checkpoint"
        pcs = []
    # ``pcs`` is the path retained up to the guest checkpoint; any terminal
    # sentinel and post-checkpoint callbacks are deliberately excluded.  The
    # two runner counters describe that same retained path.
    trace_records_total = len(pcs)
    state = _final_fp_state(uc, riscv_const, require_fp_state=require_fp_state)
    if details.get("trace_gap") is not None:
        state["adapter_execution_error"] = details["trace_gap"]
    state["observer_fields"] = observed_state_fields(state)
    details.update(
        execution_status="completed-ebreak",
        decoded_instruction_count=decoded_instruction_count,
        trace_records_total=trace_records_total,
        trace_truncated=trace_truncated,
        trace_available=bool(pcs),
        trace_mechanism="UC_HOOK_CODE",
        path_identity=_path_identity_witness(tuple(pcs)),
    )
    observation_state = {"extra_state": state}
    return stdout, tuple(pcs), details, state, observation_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a framework RV64 mailbox capsule through Unicorn")
    parser.add_argument("elf")
    parser.add_argument("--mailbox", default=os.environ.get("RV_CAPSULE_MAILBOX"))
    parser.add_argument("--observation-size", default=os.environ.get("RV_CAPSULE_OBSERVATION_SIZE"), type=int)
    parser.add_argument("--test-memory", default=os.environ.get("RV_CAPSULE_TEST_MEMORY"))
    parser.add_argument("--memory-size", default=os.environ.get("RV_CAPSULE_MEMORY_SIZE"), type=int)
    args = parser.parse_args(argv)
    if args.mailbox is None or args.observation_size is None:
        return runner_contract_gap("capsule mailbox and observation size are required")
    try:
        stdout, pcs, details, state, observation_state = _run_capsule(
            Path(args.elf).resolve(),
            int(args.mailbox, 0),
            args.observation_size,
            test_memory=(int(args.test_memory, 0) if args.test_memory is not None else None),
            memory_size=args.memory_size,
        )
    except Exception as exc:
        return runner_contract_gap(f"unicorn runner failed: {type(exc).__name__}: {exc}")
    sys.stdout.buffer.write(stdout)
    sys.stderr.write(f"RV_TOOL_VERSION=unicorn {details['unicorn_python_version']}\n")
    sys.stderr.write("RV_OBSERVATION_STATE=" + json.dumps(observation_state, separators=(",", ":")) + "\n")
    sys.stderr.write(f"RV_EXECUTED_PCS={json.dumps([hex(pc) for pc in pcs], separators=(',', ':'))}\n")
    sys.stderr.write(f"RV_INSTRUCTION_COUNT={len(pcs)}\n")
    # The common parser compares these two markers as one path-count
    # contract.  The finalized mailbox path is already represented by `pcs`;
    # do not let an adapter-internal sentinel/decode count turn a complete
    # observation into a transport gap.
    sys.stderr.write(f"RV_TOTAL_GUEST_INSTRUCTION_COUNT={len(pcs)}\n")
    sys.stderr.write("RV_TRANSLATION_EVIDENCE=" + json.dumps({"backend": "unicorn-riscv64", "expected_path": "either", "tested_pc_seen": False, "tested_pc_translated": False, "tested_pc_executed": False, "execution_count": len(pcs), "details": details}, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

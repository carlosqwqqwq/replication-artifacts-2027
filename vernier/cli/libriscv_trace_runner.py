import argparse
import json
import math
import os
import re
import signal as signal_module
import subprocess
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.execution_environment import (
    EM_RISCV,
    ExecutionEnvironmentError,
    load_target_binary_manifest,
    require_execution_plane,
    resolve_preferred_libriscv_translated_manifest,
    resolve_x86_64_runner,
    validate_elf_machine,
)
from framework.adapters.runner import observed_state_fields
from framework.direct_case import (
    _SIGNAL_CODE_BY_NAME as _SIGNAL_NUMBER_BY_NAME,
)
from framework.cli._gdb_remote import (
    LIBRISCV_TRANSLATED_FP_LAYOUT,
    _Flock,
    _connect_client,
    collect_final_fp_state,
    exit_checkpoint_pc,
)
from framework.adapters.trace_common import (
    _binary_tool_version,
    _clip_trace_at_checkpoint,
    _execute,
    _observation_frame_prefix,
    _path_identity_alias,
    _target_binary_details,
    _target_configuration as _shared_target_configuration,
)
from framework.paths import EXTERNAL_ROOT
from framework._util import (
    canonical_digest,
    elf_executable_loads,
    pc_digest,
    pc_int as _pc_int,
    pc_path_in_ranges,
    runner_contract_gap as _fail,
    sha256_file,
)
from framework.spec_definedness import profile_requires_fp_state


TRACE_PC_RE = re.compile(
    # RVOBS1 may split the ``instr`` token while it is written to stdout.
    r"^f\s+.*?\bpc\s+0x([0-9a-fA-F]+)\s+ins",
    re.IGNORECASE | re.MULTILINE,
)
INSTRUCTION_COUNT_RE = re.compile(r"Instructions executed:\s+([0-9]+)")
MACHINE_EXCEPTION_RE = re.compile(
    r"\[(?:0x)?(?P<pc>[0-9a-fA-F]+)\](?P<instruction>[^\n]*)\n>>> Machine exception\s+(?P<code>[0-9]+):\s*(?P<message>[^\n]*?)"
    r"(?:\s*\(data:\s*0x(?P<data>[0-9a-fA-F]+)\))?(?:\r?\n|$)",
    re.IGNORECASE,
)
# libriscv machine exception code -> linux-user guest signal.  The emulator
# prints its own exception enum plus an optional ``(data: 0x...)`` value; the
# framework projects that onto the guest signal and architectural trap state.
# Numbering is libriscv's own (rvlinux, see lib/libriscv/types.hpp), not the
# RISC-V mcause numbering.  Only the entries below are guest-visible execution
# exceptions; loader/runtime failures stay as raw machine exceptions.
_SIGNAL_BY_MACHINE_EXCEPTION_CODE = {
    0: "SIGILL",   # ILLEGAL_OPCODE
    1: "SIGILL",   # ILLEGAL_OPERATION
    2: "SIGSEGV",  # PROTECTION_FAULT
    3: "SIGSEGV",
    4: "SIGBUS",
    5: "SIGILL",  # UNIMPLEMENTED_INSTRUCTION_LENGTH
    6: "SIGILL",
    9: "SIGBUS",
}
_GUEST_CAUSE_BY_MACHINE_EXCEPTION_CODE = {
    0: 2,  # ILLEGAL_OPCODE -> illegal instruction
    1: 2,  # ILLEGAL_OPERATION -> illegal instruction
    3: 1,  # EXECUTION_SPACE_PROTECTION_FAULT -> instruction access fault
    4: 0,  # MISALIGNED_INSTRUCTION -> instruction-address-misaligned
    5: 2,  # UNIMPLEMENTED_INSTRUCTION_LENGTH -> illegal instruction
    6: 2,  # UNIMPLEMENTED_INSTRUCTION -> illegal instruction
}
# machine exception cause -> linux-user si_code 语义投影。libriscv 不打印
# si_code，且 data 对保护/对齐异常未统一携带完整地址，因此只投影确定的原因。
_SI_CODE_BY_CAUSE = {
    0: (1, "BUS_ADRALN"),
    1: (2, "SEGV_ACCERR"),
    2: (1, "ILL_ILLOPC"),
    4: (1, "BUS_ADRALN"),
    5: (2, "SEGV_ACCERR"),
    6: (1, "BUS_ADRALN"),
    7: (2, "SEGV_ACCERR"),
    13: (2, "SEGV_ACCERR"),
    15: (2, "SEGV_ACCERR"),
}
_MEMORY_STORE_MNEMONICS = tuple(
    "sb sh sw sd sq cbo.zero c.sb c.sh c.sw c.sd c.sq c.fsw c.fsd c.fsq sc. amo "
    "vse vsse vso vsu vsm vs1 vs2 vs4 vs8 fsb fsh fsw fsd fsq".split()
)
_MEMORY_LOAD_MNEMONICS = tuple(
    "lb lh lw ld lq c.lb c.lh c.lw c.ld c.lq c.flw c.fld c.flq lr. vl flb flh "
    "flw fld flq".split()
)


def _child_timeout_seconds() -> float:
    try:
        value = float(os.environ.get("RQ1_BACKEND_TIMEOUT_SECONDS", "60"))
    except (TypeError, ValueError):
        return 60.0
    return value if math.isfinite(value) and value > 0 else 60.0


CHILD_TIMEOUT_SEC = _child_timeout_seconds()
_LIBRISCV_GDB_LOCK_PATH = "/tmp/migration-emi-libriscv-gdb.lock"
_LIBRISCV_GDB_PORT = 2159


def _mode_args(mode: str) -> list[str]:
    if mode == "interpreter":
        # -n alone permits the fast interpreter's coarse protection path;
        # -1 makes libriscv stop at the exact guest instruction/exception.
        return ["-n", "-1"]
    if mode == "translated":
        return []
    raise ValueError(f"unsupported libriscv mode: {mode}")


def _count_args(mode: str) -> list[str]:
    return [*_mode_args(mode), "-a"] if mode == "interpreter" else ["-n", "-a"]


def parse_libriscv_trace(text: str) -> tuple[int, ...]:
    pcs = []
    for line in text.splitlines():
        if len(line) < 2 or line[0].lower() != "f" or not line[1].isspace():
            continue
        match = TRACE_PC_RE.match(line)
        if match is None:
            if re.search(r"\bpc\b", line, re.IGNORECASE):
                return ()
            continue
        try:
            pc = _pc_int("0x" + match.group(1))
        except ValueError:
            return ()
        if pc & 1:
            return ()
        pcs.append(pc)
    return tuple(pcs)


def _machine_exception_state(stdout: bytes) -> dict[str, object] | None:
    match = MACHINE_EXCEPTION_RE.search(stdout.decode("utf-8", errors="replace"))
    if match is None:
        return None
    try:
        fault_pc = _pc_int("0x" + match.group("pc"))
    except ValueError:
        return None
    code = int(match.group("code"))
    message = match.group("message").strip()
    message_lower = message.lower()
    instruction = match.group("instruction").lower().split()
    raw_instruction = (
        instruction[0]
        if instruction and re.fullmatch(r"(?:0x)?[0-9a-f]+", instruction[0])
        else None
    )
    mnemonic = (
        instruction[1]
        if raw_instruction is not None and len(instruction) > 1
        else instruction[0]
        if instruction
        else ""
    )
    is_ebreak = code == 7 and message_lower.startswith("ebreak instruction")
    cause = 3 if is_ebreak else _GUEST_CAUSE_BY_MACHINE_EXCEPTION_CODE.get(code)
    if code == 2:
        if "write" in message_lower or mnemonic.startswith(_MEMORY_STORE_MNEMONICS):
            cause = 15
        elif "read" in message_lower or mnemonic.startswith(_MEMORY_LOAD_MNEMONICS):
            cause = 13
    elif code == 9:
        if "write" in message_lower or mnemonic.startswith(_MEMORY_STORE_MNEMONICS):
            cause = 6
        elif "read" in message_lower or mnemonic.startswith(_MEMORY_LOAD_MNEMONICS):
            cause = 4
    # fault_pc 取自 ``[0x...]`` 指令行：rvlinux 对多数异常只打印异常前最后
    # 一条已执行指令（如 cause 6/9），EBREAK 场景才打印异常指令本身；因此
    # fault_pc 语义为“异常前最后执行 PC”。data 保留为原始异常证据，只有
    # 来源明确是完整地址时才投影为 fault_address。
    signal_name = (
        "SIGTRAP" if is_ebreak
        else _SIGNAL_BY_MACHINE_EXCEPTION_CODE.get(code)
    )
    si_code, si_code_name = _SI_CODE_BY_CAUSE.get(cause, (None, None))
    state: dict[str, object] = {
        "signal": signal_name,
        "signal_code": _SIGNAL_NUMBER_BY_NAME.get(signal_name) if signal_name else None,
        "fault_pc": fault_pc,
        "extra_state": {
            "libriscv_machine_exception": message,
            "libriscv_machine_exception_code": code,
        },
    }
    if fault_pc & 1:
        state["extra_state"]["fault_pc_observer_gap"] = (
            "libriscv exception PC is unaligned"
        )
    if signal_name is not None:
        state["extra_state"]["guest_trap"] = "delivered"
        state["extra_state"]["trap_observer"] = "libriscv-machine-exception"
        state["extra_state"]["guest_epc"] = fault_pc
        state["extra_state"]["guest_epc_observer"] = "libriscv-machine-exception"
    if cause is not None:
        state["extra_state"]["trap.cause"] = cause
        state["extra_state"]["guest_cause"] = cause
    if raw_instruction is not None and code in {0, 1, 5, 6} and cause == 2:
        try:
            tval = _pc_int("0x" + raw_instruction)
        except ValueError:
            pass
        else:
            state["extra_state"]["guest_tval"] = tval
            state["extra_state"]["guest_tval_observer"] = "libriscv-machine-exception"
    if si_code is not None:
        state["extra_state"]["signal.si_code"] = si_code
        state["extra_state"]["signal.si_code_observer"] = "libriscv-machine-exception"
        state["extra_state"]["signal.si_code_name"] = si_code_name
        state["extra_state"]["signal.si_code_name_observer"] = "libriscv-machine-exception"
    raw_data = match.group("data")
    if raw_data is not None:
        try:
            tval = _pc_int("0x" + raw_data)
            state["extra_state"]["libriscv_machine_exception_data"] = tval
            if code == 7 and cause == 3:
                state["extra_state"]["guest_tval"] = tval
                state["extra_state"]["guest_tval_observer"] = "libriscv-machine-exception"
            if code == 2 and cause in (13, 15):
                state["extra_state"]["guest_tval_gap"] = (
                    "libriscv protection data may be a virtual-page base"
                )
        except ValueError:
            pass
    # 只有控制流/指令未对齐的 data 明确是地址；保护/数据对齐
    # 路径还会传页基址或页内 offset，保留 raw data，不冒充完整地址。
    if raw_data is not None and code == 3:
        state["extra_state"]["guest_tval_gap"] = (
            "libriscv execution protection data may be a virtual-page base"
        )
    if raw_data is not None and code == 4:
        try:
            tval = _pc_int("0x" + raw_data)
            state["fault_address"] = hex(tval)
            state["extra_state"]["signal.si_addr"] = state["fault_address"]
            state["extra_state"]["signal.si_addr_observer"] = "libriscv-machine-exception"
            state["extra_state"]["guest_tval"] = tval
            state["extra_state"]["guest_tval_observer"] = "libriscv-machine-exception"
        except ValueError:
            pass
    if raw_data is not None and code == 9:
        state["extra_state"]["guest_tval_gap"] = (
            "libriscv alignment data may be a page offset"
        )
    state["extra_state"]["observer_fields"] = observed_state_fields(
        state["extra_state"]
    )
    return state


def _gdb_trace(
    rvlinux: list[str], elf: str, mode: str, *, terminal_pc: int | None = None,
) -> tuple[tuple[int, ...], dict, dict]:
    if mode == "translated":
        return _translated_trace(rvlinux, elf)
    command = [*rvlinux, *_mode_args(mode), "-g", "-F", "-s", elf]
    pcs: list[int] = []
    reached = False
    error = None
    proc = None
    client = None
    stdout = b""
    stderr = b""
    lock = _Flock(_LIBRISCV_GDB_LOCK_PATH)
    try:
        stop_pc = terminal_pc if terminal_pc is not None else exit_checkpoint_pc(elf)
        lock.__enter__()
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        client = _connect_client(_LIBRISCV_GDB_PORT, proc, CHILD_TIMEOUT_SEC)
        reply = client.request("?")
        while True:
            if not reply or reply[0] not in {"S", "T", "X"}:
                error = f"unexpected-stop-reply:{reply!r}"
                break
            pc = client.read_register_u64(32)
            if pc & 1:
                error = f"unaligned-pc:{pc:#x}"
                break
            pcs.append(pc)
            if pc == stop_pc:
                reached = True
                break
            reply = client.request("s")
    except (
        ConnectionError, OSError, RuntimeError, subprocess.TimeoutExpired,
        TimeoutError, ValueError,
    ) as exc:
        error = str(exc)
    finally:
        if client is not None:
            client.close()
        if proc is not None:
            if proc.poll() is None:
                try:
                    pid = getattr(proc, "pid", None)
                    if os.name == "posix" and isinstance(pid, int):
                        os.killpg(pid, signal_module.SIGKILL)
                    else:
                        proc.kill()
                except (OSError, ProcessLookupError):
                    pass
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        lock.__exit__(None, None, None)
    try:
        loads = elf_executable_loads(Path(elf))
    except (OSError, ValueError):
        loads = ()
    outside = bool(pcs) and not pc_path_in_ranges(pcs, loads)
    if outside:
        pcs = []
    trace_failed = bool(error or outside or not reached or not pcs)
    details = {
        "trace_returncode": 0 if reached else proc.returncode if proc is not None else None,
        "guest_trap": terminal_pc is not None and reached,
        "trace_failed": trace_failed,
        "trace_parse_error": False,
        "trace_checkpoint_error": None if reached else error or "checkpoint-not-reached",
        "trace_stop": "guest-exception" if terminal_pc is not None else "exit-checkpoint",
        "trace_pc_outside_elf": outside,
        "exception_pc_outside_elf": False,
        "granularity": "gdb-single-step-pc" if pcs else "missing",
        "trace_mechanism": "gdb-rsp-single-step",
        "translator_event_trace": None,
        "translated_pcs": [],
        "translated_pc_digest": None,
        "translation_trace_digest": None,
        "target_path_digest": canonical_digest({
            "backend": f"libriscv-{mode}",
            "granularity": "gdb-single-step-pc" if pcs else "missing",
            "executed_pcs": [hex(pc) for pc in pcs],
        }),
        "target_path_digest_level": "libriscv-gdb-single-step-pc-sequence",
    }
    return tuple(pcs), details, {
        "exit_code": details["trace_returncode"],
        "stdout_hex": stdout.hex(),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "command_fingerprint": canonical_digest(command),
    }


def _translated_trace(rvlinux: list[str], elf: str) -> tuple[tuple[int, ...], dict, dict]:
    command = [*rvlinux, "--no-background", "-T", "-s", elf]
    proc = _execute(command, None)
    text = proc.stdout.decode("utf-8", errors="replace") + "\n" + proc.stderr.decode("utf-8", errors="replace")
    exception = _machine_exception_state(proc.stdout + b"\n" + proc.stderr)
    guest_trap = bool(exception and exception.get("extra_state", {}).get("guest_trap") == "delivered")
    pcs = parse_libriscv_trace(text)
    pcs, checkpoint_error = _clip_trace_at_checkpoint(pcs, proc.stdout)
    if _observation_frame_prefix(proc.stdout) is None and checkpoint_error is None:
        try:
            checkpoint = exit_checkpoint_pc(elf)
            if checkpoint not in pcs:
                checkpoint_error = "exit_checkpoint is absent from translation trace"
            else:
                pcs = tuple(pcs[:pcs.index(checkpoint) + 1])
        except (OSError, RuntimeError, ValueError) as exc:
            checkpoint_error = str(exc)
    parse_error = bool(re.search(r"(?mi)^f\s", text)) and not pcs
    try:
        loads = elf_executable_loads(Path(elf))
    except (OSError, ValueError):
        loads = ()
    outside = bool(pcs) and not pc_path_in_ranges(pcs, loads)
    exception_outside = False
    if exception is not None:
        try:
            exception_outside = not pc_path_in_ranges((_pc_int(exception["fault_pc"]),), loads)
        except (KeyError, TypeError, ValueError):
            exception_outside = True
    if outside:
        pcs = ()
    trace_failed = (
        (proc.returncode != 0 and exception is None)
        or outside or exception_outside or not pcs or parse_error or checkpoint_error is not None
    )
    details = {
        "trace_returncode": proc.returncode,
        "guest_trap": guest_trap,
        "exception_state": exception,
        "trace_failed": trace_failed,
        "trace_parse_error": parse_error,
        "trace_checkpoint_error": checkpoint_error,
        "trace_pc_outside_elf": outside,
        "exception_pc_outside_elf": exception_outside,
        "granularity": "translated-per-pc" if pcs else "missing",
        "trace_mechanism": "libriscv-translator-trace",
        "translator_event_trace": None,
        "translated_pcs": [hex(pc) for pc in pcs],
        "translated_pc_digest": pc_digest(pcs),
        "translation_trace_digest": pc_digest(pcs),
        "target_path_digest": canonical_digest({
            "backend": "libriscv-translated",
            "granularity": "translated-per-pc" if pcs else "missing",
            "translated_or_executed": [hex(pc) for pc in pcs],
        }),
        "target_path_digest_level": "libriscv-translate-trace-pc-sequence",
    }
    return pcs, details, {
        "exit_code": proc.returncode,
        "stdout_hex": proc.stdout.hex(),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
        "command_fingerprint": canonical_digest(command),
    }


def _evidence(
    mode: str, executed: tuple[int, ...], total_count: int | None, trace_details: dict,
) -> dict:
    trace_returncode = trace_details.get("trace_returncode")
    trap_path = (
        trace_details.get("guest_trap") is True
        and bool(executed)
        and trace_details.get("trace_pc_outside_elf") is not True
        and trace_details.get("exception_pc_outside_elf") is not True
    )
    trace_failed = (bool(trace_details.get("trace_failed")) and not trap_path) or (
        "trace_returncode" in trace_details
        and (trace_returncode is None or trace_returncode != 0)
        and trace_details.get("guest_trap") is not True
    )
    count_attempted = mode == "translated" and "accurate_count_returncode" in trace_details
    count_missing = count_attempted and total_count is None
    count_failed = count_attempted and trace_details.get("accurate_count_returncode") != 0
    count_gap = count_failed or count_missing
    executed_set = set(executed)
    translated = mode == "translated"
    trace_available = trace_details.get("granularity") in {"gdb-single-step-pc", "translated-per-pc"}
    path_available = trace_available and not trace_failed and bool(executed)
    count_mismatch = (
        count_attempted
        and not count_gap
        and path_available
        and not trap_path
        and total_count != len(executed)
    )
    if count_mismatch and not translated:
        trace_failed = True
        path_available = False
    effective_count = (
        len(executed)
        if path_available
        else None if count_failed or count_missing else total_count
    )
    fallback_count = (
        int(effective_count or 0)
        if translated and not trace_failed and not executed
        else 0
    )
    return {
        "backend": "libriscv-translated" if translated else "libriscv-interpreter",
        "expected_path": "translated" if translated else "interpreter",
        "tested_pc_seen": False,
        "tested_pc_translated": False,
        "tested_pc_executed": False,
        "translated_count": len(executed_set) if path_available else 0,
        "execution_count": effective_count if effective_count is not None else 0,
        "interpreter_count": 0 if translated else int(effective_count or 0),
        "fallback_count": fallback_count,
        "details": {
            "source": "libriscv-runner-wrapper",
            **trace_details,
            "trace_checkpoint_error": None if trap_path else trace_details.get("trace_checkpoint_error"),
            "translator_event_trace": None,
            "path_classification": (
                "trace-failed"
                if trace_failed
                else "translated-gdb-single-step"
                if translated and path_available and executed
                and trace_details.get("granularity") == "gdb-single-step-pc"
                else "interpreter-gdb-single-step"
                if not translated and path_available
                else "translated-per-pc"
                if translated and path_available and executed
                else "fallback-no-translated-pc"
                if fallback_count
                else "unavailable"
            ),
            "tested_pc_count": 0,
            "tested_executed_count": 0,
            "total_guest_instruction_count": effective_count,
            "accurate_count_raw": total_count,
            "trace_failed": trace_failed,
            "count_failed": count_failed,
            "count_mismatch": count_mismatch,
            "count_observer": (
                "gap" if count_gap else "observed"
                if count_attempted else "not-requested"
            ),
            "count_gap": (
                "accurate-count-command-failed" if count_failed
                else "accurate-count-output-missing" if count_missing
                else None
            ),
            "path_identity": _path_identity_alias(
                () if trace_failed else executed,
                trace_details,
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run libriscv with framework trace evidence")
    parser.add_argument("--mode", choices=("interpreter", "translated"), default="translated")
    parser.add_argument("--rvlinux-bin")
    parser.add_argument("elf")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    elf = str(Path(args.elf).resolve())
    try:
        require_execution_plane()
        validate_elf_machine(Path(elf), EM_RISCV, "guest")
        if args.rvlinux_bin:
            binary = resolve_x86_64_runner(args.rvlinux_bin, "rvlinux")
        else:
            manifest = resolve_preferred_libriscv_translated_manifest() if args.mode == "translated" else None
            binary = str(
                load_target_binary_manifest(
                    manifest,
                    expected_backend=f"libriscv-{args.mode}",
                    require_build_attested=True,
                ).binary_path
            ) if manifest is not None else resolve_x86_64_runner(
                str(EXTERNAL_ROOT / "binary-translation-tools" / "libriscv" / "emulator" / ".build" / "rvlinux"),
                "rvlinux",
            )
        rvlinux = [binary]
        guest_binary_sha256 = sha256_file(Path(elf))
    except (ExecutionEnvironmentError, OSError, ValueError) as exc:
        return _fail(str(exc))
    target_binary_details = _target_binary_details(
        rvlinux[0],
        expected_backend=f"libriscv-{args.mode}",
    )
    tool_version = _binary_tool_version(rvlinux[0])
    if tool_version is None and target_binary_details.get("target_identity_status") == "verified":
        tool_version = f"libriscv-source-{target_binary_details['target_source_commit']}"

    isa_profile = (
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
    ).strip().lower()

    try:
        mode_args = _mode_args(args.mode)
        main_command = [*rvlinux, *mode_args, "-s", elf]
        main_proc = _execute(main_command, CHILD_TIMEOUT_SEC)
        observation_state = _machine_exception_state(main_proc.stdout + b"\n" + main_proc.stderr)
        terminal_pc = (
            _pc_int(observation_state["fault_pc"])
            if isinstance(observation_state, dict)
            and isinstance(observation_state.get("fault_pc"), int)
            else None
        )
        count_command = [*rvlinux, *_count_args(args.mode), elf]
        count_proc = _execute(count_command, CHILD_TIMEOUT_SEC)
        count_text = "\n".join(p.decode("utf-8", errors="replace") for p in (count_proc.stdout, count_proc.stderr))
        total_count = (
            int(match.group(1))
            if (match := INSTRUCTION_COUNT_RE.search(count_text))
            else None
        )
        count_details = {"accurate_count_returncode": count_proc.returncode}
        count_run = {
            "exit_code": count_proc.returncode,
            "stdout_hex": (_observation_frame_prefix(count_proc.stdout) or count_proc.stdout).hex(),
            "stderr": count_proc.stderr.decode("utf-8", errors="replace"),
            "command_fingerprint": canonical_digest(count_command),
        }
        executed, trace_details, trace_run = _gdb_trace(
            rvlinux, elf, args.mode, terminal_pc=terminal_pc,
        )
        fp_required = profile_requires_fp_state(isa_profile) and os.environ.get(
            "RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "1"
        ) == "1"
        if not fp_required:
            # Integer-only profiles have no architectural FPR/FFLAGS/FRM state;
            # declare the gap instead of paying a failing GDB round-trip.
            fp_state, fp_state_error = None, None
            fp_gap_declared = {"fp_observer": "not-required"}
        else:
            # libriscv FP observer (N-4): both execution modes' rvlinux GDB
            # RSP stubs expose the
            # architectural FP state directly -- f0..f31 = regnums 33..64
            # (8-byte rawbits), fflags = 66, frm = 67, fcsr = 68 (4-byte
            # uint32, see libriscv lib/libriscv/rsp_server.hpp
            # handle_readreg).  The collector drives the stub to the
            # exit_checkpoint sentinel and reads FPR rawbits/fflags/frm back
            # through single-register packets, matching the capsule adapter's
            # fpr_rawbits+fflags+frm observation.  On any failure the state is
            # not forwarded (no placeholder zeros), so the oracle reports the
            # named target-final-fp-state-missing gap.
            fp_state, fp_state_error = collect_final_fp_state(
                [*rvlinux, "-g", *mode_args, "-s", elf],
                port=_LIBRISCV_GDB_PORT,
                elf_path=elf,
                static_layout=LIBRISCV_TRANSLATED_FP_LAYOUT,
                lock_path=_LIBRISCV_GDB_LOCK_PATH,
            )
            fp_gap_declared = None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _fail(f"libriscv trace runner failed: {exc}")
    if observation_state is None and isinstance(trace_details.get("exception_state"), dict):
        observation_state = trace_details["exception_state"]
    observation_frame = _observation_frame_prefix(main_proc.stdout)
    terminal_ebreak = (
        observation_frame is not None
        and isinstance(observation_state, dict)
        and observation_state.get("extra_state", {}).get(
            "libriscv_machine_exception_code"
        ) == 7
    )
    if re.search(rb"(?m)^\s*Exception:", main_proc.stdout + b"\n" + main_proc.stderr) and observation_state is None:
        return _fail("libriscv reported a runtime exception")
    if (isinstance(observation_state, dict)
            and observation_state.get("extra_state", {}).get("guest_trap") == "delivered"):
        trace_details["guest_trap"] = True

    # rvlinux appends its EBREAK diagnostic after the binary frame.  Keep the
    # framed observation on stdout; the diagnostic is already projected below.
    if terminal_ebreak:
        # ebreak 是 harness checkpoint，不应作为正常 RVOBS1 结果的 OS signal。
        observation_state["signal"] = None
        observation_state["signal_code"] = None
    sys.stdout.buffer.write(observation_frame or main_proc.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(main_proc.stderr)
    if main_proc.stderr and not main_proc.stderr.endswith(b"\n"):
        sys.stderr.write("\n")
    # rvlinux builds have emitted machine exceptions on either stream across
    # versions; parse both, while keeping stdout/stderr transport separate in
    # the execution evidence below.
    observation_state = observation_state or {"extra_state": {}}
    extra_state = observation_state.setdefault("extra_state", {})
    if fp_gap_declared is not None:
        extra_state.update(fp_gap_declared)
    if fp_state is not None:
        extra_state.update(fp_state)
    elif fp_state_error is not None:
        extra_state.update(
            fp_observer="not-observed",
            fp_observer_gap=f"libriscv-final-fp-state-unavailable: {fp_state_error}",
            adapter_execution_error=f"final-fp-state-unavailable: {fp_state_error}",
        )
        sys.stderr.write(f"RV_FINAL_FP_STATE_STATUS=unavailable reason={fp_state_error}\n")
    guest_trap = extra_state.get("guest_trap") == "delivered"
    extra_state["observer_fields"] = observed_state_fields(extra_state)
    if observation_state.get("signal") is not None or extra_state:
        sys.stderr.write("RV_OBSERVATION_STATE=" + json.dumps(observation_state, separators=(",", ":")) + "\n")
    if executed:
        sys.stderr.write(f"RV_EXECUTED_PCS={json.dumps([hex(pc) for pc in executed], separators=(',', ':'))}\n")
        sys.stderr.write(f"RV_INSTRUCTION_COUNT={len(executed)}\n")
    if args.mode == "translated" and total_count is not None and total_count > 0 and count_proc.returncode == 0:
        sys.stderr.write(f"RV_TOTAL_GUEST_INSTRUCTION_COUNT={total_count}\n")
    if tool_version is not None:
        sys.stderr.write(f"RV_TOOL_VERSION={tool_version}\n")
    try:
        target_configuration = _shared_target_configuration(
            {
                "isa_profile": isa_profile,
                "execution_mode": args.mode,
                "guest_binary_sha256": guest_binary_sha256,
            },
            target_binary_details,
        )
        evidence = _evidence(
            args.mode,
            executed,
            total_count,
            {
                **count_details,
                **trace_details,
                **target_binary_details,
                "configuration_identity": target_configuration,
            },
        )
    except ValueError as exc:
        return _fail(str(exc))
    sys.stderr.write("RV_TRANSLATION_EVIDENCE=" + json.dumps(evidence, separators=(",", ":")) + "\n")
    trace_markers = ""
    if executed:
        trace_markers = (
            "\nRV_EXECUTED_PCS="
            + json.dumps([hex(pc) for pc in executed], separators=(",", ":"))
            + f"\nRV_INSTRUCTION_COUNT={len(executed)}"
            + (
                f"\nRV_TOTAL_GUEST_INSTRUCTION_COUNT={total_count}\n"
                if args.mode == "translated" and total_count is not None and total_count > 0 and count_proc.returncode == 0 else "\n"
            )
        )
    elif args.mode == "translated" and total_count is not None and total_count > 0 and count_proc.returncode == 0:
        trace_markers = f"\nRV_TOTAL_GUEST_INSTRUCTION_COUNT={total_count}\n"
    trace_stderr = (
        trace_run["stderr"] + trace_markers
        if trace_run is not None else ""
    )
    observation_marker = (
        "RV_OBSERVATION_STATE="
        + json.dumps(observation_state, separators=(",", ":"))
        + "\n"
        if guest_trap else ""
    )
    trace_stderr += observation_marker
    count_stderr = count_run["stderr"] + observation_marker
    sys.stderr.write(
        "RV_EXECUTION_EVIDENCE="
        + json.dumps(
            {
                "schema_version": "runner-execution-evidence-v1",
                "main_command_fingerprint": canonical_digest(main_command),
                "trace_command_fingerprint": (
                    trace_run["command_fingerprint"] if trace_run is not None else canonical_digest(main_command)
                ),
                "count_command_fingerprint": count_run["command_fingerprint"],
                "trace_run": (
                    {
                        "exit_code": trace_run["exit_code"],
                        "stdout_hex": trace_run["stdout_hex"],
                        "stderr": trace_stderr,
                    }
                    if trace_run is not None
                    else None
                ),
                "count_run": {
                    "exit_code": count_run["exit_code"],
                    "stdout_hex": count_run["stdout_hex"],
                    "stderr": count_stderr,
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stderr.flush()
    return 0 if terminal_ebreak else main_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

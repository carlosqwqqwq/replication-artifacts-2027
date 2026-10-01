import atexit
import json
import math
import os
import re
import signal as signal_module
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.execution_environment import (
    EM_RISCV,
    ExecutionEnvironmentError,
    load_target_binary_manifest,
    resolve_current_container_source_verified_manifest,
    require_execution_plane,
    resolve_x86_64_runner,
    validate_elf_machine,
)
from framework.adapters.rsp import _PORT_STARTUP_LOCK, free_port
from framework.adapters.runner import _observation_frame, observed_state_fields
from framework.cli._gdb_remote import collect_final_fp_state
from framework.direct_case import (
    _SIGNAL_CODE_BY_NAME as _SIGNAL_NUMBER_BY_NAME,
    _SIGNAL_NAME_BY_TEXT,
)
from framework.spec_definedness import (
    enabled_extensions, is_canonical_isa_profile, x_register_fp_extensions,
)
from framework._util import (
    canonical_digest,
    elf_executable_loads,
    pc_digest,
    pc_int as _pc_int,
    pc_path_in_ranges,
    runner_contract_gap as _fail,
    sha256_file,
)
from framework.adapters.trace_common import (
    _binary_tool_version,
    _clip_trace_at_checkpoint,
    _execute,
    _path_identity_alias,
    _target_binary_details,
    _target_configuration as _shared_target_configuration,
)



def _child_timeout_seconds() -> float:
    try:
        value = float(os.environ.get("RQ1_BACKEND_TIMEOUT_SECONDS", "30"))
    except (TypeError, ValueError):
        return 30.0
    return value if math.isfinite(value) and value > 0 else 30.0


CHILD_TIMEOUT_SEC = _child_timeout_seconds()
# Final FP/RVV state is an optional observer layered on top of the completed
# mailbox/trace run.  It must not inherit a 180-second guest timeout: a GDB
# stub that cannot reach the sentinel should produce an observer gap quickly.
FINAL_STATE_TIMEOUT_SEC = min(CHILD_TIMEOUT_SEC, 20.0)
_QEMU_RV32_CPU = {
    "rv32i": "rv32,m=false,a=false,zawrs=false,zicsr=false,f=false,zfa=false,d=false,zifencei=false",
    "rv32im": "rv32,m=true,a=false,zawrs=false,zicsr=false,f=false,zfa=false,d=false,zifencei=false",
    "rv32imc": "rv32,m=true,c=true,a=false,zawrs=false,zicsr=false,f=false,zfa=false,d=false,zifencei=false",
    "rv32ia": "rv32,m=false,a=true,c=false,zawrs=false,zicsr=false,f=false,zfa=false,d=false,zifencei=false",
    "rv32imac": "rv32,m=true,a=true,c=true,zawrs=false,zicsr=false,f=false,zfa=false,d=false,zifencei=false",
}
_QEMU_RV32_ORDER = (
    "m", "a", "f", "d", "c", "v", "zcb", "zcmp", "zcmt", "zicsr",
    "zifencei", "zawrs", "zacas", "zba", "zbb", "zbc", "zbs", "zfinx", "zdinx", "zhinx",
    "xtheadbb", "xtheadbs",
)
_QEMU_RV64_ORDER = (
    "m", "a", "f", "d", "q", "v", "c", "zabha", "zacas", "zalasr", "zba",
    "zbb", "zbc", "zbkb", "zbkx", "zbs", "zcb", "zcmop", "zcmp", "zcmt", "zfa",
    "zfbfmin", "zfhmin", "zfh", "zvfh", "zicbom", "zicbop", "zicboz", "zicntr", "zicond",
    "zicsr", "zifencei", "zawrs", "zihintpause", "zihpm", "zimop", "zk", "zknh",
    "zks", "zkt",
)


def _stage_guest_elf(guest_elf: str) -> str:
    with tempfile.NamedTemporaryFile(prefix="qemu-guest-", suffix=".elf", delete=False) as temp_guest:
        execution_elf = Path(temp_guest.name)
    atexit.register(execution_elf.unlink, missing_ok=True)
    shutil.copyfile(guest_elf, execution_elf)
    execution_elf.chmod(0o700)
    return str(execution_elf)


def _configured_tested_pcs() -> tuple[int, ...]:
    values = []
    for raw in os.environ.get("RV_TESTCASE_RISK_PCS", "").split(","):
        if not raw:
            continue
        try:
            value = _pc_int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0 and not value & 1:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _qemu_cpu_for_isa(
    isa_profile: str | None, disabled_extensions: tuple[str, ...] = (),
) -> str | None:
    if not is_canonical_isa_profile(isa_profile):
        return None
    disabled = {str(item).lower() for item in disabled_extensions}
    profile = isa_profile.lower()
    vendor = next(
        (name for name in ("xtheadbb", "xtheadbs")
         if name in enabled_extensions(profile) and name not in disabled),
        None,
    )
    if profile.startswith("rv32") and vendor:
        return f"rv32,{vendor}=true"
    if isa_profile in _QEMU_RV32_CPU and not disabled:
        return _QEMU_RV32_CPU[isa_profile]
    enabled = enabled_extensions(profile) - disabled
    if profile.startswith("rv32"):
        if "e" in enabled or set(enabled) - set(_QEMU_RV32_ORDER) - {"i", "e"}:
            return None
        if any(ext in enabled for ext in ("zfinx", "zdinx", "zhinx")):
            return None
        effective = enabled | ({"zicsr"} if {"f", "d"} & enabled else set())
        properties = [f"{ext}=true" for ext in _QEMU_RV32_ORDER if ext in effective]
        properties += [
            f"{ext}=false" for ext in ("m", "a", "c", "zawrs", "zicsr", "f", "zfa", "d", "zifencei", "xtheadbb", "xtheadbs")
            if ext not in effective
        ]
        properties += [
            f"{ext}=false" for ext in disabled
            if ext in _QEMU_RV32_ORDER and ext not in {"m", "a", "c", "zawrs", "zicsr", "f", "zfa", "d", "zifencei"}
        ]
        return ",".join(("rv32", *properties))
    if not profile.startswith("rv64") or x_register_fp_extensions(profile):
        return None
    if "g" in enabled:
        enabled = frozenset((enabled - {"g"}) | {"i", "m", "a", "f", "d"})
    if any(token.startswith(("zve", "zvl")) for token in enabled):
        enabled = frozenset((*enabled, "v"))
    if "zicbo" in enabled:
        enabled = frozenset((*enabled, "zicbom", "zicbop", "zicboz"))
    if set(enabled) - set(_QEMU_RV64_ORDER) - {"i", "zca", "zicbo"}:
        return None
    properties = [f"{ext}={'true' if ext in enabled else 'false'}" for ext in ("m", "a", "f", "d")]
    compressed = "c" in enabled or any(token.startswith("zc") for token in enabled)
    properties.append(f"c={'true' if compressed else 'false'}")
    properties += [
        f"{ext}=true" for ext in _QEMU_RV64_ORDER
        if ext in enabled and ext not in {"m", "a", "f", "d", "c"}
    ]
    properties += [
        f"{ext}=false" for ext in ("zfa", "zawrs", *disabled)
        if ext in _QEMU_RV64_ORDER
        and ext not in {"m", "a", "f", "d", "c"}
        and ext not in enabled
    ]
    if "v" in enabled:
        properties.append("vlen=256")
    return ",".join(("rv64", *properties))


_QEMU_TARGET_SIGNAL_RE = re.compile(
    r"uncaught target signal\s+(?P<code>[0-9]+)\s+\((?P<name>[^)]+)\)",
    re.IGNORECASE,
)
_QEMU_HOST_ABORT_RE = re.compile(
    r"unhandled CPU exception\s+(?P<cause>0x[0-9a-fA-F]+|[0-9]+)",
    re.IGNORECASE,
)
_QEMU_INTERNAL_SIGNAL_RE = re.compile(
    r"QEMU internal SIG\w+\s*\{[^}]*addr=(?P<addr>0x[0-9a-fA-F]+|[0-9]+)",
    re.IGNORECASE,
)
_QEMU_STRACE_SIGNAL_RE = re.compile(
    r"^---\s+(?P<sig>SIG\w+)\s+\{si_signo=(?P<signo>SIG\w+|[0-9]+),\s*si_code=(?P<code>-?[0-9]+|[A-Za-z_][A-Za-z0-9_]*),\s*si_addr=(?P<addr>0x[0-9a-fA-F]+|[0-9]+|NULL)\}\s*---$",
    re.IGNORECASE,
)
_SI_CODE_BY_NAME = {
    "SI_USER": 0,
    "SI_KERNEL": 0x80,
    "SI_QUEUE": -1,
    "SI_TIMER": -2,
    "SI_MESGQ": -3,
    "SI_ASYNCIO": -4,
    "SI_SIGIO": -5,
    "SI_TKILL": -6,
}
# linux-user si_code 数值 -> 语义名，按信号区分（linux-user/syscall_defs.h）。
# si_code 只在对应信号下有定义（SEGV_ACCERR=2 与 BUS_ADRALN=1 数值相同），
# 因此键是 (signal, code)。QEMU 11.0.x 实测：cbo.zero 只读页 fault ->
# SIGSEGV/SEGV_ACCERR(2)；misaligned LR -> SIGBUS/BUS_ADRALN(1)。
_SI_CODE_NAME_BY_SIGNAL = {
    "SIGILL": {1: "ILL_ILLOPC", 2: "ILL_ILLOPN", 3: "ILL_ILLADR"},
    "SIGSEGV": {1: "SEGV_MAPERR", 2: "SEGV_ACCERR"},
    "SIGBUS": {1: "BUS_ADRALN", 2: "BUS_ADRERR", 3: "BUS_OBJERR"},
    "SIGTRAP": {1: "TRAP_BRKPT", 2: "TRAP_TRACE"},
    "SIGFPE": {1: "FPE_INTDIV", 2: "FPE_INTOVF", 3: "FPE_FLTDIV"},
}
_SI_CODE_BY_NAME.update(
    {
        name: code
        for signal_codes in _SI_CODE_NAME_BY_SIGNAL.values()
        for code, name in signal_codes.items()
    }
)
# QEMU linux-user uncaught signal → RISC-V user-mode trap cause 最小映射。
# QEMU 11.0.50 实测（qemu-cpu-probe-2026-08-10.txt）与 RISC-V privileged
# spec 一致：SIGILL=illegal instruction(cause 2)、SIGTRAP=breakpoint(cause 3)。
_TRAP_CAUSE_BY_SIGNAL = {
    "SIGILL": 2,
    "SIGTRAP": 3,
}
_LOAD_OPCODES = {0x03, 0x07}
_STORE_OPCODES = {0x0F, 0x23, 0x27, 0x2F}


def _memory_trap_cause(signal_name: object, instruction_word: object) -> int | None:
    if type(instruction_word) is not int:
        return None
    opcode = instruction_word & 0x7F
    is_atomic_load = opcode == 0x2F and (instruction_word >> 27) & 0x1F == 0b00010
    if opcode in _LOAD_OPCODES or is_atomic_load:
        return {"SIGSEGV": 13, "SIGBUS": 4}.get(signal_name)
    if opcode in _STORE_OPCODES:
        return {"SIGSEGV": 15, "SIGBUS": 6}.get(signal_name)
    return None


def parse_qemu_trace_details(
    text: str,
) -> tuple[tuple[int, ...], tuple[int, ...], str | None]:
    executed: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.lstrip()
        if not line.startswith("Trace "):
            continue
        bracket_start = line.find("[")
        bracket_end = line.find("]", bracket_start + 1)
        if bracket_start < 0 or bracket_end < 0:
            return tuple(dict.fromkeys(executed)), tuple(executed), \
                f"malformed-row-at-line-{line_number}:{line}"
        fields = line[bracket_start + 1 : bracket_end].split("/")
        if len(fields) < 2:
            return tuple(dict.fromkeys(executed)), tuple(executed), \
                f"malformed-row-at-line-{line_number}:{line}"
        raw_pc = fields[1]
        try:
            pc = _pc_int(raw_pc if raw_pc.lower().startswith("0x") else "0x" + raw_pc)
        except ValueError:
            return tuple(dict.fromkeys(executed)), tuple(executed), \
                f"invalid-pc-at-line-{line_number}:{line}"
        if pc & 1:
            return tuple(dict.fromkeys(executed)), tuple(executed), \
                f"unaligned-pc-at-line-{line_number}:{line}"
        executed.append(pc)
    sequence = tuple(dict.fromkeys(executed))
    return sequence, tuple(executed), None


def parse_qemu_cpu_state_trace(
    text: str, executed: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    frames = []
    pc = None
    registers = {}
    for line in text.splitlines():
        match = re.match(r"\s+pc\s+([0-9a-fA-F]+)", line)
        if match:
            if pc is not None and len(registers) == 32:
                frames.append((pc, tuple(registers[index] for index in range(32))))
            pc, registers = int(match.group(1), 16), {}
            continue
        for match in re.finditer(r"x(\d+)/[^\s]+\s+([0-9a-fA-F]+)", line):
            registers[int(match.group(1))] = int(match.group(2), 16)
    if pc is not None and len(registers) == 32:
        frames.append((pc, tuple(registers[index] for index in range(32))))
    count = len(executed)
    if len(frames) < count + 1 or tuple(item[0] for item in frames[:count]) != executed:
        return ()
    return tuple(
        {
            "pc": executed[index],
            "before_gpr": list(frames[index][1]),
            "after_gpr": list(frames[index + 1][1]),
            "after_pc": frames[index + 1][0],
        }
        for index in range(count)
    )


def _instruction_word_at_pc(elf: str, pc: int) -> int | None:
    data = Path(elf).read_bytes()
    # ELF32/ELF64 headers need at least the first 58 bytes.  A truncated or
    # malformed artifact is an evidence gap, not an adapter exception.
    if len(data) < 58 or data[:4] != b"\x7fELF" or data[5] != 1:
        return None
    if data[4] == 2:
        phoff, phentsize, phnum = (
            int.from_bytes(data[32:40], "little"),
            int.from_bytes(data[54:56], "little"),
            int.from_bytes(data[56:58], "little"),
        )
        fmt, fields = "<IIQQQQQQ", (0, 2, 3, 5, 6, 1)
    elif data[4] == 1:
        phoff, phentsize, phnum = (
            int.from_bytes(data[28:32], "little"),
            int.from_bytes(data[42:44], "little"),
            int.from_bytes(data[44:46], "little"),
        )
        fmt, fields = "<IIIIIIII", (0, 1, 2, 4, 5, 6)
    else:
        return None
    if phentsize != struct.calcsize(fmt):
        return None
    for index in range(phnum):
        entry_offset = phoff + index * phentsize
        if entry_offset < 0 or entry_offset + phentsize > len(data):
            return None
        values = struct.unpack_from(fmt, data, entry_offset)
        p_type, file_offset, virtual, file_size, _memory_size, flags = (
            values[position] for position in fields
        )
        if p_type != 1 or not flags & 1 or not virtual <= pc < virtual + file_size:
            continue
        offset = file_offset + pc - virtual
        if offset + 2 > len(data):
            return None
        size = 2 if int.from_bytes(data[offset:offset + 2], "little") & 3 != 3 else 4
        return int.from_bytes(data[offset:offset + size], "little") if offset + size <= len(data) else None
    return None


def _trace_run_directory(trace_root: Path, backend: str) -> Path:
    parent = trace_root / "traces" / backend
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="trace-", dir=parent))


def _trace_artifact_record(
    path: Path, trace_root: Path, role: str, returncode: int,
) -> dict[str, object]:
    recorded = path.is_file()
    size = path.stat().st_size if recorded else None
    return {
        "role": role,
        "status": "recorded" if recorded else "missing",
        "path": path.relative_to(trace_root).as_posix(),
        "absolute_path": str(path.resolve()),
        "bytes": size,
        "sha256": sha256_file(path) if recorded else None,
        "returncode": returncode,
    }


def _execute_full_trace(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    # The outer campaign owns cancellation; an inner trace timeout would make
    # the saved QEMU log an arbitrarily truncated prefix.
    return subprocess.run(command, capture_output=True, check=False)


def _trace_result(
    qemu: list[str], elf: str, backend: str, trace_root: Path,
) -> tuple[tuple[int, ...], tuple[int, ...], dict, dict, tuple[dict[str, object], ...]]:
    trace_root = trace_root.resolve()
    trace_artifacts = []
    trace_dir = _trace_run_directory(trace_root, backend)
    log_path = trace_dir / "qemu.log"
    command = [*qemu, "-one-insn-per-tb", "-d", "exec,nochain", "-D", str(log_path), elf]
    proc = _execute_full_trace(command)
    granularity = "per-instruction-exec"
    actual_command = command
    actual_log_path = log_path
    if proc.returncode != 0 and b"one-insn-per-tb" in (proc.stdout + proc.stderr).lower():
        trace_artifacts.append(_trace_artifact_record(
            log_path, trace_root, "per-instruction-exec", proc.returncode,
        ))
        granularity = "tb-exec"
        trace_dir = _trace_run_directory(trace_root, backend)
        actual_log_path = trace_dir / "qemu.log"
        actual_command = [
            *qemu, "-d", "exec,nochain", "-D", str(actual_log_path), elf,
        ]
        proc = _execute_full_trace(actual_command)
    trace_artifacts.append(_trace_artifact_record(
        actual_log_path, trace_root, granularity, proc.returncode,
    ))
    text = actual_log_path.read_text(encoding="utf-8", errors="replace") if actual_log_path.exists() else ""
    translated, traced_pcs, trace_parse_error_detail = parse_qemu_trace_details(text)
    trace_parse_error = trace_parse_error_detail is not None
    if granularity == "per-instruction-exec":
        executed, trace_checkpoint_error = _clip_trace_at_checkpoint(
            traced_pcs, proc.stdout
        )
        translated = tuple(dict.fromkeys(executed))
    else:
        translated, trace_checkpoint_error = _clip_trace_at_checkpoint(
            translated, proc.stdout
        )
        executed = ()
    trace_pc_outside_elf = False
    if translated or executed:
        try:
            executable_loads = elf_executable_loads(Path(elf))
        except (OSError, ValueError):
            trace_pc_outside_elf = True
        else:
            trace_pc_outside_elf = not pc_path_in_ranges(
                (*translated, *executed), executable_loads
            )
        if trace_pc_outside_elf:
            translated = executed = ()
    terminal_ebreak = (
        not trace_parse_error
        and
        proc.returncode == -_SIGNAL_NUMBER_BY_NAME["SIGTRAP"]
        and bool(traced_pcs)
        and _observation_frame(proc.stdout) is not None
        and not trace_pc_outside_elf
        and _instruction_word_at_pc(elf, traced_pcs[-1]) in {0x9002, 0x00100073}
    )
    state_trace = ()
    if os.environ.get("RV_QEMU_STATE_TRACE") == "1" and executed:
        state_dir = _trace_run_directory(trace_root, backend)
        state_log = state_dir / "qemu-cpu.log"
        state_proc = _execute_full_trace(
            [*qemu, "-one-insn-per-tb", "-d", "cpu", "-D", str(state_log), elf]
        )
        trace_artifacts.append(_trace_artifact_record(
            state_log, trace_root, "cpu-state", state_proc.returncode,
        ))
        state_text = state_log.read_text(encoding="utf-8", errors="replace") if state_log.exists() else ""
        state_trace = parse_qemu_cpu_state_trace(state_text, executed)
    execution_exit_code = 0 if terminal_ebreak else proc.returncode
    stderr = proc.stderr.decode("utf-8", errors="replace")
    trace_host_abort_pc = None
    if _QEMU_HOST_ABORT_RE.search(stderr):
        match = re.search(r"\bpc[ \t]+([0-9a-fA-F]+)", stderr)
        if match is not None:
            trace_host_abort_pc = hex(int(match[1], 16))
    fingerprint_command = list(actual_command)
    fingerprint_command[fingerprint_command.index("-D") + 1] = "<qemu-trace-log>"
    actual_trace_record = next(
        row for row in reversed(trace_artifacts) if row["role"] == granularity
    )
    return translated, executed, {
        "trace_returncode": proc.returncode,
        "guest_exit": _guest_exit_observed(stderr),
        "trace_host_abort_pc": trace_host_abort_pc,
        "trace_termination": (
            {
                "signal": "SIGTRAP",
                "signal_code": 5,
                "guest_trap": "delivered",
                "guest_cause": 3,
                "trap_cause": 3,
                "guest_tval": 0,
                "guest_tval_observer": "qemu-linux-user-ebreak",
                "trap_observer": "qemu-linux-user-ebreak",
                "qemu_uncaught_target_signal": False,
            }
            if terminal_ebreak else _trap_signal_state(stderr) or _trace_process_signal_state(proc.returncode)
        ),
        "terminal_ebreak": terminal_ebreak,
        "trace_parse_error": trace_parse_error,
        "trace_parse_error_detail": trace_parse_error_detail,
        "trace_prefix_pcs": [hex(pc) for pc in traced_pcs],
        "trace_diagnostic": text if trace_parse_error else None,
        "trace_artifacts": trace_artifacts,
        "trace_checkpoint_error": trace_checkpoint_error,
        "trace_pc_outside_elf": trace_pc_outside_elf,
        "granularity": granularity,
        "log_bytes": actual_trace_record["bytes"],
        "tb_block_count": len(re.findall(r"^IN:\s", text, flags=re.MULTILINE)),
        "translated_pcs": [hex(pc) for pc in translated],
        "translated_pc_digest": pc_digest(translated),
        "executed_pc_digest": pc_digest(executed),
        "target_path_digest": canonical_digest({
            "backend": backend,
            "granularity": granularity,
            "translated": [hex(pc) for pc in translated],
            "executed": [hex(pc) for pc in executed],
        }),
        "target_path_digest_level": "qemu-user-log-pc-sequence",
    }, {
        "exit_code": execution_exit_code,
        "stdout_hex": proc.stdout.hex(),
        "stderr": stderr,
        "command_fingerprint": canonical_digest(fingerprint_command),
    }, state_trace


def _evidence(
    backend: str, translated: tuple[int, ...], executed: tuple[int, ...],
    trace_details: dict, tested_pcs: tuple[int, ...] = (),
) -> dict:
    trace_returncode = trace_details.get("trace_returncode")
    termination = trace_details.get("trace_termination")
    if not isinstance(termination, dict):
        termination = None
    trace_host_abort = (
        termination.get("host_abort")
        if isinstance(termination, dict) and termination.get("host_abort") is not None
        else None
    )
    host_abort_pc = None
    if trace_host_abort == "unhandled-cpu-exception":
        raw_host_abort_pc = trace_details.get("trace_host_abort_pc")
        if raw_host_abort_pc is not None:
            try:
                host_abort_pc = _pc_int(raw_host_abort_pc)
            except (TypeError, ValueError):
                host_abort_pc = None
    host_abort_path_observed = (
        trace_host_abort == "unhandled-cpu-exception"
        and type(trace_returncode) is int and trace_returncode > 0
        and trace_details.get("trace_parse_error") is not True
        and trace_details.get("trace_checkpoint_error") is None
        and trace_details.get("trace_pc_outside_elf") is not True
        and trace_details.get("granularity") == "per-instruction-exec"
        and bool(executed)
        and host_abort_pc == executed[-1]
    )
    if trace_host_abort is None and type(trace_returncode) is int and trace_returncode < 0:
        try:
            trace_signal = signal_module.Signals(-trace_returncode).name
        except ValueError:
            trace_signal = str(-trace_returncode)
        if not (
            isinstance(termination, dict)
            and termination.get("guest_trap") == "delivered"
            and termination.get("signal") == trace_signal
            and trace_details.get("main_trap_signal") == trace_signal
        ):
            # 主执行已确认同一 guest trap 时，trace 终止保留为 trap 观察。
            trace_host_abort = f"qemu-trace-signal:{trace_signal}"
    trace_failed = "trace_returncode" in trace_details and (
        trace_returncode is None
        or trace_details.get("trace_parse_error") is True
        or trace_details.get("trace_checkpoint_error") is not None
        or trace_details.get("trace_pc_outside_elf") is True
        or (
            trace_details.get("granularity") in {"per-instruction-exec", "tb-exec"}
            and not (translated or executed)
        )
        or trace_returncode not in (0,)
        and (
            not (translated or executed)
            or termination is not None
            and termination.get("host_abort") is not None
            and not host_abort_path_observed
            or trace_returncode < 0 and termination is None
            or (
                trace_returncode > 0
                and termination is None
                and trace_details.get("guest_exit") is not True
            )
        )
    )
    if trace_failed:
        translated = executed = ()
    translated_set = set(translated)
    tested_set = set(tested_pcs)
    tested_translated = tested_set <= translated_set if tested_set else False
    tested_executed = tested_set <= set(executed) if tested_set else False
    tested_seen = tested_set <= (translated_set | set(executed)) if tested_set else False
    path_details = (
        {**trace_details, "granularity": "trace-failed"}
        if trace_failed
        else trace_details
    )
    path_identity = _path_identity_alias(
        () if trace_failed else executed,
        path_details,
        require_instruction_trace=True,
    )
    path_executed = path_identity.get("executed_pc")
    if isinstance(path_executed, dict):
        path_executed["tested_pc_seen"] = tested_seen
    details = {
        **trace_details,
        "trace_failed": trace_failed,
        **({"host_abort": trace_host_abort} if trace_host_abort is not None else {}),
        **({"host_abort_path_observed": host_abort_path_observed}
           if trace_host_abort is not None else {}),
        **({"host_abort_pc": hex(host_abort_pc)} if host_abort_pc is not None else {}),
        "path_identity": path_identity,
    }
    return {
        "backend": backend,
        "expected_path": "tcg",
        "tested_pc_seen": tested_seen,
        "tested_pc_translated": tested_translated,
        "tested_pc_executed": tested_executed,
        "translated_count": len(translated_set),
        "execution_count": len(executed),
        "details": {
            "source": "qemu-user-log",
            "trace_mechanism": "qemu -d exec,nochain",
            "tested_pc_count": len(tested_set),
            "tested_translated_count": len(tested_set & translated_set),
            "tested_executed_count": len(tested_set & set(executed)),
            "tested_pc_execution_inferred": False,
            "tested_pcs": [hex(pc) for pc in sorted(tested_set)],
            **details,
        },
    }


def _guest_exit_observed(raw_stderr: str) -> bool:
    return bool(re.search(r"\b(?:exit_group|exit)\s*\(", raw_stderr))


def _trap_signal_state(raw_stderr: str) -> dict[str, object] | None:
    """把 QEMU 的 guest signal / host abort 折叠成字段化 trap 事件。

    QEMU linux-user 对 guest 未处理信号打印 ``uncaught target signal``；
    对 guest 异常无法投递时打印 ``unhandled CPU exception``（host abort，
    D-446 的 misaligned sc.w 即此路径）；QEMU 自身崩溃打印
    ``QEMU internal SIG... {addr=...}``。三类必须分开记账：guest signal
    是 guest trap 交付，host abort 是 emulator 自身失败。
    """
    internal = _QEMU_INTERNAL_SIGNAL_RE.search(raw_stderr)
    if internal is not None:
        try:
            addr = _pc_int(internal.group("addr"))
        except ValueError:
            addr = None
        state: dict[str, object] = {
            "signal": None,
            "signal_code": None,
            "host_abort": "qemu-internal-signal",
        }
        if addr is not None:
            state["fault_address"] = hex(addr)
        return state
    host_abort = _QEMU_HOST_ABORT_RE.search(raw_stderr)
    if host_abort is not None:
        raw_cause = host_abort.group("cause")
        try:
            cause = int(raw_cause, 16 if raw_cause.lower().startswith("0x") else 10)
        except ValueError:
            cause = None
        return {
            "signal": None,
            "signal_code": None,
            "host_abort": "unhandled-cpu-exception",
            **({"qemu_exception_cause": cause} if cause is not None else {}),
        }
    match = _QEMU_TARGET_SIGNAL_RE.search(raw_stderr)
    if match is None:
        return None
    signal_number = int(match.group("code"))
    raw_name = match.group("name").strip()
    signal_name = (
        raw_name.upper()
        if raw_name.upper().startswith("SIG")
        else _SIGNAL_NAME_BY_TEXT.get(raw_name.lower())
    )
    if signal_name is None:
        try:
            signal_name = signal_module.Signals(signal_number).name
        except (TypeError, ValueError):
            signal_name = None
    if signal_name is None or (
        signal_name not in _SIGNAL_NUMBER_BY_NAME
        and getattr(signal_module, signal_name, None) is None
    ):
        return None
    siginfo = _strace_siginfo_state(raw_stderr, signal_name)
    user_signal = (
        siginfo is not None
        and siginfo.get("signal") == signal_name
        and isinstance(siginfo.get("si_code"), int)
        and siginfo["si_code"] <= 0
    )
    trap_cause = None if user_signal else _TRAP_CAUSE_BY_SIGNAL.get(signal_name) if signal_name else None
    state = {
        "signal": signal_name,
        "signal_code": signal_number,
        "qemu_uncaught_target_signal": not user_signal,
    }
    if not user_signal:
        state.update(guest_trap="delivered", trap_observer="qemu-linux-user-signal")
    if trap_cause is not None:
        state["trap_cause"] = trap_cause
        state["guest_cause"] = trap_cause
    return state


def _trace_process_signal_state(returncode: int) -> dict[str, object] | None:
    if returncode >= 0:
        return None
    try:
        signal_name = signal_module.Signals(-returncode).name
    except (TypeError, ValueError):
        return None
    return {
        "signal": signal_name,
        "signal_code": -returncode,
        "host_abort": f"qemu-process-signal:{signal_name}",
    }


def _expected_trap_sigill_state(
    trace_termination: object, executed: tuple[int, ...],
    main_trap_state: object, elf: str, *, expected_trap: bool,
    risk_pcs: tuple[int, ...] = (),
) -> dict[str, object] | None:
    if not (
        expected_trap
        and isinstance(trace_termination, dict)
        and trace_termination.get("host_abort") == "qemu-process-signal:SIGILL"
        and executed
        and isinstance(main_trap_state, dict)
        and main_trap_state.get("signal") == "SIGILL"
        and main_trap_state.get("signal_code") == _SIGNAL_NUMBER_BY_NAME["SIGILL"]
        and main_trap_state.get("guest_trap") == "delivered"
        and main_trap_state.get("guest_cause") == 2
        and main_trap_state.get("trap_cause") == 2
        and main_trap_state.get("host_abort") is None
    ):
        return None
    fault_pc = executed[-1]
    if risk_pcs and fault_pc not in risk_pcs:
        return None
    main_epcs = []
    try:
        for field in ("guest_epc", "trap.epc"):
            if field in main_trap_state:
                main_epcs.append(_pc_int(main_trap_state[field]))
    except (TypeError, ValueError):
        return None
    if not main_epcs or any(pc != fault_pc for pc in main_epcs):
        return None
    tval = _instruction_word_at_pc(elf, fault_pc)
    state: dict[str, object] = {
        "signal": "SIGILL",
        "signal_code": 4,
        "guest_trap": "delivered",
        "guest_cause": 2,
        "trap_cause": 2,
        "guest_epc": fault_pc,
        "trap.epc": fault_pc,
        "guest_tval_observer": "qemu-exec-trace" if tval is not None else "unavailable",
        "trap_observer": "qemu-process-signal+qemu-exec-trace",
        "qemu_uncaught_target_signal": True,
        "expected_trap_process_signal": "qemu-process-signal:SIGILL",
    }
    if tval is not None:
        state["guest_tval"] = tval
        state["trap.tval"] = tval
    return state


def _trap_state_from_siginfo(
    returncode: int, siginfo: dict[str, object] | None, elf: str,
    fault_pc: int | None = None,
) -> dict[str, object] | None:
    if returncode >= 0 or not isinstance(siginfo, dict):
        return None
    signal_name = siginfo.get("signal")
    signal_number = _SIGNAL_NUMBER_BY_NAME.get(signal_name)
    if signal_number is None or returncode != -signal_number:
        return None
    if siginfo.get("si_code", 0) <= 0:
        return None
    valid_siginfo_codes = {
        "SIGILL": {1, 2},  # ILL_ILLOPC / ILL_ILLOPN
        "SIGSEGV": {1, 2},  # SEGV_MAPERR / SEGV_ACCERR
        "SIGBUS": {1},  # BUS_ADRALN; other SIGBUS codes are not misalignment
    }
    if signal_name in valid_siginfo_codes and (
        type(siginfo.get("si_code")) is not int
        or siginfo["si_code"] not in valid_siginfo_codes[signal_name]
    ):
        return None
    if signal_name in {"SIGSEGV", "SIGBUS"}:
        if fault_pc is None:
            return None
        instruction_word = _instruction_word_at_pc(elf, fault_pc)
        cause = _memory_trap_cause(signal_name, instruction_word)
        if cause is None:
            return None
        if cause in {4, 6, 13, 15}:
            try:
                tval = _pc_int(siginfo["si_addr"])
            except (KeyError, TypeError, ValueError):
                tval = None
            tval_observer = "qemu-strace-si_addr"
        else:
            tval = instruction_word
            tval_observer = "qemu-elf-code-at-fault-pc"
    else:
        cause = _TRAP_CAUSE_BY_SIGNAL.get(signal_name)
        if cause is None:
            return None
    state: dict[str, object] = {
        "signal": signal_name,
        "signal_code": signal_number,
        "guest_trap": "delivered",
        "guest_cause": cause,
        "trap_cause": cause,
        "trap_observer": "qemu-strace-si_addr+elf",
        "qemu_uncaught_target_signal": True,
    }
    if signal_name in {"SIGSEGV", "SIGBUS"}:
        state.update({
            "guest_epc": fault_pc,
            "trap.epc": fault_pc,
            "fault_address": siginfo.get("si_addr"),
            "trap_observer": "qemu-strace-si_addr+qemu-exec-trace",
        })
        if tval is not None:
            state.update({
                "guest_tval": tval,
                "guest_tval_observer": tval_observer,
                "trap.tval": tval,
            })
        return state
    try:
        fault_pc = _pc_int(siginfo["si_addr"])
    except (KeyError, TypeError, ValueError):
        fault_pc = None
    if fault_pc is None:
        return state
    state.update({"guest_epc": fault_pc, "trap.epc": fault_pc})
    tval = _instruction_word_at_pc(elf, fault_pc)
    if tval is not None:
        state.update({
            "guest_tval": tval,
            "guest_tval_observer": "qemu-elf-code-at-si_addr",
            "trap.tval": tval,
        })
    return state


def _attach_trace_observation(
    observation_state: dict[str, object],
    executed: tuple[int, ...],
    trace_details: dict[str, object],
) -> None:
    if not executed:
        return
    extra_state = observation_state.setdefault("extra_state", {})
    if (
        extra_state.get("guest_trap") != "delivered"
        and extra_state.get("host_abort") is None
        and extra_state.get("qemu_uncaught_target_signal") is not True
    ):
        return
    if trace_details.get("granularity") == "per-instruction-exec":
        observation_state.setdefault("fault_pc", hex(executed[-1]))
        if extra_state.get("guest_trap") == "delivered":
            extra_state.setdefault("guest_epc", executed[-1])
            extra_state.setdefault("trap.epc", executed[-1])
        extra_state.setdefault("fault_pc_source", "qemu-exec-trace-last-pc")
    else:
        extra_state.setdefault(
            "fault_pc_observer_gap",
            f"trace-granularity:{trace_details.get('granularity', 'missing')}",
        )


def _strace_siginfo_state(
    raw_stderr: str, expected_signal: str | None = None,
) -> dict[str, object] | None:
    """从 QEMU ``-strace`` 的 siginfo 行提取 si_code/si_addr。

    ``uncaught target signal`` 只有信号编号；si_code（如 SEGV_ACCERR=2）
    与 fault address 只出现在 ``-strace`` 的 ``--- SIGXXX {si_signo=...} ---``
    行里。两类证据必须能对得上：siginfo 的 si_signo 即该次投递的信号。
    仅当信号名匹配时合并，避免把 host stderr 的其它 ``SIG...`` 行误判。
    """
    siginfo: dict[str, object] | None = None
    for line in raw_stderr.splitlines():
        match = _QEMU_STRACE_SIGNAL_RE.match(line.strip())
        if match is None:
            continue
        sig_name = match.group("sig").upper()
        if expected_signal is not None and sig_name != expected_signal.upper():
            continue
        try:
            raw_signo = match.group("signo")
            if raw_signo.upper().startswith("SIG"):
                if raw_signo.upper() != sig_name:
                    continue
            elif sig_name in _SIGNAL_NUMBER_BY_NAME and int(raw_signo) != _SIGNAL_NUMBER_BY_NAME[sig_name]:
                continue
            raw_code = match.group("code")
            numeric_code = raw_code.lstrip("-").isdigit()
            code = (
                int(raw_code)
                if numeric_code
                else _SI_CODE_BY_NAME.get(raw_code.upper())
            )
            if code is None:
                continue
            raw_addr = match.group("addr")
            addr = 0 if raw_addr.upper() == "NULL" else _pc_int(raw_addr)
        except ValueError:
            continue
        raw_siginfo = {
            "signal": sig_name,
            "si_code": code,
            "si_code_name": (
                raw_code.upper()
                if not numeric_code
                else _SI_CODE_NAME_BY_SIGNAL.get(sig_name, {}).get(code)
            ),
            "si_addr": hex(addr) if addr is not None else None,
        }
        if siginfo is None:
            siginfo = raw_siginfo
    return siginfo


def _siginfo_fault_address(
    signal_name: object, siginfo: dict[str, object] | None,
) -> object | None:
    return (
        siginfo.get("si_addr")
        if signal_name in {"SIGSEGV", "SIGBUS"}
        and siginfo is not None
        and isinstance(siginfo.get("si_code"), int)
        and siginfo["si_code"] > 0
        else None
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    qemu_override = None
    backend = "qemu-riscv64"
    while len(args) >= 2 and args[0] in {"--backend", "--qemu-bin"}:
        if args[0] == "--backend":
            backend = args[1]
        else:
            qemu_override = args[1]
        args = args[2:]
    if args in (["-h"], ["--help"]):
        print("usage: qemu_trace_runner.py [--backend <qemu-riscv32|qemu-riscv64>] [--qemu-bin <path>] <riscv-elf>")
        return 0
    if len(args) != 1:
        return _fail("usage: qemu_trace_runner.py [--backend <qemu-riscv32|qemu-riscv64>] [--qemu-bin <path>] <riscv-elf>")
    if backend not in {"qemu-riscv32", "qemu-riscv64"}:
        return _fail(f"unsupported qemu backend: {backend}")
    try:
        volatile_gpr_indices = tuple(sorted(
            index for index in {
                int(item) for item in os.environ.get("RV_VOLATILE_GPR_INDICES", "").split(",")
                if item
            }
            if 0 <= index < 32
        ))
    except ValueError:
        volatile_gpr_indices = ()
    tested_pcs = _configured_tested_pcs()
    guest_elf = str(Path(args[0]).resolve())
    configured_run_root = os.environ.get("RQ1_RUN")
    trace_root = (
        Path(configured_run_root).resolve()
        if configured_run_root and Path(configured_run_root).is_dir()
        else Path(guest_elf).parent
    )
    try:
        require_execution_plane()
        validate_elf_machine(Path(guest_elf), EM_RISCV, "guest")
        if qemu_override:
            binary = resolve_x86_64_runner(qemu_override, backend)
        else:
            manifest = resolve_current_container_source_verified_manifest(
                expected_backend=backend
            )
            binary = str(
                load_target_binary_manifest(
                    manifest,
                    expected_backend=backend,
                    require_source_verified=True,
                ).binary_path
            ) if manifest is not None else resolve_x86_64_runner(backend, backend)
        qemu = [binary]
        guest_binary_sha256 = sha256_file(Path(guest_elf))
        # The Windows bind mount is readable but not reliably executable-mappable
        # for source-built QEMU.  Copy the immutable guest into the container
        # tmpfs; its bytes and identity remain those of the original ELF.
        elf = _stage_guest_elf(guest_elf)
    except (ExecutionEnvironmentError, OSError, ValueError) as exc:
        return _fail(str(exc))
    configured_cpu = (os.environ.get("RV_QEMU_CPU") or "").strip() or None
    cpu_profile = configured_cpu
    isa_profile = (
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE")
        or ""
    ).strip() or None
    expected_trap = os.environ.get("RV_TESTCASE_EXPECTED_TRAP") == "1"
    disabled_extensions = tuple(
        item.strip().lower()
        for item in os.environ.get("RV_TESTCASE_DISABLED_EXTENSIONS", "").split(",")
        if item.strip()
    )
    vector_required = any(
        token == "v" or token.startswith(("zv", "zve", "zvl"))
        for token in enabled_extensions(isa_profile or "")
    )
    if cpu_profile is None and isa_profile:
        cpu_profile = _qemu_cpu_for_isa(isa_profile, disabled_extensions)
    target_binary_details = _target_binary_details(qemu[0], expected_backend=backend)
    tool_version = (
        f"qemu-source-{target_binary_details['target_source_commit']}"
        if target_binary_details.get("target_identity_status") == "verified"
        else _binary_tool_version(qemu[0])
    )
    target_configuration = _shared_target_configuration(
        {
            "isa_profile": isa_profile,
            "cpu_profile": cpu_profile or "qemu-default",
            "observation_mode": "full",
            "guest_binary_sha256": guest_binary_sha256,
            "guest_execution_mode": "container-tmpfs-copy",
            "cpu_profile_source": (
                "RV_QEMU_CPU" if configured_cpu
                else "isa-profile-derived" if cpu_profile
                else "qemu-default"
            ),
        },
        target_binary_details,
    )
    try:
        qemu_command = [*qemu, *(('-cpu', cpu_profile) if cpu_profile else ())]
        # -strace 让 linux-user 打印 siginfo（si_code/si_addr），这是
        # D-445（cbo.zero fault address）与 D-446（guest signal vs host
        # abort）观察的必需输入；trace/FP 两趟不需要，避免 stderr 膨胀。
        main_command = [*qemu_command, "-strace", elf]
        main_proc = _execute(main_command, CHILD_TIMEOUT_SEC)
        translated, executed, trace_details, trace_run, state_trace = _trace_result(
            qemu_command, elf, backend, trace_root)
        if state_trace and not volatile_gpr_indices:
            volatile_gpr_indices = (2,)
        main_stderr = main_proc.stderr.decode("utf-8", errors="replace")
        terminal_ebreak = trace_details.get("terminal_ebreak") is True
        siginfo = _strace_siginfo_state(main_stderr)
        guest_trap = _trap_signal_state(main_stderr) is not None or (
            isinstance(siginfo, dict)
            and isinstance(siginfo.get("si_code"), int)
            and siginfo["si_code"] > 0
        )
        fp_required = os.environ.get("RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "0") == "1"
        final_state_required = (
            not expected_trap
            and (
                fp_required
                or vector_required
            )
        )
        if (guest_trap and not terminal_ebreak) or not final_state_required:
            # 没有声明 FP/RVV 终态的路线只需要 trace + trap/exit 观察；
            # 再启动一趟 QEMU GDB stub 会把一个标量 case 的耗时放大到
            # CHILD_TIMEOUT，甚至在 exit_checkpoint 前单步百万次。
            fp_state, fp_state_error = None, None
        else:
            with _PORT_STARTUP_LOCK:
                fp_port = free_port()
                fp_state, fp_state_error = collect_final_fp_state(
                    [*qemu_command, "-g", str(fp_port), elf],
                    port=fp_port,
                    elf_path=elf,
                    timeout_sec=FINAL_STATE_TIMEOUT_SEC,
                )
                if vector_required and (
                    not isinstance(fp_state, dict)
                    or not {
                        "vector.vl", "vector.vtype", "vector.vstart",
                        "vector.vlenb", "vector.registers",
                    } <= fp_state.keys()
                ):
                    fp_state = None
                    fp_state_error = fp_state_error or "gdb-vector-register-layout-incomplete"
                if fp_required and (
                    not isinstance(fp_state, dict)
                    or "fpr_rawbits" not in fp_state
                ):
                    fp_state = None
                    fp_state_error = fp_state_error or "gdb-fp-register-layout-incomplete"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _fail(f"qemu trace runner failed: {exc}")

    sys.stdout.buffer.write(main_proc.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(main_proc.stderr)
    if main_proc.stderr and not main_proc.stderr.endswith(b"\n"):
        sys.stderr.write("\n")
    stderr_text = main_proc.stderr.decode("utf-8", errors="replace") if main_proc.stderr else ""
    guest_exit = main_proc.returncode > 0 and _guest_exit_observed(stderr_text)
    trap_state = _trap_signal_state(stderr_text)
    siginfo_state = _strace_siginfo_state(
        stderr_text,
        trap_state.get("signal") if trap_state is not None else None,
    )
    siginfo_trap_state = _trap_state_from_siginfo(
        main_proc.returncode, siginfo_state, elf,
        executed[-1] if executed else None,
    )
    if siginfo_trap_state is not None and not (
        isinstance(trap_state, dict) and trap_state.get("host_abort") is not None
    ):
        trap_state = {**(trap_state or {}), **siginfo_trap_state}
    trace_termination = trace_details.get("trace_termination")
    expected_state = _expected_trap_sigill_state(
        trace_termination, executed, trap_state, elf,
        expected_trap=expected_trap,
        risk_pcs=tested_pcs,
    )
    if expected_state is not None:
        trace_details["trace_termination"] = expected_state
        trap_state = {**expected_state, **(trap_state or {})}
    if (
        terminal_ebreak
        and main_proc.returncode == -_SIGNAL_NUMBER_BY_NAME["SIGTRAP"]
        and isinstance(siginfo_state, dict)
        and siginfo_state.get("signal") == "SIGTRAP"
        and siginfo_state.get("si_code") == 1
    ):
        trap_state = {
            "signal": "SIGTRAP",
            "signal_code": 5,
            "guest_trap": "delivered",
            "guest_cause": 3,
            "trap_cause": 3,
            "guest_tval": 0,
            "guest_tval_observer": "qemu-linux-user-ebreak",
            "trap_observer": "qemu-linux-user-ebreak",
            "qemu_uncaught_target_signal": False,
        }
    if (
        isinstance(trap_state, dict)
        and trap_state.get("guest_trap") == "delivered"
        and executed
        and trace_details.get("trace_returncode") == main_proc.returncode
        and trace_details.get("trace_termination") is None
    ):
        trace_details["trace_termination"] = dict(trap_state)
    trace_details["main_guest_exit"] = guest_exit
    try:
        evidence = _evidence(
            backend,
            translated,
            executed,
            {
                **trace_details,
                **target_binary_details,
                "configuration_identity": target_configuration,
                "main_trap_signal": (
                    trap_state.get("signal") if isinstance(trap_state, dict) else None
                ),
            },
            tested_pcs,
        )
    except ValueError as exc:
        return _fail(str(exc))
    reported_executed = executed if not evidence["details"].get("trace_failed") else ()
    fp_observation_state = dict(fp_state) if fp_state is not None else {}
    if fp_state is None and fp_state_error is not None and (
        fp_required or vector_required
    ):
        fp_observation_state = {
            "fp_observer": (
                "not-observed"
                if fp_required
                else "not-required"
            ),
            "fp_observer_gap": f"qemu-final-fp-state-unavailable: {fp_state_error}",
            "adapter_execution_error": f"final-fp-state-unavailable: {fp_state_error}",
            **(
                {
                    "vector_observer_status": "gap",
                    "vector_observer_gap": fp_state_error,
                }
                if vector_required
                else {}
            ),
        }
    observation_state: dict[str, object] | None = None
    siginfo_fault_address: int | None = None
    if trap_state is not None:
        extra_state = dict(fp_observation_state)
        if siginfo_state is not None and trap_state.get("signal") == siginfo_state.get("signal"):
            # si_code 只在同一信号下有意义：siginfo 行信号必须与投递信号一致。
            extra_state["signal.si_code"] = siginfo_state["si_code"]
            extra_state["signal.si_code_observer"] = "qemu-strace"
            si_code_name = siginfo_state.get("si_code_name")
            if si_code_name is not None:
                extra_state["signal.si_code_name"] = si_code_name
                extra_state["signal.si_code_name_observer"] = "qemu-strace"
            if siginfo_state.get("si_addr") is not None:
                extra_state["signal.si_addr"] = siginfo_state["si_addr"]
                extra_state["signal.si_addr_observer"] = "qemu-strace"
                # 只有内存 fault 信号的 si_addr 才是 guest fault address；
                # SIGILL/SIGTRAP 的 si_addr 是指令位置，不能当作 tval。
                siginfo_fault_address = _siginfo_fault_address(
                    trap_state.get("signal"), siginfo_state
                )
        trap_cause = trap_state.get("trap_cause")
        if trap_cause is not None:
            extra_state["trap.cause"] = trap_cause
        extra_state.update({
            field: trap_state[field]
            for field in (
                "guest_trap", "guest_cause", "guest_epc", "trap_observer",
                "guest_tval", "guest_tval_observer",
                "qemu_exception_cause", "host_abort", "qemu_uncaught_target_signal",
                "expected_trap_process_signal",
            )
            if trap_state.get(field) is not None
        })
        signal = trap_state.get("signal")
        signal_code = trap_state.get("signal_code")
        observation_state = {
            "signal": signal,
            "signal_code": signal_code,
            "extra_state": extra_state,
        }
        if trap_state.get("guest_trap") == "delivered" and not terminal_ebreak:
            observation_state["fault"] = True
        if (
            trap_state.get("guest_trap") == "delivered"
            and trap_state.get("signal") in {"SIGILL", "SIGTRAP"}
            and siginfo_state is not None
            and siginfo_state.get("si_code", 0) > 0
            and siginfo_state.get("si_addr") is not None
        ):
            fault_pc = _pc_int(siginfo_state["si_addr"])
            observation_state["fault_pc"] = fault_pc
            extra_state["guest_epc"] = fault_pc
            extra_state["trap.epc"] = fault_pc
            extra_state["fault_pc_source"] = "qemu-strace-si_addr"
        if trap_state.get("guest_trap") == "delivered" and reported_executed:
            fault_pc = reported_executed[-1]
            observation_state.setdefault("fault_pc", fault_pc)
            extra_state.setdefault("guest_epc", fault_pc)
            extra_state.setdefault("trap.epc", fault_pc)
            extra_state.setdefault("fault_pc_source", "qemu-exec-trace-last-pc")
        if siginfo_fault_address is not None:
            observation_state["fault_address"] = siginfo_fault_address
        fault_address = trap_state.get("fault_address")
        if fault_address is not None:
            observation_state["fault_address"] = fault_address
        # fault_pc = the last executed guest PC when the trace exists; the
        # runner never synthesizes a PC from an empty/partial trace.
        if reported_executed and signal is None and fault_address is None:
            observation_state["fault_pc"] = hex(reported_executed[-1])
    elif siginfo_state is not None:
        # guest 自行捕获信号时（D-445 的 repro 安装 SA_SIGINFO handler），
        # QEMU 不打印 uncaught target signal；-strace 的 siginfo 行是唯一
        # 信号观察来源，必须独立成观察。
        signal_name = siginfo_state["signal"]
        extra_state = dict(fp_observation_state)
        extra_state["signal.si_code"] = siginfo_state["si_code"]
        extra_state["signal.si_code_observer"] = "qemu-strace"
        if siginfo_state.get("si_code_name") is not None:
            extra_state["signal.si_code_name"] = siginfo_state["si_code_name"]
            extra_state["signal.si_code_name_observer"] = "qemu-strace"
        extra_state["signal.si_addr"] = siginfo_state["si_addr"]
        extra_state["signal.si_addr_observer"] = "qemu-strace"
        observation_state = {
            "signal": signal_name,
            "signal_code": _SIGNAL_NUMBER_BY_NAME.get(signal_name),
            "extra_state": extra_state,
        }
        fault_address = _siginfo_fault_address(signal_name, siginfo_state)
        if fault_address is not None:
            observation_state["fault_address"] = fault_address
    elif fp_observation_state:
        observation_state = {"extra_state": fp_observation_state}
    if state_trace:
        observation_state = observation_state or {"extra_state": {}}
        observation_state.setdefault("extra_state", {})["state_trace"] = list(state_trace)
    if terminal_ebreak and observation_state is not None:
        # ebreak is the harness checkpoint, not an OS signal.  Keep the
        # guest-trap witness in extra_state while returning a normal exit.
        observation_state["signal"] = None
        observation_state["signal_code"] = None
    if observation_state is not None:
        _attach_trace_observation(observation_state, reported_executed, trace_details)
    if (
        guest_exit
        and trap_state is None
        and siginfo_state is None
        and (translated or reported_executed)
    ):
        observation_state = observation_state or {"extra_state": {}}
        observation_state.setdefault("extra_state", {})[
            "guest_exit_observer"
        ] = "qemu-linux-user"
    trace_host_abort = evidence["details"].get("host_abort")
    if trace_host_abort is not None:
        observation_state = observation_state or {"extra_state": {}}
        observation_state.setdefault("extra_state", {}).update({
            "host_abort": trace_host_abort,
            "host_abort_observer": "qemu-trace-process",
        })
    if observation_state is not None:
        extra_state = observation_state.setdefault("extra_state", {})
        if volatile_gpr_indices:
            extra_state["volatile_gpr_indices"] = list(volatile_gpr_indices)
        extra_state["observer_fields"] = observed_state_fields(extra_state)
        sys.stderr.write(
            "RV_OBSERVATION_STATE="
            + json.dumps(observation_state, separators=(",", ":"))
            + "\n"
        )
    elif fp_state_error is not None:
        sys.stderr.write(f"RV_FINAL_FP_STATE_STATUS=unavailable reason={fp_state_error}\n")
    if reported_executed:
        sys.stderr.write(f"RV_EXECUTED_PCS={json.dumps([hex(pc) for pc in reported_executed], separators=(',', ':'))}\n")
        sys.stderr.write(f"RV_INSTRUCTION_COUNT={len(reported_executed)}\n")
        sys.stderr.write(f"RV_TOTAL_GUEST_INSTRUCTION_COUNT={len(reported_executed)}\n")
    if tool_version is not None:
        sys.stderr.write(f"RV_TOOL_VERSION={tool_version}\n")
    sys.stderr.write(
            "RV_TRANSLATION_EVIDENCE="
            + json.dumps(
                evidence,
            separators=(",", ":"),
        )
        + "\n"
    )
    trace_markers = ""
    if reported_executed:
        trace_markers = (
            "\nRV_EXECUTED_PCS="
            + json.dumps([hex(pc) for pc in reported_executed], separators=(",", ":"))
            + f"\nRV_INSTRUCTION_COUNT={len(reported_executed)}"
            + f"\nRV_TOTAL_GUEST_INSTRUCTION_COUNT={len(reported_executed)}\n"
        )
    observation_marker = (
        "\nRV_OBSERVATION_STATE="
        + json.dumps(observation_state, separators=(",", ":"))
        if observation_state is not None else ""
    )
    trace_stderr = trace_run["stderr"] + trace_markers + observation_marker
    count_stderr = main_proc.stderr.decode("utf-8", errors="replace") + observation_marker
    execution_exit_code = 0 if terminal_ebreak else main_proc.returncode
    sys.stderr.write(
        "RV_EXECUTION_EVIDENCE="
        + json.dumps(
            {
                "schema_version": "runner-execution-evidence-v1",
                "main_command_fingerprint": canonical_digest(main_command),
                "trace_command_fingerprint": trace_run["command_fingerprint"],
                "count_command_fingerprint": canonical_digest(main_command),
                "trace_run": {
                    "exit_code": trace_run["exit_code"],
                    "stdout_hex": trace_run["stdout_hex"],
                    "stderr": trace_stderr,
                },
                "count_run": {
                    "exit_code": execution_exit_code,
                    "stdout_hex": main_proc.stdout.hex(),
                    "stderr": count_stderr,
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stderr.flush()
    return execution_exit_code


if __name__ == "__main__":
    raise SystemExit(main())

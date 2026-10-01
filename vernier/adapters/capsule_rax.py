import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.adapters.contracts import (
    PATH_IDENTITY_WITNESS_CONTRACT,
    PATH_IDENTITY_WITNESS_STATUS_UNAVAILABLE,
    backend_spec,
    profile_requires_fp_state,
)
from framework.capsule_csr_facts import ADDRESS_FAULT_MCAUSE as _ADDRESS_FAULT_CAUSES
from framework.adapters.runner import (
    _empty_fp_state,
    _observation_frame,
    canonical_trap_state,
    observed_state_fields,
)
from framework.adapters.trace_common import (
    _repeated_pc_cycle, _target_binary_details, _target_configuration,
)
from framework.direct_case import (
    INLINE_EXECUTED_PC_LIMIT, OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC,
)
from framework.direct_elf import symbol_offsets_from_elf
from framework._util import (
    elf_load_segments,
    git_head,
    git_worktree_clean,
    is_riscv_elf,
    pc_digest,
    pc_path,
    sha256_file,
    runner_contract_gap,
)
from framework.execution_identity import is_allowlisted_patched_identity


# RAX's compatibility build exposes the shared RVOBS1 frame at VMM shutdown;
# the upstream executed-PC stream remains unavailable.
RAX_SOURCE_COMMIT = "d7ce788225d9f0439d59d14b9b173b991388d7cf"
RAX_MEMORY = "128M"
RAX_TIMEOUT_SEC = 120
_RAX_TRAP_RE = re.compile(
    r"(?:gsc\s+)?riscv trap:\s+cause=(?P<cause>0x[0-9a-f]+|[0-9]+)\s+"
    r"tval=(?P<tval>0x[0-9a-f]+|[0-9]+)\s+"
    r"pc=(?P<pc>0x[0-9a-f]+|[0-9]+)",
    re.IGNORECASE,
)
_RAX_SHUTDOWN_RE = re.compile(
    r"vCPU shutdown(?: \(unsupported architecture\))?",
    re.IGNORECASE,
)
_RAX_CONFIG_CONTRACT = "rax-execution-config-v1"
_RAX_TRACE_RE = re.compile(rb"(?m)^\s*INS\s+0x([0-9a-fA-F]+)\s+")
_RAX_FP_STATE_RE = re.compile(
    rb"^RQ1_FPSTATE=fpr:(?P<fpr>[0-9a-fA-F]{16}(?:,[0-9a-fA-F]{16}){31})"
    rb";fcsr:(?P<fcsr>[0-9a-fA-F]{8})$"
)


def _rvobs1_frame(stdout: bytes) -> bytes | None:
    """Decode either a raw frame or RAX's ``RQ1_RVOBS1=`` transport line."""
    frame = _observation_frame(stdout)
    encoded = re.findall(rb"(?m)^RQ1_RVOBS1=([0-9a-fA-F]+)\s*$", stdout)
    if encoded:
        if len(encoded) != 1:
            raise RuntimeError("RAX emitted conflicting RVOBS1 frames")
        try:
            decoded = bytes.fromhex(encoded[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("RAX RVOBS1 transport is not valid hex") from exc
        transported = _observation_frame(decoded)
        if transported is None:
            raise RuntimeError("RAX RVOBS1 transport frame is malformed")
        if frame is not None and frame != transported:
            raise RuntimeError("RAX emitted conflicting raw and transported RVOBS1 frames")
        frame = transported
    return frame


def _parse_final_fp_state(stdout: bytes) -> dict[str, object] | None:
    lines = [line for line in stdout.splitlines() if line.startswith(b"RQ1_FPSTATE=")]
    if not lines:
        return None
    if len(lines) != 1:
        raise RuntimeError("RAX emitted conflicting FP state records")
    match = _RAX_FP_STATE_RE.fullmatch(lines[0])
    if match is None:
        raise RuntimeError("RAX FP state record is malformed")
    rawbits = [int(value, 16) for value in match.group("fpr").decode("ascii").split(",")]
    fcsr = int(match.group("fcsr"), 16)
    if fcsr > 0xFF:
        raise RuntimeError(f"RAX FCSR value is out of range: {fcsr:#x}")
    return {
        "fpr_rawbits": rawbits,
        "fflags": fcsr & 0x1F,
        "frm": (fcsr >> 5) & 0x7,
    }


def _parse_guest_trap(stderr: bytes) -> dict[str, object] | None:
    """Parse RAX's existing guest-trap diagnostic into structured state.

    The emulator writes this line before returning a non-zero status.  It is
    the only trap channel currently available from the RAX CLI; malformed or
    conflicting lines are rejected instead of being guessed into a clean run.
    """

    parsed = []
    for match in _RAX_TRAP_RE.finditer(stderr.decode("utf-8", errors="replace")):
        values = tuple(
            int(raw, 16 if raw.lower().startswith("0x") else 10)
            for raw in (match.group(name) for name in ("cause", "tval", "pc"))
        )
        if any(value < 0 or value > ((1 << 64) - 1) for value in values):
            raise RuntimeError("RAX guest-trap field is out of range")
        if values[2] & 1:
            raise RuntimeError("RAX guest-trap PC is unaligned")
        parsed.append(values)
    if not parsed:
        return None
    if len(set(parsed)) != 1:
        raise RuntimeError("RAX emitted conflicting guest-trap diagnostics")
    cause, tval, pc = parsed[0]
    return {
        "guest_trap": "delivered",
        "guest_cause": cause,
        "guest_epc": pc,
        "guest_tval": tval,
        "guest_tval_observer": "rax-cli-stderr",
        "fault_pc": pc,
        "trap_observer": "rax-cli-stderr",
        **({"fault_address": tval} if cause in _ADDRESS_FAULT_CAUSES else {}),
    }


def _configuration_identity(
    *,
    source_commit: str,
    rax_sha256: str,
    capsule_sha256: str,
    observation_size: int,
    isa_profile: str = "",
    target_details: dict[str, object] | None = None,
    trace_available: bool = False,
    trace_gap: str | None = "upstream-riscv-pc-trace-unavailable",
) -> dict[str, object]:
    """Return the small, reproducible identity of one RAX execution setup."""

    isa_profile = str(isa_profile or "").strip().lower()
    payload = {
        "contract": _RAX_CONFIG_CONTRACT,
        "source_commit": source_commit,
        "rax_binary_sha256": rax_sha256,
        "target_binary_sha256": rax_sha256,
        "capsule_sha256": capsule_sha256,
        "arch": "riscv64",
        "isa_profile": isa_profile,
        "backend": "rax-riscv64",
        "memory": RAX_MEMORY,
        "observer_transport": "rax-rvobs1-vmm-mailbox",
        "observer_status": "observed",
        "observer_gap": trace_gap,
        "observation_size": observation_size,
        "pc_trace": "rax-sde-trace" if trace_available else "upstream-riscv-pc-trace-unavailable",
    }
    return _target_configuration(
        payload,
        target_details or {
            "target_binary_sha256": rax_sha256,
            "target_identity_status": "missing",
            "target_identity": None,
            "target_identity_digest": None,
        },
    )


def _path_identity_witness(pcs: tuple[int, ...] = ()) -> dict[str, object]:
    """Describe the guest-PC evidence emitted by the RAX trace side channel."""
    if pcs:
        sequence, digest = pc_path(pcs)
        return {
            "contract": PATH_IDENTITY_WITNESS_CONTRACT,
            "status": "observed",
            "digest": digest,
            "evidence": "rax-sde-trace",
            "executed_pc": {
                "observed": True,
                "tested_pc_seen": False,
                "count": len(pcs),
                "digest": digest,
            },
            "executed_pcs": sequence,
            "jit_path": {"observed": False, "evidence": "not-used-by-emulator-backend"},
        }
    return {
        "contract": PATH_IDENTITY_WITNESS_CONTRACT,
        "status": PATH_IDENTITY_WITNESS_STATUS_UNAVAILABLE,
        "executed_pc": {
            "observed": False,
            "tested_pc_seen": False,
            "evidence": "upstream-riscv-pc-trace-unavailable",
        },
        "jit_path": {"observed": False, "evidence": "not-used-by-emulator-backend"},
    }


def _read_guest_trace(path: Path) -> tuple[tuple[int, ...], str | None]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return (), f"rax-trace-read-error:{type(exc).__name__}"
    if not payload:
        return (), "rax-sde-trace-empty"
    pcs = tuple(int(match.group(1), 16) for match in _RAX_TRACE_RE.finditer(payload))
    if not pcs:
        return (), "rax-sde-trace-format-empty"
    if any(pc & 1 for pc in pcs):
        return (), "rax-sde-trace-unaligned-pc"
    return pcs, None


def _trace_pc_cycle_loop(
    path: Path | None, *, repeats: int = 32, max_period: int = 64,
    ignored_ranges: tuple[tuple[int, int], ...] = (),
) -> int | None:
    """Detect a stuck direct capsule before RAX's 120-second timeout."""
    if path is None or repeats < 1 or max_period < 1:
        return None
    try:
        size = path.stat().st_size
        if size <= 0:
            return None
        with path.open("rb") as stream:
            stream.seek(max(0, size - 65536))
            payload = stream.read()
    except OSError:
        return None
    pcs = tuple(int(match.group(1), 16) for match in _RAX_TRACE_RE.finditer(payload))
    repeated = _repeated_pc_cycle(pcs, repeats=repeats, max_period=max_period)
    if repeated is not None and any(
        start <= repeated < end for start, end in ignored_ranges
    ):
        return None
    return repeated


def _observer_copy_loop_ranges(elf: Path) -> tuple[tuple[int, int], ...]:
    """Return bounded RVOBS1 copy-helper ranges, when the capsule exposes them."""
    _offsets, symbols = symbol_offsets_from_elf(
        elf, ("rvobs_copy_loop", "rvobs_copy_done"), allow_before_start=True,
    )
    try:
        start = int(symbols["rvobs_copy_loop"], 16)
        end = int(symbols["rvobs_copy_done"], 16)
    except (KeyError, TypeError, ValueError):
        return ()
    return ((start, end),) if start < end else ()


def _terminate_rax_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the RAX process group, then fall back to the process handle."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass


def _run(
    elf: Path,
    rax: Path,
    source: Path,
    observation_size: int,
) -> tuple[bytes, dict[str, object], tuple[int, ...]]:
    if not elf.is_file():
        raise RuntimeError(f"missing capsule ELF: {elf}")
    if not is_riscv_elf(elf):
        raise ValueError("expected RISC-V ELF")
    elf_load_segments(elf)
    ignored_trace_ranges = _observer_copy_loop_ranges(elf)
    if not rax.is_file():
        raise RuntimeError(f"missing RAX binary: {rax}")
    source_commit = git_head(source)
    source_clean = git_worktree_clean(source)
    target_details = _target_binary_details(str(rax), expected_backend="rax-riscv64")
    target_identity = target_details.get("target_identity")
    target_attested_build = (
        target_details.get("target_identity_status") == "verified"
        and isinstance(target_identity, dict)
        and target_identity.get("source_commit") == RAX_SOURCE_COMMIT
        and (
            (
                not target_identity.get("patch_lineage")
                and target_identity.get("source_provenance")
                in {"verified-build", "verified-git-archive"}
            )
            or is_allowlisted_patched_identity(target_identity)
        )
    )
    if source_commit != RAX_SOURCE_COMMIT or (not source_clean and not target_attested_build):
        raise RuntimeError(
            "RAX source checkout is not pinned to "
            f"{RAX_SOURCE_COMMIT}: {source_commit or 'missing'}"
        )
    if observation_size < OBSERVATION_HEADER_SIZE:
        raise RuntimeError("RAX UART RVOBS1 frame is too small")
    profile = str(
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
    ).strip().lower()
    if not profile.startswith(("rv32", "rv64")):
        raise RuntimeError("RV_TESTCASE_ISA_PROFILE is required")

    child_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        child_env.pop(name, None)
    child_env.pop("RAX_MACHINE", None)
    for name in tuple(child_env):
        if name.startswith("RAX_GSC_"):
            child_env.pop(name)
    child_env.setdefault("RUST_LOG", "error")
    trace_path = None
    trace_value = child_env.get("RAX_TRACE_PATH")
    if isinstance(trace_value, str) and trace_value.strip():
        trace_path = Path(trace_value.replace("%p", str(os.getpid())))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(rax),
        "--arch",
        "riscv64",
        "--backend",
        "emulator",
        "--memory",
        RAX_MEMORY,
        "--kernel",
        str(elf),
    ]
    if trace_path is not None:
        command.extend(("--trace", str(trace_path)))
    if trace_path is None:
        # Keep the small non-trace caller contract synchronous.  Direct
        # coverage runs provide a trace path and use the live watchdog below.
        try:
            proc = subprocess.run(
                command,
                cwd=str(rax.parent),
                timeout=RAX_TIMEOUT_SEC,
                capture_output=True,
                check=False,
                env=child_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("RAX CLI timed out") from exc
    else:
        process = subprocess.Popen(
            command,
            cwd=str(rax.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=os.name == "posix",
        )
        trace_guard_stop = threading.Event()
        trace_guard_pc: list[int] = []

        def watch_trace_loop() -> None:
            while not trace_guard_stop.is_set() and process.poll() is None:
                repeated_pc = _trace_pc_cycle_loop(
                    trace_path, ignored_ranges=ignored_trace_ranges,
                )
                if repeated_pc is not None:
                    trace_guard_pc.append(repeated_pc)
                    _terminate_rax_process(process)
                    return
                trace_guard_stop.wait(0.05)

        trace_guard_thread = threading.Thread(
            target=watch_trace_loop,
            name="rq1-rax-pc-loop-watch",
            daemon=True,
        )
        trace_guard_thread.start()
        try:
            try:
                stdout, stderr = process.communicate(timeout=RAX_TIMEOUT_SEC)
            except subprocess.TimeoutExpired as exc:
                _terminate_rax_process(process)
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        process.kill()
                    stdout, stderr = process.communicate()
                raise RuntimeError("RAX CLI timed out") from exc
        finally:
            trace_guard_stop.set()
            trace_guard_thread.join(timeout=1)
        if trace_guard_pc:
            raise RuntimeError(f"RAX guest-pc-loop:0x{trace_guard_pc[0]:x}")
        proc = subprocess.CompletedProcess(
            command, process.returncode, stdout=stdout, stderr=stderr,
        )
    executed_pcs: tuple[int, ...] = ()
    trace_gap = "upstream-riscv-pc-trace-unavailable"
    trace_sha256 = None
    if trace_path is not None:
        executed_pcs, trace_gap = _read_guest_trace(trace_path)
        if trace_path.is_file():
            trace_sha256 = sha256_file(trace_path)
        if executed_pcs:
            trace_gap = None
    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    final_fp_state = _parse_final_fp_state(proc.stdout)
    tohost_values = re.findall(
        r"(?m)^\s*RQ1_TOHOST_VALUE\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*$",
        stdout_text,
    )
    if len({int(value, 0) for value in tohost_values}) > 1:
        raise RuntimeError("RAX emitted conflicting tohost values")
    tohost_value = int(tohost_values[0], 0) if tohost_values else None
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    tohost_failure = re.search(
        r"riscv tohost failure:\s+value=(0x[0-9a-fA-F]+|[0-9]+)",
        stderr_text,
        re.IGNORECASE,
    )
    if tohost_value is None and tohost_failure is not None:
        tohost_value = int(tohost_failure.group(1), 0)
    guest_trap = _parse_guest_trap(proc.stderr)
    shutdown_reported = (
        guest_trap is None
        and _RAX_SHUTDOWN_RE.search(
            proc.stderr.decode("utf-8", errors="replace")
        ) is not None
    )
    if guest_trap is not None and proc.returncode < 0:
        raise RuntimeError("RAX process was signal-terminated while reporting a guest trap")
    if guest_trap is not None and proc.returncode == 0:
        raise RuntimeError("RAX reported a guest trap with zero exit status")
    if proc.returncode != 0 and guest_trap is None and tohost_failure is None:
        raise RuntimeError(f"RAX CLI exited with {proc.returncode}: {stderr_text.strip()}")
    observation_frame = None
    if guest_trap is None and tohost_failure is None:
        observation_frame = _rvobs1_frame(proc.stdout)
        if observation_frame is None:
            if OBSERVATION_MAGIC in proc.stdout or b"RQ1_RVOBS1=" in proc.stdout:
                raise RuntimeError("RAX UART RVOBS1 frame is too small")
            raise RuntimeError(
                "upstream RAX RISC-V exposes neither RVOBS1 nor executed-PC evidence"
            )
        if observation_frame is not None:
            if len(observation_frame) < OBSERVATION_HEADER_SIZE:
                raise RuntimeError("RAX UART RVOBS1 frame is too small")
            memory_size = int.from_bytes(
                observation_frame[16 + 32 * 8:OBSERVATION_HEADER_SIZE], "little"
            )
            frame_size = OBSERVATION_HEADER_SIZE + memory_size
            if frame_size != observation_size or len(observation_frame) != frame_size:
                raise RuntimeError(
                    f"RAX UART RVOBS1 size mismatch: got {len(observation_frame)}, expected {observation_size}"
                )
    if observation_frame is not None:
        checkpoint_pc = int.from_bytes(observation_frame[8:16], "little")
        if checkpoint_pc & 1:
            raise RuntimeError("RAX UART RVOBS1 checkpoint is unaligned")
        if checkpoint_pc == 0:
            raise RuntimeError("RAX UART RVOBS1 checkpoint is not finalized")
    # The VMM prints a line-oriented transport record.  The common observation
    # parser consumes the canonical raw frame, so strip the transport wrapper
    # before returning it to the dispatcher's stdout.
    output = b"" if guest_trap is not None or tohost_failure is not None else observation_frame or b""
    rax_sha256 = sha256_file(rax)
    capsule_sha256 = sha256_file(elf)
    configuration = _configuration_identity(
        source_commit=source_commit,
        rax_sha256=rax_sha256,
        capsule_sha256=capsule_sha256,
        observation_size=observation_size,
        isa_profile=profile,
        target_details=target_details,
        trace_available=bool(executed_pcs),
        trace_gap=trace_gap,
    )
    return output, {
        "source": "rax-cli-emulator",
        "rax_source_commit": source_commit,
        "rax_source_clean": source_clean,
        "rax_source_provenance": (
            "target-identity-attested-patched"
            if target_attested_build and isinstance(target_identity, dict)
            and target_identity.get("patch_lineage")
            else "target-identity-attested" if target_attested_build
            else "clean-checkout"
        ),
        "rax_binary_sha256": rax_sha256,
        "capsule_sha256": capsule_sha256,
        "dependency_family": backend_spec("rax-riscv64").dependency_family,
        "arch": "riscv64",
        "backend": "rax-riscv64",
        **{
            key: target_details.get(key)
            for key in (
                "target_binary_sha256",
                "target_identity_status",
                "target_identity",
                "target_identity_digest",
            )
        },
        "memory": RAX_MEMORY,
        "observer_transport": "rax-rvobs1-vmm-mailbox",
        "observer_status": "observed",
        "observer_gap": trace_gap,
        **({"final_fp_state": final_fp_state} if final_fp_state is not None else {}),
        "tohost_value": tohost_value,
        "tohost_failure": tohost_failure is not None,
        **({
            "guest_status": "failure",
            "guest_exit_channel": "rax-vcpu-tohost",
            "termination": "guest-tohost-failure",
            "terminal_observed": True,
        } if tohost_failure is not None else {}),
        "trace_available": bool(executed_pcs),
        "trace_gap": trace_gap,
        "trace_complete": bool(executed_pcs) and trace_gap is None,
        "trace_mechanism": "rax-sde-trace" if trace_path is not None else None,
        "trace_records_total": len(executed_pcs),
        **({
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha256,
        } if trace_path is not None else {}),
        "path_identity": _path_identity_witness(executed_pcs),
        "configuration_identity": configuration,
        "execution_status": (
            "guest-trap" if guest_trap is not None
            else "guest-shutdown" if shutdown_reported
            else "guest-tohost-failure" if tohost_failure is not None
            else "completed"
        ),
        "exit_code": proc.returncode,
        "process_returncode": proc.returncode,
        **(guest_trap or {}),
    }, executed_pcs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a framework RV64 capsule through RAX")
    parser.add_argument("elf")
    parser.add_argument("--rax-bin", default=os.environ.get("RAX_BINARY"))
    parser.add_argument("--rax-source", default=os.environ.get("RAX_SOURCE"))
    parser.add_argument("--observation-size", default=os.environ.get("RV_CAPSULE_OBSERVATION_SIZE"), type=int)
    args = parser.parse_args(argv)
    if args.observation_size is None:
        return runner_contract_gap("RAX UART observer size is required")
    if not args.rax_bin or not args.rax_source:
        return runner_contract_gap("RAX upstream binary and source paths are required")
    try:
        stdout, details, executed_pcs = _run(
            Path(args.elf).resolve(),
            Path(args.rax_bin).resolve(),
            Path(args.rax_source).resolve(),
            args.observation_size,
        )
    except Exception as exc:
        return runner_contract_gap(f"RAX runner failed: {type(exc).__name__}: {exc}")
    final_fp_state = details.pop("final_fp_state", None)
    sys.stdout.buffer.write(stdout)
    sys.stdout.buffer.flush()
    sys.stderr.write(f"RV_TOOL_VERSION=rax-{RAX_SOURCE_COMMIT[:12]}\n")
    if details.get("tohost_value") is not None:
        sys.stderr.write(f"RQ1_TOHOST_VALUE={int(details['tohost_value'])}\n")
    if details.get("tohost_failure"):
        sys.stderr.write(
            f"riscv tohost failure: value=0x{int(details['tohost_value']):x}\n"
        )
    if executed_pcs:
        trace_reference = None
        trace_value = details.get("trace_path")
        if len(executed_pcs) > INLINE_EXECUTED_PC_LIMIT and isinstance(trace_value, str):
            trace = Path(trace_value)
            if trace.is_file():
                trace_reference = {
                    "format": "text-pc",
                    "absolute_path": str(trace.resolve()),
                    "pc_count": len(executed_pcs),
                    "pc_digest": pc_digest(executed_pcs),
                    "bytes": trace.stat().st_size,
                    "sha256": details.get("trace_sha256"),
                }
        if trace_reference is not None:
            sys.stderr.write(
                "RV_EXECUTED_PCS_FILE="
                + json.dumps(trace_reference, separators=(",", ":"))
                + "\n"
            )
        else:
            sys.stderr.write(
                "RV_EXECUTED_PCS="
                + json.dumps([hex(pc) for pc in executed_pcs], separators=(",", ":"))
                + "\n"
            )
        sys.stderr.write(f"RV_INSTRUCTION_COUNT={len(executed_pcs)}\n")
        sys.stderr.write(f"RV_TOTAL_GUEST_INSTRUCTION_COUNT={len(executed_pcs)}\n")
    profile = str(
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
    ).strip().lower()
    require_final_fp_state = os.environ.get("RV_TESTCASE_REQUIRE_FINAL_FP_STATE") == "1"
    if profile_requires_fp_state(profile) and require_final_fp_state:
        fp_state = (
            {
                **final_fp_state,
                "fp_observer": "observed",
                "mstatus_fs": None,
                "mstatus_fs_observer": "gap",
                "mstatus_fs_gap": "rax-uart-no-csr-readback",
            }
            if isinstance(final_fp_state, dict) else
            {
                "fpr_rawbits": None,
                "fflags": None,
                "frm": None,
                "fp_observer": "not-observed",
                "fp_observer_gap": "final-fp-state-unavailable: rax-uart-no-csr-readback",
                "mstatus_fs": None,
                "mstatus_fs_observer": "gap",
                "mstatus_fs_gap": "rax-uart-no-csr-readback",
            }
        )
    else:
        fp_state = {**_empty_fp_state(), "mstatus_fs_gap": None}
    trap_state = {
        key: details[key]
        for key in (
            "guest_trap",
            "guest_cause",
            "guest_epc",
            "guest_epc_observer",
            "guest_tval",
            "guest_tval_observer",
            "trap_observer",
            "fault_pc",
            "fault_address",
        )
        if key in details
    }
    observation_extra_state = {**fp_state, **trap_state}
    for name in (
        "tohost_value", "tohost_failure", "guest_status", "guest_exit_channel",
        "termination", "terminal_observed",
    ):
        if details.get(name) is not None:
            observation_extra_state[name] = details[name]
    observation_extra_state.update(canonical_trap_state(trap_state))
    if details.get("adapter_execution_error"):
        observation_extra_state["adapter_execution_error"] = details[
            "adapter_execution_error"
        ]
    observation_extra_state["observer_fields"] = observed_state_fields(
        observation_extra_state
    )
    sys.stderr.write(
        "RV_OBSERVATION_STATE="
        + json.dumps(
            {"extra_state": observation_extra_state, **trap_state},
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stderr.write(
        "RV_TRANSLATION_EVIDENCE="
        + json.dumps(
            {
                "backend": "rax-riscv64",
                "expected_path": "interpreter",
                "tested_pc_seen": False,
                "tested_pc_translated": False,
                "tested_pc_executed": False,
                "execution_count": len(executed_pcs),
                "details": {"tested_pc_count": 0, **details},
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return int(details.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())

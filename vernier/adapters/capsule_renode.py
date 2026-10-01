import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.adapters.contracts import (
    PATH_IDENTITY_WITNESS_CONTRACT,
    backend_spec,
    capsule_cpu_profile,
    profile_requires_fp_state,
)
from framework.capsule_csr_facts import ADDRESS_FAULT_MCAUSE as _ADDRESS_FAULT_CAUSES
from framework.adapters.rsp import RspClient, _PORT_STARTUP_LOCK, free_port
from framework.adapters.runner import _empty_fp_state, _read_mailbox, observed_state_fields
from framework.adapters.trace_common import (
    _repeated_pc_cycle, _target_binary_details, _target_configuration,
)
from framework.direct_case import (
    INLINE_EXECUTED_PC_LIMIT,
    OBSERVATION_HEADER_SIZE,
    OBSERVATION_MAGIC,
)
from framework.paths import LOCAL_ROOT
from framework._util import elf_load_segments, elf_executable_loads as _elf_executable_loads, git_head, git_worktree_clean, is_riscv_elf, pc_path, pc_path_in_ranges as _trace_pcs_in_loads, runner_contract_gap, sha256_file


DEFAULT_DOTNET = LOCAL_ROOT / "toolchains" / "dotnet-8" / "dotnet"
if not DEFAULT_DOTNET.is_file():
    DEFAULT_DOTNET = Path(os.environ.get("RQ1_DEPS", "/path/to/deps")) \
        / "toolchains" / "dotnet-8" / "dotnet"
DEFAULT_RENODE_HOME = LOCAL_ROOT / "external" / "binary-translation-tools" / "Renode" / "output" / "bin" / "Release"
DEFAULT_RENODE_DLL = DEFAULT_RENODE_HOME / "Renode.dll"
DEFAULT_RENODE_SOURCE = LOCAL_ROOT / "external" / "binary-translation-tools" / "Renode"
DEFAULT_TLIB_SOURCE = (
    LOCAL_ROOT
    / "external"
    / "binary-translation-tools"
    / "Renode"
    / "src"
    / "Infrastructure"
    / "src"
    / "Emulator"
    / "Cores"
    / "tlib"
)
_CONFIGURATION_IDENTITY_CONTRACT = "renode-riscv64-execution-config-v1"
def _elf_loads(path: Path) -> tuple[int, tuple[tuple[int, int], ...]]:
    entry, segments = elf_load_segments(path)
    return entry, tuple((virtual, memory_size) for virtual, _data, memory_size in segments)


def _path_identity_witness(
    pcs: tuple[int, ...], *, evidence: str = "Renode-CreateExecutionTracing-PC",
    trace_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the observed Renode path witness, or a gap."""
    if any(pc & 1 for pc in pcs):
        pcs = ()
    sequence, digest = pc_path(pcs)
    if not sequence:
        return {
            "contract": PATH_IDENTITY_WITNESS_CONTRACT,
            "status": "unavailable",
            "executed_pc": {"observed": False, "tested_pc_seen": False, "count": 0},
            "trace_path": {"observed": False, "evidence": "renode-execution-tracer-empty"},
            "jit_path": {"observed": False, "evidence": "renode-no-translation-block-trace"},
        }
    result = {
        "contract": PATH_IDENTITY_WITNESS_CONTRACT,
        "status": "observed",
        "digest": digest,
        "executed_pc": {
            "observed": True,
            "tested_pc_seen": False,
            "count": len(sequence),
            "digest": digest,
            "evidence": evidence,
        },
        "trace_path": {
            "observed": evidence.startswith("Renode-CreateExecutionTracing-PC"),
            "evidence": evidence,
        },
        "jit_path": {
            "observed": False,
            "evidence": "renode-trace-does-not-expose-translation-block-identity",
        },
        "executed_pcs": sequence,
    }
    if len(sequence) > INLINE_EXECUTED_PC_LIMIT and isinstance(trace_artifact, dict):
        result["executed_pcs"] = []
        result["executed_pcs_artifact"] = {
            **trace_artifact,
            "format": "text-pc",
            "pc_count": len(sequence),
            "pc_digest": digest,
        }
    return result


def _configuration_identity(
    *,
    profile: str,
    cpu_type: str,
    entry: int,
    loads: tuple[tuple[int, int], ...],
    mailbox: int,
    observation_size: int,
    capsule_sha256: str,
    renode_source_commit: str | None,
    tlib_source_commit: str | None,
    renode_runtime_sha256: str,
    dotnet_host_sha256: str,
    target_details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a stable identity for the actual Renode execution setup.

    The transient TCP port and trace path are deliberately excluded from the
    digest because they are transport details.  The mailbox address remains in
    the payload because it is part of the capsule contract.  The outer
    campaign records the single execution observation and its identity.
    """
    profile = str(profile or "").strip().lower()
    payload = {
        "contract": _CONFIGURATION_IDENTITY_CONTRACT,
        "backend": "renode-riscv64",
        "isa_profile": profile,
        "cpu_type": cpu_type,
        "entry": entry,
        "load_segments": [list(item) for item in loads],
        "mailbox": mailbox,
        "observation_size": observation_size,
        "capsule_sha256": capsule_sha256,
        "renode_source_commit": renode_source_commit,
        "tlib_source_commit": tlib_source_commit,
        "renode_runtime_sha256": renode_runtime_sha256,
        "dotnet_host_sha256": dotnet_host_sha256,
        "execution_model": "bare-metal-capsule",
        "trace": "Renode-CreateExecutionTracing-PC",
        "trace_format": "text-pc",
        "target_execution_binary_sha256": target_details.get(
            "target_execution_binary_sha256"
        ),
        "target_execution_identity_digest": target_details.get(
            "target_execution_identity_digest"
        ),
    }
    return _target_configuration(
        payload,
        target_details or {
            "target_binary_sha256": renode_runtime_sha256,
            "target_identity_status": "missing",
            "target_identity": None,
            "target_identity_digest": None,
        },
    )


def _connect(
    port: int, proc: subprocess.Popen[bytes], deadline: float,
) -> RspClient:
    # The caller owns one backend deadline for startup and all RSP requests.
    last: OSError | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = ""
            if proc.stderr:
                try:
                    proc.stderr.flush()
                    proc.stderr.seek(0)
                except (OSError, AttributeError):
                    pass
                try:
                    stderr = proc.stderr.read().decode(errors="replace").strip()
                except (OSError, AttributeError):
                    stderr = ""
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"Renode exited before GDB connection: {proc.returncode}{detail}")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock = socket.create_connection(
                ("127.0.0.1", port), timeout=min(1, remaining),
            )
            sock.settimeout(None)
            return RspClient(sock, deadline=deadline)
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    if time.monotonic() >= deadline:
        raise TimeoutError("Renode backend deadline expired during GDB connection")
    raise ConnectionError(f"Renode GDB connection timeout: {last}")


def _backend_timeout_seconds() -> float:
    try:
        budget = float(os.environ.get("RQ1_BACKEND_TIMEOUT_SECONDS", "180"))
    except ValueError:
        budget = 180.0
    return min(600.0, max(1.0, budget))


def _logical_target_details(
    runtime_details: dict[str, object], logical_binary_path: str | None,
) -> dict[str, object]:
    """Pin coverage executions to the source-verified simulator identity."""
    if not logical_binary_path:
        return runtime_details
    logical = _target_binary_details(
        logical_binary_path, expected_backend="renode-riscv64",
    )
    if logical.get("target_identity_status") != "verified":
        raise RuntimeError("logical Renode target identity is not verified")
    if runtime_details.get("target_identity_status") != "verified":
        raise RuntimeError("Renode execution target identity is not verified")
    return {
        **logical,
        "target_execution_binary_sha256": runtime_details.get("target_binary_sha256"),
        "target_execution_identity_digest": runtime_details.get("target_identity_digest"),
    }


def _process_descendants(pid: int) -> tuple[int, ...]:
    """Return direct and nested children using the Linux proc contract."""
    pending = [pid]
    descendants: list[int] = []
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text(
                encoding="ascii",
            ).split()
        except (OSError, UnicodeError):
            continue
        for value in children:
            try:
                child = int(value)
            except ValueError:
                continue
            if child not in descendants:
                descendants.append(child)
                pending.append(child)
    return tuple(descendants)


def _finish_process(proc: subprocess.Popen[bytes], *, coverage: bool) -> None:
    """Flush dotnet coverage before terminating the Renode process tree."""
    if coverage and os.name == "posix":
        try:
            flush_timeout = float(
                os.environ.get("RQ1_DOTNET_COVERAGE_FLUSH_TIMEOUT_SECONDS", "240")
            )
        except ValueError:
            flush_timeout = 240.0
        flush_timeout = min(300.0, max(1.0, flush_timeout))
        try:
            if proc.poll() is None:
                # Renode has already received the debugger ``quit`` command.
                # Let dotnet-coverage observe the normal child exit and flush
                # Cobertura first; SIGINTing the collector while it is still
                # serializing produces a valid-looking but empty report.
                proc.wait(timeout=flush_timeout)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if proc.poll() is None:
                    # dotnet-coverage is the parent of Renode. Stop only the
                    # instrumented child first; the collector then observes a
                    # normal child exit and can serialize Cobertura fully.
                    for child in reversed(_process_descendants(proc.pid)):
                        try:
                            os.kill(child, signal.SIGTERM)
                        except (OSError, ProcessLookupError):
                            pass
                    proc.wait(timeout=flush_timeout)
            except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGINT)
                        proc.wait(timeout=flush_timeout)
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                    pass
    try:
        if proc.poll() is None:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
    except (OSError, ProcessLookupError):
        pass


def _render_resc(
    elf_path: Path,
    port: int,
    cpu_type: str,
    trace_path: Path,
) -> str:
    entry, loads = _elf_loads(elf_path)
    memory_ranges: list[tuple[int, int]] = []
    for virtual, size in sorted(loads):
        start = virtual & ~0xFFF
        end = (virtual + size + 0xFFF) & ~0xFFF
        if memory_ranges and start <= memory_ranges[-1][1]:
            memory_ranges[-1] = (memory_ranges[-1][0], max(memory_ranges[-1][1], end))
        else:
            memory_ranges.append((start, end))
    memory_nodes = "\n".join(
        f'        machine LoadPlatformDescriptionFromString "mem{index}: Memory.MappedMemory @ sysbus {hex(start)} {{ size: {hex(end - start)} }}"'
        for index, (start, end) in enumerate(memory_ranges)
    )
    return textwrap.dedent(
        f"""
        using sysbus
        mach create
        machine LoadPlatformDescriptionFromString "cpu: CPU.RiscV64 @ sysbus {{ cpuType: \\"{cpu_type}\\"; timeProvider: empty }}"
{memory_nodes}
        machine LoadPlatformDescriptionFromString "trap: Memory.MappedMemory @ sysbus 0x1000 {{ size: 0x1000 }}"
        sysbus LoadELF @{elf_path}
        sysbus.cpu PC {hex(entry)}
        sysbus WriteDoubleWord 0x1010 0x00100073
        sysbus.cpu MTVEC 0x1010
        sysbus.cpu AddHook 0x1010 "self.Pause()"
        # ExecutionTracer flushes complete translation blocks at block end.
        # The capsule terminates by trapping at the final ebreak, so the
        # default large block would lose the whole test/exit block before its
        # end hook runs.  One-instruction blocks preserve the native tracer
        # contract and make the observed PC path complete.
        sysbus.cpu MaximumBlockSize 1
        # The native tracer is installed before StartGdbServer, so it covers
        # the whole autostart run, including code executed before GDB attaches.
        sysbus.cpu CreateExecutionTracing \"rq1-pc\" @{trace_path} PC
        machine StartGdbServer {port}
        """
    ).strip()


def _flush_and_read_trace(
    rsp: RspClient,
    trace_path: Path,
    *,
    stop_pcs: tuple[int, ...] = (),
) -> tuple[tuple[int, ...], bool, str | None, str]:
    """Read after Renode acknowledges that its trace writer has stopped."""
    try:
        trace_stopped = rsp.qrcmd(
            "sysbus.cpu DisableExecutionTracing", timeout=None,
        ) == "OK"
    except (ConnectionError, TimeoutError):
        trace_stopped = False
    pcs, parse_error, diagnostic = _parse_executed_pcs(trace_path)
    # A missing stop acknowledgement or a malformed row leaves only a prefix
    # whose end is unknown.  Keep the raw diagnostic, but never expose that
    # prefix as an observed execution path.
    if not trace_stopped or parse_error is not None:
        pcs = ()
    return _trace_through_stop(pcs, stop_pcs), trace_stopped, parse_error, diagnostic


def _trace_through_stop(
    pcs: tuple[int, ...], stop_pcs: tuple[int, ...]
) -> tuple[int, ...]:
    indexes = [index for index, pc in enumerate(pcs) if pc in stop_pcs]
    return pcs[: indexes[-1] + 1] if indexes else pcs


def _outside_elf_pc_loop(
    path: Path | None, executable_loads: tuple[tuple[int, int], ...], *,
    repeats: int = 32,
) -> int | None:
    """Return a repeated PC outside the executable ELF from the live trace tail."""
    if path is None or not executable_loads or repeats < 1:
        return None
    try:
        size = path.stat().st_size
        if size <= 0:
            return None
        with path.open("rb") as stream:
            stream.seek(max(0, size - 65536))
            lines = stream.read().splitlines()
    except OSError:
        return None
    repeated_pc = None
    repeated = 0
    for line in reversed(lines):
        value = line.strip()
        if not value.lower().startswith(b"0x"):
            continue
        try:
            pc = int(value, 16)
        except ValueError:
            return None
        if _trace_pcs_in_loads((pc,), executable_loads):
            return None
        if repeated_pc is None:
            repeated_pc = pc
        elif pc != repeated_pc:
            return None
        repeated += 1
        if repeated >= repeats:
            return repeated_pc
    return None


def _repeated_pc_loop(path: Path | None, *, repeats: int = 32) -> int | None:
    """Return a PC from a short cycle in the live trace tail.

    The old guard only recognized ``pc, pc, ...``.  Renode can instead rotate
    through a small trap/dispatch block, which still makes no semantic
    progress but previously ran until the outer 180-second timeout.
    """
    if path is None or repeats < 1:
        return None
    try:
        size = path.stat().st_size
        if size <= 0:
            return None
        with path.open("rb") as stream:
            stream.seek(max(0, size - 65536))
            lines = stream.read().splitlines()
    except OSError:
        return None
    pcs: list[int] = []
    for line in reversed(lines):
        value = line.strip()
        if not value.lower().startswith(b"0x"):
            continue
        try:
            pc = int(value, 16)
        except ValueError:
            return None
        pcs.append(pc)
    if not pcs:
        return None
    return _repeated_pc_cycle(tuple(reversed(pcs)), repeats=repeats)


def _append_mailbox_next_pc(
    pcs: tuple[int, ...], checkpoint_pc: int, executable_loads: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], bool]:
    """Close a complete trace at the PC after the mailbox store.

    RVGEN's compact capsule writes RVOBS1 as its final instruction. Renode's
    execution tracer records instruction start PCs, while the mailbox stores
    the architectural next PC. In that valid case the checkpoint is absent
    from the trace by construction; accept only an aligned 2/4-byte successor
    that remains inside the verified executable ELF range.
    """
    if not pcs or checkpoint_pc in pcs or checkpoint_pc & 1:
        return pcs, False
    delta = checkpoint_pc - pcs[-1]
    if delta not in {2, 4} or not _trace_pcs_in_loads((checkpoint_pc,), executable_loads):
        return pcs, False
    return (*pcs, checkpoint_pc), True


def _append_trap_pc(
    pcs: tuple[int, ...], csr_trap_state: dict[str, object]
) -> tuple[tuple[int, ...], bool]:
    epc = csr_trap_state.get("guest_epc")
    if (
        csr_trap_state.get("guest_trap") == "delivered"
        and type(epc) is int
        and (not pcs or pcs[-1] != epc)
    ):
        return (*pcs, epc), True
    return pcs, False


def _persistent_trace_path(elf: Path) -> tuple[Path, Path]:
    configured_run_root = os.environ.get("RQ1_RUN")
    trace_root = (
        Path(configured_run_root).resolve()
        if configured_run_root and Path(configured_run_root).is_dir()
        else elf.resolve().parent
    )
    trace_parent = trace_root / "traces" / "renode-riscv64"
    trace_parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(tempfile.mkdtemp(prefix="trace-", dir=trace_parent))
    return trace_root, trace_dir / "renode-executed.trace"


def _trace_artifact_record(path: Path, trace_root: Path) -> dict[str, object]:
    recorded = path.is_file()
    digest = hashlib.sha256()
    if recorded:
        with path.open("rb") as trace_file:
            for chunk in iter(lambda: trace_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "status": "recorded" if recorded else "missing",
        "path": path.relative_to(trace_root).as_posix(),
        "absolute_path": str(path.resolve()),
        "bytes": path.stat().st_size if recorded else None,
        "sha256": digest.hexdigest() if recorded else None,
    }


def _drain_pending(rsp: RspClient) -> str | None:
    """Swallow packets Renode may have queued on the wire.

    A resume on a machine that already finished during autostart never gets a
    stop reply, but the abort notification can still arrive late; drain it so
    the subsequent mailbox reads are not desynchronized.
    """
    previous = rsp.sock.gettimeout()
    stop_reply = None
    try:
        rsp.sock.settimeout(0.2)
        while True:
            try:
                reply = rsp.read_reply()
            except (TimeoutError, ConnectionError):
                break
            if _is_guest_trap_stop(reply):
                stop_reply = reply
        return stop_reply
    finally:
        rsp.sock.settimeout(previous)


def _resume_to_stop(rsp: RspClient) -> str | None:
    try:
        initial = rsp.request("?")
        if _is_guest_trap_stop(initial) and initial.strip().lower()[1:3] != "05":
            return initial
    except TimeoutError:
        pending = _drain_pending(rsp)
        if pending is not None and pending.strip().lower()[1:3] != "05":
            return pending
    while True:
        try:
            reply = rsp.request("c")
        except TimeoutError:
            return _drain_pending(rsp)
        if reply is None:
            return reply
        # The first S05 can be the GDB attach stop; after one continue, a
        # repeated S05 is already the terminal trap/stop.  Continuing it
        # forever leaves the campaign process alive with no child target.
        if _is_guest_trap_stop(reply) or reply.strip().lower()[1:3] != "05":
            return reply


def _is_guest_trap_stop(reply: str | None) -> bool:
    if not isinstance(reply, str):
        return False
    value = reply.strip().lower()
    try:
        return value[:1] in {"s", "t", "x"} and int(value[1:3], 16) in {4, 5, 6, 7, 0xB}
    except ValueError:
        return False


def _parse_executed_pcs(
    trace_path: Path,
) -> tuple[tuple[int, ...], str | None, str]:
    """Parse a TraceFormat.PC trace file into an executed-PC sequence.

    Keep the valid prefix and a short diagnostic when a row is malformed.  The
    complete trace remains at ``trace_path``; it must not be copied into the
    JSON response or resident-session memory.
    """
    try:
        stream = trace_path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return (), f"trace-read-error:{type(exc).__name__}:{exc}", ""
    pcs: list[int] = []
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            diagnostic_line = line[:256]
            if not line.lower().startswith("0x"):
                return (
                    tuple(pcs),
                    f"malformed-trace-line-{line_number}:{diagnostic_line}",
                    "",
                )
            try:
                pc = int(line, 16)
            except ValueError:
                return (
                    tuple(pcs),
                    f"invalid-pc-at-line-{line_number}:{diagnostic_line}",
                    "",
                )
            if pc >= 1 << 64:
                return (
                    tuple(pcs),
                    f"pc-out-of-range-at-line-{line_number}:{diagnostic_line}",
                    "",
                )
            if pc & 1:
                return (
                    tuple(pcs),
                    f"unaligned-pc-at-line-{line_number}:{diagnostic_line}",
                    "",
                )
            pcs.append(pc)
    return tuple(pcs), None, ""


# Renode exposes the standard RISC-V GDB register map: x0-x31 = 0-31,
# pc = 32, f0-f31 = 33-64, fflags = 65, frm = 66 (see tlib
# cpu_registers.h and RegisterDescription.AddFpuFeature).
_RENODE_FPR_FIRST = 33
_RENODE_FFLAGS = 65
_RENODE_FRM = 66
# Renode exposes the mstatus CSR as GDB register 833 (CSR 0x341); the
# register is present in the tlib mapping, so the p command returns the real
# CSR value instead of the stub's placeholder-zero fallback for
# feature-only registers (ReadRegisterCommand.cs).
_RENODE_MSTATUS = 0x341
# M-1 (D-050/D-054/D-059): Renode tlib machine trap CSRs and privilege level
# at fixed GDB regnums (tlib/arch/riscv/cpu_registers.h: MEPC=0x382,
# MCAUSE=0x383, MTVAL=0x384, PRIV=4161).  The same register-map source owns
# MSTATUS=0x341 above, so these numbers are not assumed from the ISA spec.
_RENODE_MEPC = 0x382
_RENODE_MCAUSE = 0x383
_RENODE_MTVAL = 0x384
_RENODE_PRIV = 4161
_RENODE_DEFAULT_TRAP_VECTOR = 0x1010


def _read_mstatus_fs(rsp: RspClient) -> tuple[int | None, str, str | None]:
    """Read mstatus.FS (bits 13-14) through the GDB RSP single-register packet.

    A failed read is a named gap (fs-observation-unavailable); placeholder
    zeros are never fabricated into the compare.  The reply must be exactly
    one 64-bit register (16 hex bytes); a shorter reply (for example the
    stub's zero-fill of a feature-only register) fails closed instead of
    being misread as observed state.
    """
    reply = rsp.request(f"p{_RENODE_MSTATUS:x}")
    if reply.startswith("E") or len(reply) != 16:
        return (
            None,
            "gap",
            f"fs-observation-unavailable: gdb p{_RENODE_MSTATUS:x} failed: {reply}",
        )
    try:
        value = int.from_bytes(bytes.fromhex(reply), "little")
    except ValueError:
        return (
            None,
            "gap",
            f"fs-observation-unavailable: gdb p{_RENODE_MSTATUS:x} returned malformed hex: {reply}",
        )
    return (value >> 13) & 0x3, "observed", None


def _read_fp_state(rsp: RspClient, profile: str) -> dict[str, object]:
    """Read the real FPR/fflags/frm through GDB RSP single-register packets.

    Returns the observed FP state only when the executing profile actually
    requires it; otherwise the named gap is returned so the oracle never
    mistakes absent observation for architectural zeros.
    """
    if (
        not profile_requires_fp_state(profile)
        or os.environ.get("RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "0") != "1"
    ):
        return _empty_fp_state()
    mstatus_fs, mstatus_fs_observer, mstatus_fs_gap = _read_mstatus_fs(rsp)
    if mstatus_fs in (None, 0):
        return {
            "fpr_rawbits": None,
            "fflags": None,
            "frm": None,
            "fp_observer": "not-observed",
            "fp_observer_gap": (
                "renode-fp-observer-disabled: mstatus.FS=0"
                if mstatus_fs == 0 else mstatus_fs_gap
            ),
            "mstatus_fs": mstatus_fs,
            "mstatus_fs_observer": mstatus_fs_observer,
            "mstatus_fs_gap": mstatus_fs_gap,
        }
    rawbits = []
    for index in range(32):
        reply = rsp.request(f"p{_RENODE_FPR_FIRST + index:x}")
        if reply.startswith("E") or len(reply) != 16:
            raise RuntimeError(f"Renode GDB FPR read failed for f{index}: {reply}")
        rawbits.append(int.from_bytes(bytes.fromhex(reply), "little"))
    fflags_reply = rsp.request(f"p{_RENODE_FFLAGS:x}")
    frm_reply = rsp.request(f"p{_RENODE_FRM:x}")

    def _control_value(reply: str, name: str) -> int:
        # RiscVRegisterDescription advertises the RV64 FFLAGS/FRM registers
        # at the GDB register width, while older tlib builds may return the
        # architectural 32-bit payload.  Accept exactly those two widths;
        # arbitrary short replies must stay a named runner gap instead of
        # being silently turned into zero by zfill().
        if reply.startswith("E") or len(reply) not in (8, 16):
            raise RuntimeError(f"Renode GDB {name} read failed: {reply}")
        try:
            return int.from_bytes(bytes.fromhex(reply), "little")
        except ValueError as exc:
            raise RuntimeError(f"Renode GDB {name} reply is malformed: {reply}") from exc

    fflags = _control_value(fflags_reply, "fflags")
    frm = _control_value(frm_reply, "frm")
    if fflags > 0x1F or frm > 0x7:
        raise RuntimeError(f"Renode FP flags out of range: fflags={fflags:#x} frm={frm:#x}")
    # Observed-state marker: main() otherwise defaults a successful read to
    # "not-observed", which makes the strict oracle treat real FPR/flags
    # evidence as a declared gap and drops the FP compare entirely.
    return {
        "fpr_rawbits": rawbits,
        "fflags": fflags,
        "frm": frm,
        "fp_observer": "observed",
        "mstatus_fs": mstatus_fs,
        "mstatus_fs_observer": mstatus_fs_observer,
        "mstatus_fs_gap": mstatus_fs_gap,
    }


def _read_csr_trap_state(
    rsp: RspClient,
    *,
    trap_stop: bool = False,
    mailbox_ready: bool = False,
) -> dict[str, object]:
    """Read mepc/mcause/mtval/priv through GDB RSP single-register packets.

    M-1 observer extension for D-050 (ialign-csr), D-054 (jalr-mtval) and
    D-059 (mstatus.TW): the trap-context CSRs are only meaningful when the
    guest took a synchronous exception, so every read fails closed
    (``csr-observation-unavailable``) rather than fabricating architectural
    values into the compare.
    """
    result: dict[str, object] = {}
    registers = (("priv", _RENODE_PRIV),) if mailbox_ready else (
        ("mepc", _RENODE_MEPC),
        ("mcause", _RENODE_MCAUSE),
        ("mtval", _RENODE_MTVAL),
        ("priv", _RENODE_PRIV),
    )
    for name, regnum in registers:
        reply = rsp.request(f"p{regnum:x}")
        if reply.startswith("E") or not reply:
            result[f"csr.{name}"] = None
            result[f"csr.{name}_observer"] = "gap"
            result[f"csr.{name}_gap"] = (
                f"csr-observation-unavailable: gdb p{regnum:x} failed: {reply}"
            )
            continue
        if len(reply) != 16:
            result[f"csr.{name}"] = None
            result[f"csr.{name}_observer"] = "gap"
            result[f"csr.{name}_gap"] = (
                f"csr-observation-unavailable: gdb p{regnum:x} returned incomplete 64-bit value: {reply}"
            )
            continue
        try:
            value = int.from_bytes(bytes.fromhex(reply), "little")
        except ValueError:
            result[f"csr.{name}"] = None
            result[f"csr.{name}_observer"] = "gap"
            result[f"csr.{name}_gap"] = (
                f"csr-observation-unavailable: gdb p{regnum:x} returned malformed hex: {reply}"
            )
            continue
        result[f"csr.{name}"] = value
        result[f"csr.{name}_observer"] = "observed"
    if mailbox_ready:
        return result
    # A stop packet alone does not prove a guest trap: Renode also reports
    # debugger SIGTRAP as S05.  Cause zero needs a nonzero mtval to identify
    # an instruction-address-misaligned exception.
    required_csr_observed = all(
        result.get(f"csr.{name}_observer") == "observed"
        for name in ("mepc", "mcause")
    )
    cause = result.get("csr.mcause")
    epc = result.get("csr.mepc")
    tval = result.get("csr.mtval")
    delivered = (
        trap_stop
        and required_csr_observed
        and type(cause) is int
        and not cause & (1 << 63)
        and type(epc) is int
        and not epc & 1
        and (cause != 0 or type(tval) is int and tval != 0)
    )
    if delivered:
        result["guest_trap"] = "delivered"
        result["trap_observer"] = "renode-gdb-csr"
        for source, canonical, guest_key in (
            ("mepc", "trap.epc", "guest_epc"),
            ("mcause", "trap.cause", "guest_cause"),
            ("mtval", "trap.tval", "guest_tval"),
        ):
            if result.get(f"csr.{source}_observer") == "observed":
                result[canonical] = result[f"csr.{source}"]
                result[f"{canonical}_observer"] = "observed"
                result[guest_key] = result[f"csr.{source}"]
                result[f"{guest_key}_observer"] = "renode-gdb-csr"
        result["fault_pc"] = result.get("guest_epc")
        if result.get("guest_cause") in _ADDRESS_FAULT_CAUSES and type(result.get("guest_tval")) is int:
            result["fault_address"] = result.get("guest_tval")
    else:
        for name in ("mepc", "mcause", "mtval"):
            if result.get(f"csr.{name}_observer") != "observed":
                continue
            result[f"csr.{name}"] = None
            result[f"csr.{name}_observer"] = "gap"
            result[f"csr.{name}_gap"] = (
                "csr-observation-unavailable: trap context was not delivered"
            )
    return result


def _read_stop_pc(rsp: RspClient) -> int | None:
    reply = rsp.request("p20")
    if reply.startswith("E") or len(reply) != 16:
        return None
    try:
        return int.from_bytes(bytes.fromhex(reply), "little")
    except ValueError:
        return None


def _run(
    elf: Path,
    mailbox: int,
    observation_size: int,
    dotnet: Path,
    renode_dll: Path,
) -> tuple[bytes, dict[str, object], tuple[int, ...]]:
    deadline_value = os.environ.get("RQ1_BACKEND_DEADLINE_MONOTONIC")
    backend_deadline = (
        float(deadline_value) if deadline_value
        else time.monotonic() + _backend_timeout_seconds()
    )
    if not elf.is_file():
        raise RuntimeError(f"missing capsule ELF: {elf}")
    if observation_size < OBSERVATION_HEADER_SIZE:
        raise ValueError("Renode RVOBS1 frame is too small")
    if not is_riscv_elf(elf):
        raise ValueError("expected RISC-V ELF")
    executable_loads = _elf_executable_loads(elf)
    if not dotnet.is_file():
        raise RuntimeError(f"missing dotnet host: {dotnet}")
    if not renode_dll.is_file():
        raise RuntimeError(f"missing Renode runtime: {renode_dll}")
    logical_target_path = os.environ.get("RQ1_TARGET_LOGICAL_BINARY_PATH", "").strip() or None
    runtime_target_details = _target_binary_details(
        str(renode_dll), expected_backend="renode-riscv64"
    )
    target_details = _logical_target_details(
        runtime_target_details,
        logical_target_path,
    )
    target_identity = target_details.get("target_identity")
    target_identity_verified = (
        target_details.get("target_identity_status") == "verified"
        and isinstance(target_identity, dict)
        and target_identity.get("source_commit") == "d66b0c2aa3d420408eccecfd1d3bab0fd702a6db"
    )
    renode_source = Path(os.environ.get("RENODE_SOURCE", str(DEFAULT_RENODE_SOURCE))).resolve()
    tlib_source = Path(os.environ.get("RENODE_TLIB_SOURCE", str(DEFAULT_TLIB_SOURCE))).resolve()
    if (not renode_source.is_dir() or not tlib_source.is_dir()) and not target_identity_verified:
        raise RuntimeError("Renode source checkout is unavailable")
    renode_source_clean = target_identity_verified or git_worktree_clean(renode_source)
    tlib_source_clean = target_identity_verified or git_worktree_clean(tlib_source)
    coverage_launcher_value = os.environ.get("RQ1_DOTNET_COVERAGE_LAUNCHER", "").strip()
    coverage_output_value = os.environ.get("RQ1_DOTNET_COVERAGE_OUTPUT", "").strip()
    coverage_include_files = os.environ.get("RQ1_DOTNET_COVERAGE_INCLUDE_FILES", "").strip()
    coverage_session_id = os.environ.get("RQ1_DOTNET_COVERAGE_SESSION_ID", "").strip()
    if bool(coverage_launcher_value) != bool(coverage_output_value):
        raise RuntimeError("Renode dotnet coverage launcher/output must be configured together")
    if coverage_launcher_value and not logical_target_path:
        raise RuntimeError(
            "Renode dotnet coverage requires RQ1_TARGET_LOGICAL_BINARY_PATH"
        )
    coverage_launcher = Path(coverage_launcher_value) if coverage_launcher_value else None
    coverage_output = Path(coverage_output_value) if coverage_output_value else None
    if coverage_launcher is not None:
        if not coverage_launcher.is_file():
            raise RuntimeError(f"missing dotnet coverage launcher: {coverage_launcher}")
        coverage_output.parent.mkdir(parents=True, exist_ok=True)
        coverage_output.unlink(missing_ok=True)
    trace_root, trace_path = _persistent_trace_path(elf)
    # Renode's startup cleanup removes /tmp entries named "renode-*"; keep
    # its short-lived control files outside that prefix.
    with tempfile.TemporaryDirectory(prefix="capsule-run-") as tmp:
        tmpdir = Path(tmp)
        resc_path = tmpdir / "renode.resc"
        # The native tracer is installed before StartGdbServer, so it covers the
        # whole autostart run, including code executed before GDB attaches.
        profile = (
            os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
            or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
        ).strip().lower()
        if not profile:
            raise RuntimeError("RV_TESTCASE_ISA_PROFILE is required for Renode CPU selection")
        cpu_type = capsule_cpu_profile(profile)
        if cpu_type is None:
            raise RuntimeError(f"Renode CPU profile is outside the validated capsule contract: {profile}")
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(dotnet.parent), env.get("PATH", "")])
        env["DOTNET_ROOT"] = str(dotnet.parent)
        proc: subprocess.Popen[bytes] | None = None
        stderr_file = None
        rsp: RspClient | None = None
        trace_guard_stop = threading.Event()
        trace_guard_pc: list[int] = []
        trace_guard_reason: list[str] = []
        trace_guard_thread = None

        def start() -> tuple[int, str, subprocess.Popen[bytes], RspClient]:
            nonlocal proc
            with _PORT_STARTUP_LOCK:
                port = free_port()
                resc_text = _render_resc(elf, port, cpu_type, trace_path) + "\n"
                resc_path.write_text(resc_text, encoding="utf-8")
                command = [
                    str(dotnet), str(renode_dll), "--disable-gui", "--console",
                    "--plain", "-e", f"include @{resc_path}",
                ]
                if coverage_launcher is not None and coverage_output is not None:
                    if coverage_session_id:
                        remaining_ms = max(
                            1, int((backend_deadline - time.monotonic()) * 1000),
                        )
                        command = [
                            str(coverage_launcher), "connect", coverage_session_id,
                            *command, "--timeout",
                            str(remaining_ms),
                        ]
                    else:
                        command = [
                            str(coverage_launcher), "collect", "--nologo", "-f", "cobertura",
                            *( ["--include-files", coverage_include_files]
                               if coverage_include_files else [] ),
                            "-o", str(coverage_output), "--", *command,
                        ]
                # The coverage wrapper can keep its child alive while it
                # serializes the Cobertura report.  A PIPE here lets that
                # detached child keep the capsule runner's stderr open, so
                # the parent `communicate()` waits until the outer target
                # deadline even after the RVOBS1 result is complete.  Keep
                # diagnostics in the case temp directory instead; the
                # framework result itself is emitted by this process.
                nonlocal stderr_file
                stderr_destination = subprocess.PIPE
                if coverage_launcher is not None:
                    stderr_file = tempfile.TemporaryFile(
                        mode="w+b", prefix="renode-stderr-", dir=tmpdir,
                    )
                    stderr_destination = stderr_file
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_destination,
                    env=env,
                    cwd=str(renode_dll.parent),
                    # capsule_dispatch already owns the outer process group.
                    # Keep the coverage wrapper in that group so the parent
                    # Target deadline can terminate Renode and dotnet-coverage
                    # together instead of leaving the collector detached.
                    start_new_session=False,
                )
                return port, resc_text, proc, _connect(
                    port, proc, backend_deadline,
                )

        try:
            port, resc_text, proc, rsp = start()
            trace_enabled = True
            def watch_invalid_pc_loop() -> None:
                while not trace_guard_stop.is_set() and proc is not None \
                        and proc.poll() is None:
                    invalid_pc = _outside_elf_pc_loop(trace_path, executable_loads)
                    if invalid_pc is not None:
                        trace_guard_pc.append(invalid_pc)
                        trace_guard_reason.append("guest-pc-outside-elf-loop")
                        try:
                            rsp.sock.sendall(b"\x03")
                        except OSError:
                            pass
                        return
                    # RVGEN single capsules are finite straight-line witnesses;
                    # do not apply this heuristic to general program-route code.
                    if os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME"):
                        repeated_pc = _repeated_pc_loop(trace_path)
                        if repeated_pc is not None:
                            trace_guard_pc.append(repeated_pc)
                            trace_guard_reason.append("guest-pc-loop")
                            try:
                                rsp.sock.sendall(b"\x03")
                            except OSError:
                                pass
                            return
                    trace_guard_stop.wait(0.05)

            trace_guard_thread = threading.Thread(
                target=watch_invalid_pc_loop,
                name="rq1-renode-invalid-pc-watch", daemon=True,
            )
            trace_guard_thread.start()
            # Resume the machine: the capsule runs to its terminal ebreak,
            # traps into Renode's default mtvec (0x1010) and aborts, which the
            # GDB stub reports as X06.  A capsule that already finished during
            # autostart (the machine starts when the GDB client connects) never
            # replies to the resume, so a timeout is tolerated and the
            # finalized mailbox frame is the arbiter.
            resume_reply = _resume_to_stop(rsp)
            raw = _read_mailbox(rsp, mailbox, observation_size, "Renode")
            checkpoint_pc = int.from_bytes(raw[8:16], "little") if len(raw) >= 16 else 0
            mailbox_ready = raw.startswith(OBSERVATION_MAGIC) and checkpoint_pc != 0
            trap_stop = _is_guest_trap_stop(resume_reply) or (
                isinstance(resume_reply, str)
                and resume_reply.strip().lower()[:3] in {"s00", "s02"}
            )
            csr_trap_state = _read_csr_trap_state(
                rsp,
                trap_stop=trap_stop,
                mailbox_ready=mailbox_ready,
            )
            stop_pc = None
            if not mailbox_ready and isinstance(resume_reply, str) \
                    and resume_reply.strip().lower()[:3] == "s00":
                try:
                    stop_pc = _read_stop_pc(rsp)
                except (ConnectionError, TimeoutError):
                    pass
            if not mailbox_ready:
                raw = b""
            executed_pcs, trace_stopped, trace_parse_error, trace_diagnostic = (
                _flush_and_read_trace(
                    rsp,
                    trace_path,
                    stop_pcs=(
                        (checkpoint_pc,) if mailbox_ready
                        else (
                            (csr_trap_state["guest_epc"],)
                            if type(csr_trap_state.get("guest_epc")) is int
                            else ()
                        )
                    ),
                )
                if trace_enabled else ((), False, None, "")
            )
            trace_artifact = _trace_artifact_record(trace_path, trace_root)
            trace_pcs = executed_pcs
            entry, loads = _elf_loads(elf)
            trap_epc = csr_trap_state.get("guest_epc")
            trace_pc_outside_elf = bool(trace_pcs) and not _trace_pcs_in_loads(
                trace_pcs, executable_loads
            )
            trap_pc_outside_elf = (
                type(trap_epc) is int
                and not _trace_pcs_in_loads((trap_epc,), executable_loads)
            )
            if trace_pc_outside_elf or trap_pc_outside_elf:
                executed_pcs, trap_pc_added = (), False
                checkpoint_after_trace = False
            else:
                executed_pcs, checkpoint_after_trace = _append_mailbox_next_pc(
                    executed_pcs, checkpoint_pc, executable_loads,
                )
                executed_pcs, trap_pc_added = _append_trap_pc(executed_pcs, csr_trap_state)
            trace_available = (
                trace_enabled
                and bool(trace_pcs)
                and not trace_pc_outside_elf
                and not trap_pc_outside_elf
            )
            mailbox_checkpoint_only = (
                mailbox_ready
                and not executed_pcs
                and _trace_pcs_in_loads((checkpoint_pc,), executable_loads)
            )
            if mailbox_checkpoint_only:
                executed_pcs = (checkpoint_pc,)
            gdb_stop_pc = (
                stop_pc if type(stop_pc) is int
                and (
                    _trace_pcs_in_loads((stop_pc,), executable_loads)
                    or stop_pc == _RENODE_DEFAULT_TRAP_VECTOR
                ) else None
            )
            target_stop_observed = (
                not mailbox_ready
                and csr_trap_state.get("guest_trap") != "delivered"
                and isinstance(resume_reply, str)
                and resume_reply.strip().lower()[:3] == "s00"
                and gdb_stop_pc is not None
            )
            if target_stop_observed and not executed_pcs:
                executed_pcs = (gdb_stop_pc,)
            if not mailbox_ready and csr_trap_state.get("guest_trap") != "delivered" \
                    and not target_stop_observed:
                if trace_guard_pc:
                    raise RuntimeError(
                        f"{trace_guard_reason[0] if trace_guard_reason else 'guest-pc-loop'}:"
                        f"0x{trace_guard_pc[0]:x}"
                    )
                raise RuntimeError(
                    "Renode stopped without finalized mailbox or guest trap: "
                    f"reply={resume_reply!r}, trace={trace_pcs!r}, "
                    f"stop_pc={stop_pc!r}, csr={csr_trap_state!r}"
                )
            # A GDB stop is terminal/path evidence only.  Do not synthesize
            # an RVOBS1 frame from the stop PC: consumers must never confuse
            # "Renode stopped" with a guest-written architectural snapshot.
            try:
                fp_state = _read_fp_state(rsp, profile)
            except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
                fp_state = {
                    "fpr_rawbits": None,
                    "fflags": None,
                    "frm": None,
                    "fp_observer": "not-observed",
                    "adapter_execution_error": f"final-fp-state-unavailable: {exc}",
                    "mstatus_fs": None,
                    "mstatus_fs_observer": "gap",
                    "mstatus_fs_gap": f"renode-fp-observer-unavailable: {exc}",
                }
            renode_source_commit = (
                target_identity.get("source_commit") if target_identity_verified
                else git_head(renode_source) if renode_source_clean else None
            )
            tlib_source_commit = (
                target_identity.get("source_commit") if target_identity_verified
                else git_head(tlib_source) if tlib_source_clean else None
            )
            renode_runtime_sha256 = sha256_file(renode_dll)
            dotnet_host_sha256 = sha256_file(dotnet)
            capsule_sha256 = sha256_file(elf)
            target_details = _logical_target_details(
                _target_binary_details(
                    str(renode_dll), expected_backend="renode-riscv64"
                ),
                logical_target_path,
            )
            path_identity = _path_identity_witness(
                executed_pcs,
                trace_artifact=trace_artifact,
                evidence=(
                    "RVOBS1-checkpoint" if mailbox_checkpoint_only
                    else "GDB-stop-PC" if target_stop_observed and not trace_pcs
                    else "Renode-CreateExecutionTracing-PC+RVOBS1-next-PC"
                    if checkpoint_after_trace
                    else "Renode-CreateExecutionTracing-PC"
                ),
            )
            if trap_pc_added:
                path_identity["executed_pc"]["evidence"] = (
                    "Renode-CreateExecutionTracing-PC+renode-gdb-csr"
                    if trace_pcs
                    else "renode-gdb-csr-trap-pc"
                )
                path_identity["trace_path"]["observed"] = bool(trace_pcs)
                path_identity["trace_path"]["evidence"] = (
                    "Renode-CreateExecutionTracing-PC"
                    if trace_pcs
                    else "renode-gdb-csr-trap-pc"
                )
            configuration_identity = _configuration_identity(
                profile=profile,
                cpu_type=cpu_type,
                entry=entry,
                loads=loads,
                mailbox=mailbox,
                observation_size=observation_size,
                capsule_sha256=capsule_sha256,
                renode_source_commit=renode_source_commit,
                tlib_source_commit=tlib_source_commit,
                renode_runtime_sha256=renode_runtime_sha256,
                dotnet_host_sha256=dotnet_host_sha256,
                target_details=target_details,
            )
            if not renode_source_clean or not tlib_source_clean:
                configuration_identity["identity_digest"] = None
            return raw, {
                "source": "renode-dotnet-gdb-rsp",
                "renode_source_commit": renode_source_commit,
                "tlib_source_commit": tlib_source_commit,
                "renode_source_clean": renode_source_clean,
                "tlib_source_clean": tlib_source_clean,
                "isa_profile": profile,
                "cpu_type": cpu_type,
                **{
                    key: target_details.get(key)
                    for key in (
                        "target_binary_sha256",
                        "target_identity_status",
                        "target_identity",
                        "target_identity_digest",
                        "target_execution_binary_sha256",
                        "target_execution_identity_digest",
                    )
                },
                "renode_runtime_sha256": renode_runtime_sha256,
                "dotnet_host_sha256": dotnet_host_sha256,
                "capsule_sha256": capsule_sha256,
                "gdb_bind": f"127.0.0.1:{port}",
                "execution_status": (
                    "completed-mailbox-checkpoint" if mailbox_checkpoint_only
                    else "completed-mailbox" if mailbox_ready
                    else "target-stop" if target_stop_observed else "guest-trap"
                ),
                "exit_code": 0 if mailbox_ready or target_stop_observed else 1,
                "target_stop_observed": target_stop_observed,
                "target_stop_pc": (
                    (gdb_stop_pc or executed_pcs[-1]) if target_stop_observed else None
                ),
                "dependency_family": backend_spec("renode-riscv64").dependency_family,
                "fpr_observed": fp_state.get("fp_observer") == "observed",
                "trace_available": trace_available,
                "renode_trace_complete": bool(trace_stopped and trace_parse_error is None),
                "renode_trace_parse_error": trace_parse_error,
                "renode_trace_diagnostic": trace_diagnostic,
                "renode_trace_stop_ack": trace_stopped,
                "renode_trace_artifact": trace_artifact,
                "renode_checkpoint_after_trace": checkpoint_after_trace,
                "trace_pc_outside_elf": trace_pc_outside_elf,
                "trap_pc_outside_elf": trap_pc_outside_elf,
                "renode_trace_mechanism": "Renode-CreateExecutionTracing-PC" if trace_enabled else None,
                "renode_executed_pc_count": len(executed_pcs),
                "renode_executed_pc_digest": path_identity.get("digest"),
                "path_identity": path_identity,
                "configuration_identity": configuration_identity,
                "fpr_available": fp_state.get("fp_observer") == "observed",
                "resc_sha256": hashlib.sha256(resc_text.encode("utf-8")).hexdigest(),
                "fp_state": fp_state,
                "csr_trap_state": csr_trap_state,
            }, executed_pcs
        finally:
            trace_guard_stop.set()
            if trace_guard_thread is not None:
                trace_guard_thread.join(timeout=1)
            if rsp is not None:
                try:
                    # dotnet-coverage emits the Cobertura report when the
                    # instrumented Renode exits normally.  Sending SIGINT to
                    # the collector first leaves an empty ``<packages />``
                    # report, so ask Renode itself to quit before the generic
                    # process-tree fallback below.
                    if coverage_launcher is not None:
                        try:
                            rsp.qrcmd("quit", timeout=5)
                        except (ConnectionError, OSError, TimeoutError):
                            pass
                    rsp.close()
                except Exception:
                    pass
            if proc is not None:
                _finish_process(proc, coverage=coverage_launcher is not None)
                try:
                    proc.communicate(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        proc.communicate(timeout=3)
                    except (OSError, subprocess.TimeoutExpired):
                        # dotnet/Renode may keep worker threads alive after kill;
                        # a lingering child must never mask the mailbox/trace
                        # result that was already produced above.
                        pass
                if stderr_file is not None:
                    try:
                        stderr_file.close()
                    except OSError:
                        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a framework capsule through Renode and its GDB stub")
    parser.add_argument("elf")
    parser.add_argument("--mailbox", default=os.environ.get("RV_CAPSULE_MAILBOX"))
    parser.add_argument("--observation-size", default=os.environ.get("RV_CAPSULE_OBSERVATION_SIZE"), type=int)
    parser.add_argument("--dotnet", default=os.environ.get("RENODE_DOTNET", str(DEFAULT_DOTNET)))
    parser.add_argument("--renode", default=os.environ.get("RENODE_DLL", str(DEFAULT_RENODE_DLL)))
    args = parser.parse_args(argv)
    if args.mailbox is None or args.observation_size is None:
        return runner_contract_gap("capsule mailbox and observation size are required")
    try:
        stdout, details, executed_pcs = _run(
            Path(args.elf).resolve(),
            int(args.mailbox, 0),
            args.observation_size,
            Path(args.dotnet).resolve(),
            Path(args.renode).resolve(),
        )
    except Exception as exc:
        return runner_contract_gap(f"Renode runner failed: {type(exc).__name__}: {exc}")
    sys.stdout.buffer.write(stdout)
    sys.stderr.write("RV_TOOL_VERSION=renode-dotnet-gdb-rsp\n")
    if executed_pcs:
        trace_artifact = details.get("renode_trace_artifact")
        if len(executed_pcs) > INLINE_EXECUTED_PC_LIMIT \
                and isinstance(trace_artifact, dict):
            trace_reference = dict(trace_artifact)
            trace_reference.update({
                "format": "text-pc",
                "pc_count": len(executed_pcs),
                "pc_digest": details.get("renode_executed_pc_digest"),
            })
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
    fp_state = details.get("fp_state", {})
    observation_extra_state = dict(fp_state)
    observation_extra_state.update(details.get("csr_trap_state", {}))
    if details.get("target_stop_observed") is True:
        observation_extra_state.update({
            "target_stop_observed": True,
            "target_stop_pc": details.get("target_stop_pc"),
            "target_stop_observer": "renode-gdb-rsp",
        })
    observed_fields = observed_state_fields(observation_extra_state)
    observation_extra_state["observer_fields"] = (
        observed_fields
        if observation_extra_state.get("guest_trap") == "delivered"
        else [field for field in observed_fields if not field.startswith("trap.") or field in observation_extra_state]
    )
    if details.get("target_stop_observed") is True:
        observation_extra_state["observer_fields"].append("target_stop")
    observation_state = {"extra_state": observation_extra_state}
    for field in ("fault_pc", "fault_address"):
        if field in observation_extra_state:
            observation_state[field] = observation_extra_state[field]
    sys.stderr.write(
        "RV_OBSERVATION_STATE="
        + json.dumps(
            observation_state,
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stderr.write(
        "RV_TRANSLATION_EVIDENCE="
        + json.dumps(
            {
                "backend": "renode-riscv64",
                "expected_path": "either",
                "tested_pc_seen": False,
                "tested_pc_translated": False,
                "tested_pc_executed": False,
                "execution_count": len(executed_pcs),
                "details": details,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return int(details.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())

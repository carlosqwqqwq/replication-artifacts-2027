from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.adapters.contracts import (
    PATH_IDENTITY_WITNESS_CONTRACT,
    PATH_IDENTITY_WITNESS_STATUS_OBSERVED,
    PATH_IDENTITY_WITNESS_STATUS_UNAVAILABLE,
    backend_spec,
    capsule_cpu_profile,
    profile_requires_fp_state,
)
from framework.adapters.rsp import RspClient, _PORT_STARTUP_LOCK, free_port
from framework.adapters.rvvm_rsp import set_elf_entry_pc
from framework.adapters.runner import (
    GUEST_TRAP_BRIDGE_FIELDS, _read_mailbox, observed_state_fields,
)
from framework.adapters.trace_common import _repeated_pc_cycle
from framework.direct_case import (
    INLINE_EXECUTED_PC_LIMIT, OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC,
)
from framework.direct_elf import symbol_offsets_from_elf
from framework.execution_identity import TargetBinaryIdentity, is_allowlisted_patched_identity
from framework.paths import LOCAL_ROOT
from framework._util import canonical_digest, elf_executable_loads, elf_load_segments, git_head, git_worktree_clean, is_riscv_elf, pc_digest, pc_path, pc_path_in_ranges, runner_contract_gap, sha256_file


DEFAULT_RVVM_SOURCE = LOCAL_ROOT / "simulator-sources" / "RVVM"
DEFAULT_RVVM = (
    LOCAL_ROOT / "target-store" / "rvvm-riscv64"
    / "9f432733d1d1ca1ad86239da9c5b4f5731afa4749da426e9ef9cef9791c1f140"
    / "rvvm_x86_64_wrapper"
)
_MAX_STOP_REPLY_DRAIN = 64


def _report_stage(stage: str) -> None:
    print(f"RVVM_STAGE={stage}", file=sys.stderr, flush=True)


def _strip_ecall_shim(
    elf: Path, pcs: tuple[int, ...],
) -> tuple[tuple[int, ...], int | None, int | None]:
    try:
        _entry, loads = elf_load_segments(elf)
        address = min(base for base, _data, _size in loads) + (256 << 20) - 0x1000
    except (OSError, ValueError):
        return pcs, None, None
    filtered = tuple(
        pc for pc in pcs if not address <= pc < address + _RVVM_ECALL_SHIM_SIZE
    )
    return filtered, len(pcs) - len(filtered), address


def _connect(
    port: int, proc: subprocess.Popen[bytes], deadline: float,
) -> RspClient:
    last: OSError | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            diagnostic = ""
            if proc.stderr is not None:
                diagnostic = proc.stderr.read().decode("utf-8", errors="replace").strip()
            suffix = f": {diagnostic}" if diagnostic else ""
            raise ConnectionError(
                f"RVVM exited before GDB connection: {proc.returncode}{suffix}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            sock = socket.create_connection(
                ("127.0.0.1", port),
                timeout=min(0.5, remaining),
            )
            sock.settimeout(None)
            return RspClient(sock, deadline=deadline)
        except OSError as exc:
            last = exc
            time.sleep(0.001)
    if time.monotonic() >= deadline:
        raise TimeoutError("RVVM backend deadline expired during GDB connection")
    raise ConnectionError(f"RVVM GDB connection timeout: {last}")


def _rvvm_command(
    rvvm: Path, firmware: Path, profile: str, harts: int, port: int,
    jit: bool, *, stepcov: bool = False, trace_path: Path | None = None,
) -> list[str]:
    command = [
        str(rvvm), str(firmware), "-isa", profile, "-harts", str(harts),
        "-nogui", "-nonet", "-nosound", "-noisolation",
        "-gdbstub", f"127.0.0.1:{port}",
    ]
    if not jit:
        command.append("-nojit")
    if stepcov:
        command.append("-rq1-stepcov")
    if trace_path is not None:
        command.extend(("-rq1-trace", str(trace_path)))
    return command



# Optional RVVM qRcmd trace-channel protocol constants.
_TRACE_ENABLED_REPLY = "enabled"
_STATE_TRACE_PAGE = 8
_STATE_TRACE_FIELDS = (
    "pc", "priv_mode", "csr.status", "csr.ie", "csr.ip", "csr.isa",
    "csr.fcsr", "csr.vcsr", "csr.vtype", "csr.mepc", "csr.mcause",
    "csr.mtval", "csr.mtvec", "reservation.valid", "reservation.addr",
    "reservation.value", "mmu.mode", "mmu.root",
)
_RVVM_ECALL_SHIM_SIZE = 16
_RVVM_CONFIG_CONTRACT = "rvvm-riscv64-configuration-v1"


def _path_identity_witness(
    executed_pcs: tuple[int, ...], *, jit: bool
) -> dict[str, object]:
    """Summarize an optional GDB PC path and selected RVVM execution mode."""
    if any(pc & 1 for pc in executed_pcs):
        executed_pcs = ()
    sequence, digest = pc_path(executed_pcs)
    mode = "jit" if jit else "interpreter"
    trace_evidence = "rvvm-gdb-rsp-pc-trace"
    if not sequence:
        return {
            "contract": PATH_IDENTITY_WITNESS_CONTRACT,
            "status": PATH_IDENTITY_WITNESS_STATUS_UNAVAILABLE,
            "digest": None,
            "path_digest": None,
            "target_path_digest": None,
            "executed_pc": {
                "observed": False,
                "count": 0,
                "digest": None,
                "evidence": (
                    "rvvm-jit-pc-trace-is-compile-time"
                    if jit
                    else "rvvm-gdb-rsp-pc-trace-unavailable"
                ),
            },
            "jit_path": {
                "observed": False,
                "mode": mode,
                "evidence": (
                    "rvvm-jit-pc-trace-is-compile-time"
                    if jit
                    else "rvvm-launch-mode-without-pc-trace"
                ),
            },
        }
    return {
        "contract": PATH_IDENTITY_WITNESS_CONTRACT,
        "status": PATH_IDENTITY_WITNESS_STATUS_OBSERVED,
        # Keep the three names accepted by the common adapter gate.  They
        # describe one digest, not three independent observations.
        "digest": digest,
        "path_digest": digest,
        "target_path_digest": digest,
        "executed_pc": {
            "observed": True,
            "count": len(sequence),
            "digest": digest,
            "evidence": trace_evidence,
        },
        "jit_path": {
            "observed": jit,
            "mode": mode,
            "evidence": "rvvm-gdb-rsp-pc-trace:jit-block-compilation-path" if jit else "rvvm-gdb-rsp-pc-trace:interpreter-path",
        },
        "executed_pcs": sequence,
    }


def _configuration_identity(
    *,
    profile: str,
    source_commit: str | None,
    rvvm_sha256: str,
    firmware_sha256: str,
    mailbox: int,
    observation_size: int,
    jit: bool,
    trace: str = "gdb-rsp:rvvm:pc-trace",
    capsule_mode: str = "linux-user-ecall",
    capsule_sha256: str | None = None,
    target_identity: TargetBinaryIdentity | None = None,
) -> dict[str, object]:
    """Return the identity of an RVVM execution configuration."""
    profile = str(profile or "").strip().lower()
    payload = {
        "contract": _RVVM_CONFIG_CONTRACT,
        "backend": "rvvm-riscv64",
        "isa_profile": profile,
        "source_commit": source_commit,
        "rvvm_binary_sha256": rvvm_sha256,
        "firmware_sha256": firmware_sha256,
        "capsule_sha256": capsule_sha256,
        "mailbox": mailbox,
        "observation_size": observation_size,
        "translation_path": "jit" if jit else "interpreter",
        "capsule_mode": capsule_mode,
        "trace": trace,
    }
    if target_identity is not None:
        payload.update({
            "target_identity_status": "verified",
            "target_identity_digest": target_identity.identity_digest,
            "target_binary_sha256": target_identity.binary_sha256,
            "target_source_provenance": target_identity.source_provenance,
        })
    return {**payload, "identity_digest": canonical_digest(payload)}


def _target_identity(rvvm: Path, binary_sha256: str) -> TargetBinaryIdentity | None:
    try:
        data = json.loads(rvvm.with_name("target-identity.json").read_text(encoding="utf-8"))
        identity = TargetBinaryIdentity.from_dict(data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return (
        identity
        if identity.backend == "rvvm-riscv64"
        and identity.binary_sha256 == binary_sha256
        and (
            identity.is_source_verified
            or is_allowlisted_patched_identity(identity)
        )
        else None
    )


def _trace_supported(rsp: RspClient, kind: str) -> bool:
    try:
        return _qrcmd(rsp, f"rvvm:{kind}-trace", timeout=None) == _TRACE_ENABLED_REPLY
    except (ConnectionError, TimeoutError, ValueError):
        return False


def _drain_pending_stop_replies(rsp: RspClient) -> None:
    pending = getattr(rsp, "_pending_replies", None)
    while isinstance(pending, list) and pending and pending[0].startswith(("S", "T", "X")):
        rsp.read_reply()
        pending = rsp._pending_replies


def _qrcmd(rsp: RspClient, command: str, *, timeout: float | None = None) -> str:
    reply = rsp.qrcmd(command, timeout=timeout)
    while reply.startswith(("S", "T", "X")) or (reply.startswith("O") and reply != "OK"):
        reply = rsp.read_reply()
    _drain_pending_stop_replies(rsp)
    return reply


def _clear_trace(rsp: RspClient) -> None:
    reply = _qrcmd(rsp, "rvvm:pc-trace clear", timeout=None)
    if reply != "OK":
        raise RuntimeError(f"RVVM pc-trace clear failed: {reply!r}")


class _TraceReadError(RuntimeError):
    def __init__(self, message: str, prefix: tuple) -> None:
        super().__init__(message)
        self.prefix = prefix


def _read_bulk_pc_trace(path: Path) -> tuple[tuple[int, ...], str | None]:
    """Read the complete ``INS 0x...`` trace emitted by the stepcov build."""
    try:
        text = path.read_text(encoding="ascii")
    except OSError as exc:
        return (), f"RVVM instruction trace unavailable: {exc}"
    pcs: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*INS\s+(0x[0-9a-fA-F]+)\s*", line)
        if match is None:
            return (), f"RVVM instruction trace malformed at line {line_number}"
        pc = int(match.group(1), 16)
        # stepcov leaves zero-filled records after the capsule stops.  Zero is
        # not a valid address in the bare capsule image; trim the sentinel so
        # one padded trace cannot invalidate an otherwise complete observation.
        if pc == 0:
            break
        # stepcov can record the initial PC twice at the capsule boundary.
        # Consecutive duplicate instruction PCs cannot be a real RISC-V
        # instruction sequence, so normalize this recorder artifact before
        # comparing the original and variant observations.
        if pcs and pcs[-1] == pc:
            continue
        pcs.append(pc)
    if not pcs:
        return (), "RVVM instruction trace is empty"
    return tuple(pcs), None


def _outside_elf_trace_loop(
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
    pcs = []
    for line in lines:
        match = re.fullmatch(rb"\s*INS\s+(0x[0-9a-fA-F]+)\s*", line)
        if match is None:
            continue
        pc = int(match.group(1), 16)
        pcs.append(pc)
    repeated_pc = _repeated_pc_cycle(tuple(pcs), repeats=repeats)
    if repeated_pc is None or pc_path_in_ranges((repeated_pc,), executable_loads):
        return None
    return repeated_pc


def _trace_pc_cycle_loop(
    path: Path | None, *, repeats: int = 32, max_period: int = 64,
    ignored_ranges: tuple[tuple[int, int], ...] = (),
) -> int | None:
    """Return a short repeated PC cycle, including cycles inside the ELF."""
    if path is None or repeats < 1 or max_period < 1:
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
    pcs = tuple(
        int(match.group(1), 16)
        for line in lines
        for match in (re.fullmatch(rb"\s*INS\s+(0x[0-9a-fA-F]+)\s*", line),)
        if match is not None
    )
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


def _read_pc_trace(
    rsp: RspClient,
    *,
    normalize_jalr_bit0: bool = False,
    unaligned_pcs: list[int] | None = None,
) -> tuple[int, ...]:
    """Read the recorded executed-PC sequence through qRcmd pages.

    An overflow marker cannot identify the final occurrence of a repeated PC,
    so it is not a complete path.
    """
    try:
        count_reply = _qrcmd(rsp, "rvvm:pc-trace count", timeout=None)
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        raise _TraceReadError(f"RVVM pc-trace count read failed: {exc}", ()) from exc
    count_parts = count_reply.strip().split()
    overflow = len(count_parts) == 2 and count_parts[1] == "overflow"
    if len(count_parts) != 1 and not overflow:
        raise _TraceReadError(f"RVVM pc-trace count malformed: {count_reply!r}", ())
    try:
        count = int(count_parts[0])
    except ValueError:
        raise _TraceReadError(f"RVVM pc-trace count malformed: {count_reply!r}", ())
    if count < 0:
        raise _TraceReadError(f"RVVM pc-trace count out of range: {count_reply!r}", ())
    pcs: list[int] = []
    offset = 0
    while offset < count:
        try:
            page = _qrcmd(rsp, f"rvvm:pc-trace read {offset}", timeout=None)
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            raise _TraceReadError(
                f"RVVM pc-trace page read failed at {offset}: {exc}", tuple(pcs),
            ) from exc
        complete_length = len(page) - len(page) % 16
        if not page:
            raise _TraceReadError(f"RVVM pc-trace page empty at {offset}", tuple(pcs))
        for start in range(0, complete_length, 16):
            raw_pc = page[start : start + 16]
            try:
                pc = int(raw_pc, 16)
            except ValueError as exc:
                raise _TraceReadError(
                    f"RVVM pc-trace page malformed at record {len(pcs)}: {page!r}",
                    tuple(pcs),
                ) from exc
            if len(pcs) >= count:
                raise _TraceReadError(
                    f"RVVM pc-trace page exceeds declared count at {offset}: {page!r}",
                    tuple(pcs),
                )
            if pc & 1:
                if not normalize_jalr_bit0:
                    raise _TraceReadError(
                        f"RVVM pc-trace contains unaligned PC at {offset + len(pcs)}: {raw_pc}",
                        tuple(pcs),
                    )
                if unaligned_pcs is not None:
                    unaligned_pcs.append(pc)
                pc &= ~1
            pcs.append(pc)
        if len(page) != complete_length:
            raise _TraceReadError(
                f"RVVM pc-trace page has trailing data at {offset}: {page!r}",
                tuple(pcs),
            )
        if len(pcs) == offset:
            raise _TraceReadError(
                f"RVVM pc-trace page empty before declared end at {offset}: {page!r}",
                tuple(pcs),
            )
        offset = len(pcs)
    if len(pcs) != count:
        raise _TraceReadError(
            f"RVVM pc-trace length mismatch: {len(pcs)}!={count}", tuple(pcs),
        )
    if overflow:
        raise _TraceReadError(
            f"RVVM pc-trace overflow after {count} retained records: {count_reply!r}",
            tuple(pcs),
        )
    # The qRcmd recorder can repeat the initial PC at the capsule boundary,
    # just like the stepcov file recorder.  It is a recorder event, not a
    # second architectural instruction; keep observation paths comparable.
    normalized: list[int] = []
    for pc in pcs:
        if normalized and normalized[-1] == pc:
            continue
        normalized.append(pc)
    return tuple(normalized)


def _optional_pc_trace(
    rsp: RspClient, *, unaligned_pcs: list[int] | None = None,
) -> tuple[tuple[int, ...], str | None]:
    try:
        pcs = _read_pc_trace(
            rsp, normalize_jalr_bit0=True, unaligned_pcs=unaligned_pcs,
        )
    except _TraceReadError as exc:
        return tuple(int(pc) for pc in exc.prefix), f"RVVM executed-PC trace incomplete: {exc}"
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        return (), f"RVVM executed-PC trace incomplete: {exc}"
    return (pcs, None) if pcs else ((), "RVVM executed-PC trace is empty")


def _read_state_trace(rsp: RspClient) -> tuple[dict[str, int], ...]:
    try:
        count_reply = _qrcmd(rsp, "rvvm:state-trace count", timeout=None)
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        raise _TraceReadError(f"RVVM state-trace count read failed: {exc}", ()) from exc
    count_parts = count_reply.strip().split()
    overflow = len(count_parts) == 2 and count_parts[1] == "overflow"
    if len(count_parts) != 1 and not overflow:
        raise _TraceReadError(f"RVVM state-trace count malformed: {count_reply!r}", ())
    try:
        count = int(count_parts[0])
    except ValueError as exc:
        raise _TraceReadError(f"RVVM state-trace count malformed: {count_reply!r}", ()) from exc
    if count < 0:
        raise _TraceReadError(f"RVVM state-trace count out of range: {count_reply!r}", ())
    entries: list[dict[str, int]] = []
    offset = 0
    while offset < count:
        try:
            page = _qrcmd(rsp, f"rvvm:state-trace read {offset}", timeout=None)
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            raise _TraceReadError(
                f"RVVM state-trace page read failed at {offset}: {exc}", tuple(entries),
            ) from exc
        rows = page.split("|")
        for row_index, row in enumerate(rows):
            if row_index >= _STATE_TRACE_PAGE or len(entries) >= count:
                raise _TraceReadError(
                    f"RVVM state-trace page exceeds declared page/count at {offset}: {page!r}",
                    tuple(entries),
                )
            values = row.split(";")
            if len(values) != len(_STATE_TRACE_FIELDS) or any(len(value) != 16 for value in values):
                raise _TraceReadError(
                    f"RVVM state-trace row malformed at record {len(entries)}: {row!r}",
                    tuple(entries),
                )
            try:
                entry = {
                    name: int(value, 16)
                    for name, value in zip(_STATE_TRACE_FIELDS, values)
                }
            except ValueError as exc:
                raise _TraceReadError(
                    f"RVVM state-trace row malformed at record {len(entries)}: {row!r}",
                    tuple(entries),
                ) from exc
            if entry["reservation.valid"] not in (0, 1):
                raise _TraceReadError(
                    f"RVVM state-trace reservation.valid out of range at record {len(entries)}: {row!r}",
                    tuple(entries),
                )
            entries.append(entry)
        if not page:
            raise _TraceReadError(
                f"RVVM state-trace page empty before declared end at {offset}", tuple(entries),
            )
        offset = len(entries)
    if len(entries) != count:
        raise _TraceReadError(
            "RVVM state-trace length does not match declared count: "
            f"{len(entries)}!={count}", tuple(entries),
        )
    if overflow:
        raise _TraceReadError(
            f"RVVM state-trace overflow after {count} retained records: {count_reply!r}",
            tuple(entries),
        )
    return tuple(entries)


def _run(
    elf: Path,
    mailbox: int,
    observation_size: int,
    rvvm: Path,
    *,
    source: Path = DEFAULT_RVVM_SOURCE,
    jit: bool = True,
    harts: int = 1,
    trace_path: Path | None = None,
) -> tuple[bytes, dict[str, object], tuple[int, ...]]:
    firmware = elf
    if not firmware.is_file():
        raise RuntimeError(f"missing capsule ELF: {firmware}")
    if observation_size < OBSERVATION_HEADER_SIZE:
        raise ValueError("RVVM RVOBS1 frame is too small")
    if not is_riscv_elf(elf):
        raise ValueError("expected RISC-V ELF")
    elf_load_segments(elf)
    executable_loads = elf_executable_loads(elf)
    ignored_trace_ranges = _observer_copy_loop_ranges(elf)
    profile = capsule_cpu_profile(str(
        os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or os.environ.get("RV_TESTCASE_ISA_PROFILE", "")
    ).strip().lower())
    if profile is None:
        raise RuntimeError(
            "RV_TESTCASE_ISA_PROFILE is required and must select an rv32/rv64 capsule"
        )
    if not rvvm.is_file():
        raise RuntimeError(f"missing RVVM executable: {rvvm}")
    rvvm_binary_sha256 = sha256_file(rvvm)
    target_identity = _target_identity(rvvm, rvvm_binary_sha256)
    if target_identity is None:
        raise RuntimeError("RVVM target identity is missing, unverified, or source-patched")
    if target_identity.source_repository:
        target_source = Path(target_identity.source_repository)
        if target_source.is_dir():
            source = target_source
    source_clean = git_worktree_clean(source) if source.is_dir() else None
    deadline_value = os.environ.get("RQ1_BACKEND_DEADLINE_MONOTONIC")
    if deadline_value:
        deadline = float(deadline_value)
    else:
        try:
            backend_budget = float(os.environ.get("RQ1_BACKEND_TIMEOUT_SECONDS", "300"))
        except ValueError:
            backend_budget = 300.0
        deadline = time.monotonic() + max(0.001, backend_budget)
    proc: subprocess.Popen[bytes] | None = None
    rsp: RspClient | None = None
    phase = "RVVM process launch"
    _report_stage(phase)
    try:
        last_error: Exception | None = None
        for _ in range(2):
            if deadline is not None and time.monotonic() >= deadline:
                if last_error is None:
                    last_error = TimeoutError("RVVM backend execution deadline expired")
                break
            rsp = None
            try:
                if proc is not None and proc.poll() is not None:
                    proc = None
                if proc is None:
                    with _PORT_STARTUP_LOCK:
                        port = free_port()
                        command = _rvvm_command(
                            rvvm, firmware, profile, harts, port, jit,
                            stepcov=os.environ.get("RQ1_ENABLE_COVERAGE")
                                in {"1", "true", "yes"},
                            trace_path=trace_path,
                        )
                        env = os.environ.copy()
                        env["LD_LIBRARY_PATH"] = os.pathsep.join(
                            filter(None, (str(rvvm.parent), env.get("LD_LIBRARY_PATH")))
                        )
                        gdbcompat = rvvm.with_name("gdbcompat.so")
                        if gdbcompat.is_file():
                            env["LD_PRELOAD"] = os.pathsep.join(
                                filter(None, (str(gdbcompat), env.get("LD_PRELOAD")))
                            )
                        stop_flush = Path(env.get("RQ1_RVVM_STOP_FLUSH", ""))
                        if stop_flush.is_file():
                            env["LD_PRELOAD"] = os.pathsep.join(
                                filter(None, (str(stop_flush), env.get("LD_PRELOAD")))
                            )
                        proc = subprocess.Popen(
                            command,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            env=env,
                            start_new_session=os.name == "posix",
                        )
                        phase = "GDB connection"
                        _report_stage(phase)
                        rsp = _connect(port, proc, deadline)
                else:
                    phase = "GDB reconnection"
                    _report_stage(phase)
                    rsp = _connect(port, proc, deadline)
                phase = "GDB pause"
                _report_stage(phase)
                halt = rsp.request("?")
                if not halt.startswith(("S", "T", "X")):
                    raise RuntimeError(f"RVVM did not pause for ELF setup: {halt!r}")
                # RVVM may send a second stop packet while answering ``?``.
                # Consume it before the first register query; otherwise the
                # pending stop is mistaken for a hexadecimal ``g`` packet.
                _drain_pending_stop_replies(rsp)
                def setup_request(payload: str) -> str:
                    reply = rsp.request(payload)
                    for _ in range(_MAX_STOP_REPLY_DRAIN):
                        if not reply.startswith(("S", "T", "X")):
                            return reply
                        reply = rsp.read_reply()
                    raise RuntimeError("RVVM ELF setup stop-reply drain exceeded limit")

                coverage_requested = os.environ.get("RQ1_ENABLE_COVERAGE") in {
                    "1", "true", "yes",
                }
                bulk_trace = trace_path is not None
                trace_error = (
                    None if not coverage_requested else
                    None if bulk_trace else
                    None if _trace_supported(rsp, "pc") else
                    "pc-trace-channel-unavailable"
                )
                trace_supported = (
                    coverage_requested and not bulk_trace and trace_error is None
                )
                stepcov_supported = False
                executed_pcs: tuple[int, ...] = ()
                unaligned_trace_pcs: list[int] = []
                state_trace: tuple[dict[str, int], ...] = ()
                state_trace_error: str | None = None
                initial_frame = None
                if os.environ.get("RVVM_BARE_METAL_CAPSULE") == "1":
                    # Bare capsules finalize RVOBS1 before opening the GDB
                    # socket.  Keep the stopped guest state and consume that
                    # frame instead of resetting PC to the ELF entry point.
                    phase = "initial register and mailbox read"
                    _report_stage(phase)
                    initial_registers = _request_without_stop(rsp, "g")
                    register_hex_width = 16 if profile.startswith("rv64") else 8
                    if len(initial_registers) < 33 * register_hex_width:
                        raise RuntimeError("RVVM GDB register packet is incomplete")
                    initial_raw = _read_mailbox(
                        rsp, mailbox, observation_size, "RVVM",
                    )
                    initial_checkpoint = (
                        int.from_bytes(initial_raw[8:16], "little")
                        if len(initial_raw) >= 16 else 0
                    )
                    if initial_raw.startswith(OBSERVATION_MAGIC) and initial_checkpoint != 0:
                        initial_frame = initial_raw, initial_registers
                trace_guard_pc: list[int] = []
                trace_guard_reason: list[str] = []
                if initial_frame is None:
                    phase = "ELF register setup"
                    _report_stage(phase)
                    set_elf_entry_pc(firmware, profile, setup_request)
                    if coverage_requested and not trace_supported and not bulk_trace:
                        stepcov_supported = (
                            _request_without_stop(rsp, "qRQ1SingleStep") == "OK"
                        )
                    if stepcov_supported:
                        phase = "RVOBS1 step trace"
                        _report_stage(phase)
                        raw, registers, executed_pcs = _run_stepcov_until_frame(
                            rsp, mailbox, observation_size, profile,
                        )
                        trace_error = None
                        stop_reply = ""
                    else:
                        phase = "RVOBS1 frame read"
                        _report_stage(phase)
                        trace_guard_stop = threading.Event()
                        trace_guard_thread = None
                        if bulk_trace and trace_path is not None:
                            def watch_invalid_pc_loop() -> None:
                                while not trace_guard_stop.is_set() and proc is not None \
                                        and proc.poll() is None:
                                    invalid_pc = _outside_elf_trace_loop(
                                        trace_path, executable_loads,
                                    )
                                    if invalid_pc is not None:
                                        trace_guard_pc.append(invalid_pc)
                                        trace_guard_reason.append(
                                            "guest-pc-outside-elf-loop"
                                        )
                                        try:
                                            rsp.sock.sendall(b"\x03")
                                        except OSError:
                                            pass
                                        return
                                    # A direct RVGEN capsule is a finite
                                    # straight-line witness.  Some failures
                                    # rotate through an in-ELF trap/dispatch
                                    # block, so the outside-ELF check alone is
                                    # insufficient to release the GDB continue.
                                    if os.environ.get("RV_TESTCASE_ISA_PROFILE_NAME"):
                                        repeated_pc = _trace_pc_cycle_loop(
                                            trace_path,
                                            ignored_ranges=ignored_trace_ranges,
                                        )
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
                                name="rq1-rvvm-invalid-pc-watch", daemon=True,
                            )
                            trace_guard_thread.start()
                        try:
                            raw, registers, stop_reply = _read_finalized_frame(
                                rsp, mailbox, observation_size, profile,
                                clear_trace=trace_supported,
                            )
                        finally:
                            trace_guard_stop.set()
                            if trace_guard_thread is not None:
                                trace_guard_thread.join(timeout=1)
                else:
                    raw, registers = initial_frame
                    stop_reply = ""
                checkpoint_pc = int.from_bytes(raw[8:16], "little") if len(raw) >= 16 else 0
                mailbox_ready = raw.startswith(OBSERVATION_MAGIC) and checkpoint_pc != 0
                stop_trap = _guest_trap_from_stop_reply(stop_reply, registers, profile)
                mailbox_trap = (
                    _guest_trap_from_mailbox(raw, profile)
                    if mailbox_ready and os.environ.get("RV_TESTCASE_EXPECTED_TRAP") == "1"
                    else {}
                )
                if os.environ.get("RV_TESTCASE_EXPECTED_TRAP") == "1" and mailbox_ready and not mailbox_trap:
                    try:
                        phase = "trap state read"
                        mailbox_trap = _guest_trap_from_architectural_state(
                            _read_architectural_state(rsp)
                        )
                    except (RuntimeError, ValueError):
                        mailbox_trap = {}
                    if not mailbox_trap:
                        raise RuntimeError("RVVM trap mailbox marker is incomplete")
                if trace_supported:
                    phase = "PC/state trace read"
                    _report_stage(phase)
                    if harts > 1 and mailbox_ready:
                        _halt_all_harts(rsp)
                    executed_pcs, trace_error = _optional_pc_trace(
                        rsp, unaligned_pcs=unaligned_trace_pcs,
                    )
                    if _trace_supported(rsp, "state"):
                        try:
                            state_trace = tuple(
                                {**entry, "pc": entry["pc"] & ~1}
                                for entry in _read_state_trace(rsp)
                            )
                        except (RuntimeError, ConnectionError, TimeoutError) as exc:
                            state_trace_error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, _TraceReadError):
                                state_trace = tuple(exc.prefix)
                    else:
                        state_trace_error = "state-trace-channel-unavailable"
                    if state_trace and len(state_trace) != len(executed_pcs):
                        state_trace_error = (
                            "state-trace-length-mismatch:"
                            f"{len(state_trace)}!={len(executed_pcs)}"
                        )
                if bulk_trace:
                    executed_pcs, trace_error = _read_bulk_pc_trace(trace_path)
                    if trace_guard_pc and not mailbox_ready:
                        trace_error = (
                            f"{trace_guard_reason[0] if trace_guard_reason else 'guest-pc-loop'}:"
                            f"0x{trace_guard_pc[0]:x}"
                        )
                raw_executed_pcs = executed_pcs
                executed_pcs, shim_pcs, shim_address = _strip_ecall_shim(elf, raw_executed_pcs)
                if state_trace and shim_pcs and len(state_trace) == len(raw_executed_pcs):
                    address = shim_address
                    if address is not None:
                        state_trace = tuple(
                            entry for pc, entry in zip(raw_executed_pcs, state_trace)
                            if not (
                                address <= pc < address + _RVVM_ECALL_SHIM_SIZE
                                or address <= entry["pc"] < address + _RVVM_ECALL_SHIM_SIZE
                            )
                        )
                if state_trace and len(state_trace) != len(executed_pcs):
                    state_trace_error = (
                        "state-trace-length-mismatch-after-shim-filter:"
                        f"{len(state_trace)}!={len(executed_pcs)}"
                    )
                if state_trace and len(state_trace) == len(executed_pcs):
                    state_trace = tuple(
                        {**entry, "after_pc": entry["pc"], "pc": pc}
                        for pc, entry in zip(executed_pcs, state_trace)
                    )
                if not mailbox_ready:
                    trace_error = trace_error or (
                        "RVVM mailbox frame has no finalized checkpoint"
                        if raw.startswith(OBSERVATION_MAGIC)
                        else "RVVM mailbox never exposed a finalized RVOBS1 frame"
                    )
                trap_trace = (
                    bool(state_trace)
                    and type(state_trace[-1].get("csr.mcause")) is int
                    and state_trace[-1]["csr.mcause"] < (1 << 63)
                    and state_trace[-1]["csr.mcause"] != 0
                )
                if executed_pcs:
                    try:
                        capsule_loads = elf_executable_loads(elf)
                    except (OSError, ValueError):
                        trace_error = trace_error or "RVVM capsule ELF cannot validate PC trace"
                        executed_pcs = ()
                    else:
                        if not pc_path_in_ranges(executed_pcs, capsule_loads):
                            first_outside = next(
                                (index for index, pc in enumerate(executed_pcs)
                                 if not pc_path_in_ranges((pc,), capsule_loads)),
                                len(executed_pcs),
                            )
                            terminal_pcs = tuple(executed_pcs[first_outside:])
                            trap_vector = (
                                state_trace[-1].get("csr.mtvec", 0) & ~3
                                if state_trace else 0
                            )
                            if (
                                trap_trace
                                and terminal_pcs
                                and all(pc in {0, trap_vector} for pc in terminal_pcs)
                            ):
                                executed_pcs = executed_pcs[:first_outside]
                            else:
                                trace_error = "RVVM PC trace leaves capsule executable image"
                                executed_pcs = ()
                phase = "final FP state read"
                _report_stage(phase)
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
                        "mstatus_fs_gap": f"rvvm-fp-observer-unavailable: {exc}",
                    }
                phase = "final architectural state read"
                _report_stage(phase)
                try:
                    architectural_state = _read_architectural_state(rsp)
                except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
                    architectural_state = {
                        "status": "gap",
                        "error": f"final-state-unavailable: {exc}",
                        "fields": {},
                    }
                if (
                    architectural_state.get("status") != "observed"
                    and state_trace
                    and state_trace_error is None
                    and len(state_trace) == len(executed_pcs)
                    and executed_pcs
                    and state_trace[-1].get("pc") == executed_pcs[-1]
                ):
                    architectural_state = _architectural_state_from_trace(
                        state_trace[-1], evidence="rvvm-state-trace-final"
                    )
                path_pcs = executed_pcs if mailbox_ready or architectural_state.get("status") == "observed" else ()
                trace_available = bool(path_pcs)
                rvvm_source_commit = (
                    git_head(source) if source_clean is True
                    else target_identity.source_commit if target_identity is not None else None
                )
                firmware_sha256 = sha256_file(firmware)
                capsule_sha256 = sha256_file(elf)
                path_identity = _path_identity_witness(path_pcs, jit=jit)
                configuration = _configuration_identity(
                    profile=profile,
                    source_commit=rvvm_source_commit,
                    rvvm_sha256=rvvm_binary_sha256,
                    firmware_sha256=firmware_sha256,
                    capsule_sha256=capsule_sha256,
                    mailbox=mailbox,
                    observation_size=observation_size,
                    jit=jit,
                    trace=(
                        "file:rvvm:instruction-pc"
                        if bulk_trace else
                        "gdb-rsp:rvvm:pc-trace"
                        if trace_supported
                        else "gdb-rsp:rvvm:qRQ1SingleStep"
                        if stepcov_supported else "gdb-rsp:mailbox-only"
                    ),
                    capsule_mode=(
                        "bare-metal-no-ecall"
                        if os.environ.get("RVVM_BARE_METAL_CAPSULE") == "1"
                        else "linux-user-ecall"
                    ),
                    target_identity=target_identity,
                )
                if rvvm_source_commit is None:
                    configuration["identity_digest"] = None
                return (raw if mailbox_ready else b""), {
                    **stop_trap,
                    **mailbox_trap,
                    "source": "rvvm-cli-gdb-rsp",
                    "translation_path": "jit" if jit else "interpreter",
                    "capsule_mode": (
                        "bare-metal-no-ecall"
                        if os.environ.get("RVVM_BARE_METAL_CAPSULE") == "1"
                        else "linux-user-ecall"
                    ),
                    "hart_count": harts,
                    "rvvm_source_commit": rvvm_source_commit,
                    "rvvm_source_clean": source_clean,
                    "target_identity": (
                        target_identity.to_dict() if target_identity is not None else None
                    ),
                    "rvvm_binary_sha256": rvvm_binary_sha256,
                    "firmware_sha256": firmware_sha256,
                    "dependency_family": backend_spec("rvvm-riscv64").dependency_family,
                    "gpr_packet_bytes": len(registers) // 2,
                    "trace_available": trace_available,
                    # The bind port is transport-only; omit it so execution
                    # identity reflects the target setup rather than a host socket.
                    "trace_status": (
                        "partial" if trace_available and trace_error else
                        "observed" if trace_available else
                        "unavailable" if trace_error == "pc-trace-channel-unavailable"
                        else "gap"
                    ),
                    "rvvm_trace_complete": bool(trace_available and trace_error is None),
                    "trace_gap": (
                        None
                        if trace_available
                        else (
                            trace_error
                            or "empty-executed-pc-trace"
                        )
                    ),
                    "observer_gaps": {
                        **({"state_trace": state_trace_error} if state_trace_error else {}),
                        **({
                            "csr": architectural_state.get("error"),
                            "reservation": architectural_state.get("error"),
                        } if architectural_state.get("status") != "observed" else {}),
                        **({"mailbox": trace_error} if not mailbox_ready else {}),
                    },
                    "mailbox_status": "ready" if mailbox_ready else "gap",
                    # The finalized RVOBS1 frame is written before the
                    # wrapper's terminating Linux ecall.  RVVM therefore
                    # reports mcause=11 for a completed capsule; the frame,
                    # not that terminal ecall, decides whether execution
                    # completed.
                    "observer_complete": mailbox_ready,
                    "exit_code": 0 if mailbox_ready else 1,
                    # RVOBS1 is the required observation.  PC/state and FP
                    # GDB channels are optional; their absence is recorded
                    # without invalidating a complete mailbox frame.
                    "rvvm_trace_mechanism": (
                        "stepcov-instruction-trace-file"
                        if bulk_trace else
                        "gdbstub-qRcmd-pc-trace"
                        if trace_supported
                        else "gdbstub-qRQ1SingleStep-pc"
                        if stepcov_supported else "gdb-rsp-mailbox-only"
                    ),
                    "rvvm_stepcov_supported": stepcov_supported,
                    "trace_path": str(trace_path) if trace_path is not None else None,
                    "trace_records": len(executed_pcs),
                    "trace_size_bytes": (
                        trace_path.stat().st_size
                        if trace_path is not None and trace_path.is_file() else 0
                    ),
                    "trace_sha256": (
                        sha256_file(trace_path)
                        if trace_path is not None and trace_path.is_file() else None
                    ),
                    "rvvm_executed_pc_count": len(path_pcs),
                    "rvvm_executed_pc_digest": path_identity["digest"],
                    "path_identity": path_identity,
                    "configuration_identity": configuration,
                    "rvvm_trace_error": trace_error,
                    "rvvm_unaligned_pc_trace": [hex(pc) for pc in unaligned_trace_pcs],
                    "state_trace": state_trace,
                    "state_trace_status": (
                        "partial" if state_trace and state_trace_error else
                        "observed" if state_trace else
                        "unavailable" if not trace_supported
                        or state_trace_error == "state-trace-channel-unavailable"
                        else "gap"
                    ),
                    "state_trace_error": state_trace_error,
                    "state_trace_fields": list(_STATE_TRACE_FIELDS),
                    "rvvm_invalid_pc_loop": (
                        hex(trace_guard_pc[0]) if trace_guard_pc else None
                    ),
                    "rvvm_ecall_shim": {
                        "filtered_pc_count": shim_pcs,
                        "observer": "rvvm-wrapper-machine-ecall-skip",
                    },
                    "fpr_available": fp_state.get("fp_observer") == "observed",
                    "fp_state": fp_state,
                    "architectural_state": architectural_state,
                    **stop_trap,
                    **mailbox_trap,
                }, path_pcs
            except (ConnectionError, TimeoutError, RuntimeError) as exc:
                last_error = RuntimeError(f"{phase}: {type(exc).__name__}: {exc}")
                if rsp is not None:
                    rsp.close()
                    rsp = None
                if proc is not None:
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        _stdout, stderr = proc.communicate(timeout=3)
                    except (OSError, subprocess.TimeoutExpired):
                        # A broken GDB session must not leave a child behind
                        # or mask the original connection error.
                        try:
                            proc.wait(timeout=1)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                        stderr = b""
                    if stderr:
                        last_error = RuntimeError(
                            f"{exc}; RVVM stderr: {stderr.decode(errors='replace').strip()}"
                        )
                    proc = None
        raise RuntimeError(f"RVVM GDB session failed after retry: {last_error}")
    finally:
        if rsp is not None:
            try:
                rsp.request("k")
            except (ConnectionError, OSError, TimeoutError, RuntimeError):
                pass
            rsp.close()
            rsp = None
        if proc is not None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            except (OSError, ProcessLookupError):
                pass
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
                    proc.communicate(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass


def _read_finalized_frame(
    rsp: RspClient,
    mailbox: int,
    observation_size: int,
    profile: str,
    *,
    clear_trace: bool = False,
) -> tuple[bytes, str, str]:
    """Read a finalized RVOBS1 mailbox frame from one RVVM GDB session.

    The capsule image ships ``result_buffer`` pre-initialized with the
    observation magic, so a magic match alone is not a completed frame: the
    checkpoint field stays zero until ``exit_checkpoint`` writes it.  Stop,
    read registers, and read the mailbox; continue once if the frame is not
    finalized yet.
    """
    raw = b""
    registers = ""
    register_hex_width = 16 if profile.startswith("rv64") else 8
    for command in ("?", "c"):
        halt = rsp.request(command) if command == "?" else _continue_to_stop(rsp)
        if not halt.startswith(("S", "T", "X")):
            raise RuntimeError(f"RVVM did not stop on capsule ebreak: {halt}")
        _drain_pending_stop_replies(rsp)
        registers = rsp.request("g")
        if len(registers) < 33 * register_hex_width:
            raise RuntimeError("RVVM GDB register packet is incomplete")
        raw = _read_mailbox(rsp, mailbox, observation_size, "RVVM")
        checkpoint_pc = int.from_bytes(raw[8:16], "little") if len(raw) >= 16 else 0
        if raw.startswith(OBSERVATION_MAGIC) and checkpoint_pc != 0:
            return raw, registers, halt
        if halt[1:3].lower() == "04":
            return raw, registers, halt
        if command == "?" and clear_trace:
            _clear_trace(rsp)
    return raw if raw.startswith(OBSERVATION_MAGIC) else b"", registers, halt


def _continue_to_stop(rsp: RspClient) -> str:
    """Interrupt a stuck GDB continue, then let the mailbox decide readiness."""

    previous_timeout = rsp.sock.gettimeout()
    try:
        return rsp.request("c")
    except TimeoutError:
        rsp.sock.sendall(b"\x03")
        stop_reply = rsp.read_reply()
        rsp.sock.settimeout(0.2)
        # ponytail: 固定 64 个 stop 包上限；需要更大批量时再改为协议序号。
        for _ in range(_MAX_STOP_REPLY_DRAIN):
            try:
                reply = rsp.read_reply()
            except (ConnectionError, TimeoutError):
                break
            if reply.startswith(("S", "T", "X")):
                stop_reply = reply
        else:
            raise TimeoutError("RVVM stop reply drain exceeded limit")
        return stop_reply
    finally:
        rsp.sock.settimeout(previous_timeout)


def _halt_all_harts(rsp: RspClient) -> None:
    rsp.sock.sendall(b"\x03")
    halt = rsp.read_reply()
    if not halt.startswith(("S", "T", "X")):
        raise RuntimeError(f"RVVM multi-hart interrupt did not stop: {halt}")
    _drain_pending_stop_replies(rsp)


def _request_without_stop(rsp: RspClient, payload: str) -> str:
    """Read a data reply when RVVM also queues a stop notification."""
    reply = rsp.request(payload)
    for _ in range(_MAX_STOP_REPLY_DRAIN):
        if not reply.startswith(("S", "T", "X")):
            return reply
        reply = rsp.read_reply()
    raise RuntimeError(f"RVVM {payload!r} reply kept returning stop packets")


def _request_single_step(rsp: RspClient) -> str:
    """Execute one guest instruction through the existing stepcov GDB hook."""
    reply = rsp.request("s")
    for _ in range(_MAX_STOP_REPLY_DRAIN):
        if reply.startswith(("S", "T", "X")):
            return reply
        reply = rsp.read_reply()
    raise RuntimeError("RVVM single-step did not return a stop packet")


def _run_stepcov_until_frame(
    rsp: RspClient,
    mailbox: int,
    observation_size: int,
    profile: str,
) -> tuple[bytes, str, tuple[int, ...]]:
    """Collect PCs before each GDB step until the capsule seals RVOBS1.

    The stepcov build starts executing before a client can attach, so the
    qRcmd trace channel cannot recover short capsules after the fact.  Its
    explicit ``qRQ1SingleStep`` capability lets us stop at each instruction;
    this loop is the smallest adapter for that existing protocol.
    """
    register_hex_width = 16 if profile.startswith("rv64") else 8
    pcs: list[int] = []
    while True:
        registers = _request_without_stop(rsp, "g")
        if len(registers) < 33 * register_hex_width:
            raise RuntimeError("RVVM GDB register packet is incomplete during stepcov")
        raw_registers = bytes.fromhex(registers[:33 * register_hex_width])
        pc_offset = 32 * (8 if profile.startswith("rv64") else 4)
        pc_width = 8 if profile.startswith("rv64") else 4
        pc = int.from_bytes(raw_registers[pc_offset:pc_offset + pc_width], "little")
        if pc & 1:
            raise RuntimeError("RVVM stepcov PC is unaligned")
        pcs.append(pc)
        stop = _request_single_step(rsp)
        header = _read_mailbox(rsp, mailbox, OBSERVATION_HEADER_SIZE, "RVVM")
        checkpoint_pc = int.from_bytes(header[8:16], "little") if len(header) >= 16 else 0
        if header.startswith(OBSERVATION_MAGIC) and checkpoint_pc != 0:
            raw = _read_mailbox(rsp, mailbox, observation_size, "RVVM")
            if raw.startswith(OBSERVATION_MAGIC):
                return raw, _request_without_stop(rsp, "g"), tuple(pcs)
        if not stop.startswith(("S", "T", "X")):
            raise RuntimeError(f"RVVM single-step returned malformed stop: {stop!r}")


# RVVM's expanded GDB stub (src/core/gdbstub.c, GDB_FREG_BASE=33,
# GDB_FFLAGS_REG=65, GDB_FRM_REG=66) follows the RISC-V GDB register map:
# x0-x31 = 0-31, pc = 32, f0-f31 = 33-64, fflags = 65, frm = 66.
_RVVM_FPR_FIRST = 33
_RVVM_FFLAGS = 65
_RVVM_FRM = 66


def _read_fp_state(rsp: RspClient, profile: str) -> dict[str, object]:
    """Read the real FPR/fflags/frm through GDB RSP single-register packets.

    Uses the common capsule contract: when the executing profile requires
    FP state, every `pNN` read must return an 8-byte little-endian value, and
    fflags/frm must stay inside their architectural ranges.  Otherwise the
    named gap is returned so the oracle never mistakes absent observation for
    architectural zeros.
    """
    if (
        not profile_requires_fp_state(profile)
        or os.environ.get("RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "1") != "1"
    ):
        return {
            "fp_observer": "not-required",
            "mstatus_fs": None,
            "mstatus_fs_observer": "not-required",
            "mstatus_fs_gap": None,
        }
    hex_digits = frozenset("0123456789abcdefABCDEF")
    rawbits = []
    for index in range(32):
        reply = rsp.request(f"p{_RVVM_FPR_FIRST + index:x}")
        if reply.startswith("E") or len(reply) != 16:
            raise RuntimeError(f"RVVM GDB FPR read failed for f{index}: {reply}")
        if any(char not in hex_digits for char in reply):
            raise RuntimeError(f"RVVM GDB FPR read malformed for f{index}: {reply}")
        try:
            rawbits.append(int.from_bytes(bytes.fromhex(reply), "little"))
        except ValueError as exc:
            raise RuntimeError(f"RVVM GDB FPR read malformed for f{index}: {reply}") from exc
    fflags_reply = rsp.request(f"p{_RVVM_FFLAGS:x}")
    frm_reply = rsp.request(f"p{_RVVM_FRM:x}")
    if (
        fflags_reply.startswith("E")
        or frm_reply.startswith("E")
        or len(fflags_reply) not in {8, 16}
        or len(frm_reply) not in {8, 16}
    ):
        raise RuntimeError(
            "RVVM GDB fflags/frm read failed: "
            f"fflags={fflags_reply!r} frm={frm_reply!r}"
        )
    if any(char not in hex_digits for char in fflags_reply + frm_reply):
        raise RuntimeError("RVVM GDB fflags/frm reply is malformed")
    # RVVM's stub replies with a 4-byte LE value for fflags/frm (8 hex chars),
    # not an 8-byte GPR-sized value; padding to 16 chars would shift the
    # payload into the high 32 bits and decode as 0x1_0000_0000 when flags
    # are 0x1.
    try:
        fflags = int.from_bytes(bytes.fromhex(fflags_reply), "little")
        frm = int.from_bytes(bytes.fromhex(frm_reply), "little")
    except ValueError as exc:
        raise RuntimeError("RVVM GDB fflags/frm reply is malformed") from exc
    if fflags > 0x1F or frm > 0x7:
        raise RuntimeError(f"RVVM FP flags out of range: fflags={fflags:#x} frm={frm:#x}")
    # Explicit observer marker: main() otherwise defaults a successful read
    # to "not-observed", which makes the strict oracle treat real FPR/flags
    # evidence as a declared gap and drops the FP compare entirely.
    return {
        "fpr_rawbits": rawbits,
        "fflags": fflags,
        "frm": frm,
        "fp_observer": "observed",
        "mstatus_fs": None,
        "mstatus_fs_observer": "gap",
        "mstatus_fs_gap": "rvvm-gdbstub-no-csr-readback",
    }


def _read_architectural_state(rsp: RspClient) -> dict[str, object]:
    """Read final CSR, privilege and reservation state from RVVM."""
    reply = _qrcmd(rsp, "rvvm:state")
    if reply.startswith("E"):
        raise RuntimeError(f"RVVM state read failed: {reply}")
    raw: dict[str, int] = {}
    for item in reply.split(";"):
        if not item:
            continue
        key, separator, value = item.partition("=")
        if not separator or not key or len(value) != 16:
            raise ValueError(f"RVVM state reply is malformed: {item!r}")
        try:
            raw[key] = int.from_bytes(bytes.fromhex(value), "little")
        except ValueError as exc:
            raise ValueError(f"RVVM state value is malformed: {item!r}") from exc
    return _architectural_state_from_trace(raw)


def _architectural_state_from_trace(
    entry: dict[str, int], *, evidence: str | None = None,
) -> dict[str, object]:
    raw = dict(entry)
    required = {
        "priv_mode", "csr.status", "csr.mepc", "csr.mcause", "csr.mtval",
        "csr.mtvec", "reservation.valid", "reservation.addr", "reservation.value",
    }
    if not required <= raw.keys():
        raise ValueError(f"RVVM state fields are incomplete: {sorted(required - raw.keys())}")
    if raw["reservation.valid"] not in (0, 1):
        raise ValueError("RVVM reservation.valid is out of range")
    if "after_pc" in entry and "pc" in raw:
        raw["pc"] = entry["after_pc"]
    fields = {
        "csr.priv": raw["priv_mode"], "csr.priv_observer": "observed",
        "csr.status": raw["csr.status"], "csr.status_observer": "observed",
        "csr.mepc": raw["csr.mepc"], "csr.mepc_observer": "observed",
        "csr.mcause": raw["csr.mcause"], "csr.mcause_observer": "observed",
        "csr.mtval": raw["csr.mtval"], "csr.mtval_observer": "observed",
        "csr.mtvec": raw["csr.mtvec"], "csr.mtvec_observer": "observed",
        "reservation.valid": bool(raw["reservation.valid"]),
        "reservation.valid_observer": "observed",
        "reservation.address": raw["reservation.addr"],
        "reservation.address_observer": "observed",
        "reservation.result": raw["reservation.value"],
        "reservation.result_observer": "observed",
    }
    for source, target in {
        "csr.ie": "csr.ie", "csr.ip": "csr.ip", "csr.isa": "csr.isa",
        "csr.fcsr": "csr.fcsr", "csr.vcsr": "csr.vcsr", "csr.vtype": "csr.vtype",
    }.items():
        if source in raw:
            fields[target] = raw[source]
            fields[f"{target}_observer"] = "observed"
    result = {"status": "observed", "raw": raw, "fields": fields}
    if evidence is not None:
        result["evidence"] = evidence
    return result


def _guest_trap_from_architectural_state(
    architectural_state: dict[str, object],
) -> dict[str, object]:
    if architectural_state.get("status") != "observed":
        return {}
    raw = architectural_state.get("raw", {})
    if not isinstance(raw, dict) or type(raw.get("csr.mcause")) is not int:
        return {}
    raw_cause = int(raw["csr.mcause"])
    if raw_cause & (1 << 63):
        return {}
    cause = raw_cause & ((1 << 63) - 1)
    if type(raw.get("csr.mepc")) is not int or type(raw.get("csr.mtval")) is not int:
        return {}
    return {
        "guest_trap": "delivered",
        "guest_cause": cause,
        "guest_epc": int(raw["csr.mepc"]),
        "guest_tval": int(raw["csr.mtval"]),
        "guest_tval_observer": "rvvm-state-csr",
        "trap_observer": "rvvm-state-csr",
        "fault_pc": int(raw["csr.mepc"]),
    }


def _guest_trap_from_mailbox(raw: bytes, profile: str) -> dict[str, object]:
    if not raw.startswith(OBSERVATION_MAGIC) or len(raw) < OBSERVATION_HEADER_SIZE:
        return {}
    values = {
        name: int.from_bytes(raw[16 + index * 8:24 + index * 8], "little")
        for name, index in (("guest_cause", 31), ("guest_epc", 30), ("guest_tval", 28))
    }
    if profile.startswith("rv32") and any(value >= 1 << 32 for value in values.values()):
        return {}
    if values["guest_cause"] >= 1 << 63 or values["guest_epc"] == 0:
        return {}
    return {
        "guest_trap": "delivered",
        **values,
        "guest_tval_observer": "rvvm-trap-handler-csr",
        "trap_observer": "rvvm-trap-handler-csr",
        "fault_pc": values["guest_epc"],
    }


def _guest_trap_from_stop_reply(
    stop_reply: str, registers: str, profile: str,
) -> dict[str, object]:
    """Preserve a guest exception reported by RVVM's GDB stop packet."""
    if not stop_reply.startswith("S") or len(stop_reply) < 3:
        return {}
    try:
        signal_number = int(stop_reply[1:3], 16)
    except ValueError:
        return {}
    # S05 is RVVM's normal debugger stop/breakpoint path, not evidence of an
    # architectural ebreak.  Only the unambiguous illegal-instruction stop is
    # promoted here; other faults still need CSR or mailbox evidence.
    cause = {4: 2}.get(signal_number)
    if cause is None:
        return {}
    width = 16 if profile.startswith("rv64") else 8
    if len(registers) < 33 * width:
        return {}
    try:
        raw = bytes.fromhex(registers[:33 * width])
    except ValueError:
        return {}
    pc_offset = 32 * (8 if profile.startswith("rv64") else 4)
    pc_width = 8 if profile.startswith("rv64") else 4
    guest_epc = int.from_bytes(raw[pc_offset:pc_offset + pc_width], "little")
    if guest_epc == 0 or guest_epc & 1:
        return {}
    return {
        "guest_trap": "delivered",
        "guest_cause": cause,
        "guest_epc": guest_epc,
        "guest_tval_observer": "unavailable",
        "trap_observer": "rvvm-gdb-stop",
        "fault_pc": guest_epc,
        "signal": signal.Signals(signal_number).name,
        "signal_code": signal_number,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a framework capsule through RVVM and its existing GDB stub")
    parser.add_argument("elf")
    parser.add_argument("--mailbox", default=os.environ.get("RV_CAPSULE_MAILBOX"))
    parser.add_argument("--observation-size", default=os.environ.get("RV_CAPSULE_OBSERVATION_SIZE"), type=int)
    parser.add_argument("--rvvm", default=os.environ.get("RVVM_BINARY", str(DEFAULT_RVVM)))
    parser.add_argument("--rvvm-source", default=os.environ.get("RVVM_SOURCE", str(DEFAULT_RVVM_SOURCE)))
    parser.add_argument("--no-jit", action="store_true", default=os.environ.get("RVVM_NO_JIT", "1") == "1")
    parser.add_argument("--harts", type=int, default=int(os.environ.get("RVVM_HARTS", "1")))
    args = parser.parse_args(argv)
    if args.mailbox is None or args.observation_size is None:
        return runner_contract_gap("capsule mailbox and observation size are required")
    jit = not args.no_jit
    trace_value = os.environ.get("RQ1_RVVM_TRACE_PATH", "").strip()
    trace_path = (
        Path(trace_value.replace("%p", str(os.getpid()))).resolve()
        if trace_value else None
    )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stdout, details, executed_pcs = _run(
            Path(args.elf).resolve(),
            int(args.mailbox, 0),
            args.observation_size,
            Path(args.rvvm).resolve(),
            source=Path(args.rvvm_source).resolve(),
            jit=jit,
            harts=args.harts,
            trace_path=trace_path,
        )
    except Exception as exc:
        return runner_contract_gap(f"RVVM runner failed: {type(exc).__name__}: {exc}")
    _report_stage("serialize-observation")
    sys.stdout.buffer.write(stdout)
    translation_path = details.get("translation_path", "jit")
    sys.stderr.write(f"RV_TOOL_VERSION=rvvm-cli-gdb-rsp:{translation_path}\n")
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
    fp_state = details.get("fp_state", {})
    observation_extra_state = dict(fp_state)
    volatile_gprs = tuple(
        int(value)
        for value in os.environ.get("RV_VOLATILE_GPR_INDICES", "").split(",")
        if value
    )
    if volatile_gprs:
        observation_extra_state["volatile_gpr_indices"] = volatile_gprs
    architectural_state = details.get("architectural_state", {})
    if isinstance(architectural_state, dict) and architectural_state.get("status") == "observed":
        observation_extra_state.update(architectural_state.get("fields", {}))
        observation_extra_state["state_observer"] = "observed"
        if details.get("mailbox_status") != "ready" or os.environ.get("RV_TESTCASE_EXPECTED_TRAP") == "1":
            for key, value in _guest_trap_from_architectural_state(architectural_state).items():
                details.setdefault(key, value)
    if details.get("state_trace"):
        observation_extra_state["state_trace"] = details["state_trace"]
    if observer_gaps := details.get("observer_gaps"):
        observation_extra_state["observer_gaps"] = observer_gaps
    if fp_state.get("adapter_execution_error"):
        observation_extra_state["adapter_execution_error"] = fp_state[
            "adapter_execution_error"
        ]
    for key in (*GUEST_TRAP_BRIDGE_FIELDS, "guest_tval_observer", "trap_observer"):
        if details.get(key) is not None:
            observation_extra_state[key] = details[key]
    if unaligned_pcs := details.get("rvvm_unaligned_pc_trace"):
        observation_extra_state.update(
            {"control.target": int(unaligned_pcs[0], 0),
             "control.target_observer": "rvvm-pc-trace-raw-unaligned"}
        )
    observation_state = {"extra_state": observation_extra_state}
    if details.get("signal") is not None:
        observation_state["signal"] = details["signal"]
        observation_state["signal_code"] = details.get("signal_code")
    for key in ("fault_pc", "fault_address"):
        if details.get(key) is not None:
            observation_state[key] = details[key]
    observation_extra_state["observer_fields"] = observed_state_fields(observation_extra_state)
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
                "backend": "rvvm-riscv64",
                "expected_path": "jit" if jit else "interpreter",
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

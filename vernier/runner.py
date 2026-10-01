#!/usr/bin/env python3
"""RQ1 对比实验的独立控制面；只依赖 Python 标准库和外部工具。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from analysis.elf_features import (
    DECODER_VERSION, behavior_signature, decode_elf, reference_coverage,
)
from framework.direct_case import (
    OBSERVATION_HEADER_SIZE,
    OBSERVATION_MAGIC,
    MAX_OBSERVATION_MEMORY_SIZE,
    _observation_frame_end,
)
from analysis.simulator_coverage import (
    BATCH_SCHEMA, SCHEMA, SOURCE_SCHEMA, aggregate_simulator_coverage,
    gate_experiment_unions, summarize_experiment_union, summarize_simulator_coverage,
    summarize_source_experiment_union, target_source_coverage,
)
from analysis.rv_instruction_coverage import (
    EXPERIMENT_COVERAGE_PROFILE,
    EXPERIMENT_PRIVILEGE_MODE,
    OPCODE_CATALOG_DENOMINATOR,
    OPCODE_CATALOG_FORM_COUNT,
    OPCODE_CATALOG_ID,
    OPCODE_CATALOG_PARTITION_KEYS,
    OPCODE_CATALOG_REGISTRY_SHA256,
    OPCODE_KEY_SCHEMA,
    aggregate_summary as aggregate_rv_instruction_summary,
    opcode_catalog_case_coverage,
    registry_for_profile as rv_registry_for_profile,
    summarize_opcode_catalog_target_rows,
)
from analysis.source_coverage import (
    GCOV_FLUSH_INCOMPLETE_MARKER, collect_source_coverage_batch,
    compress_dotnet_report, runtime_env,
)
from framework.framework_coverage import (
    _coverage_target, finalize_framework_coverage,
    finalize_framework_source_coverage_batch, merge_framework_coverage_summaries,
    merge_framework_raw_coverage_batches, wait_for_dotnet_coverage,
)
from framework.evidence.target_coverage import summarize_target_coverage_by_target
from framework.target_evidence import (
    compact_record, iter_materialized_records, make_evidence_ref,
    materialize_record,
)
from framework.execution_identity import is_allowlisted_patched_identity
from framework.adapters.program_observer import (
    _add_torture_observer, _localize_tohost, build_linux_user,
)
from framework.adapters.runner import _finalized_observation_frame, _observation_frame
from framework.adapters.rvvm_rsp import set_elf_entry_pc
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from decfuzzer_main_adapter import run as run_decfuzzer_main
from framework.direct_elf import symbol_offsets_from_elf
from framework.native_reference_gate import NATIVE_REFERENCE_PARALLELISM, _NativeReferenceLock
from comparison_observations import (
    DIFFERENTIAL_MATCH, DIFFERENTIAL_MISMATCH, DIFFERENTIAL_UNVERIFIED,
    _canonical_differential_status,
    _differential_record, _guest_terminal_evidence, _observation_record,
    _process_executed, _target_dispatch_event,
    _target_testable,
    _trusted_observation_outcome, _trusted_reference_board,
    _valid_commit, _valid_hex,
)
from comparison_artifacts import (
    digest, load_json, sha256, write_event_ledger,
    write_json, write_json_replace, write_text_once,
)
from comparison_state import (
    _event_is_expected_timebox, _event_is_pending, _load_partial_queue_entries,
    _execution_event_closed, _metric_ratio,
    _nonnegative_index, _partial_queue_counts, _queue_accounting, _target_rates, _tool_record,
)
from comparison_queue import (
    _generation_method_coverage_ok, _queue_candidate_identity_ok,
    _queue_candidate_indices_ok, _queue_entries_artifacts_ok,
)


if __name__ == "__main__":
    sys.modules.setdefault("runner", sys.modules[__name__])


HERE = Path(__file__).resolve().parent

_TRACE_INS_PC_RE = re.compile(rb"\bINS\s+(?:0x)?([0-9a-fA-F]+)\b")
_TRACE_PLAIN_PC_RE = re.compile(rb"^\s*0x([0-9a-fA-F]+)\s*$")
DEFAULT_CONFIG = HERE / "config" / "rq1-comparison-isolated-v1.json"
DEFAULT_RUN_SECONDS = 1800
DEFAULT_TARGET_TIMEOUT_SECONDS = 180
# GCOV/LLVM source profiles are converted in bounded 128-case batches.
# ReportGenerator branch IDs depend on its complete Cobertura input set, so
# .NET coverage stays in one method/target batch.
FRAMEWORK_COVERAGE_BATCH_SIZE = 128
_FRAMEWORK_DOTNET_COVERAGE_SERVERS: dict[str, dict[str, Any]] = {}
# Canonical Ours campaigns keep a bounded online MCMC chain.  The open
# capability change removes ISA/extension gates around that chain; it does not
# replace the framework's EMI or model feedback loop.  The value lives in
# framework.step_policy so entry points cannot drift independently.
DEPS = Path(os.environ.get("RQ1_DEPS", "/path/to/deps"))
PAGE = 0x1000
# Coverage evidence is collected to completion.  Cleanup is handled through
# process ownership; complete command output is written to the run log.
# ponytail: 全局锁只保护共享加载环境；当前仅有一个 Unicorn target。
_UNICORN_ENV_LOCK = threading.Lock()
# Target worker 共享 artifact cache，锁住写入和 hard-link 物化的临界区。
_ARTIFACT_FEATURE_CACHE_LOCK = threading.Lock()
# Docker stop 先向 runner 发送 SIGTERM。记录请求并让当前子进程、目标
# worker 和收尾阶段协作退出；不要让 PID 1 直接退出而遗留模拟器子进程。
_STOP_REQUESTED = threading.Event()
_STOP_SIGNAL: int | None = None


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    _STOP_REQUESTED.set()


def stop_requested() -> bool:
    """Return whether the current run received an external stop request."""
    return _STOP_REQUESTED.is_set()


def stop_reason() -> str:
    """Return the stable ledger reason for an externally stopped run."""
    return "external-signal" if stop_requested() else ""


def _install_stop_handlers() -> dict[int, Any]:
    """Install the CLI stop handlers and return the previous handlers."""
    _STOP_REQUESTED.clear()
    global _STOP_SIGNAL
    _STOP_SIGNAL = None
    previous = {}
    if threading.current_thread() is not threading.main_thread():
        return previous
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_stop_signal)
    return previous


def _restore_stop_handlers(previous: dict[int, Any]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


# comparison 面只包含 5 个外部对比工具；两个 Ours 方法各有独立 campaign。
ACTIVE_METHODS = (
    ("B-RVDV", "program"),
    ("B-TORTURE", "program"),
    ("B-CSMITH", "program"),
    ("B-GEMI", "program"),
    ("Fuzz4All", "program"),
)
FRAMEWORK_METHODS = (("Ours-RVGEN-Direct", "single"), ("Ours-Program-Full", "program"))
TARGETS = (
    "T-QEMU", "T-LRSV-INT", "T-LRSV-TRANS", "T-UNICORN",
    "T-RENODE", "T-RAX", "T-RVVM",
)
TARGET_MODELS = {
    "T-QEMU": "linux-user",
    "T-LRSV-INT": "linux-user-interpreter",
    "T-LRSV-TRANS": "linux-user-translated",
    "T-UNICORN": "bare-metal-capsule",
    "T-RENODE": "system-bare-metal",
    "T-RAX": "emulator-bare-metal",
    "T-RVVM": "vm-interpreter-bare-metal",
}
TARGET_KINDS = {
    "T-QEMU": "qemu", "T-LRSV-INT": "libriscv", "T-LRSV-TRANS": "libriscv",
    "T-UNICORN": "unicorn", "T-RENODE": "renode", "T-RAX": "rax", "T-RVVM": "rvvm",
}
TARGET_BACKENDS = {
    "T-QEMU": "qemu-riscv64", "T-LRSV-INT": "libriscv-interpreter",
    "T-LRSV-TRANS": "libriscv-translated", "T-UNICORN": "unicorn-riscv64",
    "T-RENODE": "renode-riscv64", "T-RAX": "rax-riscv64", "T-RVVM": "rvvm-riscv64",
}
REQUIRED_DEPENDENCY_PINS = ("riscv_dv", "torture", "decfuzzer", "csmith", "fuzz4all")
FORBIDDEN_COMMAND_WORDS = {"docker", "podman", "kill", "pkill", "taskkill", "systemctl"}
def simulator_coverage_enabled() -> bool:
    return os.environ.get("RQ1_ENABLE_COVERAGE", "1").lower() in {"1", "true", "yes"}


def rv_instruction_metrics_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether the legacy fixed RV guest metrics are part of a run.

    Source coverage and observation are the active RQ1 evidence channels.  The
    fixed RV instruction registry remains available for replaying old results,
    but an active configuration can explicitly disable it without making the
    registry a generation or finalization gate.
    """
    contract = config.get("rv_instruction_coverage") if isinstance(config, Mapping) else None
    return not (isinstance(contract, Mapping) and contract.get("enabled") is False)


def _configured_target_timeout(
    config: Mapping[str, Any], override: int | None = None,
    target: Mapping[str, Any] | None = None,
) -> int:
    """Return the canonical budget, honoring an explicit target budget."""
    limits = config.get("limits") if isinstance(config, Mapping) else None
    limits = limits if isinstance(limits, Mapping) else {}
    target_budget = target.get("timeout_seconds") if isinstance(target, Mapping) else None
    value = (
        override if override is not None else
        target_budget if target_budget is not None else
        limits.get("target_timeout_seconds", DEFAULT_TARGET_TIMEOUT_SECONDS)
    )
    if type(value) is not int or value <= 0:
        raise ValueError("target timeout must be a positive integer")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nonnegative_int(value: object, *, default: int = 0) -> int:
    """Normalize adapter counters and indices without accepting booleans/fractions."""
    if isinstance(value, bool):
        return default
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed >= 0 else default


def _complete_observation_frame(result: dict[str, Any], raw: str | bytes) -> bool:
    # Unicorn and RVVM verify RVOBS1 by reading the guest mailbox directly;
    # their stdout is intentionally empty.  Prefer that structured proof even
    # when a generic stdout field records that no serialized frame was seen.
    if (
        result.get("observation_channel") == "rvobs1-frame"
        and result.get("observer_complete") is True
        and result.get("observation_frame_verified") is True
        and _valid_hex(result.get("observation_frame_sha256"), 64)
    ):
        return True
    declared = result.get("stdout_observation_frame_complete")
    if declared is not None:
        return (
            declared is True
            and result.get("observation_frame_verified") is True
            and _valid_hex(result.get("observation_frame_sha256"), 64)
        )
    if isinstance(raw, (bytes, bytearray)):
        return _finalized_observation_frame(bytes(raw)) is not None
    if OBSERVATION_MAGIC.decode("latin1") not in raw:
        return False
    try:
        return _finalized_observation_frame(raw.encode("latin1")) is not None
    except UnicodeEncodeError:
        return False


def _libriscv_compat_observation_frame(
    frame: bytes | None, trace_record_mode: str | None,
) -> tuple[bytes | None, dict[str, str] | None]:
    """Normalize libriscv's hardwired x0 write while retaining raw evidence."""
    if (
        frame is None
        or trace_record_mode not in {"libriscv-interpreter", "libriscv-translated"}
        or len(frame) < OBSERVATION_HEADER_SIZE
    ):
        return frame, None
    checkpoint = struct.unpack_from("<Q", frame, 8)[0]
    x0 = struct.unpack_from("<Q", frame, 16)[0]
    if checkpoint == 0 or checkpoint & 1 or x0 == 0:
        return frame, None
    normalized = bytearray(frame)
    struct.pack_into("<Q", normalized, 16, 0)
    return bytes(normalized), {
        "contract": "rvobs1-libriscv-x0-compat-v1",
        "backend": trace_record_mode,
        "field": "x0",
        "raw_x0": f"0x{x0:x}",
        "normalized_x0": "0x0",
    }


def _ascii_observation_frame(raw: str) -> bytes | None:
    """Decode the line-oriented RVOBS1 transport used by RAX/Renode."""
    lines = re.findall(r"(?m)^\s*RQ1_RVOBS1=([0-9a-fA-F]+)\s*$", raw)
    if len(lines) != 1:
        return None
    try:
        frame = bytes.fromhex(lines[0])
    except ValueError:
        return None
    return frame if _finalized_observation_frame(frame) == frame else None


def _stream_observation_frame(
    stream: Any, size: int, offsets: list[int],
) -> bytes | None:
    """Return one complete frame embedded in a mixed trace stream.

    LRSV writes RVOBS1 before its debug trace.  The frame contract remains
    strict (one complete, non-nested frame); only its position in the stream
    is relaxed for the external trace path.
    """
    candidates = []
    incomplete = []
    for offset in offsets:
        stream.seek(offset)
        header = stream.read(OBSERVATION_HEADER_SIZE)
        frame_end = _observation_frame_end(header, 0)
        if frame_end is None or offset + frame_end > size:
            incomplete.append((offset, None if frame_end is None else offset + frame_end))
            continue
        candidates.append((offset, offset + frame_end))
    top_level = [
        item for item in candidates
        if not any(other[0] < item[0] < other[1] for other in candidates)
    ]
    if len(top_level) != 1:
        return None
    start, end = top_level[0]
    # A magic sequence inside the selected memory payload is data.  Any
    # marker outside it is a second/truncated frame and invalidates the
    # observation instead of silently choosing one frame.
    if any(offset < start or offset >= end for offset, _ in incomplete):
        return None
    if any(offset < start or offset >= end for offset, _ in candidates
           if offset != start):
        return None
    stream.seek(start)
    frame = stream.read(end - start)
    return frame if len(frame) == end - start else None


def _stream_observation_frame_complete(
    stream: Any, size: int, offsets: list[int],
) -> bool:
    frame = _stream_observation_frame(stream, size, offsets)
    return frame is not None and _finalized_observation_frame(frame) is not None


def _semanticize_result(target: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    raw_value = result.get("stdout", "")
    raw_bytes = bytes(raw_value) if isinstance(raw_value, (bytes, bytearray)) else None
    raw = raw_bytes.decode("utf-8", "replace") if raw_bytes is not None else str(raw_value)
    kind = target.get("kind")
    frame = _complete_observation_frame(result, raw_bytes if raw_bytes is not None else raw)
    guest_terminal = _guest_terminal_evidence(result)
    qemu_normal_exit = (
        kind == "qemu"
        and result.get("status") == "passed"
        and result.get("returncode") == 0
    )
    unsupported = _unsupported_isa_result(result)
    if unsupported is not None and not (
        frame or guest_terminal or qemu_normal_exit
        or kind == "qemu" and result.get("qemu_expected_trap_verified") is True
    ):
        return unsupported
    invalid_frame_reason = result.get("observation_frame_invalid_reason")
    if (
        kind in {"qemu", "libriscv", "unicorn"}
        and result.get("status") == "passed"
        and isinstance(invalid_frame_reason, str)
    ):
        # A complete frame with an architecturally impossible x0 is a guest
        # execution failure, not a missing observer.  Keep it visible so an
        # interpreter defect is not silently removed from the denominator.
        return {**result, "status": "failed", "reason": invalid_frame_reason,
                "observer_complete": False, "terminal_observed": True,
                "termination": "guest-invalid-state",
                "guest_exit_channel": f"{kind}-rvobs1",
                "guest_status": "failure"}
    if kind in {"qemu", "libriscv", "unicorn"} \
            and result.get("status") in {"passed", "failed"} and not frame:
        has_trap_result = (
            result.get("semantic_outcome") in {"trap", "expected-trap"}
            or result.get("guest_trap") == "delivered"
            or type(result.get("guest_cause", result.get("trap.cause"))) is int
        )
        terminal_observed = guest_terminal or qemu_normal_exit
        missing_state = (
            result.get("status") == "passed"
            and not has_trap_result
            and not terminal_observed
        )
        semantic = {
            **result,
            "status": "gap" if missing_state else result.get("status"),
            **({
                "execution_status": result.get("execution_status") or "passed",
                "execution_reason": result.get("reason"),
                "reason": result.get("reason") or "observation-frame-missing",
            } if missing_state else {}),
            "observer_complete": False,
            "observation_gap": "observation-frame-missing",
            "terminal_observed": terminal_observed,
            "program_stdout_sha256": _sha256_text(raw),
            "observation_channel": "terminal",
        }
        if terminal_observed:
            if (semantic.get("status") == "passed" or missing_state) \
                    and not semantic.get("guest_status"):
                semantic["guest_status"] = "pass"
        return semantic
    if kind == "renode" or frame:
        semantic = ""
    elif kind == "libriscv":
        semantic = "\n".join(line for line in raw.splitlines()
                               if not line.lstrip().startswith((">>", "f f_")))
        if semantic:
            semantic += "\n"
    else:
        semantic = raw
    frame_digest = result.get("observation_frame_sha256")
    if frame and not isinstance(frame_digest, str) and raw_bytes is not None:
        located = _finalized_observation_frame(raw_bytes)
        if located is not None:
            frame_digest = hashlib.sha256(located).hexdigest()
    updates = {"program_stdout_sha256": _sha256_text(semantic)}
    if frame and isinstance(frame_digest, str):
        updates.update({
            "observation_frame_sha256": frame_digest,
            "observation_channel": "rvobs1-frame",
            "observation_frame_verified": True,
            "program_stdout_sha256": frame_digest,
        })
    elif result.get("observation_channel") == "rvobs1-frame" \
            and isinstance(frame_digest, str):
        updates["program_stdout_sha256"] = frame_digest
    elif result.get("observer_complete") is True:
        # A terminal marker (tohost/exit/trace stop) is not a state snapshot.
        # Adapters used to promote it to a complete observation here; keep the
        # terminal fact but make the missing state evidence explicit.
        updates.update({
            "observer_complete": False,
            "observation_channel": "terminal",
        })
    if frame:
        # RVOBS1 is state evidence and is independent of the process exit code.
        # A non-zero exit or timeout remains a separate terminal/outcome fact.
        updates["observer_complete"] = True
        if result.get("terminal_observed") is True or result.get("status") == "passed":
            updates["terminal_observed"] = True
    return {**result, **updates}




def _safe_process(argv: list[str]) -> str | None:
    if not argv:
        return "empty-command"
    joined = " ".join(str(item).lower() for item in argv)
    first = Path(str(argv[0])).name.lower()
    return "forbidden-process-command" if first in FORBIDDEN_COMMAND_WORDS or any(
        re.search(rf"(^|[^a-z]){re.escape(word)}([^a-z]|$)", joined)
        for word in FORBIDDEN_COMMAND_WORDS
    ) else None


def _persist_run(out: Path, label: str, argv: list[str], result: dict[str, Any], *,
                 cwd: Path | None = None,
                 env: dict[str, str] | None = None) -> dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    stdout = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else str(stdout)
    stderr = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    stdout_path = out / "logs" / f"{safe}.stdout"
    stderr_path = out / "logs" / f"{safe}.stderr"
    command_path = out / "logs" / f"{safe}.command.json"
    # run_cmd may already have streamed the complete bytes into these paths.
    # Keep them intact; rewriting a full log with an in-memory preview would
    # silently turn complete evidence into a truncated result.
    stdout_capture_complete = (
        result.get("stdout_capture_complete") is True
        and result.get("raw_stdout_sha256") == sha256(stdout_path)
    )
    stderr_capture_complete = (
        result.get("stderr_capture_complete") is True
        and result.get("raw_stderr_sha256") == sha256(stderr_path)
    )
    # Compact trace collection may have created a partial capture file before
    # its reader is stopped by a timeout/deadline.  Preserve that evidence;
    # never try to recreate the same path with exclusive creation.
    if not stdout_capture_complete and not stdout_path.exists():
        try:
            write_text_once(stdout_path, stdout)
        except FileExistsError:
            # The streaming capture can create the file between the existence
            # check and this fallback. Keep that capture instead of turning a
            # completed target attempt into a runner error.
            pass
    if not stderr_capture_complete and not stderr_path.exists():
        try:
            write_text_once(stderr_path, stderr)
        except FileExistsError:
            pass
    command_digest = digest(argv)
    environment_keys_sha256 = digest(sorted(env)) if env is not None else None
    write_json(command_path, {"argv": argv, "cwd": str(cwd) if cwd else None,
                              "command_sha256": command_digest,
                              "environment_keys_sha256": environment_keys_sha256})
    return {
        **result,
        "command": argv,
        "cwd": str(cwd) if cwd else None,
        "command_sha256": command_digest,
        "environment_keys_sha256": environment_keys_sha256,
        "stdout_sha256": (
            result.get("raw_stdout_sha256")
            if result.get("stdout_capture_complete") is True else _sha256_text(stdout)
        ),
        "stderr_sha256": (
            result.get("raw_stderr_sha256")
            if result.get("stderr_capture_complete") is True else _sha256_text(stderr)
        ),
        "raw_stdout_sha256": result.get("raw_stdout_sha256") or _sha256_text(stdout),
        "raw_stderr_sha256": result.get("raw_stderr_sha256") or _sha256_text(stderr),
        "stdout_path": str(stdout_path.relative_to(out)),
        "stderr_path": str(stderr_path.relative_to(out)),
        "command_path": str(command_path.relative_to(out)),
    }


def _path_contains_symlink(path: Path) -> bool:
    current = path.absolute()
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _prepare_output(path: Path, *, allowed_files: set[str] | None = None) -> Path:
    if _path_contains_symlink(path):
        raise FileExistsError(f"run output path contains a symlink: {path}")
    path = path.resolve()
    if path == HERE or HERE in path.parents:
        raise FileExistsError(f"run output must be outside comparison code root: {path}")
    allowed = {
        "execution-manifest.json", "config.snapshot.json", "container-inspect.json",
        "wrapper-logs", "wrapper-result.json", "source-snapshot", "logs", "framework",
        "continuation-seed", "control",
        ".rq1-run.lock", "tmp",
        ".rq1-heartbeat.json", *(allowed_files or set())
    }
    resumable_framework = False
    manifest = path / "execution-manifest.json"
    if manifest.is_file() and not manifest.is_symlink():
        try:
            manifest_value = load_json(manifest)
        except (OSError, TypeError, ValueError):
            manifest_value = {}
        resumable_framework = (
            isinstance(manifest_value, dict)
            and manifest_value.get("action") == "framework-run"
            and manifest_value.get("run_id") == path.name
            and (path / "framework").is_dir()
        )
    if path.exists() and (
        not path.is_dir()
        or (not resumable_framework and any(item.name not in allowed for item in path.iterdir()))
    ):
        raise FileExistsError(f"run output is not empty: {path}")
    temporary = path / "tmp"
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_dir()):
        raise FileExistsError(f"run temporary path is not a directory: {temporary}")
    path.mkdir(parents=True, exist_ok=True)
    if manifest.is_symlink():
        raise FileExistsError(f"run manifest is a symlink: {manifest}")
    if manifest.is_file():
        try:
            data = load_json(manifest)
        except (OSError, TypeError, ValueError):
            raise FileExistsError(f"run manifest is invalid: {manifest}")
        if data.get("run_id") != path.name:
            raise FileExistsError(f"run manifest belongs to another run: {manifest}")
    lock = path / ".rq1-run.lock"
    if lock.is_symlink() or (lock.exists() and not lock.is_file()):
        raise FileExistsError(f"run lock is not a regular file: {lock}")
    if not lock.exists():
        write_text_once(lock, "rq1-comparison-run-v1\n")
    return path


def _freeze_config(out: Path, config_path: Path) -> Path:
    """在 run root 保存原始配置字节，后续收尾只认这份快照。"""
    snapshot = Path(out) / "config.snapshot.json"
    expected = sha256(config_path.resolve())
    if expected is None:
        raise ValueError("config file is missing")
    if snapshot.exists():
        if snapshot.is_symlink() or not snapshot.is_file() or sha256(snapshot) != expected:
            raise ValueError("config snapshot does not match the requested config")
    else:
        shutil.copyfile(config_path.resolve(), snapshot)
    manifest_path = Path(out) / "execution-manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("config_sha256") not in (None, expected):
            raise ValueError("config does not match execution manifest")
        manifest["config_snapshot"] = "config.snapshot.json"
        write_json_replace(manifest_path, manifest)
    return snapshot


def _load_frozen_config(out: Path, config_path: Path) -> tuple[dict[str, Any], Path]:
    snapshot = Path(out) / "config.snapshot.json"
    if snapshot.is_file():
        manifest = load_json(Path(out) / "execution-manifest.json")
        if manifest.get("config_sha256") not in (None, sha256(snapshot)):
            raise ValueError("config snapshot does not match execution manifest")
        return load_json(snapshot), snapshot
    return load_json(config_path), config_path.resolve()


def _release_run_lock(path: Path) -> None:
    lock = path / ".rq1-run.lock"
    try:
        # wrapper 写入 JSON lifecycle lock；容器内 runner 不拥有它。
        if lock.read_text(encoding="utf-8") != "rq1-comparison-run-v1\n":
            return
        lock.unlink()
    except (OSError, UnicodeError):
        pass


def dep(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        try:
            return DEPS / path.relative_to("/path/to/deps")
        except ValueError:
            return path
    return DEPS / path


def _flush_gcov(binary: Path) -> None:
    try:
        library = ctypes.CDLL(str(binary))
        for name in ("__gcov_dump", "__gcov_flush"):
            function = getattr(library, name, None)
            if function:
                function()
                return
    except (AttributeError, OSError):
        pass


def _run_in_env(env: dict[str, str] | None, callback):
    if not env:
        return callback()
    with _UNICORN_ENV_LOCK:
        old = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            return callback()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _dependency_present(config: dict[str, Any], name: str) -> bool:
    path = dep(config.get("dependencies", {}).get(name))
    pin = config.get("dependency_pins", {}).get(name, {})
    kind = pin.get("kind")
    if not path:
        return False
    if kind == "git":
        try:
            head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False)
        except OSError:
            return False
        return path.is_dir() and head.returncode == 0 and (
            not pin.get("commit") or head.stdout.strip() == pin["commit"])
    if kind == "binary":
        return path.is_file() and (
            not pin.get("binary_sha256")
            or sha256(path) == str(pin["binary_sha256"]).lower()
        )
    return path.exists()


def command(name: str) -> str | None:
    return shutil.which(name)


def _cross_compiler_include_flags(compiler: str | None) -> list[str]:
    """Expose compiler headers to freestanding C artifact attempts.

    ``--float`` cases legitimately include ``float.h``/``math.h``.  Keep
    ``-nostdinc`` for the runtime contract, but add the cross compiler's own
    header directories so an ELF conversion attempt is not rejected merely by
    the absence of those standard headers.  Missing paths simply leave the
    best-effort conversion to report an artifact gap.
    """
    if not compiler:
        return []
    paths: list[Path] = []
    try:
        include = subprocess.run(
            [compiler, "-print-file-name=include"],
            capture_output=True, text=True, check=False, timeout=5,
        ).stdout.strip()
        if include:
            paths.append(Path(include))
        sysroot = subprocess.run(
            [compiler, "-print-sysroot"],
            capture_output=True, text=True, check=False, timeout=5,
        ).stdout.strip()
        if sysroot and sysroot != "/":
            paths.append(Path(sysroot) / "usr" / "include")
        triple = Path(compiler).name.removesuffix("-gcc")
        if triple:
            paths.append(Path("/usr") / triple / "include")
    except (OSError, subprocess.SubprocessError):
        return []
    flags: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir() and path not in seen:
            seen.add(path)
            flags.extend(("-isystem", str(path)))
    return flags


def _descendants(root: int) -> set[int]:
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            ppid = int(text.rsplit(")", 1)[1].split()[1])
            parents[int(entry.name)] = ppid
        except (OSError, ValueError, IndexError):
            pass
    found, pending = set(), [root]
    while pending:
        parent = pending.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                pending.append(pid)
    return found


def _terminate_process_tree(pid: int) -> set[int]:
    victims = _descendants(pid)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    _signal_pids(victims, signal.SIGTERM)
    return victims


def _group_members(pgid: int | None) -> set[int]:
    if os.name != "posix" or not pgid:
        return set()
    members = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tail = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(tail[2]) == pgid:
                members.add(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def _signal_pids(pids: set[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _kill_process_tree(pid: int) -> set[int]:
    victims = _descendants(pid)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    _signal_pids(victims, signal.SIGKILL)
    return victims

def _enable_child_subreaper() -> None:
    if os.name != "posix":
        return
    try:
        ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
    except (AttributeError, OSError):
        pass


def _reap_children(pids: set[int] | None = None) -> None:
    if os.name != "posix" or not pids:
        return
    pending = set(pids)
    deadline = time.monotonic() + 2
    while pending and time.monotonic() < deadline:
        for pid in tuple(pending):
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
                if waited:
                    pending.discard(pid)
            except (ChildProcessError, OSError):
                pending.discard(pid)
        if pending:
            time.sleep(0.02)


def _reap_adopted_children(timeout: float = 2.0) -> int:
    """Reap descendants adopted by this runner's Linux subreaper.

    A timed-out framework child can die before its Renode/RAX helper children
    finish.  Because this process is a subreaper, those helpers become our
    children and otherwise remain zombies after the lane has already produced
    its durable partial result.  Call this only after all lane/finalizer
    futures have completed, so it cannot steal a live ``Popen`` child from a
    concurrent lane.
    """
    if os.name != "posix":
        return 0
    deadline = time.monotonic() + max(0.0, float(timeout))
    reaped = 0
    while True:
        try:
            waited, _status = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            break
        if waited:
            reaped += 1
            continue
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    return reaped


def _cleanup_processes(
    process_pid: int | None, process_group: int | None, victims: set[int],
) -> None:
    if process_group and os.name == "posix":
        members = _group_members(process_group)
        victims.update(members)
        _signal_pids(members, signal.SIGKILL)
    if process_pid and os.name == "posix":
        residual = _descendants(process_pid)
        if residual:
            victims.update(residual)
            _kill_process_tree(process_pid)
    _reap_children(victims)


_enable_child_subreaper()






def _trace_writer_pids(path: Path, excluded: set[int] | None = None) -> set[int]:
    """查找仍持有本 case 原始 trace 文件的进程。"""
    try:
        expected = path.stat()
    except OSError:
        return set()
    excluded = set(excluded or ()) | {os.getpid()}
    writers: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    current = os.stat(fd)
                except OSError:
                    continue
                if (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino):
                    writers.add(int(entry.name))
                    break
        except OSError:
            continue
    return writers


def _outside_elf_pc_loop(
    path: Path | None, allowed_pcs: set[int] | None, *, repeats: int = 32,
    plain_pc_lines: bool = False,
) -> int | None:
    """Return a PC after an instruction trace remains outside the capsule ELF."""
    if path is None or not allowed_pcs or repeats < 1:
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
        match = _TRACE_INS_PC_RE.search(line)
        if match is None and plain_pc_lines:
            match = _TRACE_PLAIN_PC_RE.fullmatch(line.strip())
        if not match:
            continue
        try:
            pc = int(match.group(1), 16)
        except ValueError:
            return None
        if pc in allowed_pcs:
            return None
        if repeated_pc is None:
            repeated_pc = pc
        elif pc != repeated_pc:
            return None
        repeated += 1
        if repeated >= repeats:
            return repeated_pc
    return None


def _pipe_writer_pids(stream: Any, excluded: set[int] | None = None) -> set[int]:
    """Find processes that still hold the write end of a captured pipe."""
    if os.name != "posix" or stream is None:
        return set()
    try:
        expected = os.fstat(stream.fileno())
    except (OSError, ValueError, AttributeError):
        return set()
    excluded = set(excluded or ()) | {os.getpid()}
    writers: set[int] = set()
    try:
        proc_entries = Path("/proc").iterdir()
    except OSError:
        return writers
    for entry in proc_entries:
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    current = os.stat(fd)
                except OSError:
                    continue
                if (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino):
                    writers.add(int(entry.name))
                    break
        except OSError:
            continue
    return writers


def _cleanup_trace_writers(path: Path | None, excluded: set[int]) -> dict[str, Any]:
    """目标结束后关闭脱离进程组的 trace writer，不截断已有证据。"""
    if path is None or not path.exists():
        return {"status": "not-found", "observed": set(), "remaining": set()}
    observed: set[int] = set()
    remaining: set[int] = set()
    # 给已经脱离进程组、但正在切换文件描述符的 writer 一个很短的
    # 收口窗口；这是生命周期清理，不是 trace 数据量限制。
    for _ in range(3):
        current = _trace_writer_pids(path, excluded)
        observed.update(current)
        if current:
            _signal_pids(current, signal.SIGTERM)
            time.sleep(0.05)
            remaining = _trace_writer_pids(path, excluded)
            if remaining:
                _signal_pids(remaining, signal.SIGKILL)
                time.sleep(0.05)
                remaining = _trace_writer_pids(path, excluded)
        else:
            remaining = set()
            time.sleep(0.05)
    return {
        "status": "gap" if remaining else "clean" if not observed else "killed",
        "observed": observed,
        "remaining": remaining,
    }


def run_cmd(argv: list[str], timeout: float | None, *, cwd: Path | None = None,
            env: dict[str, str] | None = None, stdin_text: str | None = None,
            timeout_stdin_text: str | None = None,
            stdout_output_path: Path | None = None,
            stderr_output_path: Path | None = None,
            trace_path: Path | None = None,
            trace_watch_pcs: set[int] | None = None,
            trace_watch_plain_pcs: bool = False,
            trace_compact_path: Path | None = None,
            trace_record_mode: str | None = None,
            stop_on_output: tuple[str, ...] | None = None,
            stop_on_output_stdin_text: str | None = None,
            trace_watch_signal: int = signal.SIGINT,
            graceful_shutdown_seconds: float | None = None,
            graceful_shutdown_fallback_signal: int | None = None,
            graceful_shutdown_fallback_after_seconds: float | None = None,
            timeout_graceful_shutdown_seconds: float | None = None) -> dict[str, Any]:
    if reason := _safe_process(argv):
        return {"status": "gap", "reason": reason, "returncode": None,
                "stdout": "", "stderr": "", "process_started": False,
                "process_executed": False, "terminal_observed": False}
    if stop_requested():
        return {"status": "gap", "reason": stop_reason(), "reason_code": stop_reason(),
                "returncode": None, "stdout": "", "stderr": "",
                "process_started": False, "process_executed": False,
                "terminal_observed": False, "termination": stop_reason()}
    if trace_compact_path is not None:
        return _run_compact_trace_cmd(
            argv, timeout, cwd=cwd, env=env, stdin_text=stdin_text,
            compact_path=trace_compact_path, trace_record_mode=trace_record_mode,
            stdout_output_path=stdout_output_path,
            stderr_output_path=stderr_output_path,
        )
    started = time.monotonic()
    victims: set[int] = set()
    process_started = False
    process_pid = None
    process_group = None
    termination = "launch-failed"
    external_stop_requested = False
    trace_watch_pc = None

    try:
        # 逐指令 trace 的 stdout 可以涨到几十 GB；容器 /tmp 是内存 tmpfs，
        # 落在那里会撑爆整个容器，并连带打死同容器里的 K1 参考。
        scratch_root = os.environ.get("RQ1_RUN")
        scratch_root = scratch_root if scratch_root and os.path.isdir(scratch_root) else None
        with tempfile.TemporaryFile(dir=scratch_root) as stdout_file, \
                tempfile.TemporaryFile(dir=scratch_root) as stderr_file:
            process = subprocess.Popen(
                argv,
                cwd=cwd, env=env,
                stdin=subprocess.PIPE
                if stdin_text is not None or timeout_stdin_text is not None
                or stop_on_output_stdin_text is not None else None,
                stdout=stdout_file, stderr=stderr_file,
                start_new_session=os.name == "posix",
            )
            process_pid = process.pid
            process_started = True
            process_group = os.getpgid(process.pid) if os.name == "posix" else None
            graceful_shutdown_requested = False
            graceful_shutdown_deadline: float | None = None
            graceful_shutdown_fallback_at: float | None = None
            graceful_shutdown_fallback_sent = False
            early_stop_event = threading.Event()
            early_stop_pattern: list[str] = []
            stop_watcher = None

            def request_early_stop(stdin_command: str | None = None) -> None:
                nonlocal graceful_shutdown_requested, graceful_shutdown_deadline
                nonlocal graceful_shutdown_fallback_at
                if graceful_shutdown_deadline is not None:
                    return
                sent_command = False
                if stdin_command is not None and process.stdin is not None:
                    try:
                        process.stdin.write(stdin_command.encode())
                        process.stdin.flush()
                        process.stdin.close()
                        sent_command = True
                    except (BrokenPipeError, OSError):
                        pass
                if (not sent_command or graceful_shutdown_seconds is None) \
                        and process.poll() is None and os.name == "posix":
                    try:
                        os.killpg(process_group or os.getpgid(process.pid), trace_watch_signal)
                    except (ProcessLookupError, PermissionError):
                        pass
                graceful_shutdown_requested = True
                if graceful_shutdown_seconds is not None:
                    now = time.monotonic()
                    graceful_shutdown_deadline = (
                        now + max(0.0, float(graceful_shutdown_seconds))
                    )
                    if graceful_shutdown_fallback_signal is not None:
                        fallback_delay = (
                            1.0 if graceful_shutdown_fallback_after_seconds is None
                            else max(0.0, float(graceful_shutdown_fallback_after_seconds))
                        )
                        graceful_shutdown_fallback_at = now + min(
                            fallback_delay, max(0.0, float(graceful_shutdown_seconds)),
                        )

            def wait_for_graceful_shutdown() -> None:
                nonlocal graceful_shutdown_fallback_sent
                if process.poll() is None and graceful_shutdown_deadline is not None:
                    while process.poll() is None:
                        now = time.monotonic()
                        if (
                            not graceful_shutdown_fallback_sent
                            and graceful_shutdown_fallback_signal is not None
                            and graceful_shutdown_fallback_at is not None
                            and now >= graceful_shutdown_fallback_at
                        ):
                            # A collector such as dotnet-coverage must outlive
                            # its simulator child so it can serialize the
                            # profile after the guest is stopped.
                            _signal_pids(
                                _descendants(process.pid),
                                graceful_shutdown_fallback_signal,
                            )
                            graceful_shutdown_fallback_sent = True
                            continue
                        remaining = graceful_shutdown_deadline - now
                        if remaining <= 0:
                            break
                        if (
                            not graceful_shutdown_fallback_sent
                            and graceful_shutdown_fallback_at is not None
                        ):
                            remaining = min(
                                remaining,
                                max(0.001, graceful_shutdown_fallback_at - now),
                            )
                        try:
                            process.wait(timeout=min(0.1, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                if process.poll() is None:
                    if os.name == "posix":
                        try:
                            os.killpg(
                                process_group or os.getpgid(process.pid), signal.SIGINT,
                            )
                        except (ProcessLookupError, PermissionError):
                            pass
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            victims.update(_kill_process_tree(process.pid))
                    else:
                        try:
                            process.terminate()
                        except (OSError, ProcessLookupError):
                            pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass

            stop_patterns = tuple(
                (pattern, pattern.encode("utf-8"))
                for pattern in (stop_on_output or ())
                if pattern
            )
            if stop_patterns and os.name == "posix":
                def watch_output() -> None:
                    offsets = [0, 0]
                    tails = [b"", b""]
                    terminal_seen = False
                    frame_started = [False, False]
                    frame_complete = False
                    terminal_markers = (b"RQ1_TOHOST", b"RQ1_EXCEPTION")
                    max_tail = max(
                        *(len(encoded) for _, encoded in stop_patterns),
                        *(len(marker) for marker in terminal_markers),
                    ) - 1
                    streams = (stdout_file.fileno(), stderr_file.fileno())
                    while not early_stop_event.is_set():
                        pending = []
                        for index, file_descriptor in enumerate(streams):
                            try:
                                size = os.fstat(file_descriptor).st_size
                                if size <= offsets[index]:
                                    continue
                                chunk = os.pread(
                                    file_descriptor,
                                    size - offsets[index],
                                    offsets[index],
                                )
                                offsets[index] = size
                            except OSError:
                                continue
                            data = tails[index] + chunk
                            terminal_seen = terminal_seen or any(
                                marker in data
                                for marker in terminal_markers
                            )
                            tails[index] = data[-max_tail:] if max_tail else b""
                            pending.append((index, data))
                        # Read both pipes before acting on a stop marker.  Renode
                        # writes the terminal mailbox marker to stdout and may log
                        # CPU abort to stderr; acting on stderr first can interrupt
                        # the mailbox/frame flush and turn a valid result into a gap.
                        for index, data in pending:
                            marker_at = data.find(b"RQ1_RVOBS1=")
                            if marker_at >= 0:
                                frame_started[index] = True
                                frame_complete = frame_complete or b"\n" in data[
                                    marker_at + len(b"RQ1_RVOBS1="):
                                ]
                            elif frame_started[index] and b"\n" in data:
                                frame_complete = True
                        # The marker and the rest of the mailbox line can be
                        # written by separate stdout calls.  Once the later
                        # chunk closes that line, keep the safe point latched
                        # even though the marker is no longer in this chunk.
                        if frame_complete and any(
                                pattern == "RQ1_RVOBS1="
                                for pattern, _ in stop_patterns):
                            early_stop_pattern.append("RQ1_RVOBS1=")
                            request_early_stop(stop_on_output_stdin_text)
                            early_stop_event.set()
                            return
                        for _, data in pending:
                            for pattern, encoded in stop_patterns:
                                if encoded not in data:
                                    continue
                                # A complete Renode mailbox frame is the safe
                                # point to interrupt dotnet-coverage: the
                                # guest evidence is already captured and the
                                # collector gets a chance to write Cobertura.
                                # Other stop markers remain suppressed after a
                                # tohost marker so Renode can flush the frame.
                                frame_stop = encoded == b"RQ1_RVOBS1="
                                if frame_stop and not frame_complete:
                                    continue
                                if (
                                    terminal_seen and not frame_stop
                                    and pattern != "RQ1_EXCEPTION"
                                ):
                                    continue
                                early_stop_pattern.append(pattern)
                                request_early_stop(stop_on_output_stdin_text)
                                early_stop_event.set()
                                return
                        if process.poll() is not None:
                            return
                        time.sleep(0.01)

                stop_watcher = threading.Thread(
                    target=watch_output, name="rq1-output-stop-watcher", daemon=True,
                )
                stop_watcher.start()
            if stdin_text is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin_text.encode())
                    if timeout_stdin_text is None:
                        process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            deadline = math.inf if timeout is None else (
                started + max(0.001, float(timeout))
            )
            while process.poll() is None:
                if stop_requested():
                    external_stop_requested = True
                    if os.name == "posix":
                        victims |= _terminate_process_tree(process.pid)
                    else:
                        try:
                            process.terminate()
                        except (OSError, ProcessLookupError):
                            pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            victims |= _kill_process_tree(process.pid)
                        else:
                            try:
                                process.kill()
                            except (OSError, ProcessLookupError):
                                pass
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                    _signal_pids(victims, signal.SIGKILL)
                    status, returncode = "gap", process.returncode
                    termination = stop_reason()
                    break
                if graceful_shutdown_deadline is not None \
                        and time.monotonic() >= graceful_shutdown_deadline \
                        and process.poll() is None:
                    wait_for_graceful_shutdown()
                    status, returncode = "failed", process.returncode
                    termination = "graceful-stop-timeout"
                    break
                if (
                    graceful_shutdown_deadline is not None
                    and not graceful_shutdown_fallback_sent
                    and graceful_shutdown_fallback_signal is not None
                    and graceful_shutdown_fallback_at is not None
                    and time.monotonic() >= graceful_shutdown_fallback_at
                ):
                    _signal_pids(
                        _descendants(process.pid), graceful_shutdown_fallback_signal,
                    )
                    graceful_shutdown_fallback_sent = True
                trace_watch_pc = _outside_elf_pc_loop(
                    trace_path, trace_watch_pcs,
                    plain_pc_lines=trace_watch_plain_pcs,
                )
                if trace_watch_pc is not None:
                    early_stop_pattern.append(
                        f"guest-pc-outside-elf-loop:0x{trace_watch_pc:x}"
                    )
                    early_stop_event.set()
                    request_early_stop(timeout_stdin_text)
                    status, returncode = "failed", process.returncode
                    termination = "guest-invalid-state"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    graceful_shutdown_requested = False
                    timeout_shutdown_sent = False
                    if timeout_stdin_text is not None and process.stdin is not None:
                        try:
                            process.stdin.write(timeout_stdin_text.encode())
                            process.stdin.flush()
                            process.stdin.close()
                            graceful_shutdown_requested = True
                            timeout_shutdown_sent = True
                        except (BrokenPipeError, OSError):
                            pass
                    if timeout_shutdown_sent and timeout_graceful_shutdown_seconds is not None:
                        now = time.monotonic()
                        graceful_shutdown_deadline = (
                            now + max(0.0, float(timeout_graceful_shutdown_seconds))
                        )
                        if graceful_shutdown_fallback_signal is not None:
                            fallback_delay = (
                                1.0 if graceful_shutdown_fallback_after_seconds is None
                                else max(0.0, float(graceful_shutdown_fallback_after_seconds))
                            )
                            graceful_shutdown_fallback_at = now + min(
                                fallback_delay,
                                max(0.0, float(timeout_graceful_shutdown_seconds)),
                            )
                        wait_for_graceful_shutdown()
                    if process.poll() is None and os.name == "posix":
                        victims |= _terminate_process_tree(process.pid)
                    try:
                        process.wait(timeout=1 if process.poll() is None else 0)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            victims |= _kill_process_tree(process.pid)
                        else:
                            process.kill()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                    _signal_pids(victims, signal.SIGKILL)
                    status, returncode = "timeout", 124
                    termination = "timeout"
                    break
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            else:
                status, returncode = (
                    "passed" if process.returncode == 0 else "failed",
                    process.returncode,
                )
                termination = "exit"
            if termination == "guest-invalid-state" and process.poll() is None:
                if graceful_shutdown_deadline is None:
                    graceful_shutdown_deadline = time.monotonic() + 1.0
                wait_for_graceful_shutdown()
            if process.poll() is not None:
                returncode = process.returncode
            if stop_watcher is not None:
                stop_watcher.join(timeout=1.0)
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            # 目标退出后先收掉同一进程组和派生进程，再处理可能脱离进程组
            # 的 trace writer；否则它们会在 guest 已结束后继续写盘。
            _cleanup_processes(process_pid, process_group, victims)
            trace_cleanup = _cleanup_trace_writers(
                trace_path, victims | ({process_pid} if process_pid else set()),
            )
            _reap_children(trace_cleanup["observed"])
            trace_size = 0
            if trace_path is not None:
                try:
                    trace_size = trace_path.stat().st_size
                except OSError:
                    pass
            stdout_file.seek(0, 2)
            stdout_size = stdout_file.tell()
            stderr_file.seek(0, 2)
            stderr_size = stderr_file.tell()
            def file_sha256(stream):
                stream.seek(0)
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b""):
                    value.update(block)
                return value.hexdigest()
            raw_stdout_sha256 = file_sha256(stdout_file)
            raw_stderr_sha256 = file_sha256(stderr_file)
            stdout_frame_offsets = []
            stdout_file.seek(0)
            carry = b""
            scan_offset = 0
            for block in iter(lambda: stdout_file.read(1 << 20), b""):
                data = carry + block
                search_from = 0
                while True:
                    found = data.find(OBSERVATION_MAGIC, search_from)
                    if found < 0:
                        break
                    stdout_frame_offsets.append(
                        scan_offset - len(carry) + found
                    )
                    search_from = found + 1
                carry = data[-(len(OBSERVATION_MAGIC) - 1):]
                scan_offset += len(block)
            stdout_contains_observation_frame = bool(stdout_frame_offsets)
            raw_observation_frame = _stream_observation_frame(
                stdout_file, stdout_size, stdout_frame_offsets,
            )
            observation_frame, frame_compatibility = (
                _libriscv_compat_observation_frame(
                    raw_observation_frame, trace_record_mode,
                )
            )
            raw_frame_sha256 = (
                hashlib.sha256(raw_observation_frame).hexdigest()
                if frame_compatibility is not None else None
            )
            if observation_frame is not None:
                observation_frame = _finalized_observation_frame(observation_frame)
            stdout_observation_frame_complete = observation_frame is not None
            observation_frame_invalid_reason = None
            if raw_observation_frame is not None and observation_frame is None:
                if len(raw_observation_frame) >= OBSERVATION_HEADER_SIZE \
                        and struct.unpack_from("<Q", raw_observation_frame, 16)[0] != 0:
                    observation_frame_invalid_reason = "guest-x0-nonzero"
            if stdout_output_path is not None:
                stdout_output_path.parent.mkdir(parents=True, exist_ok=True)
                with stdout_output_path.open("wb") as capture:
                    stdout_file.seek(0)
                    shutil.copyfileobj(stdout_file, capture)
            stdout_file.seek(0)
            stdout_text = stdout_file.read().decode("utf-8", "replace")
            ascii_observation_frame = _ascii_observation_frame(stdout_text)
            if observation_frame is None and ascii_observation_frame is not None:
                observation_frame = ascii_observation_frame
                stdout_contains_observation_frame = True
                stdout_observation_frame_complete = True
                observation_frame_invalid_reason = None
            stderr_file.seek(0)
            stderr_text = stderr_file.read().decode("utf-8", "replace")
            if stderr_output_path is not None:
                stderr_output_path.parent.mkdir(parents=True, exist_ok=True)
                with stderr_output_path.open("wb") as capture:
                    stderr_file.seek(0)
                    shutil.copyfileobj(stderr_file, capture)
            result = {
                "status": status, "returncode": returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_contains_observation_frame": stdout_contains_observation_frame,
                "stdout_observation_frame_complete": stdout_observation_frame_complete,
                "observation_frame_verified": stdout_observation_frame_complete,
                "observation_frame_invalid_reason": observation_frame_invalid_reason,
                "observation_frame_compatibility": frame_compatibility,
                "observation_frame_raw_sha256": raw_frame_sha256,
                "observation_frame_sha256": (
                    hashlib.sha256(observation_frame).hexdigest()
                    if observation_frame is not None else None
                ),
                "raw_stdout_sha256": raw_stdout_sha256,
                "raw_stderr_sha256": raw_stderr_sha256,
                "stdout_bytes_total": stdout_size,
                "stderr_bytes_total": stderr_size,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "stdout_capture_complete": True,
                "stderr_capture_complete": True,
                "trace_truncated": status == "timeout" or external_stop_requested
                or trace_watch_pc is not None
                or trace_cleanup["status"] == "killed",
                "trace_complete": (
                    trace_path is None or (
                        status not in {"timeout", "gap"}
                        and trace_watch_pc is None
                        and trace_cleanup["status"] == "clean"
                        and trace_size > 0
                    )
                ),
                "trace_cleanup_status": trace_cleanup["status"],
                "trace_writer_pids": sorted(trace_cleanup["observed"]),
                "trace_writer_remaining": sorted(trace_cleanup["remaining"]),
                "graceful_shutdown_requested": graceful_shutdown_requested,
                "external_stop_requested": external_stop_requested,
                "early_stop_requested": bool(early_stop_pattern),
                "early_stop_pattern": early_stop_pattern[0] if early_stop_pattern else None,
                "elapsed_s": round(time.monotonic() - started, 6),
                "process_started": process_started,
                "process_executed": process_started,
                # Host-process exit is not a guest terminal.  A wall timeout
                # is the one exception: the host has a classified terminal,
                # while normal exits still require adapter evidence.
                "terminal_observed": status == "timeout",
                "process_pid": process_pid,
                "termination": termination,
                "descendant_count": len(victims),
            }
            if status == "timeout":
                result["reason"] = "wall-timeout"
            elif trace_watch_pc is not None:
                result.update(
                    reason="guest-pc-outside-elf-loop",
                    terminal_observed=True,
                    observer_complete=False,
                    guest_status="failure",
                    observation_gap="guest PC repeated outside the executable capsule ELF",
                    trace_reason=f"guest-pc-outside-elf-loop:0x{trace_watch_pc:x}",
                )
            elif external_stop_requested:
                result["reason"] = stop_reason()
                result["reason_code"] = stop_reason()
            if trace_path is not None:
                result["trace_size_bytes"] = trace_size
            if not result["trace_complete"]:
                result["trace_reason"] = (
                    f"guest-pc-outside-elf-loop:0x{trace_watch_pc:x}"
                    if trace_watch_pc is not None else
                    "trace-writer-cleanup-gap"
                    if trace_cleanup["status"] == "gap"
                    else "trace-writer-killed-after-prefix"
                    if trace_cleanup["status"] == "killed"
                    else "trace-file-missing"
                )
            return result
    except FileNotFoundError as exc:
        return {"status": "gap", "reason": "command-missing", "error": str(exc),
                "returncode": None, "stdout": "", "stderr": "",
                "process_started": False, "process_executed": False,
                "terminal_observed": False,
                "termination": "launch-failed"}
    finally:
        _cleanup_processes(process_pid, process_group, victims)


def _run_compact_trace_cmd(
    argv: list[str], timeout: float | None, *, cwd: Path | None = None,
    env: dict[str, str] | None = None, stdin_text: str | None = None,
    compact_path: Path | None = None,
    trace_record_mode: str | None = None,
    stdout_output_path: Path | None = None,
    stderr_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a verbose target while retaining only its executed-PC evidence.

    libriscv exposes one text record per dynamic instruction. A tight loop can
    repeat a PC millions of times, so this collector drains the full stream,
    counts records, and retains each unique PC once.
    """
    patterns = {
        "libriscv-translated": re.compile(
            rb"(?m)^f [^\r\n]+\s+pc\s+0x([0-9a-fA-F]+)\s+instr\s+[0-9a-fA-F]+"
        ),
        "libriscv-interpreter": re.compile(
            rb"(?m)^\s*\[(?:0[xX])?([0-9a-fA-F]+)\]\s+[0-9a-fA-F]+"
        ),
    }
    pattern = patterns.get(trace_record_mode or "")
    if compact_path is None or pattern is None:
        return {"status": "gap", "reason": "compact-trace-configuration-invalid",
                "returncode": None, "stdout": "", "stderr": "",
                "process_started": False, "process_executed": False,
                "terminal_observed": False}
    if stop_requested():
        return {"status": "gap", "reason": stop_reason(), "reason_code": stop_reason(),
                "returncode": None, "stdout": "", "stderr": "",
                "process_started": False, "process_executed": False,
                "terminal_observed": False, "termination": stop_reason()}
    started = time.monotonic()
    process = None
    process_group = None
    process_pid = None
    victims: set[int] = set()
    process_started = False
    status = "launch-failed"
    returncode = None
    termination = "launch-failed"
    external_stop_requested = False
    stdout_hash = hashlib.sha256()
    stderr_hash = hashlib.sha256()
    stdout_total = [0]
    stderr_total = [0]
    trace_records_total = [0]
    trace_seen: set[int] = set()
    trace_error: list[str] = []
    capture_error: list[str] = []
    stdout_capture_complete = [False]
    stderr_capture_complete = [False]
    diagnostic_pattern = re.compile(
        rb"instruction count limit reached|fuel exhausted|machine exception|"
        rb"unimplemented|exception|riscv trap|riscv tohost",
        re.IGNORECASE,
    )
    # 原始 stdout/stderr 已完整写入文件；诊断摘要只保留每个通道第一条
    # 命中行，避免异常循环把同一段 trace 再复制到内存中。
    diagnostic_lines: dict[str, bytes] = {}
    stdout_frame_marker = False
    stdout_frame_candidate = bytearray()
    stdout_frame_search_tail = b""
    stdout_frame_raw: bytes | None = None
    observation_frame_error = None
    stdout_frame_invalid = False

    def feed_frame(block: bytes) -> None:
        nonlocal stdout_frame_marker, stdout_frame_search_tail, stdout_frame_raw
        nonlocal observation_frame_error, stdout_frame_invalid
        if stdout_frame_raw is not None or stdout_frame_invalid:
            return
        merged = stdout_frame_search_tail + block
        if not stdout_frame_candidate:
            index = merged.find(OBSERVATION_MAGIC)
            if index < 0:
                stdout_frame_search_tail = merged[-(len(OBSERVATION_MAGIC) - 1):]
                return
            stdout_frame_marker = True
            stdout_frame_candidate.extend(merged[index:])
            stdout_frame_search_tail = b""
        else:
            stdout_frame_candidate.extend(block)
        if len(stdout_frame_candidate) < OBSERVATION_HEADER_SIZE:
            return
        frame_end = _observation_frame_end(stdout_frame_candidate, 0)
        if frame_end is None:
            stdout_frame_invalid = True
            stdout_frame_candidate.clear()
            observation_frame_error = "observation-frame-memory-too-large"
            return
        if len(stdout_frame_candidate) >= frame_end:
            stdout_frame_raw = bytes(stdout_frame_candidate[:frame_end])

    def compact_line(pc: int) -> bytes:
        if trace_record_mode == "libriscv-translated":
            return f"f rq1-compact pc 0x{pc:x} instr 00000000\n".encode()
        return f"[0x{pc:x}] 00000000\n".encode()

    def remember_diagnostic(line: bytes, channel: str) -> None:
        if channel not in diagnostic_lines and diagnostic_pattern.search(line):
            diagnostic_lines[channel] = line

    def open_capture(path: Path | None, channel: str) -> Any | None:
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open("wb")
        except (OSError, ValueError) as exc:
            capture_error.append(f"{channel}-capture-open:{type(exc).__name__}:{exc}")
            return None

    def write_capture(capture: Any | None, block: bytes, channel: str) -> Any | None:
        if capture is None:
            return None
        try:
            capture.write(block)
        except (OSError, ValueError) as exc:
            capture_error.append(f"{channel}-capture-write:{type(exc).__name__}:{exc}")
            try:
                capture.close()
            except (OSError, ValueError):
                pass
            return None
        return capture

    def close_capture(capture: Any | None, channel: str, complete: list[bool]) -> None:
        if capture is None:
            return
        try:
            capture.close()
        except (OSError, ValueError) as exc:
            capture_error.append(f"{channel}-capture-close:{type(exc).__name__}:{exc}")
        else:
            complete[0] = True

    def consume_line(line: bytes, writer: Any) -> None:
        match = pattern.search(line)
        if match is None:
            return
        try:
            pc = int(match[1], 16)
        except (TypeError, ValueError):
            return
        trace_records_total[0] += 1
        if pc not in trace_seen:
            rendered = compact_line(pc)
            if writer is not None:
                writer.write(rendered)
            trace_seen.add(pc)

    def drain_stdout_with_trace(stream: Any) -> None:
        writer = None
        capture = open_capture(stdout_output_path, "stdout")
        try:
            compact_path.parent.mkdir(parents=True, exist_ok=True)
            writer = compact_path.open("wb")
            header = b"# RQ1-COMPACT-PC-V1; unique executed PCs\n"
            writer.write(header)
        except (OSError, ValueError) as exc:
            trace_error.append(f"trace-writer-open:{type(exc).__name__}:{exc}")
            if writer is not None:
                try:
                    writer.close()
                except (OSError, ValueError):
                    pass
                writer = None

        def consume_trace_line(line: bytes) -> None:
            nonlocal writer
            try:
                consume_line(line, writer)
            except (OSError, ValueError) as exc:
                trace_error.append(f"trace-writer-write:{type(exc).__name__}:{exc}")
                if writer is not None:
                    try:
                        writer.close()
                    except (OSError, ValueError):
                        pass
                    writer = None

        try:
            line_buffer = bytearray()
            while block := stream.read(1 << 20):
                stdout_total[0] += len(block)
                stdout_hash.update(block)
                capture = write_capture(capture, block, "stdout")
                feed_frame(block)
                line_buffer.extend(block)
                while True:
                    newline = line_buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(line_buffer[:newline + 1])
                    del line_buffer[:newline + 1]
                    remember_diagnostic(line, "stdout")
                    consume_trace_line(line)
            if line_buffer:
                remember_diagnostic(bytes(line_buffer), "stdout")
                consume_trace_line(bytes(line_buffer))
        except (OSError, ValueError) as exc:
            trace_error.append(f"stdout-reader:{type(exc).__name__}:{exc}")
        finally:
            close_capture(capture, "stdout", stdout_capture_complete)
            if writer is not None:
                try:
                    writer.close()
                except (OSError, ValueError) as exc:
                    trace_error.append(f"trace-writer-close:{type(exc).__name__}:{exc}")

    def drain_stderr(stream: Any) -> None:
        capture = open_capture(stderr_output_path, "stderr")
        try:
            line_buffer = bytearray()
            while block := stream.read(1 << 20):
                stderr_total[0] += len(block)
                stderr_hash.update(block)
                capture = write_capture(capture, block, "stderr")
                line_buffer.extend(block)
                while True:
                    newline = line_buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(line_buffer[:newline + 1])
                    del line_buffer[:newline + 1]
                    remember_diagnostic(line, "stderr")
            if line_buffer:
                remember_diagnostic(bytes(line_buffer), "stderr")
        except (OSError, ValueError) as exc:
            capture_error.append(f"stderr-reader:{type(exc).__name__}:{exc}")
        finally:
            close_capture(capture, "stderr", stderr_capture_complete)

    trace_thread = None
    stderr_thread = None

    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        process_pid = process.pid
        process_started = True
        process_group = os.getpgid(process.pid) if os.name == "posix" else None
        trace_thread = threading.Thread(
            target=drain_stdout_with_trace, args=(process.stdout,),
            name="rq1-compact-trace-reader", daemon=True,
        )
        trace_thread.start()
        stderr_thread = threading.Thread(
            target=drain_stderr, args=(process.stderr,),
            name="rq1-stderr-reader", daemon=True,
        )
        stderr_thread.start()
        if stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin_text.encode())
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        deadline = math.inf if timeout is None else (
            time.monotonic() + max(0.001, float(timeout))
        )
        while process.poll() is None:
            now = time.monotonic()
            if stop_requested():
                external_stop_requested = True
                if os.name == "posix":
                    victims |= _terminate_process_tree(process.pid)
                else:
                    try:
                        process.terminate()
                    except (OSError, ProcessLookupError):
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        victims |= _kill_process_tree(process.pid)
                    else:
                        try:
                            process.kill()
                        except (OSError, ProcessLookupError):
                            pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                _signal_pids(victims, signal.SIGKILL)
                status, returncode, termination = "gap", process.returncode, stop_reason()
                break
            if now >= deadline:
                if os.name == "posix":
                    victims |= _terminate_process_tree(process.pid)
                else:
                    try:
                        process.terminate()
                    except (OSError, ProcessLookupError):
                        pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        victims |= _kill_process_tree(process.pid)
                    else:
                        try:
                            process.kill()
                        except (OSError, ProcessLookupError):
                            pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                _signal_pids(victims, signal.SIGKILL)
                status, returncode, termination = "timeout", 124, "timeout"
                break
            time.sleep(min(0.01, max(0.001, deadline - now)))
        else:
            returncode = process.returncode
            status = "passed" if returncode == 0 else "failed"
            termination = "exit"
    except FileNotFoundError as exc:
        trace_error.append(f"launch:{exc}")
    except (OSError, ValueError) as exc:
        trace_error.append(f"launch:{type(exc).__name__}:{exc}")
    finally:
        _cleanup_processes(process_pid, process_group, victims)
        # A normal exit must drain every byte before returning.  A detached
        # descendant can keep the inherited stdout pipe open forever after the
        # parent exits, so detect and terminate that writer before joining.
        detached_writers = set()
        excluded = victims | ({process_pid} if process_pid else set())
        for stream in (
            getattr(process, "stdout", None),
            getattr(process, "stderr", None),
        ):
            detached_writers.update(_pipe_writer_pids(stream, excluded))
        if detached_writers:
            victims.update(detached_writers)
            _signal_pids(detached_writers, signal.SIGTERM)
            time.sleep(0.05)
            remaining_writers = set()
            for stream in (
                getattr(process, "stdout", None),
                getattr(process, "stderr", None),
            ):
                remaining_writers.update(_pipe_writer_pids(stream, excluded))
            if remaining_writers:
                victims.update(remaining_writers)
                _signal_pids(remaining_writers, signal.SIGKILL)
            trace_error.append("detached-pipe-writer")
        if status in {"timeout", "gap"} or detached_writers:
            for stream in (
                getattr(process, "stdout", None),
                getattr(process, "stderr", None),
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        for thread in (trace_thread, stderr_thread):
            if thread is not None:
                if status in {"timeout", "gap"} or detached_writers:
                    thread.join(timeout=1)
                else:
                    thread.join()
        for stream in (
            getattr(process, "stdout", None),
            getattr(process, "stderr", None),
        ):
            if stream is not None:
                stream.close()
        _reap_children(victims)

    if stdout_frame_candidate and stdout_frame_raw is None:
        observation_frame_error = "observation-frame-incomplete"
    raw_observation_frame = None if stdout_frame_raw == b"" else stdout_frame_raw
    observation_frame, frame_compatibility = _libriscv_compat_observation_frame(
        raw_observation_frame, trace_record_mode,
    )
    raw_frame_sha256 = (
        hashlib.sha256(raw_observation_frame).hexdigest()
        if frame_compatibility is not None else None
    )
    observation_complete = (
        observation_frame is not None
        and _finalized_observation_frame(observation_frame) == observation_frame
    )
    invalid_reason = None
    if raw_observation_frame is not None and not observation_complete:
        if len(raw_observation_frame) >= OBSERVATION_HEADER_SIZE \
                and struct.unpack_from("<Q", raw_observation_frame, 16)[0] != 0:
            invalid_reason = "guest-x0-nonzero"
    try:
        compact_size = compact_path.stat().st_size if compact_path.is_file() else 0
    except OSError:
        compact_size = 0
    result = {
        "status": status if status != "launch-failed" else "gap",
        "returncode": returncode,
        # Raw stdout/stderr are written to their requested files while the
        # stream is drained.  Keep no in-memory prefix/tail that could be
        # mistaken for the evidence artifact; semantic diagnostic lines are
        # retained separately and completely below.
        "stdout": "",
        "stderr": "",
        "stdout_diagnostics": diagnostic_lines.get("stdout", b"").decode("utf-8", "replace"),
        "stderr_diagnostics": diagnostic_lines.get("stderr", b"").decode("utf-8", "replace"),
        "stdout_contains_observation_frame": stdout_frame_marker,
        "stdout_observation_frame_complete": observation_complete,
        "observation_frame_verified": observation_complete,
        "observation_frame_error": observation_frame_error,
        "observation_frame_invalid_reason": invalid_reason,
        "observation_frame_compatibility": frame_compatibility,
        "observation_frame_raw_sha256": raw_frame_sha256,
        "observation_frame_sha256": (
            hashlib.sha256(observation_frame).hexdigest()
            if observation_complete else None
        ),
        "raw_stdout_sha256": stdout_hash.hexdigest(),
        "raw_stderr_sha256": stderr_hash.hexdigest(),
        "stdout_bytes_total": stdout_total[0],
        "stderr_bytes_total": stderr_total[0],
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_capture_complete": stdout_capture_complete[0],
        "stderr_capture_complete": stderr_capture_complete[0],
        "elapsed_s": round(time.monotonic() - started, 6),
        "process_started": process_started,
        "process_executed": process_started,
        "terminal_observed": status == "timeout",
        "process_pid": process_pid,
        "termination": termination,
        "external_stop_requested": external_stop_requested,
        "descendant_count": len(victims),
        "trace_capture_mode": "unique-pc-stream",
        "trace_records": len(trace_seen),
        "trace_records_total": trace_records_total[0],
        "trace_unique_pcs": len(trace_seen),
        "trace_size_bytes": compact_size,
        "trace_truncated": status in {"timeout", "gap"} or bool(trace_error),
        "trace_complete": status not in {"timeout", "gap"} and not trace_error,
        "trace_cleanup_status": "not-required",
        "trace_writer_pids": [],
        "trace_writer_remaining": [],
    }
    if trace_error:
        result.update({
            "trace_reason": "trace-capture-failed",
            "trace_capture_error": "; ".join(trace_error),
        })
    if capture_error:
        result["capture_error"] = "; ".join(capture_error)
    if status == "timeout":
        result["reason"] = "wall-timeout"
    elif external_stop_requested:
        result["reason"] = stop_reason()
        result["reason_code"] = stop_reason()
    return result


def _run_logged(out: Path | None, label: str, argv: list[str], timeout: float | None, *,
                cwd: Path | None = None, env: dict[str, str] | None = None,
                stdin_text: str | None = None,
                timeout_stdin_text: str | None = None,
                stdout_output_path: Path | None = None,
                trace_path: Path | None = None,
                trace_watch_pcs: set[int] | None = None,
                trace_watch_plain_pcs: bool = False,
                trace_compact_path: Path | None = None,
                trace_record_mode: str | None = None,
                stop_on_output: tuple[str, ...] | None = None,
                stop_on_output_stdin_text: str | None = None,
                trace_watch_signal: int = signal.SIGINT,
                graceful_shutdown_seconds: float | None = None,
                graceful_shutdown_fallback_signal: int | None = None,
                graceful_shutdown_fallback_after_seconds: float | None = None,
                timeout_graceful_shutdown_seconds: float | None = None) -> dict[str, Any]:
    checkpoint = None
    capture_stdout_path = stdout_output_path
    capture_stderr_path = None
    if out:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        capture_stdout_path = capture_stdout_path or out / "logs" / f"{safe}.stdout"
        capture_stderr_path = out / "logs" / f"{safe}.stderr"
        checkpoint = out / "logs" / f"{safe}.start.json"
        write_json(checkpoint, {"argv": argv, "cwd": str(cwd) if cwd else None,
                                "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        result = run_cmd(argv, timeout, cwd=cwd, env=env, stdin_text=stdin_text,
                         timeout_stdin_text=timeout_stdin_text,
                         stdout_output_path=capture_stdout_path,
                         stderr_output_path=capture_stderr_path, trace_path=trace_path,
                         trace_watch_pcs=trace_watch_pcs,
                         trace_watch_plain_pcs=trace_watch_plain_pcs,
                         trace_compact_path=trace_compact_path,
                         trace_record_mode=trace_record_mode,
                         stop_on_output=stop_on_output,
                         stop_on_output_stdin_text=stop_on_output_stdin_text,
                         trace_watch_signal=trace_watch_signal,
                         graceful_shutdown_seconds=graceful_shutdown_seconds,
                         graceful_shutdown_fallback_signal=graceful_shutdown_fallback_signal,
                         graceful_shutdown_fallback_after_seconds=(
                             graceful_shutdown_fallback_after_seconds
                         ),
                         timeout_graceful_shutdown_seconds=timeout_graceful_shutdown_seconds)
        return _persist_run(out, label, argv, result, cwd=cwd, env=env) if out else result
    finally:
        if checkpoint and checkpoint.is_file():
            checkpoint.unlink()


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}


_expected_timebox_event = _event_is_expected_timebox


def _shared_deadline_censored(result: Mapping[str, Any] | None,
                              deadline: float | None) -> bool:
    """Identify a stage stopped by the shared experiment cutoff."""
    if not isinstance(result, Mapping):
        return False
    if result.get("deadline_censored") is True or result.get("run_deadline_censored") is True:
        return True
    reason = result.get("reason")
    if reason in {"generation-deadline-reached", "run-wall-clock-exhausted"}:
        return True
    return bool(
        deadline is not None
        and time.monotonic() >= deadline
        and result.get("status") == "timeout"
        and reason in {"wall-timeout", "generation-deadline-reached"}
    )


def _csmith_source_capture_complete(result: Mapping[str, Any]) -> bool:
    """Reject a Csmith candidate whose stdout capture was explicitly partial."""
    return (
        isinstance(result, Mapping)
        and result.get("stdout_truncated") is not True
        and result.get("stdout_capture_complete") is not False
    )


def _result_diagnostic_text(result: dict[str, Any]) -> str:
    """Return semantic command text without replacing the raw output artifact."""
    values = []
    for name in ("stdout", "stderr", "stdout_diagnostics", "stderr_diagnostics"):
        value = result.get(name)
        if value in (None, "", b""):
            continue
        values.append(
            value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        )
    return "\n".join(values)


_UNSUPPORTED_ISA_PATTERN = re.compile(
    r"(?:illegal\s+(?:instruction|opcode)|invalid\s+instruction|"
    r"unimplemented\s+(?:instruction|opcode)|unsupported\s+"
    r"(?:instruction|opcode|extension|architecture)|unknown\s+instruction)",
    re.IGNORECASE,
)


def _unsupported_isa_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Classify a target's explicit unsupported ISA diagnostic.

    This diagnostic does not prove a guest terminal.  Preserve it as a reason
    so the event layer can distinguish an ordinary skip from an unverified
    expected-trap case.
    """
    if not _UNSUPPORTED_ISA_PATTERN.search(_result_diagnostic_text(result)):
        return None
    return {
        **result,
        "status": "failed",
        "reason": "unsupported-isa",
        "reason_code": "unsupported-isa",
        "unsupported_isa": True,
        "observer_complete": False,
        "terminal_observed": False,
        "termination": "unsupported-isa",
        "observation_gap": "target does not implement the emitted ISA/extension",
    }


def _target_record(
    target: dict[str, Any], result: dict[str, Any], identity: dict[str, Any],
    reference: dict[str, Any], feature_reference: dict[str, str],
) -> dict[str, Any]:
    """把一个 adapter 结果投影成统一的 Target observation 记录。"""
    attempted = _process_executed(result)
    record = {
        "id": target["id"], "kind": target["kind"], "identity": identity,
        "artifact_features": feature_reference,
        "static_evidence": feature_reference,
        "reference_id": reference.get("id"),
        "reference_status": reference.get("status"),
        **_result_summary(result),
    }
    for field in (
        "process_started", "process_executed", "executed", "observer_complete", "target_observed",
        "observation_status", "outcome", "terminal", "terminal_observed",
        "terminal_outcome",
        "target_tested", "target_outcome", "semantic_outcome", "guest_cause",
        "guest_epc", "guest_tval", "guest_status", "tohost_value",
        "program_stdout_sha256",
        "observation_channel", "observation_frame_sha256",
        "observation_frame_verified", "returncode",
    ):
        record.pop(field, None)
    record["target_attempted"] = attempted
    record["target_dispatch"] = _target_dispatch_event(target["id"])
    record.update(_observation_record(result))
    return record


def _build_linux_capsule(
    source: Path, output: Path, params: dict[str, Any], *,
    deadline: float | None = None,
) -> dict[str, Any]:
    if deadline is not None and math.isfinite(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"status": "timeout", "reason": "generation-deadline-reached"}
        params = {**params, "timeout": remaining}
    try:
        build_linux_user(source, output, params)
    except (OSError, RuntimeError, ValueError) as error:
        return {"status": "gap", "reason": str(error)}
    return {"status": "passed", "output": str(output)}


def _ensure_rax_entry(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if not re.search(r"(?m)^\s*_start:\s*$", text):
        return
    localized = _localize_tohost(text)
    aliases = []
    if not re.search(r"(?m)^\s*(?:\.globl\s+)?test_0\s*$", localized):
        aliases += [".globl test_0", ".set test_0, _start"]
    if not re.search(r"(?m)^\s*(?:\.globl\s+)?test_entry\s*(?::|$)", localized):
        aliases += [".globl test_entry", ".set test_entry, _start"]
    if aliases or localized != text:
        source.write_text("\n".join(aliases) + ("\n" if aliases else "") + localized, encoding="utf-8")


_RENODE_DEFAULT_TRAP_VECTOR = 0x1010


def _renode_result(result: dict[str, Any]) -> dict[str, Any]:
    trace_pc_guard = (
        result.get("termination") == "guest-invalid-state"
        and result.get("reason") == "guest-pc-outside-elf-loop"
    )
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    trace_aborted = result.get("termination") == "trace-limit" \
        or result.get("trace_limit_exceeded") is True
    if trace_aborted:
        # 轨迹异常只影响覆盖率，不应抹掉已经拿到的客体终态。
        result = {
            **result,
            "trace_truncated": True,
            "trace_complete": False,
            "trace_reason": result.get("trace_reason") or "trace-limit-exceeded",
        }
    frame_lines = re.findall(r"(?m)^\s*RQ1_RVOBS1=([0-9a-fA-F]+)\s*$", text)
    value_match = re.search(r"RQ1_TOHOST_VALUE\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)", text)
    tohost_marker = re.search(r"(?m)^\s*RQ1_TOHOST\s*$", text) is not None
    value = int(value_match.group(1), 0) if value_match else None
    frame = b""
    if len(frame_lines) == 1:
        try:
            frame = bytes.fromhex(frame_lines[0])
        except ValueError:
            frame = b""
    finalized_frame = bool(frame and _finalized_observation_frame(frame) == frame)
    guard_pc_match = re.search(
        r"guest-pc-outside-elf-loop:0x([0-9a-fA-F]+)",
        str(result.get("trace_reason") or ""),
    )
    default_trap_sentinel = bool(
        guard_pc_match
        and int(guard_pc_match.group(1), 16) == _RENODE_DEFAULT_TRAP_VECTOR
    )
    terminal_protocol_evidence = (
        (finalized_frame and value_match is not None)
        or (tohost_marker and value_match is not None)
    )
    terminal_trace_closed = bool(
        trace_pc_guard
        and default_trap_sentinel
        and terminal_protocol_evidence
        and result.get("trace_cleanup_status") == "clean"
        and type(result.get("trace_size_bytes")) is int
        and result["trace_size_bytes"] > 0
    )
    if trace_pc_guard and not (default_trap_sentinel and terminal_protocol_evidence):
        return {
            **result,
            "status": "failed",
            "terminal_observed": True,
            "observer_complete": False,
            "guest_status": "failure",
            "guest_exit_channel": "renode-trace-pc-guard",
        }
    if trace_pc_guard:
        # The terminal trap is outside PT_LOAD by design. Keep execution
        # outcome from the verified protocol. A clean trace writer and the
        # explicit terminal protocol close the guest ELF prefix; coverage
        # parsing drops the external sentinel suffix without changing the file.
        close_reason = result.get("trace_reason") or (
            f"guest-pc-outside-elf-loop:0x{_RENODE_DEFAULT_TRAP_VECTOR:x}"
        )
        result = {
            **result,
            "trace_truncated": not terminal_trace_closed,
            "trace_complete": terminal_trace_closed,
            "trace_reason": close_reason,
            **({"trace_terminal_close_reason": close_reason}
               if terminal_trace_closed else {}),
        }
    if finalized_frame:
        frame_digest = hashlib.sha256(frame).hexdigest()
        # The frame proves that the guest state was captured, but it does not
        # encode the tohost result.  Use the explicit value when present; an
        # old bare marker remains an outcome gap instead of being promoted to
        # pass by the frame branch.
        if value is None:
            return {**result, "status": "gap",
                    "execution_status": result.get("execution_status") or result.get("status"),
                    "execution_reason": result.get("reason"),
                    "returncode": result.get("returncode"),
                    "reason": "observer-tohost-value-missing",
                    "tohost_value": None, "guest_status": "unknown",
                    "terminal_observed": False, "termination": "guest-tohost",
                    # A valid frame is useful evidence, but without the
                    # protocol result value this event is not comparable.
                    "observer_complete": False, "observation_channel": "rvobs1-frame",
                    "observation_frame_verified": True,
                    "observation_frame_sha256": frame_digest,
                    "program_stdout_sha256": frame_digest}
        failed = value != 1
        return {**result, "status": "failed" if failed else "passed",
                "returncode": 1 if failed else 0,
                "reason": "guest-tohost-failure" if failed else "guest-tohost",
                "tohost_value": value, "guest_status": "failure" if failed else "pass",
                "terminal_observed": True,
                "termination": "guest-tohost",
                "observer_complete": True, "observation_channel": "rvobs1-frame",
                "observation_frame_verified": True,
                "observation_frame_sha256": frame_digest,
                "program_stdout_sha256": frame_digest}
    unsupported = _unsupported_isa_result(result)
    if tohost_marker and value_match:
        # 只有协议值才能证明是 pass；任意写入 marker 不能替代终态值。
        updates = {"status": "passed" if value == 1 else "failed",
                   "returncode": 0 if value == 1 else 1,
                   "reason": "guest-tohost" if value == 1 else "guest-tohost-failure",
                   "tohost_value": value, "guest_status": "pass" if value == 1 else "failure",
                   "observer_complete": False, "terminal_observed": True,
                   "termination": "guest-tohost",
                   "observation_gap": "Renode did not emit a complete RVOBS1 frame"}
        if result.get("status") == "timeout":
            updates.update(
                termination="guest-terminal-after-timeout",
                host_timeout_after_terminal=True,
            )
        return {**result, **updates}
    if tohost_marker:
        return {**result, "status": "gap",
                "execution_status": result.get("execution_status") or result.get("status"),
                "execution_reason": result.get("reason"),
                "reason": "observer-tohost-value-missing",
                "observer_complete": False, "terminal_observed": False}
    if "RQ1_EXCEPTION" in text:
        return {**result, "status": "failed", "returncode": result.get("returncode") or 1,
                "reason": "guest-exception", "observer_complete": False,
                "terminal_observed": True, "termination": "guest-exception",
                "guest_exit_channel": "renode-monitor", "guest_status": "failure"}
    if "CPU abort" in text:
        return {**result, "status": "failed", "returncode": result.get("returncode") or 1,
                "reason": "guest-exception", "observer_complete": False,
                "terminal_observed": True, "termination": "guest-exception",
                "guest_exit_channel": "renode-log", "guest_status": "failure"}
    if result.get("status") == "timeout":
        return {**result, "termination": "target-timeout"}
    if unsupported is not None:
        return unsupported
    if result.get("status") != "passed":
        return result
    return {**result, "status": "gap",
            "execution_status": result.get("execution_status") or result.get("status"),
            "execution_reason": result.get("reason"),
            "reason": "observer-tohost-missing", "observer_complete": False}



def _csmith_portable_smoke(out: Path, bare_elf: Path, linux_elf: Path,
                           qemu: Path | None, renode: Path | None,
                           dotnet: Path | None, timeout: int,
                           deadline: float | None = None) -> dict[str, Any]:
    def stage_budget() -> int:
        remaining = timeout if deadline is None else min(
            timeout, max(0, math.floor(deadline - time.monotonic()))
        )
        return max(0, remaining)

    qemu_timeout = stage_budget()
    if not qemu:
        qemu_result = {"status": "gap", "reason": "qemu-smoke-missing"}
    elif qemu_timeout <= 0:
        qemu_result = {"status": "timeout", "reason": "generation-deadline-reached"}
    else:
        qemu_result = _run_logged(
            out, f"tool-B-CSMITH-qemu-smoke-{bare_elf.stem}",
            [str(qemu), str(linux_elf)], qemu_timeout,
        )
    if qemu_result.get("status") != "passed":
        return {"passed": False, "qemu": _result_summary(qemu_result)}
    script = out / "csmith-renode-smoke" / f"{bare_elf.stem}.resc"
    script.parent.mkdir(parents=True, exist_ok=True)
    try:
        _, smoke_symbols = symbol_offsets_from_elf(
            bare_elf, ("tohost",), allow_before_start=True,
        )
        hook = hex(int(smoke_symbols["tohost"], 0))
    except (KeyError, OSError, ValueError):
        return {"passed": False, "qemu": _result_summary(qemu_result),
                "renode": {"status": "gap", "reason": "renode-smoke-tohost-missing"}}
    quote, nl = chr(34), chr(10)
    try:
        _, segments = elf_segments(bare_elf)
        memory_base = min(address & ~(PAGE - 1) for address, _, _, _ in segments)
        memory_end = max(address + max(file_size, mem_size)
                         for address, _, file_size, mem_size in segments)
        memory_size = max(PAGE, ((memory_end - memory_base + PAGE - 1) // PAGE) * PAGE)
    except (OSError, struct.error, ValueError):
        memory_base, memory_size = 0x80000000, 0x01000000
    script_text = nl.join((
        "mach create",
        'machine LoadPlatformDescriptionFromString """',
        "cpu: CPU.RiscV64 @ sysbus",
        '    cpuType: "rv64im_zicsr"',
        "    allowUnalignedAccesses: true",
        "    timeProvider: empty",
        f'mem: Memory.MappedMemory @ sysbus 0x{memory_base:x}',
        f"    size: 0x{memory_size:x}",
        "trap: Memory.MappedMemory @ sysbus 0x1000",
        "    size: 0x1000",
        '"""',
        f"sysbus LoadELF @{bare_elf}",
        "sysbus WriteDoubleWord 0x1010 0x00100073",
        "sysbus.cpu MTVEC 0x1010",
        "python " + chr(34) + "monitor.Machine.UserState = 'RQ1_WAIT'" + chr(34),
        f"sysbus.cpu AddHookAtInterruptBegin {quote}if monitor.Machine.UserState != 'RQ1_DONE': print('RQ1_EXCEPTION'); monitor.Machine.UserState = 'RQ1_EXCEPTION'{quote}",
        f"sysbus AddWatchpointHook {hook} Word Write {quote}if monitor.Machine.UserState != 'RQ1_DONE' and value != 0: print('RQ1_TOHOST_VALUE=' + str(value)); print('RQ1_TOHOST'); monitor.Machine.UserState = 'RQ1_DONE'{quote}",
        f"sysbus AddWatchpointHook {hook} DoubleWord Write {quote}if monitor.Machine.UserState != 'RQ1_DONE' and value != 0: print('RQ1_TOHOST_VALUE=' + str(value)); print('RQ1_TOHOST'); monitor.Machine.UserState = 'RQ1_DONE'{quote}",
        f"sysbus AddWatchpointHook {hook} QuadWord Write {quote}if monitor.Machine.UserState != 'RQ1_DONE' and value != 0: print('RQ1_TOHOST_VALUE=' + str(value)); print('RQ1_TOHOST'); monitor.Machine.UserState = 'RQ1_DONE'{quote}",
        "start",
        'set wait_script """',
        "import time",
        "machine = monitor.Machine",
        "while machine is not None and machine.UserState not in ('RQ1_DONE', 'RQ1_EXCEPTION'):",
        "    time.sleep(0.001)",
        f"if machine is not None and machine.UserState == 'RQ1_DONE':",
        f"    while sum(int(machine.SystemBus.ReadByte({int(smoke_symbols['tohost'], 0)} + i)) << (8 * i) for i in range(8)) == 0:",
        "        time.sleep(0.001)",
        "if machine is not None:",
        "    machine.Pause()",
        '"""',
        "python $wait_script",
        "quit",
        "",
    ))
    write_text_once(script, script_text)
    renode_timeout = stage_budget()
    renode_result = _run_logged(
        out, f"tool-B-CSMITH-renode-smoke-{bare_elf.stem}",
        [str(dotnet), str(renode), "--disable-gui", "--plain", "--console", str(script)],
        renode_timeout, cwd=renode.parent if renode else None,
        # Renode's redirected console reader spins on EOF. Keep its input
        # pipe open and idle so a long trace cannot grow that queue forever.
        stdin_text="", timeout_stdin_text="",
        env={**os.environ, "DOTNET_ROOT": str(dotnet.parent) if dotnet else "",
             "DOTNET_CLI_HOME": "/tmp", "HOME": "/tmp",
             "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"},
    ) if renode and dotnet and renode_timeout > 0 else {
        "status": "timeout" if deadline is not None and renode_timeout <= 0 else "gap",
        "reason": "generation-deadline-reached"
        if deadline is not None and renode_timeout <= 0 else "renode-smoke-missing",
    }
    renode_result = _renode_result(renode_result)
    text = f"{renode_result.get('stdout', '')}{nl}{renode_result.get('stderr', '')}"
    return {"passed": renode_result.get("status") == "passed" and "RQ1_TOHOST" in text,
            "qemu": _result_summary(qemu_result), "renode": _result_summary(renode_result)}


def _libriscv_result(result: dict[str, Any]) -> dict[str, Any]:
    text = _result_diagnostic_text(result)
    if re.search(r"instruction count limit reached|fuel exhausted", text, re.IGNORECASE):
        return {**result, "status": "timeout", "returncode": 124,
                "reason": "instruction-fuel-exhausted", "termination": "simulator-fuel",
                "fuel_exhausted": True, "trace_complete": False}
    if result.get("status") in {"timeout", "gap"}:
        return result
    if re.search(r"machine exception|unimplemented|exception", text, re.IGNORECASE):
        return {**result, "status": "failed", "returncode": result.get("returncode") or 1,
                "reason": "guest-exception", "observer_complete": False,
                "terminal_observed": True, "termination": "guest-exception",
                "guest_exit_channel": "libriscv-exception", "guest_status": "failure"}
    if result.get("status") != "passed":
        return result
    return result


def _expected_outcome_result(result: dict[str, Any], expected_outcome: str | None) -> dict[str, Any]:
    if expected_outcome != "trap" or result.get("status") == "timeout":
        return result
    has_trap_evidence = (
        result.get("guest_trap") == "delivered"
        or type(result.get("guest_cause")) is int
    )
    if (result.get("status") == "failed"
            and result.get("returncode") not in {0, None}
            and has_trap_evidence):
        return {**result, "status": "passed", "reason": "expected-trap",
                "semantic_outcome": "expected-trap"}
    return result


def _qemu_expected_trap_evidence(
    target: dict[str, Any], result: dict[str, Any], expected_outcome: str | None,
    elf: Path | None, out: Path,
) -> dict[str, Any]:
    """Attribute expected SIGILL from QEMU's structured guest siginfo."""
    if (
        target.get("kind") != "qemu"
        or expected_outcome != "trap"
        or result.get("returncode") != -signal.SIGILL
        or elf is None
        or not elf.is_file()
    ):
        return result

    try:
        from framework.cli.qemu_trace_runner import (
            _strace_siginfo_state, _trap_state_from_siginfo,
        )
        stderr_parts = []
        for name in ("stderr", "stderr_diagnostics"):
            value = result.get(name)
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if isinstance(value, str) and value:
                stderr_parts.append(value)
        diagnostics = "\n".join(stderr_parts)
        siginfo = _strace_siginfo_state(diagnostics, "SIGILL")
        trace_value = result.get("trace_path")
        if siginfo is None and isinstance(trace_value, str) and trace_value:
            trace_path = Path(trace_value)
            if not trace_path.is_absolute():
                trace_path = out / trace_path
            try:
                # QEMU -D redirects -strace signal lines into the run log.
                with trace_path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        siginfo = _strace_siginfo_state(line, "SIGILL")
                        if siginfo is not None:
                            break
            except OSError:
                pass
        trap = _trap_state_from_siginfo(
            result["returncode"], siginfo, str(elf),
        )
    except (ImportError, OSError, TypeError, ValueError, struct.error):
        return result
    fault_pc = trap.get("guest_epc") if isinstance(trap, dict) else None
    if (
        not isinstance(trap, dict)
        or trap.get("guest_cause") != 2
        or type(fault_pc) is not int
        or fault_pc < 0
        or fault_pc >= 1 << 64
        or fault_pc & 1
    ):
        return result

    instruction_word = trap.get("guest_tval")
    siginfo = siginfo if isinstance(siginfo, dict) else {}
    evidence = {
        "signal": "SIGILL",
        "signal_code": siginfo.get("si_code"),
        "signal_code_name": siginfo.get("si_code_name"),
        "pc_source": "qemu-strace-si_addr",
        "execution_pc": f"0x{fault_pc:x}",
        "elf_path": str(elf),
        "elf_instruction_observed": instruction_word is not None,
    }
    if instruction_word is not None:
        evidence["instruction_word"] = f"0x{instruction_word:x}"
    return {
        **result,
        "qemu_expected_trap_verified": True,
        "qemu_expected_trap_evidence": evidence,
        "guest_status": "failure",
        "terminal_observed": True,
        "termination": "guest-trap",
        "observer_complete": False,
        "guest_exit_channel": "qemu-strace-siginfo",
        **trap,
    }


def _rax_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("termination") == "guest-invalid-state" and result.get("reason") == "guest-pc-outside-elf-loop":
        return {
            **result,
            "status": "failed",
            "terminal_observed": True,
            "observer_complete": False,
            "guest_status": "failure",
            "guest_exit_channel": "rax-trace-pc-guard",
        }
    if result.get("status") == "timeout":
        return result
    text = _result_diagnostic_text(result)
    tohost_match = re.search(
        r"(?m)^\s*RQ1_TOHOST_VALUE\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*$",
        text,
    )
    tohost_value = int(tohost_match.group(1), 0) if tohost_match else None
    trap = re.search(
        r"riscv trap:\s*cause=(0x[0-9a-fA-F]+|[0-9]+)\b", text, re.IGNORECASE,
    )
    if trap:
        cause_text = trap.group(1)
        cause = int(cause_text, 16) if cause_text.lower().startswith("0x") else int(cause_text)
        return {**result, "status": "failed", "returncode": result.get("returncode") or 1,
                "reason": "rax-guest-trap", "observer_complete": False,
                "guest_trap": "delivered", "guest_cause": cause,
                "guest_exit_channel": "rax-trap-log", "guest_status": "failure",
                "terminal_observed": True, "termination": "guest-trap"}
    tohost_failure = re.search(
        r"(?:unsupported )?riscv tohost (?:failure: value=|value:\s*)(0x[0-9a-fA-F]+|[0-9]+)",
        text, re.IGNORECASE,
    )
    if tohost_failure:
        value = int(tohost_failure.group(1), 0)
        return {**result, "status": "failed", "returncode": result.get("returncode") or 1,
                "reason": "guest-tohost-failure", "tohost_value": value,
                "guest_status": "failure", "guest_exit_channel": "rax-vcpu-tohost",
                "terminal_observed": True, "termination": "guest-tohost-failure",
                "observer_complete": False}
    if (
        result.get("status") in {"passed", "failed"}
        and result.get("stdout_observation_frame_complete") is True
        and result.get("observation_frame_verified") is True
        and _valid_hex(result.get("observation_frame_sha256"), 64)
    ):
        failed = (
            result.get("status") == "failed"
            or result.get("returncode") not in {0, None}
            or (tohost_value is not None and tohost_value != 1)
        )
        return {
            **result,
            "status": "failed" if failed else "passed",
            "reason": result.get("reason") if failed else "rax-rvobs1-frame",
            "tohost_value": tohost_value if tohost_value is not None else result.get("tohost_value"),
            "observer_complete": True,
            "terminal_observed": True,
            "guest_status": "failure" if failed else "pass",
            "guest_exit_channel": "rax-rvobs1-uart",
            "termination": "rax-rvobs1-frame",
            "observation_channel": "rvobs1-frame",
            "observation_frame_verified": True,
        }
    if result.get("status") in {"passed", "failed"}:
        shutdown = re.search(
            r"vCPU shutdown \(unsupported architecture\)|unsupported architecture|unsupported riscv",
            text, re.IGNORECASE,
        )
        return {
            **result,
            "status": "gap",
            "reason": "rax-shutdown-origin-unverified" if shutdown
                      else "rax-guest-outcome-unverified",
            "reason_code": "unsupported-isa" if shutdown else "rax-guest-outcome-unverified",
            "unsupported_isa": bool(shutdown),
            "observer_complete": False,
            "terminal_observed": False,
            "guest_status": "unknown",
            "guest_exit_channel": "rax-vcpu-shutdown" if shutdown else "rax-process-exit",
            "termination": "unsupported-isa" if shutdown else "rax-process-exit",
        }
    return result


def _renode_trace_setup(trace_path: Path) -> str:
    """Render Renode's native PC tracer for ordinary target observation."""
    return f'''sysbus.cpu CreateExecutionTracing "rq1-pc" @{trace_path} PC
'''


def _riscv_dv_trial(root: Path, out: Path, timeout: float | None, seed: int, label: str,
                    profile: dict[str, Any] | None = None, *,
                    deadline: float | None = None) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="rq1-rvdv-", dir="/tmp"))
    try:
        return _riscv_dv_trial_impl(
            root, out, timeout, seed, label, profile=profile, temp_root=temp_root,
            deadline=deadline,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _riscv_dv_trial_impl(root: Path, out: Path, timeout: float | None, seed: int, label: str,
                         profile: dict[str, Any] | None = None, *,
                         temp_root: Path,
                         deadline: float | None = None) -> dict[str, Any]:
    result_case = out / "tools" / "riscv-dv" / label / f"seed-{seed}"
    case = temp_root / label / f"seed-{seed}"
    case.mkdir(parents=True)
    profile = profile or {}
    isa = str(profile.get("isa", "rv64imc_zicsr"))
    mabi = str(profile.get("mabi", "lp64"))
    target = str(profile.get("target", isa.split("_", 1)[0]))
    deadline = deadline if deadline is not None else (
        time.monotonic() + timeout if timeout is not None else math.inf
    )
    tests = profile.get("tests") if isinstance(profile.get("tests"), list) else []
    selected = tests[seed % len(tests)] if tests else {}
    test_name = selected.get("name", "riscv_rand_instr_test")
    opts = [f"+instr_cnt={int(selected.get('instr_cnt', 10000))}",
            f"+num_of_sub_program={int(selected.get('sub_programs', 5))}",
            f"+boot_mode={profile.get('boot_mode', 'm')}",
            # Linux-user and bare target capsules both need a deterministic
            # architectural stack; otherwise riscv-dv randomizes x2 in init.
            "+fix_sp=1"]
    for index, item in enumerate(selected.get("directed", [])):
        opts.append(f"+directed_instr_{index}={item[0]},{int(item[1])}")
    for name in ("no_fence", "no_data_page", "no_branch_jump", "no_csr_instr", "bare_program_mode",
                 "enable_floating_point", "enable_vector_extension"):
        if name in profile:
            opts.append(f"+{name}={int(profile[name])}")
    iterations = max(1, int(profile.get("iterations", 1)))
    testlist = case / "pretrial-testlist.yaml"
    write_text_once(testlist, "- test: %s\n  gen_opts: >\n    %s\n  iterations: %d\n  gen_test: riscv_instr_base_test\n  rtl_test: core_base_test\n" %
                    (test_name, "\n    ".join(opts), iterations))
    generation_timeout = deadline - time.monotonic()
    if generation_timeout <= 0:
        return {
            "generator_invoked": False,
            "generation_status": "timeout",
            "generation_reason": "generation-deadline-reached",
            "generated": False,
            "built": False,
            "candidate_artifacts": [],
            "candidate_count": 0,
        }
    generation_limit = (
        max(1, math.floor(generation_timeout))
        if math.isfinite(generation_timeout) else None
    )
    compat = HERE / "container" / "assets" / "rvdv"
    compat_runner = compat / "run_compat.py"
    batch_size = max(1, int(profile.get("batch_size", min(iterations, 4))))
    # Use the pinned target directory as an explicit custom target.  This keeps
    # the requested ABI/ISA instead of letting run.py replace it with the
    # predefined target default, while pyflow still imports the matching
    # pygen target module by name.
    custom_target = root / "target" / target
    command_line = [sys.executable, str(compat_runner), str(root),
                    "--target", target, "--custom_target", str(custom_target),
                    "--isa", isa, "--mabi", mabi, "--output", str(case),
                    "--testlist", str(testlist),
                    "--test", test_name, "--iterations", str(iterations),
                    "--batch_size", str(batch_size),
                    "--simulator", "pyflow", "--steps", "gen",
                    *(["--gen_timeout", str(generation_limit)]
                      if generation_limit is not None else []),
                    "--seed" if iterations == 1 else "--start_seed", str(seed)]
    pygen = root / "pygen"
    rdv_deps = root / ".analysis-out" / "rdv-deps"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        filter(None, (str(compat), str(rdv_deps), str(pygen), os.environ.get("PYTHONPATH"))))}
    env.pop("RQ1_RVDV_CONTINUOUS_GENERATION", None)
    if generation_limit is None:
        env["RQ1_RVDV_CONTINUOUS_GENERATION"] = "1"
    run_remaining = (
        None if math.isinf(deadline) else deadline - time.monotonic()
    )
    run = (
        _run_logged(out, f"tool-{label}-generation", command_line,
                    run_remaining, cwd=root, env=env)
        if run_remaining is None or run_remaining > 0 else
        {"status": "timeout", "reason": "generation-deadline-reached",
         "returncode": 124}
    )
    sources = sorted(case.rglob("*.S"), key=lambda path: path.stat().st_size if path.is_file() else 0) \
        if case.is_dir() else []
    compiler = command("riscv64-linux-gnu-gcc")
    include = root / "user_extension"
    linker = root / "scripts" / "link.ld"
    result_case.mkdir(parents=True, exist_ok=True)
    candidates = []
    builds = []
    linux_builds = []
    for index, source_path in enumerate(sources):
        if time.monotonic() >= deadline:
            break
        destination = result_case / "asm_test" / f"{label}-{index:04d}.S"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        _ensure_rax_entry(destination)
        os.chmod(destination, 0o755)
        elf_path = destination.with_name(f"{label}-{index:04d}-bare.elf")
        for name in ("user_init.s", "user_define.h"):
            dependency = include / name
            if dependency.is_file():
                shutil.copyfile(dependency, destination.parent / name)
        # RISC-V-DV's bare_program_mode is an integration fragment, not a
        # standalone executable.  Build the bare role through the same small
        # RVOBS1 harness used for the Linux role, replacing its exit syscall
        # with tohost=1.  This preserves the generated body while giving all
        # bare simulators one observable terminal/state contract.
        build = _build_linux_capsule(
            destination, elf_path,
            {"isa": isa, "mabi": mabi, "harness": "riscv-dv",
             "gcc_opts": "-mcmodel=medany -fno-pie",
             "linker": str(linker), "transport": "tohost", "bare_metal": True},
            deadline=deadline,
        ) if compiler and include.is_dir() and linker.is_file() else {
            "status": "gap", "reason": "rvdv-cross-compile-input-missing"
        }
        builds.append(build)
        linux_elf = destination.with_name(f"{label}-{index:04d}-linux.elf")
        if destination.is_file():
            if build.get("status") == "passed" and elf_path.is_file():
                os.chmod(elf_path, 0o755)
            linux_build = _build_linux_capsule(
                destination, linux_elf,
                {"isa": isa, "mabi": mabi, "harness": "riscv-dv",
                 "gcc_opts": "-mcmodel=medany -fno-pie", "text_address": "0x10000"},
                deadline=deadline,
            )
        else:
            linux_build = {"status": "gap", "reason": "bare-elf-build-failed"}
        linux_builds.append(linux_build)
        bare_ok = build.get("status") == "passed" and elf_path.is_file()
        linux_ok = linux_build.get("status") == "passed" and linux_elf.is_file()
        if bare_ok or linux_ok:
            candidate = {
                "index": index, "source": str(destination.relative_to(out)),
                "source_sha256": sha256(destination),
            }
            if bare_ok:
                candidate.update(
                    elf=str(elf_path.relative_to(out)), elf_sha256=sha256(elf_path),
                    artifact_isa=isa,
                    bare_harness=str(elf_path.with_suffix(".S").relative_to(out)),
                    bare_harness_sha256=sha256(elf_path.with_suffix(".S")),
                )
            if linux_ok:
                candidate.update(
                    linux_elf=str(linux_elf.relative_to(out)),
                    linux_elf_sha256=sha256(linux_elf), linux_artifact_isa=isa,
                )
            candidates.append(candidate)
    candidate_artifacts = candidates
    last_candidate = candidates[-1] if candidates else {}
    source = out / last_candidate["source"] if candidates else None
    elf = out / last_candidate["elf"] if last_candidate.get("elf") else result_case / "asm_test" / "missing.elf"
    linux_elf = out / last_candidate["linux_elf"] if last_candidate.get("linux_elf") else result_case / "asm_test" / "missing-linux.elf"
    source_hash = sha256(source) if source else None
    build = builds[-1] if builds else {"status": "gap", "reason": "rvdv-source-missing"}
    linux_build = linux_builds[-1] if linux_builds else {"status": "gap", "reason": "rvdv-source-missing"}
    generation_reason = (
        "generation-deadline-reached" if time.monotonic() >= deadline else
        stop_reason() if stop_requested() else
        "riscv-dv-generator-timeout" if run.get("status") == "timeout" else
        "generation-failed" if run.get("status") == "failed" else
        "riscv-dv-source-missing" if not sources else None
    )
    result = {"generator_invoked": True, "generator": _result_summary(run),
              "generation_status": run.get("status"),
              "generation_reason": generation_reason,
              "requested_iterations": iterations, "batch_size": batch_size,
              "instr_cnt": int(selected.get("instr_cnt", 10000)),
              "boot_mode": profile.get("boot_mode", "m"),
              "source_generated": len(sources),
              "bare_built": sum(1 for item in builds if item.get("status") == "passed"),
              "linux_built": sum(1 for item in linux_builds if item.get("status") == "passed"),
              "source": str(source.relative_to(out)) if source else None,
              "source_sha256": source_hash,
              "generated": bool(sources),
              "cross_compile": _result_summary(build),
              "elf": str(elf.relative_to(out)) if elf.is_file() else None,
              "elf_sha256": sha256(elf),
              "linux_cross_compile": _result_summary(linux_build),
              "linux_elf": str(linux_elf.relative_to(out)) if linux_elf.is_file() else None,
              "linux_elf_sha256": sha256(linux_elf),
              "built": bool(candidate_artifacts), "candidate_artifacts": candidate_artifacts,
              "candidate_count": len(candidate_artifacts), "pair_ready": bool(candidate_artifacts),
              "artifact_isa": isa, "linux_artifact_isa": isa, "generator_profile": profile,
              "compatibility_module": str(compat.relative_to(HERE)),
              "compatibility_module_sha256": sha256(compat / "imp.py"),
              "compatibility_runner": str(compat_runner.relative_to(HERE)),
              "compatibility_runner_sha256": sha256(compat_runner)}
    return result



def _torture_trial(root: Path, out: Path, timeout: float | None, profile: dict[str, Any] | None = None,
                   case_index: int = 0, *, deadline: float | None = None) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="rq1-torture-", dir="/tmp"))
    try:
        return _torture_trial_impl(
            root, out, timeout, profile=profile, case_index=case_index,
            temp_root=temp_root, deadline=deadline,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _torture_trial_impl(root: Path, out: Path, timeout: float | None,
                        profile: dict[str, Any] | None = None,
                        case_index: int = 0, *, temp_root: Path,
                        deadline: float | None = None) -> dict[str, Any]:
    deadline = deadline if deadline is not None else (
        time.monotonic() + timeout if timeout is not None else math.inf
    )
    def run_phase(label: str, argv: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
        remaining = None if math.isinf(deadline) else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return {"status": "timeout", "reason": "generation-deadline-reached",
                    "returncode": 124}
        return _run_logged(out, label, argv, remaining, cwd=cwd)

    result_root = out / "tools" / "torture" / f"case-{case_index:04d}"
    work = temp_root / "work"
    shutil.copytree(root, work, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "output"))
    (work / "output").mkdir()
    config = work / "rq1-pretrial.config"
    shutil.copyfile(HERE / "container" / "assets" / "torture-pretrial.config", config)
    profile = profile or {}
    config_text = config.read_text(encoding="utf-8")
    def set_torture_option(name: str, value: object) -> None:
        nonlocal config_text
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        config_text, count = re.subn(
            rf"(?m)^torture\.generator\.{re.escape(name)}\s+[^\n]+$",
            f"torture.generator.{name} {encoded}",
            config_text,
        )
        if count != 1:
            raise ValueError(f"torture option missing: {name}")
    for name in ("nseqs", "memsize", "fprnd", "amo", "mul", "divider", "segment", "loop", "loop_size"):
        if name in profile:
            set_torture_option(name, profile[name])
    vector = profile.get("vector")
    if isinstance(vector, dict):
        for name, value in vector.items():
            set_torture_option(f"vec.{name}", value)
    mix = profile.get("mix")
    if isinstance(mix, dict):
        for name, value in mix.items():
            set_torture_option(f"mix.{name}", value)
    config.write_text(config_text, encoding="utf-8")
    java = os.environ.get("RQ1_JAVA17") or command("java")
    cache_value = os.environ.get("RQ1_TORTURE_CACHE")
    cache = Path(cache_value) if cache_value else None
    cache_copy = temp_root / "cache"
    cache_dirs = ("boot", "coursier", "global", "ivy2", "env", "precompiled")
    cache_mode = None
    cache_extract = {"status": "gap", "reason": "cache-not-provided"}
    if cache and (cache / "cache.tar.gz").is_file():
        cache_copy.mkdir()
        tar = command("tar")
        cache_extract = run_phase(
            f"tool-B-TORTURE-{case_index:04d}-cache-extract",
            [tar, "-xzf", str(cache / "cache.tar.gz"), "-C", str(cache_copy)],
        ) if tar else {"status": "gap", "reason": "tar-runtime-missing"}
        cache_mode = "archive-to-private-tmpfs" if cache_extract["status"] == "passed" else None
    elif cache and cache.is_dir() and all((cache / name).is_dir() for name in cache_dirs):
        cache_copy.mkdir()
        copier = command("cp")
        cache_extract = run_phase(
            f"tool-B-TORTURE-{case_index:04d}-cache-copy",
            [copier, "-a", *[str(cache / name) for name in cache_dirs], str(cache_copy)],
        ) if copier else {"status": "gap", "reason": "copy-runtime-missing"}
        cache_mode = "directory-to-private-tmpfs" if cache_extract["status"] == "passed" else None
    env_source = cache_copy / "env"
    if env_source.is_dir():
        shutil.copytree(env_source, work / "env", dirs_exist_ok=True)
    for cached, destination in (("generator-target", "generator/target"),
                                ("project-target", "project/target")):
        path = cache_copy / "precompiled" / cached
        if path.is_dir():
            shutil.copytree(path, work / destination, dirs_exist_ok=True)
    sbt_root = cache_copy if cache_extract["status"] == "passed" else temp_root / "sbt"
    command_line = ([java, "-Xmx6G", "-Xss8M", f"-Dsbt.boot.directory={sbt_root}/boot",
                     f"-Dsbt.global.base={sbt_root}/global", f"-Dsbt.ivy.home={sbt_root}/ivy2",
                     f"-Dsbt.coursier.home={sbt_root}/coursier", "-jar", "sbt-launch.jar",
                     "generator/run -C rq1-pretrial.config -o rq1_pretrial"] if java else [])
    run = run_phase(f"tool-B-TORTURE-{case_index:04d}-generation",
                    command_line, cwd=work) if command_line else {
        "status": "gap", "reason": "java-runtime-missing"}
    source = work / "output" / "rq1_pretrial.S"
    if not source.is_file():
        source = None
    generator_source_path = None
    generator_source_sha256 = None
    if source:
        generator_source_path = work / "output" / "rq1_pretrial.generated.S"
        generator_source_sha256 = sha256(source)
        shutil.copyfile(source, generator_source_path)
        _ensure_rax_entry(source)
    original_source_text = source.read_text(encoding="utf-8") if source else None
    source_hash = sha256(source) if source else None
    precompat_source_path = None
    precompat_source_sha256 = source_hash
    if source:
        precompat_source_path = work / "output" / "rq1_pretrial.precompat.S"
        precompat_source_path.write_text(original_source_text, encoding="utf-8")
    compat = HERE / "container" / "assets" / "torture-exit-transport-v2.json"
    header = work / "env" / "p" / "riscv_test.h"
    compat_data = load_json(compat) if compat.is_file() else {}
    rules = compat_data.get("rules", [])
    header_before = sha256(header) if header.is_file() else None
    applied = []
    compat_ready = bool(source) and header.is_file() and bool(rules)
    if compat_ready:
        # 统一退出信号：ecall -> torture 自身的 tohost 协议。特权态测试主体
        # （csr/mret）与 header 其余语义保持上游原样，不再为最弱 Target 削弱输入。
        for path in (header, source):
            text = path.read_text(encoding="utf-8")
            for rule in rules:
                text, count = re.subn(
                    rule["pattern"],
                    lambda _m, value=rule["replacement"]: value,
                    text, flags=re.MULTILINE)
                applied.append({"file": path.name, "pattern": rule["pattern"], "count": count})
            path.write_text(text, encoding="utf-8")
        source_hash = sha256(source)
        encoding = work / "env" / "encoding.h"
        if encoding.is_file():
            shutil.copyfile(encoding, work / "output" / "encoding.h")
    # 规则命中数是合同的判据：0 命中说明退出协议没被改写，
    # 只检查「规则加载成功」会把空操作报成 passed。
    rule_hits = sum(int(item.get("count") or 0) for item in applied)
    compat_ready = compat_ready and rule_hits > 0
    observer_transport: dict[str, Any] = {
        "status": "gap", "reason": "torture-observer-not-bound",
    }
    if compat_ready:
        try:
            observer_transport = _add_torture_observer(
                source, header,
                xlen=32 if str(profile.get("isa", "rv64")).lower().startswith("rv32") else 64,
            )
        except (OSError, RuntimeError, ValueError) as error:
            observer_transport = {
                "status": "gap",
                "reason": "torture-observer-binding-failed",
                "error": f"{type(error).__name__}: {error}",
            }
        compat_ready = compat_ready and observer_transport.get("status") == "passed"
        if compat_ready:
            # The observer is part of the final bare source artifact; record
            # its digest after injection so the lineage points to the bytes
            # that were actually compiled.
            source_hash = sha256(source)
    compatibility = {
        "status": "passed" if compat_ready else "gap",
        "rule_hits": rule_hits,
        "module": str(compat.relative_to(HERE)),
        "module_sha256": sha256(compat),
        "header_before_sha256": header_before,
        "header_after_sha256": sha256(header) if compat_ready else None,
        "rules": applied,
        "reason": None if compat_ready else (
            observer_transport.get("reason")
            if observer_transport.get("status") != "passed" else
            "torture-exit-transport-no-rule-hit" if not rule_hits
            else "torture-exit-transport-application-failed"),
        "observation_transport": observer_transport,
    }
    bare_header_snapshot_path = None
    bare_header_snapshot_sha256 = None
    if header.is_file():
        bare_header_snapshot_path = (
            work / "output" / "torture-lineage" / "bare-riscv_test.h"
        )
        bare_header_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, bare_header_snapshot_path)
        bare_header_snapshot_sha256 = sha256(bare_header_snapshot_path)
    compiler = command("riscv64-linux-gnu-gcc")
    include = work / "env" / "p"
    linker = include / "link.ld"
    elf = work / "output" / "rq1_pretrial.elf"
    build = run_phase(
        f"tool-B-TORTURE-{case_index:04d}-cross-compile",
        [compiler, "-static", "-mcmodel=medany", "-fno-pie", "-no-pie", "-nostdlib", "-nostartfiles",
         f"-I{include}", f"-T{linker}", str(source), "-o", str(elf),
         f"-march={profile.get('isa', 'rv64i_zicsr')}", f"-mabi={profile.get('mabi', 'lp64')}",
         "-Wl,--build-id=none", "-Wl,--no-relax"],
    ) if source and compiler and include.is_dir() and linker.is_file() and compat_ready else {
        "status": "gap", "reason": "torture-test-env-missing"
    }
    linux_preprocess = {"status": "gap", "reason": "torture-linux-adapter-missing"}
    linux_build = {"status": "gap", "reason": "torture-linux-preprocess-failed"}
    linux_dir = work / "output" / "torture-linux"
    linux_elf = linux_dir / "rq1_pretrial-linux.elf"
    linux_source = None
    linux_preprocessed_raw = None
    linux_source_sha256 = None
    linux_header_sha256 = None
    linux_preprocessed_sha256 = None
    linux_adapter_source_sha256 = None
    linux_compatibility = {
        "status": "not-applied",
        "rewrites": 0,
        "contract": "libriscv-translated-loop-counter-v1",
    }
    linux_header_asset = HERE / "container" / "assets" / "torture-linux-riscv_test.h"
    if source and compiler and linux_header_asset.is_file():
        linux_dir.mkdir(parents=True, exist_ok=True)
        linux_source = linux_dir / "torture-source.S"
        linux_pre = linux_dir / "torture-preprocessed.S"
        linux_preprocessed_raw = linux_dir / "torture-preprocessed.raw.S"
        linux_source.write_text(original_source_text, encoding="utf-8")
        linux_source_sha256 = sha256(linux_source)
        shutil.copyfile(linux_header_asset, linux_dir / "riscv_test.h")
        linux_header_sha256 = sha256(linux_dir / "riscv_test.h")
        linux_preprocess = run_phase(
            f"tool-B-TORTURE-{case_index:04d}-linux-preprocess",
            [compiler, "-E", "-P", "-I", str(linux_dir), str(linux_source), "-o", str(linux_pre)],
            cwd=linux_dir,
        )
        if linux_preprocess.get("status") == "passed":
            shutil.copyfile(linux_pre, linux_preprocessed_raw)
            linux_preprocessed_sha256 = sha256(linux_preprocessed_raw)
            adapted = linux_pre.read_text(encoding="utf-8").replace(";", "\n")
            adapted = re.sub(r"(?m)^\s*j test_start\s*$", "", adapted, count=1)
            adapted, renamed = re.subn(
                r"(?m)^test_start:\s*$", "torture_test_entry:", adapted, count=1,
            )
            if renamed == 1 and "test_start" not in adapted:
                adapted, loop_counter_rewrites = re.subn(
                    r"(?m)^(\s*)lw x2, 0\(x1\)\n"
                    r"\1addi x3, x2, -1\n"
                    r"\1sw x3, 0\(x1\)\n"
                    r"\1bnez x2, pseg_0$",
                    r"\1lw x5, 0(x1)\n"
                    r"\1addi x6, x5, -1\n"
                    r"\1sw x6, 0(x1)\n"
                    r"\1bnez x5, pseg_0",
                    adapted,
                    count=1,
                )
                linux_compatibility = {
                    "status": "applied" if loop_counter_rewrites else "not-needed",
                    "rewrites": loop_counter_rewrites,
                    "contract": "libriscv-translated-loop-counter-v1",
                    "original_registers": ["x2", "x3"],
                    "adapted_registers": ["x5", "x6"],
                }
                linux_pre.write_text(adapted, encoding="utf-8")
                linux_adapter_source_sha256 = sha256(linux_pre)
                linux_build = _build_linux_capsule(
                    linux_pre, linux_elf,
                    {"isa": str(profile.get("isa", "rv64i_zicsr")),
                     "mabi": str(profile.get("mabi", "lp64")), "harness": "custom",
                     "sail_prologue_lines": ["  j torture_test_entry"],
                     "gcc_opts": "-mcmodel=medany -fno-pie", "text_address": "0x10000"},
                    deadline=deadline,
                )
    source_rel = source.relative_to(work / "output") if source else None
    generator_source_rel = (
        generator_source_path.relative_to(work / "output")
        if generator_source_path and generator_source_path.is_file() else None
    )
    precompat_source_rel = (
        precompat_source_path.relative_to(work / "output")
        if precompat_source_path and precompat_source_path.is_file() else None
    )
    bare_header_rel = (
        bare_header_snapshot_path.relative_to(work / "output")
        if bare_header_snapshot_path and bare_header_snapshot_path.is_file() else None
    )
    elf_rel = elf.relative_to(work / "output") if elf.is_file() else None
    linux_source_rel = (
        linux_source.relative_to(work / "output")
        if linux_source and linux_source.is_file() else None
    )
    linux_header_path = linux_dir / "riscv_test.h"
    linux_header_rel = (
        linux_header_path.relative_to(work / "output")
        if linux_header_path.is_file() else None
    )
    linux_preprocessed_rel = (
        linux_preprocessed_raw.relative_to(work / "output")
        if linux_preprocessed_raw and linux_preprocessed_raw.is_file() else None
    )
    linux_adapter_source_path = linux_dir / "torture-preprocessed.S"
    linux_adapter_source_rel = (
        linux_adapter_source_path.relative_to(work / "output")
        if linux_adapter_source_sha256 else None
    )
    linux_rel = linux_elf.relative_to(work / "output") if linux_elf.is_file() else None
    if (work / "output").is_dir():
        shutil.copytree(work / "output", result_root / "output", dirs_exist_ok=True)
    source = result_root / "output" / source_rel if source_rel else None
    def lineage_artifact(
        relative: Path | None, checksum: str | None,
    ) -> dict[str, str] | None:
        if relative is None or checksum is None:
            return None
        artifact = result_root / "output" / relative
        return {"path": str(artifact.relative_to(out)), "sha256": checksum}

    source_lineage = {
        "generator_output": lineage_artifact(
            generator_source_rel, generator_source_sha256,
        ),
        "pre_compat": lineage_artifact(precompat_source_rel, precompat_source_sha256),
        "compat_rewritten": lineage_artifact(source_rel, source_hash),
        "bare_header_after_compat": lineage_artifact(
            bare_header_rel, bare_header_snapshot_sha256,
        ),
        "linux_adapter_input": lineage_artifact(linux_source_rel, linux_source_sha256),
        "linux_header": lineage_artifact(linux_header_rel, linux_header_sha256),
        "linux_preprocessed": lineage_artifact(
            linux_preprocessed_rel, linux_preprocessed_sha256,
        ),
        "linux_adapter_output": lineage_artifact(
            linux_adapter_source_rel, linux_adapter_source_sha256,
        ),
    }
    source_lineage = {
        name: artifact for name, artifact in source_lineage.items()
        if artifact is not None
    }
    elf = result_root / "output" / elf_rel if elf_rel else result_root / "output" / "rq1_pretrial.elf"
    linux_elf = result_root / "output" / linux_rel if linux_rel else result_root / "output" / "torture-linux" / "rq1_pretrial-linux.elf"
    linux_built = linux_build.get("status") == "passed" and linux_elf.is_file()
    reason = (
        "torture-cache-extract-timeout" if cache_extract.get("status") == "timeout" else
        "torture-generator-timeout" if run.get("status") == "timeout" else
        "torture-source-missing" if not source else
        "torture-bare-build-failed" if build.get("status") != "passed" else
        "torture-linux-build-failed" if not linux_built else None
    )
    return {"generator_invoked": bool(command_line), "java_runtime": java,
            "cache_mode": cache_mode,
            "cache_extract": _result_summary(cache_extract),
            "cache_manifest_sha256": sha256(cache / "cache-manifest.json") if cache and cache.is_dir() else None,
            "test_env_manifest_sha256": sha256(cache / "env-manifest.json") if cache and cache.is_dir() else None,
            "config_sha256": sha256(config),
            "generator": _result_summary(run),
            "reason": reason,
            "source": str(source.relative_to(out)) if source else None,
            "source_sha256": source_hash,
            "source_lineage_schema": "rq1-torture-source-lineage-v1",
            "source_lineage": source_lineage,
            "generated": bool(source) and run.get("status") == "passed",
            "compatibility": compatibility,
            "cross_compile": _result_summary(build), "elf": str(elf.relative_to(out)) if elf.is_file() else None,
            "elf_sha256": sha256(elf), "built": build.get("status") == "passed" and elf.is_file(),
            "linux_preprocess": _result_summary(linux_preprocess),
            "linux_compatibility": linux_compatibility,
            "linux_cross_compile": _result_summary(linux_build),
            "linux_elf": str(linux_elf.relative_to(out)) if linux_built else None,
            "linux_elf_sha256": sha256(linux_elf), "linux_built": linux_built,
            "artifact_isa": str(profile.get("isa", "rv64i_zicsr")),
            "generator_profile": profile}


def _provenance(config: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = config if isinstance(config, dict) else {}
    policy = config.get("execution_policy") if isinstance(config.get("execution_policy"), dict) else {}
    wrapper_path = HERE / "container" / "run-linux.sh"
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path.resolve()),
        "config_value_sha256": digest(config),
        "runner_sha256": sha256(HERE / "runner.py"),
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": sha256(wrapper_path),
        "image_digest": os.environ.get("RQ1_IMAGE_DIGEST"),
        "expected_image_digest": policy.get(
            "coverage_container_image_digest" if simulator_coverage_enabled()
            else "container_image_digest"
        ),
        "execution_plane": os.environ.get("RQ1_EXECUTION_PLANE", "unknown"),
        "network": os.environ.get("RQ1_NETWORK", "unknown"),
    }


def _identity(path: Path | None, binary: Path | None, target: dict[str, Any], *,
              verify_binary: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {}
    error = None
    file_present = bool(path and path.is_file())
    if file_present:
        try:
            loaded = load_json(path)
            if not isinstance(loaded, dict):
                raise ValueError("identity JSON must be an object")
            data = loaded
        except (OSError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    actual = sha256(binary) if binary and binary.is_file() else None
    payload_sha = data.get("binary_sha256")
    expected = target.get("binary_sha256")
    identity_digest = target.get("identity_digest")
    source_commit = target.get("commit")
    binary_present = bool(binary and binary.is_file())
    commit_match = bool(
        _valid_commit(data.get("source_commit")) and _valid_commit(source_commit)
        and data.get("source_commit") == source_commit
    )
    source_provenance = data.get("source_provenance")
    has_source_patch = bool(data.get("patch_lineage")) or source_provenance in {
        "patch-validated-build", "verified-git-archive+controlled-patch",
    }
    allowlisted_patch = is_allowlisted_patched_identity(data)
    source_provenance_ok = allowlisted_patch or (not has_source_patch and (
        data.get("schema_version") != "target-binary-identity-v3"
        or source_provenance in {
            "verified-build", "verified-checkout-clean-tree", "verified-git-archive",
        }
    ))
    expected_backend = TARGET_BACKENDS.get(target.get("id"))
    backend_match = expected_backend is None or data.get("backend") == expected_backend
    binary_match = None if not (expected or payload_sha) else bool(
        actual and ((not expected) or expected == actual) and ((not payload_sha) or payload_sha == actual))
    identity_match = None if not identity_digest else bool(
        re.fullmatch(r"[0-9a-f]{64}", str(identity_digest))
        and data.get("identity_digest") == identity_digest)
    reason = (
        "identity-file-invalid" if error else
        "identity-file-missing" if not file_present else
        "identity-payload-empty" if not data else
        "identity-binary-missing" if verify_binary and not binary_present else
        "identity-binary-sha-mismatch" if verify_binary and binary_match is False else
        "source-commit-mismatch" if not commit_match else
        "identity-digest-mismatch" if identity_digest and identity_match is False else
        "target-source-patch-forbidden" if has_source_patch and not allowlisted_patch else
        "source-provenance-unverified" if not source_provenance_ok else
        "identity-backend-mismatch" if not backend_match else None
    )
    return {
        "path": str(path) if path else None,
        "present": bool(data),
        "file_present": file_present,
        "identity_digest": data.get("identity_digest"),
        "expected_identity_digest": identity_digest,
        "identity_payload_digest": None,
        "identity_payload_match": identity_match,
        "expected_binary_sha256": expected,
        "actual_binary_sha256": actual,
        "binary_match": binary_match,
        "binary_sha256_verified": binary_match is True,
        "identity_match": identity_match,
        "container_image": data.get("container_image"),
        "container_image_digest": data.get("container_image_digest"),
        "source_commit": data.get("source_commit"),
        "expected_source_commit": source_commit,
        "commit_match": commit_match,
        "source_provenance": source_provenance,
        "patch_lineage": data.get("patch_lineage"),
        "allowlisted_patch": allowlisted_patch,
        "source_provenance_ok": source_provenance_ok,
        "backend": data.get("backend"),
        "expected_backend": expected_backend,
        "backend_match": backend_match,
        "error": error,
        "reason_code": reason,
        "status": ("verified" if verify_binary else "recorded")
        if data and (binary_present if verify_binary else True) and (binary_match is not False if verify_binary else True)
        and commit_match and (identity_match is not False if identity_digest else True)
        and source_provenance_ok and backend_match
         else "gap",
    }


def _target_stratum(execution_model: object) -> str:
    return (
        "linux-user"
        if str(execution_model or "").startswith("linux-user")
        else "bare-metal"
    )


def _reference_for_target(config: dict[str, Any], target: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stratum = _target_stratum(target.get("execution_model"))
    refs = config.get("references", {})
    return stratum, refs.get(stratum, {})


# K1 只执行候选的 linux-user ELF；bare-metal 形态登记为具名 gap 而不是失败。
BOARD_LINUX_USER_ONLY = "board-runs-linux-user-elf-only"


def _board_reference_identity(status: str = "recorded", source: str | None = None) -> dict[str, Any]:
    return {
        "id": "R-K1-BOARD", "engine": "k1-board", "backend": "native-rv64",
        "target": "T-K1-BOARD", "status": status,
        "source": source or "framework.native.rv64_runner",
        "transport": "jtag-openocd-gdb" if source == "framework.native.k1_jtag_runner" else "ssh-linux-user",
        "ssh_alias": os.environ.get("RQ1_NATIVE_SSH_ALIAS", "k1-board"),
    }


def _board_reference_gap(
    stratum: str, reason: str, returncode: int | None = None,
    failure_class: str | None = None,
) -> dict[str, Any]:
    identity = _board_reference_identity("gap")
    return {
        "id": identity["id"], "engine": identity["engine"], "backend": identity["backend"],
        "target": identity["target"], "stratum": stratum, "status": "gap",
        "observer_complete": False, "returncode": returncode, "terminal": "gap",
        "reason": reason, "identity": identity, "reference_identity": identity,
        **({"failure_class": failure_class} if failure_class else {}),
    }


def _board_reference_operational_warnings(
    text: str, remote_dir_marker_path: Path | None = None,
) -> list[dict[str, str]]:
    warnings = []
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "NATIVE_RV64_CLEANUP_GAP":
            warnings.append({"kind": "remote-cleanup-gap", "detail": value})
        elif key == "NATIVE_RV64_TRACE_CANCEL_GAP":
            warnings.append({"kind": "remote-trace-cancel-gap", "detail": value})
        elif key == "NATIVE_RV64_PRESERVED_REMOTE_DIR":
            warnings.append({"kind": "remote-run-preserved", "path": value})
    if remote_dir_marker_path is not None:
        try:
            remote_dir = remote_dir_marker_path.read_text(encoding="utf-8").strip()
        except OSError:
            remote_dir = ""
        if remote_dir and not any(
            warning.get("kind") == "remote-run-preserved"
            and warning.get("path") == remote_dir
            for warning in warnings
        ):
            warnings.append({"kind": "remote-run-preserved", "path": remote_dir})
    return warnings


def _cancel_active_k1_trace(
    env: dict[str, str], *, marker_wait_seconds: float = 0.0,
) -> tuple[int, bool] | None:
    try:
        from framework.native.rv64_runner import cancel_active_remote_trace

        return cancel_active_remote_trace(env, marker_wait_seconds=marker_wait_seconds)
    except Exception:
        return None


def _with_board_reference_warnings(
    record: dict[str, Any], text: str = "",
    remote_dir_marker_path: Path | None = None,
    trace_cancel_result: tuple[int, bool] | None = None,
) -> dict[str, Any]:
    warnings = _board_reference_operational_warnings(text, remote_dir_marker_path)
    if trace_cancel_result is not None and not trace_cancel_result[1]:
        warnings.append({
            "kind": "remote-trace-cancel-gap",
            "detail": f"process-group-{trace_cancel_result[0]}",
        })
    if warnings:
        record["operational_warnings"] = warnings
    return record


def _board_reference_record(
    out: Path, stratum: str, elf: Path | None, timeout: int,
    *, run_deadline: float | None = None,
) -> dict[str, Any]:
    # K1 是原生 RISC-V Linux 主机：候选的两个 ELF 里只执行 linux-user 形态，
    # bare-metal 形态需要板上 JTAG 探针，本机没有，登记具名 gap。
    if stratum != "linux-user":
        return _board_reference_gap(stratum, BOARD_LINUX_USER_ONLY)
    if not elf or not elf.is_file():
        return _board_reference_gap(stratum, "reference-artifact-missing")
    ssh_config = Path(os.environ.get("RQ1_NATIVE_SSH_CONFIG", ""))
    known_hosts = Path(os.environ.get("RQ1_NATIVE_KNOWN_HOSTS", ""))
    if not ssh_config.is_file() or not known_hosts.is_file():
        return _board_reference_gap(
            stratum, "k1-board-ssh-material-missing",
            failure_class="k1-transport-unavailable",
        )
    remaining = None if run_deadline is None else run_deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        return _board_reference_gap(stratum, "run-wall-clock-exhausted")
    guest_timeout = max(1, int(timeout))
    if remaining is not None:
        guest_timeout = max(1, min(guest_timeout, math.floor(remaining)))
    remote_timeout = max(2, guest_timeout * 2 + 30)
    if remaining is not None:
        remote_timeout = max(1, min(remote_timeout, math.floor(remaining)))
    # K1 has its own guest-startup timeout. A fixed host wall limit here would
    # cap the full ptrace replay; only the enclosing run deadline may do that.
    host_timeout = None if remaining is None else max(0.001, remaining)
    env = {**os.environ, "RQ1_NATIVE_SSH_CONFIG": str(ssh_config),
           "RQ1_NATIVE_KNOWN_HOSTS": str(known_hosts),
           "RQ1_NATIVE_GUEST_TIMEOUT": str(guest_timeout),
           "RQ1_NATIVE_REMOTE_TIMEOUT": str(remote_timeout)}
    command = [sys.executable, "-m", "framework.native.rv64_runner",
               "--stratum", stratum, str(elf)]
    request_deadline = (
        None if host_timeout is None else time.monotonic() + host_timeout
    )
    lock_deadline = (
        request_deadline
        if request_deadline is not None
        else time.monotonic() + float(remote_timeout * 3 + 150)
    )
    reference_lock = None
    remote_dir_marker_path = None
    trace_cancel_result: tuple[int, bool] | None = None
    while reference_lock is None:
        if stop_requested():
            return _board_reference_gap(stratum, stop_reason())
        remaining_lock = lock_deadline - time.monotonic()
        if remaining_lock <= 0:
            reason = (
                "run-wall-clock-exhausted"
                if run_deadline is not None and time.monotonic() >= run_deadline
                else "k1-board-reference-lock-timeout"
            )
            return _board_reference_gap(
                stratum, reason,
                failure_class="k1-timeout"
                if reason == "k1-board-reference-lock-timeout" else None,
            )
        attempt = _NativeReferenceLock(
            min(lock_deadline, time.monotonic() + min(0.1, remaining_lock)),
        )
        try:
            attempt.__enter__()
            reference_lock = attempt
        except TimeoutError:
            continue
        except OSError as error:
            return _board_reference_gap(
                stratum,
                f"k1-board-reference-lock-unavailable:{type(error).__name__}",
                failure_class="k1-transport-unavailable",
            )
    try:
        # ponytail: share the measured 8-slot K1 cap across reference callers.
        if stop_requested():
            return _board_reference_gap(stratum, stop_reason())
        if run_deadline is not None and time.monotonic() >= run_deadline:
            return _board_reference_gap(stratum, "run-wall-clock-exhausted")
        ref_dir = out / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        remote_dir_marker_path = ref_dir / (
            f"k1-board-{stratum}-{uuid.uuid4().hex}.remote-dir"
        )
        env["RQ1_NATIVE_REMOTE_DIR_MARKER"] = str(remote_dir_marker_path)
        env["RQ1_NATIVE_REMOTE_TRACE_PID_MARKER"] = str(
            remote_dir_marker_path.with_suffix(".trace-pid")
        )
        env["RQ1_NATIVE_REFERENCE_LOCK_HELD"] = "1"
        process = subprocess.Popen(
            command, cwd=HERE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        process_timeout = (
            None if request_deadline is None
            else max(0.001, request_deadline - time.monotonic())
        )
        if run_deadline is not None:
            remaining_run = max(0.001, run_deadline - time.monotonic())
            process_timeout = (
                remaining_run
                if process_timeout is None else min(process_timeout, remaining_run)
            )
        external_stop_requested = False
        stop_watcher_stop = threading.Event()
        victims: set[int] = set()

        def process_running() -> bool:
            poll = getattr(process, "poll", None)
            if callable(poll):
                return poll() is None
            return getattr(process, "returncode", None) is None

        def watch_external_stop() -> None:
            nonlocal external_stop_requested, trace_cancel_result
            while not stop_watcher_stop.wait(0.05):
                if stop_requested() and process_running():
                    external_stop_requested = True
                    trace_cancel_result = _cancel_active_k1_trace(
                        env, marker_wait_seconds=1.0,
                    )
                    process_pid = getattr(process, "pid", None)
                    victims.update(
                        _terminate_process_tree(process_pid)
                        if isinstance(process_pid, int) else set()
                    )
                    return

        stop_watcher = threading.Thread(
            target=watch_external_stop, name="rq1-board-stop-watcher", daemon=True,
        )
        stop_watcher.start()
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            trace_cancel_result = _cancel_active_k1_trace(
                env, marker_wait_seconds=0.5,
            )
            process_pid = getattr(process, "pid", None)
            victims.update(
                _terminate_process_tree(process_pid)
                if isinstance(process_pid, int) else set()
            )
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if isinstance(process_pid, int):
                    victims |= _kill_process_tree(process_pid)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            _signal_pids(victims, signal.SIGKILL)
        finally:
            stop_watcher_stop.set()
            stop_watcher.join(timeout=0.2)
        if external_stop_requested:
            return _with_board_reference_warnings(
                _board_reference_gap(stratum, stop_reason()),
                remote_dir_marker_path=remote_dir_marker_path,
                trace_cancel_result=trace_cancel_result,
            )
        if timed_out:
            reason = (
                "run-wall-clock-exhausted"
                if run_deadline is not None and time.monotonic() >= run_deadline
                else "k1-board-reference-timeout"
            )
            return _with_board_reference_warnings(
                _board_reference_gap(
                    stratum, reason,
                    failure_class="k1-timeout" if reason == "k1-board-reference-timeout" else None,
                ),
                remote_dir_marker_path=remote_dir_marker_path,
                trace_cancel_result=trace_cancel_result,
            )
        process = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        trace_cancel_result = _cancel_active_k1_trace(
            env, marker_wait_seconds=0.5,
        )
        process_pid = getattr(process, "pid", None)
        victims = (
            _terminate_process_tree(process_pid)
            if isinstance(process_pid, int) else set()
        )
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if isinstance(process_pid, int):
                victims |= _kill_process_tree(process_pid)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        _signal_pids(victims, signal.SIGKILL)
        reason = (
            "run-wall-clock-exhausted"
            if run_deadline is not None and time.monotonic() >= run_deadline
            else "k1-board-reference-timeout"
        )
        return _with_board_reference_warnings(
            _board_reference_gap(
                stratum, reason,
                failure_class="k1-timeout" if reason == "k1-board-reference-timeout" else None,
            ),
            remote_dir_marker_path=remote_dir_marker_path,
            trace_cancel_result=trace_cancel_result,
        )
    except OSError as exc:
        return _with_board_reference_warnings(
            _board_reference_gap(
                stratum, f"k1-board-reference-launch-failed:{type(exc).__name__}",
            ),
            remote_dir_marker_path=remote_dir_marker_path,
        )
    finally:
        reference_lock.__exit__(None, None, None)
    stdout = process.stdout or b""
    stderr = process.stderr or b""
    ref_dir = out / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = ref_dir / f"k1-board-{stratum}.stdout.bin"
    stderr_path = ref_dir / f"k1-board-{stratum}.stderr.log"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    text = stderr.decode("utf-8", "replace")
    operational_warnings = _board_reference_operational_warnings(
        text, remote_dir_marker_path,
    )
    gap = next((line.split("=", 1)[1] for line in text.splitlines()
                if line.startswith("RV_NATIVE_REFERENCE_GAP=")), None)
    if gap is None:
        # rv64_runner 的契约失败走 RV_RUNNER_CONTRACT_GAP；不透传就会被吞成
        # 笼统的 k1-board-reference-contract-missing，744 次失败无法归因。
        gap = next((line.split("=", 1)[1] for line in text.splitlines()
                    if line.startswith("RV_RUNNER_CONTRACT_GAP=")), None)
    if gap:
        reason = gap
        failure_class = None
        try:
            payload = json.loads(gap)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            reason = str(payload.get("reason") or gap)
            failure_class = payload.get("failure_class")
        if failure_class not in {"k1-timeout", "k1-transport-unavailable"}:
            failure_class = None
        record = _board_reference_gap(
            stratum, reason, process.returncode, failure_class,
        )
        record.update({
            "stdout_path": str(stdout_path.relative_to(out)),
            "stderr_path": str(stderr_path.relative_to(out)),
            "artifact_sha256": sha256(elf),
        })
        return _with_board_reference_warnings(
            record, text, remote_dir_marker_path,
        )
    marker = re.search(
        r"NATIVE_RV64_REFERENCE .*remote_exit_code=(-?\d+)(?: primary_timeout=(true|false))?",
        text,
    )
    identity = None
    for line in text.splitlines():
        if line.startswith("RV_NATIVE_REFERENCE_IDENTITY="):
            try:
                identity = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                identity = None
    if not marker:
        record = _board_reference_gap(
            stratum, "k1-board-reference-contract-missing", process.returncode,
        )
        return _with_board_reference_warnings(
            record, text, remote_dir_marker_path,
        )
    local_sha = sha256(elf)
    identity = {
        **_board_reference_identity("recorded"),
        **(identity or {}),
    }
    remote_code = int(marker.group(1))
    primary_timed_out = (
        marker.group(2) == "true" if marker.group(2) is not None else remote_code == 124
    )
    status = "timeout" if primary_timed_out else "passed" if remote_code == 0 else "failed"
    terminal = "timeout" if primary_timed_out else "complete_observation" if remote_code == 0 else (
        "crash" if remote_code < 0 else "nonzero-exit")
    pcs = []
    trace_status = "unavailable"
    for line in text.splitlines():
        if line.startswith("RV_EXECUTED_PCS="):
            try:
                pcs = [int(item, 0) for item in json.loads(line.split("=", 1)[1])]
            except (TypeError, ValueError, json.JSONDecodeError):
                pcs = []
        elif line.startswith("NATIVE_RV64_TRACE_STATUS="):
            trace_status = line.split("=", 1)[1]
    trace_complete = trace_status.startswith("observed-runner-trace") and bool(pcs)
    frame = None
    frame_bytes = _observation_frame(stdout)
    if frame_bytes is not None:
        try:
            checkpoint = struct.unpack_from("<Q", frame_bytes, 8)[0]
            memory_size = struct.unpack_from("<Q", frame_bytes, 16 + 32 * 8)[0]
            memory_end = OBSERVATION_HEADER_SIZE + memory_size
            gpr = tuple(struct.unpack_from("<Q", frame_bytes, 16 + index * 8)[0] for index in range(32))
            if checkpoint and not checkpoint & 1 and gpr[0] == 0 and len(frame_bytes) == memory_end:
                memory = frame_bytes[OBSERVATION_HEADER_SIZE:memory_end]
                frame = {"checkpoint_pc": checkpoint, "gpr_sha256": hashlib.sha256(
                    b"".join(value.to_bytes(8, "little") for value in gpr)).hexdigest(),
                    "memory_size": memory_size, "memory_sha256": hashlib.sha256(memory).hexdigest()}
        except (IndexError, struct.error, ValueError):
            frame = None
    elif remote_code == 0:
        return _board_reference_gap(stratum, "k1-board-observation-frame-missing", remote_code)
    if frame is None and remote_code == 0:
        return _board_reference_gap(stratum, "k1-board-observation-frame-invalid", remote_code)
    frame_verified = frame is not None
    raw_sha = hashlib.sha256(stdout).hexdigest()
    program_sha = hashlib.sha256(frame_bytes).hexdigest() if frame_verified else raw_sha
    observation = {
        "schema_version": "rq1-terminal-observation-v1", "terminal": terminal,
        "returncode": remote_code, "stdout_sha256": program_sha,
        "raw_stdout_sha256": raw_sha, "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "observation_channel": "rvobs1-frame" if frame_verified else "terminal",
        "observation_frame_sha256": program_sha if frame_verified else None,
        "observation_frame_verified": frame_verified,
        "trace_status": trace_status, "trace_records": len(pcs), "trace_pcs": pcs,
        "terminal_observed": True,
        **(frame or {}),
    }
    return {
        "id": "R-K1-BOARD", "engine": "k1-board", "backend": "native-rv64",
        "target": "T-K1-BOARD", "stratum": stratum, "status": status,
        "observer_complete": frame_verified, "returncode": remote_code, "terminal": terminal,
        **({"failure_class": "k1-timeout"} if primary_timed_out else {}),
        "program_stdout_sha256": program_sha, "stdout_sha256": program_sha,
        "observation_channel": "rvobs1-frame" if frame_verified else "terminal",
        "observation_frame_sha256": program_sha if frame_verified else None,
        "observation_frame_verified": frame_verified,
        "terminal_observed": True,
        "raw_stdout_sha256": raw_sha, "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "trace_status": trace_status, "trace_records": len(pcs), "trace_pcs": pcs,
        "identity": identity, "reference_identity": identity, "observation": observation,
        **({"operational_warnings": operational_warnings} if operational_warnings else {}),
        "stdout_path": str(stdout_path.relative_to(out)), "stderr_path": str(stderr_path.relative_to(out)),
        "artifact_sha256": local_sha,
        "trusted_reference_signature": behavior_signature({
            "trace_pcs": pcs, "outcome": "exit", "exit_code": remote_code,
        }) if frame_verified and trace_complete else None,
    }


def _run_manifest_gate(
    out: Path, config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    path = out / "execution-manifest.json"
    try:
        data = load_json(path)
    except (OSError, TypeError, ValueError):
        return {"status": "gap", "path": str(path)}, ["manifest:invalid"]
    ordinary_network = (config or {}).get("execution_policy", {}).get("network", "none")
    resource_limits = (config or {}).get("execution_policy", {}).get("resource_limits")
    framework_face = data.get("experiment_face") == "framework" \
        or data.get("action") == "framework-run"
    configured_methods = (config or {}).get(
        "framework_methods" if framework_face else "methods", []
    )
    selected_method = data.get("method_filter")
    if selected_method:
        selected = [method for method in configured_methods
                    if isinstance(method, dict) and method.get("id") == selected_method]
        expected_network = (selected[0].get("network") or ordinary_network) if selected else ordinary_network
    else:
        networks = {
            method.get("network") or ordinary_network
            for method in configured_methods if isinstance(method, dict)
        }
        expected_network = next(iter(networks)) if len(networks) == 1 else data.get("network", ordinary_network)
    configured_networks = {
        method.get("network") or ordinary_network
        for method in configured_methods if isinstance(method, dict)
    }
    expected_network_shape = "selected" if selected_method else (
        "mixed" if len(configured_networks) > 1 else "uniform"
    )
    manifest_resources = data.get("resources")
    profile = data.get("resource_profile")
    profile_limits = profile.get("limits") if isinstance(profile, dict) else None
    expected_resources = profile_limits if isinstance(profile_limits, dict) else resource_limits
    resources_ok = not isinstance(expected_resources, dict) or (
        isinstance(manifest_resources, dict)
        and manifest_resources.get("enforcement") == "docker-cgroup"
        and all(
            manifest_resources.get(field) == expected_resources.get(field)
            for field in ("cpus", "memory", "memory_swap", "pids")
        )
    )
    manifest_methods = data.get("methods")
    method_isolation = data.get("method_isolation")
    method_isolation_ok = (
        data.get("action") == "doctor" and not isinstance(method_isolation, dict)
    )
    if isinstance(method_isolation, dict):
        method_isolation_ok = (
            method_isolation.get("mode") == (
                "one-method-per-container" if selected_method else "diagnostic"
            )
            and method_isolation.get("method") == (selected_method or None)
            and method_isolation.get("method_count") == (
                len(manifest_methods) if isinstance(manifest_methods, list) else None
            )
            and (not selected_method or method_isolation.get("method_count") == 1)
            and (
                not selected_method
                or (
                    isinstance(manifest_methods, list)
                    and len(manifest_methods) == 1
                    and isinstance(manifest_methods[0], dict)
                    and manifest_methods[0].get("id") == selected_method
                )
            )
        )
    configured_targets = {
        str(target.get("id")): target
        for target in (config or {}).get("targets", [])
        if isinstance(target, dict) and target.get("id")
    }
    expected_target_ids = set(configured_targets)
    if framework_face and data.get("feedback_target"):
        feedback_target = str(data.get("feedback_target"))
        expected_target_ids = (
            set(configured_targets)
            if feedback_target == "all" else {feedback_target}
        )
    manifest_target_rows = data.get("targets")
    target_identity_fields = (
        "kind", "execution_model", "translation_mode", "commit", "binary",
        "binary_sha256", "coverage_binary", "coverage_binary_sha256",
        "identity", "identity_digest", "simulator_coverage",
    )
    target_coverage_identity_ok = (
        isinstance(manifest_target_rows, list)
        and len({str(row.get("id")) for row in manifest_target_rows if isinstance(row, dict)})
        == len(manifest_target_rows)
        and {str(row.get("id")) for row in manifest_target_rows if isinstance(row, dict)}
        == expected_target_ids
        and all(
            isinstance(row, dict)
            and str(row.get("id")) in configured_targets
            and all(row.get(field) == configured_targets[str(row.get("id"))].get(field)
                    for field in target_identity_fields)
            for row in manifest_target_rows
        )
    )
    checks = {
        "schema": data.get("schema_version") == "rq1-comparison-run-manifest-v1",
        "run_id": data.get("run_id") == out.name,
        "action": data.get("action") in {
            "run", "doctor", "generate", "execute-existing-queues", "framework-run",
        },
        # Coverage is an optional evidence side-channel.  The manifest still
        # records the actual switch so a hand-written/legacy manifest cannot
        # silently omit the execution mode, but it must not block Target
        # observation itself.
        "coverage_policy": data.get("action") not in {
            "run", "execute-existing-queues", "framework-run",
        } or type(data.get("coverage_enabled")) is bool,
        "method_filter": not selected_method or any(
            isinstance(method, dict) and method.get("id") == selected_method
            for method in configured_methods
        ),
        "method_isolation": method_isolation_ok,
        "duration": type(data.get("duration_seconds")) is int and data["duration_seconds"] > 0,
        "source_commit": bool(re.fullmatch(r"[0-9a-f]{40}", str(data.get("source_commit", "")))),
        # Framework runs keep dependency identity as provenance.  The minimal
        # prototype must still generate cases and Target evidence when a pin
        # is unavailable; external replay keeps the stricter check.
        "dependency_identity": data.get("dependency_identity_ok") is True
        or data.get("action") in {"doctor", "framework-run"},
        "image": bool(data.get("image_id")),
        "config_snapshot": data.get("config_snapshot") == "config.snapshot.json"
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(data.get("config_sha256", "")))),
        "network": data.get("network") == expected_network,
        "network_shape": data.get("network_shape") == expected_network_shape,
        "resources": resources_ok,
        "code_ro": any(item.get("container") == "/opt/rq1/comparison" and item.get("mode") == "ro"
                        for item in data.get("mounts", []) if isinstance(item, dict)),
        "generation_source": data.get("action") != "execute-existing-queues" or (
            bool(data.get("generation_run_id")) and data.get("generation_root") == "/opt/generation"
            and any(item.get("container") == "/opt/generation" and item.get("mode") == "ro"
                    for item in data.get("mounts", []) if isinstance(item, dict))
        ),
        "run_rw": any(item.get("container", "").startswith("/opt/runs/") and item.get("mode") == "rw"
                       for item in data.get("mounts", []) if isinstance(item, dict)),
        "target_identity_snapshot": isinstance(data.get("targets"), list)
        and bool(data.get("targets")),
        # The manifest must pin both the executable Target and the optional
        # coverage executable/library declaration back to the frozen config.
        "target_coverage_identity": target_coverage_identity_ok,
    }
    blockers = [f"manifest:{name}" for name, ok in checks.items() if not ok]
    return {"status": "verified" if not blockers else "gap", "path": str(path), "checks": checks}, blockers


def _config_profile_gaps(config: dict[str, Any]) -> list[str]:
    """检查生成 profile 的字段类型和与适配器实现有关的约束。"""
    gaps: list[str] = []

    def integer(value: object, path: str, minimum: int = 1) -> int | None:
        if type(value) is not int or value < minimum:
            gaps.append(path)
            return None
        return value

    def text(value: object, path: str) -> str | None:
        if not isinstance(value, str) or not value.strip():
            gaps.append(path)
            return None
        return value

    methods = config.get("methods")
    if not isinstance(methods, list):
        return ["methods:not-list"]
    declared = [*methods]
    framework_methods = config.get("framework_methods")
    if isinstance(framework_methods, list):
        declared.extend(framework_methods)
    for method in declared:
        if not isinstance(method, dict):
            gaps.append("method:not-object")
            continue
        method_id = str(method.get("id") or "<missing>")
        profile = method.get("generator_profile")
        if not isinstance(profile, dict):
            gaps.append(f"{method_id}:generator_profile")
            continue
        # `lane` is provenance for the generator profile.  It is not a
        # shared ISA/ABI allowlist: each tool may emit the profile it supports
        # and a target reports unsupported instructions at execution time.
        lane = str(method.get("lane") or "").split("/", 1)
        lane_isa, lane_mabi = lane if len(lane) == 2 else (None, None)
        profile_isa = profile.get("isa")
        profile_mabi = profile.get("mabi")
        if profile_isa is not None:
            text(profile_isa, f"{method_id}:isa")
        if profile_mabi is not None:
            text(profile_mabi, f"{method_id}:mabi")

        if method_id == "B-RVDV":
            # These are generator-owned knobs.  Validate their shape so a
            # malformed profile fails early, but never turn one upstream
            # value into an RQ1-wide capability gate.  The selected profile
            # is passed through verbatim and unsupported instructions are a
            # target runtime result.
            target = text(profile.get("target"), f"{method_id}:target")
            isa = text(profile.get("isa"), f"{method_id}:isa")
            mabi = text(profile.get("mabi"), f"{method_id}:mabi")
            boot_mode = text(profile.get("boot_mode"), f"{method_id}:boot_mode")
            if target == "rv64imc" and boot_mode != "m":
                gaps.append(f"{method_id}:boot_mode-target")
            iterations = integer(profile.get("iterations"), f"{method_id}:iterations")
            batch_size = integer(profile.get("batch_size"), f"{method_id}:batch_size")
            if iterations is not None and batch_size is not None and batch_size > iterations:
                gaps.append(f"{method_id}:batch_size-iterations")
            tests = profile.get("tests")
            if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
                gaps.append(f"{method_id}:tests")
            else:
                integer(tests[0].get("instr_cnt"), f"{method_id}:tests[0].instr_cnt")
                integer(tests[0].get("sub_programs"), f"{method_id}:tests[0].sub_programs")
            if not isinstance(profile.get("incremental_batches"), bool):
                gaps.append(f"{method_id}:incremental_batches")
            for field in ("no_csr_instr", "bare_program_mode"):
                if type(profile.get(field)) is not int or profile[field] not in (0, 1):
                    gaps.append(f"{method_id}:{field}")
        elif method_id == "B-TORTURE":
            for field in ("nseqs", "memsize", "loop_size"):
                integer(profile.get(field), f"{method_id}:{field}")
            mix = profile.get("mix")
            if not isinstance(mix, dict) or not mix or any(
                type(value) is not int or not 0 <= value <= 100
                for value in mix.values()
            ) or isinstance(mix, dict) and sum(mix.values()) != 100:
                gaps.append(f"{method_id}:mix")
        elif method_id == "B-CSMITH":
            if not isinstance(profile.get("args"), list):
                gaps.append(f"{method_id}:args")
            integer(profile.get("opt_level"), f"{method_id}:opt_level", 0)
        elif method_id == "B-GEMI":
            if not isinstance(profile.get("args"), list):
                gaps.append(f"{method_id}:args")
            if not isinstance(profile.get("profiles"), list) or not profile["profiles"]:
                gaps.append(f"{method_id}:profiles")
            if "attempt_timeout_seconds" in profile:
                integer(
                    profile.get("attempt_timeout_seconds"),
                    f"{method_id}:attempt_timeout_seconds", 1,
                )
            probability = profile.get("probability_live_code_mutate")
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                gaps.append(f"{method_id}:probability_live_code_mutate")
            for field in ("engine", "decompiler", "main"):
                text(profile.get(field), f"{method_id}:{field}")
        elif method_id in {"Ours-RVGEN-Direct", "Ours-Program-Full"}:
            # 10 is the default profile value.  A run may select a smaller
            # positive per-case budget (for example 5 or 3); the frozen
            # config snapshot records the value used by that container.
            integer(profile.get("steps"), f"{method_id}:steps")
            if profile.get("mcmc") is not True:
                gaps.append(f"{method_id}:mcmc-required")
            if not isinstance(profile.get("small_model"), str) or not profile["small_model"].strip():
                gaps.append(f"{method_id}:small-model-required")
            integer(
                profile.get(
                    "parallel_lanes", FRAMEWORK_CANONICAL_PARALLEL_LANES,
                ),
                f"{method_id}:parallel_lanes",
            )
            # Lane count is a resource knob, not a method-semantics gate.
            # Program-Full may use fewer producer lanes than RVGEN when the
            # single-container memory budget cannot hold both campaigns at
            # the same concurrency.
            if "beta" in profile and (
                type(profile["beta"]) not in (int, float)
                or not math.isfinite(profile["beta"])
                or profile["beta"] <= 0
            ):
                gaps.append(f"{method_id}:beta")
        elif method_id == "Fuzz4All":
            for field in ("batch_size", "max_length"):
                integer(profile.get(field), f"{method_id}:{field}")
            remote_predict = integer(
                profile.get("remote_num_predict"),
                f"{method_id}:remote_num_predict",
            )
            remote_context = integer(
                profile.get("remote_num_ctx"),
                f"{method_id}:remote_num_ctx",
            )
            if (
                remote_predict is not None and remote_context is not None
                and remote_context <= remote_predict
            ):
                gaps.append(f"{method_id}:remote_num_ctx-too-small")
            temperature = profile.get("temperature")
            if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
                gaps.append(f"{method_id}:temperature")
            strategy = profile.get("prompt_strategy")
            if type(strategy) is not int or strategy not in (-1, 0, 1, 2, 3):
                gaps.append(f"{method_id}:prompt_strategy")
            if ("use_hand_written_prompt" in profile
                    and profile.get("use_hand_written_prompt") is not True):
                gaps.append(f"{method_id}:use_hand_written_prompt")
            if ("no_input_prompt" in profile
                    and profile.get("no_input_prompt") is not False):
                gaps.append(f"{method_id}:no_input_prompt")
            text(method.get("model"), f"{method_id}:model")
    return gaps


def validate_config(config: dict[str, Any]) -> list[str]:
    """在生成/执行前一次性检查 canonical config 的结构和跨字段约束。"""
    if not isinstance(config, dict):
        return ["config"]
    gaps: list[str] = []

    def add(path: str, condition: bool) -> None:
        if not condition and path not in gaps:
            gaps.append(path)

    add("schema_version", config.get("schema_version") == "rq1-comparison-isolated-v1")
    rv_contract = config.get("rv_instruction_coverage")
    if rv_contract is not None and rv_instruction_metrics_enabled(config):
        add(
            "rv-instruction-coverage:schema",
            isinstance(rv_contract, dict)
            and rv_contract.get("schema_version")
            == "rq1-rv-instruction-coverage-contract-v1",
        )
        add(
            "rv-instruction-coverage:profile",
            isinstance(rv_contract, dict)
            and rv_contract.get("isa_profile") == EXPERIMENT_COVERAGE_PROFILE
            and rv_contract.get("privilege_mode") == EXPERIMENT_PRIVILEGE_MODE
            and rv_contract.get("xlen") == 64,
        )
        try:
            rv_registry = rv_registry_for_profile(
                EXPERIMENT_COVERAGE_PROFILE, EXPERIMENT_PRIVILEGE_MODE,
            )
            expected_form_count = len(rv_registry["form_units"])
            expected_opcode_count = len(rv_registry["opcode_units"])
        except (TypeError, ValueError):
            expected_form_count = expected_opcode_count = -1
        add(
            "rv-instruction-coverage:denominator",
            isinstance(rv_contract, dict)
            and rv_contract.get("form_denominator") == expected_form_count
            and rv_contract.get("opcode_denominator") == expected_opcode_count,
        )
    opcode_contract = config.get("rv_opcode_catalog_coverage")
    add(
        "rv-opcode-catalog-coverage:schema",
        isinstance(opcode_contract, dict)
        and opcode_contract.get("schema_version")
        == "rq1-rv-opcode-catalog-coverage-contract-v1"
        and opcode_contract.get("enabled") is True,
    )
    add(
        "rv-opcode-catalog-coverage:registry",
        isinstance(opcode_contract, dict)
        and opcode_contract.get("catalog") == OPCODE_CATALOG_ID
        and opcode_contract.get("key_schema") == OPCODE_KEY_SCHEMA
        and opcode_contract.get("registry_sha256") == OPCODE_CATALOG_REGISTRY_SHA256
        and opcode_contract.get("form_count") == OPCODE_CATALOG_FORM_COUNT
        and opcode_contract.get("opcode_denominator") == OPCODE_CATALOG_DENOMINATOR
        and opcode_contract.get("aggregation")
        == "method-generated-and-method-target-unique-set-union"
        and opcode_contract.get("partition_denominators") == {
            partition: len(units)
            for partition, units in OPCODE_CATALOG_PARTITION_KEYS.items()
        },
    )
    methods = config.get("methods")
    targets = config.get("targets")
    add("methods", isinstance(methods, list) and bool(methods))
    add("targets", isinstance(targets, list) and bool(targets))
    method_shape = (isinstance(methods, list)
                    and [(item.get("id"), item.get("route")) for item in methods
                         if isinstance(item, dict)] == list(ACTIVE_METHODS))
    target_ids = [item.get("id") for item in targets if isinstance(item, dict)] \
        if isinstance(targets, list) else []
    add("method-shape", method_shape)
    add(
        "target-shape",
        isinstance(targets, list)
        and len(targets) == len(TARGETS)
        and all(isinstance(item, dict) for item in targets)
        and target_ids == list(TARGETS),
    )
    for target in targets if isinstance(targets, list) else ():
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("id") or "<missing>")
        expected_kind = TARGET_KINDS.get(target_id)
        expected_model = TARGET_MODELS.get(target_id)
        expected_translation = "interpreter" if target_id == "T-RVVM" else None
        add(f"target:{target_id}:kind", target.get("kind") == expected_kind)
        add(
            f"target:{target_id}:execution-model",
            target.get("execution_model") == expected_model,
        )
        add(
            f"target:{target_id}:translation-mode",
            target.get("translation_mode") == expected_translation,
        )
        if "timeout_seconds" in target:
            add(
                f"target:{target_id}:timeout",
                type(target.get("timeout_seconds")) is int
                and target.get("timeout_seconds") > 0,
            )
        add(
            f"target:{target_id}:binary",
            isinstance(target.get("binary"), str) and bool(target["binary"].strip()),
        )
        add(
            f"target:{target_id}:identity",
            isinstance(target.get("identity"), str) and bool(target["identity"].strip())
            and isinstance(target.get("identity_digest"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", target["identity_digest"]) is not None,
        )
        add(
            f"target:{target_id}:coverage-binary",
            isinstance(target.get("coverage_binary"), str)
            and bool(target["coverage_binary"].strip()),
        )
        add(
            f"target:{target_id}:coverage-binary-sha256",
            isinstance(target.get("coverage_binary_sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", target["coverage_binary_sha256"]) is not None,
        )
        source_coverage = target.get("source_coverage")
        if isinstance(source_coverage, dict) and source_coverage.get("source_commit"):
            add(
                f"target:{target_id}:source-commit-match",
                source_coverage.get("source_commit") == target.get("commit"),
            )
    add("lane", isinstance(config.get("lane"), dict))
    isa_policy = config.get("isa_policy")
    add(
        "isa_policy",
        isinstance(isa_policy, dict)
        and isa_policy.get("mode") in {"open", "tool-profile"},
    )
    add("dependencies", isinstance(config.get("dependencies"), dict))
    policy = config.get("execution_policy")
    add("execution_policy", isinstance(policy, dict))
    if isinstance(policy, dict):
        add("execution_policy:coverage_enabled", policy.get("coverage_enabled") is True)
        ordinary_network = policy.get("network", "none")
        add("execution_policy:network", ordinary_network in {"none", "bridge"})
        resource_limits = policy.get("resource_limits")
        add("execution_policy:resource_limits", isinstance(resource_limits, dict))
        if isinstance(resource_limits, dict):
            cpus = resource_limits.get("cpus")
            add(
                "execution_policy:resource_limits:cpus",
                type(cpus) in (int, float) and math.isfinite(cpus) and cpus > 0,
            )
            for field in ("memory", "memory_swap"):
                value = resource_limits.get(field)
                add(
                    f"execution_policy:resource_limits:{field}",
                    isinstance(value, str) and bool(value.strip()),
                )
            pids = resource_limits.get("pids")
            add(
                "execution_policy:resource_limits:pids",
                type(pids) is int and pids > 0,
            )
        if isinstance(methods, list):
            method_networks = {
                str(method.get("id")): method.get("network") or ordinary_network
                for method in methods if isinstance(method, dict)
            }
            add("execution_policy:method-network", all(
                value == ordinary_network for value in method_networks.values()
            ))

    limits = config.get("limits")
    if not isinstance(limits, dict):
        gaps.append("limits")
    else:
        for field in ("target_timeout_seconds",):
            value = limits.get(field)
            add(f"limits:{field}", type(value) is int and value > 0)
    profile_gaps = _config_profile_gaps(config)
    gaps.extend(f"profile:{gap}" for gap in profile_gaps if f"profile:{gap}" not in gaps)
    framework_methods = config.get("framework_methods")
    add("framework_methods", isinstance(framework_methods, list) and bool(framework_methods))
    framework_method_shape = (
        isinstance(framework_methods, list)
        and [(item.get("id"), item.get("route")) for item in framework_methods
             if isinstance(item, dict)] == list(FRAMEWORK_METHODS)
    )
    add("framework-method-shape", framework_method_shape)
    declared = list(methods if isinstance(methods, list) else [])
    if isinstance(framework_methods, list):
        declared += [item for item in framework_methods if isinstance(item, dict)]
    if declared:
        method_ids = {str(item.get("id")) for item in declared if isinstance(item, dict)}
        dependencies = config.get("dependencies", {})
        # Applicability is retained as descriptive metadata for old reports.
        # The open-capability queue fans every generated case out to every
        # declared target, so a partial applicability table cannot block a
        # method before execution.
        applicability = config.get("applicability")
        for method in declared:
            if not isinstance(method, dict):
                continue
            method_id = str(method.get("id") or "<missing>")
            required = method.get("requires")
            add(f"{method_id}:requires", isinstance(required, list))
            if isinstance(required, list) and isinstance(dependencies, dict):
                for name in required:
                    add(f"{method_id}:dependency:{name}", name in dependencies)
            route = method.get("route")
            lane = method.get("lane")
            add(f"{method_id}:route", route in {"program", "single"})
            if lane not in (None, ""):
                add(f"{method_id}:lane", isinstance(lane, str) and "/" in lane)
            if method_id in {item[0] for item in FRAMEWORK_METHODS}:
                feedback_targets = method.get("feedback_targets")
                add(
                    f"{method_id}:feedback_targets",
                    isinstance(feedback_targets, list) and bool(feedback_targets),
                )
                if isinstance(feedback_targets, list):
                    for target_id in feedback_targets:
                        add(
                            f"{method_id}:unknown-feedback-target:{target_id}",
                            target_id in target_ids,
                        )
                        add(
                            f"{method_id}:unsupported-feedback-target:{target_id}",
                            target_id in FRAMEWORK_TARGET_BACKENDS,
                        )
            if "network" in method:
                add(f"{method_id}:network", method.get("network") in {"none", "bridge"})
            timeout = method.get("timeout_seconds")
            if timeout is not None:
                add(f"{method_id}:timeout_seconds", type(timeout) is int and timeout > 0)
            # Applicability is advisory metadata in the open-capability run.
            # Do not turn a partial declaration into a generation/target gate.
    return gaps


def doctor(
    config: dict[str, Any], *, provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config if isinstance(config, dict) else {}
    methods = config.get("methods") if isinstance(config.get("methods"), list) else []
    framework_methods = config.get("framework_methods") \
        if isinstance(config.get("framework_methods"), list) else []
    targets = config.get("targets") if isinstance(config.get("targets"), list) else []
    method_shape = [(item.get("id"), item.get("route")) for item in methods if isinstance(item, dict)] == list(ACTIVE_METHODS)
    framework_method_shape = [
        (item.get("id"), item.get("route"))
        for item in framework_methods if isinstance(item, dict)
    ] == list(FRAMEWORK_METHODS)
    config_gaps = validate_config(config)
    missing_dependencies = [name for name in REQUIRED_DEPENDENCY_PINS if not _dependency_present(config, name)]
    missing_targets = [
        item.get("id") if isinstance(item, dict) else "<invalid>"
        for item in targets
        if not isinstance(item, dict)
        or not dep(item.get("binary"))
        or not dep(item.get("binary")).exists()
    ]
    profile_gaps = _config_profile_gaps(config)
    target_identity = {}
    for target in targets:
        if not isinstance(target, dict):
            target_identity["<invalid>"] = {"status": "invalid"}
            continue
        target_id = str(target.get("id") or "<missing>")
        identity = _identity(dep(target.get("identity")), dep(target.get("binary")), target)
        target_identity[target_id] = {
            "status": identity.get("status"),
            "reason_code": identity.get("reason_code"),
            "binary_match": identity.get("binary_match"),
            "identity_match": identity.get("identity_match"),
        }
    identity_gaps = [
        f"target:{target_id}:identity"
        for target_id, identity in target_identity.items()
        if identity.get("status") != "verified"
    ]
    coverage_asset_gaps = []
    if simulator_coverage_enabled():
        for target in targets:
            if not isinstance(target, dict):
                coverage_asset_gaps.append("target:<invalid>:coverage-config")
                continue
            target_id = str(target.get("id") or "<missing>")
            if not dep(target.get("coverage_binary")) or not dep(target.get("coverage_binary")).is_file():
                coverage_asset_gaps.append(f"target:{target_id}:coverage-binary")
            coverage_spec = target.get("simulator_coverage") if isinstance(
                target.get("simulator_coverage"), dict
            ) else {}
            for field in ("plugin", "library"):
                value = coverage_spec.get(field)
                if value and (not dep(value) or not dep(value).is_file()):
                    coverage_asset_gaps.append(f"target:{target_id}:coverage-{field}")
            source_spec = target.get("source_coverage") if isinstance(
                target.get("source_coverage"), dict
            ) else {}
            if not source_spec.get("profile"):
                coverage_asset_gaps.append(f"target:{target_id}:source-coverage-profile")
            for field in ("source_root", "build_root"):
                value = source_spec.get(field)
                if value and (not dep(value) or not dep(value).is_dir()):
                    coverage_asset_gaps.append(f"target:{target_id}:source-coverage-{field}")
            collector = str(source_spec.get("collector", "gcov"))
            tool_fields = {
                "dotnet": ("launcher", "reportgenerator"),
                "llvm": ("profdata_tool", "cov_tool"),
            }.get(collector, ())
            tool_defaults = {
                "launcher": "dotnet-coverage",
                "reportgenerator": "reportgenerator",
                "profdata_tool": "llvm-profdata",
                "cov_tool": "llvm-cov",
            }
            for field in tool_fields:
                value = source_spec.get(field)
                available = dep(value).is_file() if value and dep(value) else bool(
                    shutil.which(tool_defaults[field])
                )
                if not available:
                    coverage_asset_gaps.append(f"target:{target_id}:source-coverage-{field}")
    blockers = list(config_gaps)
    blockers += [f"dependency:{name}" for name in missing_dependencies]
    blockers += [f"target:{name}:binary" for name in missing_targets]
    blockers += identity_gaps
    blockers += coverage_asset_gaps
    return {
        "schema_version": "rq1-comparison-doctor-v1",
        "status": "ready" if not blockers else "gap",
        "provenance": provenance or _provenance(config),
        "method_shape_ok": method_shape,
        "framework_method_shape_ok": framework_method_shape,
        "config_valid": not config_gaps,
        "config_gaps": config_gaps,
        "profile_contract_ok": not profile_gaps,
        "profile_gaps": profile_gaps,
        "missing_dependencies": missing_dependencies,
        "missing_targets": missing_targets,
        "target_identity": target_identity,
        "coverage_asset_gaps": coverage_asset_gaps,
        "blockers": blockers,
    }


def elf_segments(path: Path) -> tuple[int, list[tuple[int, int, int, int]]]:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("expected little-endian ELF64")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize = struct.unpack_from("<H", data, 54)[0]
    phnum = struct.unpack_from("<H", data, 56)[0]
    segments = []
    for index in range(phnum):
        fields = struct.unpack_from("<IIQQQQQQ", data, phoff + index * phentsize)
        # PT_LOAD may be BSS-only (p_filesz == 0, p_memsz > 0).  Dropping it
        # changes the guest address space and can make a valid image fail only
        # in the in-process adapters.
        if fields[0] == 1 and (fields[5] or fields[6]):
            segments.append((fields[3], fields[2], fields[5], fields[6]))
    return struct.unpack_from("<Q", data, 24)[0], segments


def _format_command(values: list[str], mapping: dict[str, Path | None]) -> list[str]:
    return [value.format(**{key: str(path) if path else "" for key, path in mapping.items()})
            for value in values]


def _run_unicorn(binary: Path, elf: Path, timeout: int, coverage: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    if stop_requested():
        return {
            "status": "gap", "reason": stop_reason(), "reason_code": stop_reason(),
            "returncode": None, "observer_complete": False,
            "process_started": False, "process_executed": False,
            "elapsed_s": round(time.monotonic() - started, 6),
        }
    old_library_path = os.environ.get("LD_LIBRARY_PATH")
    old_unicorn_path = os.environ.get("LIBUNICORN_PATH")
    try:
        os.environ["LD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LIBUNICORN_PATH"] = str(binary.parent)
        ctypes.CDLL(str(binary))
        # The pinned Unicorn core is 2.1.x, while the host image may expose a
        # 2.0.x Python binding.  Import the binding shipped with the pinned
        # source tree so the core/binding API versions remain identical.
        binding_roots = [
            DEPS / "simulator-sources" / "unicorn" / "bindings" / "python",
            DEPS / "python",
        ]
        binding_root = next(
            (root for root in binding_roots if (root / "unicorn").is_dir()), None,
        )
        if binding_root is not None:
            binding_prefix = str(binding_root.resolve())
            loaded = sys.modules.get("unicorn")
            loaded_file = getattr(loaded, "__file__", "") if loaded else ""
            if loaded and not str(loaded_file).startswith(binding_prefix):
                for name in tuple(sys.modules):
                    if name == "unicorn" or name.startswith("unicorn."):
                        sys.modules.pop(name, None)
            if str(binding_root) not in sys.path:
                sys.path.insert(0, str(binding_root))
        from unicorn import (UC_ARCH_RISCV, UC_HOOK_BLOCK, UC_HOOK_CODE, UC_HOOK_INTR,
                             UC_HOOK_MEM_READ, UC_HOOK_MEM_READ_UNMAPPED,
                             UC_HOOK_MEM_WRITE, UC_HOOK_MEM_WRITE_UNMAPPED,
                             UC_MODE_RISCV64, Uc)  # type: ignore
        import unicorn.riscv_const as riscv_const
        from unicorn.riscv_const import UC_RISCV_REG_PC
    except (OSError, ImportError, AttributeError) as exc:
        return {"status": "gap", "reason": "unicorn-python-runtime-missing", "error": str(exc),
                "process_started": False, "process_executed": False,
                "elapsed_s": round(time.monotonic() - started, 6)}
    finally:
        if old_library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = old_library_path
        if old_unicorn_path is None:
            os.environ.pop("LIBUNICORN_PATH", None)
        else:
            os.environ["LIBUNICORN_PATH"] = old_unicorn_path
    try:
        entry, segments = elf_segments(elf)
        _, symbols = symbol_offsets_from_elf(
            elf, ("tohost", "mtvec_handler", "ecall_handler", "test_done",
                  "obs_buf", "result_buffer"),
            allow_before_start=True,
        )
        tohost = int(symbols["tohost"], 0) if "tohost" in symbols else None
        has_guest_trap = "mtvec_handler" in symbols
        test_done = int(symbols["test_done"], 0) if "test_done" in symbols else None
        has_rvdv_terminal = (
            has_guest_trap and test_done is not None and "ecall_handler" in symbols
        )
        mailbox_name = "obs_buf" if "obs_buf" in symbols else "result_buffer"
        mailbox = int(symbols[mailbox_name], 0) if mailbox_name in symbols else None
        library_path = str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = library_path
        os.environ["LIBUNICORN_PATH"] = str(binary.parent)
        try:
            uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)
        finally:
            if old_library_path is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = old_library_path
            if old_unicorn_path is None:
                os.environ.pop("LIBUNICORN_PATH", None)
            else:
                os.environ["LIBUNICORN_PATH"] = old_unicorn_path
        data = elf.read_bytes()
        for address, offset, size, memory_size in segments:
            base = address & ~(PAGE - 1)
            end = (address + memory_size + PAGE - 1) & ~(PAGE - 1)
            uc.mem_map(base, end - base)
            if size:
                uc.mem_write(address, data[offset:offset + size])
        uc.mem_map(0x90000000, PAGE * 16)
        stop = "running"
        tohost_value = 0
        timed_out = False
        external_stop_requested = False
        deadline = time.monotonic() + timeout
        trace_pcs: list[int] = []
        trace_seen: set[int] = set()
        block_pcs: list[int] = []
        block_seen: set[int] = set()
        trace_records_total = 0
        event_counts: dict[str, int] = {}
        def event(name: str) -> None:
            event_counts[name] = event_counts.get(name, 0) + 1
        def hook(_uc: Any, _address: int, _size: int, _user: Any) -> None:
            nonlocal stop, timed_out, trace_records_total, tohost_value
            if time.monotonic() >= deadline:
                timed_out = True
                stop = "timeout"
                _uc.emu_stop()
                return
            if coverage:
                trace_records_total += 1
                if _address not in trace_seen:
                    trace_seen.add(_address)
                    trace_pcs.append(_address)
                event("instruction-exec")
            raw = int.from_bytes(_uc.mem_read(_address, 4), "little")
            if has_rvdv_terminal and _address == test_done and mailbox is not None:
                # Unicorn's RISC-V backend reports ECALL through UC_HOOK_INTR but
                # does not enter the machine-mode trap handler.  At the DV
                # terminal label, synthesize the same RVOBS1 checkpoint that the
                # guest handler writes; this keeps this in-process adapter
                # aligned with the machine-mode target's observable contract.
                gpr_regs = tuple(
                    getattr(riscv_const, f"UC_RISCV_REG_X{index}")
                    for index in range(32)
                )
                frame = bytearray(OBSERVATION_HEADER_SIZE)
                frame[:8] = OBSERVATION_MAGIC
                struct.pack_into("<Q", frame, 8, _address)
                for index, register in enumerate(gpr_regs):
                    value = 0 if index == 0 else int(_uc.reg_read(register))
                    struct.pack_into("<Q", frame, 16 + index * 8, value)
                _uc.mem_write(mailbox, bytes(frame))
                if tohost is not None:
                    _uc.mem_write(tohost, (1).to_bytes(8, "little"))
                    tohost_value = 1
                stop = "tohost"
                _uc.emu_stop()
                return
            if raw == 0x00100073 or raw & 0xffff == 0x9002:
                stop = "ebreak"
                _uc.emu_stop()
        def block(_uc: Any, _address: int, _size: int, _user: Any) -> None:
            if coverage:
                if _address not in block_seen:
                    block_seen.add(_address)
                    block_pcs.append(_address)
                event("basic-block-exec")
        def memory(_uc: Any, access: int, _address: int, _size: int, _value: int, _user: Any) -> None:
            nonlocal stop, tohost_value
            if coverage:
                event("memory-read" if access == 16 else "memory-write")
            if tohost is not None and access != 16 and _address <= tohost < _address + _size:
                tohost_value = _value
                if tohost_value:
                    stop = "tohost"
                    _uc.emu_stop()

        def unmapped(_uc: Any, _access: int, address: int, size: int,
                     _value: int, _user: Any) -> bool:
            # RISC-V-DV bare tests use a sparse physical-memory model. Renode's
            # mapped bus returns zero for an unmapped data read, so provide the
            # same zero-backed page lazily in Unicorn. Do not apply this to
            # ordinary capsules, where an unmapped access is a guest fault.
            if not has_guest_trap:
                return False
            start = address & ~(PAGE - 1)
            end = (address + max(1, size) + PAGE - 1) & ~(PAGE - 1)
            try:
                _uc.mem_map(start, end - start)
            except Exception:
                return False
            if coverage:
                event("memory-unmapped-page")
            return True
        def interrupt(_uc: Any, _number: int, _user: Any) -> None:
            # With a guest trap vector, ECALL is an input to the guest's
            # handler.  Stopping at the interrupt hook skips the handler's
            # tohost write and turns a valid RISC-V-DV completion into a
            # false Unicorn failure.  The memory hook observes the eventual
            # terminal write; capsules without a trap vector still use their
            # own ebreak/tohost path.
            if coverage:
                event("interrupt")
        def stop_for_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            uc.emu_stop()

        def stop_for_external_signal() -> None:
            nonlocal external_stop_requested, stop
            external_stop_requested = True
            stop = stop_reason()
            uc.emu_stop()

        def watch_external_stop() -> None:
            while not stop_watcher_stop.wait(0.05):
                if stop_requested():
                    stop_for_external_signal()
                    return

        def observation_frame_state() -> dict[str, Any]:
            if mailbox is None:
                return {"observation_gap": "Unicorn observation mailbox symbol is missing"}
            try:
                header = bytes(uc.mem_read(mailbox, OBSERVATION_HEADER_SIZE))
                if not header.startswith(OBSERVATION_MAGIC):
                    return {"observation_gap": "Unicorn obs_buf did not expose RVOBS1"}
                memory_size = struct.unpack_from(
                    "<Q", header, 16 + 32 * 8,
                )[0]
                if memory_size > MAX_OBSERVATION_MEMORY_SIZE:
                    return {"observation_gap": "Unicorn RVOBS1 memory is too large"}
                frame = bytes(uc.mem_read(
                    mailbox, OBSERVATION_HEADER_SIZE + memory_size,
                ))
                if _finalized_observation_frame(frame) != frame:
                    return {"observation_gap": "Unicorn RVOBS1 frame is incomplete or unfinalized"}
            except Exception as exc:  # noqa: BLE001 - adapter gap is persisted
                return {"observation_gap": f"Unicorn obs_buf read failed: {type(exc).__name__}: {exc}"}
            return {
                "observer_complete": True,
                "observation_channel": "rvobs1-frame",
                "observation_frame_verified": True,
                "observation_frame_sha256": hashlib.sha256(frame).hexdigest(),
            }
        if coverage or tohost is None or has_rvdv_terminal:
            uc.hook_add(UC_HOOK_CODE, hook)
        if coverage:
            uc.hook_add(UC_HOOK_BLOCK, block)
            uc.hook_add(UC_HOOK_MEM_READ, memory)
        if coverage or has_guest_trap:
            uc.hook_add(UC_HOOK_INTR, interrupt)
        if has_guest_trap:
            uc.hook_add(
                UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED,
                unmapped,
            )
        uc.hook_add(UC_HOOK_MEM_WRITE, memory)
        timer = threading.Timer(timeout, stop_for_timeout)
        timer.daemon = True
        timer.start()
        stop_watcher_stop = threading.Event()
        stop_watcher = threading.Thread(
            target=watch_external_stop, name="rq1-unicorn-stop-watcher", daemon=True,
        )
        stop_watcher.start()
        emulation_started = False
        try:
            emulation_started = True
            uc.emu_start(entry, 0)
        except Exception as exc:
            fault = {}
            try:
                pc = int(uc.reg_read(UC_RISCV_REG_PC))
                fault = {"fault_pc": hex(pc),
                         "fault_instruction": hex(int.from_bytes(uc.mem_read(pc, 4), "little"))}
            except Exception:
                pass
            if external_stop_requested:
                result = {
                    "status": "gap", "reason": stop_reason(),
                    "reason_code": stop_reason(), "returncode": None,
                    "stop": stop, "error": str(exc), "observer_complete": False,
                    "terminal_observed": False, "termination": stop_reason(),
                    "process_started": emulation_started,
                    "process_executed": emulation_started,
                    **fault,
                }
            else:
                result = {"status": "failed", "reason": "unicorn-exception", "returncode": -1,
                          "stop": type(exc).__name__, "error": str(exc), "observer_complete": False,
                          "terminal_observed": True, "termination": "guest-exception",
                          "guest_exit_channel": "unicorn-exception", "guest_status": "failure",
                          **fault}
            result.update(observation_frame_state())
        else:
            if external_stop_requested:
                result = {"status": "gap", "reason": stop_reason(),
                          "reason_code": stop_reason(), "returncode": None,
                          "stop": stop, "observer_complete": False,
                          "terminal_observed": False, "termination": stop_reason()}
            elif timed_out:
                result = {"status": "timeout", "reason": "wall-timeout", "returncode": 124,
                          "stop": stop, "observer_complete": False}
                # The host budget can fire just after the guest finalized its
                # mailbox.  Preserve a validated frame as observation evidence;
                # the timeout remains the terminal outcome and is handled by
                # the comparison layer separately.
                result.update(observation_frame_state())
            elif stop in {"ebreak", "ecall", "tohost"}:
                passed = tohost_value == 1 if tohost is not None else stop in {"ebreak", "ecall"}
                result = {"status": "passed" if passed else "failed",
                          "returncode": 0 if passed else 1, "stop": stop,
                          "tohost_value": tohost_value,
                          # A tohost/ebreak stop is terminal evidence only;
                          # observation_frame_state() promotes it only when
                          # the guest-written mailbox frame validates.
                          "observer_complete": False,
                          "terminal_observed": True,
                          "termination": (
                              "guest-tohost" if stop == "tohost" else
                              "guest-ebreak" if stop == "ebreak" else "guest-ecall"
                          )}
            else:
                result = {"status": "failed", "reason": "guest-natural-stop",
                          "returncode": 1, "stop": "natural-stop", "observer_complete": False,
                          "terminal_observed": True, "termination": "guest-natural-stop",
                          "guest_exit_channel": "unicorn-natural-stop",
                          "guest_status": "failure"}
            result.update(observation_frame_state())
        finally:
            timer.cancel()
            stop_watcher_stop.set()
            stop_watcher.join(timeout=0.2)
        result.update({
            "process_started": emulation_started,
            "process_executed": emulation_started,
            "external_stop_requested": external_stop_requested,
        })
        if coverage:
            result.update({"trace_pcs": trace_pcs, "block_pcs": block_pcs,
                           "trace_records": len(trace_pcs),
                           "trace_records_total": trace_records_total,
                           "event_counts": event_counts})
        result["elapsed_s"] = round(time.monotonic() - started, 6)
        return result
    except Exception as exc:
        return {"status": "gap", "reason": "unicorn-elf-load", "returncode": None,
                "error": str(exc), "observer_complete": False,
                "process_started": False, "process_executed": False,
                "elapsed_s": round(time.monotonic() - started, 6)}

def _run_unicorn_isolated(binary: Path, elf: Path, timeout: float,
                           coverage: bool) -> dict[str, Any]:
    started = time.monotonic()
    timeout = max(0.001, float(timeout))
    if not coverage:
        return _run_unicorn(binary, elf, timeout, coverage)
    child_code = (
        "import json,sys;"
        "from pathlib import Path;"
        "from runner import _flush_gcov,_run_unicorn;"
        "binary=Path(sys.argv[1]);"
        "result=_run_unicorn(binary,Path(sys.argv[2]),float(sys.argv[3]),True);"
        "_flush_gcov(binary);"
        "print(json.dumps(result));"
        "sys.exit(0)"
    )
    env = {**os.environ, "PYTHONPATH": str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    child = None
    try:
        child = run_cmd(
            [sys.executable, "-c", child_code, str(binary), str(elf), str(timeout)],
            timeout=timeout,
            cwd=HERE, env=env,
        )
        if child.get("reason") == "external-signal":
            return {
                **child, "reason": "external-signal", "reason_code": "external-signal",
                "elapsed_s": round(time.monotonic() - started, 6),
            }
        if child.get("status") == "timeout":
            return {
                "status": "timeout", "reason": "wall-timeout", "returncode": 124,
                "process_started": child.get("process_started") is True,
                "process_executed": child.get("process_executed") is True,
                "stdout": child.get("stdout", ""), "stderr": child.get("stderr", ""),
                "elapsed_s": round(time.monotonic() - started, 6),
            }
        stdout = child.get("stdout", "") or ""
        stderr = child.get("stderr", "") or ""
        output_lines = stdout.splitlines()
        result_line = next((line.strip() for line in reversed(output_lines) if line.strip()), None)
        if result_line is None:
            child_stderr = stderr.strip()
            reason = (
                "unicorn-child-import-error"
                if "ImportError" in child_stderr or "ModuleNotFoundError" in child_stderr
                else "unicorn-child-no-result"
            )
            return {
                "status": "gap", "reason": reason,
                "error": child_stderr or "Unicorn child produced no JSON result",
                "returncode": child.get("returncode"),
                "process_started": child.get("process_started") is True,
                "process_executed": child.get("process_executed") is True,
                "stdout": stdout, "stderr": stderr,
                "child_stdout": stdout, "child_stderr": stderr,
                "elapsed_s": round(time.monotonic() - started, 6),
            }
        result = json.loads(result_line)
        if isinstance(result, dict):
            result.setdefault("returncode", child.get("returncode"))
            result.setdefault("process_started", child.get("process_started") is True)
            result.setdefault("process_executed", child.get("process_executed") is True)
            result["child_stdout"] = stdout
            result["child_stderr"] = stderr
            result.setdefault("elapsed_s", round(time.monotonic() - started, 6))
            return result
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": "wall-timeout", "returncode": 124,
                "process_started": True, "process_executed": True,
                "elapsed_s": round(time.monotonic() - started, 6)}
    except (OSError, TypeError, ValueError) as exc:
        # 子进程输出不是 JSON 时保留原始 stderr；不要把真正的导入/运行
        # 错误覆盖成 ``list index out of range``。
        child_stderr = child.get("stderr", "") if isinstance(child, dict) else ""
        reason = (
            "unicorn-child-import-error"
            if "ImportError" in child_stderr or "ModuleNotFoundError" in child_stderr
            else "unicorn-child-result-error"
        )
        return {"status": "gap", "reason": reason, "error": str(exc),
                "returncode": child.get("returncode") if isinstance(child, dict) else None,
                "process_started": child.get("process_started") is True if isinstance(child, dict) else False,
                "process_executed": child.get("process_executed") is True if isinstance(child, dict) else False,
                "stdout": "",
                "stderr": child_stderr,
                "child_stdout": "",
                "child_stderr": child_stderr,
                "elapsed_s": round(time.monotonic() - started, 6)}


def _rvvm_read_packet(sock: socket.socket, timeout: float) -> str:
    deadline = time.monotonic() + timeout

    def receive(size: int) -> bytes:
        while True:
            if stop_requested():
                raise RuntimeError(stop_reason())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RVVM RSP packet timeout")
            sock.settimeout(min(0.5, remaining))
            try:
                return sock.recv(size)
            except socket.timeout:
                continue

    while True:
        item = receive(1)
        if not item:
            raise ConnectionError("RVVM RSP connection closed")
        if item == b"$":
            break
    body = bytearray()
    while True:
        item = receive(1)
        if not item:
            raise ConnectionError("RVVM RSP packet truncated")
        if item == b"#":
            break
        body.extend(item)
    checksum_bytes = bytearray()
    while len(checksum_bytes) < 2:
        item = receive(2 - len(checksum_bytes))
        if not item:
            raise RuntimeError("RVVM RSP checksum truncated")
        checksum_bytes.extend(item)
    try:
        checksum = int(bytes(checksum_bytes).decode("ascii"), 16)
    except ValueError as exc:
        raise RuntimeError("RVVM RSP checksum is malformed") from exc
    if checksum != sum(body) & 0xff:
        sock.sendall(b"-")
        raise RuntimeError("RVVM RSP checksum mismatch")
    sock.sendall(b"+")
    return bytes(body).decode("ascii", "replace")


def _rvvm_request(sock: socket.socket, payload: str, timeout: float,
                  pending: list[str], stop_reply: bool = False) -> str:
    raw = payload.encode("ascii")
    sock.sendall(b"$" + raw + b"#" + f"{sum(raw) & 0xff:02x}".encode("ascii"))
    while True:
        reply = _rvvm_read_packet(sock, timeout)
        if reply.startswith("O") and reply != "OK":
            continue
        if not stop_reply and reply.startswith(("S", "T", "X")):
            pending.append(reply)
            continue
        return reply


def _rvvm_qrcmd(sock: socket.socket, command: str, timeout: float,
                pending: list[str]) -> str:
    return _rvvm_request(sock, "qRcmd," + command.encode("utf-8").hex(), timeout, pending)


def _rvvm_instruction_kind(code: bytes) -> str | None:
    """Classify the two guest instructions that end the bare probe."""
    if code[:4] == b"s\x00\x10\x00" or code[:2] == b"\x02\x90":
        return "ebreak"
    if code[:4] == b"s\x00\x00\x00":
        return "ecall"
    return None


def _rvvm_packet_pc(registers: str) -> int | None:
    """Decode PC from the 64-bit register packet emitted by stepcov RVVM.

    The wrapper follows the RSP x0..x31,pc order.  Keep this in one helper so
    the stop classifier and the trace probe cannot drift apart.
    """
    if len(registers) < 33 * 16:
        return None
    try:
        raw = bytes.fromhex(registers[:33 * 16])
    except ValueError:
        return None
    return int.from_bytes(raw[32 * 8:33 * 8], "little")


def _rvvm_stepcov_mode(coverage: bool, capability_reply: str | None) -> tuple[bool, str | None]:
    if not coverage:
        return False, None
    return (True, None) if capability_reply == "OK" else (
        False, "rvvm-single-step-unsupported",
    )


def _rvvm_stop_kind(sock: socket.socket, timeout: float,
                    pending: list[str], pc: int | None = None) -> str | None:
    if pc is None:
        registers = _rvvm_request(sock, "g", timeout, pending)
        pc = _rvvm_packet_pc(registers)
        if pc is None:
            return None
        start = max(0, pc - 64)
        try:
            code = bytes.fromhex(_rvvm_request(sock, f"m{start:x},128", timeout, pending))
        except (ValueError, RuntimeError):
            code = b""
        offset = pc - start
        kind = _rvvm_instruction_kind(code[offset:])
    else:
        try:
            code = bytes.fromhex(_rvvm_request(sock, f"m{pc:x},4", timeout, pending))
        except (ValueError, RuntimeError):
            code = b""
        offset = 0
        kind = _rvvm_instruction_kind(code)
    if kind is not None:
        return kind
    try:
        state = _rvvm_qrcmd(sock, "rvvm:state", timeout, pending)
        cause = next(
            value for key, sep, value in (item.partition("=") for item in state.split(";"))
            if sep and key == "csr.mcause"
        )
        value = int.from_bytes(bytes.fromhex(cause), "little") & ((1 << 63) - 1)
        if value:
            return "ecall" if value in {8, 9, 11} else "exception"
    except (StopIteration, ValueError, RuntimeError):
        pass
    return "exception" if code[offset:offset + 4] == b"\xff\xff\xff\xff" else None


def _rvvm_pc(sock: socket.socket, timeout: float, pending: list[str]) -> int | None:
    registers = _rvvm_request(sock, "g", timeout, pending)
    return _rvvm_packet_pc(registers)


def _rvvm_read_memory(
    sock: socket.socket, address: int, size: int, timeout: float, pending: list[str],
) -> bytes:
    if type(address) is not int or address < 0 or type(size) is not int or size < 0:
        raise ValueError("invalid RVVM memory read")
    data = bytearray()
    for offset in range(0, size, 64):
        length = min(64, size - offset)
        reply = _rvvm_request(sock, f"m{address + offset:x},{length:x}", timeout, pending)
        if reply.startswith("E"):
            raise RuntimeError(f"RVVM memory read failed: {reply}")
        try:
            chunk = bytes.fromhex(reply)
        except ValueError as exc:
            raise RuntimeError("RVVM memory read returned invalid hex") from exc
        if len(chunk) != length:
            raise RuntimeError("RVVM memory read returned a short packet")
        data.extend(chunk)
    return bytes(data)


def _run_rvvm_probe(binary: Path, elf: Path, timeout: int, isa: str = "rv64i",
                     coverage: bool = False,
                     translation_mode: str = "interpreter",
                     environment: dict[str, str] | None = None,
                     trace_path: Path | None = None,
                     trace_watch_pcs: set[int] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    if stop_requested():
        return {
            "status": "gap", "reason": stop_reason(), "reason_code": stop_reason(),
            "returncode": None, "observer_complete": False,
            "process_started": False, "process_executed": False,
            "elapsed_s": round(time.monotonic() - started, 6),
        }
    process = None
    process_group = None
    sock = None
    pending: list[str] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    if translation_mode not in {"interpreter", "jit"}:
        raise ValueError(f"unsupported RVVM translation mode: {translation_mode}")
    command_line = [str(binary), str(elf), "-isa", isa, "-harts", "1",
                    "-nogui", "-nonet", "-nosound", "-noisolation",
                    *(["-nojit"] if translation_mode == "interpreter" else []),
                    *( ["-rq1-stepcov"] if coverage else []),
                    *( ["-rq1-trace", str(trace_path)] if trace_path else []),
                    "-gdbstub", f"127.0.0.1:{port}"]
    result = None
    stage = "spawn"
    external_stop_requested = False
    trace_pcs: list[int] = []
    trace_seen: set[int] = set()
    trace_records_total = 0
    # 必须在 try 之外初始化：连接/暂停阶段抛错时下面的 finally 仍会读它。
    trace_truncated = False
    trace_truncation_reason = None
    cleanup_attempted = False
    cleanup_status = "not-started"
    cleanup_error = None
    cleanup_killed: set[int] = set()
    remaining_descendants: set[int] = set()
    _, symbols = symbol_offsets_from_elf(
        elf, ("tohost", "obs_buf", "result_buffer"), allow_before_start=True,
    )
    tohost = int(symbols["tohost"], 0) if "tohost" in symbols else None
    mailbox_name = "obs_buf" if "obs_buf" in symbols else "result_buffer"
    mailbox = int(symbols[mailbox_name], 0) if mailbox_name in symbols else None
    stop_flush = Path((environment or os.environ).get("RQ1_RVVM_STOP_FLUSH", ""))
    graceful_stop = coverage and stop_flush.is_file()
    coverage_flush_status = "process-not-started" if coverage else "coverage-disabled"
    coverage_flush_signal = None
    coverage_flush_returncode = None
    coverage_flush_forced_cleanup = False
    coverage_flush_marker_written = False
    coverage_flush_marker_error = None
    trace_guard_stop = threading.Event()
    trace_guard_pc: list[int] = []
    trace_guard_thread = None
    try:
        env = {**(environment or os.environ), "LD_LIBRARY_PATH": os.pathsep.join(
            filter(None, (str(binary.parent), os.environ.get("LD_LIBRARY_PATH"))))}
        gdbcompat = binary.with_name("gdbcompat.so")
        if gdbcompat.is_file():
            env["LD_PRELOAD"] = os.pathsep.join(filter(None, (str(gdbcompat), env.get("LD_PRELOAD"))))
        if graceful_stop:
            env["LD_PRELOAD"] = os.pathsep.join(filter(None, (str(stop_flush), env.get("LD_PRELOAD"))))
        process = subprocess.Popen(command_line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=os.name == "posix")
        process_group = os.getpgid(process.pid) if os.name == "posix" else None
        stage = "connect"
        deadline = started + timeout
        while True:
            if stop_requested():
                external_stop_requested = True
                raise RuntimeError(stop_reason())
            if process.poll() is not None:
                raise RuntimeError(f"RVVM exited before GDB connection: {process.returncode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RVVM GDB connection timeout")
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=min(0.5, remaining))
                break
            except OSError:
                continue
        def remaining() -> float:
            nonlocal external_stop_requested
            if stop_requested():
                external_stop_requested = True
                raise RuntimeError(stop_reason())
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("RVVM execution timeout")
            return value

        def attach_observation_frame() -> None:
            if result is None or result.get("status") not in {"passed", "failed"}:
                return
            if mailbox is None:
                result["observer_complete"] = False
                result["observation_gap"] = "RVVM observation mailbox symbol is missing"
                if _guest_terminal_evidence(result):
                    result["observation_channel"] = "terminal"
                return
            try:
                header = _rvvm_read_memory(
                    sock, mailbox, OBSERVATION_HEADER_SIZE,
                    remaining(), pending,
                )
                if not header.startswith(OBSERVATION_MAGIC):
                    raise RuntimeError("RVVM mailbox did not expose RVOBS1")
                checkpoint = struct.unpack_from("<Q", header, 8)[0]
                memory_size = struct.unpack_from("<Q", header, 16 + 32 * 8)[0]
                if checkpoint == 0 or checkpoint & 1 or memory_size > MAX_OBSERVATION_MEMORY_SIZE:
                    raise RuntimeError("RVVM mailbox RVOBS1 header is not finalized")
                frame = _rvvm_read_memory(
                    sock, mailbox, OBSERVATION_HEADER_SIZE + memory_size,
                    remaining(), pending,
                )
                if _finalized_observation_frame(frame) != frame:
                    raise RuntimeError("RVVM mailbox RVOBS1 frame is invalid or unfinalized")
            except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
                result["observer_complete"] = False
                result["observation_gap"] = str(exc)
                if _guest_terminal_evidence(result):
                    result["observation_channel"] = "terminal"
                return
            result.update({
                "observer_complete": True,
                "observation_channel": "rvobs1-frame",
                "observation_frame_verified": True,
                "observation_frame_sha256": hashlib.sha256(frame).hexdigest(),
            })

        stage = "pause"
        sock.sendall(b"\x03")
        # The RVVM debugger can connect before a busy vCPU reaches a stop
        # point. Bound this initial interrupt by the target's remaining budget
        # instead of the short poll interval used by ordinary RSP packets.
        if not _rvvm_read_packet(sock, remaining()).startswith(("S", "T", "X")):
            raise RuntimeError("RVVM pause failed")
        # bare capsule 可能在 GDB 客户端连上前就跑完并写掉 tohost。若这里
        # 无条件继续 c，RVVM 会从 ebreak 后的自旋循环重新运行，随后被误记为
        # S05/guest-exception 或 timeout；先读取终态，只有 tohost 仍为 0 才继续。
        if tohost is not None:
            try:
                head = _rvvm_request(sock, f"m{tohost:x},8", remaining(), pending)
                head_value = int.from_bytes(bytes.fromhex(head[:16]), "little")
            except (RuntimeError, ValueError):
                head_value = 0
            if head_value:
                result = {"status": "passed" if head_value == 1 else "failed",
                          "reason": "guest-tohost" if head_value == 1 else "guest-tohost-failure",
                          "returncode": 0 if head_value == 1 else 1,
                          "elapsed_s": round(time.monotonic() - started, 6),
                          "tohost_value": head_value,
                          "guest_status": "pass" if head_value == 1 else "failure",
                          "observer_complete": True, "terminal_observed": True,
                          "termination": "guest-tohost",
                          "command": command_line}
                # tohost 在 attach 前写入时，终态可用但 PC trace 不完整；这只
                # 影响 coverage，不影响 Target 终态。
                trace_truncated = False
                if coverage:
                    trace_truncation_reason = "rvvm-attached-after-guest-exit"
        bulk_trace = coverage and trace_path is not None
        stepcov_enabled = False
        if result is None:
            set_elf_entry_pc(
                elf, isa,
                lambda packet: _rvvm_request(sock, packet, remaining(), pending),
            )
            capability_reply = (
                _rvvm_request(sock, "qRQ1SingleStep", remaining(), pending)
                if coverage and not bulk_trace else None
            )
            stepcov_enabled, trace_capability_gap = _rvvm_stepcov_mode(
                coverage and not bulk_trace, capability_reply,
            )
            if trace_capability_gap:
                trace_truncation_reason = trace_capability_gap
        stage = "continue"
        if bulk_trace and trace_watch_pcs:
            def watch_trace_pc() -> None:
                while not trace_guard_stop.is_set() and process.poll() is None:
                    invalid_pc = _outside_elf_pc_loop(trace_path, trace_watch_pcs)
                    if invalid_pc is not None:
                        trace_guard_pc.append(invalid_pc)
                        try:
                            sock.sendall(b"\x03")
                        except OSError:
                            pass
                        return
                    trace_guard_stop.wait(0.1)

            trace_guard_thread = threading.Thread(
                target=watch_trace_pc, name="rq1-rvvm-invalid-pc-watch", daemon=True,
            )
            trace_guard_thread.start()
        while result is None:
            pc = _rvvm_pc(
                sock, remaining(), pending
            ) if stepcov_enabled else None
            try:
                request = "s" if stepcov_enabled else "c"
                stop = _rvvm_request(
                    sock, request,
                    remaining(), pending,
                    stop_reply=True,
                )
            except TimeoutError:
                stage = "interrupt"
                sock.sendall(b"\x03")
                stop = _rvvm_read_packet(sock, remaining())
            if trace_guard_pc:
                invalid_pc = trace_guard_pc[0]
                result = {
                    "status": "failed",
                    "reason": "guest-pc-outside-elf-loop",
                    "returncode": -1,
                    "elapsed_s": round(time.monotonic() - started, 6),
                    "guest_stop": stop,
                    "guest_status": "failure",
                    "observer_complete": False,
                    "terminal_observed": True,
                    "termination": "guest-invalid-state",
                    "observation_gap": "guest PC repeated outside the executable capsule ELF",
                    "command": command_line,
                }
                trace_truncated = True
                trace_truncation_reason = f"guest-pc-outside-elf-loop:0x{invalid_pc:x}"
                break
            if pc is not None:
                trace_records_total += 1
                if pc not in trace_seen:
                    trace_seen.add(pc)
                    trace_pcs.append(pc)
            if tohost is not None:
                try:
                    raw = _rvvm_request(
                        sock, f"m{tohost:x},8",
                        remaining(), pending,
                    )
                    value = int.from_bytes(bytes.fromhex(raw[:16]), "little")
                except (RuntimeError, ValueError):
                    value = 0
                if value:
                    result = {"status": "passed" if value == 1 else "failed",
                              "reason": "guest-tohost" if value == 1 else "guest-tohost-failure",
                              "returncode": 0 if value == 1 else 1,
                              "elapsed_s": round(time.monotonic() - started, 6),
                              "guest_stop": stop, "tohost_value": value,
                              "guest_status": "pass" if value == 1 else "failure",
                              "observer_complete": True, "terminal_observed": True,
                              "termination": "guest-tohost",
                              "command": command_line}
                    break
            kind = _rvvm_stop_kind(
                sock, remaining(), pending, pc=pc
            ) \
                if stop.startswith(("S05", "T05")) else None
            if kind == "ebreak":
                value = None
                if tohost is not None:
                    try:
                        raw = _rvvm_request(
                            sock, f"m{tohost:x},8",
                            remaining(), pending,
                        )
                        value = int.from_bytes(bytes.fromhex(raw[:16]), "little")
                    except (RuntimeError, ValueError):
                        value = 0
                passed = value == 1 if value is not None else tohost is None
                result = {"status": "passed" if passed else "failed",
                          "reason": (
                              "guest-tohost" if value == 1 else
                              "guest-tohost-failure" if value is not None else
                              "guest-ebreak" if passed else "guest-ebreak-before-tohost"
                          ),
                          "returncode": 0 if passed else 1,
                          "elapsed_s": round(time.monotonic() - started, 6),
                          "guest_stop": stop, "tohost_value": value,
                          "guest_status": "pass" if passed else "failure",
                          "observer_complete": True, "terminal_observed": True,
                          "termination": (
                              "guest-tohost" if value == 1 else
                              "guest-tohost-failure" if value is not None else
                              "guest-ebreak"
                          ),
                          "command": command_line}
                break
            if kind == "ecall":
                continue
            if kind == "exception":
                result = {"status": "failed", "reason": "guest-exception", "returncode": -1,
                          "elapsed_s": round(time.monotonic() - started, 6),
                          "guest_stop": stop, "observer_complete": False,
                          "terminal_observed": True, "termination": "guest-exception",
                          "guest_status": "failure",
                          "command": command_line}
                break
            if not stop.startswith(("S02", "T02", "S05", "T05")):
                if stop.startswith(("S", "T", "X")):
                    result = {"status": "failed", "reason": "guest-exception", "returncode": -1,
                              "elapsed_s": round(time.monotonic() - started, 6),
                              "guest_stop": stop, "observer_complete": False,
                              "terminal_observed": True, "termination": "guest-exception",
                              "guest_status": "failure",
                              "command": command_line}
                    break
                raise RuntimeError(f"RVVM did not stop on guest termination: {stop}")
        attach_observation_frame()
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        external = external_stop_requested or stop_requested() or str(exc) == "external-signal"
        result = ({
            "status": "failed",
            "reason": "guest-pc-outside-elf-loop",
            "returncode": -1,
            "stage": stage,
            "elapsed_s": round(time.monotonic() - started, 6),
            "terminal_observed": True,
            "observer_complete": False,
            "guest_status": "failure",
            "termination": "guest-invalid-state",
            "observation_gap": "guest PC repeated outside the executable capsule ELF",
            "command": command_line,
        } if trace_guard_pc else {
            "status": "gap" if external else "timeout" if isinstance(exc, TimeoutError) else "gap",
            "reason": stop_reason() if external else (
                "wall-timeout" if isinstance(exc, TimeoutError)
                else "rvvm-gdb-probe-failed"
            ),
            "reason_code": stop_reason() if external else None,
            "stage": stage, "error": str(exc),
            "returncode": None if external else 124 if isinstance(exc, TimeoutError) else None,
            "terminal_observed": False if external else None,
            "observer_complete": False if external else None,
            "elapsed_s": round(time.monotonic() - started, 6),
            "command": command_line,
        })
    finally:
        trace_guard_stop.set()
        if trace_guard_thread is not None:
            trace_guard_thread.join(timeout=0.2)
        # Flush gcov before closing the GDB socket.  Closing the socket first
        # can let RVVM exit through its normal error path, bypassing the
        # preloaded stop-flush handler and leaving a valid target run without
        # source-coverage data.
        if coverage and process is not None:
            if process.poll() is not None:
                coverage_flush_status = "process-exited-before-stop"
                coverage_flush_returncode = process.returncode
            elif not graceful_stop:
                coverage_flush_status = "flush-helper-unavailable"
            else:
                try:
                    process.terminate()
                    coverage_flush_signal = "SIGTERM"
                except (OSError, ProcessLookupError):
                    coverage_flush_status = "flush-signal-failed"
                else:
                    # 常见 case 两秒内完成；RAID 忙时再给最多八秒写完 libgcov
                    # 数据。总等待有界，超时仍记录 marker 并回收进程，不会卡住整轮。
                    for grace_seconds in (2.0, 8.0):
                        try:
                            process.wait(timeout=grace_seconds)
                        except subprocess.TimeoutExpired:
                            continue
                        coverage_flush_returncode = process.returncode
                        coverage_flush_status = (
                            "graceful-signal-exit"
                            if process.returncode == 0 else "signal-exit-nonzero"
                        )
                        break
                    else:
                        coverage_flush_status = "flush-timeout"
        if sock is not None:
            sock.close()
        if process is not None:
            cleanup_attempted = True
            root_pid = process.pid
            cleanup_status = "clean"
            try:
                if process.poll() is None:
                    coverage_flush_forced_cleanup = coverage
                    cleanup_killed.update(_kill_process_tree(root_pid))
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    cleanup_killed.update(_kill_process_tree(root_pid))
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        cleanup_status = "gap"
                        cleanup_error = "rvvm-root-process-did-not-exit"
                residual = _descendants(root_pid)
                if process_group:
                    residual.update(_group_members(process_group))
                residual.discard(os.getpid())
                if residual:
                    cleanup_killed.update(residual)
                    _signal_pids(residual, signal.SIGKILL)
                    _reap_children(residual)
                    residual = _descendants(root_pid)
                    if process_group:
                        residual.update(_group_members(process_group))
                    residual.discard(os.getpid())
                remaining_descendants = residual
                if remaining_descendants:
                    cleanup_status = "gap"
                    cleanup_error = "rvvm-descendants-remain"
                if process.poll() is None:
                    cleanup_status = "gap"
                    cleanup_error = cleanup_error or "rvvm-root-process-remains"
            except (OSError, ProcessLookupError, PermissionError) as exc:
                cleanup_status = "gap"
                cleanup_error = f"{type(exc).__name__}: {exc}"
            _reap_children(cleanup_killed)
            if result is not None:
                result["stdout"] = ""
                result["stderr"] = ""
                if coverage:
                    result["trace_pcs"] = trace_pcs
                    result["trace_records"] = len(trace_pcs)
                    result["trace_records_total"] = trace_records_total
                    result["trace_truncated"] = trace_truncated
                    if trace_truncated:
                        result["trace_reason"] = trace_truncation_reason
                    elif trace_truncation_reason:
                        result["trace_reason"] = trace_truncation_reason
                    result["event_counts"] = {"instruction-exec": trace_records_total}
                result.update({
                    "process_pid": root_pid,
                    "process_group": process_group,
                    "cleanup_attempted": cleanup_attempted,
                    "cleanup_status": cleanup_status,
                    "cleanup_killed_pids": sorted(cleanup_killed),
                    "remaining_descendants": sorted(remaining_descendants),
                    "cleanup_error": cleanup_error,
                    "external_stop_requested": external_stop_requested,
                })
    if result is not None and "cleanup_attempted" not in result:
        result.update({
            "process_pid": process.pid if process is not None else None,
            "process_group": process_group,
            "cleanup_attempted": cleanup_attempted,
            "cleanup_status": cleanup_status,
            "cleanup_killed_pids": sorted(cleanup_killed),
            "remaining_descendants": sorted(remaining_descendants),
            "cleanup_error": cleanup_error,
            "external_stop_requested": external_stop_requested,
        })
    if result is None:
        result = {"status": "gap", "reason": "rvvm-gdb-probe-empty",
                  "command": command_line}
        result.update({
            "process_pid": None, "process_group": process_group,
            "cleanup_attempted": cleanup_attempted,
            "cleanup_status": cleanup_status,
            "cleanup_killed_pids": sorted(cleanup_killed),
            "remaining_descendants": sorted(remaining_descendants),
            "cleanup_error": cleanup_error,
        })
    flush_returncode = (
        process.returncode if process is not None else coverage_flush_returncode
    )
    flush_incomplete = coverage and (
        coverage_flush_status not in {
            "graceful-signal-exit", "process-exited-before-stop",
        }
        or coverage_flush_status == "process-exited-before-stop"
        and flush_returncode not in {0, None}
    )
    if flush_incomplete:
        flush_environment = environment or os.environ
        gcov_prefix = flush_environment.get("GCOV_PREFIX")
        if gcov_prefix:
            try:
                marker_root = Path(gcov_prefix)
                marker_root.mkdir(parents=True, exist_ok=True)
                marker_path = marker_root / GCOV_FLUSH_INCOMPLETE_MARKER
                marker_path.write_text(json.dumps({
                    "schema_version": "rq1-gcov-flush-incomplete-v1",
                    "coverage_flush_status": coverage_flush_status,
                    "coverage_flush_signal": coverage_flush_signal,
                    "coverage_flush_forced_cleanup": coverage_flush_forced_cleanup,
                    "coverage_flush_returncode": flush_returncode,
                }, sort_keys=True) + "\n", encoding="utf-8")
                coverage_flush_marker_written = True
            except OSError:
                coverage_flush_marker_error = "gcov-flush-marker-write-failed"
        else:
            coverage_flush_marker_error = "gcov-prefix-unavailable"
    result.update({
        "coverage_flush_status": coverage_flush_status,
        "coverage_flush_signal": coverage_flush_signal,
        "coverage_flush_returncode": (
            process.returncode if process is not None else coverage_flush_returncode
        ),
        "coverage_flush_forced_cleanup": coverage_flush_forced_cleanup,
        "coverage_flush_marker_written": coverage_flush_marker_written,
        "coverage_flush_marker_error": coverage_flush_marker_error,
    })
    if coverage and trace_path is not None and result is not None:
        try:
            trace_size = trace_path.stat().st_size
            with trace_path.open("rb") as stream:
                trace_records = sum(1 for _ in stream)
        except OSError:
            trace_size = 0
            trace_records = 0
        result["trace_records"] = trace_records
        result["trace_records_total"] = trace_records
        result["trace_size_bytes"] = trace_size
        result["trace_complete"] = bool(
            trace_size and result.get("terminal_observed") is True
            and result.get("status") in {"passed", "failed"}
            and result.get("trace_truncated") is not True
        )
        result["trace_status"] = (
            "observed-runner-trace" if result["trace_complete"] else "unavailable"
        )
        if not result["trace_complete"]:
            result["trace_reason"] = result.get("trace_reason") or "rvvm-instruction-trace-incomplete"
    if coverage and trace_path is None and result is not None \
            and not trace_pcs and result.get("status") in {"passed", "failed"}:
        # PC trace/mailbox completeness affects coverage and comparison
        # evidence, not the guest's independently observed terminal.
        result["trace_status"] = "unavailable"
        result["trace_gap_reason"] = trace_truncation_reason or "rvvm-pc-trace-empty"
    return result

def _target_gap(target: dict[str, Any], reason: str,
                 identity: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"id": target["id"], "kind": target["kind"], "status": "gap", "reason": reason,
           "identity": identity}
    row["target_attempted"] = False
    row["target_dispatch"] = _target_dispatch_event(str(target["id"]))
    return row


def _event_observation_recorded(event: Mapping[str, Any]) -> bool:
    observation = event.get("observation")
    if isinstance(observation, Mapping):
        return any(
            name != "schema_version" and value is not None
            for name, value in observation.items()
        )
    return (
        event.get("observer_complete") is True
        or event.get("terminal_observed") is True
        or _trusted_observation_outcome(dict(event)) is not None
    )


def _event_terminal_observed(event: Mapping[str, Any]) -> bool:
    outcome = _trusted_observation_outcome(dict(event))
    if outcome is not None:
        return outcome in {"normal", "trap", "crash", "nonzero-exit"}
    return event.get("terminal_observed") is True


def _censor_shared_run_deadline(
    result: dict[str, Any], limited_by_run_deadline: bool,
) -> dict[str, Any]:
    """Map a host timeout to pending only when the shared run cutoff bound it."""
    if not (limited_by_run_deadline and result.get("status") == "timeout"
            and result.get("reason") == "wall-timeout"):
        return result
    return {
        **result,
        "runner_status": "timeout",
        "runner_reason": result.get("reason"),
        "runner_terminal_observed": result.get("terminal_observed"),
        "status": "gap",
        "reason": "run-wall-clock-exhausted",
        "reason_code": "run-wall-clock-exhausted",
        "terminal_observed": False,
        "observer_complete": False,
        "target_outcome": None,
        "semantic_outcome": None,
        "guest_trap": None,
        "guest_cause": None,
        "guest_exit_channel": None,
        "guest_status": None,
        "guest_stop": None,
        "tohost_value": None,
        "termination": "run-deadline-censored",
        "observation_gap": "run-wall-clock-exhausted",
        "run_deadline_censored": True,
    }


def _targets_passed(config: dict[str, Any], rows: list[dict[str, Any]],
                         expected_targets: set[str] | None = None) -> bool:
    allowed = (set(expected_targets) if expected_targets is not None
               else {str(item.get("id")) for item in config.get("targets", [])
                     if isinstance(item, dict) and item.get("id")})
    ids = [row.get("id") for row in rows]
    return (all(isinstance(item, str) for item in ids)
            and len(ids) == len(set(ids)) and set(ids) == allowed
            and all(_target_testable(row) for row in rows))


def _target_run_status(config: dict[str, Any], targets: list[dict[str, Any]],
                       rows: list[dict[str, Any]], *,
                       reference_pending: bool = False) -> str:
    """Separate valid terminal observations from an all-pass target run."""
    expected = {
        str(target.get("id")) for target in targets
        if isinstance(target, dict) and target.get("id")
    }
    valid = len(rows) == len(targets) and _targets_passed(config, rows, expected)
    if not valid or reference_pending:
        return "gap"
    return "passed" if all(row.get("status") == "passed" for row in rows) else "partial"


def _materialize_dotnet_runtime(
    out: Path, target_id: str, source_binary: Path,
) -> Path:
    """Copy a .NET coverage runtime out of the read-only dependency mount."""
    destination = out / "coverage-binaries" / target_id
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".rq1-dotnet-runtime-complete-v1"
    source_digest = sha256(source_binary)
    if (
        source_digest
        and marker.is_file()
        and (destination / source_binary.name).is_file()
    ):
        try:
            if marker.read_text(encoding="utf-8").strip() == source_digest:
                return destination
        except OSError:
            pass
    shutil.copytree(
        source_binary.parent, destination, dirs_exist_ok=True,
        copy_function=shutil.copy2,
    )
    materialized = destination / source_binary.name
    if not materialized.is_file() or sha256(materialized) != source_digest:
        raise OSError("dotnet coverage runtime copy is incomplete")
    marker.write_text(f"{source_digest}\n", encoding="utf-8")
    return destination


def _dotnet_collector_env(env: Mapping[str, str]) -> dict[str, str]:
    """Keep .NET diagnostic IPC on the system temporary filesystem."""
    result = dict(env)
    result["TMPDIR"] = "/tmp"
    return result


def _renode_stop_on_output(
    *, mailbox: int | None, trace_active: bool = False,
) -> tuple[str, ...]:
    """Stop on guest evidence unless an active trace needs its flush command."""
    if trace_active:
        return ()
    return ("RQ1_RVOBS1=", "RQ1_EXCEPTION") if mailbox is not None else (
        "CPU abort", "RQ1_EXCEPTION",
    )


def run_targets(config: dict[str, Any], out: Path, *,
                 provenance: dict[str, Any] | None = None,
                 artifacts: dict[str, Path] | None = None,
                 bare_isa: str | None = None,
                 reference_override: dict[str, Any] | None = None,
                 target_ids: set[str] | None = None,
                 target_timeout_seconds: int | None = None,
                 expected_outcome: str | None = None,
                 run_deadline: float | None = None,
                 coverage_root: Path | None = None,
                 feature_cache_root: Path | None = None,
                 collect_source: bool = True,
                 skip_reference: bool = False,
                 include_guest_metrics: bool | None = None,
                 coverage_side_on_timeout: bool = False) -> dict[str, Any]:
    coverage_enabled = simulator_coverage_enabled()
    if include_guest_metrics is None:
        include_guest_metrics = rv_instruction_metrics_enabled(config)
    deferred_renode_coverage = coverage_root is not None and not collect_source
    coverage_root = Path(coverage_root or out / "coverage-raw")
    if not isinstance(config, dict):
        return {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap", "reason": "config-invalid",
            "provenance": provenance or {},
            "reference": {}, "references": {}, "targets": [],
        }
    if not isinstance(artifacts, Mapping) or not artifacts:
        return {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap", "reason": "artifacts-required",
            "provenance": provenance or _provenance(config),
            "reference": {}, "references": {}, "targets": [],
        }
    def artifact_path(value: object) -> Path | None:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            try:
                return Path(value)
            except (OSError, TypeError, ValueError):
                return None
        return None

    raw_user_elf = artifacts.get("linux-user")
    raw_bare_elf = artifacts.get("bare-metal")
    user_elf = artifact_path(raw_user_elf)
    bare_elf = artifact_path(raw_bare_elf)
    if (raw_user_elf is not None and user_elf is None) or (
        raw_bare_elf is not None and bare_elf is None
    ):
        return {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap", "reason": "artifacts-invalid",
            "provenance": provenance or _provenance(config),
            "reference": {}, "references": {}, "targets": [],
        }
    rvvm_elf = bare_elf
    # Keep the candidate's complete ISA/ABI profile for ELF decoding.  The
    # old fallback used the global lane ABI even when a tool emitted lp64d,
    # which silently projected every artifact through the common lane.
    # `config.lane` is only a legacy metadata fallback for hand-written
    # callers that omit artifact identity; queue-produced cases carry their
    # own generator profile here.
    artifact_profile = str(
        bare_isa or f"{config['lane']['isa']}/{config['lane']['mabi']}"
    ).strip()
    if "/" not in artifact_profile:
        artifact_profile = f"{artifact_profile}/{config['lane']['mabi']}"
    artifact_features: dict[str, dict[str, Any]] = {}
    execution_binaries: dict[str, Path | None] = {}
    coverage_profile_dirs: dict[str, Path] = {}
    coverage_profile_ids: dict[str, str] = {}
    for stratum, elf in (("bare-metal", bare_elf), ("linux-user", user_elf)):
        if elf and elf.is_file():
            feature_path = out / "artifacts" / f"{stratum.replace('-', '_')}.features.json"
            if feature_cache_root:
                cache_root = Path(feature_cache_root)
                privilege_mode = "machine" if stratum == "bare-metal" else "user"
                artifact_sha = sha256(elf) or ""
                cache_key = hashlib.sha256("\0".join((
                    artifact_sha, DECODER_VERSION, artifact_profile,
                    privilege_mode, stratum,
                )).encode()).hexdigest()
                cache_path = cache_root / f"{cache_key}.features.json"
                with _ARTIFACT_FEATURE_CACHE_LOCK:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    try:
                        feature = load_json(cache_path) if cache_path.is_file() else {}
                    except (OSError, TypeError, ValueError):
                        feature = {}
                    cache_hit = (
                        feature.get("elf_sha256") == artifact_sha
                        and feature.get("decoder_version") == DECODER_VERSION
                        and feature.get("isa_profile") == artifact_profile.split("/", 1)[0]
                        and feature.get("privilege_mode") == privilege_mode
                        and isinstance(feature.get("instructions"), list)
                        and bool(feature["instructions"])
                    )
                    if not cache_hit:
                        feature = decode_elf(
                            elf, artifact_profile, privilege_mode=privilege_mode,
                        )
                    if not cache_hit and (
                        feature.get("elf_sha256") == artifact_sha
                        and feature.get("decoder_version") == DECODER_VERSION
                        and isinstance(feature.get("instructions"), list)
                        and bool(feature["instructions"])
                    ):
                        write_json_replace(cache_path, feature)
                        cache_hit = True
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                    if not feature_path.exists():
                        try:
                            if cache_hit:
                                os.link(cache_path, feature_path)
                            else:
                                write_json_replace(feature_path, feature)
                        except OSError:
                            write_json_replace(feature_path, feature)
            else:
                feature = decode_elf(
                    elf, artifact_profile,
                    privilege_mode="machine" if stratum == "bare-metal" else "user",
                )
                write_json(feature_path, feature)
            artifact_features[stratum] = feature
    def feature_ref(stratum: str) -> dict[str, str]:
        return {"ref": f"artifacts/{stratum.replace('-', '_')}.features.json"}
    bare_isa = artifact_profile.split("/", 1)[0]
    results = []
    try:
        timeout = _configured_target_timeout(config, target_timeout_seconds)
    except ValueError as error:
        return {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap", "reason": "config-invalid",
            "error": str(error), "provenance": provenance or _provenance(config),
            "reference": {}, "references": {}, "targets": [],
        }
    target_timeouts: dict[str, int] = {}

    def target_budget_details(target_id: str | None = None) -> tuple[float | None, bool]:
        budget = target_timeouts.get(target_id, timeout)
        if run_deadline is None:
            return float(budget), False
        remaining = run_deadline - time.monotonic()
        if remaining <= 0:
            return None, True
        return max(0.001, min(float(budget), remaining)), remaining <= budget

    def run_expired() -> bool:
        return run_deadline is not None and time.monotonic() >= run_deadline
    all_targets = config.get("targets", [])
    all_targets = all_targets if isinstance(all_targets, (list, tuple)) else ()
    targets = [target for target in all_targets
               if isinstance(target, dict)
               if target_ids is None or target.get("id") in target_ids]
    try:
        target_timeouts = {
            str(target["id"]): _configured_target_timeout(
                config, target_timeout_seconds, target,
            )
            for target in targets
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap", "reason": "config-invalid",
            "error": str(error), "provenance": provenance or _provenance(config),
            "reference": {}, "references": {}, "targets": [],
        }
    # 板子只提供 linux-user 形态的 reference；bare-metal 形态登记具名 gap，
    # 只在 board 覆盖的 stratum 上要求差分闭合。
    board_strata = ("linux-user",) if str(
        (config.get("board_differential") or {}).get("coverage", "")
    ).startswith("linux-user") else ("bare-metal", "linux-user")

    def collect_reference_records() -> dict[str, dict[str, Any]]:
        reference_records = {}
        for stratum in ("bare-metal", "linux-user"):
            if stratum == "linux-user" and not any(
                    str(target.get("execution_model", "")).startswith("linux-user") for target in targets):
                continue
            override = (
                reference_override.get(stratum)
                if isinstance(reference_override, dict)
                and isinstance(reference_override.get(stratum), dict)
                else None
            )
            if override is None:
                override = _board_reference_record(
                    out, stratum,
                    bare_elf if stratum == "bare-metal" else user_elf,
                    timeout, run_deadline=run_deadline,
                )
            record = {
                **override,
                "stratum": stratum,
                "artifact_features": feature_ref(stratum),
                "static_evidence": feature_ref(stratum),
            }
            record["trusted_reference_coverage"] = (
                reference_coverage(
                    artifact_features.get(stratum) or {},
                    {"trace_pcs": record.get("trace_pcs", []),
                     "outcome": record.get("terminal"),
                     "exit_code": record.get("returncode")},
                )
                if _trusted_reference_board(record)
                else {
                    "schema_version": "rq1-trusted-reference-coverage-v1",
                    "status": "gap", "reason": "trusted-reference-board-required",
                }
            )
            reference_records[stratum] = record
        return reference_records

    if len(targets) > 1:
        def run_target_worker(target: dict[str, Any]) -> dict[str, Any]:
            worker_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target["id"]))
            worker_root = out / "target-workers" / worker_name
            try:
                return run_targets(
                    config, worker_root, provenance=provenance, artifacts=artifacts,
                    bare_isa=artifact_profile,
                    reference_override=reference_override, target_ids={target["id"]},
                    target_timeout_seconds=target_timeout_seconds,
                    expected_outcome=expected_outcome,
                    run_deadline=run_deadline, coverage_root=coverage_root,
                    collect_source=collect_source,
                    skip_reference=True,
                    coverage_side_on_timeout=coverage_side_on_timeout,
                )
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                return {"status": "gap", "reason": f"target-worker-error:{type(exc).__name__}",
                        "error": str(exc)[:500],
                        "error_detail": {"exception_type": type(exc).__name__,
                                          "message": str(exc)[:500],
                                          "errno": getattr(exc, "errno", None),
                                          "target": str(target.get("id")),
                                          "worker_root": str(worker_root)},
                        "targets": [], "references": {}}

        reference_pool = ThreadPoolExecutor(max_workers=1) if not skip_reference else None
        reference_future = reference_pool.submit(collect_reference_records) if reference_pool else None
        pool = ThreadPoolExecutor(max_workers=len(targets))
        futures = {pool.submit(run_target_worker, target): target for target in targets}
        all_futures = set(futures)
        if reference_future is not None:
            all_futures.add(reference_future)
        remaining = None if run_deadline is None else max(0, run_deadline - time.monotonic())
        done: set[Any] = set()
        pending_futures = set(all_futures)
        while pending_futures:
            wait_timeout = 0.1 if remaining is None else min(0.1, remaining)
            newly_done, pending_futures = wait(pending_futures, timeout=wait_timeout)
            done.update(newly_done)
            if stop_requested():
                break
            if remaining is not None:
                remaining = max(0, remaining - wait_timeout)
                if remaining <= 0:
                    break
        reference_pending = reference_future is not None and reference_future not in done
        try:
            merged_references = reference_future.result() if reference_future is not None and reference_future in done else {}
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            merged_references, reference_pending = {}, True
        # Join every worker before returning so it cannot keep writing target
        # evidence after the caller seals the ledger. Then retain the worker's
        # actual attempted/pending state instead of replacing it with an
        # unattempted synthetic gap.
        pool.shutdown(wait=True, cancel_futures=True)
        worker_results = []
        for future, target in futures.items():
            try:
                worker_results.append(future.result())
            except Exception as exc:
                worker_results.append({
                    "status": "gap",
                    "reason": f"target-worker-error:{type(exc).__name__}",
                    "error": str(exc)[:500],
                    "targets": [], "references": {},
                })
        if reference_pool is not None:
            reference_pool.shutdown(wait=True, cancel_futures=True)
        merged_targets = [
            row
            for result in worker_results
            for row in result.get("targets", [])
            if isinstance(row, dict)
        ]
        if not skip_reference and any(
            _reference_for_target(config, target)[0] in board_strata
            and not _trusted_reference_board(
                merged_references.get(_reference_for_target(config, target)[0], {}))
            for target in targets
        ):
            reference_pending = True
        for row in merged_targets:
            target = next(item for item in targets if item.get("id") == row.get("id"))
            stratum, _ = _reference_for_target(config, target)
            reference = merged_references.get(stratum, {})
            row.update({
                "reference_id": reference.get("id"),
                "reference_status": reference.get("status"),
                "reference_identity": reference.get("identity"),
                "trusted_reference_coverage": reference.get("trusted_reference_coverage"),
                "trusted_reference_signature": reference.get("trusted_reference_signature"),
            })
            row["differential"] = _differential_record(reference, row)
        merged_targets.sort(key=lambda row: targets.index(next(
            target for target in targets if target.get("id") == row.get("id")
        )))
        target_result = {
            "target_timeout_seconds": timeout,
            "schema_version": "rq1-comparison-target-run-v1",
            "provenance": provenance or _provenance(config),
            "status": _target_run_status(
                config, targets, merged_targets, reference_pending=reference_pending,
            ),
            "reference": merged_references.get("bare-metal", {}),
            "references": merged_references,
            "targets": merged_targets,
        }
        if coverage_enabled and collect_source:
            summary = aggregate_simulator_coverage(
                merged_targets, include_guest_metrics=include_guest_metrics,
            )
            write_json(out / "coverage" / "summary.json", summary)
            target_result["coverage"] = summary
            target_result["coverage_timing"] = {
                "target_execution_seconds": round(sum(
                    float(row.get("elapsed_s") or 0) for row in merged_targets
                ), 6),
                "collection_deferred": not collect_source,
            }
        return target_result

    reference_records = {} if skip_reference else collect_reference_records()
    reference_record = reference_records.get("bare-metal", {})
    for target in targets:
        stratum, _ = _reference_for_target(config, target)
        target_reference = reference_records.get(stratum, {})
        binary = dep(target.get("binary"))
        coverage_binary = dep(target.get("coverage_binary")) if coverage_enabled else binary
        expected_coverage_sha = target.get("coverage_binary_sha256")
        coverage_binary_sha_matches = bool(
            coverage_enabled and coverage_binary and coverage_binary.is_file()
            and isinstance(expected_coverage_sha, str)
            and sha256(coverage_binary) == expected_coverage_sha.lower()
        )
        target_coverage_enabled = coverage_enabled and bool(
            target.get("coverage_binary")
            and coverage_binary
            and coverage_binary.is_file()
            and coverage_binary_sha_matches
        )
        exec_binary = coverage_binary if target_coverage_enabled else binary
        execution_binaries[str(target["id"])] = exec_binary
        profile_id = hashlib.sha256(
            f"{Path(out).resolve()}:{time.time_ns()}:{threading.get_ident()}".encode()
        ).hexdigest()
        raw_dir = (
            coverage_root / str(target.get("id")) / "attempts" / profile_id
            if target_coverage_enabled else None
        )
        if raw_dir is not None:
            coverage_profile_dirs[str(target["id"])] = raw_dir
            coverage_profile_ids[str(target["id"])] = profile_id
        run_env = ({**os.environ, **runtime_env(target, raw_dir)}
                   if raw_dir else None)
        rax_stop_flush_enabled = False
        if target_coverage_enabled and target.get("kind") == "rax" and run_env is not None:
            stop_flush = Path(run_env.get("RQ1_RVVM_STOP_FLUSH", ""))
            if stop_flush.is_file():
                run_env = dict(run_env)
                run_env["LD_PRELOAD"] = os.pathsep.join(filter(
                    None, (str(stop_flush), run_env.get("LD_PRELOAD", "")),
                ))
                rax_stop_flush_enabled = True
        identity = _identity(dep(target.get("identity")), binary, target)
        artifact = user_elf if stratum == "linux-user" else (rvvm_elf if target["kind"] == "rvvm" else bare_elf)
        feature_instructions = artifact_features.get(stratum, {}).get("instructions", [])
        trace_watch_pcs = {
            item["address"] for item in feature_instructions
            if isinstance(item, dict) and type(item.get("address")) is int
        }
        if not artifact or not artifact.is_file():
            results.append(_target_gap(target, "stratum-artifact-missing", identity))
            continue
        if not binary or not binary.is_file():
            results.append(_target_gap(target, "binary-missing", identity))
            continue
        if (identity.get("patch_lineage") or identity.get("source_provenance") in {
            "patch-validated-build", "verified-git-archive+controlled-patch",
        }) and not is_allowlisted_patched_identity(identity):
            results.append(_target_gap(target, "target-source-patch-forbidden", identity))
            continue
        supported_widths = target.get("supported_isa_widths")
        profile_width = re.match(r"^rv(32|64)", artifact_profile.lower())
        if isinstance(supported_widths, list) and profile_width:
            artifact_xlen = int(profile_width.group(1))
            if artifact_xlen not in supported_widths:
                row = _target_gap(target, "target-isa-width-unsupported", identity)
                row.update({
                    "target_applicability": "unsupported-isa-width",
                    "artifact_xlen": artifact_xlen,
                    "supported_isa_widths": supported_widths,
                })
                results.append(row)
                continue
        if run_expired():
            results.append(_target_gap(target, "run-wall-clock-exhausted", identity))
            continue
        if target["kind"] == "unicorn":
            command_line = ["python-unicorn-api", str(exec_binary), str(bare_elf)]
            target_timeout, deadline_limited = target_budget_details(str(target["id"]))
            if target_timeout is None:
                results.append(_target_gap(target, "run-wall-clock-exhausted", identity))
                continue
            def run_unicorn():
                return _run_unicorn_isolated(
                    exec_binary, bare_elf, target_timeout, coverage_enabled,
                )
            result = _run_in_env(run_env, run_unicorn)
            result = _expected_outcome_result(
                _semanticize_result(target, _persist_run(out, target["id"], command_line, result, env=run_env)),
                expected_outcome,
            )
            result = _censor_shared_run_deadline(result, deadline_limited)
            record = _target_record(
                target, result, identity, target_reference, feature_ref(stratum),
            )
            results.append(record)
            continue
        launcher = dep(target.get("launcher"))
        mapping = {"binary": exec_binary, "user_elf": user_elf, "bm_elf": bare_elf, "launcher": launcher}
        run_cwd = None
        coverage_trace = None
        coverage_output = None
        normal_command_line = None
        normal_run_cwd = None
        normal_run_env = None
        coverage_side_command_line = None
        coverage_side_watch_command_line = None
        coverage_side_trace = None
        coverage_watch_script = None
        coverage_side_cwd = None
        coverage_side_env = None
        trace_setup = trace_stop = ""
        if target["kind"] == "renode":
            target_dir = out / "targets"
            target_dir.mkdir(parents=True, exist_ok=True)
            if target_coverage_enabled:
                trace_dir = out / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                coverage_trace = trace_dir / f"{target['id']}.renode.pc"
                trace_setup = _renode_trace_setup(coverage_trace)
                trace_stop = "sysbus.cpu DisableExecutionTracing\n"
            script = target_dir / "renode.resc"
            entry = 0x80000000
            try:
                entry, segments = elf_segments(bare_elf)
                memory_base = min(address & ~(PAGE - 1) for address, _, _, _ in segments)
                memory_end = max(address + max(file_size, mem_size)
                                 for address, _, file_size, mem_size in segments)
                memory_size = max(PAGE, ((memory_end - memory_base + PAGE - 1) // PAGE) * PAGE)
            except (OSError, struct.error, ValueError):
                memory_base, memory_size = 0x80000000, 0x01000000
            _, symbols = symbol_offsets_from_elf(
                bare_elf, ("tohost", "mtvec_handler", "obs_buf", "result_buffer"),
                allow_before_start=True,
            )
            has_tohost = "tohost" in symbols
            tohost_hook = hex(int(symbols["tohost"], 0)) if has_tohost else None
            has_guest_trap = "mtvec_handler" in symbols
            mailbox_name = "obs_buf" if "obs_buf" in symbols else "result_buffer"
            mailbox = int(symbols[mailbox_name], 0) if mailbox_name in symbols else None
            extensions = {str(item.get("extension", "")).lower()
                          for item in artifact_features.get("bare-metal", {}).get("instructions", [])
                          if item.get("extension")}
            renode_isa = bare_isa
            if extensions:
                base = "rv64" if bare_isa.startswith("rv64") else "rv32"
                renode_isa = base + "".join(ext for ext in "imafdcv" if ext in extensions)
            if "zicsr" in bare_isa.lower().split("_")[1:]:
                renode_isa += "_zicsr"
            trap_platform = (
                "trap: Memory.MappedMemory @ sysbus 0x1000\n"
                "    size: 0x1000\n"
                if not has_guest_trap else ""
            )
            trap_setup = (
                # A missing guest trap handler still needs a non-faulting sink:
                # an ebreak here re-enters the same address and aborts Renode
                # before the tohost wait script can read the committed value.
                "sysbus WriteDoubleWord 0x1010 0x0000006f\n"
                "sysbus.cpu MTVEC 0x1010\n"
                if not has_guest_trap else ""
            )
            watch_action = (
                "if monitor.Machine.UserState != 'RQ1_DONE' and value != 0: "
                "monitor.Machine.UserState = 'RQ1_DONE'"
            )
            watch_hook = (
                "\n".join(
                    f"sysbus AddWatchpointHook {tohost_hook} {width} Write \"{watch_action}\""
                    for width in ("Word", "DoubleWord", "QuadWord")
                )
                if has_tohost else ""
            )
            interrupt_hook = (
                ""
                if has_tohost else
                '''sysbus.cpu AddHookAtInterruptBegin "if monitor.Machine.UserState != 'RQ1_DONE': print('RQ1_SIM_EVENT:interrupt'); print('RQ1_EXCEPTION'); monitor.Machine.UserState = 'RQ1_EXCEPTION'"'''
            )
            mailbox_dump = ""
            if mailbox is not None:
                mailbox_dump = (
                    "\nif machine is not None and machine.UserState == 'RQ1_DONE':\n"
                    f"    header = [int(machine.SystemBus.ReadByte({mailbox} + i)) for i in range({OBSERVATION_HEADER_SIZE})]\n"
                    "    magic = [0x52, 0x56, 0x4f, 0x42, 0x53, 0x31, 0x00, 0x00]\n"
                    f"    memory_size = sum(header[i] << (8 * (i - {16 + 32 * 8})) for i in range({16 + 32 * 8}, {OBSERVATION_HEADER_SIZE})) if header[:8] == magic else -1\n"
                    f"    if 0 <= memory_size <= {MAX_OBSERVATION_MEMORY_SIZE}:\n"
                    f"        frame = header + [int(machine.SystemBus.ReadByte({mailbox} + i)) for i in range({OBSERVATION_HEADER_SIZE}, {OBSERVATION_HEADER_SIZE} + memory_size)]\n"
                    "        print('RQ1_RVOBS1=' + ''.join('{:02x}'.format(value) for value in frame))\n"
                )
            tohost_dump = ""
            if has_tohost:
                tohost_dump = (
                    "\nif machine is not None and machine.UserState == 'RQ1_DONE':\n"
                    f"    tohost_value = sum(int(machine.SystemBus.ReadByte({int(symbols['tohost'], 0)} + i)) << (8 * i) for i in range(8))\n"
                    "    print('RQ1_TOHOST_VALUE=' + str(tohost_value))\n"
                    "    if tohost_value != 0:\n"
                    "        print('RQ1_TOHOST')\n"
                )
            tohost_commit_wait = (
                'if machine is not None and machine.UserState == \'RQ1_DONE\':\n'
                f'    while sum(int(machine.SystemBus.ReadByte({int(symbols["tohost"], 0)} + i)) << (8 * i) for i in range(8)) == 0:\n'
                '        time.sleep(0.001)\n'
                if has_tohost else ''
            )
            wait_command = (
                'set wait_script """\nimport time\n'
                'machine = monitor.Machine\n'
                "while machine is not None and machine.UserState not in ('RQ1_DONE', 'RQ1_EXCEPTION'):\n"
                '    time.sleep(0.001)\n'
                + tohost_commit_wait
                + 'if machine is not None:\n'
                + '    machine.Pause()\n'
                + tohost_dump
                + mailbox_dump
                + '"""\npython $wait_script'
                if has_tohost or has_guest_trap else ""
            )
            def render_renode_script(path: Path, setup: str, stop: str) -> None:
                write_text_once(path, f'''mach create
machine LoadPlatformDescriptionFromString """
cpu: CPU.RiscV64 @ sysbus
    cpuType: "{renode_isa}"
    allowUnalignedAccesses: true
    timeProvider: empty
mem: Memory.MappedMemory @ sysbus 0x{memory_base:x}
    size: 0x{memory_size:x}
{trap_platform}"""
sysbus LoadELF @{bare_elf}
sysbus.cpu PC 0x{entry:x}
{trap_setup}
{setup}python "monitor.Machine.UserState = 'RQ1_WAIT'"
{interrupt_hook}
{watch_hook}
start
# no extra watch hook
{wait_command}
{stop} quit
''')

            render_renode_script(script, trace_setup, trace_stop)
            coverage_script: Path | None = None
            if target_coverage_enabled and target.get("source_coverage", {}).get("collector") == "dotnet":
                # Keep ordinary coverage runs light; timeout/invalid-PC replay
                # uses a separate native trace to stop on an out-of-ELF loop.
                coverage_script = target_dir / "renode-coverage.resc"
                render_renode_script(coverage_script, "", "")
                coverage_watch_script = target_dir / "renode-coverage-watch.resc"
                coverage_side_trace = trace_dir / f"{target['id']}-coverage-side.renode.pc"
                render_renode_script(
                    coverage_watch_script,
                    _renode_trace_setup(coverage_side_trace),
                    trace_stop,
                )
            # The Renode distribution is already selected by the target
            # binary path in config; a second hard-coded runtime directory
            # can silently execute a different build.
            run_root = binary.parent
            dotnet_root = Path(os.environ.get(
                "RQ1_DOTNET_RUNTIME",
                str(launcher.parent) if launcher else "/tmp/dotnet-8",
            ))
            if target_coverage_enabled:
                # dotnet-coverage needs a writable runtime tree. The external
                # comparison mounts simulator artifacts read-only, while the
                # framework path already materializes this cache.
                runtime_cache_root = (
                    coverage_root.parent if coverage_root is not None else out
                )
                run_root = _materialize_dotnet_runtime(
                    runtime_cache_root, str(target["id"]), exec_binary,
                )
                run_binary = run_root / exec_binary.name
            else:
                run_binary = run_root / binary.name
            run_launcher = (
                dotnet_root / launcher.name if target_coverage_enabled and launcher
                else launcher
            )
            if not run_binary.is_file():
                results.append(_target_gap(target, "runtime-binary-missing", identity))
                continue
            run_cwd = run_root
            run_env = {
                **(run_env or os.environ),
                "PATH": os.pathsep.join(filter(
                    None, (str(dotnet_root), (run_env or os.environ).get("PATH", "")),
                )),
                "DOTNET_ROOT": str(dotnet_root),
                "DOTNET_ROOT_X64": str(dotnet_root),
                "DOTNET_CLI_HOME": "/tmp", "HOME": "/tmp",
                "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
            }
            if deferred_renode_coverage and target_coverage_enabled:
                # The target verdict must come from the ordinary Renode
                # runtime.  The instrumented runtime is retained as a
                # separate source-coverage side run below; otherwise a slow
                # Cobertura flush turns a valid mailbox observation into a
                # target timeout in every external method.
                normal_run_cwd = binary.parent
                normal_run_env = dict(run_env)
                normal_command_line = [
                    str(launcher), str(binary), "--disable-gui", "--plain",
                    "--console", str(script),
                ]
            if target["kind"] == "renode" and target_coverage_enabled:
                # Keep the collector's IPC endpoint on the short system
                # temporary filesystem; guest traces remain explicitly
                # run-owned below.
                run_env = _dotnet_collector_env(run_env)
            coverage_launcher = dep((target.get("source_coverage") or {}).get("launcher"))
            coverage_output = (
                raw_dir / f"{digest(str(out.resolve()))}.cobertura.xml"
                if raw_dir else None
            )
            coverage_include_files = [run_binary]
            infrastructure = run_binary.with_name("Infrastructure.dll")
            if infrastructure.is_file():
                coverage_include_files.append(infrastructure)
            if coverage_output:
                # A resumed case reuses its deterministic path. Remove only
                # that case's previous reports so an old attempt cannot be
                # counted alongside a new one.
                for stale in (coverage_output, Path(str(coverage_output) + ".gz")):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
            command_line = (
                [str(coverage_launcher), "collect", "--nologo", "-f", "cobertura",
                 "--include-files", ",".join(str(path) for path in coverage_include_files),
                 "-o", str(coverage_output), "--", str(run_launcher), str(run_binary),
                 "--disable-gui", "--plain", "--console",
                 str(coverage_script or script)]
                if target_coverage_enabled and coverage_launcher and coverage_output
                and (target.get("source_coverage") or {}).get("collector") == "dotnet"
                else [str(run_launcher), str(run_binary), "--disable-gui", "--plain",
                      "--console", str(script)]
            )
            if deferred_renode_coverage and target_coverage_enabled:
                coverage_side_command_line = list(command_line)
                if coverage_watch_script is not None:
                    coverage_side_watch_command_line = list(command_line)
                    coverage_side_watch_command_line[-1] = str(coverage_watch_script)
                coverage_side_cwd = run_cwd
                coverage_side_env = dict(run_env)
        elif target["kind"] == "qemu":
            if target_coverage_enabled:
                plugin = dep((target.get("simulator_coverage") or {}).get("plugin"))
                if not plugin or not plugin.is_file():
                    target_coverage_enabled = False
                    exec_binary = binary
                    execution_binaries[str(target["id"])] = binary
                    raw_dir = None
                    run_env = None
                else:
                    trace_dir = out / "traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    adapter = str((target.get("simulator_coverage") or {}).get("adapter", ""))
                    if adapter == "qemu-drcov-basic-block":
                        coverage_trace = trace_dir / f"{target['id']}.qemu.drcov"
                        command_line = [
                            str(exec_binary), "-plugin",
                            f"{plugin},filename={coverage_trace}", str(user_elf),
                        ]
                    else:
                        coverage_trace = trace_dir / f"{target['id']}.qemu.pc"
                        command_line = [
                            str(exec_binary), "-d", "plugin", "-D", str(coverage_trace),
                            "-plugin", str(plugin), str(user_elf),
                        ]
            if not target_coverage_enabled:
                command_line = [str(exec_binary), str(user_elf)]
            if expected_outcome == "trap":
                # Trap attribution uses QEMU siginfo; it is independent of the coverage trace.
                command_line.insert(-1, "-strace")
        elif target["kind"] == "libriscv":
            translated = target["id"] == "T-LRSV-TRANS"
            if translated and target_coverage_enabled:
                command_line = [str(exec_binary), "--silent", "--accurate", "--trace", str(user_elf)]
            elif translated:
                command_line = [str(exec_binary), "--silent", "--accurate", str(user_elf)]
            else:
                command_line = [str(exec_binary), "--silent", "--accurate", "--no-translate",
                                *(["--debug", "--from-start"] if target_coverage_enabled else []),
                                str(user_elf)]
        elif target["kind"] == "rax":
            if target_coverage_enabled:
                trace_dir = out / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                coverage_trace = trace_dir / f"{target['id']}.rax"
            command_line = [str(exec_binary), "--arch", "riscv64", "--backend", "emulator",
                            "--memory", "128M",
                            *(["--trace", str(coverage_trace)]
                              if target_coverage_enabled and coverage_trace else []),
                            "--kernel", str(bare_elf)]
        else:
            command_line = _format_command(target.get("probe_command", []), mapping)
        if target["kind"] == "libriscv":
            translated = target["id"] == "T-LRSV-TRANS"
            if target_coverage_enabled:
                trace_dir = out / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                coverage_trace = trace_dir / f"{target['id']}.libriscv.log"
            target_timeout, deadline_limited = target_budget_details(str(target["id"]))
            if target_timeout is None:
                results.append(_target_gap(target, "run-wall-clock-exhausted", identity))
                continue
            result = _run_logged(
                out, target["id"], command_line, target_timeout, env=run_env,
                stdin_text="\n" if target_coverage_enabled and not translated else None,
                trace_compact_path=coverage_trace if target_coverage_enabled else None,
                trace_record_mode=(
                    "libriscv-translated" if translated else "libriscv-interpreter"
                ),
            )
            if target_coverage_enabled:
                # 只有 Target wall-timeout、收集线程异常或证据解析失败才把
                # trace 标为不完整；正常运行不会因输出量被截断。
                result["trace_truncated"] = (
                    result.get("termination") == "simulator-fuel"
                    or result.get("reason") == "wall-timeout"
                    or bool(result.get("trace_truncated")))
                result["trace_path"] = str(coverage_trace.relative_to(out))
            result = _libriscv_result(result)
            fallback_deadline_limited = False
            if (
                translated
                and result.get("reason") == "instruction-fuel-exhausted"
                and user_elf is not None
            ):
                fallback_timeout, fallback_deadline_limited = target_budget_details(
                    str(target["id"]),
                )
                if fallback_timeout is not None:
                    fallback_command = [
                        str(exec_binary), "--silent", "--accurate", "--no-translate",
                        str(user_elf),
                    ]
                    fallback = _run_logged(
                        out, f"{target['id']}-interpreter-fallback", fallback_command,
                        fallback_timeout, env=run_env,
                        trace_record_mode="libriscv-interpreter",
                    )
                    fallback = _libriscv_result(fallback)
                    # Promote the raw verified frame before deciding whether
                    # this fallback is usable; _run_logged has not yet passed
                    # through the generic semantic layer at this point.
                    fallback = _semanticize_result(target, fallback)
                    fallback["translation_fallback"] = {
                        "status": fallback.get("status"),
                        "primary_reason": result.get("reason"),
                        "primary_termination": result.get("termination"),
                        "primary_command": command_line,
                        "fallback_command": fallback_command,
                    }
                    if target_coverage_enabled:
                        fallback["trace_path"] = str(coverage_trace.relative_to(out))
                        fallback["trace_truncated"] = True
                        fallback["trace_complete"] = False
                        fallback["trace_reason"] = "translation-fallback-no-translate"
                    if (
                        fallback.get("status") == "passed"
                        and fallback.get("observer_complete") is True
                    ):
                        result = fallback
            result = _expected_outcome_result(
                _semanticize_result(target, result), expected_outcome,
            )
            result = _censor_shared_run_deadline(
                result, deadline_limited or fallback_deadline_limited,
            )
            record = _target_record(
                target, result, identity, target_reference, feature_ref(stratum),
            )
            results.append(record)
            continue
        if target["kind"] == "rvvm":
            if target_coverage_enabled:
                trace_dir = out / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                coverage_trace = trace_dir / f"{target['id']}.rvvm.pc"
            target_timeout, deadline_limited = target_budget_details(str(target["id"]))
            if target_timeout is None:
                results.append(_target_gap(target, "run-wall-clock-exhausted", identity))
                continue
            result = _run_rvvm_probe(exec_binary, rvvm_elf, target_timeout,
                                     bare_isa or target_reference.get("machine_isa", "rv64i"),
                                     target_coverage_enabled,
                                     str(target.get("translation_mode", "interpreter")),
                                     run_env,
                                     trace_path=coverage_trace if target_coverage_enabled else None,
                                     trace_watch_pcs=trace_watch_pcs)
            if coverage_trace:
                result["trace_path"] = str(coverage_trace.relative_to(out))
            result = _expected_outcome_result(
                _semanticize_result(target, _persist_run(out, target["id"], result.pop("command"), result, env=run_env)),
                expected_outcome,
            )
            result = _censor_shared_run_deadline(result, deadline_limited)
            record = _target_record(
                target, result, identity, target_reference, feature_ref(stratum),
            )
            results.append(record)
            continue
        if not command_line or any(not value for value in command_line):
            row = _target_gap(target, "probe-command-missing", identity)
            row["command"] = command_line or None
            results.append(row)
            continue
        target_timeout, deadline_limited = target_budget_details(str(target["id"]))
        if target_timeout is None:
            results.append(_target_gap(target, "run-wall-clock-exhausted", identity))
            continue
        execution_command = normal_command_line or command_line
        execution_cwd = normal_run_cwd or run_cwd
        execution_env = normal_run_env or run_env
        result = _run_logged(
            out, target["id"], execution_command, target_timeout,
            cwd=execution_cwd, env=execution_env,
            stdin_text="" if target["kind"] == "renode" else None,
            timeout_stdin_text=(
                "quit\n" if target["kind"] == "renode" and target_coverage_enabled
                else "" if target["kind"] == "renode" else None
            ),
            trace_path=coverage_trace if target_coverage_enabled else None,
            trace_watch_pcs=(
                trace_watch_pcs if target["kind"] in {"rax", "renode"} else None
            ),
            trace_watch_plain_pcs=target["kind"] == "renode",
            stop_on_output=(
                _renode_stop_on_output(
                    mailbox=mailbox, trace_active=coverage_trace is not None,
                )
                if target["kind"] == "renode" else None
            ),
            stop_on_output_stdin_text=(
                "quit\n" if target["kind"] == "renode"
                and target_coverage_enabled and coverage_trace is None
                else None
            ),
            trace_watch_signal=(
                signal.SIGTERM if rax_stop_flush_enabled else signal.SIGINT
            ),
            graceful_shutdown_seconds=(
                2.0 if target["kind"] == "renode" and target_coverage_enabled
                else 5.0 if rax_stop_flush_enabled else 1.0
            ),
        )
        run_status = result.get("status")
        run_timeout = float(target_timeouts.get(str(target["id"]), timeout))
        run_remaining = (
            None if run_deadline is None
            else max(0.0, run_deadline - time.monotonic())
        )
        side_is_eligible = run_status in {"passed", "failed"} or (
            coverage_side_on_timeout and run_status == "timeout"
        )
        if coverage_side_command_line is not None and side_is_eligible:
            side_timeout = run_timeout if run_remaining is None else min(
                run_timeout, run_remaining,
            )
            if side_timeout <= 0:
                result["coverage_side"] = {
                    "status": "gap", "reason": "run-wall-clock-exhausted",
                    "attempted": False, "report_observed": False,
                }
            else:
                side_needs_loop_guard = (
                    run_status == "timeout"
                    or result.get("reason") == "guest-pc-outside-elf-loop"
                    or result.get("termination") == "guest-invalid-state"
                )
                side_watch_enabled = (
                    side_needs_loop_guard
                    and coverage_side_watch_command_line is not None
                )
                side_command_line = (
                    coverage_side_watch_command_line
                    if side_watch_enabled
                    else coverage_side_command_line
                )
                side_trace_path = (
                    coverage_side_trace
                    if side_watch_enabled
                    else None
                )
                side_result = _run_logged(
                    out, f"{target['id']}-coverage-side", side_command_line,
                    side_timeout, cwd=coverage_side_cwd, env=coverage_side_env,
                    stdin_text="", timeout_stdin_text="quit\n",
                    stop_on_output=_renode_stop_on_output(
                        mailbox=mailbox, trace_active=side_trace_path is not None,
                    ),
                    stop_on_output_stdin_text=(
                        "quit\n" if side_trace_path is None else None
                    ),
                    graceful_shutdown_seconds=30.0,
                    graceful_shutdown_fallback_signal=signal.SIGINT,
                    graceful_shutdown_fallback_after_seconds=1.0,
                    timeout_graceful_shutdown_seconds=30.0,
                    trace_path=side_trace_path,
                    trace_watch_pcs=trace_watch_pcs if side_trace_path else None,
                    trace_watch_plain_pcs=side_trace_path is not None,
                )
                result["coverage_side"] = {
                    "status": side_result.get("status"),
                    "reason": side_result.get("reason"),
                    "elapsed_s": side_result.get("elapsed_s"),
                    "attempted": side_result.get("process_started") is True,
                    "report_observed": bool(coverage_output and coverage_output.is_file()),
                    "trace_path": side_result.get("trace_path"),
                }
        if target.get("kind") == "renode" and coverage_output \
                and coverage_output.is_file():
            try:
                compress_dotnet_report(coverage_output)
            except OSError as exc:
                result["source_coverage_compression_error"] = type(exc).__name__
        if coverage_trace:
            result["trace_path"] = str(coverage_trace.relative_to(out))
        if target["kind"] == "renode":
            result = _renode_result(result)
            if (
                target_coverage_enabled
                and result.get("status") in {"passed", "failed"}
                and result.get("observer_complete") is True
                and result.get("terminal_observed") is True
                and result.get("termination") in {
                    "guest-tohost", "guest-terminal-after-timeout",
                }
                and result.get("trace_size_bytes", 0) > 0
                and result.get("trace_cleanup_status") == "clean"
                and result.get("termination") != "trace-limit"
                and result.get("trace_limit_exceeded") is not True
            ):
                # dotnet-coverage may outlive a guest that already emitted a
                # complete RVOBS1 frame.  The collector timeout is a host
                # artifact flush issue, not an incomplete guest trace.
                result.update(
                    trace_complete=True,
                    trace_truncated=False,
                    trace_reason=result.get("trace_terminal_close_reason"),
                )
            if mailbox is None and result.get("terminal_observed") is True \
                    and result.get("observer_complete") is not True:
                result["observation_gap"] = "Renode observation mailbox symbol is missing"
        elif target["kind"] == "rax":
            result = _rax_result(result)
        if target["kind"] == "qemu":
            result = _qemu_expected_trap_evidence(
                target, result, expected_outcome, user_elf, out,
            )
        result = _expected_outcome_result(
            _semanticize_result(target, result), expected_outcome,
        )
        result = _censor_shared_run_deadline(result, deadline_limited)
        record = _target_record(
            target, result, identity, target_reference, feature_ref(stratum),
        )
        if target.get("adapter_only"):
            if result["status"] == "passed":
                record["status"] = "adapter-probe"
                record["adapter_probe"] = True
            elif result["status"] == "timeout" or result.get("returncode") in {124, 137, -9}:
                record["status"] = "adapter-probe-timeout"
                record["adapter_probe"] = True
                record["reason_code"] = "adapter-probe-timeout"
        results.append(record)
        (out / "targets").mkdir(parents=True, exist_ok=True)
    for row in results:
        target = next((item for item in targets if item.get("id") == row.get("id")), {})
        stratum, _ = _reference_for_target(config, target) if target else (None, {})
        feature = artifact_features.get(stratum or "") or {}
        row["isa_profile"] = feature.get("isa_profile") or bare_isa
        row["privilege_mode"] = feature.get("privilege_mode") or (
            "machine" if stratum == "bare-metal" else "user"
        )
        row["target_timeout_seconds"] = target_timeouts.get(str(row.get("id")), timeout)
        row["global_deadline_enforced"] = run_deadline is not None
        profile_id = coverage_profile_ids.get(str(row.get("id")))
        if profile_id and row.get("target_attempted") is True:
            row["source_coverage_profile_id"] = profile_id
        reference = reference_records.get(stratum, {})
        row.setdefault("backend", row.get("id"))
        row.setdefault("stratum", stratum)
        row.setdefault("reference_id", reference.get("id"))
        row.setdefault("reference_identity", reference.get("identity"))
        row.setdefault("target_identity", row.get("identity"))
        artifact = user_elf if stratum == "linux-user" else (
            rvvm_elf if target.get("kind") == "rvvm" else bare_elf)
        row["artifact_sha256"] = sha256(artifact) if artifact and artifact.is_file() else None
        execution_binary = execution_binaries.get(str(row.get("id")))
        row["execution_binary"] = (
            str(execution_binary) if execution_binary and execution_binary.is_file() else None
        )
        coverage_binary_path = dep(target.get("coverage_binary")) if coverage_enabled else None
        coverage_runtime_enabled = bool(
            coverage_enabled
            and target.get("coverage_binary")
            and coverage_binary_path
            and row["execution_binary"] == str(coverage_binary_path)
        )
        coverage_gap = None
        if coverage_enabled and not coverage_runtime_enabled:
            if not target.get("coverage_binary"):
                coverage_gap = "coverage-binary-not-explicit"
            elif not coverage_binary_path or not coverage_binary_path.is_file():
                coverage_gap = "coverage-binary-missing"
            elif not isinstance(expected_coverage_sha, str):
                coverage_gap = "coverage-binary-sha-not-declared"
            elif not coverage_binary_sha_matches:
                coverage_gap = "coverage-binary-sha-mismatch"
            else:
                coverage_gap = "coverage-runtime-unavailable"
                if target.get("kind") == "qemu":
                    plugin = dep((target.get("simulator_coverage") or {}).get("plugin"))
                    if not plugin or not plugin.is_file():
                        coverage_gap = "coverage-plugin-missing"
                if target.get("kind") == "rvvm":
                    spec = target.get("simulator_coverage") or {}
                    library = dep(spec.get("library"))
                    expected = spec.get("library_sha256")
                    if expected and (not library or sha256(library) != str(expected).lower()):
                        coverage_gap = "coverage-library-sha-mismatch"
        row["coverage_runtime_enabled"] = coverage_runtime_enabled
        row["coverage_gap"] = coverage_gap
        row["execution_binary_kind"] = "coverage" if coverage_runtime_enabled else "target"
        row["differential"] = _differential_record(reference, row)
        if coverage_enabled and coverage_gap:
            row["simulator_coverage"] = {
                "schema_version": SCHEMA,
                "status": "gap", "reason": coverage_gap,
            }
            feature = artifact_features.get(stratum)
            row["simulator_coverage"]["rv_opcode_catalog_coverage"] = (
                opcode_catalog_case_coverage(feature, row["simulator_coverage"])
            )
            row["coverage_status"] = "gap"
            row["simulator_source_coverage"] = {
                "schema_version": SOURCE_SCHEMA,
                "metric_family": "SimSrcCov", "status": "gap",
                "reason": coverage_gap,
            }
        elif coverage_enabled:
            if collect_source:
                feature = artifact_features.get(stratum)
                row["simulator_coverage"] = summarize_simulator_coverage(
                    target, row, out, feature)
                source_binary = dep(target.get("coverage_binary"))
                collection = collect_source_coverage_batch(
                    out, target, source_binary,
                    coverage_profile_dirs.get(
                        str(target.get("id")), coverage_root / str(target.get("id")),
                    ),
                )
                row["source_coverage_collection"] = collection
                row["coverage_binary"] = target.get("coverage_binary")
                row["source_commit_observed"] = collection.get("source_commit_observed")
                row["source_commit_expected"] = collection.get("source_commit_expected")
                row["source_commit_match"] = collection.get("source_commit_match")
                row["coverage_binary_sha"] = collection.get("coverage_binary_sha")
                row["coverage_binary_expected_sha"] = collection.get("coverage_binary_expected_sha")
                row["coverage_binary_sha_match"] = collection.get("coverage_binary_sha_match")
                row["coverage_binary_sha256"] = collection.get("coverage_binary_sha256")
                row["coverage_binary_expected_sha256"] = collection.get("coverage_binary_expected_sha256")
                row["coverage_binary_sha256_match"] = collection.get("coverage_binary_sha256_match")
                row["coverage_binary_source_commit"] = collection.get("coverage_binary_source_commit")
                row["coverage_binary_expected_source_commit"] = collection.get("coverage_binary_expected_source_commit")
                source_result = target_source_coverage(
                    out, target, row.get("target_identity"),
                    coverage_binary_sha=row.get("coverage_binary_sha"),
                    source_commit_observed=row.get("source_commit_observed"),
                    coverage_binary_sha256=row.get("coverage_binary_sha256"),
                    collection_status=collection.get("status"),
                    collection_reason=collection.get("reason"))
                if collection.get("warnings"):
                    source_result["warnings"] = collection["warnings"]
                row["simulator_source_coverage"] = source_result
            else:
                row["simulator_coverage"] = None
                row["simulator_coverage_deferred"] = True
                row["coverage_status"] = "deferred"
                row["simulator_source_coverage"] = {
                    "schema_version": SOURCE_SCHEMA,
                    "metric_family": "SimSrcCov", "status": "deferred",
                    "reason": "run-level-source-coverage-collection"
                }
        else:
            row["simulator_coverage"] = None
            row["coverage_status"] = "disabled"

    target_result_timeout = (
        next(iter(target_timeouts.values())) if len(target_timeouts) == 1 else timeout
    )
    target_result = {
        "target_timeout_seconds": target_result_timeout,
        "target_timeout_seconds_by_target": target_timeouts,
        "schema_version": "rq1-comparison-target-run-v1",
        "provenance": provenance or _provenance(config),
        "status": _target_run_status(config, targets, results),
        "reference": reference_record,
        "references": reference_records, "targets": results,
    }
    if coverage_enabled and collect_source:
        summary = aggregate_simulator_coverage(
            results, include_guest_metrics=include_guest_metrics,
        )
        write_json(out / "coverage" / "summary.json", summary)
        target_result["coverage"] = summary
        target_result["coverage_timing"] = {
            "target_execution_seconds": round(sum(
                float(row.get("elapsed_s") or 0) for row in results
            ), 6),
            "source_collection_seconds": round(sum(
                float((row.get("source_coverage_collection") or {}).get("elapsed_s") or 0)
                for row in results
            ), 6),
            "collection_deferred": not collect_source,
        }
    elif coverage_enabled:
        target_result["coverage"] = {
            "schema_version": SCHEMA,
            "status": "deferred", "reason": "run-level-coverage-finalization",
        }
        target_result["coverage_timing"] = {
            "target_execution_seconds": round(sum(
                float(row.get("elapsed_s") or 0) for row in results
            ), 6),
            "collection_deferred": True,
            "guest_collection_deferred": True,
        }
        target_result["coverage_status"] = "deferred"
    else:
        target_result["coverage"] = {
            "schema_version": BATCH_SCHEMA,
            "status": "disabled", "reason": "coverage-disabled",
            "method_filter": provenance.get("method_filter") if provenance else None,
            "experiment_unions": [], "targets": [], "source_targets": [],
        }
        target_result["coverage_status"] = "disabled"
    return target_result


def _generation_continuation(method_filter: str) -> dict[str, Any]:
    """Read an append-only generation cursor supplied by the host wrapper."""
    def nonnegative(name: str) -> int:
        raw = os.environ.get(name, "")
        if raw == "":
            return 0
        if not raw.isdigit():
            raise ValueError(f"{name} must be a non-negative integer")
        return int(raw)

    start_index = nonnegative("RQ1_GENERATION_START_INDEX")
    start_batch = nonnegative("RQ1_GENERATION_START_BATCH")
    attempt_offset = nonnegative("RQ1_GENERATION_ATTEMPT_OFFSET")
    candidate_index_offset = nonnegative("RQ1_GENERATION_CANDIDATE_OFFSET")
    parent_run_id = os.environ.get("RQ1_GENERATION_PARENT_RUN_ID") or None
    parent_batch_id = os.environ.get("RQ1_GENERATION_PARENT_BATCH_ID") or None
    parent_source_sha256 = os.environ.get("RQ1_GENERATION_PARENT_SOURCE_SHA256") or None
    if start_batch and method_filter != "B-RVDV":
        raise ValueError("RQ1_GENERATION_START_BATCH only applies to B-RVDV")
    if start_index and method_filter not in {"B-TORTURE", "B-CSMITH"}:
        raise ValueError("RQ1_GENERATION_START_INDEX only applies to B-TORTURE/B-CSMITH")
    if attempt_offset and method_filter != "B-GEMI":
        raise ValueError("RQ1_GENERATION_ATTEMPT_OFFSET only applies to B-GEMI")
    enabled = any((start_index, start_batch, attempt_offset, candidate_index_offset,
                   parent_run_id, parent_batch_id, parent_source_sha256))
    return {
        "schema_version": "rq1-generation-continuation-v1",
        "enabled": enabled,
        "method": method_filter,
        "parent_run_id": parent_run_id,
        "parent_batch_id": parent_batch_id,
        "parent_source_sha256": parent_source_sha256,
        "start_index": start_index,
        "start_batch": start_batch,
        "attempt_offset": attempt_offset,
        "candidate_index_offset": candidate_index_offset,
    }


def run_tools(config: dict[str, Any], out: Path, *,
               provenance: dict[str, Any] | None = None, seed: int = 303,
               timeout_override: int | None = None,
               method_filter: str | None = None,
               target_timeout_override: int | None = None,
               generation_only: bool = False,
               on_candidate: Any = None,
               run_deadline: float | None = None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("comparison tool runner requires an object config")
    if not isinstance(method_filter, str) or not method_filter:
        raise ValueError("comparison tool runner requires one --method")
    continuation = _generation_continuation(method_filter)
    start_index = continuation["start_index"]
    start_batch = continuation["start_batch"]
    attempt_offset = continuation["attempt_offset"]
    candidate_index_offset = continuation["candidate_index_offset"]
    tools_dir = out / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    def append(row: dict[str, Any], started: float | None = None) -> None:
        started = method_started if started is None else started
        declared = next(
            (item for item in config.get("methods", ())
             if isinstance(item, dict) and item.get("id") == row.get("method")),
            {},
        )
        row.setdefault("route", declared.get("route"))
        row.setdefault("lane", declared.get("lane"))
        row.setdefault("seed", seed)
        row.setdefault("generation_continuation", continuation)
        row.setdefault("phase", "generation-only" if generation_only else "generation+execution")
        row.setdefault("generation_mode", "continuous-until-experiment-end")
        row.setdefault("generation_attempted", bool(row.get("generator_invoked") or row.get("generated")))
        applicability = config.get("applicability")
        scope = applicability.get(row.get("method"), {}) \
            if isinstance(applicability, dict) else {}
        row.setdefault(
            "applicable_strata",
            scope.get("strata", ["bare-metal", "linux-user"])
            if isinstance(scope, dict) else ["bare-metal", "linux-user"],
        )
        row["method_elapsed_s"] = round(time.monotonic() - started, 6)
        if generation_pause_gate is not None:
            row["active_execution_seconds"] = round(max(
                0.0, row["method_elapsed_s"] - generation_pause_gate.paused_seconds,
            ), 6)
            row["paused_seconds"] = round(generation_pause_gate.paused_seconds, 6)
            row["segment_count"] = generation_pause_gate.segment_index
        rows.append(row)
        if generation_pause_gate is not None:
            finish_generation_control()
    methods = [
        method for method in config.get("methods", [])
        if isinstance(method, dict)
        and method.get("id") == method_filter
    ]
    framework_ids = {item[0] for item in FRAMEWORK_METHODS}
    if method_filter in framework_ids:
        raise ValueError("Ours methods must use framework-run, not the comparison queue pipeline")
    # The outer decoupled campaign owns pause/resume when it supplies a
    # deadline.  Direct unit/diagnostic callers with an explicit timeout must
    # finish at that deadline instead of entering an unowned pause gate.
    segment_control_enabled = (
        generation_only and timeout_override is None and run_deadline is None
    )
    deadline = None if segment_control_enabled else (
        float(run_deadline) if run_deadline is not None else
        time.monotonic() + int(timeout_override)
        if timeout_override is not None else None
    )
    default_target_timeout = _configured_target_timeout(config, target_timeout_override)
    configured_method_shape = [
        (method.get("id"), method.get("route"))
        for method in config.get("methods", []) if isinstance(method, dict)
    ] == list(ACTIVE_METHODS)
    selected_method_shape = (
        configured_method_shape
        and
        len(methods) == 1
        and methods[0].get("id") == method_filter
    )
    if not selected_method_shape:
        raise ValueError("comparison pipeline requires exactly one declared external method")
    deps = config.get("dependencies", {})
    timeout = DEFAULT_RUN_SECONDS
    generation_pause_gate = None
    generation_pause_control_root: Path | None = None
    generation_pause_clock_path: Path | None = None
    generation_pause_segment_index = 1
    generation_pause_last_sequence = 0
    generation_pause_method_started: float | None = None
    generation_pause_cursor: dict[str, Any] = {
        "phase": "generation", "case_index": -1,
        "last_completed_case_index": None,
        "last_completed_case_id": None,
        "active_execution_seconds": 0.0,
        "segment_index": 1,
    }

    def control_root_for_tools(path: Path) -> Path:
        for ancestor in (path, *path.parents):
            if ancestor.name == "run-tools":
                return ancestor.parent
        return path

    def start_generation_control(method_id: str, segment_seconds: int,
                                 method_started_at: float) -> None:
        nonlocal generation_pause_gate, generation_pause_control_root
        nonlocal generation_pause_clock_path, generation_pause_last_sequence
        nonlocal generation_pause_method_started
        if not segment_control_enabled:
            return
        from decoupled_pipeline import _CaseBoundaryPause

        generation_pause_control_root = control_root_for_tools(out)
        control_dir = generation_pause_control_root / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        execution = SimpleNamespace(out=generation_pause_control_root, deadline=math.inf)
        generation_pause_gate = _CaseBoundaryPause(
            execution, method_filter=method_id, participants={"lane-00"},
            segment_duration_seconds=max(1, int(segment_seconds)),
        )
        generation_pause_last_sequence = generation_pause_gate._request()[0]
        generation_pause_method_started = method_started_at
        generation_pause_clock_path = control_dir / "generation-segment.json"
        publish_generation_clock(method_started_at)
        generation_case_boundary(None, None)

    def publish_generation_clock(method_started_at: float) -> None:
        if generation_pause_gate is None or generation_pause_clock_path is None:
            return
        sequence, desired, _ = generation_pause_gate._request()
        write_json_replace(generation_pause_clock_path, {
            "schema_version": "rq1-generation-segment-clock-v1",
            "method_filter": method_filter,
            "state": "running" if desired == "running" else "pausing",
            "request_sequence": sequence,
            "segment_index": generation_pause_segment_index,
            "segment_started_monotonic": generation_pause_gate.segment_started,
            "segment_duration_seconds": generation_pause_gate.segment_duration_seconds,
            "segment_deadline_monotonic": (
                generation_pause_gate.segment_deadline
                if hasattr(generation_pause_gate, "segment_deadline") else
                generation_pause_gate.segment_started
                + generation_pause_gate.segment_duration_seconds
            ),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    def generation_case_boundary(case_id: str | None, case_index: int | None) -> None:
        nonlocal generation_pause_segment_index, generation_pause_last_sequence
        if generation_pause_gate is None:
            return
        now = time.monotonic()
        started = generation_pause_method_started or now
        active = max(0.0, now - started - generation_pause_gate.paused_seconds)
        if generation_pause_gate.pause_started is not None:
            active = max(0.0, active - (now - generation_pause_gate.pause_started))
        cursor = {
            "phase": "generation", "case_index": case_index,
            "last_completed_case_index": case_index,
            "last_completed_case_id": case_id,
            "next_case_index": None if case_index is None else case_index + 1,
            "next_case_id": None if case_id is None else f"{method_filter}-case-{case_index + 1:08d}",
            "active_execution_seconds": round(active, 6),
            "segment_index": generation_pause_segment_index,
        }
        checkpoint_path = out / "tools" / "generation-progress.json"
        write_json_replace(checkpoint_path, {
            "schema_version": "rq1-generation-progress-v1",
            "method": method_filter, "lane": "lane-00", **cursor,
            "paused_seconds": round(generation_pause_gate.paused_seconds, 6),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        generation_pause_cursor.update(cursor)
        generation_pause_gate.checkpoint("lane-00", dict(generation_pause_cursor))
        if stop_requested():
            return
        after_sequence = generation_pause_gate._request()[0]
        generation_pause_last_sequence = max(
            generation_pause_last_sequence, after_sequence,
        )
        generation_pause_segment_index = generation_pause_gate.segment_index
        generation_pause_cursor["segment_index"] = generation_pause_segment_index
        publish_generation_clock(now)
        generation_pause_gate.cursors["lane-00"] = dict(generation_pause_cursor)
        generation_pause_gate._write_status("running")

    def finish_generation_control() -> None:
        if generation_pause_gate is None:
            return
        generation_pause_gate.finish("lane-00", dict(generation_pause_cursor))
        generation_pause_gate._write_status("completed")
        if generation_pause_clock_path is not None:
            sequence, _, _ = generation_pause_gate._request()
            write_json_replace(generation_pause_clock_path, {
                "schema_version": "rq1-generation-segment-clock-v1",
                "method_filter": method_filter, "state": "completed",
                "request_sequence": sequence,
                "segment_index": generation_pause_segment_index,
                "segment_started_monotonic": generation_pause_gate.segment_started,
                "segment_duration_seconds": generation_pause_gate.segment_duration_seconds,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })

    for method in methods:
        method_started = time.monotonic()
        method_id = method["id"]
        if deadline is not None and time.monotonic() >= deadline:
            append(_tool_record(method_id, "gap", "campaign-time-exhausted",
                                     generator_invoked=False, generated=False,
                                     built=False, candidate_count=0))
            continue
        required = method.get("requires", [])
        missing = [str(dep(deps.get(name)) or name) for name in required
                   if not _dependency_present(config, name)]
        if missing:
            append(_tool_record(method_id, "gap", "dependency-missing", missing=missing))
            continue
        method_timeout = int(method.get("timeout_seconds", timeout))
        if segment_control_enabled and timeout_override is not None:
            method_timeout = max(1, int(timeout_override))
        if deadline is not None:
            # The absolute campaign deadline includes orchestration setup and
            # cannot be restarted as a fresh per-method timeout.
            remaining = max(0, math.floor(deadline - time.monotonic()))
            method_timeout = min(
                method_timeout if timeout_override is None else int(timeout_override),
                remaining,
            )
        if method_timeout <= 0:
            append(_tool_record(method_id, "gap", "method-budget-exhausted",
                                generator_invoked=False, generated=False,
                                built=False, candidate_count=0), method_started)
            continue
        start_generation_control(method_id, method_timeout, method_started)
        if segment_control_enabled:
            method_deadline = math.inf
            tool_timeout = None
        else:
            method_deadline = method_started + method_timeout
            tool_timeout = method_timeout
            if deadline is not None:
                method_deadline = min(method_deadline, deadline)
        if method_id == "Fuzz4All":
            adapter = HERE / method.get("adapter", "adapters/fuzz4all_adapter.py")
            fuzz_root = dep(deps.get("fuzz4all"))
            adapter_available = bool(fuzz_root and fuzz_root.is_dir() and adapter.is_file())
            if not adapter_available:
                append(_tool_record(
                    method_id, "gap", "fuzz4all-adapter-or-checkout-missing",
                    generator_invoked=False, generated=False, built=False,
                    candidate_count=0, network=method.get("network", "bridge"),
                    model=method.get("model"),
                ))
                continue
            fuzz_env = {**os.environ,
                        "RQ1_RUN": str(out),
                        "RQ1_CONFIG_SNAPSHOT": str(out.parents[2] / "config.snapshot.json"),
                        "RQ1_FUZZ4ALL_ROOT": str(fuzz_root),
                        "OLLAMA_HOST": (
                            os.environ.get("OLLAMA_HOST")
                            or method.get("model_endpoint")
                            or "http://133.133.135.123:11434"
                        ),
                        "RQ1_NETWORK": method.get("network", "bridge"),
                        "RQ1_FUZZ4ALL_TARGET_TIMEOUT": str(default_target_timeout),
                        "RQ1_FUZZ4ALL_GENERATION_ONLY": "1" if generation_only else "0",
                        "RQ1_FUZZ4ALL_CONTROLLED_GENERATION": (
                            "1" if segment_control_enabled else "0"
                        ),
                        "PYTHONPATH": os.pathsep.join(filter(None,
                            (str(fuzz_root), os.environ.get("PYTHONPATH"))))}
            if tool_timeout is not None:
                fuzz_env["RQ1_FUZZ4ALL_TIMEOUT"] = str(tool_timeout)
            else:
                fuzz_env.pop("RQ1_FUZZ4ALL_TIMEOUT", None)
            control_dir = (
                (generation_pause_control_root or control_root_for_tools(out)) / "control"
                if generation_only else None
            )
            ack_path = control_dir / "fuzz4all-ack.json" if control_dir else None
            remote_status_path = (
                control_dir / "fuzz4all-remote-status.json" if control_dir else None
            )
            generated_dir = out / "fuzz4all" / "generated"
            if control_dir:
                fuzz_env.update({
                    "RQ1_CONTROL_DIR": str(control_dir),
                    "RQ1_FUZZ4ALL_ACK_PATH": str(ack_path),
                    "RQ1_FUZZ4ALL_REMOTE_STATUS_PATH": str(remote_status_path),
                    "RQ1_FUZZ4ALL_GENERATED_DIR": str(generated_dir),
                })
                if generation_pause_clock_path is not None:
                    fuzz_env["RQ1_GENERATION_CLOCK"] = str(generation_pause_clock_path)
            result_path = out / "fuzz4all" / "fuzz4all-result.json"
            partial_result_path = out / "fuzz4all" / "fuzz4all-result.partial.json"
            acknowledged_fuzz_indices: set[int] = set()
            fuzz_ack_count = 0

            def normalized_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
                if not isinstance(payload, dict):
                    return []
                normalized = []
                raw_candidates = payload.get("candidate_results", [])
                if not isinstance(raw_candidates, (list, tuple)):
                    return []
                for candidate in raw_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    item = dict(candidate)
                    for field in ("source", "elf", "linux_elf", "target_result"):
                        value = item.get(field)
                        if isinstance(value, str):
                            item[field] = str(Path("fuzz4all") / value)
                    normalized.append(item)
                return normalized

            def continuation_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
                item = dict(candidate)
                index = item.get("index")
                if type(index) is not int or index < 0:
                    index = 0
                item["index"] = candidate_index_offset + index
                return item

            def emit_fuzz_candidates(payload: dict[str, Any]) -> None:
                nonlocal fuzz_ack_count
                candidates = normalized_candidates(payload)
                candidates.sort(key=lambda item: (
                    item.get("index") if type(item.get("index")) is int else math.inf
                ))
                for fallback_index, candidate in enumerate(candidates):
                    index = candidate.get("index")
                    if type(index) is not int or index < 0:
                        index = fallback_index
                    if index in acknowledged_fuzz_indices:
                        continue
                    usable = (
                        candidate.get("usable_artifact") is True
                        or candidate.get("complete_chain") is True
                    )
                    if usable and on_candidate is not None:
                        on_candidate(method_id, continuation_candidate(candidate))
                    acknowledged_fuzz_indices.add(index)
                    fuzz_ack_count = max(fuzz_ack_count, index + 1)
                    if ack_path is not None:
                        write_json_replace(ack_path, {
                            "schema_version": "rq1-generation-candidate-ack-v1",
                            "method": method_id,
                            "acknowledged_count": fuzz_ack_count,
                            "last_acknowledged_index": fuzz_ack_count - 1,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        })

            def pause_at_fuzz_batch_boundary() -> None:
                if not segment_control_enabled or remote_status_path is None:
                    return
                try:
                    remote_status = load_json(remote_status_path)
                except (OSError, TypeError, ValueError):
                    return
                if not isinstance(remote_status, dict) or remote_status.get("state") not in {
                    "waiting-for-ack", "waiting-for-control",
                }:
                    return
                discovered = remote_status.get("candidate_count_discovered")
                if type(discovered) is not int or discovered > fuzz_ack_count:
                    return
                triggered, _, _ = generation_pause_gate._pause_trigger()
                if triggered:
                    generation_case_boundary(
                        f"Fuzz4All-case-{fuzz_ack_count - 1:08d}"
                        if fuzz_ack_count else None,
                        fuzz_ack_count - 1 if fuzz_ack_count else None,
                    )

            invocation_box: dict[str, dict[str, Any]] = {}

            def invoke_fuzz4all() -> None:
                invocation_box["result"] = _run_logged(
                    out, "tool-Fuzz4All-adapter", [sys.executable, str(adapter)],
                    tool_timeout, env=fuzz_env)

            invocation_thread = threading.Thread(
                target=invoke_fuzz4all, name="rq1-fuzz4all-adapter",
                daemon=True,
            )
            invocation_thread.start()
            while invocation_thread.is_alive():
                try:
                    partial = load_json(partial_result_path)
                except (OSError, TypeError, ValueError):
                    partial = {}
                emit_fuzz_candidates(partial)
                pause_at_fuzz_batch_boundary()
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            # _run_logged enforces the process deadline. Do not return to the
            # caller while its worker can still write run artifacts.
            invocation_thread.join()
            invocation = invocation_box.get("result", {
                "status": "gap", "reason": "fuzz4all-adapter-no-result",
            })
            if not isinstance(invocation, dict):
                invocation = {
                    "status": "gap",
                    "reason": "fuzz4all-adapter-result-invalid",
                }
            fuzz = {}
            for candidate_path in (result_path, partial_result_path):
                if not candidate_path.is_file():
                    continue
                try:
                    payload = load_json(candidate_path)
                except (OSError, TypeError, ValueError):
                    continue
                if isinstance(payload, dict):
                    fuzz = payload
                    break
            emit_fuzz_candidates(fuzz)
            pause_at_fuzz_batch_boundary()
            candidates = [continuation_candidate(item) for item in normalized_candidates(fuzz)]
            complete = [item for item in candidates if item.get("usable_artifact") is True
                        or item.get("complete_chain") is True]
            def nonnegative_count(value: object) -> int | None:
                return value if type(value) is int and value >= 0 else None

            discovered = nonnegative_count(fuzz.get("candidate_count_discovered"))
            tested = nonnegative_count(fuzz.get("candidate_count_tested"))
            if tested is None:
                tested = len(candidates)
            pending = nonnegative_count(fuzz.get("candidate_count_pending"))
            if pending is None and discovered is not None:
                pending = max(0, discovered - tested)
            complete_count = nonnegative_count(fuzz.get("complete_candidate_count"))
            if complete_count is None:
                complete_count = sum(
                    1 for item in candidates
                    if item.get("complete_chain") is True
                )
            duplicate_sources = [
                item for item in candidates
                if item.get("failure_reason") == "duplicate-source"
            ]
            candidate_failures = [
                item for item in candidates
                if not item.get("usable_artifact") and not item.get("complete_chain")
                and item.get("failure_reason") != "duplicate-source"
                and item.get("deadline_censored") is not True
            ]
            deadline_rejected_candidates = sum(
                item.get("deadline_censored") is True for item in candidates
            )
            deadline_censored = (
                fuzz.get("deadline_censored") is True
                or _shared_deadline_censored(invocation, method_deadline)
            )
            interrupted = (
                stop_requested()
                or invocation.get("reason") == "external-signal"
                or fuzz.get("reason_code") == "external-signal"
            )
            first = complete[0] if complete else (candidates[0] if candidates else {})
            ok = (invocation.get("status") == "passed"
                  and fuzz.get("status") == "passed")
            lane = str(method.get("lane", "rv64imc/lp64"))
            lane_parts = lane.split("/", 1)
            generator_profile = method.get("generator_profile")
            generator_profile = generator_profile if isinstance(generator_profile, dict) else {}
            isa = str(generator_profile.get("isa") or lane_parts[0])
            mabi = str(generator_profile.get("mabi") or (
                lane_parts[1] if len(lane_parts) > 1 else "lp64"
            ))
            append(_tool_record(
                method_id,
                "passed" if ok else "partial" if complete or deadline_censored or interrupted else "gap",
                None if ok else "generation-deadline-reached" if deadline_censored else
                stop_reason() if interrupted else
                str(fuzz.get("reason_code") or invocation.get("reason")
                    or "fuzz4all-candidate-chain-gap"),
                generator_invoked=True, generated=bool(candidates), built=bool(complete),
                candidate_stream_invoked=True, pq_chain_complete=ok,
                candidate_artifacts=complete, candidate_count=len(complete),
                attempt_count=tested,
                request_count=fuzz.get("request_count"),
                generation_call_count=fuzz.get("generation_call_count"),
                model_response_duplicate_count=fuzz.get("model_response_duplicate_count"),
                duplicate_retry_count=fuzz.get("duplicate_retry_count"),
                model_response_truncated_count=fuzz.get("model_response_truncated_count"),
                truncation_retry_count=fuzz.get("truncation_retry_count"),
                source_built_count=fuzz.get("source_built_count", sum(
                    item.get("source_built") is True for item in candidates
                    if isinstance(item, dict)
                )),
                candidate_build_attempt_count=fuzz.get(
                    "candidate_build_attempt_count", sum(
                        item.get("build_attempted") is True for item in candidates
                        if isinstance(item, dict)
                    )),
                pair_built_count=sum(
                    1 for item in complete
                    if item.get("elf") or item.get("linux_elf")
                ), accepted_count=len(complete),
                candidate_failure_count=fuzz.get(
                    "candidate_failure_count", len(candidate_failures)),
                artifact_gap_count=fuzz.get(
                    "artifact_gap_count", sum(
                        item.get("usable_artifact") is True
                        and item.get("complete_chain") is not True
                        for item in candidates if isinstance(item, dict)
                    )),
                candidate_failures=fuzz.get("candidate_failures", candidate_failures),
                deadline_rejected_candidate_count=fuzz.get(
                    "deadline_rejected_candidate_count", deadline_rejected_candidates),
                duplicate_source_count=fuzz.get(
                    "duplicate_source_count", len(duplicate_sources)),
                candidate_count_discovered=discovered,
                candidate_count_tested=tested,
                candidate_count_pending=pending,
                complete_candidate_count=complete_count,
                source=first.get("source"), source_sha256=first.get("source_sha256"),
                elf=first.get("elf"), elf_sha256=first.get("elf_sha256"),
                linux_elf=first.get("linux_elf"), linux_elf_sha256=first.get("linux_elf_sha256"),
                artifact_isa=isa, linux_artifact_isa=isa, artifact_mabi=mabi,
                network=method.get("network", "bridge"), model=method.get("model"),
                request_mode=fuzz.get("request_mode"), request_model=fuzz.get("model"),
                model_endpoint=fuzz.get("model_endpoint"),
                generation_request_trace=fuzz.get("generation_request_trace"),
                generation_partial=not ok,
                adapter_result=(str(result_path.relative_to(out))
                                if result_path.is_relative_to(out) else str(result_path))
                if result_path.is_file() else None,
                generation=fuzz.get("generation"), elapsed_s=fuzz.get("elapsed_s"),
                deadline_censored=deadline_censored,
            ))
            continue
        if method_id == "B-RVDV":
            profile = method.get("generator_profile") if isinstance(method.get("generator_profile"), dict) else {}
            batch_iterations = max(1, int(profile.get("iterations", 1)))
            incremental_batches = bool(profile.get("incremental_batches", False))
            trial_iterations = min(batch_iterations, max(1, int(profile.get("batch_size", batch_iterations)))) \
                if incremental_batches else batch_iterations
            phase_deadline = method_deadline
            trials, candidates = [], []
            batch = start_batch
            while not stop_requested() and time.monotonic() < phase_deadline:
                generation_case_boundary(
                    f"B-RVDV-trial-{batch - 1:08d}" if batch else None,
                    batch - 1 if batch else None,
                )
                remaining = (
                    None if math.isinf(phase_deadline)
                    else max(0, math.floor(phase_deadline - time.monotonic()))
                )
                if remaining is not None and remaining <= 0:
                    break
                trial = _riscv_dv_trial(
                    dep(deps["riscv_dv"]), out,
                    remaining,
                    seed + batch * trial_iterations,
                    f"{method_id}-{batch:04d}",
                    {**profile, "iterations": trial_iterations, "batch_size": trial_iterations},
                    deadline=phase_deadline,
                )
                if not isinstance(trial, dict):
                    trial = _tool_record(
                        method_id, "gap", "riscv-dv-trial-result-invalid",
                        candidate_artifacts=[], candidate_count=0,
                    )
                trials.append(trial)
                batch_candidates = trial.get("candidate_artifacts") or []
                if not isinstance(batch_candidates, (list, tuple)):
                    batch_candidates = []
                batch_candidates = [
                    dict(candidate) for candidate in batch_candidates
                    if isinstance(candidate, dict)
                ]
                candidates.extend(batch_candidates)
                if on_candidate is not None:
                    for candidate in batch_candidates:
                        if isinstance(candidate.get("index"), int):
                            candidate["index"] += (
                                candidate_index_offset
                                + len(candidates) - len(batch_candidates)
                            )
                        on_candidate(method_id, candidate)
                generator = trial.get("generator")
                generator_status = (
                    generator.get("status")
                    if isinstance(generator, dict) else None
                )
                generator_status = generator_status or trial.get("generation_status")
                generator_failed = generator_status == "failed"
                batch += 1
                generation_case_boundary(f"B-RVDV-trial-{batch - 1:08d}", batch - 1)
                if generator_failed:
                    # A non-zero generator exit may leave usable source files.
                    # Keep those candidates, but do not report the batch as a
                    # clean generation pass or retry a failed invocation as if
                    # it had completed normally.
                    break
                if not trial.get("candidate_artifacts"):
                    if isinstance(generator, dict) and generator.get("status") == "timeout":
                        break
                    continue
            for index, item in enumerate(candidates):
                item["index"] = candidate_index_offset + index
            trial_totals = {
                field: sum(_nonnegative_int(trial.get(field)) for trial in trials)
                for field in ("source_generated", "bare_built", "linux_built")
            }
            aggregate = dict(trials[-1] if trials else {})
            generator_timed_out = any(
                isinstance(item.get("generator"), dict)
                and item["generator"].get("status") == "timeout"
                for item in trials
            )
            generator_failed = any(
                item.get("generation_status") == "failed"
                or isinstance(item.get("generator"), dict)
                and item["generator"].get("status") == "failed"
                for item in trials
            )
            budget_exhausted = time.monotonic() >= phase_deadline
            deadline_censored = budget_exhausted or any(
                _shared_deadline_censored(item.get("generator"), phase_deadline)
                for item in trials
            )
            interrupted = stop_requested()
            partial_reason = (
                "generation-deadline-reached" if deadline_censored else
                stop_reason() if interrupted else
                "generator-timeout" if generator_timed_out else
                "generation-failed" if generator_failed else
                "generation-budget-exhausted" if budget_exhausted else
                None if candidates else "riscv-dv-no-candidate"
            )
            aggregate.update(
                **trial_totals,
                attempt_count=sum(_nonnegative_int(trial.get("requested_iterations"))
                                  for trial in trials),
                source_built_count=trial_totals["source_generated"],
                pair_built_count=len(candidates), accepted_count=len(candidates),
                generator_invoked=bool(trials), generated=bool(candidates),
                built=bool(candidates), candidate_artifacts=candidates,
                candidate_count=len(candidates),
                candidate_count_discovered=len(candidates),
                generation_partial=partial_reason is not None,
                generation_partial_reason=partial_reason,
                deadline_censored=deadline_censored,
                generator={"status": "partial" if partial_reason else "passed" if candidates else "gap",
                           "batches": len(trials),
                           "iterations_per_batch": trial_iterations,
                           "requested_iterations": batch_iterations,
                           "batch_size": trial_iterations,
                           "instr_cnt": _nonnegative_int(
                               (profile.get("tests") or [{}])[0].get("instr_cnt", 10000),
                               default=10000,
                           )
                           if isinstance(profile.get("tests"), list) and profile.get("tests") else None,
                           "boot_mode": profile.get("boot_mode", "m"),
                           "candidate_count": len(candidates), **trial_totals},
                artifact_gap_count=max(
                    0, trial_totals["source_generated"] - len(candidates),
                ),
                trials=trials, generator_profile=profile,
            )
            append(_tool_record(
                method_id,
                "passed" if candidates and not aggregate.get("generation_partial") else
                "partial" if candidates or aggregate.get("deadline_censored") or interrupted else "gap",
                None if not aggregate.get("generation_partial") else aggregate.get(
                    "generation_partial_reason", "riscv-dv-generation-or-cross-compile-failed"
                ),
                **aggregate,
            ))
            continue
        if method_id == "B-TORTURE":
            profile = method.get("generator_profile") if isinstance(method.get("generator_profile"), dict) else {}
            phase_deadline = method_deadline
            trials = []
            candidates = []
            while not stop_requested() and time.monotonic() < phase_deadline:
                case_index = start_index + len(trials)
                generation_case_boundary(
                    f"B-TORTURE-case-{case_index - 1:08d}" if case_index else None,
                    case_index - 1 if case_index else None,
                )
                remaining = (
                    None if math.isinf(phase_deadline)
                    else max(0, math.floor(phase_deadline - time.monotonic()))
                )
                if remaining is not None and remaining <= 0:
                    break
                trial = _torture_trial(
                    dep(deps["torture"]), out, remaining, profile,
                    case_index=case_index, deadline=phase_deadline,
                )
                if not isinstance(trial, dict):
                    trial = _tool_record(
                        method_id, "gap", "torture-trial-result-invalid",
                        candidate_artifacts=[], candidate_count=0,
                    )
                trials.append(trial)
                has_artifact = (
                    trial.get("built") and trial.get("elf")
                ) or (
                    trial.get("linux_built") and trial.get("linux_elf")
                )
                if not (trial.get("source") and trial.get("source_sha256") and has_artifact):
                    generation_case_boundary(
                        f"B-TORTURE-case-{case_index:08d}", case_index,
                    )
                    continue
                candidate = {
                    "index": candidate_index_offset + len(candidates),
                    "source": trial.get("source"),
                    "source_sha256": trial.get("source_sha256"),
                    "artifact_isa": trial.get("artifact_isa"),
                    "linux_artifact_isa": trial.get("artifact_isa"),
                }
                if trial.get("built") and trial.get("elf"):
                    candidate.update(
                        elf=trial.get("elf"), elf_sha256=trial.get("elf_sha256")
                    )
                if trial.get("linux_built") and trial.get("linux_elf"):
                    candidate.update(
                        linux_elf=trial.get("linux_elf"),
                        linux_elf_sha256=trial.get("linux_elf_sha256"),
                    )
                candidates.append(candidate)
                if on_candidate is not None:
                    on_candidate(method_id, candidate)
                generation_case_boundary(
                    f"B-TORTURE-case-{case_index:08d}", case_index,
                )
            last = trials[-1] if trials else {}
            last_candidate = candidates[-1] if candidates else {}
            def trial_deadline_censored(trial: Mapping[str, Any]) -> bool:
                return any(
                    _shared_deadline_censored(trial.get(name), phase_deadline)
                    for name in (
                        "generator", "cross_compile", "linux_cross_compile",
                        "cache_extract", "linux_preprocess",
                    )
                )

            interrupted = stop_requested()
            deadline_censored = (
                time.monotonic() >= phase_deadline
                or any(trial_deadline_censored(trial) for trial in trials)
            )
            trial_failed = any(
                not (trial.get("generated") and (
                    trial.get("built") or trial.get("linux_built")
                ))
                and not trial_deadline_censored(trial)
                for trial in trials
            )
            status = (
                "partial" if candidates and (trial_failed or deadline_censored or interrupted)
                else "passed" if candidates
                else "partial" if deadline_censored or interrupted else "gap"
            )
            reason = None if status == "passed" else (
                "generation-deadline-reached" if deadline_censored else
                stop_reason() if interrupted else
                "torture-generation-or-cross-compile-failed"
            )
            append(_tool_record(
                method_id, status, reason,
                generator_invoked=any(item.get("generator_invoked") for item in trials),
                generated=bool(candidates), built=bool(candidates),
                generator={"status": status, "cases_attempted": len(trials),
                           "cases_built": len(candidates)},
                attempt_count=len(trials),
                source_built_count=sum(bool(item.get("source")) for item in trials),
                pair_built_count=len(candidates), accepted_count=len(candidates),
                trials=trials, source=last_candidate.get("source") or last.get("source"),
                source_sha256=last_candidate.get("source_sha256") or last.get("source_sha256"),
                elf=last_candidate.get("elf") or last.get("elf"),
                elf_sha256=last_candidate.get("elf_sha256") or last.get("elf_sha256"),
                linux_elf=last_candidate.get("linux_elf") or last.get("linux_elf"),
                linux_elf_sha256=(last_candidate.get("linux_elf_sha256")
                                  or last.get("linux_elf_sha256")),
                cross_compile=last.get("cross_compile"),
                linux_cross_compile=last.get("linux_cross_compile"),
                cache_extract=last.get("cache_extract"),
                compatibility=last.get("compatibility"),
                artifact_gap_count=max(
                    0,
                    sum(bool(item.get("source")) for item in trials) - len(candidates),
                ),
                artifact_isa=last.get("artifact_isa"), linux_artifact_isa=last.get("artifact_isa"),
                generator_profile=profile,
                candidate_artifacts=candidates, candidate_count=len(candidates),
                candidate_count_discovered=len(candidates),
                generation_partial=status != "passed",
                generation_partial_reason=reason,
                deadline_censored=deadline_censored,
            ))
            continue
        if method_id == "B-CSMITH":
            csmith = dep(deps["csmith"])
            profile = method.get("generator_profile") if isinstance(method.get("generator_profile"), dict) else {}
            csmith_args = [str(item) for item in profile.get("args", [])]
            lane = str(method.get("lane", "rv64imc/lp64")).split("/", 1)
            isa = str(profile.get("isa") or lane[0])
            mabi = str(profile.get("mabi") or (lane[1] if len(lane) > 1 else "lp64"))
            opt_level = int(profile.get("opt_level", 0))
            compiler = command("riscv64-linux-gnu-gcc")
            compiler_include_flags = _cross_compiler_include_flags(compiler)
            native_compiler = command("gcc")
            include = csmith.parent.parent / "include" if csmith else None
            freestanding_include = HERE / "assets" / "freestanding"
            runtime = HERE / "assets" / "csmith_runtime.c"
            capsule = HERE / "assets" / "csmith_capsule.S"
            linux_capsule = HERE / "assets" / "csmith_linux_capsule.S"
            qemu_binary = next((dep(target.get("binary")) for target in config.get("targets", [])
                                if target.get("id") == "T-QEMU"), None)
            renode_binary = next((dep(target.get("binary")) for target in config.get("targets", [])
                                  if target.get("id") == "T-RENODE"), None)
            dotnet_binary = dep(deps.get("dotnet"))
            phase_deadline = method_deadline
            def remaining() -> int | None:
                return (
                    None if math.isinf(phase_deadline)
                    else max(0, math.floor(phase_deadline - time.monotonic()))
                )
            candidates = []
            failure_counts = Counter()
            last = {}
            source_built_count = 0
            index = start_index
            segment_attempt_count = 0
            deadline_censored = False

            while not stop_requested() and time.monotonic() < phase_deadline:
                generation_case_boundary(
                    f"B-CSMITH-case-{index - 1:08d}" if index else None,
                    index - 1 if index else None,
                )
                source = tools_dir / f"csmith-{index:05d}.c"
                run = _run_logged(
                    out, f"tool-B-CSMITH-generator-{index:05d}",
                    [str(csmith), "--seed", str(seed + index), *csmith_args], remaining())
                if time.monotonic() >= phase_deadline:
                    deadline_censored = True
                    generation_case_boundary(f"B-CSMITH-case-{index:08d}", index)
                    index += 1
                    break
                if stop_requested() or run.get("reason") == "external-signal":
                    generation_case_boundary(f"B-CSMITH-case-{index:08d}", index)
                    index += 1
                    break
                if run.get("status") != "passed" or not run.get("stdout", "").strip():
                    failure_counts["csmith-generator-failed"] += 1
                    generation_case_boundary(f"B-CSMITH-case-{index:08d}", index)
                    index += 1
                    continue
                if not _csmith_source_capture_complete(run):
                    # A partial Csmith stream is not a source candidate. Keep
                    # the raw command evidence and move to the next seed.
                    failure_counts["csmith-source-capture-incomplete"] += 1
                    generation_case_boundary(f"B-CSMITH-case-{index:08d}", index)
                    index += 1
                    continue
                write_text_once(source, run["stdout"])
                source_built_count += 1
                native_elf = tools_dir / f"csmith-{index:05d}.native"
                cross_elf = tools_dir / f"csmith-{index:05d}.elf"
                linux_elf = tools_dir / f"csmith-{index:05d}-linux.elf"
                native = _run_logged(
                    out, f"tool-B-CSMITH-native-compile-{index:05d}",
                    [native_compiler, f"-O{opt_level}", str(source), "-I", str(include), "-lm", "-o", str(native_elf)], remaining(),
                ) if native_compiler and include and include.is_dir() else {"status": "gap", "reason": "native-compile-input-missing"}
                # 保留 GCC 数学 builtin，使 Csmith safe-math 的 fabsf 能落到 RISC-V F 指令。
                cross = _run_logged(
                    out, f"tool-B-CSMITH-cross-compile-{index:05d}",
                    [compiler, f"-O{opt_level}", "-mcmodel=medany", "-msmall-data-limit=0", "-fno-stack-protector",
                     "-fno-pic", "-ffunction-sections", "-fdata-sections", "-nostdinc", "-nostdlib",
                     "-nostartfiles", "-nodefaultlibs", "-static", "-no-pie", f"-march={isa}", f"-mabi={mabi}",
                     *compiler_include_flags,
                     "-DCSMITH_MINIMAL", "-DNO_PRINTF", "-DNOT_PRINT_CHECKSUM",
                     "-I", str(freestanding_include), "-I", str(include),
                     str(source), str(runtime), str(capsule), "-Wl,--gc-sections", "-Wl,--build-id=none", "-Wl,--no-relax",
                     "-Wl,-Ttext=0x80001000", "-Wl,-Tdata=0x81000000", "-Wl,-e,_start", "-o", str(cross_elf)], remaining(),
                ) if compiler and include and include.is_dir() and runtime.is_file() and capsule.is_file() else {
                    "status": "gap", "reason": "cross-runtime-missing"}
                linux_cross = _run_logged(
                    out, f"tool-B-CSMITH-linux-cross-compile-{index:05d}",
                    [compiler, f"-O{opt_level}", "-mcmodel=medany", "-msmall-data-limit=0",
                     "-fno-stack-protector", "-fno-pic", "-ffunction-sections", "-fdata-sections",
                     "-nostdinc", "-nostdlib", "-nostartfiles", "-nodefaultlibs", "-static", "-no-pie",
                     f"-march={isa}", f"-mabi={mabi}", "-DCSMITH_MINIMAL", "-DNO_PRINTF",
                     *compiler_include_flags,
                     "-DNOT_PRINT_CHECKSUM", "-I", str(freestanding_include),
                     "-I", str(include), str(source), str(runtime), str(linux_capsule),
                     "-Wl,--gc-sections", "-Wl,--build-id=none", "-Wl,--no-relax", "-Wl,-Ttext=0x12000",
                     "-Wl,-Tdata=0x400000", "-Wl,-e,_start", "-o", str(linux_elf)], remaining(),
                ) if compiler and include and include.is_dir() and runtime.is_file() and linux_capsule.is_file() else {
                    "status": "gap", "reason": "linux-cross-runtime-missing"}
                bare_ok = cross.get("status") == "passed" and cross_elf.is_file()
                linux_ok = linux_cross.get("status") == "passed" and linux_elf.is_file()
                compile_ok = bare_ok or linux_ok
                dual_ok = bare_ok and linux_ok
                last = {"source": source, "native": native, "cross": cross, "linux_cross": linux_cross,
                        "elf": cross_elf, "linux_elf": linux_elf}
                stage_deadline_censored = any(
                    _shared_deadline_censored(stage, phase_deadline)
                    for stage in (run, native, cross, linux_cross)
                )
                deadline_censored = deadline_censored or stage_deadline_censored
                smoke = (
                    _csmith_portable_smoke(
                        out, cross_elf, linux_elf, qemu_binary, renode_binary,
                        dotnet_binary, remaining(), deadline=phase_deadline,
                    ) if dual_ok and not generation_only else
                    {"status": "deferred", "passed": None,
                     "reason": "target-portability-smoke-deferred"}
                    if compile_ok else {"status": "gap", "passed": False,
                                        "reason": "pair-build-failed"}
                )
                last["smoke"] = smoke
                smoke_deadline_censored = any(
                    _shared_deadline_censored(smoke.get(name), phase_deadline)
                    for name in ("qemu", "renode")
                ) if isinstance(smoke, Mapping) else False
                deadline_censored = deadline_censored or smoke_deadline_censored
                if (not stop_requested() and not compile_ok
                        and not stage_deadline_censored):
                    failure_counts["csmith-cross-compile-failed"] += 1
                elif (not stop_requested() and not stage_deadline_censored
                      and not smoke_deadline_censored and smoke.get("passed") is False):
                    if (smoke.get("qemu") or {}).get("status") != "passed":
                        failure_counts["csmith-qemu-smoke-failed"] += 1
                    elif (smoke.get("renode") or {}).get("status") != "passed":
                        failure_counts["csmith-renode-smoke-failed"] += 1
                    else:
                        failure_counts["csmith-smoke-failed"] += 1
                # Target smoke is a portability diagnostic, not a source/
                # pair-construction gate.  Keep compile-valid pairs so
                # target failures are counted at the target layer.
                ok = compile_ok
                if ok:
                    candidate = {
                        "index": candidate_index_offset + len(candidates),
                        "generation_attempt_index": index,
                        "source": str(source.relative_to(out)),
                        "source_sha256": sha256(source), "artifact_isa": isa,
                        "linux_artifact_isa": isa,
                        "portable_smoke": smoke,
                        "portable_smoke_passed": smoke.get("passed")
                        if isinstance(smoke.get("passed"), bool) else None,
                    }
                    if bare_ok:
                        candidate.update(
                            elf=str(cross_elf.relative_to(out)),
                            elf_sha256=sha256(cross_elf),
                        )
                    if linux_ok:
                        candidate.update(
                            linux_elf=str(linux_elf.relative_to(out)),
                            linux_elf_sha256=sha256(linux_elf),
                        )
                    candidates.append(candidate)
                    if on_candidate is not None:
                        on_candidate(method_id, candidate)
                if stage_deadline_censored or smoke_deadline_censored:
                    generation_case_boundary(f"B-CSMITH-case-{index:08d}", index)
                    index += 1
                    break
                index += 1
                segment_attempt_count += 1
                generation_case_boundary(f"B-CSMITH-case-{index - 1:08d}", index - 1)
            if candidates:
                source = out / candidates[-1]["source"]
                cross_elf = (
                    out / candidates[-1]["elf"]
                    if candidates[-1].get("elf") else None
                )
                linux_elf = (
                    out / candidates[-1]["linux_elf"]
                    if candidates[-1].get("linux_elf") else None
                )
            else:
                source = last.get("source")
                cross_elf = last.get("elf")
                linux_elf = last.get("linux_elf")
            ok = bool(candidates)
            deadline_censored = deadline_censored or time.monotonic() >= phase_deadline
            interrupted = stop_requested()
            failure_reason = (
                "generation-deadline-reached" if deadline_censored else
                stop_reason() if interrupted else
                max(failure_counts, key=failure_counts.get)
                if failure_counts else "generation-budget-exhausted"
            )
            generation_partial = deadline_censored or interrupted or not ok
            segment_attempt_count = max(0, index - start_index)
            status = (
                "partial" if ok and generation_partial
                else "passed" if ok else
                "partial" if deadline_censored or interrupted else "gap"
            )
            append(_tool_record(
                method_id, status,
                None if status == "passed" else failure_reason,
                generator_invoked=True, generated=bool(last),
                generator={"status": "partial" if generation_partial else "passed" if ok else "gap",
                            "cases_attempted": segment_attempt_count,
                            "cases_built": len(candidates),
                            "cases_rejected": index - len(candidates)},
                attempt_count=segment_attempt_count, global_attempt_count=index,
                generation_cursor={
                    "start_index": start_index,
                    "next_index": index,
                    "next_seed": seed + index,
                },
                source_built_count=source_built_count,
                pair_built_count=len(candidates), accepted_count=len(candidates),
                smoke=last.get("smoke"),
                native_compile=_result_summary(last.get("native", {})),
                cross_compile=_result_summary(last.get("cross", {})),
                linux_cross_compile=_result_summary(last.get("linux_cross", {})),
                source=str(source.relative_to(out)) if source else None,
                source_sha256=sha256(source) if source else None,
                elf=str(cross_elf.relative_to(out)) if cross_elf and cross_elf.is_file() else None,
                elf_sha256=sha256(cross_elf) if cross_elf else None,
                built=bool(candidates), artifact_isa=isa,
                linux_elf=str(linux_elf.relative_to(out)) if linux_elf and linux_elf.is_file() else None,
                linux_elf_sha256=sha256(linux_elf) if linux_elf else None,
                linux_built=any(item.get("linux_elf") for item in candidates), linux_artifact_isa=isa,
                compile_valid_count=len(candidates),
                smoke_rejected_count=sum(
                    1 for item in candidates if item.get("portable_smoke_passed") is False
                ),
                smoke_is_diagnostic=True,
                candidate_artifacts=candidates, candidate_count=len(candidates),
                candidate_count_discovered=len(candidates),
                generation_partial=generation_partial,
                generation_partial_reason=failure_reason,
                deadline_censored=deadline_censored,
                artifact_gap_count=max(0, source_built_count - len(candidates)),
                failure_counts=dict(failure_counts),
            ))
            continue
        if method_id == "B-GEMI":
            profile = method.get("generator_profile") if isinstance(method.get("generator_profile"), dict) else {}
            csmith = dep(deps["csmith"])
            runtime = HERE / "assets" / "csmith_runtime.c"
            capsule = HERE / "assets" / "csmith_capsule.S"
            linux_capsule = HERE / "assets" / "csmith_linux_capsule.S"
            qemu_binary = next((dep(target.get("binary")) for target in config.get("targets", [])
                                if target.get("id") == "T-QEMU"), None)
            def checkpoint_gemi_attempt(checkpoint: dict[str, Any]) -> None:
                attempt_index = checkpoint.get("attempt")
                case_index = (
                    attempt_index - 1 if type(attempt_index) is int else
                    max(0, int(checkpoint.get("attempt_count", 1)) - 1)
                )
                case_index += attempt_offset
                checkpoint = {**checkpoint, "global_attempt": case_index + 1}
                checkpoint_path = tools_dir / "B-GEMI-generation-progress.json"
                write_json_replace(checkpoint_path, {
                    "schema_version": "rq1-generation-progress-v1",
                    "method": method_id, "lane": "lane-00",
                    "last_case_completed": checkpoint,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                generation_case_boundary(
                    f"B-GEMI-attempt-{case_index:08d}", case_index,
                )

            trial = run_decfuzzer_main(
                csmith=csmith, decfuzzer=dep(deps["decfuzzer"]), out=out, timeout=tool_timeout,
                seed=seed + attempt_offset, generator_profile=profile, csmith_include=csmith.parent.parent / "include",
                target_runtime=runtime, capsule=capsule,
                linux_capsule=linux_capsule, run_logged=_run_logged, smoke_binary=qemu_binary,
                write_once=write_text_once, sha256=sha256, command=command,
                artifact_isa=str(profile.get("isa") or str(method.get("lane", "rv64imc/lp64")).split("/", 1)[0]),
                artifact_mabi=str(profile.get("mabi") or str(method.get("lane", "rv64imc/lp64")).split("/", 1)[-1]),
                opt_level=int(profile.get("opt_level", 0)), run_smoke=not generation_only,
                should_stop=stop_requested,
                on_candidate=(
                    (lambda candidate: on_candidate(method_id, candidate))
                    if on_candidate is not None else None
                ),
                on_case_checkpoint=checkpoint_gemi_attempt,
                continuous_generation=segment_control_enabled,
                run_deadline=method_deadline,
                candidate_index_offset=candidate_index_offset,
            )
            if not isinstance(trial, dict):
                trial = _tool_record(
                    method_id, "gap", "decfuzzer-result-invalid",
                    candidate_artifacts=[], candidate_count=0,
                )
            raw_candidates = trial.get("candidate_artifacts")
            candidates = [candidate for candidate in (
                raw_candidates if isinstance(raw_candidates, list) else []
            )
                          if isinstance(candidate, dict)
                          and candidate.get("source")
                          and any(candidate.get(name) for name in ("elf", "linux_elf"))]
            candidate_gates = []
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                candidate_index = _nonnegative_int(candidate.get("index", index), default=index)
                gate = {"status": "deferred", "reason": "reference-deferred",
                        "observations": {}, "equivalent_outcome": None,
                        "deferred": True, "reference_equivalent": None}
                candidate["reference_gate"] = gate
                candidate_gates.append({"index": candidate_index,
                                        "status": gate.get("status"),
                                        "reason": gate.get("reason")})
            valid = [candidate for candidate in candidates
                     if isinstance(candidate, dict)
                     and candidate.get("reference_gate", {}).get("status") in {"passed", "deferred"}]
            trial["candidate_artifacts"] = valid
            trial["valid_candidate_count"] = len(valid)
            reference_gate = {
                "status": "deferred" if valid else "gap",
                "candidates": candidate_gates,
                "reason": None if valid else "gemi-reference-equivalence-failed",
                "deferred": bool(valid),
            }
            ok = trial.get("status") == "passed" and bool(valid)
            trial["pq_chain_complete"] = bool(valid)
            trial_fields = {key: value for key, value in trial.items()
                            if key not in {"status", "reason", "generation_partial"}}
            trial_fields.update(
                attempt_count=_nonnegative_int(
                    trial.get(
                        "candidate_count_discovered",
                        trial.get("candidate_count", len(candidates)),
                    ),
                    default=len(candidates),
                ),
                source_built_count=_nonnegative_int(
                    trial.get("candidate_count_discovered", len(candidates)),
                    default=len(candidates),
                ),
                pair_built_count=len(valid), accepted_count=len(valid),
                generation_cursor={
                    "attempt_offset": attempt_offset,
                    "next_parent_seed": seed + attempt_offset + _nonnegative_int(
                        trial.get("attempt_count"), default=0,
                    ),
                    "candidate_index_offset": candidate_index_offset,
                    "next_candidate_index": candidate_index_offset + len(valid),
                },
            )
            trial_deadline_censored = trial.get("deadline_censored") is True
            interrupted = stop_requested() or trial.get("reason") == "external-signal"
            trial_status = (
                "passed" if ok else
                "partial" if valid or trial_deadline_censored or interrupted else "gap"
            )
            trial_reason = (
                None if ok else
                "generation-deadline-reached" if trial_deadline_censored else
                stop_reason() if interrupted else
                "decfuzzer-main-build-or-reference-failed"
            )
            append(_tool_record(
                method_id, trial_status, trial_reason,
                generation_partial=not ok,
                **trial_fields, reference_gate=reference_gate,
            ))
            continue
    row = rows[0] if len(rows) == 1 else {}
    selected_method_shape = (
        selected_method_shape and row.get("method") == method_filter
    )
    complete = selected_method_shape and row.get("status") == "passed"
    partial = selected_method_shape and row.get("status") in {"passed", "partial"}
    overall_status = "passed" if complete else "partial" if partial else "gap"
    return {"schema_version": "rq1-comparison-tool-run-v1",
            "provenance": provenance or _provenance(config),
            "method_shape_ok": selected_method_shape,
            "method_filter": method_filter,
            "status": overall_status,
            "generation_continuation": continuation,
            "methods": rows}


# RQ1 target id -> framework execution backend。
FRAMEWORK_TARGET_BACKENDS = {
    "T-QEMU": "qemu-riscv64", "T-LRSV-INT": "libriscv-interpreter",
    "T-LRSV-TRANS": "libriscv-translated",
    "T-UNICORN": "unicorn-riscv64", "T-RENODE": "renode-riscv64",
    "T-RAX": "rax-riscv64", "T-RVVM": "rvvm-riscv64",
}


def _framework_manifest(out: Path, *, method: str, feedback_target: str) -> tuple[dict[str, Any], bool]:
    """读取 wrapper manifest；framework-run 不能复用 comparison 方法清单。"""
    path = out / "execution-manifest.json"
    if not path.is_file():
        return {}, False
    manifest = load_json(path)
    methods = manifest.get("methods")
    targets = manifest.get("targets")
    method_ids = [item.get("id") for item in methods if isinstance(item, dict)] \
        if isinstance(methods, list) else []
    target_ids = [item.get("id") for item in targets if isinstance(item, dict)] \
        if isinstance(targets, list) else []
    expected_target_ids = list(TARGETS) if feedback_target == "all" else [feedback_target]
    verified = (
        manifest.get("experiment_face") == "framework"
        and manifest.get("action") == "framework-run"
        and manifest.get("method_filter") == method
        and manifest.get("feedback_target") == feedback_target
        and method_ids == [method]
        and target_ids == expected_target_ids
    )
    if not verified:
        raise ValueError("framework manifest does not describe the selected standalone campaign")
    return manifest, True


def _framework_chain_events(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    chain = campaign.get("chain")
    events = chain.get("events") if isinstance(chain, dict) else None
    if isinstance(events, list):
        return [dict(item) for item in events if isinstance(item, dict)]
    return []


def _framework_write_chain_artifacts(
    out: Path, method_dir: Path, campaign: dict[str, Any],
    record: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    """把当前 JSON campaign 投影成独立 chain/integrity/seal 产物。"""
    chain_dir = method_dir / "chain"
    chain_dir.mkdir(parents=True, exist_ok=True)
    events = _framework_chain_events(campaign)
    events_path = chain_dir / "events.jsonl"
    write_event_ledger(events_path, events)
    target_records = campaign.get("target")
    target_records = target_records if isinstance(target_records, list) else []
    target_records_by_target = campaign.get("target_records_by_target")
    target_records_by_target = (
        target_records_by_target if isinstance(target_records_by_target, Mapping) else {}
    )

    def target_snapshot_for_state(target: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: target.get(key)
            for key in (
                "status", "candidate", "target_result_status", "failure_class",
                "state_sha256", "parent_sha256", "target_feedback",
                "reference_feedback", "comparison",
            ) if key in target
        }

    state_paths: list[Path] = []
    for index, event in enumerate(events):
        step = event.get("step", index)
        try:
            step_number = int(step)
        except (TypeError, ValueError):
            step_number = index
        target_snapshot = None
        target_ref = event.get("target")
        target_index = target_ref.get("target_record_index") \
            if isinstance(target_ref, dict) else None
        if isinstance(target_index, int) and 0 <= target_index < len(target_records):
            target = target_records[target_index]
            if isinstance(target, dict):
                target_snapshot = target_snapshot_for_state(target)
        target_snapshots = {}
        target_refs = event.get("targets")
        if isinstance(target_refs, Mapping):
            for target_id, reference in target_refs.items():
                records = target_records_by_target.get(target_id)
                index = (
                    reference.get("target_record_index")
                    if isinstance(reference, Mapping) else None
                )
                if (
                    isinstance(records, list) and type(index) is int
                    and 0 <= index < len(records)
                    and isinstance(records[index], Mapping)
                ):
                    target_snapshots[str(target_id)] = target_snapshot_for_state(
                        records[index],
                    )
        state = {
            "schema_version": "rq1-framework-state-v1",
            "method": method_dir.name,
            "step": step_number,
            "status": event.get("status"),
            "gate": event.get("gate"),
            "parent_sha256": event.get("parent_sha256"),
            "candidate_sha256": event.get("candidate_sha256"),
            "accepted": event.get("status") == "accepted",
            "mcmc_feedback_source": event.get("feedback_source")
            or campaign.get("mcmc_feedback_source"),
            "mcmc_chain_mode": campaign.get("mcmc_chain_mode"),
            "reference_feedback": event.get("reference_feedback"),
            "reference_path_score": event.get("reference_path_score"),
            "reference_reward_status": event.get("reference_reward_status"),
            "target": target_snapshot,
            "targets": target_snapshots,
            "target_available": event.get("target_available"),
        }
        state_path = chain_dir / f"state-{step_number:06d}.json"
        write_json_replace(state_path, state)
        state_paths.append(state_path)

    tracked = [
        method_dir / name for name in (
            "campaign-result.json", "campaign-result.partial.json",
            "candidate-ledger.json", "target-coverage.json",
            "coverage/summary.json",
        ) if (method_dir / name).is_file()
    ]
    source_profile_root = method_dir / "coverage" / "simulator-source"
    if source_profile_root.is_dir():
        tracked.extend(
            path for path in sorted(source_profile_root.rglob("*"))
            if path.is_file()
        )
    target_summary_root = method_dir / "coverage" / "targets"
    if target_summary_root.is_dir():
        tracked.extend(
            path for path in sorted(target_summary_root.rglob("summary.json"))
            if path.is_file()
        )
    tracked += [events_path, *state_paths]
    file_digests = {
        str(path.relative_to(out)): sha256(path)
        for path in tracked if path.is_file()
    }
    counts = campaign.get("counts") if isinstance(campaign.get("counts"), dict) else {}
    counts = dict(counts)
    event_status_counts = dict(Counter(str(event.get("status")) for event in events))
    manifest_ok = bool(manifest)
    # proposal gap/timeout 是合法终态；零步失败也可封存，只要其终态和
    # 空 target 列表被 campaign 明确记录。online_feedback 另行表示实验质量。
    counts = campaign.get("counts")
    terminal_zero_step_gap = (
        not events
        and campaign.get("status") in {
            "generation-gap", "seed-miss", "profile-gap", "reference-gap",
            "transport-gap", "case-skipped", "artifact-gap",
        }
        and isinstance(counts, dict)
        and type(counts.get("steps")) is int
        and counts["steps"] == 0
        and campaign.get("target") == []
    )
    chain_closed = bool(manifest_ok and file_digests and (events or terminal_zero_step_gap))
    integrity = {
        "schema_version": "rq1-framework-integrity-v1",
        "status": "verified" if chain_closed else "partial",
        "experiment_face": "framework", "method": record.get("method"),
        "feedback_target": record.get("feedback_target"),
        "chain_event_count": len(events), "chain_state_count": len(state_paths),
        "counts": counts, "event_status_counts": event_status_counts,
        "identity": {
            key: manifest.get(key)
            for key in (
                "source_commit", "config_sha256", "image_id", "dependency_identity",
                "execution_plane", "network", "targets",
            ) if key in manifest
        },
        "files": file_digests,
    }
    integrity_path = method_dir / "integrity.json"
    write_json_replace(integrity_path, integrity)
    seal = {
        "schema_version": "rq1-framework-seal-v1",
        "sealed": chain_closed,
        "status": "sealed" if chain_closed else "partial",
        "experiment_face": "framework", "method": record.get("method"),
        "feedback_target": record.get("feedback_target"),
        "chain_events": str(events_path.relative_to(out)),
        "integrity": str(integrity_path.relative_to(out)),
        "integrity_sha256": sha256(integrity_path),
    }
    seal_path = method_dir / "seal.json"
    write_json_replace(seal_path, seal)
    integrity_verified = _framework_verify_chain_artifacts(out, method_dir)
    return {
        "chain_events": str(events_path.relative_to(out)),
        "state_count": len(state_paths),
        "integrity": str(integrity_path.relative_to(out)),
        "seal": str(seal_path.relative_to(out)),
        "sealed": seal["sealed"],
        "integrity_verified": integrity_verified,
    }


def _framework_verify_chain_artifacts(out: Path, method_dir: Path) -> bool:
    """重算 framework chain 的文件摘要和 seal 引用。"""
    try:
        manifest = load_json(out / "execution-manifest.json")
        integrity = load_json(method_dir / "integrity.json")
        seal = load_json(method_dir / "seal.json")
        if integrity.get("status") != "verified" or seal.get("status") != "sealed":
            return False
        if (
            integrity.get("experiment_face") != "framework"
            or integrity.get("method") != manifest.get("method_filter")
            or integrity.get("feedback_target") != manifest.get("feedback_target")
            or seal.get("experiment_face") != "framework"
            or seal.get("method") != manifest.get("method_filter")
            or seal.get("feedback_target") != manifest.get("feedback_target")
        ):
            return False
        identity = integrity.get("identity")
        if not isinstance(identity, dict) or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            return False
        files = integrity.get("files")
        if not isinstance(files, dict):
            return False
        for relative, expected in files.items():
            path = out / relative
            if not path.is_file() or sha256(path) != expected:
                return False
        events_path = out / str(seal.get("chain_events"))
        integrity_path = out / str(seal.get("integrity"))
        if not events_path.is_file() or not integrity_path.is_file():
            return False
        event_records = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if not isinstance(event, dict):
                    return False
                event_records.append(event)
        campaign_path = method_dir / "campaign-result.json"
        if not campaign_path.is_file():
            if len(event_records) != 1:
                return False
            event_count = len(event_records)
            event = event_records[0]
            status = event.get("status")
            entry_status = event.get("entry_status")
            reason = event.get("reason")
            elapsed = event.get("elapsed_s")
            if (
                event.get("event_type") != "case-terminal"
                or status not in {"right-censored", "case-timeout", "case-failure"}
                or event.get("target_available") is not False
                or not isinstance(event.get("case_id"), str)
                or not isinstance(reason, str) or not reason
                or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or elapsed < 0
            ):
                return False
            if status == "right-censored":
                gate = event.get("gate")
                censor_stage = event.get("censor_stage")
                if not isinstance(censor_stage, str) or not censor_stage:
                    return False
                if gate == "shared-wall-clock-deadline":
                    if (
                        reason not in _FRAMEWORK_DEADLINE_CENSOR_REASONS
                        or entry_status not in {"gap", "timeout"}
                    ):
                        return False
                elif gate == "external-signal":
                    if reason != "external-signal" or entry_status not in {"gap", "timeout"}:
                        return False
                elif gate == "case-process-terminal":
                    if (
                        reason != "recovery-case-record-missing"
                        or entry_status != "timeout"
                    ):
                        return False
                else:
                    return False
            if status == "case-timeout" and (
                event.get("gate") != "case-process-terminal"
                or entry_status != "timeout"
            ):
                return False
            if status == "case-failure" and (
                event.get("gate") != "case-process-terminal"
                or not isinstance(entry_status, str)
            ):
                return False
            event_status_counts = dict(Counter(str(item.get("status")) for item in event_records))
            counts = integrity.get("counts")
            if (
                not isinstance(counts, dict)
                or integrity.get("event_status_counts") != event_status_counts
                or event_count != integrity.get("chain_event_count")
                or len(list((method_dir / "chain").glob("state-*.json")))
                != integrity.get("chain_state_count", integrity.get("chain_event_count"))
                or seal.get("integrity_sha256") != sha256(integrity_path)
                or seal.get("sealed") is not True
            ):
                return False
            terminal_counts = {
                "steps": event_count, "accepted": 0, "mh_rejected": 0,
                "mh_attempts": 0,
            }
            return not any(
                key in counts and counts[key] != expected
                for key, expected in terminal_counts.items()
            )
        campaign = load_json(campaign_path)
        reference_records = campaign.get("reference") if isinstance(campaign, dict) else None
        target_records = campaign.get("target") if isinstance(campaign, dict) else None
        target_records_by_target = (
            campaign.get("target_records_by_target")
            if isinstance(campaign, dict) else None
        )
        if not isinstance(reference_records, list) or not isinstance(target_records, list):
            return False
        mapped_target_records = isinstance(target_records_by_target, Mapping)
        if mapped_target_records and sum(
            len(records) for records in target_records_by_target.values()
            if isinstance(records, list)
        ) != len(target_records):
            return False
        raw_target_queue = campaign.get("raw_coverage_only") is True
        referenced_targets = set()
        referenced_target_records = set()
        tested_target_statuses = {"clean", "target-mismatch-candidate", "target-probe-tested"}
        reference_driven = campaign.get("mcmc_feedback_source") == "reference"
        if raw_target_queue:
            if target_records or (
                mapped_target_records
                and any(
                    not isinstance(records, list) or records
                    for records in target_records_by_target.values()
                )
            ):
                return False
            for event in event_records:
                if event.get("target") is not None or event.get("targets") is not None:
                    return False
                reference_pointer = event.get("reference")
                reference_required = (
                    event.get("status") in {"accepted", "mh-rejected", "reference-rejected"}
                    or event.get("gate") in {"reference-passed", "reference-rejected"}
                )
                if reference_pointer is None:
                    if reference_required:
                        return False
                    continue
                if not isinstance(reference_pointer, Mapping) \
                        or type(reference_pointer.get("reference_index")) is not int:
                    return False
                reference_index = reference_pointer["reference_index"]
                if not 0 <= reference_index < len(reference_records):
                    return False
                reference_record = reference_records[reference_index]
                reference_result = (
                    reference_record.get("reference")
                    if isinstance(reference_record, Mapping) else None
                )
                if not isinstance(reference_result, Mapping):
                    return False
                if event.get("gate") == "reference-passed" \
                        and reference_result.get("status") != "equivalent":
                    return False
        for event in event_records:
            gate = event.get("gate")
            status = event.get("status")
            pointer = event.get("target")
            target_pointers = event.get("targets")
            needs_target = not raw_target_queue and (
                pointer is not None or target_pointers is not None
                or gate in {"reference-passed", "target-gap", "target-probe-gap", "case-skipped", "artifact-gap"}
                or status in {"accepted", "mh-rejected"}
            )
            if not needs_target:
                continue
            if mapped_target_records:
                if not isinstance(target_pointers, Mapping) or not target_pointers \
                        or set(target_pointers) != set(target_records_by_target):
                    return False
                reference_pointer = event.get("reference")
                if not isinstance(reference_pointer, Mapping) \
                        or type(reference_pointer.get("reference_index")) is not int:
                    return False
                reference_index = reference_pointer["reference_index"]
                if not 0 <= reference_index < len(reference_records):
                    return False
                reference_record = reference_records[reference_index]
                reference_result = (
                    reference_record.get("reference")
                    if isinstance(reference_record, Mapping) else None
                )
                if not isinstance(reference_result, Mapping):
                    return False
                if gate == "reference-passed" and reference_result.get("status") != "equivalent":
                    return False
                if event.get("target_available") is not True:
                    return False
                target_observations = event.get("target_observations")
                if not isinstance(target_observations, Mapping) \
                        or set(target_observations) != set(target_pointers):
                    return False
                for target_id, target_pointer in target_pointers.items():
                    records = target_records_by_target.get(target_id)
                    target_index = (
                        target_pointer.get("target_record_index")
                        if isinstance(target_pointer, Mapping) else None
                    )
                    reference_key = (target_id, target_index)
                    if (
                        not isinstance(records, list)
                        or type(target_index) is not int
                        or not 0 <= target_index < len(records)
                        or reference_key in referenced_target_records
                    ):
                        return False
                    target_record = records[target_index]
                    if not isinstance(target_record, Mapping) or (
                        target_record.get("target_id") != target_id
                        or target_record.get("state_sha256") != event.get("candidate_sha256")
                        or target_record.get("parent_sha256") != event.get("parent_sha256")
                        or target_record.get("status") not in {
                            "clean", "target-tested", "target-mismatch-candidate", "target-gap",
                            "target-probe-tested", "target-probe-gap", "case-skipped",
                            "artifact-gap", "raw-recorded",
                        }
                    ):
                        return False
                    if status in {"accepted", "mh-rejected"} and not reference_driven \
                            and target_record.get("status") not in tested_target_statuses:
                        return False
                    referenced_target_records.add(reference_key)
                continue
            if not isinstance(pointer, dict) or type(pointer.get("target_record_index")) is not int:
                return False
            target_index = pointer["target_record_index"]
            if not 0 <= target_index < len(target_records) or target_index in referenced_targets:
                return False
            target_record = target_records[target_index]
            if not isinstance(target_record, dict):
                return False
            reference_pointer = event.get("reference")
            if status in {"case-skipped", "artifact-gap"}:
                # The target rejected this case before a reference comparison;
                # the raw target result is the complete terminal evidence.
                reference_record = None
            else:
                if not isinstance(reference_pointer, dict) \
                        or type(reference_pointer.get("reference_index")) is not int:
                    return False
                reference_index = reference_pointer["reference_index"]
                if not 0 <= reference_index < len(reference_records):
                    return False
                reference_record = reference_records[reference_index]
                if not isinstance(reference_record, dict) \
                        or not isinstance(reference_record.get("reference"), dict) \
                        or reference_record["reference"].get("status") != "equivalent":
                    return False
            if (
                event.get("target_available") is not True
                or target_record.get("state_sha256") != event.get("candidate_sha256")
                or target_record.get("parent_sha256") != event.get("parent_sha256")
                or target_record.get("status") not in {
                    "clean", "target-tested", "target-mismatch-candidate", "target-gap",
                    "target-probe-tested", "target-probe-gap", "case-skipped", "artifact-gap",
                    "raw-recorded",
                }
            ):
                return False
            if status in {"accepted", "mh-rejected"} \
                    and target_record.get("status") not in tested_target_statuses:
                return False
            referenced_targets.add(target_index)
        if not raw_target_queue:
            if mapped_target_records:
                expected_target_records = {
                    (target_id, index)
                    for target_id, records in target_records_by_target.items()
                    if isinstance(records, list)
                    for index in range(len(records))
                }
                if referenced_target_records != expected_target_records:
                    return False
            elif referenced_targets != set(range(len(target_records))):
                return False
        event_count = len(event_records)
        event_status_counts = dict(Counter(str(event.get("status")) for event in event_records))
        counts = integrity.get("counts")
        if not isinstance(counts, dict) or integrity.get("event_status_counts") != event_status_counts:
            return False
        count_checks = {
            "steps": event_count,
            "accepted": event_status_counts.get("accepted", 0),
            "mh_rejected": event_status_counts.get("mh-rejected", 0),
            "mh_attempts": sum(event.get("mh_attempt") is True for event in event_records),
        }
        if any(key in counts and counts[key] != value for key, value in count_checks.items()):
            return False
        return (
            event_count == integrity.get("chain_event_count")
            and len(list((method_dir / "chain").glob("state-*.json")))
            == integrity.get("chain_state_count", integrity.get("chain_event_count"))
            and seal.get("integrity_sha256") == sha256(integrity_path)
            and seal.get("sealed") is True
        )
    except (OSError, TypeError, ValueError):
        return False


_FRAMEWORK_SEED_STRIDE = 1_000_003
# Single-Target runs keep one execution slot; the all-Target fanout sets seven.
_FRAMEWORK_TARGET_EXECUTION_PARALLELISM = 1
# Default only.  Per-method lane count is a resource setting and may be lower
# for a memory-heavy producer such as Program-Full.
FRAMEWORK_CANONICAL_PARALLEL_LANES = 5


def _framework_parallel_lanes(profile: dict[str, Any]) -> int:
    value = profile.get(
        "parallel_lanes", FRAMEWORK_CANONICAL_PARALLEL_LANES,
    )
    if type(value) is not int or value < 1:
        raise ValueError("framework parallel_lanes must be a positive integer")
    return value


def _framework_case_count(profile: dict[str, Any]) -> int | None:
    if "case_count" not in profile:
        return None
    value = profile["case_count"]
    if type(value) is not int or value < 1:
        raise ValueError("framework case_count must be a positive integer")
    return value


def _framework_program_seed_sequence(
    config: Mapping[str, Any], catalog_root: str | Path, manifest_path: str | Path,
) -> dict[str, Any]:
    """Load one pinned B-RVDV case order for Program-Full producer lanes."""
    from case_catalog import load_catalog_queues

    catalog_root = Path(catalog_root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    queues, rows, manifest, artifact_root = load_catalog_queues(
        catalog_root, dict(config), method_filter="B-RVDV",
        manifest_path=manifest_path,
    )
    artifact_root = Path(artifact_root).resolve(strict=True)
    pinned_manifest = load_json(manifest_path)
    current_manifest = load_json(catalog_root / "catalog-manifest.json")
    revision = pinned_manifest.get("revision") if isinstance(pinned_manifest, Mapping) else None
    catalog_id = pinned_manifest.get("catalog_id") if isinstance(pinned_manifest, Mapping) else None
    if (
        type(revision) is not int or revision < 1
        or not isinstance(catalog_id, str) or not catalog_id
    ):
        raise ValueError("pinned B-RVDV case catalog identity is invalid")
    if (
        not isinstance(current_manifest, Mapping)
        or current_manifest.get("catalog_id") != catalog_id
    ):
        raise ValueError("pinned B-RVDV manifest belongs to a different catalog")
    if manifest.get("catalog_revision") != revision or manifest.get("run_id") != catalog_id:
        raise ValueError("pinned B-RVDV catalog identity changed while loading its queues")

    by_target: dict[str, list[dict[str, Any]]] = {}
    for target_id in TARGETS:
        records = []
        for entry in queues.get(target_id, ()):
            if not isinstance(entry, Mapping) or entry.get("method") != "B-RVDV":
                continue
            candidate = entry.get("candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            case_id = entry.get("case_id")
            case_number = entry.get("case_number")
            source_sha = candidate.get("source_sha256")
            if (
                not isinstance(case_id, str) or not case_id
                or type(case_number) is not int or case_number < 1
                or not isinstance(source_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_sha.lower()) is None
            ):
                raise ValueError(f"B-RVDV catalog entry is incomplete for {target_id}")
            records.append({
                "case_id": case_id,
                "case_number": case_number,
                "source_sha256": source_sha.lower(),
                "candidate": candidate,
                "tool_root": entry.get("tool_root"),
                "batch_id": entry.get("batch_id"),
            })
        records.sort(key=lambda item: item["case_number"])
        by_target[target_id] = records

    sequence = by_target.get(TARGETS[0], [])
    if not sequence:
        raise ValueError("pinned case catalog contains no B-RVDV cases")
    expected = [
        (item["case_id"], item["case_number"], item["source_sha256"])
        for item in sequence
    ]
    if any(
        [
            (item["case_id"], item["case_number"], item["source_sha256"])
            for item in by_target[target_id]
        ] != expected
        for target_id in TARGETS[1:]
    ):
        raise ValueError("B-RVDV Target queues do not share the same ordered source sequence")

    cases = []
    for index, item in enumerate(sequence):
        source = item["candidate"].get("source")
        tool_root = item.get("tool_root")
        if not isinstance(source, str) or not source:
            raise ValueError(f"B-RVDV source path is missing for {item['case_id']}")
        if not isinstance(tool_root, str) or not tool_root:
            raise ValueError(f"B-RVDV tool root is missing for {item['case_id']}")
        source_path = Path(source)
        tool_path = Path(tool_root)
        source_path = (tool_path / source_path).resolve(strict=True) \
            if not source_path.is_absolute() else source_path.resolve(strict=True)
        if not source_path.is_relative_to(artifact_root) or source_path.suffix.lower() != ".s":
            raise ValueError(f"B-RVDV source path escapes the catalog for {item['case_id']}")
        if not source_path.is_file():
            raise ValueError(f"B-RVDV source file is missing for {item['case_id']}")
        candidate = item["candidate"]
        cases.append({
            "sequence_index": index,
            "case_id": item["case_id"],
            "case_number": item["case_number"],
            "source_path": str(source_path),
            "source_relative_path": source_path.relative_to(artifact_root).as_posix(),
            "source_sha256": item["source_sha256"],
            "artifact_isa": candidate.get("artifact_isa"),
            "batch_id": item.get("batch_id"),
        })
    declared_count = rows.get("B-RVDV", {}).get("candidate_count")
    if type(declared_count) is int and declared_count != len(cases):
        raise ValueError("B-RVDV catalog count does not match its ordered case sequence")
    return {
        "schema_version": "rq1-program-seed-sequence-v1",
        "catalog_id": catalog_id,
        "catalog_revision": revision,
        "catalog_manifest_sha256": sha256(manifest_path),
        "method": "B-RVDV",
        "case_count": len(cases),
        "cases": cases,
    }


def _framework_count(counts: dict[str, Any], name: str) -> int:
    if not isinstance(counts, dict):
        return 0
    return _nonnegative_int(counts.get(name, 0), default=0)


def _framework_rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": _metric_ratio(numerator, denominator),
    }


def _framework_accepted_per_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        case for case in cases
        if case.get("campaign_result_recorded") is True
        and case.get("right_censored") is not True
    ]
    accepted = sum(_framework_count(case, "accepted") for case in completed)
    return _framework_rate(accepted, len(completed))


def _framework_target_progress_snapshot(
    method_dir: Path, target_ids: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    """Read Target-owned progress without waiting for a case result.

    The queue manifest is producer-owned and immutable after publication.
    Consumer completion therefore comes from each Target's result directory,
    never from a shared manifest state written by another Target.
    """
    progress: dict[str, dict[str, Any]] = {
        str(target_id): {
            "queued_cases": 0,
            "completed_cases": 0,
            "pending_cases": 0,
            "record_count": 0,
            "baseline_count": 0,
            "pending_execution_count": 0,
            "counts": {},
        }
        for target_id in target_ids
    }
    def safe_target(value: object) -> str:
        return "".join(
            char if char.isalnum() or char in ".-_" else "_"
            for char in str(value)
        ) or "unknown"

    # Formal framework runs maintain one producer state and one consumer
    # aggregate per Target. Progress is an observation of those aggregates,
    # never a scheduler input and never a reason to inspect old case files.
    queue_root = Path(method_dir) / "target-queues"
    queue_roots = {
        str(target_id): queue_root / safe_target(target_id)
        for target_id in target_ids
    }
    for target_id, root in queue_roots.items():
        try:
            state = load_json(root / "state.json")
        except (OSError, TypeError, ValueError):
            state = {}
        try:
            aggregate = load_json(root / "progress.json")
        except (OSError, TypeError, ValueError):
            aggregate = {}
        row = progress[target_id]
        try:
            worker_status = load_json(root / "worker-status.json")
        except (OSError, TypeError, ValueError):
            worker_status = {}
        published = _nonnegative_int(state.get("published_count"), default=0) \
            if isinstance(state, Mapping) else 0
        worker_has_accounting = isinstance(worker_status, Mapping) and any(
            name in worker_status
            for name in ("published_count", "completed_count", "pending_count")
        )
        accounting = worker_status if worker_has_accounting else aggregate
        completed = _nonnegative_int(
            accounting.get("completed_count"), default=0,
        ) if isinstance(accounting, Mapping) else 0
        pending = _nonnegative_int(
            accounting.get("pending_count"), default=max(0, published - completed),
        ) if isinstance(accounting, Mapping) else max(0, published - completed)
        row.update({
            "queued_cases": published,
            "completed_cases": completed,
            "pending_cases": pending,
            "pending_execution_count": pending,
            "progress_source": "target-queue-aggregate",
            "consumer_ownership": (
                "resident-service" if worker_has_accounting else "queue-aggregate"
            ),
            "worker_status": (
                worker_status.get("status")
                if isinstance(worker_status, Mapping) else None
            ),
            "worker_error": (
                worker_status.get("error")
                if isinstance(worker_status, Mapping) else None
            ),
            "error_count": _nonnegative_int(
                worker_status.get("error_count")
                if isinstance(worker_status, Mapping) else 0,
                default=0,
            ),
            "publish_timing": (
                state.get("publish_timing")
                if isinstance(state, Mapping)
                and isinstance(state.get("publish_timing"), Mapping) else {}
            ),
        })
        if isinstance(aggregate, Mapping):
            row["record_count"] = _nonnegative_int(
                aggregate.get("record_count"), default=0,
            )
            row["baseline_count"] = _nonnegative_int(
                aggregate.get("baseline_count"), default=0,
            )
            counts = aggregate.get("counts")
            if isinstance(counts, Mapping):
                row["counts"] = {
                    str(name): _nonnegative_int(value, default=0)
                    for name, value in counts.items()
                    if isinstance(name, str)
                }
    return progress


def _framework_target_publication_snapshot(
    method_dir: Path, target_ids: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    """Return producer-owned queue counts without reading Target execution state."""
    root = Path(method_dir) / "target-queues"
    result: dict[str, dict[str, Any]] = {}
    for target_id in target_ids:
        safe = "".join(
            char if char.isalnum() or char in ".-_" else "_"
            for char in str(target_id)
        ) or "unknown"
        state = {}
        try:
            state = load_json(root / safe / "state.json")
        except (OSError, TypeError, ValueError):
            pass
        published = _nonnegative_int(
            state.get("published_count") if isinstance(state, Mapping) else 0,
            default=0,
        )
        result[str(target_id)] = {
            "target_id": str(target_id),
            "queued_cases": published,
            "completed_cases": None,
            "pending_cases": None,
            "progress_source": "producer-publication-only",
            "consumer_ownership": "external",
        }
    return result


def _framework_target_queue_completion(
    method_dir: Path, target_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Return producer close state and the independent Target drain state."""
    progress = _framework_target_progress_snapshot(method_dir, target_ids)
    queue_root = Path(method_dir) / "target-queues"
    closed = (queue_root / "producer-closed.json").is_file()
    try:
        drain_marker = load_json(queue_root / "drain-status.json")
    except (OSError, TypeError, ValueError):
        drain_marker = {}
    drain_status = (
        str(drain_marker.get("status"))
        if isinstance(drain_marker, Mapping)
        and drain_marker.get("status") in {"draining", "drained", "partial", "error"}
        else None
    )
    pending_values = [
        row.get("pending_cases") for row in progress.values()
        if isinstance(row.get("pending_cases"), int)
    ]
    pending_count = sum(pending_values) if len(pending_values) == len(progress) else None
    # ``partial`` is the normal explicit-stop/timebox state: pending queue
    # lines are intentionally preserved and a resident worker may have been
    # interrupted while closing its native session.  Only a real worker
    # error or supervisor drain error is an error signal.
    worker_errors = {
        target_id: row.get("worker_error") or "target-worker-error"
        for target_id, row in progress.items()
        if row.get("worker_error")
        or (
            drain_status != "partial"
            and _nonnegative_int(row.get("error_count"), default=0) > 0
        )
    }
    if drain_status == "error":
        worker_errors["__supervisor__"] = "target-drain-error"
    drain_ready = drain_status in {None, "drained", "partial"}
    drained = closed and pending_count == 0 and not worker_errors and drain_ready
    return {
        "schema_version": "rq1-target-queue-completion-v1",
        "status": (
            "drained" if drained else
            "drain-error" if drain_status == "error" else
            "partial" if drain_status == "partial" else
            "producer-closed" if closed else
            "producer-open"
        ),
        "producer_closed": closed,
        "drained": drained,
        "drain_status": drain_status,
        "pending_count": pending_count,
        "worker_errors": worker_errors,
        "targets": progress,
    }


def _framework_close_target_queues(
    method_dir: Path, target_ids: tuple[str, ...] | list[str],
) -> None:
    """Seal publication, without claiming that Target work is complete."""
    method_dir = Path(method_dir)
    root = method_dir / "target-queues"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = _framework_target_publication_snapshot(method_dir, target_ids)
    write_json_replace(root / "producer-closed.json", {
        "schema_version": "rq1-target-queue-producer-closed-v1",
        "target_ids": list(target_ids),
        "published_counts": {
            target_id: row.get("queued_cases", 0)
            for target_id, row in snapshot.items()
        },
        "closed_at_epoch": time.time(),
    })


_FRAMEWORK_ROUTE_METRICS = (
    "reachable", "generation_gap", "generated", "seed_valid", "rewrite_valid",
    "reference_valid", "profile_gap", "transport_gap", "seed_miss",
    "reference_rejected", "target_attempted", "target_tested", "target_gap",
    "target_raw_recorded", "target_comparison_pending",
    "artifact_gap",
    "target_unsupported",
    "mismatch_candidate", "confirmed", "clean",
)


def _framework_route_status(rows: list[dict[str, Any]]) -> str:
    metrics = [
        values for row in rows
        for values in row.get("methods", {}).values()
        if isinstance(values, dict)
    ]
    if any(row.get("mapping_conflict") for row in rows):
        return "mapping-gap"
    for status, field in (
        ("confirmed", "confirmed"), ("target-unsupported", "target_unsupported"),
        ("artifact-gap", "artifact_gap"), ("target-gap", "target_gap"),
        ("target-mismatch-candidate", "mismatch_candidate"), ("clean", "clean"),
        ("target-tested", "target_tested"),
        ("raw-recorded", "target_raw_recorded"),
        ("reference-rejected", "reference_rejected"),
        ("reference-valid", "reference_valid"), ("transport-gap", "transport_gap"),
        ("profile-gap", "profile_gap"), ("seed-miss", "seed_miss"),
        ("reachable", "reachable"), ("generation-gap", "generation_gap"),
    ):
        if any(_framework_count(values, field) for values in metrics):
            return status
    return "unexecuted"


_FRAMEWORK_COMPLETE_TARGET_ROUTE_STATUSES = frozenset({
    "clean", "target-tested", "target-mismatch-candidate", "target-probe-tested",
    "confirmed",
})


def _framework_target_campaign_status(
    assigned_statuses: Counter[str], *, missing: int, plan_conflict: bool,
) -> str:
    """把每个已分配 route 的终态合并为真实的 campaign 覆盖状态。"""
    total = sum(assigned_statuses.values())
    complete = sum(
        count for status, count in assigned_statuses.items()
        if status in _FRAMEWORK_COMPLETE_TARGET_ROUTE_STATUSES
    )
    if not total:
        return "gap"
    if complete == total and not missing and not plan_conflict:
        return "observed"
    return "partial" if complete else "gap"


def _framework_merge_target_coverage(
    case_dirs: list[Path], case_count: int,
) -> dict[str, Any]:
    summaries = []
    for case_dir in case_dirs:
        path = case_dir / "target-coverage.json"
        try:
            summary = load_json(path)
        except (OSError, TypeError, ValueError):
            continue
        if isinstance(summary, dict) and isinstance(summary.get("rows"), list):
            summaries.append(summary)

    missing = max(0, case_count - len(summaries))
    campaign = {
        "aggregation": "route-wise-union-of-case-evidence",
        "case_count": case_count,
        "coverage_summary_case_count": len(summaries),
        "missing_case_count": missing,
        "assigned_route_status_counts": {},
        "status": "gap",
    }
    if not summaries:
        reason = "framework-case-target-coverage-missing"
        return {
            "schema": "target-coverage-matrix-v1", "source_schema": None,
            "status": "gap", "route_audit_status": "unavailable",
            "route_audit_gaps": [reason], "denominators": {},
            "results": {"methods": {}, "statuses": {}}, "counts": {},
            "methods": {}, "route_matrix": {}, "plan_completion": {
                "status": "incomplete", "gaps": [reason],
            }, "rows": [], "parallel_campaign": campaign, "reason": reason,
        }

    result = deepcopy(summaries[0])
    base_rows = result.get("rows", [])

    def route_key(row: dict[str, Any]) -> tuple[object, ...]:
        return row.get("root_key"), row.get("lineage_key"), row.get("route")

    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    plan_signature = (
        result.get("source_schema"), result.get("denominators"),
        tuple((route_key(row), row.get("route_assignment_status"), row.get("recipe_status"))
              for row in base_rows if isinstance(row, dict)),
    )
    plan_conflict = False
    assigned_statuses: Counter[str] = Counter()
    audit_gaps: list[set[str]] = []
    for summary in summaries:
        rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
        signature = (
            summary.get("source_schema"), summary.get("denominators"),
            tuple((route_key(row), row.get("route_assignment_status"), row.get("recipe_status"))
                  for row in rows),
        )
        plan_conflict |= signature != plan_signature
        audit_gaps.append({
            gap for gap in summary.get("route_audit_gaps", ()) if isinstance(gap, str)
        })
        for row in rows:
            grouped[route_key(row)].append(row)
            if row.get("route_assignment_status") == "assigned":
                assigned_statuses[str(row.get("evidence_status") or row.get("status") or "unknown")] += 1

    merged_rows = []
    for original in base_rows:
        if not isinstance(original, dict):
            continue
        key = route_key(original)
        cases = grouped.get(key, [])
        if len(cases) != len(summaries):
            plan_conflict = True
        row = deepcopy(original)
        original_methods = row.get("methods")
        original_methods = original_methods if isinstance(original_methods, dict) else {}
        method_names = set(original_methods)
        for case in cases:
            method_data = case.get("methods")
            if isinstance(method_data, dict):
                method_names.update(method_data)
        merged_methods = {}
        for method_name in sorted(method_names):
            values_list = [
                case.get("methods", {}).get(method_name, {}) for case in cases
                if isinstance(case.get("methods"), dict)
            ]
            values = dict(original_methods.get(method_name, {}))
            for field in _FRAMEWORK_ROUTE_METRICS:
                values[field] = max(
                    (_framework_count(item, field) for item in values_list), default=0,
                )
            values["cost"] = None
            if "unit_cost" in values or any("unit_cost" in item for item in values_list):
                values["unit_cost"] = None
            merged_methods[method_name] = values
        row["methods"] = merged_methods
        row["mapping_conflict"] = any(case.get("mapping_conflict") for case in cases)
        row_statuses = Counter(
            str(case.get("evidence_status") or case.get("status") or "unknown")
            for case in cases
        )
        row["case_status_counts"] = dict(sorted(row_statuses.items()))
        row["evidence"] = sorted({
            evidence for case in cases
            for evidence in case.get("evidence", ())
            if isinstance(evidence, str) and evidence
        })
        row["status"] = _framework_route_status([row])
        row["evidence_status"] = row["status"]
        attempted = any(case.get("attempt_status") == "attempted" for case in cases)
        tested = any(
            isinstance(case.get("route_attempt"), dict)
            and case["route_attempt"].get("target_tested") is True
            for case in cases
        )
        row["attempt_status"] = "attempted" if attempted else "unattempted"
        row["route_attempt"] = ({
            "status": "attempted", "source": "parallel-case-aggregate",
            "target_attempted": attempted, "target_tested": tested,
        } if attempted else None)
        merged_rows.append(row)

    methods = {}
    method_names = {name for row in merged_rows for name in row.get("methods", {})}
    for method_name in sorted(method_names):
        totals = {
            field: sum(
                _framework_count(row.get("methods", {}).get(method_name, {}), field)
                for row in merged_rows
            )
            for field in _FRAMEWORK_ROUTE_METRICS
        }
        totals.update(cost=None, unit_cost=None)
        methods[method_name] = totals
    status_counts = {
        "unexecuted": sum(row.get("status") == "unexecuted" for row in merged_rows),
        "mapping-gap": sum(row.get("status") == "mapping-gap" for row in merged_rows),
    }
    for label, field in (
        ("reachable", "reachable"), ("generation-gap", "generation_gap"),
        ("profile-gap", "profile_gap"), ("transport-gap", "transport_gap"),
        ("seed-miss", "seed_miss"), ("seed-valid", "seed_valid"),
        ("reference-rejected", "reference_rejected"),
        ("reference-valid", "reference_valid"),
        ("artifact-gap", "artifact_gap"),
        ("target-unsupported", "target_unsupported"),
        ("target-gap", "target_gap"),
        ("target-tested", "target_tested"),
        ("raw-recorded", "target_raw_recorded"),
        ("target-mismatch-candidate", "mismatch_candidate"),
        ("clean", "clean"), ("confirmed", "confirmed"),
    ):
        status_counts[label] = sum(
            any(_framework_count(values, field) for values in row.get("methods", {}).values())
            for row in merged_rows
        )
    counts = dict(result.get("counts", {}))
    counts.update(status_counts)
    counts["generated"] = sum(
        any(_framework_count(values, "generated") for values in row.get("methods", {}).values())
        for row in merged_rows
    )
    result.update(
        rows=merged_rows, methods=methods,
        results={"methods": methods, "statuses": status_counts},
        counts=counts,
        status="gap",
    )
    merged_audit_gap_set = set().union(*(
        gaps - {"route-attempt"} for gaps in audit_gaps
    )) if audit_gaps else set()
    route_matrix = result.get("route_matrix")
    route_attempt_aggregated = False
    if isinstance(route_matrix, dict):
        matrix_rows = route_matrix.get("rows")
        if isinstance(matrix_rows, list):
            merged_by_route = {route_key(row): row for row in merged_rows}
            for matrix_row in matrix_rows:
                if not isinstance(matrix_row, dict):
                    continue
                merged_row = merged_by_route.get(route_key(matrix_row))
                if merged_row is not None:
                    matrix_row["attempt_status"] = merged_row["attempt_status"]
                    matrix_row["route_attempt"] = merged_row["route_attempt"]
            matrix_counts = route_matrix.get("counts")
            if isinstance(matrix_counts, dict):
                eligible = [row for row in matrix_rows if (
                    isinstance(row, dict) and row.get("status") == "recipe-ready"
                )]
                matrix_counts["route_unattempted"] = sum(
                    row.get("attempt_status") == "unattempted" for row in eligible
                )
                matrix_counts["route_attempt_complete"] = all(
                    row.get("attempt_status") == "attempted" for row in eligible
                )
                route_attempt_aggregated = True
                if matrix_counts["route_attempt_complete"]:
                    merged_audit_gap_set.discard("route-attempt")
                else:
                    merged_audit_gap_set.add("route-attempt")
    if not route_attempt_aggregated and any(
        "route-attempt" in gaps for gaps in audit_gaps
    ):
        merged_audit_gap_set.add("route-attempt")
    merged_audit_gaps = sorted(merged_audit_gap_set)
    result["route_audit_gaps"] = merged_audit_gaps
    result["route_audit_status"] = "complete" if not merged_audit_gaps else "incomplete"
    if isinstance(route_matrix, dict):
        route_matrix["route_audit_gaps"] = merged_audit_gaps
        route_matrix["route_audit_status"] = result["route_audit_status"]
    campaign["assigned_route_status_counts"] = dict(sorted(assigned_statuses.items()))
    campaign["status"] = _framework_target_campaign_status(
        assigned_statuses, missing=missing, plan_conflict=plan_conflict,
    )
    if plan_conflict:
        campaign["reason"] = "framework-case-target-coverage-plan-conflict"
    elif missing:
        campaign["reason"] = "framework-case-target-coverage-incomplete"
    elif campaign["status"] != "observed":
        campaign["reason"] = "framework-case-target-coverage-execution-gap"
    result["status"] = campaign["status"]
    result["campaign_status"] = campaign["status"]
    result["parallel_campaign"] = campaign
    return result


def _framework_recover_case_target_coverage(
    case_dir: Path, method: str, campaign: dict[str, Any],
) -> dict[str, Any]:
    output = case_dir / "target-coverage.json"
    ledger_name = (
        "rvgen-rq1-program-recipes.json"
        if method == "Ours-Program-Full" else "rvgen-rq1-root-recipes.json"
    )
    recipe_path = HERE / "framework" / "evidence" / ledger_name
    try:
        recipe_ledger = load_json(recipe_path)
        candidate_ledger = load_json(case_dir / "candidate-ledger.json")
    except (OSError, TypeError, ValueError):
        result = {
            "output": str(output), "status": "gap",
            "reason": "framework-case-target-coverage-input-missing",
        }
        campaign["target_coverage"] = result
        return result
    if not isinstance(recipe_ledger, Mapping) or not isinstance(candidate_ledger, Mapping):
        result = {
            "output": str(output), "status": "gap",
            "reason": "framework-case-target-coverage-input-missing",
        }
        campaign["target_coverage"] = result
        return result
    try:
        target_record_groups = candidate_ledger.get("target_records_by_target", {})
        target_record_groups = (
            target_record_groups
            if isinstance(target_record_groups, Mapping) else {}
        )
        single_target_records = candidate_ledger.get("target_records")
        if not isinstance(single_target_records, (list, tuple)):
            single_target_records = candidate_ledger.get("target", ())

        def has_compact_record(value: object) -> bool:
            return isinstance(value, Mapping) and isinstance(
                value.get("target_evidence_ref"), Mapping,
            )

        compact_evidence = any(
            has_compact_record(record)
            for records in target_record_groups.values()
            if isinstance(records, (list, tuple))
            for record in records
        ) or any(
            has_compact_record(record)
            for record in (
                candidate_ledger.get("target_baselines_by_target", {}).values()
                if isinstance(candidate_ledger.get("target_baselines_by_target"), Mapping)
                else ()
            )
        ) or any(
            has_compact_record(record) for record in single_target_records
            if isinstance(single_target_records, (list, tuple))
        ) or has_compact_record(candidate_ledger.get("target_baseline"))
        summary = (
            summarize_target_coverage_by_target(
                recipe_ledger, candidate_ledger, evidence_root=case_dir,
            )
            if compact_evidence else
            summarize_target_coverage_by_target(recipe_ledger, candidate_ledger)
        )
        if not isinstance(summary, dict) or not isinstance(summary.get("rows"), list):
            raise ValueError("target coverage summary is incomplete")
        write_json_replace(output, summary)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        result = {
            "output": str(output), "status": "gap",
            "reason": f"framework-case-target-coverage-error:{type(error).__name__}",
        }
        campaign["target_coverage"] = result
        return result
    result = {"output": str(output), "counts": summary.get("counts", {})}
    campaign["target_coverage"] = result
    return result


_FRAMEWORK_DEADLINE_CENSOR_REASONS = frozenset({
    "run-wall-clock-exhausted", "execution-wall-clock-exhausted",
    "campaign-wall-clock-exhausted", "framework-target-wall-clock-exhausted",
    "shared-deadline", "shared-deadline-while-case-running",
    "target-wall-clock-pending",
})


def _framework_case_end_state(
    *, campaign_recorded: bool, entry_status: Any, deadline_reached: bool,
    entry_reason: Any = None,
) -> dict[str, Any]:
    # A deadline watchdog and an operator both reach the child through SIGTERM.
    # ``run_cmd`` reports ``external-signal`` for either path; use the shared
    # deadline to retain whether the lane stopped for its time box or operator.
    if not campaign_recorded and (
        entry_reason == "external-signal"
        or entry_reason in _FRAMEWORK_DEADLINE_CENSOR_REASONS
    ):
        return {
            "lane_stop_reason": (
                "external-signal"
                if entry_reason == "external-signal" and not deadline_reached
                else "deadline"
            ),
            "case_outcome": "right-censored",
        }
    if entry_status == "timeout":
        if deadline_reached:
            return {
                "lane_stop_reason": "deadline",
                "case_outcome": "campaign-result" if campaign_recorded else "right-censored",
            }
        # A timeout before the shared deadline is a terminal case, not a
        # terminal lane. The caller writes its synthetic chain record and
        # immediately gives the lane a fresh seed.
        return {
            "lane_stop_reason": None,
            "case_outcome": "campaign-result" if campaign_recorded else "case-timeout",
        }
    if not campaign_recorded:
        return {"lane_stop_reason": "case-failure", "case_outcome": "case-failure"}
    return {"lane_stop_reason": None, "case_outcome": "campaign-result"}


def _framework_campaign_result_complete(campaign: Mapping[str, Any]) -> bool:
    """只有 finalization 完成的文件才可作为 campaign 结果。"""
    if not isinstance(campaign, Mapping) or not campaign:
        return False
    finalization = campaign.get("finalization")
    return not isinstance(finalization, Mapping) or (
        finalization.get("status") == "complete"
    )


def _framework_raw_only_campaign(
    campaign: Mapping[str, Any], *, coverage_enabled: bool | None = None,
) -> dict[str, Any]:
    """Mark a case complete while leaving coverage aggregation to offline work."""
    result = dict(campaign)
    if coverage_enabled is None:
        finalization = result.get("finalization")
        framework_coverage = result.get("framework_coverage")
        coverage_enabled = (
            simulator_coverage_enabled()
            or isinstance(finalization, Mapping)
            and finalization.get("coverage_enabled") is True
            or isinstance(framework_coverage, Mapping)
            and framework_coverage.get("status") not in {None, "disabled"}
        )
    coverage_status = "deferred" if coverage_enabled else "disabled"
    result["target_coverage"] = {
        "schema": "target-coverage-matrix-v1",
        "status": "deferred",
        "reason": "raw-coverage-only",
    }
    result["framework_coverage"] = {
        "schema_version": "rq1-framework-coverage-v1",
        "status": coverage_status,
        "artifact_complete": False,
        "reason": (
            "raw-coverage-only" if coverage_status == "deferred"
            else "coverage-disabled"
        ),
    }
    result["raw_coverage_only"] = True
    result["finalization"] = {
        "status": "complete",
        "coverage_enabled": coverage_status == "deferred",
        "framework_coverage_status": coverage_status,
        "reason": "raw-coverage-only" if coverage_status == "deferred" else None,
    }
    return result


def _framework_campaign_core_recorded(campaign: Mapping[str, Any]) -> bool:
    """判断 pending 文件是否至少包含已完成的在线链核心结果。"""
    return isinstance(campaign, Mapping) and bool(campaign) and any(
        isinstance(campaign.get(name), Mapping)
        or isinstance(campaign.get(name), list)
        or isinstance(campaign.get(name), str)
        for name in ("counts", "chain", "ledger_path")
    )


def _framework_read_chain_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        checkpoint = load_json(path)
    except (OSError, TypeError, ValueError):
        return None
    campaign = checkpoint.get("campaign") if isinstance(checkpoint, dict) else None
    plan = checkpoint.get("target_replay_plan") if isinstance(checkpoint, dict) else None
    return checkpoint if (
        isinstance(checkpoint, dict)
        and checkpoint.get("schema_version") == "rq1-framework-chain-checkpoint-v1"
        and checkpoint.get("checkpoint_stage") in {
            "reference-chain-closed", "target-replay-progress",
        }
        and _framework_campaign_core_recorded(campaign)
        and isinstance(plan, dict)
    ) else None


def _framework_recover_chain_checkpoint(
    checkpoint: Mapping[str, Any], *, coverage_finalization_requested: bool,
    case_dir: Path | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Keep a closed reference chain and right-censor Target work on interruption."""
    campaign = dict(checkpoint["campaign"])
    plan = checkpoint.get("target_replay_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    target_ids = [str(item) for item in plan.get("target_ids", ()) if isinstance(item, str)]
    pairs = [item for item in plan.get("pairs", ()) if isinstance(item, Mapping)]
    queue_path = plan.get("queue_path")
    queue_path = queue_path if isinstance(queue_path, str) and queue_path else None
    pair_count = len(pairs)
    mode = plan.get("mode")
    recovery_reason = stop_reason or "campaign-wall-clock-exhausted"
    deadline_censored = (
        stop_reason is None
        or stop_reason in _FRAMEWORK_DEADLINE_CENSOR_REASONS
        or stop_reason in {"deadline", "wall-timeout"}
    )
    pending_failure = (
        "campaign-wall-clock-exhausted" if deadline_censored else recovery_reason
    )
    target_records_by_target: dict[str, list[dict[str, Any]]] = {}
    completed_candidate_records = 0
    pending_candidate_records = 0
    durable_records: dict[str, dict[int, dict[str, Any]]] = {}
    durable_baselines: dict[str, dict[str, Any]] = {}
    progress_sequence = -1
    progress_counts: dict[str, Any] = {}

    if case_dir is not None:
        case_dir = Path(case_dir)
        try:
            progress = load_json(case_dir / "campaign-progress.json")
        except (OSError, TypeError, ValueError):
            progress = {}
        if (
            progress.get("schema_version") == "rq1-framework-campaign-progress-v1"
            and type(progress.get("progress_sequence")) is int
            and isinstance(progress.get("counts"), Mapping)
        ):
            progress_sequence = progress["progress_sequence"]
            progress_counts = dict(progress["counts"])
        result_root = case_dir / "target-replay-queue" / "results"

        def load_target_result(
            target_index: int, target_id: str, filename: str,
            pair_index: int | None,
        ) -> dict[str, Any] | None:
            nonlocal progress_sequence, progress_counts
            try:
                value = load_json(
                    result_root / f"target-{target_index:03d}" / filename,
                )
            except (OSError, TypeError, ValueError):
                return None
            record = value.get("record") if isinstance(value, Mapping) else None
            sequence = value.get("progress_sequence") if isinstance(value, Mapping) else None
            counts = value.get("counts") if isinstance(value, Mapping) else None
            if (
                not isinstance(value, Mapping)
                or value.get("schema_version") != "rq1-target-replay-result-v1"
                or value.get("target_id") != target_id
                or value.get("pair_index") != pair_index
                or not isinstance(record, Mapping)
            ):
                return None
            if type(sequence) is int and sequence > progress_sequence \
                    and isinstance(counts, Mapping):
                progress_sequence = sequence
                progress_counts = dict(counts)
            recovered = dict(record)
            if not coverage_finalization_requested and case_dir is not None:
                record_path = result_root / f"target-{target_index:03d}" / filename
                try:
                    compact_record(
                        recovered,
                        make_evidence_ref(
                            case_dir, record_path,
                            target_id=target_id, pair_index=pair_index,
                        ),
                    )
                except (OSError, TypeError, ValueError):
                    # Recovery must remain usable when an older sidecar has no
                    # evidence reference; the inline record is still valid.
                    pass
            return recovered

        for target_index, target_id in enumerate(target_ids):
            by_pair = {}
            for pair_index in range(pair_count):
                record = load_target_result(
                    target_index, target_id,
                    f"candidate-{pair_index:06d}.json", pair_index,
                )
                if record is not None:
                    by_pair[pair_index] = record
            durable_records[target_id] = by_pair
            baseline = load_target_result(
                target_index, target_id, "baseline.json", None,
            )
            if baseline is not None:
                durable_baselines[target_id] = baseline
        if progress_counts:
            campaign["counts"] = progress_counts

    def pending_record(pair: Mapping[str, Any], target_id: str | None) -> dict[str, Any]:
        return {
            "step": pair.get("step"),
            "reference_index": pair.get("reference_index"),
            "state_sha256": pair.get("state_sha256"),
            "parent_sha256": pair.get("parent_sha256"),
            **({"target_id": target_id} if target_id else {}),
            "status": "target-gap",
            "terminal_disposition": "right-censored",
            "target_attempted": True,
            "target_result_status": pending_failure,
            "failure_class": pending_failure,
            "right_censored": True,
            "censor_stage": "target-execution",
            "censor_reason": pending_failure,
            **({"deadline_censored": True} if deadline_censored else {}),
            "comparison": {
                "status": "target-gap",
                "comparison_mode": "cross-backend",
                "differences": {"reason": "case-interrupted-before-target-replay-completed"},
            },
        }

    if mode == "mapped":
        existing = campaign.get("target_records_by_target")
        existing = existing if isinstance(existing, Mapping) else {}
        for target_id in target_ids:
            raw = existing.get(target_id, ())
            raw = raw if isinstance(raw, (list, tuple)) else ()
            records = [dict(item) for item in raw if isinstance(item, Mapping)]
            records_by_pair = {
                pair_index: record for pair_index, record in enumerate(records)
            }
            records_by_pair.update(durable_records.get(target_id, {}))
            rows = []
            for pair_index, pair in enumerate(pairs):
                if pair_index in records_by_pair:
                    record = records_by_pair[pair_index]
                    completed_candidate_records += 1
                    record.update({
                        "step": pair.get("step"),
                        "reference_index": pair.get("reference_index"),
                        "state_sha256": pair.get("state_sha256"),
                        "parent_sha256": pair.get("parent_sha256"),
                        "target_id": target_id,
                    })
                    rows.append(record)
                else:
                    rows.append(pending_record(pair, target_id))
                    pending_candidate_records += 1
            target_records_by_target[target_id] = rows
        campaign["target_records_by_target"] = target_records_by_target
        campaign["target"] = [
            dict(record)
            for target_id in target_ids
            for record in target_records_by_target[target_id]
        ]
    elif mode == "single":
        raw = campaign.get("target")
        raw = raw if isinstance(raw, (list, tuple)) else ()
        records = [dict(item) for item in raw if isinstance(item, Mapping)]
        target_id = target_ids[0] if target_ids else "target"
        records_by_pair = {
            pair_index: record for pair_index, record in enumerate(records)
        }
        records_by_pair.update(durable_records.get(target_id, {}))
        rows = []
        for pair_index, pair in enumerate(pairs):
            if pair_index in records_by_pair:
                record = records_by_pair[pair_index]
                completed_candidate_records += 1
                record.update({
                    "step": pair.get("step"),
                    "reference_index": pair.get("reference_index"),
                    "state_sha256": pair.get("state_sha256"),
                    "parent_sha256": pair.get("parent_sha256"),
                })
                rows.append(record)
            else:
                rows.append(pending_record(pair, target_id))
                pending_candidate_records += 1
        campaign["target"] = rows

    baseline_target_ids = {
        str(item) for item in plan.get("baseline_target_ids", ())
        if isinstance(item, str)
    }
    pending_baselines = 0
    if mode == "mapped":
        baselines = campaign.get("target_baselines_by_target")
        baselines = dict(baselines) if isinstance(baselines, Mapping) else {}
        for target_id in baseline_target_ids:
            if target_id in durable_baselines:
                baselines[target_id] = durable_baselines[target_id]
            if isinstance(baselines.get(target_id), Mapping):
                continue
            baselines[target_id] = {
                "target_id": target_id,
                "status": "target-gap",
                "target_result_status": pending_failure,
                "failure_class": pending_failure,
                "right_censored": True,
                "censor_stage": "target-baseline",
                "censor_reason": pending_failure,
                **({"deadline_censored": True} if deadline_censored else {}),
                "comparison": {
                    "status": "target-gap",
                    "comparison_mode": "cross-backend",
                    "differences": {"reason": pending_failure},
                },
            }
            pending_baselines += 1
        campaign["target_baselines_by_target"] = baselines
    elif mode == "single" and baseline_target_ids:
        target_id = target_ids[0] if target_ids else "target"
        if target_id in durable_baselines:
            campaign["target_baseline"] = durable_baselines[target_id]
        if not isinstance(campaign.get("target_baseline"), Mapping):
            campaign["target_baseline"] = {
                "status": "target-gap",
                "target_result_status": pending_failure,
                "failure_class": pending_failure,
                "right_censored": True,
                "censor_stage": "target-baseline",
                "censor_reason": pending_failure,
                **({"deadline_censored": True} if deadline_censored else {}),
                "comparison": {
                    "status": "target-gap",
                    "comparison_mode": "cross-backend",
                    "differences": {"reason": pending_failure},
                },
            }
            pending_baselines = 1

    chain = campaign.get("chain")
    chain = dict(chain) if isinstance(chain, Mapping) else {}
    events = [dict(item) for item in chain.get("events", ()) if isinstance(item, Mapping)]
    pair_by_step = {
        pair.get("step"): pair for pair in pairs if pair.get("step") is not None
    }
    for event in events:
        pair = pair_by_step.get(event.get("step"))
        if pair is None:
            continue
        if mode == "mapped":
            links = {}
            observations = {}
            pair_index = int(pair.get("pair_index", -1))
            for target_id in target_ids:
                records = target_records_by_target.get(target_id, ())
                if pair_index < 0 or pair_index >= len(records):
                    continue
                record = records[pair_index]
                links[target_id] = {"target_record_index": pair_index}
                observations[target_id] = {
                    key: record.get(key) for key in (
                        "status", "target_result_status", "failure_class",
                        "comparison", "deadline_censored",
                    ) if key in record
                }
            event["targets"] = links
            event["target_observations"] = observations
            event["target"] = None
        elif mode == "single":
            pair_index = int(pair.get("pair_index", -1))
            records = campaign.get("target", ())
            if 0 <= pair_index < len(records):
                event["target"] = {"target_record_index": pair_index}
    chain["events"] = events
    campaign["chain"] = chain

    missing = pending_candidate_records + pending_baselines
    counts = campaign.get("counts")
    counts = dict(counts) if isinstance(counts, Mapping) else {}
    if missing:
        counts["target_attempted"] = _framework_count(counts, "target_attempted") + missing
        counts["target_gap"] = _framework_count(counts, "target_gap") + missing
        if deadline_censored:
            counts["target_wall_clock_pending"] = (
                _framework_count(counts, "target_wall_clock_pending") + missing
            )
    campaign["counts"] = counts
    campaign["status"] = "target-gap" if missing else campaign.get("status", "reference-valid")
    if missing:
        campaign["reason"] = recovery_reason
    target_replay = campaign.get("target_replay")
    target_replay = dict(target_replay) if isinstance(target_replay, Mapping) else {}
    target_replay.update(
        status="right-censored" if missing else "complete",
        pending_pair_count=pair_count,
        pending_execution_count=missing,
        completed_record_count=completed_candidate_records,
        record_count=(sum(map(len, target_records_by_target.values()))
                      if mode == "mapped" else len(campaign.get("target", ()))),
        record_count_by_target={
            target_id: len(records)
            for target_id, records in target_records_by_target.items()
        },
        queue_persisted=queue_path is not None,
        **({"queue_path": queue_path} if queue_path is not None else {}),
    )
    campaign["target_replay"] = target_replay
    campaign["checkpoint_recovery"] = {
        "status": "right-censored" if missing else "recovered",
        "checkpoint_stage": checkpoint.get("checkpoint_stage"),
        "stop_reason": recovery_reason,
        "pending_target_execution_count": missing,
        "target_queue_persisted": queue_path is not None,
        **({"target_queue_path": queue_path} if queue_path is not None else {}),
    }
    if coverage_finalization_requested:
        campaign["framework_coverage"] = {
            "schema_version": "rq1-framework-coverage-v1",
            "status": "deferred", "artifact_complete": False,
            "reason": "reference-checkpoint-recovered",
        }
        campaign["finalization"] = {
            "status": "deferred", "coverage_enabled": True,
            "framework_coverage_status": "deferred",
            "reason": "reference-checkpoint-recovered",
        }
    else:
        campaign["framework_coverage"] = {
            "schema_version": "rq1-framework-coverage-v1",
            "status": "disabled", "artifact_complete": False,
            "reason": "coverage-disabled",
        }
        campaign["finalization"] = {
            "status": "complete", "coverage_enabled": False,
            "framework_coverage_status": "disabled",
        }
    return campaign


def _framework_terminal_case_campaign(
    record: dict[str, Any], outcome: str,
) -> dict[str, Any]:
    entry = record.get("entry") if isinstance(record.get("entry"), dict) else {}
    reason = record.get("censor_reason") or entry.get("reason")
    # Operator stops right-censor a case, but they are not target wall-clock
    # observations and must not enter that counter.
    counts = (
        {"target_wall_clock_pending": 1}
        if outcome == "right-censored" and reason in _FRAMEWORK_DEADLINE_CENSOR_REASONS
        else {}
    )
    gate = (
        "shared-wall-clock-deadline"
        if outcome == "right-censored" and reason in _FRAMEWORK_DEADLINE_CENSOR_REASONS else
        "external-signal"
        if outcome == "right-censored" and reason == "external-signal" else
        "case-process-terminal"
    )
    event = {
        "step": 0,
        "event_type": "case-terminal",
        "status": outcome,
        "gate": gate,
        "case_id": record.get("case_id"),
        "target_available": False,
        "reason": reason,
        "censor_stage": record.get("censor_stage"),
        "elapsed_s": record.get("elapsed_s"),
        "entry_status": entry.get("status"),
        "returncode": entry.get("returncode"),
        "termination": entry.get("termination"),
    }
    return {"chain": {"events": [event]}, "counts": counts, "target": []}


def _framework_case_chain_closed(case: dict[str, Any]) -> bool:
    chain = case.get("chain")
    return (
        isinstance(chain, dict)
        and chain.get("sealed") is True
        and chain.get("integrity_verified") is True
    )


def _framework_parallel_execution_complete(
    lanes: Any, chain: Any,
) -> bool:
    return (
        isinstance(lanes, list)
        and bool(lanes)
        and all(
            isinstance(lane, dict) and lane.get("execution_complete") is True
            for lane in lanes
        )
        and isinstance(chain, dict)
        and chain.get("sealed") is True
        and chain.get("integrity_verified") is True
    )


def _framework_campaign_reason(campaign: dict[str, Any]) -> str | None:
    """Keep the first concrete Target gap reason in the case summary."""
    target_records = campaign.get("target") if isinstance(campaign, dict) else None
    if not isinstance(target_records, list):
        return None
    for target in target_records:
        if not isinstance(target, dict) or target.get("status") not in {
            "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
        }:
            continue
        for name in ("failure_class", "target_result_status"):
            value = target.get(name)
            if isinstance(value, str) and value:
                return value
    return None


_FRAMEWORK_ONLINE_FEEDBACK_GAP_COUNTERS = (
    "profile_gap", "proposal_gap", "parameter_gap", "transport_gap",
    "reference_gap", "refresh_gap", "target_gap", "target_gap_event",
    "target_transport_gap", "target_wall_clock_pending",
)


def _framework_emi_active(campaign: Mapping[str, Any]) -> bool:
    """Require persisted EMI action evidence for a formal framework case."""
    if campaign.get("emi_enabled") is False:
        return False
    root_binding = campaign.get("root_binding")
    module_toggles = (
        root_binding.get("module_toggles")
        if isinstance(root_binding, Mapping) else None
    )
    if isinstance(module_toggles, Mapping) and module_toggles.get("emi") is False:
        return False
    chain = campaign.get("chain") if isinstance(campaign, Mapping) else None
    events = chain.get("events") if isinstance(chain, Mapping) else None
    # 新账本保存候选 ID；旧账本/最小 fixture 保存显式 local-rewrite kind。
    # 两种形式都表示 EMI 动作已经物化并持久化。
    return isinstance(events, list) and any(
        isinstance(event, Mapping)
        and isinstance(event.get("action"), Mapping)
        and (
            event["action"].get("candidate_id")
            or event["action"].get("kind") == "local-rewrite"
        )
        for event in events
    )


def _framework_small_model_rewrite_active(
    method: str, campaign: Mapping[str, Any],
) -> bool:
    """Require an accepted small-model rewrite for both Ours routes."""
    if method not in {"Ours-RVGEN-Direct", "Ours-Program-Full"}:
        return True
    root_binding = campaign.get("root_binding")
    module_toggles = (
        root_binding.get("module_toggles")
        if isinstance(root_binding, Mapping) else None
    )
    if isinstance(module_toggles, Mapping) and module_toggles.get("small_model") is False:
        return False
    root = campaign.get("root") if isinstance(campaign, Mapping) else None
    supply = root.get("supply") if isinstance(root, Mapping) else None
    hint = root.get("hint") if isinstance(root, Mapping) else None
    edits = hint.get("edits") if isinstance(hint, Mapping) else None
    return (
        isinstance(root, Mapping)
        and root.get("mode") == "model-on"
        and isinstance(supply, Mapping)
        and supply.get("status") == "accepted"
        and supply.get("accepted") == 1
        and isinstance(edits, list)
        and bool(edits)
    )


def _framework_online_feedback_ready(
    *, manifest_verified: bool, campaign_status: Any, entry_status: Any,
    counts: dict[str, Any], mcmc: bool = True, emi_active: bool = True,
    feedback_source: Any = "target",
    small_model_rewrite_active: bool = True,
) -> bool:
    """Report producer feedback as soon as its own reference loop is ready.

    In the formal reference-driven route, K1/QEMU is the feedback source.  A
    Target job is only an asynchronously queued observation and must not be a
    prerequisite for the next producer case.  The inline compatibility route
    still requires a Target observation because it executes that Target in the
    producer process.
    """
    reference_driven = feedback_source == "reference"
    target_feedback_ready = reference_driven or _framework_count(
        counts, "target_tested",
    ) > 0
    if (
        not mcmc
        or not manifest_verified
        or not reference_driven and campaign_status not in {
            "clean", "target-tested", "target-mismatch-candidate",
        }
        or entry_status != "passed"
        or not target_feedback_ready
        or _framework_count(counts, "mh_attempts") <= 0
        or not emi_active
        or not small_model_rewrite_active
    ):
        return False
    gap_counters = (
        tuple(name for name in _FRAMEWORK_ONLINE_FEEDBACK_GAP_COUNTERS
              if name not in {
                  "target_gap", "target_gap_event", "target_transport_gap",
                  "target_wall_clock_pending",
              })
        if reference_driven else _FRAMEWORK_ONLINE_FEEDBACK_GAP_COUNTERS
    )
    return not any(
        _framework_count(counts, name) != 0
        for name in gap_counters
    )


def _framework_formal_ready(
    *, method: str, manifest_verified: bool, execution_complete: bool,
    status: Any, coverage_enabled: bool, coverage_status: Any,
    coverage_artifact_complete: bool, online_feedback: bool,
    profile: Mapping[str, Any] | None,
    mcmc: bool = True,
) -> bool:
    """Apply the one formal gate shared by live runs and finalization."""
    valid_profile = (
        isinstance(profile, Mapping)
        and (
            type(profile.get("steps")) is int
            and profile.get("steps") > 0
            and profile.get("mcmc") is True
        )
    )
    return all((
        manifest_verified,
        execution_complete,
        status == "passed",
        coverage_enabled,
        coverage_status == "recorded",
        coverage_artifact_complete is True,
        mcmc is True,
        online_feedback is True,
        method not in {"Ours-RVGEN-Direct", "Ours-Program-Full"}
        or valid_profile,
    ))


def _framework_comparison_status(campaign_status: Any) -> str:
    if campaign_status in {"clean", "target-tested"}:
        return "equivalent"
    if campaign_status == "target-mismatch-candidate":
        return "mismatch"
    if campaign_status in {
        "target-gap", "target-probe-tested", "target-probe-gap",
        "case-skipped", "artifact-gap", "unsupported-isa",
        "reference-valid", "reference-gap", "profile-gap", "proposal-gap", "seed-miss",
        "transport-gap", "reference-rejected", "emi-rejected", "profiled",
    }:
        return "gap"
    return "unknown"


def _framework_aggregate_campaign_status(counts: Mapping[str, Any]) -> str:
    """保留只有 Target 基线已测试的闭环终态。"""
    return (
        "proposal-gap" if _framework_count(counts, "proposal_gap") else
        "artifact-gap" if _framework_count(counts, "artifact_gap") else
        "target-gap" if _framework_count(counts, "target_gap") else
        "target-mismatch-candidate"
        if _framework_count(counts, "target_mismatch_candidate") else
        "profile-gap" if _framework_count(counts, "profile_gap") else
        "seed-miss" if _framework_count(counts, "seed_miss") else
        "clean" if _framework_count(counts, "clean") else
        "target-tested" if _framework_count(counts, "target_tested") else
        "case-skipped" if _framework_count(counts, "case_skipped") else
        "reference-valid" if _framework_count(counts, "reference_valid") else
        "transport-gap" if _framework_count(counts, "transport_gap") else
        "reference-gap" if _framework_count(counts, "reference_gap") else
        "unknown"
    )


def _framework_parallel_case_record(
    case_dir: Path, *, method: str, feedback_target: str, backend: str,
    spec: dict[str, Any], profile: dict[str, Any], manifest_verified: bool,
    campaign: dict[str, Any], entry: dict[str, Any], lane_id: int,
    case_index: int, seed: int, root_case_index: int | None,
    target_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    campaign = campaign if isinstance(campaign, dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    target_coverage = campaign.get("target_coverage") \
        if isinstance(campaign.get("target_coverage"), dict) else None
    target_coverage_path = case_dir / "target-coverage.json"
    if target_coverage is None and target_coverage_path.is_file():
        try:
            target_coverage = load_json(target_coverage_path)
        except (OSError, TypeError, ValueError):
            target_coverage = None
    framework_coverage = campaign.get("framework_coverage") \
        if isinstance(campaign.get("framework_coverage"), dict) else None
    counts = campaign.get("counts") if isinstance(campaign.get("counts"), dict) else {}
    counts = dict(counts)
    target_wall_clock_pending = _framework_count(
        counts, "target_wall_clock_pending",
    ) > 0
    target_records_by_id = campaign.get("target_records_by_target")
    target_baselines_by_id = campaign.get("target_baselines_by_target")
    target_wall_clock_pending_by_target = None
    if isinstance(target_records_by_id, Mapping) or isinstance(target_baselines_by_id, Mapping):
        target_records_by_id = (
            target_records_by_id if isinstance(target_records_by_id, Mapping) else {}
        )
        target_baselines_by_id = (
            target_baselines_by_id if isinstance(target_baselines_by_id, Mapping) else {}
        )

        def target_censored(record: object) -> int:
            return int(
                isinstance(record, Mapping)
                and (
                    record.get("deadline_censored") is True
                    or record.get("failure_class") == "campaign-wall-clock-exhausted"
                )
            )

        target_wall_clock_pending_by_target = {}
        for target_id in set(target_records_by_id) | set(target_baselines_by_id):
            records = target_records_by_id.get(target_id, ())
            records = records if isinstance(records, (list, tuple)) else ()
            target_wall_clock_pending_by_target[str(target_id)] = sum(
                target_censored(record) for record in records
            ) + target_censored(target_baselines_by_id.get(target_id))
    # A direct campaign is authoritative.  The profile field remains
    # readable for legacy manifests that predate the direct mode.
    mcmc_enabled = (
        campaign.get("mcmc") is True
        if isinstance(campaign, Mapping) and "mcmc" in campaign
        else profile.get("mcmc") is True
    )
    emi_active = _framework_emi_active(campaign)
    small_model_rewrite_active = (
        _framework_small_model_rewrite_active(method, campaign)
        if method in {"Ours-RVGEN-Direct", "Ours-Program-Full"} else None
    )
    root_binding = campaign.get("root_binding")
    module_toggles = (
        root_binding.get("module_toggles")
        if isinstance(root_binding, Mapping) else None
    )
    module_toggles = dict(module_toggles) if isinstance(module_toggles, Mapping) else None
    seed_case = root_binding.get("seed_case") if isinstance(root_binding, Mapping) else None
    seed_case = seed_case if isinstance(seed_case, Mapping) else None
    online_feedback = _framework_online_feedback_ready(
        manifest_verified=manifest_verified,
        campaign_status=campaign.get("status"),
        entry_status=entry.get("status"),
        counts=counts, mcmc=mcmc_enabled, emi_active=emi_active,
        feedback_source=campaign.get("mcmc_feedback_source"),
        small_model_rewrite_active=(
            True if small_model_rewrite_active is None else small_model_rewrite_active
        ),
    )
    case_reason = (
        entry.get("reason")
        or campaign.get("reason")
        or _framework_campaign_reason(campaign)
    )
    if not campaign and not case_reason:
        case_reason = "framework-case-result-missing"
    effective_target_timeout = target_timeout_seconds
    if isinstance(campaign, dict) and campaign.get("target_timeout_seconds") is not None:
        effective_target_timeout = campaign["target_timeout_seconds"]
    return {
        "schema_version": "rq1-framework-case-v1",
        "case_id": f"lane-{lane_id:02d}-case-{case_index:06d}",
        "lane_id": lane_id,
        "case_index": case_index,
        "seed": seed,
        "root_case_index": root_case_index,
        "root_case_mode": (
            "pre-generated-rvdv-sequence"
            if method == "Ours-Program-Full" and seed_case is not None else
            "fresh-program-per-case"
            if method == "Ours-Program-Full"
            else "seeded-random-rvgen-root-pool"
        ),
        "seed_case": dict(seed_case) if seed_case is not None else None,
        "method": method,
        "feedback_target": feedback_target,
        "target_backend": backend,
        "target_timeout_seconds": effective_target_timeout,
        "route": spec.get("route"),
        "lane": spec.get("lane"),
        "steps": _framework_count(counts, "steps"),
        "step_budget": int(profile.get("steps", 0) or 0),
        "steps_per_case_budget": int(profile.get("steps", 0) or 0),
        "total_steps_completed": _framework_count(counts, "steps"),
        "mcmc": mcmc_enabled,
        "module_toggles": module_toggles,
        "mcmc_feedback_source": campaign.get("mcmc_feedback_source"),
        "mcmc_chain_mode": campaign.get("mcmc_chain_mode"),
        "target_execution_mode": campaign.get("target_execution_mode"),
        "target_replay": campaign.get("target_replay"),
        "emi_active": emi_active,
        "small_model_rewrite_active": small_model_rewrite_active,
        "mh_attempts": _framework_count(counts, "mh_attempts"),
        "accepted": _framework_count(counts, "accepted"),
        "rejected": _framework_count(counts, "mh_rejected"),
        "target_attempted": _framework_count(counts, "target_attempted"),
        "target_tested": _framework_count(counts, "target_tested"),
        "target_gap": _framework_count(counts, "target_gap"),
        "target_unsupported": _framework_count(counts, "target_unsupported"),
        "case_skipped": _framework_count(counts, "case_skipped"),
        "target_wall_clock_pending": _framework_count(
            counts, "target_wall_clock_pending"
        ),
        **({"target_wall_clock_pending_by_target": target_wall_clock_pending_by_target}
           if target_wall_clock_pending_by_target is not None else {}),
        "campaign_result_recorded": bool(campaign),
        # Target replay is a per-Target stream.  A pending Renode/RAX item is
        # not a reason to right-censor the reference case or hide completed
        # QEMU/LRSV cases; the pending amount is reported separately below.
        "target_replay_pending": target_wall_clock_pending,
        "right_censored": False,
        "censor_stage": None,
        "censor_reason": None,
        "censor_elapsed_s": None,
        "target_mismatch_candidate": _framework_count(
            counts, "target_mismatch_candidate"
        ),
        "reference_valid": _framework_count(counts, "reference_valid"),
        "counts": counts,
        "target_coverage": target_coverage,
        "framework_coverage": framework_coverage,
        "coverage_status": framework_coverage.get("status")
        if framework_coverage else "gap",
        "coverage_artifact_complete": framework_coverage.get("artifact_complete") is True
        if framework_coverage else False,
        "manifest_verified": manifest_verified,
        "campaign_status": campaign.get("status"),
        "comparison_status": _framework_comparison_status(campaign.get("status")),
        "reason": case_reason,
        "entry": {
            "status": entry.get("status"), "reason": entry.get("reason"),
            "returncode": entry.get("returncode"),
            "termination": entry.get("termination"),
        },
        "online_feedback": online_feedback,
        "status": "passed" if online_feedback else "partial" if campaign else "gap",
        "case_outcome": "campaign-result" if campaign else "unresolved",
    }


def _framework_verify_parallel_chain_artifacts(out: Path, method_dir: Path) -> bool:
    """验证并行 campaign 的每一条独立 case chain。"""
    try:
        manifest = load_json(out / "execution-manifest.json")
        parallel = load_json(method_dir / "parallel-result.json")
        integrity = load_json(method_dir / "parallel-integrity.json")
        seal = load_json(method_dir / "parallel-seal.json")
        if (
            parallel.get("method") != manifest.get("method_filter")
            or parallel.get("feedback_target") != manifest.get("feedback_target")
            or integrity.get("status") != "verified"
            or seal.get("status") != "sealed"
            or seal.get("sealed") is not True
            or seal.get("integrity_sha256") != sha256(method_dir / "parallel-integrity.json")
        ):
            return False
        lanes = parallel.get("lanes")
        if not isinstance(lanes, list):
            return False
        cases = [
            case for lane in lanes if isinstance(lane, dict)
            for case in lane.get("cases", ())
            if isinstance(case, dict)
        ]
        if not cases or integrity.get("case_count") != len(cases):
            return False
        if integrity.get("counts") != parallel.get("counts"):
            return False
        files = integrity.get("files")
        if not isinstance(files, dict):
            return False
        for relative, expected in files.items():
            path = out / str(relative)
            if not path.is_file() or sha256(path) != expected:
                return False
        for case in cases:
            case_dir = out / str(case.get("case_dir"))
            case_result = out / str(case.get("case_result"))
            if not case_dir.is_dir() or not case_result.is_file():
                return False
            if load_json(case_result) != case:
                return False
            if not _framework_verify_chain_artifacts(out, case_dir):
                return False
        return integrity.get("sealed_case_count") == sum(
            case.get("chain", {}).get("sealed") is True
            for case in cases if isinstance(case.get("chain"), dict)
        )
    except (OSError, TypeError, ValueError):
        return False


def _framework_write_parallel_chain_artifacts(
    out: Path, method_dir: Path, parallel: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    cases = [
        case for lane in parallel.get("lanes", ()) if isinstance(lane, dict)
        for case in lane.get("cases", ()) if isinstance(case, dict)
    ]
    tracked: dict[str, str | None] = {}
    coverage_summary = method_dir / "coverage" / "summary.json"
    if coverage_summary.is_file():
        tracked[str(coverage_summary.relative_to(out))] = sha256(coverage_summary)
    target_coverage = method_dir / "target-coverage.json"
    if target_coverage.is_file():
        tracked[str(target_coverage.relative_to(out))] = sha256(target_coverage)
    for case in cases:
        for relative in (
            case.get("case_result"),
            case.get("chain", {}).get("integrity") if isinstance(case.get("chain"), dict) else None,
            case.get("chain", {}).get("seal") if isinstance(case.get("chain"), dict) else None,
        ):
            if not isinstance(relative, str):
                continue
            path = out / relative
            if path.is_file():
                tracked[relative] = sha256(path)
    all_case_chains_verified = bool(cases) and all(
        isinstance(case.get("chain"), dict)
        and case["chain"].get("integrity_verified") is True
        for case in cases
    )
    integrity = {
        "schema_version": "rq1-framework-parallel-integrity-v1",
        "status": "verified" if manifest and all_case_chains_verified else "partial",
        "experiment_face": "framework",
        "method": parallel.get("method"),
        "feedback_target": parallel.get("feedback_target"),
        "case_count": len(cases),
        "sealed_case_count": sum(
            case.get("chain", {}).get("sealed") is True
            for case in cases if isinstance(case.get("chain"), dict)
        ),
        "counts": dict(parallel.get("counts", {})),
        "files": tracked,
        "identity": {
            key: manifest.get(key)
            for key in (
                "source_commit", "config_sha256", "image_id", "dependency_identity",
                "execution_plane", "network", "targets",
            ) if key in manifest
        },
    }
    integrity_path = method_dir / "parallel-integrity.json"
    write_json_replace(integrity_path, integrity)
    sealed = integrity["status"] == "verified" and integrity["sealed_case_count"] == len(cases)
    seal = {
        "schema_version": "rq1-framework-parallel-seal-v1",
        "sealed": sealed,
        "status": "sealed" if sealed else "partial",
        "experiment_face": "framework",
        "method": parallel.get("method"),
        "feedback_target": parallel.get("feedback_target"),
        "integrity": str(integrity_path.relative_to(out)),
        "integrity_sha256": sha256(integrity_path),
    }
    seal_path = method_dir / "parallel-seal.json"
    write_json_replace(seal_path, seal)
    verified = _framework_verify_parallel_chain_artifacts(out, method_dir)
    return {
        "mode": "parallel-lanes",
        "case_count": len(cases),
        "sealed_case_count": integrity["sealed_case_count"],
        "integrity": str(integrity_path.relative_to(out)),
        "seal": str(seal_path.relative_to(out)),
        "sealed": sealed,
        "integrity_verified": verified,
    }


def _framework_start_dotnet_coverage_server(
    out: Path, method_dir: Path, target_config: Mapping[str, Any], method: str,
) -> str | None:
    target_spec = target_config.get("target_spec")
    target_spec = target_spec if isinstance(target_spec, Mapping) else {}
    source = target_spec.get("source_coverage")
    if not isinstance(source, Mapping) or source.get("collector") != "dotnet":
        return None
    launcher = dep(source.get("launcher"))
    binary = dep(target_config.get("coverage_binary"))
    identity_binary = dep(target_config.get("binary_path"))
    if launcher is None or not launcher.is_file() or binary is None or identity_binary is None:
        return None

    batch_root = method_dir / "coverage-batches" / "T-RENODE" / "campaign"
    raw_dir = batch_root / "raw"
    source_dir = batch_root / "simulator-source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    source_config = dict(source)
    source_config.update({
        "profile": str(source_dir / "lcov.info"),
        "identity": str(source_dir / "identity.json"),
    })
    target_data = {
        "id": "T-RENODE", "kind": target_spec.get("kind"),
        "commit": target_spec.get("commit"),
        "coverage_binary": str(binary),
        "coverage_binary_sha256": target_spec.get("coverage_binary_sha256"),
        "source_coverage": source_config,
    }
    cache_dir = out / "coverage-binaries" / "T-RENODE"
    materialize_config = {
        "backend": target_config.get("backend"),
        "binary_path": str(binary), "identity_binary_path": str(identity_binary),
        "coverage_binary_cache": str(cache_dir),
        "coverage_binary_sha256": target_spec.get("coverage_binary_sha256"),
        "target": target_data, "source_coverage": source_config,
        "raw_dir": str(raw_dir),
    }
    runner_binary, _ = _coverage_target(materialize_config, identity_binary)
    include_files = [runner_binary]
    infrastructure = runner_binary.with_name("Infrastructure.dll")
    if infrastructure.is_file():
        include_files.append(infrastructure)

    session_id = (
        f"rq1-{str(method).replace('/', '-')}-{os.getpid()}-{time.time_ns()}"
    )
    report = raw_dir / "renode.cobertura.xml"
    log_path = batch_root / "dotnet-coverage-server.log"
    env = os.environ.copy()
    # Client IPC is resolved under TMPDIR; keep both ends outside the run root.
    env["TMPDIR"] = "/tmp"
    runtime = env.get("RQ1_DOTNET_RUNTIME")
    if runtime:
        env["DOTNET_ROOT"] = runtime
        env["PATH"] = runtime + os.pathsep + env.get("PATH", "")
    command = [
        str(launcher), "collect", "--server-mode", "--session-id", session_id,
        "--include-files", ",".join(map(str, include_files)),
        "--output", str(report), "--output-format", "cobertura", "--nologo",
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command, cwd=runner_binary.parent, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("dotnet-coverage server exited during startup")
            if "SessionId:" in log_path.read_text(encoding="utf-8", errors="replace"):
                _FRAMEWORK_DOTNET_COVERAGE_SERVERS[session_id] = {
                    "process": process, "launcher": launcher, "cwd": runner_binary.parent,
                    "env": env, "report": report,
                }
                return session_id
            time.sleep(0.05)
        raise TimeoutError("dotnet-coverage server startup timed out")
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise


def _framework_stop_dotnet_coverage_server(session_id: str) -> None:
    state = _FRAMEWORK_DOTNET_COVERAGE_SERVERS.get(session_id)
    if not isinstance(state, Mapping):
        return
    launcher = state["launcher"]
    process = state["process"]
    try:
        result = subprocess.run(
            [str(launcher), "shutdown", session_id, "--nologo", "--timeout", "60000"],
            cwd=state["cwd"], env=state["env"],
            capture_output=True, text=True, timeout=90, check=False,
        )
        process.wait(timeout=90)
        if result.returncode or process.returncode:
            raise RuntimeError("dotnet-coverage server shutdown failed")
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise
    finally:
        _FRAMEWORK_DOTNET_COVERAGE_SERVERS.pop(session_id, None)


def _framework_coverage_case_attempted(
    case_dir: Path, target_id: str, *, single_target: bool,
) -> bool:
    try:
        campaign = load_json(case_dir / "campaign-result.json")
    except (OSError, TypeError, ValueError):
        # A time-boxed raw-only case may be interrupted after the resident
        # Target has written its sidecar but before the producer checkpoint is
        # materialized.  The Target-owned raw file is still authoritative.
        campaign = {}
    if not isinstance(campaign, Mapping):
        campaign = {}
    records_by_target = campaign.get("target_records_by_target")
    baselines_by_target = campaign.get("target_baselines_by_target")
    if isinstance(records_by_target, Mapping) or isinstance(baselines_by_target, Mapping):
        records = records_by_target.get(target_id, ()) \
            if isinstance(records_by_target, Mapping) else ()
        baseline = baselines_by_target.get(target_id) \
            if isinstance(baselines_by_target, Mapping) else None
    else:
        records, baseline = (), None
    if bool(records) or isinstance(baseline, Mapping):
        return True
    if single_target and (
        bool(campaign.get("target", ()))
        or isinstance(campaign.get("target_baseline"), Mapping)
    ):
        return True
    # Raw-only producer campaigns intentionally contain no target records.
    # The resident Target raw sidecar is the authoritative evidence that this
    # case reached the simulator and must count toward batch expected cases.
    try:
        method_root = case_dir.resolve().parents[3]
        case_relative = case_dir.resolve().relative_to(method_root)
        safe_target = "".join(
            char if char.isalnum() or char in ".-_" else "_"
            for char in str(target_id)
        ) or "unknown"
        raw_root = (
            method_root / "target-queues" / safe_target
            / "raw" / "cases" / case_relative
        )
    except (OSError, ValueError, IndexError):
        return False
    return any(
        path.is_file()
        and not path.name.endswith(".raw-artifact-manifest.json")
        and (path.name == "baseline.json" or path.name.startswith("candidate-"))
        for path in raw_root.glob("*.json")
    )


def _framework_resolve_persisted_run_paths(
    value: Any, out: Path,
) -> Any:
    """Resolve container run paths after a run is opened on the host."""
    if isinstance(value, Mapping):
        return {
            key: _framework_resolve_persisted_run_paths(item, out)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_framework_resolve_persisted_run_paths(item, out) for item in value]
    if not isinstance(value, str):
        return value
    prefix = Path("/opt/runs") / out.name
    try:
        return str(out / Path(value).relative_to(prefix))
    except ValueError:
        return value


def _framework_finalize_source_coverage_batches(
    out: Path, method: str, feedback_target: str, cases: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    """Run source collectors once per target batch after all case collectors stop."""
    config_dir = out / "framework-coverage-configs" / method.replace("/", "-")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        try:
            lane_id = int(case.get("lane_id", 0))
            case_index = int(case.get("case_index", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        config_path = config_dir / f"lane-{lane_id:02d}-case-{case_index:06d}.json"
        try:
            case_config = load_json(config_path)
        except (OSError, TypeError, ValueError):
            continue
        target_items = case_config.get("targets") \
            if feedback_target == "all" and isinstance(case_config, Mapping) else None
        if isinstance(target_items, Mapping):
            items = target_items.items()
        else:
            items = ((feedback_target, case_config),)
        case_dir_value = case.get("case_dir")
        if not isinstance(case_dir_value, str):
            continue
        case_dir = out / case_dir_value
        for target_id, target_item in items:
            if not isinstance(target_id, str) or not isinstance(target_item, Mapping):
                continue
            config = target_item.get("coverage_config")
            config = config if isinstance(config, Mapping) else target_item
            config = _framework_resolve_persisted_run_paths(dict(config), out)
            summary_value = config.get("coverage_batch_summary_path")
            if not isinstance(summary_value, str) or not summary_value:
                continue
            key = (target_id, summary_value)
            group = groups.setdefault(key, {
                "config": dict(config), "case_count": 0,
            })
            # A published-but-pending Target job has no raw sidecar and must
            # not become a source-coverage input or expected case.
            attempted = _framework_coverage_case_attempted(
                case_dir, target_id, single_target=feedback_target != "all",
            )
            group["case_count"] += int(attempted)

    finalized: dict[str, dict[str, object]] = {}
    for (_target_id, summary_value), group in sorted(groups.items()):
        config = group["config"]
        session_id = config.get("coverage_batch_session_id")
        if isinstance(session_id, str) and session_id:
            try:
                _framework_stop_dotnet_coverage_server(session_id)
            except Exception as error:
                config["coverage_batch_session_shutdown_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        attempted = int(group["case_count"])
        expected_inputs = (
            int(attempted > 0)
            if config.get("dotnet_coverage_session_mode") == "server"
            else attempted
        )
        try:
            finalize_framework_source_coverage_batch(
                config,
                expected_coverage_cases=attempted,
                expected_collection_inputs=expected_inputs,
            )
        except Exception as error:
            summary_path = Path(summary_value)
            error_reason = f"framework-coverage-batch-error:{type(error).__name__}"
            target_data = config.get("target")
            target_stratum = (
                target_data.get("stratum")
                if isinstance(target_data, Mapping) else config.get("stratum")
            )
            write_json_replace(summary_path, {
                "schema_version": "rq1-framework-source-coverage-batch-v1",
                "status": "gap",
                "reason": error_reason,
                "batch_id": config.get("coverage_batch_id"),
                "target": _target_id,
                "case_count": attempted,
                "source_targets": [{
                    "method": config.get("method"),
                    "route": config.get("route"),
                    "lane": config.get("lane"),
                    "stratum": target_stratum,
                    "target": _target_id,
                    "scope": "target-batch",
                    "batch_id": config.get("coverage_batch_id"),
                    "cases": attempted,
                    "simulator_source_coverage": {
                        "status": "gap", "reason": error_reason,
                    },
                }],
            })
        finally:
            _framework_write_batch_raw_manifest(out, _target_id, config)
        if attempted:
            target_result = finalized.setdefault(_target_id, {
                "summary_paths": [],
            })
            target_result["summary_paths"].append(Path(summary_value))
    return finalized


def _framework_write_batch_raw_manifest(
    out: Path, target_id: str, config: Mapping[str, Any],
) -> None:
    """Index final Target raw files after the resident session has flushed."""
    raw_value = config.get("coverage_batch_raw_dir") or config.get("raw_dir")
    if not isinstance(raw_value, str) or not raw_value:
        return
    raw_dir = Path(raw_value)
    if not raw_dir.is_absolute():
        raw_dir = out / raw_dir
    try:
        raw_dir = raw_dir.resolve()
        raw_dir.relative_to(out.resolve())
    except (OSError, ValueError):
        return
    files: list[dict[str, Any]] = []
    source = config.get("source_coverage")
    collector = str(source.get("collector", "gcov")) \
        if isinstance(source, Mapping) else "gcov"

    def is_runtime_input(path: Path) -> bool:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if collector in {"gcov", "lcov"}:
            return (
                path.name == GCOV_FLUSH_INCOMPLETE_MARKER
                or (path.name.endswith(".gcda")
                    and not path.name.endswith(".tmp.gcda") and size > 0)
            )
        if collector == "llvm":
            return path.name.endswith(".profraw") and size > 0
        if collector == "dotnet":
            return path.name.endswith((
                ".coverage", ".cobertura.xml", ".cobertura.xml.gz",
            )) and size > 0
        return False

    runtime_file_count = 0
    if raw_dir.is_dir():
        for path in sorted(item for item in raw_dir.rglob("*") if item.is_file()):
            # 编译器临时对象不是 coverage 输入，可能在 job manifest 写入后被清理。
            if path.suffix == ".o":
                continue
            try:
                relative = path.resolve().relative_to(out.resolve())
                if is_runtime_input(path):
                    runtime_file_count += 1
                files.append({
                    "path": str(relative),
                    "kind": path.suffix.lstrip(".") or "raw",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "flush_status": "complete",
                    "source": "coverage-batch",
                })
            except (OSError, ValueError):
                continue
    manifest_path = raw_dir.parent / "raw-artifact-manifest.json"
    try:
        relative_manifest = str(manifest_path.resolve().relative_to(out.resolve()))
        relative_raw = str(raw_dir.relative_to(out.resolve()))
    except (OSError, ValueError):
        return
    write_json_replace(manifest_path, {
        "schema_version": "rq1-target-raw-artifact-manifest-v1",
        "target_id": target_id,
        "batch_id": config.get("coverage_batch_id"),
        # Static .gcno notes are collector setup, not evidence that a Target
        # job ran.  A pending-only batch must stay ``missing`` even if an old
        # collector left those notes under the batch root.
        "status": "raw-ready" if runtime_file_count else "missing",
        "files": files,
        "file_count": len(files),
        "recorded_file_count": runtime_file_count,
        "runtime_input_file_count": runtime_file_count,
        "coverage_raw_roots": [relative_raw],
        "flush_status": "complete",
        "finalized_after_session_close": True,
        "path": relative_manifest,
    })


def _framework_finalize_raw_only_batches(
    out: Path, method: str, feedback_target: str,
    parallel: Mapping[str, Any], *, include_guest_metrics: bool,
) -> dict[str, Any] | None:
    """Finalize stable raw-only coverage after drain or a closed stop."""
    queue_completion = parallel.get("target_queue_completion")
    if not isinstance(queue_completion, Mapping):
        return None
    targets = queue_completion.get("targets")
    stopped_after_close = (
        queue_completion.get("producer_closed") is True
        and isinstance(targets, Mapping) and bool(targets)
        and all(
            isinstance(item, Mapping)
            and item.get("worker_status") in {"stopped", "finished"}
            for item in targets.values()
        )
    )
    if queue_completion.get("drained") is not True and not stopped_after_close:
        return None
    lanes = parallel.get("lanes")
    if not isinstance(lanes, list):
        return None
    cases = [
        case for lane in lanes if isinstance(lane, Mapping)
        for case in lane.get("cases", ()) if isinstance(case, dict)
    ]
    finalized_batches = _framework_finalize_source_coverage_batches(
        out, method, feedback_target, cases,
    )
    method_dir = out / "framework" / method.replace("/", "-")
    if feedback_target == "all":
        summaries = {
            target: merge_framework_raw_coverage_batches(
                method_dir / "targets" / target,
                [
                    Path(path) for path in (
                        finalized_batches.get(target, {}).get("summary_paths", ())
                        if isinstance(finalized_batches.get(target), Mapping) else ()
                    )
                ],
                target,
                include_guest_metrics=include_guest_metrics,
            )
            for target in TARGETS
        }
        all_recorded = all(
            item.get("status") == "recorded"
            and item.get("artifact_complete") is True
            for item in summaries.values()
        )
        any_evidence = any(
            item.get("status") in {"recorded", "partial"}
            for item in summaries.values()
        )
        return {
            "schema_version": "rq1-framework-coverage-matrix-v1",
            "status": "recorded" if all_recorded else "partial" if any_evidence else "gap",
            "artifact_complete": all_recorded,
            "targets": summaries,
        }
    info = finalized_batches.get(feedback_target, {})
    summary = merge_framework_raw_coverage_batches(
        method_dir,
        [
            Path(path) for path in (
                info.get("summary_paths", ())
                if isinstance(info, Mapping) else ()
            )
        ],
        feedback_target,
        include_guest_metrics=include_guest_metrics,
    )
    return dict(summary) if isinstance(summary, Mapping) else None


def _assemble_framework_parallel_campaign(
    out: Path, method_dir: Path, *, method: str, feedback_target: str,
    backend: str,
    profile: dict[str, Any], manifest: dict[str, Any], lane_records: list[dict[str, Any]],
    target_timeout_seconds: int | None, parallel_lanes: int,
    target_execution_parallelism: int,
    case_queue_capacity: int | None = None,
    case_count_limit: int | None = None,
    case_finalization_mode: str = "inline",
    include_guest_metrics: bool = True,
    raw_coverage_only: bool = False,
    ablation_mode: bool | None = None,
    active_execution_seconds: float | None = None,
    paused_seconds: float = 0.0,
    segment_count: int = 1,
) -> dict[str, Any]:
    """Build the shared parallel summary from durable lane checkpoints."""
    if ablation_mode is None:
        try:
            recorded_toggles = load_json(method_dir / "module-toggles.json")
        except (OSError, TypeError, ValueError):
            recorded_toggles = None
        toggle_names = ("small_model", "emi", "mcmc")
        ablation_mode = (
            isinstance(recorded_toggles, Mapping)
            and all(type(recorded_toggles.get(name)) is bool for name in toggle_names)
            and not all(recorded_toggles[name] for name in toggle_names)
        )
    lane_records = sorted(lane_records, key=lambda item: int(item.get("lane_id", 0)))
    cases = [
        case for lane in lane_records
        for case in lane.get("cases", ()) if isinstance(case, dict)
    ]
    aggregate_counts: Counter[str] = Counter()
    for case in cases:
        counts = case.get("counts")
        if not isinstance(counts, dict):
            continue
        for name, value in counts.items():
            if isinstance(name, str) and name:
                aggregate_counts[name] += _nonnegative_int(value, default=0)
    counts = dict(aggregate_counts)
    case_count = len(cases)
    campaign_result_count = sum(
        case.get("campaign_result_recorded") is True for case in cases
    )
    right_censored_case_count = sum(
        case.get("right_censored") is True for case in cases
    )
    campaign_finalization_pending_case_count = sum(
        case.get("campaign_finalization_pending") is True for case in cases
    )
    case_failure_count = sum(
        case.get("case_outcome") in {"case-timeout", "case-failure"}
        for case in cases
    )
    completed_cases = [
        case for case in cases
        if case.get("campaign_result_recorded") is True
        and case.get("right_censored") is not True
    ]
    accepted_per_case = _framework_accepted_per_case(cases)
    feedback_cases = [
        case for case in cases if case.get("case_outcome") != "right-censored"
    ]
    mcmc_active = bool(feedback_cases) and all(
        case.get("mcmc") is True for case in feedback_cases
    )
    def _case_mode(name: str) -> str | None:
        values = {
            str(case.get(name)) for case in feedback_cases
            if isinstance(case.get(name), str) and case.get(name)
        }
        return next(iter(values)) if len(values) == 1 else "mixed" if values else None

    mcmc_feedback_source = _case_mode("mcmc_feedback_source")
    mcmc_chain_mode = _case_mode("mcmc_chain_mode")
    target_execution_mode = _case_mode("target_execution_mode")
    online_feedback = bool(feedback_cases) and all(
        case.get("online_feedback") is True for case in feedback_cases
    )
    execution_complete = bool(lane_records) and all(
        lane.get("execution_complete") is True for lane in lane_records
    )
    lane_stop_reasons = [
        {"lane_id": lane.get("lane_id"), "reason": lane.get("stop_reason")}
        for lane in lane_records
    ]
    distinct_stop_reasons = {lane.get("stop_reason") for lane in lane_records}
    stop_reason = (
        next(iter(distinct_stop_reasons)) if len(distinct_stop_reasons) == 1
        else "mixed-stop-reasons"
    )
    aggregate_campaign_status = _framework_aggregate_campaign_status(counts)
    case_dirs = [
        out / str(case["case_dir"])
        for case in cases if isinstance(case.get("case_dir"), str)
    ]
    if raw_coverage_only:
        framework_coverage = {
            "schema_version": (
                "rq1-framework-coverage-matrix-v1"
                if feedback_target == "all" else "rq1-framework-coverage-v1"
            ),
            "status": "deferred" if simulator_coverage_enabled() else "disabled",
            "artifact_complete": False,
            "reason": "raw-coverage-only" if simulator_coverage_enabled()
            else "coverage-disabled",
            "data_mode": "simulator-raw-evidence",
            "raw_evidence": {
                "coverage_batches": str(
                    (method_dir / "coverage-batches").relative_to(out)
                ),
                "case_configs": str(
                    (out / "framework-coverage-configs" / method.replace("/", "-")).relative_to(out)
                ),
            },
        }
    elif simulator_coverage_enabled():
        _framework_finalize_source_coverage_batches(
            out, method, feedback_target, cases,
        )
        if feedback_target == "all":
            target_coverage_summaries = {
                target: merge_framework_coverage_summaries(
                    method_dir / "targets" / target,
                    case_dirs,
                    target,
                    include_guest_metrics=include_guest_metrics,
                )
                for target in TARGETS
            }
            all_recorded = all(
                item.get("status") == "recorded"
                and item.get("artifact_complete") is True
                for item in target_coverage_summaries.values()
            )
            any_evidence = any(
                item.get("status") in {"recorded", "partial"}
                for item in target_coverage_summaries.values()
            )
            framework_coverage = {
                "schema_version": "rq1-framework-coverage-matrix-v1",
                "status": "recorded" if all_recorded else "partial" if any_evidence else "gap",
                "artifact_complete": all_recorded,
                "targets": target_coverage_summaries,
            }
        else:
            framework_coverage = merge_framework_coverage_summaries(
                method_dir, case_dirs, feedback_target,
                include_guest_metrics=include_guest_metrics,
            )
    else:
        framework_coverage = {
            "schema_version": "rq1-framework-coverage-v1",
            "status": "disabled", "artifact_complete": False,
            "reason": "coverage-disabled",
        }
    target_coverage = (
        {
            "schema": "target-coverage-matrix-v1",
            "status": "deferred",
            "reason": "raw-coverage-only",
            "case_count": case_count,
        }
        if raw_coverage_only else _framework_merge_target_coverage(case_dirs, case_count)
    )
    if not raw_coverage_only:
        write_json_replace(method_dir / "target-coverage.json", target_coverage)
    # Formal producer never uses Target progress as an admission gate, but it
    # exposes the resident services' durable counters for observability.
    target_progress = _framework_target_progress_snapshot(
        method_dir,
        TARGETS if feedback_target == "all" else (feedback_target,),
    )
    target_queue_completion = _framework_target_queue_completion(
        method_dir,
        TARGETS if feedback_target == "all" else (feedback_target,),
    ) if raw_coverage_only else {
        "schema_version": "rq1-target-queue-completion-v1",
        "status": "not-applicable",
        "producer_closed": False,
        "pending_count": 0,
        "worker_errors": {},
        "targets": target_progress,
    }
    # ``execution_complete`` describes the reference producer and its
    # MCMC/EMI chains only. Target queues are independent consumers; their
    # backlog is reported in ``target_queue_completion`` and must not turn a
    # completed producer into a blocked campaign.
    target_ids_for_record = TARGETS if feedback_target == "all" else (feedback_target,)
    # The producer records the deployment boundary, not a simulator handshake.
    # Target services own their process identity and raw execution status.
    session_mode = "external-target-service" if raw_coverage_only else "adapter-callback"
    session_modes = {
        str(target_id): session_mode for target_id in target_ids_for_record
    }
    target_session_mode = session_mode
    parallel = {
        "schema_version": "rq1-framework-parallel-v1",
        "status": (
            "passed"
            if execution_complete and case_failure_count == 0
            and (online_feedback or ablation_mode)
            else "partial" if cases else "gap"
        ),
        "ablation_mode": ablation_mode,
        "method": method, "feedback_target": feedback_target,
        "target_backend": backend, "parallel_lanes": parallel_lanes,
        "case_count_limit": case_count_limit,
        "target_timeout_seconds": target_timeout_seconds,
        "reference_parallel_limit": NATIVE_REFERENCE_PARALLELISM,
        "resource_policy": (
            "one reference-driven MCMC chain per case; native-rv64/K1 reference "
            f"requests share a {NATIVE_REFERENCE_PARALLELISM}-slot cross-container gate; "
            "case admission is per-lane after the reference checkpoint, with no "
            "global case slot held by Target replay; the framework reads the "
            "K1/QEMU reference result before EMI/MCMC; after the reference "
            "checkpoint, Target jobs are published to independent per-target "
            "queues and never gate the next case"
        ),
        "target_execution_parallelism": target_execution_parallelism,
        "case_queue_capacity": case_queue_capacity,
        "target_queue": "independent-per-target-queue",
        "target_service": (
            "external-per-target-services"
            if raw_coverage_only else "inline-compatibility"
        ),
        "target_session_mode": target_session_mode,
        "target_session_modes": session_modes,
        "target_consumer_ownership": (
            "external" if raw_coverage_only else "case-process-compatibility"
        ),
        "target_progress": target_progress,
        "target_queue_completion": target_queue_completion,
        "case_finalization_mode": case_finalization_mode,
        "raw_coverage_only": raw_coverage_only,
        "active_execution_seconds": active_execution_seconds,
        "paused_seconds": round(paused_seconds, 6),
        "segment_count": segment_count,
        "case_transition": "new-case-after-reference-chain-checkpoint",
        "seed_policy": "base_seed + lane_id * 1000003 + case_index",
        "lanes": lane_records, "case_count": case_count,
        "campaign_result_count": campaign_result_count,
        "completed_case_count": len(completed_cases),
        "right_censored_case_count": right_censored_case_count,
        "campaign_finalization_pending_case_count": (
            campaign_finalization_pending_case_count
        ),
        "case_failure_count": case_failure_count,
        "counts": counts,
        "steps": _framework_count(counts, "steps"),
        "reference_chain_steps": _framework_count(counts, "steps"),
        "step_budget": int(profile.get("steps", 0) or 0),
        "steps_per_case_budget": int(profile.get("steps", 0) or 0),
        "total_steps_completed": _framework_count(counts, "steps"),
        "mcmc_feedback_source": mcmc_feedback_source,
        "mcmc_chain_mode": mcmc_chain_mode,
        "target_execution_mode": target_execution_mode,
        "accepted_per_case": accepted_per_case,
        "target_wall_clock_pending": _framework_count(
            counts, "target_wall_clock_pending"
        ),
        "campaign_status": aggregate_campaign_status,
        "comparison_status": _framework_comparison_status(aggregate_campaign_status),
        "target_coverage": target_coverage,
        "mcmc": mcmc_active,
        "online_feedback": online_feedback,
        "framework_coverage": framework_coverage,
        "coverage_status": framework_coverage.get("status"),
        "coverage_artifact_complete": framework_coverage.get("artifact_complete") is True,
        "execution_complete": execution_complete,
        "stop_reason": stop_reason,
        "lane_stop_reasons": lane_stop_reasons,
    }
    method_dir.mkdir(parents=True, exist_ok=True)
    chain = {
        "mode": "parallel-lanes",
        "case_count": case_count,
        "sealed_case_count": sum(
            case.get("chain", {}).get("sealed") is True
            for case in cases if isinstance(case.get("chain"), dict)
        ),
        "integrity": str((method_dir / "parallel-integrity.json").relative_to(out)),
        "seal": str((method_dir / "parallel-seal.json").relative_to(out)),
        "sealed": bool(cases) and all(
            case.get("chain", {}).get("sealed") is True
            for case in cases if isinstance(case.get("chain"), dict)
        ),
        "integrity_verified": False,
    }
    parallel["chain"] = chain
    write_json_replace(method_dir / "parallel-result.json", parallel)
    parallel["chain"] = _framework_write_parallel_chain_artifacts(
        out, method_dir, parallel, manifest,
    )
    write_json_replace(method_dir / "parallel-result.json", parallel)
    return parallel


def _recover_framework_parallel_campaign(
    out: Path, config: dict[str, Any], manifest: dict[str, Any], *,
    recovery_reason: str | None = None,
) -> dict[str, Any] | None:
    """Rebuild the parallel summary from durable lane and case checkpoints."""
    method = manifest.get("method_filter")
    feedback_target = manifest.get("feedback_target")
    if not isinstance(method, str) or not isinstance(feedback_target, str):
        return None
    specs = {
        item.get("id"): item for item in config.get("framework_methods", ())
        if isinstance(item, dict)
    }
    spec = specs.get(method)
    if not isinstance(spec, dict):
        return None
    profile = spec.get("generator_profile")
    profile = profile if isinstance(profile, dict) else {}
    try:
        lane_count = _framework_parallel_lanes(profile)
        case_count_limit = _framework_case_count(profile)
    except (TypeError, ValueError):
        return None
    backend = (
        "multi-target" if feedback_target == "all"
        else FRAMEWORK_TARGET_BACKENDS.get(feedback_target)
    )
    if not isinstance(backend, str):
        return None
    method_dir = out / "framework" / method.replace("/", "-")
    # framework-run persists raw coverage inputs for offline aggregation.  A
    # stop before parallel-result.json/run-result.json is written must retain
    # that mode during host recovery as well.
    raw_coverage_only = manifest.get("action") == "framework-run"
    previous_active_seconds = None
    previous_paused_seconds = 0.0
    previous_segment_count = 1
    try:
        previous_parallel = load_json(method_dir / "parallel-result.json")
    except (OSError, TypeError, ValueError):
        previous_parallel = {}
    if isinstance(previous_parallel, Mapping):
        raw_coverage_only = raw_coverage_only or (
            previous_parallel.get("raw_coverage_only") is True
        )
        previous_active_seconds = previous_parallel.get("active_execution_seconds")
        previous_paused_seconds = previous_parallel.get("paused_seconds", 0.0)
        previous_segment_count = previous_parallel.get("segment_count", 1)
    if not raw_coverage_only:
        try:
            previous_run = load_json(out / "run-result.json")
        except (OSError, TypeError, ValueError):
            previous_run = {}
        raw_coverage_only = (
            isinstance(previous_run, Mapping)
            and previous_run.get("raw_coverage_only") is True
        )
    lane_records: list[dict[str, Any]] = []
    found_lane = False
    base_seed = _nonnegative_int(config.get("experiment_seed", 303), default=303)
    try:
        _, manifest_verified = _framework_manifest(
            out, method=method, feedback_target=feedback_target,
        )
    except (OSError, TypeError, ValueError):
        manifest_verified = False

    def case_index_from(path: Path) -> int | None:
        match = re.fullmatch(r"case-(\d{6})", path.name)
        return int(match.group(1)) if match else None

    for lane_id in range(lane_count):
        lane_dir = method_dir / "lanes" / f"lane-{lane_id:02d}"
        lane_path = lane_dir / "lane-result.json"
        lane = {}
        if lane_path.is_file():
            try:
                lane = load_json(lane_path)
            except (OSError, TypeError, ValueError):
                lane = {}
        if not isinstance(lane, dict):
            lane = {}
        if lane_path.is_file():
            found_lane = True
        lane.setdefault("lane_id", lane_id)
        try:
            recorded_cases = lane.get("cases", ())
        except AttributeError:
            recorded_cases = ()
        cases: list[dict[str, Any]] = []
        recorded_dirs: set[Path] = set()
        for item in recorded_cases:
            if not isinstance(item, dict):
                continue
            case = dict(item)
            relative = case.get("case_result")
            if isinstance(relative, str):
                try:
                    persisted = load_json(out / relative)
                except (OSError, TypeError, ValueError):
                    persisted = None
                if isinstance(persisted, dict):
                    case = persisted
            # Older Program-Full artifacts recorded the terminal status but
            # omitted the corresponding seed_miss counter. Recover the
            # deterministic derived count from the persisted status.
            case_dir = case.get("case_dir")
            campaign_path = out / str(case_dir) / "campaign-result.json" \
                if isinstance(case_dir, str) else None
            try:
                campaign = load_json(campaign_path) if campaign_path else {}
            except (OSError, TypeError, ValueError):
                campaign = {}
            if (
                isinstance(campaign, dict)
                and campaign.get("status") == "seed-miss"
                and isinstance(case.get("counts"), dict)
                and not _framework_count(case["counts"], "seed_miss")
            ):
                case["counts"] = {**case["counts"], "seed_miss": 1}
                case["derived_count_repair"] = "seed-miss-status-v1"
                if isinstance(relative, str):
                    write_json_replace(out / relative, case)
            if isinstance(case_dir, str):
                recorded_dirs.add((out / case_dir).resolve())
            cases.append(case)

        # The runner creates the case directory before launching the child.
        # A stop between that mkdir and the lane checkpoint used to hide the
        # tail case from every aggregate. Recover it as a real case when its
        # result exists, otherwise as an explicit right-censored case.
        cases_dir = lane_dir / "cases"
        orphan_dirs = sorted(
            (path for path in cases_dir.glob("case-*") if path.is_dir()),
            key=lambda path: path.name,
        ) if cases_dir.is_dir() else []
        for case_dir in orphan_dirs:
            case_index = case_index_from(case_dir)
            if case_index is None or case_dir.resolve() in recorded_dirs:
                continue
            found_lane = True
            case_result_path = case_dir / "case-result.json"
            try:
                recovered_case = load_json(case_result_path)
            except (OSError, TypeError, ValueError):
                recovered_case = None
            if isinstance(recovered_case, dict):
                case = recovered_case
            else:
                campaign_path = case_dir / "campaign-result.json"
                try:
                    campaign = load_json(campaign_path)
                except (OSError, TypeError, ValueError):
                    campaign = {}
                checkpoint = _framework_read_chain_checkpoint(
                    case_dir / "reference-chain-checkpoint.json",
                )
                if not _framework_campaign_core_recorded(campaign) and checkpoint is not None:
                    campaign = _framework_recover_chain_checkpoint(
                        checkpoint,
                        coverage_finalization_requested=not raw_coverage_only,
                        case_dir=case_dir,
                        stop_reason=recovery_reason or lane.get("stop_reason"),
                    )
                    write_json_replace(campaign_path, campaign)
                seed = base_seed + lane_id * _FRAMEWORK_SEED_STRIDE + case_index
                root_case_index = (
                    lane_id + case_index * lane_count
                    if method == "Ours-RVGEN-Direct" else None
                )
                if _framework_campaign_core_recorded(campaign):
                    if raw_coverage_only:
                        campaign = _framework_raw_only_campaign(
                            campaign,
                            coverage_enabled=manifest.get("coverage_enabled") is True,
                        )
                        write_json_replace(campaign_path, campaign)
                    case = _framework_parallel_case_record(
                        case_dir, method=method, feedback_target=feedback_target,
                        backend=backend, spec=spec, profile=profile,
                        manifest_verified=manifest_verified, campaign=campaign,
                        entry={
                            "status": "gap",
                            "reason": recovery_reason or "recovery-case-record-missing",
                            "termination": "recovery",
                        }, lane_id=lane_id, case_index=case_index,
                        seed=seed, root_case_index=root_case_index,
                        target_timeout_seconds=_configured_target_timeout(config, None),
                    )
                    case["recovery"] = "orphan-case-record-recovered"
                    case["case_outcome"] = "campaign-result"
                    case["chain"] = _framework_write_chain_artifacts(
                        out, case_dir, campaign, case, manifest,
                    )
                else:
                    lane_stop_reason = lane.get("stop_reason")
                    lane_stop_reason = lane_stop_reason or recovery_reason
                    censor_reason = (
                        "external-signal" if lane_stop_reason == "external-signal" else
                        "shared-deadline-while-case-running"
                        if lane_stop_reason == "deadline" else
                        "recovery-case-record-missing"
                    )
                    case = _framework_parallel_case_record(
                        case_dir, method=method, feedback_target=feedback_target,
                        backend=backend, spec=spec, profile=profile,
                        manifest_verified=manifest_verified, campaign={},
                        entry={
                            "status": "timeout", "returncode": 124,
                            "reason": "recovery-case-record-missing",
                            "termination": "recovery",
                        }, lane_id=lane_id, case_index=case_index,
                        seed=seed, root_case_index=root_case_index,
                        target_timeout_seconds=_configured_target_timeout(config, None),
                    )
                    case.update(
                        elapsed_s=0.0,
                        censor_elapsed_s=0.0,
                        right_censored=True,
                        censor_stage="recovery",
                        censor_reason=censor_reason,
                        status="right-censored",
                        case_outcome="right-censored",
                    )
                    terminal = _framework_terminal_case_campaign(
                        case, "right-censored",
                    )
                    case["counts"] = dict(terminal["counts"])
                    case["target_wall_clock_pending"] = _framework_count(
                        case["counts"], "target_wall_clock_pending",
                    )
                    case["chain"] = _framework_write_chain_artifacts(
                        out, case_dir, terminal, case, manifest,
                    )
            if (
                case.get("right_censored") is True
                or case.get("case_outcome") == "right-censored"
            ):
                case.update(
                    coverage_status="partial",
                    coverage_reason="framework-right-censored",
                    coverage_artifact_complete=False,
                )
            case["case_dir"] = str(case_dir.relative_to(out))
            case["case_result"] = str(case_result_path.relative_to(out))
            write_json_replace(case_result_path, case)
            recorded_dirs.add(case_dir.resolve())
            cases.append(case)
        cases.sort(key=lambda item: int(item.get("case_index", 0)))
        lane["cases"] = cases
        lane["completed_cases"] = len(cases)
        lane["status"] = "completed" if cases else "gap"
        if not isinstance(lane.get("stop_reason"), str) or not lane["stop_reason"]:
            lane["stop_reason"] = recovery_reason or "recovery-missing-lane"
        if lane["stop_reason"] in {
            "deadline", "case-complete", "case-count", "case-timeout", "case-failure",
        }:
            lane["execution_complete"] = bool(cases) and all(
                _framework_case_chain_closed(case) for case in cases
            )
        else:
            lane["execution_complete"] = False
        lane_dir.mkdir(parents=True, exist_ok=True)
        write_json_replace(lane_path, lane)
        lane_records.append(lane)
    if not found_lane or not any(lane.get("cases") for lane in lane_records):
        return None
    duration = manifest.get("duration_seconds")
    try:
        duration = float(duration)
    except (TypeError, ValueError, OverflowError):
        duration = float(DEFAULT_RUN_SECONDS)
    return _assemble_framework_parallel_campaign(
        out, method_dir, method=method, feedback_target=feedback_target,
        backend=backend, profile=profile, manifest=manifest,
        lane_records=lane_records,
        target_timeout_seconds=_configured_target_timeout(config, None),
        parallel_lanes=lane_count,
        target_execution_parallelism=_FRAMEWORK_TARGET_EXECUTION_PARALLELISM,
        case_count_limit=case_count_limit,
        case_finalization_mode="raw-only" if raw_coverage_only else "inline",
        include_guest_metrics=rv_instruction_metrics_enabled(config),
        raw_coverage_only=raw_coverage_only,
        active_execution_seconds=previous_active_seconds,
        paused_seconds=float(previous_paused_seconds or 0.0),
        segment_count=max(1, int(previous_segment_count or 1)),
    )


def _framework_finalize_case_coverage(
    case_dir: Path, campaign: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Finish one deferred case and persist an explicit terminal state."""
    if isinstance(config, Mapping) and isinstance(config.get("targets"), Mapping):
        target_configs = config["targets"]
        target_results: dict[str, dict[str, Any]] = {}
        target_summaries = []
        target_framework_by_id = {}
        target_records_by_id = campaign.get("target_records_by_target")
        target_records_by_id = target_records_by_id if isinstance(target_records_by_id, Mapping) else {}
        target_baselines = campaign.get("target_baselines_by_target")
        target_baselines = target_baselines if isinstance(target_baselines, Mapping) else {}
        target_items = [
            (target_id, target_config)
            for target_id, target_config in target_configs.items()
            if isinstance(target_id, str) and isinstance(target_config, Mapping)
        ]

        def finalize_target(item):
            target_id, target_config = item
            nested_coverage_config = target_config.get("coverage_config")
            if isinstance(nested_coverage_config, Mapping):
                target_config = nested_coverage_config
            try:
                target_records = list(iter_materialized_records(
                    case_dir, target_records_by_id.get(target_id, ()),
                ))
                target_baseline = target_baselines.get(target_id)
                target_baseline = (
                    materialize_record(case_dir, target_baseline)
                    if isinstance(target_baseline, Mapping) else target_baseline
                )
            except (OSError, TypeError, ValueError) as error:
                target_records = []
                target_baseline = {
                    "status": "target-gap",
                    "failure_class": "target-evidence-gap",
                    "reason": f"target-evidence:{type(error).__name__}",
                    "comparison": {
                        "status": "target-gap",
                        "comparison_mode": "cross-backend",
                        "differences": {"target_evidence": str(error)},
                    },
                }
            target_campaign = {
                **dict(campaign),
                "target": target_records,
                "target_baseline": target_baseline,
            }
            if target_config.get("enabled", True):
                try:
                    wait_result = wait_for_dotnet_coverage(target_config)
                    if wait_result.get("wait_timeout") is True:
                        output = wait_result.get("output")
                        if isinstance(output, str):
                            write_json_replace(
                                Path(output).with_name("dotnet-coverage-wait-timeout.json"),
                                {"status": "gap", "reason": "dotnet-coverage-task-wait-timeout"},
                            )
                        target_result = {
                            "schema_version": "rq1-framework-coverage-v1",
                            "status": "gap", "artifact_complete": False,
                            "reason": "dotnet-coverage-task-wait-timeout",
                        }
                    else:
                        target_result = finalize_framework_coverage(
                            case_dir, target_campaign, target_config,
                        )
                except Exception as error:
                    target_result = {
                        "schema_version": "rq1-framework-coverage-v1",
                        "status": "gap", "artifact_complete": False,
                        "reason": f"framework-coverage-error:{type(error).__name__}",
                    }
            else:
                target_result = {
                    "schema_version": "rq1-framework-coverage-v1",
                    "status": "disabled", "artifact_complete": False,
                    "reason": "coverage-disabled",
                }
            target_result["target_wall_clock_pending"] = sum(
                item.get("deadline_censored") is True
                or item.get("failure_class") == "campaign-wall-clock-exhausted"
                for item in target_campaign.get("target", ())
                if isinstance(item, Mapping)
            ) + int(
                isinstance(target_campaign.get("target_baseline"), Mapping)
                and (
                    target_campaign["target_baseline"].get("deadline_censored") is True
                    or target_campaign["target_baseline"].get("failure_class")
                    == "campaign-wall-clock-exhausted"
                )
            )
            summary_path = target_result.get("summary")
            try:
                target_summary = load_json(case_dir / str(summary_path))
            except (OSError, TypeError, ValueError):
                target_summary = None
            target_framework = {}
            if isinstance(target_summary, dict):
                framework_record = target_summary.get("framework")
                target_framework = (
                    dict(framework_record)
                    if isinstance(framework_record, Mapping) else {}
                )
            return target_id, target_result, target_summary, target_framework

        if target_items:
            compact_evidence = any(
                isinstance(record, Mapping)
                and isinstance(record.get("target_evidence_ref"), Mapping)
                for records in target_records_by_id.values()
                if isinstance(records, (list, tuple))
                for record in records
            ) or any(
                isinstance(record, Mapping)
                and isinstance(record.get("target_evidence_ref"), Mapping)
                for record in target_baselines.values()
            )
            # Compact campaigns materialize one target's raw observations at a
            # time.  Legacy inline campaigns retain the historical parallel
            # finalizer, so their behavior and throughput remain unchanged.
            with ThreadPoolExecutor(
                max_workers=(
                    1 if compact_evidence
                    else min(len(target_items), len(TARGETS))
                ),
                thread_name_prefix="framework-target-finalizer",
            ) as target_pool:
                finalized = target_pool.map(finalize_target, target_items)
                for target_id, target_result, target_summary, target_framework in finalized:
                    target_results[target_id] = target_result
                    target_framework_by_id[target_id] = target_framework
                    if isinstance(target_summary, dict):
                        target_summaries.append(target_summary)

        combined_summary = {
            "schema_version": BATCH_SCHEMA,
            "status": (
                "observed" if target_summaries
                and all(item.get("status") == "observed" for item in target_summaries)
                else "partial" if any(
                    item.get("status") in {"observed", "partial"}
                    for item in target_summaries
                ) else "gap"
            ),
            "comparability": "target-local-corpus",
            "experiment_unions": [],
            "targets": [
                row for item in target_summaries
                for row in item.get("targets", ()) if isinstance(row, Mapping)
            ],
            "source_targets": [
                row for item in target_summaries
                for row in item.get("source_targets", ()) if isinstance(row, Mapping)
            ],
            "framework": {
                "target": "all",
                "target_count": len(target_configs),
                "recorded_target_count": sum(
                    result.get("status") == "recorded"
                    for result in target_results.values()
                ),
                "trace_incomplete_cases": sum(
                    item.get("framework", {}).get("trace_incomplete_cases", 0)
                    for item in target_summaries
                    if isinstance(item.get("framework"), Mapping)
                ),
                "target_wall_clock_pending": sum(
                    int(result.get("target_wall_clock_pending", 0) or 0)
                    for result in target_results.values()
                ),
                "targets": {
                    target_id: target_framework_by_id.get(target_id, {})
                    for target_id in target_results
                },
            },
        }
        combined_path = case_dir / "coverage" / "summary.json"
        write_json_replace(combined_path, combined_summary)
        coverage_enabled = any(
            item.get("enabled", True) is True
            for item in target_configs.values() if isinstance(item, Mapping)
        )
        all_recorded = bool(target_results) and all(
            item.get("status") == "recorded"
            and item.get("artifact_complete") is True
            for item in target_results.values()
        )
        coverage = {
            "schema_version": "rq1-framework-coverage-matrix-v1",
            "status": (
                "disabled" if not coverage_enabled else
                "recorded" if all_recorded else
                "partial" if any(item.get("status") in {"recorded", "partial"}
                                 for item in target_results.values()) else "gap"
            ),
            "artifact_complete": all_recorded,
            "summary": str(combined_path.relative_to(case_dir)),
            "targets": target_results,
        }
        completed = dict(campaign)
        completed["framework_coverage"] = coverage
        completed["finalization"] = {
            "status": "complete",
            "coverage_enabled": coverage_enabled,
            "framework_coverage_status": coverage.get("status"),
        }
        write_json_replace(case_dir / "campaign-result.json", completed)
        return completed
    if config is None:
        coverage = {
            "schema_version": "rq1-framework-coverage-v1",
            "status": "gap", "artifact_complete": False,
            "reason": "framework-coverage-config-missing",
        }
        coverage_enabled = True
    elif config.get("enabled", True):
        try:
            coverage_campaign = dict(campaign)
            records = coverage_campaign.get("target")
            if isinstance(records, (list, tuple)):
                coverage_campaign["target"] = list(
                    iter_materialized_records(case_dir, records),
                )
            baseline = coverage_campaign.get("target_baseline")
            if isinstance(baseline, Mapping):
                coverage_campaign["target_baseline"] = materialize_record(
                    case_dir, baseline,
                )
            coverage = finalize_framework_coverage(
                case_dir, coverage_campaign, config,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            coverage = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": "gap", "artifact_complete": False,
                "reason": f"framework-coverage-error:{type(error).__name__}",
            }
        coverage_enabled = True
    else:
        coverage = {
            "schema_version": "rq1-framework-coverage-v1",
            "status": "disabled", "artifact_complete": False,
            "reason": "coverage-disabled",
        }
        coverage_enabled = False
    completed = dict(campaign)
    completed["framework_coverage"] = coverage
    completed["finalization"] = {
        "status": "complete", "coverage_enabled": coverage_enabled,
        "framework_coverage_status": coverage.get("status"),
        "reason": coverage.get("reason"),
    }
    write_json_replace(case_dir / "campaign-result.json", completed)
    return completed


def _framework_merge_target_queue_sidecars(
    out: Path, method_dir: Path, target_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Expose producer close state; Target result merge belongs to offline analysis."""
    completion = _framework_target_queue_completion(method_dir, target_ids)
    completion["merged_case_count"] = 0
    completion["merge_owner"] = "offline-target-analysis"
    return completion


def _framework_reproject_finalized_case(
    out: Path, case_dir: Path, record: dict[str, Any],
    campaign: dict[str, Any], *, method: str, feedback_target: str,
    backend: str, spec: dict[str, Any], profile: dict[str, Any],
    manifest: dict[str, Any], manifest_verified: bool,
    target_timeout_seconds: int | None,
) -> None:
    """Refresh case/chain checkpoints after coverage finalization."""
    refreshed = _framework_parallel_case_record(
        case_dir, method=method, feedback_target=feedback_target,
        backend=backend, spec=spec, profile=profile,
        manifest_verified=manifest_verified, campaign=campaign,
        entry=record.get("entry") if isinstance(record.get("entry"), dict) else {},
        lane_id=int(record.get("lane_id", 0)),
        case_index=int(record.get("case_index", 0)),
        seed=int(record.get("seed", 0)),
        root_case_index=record.get("root_case_index"),
        target_timeout_seconds=target_timeout_seconds,
    )
    for name in (
        "elapsed_s", "censor_elapsed_s", "censor_stage", "censor_reason",
        "right_censored", "case_outcome", "recovery", "case_dir",
    ):
        if name in record:
            refreshed[name] = record[name]
    if refreshed.get("right_censored") is True:
        refreshed.update(
            coverage_status="partial",
            coverage_reason="framework-right-censored",
            coverage_artifact_complete=False,
        )
    finalization = campaign.get("finalization")
    if isinstance(finalization, dict):
        refreshed["campaign_finalization_status"] = finalization.get("status")
    refreshed["chain"] = _framework_write_chain_artifacts(
        out, case_dir, campaign, refreshed, manifest,
    )
    refreshed["case_result"] = str(
        (case_dir / "case-result.json").relative_to(out)
    )
    record.clear()
    record.update(refreshed)
    write_json_replace(case_dir / "case-result.json", record)


def _framework_finalize_pending_cases(
    out: Path, config: dict[str, Any], manifest: dict[str, Any],
    parallel: dict[str, Any], *, manifest_verified: bool,
) -> dict[str, Any]:
    """Finish deferred case coverage from inline or sidecar evidence."""
    method = manifest.get("method_filter")
    feedback_target = manifest.get("feedback_target")
    if not isinstance(method, str) or not isinstance(feedback_target, str):
        return parallel
    specs = {
        item.get("id"): item for item in config.get("framework_methods", ())
        if isinstance(item, dict)
    }
    spec = specs.get(method)
    if not isinstance(spec, dict):
        return parallel
    profile = spec.get("generator_profile")
    profile = profile if isinstance(profile, dict) else {}
    backend = (
        "multi-target" if feedback_target == "all"
        else FRAMEWORK_TARGET_BACKENDS.get(feedback_target)
    )
    if not isinstance(backend, str):
        return parallel
    method_dir = out / "framework" / method.replace("/", "-")
    config_dir = out / "framework-coverage-configs" / method.replace("/", "-")
    lane_records = parallel.get("lanes")
    if not isinstance(lane_records, list):
        return parallel
    changed = False
    changed_lanes: set[int] = set()
    target_timeout_seconds = _configured_target_timeout(config, None)
    case_count_limit = _framework_case_count(profile)
    for lane in lane_records:
        if not isinstance(lane, dict):
            continue
        for record in lane.get("cases", ()):
            if not isinstance(record, dict):
                continue
            case_dir_value = record.get("case_dir")
            if not isinstance(case_dir_value, str):
                continue
            case_dir = out / case_dir_value
            try:
                campaign = load_json(case_dir / "campaign-result.json")
            except (OSError, TypeError, ValueError):
                continue
            if not _framework_campaign_core_recorded(campaign):
                continue
            campaign_complete = _framework_campaign_result_complete(campaign)
            if campaign_complete:
                coverage = campaign.get("framework_coverage")
                coverage_status = (
                    coverage.get("status")
                    if isinstance(coverage, Mapping) else "gap"
                )
                coverage_complete = (
                    coverage.get("artifact_complete") is True
                    if isinstance(coverage, Mapping) else False
                )
                stale = (
                    record.get("campaign_finalization_pending") is True
                    or record.get("campaign_finalization_status") not in {
                        None, "complete",
                    }
                    or record.get("coverage_status") != coverage_status
                    or record.get("coverage_artifact_complete") is not coverage_complete
                )
                if not stale:
                    continue
                lane_id = int(record.get("lane_id", lane.get("lane_id", 0)))
                _framework_reproject_finalized_case(
                    out, case_dir, record, campaign, method=method,
                    feedback_target=feedback_target, backend=backend, spec=spec,
                    profile=profile, manifest=manifest,
                    manifest_verified=manifest_verified,
                    target_timeout_seconds=target_timeout_seconds,
                )
                changed = True
                changed_lanes.add(lane_id)
                continue
            lane_id = int(record.get("lane_id", lane.get("lane_id", 0)))
            case_index = int(record.get("case_index", 0))
            config_path = config_dir / (
                f"lane-{lane_id:02d}-case-{case_index:06d}.json"
            )
            try:
                case_config = load_json(config_path)
            except (OSError, TypeError, ValueError):
                case_config = None
            campaign = _framework_finalize_case_coverage(
                case_dir, campaign, case_config,
            )
            # Formal raw-only runs leave Target raw evidence to the simulator
            # side and offline analysis; producer recovery must not read or
            # reinterpret Target sidecars.
            if campaign.get("raw_coverage_only") is True:
                write_json_replace(case_dir / "campaign-result.json", campaign)
            _framework_reproject_finalized_case(
                out, case_dir, record, campaign, method=method,
                feedback_target=feedback_target, backend=backend, spec=spec,
                profile=profile, manifest=manifest,
                manifest_verified=manifest_verified,
                target_timeout_seconds=target_timeout_seconds,
            )
            changed = True
            changed_lanes.add(lane_id)
    for lane_id in sorted(changed_lanes):
        write_json_replace(
            method_dir / "lanes" / f"lane-{lane_id:02d}" / "lane-result.json",
            next(
                lane for lane in lane_records
                if isinstance(lane, dict) and lane.get("lane_id") == lane_id
            ),
        )
    if not changed:
        return parallel
    duration = manifest.get("duration_seconds")
    try:
        duration = float(duration)
    except (TypeError, ValueError, OverflowError):
        duration = float(DEFAULT_RUN_SECONDS)
    return _assemble_framework_parallel_campaign(
        out, method_dir, method=method, feedback_target=feedback_target,
        backend=backend, profile=profile, manifest=manifest,
        lane_records=lane_records,
        target_timeout_seconds=target_timeout_seconds,
        parallel_lanes=len(lane_records),
        target_execution_parallelism=_FRAMEWORK_TARGET_EXECUTION_PARALLELISM,
        case_count_limit=case_count_limit,
        case_finalization_mode="recovered-after-interruption",
        include_guest_metrics=rv_instruction_metrics_enabled(config),
    )


def _run_framework_parallel_campaign(
    out: Path, method_dir: Path, *, method: str, feedback_target: str,
    backend: str, spec: dict[str, Any], profile: dict[str, Any],
    manifest: dict[str, Any], manifest_verified: bool,
    segment_duration_seconds: int,
    base_seed: int, command_builder: Any, target_timeout_seconds: int | None = None,
    case_finalizer: Any = None, finalization_workers: int = 1,
    target_execution_parallelism: int | None = None,
    include_guest_metrics: bool = True,
    raw_coverage_only: bool = False,
    ablation_mode: bool = False,
    case_count_limit_override: int | None = None,
) -> dict[str, Any]:
    lanes = _framework_parallel_lanes(profile)
    case_count_limit = _framework_case_count(profile)
    if case_count_limit_override is not None:
        if type(case_count_limit_override) is not int or case_count_limit_override < 1:
            raise ValueError("framework case count limit override must be positive")
        case_count_limit = (
            case_count_limit_override if case_count_limit is None
            else min(case_count_limit, case_count_limit_override)
        )
    if raw_coverage_only:
        target_ids = TARGETS if feedback_target == "all" else (feedback_target,)
        for target_id in target_ids:
            safe_target = "".join(
                char if char.isalnum() or char in ".-_" else "_"
                for char in str(target_id)
            ) or "unknown"
            (method_dir / "target-queues" / safe_target).mkdir(
                parents=True, exist_ok=True,
            )
    target_limit = max(1, int(
        target_execution_parallelism or _FRAMEWORK_TARGET_EXECUTION_PARALLELISM
    ))
    segment_duration = [max(1, int(segment_duration_seconds))]
    started_mono = time.monotonic()
    previous_active_seconds = 0.0
    previous_paused_seconds = 0.0
    previous_segment_count = 0
    try:
        previous_parallel = load_json(method_dir / "parallel-result.json")
    except (OSError, TypeError, ValueError):
        previous_parallel = {}
    if isinstance(previous_parallel, Mapping):
        try:
            previous_active_seconds = max(
                0.0, float(previous_parallel.get("active_execution_seconds", 0.0) or 0.0)
            )
        except (TypeError, ValueError, OverflowError):
            previous_active_seconds = 0.0
        try:
            previous_paused_seconds = max(
                0.0, float(previous_parallel.get("paused_seconds", 0.0) or 0.0)
            )
        except (TypeError, ValueError, OverflowError):
            previous_paused_seconds = 0.0
        try:
            previous_segment_count = max(
                0, int(previous_parallel.get("segment_count", 0) or 0)
            )
        except (TypeError, ValueError, OverflowError):
            previous_segment_count = 0
    from decoupled_pipeline import _CaseBoundaryPause

    pause_execution = SimpleNamespace(out=out, deadline=float("inf"))
    pause_gate = _CaseBoundaryPause(
        pause_execution, method_filter=method,
        participants={f"lane-{lane_id:02d}" for lane_id in range(lanes)},
        segment_duration_seconds=segment_duration[0],
    )
    _requested_sequence, requested_state, requested_duration = pause_gate._request()
    if requested_state == "running" and type(requested_duration) is int \
            and requested_duration > 0:
        segment_duration[0] = requested_duration
        pause_gate.segment_duration_seconds = requested_duration
        pause_gate.segment_deadline = (
            pause_gate.segment_started + requested_duration
        )
    pause_gate.segment_index = previous_segment_count + 1
    pause_gate._write_status("running")
    segment_deadline = [pause_gate.segment_deadline]
    segment_lock = threading.Lock()
    status_lock = threading.Lock()
    status_last_written = [0.0]
    last_resume_sequence = [pause_gate._request()[0]]
    segment_count = [previous_segment_count + 1]

    def persist_control_status() -> None:
        """Refresh the timebox while a slow case is still in flight."""
        now = time.monotonic()
        with status_lock:
            if now - status_last_written[0] < 1.0:
                return
            status_last_written[0] = now
        pause_gate._write_status(
            "paused" if pause_gate.pause_started is not None else "running"
        )

    def pause_requested() -> bool:
        return time.monotonic() >= segment_deadline[0] or pause_gate._pause_trigger()[0]

    def resume_next_segment() -> bool:
        sequence, desired, _ = pause_gate._request()
        if stop_requested():
            return False
        with segment_lock:
            if sequence < last_resume_sequence[0]:
                raise ValueError("resume request sequence did not advance")
            if desired != "running":
                raise ValueError("resume request must set desired_state to running")
            segment_duration[0] = pause_gate.segment_duration_seconds
            segment_deadline[0] = pause_gate.segment_deadline
            segment_count[0] = pause_gate.segment_index
            last_resume_sequence[0] = max(last_resume_sequence[0], sequence)
        return True

    def request_segment_pause() -> None:
        """Persist the timebox as a resumable pause request."""
        path = pause_gate.request_path
        try:
            current = load_json(path)
        except (OSError, TypeError, ValueError):
            current = {}
        if not isinstance(current, Mapping):
            current = {}
        sequence = current.get("sequence", 0)
        if type(sequence) is not int or sequence < 0:
            sequence = 0
        if current.get("desired_state") == "pause":
            return
        write_json_replace(path, {
            "schema_version": "rq1-case-boundary-request-v1",
            "sequence": sequence + 1,
            "desired_state": "pause",
            "reason": "segment-budget-expired",
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    method_dir.mkdir(parents=True, exist_ok=True)
    lanes_dir = method_dir / "lanes"
    finalizer_pool = None
    finalization_jobs: list[tuple[dict[str, Any], Path, Any]] = []
    finalization_jobs_lock = threading.Lock()
    target_progress_last_written = [0.0]
    target_progress_ids = tuple(
        TARGETS if feedback_target == "all" else (feedback_target,)
    )
    def persist_target_progress() -> None:
        """Publish best-effort per-Target counts without making them a scheduler gate."""
        now = time.monotonic()
        if now - target_progress_last_written[0] < 0.5:
            return
        target_progress_last_written[0] = now
        snapshot = _framework_target_progress_snapshot(method_dir, target_progress_ids)
        # Atomic replacement prevents concurrent lane snapshots from producing
        # a partial JSON file; the snapshot itself is a derived view.
        write_json_replace(method_dir / "target-progress.json", {
            "schema_version": "rq1-framework-target-progress-v1",
            "targets": snapshot,
        })

    if callable(case_finalizer):
        try:
            worker_count = max(1, min(int(finalization_workers), lanes))
        except (TypeError, ValueError, OverflowError):
            worker_count = 1
        finalizer_pool = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="framework-finalizer",
        )
    case_env = os.environ.copy()
    if raw_coverage_only:
        case_env["RQ1_RAW_COVERAGE_ONLY"] = "1"
        # Direct/Program-Full publish immutable entries to one durable queue
        # log per Target. Consumers are external services; this producer never
        # starts, stops, joins, or drains them.
        case_env["RQ1_TARGET_QUEUE_ONLY"] = "1"
        case_env["RQ1_TARGET_QUEUE_ROOT"] = str(method_dir / "target-queues")
    if finalizer_pool is not None:
        # The online deadline belongs to generation/reference/Target.  Slow
        # coverage materialization is queued after a case's core result is
        # durable, so it cannot prevent the next case from starting.
        case_env["RQ1_DEFER_FINALIZE"] = "1"

    lane_records: list[dict[str, Any]] = []
    # The case process exits after its reference chain/checkpoint and durable
    # Target plan are written. The only producer ordering is the per-lane
    # reference-chain checkpoint.
    case_queue_capacity = None

    def submit_case(
        label: str, command: list[str], timeout: float | None, env: dict[str, str],
    ) -> Future:
        """Run one case without making its Target tail a scheduler worker.

        The segment deadline controls admission only.  A launched case owns
        its reference checkpoint and must finish that checkpoint before the
        producer can pause; applying the segment remainder as a subprocess
        timeout would turn a normal boundary pause into case termination.
        """
        result: Future = Future()

        def invoke() -> None:
            try:
                result.set_result(_run_logged(out, label, command, timeout, env=env))
            except BaseException as error:  # surface worker failures via Future.result()
                result.set_exception(error)

        threading.Thread(
            target=invoke, name=f"{label}-process", daemon=False,
        ).start()
        return result

    def run_lane(lane_id: int) -> dict[str, Any]:
        lane_dir = lanes_dir / f"lane-{lane_id:02d}"
        lane_dir.mkdir(parents=True, exist_ok=True)
        existing_lane: dict[str, Any] = {}
        try:
            persisted_lane = load_json(lane_dir / "lane-result.json")
            if isinstance(persisted_lane, dict):
                existing_lane = persisted_lane
        except (OSError, TypeError, ValueError):
            existing_lane = {}
        existing_cases = [
            dict(item) for item in existing_lane.get("cases", ())
            if isinstance(item, dict)
        ]
        existing_indices = [
            int(item["case_index"]) for item in existing_cases
            if type(item.get("case_index")) is int and item["case_index"] >= 0
        ]
        next_case_index = max(
            [*existing_indices, int(existing_lane.get("next_case_index", 0) or 0)],
            default=0,
        )
        lane = {
            "lane_id": lane_id, "status": "running", "cases": [],
            "stop_reason": None, "execution_complete": False,
            "next_case_index": next_case_index,
            "next_case_id": f"lane-{lane_id:02d}-case-{next_case_index:06d}",
            "last_completed_case_index": None, "last_completed_case_id": None,
        }
        lane.update({
            key: value for key, value in existing_lane.items()
            if key not in {"status", "stop_reason", "execution_complete", "cases"}
        })
        lane["cases"] = existing_cases
        lane["status"] = "running"
        lane["stop_reason"] = None
        lane["execution_complete"] = False
        lane["launched_cases"] = max(
            next_case_index, int(existing_lane.get("launched_cases", 0) or 0),
        )
        active_jobs: list[dict[str, Any]] = []
        last_launched: dict[str, Any] | None = None
        stop_launching = False
        pause_pending = False

        def cursor() -> dict[str, Any]:
            last_index = next_case_index - 1
            return {
                "lane_id": lane_id,
                "next_position": next_case_index,
                "last_completed_case_index": last_index if last_index >= 0 else None,
                "last_completed_case_id": (
                    f"lane-{lane_id:02d}-case-{last_index:06d}"
                    if last_index >= 0 else None
                ),
                "next_case_index": next_case_index,
                "next_case_id": f"lane-{lane_id:02d}-case-{next_case_index:06d}",
                "next_seed": base_seed + lane_id * _FRAMEWORK_SEED_STRIDE + next_case_index,
                "segment_index": segment_count[0],
                "active_execution_seconds": round(max(
                    0.0, previous_active_seconds + time.monotonic() - started_mono
                    - pause_gate.paused_seconds,
                ), 6),
            }

        def launch_case(index: int) -> dict[str, Any]:
            case_dir = lane_dir / "cases" / f"case-{index:06d}"
            # A process can die after creating the case directory but before
            # the lane checkpoint records it.  Keep that partial evidence in
            # a non-canonical quarantine directory and reuse the canonical
            # case number on resume; otherwise the first resume stops on
            # FileExistsError and silently loses the lane suffix.
            if case_dir.exists():
                if not case_dir.is_dir():
                    raise RuntimeError(f"existing-case-path-not-directory:{case_dir}")
                durable = any(
                    (case_dir / name).is_file()
                    for name in (
                        "campaign-result.json",
                        "reference-chain-checkpoint.json",
                        "case-result.json",
                    )
                )
                if durable:
                    raise RuntimeError(
                        f"existing-case-needs-recovery:{case_dir}"
                    )
                quarantine = case_dir.with_name(
                    f"{case_dir.name}.interrupted-{time.time_ns()}"
                )
                os.replace(case_dir, quarantine)
                write_json_replace(quarantine / "recovery.json", {
                    "schema_version": "rq1-framework-orphan-case-v1",
                    "original_case_dir": str(case_dir.relative_to(out)),
                    "quarantined_case_dir": str(quarantine.relative_to(out)),
                    "reason": "resume-before-lane-checkpoint",
                    "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
                })
            case_dir.mkdir(parents=True, exist_ok=False)
            seed = base_seed + lane_id * _FRAMEWORK_SEED_STRIDE + index
            root_case_index = (
                lane_id + index * lanes
                if method == "Ours-RVGEN-Direct" else None
            )
            job: dict[str, Any] = {
                "case_index": index, "case_dir": case_dir, "seed": seed,
                "root_case_index": root_case_index,
                "started": time.monotonic(),
                "checkpoint_path": case_dir / "reference-chain-checkpoint.json",
                "env": {
                    **case_env,
                    "RQ1_CHAIN_CHECKPOINT_PATH": str(
                        case_dir / "reference-chain-checkpoint.json"
                    ),
                },
                "future": None, "entry": None,
            }
            try:
                command = command_builder(
                    case_dir, seed, None, lane_id, index,
                )
                job["future"] = submit_case(
                    f"framework-case-{lane_id:02d}-{index:06d}",
                    command, None, job["env"],
                )
            except (OSError, RuntimeError, ValueError) as error:
                job["entry"] = {
                    "status": "gap", "reason": f"case-launch:{error}",
                }
            return job

        def finish_case(job: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
            deadline_reached = time.monotonic() >= segment_deadline[0]
            entry_reason = entry.get("reason")
            recovery_stop_reason = (
                "execution-wall-clock-exhausted"
                if entry_reason == "external-signal" and deadline_reached else
                "external-signal"
                if entry_reason == "external-signal" else
                entry_reason if isinstance(entry_reason, str) else None
            )
            case_dir = job["case_dir"]
            case_index = int(job["case_index"])
            seed = int(job["seed"])
            root_case_index = job["root_case_index"]
            result_path = case_dir / "campaign-result.json"
            if result_path.is_file():
                try:
                    campaign = load_json(result_path)
                except (OSError, TypeError, ValueError):
                    campaign = {}
            else:
                campaign = {}
            campaign = campaign if _framework_campaign_core_recorded(campaign) else {}
            if not campaign:
                checkpoint = _framework_read_chain_checkpoint(job["checkpoint_path"])
                if checkpoint is not None:
                    campaign = _framework_recover_chain_checkpoint(
                        checkpoint,
                        coverage_finalization_requested=callable(case_finalizer),
                        case_dir=case_dir,
                        stop_reason=recovery_stop_reason,
                    )
                    write_json_replace(result_path, campaign)
                    try:
                        ledger_path = case_dir / "candidate-ledger.json"
                        ledger = load_json(ledger_path)
                    except (OSError, TypeError, ValueError):
                        ledger = None
                    if isinstance(ledger, dict):
                        chain = campaign.get("chain")
                        ledger["events"] = (
                            chain.get("events", []) if isinstance(chain, Mapping) else []
                        )
                        for name in (
                            "target", "target_records_by_target", "target_baseline",
                            "target_baselines_by_target", "target_replay", "counts",
                        ):
                            if name in campaign:
                                ledger[name] = campaign[name]
                        write_json_replace(ledger_path, ledger)
                    if not raw_coverage_only:
                        _framework_recover_case_target_coverage(
                            case_dir, method, campaign,
                        )
                    write_json_replace(result_path, campaign)
            campaign = campaign if _framework_campaign_core_recorded(campaign) else {}
            if raw_coverage_only and campaign:
                campaign = _framework_raw_only_campaign(campaign)
                write_json_replace(result_path, campaign)
            campaign_finalization_pending = bool(
                campaign and not _framework_campaign_result_complete(campaign)
            )
            if campaign_finalization_pending:
                campaign_finalization = campaign.get("finalization")
                campaign_finalization_status = (
                    campaign_finalization.get("status")
                    if isinstance(campaign_finalization, Mapping) else None
                )
            else:
                campaign_finalization_status = None
            record = _framework_parallel_case_record(
                case_dir, method=method, feedback_target=feedback_target,
                backend=backend, spec=spec, profile=profile,
                manifest_verified=manifest_verified, campaign=campaign,
                entry=entry, lane_id=lane_id, case_index=case_index,
                seed=seed, root_case_index=root_case_index,
                target_timeout_seconds=target_timeout_seconds,
            )
            if campaign_finalization_pending:
                record["campaign_finalization_pending"] = True
                record["campaign_finalization_status"] = (
                    campaign_finalization_status or "pending"
                )
            record["elapsed_s"] = round(time.monotonic() - job["started"], 6)
            if record["right_censored"]:
                record["censor_elapsed_s"] = record["elapsed_s"]
            record["case_dir"] = str(case_dir.relative_to(out))
            end_state = _framework_case_end_state(
                campaign_recorded=bool(campaign),
                entry_status=entry.get("status"),
                entry_reason=entry.get("reason"),
                deadline_reached=deadline_reached,
            )
            if entry.get("reason") == "external-signal":
                end_state = {
                    "lane_stop_reason": "external-signal",
                    "case_outcome": "campaign-result" if campaign else "right-censored",
                }
            record["case_outcome"] = end_state["case_outcome"]
            terminal_campaign = None
            if end_state["case_outcome"] == "right-censored":
                external_stop = end_state["lane_stop_reason"] == "external-signal"
                censor_reason = (
                    "external-signal" if external_stop else
                    entry.get("reason")
                    if entry.get("reason") in _FRAMEWORK_DEADLINE_CENSOR_REASONS else
                    "shared-deadline-while-case-running"
                )
                record.update(
                    right_censored=True,
                    censor_stage=(
                        "campaign-process-external-interruption"
                        if external_stop else "campaign-process-deadline"
                    ),
                    censor_reason=censor_reason,
                    censor_elapsed_s=record["elapsed_s"],
                    status="right-censored",
                )
                terminal_campaign = _framework_terminal_case_campaign(
                    record, "right-censored",
                )
            elif end_state["case_outcome"] in {"case-timeout", "case-failure"}:
                record["status"] = "gap"
                terminal_campaign = _framework_terminal_case_campaign(
                    record, str(end_state["case_outcome"]),
                )
            if record["right_censored"]:
                record.update(
                    coverage_status="partial",
                    coverage_reason="framework-right-censored",
                    coverage_artifact_complete=False,
                )
            if terminal_campaign is not None:
                terminal_counts = terminal_campaign.get("counts")
                record["counts"] = (
                    dict(terminal_counts) if isinstance(terminal_counts, dict) else {}
                )
                record["target_wall_clock_pending"] = _framework_count(
                    record["counts"], "target_wall_clock_pending",
                )
            if campaign:
                record["chain"] = _framework_write_chain_artifacts(
                    out, case_dir, campaign, record, manifest,
                )
            elif terminal_campaign is not None:
                record["chain"] = _framework_write_chain_artifacts(
                    out, case_dir, terminal_campaign, record, manifest,
                )
            else:
                record["chain"] = {
                    "chain_events": None, "state_count": 0,
                    "integrity": None, "seal": None,
                    "sealed": False, "integrity_verified": False,
                }
            record["case_result"] = str(
                (case_dir / "case-result.json").relative_to(out)
            )
            lane["cases"].append(record)
            lane["cases"].sort(key=lambda item: int(item.get("case_index", 0)))
            lane["completed_cases"] = len(lane["cases"])
            lane.update(cursor())
            write_json_replace(case_dir / "case-result.json", record)
            write_json_replace(lane_dir / "lane-result.json", lane)
            if (
                finalizer_pool is not None
                and campaign
                and not _framework_campaign_result_complete(campaign)
            ):
                future = finalizer_pool.submit(
                    case_finalizer, case_dir, dict(campaign), dict(record),
                )
                with finalization_jobs_lock:
                    finalization_jobs.append((record, case_dir, future))
            return end_state

        try:
            while active_jobs or not stop_launching:
                persist_control_status()
                made_progress = False
                for job in list(active_jobs):
                    future = job["future"]
                    done = future.done() if future is not None else True
                    if not done:
                        continue
                    try:
                        entry = job["entry"] if future is None else future.result()
                    except Exception as error:
                        entry = {"status": "gap", "reason": f"case-worker:{error}"}
                    entry = entry if isinstance(entry, dict) else {
                        "status": "gap", "reason": "case-worker-returned-no-result",
                    }
                    active_jobs.remove(job)
                    end_state = finish_case(job, entry)
                    persist_target_progress()
                    job["completed"] = True
                    made_progress = True
                    if end_state["lane_stop_reason"] == "deadline":
                        pause_pending = True
                        stop_launching = True
                        lane["stop_reason"] = None
                    elif end_state["lane_stop_reason"] is not None:
                        stop_launching = True
                        lane["stop_reason"] = end_state["lane_stop_reason"]

                if stop_requested():
                    stop_launching = True
                    lane["stop_reason"] = stop_reason()
                remaining = segment_deadline[0] - time.monotonic()
                if remaining <= 0:
                    stop_launching = True
                    pause_pending = True
                    lane["stop_reason"] = None

                count_reached = (
                    case_count_limit is not None
                    and next_case_index >= case_count_limit
                )
                if count_reached:
                    stop_launching = True
                    pause_pending = False
                    lane["stop_reason"] = lane.get("stop_reason") or "case-count"
                elif not stop_requested() and pause_requested():
                    stop_launching = True
                    pause_pending = True

                latest_ready = (
                    last_launched is None
                    or last_launched.get("completed") is True
                    or last_launched["future"] is None
                    or last_launched["future"].done()
                    # The checkpoint writer replaces the file atomically;
                    # only parse it if recovery is actually needed.
                    or last_launched["checkpoint_path"].is_file()
                )
                can_launch = (
                    not stop_launching
                    and remaining > 0
                    and not count_reached
                    and latest_ready
                )
                if can_launch:
                    try:
                        job = launch_case(next_case_index)
                    except (OSError, RuntimeError, ValueError) as error:
                        lane["stop_reason"] = (
                            f"scheduler-error:{type(error).__name__}:{error}"
                        )
                        stop_launching = True
                    else:
                        active_jobs.append(job)
                        last_launched = job
                        next_case_index += 1
                        lane["launched_cases"] = next_case_index
                        made_progress = True

                if not active_jobs and stop_launching:
                    if pause_pending and lane.get("stop_reason") is None \
                            and not stop_requested():
                        stop_after_pause = (
                            time.monotonic() >= segment_deadline[0]
                            or pause_gate.stop_after_pause
                        )
                        if stop_after_pause:
                            request_segment_pause()
                        lane["status"] = "paused"
                        lane.update(cursor())
                        write_json_replace(lane_dir / "lane-result.json", lane)
                        pause_gate.checkpoint(
                            f"lane-{lane_id:02d}", cursor(),
                            stop_after_pause=stop_after_pause,
                        )
                        if stop_requested():
                            lane["stop_reason"] = stop_reason()
                            break
                        if stop_after_pause:
                            lane["stop_reason"] = "deadline"
                            break
                        if not resume_next_segment():
                            lane["stop_reason"] = stop_reason()
                            break
                        lane["status"] = "running"
                        lane["stop_reason"] = None
                        pause_pending = False
                        stop_launching = False
                        lane.update(cursor())
                        write_json_replace(lane_dir / "lane-result.json", lane)
                        continue
                    break
                if not made_progress:
                    time.sleep(0.01)
        except Exception as error:
            lane["stop_reason"] = (
                f"scheduler-error:{type(error).__name__}:{error}"
            )
        finally:
            lane["cases"].sort(key=lambda item: int(item.get("case_index", 0)))
            lane["status"] = "completed" if lane["cases"] else "gap"
            lane["execution_complete"] = (
                lane["stop_reason"] in {
                    "deadline", "case-complete", "case-count", "case-timeout", "case-failure",
                }
                and bool(lane["cases"])
                and all(_framework_case_chain_closed(case) for case in lane["cases"])
            )
            lane.update(cursor())
            write_json_replace(lane_dir / "lane-result.json", lane)
            pause_gate.finish(f"lane-{lane_id:02d}", cursor())
        return lane

    try:
        with ThreadPoolExecutor(
            max_workers=lanes, thread_name_prefix="framework-lane",
        ) as pool:
            futures = [pool.submit(run_lane, lane_id) for lane_id in range(lanes)]
            lane_records = [future.result() for future in futures]
    finally:
        if finalizer_pool is not None:
            finalizer_pool.shutdown(wait=True)
        # The framework subprocesses may leave detached simulator helpers
        # behind when the shared deadline interrupts a case.  All lane and
        # coverage futures are quiescent here, so reaping adopted children is
        # safe and keeps a completed run from being held open by zombies.
        _reap_adopted_children()
        persist_target_progress()
        if raw_coverage_only:
            _framework_close_target_queues(method_dir, target_progress_ids)

    # Re-project finalized campaigns into the per-case record and chain.  The
    # initial checkpoint remains useful if the process is interrupted while
    # this post-processing phase is running; a later framework finalize can
    # recover any still-pending case from campaign-result.json.
    lanes_by_id = {
        lane.get("lane_id"): lane for lane in lane_records
        if isinstance(lane, dict)
    }
    for record, case_dir, future in finalization_jobs:
        try:
            campaign = future.result()
            if not isinstance(campaign, dict):
                raise ValueError("framework case finalizer returned no campaign")
        except Exception as error:
            campaign = {}
            try:
                campaign = load_json(case_dir / "campaign-result.json")
            except (OSError, TypeError, ValueError):
                pass
            campaign = dict(campaign) if isinstance(campaign, dict) else {}
            campaign["framework_coverage"] = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": "gap", "artifact_complete": False,
                "reason": f"framework-coverage-error:{type(error).__name__}",
            }
            campaign["finalization"] = {
                "status": "complete", "coverage_enabled": True,
                "framework_coverage_status": "gap",
                "reason": "coverage-finalizer-failed",
            }
            write_json_replace(case_dir / "campaign-result.json", campaign)
        lane_id = int(record.get("lane_id", 0))
        _framework_reproject_finalized_case(
            out, case_dir, record, campaign, method=method,
            feedback_target=feedback_target, backend=backend, spec=spec,
            profile=profile, manifest=manifest,
            manifest_verified=manifest_verified,
            target_timeout_seconds=target_timeout_seconds,
        )
        lane = lanes_by_id.get(lane_id)
        if isinstance(lane, dict):
            write_json_replace(
                lanes_dir / f"lane-{lane_id:02d}" / "lane-result.json", lane,
            )
    return _assemble_framework_parallel_campaign(
        out, method_dir, method=method, feedback_target=feedback_target,
        backend=backend, profile=profile, manifest=manifest,
        lane_records=lane_records, target_timeout_seconds=target_timeout_seconds,
        parallel_lanes=lanes,
        target_execution_parallelism=target_limit,
        case_queue_capacity=case_queue_capacity,
        case_count_limit=case_count_limit,
        case_finalization_mode=(
            "deferred-after-case" if finalizer_pool is not None else "raw-only"
        ),
        include_guest_metrics=include_guest_metrics,
        raw_coverage_only=raw_coverage_only,
        ablation_mode=ablation_mode,
        active_execution_seconds=max(
            0.0, previous_active_seconds + time.monotonic() - started_mono
            - pause_gate.paused_seconds,
        ),
        paused_seconds=previous_paused_seconds + pause_gate.paused_seconds,
        segment_count=segment_count[0],
    )


def _run_framework_campaign_single(
    config: dict[str, Any], out: Path, *, method: str,
    feedback_target: str = "T-LRSV-INT",
    timeout_seconds: int | None = None,
    target_timeout_override: int | None = None,
    program_seed_catalog_root: Path | None = None,
    program_seed_catalog_manifest: Path | None = None,
    small_model_enabled: bool = True,
    emi_enabled: bool = True,
    mcmc_enabled: bool = True,
) -> dict[str, Any]:
    """运行一条 reference-driven Ours case 链，再回放到当前 Target。"""
    if any(
        type(value) is not bool
        for value in (small_model_enabled, emi_enabled, mcmc_enabled)
    ):
        raise ValueError("framework module toggles must be booleans")
    if not emi_enabled and mcmc_enabled:
        raise ValueError("EMI-off runs require MCMC to be off")
    coverage_enabled = simulator_coverage_enabled()
    coverage_gap = None
    specs = {item.get("id"): item for item in config.get("framework_methods", [])
             if isinstance(item, dict)}
    if method not in specs:
        raise ValueError("framework-run requires a declared framework method")
    spec = specs[method]
    allowed_feedback_targets = spec.get("feedback_targets")
    target_ids = TARGETS if feedback_target == "all" else (feedback_target,)
    if not isinstance(allowed_feedback_targets, list) \
            or any(target not in allowed_feedback_targets for target in target_ids):
        raise ValueError(
            f"feedback target {feedback_target!r} is not declared for {method}"
        )
    target_specs = {
        str(item.get("id")): item
        for item in config.get("targets", ()) if isinstance(item, dict)
    }
    target_configs: dict[str, dict[str, Any]] = {}
    target_timeout_by_id: dict[str, int] = {}
    for target_id in target_ids:
        target_spec = target_specs.get(target_id, {})
        target_backend = FRAMEWORK_TARGET_BACKENDS.get(target_id)
        if target_backend is None:
            raise ValueError("unknown feedback target: " + str(target_id))
        # Use the binary and identity declared by the frozen config.
        target_binary = dep(target_spec.get("binary"))
        if target_binary is None or not target_binary.is_file():
            raise ValueError(
                "target-identity-unverified: declared target binary is missing: "
                + str(target_spec.get("binary")))
        expected_binary_sha = target_spec.get("binary_sha256")
        if expected_binary_sha and sha256(target_binary) != str(expected_binary_sha).lower():
            raise ValueError("target-identity-unverified: declared target binary digest mismatch")
        target_identity = _identity(
            dep(target_spec.get("identity")), target_binary, target_spec,
        )
        if target_identity.get("status") != "verified":
            raise ValueError(
                "target-identity-unverified: "
                + str(target_identity.get("reason_code") or "identity-check-failed")
            )
        coverage_binary = dep(target_spec.get("coverage_binary")) if coverage_enabled else None
        if coverage_enabled and (coverage_binary is None or not coverage_binary.is_file()):
            coverage_gap = coverage_gap or f"{target_id}:framework-coverage-binary-missing"
        source_coverage_spec = target_spec.get("source_coverage") if coverage_enabled else {}
        if coverage_enabled and not isinstance(source_coverage_spec, dict):
            coverage_gap = coverage_gap or f"{target_id}:framework-source-coverage-not-configured"
            source_coverage_spec = {}
        target_timeout_by_id[target_id] = _configured_target_timeout(
            config, target_timeout_override, target_spec,
        )
        target_configs[target_id] = {
            "target_spec": target_spec,
            "backend": target_backend,
            "binary_path": str(target_binary),
            "binary_sha256": expected_binary_sha,
            "identity_digest": target_spec.get("identity_digest"),
            "coverage_binary": str(coverage_binary) if coverage_binary else None,
            "target_timeout_seconds": target_timeout_by_id[target_id],
        }
    target_spec = target_configs[target_ids[0]]["target_spec"]
    backend = (
        "multi-target" if feedback_target == "all"
        else target_configs[feedback_target]["backend"]
    )
    target_binary = (
        Path(target_configs[feedback_target]["binary_path"])
        if feedback_target != "all" else None
    )
    coverage_binary = (
        Path(target_configs[feedback_target]["coverage_binary"])
        if feedback_target != "all" and target_configs[feedback_target]["coverage_binary"]
        else None
    )
    source_coverage_spec = target_spec.get("source_coverage") if coverage_enabled else {}
    source_coverage_spec = source_coverage_spec if isinstance(source_coverage_spec, dict) else {}
    profile = spec.get("generator_profile") if isinstance(spec.get("generator_profile"), dict) else {}
    # The outer target/coverage path is open-capability. Ours keeps one
    # reference-driven MCMC/EMI/model chain inside each generated case; the
    # selected Target consumes the completed chain during post-chain replay.
    target_timeout_seconds = _configured_target_timeout(
        config, target_timeout_override, target_spec,
    )
    if feedback_target == "all":
        target_timeout_seconds = max(target_timeout_by_id.values())
    out = Path(out).resolve()
    manifest, manifest_verified = _framework_manifest(
        out, method=method, feedback_target=feedback_target,
    )
    segment_duration = (
        DEFAULT_RUN_SECONDS if timeout_seconds is None else int(timeout_seconds)
    )
    if segment_duration <= 0:
        raise ValueError("framework-run requires a positive segment duration")
    method_dir = out / "framework" / str(method).replace("/", "-")
    method_dir.mkdir(parents=True, exist_ok=True)
    parallel_lanes = _framework_parallel_lanes(profile)
    seed_sequence = None
    if method == "Ours-Program-Full":
        if program_seed_catalog_root is None or program_seed_catalog_manifest is None:
            raise ValueError("Ours-Program-Full requires a pinned B-RVDV case catalog")
        if parallel_lanes != 1:
            raise ValueError("B-RVDV Program-Full input requires one producer lane")
        seed_sequence = _framework_program_seed_sequence(
            config, program_seed_catalog_root, program_seed_catalog_manifest,
        )
        sequence_path = method_dir / "program-seed-sequence.json"
        if sequence_path.is_file():
            if load_json(sequence_path) != seed_sequence:
                raise ValueError("Program-Full seed sequence changed across resume")
        else:
            write_json_replace(sequence_path, seed_sequence)
    elif program_seed_catalog_root is not None or program_seed_catalog_manifest is not None:
        raise ValueError("B-RVDV seed catalogs only apply to Ours-Program-Full")
    module_toggles = {
        "small_model": small_model_enabled,
        "emi": emi_enabled,
        "mcmc": mcmc_enabled,
    }
    toggles_path = method_dir / "module-toggles.json"
    if toggles_path.is_file():
        if load_json(toggles_path) != module_toggles:
            raise ValueError("framework module toggles changed across resume")
    else:
        write_json_replace(toggles_path, module_toggles)
    ablation_mode = not all(module_toggles.values())
    base_seed = int(config.get("experiment_seed", 303))
    deps = config.get("dependencies", {})
    checkout = dep(deps.get("riscv_dv"))
    coverage_config_dir = out / "framework-coverage-configs" / str(method).replace("/", "-")
    # 每个 case 保留模拟器原始 profile、trace 和身份；覆盖率汇总交给离线流程。
    isa_profile = str(
        profile.get("isa") or profile.get("isa_profile")
        or str(spec.get("lane") or "rv64i/lp64").split("/", 1)[0]
    ).strip().lower()

    def coverage_config(case_dir: Path, lane_id: int, case_index: int) -> Path:
        global_case_index = lane_id + case_index * parallel_lanes

        def batch_id_for(target_id: str, source_coverage: Mapping[str, Any]) -> str:
            batch_index = (
                0 if source_coverage.get("collector") == "dotnet"
                else global_case_index // FRAMEWORK_COVERAGE_BATCH_SIZE
            )
            return f"batch-{batch_index:05d}"

        def batch_paths(target_id: str, batch_id: str) -> tuple[Path, Path, Path]:
            batch_root = method_dir / "coverage-batches" / target_id / batch_id
            batch_raw_dir = batch_root / "raw"
            case_raw_dir = batch_raw_dir / f"case-{global_case_index:08d}" / "raw"
            return batch_root, batch_raw_dir, case_raw_dir

        if feedback_target == "all":
            target_runner_configs: dict[str, dict[str, Any]] = {}
            for target_id, row in target_configs.items():
                spec_for_target = row["target_spec"]
                source_for_target = spec_for_target.get("source_coverage") \
                    if coverage_enabled else {}
                source_for_target = dict(source_for_target) \
                    if isinstance(source_for_target, dict) else {}
                batch_id = batch_id_for(target_id, source_for_target)
                batch_root, batch_raw_dir, case_raw_dir = batch_paths(target_id, batch_id)
                for name in ("source_root", "build_root"):
                    value = source_for_target.get(name)
                    if value:
                        source_for_target[name] = str(dep(value) or value)
                source_for_target.update({
                    "profile": str(
                        batch_root / "simulator-source" / "lcov.info"
                    ),
                    "identity": str(
                        batch_root / "simulator-source" / "identity.json"
                    ),
                })
                target_identity = {
                    "id": target_id, "kind": spec_for_target.get("kind"),
                    "commit": spec_for_target.get("commit"), "lane": spec.get("lane"),
                    "isa_profile": isa_profile,
                    "execution_model": spec_for_target.get("execution_model"),
                    "translation_mode": spec_for_target.get("translation_mode"),
                    "identity_digest": spec_for_target.get("identity_digest"),
                    "binary_sha256": spec_for_target.get("binary_sha256"),
                    "coverage_binary": row.get("coverage_binary"),
                    "coverage_binary_sha256": spec_for_target.get("coverage_binary_sha256"),
                    "source_coverage": source_for_target,
                }
                coverage = {
                    "schema_version": "rq1-framework-coverage-config-v1",
                    "enabled": coverage_enabled,
                    "rv_instruction_coverage_enabled": rv_instruction_metrics_enabled(config),
                    "method": method,
                    "target_id": target_id, "backend": row["backend"],
                    "kind": spec_for_target.get("kind"), "lane": spec.get("lane"),
                    "isa_profile": isa_profile,
                    "binary_path": row.get("coverage_binary"),
                    "coverage_binary_cache": str(
                        out / "coverage-binaries" / target_id
                    ) if row.get("coverage_binary") else None,
                    "identity_binary_path": row["binary_path"],
                    "raw_dir": str(case_raw_dir),
                    "coverage_batch_id": batch_id,
                    "coverage_batch_scope": "target-batch",
                    "coverage_batch_raw_dir": str(batch_raw_dir),
                    "coverage_batch_summary_path": str(batch_root / "summary.json"),
                    "coverage_batch_session_id": None,
                    "dotnet_coverage_session_mode": None,
                    "summary_path": str(
                        case_dir / "coverage" / "targets" / target_id / "summary.json"
                    ),
                    "target": target_identity,
                    "source_coverage": source_for_target,
                }
                identity_section = {
                    "backend": row["backend"],
                    "binary_path": row["binary_path"],
                    "identity_resolution": "runtime",
                    "expected": {
                        "backend": row["backend"],
                        "identity_digest": row.get("identity_digest"),
                        "binary_sha256": row.get("binary_sha256"),
                    },
                }
                target_runner_configs[target_id] = {
                    "backend": row["backend"],
                    "binary_path": row["binary_path"],
                    "identity_digest": row.get("identity_digest"),
                    "binary_sha256": row.get("binary_sha256"),
                    "identity_section": identity_section,
                    "coverage_config": coverage,
                    "target_timeout_seconds": row["target_timeout_seconds"],
                }
            path = coverage_config_dir / f"lane-{lane_id:02d}-case-{case_index:06d}.json"
            write_json_replace(path, {
                "schema_version": "rq1-framework-target-configs-v1",
                "targets": target_runner_configs,
            })
            return path
        source_coverage = dict(source_coverage_spec)
        batch_id = batch_id_for(feedback_target, source_coverage)
        batch_root, batch_raw_dir, case_raw_dir = batch_paths(feedback_target, batch_id)
        for name in ("source_root", "build_root"):
            value = source_coverage.get(name)
            if value:
                source_coverage[name] = str(dep(value) or value)
        source_coverage.update({
            "profile": str(batch_root / "simulator-source" / "lcov.info"),
            "identity": str(batch_root / "simulator-source" / "identity.json"),
        })
        target = {
            "id": feedback_target, "kind": target_spec.get("kind"),
            "commit": target_spec.get("commit"), "lane": spec.get("lane"),
            "isa_profile": isa_profile,
            "execution_model": target_spec.get("execution_model"),
            "translation_mode": target_spec.get("translation_mode"),
            "identity_digest": target_spec.get("identity_digest"),
            "binary_sha256": target_spec.get("binary_sha256"),
            "coverage_binary": str(coverage_binary) if coverage_binary else None,
            "coverage_binary_sha256": target_spec.get("coverage_binary_sha256"),
            "source_coverage": source_coverage,
        }
        value = {
            "schema_version": "rq1-framework-coverage-config-v1",
            "enabled": coverage_enabled,
            "rv_instruction_coverage_enabled": rv_instruction_metrics_enabled(config),
            "method": method,
            "target_id": feedback_target, "backend": backend,
            "kind": target_spec.get("kind"), "lane": spec.get("lane"),
            "isa_profile": isa_profile,
            "binary_path": str(coverage_binary) if coverage_binary else None,
            "coverage_binary_cache": str(
                out / "coverage-binaries" / feedback_target
            ) if coverage_binary else None,
            "identity_binary_path": str(target_binary),
            "raw_dir": str(case_raw_dir),
            "coverage_batch_id": batch_id,
            "coverage_batch_scope": "target-batch",
            "coverage_batch_raw_dir": str(batch_raw_dir),
            "coverage_batch_summary_path": str(batch_root / "summary.json"),
            "coverage_batch_session_id": None,
            "dotnet_coverage_session_mode": None,
            "target": target,
            "source_coverage": source_coverage,
        }
        path = coverage_config_dir / f"lane-{lane_id:02d}-case-{case_index:06d}.json"
        write_json_replace(path, value)
        return path

    def build_command(
        case_dir: Path, case_seed: int, chain_seconds: float | None,
        lane_id: int, case_index: int,
    ) -> list[str]:
        if method == "Ours-Program-Full":
            cli = HERE / "framework" / "cli" / "run_program_campaign.py"
            ledger = HERE / "framework" / "evidence" / "rvgen-rq1-program-recipes.json"
            command = [sys.executable, str(cli), "--ledger", str(ledger),
                       "--root-key", "rq1-rvgen-program",
                       "--lineage-key", "rq1-rvgen-program-direct",
                       "--checkout", str(checkout), "--output", str(case_dir),
                       "--method", "B0", "--rule", "R3", "--state", "gpr",
                       "--observer", "RVOBS1",
                       "--steps", str(int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS))),
                       "--open-capability",
                       "--seed", str(case_seed),
                       "--target-timeout-seconds", str(target_timeout_seconds)]
            if feedback_target != "all":
                command += [
                    "--target-backend", backend,
                    "--target-id", feedback_target,
                ]
            pyflow = [path for path in (
                HERE / "container" / "assets" / "rvdv",
                checkout / ".analysis-out" / "rdv-deps" if checkout else None,
            ) if path is not None and path.is_dir()]
            if pyflow:
                command += ["--pyflow-pythonpath", os.pathsep.join(map(str, pyflow))]
            seed_case = seed_sequence["cases"][case_index]
            command += [
                "--seed-source", str(seed_case["source_path"]),
                "--seed-source-sha256", str(seed_case["source_sha256"]),
                "--seed-case-id", str(seed_case["case_id"]),
                "--seed-case-number", str(seed_case["case_number"]),
                "--seed-sequence-index", str(seed_case["sequence_index"]),
                "--seed-catalog-id", str(seed_sequence["catalog_id"]),
                "--seed-catalog-revision", str(seed_sequence["catalog_revision"]),
                "--seed-catalog-manifest-sha256",
                str(seed_sequence["catalog_manifest_sha256"]),
            ]
            if isinstance(seed_case.get("artifact_isa"), str):
                command += ["--seed-artifact-isa", seed_case["artifact_isa"]]
        else:
            cli = HERE / "framework" / "cli" / "run_single_campaign.py"
            ledger = HERE / "framework" / "evidence" / "rvgen-rq1-root-recipes.json"
            command = [sys.executable, str(cli), "--ledger", str(ledger),
                       "--root-key", "rq1-rvgen-rv64i-addi",
                       "--lineage-key", "rq1-rvgen-scalar-addi",
                       "--output", str(case_dir), "--method", "S1", "--rule", "R3",
                       "--observer", "RVOBS1",
                       "--steps", str(int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS))),
                       "--open-capability",
                       "--open-catalog",
                       "--seed", str(case_seed),
                       "--target-timeout-seconds", str(target_timeout_seconds),
                       "--case-index", str(lane_id + case_index * parallel_lanes),
                       ]
            if feedback_target != "all":
                command += [
                    "--target-backend", backend,
                    "--target-id", feedback_target,
                ]
        command.append(
            "--use-small-model" if small_model_enabled else "--no-small-model"
        )
        if not emi_enabled:
            command.append("--no-emi")
        if chain_seconds is not None:
            command += ["--max-seconds", str(max(1, math.ceil(chain_seconds)))]
        config_path = coverage_config(case_dir, lane_id, case_index)
        if feedback_target == "all":
            command += ["--target-configs", str(config_path)]
        else:
            command += ["--coverage-config", str(config_path)]
        if profile.get("small_model") not in (None, ""):
            command += ["--small-model", str(profile["small_model"])]
        if profile.get("target_replay_budget_seconds") not in (None, ""):
            command += [
                "--target-replay-budget-seconds",
                str(profile["target_replay_budget_seconds"]),
            ]
        command += [
            "--reference-path-reward-version",
            str(profile.get("reference_path_reward_version", "path-v2")),
        ]
        # 小模型、EMI、MCMC 的本次运行开关写入 run manifest，命令行只
        # 覆盖对应模块；配置中的正式 profile 保持不变。
        if profile.get("mcmc", True) is False or not mcmc_enabled:
            command.append("--no-mcmc")
        if profile.get("beta") not in (None, ""):
            command += ["--beta", str(profile["beta"])]
        if target_binary and feedback_target != "all":
            command += ["--target-binary", str(target_binary)]
        return command

    parallel = _run_framework_parallel_campaign(
        out, method_dir, method=method, feedback_target=feedback_target,
        backend=backend, spec=spec, profile=profile, manifest=manifest,
        manifest_verified=manifest_verified,
        segment_duration_seconds=segment_duration,
        base_seed=base_seed, command_builder=build_command,
        target_timeout_seconds=target_timeout_seconds,
        target_execution_parallelism=(
            len(target_ids) if feedback_target == "all"
            else _FRAMEWORK_TARGET_EXECUTION_PARALLELISM
        ),
        include_guest_metrics=rv_instruction_metrics_enabled(config),
        raw_coverage_only=True,
        ablation_mode=ablation_mode,
        case_count_limit_override=(
            len(seed_sequence["cases"]) if seed_sequence is not None else None
        ),
    )
    execution_complete = parallel.get("execution_complete") is True
    counts = parallel["counts"]
    framework_coverage = parallel.get("framework_coverage") \
        if isinstance(parallel.get("framework_coverage"), dict) else {
            "status": "gap", "artifact_complete": False,
            "reason": "framework-coverage-record-missing",
        }
    if coverage_gap:
        framework_coverage = {
            **framework_coverage,
            "status": "gap", "artifact_complete": False,
            "reason": coverage_gap,
        }
    coverage_gap_value = (
        framework_coverage.get("reason")
        if framework_coverage.get("status") == "gap" else None
    )
    target_coverage = parallel["target_coverage"]
    record = {
        "schema_version": "rq1-framework-run-v1", "run_id": out.name,
        "method": method, "face": "framework", "experiment_face": "framework",
        "module_toggles": module_toggles,
        "ablation_mode": ablation_mode,
        "feedback_target": feedback_target, "target_backend": backend,
        "route": spec.get("route"), "lane": spec.get("lane"),
        "steps": _framework_count(counts, "steps"),
        "reference_chain_steps": _framework_count(counts, "steps"),
        "step_budget": int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS)),
        "steps_per_case_budget": int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS)),
        "total_steps_completed": _framework_count(counts, "steps"),
        "configured_step_budget": int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS)),
        "mcmc": parallel.get("mcmc") is True,
        "configured_mcmc": profile.get("mcmc", True) is True,
        "mcmc_feedback_source": parallel.get("mcmc_feedback_source"),
        "mcmc_chain_mode": parallel.get("mcmc_chain_mode"),
        "target_execution_mode": parallel.get("target_execution_mode"),
        "parallel": True, "parallel_lanes": parallel["parallel_lanes"],
        "target_timeout_seconds": parallel.get(
            "target_timeout_seconds", target_timeout_seconds,
        ),
        "target_timeout_seconds_by_target": dict(target_timeout_by_id),
        "target_execution_parallelism": parallel["target_execution_parallelism"],
        "target_queue": parallel["target_queue"],
        "target_session_mode": parallel.get("target_session_mode"),
        "target_session_modes": parallel.get("target_session_modes", {}),
        "target_progress": parallel.get("target_progress", {}),
        "target_queue_completion": parallel.get("target_queue_completion", {}),
        "case_queue_capacity": parallel.get("case_queue_capacity"),
        "case_finalization_mode": parallel.get("case_finalization_mode", "inline"),
        "case_count": parallel["case_count"],
        "raw_coverage_only": parallel.get("raw_coverage_only") is True,
        "active_execution_seconds": parallel.get("active_execution_seconds"),
        "paused_seconds": parallel.get("paused_seconds"),
        "segment_count": parallel.get("segment_count"),
        "campaign_result_count": parallel["campaign_result_count"],
        "completed_case_count": parallel["completed_case_count"],
        "right_censored_case_count": parallel["right_censored_case_count"],
        "campaign_finalization_pending_case_count": parallel.get(
            "campaign_finalization_pending_case_count", 0
        ),
        "case_failure_count": parallel["case_failure_count"],
        "accepted_per_case": parallel["accepted_per_case"],
        "reference_parallel_limit": parallel["reference_parallel_limit"],
        "resource_policy": parallel["resource_policy"],
        "mh_attempts": _framework_count(counts, "mh_attempts"),
        "accepted": _framework_count(counts, "accepted"),
        "rejected": _framework_count(counts, "mh_rejected"),
        "target_attempted": _framework_count(counts, "target_attempted"),
        "target_tested": _framework_count(counts, "target_tested"),
        "target_gap": _framework_count(counts, "target_gap"),
        "target_unsupported": _framework_count(counts, "target_unsupported"),
        "case_skipped": _framework_count(counts, "case_skipped"),
        "target_wall_clock_pending": _framework_count(
            counts, "target_wall_clock_pending"
        ),
        "target_mismatch_candidate": _framework_count(
            counts, "target_mismatch_candidate"
        ),
        "reference_valid": _framework_count(counts, "reference_valid"),
        "counts": dict(counts), "target_coverage": target_coverage,
        "framework_coverage": framework_coverage,
        "coverage_status": framework_coverage.get("status"),
        "coverage_artifact_complete": framework_coverage.get("artifact_complete") is True,
        "manifest_verified": manifest_verified,
        "campaign_status": parallel["campaign_status"],
        "comparison_status": parallel["comparison_status"],
        "entry": {
            "status": "passed" if execution_complete else "partial",
            "reason": parallel["stop_reason"],
        },
        "stop_reason": parallel["stop_reason"],
        "lane_stop_reasons": parallel["lane_stop_reasons"],
        "online_feedback": parallel["online_feedback"],
        "status": parallel["status"], "chain": parallel["chain"],
        "execution_complete": execution_complete,
    }
    record["coverage_enabled"] = coverage_enabled
    record["coverage_status"] = framework_coverage.get("status")
    record["coverage_gap"] = coverage_gap_value
    formal_ready = _framework_formal_ready(
        method=method, manifest_verified=manifest_verified,
        execution_complete=execution_complete, status=record["status"],
        coverage_enabled=coverage_enabled,
        coverage_status=framework_coverage.get("status"),
        coverage_artifact_complete=framework_coverage.get("artifact_complete") is True,
        online_feedback=parallel["online_feedback"], profile=profile,
        mcmc=record["mcmc"],
    )
    record["formal_ready"] = formal_ready
    run_result = {
        "schema_version": "rq1-run-result-v2", "run_id": out.name,
        "phase": "framework-online", "pipeline": "framework-online-chain-v1",
        "status": record["status"], "event_count": record["target_attempted"],
        "formal_ready": formal_ready,
        "module_toggles": module_toggles,
        "ablation_mode": ablation_mode,
        "experiment_face": "framework", "manifest_verified": manifest_verified,
        "chain": parallel["chain"], "counts": dict(counts),
        "coverage_enabled": coverage_enabled,
        "coverage_status": record["coverage_status"],
        "coverage_gap": record.get("coverage_gap"),
        "coverage_artifact_complete": record["coverage_artifact_complete"],
        "raw_coverage_only": parallel.get("raw_coverage_only") is True,
        "raw_coverage": framework_coverage.get("raw_evidence"),
        "active_execution_seconds": parallel.get("active_execution_seconds"),
        "paused_seconds": parallel.get("paused_seconds"),
        "segment_count": parallel.get("segment_count"),
        "framework_coverage": framework_coverage,
        "target_coverage": target_coverage,
        "campaign_status": record["campaign_status"],
        "comparison_status": record["comparison_status"],
        # 在线 chain 与两类 coverage 共用同一份 child-case 证据。
        "execution_complete": execution_complete,
        "online_feedback": parallel["online_feedback"],
        "parallel": True, "parallel_lanes": parallel["parallel_lanes"],
        "target_timeout_seconds": parallel.get(
            "target_timeout_seconds", target_timeout_seconds,
        ),
        "target_timeout_seconds_by_target": dict(target_timeout_by_id),
        "stop_reason": parallel["stop_reason"],
        "lane_stop_reasons": parallel["lane_stop_reasons"],
        "target_execution_parallelism": parallel["target_execution_parallelism"],
        "target_queue": parallel["target_queue"],
        "target_progress": parallel.get("target_progress", {}),
        "target_queue_completion": parallel.get("target_queue_completion", {}),
        "case_queue_capacity": parallel.get("case_queue_capacity"),
        "case_finalization_mode": parallel.get("case_finalization_mode", "inline"),
        "case_count": parallel["case_count"],
        "campaign_result_count": parallel["campaign_result_count"],
        "completed_case_count": parallel["completed_case_count"],
        "right_censored_case_count": parallel["right_censored_case_count"],
        "campaign_finalization_pending_case_count": parallel.get(
            "campaign_finalization_pending_case_count", 0
        ),
        "case_failure_count": parallel["case_failure_count"],
        "steps": parallel["steps"],
        "reference_chain_steps": parallel["steps"],
        "step_budget": parallel["step_budget"],
        "steps_per_case_budget": parallel.get(
            "steps_per_case_budget", parallel["step_budget"],
        ),
        "total_steps_completed": parallel.get(
            "total_steps_completed", _framework_count(counts, "steps"),
        ),
        "mcmc": record["mcmc"],
        "configured_mcmc": record["configured_mcmc"],
        "mcmc_feedback_source": record.get("mcmc_feedback_source"),
        "mcmc_chain_mode": record.get("mcmc_chain_mode"),
        "target_execution_mode": record.get("target_execution_mode"),
        "reference_parallel_limit": parallel["reference_parallel_limit"],
        "resource_policy": parallel["resource_policy"],
        **{key: record[key] for key in (
            "method", "feedback_target", "target_backend", "mh_attempts",
            "accepted", "rejected", "target_attempted", "target_tested",
            "target_gap", "target_unsupported", "case_skipped",
            "target_wall_clock_pending", "accepted_per_case",
        )},
    }
    write_json_replace(out / "run-result.json", run_result)
    return record


def _framework_matrix_target_manifest(
    out: Path, method: str, feedback_target: str,
) -> tuple[Path, dict[str, Any]]:
    manifest = load_json(out / "execution-manifest.json")
    targets = manifest.get("targets")
    target = next(
        (item for item in targets if isinstance(item, dict)
         and item.get("id") == feedback_target),
        None,
    ) if isinstance(targets, list) else None
    if target is None:
        raise ValueError(f"framework matrix target is missing: {feedback_target}")
    methods = manifest.get("methods")
    method_row = next(
        (item for item in methods if isinstance(item, dict)
         and item.get("id") == method),
        None,
    ) if isinstance(methods, list) else None
    if method_row is None:
        raise ValueError(f"framework matrix method is missing: {method}")
    target_out = _framework_matrix_target_root(out, method, feedback_target)
    target_out.mkdir(parents=True, exist_ok=False)
    target_method_row = {
        **method_row,
        "targets": [feedback_target],
    }
    target_manifest = {
        **manifest,
        "run_id": target_out.name,
        "feedback_target": feedback_target,
        "methods": [target_method_row],
        "targets": [target],
        "matrix_parent_run_id": out.name,
        "matrix_parent": str(out),
    }
    write_json_replace(target_out / "execution-manifest.json", target_manifest)
    config_snapshot = out / "config.snapshot.json"
    if config_snapshot.is_file():
        shutil.copyfile(config_snapshot, target_out / "config.snapshot.json")
    return target_out, target_manifest


def _framework_matrix_target_root(
    out: Path, method: str, feedback_target: str,
) -> Path:
    if feedback_target not in TARGETS:
        raise ValueError(f"unknown framework matrix target: {feedback_target}")
    root = Path(out).resolve()
    target_root = root / "framework-targets" / method.replace("/", "-") / feedback_target
    try:
        target_root.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("framework matrix target escapes run root") from error
    return target_root


def _framework_matrix_write_coverage_summary(
    out: Path, method: str, results: Mapping[str, Mapping[str, Any]], *,
    include_guest_metrics: bool = True,
) -> dict[str, Any] | None:
    """Write one method-level coverage summary for a complete Target matrix."""
    rows: list[dict[str, Any]] = []
    source_targets: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    child_target_scopes: dict[str, set[tuple[str, str]]] = {}
    method_dir = out / "framework" / method.replace("/", "-")
    for target in TARGETS:
        target_root = _framework_matrix_target_root(out, method, target)
        target_result = results.get(target, {})
        coverage = target_result.get("framework_coverage")
        relative = (
            coverage.get("summary")
            if isinstance(coverage, Mapping) else None
        )
        if not isinstance(relative, str) or not relative:
            relative = "coverage/summary.json"
        summary_path = target_root / "framework" / method.replace("/", "-") / relative
        try:
            summary = load_json(summary_path)
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(summary, dict) or summary.get("schema_version") != BATCH_SCHEMA:
            continue
        target_summaries.append(summary)
        child_target_scopes[target] = {
            (str(row.get("target") or ""), str(row.get("stratum") or ""))
            for row in summary.get("targets", ())
            if isinstance(row, dict) and row.get("target")
        }
        for row in summary.get("targets", ()):
            if isinstance(row, dict):
                rows.append(deepcopy(row))
        for row in summary.get("source_targets", ()):
            if isinstance(row, dict):
                source_targets.append(deepcopy(row))
    if not rows:
        return None

    if not include_guest_metrics:
        # A matrix can be rebuilt from a child produced by an older process.
        # Do not leak its historical fixed-ISA guest fields back into the
        # active source-coverage summary merely because they are present in
        # the persisted row.
        for row in rows:
            row["guest"] = {}
            row["coverage_basis"] = {"SimSrcCov": "target-run-source-profile"}

    profiles = {
        profile_id
        for row in rows
        for profile_id in [
            (row.get("coverage_basis") or {}).get("coverage_profile_id")
        ]
        if isinstance(profile_id, str) and profile_id
    }
    case_artifact_pairs = sum(
        int((row.get("cases") or 0))
        for row in rows
        if type(row.get("cases")) is int and row.get("cases", 0) >= 0
    )
    expected_targets = set(TARGETS)
    expected_target_scopes = {
        target: (target, _target_stratum(TARGET_MODELS[target]))
        for target in expected_targets
    }
    union = (
        summarize_experiment_union(
            method, rows, profile_ids=profiles,
            case_artifact_pairs=case_artifact_pairs,
        )
        if include_guest_metrics else
        summarize_source_experiment_union(
            method, rows, source_rows=source_targets,
            case_artifact_pairs=case_artifact_pairs,
            expected_target_scopes=expected_target_scopes.values(),
        )
    )
    observed_target_scopes = {
        (str(row.get("target") or ""), str(row.get("stratum") or ""))
        for row in rows if row.get("target")
    }
    matrix_complete = len(target_summaries) == len(expected_targets) \
        and observed_target_scopes == set(expected_target_scopes.values()) \
        and all(
            child_target_scopes.get(target) == {scope}
            for target, scope in expected_target_scopes.items()
        )
    summary = {
        "schema_version": BATCH_SCHEMA,
        "status": (
            "observed" if matrix_complete
            and all(item.get("status") == "observed" for item in target_summaries)
            else "partial" if target_summaries else "gap"
        ),
        "comparability": "paired-corpus",
        "experiment_unions": [union],
        "targets": rows,
        "source_targets": source_targets,
        "framework": {
            "case_count": sum(
                int((item.get("framework") or {}).get("case_count") or 0)
                for item in target_summaries
            ),
            "coverage_summary_case_count": sum(
                int((item.get("framework") or {}).get(
                    "coverage_summary_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "completed_case_count": sum(
                int((item.get("framework") or {}).get(
                    "completed_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "right_censored_case_count": sum(
                int((item.get("framework") or {}).get(
                    "right_censored_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "missing_case_count": sum(
                int((item.get("framework") or {}).get(
                    "missing_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "missing_campaign_coverage_case_count": sum(
                int((item.get("framework") or {}).get(
                    "missing_campaign_coverage_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "incomplete_guest_case_count": sum(
                int((item.get("framework") or {}).get(
                    "incomplete_guest_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "campaign_incomplete_case_count": sum(
                int((item.get("framework") or {}).get(
                    "campaign_incomplete_case_count", 0
                ) or 0)
                for item in target_summaries
            ),
            "trace_incomplete_cases": sum(
                int((item.get("framework") or {}).get(
                    "trace_incomplete_cases", 0
                ) or 0)
                for item in target_summaries
            ),
            "target_wall_clock_pending": sum(
                int((item.get("framework") or {}).get(
                    "target_wall_clock_pending", 0
                ) or 0)
                for item in target_summaries
            ),
            "target": "all",
            "target_count": len(target_summaries),
        },
    }
    summarize_opcode_catalog_target_rows(
        summary, expected_target_scopes=expected_target_scopes.values(),
    )

    if include_guest_metrics:
        # Each child summary has already unioned its own cases. Present one
        # logical entry per Target to the RV aggregator so it does not mistake
        # a target-union row for a missing per-case record.
        rv_view = {
            "targets": [
                {**deepcopy(row), "cases": 1, "observed_cases": 1}
                for row in rows
            ],
            "experiment_unions": [deepcopy(union)],
        }
        has_legacy_guest_metrics = any(
            isinstance(row.get("rv_instruction_coverage"), Mapping)
            or isinstance(row.get("guest"), Mapping)
            and any(
                name in row.get("guest", {})
                for name in ("PCov", "ICov-encoding", "ICov-type", "CCov")
            )
            for row in rv_view["targets"] if isinstance(row, Mapping)
        )
        if has_legacy_guest_metrics:
            aggregate_rv_instruction_summary(rv_view, rv_view["targets"])
        by_key = {
            tuple(str(row.get(name) or "") for name in (
                "method", "route", "lane", "stratum", "target",
            )): row
            for row in rv_view["targets"]
            if isinstance(row, dict)
        }
        for row in rows:
            merged = by_key.get(tuple(str(row.get(name) or "") for name in (
                "method", "route", "lane", "stratum", "target",
            )))
            if not isinstance(merged, dict):
                continue
            if isinstance(merged.get("rv_instruction_coverage"), Mapping):
                row["rv_instruction_coverage"] = merged["rv_instruction_coverage"]
            merged_guest = merged.get("guest")
            guest = row.get("guest")
            if isinstance(merged_guest, Mapping) and isinstance(guest, dict):
                for name in ("GenCov", "ExecCov", "OpcodeCov"):
                    if isinstance(merged_guest.get(name), Mapping):
                        guest[name] = merged_guest[name]
                if isinstance(merged_guest.get("metrics"), Mapping):
                    guest["metrics"] = {
                        **(
                            guest.get("metrics", {})
                            if isinstance(guest.get("metrics"), Mapping) else {}
                        ),
                        **merged_guest["metrics"],
                    }
        summary["experiment_unions"] = rv_view["experiment_unions"]
        if isinstance(rv_view.get("rv_instruction_coverage"), Mapping):
            summary["rv_instruction_coverage"] = rv_view["rv_instruction_coverage"]
    else:
        summary["experiment_unions"] = [union]
    if not matrix_complete and include_guest_metrics:
        # Per-Target observations remain usable, while a union that omits any
        # configured Target must not retain a complete experiment percentage.
        gate_experiment_unions(
            summary, "partial", "framework-target-matrix-incomplete",
            include_targets=False,
        )
    summary_path = method_dir / "coverage" / "summary.json"
    write_json_replace(summary_path, summary)
    return {
        "summary": str(summary_path.relative_to(out)),
        "status": summary["status"],
        "artifact_complete": summary_path.is_file(),
    }


def _framework_matrix_record(
    config: dict[str, Any], out: Path, *, method: str,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild a persisted legacy result whose Target lanes owned separate chains."""
    target_results: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    persisted_results: dict[str, Mapping[str, Any]] = {}
    for target, in_memory_result in results.items():
        target_root = _framework_matrix_target_root(out, method, target)
        try:
            persisted_result = load_json(target_root / "run-result.json")
        except (OSError, TypeError, ValueError):
            persisted_result = None
        persisted_results[target] = (
            {**in_memory_result, **persisted_result}
            if isinstance(persisted_result, dict) else in_memory_result
        )
    results = persisted_results
    for target, result in results.items():
        target_root = _framework_matrix_target_root(out, method, target)
        for name, value in (result.get("counts", {}) if isinstance(result.get("counts"), dict) else {}).items():
            if isinstance(name, str) and type(value) is int and value >= 0:
                counts[name] += value
        target_results[target] = {
            "status": result.get("status"),
            "execution_complete": result.get("execution_complete") is True,
            "formal_ready": result.get("formal_ready") is True,
            "target_attempted": _framework_count(result, "target_attempted"),
            "target_tested": _framework_count(result, "target_tested"),
            "target_gap": _framework_count(result, "target_gap"),
            "target_unsupported": _framework_count(result, "target_unsupported"),
            "case_skipped": _framework_count(result, "case_skipped"),
            "target_wall_clock_pending": _framework_count(
                result, "target_wall_clock_pending",
            ),
            "target_backend": result.get("target_backend"),
            "stop_reason": result.get("stop_reason"),
            "run_result": str((target_root / "run-result.json").relative_to(out)),
            "framework_root": str(target_root.relative_to(out)),
            "chain": result.get("chain"),
            "coverage_status": result.get("coverage_status"),
            "coverage_artifact_complete": result.get("coverage_artifact_complete") is True,
        }
    target_chain = {
        target: row.get("chain") for target, row in target_results.items()
    }
    def uniform_target_field(name: str) -> object:
        values = {
            result.get(name)
            for result in results.values()
            if isinstance(result.get(name), str) and result.get(name)
        }
        if len(values) == 1:
            return next(iter(values))
        if values:
            return "mixed"
        return None

    profile = next(
        (
            item.get("generator_profile")
            for item in config.get("framework_methods", ())
            if isinstance(item, dict) and item.get("id") == method
            and isinstance(item.get("generator_profile"), dict)
        ),
        {},
    )
    step_budget = int(profile.get("steps", FRAMEWORK_CANONICAL_STEPS) or 0)
    steps_by_target = {
        target: result.get("steps")
        for target, result in results.items()
        if isinstance(result.get("steps"), int)
    }
    step_budget_by_target = {
        target: result.get("step_budget")
        for target, result in results.items()
        if isinstance(result.get("step_budget"), int)
    }
    reference_chain_steps_by_target = {
        target: result.get("reference_chain_steps", result.get("steps"))
        for target, result in results.items()
        if isinstance(result.get("reference_chain_steps", result.get("steps")), int)
    }
    execution_complete = bool(target_results) and all(
        row["execution_complete"] for row in target_results.values()
    )
    formal_ready = bool(target_results) and all(
        row["formal_ready"] for row in target_results.values()
    )
    passed = bool(target_results) and all(
        row["status"] == "passed" for row in target_results.values()
    )
    coverage_recorded = bool(target_results) and all(
        row["coverage_status"] == "recorded"
        and row["coverage_artifact_complete"]
        for row in target_results.values()
    )
    coverage_disabled = bool(target_results) and all(
        row["coverage_status"] == "disabled" for row in target_results.values()
    )
    coverage_partial = bool(target_results) and all(
        row["coverage_status"] in {"recorded", "partial"}
        for row in target_results.values()
    ) and any(
        row["coverage_status"] == "partial" for row in target_results.values()
    )
    framework_coverage = {
        "schema_version": "rq1-framework-coverage-matrix-v1",
        "status": (
            "disabled" if coverage_disabled else
            "recorded" if coverage_recorded else
            "partial" if coverage_partial else "gap"
        ),
        "artifact_complete": coverage_recorded,
        "targets": {
            target: result.get("framework_coverage")
            for target, result in results.items()
        },
    }
    coverage_flags = {
        result.get("coverage_enabled")
        for result in results.values()
        if isinstance(result.get("coverage_enabled"), bool)
    }
    coverage_enabled = (
        next(iter(coverage_flags))
        if len(coverage_flags) == 1 else simulator_coverage_enabled()
    )
    record = {
        "schema_version": "rq1-framework-matrix-v1",
        "run_id": out.name,
        "method": method,
        "face": "framework",
        "experiment_face": "framework",
        "feedback_target": "all",
        "target_backends": {
            target: result.get("target_backend") for target, result in results.items()
        },
        "route": next(
            (item.get("route") for item in config.get("framework_methods", ())
             if isinstance(item, dict) and item.get("id") == method),
            None,
        ),
        "target_results": target_results,
        "counts": dict(counts),
        # Matrix counters sum the seven isolated Target lanes.  Keep the
        # per-lane values beside them so ``steps=10`` is never mistaken for a
        # matrix-wide step count or for one shared in-memory chain.
        "matrix_aggregation": "sum-target-lanes",
        "chain_scope": "isolated-target-lanes",
        "steps": sum(steps_by_target.values()),
        "step_budget": step_budget,
        "steps_per_case_budget": step_budget,
        "total_steps_completed": sum(steps_by_target.values()),
        "configured_step_budget": step_budget,
        "steps_by_target": steps_by_target,
        "step_budget_by_target": step_budget_by_target,
        "reference_chain_steps_by_target": reference_chain_steps_by_target,
        "mcmc": bool(target_results) and all(
            result.get("mcmc") is True for result in results.values()
        ),
        "configured_mcmc": bool(profile.get("mcmc", True)),
        "mcmc_feedback_source": uniform_target_field("mcmc_feedback_source"),
        "mcmc_chain_mode": uniform_target_field("mcmc_chain_mode"),
        "target_execution_mode": uniform_target_field("target_execution_mode"),
        "status": "passed" if passed else "partial",
        "execution_complete": execution_complete,
        "formal_ready": formal_ready,
        "event_count": _framework_count(dict(counts), "target_attempted"),
        "target_attempted": _framework_count(dict(counts), "target_attempted"),
        "target_tested": _framework_count(dict(counts), "target_tested"),
        "target_gap": _framework_count(dict(counts), "target_gap"),
        "target_unsupported": _framework_count(dict(counts), "target_unsupported"),
        "case_skipped": _framework_count(dict(counts), "case_skipped"),
        "target_wall_clock_pending": _framework_count(
            dict(counts), "target_wall_clock_pending",
        ),
        "accepted": _framework_count(dict(counts), "accepted"),
        "rejected": _framework_count(dict(counts), "mh_rejected"),
        "mh_attempts": _framework_count(dict(counts), "mh_attempts"),
        "case_count": sum(_framework_count(result, "case_count") for result in results.values()),
        "campaign_result_count": sum(
            _framework_count(result, "campaign_result_count") for result in results.values()
        ),
        "completed_case_count": sum(
            _framework_count(result, "completed_case_count") for result in results.values()
        ),
        "right_censored_case_count": sum(
            _framework_count(result, "right_censored_case_count") for result in results.values()
        ),
        "case_failure_count": sum(
            _framework_count(result, "case_failure_count") for result in results.values()
        ),
        "framework_coverage": framework_coverage,
        "coverage_enabled": coverage_enabled,
        "coverage_status": framework_coverage["status"],
        "coverage_artifact_complete": framework_coverage["artifact_complete"],
        "target_coverage": {
            target: result.get("target_coverage") for target, result in results.items()
        },
        "online_feedback": bool(target_results) and all(
            result.get("online_feedback") is True for result in results.values()
        ),
        "chain": {
            "mode": "framework-target-matrix",
            "targets": target_chain,
            "sealed": bool(target_results) and all(
                isinstance(chain, dict) and chain.get("sealed") is True
                for chain in target_chain.values()
            ),
            "integrity_verified": bool(target_results) and all(
                isinstance(chain, dict) and chain.get("integrity_verified") is True
                for chain in target_chain.values()
            ),
        },
        "target_execution_parallelism": {
            target: result.get("target_execution_parallelism")
            for target, result in results.items()
        },
        "stop_reasons": {
            target: result.get("stop_reason") for target, result in results.items()
        },
    }
    method_dir = out / "framework" / method.replace("/", "-")
    method_dir.mkdir(parents=True, exist_ok=True)
    matrix_coverage = _framework_matrix_write_coverage_summary(
        out, method, results,
        include_guest_metrics=rv_instruction_metrics_enabled(config),
    )
    if matrix_coverage is not None:
        framework_coverage["summary"] = matrix_coverage["summary"]
    write_json_replace(method_dir / "matrix-result.json", record)
    write_json_replace(out / "run-result.json", record)
    return record


def _framework_matrix_target_ids(config: Mapping[str, Any], method: str) -> tuple[str, ...]:
    spec = next(
        (item for item in config.get("framework_methods", ())
         if isinstance(item, dict) and item.get("id") == method),
        None,
    )
    declared = spec.get("feedback_targets") if isinstance(spec, dict) else None
    targets = tuple(target for target in TARGETS if target in declared) \
        if isinstance(declared, list) else ()
    if targets != TARGETS:
        raise ValueError(
            f"framework matrix requires all Targets for {method}: {list(targets)}"
        )
    return targets


def _framework_verify_matrix_artifacts(out: Path, method: str) -> bool:
    """Verify a legacy Target matrix whose lanes each stored an isolated campaign."""
    try:
        manifest = load_json(out / "execution-manifest.json")
        result = load_json(out / "run-result.json")
        if (
            manifest.get("feedback_target") != "all"
            or manifest.get("method_filter") != method
            or result.get("schema_version") != "rq1-framework-matrix-v1"
        ):
            return False
        target_results = result.get("target_results")
        if not isinstance(target_results, dict) or set(target_results) != set(TARGETS):
            return False
        if result.get("chain", {}).get("sealed") is not True \
                or result.get("chain", {}).get("integrity_verified") is not True:
            return False
        for target in TARGETS:
            row = target_results.get(target)
            if not isinstance(row, dict):
                return False
            target_root = _framework_matrix_target_root(out, method, target)
            if (
                row.get("framework_root") != str(target_root.relative_to(out))
                or row.get("run_result") != str(
                    (target_root / "run-result.json").relative_to(out)
                )
            ):
                return False
            target_method_dir = target_root / "framework" / method.replace("/", "-")
            if not _framework_verify_parallel_chain_artifacts(
                target_root, target_method_dir
            ):
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def run_framework_campaign(
    config: dict[str, Any], out: Path, *, method: str,
    feedback_target: str = "T-LRSV-INT",
    timeout_seconds: int | None = None,
    target_timeout_override: int | None = None,
    program_seed_catalog_root: Path | None = None,
    program_seed_catalog_manifest: Path | None = None,
    small_model_enabled: bool = True,
    emi_enabled: bool = True,
    mcmc_enabled: bool = True,
) -> dict[str, Any]:
    """运行一条 MCMC case 链，并并行回放其候选到所选 Target 集合。"""
    seed_catalog_args = {}
    if program_seed_catalog_root is not None or program_seed_catalog_manifest is not None:
        seed_catalog_args = {
            "program_seed_catalog_root": program_seed_catalog_root,
            "program_seed_catalog_manifest": program_seed_catalog_manifest,
        }
    return _run_framework_campaign_single(
        config, out, method=method, feedback_target=feedback_target,
        timeout_seconds=timeout_seconds,
        target_timeout_override=target_timeout_override,
        small_model_enabled=small_model_enabled,
        emi_enabled=emi_enabled,
        mcmc_enabled=mcmc_enabled,
        **seed_catalog_args,
    )


def _finalize_framework_matrix_campaign(
    out: Path, config_path: Path, *, manifest: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Finalize an older per-Target campaign matrix already present on disk."""
    out = Path(out).resolve()
    config, _ = _load_frozen_config(out, config_path)
    method_spec = next(
        (item for item in config.get("framework_methods", ())
         if isinstance(item, dict) and item.get("id") == method),
        {},
    )
    profile = method_spec.get("generator_profile") \
        if isinstance(method_spec.get("generator_profile"), dict) else None
    target_finalizations: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        target_root = _framework_matrix_target_root(out, method, target)
        if not (target_root / "execution-manifest.json").is_file():
            target_finalizations[target] = {
                "status": "partial",
                "reason": "matrix-target-manifest-missing",
                "root": str(target_root.relative_to(out)),
            }
            continue
        try:
            child = finalize_framework_campaign(target_root, config_path)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            child = {
                "status": "partial",
                "reason": f"matrix-target-finalize:{type(error).__name__}",
                "error": str(error)[:500],
            }
        target_finalizations[target] = {
            **child,
            "root": str(target_root.relative_to(out)),
        }

    try:
        top_result = load_json(out / "run-result.json")
    except (OSError, TypeError, ValueError):
        top_result = {}
    persisted_results: dict[str, Mapping[str, Any]] = {}
    for target in TARGETS:
        target_root = _framework_matrix_target_root(out, method, target)
        try:
            child_result = load_json(target_root / "run-result.json")
        except (OSError, TypeError, ValueError):
            child_result = None
        if isinstance(child_result, dict):
            persisted_results[target] = child_result

    # The parent can be killed after all Target workers have durably written
    # their child ledgers but before the matrix result itself is written.
    # Rebuild that one missing parent record from the child results so the
    # wrapper can classify the unfinished suffix as right-censored instead of
    # turning a recoverable timebox into a container failure.
    child_coverage_statuses = {
        child.get("coverage_status")
        for child in persisted_results.values()
        if isinstance(child, Mapping)
    }
    child_target_wall_clock_pending = sum(
        _framework_count(child, "target_wall_clock_pending")
        for child in persisted_results.values()
    )
    rebuild_matrix = (
        not isinstance(top_result, dict)
        or not isinstance(top_result.get("target_results"), dict)
        or (
            top_result.get("coverage_status") == "gap"
            and child_coverage_statuses
            and child_coverage_statuses <= {"recorded", "partial"}
        )
        or (
            child_target_wall_clock_pending > 0
            and top_result.get("target_wall_clock_pending")
            != child_target_wall_clock_pending
        )
    )
    if rebuild_matrix:
        try:
            top_result = _framework_matrix_record(
                config, out, method=method, results=persisted_results,
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            top_result = {}
    matrix_coverage = _framework_matrix_write_coverage_summary(
        out, method, persisted_results,
        include_guest_metrics=rv_instruction_metrics_enabled(config),
    )
    if matrix_coverage is not None and isinstance(top_result, dict):
        framework_coverage = top_result.get("framework_coverage")
        if not isinstance(framework_coverage, dict):
            framework_coverage = {}
            top_result["framework_coverage"] = framework_coverage
        framework_coverage["summary"] = matrix_coverage["summary"]
        write_json_replace(out / "run-result.json", top_result)
    manifest_verified = (
        manifest.get("experiment_face") == "framework"
        and manifest.get("action") == "framework-run"
        and manifest.get("method_filter") == method
        and manifest.get("feedback_target") == "all"
    )
    matrix_verified = _framework_verify_matrix_artifacts(out, method)
    wrapper = {}
    try:
        wrapper = load_json(out / "wrapper-result.json")
    except (OSError, TypeError, ValueError):
        pass
    wrapper_verified = (
        os.environ.get("RQ1_FINALIZE_IN_CONTAINER") == "1"
        and not (out / "wrapper-result.json").is_file()
    ) or (
        wrapper.get("status") == "completed"
        and wrapper.get("exit_code") == 0
    )
    coverage_complete = (
        top_result.get("coverage_enabled") is False
        or (
            top_result.get("coverage_status") == "recorded"
            and top_result.get("coverage_artifact_complete") is True
        )
    )
    all_children_sealed = bool(target_finalizations) and all(
        row.get("status") == "sealed"
        for row in target_finalizations.values()
    )
    formal_ready = _framework_formal_ready(
        method=method, manifest_verified=manifest_verified,
        execution_complete=top_result.get("execution_complete") is True,
        status=top_result.get("status"),
        coverage_enabled=top_result.get("coverage_enabled") is True,
        coverage_status=top_result.get("coverage_status"),
        coverage_artifact_complete=top_result.get(
            "coverage_artifact_complete"
        ) is True,
        online_feedback=top_result.get("online_feedback") is True,
        profile=profile,
        mcmc=top_result.get("mcmc", profile.get("mcmc") is True),
    )
    status = (
        "sealed"
        if manifest_verified and matrix_verified and wrapper_verified
        and all_children_sealed and top_result.get("status") == "passed"
        and top_result.get("execution_complete") is True
        and coverage_complete and formal_ready
        else "partial"
    )
    result = {
        "schema_version": "rq1-framework-matrix-finalize-v1",
        "run_id": out.name,
        "experiment_face": "framework",
        "method": method,
        "feedback_target": "all",
        "status": status,
        "manifest_verified": manifest_verified,
        "chain_verified": matrix_verified,
        "wrapper_verified": wrapper_verified,
        "run_status": top_result.get("status"),
        "execution_complete": top_result.get("execution_complete") is True,
        "formal_ready": formal_ready,
        "coverage_enabled": top_result.get("coverage_enabled"),
        "coverage_status": top_result.get("coverage_status"),
        "coverage_artifact_complete": (
            top_result.get("coverage_artifact_complete") is True
        ),
        "target_finalizations": target_finalizations,
        "targets_sealed": sum(
            row.get("status") == "sealed"
            for row in target_finalizations.values()
        ),
    }
    write_json_replace(out / "framework-finalize.json", result)
    return result


def _recovery_partial_reason(out: Path, result: dict[str, Any]) -> str:
    """Use the wrapper's normalized stop provenance before legacy markers."""
    for relative in (
        "stop-provenance.json",
        "wall-clock-exhaustion.json",
        "external-interruption.json",
    ):
        try:
            marker = load_json(Path(out) / relative)
        except (OSError, TypeError, ValueError):
            continue
        reason = marker.get("reason_code") if isinstance(marker, dict) else None
        if isinstance(reason, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", reason):
            return reason
    recorded_stop = result.get("stop_reason")
    if isinstance(recorded_stop, str) and recorded_stop:
        return (
            "execution-wall-clock-exhausted"
            if recorded_stop == "deadline" else recorded_stop
        )
    recorded_reason = result.get("partial_reason")
    if isinstance(recorded_reason, str) and recorded_reason not in {
        "", "interrupted-unknown",
    }:
        return recorded_reason
    return "interrupted-unknown"


def finalize_framework_campaign(
    out: Path, config_path: Path = DEFAULT_CONFIG, *,
    recovery_partial: bool = False,
) -> dict[str, Any]:
    """重算并封存一个 framework campaign，不读取 comparison queue/ledger。"""
    out = Path(out).resolve()
    config, _ = _load_frozen_config(out, config_path)
    try:
        manifest = load_json(out / "execution-manifest.json")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("framework finalize manifest is invalid") from exc
    method = manifest.get("method_filter")
    feedback_target = manifest.get("feedback_target")
    if not isinstance(method, str) or not isinstance(feedback_target, str):
        raise ValueError("framework finalize requires method and feedback target")
    recovery_reason = (
        _recovery_partial_reason(out, {}) if recovery_partial else None
    )
    method_spec = next(
        (item for item in config.get("framework_methods", ())
         if isinstance(item, dict) and item.get("id") == method),
        {},
    )
    profile = method_spec.get("generator_profile") \
        if isinstance(method_spec.get("generator_profile"), dict) else None
    if feedback_target == "all":
        try:
            existing_result = load_json(out / "run-result.json")
        except (OSError, TypeError, ValueError):
            existing_result = {}
        if existing_result.get("schema_version") == "rq1-framework-matrix-v1":
            return _finalize_framework_matrix_campaign(
                out, config_path, manifest=manifest, method=method,
            )
    _, manifest_verified = _framework_manifest(
        out, method=method, feedback_target=feedback_target,
    )
    method_dir = out / "framework" / method.replace("/", "-")
    parallel_result_path = method_dir / "parallel-result.json"
    is_parallel = parallel_result_path.is_file()
    recovered_parallel = False
    chain_verified = (
        _framework_verify_parallel_chain_artifacts(out, method_dir)
        if is_parallel else _framework_verify_chain_artifacts(out, method_dir)
    )
    if is_parallel:
        try:
            parallel = load_json(parallel_result_path)
        except (OSError, TypeError, ValueError):
            parallel = {}
    else:
        parallel = {}
    if is_parallel or not chain_verified:
        recovered = _recover_framework_parallel_campaign(
            out, config, manifest, recovery_reason=recovery_reason,
        )
        if recovered is not None:
            parallel = recovered
            is_parallel = True
            recovered_parallel = True
            chain_verified = _framework_verify_parallel_chain_artifacts(
                out, method_dir,
    )
    parallel_rebuilt = False
    target_queue_completion: dict[str, Any] = {}
    if is_parallel:
        target_ids = (
            tuple(TARGETS) if feedback_target == "all" else (feedback_target,)
        )
        sidecar_close = _framework_merge_target_queue_sidecars(
            out, method_dir, target_ids,
        )
        target_queue_completion = sidecar_close
        parallel_rebuilt = bool(sidecar_close.get("merged_case_count"))
    if is_parallel and isinstance(parallel, dict):
        original_parallel = parallel
        parallel = _framework_finalize_pending_cases(
            out, config, manifest, parallel,
            manifest_verified=manifest_verified,
        )
        parallel_rebuilt = parallel is not original_parallel or parallel_rebuilt
        chain_verified = _framework_verify_parallel_chain_artifacts(
            out, method_dir,
        )
        if not target_queue_completion:
            target_queue_completion = _framework_target_queue_completion(
                method_dir,
                tuple(TARGETS) if feedback_target == "all" else (feedback_target,),
            )
        if parallel.get("raw_coverage_only") is True:
            parallel["target_queue_completion"] = target_queue_completion
            finalized_coverage = None
            if manifest.get("coverage_enabled") is True:
                finalized_coverage = _framework_finalize_raw_only_batches(
                    out, method, feedback_target, parallel,
                    include_guest_metrics=rv_instruction_metrics_enabled(config),
                )
            if isinstance(finalized_coverage, Mapping):
                parallel_rebuilt = True
                parallel["framework_coverage"] = dict(finalized_coverage)
                parallel["coverage_status"] = finalized_coverage.get("status", "gap")
                parallel["coverage_artifact_complete"] = (
                    finalized_coverage.get("artifact_complete") is True
                )
                parallel["raw_coverage_only"] = False
                parallel["case_finalization_mode"] = "offline-finalized"
            # A finalizer may run while Target consumers are still processing.
            # Preserve producer completion and expose Target backlog separately.
            parallel["execution_complete"] = (
                parallel.get("execution_complete") is True
            )
            write_json_replace(parallel_result_path, parallel)
    execution_complete = (
        _framework_parallel_execution_complete(
            parallel.get("lanes"), parallel.get("chain"),
        ) if is_parallel else False
    )
    run_result = {}
    run_result_path = out / "run-result.json"
    if run_result_path.is_file():
        try:
            run_result = load_json(run_result_path)
        except (OSError, TypeError, ValueError):
            run_result = {}
    if not run_result and isinstance(parallel, dict):
        counts = dict(parallel.get("counts", {}))
        run_result = {
            "schema_version": "rq1-run-result-v2",
            "run_id": out.name,
            "phase": "framework-online",
            "pipeline": "framework-online-chain-v1",
            "status": parallel.get("status", "partial"),
            "event_count": _framework_count(counts, "target_attempted"),
            "formal_ready": False,
            "experiment_face": "framework",
            "manifest_verified": manifest_verified,
            "chain": parallel.get("chain"),
            "counts": counts,
            "coverage_enabled": manifest.get("coverage_enabled") is True,
            "coverage_status": parallel.get("coverage_status"),
            "coverage_artifact_complete": (
                parallel.get("coverage_artifact_complete") is True
            ),
            "raw_coverage_only": parallel.get("raw_coverage_only") is True,
            "raw_coverage": (
                parallel.get("framework_coverage", {}).get("raw_evidence")
                if isinstance(parallel.get("framework_coverage"), Mapping) else None
            ),
            "active_execution_seconds": parallel.get("active_execution_seconds"),
            "paused_seconds": parallel.get("paused_seconds"),
            "segment_count": parallel.get("segment_count"),
            "framework_coverage": parallel.get("framework_coverage"),
            "target_coverage": parallel.get("target_coverage"),
            "target_queue_completion": parallel.get("target_queue_completion", target_queue_completion),
            "campaign_status": parallel.get("campaign_status"),
            "comparison_status": parallel.get("comparison_status"),
            "execution_complete": parallel.get("execution_complete") is True,
            "online_feedback": parallel.get("online_feedback") is True,
            "parallel": True,
            "parallel_lanes": parallel.get("parallel_lanes"),
            "steps": parallel.get("steps"),
            "step_budget": parallel.get("step_budget"),
            "mh_attempts": _framework_count(counts, "mh_attempts"),
            "accepted": _framework_count(counts, "accepted"),
            "rejected": _framework_count(counts, "mh_rejected"),
            "target_attempted": _framework_count(counts, "target_attempted"),
            "target_tested": _framework_count(counts, "target_tested"),
            "target_gap": _framework_count(counts, "target_gap"),
            "target_unsupported": _framework_count(counts, "target_unsupported"),
            "case_skipped": _framework_count(counts, "case_skipped"),
            "target_wall_clock_pending": _framework_count(
                counts, "target_wall_clock_pending"
            ),
            "accepted_per_case": parallel.get("accepted_per_case"),
            "method": method,
            "feedback_target": feedback_target,
            "target_backend": parallel.get("target_backend"),
            "stop_reason": parallel.get("stop_reason"),
        }
    if (recovered_parallel or parallel_rebuilt) and isinstance(run_result, dict):
        run_result.update({
            "status": parallel.get("status", "partial"),
            "execution_complete": execution_complete,
            "chain": parallel.get("chain"),
            "counts": dict(parallel.get("counts", {})),
            "campaign_status": parallel.get("campaign_status"),
            "comparison_status": parallel.get("comparison_status"),
            "coverage_status": parallel.get("coverage_status"),
            "coverage_artifact_complete": parallel.get(
                "coverage_artifact_complete", False,
            ),
            "raw_coverage_only": parallel.get("raw_coverage_only") is True,
            "raw_coverage": (
                parallel.get("framework_coverage", {}).get("raw_evidence")
                if isinstance(parallel.get("framework_coverage"), Mapping) else None
            ),
            "active_execution_seconds": parallel.get("active_execution_seconds"),
            "paused_seconds": parallel.get("paused_seconds"),
            "segment_count": parallel.get("segment_count"),
            "framework_coverage": parallel.get("framework_coverage"),
            "target_coverage": parallel.get("target_coverage"),
            "target_queue_completion": parallel.get("target_queue_completion", target_queue_completion),
            "online_feedback": parallel.get("online_feedback") is True,
            "formal_ready": _framework_formal_ready(
                method=method, manifest_verified=manifest_verified,
                execution_complete=execution_complete,
                status=parallel.get("status"),
                coverage_enabled=run_result.get("coverage_enabled") is True,
                coverage_status=parallel.get("coverage_status"),
                coverage_artifact_complete=parallel.get(
                    "coverage_artifact_complete"
                ) is True,
                online_feedback=parallel.get("online_feedback") is True,
                profile=profile,
                mcmc=parallel.get("mcmc", profile.get("mcmc") is True),
            ),
            "case_finalization_mode": parallel.get("case_finalization_mode"),
            "case_count": parallel.get("case_count"),
            "campaign_result_count": parallel.get("campaign_result_count"),
            "completed_case_count": parallel.get("completed_case_count"),
            "right_censored_case_count": parallel.get("right_censored_case_count"),
            "campaign_finalization_pending_case_count": parallel.get(
                "campaign_finalization_pending_case_count"
            ),
            "case_failure_count": parallel.get("case_failure_count"),
            "target_wall_clock_pending": parallel.get("target_wall_clock_pending"),
        })
        write_json_replace(out / "run-result.json", run_result)
    if is_parallel and isinstance(run_result, dict):
        formal_ready = _framework_formal_ready(
            method=method, manifest_verified=manifest_verified,
            execution_complete=execution_complete,
            status=run_result.get("status"),
            coverage_enabled=run_result.get("coverage_enabled") is True,
            coverage_status=run_result.get("coverage_status"),
            coverage_artifact_complete=run_result.get(
                "coverage_artifact_complete"
            ) is True,
            online_feedback=(
                parallel.get("online_feedback") is True
                if isinstance(parallel, dict)
                else run_result.get("online_feedback") is True
            ),
            profile=profile,
            mcmc=run_result.get("mcmc", profile.get("mcmc") is True),
        )
        if run_result.get("formal_ready") != formal_ready:
            run_result["formal_ready"] = formal_ready
            write_json_replace(out / "run-result.json", run_result)
    wrapper = {}
    wrapper_path = out / "wrapper-result.json"
    if wrapper_path.is_file():
        try:
            wrapper = load_json(wrapper_path)
        except (OSError, TypeError, ValueError):
            wrapper = {}
    wrapper_verified = (
        os.environ.get("RQ1_FINALIZE_IN_CONTAINER") == "1"
        and not wrapper_path.is_file()
    ) or (
        wrapper.get("status") == "completed"
        and wrapper.get("exit_code") == 0
    )
    raw_coverage_only = run_result.get("raw_coverage_only") is True
    framework_coverage = run_result.get("framework_coverage")
    # raw-only is an evidence-preserving pause mode, not a sealed coverage
    # result.  It becomes sealable only after the drain-gated collector above
    # changes the run back to an aggregated coverage result.
    raw_coverage_complete = False
    framework_coverage_complete = (
        run_result.get("coverage_enabled") is False
        or raw_coverage_complete
        or (
            run_result.get("coverage_artifact_complete") is True
            and run_result.get("coverage_status") == "recorded"
        )
    )
    status = "sealed" if manifest_verified and chain_verified and wrapper_verified \
        and run_result.get("status") == "passed" and framework_coverage_complete else "partial"
    result = {
        "schema_version": "rq1-framework-finalize-v1",
        "run_id": out.name, "experiment_face": "framework", "method": method,
        "feedback_target": feedback_target, "status": status,
        "raw_coverage_only": raw_coverage_only,
        "raw_coverage": (
            framework_coverage.get("raw_evidence")
            if isinstance(framework_coverage, Mapping) else None
        ),
        "manifest_verified": manifest_verified,
        "chain_verified": chain_verified,
        "wrapper_verified": wrapper_verified,
        "run_status": run_result.get("status"),
        "campaign_status": run_result.get("campaign_status"),
        "comparison_status": run_result.get("comparison_status"),
        "execution_complete": run_result.get("execution_complete") is True,
        "online_feedback": run_result.get("online_feedback") is True,
        "coverage_enabled": run_result.get("coverage_enabled"),
        "coverage_status": run_result.get("coverage_status"),
        "coverage_artifact_complete": run_result.get("coverage_artifact_complete") is True,
        "coverage_gap": run_result.get("coverage_gap"),
        "parallel": is_parallel,
        "case_count": parallel.get("case_count") if is_parallel else None,
        "accepted_per_case": run_result.get("accepted_per_case"),
        "seal": str((method_dir / ("parallel-seal.json" if is_parallel else "seal.json")).relative_to(out))
        if (method_dir / ("parallel-seal.json" if is_parallel else "seal.json")).is_file()
        else None,
        "finalization_state": "running",
    }
    # Publish a non-sealed progress record before the expensive offline
    # coverage/mismatch passes.  If the supervisor deadline kills a collector,
    # the run still has a truthful framework-finalize.json to audit.
    final_status = result["status"]
    result["status"] = "partial"
    write_json_replace(out / "framework-finalize.json", result)
    try:
        from v2.offline_coverage import merge_run_coverage

        v2_coverage = merge_run_coverage(out)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        v2_coverage = {
            "status": "gap",
            "reason": f"v2-coverage-finalize:{type(error).__name__}",
        }
    try:
        from v2.offline_mismatch import analyze_run

        v2_mismatch = analyze_run(out)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        v2_mismatch = {
            "schema_version": "rq1-v2-mismatch-summary-v1",
            "status": "error",
            "reason": f"v2-mismatch-finalize:{type(error).__name__}",
        }
    result["v2_coverage"] = v2_coverage
    result["v2_mismatch"] = v2_mismatch
    result["status"] = final_status
    result["finalization_state"] = "complete"
    if isinstance(run_result, dict):
        run_result["v2_coverage"] = v2_coverage
        run_result["v2_mismatch"] = v2_mismatch
        write_json_replace(out / "run-result.json", run_result)
    write_json_replace(out / "framework-finalize.json", result)
    return result


def _formal_differential_status(event: Mapping[str, Any]) -> object:
    differential = event.get("differential")
    status = _canonical_differential_status(
        differential.get("status") if isinstance(differential, dict) else None
    )
    if not isinstance(differential, dict) or not isinstance(status, str) \
            or status not in (DIFFERENTIAL_MATCH, DIFFERENTIAL_MISMATCH):
        return status
    comparison = differential.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    differences = comparison.get("differences")
    valid_differences = isinstance(differences, (Mapping, list, tuple))
    if (
        event.get("target_attempted") is not True
        or _event_is_pending(event)
        or event.get("reference_join_status") != "joined"
        or event.get("formal_eligible") is not True
        or any(event.get(name) for name in (
            "comparison_qualification_gaps", "qualification_gaps", "missing_fields",
            "validation_gap", "contract_error", "target_execution_contract_error",
        ))
        or differential.get("qualification_gaps")
        or differential.get("missing_fields")
        or differential.get("validation_gap")
        or differential.get("contract_error")
        or comparison.get("qualification_gaps")
        or comparison.get("missing_fields")
        or comparison.get("validation_gap")
        or comparison.get("contract_error")
        or comparison.get("target_execution_contract_error")
        or not valid_differences
        or status == DIFFERENTIAL_MISMATCH and (
            comparison.get("status") != "non-equivalent"
            or not differences
            or not isinstance(differential.get("mismatch_id"), str)
            or not differential["mismatch_id"].strip()
        )
        or status == DIFFERENTIAL_MATCH and (
            comparison.get("status") != "equivalent"
            or bool(differences)
            or differential.get("mismatch_id") is not None
        )
    ):
        return DIFFERENTIAL_UNVERIFIED
    return status


def _differential_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    discrepancies = []
    for event in events:
        if not isinstance(event, dict):
            continue
        # This synthetic event records an empty generation result for an
        # applicable target. It is a generation/queue gap, not a target
        # observation, so counting it as a missing differential status would
        # corrupt mismatch and match denominators in run-result.json.
        if event.get("reason_code") == "generation-produced-no-candidate":
            continue
        # A differential belongs to an attempted Target observation.  Keep
        # stale fields from interrupted/legacy unattempted rows out of the
        # result counters even when the ledger itself is retained verbatim.
        if event.get("target_attempted") is not True:
            continue
        if _event_is_pending(event):
            continue
        status = _formal_differential_status(event)
        status_counts[status if isinstance(status, str) and status else "missing"] += 1
        if status == DIFFERENTIAL_MISMATCH:
            discrepancies.append(event)
    return {
        "differential_status_counts": dict(sorted(status_counts.items())),
        "mismatch_candidate_count": len(discrepancies),
        "discrepancy_event_count": len(discrepancies),
        "formal_discrepancy_event_count": sum(
            event.get("formal_eligible") is True for event in discrepancies
        ),
    }


def _execution_window_within_budget(result: dict[str, Any], duration: int) -> bool:
    """Validate active time, using the run's declared segment policy."""
    value = result.get("execution_window_seconds")
    # Historical runs stored only wall_clock_seconds, which included all
    # finalization work and therefore remains the only available measurement.
    if value is None:
        value = result.get("wall_clock_seconds")
    if result.get("segment_budget_mode") == "case-boundary-pause-resume":
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0
            and result.get("execution_budget_seconds") is None
            and type(result.get("segment_count")) is int
            and result["segment_count"] > 0
        )
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and 0 < value <= duration
    )


def _board_reference_named_gap(event: dict[str, Any]) -> bool:
    """Recognize candidate-local board gaps and the shared QEMU reference cell."""
    fallback = event.get("reference_fallback")
    shared_qemu_cell = (
        event.get("reference_join_status") == "shared-reference-target"
        and event.get("target") == "T-QEMU"
        and event.get("reference_id") == "R-QEMU-FALLBACK"
        and isinstance(fallback, dict)
        and fallback.get("status") == "used"
        and fallback.get("source_target") == "T-QEMU"
    )
    return (
        shared_qemu_cell
        or (
            event.get("reference_join_status") == "reference-untrusted"
            and event.get("reference_status") in {"gap", "failed", "timeout"}
        )
    )


def _normalize_coverage_profile_refs(out: Path, coverage: dict[str, Any]) -> bool:
    """Rewrite legacy container paths to references usable from the run root."""
    rows = coverage.get("source_targets")
    if not isinstance(rows, list):
        return False
    changed = False
    run_root = out.resolve()

    def normalize(value: object) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            relative = path.resolve().relative_to(run_root)
        except (OSError, ValueError):
            relative = None
        if relative is not None and (run_root / relative).is_file():
            return str(relative)
        # Older container finalizers wrote /opt/runs/<run-id>/... into the
        # host-visible summary. Resolve that stable suffix when the host copy
        # is available, without guessing arbitrary absolute paths.
        parts = path.parts
        for index, part in enumerate(parts[:-1]):
            if part != out.name:
                continue
            candidate_relative = Path(*parts[index + 1:])
            if (run_root / candidate_relative).is_file():
                return str(candidate_relative)
        return value

    for row in rows:
        if not isinstance(row, dict):
            continue
        source = row.get("simulator_source_coverage")
        if not isinstance(source, dict):
            continue
        for name in ("profile_path", "profile_identity_path"):
            original = source.get(name)
            normalized = normalize(original)
            if normalized is not None and normalized != original:
                source[name] = normalized
                changed = True
    return changed


def finalize_campaign(
    out: Path, config_path: Path = DEFAULT_CONFIG, *,
    require_final_identity: bool = True,
    recovery_partial: bool = False,
) -> dict[str, Any]:
    out = Path(out).resolve()
    config, config_path = _load_frozen_config(out, config_path)
    config_gaps = validate_config(config)
    if config_gaps:
        raise ValueError("invalid configuration: " + ",".join(config_gaps))
    try:
        manifest = load_json(out / "execution-manifest.json")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("comparison finalize manifest is invalid") from exc
    if manifest.get("action") not in {"run", "execute-existing-queues"}:
        raise ValueError("comparison finalize requires a target-execution run")
    result_path = out / "run-result.json"
    if result_path.is_file():
        try:
            result = load_json(result_path)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("comparison finalize run result is invalid") from exc
    else:
        partial_result_path = out / "execution-result.partial.json"
        if partial_result_path.is_file():
            try:
                result = load_json(partial_result_path)
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError("comparison finalize partial result is invalid") from exc
        else:
            result = {}
    generation_source = result.get("generation_source")
    generation_source_method = (
        generation_source.get("method_filter")
        if isinstance(generation_source, dict) else None
    )
    selected_method = (
        result.get("method_filter")
        or generation_source_method
        or manifest.get("method_filter")
    )
    if not isinstance(selected_method, str) or not selected_method:
        raise ValueError("comparison finalize requires one --method")
    recovery_reason = None
    if recovery_partial:
        # The host wrapper writes a wall-clock marker for its own watchdog and
        # an external-signal marker when a container is stopped out of band.
        # Pending entries alone do not identify why a run stopped.
        recovery_reason = _recovery_partial_reason(out, result)
    if not result_path.is_file():
        # A killed runner may leave only the streaming partial checkpoint.  Keep
        # a stable result entry for the recovery seal; it remains an explicit
        # partial and never counts as a completed execution.
        result = {
            **result,
            "schema_version": "rq1-run-result-v2",
            "run_id": out.name,
            "result_file": "run-result.json",
            "method_filter": selected_method,
            "status": "partial",
            "execution_complete": False,
            "formal_ready": False,
            "partial_recovery": "execution-result.partial.json"
            if (out / "execution-result.partial.json").is_file()
            else "ledger/events.partial.jsonl",
        }
        write_json_replace(result_path, result)
    if recovery_reason:
        producer_status = result.get("producer_status")
        if isinstance(producer_status, dict) and producer_status.get("status") == "running":
            result["producer_status"] = {
                **producer_status, "status": "interrupted", "reason": recovery_reason,
            }
        result["partial_reason"] = recovery_reason
        for partial_path in (
            out / "generation-result.partial.json",
            out / "generation" / "candidate-queues.partial.json",
            out / "execution-result.partial.json",
        ):
            try:
                partial = load_json(partial_path)
            except (OSError, TypeError, ValueError):
                continue
            changed = False
            producer = partial.get("producer")
            if isinstance(producer, dict) and producer.get("status") == "running":
                partial["producer"] = {
                    **producer, "status": "interrupted", "reason": recovery_reason,
                }
                changed = True
            producer = partial.get("producer_status")
            if isinstance(producer, dict) and producer.get("status") == "running":
                partial["producer_status"] = {
                    **producer, "status": "interrupted", "reason": recovery_reason,
                }
                changed = True
            if "partial_reason" in partial and partial.get("partial_reason") != recovery_reason:
                partial["partial_reason"] = recovery_reason
                changed = True
            if changed:
                write_json_replace(partial_path, partial)
    final_events_path = out / "ledger" / "events.jsonl"
    events_path = final_events_path
    if not events_path.is_file():
        events_path = out / "ledger" / "events.partial.jsonl"
    final_event_ledger_complete = final_events_path.is_file()
    events = []
    ledger_parse_errors = []
    if events_path.is_file():
        try:
            ledger_lines = events_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            ledger_lines = []
            ledger_parse_errors.append(0)
        for line_number, line in enumerate(ledger_lines, start=1):
            if line.strip():
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    ledger_parse_errors.append(line_number)
                    continue
                if not isinstance(event, dict):
                    ledger_parse_errors.append(line_number)
                    continue
                events.append(event)
    ledger_parse_error_count = len(ledger_parse_errors)
    ledger_seal_path = out / "ledger" / "seal.json"
    ledger_seal_parse_error = False
    if ledger_seal_path.is_file():
        try:
            ledger_seal = load_json(ledger_seal_path)
        except (OSError, TypeError, ValueError):
            ledger_seal = {}
            ledger_seal_parse_error = True
    else:
        ledger_seal = {}
    coverage_registry_path = out / "coverage-registry.json"
    rv_metrics_enabled = rv_instruction_metrics_enabled(config)
    coverage_registry_parse_error = False
    try:
        coverage_registry = load_json(coverage_registry_path)
    except (OSError, TypeError, ValueError):
        coverage_registry = {}
        coverage_registry_parse_error = True
    coverage_registry_sha256 = (
        digest(coverage_registry)
        if isinstance(coverage_registry, dict) and coverage_registry else None
    )
    coverage_registry_sealed = bool(
        isinstance(coverage_registry, dict)
        and coverage_registry
        and all(
            isinstance(value, dict) and value.get("sealed") is True
            for value in coverage_registry.values()
        )
    )
    registry_event_identity_ok = (
        True if not rv_metrics_enabled else all(
            isinstance(event, dict)
            and event.get("coverage_registry_sha256") == coverage_registry_sha256
            and event.get("coverage_registry_sealed") is True
            for event in events
        )
    )
    coverage_registry_complete = (
        True if not rv_metrics_enabled else bool(
            coverage_registry_path.is_file()
            and not coverage_registry_parse_error
            and coverage_registry_sha256
            and coverage_registry_sealed
            and isinstance(ledger_seal.get("registry_sha256"), str)
            and ledger_seal.get("registry_sha256") == coverage_registry_sha256
            and result.get("coverage_registry_sha256") == coverage_registry_sha256
            and registry_event_identity_ok
        )
    )
    ledger_seal_basis = dict(ledger_seal)
    declared_ledger_seal_sha = ledger_seal_basis.pop("seal_sha256", None)
    ledger_seal_ok = bool(
        ledger_seal_path.is_file()
        and events_path.is_file()
        and ledger_seal.get("schema_version") == "rq1-ledger-seal-v1"
        and ledger_seal.get("run_id") == out.name
        and ledger_seal.get("method_filter") == selected_method
        and isinstance(ledger_seal.get("event_ledger_sha256"), str)
        and coverage_registry_complete
        and final_event_ledger_complete
        and events_path == final_events_path
        and ledger_seal.get("event_ledger_sha256") == sha256(events_path)
        and ledger_parse_error_count == 0
        and not ledger_seal_parse_error
        and isinstance(declared_ledger_seal_sha, str)
        and declared_ledger_seal_sha == digest(ledger_seal_basis)
    )
    methods = [
        method for method in config.get("methods", [])
        if isinstance(method, dict)
        and (selected_method is None or method.get("id") == selected_method)
    ]
    generation_result_path = out / "generation-result.json"
    generation_queue_path = out / "generation" / "candidate-queues.json"
    try:
        generation_result_artifact = load_json(generation_result_path)
    except (OSError, TypeError, ValueError):
        generation_result_artifact = {}
    try:
        generation_queue_artifact = load_json(generation_queue_path)
    except (OSError, TypeError, ValueError):
        generation_queue_artifact = {}
    expected_target_ids = {
        str(target.get("id")) for target in config.get("targets", [])
        if isinstance(target, dict) and target.get("id")
    }
    generation_result_artifact_ok = (
        generation_result_artifact.get("schema_version") == "rq1-generation-result-v2"
        and generation_result_artifact.get("run_id") == out.name
        and generation_result_artifact.get("method_filter") == selected_method
        and generation_result_artifact.get("phase") in {
            "streaming-generation+execution", "generation+execution",
        }
        and isinstance(generation_result_artifact.get("generation"), dict)
        and generation_result_artifact["generation"].get("queue_manifest")
        == "generation/candidate-queues.json"
        and type(generation_result_artifact.get("generation_sealed")) is bool
    )
    queue_manifest_methods = generation_queue_artifact.get("methods")
    queue_manifest_queues = generation_queue_artifact.get("queues")
    generation_queue_artifact_ok = (
        generation_queue_artifact.get("schema_version") == "rq1-generation-queue-v1"
        and isinstance(generation_queue_artifact.get("generation_run_id"), str)
        and bool(generation_queue_artifact.get("generation_run_id"))
        and generation_queue_artifact.get("method_filter") == selected_method
        and isinstance(queue_manifest_methods, dict)
        and set(queue_manifest_methods) == {selected_method}
        and isinstance(queue_manifest_queues, dict)
        and set(queue_manifest_queues) == expected_target_ids
        and all(isinstance(value, dict) for value in queue_manifest_queues.values())
    )
    expected_queue_paths = {
        out / "queues" / f"{target_id}.json" for target_id in expected_target_ids
    }
    queue_files_complete = bool(expected_queue_paths) and all(
        path.is_file() for path in expected_queue_paths
    )
    queue_artifact_shape_ok = True
    queue_parse_error_count = 0
    loaded_queue_targets: set[str] = set()
    queue_entries: dict[str, list[Any]] = {}
    for path in sorted((out / "queues").glob("*.json")):
        try:
            queue = load_json(path)
        except (OSError, TypeError, ValueError):
            queue_parse_error_count += 1
            queue_artifact_shape_ok = False
            continue
        target = queue.get("target")
        raw_entries = queue.get("entries", [])
        queue_artifact_shape_ok = queue_artifact_shape_ok and (
            queue.get("schema_version") == "rq1-target-queue-v1"
            and path.stem == str(target)
            and str(target) in expected_target_ids
            and str(target) not in loaded_queue_targets
            and queue.get("method_filter") == selected_method
            and isinstance(raw_entries, list)
        )
        if target is None or str(target) in loaded_queue_targets:
            continue
        loaded_queue_targets.add(str(target))
        queue_entries[str(target)] = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    queue_artifact_shape_ok = queue_artifact_shape_ok and loaded_queue_targets == expected_target_ids
    queue_source = "final"
    queue_partial_recovered = False
    queue_recovery_reason = None
    recovered_pending_count = 0
    snapshot_queue_counts: dict[str, int] = {}
    if not queue_files_complete or not queue_artifact_shape_ok:
        partial_entries, partial_reason = _load_partial_queue_entries(
            out / "generation" / "candidate-queues.partial.json",
            run_id=out.name, method_filter=selected_method,
            expected_target_ids=expected_target_ids,
        )
        if partial_reason is None:
            queue_entries = partial_entries
            queue_source = "partial-snapshot"
            queue_partial_recovered = True
            # A signal may arrive before the streaming workers can append the
            # normal right-censor events.  Materialize those missing entries
            # as pending in the partial ledger, so the recovered metrics and
            # their source bytes have the same queue/event key set.
            if not final_event_ledger_complete and events_path == out / "ledger" / "events.partial.jsonl" \
                    and ledger_parse_error_count == 0:
                from comparison_events import _make_event

                method_spec = next(
                    (method for method in methods if method.get("id") == selected_method),
                    {},
                )
                target_ids_by_name = {
                    str(target.get("id")): target
                    for target in config.get("targets", [])
                    if isinstance(target, dict) and target.get("id")
                }
                existing_event_keys = set()
                def recovery_key_part(value: object) -> str | None:
                    if value is None:
                        return None
                    if isinstance(value, (str, int, float, bool)):
                        return str(value)
                    return None
                for event in events:
                    index = _nonnegative_index(event.get("stream_index"))
                    if index is None:
                        continue
                    existing_event_keys.add((
                        recovery_key_part(event.get("target")),
                        recovery_key_part(event.get("method")),
                        recovery_key_part(event.get("route")),
                        recovery_key_part(event.get("lane")), index,
                    ))
                for target_id, entries in queue_entries.items():
                    target = target_ids_by_name.get(str(target_id))
                    if not target:
                        queue_recovery_reason = "partial-queue-target-missing"
                        continue
                    for entry in entries:
                        candidate = entry.get("candidate", {})
                        index = _nonnegative_index(
                            candidate.get("index") if isinstance(candidate, dict) else None
                        )
                        if index is None:
                            queue_recovery_reason = "partial-queue-entry-index"
                            continue
                        key = (
                            recovery_key_part(target_id),
                            recovery_key_part(entry.get("method")),
                            recovery_key_part(entry.get("route")),
                            recovery_key_part(entry.get("lane")), index,
                        )
                        if key in existing_event_keys:
                            continue
                        events.append(_make_event(
                            method=method_spec, target_id=str(target_id),
                            entry={**entry, "campaign_id": out.name}, result=None,
                            registry_sha=coverage_registry_sha256 or "",
                            registry_sealed=coverage_registry_sealed,
                            config=config,
                            reason_override=recovery_reason or "run-wall-clock-exhausted",
                            target_attempted=False,
                        ))
                        existing_event_keys.add(key)
                        recovered_pending_count += 1
                if recovered_pending_count:
                    write_event_ledger(events_path, events)
        else:
            queue_source = "unavailable"
            queue_recovery_reason = partial_reason
            snapshot_queue_counts, counts_reason = _partial_queue_counts(
                out / "generation" / "candidate-queues.partial.json",
                run_id=out.name, method_filter=selected_method,
                expected_target_ids=expected_target_ids,
            )
            if counts_reason is None:
                # Counts are useful generation evidence, but without entry
                # identity they cannot close the queue/event join.
                queue_source = "partial-counts"
    queue_entries_artifacts_ok = _queue_entries_artifacts_ok(
        out, config, queue_entries, selected_method,
    )
    queue_candidate_identity_ok = _queue_candidate_identity_ok(
        config, queue_entries, selected_method,
    )
    queue_artifact_shape_ok = queue_artifact_shape_ok and queue_entries_artifacts_ok
    generation_artifacts_complete = (
        generation_result_artifact_ok
        and generation_queue_artifact_ok
        and queue_files_complete
        and queue_artifact_shape_ok
        and queue_parse_error_count == 0
    )
    accounting = _queue_accounting(queue_entries, events)
    execution_events = accounting["execution_events"]
    valid_event_records = [event for event in events if isinstance(event, dict)]
    def key_text(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return None

    actual = {
        (key_text(event.get("target")), key_text(event.get("method")),
         key_text(event.get("route")), key_text(event.get("lane")))
        for event in execution_events
    }
    queue_keys = accounting["_queue_keys"]
    dispatched_events = accounting["dispatched_events"]
    queue_drained = accounting["queue_drained"]
    queue_accounting_ok = accounting["queue_accounting_complete"]
    terminal_outcomes = {
        "normal", "complete_observation", "trap", "expected-trap",
        "crash", "nonzero-exit", "timeout",
    }
    attempted_events = accounting["attempted_events"]
    parent_stages = [
        event.get("parent_execution") for event in execution_events
        if isinstance(event.get("parent_execution"), dict)
        and event["parent_execution"].get("invoked") is True
    ]
    parent_executions = [
        event.get("parent_execution") for event in execution_events
        if isinstance(event.get("parent_execution"), dict)
        and event["parent_execution"].get("attempted") is True
    ]
    parent_invocation_count = len(parent_stages)
    parent_attempted_count = len(parent_executions)
    parent_observed_count = sum(_event_observation_recorded(item) for item in parent_executions)
    parent_error_count = sum(
        item.get("status") in {"gap", "error"}
        or (item.get("status") == "failed" and item.get("outcome") not in terminal_outcomes)
        for item in parent_stages
    )
    parent_terminal_failure_count = sum(
        item.get("status") == "failed" and item.get("outcome") in terminal_outcomes
        for item in parent_stages
    )
    parent_not_started_count = parent_invocation_count - parent_attempted_count
    parent_outcome_counts = dict(Counter(
        key_text(item.get("outcome")) or "unknown" for item in parent_executions
    ))
    terminal_events = [event for event in execution_events if event.get("outcome") in terminal_outcomes]
    method_counts = {
        method["id"]: sum(
            event.get("method") == method["id"]
            and event.get("status") == "passed"
            and event.get("outcome") in terminal_outcomes
            for event in valid_event_records
        )
        for method in methods
    }
    method_attempted_counts = dict(Counter(
        key_text(event.get("method")) or "unknown" for event in attempted_events
    ))
    method_observed_counts = dict(Counter(
        key_text(event.get("method")) or "unknown" for event in attempted_events
        if _event_observation_recorded(event)
    ))
    formal_counts = dict(Counter(
        key_text(event.get("method")) or "unknown"
        for event in terminal_events if event.get("formal_eligible") is True
    ))
    method_outcome_counts = {
        method: dict(Counter(
            key_text(event.get("outcome")) or "unknown"
            for event in terminal_events if key_text(event.get("method")) == method
        ))
        for method in {key_text(event.get("method")) or "unknown" for event in terminal_events}
    }
    target_outcome_counts = {
        target: dict(Counter(
            key_text(event.get("outcome")) or "unknown"
            for event in terminal_events if key_text(event.get("target")) == target
        ))
        for target in {key_text(event.get("target")) or "unknown" for event in terminal_events}
    }
    def event_label(event: dict[str, Any], name: str) -> str:
        value = event.get(name)
        return str(value) if value not in (None, "") else "unknown"

    event_status_counts = dict(Counter(
        event_label(event, "status") for event in execution_events
    ))
    event_outcome_counts = dict(Counter(
        event_label(event, "outcome") for event in execution_events
    ))
    attempted_outcome_counts = dict(Counter(
        event_label(event, "outcome") for event in attempted_events
    ))
    method_status_counts = {
        method: dict(Counter(event_label(event, "status") for event in execution_events
                            if key_text(event.get("method")) == method))
        for method in {key_text(event.get("method")) or "unknown" for event in execution_events}
    }
    method_all_outcome_counts = {
        method: dict(Counter(event_label(event, "outcome") for event in execution_events
                             if key_text(event.get("method")) == method))
        for method in {key_text(event.get("method")) or "unknown" for event in execution_events}
    }
    target_status_counts = {
        target: dict(Counter(event_label(event, "status") for event in execution_events
                             if key_text(event.get("target")) == target))
        for target in {key_text(event.get("target")) or "unknown" for event in execution_events}
    }
    target_all_outcome_counts = {
        target: dict(Counter(event_label(event, "outcome") for event in execution_events
                             if key_text(event.get("target")) == target))
        for target in {key_text(event.get("target")) or "unknown" for event in execution_events}
    }
    event_metric_counts = {
        "event_status_counts": event_status_counts,
        "event_outcome_counts": event_outcome_counts,
        "attempted_outcome_counts": attempted_outcome_counts,
        "unsupported_isa_count": sum(
            event.get("outcome") == "case-skipped"
            and event.get("reason_code") == "unsupported-isa"
            for event in execution_events
        ),
        "method_status_counts": method_status_counts,
        "method_all_outcome_counts": method_all_outcome_counts,
        "target_status_counts": target_status_counts,
        "target_all_outcome_counts": target_all_outcome_counts,
    }
    timeout_count = sum(event.get("outcome") == "timeout" for event in attempted_events)
    observation_recorded_count = sum(
        _event_observation_recorded(event) for event in attempted_events
    )
    pending_keys = accounting["_pending_keys"]
    queued_by_target = Counter(str(key[0]) for key in queue_keys)
    pending_by_target = Counter(str(key[0]) for key in pending_keys)
    if not queue_accounting_ok and snapshot_queue_counts:
        queued_by_target = Counter(snapshot_queue_counts)
        completed_event_keys = {
            (key_text(event.get("target")), key_text(event.get("method")),
             key_text(event.get("route")), key_text(event.get("lane")),
             int(event.get("stream_index")))
            for event in dispatched_events
            if isinstance(event.get("stream_index"), int)
            and not isinstance(event.get("stream_index"), bool)
        }
        pending_by_target = Counter({
            target_id: max(
                count - sum(key[0] == target_id for key in completed_event_keys),
                0,
            )
            for target_id, count in snapshot_queue_counts.items()
        })
    reported_queued_count = (
        sum(snapshot_queue_counts.values())
        if not queue_accounting_ok and snapshot_queue_counts
        else accounting["queued_count"]
    )
    reported_pending_count = (
        sum(pending_by_target.values())
        if not queue_accounting_ok and snapshot_queue_counts
        else accounting["pending_count"]
    )
    target_rates_by_target = {}
    for target_id in sorted(queued_by_target):
        target_events = [event for event in attempted_events
                         if str(event.get("target")) == target_id]
        target_rates_by_target[target_id] = _target_rates(
            queued=queued_by_target[target_id],
            attempted=len(target_events),
            observation_recorded=sum(
                _event_observation_recorded(item) for item in target_events
            ),
            pending=pending_by_target[target_id],
            timeout=sum(item.get("outcome") == "timeout" for item in target_events),
        )
    target_rates = _target_rates(
        queued=reported_queued_count, attempted=len(attempted_events),
        observation_recorded=observation_recorded_count,
        pending=reported_pending_count, timeout=timeout_count,
    )
    event_ids = [event.get("event_id") for event in execution_events]
    event_ids_ok = (
        all(isinstance(event_id, str) and event_id for event_id in event_ids)
        and len(event_ids) == len(set(event_ids))
    )
    event_terminals_ok = bool(execution_events) and all(
        _execution_event_closed(event) or _expected_timebox_event(event)
        for event in execution_events
    )
    queued_pairs = {(target, method, route, lane)
                    for target, method, route, lane, _ in queue_keys}
    method_target_coverage_ok = actual == queued_pairs and event_ids_ok
    generation_queue_pairs = {
        (method, route, target, lane)
        for target, method, route, lane in queued_pairs
    }
    generation_payload = result.get("generation")
    generation_payload = generation_payload if isinstance(generation_payload, dict) else {}
    result_methods_payload = result.get("methods")
    result_methods_payload = (
        result_methods_payload if isinstance(result_methods_payload, (list, tuple)) else ()
    )
    generation_method_coverage_ok = _generation_method_coverage_ok(
        config,
        {"status": generation_payload.get("status", result.get("status")),
         "generation_sealed": result.get("generation_sealed"),
         "method_filter": result.get("method_filter") or generation_source_method,
         "methods": result_methods_payload},
        generation_queue_pairs,
        dict(Counter(key[1] for key in queue_keys)),
        require_seal=True,
    )
    coverage_path = out / "coverage" / "summary.json"
    coverage_case_results_path = out / "coverage" / "case-results.jsonl.gz"
    target_execution_action = manifest.get("action") in {"run", "execute-existing-queues"}
    result_coverage_enabled = result.get("coverage_enabled")
    coverage_enabled = (
        result_coverage_enabled
        if isinstance(result_coverage_enabled, bool)
        else manifest.get("coverage_enabled") is True
    )
    coverage_policy_ok = not target_execution_action or (
        isinstance(manifest.get("coverage_enabled"), bool)
        and isinstance(result_coverage_enabled, bool)
    )
    # Disabled coverage is an explicit mode, not a completed coverage
    # artifact.  It must not block execution completeness, but it cannot make
    # the run formal-ready or silently enter a coverage table.
    coverage_artifact_complete = False
    coverage_case_results_complete = False
    coverage_method_boundary_ok = True
    if coverage_enabled:
        try:
            coverage = load_json(coverage_path)
            if _normalize_coverage_profile_refs(out, coverage):
                write_json_replace(coverage_path, coverage)
            coverage_groups = (coverage.get("execution_scope") or {}).get("groups")
            coverage_method_boundary_ok = (
                coverage.get("method_filter") == selected_method
                and isinstance(coverage_groups, list)
                and all(
                    isinstance(group, dict) and group.get("method") == selected_method
                    for group in coverage_groups
                )
            )
            experiment_unions = coverage.get("experiment_unions")
            coverage_method_union_ok = (
                isinstance(experiment_unions, list)
                and len(experiment_unions) == 1
                and isinstance(experiment_unions[0], dict)
                and experiment_unions[0].get("method") == selected_method
                and experiment_unions[0].get("scope")
                == "whole-method-run-all-cases-and-targets"
            )
            coverage_artifact_complete = (
                result.get("coverage_artifact_complete") is True
                and coverage.get("schema_version") == BATCH_SCHEMA
                and isinstance(coverage.get("targets"), list)
                and coverage_method_boundary_ok
                and coverage_method_union_ok
            )
            case_results = coverage.get("case_results")
            coverage_case_results_complete = (
                coverage_case_results_path.is_file()
                and isinstance(case_results, dict)
                and case_results.get("path") == str(coverage_case_results_path.relative_to(out))
                and case_results.get("sha256") == sha256(coverage_case_results_path)
                and case_results.get("replay_match") is True
            )
            coverage_artifact_complete = coverage_artifact_complete and coverage_case_results_complete
        except (OSError, TypeError, ValueError):
            coverage_artifact_complete = False
            coverage_method_boundary_ok = False
    result_method_rows = {
        str(item.get("method")): item for item in result_methods_payload
        if isinstance(item, dict) and item.get("method")
    }
    expected_method_ids = {str(method.get("id")) for method in methods
                           if isinstance(method, dict)}
    result_methods = set(result_method_rows)
    methods_ok = result_methods == expected_method_ids
    queue_candidate_indices_ok = _queue_candidate_indices_ok(
        config, result_method_rows, queue_entries, selected_method,
    )
    queue_candidate_indices_ok = queue_candidate_indices_ok and queue_candidate_identity_ok
    # ``cases`` is an upper bound for a fixed-time campaign.  The shared
    # generation coverage gate already requires every method/target pair and
    # a sealed queue; it deliberately accepts method-level ``partial`` rows.
    generation_shape_ok = (
        generation_method_coverage_ok and queue_candidate_indices_ok
        and queue_candidate_identity_ok and methods_ok
    )
    result_method = result.get("method_filter")
    method_boundary_ok = (
        selected_method == manifest.get("method_filter")
        and (result_method is None or result_method == selected_method)
        and (generation_source_method is None or generation_source_method == selected_method)
    )
    # finalize 可能处理执行阶段先落下的 partial 快照；最终状态由下面的
    # 结构、身份、覆盖率和事件闭合条件重新计算，不把中间状态当成硬失败。
    result_status_ok = result.get("status") in {"completed", "partial"}
    wrapper_path = out / "wrapper-result.json"
    if wrapper_path.is_file():
        try:
            wrapper = load_json(wrapper_path)
        except (OSError, TypeError, ValueError):
            wrapper = {}
    else:
        wrapper = {}
    # standalone `finalize` 要求 wrapper 正常完成；源码 provenance 仅记录，
    # 不因运行期间源码变化而撤销已完成实验。
    wrapper_ok = (
        not require_final_identity and not wrapper_path.is_file()
    ) or (
        bool(wrapper) and wrapper.get("status") == "completed"
        and wrapper.get("method_filter") == selected_method
        and wrapper.get("method_count") == 1
    )
    manifest_check, _ = _run_manifest_gate(out, config)
    identity_ok = (
        manifest_check.get("status") == "verified"
        and
        isinstance(manifest.get("source_commit"), str)
        and bool(manifest.get("source_commit"))
        and manifest.get("dependency_identity_ok") is True
        and isinstance(manifest.get("image_id"), str)
        and manifest.get("config_sha256") == sha256(config_path)
        and wrapper_ok
    )
    def stamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    host_start = stamp(manifest.get("started_at_utc"))
    result_start = stamp(result.get("started_at_utc"))
    result_end = stamp(result.get("ended_at_utc"))
    wrapper_end = stamp(wrapper.get("finished_at_utc"))
    manifest_duration = manifest.get("duration_seconds")
    manifest_duration = (
        manifest_duration
        if type(manifest_duration) is int and manifest_duration > 0
        else 0
    )
    clock_ok = (
        host_start is not None and result_start is not None
        and result_end is not None
        and host_start <= result_start <= result_end
        and (wrapper_end is None or (result_end - wrapper_end).total_seconds() <= 1)
        and _execution_window_within_budget(
            result, manifest_duration,
        )
    )
    differential_metrics = _differential_metrics(valid_event_records)
    # reference 未覆盖的 stratum 是具名缺口，不把整轮判定成未完成。
    board_strata = ("linux-user",) if str(
        (config.get("board_differential") or {}).get("coverage", "")
    ).startswith("linux-user") else ("bare-metal", "linux-user")
    board_events = [
        event for event in attempted_events
        if isinstance(event, dict)
        and event.get("stratum") in board_strata
        and not _event_is_expected_timebox(event)
    ]
    def formal_differential_event_ok(event: dict[str, Any]) -> bool:
        return _formal_differential_status(event) in (
            DIFFERENTIAL_MATCH, DIFFERENTIAL_MISMATCH,
        )
    # 板卡跑过这个 case 但没能建立基线是输入自身的具名缺口，不阻断整轮；
    # 完全没有 board 结果的 reference-missing 仍然阻断。
    board_named_gap_events = [
        event for event in board_events if _board_reference_named_gap(event)
    ]
    board_events = [
        event for event in board_events if not _board_reference_named_gap(event)
    ]
    board_requested = manifest.get("board_requested") is True
    if not board_requested:
        # ``--without-board`` is an explicit experiment face.  There is no
        # board reference stream to join in this mode, so board differential
        # evidence is not applicable rather than missing.  Keeping these
        # gates true lets the target-only comparison contract be judged by
        # its queue, observation, and coverage evidence.
        differential_evidence_complete = True
        reference_execution_complete = True
    else:
        differential_evidence_complete = bool(board_events) and all(
            formal_differential_event_ok(event)
            for event in board_events
        )
        reference_execution_complete = bool(board_events) and all(
            formal_differential_event_ok(event)
            and event.get("formal_eligible") is True
            for event in board_events
        )
    # Execution completeness is the observation/queue contract.  Coverage is
    # a separate evidence side-channel and only affects formal readiness; a
    # simulator run must remain auditable when coverage is disabled or partial.
    execution_complete = all((
        method_target_coverage_ok, generation_method_coverage_ok,
        generation_artifacts_complete,
        coverage_registry_complete,
        queue_accounting_ok,
        ledger_seal_ok,
        accounting["unattempted_error_count"] == 0,
        event_terminals_ok, methods_ok, generation_shape_ok, result_status_ok, identity_ok,
        method_boundary_ok, clock_ok,
    ))
    formal_ready = all((
        execution_complete, differential_evidence_complete,
        reference_execution_complete, coverage_policy_ok, coverage_artifact_complete,
    ))
    if recovery_partial:
        # 宿主只能保存中断时的证据；即使队列恰好排空，也不能把缺少
        # 容器内正式收尾的恢复结果提升为 completed/sealed。
        execution_complete = False
        formal_ready = False
    if result_path.is_file():
        result.update(
            status="completed" if execution_complete else "partial",
            execution_complete=execution_complete,
            method_filter=selected_method,
            event_count=len(events),
            ledger_event_count=len(events),
            execution_event_count=accounting["event_count"],
            event_ledger_sha256=sha256(events_path),
            ledger_seal_sha256=sha256(ledger_seal_path),
            ledger_parse_error_count=ledger_parse_error_count,
            ledger_seal_parse_error=ledger_seal_parse_error,
            final_event_ledger_complete=final_event_ledger_complete,
            partial_reason=recovery_reason or result.get("partial_reason"),
            coverage_registry_sha256=coverage_registry_sha256,
            coverage_registry_file_sha256=(
                sha256(coverage_registry_path) if coverage_registry_path.is_file() else None
            ),
            coverage_registry_complete=coverage_registry_complete,
            formal_ready=formal_ready,
            method_target_coverage_complete=method_target_coverage_ok,
            generation_method_coverage_complete=generation_method_coverage_ok,
            generation_artifacts_complete=generation_artifacts_complete,
            queue_artifact_shape_complete=queue_artifact_shape_ok,
            queue_entries_artifacts_complete=queue_entries_artifacts_ok,
            queue_candidate_identity_complete=queue_candidate_identity_ok,
            queue_parse_error_count=queue_parse_error_count,
            coverage_artifact_complete=coverage_artifact_complete,
            coverage_method_boundary_complete=coverage_method_boundary_ok,
            coverage_policy_ok=coverage_policy_ok,
            method_boundary_complete=method_boundary_ok,
            identity_complete=identity_ok,
            queued_count=reported_queued_count,
            dispatched_count=accounting["dispatched_count"],
            processed_count=accounting["processed_count"],
            unattempted_error_count=accounting["unattempted_error_count"],
            pending_count=reported_pending_count,
            right_censored_count=accounting["right_censored_count"],
            queue_accounting_complete=queue_accounting_ok,
            queue_count_exact=queue_accounting_ok,
            queue_count_source=("queue-entries" if queue_accounting_ok else
                                "partial-snapshot-counts" if snapshot_queue_counts else
                                "unavailable"),
            ledger_seal_complete=ledger_seal_ok,
            queue_drained=queue_drained,
            queue_source=queue_source,
            queue_partial_recovered=queue_partial_recovered,
            queue_recovery_reason=queue_recovery_reason,
            recovered_pending_count=recovered_pending_count,
            target_attempted_count=len(attempted_events),
            target_observed_count=sum(
                _event_observation_recorded(event) for event in attempted_events
            ),
            target_terminal_observed_count=sum(
                _event_terminal_observed(event) for event in attempted_events
            ),
            target_rates=target_rates,
            target_rates_by_target=target_rates_by_target,
            target_invocation_count=len(attempted_events) + parent_attempted_count,
            parent_invocation_count=parent_invocation_count,
            parent_attempted_count=parent_attempted_count,
            parent_observed_count=parent_observed_count,
            parent_error_count=parent_error_count,
            parent_harness_error_count=parent_error_count,
            parent_terminal_failure_count=parent_terminal_failure_count,
            parent_not_started_count=parent_not_started_count,
            parent_outcome_counts=parent_outcome_counts,
            method_attempted_counts=method_attempted_counts,
            method_observed_counts=method_observed_counts,
            formal_counts=formal_counts,
            **differential_metrics,
            differential_evidence_complete=differential_evidence_complete,
            reference_execution_complete=reference_execution_complete,
            reference_strata=list(board_strata),
            method_outcome_counts=method_outcome_counts,
            target_outcome_counts=target_outcome_counts,
            **event_metric_counts,
            coverage_case_results=str(coverage_case_results_path.relative_to(out))
            if coverage_enabled and coverage_case_results_path.is_file() else None,
            coverage_case_results_sha256=sha256(coverage_case_results_path)
            if coverage_enabled and coverage_case_results_path.is_file() else None,
        )
        write_json_replace(result_path, result)
    derived = out / "derived"
    derived_summary = {
        "execution_complete": execution_complete,
        "formal_ready": formal_ready,
        "method_counts": method_counts,
        "method_attempted_counts": method_attempted_counts,
        "method_observed_counts": method_observed_counts,
        "formal_counts": formal_counts,
        "method_outcome_counts": method_outcome_counts,
        "target_outcome_counts": target_outcome_counts,
        **event_metric_counts,
        "target_rates": target_rates,
        "target_rates_by_target": target_rates_by_target,
        "target_attempted_count": len(attempted_events),
        "target_observed_count": sum(
            _event_observation_recorded(event) for event in attempted_events
        ),
        "target_terminal_observed_count": sum(
            _event_terminal_observed(event) for event in attempted_events
        ),
        "target_invocation_count": len(attempted_events) + parent_attempted_count,
        "parent_invocation_count": parent_invocation_count,
        "parent_attempted_count": parent_attempted_count,
        "parent_observed_count": parent_observed_count,
        "parent_error_count": parent_error_count,
        "parent_harness_error_count": parent_error_count,
        "parent_terminal_failure_count": parent_terminal_failure_count,
        "parent_not_started_count": parent_not_started_count,
        "parent_outcome_counts": parent_outcome_counts,
        "ledger_parse_error_count": ledger_parse_error_count,
        "ledger_seal_parse_error": ledger_seal_parse_error,
        "final_event_ledger_complete": final_event_ledger_complete,
        "partial_reason": recovery_reason or result.get("partial_reason"),
        "queued_count": reported_queued_count,
        "dispatched_count": accounting["dispatched_count"],
        "processed_count": accounting["processed_count"],
        "unattempted_error_count": accounting["unattempted_error_count"],
        "pending_count": reported_pending_count,
        "right_censored_count": accounting["right_censored_count"],
        "queue_accounting_complete": queue_accounting_ok,
        "queue_count_exact": queue_accounting_ok,
        "queue_count_source": ("queue-entries" if queue_accounting_ok else
                                "partial-snapshot-counts" if snapshot_queue_counts else
                                "unavailable"),
        "ledger_seal_complete": ledger_seal_ok,
        "queue_drained": queue_drained,
        "queue_source": queue_source,
        "queue_partial_recovered": queue_partial_recovered,
        "queue_recovery_reason": queue_recovery_reason,
        "recovered_pending_count": recovered_pending_count,
        "event_terminal_complete": event_terminals_ok,
        "method_target_coverage_complete": method_target_coverage_ok,
        "generation_method_coverage_complete": generation_method_coverage_ok,
        "generation_artifacts_complete": generation_artifacts_complete,
        "queue_artifact_shape_complete": queue_artifact_shape_ok,
        "queue_entries_artifacts_complete": queue_entries_artifacts_ok,
        "queue_candidate_identity_complete": queue_candidate_identity_ok,
        "queue_parse_error_count": queue_parse_error_count,
        "coverage_artifact_complete": coverage_artifact_complete,
        "coverage_policy_ok": coverage_policy_ok,
        "generation_shape_complete": generation_shape_ok,
        "queue_candidate_indices_complete": queue_candidate_indices_ok,
        "coverage_case_results_complete": coverage_case_results_complete,
        "coverage_enabled": coverage_enabled,
        "differential_evidence_complete": differential_evidence_complete,
        "reference_execution_complete": reference_execution_complete,
        "reference_strata": list(board_strata),
        "board_reference_named_gap_count": len(board_named_gap_events),
        "source_commit": manifest.get("source_commit"),
        "image_id": manifest.get("image_id"),
        "config_sha256": sha256(config_path),
        "coverage_registry_sha256": coverage_registry_sha256,
        "coverage_registry_file_sha256": (
            sha256(coverage_registry_path) if coverage_registry_path.is_file() else None
        ),
        "coverage_registry_complete": coverage_registry_complete,
        "rv_instruction_metrics_enabled": rv_metrics_enabled,
        "coverage_registry_required": rv_metrics_enabled,
        **differential_metrics,
    }
    campaign = {
        "schema_version": "rq1-campaign-complete-v1",
        "status": "complete" if execution_complete else "gap",
        "run_id": out.name,
        "method_filter": selected_method,
        "method_count": len(methods),
        "event_count": len(events),
        "coverage_method_boundary_complete": coverage_method_boundary_ok,
        "method_boundary_complete": method_boundary_ok,
        "methods_complete": methods_ok,
        "result_complete": result_status_ok,
        "identity_complete": identity_ok,
        "clock_complete": clock_ok,
        "event_ledger_sha256": sha256(events_path),
        "ledger_seal_sha256": sha256(ledger_seal_path),
        "coverage_summary_sha256": sha256(coverage_path)
        if coverage_enabled and coverage_path.is_file() else None,
        "coverage_case_results_sha256": sha256(coverage_case_results_path)
        if coverage_enabled and coverage_case_results_path.is_file() else None,
        "started_at_utc": result.get("started_at_utc"),
        "ended_at_utc": result.get("ended_at_utc"),
        **derived_summary,
    }
    write_json_replace(derived / "campaign-complete.json", campaign)
    ledger_name = str(events_path.relative_to(out))
    queue_file_names = {
        str(path.relative_to(out))
        for path in (out / "queues").glob("*.json")
    }
    queue_file_names.update(
        f"queues/{target.get('id')}.json"
        for target in config.get("targets", [])
        if isinstance(target, dict) and target.get("id")
    )
    files = {
        name: sha256(out / name)
        for name in (
            "execution-manifest.json",
            "coverage-registry.json",
            "generation-result.json",
            "generation/candidate-queues.json",
            *sorted(queue_file_names),
            "run-result.json",
            ledger_name,
            "derived/campaign-complete.json",
        )
    }
    if (out / "config.snapshot.json").is_file():
        files["config.snapshot.json"] = sha256(out / "config.snapshot.json")
    if ledger_seal_path.is_file():
        files["ledger/seal.json"] = sha256(ledger_seal_path)
    if coverage_path.is_file():
        files["coverage/summary.json"] = sha256(coverage_path)
    if coverage_case_results_path.is_file():
        files["coverage/case-results.jsonl.gz"] = sha256(coverage_case_results_path)
    source_profile_root = out / "coverage-profiles"
    if source_profile_root.is_dir():
        # SimSrcCov 的汇总不能脱离实际 LCOV 和 profile identity；把这些
        # run-owned 文件纳入同一 integrity.files，避免只封存摘要而报告被替换。
        files.update({
            str(path.relative_to(out)): sha256(path)
            for path in sorted(source_profile_root.rglob("*"))
            if path.is_file()
        })
    integrity = {
        "schema_version": "rq1-integrity-v1",
        "status": "passed" if execution_complete else "gap",
        "files": files,
        **derived_summary,
    }
    write_json_replace(derived / "integrity.json", integrity)
    seal = {
        "schema_version": "rq1-campaign-seal-v1",
        "sealed": execution_complete,
        "execution_complete": execution_complete,
        "formal_ready": formal_ready,
        "run_id": out.name,
        "method_filter": selected_method,
        "campaign_complete_sha256": sha256(derived / "campaign-complete.json"),
        "integrity_sha256": sha256(derived / "integrity.json"),
        "event_ledger_sha256": sha256(events_path),
        "ledger_seal_sha256": sha256(ledger_seal_path),
        "final_event_ledger_complete": final_event_ledger_complete,
        "queue_source": queue_source,
        "queue_partial_recovered": queue_partial_recovered,
        "queue_recovery_reason": queue_recovery_reason,
        "recovered_pending_count": recovered_pending_count,
        "queue_candidate_identity_complete": queue_candidate_identity_ok,
        "coverage_registry_file_sha256": (
            sha256(coverage_registry_path) if coverage_registry_path.is_file() else None
        ),
        "coverage_summary_sha256": sha256(coverage_path)
        if coverage_enabled and coverage_path.is_file() else None,
        "coverage_case_results_sha256": sha256(coverage_case_results_path)
        if coverage_enabled and coverage_case_results_path.is_file() else None,
    }
    seal["seal_sha256"] = digest(seal)
    write_json_replace(out / "seal.json", seal)
    return {**campaign, "status": "sealed" if execution_complete else "partial"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ1 minimal comparison runner")
    parser.add_argument("command", choices=["self-check", "doctor", "generate", "run", "execute-existing-queues", "framework-run", "finalize"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--target-timeout-seconds", type=int)
    parser.add_argument("--method")
    parser.add_argument("--feedback-target", default="T-LRSV-INT")
    parser.add_argument("--generation-root", type=Path)
    parser.add_argument("--case-catalog", type=Path)
    parser.add_argument("--case-catalog-manifest", type=Path)
    parser.add_argument("--program-seed-catalog", type=Path)
    parser.add_argument("--program-seed-catalog-manifest", type=Path)
    module_group = parser.add_mutually_exclusive_group()
    module_group.add_argument(
        "--use-small-model", dest="small_model_enabled", action="store_true",
    )
    module_group.add_argument(
        "--no-small-model", dest="small_model_enabled", action="store_false",
    )
    parser.set_defaults(small_model_enabled=None)
    parser.add_argument("--no-emi", action="store_true")
    parser.add_argument("--no-mcmc", action="store_true")
    parser.add_argument(
        "--allow-missing-final-identity", action="store_true",
        help="仅供宿主 wrapper 收尾 partial run；不提升执行或正式证据",
    )
    parser.add_argument(
        "--recovery-partial", action="store_true",
        help="仅供宿主恢复；即使事实齐全也保持 partial，不生成正式完成状态",
    )
    args = parser.parse_args(argv)

    if args.command == "self-check":
        from runner_self_check import run_self_check
        print(json.dumps(run_self_check(), ensure_ascii=False, indent=2))
        return 0
    if args.command in {"generate", "run", "execute-existing-queues"} and not args.method:
        print(f"{args.command} requires --method for one-method-per-container execution", file=sys.stderr)
        return 2
    if args.case_catalog_manifest is not None and args.command != "execute-existing-queues":
        print("--case-catalog-manifest only applies to execute-existing-queues", file=sys.stderr)
        return 2
    if (args.program_seed_catalog is None) != (args.program_seed_catalog_manifest is None):
        print("Program-Full requires both --program-seed-catalog and its pinned manifest", file=sys.stderr)
        return 2
    if args.command != "framework-run" and (
        args.program_seed_catalog is not None
        or args.program_seed_catalog_manifest is not None
    ):
        print("Program-Full seed catalogs only apply to framework-run", file=sys.stderr)
        return 2
    if args.command == "framework-run" and args.method == "Ours-Program-Full" and (
        args.program_seed_catalog is None
        or args.program_seed_catalog_manifest is None
    ):
        print("Ours-Program-Full requires a pinned B-RVDV seed catalog", file=sys.stderr)
        return 2
    if args.command == "framework-run" and args.method != "Ours-Program-Full" and (
        args.program_seed_catalog is not None
        or args.program_seed_catalog_manifest is not None
    ):
        print("B-RVDV seed catalogs only apply to Ours-Program-Full", file=sys.stderr)
        return 2
    module_toggle_requested = (
        args.small_model_enabled is not None or args.no_emi or args.no_mcmc
    )
    if module_toggle_requested and args.command != "framework-run":
        print("small-model, EMI and MCMC toggles only apply to framework-run", file=sys.stderr)
        return 2
    if module_toggle_requested and args.method not in {
        "Ours-RVGEN-Direct", "Ours-Program-Full",
    }:
        print("module toggles only apply to Ours framework methods", file=sys.stderr)
        return 2
    if args.command == "framework-run" and args.no_emi and not args.no_mcmc:
        print("--no-emi requires --no-mcmc because MCMC uses EMI proposals", file=sys.stderr)
        return 2
    if args.output is None:
        print("rq1 comparison: --output is required", file=sys.stderr)
        return 2
    stop_handlers = _install_stop_handlers()
    try:
        if args.command == "finalize":
            output = args.output.resolve()
            config, config_path = _load_frozen_config(output, args.config)
        else:
            output = _prepare_output(args.output)
            config_path = _freeze_config(output, args.config)
            config = load_json(config_path)
        config_gaps = validate_config(config)
        framework_finalize = (
            args.command == "finalize"
            and load_json(output / "execution-manifest.json").get("experiment_face")
            == "framework"
        )
        if config_gaps and args.command != "doctor" and not framework_finalize:
            raise ValueError("invalid configuration: " + ",".join(config_gaps))
        provenance = _provenance(config, config_path)
        if args.command in {"run", "generate", "execute-existing-queues"}:
            from decoupled_pipeline import run_decoupled_campaign

        if args.command == "finalize":
            manifest = load_json(output / "execution-manifest.json")
            result = (
                finalize_framework_campaign(
                    output, args.config, recovery_partial=args.recovery_partial,
                )
                if manifest.get("experiment_face") == "framework"
                else finalize_campaign(
                    output, args.config,
                    require_final_identity=not args.allow_missing_final_identity,
                    recovery_partial=args.recovery_partial,
                )
            )
        elif args.command == "doctor":
            result = doctor(config, provenance=provenance)
            write_json(output / "doctor.json", result)
        elif args.command in {"run", "execute-existing-queues"}:
            if args.command == "execute-existing-queues" and (
                (args.generation_root is None) == (args.case_catalog is None)
            ):
                raise ValueError(
                    "execute-existing-queues requires exactly one of "
                    "--generation-root or --case-catalog"
                )
            if args.generation_root is not None and args.case_catalog is not None:
                raise ValueError("choose one of --generation-root or --case-catalog")
            if args.case_catalog_manifest is not None and args.case_catalog is None:
                raise ValueError("--case-catalog-manifest requires --case-catalog")
            result = run_decoupled_campaign(
                config, output, provenance=provenance,
                generation_root=(
                    args.generation_root
                    if args.command == "execute-existing-queues" else None
                ),
                case_catalog_root=(
                    args.case_catalog
                    if args.command == "execute-existing-queues" else None
                ),
                case_catalog_manifest=(
                    args.case_catalog_manifest
                    if args.command == "execute-existing-queues" else None
                ),
                target_timeout_seconds=args.target_timeout_seconds,
                method_filter=args.method,
                seed=int(config.get("experiment_seed", 303)),
                defer_source_coverage=(
                    args.command in {"run", "execute-existing-queues"}
                ),
            )
            if not (output / "run-result.json").is_file():
                write_json(output / "run-result.json", result)
            if stop_requested() or args.command in {"run", "execute-existing-queues"} \
                    or os.environ.get("RQ1_DEFER_FINALIZE") == "1":
                result = {**result, "finalize_status": "deferred-to-wrapper"}
            else:
                final = finalize_campaign(
                    output, args.config, require_final_identity=False,
                )
                write_json_replace(output / "finalization-result.json", {
                    "schema_version": "rq1-finalization-result-v1",
                    "status": "sealed" if final.get("status") == "sealed" else "partial",
                    "exit_code": 0 if final.get("status") == "sealed" else 125,
                    "method_filter": final.get("method_filter"),
                    "run_status": final.get("status"),
                    "seal_status": final.get("execution_complete") is True,
                    "finalized_in_container": True,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                result = {**result, "finalize_status": final.get("status")}
        elif args.command == "framework-run":
            if args.method is None:
                raise ValueError("--method is required for framework-run")
            os.environ["RQ1_RAW_COVERAGE_ONLY"] = "1"
            result = run_framework_campaign(
                config, output, method=args.method,
                feedback_target=args.feedback_target,
                timeout_seconds=args.duration_seconds,
                target_timeout_override=args.target_timeout_seconds,
                program_seed_catalog_root=args.program_seed_catalog,
                program_seed_catalog_manifest=args.program_seed_catalog_manifest,
                small_model_enabled=(
                    True if args.small_model_enabled is None
                    else args.small_model_enabled
                ),
                emi_enabled=not args.no_emi,
                mcmc_enabled=not args.no_mcmc,
            )
        elif args.command == "generate":
            result = run_decoupled_campaign(
                config, output, provenance=provenance,
                generation_seconds=args.duration_seconds,
                execute_targets=False, method_filter=args.method,
                seed=int(config.get("experiment_seed", 303)),
            )
            if not (output / "generation-result.json").is_file():
                write_json(output / "generation-result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        deferred_finalize = (
            args.command in {"run", "execute-existing-queues"}
            and result.get("finalize_status") == "deferred-to-wrapper"
        )
        status = result.get("finalize_status") if args.command in {"run", "execute-existing-queues"} else result.get("status")
        if deferred_finalize:
            status = result.get("status")
        if args.command in {"run", "execute-existing-queues", "finalize"}:
            if deferred_finalize:
                return 0 if status == "completed" else 125 if status == "partial" else 2
            return 0 if status == "sealed" else 125 if status == "partial" else 2
        return 0 if status in {"ready", "passed", "completed", "partial", "sealed"} else 2
    except (IndexError, KeyError, OSError, StopIteration, TypeError, ValueError) as exc:
        print(f"rq1 comparison: invalid input or run state: {exc}", file=sys.stderr)
        return 2
    finally:
        _restore_stop_handlers(stop_handlers)
        _release_run_lock(args.output)

if __name__ == "__main__":
    raise SystemExit(main())

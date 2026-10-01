from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from functools import cache
import shutil
import signal as signal_module
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
import re

from .direct_case import (
    OBSERVATION_HEADER_SIZE,
    OBSERVATION_MAGIC,
    NATIVE_OBSERVER_TEMP_GPRS,
    Observation,
    _SIGNAL_CODE_BY_NAME,
    _SIGNAL_NAME_BY_TEXT,
    _TRAP_FIELD_ALIASES,
    _TRANSLATION_PATHS,
    TranslationEvidence,
    _checkpoint_state_chain_valid,
    _runner_contract_gap_observation as _gap_observation,
    _valid_window_checkpoint,
    native_trace_stdout_mismatch_is_compatible,
)
from .execution_environment import require_execution_plane
from .execution_identity import (
    ExecutionEvidenceBundle,
    LIVE_NATIVE_REFERENCE_EVIDENCE_TIER,
    LIVE_NATIVE_REFERENCE_IDENTITY_SOURCE,
    LIVE_NATIVE_REFERENCE_PROVENANCE,
    NativeReferenceIdentity,
)
from .adapters.runner import (
    CAPSULE_MAILBOX_ENV,
    CAPSULE_MEMORY_SIZE_ENV,
    CAPSULE_OBSERVATION_SIZE_ENV,
    CAPSULE_TEST_MEMORY_ENV,
    GUEST_TRAP_BRIDGE_FIELDS,
    LINUX_MEMORY_SIZE_ENV,
    LINUX_TEST_MEMORY_ENV,
    _observation_frame,
    _observation_frame_end,
    canonical_trap_state,
    observed_state_fields,
    terminal_ebreak_observed as _terminal_ebreak_observed,
    source_pc_map as _source_pc_map,
)
from .adapters.contracts import (
    CAPSULE_BACKENDS, backend_capability_gap, backend_spec, is_execution_backend,
)
from .paths import FRAMEWORK_RUNS_ROOT, REPOSITORY_ROOT
from .native_reference_gate import (
    MAX_NATIVE_REFERENCE_PARALLELISM,
    NATIVE_REFERENCE_PARALLELISM,
    NativeReferenceLock as _NativeReferenceLock,
    configured_native_reference_parallelism as _configured_native_reference_parallelism,
)
from .spec_definedness import is_canonical_isa_profile
from .reference_fallback import K1_QEMU_FALLBACK_FAILURE_CLASSES
from ._util import (
    canonical_digest, execute_process as _run_backend_process,
    is_sha256_digest,
    pc_digest as _trace_pc_digest,
    pc_int as _trace_int,
    sha256_file as binary_sha256,
)


DEFAULT_TIMEOUT_SEC = 300
_BACKEND_TIMEOUT_SEC = {
    "unicorn-riscv64": 300,
    # Renode's first .NET/tlib start can exceed the short capsule timeout;
    # its adapter bounds the actual GDB handshake within this budget.
    "renode-riscv64": 600,
    "rax-riscv64": 300,
    # RVVM PC-trace paging is GDB-RSP round-trip bound; long custom programs
    # need more wall time to drain a complete trace.
    "rvvm-riscv64": 300,
    # libriscv runs main/count/trace plus a GDB RSP FP observation that
    # serializes on the fixed 2159 stub port; under parallel lab load the
    # tail worker can exceed 120s. Lock wait and backend execution share one
    # process deadline, so queueing cannot extend the configured attempt time.
    "libriscv-translated": 300,
}
_QEMU_TARGET_SIGNAL_RE = re.compile(
    r"uncaught target signal\s+(?P<code>[0-9]+)\s+\((?P<name>[^)]+)\)",
    re.IGNORECASE,
)


_CAPSULE_SIGNAL_BY_CAUSE = {
    0: "SIGBUS", 1: "SIGSEGV", 2: "SIGILL", 3: "SIGTRAP",
    4: "SIGBUS", 5: "SIGSEGV", 6: "SIGBUS", 7: "SIGSEGV",
}
def _framework_cli_script(name: str) -> str:
    return str((REPOSITORY_ROOT / "framework" / "cli" / name).resolve())


def command_for_backend(
    backend: str,
    elf_path: Path,
    backend_binary_path: str | None = None,
) -> tuple[str, ...]:
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    elf = str(elf_path.resolve())
    if backend == "native-rv64":
        machine = os.uname().machine if hasattr(os, "uname") else (
            os.environ.get("PROCESSOR_ARCHITEW6432")
            or os.environ.get("PROCESSOR_ARCHITECTURE", "")
        )
        if machine.lower() in {"riscv64", "riscv"}:
            return (elf,)
        return (sys.executable, "-m", "framework.native.rv64_runner", elf)
    if is_execution_backend(backend):
        spec = backend_spec(backend)
        if spec.runner_entrypoint_kind == "script":
            command = [sys.executable, _framework_cli_script(spec.runner_entrypoint)]
        elif spec.runner_entrypoint_kind == "module":
            command = [sys.executable, "-m", spec.runner_entrypoint]
        else:
            raise ValueError(f"unsupported runner entrypoint kind: {spec.runner_entrypoint_kind}")
        if spec.runner_pass_backend:
            command.extend(("--backend", backend))
        if spec.runner_mode:
            command.extend(("--mode", spec.runner_mode))
        if backend_binary_path and spec.runner_binary_option:
            command.extend((spec.runner_binary_option, backend_binary_path))
        command.append(elf)
        return tuple(command)
    raise ValueError(f"unknown direct backend: {backend}")


@cache
def _tool_version(command: tuple[str, ...]) -> str | None:
    """Memoize the version of the actual backend binary, not its Python wrapper."""
    for flag in ("--qemu-bin", "--rvlinux-bin"):
        try:
            return _executable_tool_version(command[command.index(flag) + 1])
        except (ValueError, IndexError):
            continue
    return None if Path(command[0]).name.lower().startswith(("python", "pypy")) else _executable_tool_version(command[0])


@cache
def _executable_tool_version(executable: str) -> str | None:
    if shutil.which(executable) is None and not Path(executable).exists():
        return None
    try:
        proc = subprocess.run(
            [executable, "--version"],
            timeout=2,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return executable
    if proc.returncode != 0:
        return executable
    text = (proc.stdout or proc.stderr).decode("utf-8", errors="replace").strip()
    return text.splitlines()[0] if text else executable


def _binary_hash(path: Path) -> str | None:
    try:
        return binary_sha256(path) if path.is_file() else None
    except OSError:
        return None

def _augment_source_pc_map(source_pc_map, observations):
    instructions = source_pc_map.get("instructions") if isinstance(source_pc_map, dict) else None
    lines = source_pc_map.get("lines") if isinstance(source_pc_map, dict) else None
    if not isinstance(instructions, list) or not isinstance(lines, dict):
        return source_pc_map
    starts = {item["pc"] for item in instructions if isinstance(item, dict)
              and type(item.get("pc")) is int}
    owners = {}
    for item in instructions:
        if (
            not isinstance(item, dict)
            or type(item.get("pc")) is not int
            or type(item.get("size")) is not int
        ):
            continue
        for offset in range(0, item["size"], 2):
            owners.setdefault(item["pc"] + offset, item["pc"])
    line_for = {pc: line for line, pcs in lines.items() for pc in pcs}
    additions = {}
    for observation in observations:
        for pc in observation.executed_pcs:
            line = line_for.get(owners.get(pc)) if pc not in starts else None
            if line is not None:
                additions.setdefault(line, []).append(pc)
    if not additions:
        return source_pc_map
    result = {**source_pc_map, "lines": {line: list(pcs) for line, pcs in lines.items()}}
    for line, pcs in additions.items():
        result["lines"].setdefault(line, []).extend(pcs)
        result["lines"][line] = sorted(set(result["lines"][line]))
    # ponytail: overlap paths keep source-line evidence; byte mapping stays linear-only.
    result.pop("instructions", None)
    return result


def _attach_source_pc_map(backend: str, executable: Path, observations: list[Observation]):
    if not backend.startswith("qemu-riscv") or not observations:
        return observations
    missing = object()
    source_pc_map = next(
        (
            observation.translation_evidence.details["source_pc_map"]
            for observation in observations
            if observation.translation_evidence is not None
            and "source_pc_map" in observation.translation_evidence.details
        ),
        missing,
    )
    if source_pc_map is missing:
        source_pc_map = _source_pc_map(executable)
    if source_pc_map is None:
        return observations
    source_pc_map = _augment_source_pc_map(source_pc_map, observations)
    source_digest = source_pc_map.get("binary_sha256") if isinstance(source_pc_map, dict) else None
    result = []
    for observation in observations:
        if source_digest != observation.binary_sha256:
            result.append(_source_pc_map_gap(observation))
            continue
        evidence = observation.translation_evidence
        if evidence is not None and evidence.details.get("source_pc_map") == source_pc_map:
            result.append(observation)
            continue
        result.append(replace(
            observation,
            translation_evidence=(
                replace(evidence, details={**evidence.details, "source_pc_map": source_pc_map})
                if evidence is not None else TranslationEvidence(
                    observation.backend, "either", details={"source_pc_map": source_pc_map}
                )
            ),
        ))
    return result


def _source_pc_map_gap(observation: Observation) -> Observation:
    """Drop a mismatched auxiliary map without discarding guest observation."""
    extra_state = dict(observation.extra_state)
    previous_gaps = extra_state.get("observer_gaps")
    observer_gaps = list(previous_gaps) if isinstance(previous_gaps, (list, tuple)) else []
    reason = "source-pc-map:binary-identity-mismatch"
    if reason not in observer_gaps:
        observer_gaps.append(reason)
    extra_state["observer_gaps"] = observer_gaps
    evidence = observation.translation_evidence
    if evidence is None:
        return replace(observation, extra_state=extra_state)
    details = dict(evidence.details)
    details.pop("source_pc_map", None)
    details["source_pc_map_gap"] = "binary-identity-mismatch"
    return replace(
        observation,
        extra_state=extra_state,
        translation_evidence=replace(evidence, details=details),
    )


def _execution_evidence_gap(
    observation: Observation, reason: str, *, path_gap: bool = False,
) -> Observation:
    """Record auxiliary-runner evidence failure while retaining the main run."""
    extra_state = dict(observation.extra_state)
    previous_gaps = extra_state.get("observer_gaps")
    observer_gaps = list(previous_gaps) if isinstance(previous_gaps, (list, tuple)) else []
    marker = f"execution-evidence:{reason}"
    if marker not in observer_gaps:
        observer_gaps.append(marker)
    extra_state["observer_gaps"] = observer_gaps
    evidence = observation.translation_evidence
    if evidence is None:
        return replace(observation, extra_state=extra_state)
    details = dict(evidence.details)
    details["execution_evidence_gap"] = reason
    if path_gap:
        details["execution_evidence_path_divergence"] = True
        details["trace_failed"] = True
    return replace(
        observation,
        extra_state=extra_state,
        translation_evidence=replace(evidence, details=details),
    )


def _timeout_observation(
    backend: str,
    raw_stderr: str,
    binary_hash: str | None,
    tool_version: str | None,
    profile_id: str | None,
    input_id: str | None,
    *,
    raw_stdout: str,
    capsule_memory_size: int | None = None,
) -> Observation:
    try:
        stdout = bytes.fromhex(raw_stdout)
    except ValueError:
        stdout = b""
    if _observation_frame(stdout) is not None:
        parsed = parse_observation_stdout(
            backend,
            1,
            stdout,
            raw_stderr.encode("utf-8", errors="replace"),
            binary_hash,
            tool_version,
            profile_id,
            input_id,
        )
        snapshot = parsed.memory_delta.get("test-memory")
        capsule_memory_matches = (
            backend not in CAPSULE_BACKENDS
            or capsule_memory_size is not None
            and isinstance(snapshot, str)
            and len(snapshot) // 2 == capsule_memory_size
        )
        frame_valid = (
            parsed.contract_error is None
            and parsed.outcome in {"normal", "completed", "trap", "nonzero-exit"}
            and parsed.checkpoint_pc is not None
            and all(value is not None for value in parsed.gpr)
            and isinstance(snapshot, str)
            and parsed.memory_digest is not None
            and capsule_memory_matches
        )
        if frame_valid:
            extra_state = dict(parsed.extra_state)
            extra_state["execution_status"] = "timeout-after-RVOBS1-frame"
            if backend == "native-rv64":
                extra_state["runner_failure_class"] = "k1-timeout"
            return replace(
                parsed,
                outcome="timeout",
                exit_code=None,
                extra_state=extra_state,
            )
    observation = _gap_observation(
        backend, "", raw_stderr, binary_hash, profile_id, input_id,
        raw_stdout=raw_stdout,
        translation_evidence=_translation_evidence(backend, raw_stderr),
        tool_version=tool_version,
    )
    identity, identity_error = _reference_identity_contract(raw_stderr)
    executed_pcs, instruction_count, trace_error = _runner_trace(raw_stderr, backend)
    extra_state = dict(observation.extra_state or {})
    if identity is not None:
        extra_state["reference_identity"] = identity
    if backend == "native-rv64":
        extra_state["runner_failure_class"] = "k1-timeout"
    contract_error = identity_error
    return replace(
        observation,
        outcome="timeout",
        contract_error=contract_error,
        executed_pcs=executed_pcs,
        instruction_count=instruction_count,
        extra_state=extra_state,
    )


def _runner_total_guest_count(raw_stderr: str) -> int | None:
    if not isinstance(raw_stderr, str):
        return None
    prefix = "RV_TOTAL_GUEST_INSTRUCTION_COUNT="
    lines = [line for line in raw_stderr.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        return None
    try:
        value = int(lines[0].removeprefix(prefix), 0)
    except ValueError:
        return None
    return value if value >= 0 else None


def _runner_window_checkpoints(raw_stderr: str) -> tuple[list[dict] | None, str | None]:
    if not isinstance(raw_stderr, str):
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS stderr must be text"
    prefix = "RV_NATIVE_WINDOW_CHECKPOINTS="
    lines = [line for line in raw_stderr.splitlines() if line.startswith(prefix)]
    if len(lines) > 1:
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS must appear exactly once"
    if not lines:
        return None, None
    try:
        data = json.loads(lines[0].removeprefix(prefix))
    except json.JSONDecodeError:
        return None, "invalid RV_NATIVE_WINDOW_CHECKPOINTS json"
    if not isinstance(data, list):
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS must be a JSON list"
    if not data:
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS must not be empty"
    for item in data:
        if not isinstance(item, dict):
            return None, "RV_NATIVE_WINDOW_CHECKPOINTS item must be a JSON object"
        if not _valid_window_checkpoint(item):
            return None, "RV_NATIVE_WINDOW_CHECKPOINTS item has invalid shape"
    if any(
        _trace_int(left["after_pc"]) != _trace_int(right["pc"])
        for left, right in zip(data, data[1:])
    ):
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS pc chain is not contiguous"
    try:
        if not _checkpoint_state_chain_valid(data):
            return None, "RV_NATIVE_WINDOW_CHECKPOINTS state chain is not contiguous"
    except (TypeError, ValueError):
        return None, "RV_NATIVE_WINDOW_CHECKPOINTS state chain is invalid"
    return data, None


def _runner_contract_gap_error(raw_stderr: str) -> str | None:
    if not isinstance(raw_stderr, str):
        return "runner stderr must be text"
    prefix = "RV_RUNNER_CONTRACT_GAP="
    lines = [line for line in raw_stderr.splitlines() if line.startswith(prefix)]
    if len(lines) > 1:
        return "RV_RUNNER_CONTRACT_GAP must appear exactly once"
    for line in lines:
        raw = line.removeprefix(prefix).strip()
        if not raw:
            return "runner reported a contract gap"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            reason = data.get("reason") or data.get("message") or data.get("error")
            if reason is not None:
                return str(reason)
        return "runner reported a contract gap"
    return None


def _runner_failure_class(raw_stderr: str) -> str | None:
    if not isinstance(raw_stderr, str):
        return None
    lines = [
        line.removeprefix("RV_RUNNER_CONTRACT_GAP=").strip()
        for line in raw_stderr.splitlines()
        if line.startswith("RV_RUNNER_CONTRACT_GAP=")
    ]
    if len(lines) != 1:
        return None
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    failure_class = payload.get("failure_class") if isinstance(payload, dict) else None
    return (
        failure_class
        if isinstance(failure_class, str)
        and failure_class in K1_QEMU_FALLBACK_FAILURE_CLASSES
        else None
    )


def _runner_pc_artifact(reference: object) -> tuple[int, ...]:
    if not isinstance(reference, dict):
        raise ValueError("RV_EXECUTED_PCS_FILE must be a JSON object")
    path_value = reference.get("absolute_path") or reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("RV_EXECUTED_PCS_FILE path is missing")
    pcs: list[int] = []
    with Path(path_value).open("r", encoding="ascii") as stream:
        for line in stream:
            value = line.strip()
            if not value:
                continue
            pc = _trace_int(value)
            if pc <= 0 or pc & 1:
                raise ValueError("RV_EXECUTED_PCS_FILE contains an invalid PC")
            pcs.append(pc)
    expected_count = reference.get("pc_count")
    if expected_count is not None:
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("RV_EXECUTED_PCS_FILE count is invalid")
        if expected_count > len(pcs):
            raise ValueError("RV_EXECUTED_PCS_FILE count does not match trace")
        if expected_count < len(pcs):
            # A simulator can append its final stop/flush PCs after the adapter
            # has published the stable prefix. Keep the published evidence,
            # but only when its digest proves that the extra lines are a suffix.
            expected_digest = reference.get("pc_digest")
            if not isinstance(expected_digest, str) or not expected_digest:
                raise ValueError("RV_EXECUTED_PCS_FILE prefix digest is missing")
            prefix = tuple(pcs[:expected_count])
            if _trace_pc_digest(prefix) != expected_digest:
                raise ValueError("RV_EXECUTED_PCS_FILE prefix digest mismatch")
            pcs = list(prefix)
    return tuple(pcs)


def _runner_trace(
    raw_stderr: str, backend: str,
) -> tuple[tuple[int, ...], int | None, str | None]:
    if not isinstance(raw_stderr, str):
        return (), None, "runner stderr must be text"
    executed_pcs: tuple[int, ...] = ()
    instruction_count: int | None = None
    saw_pcs = False
    saw_count = False
    for line in raw_stderr.splitlines():
        if line.startswith("RV_EXECUTED_PCS="):
            if saw_pcs:
                return (), None, "RV_EXECUTED_PCS must appear exactly once"
            saw_pcs = True
            try:
                data = json.loads(line.removeprefix("RV_EXECUTED_PCS="))
                if not isinstance(data, list):
                    return (), None, "RV_EXECUTED_PCS must be a JSON list"
                executed_pcs = tuple(_trace_int(item) for item in data)
            except (TypeError, ValueError):
                return (), None, "invalid RV_EXECUTED_PCS"
            if any(pc <= 0 for pc in executed_pcs):
                return (), None, "RV_EXECUTED_PCS contains zero PC"
            if any(pc & 1 for pc in executed_pcs):
                return (), None, "RV_EXECUTED_PCS contains unaligned PC"
            if backend == "qemu-riscv32" and any(pc >= 1 << 32 for pc in executed_pcs):
                return (), None, "RV32 RV_EXECUTED_PCS exceeds XLEN"
        elif line.startswith("RV_EXECUTED_PCS_FILE="):
            if saw_pcs:
                return (), None, "RV_EXECUTED_PCS must appear exactly once"
            saw_pcs = True
            try:
                reference = json.loads(line.removeprefix("RV_EXECUTED_PCS_FILE="))
                executed_pcs = _runner_pc_artifact(reference)
            except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
                return (), None, "invalid RV_EXECUTED_PCS_FILE"
            if backend == "qemu-riscv32" and any(pc >= 1 << 32 for pc in executed_pcs):
                return (), None, "RV32 RV_EXECUTED_PCS_FILE exceeds XLEN"
        elif line.startswith("RV_INSTRUCTION_COUNT="):
            if saw_count:
                return (), None, "RV_INSTRUCTION_COUNT must appear exactly once"
            saw_count = True
            try:
                instruction_count = int(line.removeprefix("RV_INSTRUCTION_COUNT="), 0)
            except ValueError:
                return (), None, "invalid RV_INSTRUCTION_COUNT"
            if instruction_count < 0:
                return (), None, "invalid RV_INSTRUCTION_COUNT"
    total_lines = [
        line for line in raw_stderr.splitlines()
        if line.startswith("RV_TOTAL_GUEST_INSTRUCTION_COUNT=")
    ]
    total = _runner_total_guest_count(raw_stderr) if len(total_lines) == 1 else None
    if total_lines and total is None:
        return (), None, (
            "RV_TOTAL_GUEST_INSTRUCTION_COUNT must appear exactly once"
            if len(total_lines) != 1 else "invalid RV_TOTAL_GUEST_INSTRUCTION_COUNT"
        )
    if instruction_count is None:
        instruction_count = total
    elif total is not None and instruction_count != total and backend != "libriscv-translated":
        return (), None, "RV_INSTRUCTION_COUNT does not match RV_TOTAL_GUEST_INSTRUCTION_COUNT"
    if saw_pcs and instruction_count is not None and instruction_count != len(executed_pcs):
        return (), None, "RV_INSTRUCTION_COUNT does not match RV_EXECUTED_PCS"
    if instruction_count is None and executed_pcs:
        instruction_count = len(executed_pcs)
    return executed_pcs, instruction_count, None


def _runner_observation_state(raw_stderr: str) -> tuple[dict[str, object] | None, str | None]:
    if not isinstance(raw_stderr, str):
        return None, "runner stderr must be text"
    prefix = "RV_OBSERVATION_STATE="
    lines = raw_stderr.splitlines()
    if sum(line.startswith(prefix) for line in lines) > 1:
        return None, "RV_OBSERVATION_STATE must appear exactly once"
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None, "invalid RV_OBSERVATION_STATE"
        if not isinstance(data, dict):
            return None, "RV_OBSERVATION_STATE must be a JSON object"
        extra_state = data.get("extra_state", {})
        if extra_state is None:
            extra_state = {}
        if not isinstance(extra_state, dict):
            return None, "RV_OBSERVATION_STATE.extra_state must be a JSON object"
        try:
            signal = data.get("signal")
            if signal is not None and not isinstance(signal, str):
                return None, "invalid RV_OBSERVATION_STATE.signal"
            if signal is not None and signal.strip().lower() == "uc-exception":
                guest_trap = data.get("guest_trap", extra_state.get("guest_trap"))
                guest_cause = data.get("guest_cause", extra_state.get("guest_cause"))
                if guest_trap == "delivered" and type(guest_cause) is int:
                    signal = _CAPSULE_SIGNAL_BY_CAUSE.get(guest_cause, signal)
            fault_pc = data.get("fault_pc")
            fault_address = data.get("fault_address")
            signal_code = data.get("signal_code")
            if signal_code is not None and (
                not isinstance(signal_code, int)
                or isinstance(signal_code, bool)
                or signal_code <= 0
            ):
                return None, "invalid RV_OBSERVATION_STATE.signal_code"
            if signal is not None:
                normalized = _normalized_signal_name(signal, signal_code)
                if normalized is None:
                    if signal_code is not None or not signal.strip():
                        return None, "invalid RV_OBSERVATION_STATE.signal"
                    signal = signal.strip()
                else:
                    signal = normalized
                normalized_code = _normalized_signal_name(None, signal_code)
                if (
                    signal_code is not None
                    and normalized_code is not None
                    and normalized_code != signal
                ):
                    return None, "RV_OBSERVATION_STATE.signal does not match signal_code"
                if (
                    signal in _SIGNAL_CODE_BY_NAME
                    and signal_code is not None
                    and _normalized_signal_name(None, signal_code) != signal
                ):
                    return None, "RV_OBSERVATION_STATE.signal does not match signal_code"
            # Preserve the guest exception bridge fields emitted by capsule
            # adapters so the definedness oracle can compare
            # the delivered trap cause instead of guessing from a signal.
            for key in (
                *GUEST_TRAP_BRIDGE_FIELDS,
                "guest_trap_error", "trap_observer", "guest_tval_observer",
                "fault_address_observer", "memory_fault_observer",
                "memory_access", "memory_size",
                "observer_fields",
            ):
                if key in data and data[key] is not None:
                    if key in extra_state and extra_state[key] != data[key]:
                        return None, f"RV_OBSERVATION_STATE.{key} conflicts with extra_state"
                    extra_state[key] = data[key]
            if extra_state.get("guest_trap") not in (None, "delivered", "not-observed"):
                return None, "invalid RV_OBSERVATION_STATE.guest_trap"
            for aliases in _TRAP_FIELD_ALIASES.values():
                values = [
                    extra_state[alias]
                    for alias in aliases
                    if alias in extra_state and extra_state[alias] is not None
                ]
                if any(type(value) is not int or not 0 <= value < (1 << 64) for value in values):
                    return None, "invalid RV_OBSERVATION_STATE trap field"
                if len(values) > 1 and len(set(values)) != 1:
                    return None, "RV_OBSERVATION_STATE trap aliases conflict"
            if any(
                type(value) is int and value & 1
                for value in (
                    extra_state.get(alias)
                    for alias in _TRAP_FIELD_ALIASES["trap.epc"]
                )
                if value is not None
            ):
                return None, "RV_OBSERVATION_STATE trap EPC is unaligned"
            observer_fields = extra_state.get("observer_fields")
            if observer_fields is not None and (
                not isinstance(observer_fields, (list, tuple))
                or any(not isinstance(field, str) or not field for field in observer_fields)
            ):
                return None, "invalid RV_OBSERVATION_STATE.observer_fields"
            for field in ("fault_pc", "fault_address"):
                extra_value = extra_state.get(field)
                if extra_value is None:
                    continue
                try:
                    extra_value = _trace_int(extra_value)
                except (TypeError, ValueError):
                    return None, f"invalid RV_OBSERVATION_STATE.extra_state.{field}"
                extra_state[field] = extra_value
                if data.get(field) is not None and extra_value != _trace_int(data[field]):
                    return None, f"RV_OBSERVATION_STATE.{field} does not match extra_state"
            extra_state.update(canonical_trap_state(extra_state))
            return {
                "signal": signal,
                "signal_code": signal_code,
                "fault_pc": _trace_int(fault_pc) if fault_pc is not None else None,
                "fault_address": _trace_int(fault_address) if fault_address is not None else None,
                "extra_state": dict(extra_state),
            }, None
        except (TypeError, ValueError):
            return None, "invalid RV_OBSERVATION_STATE field"
    return None, None


def _normalized_signal_name(
    raw_name: str | None,
    signal_number: int | None,
) -> str | None:
    if isinstance(raw_name, str):
        stripped = raw_name.strip()
        if stripped:
            candidate = stripped.upper()
            if candidate.startswith("SIG"):
                return candidate if (
                    candidate in _SIGNAL_CODE_BY_NAME
                    or getattr(signal_module, candidate, None) is not None
                ) else None
            mapped = _SIGNAL_NAME_BY_TEXT.get(stripped.lower())
            if mapped is not None:
                return mapped
    if isinstance(signal_number, int) and signal_number > 0:
        for name, number in _SIGNAL_CODE_BY_NAME.items():
            if number == signal_number:
                return name
        try:
            return signal_module.Signals(signal_number).name
        except ValueError:
            return None
    return None


def _stderr_signal_state(raw_stderr: str) -> dict[str, object]:
    if not isinstance(raw_stderr, str):
        return {}
    match = _QEMU_TARGET_SIGNAL_RE.search(raw_stderr)
    if match is not None:
        signal_number = int(match.group("code"))
        signal_name = _normalized_signal_name(match.group("name"), signal_number)
        if signal_name is not None and (
            _normalized_signal_name(None, signal_number) in (None, signal_name)
        ):
            return {
                "signal": signal_name,
                "signal_code": signal_number,
            }
    for line in raw_stderr.splitlines():
        text_line = line.strip().lower()
        for text, signal_name in _SIGNAL_NAME_BY_TEXT.items():
            if text_line not in {text, f"{text} (core dumped)"}:
                continue
            try:
                signal_number = int(getattr(signal_module, signal_name))
            except (TypeError, ValueError, AttributeError):
                signal_number = None
            return {
                "signal": signal_name,
                "signal_code": signal_number,
            }
    return {}


def _derived_signal_state(
    exit_code: int | None,
    observation_state: dict[str, object],
    raw_stderr: str,
) -> dict[str, object]:
    if observation_state.get("signal") is not None:
        return observation_state
    stderr_state = (
        _stderr_signal_state(raw_stderr)
        if isinstance(exit_code, int) and exit_code != 0
        else {}
    )
    if stderr_state:
        return {
            **observation_state,
            "signal": stderr_state.get("signal"),
            "signal_code": stderr_state.get("signal_code"),
        }
    if not isinstance(exit_code, int):
        return observation_state
    if exit_code < 0:
        signal_number = -exit_code
    else:
        return observation_state
    signal_name = _normalized_signal_name(None, signal_number)
    if signal_name is None:
        return observation_state
    signal_code = observation_state.get("signal_code")
    if not isinstance(signal_code, int) or isinstance(signal_code, bool) or signal_code <= 0:
        signal_code = signal_number
    return {
        **observation_state,
        "signal": signal_name,
        "signal_code": signal_code,
    }


def _translation_evidence(
    backend: str,
    raw_stderr: str,
) -> TranslationEvidence | None:
    parsed = _parse_translation_evidence(raw_stderr, backend)
    runner_details: dict = {}
    total = _runner_total_guest_count(raw_stderr)
    if total is not None:
        runner_details["total_guest_instruction_count"] = total
    checkpoints, checkpoint_error = _runner_window_checkpoints(raw_stderr)
    if checkpoints is not None:
        runner_details["native_window_checkpoints"] = checkpoints
        runner_details["native_window_checkpoint_digest"] = canonical_digest(
            {"native_window_checkpoints": checkpoints}
        )
    elif checkpoint_error is not None:
        runner_details["native_window_checkpoint_error"] = checkpoint_error
    if parsed is not None:
        return replace(parsed, details={**parsed.details, **runner_details}) if runner_details else parsed
    if runner_details:
        return TranslationEvidence(
            backend=backend,
            expected_path="reference" if backend == "native-rv64" else "either",
            details=runner_details,
        )
    if backend == "libriscv-translated":
        return TranslationEvidence(
            backend=backend,
            expected_path="translated",
            tested_pc_seen=False,
            tested_pc_translated=False,
            tested_pc_executed=False,
            details={
                "source": "rvlinux -s translated mode",
                "granularity": "backend-mode",
                "per_pc_trace": False,
                "admissible_for_translated_path_gate": False,
            },
        )
    return None


def _parse_translation_evidence(raw_stderr: str, backend: str) -> TranslationEvidence | None:
    prefix = "RV_TRANSLATION_EVIDENCE="
    lines = raw_stderr.splitlines()
    if sum(line.startswith(prefix) for line in lines) > 1:
        return None
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        # Evidence is emitted by the runner selected by the caller.  A
        # mismatched backend label must not be accepted as a valid path
        # witness for this execution stream.
        declared_backend = data.get("backend")
        if declared_backend is not None and declared_backend != backend:
            return None
        expected_path = data.get("expected_path", "either")
        if not isinstance(expected_path, str) or expected_path not in _TRANSLATION_PATHS:
            return None
        try:
            evidence = TranslationEvidence.from_dict(
                {
                    "backend": data.get("backend", backend),
                    "expected_path": expected_path,
                    "tested_pc_seen": data.get("tested_pc_seen", False),
                    "tested_pc_translated": data.get("tested_pc_translated", False),
                    "tested_pc_executed": data.get("tested_pc_executed", False),
                    "translated_count": data.get("translated_count", 0),
                    "execution_count": data.get("execution_count", 0),
                    "interpreter_count": data.get("interpreter_count", 0),
                    "fallback_count": data.get("fallback_count", 0),
                    "details": data.get("details", {}),
                }
            )
            assert evidence is not None
            return replace(
                evidence,
                details={"source": "runner-contract", **dict(evidence.details)},
            )
        except (AssertionError, TypeError, ValueError):
            return None
    return None


def _translation_contract_error(raw_stderr: str, backend: str) -> str | None:
    prefix = "RV_TRANSLATION_EVIDENCE="
    seen = False
    for line in raw_stderr.splitlines():
        if not line.startswith(prefix):
            continue
        if seen:
            return "RV_TRANSLATION_EVIDENCE must appear exactly once"
        seen = True
        try:
            data = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return "invalid RV_TRANSLATION_EVIDENCE"
        if not isinstance(data, dict):
            return "RV_TRANSLATION_EVIDENCE must be a JSON object"
        for key in ("translated_count", "execution_count", "interpreter_count", "fallback_count"):
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return f"invalid RV_TRANSLATION_EVIDENCE.{key}"
        for key in ("tested_pc_seen", "tested_pc_translated", "tested_pc_executed"):
            if key in data and not isinstance(data[key], bool):
                return f"invalid RV_TRANSLATION_EVIDENCE.{key}"
        if "backend" in data and (not isinstance(data["backend"], str) or not data["backend"]):
            return "invalid RV_TRANSLATION_EVIDENCE.backend"
        if backend is not None and data.get("backend") not in (None, backend):
            return "RV_TRANSLATION_EVIDENCE.backend does not match backend"
        if "expected_path" in data and (
            not isinstance(data["expected_path"], str)
            or data["expected_path"] not in _TRANSLATION_PATHS
        ):
            return "invalid RV_TRANSLATION_EVIDENCE.expected_path"
        details = data.get("details", {})
        if not isinstance(details, dict):
            return "RV_TRANSLATION_EVIDENCE.details must be a JSON object"
    return None


def _execution_evidence_contract(raw_stderr: str) -> tuple[dict | None, str | None]:
    prefix = "RV_EXECUTION_EVIDENCE="
    lines = raw_stderr.splitlines()
    if sum(line.startswith(prefix) for line in lines) > 1:
        return None, "RV_EXECUTION_EVIDENCE must appear exactly once"
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None, "invalid RV_EXECUTION_EVIDENCE"
        if not isinstance(data, dict):
            return None, "RV_EXECUTION_EVIDENCE must be a JSON object"
        if data.get("schema_version") != "runner-execution-evidence-v1":
            return None, "unsupported RV_EXECUTION_EVIDENCE schema"
        for key in ("main_command_fingerprint", "trace_command_fingerprint", "count_command_fingerprint"):
            value = data.get(key)
            if not is_sha256_digest(value):
                return None, f"invalid RV_EXECUTION_EVIDENCE.{key}"
        for key in ("trace_run", "count_run"):
            value = data.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                return None, f"RV_EXECUTION_EVIDENCE.{key} must be an object or null"
            if not isinstance(value.get("stdout_hex"), str):
                return None, f"RV_EXECUTION_EVIDENCE.{key}.stdout_hex must be a string"
            if not isinstance(value.get("stderr"), str):
                return None, f"RV_EXECUTION_EVIDENCE.{key}.stderr must be a string"
            exit_code = value.get("exit_code")
            if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                return None, f"RV_EXECUTION_EVIDENCE.{key}.exit_code must be an integer"
        return data, None
    return None, None


def _native_reference_identity_contract(raw_stderr: str) -> tuple[dict | None, str | None]:
    prefix = "RV_NATIVE_REFERENCE_IDENTITY="
    lines = raw_stderr.splitlines()
    if sum(line.startswith(prefix) for line in lines) > 1:
        return None, "RV_NATIVE_REFERENCE_IDENTITY must appear exactly once"
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None, "invalid RV_NATIVE_REFERENCE_IDENTITY"
        if not isinstance(data, dict):
            return None, "RV_NATIVE_REFERENCE_IDENTITY must be a JSON object"
        try:
            identity = NativeReferenceIdentity.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None, "invalid RV_NATIVE_REFERENCE_IDENTITY"
        return identity.to_dict(), None
    return None, None


def _reference_identity_contract(raw_stderr: str) -> tuple[dict | None, str | None]:
    prefix = "RV_REFERENCE_IDENTITY="
    lines = [line for line in raw_stderr.splitlines() if line.startswith(prefix)]
    if len(lines) > 1:
        return None, "RV_REFERENCE_IDENTITY must appear exactly once"
    if not lines:
        return None, None
    try:
        data = json.loads(lines[0].removeprefix(prefix))
    except json.JSONDecodeError:
        return None, "invalid RV_REFERENCE_IDENTITY"
    if not isinstance(data, dict) or not isinstance(data.get("backend"), str):
        return None, "invalid RV_REFERENCE_IDENTITY"
    commit = data.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        return None, "invalid RV_REFERENCE_IDENTITY.source_commit"
    for name in ("binary_sha256", "identity_digest"):
        if not is_sha256_digest(data.get(name)):
            return None, f"invalid RV_REFERENCE_IDENTITY.{name}"
    return data, None


def _execution_run_observation(
    backend: str,
    record: dict | None,
    binary_hash: str,
    tool_version: str | None,
    profile_id: str | None,
    input_id: str | None,
    *,
    require_executed_pcs: bool = True,
) -> Observation | None:
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("stdout_hex"), str)
        or not isinstance(record.get("stderr"), str)
        or not isinstance(record.get("exit_code"), int)
        or isinstance(record.get("exit_code"), bool)
    ):
        return None
    try:
        stdout = bytes.fromhex(record["stdout_hex"])
    except ValueError:
        return None
    stdout = _observation_frame(stdout) or stdout
    stderr = record["stderr"].encode("utf-8")
    return parse_observation_stdout(
        backend,
        record["exit_code"],
        stdout,
        stderr,
        binary_hash,
        tool_version,
        profile_id,
        input_id,
        require_executed_pcs=require_executed_pcs,
    )


def _observation_frame_gap_reason(stdout: bytes) -> str | None:
    """Explain a RVOBS1 marker that is not a complete tail frame."""
    offset = stdout.find(OBSERVATION_MAGIC)
    if offset < 0:
        return None
    frame_end = _observation_frame_end(stdout, offset)
    if frame_end is None:
        return "truncated RVOBS1 observation header"
    if len(stdout) < frame_end:
        return "truncated RVOBS1 memory snapshot"
    return "trailing bytes after RVOBS1 memory snapshot"


def _attach_native_reference_identity(
    observation: Observation,
    raw_stderr: str,
    binary_hash: str,
) -> Observation:
    contract, contract_error = _native_reference_identity_contract(raw_stderr)
    if contract_error is not None:
        extra_state = dict(observation.extra_state)
        previous_gaps = extra_state.get("observer_gaps")
        observer_gaps = list(previous_gaps) if isinstance(previous_gaps, (list, tuple)) else []
        marker = f"native-reference-identity:{contract_error}"
        if marker not in observer_gaps:
            observer_gaps.append(marker)
        extra_state["observer_gaps"] = observer_gaps
        evidence = observation.translation_evidence
        if evidence is None:
            return replace(observation, extra_state=extra_state)
        details = dict(evidence.details)
        details["native_reference_identity_gap"] = contract_error
        return replace(
            observation,
            extra_state=extra_state,
            translation_evidence=replace(evidence, details=details),
        )
    if contract is None:
        return observation
    details = {
        "native_reference_identity": contract,
        "native_reference_identity_source": LIVE_NATIVE_REFERENCE_IDENTITY_SOURCE,
    }
    if observation.backend == "native-rv64":
        details.update(
            {
                "native_artifact_sha256": binary_hash,
                "native_reference_provenance": LIVE_NATIVE_REFERENCE_PROVENANCE,
                "native_reference_evidence_tier": LIVE_NATIVE_REFERENCE_EVIDENCE_TIER,
            }
        )
    if observation.translation_evidence is None:
        evidence = TranslationEvidence(
            backend=observation.backend,
            expected_path="reference" if observation.backend == "native-rv64" else "either",
            details=details,
        )
    else:
        evidence = replace(
            observation.translation_evidence,
            details={**observation.translation_evidence.details, **details},
        )
    return replace(observation, translation_evidence=evidence)


def _attach_execution_evidence(
    observation: Observation,
    raw_stderr: str,
    binary_hash: str,
    tool_version: str | None,
    profile_id: str | None,
    input_id: str | None,
) -> Observation:
    contract, contract_error = _execution_evidence_contract(raw_stderr)
    if contract_error is not None:
        return _execution_evidence_gap(observation, contract_error)
    if contract is None:
        if observation.backend in {
            "qemu-riscv32", "qemu-riscv64", "libriscv-translated",
        }:
            return _execution_evidence_gap(
                observation, "missing execution evidence", path_gap=True,
            )
        return observation
    fingerprints = tuple(
        str(contract[key])
        for key in (
            "main_command_fingerprint",
            "trace_command_fingerprint",
            "count_command_fingerprint",
        )
    )
    if observation.backend in {"qemu-riscv32", "qemu-riscv64"}:
        if fingerprints[0] != fingerprints[2] or fingerprints[0] == fingerprints[1]:
            return _execution_evidence_gap(
                observation, "execution evidence command roles are incoherent",
                path_gap=True,
            )
    elif observation.backend.startswith("libriscv-") and len(set(fingerprints)) != 3:
        return _execution_evidence_gap(
            observation, "execution evidence command roles are incoherent",
            path_gap=True,
        )
    trace_record = contract.get("trace_run")
    count_record = contract.get("count_run")
    if count_record is None or trace_record is None:
        missing = "count" if count_record is None else "trace"
        return _execution_evidence_gap(
            observation, f"missing {missing} execution evidence",
            path_gap=missing == "trace",
        )
    trace_observation = (
        observation
        if trace_record is None
        else _execution_run_observation(
            observation.backend,
            trace_record,
            binary_hash,
            tool_version,
            profile_id,
            input_id,
            require_executed_pcs=observation.backend != "libriscv-translated",
        )
    )
    if (
        observation.backend.startswith("libriscv-")
        and trace_observation is not None
        and isinstance(trace_record, dict)
        and trace_record.get("exit_code") == 0
        and observation.executed_pcs
        and tuple(trace_observation.executed_pcs) == tuple(observation.executed_pcs)
        and observation.translation_evidence is not None
        and observation.translation_evidence.details.get("trace_failed") is False
        and observation.translation_evidence.details.get("granularity")
        in {"gdb-single-step-pc", "translated-per-pc"}
    ):
        # libriscv 的 trace command 只负责提供 PC path，不重复输出 RVOBS1
        # frame；主 command 已经提供语义状态。两者 PC 序列一致时，复用主
        # observation，避免把合法的 path-only trace 误判为缺 frame。
        trace_observation = observation
    if observation.backend == "libriscv-translated" \
            and trace_observation is not None and trace_observation.contract_error in {
                "stdout-only observation", "trailing bytes after RVOBS1 memory snapshot",
            }:
        details = observation.translation_evidence.details \
                if observation.translation_evidence is not None else {}
        if details.get("trace_failed") is False and isinstance(trace_record, dict) \
                and trace_record.get("exit_code") == 0 and (
            details.get("translated_pcs")
            or details.get("granularity") in {"gdb-single-step-pc", "translated-per-pc"}
        ):
            # The trace run owns the path; the main run owns RVOBS1 state.
            trace_observation = observation
    if (
        observation.backend in {"qemu-riscv32", "qemu-riscv64"}
        and observation.extra_state.get("target_host_abort_observed") is True
        and trace_observation is not None
        and trace_observation.contract_error == "adapter host abort: unhandled-cpu-exception"
        and isinstance(trace_record, dict)
        and trace_record.get("exit_code") == observation.exit_code
    ):
        trace_pcs, _, trace_error = _runner_trace(
            trace_record.get("stderr", ""), observation.backend,
        )
        if trace_error is None and trace_pcs == observation.executed_pcs:
            # QEMU 的独立 trace command 不重复输出 translation evidence；主命令
            # 已经证明完整路径，trace 只复核同一 host-abort 和 PC 序列。
            trace_observation = observation
    host_abort_count_diagnostic = (
        observation.backend in {"qemu-riscv32", "qemu-riscv64"}
        and observation.extra_state.get("target_host_abort_observed") is True
        and isinstance(count_record, dict)
        and count_record.get("exit_code") == observation.exit_code
        and '"host_abort":"unhandled-cpu-exception"' in count_record.get("stderr", "")
    )
    count_observation = (
        observation
        if host_abort_count_diagnostic else
        _execution_run_observation(
            observation.backend,
            count_record,
            binary_hash,
            tool_version,
            profile_id,
            input_id,
            require_executed_pcs=False,
        )
    )
    if (
        observation.backend == "libriscv-translated"
        and isinstance(count_record, dict)
        and count_record.get("exit_code") == 0
        and count_observation is not None
        and count_observation.contract_error == "stdout-only observation"
    ):
        count_observation = observation
    if trace_observation is None or count_observation is None:
        return _execution_evidence_gap(
            observation,
            "malformed execution evidence run",
            path_gap=trace_observation is None,
        )
    if trace_observation.contract_error is not None:
        return _execution_evidence_gap(
            observation, "trace-observation-incoherent", path_gap=True,
        )
    if count_observation.contract_error is not None:
        return _execution_evidence_gap(observation, "count-observation-incoherent")
    bundle = ExecutionEvidenceBundle.create(
        main_observation=observation.to_dict(),
        trace_observation=trace_observation.to_dict(),
        count_observation=count_observation.to_dict(),
        main_command_fingerprint=str(contract["main_command_fingerprint"]),
        trace_command_fingerprint=str(contract["trace_command_fingerprint"]),
        count_command_fingerprint=str(contract["count_command_fingerprint"]),
        count_state_comparable=(
            not observation.backend.startswith("libriscv-")
            and not host_abort_count_diagnostic
        ),
    )
    if not bundle.coherent:
        updated = observation
        if observation.translation_evidence is not None:
            details = dict(observation.translation_evidence.details)
            details["execution_evidence_bundle"] = bundle.to_dict()
            details["execution_evidence_source"] = "runner-contract"
            updated = replace(
                observation,
                translation_evidence=replace(
                    observation.translation_evidence, details=details,
                ),
            )
        return _execution_evidence_gap(
            updated,
            "execution evidence is incoherent",
            path_gap=any(
                failure == "trace-state-mismatch" or failure.endswith("-path-mismatch")
                for failure in bundle.failures
            ),
        )
    if observation.translation_evidence is None:
        if observation.backend in {"qemu-riscv32", "qemu-riscv64", "libriscv-translated"}:
            return _execution_evidence_gap(
                observation, "missing translation evidence", path_gap=True,
            )
        return observation
    details = dict(observation.translation_evidence.details)
    details["execution_evidence_bundle"] = bundle.to_dict()
    details["execution_evidence_source"] = "runner-contract"
    return replace(
        observation,
        translation_evidence=replace(observation.translation_evidence, details=details),
    )


def _parse_observation_stdout(
    backend: str,
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
    binary_hash: str,
    tool_version: str | None,
    profile_id: str | None,
    input_id: str | None,
    *,
    require_executed_pcs: bool = True,
    allow_state_only: bool = False,
    volatile_gpr_indices: tuple[int, ...] = (),
) -> Observation:
    raw_stderr = stderr.decode("utf-8", errors="replace")
    def gap(reason: str, **kwargs: object) -> Observation:
        return _gap_observation(
            backend, reason, raw_stderr, binary_hash, profile_id, input_id, **kwargs
        )

    if type(exit_code) is not int:
        return gap(
            "missing or invalid process exit code",
            raw_stdout=stdout.hex(),
            tool_version=tool_version,
        )
    observation_frame = _observation_frame(stdout)
    has_complete_observation_frame = observation_frame is not None
    observer_gaps: list[str] = []
    executed_pcs, instruction_count, trace_error = _runner_trace(raw_stderr, backend)
    if trace_error is not None:
        executed_pcs, instruction_count = (), None
    translation_evidence = _translation_evidence(backend, raw_stderr)
    runner_gap_error = _runner_contract_gap_error(raw_stderr)
    if runner_gap_error is not None:
        observation = gap(
            runner_gap_error,
            exit_code=exit_code,
            raw_stdout=stdout.hex(),
            executed_pcs=executed_pcs,
            instruction_count=instruction_count,
            translation_evidence=translation_evidence,
            tool_version=tool_version,
        )
        failure_class = _runner_failure_class(raw_stderr)
        if backend == "native-rv64" and failure_class is not None:
            return replace(
                observation,
                extra_state={
                    **dict(observation.extra_state or {}),
                    "runner_failure_class": failure_class,
                },
            )
        return observation
    deferred_translation = (
        backend in {"qemu-riscv32", "qemu-riscv64"}
        and translation_evidence is not None
        and translation_evidence.details.get("source") == "disabled"
        and translation_evidence.details.get("granularity") == "none"
    )

    def traced_gap(reason: str, **kwargs: object) -> Observation:
        context = {
            "exit_code": exit_code,
            "raw_stdout": stdout.hex(),
            "executed_pcs": executed_pcs,
            "instruction_count": instruction_count,
            "translation_evidence": translation_evidence,
            "tool_version": tool_version,
        }
        context.update(kwargs)
        return gap(reason, **context)

    state_only_trace = (
        backend == "rax-riscv64"
        and not executed_pcs
        and translation_evidence is not None
        and translation_evidence.details.get("trace_available") is False
        and bool(translation_evidence.details.get("trace_gap"))
    )
    observation_state, observation_state_error = _runner_observation_state(raw_stderr)
    if observation_state_error is not None:
        if (not has_complete_observation_frame
                or observation_state_error
                == "RV_OBSERVATION_STATE.signal does not match signal_code"):
            return traced_gap(observation_state_error)
        observer_gaps.append(f"observation-state:{observation_state_error}")
        observation_state = {}
    observation_state = _derived_signal_state(exit_code, observation_state or {}, raw_stderr)
    extra_state = dict(observation_state.get("extra_state", {}))

    def note_observer_gap(reason: str) -> None:
        previous_gaps = extra_state.get("observer_gaps")
        gaps = list(previous_gaps) if isinstance(previous_gaps, (list, tuple)) else []
        if reason not in gaps:
            gaps.append(reason)
        extra_state["observer_gaps"] = gaps

    def note_path_evidence_gap(reason: str) -> None:
        nonlocal translation_evidence
        note_observer_gap(f"path:{reason}")
        evidence = translation_evidence
        if evidence is None:
            return
        details = dict(evidence.details)
        details["path_evidence_gap"] = reason
        details["execution_evidence_path_divergence"] = True
        details["trace_failed"] = True
        translation_evidence = replace(evidence, details=details)

    native_fp_state_fields = frozenset({
        "fpr_rawbits", "before_fpr_rawbits", "after_fpr_rawbits",
        "fflags", "before_fflags", "after_fflags", "csr.fflags",
        "frm", "before_frm", "after_frm", "csr.frm",
    })

    def clear_native_fp_evidence() -> None:
        for name in native_fp_state_fields:
            extra_state.pop(name, None)
        declared_fields = extra_state.get("observer_fields")
        if isinstance(declared_fields, (list, tuple)):
            extra_state["observer_fields"] = [
                field for field in declared_fields if field not in native_fp_state_fields
            ]
        observer_fields[:] = [
            field for field in observer_fields if field not in native_fp_state_fields
        ]

    def drop_native_fp_evidence(reason: str) -> None:
        clear_native_fp_evidence()
        note_observer_gap(f"native-checkpoints:{reason}")
        extra_state["fp_observer"] = "not-observed"
        extra_state["fp_observer_gap"] = reason

    if trace_error is not None:
        note_path_evidence_gap(f"runner trace parse: {trace_error}")
    if observer_gaps:
        previous_gaps = extra_state.get("observer_gaps")
        extra_state["observer_gaps"] = [
            *(previous_gaps if isinstance(previous_gaps, (list, tuple)) else ()),
            *observer_gaps,
        ]
    if volatile_gpr_indices:
        extra_state["volatile_gpr_indices"] = list(volatile_gpr_indices)
    if backend == "native-rv64":
        trace_status = [
            line.partition("=")[2].strip()
            for line in raw_stderr.splitlines()
            if line.startswith("NATIVE_RV64_TRACE_STATUS=")
        ]
        if len(trace_status) == 1 and trace_status[0]:
            extra_state["native_trace_status"] = trace_status[0]
    reference_identity, reference_identity_error = _reference_identity_contract(raw_stderr)
    if reference_identity_error is not None:
        if not has_complete_observation_frame:
            return traced_gap(reference_identity_error)
        note_observer_gap(f"reference-identity:{reference_identity_error}")
    if reference_identity is not None:
        extra_state["reference_identity"] = reference_identity
    declared_observer_fields = extra_state.get("observer_fields")
    declared_observer_fields = (
        declared_observer_fields
        if isinstance(declared_observer_fields, (list, tuple))
        else ()
    )
    observer_fields = list(dict.fromkeys(
        field
        for field in (
            *declared_observer_fields,
            *observed_state_fields(extra_state),
        )
        if isinstance(field, str)
    ))
    signal = observation_state.get("signal")
    signal_code = observation_state.get("signal_code")

    def drop_signal_evidence() -> None:
        nonlocal signal, signal_code
        observation_state["signal"] = None
        observation_state["signal_code"] = None
        extra_state.pop("signal", None)
        extra_state.pop("signal_code", None)
        declared_fields = extra_state.get("observer_fields")
        if isinstance(declared_fields, (list, tuple)):
            extra_state["observer_fields"] = [
                field for field in declared_fields
                if field not in {"signal", "signal_code"}
            ]
        signal = signal_code = None

    expected_signal_code = _SIGNAL_CODE_BY_NAME.get(signal)
    if (
        expected_signal_code is not None
        and signal_code not in (None, expected_signal_code)
        or exit_code < 0 and expected_signal_code is not None
        and -exit_code != expected_signal_code
    ):
        return traced_gap("observation signal does not match process signal")
    fault_pc = observation_state.get("fault_pc")
    fault_address = observation_state.get("fault_address")
    if type(fault_pc) is not int and type(extra_state.get("fault_pc")) is int:
        fault_pc = extra_state["fault_pc"]
    if type(fault_address) is not int and type(extra_state.get("fault_address")) is int:
        fault_address = extra_state["fault_address"]
    native_stop_claimed = (
        "target_stop_observed" in extra_state
        or "target_stop_observer" in extra_state
    )
    native_target_stop_observed = False
    if native_stop_claimed:
        expected_code = _SIGNAL_CODE_BY_NAME.get(signal)
        native_target_stop_observed = (
            backend == "native-rv64"
            and extra_state.get("target_stop_observed") is True
            and extra_state.get("target_stop_observer")
            == "native-ptrace-terminal-signal"
            and isinstance(signal, str)
            and type(signal_code) is int
            and signal_code > 0
            and expected_code == signal_code
            and type(fault_pc) is int
            and fault_pc >= 0
            and not fault_pc & 1
            and fault_pc in executed_pcs
            and exit_code != 0
            and exit_code == 128 + signal_code
        )
        if not native_target_stop_observed:
            return traced_gap(
                "native ptrace terminal-signal evidence is incomplete or inconsistent",
            )
    if backend == "qemu-riscv32":
        xlen_limit = 1 << 32
        invalid_fault_pc = type(fault_pc) is int and fault_pc >= xlen_limit
        invalid_fault_address = type(fault_address) is int and fault_address >= xlen_limit
        invalid_trap_aliases = {
            alias
            for name in ("trap.cause", "trap.epc", "trap.tval")
            for alias in _TRAP_FIELD_ALIASES[name]
            if type(extra_state.get(alias)) is int
            and extra_state[alias] >= xlen_limit
        }
        if invalid_fault_pc or invalid_fault_address or invalid_trap_aliases:
            note_observer_gap("trap-or-fault:RV32 field exceeds XLEN")
            if invalid_fault_pc:
                fault_pc = None
                extra_state.pop("fault_pc", None)
            if invalid_fault_address:
                fault_address = None
                extra_state.pop("fault_address", None)
            invalid_trap_identity = bool(
                invalid_trap_aliases.intersection(
                    (*_TRAP_FIELD_ALIASES["trap.cause"], *_TRAP_FIELD_ALIASES["trap.epc"])
                )
            )
            for alias in invalid_trap_aliases:
                extra_state.pop(alias, None)
            if invalid_trap_identity:
                note_observer_gap("trap:RV32 cause or EPC exceeds XLEN")
    adapter_execution_error = extra_state.get("adapter_execution_error")
    guest_trap_observed = extra_state.get("guest_trap") == "delivered"
    terminal_ebreak = _terminal_ebreak_observed(extra_state)
    trap_state = canonical_trap_state(extra_state)
    structured_target_signal = (
        extra_state.get("qemu_uncaught_target_signal") is True
        and isinstance(signal, str)
        and bool(signal.strip())
        and type(signal_code) is int
    ) or native_target_stop_observed
    evidence_details = (
        translation_evidence.details
        if translation_evidence is not None else {}
    )
    path_identity = evidence_details.get("path_identity")
    host_abort_path_observed = (
        backend in {"qemu-riscv32", "qemu-riscv64"}
        and extra_state.get("host_abort") == "unhandled-cpu-exception"
        and evidence_details.get("host_abort_path_observed") is True
        and evidence_details.get("trace_failed") is False
        and isinstance(path_identity, dict)
        and path_identity.get("status") == "observed"
        and bool(executed_pcs)
    )
    try:
        qemu_signal_si_addr = _trace_int(extra_state["signal.si_addr"])
    except (KeyError, TypeError, ValueError):
        qemu_signal_si_addr = None
    qemu_guest_signal_abort_observed = (
        backend in {"qemu-riscv32", "qemu-riscv64"}
        and extra_state.get("host_abort_observer") == "qemu-trace-process"
        and extra_state.get("host_abort") == f"qemu-process-signal:{signal}"
        and structured_target_signal
        and guest_trap_observed
        and signal_code == _SIGNAL_CODE_BY_NAME.get(signal)
        and type(trap_state.get("trap.cause")) is int
        and type(trap_state.get("trap.epc")) is int
        and trap_state["trap.epc"] >= 0
        and not trap_state["trap.epc"] & 1
        and fault_pc in (None, trap_state["trap.epc"])
        and (
            (
                signal == "SIGILL"
                and trap_state["trap.cause"] == 2
                and type(extra_state.get("signal.si_code")) is int
                and (
                    extra_state["signal.si_code"] == 1
                    and extra_state.get("signal.si_code_name") == "ILL_ILLOPC"
                    or extra_state["signal.si_code"] == 2
                    and extra_state.get("signal.si_code_name") == "ILL_ILLOPN"
                )
                and extra_state.get("signal.si_code_observer") == "qemu-strace"
                and extra_state.get("signal.si_code_name_observer") == "qemu-strace"
                and extra_state.get("signal.si_addr_observer") == "qemu-strace"
                and qemu_signal_si_addr == trap_state["trap.epc"]
                and extra_state.get("expected_trap_process_signal")
                == extra_state.get("host_abort")
            )
            or (
                signal == "SIGSEGV"
                and trap_state["trap.cause"] in {13, 15}
                and type(fault_address) is int
                and type(extra_state.get("signal.si_code")) is int
                and (
                    extra_state["signal.si_code"] == 1
                    and extra_state.get("signal.si_code_name") == "SEGV_MAPERR"
                    or extra_state["signal.si_code"] == 2
                    and extra_state.get("signal.si_code_name") == "SEGV_ACCERR"
                )
                and extra_state.get("signal.si_code_observer") == "qemu-strace"
                and extra_state.get("signal.si_code_name_observer") == "qemu-strace"
                and extra_state.get("signal.si_addr_observer") == "qemu-strace"
                and qemu_signal_si_addr == fault_address
                and trap_state.get("trap.tval") == fault_address
            )
            or (
                signal == "SIGBUS"
                and trap_state["trap.cause"] in {4, 6}
                and type(fault_address) is int
                and type(extra_state.get("signal.si_code")) is int
                and extra_state.get("signal.si_code") == 1
                and extra_state.get("signal.si_code_name") == "BUS_ADRALN"
                and extra_state.get("signal.si_code_observer") == "qemu-strace"
                and extra_state.get("signal.si_code_name_observer") == "qemu-strace"
                and extra_state.get("signal.si_addr_observer") == "qemu-strace"
                and qemu_signal_si_addr == fault_address
                and trap_state.get("trap.tval") == fault_address
            )
        )
    )
    if extra_state.get("host_abort") is not None and not (
        terminal_ebreak
        and extra_state["host_abort"] == "qemu-trace-signal:SIGTRAP"
        or host_abort_path_observed
        or qemu_guest_signal_abort_observed
    ):
        if has_complete_observation_frame:
            note_observer_gap(f"host-abort:{extra_state['host_abort']}")
            extra_state.pop("host_abort", None)
        else:
            return traced_gap(f"adapter host abort: {extra_state['host_abort']}")
    if host_abort_path_observed:
        extra_state["target_host_abort_observed"] = True
        extra_state["target_host_abort_observer"] = "qemu-trace-process"
    if qemu_guest_signal_abort_observed:
        fault_pc = trap_state["trap.epc"]
    if guest_trap_observed and any(
        type(trap_state.get(field)) is not int
        for field in ("trap.cause", "trap.epc")
    ):
        note_observer_gap("trap:incomplete-trap-observation")
    state_frame_without_trace = has_complete_observation_frame and (
        backend == "native-rv64"
        or backend == "rvvm-riscv64"
        and evidence_details.get("observer_complete") is True
        and evidence_details.get("mailbox_status") == "ready"
    )
    missing_executed_pcs = require_executed_pcs and exit_code == 0 and not executed_pcs and (
        backend == "native-rv64" or is_execution_backend(backend)
    ) and not deferred_translation and not state_only_trace and not state_frame_without_trace
    observation_frame_gap = (
        None
        if has_complete_observation_frame
        else _observation_frame_gap_reason(stdout)
    )
    structured_guest_exception = (
        exit_code == 0
        and guest_trap_observed
        and not terminal_ebreak
        and not has_complete_observation_frame
    )
    if extra_state.get("qemu_uncaught_target_signal") is True and not structured_target_signal:
        if has_complete_observation_frame:
            note_observer_gap("signal:qemu-target-signal-evidence-incomplete")
            extra_state.pop("qemu_uncaught_target_signal", None)
            drop_signal_evidence()
        else:
            return traced_gap("qemu target signal evidence is incomplete")
    rax_tohost_failure = (
        backend == "rax-riscv64"
        and extra_state.get("guest_exit_channel") == "rax-vcpu-tohost"
        and extra_state.get("guest_status") == "failure"
        and extra_state.get("terminal_observed") is True
        and type(extra_state.get("tohost_value")) is int
        and extra_state.get("tohost_value") != 1
    )
    guest_exit_observed = extra_state.get("guest_exit_observer") == "qemu-linux-user"
    if missing_executed_pcs:
        note_path_evidence_gap("missing RV_EXECUTED_PCS")
    if (
        observation_frame_gap is not None
        and exit_code != 0
        and not guest_trap_observed
        and not structured_target_signal
        and not host_abort_path_observed
        and not guest_exit_observed
    ):
        return traced_gap(observation_frame_gap)
    if exit_code == 0 and signal is not None and not (
        guest_trap_observed
        or structured_target_signal
        or guest_exit_observed
        or adapter_execution_error
    ):
        return traced_gap("successful exit carries an unobserved process signal")
    # FP readback is optional unless the route declared FP fields.  Preserve a
    # complete RVOBS1 frame and let the semantic projection report a missing FP
    # field when it is relevant.
    optional_observer_error = isinstance(adapter_execution_error, str) and (
        adapter_execution_error.startswith("final-fp-state-unavailable:")
        or "fp-observer" in adapter_execution_error
        or "rax-uart-no-csr-readback" in adapter_execution_error
    )
    auxiliary_path_error = (
        isinstance(adapter_execution_error, str)
        and evidence_details.get("trace_gap") == adapter_execution_error
    )
    if adapter_execution_error and not optional_observer_error and not (
        structured_guest_exception or guest_trap_observed
        or structured_target_signal
    ):
        if has_complete_observation_frame and auxiliary_path_error:
            note_path_evidence_gap(adapter_execution_error)
        else:
            return traced_gap(f"adapter execution failed: {adapter_execution_error}")
    if adapter_execution_error and optional_observer_error and has_complete_observation_frame:
        note_observer_gap(f"adapter:{adapter_execution_error}")
    if backend.startswith("libriscv-") and (
        "libriscv_machine_exception_code" in extra_state
        and extra_state.get("guest_trap") != "delivered"
    ):
        return traced_gap(
            "libriscv reported a non-guest machine exception",
        )
    if backend.startswith("libriscv-") and not guest_trap_observed and (
        b">>> Exception:" in stdout
        or ">>> Exception:" in raw_stderr
        or re.search(rb"(?m)^\s*Exception:", stdout)
        or re.search(r"(?m)^\s*Exception:", raw_stderr)
    ):
        return traced_gap(
            "libriscv reported a runtime exception",
        )
    if backend == "native-rv64":
        # Native FP state is admissible only when this parser validated the
        # complete checkpoint chain.  Runner metadata may contain a fallback
        # FP snapshot, but that snapshot is not interchangeable with a chain.
        clear_native_fp_evidence()
        checkpoints, checkpoint_error = _runner_window_checkpoints(raw_stderr)
        native_trace_status = extra_state.get("native_trace_status")
        fp_checkpoint_error = None
        if not isinstance(native_trace_status, str) or not native_trace_status.startswith(
            "observed-runner-trace"
        ):
            fp_checkpoint_error = "native trace is incomplete"
            checkpoints = None
        elif checkpoints is None:
            fp_checkpoint_error = checkpoint_error or "native window checkpoints are missing"
        else:
            checkpoint_pcs = tuple(_trace_int(item["pc"]) for item in checkpoints)
            if trace_error is not None:
                fp_checkpoint_error = "executed PC trace is invalid"
                checkpoints = None
            elif checkpoint_pcs != executed_pcs:
                note_path_evidence_gap("native-checkpoints:PC sequence does not match executed PCs")
                fp_checkpoint_error = "PC sequence does not match executed PCs"
                checkpoints = None
        if fp_checkpoint_error is not None:
            drop_native_fp_evidence(fp_checkpoint_error)
        if checkpoints:
            complete_fp_chain = all(
                all(checkpoint.get(name) is not None for name in (
                    "before_fpr_rawbits", "after_fpr_rawbits",
                    "before_fflags", "after_fflags",
                    "before_frm", "after_frm",
                ))
                for checkpoint in checkpoints
            )
            if not complete_fp_chain:
                drop_native_fp_evidence("checkpoint FP state is incomplete")
            else:
                final = checkpoints[-1]
                extra_state.update(
                    {
                        "fpr_rawbits": [
                            _trace_int(value) for value in final["after_fpr_rawbits"]
                        ],
                        "fflags": _trace_int(final["after_fflags"]),
                        "frm": _trace_int(final["after_frm"]),
                        "fp_observer": "observed",
                    }
                )
                extra_state.pop("fp_observer_gap", None)
                fp_observer_fields = ("fpr_rawbits", "fflags", "frm")
                declared_fields = extra_state.get("observer_fields")
                declared_fields = (
                    list(declared_fields)
                    if isinstance(declared_fields, (list, tuple)) else []
                )
                extra_state["observer_fields"] = list(dict.fromkeys(
                    (*declared_fields, *fp_observer_fields)
                ))
                observer_fields[:] = list(dict.fromkeys(
                    (*observer_fields, *fp_observer_fields)
                ))
    translation_error = _translation_contract_error(raw_stderr, backend)
    if translation_error is not None:
        translation_evidence = None
        note_observer_gap(f"translation-evidence:{translation_error}")
    if exit_code != 0:
        if (
            backend.startswith(("qemu-", "libriscv-"))
            and not guest_trap_observed
            and not structured_target_signal
            and not host_abort_path_observed
            and not guest_exit_observed
            and not has_complete_observation_frame
        ):
            return replace(
                traced_gap(
                    "adapter nonzero exit lacks structured guest termination evidence",
                ),
                signal=observation_state.get("signal"),
                signal_code=observation_state.get("signal_code"),
            )
        if backend in CAPSULE_BACKENDS and not (
            guest_trap_observed or rax_tohost_failure or has_complete_observation_frame
        ):
            return replace(
                traced_gap(
                    "adapter nonzero exit lacks guest-trap or adapter signal evidence",
                    extra_state={**extra_state, "observer_fields": observer_fields},
                ),
                signal=observation_state.get("signal"),
                signal_code=observation_state.get("signal_code"),
                fault_pc=fault_pc,
                fault_address=fault_address,
            )
        if not has_complete_observation_frame:
            observation = Observation(
                backend=backend,
                outcome=(
                    "trap"
                    if (guest_trap_observed and not terminal_ebreak)
                    or native_target_stop_observed
                    else "nonzero-exit"
                ),
                exit_code=exit_code,
                checkpoint_pc=None,
                gpr=(None,) * 32,
                memory_delta={},
                memory_digest=None,
                signal=observation_state.get("signal"),
                signal_code=observation_state.get("signal_code"),
                fault_pc=fault_pc,
                fault_address=fault_address,
                executed_pcs=executed_pcs,
                instruction_count=instruction_count,
                translation_evidence=translation_evidence,
                profile_id=profile_id,
                input_id=input_id,
                raw_stdout=stdout.hex(),
                raw_stderr=raw_stderr,
                binary_sha256=binary_hash,
                tool_version=tool_version,
                extra_state=extra_state,
                contract_error=None,
            )
            observation = _attach_native_reference_identity(
                observation,
                raw_stderr,
                binary_hash,
            )
            return _attach_execution_evidence(
                observation,
                raw_stderr,
                binary_hash,
                tool_version,
                profile_id,
                input_id,
            )
    # Use the validated frame even when it is followed by diagnostics.  The
    # shared locator deliberately accepts prefix/suffix bytes; retaining the
    # whole stdout here would turn a valid frame-at-start-plus-trace stream
    # into a false "trailing bytes" gap.
    observation_stdout = b"" if structured_guest_exception else (
        observation_frame or stdout
    )
    if not observation_stdout.startswith(OBSERVATION_MAGIC):
        if structured_guest_exception:
            observation = replace(
                traced_gap("guest exception observation"),
                outcome="trap",
                signal=observation_state.get("signal"),
                signal_code=observation_state.get("signal_code"),
                fault_pc=fault_pc,
                fault_address=fault_address,
                extra_state={
                    **extra_state,
                    "observer_fields": [*observer_fields, "guest_exception"],
                },
                contract_error=None,
            )
            observation = _attach_native_reference_identity(
                observation, raw_stderr, binary_hash,
            )
            return _attach_execution_evidence(
                observation, raw_stderr, binary_hash, tool_version, profile_id, input_id,
            )
        if (
            allow_state_only
            and backend == "rax-riscv64"
            and extra_state.get("state_only_observation") is True
            and exit_code == 0
            and executed_pcs
            and observation_frame_gap is None
        ):
            observation = Observation(
                backend=backend,
                outcome="normal",
                exit_code=exit_code,
                checkpoint_pc=None,
                gpr=(None,) * 32,
                memory_delta={},
                memory_digest=None,
                signal=observation_state.get("signal"),
                signal_code=observation_state.get("signal_code"),
                fault_pc=fault_pc,
                fault_address=fault_address,
                executed_pcs=executed_pcs,
                instruction_count=instruction_count,
                translation_evidence=translation_evidence,
                profile_id=profile_id,
                input_id=input_id,
                raw_stdout=stdout.hex(),
                raw_stderr=raw_stderr,
                binary_sha256=binary_hash,
                tool_version=tool_version,
                extra_state={
                    **extra_state,
                    "state_only_observation": True,
                    "observer_fields": [*observer_fields, "executed_pcs", "outcome"],
                },
                contract_error=None,
            )
            return _attach_native_reference_identity(observation, raw_stderr, binary_hash)
        if exit_code == 0 and backend in {
            "native-rv64",
            "qemu-riscv32",
            "qemu-riscv64",
            "libriscv-translated",
        }:
            observation = Observation(
                backend=backend,
                outcome="normal",
                exit_code=0,
                checkpoint_pc=None,
                gpr=(None,) * 32,
                memory_delta={},
                memory_digest=None,
                signal=None,
                signal_code=None,
                fault_pc=None,
                fault_address=None,
                executed_pcs=executed_pcs,
                instruction_count=instruction_count,
                translation_evidence=translation_evidence,
                profile_id=profile_id,
                input_id=input_id,
                raw_stdout=stdout.hex(),
                raw_stderr=raw_stderr,
                binary_sha256=binary_hash,
                tool_version=tool_version,
                extra_state={
                    **extra_state,
                    "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "stdout_observer": "process-stdout",
                    "observer_fields": list(dict.fromkeys((
                        *observer_fields, "stdout", "outcome",
                    ))),
                },
                contract_error=None,
            )
            observation = _attach_native_reference_identity(
                observation, raw_stderr, binary_hash,
            )
            return _attach_execution_evidence(
                observation, raw_stderr, binary_hash, tool_version, profile_id, input_id,
            )
        if backend in {
            "native-rv64",
            "qemu-riscv32",
            "qemu-riscv64",
            "libriscv-translated",
        }:
            observation = traced_gap(
                "stdout-only observation",
                extra_state={
                    **extra_state,
                    "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "stdout_observer": "process-stdout",
                    "observer_fields": [*observer_fields, "stdout"],
                },
            )
            observation = _attach_native_reference_identity(
                observation, raw_stderr, binary_hash,
            )
            return _attach_execution_evidence(
                observation, raw_stderr, binary_hash, tool_version, profile_id, input_id,
            )
        return traced_gap(
            (
                "expected-trap-observation-unsupported: ebreak"
                if b"machine exception" in stdout.lower()
                and b"ebreak" in stdout.lower()
                else "missing RVOBS1 observation blob"
            ),
        )
    if len(observation_stdout) < OBSERVATION_HEADER_SIZE:
        return traced_gap("truncated RVOBS1 observation header")
    checkpoint_pc = struct.unpack_from("<Q", observation_stdout, 8)[0]
    if checkpoint_pc == 0 or checkpoint_pc & 1:
        return traced_gap(
            "unfinalized RVOBS1 checkpoint" if checkpoint_pc == 0 else "unaligned RVOBS1 checkpoint",
        )
    if backend == "qemu-riscv32" and checkpoint_pc >= 1 << 32:
        return traced_gap("RV32 RVOBS1 checkpoint exceeds XLEN")
    gpr = tuple(struct.unpack_from("<Q", observation_stdout, 16 + index * 8)[0] for index in range(32))
    if backend == "qemu-riscv32" and any(value > 0xFFFFFFFF for value in gpr):
        return traced_gap("RV32 RVOBS1 GPR exceeds XLEN")
    memory_size = struct.unpack_from("<Q", observation_stdout, 16 + 32 * 8)[0]
    memory_start = OBSERVATION_HEADER_SIZE
    memory_end = memory_start + memory_size
    if len(observation_stdout) < memory_end:
        return traced_gap(
            "truncated RVOBS1 memory snapshot",
        )
    if len(observation_stdout) != memory_end:
        return traced_gap(
            "trailing bytes after RVOBS1 memory snapshot",
        )
    memory = observation_stdout[memory_start:memory_end]
    memory_delta = {"test-memory": memory.hex()}
    memory_digest = hashlib.sha256(memory).hexdigest()
    if backend == "native-rv64" and checkpoints:
        final = checkpoints[-1]
        final_gpr = tuple(_trace_int(value) for value in final["after_gpr"])
        # ptrace trace is a separate run; only combine its FP state with RVOBS1
        # when final memory and architectural GPR state agree.
        trace_state_mismatch = not native_trace_stdout_mismatch_is_compatible(extra_state)
        if _trace_int(final["after_pc"]) != checkpoint_pc:
            drop_native_fp_evidence("final PC does not match RVOBS1")
            checkpoints = None
        elif trace_state_mismatch:
            drop_native_fp_evidence("trace-run final state differs from RVOBS1")
            checkpoints = None
        elif any(
            index not in NATIVE_OBSERVER_TEMP_GPRS
            and final_gpr[index] != gpr[index]
            for index in range(len(gpr))
        ):
            drop_native_fp_evidence("final GPR state does not match RVOBS1")
            checkpoints = None
    checkpoint_reached = checkpoint_pc in executed_pcs or (
        backend == "native-rv64"
        and checkpoints
        and _trace_int(checkpoints[-1]["after_pc"]) == checkpoint_pc
    )
    if executed_pcs and not checkpoint_reached:
        note_path_evidence_gap("RVOBS1 checkpoint is absent from execution trace")
    if gpr[0] != 0:
        return traced_gap(
            "invalid RVOBS1 x0",
        )
    observation = Observation(
        backend=backend,
        outcome="trap" if (
            extra_state.get("target_stop_observed") is True
            or (
                guest_trap_observed
                and not terminal_ebreak
            )
        ) else "normal" if exit_code == 0 else "nonzero-exit",
        exit_code=exit_code,
        checkpoint_pc=checkpoint_pc,
        gpr=gpr,
        memory_delta=memory_delta,
        memory_digest=memory_digest,
        signal=observation_state.get("signal"),
        signal_code=observation_state.get("signal_code"),
        fault_pc=fault_pc,
        fault_address=fault_address,
        executed_pcs=executed_pcs,
        instruction_count=instruction_count,
        translation_evidence=translation_evidence,
        profile_id=profile_id,
        input_id=input_id,
        raw_stdout=stdout.hex(),
        raw_stderr=raw_stderr,
        binary_sha256=binary_hash,
        tool_version=tool_version,
        extra_state=extra_state,
        contract_error=None,
    )
    observation = _attach_native_reference_identity(
        observation,
        raw_stderr,
        binary_hash,
    )
    return _attach_execution_evidence(
        observation,
        raw_stderr,
        binary_hash,
        tool_version,
        profile_id,
        input_id,
    )


def parse_observation_stdout(
    backend: str,
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
    binary_hash: str,
    tool_version: str | None,
    profile_id: str | None = None,
    input_id: str | None = None,
    *,
    require_executed_pcs: bool = True,
    allow_state_only: bool = False,
    volatile_gpr_indices: tuple[int, ...] = (),
) -> Observation:
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    safe_hash = binary_hash if is_sha256_digest(binary_hash) else None
    safe_profile = profile_id if isinstance(profile_id, str) and profile_id.strip() else None
    safe_input = input_id if isinstance(input_id, str) and input_id.strip() else None
    safe_version = tool_version if isinstance(tool_version, str) and tool_version.strip() else None
    if not isinstance(stdout, (bytes, bytearray, memoryview)):
        return _gap_observation(
            backend, "stdout must be bytes", "", safe_hash, safe_profile, safe_input,
            raw_stdout="", tool_version=safe_version,
        )
    if not isinstance(stderr, (bytes, bytearray, memoryview)):
        return _gap_observation(
            backend, "stderr must be bytes", "", safe_hash, safe_profile, safe_input,
            raw_stdout=bytes(stdout).hex(), tool_version=safe_version,
        )
    stdout = bytes(stdout)
    stderr = bytes(stderr)
    try:
        return _parse_observation_stdout(
            backend, exit_code, stdout, stderr, safe_hash, safe_version,
            safe_profile, safe_input, require_executed_pcs=require_executed_pcs,
            allow_state_only=allow_state_only,
            volatile_gpr_indices=volatile_gpr_indices,
        )
    except (AttributeError, IndexError, KeyError, OverflowError, struct.error, TypeError, ValueError) as exc:
        return _gap_observation(
            backend,
            f"invalid observation payload: {exc}",
            stderr.decode("utf-8", errors="replace"),
            safe_hash,
            safe_profile,
            safe_input,
            raw_stdout=stdout.hex(),
            tool_version=safe_version,
        )


def run_backend(
    backend: str,
    elf_path: Path,
    profile_id: str | None = None,
    input_id: str | None = None,
    runner_env: dict[str, str] | None = None,
    backend_binary_path: str | None = None,
    timeout_seconds: float | None = None,
) -> list[Observation]:
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    if runner_env is not None and not isinstance(runner_env, dict):
        raise ValueError("runner_env must be an object")
    deadline = None
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be non-negative")
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("timeout_seconds must be non-negative") from error
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + timeout_seconds
    require_execution_plane()
    elf_path = elf_path.resolve()
    binary_hash = _binary_hash(elf_path)
    def gap(reason: str, stderr: str = "", **kwargs: object) -> Observation:
        return _gap_observation(
            backend, reason, stderr, binary_hash, profile_id, input_id, **kwargs
        )

    def gaps(reason: str, **kwargs: object) -> list[Observation]:
        return [gap(reason, **kwargs)]

    command = command_for_backend(backend, elf_path, backend_binary_path=backend_binary_path)
    executable = command[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        return gaps(f"missing executable: {executable}")

    version = _tool_version(command)
    observations: list[Observation] = []
    runner_cwd = str(REPOSITORY_ROOT if backend == "native-rv64" else elf_path.parent)
    child_env = os.environ.copy()
    supplied_env = {
        str(key): str(value) for key, value in (runner_env or {}).items()
    }
    if backend == "rax-riscv64":
        # RAX 的 trace feature 会输出 guest PC；没有源码覆盖率 raw dir 时也保留完整 case trace。
        supplied_env.setdefault(
            "RAX_TRACE_PATH",
            str(elf_path.parent / "rax-%p.trace"),
        )
    volatile_gpr_indices = tuple(sorted({
        int(value.strip())
        for value in supplied_env.get("RV_VOLATILE_GPR_INDICES", "").split(",")
        if value.strip().isdigit() and int(value.strip()) < 32
    }))
    configured_profile = (
        supplied_env.get("RV_TESTCASE_ISA_PROFILE_NAME")
        or supplied_env.get("RV_TESTCASE_ISA_PROFILE")
    )
    if (
        supplied_env.get("RV_TESTCASE_ISA_PROFILE_NAME") is not None
        and supplied_env.get("RV_TESTCASE_ISA_PROFILE") is not None
        and supplied_env["RV_TESTCASE_ISA_PROFILE_NAME"].strip().lower()
        != supplied_env["RV_TESTCASE_ISA_PROFILE"].strip().lower()
    ):
        return gaps("runner ISA profile environment variables conflict")
    capsule_memory_size = None
    if backend in CAPSULE_BACKENDS and CAPSULE_MEMORY_SIZE_ENV in supplied_env:
        try:
            capsule_memory_size = int(supplied_env[CAPSULE_MEMORY_SIZE_ENV], 0)
        except ValueError:
            capsule_memory_size = -1
        if capsule_memory_size < 0:
            return gaps("invalid capsule memory size configuration")
    if (
        isinstance(profile_id, str)
        and is_canonical_isa_profile(profile_id.strip().lower())
        and isinstance(configured_profile, str)
        and configured_profile.strip().lower() != profile_id.strip().lower()
    ):
        return gaps("testcase ISA profile does not match runner configuration")
    if backend != "native-rv64" and (reason := backend_capability_gap(
        backend,
        binary_path=backend_binary_path,
        runner_env=supplied_env,
    )):
        return gaps(f"backend-capability-gap: {reason}")
    allow_state_only = supplied_env.get("RV_TESTCASE_ALLOW_STATE_ONLY") == "1"
    child_env.update(supplied_env)
    run_root = child_env.get("RQ1_RUN")
    if (
        "TMPDIR" not in supplied_env
        and isinstance(run_root, str)
        and run_root
        and Path(run_root).is_dir()
    ):
        # Formal containers mount the run directory read-write but keep the
        # image root read-only; qemu/libriscv helper scripts inherit this path.
        child_env["TMPDIR"] = run_root
    for name in (
        "RV_QEMU_CPU", "RV_QEMU_STATE_TRACE", "RV_TESTCASE_ISA_PROFILE",
        "RV_TESTCASE_ISA_PROFILE_NAME",
        "RV_TESTCASE_REQUIRE_FINAL_FP_STATE", "RV_TESTCASE_ALLOW_STATE_ONLY", "RV_VOLATILE_GPR_INDICES",
        "RV_TESTCASE_EXPECTED_TRAP",
        "RV_TESTCASE_RISK_PCS",
        "GCOV_PREFIX", "GCOV_PREFIX_STRIP", "LLVM_PROFILE_FILE", "RAX_TRACE_PATH",
        "RQ1_DOTNET_COVERAGE_LAUNCHER", "RQ1_DOTNET_COVERAGE_OUTPUT",
        "RQ1_DOTNET_COVERAGE_INCLUDE_FILES", "RQ1_DOTNET_COVERAGE_SESSION_ID",
        "RQ1_TARGET_LOGICAL_BINARY_PATH",
        "RQ1_BACKEND_DEADLINE_MONOTONIC",
        "RVVM_BINARY", "RVVM_SOURCE", "RVVM_HARTS", "RVVM_NO_JIT",
        "RVVM_BARE_METAL_CAPSULE", "RQ1_RVVM_STOP_FLUSH", "RQ1_RVVM_TRACE_PATH",
        LINUX_TEST_MEMORY_ENV, LINUX_MEMORY_SIZE_ENV,
        CAPSULE_MAILBOX_ENV, CAPSULE_OBSERVATION_SIZE_ENV,
        CAPSULE_TEST_MEMORY_ENV, CAPSULE_MEMORY_SIZE_ENV,
    ):
        if name not in supplied_env:
            child_env.pop(name, None)
    # ``profile_id`` identifies the observation/profile record, not the ISA
    # configuration.  Formal campaign callers provide the canonical ISA as
    # ``RV_TESTCASE_ISA_PROFILE_NAME``; legacy adapter callers may still pass
    # the ISA itself in ``profile_id``.  Never leak a value such as
    # ``case/native-profile`` into adapter configuration.
    if "RV_TESTCASE_ISA_PROFILE" not in supplied_env:
        canonical_profile = supplied_env.get("RV_TESTCASE_ISA_PROFILE_NAME")
        if canonical_profile:
            child_env["RV_TESTCASE_ISA_PROFILE"] = canonical_profile.strip().lower()
        elif (
            isinstance(profile_id, str)
            and is_canonical_isa_profile(profile_id.strip().lower())
        ):
            child_env["RV_TESTCASE_ISA_PROFILE"] = profile_id.strip().lower()
        else:
            child_env.pop("RV_TESTCASE_ISA_PROFILE", None)
    if is_execution_backend(backend):
        child_env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(REPOSITORY_ROOT), child_env.get("PYTHONPATH")) if value
        )
    start = time.monotonic()
    process_timeout: float | None = _BACKEND_TIMEOUT_SEC.get(backend, DEFAULT_TIMEOUT_SEC)
    lock_deadline = start + process_timeout
    native_timeout_diagnostics: list[str] = []
    if backend == "native-rv64":
        # K1 owns its guest timeout; a second Direct-level default wall limit
        # would truncate a complete ptrace replay. An explicit campaign budget
        # still bounds the request and triggers remote trace cancellation.
        process_timeout = None if deadline is None else deadline - start
        try:
            guest_timeout = int(child_env.get("RQ1_NATIVE_GUEST_TIMEOUT", "180"))
            remote_timeout = int(child_env.get(
                "RQ1_NATIVE_REMOTE_TIMEOUT", str(guest_timeout * 2 + 30),
            ))
        except ValueError:
            guest_timeout, remote_timeout = 180, 390
        lock_deadline = start + max(1, remote_timeout * 3 + 150)
    if deadline is not None:
        remaining = deadline - start
        if remaining <= 0:
            observations.append(_timeout_observation(
                backend, "", binary_hash, version, profile_id, input_id,
                raw_stdout="",
            ))
            return _attach_source_pc_map(backend, elf_path, observations)
        process_timeout = remaining if process_timeout is None else min(process_timeout, remaining)
    process_deadline = start + process_timeout if process_timeout is not None else None
    try:
        lock = (
            _NativeReferenceLock(process_deadline or lock_deadline)
            if backend == "native-rv64" else nullcontext()
        )
        with lock:
            remaining = None if process_deadline is None else process_deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                if backend == "native-rv64":
                    raise TimeoutError("native-reference-lock-timeout")
                raise subprocess.TimeoutExpired(list(command), process_timeout)
            process_env = dict(child_env)
            timeout_callback = None
            if backend == "native-rv64":
                # The parent owns the shared K1 slot while the child runs.
                process_env["RQ1_NATIVE_REFERENCE_LOCK_HELD"] = "1"
                if not (
                    process_env.get("RQ1_NATIVE_REMOTE_DIR_MARKER")
                    and process_env.get("RQ1_NATIVE_REMOTE_TRACE_PID_MARKER")
                ):
                    marker_base = (
                        FRAMEWORK_RUNS_ROOT / "native-reference-markers"
                        / f"k1-{os.getpid()}-{uuid.uuid4().hex}"
                    )
                    try:
                        marker_base.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as error:
                        observations.append(gap(
                            "native-reference-cancel-marker-unavailable",
                            str(error), tool_version=version,
                            extra_state={"runner_failure_class": "k1-transport-unavailable"},
                        ))
                        return _attach_source_pc_map(backend, elf_path, observations)
                    process_env["RQ1_NATIVE_REMOTE_DIR_MARKER"] = str(marker_base) + ".remote-dir"
                    process_env["RQ1_NATIVE_REMOTE_TRACE_PID_MARKER"] = str(marker_base) + ".trace-pid"

                def cancel_native_trace() -> None:
                    try:
                        from framework.native.rv64_runner import cancel_active_remote_trace

                        result = cancel_active_remote_trace(
                            process_env, marker_wait_seconds=0.5,
                        )
                    except Exception:
                        result = None
                    if result is not None:
                        process_group, succeeded = result
                        native_timeout_diagnostics.append(
                            f"NATIVE_RV64_TRACE_CANCEL_"
                            f"{'SUCCEEDED' if succeeded else 'GAP'}={process_group}"
                        )
                    elif Path(process_env.get("RQ1_NATIVE_REMOTE_TRACE_PID_MARKER", "")).is_file():
                        native_timeout_diagnostics.append(
                            "NATIVE_RV64_TRACE_CANCEL_GAP=cancel-marker-unavailable"
                        )
                    marker = process_env.get("RQ1_NATIVE_REMOTE_DIR_MARKER")
                    if marker:
                        try:
                            remote_dir = Path(marker).read_text(encoding="utf-8").strip()
                        except OSError:
                            remote_dir = ""
                        if remote_dir:
                            native_timeout_diagnostics.append(
                                f"NATIVE_RV64_PRESERVED_REMOTE_DIR={remote_dir}"
                            )

                timeout_callback = cancel_native_trace
            if backend.startswith(("libriscv-", "qemu-", "renode-")):
                # The trace helper's phase timeout must not be shorter than
                # the enclosing target attempt.  The parent owns the deadline.
                process_env["RQ1_BACKEND_TIMEOUT_SECONDS"] = str(remaining)
            if backend == "rvvm-riscv64":
                process_env["RQ1_BACKEND_DEADLINE_MONOTONIC"] = str(process_deadline)
            proc = _run_backend_process(
                list(command), cwd=runner_cwd, env=process_env,
                timeout=remaining,
                **({"on_timeout": timeout_callback} if timeout_callback is not None else {}),
            )
        elapsed_stderr = proc.stderr + f"\nELAPSED_MS={int((time.monotonic() - start) * 1000)}".encode()
        prefix = "RV_TOOL_VERSION="
        version_lines = [
            line
            for line in elapsed_stderr.decode("utf-8", errors="replace").splitlines()
            if line.startswith(prefix)
        ]
        if len(version_lines) > 1:
            observations.append(gap(
                "RV_TOOL_VERSION must appear exactly once",
                elapsed_stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode,
                raw_stdout=proc.stdout.hex(),
                tool_version=version,
            ))
            return _attach_source_pc_map(backend, elf_path, observations)
        effective_version = (
            version_lines[0].removeprefix(prefix).strip() or None
            if version_lines
            else None
        ) or version
        parsed = parse_observation_stdout(
            backend, proc.returncode, proc.stdout, elapsed_stderr,
            binary_hash, effective_version, profile_id, input_id,
            allow_state_only=allow_state_only,
            volatile_gpr_indices=volatile_gpr_indices,
        )
        if capsule_memory_size is not None and parsed.outcome == "normal" and not allow_state_only:
            snapshot = parsed.memory_delta.get("test-memory")
            if not isinstance(snapshot, str) or len(snapshot) // 2 != capsule_memory_size:
                parsed = gap(
                    "capsule memory snapshot size does not match configuration",
                    parsed.raw_stderr,
                    exit_code=proc.returncode,
                    raw_stdout=proc.stdout.hex(),
                    executed_pcs=parsed.executed_pcs,
                    instruction_count=parsed.instruction_count,
                    translation_evidence=parsed.translation_evidence,
                    tool_version=effective_version,
                )
        observations.append(parsed)
    except subprocess.TimeoutExpired as exc:
        raw_stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
        if native_timeout_diagnostics:
            raw_stderr += ("\n" + "\n".join(native_timeout_diagnostics) + "\n").encode()
        observations.append(_timeout_observation(
            backend,
            raw_stderr.decode("utf-8", errors="replace"),
            binary_hash,
            version,
            profile_id,
            input_id,
            raw_stdout=(exc.stdout if isinstance(exc.stdout, bytes) else b"").hex(),
            capsule_memory_size=capsule_memory_size,
        ))
    except TimeoutError as exc:
        observations.append(gap(
            str(exc), str(exc), tool_version=version,
            **({"extra_state": {"runner_failure_class": "k1-timeout"}}
               if backend == "native-rv64" else {}),
        ))
    except OSError as exc:
        observations.append(gap(
            str(exc), str(exc), tool_version=version,
            **({"extra_state": {"runner_failure_class": "k1-transport-unavailable"}}
               if backend == "native-rv64" else {}),
        ))
    return _attach_source_pc_map(backend, elf_path, observations)

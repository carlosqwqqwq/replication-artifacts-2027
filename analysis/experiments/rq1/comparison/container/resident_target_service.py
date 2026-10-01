#!/usr/bin/env python3
"""Consume one Target queue through one long-lived simulator session.

This is container-side runtime code.  It never imports the case-generation
framework and never starts a simulator for an individual job.  The configured
``session_command`` must implement ``rq1-target-session-v1`` over JSONL:
``hello`` once, then repeated ``run`` requests, and ``close`` on shutdown.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


PROTOCOL = "rq1-target-session-v1"


def _safe(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in ".-_" else "_"
        for char in str(value)
    ) or "unknown"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> tuple[int, str]:
    size = 0
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            sha.update(chunk)
    return size, sha.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


def _relative_run_paths(value: object, run_root: Path) -> object:
    """Keep persisted evidence independent from the container mount path."""
    if isinstance(value, Mapping):
        return {
            key: _relative_run_paths(item, run_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relative_run_paths(item, run_root) for item in value]
    if not isinstance(value, str):
        return value
    try:
        return str(Path(value).resolve().relative_to(run_root.resolve()))
    except (OSError, ValueError):
        return value


def _command(value: object) -> list[str] | None:
    if isinstance(value, (list, tuple)) and value:
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return None


class ResidentSession:
    """One request/response stream and one simulator process."""

    def __init__(self, command: list[str], *, metadata: Mapping[str, Any], stderr: Path):
        self.command = command
        self.metadata = dict(metadata)
        self.stderr = stderr
        self.process: subprocess.Popen[str] | None = None
        self.stderr_handle = None
        self.last_timing: dict[str, Any] = {}
        self.last_coverage: dict[str, Any] = {}

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def start(self) -> None:
        if self.process is not None:
            return
        self.stderr.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = self.stderr.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_handle,
                text=True,
                bufsize=1,
            )
        except BaseException:
            self.stderr_handle.close()
            self.stderr_handle = None
            raise
        try:
            response = self.request({
                "op": "hello",
                "protocol": PROTOCOL,
                "metadata": self.metadata,
            }, timeout=10.0)
            if response.get("status") not in {"ready", "ok"} \
                    and response.get("ready") is not True:
                raise RuntimeError("resident session handshake was not accepted")
        except BaseException:
            self.close()
            raise

    def request(self, payload: Mapping[str, Any], *, timeout: float | None) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("resident session is not connected")
        process.stdin.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")
        process.stdin.flush()
        readable, _, _ = select.select([process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("resident session response timed out")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("resident session exited without a response")
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RuntimeError("resident session response is not an object")
        response = dict(value)
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("error") or "resident session error"))
        timing = response.get("timing")
        self.last_timing = dict(timing) if isinstance(timing, Mapping) else {}
        coverage = response.get("coverage")
        self.last_coverage = dict(coverage) if isinstance(coverage, Mapping) else {}
        return response

    def run(self, artifact: Mapping[str, Any], *, job_id: str, metadata: Mapping[str, Any],
            timeout: float | None) -> dict[str, Any]:
        response = self.request({
            "op": "run", "job_id": job_id, "reset": True, "load": True,
            "collect": True, "artifact": artifact, "metadata": dict(metadata),
        }, timeout=timeout)
        result = response.get("result", response)
        return {
            "observations": result.get("observations", result)
            if isinstance(result, Mapping) else result,
            "response": _jsonable(response),
            "timing": _jsonable(self.last_timing),
            "coverage": _jsonable(self.last_coverage),
        }

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"op":"close"}\n')
                process.stdin.flush()
            # A resident coverage worker may be shutting down dotnet-coverage
            # and serializing the batch Cobertura report. The collector's
            # shutdown command can use its full minute timeout before the
            # worker exits; killing it at the old short bound loses Renode's
            # final report after all Target jobs already succeeded.
            process.wait(timeout=210)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.terminate()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            if self.stderr_handle is not None:
                try:
                    self.stderr_handle.close()
                except (OSError, ValueError):
                    pass
                self.stderr_handle = None


class TargetWorker:
    def __init__(self, method_dir: Path, target_id: str, runtime_path: Path,
                 *, poll_seconds: float = 0.05, once: bool = False):
        self.method_dir = method_dir.resolve()
        self.target_id = target_id
        self.runtime_path = runtime_path.resolve()
        self.root = self.method_dir / "target-queues" / _safe(target_id)
        self.queue_path = self.root / "queue.jsonl"
        self.cursor_path = self.root / "cursor.json"
        self.poll_seconds = max(0.01, poll_seconds)
        self.once = once
        self.stop_requested = False
        self.stop_after_current = False
        self.session: ResidentSession | None = None
        self.session_generation = 0
        self.session_start_count = 0
        self.last_simulator_pid: int | None = None
        self.completed = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_error_job_id: str | None = None

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True
        # Interrupt a backend read immediately.  The Target service owns its
        # simulator session, so an explicit container stop may close that
        # session directly; normal queue consumption never takes this path.
        if self.session is not None:
            self.session.close()

    def request_stop_after_current(self, _signum: int, _frame: object) -> None:
        """Finish the current queue item, then close the resident session."""
        self.stop_requested = True
        self.stop_after_current = True

    def runtime(self) -> dict[str, Any]:
        manifest = _read(self.runtime_path)
        targets = manifest.get("targets")
        value = targets.get(self.target_id) if isinstance(targets, Mapping) else None
        return dict(value) if isinstance(value, Mapping) else {}

    def runtime_manifest(self) -> dict[str, Any]:
        return _read(self.runtime_path)

    def _start_session(self) -> None:
        if self.session is not None:
            return
        runtime = self.runtime()
        command = _command(runtime.get("session_command"))
        if command is None:
            raise RuntimeError("resident-session-required: session_command is missing")
        self.session = ResidentSession(
            command,
            metadata={
                "target_id": self.target_id,
                "identity": runtime.get("identity"),
                "coverage_config": runtime.get("coverage_config"),
                "rv_instruction_coverage": (
                    self.runtime_manifest().get("rv_instruction_coverage")
                ),
                "rv_opcode_catalog_coverage": (
                    self.runtime_manifest().get("rv_opcode_catalog_coverage")
                ),
            },
            stderr=self.root / "session.stderr.log",
        )
        self.session.start()
        self.session_generation += 1
        self.session_start_count += 1
        self.last_simulator_pid = self.session.pid
        self._status("running")

    def _status(self, status: str, *, error: str | None = None) -> None:
        if error:
            self.last_error = error[:1000]
        session = self.session
        queue_state = _read(self.root / "state.json")
        published_count = queue_state.get("published_count", 0)
        try:
            published_count = max(0, int(published_count))
        except (TypeError, ValueError, OverflowError):
            published_count = 0
        cursor = _read(self.cursor_path)
        pending_count = max(0, published_count - self.completed)
        session_value: dict[str, Any] = {
            "schema_version": "rq1-target-session-status-v1",
            "target_id": self.target_id,
            "worker_status": status,
            "simulator_session_mode": "resident-session-service",
            "backend_session_scope": "per-target-session",
            "session_generation": self.session_generation,
            "start_count": self.session_start_count,
            "simulator_pid": session.pid if session is not None else None,
            "last_simulator_pid": self.last_simulator_pid,
            "simulator_pid_active": bool(session is not None and session.pid),
            "session_command": (
                list(session.command) if session is not None else self.runtime().get("session_command")
            ),
            "updated_at_epoch": time.time(),
        }
        if error:
            session_value["session_start_error"] = error[:1000]
        value: dict[str, Any] = {
            "schema_version": "rq1-resident-target-worker-v1",
            "target_id": self.target_id,
            "status": status,
            "pid": os.getpid(),
            "published_count": published_count,
            "completed_count": self.completed,
            "pending_count": pending_count,
            "cursor_byte_offset": self._cursor(),
            "last_job_id": cursor.get("last_job_id"),
            "simulator_pid": session.pid if session else None,
            "simulator_pid_active": bool(session is not None and session.pid),
            "backend_session_scope": "per-target-session",
            "simulator_session_mode": "resident-session-service",
            "session_generation": self.session_generation,
            "start_count": self.session_start_count,
            "completed_count": self.completed,
            "error_count": self.errors,
            "last_error": self.last_error,
            "last_error_job_id": self.last_error_job_id,
            "updated_at_epoch": time.time(),
        }
        if error:
            value["error"] = error[:1000]
        _write(self.root / "worker-status.json", value)
        _write(
            self.method_dir / "target-sessions" / f"{_safe(self.target_id)}.json",
            session_value,
        )

    def _cursor(self) -> int:
        value = _read(self.cursor_path).get("byte_offset", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _next(self) -> tuple[dict[str, Any], int] | None:
        if not self.queue_path.is_file():
            return None
        offset = self._cursor()
        with self.queue_path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.readline()
        if not raw or not raw.endswith(b"\n"):
            return None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, Mapping) or value.get("target_id") != self.target_id:
            raise RuntimeError("Target queue identity mismatch")
        return dict(value), offset + len(raw)

    def _artifact(self, record: Mapping[str, Any], case_root: Path) -> dict[str, Any]:
        value = dict(record)
        for name in ("source_path", "executable_path", "bare_executable_path", "linux_executable_path"):
            path = value.get(name)
            if isinstance(path, str) and path and not Path(path).is_absolute():
                value[name] = str((case_root / path).resolve())
        return value

    def _raw_case_root(self, entry: Mapping[str, Any]) -> Path:
        """Keep raw evidence separate when candidate IDs repeat across cases."""
        case_root = entry.get("case_root")
        if not isinstance(case_root, str) or not case_root.strip():
            return self.root / "raw"
        parts = [
            _safe(part) for part in case_root.strip("/").split("/")
            if part not in {"", ".", ".."}
        ]
        return self.root / "raw" / "cases" / Path(*parts)

    def _case_coverage_config(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Load raw/profile routing for this case from the producer snapshot."""
        fallback = self.runtime().get("coverage_config")
        fallback = dict(fallback) if isinstance(fallback, Mapping) else {}
        case_root = entry.get("case_root")
        if not isinstance(case_root, str):
            return fallback
        parts = Path(case_root).parts
        lane = next((part for part in parts if part.startswith("lane-")), None)
        case = next((part for part in parts if part.startswith("case-")), None)
        if lane is None or case is None:
            return fallback
        try:
            lane_id = int(lane.removeprefix("lane-"))
            case_id = int(case.removeprefix("case-"))
        except ValueError:
            return fallback
        config_path = (
            self.method_dir.parent.parent / "framework-coverage-configs"
            / self.method_dir.name / f"lane-{lane_id:02d}-case-{case_id:06d}.json"
        )
        document = _read(config_path)
        targets = document.get("targets")
        target = targets.get(self.target_id) if isinstance(targets, Mapping) else None
        value = target.get("coverage_config") if isinstance(target, Mapping) else None
        return dict(value) if isinstance(value, Mapping) else fallback

    @staticmethod
    def _artifact_gap(job: Mapping[str, Any]) -> str | None:
        declared = job.get("artifact_gap")
        if isinstance(declared, str) and declared:
            return declared
        if job.get("job_type") == "baseline":
            return None if isinstance(job.get("artifact_record"), Mapping) else (
                "baseline-artifact-record-missing"
            )
        records = job.get("artifact_records")
        if not isinstance(records, Mapping):
            return "reference-pair-artifacts-missing"
        missing = [
            name for name in ("reference_original", "reference_variant")
            if not isinstance(records.get(name), Mapping)
        ]
        return (
            "reference-pair-artifacts-incomplete:" + ",".join(missing)
            if missing else None
        )

    @staticmethod
    def _observation_gap(value: object) -> str | None:
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            extra = row.get("extra_state")
            if isinstance(extra, Mapping) and extra.get("target_unsupported"):
                continue
            error = row.get("contract_error")
            outcome = str(row.get("outcome") or "").strip().lower()
            if error:
                return str(error)
            if outcome in {
                "runner-contract-gap", "transport-gap", "timeout",
                "unavailable", "failed", "error",
            }:
                return outcome
        return None

    @staticmethod
    def _observation_unsupported(value: object) -> str | None:
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            extra = row.get("extra_state")
            if isinstance(extra, Mapping) and extra.get("target_unsupported"):
                return str(extra["target_unsupported"])
        return None

    def _write_result(self, entry: Mapping[str, Any], next_offset: int,
                      result: Mapping[str, Any]) -> None:
        safe_result = _jsonable(result)
        full_result_value = (
            dict(safe_result) if isinstance(safe_result, Mapping)
            else {"value": safe_result}
        )
        # Raw observations and execution responses can contain complete guest
        # traces. They already have an immutable owner in raw_evidence_path;
        # duplicating them in both ``record`` and ``result`` turns one replay
        # result into tens of megabytes and makes offline analysis reread the
        # same trace repeatedly. Keep the queue result as a compact index.
        result_value = dict(full_result_value)
        for name in ("raw_observations", "raw_executions"):
            result_value.pop(name, None)
        if "raw_evidence_path" in full_result_value:
            result_value["raw_result_mode"] = "sidecar"
        result_value["raw_observation_phases"] = sorted(
            full_result_value.get("raw_observations", {}).keys()
            if isinstance(full_result_value.get("raw_observations"), Mapping)
            else full_result_value.get("raw_executions", {}).keys()
            if isinstance(full_result_value.get("raw_executions"), Mapping)
            else ()
        )
        result_value.setdefault("target_id", self.target_id)
        result_value.setdefault("target_result_status", result_value.get("status", "target-gap"))
        result_value.setdefault("target_attempted", True)
        result_path_value = entry.get("result_path")
        if not isinstance(result_path_value, str) or not result_path_value:
            raise ValueError("Target queue entry has no result_path")
        result_path = (self.method_dir / result_path_value).resolve()
        result_path.relative_to(self.method_dir)
        completed = self.completed + 1
        _write(result_path, {
            "schema_version": "rq1-target-replay-result-v1",
            "target_id": self.target_id,
            "job_id": entry.get("job_id"),
            "pair_index": entry.get("pair_index"),
            "progress_sequence": completed - 1,
            "counts": {},
            "record": result_value,
            "result": result_value,
            "simulator_session_mode": "resident-session-service",
            "backend_session_scope": "per-target-session",
        })
        _write(self.cursor_path, {
            "schema_version": "rq1-target-queue-cursor-v1",
            "target_id": self.target_id,
            "byte_offset": next_offset,
            "completed_jobs": completed,
            "last_job_id": entry.get("job_id"),
            "updated_at_epoch": time.time(),
        })
        self.completed = completed

    def _write_raw_evidence(
        self, entry: Mapping[str, Any], job_id: str,
        executions: Mapping[str, Any], coverage_config: Mapping[str, Any],
        raw_manifest: Mapping[str, Any],
    ) -> str:
        manifest = self.runtime_manifest()
        run_root = self.method_dir.parent.parent
        raw_path = self._raw_case_root(entry) / f"{_safe(job_id)}.json"
        _write(raw_path, {
            "schema_version": "rq1-target-raw-execution-v1",
            "target_id": self.target_id,
            "job_id": job_id,
            "case_root": entry.get("case_root"),
            "backend_session_scope": "per-target-session",
            "session_pid": self.session.pid if self.session else None,
            "simulator_coverage": _jsonable(
                _relative_run_paths(coverage_config, run_root)
            ),
            "rv_instruction_coverage": manifest.get("rv_instruction_coverage"),
            "rv_opcode_catalog_coverage": manifest.get("rv_opcode_catalog_coverage"),
            "executions": _relative_run_paths(_jsonable(executions), run_root),
            "raw_artifact_manifest": _jsonable(raw_manifest),
            "recorded_at_epoch": time.time(),
        })
        return str(raw_path.relative_to(self.method_dir))

    def _write_raw_artifact_manifest(
        self, entry: Mapping[str, Any], job_id: str,
        executions: Mapping[str, Any],
        coverage_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Index Target-side and simulator-side raw files for this job.

        The resident session writes observations below ``target-queues`` but
        source-coverage adapters write their counters/traces below the
        configured coverage batch directory.  Both are run-owned evidence;
        indexing only the former makes a successful simulator collection look
        like a missing artifact.
        """
        case_root = self._raw_case_root(entry)
        files: list[dict[str, Any]] = []
        seen: set[Path] = set()
        coverage_roots: list[str] = []

        def add_root(value: object) -> None:
            if not isinstance(value, str) or not value.strip():
                return
            path = Path(value)
            if not path.is_absolute():
                path = self.method_dir / path
            try:
                path = path.resolve()
                path.relative_to(self.method_dir)
            except (OSError, ValueError):
                return
            if not path.is_dir():
                return
            coverage_roots.append(str(path.relative_to(self.method_dir)))
            for item in sorted(path.rglob("*")):
                if not item.is_file():
                    continue
                try:
                    resolved = item.resolve()
                    relative = resolved.relative_to(self.method_dir)
                except (OSError, ValueError):
                    continue
                # 编译器临时对象不是 coverage 输入，任务结束后会被清理；不把它写进
                # job manifest，避免收尾时把正常清理误判成 raw 缺失。
                if resolved.suffix == ".o":
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                size, sha = _file_digest(resolved)
                files.append({
                    "path": str(relative),
                    "kind": resolved.suffix.lstrip(".") or "raw",
                    "size_bytes": size,
                    "sha256": sha,
                    "flush_status": "complete",
                    "source": "simulator-coverage",
                })

        for phase in executions:
            phase_root = case_root / _safe(job_id) / _safe(phase)
            if phase_root.is_dir():
                for path in sorted(item for item in phase_root.rglob("*") if item.is_file()):
                    try:
                        resolved = path.resolve()
                        relative = resolved.relative_to(self.method_dir)
                    except (OSError, ValueError):
                        continue
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    size, sha = _file_digest(resolved)
                    files.append({
                        "path": str(relative),
                        "kind": resolved.suffix.lstrip(".") or "raw",
                        "size_bytes": size,
                        "sha256": sha,
                        "flush_status": "complete",
                        "source": "target-observation",
                    })

        coverage_config = coverage_config if isinstance(coverage_config, Mapping) else {}
        raw_names = ["raw_dir"]
        # Renode server-mode writes its Cobertura report at the batch root.
        # Keep this job manifest case-local; offline coverage may inspect the
        # shared batch root later, after the session has been flushed.
        for name in raw_names:
            add_root(coverage_config.get(name))
        files.sort(key=lambda item: str(item.get("path", "")))
        manifest_path = case_root / f"{_safe(job_id)}.raw-artifact-manifest.json"
        manifest = {
            "schema_version": "rq1-target-raw-artifact-manifest-v1",
            "target_id": self.target_id,
            "job_id": job_id,
            "case_root": entry.get("case_root"),
            "status": "raw-ready" if files else "missing",
            "files": files,
            "file_count": len(files),
            "recorded_file_count": len(files),
            "coverage_raw_roots": coverage_roots,
            "path": str(manifest_path.relative_to(self.method_dir)),
        }
        _write(manifest_path, manifest)
        return manifest

    def _process(self, entry: Mapping[str, Any], next_offset: int) -> None:
        case_root_value = entry.get("case_root")
        job_path_value = entry.get("job_path")
        if not isinstance(case_root_value, str) or not isinstance(job_path_value, str):
            raise ValueError("Target queue entry has no case/job path")
        case_root = (self.method_dir / case_root_value).resolve()
        job_path = (case_root / job_path_value).resolve()
        case_root.relative_to(self.method_dir)
        job_path.relative_to(case_root)
        job = _read(job_path)
        job_id = str(entry.get("job_id") or job_path.stem)
        if artifact_gap := self._artifact_gap(job):
            self._write_result(entry, next_offset, {
                "status": "target-gap",
                "failure_class": "artifact-gap",
                "reason": artifact_gap,
                "target_attempted": False,
                "target": {
                    "status": "target-gap",
                    "differences": {"target_error": artifact_gap},
                },
            })
            self._status("running")
            return
        runtime = self.runtime()
        case_coverage_config = self._case_coverage_config(entry)
        timeout_value = runtime.get("timeout_seconds")
        timeout = float(timeout_value) if isinstance(timeout_value, (int, float)) else None
        session_timeout = timeout
        source = case_coverage_config.get("source_coverage")
        if isinstance(source, Mapping) and source.get("collector") == "dotnet":
            # The normal Renode run keeps the target budget.  Its separate
            # Cobertura replay needs the existing collector wait window too;
            # otherwise the JSONL session kills a healthy replay at 180s.
            value = case_coverage_config.get("coverage_timeout_seconds", 300)
            try:
                coverage_timeout = min(1800.0, max(1.0, float(value)))
            except (TypeError, ValueError, OverflowError):
                coverage_timeout = 180.0
            session_timeout = (
                None if timeout is None else timeout + 2.0 * coverage_timeout + 70.0
            )
        self._start_session()

        def run(record: object, request_id: str, phase: str) -> dict[str, Any]:
            if not isinstance(record, Mapping):
                raise ValueError("Target artifact record is missing")
            raw_root = self._raw_case_root(entry) / _safe(job_id) / _safe(phase)
            raw_root.mkdir(parents=True, exist_ok=True)
            return self.session.run(
                self._artifact(record, case_root), job_id=request_id,
                metadata={
                    "target_id": self.target_id,
                    "job_id": request_id,
                    "phase": phase,
                    "raw_root": str(raw_root),
                    "coverage_config": case_coverage_config,
                    "timeout_seconds": timeout,
                    "rv_instruction_coverage": (
                        self.runtime_manifest().get("rv_instruction_coverage")
                    ),
                    "rv_opcode_catalog_coverage": (
                        self.runtime_manifest().get("rv_opcode_catalog_coverage")
                    ),
                },
                timeout=session_timeout,
            )

        if job.get("job_type") == "baseline":
            raw = run(job.get("artifact_record"), job_id, "baseline")
            raw_executions = {"baseline": raw}
            unsupported_reason = self._observation_unsupported(raw.get("observations"))
            gap_reason = None if unsupported_reason else self._observation_gap(
                raw.get("observations")
            )
            result_status = (
                "case-skipped" if unsupported_reason else
                "target-gap" if gap_reason else "raw-recorded"
            )
            result = {
                "status": result_status, "raw_observations": raw["observations"],
                "raw_executions": raw_executions,
                "target": {"status": result_status, "observations": raw["observations"]},
                "comparison_pending": result_status == "raw-recorded",
            }
        else:
            records = job.get("artifact_records")
            if not isinstance(records, Mapping):
                raise ValueError("Target artifact records are missing")
            original = run(records.get("reference_original"), job_id + ":original", "original")
            variant = run(records.get("reference_variant"), job_id + ":variant", "variant")
            raw_executions = {"original": original, "variant": variant}
            unsupported_reason = (
                self._observation_unsupported(original.get("observations"))
                or self._observation_unsupported(variant.get("observations"))
            )
            gap_reason = None if unsupported_reason else (
                self._observation_gap(original.get("observations"))
                or self._observation_gap(variant.get("observations"))
            )
            result_status = (
                "case-skipped" if unsupported_reason else
                "target-gap" if gap_reason else "raw-recorded"
            )
            result = {
                "status": result_status,
                "raw_observations": {
                    "original": original["observations"],
                    "variant": variant["observations"],
                },
                "raw_executions": {"original": original, "variant": variant},
                "target": {
                    "status": result_status,
                    "observations": {
                        "original": original["observations"],
                        "variant": variant["observations"],
                    },
                },
                "comparison_pending": result_status == "raw-recorded",
            }
        if unsupported_reason:
            result["reason"] = unsupported_reason
            result["failure_class"] = "target-unsupported"
        if gap_reason:
            result["failure_class"] = "runner-contract-gap"
            result["reason"] = gap_reason
        raw_manifest = self._write_raw_artifact_manifest(
            entry, job_id, raw_executions, case_coverage_config,
        )
        result["raw_artifact_manifest"] = raw_manifest
        result["coverage"] = {
            "status": "raw-recorded" if raw_manifest["status"] == "raw-ready" else "missing",
            "raw_capture_status": raw_manifest["status"],
            "raw_artifact_manifest": raw_manifest["path"],
        }
        result["raw_evidence_path"] = self._write_raw_evidence(
            entry, job_id, raw_executions, case_coverage_config, raw_manifest,
        )
        self._write_result(entry, next_offset, result)
        self._status("running")

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        cursor = _read(self.cursor_path)
        try:
            self.completed = max(0, int(cursor.get("completed_jobs", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            self.completed = 0
        self._status("starting")
        while not self.stop_requested:
            try:
                item = self._next()
                if item is None:
                    if self.once:
                        break
                    time.sleep(self.poll_seconds)
                    continue
                entry, next_offset = item
                try:
                    self._process(entry, next_offset)
                except Exception as error:
                    if self.stop_requested and not self.stop_after_current:
                        # Closing a native session can interrupt the current
                        # read.  That is an explicit partial stop, not a
                        # simulator/worker error; leave the queue line for a
                        # later resume and keep error_count unchanged.
                        if self.session is not None:
                            self.session.close()
                            self.session = None
                        self._status("stopping")
                        break
                    self.errors += 1
                    self.last_error_job_id = str(entry.get("job_id") or "") or None
                    # A protocol timeout or malformed response leaves the
                    # stream state unknown.  Do not reuse that session for the
                    # next queue item; the next valid item will start a new
                    # resident generation for this Target.
                    if self.session is not None:
                        self.session.close()
                        self.session = None
                    self._status("error", error=f"{type(error).__name__}: {error}")
                    # A malformed job or absent service is Target-local. Do
                    # not advance its cursor, so a later resume can retry it.
                    if isinstance(error, RuntimeError) and str(error).startswith("resident-session-required"):
                        self._status("missing-session-command", error=str(error))
                        return 2
                    artifact_gap = (
                        isinstance(error, ValueError)
                        and "Target artifact record is missing" in str(error)
                    )
                    result = {
                        "status": "target-gap",
                        "failure_class": "artifact-gap" if artifact_gap else "target-worker-error",
                        "reason": f"{type(error).__name__}: {error}",
                        "target_attempted": not artifact_gap,
                    }
                    self._write_result(entry, next_offset, result)
            except Exception as error:
                if self.stop_requested:
                    self._status("stopping")
                    break
                self.errors += 1
                self._status("error", error=f"{type(error).__name__}: {error}")
                time.sleep(self.poll_seconds)
        if self.session is not None:
            self.session.close()
            self.session = None
        self._status("stopped")
        return 0 if self.errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-dir", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    worker = TargetWorker(args.method_dir, args.target, args.runtime_config, once=args.once)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, worker.request_stop)
    signal.signal(signal.SIGUSR1, worker.request_stop_after_current)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())

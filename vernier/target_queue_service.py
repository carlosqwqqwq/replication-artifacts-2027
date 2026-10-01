"""独立 Target 队列消费者。

The Direct producer owns reference/EMI/MCMC.  This module owns only the
durable Target job streams and their per-Target consumers.  A consumer sends
artifacts to its simulator session and persists raw observations/timing/
coverage; it does not run EMI, compare against reference observations, or
publish a campaign verdict.  It deliberately does not expose a campaign-wide
completion future: a Target result is a sidecar for ``case_id + target_id``.

Formal stream-only runs use a resident TargetSession when one is configured.
Until a native service is deployed for a Target, the queue records the explicit
``resident-session-required/unavailable`` gap and does not fall back to a
per-job callback. T-UNICORN's embedded worker remains a resident session.
The mode is recorded truthfully in every result.

The producer manifest is an immutable input plan.  A consumer never changes
``target-replay-queue.json`` and never updates another Target's progress;
completion is represented only by this Target's result sidecars.
"""

from __future__ import annotations

from collections.abc import Mapping
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from framework._util import (
    atomic_write_json, canonical_digest, json_safe, read_json_object,
)
from framework.adapters.program import BuiltProgram
from framework.program_runner import backend_runner
from framework.target_session import (
    ResidentSessionRequiredError, TargetSession, build_target_session,
)


def _safe(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value)
    ) or "unknown"


def _built_program(record: Mapping[str, object]) -> BuiltProgram:
    if not isinstance(record, Mapping):
        raise ValueError("target artifact record must be an object")
    materialization = record.get("materialization")
    materialization = materialization if isinstance(materialization, Mapping) else {}
    linux = materialization.get("linux_user")
    linux = linux if isinstance(linux, Mapping) else {}
    bare = materialization.get("bare_metal")
    bare = bare if isinstance(bare, Mapping) else {}
    executable = record.get("executable_path")
    source = record.get("source_path")
    if not isinstance(source, str) or not isinstance(executable, str):
        raise ValueError("target artifact paths are missing")
    linux_path = record.get("linux_executable_path")
    if linux_path is None and linux.get("status") == "built":
        linux_path = linux.get("path")
    bare_path = record.get("bare_executable_path")
    if bare_path is None:
        bare_path = bare.get("path")
    return BuiltProgram(
        Path(source), Path(executable),
        str(record.get("source_sha256") or ""),
        str(record.get("executable_sha256") or ""),
        record.get("parent_sha256") if isinstance(record.get("parent_sha256"), str) else None,
        record.get("run_params", {}) if isinstance(record.get("run_params"), Mapping) else {},
        Path(bare_path) if isinstance(bare_path, str) else None,
        record.get("bare_executable_sha256")
        if isinstance(record.get("bare_executable_sha256"), str) else bare.get("sha256"),
        str(bare.get("status") or "not-requested"),
        bare.get("reason") if isinstance(bare.get("reason"), str) else None,
        Path(linux_path) if isinstance(linux_path, str) else None,
        record.get("linux_executable_sha256")
        if isinstance(record.get("linux_executable_sha256"), str) else linux.get("sha256"),
        str(linux.get("status") or "not-requested"),
        linux.get("reason") if isinstance(linux.get("reason"), str) else None,
    )


def _target_record(
    job: Mapping[str, object], target_id: str, result: Mapping[str, object],
) -> dict[str, object]:
    target = result.get("target")
    target = dict(target) if isinstance(target, Mapping) else {
        "status": "target-gap",
        "differences": {"target_error": result.get("reason")},
    }
    pair_status = str(result.get("status") or "target-gap")
    status = (
        "raw-recorded" if pair_status == "raw-recorded" else
        "clean" if pair_status == "clean" else
        "target-mismatch-candidate" if pair_status == "target-mismatch-candidate" else
        "target-tested" if pair_status == "target-tested" else
        "case-skipped" if pair_status == "case-skipped" else
        "target-gap"
    )
    artifacts = result.get("artifacts")
    target_artifact = (
        artifacts.get("target_variant")
        if isinstance(artifacts, Mapping) else None
    )
    if target_artifact is None and isinstance(artifacts, Mapping):
        target_artifact = artifacts.get("reference_variant")
    return {
        "step": job.get("step"),
        "reference_index": job.get("reference_index"),
        "state_sha256": job.get("state_sha256"),
        "parent_sha256": job.get("parent_sha256"),
        "target_id": target_id,
        "status": status,
        "terminal_disposition": status,
        "target_attempted": result.get("target_attempted") is not False,
        "target_result_status": pair_status,
        "comparison_qualified": status in {"clean", "target-mismatch-candidate"},
        "comparison_pending": status == "raw-recorded",
        "comparison": target,
        **({"raw_observations": result["raw_observations"]}
           if "raw_observations" in result else {}),
        **({"raw_executions": result["raw_executions"]}
           if "raw_executions" in result else {}),
        "artifact": target_artifact,
        **({"failure_class": result["failure_class"]}
           if isinstance(result.get("failure_class"), str) else {}),
        **({"coverage_evidence": dict(result["coverage_evidence"])}
           if isinstance(result.get("coverage_evidence"), Mapping) else {}),
        **({"deadline_censored": True}
           if result.get("deadline_censored") is True else {}),
    }


def _progress_delta(record: Mapping[str, object], *, baseline: bool) -> dict[str, int]:
    status = record.get("status")
    attempted = record.get("target_attempted") is not False
    qualified = record.get("comparison_qualified") is True
    tested = (
        status == "target-tested" and qualified
        if baseline else
        status in {"clean", "target-mismatch-candidate"} and qualified
    )
    raw_recorded = status == "raw-recorded"
    return {
        "target_attempted": int(attempted),
        "target_tested": int(tested),
        "target_gap": int(
            attempted and not tested
            and status not in {"case-skipped", "raw-recorded"}
        ),
        "target_raw_recorded": int(raw_recorded),
        "target_comparison_pending": int(raw_recorded),
        "target_unsupported": int(attempted and status == "case-skipped"),
        "case_skipped": int(attempted and status == "case-skipped"),
        "target_mismatch_candidate": int(
            tested and status == "target-mismatch-candidate"
        ),
        "target_transport_gap": int(
            record.get("failure_class") == "transport-gap"
            or record.get("target_result_status") == "transport-gap"
        ),
        "target_wall_clock_pending": int(
            record.get("deadline_censored") is True
            or record.get("failure_class") == "campaign-wall-clock-exhausted"
        ),
        "clean": int(tested and status == "clean"),
        "artifact_gap": int(status == "artifact-gap"),
    }


class TargetQueueCoordinator:
    """Consume all case queues in one campaign with one worker per Target."""

    def __init__(
        self, method_dir: str | Path, target_ids: tuple[str, ...],
        *, owner: str = "standalone", stream_only: bool | None = None,
    ):
        self.method_dir = Path(method_dir).resolve()
        self.target_ids = tuple(str(value) for value in target_ids)
        self.owner = str(owner)
        self._stream_only = (
            bool(stream_only)
            if stream_only is not None else
            (self.method_dir / "target-queues").is_dir()
        )
        self._stop = threading.Event()
        self._drain = threading.Event()
        self._threads: list[threading.Thread] = []
        self._owner_fds: dict[str, object] = {}
        self._claimed: set[tuple[str, str]] = set()
        self._session_cache: dict[str, tuple[str, TargetSession]] = {}
        self._session_status: dict[str, tuple[str, str]] = {}
        self._session_start_seconds: dict[str, float] = {}
        self._session_generation: dict[str, int] = {}
        self._session_start_count: dict[str, int] = {}
        self._session_last_pid: dict[str, int] = {}
        self._session_last_identity: dict[str, Mapping[str, object]] = {}
        self._session_start_errors: dict[str, str] = {}
        self._session_unavailable: set[str] = set()
        self._coverage_prewarm_status: dict[str, str] = {}
        self._coverage_prewarm_errors: dict[str, str] = {}
        self._job_start: dict[str, tuple[float, float]] = {}
        self._cursor_offsets: dict[str, int] = {}
        self._worker_status: dict[str, str] = {}
        self._worker_errors: dict[str, dict[str, str]] = {}
        self.errors: list[dict[str, str]] = []

    def _session_descriptor(self, target_id: str) -> tuple[str, str]:
        return self._session_status.get(
            target_id,
            (
                ("resident-session-required", "unavailable")
                if self._stream_only else
                ("adapter-callback", "per-job")
            ),
        )

    def _write_session_status(self, target_id: str, worker_status: str) -> None:
        """Persist truthful worker/session metadata for one Target."""
        mode, scope = self._session_descriptor(target_id)
        session = self._session_cache.get(target_id)
        process = getattr(session[1], "process", None) if session else None
        live_pid = getattr(process, "pid", None)
        session_object = session[1] if session else None
        last_pid = self._session_last_pid.get(target_id)
        if session_object is not None:
            observed_last_pid = getattr(session_object, "last_session_pid", None)
            if isinstance(observed_last_pid, int):
                last_pid = observed_last_pid
                self._session_last_pid[target_id] = observed_last_pid
            observed_identity = getattr(
                session_object, "last_session_identity", None,
            )
            if isinstance(observed_identity, Mapping):
                last_identity = dict(observed_identity)
                self._session_last_identity[target_id] = last_identity
            else:
                last_identity = self._session_last_identity.get(target_id)
        else:
            last_identity = self._session_last_identity.get(target_id)
        simulator_pid = live_pid if isinstance(live_pid, int) else last_pid
        atomic_write_json(self.method_dir / "target-sessions" / f"{_safe(target_id)}.json", {
            "schema_version": "rq1-target-session-status-v1",
            "target_id": target_id,
            "worker_status": worker_status,
            "session_generation": self._session_generation.get(target_id, 0),
            "start_count": self._session_start_count.get(target_id, 0),
            "start_count_scope": "target-consumer-process",
            # Keep the last process identity after a normal stop.  The
            # ``simulator_pid_active`` bit distinguishes a live session from
            # historical evidence without erasing the identity at shutdown.
            "simulator_pid": simulator_pid,
            "simulator_pid_active": isinstance(live_pid, int),
            "last_simulator_pid": last_pid,
            "last_session_identity": last_identity,
            "owner": self.owner,
            "simulator_session_mode": mode,
            "backend_session_scope": scope,
            "coverage_runtime_prewarm": self._coverage_prewarm_status.get(
                target_id, "not-configured",
            ),
            "coverage_runtime_prewarm_error": self._coverage_prewarm_errors.get(
                target_id,
            ),
            "session_start_seconds": self._session_start_seconds.get(target_id),
            "session_start_error": self._session_start_errors.get(target_id),
            "updated_at_epoch": time.time(),
        })

    def _write_worker_status(
        self, target_id: str, status: str, *, error: str | None = None,
    ) -> None:
        """Persist one Target worker's lifecycle and queue counters."""
        root = self._target_queue_root(target_id)
        root.mkdir(parents=True, exist_ok=True)
        try:
            state = read_json_object(root / "state.json")
        except (OSError, TypeError, ValueError):
            state = {}
        try:
            progress = read_json_object(root / "progress.json")
        except (OSError, TypeError, ValueError):
            progress = {}
        try:
            published = max(0, int(state.get("published_count", 0)))
        except (TypeError, ValueError, OverflowError):
            published = 0
        try:
            completed = max(0, int(progress.get("completed_count", 0)))
        except (TypeError, ValueError, OverflowError):
            completed = 0
        self._worker_status[target_id] = status
        payload: dict[str, object] = {
            "schema_version": "rq1-target-worker-status-v1",
            "target_id": target_id,
            "status": status,
            "owner": self.owner,
            "pid": os.getpid(),
            "published_count": published,
            "completed_count": completed,
            "pending_count": max(0, published - completed),
            "updated_at_epoch": time.time(),
        }
        if error:
            payload["error"] = error[:1000]
        elif target_id in self._worker_errors:
            payload["error"] = self._worker_errors[target_id]["error"]
        atomic_write_json(root / "worker-status.json", payload)

    @property
    def worker_errors(self) -> dict[str, dict[str, str]]:
        return dict(self._worker_errors)

    def start(self) -> None:
        if self._threads or self._owner_fds:
            raise RuntimeError("Target queue coordinator already started")
        self.method_dir.mkdir(parents=True, exist_ok=True)
        if self._stream_only:
            for target_id in self.target_ids:
                self._target_queue_root(target_id).mkdir(parents=True, exist_ok=True)
        session_root = self.method_dir / "target-sessions"
        session_root.mkdir(parents=True, exist_ok=True)
        try:
            for target_id in self.target_ids:
                owner_path = self.method_dir / (
                    f".target-consumer-{_safe(target_id)}.lock"
                )
                try:
                    owner_fd = owner_path.open("a+", encoding="utf-8")
                except OSError as error:
                    # A broken/unavailable Target is local state. Do not
                    # prevent the other Target workers from consuming their
                    # own queues.
                    message = f"{type(error).__name__}: {error}"[:1000]
                    self._worker_errors[target_id] = {
                        "target_id": target_id, "error": message,
                    }
                    self._worker_status[target_id] = "error"
                    self._write_worker_status(target_id, "error", error=message)
                    self._write_session_status(target_id, "error")
                    continue
                try:
                    fcntl.flock(owner_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    owner_fd.close()
                    message = (
                        f"Target {target_id} already has an active consumer: "
                        f"{self.method_dir}"
                    )[:1000]
                    self._worker_errors[target_id] = {
                        "target_id": target_id, "error": message,
                    }
                    self._worker_status[target_id] = "error"
                    self._write_worker_status(target_id, "error", error=message)
                    self._write_session_status(target_id, "error")
                    continue
                owner_fd.seek(0)
                owner_fd.truncate()
                owner_fd.write(f"owner={self.owner} pid={os.getpid()}\n")
                owner_fd.flush()
                self._owner_fds[target_id] = owner_fd
        except BaseException:
            self._release_owner()
            raise
        try:
            for target_id in self.target_ids:
                if target_id not in self._owner_fds:
                    continue
                self._write_worker_status(target_id, "running")
                self._write_session_status(target_id, "running")
                thread = threading.Thread(
                    target=self._worker, args=(target_id,),
                    name=f"target-queue-{_safe(target_id)}", daemon=False,
                )
                try:
                    thread.start()
                except BaseException as error:
                    message = f"{type(error).__name__}: {error}"[:1000]
                    self._worker_errors[target_id] = {
                        "target_id": target_id, "error": message,
                    }
                    self._worker_status[target_id] = "error"
                    self._write_worker_status(target_id, "error", error=message)
                    self._write_session_status(target_id, "error")
                    continue
                self._threads.append(thread)
        except BaseException:
            self._stop.set()
            for thread in self._threads:
                thread.join()
            self._threads.clear()
            self._release_owner()
            raise

    def _prewarm_target_runtime(self, target_id: str) -> None:
        """Warm immutable coverage runtime once before this Target worker."""
        self._coverage_prewarm_status[target_id] = "not-configured"
        manifest_path = self.method_dir / "target-runtime-configs.json"
        try:
            manifest = read_json_object(manifest_path)
            targets = manifest.get("targets")
            entry = targets.get(target_id) if isinstance(targets, Mapping) else None
            coverage = entry.get("coverage_config") \
                if isinstance(entry, Mapping) else None
            if not isinstance(coverage, Mapping) or not coverage.get("enabled", True):
                self._coverage_prewarm_status[target_id] = "disabled"
                return
            from framework.framework_coverage import prewarm_coverage_target

            prewarm_coverage_target(coverage)
            self._coverage_prewarm_status[target_id] = "warmed"
        except FileNotFoundError:
            # A standalone consumer may be started before the producer has
            # written the run-level manifest. The first job retries this one
            # startup-only operation; it is never part of a simulator job.
            self._coverage_prewarm_status[target_id] = "pending-manifest"
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._coverage_prewarm_status[target_id] = "gap"
            self._coverage_prewarm_errors[target_id] = (
                f"{type(error).__name__}: {error}"[:1000]
            )
            self.errors.append({
                "target_id": target_id,
                "job": "coverage-prewarm",
                "error": f"{type(error).__name__}: {error}"[:1000],
            })

    def _prime_target_session(self, target_id: str) -> None:
        """Start this Target's resident session before its first queue item.

        The runtime manifest is immutable and is written before the supervisor
        starts this consumer.  Priming here makes the lifecycle explicit:
        coverage/runtime warm-up and simulator session startup happen once on
        the Target worker, while each later queue item only submits a reset /
        load / collect request.  A missing native service remains a Target-
        local gap and never falls back to a per-job simulator.
        """
        try:
            manifest = read_json_object(
                self.method_dir / "target-runtime-configs.json",
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return
        targets = manifest.get("targets")
        entry = targets.get(target_id) if isinstance(targets, Mapping) else None
        if not isinstance(entry, Mapping):
            return
        backend = entry.get("backend")
        binary = entry.get("binary_path")
        identity = entry.get("identity_section")
        if not isinstance(backend, str) or not isinstance(identity, Mapping):
            return
        config: dict[str, object] = {
            "backend": backend,
            "binary_path": binary,
            "identity_section": dict(identity),
            "session_command": entry.get("session_command"),
            "coverage_config": entry.get("coverage_config"),
        }
        if not config.get("session_command") and backend not in {
            "unicorn", "unicorn-riscv64",
        }:
            self._session_unavailable.add(target_id)
            self._session_status[target_id] = (
                "resident-session-required", "unavailable",
            )
            self._write_session_status(target_id, "running")
            return
        try:
            self._target_runner(target_id, config)
        except ResidentSessionRequiredError:
            self._session_unavailable.add(target_id)
            self._session_status[target_id] = (
                "resident-session-required", "unavailable",
            )
            self._write_session_status(target_id, "running")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._session_unavailable.add(target_id)
            self._session_start_errors[target_id] = (
                f"{type(error).__name__}: {error}"[:1000]
            )
            self._session_status[target_id] = (
                "resident-session-required", "unavailable",
            )
            self._write_session_status(target_id, "running")

    def _release_owner(self) -> None:
        owner_fds = self._owner_fds
        self._owner_fds = {}
        for owner_fd in owner_fds.values():
            try:
                fcntl.flock(owner_fd.fileno(), fcntl.LOCK_UN)
            finally:
                owner_fd.close()

    def stop(self, *, drain: bool = False) -> None:
        if drain:
            self._drain.set()
        else:
            self._stop.set()
        # Close resident sessions before joining workers. A worker may be
        # blocked in a native session read; waiting first would make a normal
        # pause/stop look like a deadlock and keep the container alive.
        for target_id, (_fingerprint, session) in list(self._session_cache.items()):
            try:
                session.close()
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                self.errors.append({
                    "target_id": target_id,
                    "job": "session-close",
                    "error": f"{type(error).__name__}: {error}"[:1000],
                })
        self._session_cache.clear()
        for thread in self._threads:
            thread.join()
        for target_id in self.target_ids:
            status = self._worker_status.get(target_id)
            if status != "error":
                status = "drained" if drain else "stopped"
            self._write_worker_status(target_id, status)
            self._write_session_status(target_id, "stopped")
        self._release_owner()

    def _queue_paths(self) -> list[Path]:
        return sorted(
            self.method_dir.glob("lanes/lane-*/cases/case-*/target-replay-queue.json")
        )

    def _target_queue_root(self, target_id: str) -> Path:
        return self.method_dir / "target-queues" / _safe(target_id)

    def _target_queue_log(self, target_id: str) -> Path:
        return self._target_queue_root(target_id) / "queue.jsonl"

    def _target_cursor_path(self, target_id: str) -> Path:
        return self._target_queue_root(target_id) / "cursor.json"

    def _cursor_offset(self, target_id: str) -> int:
        if target_id in self._cursor_offsets:
            return self._cursor_offsets[target_id]
        path = self._target_cursor_path(target_id)
        try:
            cursor = read_json_object(path)
            value = cursor.get("byte_offset", 0)
            offset = max(0, int(value))
        except (OSError, TypeError, ValueError):
            offset = 0
        self._cursor_offsets[target_id] = offset
        return offset

    def _advance_cursor(
        self, target_id: str, offset: int, index: Mapping[str, object],
    ) -> None:
        self._cursor_offsets[target_id] = max(0, int(offset))
        atomic_write_json(self._target_cursor_path(target_id), {
            "schema_version": "rq1-target-queue-cursor-v1",
            "target_id": target_id,
            "byte_offset": self._cursor_offsets[target_id],
            "last_job_id": index.get("job_id"),
            "updated_at_epoch": time.time(),
        })

    def _has_target_queue_log(self, target_id: str) -> bool:
        return self._target_queue_log(target_id).is_file()

    def _next_published_job(
        self, target_id: str,
    ) -> tuple[Path, Path, dict[str, object], int] | None:
        """Read exactly one new record from a Target-owned append-only log."""
        queue_path = self._target_queue_log(target_id)
        if not queue_path.is_file():
            return None
        offset = self._cursor_offset(target_id)
        try:
            with queue_path.open("rb") as stream:
                stream.seek(offset)
                raw = stream.readline()
        except OSError:
            return None
        if not raw:
            return None
        # A producer may be between the two halves of an append. Retry on the
        # next poll instead of advancing past a partial record.
        if not raw.endswith(b"\n"):
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError):
            raise RuntimeError(
                f"invalid Target queue record at {queue_path}:{offset}"
            ) from None
        if not isinstance(value, Mapping) or value.get("target_id") != target_id:
            raise RuntimeError(
                f"Target queue record identity mismatch at {queue_path}:{offset}"
            )
        case_root = value.get("case_root")
        relative_job = value.get("job_path")
        if not isinstance(case_root, str) or not isinstance(relative_job, str):
            raise RuntimeError("Target queue record has no case/job path")
        case_path = (self.method_dir / case_root).resolve()
        job_path = (case_path / relative_job).resolve()
        if not case_path.is_relative_to(self.method_dir) \
                or not job_path.is_relative_to(case_path):
            raise RuntimeError("Target queue record escapes method directory")
        index = dict(value)
        index["_case_root"] = str(case_path)
        index["_result_path"] = value.get("result_path")
        index["_queue_manifest_path"] = value.get("queue_manifest_path")
        index["_published"] = True
        index["_queue_offset"] = offset
        index["_queue_next_offset"] = offset + len(raw)
        return queue_path, job_path, index, offset + len(raw)

    def _published_queue_pending(self, target_id: str) -> bool:
        path = self._target_queue_log(target_id)
        try:
            return path.is_file() and path.stat().st_size > self._cursor_offset(target_id)
        except OSError:
            return False

    def _result_path(
        self, queue_path: Path, index: Mapping[str, object], target_id: str,
        queue: Mapping[str, object],
    ) -> Path:
        explicit = index.get("_result_path") or index.get("result_path")
        if isinstance(explicit, str) and explicit:
            candidate = (self.method_dir / explicit).resolve()
            if candidate.is_relative_to(self.method_dir):
                return candidate
        if index.get("_published"):
            raise ValueError("formal Target queue entry has no result_path")
        target_ids = queue.get("target_ids", self.target_ids)
        target_ids = (
            list(target_ids)
            if isinstance(target_ids, (list, tuple)) else list(self.target_ids)
        )
        slot = target_ids.index(target_id) if target_id in target_ids else self.target_ids.index(target_id)
        job_id = _safe(index.get("job_id") or "job")
        case_root = index.get("_case_root")
        result_root = (
            Path(case_root)
            if isinstance(case_root, str) else queue_path.parent
        )
        return (
            result_root / "target-replay-queue" / "results"
            / f"target-{slot:03d}" / f"{job_id}.json"
        )

    def _manifest_for_entry(self, index: Mapping[str, object]) -> dict[str, object]:
        value = index.get("_queue_manifest_path") or index.get("queue_manifest_path")
        if not isinstance(value, str) or not value:
            return {}
        try:
            return read_json_object((self.method_dir / value).resolve())
        except (OSError, TypeError, ValueError):
            return {}

    def _entry_expired(self, index: Mapping[str, object]) -> bool:
        deadline = index.get("target_replay_deadline_epoch")
        if deadline is None and not index.get("_published"):
            deadline = self._manifest_for_entry(index).get(
                "target_replay_deadline_epoch"
            )
        try:
            return deadline is not None and time.time() >= float(deadline)
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _queue_expired(queue: Mapping[str, object]) -> bool:
        value = queue.get("target_replay_deadline_epoch")
        try:
            return value is not None and time.time() >= float(value)
        except (TypeError, ValueError, OverflowError):
            return False

    def _jobs_for(self, target_id: str) -> list[tuple[Path, Path, dict[str, object]]]:
        result = []
        for queue_path in self._queue_paths():
            try:
                queue = read_json_object(queue_path)
            except (OSError, TypeError, ValueError):
                continue
            if queue.get("status") not in {
                "chain-closed", "pending", "target-replay-progress",
            }:
                # Do not consume a producer snapshot while the chain is still
                # adding jobs; otherwise a worker result can be overwritten.
                continue
            jobs = queue.get("jobs", ())
            if not isinstance(jobs, (list, tuple)):
                jobs = ()
            for item in jobs:
                if not isinstance(item, Mapping):
                    continue
                states = item.get("target_states")
                if isinstance(states, Mapping):
                    # Compatibility with manifests produced by the old inline
                    # replay path. New queue-only manifests omit this mutable
                    # field completely.
                    if states.get(target_id) != "pending":
                        continue
                elif target_id not in queue.get("target_ids", self.target_ids):
                    continue
                job_path = item.get("job_path")
                if not isinstance(job_path, str):
                    continue
                candidate = queue_path.parent / job_path
                if candidate.is_file() and not self._result_path(
                    queue_path, item, target_id, queue,
                ).is_file():
                    result.append((queue_path, candidate, dict(item)))
            baseline = queue.get("baseline")
            baseline_states = (
                baseline.get("target_states")
                if isinstance(baseline, Mapping) else None
            )
            baseline_path = (
                baseline.get("job_path")
                if isinstance(baseline, Mapping) else None
            )
            if (
                isinstance(baseline_states, Mapping)
                and baseline_states.get(target_id) == "pending"
                and isinstance(baseline_path, str)
            ):
                candidate = queue_path.parent / baseline_path
                index = {
                    "job_id": "baseline",
                    "job_path": baseline_path,
                    "job_type": "baseline",
                    "state_sha256": queue.get("root_sha256"),
                    "parent_sha256": None,
                    "enqueued_at_epoch": baseline.get("enqueued_at_epoch"),
                }
                if candidate.is_file() and not self._result_path(
                    queue_path, index, target_id, queue,
                ).is_file():
                    result.append((
                        queue_path, candidate, index,
                    ))
            elif (
                not isinstance(baseline_states, Mapping)
                and target_id in queue.get("target_ids", self.target_ids)
                and isinstance(baseline_path, str)
            ):
                candidate = queue_path.parent / baseline_path
                index = {
                    "job_id": "baseline",
                    "job_path": baseline_path,
                    "job_type": "baseline",
                    "state_sha256": queue.get("root_sha256"),
                    "parent_sha256": None,
                    "enqueued_at_epoch": baseline.get("enqueued_at_epoch"),
                }
                if candidate.is_file() and not self._result_path(
                    queue_path, index, target_id, queue,
                ).is_file():
                    result.append((queue_path, candidate, index))
        return result

    @staticmethod
    def _timing(
        index: Mapping[str, object], started_epoch: float,
        started_monotonic: float, *, session_start_seconds: float | None = None,
        session_timing: Mapping[str, object] | None = None,
        target_execution_seconds: float | None = None,
        session_mode: str | None = None,
    ) -> dict[str, object]:
        finished_epoch = time.time()
        enqueued = index.get("enqueued_at_epoch")
        session_timing = (
            dict(session_timing) if isinstance(session_timing, Mapping) else {}
        )
        stage_timings = {
            "queue_wait_seconds": (
                round(max(0.0, started_epoch - float(enqueued)), 6)
                if isinstance(enqueued, (int, float)) else None
            ),
            "session_start_seconds": session_start_seconds,
            "reset_load_seconds": session_timing.get("reset_load_seconds"),
            "guest_execution_seconds": session_timing.get("guest_execution_seconds"),
            "trace_flush_read_seconds": session_timing.get(
                "trace_flush_read_seconds"
            ),
            "coverage_setup_seconds": session_timing.get("coverage_setup_seconds"),
            "target_execution_seconds": target_execution_seconds,
        }
        return {
            "enqueued_at_epoch": enqueued,
            "started_at_epoch": started_epoch,
            "finished_at_epoch": finished_epoch,
            "session_start_seconds": session_start_seconds,
            "target_execution_seconds": target_execution_seconds,
            "execution_seconds": round(
                max(0.0, time.monotonic() - started_monotonic), 6,
            ),
            "stage_timings": stage_timings,
            "session_timing_source": (
                "resident-session-response" if session_timing else
                "resident-session-required" if session_mode == "resident-session-required" else
                "adapter-callback-contract"
            ),
        }

    def _target_runner(
        self, target_id: str, config: Mapping[str, object],
        *, job_id: str | None = None,
    ) -> Callable[[BuiltProgram], object]:
        backend = config.get("backend")
        binary = config.get("binary_path")
        identity = config.get("identity_section")
        if not isinstance(backend, str) or not isinstance(identity, Mapping):
            raise ValueError("target backend identity is incomplete")
        target_recipe = config.get("target_recipe")
        target_recipe = (
            dict(target_recipe) if isinstance(target_recipe, Mapping) else {
                "target": {"backend": backend},
                "observer": {"fields": config.get("observer_fields", ())},
            }
        )
        fingerprint_payload: dict[str, object] = {
            "backend": backend,
            "binary_path": binary,
            "identity": identity,
            "session_command": config.get("session_command"),
        }
        # Resident services receive coverage routing per job.  The legacy
        # callback has no submit protocol, so keep its configured routing in
        # the compatibility fingerprint.
        embedded_resident = backend in {"unicorn", "unicorn-riscv64"}
        if not config.get("session_command") and not embedded_resident:
            fingerprint_payload["coverage_config"] = config.get("coverage_config")
        fingerprint = canonical_digest(fingerprint_payload)
        runner: Callable[..., object] | None = None

        # A Target consumer can start before the producer has published the
        # immutable runtime manifest. Retry that one-time prewarm at the first
        # job boundary; it remains outside the simulator request itself.
        if self._coverage_prewarm_status.get(target_id) == "pending-manifest":
            self._prewarm_target_runtime(target_id)

        if target_id in self._session_unavailable and target_id not in self._session_cache:
            detail = self._session_start_errors.get(
                target_id, "resident session command is unavailable",
            )
            raise ResidentSessionRequiredError(detail)

        def fallback_runner(
            artifact: BuiltProgram, *, timeout_seconds: float | None = None,
        ):
            nonlocal runner
            if runner is None:
                runner = backend_runner(
                    backend, binary if isinstance(binary, str) else None,
                    identity, target_recipe,
                    coverage_config=(
                        config.get("coverage_config")
                        if isinstance(config.get("coverage_config"), Mapping) else None
                    ),
                )
            timeout = config.get("target_timeout_seconds")
            return runner(
                artifact,
                timeout_seconds=(
                    min(float(timeout), float(timeout_seconds))
                    if isinstance(timeout, (int, float))
                    and timeout_seconds is not None else
                    float(timeout) if isinstance(timeout, (int, float)) else
                    timeout_seconds
                ),
            )

        cached = self._session_cache.get(target_id)
        resident_restart = bool(
            cached is not None
            and cached[0] == fingerprint
            and cached[1].mode == "resident-session-service"
            and getattr(cached[1], "process", None) is None
        )
        if cached is None or cached[0] != fingerprint or resident_restart:
            if cached is not None:
                if cached[0] != fingerprint:
                    try:
                        cached[1].close()
                    except Exception:
                        pass
                    session = build_target_session(
                        config, fallback_runner,
                        allow_callback=(
                            not self._stream_only
                            and not bool(config.get("session_command"))
                        ),
                    )
                else:
                    session = cached[1]
            else:
                session = build_target_session(
                    config, fallback_runner,
                    allow_callback=(
                        not self._stream_only
                        and not bool(config.get("session_command"))
                    ),
                )
            session_started = time.monotonic()
            session.start()
            self._session_start_seconds[target_id] = round(
                max(0.0, time.monotonic() - session_started), 6,
            )
            self._session_cache[target_id] = (fingerprint, session)
            self._session_status[target_id] = (session.mode, session.scope)
            self._session_generation[target_id] = (
                self._session_generation.get(target_id, 0) + 1
            )
            self._session_start_count[target_id] = (
                self._session_start_count.get(target_id, 0) + 1
            )
            self._write_session_status(target_id, "running")
        session = self._session_cache[target_id][1]

        def target_runner(artifact: BuiltProgram):
            timeout = config.get("target_timeout_seconds")
            metadata: dict[str, object] = {
                "target_id": target_id,
                "job_id": job_id,
                # Recipe/options are job input, not session identity.  A
                # resident process must be able to reset/load different
                # cases without being restarted for every recipe.
                "target_recipe": target_recipe,
            }
            coverage_config = config.get("coverage_config")
            if isinstance(coverage_config, Mapping):
                metadata["coverage_config"] = dict(coverage_config)
            return session.submit(
                artifact,
                job_id=job_id,
                metadata=metadata,
                timeout_seconds=(
                    float(timeout) if isinstance(timeout, (int, float)) else None
                ),
            )

        return target_runner

    def _process_job(
        self, target_id: str, queue_path: Path, job_path: Path,
        index: Mapping[str, object], *, next_offset: int | None = None,
    ) -> None:
        result_path = self._result_path(queue_path, index, target_id, {})
        if result_path.is_file():
            if next_offset is not None:
                self._advance_cursor(target_id, next_offset, index)
            return
        started_epoch = time.time()
        started_monotonic = time.monotonic()
        try:
            expired = self._entry_expired(index) if next_offset is not None else (
                self._queue_expired(read_json_object(queue_path))
            )
            if expired:
                self._finish(
                    queue_path, index, target_id,
                    {
                        "status": "target-gap",
                        "failure_class": "campaign-wall-clock-exhausted",
                        "deadline_censored": True,
                        "target": {
                            "status": "target-gap",
                            "differences": {
                                "target_error": "target replay deadline expired",
                            },
                        },
                    },
                    timing=self._timing(
                        index, started_epoch, started_monotonic,
                        session_mode=self._session_descriptor(target_id)[0],
                    ),
                )
            else:
                # Keep the public/mocked _execute signature stable.  The
                # worker-local start timestamps are read by the real method
                # only for timing; compatibility tests and adapters can still
                # replace _execute with the original four-argument callback.
                self._job_start[target_id] = (started_epoch, started_monotonic)
                self._execute(queue_path, job_path, index, target_id)
        except Exception as error:  # every claimed job gets a sidecar
            # A normal container pause closes resident sessions while a worker
            # may still be inside submit().  That is not a Target failure:
            # leave the byte cursor untouched so resume retries this exact job.
            # Real backend errors, which happen while the coordinator is
            # running, still become the Target-local durable gap below.
            if self._stop.is_set() and not self._result_path(
                queue_path, index, target_id, {},
            ).is_file():
                return
            self.errors.append({
                "target_id": target_id,
                "job": str(job_path),
                "error": f"{type(error).__name__}: {error}"[:1000],
            })
            failure_class = (
                "resident-session-required"
                if isinstance(error, ResidentSessionRequiredError)
                else "target-worker-error"
            )
            self._finish(
                queue_path, index, target_id, {
                    "status": "target-gap",
                    "failure_class": failure_class,
                    "reason": str(error),
                },
                timing=self._timing(
                    index, started_epoch, started_monotonic,
                    session_mode=self._session_descriptor(target_id)[0],
                ),
            )
        finally:
            self._job_start.pop(target_id, None)
            # A cursor is an acknowledgement, not a claim.  If the result
            # sidecar did not become durable, leave the offset unchanged so a
            # restart retries the job (or observes the already durable sidecar).
            if next_offset is not None and self._result_path(
                queue_path, index, target_id, {},
            ).is_file():
                self._advance_cursor(target_id, next_offset, index)

    def _worker(self, target_id: str) -> None:
        try:
            # Each Target warms its own immutable runtime on its own worker;
            # a slow cache setup cannot delay another Target's worker start.
            self._prewarm_target_runtime(target_id)
            self._prime_target_session(target_id)
            self._write_session_status(target_id, "running")
            while not self._stop.is_set():
                if self._stream_only or self._has_target_queue_log(target_id):
                    item = self._next_published_job(target_id)
                    if item is not None:
                        queue_path, job_path, index, next_offset = item
                        self._process_job(
                            target_id, queue_path, job_path, index,
                            next_offset=next_offset,
                        )
                        continue
                    if self._drain.is_set() and not self._published_queue_pending(target_id):
                        break
                    time.sleep(0.05)
                    continue

                # Compatibility path for manifests produced before the Target
                # level queue log existed.
                jobs = self._jobs_for(target_id)
                submitted = False
                for queue_path, job_path, index in jobs:
                    key = (target_id, str(job_path))
                    if key in self._claimed:
                        continue
                    self._claimed.add(key)
                    submitted = True
                    self._process_job(target_id, queue_path, job_path, index)
                if self._drain.is_set() and not self._jobs_for(target_id):
                    break
                if not submitted:
                    time.sleep(0.05)
        except Exception as error:  # queue corruption/worker failure is durable
            message = f"{type(error).__name__}: {error}"[:1000]
            self._worker_errors[target_id] = {
                "target_id": target_id, "error": message,
            }
            self.errors.append({
                "target_id": target_id, "job": "worker-loop", "error": message,
            })
            self._write_worker_status(target_id, "error", error=message)
        else:
            self._write_worker_status(
                target_id, "drained" if self._drain.is_set() else "stopped",
            )

    def _execute(
        self, queue_path: Path, job_path: Path,
        index: Mapping[str, object], target_id: str,
        *, started_epoch: float | None = None,
        started_monotonic: float | None = None,
    ) -> None:
        job = read_json_object(job_path)
        configs = job.get("target_configs")
        config = configs.get(target_id) if isinstance(configs, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError(f"target config missing: {target_id}")
        identity = config.get("identity_section")
        if not isinstance(identity, Mapping):
            raise ValueError("target backend identity is incomplete")
        current_start = self._job_start.get(target_id)
        started_epoch = (
            started_epoch if started_epoch is not None
            else current_start[0] if current_start is not None else time.time()
        )
        started_monotonic = (
            started_monotonic if started_monotonic is not None
            else current_start[1] if current_start is not None else time.monotonic()
        )
        target_runner = self._target_runner(
            target_id, config, job_id=str(index.get("job_id") or job_path.stem),
        )

        def execute_raw(artifact: BuiltProgram) -> dict[str, object]:
            """Send one artifact to the simulator and retain its raw reply."""
            started = time.monotonic()
            response = target_runner(artifact)
            safe_response = json_safe(response)
            if isinstance(safe_response, Mapping):
                observations = safe_response.get("observations")
                if observations is None and "result" in safe_response:
                    observations = safe_response.get("result")
                if observations is None and any(
                    name in safe_response
                    for name in ("outcome", "exit_code", "contract_error", "backend")
                ):
                    observations = safe_response
            else:
                observations = safe_response
            observations = (
                list(observations) if isinstance(observations, (list, tuple))
                else [observations] if isinstance(observations, Mapping) else []
            )
            session = self._session_cache.get(target_id)
            session_object = session[1] if session else None
            return {
                "status": "raw-recorded",
                "observations": observations,
                "response": safe_response,
                "coverage": json_safe(
                    getattr(session_object, "last_coverage", {})
                    if session_object is not None else {}
                ),
                "session_timing": json_safe(
                    getattr(session_object, "last_timing", {})
                    if session_object is not None else {}
                ),
                "execution_seconds": round(
                    max(0.0, time.monotonic() - started), 6,
                ),
            }

        def finish(
            result: Mapping[str, object], target_execution_seconds: float,
        ) -> None:
            session = self._session_cache.get(target_id)
            session_object = session[1] if session else None
            session_timing = (
                getattr(session_object, "last_timing", {})
                if session_object is not None else {}
            )
            session_mode = self._session_descriptor(target_id)[0]
            coverage = (
                getattr(session_object, "last_coverage", {})
                if session_object is not None else {}
            )
            result_value = dict(result)
            if session_mode == "resident-session-service":
                result_value["coverage_evidence"] = (
                    dict(coverage) if isinstance(coverage, Mapping) and coverage
                    else {
                        "status": "not-reported",
                        "reason": "resident-session-did-not-report-coverage",
                    }
                )
            self._finish(
                queue_path, index, target_id, result_value,
                timing=self._timing(
                    index, started_epoch, started_monotonic,
                    session_start_seconds=self._session_start_seconds.get(target_id),
                    session_timing=session_timing,
                    target_execution_seconds=target_execution_seconds,
                    session_mode=session_mode,
                ),
            )

        artifact_gap = job.get("artifact_gap")
        if not isinstance(artifact_gap, str) or not artifact_gap:
            if job.get("job_type") == "baseline":
                artifact_gap = (
                    None if isinstance(job.get("artifact_record"), Mapping)
                    else "baseline-artifact-record-missing"
                )
            else:
                records = job.get("artifact_records")
                missing = [
                    name for name in ("reference_original", "reference_variant")
                    if not isinstance(records.get(name), Mapping)
                ] if isinstance(records, Mapping) else [
                    "reference_original", "reference_variant",
                ]
                artifact_gap = (
                    "reference-pair-artifacts-incomplete:" + ",".join(missing)
                    if missing else None
                )
        if artifact_gap:
            finish({
                "status": "target-gap",
                "failure_class": "artifact-gap",
                "reason": artifact_gap,
                "target_attempted": False,
                "target": {
                    "status": "target-gap",
                    "differences": {"target_error": artifact_gap},
                },
            }, 0.0)
            return

        if job.get("job_type") == "baseline":
            artifact_record = job.get("artifact_record")
            if not isinstance(artifact_record, Mapping):
                raise ValueError("baseline target job has no artifact record")
            artifact = _built_program(artifact_record)
            target_started = time.monotonic()
            raw = execute_raw(artifact)
            baseline_result = {
                **raw,
                "target": {
                    "status": "raw-recorded",
                    "observations": raw["observations"],
                },
                "comparison_pending": True,
            }
            finish(baseline_result, max(0.0, time.monotonic() - target_started))
            return
        artifact_records = job.get("artifact_records")
        if not isinstance(artifact_records, Mapping):
            raise ValueError("candidate target job has no simulator artifacts")
        original_record = artifact_records.get("reference_original")
        variant_record = artifact_records.get("reference_variant")
        if not isinstance(original_record, Mapping) or not isinstance(variant_record, Mapping):
            raise ValueError("candidate target job simulator artifacts are incomplete")
        target_started = time.monotonic()
        original_raw = execute_raw(_built_program(original_record))
        variant_raw = execute_raw(_built_program(variant_record))
        result = {
            "status": "raw-recorded",
            "raw_observations": {
                "original": original_raw["observations"],
                "variant": variant_raw["observations"],
            },
            "raw_executions": {
                "original": original_raw,
                "variant": variant_raw,
            },
            "target": {
                "status": "raw-recorded",
                "observations": {
                    "original": original_raw["observations"],
                    "variant": variant_raw["observations"],
                },
            },
            "artifacts": {
                "reference_original": dict(original_record),
                "reference_variant": dict(variant_record),
            },
            "comparison_pending": True,
        }
        finish(result, max(0.0, time.monotonic() - target_started))

    def _update_target_progress(
        self, target_id: str, job_id: str, record: Mapping[str, object],
    ) -> None:
        """Update one O(1)-read aggregate for runner observability."""
        root = self._target_queue_root(target_id)
        progress_path = root / "progress.json"
        try:
            progress = read_json_object(progress_path) if progress_path.is_file() else {}
        except (OSError, TypeError, ValueError):
            progress = {}
        try:
            state = read_json_object(root / "state.json")
        except (OSError, TypeError, ValueError):
            state = {}
        counts = dict(progress.get("counts", {})) \
            if isinstance(progress.get("counts"), Mapping) else {}
        for name, value in _progress_delta(record, baseline=job_id == "baseline").items():
            counts[name] = int(counts.get(name, 0) or 0) + value
        status = record.get("status")
        status_name = status if isinstance(status, str) else "other"
        status_counts = dict(progress.get("status_counts", {})) \
            if isinstance(progress.get("status_counts"), Mapping) else {}
        status_counts[status_name] = int(status_counts.get(status_name, 0) or 0) + 1
        published = state.get("published_count", 0)
        try:
            published = max(0, int(published))
        except (TypeError, ValueError, OverflowError):
            published = 0
        completed = int(progress.get("completed_count", 0) or 0) + 1
        atomic_write_json(progress_path, {
            "schema_version": "rq1-target-queue-progress-v1",
            "target_id": target_id,
            "published_count": published,
            "completed_count": completed,
            "pending_count": max(0, published - completed),
            "record_count": int(progress.get("record_count", 0) or 0)
            + int(job_id != "baseline"),
            "baseline_count": int(progress.get("baseline_count", 0) or 0)
            + int(job_id == "baseline"),
            "counts": counts,
            "status_counts": status_counts,
            "last_job_id": job_id,
            "cursor_byte_offset": self._cursor_offsets.get(target_id, 0),
            "updated_at_epoch": time.time(),
        })

    def _finish(
        self, queue_path: Path, index: Mapping[str, object],
        target_id: str, result: Mapping[str, object],
        *, timing: Mapping[str, object] | None = None,
    ) -> None:
        job_id = str(index.get("job_id") or "job")
        case_root_value = index.get("_case_root")
        case_root = (
            Path(case_root_value)
            if isinstance(case_root_value, str) else queue_path.parent
        )
        # Formal stream entries carry their result path and target slot.  Do
        # not read the producer manifest or any other Target's state.
        queue = {} if index.get("_published") else self._manifest_for_entry(index)
        if not queue and not index.get("_published"):
            queue = read_json_object(queue_path)
        target_ids = queue.get("target_ids", self.target_ids)
        target_ids = (
            list(target_ids)
            if isinstance(target_ids, (list, tuple)) else list(self.target_ids)
        )
        slot_value = index.get("target_slot")
        slot = (
            int(slot_value) if type(slot_value) is int else
            target_ids.index(target_id)
            if target_id in target_ids else self.target_ids.index(target_id)
        )
        if index.get("_published"):
            result_path = self._result_path(queue_path, index, target_id, queue)
            result_dir = result_path.parent
        else:
            result_dir = (
                case_root / "target-replay-queue" / "results" / f"target-{slot:03d}"
            )
            result_path = result_dir / f"{_safe(job_id)}.json"
        result_dir.mkdir(parents=True, exist_ok=True)
        record = (
            {
                **dict(result),
                "target_id": target_id,
                "target_result_status": result.get("status", "target-gap"),
                "target_attempted": True,
            }
            if job_id == "baseline" else
            _target_record(index, target_id, result)
        )
        timing_record = dict(timing or {})
        timing_record["persist_started_at_epoch"] = time.time()
        record["timing"] = timing_record
        session_mode, session_scope = self._session_descriptor(target_id)
        timing_path = result_dir / f"{_safe(job_id)}.timing.json"
        timing_record["timing_sidecar"] = str(
            timing_path.relative_to(
                self.method_dir if index.get("_published") else case_root
            )
        )

        progress_path = (
            self._target_queue_root(target_id) / "progress.json"
            if index.get("_published") else result_dir / "progress.json"
        )
        try:
            progress = read_json_object(progress_path) if progress_path.is_file() else {}
        except (OSError, TypeError, ValueError):
            progress = {}
        progress_counts = dict(progress.get("counts", {})) \
            if isinstance(progress.get("counts"), Mapping) else {}
        delta = _progress_delta(record, baseline=job_id == "baseline")
        for name, value in delta.items():
            progress_counts[name] = int(progress_counts.get(name, 0) or 0) + value
        status_value = record.get("status")
        status_name = status_value if isinstance(status_value, str) else "other"
        status_counts = dict(progress.get("status_counts", {})) \
            if isinstance(progress.get("status_counts"), Mapping) else {}
        status_counts[status_name] = int(status_counts.get(status_name, 0) or 0) + 1
        progress_sequence = int(progress.get("progress_sequence", -1) or -1) + 1
        atomic_write_json(result_path, {
            "schema_version": "rq1-target-replay-result-v1",
            "target_id": target_id,
            "job_id": job_id,
            "pair_index": index.get("pair_index"),
            "progress_sequence": progress_sequence,
            "counts": progress_counts,
            "result": dict(result),
            "record": record,
            "simulator_session_mode": session_mode,
            "backend_session_scope": session_scope,
            "queue_manifest_immutable": True,
            "timing": timing_record,
        })
        if not index.get("_published"):
            atomic_write_json(result_dir / "progress.json", {
                "schema_version": "rq1-target-progress-v1",
                "target_id": target_id,
                "progress_sequence": progress_sequence,
                "record_count": int(progress.get("record_count", 0) or 0)
                + int(job_id != "baseline"),
                "baseline_count": int(progress.get("baseline_count", 0) or 0)
                + int(job_id == "baseline"),
                "counts": progress_counts,
                "status_counts": status_counts,
                "updated_at_epoch": time.time(),
            })
        timing_record["persisted_at_epoch"] = time.time()
        timing_record["result_persistence_seconds"] = round(
            max(0.0, timing_record["persisted_at_epoch"]
                - timing_record["persist_started_at_epoch"]), 6,
        )
        timing_record["stage_timings"] = {
            **(
                dict(timing_record.get("stage_timings", {}))
                if isinstance(timing_record.get("stage_timings"), Mapping) else {}
            ),
            "result_persistence_seconds": timing_record["result_persistence_seconds"],
        }
        # The result is published before the cursor advances.  Keep the exact
        # persistence interval in a tiny sidecar so the main result need not be
        # rewritten (and the handoff remains one-way and cheap).
        atomic_write_json(timing_path, {
            "schema_version": "rq1-target-job-timing-v1",
            "target_id": target_id,
            "job_id": job_id,
            "timing": timing_record,
        })
        try:
            self._update_target_progress(target_id, job_id, record)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            # Aggregate progress is an observability sidecar.  A failed
            # aggregate write must not turn a durable result into a retry or
            # block the Target cursor.
            self.errors.append({
                "target_id": target_id,
                "job": f"{job_id}:progress",
                "error": f"{type(error).__name__}: {error}"[:1000],
            })


__all__ = ["TargetQueueCoordinator"]

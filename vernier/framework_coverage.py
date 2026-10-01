"""Framework target coverage: one small adapter shared by both Ours routes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

from analysis.elf_features import decode_elf, metric_registry_for_profile
from analysis.simulator_coverage import (
    BATCH_SCHEMA,
    SCHEMA,
    aggregate_simulator_coverage, _pcov_unit_digest,
    _nonnegative_count, _pc_value, _summarize_pcov_run, summarize_experiment_union,
    summarize_source_experiment_union, summarize_simulator_coverage,
    target_source_coverage,
)
from analysis.rv_instruction_coverage import (
    RV_INSTRUCTION_METRICS,
    aggregate_opcode_catalog_summary,
    aggregate_summary as aggregate_rv_instruction_coverage,
    summarize_opcode_catalog_target_rows,
)
from analysis.source_coverage import (
    _error_summary, _merge_lcov_profiles, collect_source_coverage_batch,
    coverage_input_files,
)
from framework._util import atomic_write_json, read_json_object, sha256_file
from framework.execution_identity import TargetBinaryIdentity


_DOTNET_COVERAGE_COMPLETED: set[str] = set()
_DOTNET_COVERAGE_COMPLETED_LOCK = threading.Lock()
_DOTNET_COVERAGE_INFLIGHT: set[str] = set()
_DOTNET_COVERAGE_ATTEMPTED: set[str] = set()
_DOTNET_COVERAGE_THREADS: dict[str, threading.Thread] = {}
# Immutable target runtime setup is worker-local state.  The key deliberately
# excludes raw_dir: raw/trace/TMPDIR belong to one job, while the instrumented
# binary and its runtime tree belong to the Target worker.
_PREWARMED_COVERAGE_TARGETS: dict[
    tuple[str, str], tuple[Path, TargetBinaryIdentity]
] = {}


class DotnetCoverageServer:
    """One batch-scoped collector for the resident Renode Target worker."""

    def __init__(self, config: Mapping[str, object]):
        self.config = _effective_coverage_config(config)
        self.process: subprocess.Popen[bytes] | None = None
        self.launcher: Path | None = None
        self.session_id = ""
        self.report: Path | None = None
        self.log_path: Path | None = None
        self._environment: dict[str, str] = {}

    def _server_config(self) -> dict[str, object]:
        raw_value = self.config.get("coverage_batch_raw_dir")
        if raw_value in (None, ""):
            raw_value = self.config.get("raw_dir")
        raw_dir = _path(raw_value)
        return {**self.config, "raw_dir": str(raw_dir)}

    def _write_status(self, status: str, **fields: object) -> None:
        if self.report is None:
            return
        path = self.report.parent / "dotnet-coverage-server.json"
        try:
            atomic_write_json(
                path,
                {
                    "schema_version": "rq1-dotnet-coverage-server-v1",
                    "status": status,
                    "session_id": self.session_id or None,
                    "report": str(self.report),
                    "log": str(self.log_path) if self.log_path else None,
                    **fields,
                },
            )
        except OSError:
            pass

    def start(self) -> str:
        if self.process is not None:
            return self.session_id
        source = self.config.get("source_coverage")
        source = source if isinstance(source, Mapping) else {}
        launcher_value = source.get("launcher")
        if not isinstance(launcher_value, str) or not launcher_value:
            raise FileNotFoundError("dotnet coverage launcher is missing")
        self.launcher = _path(launcher_value)
        if not self.launcher.is_file():
            raise FileNotFoundError(str(self.launcher))
        server_config = self._server_config()
        raw_dir = _path(server_config.get("raw_dir"))
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.report = raw_dir / "renode.cobertura.xml"
        self.report.unlink(missing_ok=True)
        self.log_path = raw_dir / "dotnet-coverage-server.log"
        self.session_id = f"rq1-{os.getpid()}-{uuid.uuid4().hex}"
        identity_binary = _path(server_config.get("identity_binary_path"))
        runner_binary, runner_env = _coverage_target(server_config, identity_binary)
        include_files = [runner_binary]
        infrastructure = runner_binary.with_name("Infrastructure.dll")
        if infrastructure.is_file():
            include_files.append(infrastructure)
        runtime = os.environ.get("RQ1_DOTNET_RUNTIME", "")
        self._environment = os.environ.copy()
        self._environment.update({
            # dotnet-coverage uses a Unix-domain socket for ``connect``;
            # long run-root paths exceed Linux's 108-byte socket limit.
            # Reports and guest traces remain run-owned; only IPC uses /tmp.
            "TMPDIR": "/tmp",
            "DOTNET_ROOT": runtime or self._environment.get("DOTNET_ROOT", ""),
        })
        if runtime:
            self._environment["PATH"] = os.pathsep.join(
                [runtime, self._environment.get("PATH", "")]
            )
        for name in ("RQ1_TARGET_LOGICAL_BINARY_PATH",):
            if name in runner_env:
                self._environment[name] = runner_env[name]
        raw_log = self.log_path.open("ab")
        command = [
            str(self.launcher), "collect", "--server-mode",
            "--session-id", self.session_id,
            "--include-files", ",".join(map(str, include_files)),
            "--output", str(self.report), "--output-format", "cobertura",
            "--nologo",
        ]
        try:
            self.process = subprocess.Popen(
                command, cwd=runner_binary.parent, env=self._environment,
                stdout=raw_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            raw_log.close()
        self._write_status("starting", pid=self.process.pid if self.process else None)
        deadline = time.monotonic() + 10.0
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("dotnet-coverage server exited during startup")
                text = self.log_path.read_text(encoding="utf-8", errors="replace")
                if "SessionId:" in text:
                    self._write_status("running", pid=self.process.pid)
                    return self.session_id
                time.sleep(0.05)
            raise TimeoutError("dotnet-coverage server startup timed out")
        except Exception:
            self.stop()
            raise

    def stop(self) -> dict[str, object]:
        process = self.process
        self.process = None
        result: dict[str, object] = {"status": "not-running"}
        if process is None:
            return result
        error: str | None = None
        try:
            if process.poll() is None and self.launcher is not None:
                shutdown = subprocess.run(
                    [str(self.launcher), "shutdown", self.session_id,
                     "--nologo", "--timeout", "60000"],
                    cwd=str(self.report.parent if self.report else Path.cwd()),
                    env=self._environment, capture_output=True,
                    timeout=90, check=False,
                )
                if shutdown.returncode:
                    error = f"shutdown-exit:{shutdown.returncode}"
            process.wait(timeout=90)
        except Exception as exc:  # preserve raw report and expose the gap
            error = f"{type(exc).__name__}: {exc}"
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except OSError:
                    pass
        status = "shutdown-gap" if error else "stopped"
        result.update({
            "status": status,
            "returncode": process.returncode,
            "error": error,
            "report_observed": _dotnet_report_observed(str(self.report))
            if self.report else False,
        })
        self._write_status(
            status, pid=process.pid,
            **{key: value for key, value in result.items() if key != "status"},
        )
        return result


def _dotnet_coverage_key(env: Mapping[str, str] | None) -> str | None:
    if not isinstance(env, Mapping):
        return None
    value = env.get("RQ1_DOTNET_COVERAGE_OUTPUT")
    return str(value) if isinstance(value, str) and value else None


def _dotnet_report_observed(path: str | None) -> bool:
    """Accept only a non-empty Cobertura report, not dotnet's empty marker."""
    if not path:
        return False
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return False
    return "<package " in text or "<package>" in text


def _dotnet_coverage_already_completed(key: str | None) -> bool:
    if key is None:
        return False
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        return key in _DOTNET_COVERAGE_COMPLETED


def _dotnet_coverage_already_attempted(key: str | None) -> bool:
    """Avoid replaying a known collector gap on every target invocation."""
    if key is None:
        return False
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        if (
            key in _DOTNET_COVERAGE_COMPLETED
            or key in _DOTNET_COVERAGE_ATTEMPTED
            or key in _DOTNET_COVERAGE_INFLIGHT
        ):
            return True
    try:
        status = read_json_object(_dotnet_coverage_status_path(Path(key))).get("status")
    except (OSError, TypeError, ValueError):
        return False
    return status in {"gap", "recorded", "client-recorded"}


def _mark_dotnet_coverage_attempted(key: str | None) -> None:
    if key is None:
        return
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        _DOTNET_COVERAGE_ATTEMPTED.add(key)


def _mark_dotnet_coverage_completed(key: str | None) -> None:
    if key is None:
        return
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        _DOTNET_COVERAGE_COMPLETED.add(key)


def _has_terminal_observation(value: object) -> bool:
    """Return whether a normal (non-collector) target result is usable."""
    rows = value if isinstance(value, (list, tuple)) else (value,)
    for item in rows:
        if isinstance(item, str):
            return bool(item)
        outcome = item.get("outcome") if isinstance(item, Mapping) else getattr(
            item, "outcome", None,
        )
        if outcome not in {None, "timeout", "gap", "runner-contract-gap", "unavailable"}:
            return True
    return False


def _dotnet_coverage_timeout_seconds(config: Mapping[str, object]) -> float:
    """Return the independent budget for the source-coverage side run."""
    value = config.get("coverage_timeout_seconds")
    if value in (None, ""):
        # A managed Renode replay can finish the guest mailbox before the
        # .NET profiler flushes its report. Keep this side-run budget
        # independent from the normal Target budget; 180 seconds is too
        # short for long Program-Full capsules on a loaded host.
        value = os.environ.get("RQ1_DOTNET_COVERAGE_TIMEOUT_SECONDS", "300")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        value = 180.0
    return min(1800.0, max(1.0, value))


def _dotnet_coverage_wait_seconds(config: Mapping[str, object]) -> float:
    """Cover backend execution and the Cobertura flush window."""
    return 2.0 * _dotnet_coverage_timeout_seconds(config) + 65.0


def _dotnet_coverage_output(config: Mapping[str, object]) -> Path | None:
    raw_value = config.get("raw_dir")
    if raw_value in (None, ""):
        return None
    return _path(raw_value) / "renode.cobertura.xml"


def _dotnet_coverage_status_path(output: Path) -> Path:
    return output.parent / "dotnet-coverage-task.json"


def _dotnet_coverage_task_status(output: Path) -> str | None:
    try:
        status = read_json_object(_dotnet_coverage_status_path(output)).get("status")
    except (OSError, TypeError, ValueError):
        return None
    return status if isinstance(status, str) else None


def _write_dotnet_coverage_status(
    output: Path, status: str, *, started: float, **fields: object,
) -> None:
    try:
        atomic_write_json(
            _dotnet_coverage_status_path(output),
            {
                "schema_version": "rq1-dotnet-coverage-task-v1",
                "status": status,
                "output": str(output),
                "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
                **fields,
            },
        )
    except OSError:
        # A collector failure must remain a coverage gap, never a target gap.
        pass


def _dotnet_coverage_client_recorded(output: Path) -> bool:
    try:
        status = read_json_object(_dotnet_coverage_status_path(output)).get("status")
    except (OSError, TypeError, ValueError):
        return False
    return status in {"recorded", "client-recorded"}


def _run_dotnet_coverage_task(
    key: str,
    output: Path,
    timeout_seconds: float,
    config: Mapping[str, object],
    backend: str,
    executable: Path,
    profile_id: str | None,
    input_id: str | None,
    env: dict[str, str],
    binary: str | None,
) -> None:
    """Run a coverage pass while preserving the normal target observation."""
    from framework.direct_runner import run_backend

    started = time.monotonic()
    _mark_dotnet_coverage_attempted(key)
    _write_dotnet_coverage_status(
        output, "running", started=started, timeout_seconds=timeout_seconds,
    )
    try:
        result: object = run_backend(
            backend, executable, profile_id=profile_id, input_id=input_id,
            runner_env=env, backend_binary_path=binary,
            timeout_seconds=timeout_seconds,
        )
        session_id = env.get("RQ1_DOTNET_COVERAGE_SESSION_ID")
        report_observed = _dotnet_report_observed(str(output))
        client_completed = bool(session_id) and _has_terminal_observation(result)
        observed = report_observed or client_completed
        observations = result if isinstance(result, (list, tuple)) else (result,)
        observation_outcomes = []
        observation_errors = []
        for item in observations:
            if isinstance(item, Mapping):
                outcome = item.get("outcome")
                error = item.get("contract_error")
            else:
                outcome = getattr(item, "outcome", None)
                error = getattr(item, "contract_error", None)
            observation_outcomes.append(outcome)
            if isinstance(error, str) and error:
                observation_errors.append(error)
        if observed:
            _mark_dotnet_coverage_completed(key)
        _write_dotnet_coverage_status(
            output,
            "client-recorded" if client_completed and not report_observed
            else "recorded" if observed else "gap",
            started=started,
            report_observed=report_observed,
            session_id=session_id,
            client_completed=client_completed,
            observation_outcomes=observation_outcomes,
            observation_errors=observation_errors,
        )
    except Exception as error:  # pragma: no cover - exercised by real adapters
        _write_dotnet_coverage_status(
            output, "gap", started=started, report_observed=False,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        with _DOTNET_COVERAGE_COMPLETED_LOCK:
            _DOTNET_COVERAGE_INFLIGHT.discard(key)
            _DOTNET_COVERAGE_THREADS.pop(key, None)


def _schedule_dotnet_coverage(
    *,
    key: str | None,
    output: Path | None,
    config: Mapping[str, object],
    backend: str,
    executable: Path,
    profile_id: str | None,
    input_id: str | None,
    env: dict[str, str],
    binary: str | None,
) -> bool:
    """Schedule at most one .NET coverage task for a case-local output path."""
    if key is None or output is None:
        return False
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        if (
            key in _DOTNET_COVERAGE_COMPLETED
            or key in _DOTNET_COVERAGE_ATTEMPTED
            or key in _DOTNET_COVERAGE_INFLIGHT
        ):
            return False
        _DOTNET_COVERAGE_INFLIGHT.add(key)
        worker = threading.Thread(
            target=_run_dotnet_coverage_task,
            args=(
                key, output, _dotnet_coverage_timeout_seconds(config), config,
                backend, executable, profile_id, input_id, dict(env), binary,
            ),
            name="rq1-dotnet-coverage",
            daemon=not bool(env.get("RQ1_DOTNET_COVERAGE_SESSION_ID")),
        )
        _DOTNET_COVERAGE_THREADS[key] = worker
    try:
        worker.start()
    except RuntimeError:
        with _DOTNET_COVERAGE_COMPLETED_LOCK:
            _DOTNET_COVERAGE_INFLIGHT.discard(key)
            _DOTNET_COVERAGE_THREADS.pop(key, None)
        raise
    return True


def wait_for_dotnet_coverage(
    config: Mapping[str, object] | None, timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Wait for the case-local .NET coverage task before finalizing its case."""
    if not isinstance(config, Mapping):
        return {"status": "not-configured"}
    output = _dotnet_coverage_output(config)
    source = config.get("source_coverage")
    if output is None or not isinstance(source, Mapping) or source.get("collector") != "dotnet":
        return {"status": "not-configured"}
    key = str(output)
    with _DOTNET_COVERAGE_COMPLETED_LOCK:
        worker = _DOTNET_COVERAGE_THREADS.get(key)
    if worker is None:
        task_status = _dotnet_coverage_task_status(output)
        wait = (
            _dotnet_coverage_wait_seconds(config)
            if timeout_seconds is None else max(0.0, float(timeout_seconds))
        )
        deadline = time.monotonic() + wait
        while task_status == "running" and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            task_status = _dotnet_coverage_task_status(output)
        wait_timeout = task_status == "running"
        observed = _dotnet_report_observed(key) or _dotnet_coverage_client_recorded(output)
        if wait_timeout:
            status = "running"
        elif observed or task_status in {"recorded", "client-recorded"}:
            status = "recorded"
        elif task_status == "gap":
            status = "gap"
        else:
            status = "not-started"
        return {
            "status": status, "output": str(output),
            "report_observed": observed, "wait_timeout": wait_timeout,
        }
    wait = (
        _dotnet_coverage_wait_seconds(config)
        if timeout_seconds is None else max(0.0, float(timeout_seconds))
    )
    worker.join(wait)
    alive = worker.is_alive()
    observed = _dotnet_report_observed(key) or _dotnet_coverage_client_recorded(output)
    return {
        "status": "running" if alive else ("recorded" if observed else "gap"),
        "output": str(output), "report_observed": observed,
        "wait_timeout": alive,
    }


def _path(value: object, *, base: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        dependency_root = os.environ.get("RQ1_DEPS")
        if dependency_root:
            try:
                return Path(dependency_root) / path.relative_to("/path/to/deps")
            except ValueError:
                pass
        run_root_value = os.environ.get("RQ1_RUN_ROOT")
        if run_root_value:
            run_root = Path(run_root_value)
            prefix = Path("/opt/runs") / run_root.name
            try:
                return run_root / path.relative_to(prefix)
            except ValueError:
                pass
        try:
            if path.exists():
                return path
        except OSError:
            pass
        return path
    return (base or Path.cwd()) / path


def _effective_coverage_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    nested = config.get("coverage_config")
    return {
        **dict(config),
        **(dict(nested) if isinstance(nested, Mapping) else {}),
    }


def _base_target_identity(config: Mapping[str, object]) -> TargetBinaryIdentity:
    config = _effective_coverage_config(config)
    identity_binary = _path(config.get("identity_binary_path"))
    if not identity_binary.is_file():
        raise FileNotFoundError("framework coverage target binary is missing")
    identity = TargetBinaryIdentity.from_dict(read_json_object(
        identity_binary.with_name("target-identity.json"),
    ))
    if sha256_file(identity_binary) != identity.binary_sha256:
        raise ValueError("framework coverage identity binary does not match its manifest")
    target = config.get("target")
    expected_binary_sha = config.get("binary_sha256")
    expected_identity_digest = config.get("identity_digest")
    if isinstance(target, Mapping):
        expected_binary_sha = expected_binary_sha or target.get("binary_sha256")
        expected_identity_digest = expected_identity_digest or target.get("identity_digest")
    if isinstance(expected_binary_sha, str) and identity.binary_sha256 != expected_binary_sha:
        raise ValueError("framework coverage identity does not match the declared target SHA")
    if isinstance(expected_identity_digest, str) \
            and identity.identity_digest != expected_identity_digest:
        raise ValueError("framework coverage identity does not match the declared target digest")
    expected_backend = config.get("backend")
    if expected_backend not in (None, identity.backend):
        raise ValueError("framework coverage target backend does not match its identity")
    return identity


def coverage_target_identity(config: Mapping[str, object]) -> TargetBinaryIdentity:
    """Resolve the exact identity written beside the coverage execution binary."""
    config = _effective_coverage_config(config)
    binary = _path(config.get("binary_path"))
    if not binary.is_file():
        raise FileNotFoundError("framework coverage target binary is missing")
    identity = _base_target_identity(config)
    binary_sha256 = sha256_file(binary)
    declared_sha256 = config.get("coverage_binary_sha256")
    target = config.get("target")
    if not isinstance(declared_sha256, str) and isinstance(target, Mapping):
        declared_sha256 = target.get("coverage_binary_sha256")
    if declared_sha256 not in (None, "") and str(declared_sha256).lower() != binary_sha256:
        raise ValueError("framework coverage binary does not match its declared SHA")
    return replace(identity, binary_sha256=binary_sha256)


def coverage_observation_identity(config: Mapping[str, object]) -> TargetBinaryIdentity:
    """Resolve identity of the executable that produces Target observations."""
    config = _effective_coverage_config(config)
    source = config.get("source_coverage")
    collector = source.get("collector") if isinstance(source, Mapping) else None
    if collector == "dotnet":
        return _base_target_identity(config)
    return coverage_target_identity(config)


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize one immutable runtime file without serializing execution."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return
    except FileExistsError:
        return
    except OSError:
        pass
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prewarm_coverage_target(
    config: Mapping[str, object],
) -> tuple[Path, TargetBinaryIdentity]:
    """Prepare one Target's immutable coverage runtime once per worker.

    ``raw_dir`` is intentionally absent from the cache key.  It is a job-local
    output path and must never make one Target's coverage setup depend on a
    different case or Target.
    """
    config = _effective_coverage_config(config)
    binary = _path(config.get("binary_path"))
    identity_binary = _path(config.get("identity_binary_path"))
    if not binary.is_file() or not identity_binary.is_file():
        raise FileNotFoundError("framework coverage target binary is missing")
    cache_value = config.get("coverage_binary_cache")
    cache_dir = (
        _path(cache_value).resolve()
        if isinstance(cache_value, (str, Path)) and str(cache_value) else None
    )
    key = (str(binary.resolve()), str(cache_dir) if cache_dir is not None else "")
    cached = _PREWARMED_COVERAGE_TARGETS.get(key)
    if cached is not None and cached[0].is_file():
        return cached

    coverage_identity = coverage_target_identity(config)
    if cache_dir is None:
        result = (binary, coverage_identity)
        _PREWARMED_COVERAGE_TARGETS[key] = result
        return result

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_binary = cache_dir / binary.name
    if cached_binary.is_file() and sha256_file(cached_binary) != coverage_identity.binary_sha256:
        cached_binary.unlink()
    _link_or_copy(binary, cached_binary)

    if binary.suffix.lower() == ".dll":
        sidecars = (
            path for path in binary.parent.rglob("*")
            if path.is_file()
            and path.relative_to(binary.parent)
            not in {Path(binary.name), Path("target-identity.json")}
        )
        for sidecar in sidecars:
            relative = sidecar.relative_to(binary.parent)
            destination = cache_dir / relative
            if destination.is_file() and sha256_file(destination) != sha256_file(sidecar):
                destination.unlink()
            _link_or_copy(sidecar, destination)
    else:
        for sidecar in binary.parent.glob("*.so*"):
            if sidecar.is_file() and sidecar.name != binary.name:
                destination = cache_dir / sidecar.name
                if destination.is_file() and sha256_file(destination) != sha256_file(sidecar):
                    destination.unlink()
                _link_or_copy(sidecar, destination)

    atomic_write_json(cache_dir / "target-identity.json", coverage_identity.to_dict())
    result = (cached_binary, coverage_identity)
    _PREWARMED_COVERAGE_TARGETS[key] = result
    return result


def _coverage_target(config: Mapping[str, object], executable: Path) -> tuple[Path, dict[str, str]]:
    config = _effective_coverage_config(config)
    binary = _path(config.get("binary_path"))
    identity_binary = _path(config.get("identity_binary_path"))
    if not binary.is_file() or not identity_binary.is_file():
        raise FileNotFoundError("framework coverage target binary is missing")
    source_binary, coverage_identity = prewarm_coverage_target(config)
    binary_sha256 = coverage_identity.binary_sha256
    cache_value = config.get("coverage_binary_cache")

    # A .NET coverage build is a complete runtime tree, not a single shared
    # object. Re-copying its ~800 sidecars into every case consumes the case
    # wall-clock budget before Renode is launched. Reuse the run-level cache
    # directly and only refresh its tiny identity manifest.
    reuse_cached_runtime = (
        binary.suffix.lower() == ".dll"
        and isinstance(cache_value, (str, Path))
        and str(cache_value)
        and source_binary.parent == _path(cache_value)
    )
    if isinstance(cache_value, (str, Path)) and str(cache_value):
        runner_binary = source_binary
    else:
        target_dir = executable.parent / ".coverage-target" / coverage_identity.backend
        target_dir.mkdir(parents=True, exist_ok=True)
        runner_binary = target_dir / binary.name
        if runner_binary.is_file() and sha256_file(runner_binary) != binary_sha256:
            runner_binary.unlink()
        if not runner_binary.is_file():
            try:
                os.link(source_binary, runner_binary)
            except OSError:
                shutil.copy2(source_binary, runner_binary)
    if not (isinstance(cache_value, (str, Path)) and str(cache_value)):
        atomic_write_json(runner_binary.with_name("target-identity.json"), coverage_identity.to_dict())
    if binary.suffix.lower() == ".dll" and not reuse_cached_runtime:
        for sidecar in source_binary.parent.rglob("*"):
            relative = sidecar.relative_to(source_binary.parent)
            if not sidecar.is_file() or relative in {Path(binary.name), Path("target-identity.json")}:
                continue
            target_sidecar = target_dir / relative
            target_sidecar.parent.mkdir(parents=True, exist_ok=True)
            if target_sidecar.is_file() and sha256_file(target_sidecar) != sha256_file(sidecar):
                target_sidecar.unlink()
            if not target_sidecar.is_file():
                try:
                    os.link(sidecar, target_sidecar)
                except OSError:
                    shutil.copy2(sidecar, target_sidecar)
    elif not reuse_cached_runtime:
        # ELF launchers such as RVVM resolve sibling shared objects from the
        # launcher's directory. Keep the case-local runner self-contained.
        for sidecar in source_binary.parent.glob("*.so*"):
            if sidecar.name == binary.name or not sidecar.is_file():
                continue
            target_sidecar = runner_binary.parent / sidecar.name
            if target_sidecar.is_file() and sha256_file(target_sidecar) != sha256_file(sidecar):
                target_sidecar.unlink()
            if not target_sidecar.is_file():
                try:
                    os.link(sidecar, target_sidecar)
                except OSError:
                    shutil.copy2(sidecar, target_sidecar)

    source = config.get("source_coverage")
    source = source if isinstance(source, Mapping) else {}
    raw_value = config.get("raw_dir")
    raw_dir = _path(raw_value) if raw_value not in (None, "") else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = raw_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    collector = str(source.get("collector", "gcov"))
    env = (
        {"GCOV_PREFIX": str(raw_dir),
         "GCOV_PREFIX_STRIP": str(source.get("gcov_prefix_strip", 2))}
        if raw_dir is not None and collector in {"gcov", "lcov"} else
        {"LLVM_PROFILE_FILE": str(raw_dir / "%m-%p.profraw")}
        if raw_dir is not None and collector == "llvm" else {}
    )
    if raw_dir is not None:
        # The simulator may create helper files during one job.  Keep those
        # files beside that job's raw/trace evidence, never in a shared run
        # or host temporary directory.
        env["TMPDIR"] = str(tmp_dir)
    target = config.get("target")
    target_backend = config.get("backend")
    if target_backend is None and isinstance(target, Mapping):
        target_backend = target.get("backend")
    if raw_dir is not None and (target_backend == "rax-riscv64" or (
        isinstance(target, Mapping) and target.get("id") == "T-RAX"
    )):
        # RAX's guest PC trace is an execution side channel, separate from
        # LLVM's profraw stream.  Keep both in the same case-local raw tree.
        env["RAX_TRACE_PATH"] = str(raw_dir / "rax-%p.trace")
    if raw_dir is not None and target_backend == "renode-riscv64" and collector == "dotnet":
        # The coverage Renode is a separately-built executable.  Keep the
        # source-verified target identity available to the adapter so the
        # instrumented collector and ordinary runner identify one target.
        env["RQ1_TARGET_LOGICAL_BINARY_PATH"] = str(identity_binary)
        launcher = source.get("launcher")
        if isinstance(launcher, str) and launcher:
            env["RQ1_DOTNET_COVERAGE_LAUNCHER"] = launcher
            env["RQ1_DOTNET_COVERAGE_OUTPUT"] = str(raw_dir / "renode.cobertura.xml")
            session_id = config.get("coverage_batch_session_id")
            if isinstance(session_id, str) and session_id:
                env["RQ1_DOTNET_COVERAGE_SESSION_ID"] = session_id
                # The connect client also uses a Unix-domain socket. Keep its
                # IPC path short while all report/trace paths stay case-local.
                env["TMPDIR"] = "/tmp"
            include_files = [runner_binary]
            infrastructure = runner_binary.with_name("Infrastructure.dll")
            if infrastructure.is_file():
                include_files.append(infrastructure)
            env["RQ1_DOTNET_COVERAGE_INCLUDE_FILES"] = ",".join(
                str(path) for path in include_files
            )
    if target_backend == "rvvm-riscv64":
        # The launcher is dynamically linked; do not let a normal-target
        # librvvm.so.0 shadow the stepcov sidecar selected above.
        env["LD_LIBRARY_PATH"] = str(runner_binary.parent)
        if raw_dir is not None:
            # The stepcov build writes one complete guest-PC record per line.
            # The framework adapter consumes this file after the mailbox
            # closes, so it does not need one GDB round trip per instruction.
            env["RQ1_RVVM_TRACE_PATH"] = str(raw_dir / "rvvm-%p.pc")
    return runner_binary, env


def run_backend_with_coverage(
    backend: str, executable: Path, *, profile_id: str | None = None,
    input_id: str | None = None, runner_env: dict[str, str] | None = None,
    backend_binary_path: str | None = None,
    coverage_config: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
    campaign_deadline_monotonic: float | None = None,
) -> object:
    """Run one target request, optionally selecting the instrumented binary."""
    from framework.direct_runner import run_backend

    config = coverage_config if isinstance(coverage_config, Mapping) else None
    try:
        # One attempt owns one absolute deadline.  Queueing for a shared
        # target is part of that attempt; otherwise a fast simulator can
        # still consume an unbounded amount of wall clock before it starts.
        now = time.monotonic()
        attempt_deadline = campaign_deadline_monotonic
        if timeout_seconds is not None:
            timeout_deadline = now + max(0.0, float(timeout_seconds))
            attempt_deadline = (
                timeout_deadline
                if attempt_deadline is None
                else min(float(attempt_deadline), timeout_deadline)
            )
        started = time.monotonic()
        execution_timeout = timeout_seconds
        if attempt_deadline is not None:
            deadline_remaining = max(
                0.0, float(attempt_deadline) - started,
            )
            execution_timeout = min(
                float(timeout_seconds), deadline_remaining,
            ) if timeout_seconds is not None else deadline_remaining
        coverage_setup_gap = None
        if config and config.get("enabled", True):
            try:
                selected, coverage_env = _coverage_target(
                    config, Path(executable).resolve(),
                )
                env = {**(runner_env or {}), **coverage_env}
                binary = str(selected)
            except (KeyError, OSError, TypeError, ValueError) as error:
                coverage_setup_gap = {
                    "schema_version": "rq1-framework-coverage-setup-gap-v1",
                    "reason": "coverage-setup-gap",
                    "error_type": type(error).__name__,
                }
                env, binary = runner_env, backend_binary_path
                try:
                    raw_dir = config.get("raw_dir")
                    if raw_dir:
                        marker = _path(raw_dir).parent / "coverage-setup-gap.json"
                        atomic_write_json(marker, coverage_setup_gap)
                except OSError:
                    pass
        else:
            env, binary = runner_env, backend_binary_path
        def invoke(
                invoke_env: dict[str, str] | None,
                invoke_binary: str | None,
        ) -> object:
            remaining = None if execution_timeout is None else max(
                0.0, float(execution_timeout) - (time.monotonic() - started)
            )
            return run_backend(
                backend, Path(executable), profile_id=profile_id, input_id=input_id,
                runner_env=invoke_env, backend_binary_path=invoke_binary,
                timeout_seconds=remaining,
            )

        collector = (
            config.get("source_coverage", {}).get("collector")
            if isinstance(config, Mapping)
            and isinstance(config.get("source_coverage"), Mapping)
            else None
        )
        # One ordinary run records the target observation. A .NET source
        # profile uses one additional instrumented run through the collector.
        if (
            collector == "dotnet"
            and isinstance(backend_binary_path, str)
            and backend_binary_path
        ):
            normal_env = dict(runner_env or {})
            for name in (
                "RQ1_DOTNET_COVERAGE_LAUNCHER",
                "RQ1_DOTNET_COVERAGE_OUTPUT",
                "RQ1_DOTNET_COVERAGE_INCLUDE_FILES",
                "RQ1_DOTNET_COVERAGE_SESSION_ID",
            ):
                normal_env.pop(name, None)
            coverage_key = _dotnet_coverage_key(env)
            if _dotnet_coverage_already_attempted(coverage_key):
                return invoke(normal_env, backend_binary_path)
            normal = invoke(normal_env, backend_binary_path)
            coverage_output = _dotnet_coverage_output(config)
            if coverage_output is not None:
                # raw-only defers conversion, not collection.  The resident
                # Target still needs one instrumented replay so the later
                # offline collector has a Cobertura input to convert.
                _schedule_dotnet_coverage(
                    key=coverage_key,
                    output=coverage_output,
                    config=config,
                    backend=backend,
                    executable=Path(executable),
                    profile_id=profile_id,
                    input_id=input_id,
                    env=env, binary=binary,
                )
                return normal
            # Unit-level callers may provide a reduced config without a case
            # raw directory. Keep that legacy contract synchronous.
            covered = invoke(env, binary)
            if covered:
                _mark_dotnet_coverage_completed(
                    coverage_key
                    if _dotnet_report_observed(
                        env.get("RQ1_DOTNET_COVERAGE_OUTPUT")
                    ) else None
                )
            if _has_terminal_observation(normal):
                return normal
            return covered or normal
        return invoke(env, binary)
    except Exception:
        raise


def _local_artifact_path(path: Path, output: Path | None) -> Path:
    """Resolve a container-recorded case path against the local run mount."""
    if path.is_file() or output is None or not path.is_absolute():
        return path
    container_parts = path.parts
    marker = next(
        (container_parts.index(name) for name in ("framework-targets", "framework")
         if name in container_parts),
        None,
    )
    if marker is None:
        return path
    local_output = output.resolve()
    local_marker = next(
        (local_output.parts.index(name) for name in ("framework-targets", "framework")
         if name in local_output.parts),
        None,
    )
    if local_marker is None:
        return path
    local_run_root = Path(*local_output.parts[:local_marker])
    return local_run_root / Path(*container_parts[marker:])


def _artifact_paths(
    value: object, *, stratum: str | None = None, output: Path | None = None,
) -> tuple[Path, ...]:
    if not isinstance(value, Mapping):
        return ()
    result = []
    names = (
        ("bare_executable_path", "executable_path", "linux_executable_path")
        if stratum == "bare-metal" else
        ("linux_executable_path", "executable_path", "bare_executable_path")
    )
    for name in names:
        path = value.get(name)
        if isinstance(path, str) and path:
            candidate = _local_artifact_path(Path(path), output)
            if not candidate.is_file():
                continue
            candidate = candidate.resolve()
            if candidate not in result:
                result.append(candidate)
    return tuple(result)


def _campaign_artifacts(
    campaign: Mapping[str, object], output: Path, *, stratum: str | None = None,
) -> tuple[Path, ...]:
    result: list[Path] = []

    def add(value: object) -> None:
        for path in _artifact_paths(value, stratum=stratum, output=output):
            if path not in result:
                result.append(path)

    for item in campaign.get("generation_artifacts", ()):
        add(item)
    for item in campaign.get("reference", ()):
        if isinstance(item, Mapping):
            for artifact in (item.get("artifacts") or {}).values():
                add(artifact)
    for path in (
        output / "profile" / "program.elf", output / "program.elf",
        output / "generation" / "program.elf",
    ):
        if path.is_file() and path.resolve() not in result:
            result.append(path.resolve())
    return tuple(result)


def _target_stratum(target: Mapping[str, object] | None) -> str:
    """Derive the execution stratum from the target contract.

    Framework coverage is grouped by stratum.  Keeping this derivation here
    prevents bare-metal capsule rows from being silently filed under the
    Linux-user bucket when a target record omits an explicit ``stratum``.
    """
    if not isinstance(target, Mapping):
        return "linux-user"
    explicit = target.get("stratum")
    if explicit in {"bare-metal", "linux-user"}:
        return str(explicit)
    model = str(target.get("execution_model") or "").lower()
    kind = str(target.get("kind") or "").lower()
    target_id = str(target.get("id") or "").upper()
    if (
        "bare" in model
        or kind in {"unicorn", "renode", "rax", "rvvm"}
        or target_id in {"T-UNICORN", "T-RENODE", "T-RAX", "T-RVVM"}
    ):
        return "bare-metal"
    return "linux-user"


def _framework_feature_for_coverage(
    artifact: Path, isa_profile: str,
    observation: Mapping[str, object] | None = None,
    feature_cache: dict[Path, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Decode one guest ELF with the same projections used by final coverage."""
    cache = feature_cache if feature_cache is not None else {}
    if artifact not in cache:
        cache[artifact] = decode_elf(
            artifact, isa_profile, privilege_mode="user",
        )
    feature = cache[artifact]
    trap_observation = observation if isinstance(observation, Mapping) else {}
    unknown = [
        item for item in feature.get("instructions", ())
        if isinstance(item, Mapping) and item.get("extension") == "unknown"
    ]
    unknown_names = {str(item.get("mnemonic")) for item in unknown}
    trap_scaffold = (
        trap_observation.get("outcome") == "trap"
        and bool(unknown)
        and all(name == ".word" or name.startswith("csr") for name in unknown_names)
    )
    if trap_scaffold:
        instructions = []
        for item in feature.get("instructions", ()):
            if not isinstance(item, Mapping):
                instructions.append(item)
                continue
            if item.get("extension") != "unknown":
                instructions.append(item)
                continue
            if item.get("mnemonic") == ".word":
                instructions.append({
                    **item,
                    "coverage_excluded_metrics": [
                        "GenCov", "ExecCov", "ICov-encoding",
                        "ICov-type", "CCov",
                    ],
                })
            else:
                instructions.append({
                    **item,
                    "coverage_excluded_metrics": [
                        "PCov", "GenCov", "ExecCov", "ICov-encoding",
                        "ICov-type", "CCov",
                    ],
                })
        feature = {
            **feature,
            "status": "ok",
            "reason": None,
            "instructions": instructions,
            "coverage_scaffold_unknown_instructions": sorted(unknown_names),
        }
    bounds = _program_body_bounds(observation or {})
    if bounds:
        start, end = bounds
        full_instructions = feature.get("instructions", ())
        instructions = [
            item for item in full_instructions
            if isinstance(item, Mapping)
            and type(item.get("address")) is int
            and start <= item["address"] < end
        ]
        if instructions:
            feature = {
                **feature,
                # Catalog metrics use the complete ELF map and trace; legacy RV
                # metrics keep the program-body projection.
                "opcode_catalog_static_instructions": full_instructions,
                "opcode_catalog_execution_instructions": full_instructions,
                "instructions": instructions,
            }
    return feature


def _coverage_row(
    observation: Mapping[str, object], *, artifact: Path, campaign: Mapping[str, object],
    target: Mapping[str, object], case_id: str, method: str | None = None,
) -> dict[str, object]:
    outcome = observation.get("outcome")
    complete = outcome in {
        "normal", "completed", "trap", "expected-trap", "nonzero-exit",
    }
    explicit_attempted = observation.get("target_attempted")
    attempted = (
        explicit_attempted
        if type(explicit_attempted) is bool
        else any(
            observation.get(name) is True
            for name in (
                "process_started", "process_executed", "executed",
                "terminal_observed",
            )
        ) or complete or observation.get("outcome") in {
            "timeout", "crash", "gap",
        }
    )
    explicit_terminal = observation.get("terminal_observed")
    terminal_observed = (
        explicit_terminal
        if type(explicit_terminal) is bool
        else complete or outcome in {"timeout", "crash", "gap"}
    )
    explicit_started = observation.get("process_started")
    process_started = (
        explicit_started
        if type(explicit_started) is bool
        else bool(attempted)
    )
    explicit_executed = observation.get("process_executed")
    process_executed = explicit_executed if type(explicit_executed) is bool else None
    if process_executed is None and type(observation.get("executed")) is bool:
        process_executed = observation["executed"]
    explicit_complete = observation.get("observer_complete")
    observer_complete = (
        explicit_complete
        if type(explicit_complete) is bool
        else complete and attempted and terminal_observed
        and not observation.get("contract_error")
    )
    if not attempted:
        observer_complete = False
    translation = observation.get("translation_evidence")
    translation_details = (
        translation.get("details")
        if isinstance(translation, Mapping)
        and isinstance(translation.get("details"), Mapping)
        else {}
    )
    trace_states = [
        value for value in (
            observation.get("trace_complete"),
            translation_details.get("trace_complete"),
            translation_details.get("renode_trace_complete"),
            translation_details.get("rvvm_trace_complete"),
        )
        if isinstance(value, bool)
    ]
    if translation_details.get("trace_parse_error") is True:
        trace_states.append(False)
    trace_complete = all(trace_states) if trace_states else None
    # Renode may return a valid RVOBS1 checkpoint after its asynchronous
    # instruction hook produced no PC trace.  That checkpoint proves state,
    # not executed-instruction coverage; never feed it into PCov/ICov/CCov.
    trace_available = observation.get("trace_available")
    if trace_available is None:
        trace_available = translation_details.get("trace_available")
    target_outcome = (
        "expected-trap" if outcome in {"trap", "expected-trap"}
        else "nonzero-exit" if outcome == "nonzero-exit"
        else "complete_observation" if outcome in {"normal", "completed"}
        else None
    )
    pcs = observation.get("executed_pcs")
    catalog_pcs = None
    catalog_trace_invalid = False
    if isinstance(pcs, (list, tuple)):
        # Opcode coverage is a PC set, so keep unique addresses and retain one
        # separate invalid-evidence flag instead of copying the full trace.
        catalog_pcs = []
        catalog_seen: set[int] = set()
        for value in pcs:
            pc = _pc_value(value)
            if pc is None:
                catalog_trace_invalid = True
            elif pc not in catalog_seen:
                catalog_seen.add(pc)
                catalog_pcs.append(pc)
        catalog_pcs = tuple(catalog_pcs)
    if trace_available is False:
        pcs = ()
        catalog_pcs = ()
        catalog_trace_invalid = False
    body_bounds = _program_body_bounds(observation)
    if body_bounds and isinstance(pcs, (list, tuple)):
        start, end = body_bounds
        pcs = tuple(pc for pc in pcs if type(pc) is int and start <= pc < end)
    stratum = _target_stratum(target)
    return {
        "method": method or campaign.get("method") or "framework",
        "route": campaign.get("root_binding", {}).get("route")
        if isinstance(campaign.get("root_binding"), Mapping) else None,
        "lane": target.get("lane") or campaign.get("root_binding", {}).get("lane")
        if isinstance(campaign.get("root_binding"), Mapping) else target.get("lane"),
        "isa_profile": target.get("isa_profile"),
        "stratum": stratum,
        "target": target.get("id"), "case_id": case_id,
        "target_identity_id": (
            target.get("identity_digest") or target.get("binary_sha256")
            or target.get("commit")
        ),
        **({"coverage_only": True} if observation.get("coverage_only") is True else {}),
        "artifact_sha256": observation.get("guest_elf_sha256") or sha256_file(artifact),
        **({"trace_path": trace_path} if isinstance(
            trace_path := (
                observation.get("trace_path") or translation_details.get("trace_path")
            ), str
        ) and trace_path else {}),
        "trace_pcs": pcs if isinstance(pcs, (list, tuple)) else [],
        **({"opcode_catalog_trace_pcs": catalog_pcs}
           if isinstance(catalog_pcs, (list, tuple)) else {}),
        **({"opcode_catalog_trace_invalid": True}
           if catalog_trace_invalid else {}),
        "trace_scope_projected": body_bounds is not None,
        "target_outcome": target_outcome,
        # Preserve execution facts supplied by the adapter.  A missing PC
        # trace is a coverage gap after an attempted run, while an explicit
        # ``target_attempted=false`` remains unattempted.
        "target_attempted": bool(attempted),
        "process_started": process_started,
        "process_executed": process_executed,
        "executed": process_executed,
        "terminal_observed": bool(terminal_observed),
        **({"trace_complete": trace_complete} if trace_complete is not None else {}),
        "trace_reason": observation.get("trace_reason")
        or translation_details.get("trace_gap")
        or translation_details.get("trace_parse_error_detail")
        or ("trace-parse-error" if translation_details.get("trace_parse_error") is True else None)
        or ("trace-unavailable" if trace_available is False else None),
        "observer_complete": bool(observer_complete),
        "observation": dict(observation),
    }


def _observations(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(item for item in value if isinstance(item, Mapping)) \
        if isinstance(value, (list, tuple)) else ()


def _program_body_bounds(observation: Mapping[str, object]) -> tuple[int, int] | None:
    extra = observation.get("extra_state")
    path = extra.get("program_test_path") if isinstance(extra, Mapping) else None
    if not isinstance(path, Mapping) or path.get("status") != "observed":
        return None
    start, end = path.get("start_pc"), path.get("end_pc")
    return (start, end) if type(start) is int and type(end) is int and start < end else None


def _trace_incomplete(row: Mapping[str, object]) -> bool:
    """A partial trace cannot be promoted to a complete coverage artifact."""
    if row.get("trace_complete") is False or row.get("trace_cleanup_status") == "gap":
        return True
    if any(row.get(name) is True for name in ("trace_truncated", "trace_limit_exceeded")):
        return True
    observation = row.get("observation")
    if isinstance(observation, Mapping) and any(
        observation.get(name) is True
        for name in ("trace_truncated", "trace_limit_exceeded")
    ):
        return True
    if isinstance(observation, Mapping) and (
        observation.get("trace_complete") is False
        or observation.get("trace_cleanup_status") == "gap"
    ):
        return True
    coverage = row.get("simulator_coverage")
    if isinstance(coverage, Mapping):
        events = coverage.get("simulator_events")
        trace = events.get("trace") if isinstance(events, Mapping) else None
        return isinstance(trace, Mapping) and trace.get("truncated") is True
    return False


def _deadline_censored(value: object) -> bool:
    """Recognize only an explicit shared-deadline observation as right-censored."""
    if not isinstance(value, Mapping):
        return False
    pending_reasons = {
        "run-wall-clock-exhausted", "execution-wall-clock-exhausted",
        "campaign-wall-clock-exhausted", "framework-target-wall-clock-exhausted",
    }
    pending_outcomes = {
        "pending", "right-censored", "right_censored",
        "deadline-censored", "deadline_censored",
    }
    pending_statuses = pending_outcomes
    pending_terminations = pending_reasons | {"run-deadline-censored"}
    def matches(item: object, allowed: set[str]) -> bool:
        return isinstance(item, str) and item in allowed

    sources: list[Mapping[str, object]] = [value]
    observation = value.get("observation")
    if isinstance(observation, Mapping):
        sources.append(observation)
    stages = value.get("stage_records")
    if isinstance(stages, Mapping) and isinstance(stages.get("target"), Mapping):
        sources.append(stages["target"])
    external_signal = any(
        matches(source.get(name), {"external-signal"})
        for source in sources
        for name in (
            "reason_code", "pending_reason", "reason", "failure_class",
            "termination",
        )
    )
    explicit_deadline = any(
        source.get(name) is True
        for source in sources
        for name in (
            "deadline_censored", "run_deadline_censored",
        )
    ) or any(
        matches(source.get("status"), {"deadline-censored", "deadline_censored"})
        or matches(source.get("failure_class"), pending_reasons)
        or matches(source.get("termination"), pending_terminations)
        or any(matches(source.get(name), pending_reasons) for name in (
            "reason_code", "pending_reason", "reason",
        ))
        for source in sources
    )
    if external_signal and not explicit_deadline:
        return False
    for source in sources:
        if (
            matches(source.get("status"), pending_statuses)
            or matches(source.get("target_outcome"), pending_outcomes)
            or source.get("right_censored") is True
            or source.get("deadline_censored") is True
            or source.get("run_deadline_censored") is True
            or matches(source.get("failure_class"), pending_reasons)
            or matches(source.get("case_outcome"), pending_outcomes)
            or matches(source.get("outcome"), pending_outcomes)
            or matches(source.get("termination"), pending_terminations)
        ):
            return True
        if (
            any(matches(reason, pending_reasons) for reason in (
                source.get("reason_code"), source.get("pending_reason"),
                source.get("reason"),
            ))
            and source.get("terminal_observed") is not True
            and (
                source.get("outcome") is None
                or matches(source.get("outcome"), {"gap", "pending"})
            )
        ):
            return True
    return False


def _external_signal_censored(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    sources: list[Mapping[str, object]] = [value]
    observation = value.get("observation")
    if isinstance(observation, Mapping):
        sources.append(observation)
    stages = value.get("stage_records")
    if isinstance(stages, Mapping) and isinstance(stages.get("target"), Mapping):
        sources.append(stages["target"])
    for source in sources:
        if not any(source.get(name) == "external-signal" for name in (
            "reason_code", "pending_reason", "reason", "failure_class",
            "termination",
        )):
            continue
        if source.get("terminal_observed") is True:
            continue
        if source.get("outcome") in {
            None, "gap", "pending", "right-censored", "right_censored",
        }:
            return True
    return False


def _right_censored(row: Mapping[str, object]) -> bool:
    if (
        row.get("right_censored") is True
        or _deadline_censored(row)
        or _external_signal_censored(row)
    ):
        return True
    observation = row.get("observation")
    return _deadline_censored(observation)


def _mark_rv_coverage_incomplete(
    summary: dict[str, object], reason: str, *, include_targets: bool = True,
) -> None:
    """Keep RV evidence while preventing incomplete-run ratios."""
    names = ("targets", "experiment_unions") if include_targets else (
        "experiment_unions",
    )
    containers = [
        item for name in names
        for item in summary.get(name, ())
        if isinstance(item, dict)
    ]
    for container in containers:
        values: list[tuple[object, dict[str, object] | None]] = []
        rv = container.get("rv_instruction_coverage")
        if isinstance(rv, dict):
            values.append((rv.get("metrics"), rv))
        guest = container.get("guest")
        if isinstance(guest, dict):
            values.append((guest, None))
            values.append((guest.get("metrics"), None))
        for metrics, envelope in values:
            if not isinstance(metrics, dict):
                continue
            changed = False
            has_gap = False
            for name in RV_INSTRUCTION_METRICS:
                metric = metrics.get(name)
                if not isinstance(metric, dict):
                    continue
                if metric.get("status") == "gap":
                    has_gap = True
                elif metric.get("status") in {"observed", "partial"}:
                    metric.update(status="partial", value=None, reason=reason)
                    changed = True
            if envelope is not None and changed:
                envelope["status"] = "gap" if has_gap else "partial"
                envelope["reason"] = reason


def finalize_framework_source_coverage_batch(
    config: Mapping[str, object], *, expected_coverage_cases: int | None = None,
    expected_collection_inputs: int | None = None,
) -> dict[str, object]:
    """Convert source profiles once for one framework case batch."""
    config = _effective_coverage_config(config)
    summary_value = config.get("coverage_batch_summary_path")
    raw_value = config.get("coverage_batch_raw_dir")
    if not summary_value or not raw_value:
        return {"status": "gap", "reason": "framework-coverage-batch-not-configured"}
    summary_path = _path(summary_value)
    output = summary_path.parent
    raw_dir = _path(raw_value)
    target_data = config.get("target")
    target_data = dict(target_data) if isinstance(target_data, Mapping) else {}
    target_data.setdefault("id", config.get("target_id"))
    target_data.setdefault("kind", config.get("kind", "qemu"))
    target_data.setdefault("stratum", config.get("stratum"))
    target_data["stratum"] = _target_stratum(target_data)
    source = config.get("source_coverage")
    source_config = dict(source) if isinstance(source, Mapping) else {}
    for key in (
        "profile", "identity", "source_root", "build_root", "launcher",
        "reportgenerator", "profdata_tool", "cov_tool", "gcov",
    ):
        value = source_config.get(key)
        if value:
            source_config[key] = str(_path(value))
    target_data["source_coverage"] = source_config
    target_data["coverage_binary"] = str(_path(config.get("binary_path")))
    collector = str(target_data["source_coverage"].get("collector", ""))
    session_mode = config.get("dotnet_coverage_session_mode") == "server"
    # The resident Target worker owns the .NET collector process, so the
    # immutable per-case config cannot contain its runtime session id. Read
    # the batch marker written by that worker before validating the report.
    server_marker = raw_dir / "dotnet-coverage-server.json"
    if collector == "dotnet" and server_marker.is_file():
        try:
            marker = read_json_object(server_marker)
        except (OSError, TypeError, ValueError):
            marker = {}
        marker_status = marker.get("status")
        marker_session = marker.get("session_id")
        if marker_status in {"starting", "running", "stopped", "shutdown-gap"}:
            session_mode = True
            config = {
                **config,
                "coverage_batch_session_id": marker_session,
                "dotnet_coverage_session_mode": "server",
            }
    expected_inputs = (
        expected_collection_inputs
        if expected_collection_inputs is not None else expected_coverage_cases
    )
    if session_mode and collector == "dotnet":
        expected_inputs = 1 if expected_coverage_cases else 0
    wait_timeout_markers = (
        list(raw_dir.rglob("dotnet-coverage-wait-timeout.json"))
        if collector == "dotnet" else []
    )
    if config.get("coverage_batch_session_shutdown_error"):
        collection = {
            "status": "gap",
            "reason": "dotnet-coverage-session-shutdown-failed",
            "error": str(config["coverage_batch_session_shutdown_error"]),
        }
    elif expected_coverage_cases == 0:
        collection = {
            "status": "gap",
            "reason": "framework-source-coverage-no-target-executions",
        }
    else:
        collection = collect_source_coverage_batch(
            output, target_data, _path(config.get("binary_path")), raw_dir,
            expected_coverage_cases=expected_inputs,
        )
    if expected_inputs and collector in {"dotnet", "llvm"}:
        input_count = len(coverage_input_files(raw_dir, collector))
        mismatch = (
            input_count != expected_inputs if collector == "dotnet"
            else input_count < expected_inputs
        )
        if mismatch and collection.get("status") in {"observed", "partial"}:
            collection.update(
                status="partial",
                reason=(
                    "dotnet-cobertura-report-count-mismatch"
                    if collector == "dotnet" else "llvm-profraw-count-mismatch"
                ),
                expected_coverage_cases=expected_inputs,
                expected_target_cases=expected_coverage_cases,
                observed_coverage_inputs=input_count,
            )
    session_id = config.get("coverage_batch_session_id")
    if session_mode and expected_coverage_cases and isinstance(session_id, str):
        completed_clients = 0
        for status_path in raw_dir.rglob("dotnet-coverage-task.json"):
            try:
                status_record = read_json_object(status_path)
            except (OSError, TypeError, ValueError):
                continue
            if (
                status_record.get("session_id") == session_id
                and status_record.get("status") in {"client-recorded", "recorded"}
                and status_record.get("client_completed") is True
            ):
                completed_clients += 1
        if completed_clients == 0 \
                and collection.get("status") in {"observed", "partial"}:
            collection.update(
                status="partial",
                reason="dotnet-coverage-client-record-missing",
                expected_target_cases=expected_coverage_cases,
                completed_clients=completed_clients,
            )
    if wait_timeout_markers:
        collection["wait_timeout_cases"] = len(wait_timeout_markers)
        if collection.get("status") in {"observed", "partial"}:
            collection.update(
                status="partial",
                reason="dotnet-coverage-task-wait-timeout",
            )
    source_result = target_source_coverage(
        output, target_data,
        coverage_binary_sha=collection.get("coverage_binary_sha"),
        source_commit_observed=collection.get("source_commit_observed"),
        coverage_binary_sha256=collection.get("coverage_binary_sha256"),
        collection_status=collection.get("status"),
        collection_reason=collection.get("reason"),
    )
    if collection.get("warnings"):
        source_result["warnings"] = sorted(set(collection["warnings"]))
    setup_gaps = sorted(raw_dir.rglob("coverage-setup-gap.json")) \
        if raw_dir.is_dir() else []
    if setup_gaps and source_result.get("status") == "observed":
        source_result.update(
            status="partial", reason="framework-coverage-setup-gap",
        )
    batch_id = str(config.get("coverage_batch_id") or summary_path.parent.name)
    batch_scope = str(config.get("coverage_batch_scope") or "target-batch")
    target_id = str(target_data.get("id") or "")
    target_row = {
        "method": config.get("method"), "route": config.get("route"),
        "lane": config.get("lane"), "stratum": target_data["stratum"],
        "target": target_id, "scope": batch_scope, "batch_id": batch_id,
        "cases": expected_coverage_cases,
        "simulator_source_coverage": source_result,
    }
    summary = {
        "schema_version": "rq1-framework-source-coverage-batch-v1",
        "status": source_result.get("status"), "batch_id": batch_id,
        "target": target_id, "case_count": expected_coverage_cases,
        "collection": collection, "source_targets": [target_row],
        "framework": {
            "scope": batch_scope, "batch_id": batch_id,
            "case_count": expected_coverage_cases,
            "coverage_setup_gap_count": len(setup_gaps),
        },
    }
    atomic_write_json(summary_path, summary)
    return summary


def finalize_framework_coverage(
    output: Path, campaign: Mapping[str, object], config: Mapping[str, object],
) -> dict[str, object]:
    """Materialize both guest instruction and simulator-source coverage."""
    config = _effective_coverage_config(config)
    output = Path(output).resolve()
    target_data = config.get("target")
    target_data = dict(target_data) if isinstance(target_data, Mapping) else {}
    target_data.setdefault("id", config.get("target_id"))
    target_data.setdefault("kind", config.get("kind", "qemu"))
    target_data.setdefault("execution_model", config.get("execution_model"))
    target_data.setdefault("stratum", config.get("stratum"))
    target_data.setdefault("privilege_mode", "user")
    target_data["stratum"] = _target_stratum(target_data)
    lane = str(config.get("lane") or target_data.get("lane") or "rv64i/lp64")
    isa_profile = str(
        config.get("isa_profile") or config.get("isa")
        or target_data.get("isa_profile") or lane.split("/", 1)[0]
    ).strip().lower()
    target_data["isa_profile"] = isa_profile
    target_data["coverage_binary"] = str(_path(config.get("binary_path")))
    source = config.get("source_coverage")
    target_data["source_coverage"] = dict(source) if isinstance(source, Mapping) else {}
    artifacts = _campaign_artifacts(campaign, output, stratum=target_data["stratum"])
    method = str(config.get("method") or campaign.get("method") or "framework")
    feature_cache: dict[Path, dict[str, object]] = {}

    def feature_for(
        artifact: Path, observation: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        # Cache only the decoded ELF; observation-specific projections must
        # not leak into another case that reuses this artifact.
        return _framework_feature_for_coverage(
            artifact, isa_profile, observation, feature_cache,
        )

    records: list[dict[str, object]] = []
    case_id = str(campaign.get("seed") or output.name)
    include_guest_metrics = config.get("rv_instruction_coverage_enabled") is not False

    def coverage_for(
        row: dict[str, object], artifact: Path, observation: Mapping[str, object],
    ) -> dict[str, object]:
        if not include_guest_metrics:
            return {
                "schema_version": SCHEMA,
                "status": "disabled",
                "reason": "guest-metrics-disabled",
            }
        return summarize_simulator_coverage(
            target_data, row, output, feature_for(artifact, observation),
        )

    baseline = campaign.get("target_baseline")
    baseline_artifact = artifacts[0] if artifacts else None
    if isinstance(baseline, Mapping) and baseline_artifact is not None:
        for index, observation in enumerate(_observations(baseline.get("observations"))):
            if baseline.get("coverage_only") is True:
                observation = {
                    **observation,
                    "target_attempted": True,
                    "coverage_only": True,
                }
            row = _coverage_row(
                observation, artifact=baseline_artifact, campaign=campaign,
                target=target_data, case_id=f"{case_id}:baseline:{index}",
                method=method,
            )
            row["simulator_coverage"] = coverage_for(
                row, baseline_artifact, observation,
            )
            records.append(row)

    references = campaign.get("reference")
    targets = campaign.get("target")
    if isinstance(references, list) and isinstance(targets, list):
        for target_index, target_record in enumerate(targets):
            if not isinstance(target_record, Mapping):
                continue
            comparison = target_record.get("comparison")
            observations = comparison.get("observations") if isinstance(comparison, Mapping) else None
            if not isinstance(observations, Mapping):
                continue
            reference_index, reference_index_valid = _nonnegative_count(
                target_record.get("reference_index"), missing=None,
            )
            if not reference_index_valid or reference_index is None:
                continue
            reference = references[reference_index] if 0 <= reference_index < len(references) else {}
            pair = reference.get("artifacts", {}) if isinstance(reference, Mapping) else {}
            reference_paths = {
                "original": next(iter(_artifact_paths(
                    pair.get("reference_original"), stratum=target_data["stratum"],
                    output=output,
                )), None),
                "variant": next(iter(_artifact_paths(
                    pair.get("reference_variant"), stratum=target_data["stratum"],
                    output=output,
                )), None),
            } if isinstance(pair, Mapping) else {"original": None, "variant": None}
            target_paths = {
                "original": next(iter(_artifact_paths(
                    target_record.get("target_original_artifact"),
                    stratum=target_data["stratum"], output=output,
                )), None),
                "variant": next(iter(_artifact_paths(
                    target_record.get("artifact"),
                    stratum=target_data["stratum"], output=output,
                )), None),
            }
            for side in ("original", "variant"):
                artifact = target_paths[side] or reference_paths[side] or baseline_artifact
                if artifact is None:
                    continue
                for observation_index, observation in enumerate(_observations(observations.get(side))):
                    if target_record.get("coverage_only") is True:
                        observation = {
                            **observation,
                            "target_attempted": True,
                            "coverage_only": True,
                        }
                    row = _coverage_row(
                        observation, artifact=artifact, campaign=campaign,
                        target=target_data,
                        case_id=f"{case_id}:target:{target_index}:{side}:{observation_index}",
                        method=method,
                    )
                    row["right_censored"] = _right_censored(target_record)
                    row["simulator_coverage"] = coverage_for(
                        row, artifact, observation,
                    )
                    records.append(row)

    batch_summary_path = config.get("coverage_batch_summary_path")
    batch_raw_dir = config.get("coverage_batch_raw_dir")
    coverage_batch = None
    if batch_summary_path and batch_raw_dir:
        coverage_batch = {
            "id": str(config.get("coverage_batch_id") or ""),
            "summary_path": str(_path(batch_summary_path)),
            "scope": str(config.get("coverage_batch_scope") or "target-batch"),
        }
        collection = {
            "status": "deferred",
            "reason": "framework-source-coverage-batch-pending",
        }
        source_result = None
    else:
        collection = collect_source_coverage_batch(
            output, target_data, _path(config.get("binary_path")),
            _path(config.get("raw_dir")),
        )
        source_result = target_source_coverage(
            output, target_data,
            coverage_binary_sha=collection.get("coverage_binary_sha"),
            source_commit_observed=collection.get("source_commit_observed"),
            coverage_binary_sha256=collection.get("coverage_binary_sha256"),
            collection_status=collection.get("status"),
            collection_reason=collection.get("reason"),
        )
        if collection.get("warnings"):
            source_result["warnings"] = sorted(set(collection["warnings"]))
    setup_gap = None
    setup_gap_paths = [output / "coverage-setup-gap.json"]
    raw_dir = config.get("raw_dir")
    if raw_dir:
        setup_gap_paths.append(_path(raw_dir).parent / "coverage-setup-gap.json")
    for setup_gap_path in setup_gap_paths:
        if not setup_gap_path.is_file():
            continue
        try:
            setup_gap = read_json_object(setup_gap_path).get("reason")
        except (OSError, TypeError, ValueError):
            setup_gap = "coverage-setup-gap"
        break
    # Coverage is evidence from the target process.  Its eligibility is not a
    # differential verdict and must not depend on RVOBS1/rich observer fields.
    complete_records = [
        row for row in records
        if row.get("target_attempted") is True
        and (
            (
                include_guest_metrics
                and isinstance(row.get("simulator_coverage"), Mapping)
                and row["simulator_coverage"].get("status") in {"observed", "partial"}
                and not _trace_incomplete(row)
            )
            or (
                not include_guest_metrics
                and not _right_censored(row)
            )
        )
    ]
    if records and source_result is not None:
        records[0]["simulator_source_coverage"] = source_result
    # Every parseable observation contributes its covered units. Completeness
    # gates the reported status/value, never the numerator collected so far.
    summary = aggregate_simulator_coverage(
        records, include_guest_metrics=include_guest_metrics,
    )
    if not coverage_batch and not summary.get("source_targets"):
        summary["source_targets"] = [{
            "target": target_data.get("id"), "stratum": target_data["stratum"],
            "scope": "target-run", "simulator_source_coverage": source_result,
        }]
    campaign_targets = campaign.get("target")
    campaign_targets = campaign_targets if isinstance(campaign_targets, list) else ()
    right_censored_case_count = sum(
        1 for item in campaign_targets
        if isinstance(item, Mapping) and _right_censored(item)
    )
    if not right_censored_case_count:
        right_censored_case_count = sum(_right_censored(row) for row in records)
    summary["framework"] = {
        "method": method,
        "simulator_source_coverage": "instrumented-target-profile -> SimSrcCov",
        "target": target_data.get("id"),
        "trace_incomplete_cases": sum(_trace_incomplete(row) for row in records),
        "completed_case_count": len(complete_records),
        "right_censored_case_count": right_censored_case_count,
        "target_wall_clock_pending": sum(
            _deadline_censored(item)
            for item in campaign_targets if isinstance(item, Mapping)
        ),
    }
    if coverage_batch:
        summary["framework"]["coverage_batch"] = coverage_batch
    summary_path = _path(config.get("summary_path")) \
        if config.get("summary_path") else output / "coverage" / "summary.json"
    atomic_write_json(summary_path, summary)
    guest_ok = (
        (not include_guest_metrics and bool(records))
        or (
            include_guest_metrics
            and bool(complete_records)
            and summary.get("status") in {"observed", "partial"}
        )
    )
    source_ok = bool(coverage_batch) or (
        isinstance(source_result, Mapping)
        and source_result.get("status") in {"observed", "partial", "NA"}
        and collection.get("status") == "observed"
    )
    trace_incomplete = summary["framework"]["trace_incomplete_cases"] > 0
    target_wall_clock_pending = summary["framework"]["target_wall_clock_pending"] > 0
    right_censored = summary["framework"]["right_censored_case_count"] > 0
    status = "recorded" if (
        guest_ok and source_ok and not setup_gap
        and (not include_guest_metrics or not trace_incomplete)
        and not target_wall_clock_pending and not right_censored
    ) else "gap"
    reason = None if status == "recorded" else (
        setup_gap or (
            "trace-source-truncated" if include_guest_metrics and trace_incomplete else
            "framework-target-wall-clock-exhausted" if target_wall_clock_pending else
            "framework-right-censored" if right_censored else
            "framework-target-observation-missing" if not records else
            source_result.get("reason") if isinstance(source_result, Mapping) and not source_ok
            else "framework-guest-coverage-missing"
        )
    )
    return {
        "schema_version": "rq1-framework-coverage-v1",
        "status": status, "artifact_complete": status == "recorded",
        "summary": str(summary_path.relative_to(output)),
        "target_status": summary.get("status"), "target_records": len(records),
        "completed_case_count": len(complete_records),
        "right_censored_case_count": right_censored_case_count,
        "source_collection": collection, "simulator_source_coverage": source_result,
        "coverage_batch": coverage_batch,
        "reason": reason,
    }


def _merge_metric(rows: list[Mapping[str, object]], name: str) -> dict[str, object]:
    metrics = [row.get(name) for row in rows if isinstance(row.get(name), Mapping)]
    statuses = [metric.get("status") for metric in metrics]
    profile_ids = {str(metric.get("coverage_profile_id")) for metric in metrics
                   if metric.get("coverage_profile_id")}
    missing_metrics = max(0, len(rows) - len(metrics))
    if name == "PCov":
        status = (
            "gap" if any(item == "gap" for item in statuses) else
            "partial" if any(item == "partial" for item in statuses) else
            "observed" if statuses and all(item == "observed" for item in statuses) else
            "NA"
        )
        eligible: set[str] = set()
        covered: set[str] = set()
        eligible_by_artifact: dict[str, set[str]] = {}
        identity_missing_count = 0
        static_map_missing_count = 0
        invalid_unit_count = 0
        registry_shas = {
            str(metric.get("coverage_registry_sha256")) for metric in metrics
            if metric.get("coverage_registry_sha256")
        }
        static_map_conflict = False
        unit_pattern = re.compile(r"([0-9a-f]{64})@0x(0|[1-9a-f][0-9a-f]*)\Z")
        for metric in metrics:
            raw_eligible_count = metric.get("eligible")
            raw_covered_count = metric.get("covered")
            try:
                eligible_count, eligible_valid = _nonnegative_count(
                    raw_eligible_count, missing=None,
                )
                covered_count, covered_valid = _nonnegative_count(
                    raw_covered_count, missing=None,
                )
                if not eligible_valid or not covered_valid:
                    raise ValueError
                auxiliary_counts = [
                    metric.get("identity_missing_record_count"),
                    metric.get("static_map_missing_record_count"),
                    metric.get("invalid_unit_record_count"),
                ]
                auxiliary_values = []
                for value in auxiliary_counts:
                    parsed, valid = _nonnegative_count(value)
                    if not valid:
                        raise ValueError
                    auxiliary_values.append(parsed or 0)
                identity_missing_count += auxiliary_values[0]
                static_map_missing_count += auxiliary_values[1]
                invalid_unit_count += auxiliary_values[2]
                artifact_count, artifact_valid = _nonnegative_count(
                    metric.get("artifact_count"), missing=None,
                )
                hit_artifact_count, hit_artifact_valid = _nonnegative_count(
                    metric.get("artifacts_with_hits"), missing=None,
                )
                if not artifact_valid or not hit_artifact_valid:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                invalid_unit_count += 1
                continue
            eligible_values = metric.get("eligible_units")
            covered_values = metric.get("covered_units")
            if ((name == "PCov" and metric.get("aggregation") != "run-union-elf-pc")
                    or not isinstance(eligible_values, (list, tuple))
                    or not isinstance(covered_values, (list, tuple))
                    or not all(isinstance(unit, str)
                               for unit in [*eligible_values, *covered_values])):
                invalid_unit_count += 1
                continue
            eligible_set = set(eligible_values)
            covered_set = set(covered_values)
            valid = (
                not isinstance(raw_eligible_count, bool)
                and not isinstance(raw_covered_count, bool)
                and len(eligible_set) == len(eligible_values)
                and len(covered_set) == len(covered_values)
                and len(eligible_set) == eligible_count
                and len(covered_set) == covered_count
                and covered_set <= eligible_set
                and metric.get("eligible_units_sha256") == _pcov_unit_digest(eligible_set)
                and metric.get("covered_units_sha256") == _pcov_unit_digest(covered_set)
                and artifact_count == len({unit[:64] for unit in eligible_set})
                and hit_artifact_count == len({unit[:64] for unit in covered_set})
                and metric.get("status") in {"observed", "partial", "gap", "NA"}
            )
            per_artifact: dict[str, set[str]] = {}
            if valid:
                for unit in eligible_set | covered_set:
                    match = unit_pattern.fullmatch(unit)
                    if match is None or int(match.group(2), 16) >= 1 << 64:
                        valid = False
                        break
                for unit in eligible_set:
                    match = unit_pattern.fullmatch(unit)
                    if match:
                        per_artifact.setdefault(match.group(1), set()).add(unit)
            structural_valid = valid
            metric_status = metric.get("status")
            if valid and metric_status == "observed":
                try:
                    valid = (
                        eligible_count > 0 and metric.get("value") is not None
                        and abs(float(metric["value"]) - covered_count / eligible_count) <= 1.1e-6
                    )
                except (KeyError, TypeError, ValueError):
                    valid = False
            elif valid and metric.get("value") is not None:
                valid = False
            if structural_valid:
                # Preserve structurally verified units as numerator evidence;
                # a stale persisted ratio still forces a gap status.
                eligible.update(eligible_set)
                covered.update(covered_set)
                for artifact, artifact_units in per_artifact.items():
                    previous = eligible_by_artifact.get(artifact)
                    if previous is not None and previous != artifact_units:
                        static_map_conflict = True
                    else:
                        eligible_by_artifact[artifact] = artifact_units
            if not valid:
                invalid_unit_count += 1
                continue
        if len(eligible) != sum(len(units) for units in eligible_by_artifact.values()):
            static_map_conflict = True
        if static_map_conflict:
            status = "gap"
        if missing_metrics or identity_missing_count or static_map_missing_count or invalid_unit_count:
            status = "gap"
        merged = _summarize_pcov_run(
            covered, eligible, status,
            identity_missing_records=identity_missing_count,
            static_map_missing_records=static_map_missing_count,
            invalid_unit_records=invalid_unit_count + missing_metrics,
        )
        if len(profile_ids) == 1:
            merged["coverage_profile_id"] = next(iter(profile_ids))
        if len(registry_shas) == 1:
            merged["coverage_registry_sha256"] = next(iter(registry_shas))
        if static_map_conflict:
            merged.update(status="gap", value=None, reason="pcov-static-map-conflict")
        elif missing_metrics:
            merged.update(status="gap", value=None, reason="framework-pcov-metric-missing")
        elif status == "gap" and not merged.get("reason"):
            merged["reason"] = next((
                str(metric.get("reason")) for metric in metrics if metric.get("reason")
            ), "framework-pcov-coverage-gap")
        return merged

    profile_conflict = len(profile_ids) != 1
    profile_id = next(iter(profile_ids), None)
    registry = None
    if profile_id:
        parts = profile_id.split("|")
        if len(parts) == 3:
            try:
                candidate_registry = metric_registry_for_profile(parts[0], parts[1])
            except ValueError:
                candidate_registry = None
            if (candidate_registry is not None
                    and candidate_registry.get("metric_profile_id") == profile_id):
                registry = candidate_registry
            else:
                profile_conflict = True
        else:
            profile_conflict = True
    expected = set(registry["metrics"][name]) if registry is not None else set()
    covered: set[str] = set()
    invalid_units = 0
    status_values = []
    for metric in metrics:
        metric_status = metric.get("status")
        status_values.append(metric_status)
        raw_eligible = metric.get("eligible_units")
        raw_covered = metric.get("covered_units")
        if not isinstance(raw_eligible, (list, tuple)) or not isinstance(
            raw_covered, (list, tuple)
        ) or not all(isinstance(unit, str) for unit in [*raw_eligible, *raw_covered]):
            invalid_units += 1
            continue
        eligible_units = set(raw_eligible)
        covered_units = set(raw_covered)
        try:
            eligible_count, eligible_valid = _nonnegative_count(
                metric.get("eligible"), missing=None,
            )
            covered_count, covered_valid = _nonnegative_count(
                metric.get("covered"), missing=None,
            )
            if not eligible_valid or not covered_valid:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            invalid_units += 1
            continue
        valid = (
            eligible_count is not None and covered_count is not None
            and len(eligible_units) == len(raw_eligible)
            and len(covered_units) == len(raw_covered)
            and eligible_count == len(eligible_units)
            and covered_count == len(covered_units)
            and covered_units <= eligible_units
            and metric_status in {"observed", "partial", "gap", "NA"}
            and registry is not None and eligible_units == expected
            and metric.get("coverage_profile_id") == profile_id
            and metric.get("registry_sha256") == _pcov_unit_digest(expected)
        )
        structural_valid = valid
        if valid:
            if metric_status == "observed":
                try:
                    valid = (
                        eligible_count > 0 and metric.get("value") is not None
                        and abs(float(metric["value"]) - covered_count / eligible_count) <= 1.1e-6
                    )
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    valid = False
            elif metric.get("value") is not None:
                valid = False
        if structural_valid:
            # Preserve structurally verified units as numerator evidence;
            # a stale persisted ratio still forces a gap status.
            covered.update(covered_units & expected)
        if not valid:
            invalid_units += 1
            continue

    if not metrics:
        status = "NA"
    elif profile_conflict or invalid_units:
        status = "gap"
    elif any(item == "gap" for item in status_values):
        status = ("partial" if any(item in {"observed", "partial"} for item in status_values)
                  else "gap")
    elif missing_metrics or any(item in {"partial", "NA"} for item in status_values):
        status = "partial" if any(item in {"observed", "partial"} for item in status_values) else "NA"
    elif all(item == "observed" for item in status_values):
        status = "observed"
    else:
        status = "NA"
    result = {
        "covered": len(covered), "eligible": len(expected),
        "value": round(len(covered) / len(expected), 6)
        if expected and status == "observed" else None,
        "status": status, "covered_units": sorted(covered),
        "eligible_units": sorted(expected), "coverage_profile_id": profile_id,
        "registry_sha256": _pcov_unit_digest(expected) if expected else None,
    }
    if invalid_units:
        result["reason"] = "fixed-profile-unit-set-invalid"
    elif profile_conflict:
        result["reason"] = "coverage-profile-or-denominator-mismatch"
    elif missing_metrics:
        result["reason"] = "fixed-profile-metric-missing-from-case"
    elif status != "observed":
        result["reason"] = "framework-case-coverage-incomplete"
    return result

def _source_identity_key(identity: Mapping[str, object]) -> tuple[object, ...]:
    """覆盖 profile 的源码身份必须包含完整统计口径。"""
    def values(name: str) -> tuple[str, ...]:
        value = identity.get(name, ())
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return (str(value),) if value not in (None, "") else ()

    return (
        identity.get("source_repository"),
        identity.get("source_commit_expected"),
        identity.get("target_commit"),
        identity.get("coverage_binary_expected_sha256"),
        identity.get("source_root"),
        values("source_scope"),
        values("source_scope_exclude"),
    )


def merge_framework_coverage_summaries(
    output: Path, case_dirs: list[Path], target: str, *,
    include_guest_metrics: bool = True,
) -> dict[str, object]:
    """Union per-case guest units and simulator-source LCOV profiles."""
    summaries = []
    summary_cases: list[tuple[Path, Mapping[str, object]]] = []
    statuses = []
    campaign_incomplete_cases: set[Path] = set()
    right_censored_cases: set[Path] = set()
    source_case_statuses: list[str | None] = []
    source_incomplete_cases: set[Path] = set()
    case_level_censored: set[Path] = set()
    incomplete_guest_cases: set[Path] = set()
    target_strata: set[str] = set()
    trace_incomplete_cases = 0
    trace_incomplete_by_case: dict[Path, int] = defaultdict(int)
    target_wall_clock_pending = 0
    missing_target_summary_case_count = 0
    source_batch_cache: dict[Path, Mapping[str, object] | None] = {}

    def _case_target_framework(summary: Mapping[str, object]) -> Mapping[str, object]:
        framework = summary.get("framework")
        if isinstance(framework, Mapping) and isinstance(framework.get("targets"), Mapping):
            framework = framework["targets"].get(target, {})
        return framework if isinstance(framework, Mapping) else {}

    def _source_batch_summary(
        case_dir: Path, summary: Mapping[str, object],
    ) -> tuple[Mapping[str, object] | None, Path | None]:
        reference = _case_target_framework(summary).get("coverage_batch")
        if not isinstance(reference, Mapping):
            return None, None
        value = reference.get("summary_path")
        if not isinstance(value, str) or not value:
            return None, None
        path = Path(value)
        path = _path(path) if path.is_absolute() else case_dir / path
        path = path.resolve()
        if path not in source_batch_cache:
            try:
                source_batch_cache[path] = read_json_object(path)
            except (OSError, TypeError, ValueError):
                source_batch_cache[path] = None
        return source_batch_cache[path], path

    for case_dir in case_dirs:
        try:
            campaign = read_json_object(case_dir / "campaign-result.json")
        except (OSError, TypeError, ValueError):
            statuses.append(None)
            campaign_incomplete_cases.add(case_dir)
        else:
            coverage = campaign.get("framework_coverage")
            coverage = coverage if isinstance(coverage, Mapping) else {}
            coverage_by_target = coverage.get("targets")
            target_coverage = (
                coverage_by_target.get(target)
                if isinstance(coverage_by_target, Mapping) else None
            )
            target_coverage = target_coverage if isinstance(target_coverage, Mapping) else coverage
            coverage_status = target_coverage.get("status")
            statuses.append(coverage_status)
            if coverage_status != "recorded":
                campaign_incomplete_cases.add(case_dir)
            counts = campaign.get("counts")
            if isinstance(target_coverage.get("target_wall_clock_pending"), int):
                if target_coverage["target_wall_clock_pending"]:
                    right_censored_cases.add(case_dir)
            elif isinstance(counts, Mapping) and not isinstance(coverage_by_target, Mapping):
                pending, pending_valid = _nonnegative_count(
                    counts.get("target_wall_clock_pending"), missing=None,
                )
                if pending_valid and pending:
                    right_censored_cases.add(case_dir)
            reason = target_coverage.get("reason")
            if include_guest_metrics and isinstance(reason, str) and reason in {
                    "trace-source-truncated", "framework-target-wall-clock-exhausted",
                    "framework-target-observation-missing", "framework-guest-coverage-missing",
                }:
                incomplete_guest_cases.add(case_dir)
            elif include_guest_metrics and coverage_status != "recorded" \
                    and not isinstance(reason, str):
                # An unclassified non-recorded result cannot establish a closed
                # guest run. Known source-only reasons remain independent from
                # guest unit evidence.
                incomplete_guest_cases.add(case_dir)
        try:
            case_result = read_json_object(case_dir / "case-result.json")
        except (OSError, TypeError, ValueError):
            case_result = {}
        pending_by_target = case_result.get("target_wall_clock_pending_by_target")
        target_pending_valid = isinstance(pending_by_target, Mapping)
        target_pending = None
        if target_pending_valid:
            target_pending, target_pending_valid = _nonnegative_count(
                pending_by_target.get(target, 0), missing=0,
            )
        case_right_censored = (
            bool(target_pending) if target_pending_valid
            else case_result.get("right_censored") is True
            or case_result.get("case_outcome") == "right-censored"
        )
        if case_right_censored:
            right_censored_cases.add(case_dir)
            case_level_censored.add(case_dir)
            incomplete_guest_cases.add(case_dir)
        try:
            summary = read_json_object(case_dir / "coverage" / "summary.json")
        except (OSError, TypeError, ValueError):
            source_case_statuses.append(None)
            source_incomplete_cases.add(case_dir)
            incomplete_guest_cases.add(case_dir)
            continue
        if summary.get("schema_version") != BATCH_SCHEMA:
            source_case_statuses.append(None)
            source_incomplete_cases.add(case_dir)
            incomplete_guest_cases.add(case_dir)
            continue
        summaries.append(summary)
        summary_cases.append((case_dir, summary))
        if not any(
            isinstance(row, Mapping) and str(row.get("target")) == target
            for row in summary.get("targets", ())
        ):
            missing_target_summary_case_count += 1
        for row in summary.get("targets", ()):
            if isinstance(row, Mapping) and str(row.get("target")) == target:
                row_stratum = row.get("stratum")
                if row_stratum in {"bare-metal", "linux-user"}:
                    target_strata.add(str(row_stratum))
        framework = summary.get("framework")
        if isinstance(framework, Mapping) and isinstance(framework.get("targets"), Mapping):
            framework = framework["targets"].get(target, {})
        if not isinstance(framework, Mapping):
            if include_guest_metrics:
                incomplete_guest_cases.add(case_dir)
        else:
            for name, total in (
                ("trace_incomplete_cases", "trace"),
                ("target_wall_clock_pending", "pending"),
                ("right_censored_case_count", "right-censored"),
            ):
                count, valid_count = _nonnegative_count(framework.get(name))
                if not valid_count:
                    count = 1
                else:
                    count = count or 0
                if total == "trace":
                    trace_incomplete_cases += count
                    trace_incomplete_by_case[case_dir] += count
                elif total == "pending":
                    target_wall_clock_pending += count
                    if count:
                        right_censored_cases.add(case_dir)
                elif total == "right-censored" and count:
                    right_censored_cases.add(case_dir)
                if count:
                    incomplete_guest_cases.add(case_dir)
        target_rows = summary.get("targets")
        target_rows = target_rows if isinstance(target_rows, (list, tuple)) else ()
        target_rows = [
            row for row in target_rows
            if isinstance(row, Mapping) and str(row.get("target")) == target
        ]
        # A gap in one metric (for example PCov identity) must not erase other
        # metrics whose own observation and denominator are valid.
        if not target_rows:
            incomplete_guest_cases.add(case_dir)
        batch_summary, _batch_path = _source_batch_summary(case_dir, summary)
        source_container = (
            batch_summary.get("source_targets", ())
            if isinstance(batch_summary, Mapping) else summary.get("source_targets", ())
        )
        source_rows = []
        for row in source_container:
            if not isinstance(row, Mapping) or str(row.get("target")) != target:
                continue
            source = row.get("simulator_source_coverage")
            if isinstance(source, Mapping):
                source_rows.append(str(source.get("status") or "gap"))
            else:
                source_rows.append("gap")
        source_status = (
            "observed" if source_rows and all(item == "observed" for item in source_rows)
            else "partial" if any(item in {"observed", "partial"} for item in source_rows)
            else "NA" if source_rows and all(item == "NA" for item in source_rows)
            else "gap" if source_rows else None
        )
        if source_status not in {"observed", "partial", "NA"}:
            source_incomplete_cases.add(case_dir)
        source_case_statuses.append(source_status)

    completed_summaries = [
        summary for case_dir, summary in summary_cases
        if case_dir not in incomplete_guest_cases
        and case_dir not in campaign_incomplete_cases
    ]
    # Every parseable case contributes its units. Incomplete prefixes remain
    # useful numerator evidence; status and ratios are gated below.
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    sources: list[Mapping[str, object]] = []
    source_batch_keys: set[tuple[str, str]] = set()
    for case_dir, summary in summary_cases:
        for row in summary.get("targets", ()):
            if isinstance(row, Mapping) and str(row.get("target")) == target:
                key = tuple(str(row.get(name) or "") for name in ("method", "route", "lane", "stratum", "target"))
                grouped[key].append(row)
    for case_dir, summary in summary_cases:
        batch_summary, batch_path = _source_batch_summary(case_dir, summary)
        source_container = (
            batch_summary.get("source_targets", ())
            if isinstance(batch_summary, Mapping) else summary.get("source_targets", ())
        )
        profile_base = batch_path.parent if batch_path is not None else case_dir
        for row in source_container:
            if isinstance(row, Mapping) and str(row.get("target")) == target:
                source = row.get("simulator_source_coverage")
                if isinstance(source, Mapping):
                    source = dict(source)
                    source.update({
                        name: row.get(name)
                        for name in (
                            "method", "route", "lane", "stratum", "target",
                            "scope", "batch_id", "cases",
                        )
                        if name in row
                    })
                    profile = source.get("profile_path")
                    if isinstance(profile, str) and not Path(profile).is_absolute():
                        source["profile_path"] = str((profile_base / profile).resolve())
                    batch_key = (
                        str(source.get("target") or target),
                        str(source.get("batch_id") or ""),
                    )
                if source.get("scope") in {"target-batch", "method-run"} \
                        and batch_key in source_batch_keys:
                    continue
                if source.get("scope") in {"target-batch", "method-run"}:
                    source_batch_keys.add(batch_key)
                    sources.append(source)

    missing_case_count = max(0, len(case_dirs) - len(summaries))
    missing_campaign_coverage_case_count = sum(status is None for status in statuses)
    incomplete_guest_case_count = len(incomplete_guest_cases)
    campaign_incomplete_case_count = len(campaign_incomplete_cases)
    timebox_trace_incomplete_cases = sum(
        count for case_dir, count in trace_incomplete_by_case.items()
        if case_dir in right_censored_cases
    )
    unexpected_trace_incomplete_cases = max(
        0, trace_incomplete_cases - timebox_trace_incomplete_cases,
    )
    # 整轮时限造成的右删失是预设停止点，不是覆盖率缺口：它只说明这一轮在
    # 共享截止线上结束，不能把已闭合样例的比率降级。只有非时限删失的缺口
    # 才算覆盖率不完整。
    timebox_censored_cases = set(case_level_censored)
    unexpected_incomplete_guest = incomplete_guest_cases - timebox_censored_cases
    unexpected_missing_campaign = {
        case_dir for case_dir, status in zip(case_dirs, statuses)
        if status is None
    } - timebox_censored_cases
    # Guest/observer completeness is a separate channel from simulator source
    # coverage.  The latter is allowed to proceed even when a target has no
    # RVOBS1/guest-PC frame (for example RAX/RVVM).  Source completeness is
    # computed after the per-case source profiles are collected below.
    guest_coverage_incomplete = bool(
        unexpected_incomplete_guest or unexpected_missing_campaign
    )
    coverage_incomplete = guest_coverage_incomplete if include_guest_metrics else False
    rows = []
    for (method, route, lane, stratum, target_id), items in sorted(grouped.items()):
        statuses_set = {str(item.get("status")) for item in items}
        status = "observed" if statuses_set == {"observed"} else (
            "partial" if statuses_set & {"observed", "partial"} else
            "gap" if statuses_set == {"gap"} else "NA"
        )
        guest = ({name: _merge_metric(
            [item.get("guest", {}) for item in items if isinstance(item.get("guest"), Mapping)], name
        ) for name in ("PCov", "ICov-encoding", "ICov-type", "CCov")}
            if include_guest_metrics else {})
        target_identity_ids = {
            str(item.get("target_identity_id")) for item in items
            if item.get("target_identity_id")
        }
        identity_mismatch = include_guest_metrics and len(target_identity_ids) > 1
        if coverage_incomplete \
                and status in {"observed", "partial"}:
            # Preserve counts and units from available samples, but do not
            # expose their ratio as a complete-run value.
            status = "partial"
            for metric in guest.values():
                if metric.get("status") in {"observed", "partial"}:
                    metric["status"] = "partial"
                    metric["value"] = None
                    metric.setdefault("reason", "framework-case-coverage-incomplete")
        metric_gap_reasons = [
            str(metric.get("reason") or "framework-guest-coverage-gap")
            for metric in guest.values() if metric.get("status") == "gap"
        ]
        profile_mismatch = any(
            metric.get("status") == "gap"
            and "mismatch" in str(metric.get("reason") or "")
            for metric in guest.values()
        )
        if profile_mismatch or identity_mismatch:
            status = "gap"
            reason = (
                "target-identity-mismatch" if identity_mismatch
                else "coverage-profile-or-denominator-mismatch"
            )
            for name, metric in guest.items():
                if name == "PCov" and not identity_mismatch:
                    continue
                metric.update(value=None, status="gap", reason=reason)
        elif metric_gap_reasons:
            status = "gap"
            reason = sorted(metric_gap_reasons)[0]
        else:
            reason = None
        events: dict[str, int] = defaultdict(int)
        for item in items:
            raw_events = item.get("simulator_event_counts")
            if raw_events is None:
                continue
            if not isinstance(raw_events, Mapping):
                reason = reason or "framework-simulator-event-counts-invalid"
                status = "gap"
                continue
            for name, value in raw_events.items():
                if not isinstance(name, str):
                    reason = reason or "framework-simulator-event-counts-invalid"
                    status = "gap"
                    continue
                count, valid = _nonnegative_count(value, missing=None)
                if not valid or count is None:
                    reason = reason or "framework-simulator-event-counts-invalid"
                    status = "gap"
                    continue
                events[str(name)] += count

        def count_field(item: Mapping[str, object], name: str) -> int:
            value = item.get(name)
            count, valid = _nonnegative_count(value)
            return count or 0 if valid else 0

        rows.append({
            "method": method or None, "route": route or None, "lane": lane or None,
            "stratum": stratum or None, "target": target_id,
            "target_identity_id": next(iter(target_identity_ids), None),
            "cases": sum(count_field(item, "cases") for item in items),
            "observed_cases": sum(count_field(item, "observed_cases") for item in items),
            "incomplete_cases": sum(count_field(item, "incomplete_cases") for item in items),
            "missing_coverage_summary_cases": missing_case_count,
            "status": status, "simulator_event_counts": dict(sorted(events.items())),
            "reason": reason,
            "coverage_basis": (
                {
                    "PCov": "run-union-elf-pc", "ICov-encoding": "fixed-profile-universe",
                    "ICov-type": "fixed-profile-universe", "CCov": "fixed-profile-universe",
                    "coverage_profile_id": next((
                        metric.get("coverage_profile_id") for metric in guest.values()
                        if metric.get("coverage_profile_id")
                    ), None),
                    "coverage_registry_sha256": next((
                        metric.get("coverage_registry_sha256") or metric.get("registry_sha256")
                        for metric in guest.values()
                        if metric.get("coverage_registry_sha256") or metric.get("registry_sha256")
                    ), None),
                }
                if include_guest_metrics else {
                    "SimSrcCov": "target-run-source-profile",
                }
            ),
            "guest": guest,
            "translator_coverage": {
                "covered": None, "eligible": None, "value": None,
                "status": "NA", "reason": "translator-feature-channel-unavailable",
            },
        })
    source = max(
        sources,
        key=lambda item: {"observed": 3, "partial": 2, "gap": 1}.get(
            str(item.get("status")), 0
        ),
        default=None,
    )
    source_profiles: dict[tuple[object, ...], list[Path]] = defaultdict(list)
    all_source_profiles: set[Path] = set()
    for item in sources:
        profile = item.get("profile_path")
        identity = item.get("identity")
        if not isinstance(profile, str):
            continue
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = output / profile_path
        if not profile_path.is_file():
            continue
        if not isinstance(identity, Mapping):
            continue
        key = _source_identity_key(identity)
        path = profile_path.resolve()
        all_source_profiles.add(path)
        if path not in source_profiles[key]:
            source_profiles[key].append(path)
    source_incomplete = bool(
        len(source_case_statuses) != len(case_dirs)
        or (source_incomplete_cases - timebox_censored_cases)
    )
    source_coverage_incomplete = source_incomplete
    if not include_guest_metrics:
        # Do not reuse guest trace/observer gaps as source-profile gaps.  A
        # missing source profile still remains incomplete through this flag.
        coverage_incomplete = source_coverage_incomplete
    source_profile_complete = False
    if source_profiles:
        identity_key, profiles = max(source_profiles.items(), key=lambda item: len(item[1]))
        if source is None or not isinstance(source.get("identity"), Mapping):
            source = None
        else:
            source_identity = source["identity"]
            source_key = _source_identity_key(source_identity)
            if source_key != identity_key:
                source = next(
                    (
                        item for item in sources
                        if isinstance(item.get("identity"), Mapping)
                        and _source_identity_key(item["identity"]) == identity_key
                    ),
                    source,
                )
        if source is not None:
            identity = source.get("identity")
            merged_profile = output / "coverage" / "simulator-source" / "lcov.info"
            code, error = _merge_lcov_profiles(profiles, merged_profile)
            if code == 0 and isinstance(identity, Mapping):
                source_target = {
                    "id": target,
                    "commit": identity.get("target_commit"),
                    "coverage_binary": identity.get("coverage_binary"),
                    "coverage_binary_sha256": identity.get(
                        "coverage_binary_expected_sha256"
                    ),
                    "source_coverage": {
                        "profile": str(merged_profile),
                        "format": identity.get("profile_format", "lcov"),
                        "source_repository": identity.get("source_repository"),
                        "source_commit": identity.get("source_commit_expected"),
                        "target_commit": identity.get("target_commit"),
                        "coverage_binary_sha256": identity.get(
                            "coverage_binary_expected_sha256"
                        ),
                        "source_scope": identity.get("source_scope", []),
                        "source_scope_exclude": identity.get(
                            "source_scope_exclude", []
                        ),
                        "source_root": identity.get("source_root"),
                        "toolchain": identity.get("toolchain"),
                        "build_flags": identity.get("build_flags", []),
                    },
                }
                source = target_source_coverage(
                    output,
                    source_target,
                    identity=dict(identity),
                    source_commit_observed=identity.get("source_commit_observed"),
                    coverage_binary_sha256=identity.get("coverage_binary_sha256"),
                )
                if (
                    len(profiles) < len(all_source_profiles)
                    or source_incomplete
                ):
                    source["status"] = "partial" if source.get("status") == "observed" else source.get("status")
                    source["reason"] = source.get("reason") or "framework-source-coverage-incomplete"
                source_profile_complete = (
                    source.get("status") in {"observed", "partial", "NA"}
                    and len(profiles) == len(all_source_profiles)
                    and not source_incomplete
                )
            elif source is not None:
                source = {
                    **source,
                    "status": "gap",
                    "reason": error or "framework-source-coverage-merge-failed",
                }
    elif source is not None and source.get("status") in {"observed", "partial"}:
        source = {
            **source,
            "status": "gap",
            "reason": "framework-source-coverage-profile-missing",
        }
    if isinstance(source, Mapping) and source.get("status") == "gap":
        source = {
            **source,
            "reason": _error_summary(str(source.get("reason") or ""))
            or "framework-source-coverage-incomplete",
        }
    if target_wall_clock_pending:
        if isinstance(source, Mapping) and source.get("status") in {"observed", "partial"}:
            source = {
                **source,
                "status": "partial",
                "reason": "framework-target-wall-clock-exhausted",
            }
        source_profile_complete = False
    complete = (
        bool(statuses)
        and (
            not include_guest_metrics
            or all(
                status == "recorded" or case_dir in timebox_censored_cases
                for case_dir, status in zip(case_dirs, statuses)
            )
        )
        and source_profile_complete
        and (
            (
                not include_guest_metrics
                and target_wall_clock_pending == 0
                and not source_coverage_incomplete
            )
            or (
                include_guest_metrics
                and unexpected_trace_incomplete_cases == 0
                and target_wall_clock_pending == 0
                and not unexpected_incomplete_guest
            )
        )
        and len(summaries) >= len(case_dirs) - len(timebox_censored_cases)
        and bool(rows) and all(row.get("status") == "observed" for row in rows)
    )
    guest_evidence = any(
        metric.get("eligible", 0)
        and metric.get("status") in {"observed", "partial"}
        for row in rows for metric in row.get("guest", {}).values()
    )
    source_evidence = bool(source_profiles) and isinstance(source, Mapping) \
        and any(
            metric.get("eligible", 0)
            for metric in (source.get("metrics") or {}).values()
            if isinstance(metric, Mapping)
        )
    evidence_available = guest_evidence or source_evidence
    timebox_only = bool(case_dirs) and len(right_censored_cases) == len(case_dirs) \
        and unexpected_trace_incomplete_cases == 0 \
        and set(incomplete_guest_cases) <= right_censored_cases \
        and set(campaign_incomplete_cases) <= right_censored_cases \
        and not any(row.get("status") == "gap" for row in rows)
    summary_status = (
        "observed" if complete else
        "gap" if any(row.get("status") == "gap" for row in rows) else
        "partial" if evidence_available or timebox_only else "gap"
    )
    status = (
        "recorded" if complete else
        "partial" if evidence_available or timebox_only else "gap"
    )
    reason = None if complete else (
        "trace-source-truncated"
        if include_guest_metrics and unexpected_trace_incomplete_cases else
        "framework-target-wall-clock-exhausted" if target_wall_clock_pending else
        "framework-right-censored" if right_censored_cases else
        "framework-campaign-coverage-incomplete"
        if include_guest_metrics and campaign_incomplete_case_count else
        "framework-source-coverage-incomplete"
        if not include_guest_metrics and source_coverage_incomplete else
        "framework-case-coverage-incomplete" if incomplete_guest_case_count else
        _error_summary(str(source.get("reason") or ""))
        if isinstance(source, Mapping) and source.get("status") == "gap" else
        "framework-source-coverage-incomplete"
        if source is not None and source.get("status") == "gap" else
        "framework-case-coverage-incomplete"
    )
    experiment_unions = []
    evidence_case_counts: dict[str, int] = defaultdict(int)
    for items in grouped.values():
        for case_row in items:
            count, valid = _nonnegative_count(case_row.get("cases"))
            if valid:
                evidence_case_counts[str(case_row.get("method") or "")] += int(count or 0)
    evidence_rows = [item for items in grouped.values() for item in items]
    for method_key in sorted({str(item.get("method") or "") for item in evidence_rows}):
        method_rows = [
            item for item in evidence_rows if (item.get("method") or "") == method_key
        ]
        if include_guest_metrics:
            union = summarize_experiment_union(
                method_key or None, method_rows,
                profile_ids={
                    str(profile_id)
                    for row in method_rows
                    for profile_id in [(row.get("coverage_basis") or {}).get(
                        "coverage_profile_id")]
                    if profile_id
                },
            )
        else:
            method_sources = [
                item for item in sources
                if item.get("method") == method_key
            ]
            union = summarize_source_experiment_union(
                method_key or None, method_rows, source_rows=method_sources,
            )
        if coverage_incomplete and include_guest_metrics:
            # The fixed denominator and all safe numerator units remain, but
            # the run-level ratio is not a complete campaign result.
            union["status"] = "partial"
            for metric in (union.get("guest") or {}).values():
                if isinstance(metric, dict) and metric.get("status") in {
                    "observed", "partial",
                }:
                    metric.update(
                        status="partial", value=None,
                        reason="framework-case-coverage-incomplete",
                    )
        # The union contains every parseable case. The run-level completion
        # gate remains separate in ``framework.completed_case_count``.
        union["target_case_records"] = evidence_case_counts.get(method_key, 0)
        experiment_unions.append(union)
    target_stratum = (
        next(iter(target_strata)) if len(target_strata) == 1
        else _target_stratum({"id": target})
    )
    summary = {
        "schema_version": BATCH_SCHEMA,
        "status": summary_status, "reason": reason,
        "guest_metrics_enabled": include_guest_metrics,
        "comparability": "target-local-corpus",
        "experiment_unions": experiment_unions,
        "targets": rows,
        "source_targets": ([{"target": target, "stratum": target_stratum, "scope": "method-run",
                              "simulator_source_coverage": source}] if source else []),
        "framework": {
            "case_count": len(case_dirs),
            "coverage_summary_case_count": len(summaries),
            "completed_case_count": len(completed_summaries),
            "right_censored_case_count": len(right_censored_cases),
            "missing_case_count": missing_case_count,
            "missing_target_summary_case_count": missing_target_summary_case_count,
            "missing_campaign_coverage_case_count": missing_campaign_coverage_case_count,
            "incomplete_guest_case_count": incomplete_guest_case_count,
            "campaign_incomplete_case_count": campaign_incomplete_case_count,
            "target": target,
            "trace_incomplete_cases": trace_incomplete_cases,
            "target_wall_clock_pending": target_wall_clock_pending,
        },
    }
    rv_records = []
    opcode_catalog_records = []
    for _case_dir, case_summary in summary_cases:
        for case_row in case_summary.get("targets", ()):
            if not isinstance(case_row, Mapping):
                continue
            if str(case_row.get("target")) != target:
                continue
            rv = case_row.get("rv_instruction_coverage")
            if isinstance(rv, Mapping):
                rv_records.append({
                    "method": case_row.get("method"),
                    "route": case_row.get("route"),
                    "lane": case_row.get("lane"),
                    "stratum": case_row.get("stratum"),
                    "target": case_row.get("target"),
                    "simulator_coverage": {"rv_instruction_coverage": rv},
                })
            opcode_catalog = case_row.get("rv_opcode_catalog_coverage")
            if isinstance(opcode_catalog, Mapping):
                opcode_catalog_records.append({
                    "method": case_row.get("method"),
                    "route": case_row.get("route"),
                    "lane": case_row.get("lane"),
                    "stratum": case_row.get("stratum"),
                    "target": case_row.get("target"),
                    "rv_opcode_catalog_coverage": opcode_catalog,
                })
    if include_guest_metrics:
        aggregate_rv_instruction_coverage(summary, rv_records)
    if include_guest_metrics:
        aggregate_opcode_catalog_summary(summary, opcode_catalog_records)
    if missing_case_count or missing_target_summary_case_count:
        incomplete_reason = (
            "framework-case-coverage-incomplete" if missing_case_count
            else "framework-target-case-coverage-missing"
        )
        for row in summary.get("targets", ()):
            if not isinstance(row, dict):
                continue
            catalog = row.get("rv_opcode_catalog_coverage")
            metrics = catalog.get("metrics") if isinstance(catalog, Mapping) else None
            if not isinstance(metrics, dict):
                continue
            for metric in metrics.values():
                if not isinstance(metric, dict) or metric.get("status") == "gap":
                    continue
                metric.update(
                    status="partial", value=None,
                    reason=incomplete_reason,
                )
            catalog_statuses = {
                metric.get("status") for metric in metrics.values()
                if isinstance(metric, Mapping)
            }
            catalog["status"] = (
                "gap" if "gap" in catalog_statuses else
                "partial" if catalog_statuses else "gap"
            )
        summarize_opcode_catalog_target_rows(summary)
    if include_guest_metrics and coverage_incomplete:
        _mark_rv_coverage_incomplete(
            summary, reason or "framework-case-coverage-incomplete",
            include_targets=False,
        )
    output = Path(output).resolve()
    summary_path = output / "coverage" / "summary.json"
    atomic_write_json(summary_path, summary)
    return {
        "schema_version": "rq1-framework-coverage-v1", "status": status,
        "artifact_complete": complete, "summary": str(summary_path.relative_to(output)),
        "case_count": len(case_dirs),
        "coverage_summary_case_count": len(summaries),
        "completed_case_count": len(completed_summaries),
        "right_censored_case_count": len(right_censored_cases),
        "missing_case_count": missing_case_count,
        "missing_target_summary_case_count": missing_target_summary_case_count,
        "missing_campaign_coverage_case_count": missing_campaign_coverage_case_count,
        "incomplete_guest_case_count": incomplete_guest_case_count,
        "campaign_incomplete_case_count": campaign_incomplete_case_count,
        "target_wall_clock_pending": target_wall_clock_pending,
        "reason": reason,
    }


def merge_framework_raw_coverage_batches(
    output: Path, batch_summary_paths: list[Path], target: str, *,
    include_guest_metrics: bool = False,
) -> dict[str, object]:
    """Union source profiles from batches that contain real Target executions.

    Raw-only framework runs do not create one ``coverage/summary.json`` per
    producer case.  Reusing ``merge_framework_coverage_summaries`` for those
    runs therefore treats every pending producer case as a missing coverage
    case.  Batch summaries are the correct unit here: their ``case_count`` is
    already filtered by the Target-owned raw sidecar boundary.
    """
    output = Path(output).resolve()
    entries: list[dict[str, object]] = []
    for summary_path in sorted({Path(path).resolve() for path in batch_summary_paths}):
        try:
            summary = read_json_object(summary_path)
        except (OSError, TypeError, ValueError):
            continue
        if summary.get("schema_version") not in {
            BATCH_SCHEMA, "rq1-framework-source-coverage-batch-v1",
        }:
            continue
        rows = summary.get("source_targets")
        rows = rows if isinstance(rows, (list, tuple)) else ()
        for row in rows:
            if not isinstance(row, Mapping) or str(row.get("target")) != target:
                continue
            coverage = row.get("simulator_source_coverage")
            if not isinstance(coverage, Mapping):
                continue
            count, valid_count = _nonnegative_count(row.get("cases"), missing=0)
            if not valid_count or not count:
                # A zero-case summary is a collector state record, not source
                # coverage evidence for a pending queue item.
                continue
            normalized_coverage = dict(coverage)
            profile_value = normalized_coverage.get("profile_path")
            if isinstance(profile_value, str) and profile_value:
                profile_path = Path(profile_value)
                if not profile_path.is_absolute():
                    profile_path = summary_path.parent / profile_path
                normalized_coverage["profile_path"] = str(profile_path.resolve())
            entry = dict(row)
            entry["cases"] = int(count or 0)
            entry["simulator_source_coverage"] = normalized_coverage
            entries.append(entry)
            break

    def coverage_status(entry: Mapping[str, object]) -> str:
        coverage = entry.get("simulator_source_coverage")
        value = coverage.get("status") if isinstance(coverage, Mapping) else None
        return value if value in {"observed", "partial", "gap", "NA"} else "gap"

    groups: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for entry in entries:
        key = tuple(str(entry.get(name) or "") for name in (
            "method", "route", "lane", "stratum", "target",
        ))
        group = groups.setdefault(key, {
            "first": entry, "cases": 0, "observed_cases": 0,
            "incomplete_cases": 0, "statuses": [],
        })
        count = int(entry.get("cases", 0) or 0)
        status = coverage_status(entry)
        group["cases"] += count
        group["observed_cases"] += count if status == "observed" else 0
        group["incomplete_cases"] += count if status != "observed" else 0
        group["statuses"].append(status)

    valid_sources: list[tuple[Mapping[str, object], Path, Mapping[str, object]]] = []
    invalid_source_count = 0
    source_profiles: dict[tuple[object, ...], list[Path]] = defaultdict(list)
    all_source_profiles: set[Path] = set()
    for entry in entries:
        coverage = entry.get("simulator_source_coverage")
        profile_value = coverage.get("profile_path") \
            if isinstance(coverage, Mapping) else None
        identity = coverage.get("identity") \
            if isinstance(coverage, Mapping) else None
        profile_path = Path(profile_value) \
            if isinstance(profile_value, str) and profile_value else None
        if profile_path is None or not profile_path.is_file() \
                or not isinstance(identity, Mapping):
            invalid_source_count += 1
            continue
        profile_path = profile_path.resolve()
        all_source_profiles.add(profile_path)
        identity_key = _source_identity_key(identity)
        if profile_path not in source_profiles[identity_key]:
            source_profiles[identity_key].append(profile_path)
        valid_sources.append((
            coverage if isinstance(coverage, Mapping) else {},
            profile_path, identity,
        ))

    source: Mapping[str, object] | None = None
    source_profile_complete = False
    merge_error: str | None = None
    if source_profiles:
        identity_key, profiles = max(
            source_profiles.items(), key=lambda item: len(item[1]),
        )
        selected = next(
            (item for item in valid_sources
             if _source_identity_key(item[2]) == identity_key),
            None,
        )
        if selected is not None:
            _coverage, _profile, identity = selected
            merged_profile = output / "coverage" / "simulator-source" / "lcov.info"
            if len(profiles) == 1:
                merged_profile.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(profiles[0], merged_profile)
                    code, merge_error = 0, None
                except OSError as error:
                    code, merge_error = 2, f"lcov-profile-copy-failed:{type(error).__name__}"
            else:
                code, merge_error = _merge_lcov_profiles(profiles, merged_profile)
            if code == 0:
                identity_path = merged_profile.with_name("identity.json")
                atomic_write_json(identity_path, dict(identity))
                source_target = {
                    "id": target,
                    "commit": identity.get("target_commit"),
                    "coverage_binary": identity.get("coverage_binary"),
                    "coverage_binary_sha256": identity.get(
                        "coverage_binary_expected_sha256"
                        or "coverage_binary_sha256"
                    ),
                    "source_coverage": {
                        "profile": str(merged_profile),
                        "identity": str(identity_path),
                        "format": identity.get("profile_format", "lcov"),
                        "source_repository": identity.get("source_repository"),
                        "source_commit": identity.get("source_commit_expected"),
                        "target_commit": identity.get("target_commit"),
                        "coverage_binary_sha256": identity.get(
                            "coverage_binary_expected_sha256"
                            or "coverage_binary_sha256"
                        ),
                        "source_scope": identity.get("source_scope", []),
                        "source_scope_exclude": identity.get(
                            "source_scope_exclude", []
                        ),
                        "source_root": identity.get("source_root"),
                        "toolchain": identity.get("toolchain"),
                        "build_flags": identity.get("build_flags", []),
                    },
                }
                source_value = target_source_coverage(
                    output, source_target, identity=dict(identity),
                    source_commit_observed=identity.get("source_commit_observed"),
                    coverage_binary_sha256=identity.get("coverage_binary_sha256"),
                )
                source = source_value
                source_profile_complete = (
                    not invalid_source_count
                    and len(profiles) == len(all_source_profiles)
                    and all(coverage_status(entry) == "observed" for entry in entries)
                    and source_value.get("status") in {"observed", "partial", "NA"}
                )
                if not source_profile_complete:
                    source = {
                        **source_value,
                        "status": "partial"
                        if source_value.get("status") in {"observed", "partial"}
                        else source_value.get("status"),
                        "reason": source_value.get("reason")
                        or "framework-source-coverage-incomplete",
                    }
            else:
                merge_error = merge_error or "framework-source-coverage-merge-failed"

    if source is None and entries:
        candidate = max(
            (entry.get("simulator_source_coverage") for entry in entries),
            key=lambda item: {"observed": 3, "partial": 2, "gap": 1}.get(
                str(item.get("status")) if isinstance(item, Mapping) else "gap", 0,
            ),
            default={},
        )
        source = {
            **(dict(candidate) if isinstance(candidate, Mapping) else {}),
            "status": "gap",
            "reason": merge_error or "framework-source-coverage-profile-missing",
        }

    target_rows: list[dict[str, object]] = []
    source_row: dict[str, object] | None = None
    for key, group in sorted(groups.items()):
        first = group["first"]
        statuses = group["statuses"]
        status = (
            "observed" if statuses and all(item == "observed" for item in statuses)
            else "partial" if any(item in {"observed", "partial"} for item in statuses)
            else "gap"
        )
        target_rows.append({
            "method": first.get("method"), "route": first.get("route"),
            "lane": first.get("lane"), "stratum": first.get("stratum"),
            "target": target, "cases": group["cases"],
            "observed_cases": group["observed_cases"],
            "incomplete_cases": group["incomplete_cases"],
            "missing_coverage_summary_cases": group["incomplete_cases"],
            "status": status, "guest": {},
            "coverage_basis": {"SimSrcCov": "target-run-source-profile"},
            "translator_coverage": {
                "covered": None, "eligible": None, "value": None,
                "status": "NA", "reason": "translator-feature-channel-unavailable",
            },
        })

    if source is not None and target_rows:
        first = target_rows[0]
        source_row = {
            "method": first.get("method"), "route": first.get("route"),
            "lane": first.get("lane"), "stratum": first.get("stratum"),
            "target": target, "scope": "method-run", "batch_id": "raw-merged",
            "cases": sum(int(row.get("cases", 0) or 0) for row in target_rows),
            "simulator_source_coverage": source,
        }

    method_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in target_rows:
        method_rows[str(row.get("method") or "")].append(row)
    experiment_unions = []
    for method_name, rows in sorted(method_rows.items()):
        experiment_unions.append(summarize_source_experiment_union(
            method_name or None, rows,
            source_rows=[source_row] if source_row is not None else [],
        ))

    source_metrics = source.get("metrics") if isinstance(source, Mapping) else None
    source_metrics = source_metrics if isinstance(source_metrics, Mapping) else {}
    evidence_available = source is not None and any(
        isinstance(metric, Mapping) and metric.get("eligible")
        for metric in source_metrics.values()
    )
    complete = bool(entries) and bool(target_rows) and source_profile_complete \
        and all(row.get("status") == "observed" for row in target_rows) \
        and not include_guest_metrics
    reason = None if complete else (
        "framework-raw-only-guest-metrics-not-materialized"
        if include_guest_metrics else
        "framework-source-coverage-no-target-executions" if not entries else
        "framework-source-coverage-incomplete"
    )
    summary_status = "observed" if complete else "partial" if evidence_available else "gap"
    status = "recorded" if complete else "partial" if evidence_available else "gap"
    total_cases = sum(int(row.get("cases", 0) or 0) for row in target_rows)
    observed_cases = sum(int(row.get("observed_cases", 0) or 0) for row in target_rows)
    summary = {
        "schema_version": BATCH_SCHEMA,
        "status": summary_status, "reason": reason,
        "guest_metrics_enabled": include_guest_metrics,
        "comparability": "target-local-corpus",
        "experiment_unions": experiment_unions,
        "targets": target_rows,
        "source_targets": [source_row] if source_row is not None else [],
        "framework": {
            "case_count": total_cases,
            "coverage_summary_case_count": len(entries),
            "completed_case_count": observed_cases,
            "right_censored_case_count": 0,
            "missing_case_count": max(0, total_cases - observed_cases),
            "missing_target_summary_case_count": 0,
            "missing_campaign_coverage_case_count": 0,
            "incomplete_guest_case_count": 0,
            "campaign_incomplete_case_count": 0,
            "target": target,
            "trace_incomplete_cases": 0,
            "target_wall_clock_pending": 0,
        },
    }
    summary_path = output / "coverage" / "summary.json"
    atomic_write_json(summary_path, summary)
    return {
        "schema_version": "rq1-framework-coverage-v1",
        "status": status,
        "artifact_complete": complete,
        "summary": str(summary_path.relative_to(output)),
        "case_count": total_cases,
        "coverage_summary_case_count": len(entries),
        "completed_case_count": observed_cases,
        "right_censored_case_count": 0,
        "missing_case_count": max(0, total_cases - observed_cases),
        "missing_target_summary_case_count": 0,
        "target_wall_clock_pending": 0,
        "reason": reason,
    }


__all__ = [
    "DotnetCoverageServer",
    "finalize_framework_coverage", "finalize_framework_source_coverage_batch",
    "merge_framework_raw_coverage_batches",
    "merge_framework_coverage_summaries",
    "prewarm_coverage_target", "run_backend_with_coverage", "wait_for_dotnet_coverage",
]

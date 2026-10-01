"""Target resident-session seam。

正式 Target 优先使用已配置的 native JSONL service，或本进程内的嵌入式
worker。one-shot callback 只供显式选择的 legacy inline 兼容路径使用；该模式
始终标记为 ``adapter-callback/per-job``，不会伪装成常驻 session。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
from typing import Protocol


class ResidentSessionRequiredError(RuntimeError):
    """Target 没有常驻执行面，不能退回 per-job。"""


class TargetSession(Protocol):
    mode: str
    scope: str
    last_timing: Mapping[str, object]
    last_coverage: Mapping[str, object]
    last_session_pid: int | None
    last_session_identity: Mapping[str, object] | None

    def start(self) -> None: ...

    def submit(
        self, artifact: object, *, timeout_seconds: float | None = None,
        job_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> object: ...

    def close(self) -> None: ...


def _artifact_payload(artifact: object) -> dict[str, object]:
    value = artifact if isinstance(artifact, Mapping) else None
    get = value.get if value is not None else lambda name, default=None: getattr(
        artifact, name, default,
    )
    params = get("run_params", {})
    params = dict(params) if isinstance(params, Mapping) else {}
    return {
        "source_path": str(get("source_path", "")),
        "executable_path": str(get("executable_path", "")),
        "bare_executable_path": (
            str(get("bare_executable_path"))
            if get("bare_executable_path") is not None else None
        ),
        "linux_executable_path": (
            str(get("linux_executable_path"))
            if get("linux_executable_path") is not None else None
        ),
        "source_sha256": get("source_sha256"),
        "executable_sha256": get("executable_sha256"),
        "run_params": params,
    }


class CallbackTargetSession:
    """Explicit per-job compatibility adapter; never labeled as resident."""

    mode = "adapter-callback"
    scope = "per-job"
    last_timing: Mapping[str, object] = {}
    last_coverage: Mapping[str, object] = {}
    last_session_pid: int | None = None
    last_session_identity: Mapping[str, object] | None = None

    def __init__(self, runner: Callable[..., object]):
        self._runner = runner

    def start(self) -> None:
        return None

    def submit(
        self, artifact: object, *, timeout_seconds: float | None = None,
        job_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> object:
        return self._runner(artifact, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        return None


class JsonLineTargetSession:
    """Resident simulator process speaking one request/one JSON response lines."""

    mode = "resident-session-service"
    scope = "per-target-session"

    def __init__(
        self, command: Sequence[str], *, env: Mapping[str, str] | None = None,
        session_metadata: Mapping[str, object] | None = None,
    ):
        if not command:
            raise ValueError("resident target session command is empty")
        self.command = tuple(str(item) for item in command)
        self.env = dict(env or {})
        self.session_metadata = dict(session_metadata or {})
        self.process: subprocess.Popen[str] | None = None
        self.last_timing: Mapping[str, object] = {}
        self.last_coverage: Mapping[str, object] = {}
        self.session_pid: int | None = None
        self.last_session_pid: int | None = None
        self.last_session_identity: Mapping[str, object] | None = None
        self.library_lifetime = "process"

    def _read_response(self, timeout_seconds: float | None) -> Mapping[str, object]:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("resident target session is not connected")
        readable, _, _ = select.select(
            [process.stdout], [], [], timeout_seconds,
        )
        if not readable:
            raise TimeoutError("resident target session response timed out")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("resident target session exited without a response")
        try:
            response = json.loads(line)
        except (TypeError, ValueError) as error:
            raise RuntimeError("resident target session returned invalid JSON") from error
        if not isinstance(response, Mapping):
            raise RuntimeError("resident target session response is not an object")
        return response

    @classmethod
    def from_config(
        cls, value: object, *,
        session_metadata: Mapping[str, object] | None = None,
    ) -> "JsonLineTargetSession | None":
        if isinstance(value, (list, tuple)):
            return cls(value, session_metadata=session_metadata)
        if isinstance(value, str) and value.strip():
            return cls(shlex.split(value), session_metadata=session_metadata)
        return None

    def start(self) -> None:
        if self.process is not None:
            return
        environment = os.environ.copy()
        environment.update(self.env)
        self.process = subprocess.Popen(
            list(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.session_pid = self.process.pid
        self.last_session_pid = self.session_pid
        try:
            if self.process.stdin is None:
                raise RuntimeError("resident target session stdin is unavailable")
            self._send({
                "op": "hello",
                "protocol": "rq1-target-session-v1",
                "metadata": self.session_metadata,
            })
            response = self._read_response(5.0)
            if response.get("status") not in {"ready", "ok"} \
                    and response.get("ready") is not True:
                raise RuntimeError("resident target session handshake was not accepted")
        except Exception:
            self.close()
            raise

    def submit(
        self, artifact: object, *, timeout_seconds: float | None = None,
        job_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> object:
        if self.process is None:
            self.start()
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("resident target session is not connected")
        self._send({
            "op": "run",
            "job_id": job_id,
            "reset": True,
            "load": True,
            "collect": True,
            "artifact": _artifact_payload(artifact),
            "metadata": dict(metadata or {}),
        })
        wait_seconds = None if timeout_seconds is None else max(0.0, float(timeout_seconds))
        try:
            response = self._read_response(wait_seconds)
        except Exception:
            # A timed-out or malformed response makes the request/response
            # stream unsynchronizable. Close it so the next job can create a
            # fresh generation instead of consuming a stale reply.
            self.close()
            raise
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("error") or "resident target session error"))
        timing = response.get("timing")
        self.last_timing = dict(timing) if isinstance(timing, Mapping) else {}
        coverage = response.get("coverage")
        self.last_coverage = dict(coverage) if isinstance(coverage, Mapping) else {}
        for name in ("session_pid", "backend_session_scope", "coverage"):
            if name in response:
                self.last_timing[name] = response[name]
        identity = response.get("session_identity") or response.get("native_identity")
        self.last_session_identity = (
            dict(identity) if isinstance(identity, Mapping) else None
        )
        self.last_timing.setdefault("session_pid", self.session_pid)
        self.last_timing.setdefault("library_lifetime", self.library_lifetime)
        return response.get("result", response)

    def _send(self, payload: Mapping[str, object]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("resident target session is not connected")
        try:
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            process.stdin.flush()
        except (OSError, ValueError):
            # BrokenPipe/flush errors invalidate the request stream.  Clear it
            # now so the next submit starts a new resident session.
            self.close()
            raise

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"op":"close"}\n')
                    process.stdin.flush()
                except (OSError, ValueError):
                    pass
            # Keep the worker alive long enough to flush a batch-scoped
            # coverage collector before terminating the resident session.
            process.wait(timeout=90)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            self.last_session_pid = process.pid
            self.session_pid = None


def build_target_session(
    config: Mapping[str, object],
    fallback_runner: Callable[..., object],
    *, allow_callback: bool = False,
) -> TargetSession:
    """Create a resident session; callback fallback is explicit only."""
    command = config.get("session_command")
    metadata = {
        name: config[name]
        for name in ("backend", "identity_section", "target_recipe", "coverage_config")
        if name in config
    }
    session = JsonLineTargetSession.from_config(
        command, session_metadata=metadata,
    )
    if session is not None:
        return session
    backend = str(config.get("backend") or "").strip().lower()
    if backend in {"unicorn", "unicorn-riscv64"}:
        worker = Path(__file__).with_name("target_session_worker.py")
        return JsonLineTargetSession(
            (sys.executable, "-u", str(worker), "--backend", "unicorn-riscv64"),
            session_metadata=metadata,
        )
    if allow_callback:
        return CallbackTargetSession(fallback_runner)
    raise ResidentSessionRequiredError(
        f"Target backend {backend or '<unknown>'} requires session_command "
        "for a resident native service; per-job callback is disabled"
    )


__all__ = [
    "CallbackTargetSession", "JsonLineTargetSession", "TargetSession",
    "ResidentSessionRequiredError", "build_target_session",
]

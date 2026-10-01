import json
import os
import posixpath
import queue
import shlex
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from framework.adapters.runner import _observation_frame
from framework.direct_case import NATIVE_OBSERVER_TEMP_GPRS, OBSERVATION_HEADER_SIZE
from framework.execution_environment import resolve_native_reference_identity
from framework.native_reference_gate import NativeReferenceLock
from framework.paths import FRAMEWORK_RUNS_ROOT, LOCAL_ROOT, REPOSITORY_ROOT
from framework._util import pc_int, sha256_file
from framework.reference_fallback import K1_QEMU_FALLBACK_FAILURE_CLASSES


SERVER_SSH_CONFIG = Path("/path/to/ssh/config")
SSH_CONFIG = os.environ.get("RQ1_NATIVE_SSH_CONFIG") or (
    str(SERVER_SSH_CONFIG) if os.name != "nt" and SERVER_SSH_CONFIG.is_file() else None
)
SSH_ALIAS = os.environ.get("RQ1_NATIVE_SSH_ALIAS", "k1-board")
REMOTE_HOST = os.environ.get("RQ1_NATIVE_SSH_HOST", "133.133.137.252")
REMOTE_USER = os.environ.get("RQ1_NATIVE_SSH_USER", "user")
REMOTE_STORAGE_ROOT = "/mnt/build/wangyang"
REMOTE_ROOT = os.environ.get(
    "RQ1_NATIVE_REMOTE_ROOT",
    f"{REMOTE_STORAGE_ROOT}/rq1-runs" if SSH_CONFIG else "/home/user/riscv-experiments",
)
REMOTE_ROOT_IS_EPHEMERAL = (
    bool(os.environ.get("RQ1_NATIVE_REMOTE_ROOT"))
    and posixpath.normpath(REMOTE_ROOT) != f"{REMOTE_STORAGE_ROOT}/rq1-runs"
)
REMOTE_PREFIX = f"{REMOTE_ROOT}/direct-rv64-run-"
REMOTE_MIN_FREE_KIB = int(os.environ.get("RQ1_NATIVE_MIN_FREE_KIB", "1048576"))
REMOTE_SYSTEM_MIN_FREE_KIB = int(os.environ.get("RQ1_NATIVE_SYSTEM_MIN_FREE_KIB", "3145728"))
DEFAULT_TIMEOUT_SEC = int(os.environ.get("RQ1_NATIVE_GUEST_TIMEOUT", "180"))
# 远端流的每个阶段分别限时；父进程汇总各阶段预算。
REMOTE_EXEC_TIMEOUT_SEC = int(os.environ.get(
    "RQ1_NATIVE_REMOTE_TIMEOUT", str(DEFAULT_TIMEOUT_SEC * 2 + 30),
))
_ELF_MAGIC = b"\x7fELF"
_EM_RISCV = 243
_REMOTE_TRANSPORT_MARKERS = (
    "connection timed out", "operation timed out", "no route to host",
    "network is unreachable", "connection refused", "connection reset by peer",
    "connection closed by", "could not resolve hostname",
    "name or service not known", "temporary failure in name resolution",
    "host key verification failed", "remote host identification has changed",
    "permission denied (", "kex_exchange_identification",
    "ssh_exchange_identification", "broken pipe", "lost connection",
    # K1 storage preflight failures make the reference transport unusable for
    # this campaign and must trigger the declared QEMU fallback.
    "k1 storage root is missing or not writable",
    "k1 storage free space below threshold",
    "k1 system free space below threshold",
    "k1 data free-space probe failed",
    "k1 system free-space probe failed",
)


class _RemoteTransportError(RuntimeError):
    """SSH could not complete a K1 reference request."""


class _RemoteExecutionTimeout(RuntimeError):
    """K1 远端执行阶段超过响应预算。"""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(phase)


def _remote_transport_failure_class(detail: bytes | str) -> str | None:
    text = detail.decode("utf-8", errors="replace") if isinstance(detail, bytes) else str(detail)
    lowered = text.lower()
    return (
        "k1-transport-unavailable"
        if any(marker in lowered for marker in _REMOTE_TRANSPORT_MARKERS)
        else None
    )


def _rv64_elf_error(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError as exc:
        return f"read-error:{exc}"
    if len(header) < 20 or header[:4] != _ELF_MAGIC:
        return "invalid-elf-header"
    if header[4] != 2:
        return "elf64-required"
    if header[5] != 1:
        return "little-endian-required"
    if int.from_bytes(header[18:20], "little") != _EM_RISCV:
        return "riscv-required"
    return None


def _extract_results_archive(archive_path: Path, destination: Path) -> None:
    """Extract only regular files/directories inside the temporary result root."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            normalized_name = member.name.replace("\\", "/")
            member_path = (destination / normalized_name).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(f"native result archive path escapes root: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"native result archive contains unsafe member: {member.name}")
            archive.extract(member, destination)


def _ssh_options(known_hosts: Path, *, multiplex: bool = True) -> list[str]:
    if SSH_CONFIG:
        options = [
            "-F", SSH_CONFIG,
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=yes",
            f"-oUserKnownHostsFile={known_hosts}",
            "-oConnectTimeout=10",
        ]
    else:
        options = [
            "-oBatchMode=no",
            "-oPreferredAuthentications=password",
            "-oPubkeyAuthentication=no",
            "-oKbdInteractiveAuthentication=no",
            "-oStrictHostKeyChecking=yes",
            f"-oUserKnownHostsFile={known_hosts}",
            "-oNumberOfPasswordPrompts=1",
            "-oConnectTimeout=10",
        ]
    if os.name != "nt" and multiplex:
        options.extend((
            "-oControlMaster=auto",
            "-oControlPersist=30",
            "-oControlPath=/tmp/rq1-k1-%C",
        ))
    else:
        options.extend((
            "-oControlMaster=no",
            "-oControlPersist=no",
            "-oControlPath=none",
        ))
    options.extend((
        "-oServerAliveInterval=15",
        "-oServerAliveCountMax=3",
    ))
    return options


def _run_remote_process(
    args: list[str], env: dict[str, str], timeout: float | None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except OSError as error:
        raise _RemoteTransportError(
            f"{Path(args[0]).name}-unavailable:{type(error).__name__}"
        ) from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Descendants can retain the SSH pipes after the process group is
        # killed.  Reap the direct child within a short bound, then close the
        # streams so a timed out transport never turns into an unbounded
        # cleanup wait.
        try:
            process.communicate(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise _RemoteTransportError(f"{Path(args[0]).name}-timeout") from None
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def _remote_target() -> str:
    return SSH_ALIAS if SSH_CONFIG else f"{REMOTE_USER}@{REMOTE_HOST}"


def _remote_path_is_under(path: str, root: str) -> bool:
    path = posixpath.normpath(path)
    root = posixpath.normpath(root)
    return path == root or path.startswith(root + "/")


def _remote_storage_contract_error() -> str | None:
    if not SSH_CONFIG:
        return None
    if not _remote_path_is_under(REMOTE_ROOT, REMOTE_STORAGE_ROOT):
        return (
            "native RV64 remote root must stay under "
            f"{REMOTE_STORAGE_ROOT}: {REMOTE_ROOT}"
        )
    if not posixpath.isabs(REMOTE_ROOT):
        return f"native RV64 remote root must be absolute: {REMOTE_ROOT}"
    return None


def _remote_trace_helper_cache_path(helper: Path) -> tuple[str, str] | None:
    if not SSH_CONFIG or os.environ.get("RQ1_NATIVE_TRACE_HELPER_CACHE") != "1":
        return None
    try:
        digest = sha256_file(helper)
    except OSError:
        return None
    cache_root = posixpath.join(REMOTE_STORAGE_ROOT, ".rq1-native-helper-cache")
    return f"{cache_root}/rv64-ptrace-trace-{digest}", digest


def _remote_storage_preflight(
    qdir: str, *, helper_cache: tuple[str, str] | None = None,
) -> str:
    storage_root = shlex.quote(REMOTE_STORAGE_ROOT)
    run_root = shlex.quote(REMOTE_ROOT)
    command = (
        f"storage_root={storage_root}; run_root={run_root}; "
        "test -d \"$storage_root\" && test -w \"$storage_root\" || "
        "{ printf '%s\\n' 'K1 storage root is missing or not writable' >&2; exit 73; }; "
        "data_free=$(df -Pk \"$storage_root\" | awk 'END {print $4}'); "
        "system_free=$(df -Pk / | awk 'END {print $4}'); "
        "case \"$data_free\" in ''|*[!0-9]*) "
        "printf '%s\\n' 'K1 data free-space probe failed' >&2; exit 73;; esac; "
        "case \"$system_free\" in ''|*[!0-9]*) "
        "printf '%s\\n' 'K1 system free-space probe failed' >&2; exit 73;; esac; "
        f"printf 'K1_STORAGE_DATA_FREE_KIB=%s\\nK1_STORAGE_ROOT_FREE_KIB=%s\\n' "
        "\"$data_free\" \"$system_free\" >&2; "
        f"if [ \"$data_free\" -lt {REMOTE_MIN_FREE_KIB} ]; then "
        f"printf 'K1 storage free space below threshold: %s KiB < %s KiB\\n' "
        f"\"$data_free\" {REMOTE_MIN_FREE_KIB} >&2; exit 73; fi; "
        f"if [ \"$system_free\" -lt {REMOTE_SYSTEM_MIN_FREE_KIB} ]; then "
        f"printf 'K1 system free space below threshold: %s KiB < %s KiB\\n' "
        f"\"$system_free\" {REMOTE_SYSTEM_MIN_FREE_KIB} >&2; exit 73; fi; "
        "machine=$(uname -m); kernel=$(uname -r); "
        "isa=$(grep -m1 '^isa' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'); "
        "printf 'RV_NATIVE_MACHINE=%s\\nRV_NATIVE_KERNEL=%s\\nRV_NATIVE_ISA=%s\\n' "
        "\"$machine\" \"$kernel\" \"$isa\" >&2; "
        f"mkdir -p -m 700 {run_root} {qdir}"
    )
    if helper_cache is None:
        return command
    cache_path, digest = helper_cache
    cache_root = shlex.quote(posixpath.dirname(cache_path))
    cache_file = shlex.quote(cache_path)
    expected_digest = shlex.quote(digest)
    return command + (
        f" && {{ cache_status=miss; if mkdir -p -m 700 {cache_root} 2>/dev/null; then "
        f"if [ -f {cache_file} ] && [ ! -L {cache_file} ]; then "
        f"actual=$(sha256sum {cache_file} 2>/dev/null | awk '{{print $1}}'); "
        f"if [ \"$actual\" = {expected_digest} ]; then "
        "cache_status=hit; "
        f"else rm -f -- {cache_file} 2>/dev/null || true; fi; "
        f"elif [ -e {cache_file} ] || [ -L {cache_file} ]; then "
        f"rm -f -- {cache_file} 2>/dev/null || true; fi; fi; "
        "printf 'K1_TRACE_HELPER_CACHE=%s\\n' \"$cache_status\" >&2; }"
    )


def _remote_trace_helper_cache_store(
    remote_helper: str, helper_cache: tuple[str, str],
) -> str:
    cache_path, digest = helper_cache
    cache_root = shlex.quote(posixpath.dirname(cache_path))
    cache_file = shlex.quote(cache_path)
    request_file = shlex.quote(remote_helper)
    expected = shlex.quote(digest)
    return (
        "if [ \"$cache_status\" = hit ]; then true; else "
        f"mkdir -p -m 700 {cache_root} 2>/dev/null && "
        f"{{ if [ ! -e {cache_file} ] && [ ! -L {cache_file} ]; then "
        f"ln -- {request_file} {cache_file} 2>/dev/null || true; fi; "
        f"test -f {cache_file} && test ! -L {cache_file} && "
        f"actual=$(sha256sum {cache_file} 2>/dev/null | awk '{{print $1}}') && "
        f"test \"$actual\" = {expected}; }}; fi"
    )


def _remote_exec(
    command: str,
    env: dict[str, str],
    known_hosts: Path,
    timeout: float | None = REMOTE_EXEC_TIMEOUT_SEC,
    *,
    multiplex: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return _run_remote_process(
        ["ssh", *_ssh_options(known_hosts, multiplex=multiplex), _remote_target(), command],
        env, timeout,
    )


def _cancel_remote_trace_group(
    remote_dir: str,
    process_group: int,
    env: dict[str, str],
    known_hosts: Path,
) -> bool:
    if (
        not remote_dir.startswith(REMOTE_PREFIX)
        or not _remote_path_is_under(remote_dir, REMOTE_ROOT)
        or process_group <= 1
    ):
        return False
    pid_file = shlex.quote(posixpath.join(remote_dir, "run-0", "trace.pid"))
    expected_cwd = shlex.quote(remote_dir)
    expected = shlex.quote(str(process_group))
    command = (
        f"pid_file={pid_file}; expected={expected}; "
        f"expected_cwd={expected_cwd}; "
        "if [ ! -f \"$pid_file\" ] || [ \"$(cat -- \"$pid_file\")\" != \"$expected\" ]; then exit 1; fi; "
        "if ! /bin/kill -0 -- -\"$expected\" 2>/dev/null; then exit 0; fi; "
        "stat_file=/proc/$expected/stat; cwd_file=/proc/$expected/cwd; "
        "[ -r \"$stat_file\" ] && [ -e \"$cwd_file\" ] || exit 1; "
        "group=$(awk '{sub(/^.*\\) /, \"\"); print $3}' \"$stat_file\" 2>/dev/null) || exit 1; "
        "session=$(awk '{sub(/^.*\\) /, \"\"); print $4}' \"$stat_file\" 2>/dev/null) || exit 1; "
        "cwd=$(readlink -- \"$cwd_file\" 2>/dev/null) || exit 1; "
        "[ \"$group\" = \"$expected\" ] && [ \"$session\" = \"$expected\" ] && "
        "[ \"$cwd\" = \"$expected_cwd\" ] || exit 1; "
        "/bin/kill -TERM -- -\"$expected\" 2>/dev/null || true; "
        "tries=0; while /bin/kill -0 -- -\"$expected\" 2>/dev/null && [ \"$tries\" -lt 20 ]; do "
        "sleep 0.1; tries=$((tries + 1)); done; "
        "if /bin/kill -0 -- -\"$expected\" 2>/dev/null; then "
        "/bin/kill -KILL -- -\"$expected\" 2>/dev/null || true; sleep 0.1; fi; "
        "if /bin/kill -0 -- -\"$expected\" 2>/dev/null; then exit 75; fi; "
        "exit 0"
    )
    try:
        result = _remote_exec(
            command, env, known_hosts, timeout=5, multiplex=False,
        )
    except _RemoteTransportError:
        return False
    return result.returncode == 0


def cancel_active_remote_trace(
    env: dict[str, str], *, marker_wait_seconds: float = 0.0,
) -> tuple[int, bool] | None:
    remote_dir_marker = env.get("RQ1_NATIVE_REMOTE_DIR_MARKER")
    trace_pid_marker = env.get("RQ1_NATIVE_REMOTE_TRACE_PID_MARKER")
    if not remote_dir_marker or not trace_pid_marker:
        return None
    try:
        remote_dir = Path(remote_dir_marker).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    process_group = None
    attempts = max(1, int(marker_wait_seconds * 20) + 1)
    for attempt in range(attempts):
        try:
            value = Path(trace_pid_marker).read_text(encoding="ascii").strip()
            if value.isdecimal() and int(value) > 1:
                process_group = int(value)
                break
        except OSError:
            pass
        if attempt + 1 < attempts:
            time.sleep(0.05)
    if process_group is None:
        return None
    try:
        success = _cancel_remote_trace_group(
            remote_dir, process_group, env,
            Path(env["RQ1_NATIVE_KNOWN_HOSTS"]),
        )
    except Exception:
        success = False
    return process_group, success


def _kill_remote_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _remote_stream_process(
    command: str,
    input_archive: Path,
    result_archive: Path,
    env: dict[str, str],
    known_hosts: Path,
    timeout: float = REMOTE_EXEC_TIMEOUT_SEC,
    on_remote_dir_created: Callable[[], None] | None = None,
    on_remote_trace_pid: Callable[[int], None] | None = None,
    on_remote_trace_finished: Callable[[int], None] | None = None,
    cancel_remote_trace: Callable[[int], bool] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Stream one K1 request; the guest startup is bounded, full trace is not."""
    args = ["ssh", *_ssh_options(known_hosts), _remote_target(), command]
    input_size = input_archive.stat().st_size
    input_stream = input_archive.open("rb")
    try:
        output_stream = result_archive.open("wb")
    except OSError:
        input_stream.close()
        raise
    try:
        process = subprocess.Popen(
            args, env=env, stdin=input_stream, stdout=output_stream,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        input_stream.close()
        output_stream.close()
        raise _RemoteTransportError(
            f"ssh-unavailable:{type(error).__name__}"
        ) from error

    stderr_lines: list[bytes] = []
    stderr_events: queue.Queue[bytes | None] = queue.Queue()

    def collect_stderr() -> None:
        try:
            if process.stderr is not None:
                for line in iter(process.stderr.readline, b""):
                    stderr_events.put(line)
        finally:
            stderr_events.put(None)

    reader = threading.Thread(target=collect_stderr, name="k1-ssh-stderr", daemon=True)
    reader.start()
    expected_bytes: int | None = None
    result_sent = False
    stderr_closed = False
    stage = "input"
    remote_phase = "remote-startup"
    remote_trace_pid: int | None = None
    last_result_size = 0

    def cancel_live_trace() -> bytes:
        nonlocal remote_trace_pid
        if remote_trace_pid is None or cancel_remote_trace is None:
            return b""
        process_group = remote_trace_pid
        try:
            cancelled = cancel_remote_trace(process_group)
        except Exception:
            cancelled = False
        if cancelled:
            if on_remote_trace_finished is not None:
                on_remote_trace_finished(process_group)
            remote_trace_pid = None
        return (
            f"NATIVE_RV64_TRACE_CANCEL_{'SUCCEEDED' if cancelled else 'GAP'}="
            f"{process_group}\n"
        ).encode()

    def emit_cancel_note(note: bytes) -> None:
        if not note:
            return
        stream = getattr(sys.stderr, "buffer", None)
        if stream is not None:
            stream.write(note)
        else:
            sys.stderr.write(note.decode("utf-8", errors="replace"))

    input_timeout = min(timeout, max(30.0, input_size / 65536.0))
    deadline: float | None = time.monotonic() + input_timeout
    try:
        while True:
            if stage == "result-transfer":
                received_bytes = result_archive.stat().st_size
                if received_bytes > last_result_size:
                    last_result_size = received_bytes
                    deadline = time.monotonic() + timeout
            if expected_bytes is not None and result_sent:
                received_bytes = result_archive.stat().st_size
                if received_bytes == expected_bytes:
                    break
                if received_bytes > expected_bytes:
                    raise ValueError(
                        "native RV64 result stream exceeded its declared byte count"
                    )
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                if stage in {"execution", "result-transfer"}:
                    raise _RemoteExecutionTimeout(remote_phase)
                raise _RemoteTransportError("ssh-timeout")
            try:
                line = stderr_events.get(
                    timeout=0.1 if remaining is None else min(0.1, remaining),
                )
            except queue.Empty:
                if process.poll() is not None and stderr_closed:
                    break
                continue
            if line is None:
                stderr_closed = True
                if process.poll() is not None:
                    break
                continue
            text = line.decode("utf-8", errors="replace").strip()
            if "K1_NATIVE_PACKAGE_PROGRESS" in text:
                if remote_phase != "result-packaging":
                    raise ValueError("native RV64 package progress arrived outside packaging")
                deadline = time.monotonic() + timeout
                continue
            stderr_lines.append(line)
            if text == "K1_NATIVE_INPUT_READY":
                if stage != "input":
                    raise ValueError("native RV64 result stream has an unexpected input marker")
                stage = "execution"
                deadline = time.monotonic() + timeout
            elif text == "K1_NATIVE_DIRECTORY_CREATED":
                if stage != "input":
                    raise ValueError("native RV64 result stream has an unexpected directory marker")
                if on_remote_dir_created is not None:
                    on_remote_dir_created()
            elif text.startswith("K1_NATIVE_TRACE_PID="):
                value = text.partition("=")[2]
                if stage != "execution" or not value.isdecimal() or int(value) <= 1:
                    raise ValueError("native RV64 trace process-group marker is invalid")
                remote_trace_pid = int(value)
                if on_remote_trace_pid is not None:
                    on_remote_trace_pid(remote_trace_pid)
            elif text.startswith("K1_NATIVE_PACKAGE_PID="):
                value = text.partition("=")[2]
                if (
                    remote_phase != "result-packaging"
                    or not value.isdecimal() or int(value) <= 1
                ):
                    raise ValueError("native RV64 package process-group marker is invalid")
                remote_trace_pid = int(value)
                if on_remote_trace_pid is not None:
                    on_remote_trace_pid(remote_trace_pid)
            elif text.startswith("K1_NATIVE_TRANSFER_PID="):
                value = text.partition("=")[2]
                if (
                    stage != "result-transfer"
                    or not value.isdecimal() or int(value) <= 1
                ):
                    raise ValueError("native RV64 transfer process-group marker is invalid")
                remote_trace_pid = int(value)
                if on_remote_trace_pid is not None:
                    on_remote_trace_pid(remote_trace_pid)
            elif text == "K1_NATIVE_PRIMARY_STARTED":
                remote_phase = "primary-guest"
            elif text == "K1_NATIVE_PRIMARY_FINISHED":
                remote_phase = "trace-preparation"
            elif text == "K1_NATIVE_TRACE_STARTED":
                remote_phase = "ptrace-trace"
                # The trace is part of the task's full evidence. Only an
                # enclosing campaign deadline or explicit stop may cancel it.
                deadline = None
            elif text == "K1_NATIVE_TRACE_FINISHED":
                remote_phase = "result-packaging"
                deadline = time.monotonic() + timeout
                if remote_trace_pid is not None and on_remote_trace_finished is not None:
                    on_remote_trace_finished(remote_trace_pid)
                remote_trace_pid = None
            elif text == "K1_NATIVE_PACKAGE_STARTED":
                if stage != "execution":
                    raise ValueError("native RV64 package marker arrived in the wrong stage")
                remote_phase = "result-packaging"
                deadline = time.monotonic() + timeout
            elif text == "K1_NATIVE_PACKAGE_FINISHED":
                if remote_phase != "result-packaging":
                    raise ValueError("native RV64 package completion arrived in the wrong stage")
                deadline = time.monotonic() + timeout
                if remote_trace_pid is not None and on_remote_trace_finished is not None:
                    on_remote_trace_finished(remote_trace_pid)
                remote_trace_pid = None
            elif text.startswith("K1_NATIVE_RESULT_BYTES="):
                value = text.partition("=")[2]
                if stage != "execution" or expected_bytes is not None or not value.isdecimal():
                    raise ValueError("native RV64 result stream has an invalid byte count")
                expected_bytes = int(value)
                stage = "result-transfer"
                remote_phase = "result-transfer"
                deadline = time.monotonic() + timeout
            elif text == "K1_NATIVE_RESULT_SENT":
                if stage != "result-transfer":
                    raise ValueError("native RV64 result stream has an unexpected completion marker")
                if remote_trace_pid is not None and on_remote_trace_finished is not None:
                    on_remote_trace_finished(remote_trace_pid)
                remote_trace_pid = None
                result_sent = True

        return_code = process.wait(
            timeout=None if deadline is None else max(0.0, deadline - time.monotonic()),
        )
        reader.join(timeout=1)
        result_size = result_archive.stat().st_size
        cancel_note = (
            cancel_live_trace()
            if return_code != 0 else b""
        )
        if return_code == 0 and (
            expected_bytes is None or not result_sent or result_size != expected_bytes
        ):
            raise ValueError("native RV64 result stream ended before archive verification")
        return subprocess.CompletedProcess(
            args, return_code, None, b"".join(stderr_lines) + cancel_note,
        )
    except _RemoteExecutionTimeout:
        emit_cancel_note(cancel_live_trace())
        _kill_remote_process(process)
        raise
    except subprocess.TimeoutExpired:
        _kill_remote_process(process)
        if stage in {"execution", "result-transfer"}:
            raise _RemoteExecutionTimeout(remote_phase) from None
        raise _RemoteTransportError("ssh-timeout") from None
    except BaseException:
        emit_cancel_note(cancel_live_trace())
        _kill_remote_process(process)
        raise
    finally:
        reader.join(timeout=1)
        if process.stderr is not None:
            process.stderr.close()
        input_stream.close()
        output_stream.close()


def _fail(
    message: str, detail: bytes = b"", code: int = 127, *,
    failure_class: str | None = None,
) -> int:
    payload: object = (
        {"reason": message, "failure_class": failure_class}
        if failure_class in K1_QEMU_FALLBACK_FAILURE_CLASSES else message
    )
    sys.stderr.write("RV_RUNNER_CONTRACT_GAP=" + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stderr.write(message + "\n")
    if detail:
        sys.stderr.buffer.write(detail)
        if not detail.endswith(b"\n"):
            sys.stderr.write("\n")
    return code


def _remote_connection() -> tuple[dict[str, str], Path, str] | int:
    if SSH_CONFIG:
        if contract_error := _remote_storage_contract_error():
            return _fail(contract_error)
        config = Path(SSH_CONFIG)
        known_hosts = Path(os.environ.get("RQ1_NATIVE_KNOWN_HOSTS") or config.with_name("known_hosts"))
        if not config.is_file():
            return _fail(
                f"native RV64 SSH config is missing: {config}",
                failure_class="k1-transport-unavailable",
            )
        if not known_hosts.is_file():
            return _fail(
                f"native RV64 known_hosts file is missing: {known_hosts}",
                failure_class="k1-transport-unavailable",
            )
        remote_dir = f"{REMOTE_PREFIX}{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        return os.environ.copy(), known_hosts, remote_dir
    askpass = Path(os.environ.get("RISCV_SSH_ASKPASS") or LOCAL_ROOT / "secrets" / "riscv-askpass.cmd")
    known_hosts = Path(os.environ.get("RISCV_KNOWN_HOSTS") or REPOSITORY_ROOT / ".codex" / "skills" / "use-native-riscv-environment" / "scripts" / "riscv-known-hosts")
    if not askpass.is_file():
        return _fail(
            f"native RV64 askpass is missing outside the repository: {askpass}",
            failure_class="k1-transport-unavailable",
        )
    if os.name != "nt" and not os.access(askpass, os.X_OK):
        return _fail(
            f"native RV64 askpass is not executable on this host: {askpass}",
            failure_class="k1-transport-unavailable",
        )
    if not known_hosts.is_file():
        return _fail(
            f"native RV64 known_hosts file is missing: {known_hosts}",
            failure_class="k1-transport-unavailable",
        )
    env = os.environ.copy()
    env["SSH_ASKPASS"] = str(askpass)
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env.setdefault("DISPLAY", "codex")
    return env, known_hosts, f"{REMOTE_PREFIX}{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def _parse_trace_pcs(text: str) -> tuple[tuple[int, ...], str | None]:
    pcs: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("RV_TRACE_PC="):
            continue
        raw = line.removeprefix("RV_TRACE_PC=").strip()
        try:
            pc = pc_int(raw)
        except ValueError:
            return tuple(pcs), f"invalid-pc-at-line-{line_number}:{raw}"
        if pc & 1:
            return tuple(pcs), f"unaligned-pc-at-line-{line_number}:{raw}"
        pcs.append(pc)
    return tuple(pcs), None


def _parse_terminal_signal_contract(text: str) -> tuple[int, int] | tuple[()] | None:
    signal_number = trap_pc = None
    seen: set[str] = set()
    has_signal = False
    for line in text.splitlines():
        if line.startswith("RV_TRACE_TERMINAL_SIGNAL="):
            has_signal = True
            if "signal" in seen:
                return ()
            seen.add("signal")
            try:
                signal_number = int(line.removeprefix("RV_TRACE_TERMINAL_SIGNAL="), 10)
            except ValueError:
                return ()
        elif line.startswith("RV_TRACE_FINAL_TRAP_PC="):
            if "pc" in seen:
                return ()
            seen.add("pc")
            try:
                trap_pc = pc_int(line.removeprefix("RV_TRACE_FINAL_TRAP_PC=").strip())
            except ValueError:
                return ()
    if not has_signal:
        return None
    if signal_number not in {4, 6, 7, 8, 11, 31} or trap_pc is None or trap_pc & 1:
        return ()
    return signal_number, trap_pc


def _parse_gdb_trace_contract(
    text: str, trace_status: str,
) -> tuple[tuple[int, ...], str | None]:
    pcs, parse_error = _parse_trace_pcs(text)
    if parse_error is not None:
        return pcs, parse_error
    if not trace_status.startswith("gdb-single-step:"):
        return pcs, "invalid-gdb-trace-status"
    try:
        exit_code = int(trace_status.removeprefix("gdb-single-step:"), 10)
    except ValueError:
        return pcs, "invalid-gdb-trace-exit-code"
    if exit_code != 0:
        return pcs, f"gdb-trace-exit-code-{exit_code}"
    stop_pc = None
    exit_pc = None
    steps = None
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith("RV_TRACE_STOP_PC="):
            if "stop" in seen:
                return pcs, f"duplicate-stop-pc-at-line-{line_number}"
            seen.add("stop")
            try:
                stop_pc = pc_int(line.removeprefix("RV_TRACE_STOP_PC=").strip())
            except ValueError:
                return pcs, f"invalid-stop-pc-at-line-{line_number}"
        elif line.startswith("RV_TRACE_EXIT_PC="):
            if "exit" in seen:
                return pcs, f"duplicate-exit-pc-at-line-{line_number}"
            seen.add("exit")
            try:
                exit_pc = pc_int(line.removeprefix("RV_TRACE_EXIT_PC=").strip())
            except ValueError:
                return pcs, f"invalid-exit-pc-at-line-{line_number}"
        elif line.startswith("RV_TRACE_STEPS="):
            if "steps" in seen:
                return pcs, f"duplicate-step-count-at-line-{line_number}"
            seen.add("steps")
            try:
                steps = int(line.removeprefix("RV_TRACE_STEPS=").strip(), 10)
            except ValueError:
                return pcs, f"invalid-step-count-at-line-{line_number}"
    if (
        stop_pc is not None and exit_pc is not None and stop_pc == exit_pc and not stop_pc & 1
        and steps is not None and steps >= 0 and steps == len(pcs)
    ):
        return pcs, None
    return pcs, "gdb-trace-contract-incomplete"


def _parse_trace_checkpoints(text: str) -> tuple[tuple[dict, ...], str | None]:
    checkpoints: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("RV_TRACE_CHECKPOINT="):
            continue
        try:
            data = json.loads(line.removeprefix("RV_TRACE_CHECKPOINT="))
        except json.JSONDecodeError:
            return tuple(checkpoints), f"invalid-checkpoint-json-at-line-{line_number}"
        if not isinstance(data, dict):
            return tuple(checkpoints), f"invalid-checkpoint-at-line-{line_number}"
        try:
            pc = pc_int(data.get("pc"))
            after_pc = pc_int(data.get("after_pc"))
        except (TypeError, ValueError):
            return tuple(checkpoints), f"invalid-checkpoint-pc-at-line-{line_number}"
        if pc & 1 or after_pc & 1:
            return tuple(checkpoints), f"unaligned-checkpoint-pc-at-line-{line_number}"
        for key in ("before_gpr", "after_gpr"):
            values = data.get(key)
            if not isinstance(values, list) or len(values) != 32:
                return tuple(checkpoints), f"invalid-{key}-at-line-{line_number}"
            try:
                if pc_int(values[0]) != 0:
                    return tuple(checkpoints), f"nonzero-x0-in-{key}-at-line-{line_number}"
                tuple(pc_int(item) for item in values)
            except (TypeError, ValueError):
                return tuple(checkpoints), f"invalid-{key}-at-line-{line_number}"
        for key in ("before_fpr_rawbits", "after_fpr_rawbits"):
            values = data.get(key)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != 32:
                return tuple(checkpoints), f"invalid-{key}-at-line-{line_number}"
            try:
                tuple(pc_int(item) for item in values)
            except (TypeError, ValueError):
                return tuple(checkpoints), f"invalid-{key}-at-line-{line_number}"
        for key, mask in (
            ("before_fflags", 0x1F),
            ("after_fflags", 0x1F),
            ("before_frm", 0x7),
            ("after_frm", 0x7),
        ):
            value = data.get(key)
            if value is None:
                continue
            try:
                if pc_int(value) > mask:
                    return tuple(checkpoints), f"out-of-range-{key}-at-line-{line_number}"
            except (TypeError, ValueError):
                return tuple(checkpoints), f"invalid-{key}-at-line-{line_number}"
        checkpoints.append(data)
    return tuple(checkpoints), None


def _stable_trace_final_fp_state(checkpoints: tuple[dict, ...]) -> dict[str, object] | None:
    """Recover a path-insensitive final FP state from ptrace checkpoints.

    This is only used when the ptrace trace cannot be accepted as a native
    checkpoint trace (for example because the traced stdout diverges from the
    raw run) but every checkpoint still reports the same before/after FP state.
    In that narrow case we can safely recover final FPR/FCSR as observation
    state without promoting the trace to executed-PC or checkpoint evidence.
    """
    if not checkpoints:
        return None
    stable_fpr: tuple[int, ...] | None = None
    stable_fflags: int | None = None
    stable_frm: int | None = None
    for checkpoint in checkpoints:
        before_fpr = checkpoint.get("before_fpr_rawbits")
        after_fpr = checkpoint.get("after_fpr_rawbits")
        before_fflags = checkpoint.get("before_fflags")
        after_fflags = checkpoint.get("after_fflags")
        before_frm = checkpoint.get("before_frm")
        after_frm = checkpoint.get("after_frm")
        if (
            before_fpr is None
            or after_fpr is None
            or before_fflags is None
            or after_fflags is None
            or before_frm is None
            or after_frm is None
        ):
            return None
        current_before_fpr = tuple(pc_int(value) for value in before_fpr)
        current_after_fpr = tuple(pc_int(value) for value in after_fpr)
        current_before_fflags = pc_int(before_fflags)
        current_after_fflags = pc_int(after_fflags)
        current_before_frm = pc_int(before_frm)
        current_after_frm = pc_int(after_frm)
        if (
            current_before_fflags > 0x1F
            or current_after_fflags > 0x1F
            or current_before_frm > 0x7
            or current_after_frm > 0x7
        ):
            return None
        if (
            current_before_fpr != current_after_fpr
            or current_before_fflags != current_after_fflags
            or current_before_frm != current_after_frm
        ):
            return None
        if stable_fpr is None:
            stable_fpr = current_after_fpr
            stable_fflags = current_after_fflags
            stable_frm = current_after_frm
            continue
        if (
            stable_fpr != current_after_fpr
            or stable_fflags != current_after_fflags
            or stable_frm != current_after_frm
        ):
            return None
    if stable_fpr is None or stable_fflags is None or stable_frm is None:
        return None
    return {
        "fpr_rawbits": list(stable_fpr),
        "fflags": stable_fflags,
        "frm": stable_frm,
    }


def _trace_checkpoint_contract_complete(
    text: str,
    pcs: tuple[int, ...],
    checkpoints: tuple[dict, ...],
) -> bool:
    status = None
    hits = None
    checkpoint_count = None
    seen: set[str] = set()
    for line in text.splitlines():
        if line.startswith("RV_TRACE_PC="):
            try:
                pc_int(line.removeprefix("RV_TRACE_PC=").strip())
            except ValueError:
                return False
        elif line.startswith("RV_TRACE_STATUS="):
            if "RV_TRACE_STATUS" in seen:
                return False
            seen.add("RV_TRACE_STATUS")
            status = line.removeprefix("RV_TRACE_STATUS=").strip()
        elif line.startswith("RV_TRACE_HITS="):
            if "RV_TRACE_HITS" in seen:
                return False
            seen.add("RV_TRACE_HITS")
            try:
                hits = int(line.removeprefix("RV_TRACE_HITS="), 0)
            except ValueError:
                return False
        elif line.startswith("RV_TRACE_CHECKPOINTS="):
            if "RV_TRACE_CHECKPOINTS" in seen:
                return False
            seen.add("RV_TRACE_CHECKPOINTS")
            try:
                checkpoint_count = int(line.removeprefix("RV_TRACE_CHECKPOINTS="), 0)
            except ValueError:
                return False
    if not pcs:
        return False
    return bool(
        status == "ok"
        and hits == len(pcs)
        and checkpoint_count == len(checkpoints)
        and len(pcs) == len(checkpoints)
        and all(
            pc_int(pc) == pc_int(checkpoint.get("pc"))
            for pc, checkpoint in zip(pcs, checkpoints)
        )
    )


def _trace_helper_source() -> Path:
    return REPOSITORY_ROOT / "framework" / "native_trace" / "ptrace_trace.c"


def _build_trace_helper() -> tuple[Path | None, str]:
    source = _trace_helper_source()
    if not source.is_file():
        return None, "missing-source"
    digest = sha256_file(source)
    output = FRAMEWORK_RUNS_ROOT / "native-trace" / f"rv64-ptrace-trace-{digest[:16]}"
    lock_stream = None
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_stream = output.with_name(output.name + ".lock").open("a+b")
        if os.name == "nt":
            import msvcrt

            lock_stream.seek(0, os.SEEK_END)
            if lock_stream.tell() == 0:
                lock_stream.write(b"\0")
                lock_stream.flush()
            lock_stream.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output, "cached"

        fd, temporary_name = tempfile.mkstemp(
            prefix=output.name + ".", dir=output.parent,
        )
        temporary = Path(temporary_name)
        command = (
            "riscv64-linux-gnu-gcc",
            "-O2",
            "-static",
            "-Wall",
            "-Wextra",
            "-Wno-unused-variable",
            "-o",
            str(temporary),
            str(source),
        )
        os.close(fd)
        proc = subprocess.run(
            list(command),
            cwd=str(REPOSITORY_ROOT),
            timeout=60,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not temporary.is_file():
            if output.is_file():
                return output, "cached"
            detail = (proc.stderr or proc.stdout).decode(
                "utf-8", errors="replace",
            ).strip().splitlines()
            return None, "build-failed:" + (
                detail[-1] if detail else str(proc.returncode)
            )
        os.replace(temporary, output)
        temporary = None
        return output, "built"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"build-error:{exc}"
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if lock_stream is not None:
            lock_stream.close()


def _elf_symbols(elf_path: Path) -> tuple[dict[str, int], str]:
    manifest_symbols, _ = _elf_symbols_from_build_manifest(elf_path)
    if manifest_symbols:
        return manifest_symbols, "ok"
    try:
        proc = subprocess.run(
            ["riscv64-linux-gnu-nm", "-n", str(elf_path)],
            cwd=str(elf_path.parent),
            timeout=10,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"nm-error:{exc}"
    if proc.returncode != 0:
        return {}, "nm-failed"
    symbols: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[-1]
        if name not in {"test_start", "exit_checkpoint", "_start", "test_done", "write_tohost"}:
            continue
        try:
            symbols[name] = int(parts[0], 16)
        except ValueError:
            continue
    start = symbols.get("test_start", symbols.get("_start"))
    stop = symbols.get("exit_checkpoint", symbols.get("test_done", symbols.get("write_tohost")))
    if start is not None and stop is not None:
        return {"test_start": start, "exit_checkpoint": stop}, "ok"
    if start is not None:
        return {"test_start": start}, "fallback-start"
    return symbols, "missing-symbols"


def _elf_symbols_from_build_manifest(elf_path: Path) -> tuple[dict[str, int], str]:
    manifest_path = elf_path.with_name("case.build.json")
    if not manifest_path.is_file():
        return {}, "missing-build-manifest"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"invalid-build-manifest:{exc}"
    if not isinstance(payload, dict):
        return {}, "invalid-build-manifest:payload-not-object"
    build_map = payload.get("map")
    if not isinstance(build_map, dict):
        return {}, "invalid-build-manifest:missing-map"
    runtime_symbols = build_map.get("runtime_symbol_addresses")
    if not isinstance(runtime_symbols, dict):
        return {}, "invalid-build-manifest:missing-runtime-symbol-addresses"
    symbols: dict[str, int] = {}
    for name in ("test_start", "exit_checkpoint"):
        value = runtime_symbols.get(name)
        if not isinstance(value, str) or not value:
            return {}, f"invalid-build-manifest:missing-{name}"
        try:
            symbols[name] = pc_int(value)
        except ValueError:
            return {}, f"invalid-build-manifest:bad-{name}"
    return symbols, "build-manifest"


def _fallback_pc_manifest(elf_path: Path, output_path: Path, start: int) -> tuple[bool, str]:
    data = elf_path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
        return False, "fallback-invalid-elf"
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize = struct.unpack_from("<H", data, 54)[0]
    phnum = struct.unpack_from("<H", data, 56)[0]
    for index in range(phnum):
        offset = phoff + index * phentsize
        p_type, flags, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, offset)
        if p_type != 1 or not flags & 1 or not p_vaddr <= start < p_vaddr + p_filesz:
            continue
        cursor = p_offset + start - p_vaddr
        limit = p_offset + p_filesz
        entries = []
        pc = start
        while cursor + 2 <= limit:
            halfword = data[cursor] | (data[cursor + 1] << 8)
            length = 2 if (halfword & 0x3) != 0x3 else 4
            if cursor + length > limit:
                break
            instruction = data[cursor:cursor + length]
            if instruction in (b"\x73\x00\x10\x00", b"\x02\x90"):
                if not entries:
                    return False, "fallback-no-terminal-range"
                sentinel = entries.pop()
                output_path.write_text(
                    "".join(f"0x{address:x} {size}\n" for address, size in entries)
                    + f"0x{sentinel[0]:x} {sentinel[1]} exit-sentinel\n",
                    encoding="utf-8",
                )
                return True, f"fallback-pc-manifest:{len(entries)}+exit-sentinel"
            entries.append((pc, length))
            cursor += length
            pc += length
    return False, "fallback-terminal-ebreak-missing"


def _write_pc_manifest(elf_path: Path, output_path: Path) -> tuple[bool, str]:
    symbols, status = _elf_symbols(elf_path)
    if status == "fallback-start":
        return _fallback_pc_manifest(elf_path, output_path, symbols["test_start"])
    if status != "ok":
        return False, status
    start = symbols["test_start"]
    stop = symbols["exit_checkpoint"]
    if stop <= start:
        return False, "invalid-symbol-range"
    code_bytes, status = _elf_vaddr_slice(elf_path, start, stop)
    if not code_bytes:
        return False, status
    entries = []
    cursor = 0
    pc = start
    while cursor < len(code_bytes):
        if cursor + 2 > len(code_bytes):
            return False, "truncated-instruction"
        halfword = code_bytes[cursor] | (code_bytes[cursor + 1] << 8)
        length = 2 if (halfword & 0x3) != 0x3 else 4
        if cursor + length > len(code_bytes):
            return False, "truncated-instruction"
        entries.append((pc, length))
        cursor += length
        pc += length
    prefix, sentinel_status = _elf_vaddr_slice(elf_path, stop, stop + 2)
    if len(prefix) != 2:
        return False, sentinel_status
    halfword = prefix[0] | (prefix[1] << 8)
    sentinel_length = 2 if (halfword & 0x3) != 0x3 else 4
    instruction, sentinel_status = _elf_vaddr_slice(elf_path, stop, stop + sentinel_length)
    if len(instruction) != sentinel_length:
        return False, sentinel_status
    output_path.write_text(
        "".join(
            (
                *(f"0x{pc:x} {length}\n" for pc, length in entries),
                f"0x{stop:x} {sentinel_length} exit-sentinel\n",
            )
        ),
        encoding="utf-8",
    )
    return True, f"pc-manifest:{len(entries)}+exit-sentinel"


def _elf_vaddr_slice(elf_path: Path, start: int, stop: int) -> tuple[bytes, str]:
    if stop <= start:
        return b"", "invalid-symbol-range"
    try:
        data = elf_path.read_bytes()
    except OSError as exc:
        return b"", f"elf-read-error:{exc}"
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return b"", "invalid-elf-header"
    if data[4] != 2 or data[5] != 1:
        return b"", "unsupported-elf-layout"
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    if e_shoff <= 0 or e_shentsize <= 0 or e_shnum <= 0:
        return b"", "missing-section-table"
    section_end = e_shoff + e_shentsize * e_shnum
    if section_end > len(data):
        return b"", "invalid-section-table-range"
    window_size = stop - start
    for index in range(e_shnum):
        offset = e_shoff + index * e_shentsize
        sh_addr = struct.unpack_from("<Q", data, offset + 16)[0]
        sh_offset = struct.unpack_from("<Q", data, offset + 24)[0]
        sh_size = struct.unpack_from("<Q", data, offset + 32)[0]
        if sh_size <= 0:
            continue
        if sh_addr > start or stop > sh_addr + sh_size:
            continue
        file_start = sh_offset + (start - sh_addr)
        file_stop = file_start + window_size
        if file_stop > len(data):
            return b"", "section-range-outside-file"
        return data[file_start:file_stop], "ok"
    return b"", "address-range-not-in-section"


def _safe_status(value: str) -> str:
    return "".join(character if character.isalnum() or character in ".:_-" else "_" for character in value)[:120]


def _gdb_trace_script() -> str:
    return "\n".join(
        (
            "set pagination off",
            "set confirm off",
            "set print thread-events off",
            "break test_start",
            "run",
            "delete breakpoints",
            "set $exit = &exit_checkpoint",
            "set $steps = 0",
            "while $pc != $exit",
            '  printf "RV_TRACE_PC=0x%lx\\n", $pc',
            "  stepi",
            "  set $steps = $steps + 1",
            "end",
            'printf "RV_TRACE_STOP_PC=0x%lx\\n", $pc',
            'printf "RV_TRACE_EXIT_PC=0x%lx\\n", $exit',
            'printf "RV_TRACE_STEPS=%ld\\n", $steps',
            "quit",
            "",
        )
    )


def _cleanup(remote_dir: str, env: dict[str, str], known_hosts: Path) -> bool:
    if not remote_dir.startswith(REMOTE_PREFIX) or not _remote_path_is_under(remote_dir, REMOTE_ROOT):
        return False
    cleanup = f"rm -rf -- {shlex.quote(remote_dir)}"
    if REMOTE_ROOT_IS_EPHEMERAL:
        cleanup += f"; rmdir -- {shlex.quote(REMOTE_ROOT)} 2>/dev/null || true"
    result = _remote_exec(cleanup, env, known_hosts)
    if result.returncode != 0:
        sys.stderr.write(
            f"NATIVE_RV64_CLEANUP_GAP=remote-exit-{result.returncode}\n"
        )
        if result.stderr:
            sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        return False
    marker = env.get("RQ1_NATIVE_REMOTE_DIR_MARKER")
    if marker:
        try:
            marker_path = Path(marker)
            if marker_path.read_text(encoding="utf-8").strip() == remote_dir:
                marker_path.unlink(missing_ok=True)
        except OSError:
            pass
    trace_marker = env.get("RQ1_NATIVE_REMOTE_TRACE_PID_MARKER")
    if trace_marker:
        try:
            Path(trace_marker).unlink(missing_ok=True)
        except OSError:
            pass
    return True


def _native_reference_identity_from_output(
    output: bytes,
    known_hosts: Path,
    stratum: str = "linux-user",
) -> tuple[dict | None, str | None]:
    fields: dict[str, str] = {}
    for line in output.decode("utf-8", errors="replace").splitlines():
        if line.startswith("RV_NATIVE_MACHINE="):
            fields["machine"] = line.removeprefix("RV_NATIVE_MACHINE=").strip()
        elif line.startswith("RV_NATIVE_KERNEL="):
            fields["kernel"] = line.removeprefix("RV_NATIVE_KERNEL=").strip()
        elif line.startswith("RV_NATIVE_ISA="):
            fields["isa"] = line.removeprefix("RV_NATIVE_ISA=").strip()
    if not fields.get("machine") or not fields.get("kernel") or not fields.get("isa"):
        return None, "remote-identity-incomplete"
    try:
        identity = resolve_native_reference_identity(
            fields["machine"],
            fields["kernel"],
            fields["isa"],
            known_hosts_path=known_hosts,
            runner_path=Path(__file__).resolve(),
            helper_path=_trace_helper_source(),
        )
    except Exception as exc:
        return None, str(exc)
    return {**identity.to_dict(), "stratum": stratum, "execution_model": stratum}, None


def _run_stderr_record(
    *,
    remote_dir: str,
    local_sha: str,
    index: int,
    exit_code: int,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    trace_status: str,
    trace_stdout_bytes: bytes,
    trace_stderr_text: str,
    native_identity: dict | None,
    primary_timed_out: bool = False,
) -> str:
    trace_stdout_text = trace_stdout_bytes.decode("utf-8", errors="replace")
    trace_fallback_fp_state = None
    observation_extra_state: dict[str, object] = {}
    trace_complete = False
    terminal_signal_state = None
    if trace_status.startswith("ptrace-manifest:"):
        trace_pcs, pc_parse_error = _parse_trace_pcs(trace_stderr_text)
        trace_checkpoints, checkpoint_parse_error = _parse_trace_checkpoints(trace_stderr_text)
        terminal_signal = _parse_terminal_signal_contract(trace_stderr_text)
        expected_trace_exit = 0 if terminal_signal is not None else exit_code
        expected_trace_status = f"ptrace-manifest:{expected_trace_exit}"
        complete = (
            trace_status == expected_trace_status
            and pc_parse_error is None
            and checkpoint_parse_error is None
            and _trace_checkpoint_contract_complete(
                trace_stderr_text, trace_pcs, trace_checkpoints,
            )
        )
        errors = [error for error in (pc_parse_error, checkpoint_parse_error) if error]
        if trace_status != expected_trace_status:
            errors.append(f"trace-process-status:{trace_status}")
        if terminal_signal is not None:
            if not complete or not terminal_signal:
                errors.append("terminal-signal-contract-failed")
            else:
                signal_number, trap_pc = terminal_signal
                if trap_pc != trace_pcs[-1]:
                    trace_pcs += (trap_pc,)
                trace_status = f"ptrace-signal:{signal_number}"
                terminal_signal_state = (signal_number, trap_pc)
        elif not complete:
            errors.append("checkpoint-contract-failed")
        elif trace_stdout_bytes != stdout_bytes:
            trace_stdout_matches, mismatch_summary = _compare_native_trace_stdout(
                stdout_bytes, trace_stdout_bytes,
            )
            if mismatch_summary is not None:
                observation_extra_state["native_trace_stdout_mismatch"] = mismatch_summary
                if trace_stdout_matches:
                    observation_extra_state["volatile_gpr_indices"] = (
                        mismatch_summary["gpr_diff_indices"]
                    )
            if not trace_stdout_matches:
                trace_fallback_fp_state = _stable_trace_final_fp_state(trace_checkpoints)
                errors.append("stdout-mismatch")
        trace_complete = complete and not errors
        if errors:
            trace_status = f"partial:{trace_status};{'|'.join(errors)}"
    elif trace_status.startswith("gdb-single-step:"):
        trace_pcs, parse_error = _parse_gdb_trace_contract(trace_stdout_text, trace_status)
        trace_checkpoints = ()
        trace_complete = parse_error is None
        if parse_error is not None:
            trace_status = f"partial:{trace_status};{parse_error}"
    else:
        trace_pcs = ()
        trace_checkpoints = ()

    lines = []
    if native_identity is not None:
        lines.append(
            "RV_NATIVE_REFERENCE_IDENTITY="
            + json.dumps(native_identity, sort_keys=True, separators=(",", ":"))
        )
    lines.append(
        f"NATIVE_RV64_REFERENCE remote_dir={remote_dir} execution_index={index} "
        f"elf_sha256={local_sha} remote_exit_code={exit_code} "
        f"primary_timeout={'true' if primary_timed_out else 'false'}"
    )
    if trace_pcs:
        lines.append(f"RV_EXECUTED_PCS={json.dumps([hex(pc) for pc in trace_pcs], separators=(',', ':'))}")
        lines.append(f"RV_INSTRUCTION_COUNT={len(trace_pcs)}")
    lines.append(
        f"NATIVE_RV64_TRACE_STATUS={'observed-runner-trace' if trace_complete else 'partial-runner-trace' if trace_pcs else 'unavailable'} "
        f"status={trace_status}"
    )
    if trace_checkpoints:
        lines.append(f"RV_NATIVE_WINDOW_CHECKPOINTS={json.dumps(trace_checkpoints, separators=(',', ':'))}")
    if trace_fallback_fp_state is not None:
        observation_extra_state.update(trace_fallback_fp_state)
    if terminal_signal_state is not None:
        observation_extra_state.update({
            "target_stop_observed": True,
            "target_stop_observer": "native-ptrace-terminal-signal",
            "observer_fields": ["outcome", "signal", "fault_pc"],
        })
    if observation_extra_state:
        observation_state = {"extra_state": observation_extra_state}
        if terminal_signal_state is not None:
            signal_number, trap_pc = terminal_signal_state
            observation_state.update({
                "signal": signal.Signals(signal_number).name,
                "signal_code": signal_number,
                "fault_pc": hex(trap_pc),
            })
        lines.append(
            "RV_OBSERVATION_STATE="
            + json.dumps(observation_state, separators=(",", ":"))
        )
    if not trace_complete:
        if trace_stdout_text:
            lines.append(f"NATIVE_RV64_TRACE_STDOUT={trace_stdout_text!r}")
        if trace_stderr_text:
            lines.append(f"NATIVE_RV64_TRACE_STDERR={trace_stderr_text!r}")
    program_stderr = stderr_bytes.decode("utf-8", errors="replace")
    if program_stderr:
        lines.append(program_stderr)
    return "\n".join(lines) + "\n"


def _stdout_observation_summary(stdout_bytes: bytes) -> dict[str, object] | None:
    frame = _observation_frame(stdout_bytes)
    if frame is None or len(frame) < OBSERVATION_HEADER_SIZE:
        return None
    checkpoint = struct.unpack_from("<Q", frame, 8)[0]
    if checkpoint == 0 or checkpoint & 1:
        return None
    memory_size = struct.unpack_from("<Q", frame, 16 + (32 * 8))[0]
    memory_start = OBSERVATION_HEADER_SIZE
    memory_end = memory_start + memory_size
    gpr = tuple(
        struct.unpack_from("<Q", frame, 16 + index * 8)[0]
        for index in range(32)
    )
    if len(frame) != memory_end or gpr[0] != 0:
        return None
    return {
        "checkpoint_pc": checkpoint,
        "gpr": gpr,
        "memory": frame[memory_start:memory_end],
    }


def _compare_native_trace_stdout(
    stdout_bytes: bytes, trace_stdout_bytes: bytes,
) -> tuple[bool, dict[str, object] | None]:
    observed_frame = _observation_frame(stdout_bytes)
    traced_frame = _observation_frame(trace_stdout_bytes)
    observed = _stdout_observation_summary(stdout_bytes)
    traced = _stdout_observation_summary(trace_stdout_bytes)
    if observed is None or traced is None or observed_frame is None or traced_frame is None:
        return False, None

    observed_offset = stdout_bytes.find(observed_frame)
    traced_offset = trace_stdout_bytes.find(traced_frame)
    other_bytes_equal = (
        observed_offset == traced_offset
        and stdout_bytes[:observed_offset] == trace_stdout_bytes[:traced_offset]
        and stdout_bytes[observed_offset + len(observed_frame):]
        == trace_stdout_bytes[traced_offset + len(traced_frame):]
    )
    diff_indices = [
        index for index, (left, right) in enumerate(zip(observed["gpr"], traced["gpr"]))
        if left != right
    ]
    summary: dict[str, object] = {
        "gpr_diff_indices": diff_indices,
        "memory_equal": observed["memory"] == traced["memory"],
        "checkpoint_pc_equal": observed["checkpoint_pc"] == traced["checkpoint_pc"],
        "other_bytes_equal": other_bytes_equal,
    }
    matches = (
        bool(diff_indices)
        and set(diff_indices).issubset(NATIVE_OBSERVER_TEMP_GPRS)
        and summary["memory_equal"] is True
        and summary["checkpoint_pc_equal"] is True
        and other_bytes_equal
    )
    return matches, summary


def _read_run_record(
    root: Path,
    remote_dir: str,
    local_sha: str,
    index: int,
    native_identity: dict | None,
) -> dict:
    run_dir = root / f"run-{index}"
    stdout_bytes = (run_dir / "stdout.bin").read_bytes()
    stderr_bytes = (run_dir / "stderr.txt").read_bytes()
    exit_code = int((run_dir / "exit.code").read_text(encoding="utf-8").strip())
    timeout_flag = (run_dir / "primary-timeout.flag").read_text(
        encoding="ascii",
    ).strip()
    if timeout_flag not in {"0", "1"}:
        raise ValueError("native RV64 primary timeout marker is invalid")
    primary_timed_out = timeout_flag == "1"
    trace_status = (run_dir / "trace.status").read_text(encoding="utf-8", errors="replace").strip()
    trace_stdout_bytes = (run_dir / "trace.stdout").read_bytes()
    trace_stderr_text = (run_dir / "trace.stderr").read_text(encoding="utf-8", errors="replace")
    return {
        "exit_code": exit_code,
        "primary_timed_out": primary_timed_out,
        "stdout_hex": stdout_bytes.hex(),
        "stderr": _run_stderr_record(
            remote_dir=remote_dir,
            local_sha=local_sha,
            index=index,
            exit_code=exit_code,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            trace_status=trace_status,
            trace_stdout_bytes=trace_stdout_bytes,
            trace_stderr_text=trace_stderr_text,
            native_identity=native_identity,
            primary_timed_out=primary_timed_out,
        ),
    }




def _run_unlocked(elf_path: Path, stratum: str = "linux-user") -> int:
    elf_path = elf_path.resolve()
    if not elf_path.is_file():
        return _fail(f"native RV64 runner input ELF does not exist: {elf_path}")
    if reason := _rv64_elf_error(elf_path):
        return _fail(f"native RV64 runner input ELF is invalid: {reason}")
    connection = _remote_connection()
    if isinstance(connection, int):
        return connection
    env, known_hosts, remote_dir = connection
    qdir = shlex.quote(remote_dir)
    local_sha = sha256_file(elf_path)
    trace_source = _trace_helper_source()

    with tempfile.TemporaryDirectory(prefix="native-rv64-runner-") as tmp:
        tmp_dir = Path(tmp)
        input_archive_path = tmp_dir / "inputs.tar.gz"
        results_archive_path = tmp_dir / "results.tgz"
        results_dir = tmp_dir / "results"
        pc_manifest_path = tmp_dir / "pc-manifest.txt"
        remote_dir_created = False
        remote_run_started = False
        remote_results_complete = False
        trace_helper, helper_status = _build_trace_helper()
        manifest_ok, manifest_status = _write_pc_manifest(elf_path, pc_manifest_path)
        helper_cache = (
            _remote_trace_helper_cache_path(trace_helper)
            if trace_helper is not None and manifest_ok else None
        )
        helper_cache_status = "miss" if helper_cache is not None else "disabled"

        try:
            create_command = _remote_storage_preflight(
                qdir, helper_cache=helper_cache,
            )
            remote_dir_marker = os.environ.get("RQ1_NATIVE_REMOTE_DIR_MARKER")
            remote_trace_pid_marker = os.environ.get(
                "RQ1_NATIVE_REMOTE_TRACE_PID_MARKER",
            )

            def mark_remote_dir_created() -> None:
                nonlocal remote_dir_created
                remote_dir_created = True
                if remote_dir_marker:
                    marker_path = Path(remote_dir_marker)
                    marker_path.parent.mkdir(parents=True, exist_ok=True)
                    marker_path.write_text(remote_dir + "\n", encoding="utf-8")

            def mark_remote_trace_pid(process_group: int) -> None:
                if remote_trace_pid_marker:
                    marker_path = Path(remote_trace_pid_marker)
                    marker_path.parent.mkdir(parents=True, exist_ok=True)
                    marker_path.write_text(f"{process_group}\n", encoding="ascii")

            def mark_remote_trace_finished(process_group: int) -> None:
                if remote_trace_pid_marker:
                    marker_path = Path(remote_trace_pid_marker)
                    try:
                        if marker_path.read_text(encoding="ascii").strip() == str(process_group):
                            marker_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            def cancel_remote_trace(process_group: int) -> bool:
                return _cancel_remote_trace_group(
                    remote_dir, process_group, env, known_hosts,
                )

            input_files: list[tuple[Path, str]] = [(elf_path, "case.elf")]
            helper_upload_required = False
            remote_helper_name = "rv64-ptrace-trace"
            if trace_helper is not None and manifest_ok:
                input_files.append((trace_helper, trace_helper.name))
                helper_upload_required = True
                remote_helper_name = trace_helper.name
                input_files.append((pc_manifest_path, "pc-manifest.txt"))
            elif manifest_ok and trace_source.is_file():
                input_files.extend((
                    (pc_manifest_path, "pc-manifest.txt"),
                    (trace_source, "ptrace_trace.c"),
                ))

            if trace_helper is not None and manifest_ok:
                local_helper = f"./{shlex.quote(remote_helper_name)}"
                if helper_cache is not None:
                    cached_helper = shlex.quote(helper_cache[0])
                    cached_runner = (
                        f"if [ \"$cache_status\" = hit ]; then "
                    f"run_trace ptrace-manifest {cached_helper} "
                        "./case.elf pc-manifest.txt; else "
                    )
                    cached_runner_end = "fi; "
                else:
                    cached_runner = ""
                    cached_runner_end = ""
                trace_clause = (
                    cached_runner
                    +
                    f"if chmod 700 {shlex.quote(remote_helper_name)}; then "
                    + f"run_trace ptrace-manifest {local_helper} "
                    + "./case.elf pc-manifest.txt; else "
                    + ": > \"$run_dir/trace.stdout\"; "
                    + "printf '%s\\n' 'unavailable:ptrace-helper-chmod-failed' "
                    + "> \"$run_dir/trace.status\"; fi; "
                    + cached_runner_end
                )
            elif manifest_ok and trace_source.is_file():
                trace_clause = (
                    "if gcc -O2 -static -Wall -Wextra -Wno-unused-variable "
                    "-o rv64-ptrace-trace.tmp.$$ ptrace_trace.c > \"$run_dir/trace-build.stdout\" "
                    "2> \"$run_dir/trace-build.stderr\" && "
                    "mv rv64-ptrace-trace.tmp.$$ rv64-ptrace-trace; then "
                    "if chmod 700 rv64-ptrace-trace; then "
                    "run_trace ptrace-manifest ./rv64-ptrace-trace "
                    "./case.elf pc-manifest.txt; else "
                    ": > \"$run_dir/trace.stdout\"; "
                    "printf '%s\\n' 'unavailable:ptrace-helper-chmod-failed' "
                    "> \"$run_dir/trace.status\"; fi; "
                    "else rm -f rv64-ptrace-trace.tmp.$$; : > \"$run_dir/trace.stdout\"; "
                    "cat \"$run_dir/trace-build.stderr\" > \"$run_dir/trace.stderr\"; "
                    "printf '%s\\n' 'unavailable:ptrace-remote-build-failed' "
                    "> \"$run_dir/trace.status\"; fi; "
                )
            else:
                status = shlex.quote(
                    f"unavailable:ptrace-{_safe_status(helper_status)}"
                    if trace_helper is None
                    else f"unavailable:manifest-{_safe_status(manifest_status)}"
                )
                trace_clause = (
                    ": > \"$run_dir/trace.stdout\"; : > \"$run_dir/trace.stderr\"; "
                    f"printf '%s\\n' {status} > \"$run_dir/trace.status\"; "
                )

            with tarfile.open(
                input_archive_path, "w:gz", compresslevel=1,
            ) as input_archive:
                for path, archive_name in input_files:
                    input_archive.add(path, arcname=archive_name, recursive=False)

            cache_store_clause = ""
            if helper_cache is not None and helper_upload_required:
                remote_helper = f"{remote_dir}/{remote_helper_name}"
                store_command = _remote_trace_helper_cache_store(
                    remote_helper, helper_cache,
                )
                cache_store_clause = (
                    "if [ \"$cache_status\" = hit ]; then "
                    "printf 'K1_TRACE_HELPER_CACHE=hit\\n' >&2; "
                    f"elif {store_command}; then "
                    "printf 'K1_TRACE_HELPER_CACHE=stored\\n' >&2; "
                    "else printf 'K1_TRACE_HELPER_CACHE=store-failed\\n' >&2; fi; "
                )

            timeout = DEFAULT_TIMEOUT_SEC
            trace_script = shlex.quote(_gdb_trace_script())
            remote_cmd = (
                f"cache_status=disabled; {create_command} && "
                "printf '%s\\n' K1_NATIVE_DIRECTORY_CREATED >&2 && "
                f"cd {qdir} || exit 73; "
                + (
                    "if [ \"$cache_status\" = hit ]; then "
                    "tar -xzf - case.elf pc-manifest.txt || exit 73; "
                    "else tar -xzf - || exit 73; fi; "
                    if helper_cache is not None else "tar -xzf - || exit 73; "
                )
                + "printf '%s\\n' K1_NATIVE_INPUT_READY >&2; "
                + "test -f case.elf && chmod 700 case.elf && "
                + "sha256sum case.elf > case.sha256 || exit 73; "
                + f"{cache_store_clause}"
                + "run_dir=\"run-0\"; "
                + "mkdir \"$run_dir\"; "
                + "run_trace() { "
                + "trace_kind=$1; shift; "
                + "if ! command -v setsid >/dev/null 2>&1 || [ ! -x /bin/kill ]; then "
                + "printf '%s\\n' 'unavailable:trace-process-group-unavailable' > \"$run_dir/trace.status\"; "
                + "return 127; fi; "
                + "setsid sh -c 'out=$1; err=$2; shift 2; exec \"$@\" > \"$out\" 2> \"$err\"' "
                + "sh \"$run_dir/trace.stdout\" \"$run_dir/trace.stderr\" \"$@\" & "
                + "trace_pid=$!; "
                + "printf '%s\\n' \"$trace_pid\" > \"$run_dir/trace.pid\"; "
                + "printf 'K1_NATIVE_TRACE_PID=%s\\n' \"$trace_pid\" >&2; "
                + "trap '/bin/kill -TERM -- \"-$trace_pid\" 2>/dev/null || true; "
                + "wait \"$trace_pid\" 2>/dev/null || true; exit 143' HUP TERM INT; "
                + "wait \"$trace_pid\"; trace_exit=$?; trap - HUP TERM INT; "
                + "printf '%s:%s\\n' \"$trace_kind\" \"$trace_exit\" > \"$run_dir/trace.status\"; "
                + "}; "
                + "printf '%s\\n' K1_NATIVE_PRIMARY_STARTED >&2; "
                + f"(LC_ALL=C timeout --verbose --signal=TERM --kill-after=2s {timeout}s "
                + "sh -c 'exec \"$1\" > \"$2\" 2> \"$3\"' sh ./case.elf "
                + "\"$run_dir/stdout.bin\" \"$run_dir/stderr.txt\") "
                + "2> \"$run_dir/primary-timeout.stderr\"; "
                + "status=$?; "
                + "timed_out=0; "
                + "if grep -q '^timeout:' \"$run_dir/primary-timeout.stderr\"; then timed_out=1; fi; "
                + "printf '%s\\n' \"$status\" > \"$run_dir/exit.code\"; "
                + "printf '%s\\n' \"$timed_out\" > \"$run_dir/primary-timeout.flag\"; "
                + "printf '%s\\n' K1_NATIVE_PRIMARY_FINISHED >&2; "
                + "if [ \"$timed_out\" -eq 1 ]; then "
                + ": > \"$run_dir/trace.stdout\"; "
                + ": > \"$run_dir/trace.stderr\"; "
                + "printf '%s\\n' 'unavailable:primary-timeout' > \"$run_dir/trace.status\"; "
                + "else "
                + "printf '%s\\n' K1_NATIVE_TRACE_STARTED >&2; "
                + f"{trace_clause}"
                + "if grep -q '^unavailable' \"$run_dir/trace.status\" && command -v gdb >/dev/null 2>&1; then "
                + f"printf '%s' {trace_script} > \"$run_dir/trace.gdb\"; "
                + "run_trace gdb-single-step gdb -q -nx -batch "
                + "-x \"$run_dir/trace.gdb\" ./case.elf; "
                + "fi; "
                + "fi; "
                + "printf '%s\\n' K1_NATIVE_TRACE_FINISHED >&2; "
                + "printf '%s\\n' K1_NATIVE_PACKAGE_STARTED >&2; "
                + "setsid sh -c 'exec tar --checkpoint=10000 "
                + "--checkpoint-action=echo=K1_NATIVE_PACKAGE_PROGRESS "
                + "--exclude=run-0/trace.pid -czf results.tgz case.sha256 run-0' & "
                + "package_pid=$!; "
                + "printf '%s\\n' \"$package_pid\" > \"$run_dir/trace.pid\"; "
                + "printf 'K1_NATIVE_PACKAGE_PID=%s\\n' \"$package_pid\" >&2; "
                + "wait \"$package_pid\"; package_status=$?; "
                + "[ \"$package_status\" -eq 0 ] || exit 74; "
                + "printf '%s\\n' K1_NATIVE_PACKAGE_FINISHED >&2; "
                + "result_bytes=$(wc -c < results.tgz | tr -d '[:space:]'); "
                + "case \"$result_bytes\" in ''|*[!0-9]*) exit 74;; esac; "
                + "printf 'K1_NATIVE_RESULT_BYTES=%s\\n' \"$result_bytes\" >&2; "
                + "setsid sh -c 'exec cat -- results.tgz' & "
                + "transfer_pid=$!; "
                + "printf '%s\\n' \"$transfer_pid\" > \"$run_dir/trace.pid\"; "
                + "printf 'K1_NATIVE_TRANSFER_PID=%s\\n' \"$transfer_pid\" >&2; "
                + "wait \"$transfer_pid\"; transfer_status=$?; "
                + "[ \"$transfer_status\" -eq 0 ] || exit 75; "
                + "printf '%s\\n' K1_NATIVE_RESULT_SENT >&2; exit 0"
            )
            remote_run_started = True
            # Input and complete trace archives stream over one SSH session.
            # Trace time is unbounded; package/transfer idle timeouts reset on progress.
            executed = _remote_stream_process(
                remote_cmd, input_archive_path, results_archive_path,
                env, known_hosts, timeout=REMOTE_EXEC_TIMEOUT_SEC,
                on_remote_dir_created=mark_remote_dir_created,
                on_remote_trace_pid=mark_remote_trace_pid,
                on_remote_trace_finished=mark_remote_trace_finished,
                cancel_remote_trace=cancel_remote_trace,
            )
            if executed.returncode != 0:
                return _fail(
                    "native RV64 remote execution wrapper failed", executed.stderr,
                    failure_class=_remote_transport_failure_class(executed.stderr),
                )

            if b"K1_TRACE_HELPER_CACHE=hit" in executed.stderr:
                helper_cache_status = "hit"
            elif b"K1_TRACE_HELPER_CACHE=stored" in executed.stderr:
                helper_cache_status = "stored"
            elif b"K1_TRACE_HELPER_CACHE=store-failed" in executed.stderr:
                helper_cache_status = "store-failed"

            _extract_results_archive(results_archive_path, results_dir)

            native_identity, native_identity_error = _native_reference_identity_from_output(
                executed.stderr, known_hosts, stratum,
            )
            if native_identity_error is not None:
                return _fail(
                    f"native RV64 identity probe failed: {native_identity_error}",
                    failure_class=(
                        "k1-transport-unavailable"
                        if native_identity_error.startswith("k1-transport-unavailable:")
                        else None
                    ),
                )

            remote_sha = (results_dir / "case.sha256").read_text(encoding="utf-8").split()[0].lower()
            if remote_sha != local_sha:
                return _fail(
                    f"native RV64 ELF hash mismatch: local={local_sha} remote={remote_sha}",
                    code=126,
                )

            record = _read_run_record(results_dir, remote_dir, local_sha, 0, native_identity)
            remote_results_complete = True
            if record["primary_timed_out"]:
                return _fail(
                    f"native RV64 guest execution timed out after {timeout}s",
                    code=124,
                    failure_class="k1-timeout",
                )
            sys.stderr.write(f"NATIVE_RV64_HELPER_CACHE={helper_cache_status}\n")
            sys.stdout.buffer.write(bytes.fromhex(record["stdout_hex"]))
            sys.stdout.buffer.flush()
            sys.stderr.write(record["stderr"])
            sys.stderr.flush()
            return int(record["exit_code"])
        except _RemoteExecutionTimeout as exc:
            return _fail(
                f"native RV64 K1 {exc.phase} phase exceeded its response deadline",
                code=124,
                failure_class="k1-timeout",
            )
        except _RemoteTransportError as exc:
            return _fail(
                f"native RV64 remote transport failed: {exc}",
                failure_class="k1-transport-unavailable",
            )
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
            return _fail(f"native RV64 result processing failed: {exc}")
        finally:
            if remote_dir_created:
                if remote_run_started and not remote_results_complete:
                    sys.stderr.write(f"NATIVE_RV64_PRESERVED_REMOTE_DIR={remote_dir}\n")
                else:
                    try:
                        _cleanup(remote_dir, env, known_hosts)
                    except _RemoteTransportError as cleanup_error:
                        sys.stderr.write(f"NATIVE_RV64_CLEANUP_GAP={cleanup_error}\n")


def run(elf_path: Path, stratum: str = "linux-user") -> int:
    if os.environ.get("RQ1_NATIVE_REFERENCE_LOCK_HELD") == "1":
        return _run_unlocked(elf_path, stratum)
    lock_deadline = time.monotonic() + REMOTE_EXEC_TIMEOUT_SEC * 3 + 150
    try:
        with NativeReferenceLock(lock_deadline):
            return _run_unlocked(elf_path, stratum)
    except TimeoutError:
        return _fail(
            "native RV64 shared reference gate timed out",
            code=124,
            failure_class="k1-timeout",
        )
    except OSError as exc:
        return _fail(
            f"native RV64 shared reference gate unavailable: {exc}",
            failure_class="k1-transport-unavailable",
        )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 1:
        stratum, elf = "linux-user", args[0]
    elif len(args) == 3 and args[0] == "--stratum" and args[1] in {"bare-metal", "linux-user"}:
        stratum, elf = args[1], args[2]
    else:
        return _fail("usage: python -m framework.native.rv64_runner [--stratum bare-metal|linux-user] <rv64-elf>")
    return run(Path(elf), stratum)


if __name__ == "__main__":
    raise SystemExit(main())

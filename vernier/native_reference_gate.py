from __future__ import annotations

import errno
import os
import time


NATIVE_REFERENCE_LOCK_PATH = os.environ.get(
    "RQ1_NATIVE_REFERENCE_LOCK_PATH", "/tmp/rq1-native-rv64-reference.lock"
)
# ponytail: measured K1 throughput peaks at 8; repeat the board sweep before raising the cap.
MAX_NATIVE_REFERENCE_PARALLELISM = 8


def configured_native_reference_parallelism(value: str | None) -> int:
    try:
        configured = int(value or "8")
    except ValueError as error:
        raise ValueError(
            "RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer"
        ) from error
    if configured < 1:
        raise ValueError("RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer")
    return min(configured, MAX_NATIVE_REFERENCE_PARALLELISM)


NATIVE_REFERENCE_PARALLELISM = configured_native_reference_parallelism(
    os.environ.get("RQ1_NATIVE_REFERENCE_PARALLELISM")
)


class NativeReferenceLock:
    """跨进程限制 native-rv64 reference 并发，共享 gate 兼容旧独占锁。"""

    def __init__(self, deadline: float) -> None:
        self._fd = -1
        self._slot_fd = -1
        self._deadline = deadline
        self._fcntl = None

    def _lock(self, descriptor: int, operation: int) -> None:
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native-reference-lock-timeout")
            try:
                self._fcntl.flock(
                    descriptor, operation | self._fcntl.LOCK_NB,
                )
                return
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                time.sleep(min(0.1, remaining))

    def __enter__(self) -> NativeReferenceLock:
        if os.name != "posix":
            return self
        import fcntl

        self._fcntl = fcntl
        try:
            self._fd = os.open(
                NATIVE_REFERENCE_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644,
            )
            if NATIVE_REFERENCE_PARALLELISM == 1:
                self._lock(self._fd, fcntl.LOCK_EX)
                return self

            # Legacy callers take LOCK_EX on the base file. Shared holders let
            # this version use slots without overlapping an older runner.
            self._lock(self._fd, fcntl.LOCK_SH)
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("native-reference-lock-timeout")
                for slot in range(NATIVE_REFERENCE_PARALLELISM):
                    descriptor = os.open(
                        f"{NATIVE_REFERENCE_LOCK_PATH}.slot-{slot}",
                        os.O_CREAT | os.O_RDWR, 0o644,
                    )
                    try:
                        fcntl.flock(
                            descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except OSError as error:
                        os.close(descriptor)
                        if error.errno not in (errno.EAGAIN, errno.EACCES):
                            raise
                        continue
                    self._slot_fd = descriptor
                    return self
                time.sleep(min(0.1, remaining))
        except BaseException:
            self._release()
            raise

    def __exit__(self, *_exc_info: object) -> None:
        self._release()

    def _release(self) -> None:
        if self._slot_fd >= 0:
            self._fcntl.flock(self._slot_fd, self._fcntl.LOCK_UN)
            os.close(self._slot_fd)
            self._slot_fd = -1
        if self._fd >= 0:
            self._fcntl.flock(self._fd, self._fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# Keep the existing private name available to framework and compatibility callers.
_NativeReferenceLock = NativeReferenceLock

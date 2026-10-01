import os
import socket
import time
from threading import Lock

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

_MAX_RETRIES = 3


class _PortStartupLock:
    """Serialize free_port -> child-listen across workers in one container."""

    def __init__(self) -> None:
        self._thread_lock = Lock()
        self._fd: int | None = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            if fcntl is not None:
                path = os.environ.get(
                    "RQ1_RVVM_PORT_LOCK", "/tmp/rq1-rvvm-port-startup.lock",
                )
                self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None
        finally:
            self._thread_lock.release()


_PORT_STARTUP_LOCK = _PortStartupLock()


def rsp_decode(payload: bytes) -> bytes:
    result = bytearray()
    index = 0
    while index < len(payload):
        item = payload[index]
        if item == 0x7D:
            if index + 1 == len(payload):
                raise ValueError("RSP trailing escape")
            result.append(payload[index + 1] ^ 0x20)
            index += 2
            continue
        if item == 0x2A:
            if not result or index + 1 == len(payload):
                raise ValueError("RSP invalid run-length encoding")
            count = payload[index + 1] - 29
            if count < 3 or count > 97:
                raise ValueError("RSP invalid run-length encoding")
            result.extend([result[-1]] * count)
            index += 2
            continue
        result.append(item)
        index += 1
    return bytes(result)


def rsp_encode(payload: bytes) -> bytes:
    return b"".join(
        (b"}" + bytes((item ^ 0x20,))) if item in b"#$}*" else bytes((item,))
        for item in payload
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RspClient:
    """Minimal shared RSP transport; backend startup/stop semantics stay local."""

    _max_packet_size: int | None = None

    def __init__(self, sock: socket.socket, *, deadline: float | None = None):
        self.sock = sock
        self.deadline = deadline
        self._transaction_timeout: float | None = None
        self._pending_replies: list[str] = []

    def _bound_timeout(self) -> None:
        if self.deadline is None:
            return
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RSP backend deadline expired")
        timeout = self._transaction_timeout
        self.sock.settimeout(remaining if timeout is None else min(timeout, remaining))

    def _begin_transaction(self) -> float | None:
        previous = self.sock.gettimeout()
        self._transaction_timeout = previous
        try:
            self._bound_timeout()
        except TimeoutError:
            self._transaction_timeout = None
            raise
        return previous

    def _end_transaction(self, previous: float | None) -> None:
        self.sock.settimeout(previous)
        self._transaction_timeout = None

    def _read_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            self._bound_timeout()
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("RSP closed before complete payload")
            data.extend(chunk)
        return bytes(data)

    def _read_reply(self, marker: bytes | None = None) -> tuple[bytes, bool]:
        marker = marker or self._read_exact(1)
        while marker != b"$":
            marker = self._read_exact(1)
        body = bytearray()
        escaped = run_length = False
        while True:
            try:
                item = self._read_exact(1)
            except ConnectionError:
                if b"*" in body:
                    return bytes(body), False
                raise
            if item == b"#" and not escaped and not run_length:
                break
            body.extend(item)
            if escaped or run_length:
                escaped = False
                run_length = False
            elif item == b"}":
                escaped = True
                run_length = False
            else:
                run_length = item == b"*"
            if self._max_packet_size is not None and len(body) > self._max_packet_size:
                raise ValueError("RSP packet too large")
        checksum_text = self._read_exact(2)
        try:
            received = int(checksum_text.decode("ascii"), 16)
        except ValueError:
            return bytes(body), False
        body = bytes(body)
        valid = received == (sum(body) & 0xFF)
        if valid:
            try:
                body = rsp_decode(body)
            except ValueError:
                return body, False
        return body, valid

    def _read_acked_reply(self, marker: bytes | None = None) -> str | None:
        body, valid = self._read_reply(marker)
        if not valid:
            self.sock.sendall(b"-")
            return None
        try:
            response = body.decode("utf-8")
        except UnicodeDecodeError:
            self.sock.sendall(b"-")
            return None
        self.sock.sendall(b"+")
        return response

    def request(self, payload: str) -> str:
        raw = payload.encode("ascii")
        encoded = rsp_encode(raw)
        packet = b"$" + encoded + b"#" + f"{sum(encoded) & 0xFF:02x}".encode("ascii")
        previous = self._begin_transaction()
        try:
            for _ in range(_MAX_RETRIES):
                self._bound_timeout()
                self.sock.sendall(packet)
                while True:
                    self._bound_timeout()
                    ack = self.sock.recv(1)
                    if not ack:
                        raise ConnectionError("RSP closed before packet acknowledgement")
                    if ack == b"$":
                        reply = self._read_acked_reply(ack)
                        if reply is not None:
                            self._pending_replies.append(reply)
                        continue
                    if ack in {b"+", b"-"}:
                        break
                if ack == b"-":
                    continue
                response = self._read_socket_reply()
                if self._pending_replies:
                    self._pending_replies.append(response)
                    return self._pending_replies.pop(0)
                return response
            raise ConnectionError("RSP packet was rejected after retries")
        finally:
            self._end_transaction(previous)

    def _read_socket_reply(self) -> str:
        for _ in range(_MAX_RETRIES):
            response = self._read_acked_reply()
            if response is not None:
                return response
        raise ConnectionError("RSP reply checksum failed after retries")

    def read_reply(self) -> str:
        """Read one pending RSP reply packet (acknowledging it).

        ``qRcmd`` handlers may emit console output followed by a separate
        result packet; the caller drains them with repeated calls.
        """
        previous = self._begin_transaction()
        try:
            if self._pending_replies:
                return self._pending_replies.pop(0)
            return self._read_socket_reply()
        finally:
            self._end_transaction(previous)

    def qrcmd(self, command: str, *, timeout: float | None = None) -> str:
        """Run a backend monitor command through the ``qRcmd,<hex>`` packet.

        Some stubs reply with a console-output packet (``O...``) before the
        result packet; drain the console packet when present.  Normal requests
        use the client's backend deadline; callers may pass a short timeout
        for cleanup commands.
        """
        previous = self.sock.gettimeout()
        try:
            self.sock.settimeout(timeout)
            reply = self.request("qRcmd," + command.encode("utf-8").hex())
            while reply.startswith("O") and reply != "OK":
                reply = self.read_reply()
            return reply
        finally:
            self.sock.settimeout(previous)

    def close(self) -> None:
        self.sock.close()

from __future__ import annotations

from contextlib import suppress
import os
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from framework.adapters.rsp import RspClient
from framework.execution_environment import ExecutionEnvironmentError, resolve_x86_64_runner


_CONNECT_POLL_SEC = 0.05
_DEFAULT_TIMEOUT_SEC = 20
_PC_REGNUM = 32
_GDB_LOCK_TIMEOUT_SEC = 100


class _Flock:
    """Advisory lock with a bounded wait: tail workers fail fast with
    `gdb-lock-timeout` instead of blocking a whole wave on one stub."""

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._fd = -1

    def __enter__(self) -> _Flock:
        if self._path is None:
            return self
        try:
            import errno
            import fcntl
        except ImportError:  # Windows control plane: no fcntl, skip locking
            return self
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + _GDB_LOCK_TIMEOUT_SEC
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return self
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError("gdb-lock-timeout") from exc
                time.sleep(0.1)

    def __exit__(self, *exc_info: object) -> None:
        if self._fd >= 0:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)


@dataclass(frozen=True)
class GdbRegisterLayout:
    fpr_regnums: tuple[int, ...]
    fflags_regnum: int | None
    frm_regnum: int | None
    fcsr_regnum: int | None = None
    # Zicntr 计数器（D-440 前置：QEMU target.xml riscv-csr.xml 暴露，
    # libriscv stub 无 target.xml，保持空）。
    csr_regnums: dict[str, int] = field(default_factory=dict)
    # RVV observer（QEMU riscv-vector.xml / riscv-csr.xml；libriscv 无 V 面）。
    vstart_regnum: int | None = None
    vl_regnum: int | None = None
    vtype_regnum: int | None = None
    vlenb_regnum: int | None = None
    v_regnums: tuple[int, ...] = ()

LIBRISCV_TRANSLATED_FP_LAYOUT = GdbRegisterLayout(
    fpr_regnums=tuple(range(33, 65)),
    fflags_regnum=66,
    frm_regnum=67,
    fcsr_regnum=68,
)
_FPR_ABI_TO_ARCH = dict(zip(
    "ft0 ft1 ft2 ft3 ft4 ft5 ft6 ft7 fs0 fs1 fa0 fa1 fa2 fa3 fa4 fa5 fa6 fa7 "
    "fs2 fs3 fs4 fs5 fs6 fs7 fs8 fs9 fs10 fs11 ft8 ft9 ft10 ft11".split(),
    (f"f{index}" for index in range(32)),
))


def exit_checkpoint_pc(elf_path: str) -> int:
    nm = resolve_x86_64_runner("riscv64-linux-gnu-nm", "riscv64-linux-gnu-nm")
    proc = subprocess.run(
        [nm, "-n", str(Path(elf_path).resolve())],
        timeout=_DEFAULT_TIMEOUT_SEC,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nm failed: {proc.returncode}")
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[-1] != "exit_checkpoint":
            continue
        try:
            value = int(fields[0], 16)
        except ValueError:
            continue
        if 0 <= value <= ((1 << 64) - 1):
            return value
    raise RuntimeError("nm is missing exit_checkpoint")


def _read_vector_state(client: _GdbRemoteClient, layout: GdbRegisterLayout) -> dict[str, object]:
    """Read the complete live RVV state, or return no state on a partial layout."""

    vector_fields = (
        layout.vstart_regnum,
        layout.vl_regnum,
        layout.vtype_regnum,
        layout.vlenb_regnum,
    )
    if not (
        len(layout.v_regnums) == 32
        and len(set(layout.v_regnums)) == 32
        and all(regnum is not None for regnum in (
            layout.vstart_regnum, layout.vl_regnum, layout.vtype_regnum, layout.vlenb_regnum
        ))
    ):
        if layout.v_regnums or any(regnum is not None for regnum in vector_fields):
            return {
                "vector_observer_status": "gap",
                "vector_observer_gap": "gdb-vector-register-layout-incomplete",
            }
        return {}
    try:
        vlenb = client.read_register_u64(layout.vlenb_regnum)
        if vlenb <= 0:
            return {
                "vector_observer_status": "gap",
                "vector_observer_gap": "gdb-vector-vlenb-invalid",
            }
        registers = [client.read_register_bytes(regnum) for regnum in layout.v_regnums]
        if any(
            not isinstance(value, str)
            or not value.startswith("0x")
            or len(value) != 2 + 2 * vlenb
            or any(char not in "0123456789abcdefABCDEF" for char in value[2:])
            for value in registers
        ):
            return {
                "vector_observer_status": "gap",
                "vector_observer_gap": "gdb-vector-register-width-mismatch",
            }
        state: dict[str, object] = {
            "vector.vstart": client.read_register_u64(layout.vstart_regnum),
            "vector.vl": client.read_register_u64(layout.vl_regnum),
            "vector.vtype": client.read_register_u64(layout.vtype_regnum),
            "vector.vlenb": vlenb,
            "vector.registers": registers,
        }
    except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        return {
            "vector_observer_status": "gap",
            "vector_observer_gap": f"gdb-vector-read-failed:{exc}",
        }
    state["vector_observer_status"] = "observed"
    return state


def collect_final_fp_state(
    launch_command: list[str],
    *,
    port: int,
    elf_path: str,
    static_layout: GdbRegisterLayout | None = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    lock_path: str | None = None,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        stop_pc = exit_checkpoint_pc(elf_path)
    except (ExecutionEnvironmentError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return None, f"exit-checkpoint-unavailable:{exc}"
    lock = _Flock(lock_path)
    try:
        lock.__enter__()
    except TimeoutError as exc:
        return None, str(exc)
    try:
        proc: subprocess.Popen[bytes] | None = None
        client: _GdbRemoteClient | None = None
        try:
            proc = subprocess.Popen(
                launch_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            client = _connect_client(port, proc, timeout_sec)
            client.request("qSupported:xmlRegisters=i386;qXfer:features:read+")
            layout = static_layout or discover_riscv_register_layout(client)
            if layout is None:
                return None, "fp-register-layout-unavailable"
            if client.request(f"Z0,{stop_pc:x},4") != "OK":
                return None, "gdb-breakpoint-rejected"
            stop_error = _advance_to_stop_pc(client, stop_pc, timeout_sec=timeout_sec)
            if stop_error is not None:
                return None, stop_error
            state: dict[str, object] = {}
            if layout.fpr_regnums and (
                len(layout.fpr_regnums) != 32
                or len(set(layout.fpr_regnums)) != 32
            ):
                return None, "gdb-fpr-register-layout-invalid"
            if layout.fpr_regnums:
                fpr_rawbits = []
                for index, regnum in enumerate(layout.fpr_regnums):
                    rawbits = client.read_register_bytes(regnum)
                    if len(rawbits) not in (10, 18):
                        return None, f"gdb-fpr-width-unexpected:{index}:{len(rawbits) - 2}"
                    fpr_rawbits.append(int(rawbits, 0))
                if layout.fflags_regnum is not None and layout.frm_regnum is not None:
                    fflags = client.read_register_u64(layout.fflags_regnum)
                    frm = client.read_register_u64(layout.frm_regnum)
                    if fflags > 0x1F or frm > 0x7:
                        return None, "gdb-fcsr-value-out-of-range"
                elif layout.fcsr_regnum is not None:
                    fcsr = client.read_register_u64(layout.fcsr_regnum)
                    fflags = fcsr & 0x1F
                    frm = (fcsr >> 5) & 0x7
                else:
                    return None, "gdb-fcsr-layout-unavailable"
                state["fpr_rawbits"] = fpr_rawbits
                state["fflags"] = fflags
                state["frm"] = frm
                state["fp_observer"] = "observed"
            state.update(_read_vector_state(client, layout))
            if layout.csr_regnums:
                mstatus_regnum = layout.csr_regnums.get("mstatus")
                if mstatus_regnum is not None:
                    try:
                        state["mstatus_fs"] = (client.read_register_u64(mstatus_regnum) >> 13) & 0x3
                        state["mstatus_fs_observer"] = "observed"
                    except RuntimeError as exc:
                        state.update(
                            mstatus_fs=None,
                            mstatus_fs_observer="gap",
                            mstatus_fs_gap=f"gdb-mstatus-fs-unavailable: {exc}",
                        )
                # Zicntr 读回：QEMU linux-user 对 `zicntr=false` 的 accepted
                # child-disable 配置仍退休计数器（D-440 root），字段化读回
                # 是 CSR 写回/disable 一致性的直接证据。
                for csr_name in ("cycle", "time", "instret"):
                    regnum = layout.csr_regnums.get(csr_name)
                    if regnum is not None:
                        try:
                            state[f"csr.{csr_name}"] = client.read_register_u64(regnum)
                            state[f"csr.{csr_name}_observer"] = "observed"
                        except RuntimeError as exc:
                            state[f"csr.{csr_name}_observer"] = "gap"
                            state[f"csr.{csr_name}_gap"] = f"gdb-csr-unavailable: {exc}"
            return state, None
        except (ConnectionError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return None, f"gdb-final-fp-state-failed:{exc}"
        finally:
            if client is not None:
                client.close()
            if proc is not None and proc.poll() is None:
                with suppress(OSError, ProcessLookupError, subprocess.TimeoutExpired):
                    if os.name == "posix" and isinstance(getattr(proc, "pid", None), int):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.communicate(timeout=1)
    finally:
        lock.__exit__(None, None, None)


def _advance_to_stop_pc(
    client: _GdbRemoteClient, stop_pc: int, *, timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
) -> str | None:
    """Drive the stub to the observer-entry sentinel before reading final FP state.

    Some translated-path stubs stop at the beginning of the translated block that
    contains ``exit_checkpoint`` instead of the symbol address itself.  We keep
    the strict contract, but advance with single-step until the PC is exactly the
    sentinel or it becomes clear the stub cannot reach it.
    """
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    stop_reply = client.request("c")
    previous_pc = None
    while True:
        if time.monotonic() >= deadline:
            return "gdb-stop-timeout"
        if not stop_reply or stop_reply[0] not in {"S", "T", "X"}:
            return f"gdb-stop-reply-unexpected:{stop_reply!r}"
        actual_pc = client.read_register_u64(_PC_REGNUM)
        if actual_pc == stop_pc:
            return None
        if actual_pc == previous_pc:
            return f"gdb-stop-pc-mismatch:stalled:{hex(actual_pc)}"
        previous_pc = actual_pc
        stop_reply = client.request("s")


def discover_riscv_register_layout(client: _GdbRemoteClient) -> GdbRegisterLayout | None:
    """Discover FP + RVV layout from the stub's target.xml (QEMU 11.x)."""
    xml_by_annex: dict[str, ET.Element] = {}
    try:
        xml_by_annex["target.xml"] = _parse_gdb_xml(client.read_xml_annex("target.xml"))
    except (ET.ParseError, RuntimeError, ConnectionError):
        return None
    regnums: dict[str, int] = {}

    def walk(element: ET.Element, next_regnum: int) -> int:
        for child in element:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "include":
                href = child.attrib.get("href")
                if not href:
                    continue
                if href not in xml_by_annex:
                    xml_by_annex[href] = _parse_gdb_xml(client.read_xml_annex(href))
                next_regnum = walk(xml_by_annex[href], next_regnum)
                continue
            if tag == "reg":
                name = child.attrib.get("name")
                if not name:
                    continue
                raw_regnum = child.attrib.get("regnum")
                current = int(raw_regnum, 0) if raw_regnum is not None else next_regnum
                regnums[_FPR_ABI_TO_ARCH.get(name, name)] = current
                next_regnum = current + 1
                continue
            next_regnum = walk(child, next_regnum)
        return next_regnum

    try:
        walk(xml_by_annex["target.xml"], 0)
    except (ET.ParseError, RuntimeError, ConnectionError, ValueError):
        return None
    fpr_regnums = tuple(regnums[f"f{index}"] for index in range(32)) if all(f"f{index}" in regnums for index in range(32)) else ()
    v_regnums = tuple(regnums[f"v{index}"] for index in range(32)) if all(f"v{index}" in regnums for index in range(32)) else ()
    csr_regnums = {
        name: regnums[name]
        for name in ("mstatus", "cycle", "time", "instret")
        if name in regnums
    }
    if not fpr_regnums and not (v_regnums and regnums.get("vstart") is not None) and not csr_regnums:
        return None
    return GdbRegisterLayout(
        fpr_regnums=fpr_regnums,
        fflags_regnum=regnums.get("fflags"),
        frm_regnum=regnums.get("frm"),
        fcsr_regnum=regnums.get("fcsr"),
        csr_regnums=csr_regnums,
        vstart_regnum=regnums.get("vstart"),
        vl_regnum=regnums.get("vl"),
        vtype_regnum=regnums.get("vtype"),
        vlenb_regnum=regnums.get("vlenb"),
        v_regnums=v_regnums,
    )


def _parse_gdb_xml(text: str) -> ET.Element:
    return ET.fromstring(text.replace("xi:include", "include"))


def _connect_client(port: int, proc: subprocess.Popen[bytes], timeout_sec: int) -> _GdbRemoteClient:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(
                "gdb-stub-exited:"
                f"rc={proc.returncode},stdout={stdout.decode('utf-8', errors='replace')[-200:]},"
                f"stderr={stderr.decode('utf-8', errors='replace')[-200:]}"
            )
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=_CONNECT_POLL_SEC)
            return _GdbRemoteClient(sock, timeout_sec)
        except OSError as exc:
            last_error = exc
            time.sleep(_CONNECT_POLL_SEC)
    raise ConnectionError(f"gdb-stub-connect-timeout:{last_error}")


class _GdbRemoteClient(RspClient):
    _max_packet_size = 4096

    def __init__(self, sock: socket.socket, timeout: float) -> None:
        super().__init__(sock)
        sock.settimeout(timeout)

    def close(self) -> None:
        with suppress(OSError):
            super().close()

    def _read_register_data(self, regnum: int) -> bytes:
        response = self.request(f"p{regnum:x}")
        if not response or response.startswith("E"):
            raise RuntimeError(f"gdb-read-register-failed:{regnum}:{response}")
        if len(response) % 2 != 0:
            raise RuntimeError(f"gdb-read-register-malformed:{regnum}:{response}")
        try:
            return bytes.fromhex(response)
        except ValueError as exc:
            raise RuntimeError(f"gdb-read-register-malformed:{regnum}:{response}") from exc

    def read_register_u64(self, regnum: int) -> int:
        """Read one register as a little-endian unsigned 64-bit value.

        Accepts both 8-byte replies (QEMU gdbstub XLEN registers) and 4-byte
        replies: the libriscv RSP stub returns fflags/frm/fcsr as uint32
        (see libriscv rsp_server.hpp handle_readreg, regnums 66/67/68).
        """

        data = self._read_register_data(regnum)
        if len(data) not in (4, 8):
            raise RuntimeError(f"gdb-read-register-width-unexpected:{regnum}:{len(data)}")
        return int.from_bytes(data, "little")

    def read_register_bytes(self, regnum: int) -> str:
        """Read one register as a canonical little-endian hex string (any width)."""

        data = self._read_register_data(regnum)
        return "0x" + data[::-1].hex()

    def read_xml_annex(self, annex: str) -> str:
        offset = 0
        chunks: list[str] = []
        while True:
            response = self.request(f"qXfer:features:read:{annex}:{offset:x},400")
            if not response:
                raise RuntimeError(f"gdb-empty-annex:{annex}")
            kind, payload = response[0], response[1:]
            if kind not in {"m", "l"}:
                raise RuntimeError(f"gdb-annex-read-failed:{annex}:{response}")
            if kind == "m" and not payload:
                raise RuntimeError(f"gdb-annex-empty-fragment:{annex}:{offset:x}")
            chunks.append(payload)
            offset += len(payload.encode("utf-8"))
            if kind == "l":
                return "".join(chunks)

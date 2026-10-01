from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from framework._util import sha256_file
from framework.direct_case import OBSERVATION_HEADER_SIZE
from framework.adapters.runner import _observation_frame
from framework.native.rv64_runner import _rv64_elf_error


GDB_TIMEOUT = int(os.environ.get("RQ1_K1_GDB_TIMEOUT", "180"))
GDB_PORT = int(os.environ.get("RQ1_K1_GDB_PORT", "1024"))
GDB_HOST = os.environ.get("RQ1_K1_GDB_HOST", "127.0.0.1")
REMOTE_EXIT = 125


def _identity(host: str, port: int, gdb: str | None, openocd: str | None) -> dict[str, object]:
    return {
        "id": "R-K1-BOARD",
        "engine": "k1-board",
        "backend": "native-rv64",
        "target": "T-K1-BOARD",
        "source": "framework.native.k1_jtag_runner",
        "transport": "jtag-openocd-gdb",
        "gdb_target": f"{host}:{port}",
        "gdb": gdb,
        "openocd": openocd,
    }


def _gap(elf_sha: str, reason: str, identity: dict[str, object]) -> int:
    sys.stderr.write(
        f"NATIVE_RV64_REFERENCE elf_sha256={elf_sha} remote_exit_code={REMOTE_EXIT}\n"
    )
    sys.stderr.write("RV_NATIVE_REFERENCE_IDENTITY=" + json.dumps(identity, separators=(",", ":")) + "\n")
    sys.stderr.write(f"RV_NATIVE_REFERENCE_GAP={reason}\n")
    return REMOTE_EXIT


def _tool(env_name: str, names: tuple[str, ...]) -> str | None:
    configured = os.environ.get(env_name)
    return configured or next((shutil.which(name) for name in names if shutil.which(name)), None)


def _endpoint() -> tuple[str, int, bool]:
    value = os.environ.get("RQ1_K1_GDB_SERVER")
    if value:
        host, port = value.rsplit(":", 1)
        return host, int(port), True
    configured = "RQ1_K1_GDB_HOST" in os.environ or "RQ1_K1_GDB_PORT" in os.environ
    return GDB_HOST, GDB_PORT, configured


def _wait_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _start_openocd(
    temporary: Path, host: str, port: int, openocd: str,
) -> tuple[subprocess.Popen[bytes] | None, str | None]:
    target = os.environ.get(
        "RQ1_K1_OPENOCD_TARGET",
        str(Path(__file__).with_name("spacemit-k1.cfg")),
    )
    interface = os.environ.get("RQ1_K1_OPENOCD_INTERFACE", "interface/jlink.cfg")
    command = [openocd]
    scripts = os.environ.get("RQ1_K1_OPENOCD_SCRIPTS")
    if scripts:
        command += ["-s", scripts]
    command += [
        "-f", interface,
        "-c", f"set CORES {os.environ.get('RQ1_K1_OPENOCD_CORES', '8')}",
        "-f", target,
        "-c", f"gdb_port {port}",
    ]
    log = temporary / "openocd.log"
    stream = log.open("wb")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if _wait_port(host, port, 20):
        return process, None
    try:
        process.terminate()
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        finally:
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
    stream.close()
    detail = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return None, "k1-jtag-openocd-start-failed:" + (detail[-1][:160] if detail else "no-log")


def _stop_openocd(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _elf_symbols(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        return {}
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    section_size = struct.unpack_from("<H", data, 58)[0]
    section_count = struct.unpack_from("<H", data, 60)[0]
    symbols: dict[str, int] = {}
    for index in range(section_count):
        offset = section_offset + index * section_size
        if offset + 64 > len(data):
            continue
        fields = struct.unpack_from("<IIQQQQIIQQ", data, offset)
        if fields[1] not in {2, 11} or fields[6] >= section_count:
            continue
        string_section = section_offset + fields[6] * section_size
        string_offset, string_size = struct.unpack_from("<QQ", data, string_section + 24)[0:2]
        strings = data[string_offset:string_offset + string_size]
        entry_size = fields[9] or 24
        for entry in range(0, fields[5], entry_size):
            symbol_offset = fields[4] + entry
            if symbol_offset + 24 > len(data):
                continue
            name_offset, _, _, _, value, _ = struct.unpack_from("<IBBHQQ", data, symbol_offset)
            end = strings.find(b"\0", name_offset)
            if name_offset < len(strings) and end >= 0:
                name = strings[name_offset:end].decode("ascii", errors="ignore")
                if name in {"_start", "capsule_exit", "result_buffer", "result_end", "stack_area", "stack_top"}:
                    symbols[name] = value
    return symbols


def _last_ebreak(path: Path) -> int | None:
    data = path.read_bytes()
    if len(data) < 64:
        return None
    program_offset = struct.unpack_from("<Q", data, 32)[0]
    program_size = struct.unpack_from("<H", data, 54)[0]
    program_count = struct.unpack_from("<H", data, 56)[0]
    last = None
    for index in range(program_count):
        offset = program_offset + index * program_size
        if offset + 56 > len(data):
            continue
        p_type, flags, file_offset, address, _, file_size, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, offset,
        )
        if p_type != 1 or not flags & 1:
            continue
        segment = data[file_offset:file_offset + file_size]
        cursor = 0
        while cursor + 2 <= len(segment):
            halfword = segment[cursor] | segment[cursor + 1] << 8
            size = 2 if (halfword & 3) != 3 else 4
            if cursor + size > len(segment):
                break
            instruction = segment[cursor:cursor + size]
            if instruction in {b"\x73\x00\x10\x00", b"\x02\x90"}:
                last = address + cursor
            cursor += size
    return last


def _ecalls_before(path: Path, limit: int) -> list[int]:
    data = path.read_bytes()
    program_offset = struct.unpack_from("<Q", data, 32)[0]
    program_size = struct.unpack_from("<H", data, 54)[0]
    program_count = struct.unpack_from("<H", data, 56)[0]
    calls = []
    for index in range(program_count):
        offset = program_offset + index * program_size
        if offset + 56 > len(data):
            continue
        p_type, flags, file_offset, address, _, file_size, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, offset,
        )
        if p_type != 1 or not flags & 1:
            continue
        segment = data[file_offset:file_offset + file_size]
        cursor = 0
        while cursor + 4 <= len(segment):
            halfword = segment[cursor] | segment[cursor + 1] << 8
            size = 2 if (halfword & 3) != 3 else 4
            if cursor + size > len(segment):
                break
            if size == 4 and segment[cursor:cursor + 4] == b"\x73\x00\x00\x00":
                location = address + cursor
                if location < limit:
                    calls.append(location)
            cursor += size
    return calls


def _terminal_address(path: Path, symbols: dict[str, int]) -> int | None:
    if symbols.get("capsule_exit"):
        return symbols["capsule_exit"]
    ebreak = _last_ebreak(path)
    if ebreak is None:
        return None
    calls = _ecalls_before(path, ebreak)
    if calls:
        cluster = [calls[-1]]
        for location in reversed(calls[:-1]):
            if cluster[0] - location <= 16:
                cluster.insert(0, location)
            else:
                break
        return cluster[0]
    return ebreak


def _text_address(path: Path) -> int | None:
    data = path.read_bytes()
    if len(data) < 64:
        return None
    program_offset = struct.unpack_from("<Q", data, 32)[0]
    program_size = struct.unpack_from("<H", data, 54)[0]
    program_count = struct.unpack_from("<H", data, 56)[0]
    addresses = []
    for index in range(program_count):
        offset = program_offset + index * program_size
        if offset + 56 > len(data):
            continue
        p_type, flags, _, address, _, _, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, offset,
        )
        if p_type == 1 and flags & 1:
            addresses.append(address)
    return min(addresses) if addresses else None


def _rebase_elf(path: Path, directory: Path, target: int) -> tuple[Path | None, str | None]:
    source = _text_address(path)
    if source is None or source == target:
        return path, None
    objcopy = _tool(
        "RQ1_K1_OBJCOPY",
        ("riscv64-linux-gnu-objcopy", "riscv64-unknown-elf-objcopy", "objcopy"),
    )
    if not objcopy:
        return None, "k1-jtag-objcopy-missing"
    delta = target - source
    adjustment = f"0x{delta:x}" if delta >= 0 else f"-0x{-delta:x}"
    output = directory / "k1-rebased.elf"
    result = subprocess.run(
        [objcopy, f"--change-addresses={adjustment}", str(path), str(output)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return (output, None) if result.returncode == 0 else (
        None, "k1-jtag-elf-rebase-failed",
    )


def _gdb_script(
    dump: Path, host: str, port: int, stop_expression: str, end_expression: str,
) -> str:
    dump_path = str(dump).replace("\\", "/")
    stop_condition = (
        stop_expression.removeprefix("*")
        if stop_expression.startswith("*") else f"&{stop_expression}"
    )
    return "\n".join(
        (
            "set pagination off",
            "set confirm off",
            "set print thread-events off",
            f"set remotetimeout {GDB_TIMEOUT}",
            f"target extended-remote {host}:{port}",
            "monitor reset halt",
            "load",
            "set $pc = &_start",
            f"break {stop_expression}",
            "continue",
            f"if $pc == {stop_condition}",
            '  printf "K1_RQ1_STOP=complete\\n"',
            f"  dump binary memory {dump_path} &result_buffer {end_expression}",
            "else",
            '  printf "K1_RQ1_STOP=unexpected\\n"',
            "end",
            "monitor reset halt",
            "quit",
            "",
        )
    )


def _frame(data: bytes) -> bytes | None:
    frame = _observation_frame(data)
    if frame is None or len(frame) < OBSERVATION_HEADER_SIZE:
        return None
    checkpoint = int.from_bytes(frame[8:16], "little")
    if checkpoint == 0 or checkpoint & 1:
        return None
    if int.from_bytes(frame[16:24], "little") != 0:
        return None
    return frame


def run(elf_path: Path) -> int:
    elf_path = elf_path.resolve()
    if not elf_path.is_file():
        return _gap("", "k1-jtag-elf-missing", _identity(GDB_HOST, GDB_PORT, None, None))
    elf_sha = sha256_file(elf_path)
    if reason := _rv64_elf_error(elf_path):
        return _gap(elf_sha, f"k1-jtag-invalid-elf:{reason}", _identity(GDB_HOST, GDB_PORT, None, None))

    host, port, endpoint_configured = _endpoint()
    gdb = _tool(
        "RQ1_K1_GDB",
        ("riscv64-unknown-linux-gnu-gdb", "riscv64-unknown-elf-gdb", "gdb-multiarch"),
    )
    openocd = _tool("RQ1_K1_OPENOCD", ("openocd",))
    identity = _identity(host, port, gdb, openocd)
    if not gdb:
        return _gap(elf_sha, "k1-jtag-gdb-missing", identity)
    if not endpoint_configured and not openocd:
        return _gap(elf_sha, "k1-jtag-transport-not-configured", identity)
    if not endpoint_configured and openocd:
        endpoint_configured = True

    with tempfile.TemporaryDirectory(prefix="rq1-k1-jtag-") as directory:
        temporary = Path(directory)
        dump = temporary / "observation.bin"
        script = temporary / "run.gdb"
        target_address = int(os.environ.get("RQ1_K1_TEXT_ADDRESS", "0x07200000"), 0)
        load_elf, rebase_error = _rebase_elf(elf_path, temporary, target_address)
        if rebase_error or load_elf is None:
            return _gap(elf_sha, rebase_error or "k1-jtag-elf-rebase-failed", identity)
        identity["loaded_elf_sha256"] = sha256_file(load_elf)
        identity["loaded_text_address"] = f"0x{_text_address(load_elf):x}"
        identity["source_text_address"] = f"0x{_text_address(elf_path):x}"
        identity["address_rebased"] = load_elf != elf_path
        symbols = _elf_symbols(load_elf)
        terminal = _terminal_address(load_elf, symbols)
        stop = "capsule_exit" if symbols.get("capsule_exit") else (
            f"*0x{terminal:x}" if terminal is not None else None
        )
        end = "&result_end" if symbols.get("result_end") else (
            "&stack_area" if symbols.get("stack_area") else None
        )
        if not symbols.get("_start") or not symbols.get("result_buffer") or not stop or not end:
            return _gap(elf_sha, "k1-jtag-elf-observer-symbols-missing", identity)
        script.write_text(_gdb_script(dump, host, port, stop, end), encoding="utf-8")
        openocd_process = None
        try:
            if openocd and not ("RQ1_K1_GDB_SERVER" in os.environ or "RQ1_K1_GDB_HOST" in os.environ):
                openocd_process, error = _start_openocd(temporary, host, port, openocd)
                if error:
                    return _gap(elf_sha, error, identity)
            elif not endpoint_configured:
                return _gap(elf_sha, "k1-jtag-gdb-server-missing", identity)

            try:
                result = subprocess.run(
                    [gdb, "-q", "-nx", "--batch", "-x", str(script), str(load_elf)],
                    capture_output=True,
                    timeout=GDB_TIMEOUT + 30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return _gap(elf_sha, f"k1-jtag-gdb-failed:{type(exc).__name__}", identity)
            output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            if "K1_RQ1_STOP=complete" not in output:
                return _gap(elf_sha, "k1-jtag-guest-stop", identity)
            if not dump.is_file():
                return _gap(elf_sha, "k1-jtag-observation-missing", identity)
            frame = _frame(dump.read_bytes())
            if frame is None:
                return _gap(elf_sha, "k1-jtag-observation-frame-invalid", identity)
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()
            sys.stderr.write(f"NATIVE_RV64_REFERENCE elf_sha256={elf_sha} remote_exit_code=0\n")
            sys.stderr.write("RV_NATIVE_REFERENCE_IDENTITY=" + json.dumps(identity, separators=(",", ":")) + "\n")
            sys.stderr.write("RV_EXECUTED_PCS=[]\nNATIVE_RV64_TRACE_STATUS=jtag-breakpoint\n")
            return 0
        finally:
            _stop_openocd(openocd_process)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m framework.native.k1_jtag_runner <rv64-elf>\n")
        return 2
    return run(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())

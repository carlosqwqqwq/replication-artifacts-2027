"""最小共享工具：文件哈希与严格类型校验（单一来源）。

文件哈希逻辑集中于本模块，避免跨模块漂移。
"""

import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import tempfile
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .paths import windows_extended_length_path

_TIMING_ONLY_FIELDS = frozenset(("csr.cycle", "csr.time", "csr.instret"))
_PROTECTED_ENTRY_LABELS = frozenset(
    {"_start", "main", "init", "h0_start", "test_done", "write_tohost"}
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "-inf" if value < 0 else "inf" if value > 0 else "nan"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "to_record"):
        return json_safe(value.to_record())
    if is_dataclass(value):
        return json_safe(asdict(value))
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def object_field(value: object, name: str, default: object = None) -> object:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _path_in_root(path: str | Path, root: str | Path, kind: str) -> Path:
    root = Path(root).absolute().resolve()
    candidate = Path(path).absolute()
    current = candidate
    while current != root and current.parent != current:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"{kind}-symlink-or-junction")
        current = current.parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{kind}-outside-root") from error
    return resolved


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def execute_process(
    command: list[str], timeout: float | None, *, cwd: str | None = None,
    env: dict[str, str] | None = None,
    on_timeout: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                pass
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        else:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired as cleanup_error:
            stdout = cleanup_error.output or error.output or b""
            stderr = cleanup_error.stderr or error.stderr or b""
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout or error.output, stderr=stderr or error.stderr,
        ) from error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def strip_c_comments(source: str) -> str:
    result, quote, escaped, comment = [], None, False, False
    index = 0
    while index < len(source):
        char = source[index]
        if comment:
            if source.startswith("*/", index):
                result.extend((" ", " "))
                comment, index = False, index + 2
            else:
                result.append(char if char in "\r\n" else " ")
                index += 1
            continue
        if quote is not None:
            result.append(char)
            escaped = not escaped if char == "\\" else False
            if char == quote and not escaped:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
        elif source.startswith("/*", index):
            result.extend((" ", " "))
            comment, index = True, index + 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def atomic_write_json(path: Path, data: object, *, compact: bool = False) -> None:
    path = windows_extended_length_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        handle.write(json.dumps(
            data,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _riscv_elf_identity(value: Path | bytes) -> bool:
    data = value.read_bytes() if isinstance(value, Path) else value
    return len(data) >= 20 and data[:4] == b"\x7fELF" and data[5] == 1 \
        and int.from_bytes(data[18:20], "little") == 243


def is_riscv_elf(value: Path | bytes) -> bool:
    data = value.read_bytes() if isinstance(value, Path) else value
    if not _riscv_elf_identity(data):
        return False
    minimum = {1: 52, 2: 64}.get(data[4], 0)
    return minimum > 0 and len(data) >= minimum


def elf_load_segments(path: Path) -> tuple[int, tuple[tuple[int, bytes, int], ...]]:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[5] != 1 or data[18:20] != b"\xf3\x00":
        raise ValueError("expected little-endian RISC-V capsule")
    if data[4] == 2:
        entry, phoff = struct.unpack_from("<QQ", data, 24)
        phentsize, phnum = struct.unpack_from("<HH", data, 54)
        fmt = "<IIQQQQQQ"
        expected_size = 56
        fields = (0, 2, 3, 5, 6)
    elif data[4] == 1:
        entry, phoff = struct.unpack_from("<II", data, 24)
        phentsize, phnum = struct.unpack_from("<HH", data, 42)
        fmt = "<IIIIIIII"
        expected_size = 32
        fields = (0, 1, 2, 4, 5)
    else:
        raise ValueError("unsupported ELF class")
    if phentsize != expected_size:
        raise ValueError("unexpected ELF program-header size")
    loads = []
    for index in range(phnum):
        offset = phoff + index * phentsize
        fields_data = struct.unpack_from(fmt, data, offset)
        p_type, file_offset, virtual, file_size, memory_size = (
            fields_data[index] for index in fields
        )
        if p_type == 1 and memory_size:
            loads.append((virtual, data[file_offset : file_offset + file_size], memory_size))
    if not loads:
        raise ValueError("capsule has no loadable segment")
    return entry, tuple(loads)


def elf_executable_loads(path: Path) -> tuple[tuple[int, int], ...]:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[5] != 1:
        raise ValueError("expected little-endian RISC-V ELF")
    if data[4] == 2:
        phoff = struct.unpack_from("<Q", data, 32)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 54)
        fmt = "<IIQQQQQQ"
        vaddr_index, memsz_index, flags_index = 3, 6, 1
    elif data[4] == 1:
        phoff = struct.unpack_from("<I", data, 28)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 42)
        fmt = "<IIIIIIII"
        vaddr_index, memsz_index, flags_index = 2, 5, 6
    else:
        raise ValueError("unsupported ELF class")
    if phentsize != struct.calcsize(fmt):
        raise ValueError("unexpected ELF program-header size")
    return tuple(
        (fields[vaddr_index], fields[memsz_index])
        for index in range(phnum)
        if (fields := struct.unpack_from(fmt, data, phoff + index * phentsize))[0] == 1
        and fields[flags_index] & 1
        and fields[memsz_index]
    )


def elf_writable_loads(path: Path) -> tuple[tuple[int, int], ...]:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[5] != 1:
        raise ValueError("expected little-endian RISC-V ELF")
    if data[4] == 2:
        phoff = struct.unpack_from("<Q", data, 32)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 54)
        fmt = "<IIQQQQQQ"
        vaddr_index, memsz_index, flags_index = 3, 6, 1
    elif data[4] == 1:
        phoff = struct.unpack_from("<I", data, 28)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 42)
        fmt = "<IIIIIIII"
        vaddr_index, memsz_index, flags_index = 2, 5, 6
    else:
        raise ValueError("unsupported ELF class")
    if phentsize != struct.calcsize(fmt):
        raise ValueError("unexpected ELF program-header size")
    return tuple(
        (fields[vaddr_index], fields[memsz_index])
        for index in range(phnum)
        if (fields := struct.unpack_from(fmt, data, phoff + index * phentsize))[0] == 1
        and fields[flags_index] & 2
        and fields[memsz_index]
    )


def pc_path_in_ranges(pcs, ranges) -> bool:
    return all(
        pc % 2 == 0 and any(base <= pc < base + size for base, size in ranges)
        for pc in pcs
    )


def strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def canonical_digest(data: object) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compare_mask_fields(compare_mask: object) -> tuple[str, ...]:
    fields = [
        name
        for name in ("outcome", "checkpoint_pc", "signal", "fault_pc", "fault_address")
        if getattr(compare_mask, name, False)
    ]
    fields.extend(f"gpr.x{index}" for index in getattr(compare_mask, "gpr_indices", ()))
    fields.extend(f"memory.{region}" for region in getattr(compare_mask, "memory_region_ids", ()))
    fields.extend(f"extra.{key}" for key in getattr(compare_mask, "extra_state_keys", ()))
    return tuple(dict.fromkeys(fields))


def is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def require_sha256(value: str, name: str) -> str:
    if not is_sha256_digest(value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def pc_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("PC must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = int(value, 0)
    else:
        raise ValueError("PC must be an integer")
    if parsed < 0 or parsed > (1 << 64) - 1:
        raise ValueError("PC is out of range")
    return parsed


def bounded_small_int(value: object, maximum: int, name: str) -> int:
    parsed = pc_int(value)
    if parsed > maximum:
        raise ValueError(f"{name} is out of range")
    return parsed


def pc_digest(pcs) -> str:
    return canonical_digest([hex(pc_int(pc)) for pc in pcs])


def pc_path(pcs) -> tuple[list[str], str | None]:
    sequence = [hex(pc_int(pc)) for pc in pcs]
    return sequence, pc_digest(sequence) if sequence else None


def path_through_stop(pcs, stop_pc):
    if type(stop_pc) is int and stop_pc in pcs:
        return pcs[: len(pcs) - pcs[::-1].index(stop_pc)]
    return pcs


def git_head(path: Path) -> str | None:
    git_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        git_env.pop(name, None)
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            timeout=3,
            capture_output=True,
            check=False,
            text=True,
            env=git_env,
        )
    except (OSError, subprocess.TimeoutExpired):
        git_dir = Path(path) / ".git"
        try:
            if git_dir.is_file():
                pointer = git_dir.read_text(encoding="utf-8").strip()
                if pointer.startswith("gitdir:"):
                    git_dir = (git_dir.parent / pointer[7:].strip()).resolve()
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                head = (git_dir / head[5:]).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return head if len(head) in {40, 64} and all(char in "0123456789abcdefABCDEF" for char in head) else None
    stripped = proc.stdout.strip()
    if proc.returncode == 0 and stripped:
        return stripped
    git_dir = Path(path) / ".git"
    try:
        if git_dir.is_file():
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_dir = (git_dir.parent / pointer[7:].strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (git_dir / head[5:]).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head if len(head) in {40, 64} and all(
        char in "0123456789abcdefABCDEF" for char in head
    ) else None


def git_worktree_clean(path: Path) -> bool:
    git_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        git_env.pop(name, None)
    try:
        for command in (
            ("diff", "--ignore-space-at-eol", "--quiet"),
            ("diff", "--cached", "--ignore-space-at-eol", "--quiet"),
        ):
            if subprocess.run(
                ["git", "-c", "core.filemode=false", "-C", str(path), *command], timeout=15,
                capture_output=True, check=False, text=True, env=git_env,
            ).returncode != 0:
                return False
        proc = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            timeout=15, capture_output=True, check=False, text=True, env=git_env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and not any(line.startswith("??") for line in proc.stdout.splitlines())


def runner_contract_gap(message: str) -> int:
    sys.stderr.write(
        "RV_RUNNER_CONTRACT_GAP="
        + json.dumps(message, separators=(",", ":"))
        + "\n"
    )
    sys.stderr.write(message + "\n")
    return 127

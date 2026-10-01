"""共享 runner 契约：归约观察事实并组装 runner 环境。

实际执行由各 adapter 和 execution plane 负责。
"""

from collections.abc import Sequence
import re
import struct
import subprocess
from pathlib import Path

from ..direct_case import (
    OBSERVATION_MAGIC,
    OBSERVATION_HEADER_SIZE,
    _observation_frame_end,
    _TRAP_FIELD_ALIASES,
)
from .._util import pc_int, sha256_file
from .contracts import (
    _observer_value_is_non_observed_marker,
    _observer_value_is_valid,
)


# direct_runner 会把这些 adapter 桥接字段提升到 extra_state。
GUEST_TRAP_BRIDGE_FIELDS = (
    "guest_trap",
    "guest_cause",
    "guest_epc",
    "guest_tval",
)


def guest_trap_observed(extra_state: dict | None) -> bool:
    if not isinstance(extra_state, dict):
        return False
    if extra_state.get("guest_trap") != "delivered":
        return False
    values = {}
    for canonical, aliases in _TRAP_FIELD_ALIASES.items():
        present = [
            extra_state[name] for name in aliases
            if extra_state.get(name) is not None
            and not _observer_value_is_non_observed_marker(
                extra_state.get(f"{name}_observer")
            )
        ]
        if any(type(value) is not int or not 0 <= value < (1 << 64) for value in present):
            return False
        if len(present) > 1 and len(set(present)) != 1:
            return False
        if present:
            values[canonical] = present[0]
    cause, epc = values.get("trap.cause"), values.get("trap.epc")
    return (
        type(cause) is int and 0 <= cause < 64
        and type(epc) is int and not epc & 1
    )


def terminal_ebreak_observed(
    extra_state: dict | None, *, risk_pcs: Sequence[int] | None = None,
) -> bool:
    if not guest_trap_observed(extra_state):
        return False
    trap = canonical_trap_state(extra_state)
    if trap.get("trap.cause") != 3 or extra_state.get("trap_observer") not in {
        "qemu-linux-user-ebreak", "libriscv-machine-exception",
    }:
        return False
    if extra_state.get("trap_observer") == "libriscv-machine-exception" \
            and extra_state.get("libriscv_machine_exception_code") != 7:
        return False
    if trap.get("trap.tval") not in (None, 0):
        return False
    epc = trap.get("trap.epc")
    if type(epc) is not int or not 0 <= epc < (1 << 64) or epc & 1:
        return False
    epc_key = next(
        (alias for alias in _TRAP_FIELD_ALIASES["trap.epc"] if extra_state.get(alias) is not None),
        "trap.epc",
    )
    if _observer_value_is_non_observed_marker(extra_state.get(f"{epc_key}_observer")):
        return False
    if risk_pcs is None:
        return True
    try:
        risks = {pc_int(value) for value in risk_pcs}
    except (TypeError, ValueError):
        return False
    return bool(risks) and epc not in risks


def observed_state_fields(
    extra_state: dict | None,
    trap_state: dict | None = None,
) -> list[str]:
    """Name only fields backed by a successful live observation."""

    state = dict(extra_state) if isinstance(extra_state, dict) else {}
    if isinstance(trap_state, dict):
        state.update(trap_state)
    fields: list[str] = []

    def add(name: str) -> None:
        if name not in fields:
            fields.append(name)

    def is_u64(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < (1 << 64)

    for canonical in canonical_trap_state(state):
        add(canonical)

    if state.get("fp_observer") == "observed":
        rawbits = state.get("fpr_rawbits")
        if (
            isinstance(rawbits, (list, tuple))
            and len(rawbits) == 32
            and all(is_u64(value) for value in rawbits)
        ):
            add("fpr_rawbits")
        fflags = state.get("fflags")
        if isinstance(fflags, int) and not isinstance(fflags, bool) and 0 <= fflags <= 0x1F:
            add("fflags")
        frm = state.get("frm")
        if isinstance(frm, int) and not isinstance(frm, bool) and 0 <= frm <= 0x7:
            add("frm")
    fs = state.get("mstatus_fs")
    if (
        state.get("mstatus_fs_observer") == "observed"
        and isinstance(fs, int)
        and not isinstance(fs, bool)
        and 0 <= fs <= 0x3
    ):
        add("csr.mstatus.fs")

    privilege = state.get("csr.priv")
    if (
        state.get("csr.priv_observer") == "observed"
        and is_u64(privilege)
        and 0 <= privilege <= 0x3
    ):
        add("privilege.mode")

    vector_markers = (
        state.get("vector_observer_status"),
        state.get("vector_observer"),
    )
    if "observed" in vector_markers and all(
        marker in (None, "observed") for marker in vector_markers
    ):
        vector_csrs = state.get("vector_csrs")
        vector_csrs = vector_csrs if isinstance(vector_csrs, dict) else {}
        for canonical, *aliases in (
            ("vector.vl", "vl"),
            ("vector.vtype", "vtype"),
            ("vector.vstart", "vstart"),
            ("vector.vlenb", "vlenb"),
            ("vector.registers", "vector_registers"),
            ("fflags", "fflags"),
            ("frm", "frm"),
            ("vxsat", "vxsat"),
            ("vxrm", "vxrm"),
        ):
            value = next((state.get(alias) for alias in (canonical, *aliases)
                          if state.get(alias) is not None), None)
            if value is None:
                value = vector_csrs.get(canonical.removeprefix("vector."))
            if _observer_value_is_valid(canonical, value):
                add(canonical)

    for name in ("branch_outcome", "after_pc", "control.target", "link"):
        value = state.get(name)
        if _observer_value_is_valid(name, value):
            add(name)

    for name, value in state.items():
        if not isinstance(name, str):
            continue
        if (
            name.startswith("csr.")
            and not name.endswith(("_observer", "_gap"))
            and is_u64(value)
            and _observer_value_is_valid(name, value)
            and state.get(f"{name}_observer") == "observed"
        ):
            if name == "csr.priv" and value > 0x3:
                continue
            add(name)
        elif (
            name.startswith(("memory.", "signal.", "reservation."))
            and not name.endswith(("_observer", "_gap"))
            and isinstance(state.get(f"{name}_observer"), str)
            and not _observer_value_is_non_observed_marker(state[f"{name}_observer"])
            and _observer_value_is_valid(name, value)
        ):
            add(name)
    return fields


def canonical_trap_state(trap_state: dict | None) -> dict[str, int]:
    """Project a delivered guest trap into the shared ``trap.*`` namespace."""

    if not isinstance(trap_state, dict):
        return {}
    if trap_state.get("guest_trap") != "delivered":
        return {
            name: trap_state[name]
            for name in ("trap.cause", "trap.epc", "trap.tval")
            if type(trap_state.get(name)) is int
            and 0 <= trap_state[name] < (64 if name == "trap.cause" else 1 << 64)
            and not (name == "trap.epc" and trap_state[name] & 1)
        } if all(type(trap_state.get(name)) is int for name in ("trap.cause", "trap.epc", "trap.tval")) \
            and not trap_state["trap.epc"] & 1 else {}
    result: dict[str, int] = {}
    for canonical, aliases in _TRAP_FIELD_ALIASES.items():
        values = [
            trap_state[alias]
            for alias in aliases
            if alias in trap_state
            and trap_state[alias] is not None
            and not _observer_value_is_non_observed_marker(
                trap_state.get(f"{alias}_observer")
            )
        ]
        if (
            values
            and all(
                type(value) is int
                and 0 <= value < (64 if canonical == "trap.cause" else 1 << 64)
                for value in values
            )
            and len(set(values)) == 1
            and not (canonical == "trap.epc" and values[0] & 1)
        ):
            result[canonical] = values[0]
    return result


def _empty_fp_state() -> dict[str, object]:
    return {
        "fpr_rawbits": None,
        "fflags": None,
        "frm": None,
        "fp_observer": "not-required",
        "mstatus_fs": None,
        "mstatus_fs_observer": "not-required",
        "mstatus_fs_gap": None,
    }


def _memory_fault_state(fault: dict[str, object], backend: str) -> dict[str, object]:
    observer = f"{backend}-memory-hook"
    return {
        "fault_address": int(fault["address"]),
        "fault_address_observer": observer,
        "memory_access": int(fault["access"]),
        "memory_size": int(fault["size"]),
        "memory_fault_observer": observer,
    }


def is_zbb_word(word: int) -> bool:
    word &= 0xFFFFFFFF
    opcode = word & 0x7F
    funct3 = (word >> 12) & 0x7
    funct7 = (word >> 25) & 0x7F
    return (
        (opcode == 0x33 and funct7 in {0x20, 0x05, 0x30} and funct3 in {1, 4, 5, 6, 7})
        or (opcode == 0x3B and funct7 == 0x30 and funct3 in {1, 5})
        or (opcode == 0x13 and funct7 in {0x30, 0x34} and funct3 in {1, 5})
    )


def _read_mailbox(rsp, mailbox: int, observation_size: int, backend: str) -> bytes:
    if (
        type(mailbox) is not int
        or mailbox < 0
        or type(observation_size) is not int
        or observation_size <= 0
    ):
        raise ValueError(f"{backend} GDB mailbox arguments are invalid")
    chunks = []
    # New RVVM builds accept 240-byte debug reads; retain compatibility with
    # older stubs by lowering the chunk size when the first reply is 64 bytes.
    chunk_size = 240 if backend == "RVVM" else 64
    offset = 0
    while offset < observation_size:
        size = min(chunk_size, observation_size - offset)
        packet = f"m{mailbox + offset:x},{size:x}"
        memory = rsp.request(packet)
        for _ in range(8):
            if memory[:1].lower() not in {"s", "t", "x"}:
                break
            read_reply = getattr(rsp, "read_reply", None)
            memory = read_reply() if read_reply is not None else rsp.request(packet)
        else:
            raise RuntimeError(f"{backend} GDB mailbox read kept returning stop packets")
        if not isinstance(memory, str):
            raise RuntimeError(f"{backend} GDB mailbox reply is not text")
        if memory.startswith("E"):
            raise RuntimeError(f"{backend} GDB mailbox read failed: {memory}")
        try:
            chunk = bytes.fromhex(memory)
        except ValueError as exc:
            raise RuntimeError(f"{backend} GDB mailbox reply is malformed: {memory!r}") from exc
        if len(chunk) != size:
            if backend == "RVVM" and chunk_size == 240 and len(chunk) == 64:
                chunk_size = 64
                size = len(chunk)
            else:
                raise RuntimeError(f"{backend} GDB mailbox reply size mismatch: {len(chunk)} != {size}")
        if not chunk:
            raise RuntimeError(f"{backend} GDB mailbox reply size mismatch: {len(chunk)} != {size}")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _observation_frame(stdout: bytes) -> bytes | None:
    """Return the one complete RVOBS1 frame in a bounded output stream.

    Adapters are allowed to mix diagnostics with the binary frame.  The
    frame itself is still strict: multiple top-level frames, or a truncated
    marker outside the selected frame, invalidate the observation.
    """
    if not isinstance(stdout, (bytes, bytearray)):
        return None
    stdout = bytes(stdout)
    candidates = []
    incomplete = []
    offset = stdout.find(OBSERVATION_MAGIC)
    while offset >= 0:
        frame_end = _observation_frame_end(stdout, offset)
        if frame_end is None:
            incomplete.append((offset, None))
            offset = stdout.find(OBSERVATION_MAGIC, offset + 1)
            continue
        if frame_end <= len(stdout):
            candidates.append((frame_end, offset, stdout[offset:frame_end]))
        else:
            incomplete.append((offset, frame_end))
        offset = stdout.find(OBSERVATION_MAGIC, offset + 1)
    top_level = [
        item for item in candidates
        if not any(other[1] < item[1] < other[0] for other in candidates)
    ]
    if len(top_level) != 1:
        return None
    frame_end, frame_start, frame = top_level[0]
    # A magic sequence inside the declared memory payload is ordinary guest
    # data, not a second/truncated top-level frame.  Only markers outside the
    # selected frame invalidate the observation.
    if any(
        offset < frame_start or offset >= frame_end
        for offset, _declared_end in incomplete
    ):
        return None
    if any(
        other_start < frame_start or other_end > frame_end
        for other_end, other_start, _ in candidates
        if other_start != frame_start
    ):
        return None
    return frame


def _finalized_observation_frame(stdout: bytes) -> bytes | None:
    """Return a structurally valid frame whose guest checkpoint is finalized."""

    frame = _observation_frame(stdout)
    if frame is None or len(frame) < OBSERVATION_HEADER_SIZE:
        return None
    checkpoint = struct.unpack_from("<Q", frame, 8)[0]
    x0 = struct.unpack_from("<Q", frame, 16)[0]
    if checkpoint == 0 or checkpoint & 1 or x0 != 0:
        return None
    return frame


def source_pc_map(executable: Path) -> dict[str, object] | None:
    try:
        result = subprocess.run(
            ["riscv64-linux-gnu-objdump", "-dS", "--line-numbers", str(executable)],
            capture_output=True, text=True, check=False, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines: dict[str, list[int]] = {}
    instructions: list[dict[str, object]] = []
    source_line = None
    seen: set[int] = set()
    for raw in result.stdout.splitlines():
        marker = re.fullmatch(r"\s*(?:\./)*[^:\s]+:(\d+)\s*", raw)
        if marker:
            source_line = marker[1]
            continue
        instruction = re.match(r"\s*([0-9a-f]+):\s*([0-9a-f]+)\s+", raw, re.I)
        if instruction is None:
            continue
        pc, value = int(instruction[1], 16), instruction[2].lower()
        if pc & 1:
            return None
        if pc in seen or len(value) % 2:
            return None
        seen.add(pc)
        instructions.append({"pc": pc, "size": len(value) // 2, "bytes": value})
        if source_line is not None:
            lines.setdefault(source_line, []).append(pc)
    if not instructions or not lines:
        return None
    try:
        binary_sha256 = sha256_file(executable)
    except (OSError, TypeError):
        return None
    return {
        "binary_sha256": binary_sha256,
        "lines": lines,
        "instructions": instructions,
    }


LINUX_TEST_MEMORY_ENV = "RV_LINUX_TEST_MEMORY"
LINUX_MEMORY_SIZE_ENV = "RV_LINUX_MEMORY_SIZE"
CAPSULE_MAILBOX_ENV = "RV_CAPSULE_MAILBOX"
CAPSULE_OBSERVATION_SIZE_ENV = "RV_CAPSULE_OBSERVATION_SIZE"
CAPSULE_TEST_MEMORY_ENV = "RV_CAPSULE_TEST_MEMORY"
CAPSULE_MEMORY_SIZE_ENV = "RV_CAPSULE_MEMORY_SIZE"

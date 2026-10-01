"""Small host-side ELF setup for RVVM's standard GDB RSP interface."""

from collections.abc import Callable
from pathlib import Path

from framework._util import elf_load_segments


def set_elf_entry_pc(elf: Path, profile: str, request: Callable[[str], str]) -> int:
    """Set the guest PC from ELF metadata without requiring an RVVM extension."""
    entry, _ = elf_load_segments(elf)
    width = 8 if profile.startswith("rv64") else 4 if profile.startswith("rv32") else 0
    if not width:
        raise ValueError(f"unsupported RVVM profile: {profile}")
    reply = request("g")
    try:
        registers = bytearray.fromhex(reply)
    except ValueError as exc:
        raise RuntimeError(
            f"RVVM GDB register packet is malformed: {reply[:128]!r}"
        ) from exc
    if len(registers) < 33 * width:
        raise RuntimeError("RVVM GDB register packet is incomplete")
    start = 32 * width
    registers[start : start + width] = entry.to_bytes(width, "little")
    if request("G" + registers.hex()) != "OK":
        raise RuntimeError("RVVM GDB register write rejected the ELF entry PC")
    return entry

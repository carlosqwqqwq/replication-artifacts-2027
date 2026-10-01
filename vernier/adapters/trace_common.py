"""QEMU/libriscv trace runner 共用辅助函数。"""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from framework._util import canonical_digest, execute_process, pc_digest, sha256_file
from framework.adapters.contracts import PATH_IDENTITY_WITNESS_CONTRACT
from framework.adapters.runner import _observation_frame
from framework.direct_case import OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC
from framework.execution_identity import TargetBinaryIdentity


_execute = execute_process


def _repeated_pc_cycle(
    pcs: Sequence[int], *, repeats: int = 32, max_period: int = 64,
) -> int | None:
    """Return the first PC of a short cycle repeated in the trace tail.

    A stuck guest need not execute one PC repeatedly: trap/dispatch paths often
    rotate through a small block.  The detector is deliberately restricted to
    a short tail and a bounded period so it is useful as a live watchdog while
    avoiding a general-purpose path analysis.
    """
    if repeats < 1 or max_period < 1 or len(pcs) < repeats:
        return None
    max_period = min(max_period, len(pcs) // repeats)
    for period in range(1, max_period + 1):
        window = tuple(pcs[-period * repeats:])
        cycle = window[:period]
        if all(
            window[offset:offset + period] == cycle
            for offset in range(0, len(window), period)
        ):
            return cycle[0]
    return None


def _clip_trace_at_checkpoint(
    pcs: tuple[int, ...], stdout: bytes,
) -> tuple[tuple[int, ...], str | None]:
    frame = _observation_frame(stdout) or _observation_frame_prefix(stdout)
    if frame is None:
        return pcs, None
    checkpoint = int.from_bytes(frame[8:16], "little")
    if checkpoint == 0:
        return (), "RVOBS1 checkpoint is not finalized"
    if checkpoint & 1:
        return (), "RVOBS1 checkpoint is unaligned"
    if checkpoint not in pcs:
        return (), "RVOBS1 checkpoint is absent from execution trace"
    return pcs[: len(pcs) - pcs[::-1].index(checkpoint)], None


def _observation_frame_prefix(stdout: bytes) -> bytes | None:
    offset = stdout.find(OBSERVATION_MAGIC)
    if offset < 0 or len(stdout) < offset + OBSERVATION_HEADER_SIZE:
        return None
    memory_size = int.from_bytes(
        stdout[offset + 16 + 32 * 8:offset + OBSERVATION_HEADER_SIZE], "little"
    )
    end = offset + OBSERVATION_HEADER_SIZE + memory_size
    if len(stdout) < end or stdout.find(OBSERVATION_MAGIC, end) >= 0:
        return None
    return stdout[offset:end]


def _path_identity_alias(
    pcs: tuple[int, ...],
    trace_details: dict,
    *,
    require_instruction_trace: bool = False,
) -> dict[str, object]:
    executed_digest = pc_digest(pcs) if pcs else None
    observed = bool(pcs) and (
        not require_instruction_trace
        or trace_details.get("granularity") == "per-instruction-exec"
    )
    return {
        "contract": PATH_IDENTITY_WITNESS_CONTRACT,
        "status": "observed" if observed else "unavailable",
        "digest": executed_digest if observed else None,
        "evidence": trace_details.get("granularity", "missing"),
        "executed_pc": {
            "observed": observed,
            "tested_pc_seen": False,
            "count": len(pcs),
            "digest": executed_digest if observed else None,
        },
        "executed_pcs": [hex(pc) for pc in pcs],
    }


def _binary_tool_version(path: str) -> str | None:
    try:
        process = subprocess.run(
            [path, "--version"], timeout=2, capture_output=True, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    for output in (process.stdout, process.stderr):
        for line in output.decode("utf-8", errors="replace").splitlines():
            version = line.strip()
            if version:
                return version
    return None


def _target_binary_details(path: str, *, expected_backend: str) -> dict[str, object]:
    binary = Path(path)
    try:
        binary_sha256 = sha256_file(binary)
    except OSError:
        binary_sha256 = None
    details: dict[str, object] = {
        "target_binary_path": str(binary),
        "target_binary_sha256": binary_sha256,
        "target_identity_status": "missing",
        "target_identity": None,
        "target_identity_digest": None,
    }
    if binary_sha256 is None:
        return details
    manifest_path = binary.resolve().with_name("target-identity.json")
    if not manifest_path.is_file():
        return details
    try:
        identity = TargetBinaryIdentity.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        details["target_identity_status"] = "invalid"
        return details
    if identity.backend != expected_backend or identity.binary_sha256 != binary_sha256:
        details["target_identity_status"] = "mismatch"
        return details
    if not identity.is_build_attested:
        return details
    details.update({
        "target_identity_status": "verified",
        "target_identity_path": str(manifest_path),
        "target_identity": identity.to_dict(),
        "target_identity_digest": identity.identity_digest,
        "target_source_commit": identity.source_commit,
    })
    return details


def _target_configuration(
    values: dict[str, object], target_details: dict[str, object],
) -> dict[str, object]:
    configuration = {
        **values,
        **{
            key: target_details.get(key)
            for key in (
                "target_binary_sha256", "target_identity_status", "target_identity",
                "target_identity_digest",
            )
        },
    }
    configuration["identity_digest"] = canonical_digest(configuration)
    return configuration

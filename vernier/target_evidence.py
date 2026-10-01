"""Durable Target evidence references used by memory-bounded campaigns.

The online campaign only needs the Target verdict and a small amount of
qualification metadata.  The complete observation (including raw simulator
output and executed PC paths) is written to the replay-result sidecar first.
This module keeps the transition compatible with older inline records: old
records are returned unchanged, while new records are materialized lazily from
their sidecar reference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator


TARGET_EVIDENCE_REF_SCHEMA = "rq1-target-evidence-ref-v1"
TARGET_REPLAY_RESULT_SCHEMA = "rq1-target-replay-result-v1"


def file_digest(path: Path) -> tuple[str, int]:
    """Return a streaming SHA256 and byte count without loading a large file."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def make_evidence_ref(
    root: Path, path: Path, *, target_id: str, pair_index: int | None,
) -> dict[str, object]:
    """Build an integrity-checked reference relative to one case root."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("target evidence path is outside case root") from error
    digest, size = file_digest(path)
    return {
        "schema": TARGET_EVIDENCE_REF_SCHEMA,
        "path": str(relative),
        "sha256": digest,
        "bytes": size,
        "target_id": target_id,
        "pair_index": pair_index,
    }


def _compact_comparison(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    # Keep every comparison/qualification field.  Only the two observation
    # arrays contain the large simulator payload; the full value remains in the
    # sidecar and is restored by materialize_record().
    return {
        str(key): item
        for key, item in value.items()
        if key != "observations"
    }


def compact_record(
    record: dict[str, Any], reference: Mapping[str, object],
) -> dict[str, Any]:
    """Replace heavy inline observation payloads with a durable reference."""
    compact = record.get("comparison")
    if isinstance(compact, Mapping):
        record["comparison"] = _compact_comparison(compact)
    # Baseline records keep observations at the top level; target pair records
    # keep them under comparison.  Both forms use the same sidecar contract.
    record.pop("observations", None)
    record["target_evidence_ref"] = dict(reference)
    return record


def _safe_reference_path(case_root: Path, reference: Mapping[str, object]) -> Path:
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("target evidence reference path is missing")
    root = Path(case_root).resolve()
    path = (root / path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("target evidence reference escapes case root") from error
    return path


def _read_sidecar(case_root: Path, reference: Mapping[str, object]) -> dict[str, Any]:
    path = _safe_reference_path(case_root, reference)
    expected_bytes = reference.get("bytes")
    if type(expected_bytes) is int and expected_bytes >= 0:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError("target evidence byte count mismatch")
    expected_sha256 = reference.get("sha256")
    if isinstance(expected_sha256, str) and expected_sha256:
        actual_sha256, _ = file_digest(path)
        if actual_sha256 != expected_sha256:
            raise ValueError("target evidence digest mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("target evidence sidecar must be an object")
    if payload.get("schema_version") != TARGET_REPLAY_RESULT_SCHEMA:
        raise ValueError("target evidence sidecar schema mismatch")
    expected_target = reference.get("target_id")
    if expected_target not in (None, "") and payload.get("target_id") != expected_target:
        raise ValueError("target evidence target id mismatch")
    expected_pair = reference.get("pair_index")
    if payload.get("pair_index") != expected_pair:
        raise ValueError("target evidence pair index mismatch")
    record = payload.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("target evidence record is missing")
    return dict(record)


def materialize_record(case_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a full record, accepting both old inline and new compact forms."""
    reference = record.get("target_evidence_ref")
    if not isinstance(reference, Mapping):
        return record if isinstance(record, dict) else dict(record)
    full = _read_sidecar(Path(case_root), reference)
    # Preserve chain-time fields (step, status, deadline disposition, and
    # counters) that are updated after the sidecar is written.
    for key, value in record.items():
        if key not in {"comparison", "observations"}:
            full[key] = value
    compact_comparison = record.get("comparison")
    full_comparison = full.get("comparison")
    if isinstance(full_comparison, Mapping) and isinstance(compact_comparison, Mapping):
        merged = dict(full_comparison)
        merged.update(compact_comparison)
        full["comparison"] = merged
    elif isinstance(compact_comparison, Mapping):
        full["comparison"] = dict(compact_comparison)
    return full


def iter_materialized_records(
    case_root: Path, records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    """Materialize records one at a time to bound coverage-finalization RSS."""
    if isinstance(records, Mapping):
        values = (
            item for group in records.values()
            for item in (group if isinstance(group, (list, tuple)) else (group,))
        )
    elif isinstance(records, (list, tuple)):
        values = iter(records)
    else:
        values = iter(())
    for item in values:
        if isinstance(item, Mapping):
            yield materialize_record(case_root, item)


__all__ = [
    "TARGET_EVIDENCE_REF_SCHEMA", "TARGET_REPLAY_RESULT_SCHEMA",
    "compact_record", "file_digest", "iter_materialized_records",
    "make_evidence_ref", "materialize_record",
]

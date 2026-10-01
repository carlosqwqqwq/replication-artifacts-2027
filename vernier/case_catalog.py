"""Append-only index for validated, read-only external case corpora."""

from __future__ import annotations

import fcntl
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from comparison_artifacts import digest, load_json, sha256
from comparison_queue import (
    _applicable_targets, _reusable_dependency_identity,
)
from comparison_state import (
    _CANDIDATE_PATH_FIELDS, _candidate_identity_key, _refresh_generation_metrics,
)


_CATALOG_SCHEMA = "rq1-case-catalog-v1"
_BATCH_SCHEMA = "rq1-case-catalog-batch-v1"


def _self_digest(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return digest(payload)


def _read(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lock(catalog_root: Path) -> Iterator[None]:
    catalog_root.mkdir(parents=True, exist_ok=True)
    with (catalog_root / ".catalog.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _ids(config: dict[str, Any], section: str) -> list[str]:
    return [
        str(row["id"]) for row in config.get(section, [])
        if isinstance(row, dict) and row.get("id")
    ]


def _execution_identity(execution: dict[str, Any]) -> dict[str, Any]:
    targets = execution.get("targets")
    if not isinstance(targets, list):
        targets = []
    return {
        "config_sha256": execution.get("config_sha256"),
        "image_id": execution.get("image_id"),
        "execution_plane": execution.get("execution_plane"),
        "network": execution.get("network"),
        "dependency_identity": _reusable_dependency_identity(
            execution.get("dependency_identity"),
        ),
        "dependency_identity_ok": execution.get("dependency_identity_ok"),
        "targets": [
            {key: row.get(key) for key in (
                "id", "kind", "execution_model", "translation_mode", "commit",
                "binary_sha256", "identity_digest", "coverage_binary_sha256",
            )}
            for row in targets if isinstance(row, dict)
        ],
    }


def _read_raw_queues(run_root: Path, targets: list[str]) -> dict[str, list[dict[str, Any]]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        try:
            document = _read(run_root / "queues" / f"{target}.json")
        except (OSError, TypeError, ValueError):
            document = {}
        entries = document.get("entries") if isinstance(document, dict) else None
        queues[target] = entries if isinstance(entries, list) else []
    return queues


def _tree_sha256(root: Path) -> str:
    process = subprocess.Popen(
        [
            "tar", "--sort=name", "--mtime=@0", "--mode=a+rwX", "--owner=0",
            "--group=0", "--numeric-owner", "-cf", "-", "-C", str(root), ".",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise OSError("could not read source snapshot archive")
    digest_value = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest_value.update(chunk)
    process.stdout.close()
    error = process.stderr.read()
    process.stderr.close()
    status = process.wait()
    if status:
        raise ValueError(f"could not hash source snapshot: {error.decode(errors='replace')}")
    return digest_value.hexdigest()


def _corpus_manifests(root: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    copy_path = root / "corpus-manifest.json"
    batch_path = root / "generation-batch-manifest.json"
    copied, batch = _read(copy_path), _read(batch_path)
    copy_sha, batch_sha = sha256(copy_path), sha256(batch_path)
    return copied, batch, str(copy_sha), str(batch_sha)


def _inspect_corpus(
    corpus_root: Path,
    artifact_root: Path,
    config: dict[str, Any],
    selected_methods: list[str] | None = None,
) -> dict[str, Any]:
    root = corpus_root.resolve(strict=True)
    artifact_root = artifact_root.resolve(strict=True)
    try:
        relative_root = root.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ValueError("corpus root must be below catalog_root.parent") from exc
    copied, batch, copy_sha, batch_sha = _corpus_manifests(root)
    validation = copied.get("content_validation", {})
    copied_methods = validation.get("methods", {}) if isinstance(validation, dict) else {}
    copied_snapshots = validation.get("source_snapshots", {}) if isinstance(validation, dict) else {}
    if not isinstance(copied_methods, dict):
        copied_methods = {}
    batch_methods = {
        row.get("method"): row for row in batch.get("methods", [])
        if isinstance(row, dict) and row.get("method")
    }
    methods = selected_methods if selected_methods is not None else list(copied_methods or batch_methods)
    methods = [method for method in methods if method in copied_methods or method in batch_methods]
    targets = _ids(config, "targets")
    target_identity: list[dict[str, Any]] | None = None
    inspected: dict[str, Any] = {}

    def optional_json(path: Path) -> dict[str, Any]:
        try:
            value = _read(path)
        except (OSError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    for method in methods:
        copied_row = copied_methods.get(method, {})
        batch_row = batch_methods.get(method, {})
        if not isinstance(copied_row, dict):
            copied_row = {}
        if not isinstance(batch_row, dict):
            batch_row = {}
        run_id = batch_row.get("run_id") or copied_row.get("run_id")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            continue
        run_root = root / "runs" / run_id
        snapshot_row = copied_snapshots.get(method) if isinstance(copied_snapshots, dict) else None
        config_path = run_root / "config.snapshot.json"
        config_sha = sha256(config_path) if config_path.is_file() else ""
        execution_path = run_root / "execution-manifest.json"
        execution = optional_json(execution_path)
        identity = _execution_identity(execution)
        if target_identity is None:
            target_identity = identity["targets"]
        queues: dict[str, list[dict[str, Any]]] = {}
        source_sha_by_index: dict[int, str] = {}
        queue_hashes: dict[str, str] = {}
        for target in targets:
            queue_path = run_root / "queues" / f"{target}.json"
            queue_doc = optional_json(queue_path)
            entries = queue_doc.get("entries", [])
            entries = entries if isinstance(entries, list) else []
            queues[target] = entries
            queue_hashes[target] = sha256(queue_path) if queue_path.is_file() else ""
            for offset, entry in enumerate(entries):
                candidate = entry.get("candidate") if isinstance(entry, dict) else None
                candidate = candidate if isinstance(candidate, dict) else {}
                local_index = candidate.get("index", offset)
                if type(local_index) is int and local_index >= 0:
                    source_sha_by_index.setdefault(
                        local_index, str(candidate.get("source_sha256") or ""),
                    )

        result = optional_json(run_root / "run-result.json") or optional_json(
            run_root / "generation-result.json",
        )
        result_rows = result.get("methods", [])
        method_row = next((row for row in result_rows
                           if isinstance(row, dict) and row.get("method") == method), {})
        if not isinstance(method_row, dict):
            method_row = {}
        status = copied_row.get("generation_status") or batch_row.get("generation_status") \
            or result.get("status") or "partial"

        inspected[method] = {
            "queues": queues,
            "row": method_row or {"method": method},
            "source": {
                "status": status,
                "run_id": run_id,
                "dependency_identity_ok": execution.get("dependency_identity_ok"),
                "config_sha256": config_sha,
            },
            "run_id": run_id,
            "config_sha256": str(config_sha),
            "source_snapshot_sha256": (
                snapshot_row.get("sha256") if isinstance(snapshot_row, dict)
                else batch_row.get("source_snapshot_sha256", "")
            ),
            "execution_manifest_sha256": str(sha256(execution_path))
            if execution_path.is_file() else "",
            "execution_identity": identity,
            "execution_identity_sha256": digest(identity),
            "queue_sha256": queue_hashes,
            "generation_status": status,
            "cases": [
                {"local_index": index, "source_sha256": source_sha_by_index[index]}
                for index in sorted(source_sha_by_index)
            ],
        }
    return {
        "corpus_id": copied["corpus_id"],
        "corpus_root": relative_root,
        "corpus_manifest_sha256": copy_sha,
        "generation_batch_manifest_sha256": batch_sha,
        "original_batch_status": batch.get("status"),
        "declared_methods": list(batch_methods),
        "target_identity_sha256": digest(target_identity or []),
        "methods": inspected,
    }


def _load_catalog(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest = _read(manifest_path or root / "catalog-manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("catalog manifest must be an object")
    if not isinstance(manifest.get("methods"), dict):
        manifest["methods"] = {}
    if not isinstance(manifest.get("batches"), list):
        manifest["batches"] = []
    return manifest


def register_corpus(
    catalog_root: str | Path,
    corpus_root: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append a validated corpus by reference; its case files are never copied."""
    catalog_root = Path(catalog_root).resolve()
    artifact_root = catalog_root.parent.resolve(strict=True)
    inspected = _inspect_corpus(Path(corpus_root), artifact_root, config)
    return _register_inspected(catalog_root, inspected, config)


def _register_inspected(
    catalog_root: Path, inspected: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    first_method = next(iter(inspected["methods"].values()), {})
    config_sha = first_method.get("config_sha256", "")
    config_digest = digest(config)

    with _lock(catalog_root):
        path = catalog_root / "catalog-manifest.json"
        if path.exists():
            manifest = _load_catalog(catalog_root)
            manifest.pop("compatible_config_digest", None)
            manifest["target_identity_sha256"] = inspected["target_identity_sha256"]
            for method in inspected["declared_methods"]:
                manifest["methods"].setdefault(method, {
                    "case_count": 0,
                    "next_case_number": 1,
                    "execution_identity_sha256": None,
                    "execution_identity": None,
                })
        else:
            manifest = {
                "schema_version": _CATALOG_SCHEMA,
                "catalog_id": catalog_root.name,
                "artifact_root": "..",
                "revision": 0,
                "config_sha256": config_sha,
                "config_digest": config_digest,
                "target_identity_sha256": inspected["target_identity_sha256"],
                "methods": {
                    method: {
                        "case_count": 0,
                        "next_case_number": 1,
                        "execution_identity_sha256": (
                            inspected["methods"][method]["execution_identity_sha256"]
                            if method in inspected["methods"] else None
                        ),
                        "execution_identity": (
                            inspected["methods"][method]["execution_identity"]
                            if method in inspected["methods"] else None
                        ),
                    }
                    for method in inspected["declared_methods"]
                },
                "batches": [],
            }

        methods_index: dict[str, Any] = {}
        for method, row in inspected["methods"].items():
            record = manifest["methods"].setdefault(method, {
                "case_count": 0,
                "next_case_number": 1,
                "execution_identity_sha256": None,
                "execution_identity": None,
            })
            if record.get("execution_identity_sha256") is None:
                record["execution_identity_sha256"] = row["execution_identity_sha256"]
                record["execution_identity"] = row["execution_identity"]
            first = record["next_case_number"]
            methods_index[method] = {
                "run_id": row["run_id"],
                "config_sha256": row["config_sha256"],
                "source_snapshot_sha256": row["source_snapshot_sha256"],
                "execution_manifest_sha256": row["execution_manifest_sha256"],
                "execution_identity_sha256": row["execution_identity_sha256"],
                "generation_status": row["generation_status"],
                "queue_sha256": row["queue_sha256"],
                "cases": [
                    {
                        "case_number": first + i,
                        "case_id": f"{method}-{first + i:08d}",
                        "local_index": case["local_index"],
                        "source_sha256": case["source_sha256"],
                    }
                    for i, case in enumerate(row["cases"])
                ],
            }

        sequence = manifest.get("revision", 0)
        if type(sequence) is not int:
            sequence = 0
        sequence += 1
        index_root = catalog_root / "batches"
        filename = f"{sequence:06d}-{inspected['corpus_id']}.json"
        suffix = 1
        while (index_root / filename).exists():
            filename = f"{sequence:06d}-{inspected['corpus_id']}-{suffix}.json"
            suffix += 1
        batch_index = {
            "schema_version": _BATCH_SCHEMA,
            "sequence": sequence,
            "batch_id": inspected["corpus_id"],
            "corpus_root": inspected["corpus_root"],
            "corpus_manifest_sha256": inspected["corpus_manifest_sha256"],
            "generation_batch_manifest_sha256": inspected["generation_batch_manifest_sha256"],
            "original_batch_status": inspected["original_batch_status"],
            "target_identity_sha256": inspected["target_identity_sha256"],
            "methods": methods_index,
        }
        batch_index["index_sha256"] = _self_digest(batch_index, "index_sha256")
        index_path = catalog_root / "batches" / filename
        _atomic_json(index_path, batch_index)
        os.chmod(index_path, 0o444)
        index_sha = sha256(index_path)
        manifest["batches"].append({
            "sequence": sequence,
            "batch_id": inspected["corpus_id"],
            "index_path": f"batches/{filename}",
            "index_sha256": index_sha,
            "corpus_manifest_sha256": inspected["corpus_manifest_sha256"],
            "generation_batch_manifest_sha256": inspected["generation_batch_manifest_sha256"],
        })
        manifest["revision"] = sequence
        for method, row in inspected["methods"].items():
            count = len(row["cases"])
            record = manifest["methods"][method]
            record["case_count"] = int(record.get("case_count", 0) or 0) + count
            record["next_case_number"] = int(record.get("next_case_number", 1) or 1) + count
        manifest.pop("catalog_sha256", None)
        manifest["catalog_sha256"] = _self_digest(manifest, "catalog_sha256")
        _atomic_json(path, manifest)
        return manifest


def _merge_method_rows(
    rows: dict[str, dict[str, Any]], queues: dict[str, list[dict[str, Any]]],
) -> None:
    for method, row in rows.items():
        batches = row.pop("_batch_rows")
        generation_statuses = row.pop("_generation_statuses")
        original_statuses = row.pop("_original_statuses")
        totals = {
            key: sum(item.get(key, 0) for item in batches
                     if isinstance(item.get(key), (int, float))
                     and not isinstance(item.get(key), bool))
            for key in {key for item in batches for key in item
                        if key.endswith("_count") or key in {
                            "candidate_count", "accepted_count", "accepted_candidate_count",
                            "candidate_enqueued_count", "elapsed_s", "method_elapsed_s",
                        }}
        }
        row.update(totals)
        count = len({entry["case_id"] for entries in queues.values()
                     for entry in entries if entry.get("method") == method})
        row.update({
            "candidate_count": count,
            "accepted_candidate_count": count,
            "candidate_enqueued_count": count,
            "target_queue_entry_count": sum(
                1 for entries in queues.values() for entry in entries
                if entry.get("method") == method
            ),
            "batch_count": len(batches),
            "batch_statuses": [item.get("status") for item in batches],
            "generation_batch_statuses": generation_statuses,
            "original_batch_statuses": original_statuses,
            "status": "gap" if not batches else "passed"
            if all(item.get("status") == "passed" for item in batches)
            and all(status == "passed" for status in generation_statuses) else "partial",
            "generated": any(item.get("generated") is True for item in batches),
            "built": any(item.get("built") is True for item in batches),
        })
        _refresh_generation_metrics(row)


def load_catalog_queues(
    catalog_root: str | Path,
    config: dict[str, Any],
    method_filter: str | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any], Path]:
    """Read available catalog queues and attach stable case labels where present."""
    catalog_root = Path(catalog_root).resolve(strict=True)
    artifact_root = catalog_root.parent.resolve(strict=True)
    pinned_manifest = Path(manifest_path).resolve() if manifest_path is not None else None
    manifest = _load_catalog(catalog_root, pinned_manifest)
    methods = [method_filter] if method_filter else list(manifest["methods"])
    targets = _ids(config, "targets")
    queues: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    batches_summary: list[dict[str, Any]] = []
    next_numbers = {method: 1 for method in methods}

    def optional_json(path: Path) -> dict[str, Any]:
        try:
            value = _read(path)
        except (OSError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    for batch in manifest["batches"]:
        if not isinstance(batch, dict) or not isinstance(batch.get("index_path"), str):
            continue
        index_path = (catalog_root / batch["index_path"]).resolve()
        if not index_path.is_relative_to(catalog_root) or not index_path.is_file():
            continue
        index = optional_json(index_path)
        index_methods = index.get("methods")
        if not isinstance(index_methods, dict):
            index_methods = {}
        present_methods = [method for method in methods if method in index_methods]
        if not present_methods:
            batches_summary.append({
                "batch_id": index.get("batch_id", batch.get("batch_id")),
                "original_batch_status": index.get("original_batch_status"),
                "corpus_manifest_sha256": index.get("corpus_manifest_sha256"),
                "generation_batch_manifest_sha256": index.get("generation_batch_manifest_sha256"),
                "method_count": 0,
            })
            continue
        corpus_rel = index.get("corpus_root")
        if not isinstance(corpus_rel, str):
            continue
        corpus_root = (artifact_root / corpus_rel).resolve()
        if not corpus_root.is_relative_to(artifact_root):
            continue
        for method in present_methods:
            indexed = index_methods.get(method)
            if not isinstance(indexed, dict):
                continue
            run_id = indexed.get("run_id")
            if not isinstance(run_id, str) or Path(run_id).name != run_id:
                continue
            run_root = (corpus_root / "runs" / run_id).resolve()
            if not run_root.is_relative_to(corpus_root):
                continue
            run_result = optional_json(run_root / "run-result.json")
            generation_result = optional_json(run_root / "generation-result.json")
            execution = optional_json(run_root / "execution-manifest.json")
            indexed_cases = indexed.get("cases")
            if not isinstance(indexed_cases, list):
                indexed_cases = []
            case_by_local: dict[int, dict[str, Any]] = {}
            fallback_cases: dict[int, dict[str, Any]] = {}
            for offset, case in enumerate(indexed_cases):
                if not isinstance(case, dict):
                    continue
                local_index = case.get("local_index", offset)
                if type(local_index) is not int or local_index < 0:
                    local_index = offset
                number = case.get("case_number")
                if type(number) is not int or number < 1:
                    number = next_numbers[method] + offset
                case_by_local[local_index] = {
                    "case_number": number,
                    "case_id": case.get("case_id") or f"{method}-{number:08d}",
                    "local_index": local_index,
                    "source_sha256": case.get("source_sha256"),
                }
                next_numbers[method] = max(next_numbers[method], number + 1)
            for target in targets:
                queue_path = run_root / "queues" / f"{target}.json"
                queue_doc = optional_json(queue_path)
                raw_entries = queue_doc.get("entries")
                if not isinstance(raw_entries, list):
                    continue
                for offset, raw in enumerate(raw_entries):
                    if not isinstance(raw, dict):
                        continue
                    if raw.get("method") not in (None, method):
                        continue
                    entry = dict(raw)
                    candidate = raw.get("candidate")
                    candidate = dict(candidate) if isinstance(candidate, dict) else {}
                    local_index = candidate.get("index")
                    if type(local_index) is not int or local_index < 0:
                        local_index = offset
                    case = case_by_local.get(local_index)
                    if case is None:
                        case = fallback_cases.get(local_index)
                    if case is None:
                        number = next_numbers[method] + len(fallback_cases)
                        case = {
                            "case_number": number,
                            "case_id": f"{method}-{number:08d}",
                            "local_index": local_index,
                            "source_sha256": candidate.get("source_sha256"),
                        }
                        fallback_cases[local_index] = case
                    number = case["case_number"]
                    tool_root = (run_root / str(raw.get("tool_root", ""))).resolve()
                    entry["tool_root"] = str(tool_root)
                    identity = {
                        "batch_id": index.get("batch_id", batch.get("batch_id")),
                        "batch_candidate_index": local_index,
                        "case_number": number,
                        "case_id": case["case_id"],
                    }
                    candidate.update(identity, index=number - 1)
                    entry.update(identity, candidate=candidate)
                    queues[target].append(entry)
            if fallback_cases:
                next_numbers[method] = max(
                    next_numbers[method],
                    max(case["case_number"] for case in fallback_cases.values()) + 1,
                )
            result_doc = run_result if run_result else generation_result
            method_row = next((row for row in result_doc.get("methods", [])
                               if isinstance(row, dict) and row.get("method") == method), None)
            status = indexed.get("generation_status") or (
                method_row.get("status") if isinstance(method_row, dict) else None
            ) or result_doc.get("status") or "partial"
            if not isinstance(method_row, dict):
                count = len(case_by_local) + len(fallback_cases)
                method_row = {
                    "method": method,
                    "status": status,
                    "candidate_count": count,
                    "accepted_candidate_count": count,
                    "candidate_enqueued_count": count,
                    "target_queue_entry_count": count * len(targets),
                }
            if method not in rows:
                rows[method] = dict(method_row)
                rows[method]["method"] = method
                rows[method]["_batch_rows"] = []
                rows[method]["_generation_statuses"] = []
                rows[method]["_original_statuses"] = []
            rows[method]["_batch_rows"].append(dict(method_row))
            rows[method]["_generation_statuses"].append(status)
            rows[method]["_original_statuses"].append(index.get("original_batch_status"))
            sources[method].append({
                "status": status,
                "run_id": run_id,
                "batch_id": index.get("batch_id", batch.get("batch_id")),
                "config_sha256": indexed.get("config_sha256"),
                "source_snapshot_sha256": indexed.get("source_snapshot_sha256"),
                "execution_manifest_sha256": indexed.get("execution_manifest_sha256"),
                "dependency_identity": execution.get("dependency_identity"),
                "dependency_identity_ok": execution.get("dependency_identity_ok"),
                "dependency_root": execution.get("dependency_root"),
                "image_id": execution.get("image_id"),
                "execution_plane": execution.get("execution_plane"),
                "network": execution.get("network"),
                "targets": execution.get("targets"),
            })
        batches_summary.append({
            "batch_id": index.get("batch_id", batch.get("batch_id")),
            "original_batch_status": index.get("original_batch_status"),
            "corpus_manifest_sha256": index.get("corpus_manifest_sha256"),
            "generation_batch_manifest_sha256": index.get("generation_batch_manifest_sha256"),
            "method_count": len(present_methods),
        })

    for method in methods:
        if method not in rows:
            rows[method] = {
                "method": method,
                "status": "gap",
                "reason": "no-case-batch",
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "candidate_enqueued_count": 0,
                "target_queue_entry_count": 0,
                "_batch_rows": [],
                "_generation_statuses": [],
                "_original_statuses": [],
            }
    _merge_method_rows(rows, queues)
    statuses = [source.get("status") for values in sources.values() for source in values]
    generation_source: dict[str, Any] = {
        "schema_version": _CATALOG_SCHEMA,
        "run_id": manifest.get("catalog_id", catalog_root.name),
        "root": str(catalog_root),
        "catalog_revision": manifest.get("revision", 0),
        "catalog_sha256": manifest.get("catalog_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "target_identity_sha256": manifest.get("target_identity_sha256"),
        "method_filter": method_filter,
        "status": "passed" if statuses and all(status == "passed" for status in statuses) else "partial",
        "dependency_identity_ok": all(
            source.get("dependency_identity_ok") is True
            for values in sources.values() for source in values
        ),
        "batches": batches_summary,
        "method_execution_sources": sources,
    }
    if len(methods) == 1 and sources[methods[0]]:
        for key in (
            "source_commit", "source_end_commit", "source_dirty", "source_end_dirty",
            "config_snapshot", "dependency_identity", "dependency_identity_end",
            "dependency_root", "image_id", "execution_plane", "network", "targets",
        ):
            if key in sources[methods[0]][0]:
                generation_source[key] = sources[methods[0]][0][key]
    return queues, rows, generation_source, artifact_root


def _batch_config(batch_path: Path, batch: dict[str, Any], config_path: str | Path | None) -> dict[str, Any]:
    if config_path is not None:
        return _read(Path(config_path).resolve(strict=True))
    base = batch_path.parent.parent
    for row in batch.get("methods", []):
        if not isinstance(row, dict) or not row.get("run_root"):
            continue
        run_root = Path(row["run_root"])
        if not run_root.is_absolute():
            run_root = base / run_root
        config = run_root / "config.snapshot.json"
        if config.is_file():
            return _read(config)
    raise ValueError("cannot find a method config snapshot in the generation batch")


def _batch_path(value: object, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("generation batch is missing a source path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def import_generation_batch(
    catalog_root: str | Path,
    batch_manifest_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy available method queues from a generation batch into the catalog."""
    catalog_root = Path(catalog_root).resolve()
    batch_path = Path(batch_manifest_path).resolve(strict=True)
    batch = _read(batch_path)
    if batch.get("schema_version") != "rq1-generation-batch-v1":
        raise ValueError("unsupported generation batch manifest")
    config = _batch_config(batch_path, batch, config_path)
    batch_id = batch_path.parent.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id) or batch_id in {".", ".."}:
        raise ValueError("generation batch directory name is not a valid batch id")
    artifact_root = catalog_root.parent.resolve(strict=True)
    final_root = catalog_root / "batches" / batch_id
    staging_root = catalog_root / "batches" / ".staging" / batch_id
    if any(path.is_symlink() for path in (final_root.parent, staging_root.parent)):
        raise ValueError("catalog batch and staging directories must not be symlinks")
    def optional_json(path: Path) -> dict[str, Any]:
        try:
            value = _read(path)
        except (OSError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def inspect_copy(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"import copy is not a real directory: {root}")
        copied, _, _, _ = _corpus_manifests(root)
        return copied, _inspect_corpus(root, artifact_root, config)

    def result_for(root: Path, copied: dict[str, Any], inspected: dict[str, Any]) -> dict[str, Any]:
        inspected["corpus_root"] = root.relative_to(artifact_root).as_posix()
        registered = _register_inspected(catalog_root, inspected, config)
        return {
            "catalog_root": str(catalog_root),
            "corpus_root": str(root),
            "catalog_revision": registered["revision"],
            "catalog_sha256": registered["catalog_sha256"],
            "registered_methods": {
                method: len(row["cases"]) for method, row in inspected["methods"].items()
            },
            "excluded_methods": copied.get("excluded_methods", {}),
            "original_batch_status": copied.get("original_batch_status"),
        }

    if final_root.exists():
        copied, inspected = inspect_copy(final_root)
        if (catalog_root / "catalog-manifest.json").is_file():
            manifest = _load_catalog(catalog_root)
            for entry in manifest.get("batches", []):
                if not isinstance(entry, dict) or entry.get("batch_id") != batch_id:
                    continue
                index_path = (catalog_root / str(entry.get("index_path") or "")).resolve()
                if not index_path.is_relative_to(catalog_root):
                    continue
                index = optional_json(index_path)
                return {
                    "catalog_root": str(catalog_root),
                    "corpus_root": str(final_root),
                    "catalog_revision": manifest.get("revision"),
                    "catalog_sha256": manifest.get("catalog_sha256"),
                    "registered_methods": {
                        method: len(row.get("cases", []))
                        for method, row in index.get("methods", {}).items()
                        if isinstance(row, dict)
                    },
                    "excluded_methods": copied.get("excluded_methods", {}),
                    "original_batch_status": copied.get("original_batch_status"),
                    "already_registered": True,
                }
        subprocess.run(["chmod", "-R", "a-w", "--", str(final_root)], check=True)
        return result_for(final_root, copied, inspected)

    base = batch_path.parent.parent
    targets = _ids(config, "targets")
    config_methods = set(_ids(config, "methods"))
    batch_methods = batch.get("methods")
    if not isinstance(batch_methods, list):
        raise ValueError("generation batch has no method records")
    seen_methods: set[str] = set()
    for row in batch_methods:
        if isinstance(row, dict) and isinstance(row.get("method"), str):
            if row["method"] in seen_methods:
                raise ValueError(f"duplicate generation method record: {row['method']}")
            seen_methods.add(row["method"])
    usable: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    for row in batch_methods:
        if not isinstance(row, dict) or not isinstance(row.get("method"), str):
            continue
        method = row["method"]
        try:
            if method not in config_methods:
                continue
            run_root = _batch_path(row.get("run_root"), base)
            execution = optional_json(run_root / "execution-manifest.json")
            result = optional_json(run_root / "generation-result.json")
            config_file = run_root / "config.snapshot.json"
            run_config = optional_json(config_file) or config
            snapshot = None
            try:
                snapshot = _batch_path(row.get("source_snapshot"), base)
            except (OSError, TypeError, ValueError):
                pass
            queues = _read_raw_queues(run_root, targets)
            method_rows = result.get("methods", [])
            method_row = next((item for item in method_rows
                               if isinstance(item, dict) and item.get("method") == method), {})
            if not isinstance(method_row, dict):
                method_row = {}
            source = {
                "status": row.get("generation_status") or result.get("status") or "partial",
                "dependency_identity_ok": execution.get("dependency_identity_ok"),
            }
            usable[method] = {
                "run_root": run_root,
                "snapshot": snapshot,
                "snapshot_sha256": (
                    _tree_sha256(snapshot) if snapshot is not None else ""
                ),
                "config": run_config,
                "config_sha256": sha256(config_file) if config_file.is_file() else "",
                "queues": queues,
                "method_row": method_row,
                "batch_row": row,
                "source": source,
            }
        except (OSError, TypeError, ValueError) as exc:
            excluded[method] = str(exc)[:300]
    if not usable:
        usable = {}

    if staging_root.exists():
        try:
            copied, inspected = inspect_copy(staging_root)
        except (OSError, TypeError, ValueError):
            if staging_root.is_symlink() or staging_root.parent.is_symlink():
                raise ValueError(f"refusing to remove staging symlink: {staging_root}")
            subprocess.run(["chmod", "-R", "u+rwX", "--", str(staging_root)], check=True)
            shutil.rmtree(staging_root)
        else:
            subprocess.run(["chmod", "u+w", "--", str(staging_root)], check=True)
            os.rename(staging_root, final_root)
            subprocess.run(["chmod", "-R", "a-w", "--", str(final_root)], check=True)
            return result_for(final_root, copied, inspected)

    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    (staging_root / "runs").mkdir()
    (staging_root / "source-snapshots").mkdir()
    subprocess.run(["cp", "-a", "--reflink=auto", str(batch_path),
                    str(staging_root / "generation-batch-manifest.json")], check=True)
    for method, row in usable.items():
        run_id = row["batch_row"]["run_id"]
        subprocess.run(["cp", "-a", "--reflink=auto", str(row["run_root"]),
                        str(staging_root / "runs" / run_id)], check=True)
        if row["snapshot"] is not None:
            subprocess.run(["cp", "-a", "--reflink=auto", str(row["snapshot"]),
                            str(staging_root / "source-snapshots" / run_id)], check=True)

    copied_methods: dict[str, Any] = {}
    copied_snapshots: dict[str, Any] = {}
    for method, original in usable.items():
        run_id = original["batch_row"]["run_id"]
        run_root = staging_root / "runs" / run_id
        queues = _read_raw_queues(run_root, targets)
        source = original["source"]
        method_row = original["method_row"]
        target_queues = {}
        for target in targets:
            queue_path = run_root / "queues" / f"{target}.json"
            target_queues[target] = {
                "entry_count": len(queues[target]),
                "sha256": sha256(queue_path),
                "copy_matches_source": True,
            }
        wrapper_path = run_root / "wrapper-result.json"
        wrapper = _read(wrapper_path) if wrapper_path.is_file() else {}
        copied_methods[method] = {
            "run_id": run_id,
            "original_batch_status": batch.get("status"),
            "original_usable_for_replay": original["batch_row"].get("usable_for_replay") is True,
            "wrapper_exit_code": original["batch_row"].get("wrapper_exit_code"),
            "wrapper_status": original["batch_row"].get("wrapper_status"),
            "candidate_count": method_row.get("accepted_candidate_count", len(queues[targets[0]])),
            "target_queues": target_queues,
            "content_validation": "not-checked",
            "errors": [],
            "container_exit_code": wrapper.get("exit_code"),
            "generation_status": source.get("status"),
            "generation_sealed": original["batch_row"].get("generation_sealed"),
            "queue_entry_count_per_target": {target: len(queues[target]) for target in targets},
            "dependency_identity_ok": source.get("dependency_identity_ok"),
            "config_sha256": original["config_sha256"],
            "wrapper_result_missing": not wrapper_path.is_file(),
        }
        if original["snapshot"] is not None:
            copied_snapshots[method] = {
                "path": f"source-snapshots/{run_id}",
                "sha256": original["snapshot_sha256"],
                "expected_batch_sha256": original["batch_row"].get("source_snapshot_sha256"),
            }

    corpus_manifest = {
        "schema_version": "rq1-case-corpus-copy-v1",
        "corpus_id": batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_batch_manifest": str(batch_path),
        "original_batch_manifest_sha256": sha256(staging_root / "generation-batch-manifest.json"),
        "original_batch_status": batch.get("status"),
        "original_batch_usable_method_count": batch.get("usable_method_count"),
        "content_validation": {
            "status": "not-checked",
            "validator": "case_catalog import",
            "method_count": len(copied_methods),
            "target_count": len(targets),
            "methods": copied_methods,
            "source_snapshots": copied_snapshots,
            "errors": [],
        },
        "excluded_methods": excluded,
        "replay_plan": "execute-existing-queues",
    }
    _atomic_json(staging_root / "corpus-manifest.json", corpus_manifest)
    (staging_root / "corpus-manifest.json.sha256").write_text(
        f"{sha256(staging_root / 'corpus-manifest.json')}  corpus-manifest.json\n",
        encoding="ascii",
    )
    inspected = _inspect_corpus(staging_root, artifact_root, config)
    os.rename(staging_root, final_root)
    subprocess.run(["chmod", "-R", "a-w", "--", str(final_root)], check=True)
    return result_for(final_root, corpus_manifest, inspected)


def _candidate_checkpoint_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        return [], hashlib.sha256(b"").hexdigest()
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        candidate = record.get("candidate") if isinstance(record, dict) else None
        if not isinstance(candidate, dict):
            continue
        identity = _candidate_identity_key(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(dict(candidate))
    return rows, hashlib.sha256(raw).hexdigest()


def _registered_checkpoint_prefix(catalog_root: Path, run_id: str) -> int:
    try:
        batches = _read(catalog_root / "catalog-manifest.json").get("batches", [])
    except ValueError:
        return 0
    if not isinstance(batches, list):
        return 0
    prefix = re.escape(f"{run_id}-gcp-")
    ranges = []
    for row in batches:
        batch_id = row.get("batch_id") if isinstance(row, dict) else None
        if not isinstance(batch_id, str):
            continue
        match = re.fullmatch(prefix + r"(\d+)-(\d+)-[0-9a-f]{12}", batch_id)
        if match:
            start, last = map(int, match.groups())
            if last >= start:
                ranges.append((start, last + 1))
    highwater = 0
    for start, end in sorted(ranges):
        if start > highwater:
            break
        highwater = max(highwater, end)
    return highwater


def _catalog_source_hashes(catalog_root: Path, method: str) -> set[str]:
    """Return source digests already registered for one method."""
    try:
        manifest = _read(catalog_root / "catalog-manifest.json")
    except ValueError:
        return set()
    hashes: set[str] = set()
    for entry in manifest.get("batches", []):
        if not isinstance(entry, dict):
            continue
        index_path = catalog_root / str(entry.get("index_path") or "")
        try:
            index = _read(index_path)
        except (OSError, TypeError, ValueError):
            continue
        method_row = (index.get("methods") or {}).get(method)
        if not isinstance(method_row, dict):
            continue
        for case in method_row.get("cases", []):
            if not isinstance(case, dict):
                continue
            source_sha = case.get("source_sha256")
            if isinstance(source_sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", source_sha):
                hashes.add(source_sha.lower())
    return hashes


def _checkpoint_copy_candidate_files(
    source_root: Path, destination_root: Path, method: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_relative = Path("run-tools") / "generation" / re.sub(
        r"[^A-Za-z0-9._-]+", "_", method,
    )
    source_tool_root = source_root / tool_relative
    destination_tool_root = destination_root / tool_relative
    copied_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        item = copy.deepcopy(candidate)
        for field in _CANDIDATE_PATH_FIELDS:
            raw = item.get(field)
            if not isinstance(raw, str) or not raw.strip():
                continue
            source_path = Path(raw)
            source_path = source_path if source_path.is_absolute() else source_tool_root / source_path
            if source_path.is_symlink():
                item[field] = None
                continue
            try:
                source_path = source_path.resolve(strict=True)
                relative = source_path.relative_to(source_root)
            except (OSError, ValueError):
                item[field] = None
                continue
            if not source_path.is_file() or source_path.is_symlink():
                item[field] = None
                continue
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source_path, destination)
            item[field] = os.path.relpath(destination, destination_tool_root)
        item["generation_candidate_index"] = candidate.get("index")
        item["index"] = len(copied_candidates)
        copied_candidates.append(item)
    return copied_candidates


def import_generation_checkpoint(
    catalog_root: str | Path, run_root: str | Path,
) -> dict[str, Any]:
    """Append newly available cases from a generation run."""
    catalog_root = Path(catalog_root).resolve()
    catalog_root.parent.mkdir(parents=True, exist_ok=True)
    run_path = Path(run_root)
    if run_path.is_symlink():
        raise ValueError("generation run must not be a symlink")
    run_root = run_path.resolve(strict=True)
    results_root = catalog_root.parent.parent.resolve(strict=True)
    if run_root.parent != (results_root / "runs").resolve(strict=True):
        raise ValueError("generation run must be directly below results/runs")
    control_root = run_root / "control"
    control_root.mkdir(parents=True, exist_ok=True)
    with (control_root / "generation-catalog-import.lock").open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            execution = _read(run_root / "execution-manifest.json")
            if execution.get("action") != "generate" or execution.get("run_id") != run_root.name:
                raise ValueError("checkpoint source is not the expected generate run")
            method = execution.get("method_filter")
            if not isinstance(method, str) or not method:
                raise ValueError("generation run method identity is missing")
            config_path = run_root / "config.snapshot.json"
            config = _read(config_path)
            if method not in _ids(config, "methods"):
                raise ValueError(f"generation method is not declared in its config: {method}")
            targets = _ids(config, "targets")
            if not targets:
                raise ValueError("generation config has no Targets")

            def optional_record(path: Path) -> dict[str, Any]:
                try:
                    value = _read(path) if path.is_file() else {}
                except ValueError:
                    return {}
                return value if isinstance(value, dict) else {}

            status_path = control_root / "status.json"
            status = optional_record(status_path)
            generation_path = run_root / "generation-result.json"
            result = optional_record(generation_path)
            paused = status.get("state") == "paused"
            terminal = (
                status.get("state") in {"completed", "failed", "stopped"}
                or (run_root / "wrapper-result.json").is_file()
            )
            journal_path = run_root / "generation" / "candidate-artifacts.partial.jsonl"
            journal_candidates, journal_sha = _candidate_checkpoint_rows(journal_path)
            cursor_path = control_root / "generation-catalog-import.json"
            cursor = optional_record(cursor_path)
            cursor_matches = cursor_path.is_file() and (
                cursor.get("schema_version") == "rq1-generation-catalog-import-v1"
                and cursor.get("run_id") == run_root.name
                and cursor.get("method") == method
            )
            imported_count = cursor.get("candidate_count", 0)
            cursor_recovered = (
                not cursor_matches
                or type(imported_count) is not int
                or imported_count < 0
            )
            if cursor_recovered:
                imported_count = _registered_checkpoint_prefix(catalog_root, run_root.name)

            if terminal:
                method_rows = result.get("methods")
                final_row = next((row for row in method_rows or []
                                  if isinstance(row, dict) and row.get("method") == method), {})
                final_candidates = final_row.get("candidate_artifacts")
                combined = list(journal_candidates)
                if isinstance(final_candidates, list):
                    combined.extend(item for item in final_candidates if isinstance(item, dict))
                unique_candidates: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
                for candidate in combined:
                    unique_candidates.setdefault(_candidate_identity_key(candidate), candidate)
                available_candidates = list(unique_candidates.values())
            else:
                available_candidates = journal_candidates
            if cursor_recovered:
                _atomic_json(cursor_path, {
                    "schema_version": "rq1-generation-catalog-import-v1",
                    "run_id": run_root.name, "method": method,
                    "candidate_count": imported_count,
                    "cursor_recovered": True,
                })
            candidate_slice = available_candidates[imported_count:]
            if not candidate_slice:
                return {
                    "status": "no-new-candidates", "run_id": run_root.name,
                    "imported_candidate_count": imported_count,
                    "candidate_count": len(available_candidates),
                    "catalog_root": str(catalog_root),
                }
            registered_sources = _catalog_source_hashes(catalog_root, method)
            candidates = []
            duplicate_source_count = 0
            for candidate in candidate_slice:
                source_sha = candidate.get("source_sha256")
                if (isinstance(source_sha, str)
                        and re.fullmatch(r"[0-9a-fA-F]{64}", source_sha)
                        and source_sha.lower() in registered_sources):
                    duplicate_source_count += 1
                    continue
                candidates.append(candidate)
            if not candidates:
                _atomic_json(cursor_path, {
                    "schema_version": "rq1-generation-catalog-import-v1",
                    "run_id": run_root.name, "method": method,
                    "candidate_count": len(available_candidates),
                    "duplicate_source_count": duplicate_source_count,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                return {
                    "status": "no-new-candidates", "run_id": run_root.name,
                    "imported_candidate_count": len(available_candidates),
                    "candidate_count": len(available_candidates),
                    "duplicate_source_count": duplicate_source_count,
                    "catalog_root": str(catalog_root),
                }

            snapshot_value = execution.get("source_snapshot_host_root")
            if not isinstance(snapshot_value, str) or not snapshot_value:
                raise ValueError("generation source snapshot path is missing")
            snapshot_input = Path(snapshot_value)
            if snapshot_input.is_symlink():
                raise ValueError("generation source snapshot must not be a symlink")
            snapshot_root = snapshot_input.resolve(strict=True)
            if snapshot_root.parent != (results_root / "source-snapshots").resolve(strict=True):
                raise ValueError("generation source snapshot is outside results/source-snapshots")
            if snapshot_root.name != run_root.name:
                raise ValueError("generation source snapshot run identity mismatch")
            snapshot_sha = _tree_sha256(snapshot_root)

            first = imported_count
            last = imported_count + len(candidate_slice) - 1
            checkpoint_sha = hashlib.sha256(
                json.dumps(candidates, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            checkpoint_id = (
                f"{run_root.name}-gcp-{first:08d}-{last:08d}-{checkpoint_sha[:12]}"
            )
            checkpoint_parent = results_root / "launch-logs" / "generation-checkpoints" / run_root.name
            checkpoint_parent.mkdir(parents=True, exist_ok=True)
            checkpoint_root = checkpoint_parent / checkpoint_id
            checkpoint_run_id = f"{checkpoint_id}-snapshot"
            batch_path = checkpoint_root / "generation-batch-manifest.json"
            if not checkpoint_root.exists():
                temporary_root = checkpoint_parent / f".{checkpoint_id}.tmp-{os.getpid()}"
                temporary_root.mkdir()
                try:
                    snapshot_run = temporary_root / "runs" / checkpoint_run_id
                    snapshot_run.mkdir(parents=True)
                    for relative in (
                        Path("config.snapshot.json"),
                        Path("coverage-registry.json"),
                    ):
                        source_file = run_root / relative
                        if source_file.is_file():
                            destination_file = snapshot_run / relative
                            destination_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_file, destination_file)
                    execution_copy = dict(execution)
                    execution_copy.update({
                        "run_id": checkpoint_run_id,
                        "checkpoint_source_run_id": run_root.name,
                        "generation_checkpoint_id": checkpoint_id,
                    })
                    _atomic_json(snapshot_run / "execution-manifest.json", execution_copy)

                    copied_candidates = _checkpoint_copy_candidate_files(
                        run_root, snapshot_run, method, candidates,
                    )
                    method_row = {
                        "method": method, "status": "partial",
                        "reason": "generation-checkpoint",
                        "generator_invoked": True, "generated": True, "built": True,
                        "candidate_count": len(copied_candidates),
                        "candidate_artifacts": copied_candidates,
                        "accepted_candidate_count": len(copied_candidates),
                        "accepted_count": len(copied_candidates),
                        "source_built_count": len(copied_candidates),
                        "generation_checkpoint_id": checkpoint_id,
                        "generation_checkpoint_source_run_id": run_root.name,
                    }
                    _refresh_generation_metrics(method_row)
                    method_config = next(
                        item for item in config.get("methods", [])
                        if isinstance(item, dict) and item.get("id") == method
                    )
                    queues = {target: [] for target in targets}
                    tool_root = snapshot_run / "run-tools" / "generation" / re.sub(
                        r"[^A-Za-z0-9._-]+", "_", method,
                    )
                    for candidate in copied_candidates:
                        for target in _applicable_targets(config, method):
                            artifact_available = bool(
                                candidate.get("elf") or candidate.get("linux_elf")
                            )
                            queues.setdefault(str(target), []).append({
                                "method": method,
                                "route": method_config.get("route"),
                                "lane": method_config.get("lane"),
                                "tool_root": str(tool_root.relative_to(snapshot_run)),
                                "candidate": candidate,
                                "artifact_status": "ready" if artifact_available else "artifact-gap",
                            })
                    generation_result = {
                        "schema_version": "rq1-generation-result-v1",
                        "run_id": checkpoint_run_id,
                        "phase": "generation-only", "status": "partial",
                        "method_filter": method,
                        "generation_sealed": True,
                        "generation_checkpoint": {
                            "kind": (
                                "paused-case-boundary-snapshot" if paused else
                                "terminal-suffix-snapshot" if terminal else
                                "live-candidate-snapshot"
                            ),
                            "source_run_id": run_root.name,
                            "source_candidate_start": first,
                            "source_candidate_end_exclusive": last + 1,
                            "source_candidate_journal_sha256": journal_sha,
                            "source_candidate_slice_sha256": checkpoint_sha,
                            "source_pause_reason": status.get("pause_reason"),
                        },
                        "started_at_utc": result.get("started_at_utc"),
                        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": execution.get("duration_seconds"),
                        "generation_seconds": execution.get("duration_seconds"),
                        "methods": [method_row],
                        "ledger_seal": "ledger/seal.json",
                    }
                    queue_summary: dict[str, Any] = {}
                    for target in targets:
                        entries = queues.get(target, [])
                        queue_path = snapshot_run / "queues" / f"{target}.json"
                        queue_doc = {
                            "schema_version": "rq1-target-queue-v1",
                            "target": target, "method_filter": method,
                            "generation_run_id": checkpoint_run_id,
                            "generation_source": {
                                "run_id": run_root.name,
                                "checkpoint_id": checkpoint_id,
                                "status": "partial",
                            },
                            "execution_model": "independent-target-worker",
                            "entries": entries,
                        }
                        _atomic_json(queue_path, queue_doc)
                        queue_summary[target] = {
                            "count": len(entries),
                            "status": "ready" if entries else "empty",
                        }
                    queues_summary = {
                        "schema_version": "rq1-generation-queue-v1",
                        "pipeline": "paused-generation-checkpoint-v1",
                        "generation_run_id": checkpoint_run_id,
                        "generation_source": {
                            "run_id": run_root.name,
                            "checkpoint_id": checkpoint_id,
                            "status": "partial",
                        },
                        "method_filter": method,
                        "methods": {method: {
                            "status": "partial",
                            "candidate_count": method_row["candidate_count"],
                            "accepted_candidate_count": method_row["accepted_candidate_count"],
                            "target_queue_entry_count": sum(
                                item["count"] for item in queue_summary.values()
                            ),
                        }},
                        "queues": queue_summary,
                    }
                    _atomic_json(snapshot_run / "generation" / "candidate-queues.json", queues_summary)
                    ledger = snapshot_run / "ledger" / "events.jsonl"
                    ledger.parent.mkdir(parents=True, exist_ok=True)
                    ledger.write_bytes(b"")
                    registry_sha = sha256(snapshot_run / "coverage-registry.json")
                    seal = {
                        "schema_version": "rq1-generation-ledger-seal-v1",
                        "algorithm": "sha256", "run_id": checkpoint_run_id,
                        "method_filter": method,
                        "config_sha256": execution.get("config_sha256"),
                        "registry_sha256": registry_sha,
                        "event_ledger_sha256": sha256(ledger),
                    }
                    seal["seal_sha256"] = digest(seal)
                    _atomic_json(snapshot_run / "ledger" / "seal.json", seal)
                    generation_result.update({
                        "event_count": 0,
                        "event_ledger": "ledger/events.jsonl",
                        "event_ledger_sha256": sha256(ledger),
                        "ledger_seal_sha256": sha256(snapshot_run / "ledger" / "seal.json"),
                        "coverage_registry": "coverage-registry.json",
                        "coverage_registry_sha256": registry_sha,
                        "coverage_status": "deferred",
                    })
                    _atomic_json(snapshot_run / "generation-result.json", generation_result)
                    _atomic_json(snapshot_run / "run-result.json", generation_result)
                    tool_run = {
                        "schema_version": "rq1-comparison-tool-run-v2",
                        "phase": "generation-only", "run_id": checkpoint_run_id,
                        "status": "partial", "method_filter": method,
                        "methods": [method_row],
                    }
                    _atomic_json(
                        snapshot_run / "run-tools" / "generation" / re.sub(
                            r"[^A-Za-z0-9._-]+", "_", method,
                        ) / "tool-run.json",
                        tool_run,
                    )
                    final_snapshot_run = checkpoint_root / "runs" / checkpoint_run_id
                    snapshot_row = {
                        "method": method, "run_id": checkpoint_run_id,
                        "status": "partial", "usable_for_replay": True,
                        "wrapper_exit_code": None,
                        "wrapper_status": (
                            "paused-checkpoint" if paused else
                            "terminal-suffix" if terminal else
                            "live-checkpoint"
                        ),
                        "generation_status": "partial",
                        "generation_sealed": True,
                        "candidate_count": method_row["accepted_candidate_count"],
                        "run_root": str(final_snapshot_run),
                        "source_snapshot": str(snapshot_root),
                        "source_snapshot_sha256": snapshot_sha,
                        "queue_manifest": str(final_snapshot_run / "generation" / "candidate-queues.json"),
                        "target_queues": {
                            target: {
                                "path": str(final_snapshot_run / "queues" / f"{target}.json"),
                                "entry_count": len(queues.get(target, [])),
                                "sha256": sha256(snapshot_run / "queues" / f"{target}.json"),
                            }
                            for target in targets
                        },
                    }
                    batch = {
                        "schema_version": "rq1-generation-batch-v1",
                        "status": "partial", "phase": "generation-only-checkpoint",
                        "duration_seconds": execution.get("duration_seconds"),
                        "simulator_execution": False, "board_requested": False,
                        "config_sha256": sha256(config_path),
                        "source": {
                            "commit": execution.get("source_commit"),
                            "dirty": execution.get("source_dirty"),
                            "tree_sha256": snapshot_sha,
                        },
                        "resource_profile": execution.get("resource_profile"),
                        "targets": targets,
                        "format_spec": "docs/EXTERNAL_CASE_CORPUS_FORMAT.md",
                        "started_at_utc": result.get("started_at_utc"),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "usable_method_count": 1, "failed_method_count": 0,
                        "methods": [snapshot_row],
                    }
                    _atomic_json(temporary_root / "generation-batch-manifest.json", batch)
                    os.rename(temporary_root, checkpoint_root)
                except BaseException:
                    # Keep the partial export for diagnosis and recovery.
                    raise
            else:
                batch = _read(batch_path)
                if batch.get("schema_version") != "rq1-generation-batch-v1":
                    raise ValueError(f"existing checkpoint has an invalid manifest: {checkpoint_root}")

            imported = import_generation_batch(catalog_root, batch_path, config_path)
            registered = imported.get("registered_methods", {}).get(method)
            expected_count = len(candidates)
            registered_count = registered if isinstance(registered, int) else 0
            next_cursor = {
                "schema_version": "rq1-generation-catalog-import-v1",
                "run_id": run_root.name, "method": method,
                "candidate_count": len(available_candidates),
                "last_checkpoint_id": checkpoint_id,
                "last_checkpoint_sha256": checkpoint_sha,
                "candidate_journal_sha256": journal_sha,
                "duplicate_source_count": duplicate_source_count,
                "catalog_revision": imported.get("catalog_revision"),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(cursor_path, next_cursor)
            return {
                "status": "imported", "run_id": run_root.name,
                "method": method, "checkpoint_id": checkpoint_id,
                "candidate_start": first, "candidate_end_exclusive": last + 1,
                "imported_candidate_count": expected_count,
                "registered_candidate_count": registered_count,
                "duplicate_source_count": duplicate_source_count,
                **imported,
            }
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def import_generation_run(
    catalog_root: str | Path,
    run_root: str | Path,
    source_snapshot: str | Path,
) -> dict[str, Any]:
    """Append the available queues from one standalone generation run."""
    catalog_root = Path(catalog_root).resolve(strict=True)
    artifact_base = catalog_root.parent.parent.resolve(strict=True)
    run_input = Path(run_root)
    snapshot_input = Path(source_snapshot)
    if run_input.is_symlink() or snapshot_input.is_symlink():
        raise ValueError("generation run and source snapshot must not be symlinks")
    run_root = run_input.resolve(strict=True)
    source_snapshot = snapshot_input.resolve(strict=True)
    if run_root.parent != (artifact_base / "runs").resolve(strict=True):
        raise ValueError("generation run must be directly below results/runs")
    if source_snapshot.parent != (artifact_base / "source-snapshots").resolve(strict=True):
        raise ValueError("source snapshot must be directly below results/source-snapshots")

    run_id = run_root.name
    execution = _read(run_root / "execution-manifest.json")
    result = _read(run_root / "generation-result.json")
    config_path = run_root / "config.snapshot.json"
    config = _read(config_path)
    config_sha = sha256(config_path)
    method = execution.get("method_filter")
    exit_path = run_root / "wrapper-logs" / "container.exit-code"
    try:
        exit_code = int(exit_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        exit_code = None
    if not isinstance(method, str) or not method:
        method = run_root.name

    targets = _ids(config, "targets")
    queues = _read_raw_queues(run_root, targets)
    target_queues: dict[str, dict[str, Any]] = {}
    candidate_count = max((len(queues[target]) for target in targets), default=0)
    result_row = next(
        (row for row in result.get("methods", [])
         if isinstance(row, dict) and row.get("method") == method),
        {},
    )
    if not isinstance(result_row, dict):
        result_row = {}
    for target in targets:
        queue_path = run_root / "queues" / f"{target}.json"
        target_queues[target] = {
            "path": str(queue_path),
            "entry_count": len(queues[target]),
            "sha256": sha256(queue_path) if queue_path.is_file() else "",
        }

    snapshot_sha = _tree_sha256(source_snapshot)
    batch_id = f"rq1-case-batch-{run_id}"
    batch_dir = artifact_base / "launch-logs" / batch_id
    if batch_dir.is_symlink():
        raise ValueError("generation import directory must not be a symlink")
    batch_path = batch_dir / "generation-batch-manifest.json"
    run_path = str(run_root)
    snapshot_path = str(source_snapshot)
    queue_manifest_path = str(run_root / "generation" / "candidate-queues.json")
    batch = {
        "schema_version": "rq1-generation-batch-v1",
        "status": result.get("status", "partial"),
        "phase": "generation-only",
        "duration_seconds": result.get("generation_seconds"),
        "simulator_execution": execution.get("target_execution_enabled") is True,
        "board_requested": execution.get("board_requested"),
        "config_sha256": config_sha,
        "source": {
            "commit": execution.get("source_commit"),
            "dirty": execution.get("source_dirty"),
            "tree_sha256": snapshot_sha,
        },
        "resource_profile": execution.get("resource_profile"),
        "targets": targets,
        "format_spec": "docs/EXTERNAL_CASE_CORPUS_FORMAT.md",
        "started_at_utc": result.get("started_at_utc"),
        "finished_at_utc": result.get("ended_at_utc"),
        "usable_method_count": 1,
        "failed_method_count": 0,
        "methods": [{
            "method": method,
            "run_id": run_id,
            "status": result.get("status", "partial"),
            "usable_for_replay": bool(candidate_count),
            "wrapper_exit_code": exit_code,
            "wrapper_status": result["status"],
            "generation_status": result.get("status", "partial"),
            "generation_sealed": result.get("generation_sealed"),
            "observed_seconds": result.get("observed_seconds"),
            "candidate_count": candidate_count,
            "candidate_failure_count": result_row.get("candidate_failure_count"),
            "artifact_gap_count": result_row.get("artifact_gap_count"),
            "run_root": run_path,
            "source_snapshot": snapshot_path,
            "source_snapshot_sha256": snapshot_sha,
            "source_snapshot_matches_batch": True,
            "queue_manifest": queue_manifest_path,
            "queue_manifest_sha256": sha256(run_root / "generation" / "candidate-queues.json"),
            "target_queues": target_queues,
            "dependency_identity_ok": execution.get("dependency_identity_ok"),
        }],
    }
    if batch_path.is_symlink():
        raise ValueError("generation batch manifest must not be a symlink")
    batch_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(batch_path, batch)
    imported = import_generation_batch(catalog_root, batch_path)
    imported["generation_batch_manifest"] = str(batch_path)
    imported["source_run_id"] = run_id
    return imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="maintain append-only RQ1 case catalogs")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="register an existing validated corpus by reference")
    register.add_argument("--catalog-root", required=True)
    register.add_argument("--corpus-root", required=True)
    register.add_argument("--config")
    importer = commands.add_parser(
        "import-generation-batch", help="copy available queues and register their cases",
    )
    importer.add_argument("--catalog-root", required=True)
    importer.add_argument("--batch-manifest", required=True)
    importer.add_argument("--config")
    checkpoint_importer = commands.add_parser(
        "import-generation-checkpoint",
        help="snapshot available cases from a running, paused, or terminal generation run",
    )
    checkpoint_importer.add_argument("--catalog-root", required=True)
    checkpoint_importer.add_argument("--run-root", required=True)
    run_importer = commands.add_parser(
        "import-generation-run", help="append available queues from one standalone generation run",
    )
    run_importer.add_argument("--catalog-root", required=True)
    run_importer.add_argument("--run-root", required=True)
    run_importer.add_argument("--source-snapshot", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            corpus_root = Path(args.corpus_root).resolve(strict=True)
            if args.config:
                config = _read(Path(args.config).resolve(strict=True))
            else:
                copied, _, _, _ = _corpus_manifests(corpus_root)
                methods = copied["content_validation"]["methods"]
                method = next(iter(methods))
                run_id = methods[method]["run_id"]
                config = _read(corpus_root / "runs" / run_id / "config.snapshot.json")
            manifest = register_corpus(args.catalog_root, corpus_root, config)
            print(json.dumps({
                "catalog_root": str(Path(args.catalog_root).resolve()),
                "catalog_revision": manifest["revision"],
                "catalog_sha256": manifest["catalog_sha256"],
                "case_counts": {key: row["case_count"] for key, row in manifest["methods"].items()},
            }, ensure_ascii=False, indent=2))
        elif args.command == "import-generation-run":
            result = import_generation_run(
                args.catalog_root, args.run_root, args.source_snapshot,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "import-generation-checkpoint":
            result = import_generation_checkpoint(args.catalog_root, args.run_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = import_generation_batch(
                args.catalog_root, args.batch_manifest, args.config,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"case_catalog: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

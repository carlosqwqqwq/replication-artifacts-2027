#!/usr/bin/env python3
"""Write a provisional full-batch source and RV opcode snapshot after a run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.source_coverage import (
    _identity as source_coverage_identity,
    _merge_lcov_profiles,
    collect_source_coverage_batch,
)
from analysis.simulator_coverage import target_source_coverage
from analysis.rv_instruction_coverage import (
    OPCODE_CATALOG_DENOMINATOR,
    OPCODE_CATALOG_ID,
    OPCODE_CATALOG_KEYS,
    OPCODE_CATALOG_REGISTRY_SHA256,
    OPCODE_CATALOG_SCHEMA,
    OPCODE_KEY_SCHEMA,
    aggregate_opcode_catalog_summary,
)
from comparison_coverage import _materialize_guest_coverage
from comparison_observations import _record_artifact_digest
from framework.framework_coverage import (
    _observations,
    _target_stratum,
)


METHODS = (
    "B-RVDV", "B-TORTURE", "B-CSMITH", "B-GEMI", "Fuzz4All",
    "Ours-RVGEN-Direct", "Ours-Program-Full",
)
PROFILE_SUFFIXES = {
    "gcov": (".gcda",),
    "lcov": (".gcda",),
    "llvm": (".profraw",),
    "dotnet": (".cobertura.xml", ".cobertura.xml.gz"),
}
METRIC_NAMES = ("function", "line", "branch")
RV_CATALOG_METRICS = ("GenOpcodeCov", "ExecOpcodeCov")


def _valid_catalog_metric_status(entry: Mapping[str, Any], name: str) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    if (
        entry.get("schema_version") != OPCODE_CATALOG_SCHEMA
        or entry.get("catalog_id") != OPCODE_CATALOG_ID
        or entry.get("key_schema") != OPCODE_KEY_SCHEMA
        or entry.get("registry_sha256") != OPCODE_CATALOG_REGISTRY_SHA256
    ):
        return None
    metrics = entry.get("metrics")
    metric = metrics.get(name) if isinstance(metrics, Mapping) else None
    if not isinstance(metric, Mapping):
        return None
    units = metric.get("covered_units")
    outside = metric.get("out_of_catalog_units", ())
    if not (
        isinstance(units, list)
        and all(isinstance(unit, str) for unit in units)
        and len(set(units)) == len(units)
        and set(units) <= OPCODE_CATALOG_KEYS
        and type(metric.get("covered")) is int
        and metric.get("covered") == len(units)
        and type(metric.get("eligible")) is int
        and metric.get("eligible") == OPCODE_CATALOG_DENOMINATOR
        and metric.get("registry_sha256") == OPCODE_CATALOG_REGISTRY_SHA256
        and isinstance(outside, (list, tuple))
        and all(isinstance(unit, str) for unit in outside)
    ):
        return None
    status = metric.get("status")
    return status if status in {"observed", "partial", "gap", "NA"} else None


def _catalog_has_reportable_prefix(statuses: Mapping[str, int], covered: object) -> bool:
    return bool(
        statuses.get("observed") or statuses.get("partial")
        or type(covered) is int and covered > 0
    )


def _catalog_provisional_value(
    metric: Mapping[str, Any], source_statuses: Mapping[str, int],
) -> float | None:
    covered, eligible = metric.get("covered"), metric.get("eligible")
    if (
        type(covered) is int and type(eligible) is int and eligible > 0
        and _catalog_has_reportable_prefix(source_statuses, covered)
    ):
        return round(covered / eligible, 6)
    return None


def _catalog_metric_status_counts(entries, name: str) -> Counter:
    statuses = Counter()
    for entry in entries:
        status = _valid_catalog_metric_status(entry, name)
        if status is not None:
            statuses[status] += 1
    return statuses


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _is_profile(path: Path, collector: str) -> bool:
    suffixes = PROFILE_SUFFIXES.get(collector, ())
    if collector in {"gcov", "lcov"} and path.name.endswith(".tmp.gcda"):
        return False
    return path.is_file() and any(path.name.endswith(item) for item in suffixes)


def _profile_files(root: Path, collector: str) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(
            (path for path in root.rglob("*") if _is_profile(path, collector)),
            key=lambda path: path.as_posix(),
        )
    except OSError:
        return []


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _stage_profile(
    *, source: Path, destination: Path, marker: Path, collector: str,
    run_root: Path,
) -> int:
    """Hard-link closed source-profile files into the progress-only input tree."""
    existing = _load_json(marker)
    if existing.get("staged") is True:
        return int(existing.get("file_count", 0) or 0)

    files = [path for path in _profile_files(source, collector) if _within(path, run_root)]
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        copied += 1
    _write_json(marker, {
        # Empty input directories are retried during the post-run pass so a
        # profile that is still flushing can be picked up later.
        "staged": bool(copied),
        "file_count": copied,
        "source": str(source),
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return copied


def _parse_jsonl(path: Path):
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    # The live writer may be in the middle of its last line.
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _catalog_progress(
    method: str, target_id: str, entries: list[Mapping[str, Any]],
    expected_cases: int, *, enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "disabled", "enabled": False,
            "reason": "rv-opcode-catalog-coverage-disabled",
            "metrics": {},
        }
    if expected_cases <= 0:
        return {
            "schema_version": OPCODE_CATALOG_SCHEMA,
            "catalog_id": OPCODE_CATALOG_ID,
            "key_schema": OPCODE_KEY_SCHEMA,
            "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
            "eligible": OPCODE_CATALOG_DENOMINATOR,
            "status": "waiting", "reason": "no-target-attempts-yet",
            "expected_case_count": 0, "represented_case_count": 0,
            "provisional": True,
            "metrics": {
                name: {
                    "covered": None, "eligible": OPCODE_CATALOG_DENOMINATOR,
                    "value": None, "provisional_value": None, "status": "waiting",
                }
                for name in RV_CATALOG_METRICS
            },
        }

    represented_cases = 0
    valid_metrics = Counter()
    source_statuses = {name: Counter() for name in RV_CATALOG_METRICS}
    records = []
    for entry in entries:
        count = entry.get("case_count", 1)
        if type(count) is int and count >= 0:
            represented_cases += count
        for name in RV_CATALOG_METRICS:
            status = _valid_catalog_metric_status(entry, name)
            if status is not None:
                valid_metrics[name] += 1
                source_statuses[name][status] += 1
        records.append({
            "method": method, "route": "post-execution", "lane": "all",
            "stratum": "all", "target": target_id,
            "rv_opcode_catalog_coverage": entry,
        })

    row = {
        "method": method, "route": "post-execution", "lane": "all",
        "stratum": "all", "target": target_id, "cases": expected_cases,
    }
    summary = aggregate_opcode_catalog_summary({"targets": [row]}, records)
    result_row = summary["targets"][0]
    catalog = result_row.get("rv_opcode_catalog_coverage")
    catalog = dict(catalog) if isinstance(catalog, Mapping) else {
        "schema_version": OPCODE_CATALOG_SCHEMA,
        "catalog_id": OPCODE_CATALOG_ID,
        "key_schema": OPCODE_KEY_SCHEMA,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "metrics": {}, "status": "gap",
        "reason": "opcode-catalog-evidence-missing",
    }
    catalog["expected_case_count"] = expected_cases
    catalog["represented_case_count"] = represented_cases
    catalog["eligible"] = OPCODE_CATALOG_DENOMINATOR
    catalog["provisional"] = True

    metric_statuses = []
    metrics = catalog.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    output_metrics = {}
    for name in RV_CATALOG_METRICS:
        metric = dict(metrics.get(name, {})) if isinstance(metrics.get(name), Mapping) else {}
        metric.setdefault("covered", None)
        metric.setdefault("eligible", OPCODE_CATALOG_DENOMINATOR)
        metric.setdefault("value", None)
        metric.setdefault("status", "gap")
        metric.setdefault("reason", "opcode-catalog-evidence-missing")
        covered, eligible = metric.get("covered"), metric.get("eligible")
        has_evidence = valid_metrics[name] > 0
        has_reportable_prefix = (
            source_statuses[name]["observed"] > 0
            or source_statuses[name]["partial"] > 0
            or type(covered) is int and covered > 0
        )
        metric["provisional_value"] = (
            _catalog_provisional_value(metric, source_statuses[name])
            if has_evidence else None
        )
        partitions = metric.get("partition_coverage")
        if isinstance(partitions, Mapping):
            metric["partition_coverage"] = {
                partition: {
                    **dict(partition_metric),
                    "provisional_value": (
                        round(partition_metric["covered"] / partition_metric["eligible"], 6)
                        if type(partition_metric.get("covered")) is int
                        and type(partition_metric.get("eligible")) is int
                        and partition_metric.get("eligible", 0) > 0
                        and has_evidence and has_reportable_prefix else None
                    ),
                }
                for partition, partition_metric in partitions.items()
                if isinstance(partition_metric, Mapping)
            }
        # A shortfall in the live prefix is expected while cases or coverage
        # finalizers are still running. Preserve the observed lower bound and
        # label it partial; contradictory/invalid evidence remains a gap.
        if (
            metric.get("status") == "gap"
            and represented_cases < expected_cases
            and metric.get("reason") == "case-count-mismatch"
            and has_evidence
            and source_statuses[name]["gap"] == 0
        ):
            metric["status"] = "partial"
            metric["reason"] = "opcode-catalog-evidence-pending"
        metric_statuses.append(str(metric.get("status") or "gap"))
        output_metrics[name] = metric
    catalog["metrics"] = output_metrics
    catalog["status"] = (
        "gap" if "gap" in metric_statuses else
        "partial" if "partial" in metric_statuses
        or "NA" in metric_statuses and not all(item == "NA" for item in metric_statuses) else
        "NA" if metric_statuses and all(item == "NA" for item in metric_statuses) else
        "observed"
    )
    if catalog["status"] == "partial" and represented_cases < expected_cases:
        catalog["reason"] = "opcode-catalog-evidence-pending"
    return catalog


def _method_catalog_summary(
    method: str, targets: list[Mapping[str, Any]],
    input_data: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    target_rows = []
    records = []
    generated_statuses = Counter()
    for target in targets:
        target_id = str(target.get("id") or "")
        inputs = input_data.get(target_id, {})
        expected_cases = _metric_count(
            inputs.get("catalog_expected_cases", inputs.get("attempts", 0))
        )
        if not target_id or expected_cases <= 0:
            continue
        stratum = (
            "linux-user"
            if str(target.get("execution_model", "")).startswith("linux-user")
            else "bare-metal"
        )
        target_rows.append({
            "method": method,
            "route": "post-execution",
            "lane": "all",
            "stratum": stratum,
            "target": target_id,
            "cases": expected_cases,
        })
        entries = inputs.get("catalog_entries")
        for entry in entries if isinstance(entries, list) else ():
            if isinstance(entry, Mapping):
                records.append({
                    "method": method,
                    "route": "post-execution",
                    "lane": "all",
                    "stratum": stratum,
                    "target": target_id,
                    "rv_opcode_catalog_coverage": entry,
                })
                status = _valid_catalog_metric_status(entry, "GenOpcodeCov")
                if status is not None:
                    generated_statuses[status] += 1
    if not target_rows:
        return {}
    summary = {"targets": target_rows}
    aggregate_opcode_catalog_summary(summary, records)
    result = summary.get("rv_opcode_catalog_coverage")
    if not isinstance(result, dict):
        return {}
    result["input_scope"] = "target-case opcode evidence represented in the current event ledger"
    for row in result.get("method_generated_unions", []):
        if not isinstance(row, dict) or row.get("method") != method:
            continue
        metric = (row.get("metrics") or {}).get("GenOpcodeCov")
        if isinstance(metric, dict):
            metric["provisional_value"] = _catalog_provisional_value(
                metric, generated_statuses,
            )
    return result


def _metric_count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _method_row(document: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    methods = document.get("methods")
    if isinstance(methods, list):
        return next((row for row in methods if isinstance(row, Mapping)
                     and row.get("method") == method), {})
    if isinstance(methods, Mapping):
        row = methods.get(method)
        return row if isinstance(row, Mapping) else {}
    return {}


def _guest_cache_identity(
    run_root: Path, case_root: Path, target_id: str,
    event: Mapping[str, Any], event_artifact: str | None,
    cached_identity: Mapping[str, Any] | None = None,
    *, hash_if_needed: bool = True,
) -> dict[str, Any] | None:
    result = _load_json(case_root / "target-run.json")
    rows = result.get("targets")
    target_row = next((row for row in rows if isinstance(row, Mapping)
                       and row.get("id") == target_id), {}) \
        if isinstance(rows, list) else {}
    row_artifact = _record_artifact_digest(dict(target_row))
    trace = target_row.get("trace") if isinstance(target_row, Mapping) else None
    trace = trace if isinstance(trace, Mapping) else {}
    trace_sha = (
        target_row.get("trace_sha256") or trace.get("sha256")
        or event.get("trace_sha256")
    ) if isinstance(target_row, Mapping) else event.get("trace_sha256")
    trace_value = (
        target_row.get("trace_path") or trace.get("path")
        if isinstance(target_row, Mapping) else None
    ) or event.get("trace_path")
    resolved_trace = None
    trace_size = trace_mtime_ns = None
    if isinstance(trace_value, str) and trace_value:
        resolved_trace = Path(trace_value)
        if not resolved_trace.is_absolute():
            resolved_trace = case_root / resolved_trace
        if _within(resolved_trace, run_root):
            try:
                trace_stat = resolved_trace.stat()
                trace_size, trace_mtime_ns = trace_stat.st_size, trace_stat.st_mtime_ns
            except OSError:
                resolved_trace = None
    old_identity = cached_identity if isinstance(cached_identity, Mapping) else {}
    unchanged_trace = bool(
        resolved_trace is not None
        and old_identity.get("trace_path") == str(resolved_trace.resolve())
        and old_identity.get("trace_size_bytes") == trace_size
        and old_identity.get("trace_mtime_ns") == trace_mtime_ns
        and isinstance(old_identity.get("trace_sha256"), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", old_identity["trace_sha256"])
    )
    if unchanged_trace:
        trace_sha = old_identity["trace_sha256"]
    elif resolved_trace is not None:
        if not hash_if_needed:
            return None
        trace_sha = _sha256(resolved_trace)
    if not isinstance(trace_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", trace_sha):
        return None
    artifact = event_artifact or row_artifact
    if (
        not isinstance(artifact, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", artifact)
        or row_artifact not in (None, artifact.lower())
        or not isinstance(trace_sha, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", trace_sha)
    ):
        return None
    return {
        "artifact_id": artifact.lower(),
        "target_artifact_id": row_artifact.lower() if row_artifact else None,
        "trace_sha256": trace_sha.lower(),
        "trace_path": str(resolved_trace.resolve()) if resolved_trace else None,
        "trace_size_bytes": trace_size,
        "trace_mtime_ns": trace_mtime_ns,
        "catalog_id": OPCODE_CATALOG_ID,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
    }


def _save_external_catalog_entry(
    *, run_root: Path, event: Mapping[str, Any], marker: Path,
    artifact_id: str | None, case_root: Path,
    old_identity: Mapping[str, Any] | None,
) -> None:
    coverage = event.get("simulator_coverage")
    entry = coverage.get("rv_opcode_catalog_coverage") \
        if isinstance(coverage, Mapping) else None
    if not isinstance(entry, Mapping):
        return
    identity = _guest_cache_identity(
        run_root, case_root, str(event.get("target") or ""),
        event, artifact_id, old_identity,
    )
    if identity is None:
        return
    _write_json(marker, {
        "schema_version": "rq1-opcode-evidence-v1",
        "artifact_id": artifact_id,
        "identity": identity,
        "rv_opcode_catalog_coverage": dict(entry),
    })


def _external_inputs(
    run_root: Path, method: str, targets: list[Mapping[str, Any]], progress_root: Path,
    *, defer_materialization: bool = False,
    pending_by_target: dict[str, list[tuple[Any, ...]]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    target_by_id = {str(target.get("id")): target for target in targets}
    counts = {target_id: {"attempts": 0, "profiles": 0, "data_profiles": 0,
                          "catalog_entries": [], "catalog_expected_cases": 0,
                          "outcomes": Counter(),
                          "statuses": Counter()}
              for target_id in target_by_id}
    total_outcomes, total_statuses = Counter(), Counter()
    events_path = run_root / "ledger" / "events.partial.jsonl"
    if not events_path.is_file():
        events_path = run_root / "ledger" / "events.jsonl"
    events = list(_parse_jsonl(events_path))
    materialization_candidates = []
    materialization_cache = []
    cached_cases = 0
    # ponytail: scan completed ledger events once per post-run aggregation; add a cursor only if profiling shows this scan is costly.
    for event in events:
        if not isinstance(event, dict):
            continue
        target_id = str(event.get("target") or "")
        if target_id not in target_by_id:
            continue
        coverage = event.get("simulator_coverage")
        if (isinstance(coverage, Mapping)
                and isinstance(coverage.get("rv_opcode_catalog_coverage"), Mapping)):
            continue
        index = event.get("stream_index")
        if type(index) is not int or index < 0:
            match = re.fullmatch(r"candidate-(\d+)", str(event.get("case_id") or ""))
            if not match:
                continue
            index = int(match.group(1))
        artifact_id = next((event.get(key) for key in (
            "artifact_id", "linux_elf_sha256", "bare_elf_sha256", "rvvm_elf_sha256",
        ) if isinstance(event.get(key), str)), None)
        marker = (
            progress_root / "index" / _safe(method) / "opcode-catalog"
            / _safe(target_id) / f"candidate-{index:08d}.json"
        )
        case_root = (
            run_root / "executions" / _safe(target_id) / _safe(method)
            / f"candidate-{index}"
        )
        cached = _load_json(marker)
        entry = cached.get("rv_opcode_catalog_coverage")
        cache_identity = _guest_cache_identity(
            run_root, case_root, target_id, event, artifact_id,
            cached.get("identity") if isinstance(cached.get("identity"), Mapping) else None,
            hash_if_needed=not defer_materialization,
        )
        if (cached.get("schema_version") == "rq1-opcode-evidence-v1"
                and cache_identity is not None
                and cached.get("identity") == cache_identity
                and isinstance(entry, Mapping)):
            coverage = dict(coverage) if isinstance(coverage, Mapping) else {
                "schema_version": "rq1-simulator-coverage-v4",
            }
            coverage["rv_opcode_catalog_coverage"] = dict(entry)
            event["simulator_coverage"] = coverage
            cached_cases += 1
            continue
        if event.get("target_attempted") is not True:
            continue
        if not (case_root / "target-run.json").is_file():
            continue
        materialization_candidates.append(event)
        materialization_cache.append((
            event, marker, artifact_id, cache_identity, case_root,
            cached.get("identity") if isinstance(cached.get("identity"), Mapping) else None,
        ))

    coverage_materialization = {
        "cases": 0, "parsed_cases": 0, "missing_cases": 0,
        "error_cases": 0, "elapsed_s": 0.0,
    }
    if defer_materialization:
        if pending_by_target is None:
            raise ValueError("deferred trace materialization requires a candidate map")
        for candidate in materialization_cache:
            pending_by_target.setdefault(str(candidate[0].get("target") or ""), []).append(
                candidate,
            )
        coverage_materialization["cases"] = len(materialization_candidates)
    elif materialization_candidates:
        try:
            def save_materialized_case(index: int, _event: dict[str, Any], state: str) -> None:
                if state != "parsed":
                    return
                event, marker, artifact_id, _, case_root, old_identity = (
                    materialization_cache[index]
                )
                _save_external_catalog_entry(
                    run_root=run_root, event=event, marker=marker,
                    artifact_id=artifact_id, case_root=case_root,
                    old_identity=old_identity,
                )

            coverage_materialization = _materialize_guest_coverage(
                run_root,
                [dict(target) for target in targets if isinstance(target, Mapping)],
                materialization_candidates,
                max_workers=1,
                on_case=save_materialized_case,
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError):
            coverage_materialization = {
                "cases": len(materialization_candidates),
                "error_cases": len(materialization_candidates),
                "elapsed_s": None,
            }
    coverage_materialization["cached_cases"] = cached_cases
    coverage_materialization["candidate_cases"] = len(materialization_candidates)
    for event in events:
        target_id = event.get("target")
        if target_id not in target_by_id:
            continue
        coverage = event.get("simulator_coverage")
        entry = coverage.get("rv_opcode_catalog_coverage") \
            if isinstance(coverage, Mapping) else None
        if isinstance(entry, Mapping):
            counts[target_id]["catalog_entries"].append(entry)
        if (
            isinstance(entry, Mapping)
            or event.get("artifact_available") is True
            or event.get("target_attempted") is True
        ):
            case_count = entry.get("case_count", 1) if isinstance(entry, Mapping) else 1
            counts[target_id]["catalog_expected_cases"] += (
                case_count if type(case_count) is int and case_count >= 0 else 1
            )
        if event.get("target_attempted") is not True:
            continue
        counts[target_id]["attempts"] += 1
        outcome, status = event.get("outcome"), event.get("status")
        if isinstance(outcome, str) and outcome:
            counts[target_id]["outcomes"][outcome] += 1
            total_outcomes[outcome] += 1
        if isinstance(status, str) and status:
            counts[target_id]["statuses"][status] += 1
            total_statuses[status] += 1
        profile_id = event.get("source_coverage_profile_id")
        if not isinstance(profile_id, str) or not re.fullmatch(r"[0-9a-f]{64}", profile_id):
            continue
        collector = str((target_by_id[target_id].get("source_coverage") or {}).get("collector") or "")
        source = run_root / "coverage-raw" / _safe(target_id) / "attempts" / profile_id
        key = f"external-{profile_id}"
        marker = progress_root / "index" / method / "external" / _safe(target_id) / f"{key}.json"
        counts[target_id]["profiles"] += 1
        destination = progress_root / "inputs" / _safe(target_id) / "raw" / "profiles" / key
        if _stage_profile(
            source=source, destination=destination, marker=marker,
            collector=collector, run_root=run_root,
        ):
            counts[target_id]["data_profiles"] += 1
    generation = _load_json(run_root / "generation-result.partial.json")
    if not generation:
        generation = _load_json(run_root / "generation-result.json")
    execution = _load_json(run_root / "execution-result.partial.json")
    generation_row = _method_row(generation, method)
    candidate_count = generation_row.get("candidate_count")
    if type(candidate_count) is not int:
        candidate_count = generation_row.get("accepted_candidate_count")
    execution_total = sum(item["attempts"] for item in counts.values())
    progress = {
        "kind": "external-tool",
        "generation": {
            "status": generation_row.get("generation_status")
            or generation_row.get("status") or generation.get("status"),
            "raw_candidates": generation_row.get("raw_candidate_count"),
            "candidates": candidate_count,
            "artifact_gaps": generation_row.get("artifact_gap_count"),
            "duplicates": generation_row.get("duplicate_candidate_count"),
            "deadline_rejected": generation_row.get("deadline_rejected_candidate_count"),
        },
        "execution": {
            "target_attempts": execution_total,
            "event_count": execution.get("event_count", execution_total),
            "processed_count": execution.get("processed_count"),
            "pending_count": execution.get("pending_count"),
            "outcome_counts": dict(sorted(total_outcomes.items())),
            "status_counts": dict(sorted(total_statuses.items())),
        },
        "coverage_materialization": coverage_materialization,
        "targets": {
            target_id: {
                "attempts": value["attempts"],
                "catalog_expected_cases": value["catalog_expected_cases"],
                "outcomes": dict(sorted(value["outcomes"].items())),
                "statuses": dict(sorted(value["statuses"].items())),
            }
            for target_id, value in counts.items()
        },
    }
    return counts, progress


def _materialize_external_target(
    run_root: Path, target: Mapping[str, Any], candidates: list[tuple[Any, ...]],
) -> dict[str, Any]:
    events = [item[0] for item in candidates]
    try:
        def save_materialized_case(index: int, _event: dict[str, Any], state: str) -> None:
            if state != "parsed":
                return
            event, marker, artifact_id, _, case_root, old_identity = candidates[index]
            _save_external_catalog_entry(
                run_root=run_root, event=event, marker=marker,
                artifact_id=artifact_id, case_root=case_root,
                old_identity=old_identity,
            )

        summary = _materialize_guest_coverage(
            run_root, [dict(target)], events, max_workers=1,
            on_case=save_materialized_case,
        )
        return summary
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError):
        return {
            "cases": len(candidates), "error_cases": len(candidates),
            "elapsed_s": None,
        }


def _framework_attempted_targets(
    campaign: Mapping[str, Any], target_ids: set[str],
) -> set[str]:
    summaries = campaign.get("target_summaries")
    if isinstance(summaries, Mapping):
        return {
            target_id for target_id in target_ids
            if isinstance(summary := summaries.get(target_id), Mapping)
            and (
                type(summary.get("record_count")) is int
                and summary["record_count"] > 0
                or type(summary.get("baseline_count")) is int
                and summary["baseline_count"] > 0
            )
        }
    records_by_target = campaign.get("target_records_by_target")
    baselines_by_target = campaign.get("target_baselines_by_target")
    if isinstance(records_by_target, Mapping) or isinstance(baselines_by_target, Mapping):
        records_by_target = records_by_target if isinstance(records_by_target, Mapping) else {}
        baselines_by_target = baselines_by_target if isinstance(baselines_by_target, Mapping) else {}
        return {
            target_id for target_id in target_ids
            if bool(records_by_target.get(target_id, ()))
            or isinstance(baselines_by_target.get(target_id), Mapping)
        }
    return set()


def _framework_result_attempted_targets(
    case_dir: Path, target_ids: set[str],
) -> set[str]:
    """Read resident Target result sidecars when raw-only campaign arrays are empty."""
    attempted: set[str] = set()
    try:
        method_root = case_dir.resolve().parents[3]
        case_relative = case_dir.resolve().relative_to(method_root)
    except (OSError, ValueError, IndexError):
        method_root = None
        case_relative = None
    if method_root is not None and case_relative is not None:
        for target_id in target_ids:
            raw_root = (
                method_root / "target-queues" / _safe(target_id)
                / "raw" / "cases" / case_relative
            )
            if any(
                path.is_file()
                and not path.name.endswith(".raw-artifact-manifest.json")
                and (path.name == "baseline.json" or path.name.startswith("candidate-"))
                for path in raw_root.glob("*.json")
            ):
                attempted.add(target_id)
        if attempted:
            return attempted
    result_root = case_dir / "target-replay-queue" / "results"
    for path in sorted(result_root.glob("target-*/*.json")):
        value = _load_json(path)
        target_id = value.get("target_id")
        if target_id not in target_ids:
            continue
        record = value.get("record")
        record = record if isinstance(record, Mapping) else value.get("result")
        if isinstance(record, Mapping) and record.get("target_attempted") is not False:
            attempted.add(str(target_id))
    return attempted


def _resident_result_record(
    result_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Load a compact result and its raw sidecar without duplicating traces."""
    value = _load_json(result_path)
    record = value.get("record")
    record = record if isinstance(record, Mapping) else value.get("result")
    if not isinstance(record, Mapping):
        return value, None
    record_value = dict(record)
    raw_reference = record_value.get("raw_evidence_path")
    if isinstance(raw_reference, str) and raw_reference:
        try:
            # result/.../target-N/job.json -> method root is seven parents up.
            method_root = result_path.parents[7]
            run_root = method_root.parent.parent
            raw_path = Path(raw_reference)
            if raw_path.is_absolute():
                raw_path = run_root / raw_path.relative_to(
                    Path("/opt/runs") / run_root.name
                )
            else:
                raw_path = method_root / raw_path
            raw = _load_json(raw_path)
        except (IndexError, OSError, TypeError, ValueError):
            raw = {}
        if isinstance(raw, Mapping):
            for name in ("executions", "observations"):
                if name in raw and name not in record_value:
                    record_value[name if name != "executions" else "raw_executions"] = raw[name]
            raw_executions = raw.get("executions")
            if isinstance(raw_executions, Mapping):
                record_value.setdefault("raw_executions", raw_executions)
    return value, record_value


def _raw_catalog_entry(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract one target-job RV catalog entry from resident raw executions."""
    executions = record.get("raw_executions")
    if not isinstance(executions, Mapping):
        direct = record.get("rv_opcode_catalog_coverage")
        return dict(direct) if isinstance(direct, Mapping) else None
    entries: list[Mapping[str, Any]] = []
    for execution in executions.values():
        if not isinstance(execution, Mapping):
            continue
        coverage = execution.get("coverage")
        coverage = coverage if isinstance(coverage, Mapping) else {}
        entry = coverage.get("rv_opcode_catalog_coverage")
        if isinstance(entry, Mapping):
            entries.append(entry)
    if not entries:
        return None
    if len(entries) == 1:
        result = dict(entries[0])
        result.setdefault("case_count", 1)
        return result
    # A candidate pair has original and variant executions but represents one
    # replay job. Union both phase metrics, then restore the job-level count.
    target = str(record.get("target_id") or "target")
    aggregate = aggregate_opcode_catalog_summary(
        {"targets": [{
            "method": "raw-only", "route": "target-replay",
            "lane": "all", "stratum": "all", "target": target,
            "cases": len(entries),
        }]},
        [{
            "method": "raw-only", "route": "target-replay", "lane": "all",
            "stratum": "all", "target": target,
            "simulator_coverage": {"rv_opcode_catalog_coverage": entry},
        } for entry in entries],
    )
    rows = aggregate.get("targets")
    row = rows[0] if isinstance(rows, list) and rows else {}
    result = row.get("rv_opcode_catalog_coverage") \
        if isinstance(row, Mapping) else None
    if not isinstance(result, Mapping):
        return None
    result = dict(result)
    result["case_count"] = 1
    result["aggregation_scope"] = "target-job-original-variant-union"
    return result


def _framework_raw_catalog_entries(
    run_root: Path, method: str, target_id: str,
) -> tuple[list[Mapping[str, Any]], int]:
    """Collect catalog entries directly from resident raw-evidence sidecars."""
    method_root = run_root / "framework" / _safe(method)
    entries: list[Mapping[str, Any]] = []
    expected = 0
    raw_root = method_root / "target-queues" / _safe(target_id) / "raw" / "cases"
    for path in sorted(raw_root.rglob("*.json")) if raw_root.is_dir() else ():
        if path.name.endswith(".raw-artifact-manifest.json") \
                or not (path.name == "baseline.json" or path.name.startswith("candidate-")):
            continue
        value = _load_json(path)
        if value.get("target_id") != target_id:
            continue
        raw_executions = value.get("executions")
        record = {
            "target_id": target_id,
            "raw_executions": raw_executions,
        }
        if not isinstance(raw_executions, Mapping):
            continue
        expected += 1
        entry = _raw_catalog_entry(record)
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries, expected


def _framework_raw_case_catalog_entries(
    case_dir: Path, method: str, target_id: str,
) -> tuple[list[Mapping[str, Any]], int]:
    """Read only one case's raw RV sidecars; avoid huge compact result files."""
    try:
        method_root = case_dir.resolve().parents[3]
        case_relative = case_dir.resolve().relative_to(method_root)
    except (OSError, ValueError, IndexError):
        return [], 0
    raw_root = (
        method_root / "target-queues" / _safe(target_id) / "raw" / "cases"
        / case_relative
    )
    entries: list[Mapping[str, Any]] = []
    expected = 0
    for path in sorted(raw_root.glob("*.json")) if raw_root.is_dir() else ():
        if path.name.endswith(".raw-artifact-manifest.json") \
                or not (path.name == "baseline.json" or path.name.startswith("candidate-")):
            continue
        value = _load_json(path)
        if value.get("target_id") != target_id:
            continue
        raw_executions = value.get("executions")
        if not isinstance(raw_executions, Mapping):
            expected += 1
            continue
        expected += 1
        entry = _raw_catalog_entry({
            "target_id": target_id, "raw_executions": raw_executions,
        })
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries, expected


def _framework_live_catalog_entry(
    *, case_dir: Path, method: str, target_id: str,
    lane_id: int, case_index: int, target_config: Mapping[str, Any],
    target_data: Mapping[str, Any], base_campaign: Mapping[str, Any],
    record_key: str, baseline: bool, progress_root: Path,
    source_path: Path | None = None, record: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if source_path is not None:
        try:
            stat = source_path.stat()
        except OSError:
            return None
        identity = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
        }
    else:
        identity = {"record_index": record_key}
    identity.update({
        "catalog_id": OPCODE_CATALOG_ID,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
    })
    marker = (
        progress_root / "index" / _safe(method) / "framework-opcode"
        / _safe(target_id) / f"lane-{lane_id:02d}-case-{case_index:06d}"
        / f"{_safe(record_key)}.json"
    )
    cached = _load_json(marker)
    entry = cached.get("rv_opcode_catalog_coverage")
    if (
        cached.get("schema_version") == "rq1-opcode-evidence-v1"
        and cached.get("identity") == identity
        and isinstance(entry, Mapping)
        and _valid_catalog_metric_status(entry, "GenOpcodeCov") is not None
        and _valid_catalog_metric_status(entry, "ExecOpcodeCov") is not None
    ):
        return dict(entry)

    if source_path is not None:
        sidecar, loaded_record = _resident_result_record(source_path)
        record = loaded_record
        if (
            sidecar.get("schema_version") != "rq1-target-replay-result-v1"
            or sidecar.get("target_id") != target_id
            or not isinstance(record, Mapping)
        ):
            return None
    if not isinstance(record, Mapping):
        return None

    # Resident workers already compute catalog metrics while the ELF and full
    # PC trace are still available. Prefer that durable evidence; replaying a
    # target result through the campaign finalizer loses the artifact fields
    # that raw-only deliberately omits from the producer record.
    raw_entry = _raw_catalog_entry(record)
    if (
        isinstance(raw_entry, Mapping)
        and _valid_catalog_metric_status(raw_entry, "GenOpcodeCov") is not None
        and _valid_catalog_metric_status(raw_entry, "ExecOpcodeCov") is not None
    ):
        result = dict(raw_entry)
        _write_json(marker, {
            "schema_version": "rq1-opcode-evidence-v1",
            "identity": identity,
            "rv_opcode_catalog_coverage": result,
        })
        return result

    from framework.framework_coverage import finalize_framework_coverage

    campaign = dict(base_campaign)
    campaign["method"] = method
    if not isinstance(campaign.get("reference"), list):
        campaign["reference"] = []
    if baseline:
        baseline_record = dict(record)
        if not _observations(baseline_record.get("observations")):
            baseline_record["observations"] = [{
                "target_attempted": False, "trace_available": False,
            }]
        campaign["target"] = []
        campaign["target_baseline"] = baseline_record
        target_artifact = baseline_record.get("target_artifact")
        artifacts = list(campaign.get("generation_artifacts", ()))
        if isinstance(target_artifact, Mapping):
            artifacts.append(dict(target_artifact))
        campaign["generation_artifacts"] = artifacts
    else:
        candidate = dict(record)
        comparison = candidate.get("comparison")
        comparison = dict(comparison) if isinstance(comparison, Mapping) else {}
        observations = comparison.get("observations")
        observations = dict(observations) if isinstance(observations, Mapping) else {}
        if not any(_observations(value) for value in observations.values()):
            if isinstance(candidate.get("target_original_artifact"), Mapping):
                observations["original"] = [{
                    "target_attempted": False, "trace_available": False,
                }]
            if isinstance(candidate.get("artifact"), Mapping):
                observations["variant"] = [{
                    "target_attempted": False, "trace_available": False,
                }]
        candidate["comparison"] = {**comparison, "observations": observations}
        campaign["target"] = [candidate]
        campaign["target_baseline"] = None

    summary_path = (
        case_dir / "coverage" / "post-execution"
        / f"{_safe(target_id)}-{_safe(record_key)}.json"
    )
    config = dict(target_config)
    config["target"] = dict(target_data)
    config["target_id"] = target_id
    config["method"] = method
    config["summary_path"] = str(summary_path)
    # Collect case-level guest opcode evidence here; SimSrcCov remains owned by
    # the method-level batch collector.
    config["coverage_batch_summary_path"] = str(
        config.get("coverage_batch_summary_path") or summary_path.with_suffix(".source.json")
    )
    config["coverage_batch_raw_dir"] = str(
        config.get("coverage_batch_raw_dir") or summary_path.parent / "deferred-source"
    )
    try:
        finalize_framework_coverage(case_dir, campaign, config)
        summary = _load_json(summary_path)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError):
        return None
    finally:
        summary_path.unlink(missing_ok=True)
    rows = summary.get("targets")
    selected = next((item for item in rows if isinstance(item, Mapping)
                     and item.get("target") == target_id), None) \
        if isinstance(rows, list) else None
    entry = selected.get("rv_opcode_catalog_coverage") \
        if isinstance(selected, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    result = dict(entry)
    _write_json(marker, {
        "schema_version": "rq1-opcode-evidence-v1",
        "identity": identity,
        "rv_opcode_catalog_coverage": result,
    })
    return result


def _framework_live_catalog_entries(
    *, run_root: Path, progress_root: Path, method: str,
    lane_id: int, case_index: int, case_dir: Path, target_id: str,
    target_configs: Mapping[str, Any], campaign: Mapping[str, Any],
    live_campaign: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    item = target_configs.get(target_id)
    if not isinstance(item, Mapping):
        return [], 0
    target_config = item.get("coverage_config")
    target_config = dict(target_config) if isinstance(target_config, Mapping) else dict(item)
    target_data = target_config.get("target")
    target_data = dict(target_data) if isinstance(target_data, Mapping) else {}
    target_data.setdefault("id", target_id)
    target_data.setdefault("kind", target_config.get("kind", "qemu"))
    target_data.setdefault("execution_model", target_config.get("execution_model"))
    target_data.setdefault("stratum", target_config.get("stratum"))
    target_data.setdefault("privilege_mode", "user")
    target_data["stratum"] = _target_stratum(target_data)
    lane = str(target_config.get("lane") or target_data.get("lane") or "rv64i/lp64")
    target_data["isa_profile"] = str(
        target_config.get("isa_profile") or target_config.get("isa")
        or target_data.get("isa_profile") or lane.split("/", 1)[0]
    ).strip().lower()
    target_data["coverage_binary"] = str(target_config.get("binary_path") or "")
    source = target_config.get("source_coverage")
    target_data["source_coverage"] = dict(source) if isinstance(source, Mapping) else {}
    target_names = list(target_configs)
    raw_entries, raw_expected = _framework_raw_case_catalog_entries(
        case_dir, method, target_id,
    )
    if raw_expected:
        return [dict(entry) for entry in raw_entries], raw_expected
    try:
        target_slot = target_names.index(target_id)
    except ValueError:
        target_slot = 0
    result_dir = (
        case_dir / "target-replay-queue" / "results" / f"target-{target_slot:03d}"
    )
    entries: list[dict[str, Any]] = []
    expected = 0
    if result_dir.is_dir():
        files = []
        baseline_path = result_dir / "baseline.json"
        if baseline_path.is_file():
            files.append((baseline_path, True, "baseline"))
        files.extend((path, False, path.stem)
                     for path in sorted(result_dir.glob("candidate-*.json")))
        for path, is_baseline, record_key in files:
            if not _within(path, run_root):
                expected += 1
                continue
            entry = _framework_live_catalog_entry(
                case_dir=case_dir, method=method, target_id=target_id,
                lane_id=lane_id, case_index=case_index,
                target_config=target_config, target_data=target_data,
                base_campaign=live_campaign or campaign,
                record_key=record_key, baseline=is_baseline,
                progress_root=progress_root, source_path=path,
            )
            if isinstance(entry, dict):
                entries.append(entry)
                count = entry.get("case_count", 1)
                expected += count if type(count) is int and count >= 0 else 1
            else:
                expected += 1
        target_summaries = campaign.get("target_summaries")
        target_summaries = target_summaries if isinstance(target_summaries, Mapping) else {}
        target_progress = target_summaries.get(target_id)
        if isinstance(target_progress, Mapping):
            declared_records = (
                _metric_count(target_progress.get("record_count"))
                + _metric_count(target_progress.get("baseline_count"))
            )
            expected += max(0, declared_records - len(files))
        if files:
            return entries, expected

    records_by_target = live_campaign.get("target_records_by_target")
    records_by_target = records_by_target if isinstance(records_by_target, Mapping) else {}
    records = records_by_target.get(target_id)
    if not isinstance(records, (list, tuple)):
        records_by_target = campaign.get("target_records_by_target")
        records_by_target = records_by_target if isinstance(records_by_target, Mapping) else {}
        records = records_by_target.get(target_id)
    if not isinstance(records, (list, tuple)) and len(target_names) == 1:
        records = live_campaign.get("target", ())
    baseline_by_target = live_campaign.get("target_baselines_by_target")
    baseline_by_target = baseline_by_target if isinstance(baseline_by_target, Mapping) else {}
    baseline_record = baseline_by_target.get(target_id)
    if not isinstance(baseline_record, Mapping) and len(target_names) == 1:
        candidate = live_campaign.get("target_baseline")
        baseline_record = candidate if isinstance(candidate, Mapping) else None
    if not isinstance(baseline_record, Mapping):
        baseline_by_target = campaign.get("target_baselines_by_target")
        baseline_by_target = baseline_by_target if isinstance(baseline_by_target, Mapping) else {}
        candidate = baseline_by_target.get(target_id)
        baseline_record = candidate if isinstance(candidate, Mapping) else None

    if isinstance(baseline_record, Mapping):
        entry = _framework_live_catalog_entry(
            case_dir=case_dir, method=method, target_id=target_id,
            lane_id=lane_id, case_index=case_index,
            target_config=target_config, target_data=target_data,
            base_campaign=live_campaign or campaign,
            record_key="checkpoint-baseline", baseline=True,
            progress_root=progress_root, record=baseline_record,
        )
        if isinstance(entry, dict):
            entries.append(entry)
            count = entry.get("case_count", 1)
            expected += count if type(count) is int and count >= 0 else 1
        else:
            expected += 1
    for record_index, record in enumerate(records or ()):
        if not isinstance(record, Mapping):
            expected += 1
            continue
        entry = _framework_live_catalog_entry(
            case_dir=case_dir, method=method, target_id=target_id,
            lane_id=lane_id, case_index=case_index,
            target_config=target_config, target_data=target_data,
            base_campaign=live_campaign or campaign,
            record_key=f"checkpoint-candidate-{record_index:06d}",
            baseline=False, progress_root=progress_root, record=record,
        )
        if isinstance(entry, dict):
            entries.append(entry)
            count = entry.get("case_count", 1)
            expected += count if type(count) is int and count >= 0 else 1
        else:
            expected += 1
    return entries, expected


def _framework_inputs(
    run_root: Path, method: str, targets: list[Mapping[str, Any]], progress_root: Path,
    *, pending_catalogs: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    target_by_id = {str(target.get("id")): target for target in targets}
    target_ids = set(target_by_id)
    stats = {target_id: {"attempts": 0, "profiles": 0, "data_profiles": 0,
                         "session_id": None, "session_launcher": None,
                         "catalog_entries": [], "catalog_expected_cases": 0,
                         "statuses": Counter()}
             for target_id in target_by_id}
    safe_method = _safe(method)
    method_dir = run_root / "framework" / safe_method
    config_dir = run_root / "framework-coverage-configs" / safe_method
    progress_cases: dict[tuple[int, int], Mapping[str, Any]] = {}
    live_campaigns: dict[tuple[int, int], Mapping[str, Any]] = {}
    in_progress_case_dirs: set[tuple[int, int]] = set()
    progress_campaigns = 0
    totals = Counter()
    target_attempt_counts = Counter()
    by_target_statuses = {target_id: Counter() for target_id in target_by_id}
    # ponytail: this rereads persisted case JSON during post-run aggregation (O(completed cases)); use a cursor only if profiling shows this is costly.
    for lane_dir in sorted((method_dir / "lanes").glob("lane-*")):
        lane_path = lane_dir / "lane-result.json"
        lane = _load_json(lane_path)
        lane_id = lane.get("lane_id")
        if type(lane_id) is not int:
            match = re.search(r"lane-(\d+)$", lane_dir.name)
            lane_id = int(match.group(1)) if match else None
        if lane_id is None:
            continue
        cases = lane.get("cases")
        cases = cases if isinstance(cases, list) else []
        case_dirs: dict[int, Path] = {}
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            case_index = case.get("case_index")
            case_dir_value = case.get("case_dir")
            if type(case_index) is not int:
                continue
            progress_cases[(lane_id, case_index)] = case
            if isinstance(case_dir_value, str):
                case_dir = run_root / case_dir_value
                if _within(case_dir, run_root) and case_dir.is_dir():
                    case_dirs[case_index] = case_dir
        lane_cases_dir = lane_dir / "cases"
        for case_dir in sorted(lane_cases_dir.glob("case-*")):
            match = re.fullmatch(r"case-(\d+)", case_dir.name)
            if match:
                case_dirs.setdefault(int(match.group(1)), case_dir)

        for case_index, case_dir in sorted(case_dirs.items()):
            if not _within(case_dir, run_root):
                continue
            case_key = (lane_id, case_index)
            if case_key not in progress_cases:
                in_progress_case_dirs.add(case_key)
            final_campaign = _load_json(case_dir / "campaign-result.json")
            checkpoint_campaign = {}
            if final_campaign:
                campaign = final_campaign
                live_campaign = final_campaign
            else:
                progress_campaign = _load_json(case_dir / "campaign-progress.json")
                checkpoint = _load_json(case_dir / "reference-chain-checkpoint.json")
                checkpoint_value = checkpoint.get("campaign")
                checkpoint_campaign = dict(checkpoint_value) \
                    if isinstance(checkpoint_value, Mapping) else {}
                has_progress_summary = (
                    progress_campaign.get("schema_version")
                    == "rq1-framework-campaign-progress-v1"
                    and isinstance(progress_campaign.get("counts"), Mapping)
                    and (
                        isinstance(progress_campaign.get("target_summaries"), Mapping)
                        or isinstance(progress_campaign.get("target_records_by_target"), Mapping)
                    )
                )
                campaign = progress_campaign if has_progress_summary else checkpoint_campaign
                live_campaign = checkpoint_campaign or campaign
            attempted_targets = _framework_attempted_targets(campaign, target_ids)
            if not final_campaign:
                attempted_targets |= _framework_attempted_targets(
                    live_campaign, target_ids,
                )
            attempted_targets |= _framework_result_attempted_targets(
                case_dir, target_ids,
            )
            if not attempted_targets:
                continue
            progress_campaigns += 1
            if case_key not in progress_cases:
                live_campaigns[case_key] = campaign
            config_path = config_dir / f"lane-{lane_id:02d}-case-{case_index:06d}.json"
            configs = _load_json(config_path)
            target_configs = configs.get("targets")
            if not isinstance(target_configs, Mapping):
                configured_target = configs.get("target_id")
                target_configs = (
                    {configured_target: configs}
                    if isinstance(configured_target, str) else {}
                )
            records_by_target = live_campaign.get("target_records_by_target")
            records_by_target = records_by_target if isinstance(records_by_target, Mapping) else {}
            baselines_by_target = live_campaign.get("target_baselines_by_target")
            baselines_by_target = baselines_by_target if isinstance(baselines_by_target, Mapping) else {}
            coverage_summary = _load_json(case_dir / "coverage" / "summary.json")
            summary_rows = coverage_summary.get("targets")
            summary_rows = summary_rows if isinstance(summary_rows, list) else []
            progress_target_summaries = campaign.get("target_summaries")
            progress_target_summaries = (
                progress_target_summaries
                if isinstance(progress_target_summaries, Mapping) else {}
            )
            for target_id in attempted_targets:
                stats[target_id]["attempts"] += 1
                target_records = records_by_target.get(target_id)
                target_records = target_records if isinstance(target_records, list) else []
                baseline = baselines_by_target.get(target_id)
                target_progress = progress_target_summaries.get(target_id)
                target_progress = target_progress if isinstance(target_progress, Mapping) else {}
                progress_counts = target_progress.get("counts")
                progress_counts = progress_counts if isinstance(progress_counts, Mapping) else {}
                status_counts = target_progress.get("status_counts")
                status_counts = status_counts if isinstance(status_counts, Mapping) else {}
                expected_case_count = (
                    int(target_progress.get("record_count", 0) or 0)
                    + int(target_progress.get("baseline_count", 0) or 0)
                    if target_progress else
                    len(target_records) + int(isinstance(baseline, Mapping))
                )
                if target_progress:
                    target_attempt_counts[target_id] += _metric_count(
                        progress_counts.get("target_attempted"),
                    )
                    stats[target_id]["statuses"].update(status_counts)
                    by_target_statuses[target_id].update(status_counts)
                else:
                    target_attempt_counts[target_id] += sum(
                        record.get("target_attempted") is not False
                        for record in target_records if isinstance(record, Mapping)
                    ) + int(
                        isinstance(baseline, Mapping)
                        and baseline.get("target_attempted") is not False
                    )
                    for record in [*target_records,
                                   *([baseline] if isinstance(baseline, Mapping) else [])]:
                        status = record.get("status") or record.get("target_result_status")
                        if isinstance(status, str) and status:
                            stats[target_id]["statuses"][status] += 1
                            by_target_statuses[target_id][status] += 1
                selected_summary = next((row for row in summary_rows
                                         if isinstance(row, Mapping)
                                         and row.get("target") == target_id), None)
                entry = selected_summary.get("rv_opcode_catalog_coverage") \
                    if final_campaign and isinstance(selected_summary, Mapping) else None
                if isinstance(entry, Mapping):
                    stats[target_id]["catalog_entries"].append(entry)
                    entry_count = entry.get("case_count", 1)
                    if type(entry_count) is int and entry_count >= 0:
                        expected_case_count = max(expected_case_count, entry_count)
                else:
                    live_context = {
                        "run_root": run_root, "progress_root": progress_root,
                        "method": method, "lane_id": lane_id,
                        "case_index": case_index, "case_dir": case_dir,
                        "target_id": target_id, "target_configs": target_configs,
                        "campaign": campaign, "live_campaign": live_campaign,
                    }
                    if pending_catalogs is None:
                        live_entries, live_expected = _framework_live_catalog_entries(
                            **live_context,
                        )
                        stats[target_id]["catalog_entries"].extend(live_entries)
                        expected_case_count = max(expected_case_count, live_expected)
                    else:
                        pending_catalogs.setdefault(target_id, []).append({
                            **live_context,
                            "base_expected": expected_case_count,
                        })
                stats[target_id]["catalog_expected_cases"] += expected_case_count
                target_item = target_configs.get(target_id)
                coverage = target_item.get("coverage_config") \
                    if isinstance(target_item, Mapping) else None
                if not isinstance(coverage, Mapping) and isinstance(target_item, Mapping):
                    coverage = target_item
                if not isinstance(coverage, Mapping):
                    continue
                candidate_session = coverage.get("coverage_batch_session_id")
                if isinstance(candidate_session, str) and candidate_session:
                    stats[target_id]["session_id"] = candidate_session
                    source_config = coverage.get("source_coverage")
                    if isinstance(source_config, Mapping):
                        launcher_value = source_config.get("launcher")
                        if isinstance(launcher_value, str):
                            stats[target_id]["session_launcher"] = launcher_value
                    continue
                raw_value = coverage.get("raw_dir")
                if not isinstance(raw_value, str) or not raw_value:
                    continue
                source = Path(raw_value)
                if source.is_absolute():
                    try:
                        source = run_root / source.relative_to(
                            Path("/opt/runs") / run_root.name
                        )
                    except ValueError:
                        pass
                else:
                    source = run_root / source
                if not _within(source, run_root):
                    continue
                batch_id = str(coverage.get("coverage_batch_id") or "batch")
                key = f"framework-{lane_id:02d}-{case_index:06d}-{_safe(batch_id)}"
                marker = progress_root / "index" / method / "framework" / _safe(target_id) / f"{key}.json"
                destination = progress_root / "inputs" / _safe(target_id) / "raw" / "profiles" / key
                collector = str((target_by_id[target_id].get("source_coverage") or {}).get("collector") or "")
                stats[target_id]["profiles"] += 1
                if _stage_profile(
                    source=source, destination=destination, marker=marker,
                    collector=collector, run_root=run_root,
                ):
                    stats[target_id]["data_profiles"] += 1
    count_fields = (
        "steps", "mh_attempts", "accepted", "rejected", "target_attempted",
        "target_tested", "target_gap", "target_mismatch_candidate", "reference_valid",
    )
    for case in progress_cases.values():
        for name in count_fields:
            value = case.get(name)
            if value is None and isinstance(case.get("counts"), Mapping):
                value = case["counts"].get("mh_rejected" if name == "rejected" else name)
            totals[name] += _metric_count(value)
    for campaign in live_campaigns.values():
        counts = campaign.get("counts")
        counts = counts if isinstance(counts, Mapping) else {}
        for name in count_fields:
            count_name = "mh_rejected" if name == "rejected" else name
            value = counts.get(count_name)
            if name == "steps" and value is None:
                value = campaign.get("reference_chain_steps")
            totals[name] += _metric_count(value)
    progress = {
        "kind": "ours-framework",
        "completed_cases": len(progress_cases),
        "in_progress_cases": len(in_progress_case_dirs),
        "in_progress_cases_with_campaign": len(live_campaigns),
        "cases_with_target_campaign": progress_campaigns,
        "steps": totals["steps"],
        "mh_attempts": totals["mh_attempts"],
        "accepted": totals["accepted"],
        "rejected": totals["rejected"],
        "target_attempted": totals["target_attempted"],
        "target_tested": totals["target_tested"],
        "target_gap": totals["target_gap"],
        "target_mismatch_candidate": totals["target_mismatch_candidate"],
        "reference_valid": totals["reference_valid"],
        "targets": {
            target_id: {
                "attempts": target_attempt_counts[target_id],
                "statuses": dict(sorted(by_target_statuses[target_id].items())),
            }
            for target_id in target_by_id
        },
    }
    return stats, progress


def _dependency_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(os.environ.get("RQ1_DEPS", "/path/to/deps")) / path


def _dotnet_session_snapshot(
    *, launcher_value: str | None, session_id: str | None, raw_cache: Path,
) -> tuple[bool, str | None]:
    launcher = _dependency_path(launcher_value)
    if launcher is None or not launcher.is_file() or not session_id:
        return False, "dotnet-coverage-session-unavailable"
    raw_cache.mkdir(parents=True, exist_ok=True)
    snapshot_dir = raw_cache / "live-session"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dotnet-progress-") as temporary:
        work = Path(temporary)
        binary_report = work / "session.coverage"
        cobertura_report = work / "session.cobertura.xml"
        environment = os.environ.copy()
        environment["TMPDIR"] = "/tmp"
        runtime = environment.get("RQ1_DOTNET_RUNTIME")
        if runtime:
            environment["DOTNET_ROOT"] = runtime
            environment["PATH"] = runtime + os.pathsep + environment.get("PATH", "")
        try:
            ipc_timeout_ms = int(float(
                environment.get("RQ1_BACKEND_TIMEOUT_SECONDS", "180")
            ) * 1000)
        except (OverflowError, TypeError, ValueError):
            ipc_timeout_ms = 180_000
        ipc_timeout_ms = min(180_000, max(1_000, ipc_timeout_ms))
        try:
            snapshot = subprocess.run(
                [str(launcher), "snapshot", session_id, "--nologo",
                 "--timeout", str(ipc_timeout_ms), "--output", str(binary_report)],
                capture_output=True, text=True,
                timeout=ipc_timeout_ms / 1000 + 5, check=False, env=environment,
            )
            if snapshot.returncode != 0 or not binary_report.is_file():
                detail = (snapshot.stderr or snapshot.stdout or "snapshot-command-failed").strip()
                return False, f"snapshot-returncode={snapshot.returncode}: {detail[-500:]}"
            merge = subprocess.run(
                [str(launcher), "merge", str(binary_report), "--nologo",
                 "--output", str(cobertura_report), "--output-format", "cobertura"],
                capture_output=True, text=True, timeout=600, check=False, env=environment,
            )
            if merge.returncode != 0 or not cobertura_report.is_file():
                detail = (merge.stderr or merge.stdout or "cobertura-merge-failed").strip()
                return False, f"cobertura-merge-returncode={merge.returncode}: {detail[-500:]}"
            destination = snapshot_dir / "renode.cobertura.xml"
            temporary_output = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(cobertura_report, temporary_output)
            os.replace(temporary_output, destination)
            return True, None
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"{type(error).__name__}: {error}"


def _identity_digest(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _case_profile_dirs(
    raw_cache: Path, collector: str, keys: set[str] | None = None,
) -> dict[str, Path]:
    root = raw_cache / "profiles"
    if not root.is_dir():
        return {}
    paths = (root / key for key in sorted(keys)) if keys is not None else sorted(root.iterdir())
    result = {}
    for path in paths:
        if path.is_dir() and not path.is_symlink() and _profile_files(path, collector):
            result[path.name] = path
    return result


def _profile_input_stamp(raw_cache: Path) -> str:
    """Identify staged, closed profile units using metadata only."""
    # ponytail: staged case directories are immutable after close; final sealing
    # still hashes and replays full inputs if that storage contract changes.
    root = raw_cache / "profiles"
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(
            f"{path.name}\0{stat.st_dev}\0{stat.st_ino}\0{stat.st_size}\0"
            f"{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _file_stamp(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }


def _source_collection_cache_key(
    *, target: Mapping[str, Any], source_spec: Mapping[str, Any],
    collector: str, raw_cache: Path, binary: Path | None, attempts: int,
    session_id: str | None, session_snapshot: Path | None,
) -> str:
    return _identity_digest({
        "schema_version": "rq1-source-collection-cache-v2",
        "target": target.get("id"),
        "collector": collector,
        "source_spec": dict(source_spec),
        "coverage_binary": target.get("coverage_binary"),
        "coverage_binary_stamp": _file_stamp(binary),
        "profile_input_stamp": _profile_input_stamp(raw_cache),
        "session_id": session_id,
        "live_session_sequence": attempts if session_id else None,
        "session_snapshot": _file_stamp(session_snapshot),
    })


def _persist_incremental_state(
    *, target_dir: Path, profile: Path, input_keys: list[str],
    input_statuses: Mapping[str, str], identity_digest: str,
    collection: Mapping[str, Any],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cached_profile = target_dir / "cumulative.lcov.info"
    temporary = cached_profile.with_name(cached_profile.name + f".{os.getpid()}.tmp")
    shutil.copy2(profile, temporary)
    os.replace(temporary, cached_profile)
    profile_stat = cached_profile.stat()
    _write_json(target_dir / "incremental.json", {
        "schema_version": "rq1-sim-srccov-incremental-v3",
        "collector": collection.get("collector"),
        "identity_digest": identity_digest,
        "input_keys": input_keys,
        "input_statuses": dict(input_statuses),
        "input_sequence": len(input_keys),
        "profile_size": profile_stat.st_size,
        "profile_mtime_ns": profile_stat.st_mtime_ns,
        "profile_ctime_ns": profile_stat.st_ctime_ns,
        "profile_inode": profile_stat.st_ino,
        "collection": dict(collection),
    })


def _incremental_source_collection(
    *, temporary: Path, target_copy: dict[str, Any], binary: Path | None,
    raw_cache: Path, current_identity: dict[str, Any], identity_path: Path,
    profile_path: Path,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Collect new case shards and merge their LCOV with the persisted union."""
    collector = str((target_copy.get("source_coverage") or {}).get("collector") or "")
    if collector != "gcov":
        return None, 0, "collector-not-proven-incremental"
    current_digest = _identity_digest(current_identity)
    target_dir = raw_cache.parents[2] / "targets" / _safe(target_copy.get("id"))
    state_path = target_dir / "incremental.json"
    state = _load_json(state_path)
    old_keys = state.get("input_keys")
    old_keys = old_keys if isinstance(old_keys, list) and all(
        isinstance(key, str) for key in old_keys
    ) else []
    cached_profile = target_dir / "cumulative.lcov.info"
    try:
        cached_stat = cached_profile.stat()
    except OSError:
        cached_stat = None
    cache_valid = (
        state.get("schema_version") == "rq1-sim-srccov-incremental-v3"
        and state.get("collector") == collector
        and state.get("identity_digest") == current_digest
        and bool(old_keys)
        and len(set(old_keys)) == len(old_keys)
        and state.get("input_sequence") == len(old_keys)
        and isinstance(state.get("input_statuses"), dict)
        and set(state["input_statuses"]) == set(old_keys)
        and all(status in {"observed", "partial"}
                for status in state["input_statuses"].values())
        and cached_stat is not None
        and state.get("profile_size") == cached_stat.st_size
        and state.get("profile_mtime_ns") == cached_stat.st_mtime_ns
        and state.get("profile_ctime_ns") == cached_stat.st_ctime_ns
        and state.get("profile_inode") == cached_stat.st_ino
    )
    committed_keys = list(old_keys) if cache_valid else []
    input_statuses = dict(state.get("input_statuses") or {}) if cache_valid else {}
    profile_root = raw_cache / "profiles"
    try:
        available_keys = {
            path.name for path in profile_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        } if profile_root.is_dir() else set()
    except OSError:
        available_keys = set()
    if not cache_valid:
        return (
            None, len(_case_profile_dirs(raw_cache, collector)),
            "incremental-state-seed-required",
        )
    if not set(committed_keys) <= available_keys:
        return (
            None, len(_case_profile_dirs(raw_cache, collector)),
            "incremental-input-set-shrank",
        )
    # Only a valid state may consume new closed case profile directories.
    candidate_keys = available_keys - set(committed_keys)
    profile_dirs = _case_profile_dirs(raw_cache, collector, candidate_keys)
    new_keys = sorted(profile_dirs)
    shard_paths: list[Path] = []
    shard_collections: list[dict[str, Any]] = []
    successful_keys: list[str] = []
    failed_profiles: list[dict[str, str]] = []
    spec = target_copy["source_coverage"]
    for key in new_keys:
        shard_dir = temporary / "case-shards" / key
        shard_profile = shard_dir / "case.lcov.info"
        shard_identity = shard_dir / "identity.json"
        shard_target = deepcopy(target_copy)
        shard_spec = deepcopy(spec)
        shard_spec.update(profile=str(shard_profile), identity=str(shard_identity))
        shard_target["source_coverage"] = shard_spec
        try:
            collection = collect_source_coverage_batch(
                shard_dir, shard_target, binary, profile_dirs[key],
                expected_coverage_cases=1,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            failed_profiles.append({
                "profile_id": key,
                "reason": f"incremental-case-collection-error:{type(error).__name__}",
            })
            continue
        identity = _load_json(shard_identity)
        if _identity_digest(identity) != current_digest:
            return None, len(available_keys), "incremental-identity-changed"
        if collection.get("status") not in {"observed", "partial"} \
                or not shard_profile.is_file():
            failed_profiles.append({
                "profile_id": key,
                "reason": str(collection.get("reason") or collection.get("status")
                              or "incremental-case-collection-failed"),
            })
            continue
        shard_paths.append(shard_profile)
        shard_collections.append(collection)
        successful_keys.append(key)
        input_statuses[key] = str(collection.get("status"))

    if not new_keys and cache_valid:
        shutil.copy2(cached_profile, profile_path)
        collection = dict(state.get("collection") or {})
    elif shard_paths:
        merged = temporary / "merged-cumulative.lcov.info"
        sources = ([cached_profile] if cache_valid else []) + shard_paths
        code, error = _merge_lcov_profiles(sources, merged)
        if code or not merged.is_file():
            failed_profiles.extend({
                "profile_id": key,
                "reason": f"incremental-lcov-merge-failed:{error}",
            } for key in successful_keys)
            successful_keys = []
            if cache_valid:
                shutil.copy2(cached_profile, profile_path)
                collection = dict(state.get("collection") or {})
            else:
                collection = {}
        else:
            shutil.copy2(merged, profile_path)
            collection = dict(shard_collections[-1])
    elif cache_valid:
        shutil.copy2(cached_profile, profile_path)
        collection = dict(state.get("collection") or {})
    else:
        collection = {}

    identity_path.write_text(
        json.dumps(current_identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if profile_path.is_file():
        committed_keys = sorted(set(committed_keys + successful_keys))
        failed = bool(failed_profiles)
        partial_inputs = any(status == "partial" for status in input_statuses.values())
        # Avoid rehashing the growing cumulative profile each hour. The shard
        # collector validates fresh inputs; this profile is only the union.
        collection.pop("profile_sha256", None)
        collection.update({
            "status": "partial" if failed or partial_inputs else "observed",
            "collector": collector,
            "profile": profile_path.name,
            "identity": identity_path.name,
            "raw_dir": str(raw_cache),
            "incremental": True,
            "input_sequence": len(committed_keys),
            "pending_profiles": failed_profiles,
        })
        if failed:
            collection["reason"] = "incremental-profile-pending"
        if successful_keys:
            _persist_incremental_state(
                target_dir=target_dir, profile=profile_path,
                input_keys=committed_keys, input_statuses=input_statuses,
                identity_digest=current_digest, collection=collection,
            )
        elif failed_profiles:
            previous_collection = state.get("collection")
            if previous_collection != collection:
                state.update({
                    "input_statuses": input_statuses,
                    "input_sequence": len(committed_keys),
                    "profile_size": cached_stat.st_size,
                    "profile_mtime_ns": cached_stat.st_mtime_ns,
                    "profile_ctime_ns": cached_stat.st_ctime_ns,
                    "profile_inode": cached_stat.st_ino,
                    "collection": dict(collection),
                })
                _write_json(state_path, state)
    else:
        collection = {
            "status": "gap", "collector": collector,
            "reason": (
                "incremental-profile-pending" if failed_profiles
                else "no-closed-case-profiles"
            ),
            "incremental": True, "input_sequence": 0,
            "pending_profiles": failed_profiles,
        }
    return collection, len(committed_keys), None


def _collect_target(
    *, run_root: Path, target: Mapping[str, Any], progress_root: Path,
    attempts: int, profiles: int, data_profiles: int,
    session_id: str | None = None, session_launcher: str | None = None,
    force_full_batch: bool = False,
) -> dict[str, Any]:
    target_id = str(target.get("id") or "")
    collector = str((target.get("source_coverage") or {}).get("collector") or "")
    raw_cache = progress_root / "inputs" / _safe(target_id) / "raw"
    live_session = bool(session_id and collector == "dotnet")
    snapshot_error = None
    snapshot_status = None
    snapshot_elapsed_s = None
    snapshot_path = None
    if live_session and attempts:
        snapshot_path = raw_cache / "live-session" / "renode.cobertura.xml"
        session_cache = _load_json(
            progress_root / "targets" / _safe(target_id) / "dotnet-session.json",
        )
        unchanged_snapshot = (
            session_cache.get("session_id") == session_id
            and session_cache.get("attempt_sequence") == attempts
            and snapshot_path.is_file()
        )
        if unchanged_snapshot:
            ok, snapshot_error = True, None
            snapshot_status = "reused"
        else:
            snapshot_started = time.monotonic()
            ok, snapshot_error = _dotnet_session_snapshot(
                launcher_value=session_launcher, session_id=session_id,
                raw_cache=raw_cache,
            )
            snapshot_elapsed_s = round(max(0.0, time.monotonic() - snapshot_started), 6)
            snapshot_status = "captured" if ok else "failed"
            if ok:
                _write_json(
                    progress_root / "targets" / _safe(target_id) / "dotnet-session.json",
                    {
                        "session_id": session_id,
                        "attempt_sequence": attempts,
                        "snapshot_sha256": _sha256(snapshot_path),
                        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
        profiles = int(ok)
        data_profiles = int(ok)
        if snapshot_error:
            previous = _load_json(
                progress_root / "targets" / _safe(target_id) / "latest.json",
            )
            if previous:
                previous["status"] = "stale"
                previous["stale_reason"] = "dotnet-live-session-snapshot-failed"
                previous["requested_input_sequence"] = attempts
                previous["snapshot_error"] = snapshot_error
                previous_source = previous.get("source_coverage")
                if isinstance(previous_source, dict):
                    previous_source["stale"] = True
                previous_collection = previous.get("collection")
                if not isinstance(previous_collection, dict):
                    previous_collection = {}
                    previous["collection"] = previous_collection
                previous_collection.update({
                    "snapshot_error": snapshot_error,
                    "snapshot_status": snapshot_status,
                    "snapshot_elapsed_s": snapshot_elapsed_s,
                })
                return previous
            return {
                "method": None, "target": target_id, "status": "stale",
                "attempted_cases": attempts, "profiles_seen": profiles,
                "profiles_with_data": data_profiles,
                "profiles_missing": attempts, "collector": collector,
                "input_sequence": attempts, "identity_digest": None,
                "collected_at_utc": None,
                "collection": {
                    "status": "stale",
                    "reason": "dotnet-live-session-snapshot-failed",
                    "snapshot_error": snapshot_error,
                    "snapshot_status": snapshot_status,
                    "snapshot_elapsed_s": snapshot_elapsed_s,
                },
                "source_coverage": {"status": "stale", "metrics": {}},
                "scope": "cumulative-live-session-snapshot",
            }
    if attempts <= 0:
        return {
            "method": None, "target": target_id, "status": "waiting",
            "attempted_cases": 0, "profiles_seen": profiles,
            "profiles_with_data": data_profiles, "collector": collector,
            "input_sequence": 0, "identity_digest": None,
            "collection": {"status": "waiting", "reason": "no-completed-target-attempts"},
            "source_coverage": {"status": "waiting", "metrics": {}},
            "scope": "live-session-snapshot" if live_session else "completed-attempt-profiles",
        }

    source_spec = target.get("source_coverage")
    if not isinstance(source_spec, Mapping) or collector not in PROFILE_SUFFIXES:
        return {
            "method": None, "target": target_id, "status": "gap",
            "attempted_cases": attempts, "profiles_seen": profiles,
            "profiles_with_data": data_profiles, "collector": collector,
            "input_sequence": profiles, "identity_digest": None,
            "collection": {"status": "gap", "reason": "source-coverage-not-configured"},
            "source_coverage": {"status": "gap", "metrics": {}},
        }

    binary = _dependency_path(target.get("coverage_binary"))
    cache_key = _source_collection_cache_key(
        target=target, source_spec=source_spec, collector=collector,
        raw_cache=raw_cache, binary=binary, attempts=attempts,
        session_id=session_id if live_session else None,
        session_snapshot=snapshot_path if live_session else None,
    )
    previous = _load_json(
        progress_root / "targets" / _safe(target_id) / "latest.json",
    )
    previous_collection = previous.get("collection")
    previous_source = previous.get("source_coverage")
    if (
        not force_full_batch
        and previous.get("source_collection_cache_key") == cache_key
        and isinstance(previous.get("source_collection_base"), Mapping)
        and isinstance(previous_collection, Mapping)
        and previous_collection.get("status") not in {"processing", "stale"}
        and not previous_collection.get("pending_profiles")
        and isinstance(previous_source, Mapping)
        and previous_source.get("status") != "stale"
    ):
        row = deepcopy(previous)
        row["status"] = "provisional" if row.get("status") == "processing" else row.get("status")
        row["source_collection_reused"] = True
        row.pop("trace_materialization", None)
        row["attempted_cases"] = attempts
        row["profiles_seen"] = profiles
        row["profiles_with_data"] = data_profiles
        missing_profiles = 0 if live_session else max(0, attempts - data_profiles)
        row["profiles_missing"] = missing_profiles
        base = row["source_collection_base"]
        collection = dict(previous_collection)
        collection["status"] = base.get("collection_status")
        if base.get("collection_reason"):
            collection["reason"] = base["collection_reason"]
        else:
            collection.pop("reason", None)
        source = dict(previous_source)
        source["status"] = base.get("source_status")
        if base.get("source_reason"):
            source["reason"] = base["source_reason"]
        else:
            source.pop("reason", None)
        incomplete = missing_profiles > 0 or bool(
            collector == "gcov"
            and _metric_count(row.get("input_sequence")) < data_profiles
        )
        if incomplete and collection.get("status") in {"observed", "partial"}:
            reason = (
                "coverage-progress-incremental-profile-count-incomplete"
                if collector == "gcov"
                and _metric_count(row.get("input_sequence")) < data_profiles else
                "coverage-progress-profile-count-incomplete"
            )
            collection.update({"status": "partial", "reason": reason})
            source.update({"status": "partial", "reason": reason})
        row["collection"] = collection
        source.pop("stale", None)
        row["source_coverage"] = source
        return row

    output = progress_root / "work" / _safe(target_id)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coverage-sample-", dir=output) as directory:
        temporary = Path(directory)
        target_copy = deepcopy(dict(target))
        copied_spec = deepcopy(dict(source_spec))
        profile_path = temporary / "simulator-source.lcov.info"
        identity_path = temporary / "identity.json"
        copied_spec["profile"] = str(profile_path)
        copied_spec["identity"] = str(identity_path)
        target_copy["source_coverage"] = copied_spec
        if live_session:
            collection_raw = raw_cache / "live-session"
            expected_inputs = 1
        else:
            collection_raw = raw_cache
            expected_inputs = attempts
        input_sequence = attempts if live_session else 0
        incremental_reason = None
        incremental_collection = None
        current_identity = None
        incremental_enabled = (
            not force_full_batch and not live_session and collector == "gcov"
        )
        if incremental_enabled:
            try:
                current_identity = source_coverage_identity(
                    target_copy, copied_spec, binary,
                )
                incremental_collection, input_sequence, incremental_reason = (
                    _incremental_source_collection(
                        temporary=temporary, target_copy=target_copy, binary=binary,
                        raw_cache=raw_cache, current_identity=current_identity,
                        identity_path=identity_path, profile_path=profile_path,
                    )
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                incremental_reason = f"incremental-collection-error:{type(error).__name__}"
        if incremental_collection is not None:
            collection = incremental_collection
            input_sequence = int(collection.get("input_sequence", input_sequence) or 0)
        elif incremental_enabled:
            collection = collect_source_coverage_batch(
                temporary, target_copy, binary, collection_raw,
                expected_coverage_cases=expected_inputs,
            )
            fallback_reason = incremental_reason or "incremental-collection-unavailable"
            collection = {
                **collection, "incremental": False,
                "incremental_fallback_reason": fallback_reason,
            }
            seed_identity = _load_json(identity_path)
            seed_digest = _identity_digest(seed_identity) if seed_identity else None
            case_keys = sorted(_case_profile_dirs(raw_cache, collector))
            if (
                collection.get("status") in {"observed", "partial"}
                and profile_path.is_file() and case_keys
                and isinstance(current_identity, dict)
                and seed_digest == _identity_digest(current_identity)
            ):
                input_sequence = len(case_keys)
                collection.update({
                    "input_sequence": input_sequence,
                    "incremental_seed": True,
                })
                _persist_incremental_state(
                    target_dir=progress_root / "targets" / _safe(target_id),
                    profile=profile_path, input_keys=case_keys,
                    input_statuses={key: str(collection["status"]) for key in case_keys},
                    identity_digest=seed_digest, collection=collection,
                )
            else:
                input_sequence = 0
        else:
            collection = collect_source_coverage_batch(
                temporary, target_copy, binary, collection_raw,
                expected_coverage_cases=expected_inputs,
            )
            if not live_session:
                input_sequence = len(_case_profile_dirs(raw_cache, collector))
            if incremental_reason:
                collection = {
                    **collection, "incremental": False,
                    "incremental_fallback_reason": incremental_reason,
                }
            elif not live_session and collector in {"lcov", "llvm", "dotnet"}:
                collection = {
                    **collection, "incremental": False,
                    "incremental_fallback_reason": "collector-not-proven-incremental",
                }
        if force_full_batch:
            collection = {**collection, "aggregation_mode": "full-batch"}
        identity = _load_json(identity_path)
        identity_digest = _identity_digest(identity) if identity else None
        source = target_source_coverage(
            temporary, target_copy,
            coverage_binary_sha=collection.get("coverage_binary_sha"),
            source_commit_observed=collection.get("source_commit_observed"),
            coverage_binary_sha256=collection.get("coverage_binary_sha256"),
            collection_status=collection.get("status"),
            collection_reason=collection.get("reason"),
        )
        source_collection_base = {
            "collection_status": collection.get("status"),
            "collection_reason": collection.get("reason"),
            "source_status": source.get("status"),
            "source_reason": source.get("reason"),
        }
        missing_profiles = 0 if live_session else max(0, attempts - data_profiles)
        incomplete = missing_profiles > 0 or bool(
            incremental_enabled and input_sequence < data_profiles
        )
        if incomplete and collection.get("status") in {"observed", "partial"}:
            reason = (
                "coverage-progress-incremental-profile-count-incomplete"
                if incremental_enabled and input_sequence < data_profiles
                else "coverage-progress-profile-count-incomplete"
            )
            collection = {**collection, "status": "partial", "reason": reason,
                          "attempted_cases": attempts,
                          "profiles_with_data": data_profiles}
            source = target_source_coverage(
                temporary, target_copy,
                coverage_binary_sha=collection.get("coverage_binary_sha"),
                source_commit_observed=collection.get("source_commit_observed"),
                coverage_binary_sha256=collection.get("coverage_binary_sha256"),
                collection_status="partial", collection_reason=reason,
            )
        if snapshot_status is not None:
            collection = {
                **collection,
                "snapshot_status": snapshot_status,
                "snapshot_elapsed_s": snapshot_elapsed_s,
            }
        if snapshot_error:
            collection = {**collection, "snapshot_error": snapshot_error}
        if (
            force_full_batch and collector == "gcov" and not live_session
            and profile_path.is_file() and identity_digest
            and collection.get("status") in {"observed", "partial"}
        ):
            case_keys = sorted(_case_profile_dirs(raw_cache, collector))
            if case_keys and len(case_keys) == data_profiles:
                collection.update({
                    "input_sequence": len(case_keys), "incremental_seed": True,
                })
                _persist_incremental_state(
                    target_dir=progress_root / "targets" / _safe(target_id),
                    profile=profile_path, input_keys=case_keys,
                    input_statuses={key: str(collection["status"]) for key in case_keys},
                    identity_digest=identity_digest, collection=collection,
                )
        profile_output = progress_root / "profiles" / f"{_safe(target_id)}.lcov.info"
        profile_digest = collection.get("profile_sha256")
        if profile_path.is_file():
            profile_output.parent.mkdir(parents=True, exist_ok=True)
            temporary_profile = profile_output.with_name(profile_output.name + ".tmp")
            shutil.copy2(profile_path, temporary_profile)
            os.replace(temporary_profile, profile_output)
            if not isinstance(profile_digest, str) and not collection.get("incremental"):
                digest = hashlib.sha256()
                with profile_output.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                profile_digest = digest.hexdigest()
            source["profile_path"] = str(profile_output.relative_to(run_root))
        source["provisional"] = True
        return {
            "method": None,
            "target": target_id,
            "status": "provisional",
            "attempted_cases": attempts,
            "profiles_seen": profiles,
            "profiles_with_data": data_profiles,
            "profiles_missing": missing_profiles,
            "collector": collector,
            "collection": collection,
            "source_coverage": source,
            "profile_sha256": profile_digest,
            "identity_digest": identity_digest,
            "source_collection_cache_key": cache_key,
            "source_collection_base": source_collection_base,
            "source_collection_reused": False,
            "input_sequence": input_sequence,
            "scope": "cumulative-live-session-snapshot" if live_session else "completed-attempt-profiles",
        }


def _metric(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    source = row.get("source_coverage")
    metrics = source.get("metrics") if isinstance(source, Mapping) else None
    value = metrics.get(name) if isinstance(metrics, Mapping) else None
    if not isinstance(value, Mapping):
        return {}
    metric = dict(value)
    covered, eligible = metric.get("covered"), metric.get("eligible")
    metric["provisional_value"] = (
        round(covered / eligible, 6)
        if type(covered) is int and type(eligible) is int and eligible > 0
        else None
    )
    return metric


def _write_latest_csv(
    path: Path, rows: list[dict[str, Any]], as_of: str,
    experiment_progress: Mapping[str, Any], legacy_status: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fields = [
        "as_of_utc", "method", "target", "attempted_cases", "profiles_with_data",
        "collector_status", "source_status", "coverage_sample_status", "sample_phase",
        "function_covered", "function_eligible", "function_value", "function_provisional_value",
        "line_covered", "line_eligible", "line_value", "line_provisional_value",
        "branch_covered", "branch_eligible", "branch_value", "branch_provisional_value",
        "gen_opcode_covered", "gen_opcode_eligible", "gen_opcode_value",
        "gen_opcode_provisional_value", "gen_opcode_status", "gen_opcode_partitions",
        "exec_opcode_covered", "exec_opcode_eligible", "exec_opcode_value",
        "exec_opcode_provisional_value", "exec_opcode_status", "exec_opcode_partitions",
        "legacy_rv_instruction_status", "target_outcomes", "target_statuses",
        "generated_candidates", "raw_candidates", "artifact_gaps",
        "completed_cases", "in_progress_cases", "mcmc_steps", "mh_attempts", "mh_accepted", "mh_rejected",
        "target_attempts_total", "target_tested", "target_gaps", "reason",
    ]
    if metadata:
        fields = ["run_id", "method", "target", "collected_at_utc",
                  "input_sequence", "status", "as_of_utc", "identity_digest",
                  *fields[3:]]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            source = row.get("source_coverage") or {}
            collection = row.get("collection") or {}
            record = {
                "as_of_utc": as_of,
                "method": row.get("method"),
                "target": row.get("target"),
                "attempted_cases": row.get("attempted_cases"),
                "profiles_with_data": row.get("profiles_with_data"),
                "collector_status": collection.get("status"),
                "source_status": source.get("status"),
                "coverage_sample_status": row.get("coverage_sample_status"),
                "sample_phase": row.get("sample_phase"),
                "legacy_rv_instruction_status": legacy_status.get("status"),
                "reason": collection.get("reason") or source.get("reason") or collection.get("snapshot_error"),
            }
            if metadata:
                record.update({name: metadata.get(name) for name in (
                    "run_id", "method", "target", "collected_at_utc",
                    "input_sequence", "status",
                )})
                record["identity_digest"] = row.get("identity_digest")
            for name in METRIC_NAMES:
                metric = _metric(row, name)
                record[f"{name}_covered"] = metric.get("covered")
                record[f"{name}_eligible"] = metric.get("eligible")
                record[f"{name}_value"] = metric.get("value")
                record[f"{name}_provisional_value"] = metric.get("provisional_value")
            catalog = row.get("rv_opcode_catalog_coverage") or {}
            catalog_metrics = catalog.get("metrics") or {}
            for name, prefix in (("GenOpcodeCov", "gen_opcode"), ("ExecOpcodeCov", "exec_opcode")):
                metric = catalog.get("method_generated_union") \
                    if name == "GenOpcodeCov" else catalog_metrics.get(name)
                if not isinstance(metric, Mapping):
                    metric = catalog_metrics.get(name)
                metric = metric if isinstance(metric, Mapping) else {}
                record[f"{prefix}_covered"] = metric.get("covered")
                record[f"{prefix}_eligible"] = metric.get("eligible")
                record[f"{prefix}_value"] = metric.get("value")
                record[f"{prefix}_provisional_value"] = metric.get("provisional_value")
                record[f"{prefix}_status"] = metric.get("status")
                record[f"{prefix}_partitions"] = json.dumps(
                    metric.get("partition_coverage", {}), ensure_ascii=False, separators=(",", ":"),
                )
            target_metrics = row.get("experiment_metrics") or {}
            target_progress = experiment_progress.get("targets") or {}
            target_data = target_progress.get(str(row.get("target")), {}) \
                if isinstance(target_progress, Mapping) else {}
            generation = experiment_progress.get("generation") or {}
            framework = experiment_progress if experiment_progress.get("kind") == "ours-framework" else {}
            record.update({
                "target_outcomes": json.dumps(
                    target_metrics.get("outcomes", {}),
                    ensure_ascii=False, separators=(",", ":"),
                ),
                "target_statuses": json.dumps(
                    target_metrics.get("statuses", {}),
                    ensure_ascii=False, separators=(",", ":"),
                ),
                "generated_candidates": generation.get("candidates"),
                "raw_candidates": generation.get("raw_candidates"),
                "artifact_gaps": generation.get("artifact_gaps"),
                "completed_cases": framework.get("completed_cases"),
                "in_progress_cases": framework.get("in_progress_cases"),
                "mcmc_steps": framework.get("steps"),
                "mh_attempts": framework.get("mh_attempts"),
                "mh_accepted": framework.get("accepted"),
                "mh_rejected": framework.get("rejected"),
                "target_attempts_total": target_data.get("attempts"),
                "target_tested": framework.get("target_tested"),
                "target_gaps": framework.get("target_gap"),
            })
            writer.writerow(record)
    os.replace(temporary, path)


def _write_target_latest(
    progress_root: Path, row: dict[str, Any], metadata: Mapping[str, Any],
    experiment_progress: Mapping[str, Any], legacy_status: Mapping[str, Any],
    rv_opcode_catalog_contract: Mapping[str, Any],
) -> None:
    target_dir = progress_root / "targets" / _safe(row.get("target"))
    payload = {
        **row, **metadata,
        "experiment_progress": dict(experiment_progress),
        "rv_instruction_coverage": dict(legacy_status),
        "rv_opcode_catalog_contract": dict(rv_opcode_catalog_contract),
    }
    _write_latest_csv(
        target_dir / "latest.csv", [payload],
        str(metadata.get("collected_at_utc") or ""), experiment_progress,
        legacy_status, metadata=metadata,
    )
    _write_json(target_dir / "latest.json", payload)


def _render(
    rows: list[dict[str, Any]], as_of: str,
    experiment_progress: Mapping[str, Any], legacy_status: Mapping[str, Any],
) -> None:
    print(f"coverage-progress as of {as_of} (provisional metrics)")
    print("| Target | Attempts | Profiles | Function | Line | Branch | Status |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        values = []
        for name in METRIC_NAMES:
            metric = _metric(row, name)
            covered, eligible = metric.get("covered"), metric.get("eligible")
            value = metric.get("provisional_value")
            if type(covered) is int and type(eligible) is int:
                values.append(f"{covered}/{eligible} ({value:.2%})" if type(value) in (int, float)
                              else f"{covered}/{eligible} (pending)")
            else:
                values.append("—")
        status = row.get("collection", {}).get("status")
        print(
            f"| {row.get('target')} | {row.get('attempted_cases')} | "
            f"{row.get('profiles_with_data')} | {values[0]} | {values[1]} | "
            f"{values[2]} | {status} |"
        )
    print("| Target | GenOpcodeCov method union / 1480 | ExecOpcodeCov / 1480 | Gen / Exec status |")
    print("| --- | ---: | ---: | --- |")
    for row in rows:
        catalog = row.get("rv_opcode_catalog_coverage") or {}
        metrics = catalog.get("metrics") or {}
        formatted = []
        for name in RV_CATALOG_METRICS:
            metric = (
                catalog.get("method_generated_union")
                if name == "GenOpcodeCov" else metrics.get(name)
            ) or {}
            covered, eligible = metric.get("covered"), metric.get("eligible")
            value = metric.get("provisional_value")
            if type(covered) is int and type(eligible) is int:
                shown_value = f"{value:.2%}" if type(value) in (int, float) else "pending"
                formatted.append(f"{covered}/{eligible} ({shown_value})")
            else:
                formatted.append("—")
        gen_status = (catalog.get("method_generated_union") or {}).get("status") \
            or (metrics.get("GenOpcodeCov") or {}).get("status") or "gap"
        exec_status = (metrics.get("ExecOpcodeCov") or {}).get("status") or "gap"
        print(
            f"| {row.get('target')} | {formatted[0]} | {formatted[1]} | "
            f"{gen_status} / {exec_status} |"
        )
    if experiment_progress.get("kind") == "ours-framework":
        print(
            "Ours progress: "
            f"cases={experiment_progress.get('completed_cases')} "
            f"in-progress={experiment_progress.get('in_progress_cases')} "
            f"steps={experiment_progress.get('steps')} "
            f"MH={experiment_progress.get('mh_attempts')} "
            f"accepted={experiment_progress.get('accepted')} "
            f"rejected={experiment_progress.get('rejected')} "
            f"target-tested={experiment_progress.get('target_tested')} "
            f"target-gap={experiment_progress.get('target_gap')}"
        )
    else:
        generation = experiment_progress.get("generation") or {}
        execution = experiment_progress.get("execution") or {}
        print(
            "External progress: "
            f"candidates={generation.get('candidates')} "
            f"raw={generation.get('raw_candidates')} "
            f"artifact-gaps={generation.get('artifact_gaps')} "
            f"target-attempts={execution.get('target_attempts')} "
            f"outcomes={json.dumps(execution.get('outcome_counts', {}), ensure_ascii=False)}"
        )
    print(f"Legacy GenCov/ExecCov/OpcodeCov: {legacy_status.get('status')} ({legacy_status.get('reason')})")


def sample(
    run_root: Path, expected_method: str | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest = _load_json(run_root / "execution-manifest.json")
    config = _load_json(run_root / "config.snapshot.json")
    method = str(manifest.get("method_filter") or "")
    action = str(manifest.get("action") or "")
    if method not in METHODS or (expected_method and expected_method != method):
        raise ValueError("run method does not match the seven-method matrix")
    if action not in {"run", "execute-existing-queues", "framework-run"}:
        raise ValueError("coverage progress requires comparison execution or framework-run")
    if not manifest or not config:
        raise ValueError("run manifest or config snapshot is missing")
    try:
        started = datetime.fromisoformat(
            str(manifest.get("started_at_utc") or "").replace("Z", "+00:00")
        )
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = int(manifest.get("duration_seconds"))
    except (TypeError, ValueError, OverflowError):
        started = None
        duration = 0
    wrapper_result = _load_json(run_root / "wrapper-result.json")
    wrapper_stopped = (
        wrapper_result.get("status") in {"completed", "partial", "failed"}
        if isinstance(wrapper_result, Mapping) else False
    )
    if not wrapper_stopped:
        raise ValueError("coverage snapshots are available only after execution stops")
    sample_phase = "post-execution-full-batch"
    if action in {"run", "execute-existing-queues"}:
        partial_path = run_root / "coverage-result.partial.json"
        if partial_path.is_file():
            partial_coverage = _load_json(partial_path)
            pinned_catalog_replay = (
                action == "execute-existing-queues"
                and isinstance(manifest.get("case_catalog_manifest"), Mapping)
                and bool(manifest["case_catalog_manifest"].get("file_sha256"))
            )
            stopped_partial = (
                pinned_catalog_replay and partial_coverage.get("status") == "partial"
            )
            if partial_coverage.get("status") != "deferred" and not stopped_partial:
                print("official coverage finalization has started; no provisional sample was taken")
                return {"status": "finalizing"}
    if action == "framework-run":
        batch_root = run_root / "framework" / _safe(method) / "coverage-batches"
        batch_finalizing = False
        if batch_root.is_dir():
            batch_finalizing = any(
                (batch / "summary.json").is_file()
                for target in batch_root.iterdir() if target.is_dir()
                for batch in target.iterdir() if batch.is_dir()
            )
        if batch_finalizing:
            print("official framework coverage finalization has started; no provisional sample was taken")
            return {"status": "finalizing"}
    targets = config.get("targets")
    if not isinstance(targets, list) or len(targets) != 7:
        raise ValueError("the run config must declare all seven Targets")

    progress_root = run_root / "coverage-progress"
    progress_root.mkdir(parents=True, exist_ok=True)
    with (progress_root / ".progress.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("coverage progress sample already running", file=sys.stderr)
            return {"status": "busy"}

        selected_targets = [
            target for target in targets
            if isinstance(target, Mapping) and isinstance(target.get("id"), str)
        ]
        pending_by_target: dict[str, list[tuple[Any, ...]]] = {}
        pending_framework_catalogs: dict[str, list[dict[str, Any]]] = {}
        deferred_external = action in {"run", "execute-existing-queues"}
        if deferred_external:
            input_data, experiment_progress = _external_inputs(
                run_root, method, selected_targets, progress_root,
                defer_materialization=True, pending_by_target=pending_by_target,
            )
        else:
            input_data, experiment_progress = _framework_inputs(
                run_root, method, selected_targets, progress_root,
                pending_catalogs=pending_framework_catalogs,
            )
        catalog_contract = config.get("rv_opcode_catalog_coverage")
        catalog_contract = catalog_contract if isinstance(catalog_contract, Mapping) else {}
        catalog_enabled = catalog_contract.get("enabled") is True
        legacy_contract = config.get("rv_instruction_coverage")
        legacy_contract = legacy_contract if isinstance(legacy_contract, Mapping) else {}
        legacy_enabled = legacy_contract.get("enabled") is True
        legacy_status = {
            "enabled": legacy_enabled,
            "status": "not-collected" if legacy_enabled else "disabled",
            "metrics": ["GenCov", "ExecCov", "OpcodeCov"],
            "reason": (
                "legacy-rv-instruction-metrics-not-collected-by-this-snapshot"
                if legacy_enabled else "config.rv_instruction_coverage.enabled=false"
            ),
        }
        rv_opcode_catalog_contract = {
            "schema_version": catalog_contract.get("schema_version"),
            "catalog_id": catalog_contract.get("catalog"),
            "key_schema": catalog_contract.get("key_schema"),
            "registry_sha256": catalog_contract.get("registry_sha256"),
            "form_count": catalog_contract.get("form_count"),
            "opcode_denominator": catalog_contract.get("opcode_denominator"),
            "partition_denominators": catalog_contract.get("partition_denominators"),
            "aggregation": catalog_contract.get("aggregation"),
            "enabled": catalog_enabled,
        }
        now = datetime.now(timezone.utc)
        elapsed_seconds = max(0, int((now - started).total_seconds())) if started else None
        runtime = {
            "elapsed_seconds": elapsed_seconds,
            "budget_seconds": duration or None,
            "remaining_seconds": (
                max(0, duration - elapsed_seconds)
                if duration > 0 and elapsed_seconds is not None else None
            ),
        }
        experiment_progress = {"runtime": runtime, **experiment_progress}
        has_pending_traces = (
            any(pending_by_target.values())
            or any(pending_framework_catalogs.values())
        )
        opcode_catalog_summary = {} if has_pending_traces else _method_catalog_summary(
            method, selected_targets, input_data,
        )
        generated_metric = next((
            item.get("metrics", {}).get("GenOpcodeCov")
            for item in opcode_catalog_summary.get("method_generated_unions", [])
            if isinstance(item, Mapping) and item.get("method") == method
        ), None)
        processing_union = {
            "covered": None, "eligible": OPCODE_CATALOG_DENOMINATOR,
            "value": None, "provisional_value": None, "status": "processing",
            "reason": "method-generated-union-pending",
        }
        rows: list[dict[str, Any]] = []
        collection_jobs: list[dict[str, Any]] = []
        sample_id = now.strftime("%Y%m%dT%H%M%S") \
            + f"-{time.time_ns() % 1_000_000_000:09d}"
        target_order = {
            str(target["id"]): index for index, target in enumerate(selected_targets)
        }

        def collection_order_key(target: Mapping[str, Any]) -> tuple[bool, int]:
            spec = target.get("source_coverage")
            collector = spec.get("collector") if isinstance(spec, Mapping) else None
            return str(collector or "") in {"llvm", "dotnet"}, target_order[
                str(target["id"])
            ]

        collection_targets = sorted(
            selected_targets, key=collection_order_key,
        )

        def publish_progress(status: str) -> dict[str, Any]:
            current_rows = {
                str(row.get("target")): dict(row) for row in rows
            }
            snapshot_rows = []
            for target in selected_targets:
                target_id = str(target["id"])
                row = current_rows.get(target_id)
                if row is not None:
                    row["coverage_sample_status"] = "updated"
                else:
                    previous = _load_json(
                        progress_root / "targets" / _safe(target_id) / "latest.json",
                    )
                    if (
                        previous.get("run_id") == manifest.get("run_id")
                        and previous.get("method") == method
                    ):
                        row = dict(previous)
                        row["coverage_sample_status"] = (
                            "processing" if row.get("status") == "processing"
                            else "pending"
                        )
                    else:
                        inputs = input_data.get(target_id, {})
                        waiting = _metric_count(inputs.get("attempts")) == 0
                        row = {
                            "run_id": manifest.get("run_id"), "method": method,
                            "target": target_id,
                            "collected_at_utc": None,
                            "input_sequence": _metric_count(inputs.get("profiles")),
                            "status": "waiting" if waiting else "processing",
                            "coverage_sample_status": "waiting" if waiting else "pending",
                            "attempted_cases": _metric_count(inputs.get("attempts")),
                            "profiles_seen": _metric_count(inputs.get("profiles")),
                            "profiles_with_data": _metric_count(inputs.get("data_profiles")),
                            "collection": {
                                "status": "waiting" if waiting else "processing",
                                "reason": "target-collection-not-published-yet",
                            },
                            "source_coverage": {
                                "status": "waiting" if waiting else "processing",
                                "metrics": {},
                            },
                            "rv_opcode_catalog_coverage": {
                                "status": "waiting" if waiting else "processing",
                                "eligible": OPCODE_CATALOG_DENOMINATOR,
                                "metrics": {},
                            },
                        }
                row["sample_phase"] = sample_phase
                snapshot_rows.append(row)

            as_of = datetime.now(timezone.utc).isoformat()
            payload = {
                "schema_version": "rq1-coverage-progress-v2",
                "sample_id": sample_id,
                "as_of_utc": as_of,
                "run_id": manifest.get("run_id"),
                "method": method,
                "action": action,
                "sample_phase": sample_phase,
                "duration_seconds": manifest.get("duration_seconds"),
                "source_commit": manifest.get("source_commit"),
                "source_dirty": manifest.get("source_dirty"),
                "config_sha256": manifest.get("config_sha256"),
                "status": status,
                "coverage_sample": {
                    "status": status,
                    "targets_expected": len(selected_targets),
                    "targets_updated_this_sample": len(rows),
                },
                "metrics": {
                    "source_coverage": [
                        "SimSrcCov.function", "SimSrcCov.line", "SimSrcCov.branch",
                    ],
                    "rv_opcode_catalog_coverage": list(RV_CATALOG_METRICS),
                    "rv_opcode_catalog_denominator": OPCODE_CATALOG_DENOMINATOR,
                    "rv_opcode_catalog_id": OPCODE_CATALOG_ID,
                },
                "rv_opcode_catalog_contract": rv_opcode_catalog_contract,
                "rv_opcode_catalog_coverage": opcode_catalog_summary or None,
                "rv_instruction_coverage": legacy_status,
                "experiment_progress": experiment_progress,
                "targets": snapshot_rows,
                "official_result": (
                    None if deferred_external else "coverage/summary.json"
                ),
                "progress_result": "coverage-progress/latest.json",
            }
            _write_json(progress_root / "latest.json", payload)
            _write_latest_csv(
                progress_root / "latest.csv", snapshot_rows, as_of,
                experiment_progress, legacy_status,
            )
            _render(snapshot_rows, as_of, experiment_progress, legacy_status)
            return payload

        def mark_target_processing(target: Mapping[str, Any]) -> None:
            target_id = str(target["id"])
            inputs = input_data.get(target_id, {})
            previous = _load_json(progress_root / "targets" / _safe(target_id) / "latest.json")
            matching_previous = (
                previous.get("run_id") == manifest.get("run_id")
                and previous.get("method") == method
            )
            row = dict(previous) if matching_previous else {
                "collection": {
                    "status": "processing",
                    "reason": "source-coverage-collection-pending",
                },
                "source_coverage": {"status": "processing", "metrics": {}},
            }
            input_sequence = (
                _metric_count(previous.get("input_sequence"))
                if matching_previous else _metric_count(inputs.get("profiles"))
            )
            row.update({
                "target": target_id, "method": method, "run_id": manifest.get("run_id"),
                "attempted_cases": _metric_count(inputs.get("attempts")),
                "profiles_seen": (
                    _metric_count(previous.get("profiles_seen"))
                    if matching_previous else _metric_count(inputs.get("profiles"))
                ),
                "profiles_with_data": (
                    _metric_count(previous.get("profiles_with_data"))
                    if matching_previous else _metric_count(inputs.get("data_profiles"))
                ),
                "status": "processing",
                "coverage_sample_status": "processing",
                "sample_phase": sample_phase,
                "requested_input_sequence": _metric_count(inputs.get("profiles")),
            })
            pending_trace_count = (
                len(pending_by_target.get(target_id, ()))
                + len(pending_framework_catalogs.get(target_id, ()))
            )
            if pending_trace_count:
                row["trace_materialization"] = {
                    "status": "processing", "pending_cases": pending_trace_count,
                }
            catalog = dict(row.get("rv_opcode_catalog_coverage") or {})
            if catalog_enabled:
                catalog["method_generated_union"] = dict(processing_union)
            row["rv_opcode_catalog_coverage"] = catalog
            metadata = {
                "run_id": manifest.get("run_id"), "method": method, "target": target_id,
                "collected_at_utc": previous.get("collected_at_utc")
                if matching_previous else None,
                "input_sequence": input_sequence,
                "status": "processing",
            }
            row.update(metadata)
            _write_target_latest(
                progress_root, row, metadata, experiment_progress, legacy_status,
                rv_opcode_catalog_contract,
            )

        publish_progress("processing")

        for target in collection_targets:
            target_id = str(target["id"])
            inputs = input_data.get(target_id, {})
            previous = _load_json(
                progress_root / "targets" / _safe(target_id) / "latest.json",
            )
            has_previous = (
                previous.get("run_id") == manifest.get("run_id")
                and previous.get("method") == method
            )
            mark_target_processing(target)
            for context in pending_framework_catalogs.get(target_id, []):
                base_expected = _metric_count(context.pop("base_expected", 0))
                live_entries, live_expected = _framework_live_catalog_entries(**context)
                inputs.setdefault("catalog_entries", []).extend(live_entries)
                inputs["catalog_expected_cases"] = _metric_count(
                    inputs.get("catalog_expected_cases"),
                ) + max(0, live_expected - base_expected)
            candidates = pending_by_target.get(target_id, [])
            if candidates:
                materialized = _materialize_external_target(run_root, target, candidates)
                for candidate in candidates:
                    coverage = candidate[0].get("simulator_coverage")
                    entry = coverage.get("rv_opcode_catalog_coverage") \
                        if isinstance(coverage, Mapping) else None
                    if isinstance(entry, Mapping):
                        inputs.setdefault("catalog_entries", []).append(entry)
                materialization = experiment_progress.get("coverage_materialization")
                materialization = dict(materialization) if isinstance(materialization, Mapping) else {}
                for key in (
                    "parsed_cases", "missing_cases", "identity_mismatch_cases",
                    "error_cases", "pending_cases", "right_censored_cases",
                    "not_attempted_cases",
                ):
                    value = materialized.get(key)
                    if type(value) is int:
                        materialization[key] = int(materialization.get(key, 0) or 0) + value
                elapsed = materialized.get("elapsed_s")
                if type(elapsed) in (int, float):
                    materialization["elapsed_s"] = round(
                        float(materialization.get("elapsed_s", 0) or 0) + elapsed, 6,
                    )
                experiment_progress["coverage_materialization"] = materialization

            catalog_entries = inputs.get("catalog_entries")
            catalog_entries = [entry for entry in catalog_entries
                               if isinstance(entry, Mapping)] \
                if isinstance(catalog_entries, list) else []
            expected_catalog_cases = _metric_count(
                inputs.get("catalog_expected_cases", inputs.get("attempts", 0))
            )
            catalog_progress = _catalog_progress(
                method, target_id, catalog_entries, expected_catalog_cases,
                enabled=catalog_enabled,
            )
            if has_pending_traces and catalog_enabled:
                catalog_progress["method_generated_union"] = dict(processing_union)
            elif isinstance(generated_metric, Mapping):
                catalog_progress["method_generated_union"] = dict(generated_metric)
            experiment_metrics = {
                "attempts": _metric_count(inputs.get("attempts")),
                "outcomes": dict(sorted(inputs.get("outcomes", {}).items()))
                if isinstance(inputs.get("outcomes"), Mapping) else {},
                "statuses": dict(sorted(inputs.get("statuses", {}).items()))
                if isinstance(inputs.get("statuses"), Mapping) else {},
            }
            pending_row = _load_json(
                progress_root / "targets" / _safe(target_id) / "latest.json",
            )
            pending_row.update({
                "run_id": manifest.get("run_id"), "method": method,
                "target": target_id,
                "attempted_cases": _metric_count(inputs.get("attempts")),
                "profiles_seen": (
                    _metric_count(pending_row.get("profiles_seen"))
                    if has_previous else _metric_count(inputs.get("profiles"))
                ),
                "profiles_with_data": (
                    _metric_count(pending_row.get("profiles_with_data"))
                    if has_previous else _metric_count(inputs.get("data_profiles"))
                ),
                "input_sequence": (
                    _metric_count(pending_row.get("input_sequence"))
                    if has_previous else _metric_count(inputs.get("profiles"))
                ),
                "collected_at_utc": (
                    pending_row.get("collected_at_utc") if has_previous
                    else datetime.now(timezone.utc).isoformat()
                ),
                "status": "processing",
                "coverage_sample_status": "processing",
                "requested_input_sequence": _metric_count(inputs.get("profiles")),
                "rv_opcode_catalog_coverage": catalog_progress,
                "rv_instruction_coverage": dict(legacy_status),
                "experiment_metrics": experiment_metrics,
            })
            if not has_previous:
                pending_row.update({
                    "collection": {
                        "status": "processing",
                        "reason": "source-coverage-collection-pending",
                    },
                    "source_coverage": {"status": "processing", "metrics": {}},
                })
            if (pending_by_target.get(target_id)
                    or pending_framework_catalogs.get(target_id)):
                pending_row["trace_materialization"] = {"status": "complete"}
            pending_metadata = {
                "run_id": manifest.get("run_id"), "method": method,
                "target": target_id,
                "collected_at_utc": pending_row.get("collected_at_utc"),
                "input_sequence": pending_row["input_sequence"],
                "status": "processing",
            }
            _write_target_latest(
                progress_root, pending_row, pending_metadata,
                experiment_progress, legacy_status, rv_opcode_catalog_contract,
            )
            publish_progress("processing")

            started_at = time.monotonic()
            print(
                f"coverage-collector-start target={target_id} "
                f"profiles={_metric_count(inputs.get('profiles'))} "
                "full_batch=true",
                flush=True,
            )
            try:
                row = _collect_target(
                    run_root=run_root, target=target, progress_root=progress_root,
                    attempts=int(inputs.get("attempts", 0)),
                    profiles=int(inputs.get("profiles", 0)),
                    data_profiles=int(inputs.get("data_profiles", 0)),
                    session_id=inputs.get("session_id"),
                    session_launcher=inputs.get("session_launcher"),
                    force_full_batch=True,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                reason = f"collector-error:{type(error).__name__}"
                row = {
                    "target": target_id, "status": "gap",
                    "attempted_cases": _metric_count(inputs.get("attempts")),
                    "profiles_seen": _metric_count(inputs.get("profiles")),
                    "profiles_with_data": _metric_count(inputs.get("data_profiles")),
                    "collection": {
                        "status": "gap", "reason": reason,
                        "error": str(error)[:500],
                    },
                    "source_coverage": {"status": "gap", "reason": reason, "metrics": {}},
                }
            finally:
                print(
                    f"coverage-collector-finished target={target_id} "
                    f"elapsed_s={time.monotonic() - started_at:.1f}",
                    flush=True,
                )
            source_coverage = row.get("source_coverage")
            if isinstance(source_coverage, dict):
                source_metrics = source_coverage.get("metrics")
                source_metrics = dict(source_metrics) if isinstance(source_metrics, Mapping) else {}
                for name in METRIC_NAMES:
                    metric = _metric(row, name)
                    if metric:
                        source_metrics[name] = metric
                source_coverage["metrics"] = source_metrics
            row["method"] = method
            row["sample_phase"] = sample_phase
            row["rv_opcode_catalog_coverage"] = catalog_progress
            row["coverage_sample_status"] = "updated"
            row["rv_instruction_coverage"] = dict(legacy_status)
            row["experiment_metrics"] = experiment_metrics
            target_metadata = {
                "run_id": manifest.get("run_id"),
                "method": method,
                "target": target_id,
                "collected_at_utc": row.get(
                    "collected_at_utc", datetime.now(timezone.utc).isoformat(),
                ),
                "input_sequence": _metric_count(row.get("input_sequence")),
                "status": row.get("status") or "provisional",
            }
            row.update(target_metadata)
            if (pending_by_target.get(target_id)
                    or pending_framework_catalogs.get(target_id)):
                row["trace_materialization"] = {"status": "complete"}
            rows.append(row)
            _write_target_latest(
                progress_root, row, target_metadata,
                experiment_progress, legacy_status, rv_opcode_catalog_contract,
            )
            publish_progress("processing")

        if has_pending_traces:
            opcode_catalog_summary = _method_catalog_summary(method, selected_targets, input_data)
            generated_metric = next((
                item.get("metrics", {}).get("GenOpcodeCov")
                for item in opcode_catalog_summary.get("method_generated_unions", [])
                if isinstance(item, Mapping) and item.get("method") == method
            ), None)
            union_time = datetime.now(timezone.utc).isoformat()
            for row in rows:
                catalog = row.get("rv_opcode_catalog_coverage")
                if not isinstance(catalog, dict):
                    continue
                if isinstance(generated_metric, Mapping):
                    catalog["method_generated_union"] = dict(generated_metric)
                    catalog["method_generated_union_collected_at_utc"] = union_time
                metadata = {key: row.get(key) for key in (
                    "run_id", "method", "target", "collected_at_utc",
                    "input_sequence", "status",
                )}
                _write_target_latest(
                    progress_root, row, metadata, experiment_progress,
                    legacy_status, rv_opcode_catalog_contract,
                )

        rows.sort(key=lambda row: target_order[str(row.get("target"))])
        payload = publish_progress("provisional")
        checkpoint = progress_root / "checkpoints" / f"{sample_id}.json"
        _write_json(checkpoint, payload)
        with (progress_root / "checkpoints.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=Path(os.environ.get("RQ1_RUN", "/opt/runs/unknown")),
    )
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    os.environ.setdefault("RQ1_COVERAGE_JOBS", "1")
    try:
        result = sample(args.run_root, args.method)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"coverage progress unavailable: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if result.get("status") == "provisional":
        return 0
    if result.get("status") in {"busy", "finalizing"}:
        return 4
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

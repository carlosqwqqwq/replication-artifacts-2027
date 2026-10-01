#!/usr/bin/env python3
"""Merge offline coverage for repeated Ours-RVGEN-Direct 1×7 runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

COMPARISON_ROOT = Path(__file__).resolve().parents[1]
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from analysis import coverage_offline, rv_instruction_coverage, source_coverage
from analysis.simulator_coverage import target_source_coverage
import coverage_progress


METHOD = "Ours-RVGEN-Direct"
SCHEMA = "rq1-direct-offline-coverage-v1"
SOURCE_METRICS = ("function", "line", "branch")


def _read_json(path: Path) -> dict[str, Any]:
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


def _run_record(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    execution = _read_json(root / "execution-manifest.json")
    experiment = _read_json(root / "experiment-manifest.json")
    config = _read_json(root / "config.snapshot.json")
    if execution.get("method_filter") != METHOD or execution.get("action") != "run":
        raise ValueError(f"not an Ours-RVGEN-Direct run: {root}")
    targets = config.get("targets")
    if not isinstance(targets, list):
        raise ValueError(f"run config has no target matrix: {root}")
    target_map = {
        str(item.get("id")): item for item in targets
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(target_map) != 7:
        raise ValueError(f"Direct run must contain seven unique Targets: {root}")

    progress = _read_json(root / "progress.json")
    stop = _read_json(root / "stop-provenance.json")
    run_result = _read_json(root / "run-result.json")
    wrapper = _read_json(root / "wrapper-result.json")
    reason = (
        stop.get("reason_code") or run_result.get("stop_reason")
        or run_result.get("partial_reason") or wrapper.get("reason_code")
    )
    stage = str(progress.get("stage") or "")
    interrupted = (
        stage.startswith("external-interruption")
        or reason in coverage_offline._INTERRUPTION_REASONS
        or (root / "external-interruption.json").is_file()
    )
    return {
        "run_id": str(execution.get("run_id") or root.name),
        "run_root": root,
        "method": METHOD,
        "execution": execution,
        "experiment": experiment,
        "config": config,
        "targets": target_map,
        "progress": progress,
        "stage": stage,
        "interrupted": interrupted,
        "stop_reason": reason,
        "source_progress": _read_json(root / "coverage-progress" / "latest.json"),
        "simulator_progress": _read_json(
            root / "simulator-coverage-progress" / "latest.json",
        ),
    }


def _target_queue_sha256(run: Mapping[str, Any], target_id: str) -> str | None:
    experiment = run.get("experiment")
    queues = experiment.get("target_queues") if isinstance(experiment, Mapping) else None
    queue = queues.get(target_id) if isinstance(queues, Mapping) else None
    value = queue.get("sha256") if isinstance(queue, Mapping) else None
    return value if isinstance(value, str) and value else None


def _same_target_queue(runs: list[dict[str, Any]], target_id: str) -> bool:
    digests = [_target_queue_sha256(run, target_id) for run in runs]
    return len(digests) > 1 and all(digest is not None for digest in digests) \
        and len(set(digests)) == 1


def _target_queue_position(run: Mapping[str, Any], target_id: str) -> int:
    progress = run.get("simulator_progress")
    targets = progress.get("targets") if isinstance(progress, Mapping) else None
    row = targets.get(target_id) if isinstance(targets, Mapping) else None
    value = row.get("queue_position") if isinstance(row, Mapping) else None
    return value if type(value) is int and value >= 0 else -1


def _queue_superseder(
    run: dict[str, Any], runs: list[dict[str, Any]], target_ids: list[str],
) -> dict[str, Any] | None:
    for candidate in runs:
        if candidate["run_id"] == run["run_id"]:
            continue
        if not all(_same_target_queue([run, candidate], target) for target in target_ids):
            continue
        older = [_target_queue_position(run, target) for target in target_ids]
        newer = [_target_queue_position(candidate, target) for target in target_ids]
        if all(a >= 0 and b >= a for a, b in zip(older, newer)) and any(
            b > a for a, b in zip(older, newer)
        ):
            return candidate
    return None


def _saved_progress_inputs(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = run["simulator_progress"].get("targets", {})
    inputs = {}
    for target_id in run["targets"]:
        row = targets.get(target_id, {}) if isinstance(targets, Mapping) else {}
        attempts = row.get("attempted_cases", 0)
        profiles = row.get("source_coverage_attempts", attempts)
        inputs[target_id] = {
            "attempts": attempts if type(attempts) is int else 0,
            "profiles": profiles if type(profiles) is int else 0,
            "data_profiles": None,
            "data_profiles_known": False,
            "outcomes": row.get("run_status_counts", {}),
            "statuses": row.get("simulator_coverage_status_counts", {}),
            "catalog_expected_cases": _target_queue_position(run, target_id),
        }
    return inputs


def _progress_inputs(
    run: dict[str, Any], output: Path, *, skip_legacy_replay: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    latest = run["source_progress"]
    if latest.get("run_id") == run["run_id"] and isinstance(latest.get("targets"), list):
        rows = {
            str(row.get("target")): row for row in latest["targets"]
            if isinstance(row, dict) and isinstance(row.get("target"), str)
        }
        inputs = {}
        for target_id in run["targets"]:
            row = rows.get(target_id, {})
            metrics = row.get("experiment_metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            inputs[target_id] = {
                "attempts": row.get("attempted_cases", 0),
                "profiles": row.get("profiles_seen", 0),
                "data_profiles": row.get("profiles_with_data", 0),
                "data_profiles_known": True,
                "outcomes": metrics.get("outcomes", {}),
                "statuses": metrics.get("statuses", {}),
                "catalog_expected_cases": (
                    (row.get("rv_opcode_catalog_coverage") or {}).get("expected_case_count", 0)
                    if isinstance(row.get("rv_opcode_catalog_coverage"), Mapping) else 0
                ),
            }
        return inputs, latest.get("rv_opcode_catalog_coverage") or {}

    if skip_legacy_replay:
        return _saved_progress_inputs(run), {}

    scratch = output / "work" / "staged" / run["run_id"]
    inputs, _ = coverage_progress._external_inputs(
        run["run_root"], METHOD,
        list(run["targets"].values()), scratch,
        defer_materialization=True, pending_by_target={},
    )
    catalog = coverage_progress._method_catalog_summary(
        METHOD, list(run["targets"].values()), inputs,
    )
    reduced = {}
    for target_id, item in inputs.items():
        reduced[target_id] = {
            name: item.get(name, {}) if name in {"outcomes", "statuses"}
            else int(item.get(name, 0) or 0)
            for name in ("attempts", "profiles", "data_profiles", "outcomes", "statuses",
                         "catalog_expected_cases")
        }
        reduced[target_id]["data_profiles_known"] = True
    return reduced, catalog


def _source_status(run_inputs: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = []
    for item in run_inputs:
        run_id = item["run_id"]
        if item["interrupted"]:
            reasons.append(f"{run_id}:run-interrupted")
        if not item["attempts_known"]:
            reasons.append(f"{run_id}:attempt-count-unknown")
        if item["attempts"] > item["profiles"]:
            reasons.append(f"{run_id}:attempt-profile-id-missing")
        if item.get("data_profiles_known", True) \
                and item["profiles"] > item["data_profiles"]:
            reasons.append(f"{run_id}:source-profile-input-missing")
    return ";".join(sorted(set(reasons))), reasons


def _link_profile_inputs(
    output: Path, target_id: str, collector: str,
    run_inputs: list[dict[str, Any]],
) -> tuple[Path, int, dict[str, int]]:
    work_root = Path(os.environ.get("RQ1_DIRECT_WORK_ROOT") or output / "work")
    destination = work_root / "raw" / target_id
    destination.mkdir(parents=True, exist_ok=True)
    file_count = 0
    data_profiles: dict[str, set[str]] = {}
    for run_index, item in enumerate(run_inputs):
        root = Path(item["raw_root"])
        found = data_profiles.setdefault(item["run_id"], set())
        for source in coverage_offline._raw_files(root, collector):
            try:
                resolved = source.resolve(strict=True)
                relative = source.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            try:
                has_data = resolved.stat().st_size > 0
            except OSError:
                has_data = False
            if has_data:
                parts = relative.parts
                profile_key = next((
                    f"{anchor}/{parts[index + 1]}"
                    for anchor in ("attempts", "profiles")
                    if anchor in parts
                    for index in [parts.index(anchor)]
                    if index + 1 < len(parts)
                ), str(relative.parent))
                found.add(profile_key)
            target = (
                destination / "runs" / f"{run_index:03d}-{item['run_id']}"
                / relative
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(resolved)
            except OSError:
                import shutil
                shutil.copy2(resolved, target)
            file_count += 1
    return destination, file_count, {
        run_id: len(profiles) for run_id, profiles in data_profiles.items()
    }


def _gcov_profile_inputs(
    run_inputs: list[dict[str, Any]],
) -> tuple[list[Path], int, dict[str, int], list[Path]]:
    """Index raw gcda directories without copying hundreds of thousands of links."""
    profiles: dict[tuple[str, str, str], Path] = {}
    data_profiles: dict[str, set[str]] = {}
    markers: list[Path] = []
    file_count = 0
    marker_name = source_coverage.GCOV_FLUSH_INCOMPLETE_MARKER
    for item in run_inputs:
        run_id = str(item["run_id"])
        root = Path(item["raw_root"])
        found = data_profiles.setdefault(run_id, set())
        run_root = Path(item["run_root"]).resolve()
        for source in coverage_offline._raw_files(root, "gcov"):
            try:
                resolved = source.resolve(strict=True)
                resolved.relative_to(run_root)
                relative = source.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            file_count += 1
            if source.name == marker_name:
                markers.append(resolved)
                continue
            if not source.name.endswith(".gcda"):
                continue
            parts = relative.parts
            anchor = next((name for name in ("attempts", "profiles") if name in parts), None)
            if anchor is None:
                raise ValueError(f"gcov-profile-root-missing:{relative}")
            index = parts.index(anchor)
            if index + 1 >= len(parts):
                raise ValueError(f"gcov-profile-id-missing:{relative}")
            profile_id = parts[index + 1]
            profile_root = root.joinpath(*parts[:index + 2])
            key = (run_id, anchor, profile_id)
            profiles[key] = profile_root
            if resolved.stat().st_size > 0:
                found.add(f"{anchor}/{profile_id}")
    active = [path for key, path in sorted(profiles.items())
              if key[1] + "/" + key[2] in data_profiles.get(key[0], set())]
    return active, file_count, {
        run_id: len(values) for run_id, values in data_profiles.items()
    }, markers


def _merge_gcov_profile_dirs(
    profile_dirs: list[Path], work_root: Path, target_id: str,
    markers: list[Path],
) -> tuple[Path, dict[str, Any]]:
    """Merge case-level GCDA trees before starting gcov once per object."""
    started = time.monotonic()
    tool = shutil.which("gcov-tool")
    if not tool:
        raise RuntimeError("gcov-tool-missing")
    if not profile_dirs:
        raise ValueError("gcov-profile-data-missing")
    target_work = work_root / "merged-gcov" / target_id
    if target_work.exists() and any(target_work.iterdir()):
        raise ValueError(f"gcov-merge-work-not-empty:{target_work}")
    target_work.mkdir(parents=True, exist_ok=True)
    try:
        jobs = int(os.environ.get("RQ1_GCOV_MERGE_JOBS", "2"))
        if jobs < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        jobs = 2
    jobs = min(jobs, 4)

    def reduce_profiles(inputs: list[Path], destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        current = list(inputs)
        owned = {path for path in current if path.is_relative_to(target_work)}
        level = 0
        while len(current) > 1:
            pairs = [(current[index], current[index + 1])
                     for index in range(0, len(current) - 1, 2)]
            outputs = [destination / f"merge-{level:03d}-{index:06d}"
                       for index in range(len(pairs))]

            def merge(pair: tuple[Path, Path], output: Path) -> Path:
                result = subprocess.run(
                    [tool, "merge", str(pair[0]), str(pair[1]), "--output", str(output)],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode:
                    detail = (result.stderr or result.stdout or "gcov-tool-merge-failed")
                    raise RuntimeError(detail[:800])
                return output

            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(merge, pair, output)
                           for pair, output in zip(pairs, outputs)]
                merged = [future.result() for future in futures]
            next_level = merged
            if len(current) % 2:
                next_level.append(current[-1])
            consumed = current[:len(pairs) * 2]
            for path in consumed:
                if path in owned:
                    shutil.rmtree(path)
            owned = {path for path in next_level if path.is_relative_to(target_work)}
            current = next_level
            level += 1
        final = current[0]
        if final.is_relative_to(target_work):
            return final
        copied = destination / "single-profile"
        shutil.copytree(final, copied)
        return copied

    batches_root = target_work / "batches"
    batch_results = []
    for index in range(0, len(profile_dirs), 64):
        batch = profile_dirs[index:index + 64]
        batch_results.append(reduce_profiles(
            batch, batches_root / f"batch-{index // 64:06d}",
        ))
    merged_root = reduce_profiles(batch_results, target_work / "final")
    if markers:
        marker_root = merged_root / ".rq1-flush-markers"
        for index, marker in enumerate(markers):
            destination = marker_root / f"{index:08d}" / marker_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(marker, destination)
    merged_files = sum(
        1 for path in merged_root.rglob("*.gcda")
        if not path.name.endswith(".tmp.gcda")
    )
    return merged_root, {
        "status": "merged", "tool": Path(tool).name,
        "input_profile_directories": len(profile_dirs),
        "input_incomplete_flush_markers": len(markers),
        "merged_gcda_files": merged_files,
        "batch_size": 64, "parallel_merges_per_target": jobs,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _unit_metric(
    rows: list[tuple[str, Mapping[str, Any] | None]], name: str,
    interrupted: bool,
) -> dict[str, Any]:
    covered: set[str] = set()
    outside: set[str] = set()
    valid_rows = 0
    incomplete = interrupted
    run_inputs = []
    for run_id, row in rows:
        metric = row.get(name) if isinstance(row, Mapping) else None
        units = metric.get("covered_units") if isinstance(metric, Mapping) else None
        valid = (
            isinstance(metric, Mapping)
            and isinstance(units, list)
            and all(isinstance(unit, str) for unit in units)
            and len(set(units)) == len(units)
            and set(units) <= rv_instruction_coverage.OPCODE_CATALOG_KEYS
            and type(metric.get("covered")) is int
            and metric["covered"] == len(units)
            and metric.get("eligible") == rv_instruction_coverage.OPCODE_CATALOG_DENOMINATOR
            and metric.get("registry_sha256") == rv_instruction_coverage.OPCODE_CATALOG_REGISTRY_SHA256
        )
        if valid:
            valid_rows += 1
            covered.update(units)
            values = metric.get("out_of_catalog_units", ())
            if isinstance(values, (list, tuple)):
                outside.update(str(value) for value in values)
            if metric.get("status") != "observed":
                incomplete = True
        else:
            incomplete = True
        run_inputs.append({
            "run_id": run_id,
            "status": metric.get("status", "gap") if isinstance(metric, Mapping) else "gap",
            "covered": metric.get("covered") if isinstance(metric, Mapping) else None,
            "eligible": metric.get("eligible") if isinstance(metric, Mapping) else None,
        })
    status = "gap" if valid_rows == 0 else "partial" if incomplete else "observed"
    metric = {
        "covered": len(covered),
        "eligible": rv_instruction_coverage.OPCODE_CATALOG_DENOMINATOR,
        "value": round(len(covered) / rv_instruction_coverage.OPCODE_CATALOG_DENOMINATOR, 6)
        if status == "observed" else None,
        "provisional_value": round(len(covered) / rv_instruction_coverage.OPCODE_CATALOG_DENOMINATOR, 6)
        if valid_rows else None,
        "status": status,
        "covered_units": sorted(covered),
        "registry_sha256": rv_instruction_coverage.OPCODE_CATALOG_REGISTRY_SHA256,
        "aggregation": "unique-opcode-set-union-across-runs",
        "partition_coverage": {},
        "run_inputs": run_inputs,
    }
    for partition, units in rv_instruction_coverage.OPCODE_CATALOG_PARTITION_KEYS.items():
        count = len(covered & units)
        metric["partition_coverage"][partition] = {
            "covered": count, "eligible": len(units),
            "value": round(count / len(units), 6)
            if len(units) and status == "observed" else None,
            "provisional_value": round(count / len(units), 6)
            if len(units) and valid_rows else None,
        }
    if outside:
        metric["out_of_catalog_units"] = sorted(outside)
    if status == "partial":
        metric["reason"] = "interrupted-or-incomplete-run-evidence"
    elif status == "gap":
        metric["reason"] = "opcode-catalog-evidence-missing"
    return metric


def _catalog_rows(summary: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any] | None]:
    rows = summary.get("target_runs")
    by_target = {
        str(row.get("target")): row for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("target"), str)
    } if isinstance(rows, list) else {}
    unions = summary.get("method_generated_unions")
    generated = next((row for row in unions if isinstance(row, Mapping)), None) \
        if isinstance(unions, list) else None
    return by_target, generated


def _merge_catalogs(
    runs: list[dict[str, Any]], catalogs: list[Mapping[str, Any]], target_ids: list[str],
) -> dict[str, Any]:
    indexed = [_catalog_rows(summary) for summary in catalogs]
    interrupted = any(run["interrupted"] for run in runs)
    target_runs = []
    for target_id in target_ids:
        same_queue = _same_target_queue(runs, target_id)
        source_runs = [max(runs, key=lambda run: _target_queue_position(run, target_id))] \
            if same_queue else runs
        source_run_ids = {run["run_id"] for run in source_runs}
        selected = [
            (run["run_id"], rows.get(target_id))
            for run, (rows, _generated) in zip(runs, indexed)
            if run["run_id"] in source_run_ids
        ]
        metrics = {
            name: _unit_metric(
                [(run_id, row.get("metrics", {}) if isinstance(row, Mapping) else None)
                 for run_id, row in selected],
                name, interrupted,
            )
            for name in ("GenOpcodeCov", "ExecOpcodeCov")
        }
        target_runs.append({
            "method": METHOD,
            "stratum": "bare-metal",
            "target": target_id,
            "case_count": max((int(row.get("case_count", 0) or 0)
                               for _run_id, row in selected if isinstance(row, Mapping)),
                              default=0) if same_queue else sum(
                                  int(row.get("case_count", 0) or 0)
                                  for _run_id, row in selected
                                  if isinstance(row, Mapping)
                              ),
            "case_count_aggregation": "most-advanced-identical-queue-snapshot"
            if same_queue else "sum-across-distinct-target-queues",
            "aggregation_scope": "method-target-run",
            "input_scopes": [{"run_id": run["run_id"]} for run in runs],
            "evidence_run_ids": [run_id for run_id, _row in selected],
            "metrics": metrics,
            "status": "gap" if any(m["status"] == "gap" for m in metrics.values())
            else "partial" if any(m["status"] == "partial" for m in metrics.values())
            else "observed",
        })
    identical_queues = all(_same_target_queue(runs, target) for target in target_ids)
    generated_runs = [max(runs, key=lambda run: sum(
        _target_queue_position(run, target) for target in target_ids
    ))] if identical_queues else runs
    generated_run_ids = {run["run_id"] for run in generated_runs}
    generated_rows = [
        (run["run_id"], generated.get("metrics", {})
         if isinstance(generated, Mapping) else None)
        for run, (_targets, generated) in zip(runs, indexed)
        if run["run_id"] in generated_run_ids
    ]
    generated_metric = _unit_metric(generated_rows, "GenOpcodeCov", interrupted)
    contract = runs[-1]["config"].get("rv_opcode_catalog_coverage", {})
    return {
        "schema_version": rv_instruction_coverage.OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "catalog-wide-comparison",
        "catalog_id": rv_instruction_coverage.OPCODE_CATALOG_ID,
        "key_schema": rv_instruction_coverage.OPCODE_KEY_SCHEMA,
        "registry_sha256": rv_instruction_coverage.OPCODE_CATALOG_REGISTRY_SHA256,
        "catalog_form_count": contract.get("form_count"),
        "opcode_denominator": rv_instruction_coverage.OPCODE_CATALOG_DENOMINATOR,
        "partitions": {
            partition: {"opcode_denominator": len(units)}
            for partition, units in rv_instruction_coverage.OPCODE_CATALOG_PARTITION_KEYS.items()
        },
        "aggregation": "unique-opcode-set-union-across-runs; no-case-averaging",
        "target_runs": target_runs,
        "method_generated_unions": [{
            "method": METHOD,
            "aggregation_scope": "method-generated-elf-union",
            "metrics": {"GenOpcodeCov": generated_metric},
        }],
    }


def _merge_pcov(runs: list[dict[str, Any]], target_id: str) -> dict[str, Any]:
    if _same_target_queue(runs, target_id):
        candidates = []
        for run in runs:
            targets = run["simulator_progress"].get("targets", {})
            row = targets.get(target_id) if isinstance(targets, Mapping) else None
            pcov = row.get("case_artifact_pcov") if isinstance(row, Mapping) else None
            if isinstance(pcov, Mapping):
                candidates.append((
                    _target_queue_position(run, target_id),
                    int(pcov.get("cases_with_denominator", 0) or 0),
                    str(run["run_id"]), run, pcov,
                ))
        if candidates:
            selected_position, _cases, selected_id, selected_run, selected = max(
                candidates, key=lambda item: (item[0], item[1], item[2]),
            )
            covered = selected.get("covered_addresses_across_case_artifacts")
            eligible = selected.get("eligible_addresses_across_case_artifacts")
            cases = selected.get("cases_with_denominator")
            valid = all(type(value) is int and value >= 0
                        for value in (covered, eligible, cases))
            status = selected.get("status", "gap") if valid else "gap"
            if valid and selected_run["interrupted"] and status == "observed":
                status = "partial"
            result = {
                "aggregation": "most-advanced-snapshot-for-identical-target-queue",
                "input_queue_sha256": _target_queue_sha256(runs[0], target_id),
                "selected_run_id": selected_id,
                "selected_queue_position": selected_position,
                "covered_addresses_across_case_artifacts": covered if valid else None,
                "eligible_addresses_across_case_artifacts": eligible if valid else None,
                "cases_with_denominator": cases if valid else 0,
                "value": round(covered / eligible, 6)
                if valid and eligible and status == "observed" else None,
                "provisional_value": round(covered / eligible, 6)
                if valid and eligible else None,
                "status": status,
                "reason": "identical-input-queue; prior-prefix-artifacts-not-double-counted",
                "run_inputs": [],
            }
            for run in runs:
                targets = run["simulator_progress"].get("targets", {})
                row = targets.get(target_id) if isinstance(targets, Mapping) else None
                pcov = row.get("case_artifact_pcov") if isinstance(row, Mapping) else None
                result["run_inputs"].append({
                    "run_id": run["run_id"],
                    "queue_position": _target_queue_position(run, target_id),
                    "selected_for_unique_queue_snapshot": run["run_id"] == selected_id,
                    "status": pcov.get("status", "gap")
                    if isinstance(pcov, Mapping) else "gap",
                    "cases_with_denominator": pcov.get("cases_with_denominator")
                    if isinstance(pcov, Mapping) else None,
                })
            return result

    covered = eligible = cases = 0
    evidence = 0
    incomplete = any(run["interrupted"] for run in runs)
    run_rows = []
    for run in runs:
        targets = run["simulator_progress"].get("targets", {})
        row = targets.get(target_id) if isinstance(targets, Mapping) else None
        pcov = row.get("case_artifact_pcov") if isinstance(row, Mapping) else None
        if not isinstance(pcov, Mapping):
            incomplete = True
            run_rows.append({"run_id": run["run_id"], "status": "gap"})
            continue
        c, e, n = (pcov.get("covered_addresses_across_case_artifacts"),
                   pcov.get("eligible_addresses_across_case_artifacts"),
                   pcov.get("cases_with_denominator"))
        valid = all(type(value) is int and value >= 0 for value in (c, e, n))
        if valid:
            evidence += 1
            covered += c
            eligible += e
            cases += n
        if pcov.get("status") != "observed":
            incomplete = True
        run_rows.append({
            "run_id": run["run_id"], "status": pcov.get("status", "gap"),
            "covered": c, "eligible": e, "cases_with_denominator": n,
        })
    status = "gap" if not evidence or eligible == 0 else "partial" if incomplete else "observed"
    return {
        "aggregation": "sum-per-case-PC-address-counts-across-runs",
        "covered_addresses_across_case_artifacts": covered if evidence else None,
        "eligible_addresses_across_case_artifacts": eligible if evidence else None,
        "cases_with_denominator": cases if evidence else 0,
        "value": round(covered / eligible, 6) if eligible and status == "observed" else None,
        "provisional_value": round(covered / eligible, 6) if eligible else None,
        "status": status,
        "run_inputs": run_rows,
    }


def _collect_source_target(payload: dict[str, Any]) -> dict[str, Any]:
    target_id = payload["target_id"]
    target = payload["target"]
    output = Path(payload["output"])
    run_inputs = payload["run_inputs"]
    started = time.monotonic()
    spec = target.get("source_coverage")
    spec = dict(spec) if isinstance(spec, Mapping) else {}
    collector = str(spec.get("collector") or "")
    profile_dir = output / "profiles" / target_id
    spec["profile"] = str(profile_dir / "lcov.info")
    spec["identity"] = str(profile_dir / "identity.json")
    target_copy = {**target, "source_coverage": spec}
    attempts = sum(item["attempts"] for item in run_inputs)
    expected = attempts if collector in {"llvm", "dotnet"} else None
    input_files = 0
    raw_root = output / "work" / "raw" / target_id
    gcov_merge: dict[str, Any] | None = None
    try:
        stage_started = time.monotonic()
        print(f"direct-input-staging-start target={target_id}", flush=True)
        work_root = Path(os.environ.get("RQ1_DIRECT_WORK_ROOT") or output / "work")
        if collector in {"gcov", "lcov"}:
            profile_dirs, input_files, data_profiles, markers = _gcov_profile_inputs(
                run_inputs,
            )
            print(f"direct-profile-merge-start target={target_id} "
                  f"profiles={len(profile_dirs)} files={input_files}", flush=True)
            raw_root, gcov_merge = _merge_gcov_profile_dirs(
                profile_dirs, work_root, target_id, markers,
            )
        else:
            raw_root, input_files, data_profiles = _link_profile_inputs(
                output, target_id, collector, run_inputs,
            )
        for item in run_inputs:
            item["data_profiles"] = data_profiles.get(item["run_id"], 0)
            item["data_profiles_known"] = True
        print(f"direct-collector-start target={target_id} files={input_files} "
              f"merged_gcda={(gcov_merge or {}).get('merged_gcda_files')} "
              f"staging_s={time.monotonic() - stage_started:.1f}", flush=True)
        binary = coverage_offline._dependency_path(target.get("coverage_binary"))
        lock_key = str(spec.get("build_root") or spec.get("source_root") or target_id)
        lock_path = work_root / "locks" / (
            __import__("hashlib").sha256(lock_key.encode()).hexdigest() + ".lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            collection = source_coverage.collect_source_coverage_batch(
                output, target_copy, binary, raw_root,
                expected_coverage_cases=expected,
            )
        _reason, reasons = _source_status(run_inputs)
        if input_files == 0 and attempts:
            reasons.append("source-profile-inputs-missing")
        if collection.get("status") == "partial" and collection.get("reason"):
            reasons.append(str(collection["reason"]))
        if reasons and collection.get("status") in {"observed", "partial"}:
            collection = {
                **collection, "status": "partial",
                "reason": ";".join(sorted(set(reasons))),
            }
        source = target_source_coverage(
            output, target_copy,
            coverage_binary_sha=collection.get("coverage_binary_sha"),
            source_commit_observed=collection.get("source_commit_observed"),
            coverage_binary_sha256=collection.get("coverage_binary_sha256"),
            collection_status=collection.get("status"),
            collection_reason=collection.get("reason"),
        )
        return {
            "target": target_id, "status": source.get("status", "gap"),
            "attempted_cases": attempts,
            "source_profiles": sum(item["profiles"] for item in run_inputs),
            "profiles_with_data": sum(item["data_profiles"] for item in run_inputs),
            "source_profile_files": input_files,
            "run_inputs": [
                {key: item[key] for key in (
                    "run_id", "attempts", "profiles", "data_profiles",
                    "data_profiles_known", "attempts_known", "interrupted",
                    "stop_reason",
                )}
                for item in run_inputs
            ],
            "collection": collection, "source_coverage": source,
            "gcov_profile_merge": gcov_merge,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        reason = f"offline-source-collection:{type(error).__name__}"
        return {
            "target": target_id, "status": "gap", "attempted_cases": attempts,
            "source_profiles": 0, "profiles_with_data": 0,
            "source_profile_files": input_files,
            "collection": {"status": "gap", "reason": reason,
                           "error": str(error)[:500]},
            "source_coverage": coverage_offline._source_metrics_gap(reason),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    finally:
        print(f"direct-collector-finished target={target_id} "
              f"elapsed_s={time.monotonic() - started:.1f}", flush=True)


def _target_identity_matches(runs: list[dict[str, Any]], target_id: str) -> bool:
    signatures = {
        coverage_offline._identity_signature(run["targets"][target_id])
        for run in runs
    }
    return len(signatures) == 1


def aggregate(
    run_roots: list[Path], output: Path, workers: int = 7,
    *, include_guest_metrics: bool = True,
) -> dict[str, Any]:
    runs = [_run_record(root) for root in run_roots]
    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run id")
    target_ids = sorted(runs[0]["targets"])
    if any(set(run["targets"]) != set(target_ids) for run in runs[1:]):
        raise ValueError("Direct runs do not share the same seven Targets")

    output = output.resolve()
    if any(output == run["run_root"] or run["run_root"] in output.parents for run in runs):
        raise ValueError("output must be outside every input run")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_inputs_by_target: dict[str, list[dict[str, Any]]] = {target: [] for target in target_ids}
    run_input_data: list[dict[str, dict[str, Any]]] = []
    catalog_summaries: list[Mapping[str, Any]] = []
    for run in runs:
        superseder = _queue_superseder(run, runs, target_ids)
        run["queue_superseder"] = superseder["run_id"] if superseder else None
        inputs, catalog = _progress_inputs(
            run, output, skip_legacy_replay=superseder is not None,
        )
        run_input_data.append(inputs)
        catalog_summaries.append(
            catalog if include_guest_metrics and isinstance(catalog, Mapping) else {}
        )
        progress_root = (
            output / "work" / "staged" / run["run_id"]
            if run["source_progress"].get("run_id") != run["run_id"]
            or not isinstance(run["source_progress"].get("targets"), list)
            else run["run_root"] / "coverage-progress"
        )
        for target_id in target_ids:
            values = inputs.get(target_id, {})
            raw_root = progress_root / "inputs" / target_id / "raw"
            if not raw_root.is_dir():
                raw_root = run["run_root"] / "coverage-raw" / target_id
            run_inputs_by_target[target_id].append({
                "run_id": run["run_id"], "run_root": str(run["run_root"]),
                "raw_root": str(raw_root),
                "attempts": int(values.get("attempts", 0) or 0),
                "profiles": int(values.get("profiles", 0) or 0),
                "data_profiles": values.get("data_profiles")
                if type(values.get("data_profiles")) is int else None,
                "data_profiles_known": values.get("data_profiles_known") is not False,
                "attempts_known": True,
                "interrupted": run["interrupted"],
                "stop_reason": run["stop_reason"] or run["stage"] or None,
            })

    rv_catalog = (
        _merge_catalogs(runs, catalog_summaries, target_ids)
        if include_guest_metrics else {
            "status": "disabled", "reason": "guest-metrics-disabled",
            "target_runs": [], "method_generated_unions": [],
        }
    )
    identity_matches = {target: _target_identity_matches(runs, target) for target in target_ids}
    target_rows = []
    for target_id in target_ids:
        input_rows = run_inputs_by_target[target_id]
        attempts = sum(row["attempts"] for row in input_rows)
        data_profiles_known = all(row["data_profiles_known"] for row in input_rows)
        profiles_with_data = sum(row["data_profiles"] or 0 for row in input_rows) \
            if data_profiles_known else None
        outcomes: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for inputs in run_input_data:
            data = inputs.get(target_id, {})
            for key, dest in (("outcomes", outcomes), ("statuses", statuses)):
                values = data.get(key)
                if isinstance(values, Mapping):
                    for name, count in values.items():
                        if type(count) is int:
                            dest[str(name)] = dest.get(str(name), 0) + count
        target_rows.append({
            "target": target_id,
            "identity_match": identity_matches[target_id],
            "attempted_cases": attempts,
            "source_profiles": sum(row["profiles"] for row in input_rows),
            "profiles_with_data": profiles_with_data,
            "profiles_with_data_known": data_profiles_known,
            "run_inputs": [
                {key: row[key] for key in (
                    "run_id", "attempts", "profiles", "data_profiles",
                    "data_profiles_known", "interrupted", "stop_reason",
                )}
                for row in input_rows
            ],
            "outcomes": outcomes,
            "statuses": statuses,
            "case_artifact_pcov": _merge_pcov(runs, target_id),
            "rv_opcode_catalog_coverage": next(
                (row for row in rv_catalog["target_runs"] if row["target"] == target_id),
                {"status": "disabled", "reason": "guest-metrics-disabled"},
            ) if include_guest_metrics else {
                "status": "disabled", "reason": "guest-metrics-disabled",
            },
            "source_coverage": None,
            "collection": {"status": "processing"},
        })

    random_queue_hashes = [
        run["experiment"].get("random_queue_sha256") for run in runs
    ]
    random_queue_matches = len(runs) > 1 and all(
        isinstance(value, str) and value for value in random_queue_hashes
    ) and len(set(random_queue_hashes)) == 1
    progress = {
        "schema_version": SCHEMA, "status": "processing",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD, "aggregation_scope": "one-method-seven-targets",
        "runs": [{
            "run_id": run["run_id"],
            "started_at_utc": run["execution"].get("started_at_utc"),
            "duration_seconds": run["execution"].get("duration_seconds"),
            "source_commit": run["execution"].get("source_commit"),
            "source_dirty": run["execution"].get("source_dirty"),
            "config_sha256": run["execution"].get("config_sha256"),
            "source_snapshot_sha256": run["experiment"].get("source_snapshot_sha256"),
            "random_queue_sha256": run["experiment"].get("random_queue_sha256"),
            "queue_superseded_by": run.get("queue_superseder"),
            "stage": run["stage"], "interrupted": run["interrupted"],
            "stop_reason": run["stop_reason"],
        } for run in runs],
        "input_queue_overlap": {
            "random_queue_sha256_matches": random_queue_matches,
            "target_queue_sha256_matches": {
                target_id: _same_target_queue(runs, target_id)
                for target_id in target_ids
            },
            "pcov_rule": (
                "most-advanced-queue-position-snapshot-per-target-when-queues-match"
            ),
        },
        "targets": target_rows,
    }
    _write_json(output / "progress.json", progress)

    jobs = []
    for target_id in target_ids:
        if not identity_matches[target_id]:
            continue
        target = runs[-1]["targets"][target_id]
        jobs.append({
            "target_id": target_id, "target": target,
            "run_inputs": run_inputs_by_target[target_id],
            "output": str(output),
        })
    results = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)) or 1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_collect_source_target, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job["target_id"]] = future.result()
            except Exception as error:
                target_id = job["target_id"]
                results[target_id] = {
                    "target": target_id, "status": "gap",
                    "collection": {"status": "gap",
                                   "reason": f"offline-worker-error:{type(error).__name__}",
                                   "error": str(error)[:500]},
                    "source_coverage": coverage_offline._source_metrics_gap(
                        f"offline-worker-error:{type(error).__name__}",
                    ),
                }
            for row in target_rows:
                if row["target"] == job["target_id"]:
                    row.update(results[job["target_id"]])
                    break
            progress["targets"] = target_rows
            progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(output / "progress.json", progress)
            print(f"direct-target-published target={job['target_id']} "
                  f"status={results[job['target_id']].get('status')}", flush=True)

    for target_id, matches in identity_matches.items():
        if matches:
            continue
        reason = "target-coverage-identity-mismatch"
        row = next(item for item in target_rows if item["target"] == target_id)
        row.update({
            "status": "gap", "collection": {"status": "gap", "reason": reason},
            "source_coverage": coverage_offline._source_metrics_gap(reason),
        })
    overall_status = (
        "gap" if all(row.get("status") == "gap" for row in target_rows)
        else "partial" if any(row.get("status") != "observed" for row in target_rows)
        else "observed"
    )
    summary = {
        **progress,
        "status": overall_status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": [
            "SimSrcCov.function", "SimSrcCov.line", "SimSrcCov.branch",
            "PCov.case_artifact_address_sum",
            *( ["GenOpcodeCov", "ExecOpcodeCov"] if include_guest_metrics else []),
        ],
        "rv_opcode_catalog_coverage": rv_catalog,
        "guest_metrics_enabled": include_guest_metrics,
        "pcov_aggregation_note": (
            "Identical target queues use the most advanced case-artifact snapshot to avoid double counting; distinct queues sum per-case address counts."
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "progress.json", {**progress, "status": "complete"})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True,
                        help="Direct run root; repeat to merge multiple runs")
    parser.add_argument("--output", required=True, help="New aggregate output directory")
    parser.add_argument("--workers", type=int, default=7,
                        help="parallel Target collectors (default: 7)")
    parser.add_argument(
        "--source-only", action="store_true",
        help="只汇总 SimSrcCov；RV opcode 作为 disabled 诊断保留",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = aggregate([Path(value) for value in args.run_root],
                       Path(args.output), args.workers,
                       include_guest_metrics=not args.source_only)
    print(f"direct-offline-summary={Path(args.output) / 'summary.json'} "
          f"status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

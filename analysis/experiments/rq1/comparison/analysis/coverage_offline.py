"""离线重算并合并一轮或多轮 RQ1 7×7 SimSrcCov 原始 profile。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

COMPARISON_ROOT = Path(__file__).resolve().parents[1]
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from analysis import source_coverage
from analysis import ssd_trace_stage
from analysis import simulator_coverage
from analysis.simulator_coverage import SOURCE_SCHEMA, target_source_coverage
from analysis import rv_instruction_coverage as rv_catalog
import coverage_progress as cp


SCHEMA = "rq1-offline-simulator-source-coverage-v1"
UNIQUE_SCHEMA = "rq1-offline-simsrc-unique-coverage-v1"
_SOURCE_METRICS = ("function", "line", "branch")
_PROFILE_ID = re.compile(r"[0-9a-f]{64}")
_CASE_CONFIG = re.compile(r"lane-(\d+)-case-(\d+)\.json$")
_CASE_DIR = re.compile(r"case-(\d+)$")
_INTERRUPTION_REASONS = {
    "external-signal", "resource-limit", "container-failure",
    "container-signal-killed", "interrupted-unknown",
}
_TARGET_MANIFEST_FIELDS = (
    "kind", "execution_model", "translation_mode", "commit", "binary",
    "binary_sha256", "coverage_binary", "coverage_binary_sha256",
    "simulator_coverage", "identity", "identity_digest",
)


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _identity_signature(target: Mapping[str, Any]) -> str:
    spec = target.get("source_coverage")
    spec = dict(spec) if isinstance(spec, Mapping) else {}
    spec.pop("profile", None)
    spec.pop("identity", None)
    identity = {
        "target": target.get("id"),
        "kind": target.get("kind"),
        "commit": target.get("commit"),
        "binary_sha256": target.get("binary_sha256"),
        "coverage_binary": target.get("coverage_binary"),
        "coverage_binary_sha256": target.get("coverage_binary_sha256"),
        "source_coverage": spec,
    }
    return hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode()).hexdigest()


def _execution_targets_match(execution: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    configured = config.get("targets")
    recorded = execution.get("targets")
    if not isinstance(configured, list) or not isinstance(recorded, list):
        return False
    config_by_id = {
        item.get("id"): item for item in configured if isinstance(item, dict)
    }
    recorded_by_id = {
        item.get("id"): item for item in recorded if isinstance(item, dict)
    }
    if len(config_by_id) != 7 or len(recorded_by_id) != 7 \
            or set(config_by_id) != set(recorded_by_id):
        return False
    return all(
        config_by_id[target_id].get(field) == recorded_by_id[target_id].get(field)
        for target_id in config_by_id
        for field in _TARGET_MANIFEST_FIELDS
    )


def _dependency_path(value: object) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        return Path(os.environ.get("RQ1_DEPS", "/path/to/deps")) / path
    return source_coverage._dependency_path(path)


def _load_launches(paths: list[Path]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    methods: list[str] | None = None
    targets: list[str] | None = None
    experiments = []
    run_ids: set[str] = set()
    manifest_paths: set[Path] = set()
    for value in paths:
        path = value.resolve()
        if path in manifest_paths:
            raise ValueError(f"duplicate launch manifest: {path}")
        manifest_paths.add(path)
        payload = _read_json(path)
        if payload.get("schema_version") != "rq1-long-run-launch-v1":
            raise ValueError(f"unsupported launch manifest: {path}")
        current_methods = payload.get("methods")
        current_targets = payload.get("targets")
        if not isinstance(current_methods, list) or len(current_methods) != 7 \
                or not all(isinstance(item, str) for item in current_methods) \
                or len(set(current_methods)) != 7:
            raise ValueError(f"launch manifest must list seven unique methods: {path}")
        if not isinstance(current_targets, list) or len(current_targets) != 7 \
                or not all(isinstance(item, str) for item in current_targets) \
                or len(set(current_targets)) != 7:
            raise ValueError(f"launch manifest must list seven unique Targets: {path}")
        if methods is None:
            methods, targets = list(current_methods), list(current_targets)
        elif set(methods) != set(current_methods) or set(targets) != set(current_targets):
            raise ValueError("launch manifests do not describe the same 7×7 matrix")

        runs = payload.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"launch manifest has no run list: {path}")
        results_root = path.parents[2]
        run_rows = []
        seen_methods: set[str] = set()
        for item in runs:
            if not isinstance(item, dict):
                continue
            method, run_id = item.get("method"), item.get("run_id")
            if method not in current_methods or not isinstance(run_id, str) or not run_id:
                raise ValueError(f"invalid run entry in {path}")
            if method in seen_methods or run_id in run_ids:
                raise ValueError(f"duplicate method or run id in launch manifests: {run_id}")
            seen_methods.add(method)
            run_ids.add(run_id)
            run_root = results_root / "runs" / run_id
            execution = _read_json(run_root / "execution-manifest.json")
            config = _read_json(run_root / "config.snapshot.json")
            if method.startswith("Ours-"):
                allowed_actions = {"framework-run"}
            else:
                # External tools are replayed from the pinned case catalog in
                # current 7×7 runs; older combined runs used the action "run".
                allowed_actions = {"run", "execute-existing-queues"}
            error = None
            if not run_root.is_dir():
                error = "run-root-missing"
            elif not execution or not config:
                error = "run-manifest-or-config-missing"
            elif execution.get("method_filter") != method:
                error = "run-method-mismatch"
            elif execution.get("action") not in allowed_actions:
                error = "run-action-mismatch"
            elif not _execution_targets_match(execution, config):
                error = "run-target-identity-mismatch"
            elif execution.get("coverage_enabled") is not True:
                error = "coverage-disabled"
            else:
                config_targets = config.get("targets")
                config_ids = [
                    row.get("id") for row in config_targets
                    if isinstance(row, dict)
                ] if isinstance(config_targets, list) else []
                if len(config_ids) != 7 or set(config_ids) != set(current_targets):
                    error = "run-target-matrix-mismatch"
            run_result = _read_json(run_root / "run-result.json")
            wrapper = _read_json(run_root / "wrapper-result.json")
            stop = _read_json(run_root / "stop-provenance.json")
            interrupted_reason = (
                stop.get("reason_code")
                or run_result.get("stop_reason")
                or run_result.get("partial_reason")
                or wrapper.get("reason_code")
            )
            interrupted = (
                interrupted_reason in _INTERRUPTION_REASONS
                or (run_root / "external-interruption.json").is_file()
            )
            run_rows.append({
                "run_id": run_id,
                "method": method,
                "launch_status": item.get("status"),
                "run_root": run_root,
                "execution": execution,
                "config": config,
                "run_result": run_result,
                "wrapper": wrapper,
                "interrupted": interrupted,
                "stop_reason": interrupted_reason,
                "started_at_utc": execution.get("started_at_utc"),
                "finished_at_utc": wrapper.get("finished_at_utc")
                or stop.get("finished_at_utc"),
                "error": error,
            })
        if seen_methods != set(current_methods):
            raise ValueError(f"launch manifest does not contain all seven methods: {path}")
        experiments.append({
            "launch_manifest": str(path),
            "run_prefix": payload.get("run_prefix"),
            "source_commit": payload.get("source_commit"),
            "config_sha256": payload.get("config_sha256"),
            "runs": run_rows,
        })
    return methods or [], targets or [], experiments


def _target_specs(
    experiments: list[dict[str, Any]], targets: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    signatures: dict[tuple[str, str], set[str]] = {}
    for experiment in experiments:
        for run in experiment["runs"]:
            method = str(run.get("method") or "")
            config_targets = run["config"].get("targets")
            if not isinstance(config_targets, list):
                continue
            for target in config_targets:
                if not isinstance(target, dict) or target.get("id") not in targets:
                    continue
                target_id = str(target["id"])
                key = (method, target_id)
                signatures.setdefault(key, set()).add(_identity_signature(target))
                specs.setdefault(key, target)
    for key, values in signatures.items():
        if len(values) > 1:
            specs[key] = {
                **specs.get(key, {"id": key[1]}),
                "_identity_mismatch": True,
            }
    return specs


def _run_path(value: object, run_root: Path, case_root: Path | None = None) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        return (case_root or run_root) / path
    prefix = Path("/opt/runs") / run_root.name
    try:
        return run_root / path.relative_to(prefix)
    except ValueError:
        return source_coverage._dependency_path(path)


@lru_cache(maxsize=32)
def _ledger_events(run_root_text: str) -> tuple[tuple[dict[str, Any], ...], bool]:
    run_root = Path(run_root_text)
    ledger = run_root / "ledger" / "events.jsonl"
    if not ledger.is_file():
        ledger = run_root / "ledger" / "events.partial.jsonl"
    if not ledger.is_file():
        return (), False
    events, known = [], True
    try:
        with ledger.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (TypeError, ValueError):
                    known = False
                    continue
                if isinstance(value, dict):
                    events.append(value)
                else:
                    known = False
    except (OSError, UnicodeError):
        return tuple(events), False
    return tuple(events), known


def _events_for_external_run(
    run_root: Path, target_id: str,
) -> tuple[int, bool, list[Path], dict[Path, dict[str, Any]]]:
    events, known = _ledger_events(str(run_root.resolve()))
    if not events and not known:
        return 0, False, [], {}
    raw_root = run_root / "coverage-raw" / _safe(target_id)
    attempts = 0
    profile_roots: set[Path] = set()
    profile_cases: dict[Path, dict[str, Any]] = {}
    for event in events:
        if event.get("target") != target_id or event.get("target_attempted") is not True:
            continue
        attempts += 1
        profile_id = event.get("source_coverage_profile_id")
        if isinstance(profile_id, str) and _PROFILE_ID.fullmatch(profile_id):
            profile_root = raw_root / "attempts" / profile_id
            profile_roots.add(profile_root)
            profile_cases[profile_root] = {
                "case_id": event.get("case_id")
                or (f"candidate-{event['stream_index']}"
                    if type(event.get("stream_index")) is int else profile_id),
                "stream_index": event.get("stream_index"),
                "profile_id": profile_id,
                "status": event.get("status"),
                "outcome": event.get("outcome"),
                "reason_code": event.get("reason_code"),
                "termination": event.get("termination"),
                "coverage_flush_status": event.get("coverage_flush_status"),
                "coverage_flush_returncode": event.get("coverage_flush_returncode"),
                "coverage_flush_forced_cleanup": event.get(
                    "coverage_flush_forced_cleanup",
                ),
            }
        else:
            known = False
    return attempts, known, sorted(profile_roots), profile_cases


def _campaign_attempted(case_dir: Path, target_id: str) -> bool | None:
    saved_status = None
    for name in ("campaign-progress.json", "campaign-result.json"):
        campaign = _read_json(case_dir / name)
        summaries = campaign.get("target_summaries")
        summary = summaries.get(target_id) if isinstance(summaries, Mapping) else None
        if isinstance(summary, Mapping):
            counts = summary.get("counts")
            attempted = counts.get("target_attempted") \
                if isinstance(counts, Mapping) else None
            if type(attempted) is int and attempted >= 0:
                if attempted > 0:
                    return True
                saved_status = False
            records = summary.get("record_count")
            baselines = summary.get("baseline_count")
            if saved_status is None and type(records) is int and type(baselines) is int:
                saved_status = records + baselines > 0
        records = campaign.get("target_records_by_target")
        baselines = campaign.get("target_baselines_by_target")
        if isinstance(records, Mapping) or isinstance(baselines, Mapping):
            recorded = bool(
                (records.get(target_id) if isinstance(records, Mapping) else None)
                or (baselines.get(target_id) if isinstance(baselines, Mapping) else None)
            )
            if recorded:
                return True
    checkpoint = _read_json(case_dir / "reference-chain-checkpoint.json")
    campaign = checkpoint.get("campaign")
    if isinstance(campaign, Mapping):
        records = campaign.get("target_records_by_target")
        baselines = campaign.get("target_baselines_by_target")
        if isinstance(records, Mapping) or isinstance(baselines, Mapping):
            recorded = bool(
                (records.get(target_id) if isinstance(records, Mapping) else None)
                or (baselines.get(target_id) if isinstance(baselines, Mapping) else None)
            )
            if recorded:
                return True
    # Raw-only framework campaigns deliberately leave the producer-side
    # target record arrays empty. The resident Target owns the durable result
    # sidecars, so use those as the execution fact when the campaign has no
    # target summary. A result with target_attempted=false is an artifact gap,
    # not a simulator attempt.
    try:
        method_root = case_dir.resolve().parents[3]
        case_relative = case_dir.resolve().relative_to(method_root)
        raw_root = (
            method_root / "target-queues" / _safe(target_id)
            / "raw" / "cases" / case_relative
        )
    except (OSError, ValueError, IndexError):
        raw_root = None
    if raw_root is not None and any(
        path.is_file()
        and not path.name.endswith(".raw-artifact-manifest.json")
        and (path.name == "baseline.json" or path.name.startswith("candidate-"))
        for path in raw_root.glob("*.json")
    ):
        return True

    result_seen = False
    result_root = case_dir / "target-replay-queue" / "results"
    for path in sorted(result_root.glob("target-*/*.json")):
        value = _read_json(path)
        if value.get("target_id") != target_id:
            continue
        record = value.get("record")
        record = record if isinstance(record, Mapping) else value.get("result")
        if not isinstance(record, Mapping):
            continue
        result_seen = True
        if record.get("target_attempted") is not False:
            return True
    if result_seen:
        return False
    return saved_status


def _framework_attempt_inputs(
    run: dict[str, Any], target_id: str, collector: str,
) -> tuple[int, bool, list[Path], bool, int]:
    run_root = run["run_root"]
    method = str(run["method"])
    method_dir = run_root / "framework" / _safe(method)
    config_dir = run_root / "framework-coverage-configs" / _safe(method)
    profile_roots: list[Path] = []
    attempts = 0
    known = True
    dotnet_session = False
    session_profiled_attempts = 0
    lane_case_dirs: dict[int, dict[int, Path]] = {}
    configs = sorted(config_dir.glob("lane-*-case-*.json")) if config_dir.is_dir() else []
    for config_path in configs:
        match = _CASE_CONFIG.search(config_path.name)
        if not match:
            continue
        lane_id, case_index = map(int, match.groups())
        lane_dir = method_dir / "lanes" / f"lane-{lane_id:02d}"
        if lane_id not in lane_case_dirs:
            lane = _read_json(lane_dir / "lane-result.json")
            cases = lane.get("cases")
            case_dirs = {}
            if isinstance(cases, list):
                for case in cases:
                    if not isinstance(case, dict) or type(case.get("case_index")) is not int:
                        continue
                    value = case.get("case_dir")
                    path = _run_path(value, run_root) if isinstance(value, str) else None
                    if path is not None and path.is_dir():
                        case_dirs[case["case_index"]] = path
            lane_case_dirs[lane_id] = case_dirs
        case_dir = lane_case_dirs[lane_id].get(case_index)
        if case_dir is None:
            expected_case_dir = lane_dir / "cases" / f"case-{case_index:06d}"
            if expected_case_dir.is_dir():
                case_dir = expected_case_dir
            else:
                case_dir = next((
                    path for path in (lane_dir / "cases").glob("case-*")
                    if (found := _CASE_DIR.fullmatch(path.name))
                    and int(found.group(1)) == case_index
                ), None)
        attempted = _campaign_attempted(case_dir, target_id) if case_dir else None
        if attempted is None:
            known = False
        elif attempted:
            attempts += 1

        case_config = _read_json(config_path)
        target_configs = case_config.get("targets")
        target_config = target_configs.get(target_id) \
            if isinstance(target_configs, Mapping) else None
        coverage = target_config.get("coverage_config") \
            if isinstance(target_config, Mapping) else None
        coverage = coverage if isinstance(coverage, Mapping) else target_config
        if not isinstance(coverage, Mapping):
            if attempted:
                known = False
            continue
        dotnet_session |= coverage.get("dotnet_coverage_session_mode") == "server"
        if attempted is True and coverage.get("dotnet_coverage_session_mode") == "server":
            expected_session = coverage.get("coverage_batch_session_id")
            task_records = (
                _read_json(path) for path in case_dir.rglob("dotnet-coverage-task.json")
            ) if case_dir else ()
            if any(
                record.get("status") in {"client-recorded", "recorded"}
                and record.get("client_completed") is True
                and (not expected_session or record.get("session_id") == expected_session)
                for record in task_records
            ):
                session_profiled_attempts += 1
        raw_value = coverage.get("raw_dir")
        if isinstance(raw_value, str):
            raw = _run_path(raw_value, run_root, case_dir)
            if raw is None:
                continue
            if raw.is_dir() and (attempted is True or _has_profile(raw, collector)):
                profile_roots.append(raw)
    return attempts, known, profile_roots, dotnet_session, session_profiled_attempts


def _target_run_inputs(
    run: dict[str, Any], target_id: str, collector: str,
) -> dict[str, Any]:
    run_root = run["run_root"]
    if run.get("error"):
        return {
            "run_id": run["run_id"], "run_root": str(run_root),
            "status": "missing", "reason": run["error"], "attempts": 0,
            "attempts_known": False, "profile_roots": [], "raw_roots": [],
            "profile_data_roots": [], "profile_case_metadata": {},
            "missing_profile_cases": [],
            "profiled_attempts": None, "attempt_profile_tracking": False,
            "profile_present": False, "dotnet_session": False,
            "interrupted": run["interrupted"],
        }
    if run["execution"].get("action") == "framework-run":
        attempts, known, profile_roots, session, session_profiled = _framework_attempt_inputs(
            run, target_id, collector,
        )
        profile_case_metadata = {
            path: {"case_id": path.parent.name}
            for path in profile_roots
            if _CASE_DIR.fullmatch(path.parent.name)
        }
        raw_root = run_root / "framework" / _safe(run["method"]) \
            / "coverage-batches" / _safe(target_id)
        raw_roots = [raw_root] if raw_root.is_dir() else []
        profiled = min(session_profiled, attempts) if session else None
        attempt_profile_tracking = True
        profile_present = _has_profile(raw_root, collector)
    else:
        attempts, known, profile_roots, profile_case_metadata = _events_for_external_run(
            run_root, target_id,
        )
        raw_root = run_root / "coverage-raw" / _safe(target_id)
        raw_roots = [raw_root] if raw_root.is_dir() else []
        session = False
        profiled = None
        attempt_profile_tracking = False
        profile_present = _has_profile(raw_root, collector)
    profile_data_roots = [] if collector == "dotnet" and session else [
        path for path in profile_roots if _has_profile(path, collector)
    ]
    missing_profile_cases = []
    for path in (profile_roots if not (collector == "dotnet" and session) else ()):
        if path in profile_data_roots:
            continue
        case = profile_case_metadata.get(path)
        if not case:
            continue
        flush_status = case.get("coverage_flush_status")
        if isinstance(flush_status, str) and flush_status not in {
            "graceful-signal-exit", "process-exited-before-stop",
        }:
            reason = flush_status
        elif case.get("outcome") in {"normal", "passed"}:
            reason = "raw-profile-missing-after-target-exit"
        else:
            reason = case.get("reason_code") or case.get("termination") \
                or case.get("outcome") or "raw-profile-not-written"
        missing_profile_cases.append({**case, "reason": str(reason)})
    if attempt_profile_tracking and not session:
        profiled = len(profile_data_roots)
    return {
        "run_id": run["run_id"],
        "run_root": str(run_root),
        "status": run["run_result"].get("status")
        or run["wrapper"].get("status") or run.get("launch_status"),
        "reason": run.get("stop_reason"),
        "attempts": attempts,
        "attempts_known": known,
        "profile_roots": profile_roots,
        "profile_data_roots": profile_data_roots,
        "profile_case_metadata": profile_case_metadata,
        "missing_profile_cases": missing_profile_cases,
        "raw_roots": raw_roots,
        "profiled_attempts": profiled,
        "attempt_profile_tracking": attempt_profile_tracking,
        "profile_present": profile_present,
        "dotnet_session": session,
        "interrupted": run["interrupted"],
    }


def _has_profile(root: Path, collector: str) -> bool:
    if not root.is_dir():
        return False
    if collector in {"gcov", "lcov"}:
        paths = root.rglob("*.gcda")
        return any(
            path.is_file() and not path.name.endswith(".tmp.gcda")
            and path.stat().st_size > 0 for path in paths
        )
    if collector == "llvm":
        return any(path.is_file() and path.stat().st_size > 0
                   for path in root.rglob("*.profraw"))
    if collector == "dotnet":
        return any(
            path.is_file() and path.stat().st_size > 0
            for suffix in ("*.coverage", "*.cobertura.xml", "*.cobertura.xml.gz")
            for path in root.rglob(suffix)
        )
    return False


def _raw_files(root: Path, collector: str) -> list[Path]:
    paths = source_coverage.coverage_input_files(root, collector)
    if collector in {"gcov", "lcov"}:
        return [
            path for path in paths
            if path.name == source_coverage.GCOV_FLUSH_INCOMPLETE_MARKER
            or (path.name.endswith(".gcda") and not path.name.endswith(".tmp.gcda"))
        ]
    if collector == "llvm":
        return [path for path in paths if path.name.endswith(".profraw")]
    if collector == "dotnet":
        return [path for path in paths if path.name.endswith(
            (".coverage", ".cobertura.xml", ".cobertura.xml.gz"),
        )]
    return []


def _stage_raw_inputs(
    inputs: list[dict[str, Any]], destination: Path, collector: str,
) -> int:
    count = 0
    for run_index, item in enumerate(inputs):
        for root_index, raw_root in enumerate(item["raw_roots"]):
            if not raw_root.is_dir():
                continue
            for source in _raw_files(raw_root, collector):
                try:
                    resolved = source.resolve(strict=True)
                    relative = source.relative_to(raw_root)
                except (OSError, RuntimeError, ValueError):
                    continue
                # ``_stage_cell_inputs`` may have remapped ``raw_root`` to a
                # verified NVMe copy.  In that case the resolved file is
                # intentionally outside the original run root; requiring the
                # old containment relation here silently drops every staged
                # dotnet/LLVM input.  The root itself comes from the manifest
                # and was already restricted by the cell staging step, so the
                # relative path check is the relevant invariant here.
                target = (
                    destination / "runs" / f"{run_index:03d}-{_safe(item['run_id'])}"
                    / f"root-{root_index:03d}" / relative
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.symlink_to(resolved)
                except OSError:
                    shutil.copy2(resolved, target)
                count += 1
    return count


def _dotnet_expected_inputs(inputs: list[dict[str, Any]]) -> int:
    expected = 0
    for item in inputs:
        if item["attempts"] <= 0:
            continue
        expected += 1 if item["dotnet_session"] else item["attempts"]
    return expected


def _source_metrics_gap(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA,
        "metric_family": "SimSrcCov",
        "status": "gap",
        "reason": reason,
        "identity": {},
        "metrics": {
            name: {"covered": None, "eligible": None, "value": None,
                   "status": "gap", "reason": reason}
            for name in ("function", "line", "branch")
        },
    }


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _merge_gcov_profiles(
    inputs: list[dict[str, Any]], raw_root: Path,
) -> tuple[Path, int, int, list[dict[str, Any]], int]:
    """先合并原始 GCDA，再让 gcov 只扫描一次合并结果。"""
    roots: set[Path] = set()
    profile_case_metadata: dict[Path, dict[str, Any]] = {}
    for item in inputs:
        data_roots = item.get("profile_data_roots")
        if data_roots is None:
            data_roots = [
                path for path in item["profile_roots"]
                if path.is_dir() and _has_profile(path, "gcov")
            ]
        roots.update(data_roots)
        profile_case_metadata.update(item.get("profile_case_metadata") or {})
    roots = sorted(roots)
    merged = raw_root / "merged-gcov"
    if not roots:
        merged.mkdir(parents=True, exist_ok=True)
        return merged, 0, 0, [], 0
    # ``gcov-tool merge`` creates its output directory but does not reliably
    # create the parent when several merges start at once.  Pre-create the
    # shared parent before launching pairwise workers; otherwise only the
    # first worker succeeds and the rest report ``Cannot make directory``.
    merged.mkdir(parents=True, exist_ok=True)

    input_files = 0
    flush_markers = []
    for root in roots:
        for path in root.rglob("*.gcda"):
            if not path.name.endswith(".tmp.gcda"):
                input_files += 1
        flush_markers.extend(root.rglob(source_coverage.GCOV_FLUSH_INCOMPLETE_MARKER))

    # Merge in a balanced tree.  The former left fold rewrote the complete
    # accumulated profile for every case (O(n²) I/O for long runs); pairwise
    # rounds keep the same gcov-tool semantics while reducing the work to
    # O(n log n).  Original profile roots are never modified.
    failures: list[dict[str, Any]] = []
    queue: list[Path] = list(roots)
    round_index = 0
    accepted = 0
    try:
        pair_jobs = max(1, min(4, int(os.environ.get("RQ1_GCOV_PAIR_JOBS", "2"))))
    except (TypeError, ValueError):
        pair_jobs = 2

    def merge_pair(
        left: Path, right: Path, destination: Path,
    ) -> tuple[Path, Path, Path, int, str]:
        if destination.exists():
            shutil.rmtree(destination)
        try:
            result = subprocess.run(
                ["gcov-tool", "merge", "--output", str(destination),
                 str(left), str(right)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, check=False,
            )
            code = result.returncode
            error = (result.stderr or "")[-500:]
        except OSError as exc:
            code, error = 127, str(exc)
        return left, right, destination, code, error

    while len(queue) > 1:
        next_queue: list[Path] = []
        pair_tasks: list[tuple[Path, Path, Path]] = []
        for pair_index in range(0, len(queue), 2):
            left = queue[pair_index]
            if pair_index + 1 >= len(queue):
                next_queue.append(left)
                continue
            right = queue[pair_index + 1]
            destination = merged / f"round-{round_index:04d}-{pair_index // 2:06d}"
            pair_tasks.append((left, right, destination))
        merged_pairs = []
        with ThreadPoolExecutor(max_workers=min(pair_jobs, len(pair_tasks) or 1)) as pool:
            futures = [pool.submit(merge_pair, *task) for task in pair_tasks]
            for future in futures:
                merged_pairs.append(future.result())
        for left, right, destination, code, error in merged_pairs:
            if code == 0 and destination.is_dir():
                next_queue.append(destination)
            else:
                if destination.is_dir():
                    shutil.rmtree(destination)
                failures.append({
                    "profile_root": str(right), "returncode": code,
                    "reason": error or "gcov-tool-merge-failed",
                    "case": profile_case_metadata.get(right),
                })
                # Retain the left side as evidence and leave the failed right
                # side explicitly recorded as a merge failure.
                next_queue.append(left)
        queue = next_queue
        round_index += 1
    accepted = max(0, len(roots) - len(failures))
    if queue:
        active = queue[0]
        if active.parent != merged:
            final = merged / "accum-final"
            if final.exists():
                shutil.rmtree(final)
            shutil.copytree(active, final, symlinks=True, copy_function=_link_or_copy)
            active = final
    else:
        active = merged / "accum-final"
        active.mkdir(parents=True, exist_ok=True)

    for index, marker in enumerate(flush_markers):
        destination = active / "flush-markers" / f"{index:08d}" / marker.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker, destination)
    return active, accepted, input_files, failures, len(flush_markers)


@contextmanager
def _stage_cell_inputs(
    inputs: list[dict[str, Any]], collector: str,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Move one cell's stable raw inputs to NVMe for the expensive read path."""
    if collector in {"gcov", "lcov"}:
        root_key = "profile_data_roots"
        fallback_key = "profile_roots"
    else:
        root_key = "raw_roots"
        fallback_key = "raw_roots"
    roots: list[Path] = []
    for item in inputs:
        values = item.get(root_key)
        if values is None:
            values = item.get(fallback_key, ())
        roots.extend(Path(value) for value in values or ())

    predicate = ssd_trace_stage.coverage_file_predicate(collector)
    with ssd_trace_stage.stage_roots(
        roots, predicate=predicate, prefix="rq1-ssd-profile-",
    ) as (mapping, stage):
        staged_inputs: list[dict[str, Any]] = []
        for item in inputs:
            clone = dict(item)
            values = item.get(root_key)
            if values is None:
                values = item.get(fallback_key, ())
            clone[root_key] = [
                mapping.get(Path(value).resolve(), Path(value))
                for value in values or ()
            ]
            # Growth-curve workers use per-case profile roots even when the
            # collector itself stages one raw root (LLVM/Dotnet). Remap those
            # nested roots into the verified NVMe copy as well; otherwise the
            # worker would silently read the original HDD paths.
            for nested_key in ("profile_roots", "profile_data_roots"):
                nested_values = item.get(nested_key)
                if not nested_values:
                    continue
                remapped = []
                for value in nested_values:
                    original = Path(value).resolve()
                    # Per-case roots normally equal one of the staged roots.
                    # Resolve that common case in O(1); scanning every root
                    # for every case made large RVVM cells quadratic.
                    mapped_value = mapping.get(original)
                    if mapped_value is None:
                        for source_root, staged_root in mapping.items():
                            try:
                                mapped_value = staged_root / original.relative_to(source_root)
                            except ValueError:
                                continue
                            break
                    remapped.append(mapped_value or Path(value))
                clone[nested_key] = remapped
            metadata = item.get("profile_case_metadata")
            if isinstance(metadata, Mapping) and mapping:
                remapped_metadata = {}
                for value, details in metadata.items():
                    original = Path(value).resolve()
                    mapped_value = mapping.get(original)
                    if mapped_value is None:
                        for source_root, staged_root in mapping.items():
                            try:
                                mapped_value = staged_root / original.relative_to(source_root)
                            except ValueError:
                                continue
                            break
                    remapped_metadata[mapped_value or Path(value)] = details
                clone["profile_case_metadata"] = remapped_metadata
            if root_key != "raw_roots":
                # The raw roots are retained as metadata and are not read by
                # the gcov merge.  Keep them stable so the result still points
                # to the original experiment evidence.
                clone["raw_roots"] = list(item.get("raw_roots", ()))
            staged_inputs.append(clone)
        # Keep the metadata object live so callers that retain it can observe
        # the final cleanup state written by ``stage_roots``.
        yield staged_inputs, stage


def _gcov_merge_failure_label(profile_root: Path) -> str:
    case_name = profile_root.parent.name
    return case_name if re.fullmatch(r"case-\d+", case_name) else profile_root.name


def _catalog_cache_dirs(run_root: Path, method: str, target_id: str) -> list[Path]:
    relative = Path("coverage-progress") / "index" / _safe(method) \
        / "opcode-catalog" / _safe(target_id)
    paths = [run_root / relative]
    paths.extend(path / relative for path in run_root.glob("coverage-reanalysis-*"))
    return sorted(
        (path for path in paths if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _catalog_cache_matches(
    run_root: Path, method: str, target_id: str,
    event: Mapping[str, Any], candidate: int, cached: Mapping[str, Any],
) -> bool:
    identity = cached.get("identity")
    entry = cached.get("rv_opcode_catalog_coverage")
    if not isinstance(identity, Mapping) or not isinstance(entry, Mapping):
        return False
    if (identity.get("catalog_id") != rv_catalog.OPCODE_CATALOG_ID
            or identity.get("registry_sha256") != rv_catalog.OPCODE_CATALOG_REGISTRY_SHA256):
        return False
    case_root = run_root / "executions" / _safe(target_id) / _safe(method) \
        / f"candidate-{candidate}"
    target_run = _read_json(case_root / "target-run.json")
    target_row = next((row for row in target_run.get("targets", ())
                       if isinstance(row, Mapping) and row.get("id") == target_id), {})
    event_artifact = cp._record_artifact_digest(dict(event))
    target_artifact = cp._record_artifact_digest(dict(target_row))
    expected_artifact = event_artifact or target_artifact
    if (not isinstance(expected_artifact, str)
            or identity.get("artifact_id") != expected_artifact.lower()
            or target_artifact not in (None, identity.get("target_artifact_id"))):
        return False
    trace = target_row.get("trace") if isinstance(target_row, Mapping) else None
    trace = trace if isinstance(trace, Mapping) else {}
    trace_value = (
        target_row.get("trace_path") or trace.get("path")
        or event.get("trace_path")
    )
    trace_path = _run_path(trace_value, run_root, case_root)
    cached_trace = _run_path(identity.get("trace_path"), run_root, case_root)
    try:
        stat = trace_path.stat() if trace_path is not None else None
        same_path = (
            trace_path is not None and cached_trace is not None
            and trace_path.resolve() == cached_trace.resolve()
        )
    except OSError:
        return False
    return bool(
        stat and same_path
        and identity.get("trace_size_bytes") == stat.st_size
        and identity.get("trace_mtime_ns") == stat.st_mtime_ns
        and isinstance(identity.get("trace_sha256"), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", identity["trace_sha256"])
    )


def _rv_catalog_gap(reason: str, expected_cases: int = 0) -> dict[str, Any]:
    metrics = {
        name: {
            "covered": None, "eligible": rv_catalog.OPCODE_CATALOG_DENOMINATOR,
            "value": None, "status": "gap", "reason": reason,
        }
        for name in ("GenOpcodeCov", "ExecOpcodeCov")
    }
    return {
        "schema_version": rv_catalog.OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "offline-method-target-union",
        "catalog_id": rv_catalog.OPCODE_CATALOG_ID,
        "key_schema": rv_catalog.OPCODE_KEY_SCHEMA,
        "registry_sha256": rv_catalog.OPCODE_CATALOG_REGISTRY_SHA256,
        "denominator": rv_catalog.OPCODE_CATALOG_DENOMINATOR,
        "expected_case_count": expected_cases,
        "represented_case_count": 0,
        "missing_case_count": expected_cases,
        "status": "gap",
        "reason": reason,
        "metrics": metrics,
        "run_inputs": [],
    }


def _merge_rv_catalog_entries(
    entries: list[Mapping[str, Any]], expected: int, represented: int,
) -> dict[str, Any]:
    missing = max(0, expected - represented)
    metrics = {}
    for name in ("GenOpcodeCov", "ExecOpcodeCov"):
        metric = rv_catalog._merge_catalog_metric(
            entries, name,
            force_gap=not entries,
            reason="opcode-catalog-evidence-missing" if not entries else None,
        )
        if missing and metric.get("status") in {"observed", "partial"}:
            metric.update({
                "status": "partial", "value": None,
                "reason": "opcode-catalog-case-evidence-incomplete",
            })
        metrics[name] = metric
    statuses = {metric.get("status") for metric in metrics.values()}
    return {
        "schema_version": rv_catalog.OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "offline-method-target-union",
        "catalog_id": rv_catalog.OPCODE_CATALOG_ID,
        "key_schema": rv_catalog.OPCODE_KEY_SCHEMA,
        "registry_sha256": rv_catalog.OPCODE_CATALOG_REGISTRY_SHA256,
        "denominator": rv_catalog.OPCODE_CATALOG_DENOMINATOR,
        "aggregation": "unique-opcode-set-union-across-runs",
        "expected_case_count": expected,
        "represented_case_count": represented,
        "missing_case_count": missing,
        "status": "gap" if "gap" in statuses else
                  "partial" if "partial" in statuses else "observed",
        "metrics": metrics,
    }


def _framework_catalog_entries(
    run: dict[str, Any], method: str, target_id: str,
    evidence_dir: Path | None = None,
) -> tuple[list[Mapping[str, Any]], int]:
    # Raw-only framework runs intentionally do not build an online catalog
    # summary.  A sidecar produced from the retained replay queue is an
    # explicit offline input, rather than a guessed case count.
    if evidence_dir is not None:
        sidecar = evidence_dir / f"{_safe(method)}-{_safe(target_id)}.json"
        document = _read_json(sidecar)
        if document:
            if document.get("method") not in (None, method) \
                    or document.get("target") not in (None, target_id):
                return [], 0
            rows = document.get("entries")
            entries = [
                row.get("rv_opcode_catalog_coverage")
                for row in rows
                if isinstance(row, Mapping)
                and isinstance(row.get("rv_opcode_catalog_coverage"), Mapping)
            ] if isinstance(rows, list) else []
            count = document.get("attempted_cases")
            if type(count) is not int or count < len(entries):
                count = len(entries)
            return entries, count
    # New raw-only framework runs keep the authoritative RV metrics inside
    # resident Target result sidecars. Consume them directly when no derived
    # summary/explicit evidence directory is available.
    raw_entries, raw_count = cp._framework_raw_catalog_entries(
        run["run_root"], method, target_id,
    )
    if raw_count:
        return raw_entries, raw_count
    method_root = run["run_root"] / "framework" / _safe(method)
    paths = [
        method_root / "targets" / _safe(target_id) / "coverage" / "summary.json",
        method_root / "coverage" / "summary.json",
    ]
    for path in paths:
        catalog = _read_json(path).get("rv_opcode_catalog_coverage")
        target_runs = catalog.get("target_runs") if isinstance(catalog, Mapping) else None
        entries = []
        if isinstance(target_runs, list) and isinstance(catalog, Mapping):
            identity = {
                key: catalog.get(key)
                for key in ("schema_version", "catalog_id", "key_schema", "registry_sha256")
            }
            entries = [
                {**identity, **row} for row in target_runs
                if isinstance(row, Mapping)
                and row.get("method") == method and row.get("target") == target_id
            ]
        if entries:
            return entries, sum(
                int(row.get("case_count", 0) or 0) for row in entries
                if type(row.get("case_count")) is int
            )
    return [], 0


def _external_catalog_entries(
    run: dict[str, Any], method: str, target_id: str,
) -> tuple[list[Mapping[str, Any]], int, dict[str, Any]]:
    run_root = run["run_root"]
    events, ledger_known = _ledger_events(str(run_root.resolve()))
    attempted = [event for event in events if (
        event.get("target") == target_id and event.get("target_attempted") is True
    )]
    needs_cache = any(
        not isinstance(
            (event.get("simulator_coverage") or {}).get("rv_opcode_catalog_coverage")
            if isinstance(event.get("simulator_coverage"), Mapping) else None,
            Mapping,
        )
        for event in attempted
    )
    cache_dirs = _catalog_cache_dirs(run_root, method, target_id) if needs_cache else []
    case_entries: dict[int, Mapping[str, Any]] = {}
    unresolved: list[tuple[int, dict[str, Any]]] = []
    attempted_candidates: set[int] = set()
    for event in attempted:
        candidate = event.get("stream_index")
        if type(candidate) is not int or candidate < 0:
            match = re.fullmatch(r"candidate-(\d+)", str(event.get("case_id") or ""))
            candidate = int(match.group(1)) if match else None
        if candidate is None:
            continue
        attempted_candidates.add(candidate)
        name = f"candidate-{candidate:08d}.json"
        cached_entry = None
        coverage = event.get("simulator_coverage")
        value = coverage.get("rv_opcode_catalog_coverage") \
            if isinstance(coverage, Mapping) else None
        if isinstance(value, Mapping):
            cached_entry = value
        else:
            for directory in cache_dirs:
                path = directory / name
                if not path.is_file():
                    continue
                cached = _read_json(path)
                if _catalog_cache_matches(
                    run_root, method, target_id, event, candidate, cached,
                ):
                    value = cached.get("rv_opcode_catalog_coverage")
                    if isinstance(value, Mapping):
                        cached_entry = value
                        break
        if cached_entry is not None:
            case_entries[candidate] = cached_entry
        else:
            # Progress snapshots are disposable derived state. If their case
            # cache is stale or absent, rebuild the catalog entry from the
            # retained target trace instead of silently treating that attempt
            # as missing. Keep the original run ledger untouched.
            parse_event = dict(event)
            parse_event.pop("simulator_coverage", None)
            unresolved.append((candidate, parse_event))

    materialization: dict[str, Any] = {
        "cases": 0, "parsed_cases": 0, "missing_cases": 0,
        "identity_mismatch_cases": 0, "error_cases": 0,
    }
    if unresolved:
        configured_targets = run.get("config", {}).get("targets", [])
        target = next((value for value in configured_targets
                       if isinstance(value, Mapping)
                       and value.get("id") == target_id), None)
        if isinstance(target, Mapping):
            parse_events = [event for _, event in unresolved]

            def save_materialized_case(
                index: int, event: dict[str, Any], state: str,
            ) -> None:
                if state != "parsed":
                    return
                candidate = unresolved[index][0]
                artifact_id = next((event.get(key) for key in (
                    "artifact_id", "linux_elf_sha256", "bare_elf_sha256",
                    "rvvm_elf_sha256",
                ) if isinstance(event.get(key), str)), None)
                cp._save_external_catalog_entry(
                    run_root=run_root, event=event,
                    marker=(run_root / "coverage-progress" / "index"
                            / _safe(method) / "opcode-catalog"
                            / _safe(target_id)
                            / f"candidate-{candidate:08d}.json"),
                    artifact_id=artifact_id,
                    case_root=(run_root / "executions" / _safe(target_id)
                               / _safe(method) / f"candidate-{candidate}"),
                    old_identity=None,
                )

            try:
                materialization = cp._materialize_guest_coverage(
                    run_root, [dict(target)], parse_events,
                    max_workers=1, on_case=save_materialized_case,
                )
            except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError):
                materialization = {
                    "cases": len(parse_events), "parsed_cases": 0,
                    "missing_cases": 0, "identity_mismatch_cases": 0,
                    "error_cases": len(parse_events),
                }
            for candidate, event in unresolved:
                coverage = event.get("simulator_coverage")
                value = coverage.get("rv_opcode_catalog_coverage") \
                    if isinstance(coverage, Mapping) else None
                if isinstance(value, Mapping):
                    case_entries[candidate] = value

    entries = [case_entries[index] for index in sorted(case_entries)]
    represented = set(case_entries)
    missing_candidates = sorted(attempted_candidates - represented)
    metric_status_counts: dict[str, dict[str, int]] = {}
    metric_reason_counts: dict[str, dict[str, int]] = {}
    metric_gap_case_examples: dict[str, list[int]] = {}
    for name in ("GenOpcodeCov", "ExecOpcodeCov"):
        statuses: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        bad_cases: list[int] = []
        for candidate, entry in sorted(case_entries.items()):
            metrics = entry.get("metrics")
            metric = metrics.get(name) if isinstance(metrics, Mapping) else None
            status = str(metric.get("status") or "gap") \
                if isinstance(metric, Mapping) else "missing"
            statuses[status] += 1
            if status in {"gap", "partial", "missing"}:
                bad_cases.append(candidate)
                reason = metric.get("reason") if isinstance(metric, Mapping) else None
                if isinstance(reason, str) and reason:
                    reasons[reason] += 1
                elif status == "missing":
                    reasons["opcode-catalog-metric-missing"] += 1
        if missing_candidates:
            statuses["missing"] += len(missing_candidates)
            reasons["catalog-cache-incomplete"] += len(missing_candidates)
            bad_cases.extend(missing_candidates)
        metric_status_counts[name] = dict(sorted(statuses.items()))
        metric_reason_counts[name] = dict(sorted(reasons.items()))
        metric_gap_case_examples[name] = bad_cases[:20]

    return entries, len(represented), {
        "run_id": run["run_id"],
        "attempted_cases": len(attempted),
        "attempts_known": ledger_known,
        "represented_cases": len(represented),
        "missing_cases": max(0, len(attempted) - len(represented)),
        "missing_case_examples": missing_candidates[:32],
        "metric_status_counts": metric_status_counts,
        "metric_reason_counts": metric_reason_counts,
        "metric_gap_case_examples": metric_gap_case_examples,
        "materialization": materialization,
        "interrupted": run["interrupted"],
        "reason": "catalog-cache-incomplete" if len(represented) < len(attempted) else None,
    }


def _collect_rv_catalog_cell(
    method: str, target_id: str, target: Mapping[str, Any],
    runs: list[dict[str, Any]], run_inputs: list[dict[str, Any]],
    catalog_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    if target.get("_identity_mismatch"):
        return _rv_catalog_gap("target-coverage-identity-mismatch")
    entries: list[Mapping[str, Any]] = []
    run_rows = []
    expected = 0
    represented = 0
    for run, run_input in zip(runs, run_inputs):
        attempts = int(run_input.get("attempts", 0) or 0)
        if run["execution"].get("action") == "framework-run":
            selected, count = _framework_catalog_entries(
                run, method, target_id, catalog_evidence_dir,
            )
            expected_run = count if selected else attempts
            # A raw-only sidecar contains one entry per replay case.  Online
            # summaries, in contrast, may contain an aggregated target row
            # whose ``case_count`` is larger than the number of rows.
            represented_count = (
                len(selected) if catalog_evidence_dir is not None else count
            ) if selected else 0
            expected += expected_run
            run_rows.append({
                "run_id": run["run_id"], "attempted_cases": attempts,
                "expected_cases": expected_run,
                "represented_cases": represented_count,
                "missing_cases": max(0, expected_run - represented_count),
                "interrupted": run["interrupted"],
                "reason": "framework-catalog-summary-missing" if not selected and attempts else None,
            })
        else:
            expected += attempts
            selected, represented_count, run_row = _external_catalog_entries(
                run, method, target_id,
            )
            run_rows.append(run_row)
        entries.extend(selected)
        represented += represented_count

    result = _merge_rv_catalog_entries(entries, expected, represented)
    result["run_inputs"] = run_rows
    return result


def _collect_cell(
    output: Path, method: str, target_id: str, target: dict[str, Any],
    runs: list[dict[str, Any]], catalog_evidence_dir: Path | None = None,
    *, include_guest_metrics: bool = True,
) -> dict[str, Any]:
    signature = _identity_signature(target)
    if target.get("_identity_mismatch"):
        source = _source_metrics_gap("target-coverage-identity-mismatch")
        return {
            "method": method, "target": target_id, "status": "gap",
            "reason": "target-coverage-identity-mismatch", "run_ids": [
                row["run_id"] for row in runs
            ], "source_coverage": source,
            "rv_opcode_catalog_coverage": _rv_catalog_gap(
                "target-coverage-identity-mismatch",
            ),
        }
    source_spec = target.get("source_coverage")
    collector = str(source_spec.get("collector") or "") \
        if isinstance(source_spec, Mapping) else ""
    if not collector:
        source = _source_metrics_gap("source-coverage-not-configured")
        return {
            "method": method, "target": target_id, "status": "gap",
            "reason": "source-coverage-not-configured",
            "run_ids": [row["run_id"] for row in runs],
            "source_coverage": source,
            "rv_opcode_catalog_coverage": _rv_catalog_gap(
                "source-coverage-not-configured",
            ),
        }

    run_inputs = [_target_run_inputs(row, target_id, collector) for row in runs]
    attempted = sum(item["attempts"] for item in run_inputs)
    rv_result = (
        _collect_rv_catalog_cell(
            method, target_id, target, runs, run_inputs, catalog_evidence_dir,
        ) if include_guest_metrics else
        _rv_catalog_gap("guest-metrics-disabled", expected_cases=attempted)
    )
    expected_inputs = (
        _dotnet_expected_inputs(run_inputs) if collector == "dotnet"
        else attempted if collector == "llvm" else None
    )
    attempt_profile_tracking = all(
        item["attempt_profile_tracking"] for item in run_inputs
    )
    profile_attempts = (
        sum(item["profiled_attempts"] or 0 for item in run_inputs)
        if attempt_profile_tracking else None
    )
    profile_input_count = 0
    reasons = []
    for item in run_inputs:
        if item["reason"]:
            reasons.append(f"{item['run_id']}:{item['reason']}")
        missing_profile_count = len(item.get("missing_profile_cases", ()))
        if missing_profile_count:
            reasons.append(
                f"{item['run_id']}:source-profile-missing:{missing_profile_count}"
            )
        if not item["attempts_known"]:
            reasons.append(f"{item['run_id']}:attempt-count-unknown")
        if item["interrupted"]:
            reasons.append(f"{item['run_id']}:run-interrupted")
        if item["attempts"] > 0 and not item["profile_present"]:
            reasons.append(f"{item['run_id']}:source-profile-inputs-missing")
        if item["attempt_profile_tracking"] and item["attempts"] > (
            item["profiled_attempts"] or 0
        ):
            if collector == "dotnet" and item["dotnet_session"]:
                reasons.append(f"{item['run_id']}:dotnet-session-client-count-mismatch")
            elif item["profile_present"]:
                reasons.append(f"{item['run_id']}:attempt-profile-missing")

    profile_dir = output / "profiles" / _safe(method) / _safe(target_id)
    spec = dict(source_spec) if isinstance(source_spec, Mapping) else {}
    spec["profile"] = str(profile_dir / "lcov.info")
    spec["identity"] = str(profile_dir / "identity.json")
    target_for_collection = {**target, "source_coverage": spec}
    binary = _dependency_path(target.get("coverage_binary"))

    stage_meta: dict[str, Any] = {}
    with _stage_cell_inputs(run_inputs, collector) as (staged_inputs, stage_info):
        stage_meta = dict(stage_info)
        temporary_parent = Path(stage_info["stage_dir"]) \
            if stage_info.get("stage_dir") else output
        with tempfile.TemporaryDirectory(
            prefix=f"raw-{_safe(method)}-{_safe(target_id)}-",
            dir=temporary_parent,
        ) as temporary:
            raw_root = Path(temporary)
            if collector in {"gcov", "lcov"}:
                merge_started = time.monotonic()
                collector_root, merged_profiles, profile_input_count, merge_failures, flush_count = (
                    _merge_gcov_profiles(staged_inputs, raw_root)
                )
                stage_meta["merge_seconds"] = round(
                    time.monotonic() - merge_started, 6
                )
                observed_inputs = merged_profiles
                for failure in merge_failures:
                    case = failure.get("case")
                    case_id = case.get("case_id") if isinstance(case, Mapping) else None
                    label = case_id or _gcov_merge_failure_label(
                        Path(failure["profile_root"]),
                    )
                    reasons.append(
                        f"gcov-tool-merge-failed:{label}:{failure['returncode']}"
                    )
            else:
                collector_root = raw_root
                profile_input_count = _stage_raw_inputs(
                    staged_inputs, raw_root, collector,
                )
                merged_profiles, merge_failures, flush_count = 0, [], 0
            if collector == "dotnet":
                observed_inputs = sum(
                    path.name.endswith((".cobertura.xml", ".cobertura.xml.gz"))
                    for path in _raw_files(collector_root, collector)
                )
                if observed_inputs != expected_inputs:
                    reasons.append("cobertura-input-count-mismatch")
            elif collector == "llvm":
                observed_inputs = sum(path.name.endswith(".profraw")
                                      for path in _raw_files(collector_root, collector))
                if expected_inputs is not None and observed_inputs < expected_inputs:
                    reasons.append("llvm-profraw-count-mismatch")
            try:
                collection = source_coverage.collect_source_coverage_batch(
                    output, target_for_collection, binary, collector_root,
                    expected_coverage_cases=expected_inputs,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                collection = {
                    "status": "unavailable",
                    "reason": f"offline-source-collection:{type(error).__name__}",
                }
            if profile_input_count == 0 and attempted and not any(
                reason.endswith("source-profile-inputs-missing") for reason in reasons
            ):
                reasons.append("source-profile-inputs-missing")
            if collection.get("status") == "partial" and collection.get("reason"):
                reasons.append(str(collection["reason"]))
            if reasons and collection.get("status") in {"observed", "partial"}:
                reason = ";".join(sorted(set(reasons)))
                collection = {
                    **collection, "status": "partial", "reason": reason,
                }

            source = target_source_coverage(
                output, target_for_collection,
                coverage_binary_sha=collection.get("coverage_binary_sha"),
                source_commit_observed=collection.get("source_commit_observed"),
                coverage_binary_sha256=collection.get("coverage_binary_sha256"),
                collection_status=collection.get("status"),
                collection_reason=collection.get("reason"),
            )
            collection = dict(collection)
            collection.pop("raw_dir", None)
            collection["raw_sources"] = [
                str(root) for item in run_inputs for root in item["raw_roots"]
            ]
            collection["profile_input_files"] = profile_input_count
            collection["expected_collection_inputs"] = expected_inputs
            collection["observed_collection_inputs"] = observed_inputs
            if collector in {"gcov", "lcov"}:
                collection["gcov_merge_tool"] = "gcov-tool"
                collection["gcov_merge_profiles"] = merged_profiles
                collection["gcov_merge_failures"] = merge_failures
                collection["gcov_incomplete_flush_profiles"] = flush_count
            if reasons:
                source["collection_warnings"] = sorted(set(reasons))
    stage_meta["cleanup"] = "complete"
    collection["ssd_stage"] = stage_meta

    status = str(source.get("status") or "gap")
    return {
        "method": method,
        "target": target_id,
        "status": status,
        "collector": collector,
        "identity_sha256": signature,
        "run_ids": [row["run_id"] for row in runs],
        "attempted_cases": attempted,
        "attempt_profile_tracking": attempt_profile_tracking,
        "profiled_attempt_cases": profile_attempts,
        "missing_profile_attempts": (
            max(0, attempted - profile_attempts)
            if profile_attempts is not None else None
        ),
        "run_inputs": [
            {
                "run_id": item["run_id"],
                "status": item["status"],
                "attempted_cases": item["attempts"],
                "attempts_known": item["attempts_known"],
                "profile_present": item["profile_present"],
                "profile_root_count": len(item["profile_roots"]),
                "attempt_profile_tracking": item["attempt_profile_tracking"],
                "profiled_attempt_cases": item["profiled_attempts"],
                "interrupted": item["interrupted"],
                "reason": item["reason"],
                "missing_profile_cases": item.get("missing_profile_cases", []),
            }
            for item in run_inputs
        ],
        "collection": collection,
        "source_coverage": source,
        "rv_opcode_catalog_coverage": rv_result,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _elapsed_seconds(started_at: object, finished_at: object) -> int | None:
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None or finished.tzinfo is None:
        return None
    return max(0, int((finished - started).total_seconds()))


def _lcov_units_for_cell(
    output: Path, cell: Mapping[str, Any], target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source = cell.get("source_coverage")
    source = source if isinstance(source, Mapping) else {}
    reported = source.get("metrics")
    reported = reported if isinstance(reported, Mapping) else {}
    result: dict[str, dict[str, Any]] = {}
    if source.get("status") not in {"observed", "partial"}:
        for name in _SOURCE_METRICS:
            metric = reported.get(name)
            status = metric.get("status") if isinstance(metric, Mapping) else None
            result[name] = {
                "status": "NA" if status == "NA" else "gap",
                "reason": source.get("reason") or "source-coverage-unavailable",
            }
        return result

    spec = target.get("source_coverage")
    spec = spec if isinstance(spec, Mapping) else {}
    scope = spec.get("source_scope")
    if not isinstance(scope, list) or not scope:
        return {
            name: {"status": "gap", "reason": "source-scope-not-declared"}
            for name in _SOURCE_METRICS
        }
    profile_value = source.get("profile_path")
    if not isinstance(profile_value, str) or not profile_value:
        return {
            name: {"status": "gap", "reason": "source-coverage-profile-missing"}
            for name in _SOURCE_METRICS
        }
    profile = Path(profile_value)
    if not profile.is_absolute():
        profile = output / profile
    try:
        parsed = simulator_coverage.summarize_lcov(
            profile, [str(item) for item in scope],
            str(spec["source_root"]) if spec.get("source_root") else None,
            [str(item) for item in spec.get("source_scope_exclude", ())],
            include_units=True,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return {
            name: {"status": "gap", "reason": f"lcov-parse:{type(error).__name__}"}
            for name in _SOURCE_METRICS
        }

    parsed_metrics = parsed.get("metrics")
    parsed_metrics = parsed_metrics if isinstance(parsed_metrics, Mapping) else {}
    unit_sets = parsed.get("unit_sets")
    unit_sets = unit_sets if isinstance(unit_sets, Mapping) else {}
    for name in _SOURCE_METRICS:
        metric = parsed_metrics.get(name)
        metric = metric if isinstance(metric, Mapping) else {}
        original = reported.get(name)
        original = original if isinstance(original, Mapping) else {}
        status = metric.get("status")
        sets = unit_sets.get(name)
        sets = sets if isinstance(sets, Mapping) else {}
        try:
            eligible = {
                (str(unit[0]), str(unit[1])) for unit in sets.get("eligible", ())
                if isinstance(unit, (list, tuple)) and len(unit) == 2
            }
            covered = {
                (str(unit[0]), str(unit[1])) for unit in sets.get("covered", ())
                if isinstance(unit, (list, tuple)) and len(unit) == 2
            }
            valid_pairs = (
                len(eligible) == len(sets.get("eligible", ()))
                and len(covered) == len(sets.get("covered", ()))
            )
        except TypeError:
            valid_pairs = False
            eligible, covered = set(), set()
        if status == "NA" and not eligible \
                and original.get("status") in {None, "NA"}:
            result[name] = {"status": "NA", "reason": "coverage-dimension-unavailable"}
        elif original.get("status") == "NA":
            result[name] = {"status": "gap", "reason": "source-metric-unit-mismatch"}
        elif status not in {"observed", "partial"} or original.get("status") == "gap":
            result[name] = {
                "status": "gap",
                "reason": metric.get("reason") or original.get("reason")
                or "source-coverage-profile-unavailable",
            }
        elif not valid_pairs or not covered <= eligible \
                or not eligible \
                or metric.get("eligible") != len(eligible) \
                or metric.get("covered") != len(covered) \
                or original.get("eligible") != metric.get("eligible") \
                or original.get("covered") != metric.get("covered"):
            result[name] = {"status": "gap", "reason": "lcov-unit-count-mismatch"}
        else:
            result[name] = {
                "status": "partial"
                if status == "partial" or original.get("status") == "partial"
                else "observed",
                "eligible": eligible,
                "covered": covered,
            }
    return result


def _unique_metric_gap(reason: str, missing: list[str]) -> dict[str, Any]:
    return {
        "status": "gap", "unique_covered": None, "eligible": None,
        "value": None, "method_covered": None,
        "unique_share_of_method_covered": None,
        "available_comparators": [], "missing_comparators": missing,
        "reason": reason,
    }


def _unique_unit_digest(units: set[tuple[str, str]]) -> str:
    encoded = "\n".join(
        json.dumps(unit, ensure_ascii=False, separators=(",", ":"))
        for unit in sorted(units)
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique_coverage_for_target(
    output: Path, target_id: str, methods: list[str],
    cells: Mapping[tuple[str, str], dict[str, Any]],
    target_specs: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    specs = {method: target_specs.get((method, target_id)) for method in methods}
    configured_specs = [
        spec for spec in specs.values()
        if isinstance(spec, Mapping)
        and isinstance(spec.get("source_coverage"), Mapping)
        and spec["source_coverage"].get("collector")
    ]
    identity_mismatch = any(
        spec.get("_identity_mismatch") is True for spec in configured_specs
    ) or len({_identity_signature(spec) for spec in configured_specs}) > 1
    parsed = {} if identity_mismatch else {
        method: _lcov_units_for_cell(
            output, cells[(method, target_id)], specs[method]
            if isinstance(specs[method], Mapping) else {},
        )
        for method in methods if (method, target_id) in cells
    }
    results = {}
    peers_by_method: dict[str, list[str]] = {}
    for method in methods:
        peers = [other for other in methods if other != method]
        peers_by_method[method] = peers
        missing_profile_methods = [
            peer for peer in peers
            if peer not in parsed or all(
                parsed[peer][name]["status"] not in {"observed", "partial"}
                for name in _SOURCE_METRICS
            )
        ]
        metric_results = {}
        artifact_metrics = {}
        for name in _SOURCE_METRICS:
            owner = parsed.get(method, {}).get(name, {"status": "gap"})
            peer_rows = {
                peer: parsed.get(peer, {}).get(name, {"status": "gap"})
                for peer in peers
            }
            usable_peers = {
                peer: row for peer, row in peer_rows.items()
                if row.get("status") in {"observed", "partial"}
            }
            missing = [peer for peer in peers if peer not in usable_peers]
            units: list[tuple[str, str]] = []
            if identity_mismatch:
                metric = _unique_metric_gap("target-coverage-identity-mismatch", peers)
            elif owner.get("status") == "NA":
                metric = {
                    "status": "NA", "unique_covered": None, "eligible": 0,
                    "value": None, "method_covered": None,
                    "unique_share_of_method_covered": None,
                    "available_comparators": sorted(usable_peers),
                    "missing_comparators": missing,
                    "reason": "coverage-dimension-unavailable",
                }
            elif owner.get("status") not in {"observed", "partial"}:
                metric = _unique_metric_gap(
                    str(owner.get("reason") or "method-coverage-unavailable"), missing,
                )
            elif not usable_peers:
                metric = _unique_metric_gap("no-comparator-profiles", peers)
            else:
                participating = [owner, *usable_peers.values()]
                universes = [row["eligible"] for row in participating]
                if any(universe != universes[0] for universe in universes[1:]):
                    metric = _unique_metric_gap("eligible-unit-universe-mismatch", missing)
                else:
                    covered = owner["covered"] & owner["eligible"]
                    other_covered = set().union(*(
                        row["covered"] & row["eligible"]
                        for row in usable_peers.values()
                    ))
                    unique = covered - other_covered
                    units = sorted(unique)
                    incomplete = bool(missing) or any(
                        row.get("status") == "partial" for row in participating
                    )
                    metric = {
                        "status": "partial" if incomplete else "observed",
                        "unique_covered": len(unique),
                        "eligible": len(owner["eligible"]),
                        "value": round(len(unique) / len(owner["eligible"]), 6)
                        if owner["eligible"] else None,
                        "method_covered": len(covered),
                        "unique_share_of_method_covered": round(
                            len(unique) / len(covered), 6,
                        ) if covered else None,
                        "available_comparators": sorted(usable_peers),
                        "missing_comparators": missing,
                        "unique_units_sha256": _unique_unit_digest(unique),
                    }
                    if missing:
                        metric["reason"] = "comparator-profile-incomplete"
                    elif incomplete:
                        metric["reason"] = "source-coverage-partial"
            metric_results[name] = metric
            artifact_metrics[name] = {
                "status": metric["status"],
                "unique_units": [
                    {"source": source, "unit": unit} for source, unit in units
                ],
            }

        metric_statuses = [
            row["status"] for row in metric_results.values()
            if row["status"] != "NA"
        ]
        if not metric_statuses:
            status = "NA"
        elif all(value == "gap" for value in metric_statuses):
            status = "gap"
        elif any(value in {"gap", "partial"} for value in metric_statuses):
            status = "partial"
        else:
            status = "observed"
        relative_units_path = Path("unique-coverage") / _safe(method) \
            / f"{_safe(target_id)}.json"
        _write_json(output / relative_units_path, {
            "schema_version": UNIQUE_SCHEMA,
            "method": method,
            "target": target_id,
            "comparison": "method-covered-units-minus-other-methods-union",
            "compared_methods": peers_by_method[method],
            "missing_profile_methods": missing_profile_methods,
            "metrics": artifact_metrics,
        })
        results[method] = {
            "schema_version": UNIQUE_SCHEMA,
            "status": status,
            "comparison": "exclusive-within-target-across-methods",
            "compared_methods": peers_by_method[method],
            "metrics": metric_results,
            "units_path": relative_units_path.as_posix(),
        }
    return results


def _add_unique_coverage(
    output: Path, methods: list[str], targets: list[str],
    cells: list[dict[str, Any]],
    target_specs: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    cells_by_key = {(cell["method"], cell["target"]): cell for cell in cells}
    for target_id in targets:
        results = _unique_coverage_for_target(
            output, target_id, methods, cells_by_key, target_specs,
        )
        for method, unique in results.items():
            cell = cells_by_key.get((method, target_id))
            if cell is not None:
                cell["unique_coverage"] = unique


def aggregate_launch_manifests(
    launch_paths: list[Path], output: Path, *, jobs: int = 4,
    catalog_evidence_dir: Path | None = None,
    include_guest_metrics: bool = True,
) -> dict[str, Any]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    methods, targets, experiments = _load_launches(launch_paths)
    runs = [run for experiment in experiments for run in experiment["runs"]]
    output = output.resolve()
    if catalog_evidence_dir is not None:
        catalog_evidence_dir = catalog_evidence_dir.resolve()
        if not catalog_evidence_dir.is_dir():
            raise ValueError(
                f"catalog evidence directory does not exist: {catalog_evidence_dir}"
            )
    for run in runs:
        root = run["run_root"].resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("offline output must be separate from all input run roots")
    if os.path.lexists(output):
        raise FileExistsError(f"offline output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        target_specs = _target_specs(experiments, targets)
        tasks = []
        for method in methods:
            method_runs = [run for run in runs if run["method"] == method]
            for target_id in targets:
                target = target_specs.get((method, target_id), {"id": target_id})
                tasks.append((method, target_id, target, method_runs))
        completed: dict[tuple[str, str], dict[str, Any]] = {}
        completed_count = 0

        def collect_cell(
            method: str, target_id: str, target: dict[str, Any],
            method_runs: list[dict[str, Any]],
        ) -> dict[str, Any]:
            # Keep the legacy callable shape unchanged for existing isolated
            # wrappers; the extra keyword is needed only for source-only mode.
            if include_guest_metrics:
                return _collect_cell(
                    temporary, method, target_id, target, method_runs,
                    catalog_evidence_dir,
                )
            return _collect_cell(
                temporary, method, target_id, target, method_runs,
                catalog_evidence_dir, include_guest_metrics=False,
            )

        with ThreadPoolExecutor(max_workers=min(jobs, len(tasks) or 1)) as pool:
            futures = {
                pool.submit(collect_cell, method, target_id, target, method_runs):
                    (method, target_id)
                for method, target_id, target, method_runs in tasks
            }
            for future in as_completed(futures):
                method, target_id = futures[future]
                try:
                    completed[(method, target_id)] = future.result()
                except Exception as error:
                    reason = f"offline-cell-error:{type(error).__name__}"
                    completed[(method, target_id)] = {
                        "method": method, "target": target_id, "status": "gap",
                        "reason": reason,
                        "run_ids": [row["run_id"] for row in next(
                            task[3] for task in tasks
                            if task[0] == method and task[1] == target_id
                        )],
                        "source_coverage": _source_metrics_gap(reason),
                        "rv_opcode_catalog_coverage": _rv_catalog_gap(reason),
                    }
                completed_count += 1
                cell = completed[(method, target_id)]
                source_status = (cell.get("source_coverage") or {}).get("status", "gap")
                rv_status = (cell.get("rv_opcode_catalog_coverage") or {}).get("status", "gap")
                print(
                    f"[coverage-offline] {completed_count}/{len(tasks)} "
                    f"{method} × {target_id}: SimSrcCov={source_status}, RV={rv_status}",
                    file=sys.stderr, flush=True,
                )
        cells = [completed[(method, target_id)] for method, target_id, *_ in tasks]
        _add_unique_coverage(temporary, methods, targets, cells, target_specs)
        source_statuses = [str(cell.get("status") or "gap") for cell in cells]
        rv_statuses = [
            str((cell.get("rv_opcode_catalog_coverage") or {}).get("status") or "gap")
            for cell in cells
        ]
        unique_statuses = [
            str((cell.get("unique_coverage") or {}).get("status") or "gap")
            for cell in cells
            if (cell.get("unique_coverage") or {}).get("status") != "NA"
        ]
        all_statuses = source_statuses + unique_statuses
        if include_guest_metrics:
            all_statuses += rv_statuses
        status = "recorded" if all(item == "observed" for item in all_statuses) else (
            "partial" if any(item in {"observed", "partial"} for item in all_statuses)
            else "gap"
        )
        payload = {
            "schema_version": SCHEMA,
            "status": status,
            "aggregation": "raw-profile-union-within-method-target",
            "denominator_policy": "one-target-source-scope-per-cell",
            "rv_opcode_catalog": {
                "catalog_id": rv_catalog.OPCODE_CATALOG_ID,
                "registry_sha256": rv_catalog.OPCODE_CATALOG_REGISTRY_SHA256,
                "denominator": rv_catalog.OPCODE_CATALOG_DENOMINATOR,
                "aggregation": "unique-opcode-set-union-across-runs",
            },
            "unique_coverage": {
                "schema_version": UNIQUE_SCHEMA,
                "comparison": "exclusive-within-target-across-methods",
                "aggregation": "method-covered-units-minus-other-methods-union",
                "eligible_units_policy": "require-identical-unit-universe",
                "unit_artifacts": "unique-coverage/<method>/<target>.json",
            },
            "jobs": min(jobs, len(tasks) or 1),
            "guest_metrics_enabled": include_guest_metrics,
            "catalog_evidence_dir": (
                str(catalog_evidence_dir) if catalog_evidence_dir is not None else None
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "methods": methods,
            "targets": targets,
            "matrix_cells_expected": 49,
            "matrix_cells_written": len(cells),
            "experiments": [
                {key: value for key, value in experiment.items() if key != "runs"}
                for experiment in experiments
            ],
            "runs": [
                {
                    "run_id": run["run_id"],
                    "method": run["method"],
                    "launch_status": run["launch_status"],
                    "run_status": run["run_result"].get("status")
                    or run["wrapper"].get("status") or "unavailable",
                    "interrupted": run["interrupted"],
                    "stop_reason": run["stop_reason"],
                    "planned_duration_seconds": run["execution"].get("duration_seconds"),
                    "elapsed_seconds": _elapsed_seconds(
                        run["started_at_utc"], run["finished_at_utc"],
                    ),
                    "execution_complete": (
                        run["run_result"].get("execution_complete") is True
                        or run["wrapper"].get("execution_complete") is True
                    ),
                    "error": run.get("error"),
                }
                for run in runs
            ],
            "cells": cells,
        }
        _write_json(temporary / "summary.json", payload)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="离线合并相同 7×7 RQ1 实验的 SimSrcCov 与 RV Opcode Catalog。",
    )
    parser.add_argument(
        "--launch-manifest", type=Path, action="append", required=True,
        help="results/launch-logs/<prefix>/launch-manifest.json，可重复传入",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--catalog-evidence-dir", type=Path,
        help=("raw-only framework 的离线 RV catalog sidecar 目录；"
              "文件名为 <method>-<target>.json"),
    )
    parser.add_argument(
        "--source-only", action="store_true",
        help="只汇总 SimSrcCov；RV opcode 作为 disabled 诊断保留，不阻塞 source 结果",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[4]
    deps_root = repo_root.parent.parent / "deps"
    os.environ.setdefault("RQ1_DEPS", str(deps_root))
    os.environ.setdefault("RQ1_DOTNET_RUNTIME", str(deps_root / "toolchains/dotnet-8"))
    dotnet_runtime = Path(os.environ["RQ1_DOTNET_RUNTIME"])
    if dotnet_runtime.is_dir():
        os.environ.setdefault("DOTNET_ROOT", str(dotnet_runtime))
        os.environ["PATH"] = str(dotnet_runtime) + os.pathsep + os.environ.get("PATH", "")

    try:
        result = aggregate_launch_manifests(
            args.launch_manifest, args.output, jobs=args.jobs,
            catalog_evidence_dir=args.catalog_evidence_dir,
            include_guest_metrics=not args.source_only,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(2, f"offline coverage aggregation failed: {error}\n")
    print(json.dumps({
        "status": result["status"],
        "matrix_cells": result["matrix_cells_written"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

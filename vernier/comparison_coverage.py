"""Comparison coverage aggregation and replay helpers."""

from __future__ import annotations

import gzip
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from analysis.simulator_coverage import (
    SCHEMA, aggregate_simulator_coverage, gate_experiment_unions,
    SOURCE_SCHEMA, summarize_simulator_coverage,
)
from analysis.source_coverage import (
    coverage_input_files, coverage_input_fingerprint,
    instrumented_binary_identity, SOURCE_COVERAGE_PIPELINE,
)
from comparison_artifacts import digest, sha256
from comparison_observations import _record_artifact_digest, _record_has_artifact_digest
from comparison_state import (
    _event_is_expected_timebox, _event_is_pending, _nonnegative_index,
)


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def _artifact_identity_invalid(record: object) -> bool:
    """Reject explicit identity failure or conflicting artifact aliases."""
    if not isinstance(record, dict):
        return True
    return (
        record.get("artifact_identity_ok") is False
        or (
            _record_has_artifact_digest(record)
            and _record_artifact_digest(record) is None
        )
    )


def _materialize_guest_coverage(
    out: Path, targets: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *, max_workers: int | None = None,
    on_case: Callable[[int, dict[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    """Parse every available case trace once during finalization.

    Coverage is an evidence calculation, not a timed execution phase.  The
    caller keeps the execution wall-clock window, while this function visits
    the complete event set and reports missing/invalid evidence explicitly.
    """
    started = time.monotonic()
    target_by_id = {str(target.get("id")): target for target in targets}
    feature_cache: dict[str, dict[str, Any]] = {}
    feature_lock = Lock()

    def feature_for(case_root: Path, event: dict[str, Any], row: dict[str, Any],
                    target: dict[str, Any]) -> dict[str, Any]:
        stratum = _stratum(target)
        artifact_id = (_record_artifact_digest(row) or _record_artifact_digest(event)
                       or event.get("bare_elf_sha256") or event.get("linux_elf_sha256"))
        cache_key = f"{stratum}:{artifact_id or case_root}"
        with feature_lock:
            if cache_key in feature_cache:
                return feature_cache[cache_key]
        name = f"{stratum.replace('-', '_')}.features.json"
        paths = [case_root / "artifacts" / name]
        if isinstance(artifact_id, str) and artifact_id:
            paths.extend(sorted((out / "artifact-features").glob(
                f"{artifact_id}-*-{stratum.replace('-', '_')}.features.json")))
        value: object = None
        for path in paths:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(value, dict):
                break
        feature = value if isinstance(value, dict) else {}
        with feature_lock:
            feature_cache[cache_key] = feature
        return feature

    def materialize(index: int) -> tuple[str, int]:
        event = events[index]
        # The partial ledger is an external/persisted input. A malformed row
        # must become an evidence error and leave the remaining cases
        # available for aggregation; calling ``.get`` on it here used to abort
        # the whole finalization pass.
        if not isinstance(event, dict):
            return "error", index
        if event.get("target_attempted") is not True:
            return ("pending" if _event_is_pending(event)
                    else "not-attempted"), index
        if _event_is_pending(event):
            # A process that started before the shared deadline is right
            # censored.  It is attempted, but it is no longer part of the
            # unattempted queue remainder.
            return "right-censored", index
        event_artifact = _record_artifact_digest(event)
        if _artifact_identity_invalid(event):
            return "identity", index
        existing_coverage = event.get("simulator_coverage")
        if isinstance(existing_coverage, dict):
            return (
                "existing" if existing_coverage.get("schema_version") == SCHEMA else "error",
                index,
            )
        target_id = str(event.get("target") or "")
        target = target_by_id.get(target_id)
        method = event.get("method")
        if not target or not method:
            return "missing", index
        candidate = _nonnegative_index(event.get("stream_index"))
        if candidate is None:
            return "missing", index
        case_root = (out / "executions" / _safe(target_id) / _safe(method)
                     / f"candidate-{candidate}")
        target_run = case_root / "target-run.json"
        try:
            result = json.loads(target_run.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return "missing", index
        row = _target_row(result, target_id)
        if not row:
            return "missing", index
        expected_artifact = event_artifact or event.get(
            "linux_elf_sha256" if _stratum(target) == "linux-user" else "bare_elf_sha256"
        )
        actual_artifact = _record_artifact_digest(row)
        if (not isinstance(expected_artifact, str)
                or not isinstance(actual_artifact, str)
                or expected_artifact.lower() != actual_artifact.lower()):
            return "identity", index
        try:
            feature = feature_for(case_root, event, row, target)
            coverage = summarize_simulator_coverage(target, row, case_root, feature)
            if not isinstance(coverage, dict) or coverage.get("schema_version") != SCHEMA:
                return "error", index
        except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError):
            return "error", index
        event["simulator_coverage"] = coverage
        return "parsed", index

    indexes = list(range(len(events)))
    if indexes:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            if on_case is None:
                states = list(pool.map(materialize, indexes))
            else:
                states = []
                futures = [pool.submit(materialize, index) for index in indexes]
                for future in as_completed(futures):
                    state, index = future.result()
                    try:
                        on_case(index, events[index], state)
                    except (OSError, TypeError, ValueError, RuntimeError):
                        state = "error"
                    states.append((state, index))
    else:
        states = []
    counts = Counter(state for state, _ in states)
    return {
        "cases": len(events), "parsed_cases": counts["parsed"],
        "missing_cases": counts["missing"],
        "identity_mismatch_cases": counts["identity"],
        "error_cases": counts["error"],
        "pending_cases": counts["pending"],
        "right_censored_cases": counts["right-censored"],
        "not_attempted_cases": counts["not-attempted"],
        "elapsed_s": round(time.monotonic() - started, 6),
    }

def _reusable_source_coverage(
    out: Path, target: dict[str, Any], raw_dir: Path,
    *, expected_coverage_cases: int | None = None,
) -> dict[str, Any] | None:
    """Reuse only a complete source merge whose raw input identity still matches."""
    path = out / "coverage" / _safe(target.get("id")) / "result.json"
    spec = target.get("source_coverage")
    if not isinstance(spec, dict) or not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    collection = row.get("collection") if isinstance(row, dict) else None
    source = row.get("source") if isinstance(row, dict) else None
    if not isinstance(collection, dict) or not isinstance(source, dict):
        return None
    if source.get("schema_version") != SOURCE_SCHEMA:
        return None
    source_identity = source.get("identity")
    if (not isinstance(source_identity, dict)
            or source_identity.get("collector_pipeline") != SOURCE_COVERAGE_PIPELINE):
        return None
    if collection.get("collector_pipeline") != SOURCE_COVERAGE_PIPELINE:
        return None
    if collection.get("status") != "observed" or source.get("status") not in {
        "observed", "partial", "NA", "gap",
    }:
        return None
    if collection.get("coverage_binary") != target.get("coverage_binary"):
        return None
    if collection.get("source_commit_expected") != spec.get("source_commit"):
        return None
    expected_binary_sha256 = spec.get("coverage_binary_sha256") or target.get("coverage_binary_sha256")
    observed_binary_sha256 = collection.get("coverage_binary_expected_sha256")
    binary_identity_ok = (
        observed_binary_sha256 in (None, "")
        if expected_binary_sha256 in (None, "") else
        isinstance(observed_binary_sha256, str)
        and isinstance(expected_binary_sha256, str)
        and observed_binary_sha256.lower() == expected_binary_sha256.lower()
    )
    if not binary_identity_ok or collection.get("coverage_binary_sha256_match") is not True:
        return None
    if (collection.get("instrumented_binaries_match") is not True
            or collection.get("instrumented_binaries") != instrumented_binary_identity(spec)):
        return None
    if any(
        collection.get(name) != spec.get(name, [] if name != "source_root" else None)
        for name in ("source_scope", "source_scope_exclude", "source_root")
    ):
        return None
    if (str(spec.get("collector", "gcov")) in {"dotnet", "llvm"}
            and expected_coverage_cases is not None):
        coverage_scope = collection.get("coverage_scope")
        if not isinstance(coverage_scope, dict):
            return None
        stored_expected = coverage_scope.get("coverage_eligible_cases")
        if stored_expected is None:
            stored_tested = coverage_scope.get("target_tested_cases")
            stored_inputs = coverage_scope.get("source_input_files")
            stored_expected = (
                stored_tested if isinstance(stored_tested, int) and stored_tested > 0
                else stored_inputs
            )
        if stored_expected != expected_coverage_cases:
            return None
        observed_inputs = _source_input_file_count(raw_dir, spec.get("collector"))
        inputs_incomplete = (
            observed_inputs != expected_coverage_cases
            if str(spec.get("collector")) == "dotnet" else
            observed_inputs < expected_coverage_cases
        )
        if inputs_incomplete:
            return None
    if collection.get("input_fingerprint") != coverage_input_fingerprint(raw_dir, spec):
        return None
    profile = Path(str(collection.get("profile") or ""))
    if not profile.is_absolute():
        profile = out / profile
    if not profile.is_file():
        return None
    collection = {**collection, "reused": True, "elapsed_s": 0.0}
    return {"collection": collection, "source": source}


def _stratum(target: dict[str, Any]) -> str:
    return "linux-user" if str(target.get("execution_model", "")).startswith("linux-user") else "bare-metal"

def _target_row(result: dict[str, Any], target_id: str) -> dict[str, Any]:
    rows = result.get("targets", ()) if isinstance(result, dict) else ()
    if not isinstance(rows, (list, tuple)):
        return {}
    return next((row for row in rows
                 if isinstance(row, dict) and row.get("id") == target_id), {})

def _execution_group_key(
    item: dict[str, Any], target_id: str | None = None,
    target: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, str]:
    target_name = str(target_id or item.get("target") or item.get("id") or "")
    stratum = str(item.get("stratum") or (_stratum(target) if target else ""))
    return tuple(str(item.get(name) or "") for name in ("method", "route", "lane")) + (
        stratum, target_name)


def _case_identity(
    item: dict[str, Any], *, target_id: object = None, event: bool = False,
) -> tuple[object, ...]:
    """Return the queue/event join key used by the execution ledger."""
    candidate = item if event else item.get("candidate", {})
    candidate = candidate if isinstance(candidate, dict) else {}
    raw_index = candidate.get("stream_index" if event else "index")
    # Old summary fixtures (which have no event_id) omitted stream_index; their
    # only possible case was candidate zero.  New events must carry the
    # explicit index or they remain an identity gap.
    if raw_index is None and event and not item.get("event_id"):
        # Legacy hand-written event fixtures omitted stream_index; only those
        # events can safely use the historical candidate-0 fallback. Persisted
        # queue entries must always carry an explicit candidate index.
        raw_index = 0
    index = _nonnegative_index(raw_index)
    target = target_id if target_id is not None else item.get("target")

    def identity_part(value: object) -> object:
        # Persisted ledgers can contain a malformed list/object in a key field.
        # Keep finalization fail-closed instead of crashing when the key enters
        # a set; a fresh sentinel also prevents malformed rows from joining.
        if value is None or isinstance(value, bool):
            return object()
        if isinstance(value, float) and not math.isfinite(value):
            return object()
        if isinstance(value, (str, int, float)):
            text = str(value)
            return text if text else object()
        return object()

    return (
        identity_part(target), identity_part(item.get("method")),
        identity_part(item.get("route")), identity_part(item.get("lane")),
        index if index is not None else object(),
    )


def _coverage_summary(
    events: list[dict[str, Any]], queues: dict[str, list[dict[str, Any]]],
    targets: list[dict[str, Any]], method_filter: str | None = None,
    *, aggregate_indexes: list[int] | None = None,
    neutralize_timebox: bool = False,
    include_guest_metrics: bool = True,
) -> dict[str, Any]:
    """Keep aggregate coverage values tied to the full queued-case denominator."""
    target_rows = targets if isinstance(targets, (list, tuple)) else ()
    target_by_id = {
        str(target.get("id")): target
        for target in target_rows
        if isinstance(target, dict)
    }
    queue_rows = queues if isinstance(queues, dict) else {}
    event_rows = events if isinstance(events, (list, tuple)) else ()
    groups: dict[tuple[str, str, str, str, str], Counter] = {}
    queue_keys: set[tuple[object, ...]] = set()
    event_keys: set[tuple[object, ...]] = set()
    aggregate_events: list[dict[str, Any]] = []
    if aggregate_indexes is not None:
        aggregate_indexes.clear()
    duplicate_queue_count = 0
    duplicate_event_count = 0
    orphan_event_count = 0
    artifact_identity_mismatch_count = 0
    malformed_queue_count = 0
    malformed_event_count = 0

    for target_id, raw_entries in queue_rows.items():
        entries = raw_entries if isinstance(raw_entries, (list, tuple)) else ()
        malformed_queue_count += (
            1 if raw_entries is not None and not isinstance(raw_entries, (list, tuple))
            else 0
        )
        target = target_by_id.get(str(target_id), {})
        for entry in entries:
            if not isinstance(entry, dict):
                malformed_queue_count += 1
                continue
            if method_filter is not None and entry.get("method") not in (
                None, method_filter,
            ):
                continue
            key = _execution_group_key(entry, str(target_id), target)
            group = groups.setdefault(key, Counter())
            group["queued_cases"] += 1
            case_key = _case_identity(entry, target_id=target_id)
            if case_key in queue_keys:
                duplicate_queue_count += 1
                group["identity_mismatch_cases"] += 1
            else:
                queue_keys.add(case_key)
    for event_index, event in enumerate(event_rows):
        if not isinstance(event, dict):
            malformed_event_count += 1
            continue
        if method_filter is not None and event.get("method") not in (
            None, method_filter,
        ):
            continue
        # This diagnostic explains an empty generation result; it is not a
        # queue entry and must not become a coverage case.
        if event.get("reason_code") == "generation-produced-no-candidate":
            continue
        key = _execution_group_key(event, target=target_by_id.get(str(event.get("target")), {}))
        group = groups.setdefault(key, Counter())
        case_key = _case_identity(event, event=True)
        is_new_event = case_key not in event_keys
        if is_new_event:
            event_keys.add(case_key)
        else:
            duplicate_event_count += 1
            group["identity_mismatch_cases"] += 1
        matched_queue = case_key in queue_keys
        if not matched_queue:
            orphan_event_count += 1
            group["identity_mismatch_cases"] += 1
        group["event_cases"] += 1
        outcome = event.get("outcome")
        coverage = event.get("simulator_coverage")
        source_record = event.get("simulator_source_coverage")
        if include_guest_metrics:
            case_coverage_status = (
                str(coverage.get("status", "gap"))
                if isinstance(coverage, dict) else "partial"
            )
        else:
            case_coverage_status = (
                str(source_record.get("status", "gap"))
                if isinstance(source_record, dict)
                and source_record.get("schema_version") == SOURCE_SCHEMA
                else "partial" if isinstance(coverage, dict) else "gap"
            )
        timebox_censored = (
            neutralize_timebox and matched_queue and is_new_event
            and _event_is_expected_timebox(event)
        )
        if timebox_censored:
            group["timebox_censored_cases"] += 1
        trace_incomplete = (
            not timebox_censored and include_guest_metrics
            and _trace_incomplete_event(event, coverage)
        )
        if (
            matched_queue and is_new_event
            and event.get("target_attempted") is True
            and trace_incomplete
        ):
            group["trace_incomplete_cases"] += 1
        if matched_queue and is_new_event and _event_is_pending(event):
            if event.get("target_attempted") is True:
                group["right_censored_cases"] += 1
            else:
                group["pending_cases"] += 1
        # Keep execution/observation failures visible, but do not turn a
        # complete simulator trace into a coverage failure merely because the
        # guest ended with an exception or nonzero exit.  The state-frame and
        # differential channels remain separately accounted for by the event
        # ledger.
        elif matched_queue and is_new_event and not timebox_censored:
            observation_error = (
                (isinstance(outcome, str)
                 and outcome in {"gap", "timeout", "crash", "nonzero-exit"})
                or (isinstance(event.get("status"), str)
                    and event.get("status") in {"error", "failed"})
            )
            if observation_error:
                group["observation_error_cases"] += 1
                if include_guest_metrics:
                    coverage_closed = (
                        isinstance(coverage, dict)
                        and coverage.get("schema_version") == SCHEMA
                        and coverage.get("status") == "observed"
                    )
                else:
                    coverage_closed = (
                        isinstance(source_record, dict)
                        and source_record.get("schema_version") == SOURCE_SCHEMA
                        and source_record.get("status") in {"observed", "partial", "NA"}
                    )
                if not coverage_closed:
                    group["error_cases"] += 1
        artifact_identity_invalid = _artifact_identity_invalid(event)
        if matched_queue and is_new_event and artifact_identity_invalid:
            group["identity_mismatch_cases"] += 1
            artifact_identity_mismatch_count += 1
        executable_event = (
            matched_queue and is_new_event
            and event.get("target_attempted") is True
            and not _event_is_pending(event)
            and not timebox_censored
            and not artifact_identity_invalid
        )
        # A parsed prefix is safe numerator evidence.  The trace completeness
        # flag still gates its ratio below, but dropping the prefix here would
        # make a timeout erase instructions that were already observed.
        if executable_event:
            aggregate_events.append(event)
            if aggregate_indexes is not None:
                aggregate_indexes.append(event_index)
        coverage_present = (
            isinstance(coverage, dict)
            and coverage.get("schema_version") == SCHEMA
            if include_guest_metrics else
            isinstance(source_record, dict)
            and source_record.get("schema_version") == SOURCE_SCHEMA
        )
        if executable_event and coverage_present:
            group["coverage_cases"] += 1
            if case_coverage_status == "partial":
                group["partial_coverage_cases"] += 1
            elif case_coverage_status in {"gap", "NA"}:
                group["gap_coverage_cases"] += 1

    summary = aggregate_simulator_coverage(
        aggregate_events, include_guest_metrics=include_guest_metrics,
    )
    rows = {
        _execution_group_key(row, target_id=str(row.get("target"))): row
        for row in summary.get("targets", ()) if isinstance(row, dict)
    }
    scope_groups = []
    for key, group in sorted(groups.items()):
        queued = group["queued_cases"]
        coverage_cases = group["coverage_cases"]
        timebox_censored = group["timebox_censored_cases"]
        eligible_queued = max(0, queued - timebox_censored)
        excluded = max(0, eligible_queued - coverage_cases)
        row = rows.get(key)
        counts = {
            "queued_cases": queued,
            "event_cases": group["event_cases"],
            "coverage_cases": coverage_cases,
            "excluded_cases": excluded,
            "pending_cases": group["pending_cases"],
            "right_censored_cases": group["right_censored_cases"],
            "timebox_censored_cases": timebox_censored,
            "error_cases": group["error_cases"],
            "observation_error_cases": group["observation_error_cases"],
            "partial_coverage_cases": group["partial_coverage_cases"],
            "gap_coverage_cases": group["gap_coverage_cases"],
            "trace_incomplete_cases": group["trace_incomplete_cases"],
            "identity_mismatch_cases": group["identity_mismatch_cases"],
        }
        if row is not None:
            row.setdefault("coverage_basis", {}).update(counts)
            unexpected_pending = max(0, group["pending_cases"] - timebox_censored)
            unexpected_right_censored = max(
                0, group["right_censored_cases"] - timebox_censored,
            )
            incomplete = bool(
                unexpected_pending or unexpected_right_censored
                or group["error_cases"] or excluded
                or group["trace_incomplete_cases"]
                or group["identity_mismatch_cases"]
            )
            if incomplete and row.get("status") == "observed":
                row["status"] = "partial" if coverage_cases else "gap"
                for metric in row.get("guest", {}).values():
                    if isinstance(metric, dict) and metric.get("status") == "observed":
                        metric["status"] = row["status"]
                        metric["reason"] = "batch-observation-incomplete"
                        metric["value"] = None
        else:
            unexpected_pending = max(0, group["pending_cases"] - timebox_censored)
            unexpected_right_censored = max(
                0, group["right_censored_cases"] - timebox_censored,
            )
        expected_timebox_only = bool(
            neutralize_timebox and timebox_censored
            and not (
                unexpected_pending or unexpected_right_censored
                or group["error_cases"] or excluded
                or group["trace_incomplete_cases"]
                or group["identity_mismatch_cases"]
            )
        )
        row_status = row.get("status") if row else (
            "observed" if expected_timebox_only else
            "partial" if coverage_cases or group["pending_cases"]
            or group["right_censored_cases"] else "gap")
        if group["identity_mismatch_cases"] and row_status == "observed":
            row_status = "partial" if coverage_cases else "gap"
        scope_groups.append({**{name: value for name, value in zip(
            ("method", "route", "lane", "stratum", "target"), key)},
            **counts, "status": row_status,
            **({"reason": "timebox-censored"} if expected_timebox_only else {}),
        })

    queued_cases = sum(group["queued_cases"] for group in groups.values())
    coverage_cases = sum(group["coverage_cases"] for group in groups.values())
    pending_cases = sum(group["pending_cases"] for group in groups.values())
    right_censored_cases = sum(
        group["right_censored_cases"] for group in groups.values()
    )
    timebox_censored_cases = sum(
        group["timebox_censored_cases"] for group in groups.values()
    )
    error_cases = sum(group["error_cases"] for group in groups.values())
    observation_error_cases = sum(
        group["observation_error_cases"] for group in groups.values()
    )
    trace_incomplete_cases = sum(group["trace_incomplete_cases"] for group in groups.values())
    excluded_cases = max(0, queued_cases - timebox_censored_cases - coverage_cases)
    unexpected_pending_cases = max(0, pending_cases - timebox_censored_cases)
    unexpected_right_censored_cases = max(
        0, right_censored_cases - timebox_censored_cases,
    )
    identity_mismatch_cases = (
        duplicate_queue_count + duplicate_event_count
        + len(queue_keys - event_keys) + len(event_keys - queue_keys)
        + malformed_queue_count + malformed_event_count
        + artifact_identity_mismatch_count
    )
    incomplete = bool(
        unexpected_pending_cases or unexpected_right_censored_cases or error_cases
        or excluded_cases or trace_incomplete_cases
        or identity_mismatch_cases
    )
    if incomplete:
        if summary.get("status") in {"observed", "NA"} and (
                coverage_cases or unexpected_pending_cases or unexpected_right_censored_cases
                or trace_incomplete_cases):
            summary["status"] = "partial"
        elif summary.get("status") == "observed":
            summary["status"] = "gap"
        summary["reason"] = "batch-observation-incomplete"
    elif neutralize_timebox and timebox_censored_cases:
        if summary.get("status") == "gap":
            summary["status"] = "partial"
        summary["reason"] = "timebox-censored"
    if trace_incomplete_cases:
        summary["reason"] = "trace-source-truncated"
    if identity_mismatch_cases:
        summary["reason"] = "queue-event-identity-mismatch"
    summary["execution_scope"] = {
        "denominator": "queued-target-cases",
        "queued_cases": queued_cases,
        "event_cases": sum(group["event_cases"] for group in groups.values()),
        "coverage_cases": coverage_cases,
        "excluded_cases": excluded_cases,
        "pending_cases": pending_cases,
        "right_censored_cases": right_censored_cases,
        "timebox_censored_cases": timebox_censored_cases,
        "error_cases": error_cases,
        "observation_error_cases": observation_error_cases,
        "trace_incomplete_cases": trace_incomplete_cases,
        "identity_mismatch_cases": identity_mismatch_cases,
        "duplicate_queue_cases": duplicate_queue_count,
        "duplicate_event_cases": duplicate_event_count,
        "malformed_queue_cases": malformed_queue_count,
        "malformed_event_cases": malformed_event_count,
        "orphan_event_cases": orphan_event_count,
        "unaccounted_queue_cases": len(queue_keys - event_keys),
        "queue_event_identity_complete": identity_mismatch_cases == 0,
        "status": summary.get("status"),
        "groups": scope_groups,
    }
    preserve_generation = bool(
        queued_cases > 0
        and coverage_cases == queued_cases
        and not pending_cases
        and not right_censored_cases
        and not excluded_cases
        and not identity_mismatch_cases
    )
    if incomplete:
        gate_experiment_unions(
            summary, str(summary.get("status") or "gap"),
            str(summary.get("reason") or "batch-observation-incomplete"),
            preserve_generation=preserve_generation,
            # Per-target rows were already gated against their own queue and
            # event counts above.  The campaign gap only invalidates the
            # whole-experiment union; rewriting target numerators here would
            # lose valid observations from completed targets.
            include_targets=False,
        )
    summary["method_filter"] = method_filter
    return summary

def _source_input_file_count(raw_dir: Path, collector: object) -> int:
    collector = str(collector or "gcov")
    if collector == "dotnet":
        return len(coverage_input_files(raw_dir, collector))
    pattern = {"llvm": "*.profraw", "gcov": "*.gcda", "lcov": "*.gcda"}.get(collector)
    return len({path.resolve() for path in raw_dir.rglob(pattern)}) if pattern else 0


def _trace_incomplete_event(event: dict[str, Any], coverage: object) -> bool:
    observation = event.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    if (
        event.get("trace_complete") is False
        or observation.get("trace_complete") is False
        or event.get("trace_cleanup_status") == "gap"
        or observation.get("trace_cleanup_status") == "gap"
        or any(
            event.get(name) is True or observation.get(name) is True
            for name in ("trace_truncated", "trace_limit_exceeded")
        )
    ):
        return True
    if isinstance(coverage, dict):
        trace = (coverage.get("simulator_events") or {}).get("trace", {})
        return isinstance(trace, dict) and trace.get("truncated") is True
    return False

def _coverage_case_record(event: dict[str, Any]) -> dict[str, Any]:
    """Keep the recomputable coverage facts, excluding stdout/stderr and raw traces."""
    record = {
        name: event.get(name) for name in _COVERAGE_CASE_FIELDS
        if name in event and not name.endswith("_digest")
    }
    # 不把完整目标身份复制到每条账本记录；保存紧凑摘要供重放分组。
    # 缺少这个摘要会让有效账本的重放结果与内存聚合不一致，并掩盖二进制混用。
    identity_id = event.get("target_identity_id")
    if not identity_id:
        for identity in (event.get("target_identity"), event.get("identity")):
            if not isinstance(identity, dict):
                continue
            identity_id = (
                identity.get("identity_digest")
                or identity.get("expected_identity_digest")
                or identity.get("binary_sha256")
                or identity.get("binary")
            )
            if identity_id:
                break
    if identity_id:
        record["target_identity_id"] = str(identity_id)
    # SimSrcCov is collected once per target/run for gcov, dotnet and merged
    # LLVM profiles.  Copying its full payload into every case makes a
    # target-run measurement look like a per-case measurement.  Keep only the
    # scope marker here; the complete numerator/denominator stays in
    # ``coverage/summary.json`` under ``source_targets``.
    if isinstance(event.get("simulator_source_coverage"), dict):
        record["simulator_source_coverage_scope"] = "target-run"
    if "simulator_coverage" in record:
        record["simulator_coverage_digest"] = digest(record["simulator_coverage"])
    return record

def _write_coverage_case_results(
    out: Path, events: list[dict[str, Any]], method_filter: str | None,
    *, aggregate_indexes: list[int] | None = None,
    include_guest_metrics: bool = True,
) -> dict[str, Any]:
    """Persist one compressed, case-level coverage ledger for independent replay."""
    path = out / "coverage" / "case-results.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    aggregate_index_set = set(aggregate_indexes) if aggregate_indexes is not None else None
    indexed_records = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        record = _coverage_case_record(event)
        record["coverage_aggregate_included"] = (
            True if aggregate_index_set is None
            else event_index in aggregate_index_set
        )
        indexed_records.append((record, event))
    records = [record for record, _event in indexed_records]
    expected_records = [
        event for record, event in indexed_records
        if record["coverage_aggregate_included"]
    ]
    try:
        with temp_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
                header = {
                    "schema_version": "rq1-coverage-case-results-v1",
                    "run_id": out.name, "method_filter": method_filter,
                    "record_count": len(records),
                    "aggregate_record_count": len(expected_records),
                }
                stream.write((json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n").encode())
                for record in records:
                    stream.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode())
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    def signature(summary: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
        result = {}
        for row in summary.get("targets", ()):
            if not isinstance(row, dict):
                continue
            key = tuple(str(row.get(name) or "") for name in
                        ("method", "route", "lane", "stratum", "target"))
            result[key] = {
                "status": row.get("status"),
                "cases": row.get("cases"),
                "observed_cases": row.get("observed_cases"),
                "incomplete_cases": row.get("incomplete_cases"),
                "coverage_basis": {
                    name: (row.get("coverage_basis") or {}).get(name)
                    for name in (
                        "PCov", "ICov-encoding", "ICov-type", "CCov",
                        "coverage_profile_id", "coverage_registry_sha256",
                        "cohort_id", "cohort_cases",
                    )
                },
                "events": row.get("simulator_event_counts", {}),
                "guest": {
                    name: {
                        key: metric.get(key) for key in (
                            "aggregation", "value", "status", "covered", "eligible",
                            "artifact_count", "artifacts_with_hits",
                            "covered_units_sha256", "eligible_units_sha256",
                            "covered_units", "eligible_units", "coverage_profile_id",
                            "coverage_registry_sha256", "registry_sha256", "reason",
                            "identity_missing_record_count", "static_map_missing_record_count",
                            "invalid_unit_record_count",
                        ) if key in metric
                    }
                    for name, metric in (row.get("guest") or {}).items()
                    if isinstance(metric, dict)
                },
                "experiment_unions": summary.get("experiment_unions", []),
            }
        return result

    def identity_signature(records: list[dict[str, Any]]) -> Counter[str]:
        return Counter(json.dumps(
            {name: record.get(name) for name in _COVERAGE_CASE_ID_FIELDS},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) for record in records)

    replay_match = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            header = json.loads(next(stream))
            replay_records = [json.loads(line) for line in stream if line.strip()]
        replay_flags = [
            record.get("coverage_aggregate_included") is True
            for record in replay_records
        ]
        expected_flags = [
            record.get("coverage_aggregate_included") is True
            for record in records
        ]
        replay_input = [
            record for record in replay_records
            if record.get("coverage_aggregate_included") is True
        ]
        expected_input = expected_records
        if not include_guest_metrics:
            # SimSrcCov is a target-run profile and is deliberately kept out
            # of the per-case ledger.  Replay the case identity/execution
            # scope with that target-level payload omitted on both sides; the
            # sealed summary and target result remain the source-profile
            # authority.
            replay_input = [
                {key: value for key, value in record.items()
                 if key != "simulator_source_coverage"}
                for record in replay_input
            ]
            expected_input = [
                {key: value for key, value in record.items()
                 if key != "simulator_source_coverage"}
                for record in expected_input
            ]
        replay = aggregate_simulator_coverage(
            replay_input, include_guest_metrics=include_guest_metrics,
        )
        header_match = (
            isinstance(header, dict)
            and header.get("schema_version") == "rq1-coverage-case-results-v1"
            and header.get("record_count") == len(replay_records)
            and header.get("aggregate_record_count") == sum(replay_flags)
        )
        expected = aggregate_simulator_coverage(
            expected_input, include_guest_metrics=include_guest_metrics,
        )
        aggregate_match = signature(replay) == signature(expected)
        identity_match = (
            replay_flags == expected_flags
            and identity_signature(replay_records) == identity_signature(records)
        )
        replay_match = header_match and aggregate_match and identity_match
    except (OSError, StopIteration, TypeError, ValueError):
        replay = {}
        header_match = aggregate_match = identity_match = False
    return {
        "path": str(path.relative_to(out)), "sha256": sha256(path),
        "format": "jsonl+gzip", "records": len(records),
        "replay": "aggregate_simulator_coverage",
        "replay_match": replay_match,
        "replay_checks": {
            "header": header_match,
            "aggregate_scope_and_metrics": aggregate_match,
            "case_identity_and_execution_scope": identity_match,
        },
    }
_COVERAGE_CASE_FIELDS = (
    "event_id", "method", "route", "lane", "stratum", "target", "case_id",
    "candidate_id", "stream_index", "artifact_sha256", "artifact_id", "bare_elf_sha256",
    "linux_elf_sha256", "artifact_isa", "artifact_identity_ok", "status",
    "outcome", "reason_code", "target_attempted",
    "right_censored", "deadline_censored", "case_outcome", "failure_class",
    "run_deadline_censored",
    "formal_eligible", "coverage_status", "simulator_coverage",
    "simulator_coverage_digest", "simulator_source_coverage_scope", "target_identity_id",
)

_COVERAGE_CASE_ID_FIELDS = (
    "event_id", "method", "route", "lane", "stratum", "target", "case_id",
    "candidate_id", "stream_index", "artifact_sha256", "artifact_id", "bare_elf_sha256",
    "linux_elf_sha256", "artifact_isa", "artifact_identity_ok", "status", "outcome",
    "reason_code", "target_attempted",
    "right_censored", "deadline_censored", "case_outcome", "failure_class",
    "run_deadline_censored",
    "formal_eligible", "simulator_coverage_digest",
    "simulator_source_coverage_scope", "target_identity_id",
)

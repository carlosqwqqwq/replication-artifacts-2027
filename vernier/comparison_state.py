"""Comparison 的候选、队列和计数事实。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from comparison_artifacts import load_json
from comparison_observations import _record_is_pending, _trusted_observation_outcome


_CANDIDATE_PATH_FIELDS = (
    "source", "elf", "linux_elf",
    "parent_source", "parent_elf", "parent_linux_elf",
)

_EXPECTED_TIMEBOX_REASONS = frozenset({
    "deadline", "run-wall-clock-exhausted", "execution-wall-clock-exhausted",
    "campaign-wall-clock-exhausted", "framework-target-wall-clock-exhausted",
    "run-deadline-censored", "execution-deadline-censored",
    "target-wall-clock-pending",
})


def _nonnegative_index(value: object, *, default: int | None = None) -> int | None:
    """Normalize a persisted case index without treating booleans as numbers."""
    if isinstance(value, bool):
        return default
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed >= 0 else default


def _candidate_identity_key(candidate: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    """以候选物料摘要（或路径回退）和预期终态去重。"""
    material = tuple(
        (
            f"{field}:sha256:{candidate.get(field + '_sha256').lower()}"
            if isinstance(candidate.get(field + "_sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", candidate.get(field + "_sha256")) is not None
            else f"path:{candidate.get(field) or ''}"
        )
        for field in _CANDIDATE_PATH_FIELDS
    )
    return material, str(candidate.get("expected_outcome") or "")


def _metric_ratio(
    numerator: object, denominator: object, *, bounded: bool = False,
) -> float | None:
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        return None
    try:
        numerator_value = float(numerator) if numerator is not None else None
        denominator_value = float(denominator) if denominator is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        numerator_value is None or denominator_value is None
        or not math.isfinite(numerator_value)
        or not math.isfinite(denominator_value)
        or numerator_value < 0 or denominator_value <= 0
        or (bounded and numerator_value > denominator_value)
    ):
        return None
    return round(numerator_value / denominator_value, 6)


def _generation_rates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_built = row.get("source_built_count")
    # ``pair_built_count`` is retained as a compatibility alias for old
    # ledgers.  The active metric is the number of candidates with at least
    # one usable ELF projection; a missing second conversion is a target-side
    # gap, not a generation rejection.
    pair_built = row.get("artifact_built_count", row.get("pair_built_count"))
    accepted = row.get("accepted_count")
    attempted = row.get("attempt_count")
    elapsed = row.get("method_elapsed_s", row.get("elapsed_s"))
    return {
        "source_build_rate": {
            "numerator": source_built, "denominator": attempted,
            "value": _metric_ratio(source_built, attempted, bounded=True),
        },
        "pair_build_rate": {
            "numerator": pair_built, "denominator": source_built,
            "value": _metric_ratio(pair_built, source_built, bounded=True),
        },
        "accept_rate": {
            "numerator": accepted, "denominator": pair_built,
            "value": _metric_ratio(accepted, pair_built, bounded=True),
        },
        "accepted_per_second": {
            "numerator": accepted, "denominator": elapsed,
            "value": _metric_ratio(accepted, elapsed),
        },
    }


def _refresh_generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    source_built = row.get("source_built_count")
    if source_built is None:
        source_built = row.get("source_count")
    if source_built is None:
        source_built = row.get("source_generated")
    if source_built is None and isinstance(row.get("candidate_artifacts"), list):
        artifacts = row["candidate_artifacts"]
        if all(isinstance(item, dict) and item.get("source") for item in artifacts):
            source_built = len(artifacts)
    row["source_built_count"] = source_built
    row["source_count"] = source_built
    row["generation_rates"] = _generation_rates(row)
    return row


def _event_is_pending(event: dict[str, Any]) -> bool:
    """识别右删失事件；事件可能已经启动但尚未完成。"""
    return _record_is_pending(event)


def _event_is_expected_timebox(event: dict[str, Any]) -> bool:
    """识别由本轮墙钟上限造成的中性删失，而非外部中断或真实错误。"""
    if not isinstance(event, dict):
        return False
    sources = [event]
    observation = event.get("observation")
    if isinstance(observation, dict):
        sources.append(observation)
    stages = event.get("stage_records")
    if isinstance(stages, dict) and isinstance(stages.get("target"), dict):
        sources.append(stages["target"])
    controlled_timeout = any(
        isinstance(source.get(name), str)
        and source.get(name) in {"wall-timeout", "target-timeout"}
        for source in sources
        for name in ("reason_code", "pending_reason", "reason", "failure_class", "termination")
    ) and any(source.get("outcome") == "timeout" for source in sources)
    if event.get("target_attempted") is True and controlled_timeout:
        return True
    if not _event_is_pending(event):
        return False
    for source in sources:
        if source.get("run_deadline_censored") is True:
            return True
        if any(
            isinstance(source.get(name), str)
            and source.get(name) in _EXPECTED_TIMEBOX_REASONS
            for name in (
                "reason_code", "pending_reason", "reason", "failure_class",
                "termination", "case_outcome",
            )
        ):
            return True
    return False


def _execution_event_closed(event: dict[str, Any]) -> bool:
    """判断目标事件是否已有实际结果或具名的执行缺口。"""
    if _event_is_pending(event):
        return False
    status = event.get("status")
    if not isinstance(status, str) or status not in {
        "passed", "failed", "timeout", "error", "gap",
        "adapter-probe", "adapter-probe-timeout",
    }:
        return False
    outcome = event.get("outcome")
    if isinstance(outcome, str) and outcome in {"case-skipped", "artifact-gap", "transport-gap"}:
        return isinstance(event.get("reason_code"), str)
    trusted_outcome = _trusted_observation_outcome(event)
    named_gap = (
        outcome == "gap"
        and status in {"error", "failed", "gap"}
        and isinstance(event.get("reason_code") or event.get("reason"), str)
    )
    return bool(
        named_gap
        or (
            event.get("target_attempted") is True
            and trusted_outcome is not None and isinstance(outcome, str)
            and trusted_outcome == {
                "complete_observation": "normal",
                "expected-trap": "trap",
            }.get(outcome, outcome)
        )
    )


def _tool_record(
    method: str, status: str, reason: str | None = None, **fields: Any,
) -> dict[str, Any]:
    row = {
        "method": method, "status": status,
        **({"reason": reason} if reason else {}), **fields,
    }
    generator = row.get("generator") if isinstance(row.get("generator"), dict) else {}
    accepted_value = row.get("accepted_count")
    if accepted_value is None:
        accepted_value = row.get("accepted_candidate_count")
    if accepted_value is None and isinstance(row.get("candidate_artifacts"), list):
        accepted_value = sum(
            1 for item in row["candidate_artifacts"] if isinstance(item, dict)
        )
    # Tool adapters write this field to JSON, so a malformed value must turn
    # into an auditable empty/partial row instead of aborting finalization.
    # ``bool`` is rejected explicitly because it is an ``int`` subclass.
    candidate_artifacts = row.get("candidate_artifacts")
    valid_artifact_count = (
        sum(1 for item in candidate_artifacts if isinstance(item, dict))
        if isinstance(candidate_artifacts, list) else 0
    )
    # A non-empty artifact list is the persisted candidate set.  Clamp an
    # adapter's declared count to its valid rows so malformed entries cannot
    # inflate generation or queue denominators.  Empty lists remain compatible
    # with adapters that only persist aggregate counts.
    fallback_accepted = valid_artifact_count
    accepted = _nonnegative_index(accepted_value, default=None)
    if accepted is None:
        accepted = fallback_accepted
    elif isinstance(candidate_artifacts, list) and candidate_artifacts:
        accepted = min(accepted, valid_artifact_count)
    mcmc = row.get("mcmc", generator.get("mcmc"))
    if isinstance(mcmc, bool):
        row.setdefault("search_mode", "mcmc" if mcmc else "direct")
    row.setdefault("steps", generator.get("steps"))
    row.setdefault("mh_attempts", generator.get("mh_attempts"))
    row.setdefault("accepted", generator.get("accepted"))
    row.setdefault("rejected", generator.get("rejected", generator.get("mh_rejected")))
    row.setdefault("request_count", None)
    attempt_count = row.get("attempt_count")
    if attempt_count is None:
        for value in (
            row.get("cases_attempted"), row.get("candidate_count_tested"),
            generator.get("cases_attempted"), generator.get("attempt_count"),
            generator.get("candidate_count_tested"),
        ):
            if value is not None:
                attempt_count = value
                break
    row["attempt_count"] = attempt_count
    source_count = row.get("source_count")
    if source_count is None:
        for value in (
            row.get("source_built_count"), row.get("source_generated"),
            generator.get("sources_generated"), generator.get("variant_source_count"),
        ):
            if value is not None:
                source_count = value
                break
    row["source_count"] = source_count
    pair_built_count = row.get("pair_built_count")
    if pair_built_count is None and isinstance(row.get("candidate_artifacts"), list):
        pair_built_count = sum(
            1 for item in row["candidate_artifacts"]
            if isinstance(item, dict) and (item.get("elf") or item.get("linux_elf"))
        )
    row["pair_built_count"] = pair_built_count
    artifacts = row.get("candidate_artifacts")
    if isinstance(artifacts, list):
        bare_count = sum(1 for item in artifacts if isinstance(item, dict) and item.get("elf"))
        linux_count = sum(1 for item in artifacts if isinstance(item, dict) and item.get("linux_elf"))
        artifact_count = sum(
            1 for item in artifacts
            if isinstance(item, dict) and (item.get("elf") or item.get("linux_elf"))
        )
        row.update(
            bare_built_count=bare_count, linux_built_count=linux_count,
            artifact_built_count=artifact_count, pair_built_count=artifact_count,
            bare_built=bare_count > 0, linux_built=linux_count > 0,
            pair_built=artifact_count > 0, pair_ready=artifact_count > 0,
        )
    else:
        row.setdefault("bare_built_count", int(bool(row.get("elf"))))
        row.setdefault("linux_built_count", int(bool(row.get("linux_elf"))))
        row.setdefault("bare_built", bool(row.get("elf") or row.get("built")))
        row.setdefault("linux_built", bool(row.get("linux_elf")))
        artifact_built = bool(row.get("elf") or row.get("linux_elf") or row.get("built"))
        row.setdefault("artifact_built_count", int(artifact_built))
        row.setdefault("pair_built_count", row["artifact_built_count"])
        row.setdefault("pair_built", artifact_built)
        row.setdefault("pair_ready", artifact_built)
    row.setdefault("raw_candidate_count", row.get("candidate_count") or 0)
    # These are normalized facts, not optional defaults.  Keeping a malformed
    # adapter value here would make downstream queue accounting disagree with
    # the candidate artifacts we just counted.
    row["unique_candidate_count"] = accepted
    row["accepted_count"] = accepted
    row["candidate_enqueued_count"] = accepted
    row.setdefault("target_queue_entry_count", None)
    row.setdefault("candidate_status", "usable" if accepted else "empty")
    if status == "gap" and accepted:
        row["status"] = "partial"
    row.setdefault("generation_partial", row.get("status") == "partial")
    if row.get("generation_partial"):
        row["generation_status"] = "partial"
    else:
        row.setdefault("generation_status", row.get("status"))
    if row.get("generation_partial") and not row.get("generation_partial_reason"):
        row["generation_partial_reason"] = (
            row.get("reason") or generator.get("reason") or "generation-incomplete"
        )
    return _refresh_generation_metrics(row)


def _queue_accounting(
    queue_entries: dict[str, list[Any]], events: list[dict[str, Any]],
) -> dict[str, Any]:
    """计算唯一的 queue/event 账本。"""
    def identity_part(value: object) -> str | None:
        # A damaged persisted row may put a list/object in a key field. Keep
        # the key hashable so accounting can report an invalid key instead of
        # aborting finalization; the ``None`` marker makes the key fail the
        # validity gate below.
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float)):
            text = str(value)
            return text if text else None
        return None

    def key(target_id: object, item: dict[str, Any], *, event: bool = False) -> tuple[Any, ...]:
        candidate = item if event else item.get("candidate", {})
        candidate = candidate if isinstance(candidate, dict) else {}
        return (
            identity_part(target_id), identity_part(item.get("method")),
            identity_part(item.get("route")), identity_part(item.get("lane")),
            _nonnegative_index(candidate.get("stream_index" if event else "index")),
        )

    queue_items = []
    invalid_queue_entry_count = 0
    if not isinstance(queue_entries, dict):
        queue_entries = {}
        invalid_queue_entry_count = 1
    for target_id, entries in queue_entries.items():
        if not isinstance(entries, (list, tuple)):
            invalid_queue_entry_count += 1
            continue
        for entry in entries:
            if isinstance(entry, dict):
                queue_items.append((target_id, entry))
            else:
                invalid_queue_entry_count += 1
    queue_keys = {key(target_id, entry) for target_id, entry in queue_items}

    def valid_key(item: tuple[Any, ...]) -> bool:
        return (
            item[0] is not None
            and all(item[index] is not None for index in (1, 2, 3))
            and isinstance(item[4], int) and item[4] >= 0
        )

    invalid_queue_key_count = sum(not valid_key(item) for item in queue_keys)
    event_values = events if isinstance(events, (list, tuple)) else ()
    valid_events = [event for event in event_values if isinstance(event, dict)]
    invalid_event_count = len(event_values) - len(valid_events)
    if not isinstance(events, (list, tuple)):
        invalid_event_count += 1
    execution_events = [
        event for event in valid_events
        if event.get("reason_code") != "generation-produced-no-candidate"
    ]
    event_keys = {key(event.get("target"), event, event=True) for event in execution_events}
    invalid_event_key_count = sum(not valid_key(item) for item in event_keys)
    pending_events = [event for event in execution_events if _event_is_pending(event)]
    dispatched_events = [
        event for event in execution_events
        if not _event_is_pending(event) or event.get("target_attempted") is True
    ]
    dispatched_keys = {key(event.get("target"), event, event=True) for event in dispatched_events}
    attempted_events = [event for event in dispatched_events if event.get("target_attempted") is True]
    attempted_keys = {key(event.get("target"), event, event=True) for event in attempted_events}
    error_events = [event for event in dispatched_events if event.get("target_attempted") is not True]
    right_censored_keys = {
        key(event.get("target"), event, event=True)
        for event in pending_events if event.get("target_attempted") is True
    }
    pending_keys = queue_keys - dispatched_keys
    queue_accounting_complete = (
        invalid_queue_entry_count == 0 and invalid_queue_key_count == 0
        and invalid_event_count == 0 and invalid_event_key_count == 0
        and len(queue_items) == len(queue_keys)
        and len(execution_events) == len(event_keys)
        and queue_keys == event_keys
    )
    queue_drained = queue_accounting_complete and queue_keys == dispatched_keys
    return {
        "_queue_keys": queue_keys,
        "_dispatched_keys": dispatched_keys, "_attempted_keys": attempted_keys,
        "_pending_keys": pending_keys,
        "_right_censored_keys": right_censored_keys,
        "execution_events": execution_events, "dispatched_events": dispatched_events,
        "attempted_events": attempted_events, "error_events": error_events,
        "pending_events": pending_events,
        "invalid_queue_entry_count": invalid_queue_entry_count,
        "invalid_event_count": invalid_event_count,
        "queue_accounting_complete": queue_accounting_complete,
        "queue_drained": queue_drained, "queued_count": len(queue_keys),
        "event_count": len(execution_events), "dispatched_count": len(dispatched_keys),
        "processed_count": len(attempted_keys), "unattempted_error_count": len(error_events),
        "pending_count": len(pending_keys),
        "right_censored_count": len(right_censored_keys),
    }


def _target_rates(
    *, queued: int, attempted: int, observation_recorded: int,
    pending: int, timeout: int,
) -> dict[str, dict[str, Any]]:
    return {
        "attempt_rate": {"numerator": attempted, "denominator": queued,
                          "value": _metric_ratio(attempted, queued, bounded=True)},
        "observation_rate": {"numerator": observation_recorded, "denominator": attempted,
                              "value": _metric_ratio(observation_recorded, attempted, bounded=True)},
        "pending_rate": {"numerator": pending, "denominator": queued,
                          "value": _metric_ratio(pending, queued, bounded=True)},
        "timeout_rate": {"numerator": timeout, "denominator": attempted,
                          "value": _metric_ratio(timeout, attempted, bounded=True)},
    }


def _load_partial_queue_entries(
    path: Path, *, run_id: str, method_filter: str,
    expected_target_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """Load the durable queue snapshot used by interrupted-run recovery."""
    try:
        snapshot = load_json(path)
    except (OSError, TypeError, ValueError):
        return {}, "partial-queue-snapshot-invalid"
    if snapshot.get("schema_version") != "rq1-generation-queue-partial-v1":
        return {}, "partial-queue-snapshot-schema"
    if snapshot.get("run_id") not in (None, run_id):
        return {}, "partial-queue-snapshot-run-id"
    if snapshot.get("method_filter") != method_filter:
        return {}, "partial-queue-snapshot-method"
    raw_queues = snapshot.get("queue_entries")
    if not isinstance(raw_queues, dict):
        return {}, "partial-queue-entry-identity-unavailable"
    if {str(target_id) for target_id in raw_queues} != expected_target_ids:
        return {}, "partial-queue-target-shape"
    queue_metadata = snapshot.get("queues")
    if not isinstance(queue_metadata, dict):
        return {}, "partial-queue-metadata-missing"
    queues: dict[str, list[dict[str, Any]]] = {}
    for target_id in sorted(expected_target_ids):
        entries = raw_queues.get(target_id)
        metadata = queue_metadata.get(target_id)
        if not isinstance(entries, list) or not isinstance(metadata, dict):
            return {}, f"partial-queue-entries-invalid:{target_id}"
        if metadata.get("count") != len(entries):
            return {}, f"partial-queue-count-mismatch:{target_id}"
        seen: set[tuple[Any, ...]] = set()
        checked: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return {}, f"partial-queue-entry-invalid:{target_id}"
            candidate = entry.get("candidate")
            if (
                entry.get("method") != method_filter
                or not isinstance(entry.get("route"), str)
                or not isinstance(entry.get("lane"), str)
                or not isinstance(candidate, dict)
            ):
                return {}, f"partial-queue-entry-identity:{target_id}"
            candidate_index = _nonnegative_index(candidate.get("index"))
            if candidate_index is None:
                return {}, f"partial-queue-entry-index:{target_id}"
            key = (entry.get("method"), entry.get("route"), entry.get("lane"), candidate_index)
            if key in seen:
                return {}, f"partial-queue-entry-duplicate:{target_id}"
            seen.add(key)
            checked.append(entry)
        queues[target_id] = checked
    return queues, None


def _partial_queue_counts(
    path: Path, *, run_id: str, method_filter: str,
    expected_target_ids: set[str],
) -> tuple[dict[str, int], str | None]:
    """读取没有 entry identity 时仍可相信的 queue 计数快照。

    老版本或被 SIGKILL 截断的 checkpoint 可能只落盘了 ``queues`` 的
    count，而没有 ``queue_entries``。这不能恢复 queue/event 一一对应关系，
    但也不能把已知的生成数量投影成 ``queued_count=0``。
    """
    try:
        snapshot = load_json(path)
    except (OSError, TypeError, ValueError):
        return {}, "partial-queue-snapshot-invalid"
    if snapshot.get("schema_version") != "rq1-generation-queue-partial-v1":
        return {}, "partial-queue-snapshot-schema"
    if snapshot.get("run_id") not in (None, run_id):
        return {}, "partial-queue-snapshot-run-id"
    if snapshot.get("method_filter") != method_filter:
        return {}, "partial-queue-snapshot-method"
    queues = snapshot.get("queues")
    if not isinstance(queues, dict) or set(map(str, queues)) != expected_target_ids:
        return {}, "partial-queue-target-shape"
    counts: dict[str, int] = {}
    for target_id in sorted(expected_target_ids):
        row = queues.get(target_id)
        if not isinstance(row, dict) or type(row.get("count")) is not int or row["count"] < 0:
            return {}, f"partial-queue-count-invalid:{target_id}"
        counts[target_id] = row["count"]
    return counts, None

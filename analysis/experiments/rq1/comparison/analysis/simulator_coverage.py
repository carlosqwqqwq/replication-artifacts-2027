"""统一计算 guest 和模拟器源码覆盖率；原始事件仅作诊断记录。"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
import os
import posixpath
from pathlib import Path
import re
import struct
import threading
from typing import Any, Iterable

try:
    from analysis.elf_features import (
        DECODER_VERSION, DEFAULT_PRIVILEGE_MODES, behavior_signature,
        metric_registry_for_profile,
    )
    from analysis.rv_instruction_coverage import (
        EXPERIMENT_COVERAGE_PROFILE,
        EXPERIMENT_PRIVILEGE_MODE,
        METRICS as RV_INSTRUCTION_METRICS,
        OPCODE_CATALOG_ID,
        OPCODE_CATALOG_REGISTRY_SHA256,
        aggregate_opcode_catalog_summary,
        aggregate_summary as aggregate_rv_instruction_summary,
        attach_case_metrics as attach_rv_instruction_metrics,
    )
    from comparison_observations import _record_is_pending
    from framework.spec_definedness import enabled_extensions
    from framework.execution_identity import is_allowlisted_patched_identity
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.elf_features import (
        DECODER_VERSION, DEFAULT_PRIVILEGE_MODES, behavior_signature,
        metric_registry_for_profile,
    )
    from analysis.rv_instruction_coverage import (
        EXPERIMENT_COVERAGE_PROFILE,
        EXPERIMENT_PRIVILEGE_MODE,
        METRICS as RV_INSTRUCTION_METRICS,
        OPCODE_CATALOG_ID,
        OPCODE_CATALOG_REGISTRY_SHA256,
        aggregate_opcode_catalog_summary,
        aggregate_summary as aggregate_rv_instruction_summary,
        attach_case_metrics as attach_rv_instruction_metrics,
    )
    from comparison_observations import _record_is_pending
    from framework.spec_definedness import enabled_extensions
    from framework.execution_identity import is_allowlisted_patched_identity

SCHEMA = "rq1-simulator-coverage-v4"
BATCH_SCHEMA = "rq1-simulator-coverage-batch-v6"
SOURCE_SCHEMA = "rq1-simulator-source-coverage-v2"
GUEST_METRICS = ("PCov", "ICov-encoding", "ICov-type", "CCov")
# v2 adds mandatory backend/adapter, ISA/privilege and catalog identity
# validation. Older sidecars are deliberately re-scanned rather than reused.
PC_SIDECAR_SCHEMA = "rq1-simulator-pc-sidecar-v2"


def _trace_io_slots() -> int:
    try:
        value = int(os.environ.get("RQ1_TRACE_IO_SLOTS", "1"))
    except (TypeError, ValueError):
        value = 1
    return max(1, min(4, value))


# ponytail: one shared HDD slot is the safe default; raise it only after an
# A/B measurement proves that the trace store is not the bottleneck.
_TRACE_IO_GATE = threading.BoundedSemaphore(_trace_io_slots())
# The existing native prewarm helper consumes this environment variable. Keep
# its default aligned with the shared source-side trace gate; an explicit
# operator value still wins.
os.environ.setdefault("RQ1_TRACE_SCAN_JOBS", str(_trace_io_slots()))
# Keep the legacy simulator summary and the three formal RV metrics on one
# profile string.  The formal privilege mode is owned by rv_instruction_coverage.
EXPERIMENT_UNION_PROFILE = EXPERIMENT_COVERAGE_PROFILE
FULL_EXPERIMENT_UNION_PROFILE = EXPERIMENT_COVERAGE_PROFILE

ADAPTERS = {
    "T-QEMU": {"backend": "qemu-riscv64", "trace": "qemu-instruction-pc",
                "parser": "qemu-execlog"},
    "T-LRSV-INT": {"backend": "libriscv-interpreter", "trace": "libriscv-debug-pc"},
    "T-LRSV-TRANS": {"backend": "libriscv-translated", "trace": "libriscv-trace"},
    "T-UNICORN": {"backend": "unicorn-riscv64", "trace": "unicorn-hooks"},
    "T-RENODE": {"backend": "renode-riscv64", "trace": "renode-unique-pc",
                  "parser": "renode-unique-pc"},
    "T-RAX": {"backend": "rax-riscv64",
              "trace": "rax-instruction-pc", "parser": "rax-instruction-pc"},
    "T-RVVM": {"backend": "rvvm-riscv64", "trace": "rvvm-instruction-pc",
                "parser": "rvvm-instruction-pc"},
}


def _metric(covered: int, eligible: int, *, status: str | None = None,
            reason: str | None = None) -> dict[str, Any]:
    status = status or ("observed" if eligible else "NA")
    if covered > eligible and status in {"observed", "partial"}:
        # 这是 profile/口径不一致的防御性保护；正常的 source_scope_exclude
        # 会同时移除 covered 和 eligible，不应产生超过 100% 的比例。
        status = "gap"
        reason = reason or "covered-exceeds-eligible-after-exclusion"
    value = round(covered / eligible, 6) if eligible and status in {"observed", "partial"} else None
    result = {"covered": covered, "eligible": eligible, "value": value, "status": status}
    if reason:
        result["reason"] = reason
    return result


def _metric_value(covered: Iterable[str], eligible: Iterable[str], *,
                  reason: str | None = None) -> dict[str, Any]:
    return _metric(len(set(covered)), len(set(eligible)), reason=reason)


def _nonnegative_count(value: object, *, missing: int | None = 0) -> tuple[int | None, bool]:
    """Parse a JSON counter without accepting booleans or negative values."""
    if value is None:
        return missing, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None, False
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, False
    return (parsed, True) if parsed >= 0 else (None, False)


def _hit_count(value: object) -> int:
    """Parse an LCOV hit count without truncating malformed values."""
    if isinstance(value, bool) or value is None:
        raise ValueError("coverage count must be a non-negative integer")
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        parsed = int(value.strip(), 10)
    else:
        try:
            parsed_float = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coverage count must be a non-negative integer") from exc
        if not math.isfinite(parsed_float) or parsed_float < 0 or not parsed_float.is_integer():
            raise ValueError("coverage count must be a non-negative integer")
        parsed = int(parsed_float)
    if parsed < 0:
        raise ValueError("coverage count must be a non-negative integer")
    return parsed


def _pc_value(value: object) -> int | None:
    """Normalize a trace PC and reject booleans/out-of-range integers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None
    try:
        parsed = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 0 <= parsed < 1 << 64 else None


_PCOV_UNIT_RE = re.compile(r"([0-9a-f]{64})@0x(0|[1-9a-f][0-9a-f]*)\Z")


def _pcov_unit_digest(units: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(units)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_unit_digest(units: Iterable[tuple[str, str]]) -> str:
    """Digest normalized SimSrcCov units, including source path and unit key."""
    payload = json.dumps(
        sorted({f"{source}\x00{unit}" for source, unit in units}),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_summary_digest(records: Iterable[dict[str, Any]], field: str) -> str:
    """Digest authoritative per-source LCOV summary counts."""
    units = {
        (str(record["source_key"]), f"{field}:{int(record[field])}")
        for record in records
        if record.get(field) is not None
    }
    return _source_unit_digest(units)


def _experiment_union_profile(profiles: Iterable[str]) -> str:
    """Use the full RQ1 ISA universe when a run exercises A/F/D."""
    for profile_id in profiles:
        profile = str(profile_id).split("|", 1)[0].strip().lower()
        if {"a", "f", "d"} & set(enabled_extensions(profile)):
            return FULL_EXPERIMENT_UNION_PROFILE
    return EXPERIMENT_UNION_PROFILE


def _summarize_pcov_run(
    covered_units: Iterable[str], eligible_units: Iterable[str], status: str,
    *, identity_missing_records: int = 0,
    static_map_missing_records: int = 0,
    invalid_unit_records: int = 0,
) -> dict[str, Any]:
    covered = set(covered_units)
    eligible = set(eligible_units)
    reason = None
    invalid = False
    for unit in covered | eligible:
        match = _PCOV_UNIT_RE.fullmatch(unit) if isinstance(unit, str) else None
        if match is None or int(match.group(2), 16) >= 1 << 64:
            invalid = True
            break
    if invalid or not covered <= eligible:
        status = "gap"
        invalid_unit_records += 1
        reason = "pcov-static-unit-set-invalid"
    elif status not in {"observed", "partial", "gap", "NA"}:
        status = "gap"
        invalid_unit_records += 1
        reason = "pcov-status-invalid"
    elif invalid_unit_records:
        status = "gap"
        reason = "pcov-static-unit-set-invalid"
    elif identity_missing_records:
        status = "gap"
        reason = "pcov-artifact-identity-missing"
    elif static_map_missing_records:
        status = "gap"
        reason = "pcov-static-map-missing"
    elif not eligible and status == "observed":
        status = "gap"
        static_map_missing_records += 1
        reason = "pcov-static-map-missing"
    elif status == "partial":
        reason = "batch-observation-incomplete"
    elif status == "gap":
        reason = "batch-observation-gap"
    elif status == "NA":
        reason = "no-observed-static-instruction-map"
    if identity_missing_records or static_map_missing_records or invalid_unit_records:
        status = "gap"
    metric = _metric(len(covered), len(eligible), status=status, reason=reason)
    if status != "observed":
        metric["value"] = None
    artifacts = {
        match.group(1) for unit in eligible
        if isinstance(unit, str) and (match := _PCOV_UNIT_RE.fullmatch(unit))
    }
    covered_artifacts = {
        match.group(1) for unit in covered
        if isinstance(unit, str) and (match := _PCOV_UNIT_RE.fullmatch(unit))
    }
    return {
        **metric,
        "aggregation": "run-union-elf-pc",
        "covered_units": sorted(covered),
        "eligible_units": sorted(eligible),
        "covered_units_sha256": _pcov_unit_digest(covered),
        "eligible_units_sha256": _pcov_unit_digest(eligible),
        "artifact_count": len(artifacts),
        "artifacts_with_hits": len(covered_artifacts),
        "identity_missing_record_count": identity_missing_records,
        "static_map_missing_record_count": static_map_missing_records,
        "invalid_unit_record_count": invalid_unit_records,
    }


def summarize_experiment_union(
    method: str | None, target_rows: Iterable[dict[str, Any]], *,
    profile_ids: Iterable[str] = (), case_artifact_pairs: int | None = None,
) -> dict[str, Any]:
    """Union all case and target evidence for one method's complete run."""
    rows = [row for row in (target_rows or ()) if isinstance(row, dict)]
    profiles: set[str] = set()
    invalid_profiles = False
    for profile_id in profile_ids or ():
        if isinstance(profile_id, str) and profile_id:
            profiles.add(profile_id)
        else:
            invalid_profiles = True
    declared_registry_shas: dict[str, set[str]] = {}
    source_registries: dict[str, dict[str, Any]] = {}
    modes: set[str] = set()
    for row in rows:
        basis = row.get("coverage_basis")
        if not isinstance(basis, dict):
            invalid_profiles = True
            continue
        profile_id = basis.get("coverage_profile_id")
        registry_sha = basis.get("coverage_registry_sha256")
        if not isinstance(profile_id, str) or not profile_id:
            invalid_profiles = True
            continue
        profiles.add(profile_id)
        if isinstance(registry_sha, str) and registry_sha:
            declared_registry_shas.setdefault(profile_id, set()).add(registry_sha)
        else:
            invalid_profiles = True
    for profile_id in profiles:
        parts = profile_id.split("|")
        if len(parts) != 3 or parts[1] not in DEFAULT_PRIVILEGE_MODES:
            invalid_profiles = True
            continue
        try:
            source_registry = metric_registry_for_profile(parts[0], parts[1])
        except ValueError:
            invalid_profiles = True
            continue
        if source_registry["metric_profile_id"] != profile_id:
            invalid_profiles = True
            continue
        source_registries[profile_id] = source_registry
        if declared_registry_shas.get(profile_id) != {source_registry["registry_sha256"]}:
            invalid_profiles = True
        modes.add(parts[1])
    # The legacy guest union must use the same fixed user-mode denominator as
    # the formal RV instruction metrics.  Observed modes remain diagnostics;
    # they do not change the comparison universe.
    privilege_mode = EXPERIMENT_PRIVILEGE_MODE
    if not modes:
        invalid_profiles = True
    denominator_profile = _experiment_union_profile(profiles)
    registry = metric_registry_for_profile(denominator_profile, privilege_mode)

    pcov_eligible: set[str] = set()
    pcov_covered: set[str] = set()
    pcov_statuses: list[str] = []
    pcov_identity_missing = pcov_static_missing = pcov_invalid = 0
    pcov_values_available = True
    static_maps: dict[str, set[str]] = {}
    static_map_conflict = False
    fixed_covered = {name: set() for name in GUEST_METRICS if name != "PCov"}
    fixed_statuses = {name: [] for name in fixed_covered}
    fixed_invalid = {name: False for name in fixed_covered}
    fixed_values_available = {name: True for name in fixed_covered}
    target_scopes = []
    invalid_target_counts = False
    for row in rows:
        target_scopes.append({
            name: row.get(name) for name in
            ("target", "stratum", "lane", "route", "target_identity_id")
        })
        guest = row.get("guest")
        guest = guest if isinstance(guest, dict) else {}
        pcov = guest.get("PCov")
        if not isinstance(pcov, dict):
            pcov_statuses.append("gap")
            pcov_invalid += 1
        else:
            pcov_status = pcov.get("status")
            pcov_status = (
                pcov_status if pcov_status in {"observed", "partial", "gap", "NA"}
                else "gap"
            )
            pcov_statuses.append(pcov_status)
            if pcov_status != "observed" and pcov.get("value") is None:
                pcov_values_available = False
            for field, accumulator in (
                ("identity_missing_record_count", "identity"),
                ("static_map_missing_record_count", "static"),
                ("invalid_unit_record_count", "invalid"),
            ):
                count, count_valid = _nonnegative_count(pcov.get(field))
                if not count_valid:
                    pcov_invalid += 1
                    continue
                if accumulator == "identity":
                    pcov_identity_missing += int(count or 0)
                elif accumulator == "static":
                    pcov_static_missing += int(count or 0)
                else:
                    pcov_invalid += int(count or 0)
            eligible_count, eligible_valid = _nonnegative_count(
                pcov.get("eligible"), missing=None,
            )
            covered_count, covered_valid = _nonnegative_count(
                pcov.get("covered"), missing=None,
            )
            if (not eligible_valid or not covered_valid
                    or eligible_count is None or covered_count is None):
                eligible_count = covered_count = -1
                pcov_invalid += 1
            eligible_raw, covered_raw = pcov.get("eligible_units"), pcov.get("covered_units")
            valid = (
                pcov.get("aggregation") == "run-union-elf-pc"
                and isinstance(eligible_raw, list) and isinstance(covered_raw, list)
                and all(isinstance(unit, str) for unit in (*eligible_raw, *covered_raw))
                and len(set(eligible_raw)) == len(eligible_raw)
                and len(set(covered_raw)) == len(covered_raw)
                and eligible_count >= 0 and covered_count >= 0
                and eligible_count == len(eligible_raw)
                and covered_count == len(covered_raw)
                and set(covered_raw) <= set(eligible_raw)
                and pcov.get("eligible_units_sha256") == _pcov_unit_digest(eligible_raw)
                and pcov.get("covered_units_sha256") == _pcov_unit_digest(covered_raw)
            )
            if pcov_status in {"observed", "partial"} and pcov.get("value") is not None:
                try:
                    valid = valid and eligible_count > 0 \
                        and math.isclose(
                            float(pcov["value"]), covered_count / eligible_count,
                            rel_tol=0.0, abs_tol=1.1e-6,
                        )
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    valid = False
            elif pcov_status == "observed":
                valid = False
            elif pcov.get("value") is not None:
                valid = False
            if not valid:
                pcov_invalid += 1
                pcov_statuses[-1] = "gap"
            else:
                eligible_set, covered_set = set(eligible_raw), set(covered_raw)
                pcov_eligible.update(eligible_set)
                pcov_covered.update(covered_set)
                per_artifact: dict[str, set[str]] = {}
                for unit in eligible_set:
                    match = _PCOV_UNIT_RE.fullmatch(unit)
                    if match is None:
                        continue
                    per_artifact.setdefault(match.group(1), set()).add(unit)
                for artifact, units in per_artifact.items():
                    previous = static_maps.get(artifact)
                    if previous is not None and previous != units:
                        static_map_conflict = True
                    else:
                        static_maps[artifact] = units
        for name in fixed_covered:
            metric = guest.get(name)
            if not isinstance(metric, dict):
                fixed_statuses[name].append("gap")
                fixed_invalid[name] = True
                continue
            metric_status = metric.get("status")
            metric_status = (
                metric_status if metric_status in {"observed", "partial", "gap", "NA"}
                else "gap"
            )
            fixed_statuses[name].append(metric_status)
            if metric_status != "observed" and metric.get("value") is None:
                fixed_values_available[name] = False
            covered_raw = metric.get("covered_units")
            eligible_raw = metric.get("eligible_units")
            covered_count, covered_valid = _nonnegative_count(
                metric.get("covered"), missing=None,
            )
            eligible_count, eligible_valid = _nonnegative_count(
                metric.get("eligible"), missing=None,
            )
            if not covered_valid or not eligible_valid:
                covered_count = eligible_count = -1
            valid = (
                isinstance(covered_raw, list) and isinstance(eligible_raw, list)
                and all(isinstance(unit, str) for unit in (*covered_raw, *eligible_raw))
                and len(set(covered_raw)) == len(covered_raw)
                and len(set(eligible_raw)) == len(eligible_raw)
                and covered_count >= 0 and eligible_count >= 0
                and covered_count == len(covered_raw)
                and eligible_count == len(eligible_raw)
                and set(covered_raw) <= set(eligible_raw)
                and metric.get("coverage_profile_id") == (
                    row.get("coverage_basis") or {}
                ).get("coverage_profile_id")
                and metric.get("registry_sha256") == _pcov_unit_digest(eligible_raw)
            )
            source_profile_id = (row.get("coverage_basis") or {}).get(
                "coverage_profile_id"
            )
            source_registry = (
                source_registries.get(source_profile_id)
                if isinstance(source_profile_id, str) else None
            )
            source_units = (
                set(source_registry["metrics"][name])
                if source_registry is not None else set()
            )
            valid = valid and source_registry is not None and set(eligible_raw) == source_units
            if metric_status in {"observed", "partial"} and metric.get("value") is not None:
                try:
                    valid = valid and eligible_count > 0 \
                        and math.isclose(
                            float(metric["value"]), covered_count / eligible_count,
                            rel_tol=0.0, abs_tol=1.1e-6,
                        )
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    valid = False
            elif metric_status == "observed":
                valid = False
            elif metric.get("value") is not None:
                valid = False
            if not valid:
                fixed_invalid[name] = True
                fixed_statuses[name][-1] = "gap"
            else:
                # A native lane may declare a valid superset profile. Only
                # hits in the frozen RQ1 universe enter this union.
                fixed_covered[name].update(
                    set(covered_raw) & set(registry["metrics"][name])
                )

    def merged_status(statuses: list[str], invalid: bool = False) -> str:
        allowed = {"observed", "partial", "gap", "NA"}
        if (invalid or not statuses
                or any(item not in allowed or item == "gap" for item in statuses)):
            return "gap"
        if all(item == "NA" for item in statuses):
            return "NA"
        if any(item in {"partial", "NA"} for item in statuses):
            return "partial"
        return "observed"

    if static_map_conflict:
        pcov_invalid += 1
    target_identity_ids: dict[str, set[str]] = {}
    for scope in target_scopes:
        target = str(scope.get("target") or "")
        identity = scope.get("target_identity_id")
        if target and isinstance(identity, str) and identity:
            target_identity_ids.setdefault(target, set()).add(identity)
    target_identity_conflict = any(
        len(identities) > 1 for identities in target_identity_ids.values()
    )

    pcov_status = merged_status(
        pcov_statuses, bool(pcov_identity_missing or pcov_static_missing
                            or pcov_invalid or static_map_conflict
                            or target_identity_conflict),
    )
    pcov_metric = _summarize_pcov_run(
        pcov_covered, pcov_eligible, pcov_status,
        identity_missing_records=pcov_identity_missing,
        static_map_missing_records=pcov_static_missing,
        invalid_unit_records=pcov_invalid,
    )
    if pcov_status == "partial" and pcov_values_available and pcov_eligible:
        pcov_metric["value"] = round(len(pcov_covered) / len(pcov_eligible), 6)
    if static_map_conflict:
        pcov_metric.update(status="gap", value=None, reason="pcov-static-map-conflict")
    elif target_identity_conflict:
        pcov_metric.update(status="gap", value=None, reason="target-identity-mismatch")
    pcov_metric.update(
        coverage_profile_id=registry["metric_profile_id"],
        coverage_registry_sha256=registry["registry_sha256"],
    )
    guest_metrics: dict[str, Any] = {"PCov": pcov_metric}
    for name, statuses in fixed_statuses.items():
        expected = set(registry["metrics"][name])
        covered = fixed_covered[name]
        metric_status = merged_status(
            statuses,
            fixed_invalid[name] or invalid_profiles
            or target_identity_conflict,
        )
        metric = _metric_value(covered & expected, expected)
        metric.update(
            status=metric_status,
            value=round(len(covered & expected) / len(expected), 6)
            if expected and (
                metric_status == "observed"
                or metric_status == "partial" and fixed_values_available[name]
            ) else None,
            covered_units=sorted(covered & expected),
            eligible_units=sorted(expected),
            coverage_profile_id=registry["metric_profile_id"],
            registry_sha256=_pcov_unit_digest(expected),
            denominator_profile=denominator_profile,
        )
        if metric_status != "observed":
            metric["reason"] = (
                "target-identity-mismatch" if target_identity_conflict else
                "coverage-profile-metadata-invalid" if invalid_profiles else
                f"{name}-experiment-observation-incomplete"
            )
        guest_metrics[name] = metric
    scopes_by_key = {}
    for scope in target_scopes:
        key = tuple(str(scope.get(name) or "") for name in (
            "target", "stratum", "lane", "route", "target_identity_id",
        ))
        scopes_by_key[key] = scope
    target_scopes = [scopes_by_key[key] for key in sorted(scopes_by_key)]
    row_statuses = [str(row.get("status") or "gap") for row in rows]
    all_statuses = row_statuses + [
        str(metric.get("status")) for metric in guest_metrics.values()
        if str(metric.get("status")) != "NA"
    ]
    target_case_records = 0
    for row in rows:
        count, count_valid = _nonnegative_count(row.get("cases"))
        if not count_valid:
            invalid_target_counts = True
        else:
            target_case_records += int(count or 0)
    if invalid_target_counts:
        experiment_status = "gap"
        experiment_reason = "experiment-count-invalid"
    else:
        experiment_status = merged_status(all_statuses, target_identity_conflict)
        experiment_reason = (
            "target-identity-mismatch" if target_identity_conflict else
            "coverage-profile-metadata-invalid" if invalid_profiles else
            "experiment-status-invalid" if any(item not in {
                "observed", "partial", "gap", "NA"
            } for item in all_statuses) else
            "experiment-observation-gap" if any(item == "gap" for item in all_statuses) else
            "experiment-observation-incomplete" if any(item in {"partial", "NA"}
                                                      for item in all_statuses) else
            None
        )
    return {
        "scope": "whole-method-run-all-cases-and-targets",
        "method": method,
        "status": experiment_status,
        "reason": experiment_reason,
        "targets": sorted({str(row.get("target")) for row in rows if row.get("target")}),
        "target_scopes": target_scopes,
        "target_group_count": len(rows),
        "target_case_records": target_case_records,
        "case_artifact_pairs": case_artifact_pairs,
        "profile_ids": sorted(profiles),
        "privilege_modes": sorted(modes),
        "isa_profile": denominator_profile,
        "coverage_basis": {
            "PCov": "run-union-elf-pc",
            "ICov-encoding": "fixed-profile-universe",
            "ICov-type": "fixed-profile-universe",
            "CCov": "fixed-profile-universe",
            "coverage_profile_id": registry["metric_profile_id"],
            "coverage_registry_sha256": registry["registry_sha256"],
        },
        "guest": guest_metrics,
    }


def summarize_source_experiment_union(
    method: str | None, target_rows: Iterable[dict[str, Any]], *,
    source_rows: Iterable[dict[str, Any]] = (),
    case_artifact_pairs: int | None = None,
    expected_target_scopes: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Summarize the active simulator-source coverage channel.

    The active RQ1 contract uses each target-run source profile as its
    denominator, keyed by ``(target, stratum)``. It has no guest instruction registry, privilege-mode
    universe, or decoded ELF PC set. Keeping this union separate from
    ``summarize_experiment_union`` prevents the historical PCov/ICov/CCov
    validator from turning a valid SimSrcCov result into a metadata gap.
    """
    rows = [row for row in (target_rows or ()) if isinstance(row, dict)]
    sources = [row for row in (source_rows or ()) if isinstance(row, dict)]

    def scope_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
        target = str(row.get("target") or "")
        stratum = str(row.get("stratum") or "")
        return (target, stratum) if target and stratum else None

    target_statuses: dict[tuple[str, str], list[str]] = {}
    invalid_target_scope = False
    for row in rows:
        key = scope_key(row)
        if key is None:
            invalid_target_scope = True
            continue
        value = str(row.get("status") or "gap")
        target_statuses.setdefault(key, []).append(
            value if value in {"observed", "partial", "gap", "NA"} else "gap"
        )
    row_scopes = set(target_statuses)
    declared_scopes = list(expected_target_scopes or ())
    normalized_expected = []
    invalid_expected_scope = False
    for scope in declared_scopes:
        if not isinstance(scope, (list, tuple)) or len(scope) != 2:
            invalid_expected_scope = True
            continue
        target, stratum = str(scope[0] or ""), str(scope[1] or "")
        if not target or not stratum:
            invalid_expected_scope = True
            continue
        normalized_expected.append((target, stratum))
    expected_scopes = (
        set(normalized_expected)
        if expected_target_scopes is not None else row_scopes
    )
    missing_target_scopes = expected_scopes - row_scopes
    extra_target_scopes = row_scopes - expected_scopes
    source_statuses: dict[tuple[str, str], str] = {}
    source_signatures: dict[tuple[str, str], str] = {}
    source_scope_conflicts: set[tuple[str, str]] = set()
    invalid_source_scope = False
    for row in sources:
        coverage = row.get("simulator_source_coverage", row)
        key = scope_key(row)
        if key is None:
            invalid_source_scope = True
            continue
        value = (
            str(coverage.get("status") or "gap")
            if isinstance(coverage, Mapping) else "gap"
        )
        value = value if value in {"observed", "partial", "gap", "NA"} else "gap"
        signature = json.dumps(
            coverage, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        ) if isinstance(coverage, Mapping) else "invalid-source-row"
        previous = source_statuses.get(key)
        if (previous is not None
                and (previous != value or source_signatures[key] != signature)):
            source_scope_conflicts.add(key)
        source_statuses[key] = (
            "gap" if key in source_scope_conflicts else value
        )
        source_signatures[key] = signature
    source_scopes = set(source_statuses)
    missing_source_scopes = expected_scopes - source_scopes
    extra_source_scopes = source_scopes - expected_scopes
    all_statuses = [
        *(
            status
            for key in expected_scopes
            for status in target_statuses.get(key, ())
        ),
        *(source_statuses[key] for key in expected_scopes
          if key in source_statuses),
    ]
    status = (
        "observed" if all_statuses and all(item == "observed" for item in all_statuses)
        else "gap" if any(item == "gap" for item in all_statuses)
        else "partial" if any(item in {"observed", "partial"} for item in all_statuses)
        else "NA"
    )
    scope_incomplete = bool(
        invalid_target_scope or invalid_expected_scope or invalid_source_scope
        or missing_target_scopes or extra_target_scopes
        or missing_source_scopes or extra_source_scopes or source_scope_conflicts
    )
    if scope_incomplete:
        status = "gap"
    reason = "simulator-source-coverage-scope-incomplete" if scope_incomplete else next(
        (
            str(value)
            for row in (*rows, *sources)
            for value in (row.get("reason"),)
            if isinstance(value, str) and value
        ),
        None,
    )
    target_scopes = [
        {
            name: row.get(name) for name in
            ("target", "stratum", "lane", "route", "target_identity_id")
        }
        for row in rows
    ]
    target_case_records = 0
    invalid_counts = False
    for row in rows:
        count, valid = _nonnegative_count(row.get("cases"), missing=None)
        if not valid or count is None:
            invalid_counts = True
        else:
            target_case_records += int(count)
    if invalid_counts:
        status = "gap"
        reason = reason or "experiment-count-invalid"
    elif rows and not sources:
        status = "gap"
        reason = reason or "simulator-source-coverage-missing"
    targets = sorted({str(row.get("target")) for row in rows if row.get("target")})
    return {
        "scope": "whole-method-run-all-cases-and-targets",
        "method": method,
        "status": status,
        "reason": reason,
        "targets": targets,
        "target_scopes": target_scopes,
        "target_group_count": len(rows),
        "target_case_records": target_case_records,
        "case_artifact_pairs": case_artifact_pairs,
        "source_scope_complete": not scope_incomplete,
        "missing_target_scopes": [
            {"target": target, "stratum": stratum}
            for target, stratum in sorted(missing_target_scopes)
        ],
        "extra_target_scopes": [
            {"target": target, "stratum": stratum}
            for target, stratum in sorted(extra_target_scopes)
        ],
        "missing_source_scopes": [
            {"target": target, "stratum": stratum}
            for target, stratum in sorted(missing_source_scopes)
        ],
        "extra_source_scopes": [
            {"target": target, "stratum": stratum}
            for target, stratum in sorted(extra_source_scopes)
        ],
        "conflicting_source_scopes": [
            {"target": target, "stratum": stratum}
            for target, stratum in sorted(source_scope_conflicts)
        ],
        "profile_ids": [],
        "privilege_modes": [],
        "isa_profile": None,
        "coverage_basis": {"SimSrcCov": "target-run-source-profile"},
        "guest": {},
        "source_targets": sources,
    }


def gate_experiment_unions(
    summary: dict[str, Any], status: str, reason: str, *,
    preserve_generation: bool = False,
    include_targets: bool = True,
) -> None:
    """Keep persisted ratios null after an evidence gap.

    A complete static ELF still proves ``GenCov`` when only its dynamic trace
    is incomplete.  Callers that have accounted for every queued ELF can set
    ``preserve_generation``; identity or missing-case gaps keep the default
    fail-closed behavior for all metrics.
    """
    if status not in {"partial", "gap"}:
        return
    metric_names = set(GUEST_METRICS) | set(RV_INSTRUCTION_METRICS)
    if preserve_generation:
        metric_names.discard("GenCov")

    def gate_metrics(metrics: object) -> tuple[bool, bool]:
        if not isinstance(metrics, dict):
            return False, False
        changed = has_gap = False
        for name in metric_names:
            metric = metrics.get(name)
            if not isinstance(metric, dict):
                continue
            if metric.get("status") == "gap":
                has_gap = True
                continue
            metric.update(status=status, value=None, reason=reason)
            changed = True
        return changed, has_gap

    container_names = ("targets", "experiment_unions") if include_targets else (
        "experiment_unions",
    )
    for container_name in container_names:
        for container in summary.get(container_name, ()):
            if not isinstance(container, dict):
                continue
            if container.get("status") != "gap" or status == "gap":
                container["status"] = status
            container["reason"] = reason
            guest = container.get("guest")
            gate_metrics(guest)
            if isinstance(guest, dict):
                gate_metrics(guest.get("metrics"))
            rv = container.get("rv_instruction_coverage")
            if isinstance(rv, dict):
                changed, has_gap = gate_metrics(rv.get("metrics"))
                if changed or has_gap:
                    rv["status"] = "gap" if has_gap else status
                    rv["reason"] = reason

def _resolve_output(out: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = out / path
    try:
        path = path.resolve()
        return path if path == out.resolve() or out.resolve() in path.parents else None
    except OSError:
        return None


def trace_is_hard_incomplete(row: Mapping[str, Any]) -> bool:
    observation = row.get("observation")
    observation = observation if isinstance(observation, Mapping) else {}
    return bool(
        row.get("trace_complete") is False
        or observation.get("trace_complete") is False
        or row.get("trace_cleanup_status") == "gap"
        or observation.get("trace_cleanup_status") == "gap"
    )


def _paths_from_row(
    out: Path, row: dict[str, Any], *names: str, skip_incomplete: bool = True,
) -> list[Path]:
    if skip_incomplete and trace_is_hard_incomplete(row):
        return []
    paths = []
    for name in names:
        path = _resolve_output(out, row.get(name))
        # Keep missing paths in the scan.  Filtering them here turns a missing
        # trace into an apparently unattempted run and hides the evidence gap.
        if path and path not in paths:
            paths.append(path)
    return paths


def _record_uses_patched_source(row: dict[str, Any]) -> bool:
    """Return whether an identity carries an unapproved source patch.

    Controlled, reproducibly attested simulator builds are valid experiment
    targets.  Sparse or malformed patch claims remain a coverage gap; treating
    every patch provenance as forbidden would reject the RAX/RVVM binaries
    selected by the current configuration and hide their real observations.
    """
    patched = {"patch-validated-build", "verified-git-archive+controlled-patch"}
    for identity in (row.get("target_identity"), row.get("identity")):
        if not isinstance(identity, dict):
            continue
        has_source_patch = bool(identity.get("patch_lineage")) or (
            identity.get("source_provenance") in patched
        )
        if has_source_patch and not is_allowlisted_patched_identity(identity):
            return True
    return False


def _artifact_digest_for_record(record: dict[str, Any], stratum: str) -> str | None:
    """Select one artifact identity and reject conflicting legacy aliases."""
    preferred = "linux_elf_sha256" if stratum == "linux-user" else "bare_elf_sha256"
    fields = [preferred, "artifact_sha256", "artifact_id"]
    fields.extend(("linux_elf_sha256",) if stratum == "linux-user"
                  else ("bare_elf_sha256", "elf_sha256"))
    values = [record.get(name) for name in fields if record.get(name) not in (None, "")]
    if not values or not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        for value in values
    ) or len({value.lower() for value in values if isinstance(value, str)}) != 1:
        return None
    normalized = {str(value).lower() for value in values}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _coverage_record_ineligible(record: dict[str, Any]) -> bool:
    """Reject pending or explicitly invalid events before using cached coverage."""
    if not isinstance(record, dict):
        return True
    if (
        "target_attempted" in record and record.get("target_attempted") is not True
    ) or (
        "artifact_identity_ok" in record
        and record.get("artifact_identity_ok") is not True
    ):
        return True
    return _record_is_pending(record)


PC_TRACE_PATTERNS = {
    "qemu-execlog": re.compile(
        rb'(?m)^\s*\d+,\s+0x([0-9a-fA-F]+),\s+'
        rb'0x[0-9a-fA-F]+,\s+"'
    ),
    "libriscv-translated": re.compile(
        rb"(?m)^f [^ \r\n]+\s+pc\s+0x([0-9a-fA-F]+)\s+instr\s+[0-9a-fA-F]+"),
    "libriscv-interpreter": re.compile(
        rb"(?m)^\s*\[0x([0-9a-fA-F]+)\]\s+[0-9a-fA-F]+"),
    "renode-unique-pc": re.compile(rb"(?m)^\s*0x([0-9a-fA-F]+)\s*$"),
    "rax-instruction-pc": re.compile(
        rb"(?m)^\s*INS\s+0x([0-9a-fA-F]+)\s+"
    ),
    "rvvm-instruction-pc": re.compile(
        rb"(?m)^\s*INS\s+0x([0-9a-fA-F]+)\s+"
    ),
}

QEMU_PLUGIN_RE = re.compile(r"^\s*\d+,\s+0x([0-9a-fA-F]+),\s+0x[0-9a-fA-F]+,")
DRCOV_BB_TABLE_RE = re.compile(rb"(?m)^BB Table:\s*([0-9]+)\s+bbs\r?\n")
LRSV_TRACE_RE = re.compile(r"\bpc\s+0x([0-9a-fA-F]+)\s+instr\s+([0-9a-fA-F]+)")
LRSV_INTERPRETER_RE = re.compile(r"^\s*\[0x([0-9a-fA-F]+)\]\s+[0-9a-fA-F]+")
SIM_EVENT_RE = re.compile(r"RQ1_SIM_EVENT:([A-Za-z0-9_.-]+)")
LRSV_INSTRUCTION_COUNT_RE = re.compile(r"Instructions executed:\s*([0-9]+)")
LRSV_BLOCKS_RE = re.compile(r"Emitted .*\b[0-9]+\s+blocks")
LRSV_EXCEPTION_RE = re.compile(r"Machine exception|Illegal operation|Misaligned|Unimplemented", re.I)
QEMU_EXCEPTION_RE = re.compile(r"uncaught target signal|unhandled CPU exception|QEMU internal SIG", re.I)
QEMU_SYSCALL_RE = re.compile(r"^\s*[0-9]+\s+[a-z_][a-z0-9_]*\(")
RENODE_INTERRUPT_RE = re.compile(r"interrupt|IRQ", re.I)
_RENODE_STORE_MNEMONICS = frozenset({
    "sb", "sh", "sw", "sd", "fsb", "fsh", "fsw", "fsd",
    "c.sb", "c.sh", "c.sw", "c.sd",
})
_RENODE_LOADER_MNEMONICS = frozenset({"add", "addi", "auipc", "lui", "c.addi"})
_RENODE_ZERO_STORE_RE = re.compile(
    r",\s*(?:0|0x0)\s*\(\s*([a-z][a-z0-9]*)\s*\)", re.I,
)


def _renode_terminal_pcs(feature: dict[str, Any] | None) -> set[int]:
    """Find static PCs that write the guest's nonzero tohost result.

    Renode's execution tracer keeps running briefly after the watchpoint in
    older scripts.  The static map lets the replay cut at the first observed
    terminal store, including candidates whose disassembly puts the tohost
    label on the preceding address-building instruction.
    """
    instructions = (
        feature.get("opcode_catalog_execution_instructions", feature.get("instructions"))
        if isinstance(feature, dict) else None
    )
    if not isinstance(instructions, list):
        return set()
    terminal_pcs: set[int] = set()
    for index, item in enumerate(instructions):
        if not isinstance(item, dict) or type(item.get("address")) is not int \
                or not 0 <= item.get("address") < 1 << 64:
            continue
        mnemonic = str(item.get("mnemonic", "")).lower()
        operands = str(item.get("operands", ""))
        if mnemonic in _RENODE_STORE_MNEMONICS and "<tohost>" in operands.lower():
            terminal_pcs.add(item["address"])
        if mnemonic not in _RENODE_LOADER_MNEMONICS or "<tohost>" not in operands.lower():
            continue
        register = operands.split(",", 1)[0].strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9]*", register):
            continue
        for candidate in instructions[index + 1:]:
            if not isinstance(candidate, dict) or type(candidate.get("address")) is not int \
                    or not 0 <= candidate.get("address") < 1 << 64:
                continue
            candidate_mnemonic = str(candidate.get("mnemonic", "")).lower()
            candidate_operands = str(candidate.get("operands", ""))
            if candidate_mnemonic not in _RENODE_STORE_MNEMONICS:
                continue
            if "<tohost>" in candidate_operands.lower():
                terminal_pcs.add(candidate["address"])
                continue
            match = _RENODE_ZERO_STORE_RE.search(candidate_operands)
            if match and match.group(1).lower() == register:
                terminal_pcs.add(candidate["address"])
    return terminal_pcs


def _scan_pc_trace(path: Path, adapter: str, add_pc: Any,
                   errors: list[str] | None = None) -> tuple[int, str | None]:
    pattern = PC_TRACE_PATTERNS[adapter]
    records = 0
    carry = b""
    digest = hashlib.sha256()

    def consume(data: bytes) -> None:
        nonlocal records
        for match in pattern.finditer(data):
            pc = int(match[1], 16)
            if pc >= 1 << 64:
                if errors is not None:
                    errors.append("trace-record-error")
                continue
            add_pc(pc)
            records += 1

    try:
        with path.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
                data = carry + block
                newline = data.rfind(b"\n")
                if newline < 0:
                    carry = data
                    continue
                consume(data[:newline + 1])
                carry = data[newline + 1:]
        consume(carry)
    except OSError as exc:
        if errors is not None:
            errors.append(f"trace-read-error:{type(exc).__name__}")
        # Preserve records already parsed before a late read failure; the
        # caller will mark the result incomplete using ``errors``.
        return records, None
    return records, digest.hexdigest()

def _renode_binary_trace(raw: bytes) -> tuple[list[int], int, bool]:
    if len(raw) < 10 or raw[:7] != b"ReTrace" or raw[7] != 4:
        return [], 0, False
    pc_width, has_opcode = raw[8], bool(raw[9])
    offset = 10
    multiple = False
    if has_opcode:
        if offset + 2 > len(raw):
            return [], 0, False
        multiple = bool(raw[offset])
        identifier_length = raw[offset + 1]
        offset += 2 + identifier_length
    if offset > len(raw) or not 1 <= pc_width <= 8:
        return [], 0, False
    pcs, records, block_left = [], 0, 0
    while offset < len(raw):
        if multiple and not block_left:
            if offset + 9 > len(raw):
                return pcs, records, False
            offset += 1
            block_left = int.from_bytes(raw[offset:offset + 8], "little")
            offset += 8
            if not block_left:
                return pcs, records, False
        if offset + pc_width > len(raw):
            return pcs, records, False
        pcs.append(int.from_bytes(raw[offset:offset + pc_width], "little"))
        offset += pc_width
        if has_opcode:
            if offset >= len(raw):
                return pcs, records, False
            opcode_length = raw[offset]
            offset += 1
            if offset + opcode_length > len(raw):
                return pcs, records, False
            offset += opcode_length
        if offset >= len(raw):
            return pcs, records, False
        extra = raw[offset]
        offset += 1
        while extra:
            if extra == 1:
                size = 25
            elif extra == 2:
                size = 16
            elif extra == 3:
                if offset + 3 > len(raw):
                    return pcs, records, False
                size = {2: 16, 3: 32, 4: 64}.get(raw[offset + 1], -1) + 3
                if size < 3:
                    return pcs, records, False
            else:
                return pcs, records, False
            if offset + size > len(raw):
                return pcs, records, False
            offset += size
            if offset >= len(raw):
                return pcs, records, False
            extra = raw[offset]
            offset += 1
        records += 1
        if multiple:
            block_left -= 1
    return pcs, records, True


def _scan_drcov_trace(
    path: Path, allowed_pcs: set[int] | None, record_pc: Any,
) -> tuple[int, bool, str | None, int]:
    """Read QEMU's compact DRCOV basic-block table.

    DRCOV emits one 8-byte entry per executed translation block at process
    exit, rather than one line per dynamic instruction.  Every instruction PC
    in an executed block is an executed guest PC, so expanding only against
    the static candidate map preserves the coverage numerator without
    materialising the repeated dynamic stream.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return 0, False, None, 0
    digest = hashlib.sha256(raw).hexdigest()
    header = DRCOV_BB_TABLE_RE.search(raw)
    if header is None:
        return 0, False, digest, 0
    try:
        count = int(header[1])
    except (TypeError, ValueError):
        return 0, False, digest, 0
    offset = header.end()
    expected = count * 8
    # The plugin writes exactly ``count`` fixed-size entries.  Extra bytes are
    # not silently ignored because they indicate a different or damaged format.
    valid = expected == len(raw) - offset
    available = min(count, max(0, len(raw) - offset) // 8)
    static_pcs = sorted(allowed_pcs) if allowed_pcs is not None else []
    ignored = 0
    for index in range(available):
        start, size, _module = struct.unpack_from("<IHH", raw, offset + index * 8)
        end = start + size
        mapped = 0
        if allowed_pcs is not None and size:
            position = bisect_left(static_pcs, start)
            while position < len(static_pcs) and static_pcs[position] < end:
                record_pc(static_pcs[position])
                mapped += 1
                position += 1
        if allowed_pcs is not None and mapped == 0:
            ignored += 1
    return count, valid, digest, ignored


def _sidecar_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:120] or "case"


def _sidecar_case_id(row: Mapping[str, Any]) -> object:
    for name in ("case_id", "candidate_id", "stream_index"):
        value = row.get(name)
        if value is not None and value != "":
            return value
    return "target"


def _pc_sidecar_path(out: Path, target_id: str, row: Mapping[str, Any]) -> Path:
    case_id = _sidecar_case_id(row)
    case_digest = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:16]
    return Path(out) / "coverage-pc-sidecars" / (
        f"{_sidecar_part(target_id)}-{case_digest}.json"
    )


def _trace_files_for_sidecar(out: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    files = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            stat = None
        try:
            name = str(path.relative_to(out))
        except ValueError:
            name = str(path)
        files.append({
            "path": name,
            "size_bytes": stat.st_size if stat is not None else None,
            "mtime_ns": stat.st_mtime_ns if stat is not None else None,
        })
    return files


def _declared_trace_sha(row: Mapping[str, Any], observation: Mapping[str, Any]) -> str | None:
    trace = row.get("trace")
    trace = trace if isinstance(trace, Mapping) else {}
    value = row.get("trace_sha256") or trace.get("sha256") \
        or observation.get("trace_sha256")
    return value.lower() if isinstance(value, str) \
        and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def _load_pc_sidecar(
    out: Path, target_id: str, row: Mapping[str, Any], paths: list[Path], *,
    backend: str | None = None, adapter: str | None = None,
    isa_profile: str | None = None, privilege_mode: str | None = None,
) -> tuple[list[int], Counter, int, bool, str | None, int] | None:
    path = _pc_sidecar_path(out, target_id, row)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping) \
            or payload.get("schema_version") != PC_SIDECAR_SCHEMA \
            or payload.get("target") != target_id:
        return None
    if payload.get("case_id") != _sidecar_case_id(row):
        return None
    for field, expected in (
        ("backend", backend), ("adapter", adapter),
        ("isa_profile", isa_profile), ("privilege_mode", privilege_mode),
    ):
        if expected is not None and payload.get(field) != expected:
            return None
    catalog = payload.get("catalog")
    if not isinstance(catalog, Mapping) \
            or catalog.get("catalog_id") != OPCODE_CATALOG_ID \
            or catalog.get("registry_sha256") != OPCODE_CATALOG_REGISTRY_SHA256:
        return None
    artifact = row.get("artifact_sha256") or row.get("bare_elf_sha256") \
        or row.get("linux_elf_sha256")
    if payload.get("artifact_sha256") not in (None, artifact):
        return None
    observation = row.get("observation")
    observation = observation if isinstance(observation, Mapping) else {}
    current_incomplete = (
        row.get("trace_complete") is False
        or observation.get("trace_complete") is False
        or row.get("trace_cleanup_status") == "gap"
        or observation.get("trace_cleanup_status") == "gap"
        or row.get("trace_truncated") is True
        or observation.get("trace_truncated") is True
        or row.get("trace_limit_exceeded") is True
        or observation.get("trace_limit_exceeded") is True
    )
    if payload.get("trace_complete") is True and current_incomplete:
        return None
    if payload.get("trace_complete") is not False and payload.get("trace_complete") is not True:
        return None
    if payload.get("trace_complete") is False and not current_incomplete:
        return None
    expected_files = payload.get("trace_files")
    if isinstance(expected_files, list):
        actual_files = _trace_files_for_sidecar(out, paths)
        if actual_files != expected_files:
            return None
    declared_sha = _declared_trace_sha(row, observation)
    if declared_sha and payload.get("trace_sha256") not in (None, declared_sha):
        return None
    values = payload.get("pcs")
    if not isinstance(values, list) or any(
        type(value) is not int or not 0 <= value < 1 << 64 for value in values
    ) or len(values) != len(set(values)):
        return None
    events = Counter()
    raw_events = payload.get("event_counts")
    if isinstance(raw_events, Mapping):
        for name, value in raw_events.items():
            if isinstance(name, str) and type(value) is int and value >= 0:
                events[name] = value
    events["trace-sidecar-hit"] += 1
    total = payload.get("trace_records_total", len(values))
    if type(total) is not int or total < 0:
        return None
    ignored = payload.get("ignored_records", 0)
    if type(ignored) is not int or ignored < 0:
        return None
    trace_sha = payload.get("trace_sha256")
    trace_sha = trace_sha if isinstance(trace_sha, str) \
        and re.fullmatch(r"[0-9a-fA-F]{64}", trace_sha) else None
    return values, events, total, payload.get("truncated") is True, trace_sha, ignored


def _write_pc_sidecar(
    out: Path, target: Mapping[str, Any], row: Mapping[str, Any], paths: list[Path],
    *, backend: str, adapter: str, pcs: list[int], events: Counter, total: int,
    truncated: bool, trace_sha: str | None, catalog: Mapping[str, Any] | None,
    isa_profile: str | None = None, privilege_mode: str | None = None,
) -> dict[str, Any] | None:
    target_id = str(target.get("id") or "")
    observation = row.get("observation")
    observation = observation if isinstance(observation, Mapping) else {}
    artifact = row.get("artifact_sha256") or row.get("bare_elf_sha256") \
        or row.get("linux_elf_sha256")
    payload = {
        "schema_version": PC_SIDECAR_SCHEMA,
        "target": target_id,
        "case_id": _sidecar_case_id(row),
        "backend": backend,
        "adapter": adapter,
        "artifact_sha256": artifact,
        "isa_profile": isa_profile if isa_profile is not None else (
            row.get("isa_profile") or target.get("isa_profile")
        ),
        "privilege_mode": privilege_mode if privilege_mode is not None else (
            row.get("privilege_mode") or target.get("privilege_mode")
        ),
        "trace_complete": not truncated
        and row.get("trace_truncated") is not True
        and row.get("trace_limit_exceeded") is not True
        and observation.get("trace_truncated") is not True
        and observation.get("trace_limit_exceeded") is not True
        and row.get("trace_complete") is not False
        and observation.get("trace_complete") is not False
        and row.get("trace_cleanup_status") != "gap"
        and observation.get("trace_cleanup_status") != "gap",
        "truncated": bool(truncated),
        "trace_records_total": total,
        "unique_pcs": len(pcs),
        "pcs": list(pcs),
        "trace_sha256": trace_sha,
        "trace_files": _trace_files_for_sidecar(out, paths),
        "ignored_records": max(0, int(events.get("trace-record-error", 0))),
        "event_counts": dict(sorted(events.items())),
        "catalog": {
            "catalog_id": catalog.get("catalog_id"),
            "registry_sha256": catalog.get("registry_sha256"),
        } if isinstance(catalog, Mapping) else None,
    }
    path = _pc_sidecar_path(out, target_id, row)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        raw = path.read_bytes()
        try:
            relative = str(path.relative_to(out))
        except ValueError:
            relative = str(path)
        return {
            "schema_version": PC_SIDECAR_SCHEMA,
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def _scan_trace(paths: list[Path], adapter: str, row: dict[str, Any],
                allowed_pcs: set[int] | None = None,
                terminal_pcs: set[int] | None = None,
                ignore_out_of_scope: bool = False,
                ) -> tuple[list[int], Counter, int, bool, str | None, int]:
    ignore_out_of_scope = ignore_out_of_scope or row.get("trace_scope_projected") is True
    pcs: list[int] = []
    seen_pcs: set[int] = set()
    events = Counter()
    total = 0
    ignored = 0
    renode_trace_pcs = (
        [] if adapter in {"renode", "renode-unique-pc"}
        and allowed_pcs is not None else None
    )
    renode_trace_valid = True

    def record_pc(value: int) -> None:
        nonlocal ignored
        if allowed_pcs is not None and value not in allowed_pcs:
            if not ignore_out_of_scope:
                ignored += 1
            return
        if value not in seen_pcs:
            seen_pcs.add(value)
            pcs.append(value)

    def add_pc(value: int) -> None:
        if renode_trace_pcs is not None:
            renode_trace_pcs.append(value)
            return
        record_pc(value)
    observation = row.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    trace_incomplete = (
        row.get("trace_complete") is False
        or observation.get("trace_complete") is False
        or row.get("trace_cleanup_status") == "gap"
        or observation.get("trace_cleanup_status") == "gap"
    )
    truncated = trace_incomplete or any(
        bool(row.get(name)) or bool(observation.get(name))
        for name in ("trace_truncated", "trace_limit_exceeded")
    ) or (row.get("termination") or observation.get("termination")) in {
        "trace-limit", "trace-limit-exceeded"
    }
    trace_hashes: list[str] = []
    trace_errors: list[str] = []

    # A hard writer/cleanup gap cannot become an observed guest metric by
    # reading the same prefix again.  Reuse inline PC evidence when present;
    # otherwise return a fail-closed gap without opening the raw trace.
    hard_incomplete = trace_is_hard_incomplete(row)
    if hard_incomplete and paths:
        raw_pcs = row.get("trace_pcs")
        if isinstance(raw_pcs, (list, tuple)):
            for value in raw_pcs:
                pc = _pc_value(value)
                if pc is None:
                    trace_errors.append("trace-record-error")
                else:
                    record_pc(pc)
        declared_value = row.get("trace_records_total")
        if declared_value is None:
            declared_value = observation.get("trace_records_total")
        total, total_valid = _nonnegative_count(declared_value, missing=len(pcs))
        if not total_valid:
            total = len(pcs)
            trace_errors.append("trace-record-count-error")
        if trace_errors:
            for error in trace_errors:
                events[error] += 1
        events["trace-read-skipped-incomplete"] += len(paths)
        return (
            pcs, events, int(total or 0), True,
            _declared_trace_sha(row, observation), ignored,
        )

    def remember_trace_hash(value: str | None) -> None:
        if value is not None:
            trace_hashes.append(value)

    for path in paths:
        if adapter == "qemu-drcov-basic-block":
            records, valid, path_sha, path_ignored = _scan_drcov_trace(
                path, allowed_pcs, record_pc,
            )
            remember_trace_hash(path_sha)
            if path_sha is None:
                trace_errors.append("trace-read-error")
            elif not valid:
                trace_errors.append("trace-format-error")
            total += records
            ignored += path_ignored
            events["basic-block-exec"] += records
            truncated = truncated or not valid
            continue
        if adapter in {"renode", "renode-unique-pc"}:
            try:
                with path.open("rb") as stream:
                    prefix = stream.read(7)
            except OSError:
                prefix = b""
                trace_errors.append("trace-read-error")
            if prefix == b"ReTrace":
                try:
                    raw = path.read_bytes()
                except OSError:
                    raw = b""
                    trace_errors.append("trace-read-error")
                if raw:
                    remember_trace_hash(hashlib.sha256(raw).hexdigest())
                trace_pcs, records, valid = _renode_binary_trace(raw)
                renode_trace_valid = renode_trace_valid and valid
                for pc in trace_pcs:
                    add_pc(pc)
                total += records
                events["instruction-exec"] += records
                truncated = truncated or not valid
                if not valid:
                    trace_errors.append("trace-format-error")
                continue
        if adapter in PC_TRACE_PATTERNS:
            records, path_sha = _scan_pc_trace(path, adapter, add_pc, trace_errors)
            remember_trace_hash(path_sha)
            total += records
            events["instruction-exec"] += records
            if adapter == "libriscv-translated":
                events["translation-block"] += records
            continue
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for raw_line in stream:
                    digest.update(raw_line)
                    line = raw_line.decode("utf-8", errors="replace")
                    pc = None
                    if adapter == "qemu":
                        match = QEMU_PLUGIN_RE.search(line)
                        if match:
                            pc = _pc_value(match[1])
                            if pc is not None:
                                events["instruction-exec"] += 1
                            else:
                                trace_errors.append("trace-record-error")
                    elif adapter in {"libriscv-translated", "libriscv-interpreter"}:
                        match = LRSV_TRACE_RE.search(line) or (
                            LRSV_INTERPRETER_RE.search(line)
                            if adapter == "libriscv-interpreter" else None)
                        pc = _pc_value(match[1]) if match else None
                        if match:
                            if pc is not None:
                                events["instruction-exec"] += 1
                                if adapter.endswith("translated"):
                                    events["translation-block"] += 1
                            else:
                                trace_errors.append("trace-record-error")
                    elif adapter == "renode":
                        for event_match in SIM_EVENT_RE.finditer(line):
                            events[event_match[1]] += 1
                        if "RQ1_TOHOST" in line:
                            events["memory-watchpoint"] += 1
                        if "RQ1_EXCEPTION" in line:
                            events["exception"] += 1
                        if RENODE_INTERRUPT_RE.search(line):
                            events["interrupt"] += 1
                    if pc is not None:
                        total += 1
                        add_pc(pc)
                    if adapter.startswith("libriscv"):
                        count_match = LRSV_INSTRUCTION_COUNT_RE.search(line)
                        if count_match:
                            value = int(count_match[1])
                            events["instruction-exec"] = max(events["instruction-exec"], value)
                        if LRSV_BLOCKS_RE.search(line):
                            events["translation-block"] += 1
                        if LRSV_EXCEPTION_RE.search(line):
                            events["exception"] += 1
                    if adapter == "qemu":
                        if QEMU_EXCEPTION_RE.search(line):
                            events["exception"] += 1
                        if QEMU_SYSCALL_RE.search(line):
                            events["syscall"] += 1
            remember_trace_hash(digest.hexdigest())
        except OSError:
            trace_errors.append("trace-read-error")
            continue
    raw_pcs = row.get("trace_pcs")
    if isinstance(raw_pcs, (list, tuple)) and adapter != "unavailable":
        for value in raw_pcs:
            pc = _pc_value(value)
            if pc is None:
                trace_errors.append("trace-record-error")
                continue
            total += 1
            add_pc(pc)
    # Hook adapters can publish a complete unique-PC set together with the
    # larger dynamic execution count.  The latter is a diagnostic count, not a
    # request to discard any PC evidence.
    declared_total, declared_total_valid = _nonnegative_count(
        row.get("trace_records_total"), missing=None,
    )
    if not declared_total_valid:
        declared_total = None
        trace_errors.append("trace-record-count-error")
    if renode_trace_pcs is not None:
        raw_trace_count = len(renode_trace_pcs)
        first_ignored = None
        last_mapped = None
        first_terminal = None
        for index, value in enumerate(renode_trace_pcs):
            if value in allowed_pcs:
                last_mapped = index
            elif first_ignored is None:
                first_ignored = index
            if first_terminal is None and terminal_pcs and value in terminal_pcs:
                first_terminal = index
        terminal = row.get("target_outcome") or observation.get("terminal")
        terminal_evidence = (
            row.get("termination") in {"guest-tohost", "guest-terminal-after-timeout"}
            or observation.get("termination") in {"guest-tohost", "guest-terminal-after-timeout"}
            or row.get("reason") in {"guest-tohost", "guest-tohost-failure"}
            or observation.get("reason") in {"guest-tohost", "guest-tohost-failure"}
        )
        complete = (
            row.get("observer_complete") is True
            and terminal in {
                "complete_observation", "trap", "expected-trap", "nonzero-exit",
            }
        ) or terminal_evidence
        cutoff = None
        if complete and renode_trace_valid and terminal_pcs:
            cutoff = first_terminal
            if cutoff is not None:
                del renode_trace_pcs[cutoff + 1:]
                total = max(0, total - raw_trace_count + len(renode_trace_pcs))
        if cutoff is None:
            # Renode records the few instructions used to leave the capsule
            # after the terminal watchpoint.  They are outside the candidate
            # feature map, but a contiguous suffix after a verified terminal
            # is not an execution-path mismatch.  Keep genuine in-path
            # mismatches strict.
            if (complete and renode_trace_valid and first_ignored is not None
                    and last_mapped is not None and first_ignored > last_mapped):
                del renode_trace_pcs[first_ignored:]
                total = max(0, total - raw_trace_count + len(renode_trace_pcs))
        for value in renode_trace_pcs:
            record_pc(value)
    declared_events = row.get("event_counts")
    if declared_events is not None and not isinstance(declared_events, dict):
        trace_errors.append("trace-event-count-error")
    elif isinstance(declared_events, dict):
        for name, value in declared_events.items():
            if isinstance(name, str) and type(value) is int and value >= 0:
                events[name] = max(events[name], value)
            else:
                trace_errors.append("trace-event-count-error")
    if declared_total is not None:
        total = max(total, declared_total)
    if trace_errors:
        truncated = True
        for error in trace_errors:
            events[error] += 1
    trace_sha = None
    if trace_hashes:
        trace_sha = (trace_hashes[0] if len(trace_hashes) == 1 else
                     hashlib.sha256(json.dumps(trace_hashes, separators=(",", ":")).encode()).hexdigest())
    return pcs, events, total, truncated, trace_sha, ignored


def _guest_coverage(feature: dict[str, Any] | None, pcs: list[int], total: int,
                    truncated: bool, ignored_trace_records: int,
                    evidence_reason: str | None = None,
                    isa_profile_hint: str | None = None,
                    privilege_mode_hint: str | None = None) -> dict[str, Any]:
    def empty(status: str, reason: str, registry: dict[str, Any] | None = None):
        metrics = {}
        for name in GUEST_METRICS:
            if registry is None or name == "PCov":
                eligible_units = []
            else:
                eligible_units = sorted(registry["metrics"][name])
            eligible = len(eligible_units)
            metric = _metric(0, eligible, status=status, reason=reason)
            metric["covered_units"] = []
            metric["eligible_units"] = eligible_units
            if registry:
                metric["coverage_profile_id"] = registry["metric_profile_id"]
                if name != "PCov":
                    metric["registry_sha256"] = hashlib.sha256(json.dumps(
                        metric["eligible_units"], separators=(",", ":")
                    ).encode()).hexdigest()
            metrics[name] = metric
        return {"status": status, "reason": reason,
                "trace_records": len(pcs), "trace_records_total": total,
                "mapped_records": 0, "ignored_records": ignored_trace_records,
                "metrics": metrics,
                **({"coverage_profile_id": registry["metric_profile_id"],
                   "coverage_registry_sha256": registry["registry_sha256"],
                   "isa_profile": registry["isa_profile"],
                   "privilege_mode": registry["privilege_mode"]} if registry else {})}

    feature = feature if isinstance(feature, dict) else None
    feature_profile = str(feature.get("isa_profile") or "").split("/", 1)[0].strip().lower() \
        if feature else ""
    profile_hint = str(isa_profile_hint or "").split("/", 1)[0].strip().lower()
    profile = feature_profile or profile_hint or (
        str(feature.get("lane") or "").split("/", 1)[0].strip().lower() if feature else ""
    )
    feature_mode = str(feature.get("privilege_mode") or "") if feature else ""
    mode_hint = str(privilege_mode_hint or "")
    privilege_mode = feature_mode or mode_hint or "user"
    if ((feature_profile and profile_hint and feature_profile != profile_hint)
            or (feature_mode and mode_hint and feature_mode != mode_hint)):
        try:
            registry = metric_registry_for_profile(
                feature_profile or profile_hint or profile,
                feature_mode or mode_hint or privilege_mode,
            )
        except ValueError:
            registry = None
        return empty("gap", "isa-profile-feature-mismatch", registry)
    if not profile:
        reason = "static-feature-map-missing" if not feature else (
            "static-feature-map-stale" if feature.get("decoder_version") != DECODER_VERSION
            else "isa-profile-not-canonical"
        )
        return empty("NA" if not feature else "gap", reason)
    try:
        registry = metric_registry_for_profile(profile, privilege_mode)
    except ValueError as exc:
        return empty("gap", str(exc))
    if not feature:
        return empty("NA", "static-feature-map-missing", registry)
    if feature.get("decoder_version") != DECODER_VERSION:
        return empty("gap", "static-feature-map-stale", registry)
    feature_map_partial = (
        feature.get("status", "ok") != "ok"
        and feature.get("reason") == "unknown-or-reserved-instruction-encoding"
    )
    if feature.get("status", "ok") != "ok" and not feature_map_partial:
        return empty("gap", "static-feature-map-invalid", registry)
    instructions = feature.get("instructions", [])
    if not isinstance(instructions, list) or not instructions:
        return empty("gap", "static-feature-map-empty", registry)
    # This map defines the PCov denominator. Filtering malformed rows would
    # silently shrink that denominator and turn incomplete evidence into a
    # seemingly valid observation.
    if any(
        not isinstance(item, dict)
        or type(item.get("address")) is not int
        or not 0 <= item.get("address") < 1 << 64
        or not isinstance(item.get("encoding_id"), str)
        or not item.get("encoding_id")
        for item in instructions
    ):
        return empty("gap", "static-feature-map-incomplete", registry)
    addresses = [item["address"] for item in instructions]
    if len(set(addresses)) != len(addresses):
        return empty("gap", "static-feature-map-duplicate-pc", registry)
    by_pc = {item["address"]: item for item in instructions}
    mapped = [by_pc[pc] for pc in pcs if pc in by_pc]
    reason = (evidence_reason if evidence_reason else
              "guest-pc-map-mismatch" if ignored_trace_records else
              "trace-source-truncated" if truncated else
              "guest-pc-trace-missing" if not pcs else
              "guest-pc-map-empty" if not mapped else None)

    def configuration(item: dict[str, Any]) -> set[str]:
        values = item.get("configuration_ids") or (
            [item.get("configuration_id")] if item.get("configuration_id") else []
        )
        return {str(value) for value in values if value}

    def types(item: dict[str, Any]) -> set[str]:
        values = item.get("type_ids") or (
            [item.get("type_id")] if item.get("type_id") else []
        )
        return {str(value) for value in values if value}

    def metric_eligible(item: dict[str, Any], name: str) -> bool:
        excluded = item.get("coverage_excluded_metrics", ())
        return not (
            item.get("coverage_excluded") is True
            or isinstance(excluded, (list, tuple, set, frozenset))
            and name in excluded
        )

    eligible = {
        "PCov": {
            item["address"] for item in instructions
            if metric_eligible(item, "PCov")
        },
        "ICov-encoding": set(registry["metrics"]["ICov-encoding"]),
        "ICov-type": set(registry["metrics"]["ICov-type"]),
        "CCov": set(registry["metrics"]["CCov"]),
    }
    observed = {
        "PCov": {
            item.get("address") for item in mapped
            if metric_eligible(item, "PCov")
        },
        "ICov-encoding": {str(item.get("encoding_id")) for item in mapped
                          if metric_eligible(item, "ICov-encoding")
                          if item.get("encoding_id")},
        "ICov-type": set().union(*(
            types(item) for item in mapped if metric_eligible(item, "ICov-type")
        )),
        "CCov": set().union(*(
            configuration(item) for item in mapped if metric_eligible(item, "CCov")
        )),
    }
    out_of_profile = {
        name: observed[name] - eligible[name]
        for name in ("ICov-encoding", "ICov-type", "CCov")
    }
    trace_reason = reason
    if any(out_of_profile.values()):
        reason = "feature-unit-outside-isa-profile"
    elif feature_map_partial and mapped and reason is None:
        reason = "static-feature-map-partial"
    map_partial = feature_map_partial or any(out_of_profile.values())
    trace_status = (
        "gap" if (ignored_trace_records or evidence_reason) and not mapped
        else "partial" if mapped and (truncated or ignored_trace_records or evidence_reason)
        else "observed" if mapped else "NA"
    )
    mapped_status = (
        "gap" if (ignored_trace_records or evidence_reason or any(out_of_profile.values())) and not mapped
        else "partial" if mapped and (truncated or ignored_trace_records or evidence_reason
                                       or map_partial)
        else "observed" if mapped else "NA"
    )
    metrics = {}
    for name in GUEST_METRICS:
        covered = observed[name] & eligible[name]
        metric_reason = trace_reason if name == "PCov" else reason
        metric = _metric_value(covered, eligible[name], reason=metric_reason)
        metric_status = trace_status if name == "PCov" else mapped_status
        metric["status"] = metric_status if metric["eligible"] else "NA"
        if metric["status"] != "observed":
            metric["value"] = None
        metric["covered_units"] = sorted(covered, key=str)
        metric["eligible_units"] = sorted(eligible[name], key=str)
        metric["coverage_profile_id"] = registry["metric_profile_id"]
        if name != "PCov":
            metric["registry_sha256"] = hashlib.sha256(json.dumps(
                metric["eligible_units"], separators=(",", ":")
            ).encode()).hexdigest()
        metrics[name] = metric
    return {
        "status": mapped_status, "reason": reason,
        "trace_records": len(pcs), "trace_records_total": total,
        "mapped_records": len(mapped), "ignored_records": ignored_trace_records,
        "metrics": metrics, "coverage_profile_id": registry["metric_profile_id"],
        "coverage_registry_sha256": registry["registry_sha256"],
        "isa_profile": profile, "privilege_mode": privilege_mode,
    }


def summarize_simulator_coverage(target: dict[str, Any], row: dict[str, Any], out: Path,
                                 feature: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(target, dict):
        return {
            "schema_version": SCHEMA,
            "status": "gap",
            "reason": "target-invalid",
        }
    if not isinstance(row, dict):
        return {
            "schema_version": SCHEMA,
            "status": "gap",
            "reason": "target-observation-invalid",
        }
    target_id = str(target.get("id", ""))
    spec = ADAPTERS.get(target_id)
    if not spec:
        return {"schema_version": SCHEMA, "status": "NA", "reason": "adapter-not-configured"}
    kind = str(target.get("kind", ""))
    backend = str(spec["backend"])
    source_patched = _record_uses_patched_source(row)
    adapter = "unavailable" if source_patched else spec.get("parser") or (
        ("libriscv-translated" if target_id == "T-LRSV-TRANS"
         else "libriscv-interpreter") if kind == "libriscv" else kind
    )
    trace_paths = _paths_from_row(
        out, row, "trace_path", "coverage_trace", skip_incomplete=False,
    )
    feature_map = feature if isinstance(feature, dict) else {}
    feature_instructions = feature_map.get("opcode_catalog_execution_instructions")
    if not isinstance(feature_instructions, list):
        feature_instructions = feature_map.get("instructions", [])
    sidecar_isa_profile = str(
        feature_map.get("isa_profile") or row.get("isa_profile")
        or target.get("isa_profile") or ""
    ).strip() or None
    sidecar_privilege_mode = str(
        feature_map.get("privilege_mode") or row.get("privilege_mode")
        or target.get("privilege_mode") or ""
    ).strip() or None
    allowed_pcs = {item.get("address") for item in feature_instructions
                   if isinstance(item, dict) and type(item.get("address")) is int
                   and 0 <= item.get("address") < 1 << 64} or None
    cached_scan = _load_pc_sidecar(
        out, target_id, row, trace_paths,
        backend=backend, adapter=spec["trace"],
        isa_profile=sidecar_isa_profile,
        privilege_mode=sidecar_privilege_mode,
    )
    if cached_scan is not None:
        pcs, events, total, truncated, trace_sha, ignored = cached_scan
    else:
        with _TRACE_IO_GATE:
            pcs, events, total, truncated, trace_sha, ignored = _scan_trace(
                trace_paths, adapter, row,
                allowed_pcs,
                _renode_terminal_pcs(feature) if adapter.startswith("renode") else None,
                ignore_out_of_scope=row.get("trace_scope_projected") is True,
            )
    # Restore the complete ELF PC set for the catalog metric; legacy RV
    # metrics are projected back to their existing feature scope below.
    if adapter != "unavailable":
        seen_pcs = set(pcs)
        for value in row.get("opcode_catalog_trace_pcs", ()):
            pc = _pc_value(value)
            if pc is not None and (allowed_pcs is None or pc in allowed_pcs) \
                    and pc not in seen_pcs:
                pcs.append(pc)
                seen_pcs.add(pc)
    legacy_pcs = pcs
    legacy_instructions = feature_map.get("instructions")
    if ("opcode_catalog_execution_instructions" in feature_map
            and isinstance(legacy_instructions, list)):
        legacy_allowed_pcs = {
            item.get("address") for item in legacy_instructions
            if isinstance(item, dict) and type(item.get("address")) is int
        }
        legacy_pcs = [pc for pc in pcs if pc in legacy_allowed_pcs]
    observation = row.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    outcome = row.get("target_outcome") or observation.get("terminal")
    state_observed = outcome in {
        "complete_observation", "trap", "expected-trap", "nonzero-exit",
    } and row.get("observer_complete") is True
    trace_terminal_observed = (
        (row.get("terminal_observed") is True
         or observation.get("terminal_observed") is True)
        and (row.get("termination") or observation.get("termination")) in {
            "guest-tohost", "guest-tohost-failure", "guest-trap",
            "guest-exception", "guest-ebreak", "guest-ecall",
            "guest-natural-stop", "guest-invalid-state",
        }
        and outcome != "timeout"
    )
    # PC-based guest metrics require a complete execution trace, not a complete
    # state mailbox. An explicit guest terminal closes the trace even when the
    # separate RVOBS1 state observation is unavailable.  A valid, non-truncated
    # trace also closes the guest-coverage channel when the process exits with
    # an observed exception/nonzero result; the missing architectural frame is
    # retained in the observation channel and must not erase PC/IC/CC evidence.
    trace_closed = state_observed or trace_terminal_observed
    attempted = row.get("target_attempted") is True or row.get("process_started") is True
    if pcs:
        status = "observed" if trace_closed else "partial"
    elif attempted:
        status = "gap"
    else:
        status = "NA"
    trace_error_reason = next(
        (reason for reason in (
            "trace-read-error", "trace-format-error", "trace-record-error",
            "trace-record-count-error", "trace-writer-cleanup-gap",
            "trace-file-missing",
        ) if any(name == reason or name.startswith(f"{reason}:") for name in events)),
        None,
    )
    trace_incomplete = (
        row.get("trace_complete") is False
        or observation.get("trace_complete") is False
        or row.get("trace_cleanup_status") == "gap"
        or observation.get("trace_cleanup_status") == "gap"
    )
    if trace_incomplete and trace_error_reason is None:
        trace_error_reason = (
            row.get("trace_reason") or observation.get("trace_reason")
            or "trace-writer-cleanup-gap"
        )
    if kind == "rvvm" and not pcs and trace_error_reason is None:
        trace_error_reason = row.get("trace_gap_reason") or row.get("trace_reason")
    if target_id == "T-RAX" and attempted and not pcs and trace_error_reason is None:
        trace_error_reason = "rax-riscv-pc-trace-empty"
    if source_patched and attempted:
        trace_error_reason = "target-source-patch-forbidden"
    if (
        not trace_closed and pcs and not truncated and not trace_incomplete
        and trace_error_reason is None
    ):
        trace_closed = True
    # A complete, parseable PC trace is sufficient for guest coverage even
    # when the process exits by signal before publishing an RVOBS1 frame.
    # Recompute the status after the fallback above; otherwise the initial
    # ``partial`` value leaks into the final guest summary.
    if trace_closed and pcs and not truncated and trace_error_reason is None:
        status = "observed"
    if trace_error_reason:
        status = "gap" if not pcs else "partial"
    guest = _guest_coverage(
        feature, legacy_pcs, total, truncated, ignored, trace_error_reason,
        str(row.get("isa_profile") or target.get("isa_profile") or "") or None,
        str(row.get("privilege_mode") or target.get("privilege_mode") or "") or None,
    )
    if trace_error_reason:
        # The trace failure is the direct cause of absent dynamic evidence;
        # keep it visible even when the ELF feature map is missing too.
        guest["reason"] = trace_error_reason
    if guest["status"] == "gap":
        status = "gap"
    elif guest["status"] == "partial":
        status = "partial"
    elif not pcs and status == "observed":
        status = "gap"
    if status != "observed":
        for metric in guest.get("metrics", {}).values():
            if status == "gap":
                metric["status"] = "gap"
                metric["value"] = None
                metric["reason"] = guest.get("reason") or "target-observation-gap"
            elif metric.get("status") in {"observed", "partial"}:
                metric["status"] = "partial" if status == "partial" else status
                metric["value"] = None
                metric["reason"] = guest.get("reason") or (
                    "target-observation-incomplete" if status == "partial"
                    else "target-observation-unavailable")
        guest["status"] = status
    family_reason = guest.get("reason") or (
        "target-observation-incomplete" if status == "partial" else
        "target-observation-unavailable" if status in {"NA", "gap"} else None
    )
    # 没有可用 PC 时 trace 语义为未知；声称 canonical 会把缺失伪装成语义。
    trace_semantics = "canonical-executed-instruction-pc" if pcs else None
    guest_trace_semantics = (
        "canonical-executed-instruction-pc" if legacy_pcs else None
    )
    events_record = {
        "schema_version": "rq1-simulator-events-v3", "backend": backend,
        "adapter": spec["trace"], "status": status,
        "event_counts": dict(sorted(events.items())),
        "trace": {"format": spec["trace"],
                  "paths": [str(path.relative_to(out)) for path in trace_paths],
                  "sha256": trace_sha, "records": total, "unique_pcs": len(pcs),
                  "ignored_records": ignored, "truncated": truncated,
                  "errors": sorted({error for error in events
                                     if error.startswith("trace-") and error.endswith("error")}),
                  "semantics": trace_semantics},
    }
    guest["trace_semantics"] = guest_trace_semantics
    guest["metric_scope"] = {
        "PCov": "static-instruction-location-execution-within-elf",
        "ICov-encoding": "fixed-isa-profile-instruction-form-universe",
        "ICov-type": "fixed-isa-profile-semantic-type-universe",
        "CCov": "fixed-isa-profile-configuration-class-universe",
    }
    catalog_trace_values = row.get("opcode_catalog_trace_pcs")
    if isinstance(catalog_trace_values, (list, tuple)):
        catalog_trace_values = list(catalog_trace_values)
        catalog_seen = {
            pc for value in catalog_trace_values
            if (pc := _pc_value(value)) is not None
        }
        catalog_trace_values.extend(pc for pc in pcs if pc not in catalog_seen)
    else:
        catalog_trace_values = pcs
    if row.get("opcode_catalog_trace_invalid") is True:
        catalog_trace_values.append(None)
    result = {"schema_version": SCHEMA, "status": status, "reason": family_reason,
              "backend": backend, "adapter": spec["trace"], "guest": guest,
              "behavior_signature": behavior_signature({
                  "trace_pcs": legacy_pcs, "outcome": outcome,
                  "exit_code": (row.get("observation") or {}).get("returncode"),
              }),
              "simulator_events": events_record,
              "_rv_opcode_catalog_trace_pcs": catalog_trace_values}
    result = attach_rv_instruction_metrics(feature, result)
    pc_sidecar = _write_pc_sidecar(
        out, target, row, trace_paths, backend=backend, adapter=spec["trace"],
        pcs=pcs, events=events, total=total, truncated=truncated,
        trace_sha=trace_sha,
        catalog=result.get("rv_opcode_catalog_coverage"),
        isa_profile=sidecar_isa_profile,
        privilege_mode=sidecar_privilege_mode,
    )
    if pc_sidecar is not None:
        result["pc_sidecar"] = pc_sidecar
    # The full catalog PC evidence is an in-memory input, not a second trace
    # artifact in the persisted simulator summary.
    result.pop("_rv_opcode_catalog_trace_pcs", None)
    return result


def _source_tree_identity_warning(identity: dict[str, Any]) -> str | None:
    dirty_paths = identity.get("non_generated_dirty_paths")
    if (identity.get("source_tree_status") == "dirty"
            or identity.get("source_tree_dirty") is True
            or isinstance(dirty_paths, list) and bool(dirty_paths)):
        return "source-coverage-source-tree-dirty"
    drift_reason = identity.get("source_tree_drift_reason")
    if drift_reason:
        return str(drift_reason)
    if (identity.get("source_tree_status") != "clean"
            or identity.get("source_tree_dirty") is not False
            or not isinstance(dirty_paths, list)):
        return "source-coverage-source-tree-state-unavailable"
    return None


def _source_commit_proven_mismatch(identity: dict[str, Any]) -> bool:
    observed = identity.get("source_commit_observed") or identity.get("coverage_binary_sha")
    expected = (identity.get("source_commit_expected") or identity.get("source_commit")
                or identity.get("coverage_binary_expected_sha"))
    target = identity.get("target_commit")
    return bool(
        (observed and expected and observed != expected)
        or (observed and target and observed != target)
        or (expected and target and expected != target)
    )


def aggregate_simulator_coverage(
    records: Iterable[dict[str, Any]], *, include_guest_metrics: bool = True,
) -> dict[str, Any]:
    """Aggregate target evidence and optional legacy guest metrics.

    ``SimSrcCov`` is the active RQ1 result.  The fixed PCov/ICov/CCov and RV
    instruction registry are retained by default for historical replay, but
    active runs can set ``include_guest_metrics=False`` so those fixed
    universes neither appear in the main summary nor turn a source-coverage
    result into a gap.
    """
    records = list(records or ())
    include_guest_metrics = include_guest_metrics is not False
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    source_targets: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    source_rank = {"gap": 4, "partial": 3, "observed": 2, "NA": 1, "deferred": 0}
    rv_records: list[dict[str, Any]] = []
    opcode_catalog_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        raw_coverage = record.get("simulator_coverage")
        coverage = None
        coverage_issue = None
        source_patched = _record_uses_patched_source(record)
        record_ineligible = _coverage_record_ineligible(record)
        if source_patched:
            coverage_issue = "target-source-patch-forbidden"
        elif record_ineligible and raw_coverage is not None:
            coverage_issue = "coverage-case-ineligible"
        elif isinstance(raw_coverage, dict):
            if raw_coverage.get("schema_version") == SCHEMA:
                coverage = raw_coverage
            else:
                coverage_issue = "simulator-coverage-schema-mismatch"
        elif raw_coverage is not None:
            coverage_issue = "simulator-coverage-invalid"
        else:
            coverage_issue = "simulator-coverage-missing"
        coverage_backend = raw_coverage.get("backend") if isinstance(raw_coverage, dict) else None
        if not any(record.get(name) or coverage_backend
                   for name in ("target", "id", "backend_id")):
            continue
        if (include_guest_metrics and not source_patched and not record_ineligible
                and ((isinstance(raw_coverage, dict)
                      and raw_coverage.get("schema_version") == SCHEMA)
                     or "rv_instruction_coverage" in record)):
            rv_records.append(record)
        if (include_guest_metrics and not source_patched and not record_ineligible
                and isinstance(raw_coverage, dict)
                and isinstance(raw_coverage.get("rv_opcode_catalog_coverage"), Mapping)):
            opcode_catalog_records.append(record)
        target = str(record.get("target") or record.get("id") or
                     record.get("backend_id") or coverage_backend or "")
        stratum = str(record.get("stratum") or "")
        key = tuple(str(record.get(name) or "") for name in ("method", "route", "lane")) + (
            stratum, target)
        group = groups.setdefault(key, {
            "cases": 0, "events": Counter(), "guest": {}, "statuses": [],
            "case_ids": set(), "observed_cases": 0, "incomplete_cases": 0,
            "pcov_eligible_by_artifact": {}, "pcov_covered_by_artifact": {},
            "fixed_metric_statuses": {name: [] for name in GUEST_METRICS if name != "PCov"},
            "fixed_metric_errors": {name: 0 for name in GUEST_METRICS if name != "PCov"},
            "fixed_metric_missing": {name: 0 for name in GUEST_METRICS if name != "PCov"},
            "canonical_metrics": None, "canonical_registry_sha": None,
            "pcov_identity_missing_records": 0,
            "pcov_static_map_missing_records": 0,
            "pcov_invalid_unit_records": 0,
            "pcov_missing_metric_records": 0,
            "reasons": set(), "profile_ids": set(), "registry_shas": set(),
            "metric_registry_shas": {}, "eligible_by_metric": {},
            "target_identity_ids": set(), "profile_conflict": False,
            "identity_conflict": False, "pcov_universe_conflict": False,
        })
        if not include_guest_metrics:
            # Active runs are judged by the target-produced source profile.
            # A missing/unsupported guest trace is diagnostic and must not
            # turn an otherwise valid SimSrcCov result into a guest-metric
            # failure.
            source_record = record.get("simulator_source_coverage")
            coverage_status = (
                str(source_record.get("status", "gap"))
                if isinstance(source_record, dict)
                and source_record.get("schema_version") == SOURCE_SCHEMA
                else "partial" if coverage is not None else
                "gap" if source_patched or raw_coverage is not None else "partial"
            )
        else:
            coverage_status = (
                str(coverage.get("status", "gap")) if coverage is not None
                else "gap" if source_patched or raw_coverage is not None else "partial"
            )
        if coverage_status not in {"observed", "partial", "gap", "NA"}:
            coverage_status = "gap"
            group["reasons"].add("simulator-coverage-status-invalid")
        # In the active source-only contract a guest coverage payload is
        # intentionally absent.  Do not report that deliberate absence as a
        # target-source error when SimSrcCov is present.
        if coverage_issue and not (
            not include_guest_metrics
            and coverage_issue == "simulator-coverage-missing"
            and isinstance(record.get("simulator_source_coverage"), dict)
            and record["simulator_source_coverage"].get("schema_version") == SOURCE_SCHEMA
        ):
            group["reasons"].add(coverage_issue)
        group["statuses"].append(coverage_status)
        group["cases"] += 1
        if coverage_status == "observed":
            group["observed_cases"] += 1
        else:
            group["incomplete_cases"] += 1
        artifact_id = _artifact_digest_for_record(record, stratum)
        case_id = (record.get("case_id") or record.get("candidate_id")
                   or record.get("artifact_sha256") or artifact_id)
        valid_artifact_id = (
            isinstance(artifact_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", artifact_id) is not None
        )
        valid_identity = (
            isinstance(case_id, str) and bool(case_id.strip())
            and valid_artifact_id
        )
        cohort_key = f"{case_id}|{artifact_id}" if valid_identity else None
        if cohort_key:
            group["case_ids"].add(str(cohort_key))
        identity = record.get("target_identity")
        if identity is None:
            identity = record.get("identity")
        identity = identity if isinstance(identity, dict) else {}
        if isinstance(identity, dict):
            identity_id = (identity.get("identity_digest") or identity.get("expected_identity_digest")
                           or identity.get("binary_sha256") or identity.get("binary")
                           or record.get("target_identity_id"))
            if identity_id:
                group["target_identity_ids"].add(str(identity_id))
        elif record.get("target_identity_id"):
            group["target_identity_ids"].add(str(record["target_identity_id"]))
        source = record.get("simulator_source_coverage")
        if source_patched and isinstance(source, dict):
            source = {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov",
                      "status": "gap", "reason": "target-source-patch-forbidden"}
        elif isinstance(source, dict) and source.get("schema_version") != SOURCE_SCHEMA:
            source = {
                "schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov",
                "status": "gap", "reason": "source-coverage-schema-mismatch",
                "profile_path": source.get("profile_path"),
                "identity": source.get("identity") if isinstance(source.get("identity"), dict) else {},
            }
        elif isinstance(source, dict) and source.get("status") in {"observed", "partial"}:
            source_identity = source.get("identity")
            source_identity = source_identity if isinstance(source_identity, dict) else {}
            target_commit = source_identity.get("target_commit")
            expected_source_commit = (
                source_identity.get("source_commit_expected")
                or source_identity.get("source_commit")
                or source_identity.get("coverage_binary_expected_sha")
            )
            observed_source_commit = (
                source_identity.get("source_commit_observed")
                or source_identity.get("coverage_binary_sha")
            )
            warnings = set(source.get("provenance_warnings") or ())
            tree_warning = _source_tree_identity_warning(source_identity)
            if tree_warning:
                warnings.add(tree_warning)
            # Source-tree drift is provenance only: keep the collected metric.
            identity_reason = (
                "source-coverage-source-commit-mismatch"
                if _source_commit_proven_mismatch(source_identity) else None
            )
            if not observed_source_commit or not expected_source_commit or not target_commit:
                warnings.add("source-coverage-source-commit-identity-unavailable")
            if source_identity.get("source_commit_match") is False and not identity_reason:
                warnings.add("source-commit-match-flag-unverified")
            expected_binary_sha = source_identity.get("coverage_binary_expected_sha256")
            observed_binary_sha = source_identity.get("coverage_binary_sha256")
            if not identity_reason and not expected_binary_sha:
                identity_reason = "source-coverage-binary-sha-not-declared"
            elif not identity_reason and not observed_binary_sha:
                identity_reason = "source-coverage-binary-sha-unavailable"
            elif (not identity_reason and isinstance(expected_binary_sha, str)
                  and isinstance(observed_binary_sha, str)
                  and expected_binary_sha.lower() != observed_binary_sha.lower()):
                identity_reason = "source-coverage-binary-sha-mismatch"
            if not identity_reason and source_identity.get("instrumented_binaries_match") is False:
                entries = source_identity.get("instrumented_binaries", [])
                if any(not item.get("expected_sha256") for item in entries
                       if isinstance(item, dict)):
                    identity_reason = "source-coverage-instrumented-binary-sha-not-declared"
                elif any(not item.get("sha256") for item in entries
                         if isinstance(item, dict)):
                    identity_reason = "source-coverage-instrumented-binary-sha-unavailable"
                else:
                    identity_reason = "source-coverage-instrumented-binary-sha-mismatch"
            if identity_reason:
                source = {
                    "schema_version": SOURCE_SCHEMA,
                    "metric_family": "SimSrcCov",
                    "status": "gap",
                    "reason": identity_reason,
                    "profile_path": source.get("profile_path"),
                    "identity": source_identity,
                    "provenance_warnings": sorted(warnings),
                }
            elif warnings:
                source = {**source, "provenance_warnings": sorted(warnings)}
        if isinstance(source, dict):
            current = source_targets.get(key)
            source_status = str(source.get("status"))
            if current is None or source_rank.get(source_status, -1) > source_rank.get(
                str(current.get("status")), -1
            ):
                source_targets[key] = source
            elif (source_rank.get(source_status, -1)
                  == source_rank.get(str(current.get("status")), -1) and current != source):
                current_measurement = {
                    name: value for name, value in current.items()
                    if name not in {"identity", "provenance_warnings",
                                    "provenance_identity_candidates"}
                }
                source_measurement = {
                    name: value for name, value in source.items()
                    if name not in {"identity", "provenance_warnings",
                                    "provenance_identity_candidates"}
                }
                if current_measurement == source_measurement:
                    candidates = []
                    for item in (
                        *current.get("provenance_identity_candidates", (current.get("identity"),)),
                        *source.get("provenance_identity_candidates", (source.get("identity"),)),
                    ):
                        if item is not None and item not in candidates:
                            candidates.append(item)
                    source_targets[key] = {
                        **current,
                        "provenance_warnings": sorted(set(
                            current.get("provenance_warnings", [])
                            + source.get("provenance_warnings", [])
                        )),
                        "provenance_identity_candidates": candidates,
                    }
                else:
                    source_targets[key] = {
                        "schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov",
                        "status": "gap", "reason": "simulator-source-coverage-conflict",
                        "candidates": [current, source],
                    }
        if coverage is None:
            if record.get("target_attempted") is True or record.get("process_started") is True:
                group["pcov_static_map_missing_records"] += 1
            continue
        event_record = coverage.get("simulator_events", {})
        if not isinstance(event_record, dict):
            group["reasons"].add("simulator-events-invalid")
            event_record = {}
        event_counts = event_record.get("event_counts", {})
        if isinstance(event_counts, dict):
            for event_name, event_count in event_counts.items():
                if isinstance(event_name, str) and type(event_count) is int and event_count >= 0:
                    group["events"][event_name] = max(
                        group["events"][event_name], event_count,
                    )
                else:
                    group["reasons"].add("simulator-event-count-invalid")
        elif event_counts is not None:
            group["reasons"].add("simulator-event-count-invalid")
        if not include_guest_metrics:
            continue
        guest_record = coverage.get("guest", {})
        guest_record = guest_record if isinstance(guest_record, dict) else {}
        if not isinstance(coverage.get("guest", {}), dict):
            group["reasons"].add("guest-coverage-invalid")
        profile_id = guest_record.get("coverage_profile_id")
        registry_sha = guest_record.get("coverage_registry_sha256")
        if profile_id:
            group["profile_ids"].add(str(profile_id))
        if registry_sha:
            group["registry_shas"].add(str(registry_sha))
        guest_metrics = guest_record.get("metrics")
        guest_metrics = guest_metrics if isinstance(guest_metrics, dict) else {}
        record_registry = None
        if isinstance(profile_id, str):
            parts = profile_id.split("|")
            if len(parts) == 3:
                try:
                    candidate_registry = metric_registry_for_profile(parts[0], parts[1])
                except ValueError:
                    candidate_registry = None
                if (candidate_registry is not None
                        and candidate_registry["metric_profile_id"] == profile_id
                        and candidate_registry["registry_sha256"] == registry_sha):
                    record_registry = candidate_registry
                    if group["canonical_metrics"] is None:
                        group["canonical_metrics"] = {
                            name: set(units)
                            for name, units in candidate_registry["metrics"].items()
                        }
                        group["canonical_registry_sha"] = candidate_registry["registry_sha256"]
                    elif group["canonical_registry_sha"] != candidate_registry["registry_sha256"]:
                        group["profile_conflict"] = True
                else:
                    group["profile_conflict"] = True
                    group["reasons"].add("isa-profile-registry-mismatch")
            else:
                group["profile_conflict"] = True
                group["reasons"].add("isa-profile-id-invalid")

        pcov_metric = guest_metrics.get("PCov")
        if not isinstance(pcov_metric, dict):
            group["pcov_missing_metric_records"] += 1
        else:
            raw_eligible = pcov_metric.get("eligible")
            raw_covered = pcov_metric.get("covered")
            eligible_raw = pcov_metric.get("eligible_units")
            covered_raw = pcov_metric.get("covered_units")
            metric_status = pcov_metric.get("status")
            eligible_count, eligible_valid = _nonnegative_count(
                raw_eligible, missing=None,
            )
            covered_count, covered_valid = _nonnegative_count(
                raw_covered, missing=None,
            )
            if (not eligible_valid or not covered_valid
                    or eligible_count is None or covered_count is None):
                eligible_count = covered_count = -1
            valid_arrays = isinstance(eligible_raw, (list, tuple)) and isinstance(
                covered_raw, (list, tuple)
            )
            valid_pc_types = valid_arrays and all(
                isinstance(pc, int) and not isinstance(pc, bool)
                for pc in (*eligible_raw, *covered_raw)
            )
            eligible_pcs = set(eligible_raw) if valid_pc_types else set()
            covered_pcs = set(covered_raw) if valid_pc_types else set()
            valid_metric = (
                eligible_count is not None and covered_count is not None
                and eligible_count >= 0 and covered_count >= 0 and valid_pc_types
                and len(eligible_pcs) == len(eligible_raw)
                and len(covered_pcs) == len(covered_raw)
                and all(0 <= pc < 1 << 64 for pc in eligible_pcs | covered_pcs)
                and covered_pcs <= eligible_pcs
                and eligible_count == len(eligible_pcs)
                and covered_count == len(covered_pcs)
                and metric_status in {"observed", "partial", "gap", "NA"}
            )
            structural_valid = valid_metric
            metric_value = pcov_metric.get("value")
            if valid_metric and eligible_count:
                if metric_status == "observed":
                    try:
                        valid_metric = (
                            coverage_status == "observed" and metric_value is not None
                            and math.isclose(
                                float(metric_value), covered_count / eligible_count,
                                rel_tol=0.0, abs_tol=1.1e-6,
                            )
                        )
                    except (TypeError, ValueError):
                        valid_metric = False
                elif metric_value is not None:
                    valid_metric = False
                if coverage_status == "observed" and metric_status != "observed":
                    valid_metric = False
            elif valid_metric:
                valid_metric = covered_count == 0 and not eligible_pcs and not covered_pcs
                if metric_value is not None:
                    valid_metric = False
            if structural_valid and valid_artifact_id and eligible_count > 0:
                previous = group["pcov_eligible_by_artifact"].get(artifact_id)
                if previous is not None and previous != eligible_pcs:
                    group["pcov_universe_conflict"] = True
                    group["reasons"].add("pcov-static-map-conflict")
                else:
                    group["pcov_eligible_by_artifact"][artifact_id] = eligible_pcs
                    if (coverage_status in {"observed", "partial", "gap"}
                            and metric_status in {"observed", "partial", "gap"}):
                        group["pcov_covered_by_artifact"].setdefault(
                            artifact_id, set(),
                        ).update(covered_pcs)
            if not valid_metric:
                group["pcov_invalid_unit_records"] += 1
            elif eligible_count == 0:
                if (coverage_status == "observed"
                        or record.get("target_attempted") is True
                        or record.get("process_started") is True):
                    group["pcov_static_map_missing_records"] += 1
            elif not valid_artifact_id:
                group["pcov_identity_missing_records"] += 1

        for name in ("ICov-encoding", "ICov-type", "CCov"):
            metric = guest_metrics.get(name)
            if not isinstance(metric, dict):
                group["fixed_metric_missing"][name] += 1
                group["fixed_metric_statuses"][name].append("gap")
                group["reasons"].add(f"{name}-metric-missing")
                continue
            metric_status = metric.get("status")
            group["fixed_metric_statuses"][name].append(
                metric_status if metric_status in {"observed", "partial", "gap", "NA"}
                else "gap"
            )
            raw_eligible = metric.get("eligible")
            raw_covered = metric.get("covered")
            eligible_count, eligible_valid = _nonnegative_count(
                raw_eligible, missing=None,
            )
            covered_count, covered_valid = _nonnegative_count(
                raw_covered, missing=None,
            )
            if not eligible_valid or not covered_valid:
                eligible_count = covered_count = -1
            eligible_raw = metric.get("eligible_units")
            covered_raw = metric.get("covered_units")
            valid_arrays = isinstance(eligible_raw, (list, tuple)) and isinstance(
                covered_raw, (list, tuple)
            ) and all(isinstance(unit, str) for unit in [*eligible_raw, *covered_raw])
            eligible_units = set(eligible_raw) if valid_arrays else set()
            covered_units = set(covered_raw) if valid_arrays else set()
            expected_units = (
                set(record_registry["metrics"][name]) if record_registry is not None else None
            )
            valid_metric = (
                eligible_count is not None and covered_count is not None
                and eligible_count >= 0 and covered_count >= 0 and valid_arrays
                and len(eligible_units) == len(eligible_raw)
                and len(covered_units) == len(covered_raw)
                and eligible_count == len(eligible_units)
                and covered_count == len(covered_units)
                and covered_units <= eligible_units
                and metric_status in {"observed", "partial", "gap", "NA"}
                and expected_units is not None and eligible_units == expected_units
                and metric.get("coverage_profile_id") == profile_id
                and metric.get("registry_sha256") == _pcov_unit_digest(expected_units)
            )
            structural_valid = valid_metric
            metric_value = metric.get("value")
            if valid_metric and eligible_count:
                if metric_status == "observed":
                    try:
                        valid_metric = (
                            coverage_status == "observed" and metric_value is not None
                            and math.isclose(
                                float(metric_value), covered_count / eligible_count,
                                rel_tol=0.0, abs_tol=1.1e-6,
                            )
                        )
                    except (TypeError, ValueError):
                        valid_metric = False
                elif metric_value is not None:
                    valid_metric = False
                if coverage_status == "observed" and metric_status != "observed":
                    valid_metric = False
            if structural_valid:
                bucket = group["guest"].setdefault(name, {"covered": set(), "eligible": set()})
                bucket["eligible"].update(expected_units)
                previous_eligible = group["eligible_by_metric"].setdefault(name, expected_units)
                if previous_eligible != expected_units:
                    group["profile_conflict"] = True
                group["metric_registry_shas"].setdefault(name, set()).add(
                    str(metric.get("registry_sha256"))
                )
                if (coverage_status in {"observed", "partial", "gap"}
                        and metric_status in {"observed", "partial", "gap"}):
                    bucket["covered"].update(covered_units)
            if not valid_metric:
                group["fixed_metric_errors"][name] += 1
                group["reasons"].add(f"{name}-profile-universe-or-unit-set-invalid")
                continue

    rows = []
    for (method, route, lane, stratum, target), group in sorted(groups.items()):
        status_set = set(group["statuses"])
        aggregate_status = (
            "observed" if status_set == {"observed"} else
            "gap" if status_set == {"gap"} else
            "NA" if not status_set or status_set == {"NA"} else "partial"
        )
        # SimSrcCov is a run-level target profile.  When the fixed guest
        # instruction channel is disabled, deferred guest rows must not turn
        # an otherwise observed source profile into a partial target row.
        source_record = source_targets.get((method, route, lane, stratum, target))
        if not include_guest_metrics and isinstance(source_record, Mapping):
            source_status = str(source_record.get("status") or "gap")
            if source_status in {"observed", "partial", "gap", "NA"}:
                aggregate_status = source_status
        observation_status = aggregate_status
        pcov_error = bool(
            include_guest_metrics and (
                group["pcov_identity_missing_records"]
                or group["pcov_invalid_unit_records"]
                or group["pcov_missing_metric_records"]
                or group["pcov_universe_conflict"]
                or group["pcov_static_map_missing_records"]
            )
        )
        if include_guest_metrics and group["pcov_identity_missing_records"]:
            group["reasons"].add("pcov-artifact-identity-missing")
        if include_guest_metrics and group["pcov_invalid_unit_records"]:
            group["reasons"].add("pcov-static-unit-set-invalid")
        if include_guest_metrics and group["pcov_missing_metric_records"]:
            group["reasons"].add("pcov-metric-missing")
        if include_guest_metrics and group["pcov_static_map_missing_records"]:
            group["reasons"].add("pcov-static-map-missing")
        if len(group["profile_ids"]) > 1 or len(group["registry_shas"]) > 1:
            group["profile_conflict"] = True
        if len(group["target_identity_ids"]) > 1:
            group["identity_conflict"] = True
        fixed_metric_error = include_guest_metrics and any(
            group["fixed_metric_errors"][name] or group["fixed_metric_missing"][name]
            for name in ("ICov-encoding", "ICov-type", "CCov")
        )
        if fixed_metric_error:
            group["reasons"].add("fixed-isa-metric-input-invalid")
        guest_conflict = include_guest_metrics and (
            group["profile_conflict"] or group["identity_conflict"]
            or group["pcov_universe_conflict"]
        )
        if guest_conflict or pcov_error or fixed_metric_error:
            aggregate_status = "gap"
            if group["profile_conflict"]:
                group["reasons"].add("isa-profile-mismatch")
            if group["identity_conflict"]:
                group["reasons"].add("target-identity-mismatch")
        pcov_eligible_units = {
            f"{artifact}@0x{pc:x}"
            for artifact, pcs in group["pcov_eligible_by_artifact"].items()
            for pc in pcs
        }
        pcov_covered_units = {
            f"{artifact}@0x{pc:x}"
            for artifact, pcs in group["pcov_covered_by_artifact"].items()
            for pc in pcs
        }
        guest = {}
        profile_id = next(iter(group["profile_ids"])) if len(group["profile_ids"]) == 1 else None
        registry_sha = next(iter(group["registry_shas"])) if len(group["registry_shas"]) == 1 else None
        for name in GUEST_METRICS if include_guest_metrics else ():
            bucket = group["guest"].get(name)
            if name == "PCov":
                guest[name] = _summarize_pcov_run(
                    pcov_covered_units, pcov_eligible_units, observation_status,
                    identity_missing_records=group["pcov_identity_missing_records"],
                    static_map_missing_records=group["pcov_static_map_missing_records"],
                    invalid_unit_records=group["pcov_invalid_unit_records"],
                )
                guest[name]["coverage_profile_id"] = profile_id
                guest[name]["coverage_registry_sha256"] = registry_sha
                if group["pcov_universe_conflict"]:
                    guest[name].update(
                        status="gap", value=None, reason="pcov-static-map-conflict",
                    )
                elif group["pcov_missing_metric_records"]:
                    guest[name].update(
                        status="gap", value=None, reason="pcov-metric-missing",
                    )
                elif pcov_error or group["identity_conflict"]:
                    guest[name].update(
                        status="gap", value=None,
                        reason=guest[name].get("reason") or (
                            "target-identity-mismatch" if group["identity_conflict"]
                            else "pcov-coverage-gap"
                        ),
                    )
                continue
            canonical_metrics = group["canonical_metrics"]
            expected_units = (
                set(canonical_metrics[name])
                if isinstance(canonical_metrics, dict) and name in canonical_metrics
                else set()
            )
            bucket = group["guest"].get(name)
            covered_units = set(bucket["covered"]) if bucket else set()
            statuses = group["fixed_metric_statuses"][name]
            if group["profile_conflict"] or group["identity_conflict"]:
                metric_status = "gap"
                metric_reason = sorted(group["reasons"])[0]
            elif group["fixed_metric_errors"][name] or group["fixed_metric_missing"][name]:
                metric_status = "gap"
                metric_reason = f"{name}-profile-universe-or-unit-set-invalid"
            elif not expected_units:
                metric_status = "gap" if observation_status in {"observed", "partial"} else "NA"
                metric_reason = "coverage-profile-or-registry-missing"
            elif statuses and all(item == "NA" for item in statuses):
                metric_status = "NA"
                metric_reason = "no-observed-guest-trace"
            elif observation_status == "observed" and statuses and all(
                item == "observed" for item in statuses
            ):
                metric_status = "observed"
                metric_reason = None
            elif observation_status in {"partial", "gap"}:
                metric_status = observation_status
                metric_reason = (
                    "batch-observation-incomplete" if metric_status == "partial"
                    else "batch-observation-gap"
                )
            else:
                metric_status = "partial"
                metric_reason = "batch-observation-incomplete"
            metric = _metric_value(covered_units & expected_units, expected_units)
            metric["status"] = metric_status
            metric["value"] = round(len(covered_units & expected_units) / len(expected_units), 6) \
                if expected_units and metric_status == "observed" else None
            metric["covered_units"] = sorted(covered_units & expected_units, key=str)
            metric["eligible_units"] = sorted(expected_units, key=str)
            metric["coverage_profile_id"] = profile_id
            metric["registry_sha256"] = _pcov_unit_digest(expected_units) if expected_units else None
            if metric_reason:
                metric["reason"] = metric_reason
            guest[name] = metric

        case_ids = sorted(group["case_ids"])
        cohort_id = hashlib.sha256(json.dumps(
            case_ids, separators=(",", ":")
        ).encode()).hexdigest() if case_ids else None
        target_identity_id = next(iter(group["target_identity_ids"]), None)
        rows.append({
            "method": method or None, "route": route or None, "lane": lane or None,
            "stratum": stratum or None, "target": target,
            "target_identity_id": target_identity_id,
            "cases": group["cases"], "observed_cases": group["observed_cases"],
            "incomplete_cases": group["incomplete_cases"], "status": aggregate_status,
            "reason": sorted(group["reasons"])[0] if group["reasons"] else None,
            "coverage_basis": ({
                "PCov": "run-union-elf-pc",
                "ICov-encoding": "fixed-profile-universe",
                "ICov-type": "fixed-profile-universe", "CCov": "fixed-profile-universe",
                "coverage_profile_id": profile_id,
                "coverage_registry_sha256": registry_sha,
                "cohort_id": cohort_id, "cohort_cases": len(case_ids),
            } if include_guest_metrics else {
                "SimSrcCov": "target-run-source-profile",
                "cohort_id": cohort_id, "cohort_cases": len(case_ids),
            }),
            "simulator_event_counts": dict(sorted(group["events"].items())),
            "guest": guest,
            "translator_coverage": {
                "covered": None, "eligible": None, "value": None,
                "status": "NA", "reason": "translator-feature-channel-unavailable",
            },
        })
    row_statuses = {row["status"] for row in rows}
    status = ("observed" if row_statuses == {"observed"} else
              "gap" if row_statuses == {"gap"} else
              "NA" if not row_statuses or row_statuses == {"NA"} else "partial")
    cohorts = {row["coverage_basis"]["cohort_id"] for row in rows}
    comparability = ("paired-corpus" if len({row["target"] for row in rows}) > 1
                     and cohorts and None not in cohorts and len(cohorts) == 1
                     else "target-local-fixed-profile-universe"
                     if include_guest_metrics else "target-local-source-profile")
    source_rows = [{
        "method": method or None, "route": route or None, "lane": lane or None,
        "target": target, "stratum": stratum or None, "scope": "target-run",
        "simulator_source_coverage": source,
    } for (method, route, lane, stratum, target), source in sorted(source_targets.items())]
    experiment_unions = []
    for method_key in sorted({key[0] for key in groups}):
        method_groups = [
            group for key, group in groups.items() if key[0] == method_key
        ]
        method_rows = [row for row in rows if (row.get("method") or "") == method_key]
        method_sources = [
            item for item in source_rows
            if (item.get("method") or "") == method_key
        ]
        if include_guest_metrics:
            experiment_unions.append(summarize_experiment_union(
                method_key or None,
                method_rows,
                profile_ids={
                    profile_id for group in method_groups for profile_id in group["profile_ids"]
                },
                case_artifact_pairs=len({
                    case_id for group in method_groups for case_id in group["case_ids"]
                }),
            ))
        else:
            experiment_unions.append(summarize_source_experiment_union(
                method_key or None, method_rows, source_rows=method_sources,
                case_artifact_pairs=len({
                    case_id for group in method_groups for case_id in group["case_ids"]
                }),
            ))
    summary = {"schema_version": BATCH_SCHEMA, "status": status,
               "comparability": comparability, "experiment_unions": experiment_unions,
               "targets": rows, "source_targets": source_rows,
               "guest_metrics_enabled": include_guest_metrics}
    if include_guest_metrics:
        summary = aggregate_rv_instruction_summary(summary, rv_records)
        summary = aggregate_opcode_catalog_summary(summary, opcode_catalog_records)
    return summary


def _in_scope(source: str, scopes: list[str], source_root: str | None = None,
              excludes: list[str] | None = None) -> bool:
    """按路径段匹配；scopes 任一命中即纳入，excludes 命中即排除。"""
    # 相对路径补仓库名；构建机绝对路径（含 Windows 盘符）保持原样，
    # 否则外部依赖（如 protobuf-net）会被误算进本仓库分母。
    source = posixpath.normpath(source.replace("\\", "/"))
    rooted = source.startswith("/") or (len(source) > 1 and source[1] == ":")
    if source_root and not rooted:
        if source == ".." or source.startswith("../"):
            return False
        root = posixpath.normpath(str(source_root).replace("\\", "/").strip())
        root_name = root.rstrip("/").rsplit("/", 1)[-1]
        source = f"{root_name}/{source.removeprefix('./')}"
    path = "/" + posixpath.normpath(source.replace("\\", "/")).strip("/") + "/"

    def listed(items):
        return any(
            "/" + posixpath.normpath(item.replace("\\", "/")).strip("/") + "/" in path
            for item in items
        )

    return bool(scopes) and listed(scopes) and not listed(excludes or [])


def _source_key(source: str, source_root: str | None = None) -> str:
    """把同一源码的相对/绝对 LCOV 路径归一到一个统计单元。"""
    value = posixpath.normpath(str(source).replace("\\", "/").strip())
    rooted = value.startswith("/") or (len(value) > 1 and value[1] == ":")
    root = (
        posixpath.normpath(str(source_root).replace("\\", "/").strip())
        if source_root else ""
    )
    root_name = root.rstrip("/").rsplit("/", 1)[-1] if root else ""
    if root_name and not rooted and (value == ".." or value.startswith("../")):
        value = posixpath.normpath(posixpath.join(root, value))
        rooted = True
    parts = [part for part in value.split("/") if part and part != "."]
    if not parts:
        return ""
    if root_name:
        root_parts = [part for part in root.split("/")
                      if part and part != "."]
        if (root_parts and len(parts) >= len(root_parts)
                and parts[:len(root_parts)] == root_parts):
            relative = parts[len(root_parts):]
            return "/".join([root_name, *relative])
        # Build paths often replace the absolute prefix but retain the
        # project directory.  Use its first occurrence: choosing the last
        # occurrence drops real nested components such as
        # ``libriscv/lib/libriscv/foo.cpp``.
        for index, part in enumerate(parts):
            if part == root_name:
                return "/".join(parts[index:])
        if not rooted:
            return posixpath.normpath(f"{root_name}/{'/'.join(parts)}")
    return "/".join(parts)


def _empty_source_metrics(reason: str, *, status: str = "gap") -> dict[str, dict[str, Any]]:
    """Keep function/line/branch gaps explicit instead of omitting dimensions."""
    return {
        name: _metric(0, 0, status=status, reason=reason)
        for name in ("function", "line", "branch")
    }


def summarize_lcov(profile: Path, scopes: list[str], source_root: str | None = None,
                  excludes: list[str] | None = None, *,
                  include_units: bool = False,
                  units_only: bool = False) -> dict[str, Any]:
    # source_scope_exclude 同时移除 covered 和 eligible；这样剔除无关架构
    # 不会改变保留源码的命中分子，也不会制造 >100% 的比例。
    functions: dict[tuple[str, str], int] = {}
    lines: dict[tuple[str, str], int] = {}
    branches: dict[tuple[str, str], int] = {}
    eligible_functions: set[tuple[str, str]] = set()
    eligible_lines: set[tuple[str, str]] = set()
    eligible_branches: set[tuple[str, str]] = set()
    eligible_sources: set[str] = set()
    # Keep the complete LCOV universe and its hits as an audit trail.  The
    # reported numerator is the intersection with the configured scope; the
    # excluded hit count makes an intentional scope reduction reviewable.
    profile_functions: set[tuple[str, str]] = set()
    profile_lines: set[tuple[str, str]] = set()
    profile_branches: set[tuple[str, str]] = set()
    function_scope: dict[tuple[str, str], bool] = {}
    line_scope: dict[tuple[str, str], bool] = {}
    branch_scope: dict[tuple[str, str], bool] = {}
    record_summaries: list[dict[str, Any]] = []
    raw_branch_records: list[tuple[int, str, bool, str, str, str, int]] = []
    record_id = -1
    scope_unit_conflict = False
    source = None
    source_key = ""
    eligible = False
    function_keys: dict[str, list[tuple[str, str]]] = {}
    function_cursor: dict[str, int] = {}
    records_seen = False
    record_open = False
    profile_incomplete = False
    profile_error: str | None = None

    def note_scope(mapping: dict[tuple[str, str], bool],
                   key: tuple[str, str], in_scope: bool) -> None:
        nonlocal scope_unit_conflict
        previous = mapping.get(key)
        if previous is not None and previous != in_scope:
            scope_unit_conflict = True
        else:
            mapping[key] = in_scope

    try:
        for raw in profile.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.startswith("SF:"):
                if record_open:
                    profile_incomplete = True
                source = raw[3:].strip()
                source_key = _source_key(source, source_root)
                eligible = bool(source and _in_scope(source, scopes, source_root, excludes))
                record_id += 1
                record_summaries.append({
                    "record_id": record_id,
                    "source_key": source_key,
                    "eligible": eligible,
                    "FNF": None, "FNH": None,
                    "LF": None, "LH": None,
                    "BRF": None, "BRH": None,
                })
                function_keys = {}
                function_cursor = {}
                record_open = True
                if eligible:
                    eligible_sources.add(source_key)
            elif raw == "end_of_record":
                if not record_open:
                    profile_incomplete = True
                source = None
                source_key = ""
                eligible = False
                function_keys = {}
                function_cursor = {}
                record_open = False
            elif not source:
                continue
            elif raw.startswith("FN:"):
                records_seen = True
                line, name = raw[3:].split(",", 1)
                # Function names are not unique in C#/C++ LCOV (overloads and
                # generated accessors are common).  Keep the declaration line
                # in the unit key; FNDA entries are paired in declaration order
                # below so no hit is merged into a sibling overload.
                key = (source_key, f"{line}:{name}")
                function_keys.setdefault(name, []).append(key)
                profile_functions.add(key)
                note_scope(function_scope, key, eligible)
                if eligible:
                    functions.setdefault(key, 0)
                    eligible_functions.add(key)
            elif raw.startswith("FNDA:"):
                records_seen = True
                count, name = raw[5:].split(",", 1)
                candidates = function_keys.get(name, [])
                cursor = function_cursor.get(name, 0)
                if cursor < len(candidates):
                    key = candidates[cursor]
                    function_cursor[name] = cursor + 1
                else:
                    # Preserve a malformed/compact FNDA-only record instead
                    # of silently dropping a potentially covered function.
                    key = (source_key, f"name:{name}")
                profile_functions.add(key)
                note_scope(function_scope, key, eligible)
                functions[key] = max(functions.get(key, 0), _hit_count(count))
                if eligible:
                    eligible_functions.add(key)
            elif raw.startswith("DA:"):
                records_seen = True
                line, count, *_ = raw[3:].split(",")
                key = (source_key, line)
                lines[key] = max(lines.get(key, 0), _hit_count(count))
                profile_lines.add(key)
                note_scope(line_scope, key, eligible)
                if eligible:
                    eligible_lines.add(key)
            elif raw.startswith("BRDA:"):
                records_seen = True
                line, block, branch, taken = raw[5:].split(",", 3)
                hits = 0 if taken == "-" else _hit_count(taken)
                raw_branch_records.append(
                    (record_id, source_key, eligible, line, block, branch, hits)
                )
            elif raw.startswith(("FNF:", "FNH:", "LF:", "LH:", "BRF:", "BRH:")):
                records_seen = True
                field = raw.split(":", 1)[0]
                record_summaries[-1][field] = _hit_count(
                    raw.split(":", 1)[1].strip()
                )
    except (OSError, UnicodeError, ValueError) as exc:
        # Keep all units parsed before the bad/truncated record.  Returning an
        # empty metric here used to erase valid numerator evidence and made a
        # profile defect look like a zero coverage result.
        profile_error = f"{type(exc).__name__}: {exc}"
    if record_open:
        profile_incomplete = True
    if not records_seen:
        reason = (
            "source-coverage-profile-invalid" if profile_error else
            "source-coverage-profile-truncated" if profile_incomplete else
            "source-coverage-records-missing"
        )
        result = {"schema_version": SOURCE_SCHEMA, "status": "gap",
                  "reason": reason,
                  **({"error": profile_error} if profile_error else {}),
                  "metrics": _empty_source_metrics(reason)}
        if include_units:
            result["unit_sets"] = {
                name: {"eligible": [], "covered": []}
                for name in ("function", "line", "branch")
            }
        return result

    # A .NET report may contain several records for the same physical source
    # file.  Function and line keys already collapse those records, but
    # branch identifiers are report-local and are therefore not stable across
    # duplicate records.  Normalize a duplicate file's branches by the
    # ordinal within each (line, block) group, retaining the maximum hit.
    source_record_ids: dict[str, set[int]] = {}
    source_has_eligible_record: set[str] = set()
    for record in record_summaries:
        source_record_ids.setdefault(record["source_key"], set()).add(
            int(record["record_id"])
        )
        if record.get("eligible"):
            source_has_eligible_record.add(record["source_key"])
    duplicate_sources = {
        source_key for source_key, ids in source_record_ids.items()
        if len(ids) > 1 and source_key in source_has_eligible_record
    }
    branch_ordinals: dict[tuple[int, str, str, str], int] = {}
    for (current_record_id, current_source_key, current_eligible,
         line, block, branch, hits) in raw_branch_records:
        if current_source_key in duplicate_sources:
            group = (current_record_id, current_source_key, line, block)
            ordinal = branch_ordinals.get(group, 0)
            branch_ordinals[group] = ordinal + 1
            unit = f"{line},{block},{ordinal}"
        else:
            unit = f"{line},{block},{branch}"
        key = (current_source_key, unit)
        branches[key] = max(branches.get(key, 0), hits)
        profile_branches.add(key)
        note_scope(branch_scope, key, current_eligible)
        if current_eligible:
            eligible_branches.add(key)

    profile_covered = {
        "function": {key for key, value in functions.items() if value > 0},
        "line": {key for key, value in lines.items() if value > 0},
        "branch": {key for key, value in branches.items() if value > 0},
    }
    raw_metric_values = {
        "function": (len(profile_covered["function"] & eligible_functions),
                     len(eligible_functions)),
        "line": (len(profile_covered["line"] & eligible_lines),
                  len(eligible_lines)),
        "branch": (len(profile_covered["branch"] & eligible_branches),
                   len(eligible_branches)),
    }
    summary_fields = {
        "function": ("FNH", "FNF"),
        "line": ("LH", "LF"),
        "branch": ("BRH", "BRF"),
    }
    eligible_records = [record for record in record_summaries
                        if record.get("eligible")]
    metrics = {}
    metric_digest_records: dict[str, list[dict[str, Any]]] = {}
    profile_digest_records: dict[str, list[dict[str, Any]]] = {}
    metric_basis: dict[str, str] = {}
    for name, (covered_field, eligible_field) in summary_fields.items():
        summary_usable = bool(eligible_records) and not duplicate_sources \
            and all(record.get(covered_field) is not None
                    and record.get(eligible_field) is not None
                    for record in eligible_records)
        profile_summary_usable = bool(record_summaries) and not duplicate_sources \
            and all(record.get(covered_field) is not None
                    and record.get(eligible_field) is not None
                    for record in record_summaries)
        if summary_usable:
            covered = sum(int(record[covered_field]) for record in eligible_records)
            eligible_count = sum(int(record[eligible_field]) for record in eligible_records)
            metric_digest_records[name] = eligible_records
            metric_basis[name] = "lcov-file-summary"
        else:
            covered, eligible_count = raw_metric_values[name]
            metric_basis[name] = (
                "lcov-records-deduplicated-by-source-location"
                if duplicate_sources else "lcov-records"
            )
        if profile_summary_usable:
            profile_digest_records[name] = record_summaries
        metrics[name] = _metric(covered, eligible_count)
    unit_sets = {
        "function": (eligible_functions,
                     profile_covered["function"] & eligible_functions),
        "line": (eligible_lines,
                  profile_covered["line"] & eligible_lines),
        "branch": (eligible_branches,
                   profile_covered["branch"] & eligible_branches),
    }
    if units_only:
        # SimSrcCov growth only needs the stable case-level unit sets.  The
        # final summary path still computes metrics, digests, excluded counts
        # and profile audit fields below; skipping those here avoids doing the
        # same reduction twice for every Renode case.
        available = [item for item in unit_sets.values() if item[0]]
        status = ("gap" if profile_error or scope_unit_conflict else
                  "partial" if profile_incomplete else
                  "observed" if len(available) == 3 else
                  "partial" if available else "NA")
        reason = ("source-coverage-profile-invalid" if profile_error else
                  "source-coverage-profile-truncated" if profile_incomplete else
                  "source-coverage-scope-unit-conflict" if scope_unit_conflict else
                  None if status == "observed" else
                  "source-coverage-dimension-missing" if status == "partial" else
                  "source-coverage-scope-empty")
        result = {"schema_version": SOURCE_SCHEMA, "status": status,
                  "source_files": len(eligible_sources),
                  **({"error": profile_error} if profile_error else {}),
                  "reason": reason,
                  "unit_sets": {
                      name: {
                          "eligible": [list(unit) for unit in sorted(eligible_units)],
                          "covered": [list(unit) for unit in sorted(covered_units)],
                      }
                      for name, (eligible_units, covered_units) in unit_sets.items()
                  }}
        return result
    profile_units = {
        "function": profile_functions,
        "line": profile_lines,
        "branch": profile_branches,
    }
    for name, (eligible_units, covered_units) in unit_sets.items():
        if name in metric_digest_records:
            metrics[name]["eligible_units_sha256"] = _source_summary_digest(
                metric_digest_records[name], summary_fields[name][1]
            )
            metrics[name]["covered_units_sha256"] = _source_summary_digest(
                metric_digest_records[name], summary_fields[name][0]
            )
        else:
            metrics[name]["eligible_units_sha256"] = _source_unit_digest(eligible_units)
            metrics[name]["covered_units_sha256"] = _source_unit_digest(covered_units)
        metrics[name]["metric_basis"] = metric_basis[name]
        if name in profile_digest_records:
            records = profile_digest_records[name]
            covered_field, eligible_field = summary_fields[name]
            excluded_records = [record for record in records
                                if not record.get("eligible")]
            metrics[name]["profile_units"] = sum(
                int(record[eligible_field]) for record in records
            )
            metrics[name]["profile_covered"] = sum(
                int(record[covered_field]) for record in records
            )
            metrics[name]["excluded_units"] = sum(
                int(record[eligible_field]) for record in excluded_records
            )
            metrics[name]["excluded_covered"] = sum(
                int(record[covered_field]) for record in excluded_records
            )
            metrics[name]["profile_units_sha256"] = _source_summary_digest(
                records, eligible_field
            )
            metrics[name]["profile_covered_units_sha256"] = _source_summary_digest(
                records, covered_field
            )
            metrics[name]["excluded_units_sha256"] = _source_summary_digest(
                excluded_records, eligible_field
            )
            metrics[name]["excluded_covered_units_sha256"] = _source_summary_digest(
                excluded_records, covered_field
            )
        else:
            metrics[name]["profile_units"] = len(profile_units[name])
            metrics[name]["profile_covered"] = len(profile_covered[name])
            metrics[name]["excluded_units"] = len(profile_units[name] - eligible_units)
            metrics[name]["excluded_covered"] = len(profile_covered[name] - covered_units)
            metrics[name]["profile_units_sha256"] = _source_unit_digest(profile_units[name])
            metrics[name]["profile_covered_units_sha256"] = _source_unit_digest(
                profile_covered[name]
            )
            metrics[name]["excluded_units_sha256"] = _source_unit_digest(
                profile_units[name] - eligible_units
            )
            metrics[name]["excluded_covered_units_sha256"] = _source_unit_digest(
                profile_covered[name] - covered_units
            )
        if profile_error:
            metrics[name].update(
                status="gap", value=None,
                reason="source-coverage-profile-invalid",
            )
        elif profile_incomplete:
            metrics[name].update(
                status="partial", value=None,
                reason="source-coverage-profile-truncated",
            )
    available = [item for item in metrics.values() if item["eligible"]]
    status = ("gap" if profile_error or scope_unit_conflict
              or any(item["status"] == "gap" for item in metrics.values()) else
              "partial" if profile_incomplete else
              "observed" if len(available) == 3 else
              "partial" if available else "NA")
    reason = ("source-coverage-profile-invalid" if profile_error else
              "source-coverage-profile-truncated" if profile_incomplete else
              "source-coverage-scope-unit-conflict" if scope_unit_conflict else
              "source-coverage-covered-exceeds-eligible-after-exclusion"
              if status == "gap" and any(
                  item.get("reason") == "covered-exceeds-eligible-after-exclusion"
                  for item in metrics.values()
              ) else None if status == "observed" else
              "source-coverage-dimension-missing" if status == "partial" else
              "source-coverage-scope-empty")
    result = {"schema_version": SOURCE_SCHEMA, "status": status,
              "source_files": len(eligible_sources),
              **({"error": profile_error} if profile_error else {}),
              "metrics": metrics, "reason": reason}
    if include_units:
        result["unit_sets"] = {
            name: {
                "eligible": [list(unit) for unit in sorted(eligible_units)],
                "covered": [list(unit) for unit in sorted(covered_units)],
            }
            for name, (eligible_units, covered_units) in unit_sets.items()
        }
    return result


def _source_path(out: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else out / path


def _artifact_ref(out: Path, path: Path | None) -> str | None:
    """Store run-owned artifacts as paths relative to the run root."""
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(out.resolve()))
    except (OSError, ValueError):
        return str(path)


def target_source_coverage(out: Path, target: dict[str, Any],
                           identity: dict[str, Any] | None = None,
                           coverage_binary_sha: str | None = None,
                           source_commit_observed: str | None = None,
                           coverage_binary_sha256: str | None = None,
                           collection_status: str | None = None,
                           collection_reason: str | None = None) -> dict[str, Any]:
    spec = target.get("source_coverage") if isinstance(target.get("source_coverage"), dict) else {}
    profile = _source_path(out, spec.get("profile"))
    source_identity = {}
    identity_path = _source_path(out, spec.get("identity"))
    if identity_path and identity_path.is_file():
        try:
            loaded_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if isinstance(loaded_identity, dict):
                source_identity = loaded_identity
        except (OSError, TypeError, ValueError):
            pass
    observed_source_commit = (
        source_commit_observed
        or coverage_binary_sha
        or (identity or {}).get("source_commit_observed")
        or (identity or {}).get("coverage_binary_sha")
    )
    expected_source_commit = spec.get("source_commit")
    source_commit_match = (
        observed_source_commit == expected_source_commit
        if observed_source_commit and expected_source_commit else None
    )
    observed_binary_sha256 = (
        coverage_binary_sha256
        or (identity or {}).get("coverage_binary_sha256")
    )
    expected_binary_sha256 = (
        spec.get("coverage_binary_sha256")
        or target.get("coverage_binary_sha256")
    )
    binary_sha256_match = (
        isinstance(observed_binary_sha256, str)
        and isinstance(expected_binary_sha256, str)
        and observed_binary_sha256.lower() == expected_binary_sha256.lower()
        if observed_binary_sha256 and expected_binary_sha256 else None
    )
    def identity_value(name: str):
        if isinstance(identity, dict) and name in identity:
            return identity[name]
        return source_identity.get(name)

    target_commit = target.get("commit")
    declared_target_commit = spec.get("target_commit")
    source_commit = spec.get("source_commit") or (identity or {}).get("source_commit")
    metadata = {"backend": target.get("id"), "source_repository": spec.get("source_repository"),
                "source_commit": source_commit,
                "target_commit": declared_target_commit or target_commit,
                "coverage_binary": target.get("coverage_binary"),
                "source_commit_observed": observed_source_commit,
                "source_commit_expected": expected_source_commit,
                "source_commit_match": source_commit_match,
                "source_tree_status": identity_value("source_tree_status"),
                "source_tree_dirty": identity_value("source_tree_dirty"),
                "source_tree_drift_reason": identity_value("source_tree_drift_reason"),
                "non_generated_dirty_paths": identity_value("non_generated_dirty_paths"),
                "source_dirty_path_count": identity_value("source_dirty_path_count"),
                "generated_untracked_path_count": identity_value("generated_untracked_path_count"),
                "coverage_binary_sha256": observed_binary_sha256,
                "coverage_binary_expected_sha256": expected_binary_sha256,
                "coverage_binary_sha256_match": binary_sha256_match,
                # 兼容旧 summary 字段；其值是源码提交身份，不是文件摘要。
                "coverage_binary_sha": observed_source_commit,
                "coverage_binary_expected_sha": expected_source_commit,
                "coverage_binary_sha_match": source_commit_match,
                "toolchain": spec.get("toolchain"), "build_flags": spec.get("build_flags", []),
                "collector": spec.get("collector"),
                "collector_input_format": spec.get("collector_input_format"),
                "collector_pipeline": source_identity.get("collector_pipeline"),
                "collector_tools": source_identity.get("collector_tools"),
                "instrumented_binaries": source_identity.get("instrumented_binaries", []),
                "source_scope": spec.get("source_scope", []), "source_scope_exclude": spec.get("source_scope_exclude", []), "source_root": spec.get("source_root"),
                "profile_format": spec.get("format", "lcov")}
    provenance_warnings = set((identity or {}).get("provenance_warnings") or ())
    tree_warning = _source_tree_identity_warning(metadata)
    if tree_warning:
        provenance_warnings.add(tree_warning)
    if not observed_source_commit or not expected_source_commit or not target_commit:
        provenance_warnings.add("source-coverage-source-commit-identity-unavailable")
    if _source_commit_proven_mismatch(metadata) or (
            declared_target_commit and target_commit
            and declared_target_commit != target_commit):
        provenance_warnings.add("source-coverage-source-commit-mismatch")
    metadata["provenance_warnings"] = sorted(provenance_warnings)
    if collection_status is not None and collection_status not in {"observed", "partial"}:
        reason = collection_reason or f"source-coverage-collection-{collection_status}"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason,
                "profile_path": _artifact_ref(out, profile),
                "collection_status": collection_status, "identity": metadata,
                "metrics": _empty_source_metrics(reason)}
    collection_partial = collection_status == "partial"
    if not target.get("coverage_binary"):
        reason = "coverage-binary-not-explicit"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    if not metadata["coverage_binary_expected_sha256"]:
        reason = "source-coverage-binary-sha-not-declared"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    if not metadata["coverage_binary_sha256_match"]:
        reason = "source-coverage-binary-sha-unavailable" if not observed_binary_sha256 else (
            "source-coverage-binary-sha-mismatch"
        )
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    if (spec.get("instrumented_binaries")
            and source_identity.get("instrumented_binaries_match") is not True):
        entries = source_identity.get("instrumented_binaries", [])
        if any(not item.get("expected_sha256") for item in entries
               if isinstance(item, dict)):
            reason = "source-coverage-instrumented-binary-sha-not-declared"
        elif any(not item.get("sha256") for item in entries
                 if isinstance(item, dict)):
            reason = "source-coverage-instrumented-binary-sha-unavailable"
        else:
            reason = "source-coverage-instrumented-binary-sha-mismatch"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    source_commit_mismatch = (
        _source_commit_proven_mismatch(metadata)
        or (declared_target_commit and target_commit
            and declared_target_commit != target_commit)
        or (source_commit and target_commit and source_commit != target_commit)
    )
    if source_commit_mismatch:
        reason = "source-coverage-source-commit-mismatch"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    if profile is None:
        reason = "source-coverage-profile-not-configured"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "NA",
                "reason": reason, "identity": metadata,
                "metrics": _empty_source_metrics(reason, status="NA")}
    if not profile.is_file():
        reason = "source-coverage-profile-missing"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile), "identity": metadata,
                "metrics": _empty_source_metrics(reason)}
    if str(spec.get("format", "lcov")).lower() != "lcov":
        reason = "source-coverage-format-unsupported"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile),
                "identity": metadata, "metrics": _empty_source_metrics(reason)}
    def paths(key):
        return [str(item).replace("\\", "/").strip("/")
                for item in spec.get(key, []) if str(item).strip()]

    scopes, excludes = paths("source_scope"), paths("source_scope_exclude")
    if not scopes:
        reason = "source-scope-not-declared"
        return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                "reason": reason, "profile_path": _artifact_ref(out, profile), "identity": metadata,
                "metrics": _empty_source_metrics(reason)}
    parsed = summarize_lcov(profile, scopes, metadata["source_root"], excludes)
    if (parsed.get("status") == "partial" and spec.get("collector") == "dotnet"
            and not parsed.get("metrics", {}).get("branch", {}).get("eligible")):
        parsed["reason"] = "dotnet-coverage-branch-edge-unavailable"
    if collection_partial and parsed.get("status") in {"observed", "partial"}:
        parsed["collection_status"] = "partial"
        parsed["collection_reason"] = collection_reason
        parsed["status"] = "partial"
        if collection_reason:
            parsed["reason"] = collection_reason
        for metric in parsed.get("metrics", {}).values():
            if metric.get("status") == "observed":
                metric["status"] = "partial"
                metric["reason"] = collection_reason or "source-coverage-collection-partial"
    identity_path = _source_path(out, spec.get("identity"))
    if identity_path:
        if not identity_path.is_file():
            metadata["provenance_warnings"] = sorted(set(
                [*metadata["provenance_warnings"], "source-coverage-profile-identity-unavailable"]
            ))
        else:
            try:
                profile_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                profile_identity = None
            if not isinstance(profile_identity, dict):
                metadata["provenance_warnings"] = sorted(set(
                    [*metadata["provenance_warnings"], "source-coverage-profile-identity-invalid"]
                ))
            else:
                profile_source = profile_identity.get("source_commit")
                profile_target = profile_identity.get("target_commit")
                if not profile_source or not profile_target:
                    metadata["provenance_warnings"] = sorted(set(
                        [*metadata["provenance_warnings"],
                         "source-coverage-profile-commit-identity-unavailable"]
                    ))
                elif profile_source != profile_target:
                    reason = "source-coverage-source-commit-mismatch"
                    return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                            "reason": reason, "profile_path": _artifact_ref(out, profile),
                            "profile_identity_path": _artifact_ref(out, identity_path), "identity": metadata,
                            "metrics": _empty_source_metrics(reason),
                            "profile_identity": profile_identity}
                if profile_source and source_commit and profile_source != source_commit:
                    reason = "source-coverage-source-commit-mismatch"
                    return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                            "reason": reason, "profile_path": _artifact_ref(out, profile),
                            "profile_identity_path": _artifact_ref(out, identity_path), "identity": metadata,
                            "metrics": _empty_source_metrics(reason),
                            "profile_identity": profile_identity}
                if any(name not in profile_identity
                       or profile_identity.get(name) != metadata.get(name)
                       for name in ("source_scope", "source_scope_exclude", "source_root")):
                    reason = "source-coverage-profile-scope-mismatch"
                    return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                            "reason": reason, "profile_path": _artifact_ref(out, profile),
                            "profile_identity_path": _artifact_ref(out, identity_path), "identity": metadata,
                            "metrics": _empty_source_metrics(reason),
                            "profile_identity": profile_identity}
                profile_binary_sha = profile_identity.get("coverage_binary_sha256")
                if not profile_binary_sha:
                    metadata["provenance_warnings"] = sorted(set(
                        [*metadata["provenance_warnings"],
                         "source-coverage-profile-binary-identity-unavailable"]
                    ))
                elif (isinstance(profile_binary_sha, str)
                      and isinstance(metadata["coverage_binary_sha256"], str)
                      and profile_binary_sha.lower()
                      != metadata["coverage_binary_sha256"].lower()):
                    reason = "source-coverage-binary-sha-mismatch"
                    return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "status": "gap",
                            "reason": reason, "profile_path": _artifact_ref(out, profile),
                            "profile_identity_path": _artifact_ref(out, identity_path), "identity": metadata,
                            "metrics": _empty_source_metrics(reason),
                            "profile_identity": profile_identity}
    return {"schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov", "profile_path": _artifact_ref(out, profile),
            "identity": metadata, **parsed}

def self_check() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lcov = root / "coverage.info"
        lcov.write_text("TN:\nSF:src/sim.c\nFN:1,main\nFNDA:1,main\nDA:1,1\nBRDA:1,0,0,1\nend_of_record\n", encoding="utf-8")
        source = summarize_lcov(lcov, ["src"])
        assert source["status"] == "observed" and source["metrics"]["line"]["value"] == 1.0
        duplicate_lcov = root / "duplicate.info"
        duplicate_lcov.write_text(
            "SF:src/sim.c\nFN:1,main\nFNDA:0,main\nDA:1,0\n"
            "BRDA:1,0,0,-\nend_of_record\n"
            "SF:src/sim.c\nFN:1,main\nFNDA:3,main\nDA:1,2\n"
            "BRDA:1,0,0,2\nend_of_record\n",
            encoding="utf-8",
        )
        duplicate = summarize_lcov(duplicate_lcov, ["src"])
        assert all(duplicate["metrics"][name]["covered"] == 1
                   for name in ("function", "line", "branch"))
        overload_lcov = root / "overload.info"
        overload_lcov.write_text(
            "SF:src/overload.c\nFN:10,run()\nFN:20,run()\n"
            "FNDA:0,run()\nFNDA:2,run()\nFNF:2\nFNH:1\n"
            "DA:10,0\nDA:20,1\nBRDA:20,0,0,1\nend_of_record\n",
            encoding="utf-8",
        )
        overload = summarize_lcov(overload_lcov, ["src"])
        assert overload["metrics"]["function"]["eligible"] == 2
        assert overload["metrics"]["function"]["covered"] == 1
        assert overload["metrics"]["function"]["value"] == 0.5
        aliases = root / "aliases.info"
        aliases.write_text(
            "SF:src/sim.c\nDA:1,1\nend_of_record\n"
            "SF:/tmp/project/src/sim.c\nDA:1,0\nend_of_record\n",
            encoding="utf-8",
        )
        alias_summary = summarize_lcov(aliases, ["project/src"], "/tmp/project")
        assert alias_summary["metrics"]["line"]["covered"] == 1
        assert alias_summary["metrics"]["line"]["eligible"] == 1
        relative = summarize_lcov(lcov, ["project/src"], "/tmp/project")
        # exclude 同时移除命中分子和资格分母；保留源码的覆盖结果不受影响。
        scoped = root / "scoped.info"
        scoped.write_text(
            "SF:project/src/sim.c\nDA:1,1\nend_of_record\n"
            "SF:project/src/vendor/lib.c\nDA:1,1\nend_of_record\n",
            encoding="utf-8",
        )
        unexcluded = summarize_lcov(scoped, ["project/src"])
        kept = summarize_lcov(scoped, ["project/src"], None, ["project/src/vendor"])
        assert kept["metrics"]["line"]["covered"] == 1
        assert kept["metrics"]["line"]["eligible"] == 1
        assert kept["metrics"]["line"]["value"] == 1.0
        assert kept["metrics"]["line"]["profile_covered"] == 2
        assert kept["metrics"]["line"]["excluded_covered"] == 1
        assert kept["status"] == "partial"
        all_excluded = summarize_lcov(
            scoped, ["project/src"], None, ["project/src"],
        )
        assert all_excluded["status"] == "NA"
        assert all_excluded["metrics"]["line"]["covered"] == 0
        assert all_excluded["metrics"]["line"]["eligible"] == 0
        incomplete_lcov = root / "incomplete.info"
        incomplete_lcov.write_text("SF:src/sim.c\nDA:1,1\n", encoding="utf-8")
        incomplete = summarize_lcov(incomplete_lcov, ["src"])
        assert (incomplete["status"] == "partial"
                and incomplete["reason"] == "source-coverage-profile-truncated")
        vendor = root / "vendor.libriscv.log"
        vendor.write_text("SF:project/vendor/lib.c" + chr(10) + "DA:1,1" + chr(10) + "end_of_record" + chr(10), encoding="utf-8")
        assert summarize_lcov(vendor, ["project"], None, ["project/vendor"])["status"] == "NA"
        # 构建机绝对路径（Windows 盘符）属于外部依赖，不能算进本仓库分母
        assert not _in_scope("C:\\Code\\pb-net\\x.cs", ["renode"], "/opt/renode")
        
        assert relative["status"] == "observed"
        mixed = root / "mixed.libriscv.log"
        mixed.write_bytes(
            b"f f_1000 pc 0x1000 instr 00000013\nRVOBS1\x00\x00"
            b"\nf f_1004 pc 0x1004 instr 00000013\n")
        mixed_scan = _scan_trace([mixed], "libriscv-translated", {}, {0x1000, 0x1004})
        assert mixed_scan[2] == 2 and mixed_scan[0] == [0x1000, 0x1004]
        boundary = root / "boundary.libriscv.log"
        boundary.write_bytes(
            b"x" * ((1 << 20) - 260) + b"\n"
            b"f f_2000 pc 0x2000 instr 00000013\n")
        boundary_scan = _scan_trace([boundary], "libriscv-translated", {}, {0x2000})
        assert boundary_scan[2] == 1 and boundary_scan[0] == [0x2000]
        crossing = root / "crossing.libriscv.log"
        line_start = (1 << 20) - 100
        crossing.write_bytes(
            b"x" * (line_start - 1) + b"\n"
            b"f f_3000 pc 0x3000 instr 00000013 " + b"x" * 200 + b"\n"
        )
        crossing_scan = _scan_trace([crossing], "libriscv-translated", {}, {0x3000})
        assert crossing_scan[2] == 1 and crossing_scan[0] == [0x3000]
        no_newline = root / "no-newline.libriscv.log"
        no_newline.write_bytes(b"f f_3004 pc 0x3004 instr 00000013")
        no_newline_scan = _scan_trace([no_newline], "libriscv-translated", {}, {0x3004})
        assert no_newline_scan[2] == 1 and no_newline_scan[0] == [0x3004]
        trace = root / "qemu.drcov"
        trace.write_bytes(
            b"DRCOV VERSION: 2\nDRCOV FLAVOR: drcov-64\n"
            b"Module Table: version 2, count 1\n"
            b"Columns: id, base, end, entry, path\n"
            b"0, 0x1000, 0x2000, 0x1000, case\n"
            b"BB Table: 1 bbs\n"
            + struct.pack("<IHH", 0x1000, 4, 0)
        )
        feature = {
            "status": "ok", "decoder_version": DECODER_VERSION,
            "lane": "rv64i/lp64", "isa_profile": "rv64i", "privilege_mode": "user",
            "instructions": [{
                "address": 0x1000, "encoding_id": "enc:32:rv64i:addi",
                "type_id": "type:integer-immediate", "type_ids": ["type:integer-immediate"],
                "configuration_id": "cfg:integer:immediate",
                "configuration_ids": ["cfg:integer:immediate"],
            }],
        }
        drcov_scan = _scan_trace(
            [trace], "qemu-drcov-basic-block", {}, {0x1000},
        )
        assert drcov_scan[2] == 1 and drcov_scan[0] == [0x1000]
        block_only = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_path": "qemu.drcov", "status": "passed",
             "target_outcome": "complete_observation", "observer_complete": True,
             "target_attempted": True, "process_started": True},
            root, feature,
        )
        assert block_only["status"] == "gap"
        qemu_exec = root / "qemu-execlog.pc"
        qemu_exec.write_bytes(b'0, 0x1000, 0x00000013, "addi a0, zero, 0"\n')
        result = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_path": "qemu-execlog.pc", "status": "passed",
             "target_outcome": "complete_observation", "observer_complete": True,
             "target_attempted": True, "process_started": True},
            root, feature,
        )
        assert result["guest"]["metrics"]["ICov-encoding"]["eligible"] == 52
        assert result["guest"]["metrics"]["ICov-encoding"]["value"] == round(1 / 52, 6)
        rax_trace = root / "rax-tohost.pc"
        rax_trace.write_bytes(b"INS 0x1000 compact\n")
        rax_coverage = summarize_simulator_coverage(
            {"id": "T-RAX", "kind": "rax"},
            {"trace_path": "rax-tohost.pc", "status": "gap",
             "observer_complete": False, "terminal_observed": True,
            "termination": "guest-tohost", "trace_complete": True,
            "trace_truncated": False, "trace_cleanup_status": "clean",
            "trace_pcs": [0x1000],
            "target_attempted": True, "process_started": True},
            root, feature,
        )
        assert rax_coverage["status"] == "observed"
        assert rax_coverage["guest"]["metrics"]["PCov"]["value"] == 1.0
        assert rax_coverage["guest"]["trace_records"] == 1
        rvvm_patched = summarize_simulator_coverage(
            {"id": "T-RVVM", "kind": "rvvm"},
            {"trace_pcs": [0x1000], "status": "passed",
             "target_outcome": "complete_observation", "observer_complete": True,
             "target_attempted": True, "process_started": True,
             "target_identity": {"source_provenance": "patch-validated-build"}},
            root, feature,
        )
        assert rvvm_patched["status"] == "gap"
        assert rvvm_patched["reason"] == "target-source-patch-forbidden"
        assert rvvm_patched["guest"]["trace_records"] == 0
        patched_batch = aggregate_simulator_coverage([{
            "target": "T-RVVM", "case_id": "candidate-13",
            "target_identity": {"source_provenance": "patch-validated-build"},
            "simulator_coverage": {
                "schema_version": SCHEMA, "status": "observed",
                "backend": "rvvm-riscv64", "adapter": "rvvm-gdb-rsp-pc",
                "guest": {"metrics": {"PCov": {
                    "status": "observed", "value": 1.0,
                    "eligible_units": ["pc:0x1000"],
                    "covered_units": ["pc:0x1000"],
                }}},
                "simulator_events": {"event_counts": {}},
            },
            "simulator_source_coverage": {
                "schema_version": SOURCE_SCHEMA, "metric_family": "SimSrcCov",
                "status": "observed", "metrics": {},
            },
        }])
        patched_target = patched_batch["targets"][0]
        assert patched_target["status"] == "gap"
        assert patched_target["reason"] == "target-source-patch-forbidden"
        assert patched_batch["source_targets"][0]["simulator_source_coverage"]["status"] == "gap"
        missing_patched_batch = aggregate_simulator_coverage([{
            "target": "T-RVVM", "case_id": "candidate-14",
            "target_identity": {"source_provenance": "patch-validated-build"},
        }])
        assert missing_patched_batch["status"] == "gap"
        assert missing_patched_batch["targets"][0]["reason"] == "target-source-patch-forbidden"
        nonzero_exit = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"status": "failed", "target_outcome": "nonzero-exit",
             "observer_complete": True, "target_attempted": True,
             "process_started": True, "trace_pcs": [0x1000]},
            root, feature,
        )
        assert nonzero_exit["status"] == "observed"
        stale_collection = target_source_coverage(
            root, {"id": "T-QEMU", "coverage_binary": "qemu",
                   "source_coverage": {"profile": "coverage.info"}},
            collection_status="gap", collection_reason="collector-timeout",
        )
        assert stale_collection["status"] == "gap"
        assert stale_collection["reason"] == "collector-timeout"
        trap_result = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_pcs": [0x1000], "target_outcome": "trap",
             "observer_complete": True}, root, feature,
        )
        assert trap_result["status"] == "observed"
        truncated_result = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_path": "qemu-execlog.pc", "target_outcome": "complete_observation",
             "observer_complete": True, "trace_limit_exceeded": True}, root, feature)
        assert truncated_result["status"] == "partial"
        assert truncated_result["guest"]["status"] == "partial"
        assert truncated_result["guest"]["metrics"]["ICov-encoding"]["value"] is None
        partial_mismatch = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_pcs": [0x1000, 0x2000], "target_outcome": "complete_observation",
             "observer_complete": True}, root, feature)
        assert partial_mismatch["status"] == "partial"
        assert partial_mismatch["reason"] == "guest-pc-map-mismatch"
        assert partial_mismatch["guest"]["ignored_records"] == 1
        incomplete_observation = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_pcs": [0x1000], "observer_complete": False}, root, feature,
        )
        assert incomplete_observation["status"] == "observed"
        assert incomplete_observation["guest"]["metrics"]["PCov"]["value"] == 1.0
        terminal_without_state_frame = summarize_simulator_coverage(
            {"id": "T-UNICORN", "kind": "unicorn"},
            {"status": "gap", "trace_pcs": [0x1000],
             "observer_complete": False, "terminal_observed": True,
             "termination": "guest-tohost", "guest_status": "pass"},
            root, feature,
        )
        assert terminal_without_state_frame["status"] == "observed"
        assert terminal_without_state_frame["guest"]["metrics"]["PCov"]["value"] == 1.0
        assert terminal_without_state_frame["guest"]["metrics"]["ICov-encoding"]["value"] == round(1 / 52, 6)
        stale = dict(feature)
        stale["decoder_version"] = "old"
        stale_result = summarize_simulator_coverage({"id": "T-QEMU", "kind": "qemu"},
                                                    {"trace_path": "qemu-execlog.pc",
                                                     "target_outcome": "complete_observation",
                                                     "observer_complete": True}, root, stale)
        assert stale_result["status"] == "gap"
        assert stale_result["reason"] == "static-feature-map-stale"
        assert stale_result["guest"]["metrics"]["ICov-encoding"]["eligible"] == 52
        assert stale_result["guest"]["metrics"]["ICov-encoding"]["value"] is None
        mismatch = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_pcs": [0x2000], "target_outcome": "complete_observation",
             "observer_complete": True}, root, feature)
        assert mismatch["status"] == "gap"
        assert mismatch["reason"] == "guest-pc-map-mismatch"
        assert mismatch["guest"]["metrics"]["ICov-encoding"]["eligible"] == 52
        assert mismatch["guest"]["metrics"]["ICov-encoding"]["value"] is None
        legacy = root / "qemu-legacy.log"
        legacy.write_text("Trace 0: 0x1 [00000000/00001000/00000000/00000000]\\n", encoding="utf-8")
        assert _scan_trace([legacy], "qemu", {})[2] == 0
        renode = root / "renode.trace"
        record = lambda pc, opcode: pc.to_bytes(8, "little") + bytes([len(opcode)]) + opcode + b"\0"
        renode.write_bytes(b"ReTrace" + bytes([4, 8, 1, 0, 0]) +
                          record(0x1000, b"\x13\x00\x00\x00") + record(0x1004, b"\x01\x00"))
        scanned = _scan_trace([renode], "renode", {}, {0x1000, 0x1004})
        assert scanned[2] == 2 and scanned[3] is False and scanned[0] == [0x1000, 0x1004]
        renode_repeated = root / "renode-repeated.pc"
        renode_repeated.write_text("0x1000\n0x1000\n0x1004\n", encoding="utf-8")
        repeated_scan = _scan_trace(
            [renode_repeated], "renode-unique-pc", {}, {0x1000, 0x1004},
        )
        assert repeated_scan[0] == [0x1000, 0x1004] and repeated_scan[2] == 3
        renode_feature = {
            "status": "ok", "decoder_version": DECODER_VERSION,
            "lane": "rv64imc_zicsr_zifencei/lp64",
            "isa_profile": "rv64imc_zicsr_zifencei", "privilege_mode": "machine",
            "instructions": [
                {"address": 0x1000, "mnemonic": "addi",
                 "operands": "t0,t0,0 # 1004 <tohost>",
                 "encoding_id": "enc:32:rv64i:addi"},
                {"address": 0x1004, "mnemonic": "sd", "operands": "t1,0(t0)",
                 "encoding_id": "enc:32:rv64i:sd"},
                {"address": 0x1008, "mnemonic": "c.ebreak", "operands": "",
                 "encoding_id": "enc:16:rv64c:c.ebreak"},
                {"address": 0x100A, "mnemonic": "c.j", "operands": "100a",
                 "encoding_id": "enc:16:rv64c:c.j"},
            ],
        }
        assert _renode_terminal_pcs(renode_feature) == {0x1004}
        renode_terminal = root / "renode-terminal.trace"
        renode_terminal.write_bytes(
            b"ReTrace" + bytes([4, 8, 1, 0, 0])
            + record(0x1000, b"\x13\x00\x00\x00")
            + record(0x1004, b"\x13\x00\x00\x00")
            + record(0x1008, b"\x02\x90")
            + record(0x100A, b"\x01\x00")
        )
        terminal_result = summarize_simulator_coverage(
            {"id": "T-RENODE", "kind": "renode"},
            {"trace_path": "renode-terminal.trace", "status": "passed",
             "target_outcome": "complete_observation", "observer_complete": True},
            root, renode_feature,
        )
        assert (terminal_result["guest"]["ignored_records"] == 0
                and terminal_result["guest"]["trace_records_total"] == 2
                and terminal_result["guest"]["mapped_records"] == 2)
        renode_tail = root / "renode-tail.trace"
        renode_tail.write_bytes(
            b"ReTrace" + bytes([4, 8, 1, 0, 0])
            + record(0x1000, b"\x13\x00\x00\x00")
            + record(0x1004, b"\x13\x00\x00\x00")
        )
        renode_tail_result = summarize_simulator_coverage(
            {"id": "T-RENODE", "kind": "renode"},
            {"trace_path": "renode-tail.trace", "status": "passed",
             "observer_complete": False, "terminal_observed": True,
             "termination": "guest-tohost", "reason": "guest-tohost"},
            root, {**feature, "instructions": feature["instructions"]},
        )
        assert renode_tail_result["guest"]["ignored_records"] == 0
        assert renode_tail_result["status"] == "observed"
        assert renode_tail_result["guest"]["metrics"]["PCov"]["value"] == 1.0
        missing_guest = summarize_simulator_coverage({"id": "T-QEMU", "kind": "qemu"}, {}, root, feature)
        missing_metric = missing_guest["guest"]["metrics"]["ICov-encoding"]
        assert missing_metric["status"] == "NA" and missing_metric["eligible"] == 52
        assert missing_metric["value"] is None
        attempted_without_trace = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"status": "timeout", "target_outcome": "timeout",
             "target_attempted": True, "process_started": True},
            root, feature,
        )
        assert attempted_without_trace["status"] == "gap"
        assert attempted_without_trace["guest"]["metrics"]["ICov-encoding"]["status"] == "gap"
        missing_trace = summarize_simulator_coverage(
            {"id": "T-QEMU", "kind": "qemu"},
            {"trace_path": "missing.drcov", "target_outcome": "complete_observation",
             "observer_complete": True, "target_attempted": True},
            root, feature,
        )
        assert (missing_trace["status"] == "gap"
                and missing_trace["reason"] == "trace-read-error")
        rvvm_attach_gap = summarize_simulator_coverage(
            {"id": "T-RVVM", "kind": "rvvm"},
            {"status": "passed", "target_outcome": "complete_observation",
             "observer_complete": True, "target_attempted": True,
             "process_started": True, "trace_reason": "rvvm-attached-after-guest-exit"},
            root, feature,
        )
        assert rvvm_attach_gap["status"] == "gap"
        assert rvvm_attach_gap["reason"] == "rvvm-attached-after-guest-exit"
        assert rvvm_attach_gap["guest"]["metrics"]["ICov-encoding"]["status"] == "gap"
        rvvm_no_step = summarize_simulator_coverage(
            {"id": "T-RVVM", "kind": "rvvm"},
            {"status": "passed", "target_outcome": "complete_observation",
             "observer_complete": True, "target_attempted": True,
             "process_started": True, "trace_status": "unavailable",
             "trace_gap_reason": "rvvm-single-step-unsupported"},
            root, feature,
        )
        assert (rvvm_no_step["status"] == "gap"
                and rvvm_no_step["reason"] == "rvvm-single-step-unsupported")
        assert set(result["simulator_events"]) == {
            "schema_version", "backend", "adapter", "status", "event_counts", "trace"
        }
        batch = aggregate_simulator_coverage([{
            "target": "T-QEMU", "stratum": "bare-metal", "simulator_coverage": result,
            "simulator_source_coverage": {"status": "observed", "metric_family": "SimSrcCov"},
        }])
        assert batch["targets"][0]["guest"]["ICov-type"]["eligible"] == 10
        assert batch["targets"][0]["guest"]["ICov-type"]["value"] == 0.1
        split = aggregate_simulator_coverage([
            {"method": "M", "route": "program", "lane": "rv64im/lp64",
             "stratum": "bare-metal", "target": "T-QEMU", "simulator_coverage": result},
            {"method": "M", "route": "program", "lane": "rv64im/lp64",
             "stratum": "linux-user", "target": "T-QEMU", "simulator_coverage": result},
        ])
        assert len(split["targets"]) == 2
        missing_batch = aggregate_simulator_coverage([{
            "target": "T-QEMU", "stratum": "bare-metal", "simulator_coverage": missing_guest,
        }])
        missing_batch_metric = missing_batch["targets"][0]["guest"]["ICov-encoding"]
        assert missing_batch_metric["eligible"] == 52 and missing_batch_metric["value"] is None
        assert not any("event" in key and "coverage" in key for key in batch["targets"][0])
        assert len(batch["source_targets"]) == 1
        assert "simulator_source_coverage" not in batch["targets"][0]
        unpaired = aggregate_simulator_coverage([
            {"target": "T-QEMU", "case_id": "candidate-0", "artifact_sha256": "a" * 64,
             "simulator_coverage": result},
            {"target": "T-RAX", "case_id": "candidate-0", "artifact_sha256": "b" * 64,
             "simulator_coverage": result},
        ])
        assert unpaired["comparability"] == "target-local-fixed-profile-universe"
        unidentified = aggregate_simulator_coverage([
            {"target": "T-QEMU", "case_id": "candidate-0", "simulator_coverage": result},
            {"target": "T-RAX", "case_id": "candidate-0", "simulator_coverage": result},
        ])
        assert unidentified["comparability"] == "target-local-fixed-profile-universe"
        inconsistent = aggregate_simulator_coverage([{
            "target": "T-QEMU", "case_id": "candidate-0", "artifact_sha256": "a" * 64,
            "simulator_coverage": {**result, "status": "gap"},
        }])
        # A stale case status gates the ratio but keeps structurally verified
        # numerator units available for diagnosis.
        assert inconsistent["targets"][0]["guest"]["ICov-encoding"]["covered"] == result["guest"]["metrics"]["ICov-encoding"]["covered"]
        assert inconsistent["targets"][0]["guest"]["ICov-encoding"]["value"] is None
        single = aggregate_simulator_coverage([{
            "target": "T-QEMU", "case_id": "candidate-0", "artifact_sha256": "a" * 64,
            "simulator_coverage": result,
        }])
        assert single["comparability"] == "target-local-fixed-profile-universe"
        assert single["targets"][0]["translator_coverage"] == {
            "covered": None, "eligible": None, "value": None,
            "status": "NA", "reason": "translator-feature-channel-unavailable",
        }
        legacy_batch = aggregate_simulator_coverage([{
            "target": "T-QEMU", "simulator_coverage": {**result, "schema_version": "rq1-simulator-coverage-v1"},
        }])
        assert (legacy_batch["targets"][0]["status"] == "gap"
                and legacy_batch["targets"][0]["reason"] == "simulator-coverage-schema-mismatch")
        pc_collision = aggregate_simulator_coverage([
            {"target": "T-QEMU", "case_id": "candidate-0", "artifact_sha256": "a" * 64,
             "simulator_coverage": result},
            {"target": "T-QEMU", "case_id": "candidate-1", "artifact_sha256": "b" * 64,
             "simulator_coverage": result},
        ])
        pc_metric = pc_collision["targets"][0]["guest"]["PCov"]
        assert pc_metric["eligible"] == 2 and pc_metric["artifact_count"] == 2
        assert pc_metric["value"] == 1.0
        partial_batch = aggregate_simulator_coverage([
            {"method": "M", "route": "program", "lane": "rv64im/lp64",
             "stratum": "bare-metal", "target": "T-QEMU", "case_id": "candidate-0",
             "artifact_sha256": "a" * 64, "simulator_coverage": result},
            {"method": "M", "route": "program", "lane": "rv64im/lp64",
             "stratum": "bare-metal", "target": "T-QEMU", "case_id": "candidate-1",
             "artifact_sha256": "b" * 64},
        ])
        partial_row = partial_batch["targets"][0]
        assert partial_row["status"] == "partial"
        assert partial_row["cases"] == 2
        assert partial_row["observed_cases"] == 1
        assert partial_row["incomplete_cases"] == 1
        assert partial_row["guest"]["ICov-encoding"]["status"] == "partial"
        assert partial_row["guest"]["ICov-encoding"]["value"] is None
        # per-case 记录里绝大多数 SimSrcCov 是 deferred 占位，run 级那一条
        # 才是真实数值；先到的占位不能让真实覆盖被丢掉。
        def simsrc(source):
            return {"target": "T-QEMU", "case_id": "candidate-0",
                    "artifact_sha256": "a" * 64, "simulator_coverage": result,
                    "simulator_source_coverage": source}
        deferred = {"schema_version": SOURCE_SCHEMA, "status": "deferred",
                    "reason": "run-level-source-coverage-collection"}
        source_commit = "c" * 40
        source_binary_sha256 = "b" * 64
        measured = {
            "schema_version": SOURCE_SCHEMA,
            "status": "observed", "metrics": {"line": {"value": 1.0}},
            "identity": {
                "source_commit": source_commit,
                "source_commit_expected": source_commit,
                "source_commit_observed": source_commit,
                "source_commit_match": True,
                "source_tree_status": "clean",
                "source_tree_dirty": False,
                "non_generated_dirty_paths": [],
                "target_commit": source_commit,
                "coverage_binary_expected_sha256": source_binary_sha256,
                "coverage_binary_sha256": source_binary_sha256,
                "coverage_binary_sha256_match": True,
            },
        }
        simsrc_batch = aggregate_simulator_coverage([simsrc(deferred), simsrc(measured)])
        assert len(simsrc_batch["source_targets"]) == 1
        assert simsrc_batch["source_targets"][0]["simulator_source_coverage"]["status"] == "observed"
        reverse_batch = aggregate_simulator_coverage([simsrc(measured), simsrc({"status": "gap"})])
        assert reverse_batch["source_targets"][0]["simulator_source_coverage"]["status"] == "gap"
        simsrc_only = aggregate_simulator_coverage([simsrc(deferred)])
        assert simsrc_only["source_targets"][0]["simulator_source_coverage"]["status"] == "deferred"
    return {"status": "ok", "adapters": len(ADAPTERS), "schema_version": SCHEMA}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ1 simulator coverage calculator")
    parser.add_argument("command", choices=("self-check", "lcov"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--source-scope", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
        return 0
    if not args.profile:
        parser.error("lcov requires --profile")
    print(json.dumps(summarize_lcov(args.profile, args.source_scope), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

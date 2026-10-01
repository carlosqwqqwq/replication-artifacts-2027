#!/usr/bin/env python3
"""Compare complete RQ1 coverage summaries under matched runs.

Active source-only runs compare target-local SimSrcCov; legacy runs retain the
fixed guest instruction coverage comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.elf_features import (
    CATALOG_SOURCE, DECODER_VERSION, DEFAULT_PRIVILEGE_MODES, ISA_SPEC_RELEASE,
    TAXONOMY_VERSION,
    metric_registry_for_profile, verify_sealed_entry,
)
from comparison_artifacts import digest
from analysis.simulator_coverage import (
    BATCH_SCHEMA, EXPERIMENT_UNION_PROFILE, SOURCE_SCHEMA,
    _nonnegative_count,
    _pcov_unit_digest,
)
from analysis.rv_instruction_coverage import (
    EXPERIMENT_COVERAGE_PROFILE, EXPERIMENT_PRIVILEGE_MODE,
    OPCODE_CATALOG_DENOMINATOR, OPCODE_CATALOG_ID,
    OPCODE_CATALOG_KEYS,
    OPCODE_CATALOG_REGISTRY_SHA256, OPCODE_CATALOG_SCHEMA,
    OPCODE_KEY_SCHEMA,
    compare_experiment_unions,
    registry_for_profile as rv_registry_for_profile,
)
SCHEMA = "rq1-coverage-comparison-v5"
# One experiment-wide ISA universe keeps results comparable across narrower generators.
# Unsupported forms in a run's profile earn no hits.
COMPARISON_PROFILE = EXPERIMENT_UNION_PROFILE
METRICS = ("PCov", "ICov-encoding", "ICov-type", "CCov")
METRIC_KINDS = {
    "ICov-encoding": "encoding",
    "ICov-type": "type",
    "CCov": "configuration",
}
ROW_BASIS = {
    "PCov": "run-union-elf-pc",
    "ICov-encoding": "fixed-profile-universe",
    "ICov-type": "fixed-profile-universe",
    "CCov": "fixed-profile-universe",
}
RUN_MATCH_FIELDS = (
    ("config_sha256", "configuration"),
    ("source_commit", "RQ1 source commit"),
    ("image_id", "execution image"),
    ("execution_plane", "execution plane"),
    ("network", "network policy"),
    ("platform", "host and container platform"),
    ("resources", "resource limits"),
    ("duration_seconds", "campaign duration"),
    ("experiment_seed", "experiment seed"),
)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str | None:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _artifact_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _integrity_files_match(root: Path, files: Any) -> bool:
    if not isinstance(files, Mapping) or not files:
        return False
    for relative, expected in files.items():
        path = _artifact_path(root, relative)
        if (path is None or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or _sha256_file(path) != expected):
            return False
    return True


def _comparison_campaign_issues(
    root: Path, result: Mapping[str, Any],
    method: Any, summary_path: Path,
) -> list[str]:
    issues = []
    campaign_path = root / "derived" / "campaign-complete.json"
    integrity_path = root / "derived" / "integrity.json"
    seal_path = root / "seal.json"
    campaign = _read_object(campaign_path)
    integrity = _read_object(integrity_path)
    seal = _read_object(seal_path)
    if campaign is None:
        issues.append("campaign-complete-missing-or-invalid")
    elif (
        campaign.get("schema_version") != "rq1-campaign-complete-v1"
        or campaign.get("status") != "complete"
        or campaign.get("execution_complete") is not True
        or campaign.get("run_id") != root.name
        or campaign.get("method_filter") != method
        or campaign.get("coverage_artifact_complete") is not True
    ):
        issues.append("campaign-complete-incomplete-or-identity-mismatch")
    if integrity is None:
        issues.append("campaign-integrity-missing-or-invalid")
    elif (
        integrity.get("schema_version") != "rq1-integrity-v1"
        or integrity.get("status") != "passed"
        or integrity.get("execution_complete") is not True
        or integrity.get("coverage_artifact_complete") is not True
        or integrity.get("coverage_registry_complete") is not True
    ):
        issues.append("campaign-integrity-incomplete")
    if integrity is not None and not _integrity_files_match(root, integrity.get("files")):
        issues.append("campaign-integrity-files-mismatch")
    required_files = {
        "coverage-registry.json",
        summary_path.resolve().relative_to(root.resolve()).as_posix(),
        "coverage/case-results.jsonl.gz",
    }
    if integrity is not None:
        integrity_files = integrity.get("files")
        if (not isinstance(integrity_files, Mapping)
                or not required_files <= set(integrity_files)):
            issues.append("campaign-integrity-coverage-artifacts-missing")
    if seal is None:
        issues.append("campaign-seal-missing-or-invalid")
    else:
        seal_basis = dict(seal)
        declared_seal_sha = seal_basis.pop("seal_sha256", None)
        if (
            seal.get("schema_version") != "rq1-campaign-seal-v1"
            or seal.get("sealed") is not True
            or seal.get("execution_complete") is not True
            or seal.get("run_id") != root.name
            or seal.get("method_filter") != method
        ):
            issues.append("campaign-seal-incomplete-or-identity-mismatch")
        if declared_seal_sha != digest(seal_basis):
            issues.append("campaign-seal-self-digest-mismatch")
        if (campaign is not None
                and seal.get("campaign_complete_sha256") != _sha256_file(campaign_path)):
            issues.append("campaign-seal-campaign-digest-mismatch")
        if (integrity is not None
                and seal.get("integrity_sha256") != _sha256_file(integrity_path)):
            issues.append("campaign-seal-integrity-digest-mismatch")
        registry_path = root / "coverage-registry.json"
        case_results_path = root / "coverage" / "case-results.jsonl.gz"
        if seal.get("coverage_registry_file_sha256") != _sha256_file(registry_path):
            issues.append("campaign-seal-coverage-registry-digest-mismatch")
        if seal.get("coverage_summary_sha256") != _sha256_file(summary_path):
            issues.append("campaign-seal-coverage-summary-digest-mismatch")
        if seal.get("coverage_case_results_sha256") != _sha256_file(case_results_path):
            issues.append("campaign-seal-coverage-case-results-digest-mismatch")
    if result.get("coverage_registry_sha256") != digest(
        _read_object(root / "coverage-registry.json") or {}
    ):
        issues.append("coverage-registry-result-digest-mismatch")
    return issues


def _framework_case_chain_valid(root: Path, chain: Mapping[str, Any],
                                method: str, feedback_target: Any) -> bool:
    integrity_path = _artifact_path(root, chain.get("integrity"))
    seal_path = _artifact_path(root, chain.get("seal"))
    if integrity_path is None or seal_path is None:
        return False
    integrity = _read_object(integrity_path)
    seal = _read_object(seal_path)
    if integrity is None or seal is None:
        return False
    return bool(
        integrity.get("schema_version") == "rq1-framework-integrity-v1"
        and integrity.get("status") == "verified"
        and integrity.get("experiment_face") == "framework"
        and integrity.get("method") == method
        and integrity.get("feedback_target") == feedback_target
        and seal.get("schema_version") == "rq1-framework-seal-v1"
        and seal.get("sealed") is True
        and seal.get("status") == "sealed"
        and seal.get("experiment_face") == "framework"
        and seal.get("method") == method
        and seal.get("feedback_target") == feedback_target
        and seal.get("integrity") == chain.get("integrity")
        and seal.get("integrity_sha256") == _sha256_file(integrity_path)
        and _artifact_path(root, seal.get("chain_events")) is not None
        and _integrity_files_match(root, integrity.get("files"))
    )


def _framework_campaign_issues(
    root: Path, manifest: Mapping[str, Any], result: Mapping[str, Any],
    method: Any, summary_path: Path,
) -> list[str]:
    issues = []
    feedback_target = manifest.get("feedback_target")
    if (
        result.get("method") != method
        or result.get("feedback_target") != feedback_target
        or result.get("status") != "passed"
        or result.get("execution_complete") is not True
        or result.get("coverage_status") != "recorded"
        or result.get("coverage_artifact_complete") is not True
        or result.get("formal_ready") is not True
    ):
        issues.append("framework-run-result-incomplete-or-not-formal-ready")
    if not isinstance(method, str) or not method or "/" in method or "\\" in method:
        return issues + ["framework-method-identity-invalid"]
    method_dir = root / "framework" / method.replace("/", "-")
    parallel_result_path = method_dir / "parallel-result.json"
    integrity_path = method_dir / "parallel-integrity.json"
    seal_path = method_dir / "parallel-seal.json"
    parallel_result = _read_object(parallel_result_path)
    integrity = _read_object(integrity_path)
    seal = _read_object(seal_path)
    if parallel_result is None:
        issues.append("framework-parallel-result-missing-or-invalid")
    if integrity is None:
        issues.append("framework-parallel-integrity-missing-or-invalid")
    if seal is None:
        issues.append("framework-parallel-seal-missing-or-invalid")
    if parallel_result is None or integrity is None or seal is None:
        return issues
    lanes = parallel_result.get("lanes")
    cases = [
        case for lane in lanes if isinstance(lane, Mapping)
        for case in lane.get("cases", ()) if isinstance(case, Mapping)
    ] if isinstance(lanes, list) else []
    chain = parallel_result.get("chain")
    expected_integrity = str(integrity_path.relative_to(root))
    expected_seal = str(seal_path.relative_to(root))
    valid = (
        parallel_result.get("schema_version") == "rq1-framework-parallel-v1"
        and parallel_result.get("status") == "passed"
        and parallel_result.get("execution_complete") is True
        and parallel_result.get("method") == method
        and parallel_result.get("feedback_target") == feedback_target
        and isinstance(chain, Mapping)
        and chain.get("integrity") == expected_integrity
        and chain.get("seal") == expected_seal
        and chain.get("sealed") is True
        and chain.get("integrity_verified") is True
        and integrity.get("schema_version") == "rq1-framework-parallel-integrity-v1"
        and integrity.get("status") == "verified"
        and integrity.get("experiment_face") == "framework"
        and integrity.get("method") == method
        and integrity.get("feedback_target") == feedback_target
        and integrity.get("case_count") == len(cases) > 0
        and integrity.get("sealed_case_count") == len(cases)
        and integrity.get("counts") == parallel_result.get("counts")
        and seal.get("schema_version") == "rq1-framework-parallel-seal-v1"
        and seal.get("sealed") is True
        and seal.get("status") == "sealed"
        and seal.get("experiment_face") == "framework"
        and seal.get("method") == method
        and seal.get("feedback_target") == feedback_target
        and seal.get("integrity") == expected_integrity
        and seal.get("integrity_sha256") == _sha256_file(integrity_path)
        and _integrity_files_match(root, integrity.get("files"))
    )
    if not valid:
        issues.append("framework-parallel-seal-or-integrity-invalid")
    for case in cases:
        case_result = _artifact_path(root, case.get("case_result"))
        case_chain = case.get("chain")
        if (
            case_result is None
            or _read_object(case_result) != case
            or not isinstance(case_chain, Mapping)
            or case_chain.get("sealed") is not True
            or case_chain.get("integrity_verified") is not True
            or not _framework_case_chain_valid(root, case_chain, method, feedback_target)
        ):
            issues.append("framework-parallel-case-chain-invalid")
            break
    summary_relative = summary_path.resolve().relative_to(root.resolve()).as_posix()
    if summary_relative not in integrity.get("files", {}):
        issues.append("framework-integrity-coverage-summary-missing")
    return issues


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _digest_units(units: Sequence[str]) -> str:
    payload = json.dumps(sorted(units), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coverage_profile_parts(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    basis = row.get("coverage_basis")
    basis = basis if isinstance(basis, Mapping) else {}
    profile_id = basis.get("coverage_profile_id")
    if not isinstance(profile_id, str):
        return None, None
    parts = profile_id.split("|")
    return (parts[0], parts[1]) if len(parts) == 3 else (None, None)


def _lane_abi(row: Mapping[str, Any]) -> str | None:
    lane = row.get("lane")
    if not isinstance(lane, str) or "/" not in lane:
        return None
    _isa, abi = lane.split("/", 1)
    return abi.strip().lower() or None


def _close(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.1e-6)
    except (TypeError, ValueError, OverflowError):
        return False


def _target_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("targets")
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("id"):
            return {}
        key = str(row["id"])
        if key in result:
            return {}
        result[key] = row
    return result


def _load_run(path: Path) -> dict[str, Any]:
    root = path.resolve()
    manifest = _read_object(root / "execution-manifest.json")
    summary_path = root / "coverage" / "summary.json"
    if not summary_path.is_file() and isinstance(manifest, Mapping):
        method = manifest.get("method_filter")
        if (manifest.get("action") == "framework-run"
                and isinstance(method, str) and method not in {"", ".", ".."}
                and "/" not in method and "\\" not in method):
            framework_summary = root / "framework" / method / "coverage" / "summary.json"
            if framework_summary.is_file():
                summary_path = framework_summary
    summary = _read_object(summary_path)
    result = _read_object(root / "run-result.json")
    registry_path = root / "coverage-registry.json"
    registry = _read_object(registry_path) if registry_path.exists() else None
    issues: list[str] = []
    if manifest is None:
        issues.append("execution-manifest-missing-or-invalid")
        manifest = {}
    elif manifest.get("schema_version") != "rq1-comparison-run-manifest-v1":
        issues.append("execution-manifest-schema-unsupported")
    if manifest.get("run_id") != root.name:
        issues.append("execution-manifest-run-id-mismatch")
    if manifest.get("action") not in {"run", "execute-existing-queues", "framework-run"}:
        issues.append("execution-manifest-action-not-comparable")
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        issues.append("source-commit-identity-missing-or-invalid")
    # The end-of-run source identity is observational metadata. If working
    # files change during a campaign, retain the measured coverage instead of
    # turning the comparison into a hard rejection.
    if summary is None:
        issues.append("coverage-summary-missing-or-invalid")
        summary = {}
    elif summary.get("schema_version") != BATCH_SCHEMA:
        issues.append("coverage-summary-schema-unsupported")
    if result is None:
        issues.append("run-result-missing-or-invalid")
        result = {}
    else:
        if result.get("schema_version") != "rq1-run-result-v2":
            issues.append("run-result-schema-unsupported")
        if result.get("execution_complete") is not True:
            issues.append("run-execution-incomplete")
        if result.get("coverage_artifact_complete") is not True:
            issues.append("coverage-artifact-incomplete")
    method = manifest.get("method_filter")
    experiment_union = None
    if not isinstance(method, str) or not method:
        issues.append("manifest-method-filter-missing")
    else:
        unions = summary.get("experiment_unions")
        matches = [item for item in unions if isinstance(item, Mapping)
                   and item.get("method") == method] if isinstance(unions, list) else []
        if not isinstance(unions, list) or len(unions) != 1:
            issues.append("experiment-wide-coverage-union-method-boundary-mismatch")
        if len(matches) != 1 or matches[0].get("scope") != (
            "whole-method-run-all-cases-and-targets"
        ):
            issues.append("experiment-wide-coverage-union-missing-or-ambiguous")
        elif matches[0].get("scope") == "whole-method-run-all-cases-and-targets":
            experiment_union = matches[0]
        target_rows = summary.get("targets")
        if isinstance(target_rows, list) and any(
            isinstance(row, Mapping) and row.get("method") not in (None, method)
            for row in target_rows
        ):
            issues.append("coverage-target-method-boundary-mismatch")
    if manifest.get("coverage_enabled") is not True or result.get("coverage_enabled") is False:
        issues.append("coverage-collection-disabled")
    isolation = manifest.get("method_isolation")
    if (not isinstance(isolation, Mapping)
            or isolation.get("mode") != "one-method-per-container"
            or isolation.get("method") != method
            or isolation.get("method_count") != 1):
        issues.append("method-isolation-unverified")
    if manifest.get("dependency_identity_ok") is not True:
        issues.append("method-dependency-identity-unverified")
    if result.get("method_filter") not in (None, method):
        issues.append("run-result-method-filter-mismatch")
    if result.get("coverage_enabled") not in (None, True):
        issues.append("run-result-coverage-disabled")
    if result.get("coverage_enabled") is not True:
        issues.append("run-result-coverage-status-missing")
    if manifest.get("action") in {"run", "execute-existing-queues"}:
        issues.extend(_comparison_campaign_issues(
            root, result, method, summary_path,
        ))
    elif manifest.get("action") == "framework-run":
        issues.extend(_framework_campaign_issues(
            root, manifest, result, method, summary_path,
        ))
    registry_required = manifest.get("action") in {"run", "execute-existing-queues"}
    if registry_required and not registry_path.is_file():
        issues.append("coverage-registry-missing-or-invalid")
    if registry_path.exists():
        if registry is None:
            issues.append("coverage-registry-missing-or-invalid")
        elif set(registry) != {"encoding", "type", "configuration"} or not all(
            verify_sealed_entry(registry.get(kind), kind) for kind in registry
        ):
            issues.append("coverage-registry-seal-invalid")
        elif result.get("coverage_registry_sha256") != digest(registry):
            issues.append("coverage-registry-result-digest-mismatch")
    targets = _target_rows(manifest)
    if not targets:
        issues.append("manifest-target-identities-missing-or-ambiguous")
    if experiment_union is not None:
        union_targets = experiment_union.get("targets")
        scope_targets = experiment_union.get("target_scopes")
        expected_targets = set(targets)
        if (not isinstance(union_targets, list)
                or set(str(item) for item in union_targets) != expected_targets):
            issues.append("experiment-target-set-does-not-match-manifest")
        if (not isinstance(scope_targets, list)
                or {str(item.get("target") or "") for item in scope_targets
                    if isinstance(item, Mapping)} != expected_targets):
            issues.append("experiment-target-scopes-do-not-match-manifest")
    return {
        "path": str(root), "manifest": manifest, "summary": summary,
        "result": result, "method": method, "targets": targets,
        "registry": registry, "issues": issues,
    }


def _run_pair_issues(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    issues = list(left["issues"]) + list(right["issues"])
    left_manifest, right_manifest = left["manifest"], right["manifest"]
    for field, label in RUN_MATCH_FIELDS:
        left_value = left_manifest.get(field)
        right_value = right_manifest.get(field)
        if left_value in (None, "") or right_value in (None, ""):
            issues.append(f"{label}-metadata-missing")
        elif left_value != right_value:
            issues.append(f"{label}-mismatch")
    for field, label in (
        ("deadlines.target_seconds", "per-target time budget"),
    ):
        left_value = _path_value(left_manifest, field)
        right_value = _path_value(right_manifest, field)
        if left_value is None or right_value is None:
            issues.append(f"{label}-metadata-missing")
        elif left_value != right_value:
            issues.append(f"{label}-mismatch")
    if left.get("method") == right.get("method"):
        issues.append("method-filters-identical")
    if set(left["targets"]) != set(right["targets"]):
        issues.append("target-set-mismatch")
    if left["summary"].get("method_filter") not in (None, left.get("method")):
        issues.append("baseline-summary-method-filter-mismatch")
    if right["summary"].get("method_filter") not in (None, right.get("method")):
        issues.append("candidate-summary-method-filter-mismatch")
    return sorted(set(issues))


def _target_identity_issues(left: Mapping[str, Any], right: Mapping[str, Any],
                            target: str) -> list[str]:
    first = left["targets"].get(target)
    second = right["targets"].get(target)
    if first is None or second is None:
        return ["target-identity-missing"]
    fields = (
        "id", "kind", "execution_model", "translation_mode", "commit",
        "binary_sha256", "coverage_binary_sha256", "identity_digest",
    )
    first_identity = {field: first.get(field) for field in fields}
    second_identity = {field: second.get(field) for field in fields}
    immutable = ("binary_sha256", "identity_digest", "commit")
    if not any(first.get(field) for field in immutable) or not any(
        second.get(field) for field in immutable
    ):
        return ["target-immutable-identity-incomplete"]
    if not first.get("coverage_binary_sha256") or not second.get("coverage_binary_sha256"):
        return ["target-coverage-binary-identity-missing"]
    return [] if first_identity == second_identity else ["target-identity-mismatch"]


def _run_conditions(run: Mapping[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    dependency = manifest.get("dependency_identity")
    dependency_rows = dependency.get("dependencies") if isinstance(dependency, Mapping) else None
    dependencies = {}
    if isinstance(dependency_rows, Mapping):
        dependencies = {
            str(name): {
                field: item.get(field)
                for field in ("kind", "expected_commit", "head", "expected_sha256",
                              "actual_sha256", "identity_ok", "status")
            }
            for name, item in dependency_rows.items() if isinstance(item, Mapping)
        }
    return {
        "config_sha256": manifest.get("config_sha256"),
        "source_commit": manifest.get("source_commit"),
        "source_end_commit": manifest.get("source_end_commit"),
        "source_dirty": manifest.get("source_dirty"),
        "source_end_dirty": manifest.get("source_end_dirty"),
        "image_id": manifest.get("image_id"),
        "execution_plane": manifest.get("execution_plane"),
        "network": manifest.get("network"),
        "platform": manifest.get("platform"),
        "resources": manifest.get("resources"),
        "duration_seconds": manifest.get("duration_seconds"),
        "target_seconds": _path_value(manifest, "deadlines.target_seconds"),
        "experiment_seed": manifest.get("experiment_seed"),
        "method_dependency_identity": {
            "identity_ok": dependency.get("identity_ok") if isinstance(dependency, Mapping) else None,
            "dependencies": dependencies,
        },
    }


def _row_index(summary: Mapping[str, Any]) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[str]]:
    rows = summary.get("targets")
    if not isinstance(rows, list):
        return {}, ["coverage-target-rows-missing"]
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    issues = []
    for row in rows:
        if not isinstance(row, Mapping):
            issues.append("coverage-target-row-invalid")
            continue
        key = (str(row.get("target") or ""), str(row.get("stratum") or ""))
        if not all(key):
            issues.append("coverage-target-row-key-missing")
        elif key in result:
            issues.append(f"coverage-target-row-ambiguous:{key[0]}:{key[1]}")
        else:
            result[key] = row
    return result, issues


_SOURCE_METRICS = ("function", "line", "branch")
_SOURCE_IDENTITY_FIELDS = (
    "source_commit", "source_commit_expected", "source_commit_observed",
    "coverage_binary_sha256", "coverage_binary_expected_sha256",
    "coverage_binary", "source_repository", "source_root", "source_scope",
    "source_scope_exclude", "collector", "collector_input_format",
    "collector_pipeline", "collector_tools", "instrumented_binaries",
    "instrumented_binaries_match", "toolchain", "build_flags", "profile_format",
)
_SOURCE_REQUIRED_IDENTITY_FIELDS = (
    "source_commit_expected", "source_commit_observed",
    "coverage_binary_sha256", "coverage_binary_expected_sha256",
    "source_root", "source_scope", "source_scope_exclude",
)


def _source_row_index(summary: Mapping[str, Any]) -> tuple[
    list[Mapping[str, Any]] | None, list[str]
]:
    rows = summary.get("source_targets")
    if rows is None:
        return None, []
    if not isinstance(rows, list):
        return [], ["source-coverage-target-rows-invalid"]
    result: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            issues.append("source-coverage-target-row-invalid")
            continue
        key = tuple(str(row.get(name) or "") for name in ("target", "stratum"))
        if not key[0] or not key[1]:
            issues.append("source-coverage-target-row-key-missing")
            continue
        source = row.get("simulator_source_coverage")
        source = source if isinstance(source, Mapping) else {}
        signature = (
            source.get("schema_version"), source.get("status"),
            source.get("reason"), source.get("metrics"),
            _source_identity(source),
        )
        scope = {
            "lane": str(row.get("lane") or ""),
            "route": str(row.get("route") or ""),
        }
        existing = result.get(key)
        if existing is not None:
            if existing["_source_signature"] != signature:
                issues.append(f"source-coverage-target-run-ambiguous:{key[0]}:{key[1]}")
            if scope not in existing["_input_scopes"]:
                existing["_input_scopes"].append(scope)
                existing["_input_scopes"].sort(
                    key=lambda item: (item["lane"], item["route"]),
                )
            continue
        result[key] = {
            **dict(row), "_source_key": key, "_input_scopes": [scope],
            "_source_signature": signature,
        }
    return [result[key] for key in sorted(result)], issues


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    identity = source.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    return {field: identity.get(field) for field in _SOURCE_IDENTITY_FIELDS}


def _expected_source_scope_keys(run: Mapping[str, Any]) -> tuple[set[tuple[str, str]], list[str]]:
    union = _experiment_union_for_run(run)
    if union is None:
        return set(), ["experiment-wide-coverage-union-missing"]
    scopes, issues = _experiment_scope_index(union)
    target_rows = run.get("targets")
    target_rows = target_rows if isinstance(target_rows, Mapping) else {}
    expected_targets = {str(target) for target in target_rows if str(target)}
    expected_scopes: set[tuple[str, str]] = set()
    for target, row in target_rows.items():
        execution_model = row.get("execution_model") if isinstance(row, Mapping) else None
        if not isinstance(execution_model, str) or not execution_model:
            issues.append(f"manifest-target-execution-model-missing:{target}")
            continue
        stratum = (
            "linux-user" if execution_model.startswith("linux-user")
            else "bare-metal"
        )
        expected_scopes.add((str(target), stratum))
    union_targets = union.get("targets")
    declared_targets = (
        {str(target) for target in union_targets}
        if isinstance(union_targets, list) else set()
    )
    scope_keys = {(key[0], key[1]) for key in scopes}
    scope_targets = {target for target, _stratum in scope_keys}
    if (not isinstance(union_targets, list)
            or declared_targets != expected_targets
            or scope_targets != expected_targets):
        issues.append("experiment-target-scopes-do-not-match-manifest")
    if scope_keys != expected_scopes:
        issues.append("experiment-target-strata-do-not-match-manifest")
    return expected_scopes, issues


def _compare_source_metric(
    name: str, first: Mapping[str, Any], second: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if first.get("status") != "observed":
        issues.append(f"baseline-source-{name}-status-{first.get('status') or 'missing'}")
    if second.get("status") != "observed":
        issues.append(f"candidate-source-{name}-status-{second.get('status') or 'missing'}")
    first_digest = first.get("eligible_units_sha256")
    second_digest = second.get("eligible_units_sha256")
    first_digest_valid = (
        isinstance(first_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", first_digest) is not None
    )
    second_digest_valid = (
        isinstance(second_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", second_digest) is not None
    )
    if not isinstance(first_digest, str) or not first_digest:
        issues.append(f"baseline-source-{name}-denominator-digest-missing")
    elif not first_digest_valid:
        issues.append(f"baseline-source-{name}-denominator-digest-invalid")
    if not isinstance(second_digest, str) or not second_digest:
        issues.append(f"candidate-source-{name}-denominator-digest-missing")
    elif not second_digest_valid:
        issues.append(f"candidate-source-{name}-denominator-digest-invalid")
    if first_digest_valid and second_digest_valid and first_digest != second_digest:
        issues.append(f"source-{name}-denominator-mismatch")
    first_eligible, first_eligible_valid = _nonnegative_count(
        first.get("eligible"), missing=None,
    )
    second_eligible, second_eligible_valid = _nonnegative_count(
        second.get("eligible"), missing=None,
    )
    first_covered, first_covered_valid = _nonnegative_count(
        first.get("covered"), missing=None,
    )
    second_covered, second_covered_valid = _nonnegative_count(
        second.get("covered"), missing=None,
    )
    if (not all((first_eligible_valid, second_eligible_valid,
                 first_covered_valid, second_covered_valid))
            or any(value is None for value in (
                first_eligible, second_eligible, first_covered, second_covered,
            ))):
        issues.append(f"source-{name}-counts-invalid")
        first_eligible = second_eligible = first_covered = second_covered = 0
    if (
        first_eligible <= 0 or second_eligible <= 0
        or first_covered < 0 or second_covered < 0
        or first_covered > first_eligible or second_covered > second_eligible
    ):
        issues.append(f"source-{name}-counts-invalid")
    first_raw_value, second_raw_value = first.get("value"), second.get("value")
    try:
        first_value = (
            float(first_raw_value)
            if isinstance(first_raw_value, (int, float))
            and not isinstance(first_raw_value, bool)
            else None
        )
        second_value = (
            float(second_raw_value)
            if isinstance(second_raw_value, (int, float))
            and not isinstance(second_raw_value, bool)
            else None
        )
    except (OverflowError, ValueError):
        first_value = second_value = None
    if (
        first_value is None or second_value is None
        or not math.isfinite(first_value) or not math.isfinite(second_value)
        or not 0.0 <= first_value <= 1.0 or not 0.0 <= second_value <= 1.0
        or first_eligible <= 0 or second_eligible <= 0
        or not math.isclose(first_value, first_covered / first_eligible,
                            rel_tol=0.0, abs_tol=1.1e-6)
        or not math.isclose(second_value, second_covered / second_eligible,
                            rel_tol=0.0, abs_tol=1.1e-6)
    ):
        issues.append(f"source-{name}-ratio-invalid")
    if first_eligible != second_eligible:
        issues.append(f"source-{name}-eligible-count-mismatch")
    comparable = not issues
    return {
        "status": "comparable" if comparable else "not-comparable",
        "reasons": sorted(set(issues)),
        "baseline_covered": first_covered,
        "candidate_covered": second_covered,
        "baseline_eligible": first_eligible,
        "candidate_eligible": second_eligible,
        "baseline_percent": round(first_value * 100, 4)
        if comparable and first_value is not None else None,
        "candidate_percent": round(second_value * 100, 4)
        if comparable and second_value is not None else None,
        "delta_percentage_points": round(
            (second_value - first_value) * 100, 4,
        ) if comparable and first_value is not None and second_value is not None else None,
        "baseline_denominator_sha256": first_digest,
        "candidate_denominator_sha256": second_digest,
    }


def _compare_source_targets(
    left: Mapping[str, Any], right: Mapping[str, Any], run_issues: list[str],
) -> dict[str, Any]:
    first_rows, first_issues = _source_row_index(left["summary"])
    second_rows, second_issues = _source_row_index(right["summary"])
    if first_rows is None and second_rows is None:
        return {
            "status": "not-recorded",
            "reasons": ["source-coverage-not-recorded"],
            "targets": [],
        }
    if first_rows is None or second_rows is None:
        return {
            "status": "not-comparable",
            "reasons": ["source-coverage-missing-on-one-side"],
            "targets": [],
        }
    def indexed(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {tuple(row["_source_key"]): row for row in rows}
    first_index = indexed(first_rows)
    second_index = indexed(second_rows)
    first_expected, first_scope_issues = _expected_source_scope_keys(left)
    second_expected, second_scope_issues = _expected_source_scope_keys(right)
    scope_issues = [
        *(f"baseline-{issue}" for issue in first_scope_issues),
        *(f"candidate-{issue}" for issue in second_scope_issues),
    ]
    if set(first_index) != first_expected:
        scope_issues.append("baseline-source-coverage-target-scope-mismatch")
    if set(second_index) != second_expected:
        scope_issues.append("candidate-source-coverage-target-scope-mismatch")
    targets = []
    for key in sorted(
        set(first_index) | set(second_index) | first_expected | second_expected,
    ):
        first = first_index.get(key)
        second = second_index.get(key)
        reasons = list(run_issues) + first_issues + second_issues
        if first is None:
            reasons.append("baseline-source-coverage-row-missing")
        if second is None:
            reasons.append("candidate-source-coverage-row-missing")
        if first is None or second is None:
            targets.append({
                "target": key[0], "stratum": key[1],
                "input_scopes": {
                    "baseline": first.get("_input_scopes", []) if first else [],
                    "candidate": second.get("_input_scopes", []) if second else [],
                },
                "status": "not-comparable", "reasons": sorted(set(reasons)),
                "metrics": {},
            })
            continue
        first_source = first.get("simulator_source_coverage")
        second_source = second.get("simulator_source_coverage")
        first_source = first_source if isinstance(first_source, Mapping) else {}
        second_source = second_source if isinstance(second_source, Mapping) else {}
        if first_source.get("schema_version") != SOURCE_SCHEMA:
            reasons.append("baseline-source-coverage-schema-mismatch")
        if second_source.get("schema_version") != SOURCE_SCHEMA:
            reasons.append("candidate-source-coverage-schema-mismatch")
        if first_source.get("status") != "observed":
            reasons.append(f"baseline-source-status-{first_source.get('status') or 'missing'}")
        if second_source.get("status") != "observed":
            reasons.append(f"candidate-source-status-{second_source.get('status') or 'missing'}")
        first_identity = _source_identity(first_source)
        second_identity = _source_identity(second_source)
        for field in _SOURCE_IDENTITY_FIELDS:
            if first_identity.get(field) != second_identity.get(field):
                reasons.append(f"source-coverage-identity-mismatch:{field}")
        for side, identity in (("baseline", first_identity), ("candidate", second_identity)):
            for field in _SOURCE_REQUIRED_IDENTITY_FIELDS:
                value = identity.get(field)
                # An empty exclusion list is a valid declared scope; an empty
                # source root/scope or missing identity value is not.
                missing = value in (None, "") or value == {}
                if field != "source_scope_exclude" and value == []:
                    missing = True
                if missing:
                    reasons.append(f"{side}-source-coverage-identity-missing:{field}")
        first_metrics = first_source.get("metrics")
        second_metrics = second_source.get("metrics")
        first_metrics = first_metrics if isinstance(first_metrics, Mapping) else {}
        second_metrics = second_metrics if isinstance(second_metrics, Mapping) else {}
        metrics = {
            name: _compare_source_metric(
                name,
                first_metrics.get(name) if isinstance(first_metrics.get(name), Mapping) else {},
                second_metrics.get(name) if isinstance(second_metrics.get(name), Mapping) else {},
            ) for name in _SOURCE_METRICS
        }
        for metric in metrics.values():
            reasons.extend(metric.get("reasons", []))
        targets.append({
            "target": key[0], "stratum": key[1],
            "input_scopes": {
                "baseline": first.get("_input_scopes", []),
                "candidate": second.get("_input_scopes", []),
            },
            "status": "comparable" if not reasons else "not-comparable",
            "reasons": sorted(set(reasons)),
            "identities": {"baseline": first_identity, "candidate": second_identity},
            "metrics": metrics,
        })
    status = "comparable" if targets and all(
        item.get("status") == "comparable" for item in targets
    ) and not scope_issues else "not-comparable"
    return {
        "status": status,
        "reasons": sorted(set(
            first_issues + second_issues + scope_issues
            + [reason for item in targets for reason in item.get("reasons", [])]
        )),
        "targets": targets,
    }


def _summary_uses_source_only(summary: Mapping[str, Any]) -> bool:
    """Identify the active SimSrcCov-only summary shape.

    The active contract intentionally omits the historical guest instruction
    registry.  This test is based on the sealed union basis rather than the
    current process configuration so comparison can safely replay old output.
    """
    unions = summary.get("experiment_unions")
    if (not isinstance(unions, list) or not unions
            or any(not isinstance(item, Mapping) for item in unions)):
        return False
    bases = [item.get("coverage_basis") for item in unions]
    return all(
        isinstance(basis, Mapping)
        and basis.get("SimSrcCov") == "target-run-source-profile"
        and not any(name in basis for name in METRICS)
        for basis in bases
    )


def _compare_source_experiment_union(
    left: Mapping[str, Any], right: Mapping[str, Any],
    run_issues: list[str],
) -> dict[str, Any]:
    """Compare active source profiles without inventing a guest denominator."""
    source = _compare_source_targets(left, right, run_issues)
    first = _experiment_union_for_run(left)
    second = _experiment_union_for_run(right)
    first_targets = first.get("targets", []) if isinstance(first, Mapping) else []
    second_targets = second.get("targets", []) if isinstance(second, Mapping) else []
    reasons = list(source.get("reasons", ()))
    scope_indexes = []
    scope_shapes = []
    for side, union, summary in (
        ("baseline", first, left["summary"]),
        ("candidate", second, right["summary"]),
    ):
        if union is None:
            reasons.append(f"{side}-experiment-wide-coverage-union-missing")
            scopes, scope_issues = {}, []
        else:
            if union.get("scope") != "whole-method-run-all-cases-and-targets":
                reasons.append(f"{side}-experiment-coverage-scope-mismatch")
            if union.get("status") != "observed":
                reasons.append(f"{side}-experiment-status-{union.get('status') or 'missing'}")
            scopes, scope_issues = _experiment_scope_index(union)
        reasons.extend(scope_issues)
        scope_indexes.append(scopes)
        scope_shapes.append({(key[0], key[1]) for key in scopes})
        source_rows, row_issues = _source_row_index(summary)
        reasons.extend(row_issues)
        source_keys = {
            tuple(row["_source_key"]) for row in (source_rows or ())
        }
        if source_keys != scope_shapes[-1]:
            reasons.append(f"{side}-source-coverage-target-scope-mismatch")
    if scope_shapes[0] != scope_shapes[1]:
        reasons.append("experiment-target-scope-mismatch")
    for target in sorted({key[0] for shapes in scope_shapes for key in shapes}):
        reasons.extend(_target_identity_issues(left, right, target))
    if reasons:
        source["status"] = "not-comparable"
        source["reasons"] = sorted(set(reasons))
    return {
        "scope": "whole-method-run-all-cases-and-targets",
        "status": source.get("status", "not-comparable"),
        "reasons": sorted(set(reasons)),
        "target_groups": {
            "baseline": len(scope_indexes[0]),
            "candidate": len(scope_indexes[1]),
        },
        "targets": {
            "baseline": first_targets,
            "candidate": second_targets,
        },
        "target_scopes": {
            "baseline": [
                dict(scope_indexes[0][key]) for key in sorted(scope_indexes[0])
            ],
            "candidate": [
                dict(scope_indexes[1][key]) for key in sorted(scope_indexes[1])
            ],
        },
        "source_coverage": source,
        "metrics": {
            "SimSrcCov": {
                "status": source.get("status", "not-comparable"),
                "comparison_scope": "target-local-source-profiles",
                "reasons": sorted(set(reasons)),
            },
        },
        "case_artifact_pairs": {
            "baseline": first.get("case_artifact_pairs") if isinstance(first, Mapping) else None,
            "candidate": second.get("case_artifact_pairs") if isinstance(second, Mapping) else None,
        },
    }


def _registry_for_row(row: Mapping[str, Any],
                      fragment: Mapping[str, Any] | None = None
                      ) -> tuple[dict[str, Any] | None, list[str]]:
    basis = row.get("coverage_basis")
    basis = basis if isinstance(basis, Mapping) else {}
    profile_id = basis.get("coverage_profile_id")
    registry_sha = basis.get("coverage_registry_sha256")
    if not isinstance(profile_id, str) or not profile_id or not isinstance(registry_sha, str):
        return None, ["coverage-profile-or-registry-metadata-missing"]
    parts = profile_id.split("|")
    if len(parts) != 3:
        return None, ["coverage-profile-id-invalid"]
    profile, privilege_mode, _ = parts
    try:
        registry = metric_registry_for_profile(profile, privilege_mode)
    except ValueError:
        return None, ["coverage-profile-unsupported"]
    issues = []
    if registry["metric_profile_id"] != profile_id:
        issues.append("coverage-profile-id-does-not-match-pinned-registry")
    if registry["registry_sha256"] != registry_sha:
        issues.append("coverage-registry-does-not-match-pinned-catalog")
    if isinstance(fragment, Mapping):
        profile_key = f"{profile}@{privilege_mode}"
        for name, kind in METRIC_KINDS.items():
            entry = fragment.get(kind)
            bins = entry.get("eligible_bins_by_profile") if isinstance(entry, Mapping) else None
            registered = bins.get(profile_key) if isinstance(bins, Mapping) else None
            if not isinstance(registered, list) or set(registered) != set(
                registry["metrics"][name]
            ):
                issues.append(f"{name}-sealed-registry-profile-mismatch")
    return registry, issues


def _metric_data(row: Mapping[str, Any], name: str,
                 registry: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    guest = row.get("guest")
    guest = guest if isinstance(guest, Mapping) else {}
    raw = guest.get(name)
    if not isinstance(raw, Mapping):
        return None, [f"{name}-metric-missing"]
    metric = dict(raw)
    issues = []
    if metric.get("status") != "observed":
        issues.append(f"{name}-status-{metric.get('status') or 'missing'}")
    basis = row.get("coverage_basis")
    basis = basis if isinstance(basis, Mapping) else {}
    if basis.get(name) != ROW_BASIS[name]:
        issues.append(f"{name}-aggregation-basis-mismatch")
    profile_id = basis.get("coverage_profile_id")
    if metric.get("coverage_profile_id") not in (None, profile_id):
        issues.append(f"{name}-profile-mismatch")
    if metric.get("coverage_profile_id") is None:
        issues.append(f"{name}-profile-metadata-missing")
    if name == "PCov":
        expected_registry_sha = basis.get("coverage_registry_sha256")
        if metric.get("coverage_registry_sha256") != expected_registry_sha:
            issues.append("PCov-registry-metadata-mismatch")
        for field, reason in (
            ("identity_missing_record_count", "PCov-artifact-identity-missing"),
            ("static_map_missing_record_count", "PCov-static-map-missing"),
            ("invalid_unit_record_count", "PCov-static-unit-set-invalid"),
        ):
            parsed_count, count_valid = _nonnegative_count(metric.get(field), missing=0)
            if not count_valid:
                issues.append(f"{reason}-count-invalid")
            elif parsed_count:
                issues.append(reason)
        if metric.get("aggregation") != "run-union-elf-pc":
            issues.append("PCov-aggregation-mismatch")
        raw_eligible = metric.get("eligible_units")
        raw_covered = metric.get("covered_units")
        if not isinstance(raw_eligible, list) or not isinstance(raw_covered, list):
            return None, issues + ["PCov-unit-sets-missing"]
        if not all(isinstance(unit, str) for unit in [*raw_eligible, *raw_covered]):
            return None, issues + ["PCov-unit-set-invalid"]
        eligible_units = set(raw_eligible)
        covered_units = set(raw_covered)
        unit_pattern = re.compile(r"([0-9a-f]{64})@0x(0|[1-9a-f][0-9a-f]*)\Z")
        if (len(eligible_units) != len(raw_eligible)
                or len(covered_units) != len(raw_covered)):
            issues.append("PCov-unit-set-invalid")
        else:
            for unit in eligible_units | covered_units:
                match = unit_pattern.fullmatch(unit)
                if match is None or int(match.group(2), 16) >= 1 << 64:
                    issues.append("PCov-unit-identity-invalid")
                    break
        if not covered_units <= eligible_units:
            issues.append("PCov-covered-units-invalid")
        covered_count, covered_valid = _nonnegative_count(
            metric.get("covered"), missing=None,
        )
        eligible_count, eligible_valid = _nonnegative_count(
            metric.get("eligible"), missing=None,
        )
        artifact_count, artifact_valid = _nonnegative_count(
            metric.get("artifact_count"), missing=None,
        )
        artifacts_with_hits, hits_valid = _nonnegative_count(
            metric.get("artifacts_with_hits"), missing=None,
        )
        if (not all((covered_valid, eligible_valid, artifact_valid, hits_valid))
                or any(value is None for value in (
                    covered_count, eligible_count, artifact_count, artifacts_with_hits,
                ))):
            return None, issues + ["PCov-summary-counts-invalid"]
        if covered_count != len(covered_units) or eligible_count != len(eligible_units):
            issues.append("PCov-unit-count-mismatch")
        if artifact_count != len({unit[:64] for unit in eligible_units}):
            issues.append("PCov-artifact-count-mismatch")
        if artifacts_with_hits != len({unit[:64] for unit in covered_units}):
            issues.append("PCov-hit-artifact-count-mismatch")
        if metric.get("eligible_units_sha256") != _pcov_unit_digest(eligible_units):
            issues.append("PCov-eligible-set-digest-mismatch")
        if metric.get("covered_units_sha256") != _pcov_unit_digest(covered_units):
            issues.append("PCov-covered-set-digest-mismatch")
        if metric.get("status") != "observed" or not eligible_units:
            if metric.get("value") is not None:
                issues.append("PCov-incomplete-run-union-has-value")
            return None, issues
        value = len(covered_units) / len(eligible_units)
        if not math.isfinite(value) or not _close(metric.get("value"), value):
            issues.append("PCov-run-union-value-mismatch")
        return ({
            "value": value,
            "covered_units": covered_units,
            "eligible_units": eligible_units,
            "artifact_count": artifact_count,
            "artifacts_with_hits": artifacts_with_hits,
        }, issues)

    if registry is None:
        return None, issues + [f"{name}-registry-unavailable"]
    units = metric.get("eligible_units")
    covered = metric.get("covered_units")
    if not isinstance(units, list) or not isinstance(covered, list):
        return None, issues + [f"{name}-unit-sets-missing"]
    if not all(isinstance(item, str) for item in [*units, *covered]):
        return None, issues + [f"{name}-unit-set-invalid"]
    eligible_units = list(units)
    covered_units = list(covered)
    expected = set(registry["metrics"][name])
    if len(set(eligible_units)) != len(eligible_units) or set(eligible_units) != expected:
        issues.append(f"{name}-eligible-universe-incomplete-or-invalid")
    if len(set(covered_units)) != len(covered_units) or not set(covered_units) <= set(eligible_units):
        issues.append(f"{name}-covered-units-invalid")
    eligible_count, eligible_valid = _nonnegative_count(
        metric.get("eligible"), missing=None,
    )
    covered_count, covered_valid = _nonnegative_count(
        metric.get("covered"), missing=None,
    )
    if (not eligible_valid or not covered_valid
            or eligible_count is None or covered_count is None):
        return None, issues + [f"{name}-counts-invalid"]
    if eligible_count != len(eligible_units):
        issues.append(f"{name}-eligible-count-mismatch")
    if covered_count != len(covered_units):
        issues.append(f"{name}-covered-count-mismatch")
    if not _close(metric.get("value"), len(covered_units) / len(eligible_units)
                  if eligible_units else -1):
        issues.append(f"{name}-ratio-mismatch")
    expected_hash = _digest_units(eligible_units)
    if metric.get("registry_sha256") != expected_hash:
        issues.append(f"{name}-denominator-hash-mismatch")
    if metric.get("coverage_profile_id") != registry.get("metric_profile_id"):
        issues.append(f"{name}-profile-mismatch")
    return ({"value": len(covered_units) / len(eligible_units) if eligible_units else 0,
             "covered_units": set(covered_units), "eligible_units": set(eligible_units)}, issues)


def _compare_metric(name: str, left: Mapping[str, Any], right: Mapping[str, Any],
                    left_registry: Mapping[str, Any] | None,
                    right_registry: Mapping[str, Any] | None,
                    comparison_registry: Mapping[str, Any] | None) -> dict[str, Any]:
    first, first_issues = _metric_data(left, name, left_registry)
    second, second_issues = _metric_data(right, name, right_registry)
    issues = sorted(set(first_issues + second_issues))
    _, mode_left = _coverage_profile_parts(left)
    _, mode_right = _coverage_profile_parts(right)
    if mode_left is None or mode_right is None:
        issues.append("coverage-privilege-mode-missing")
    elif mode_left != mode_right:
        issues.append("privilege-mode-mismatch")
    if first is None or second is None:
        return {"status": "not-comparable", "reasons": sorted(set(issues))}
    if name != "PCov":
        if comparison_registry is None:
            issues.append(f"{name}-comparison-universe-unavailable")
            return {"status": "not-comparable", "reasons": sorted(set(issues))}
        eligible = set(comparison_registry["metrics"][name])
        if not first["eligible_units"] <= eligible or not second["eligible_units"] <= eligible:
            issues.append(f"{name}-profile-outside-comparison-universe")
        first_covered = first["covered_units"] & eligible
        second_covered = second["covered_units"] & eligible
        common = first_covered & second_covered
        left_only = first_covered - second_covered
        right_only = second_covered - first_covered
        union = first_covered | second_covered
        first_value = len(first_covered) / len(eligible) if eligible else 0.0
        second_value = len(second_covered) / len(eligible) if eligible else 0.0
        return {
            "status": "not-comparable" if issues else "comparable",
            "reasons": sorted(set(issues)),
            "denominator_profile": COMPARISON_PROFILE,
            "comparison_registry_sha256": comparison_registry["registry_sha256"],
            "denominator_units": len(eligible),
            "baseline_profile_units": len(first["eligible_units"]),
            "candidate_profile_units": len(second["eligible_units"]),
            "baseline_unavailable_units": sorted(eligible - first["eligible_units"]),
            "candidate_unavailable_units": sorted(eligible - second["eligible_units"]),
            "baseline_covered_units": len(first_covered),
            "candidate_covered_units": len(second_covered),
            "baseline_percent": round(first_value * 100, 4),
            "candidate_percent": round(second_value * 100, 4),
            "delta_percentage_points": round((second_value - first_value) * 100, 4),
            "shared_covered_units": len(common),
            "baseline_only_units": sorted(left_only),
            "candidate_only_units": sorted(right_only),
            "covered_set_jaccard": round(len(common) / len(union), 6) if union else None,
        }
    first_covered = first["covered_units"]
    second_covered = second["covered_units"]
    first_eligible = first["eligible_units"]
    second_eligible = second["eligible_units"]
    common_covered = first_covered & second_covered
    common_eligible = first_eligible & second_eligible
    covered_union = first_covered | second_covered
    eligible_union = first_eligible | second_eligible
    return {
        "status": "not-comparable" if issues else "comparable",
        "reasons": sorted(set(issues)),
        "aggregation": "run-union-elf-pc" if name == "PCov" else ROW_BASIS[name],
        "comparison_scope": (
            "experiment-wide-ELF-PC-unions"
            if left.get("scope") == "whole-method-run-all-cases-and-targets"
            else "target-run-ELF-PC-unions"
        ),
        "baseline_percent": round(first["value"] * 100, 4),
        "candidate_percent": round(second["value"] * 100, 4),
        "delta_percentage_points": round((second["value"] - first["value"]) * 100, 4),
        "baseline_covered_units": len(first_covered),
        "candidate_covered_units": len(second_covered),
        "baseline_eligible_units": len(first_eligible),
        "candidate_eligible_units": len(second_eligible),
        "baseline_artifact_count": first["artifact_count"],
        "candidate_artifact_count": second["artifact_count"],
        "shared_covered_units": len(common_covered),
        "baseline_only_covered_units": sorted(first_covered - second_covered),
        "candidate_only_covered_units": sorted(second_covered - first_covered),
        "covered_set_jaccard": round(len(common_covered) / len(covered_union), 6)
        if covered_union else None,
        "shared_eligible_units": len(common_eligible),
        "eligible_set_jaccard": round(len(common_eligible) / len(eligible_union), 6)
        if eligible_union else None,
    }


def _compare_target(left: Mapping[str, Any], right: Mapping[str, Any],
                    key: tuple[str, str], run_issues: list[str]) -> dict[str, Any]:
    target, stratum = key
    first_index, first_issues = _row_index(left["summary"])
    second_index, second_issues = _row_index(right["summary"])
    first, second = first_index.get(key), second_index.get(key)
    issues = list(run_issues) + first_issues + second_issues
    if first is None:
        issues.append("baseline-target-row-missing")
    if second is None:
        issues.append("candidate-target-row-missing")
    if first is None or second is None:
        return {"target": target, "stratum": stratum,
                "status": "not-comparable", "reasons": sorted(set(issues))}
    first_abi, second_abi = _lane_abi(first), _lane_abi(second)
    if first_abi is None or second_abi is None:
        issues.append("abi-lane-missing")
    elif first_abi != second_abi:
        issues.append("abi-mismatch")
    issues.extend(_target_identity_issues(left, right, target))
    if first.get("target_identity_id") and second.get("target_identity_id") \
            and first.get("target_identity_id") != second.get("target_identity_id"):
        issues.append("summary-target-identity-mismatch")
    if first.get("status") != "observed":
        issues.append(f"baseline-target-status-{first.get('status') or 'missing'}")
    if second.get("status") != "observed":
        issues.append(f"candidate-target-status-{second.get('status') or 'missing'}")
    if first.get("method") not in (None, left.get("method")):
        issues.append("baseline-row-method-filter-mismatch")
    if second.get("method") not in (None, right.get("method")):
        issues.append("candidate-row-method-filter-mismatch")
    first_registry, first_registry_issues = _registry_for_row(first, left.get("registry"))
    second_registry, second_registry_issues = _registry_for_row(second, right.get("registry"))
    issues.extend(first_registry_issues + second_registry_issues)
    _, mode_left = _coverage_profile_parts(first)
    _, mode_right = _coverage_profile_parts(second)
    comparison_registry = None
    if mode_left and mode_left == mode_right:
        try:
            comparison_registry = metric_registry_for_profile(COMPARISON_PROFILE, mode_left)
        except ValueError:
            issues.append("comparison-profile-unsupported")
    elif mode_left and mode_right:
        issues.append("privilege-mode-mismatch")
    metrics = {
        name: _compare_metric(
            name, first, second, first_registry, second_registry, comparison_registry,
        )
        for name in METRICS
    }
    rv_instruction_coverage = _compare_target_rv_instruction_coverage(first, second)
    if rv_instruction_coverage.get("status") == "not-comparable":
        issues.extend(rv_instruction_coverage.get("reasons", ()))
    target_issues = sorted(set(issues))
    if target_issues:
        for metric in metrics.values():
            metric["status"] = "not-comparable"
            metric["reasons"] = sorted(set(metric.get("reasons", [])) | set(target_issues))
    return {
        "target": target, "stratum": stratum,
        "lane": first.get("lane"),
        "lanes": {"baseline": first.get("lane"), "candidate": second.get("lane")},
        "routes": {"baseline": first.get("route"), "candidate": second.get("route")},
        "target_identity_id": first.get("target_identity_id") or second.get("target_identity_id"),
        "case_exposure": {
            "baseline": {"cases": first.get("cases"), "observed_cases": first.get("observed_cases")},
            "candidate": {"cases": second.get("cases"), "observed_cases": second.get("observed_cases")},
        },
        "coverage_profile_ids": {
            "baseline": (first.get("coverage_basis") or {}).get("coverage_profile_id"),
            "candidate": (second.get("coverage_basis") or {}).get("coverage_profile_id"),
        },
        "coverage_registry_sha256s": {
            "baseline": (first.get("coverage_basis") or {}).get(
                "coverage_registry_sha256"),
            "candidate": (second.get("coverage_basis") or {}).get(
                "coverage_registry_sha256"),
        },
        "comparison_profile": COMPARISON_PROFILE,
        "comparison_registry_sha256": (
            comparison_registry.get("registry_sha256") if comparison_registry else None
        ),
        "status": "not-comparable" if target_issues or any(
            item["status"] != "comparable" for item in metrics.values()
        ) else "comparable",
        "reasons": target_issues,
        "metrics": metrics,
        "rv_instruction_coverage": rv_instruction_coverage,
    }


def _compare_target_rv_instruction_coverage(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the target's observed ISA/privilege-pair RV registry when present."""
    left = first.get("rv_instruction_coverage")
    right = second.get("rv_instruction_coverage")
    scope = "target-run-fixed-profile-union"
    if not isinstance(left, Mapping) and not isinstance(right, Mapping):
        return {"status": "not-recorded", "scope": scope, "reasons": []}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {
            "status": "not-comparable", "scope": scope,
            "reasons": ["rv-instruction-coverage-missing-on-one-side"],
        }
    result = compare_experiment_unions(
        {"rv_instruction_coverage": left},
        {"rv_instruction_coverage": right},
    )
    reasons = list(result.get("reasons", ()))
    for side, value in (("baseline", left), ("candidate", right)):
        if value.get("aggregation_scope") != scope:
            reasons.append(f"{side}-rv-instruction-coverage-scope-mismatch")
        denominator = value.get("denominator")
        legacy_registry = rv_registry_for_profile(
            EXPERIMENT_COVERAGE_PROFILE, EXPERIMENT_PRIVILEGE_MODE,
        )
        metrics = value.get("metrics")
        legacy_counts = {
            "GenCov": len(legacy_registry["form_units"]),
            "ExecCov": len(legacy_registry["form_units"]),
            "OpcodeCov": len(legacy_registry["opcode_units"]),
        }
        is_valid_legacy = (
            denominator == "pinned-isa-profile-registry"
            and value.get("coverage_profile_id") == legacy_registry["coverage_profile_id"]
            and value.get("coverage_registry_sha256")
            == legacy_registry["coverage_registry_sha256"]
            and isinstance(metrics, Mapping)
            and all(
                isinstance(metrics.get(name), Mapping)
                and metrics[name].get("eligible") == expected
                for name, expected in legacy_counts.items()
            )
        )
        if denominator != "observed-profile-pair-union-registry" and not is_valid_legacy:
            reasons.append(f"{side}-rv-instruction-coverage-denominator-mismatch")
    result["scope"] = scope
    result["status"] = "comparable" if not reasons else "not-comparable"
    result["reasons"] = sorted(set(reasons))
    return result


def _experiment_union_for_run(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    summary = run.get("summary")
    unions = summary.get("experiment_unions") if isinstance(summary, Mapping) else None
    if not isinstance(unions, list):
        return None
    matches = [item for item in unions if isinstance(item, Mapping)
               and item.get("method") == run.get("method")]
    return matches[0] if len(matches) == 1 else None


def _experiment_scope_index(union: Mapping[str, Any]) -> tuple[
    dict[tuple[str, str, str, str], Mapping[str, Any]], list[str]
]:
    rows = union.get("target_scopes")
    if not isinstance(rows, list):
        return {}, ["experiment-target-scopes-missing"]
    result = {}
    issues = []
    for row in rows:
        if not isinstance(row, Mapping):
            issues.append("experiment-target-scope-invalid")
            continue
        key = tuple(str(row.get(name) or "") for name in (
            "target", "stratum", "lane", "route",
        ))
        if not all(key):
            issues.append("experiment-target-scope-key-missing")
        elif key in result:
            issues.append(f"experiment-target-scope-ambiguous:{key[0]}:{key[1]}")
        else:
            result[key] = row
    return result, issues


def _compare_experiment_union(left: Mapping[str, Any], right: Mapping[str, Any],
                              run_issues: list[str]) -> dict[str, Any]:
    first = _experiment_union_for_run(left)
    second = _experiment_union_for_run(right)
    issues = list(run_issues)
    status_issues = []
    if first is None:
        issues.append("baseline-experiment-wide-coverage-union-missing")
    if second is None:
        issues.append("candidate-experiment-wide-coverage-union-missing")
    if first is None or second is None:
        return {"scope": "whole-method-run-all-cases-and-targets",
                "status": "not-comparable", "reasons": sorted(set(issues))}
    first_scopes, first_scope_issues = _experiment_scope_index(first)
    second_scopes, second_scope_issues = _experiment_scope_index(second)
    issues.extend(first_scope_issues + second_scope_issues)
    # Route and ISA lane identify how a method exercised a target.  The
    # experiment scope must remain comparable when generators use different
    # routes or narrower ISA lanes; target, stratum, and ABI define the
    # execution population.  ABI is the part after the lane separator.
    def scope_shape(key: tuple[str, str, str, str]) -> tuple[str, str, str]:
        lane = key[2]
        abi = lane.split("/", 1)[1].strip().lower() if "/" in lane else ""
        return key[0], key[1], abi

    first_scope_shapes = {scope_shape(key) for key in first_scopes}
    second_scope_shapes = {scope_shape(key) for key in second_scopes}
    if first_scope_shapes != second_scope_shapes:
        issues.append("experiment-target-scope-mismatch")
    for target in sorted({key[0] for key in set(first_scopes) | set(second_scopes)}):
        issues.extend(_target_identity_issues(left, right, target))
    for key in set(first_scopes) & set(second_scopes):
        first_identity = first_scopes[key].get("target_identity_id")
        second_identity = second_scopes[key].get("target_identity_id")
        if first_identity and second_identity and first_identity != second_identity:
            issues.append(f"experiment-target-identity-mismatch:{key[0]}:{key[1]}")
    if first.get("scope") != "whole-method-run-all-cases-and-targets" \
            or second.get("scope") != "whole-method-run-all-cases-and-targets":
        issues.append("experiment-coverage-scope-mismatch")
    if first.get("status") != "observed":
        status_issues.append(f"baseline-experiment-status-{first.get('status') or 'missing'}")
    if second.get("status") != "observed":
        status_issues.append(f"candidate-experiment-status-{second.get('status') or 'missing'}")
    issues.extend(status_issues)
    first_registry, first_registry_issues = _registry_for_row(first, left.get("registry"))
    second_registry, second_registry_issues = _registry_for_row(second, right.get("registry"))
    issues.extend(first_registry_issues + second_registry_issues)
    _, mode_left = _coverage_profile_parts(first)
    _, mode_right = _coverage_profile_parts(second)
    comparison_registry = None
    if mode_left and mode_left == mode_right:
        try:
            comparison_registry = metric_registry_for_profile(COMPARISON_PROFILE, mode_left)
        except ValueError:
            issues.append("comparison-profile-unsupported")
    elif mode_left and mode_right:
        issues.append("privilege-mode-mismatch")
    metrics = {
        name: _compare_metric(
            name, first, second, first_registry, second_registry, comparison_registry,
        )
        for name in METRICS
    }
    if issues:
        for metric in metrics.values():
            metric["status"] = "not-comparable"
            metric["reasons"] = sorted(set(metric.get("reasons", [])) | set(issues))
    reasons = sorted(set(issues))
    return {
        "scope": "whole-method-run-all-cases-and-targets",
        "status": "not-comparable" if reasons or any(
            metric["status"] != "comparable" for metric in metrics.values()
        ) else "comparable",
        "reasons": reasons,
        "target_groups": {"baseline": len(first_scopes), "candidate": len(second_scopes)},
        "targets": {"baseline": first.get("targets"), "candidate": second.get("targets")},
        "case_artifact_pairs": {
            "baseline": first.get("case_artifact_pairs"),
            "candidate": second.get("case_artifact_pairs"),
        },
        "comparison_profile": COMPARISON_PROFILE,
        "comparison_registry_sha256": (
            comparison_registry.get("registry_sha256") if comparison_registry else None
        ),
        "metrics": metrics,
    }


def _catalog_opcode_header_issues(
    value: Mapping[str, Any], side: str,
) -> list[str]:
    issues = []
    if value.get("schema_version") != OPCODE_CATALOG_SCHEMA:
        issues.append(f"{side}-opcode-catalog-schema-mismatch")
    if value.get("catalog_id") != OPCODE_CATALOG_ID:
        issues.append(f"{side}-opcode-catalog-id-mismatch")
    if value.get("key_schema") != OPCODE_KEY_SCHEMA:
        issues.append(f"{side}-opcode-key-schema-mismatch")
    if value.get("registry_sha256") != OPCODE_CATALOG_REGISTRY_SHA256:
        issues.append(f"{side}-opcode-catalog-registry-mismatch")
    if value.get("opcode_denominator") != OPCODE_CATALOG_DENOMINATOR:
        issues.append(f"{side}-opcode-denominator-mismatch")
    return issues


def _catalog_opcode_metric_data(
    entry: Mapping[str, Any], name: str, side: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues = []
    metrics = entry.get("metrics")
    metric = metrics.get(name) if isinstance(metrics, Mapping) else None
    if not isinstance(metric, Mapping):
        return None, [f"{side}-{name}-missing"]
    if metric.get("status") != "observed":
        issues.append(f"{side}-{name}-status-{metric.get('status') or 'missing'}")
    if metric.get("registry_sha256") != OPCODE_CATALOG_REGISTRY_SHA256:
        issues.append(f"{side}-{name}-registry-mismatch")
    eligible = metric.get("eligible")
    covered = metric.get("covered")
    units = metric.get("covered_units")
    if (type(eligible) is not int or eligible != OPCODE_CATALOG_DENOMINATOR
            or type(covered) is not int or not 0 <= covered <= OPCODE_CATALOG_DENOMINATOR
            or not isinstance(units, list)
            or any(not isinstance(unit, str) for unit in units)
            or len(set(units)) != len(units)
            or not set(units) <= OPCODE_CATALOG_KEYS
            or len(units) != covered):
        issues.append(f"{side}-{name}-counts-invalid")
    else:
        value = metric.get("value")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)
                or not math.isclose(value, covered / eligible, rel_tol=0.0, abs_tol=1.1e-6)):
            issues.append(f"{side}-{name}-ratio-invalid")
    if issues:
        return None, issues
    return {
        "covered": covered,
        "eligible": eligible,
        "covered_units": set(units),
        "value": covered / eligible,
    }, issues


def _compare_catalog_opcode_metric(
    left: Mapping[str, Any], right: Mapping[str, Any], name: str,
) -> dict[str, Any]:
    first, first_issues = _catalog_opcode_metric_data(left, name, "baseline")
    second, second_issues = _catalog_opcode_metric_data(right, name, "candidate")
    issues = sorted(set(first_issues + second_issues))
    if first is None or second is None:
        return {"status": "not-comparable", "reasons": issues}
    if first["eligible"] != second["eligible"]:
        issues.append(f"{name}-denominator-mismatch")
    shared = first["covered_units"] & second["covered_units"]
    left_only = first["covered_units"] - second["covered_units"]
    right_only = second["covered_units"] - first["covered_units"]
    union = first["covered_units"] | second["covered_units"]
    comparable = not issues
    return {
        "status": "comparable" if comparable else "not-comparable",
        "reasons": sorted(set(issues)),
        "baseline_covered": first["covered"],
        "candidate_covered": second["covered"],
        "eligible": OPCODE_CATALOG_DENOMINATOR,
        "baseline_percent": round(first["value"] * 100, 4),
        "candidate_percent": round(second["value"] * 100, 4),
        "delta_percentage_points": round(
            (second["value"] - first["value"]) * 100, 4,
        ),
        "shared_covered": len(shared),
        "baseline_only_covered": len(left_only),
        "candidate_only_covered": len(right_only),
        "covered_set_jaccard": round(len(shared) / len(union), 6) if union else None,
    }


def _catalog_method_generated_union(
    summary: Mapping[str, Any], method: str,
) -> Mapping[str, Any] | None:
    catalog = summary.get("rv_opcode_catalog_coverage")
    unions = catalog.get("method_generated_unions") if isinstance(catalog, Mapping) else None
    matches = [
        item for item in unions if isinstance(item, Mapping)
        and str(item.get("method") or "") == method
    ] if isinstance(unions, list) else []
    return matches[0] if len(matches) == 1 else None


def _catalog_target_run_index(
    summary: Mapping[str, Any], method: str,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[str]]:
    catalog = summary.get("rv_opcode_catalog_coverage")
    rows = catalog.get("target_runs") if isinstance(catalog, Mapping) else None
    if not isinstance(rows, list):
        return {}, ["opcode-catalog-target-runs-missing"]
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    issues = []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("method") or "") != method:
            continue
        key = (str(row.get("target") or ""), str(row.get("stratum") or ""))
        if not key[0] or not key[1]:
            issues.append("opcode-catalog-target-key-missing")
            continue
        if key in result:
            issues.append(f"opcode-catalog-target-ambiguous:{key[0]}:{key[1]}")
            continue
        result[key] = row
    return result, issues


def _compare_rv_opcode_catalog_coverage(
    left_summary: Mapping[str, Any], right_summary: Mapping[str, Any],
    left_method: str, right_method: str,
) -> dict[str, Any]:
    left = left_summary.get("rv_opcode_catalog_coverage")
    right = right_summary.get("rv_opcode_catalog_coverage")
    if not isinstance(left, Mapping) and not isinstance(right, Mapping):
        return {"status": "not-recorded", "reasons": ["opcode-catalog-coverage-not-recorded"]}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {
            "status": "not-comparable",
            "reasons": ["opcode-catalog-coverage-missing-on-one-side"],
        }
    issues = _catalog_opcode_header_issues(left, "baseline")
    issues.extend(_catalog_opcode_header_issues(right, "candidate"))
    generated_left = _catalog_method_generated_union(left_summary, left_method)
    generated_right = _catalog_method_generated_union(right_summary, right_method)
    if generated_left is None or generated_right is None:
        generated = {
            "status": "not-comparable",
            "reasons": ["method-generated-opcode-union-missing"],
        }
    else:
        generated = _compare_catalog_opcode_metric(
            generated_left, generated_right, "GenOpcodeCov",
        )

    first_targets, first_issues = _catalog_target_run_index(left_summary, left_method)
    second_targets, second_issues = _catalog_target_run_index(right_summary, right_method)
    issues.extend(f"baseline-{item}" for item in first_issues)
    issues.extend(f"candidate-{item}" for item in second_issues)
    if not first_targets:
        issues.append("baseline-opcode-catalog-target-runs-empty")
    if not second_targets:
        issues.append("candidate-opcode-catalog-target-runs-empty")
    target_comparisons = []
    for key in sorted(set(first_targets) | set(second_targets)):
        first = first_targets.get(key)
        second = second_targets.get(key)
        if first is None or second is None:
            target_comparisons.append({
                "target": key[0], "stratum": key[1],
                "status": "not-comparable",
                "reasons": ["opcode-catalog-target-missing-on-one-side"],
            })
            continue
        target_result = _compare_catalog_opcode_metric(
            first, second, "ExecOpcodeCov",
        )
        target_comparisons.append({
            "target": key[0], "stratum": key[1],
            **target_result,
        })
    metric_results = [generated, *target_comparisons]
    for result in metric_results:
        if result.get("status") == "comparable":
            continue
        issues.extend(
            str(reason) for reason in result.get("reasons", ())
            if isinstance(reason, str) and reason
        )
    statuses = [generated.get("status"), *(
        item.get("status") for item in target_comparisons
    )]
    comparable = not issues and bool(statuses) and all(
        status == "comparable" for status in statuses
    )
    return {
        "status": "comparable" if comparable else "not-comparable",
        "reasons": sorted(set(issues)),
        "catalog_id": OPCODE_CATALOG_ID,
        "key_schema": OPCODE_KEY_SCHEMA,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "opcode_denominator": OPCODE_CATALOG_DENOMINATOR,
        "GenOpcodeCov": generated,
        "ExecOpcodeCov_by_target": target_comparisons,
    }


def compare_run_dirs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("provide a baseline run directory and at least one candidate")
    runs = [_load_run(Path(path)) for path in run_dirs]
    baseline = runs[0]
    comparisons = []
    for candidate in runs[1:]:
        run_issues = _run_pair_issues(baseline, candidate)
        baseline_source_only = _summary_uses_source_only(baseline["summary"])
        candidate_source_only = _summary_uses_source_only(candidate["summary"])
        if baseline_source_only and candidate_source_only:
            experiment_union = _compare_source_experiment_union(
                baseline, candidate, run_issues,
            )
            source_coverage = experiment_union["source_coverage"]
            targets = source_coverage.get("targets", [])
            rv_instruction_coverage = {
                "status": "not-recorded",
                "scope": "target-run-fixed-profile-union",
                "reasons": ["active-source-only-coverage"],
            }
        else:
            first_index, first_issues = _row_index(baseline["summary"])
            second_index, second_issues = _row_index(candidate["summary"])
            keys = sorted(set(first_index) | set(second_index))
            targets = [
                _compare_target(baseline, candidate, key,
                                run_issues + first_issues + second_issues)
                for key in keys
            ]
            experiment_union = _compare_experiment_union(
                baseline, candidate, run_issues,
            )
            rv_instruction_coverage = compare_experiment_unions(
                _experiment_union_for_run(baseline),
                _experiment_union_for_run(candidate),
            )
            source_coverage = _compare_source_targets(
                baseline, candidate, run_issues,
            )
        baseline_union = _experiment_union_for_run(baseline)
        candidate_union = _experiment_union_for_run(candidate)
        rv_opcode_catalog_coverage = _compare_rv_opcode_catalog_coverage(
            baseline["summary"], candidate["summary"],
            str(baseline.get("method") or ""),
            str(candidate.get("method") or ""),
        )
        baseline_has_rv = isinstance(
            baseline_union.get("rv_instruction_coverage")
            if isinstance(baseline_union, Mapping) else None,
            Mapping,
        )
        candidate_has_rv = isinstance(
            candidate_union.get("rv_instruction_coverage")
            if isinstance(candidate_union, Mapping) else None,
            Mapping,
        )
        rv_gate_reasons = []
        if baseline_source_only != candidate_source_only:
            rv_gate_reasons.append("coverage-channel-mismatch")
        elif baseline_source_only and candidate_source_only:
            # SimSrcCov has its own fixed source profile identity and does not
            # use the historical guest registry.  The source comparison above
            # is the active gate for this pair.
            pass
        elif baseline_has_rv != candidate_has_rv:
            rv_gate_reasons.append("rv-instruction-coverage-missing-on-one-side")
        elif baseline_has_rv and rv_instruction_coverage.get("status") != "comparable":
            rv_gate_reasons.extend(rv_instruction_coverage.get("reasons", ()))
        source_gate_reasons = []
        if source_coverage.get("status") == "not-comparable":
            source_gate_reasons = [
                "source-coverage-not-comparable",
                *source_coverage.get("reasons", ()),
            ]
        if not targets:
            targets = [{"status": "not-comparable",
                        "reasons": ["no-coverage-target-rows"]}]
        comparisons.append({
            "candidate": {"run_dir": candidate["path"], "method": candidate["method"],
                          "conditions": _run_conditions(candidate)},
            # The method-level union is the comparison result.  Target rows
            # remain diagnostic breakdowns and may be ambiguous when one
            # target is exercised through multiple routes or ISA lanes.
            "status": "not-comparable" if run_issues
            or experiment_union["status"] != "comparable"
            or rv_gate_reasons or source_gate_reasons else "comparable",
            "reasons": sorted(set(
                run_issues + rv_gate_reasons + source_gate_reasons,
            )),
            "experiment_union": experiment_union,
            "rv_instruction_coverage": rv_instruction_coverage,
            "rv_opcode_catalog_coverage": rv_opcode_catalog_coverage,
            "source_coverage": source_coverage,
            "targets": targets,
        })
    active_source_only = _summary_uses_source_only(baseline["summary"])
    protocol = (
        "matched configuration, source, image, resource/time budgets and Target binaries; "
        "active comparisons use target-local fixed simulator source profiles and "
        "SimSrcCov-function/line/branch as the main coverage result; full-catalog "
        "opcode diversity is reported separately as a diagnostic"
        if active_source_only else
        "matched configuration, source, image, resource/time budgets and Target binaries; "
        f"fixed {COMPARISON_PROFILE} ISA denominators; the experiment union combines all "
        "case and target hits once per run; forms outside a run's profile count as uncovered; "
        "per-target results remain available as breakdowns"
    )
    return {
        "schema_version": SCHEMA,
        "status": "not-comparable" if any(
            item["status"] != "comparable" for item in comparisons
        ) else "comparable",
        "registry": {
            "catalog_source": CATALOG_SOURCE,
            "isa_spec_release": ISA_SPEC_RELEASE,
            "decoder_version": DECODER_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "rv_opcode_catalog": {
                "catalog_id": OPCODE_CATALOG_ID,
                "key_schema": OPCODE_KEY_SCHEMA,
                "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
                "opcode_denominator": OPCODE_CATALOG_DENOMINATOR,
            },
            "comparison_profile": COMPARISON_PROFILE,
            "comparison_profiles": {
                mode: {
                    "profile_id": metric_registry_for_profile(
                        COMPARISON_PROFILE, mode,
                    )["metric_profile_id"],
                    "registry_sha256": metric_registry_for_profile(
                        COMPARISON_PROFILE, mode,
                    )["registry_sha256"],
                    "denominator_units": {
                        name: len(metric_registry_for_profile(
                            COMPARISON_PROFILE, mode,
                        )["metrics"][name])
                        for name in METRIC_KINDS
                    },
                }
                for mode in DEFAULT_PRIVILEGE_MODES
            },
        },
        "protocol": protocol,
        "baseline": {"run_dir": baseline["path"], "method": baseline["method"],
                     "conditions": _run_conditions(baseline)},
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare complete RQ1 guest-coverage or SimSrcCov runs"
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="baseline run directory first, followed by one or more candidate run directories",
    )
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = compare_run_dirs(args.run_dirs)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report["status"] == "comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())

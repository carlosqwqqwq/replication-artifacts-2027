"""把目标 root ledger 汇总为可重算的 method-level coverage matrix。"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from collections.abc import Iterable, Mapping
from pathlib import Path

from framework._util import canonical_digest, is_sha256_digest, pc_int, pc_path
from framework.direct_case import _SIGNAL_CODE_BY_NAME
from framework.adapters.contracts import (
    PATH_IDENTITY_WITNESS_CONTRACT,
    canonical_execution_backend, is_execution_backend,
    same_execution_backend, _observer_value_is_valid,
)
from framework.adapters.riscv_dv import _normalize_generation_profile
from framework.adapters.program import (
    _comparison_qualified, _observation_outcome_valid, _observer_field_value, compare_observations,
    target_baseline_contract_valid,
)
from framework.adapters.runner import guest_trap_observed
from framework.execution_identity import (
    TargetBinaryIdentity, _target_identity_containers_match, _target_identity_parts,
)
from framework.evidence.recipe_contract import (
    _FORMAL_TARGET_STATUSES, _REFERENCE_EXPECTED_STATUSES, _recipe_contains_trap_marker,
    _recipe_gaps as _recipe_gaps_impl, _recipe_route, _recipe_route_name, _route_alias,
)
from framework.evidence.target_binding import resolve_root_recipe, root_binding_for_recipe as _root_binding_for_recipe
from framework.reference_fallback import (
    configured_k1_qemu_fallback_classes,
    qemu_fallback_policy_matches,
)
from framework.evidence.target_contract import (
    _generation_profile_digest, _normal_outcome, _positive_count, _recipe_digest,
    _recipe_target_identity, _target_claimed,
    _target_status_coherence,
)
from framework.target_evidence import iter_materialized_records, materialize_record

_contains_trap_marker = _recipe_contains_trap_marker
_recipe_gaps = _recipe_gaps_impl
root_binding_for_recipe = _root_binding_for_recipe


METHODS = ("B0", "B1", "B2", "B3")
EXPERIMENT_METHODS = ("S0", "S1", "S2", "S3", "S4", "S5")
RULES = ("M5", *(f"R{i}" for i in range(1, 12)))
ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT.with_name(ROOT.name + ".local")


FIELDS = (
    "reachable", "generation_gap", "generated", "seed_valid", "rewrite_valid", "reference_valid",
    "profile_gap", "transport_gap", "seed_miss", "reference_rejected",
    "target_attempted", "target_tested", "target_raw_recorded", "target_comparison_pending",
    "target_gap", "artifact_gap", "target_unsupported",
    "mismatch_candidate",
    "clean", "cost",
)
MAPPING_FIELDS = ("route", "rule", "state", "observer", "target")
_ROOT_BINDING_FIELDS = (
    "root_key", "lineage_key", "route", "method", "rule", "state", "observer", "target",
)
_FORMAL_LEDGER_SCHEMAS = frozenset({"coverage-target-root-ledger-v1"})
_OBSERVATION_STATUSES = frozenset({
    "normal", "completed", "trap", "nonzero-exit", "timeout", "unavailable",
    "runner-contract-gap",
})
_REFERENCE_IDENTITY_FIELDS = frozenset({
    "backend", "profile_id", "input_id", "tool_version", "source_repository",
    "source_commit", "schema_version", "binary_sha256", "identity_digest",
    "container_image", "container_image_digest", "source_provenance",
    "build_flags", "patch_lineage",
})


_CAPSULE_BACKENDS = frozenset({
    "unicorn-riscv64", "rax-riscv64", "rvvm-riscv64",
})


def _raw_target_pending(value: object) -> bool:
    """原始 Target 结果已落盘，但离线比较尚未发生。"""
    return isinstance(value, Mapping) and (
        value.get("status") == "raw-recorded"
        and value.get("comparison_pending") is True
    )


def _capsule_artifact_valid(
    observation: Mapping[str, object], executable_sha256: object,
) -> bool:
    evidence = observation.get("translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    configuration = details.get("configuration_identity") if isinstance(details, Mapping) else None
    capsule = configuration.get("capsule_sha256") if isinstance(configuration, Mapping) else None
    backend = observation.get("backend")
    if (
        backend not in _CAPSULE_BACKENDS
        or not isinstance(details, Mapping)
        or not isinstance(configuration, Mapping)
        or configuration.get("target_identity_status") != "verified"
        or not is_sha256_digest(capsule)
    ):
        return False
    if backend == "rvvm-riscv64" and details.get("capsule_mode") != "bare-metal-no-ecall":
        return False
    values = tuple(observation.get(name) for name in ("binary_sha256", "guest_elf_sha256"))
    return any(is_sha256_digest(value) for value in values) and all(
        value is None or value in {executable_sha256, capsule} for value in values
    )
def _target_expected_identity_gap(
    record: Mapping[str, object], recipe: Mapping[str, object],
) -> bool:
    expected_digest, expected_binary = _recipe_target_identity(recipe)
    if expected_binary is None and expected_digest is None:
        return False
    values = []
    pending_seen = False
    def identity_pair(value: object) -> tuple[object, object]:
        identity = _execution_identity(value)
        return identity.get("target_binary_sha256"), identity.get("target_identity_digest")

    targets = record.get("targets")
    if isinstance(targets, Mapping):
        binding = record.get("root_binding")
        target_name = binding.get("target") if isinstance(binding, Mapping) else None
        target = targets.get(target_name)
        if not isinstance(target, Mapping) and isinstance(target_name, str):
            target = next(
                (item for name, item in targets.items()
                 if isinstance(name, str)
                 and same_execution_backend(name, target_name)),
                None,
            )
        if isinstance(target, Mapping):
            if _raw_target_pending(target):
                pending_seen = True
            else:
                values.append(identity_pair(target))
    target_rows = record.get("target_records") or record.get("target")
    if isinstance(target_rows, (list, tuple)):
        for target in target_rows:
            if _raw_target_pending(target):
                pending_seen = True
                continue
            comparison = target.get("comparison") if isinstance(target, Mapping) else None
            observations = comparison.get("observations") if isinstance(comparison, Mapping) else None
            for side in ("original", "variant"):
                for observation in observations.get(side, ()) if isinstance(observations, Mapping) else ():
                    values.append(identity_pair(observation))
    target_baseline = record.get("target_baseline")
    if isinstance(target_baseline, Mapping):
        if _raw_target_pending(target_baseline):
            pending_seen = True
        else:
            for observation in target_baseline.get("observations", ()):
                if not isinstance(observation, Mapping):
                    continue
                values.append(identity_pair(observation))
    if not values and pending_seen:
        return False
    return not values or any(
        (expected_binary is not None and binary != expected_binary)
        or (expected_digest is not None and digest != expected_digest)
        for binary, digest in values
    )


def _formal_target_baseline_gap(
    record: Mapping[str, object], expected_backend: object, expected_status: object = None,
) -> tuple[str | None, int, int]:
    baseline = record.get("target_baseline")
    if not isinstance(baseline, Mapping):
        return None, 0, 0
    status = baseline.get("status")
    if not isinstance(status, str):
        return "target-status-invalid", 1, 0
    if _raw_target_pending(baseline):
        return None, 1, 0
    probe_status = status in {"target-probe-tested", "target-probe-gap"}
    if status == "case-skipped":
        return "target-unsupported", 1, 0
    if status == "artifact-gap":
        return "artifact-gap", 1, 0
    if status in {"target-gap", "transport-gap", "target-unavailable", "runner-contract-gap"}:
        return "target-status-gap", 1, 0
    if status != "target-tested" and not probe_status:
        return "target-status-invalid", 1, 0
    coverage_only = baseline.get("coverage_only") is True and status == "target-tested"
    if status == "target-tested" and not coverage_only \
            and not _comparison_qualified(baseline):
        return "target-comparison-unqualified", 1, 0
    if coverage_only and baseline.get("comparison_qualified") is True:
        return "target-coverage-only-qualified", 1, 0
    observations = baseline.get("observations")
    if not isinstance(observations, list) or not observations:
        return "target-observations-invalid", 1, 0
    if any(not isinstance(row, Mapping) or not _observation_outcome_valid(row) for row in observations):
        return "target-observations-invalid", 1, 0
    backends = tuple(
        item.get("backend") for item in observations if isinstance(item, Mapping)
    )
    if len(backends) != len(observations) or any(
        not isinstance(item, str) or not item.strip() for item in backends
    ):
        return "target-observation-invalid", 1, 0
    if isinstance(expected_backend, str) and any(
        not same_execution_backend(item, expected_backend) for item in backends
    ):
        return "target-backend-mismatch", 1, 0
    reference = record.get("reference_baseline")
    if not coverage_only and isinstance(reference, Mapping) and (
        reference.get("reference_expected_contract") == "unverified"
        or any(reference.get(name) for name in (
            "qualification_gaps", "missing_fields", "validation_gap",
            "contract_error", "target_execution_contract_error",
        ))
    ):
        return "target-reference-unqualified", 1, 0
    reference_identity = (
        _execution_identity(reference.get("reference_identity"))
        if isinstance(reference, Mapping) else {}
    )
    reference_backend = reference_identity.get("backend")
    shared_qemu_fallback = (
        reference_backend == "qemu-riscv64"
        and _recorded_qemu_reference_fallback(record)
    )
    if not coverage_only and isinstance(reference_backend, str) and any(
        same_execution_backend(item, reference_backend) for item in backends
    ) and not shared_qemu_fallback:
        return "target-backend-mismatch", 1, 0
    if coverage_only:
        if any(not is_sha256_digest(baseline.get(name)) for name in (
            "source_sha256", "executable_sha256",
        )):
            return "target-artifact-identity-mismatch", 1, 0
    else:
        if not isinstance(reference, Mapping):
            return "target-artifact-identity-mismatch", 1, 0
        for name in ("source_sha256", "executable_sha256"):
            expected = reference.get(name)
            if not is_sha256_digest(expected) or baseline.get(name) != expected:
                return "target-artifact-identity-mismatch", 1, 0
    executable_sha256 = baseline.get("executable_sha256")
    if is_sha256_digest(executable_sha256) and any(
        not isinstance(observation, Mapping)
        or is_execution_backend(observation.get("backend"))
        and not (
            _capsule_artifact_valid(observation, executable_sha256)
            or (
                is_sha256_digest(
                    observation.get("guest_elf_sha256") or observation.get("binary_sha256")
                )
                and observation.get("guest_elf_sha256") in (None, executable_sha256)
                and observation.get("binary_sha256") in (None, executable_sha256)
            )
        )
        for observation in observations
    ):
        return "target-artifact-identity-mismatch", 1, 0
    binding = record.get("root_binding")
    expected_trap = expected_status in {"trap", "nonzero-exit", "illegal-instruction"} \
        or isinstance(binding, Mapping) and binding.get("expected_trap") is True
    if any(
        not target_baseline_contract_valid(
            observation, binding if isinstance(binding, Mapping) else None,
            expected_trap=expected_trap,
        )
        for observation in observations
    ):
        return "target-observation-invalid", 1, 0
    comparison = baseline.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("comparison_mode") != "cross-backend":
        return "target-comparison-mode-invalid", 1, 0
    comparison_status = comparison.get("status")
    counts = record.get("counts")
    mismatch_claim = isinstance(counts, Mapping) \
        and _positive_count(counts.get("target_mismatch_candidate"))
    if (
        not record.get("target_records") and not record.get("targets")
        and isinstance(counts, Mapping)
    ):
        expected_mismatch = int(
            not coverage_only
            and comparison_status == "non-equivalent"
            and _comparison_qualified(baseline, "non-equivalent")
        )
        for name, expected in (
            ("target_mismatch_candidate", expected_mismatch),
            ("clean", 0),
        ):
            if name in counts and counts.get(name) != expected:
                return "target-counts-mismatch", 1, 0
    if coverage_only and comparison_status != "reference-gap":
        return "target-comparison-status-mismatch", 1, 0
    if not coverage_only and expected_status == "clean" and comparison_status != "equivalent" and not mismatch_claim:
        return "target-comparison-status-mismatch", 1, 0
    if not coverage_only and expected_status in {"target-mismatch-candidate", "target-state-mismatch-candidate"} \
            and comparison_status != "non-equivalent":
        return "target-comparison-status-mismatch", 1, 0
    if expected_status in {"normal", "completed"} and any(
        _normal_outcome(row.get("outcome"), row.get("extra_state")) != "normal"
        for row in observations
    ):
        return "target-status-mismatch", 1, 0
    if expected_status in {"trap", "nonzero-exit", "illegal-instruction"} and any(
        row.get("outcome") not in {"trap", "nonzero-exit"}
        for row in observations
    ):
        return "target-status-mismatch", 1, 0
    if coverage_only:
        return None, 1, 0
    observer_fields = (
        binding.get("comparison_fields", binding.get("observer_fields"))
        if isinstance(binding, Mapping) else None
    )
    try:
        actual = compare_observations(
            tuple(reference.get("observation")), tuple(observations), observer_fields,
            cross_backend=True,
        )
    except (TypeError, ValueError):
        return "target-observations-invalid", 1, 0
    if (
        actual.get("status") != comparison.get("status")
        or canonical_digest(actual.get("differences", ()))
        != canonical_digest(comparison.get("differences", ()))
    ):
        return "target-comparison-status-mismatch", 1, 0
    if not record.get("target_records") and not record.get("targets"):
        counts = record.get("counts") if isinstance(record.get("counts"), Mapping) else {}
        mismatch = comparison.get("status") == "non-equivalent"
        if (type(counts.get("target_mismatch_candidate")) is not int
                or counts["target_mismatch_candidate"] != int(mismatch)):
            return "target-comparison-status-mismatch", 1, 0
    return (None, 0, 0) if probe_status else (None, 1, 1)


def _formal_target_gap(record: Mapping[str, object], recipe: Mapping[str, object]) -> str | None:
    if not _target_claimed(record):
        return None
    target_recipe = recipe.get("target")
    expected_backend = (
        target_recipe.get("backend")
        if isinstance(target_recipe, Mapping) else None
    )
    if isinstance(target_recipe, Mapping) and "backend" in target_recipe \
            and not isinstance(expected_backend, str):
        return "target-backend-missing"
    reference_recipe = recipe.get("reference")
    reference_backend = (
        reference_recipe.get("backend")
        if isinstance(reference_recipe, Mapping) else None
    )
    # A shared reference/target backend weakens differential attribution but
    # does not invalidate target coverage or the raw observation record.
    root = record.get("root")
    stage_statuses = (
        root.get("status") if isinstance(root, Mapping) else None,
        record.get("terminal_status"), record.get("status"),
    )
    coverage_only_baseline = (
        isinstance(record.get("target_baseline"), Mapping)
        and record["target_baseline"].get("coverage_only") is True
        and record["target_baseline"].get("status") == "target-tested"
    )
    if any(
        isinstance(value, str) and value in {
            "generation-gap", "transport-gap", "profile-gap", "seed-miss",
        }
        or value == "reference-rejected" and not coverage_only_baseline
        for value in stage_statuses
    ):
        return None
    if _target_expected_identity_gap(record, recipe):
        return "target-expected-identity-mismatch"
    counts = record.get("counts")
    if isinstance(counts, Mapping) and type(counts.get("proposal_gap")) is int \
            and counts["proposal_gap"] > 0:
        return "proposal-gap"
    expected_status = (
        target_recipe.get("expected", {}).get("status")
        if isinstance(target_recipe, Mapping)
        and isinstance(target_recipe.get("expected"), Mapping) else None
    )
    if expected_status is not None and not isinstance(expected_status, str):
        return "target-status-invalid"
    binding = record.get("root_binding")
    coverage = record.get("framework_coverage")
    # Path identity is a coverage witness, not a prerequisite for a state
    # comparison when coverage was not requested.
    path_required = (
        isinstance(coverage, Mapping)
        and coverage.get("status") not in {None, "disabled", "not-requested"}
    )
    expected_trap = isinstance(binding, Mapping) and binding.get("expected_trap") is True
    if expected_trap and expected_status in {"normal", "completed"}:
        expected_status = "trap"
    recipe_expected_status = expected_status
    baseline_gap, baseline_attempted, baseline_tested = _formal_target_baseline_gap(
        record, expected_backend, expected_status,
    )
    if baseline_gap is not None and baseline_gap not in {"target-unsupported", "artifact-gap"}:
        return baseline_gap

    def target_rows_for_count() -> tuple[Mapping[str, object], ...]:
        rows = record.get("target_records")
        if not rows:
            rows = record.get("target")
        if not isinstance(rows, (list, tuple, Mapping)):
            rows = record.get("targets")
        if isinstance(rows, Mapping):
            rows = tuple(
                item for value in rows.values()
                for item in (value if isinstance(value, (list, tuple)) else (value,))
            )
        return tuple(item for item in rows if isinstance(item, Mapping)) \
            if isinstance(rows, (list, tuple)) else ()

    def count_gap(
        attempted: int, tested: int, *, unsupported: int = 0,
        artifact_gaps: int = 0,
    ) -> str | None:
        counts = record.get("counts")
        names = (
            "target_attempted", "target_tested", "target_gap", "target_unsupported",
        )
        if not isinstance(counts, Mapping) or not any(name in counts for name in names):
            return None
        values = tuple(counts.get(name, 0) for name in names)
        raw_artifact_gaps = counts.get("artifact_gap", 0)
        rows = target_rows_for_count()
        coverage_only_completed = sum(
            item.get("coverage_only") is True
            and item.get("target_attempted") is not False
            and (
                item.get("status") in {
                    "clean", "target-mismatch-candidate", "target-tested",
                }
                or item.get("terminal_disposition") in {
                    "clean", "target-mismatch-candidate", "target-tested",
                }
            )
            for item in rows
        )
        baseline = record.get("target_baseline")
        coverage_only_completed += int(
            isinstance(baseline, Mapping)
            and baseline.get("coverage_only") is True
            and baseline.get("status") == "target-tested"
        )
        raw_pending = sum(
            item.get("target_attempted") is not False
            for item in rows if _raw_target_pending(item)
        )
        raw_pending += int(_raw_target_pending(baseline))
        recorded_unsupported = counts.get("target_unsupported", counts.get("case_skipped", 0))
        return (
            "target-counts-invalid"
            if any(type(value) is not int or value < 0 for value in values)
            or type(recorded_unsupported) is not int or recorded_unsupported < 0
            or type(raw_artifact_gaps) is not int or raw_artifact_gaps < 0
            else "target-counts-mismatch"
            if (
                values[0] != attempted
                or values[1] != tested
                or recorded_unsupported != unsupported
                or "case_skipped" in counts and counts["case_skipped"] != unsupported
                or "artifact_gap" in counts and raw_artifact_gaps != artifact_gaps
                or values[2] + values[3] + coverage_only_completed + raw_pending
                != attempted - tested
            ) else None
        )

    def status_gap(target: Mapping[str, object]) -> tuple[str | None, str | None]:
        status = target.get("status")
        if status is None:
            return None, "target-status-missing"
        if not isinstance(status, str):
            return None, "target-status-invalid"
        if _raw_target_pending(target):
            return status, None
        coverage_only = target.get("coverage_only") is True
        if status == "target-tested" and coverage_only:
            if target.get("candidate") is True or target.get("comparison_qualified") is True:
                return status, "coverage-only-candidate-claim"
            return status, None
        if status == "case-skipped":
            return status, "target-unsupported"
        if status == "artifact-gap":
            return status, "artifact-gap"
        if status in {"target-gap", "runner-contract-gap", "target-unavailable", "transport-gap"}:
            return status, "target-status-gap"
        if status in {"target-probe-tested", "target-probe-gap"}:
            return status, None
        if status not in _FORMAL_TARGET_STATUSES:
            return status, "target-status-invalid"
        if coverage_only:
            return status, "coverage-only-candidate-claim"
        if gap := _target_status_coherence(target):
            return status, gap
        expected_comparison = (
            "equivalent" if status == "clean" else "non-equivalent"
        )
        if not _comparison_qualified(target, expected_comparison):
            return status, "target-comparison-unqualified"
        raw_status = target.get("target_result_status")
        if raw_status is not None and (
            status in _FORMAL_TARGET_STATUSES
            and target.get("coverage_only") is not True
            and raw_status != status
            or status == "target-gap" and raw_status in _FORMAL_TARGET_STATUSES
        ):
            return status, "target-result-status-mismatch"
        if recipe_expected_status in _FORMAL_TARGET_STATUSES and status != recipe_expected_status \
                and not (
                    recipe_expected_status == "clean"
                    and status in {"target-mismatch-candidate", "target-state-mismatch-candidate"}
                ):
            return status, "target-status-mismatch"
        return status, None

    targets = record.get("targets")
    candidate_rows = record.get("target_records")
    if candidate_rows is None or (
        isinstance(candidate_rows, (list, tuple)) and not candidate_rows
    ):
        target_rows = record.get("target")
        if isinstance(target_rows, (list, tuple)):
            candidate_rows = target_rows
        elif target_rows not in (None, "") and not (
            isinstance(target_rows, str)
            and isinstance(record.get("root_binding"), Mapping)
            and target_rows == record["root_binding"].get("target")
        ):
            candidate_rows = target_rows
    candidate_absent = candidate_rows is None or (
        isinstance(candidate_rows, (list, tuple)) and not candidate_rows
    )
    if (
        candidate_absent and not isinstance(targets, Mapping)
        and isinstance(baseline := record.get("target_baseline"), Mapping)
        and baseline.get("status") in {"target-probe-tested", "target-probe-gap"}
    ):
        return count_gap(0, 0)
    if baseline_attempted and not isinstance(targets, Mapping) and candidate_absent:
        baseline = record.get("target_baseline")
        baseline_status = baseline.get("status") if isinstance(baseline, Mapping) else None
        if baseline_status == "target-tested":
            identity_status = _target_identity_status(record)
            if identity_status in {"same", "missing", "invalid", "binding-mismatch"}:
                return f"target-identity-{identity_status}"
        unsupported = int(baseline_status == "case-skipped" and baseline_attempted > 0)
        artifact_gaps = int(baseline_status == "artifact-gap")
        if gap := count_gap(
            baseline_attempted, baseline_tested,
            unsupported=unsupported, artifact_gaps=artifact_gaps,
        ):
            return gap
        return baseline_gap
    if isinstance(targets, Mapping):
        binding = record.get("root_binding")
        name = binding.get("target") if isinstance(binding, Mapping) else None
        target = targets.get(name)
        if not isinstance(target, Mapping) and isinstance(name, str):
            target = next(
                (
                    item for target_name, item in targets.items()
                    if isinstance(target_name, str)
                    and same_execution_backend(target_name, name)
                ),
                None,
            )
        if not isinstance(target, Mapping):
            return "target-record-missing"
        actual_name = next(
            (target_name for target_name, value in targets.items() if value is target),
            None,
        )
        if isinstance(expected_backend, str) and not same_execution_backend(actual_name, expected_backend):
            return "target-backend-mismatch"
        if isinstance(expected_backend, str) and target.get("target_backend") not in (None, "") \
                and not same_execution_backend(target.get("target_backend"), expected_backend):
            return "target-backend-mismatch"
        status, gap = status_gap(target)
        if gap is not None:
            if status in {"case-skipped", "artifact-gap"}:
                baseline = record.get("target_baseline")
                baseline_status = baseline.get("status") if isinstance(baseline, Mapping) else None
                baseline_attempt = baseline_attempted if baseline_status in {
                    "case-skipped", "artifact-gap",
                } else 0
                target_attempt = int(target.get("target_attempted") is not False)
                count_error = count_gap(
                    baseline_attempt + target_attempt, baseline_tested,
                    unsupported=int(baseline_status == "case-skipped" and baseline_attempt > 0)
                    + int(status == "case-skipped" and target_attempt > 0),
                    artifact_gaps=int(baseline_status == "artifact-gap")
                    + int(status == "artifact-gap"),
                )
                return count_error or gap
            return gap
        identity_status = _target_identity_status(record)
        if identity_status in {"same", "missing", "invalid", "binding-mismatch"}:
            return f"target-identity-{identity_status}"
        coverage_only = target.get("coverage_only") is True
        if status in _FORMAL_TARGET_STATUSES:
            if target.get("terminal_disposition") != target.get("status"):
                return "target-terminal-status-mismatch"
            if not coverage_only:
                if path_required and target.get("target_route_kind") == "discovery" and (
                    target.get("target_path_stable") is not True
                    or not is_sha256_digest(target.get("target_path_identity"))
                ):
                    return "target-path-evidence-gap"
        if status in {"target-probe-tested", "target-probe-gap"}:
            return baseline_gap
        counts = record.get("counts")
        if (
            isinstance(counts, Mapping)
            and not coverage_only
            and status in _FORMAL_TARGET_STATUSES
            and any(name in counts for name in ("clean", "target_mismatch_candidate"))
        ):
            mismatch = status in {"target-mismatch-candidate", "target-state-mismatch-candidate"}
            if (
                counts.get("clean") != int(not mismatch)
                or counts.get("target_mismatch_candidate") != int(mismatch)
            ):
                return "target-counts-mismatch"
        if gap := count_gap(
            baseline_attempted + int(target.get("target_attempted") is not False),
            baseline_tested + int(status in _FORMAL_TARGET_STATUSES and not coverage_only),
            unsupported=int(
                isinstance(record.get("target_baseline"), Mapping)
                and record["target_baseline"].get("status") == "case-skipped"
                and baseline_attempted > 0
            ),
            artifact_gaps=int(
                isinstance(record.get("target_baseline"), Mapping)
                and record["target_baseline"].get("status") == "artifact-gap"
            ),
        ):
            return gap
        return baseline_gap
    target_rows = candidate_rows
    if not isinstance(target_rows, (list, tuple)) or not target_rows:
        return baseline_gap or "target-record-missing"
    identity_status = _target_identity_status(record)
    if identity_status in {"same", "missing", "invalid", "binding-mismatch"}:
        return f"target-identity-{identity_status}"
    attempted = tested = 0
    unsupported = artifact_gaps = 0
    terminal_gap = None
    binding = record.get("root_binding")
    for target in target_rows:
        if not isinstance(target, Mapping):
            return "target-record-invalid"
        status, gap = status_gap(target)
        row_attempted = target.get("target_attempted") is not False
        if status in {"case-skipped", "artifact-gap"}:
            attempted += int(row_attempted)
            unsupported += int(status == "case-skipped" and row_attempted)
            artifact_gaps += int(status == "artifact-gap")
            terminal_gap = terminal_gap or gap
            continue
        if gap is not None:
            return gap
        if status in _FORMAL_TARGET_STATUSES and target.get("terminal_disposition") != status:
            return "target-terminal-status-mismatch"
        if status in {"target-probe-tested", "target-probe-gap"}:
            continue
        attempted += int(row_attempted)
        if _raw_target_pending(target):
            continue
        coverage_only = target.get("coverage_only") is True
        if coverage_only and status == "target-tested":
            comparison = target.get("comparison")
            observations = comparison.get("observations") if isinstance(comparison, Mapping) else None
            if target.get("terminal_disposition") != status:
                return "target-terminal-status-mismatch"
            if not isinstance(observations, Mapping):
                return "target-observations-invalid"
            for side in ("original", "variant"):
                side_rows = observations.get(side)
                if not isinstance(side_rows, list) or not side_rows or any(
                    not isinstance(row, Mapping)
                    or row.get("outcome") not in _OBSERVATION_STATUSES
                    or not _observation_outcome_valid(row)
                    or isinstance(expected_backend, str)
                    and not same_execution_backend(row.get("backend"), expected_backend)
                    for row in side_rows
                ):
                    return "target-observation-invalid"
            continue
        if coverage_only:
            return "coverage-only-candidate-claim"
        tested += int(status in _FORMAL_TARGET_STATUSES)
        comparison = target.get("comparison")
        comparison_status = comparison.get("status") if isinstance(comparison, Mapping) else None
        comparison_expected_status = "equivalent" if status == "clean" else "non-equivalent"
        if comparison_status != comparison_expected_status:
            return "target-comparison-status-mismatch"
        observations = comparison.get("observations") if isinstance(comparison, Mapping) else None
        if not isinstance(observations, Mapping):
            return "target-observations-invalid"
        for side in ("original", "variant"):
            rows = observations.get(side)
            if not isinstance(rows, list) or not rows:
                return "target-observations-invalid"
            if any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("backend"), str) or not row["backend"].strip()
                or row.get("outcome") not in _OBSERVATION_STATUSES
                or not _observation_outcome_valid(row)
                for row in rows
            ):
                return "target-observation-invalid"
            if recipe_expected_status in {"normal", "completed"} and any(
                _normal_outcome(row.get("outcome"), row.get("extra_state")) != "normal"
                for row in rows
            ):
                return "target-status-mismatch"
            if recipe_expected_status in {"trap", "nonzero-exit", "illegal-instruction"} and any(
                row.get("outcome") not in {"trap", "nonzero-exit"}
                for row in rows
            ):
                return "target-status-mismatch"
            if isinstance(expected_backend, str) and any(
                    not same_execution_backend(row.get("backend"), expected_backend)
                    for row in rows
            ):
                return "target-backend-mismatch"
            for row in rows:
                extra_state = row.get("extra_state") if isinstance(row, Mapping) else None
                terminal_trap = guest_trap_observed(extra_state)
                evidence = row.get("translation_evidence") if isinstance(row, Mapping) else None
                details = evidence.get("details") if isinstance(evidence, Mapping) else None
                path = details.get("path_identity") if isinstance(details, Mapping) else None
                path_digest = next(
                    (path.get(name) for name in ("digest", "path_digest", "target_path_digest")
                     if isinstance(path, Mapping) and is_sha256_digest(path.get(name))),
                    None,
                )
                path_pcs = path.get("executed_pcs") if isinstance(path, Mapping) else None
                try:
                    observed_pcs = tuple(pc_int(value) for value in row.get("executed_pcs", ()))
                    witnessed_pcs = tuple(pc_int(value) for value in path_pcs)
                except (TypeError, ValueError):
                    observed_pcs = witnessed_pcs = ()
                expected_path_digest = pc_path(observed_pcs)[1] if observed_pcs else None
                witness = path.get("executed_pc") if isinstance(path, Mapping) else None
                if path_required and not terminal_trap and (
                    not isinstance(path, Mapping) or path.get("status") != "observed"
                    or path.get("contract") != PATH_IDENTITY_WITNESS_CONTRACT
                    or not witnessed_pcs or witnessed_pcs != observed_pcs
                    or path_digest != expected_path_digest
                    or not isinstance(witness, Mapping)
                    or witness.get("observed") is not True
                    or witness.get("count") != len(observed_pcs)
                    or witness.get("digest") != expected_path_digest
                ):
                    return "target-path-evidence-gap"
        observer_fields = (
            binding.get("comparison_fields", binding.get("observer_fields"))
            if isinstance(binding, Mapping) else None
        )
        try:
            actual = compare_observations(
                tuple(observations["original"]), tuple(observations["variant"]), observer_fields
            )
        except (TypeError, ValueError):
            actual = {"status": "reference-gap", "differences": []}
        if actual.get("status") == "reference-gap":
            if "differences" in comparison or isinstance(record.get("counts"), Mapping):
                return "target-observations-invalid"
        elif (
            actual.get("status") != comparison_expected_status
            or canonical_digest(actual.get("differences", []))
            != canonical_digest(comparison.get("differences", []))
        ):
            return "target-comparison-status-mismatch"
        if status == "clean":
            original, variant = observations.get("original"), observations.get("variant")
            if any(_normal_outcome(left.get("outcome"), left.get("extra_state"))
                   != _normal_outcome(right.get("outcome"), right.get("extra_state"))
                   for left, right in zip(original, variant)):
                return "target-comparison-status-mismatch"
    baseline = record.get("target_baseline")
    baseline_status = baseline.get("status") if isinstance(baseline, Mapping) else None
    if gap := count_gap(
        baseline_attempted + attempted, baseline_tested + tested,
        unsupported=unsupported + int(
            baseline_status == "case-skipped" and baseline_attempted > 0
        ),
        artifact_gaps=artifact_gaps + int(baseline_status == "artifact-gap"),
    ):
        return gap
    return terminal_gap or baseline_gap


def _is_reference_identity_field(name: object) -> bool:
    return isinstance(name, str) and (
        name in _REFERENCE_IDENTITY_FIELDS
        or name.endswith(("_sha256", "_digest", "_commit"))
    )


def _reference_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("reference_identity")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    for name, item in value.items():
        if not _is_reference_identity_field(name) or item in (None, ""):
            continue
        if name in result and result[name] != item:
            result["__identity_conflict__"] = True
        result[name] = item
    return result


def _reference_observation_identity(value: object) -> dict[str, object]:
    result = _reference_identity(value)
    nested = value.get("reference_identity") if isinstance(value, Mapping) else None
    extra_state = value.get("extra_state") if isinstance(value, Mapping) else None
    embedded = extra_state.get("reference_identity") if isinstance(extra_state, Mapping) else None
    if (isinstance(value, Mapping) and "binary_sha256" in value
            and not isinstance(nested, Mapping) and not isinstance(embedded, Mapping)):
        result.pop("binary_sha256", None)
    evidence = value.get("translation_evidence") if isinstance(value, Mapping) else None
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    identities = (
        embedded,
        nested,
        details.get("target_identity") if isinstance(details, Mapping) else None,
        (details.get("configuration_identity", {}).get("target_identity")
         if isinstance(details, Mapping)
         and isinstance(details.get("configuration_identity"), Mapping) else None),
    )
    for identity in identities:
        if not isinstance(identity, Mapping):
            continue
        for name in ("source_commit", "identity_digest", "binary_sha256"):
            if identity.get(name) not in (None, ""):
                if name in result and result[name] != identity[name]:
                    result["__identity_conflict__"] = True
                result[name] = identity[name]
    return result


def _recorded_qemu_reference_fallback(record: Mapping[str, object]) -> bool:
    baseline = record.get("reference_baseline")
    binding = record.get("root_binding")
    policy = binding.get("reference_fallback_policy") if isinstance(binding, Mapping) else None
    summary = baseline.get("reference_fallback") if isinstance(baseline, Mapping) else None
    if not isinstance(policy, Mapping) or not isinstance(summary, Mapping):
        return False
    allowed_classes = configured_k1_qemu_fallback_classes(policy.get("on"))
    fallback_reason = summary.get("fallback_reason")
    if (
        policy.get("backend") != "qemu-riscv64"
        or policy.get("target_id") != "T-QEMU"
        or allowed_classes is None
        or policy.get("selection_scope") != "whole-campaign"
        or summary.get("schema_version") != "rq1-reference-fallback-v1"
        or summary.get("preferred_backend") != "native-rv64"
        or summary.get("effective_backend") != "qemu-riscv64"
        or not qemu_fallback_policy_matches(
            summary.get("policy"), fallback_reason, allowed_classes,
        )
        or summary.get("selection_scope") != "whole-campaign"
        or summary.get("target_id") != "T-QEMU"
        or not is_sha256_digest(summary.get("k1_attempt_digest"))
        or not isinstance(baseline.get("observation"), (list, tuple))
    ):
        return False
    for observation in baseline["observation"]:
        if not isinstance(observation, Mapping):
            continue
        extra = observation.get("extra_state")
        detail = extra.get("reference_fallback") if isinstance(extra, Mapping) else None
        attempts = detail.get("k1_attempts") if isinstance(detail, Mapping) else None
        if (
            isinstance(detail, Mapping)
            and isinstance(attempts, (list, tuple))
            and detail.get("k1_attempt_digest") == summary.get("k1_attempt_digest")
            and canonical_digest(attempts) == summary.get("k1_attempt_digest")
            and any(
                isinstance(item, Mapping)
                and item.get("backend") == "native-rv64"
                and isinstance(item.get("extra_state"), Mapping)
                and item["extra_state"].get("runner_failure_class") == fallback_reason
                and item["extra_state"].get("runner_failure_class") in allowed_classes
                for item in attempts
            )
        ):
            return True
    return False


def _formal_reference_gap(record: Mapping[str, object], recipe: Mapping[str, object]) -> str | None:
    root = record.get("root")
    early_statuses = {
        root.get("status") if isinstance(root, Mapping) else None,
        record.get("terminal_status"),
        record.get("status"),
    }
    if early_statuses & {"generation-gap", "transport-gap", "profile-gap", "seed-miss"} \
            and not isinstance(record.get("reference_baseline"), Mapping):
        return None
    reference = recipe.get("reference")
    expected = reference.get("expected") if isinstance(reference, Mapping) else None
    binding = record.get("root_binding")
    runtime_identity = isinstance(binding, Mapping) \
        and binding.get("reference_identity_resolution") == "runtime"
    expected_identity = {} if runtime_identity else {
        name: value for name, value in expected.items()
        if _is_reference_identity_field(name)
    } if isinstance(expected, Mapping) else {}
    expected_status = expected.get("status") if isinstance(expected, Mapping) else None
    if "status" in (expected or {}) and expected_status not in _REFERENCE_EXPECTED_STATUSES:
        return "reference-status-invalid"
    generation = recipe.get("generation")
    generation = generation if isinstance(generation, Mapping) else {}
    initial_state = generation.get("initial_state")
    expected_text = (
        tuple(generation.get("sequence", ()))
        + ((initial_state.get("expected"),) if isinstance(initial_state, Mapping) else ())
    )
    expected_trap = (
        isinstance(generation, Mapping)
        and any(_contains_trap_marker(item) for item in expected_text)
    )
    expected_trap = expected_trap or (
        isinstance(binding, Mapping) and binding.get("expected_trap") is True
    )
    if expected_status == "reference-valid":
        expected_status = "trap" if expected_trap else "normal"
    elif expected_trap and expected_status in {"normal", "completed"}:
        expected_status = "trap"
    baseline = record.get("reference_baseline")
    if not isinstance(baseline, Mapping):
        return "reference-identity-mismatch" if expected_identity else "reference-baseline-missing"
    identity = baseline.get("reference_identity")
    rows = baseline.get("observation")
    if not isinstance(identity, Mapping) or not isinstance(rows, list) or not rows:
        return "reference-identity-mismatch" if expected_identity else "reference-baseline-invalid"
    recorded_fallback = _recorded_qemu_reference_fallback(record)
    fallback_summary = baseline.get("reference_fallback")
    if recorded_fallback and isinstance(fallback_summary, Mapping):
        effective_backend = fallback_summary.get("effective_backend")
        if isinstance(effective_backend, str):
            if "backend" in expected_identity:
                expected_identity["backend"] = effective_backend
    if any(not isinstance(row, Mapping) for row in rows):
        return "reference-observation-invalid"
    if any(not _observation_outcome_valid(row) for row in rows):
        return "reference-observation-invalid"
    source_sha256 = baseline.get("source_sha256")
    executable_sha256 = baseline.get("executable_sha256")
    if not is_sha256_digest(source_sha256) or not is_sha256_digest(executable_sha256):
        return "reference-identity-mismatch"
    expected_source = record.get("original_sha256")
    if is_sha256_digest(expected_source) and source_sha256 != expected_source:
        return "reference-identity-mismatch"
    if any(
        is_execution_backend(row.get("backend"))
        and row.get("backend") != "sail-riscv"
        and not any(
            is_sha256_digest(row.get(name)) and row.get(name) == executable_sha256
            for name in ("binary_sha256", "guest_elf_sha256")
        )
        for row in rows
    ):
        return "reference-identity-mismatch"

    def invalid_identity(value: Mapping[str, object]) -> bool:
        return any(
            name in value and (
                value[name] in (None, "")
                or not isinstance(value[name], str)
                or name in {"binary_sha256", "guest_elf_sha256", "identity_digest"}
                and not is_sha256_digest(value[name])
            )
            for name in value
        )

    if invalid_identity(identity):
        return "reference-identity-mismatch"
    observer = recipe.get("observer")
    fields = (
        binding.get("comparison_fields", binding.get("observer_fields", ()))
        if isinstance(binding, Mapping)
        else observer.get("fields", ()) if isinstance(observer, Mapping) else ()
    )

    def field_present(row: Mapping[str, object], field: object) -> bool:
        value = _observer_field_value(row, field)
        return _observer_value_is_valid(str(field), value)

    if isinstance(fields, (list, tuple)) and any(
        not all(field_present(row, field) for row in rows) for field in fields
    ):
        return "reference-observation-invalid"
    identity_values = _reference_identity(identity)
    if (is_sha256_digest(executable_sha256)
            and identity.get("backend") != "sail-riscv" and any(
        any(
            row.get(name) not in (None, "")
            and row.get(name) != executable_sha256
            for name in ("binary_sha256", "guest_elf_sha256")
        )
        for row in rows
    )):
        return "reference-identity-mismatch"
    for row in rows:
        row_identity = _reference_observation_identity(row)
        guest_binary = row.get("guest_elf_sha256")
        if guest_binary in (None, "") and not isinstance(row.get("reference_identity"), Mapping):
            guest_binary = row.get("binary_sha256")
        if (
            identity_values.get("binary_sha256") == guest_binary
            and row_identity.get("binary_sha256") not in (None, "")
            and row_identity["binary_sha256"] != guest_binary
        ):
            identity_values["binary_sha256"] = row_identity["binary_sha256"]
        for name, value in row_identity.items():
            if name in identity_values and identity_values[name] != value:
                identity_values["__identity_conflict__"] = True
            elif name not in identity_values:
                identity_values[name] = value

    def identity_matches(values: Mapping[str, object]) -> bool:
        return not values.get("__identity_conflict__") and all(
            name in values and (
                isinstance(values[name], str)
                and isinstance(expected_value, str)
                and _reference_runtime_backend(recipe, values[name])
                == _reference_runtime_backend(recipe, expected_value)
                if name == "backend" else values[name] == expected_value
            )
            for name, expected_value in expected_identity.items()
        )

    if expected_identity and (
        any(
            value is None
            or isinstance(value, str) and not value.strip()
            or (name.endswith("_sha256") or name == "identity_digest")
            and not is_sha256_digest(value)
            for name, value in expected_identity.items()
        )
        or not identity_matches(identity_values)
    ):
        return "reference-identity-mismatch"
    expected_backend = reference.get("backend") if isinstance(reference, Mapping) else None
    identity_backend = identity_values.get("backend")
    if not isinstance(identity_backend, str):
        return "reference-backend-missing"
    if recorded_fallback and isinstance(fallback_summary, Mapping):
        expected_backend = fallback_summary.get("effective_backend")
    expected_backend = _reference_runtime_backend(recipe, expected_backend)
    if _reference_runtime_backend(recipe, identity_backend) != expected_backend:
        return "reference-backend-mismatch"
    if expected_status in {"normal", "completed"} and any(
        _normal_outcome(row.get("outcome"), row.get("extra_state")) != "normal" for row in rows
        if isinstance(row, Mapping)
    ):
        return "reference-status-mismatch"
    if expected_status == "trap" and any(
        row.get("outcome") not in {"trap", "nonzero-exit"}
        for row in rows if isinstance(row, Mapping)
    ):
        return "reference-status-mismatch"
    if expected_status == "nonzero-exit" and any(
        row.get("outcome") != "nonzero-exit"
        for row in rows if isinstance(row, Mapping)
    ):
        return "reference-status-mismatch"
    if expected_status in {"timeout", "unavailable"} and any(
        row.get("outcome") != expected_status
        for row in rows if isinstance(row, Mapping)
    ):
        return "reference-status-mismatch"
    # ``executable_sha256`` identifies the guest ELF.  The runner identity's
    # ``binary_sha256`` identifies the simulator/runner binary, so compare it
    # only through an explicitly expected runtime identity above.
    if expected_status == "trap" and any(
        row.get("outcome") in {"trap", "nonzero-exit"}
        and not (
            isinstance(row.get("signal"), str)
            and row["signal"].strip() in _SIGNAL_CODE_BY_NAME
            and row.get("signal_code") in (None, _SIGNAL_CODE_BY_NAME[row["signal"].strip()])
            or isinstance(row.get("extra_state"), Mapping)
            and guest_trap_observed(dict(row["extra_state"]))
        )
        for row in rows if isinstance(row, Mapping)
    ):
        return "reference-status-mismatch"
    for row in rows:
        row_identity = _reference_observation_identity(row)
        if invalid_identity(row_identity):
            return "reference-identity-mismatch"
        if expected_identity and not identity_matches({**identity_values, **row_identity}):
            return "reference-identity-mismatch"
        row_backend = row.get("backend") or row_identity.get("backend")
        if not isinstance(row_backend, str) or _reference_runtime_backend(recipe, row_backend) != expected_backend:
            return "reference-backend-mismatch"
        if any(
            identity_values.get(name) not in (None, "")
            and row_identity.get(name) != identity_values.get(name)
            for name in ("profile_id", "input_id", "tool_version")
        ):
            return "reference-identity-mismatch"
    return None


def _reference_runtime_backend(recipe: Mapping[str, object], backend: object) -> str | None:
    raw = backend.strip().lower() if isinstance(backend, str) else None
    canonical = raw if raw in {"qemu-riscv32", "qemu-riscv64"} else (
        canonical_execution_backend(raw) if raw is not None else None
    )
    generation = recipe.get("generation")
    isa = generation.get("isa") if isinstance(generation, Mapping) else None
    if canonical == "qemu" and isinstance(isa, str):
        return "qemu-riscv32" if isa.lower().startswith("rv32") else "qemu-riscv64"
    return canonical


def import_execution_evidence(
    ledger: Mapping[str, object],
    records: Mapping[str, object] | Iterable[Mapping[str, object]] = (),
    *,
    evidence_root: str | Path | None = None,
) -> tuple[Mapping[str, object], ...]:
    """归一化显式传入的 canonical root-bound execution evidence。"""
    if isinstance(records, Mapping):
        selected = records.get("selected_execution_records")
        inherited = records.get("root_binding")
        records = [
            {**item, "root_binding": inherited}
            if isinstance(item, Mapping)
            and "root_binding" not in item
            and isinstance(inherited, Mapping) else item
            for item in selected
        ] if isinstance(selected, (list, tuple)) else (records,)
    imported = []
    for record in records or ():
        if not isinstance(record, Mapping):
            imported.append(record)
            continue
        indexed_path = record.get("record_path")
        if isinstance(indexed_path, str) and indexed_path.strip():
            root = Path(evidence_root).resolve() if evidence_root is not None else None
            raw_path = Path(indexed_path)
            path = (root / raw_path if root is not None and not raw_path.is_absolute() else raw_path).resolve()
            if root is not None:
                try:
                    path.relative_to(root)
                except ValueError:
                    imported.append({**record, "_coverage_import_gap": "indexed-record-outside-root"})
                    continue
            try:
                payload = path.read_bytes()
                actual = json.loads(payload)
            except (OSError, ValueError):
                imported.append({**record, "_coverage_import_gap": "indexed-record-invalid"})
                continue
            expected_hash = record.get("record_sha256")
            if (
                not is_sha256_digest(expected_hash)
                or hashlib.sha256(payload).hexdigest() != expected_hash
            ):
                imported.append({**record, "_coverage_import_gap": "indexed-record-hash-mismatch"})
                continue
            if not isinstance(actual, Mapping):
                imported.append({**record, "_coverage_import_gap": "indexed-record-invalid"})
                continue
            binding = actual.get("root_binding")
            actual_targets = actual.get("targets")
            target_name = binding.get("target") if isinstance(binding, Mapping) else None
            actual_target = (
                actual_targets.get(target_name)
                if isinstance(actual_targets, Mapping) and isinstance(target_name, str)
                else None
            )
            if not isinstance(actual_target, Mapping) and isinstance(actual_targets, Mapping):
                actual_target = next(
                    (value for name, value in actual_targets.items()
                     if isinstance(name, str) and isinstance(value, Mapping)
                     and same_execution_backend(name, target_name)),
                    None,
                )
            indexed_pairs = (
                ("root_key", record.get("root_key"), binding.get("root_key") if isinstance(binding, Mapping) else None),
                ("lineage_key", record.get("lineage_key"), binding.get("lineage_key") if isinstance(binding, Mapping) else None),
                ("route", record.get("route"), binding.get("route") if isinstance(binding, Mapping) else None),
                ("recipe_digest", record.get("recipe_digest"), binding.get("root_recipe_digest") if isinstance(binding, Mapping) else None),
                ("testcase_id", record.get("testcase_id"), actual.get("testcase_id")),
                ("form", record.get("form"), binding.get("form") if isinstance(binding, Mapping) else None),
                ("generation_rule_id", record.get("generation_rule_id"), binding.get("generation_rule_id") if isinstance(binding, Mapping) else None),
                ("code_sha256", record.get("code_sha256"), binding.get("code_sha256") if isinstance(binding, Mapping) else None),
                ("target_identity_digest", record.get("target_identity_digest"), actual_target.get("target_identity_digest")
                 if isinstance(actual_target, Mapping) else None),
                ("target_binary_sha256", record.get("target_binary_sha256"), actual_target.get("target_binary_sha256")
                 if isinstance(actual_target, Mapping) else None),
            )
            if any(
                left not in (None, "") and (
                    _route_alias(left) is None
                    or _route_alias(right) is None
                    or _route_alias(left) != _route_alias(right)
                    if field == "route" else left != right
                )
                for field, left, right in indexed_pairs
            ):
                imported.append({**record, "_coverage_import_gap": "indexed-record-binding-mismatch"})
                continue
            imported.extend(import_execution_evidence(
                ledger, actual, evidence_root=path.parents[2],
            ))
            continue
        if gap := _non_formal_evidence_gap(record):
            imported.append({**record, "_coverage_import_gap": gap})
            continue
        selected = record.get("selected_execution_records")
        if isinstance(selected, (list, tuple)):
            inherited = record.get("root_binding")
            children = [
                {**item, "root_binding": inherited}
                if isinstance(item, Mapping)
                and "root_binding" not in item
                and isinstance(inherited, Mapping) else item
                for item in selected
            ]
            imported.extend(import_execution_evidence(
                ledger, children, evidence_root=evidence_root,
            ))
        elif record.get("schema") == "program-candidate-ledger-v1":
            binding = record.get("root_binding")
            route = binding.get("route") if isinstance(binding, Mapping) else None
            imported.append(_canonical_program_record(
                ledger, record,
                expected_route="single" if _route_alias(route) == "single" else "program",
            ))
        else:
            imported.append(
                {**record, "_coverage_import_gap": "unsupported-execution-evidence"}
                if ledger.get("schema") in _FORMAL_LEDGER_SCHEMAS else record
            )
    return tuple(imported)


def _non_formal_evidence_gap(record: Mapping[str, object]) -> str | None:
    if "formal_evidence_eligible" in record and type(record["formal_evidence_eligible"]) is not bool:
        return "formal-evidence-eligibility-invalid"
    if record.get("formal_evidence_eligible") is False:
        return "non-formal-evidence"
    for name in ("evidence_scope", "evidence_tier", "evidence_class", "status_class"):
        value = record.get(name)
        if isinstance(value, str) and value.lower() in {
            "smoke", "historical", "engineering", "l1", "historical-known-recall",
        }:
            return "non-formal-evidence"
    return None


def _canonical_program_record(
    ledger: Mapping[str, object], record: Mapping[str, object], *,
    expected_route: str = "program",
) -> dict[str, object]:
    binding = record.get("root_binding")
    if not isinstance(binding, Mapping):
        return {**record, "_coverage_import_gap": "root-binding-missing"}
    if any(
        not isinstance(binding.get(name), str) or not binding[name].strip()
        for name in _ROOT_BINDING_FIELDS
    ):
        return {**record, "_coverage_import_gap": "root-binding-invalid"}
    binding_route = binding.get("route")
    route = _route_alias(binding_route) if binding_route is not None else None
    if route != expected_route:
        return {**record, "_coverage_import_gap": "root-binding-route-mismatch"}
    result = dict(record)
    if not isinstance(record.get("root"), Mapping) or not isinstance(
        record.get("counts"), Mapping
    ):
        result["_coverage_import_gap"] = "program-structure-missing"
        return result
    try:
        recipe = resolve_root_recipe(
            ledger, binding.get("root_key"), binding.get("lineage_key"),
            route=route,
        )
    except (TypeError, ValueError):
        return result
    target_identity_digest, target_binary_sha256 = _recipe_target_identity(recipe)
    if is_sha256_digest(target_identity_digest) and binding.get(
        "target_identity_digest"
    ) != target_identity_digest:
        result["_coverage_import_gap"] = "root-binding-target-identity-mismatch"
    if (
        "_coverage_import_gap" not in result
        and is_sha256_digest(target_binary_sha256)
        and binding.get("target_binary_sha256") != target_binary_sha256
    ):
        result["_coverage_import_gap"] = "root-binding-target-binary-mismatch"
    if result.get("route") in (None, "") and route in {"single", "program"}:
        result["route"] = _recipe_route(recipe)
    generation = recipe.get("generation")
    if (isinstance(generation, Mapping)
            and isinstance(generation.get("generation_profile"), Mapping)
            and result.get("generation_profile_digest") in (None, "")
            and binding.get("generation_profile_digest") in (None, "")):
        result["generation_profile_digest"] = _generation_profile_digest(
            generation["generation_profile"]
        )
    return result


def _execution_route_attempt(record: Mapping[str, object]) -> tuple[tuple[str, str, str], dict[str, object]] | None:
    """从已导入的执行证据提取路线级 target attempt。"""
    if record.get("_coverage_import_gap") or record.get("_root_binding_conflict"):
        return None
    binding = record.get("root_binding")
    if not isinstance(binding, Mapping):
        return None
    root_key = binding.get("root_key")
    lineage_key = binding.get("lineage_key")
    route = _route_alias(binding.get("route"))
    if any(not isinstance(value, str) or not value.strip()
           for value in (root_key, lineage_key, route)):
        return None
    counts = record.get("counts")
    claimed = (
        record.get("target_attempted") is True
        or record.get("target_tested") is True
        or record.get("target_gap") is True
        or any(
            isinstance(counts, Mapping) and _positive_count(counts.get(name))
            for name in (
                "target_attempted", "target_tested", "target_gap",
                "target_raw_recorded", "target_comparison_pending",
                "target_unsupported", "case_skipped", "target_mismatch_candidate", "clean",
            )
        )
    )
    if not claimed:
        return None
    return (root_key, lineage_key, route), {
        "status": "attempted",
        "source": "execution-evidence",
        "target_attempted": True,
        "target_tested": bool(
            record.get("target_tested") is True
            or isinstance(counts, Mapping)
            and _positive_count(counts.get("target_tested"))
        ),
    }


def _overlay_execution_route_attempts(
    route_matrix: dict[str, object],
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """把实际执行证据叠加到静态 route matrix，避免只看 recipe 元数据。"""
    attempts: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        result = _execution_route_attempt(record)
        if result is not None:
            key, attempt = result
            attempts[key] = attempt
    rows = route_matrix.get("rows")
    if not isinstance(rows, list):
        return route_matrix
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("root_key"), row.get("lineage_key"), row.get("route"))
        attempt = attempts.get(key)
        if attempt is None:
            continue
        existing = row.get("route_attempt")
        row["route_attempt"] = {
            **(dict(existing) if isinstance(existing, Mapping) else {}),
            **attempt,
        }
        row["attempt_status"] = "attempted"
    counts = route_matrix.get("counts")
    if isinstance(counts, dict):
        eligible = [row for row in rows if row.get("status") == "recipe-ready"]
        counts["route_attempt_eligible_count"] = len(eligible)
        counts["route_attempt_complete"] = all(
            row.get("attempt_status") == "attempted" for row in eligible
        )
        counts["route_unattempted"] = sum(
            row.get("attempt_status") == "unattempted" for row in eligible
        )
        gaps = [gap for gap in route_matrix.get("route_audit_gaps", ())
                if gap != "route-attempt"]
        if not counts["route_attempt_complete"]:
            gaps.append("route-attempt")
        route_matrix["route_audit_gaps"] = sorted(set(gaps))
        route_matrix["route_audit_status"] = (
            "complete" if not route_matrix["route_audit_gaps"] else "incomplete"
        )
    return route_matrix


def _roots(
    ledger: Mapping[str, object], methods: tuple[str, ...],
    *, route_matrix: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    route_records = ledger.get("route_records", ())
    if not isinstance(route_records, (list, tuple)):
        raise ValueError("target ledger route_records must be a list")
    eligible = tuple(
        route for route in route_records
        if isinstance(route, Mapping) and route.get("target_scope_included") is True
        and all(
            isinstance(route.get(name), str) and route[name].strip()
            for name in ("root_key", "lineage_key")
        )
    )
    split_bases = {
        (route.get("root_key"), route.get("lineage_key"))
        for route in eligible
        if _recipe_route_name(route) is not None
    }
    theoretical: dict[tuple[str, str, str | None], dict[str, object]] = {}
    for route in eligible:
        base_key = (route.get("root_key"), route.get("lineage_key"))
        route_name = _recipe_route_name(route)
        if route_name is None and base_key in split_bases:
            continue
        key = (*base_key, route_name)
        theoretical.setdefault(key, {
            "theoretical": {
                "tool": route.get("tool"),
                "route_status": route.get("route_status"),
                "mapping_status": route.get("mapping_status"),
                "source_locator": route.get("source_locator"),
            },
        })
    if route_matrix is None:
        from .root_route_matrix import build_root_route_matrix
        route_matrix = build_root_route_matrix(ledger)

    expanded = []
    for route_row in route_matrix["rows"]:
        key = (route_row["root_key"], route_row["lineage_key"], route_row["route"])
        source = theoretical.get(key) or (
            theoretical.get((*key[:2], None)) if key[2] == "single" else None
        )
        row = {
            "root_key": key[0], "lineage_key": key[1],
            "theoretical": dict(source["theoretical"]) if source else {
                "tool": None, "route_status": "route-unattempted",
                "mapping_status": "unassigned", "source_locator": None,
            },
        }
        recipe = route_row.get("recipe")
        raw_route = _recipe_route_name(recipe) if isinstance(recipe, Mapping) else None
        if raw_route is None and route_row["source_record_count"] == 1:
            raw_route = key[2]
        assignments = [raw_route] if isinstance(raw_route, str) and raw_route else []
        recipe_gaps = list(route_row["recipe_gaps"])
        if route_row["source_record_count"] > 1:
            recipe_gaps.append("route-record-ambiguous")
        row.update(
            route=key[2],
            route_count=route_row["source_record_count"],
            route_assignments=assignments,
            route_assignment=assignments[0] if assignments else (
                "ambiguous" if route_row["source_record_count"] > 1 else "unassigned"
            ),
            route_assignment_status=(
                "assigned" if assignments else
                "ambiguous" if route_row["source_record_count"] > 1 else "unassigned"
            ),
            recipe_status=route_row.get(
                "recipe_status",
                "recipe-ready" if route_row["status"] == "recipe-ready" else "recipe-gap",
            ),
            recipe_gaps=sorted(set(recipe_gaps)),
            attempt_status=route_row["attempt_status"],
            execution_status=route_row["execution_status"],
            evidence_status=route_row.get("evidence_status", "unobserved"),
            execution_gaps=list(route_row["execution_gaps"]),
            gap_reason=route_row.get("gap_reason"),
            route_gap=deepcopy(route_row.get("route_gap")),
            route_attempt=deepcopy(route_row.get("route_attempt")),
            source_locators=list(route_row["source_locators"]),
            mapping={field: [] for field in MAPPING_FIELDS},
            methods={method: {field: 0 for field in FIELDS} for method in methods},
            evidence=[],
        )
        if route_row.get("source_provenance"):
            row["source_provenance"] = deepcopy(route_row["source_provenance"])
        expanded.append(row)
    return sorted(
        expanded,
        key=lambda row: (
            row["root_key"], row["lineage_key"], 0 if row["route"] == "single" else 1,
        ),
    )


def summarize_target_coverage(
    ledger: Mapping[str, object],
    records: Iterable[Mapping[str, object]] = (),
    *,
    evidence_root: str | Path | None = None,
) -> dict[str, object]:
    """只接受显式 root/lineage 绑定，未绑定记录进入独立计数。"""
    from .root_route_matrix import build_root_route_matrix

    records = tuple(
        _coverage_record(record) if isinstance(record, Mapping) else record
        for record in import_execution_evidence(
            ledger, records, evidence_root=evidence_root,
        )
    )
    observed = {
        value for record in records if isinstance(record, Mapping)
        for value in (record.get("method") or record.get("method_group"),)
        if isinstance(value, str)
    }
    methods = (*METHODS, *EXPERIMENT_METHODS) if observed & set(EXPERIMENT_METHODS) else METHODS
    observed_routes = tuple(sorted({
        route for record in records
        for route in (_route_alias(record.get("route")),)
        if route is not None
    }))
    route_scope = observed_routes if len(observed_routes) == 1 else None
    route_matrix = _overlay_execution_route_attempts(
        build_root_route_matrix(
            ledger, routes=route_scope or ("single", "program"),
        ), records,
    )
    route_matrix["route_scope"] = list(route_scope or ("single", "program"))
    rows = _roots(ledger, methods, route_matrix=route_matrix)
    split_keys = {
        (row["root_key"], row["lineage_key"])
        for row in rows if row.get("route") is not None
    }
    by_key = {
        (row["root_key"], row["lineage_key"], row.get("route")): row
        for row in rows
    }
    unbound = 0
    seen_bindings: dict[tuple[object, ...], tuple[object, ...]] = {}
    seen_records: dict[tuple[object, ...], str] = {}
    mapping_gaps = sum(
        isinstance(route, Mapping)
        and route.get("target_scope_included") is True
        and any(
            not isinstance(route.get(name), str) or not route[name].strip()
            for name in ("root_key", "lineage_key")
        )
        for route in ledger.get("route_records", ())
    )
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("coverage record must be an object")
        if record.get("_root_binding_conflict"):
            mapping_gaps += 1
            continue
        if record.get("_coverage_import_gap"):
            mapping_gaps += 1
            continue
        binding = record.get("root_binding")
        if not isinstance(binding, Mapping):
            unbound += 1
            continue
        if any(
            not isinstance(binding.get(name), str) or not binding[name].strip()
            for name in _ROOT_BINDING_FIELDS
        ):
            mapping_gaps += 1
            continue
        base_key = (record.get("root_key"), record.get("lineage_key"))
        if any(not isinstance(item, str) or not item for item in base_key):
            unbound += 1
            continue
        key = (*base_key, record.get("route")) if base_key in split_keys else (*base_key, None)
        row = by_key.get(key)
        if row is None:
            mapping_gaps += 1
            continue
        if row["recipe_status"] != "recipe-ready":
            mapping_gaps += 1
            continue
        recipe = resolve_root_recipe(
            ledger, *base_key,
            route=record.get("route") if base_key in split_keys else None,
        )
        target_override = binding.get("target_backend_override") is True
        if (isinstance(binding, Mapping)
                and (
                    not isinstance(binding.get("target"), str)
                    or (
                        not target_override
                        and not same_execution_backend(
                            binding["target"], recipe["target"]["backend"],
                        )
                    )
                )):
            mapping_gaps += 1
            continue
        expected_target_identity, expected_target_binary = _recipe_target_identity(recipe)
        if not target_override and (
            is_sha256_digest(expected_target_identity)
            and binding.get("target_identity_digest") != expected_target_identity
        ):
            mapping_gaps += 1
            continue
        if not target_override and (
            is_sha256_digest(expected_target_binary)
            and binding.get("target_binary_sha256") != expected_target_binary
        ):
            mapping_gaps += 1
            continue
        if record.get("schema") == "program-candidate-ledger-v1":
            reference_record = record.get("reference_baseline")
            if not isinstance(reference_record, Mapping):
                reference_record = record.get("reference")
            reference_backend = (
                reference_record.get("backend")
                if isinstance(reference_record, Mapping) else None
            )
            if not isinstance(reference_backend, str):
                identity = (
                    reference_record.get("reference_identity")
                    if isinstance(reference_record, Mapping) else None
                )
                reference_backend = identity.get("backend") if isinstance(identity, Mapping) else None
            expected_reference = recipe.get("reference", {}).get("backend")
            fallback_reason = (
                reference_record.get("fallback_reason")
                if isinstance(reference_record, Mapping) else None
            )
            sail_fallback = (
                expected_reference == "native-rv64"
                and reference_backend == "sail-riscv"
                and (
                    isinstance(fallback_reason, str)
                    and (
                        fallback_reason.startswith("native-isa-gap:")
                        or fallback_reason == "no-current-rv64-native-reference"
                    )
                )
            )
            qemu_fallback = (
                expected_reference == "native-rv64"
                and reference_backend == "qemu-riscv64"
                and _recorded_qemu_reference_fallback(record)
            )
            reference_mismatch = (
                isinstance(expected_reference, str)
                and (
                    not isinstance(reference_backend, str)
                    or _reference_runtime_backend(recipe, reference_backend)
                    != _reference_runtime_backend(recipe, expected_reference)
                )
                and not (sail_fallback or qemu_fallback)
            )
            if reference_mismatch:
                record = {**record, "reference_valid": False}
                if record.get("target_tested") is True or record.get("target_attempted") is True:
                    record.update(
                        target_tested=False, target_gap=True,
                        mismatch_candidate=False, clean=False,
                    )
        if record.get("route") != _recipe_route(recipe):
            mapping_gaps += 1
            continue
        if record.get("root_recipe_digest") != _recipe_digest(recipe):
            row["mapping_conflict"] = True
            mapping_gaps += 1
            continue
        generation = recipe.get("generation")
        generation_profile = (
            generation.get("generation_profile")
            if isinstance(generation, Mapping)
            else None
        )
        if isinstance(generation_profile, Mapping):
            profile = dict(generation_profile)
            profile.setdefault("iterations", 1)
            accepted_profile_digests = {
                _generation_profile_digest(profile),
                canonical_digest(_normalize_generation_profile(profile)),
            }
            if record.get("generation_profile_digest") not in accepted_profile_digests:
                mapping_gaps += 1
                continue
        generation_identity_digest = (
            generation.get("generation_identity_digest")
            if isinstance(generation, Mapping) else None
        )
        if (generation_identity_digest not in (None, "runtime")
                and record.get("generation_identity_digest") != generation_identity_digest):
            mapping_gaps += 1
            continue
        if ledger.get("schema") in _FORMAL_LEDGER_SCHEMAS:
            contract_gap = (
                _formal_reference_gap(record, recipe)
                if record.get("schema") == "program-candidate-ledger-v1"
                else None
            ) or _formal_target_gap(record, recipe)
            if contract_gap is not None:
                target_claim = _target_claimed(record)
                record = {
                    **record,
                    "reference_valid": False if contract_gap.startswith("reference-") else record.get("reference_valid"),
                    "target_tested": False,
                    "target_gap": target_claim and contract_gap != "target-unsupported",
                    "target_unsupported": contract_gap == "target-unsupported",
                    "mismatch_candidate": False,
                    "clean": False,
                    "coverage_gap_reason": contract_gap,
                }
                if contract_gap.startswith("reference-"):
                    root = record.get("root")
                    counts = record.get("counts")
                    if isinstance(root, Mapping):
                        record["root"] = {**root, "status": "reference-rejected"}
                    if isinstance(counts, Mapping):
                        record["counts"] = {
                            **counts, "reference_valid": 0,
                            "reference_rejected": 1,
                        }
                    record = _coverage_record(record)
        method = record.get("method") or record.get("method_group")
        if method in (None, ""):
            mapping_gaps += 1
            continue
        if method not in methods:
            mapping_gaps += 1
            continue
        if record.get("rule") not in RULES:
            mapping_gaps += 1
            continue
        cost = record.get("cost")
        if (cost is not None
                and (not isinstance(cost, (int, float)) or isinstance(cost, bool)
                     or cost < 0 or not math.isfinite(cost))):
            raise ValueError("coverage cost must be a non-negative number")
        mapping = {
            field: tuple(
                item for item in (
                    record.get(field)
                    if isinstance(record.get(field), (list, tuple))
                    else (record.get(field),)
                )
                if isinstance(item, str) and item
            )
            for field in MAPPING_FIELDS
        }
        evidence = record.get("evidence") or record.get("evidence_path")
        has_evidence = (
            isinstance(evidence, str) and bool(evidence)
            or isinstance(evidence, (list, tuple))
            and any(isinstance(item, str) and item for item in evidence)
        )
        if not all(mapping.values()) or not has_evidence:
            mapping_gaps += 1
            continue
        if isinstance(evidence, str) and evidence not in row["evidence"]:
            row["evidence"].append(evidence)
        elif isinstance(evidence, (list, tuple)):
            row["evidence"].extend(
                item for item in evidence
                if isinstance(item, str) and item and item not in row["evidence"]
            )
        if record.get("target_stage_status") == "target-gap" \
                or record.get("target_gap") is True or record.get("artifact_gap") is True:
            gaps = (
                record.get("target_reason_code"),
                record.get("target_adapter_execution_gap"),
                record.get("coverage_gap_reason"),
            )
            specific = next(
                (gap for gap in gaps
                 if isinstance(gap, str) and gap and gap != "target-status-gap"),
                None,
            )
            if specific is not None:
                row["execution_gaps"] = [
                    item for item in row["execution_gaps"]
                    if item != "target-status-gap"
                ]
                if specific not in row["execution_gaps"]:
                    row["execution_gaps"].append(specific)
            elif "target-status-gap" not in row["execution_gaps"]:
                row["execution_gaps"].append("target-status-gap")
        counts = row["methods"][method]
        binding_key = (record.get("root_key"), record.get("lineage_key"), record.get("route"), method)
        binding_state = tuple(
            bool(record.get(name))
            for name in (
                "generated", "generation_gap", "seed_valid", "seed_miss",
                "profile_gap", "transport_gap", "reference_rejected",
                "reference_valid", "target_attempted", "target_tested", "target_gap",
                "target_raw_recorded", "target_comparison_pending", "artifact_gap",
                "target_unsupported", "mismatch_candidate", "clean", "reachable",
            )
        )
        previous_state = seen_bindings.get(binding_key)
        if previous_state is not None and previous_state != binding_state:
            row["mapping_conflict"] = True
            mapping_gaps += 1
            continue
        record_digest = canonical_digest(dict(record))
        previous_record = seen_records.get(binding_key)
        if previous_record is not None:
            if previous_record == record_digest:
                continue
            row["mapping_conflict"] = True
            mapping_gaps += 1
            continue
        seen_bindings[binding_key] = binding_state
        seen_records[binding_key] = record_digest
        def claimed(value: object) -> bool:
            return value is True or _positive_count(value)
        for field in FIELDS[:-1]:
            value = record.get(field)
            if field == "mismatch_candidate":
                value = (
                    claimed(record.get("mismatch_candidate"))
                    or claimed(record.get("target_mismatch_candidate"))
                )
            elif field == "target_attempted":
                value = (
                    claimed(record.get("target_attempted"))
                    or claimed(record.get("target_tested"))
                    or claimed(record.get("target_raw_recorded"))
                    or claimed(record.get("target_comparison_pending"))
                    or claimed(record.get("target_gap"))
                )
            counts[field] = max(counts[field], int(claimed(value)))
        if cost is None:
            counts["cost"] = None
        elif counts["cost"] is not None:
            counts["cost"] += cost
        for field, values in mapping.items():
            row["mapping"][field].extend(
                item for item in values
                if item not in row["mapping"][field]
            )
    for row in rows:
        if row.get("mapping_conflict"):
            for values in row["methods"].values():
                for field in FIELDS:
                    if field != "cost":
                        values[field] = 0
                values["cost"] = None
        values = tuple(row["methods"].values())
        row["status"] = "mapping-gap" if row.get("mapping_conflict") else (
            "artifact-gap" if any(item.get("artifact_gap", 0) for item in values)
            else "target-gap" if any(item.get("target_gap", 0) for item in values)
            else "target-mismatch-candidate" if any(item.get("mismatch_candidate", 0) for item in values)
            else "clean" if any(item.get("clean", 0) for item in values)
            else "target-tested" if any(item.get("target_tested", 0) for item in values)
            else "raw-recorded" if any(item.get("target_raw_recorded", 0) for item in values)
            else "target-unsupported" if any(item.get("target_unsupported", 0) for item in values)
            else "reference-rejected" if any(item.get("reference_rejected", 0) for item in values)
            else "reference-valid" if any(item.get("reference_valid", 0) for item in values)
            else "transport-gap" if any(item.get("transport_gap", 0) for item in values)
            else "profile-gap" if any(item.get("profile_gap", 0) for item in values)
            else "seed-miss" if any(item.get("seed_miss", 0) for item in values)
            else "reachable" if any(item.get("reachable", 0) for item in values)
            else "generation-gap" if any(item.get("generation_gap", 0) for item in values)
            else "unexecuted"
        )
        row["evidence_status"] = row["status"]
    method_summaries = {}
    for method in methods:
        costs = [row["methods"][method]["cost"] for row in rows]
        cost = None if any(value is None for value in costs) else sum(costs)
        summary = {
            field: cost if field == "cost" else
            sum(row["methods"][method][field] for row in rows)
            for field in FIELDS
        }
        summary["unit_cost"] = (
            cost / summary["target_tested"]
            if cost is not None and summary["target_tested"] else None
        )
        method_summaries[method] = summary
    status_counts = {
        "unexecuted": sum(row["status"] == "unexecuted" for row in rows),
        "mapping-gap": sum(row["status"] == "mapping-gap" for row in rows),
        **{
            label: sum(
                any(item.get(field, 0) for item in row["methods"].values())
                for row in rows
            )
            for label, field in (
                ("reachable", "reachable"), ("generation-gap", "generation_gap"),
                ("profile-gap", "profile_gap"), ("transport-gap", "transport_gap"),
                ("seed-miss", "seed_miss"),
                ("seed-valid", "seed_valid"),
                ("reference-rejected", "reference_rejected"),
                ("reference-valid", "reference_valid"),
                ("artifact-gap", "artifact_gap"),
                ("target-unsupported", "target_unsupported"),
                ("target-gap", "target_gap"), ("target-tested", "target_tested"),
                ("raw-recorded", "target_raw_recorded"),
                ("target-mismatch-candidate", "mismatch_candidate"),
                ("clean", "clean"),
            )
        },
    }
    mapping_complete = sum(
        all(row["mapping"][field] for field in MAPPING_FIELDS) for row in rows
    )
    root_count = len({row["root_key"] for row in rows})
    route_count = len(rows)
    recipe_ready = sum(row["recipe_status"] == "recipe-ready" for row in rows)
    execution_ready = sum(row["execution_status"] == "executable-ready" for row in rows)
    generated = sum(
        any(values.get("generated", 0) for values in row["methods"].values())
        for row in rows
    )
    theoretical_roots = {
        (row["root_key"], row["lineage_key"])
        for row in rows
        if row["theoretical"].get("mapping_status") == "mapped"
    }
    plan_completion = _plan_completion(
        ledger, route_matrix, rows,
        {"mapping_gap_records": mapping_gaps, "unbound_records": unbound},
    )
    return {
        "schema": "target-coverage-matrix-v1",
        "source_schema": ledger.get("schema"),
        "route_scope": list(route_matrix.get("route_scope", route_matrix.get("routes", ()))),
        "route_audit_status": route_matrix["route_audit_status"],
        "route_audit_gaps": list(route_matrix["route_audit_gaps"]),
        "denominators": {
            "canonical_roots": root_count,
            "canonical_root_routes": route_count,
            "recipe_ready_routes": recipe_ready,
            "execution_ready_routes": execution_ready,
            "semantic_mapping_complete_routes": mapping_complete,
        },
        "results": {
            "methods": method_summaries,
            "statuses": status_counts,
        },
        "counts": {
            "canonical_root_lineage": route_count,
            "canonical_roots": root_count,
            "unbound_records": unbound,
            "mapping_gap_records": mapping_gaps,
            "theoretical_mapped": len(theoretical_roots),
            "route_assigned": sum(
                row["route_assignment_status"] == "assigned" for row in rows
            ),
            "route_ambiguous": sum(
                row["route_assignment_status"] == "ambiguous" for row in rows
            ),
            "route_unassigned": sum(
                row["route_assignment_status"] == "unassigned" for row in rows
            ),
            "generated": generated,
            "recipe_ready": recipe_ready,
            "recipe_gap": route_count - recipe_ready,
            "execution_ready": execution_ready,
            "execution_gap": route_count - execution_ready,
            "semantic_mapping_complete": mapping_complete,
            "semantic_mapping_gap": route_count - mapping_complete,
            **status_counts,
        },
        "methods": method_summaries,
        "route_matrix": route_matrix,
        "plan_completion": plan_completion,
        "rows": rows,
    }


def summarize_target_coverage_by_target(
    ledger: Mapping[str, object], campaign: Mapping[str, object],
    *, evidence_root: Path | None = None,
) -> dict[str, object]:
    """把一条共享 MCMC 账本按 Target 观察侧证据分别汇总。"""
    target_bindings = campaign.get("target_bindings")
    target_records = campaign.get("target_records_by_target")
    target_baselines = campaign.get("target_baselines_by_target")
    if not all(isinstance(value, Mapping) for value in (
        target_bindings, target_records, target_baselines,
    )):
        # Single-Target campaigns use the historical top-level ``target`` /
        # ``target_baseline`` fields.  Raw-only runs compact those records in
        # exactly the same way as mapped campaigns, so hydrate them before the
        # legacy summarizer consumes their observations.
        if evidence_root is not None:
            campaign_view = dict(campaign)
            rows = campaign_view.get("target_records")
            if not isinstance(rows, (list, tuple)) or not rows:
                rows = campaign_view.get("target")
            if isinstance(rows, (list, tuple)):
                materialized = list(iter_materialized_records(evidence_root, rows))
                campaign_view["target_records"] = materialized
                campaign_view["target"] = materialized
            baseline = campaign_view.get("target_baseline")
            if isinstance(baseline, Mapping):
                campaign_view["target_baseline"] = materialize_record(
                    evidence_root, baseline,
                )
            return summarize_target_coverage(ledger, (campaign_view,))
        return summarize_target_coverage(ledger, (campaign,))
    summaries: dict[str, dict[str, object]] = {}
    for target_id, binding in target_bindings.items():
        if not isinstance(target_id, str) or not isinstance(binding, Mapping):
            continue
        rows = target_records.get(target_id, ())
        rows = (
            list(iter_materialized_records(evidence_root, rows))
            if evidence_root is not None and isinstance(rows, (list, tuple))
            else list(rows) if isinstance(rows, (list, tuple)) else []
        )
        rows = [item for item in rows if isinstance(item, Mapping)]
        baseline = target_baselines.get(target_id)
        if evidence_root is not None and isinstance(baseline, Mapping):
            baseline = materialize_record(evidence_root, baseline)
        counts = dict(campaign.get("counts", {})) if isinstance(campaign.get("counts"), Mapping) else {}
        target_counts = {
            name: 0 for name in (
                "target_attempted", "target_tested", "target_gap", "artifact_gap",
                "target_raw_recorded", "target_comparison_pending", "target_unsupported",
                "case_skipped", "target_mismatch_candidate",
                "clean", "target_transport_gap", "target_wall_clock_pending",
            )
        }
        probe_mode = binding.get("target_execution_mode") == "probe"
        for item in rows:
            status = item.get("status")
            status = status if isinstance(status, str) else None
            attempted = item.get("target_attempted") is not False
            probe = probe_mode and status in {"target-probe-tested", "target-probe-gap"}
            counted_attempt = attempted and not probe
            coverage_only = item.get("coverage_only") is True
            expected_comparison = (
                "equivalent" if status == "clean" else
                "non-equivalent" if status == "target-mismatch-candidate" else None
            )
            comparison_qualified = (
                expected_comparison is not None
                and _comparison_qualified(item, expected_comparison)
            )
            tested = (
                not probe_mode and not coverage_only and comparison_qualified
                and status in {"clean", "target-mismatch-candidate"}
            )
            raw_pending = _raw_target_pending(item)
            coverage_only_completed = coverage_only and attempted and status in {
                "clean", "target-mismatch-candidate", "target-tested",
            }
            target_counts["target_attempted"] += int(counted_attempt)
            target_counts["target_raw_recorded"] += int(counted_attempt and raw_pending)
            target_counts["target_comparison_pending"] += int(counted_attempt and raw_pending)
            target_counts["target_tested"] += int(tested)
            target_counts["target_gap"] += int(
                counted_attempt and not tested and not coverage_only_completed
                and status != "case-skipped" and not raw_pending
            )
            target_counts["target_unsupported"] += int(counted_attempt and status == "case-skipped")
            target_counts["case_skipped"] += int(counted_attempt and status == "case-skipped")
            target_counts["artifact_gap"] += int(status == "artifact-gap")
            target_counts["target_mismatch_candidate"] += int(
                not probe_mode and not coverage_only and comparison_qualified
                and status == "target-mismatch-candidate"
            )
            target_counts["clean"] += int(
                not probe_mode and not coverage_only and comparison_qualified
                and status == "clean"
            )
            target_counts["target_transport_gap"] += int(attempted and not coverage_only and (
                item.get("failure_class") == "transport-gap"
                or item.get("target_result_status") == "transport-gap"
            ))
            target_counts["target_wall_clock_pending"] += int(attempted and (
                item.get("deadline_censored") is True
                or item.get("failure_class") == "campaign-wall-clock-exhausted"
            ))
        if isinstance(baseline, Mapping):
            status = baseline.get("status")
            status = status if isinstance(status, str) else None
            comparison = baseline.get("comparison")
            comparison_status = (
                comparison.get("status") if isinstance(comparison, Mapping) else None
            )
            comparison_status = (
                comparison_status if isinstance(comparison_status, str) else None
            )
            baseline_qualified = (
                comparison_status in {"equivalent", "non-equivalent"}
                and _comparison_qualified(baseline, comparison_status)
            )
            if baseline.get("coverage_only") is True and status == "target-tested":
                target_counts["target_attempted"] += 1
            elif _raw_target_pending(baseline):
                target_counts["target_attempted"] += 1
                target_counts["target_raw_recorded"] += 1
                target_counts["target_comparison_pending"] += 1
            elif status == "case-skipped":
                target_counts.update(target_attempted=1, target_unsupported=1, case_skipped=1)
            elif status == "artifact-gap":
                target_counts.update(target_attempted=1, target_gap=1, artifact_gap=1)
            else:
                tested = int(
                    status == "target-tested"
                    and baseline_qualified
                )
                gap = int(
                    status not in {"target-tested", "target-probe-tested", "target-probe-gap"}
                    or (status == "target-tested" and not tested)
                )
                target_counts["target_attempted"] += tested + gap
                target_counts["target_tested"] += tested
                target_counts["target_gap"] += gap
                target_counts["target_mismatch_candidate"] += int(
                    tested and isinstance(comparison, Mapping)
                    and comparison.get("status") == "non-equivalent"
                )
                target_counts["target_transport_gap"] += int(
                    status == "transport-gap" or baseline.get("failure_class") == "transport-gap"
                )
                target_counts["target_wall_clock_pending"] += int(
                    baseline.get("deadline_censored") is True
                    or baseline.get("failure_class") == "campaign-wall-clock-exhausted"
                )
        counts.update(target_counts)
        target_statuses = {
            item.get("status") for item in rows
            if isinstance(item.get("status"), str)
        }
        baseline_status = baseline.get("status") if isinstance(baseline, Mapping) else None
        baseline_status = baseline_status if isinstance(baseline_status, str) else None
        coverage_only_completed = any(
            item.get("coverage_only") is True
            and item.get("target_attempted") is not False
            and item.get("status") in {
                "clean", "target-mismatch-candidate", "target-tested",
            }
            for item in rows
        )
        shared_statuses = {
            "generation-gap", "transport-gap", "profile-gap", "seed-miss",
            "reference-rejected", "reference-gap", "reference-valid", "proposal-gap",
        }
        target_status = (
            "artifact-gap" if target_counts["artifact_gap"] else
            "target-gap" if target_counts["target_gap"] else
            "target-mismatch-candidate" if target_counts["target_mismatch_candidate"] else
            "case-skipped" if (target_counts["case_skipped"] or "case-skipped" in target_statuses)
            and not target_counts["clean"] else
            "clean" if target_counts["clean"] else
            "target-probe-gap" if baseline_status == "target-probe-gap" or "target-probe-gap" in target_statuses else
            "target-probe-tested" if baseline_status == "target-probe-tested" or "target-probe-tested" in target_statuses else
            "target-tested" if coverage_only_completed else
            "raw-recorded" if target_counts["target_raw_recorded"] else
            baseline_status if baseline_status == "target-tested" else
            campaign.get("status") if campaign.get("status") in shared_statuses else
            "target-gap" if not rows and baseline is None else
            "profiled"
        )
        target_campaign = {
            **dict(campaign),
            "root_binding": dict(binding),
            "target": rows,
            "target_records": rows,
            "target_baseline": baseline,
            "counts": counts,
            "status": target_status,
            "target_bindings": {},
            "target_records_by_target": {},
            "target_baselines_by_target": {},
        }
        summaries[target_id] = summarize_target_coverage(
            ledger, (target_campaign,),
        )
    if not summaries:
        return summarize_target_coverage(ledger, (campaign,))
    result = dict(next(iter(summaries.values())))
    result["targets"] = summaries
    result["rows"] = [
        row
        for target_summary in summaries.values()
        for row in target_summary.get("rows", ())
        if isinstance(row, Mapping)
    ]
    def sum_maps(items: Iterable[Mapping[str, object]]) -> dict[str, object]:
        items = tuple(items)
        return {
            name: sum(item.get(name, 0) or 0 for item in items)
            for name in {field for item in items for field in item}
        }

    result["counts"] = sum_maps(
        item["counts"] for item in summaries.values() if isinstance(item.get("counts"), Mapping)
    )
    shared_count_fields = {
        "canonical_root_lineage", "canonical_roots", "unbound_records",
        "mapping_gap_records", "theoretical_mapped", "route_assigned",
        "route_ambiguous", "route_unassigned", "generated", "recipe_ready",
        "recipe_gap", "execution_ready", "execution_gap",
        "semantic_mapping_complete", "semantic_mapping_gap",
        "unexecuted", "mapping-gap", "reachable", "generation-gap",
        "profile-gap", "transport-gap", "seed-miss", "seed-valid",
        "reference-rejected", "reference-valid",
    }
    # Every Target summary refers to the same plan and route matrix. Keep
    # shared chain and structural totals once; target outcomes remain additive.
    first_summary = next(iter(summaries.values()))
    first_denominators = first_summary.get("denominators")
    if isinstance(first_denominators, Mapping):
        result["denominators"] = dict(first_denominators)
    first_counts = first_summary.get("counts")
    if isinstance(first_counts, Mapping):
        for name in shared_count_fields:
            if name in first_counts:
                result["counts"][name] = first_counts[name]
    method_names = {
        method for summary in summaries.values()
        for method in summary.get("methods", {})
        if isinstance(summary.get("methods"), Mapping)
    }
    methods = {}
    for method in method_names:
        values = [summary["methods"][method] for summary in summaries.values()
                  if isinstance(summary.get("methods"), Mapping)
                  and isinstance(summary["methods"].get(method), Mapping)]
        totals = sum_maps(
            {name: value for name, value in item.items() if name != "unit_cost"}
            for item in values
        )
        if values:
            for name in (
                "reachable", "generation_gap", "generated", "seed_valid",
                "rewrite_valid", "reference_valid", "profile_gap",
                "transport_gap", "seed_miss", "reference_rejected", "cost",
            ):
                if name in values[0]:
                    totals[name] = values[0][name]
        if any(item.get("cost") is None for item in values):
            totals["cost"] = None
        tested = totals.get("target_tested", 0) or 0
        totals["unit_cost"] = totals.get("cost") / tested if totals.get("cost") is not None and tested else None
        methods[method] = totals
    statuses = sum_maps(
        summary["results"]["statuses"] for summary in summaries.values()
        if isinstance(summary.get("results"), Mapping)
        and isinstance(summary["results"].get("statuses"), Mapping)
    )
    first_statuses = first_summary.get("results")
    first_statuses = first_statuses.get("statuses") if isinstance(first_statuses, Mapping) else None
    if isinstance(first_statuses, Mapping):
        for name in shared_count_fields:
            if name in first_statuses:
                statuses[name] = first_statuses[name]
    result["methods"] = methods
    result["results"] = {"methods": methods, "statuses": statuses}
    route_matrix = deepcopy(result.get("route_matrix"))
    if isinstance(route_matrix, dict) and isinstance(route_matrix.get("rows"), list):
        for row in route_matrix["rows"]:
            if not isinstance(row, dict):
                continue
            key = (row.get("root_key"), row.get("lineage_key"), row.get("route"))
            target_states = []
            for target_summary in summaries.values():
                target_rows = target_summary.get("rows", ())
                target_row = next((item for item in target_rows if (
                    item.get("root_key"), item.get("lineage_key"), item.get("route")
                ) == key), None) if isinstance(target_rows, (list, tuple)) else None
                target_states.append((
                    isinstance(target_row, Mapping)
                    and target_row.get("attempt_status") == "attempted",
                    isinstance(target_row, Mapping)
                    and isinstance(target_row.get("route_attempt"), Mapping)
                    and target_row["route_attempt"].get("target_tested") is True,
                ))
            attempted = bool(target_states) and all(state[0] for state in target_states)
            tested = bool(target_states) and all(state[1] for state in target_states)
            row["attempt_status"] = "attempted" if attempted else "unattempted"
            row["route_attempt"] = ({
                "status": "attempted", "source": "multi-target-aggregate",
                "target_attempted": attempted, "target_tested": tested,
            } if attempted else None)
        matrix_counts = route_matrix.get("counts")
        if isinstance(matrix_counts, dict):
            eligible = [row for row in route_matrix["rows"] if (
                isinstance(row, Mapping) and row.get("status") == "recipe-ready"
            )]
            target_route_states = [
                (target_summary, row)
                for target_summary in summaries.values()
                for row in eligible
            ]
            matrix_counts["route_attempt_eligible_count"] = len(target_route_states)
            matrix_counts["route_unattempted"] = sum(
                not any(
                    item.get("attempt_status") == "attempted"
                    for item in target_summary.get("rows", ())
                    if isinstance(item, Mapping)
                    and (item.get("root_key"), item.get("lineage_key"), item.get("route"))
                    == (route.get("root_key"), route.get("lineage_key"), route.get("route"))
                )
                for target_summary, route in target_route_states
            )
            matrix_counts["route_attempt_complete"] = (
                matrix_counts["route_unattempted"] == 0
            )
            gaps = {
                gap for summary in summaries.values()
                for gap in summary.get("route_audit_gaps", ())
                if isinstance(gap, str) and gap != "route-attempt"
            }
            gaps.update(gap for gap in route_matrix.get("route_audit_gaps", ())
                        if isinstance(gap, str) and gap != "route-attempt")
            if not matrix_counts["route_attempt_complete"]:
                gaps.add("route-attempt")
            route_matrix["route_audit_gaps"] = sorted(gaps)
            route_matrix["route_audit_status"] = "complete" if not gaps else "incomplete"
    if isinstance(route_matrix, Mapping):
        result["route_audit_gaps"] = list(route_matrix.get("route_audit_gaps", ()))
        result["route_audit_status"] = route_matrix.get("route_audit_status")
        result["route_matrix"] = route_matrix
        result["plan_completion"] = _plan_completion(
            ledger, route_matrix, list(result["rows"]), result["counts"],
        )
    return result


def _plan_completion(
    ledger: Mapping[str, object], route_matrix: Mapping[str, object],
    rows: list[Mapping[str, object]], counts: Mapping[str, object],
) -> dict[str, object]:
    declared = ledger.get("counts")
    expected = declared.get("canonical_target_root_count") if isinstance(declared, Mapping) else None
    gaps = []
    if not isinstance(declared, Mapping) or declared.get("canonical_root_count_frozen") is not True or expected != 253:
        gaps.append("canonical-root-count")
    matrix_counts = route_matrix["counts"]
    if matrix_counts["root_count"] != expected:
        gaps.append("canonical-root-coverage")
    if matrix_counts["source_route_record_count"] != expected * 2 if isinstance(expected, int) else True:
        gaps.append("route-records")
    if route_matrix["route_audit_status"] != "complete":
        gaps.extend(route_matrix["route_audit_gaps"])
    if matrix_counts["single_recipe_gap"] or matrix_counts["program_recipe_gap"]:
        gaps.append("route-recipes")
    if counts.get("mapping_gap_records", 0) or counts.get("unbound_records", 0):
        gaps.append("execution-binding")
    execution_terminal_statuses = {
        "generation-gap", "transport-gap", "profile-gap", "seed-miss",
        "target-gap", "artifact-gap", "target-tested",
        "target-mismatch-candidate", "clean", "reference-rejected",
        "target-probe-tested", "target-probe-gap", "target-unsupported",
    }
    if any(row["status"] not in execution_terminal_statuses for row in rows):
        gaps.append("execution-evidence")
    gaps = sorted(set(gaps))
    return {"status": "complete" if not gaps else "incomplete", "gaps": gaps}


def _coverage_record(record: Mapping[str, object]) -> Mapping[str, object]:
    binding = record.get("root_binding")
    if not isinstance(binding, Mapping):
        result = dict(record)
    else:
        conflicts = []
        for name in (
            "root_key", "lineage_key", "route", "rule", "state", "observer",
                "root_recipe_digest", "generation_profile_digest",
                "generation_identity_digest", "target_binary_sha256",
                "target_identity_digest", "target_identity",
        ):
            same = record.get(name) == binding.get(name)
            if name == "route":
                left_route, right_route = _route_alias(record.get(name)), _route_alias(binding.get(name))
                same = left_route is not None and left_route == right_route
            if record.get(name) not in (None, "") and binding.get(name) not in (None, "") \
                    and not same:
                conflicts.append(name)
        top_method = record.get("method") or record.get("method_group")
        if top_method not in (None, "") and top_method != binding.get("method"):
            conflicts.append("method")
        top_target = record.get("target")
        if (isinstance(top_target, str) and top_target
                and not same_execution_backend(top_target, binding.get("target"))):
            conflicts.append("target")
        result = {**record, **binding}
        if (canonical_route := _route_alias(result.get("route"))) is not None:
            result["route"] = canonical_route
        if conflicts:
            result["_root_binding_conflict"] = tuple(dict.fromkeys(conflicts))
    counts = record.get("counts")
    root = record.get("root")
    if not isinstance(counts, Mapping) or not isinstance(root, Mapping):
        return result
    def positive(name: str) -> int:
        return _positive_count(counts.get(name, 0))
    witness = root.get("boundary_witness")
    aseed = root.get("aseed")
    root_status = root.get("status")
    root_failure = root.get("failure")
    terminal_status = record.get("terminal_status")
    failure_status = root_failure.get("status") if isinstance(root_failure, Mapping) else None
    stage_status = next(
        (
            value for value in (root_status, terminal_status, failure_status)
            if isinstance(value, str)
            and value in {
                "generation-gap", "transport-gap", "profile-gap", "seed-miss",
                "reference-rejected",
            }
        ),
        None,
    )
    profile_gap = stage_status == "profile-gap" or positive("profile_gap")
    reference_rejected = stage_status == "reference-rejected" or positive("reference_rejected")
    failed_before_execution = stage_status is not None
    known_root_statuses = {
        "not-applied", "pending", "generated", "seed-miss", "generation-gap",
        "transport-gap", "profile-gap", "reference-rejected",
    }
    root_reachable = (
        isinstance(root_status, str) and root_status in {"not-applied", "generated"}
        and not failed_before_execution and not reference_rejected
    )
    generation_gap = (
        stage_status == "generation-gap"
        or not isinstance(root_status, str)
        or root_status not in known_root_statuses
    )
    generated = (
        isinstance(root_status, str) and root_status in {"not-applied", "generated"}
        and not failed_before_execution
        and is_sha256_digest(root.get("raw_sha256"))
        and is_sha256_digest(root.get("rewrite_sha256"))
    )
    seed_valid = isinstance(aseed, Mapping) and aseed.get("status") == "passed"
    rewrite_valid = isinstance(witness, Mapping) and witness.get("realizes") is True
    if record.get("schema") == "program-candidate-ledger-v1":
        seed_valid = seed_valid and all(
            aseed.get(name) is True
            for name in ("build", "reference_outcome_valid", "realizes")
        )
    baseline = record.get("reference_baseline")
    target_baseline = record.get("target_baseline")
    baseline_only = (
        isinstance(target_baseline, Mapping)
        and not record.get("target_records")
        and not record.get("targets")
    )
    reference_valid = (
        generated and seed_valid and rewrite_valid
        and (positive("reference_valid") or baseline_only)
        and record.get("status") != "case-skipped"
        and not (
            isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "case-skipped"
        )
    )
    if record.get("schema") == "program-candidate-ledger-v1" and root_status not in {
        "generation-gap", "transport-gap", "profile-gap", "seed-miss",
    } and (
        record.get("status") != "case-skipped"
        and not (
            isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "case-skipped"
        )
        and (
        not isinstance(baseline, Mapping)
        or not isinstance(baseline.get("observation"), (list, tuple))
        or not baseline["observation"]
        or not isinstance(baseline.get("reference_identity"), Mapping)
        or not isinstance(baseline["reference_identity"].get("backend"), str)
            or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("backend"), str)
            or not item["backend"].strip()
            or item.get("outcome") not in {"normal", "completed", "trap", "nonzero-exit"}
            for item in baseline["observation"]
        )
        )
    ):
        reference_valid = False
        reference_rejected = True
    target_claim = positive("target_tested")
    target_unsupported = bool(
        positive("target_unsupported")
        or positive("case_skipped")
        or isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "case-skipped"
        or record.get("status") == "case-skipped"
    )
    record_status = record.get("status")
    target_mismatch_claim = positive("target_mismatch_candidate") or (
        isinstance(record_status, str)
        and record_status in {
            "target-mismatch-candidate", "target-state-mismatch-candidate",
        }
    )
    target_mismatch_count = counts.get("target_mismatch_candidate", 0)
    if type(target_mismatch_count) is not int:
        target_mismatch_count = 0
    result_claim = any(
        positive(name) if name != "target_mismatch_candidate" else target_mismatch_claim
        for name in ("target_mismatch_candidate", "clean")
    )
    target_attempted = any(
        positive(name)
        for name in (
            "target_attempted", "target_tested", "target_gap", "target_unsupported",
            "target_raw_recorded", "target_comparison_pending", "case_skipped",
        )
    )
    target_tested = bool(
        generated and seed_valid and rewrite_valid and reference_valid and target_claim
    )
    count_fields = (
        "target_attempted", "target_tested", "target_gap",
        "target_mismatch_candidate", "clean",
    )
    for name in ("target_raw_recorded", "target_comparison_pending"):
        if name in counts:
            count_fields = (*count_fields, name)
    if "target_unsupported" in counts:
        count_fields = (*count_fields, "target_unsupported")
    if "case_skipped" in counts:
        count_fields = (*count_fields, "case_skipped")
    if "artifact_gap" in counts:
        count_fields = (*count_fields, "artifact_gap")
    unsupported_count = counts.get("target_unsupported", counts.get("case_skipped", 0))
    target_rows = record.get("target_records")
    if not isinstance(target_rows, (list, tuple)) or not target_rows:
        target_records_by_target = record.get("target_records_by_target")
        if isinstance(target_records_by_target, Mapping):
            target_rows = target_records_by_target
        else:
            target_rows = record.get("target")
    if isinstance(target_rows, Mapping):
        target_rows = tuple(
            item
            for value in target_rows.values()
            for item in (value if isinstance(value, (list, tuple)) else (value,))
            if isinstance(item, Mapping)
        )
    if not isinstance(target_rows, (list, tuple)):
        target_rows = ()
    raw_pending_count = sum(
        int(item.get("target_attempted") is not False)
        for item in target_rows if _raw_target_pending(item)
    ) + int(_raw_target_pending(target_baseline))
    if raw_pending_count == 0:
        raw_pending_count = max(
            value if type(value) is int and value > 0 else 0
            for value in (
                counts.get("target_raw_recorded", 0),
                counts.get("target_comparison_pending", 0),
            )
        )
    target_attempted = target_attempted or bool(raw_pending_count)
    artifact_gap_count = counts.get("artifact_gap", 0)
    if "artifact_gap" not in counts:
        artifact_gap_count = int(
            isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "artifact-gap"
        ) + sum(
            isinstance(item, Mapping) and item.get("status") == "artifact-gap"
            for item in target_rows
        )
    artifact_gap = bool(
        positive("artifact_gap")
        or isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "artifact-gap"
        or any(
            isinstance(item, Mapping) and item.get("status") == "artifact-gap"
            for item in target_rows
        )
    )
    target_gap_count = counts.get("target_gap", 0)
    coverage_only_completed = sum(
        1 for item in target_rows
        if isinstance(item, Mapping)
        and item.get("coverage_only") is True
        and item.get("target_attempted") is not False
        and (
            isinstance(item.get("status"), str)
            and item.get("status") in {
                "clean", "target-mismatch-candidate", "target-tested",
            }
            or isinstance(item.get("terminal_disposition"), str)
            and item.get("terminal_disposition") in {
                "clean", "target-mismatch-candidate", "target-tested",
            }
        )
    )
    coverage_only_completed += int(
        isinstance(target_baseline, Mapping)
        and target_baseline.get("coverage_only") is True
        and target_baseline.get("status") == "target-tested"
        and isinstance(target_baseline.get("observations"), (list, tuple))
        and bool(target_baseline.get("observations"))
    )
    baseline_clean = int(
        isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "target-tested"
        and target_baseline.get("coverage_only") is not True
        and _comparison_qualified(target_baseline, "equivalent")
    )
    qualified_row_mismatches = sum(
        item.get("coverage_only") is not True
        and isinstance(item.get("status"), str)
        and item.get("status") in {
            "target-mismatch-candidate", "target-state-mismatch-candidate",
        }
        and _comparison_qualified(item, "non-equivalent")
        for item in target_rows if isinstance(item, Mapping)
    )
    qualified_row_clean = sum(
        item.get("coverage_only") is not True
        and item.get("status") == "clean"
        and _comparison_qualified(item, "equivalent")
        for item in target_rows if isinstance(item, Mapping)
    )
    baseline_mismatch = (
        isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "target-tested"
        and target_baseline.get("coverage_only") is not True
        and _comparison_qualified(target_baseline, "non-equivalent")
    )
    comparison_qualification_gap = any(
        item.get("coverage_only") is not True
        and (
            isinstance(item.get("status"), str)
            and item.get("status") in {
                "target-mismatch-candidate", "target-state-mismatch-candidate",
            }
            and not _comparison_qualified(item, "non-equivalent")
            or item.get("status") == "clean"
            and not _comparison_qualified(item, "equivalent")
        )
        for item in target_rows if isinstance(item, Mapping)
    )
    if isinstance(target_baseline, Mapping) \
            and target_baseline.get("status") == "target-tested" \
            and target_baseline.get("coverage_only") is not True \
            and not _comparison_qualified(target_baseline):
        comparison_qualification_gap = True
    comparison_qualification_gap |= (
        target_mismatch_count != qualified_row_mismatches + int(baseline_mismatch)
        or type(counts.get("clean")) is int
        and counts["clean"] != qualified_row_clean
    )
    count_contract_gap = (
        any(type(counts.get(name)) is not int or counts[name] < 0 for name in count_fields)
        or counts["target_tested"] > counts["target_attempted"]
        or "case_skipped" in counts and counts["case_skipped"] != unsupported_count
        or type(artifact_gap_count) is not int or artifact_gap_count < 0
        or counts["target_gap"] + unsupported_count + coverage_only_completed
        + raw_pending_count != counts["target_attempted"] - counts["target_tested"]
        or target_mismatch_count + counts["clean"] + baseline_clean
        != counts["target_tested"]
        or comparison_qualification_gap
    ) if all(type(counts.get(name)) is int for name in count_fields) else True
    result.update({
        "reachable": root_reachable,
        "generation_gap": generation_gap,
        "generated": generated,
        "seed_valid": seed_valid,
        "rewrite_valid": rewrite_valid,
        "reference_valid": reference_valid,
        "profile_gap": profile_gap,
        "reference_rejected": reference_rejected,
        "transport_gap": stage_status == "transport-gap",
        "seed_miss": stage_status == "seed-miss",
        "target_attempted": target_attempted,
        "target_tested": target_tested,
        "target_raw_recorded": bool(raw_pending_count),
        "target_comparison_pending": bool(raw_pending_count),
        "target_unsupported": target_unsupported,
        "artifact_gap": artifact_gap,
        "target_gap": bool(
            positive("target_gap") and artifact_gap_count < target_gap_count
            or (
                (target_attempted or target_claim or result_claim)
                and not target_tested and not target_unsupported
                and not artifact_gap
                and not coverage_only_completed
                and not raw_pending_count
            )
        ),
        "mismatch_candidate": target_tested and target_mismatch_claim,
        "clean": target_tested and positive("clean"),
    })
    if count_contract_gap:
        result.update(
            target_tested=False, target_gap=True,
            mismatch_candidate=False, clean=False,
            _coverage_counts_gap=(
                "target-comparison-unqualified"
                if comparison_qualification_gap else "target-counts-invalid"
            ),
        )
    if reference_rejected and not positive("reference_valid"):
        result["reference_valid"] = False
    if (
        record.get("schema") == "program-candidate-ledger-v1"
        and target_claim
        and (
            not isinstance(record.get("reference_baseline"), Mapping)
            or not (
                isinstance(record.get("target_records"), (list, tuple))
                and record["target_records"]
                or isinstance(record.get("target_baseline"), Mapping)
                and not record.get("target_records")
            )
        )
    ):
        result.update(
            target_attempted=False, target_tested=False, target_gap=True,
            mismatch_candidate=False, clean=False,
        )
    if failed_before_execution or result["generation_gap"]:
        result.update(
            generated=False, seed_valid=False, rewrite_valid=False,
            reference_valid=False,
            target_attempted=target_attempted or bool(coverage_only_completed),
            target_tested=False,
            target_gap=False,
            mismatch_candidate=False, clean=False,
        )
    target_identity = _target_identity_status(record)
    if (
        target_identity in {"same", "missing", "invalid", "binding-mismatch"}
        and positive("target_tested")
        and not (failed_before_execution or result["generation_gap"])
    ):
        result.update(
            target_tested=False, target_gap=True, clean=False,
            mismatch_candidate=False, target_identity_status=target_identity,
        )
    return result


def _target_identity_status(record: Mapping[str, object]) -> str | None:
    baseline = record.get("reference_baseline")
    targets = record.get("target_records")
    if targets is None or (isinstance(targets, (list, tuple)) and not targets):
        target_rows = record.get("target")
        if isinstance(target_rows, (list, tuple)):
            targets = target_rows
        elif target_rows not in (None, "") and not (
            isinstance(target_rows, str)
            and isinstance(record.get("root_binding"), Mapping)
            and target_rows == record["root_binding"].get("target")
            ):
            targets = target_rows
    pending_target_rows = False
    if isinstance(targets, (list, tuple)):
        pending_target_rows = any(_raw_target_pending(item) for item in targets)
        non_pending_targets = tuple(
            item for item in targets
            if not _raw_target_pending(item)
        )
        if len(non_pending_targets) != len(targets):
            targets = non_pending_targets
    if isinstance(targets, Mapping):
        binding = record.get("root_binding")
        name = binding.get("target") if isinstance(binding, Mapping) else None
        target = targets.get(name)
        if not isinstance(target, Mapping) and isinstance(name, str):
            target = next(
                (
                    item for target_name, item in targets.items()
                    if isinstance(target_name, str)
                    and same_execution_backend(target_name, name)
                ),
                None,
            )
        targets = (
            [target] if isinstance(target, Mapping)
            and isinstance(target.get("comparison"), Mapping) else None
        )
    target_baseline = record.get("target_baseline")
    targets_absent = targets is None or (
        isinstance(targets, (list, tuple)) and not targets
    )
    if (_raw_target_pending(target_baseline) or pending_target_rows) and targets_absent:
        return None
    if isinstance(target_baseline, Mapping) and targets_absent:
        if target_baseline.get("status") == "case-skipped":
            return None
        if target_baseline.get("status") != "target-tested":
            return "missing"
        observations = target_baseline.get("observations")
        if not isinstance(observations, (list, tuple)) or not observations:
            return "missing"
        binding = record.get("root_binding")
        expected_digest = (
            binding.get("target_identity_digest")
            if isinstance(binding, Mapping) else None
        )
        expected_binary = (
            binding.get("target_binary_sha256")
            if isinstance(binding, Mapping) else None
        )
        expected_target = (
            binding.get("target") if isinstance(binding, Mapping) else None
        )
        for item in observations:
            identity, status = _target_binary_identity(item)
            if status is not None or identity is None:
                return status or "missing"
            observed_backend = item.get("backend") if isinstance(item, Mapping) else None
            if not isinstance(observed_backend, str) or not observed_backend:
                return "missing"
            if (
                isinstance(expected_target, str) and expected_target
                and not same_execution_backend(observed_backend, expected_target)
            ):
                return "binding-mismatch"
            if not same_execution_backend(identity.backend, observed_backend):
                return "invalid"
            if (expected_digest not in (None, "", "runtime")
                    and identity.identity_digest != expected_digest
                    or expected_binary not in (None, "", "runtime")
                    and identity.binary_sha256 != expected_binary):
                return "binding-mismatch"
        return None
    if not isinstance(baseline, Mapping) and not isinstance(targets, (list, tuple)):
        return None
    if not isinstance(targets, (list, tuple)) or not targets:
        return "missing"
    coverage_only_rows = all(
        isinstance(target, Mapping) and target.get("coverage_only") is True
        for target in targets
    )
    reference = _execution_identity(
        baseline.get("reference_identity") if isinstance(baseline, Mapping) else None
    )
    for observation in (
        baseline.get("observation", ())
        if isinstance(baseline, Mapping)
        and isinstance(baseline.get("observation"), (list, tuple)) else ()
    ):
        for name, value in _execution_identity(observation).items():
            if not reference.get(name):
                reference[name] = value
    if not reference.get("backend") and not coverage_only_rows:
        return "missing"
    references = record.get("reference")
    if not isinstance(references, (list, tuple)):
        references = record.get("references")

    def artifact_for(target: Mapping[str, object], side: str) -> Mapping[str, object] | None:
        if side == "variant":
            artifact = target.get("artifact")
            return artifact if isinstance(artifact, Mapping) else None
        original_artifact = target.get("target_original_artifact")
        if isinstance(original_artifact, Mapping):
            return original_artifact
        index = target.get("reference_index")
        reference_record = (
            references[index] if isinstance(references, (list, tuple))
            and type(index) is int and 0 <= index < len(references)
            and isinstance(references[index], Mapping) else None
        )
        artifacts = reference_record.get("artifacts") if isinstance(reference_record, Mapping) else None
        artifact = artifacts.get("reference_original") if isinstance(artifacts, Mapping) else None
        return artifact if isinstance(artifact, Mapping) else None

    def artifact_status(row: Mapping[str, object], artifact: object) -> str | None:
        if not is_execution_backend(row.get("backend")) or row.get("backend") == "sail-riscv":
            return None
        if not isinstance(artifact, Mapping):
            return "missing"
        executable = (
            artifact.get("bare_executable_sha256") or artifact.get("executable_sha256")
            if row.get("backend") in _CAPSULE_BACKENDS else
            artifact.get("linux_executable_sha256") or artifact.get("executable_sha256")
        )
        values = tuple(row.get(name) for name in ("binary_sha256", "guest_elf_sha256"))
        if not is_sha256_digest(executable):
            return "missing"
        if any(value is not None and value != executable for value in values):
            return "invalid"
        return None if any(is_sha256_digest(value) and value == executable for value in values) else "missing"

    def artifact_contract_status(target: Mapping[str, object]) -> str | None:
        artifact = target.get("artifact")
        if not isinstance(artifact, Mapping):
            return "missing"
        if any(not is_sha256_digest(artifact.get(name)) for name in (
            "source_sha256", "executable_sha256", "parent_sha256",
        )) or not isinstance(artifact.get("run_params"), Mapping):
            return "missing"
        if is_sha256_digest(target.get("state_sha256")) \
                and artifact["source_sha256"] != target["state_sha256"]:
            return "invalid"
        index = target.get("reference_index")
        reference_record = (
            references[index] if isinstance(references, (list, tuple))
            and type(index) is int and 0 <= index < len(references)
            and isinstance(references[index], Mapping) else None
        )
        artifacts = reference_record.get("artifacts") if isinstance(reference_record, Mapping) else None
        expected = artifacts.get("reference_variant") if isinstance(artifacts, Mapping) else None
        if isinstance(expected, Mapping):
            for name in ("source_sha256", "executable_sha256", "parent_sha256"):
                if expected.get(name) not in (None, artifact.get(name)):
                    return "invalid"
            if isinstance(expected.get("run_params"), Mapping) \
                    and dict(expected["run_params"]) != dict(artifact["run_params"]):
                return "invalid"
        return None

    binding = record.get("root_binding")
    expected_target = binding.get("target") if isinstance(binding, Mapping) else None
    expected_target_digest = (
        binding.get("target_identity_digest")
        if isinstance(binding, Mapping) else None
    )
    expected_target_binary = (
        binding.get("target_binary_sha256")
        if isinstance(binding, Mapping) else None
    )
    statuses = []
    identities = []
    coverage_only_seen = False
    for target in targets:
        artifact_gap = artifact_contract_status(target)
        if artifact_gap:
            return artifact_gap
        comparison = target.get("comparison") if isinstance(target, Mapping) else None
        observations = comparison.get("observations") if isinstance(comparison, Mapping) else None
        if not isinstance(observations, Mapping):
            return "missing"
        for side in ("original", "variant"):
            rows = observations.get(side)
            if not isinstance(rows, (list, tuple)) or not rows:
                return "missing"
            for observation in rows:
                target_identity = _execution_identity(observation)
                if not target_identity.get("backend"):
                    return "missing"
                if (isinstance(expected_target, str) and expected_target
                        and not same_execution_backend(target_identity["backend"], expected_target)):
                    return "binding-mismatch"
                identity, identity_status = _target_binary_identity(observation)
                if identity_status:
                    return identity_status
                if target.get("coverage_only") is True:
                    observed_backend = (
                        observation.get("backend")
                        if isinstance(observation, Mapping) else None
                    )
                    if not isinstance(observed_backend, str) or not observed_backend:
                        return "missing"
                    if (
                        isinstance(expected_target, str) and expected_target
                        and not same_execution_backend(observed_backend, expected_target)
                    ):
                        return "binding-mismatch"
                    if not same_execution_backend(identity.backend, observed_backend):
                        return "invalid"
                    if (expected_target_digest not in (None, "")
                            and identity.identity_digest != expected_target_digest
                            or expected_target_binary not in (None, "")
                            and identity.binary_sha256 != expected_target_binary):
                        return "binding-mismatch"
                    artifact_identity = _execution_identity(target.get("artifact"))
                    if any(
                        artifact_identity.get(name) not in (None, "")
                        and artifact_identity[name] != getattr(identity, attr)
                        for name, attr in (
                            ("target_binary_sha256", "binary_sha256"),
                            ("target_identity_digest", "identity_digest"),
                        )
                    ):
                        return "invalid"
                    artifact_gap = artifact_status(
                        observation, artifact_for(target, side),
                    )
                    if artifact_gap:
                        return artifact_gap
                    identities.append(identity)
                    coverage_only_seen = True
                    continue
                if same_execution_backend(target_identity["backend"], reference["backend"]):
                    distinct = any(
                        reference.get(name) not in (None, "")
                        and reference[name] != getattr(identity, attr)
                        for name, attr in (
                            ("target_binary_sha256", "binary_sha256"),
                            ("target_identity_digest", "identity_digest"),
                        )
                    )
                    qemu_fallback = (
                        _recorded_qemu_reference_fallback(record)
                        and target_identity["backend"] == "qemu-riscv64"
                    )
                    if not distinct:
                        if qemu_fallback:
                            if (
                                expected_target_digest not in (None, "")
                                and identity.identity_digest != expected_target_digest
                            ) or (
                                expected_target_binary not in (None, "")
                                and identity.binary_sha256 != expected_target_binary
                            ):
                                return "binding-mismatch"
                            artifact_identity = _execution_identity(
                                target.get("artifact") if isinstance(target, Mapping) else None
                            )
                            if any(
                                artifact_identity.get(name) not in (None, "")
                                and artifact_identity[name] != getattr(identity, attr)
                                for name, attr in (
                                    ("target_binary_sha256", "binary_sha256"),
                                    ("target_identity_digest", "identity_digest"),
                                )
                            ):
                                return "invalid"
                            artifact_gap = artifact_status(
                                observation, artifact_for(target, side),
                            )
                            if artifact_gap:
                                return artifact_gap
                            identities.append(identity)
                            statuses.append("shared-qemu-fallback")
                            continue
                        return "same"
                    identities.append(identity)
                    statuses.append(
                        "shared-qemu-fallback" if qemu_fallback else "distinct"
                    )
                    continue
                if not same_execution_backend(identity.backend, target_identity["backend"]):
                    return "invalid"
                if (expected_target_digest not in (None, "")
                        and identity.identity_digest != expected_target_digest):
                    return "binding-mismatch"
                if (expected_target_binary not in (None, "")
                        and identity.binary_sha256 != expected_target_binary):
                    return "binding-mismatch"
                artifact_identity = _execution_identity(
                    target.get("artifact") if isinstance(target, Mapping) else None
                )
                if any(
                    artifact_identity.get(name) not in (None, "")
                    and artifact_identity[name] != getattr(identity, attr)
                    for name, attr in (
                        ("target_binary_sha256", "binary_sha256"),
                        ("target_identity_digest", "identity_digest"),
                    )
                ):
                    return "invalid"
                artifact_gap = artifact_status(observation, artifact_for(target, side))
                if artifact_gap:
                    return artifact_gap
                identities.append(identity)
                statuses.append("distinct")
    if not identities or any(item != identities[0] for item in identities[1:]):
        return "invalid"
    if coverage_only_seen:
        return "coverage-only"
    return statuses[0] if statuses and len(set(statuses)) == 1 else "missing"



def _target_binary_identity(value: object) -> tuple[TargetBinaryIdentity | None, str | None]:
    if not isinstance(value, Mapping):
        return None, "missing"
    containers, payloads = _target_identity_parts(value)
    if not payloads:
        return None, "missing"
    if any(
        "target_identity_status" in container
        and container.get("target_identity_status") != "verified"
        for container in containers
    ):
        return None, "invalid"
    try:
        identities = tuple(TargetBinaryIdentity.from_dict(item) for item in payloads)
    except (KeyError, TypeError, ValueError):
        return None, "invalid"
    identity = identities[0]
    if any(item != identity for item in identities[1:]):
        return None, "invalid"
    if not is_sha256_digest(payloads[0].get("identity_digest")):
        return None, "missing"
    if not identity.source_repository or not identity.source_commit:
        return None, "missing"
    if not _target_identity_containers_match(containers, identity):
        return None, "invalid"
    return identity, None


def _execution_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("reference_identity")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    result.update({
        name: value[name]
        for name in ("backend", "tool_version", "target_binary_sha256", "target_identity_digest")
        if value.get(name) not in (None, "")
    })
    containers, payloads = _target_identity_parts(value)
    for container in containers:
        for name in ("target_binary_sha256", "target_identity_digest"):
            if container.get(name) not in (None, ""):
                result[name] = container[name]
    for target in payloads:
        for source, destination in (
            ("binary_sha256", "target_binary_sha256"),
            ("identity_digest", "target_identity_digest"),
        ):
            if target.get(source) not in (None, ""):
                result[destination] = target[source]
    return result

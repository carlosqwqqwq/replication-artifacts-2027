"""共享 campaign 的 target baseline 阶段。"""

import subprocess
from collections.abc import Callable, Mapping, Sequence

from framework._util import _compare_mask_fields, is_sha256_digest
from framework.adapters.contracts import (
    CAPSULE_BACKENDS, is_execution_backend, same_execution_backend,
)
from framework.adapters.program import (
    BuiltProgram, _artifact_sha256_for_backend, _observation_outcome_valid,
    _observation_record, _observations,
    _target_unsupported_reason, compare_observations,
    target_baseline_contract_valid, target_risk_execution_contract_error,
)
from framework.direct_case import (
    CompareMask,
    native_trace_stdout_mismatch_is_compatible,
)
from framework.execution_environment import ExecutionEnvironmentError


def _observation_input_id(row: Mapping[str, object]) -> object:
    value = row.get("input_id")
    identity = row.get("reference_identity")
    return value if value not in (None, "") else (
        identity.get("input_id") if isinstance(identity, Mapping) else None
    )


def _observation_artifact_gaps(
    baseline: BuiltProgram, observations: Sequence[object], side: str,
) -> list[str]:
    gaps: list[str] = []
    for index, item in enumerate(observations):
        row = _observation_record(item)
        backend = row.get("backend")
        if backend == "sail-riscv":
            continue
        expected = _artifact_sha256_for_backend(baseline, backend)
        recorded, guest = row.get("binary_sha256"), row.get("guest_elf_sha256")
        if is_execution_backend(backend):
            if not is_sha256_digest(expected):
                gaps.append(f"{side}-selected-artifact-missing:{index}")
            if recorded is None and guest is None:
                gaps.append(f"{side}-observation-artifact-identity-missing:{index}")
        if any(value is not None and value != expected for value in (recorded, guest)):
            evidence = row.get("translation_evidence")
            details = evidence.get("details") if isinstance(evidence, Mapping) else None
            configuration = (
                details.get("configuration_identity")
                if isinstance(details, Mapping) else None
            )
            capsule_sha = (
                details.get("capsule_sha256") if isinstance(details, Mapping) else None
            )
            if capsule_sha is None and isinstance(configuration, Mapping):
                capsule_sha = configuration.get("capsule_sha256")
            if not (
                backend == "rvvm-riscv64"
                and isinstance(details, Mapping)
                and details.get("capsule_mode") == "bare-metal-no-ecall"
                and capsule_sha in (recorded, guest)
            ):
                gaps.append(f"{side}-artifact-identity-mismatch:{index}")
    return gaps


def _reference_baseline_qualification_gaps(
    baseline: BuiltProgram, reference_record: Mapping[str, object],
    observations: Sequence[object],
) -> list[str]:
    gaps: list[str] = []
    for field in ("qualification_gaps", "missing_fields", "validation_gap", "contract_error"):
        if reference_record.get(field):
            gaps.append(f"reference-record-{field}")
    if not observations:
        gaps.append("reference-observation-missing")
    identity = reference_record.get("reference_identity")
    backend = identity.get("backend") if isinstance(identity, Mapping) else None
    if not isinstance(backend, str) or not backend.strip():
        gaps.append("reference-backend-unidentified")
    if reference_record.get("reference_expected_contract") == "unverified":
        gaps.append("reference-expected-contract-unverified")
    for field, expected in (
        ("source_sha256", baseline.source_sha256),
        ("executable_sha256", baseline.executable_sha256),
    ):
        actual = reference_record.get(field)
        if not is_sha256_digest(expected) or actual != expected:
            gaps.append(f"reference-{field}-mismatch")

    expected_input_id = baseline.run_params.get("input_id")
    recorded_input_id = identity.get("input_id") if isinstance(identity, Mapping) else None
    if (
        isinstance(expected_input_id, str) and expected_input_id.strip()
        and recorded_input_id not in (None, "")
        and recorded_input_id != expected_input_id
    ):
        gaps.append("reference-input-identity-mismatch")

    for index, item in enumerate(observations):
        row = _observation_record(item)
        row_backend = row.get("backend")
        if not isinstance(row_backend, str) or not row_backend.strip():
            gaps.append(f"reference-backend-missing:{index}")
        elif isinstance(backend, str) and not same_execution_backend(row_backend, backend):
            gaps.append(f"reference-backend-mismatch:{index}")
        if (
            row.get("outcome") not in {"normal", "completed", "nonzero-exit", "trap"}
            or not _observation_outcome_valid(row)
        ):
            gaps.append(f"reference-outcome-invalid:{index}")
        if row.get("contract_error"):
            gaps.append(f"reference-contract-error:{index}")
        if row.get("validation_gap"):
            gaps.append(f"reference-validation-gap:{index}")
        if (
            isinstance(expected_input_id, str) and expected_input_id.strip()
            and _observation_input_id(row) != expected_input_id
        ):
            gaps.append(f"reference-input-id-mismatch:{index}")
        extra = row.get("extra_state")
        if (
            isinstance(extra, Mapping)
            and not native_trace_stdout_mismatch_is_compatible(extra)
        ):
            gaps.append(f"reference-state-witness-conflict:{index}")
    gaps.extend(_observation_artifact_gaps(baseline, observations, "reference"))
    return list(dict.fromkeys(gaps))


def evaluate_target_baseline(
    baseline: BuiltProgram,
    target_runner: Callable[[BuiltProgram], object],
    *,
    baseline_record: Mapping[str, object],
    binding: Mapping[str, object] | None,
    observer_fields: Sequence[str] = (),
    expected_trap: bool = False,
    target_observations: Sequence[object] | None = None,
) -> dict[str, object]:
    """执行一次 target、保留原始观察，并做一次差分比较。

    Target baseline 总会执行并保存 observation。比较资格单独记录，不能抹掉
    raw comparison，也不能改变执行状态。
    """
    probe_mode = isinstance(binding, Mapping) and binding.get("target_execution_mode") == "probe"
    target_status = "target-probe-tested" if probe_mode else "target-tested"
    target_gap = "target-probe-gap" if probe_mode else "target-gap"
    risk_identity = dict(binding or {})
    if (
        baseline.run_params.get("route") == "single"
        or risk_identity.get("route") == "single"
    ):
        risk_identity["route"] = "single"
        risk_identity["expected_trap"] = expected_trap

    def result(
        status: str,
        target_observations: tuple[object, ...],
        comparison: Mapping[str, object],
        *, reason: str | None = None,
    ) -> dict[str, object]:
        reference_observations = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        reference_gaps = _reference_baseline_qualification_gaps(
            baseline, baseline_record, reference_observations,
        )
        comparison_gaps = list(reference_gaps)
        if comparison.get("qualification_gaps"):
            comparison_gaps.append("comparison-identity-gap")
        if comparison.get("missing_fields"):
            comparison_gaps.append("comparison-fields-missing")
        if comparison.get("target_execution_contract_error"):
            comparison_gaps.append("target-execution-contract-error")
        for index, item in enumerate(target_observations):
            row = _observation_record(item)
            if (
                row.get("outcome") not in {"normal", "completed", "nonzero-exit", "trap"}
                or not _observation_outcome_valid(row)
            ):
                comparison_gaps.append(f"target-outcome-invalid:{index}")
            if row.get("contract_error"):
                comparison_gaps.append(f"target-contract-error:{index}")
            if row.get("validation_gap"):
                comparison_gaps.append(f"target-validation-gap:{index}")
            expected_input_id = baseline.run_params.get("input_id")
            if (
                isinstance(expected_input_id, str) and expected_input_id.strip()
                and _observation_input_id(row) != expected_input_id
            ):
                comparison_gaps.append(f"target-input-id-mismatch:{index}")
            if (
                isinstance(binding, Mapping)
                and binding.get("target") not in (None, "")
                and not is_execution_backend(row.get("backend"))
            ):
                comparison_gaps.append(f"target-backend-invalid:{index}")
            if is_execution_backend(row.get("backend")) and not target_baseline_contract_valid(
                row, binding, expected_trap=expected_trap,
            ):
                comparison_gaps.append(f"target-execution-contract-invalid:{index}")
        comparison_gaps.extend(
            _observation_artifact_gaps(baseline, target_observations, "target")
        )
        comparison_status = comparison.get("status")
        if comparison_status not in {"equivalent", "non-equivalent"}:
            comparison_gaps.append("comparison-status-unqualified")
        comparison_qualified = (
            comparison_status in {"equivalent", "non-equivalent"}
            and not comparison_gaps
        )
        record = {
            "status": status,
            "source_sha256": baseline.source_sha256,
            "executable_sha256": baseline.executable_sha256,
            "reference_baseline_valid": not reference_gaps,
            "reference_baseline_qualification_gaps": reference_gaps,
            "comparison_qualified": comparison_qualified,
            "comparison_qualification_gaps": list(dict.fromkeys(comparison_gaps)),
            "observation_count": len(target_observations),
            "observations": [
                _observation_record(item) for item in target_observations
            ],
            "comparison": {
                "status": comparison.get("status"),
                "comparison_mode": comparison.get("comparison_mode"),
                "differences": comparison.get("differences", ()),
                **({"missing_fields": comparison["missing_fields"]}
                   if comparison.get("missing_fields") else {}),
                **({"qualification_gaps": comparison["qualification_gaps"]}
                   if comparison.get("qualification_gaps") else {}),
                **({"target_execution_contract_error": comparison[
                    "target_execution_contract_error"
                ]} if comparison.get("target_execution_contract_error") else {}),
            },
        }
        if reason:
            record["reason"] = reason
        return record

    try:
        target_observations = _observations(
            target_runner(baseline) if target_observations is None else target_observations
        )
        unsupported_reason = next(
            (
                reason for item in target_observations
                if (reason := _target_unsupported_reason(
                    item, expected_trap=expected_trap,
                )) is not None
            ),
            None,
        )
        if unsupported_reason is not None:
            return result(
                "case-skipped", target_observations,
                {
                    "status": "unsupported-isa",
                    "comparison_mode": "target-only",
                    "differences": {"target": unsupported_reason},
                },
                reason=unsupported_reason,
            )
        reference_observations = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        if not reference_observations:
            return result(
                target_gap, target_observations,
                {
                    "status": "reference-gap",
                    "comparison_mode": "cross-backend",
                    "differences": {"reference_observation": "missing"},
                },
                reason="reference-observation-missing",
            )
        if not target_observations:
            return result(
                target_gap, target_observations,
                {
                    "status": "reference-gap",
                    "comparison_mode": "cross-backend",
                    "differences": {"target_observation": "missing"},
                },
                reason="target-observation-missing",
            )
        single_case = baseline.run_params.get("single_case")
        raw_compare_mask = (
            single_case.get("compare_mask")
            if isinstance(single_case, Mapping) else None
        )
        compare_mask = (
            CompareMask.from_dict(dict(raw_compare_mask))
            if isinstance(raw_compare_mask, Mapping) else None
        )
        comparison_fields = (
            _compare_mask_fields(compare_mask)
            if compare_mask is not None else
            tuple(observer_fields) if observer_fields else None
        )
        if (
            compare_mask is None
            and isinstance(binding, Mapping)
            and binding.get("target") in CAPSULE_BACKENDS
        ):
            comparison_fields = (
                ("outcome",) if expected_trap else ("outcome", "exit_code")
            ) + tuple(observer_fields or ())
        if (
            isinstance(binding, Mapping)
            and binding.get("target") in CAPSULE_BACKENDS
            and comparison_fields is not None
        ):
            comparison_fields = tuple(
                field for field in comparison_fields
                if field not in {"checkpoint_pc", "fault_pc"}
            )
        comparison = compare_observations(
            reference_observations, target_observations,
            comparison_fields,
            cross_backend=True,
        )
        risk_execution_error = next(
            (
                error for item in target_observations
                if (error := target_risk_execution_contract_error(
                    _observation_record(item), risk_identity,
                )) is not None
            ),
            None,
        )
        if risk_execution_error is not None:
            comparison = {
                **comparison,
                "target_execution_contract_error": risk_execution_error,
            }
        status = (
            target_status
            if comparison.get("status") in {"equivalent", "non-equivalent"}
            else target_gap
        )
        return result(status, target_observations, comparison)
    except Exception as error:
        unsupported_reason = _target_unsupported_reason(
            error, expected_trap=expected_trap,
        )
        if unsupported_reason is not None:
            return {
                "status": "case-skipped",
                "source_sha256": baseline.source_sha256,
                "executable_sha256": baseline.executable_sha256,
                "observations": [],
                "comparison": {
                    "status": "unsupported-isa",
                    "comparison_mode": "target-only",
                    "differences": {"target": unsupported_reason},
                },
                "reason": unsupported_reason,
            }
        if "artifact-missing" in str(error) or "artifact-gap" in str(error):
            return {
                "status": "artifact-gap",
                "source_sha256": baseline.source_sha256,
                "executable_sha256": baseline.executable_sha256,
                "observations": [],
                "comparison": {
                    "status": "artifact-gap",
                    "comparison_mode": "target-only",
                    "differences": {"target": str(error)},
                },
                "reason": f"artifact-gap:{error}",
            }
        transport_error = isinstance(
            error, (ExecutionEnvironmentError, OSError, subprocess.SubprocessError),
        )
        return {
            "status": (
                ("target-probe-gap" if probe_mode else "transport-gap")
                if transport_error else target_gap
            ),
            "reason": str(error) if transport_error else f"{type(error).__name__}: {error}",
        }

"""完整程序的 EMI、MCMC、reference 和 target 单链。"""

import hashlib
import fcntl
import inspect
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from framework._util import (
    atomic_write_json, canonical_digest, is_sha256_digest, json_safe as _json_safe,
    object_field as _object_field,
)
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from framework.direct_case import expected_trap_from_dataflow
from framework import campaign_contract as _campaign_contract
from framework.campaign_route import CampaignRoute
from framework.execution_environment import ExecutionEnvironmentError
from framework.riscv_catalog import OFFICIAL_ALL_CATALOG_FORMS
from framework.reference_fallback import (
    configured_k1_qemu_fallback_classes,
    qemu_fallback_policy_matches,
)
from framework.adapters.program import (
    BuiltProgram,
    BuilderCallback,
    Program,
    _INSTRUCTION_RE,
    _comparison_qualified,
    _record_has_qualification_gap,
    _artifact_record as _built_artifact_record,
    _observation_record,
    _observations,
    _semantic_witness_contract,
    _reference_failure_status,
    _reference_exception_status,
    _build_cache_key,
    build_program,
    enumerate_case_rewrites,
    rewrite_program,
    run_emi_pair,
)
from framework.adapters.contracts import (
    route_admission_gap,
    same_execution_backend as _same_execution_backend,
)
from framework.adapters.runner import guest_trap_observed, source_pc_map as _source_pc_map
from framework.mcmc.program import _RNG_SCHEME, _validate_beta, sample_program
from framework.observability import (
    _is_observation_side_channel_field,
)
from framework.program_target_stage import (
    _reference_baseline_qualification_gaps, evaluate_target_baseline,
)
from framework.target_evidence import compact_record, make_evidence_ref
from framework.rvemi.program import (
    ProgramVariant, _logical_source_lines, _pc_value, _profile_action,
    _profile_input_matches, _source_include_files,
    _SOURCE_BINARY_PREFIX, apply_profile, build_profile, profile_actions,
    program_sha256,
)
from framework.rvgen.provider import CaseProgram
from framework.supply.model_candidate_supply import rewrite_hint_edits, supply_case_rewrites


def _artifact_gap_reason(artifacts: object) -> str | None:
    """Return a durable reason when a Target job cannot execute its pair."""
    if not isinstance(artifacts, Mapping):
        return "reference-pair-artifacts-missing"
    missing = [
        name for name in ("reference_original", "reference_variant")
        if not isinstance(artifacts.get(name), Mapping)
    ]
    return (
        "reference-pair-artifacts-incomplete:" + ",".join(missing)
        if missing else None
    )


def _compact_campaign_event(event: Mapping[str, object]) -> dict[str, object]:
    """Keep transition facts; source text and materialized action pools are redundant."""
    result = dict(event)
    result.pop("action_pool", None)
    result.pop("reverse_action_pool", None)
    action = event.get("action")
    if not isinstance(action, Mapping):
        return result
    compact_action = dict(action)
    payload = action.get("payload")
    if isinstance(payload, Mapping) and isinstance(payload.get("source"), str):
        compact_payload = dict(payload)
        source = compact_payload.pop("source")
        compact_payload["source_sha256"] = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        compact_payload["source_length"] = len(source)
        compact_action["payload"] = compact_payload
    result["action"] = compact_action
    return result


def _compact_chain_record(chain: Mapping[str, object]) -> dict[str, object]:
    """Persist the chain without retaining repeated sampler objects."""
    record = dict(chain)
    events = chain.get("events")
    if isinstance(events, list):
        record["events"] = [
            _compact_campaign_event(event)
            for event in events if isinstance(event, Mapping)
        ]
    final = chain.get("final")
    if final is not None:
        record["final"] = {
            "sha256": getattr(final, "sha256", None),
            "source_length": len(getattr(final, "source", "")),
        }
    record.pop("profile", None)
    run = chain.get("run")
    samples = getattr(run, "samples", None)
    attempts = getattr(run, "mh_attempts", None)
    if isinstance(run, Mapping):
        samples = run.get("samples")
        attempts = run.get("mh_attempts")
    record["run"] = {
        "mh_attempts": attempts if isinstance(attempts, int) else 0,
        "sample_count": len(samples) if isinstance(samples, (list, tuple)) else 0,
    }
    return _json_safe(record)


def _reference_fallback_summary(observations: Sequence[object]) -> dict[str, object] | None:
    for item in observations:
        extra = _object_field(item, "extra_state")
        fallback = extra.get("reference_fallback") if isinstance(extra, Mapping) else None
        if isinstance(fallback, Mapping):
            return dict(fallback)
    return None


def _attach_reference_fallback_evidence(
    observations: Sequence[object], fallback: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Carry the probe's original K1 attempt into the baseline once."""
    result = [_observation_record(item) for item in observations]
    if result:
        extra = result[0].get("extra_state")
        result[0]["extra_state"] = {
            **(dict(extra) if isinstance(extra, Mapping) else {}),
            "reference_fallback": dict(fallback),
        }
    return tuple(result)


def _reference_fallback_matches_binding(
    binding: Mapping[str, object] | None,
    expected_identity: Mapping[str, object],
    fallback: Mapping[str, object] | None,
) -> bool:
    policy = binding.get("reference_fallback_policy") if isinstance(binding, Mapping) else None
    attempts = fallback.get("k1_attempts") if isinstance(fallback, Mapping) else None
    allowed_classes = (
        configured_k1_qemu_fallback_classes(policy.get("on"))
        if isinstance(policy, Mapping) else None
    )
    fallback_reason = fallback.get("fallback_reason") if isinstance(fallback, Mapping) else None
    return (
        isinstance(policy, Mapping)
        and isinstance(fallback, Mapping)
        and isinstance(attempts, (list, tuple))
        and expected_identity.get("backend") == "native-rv64"
        and policy.get("backend") == "qemu-riscv64"
        and policy.get("target_id") == "T-QEMU"
        and allowed_classes is not None
        and policy.get("selection_scope") == "whole-campaign"
        and fallback.get("schema_version") == "rq1-reference-fallback-v1"
        and fallback.get("preferred_backend") == "native-rv64"
        and fallback.get("effective_backend") == policy.get("backend")
        and qemu_fallback_policy_matches(
            fallback.get("policy"), fallback_reason, allowed_classes,
        )
        and fallback.get("selection_scope") == policy.get("selection_scope")
        and fallback.get("target_id") == policy.get("target_id")
        and is_sha256_digest(fallback.get("k1_attempt_digest"))
        and canonical_digest(attempts) == fallback.get("k1_attempt_digest")
        and any(
            isinstance(item, Mapping)
            and item.get("backend") == "native-rv64"
            and isinstance(item.get("extra_state"), Mapping)
            and item["extra_state"].get("runner_failure_class") == fallback_reason
            and item["extra_state"].get("runner_failure_class") in allowed_classes
            for item in attempts
        )
    )


def _model_toggle_action(
    program: object, profile: object, hint: Mapping[str, object] | None,
    *, raw_sha256: str, rewritten_sha256: str,
) -> dict[str, object] | None:
    """Expose the accepted model edit as a reversible online EMI action."""
    if not isinstance(hint, Mapping):
        return None
    edits = hint.get("edits")
    if not isinstance(edits, list) or len(edits) != 1:
        return None
    edit = edits[0]
    if not isinstance(edit, Mapping):
        return None
    span = edit.get("source_span")
    payload = edit.get("payload")
    source_text = edit.get("source_text")
    if (
        not isinstance(span, Mapping) or type(span.get("start")) is not int
        or not isinstance(payload, Mapping)
        or not isinstance(payload.get("mnemonic"), str)
        or not isinstance(payload.get("operands"), list)
        or not all(isinstance(value, str) for value in payload["operands"])
        or not isinstance(source_text, str) or not source_text.strip()
    ):
        return None
    current_sha256 = program_sha256(program)
    target_text = (
        source_text
        if current_sha256 == rewritten_sha256 else
        f"{payload['mnemonic']} {', '.join(payload['operands'])}"
        if current_sha256 == raw_sha256 else None
    )
    if target_text is None:
        return None
    line_candidates = {
        value for value in (int(span["start"]) - 1, int(span["start"]))
        if value >= 0
    }
    row = next(
        (
            item for item in getattr(profile, "instructions", ())
            if isinstance(item, Mapping) and item.get("line") in line_candidates
        ),
        None,
    )
    if not isinstance(row, Mapping):
        return None
    occurrence_ids = tuple(
        item.get("id") for item in getattr(profile, "occurrences", ())
        if isinstance(item, Mapping) and item.get("instruction_id") == row.get("id")
        and isinstance(item.get("id"), str)
    )
    return _profile_action(
        row, profile, "R3", target_text,
        occurrence_ids[0] if len(occurrence_ids) == 1 else None,
        occurrence_ids, profile_digest=getattr(profile, "profile_digest", None),
    )


def _profile_observations(observations: Sequence[object], executable: Path) -> tuple[object, ...]:
    source_map = _source_pc_map(executable)
    if source_map is None:
        return tuple(observations)
    output: list[object] = []
    for item in observations:
        if isinstance(item, Mapping):
            evidence = item.get("translation_evidence")
            if isinstance(evidence, Mapping):
                evidence = dict(evidence)
                details = evidence.get("details")
                evidence["details"] = {
                    **(dict(details) if isinstance(details, Mapping) else {}),
                    "source_pc_map": source_map,
                }
            else:
                evidence = {
                    "backend": item.get("backend") or "native-rv64",
                    "details": {"source_pc_map": source_map},
                }
            output.append({
                **item,
                "binary_sha256": source_map["binary_sha256"],
                "translation_evidence": evidence,
            })
            continue
        # runner 产出的是 Observation dataclass；源码映射同样必须挂上，
        # 否则 build_profile 会以 executed-pc-unmapped 拒绝整条 program 链。
        evidence = _object_field(item, "translation_evidence")
        details = dict(_object_field(evidence, "details") or {})
        details["source_pc_map"] = source_map
        # contract-gap 的 observation 没有 translation_evidence；直接 replace(None)
        # 会抛 "replace() should be called on dataclass instances"，
        # 把真实的 runner-contract-gap 掩盖成 profile-gap。
        if evidence is None:
            output.append(item)
            continue
        output.append(replace(
            item,
            binary_sha256=source_map["binary_sha256"],
            translation_evidence=replace(evidence, details=details),
        ))
    return tuple(output)


def reference_witness_source_lines(
    program: Program,
    builder: BuilderCallback,
    reference_runner: Callable[[BuiltProgram], object],
    *,
    work_dir: str | Path,
    timeout_seconds: float | None = None,
) -> tuple[frozenset[int], dict[str, object]]:
    """Probe the raw program and return source lines present in its path.

    The probe is deliberately reference-only.  It supplies the model with an
    executable witness candidate pool; the normal rewritten-program profile
    remains the final admission check.
    """
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, object] = {
        "schema": "program-reference-witness-probe-v1",
        "status": "not-started",
        "program_sha256": program_sha256(program),
        "work_dir": str(root),
    }

    def observation_summary(item: object) -> dict[str, object]:
        result = {
            name: value for name in (
                "backend", "outcome", "exit_code", "instruction_count",
                "input_id", "contract_error",
            ) if (value := _object_field(item, name)) is not None
        }
        pcs = _object_field(item, "executed_pcs")
        if isinstance(pcs, (list, tuple)):
            result["executed_pc_count"] = len(pcs)
        return result

    try:
        artifact = replace(
            build_program(program, builder, work_dir=root),
            parent_sha256=None,
        )
        observations = _observations(reference_runner(
            artifact,
            **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
        ))
        if fallback := _reference_fallback_summary(observations):
            evidence["reference_fallback"] = fallback
        evidence.update({
            "source_sha256": artifact.source_sha256,
            "executable_sha256": artifact.executable_sha256,
            "executable": str(artifact.executable_path),
            "observations": [observation_summary(item) for item in observations],
        })
        if not observations:
            evidence.update(status="gap", reason="reference-observation-missing")
            return frozenset(), evidence
        if any(_object_field(item, "contract_error") for item in observations):
            evidence.update(status="gap", reason="reference-contract-error")
            return frozenset(), evidence
        profile_result = _campaign_contract.attach_execution_identity(
            build_profile(
                program,
                _profile_observations(observations, artifact.executable_path),
                input_identity=_campaign_contract.profile_identity(artifact, observations),
            ),
            artifact,
            observations,
        )
        profile = getattr(profile_result, "profile", profile_result)
        if profile is None:
            profile_reason = getattr(profile_result, "reason", None) or "profile-missing"
            # RISC-V-DV's generated #line directives can leave harness-only
            # DWARF rows outside the program profile.  For candidate admission
            # the source map plus the observed PCs is sufficient and avoids
            # turning a usable execution witness into a false profile gap.
            if profile_reason == "profile-gap:executed-pc-unmapped":
                source_map = _source_pc_map(artifact.executable_path)
                observed_pcs = {
                    parsed for item in observations
                    for value in (_object_field(item, "executed_pcs"),)
                    if isinstance(value, (list, tuple))
                    for pc in value
                    if (parsed := _pc_value(pc)) is not None
                }
                mapped_lines = source_map.get("lines") if isinstance(source_map, Mapping) else None
                if observed_pcs and isinstance(mapped_lines, Mapping):
                    logical_witness = {
                        int(line): tuple(
                            _pc_value(pc) for pc in values
                            if _pc_value(pc) is not None
                        )
                        for line, values in mapped_lines.items()
                        if str(line).isdigit() and isinstance(values, (list, tuple))
                    }
                    def instruction_key(value: str) -> tuple[str, tuple[str, ...]] | None:
                        match = _INSTRUCTION_RE.search(value.rstrip("\r\n"))
                        if match is None:
                            return None
                        return (
                            match["mnemonic"].lower().replace("_", "."),
                            tuple(
                                item.strip().lower()
                                for item in match["operands"].split(",")
                                if item.strip()
                            ),
                        )

                    harness_path = artifact.executable_path.with_suffix(".S")
                    harness_lines = (
                        harness_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if harness_path.is_file() else ()
                    )
                    harness_logical = _logical_source_lines(harness_lines)
                    executed_keys = {
                        instruction_key(line)
                        for index, line in enumerate(harness_lines)
                        if index < len(harness_logical)
                        and harness_logical[index] is not None
                        and any(
                            pc in observed_pcs
                            for pc in logical_witness.get(harness_logical[index], ())
                        )
                        and instruction_key(line) is not None
                    }
                    source_lines = _program_source(program).splitlines(keepends=True)
                    executed_lines = frozenset(
                        index + 1 for index, line in enumerate(source_lines)
                        if instruction_key(line) in executed_keys
                    )
                    mapping_mode = "harness-instruction-identity"
                    if not executed_lines:
                        logical_lines = _logical_source_lines(source_lines)
                        executed_lines = frozenset(
                            index + 1
                            for index, logical in enumerate(logical_lines)
                            if logical is not None
                            and any(
                                pc in observed_pcs
                                for pc in logical_witness.get(logical, ())
                            )
                        )
                        mapping_mode = "logical-source-line"
                    if executed_lines:
                        evidence.update({
                            "status": "recorded",
                            "profile_status": "source-map-witness",
                            "profile_gap_fallback": profile_reason,
                            "source_witness_mapping": mapping_mode,
                            "executed_source_lines": sorted(executed_lines),
                            "executed_source_line_count": len(executed_lines),
                            "reference_identity": _campaign_contract.reference_identity(observations),
                        })
                        return executed_lines, evidence
            evidence.update(
                status="gap",
                reason=profile_reason,
            )
            return frozenset(), evidence
        executed_lines = frozenset(
            row["line"] + 1
            for row in getattr(profile, "instructions", ())
            if isinstance(row, Mapping)
            and type(row.get("line")) is int
            and row.get("executed") is True
        )
        evidence.update({
            "status": "recorded",
            "profile_status": "recorded",
            "executed_source_lines": sorted(executed_lines),
            "executed_source_line_count": len(executed_lines),
            "reference_identity": _campaign_contract.reference_identity(observations),
            "profile_provenance": getattr(profile, "provenance", {}),
        })
        return executed_lines, evidence
    except Exception as error:
        evidence.update(status="gap", reason=f"{type(error).__name__}: {error}")
        return frozenset(), evidence


_EMPTY_COUNTS = {
    "steps": 0, "emi_rejected": 0, "reference_rejected": 0, "reference_gap": 0,
    "proposal_gap": 0, "seed_miss": 0, "parameter_gap": 0, "transport_gap": 0,
    "profile_gap": 0, "refresh_gap": 0, "reference_valid": 0,
    "target_attempted": 0, "target_tested": 0, "target_gap": 0,
    "target_raw_recorded": 0, "target_comparison_pending": 0,
    "artifact_gap": 0,
    "target_unsupported": 0, "case_skipped": 0,
    "target_gap_event": 0,
    "target_wall_clock_pending": 0,
    "target_transport_gap": 0, "target_mismatch_candidate": 0, "clean": 0,
    "mh_attempts": 0, "accepted": 0, "mh_rejected": 0,
}


def _compact_target_evidence_enabled() -> bool:
    """Enable bounded Target records for raw-only or explicit compact runs."""
    return os.environ.get("RQ1_COMPACT_TARGET_EVIDENCE") == "1" \
        or os.environ.get("RQ1_RAW_COVERAGE_ONLY") == "1"


def _target_baseline_counts(record: Mapping[str, object] | None) -> dict[str, int]:
    if not isinstance(record, Mapping):
        return {}
    status = record.get("status")
    if record.get("coverage_only") is True and status == "target-tested":
        return {
            "target_attempted": 1, "target_tested": 0, "target_gap": 0,
            "target_raw_recorded": 0, "target_comparison_pending": 0,
            "target_unsupported": 0, "case_skipped": 0,
            "target_mismatch_candidate": 0, "target_transport_gap": 0,
            "target_wall_clock_pending": 0,
        }
    if status == "case-skipped":
        return {
            "target_attempted": 1, "target_tested": 0, "target_gap": 0,
            "target_raw_recorded": 0, "target_comparison_pending": 0,
            "target_unsupported": 1, "case_skipped": 1,
            "target_mismatch_candidate": 0, "target_transport_gap": 0,
            "target_wall_clock_pending": 0,
        }
    if status == "artifact-gap":
        return {
            "target_attempted": 1, "target_tested": 0, "target_gap": 1,
            "target_raw_recorded": 0, "target_comparison_pending": 0,
            "artifact_gap": 1, "target_unsupported": 0, "case_skipped": 0,
            "target_mismatch_candidate": 0, "target_transport_gap": 0,
            "target_wall_clock_pending": 0,
        }
    comparison = record.get("comparison")
    if status == "raw-recorded" and record.get("comparison_pending") is True:
        return {
            "target_attempted": 1, "target_tested": 0, "target_gap": 0,
            "target_raw_recorded": 1, "target_comparison_pending": 1,
            "target_mismatch_candidate": 0, "target_transport_gap": 0,
            "target_wall_clock_pending": 0,
        }
    tested = int(status == "target-tested" and _comparison_qualified(record))
    comparison_gap = int(status == "target-tested" and not tested)
    gap = int(status not in {
        "target-tested", "target-probe-tested", "target-probe-gap",
    }) + comparison_gap
    return {
        "target_attempted": tested + gap,
        "target_tested": tested,
        "target_gap": gap,
        "target_mismatch_candidate": int(
            tested and isinstance(comparison, Mapping)
            and comparison.get("status") == "non-equivalent"
        ),
        "target_transport_gap": int(
            status == "transport-gap"
            or record.get("failure_class") == "transport-gap"
        ),
        "target_wall_clock_pending": int(
            record.get("deadline_censored") is True
            or record.get("failure_class") == "campaign-wall-clock-exhausted"
        ),
    }


def _campaign_deadline_exhausted(
    deadline: float | None, result: Mapping[str, object] | None,
) -> bool:
    """把在线 campaign 到期后的未完成 Target 标成 right-censor。"""
    if not isinstance(result, Mapping) or result.get("status") not in {
        "target-gap", "target-probe-gap", "transport-gap",
    }:
        return False

    def marked(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        extra = value.get("extra_state")
        if isinstance(extra, Mapping) and (
            extra.get("campaign_wall_clock_exhausted") is True
            or extra.get("target_timeout_origin") == "campaign-wall-clock"
        ):
            return True
        summary = value.get("target_observation_summary")
        return isinstance(summary, list) and any(
            isinstance(row, Mapping)
            and (
                row.get("campaign_wall_clock_exhausted") is True
                or row.get("target_timeout_origin") == "campaign-wall-clock"
            )
            for row in summary
        )

    target = result.get("target")
    if marked(result) or marked(target):
        return True
    observations = result.get("observations")
    if isinstance(observations, (list, tuple)) and any(
        marked(item) for item in observations
    ):
        return True
    return bool(
        deadline is not None
        and time.monotonic() >= deadline
    )


def _mark_campaign_deadline_observation(value: object) -> object:
    """给到期时返回的 observation 留下来源标记，不修改 Target 本身。"""
    extra = _object_field(value, "extra_state")
    state = dict(extra) if isinstance(extra, Mapping) else {}
    state.update({
        "campaign_wall_clock_exhausted": True,
        "target_timeout_origin": "campaign-wall-clock",
    })
    if isinstance(value, Mapping):
        return {**value, "extra_state": state}
    try:
        return replace(value, extra_state=state)
    except (TypeError, ValueError):
        return value


def _mark_campaign_deadline_result(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(_mark_campaign_deadline_observation(item) for item in value)
    if isinstance(value, list):
        return [_mark_campaign_deadline_observation(item) for item in value]
    return _mark_campaign_deadline_observation(value) if value is not None else value


def _source_includes(program: Program) -> dict[str, str]:
    path = getattr(program, "program", None)
    if path is None or Path(path).suffix.lower() != ".s":
        return {}
    path = Path(path).resolve()
    def snapshot(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return _SOURCE_BINARY_PREFIX + data.hex()
    return {
        name: snapshot(data)
        for name, data in _source_include_files(path).items()
        if name != path.name
    }


def _program_source(program: Program) -> str:
    if isinstance(program, ProgramVariant):
        return program.source
    path = Path(getattr(program, "program", program))
    return path.read_bytes().decode("utf-8") if path.suffix.lower() == ".s" else ""


def _rewrite_realizes(before: Program, after: Program, hint: Mapping[str, object]) -> bool:
    if not isinstance(hint, Mapping) or not rewrite_hint_edits(hint):
        return False
    if dict(getattr(before, "run_params", {})) != dict(getattr(after, "run_params", {})):
        return False
    def file_lines(program: Program, source_name: str | None):
        if not source_name:
            return _program_source(program).splitlines(keepends=True)
        program_path = getattr(program, "program", None)
        if program_path is None:
            return None
        try:
            base = Path(program_path).resolve().parent
            target = (base / source_name).resolve()
            target.relative_to(base)
            return target.read_bytes().decode("utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError, TypeError, ValueError):
            return None

    ordered: list[Mapping[str, object]] = []
    for edit in rewrite_hint_edits(hint):
        if not isinstance(edit, Mapping) or not isinstance(span := edit.get("source_span"), Mapping):
            return False
        source_name = edit.get("source_path")
        if source_name is not None and (not isinstance(source_name, str) or not source_name):
            return False
        payload = edit.get("payload")
        if edit.get("operator") == "replace_fragment":
            if (type(span.get("start")) is not int or type(span.get("end")) is not int
                    or span["end"] < span["start"]
                    or type(span.get("start_col")) is not int
                    or type(span.get("end_col")) is not int
                    or not isinstance(payload, Mapping)
                    or not isinstance(payload.get("lines"), list)
                    or len(payload["lines"]) != span["end"] - span["start"] + 1
                    or any(not isinstance(item, str) for item in payload["lines"])):
                return False
        elif (edit.get("operator") != "replace" or not isinstance(payload, Mapping)
              or span.get("start") != span.get("end")
              or type(span.get("start")) is not int
              or type(span.get("start_col")) is not int
              or type(span.get("end_col")) is not int
              or not isinstance(payload.get("mnemonic"), str)
              or not isinstance(payload.get("operands"), (list, tuple))
              or any(not isinstance(item, str) for item in payload["operands"])):
            return False
        ordered.append(edit)
    groups: dict[str, list[Mapping[str, object]]] = {}
    for edit in ordered:
        groups.setdefault(str(edit.get("source_path") or ""), []).append(edit)
    for source_name, edits in groups.items():
        before_lines = file_lines(before, source_name or None)
        after_lines = file_lines(after, source_name or None)
        if before_lines is None or after_lines is None:
            return False
        edits.sort(key=lambda edit: (
            edit["source_span"]["start"], edit["source_span"]["start_col"],
        ))
        if any(
            (right["source_span"]["start"], right["source_span"]["start_col"])
            < (left["source_span"]["end"], left["source_span"]["end_col"])
            for left, right in zip(edits, edits[1:])
        ):
            return False
        expected_lines = list(before_lines)
        for edit in reversed(edits):
            span, payload = edit["source_span"], edit["payload"]
            if edit.get("operator") == "replace_fragment":
                if not 1 <= span["start"] <= span["end"] <= len(expected_lines):
                    return False
                original_block = expected_lines[span["start"] - 1:span["end"]]
                if edit.get("source_sha256") not in (None, hashlib.sha256("".join(original_block).encode()).hexdigest()):
                    return False
                expected_lines[span["start"] - 1:span["end"]] = [
                    value + original[len(original.rstrip("\r\n")):]
                    for value, original in zip(payload["lines"], original_block)
                ]
                continue
            line_no, start_col, end_col = span["start"], span["start_col"], span["end_col"]
            if not 1 <= line_no <= len(expected_lines):
                return False
            original = expected_lines[line_no - 1]
            source_line = original.rstrip("\r\n")
            if not 0 <= start_col < end_col <= len(source_line):
                return False
            match = _INSTRUCTION_RE.fullmatch(source_line[start_col:end_col])
            if match is None:
                return False
            replacement = f"{match.group('indent')}{payload['mnemonic']} {', '.join(payload['operands'])}"
            expected_lines[line_no - 1] = (
                source_line[:start_col] + replacement + (match.group("comment") or "")
                + source_line[end_col:] + original[len(source_line):]
            )
        if expected_lines != after_lines:
            return False
    return True


def _apply_program_hint(
    program: object, hint: Mapping[str, object], output_path: Path,
    candidates: Sequence[Mapping[str, object]],
) -> object:
    return rewrite_program(program, hint, output_path, candidates=candidates)


def _program_profile_realizes(
    program: object, _profile: object, hint: Mapping[str, object],
) -> bool:
    return _campaign_contract.profile_realizes(program, hint)


def _program_route() -> CampaignRoute:
    def enumerate_program_rewrites(program: object) -> tuple[dict[str, object], ...]:
        return enumerate_case_rewrites(program)

    return CampaignRoute(
        "program", enumerate_program_rewrites, _apply_program_hint,
        _rewrite_realizes, build_profile, profile_actions, apply_profile,
        _program_profile_realizes,
    )


def _target_result_stage(result: Mapping[str, object], probe_mode: bool) -> tuple[str, bool]:
    """Count a Target attempt only when the pair result contains a Target result."""
    status = result.get("status")
    status = status if isinstance(status, str) else "target-gap"
    if result.get("target") is None:
        return status, False
    if status in {"case-skipped", "artifact-gap"}:
        return status, True
    if status in {"clean", "target-mismatch-candidate"} \
            and result.get("comparison_qualified") is not True:
        return ("target-probe-gap" if probe_mode else "target-gap"), True
    if status in {"clean", "target-mismatch-candidate", "target-tested"}:
        return ("target-probe-tested" if probe_mode else status), True
    return ("target-probe-gap" if probe_mode else "target-gap"), True


def _target_execution_completed(observations: object) -> bool:
    """True when Target observations prove a completed guest run."""
    rows = tuple(observations) if isinstance(observations, (list, tuple)) else ()
    return bool(rows) and all(
        isinstance(outcome := _object_field(item, "outcome"), str)
        and outcome not in {"timeout", "unavailable", "runner-contract-gap"}
        and not _object_field(item, "contract_error")
        and not _object_field(item, "validation_gap")
        for item in rows
    )


def _comparison_has_reference_gap(comparison: object) -> bool:
    if not isinstance(comparison, Mapping):
        return False
    if comparison.get("status") == "reference-gap":
        return True
    validation_gap = comparison.get("validation_gap")
    return (
        isinstance(validation_gap, Mapping)
        and validation_gap.get("status") == "reference-gap"
    )


def _target_event_fields(target_record: Mapping[str, object]) -> dict[str, object]:
    """Project only the Target verdict needed to audit a chain event."""
    if not isinstance(target_record, Mapping):
        return {}
    fields: dict[str, object] = {}
    if target_record.get("target_attempted") is False:
        fields["target_attempted"] = False
    result_status = target_record.get("target_result_status")
    if isinstance(result_status, str) and result_status:
        fields["target_result_status"] = result_status
    failure_class = target_record.get("failure_class")
    if isinstance(failure_class, str) and failure_class:
        fields["target_failure_class"] = failure_class
    comparison = target_record.get("comparison")
    if target_record.get("status") in {
        "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
    }:
        observation_reason = _target_observation_reason(comparison)
        generic = {
            "target-contract-gap", "target-runner-gap",
            "target-gap", "target-probe-gap", "case-skipped", "artifact-gap", "reference-gap",
        }
        candidates = [
            value for value in (failure_class, observation_reason, result_status)
            if isinstance(value, str) and value
        ]
        concrete = next(
            (value for value in candidates if value not in generic),
            candidates[0] if candidates else "target-gap",
        )
        fields["target_gap_reason"] = concrete
        if target_record.get("deadline_censored") is True:
            fields["target_deadline_censored"] = True
    if isinstance(comparison, Mapping):
        status = comparison.get("status")
        if isinstance(status, str) and status:
            fields["target_comparison_status"] = status
        differences = comparison.get("differences")
        if isinstance(differences, Mapping):
            fields["target_difference_keys"] = sorted(str(key) for key in differences)
        observation_summary = comparison.get("target_observation_summary")
        if isinstance(observation_summary, list):
            fields["target_observation_summary"] = observation_summary
    return fields


def _target_observation_reason(comparison: Mapping[str, object] | None) -> str | None:
    if not isinstance(comparison, Mapping):
        return None
    summary = comparison.get("target_observation_summary")
    if not isinstance(summary, list):
        return None
    for row in summary:
        if not isinstance(row, Mapping):
            continue
        for name in ("reason", "trace_reason", "contract_error"):
            value = row.get(name)
            if isinstance(value, str) and value:
                return value
    return None


def _generation_gap_result(
    *, work_dir: str | Path, root_binding: Mapping[str, object], status: str,
    reason: str, seed: int, beta: float, mcmc: bool,
) -> dict[str, object]:
    root = Path(work_dir).resolve()
    binding = dict(root_binding)
    evidence_path = binding.get("evidence_path")
    path = Path(evidence_path) if isinstance(evidence_path, str) else root / "candidate-ledger.json"
    path = path if path.is_absolute() else root / path
    counts = {**_EMPTY_COUNTS, "transport_gap": int(status == "transport-gap")}
    root_record = {
        "status": status, "failure_class": status,
        "reason": reason, "mode": "model-off", "raw_sha256": None,
        "rewrite_sha256": None, "root_sha256": None, "root_digest": None,
        "candidate_pool_digest": None, "candidate_count": 0, "candidate_pool": None, "hint": None,
        "supply": None, "i0": {},
        "failure": {"status": status, "reason": reason},
        "boundary_witness": {"status": status, "realizes": False, "reason": reason},
        "aseed": {"status": "not-started"},
    }
    payload = {
        "schema": "program-candidate-ledger-v1", "original_source": None,
        "original_sha256": None, "raw_sha256": None, "root": root_record,
        "original_run_params": {}, "search_mode": "mcmc" if mcmc else "direct",
        **({"root_binding": binding} if binding else {}),
        "root_recipe_digest": binding.get("root_recipe_digest"),
        "generation_profile_digest": binding.get("generation_profile_digest"),
        "generation_identity_digest": binding.get("generation_identity_digest"),
        "beta": beta, "seed": seed,
        "rng_scheme": _RNG_SCHEME,
        "mcmc_feedback_source": binding.get("mcmc_feedback_source", "target"),
        "mcmc_chain_mode": (
            "reference-only"
            if binding.get("mcmc_feedback_source") == "reference"
            else "target-coupled"
        ),
        "target_execution_mode": "reference-only" if binding.get(
            "mcmc_feedback_source"
        ) == "reference" else "synchronous",
        "target_replay": {
            "mode": "not-requested",
            "pending_pair_count": 0,
            "record_count": 0,
        },
        "reference_baseline": None, "initial_profile": None,
        "references": [], "target_records": [], "events": [],
        "rule_evidence": [], "generation_failure": {"status": status, "reason": reason},
        "counts": counts, "terminal_status": status,
    }
    atomic_write_json(path, payload, compact=True)
    return {
        "status": status, "original_sha256": None, "raw_sha256": None,
        "root_sha256": None, "root_digest": None, "root": root_record,
        **({"root_binding": binding} if binding else {}),
        "profile": {"status": "not-started"}, "reason": reason,
        "seed": seed, "rng_scheme": _RNG_SCHEME, "beta": beta,
        "mcmc": mcmc, "search_mode": "mcmc" if mcmc else "direct",
        "mcmc_feedback_source": binding.get("mcmc_feedback_source", "target"),
        "mcmc_chain_mode": (
            "reference-only"
            if binding.get("mcmc_feedback_source") == "reference"
            else "target-coupled"
        ),
        "target_execution_mode": "reference-only" if binding.get(
            "mcmc_feedback_source"
        ) == "reference" else "synchronous",
        "target_replay": {
            "mode": "not-requested",
            "pending_pair_count": 0,
            "record_count": 0,
        },
        "chain": None, "reference": [], "reference_baseline": None,
        "target": [], "counts": counts, "ledger_path": str(path),
    }


def _target_timeout_budget(
    campaign_deadline: float | None, target_timeout_seconds: float | None,
) -> float | None:
    remaining = (
        None if campaign_deadline is None
        else max(0.0, campaign_deadline - time.monotonic())
    )
    if target_timeout_seconds is None:
        return remaining
    return min(
        float(target_timeout_seconds),
        remaining if remaining is not None else float(target_timeout_seconds),
    )


def run_program_campaign(
    original: Program,
    builder: BuilderCallback,
    reference_runner: Callable[[BuiltProgram], object],
    target_runner: Callable[[BuiltProgram], object] | None,
    *,
    work_dir: str | Path,
    steps: int = FRAMEWORK_CANONICAL_STEPS,
    seed: int = 1,
    beta: float = 1.0,
    mcmc: bool = True,
    emi_enabled: bool = True,
    rewrite_hint: object | None = None,
    root_binding: Mapping[str, object] | None = None,
    target_replay_runners: Mapping[str, Callable[[BuiltProgram], object]] | None = None,
    target_bindings: Mapping[str, Mapping[str, object]] | None = None,
    target_job_configs: Mapping[str, Mapping[str, object]] | None = None,
    target_ids: Sequence[str] | None = None,
    route: CampaignRoute | None = None,
    candidate_pool: Sequence[Mapping[str, object]] | None = None,
    reference_fallback_evidence: Mapping[str, object] | None = None,
    max_seconds: float | None = None,
    target_timeout_seconds: float | None = None,
    target_replay_budget_seconds: float | None = None,
    reference_path_reward_version: str = "path-v2",
    bare_builder: BuilderCallback | None = None,
    open_capability: bool = False,
    chain_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """运行 reference-driven MCMC，并按模式发布或兼容回放 Target jobs。"""
    route = _program_route() if route is None else route
    if route.name not in {"single", "program"}:
        raise ValueError("unknown campaign route")
    if (
        type(seed) is not int or type(steps) is not int or steps < 0
        or type(mcmc) is not bool or type(emi_enabled) is not bool
    ):
        raise ValueError("invalid MCMC arguments")
    if not emi_enabled and mcmc:
        raise ValueError("MCMC requires EMI proposals; disable both modules together")
    _validate_beta(beta)
    if reference_path_reward_version not in {"pc-edge-v1", "path-v2"}:
        raise ValueError("unknown reference path reward version")
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if chain_checkpoint_path is not None:
        chain_checkpoint_path = Path(chain_checkpoint_path).resolve()
    campaign_deadline = (
        None if max_seconds is None
        else time.monotonic() + max(0.0, float(max_seconds))
    )
    target_replay_started = False
    target_replay_deadline: float | None = None

    def active_target_deadline() -> float | None:
        return target_replay_deadline if target_replay_started else campaign_deadline

    if target_timeout_seconds is not None:
        if isinstance(target_timeout_seconds, bool):
            raise ValueError("target_timeout_seconds must be non-negative")
        try:
            target_timeout_value = float(target_timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("target_timeout_seconds must be non-negative") from error
        if not math.isfinite(target_timeout_value) or target_timeout_value < 0:
            raise ValueError("target_timeout_seconds must be non-negative")
    if target_replay_budget_seconds is not None:
        if isinstance(target_replay_budget_seconds, bool):
            raise ValueError("target_replay_budget_seconds must be non-negative")
        try:
            target_replay_budget_value = float(target_replay_budget_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("target_replay_budget_seconds must be non-negative") from error
        if not math.isfinite(target_replay_budget_value) or target_replay_budget_value < 0:
            raise ValueError("target_replay_budget_seconds must be non-negative")
    else:
        target_replay_budget_value = None
    target_queue_requested = os.environ.get("RQ1_TARGET_QUEUE_ONLY", "0") == "1"
    requested_target_ids = tuple(target_ids or ())
    if any(
        not isinstance(target_id, str) or not target_id.strip()
        for target_id in requested_target_ids
    ) or len(set(requested_target_ids)) != len(requested_target_ids):
        raise ValueError("target_ids must contain unique non-empty strings")
    target_replay_runners = dict(target_replay_runners or {})
    target_bindings = dict(target_bindings or {})
    target_job_configs = {
        str(target_id): dict(value)
        for target_id, value in (target_job_configs or {}).items()
        if isinstance(target_id, str) and isinstance(value, Mapping)
    }
    if target_runner is not None and target_replay_runners:
        raise ValueError("target_runner and target_replay_runners are mutually exclusive")
    if any(
        not isinstance(name, str) or not name.strip() or not callable(runner)
        for name, runner in target_replay_runners.items()
    ):
        raise ValueError("target_replay_runners must map target ids to callables")
    if target_replay_runners and set(target_bindings) != set(target_replay_runners):
        raise ValueError("target_bindings must match target_replay_runners")
    queue_target_ids = requested_target_ids or tuple(target_replay_runners)
    if target_queue_requested and target_bindings and set(target_bindings) != set(queue_target_ids):
        raise ValueError("target_bindings must match queue target_ids")
    if target_job_configs and set(target_job_configs) != set(
        target_replay_runners or queue_target_ids or ({
            str(binding.get("target_id") or binding.get("target") or "target")
            if isinstance(binding, Mapping) else "target"
        } if target_runner is not None else ())
    ):
        raise ValueError("target_job_configs must match target replay targets")
    if any(not isinstance(value, Mapping) for value in target_bindings.values()):
        raise ValueError("target_bindings values must be objects")
    has_target_replay = (
        target_runner is not None
        or bool(target_replay_runners)
        or (target_queue_requested and bool(queue_target_ids))
    )
    def accepts_campaign_deadline(runner: Callable[..., object]) -> bool:
        try:
            parameters = inspect.signature(runner).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "campaign_deadline_monotonic"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    if target_runner is not None and (
        campaign_deadline is not None or target_timeout_seconds is not None
        or target_replay_budget_value is not None
    ):
        unbounded_target_runner = target_runner
        target_deadline_aware = accepts_campaign_deadline(unbounded_target_runner)

        def budgeted_target_runner(artifact: BuiltProgram) -> object:
            deadline = active_target_deadline()
            remaining = _target_timeout_budget(
                deadline, target_timeout_seconds,
            )
            if target_deadline_aware:
                result = unbounded_target_runner(
                    artifact,
                    timeout_seconds=target_timeout_seconds,
                    campaign_deadline_monotonic=deadline,
                )
            else:
                result = unbounded_target_runner(artifact, timeout_seconds=remaining)
            if deadline is not None and time.monotonic() >= deadline:
                return _mark_campaign_deadline_result(result)
            return result

        target_runner = budgeted_target_runner
    if target_replay_runners and (
        campaign_deadline is not None or target_timeout_seconds is not None
        or target_replay_budget_value is not None
    ):
        def budgeted_runner(
            runner: Callable[[BuiltProgram], object],
        ) -> Callable[[BuiltProgram], object]:
            deadline_aware = accepts_campaign_deadline(runner)

            def run(artifact: BuiltProgram):
                deadline = active_target_deadline()
                remaining = _target_timeout_budget(
                    deadline, target_timeout_seconds,
                )
                if deadline_aware:
                    result = runner(
                        artifact,
                        timeout_seconds=target_timeout_seconds,
                        campaign_deadline_monotonic=deadline,
                    )
                else:
                    result = runner(artifact, timeout_seconds=remaining)
                if deadline is not None and time.monotonic() >= deadline:
                    return _mark_campaign_deadline_result(result)
                return result
            return run

        target_replay_runners = {
            target_id: budgeted_runner(runner)
            for target_id, runner in target_replay_runners.items()
        }
    raw_program = original
    try:
        raw_source = _program_source(raw_program)
        raw_sha256 = program_sha256(raw_program)
    except (OSError, UnicodeError, ValueError) as error:
        return _generation_gap_result(
            work_dir=root,
            root_binding=root_binding if isinstance(root_binding, Mapping) else {},
            status="generation-gap",
            reason=f"source-invalid:{error}",
            seed=seed,
            beta=beta,
            mcmc=mcmc,
        )
    source_path = getattr(raw_program, "program", None)
    if (
        source_path is not None
        and getattr(raw_program, "run_params", {}).get("harness") == "custom"
        and re.search(r"(?m)^\s*(?:init|h0_start):\s*$", raw_source)
        and all(
            re.search(rf"(?m)^\s*{name}:\s*$", raw_source)
            for name in ("test_done", "write_tohost")
        )
    ):
        raw_program = CaseProgram(
            Path(source_path), {**dict(getattr(raw_program, "run_params", {})), "harness": "riscv-dv"},
        )
        original = raw_program
    if isinstance(raw_program, ProgramVariant) and route.name == "program" and rewrite_hint is not None:
        source_path = root / "generation" / "input" / "program.S"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(raw_program.source.encode("utf-8"))
        for name, data in raw_program._include_context:
            target = source_path.parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        raw_program = original = CaseProgram(source_path, raw_program.run_params)
    try:
        if not emi_enabled and rewrite_hint is None:
            candidates = ()
        else:
            direct_case = (
                open_capability and not mcmc and steps == 0 and rewrite_hint is None
            )
            if direct_case:
                candidates = ()
            elif candidate_pool is not None:
                candidates = tuple(candidate_pool)
            else:
                candidates = (
                    route.enumerate_rewrites(raw_program)
                    if route.name == "single"
                    or source_path is not None and Path(source_path).suffix.lower() == ".s"
                    else ()
                )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        return _generation_gap_result(
            work_dir=Path(work_dir).resolve(),
            root_binding=root_binding if isinstance(root_binding, Mapping) else {},
            status="generation-gap",
            reason=f"candidate-enumeration:{error}",
            seed=seed,
            beta=beta,
            mcmc=mcmc,
        )
    candidate_pool_digest = canonical_digest(list(candidates))
    run_params = dict(getattr(raw_program, "run_params", {}))
    if route.name == "program" and (
        not isinstance(run_params.get("input_id"), str)
        or not run_params["input_id"].strip()
    ):
        run_params["input_id"] = "input-" + canonical_digest({
            "contract": "program-input-identity-v1",
            "program_sha256": raw_sha256,
            "isa": run_params.get("isa") or run_params.get("isa_profile"),
            "initial_state": run_params.get("initial_state"),
        })[:24]
        if isinstance(raw_program, CaseProgram):
            raw_program = original = CaseProgram(raw_program.program, run_params)
        elif isinstance(raw_program, ProgramVariant):
            raw_program = original = replace(raw_program, run_params=run_params)
    raw_includes = _source_includes(raw_program)
    actual_single_route = route.name == "single"
    declared_route = run_params.get("route")
    if declared_route not in (None, "single", "program") or (
        declared_route is not None and declared_route != route.name
    ):
        return _generation_gap_result(
            work_dir=root,
            root_binding=root_binding if isinstance(root_binding, Mapping) else {},
            status="generation-gap",
            reason="route identity is invalid",
            seed=seed,
            beta=beta,
            mcmc=mcmc,
        )
    if root_binding is not None and not isinstance(root_binding, Mapping):
        raise ValueError("root binding is invalid")
    binding = None if root_binding is None else dict(root_binding)
    required_sequence = None
    probe_options = binding is not None and (
        binding.get("target_execution_mode") == "probe"
        and isinstance(binding.get("target"), str)
        and bool(binding["target"].strip())
    )
    if binding is not None:
        methods = (*(f"B{i}" for i in range(4)), *(f"S{i}" for i in range(6)))
        rules = ("M5", *(f"R{i}" for i in range(1, 12)))
        if any(
            not isinstance(binding.get(name), str) or not binding[name].strip()
            for name in ("root_key", "lineage_key", "route", "method", "rule", "state", "observer", "target")
        ) or binding.get("method") not in methods or binding.get("rule") not in rules:
            raise ValueError("root binding is invalid")
        if binding.get("route") not in {"single", "program"} or binding["route"] != route.name:
            raise ValueError("root binding is invalid")
    observer_fields = binding.get("observer_fields") if binding is not None else None
    if observer_fields is not None and (
        not isinstance(observer_fields, Sequence)
        or isinstance(observer_fields, (str, bytes))
        or any(not isinstance(item, str) or not item.strip() for item in observer_fields)
    ):
        raise ValueError("root binding is invalid")
    if observer_fields is not None:
        observer_fields = tuple(
            field for field in observer_fields
            if not _is_observation_side_channel_field(field)
        )
        if binding is not None:
            binding["observer_fields"] = list(observer_fields)
    comparison_fields = binding.get("comparison_fields") if binding is not None else None
    if comparison_fields is not None and (
        not isinstance(comparison_fields, Sequence)
        or isinstance(comparison_fields, (str, bytes))
        or any(not isinstance(item, str) or not item.strip() for item in comparison_fields)
    ):
        raise ValueError("root binding is invalid")
    if comparison_fields is None:
        comparison_fields = observer_fields
    if comparison_fields is not None:
        comparison_fields = tuple(
            field for field in comparison_fields
            if not _is_observation_side_channel_field(field)
        )
        if binding is not None:
            binding["comparison_fields"] = list(comparison_fields)
    if binding is not None and not probe_options:
        methods = (*(f"B{i}" for i in range(4)), *(f"S{i}" for i in range(6)))
        rules = ("M5", *(f"R{i}" for i in range(1, 12)))
        if any(
            not isinstance(binding.get(name), str) or not binding[name].strip()
            for name in ("root_key", "lineage_key", "route", "method", "rule", "state", "observer", "target")
        ) or binding["method"] not in methods or binding["rule"] not in rules:
            raise ValueError("root binding is invalid")
        if binding["route"] not in {"single", "program"}:
            raise ValueError("root binding is invalid")
        if binding["route"] != route.name:
            raise ValueError("root binding is invalid")
        if "evidence_path" in binding and (
            not isinstance(binding["evidence_path"], str) or not binding["evidence_path"].strip()
        ):
            raise ValueError("root binding is invalid")
        if any(
            name in binding and not is_sha256_digest(binding[name])
            for name in (
                "root_recipe_digest", "generation_profile_digest",
                "generation_identity_digest",
            )
        ):
            raise ValueError("root binding is invalid")
        binding.setdefault("evidence_path", str((root / "candidate-ledger.json").resolve()))
        evidence_path = Path(binding["evidence_path"]).resolve()
        raw_path = getattr(raw_program, "program", None)
        raw_path = Path(raw_path).resolve() if raw_path is not None else None
        source_paths = {path for path in (raw_path, root / "generation" / "program.S") if path is not None}
        if raw_path is not None:
            source_paths.update((raw_path.parent / name).resolve() for name in raw_includes)
        if evidence_path in source_paths:
            raise ValueError("root binding is invalid")
        required_sequence = binding.get("required_sequence")
        if (required_sequence is None
                and binding.get("root_recipe_digest") is not None
                and binding.get("sequence_mode") != "dynamic"):
            raise ValueError("root binding required sequence is missing")
    if required_sequence is not None and (
        not isinstance(required_sequence, Sequence)
        or isinstance(required_sequence, (str, bytes))
        or not required_sequence
        or any(not isinstance(item, str) or not item.strip() for item in required_sequence)
    ):
        raise ValueError("required sequence is invalid")
    if binding is not None and required_sequence is not None:
        binding["required_sequence"] = list(required_sequence)
    if binding is not None:
        semantic_witness = binding.get("semantic_witness")
        contract = semantic_witness if isinstance(semantic_witness, Mapping) else (
            _semantic_witness_contract(tuple(required_sequence or ()), semantic_witness)
            if isinstance(semantic_witness, str) else None
        )
        if contract is not None:
            binding["semantic_witness_contract"] = dict(contract)
        if not probe_options and not open_capability:
            admission_gap = route_admission_gap(
                binding, route.name, binding.get("reference_backend"),
                binding.get("target") if target_runner is not None else "unavailable",
                run_params.get("isa") or run_params.get("isa_profile"),
                binding.get("harness") or run_params.get("harness"),
                observer_fields or (),
            )
            if admission_gap is not None:
                return _generation_gap_result(
                    work_dir=root, root_binding=binding, status="transport-gap",
                    reason=f"route-admission:{admission_gap}",
                    seed=seed, beta=beta, mcmc=mcmc,
                )
    rewrite_record: dict[str, object] = {
        "status": "not-applied" if rewrite_hint is None else "pending",
        "mode": "model-off" if rewrite_hint is None else "model-on",
        "raw_sha256": raw_sha256,
        "rewrite_sha256": raw_sha256 if rewrite_hint is None else None,
        "root_sha256": raw_sha256 if rewrite_hint is None else None,
        "root_digest": None,
        "i0_digest": canonical_digest(run_params) if rewrite_hint is None else None,
        "reference_observation_digest": None,
        "candidate_pool_digest": candidate_pool_digest,
        "candidate_count": len(candidates),
        "hint": None,
        "supply": None,
        "i0": dict(run_params),
        "boundary_witness": {
            "status": "pending",
            "realizes": False,
            "candidate_pool_digest": candidate_pool_digest,
        },
        "aseed": {"status": "not-evaluated"},
    }
    if rewrite_hint is None:
        rewrite_record["root_digest"] = canonical_digest({
            "raw_sha256": raw_sha256, "rewrite_sha256": raw_sha256,
            "hint": None, "run_params": run_params,
            "i0_digest": canonical_digest(run_params),
            "reference_observation_digest": None,
            "reference_identity": None,
        })
    if rewrite_hint is not None:
        try:
            supplied = supply_case_rewrites(
                rewrite_hint,
                {"raw_sha256": raw_sha256, "candidate_pool_digest": candidate_pool_digest},
                candidates=candidates,
            )
            rewrite_record["supply"] = supplied
            if supplied["status"] != "accepted":
                reason = str(supplied.get("rejection_reason", ""))
                generation_gap = reason.startswith(("schema-error:", "action-reject:"))
                prefix = "generation-gap" if generation_gap else "seed-miss"
                raise ValueError(f"{prefix}: expected exactly one accepted CaseRewriteHint")
            selected = dict(supplied["canonical_hint"])
            original = route.apply_hint(
                raw_program, selected, root / "generation" / "program.S", candidates,
            )
            if not route.rewrite_realizes(raw_program, original, selected):
                raise ValueError("seed-miss: rewrite witness not realized")
            rewrite_sha256 = program_sha256(original)
            root_digest = canonical_digest({
                "raw_sha256": raw_sha256, "rewrite_sha256": rewrite_sha256,
                "hint": selected, "run_params": run_params,
            })
            rewrite_record.update(
                status="generated", rewrite_sha256=rewrite_sha256,
                root_sha256=rewrite_sha256, root_digest=root_digest, hint=selected,
                boundary_witness={
                    "status": "realized", "realizes": True, "hint": selected,
                    "candidate": {"edits": selected["edits"]},
                    "candidate_pool_digest": candidate_pool_digest,
                },
            )
        except Exception as error:
            detail = str(error)
            status = "seed-miss" if detail.startswith("seed-miss:") else "generation-gap"
            reason = detail if detail.startswith(f"{status}:") else f"{status}:{detail}"
            rewrite_record.update(
                status=status, root_sha256=None, root_digest=None,
                reason=reason, boundary_witness={
                    "status": status, "realizes": False,
                    "candidate_pool_digest": candidate_pool_digest,
                    "reason": reason,
                },
                aseed={"status": "not-started" if status == "generation-gap" else "not-evaluated"},
            )
    if binding is not None and isinstance(binding.get("semantic_witness"), (str, Mapping)):
        rewrite_record["boundary_witness"]["semantic_witness"] = (
            dict(binding["semantic_witness"])
            if isinstance(binding["semantic_witness"], Mapping)
            else binding["semantic_witness"]
        )
    if binding is not None and isinstance(binding.get("semantic_witness_contract"), Mapping):
        rewrite_record["boundary_witness"]["semantic_witness_contract"] = dict(
            binding["semantic_witness_contract"]
        )
    original_sha256 = raw_sha256 if original is raw_program else program_sha256(original)
    original_source = raw_source if original is raw_program else _program_source(original)
    original_includes = _source_includes(original)
    original_run_params = dict(getattr(original, "run_params", {}))
    baseline: BuiltProgram | None = None
    baseline_record: dict[str, object] | None = None
    reference_attempts: list[dict[str, object]] = []
    initial_profile_record: dict[str, object] | None = None
    references: list[dict[str, object]] = []
    reference_cache: dict[tuple[str, str], int] = {}
    artifact_cache: dict[tuple[object, ...], BuiltProgram] = {}
    target_records: list[dict[str, object]] = []
    target_records_by_target: dict[str, list[dict[str, object]]] = {
        target_id: [] for target_id in target_replay_runners
    }
    target_baseline: dict[str, object] | None = None
    target_baselines_by_target: dict[str, dict[str, object]] = {}
    target_baseline_compared = False
    target_feedback_scores: dict[str, float | None] = {}
    reference_feedback_scores: dict[str, float | None] = {}
    reference_feedback_records: dict[str, dict[str, object]] = {}
    reference_feedback_mode = (
        isinstance(binding, Mapping)
        and binding.get("mcmc_feedback_source") == "reference"
    )
    target_queue_only = target_queue_requested and reference_feedback_mode and has_target_replay
    reference_backend_label = (
        str(binding.get("reference_backend"))
        if isinstance(binding, Mapping)
        and isinstance(binding.get("reference_backend"), str)
        else "reference"
    )
    deferred_target_baseline: tuple[BuiltProgram, Mapping[str, object], bool] | None = None
    pending_target_pairs: list[dict[str, object]] = []
    target_queue_path = (
        root / "target-replay-queue.json"
        if reference_feedback_mode and has_target_replay else None
    )
    target_queue_root = root / "target-replay-queue"
    target_dispatch_root = None
    dispatch_root_value = os.environ.get("RQ1_TARGET_QUEUE_ROOT", "").strip()
    if target_queue_only:
        if not dispatch_root_value:
            raise RuntimeError(
                "formal Target queue mode requires RQ1_TARGET_QUEUE_ROOT"
            )
        target_dispatch_root = Path(dispatch_root_value).resolve()
    queued_target_ids = list(queue_target_ids)
    if not queued_target_ids and target_runner is not None:
        queued_target_ids = [
            str(binding.get("target_id") or binding.get("target") or "target")
            if isinstance(binding, Mapping) else "target"
        ]
    target_result_slots = {
        target_id: index for index, target_id in enumerate(queued_target_ids)
    }
    target_progress_sequence = 0
    # The campaign function is the sole producer.  Target consumers never
    # mutate these objects or the producer manifest, so there is no
    # framework/simulator lock to contend on.
    target_queue_baseline: dict[str, object] | None = None
    target_queue_jobs: list[dict[str, object]] = []
    target_queue_jobs_by_id: dict[str, dict[str, object]] = {}
    target_publish_gaps: list[dict[str, object]] = []
    target_replay_deadline_epoch: float | None = None
    target_progress_counts: dict[str, int] | None = None
    target_progress_summaries: dict[str, dict[str, object]] = {}

    def release_pending_target_payload(pending_target: Mapping[str, object]) -> None:
        """Drop replay-only objects after their durable result is written."""
        if isinstance(pending_target, dict):
            for key in ("base", "candidate", "reference_observations"):
                pending_target.pop(key, None)

    def queue_program_payload(program: object) -> dict[str, object]:
        include_context = getattr(program, "_include_context", ())
        if isinstance(include_context, (list, tuple)) and include_context:
            includes = [
                {"name": name, "content_hex": data.hex()}
                for name, data in include_context
                if isinstance(name, str) and isinstance(data, bytes)
            ]
        else:
            includes = [
                {"name": name, "content": value}
                for name, value in _source_includes(program).items()
            ]
        return {
            "source_sha256": program_sha256(program),
            "source": _program_source(program),
            "run_params": dict(getattr(program, "run_params", {}) or {}),
            "action_json": getattr(program, "action_json", "{}"),
            "include_context": includes,
        }

    def persist_target_queue(stage: str) -> None:
        if target_queue_path is None:
            return
        jobs = []
        for item in target_queue_jobs:
            job = {
                "job_id": item["job_id"],
                "job_path": item["job_path"],
                "pair_index": item["pair_index"],
                "reference_index": item["reference_index"],
                "state_sha256": item["state_sha256"],
                "parent_sha256": item["parent_sha256"],
                "step": item.get("step"),
                "reference_event_id": item.get("reference_event_id"),
                "publish_stage": item.get("publish_stage"),
                "enqueued_at_epoch": item.get("enqueued_at_epoch"),
            }
            if not target_queue_only:
                job["target_states"] = dict(item["target_states"])
            jobs.append(job)
        baseline = None
        if target_queue_baseline is not None:
            baseline = {
                "job_path": target_queue_baseline["job_path"],
                "reference_event_id": target_queue_baseline.get(
                    "reference_event_id"
                ),
                "publish_stage": target_queue_baseline.get("publish_stage"),
                "enqueued_at_epoch": target_queue_baseline.get(
                    "enqueued_at_epoch"
                ),
            }
            if not target_queue_only:
                baseline["target_states"] = dict(
                    target_queue_baseline["target_states"]
                )
        manifest = {
            "schema_version": "rq1-target-replay-queue-v1",
            "status": stage,
            "root_sha256": original_sha256,
            "seed": seed,
            "chain_step_budget": steps,
            "target_replay_budget_seconds": (
                None if target_queue_only else target_replay_budget_value
            ),
            "target_replay_deadline_epoch": target_replay_deadline_epoch,
            "target_queue_deadline_policy": (
                "per-target-timeout-only" if target_queue_only
                else "campaign-target-replay-budget"
            ),
            "reference_path_reward_version": reference_path_reward_version,
            "target_ids": list(queued_target_ids),
            "target_queue_ownership": "producer-plan",
            "target_completion_source": "per-target-result-sidecar",
            "baseline": baseline,
            "jobs": jobs,
        }
        if not target_queue_only:
            manifest["pending_execution_count"] = target_queue_pending_count()
        atomic_write_json(target_queue_path, _json_safe(manifest))

    def _record_target_publish_gap(
        target_id: str, job_id: str, error: BaseException,
    ) -> None:
        record = {
            "schema_version": "rq1-target-publish-gap-v1",
            "target_id": target_id,
            "job_id": job_id,
            "failure_class": "target-publish-gap",
            "reason": f"{type(error).__name__}: {error}"[:1000],
            "recorded_at_epoch": time.time(),
        }
        target_publish_gaps.append(record)
        target_root = target_dispatch_root / _safe(target_id) \
            if target_dispatch_root is not None else None
        try:
            if target_root is None:
                raise OSError("Target queue root is unavailable")
            target_root.mkdir(parents=True, exist_ok=True)
            lock_path = target_root / ".append.lock"
            gap_path = target_root / "publish-gaps.jsonl"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    with gap_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(
                            _json_safe(record), ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"),
                        ) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except (OSError, TypeError, ValueError):
            # The in-memory campaign record is still returned. A broken
            # Target queue must not turn a valid reference/MCMC result into a
            # producer failure.
            pass

    def _publish_target_job(
        *, job_id: str, job_path: Path, enqueued_at_epoch: float,
        target_ids: Sequence[str],
    ) -> None:
        """Append one immutable entry to each Target's durable queue.

        The append lock only serializes concurrent producer writes to one log
        record. It is not held while a Target runs, and it is not an admission
        gate. Consumers advance their own byte cursor after the result sidecar
        is durable.
        """
        if target_dispatch_root is None or not queued_target_ids:
            return
        try:
            method_root = target_dispatch_root.parent.resolve()
            case_root = root.resolve().relative_to(method_root)
        except ValueError:
            raise RuntimeError(
                "RQ1_TARGET_QUEUE_ROOT must be the parent method directory's target-queues"
            ) from None
        for target_id in target_ids:
            publish_started = time.monotonic()
            slot = target_result_slots[target_id]
            job_meta = (
                target_queue_jobs_by_id.get(job_id)
                if job_id != "baseline" else target_queue_baseline
            ) or {}
            target_root = target_dispatch_root / "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in str(target_id)
            )
            target_root.mkdir(parents=True, exist_ok=True)
            result_path = (
                root / "target-replay-queue" / "results"
                / f"target-{slot:03d}" / f"{job_id}.json"
            )
            entry = {
                "schema_version": "rq1-target-queue-entry-v1",
                "target_id": target_id,
                "target_slot": slot,
                "job_id": job_id,
                "case_root": str(case_root),
                "job_path": str(job_path.relative_to(root)),
                "queue_manifest_path": str(
                    target_queue_path.relative_to(method_root)
                ) if target_queue_path is not None else None,
                "result_path": str(result_path.relative_to(method_root)),
                "pair_index": job_meta.get("pair_index"),
                "reference_index": job_meta.get("reference_index"),
                "state_sha256": job_meta.get("state_sha256"),
                "parent_sha256": job_meta.get("parent_sha256"),
                "step": job_meta.get("step"),
                "reference_event_id": job_meta.get("reference_event_id"),
                "publish_stage": job_meta.get("publish_stage"),
                "enqueued_at_epoch": enqueued_at_epoch,
                "target_replay_deadline_epoch": target_replay_deadline_epoch,
            }
            payload = (
                json.dumps(
                    _json_safe(entry), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ) + "\n"
            ).encode("utf-8")
            lock_path = target_root / ".append.lock"
            queue_path = target_root / "queue.jsonl"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    append_started = time.monotonic()
                    fd = os.open(
                        queue_path,
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                        0o644,
                    )
                    try:
                        view = memoryview(payload)
                        while view:
                            view = view[os.write(fd, view):]
                        # The consumer is an independent process.  Do not make
                        # a queue entry visible before its bytes survive a
                        # producer crash or a container restart.
                        fsync_started = time.monotonic()
                        os.fsync(fd)
                        fsync_seconds = max(0.0, time.monotonic() - fsync_started)
                    finally:
                        os.close(fd)
                    append_seconds = max(0.0, time.monotonic() - append_started)
                    state_path = target_root / "state.json"
                    try:
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                    except (OSError, TypeError, ValueError):
                        state = {}
                    published_count = state.get("published_count", 0)
                    try:
                        published_count = max(0, int(published_count)) + 1
                    except (TypeError, ValueError, OverflowError):
                        published_count = 1
                    previous_timing = state.get("publish_timing")
                    previous_timing = (
                        previous_timing if isinstance(previous_timing, Mapping) else {}
                    )
                    state_write_started = time.monotonic()
                    atomic_write_json(state_path, {
                        "schema_version": "rq1-target-queue-state-v1",
                        "target_id": target_id,
                        "published_count": published_count,
                        "last_job_id": job_id,
                        "updated_at_epoch": time.time(),
                        "publish_timing": {
                            "last_seconds": round(
                                max(0.0, time.monotonic() - publish_started), 6,
                            ),
                            "last_append_seconds": round(append_seconds, 6),
                            "last_fsync_seconds": round(fsync_seconds, 6),
                            "last_state_write_seconds": round(
                                max(0.0, time.monotonic() - state_write_started), 6,
                            ),
                            "total_seconds": round(
                                float(previous_timing.get("total_seconds", 0.0) or 0.0)
                                + max(0.0, time.monotonic() - publish_started),
                                6,
                            ),
                            "total_fsync_seconds": round(
                                float(previous_timing.get("total_fsync_seconds", 0.0) or 0.0)
                                + fsync_seconds,
                                6,
                            ),
                        },
                    })
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def publish_target_job(
        *, job_id: str, job_path: Path, enqueued_at_epoch: float,
    ) -> None:
        """Publish independently; a broken Target never blocks its peers."""
        for target_id in queued_target_ids:
            try:
                _publish_target_job(
                    job_id=job_id, job_path=job_path,
                    enqueued_at_epoch=enqueued_at_epoch,
                    target_ids=(target_id,),
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                _record_target_publish_gap(target_id, job_id, error)

    def target_queue_pending_count() -> int:
        if target_queue_only:
            return 0
        return sum(
            state in {"pending", "running"}
            for item in target_queue_jobs
            for state in item["target_states"].values()
        ) + sum(
            state in {"pending", "running"}
            for state in (
                target_queue_baseline["target_states"].values()
                if target_queue_baseline is not None else ()
            )
        )

    def queue_candidate_target_pair(pending_target: dict[str, object]) -> None:
        if target_queue_path is None:
            return
        pair_index = len(target_queue_jobs)
        job_id = f"candidate-{pair_index:06d}"
        job_path = target_queue_root / "jobs" / f"{job_id}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_records = pending_target.get("artifact_records")
        artifact_gap = pending_target.get("artifact_gap")
        if not isinstance(artifact_gap, str) or not artifact_gap:
            artifact_gap = _artifact_gap_reason(artifact_records)
        reference_event_id = canonical_digest({
            "job_id": job_id,
            "pair_index": pair_index,
            "reference_index": pending_target["reference_index"],
            "state_sha256": pending_target["candidate"].sha256,
            "parent_sha256": pending_target["base_sha256"],
            "reference_observations": pending_target["reference_observations"],
        })
        atomic_write_json(job_path, _json_safe({
            "schema_version": "rq1-target-replay-job-v1",
            "job_id": job_id,
            "job_type": "candidate-pair",
            "pair_index": pair_index,
            "reference_index": pending_target["reference_index"],
            "state_sha256": pending_target["candidate"].sha256,
            "parent_sha256": pending_target["base_sha256"],
            "base": queue_program_payload(pending_target["base"]),
            "candidate": queue_program_payload(pending_target["candidate"]),
            "reference_observations": pending_target["reference_observations"],
            "reference_event_id": reference_event_id,
            "publish_stage": "reference-pair",
            "artifact_records": (
                dict(artifact_records) if isinstance(artifact_records, Mapping) else {}
            ),
            **({"artifact_gap": artifact_gap} if artifact_gap else {}),
            # Formal queue consumers own their backend/session configuration.
            # Inline compatibility replay may still carry the legacy config.
            "target_configs": {} if target_queue_only else target_job_configs,
        }))
        pending_target.update(
            job_id=job_id,
            job_path=str(job_path.relative_to(root)),
            pair_index=pair_index,
            state_sha256=pending_target["candidate"].sha256,
            parent_sha256=pending_target["base_sha256"],
            step=None,
            enqueued_at_epoch=time.time(),
            reference_event_id=reference_event_id,
            publish_stage="reference-pair",
        )
        if not target_queue_only:
            pending_target["target_states"] = {
                target_id: "pending" for target_id in (
                    list(target_replay_runners) or [
                        str(binding.get("target_id") or binding.get("target") or "target")
                    ]
                )
            }
        queued_job = {
            key: pending_target[key]
            for key in (
                "job_id", "job_path", "pair_index", "reference_index",
                "state_sha256", "parent_sha256", "step",
                "enqueued_at_epoch",
                "reference_event_id", "publish_stage",
                *(('target_states',) if not target_queue_only else ()),
            )
        }
        target_queue_jobs.append(queued_job)
        target_queue_jobs_by_id[job_id] = queued_job
        publish_target_job(
            job_id=job_id, job_path=job_path,
            enqueued_at_epoch=float(pending_target["enqueued_at_epoch"]),
        )

    def queue_target_baseline(observations: object) -> None:
        nonlocal target_queue_baseline
        if target_queue_path is None:
            return
        reference_observations = (
            [_observation_record(item) for item in observations]
            if isinstance(observations, (list, tuple)) else []
        )
        job_path = target_queue_root / "jobs" / "baseline.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        reference_event_id = canonical_digest({
            "job_id": "baseline",
            "state_sha256": original_sha256,
            "reference_observations": reference_observations,
        })
        atomic_write_json(job_path, _json_safe({
            "schema_version": "rq1-target-replay-job-v1",
            "job_id": "baseline",
            "job_type": "baseline",
            "state_sha256": original_sha256,
            "program": queue_program_payload(original),
            "reference_observations": reference_observations,
            "reference_event_id": reference_event_id,
            "publish_stage": "reference-baseline",
            "artifact_record": (
                _built_artifact_record(baseline)
                if isinstance(baseline, BuiltProgram) else None
            ),
            "target_configs": {} if target_queue_only else target_job_configs,
        }))
        target_queue_baseline = {
            "job_path": str(job_path.relative_to(root)),
            "enqueued_at_epoch": time.time(),
            "reference_event_id": reference_event_id,
            "publish_stage": "reference-baseline",
        }
        publish_target_job(
            job_id="baseline", job_path=job_path,
            enqueued_at_epoch=float(target_queue_baseline["enqueued_at_epoch"]),
        )
        if not target_queue_only:
            target_queue_baseline["target_states"] = {
                target_id: "pending" for target_id in (
                    list(target_replay_runners) or [
                        str(binding.get("target_id") or binding.get("target") or "target")
                    ]
                )
            }
        persist_target_queue("chain-running")

    def queue_target_state(job_id: str, target_id: str, status: str) -> None:
        if target_queue_path is None or target_queue_only:
            return
        job = (
            target_queue_baseline
            if job_id == "baseline" else
            target_queue_jobs_by_id.get(job_id)
        )
        if job is None:
            return
        states = job.get("target_states")
        if isinstance(states, dict) and target_id in states:
            states[target_id] = status
        # Inline compatibility path only. Queue-only consumers never call this.

    target_progress_count_fields = (
        "target_attempted", "target_tested", "target_gap", "target_unsupported",
        "target_raw_recorded", "target_comparison_pending",
        "case_skipped", "target_mismatch_candidate", "target_transport_gap",
        "target_wall_clock_pending", "clean", "artifact_gap",
    )
    target_progress_status_fields = (
        "clean", "target-mismatch-candidate", "target-tested", "target-gap",
        "raw-recorded",
        "target-probe-tested", "target-probe-gap", "case-skipped", "artifact-gap",
        "transport-gap", "reference-valid", "external-signal", "other",
    )

    def empty_target_progress_summary() -> dict[str, object]:
        return {
            "record_count": 0,
            "baseline_count": 0,
            "counts": {key: 0 for key in target_progress_count_fields},
            "status_counts": {key: 0 for key in target_progress_status_fields},
        }

    def target_progress_delta(
        record: Mapping[str, object], *, baseline: bool = False,
    ) -> dict[str, int]:
        if baseline:
            values = _target_baseline_counts(record)
            return {key: int(values.get(key, 0)) for key in target_progress_count_fields}
        status = record.get("status")
        attempted = record.get("target_attempted") is not False
        raw_pending = status == "raw-recorded" and record.get("comparison_pending") is True
        coverage_only = record.get("coverage_only") is True
        expected_comparison = (
            "equivalent" if status == "clean" else
            "non-equivalent" if status == "target-mismatch-candidate" else None
        )
        comparison_qualified = (
            expected_comparison is not None
            and _comparison_qualified(record, expected_comparison)
        )
        probe_status = probe_mode and status in {
            "target-probe-tested", "target-probe-gap",
        }
        counted_attempt = attempted and not probe_status
        tested = (
            not probe_mode and not coverage_only
            and comparison_qualified
            and status in {"clean", "target-mismatch-candidate"}
        )
        coverage_only_completed = (
            coverage_only and attempted
            and status in {"clean", "target-mismatch-candidate", "target-tested"}
        )
        return {
            "target_attempted": int(counted_attempt),
            "target_tested": int(tested),
            "target_gap": int(
                counted_attempt and not tested and not coverage_only_completed
                and status != "case-skipped" and not raw_pending
            ),
            "target_raw_recorded": int(counted_attempt and raw_pending),
            "target_comparison_pending": int(counted_attempt and raw_pending),
            "target_unsupported": int(counted_attempt and status == "case-skipped"),
            "case_skipped": int(counted_attempt and status == "case-skipped"),
            "target_mismatch_candidate": int(
                not probe_mode and not coverage_only
                and comparison_qualified
                and status == "target-mismatch-candidate"
            ),
            "target_transport_gap": int(attempted and not coverage_only and (
                record.get("failure_class") == "transport-gap"
                or record.get("target_result_status") == "transport-gap"
            )),
            "target_wall_clock_pending": int(attempted and (
                record.get("deadline_censored") is True
                or record.get("failure_class") == "campaign-wall-clock-exhausted"
            )),
            "clean": int(
                not probe_mode and not coverage_only and comparison_qualified
                and status == "clean"
            ),
            "artifact_gap": int(status == "artifact-gap"),
        }

    def target_progress_status(record: Mapping[str, object]) -> str:
        status = record.get("status")
        expected_comparison = (
            "equivalent" if status == "clean" else
            "non-equivalent" if status == "target-mismatch-candidate" else None
        )
        if status in {"clean", "target-mismatch-candidate"} \
                and not _comparison_qualified(record, expected_comparison):
            return "target-gap"
        return status if status in target_progress_status_fields[:-1] else "other"

    def persist_target_replay_progress(
        counts: Mapping[str, int], sequence: int, *, rebuild: bool = False,
    ) -> None:
        nonlocal target_progress_counts, target_progress_summaries
        # Queue-only Target progress belongs to each consumer's result
        # directory. The producer publishes the plan and does not create a
        # shared cumulative Target state file.
        if target_queue_path is None or target_queue_only:
            return
        if rebuild or target_progress_counts is None:
            target_progress_counts = dict(counts)
            target_progress_summaries = {
                target_id: empty_target_progress_summary()
                for target_id in queued_target_ids
            }
            if target_replay_runners:
                progress_records = target_records_by_target
                progress_baselines = target_baselines_by_target
            else:
                target_id = queued_target_ids[0] if queued_target_ids else "target"
                progress_records = {target_id: target_records}
                progress_baselines = (
                    {target_id: target_baseline}
                    if isinstance(target_baseline, Mapping) else {}
                )
            for target_id in queued_target_ids:
                summary = target_progress_summaries.setdefault(
                    target_id, empty_target_progress_summary(),
                )
                rows = progress_records.get(target_id, ())
                for record in rows:
                    if isinstance(record, Mapping):
                        summary["record_count"] = int(summary["record_count"]) + 1
                        status_counts = summary["status_counts"]
                        status_key = target_progress_status(record)
                        status_counts[status_key] += 1
                        values = summary["counts"]
                        for key, value in target_progress_delta(record).items():
                            values[key] += value
                baseline_record = progress_baselines.get(target_id)
                if isinstance(baseline_record, Mapping):
                    summary["baseline_count"] = 1
                    status_counts = summary["status_counts"]
                    status_counts[target_progress_status(baseline_record)] += 1
                    values = summary["counts"]
                    for key, value in target_progress_delta(
                        baseline_record, baseline=True,
                    ).items():
                        values[key] += value
        atomic_write_json(root / "campaign-progress.json", _json_safe({
            "schema_version": "rq1-framework-campaign-progress-v1",
            "progress_sequence": sequence,
            "counts": dict(target_progress_counts),
            "target_summaries": target_progress_summaries,
        }), compact=True)

    def persist_target_replay_record(
        target_id: str, pair_index: int | None, record: Mapping[str, object],
        *, baseline: bool = False,
    ) -> dict[str, object] | None:
        nonlocal target_progress_sequence, target_progress_counts
        slot = target_result_slots.get(target_id)
        if target_queue_path is None or slot is None:
            return None
        if target_progress_counts is None:
            target_progress_counts = dict(_EMPTY_COUNTS)
            target_progress_summaries.update({
                name: empty_target_progress_summary()
                for name in queued_target_ids
            })
        summary = target_progress_summaries.setdefault(
            target_id, empty_target_progress_summary(),
        )
        target_progress_sequence += 1
        name = "baseline" if baseline else f"candidate-{int(pair_index):06d}"
        record_path = (
            target_queue_root / "results" / f"target-{slot:03d}" / f"{name}.json"
        )
        persisted_record = dict(record)
        if baseline and isinstance(baseline_artifact, BuiltProgram):
            artifact_record = _built_artifact_record(baseline_artifact)
            persisted_record.setdefault("target_artifact", artifact_record)
            if isinstance(record, dict):
                record.setdefault("target_artifact", artifact_record)
        delta = target_progress_delta(record, baseline=baseline)
        next_counts = dict(target_progress_counts or _EMPTY_COUNTS)
        per_target_counts = dict(summary["counts"])
        for key, value in delta.items():
            next_counts[key] = next_counts.get(key, 0) + value
            per_target_counts[key] = per_target_counts.get(key, 0) + value
        status_counts = dict(summary["status_counts"])
        status_key = target_progress_status(record)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        next_summary = {
            "record_count": int(summary["record_count"]) + int(not baseline),
            "baseline_count": int(summary["baseline_count"]) + int(baseline),
            "counts": per_target_counts,
            "status_counts": status_counts,
        }
        atomic_write_json(record_path, _json_safe({
            "schema_version": "rq1-target-replay-result-v1",
            "target_id": target_id,
            "pair_index": pair_index,
            "progress_sequence": target_progress_sequence,
            "counts": next_counts,
            "record": persisted_record,
        }), compact=True)
        evidence_ref = make_evidence_ref(
            root, record_path, target_id=target_id, pair_index=pair_index,
        )
        if _compact_target_evidence_enabled() and isinstance(record, dict):
            compact_record(record, evidence_ref)
        target_progress_counts = next_counts
        target_progress_summaries[target_id] = next_summary
        persist_target_replay_progress(next_counts, target_progress_sequence)
        return evidence_ref

    if target_queue_path is not None:
        target_queue_root.mkdir(parents=True, exist_ok=True)
        persist_target_queue("chain-running")
    probe_mode = isinstance(binding, Mapping) and binding.get("target_execution_mode") == "probe"
    single_case = run_params.get("single_case")
    single_dataflow = (
        single_case.get("dataflow_meta", {})
        if isinstance(single_case, Mapping) else {}
    )
    expected_trap = (
        expected_trap_from_dataflow(single_dataflow)
        if actual_single_route else
        isinstance(binding, Mapping) and binding.get("expected_trap") is True
    )

    def mark_coverage_only_baseline(record: Mapping[str, object]) -> dict[str, object]:
        comparison = record.get("comparison")
        observations = record.get("observations")
        if (
            not reference_feedback_mode
            or record.get("status") not in {"target-gap", "reference-gap"}
            or not _comparison_has_reference_gap(comparison)
            or not _target_execution_completed(observations)
        ):
            return dict(record)
        return {**record, "status": "target-tested", "coverage_only": True}

    def capture_target_baseline(
        baseline_artifact: BuiltProgram,
        reference_record: Mapping[str, object],
        expected_trap: bool = False,
        *, defer_target: bool | None = None,
    ) -> dict[str, object] | None:
        """Capture the Target baseline for the route's execution mode.

        Reference-driven Ours campaigns defer this side-channel run until the
        MH chain closes. Synchronous compatibility routes keep the older
        baseline behavior.
        """
        nonlocal target_baseline, target_baseline_compared, deferred_target_baseline
        if not has_target_replay:
            return target_baseline
        if target_queue_only:
            # The formal producer only publishes the baseline job.  It never
            # constructs or invokes a simulator callback; the resident Target
            # service owns execution and raw evidence collection.
            deferred_target_baseline = (
                baseline_artifact, reference_record, expected_trap,
            )
            return target_baseline
        if reference_feedback_mode and not direct_case and defer_target is not False:
            # Reference-driven Ours 的 reference/MH 链不等待 Target。基线也留到链结束后再
            # 执行；这里始终保留最新的 reference record，避免先保存 provisional
            # 空观察后覆盖真正的 K1/QEMU baseline。
            deferred_target_baseline = (
                baseline_artifact, reference_record, expected_trap,
            )
            return target_baseline
        reference_ready = bool(_observations(reference_record.get("observation", ())))
        if target_baseline is None:
            target_baseline = mark_coverage_only_baseline(
                evaluate_target_baseline(
                    baseline_artifact, target_runner,
                    baseline_record=(
                        reference_record if reference_ready
                        else {"observation": (), "reference_identity": {}}
                    ),
                    binding=binding, observer_fields=comparison_fields,
                    expected_trap=expected_trap,
                )
            )
        elif (
            reference_ready
            and not target_baseline_compared
            and isinstance(target_baseline.get("observations"), (list, tuple))
            and target_baseline.get("observations")
        ):
            target_baseline = mark_coverage_only_baseline(
                evaluate_target_baseline(
                    baseline_artifact, target_runner,
                    baseline_record=reference_record,
                    binding=binding, observer_fields=comparison_fields,
                    expected_trap=expected_trap,
                    target_observations=target_baseline["observations"],
                )
            )
            target_baseline_compared = True
        fallback_record = reference_record.get("reference_fallback")
        if (
            isinstance(fallback_record, Mapping)
            and isinstance(binding, Mapping)
            and _same_execution_backend(
                fallback_record.get("effective_backend"), binding.get("target"),
            )
        ):
            target_baseline["reference_target_relation"] = "shared-qemu-fallback"
            if isinstance(binding, dict):
                binding["reference_target_relation"] = "shared-qemu-fallback"
        if _campaign_deadline_exhausted(active_target_deadline(), target_baseline):
            target_baseline = {
                **target_baseline,
                "failure_class": "campaign-wall-clock-exhausted",
                "deadline_censored": True,
            }
        baseline_feedback = target_feedback_value(target_baseline)
        if baseline_feedback is not None:
            target_feedback_scores[original_sha256] = baseline_feedback
        return target_baseline

    def target_feedback_value(record: Mapping[str, object] | None) -> float | None:
        if not isinstance(record, Mapping):
            return None
        if record.get("coverage_only") is True:
            return None
        status = record.get("status")
        if status in {"clean", "target-tested", "target-probe-tested"} \
                and _comparison_qualified(record, "equivalent"):
            return 0.0
        if status in {"target-mismatch-candidate", "target-tested", "target-probe-tested"} \
                and _comparison_qualified(record, "non-equivalent"):
            return 1.0
        return None

    def reference_path_feedback(
        observations: object, program: object,
    ) -> dict[str, object]:
        """Score reference path shape and root-relative control-flow novelty."""
        rows = tuple(observations) if isinstance(observations, (list, tuple)) else ()
        if not rows:
            return {"status": "unavailable", "reason": "reference-trace-missing", "score": None}
        paths = []
        backends = set()
        for item in rows:
            backend = _object_field(item, "backend")
            if isinstance(backend, str):
                backends.add(backend)
            extra = _object_field(item, "extra_state")
            extra = extra if isinstance(extra, Mapping) else {}
            if _object_field(item, "contract_error") is not None:
                return {"status": "unavailable", "reason": "reference-contract-gap", "score": None}
            if _object_field(item, "trace_complete") is False or extra.get("trace_complete") is False:
                return {"status": "unavailable", "reason": "reference-trace-incomplete", "score": None}
            if backend == "native-rv64":
                trace_status = extra.get("native_trace_status")
                if not isinstance(trace_status, str) or not trace_status.startswith(
                    "observed-runner-trace",
                ):
                    return {"status": "unavailable", "reason": "reference-trace-incomplete", "score": None}
            program_path = extra.get("program_test_path")
            if isinstance(program_path, Mapping):
                if program_path.get("status") != "observed":
                    return {"status": "unavailable", "reason": "program-path-unavailable", "score": None}
                raw_pcs = program_path.get("executed_pcs")
            else:
                raw_pcs = _object_field(item, "executed_pcs")
            if not isinstance(raw_pcs, (list, tuple)) or not raw_pcs:
                return {"status": "unavailable", "reason": "reference-trace-missing", "score": None}
            pcs = tuple(_pc_value(value) for value in raw_pcs)
            if any(value is None for value in pcs):
                return {"status": "unavailable", "reason": "reference-trace-invalid", "score": None}
            instruction_count = _object_field(item, "instruction_count")
            whole_trace = _object_field(item, "executed_pcs")
            if (
                instruction_count is not None
                and isinstance(whole_trace, (list, tuple))
                and instruction_count != len(whole_trace)
            ):
                return {"status": "unavailable", "reason": "reference-trace-count-mismatch", "score": None}
            paths.append(tuple(int(value) for value in pcs))
        if len(backends) != 1:
            return {"status": "unavailable", "reason": "reference-identity-mixed", "score": None}
        path = paths[0]
        unique_pcs = set(path)
        unique_edges = set(zip(path, path[1:]))
        control_edges = {
            edge for edge in unique_edges
            if edge[1] - edge[0] not in {2, 4}
        }
        loop_edges = {edge for edge in control_edges if edge[1] <= edge[0]}

        def traced_instruction_words(
            source_program: object, pcs: Sequence[int],
        ) -> dict[int, int] | None:
            source = _program_source(source_program)
            words: list[int] = []
            for line in source.splitlines():
                match = re.match(
                    r"\s*\.(?:word|4byte|long)\s+([^#;]+)", line,
                    re.IGNORECASE,
                )
                if match is None:
                    continue
                operands = [part.strip() for part in match.group(1).split(",")]
                if not operands or any(
                    re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", value) is None
                    for value in operands
                ):
                    return None
                words.extend(int(value, 0) for value in operands)
            addresses = set(pcs)
            if not words or not addresses:
                return None
            base_pc = min(addresses)
            if addresses != {base_pc + 4 * index for index in range(len(words))}:
                return None
            return {pc: words[(pc - base_pc) // 4] for pc in addresses}

        def instruction_forms(
            words_by_pc: Mapping[int, int] | None,
        ) -> set[tuple[int, int]] | None:
            if words_by_pc is None:
                return None
            forms: set[tuple[int, int]] = set()
            rv32_forms = tuple(
                form for form in OFFICIAL_ALL_CATALOG_FORMS
                if form.encoding_length_bytes == 4
            )
            for raw in words_by_pc.values():
                matches = [
                    form for form in rv32_forms
                    if raw & form.mask == form.match
                ]
                if not matches:
                    return None
                specificity = max(form.mask.bit_count() for form in matches)
                fixed_patterns = {
                    (form.mask, form.match) for form in matches
                    if form.mask.bit_count() == specificity
                }
                if len(fixed_patterns) != 1:
                    return None
                forms.update(fixed_patterns)
            return forms

        instruction_novelty_bits = 0
        instruction_novelty_status = "unavailable"
        baseline_observations = (
            baseline_record.get("observation")
            if isinstance(baseline_record, Mapping) else None
        )
        baseline_rows = (
            tuple(baseline_observations)
            if isinstance(baseline_observations, (list, tuple)) else ()
        )
        baseline_path: tuple[int, ...] = ()
        if len(baseline_rows) == 1:
            baseline_item = baseline_rows[0]
            baseline_extra = _object_field(baseline_item, "extra_state")
            baseline_extra = baseline_extra if isinstance(baseline_extra, Mapping) else {}
            baseline_marker = baseline_extra.get("program_test_path")
            baseline_raw_pcs = (
                baseline_marker.get("executed_pcs")
                if isinstance(baseline_marker, Mapping)
                and baseline_marker.get("status") == "observed"
                else _object_field(baseline_item, "executed_pcs")
            )
            if isinstance(baseline_raw_pcs, (list, tuple)):
                parsed = tuple(_pc_value(value) for value in baseline_raw_pcs)
                if parsed and all(value is not None for value in parsed):
                    baseline_path = tuple(int(value) for value in parsed)

        baseline_pcs = set(baseline_path)
        baseline_edges = set(zip(baseline_path, baseline_path[1:]))
        baseline_control_edges = {
            edge for edge in baseline_edges
            if edge[1] - edge[0] not in {2, 4}
        }
        baseline_loop_edges = {
            edge for edge in baseline_control_edges if edge[1] <= edge[0]
        }
        novel_pcs = unique_pcs - baseline_pcs if baseline_path else set()
        novel_edges = unique_edges - baseline_edges if baseline_path else set()
        novel_control_edges = (
            control_edges - baseline_control_edges if baseline_path else set()
        )
        novel_loop_edges = (
            loop_edges - baseline_loop_edges if baseline_path else set()
        )

        current_words = traced_instruction_words(program, path)
        baseline_words = (
            traced_instruction_words(original, baseline_path)
            if baseline_path else None
        )
        current_instruction_forms = instruction_forms(current_words)
        baseline_instruction_forms = instruction_forms(baseline_words)
        if current_instruction_forms is not None and baseline_instruction_forms is not None:
            novel_instruction_forms = (
                current_instruction_forms - baseline_instruction_forms
            )
            instruction_form_novelty_status = "observed"
        else:
            novel_instruction_forms = set()
            instruction_form_novelty_status = "unavailable"
        if current_words is not None and baseline_words is not None:
            common_pcs = current_words.keys() & baseline_words.keys()
            instruction_novelty_bits = sum(
                (current_words[pc] ^ baseline_words[pc]).bit_count()
                for pc in common_pcs
            )
            instruction_novelty_status = "observed"
        instruction_encoding_reward = min(instruction_novelty_bits, 16) / 16.0
        feature_mass = len(unique_pcs) + len(unique_edges)
        if reference_path_reward_version == "path-v2":
            feature_mass += (
                2 * len(control_edges) + len(loop_edges)
                + len(novel_pcs) + 2 * len(novel_edges)
                + 2 * len(novel_control_edges) + 2 * len(novel_loop_edges)
                + 2 * len(novel_instruction_forms)
                + instruction_encoding_reward
            )
        score_basis = (
            "pc-edge-v1" if reference_path_reward_version == "pc-edge-v1"
            else "pc-edge-root-relative-instruction-and-path-novelty-v2"
        )
        return {
            "status": "observed",
            "backend": next(iter(backends)),
            "path_pc_count": len(unique_pcs),
            "path_edge_count": len(unique_edges),
            "control_edge_count": len(control_edges),
            "loop_edge_count": len(loop_edges),
            "root_relative_novel_pc_count": len(novel_pcs),
            "root_relative_novel_edge_count": len(novel_edges),
            "root_relative_novel_control_edge_count": len(novel_control_edges),
            "root_relative_novel_loop_edge_count": len(novel_loop_edges),
            "root_relative_novel_instruction_form_count": len(novel_instruction_forms),
            "root_relative_path_status": (
                "observed" if baseline_path else "root-path-unavailable"
            ),
            "instruction_form_novelty_status": instruction_form_novelty_status,
            "executed_instruction_novelty_bits": instruction_novelty_bits,
            "bounded_instruction_encoding_reward": instruction_encoding_reward,
            "instruction_novelty_status": instruction_novelty_status,
            "trace_digest": canonical_digest(path),
            "reward_version": reference_path_reward_version,
            "score_basis": score_basis,
            "score": math.log1p(feature_mass),
        }

    def campaign_counts(
        events: Sequence[Mapping[str, object]], run: object,
        records: Sequence[Mapping[str, object]],
        target_baseline_record: Mapping[str, object] | None,
        baselines_by_target: Mapping[str, Mapping[str, object]],
    ) -> dict[str, int]:
        counts = dict(_EMPTY_COUNTS)
        attempts = getattr(run, "mh_attempts", None)
        if isinstance(run, Mapping):
            attempts = run.get("mh_attempts")
        counts["steps"] = len(events)
        counts["mh_attempts"] = attempts if isinstance(attempts, int) else 0
        reference_rows = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        counts["reference_baseline_valid"] = int(
            isinstance(baseline, BuiltProgram)
            and isinstance(baseline_record, Mapping)
            and not _reference_baseline_qualification_gaps(
                baseline, baseline_record, reference_rows,
            )
        )
        for item in events:
            status = item.get("status")
            for key, value in (
                ("emi_rejected", "emi-rejected"),
                ("reference_rejected", "reference-rejected"),
                ("parameter_gap", "parameter-gap"),
                ("transport_gap", "transport-gap"),
                ("accepted", "accepted"),
                ("mh_rejected", "mh-rejected"),
                ("target_gap_event", "target-gap"),
                ("seed_miss", "seed-miss"),
            ):
                counts[key] += int(status == value)
            counts["reference_rejected"] += int(
                item.get("gate") == "reference-rejected"
                and status != "reference-rejected"
            )
            if isinstance(status, str):
                counts["reference_gap"] += int(status.startswith("reference-gap"))
                counts["proposal_gap"] += int(status.startswith("proposal-gap"))
                counts["profile_gap"] += int(status.startswith("profile-gap"))
            counts["reference_gap"] += int(isinstance(item.get("reference_gap"), Mapping))
            refresh = item.get("profile_refresh")
            counts["refresh_gap"] += int(
                isinstance(refresh, Mapping) and refresh.get("status") == "gap"
            )
            counts["reference_valid"] += int(item.get("gate") == "reference-passed")

        for item in records:
            for key, value in target_progress_delta(item).items():
                counts[key] += value
        baseline_counts = _target_baseline_counts(target_baseline_record)
        if baselines_by_target:
            baseline_counts = {}
            for target_baseline in baselines_by_target.values():
                for name, value in _target_baseline_counts(target_baseline).items():
                    baseline_counts[name] = baseline_counts.get(name, 0) + value
        counts["target_wall_clock_pending"] += baseline_counts.get(
            "target_wall_clock_pending", 0,
        )
        counts.update(
            target_attempted=counts["target_attempted"] + baseline_counts.get("target_attempted", 0),
            target_tested=counts["target_tested"] + baseline_counts.get("target_tested", 0),
            target_gap=counts["target_gap"] + baseline_counts.get("target_gap", 0),
            target_unsupported=counts["target_unsupported"]
            + baseline_counts.get("target_unsupported", 0),
            case_skipped=counts["case_skipped"]
            + baseline_counts.get("case_skipped", 0),
            target_mismatch_candidate=counts["target_mismatch_candidate"]
            + baseline_counts.get("target_mismatch_candidate", 0),
            target_transport_gap=counts["target_transport_gap"]
            + baseline_counts.get("target_transport_gap", 0),
        )
        return counts

    def feedback_score(state: ProgramVariant) -> float | None:
        scores = reference_feedback_scores if reference_feedback_mode else target_feedback_scores
        return scores.get(state.sha256)

    def write_ledger(events: Sequence[dict[str, object]], counts: dict[str, int]) -> Path:
        # ponytail: ledger stores transitions only; offline replay stays out of runtime.
        evidence_path = binding.get("evidence_path") if binding is not None else None
        path = Path(evidence_path) if isinstance(evidence_path, str) else Path("candidate-ledger.json")
        path = path if path.is_absolute() else root / path
        baseline = None if baseline_record is None else {
            **baseline_record,
            "observation": [
                _observation_record(item)
                for item in baseline_record["observation"]
            ],
        }
        rule_evidence = _campaign_contract.rule_evidence_records(events)
        ledger_events = []
        for event in events:
            item = _compact_campaign_event(event)
            if binding is not None:
                item["root_binding"] = dict(binding)
            ledger_events.append(item)
        payload = _json_safe({
            "schema": "program-candidate-ledger-v1",
            "original_source": original_source,
            "original_sha256": original_sha256,
            **({"original_includes": original_includes} if original_includes else {}),
            **({"raw_source": raw_source} if raw_sha256 != original_sha256 else {}),
            **({"raw_includes": raw_includes} if raw_includes else {}),
            "raw_sha256": raw_sha256,
            "root": rewrite_record,
            "original_run_params": original_run_params,
            "raw_run_params": run_params,
            "search_mode": "mcmc" if mcmc else "direct",
            **({"root_binding": binding} if binding is not None else {}),
            **({"target_bindings": target_bindings} if target_bindings else {}),
            **({"root_recipe_digest": binding["root_recipe_digest"]}
               if binding is not None and "root_recipe_digest" in binding else {}),
            **({"generation_profile_digest": binding["generation_profile_digest"]}
               if binding is not None and "generation_profile_digest" in binding else {}),
            **({"generation_identity_digest": binding["generation_identity_digest"]}
               if binding is not None and "generation_identity_digest" in binding else {}),
            "beta": beta,
            "seed": seed,
            "rng_scheme": _RNG_SCHEME,
            "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
            "reference_path_reward_version": reference_path_reward_version,
            "mcmc_chain_mode": (
                "reference-only" if reference_feedback_mode else "target-coupled"
            ),
            "target_execution_mode": (
                "post-chain-replay"
                if reference_feedback_mode and has_target_replay
                else "reference-only" if reference_feedback_mode else "synchronous"
            ),
            "target_replay": {
                "mode": (
                    "post-chain" if reference_feedback_mode and has_target_replay
                    else "inline" if has_target_replay else "not-requested"
                ),
                "target_ids": list(queued_target_ids),
                "pending_pair_count": len(pending_target_pairs),
                "record_count": len(target_records),
            },
            "reference_baseline": baseline,
            **({"reference_attempts": reference_attempts} if reference_attempts else {}),
            "initial_profile": initial_profile_record,
            "references": [
                {
                    **{key: value for key, value in item.items() if key != "target"},
                    "reference": {
                        **item["reference"],
                        "observations": {
                            key: value
                            for key, value in item["reference"]["observations"].items()
                            if key != "original"
                        },
                    }
                    if isinstance(item.get("reference"), Mapping)
                    and isinstance(item["reference"].get("observations"), Mapping)
                    else item.get("reference"),
                }
                for item in references
            ],
            "target_records": list(target_records),
            "target_records_by_target": {
                target_id: list(records)
                for target_id, records in target_records_by_target.items()
            },
            "target_baseline": target_baseline,
            "target_baselines_by_target": target_baselines_by_target,
            "events": ledger_events,
            "rule_evidence": rule_evidence,
            "counts": counts,
        })
        if not events and rewrite_record.get("status") in {"generation-gap", "transport-gap"}:
            payload["generation_failure"] = rewrite_record["failure"]
            payload["terminal_status"] = payload["generation_failure"].get(
                "status", "generation-gap"
            )
        elif not events and counts.get("profile_gap", 0):
            payload["terminal_status"] = "profile-gap"
        elif not events and counts.get("reference_gap", 0):
            payload["terminal_status"] = "reference-gap"
        elif not events and rewrite_record.get("status") == "seed-miss":
            payload["terminal_status"] = "seed-miss"
        elif not events and counts.get("artifact_gap", 0):
            payload["terminal_status"] = "artifact-gap"
        elif not events and counts.get("case_skipped", 0):
            payload["terminal_status"] = "case-skipped"
        elif counts.get("seed_miss", 0):
            payload["terminal_status"] = "seed-miss"
        atomic_write_json(path, payload, compact=True)
        return path

    def gap_result(
        status: str, record: dict[str, object],
    ) -> dict[str, object]:
        profile_record = record if status == "profile-gap" else {"status": "not-started"}
        if isinstance(reason := record.get("reason"), str) and reason:
            rewrite_record["failure"] = {
                "status": status, "reason": reason,
                **({"reference": record["reference"]}
                   if isinstance(record.get("reference"), list) else {}),
            }
        if status == "transport-gap":
            rewrite_record.update(
                status="transport-gap",
                boundary_witness={
                    **rewrite_record["boundary_witness"],
                    "status": "transport-gap", "realizes": False, "reason": reason,
                },
                aseed={"status": "not-started"},
            )
        counts = dict(_EMPTY_COUNTS)
        reference_rows = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        reference_valid = int(
            isinstance(baseline, BuiltProgram)
            and isinstance(baseline_record, Mapping)
            and not _reference_baseline_qualification_gaps(
                baseline, baseline_record, reference_rows,
            )
        )
        counts["reference_valid"] = reference_valid
        counts["reference_baseline_valid"] = reference_valid
        if status == "transport-gap":
            counts["transport_gap"] = 1
        elif status == "reference-gap":
            counts["reference_gap"] = 1
        elif status == "reference-rejected":
            counts["reference_gap"] = 1
        elif status == "profile-gap" and record.get("status") == "gap":
            counts["profile_gap"] = 1
        elif status == "seed-miss":
            counts["seed_miss"] = 1
        elif status == "case-skipped":
            counts["case_skipped"] = 1
            counts["target_unsupported"] = 1
        elif status == "artifact-gap":
            counts["artifact_gap"] = 1
        counts.update(_target_baseline_counts(target_baseline))
        return {
            "status": status,
            "original_sha256": original_sha256,
            "raw_sha256": raw_sha256,
            "root_sha256": rewrite_record.get("root_sha256"),
            "root_digest": rewrite_record.get("root_digest"),
            "root": rewrite_record,
            **({"root_binding": binding} if binding is not None else {}),
            "search_mode": "mcmc" if mcmc else "direct",
            "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
            "mcmc_chain_mode": (
                "reference-only" if reference_feedback_mode else "target-coupled"
            ),
            "target_execution_mode": (
                "post-chain-replay"
                if reference_feedback_mode and has_target_replay
                else "reference-only" if reference_feedback_mode else "synchronous"
            ),
            "target_replay": {
                "mode": (
                    "post-chain" if reference_feedback_mode and has_target_replay
                    else "inline" if has_target_replay else "not-requested"
                ),
                "target_ids": list(queued_target_ids),
                "pending_pair_count": 0,
                "record_count": len(target_records),
            },
            "reference_chain_steps": 0,
            "step_budget": steps,
            "steps_per_case_budget": steps,
            "total_steps_completed": 0,
            "steps_completed": 0,
            "seed": seed, "rng_scheme": _RNG_SCHEME, "beta": beta, "mcmc": mcmc,
            "profile": profile_record,
            **({"reason": record["reason"]} if record.get("reason") else {}),
            "chain": None,
            "reference": record.get("reference", []),
            **({"reference_attempts": record["reference_attempts"]}
               if isinstance(record.get("reference_attempts"), list) else {}),
            **({"reference_fallback": record["reference_fallback"]}
               if isinstance(record.get("reference_fallback"), Mapping) else {}),
            "reference_baseline": baseline_record,
            "target": [],
            "target_baseline": target_baseline,
            "counts": counts,
            "ledger_path": str(write_ledger((), counts)),
        }

    def profile_gap(reason: str | None) -> dict[str, object]:
        record = {"status": "gap", "reason": reason}
        return gap_result("profile-gap", record)

    if rewrite_record["status"] in {"generation-gap", "seed-miss"}:
        status = rewrite_record["status"]
        return gap_result(
            status,
            {"status": status, "reason": rewrite_record.get("reason")},
        )

    # Explicit direct compatibility calls use a single case pass.  The
    # canonical Ours profile keeps this branch disabled and runs the
    # MCMC/EMI loop below; the branch exists only for API/fixture callers that
    # deliberately request ``mcmc=False, steps=0``.
    if direct_case:
        try:
            baseline = replace(
                build_program(
                    original, builder, work_dir=root / "profile",
                    bare_builder=bare_builder,
                ),
                parent_sha256=None,
            )
        except Exception as error:
            return gap_result(
                "transport-gap",
                {"status": "transport-gap", "reason": str(error)},
            )

        # This compatibility branch records Target evidence independently of
        # the optional reference.  Reference-driven Ours campaigns do not use
        # this branch for their online loop; they replay Target after MH.
        capture_target_baseline(
            baseline, {"observation": (), "reference_identity": {}}, expected_trap,
        )
        # A target rejection is terminal for this case.  The outer case loop
        # records it and continues with the next generated case.
        if (
            isinstance(target_baseline, Mapping)
            and target_baseline.get("status") in {"case-skipped", "artifact-gap"}
        ):
            target_status = target_baseline.get("status")
            return gap_result(
                target_status,
                {
                    "status": target_status,
                    "reason": target_baseline.get("reason")
                    or ("unsupported-isa" if target_status == "case-skipped" else "artifact-gap"),
                },
            )
        reference_error = None
        try:
            observations = _observations(reference_runner(baseline))
        except Exception as error:
            observations = ()
            reference_error = f"{type(error).__name__}: {error}"
        if observations:
            baseline_record = {
                "source_sha256": baseline.source_sha256,
                "executable_sha256": baseline.executable_sha256,
                "run_params": dict(baseline.run_params),
                "reference_identity": _campaign_contract.reference_identity(observations),
                "observation": observations,
            }
            capture_target_baseline(baseline, baseline_record, expected_trap)
        else:
            reference_attempts.append({
                "status": "gap",
                "reason": reference_error or "reference-observation-missing",
            })

        comparison = (
            target_baseline.get("comparison")
            if isinstance(target_baseline, Mapping) else None
        )
        comparison_status = (
            comparison.get("status") if isinstance(comparison, Mapping) else None
        )
        target_status = (
            target_baseline.get("status")
            if isinstance(target_baseline, Mapping) else None
        )
        baseline_counts = _target_baseline_counts(target_baseline)
        baseline_mismatch = baseline_counts.get("target_mismatch_candidate", 0) > 0
        baseline_comparison_qualified = (
            isinstance(target_baseline, Mapping)
            and _comparison_qualified(target_baseline, "equivalent")
        )
        campaign_status = (
            "target-mismatch-candidate"
            if baseline_mismatch else
            "clean"
            if comparison_status == "equivalent" and baseline_comparison_qualified else
            "target-gap"
            if comparison_status in {"equivalent", "non-equivalent"}
            and not baseline_comparison_qualified else
            "case-skipped"
            if target_status == "case-skipped" else
            "target-tested"
            if target_status == "target-tested" else
            "reference-valid"
            if target_runner is None and observations else
            "target-gap"
        )
        counts = dict(_EMPTY_COUNTS)
        reference_rows = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        reference_valid = int(
            isinstance(baseline, BuiltProgram)
            and isinstance(baseline_record, Mapping)
            and not _reference_baseline_qualification_gaps(
                baseline, baseline_record, reference_rows,
            )
        )
        counts["reference_baseline_valid"] = reference_valid
        counts["reference_valid"] = reference_valid
        counts["reference_gap"] = int(not reference_valid)
        counts.update(baseline_counts)
        initial_profile_record = {
            "status": "not-required",
            "reason": "direct-minimal-flow",
        }
        target_records: list[dict[str, object]] = []
        direct_event = {
            "step": 0,
            "event_type": "direct-case",
            "status": campaign_status,
            "gate": "direct-case",
            "parent_sha256": None,
            "candidate_sha256": original_sha256,
            "target_available": has_target_replay,
            "target": {"target_record_index": 0} if target_records else None,
        }
        ledger_path = write_ledger((direct_event,), counts)
        return {
            "status": campaign_status,
            "original_sha256": original_sha256,
            "raw_sha256": raw_sha256,
            "root_sha256": rewrite_record.get("root_sha256"),
            "root_digest": rewrite_record.get("root_digest"),
            "root": rewrite_record,
            **({"root_binding": binding} if binding is not None else {}),
            "beta": beta, "mcmc": False,
            "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
            "mcmc_chain_mode": (
                "reference-only" if reference_feedback_mode else "target-coupled"
            ),
            "seed": seed,
            "rng_scheme": _RNG_SCHEME, "search_mode": "direct",
            "target_timeout_seconds": target_timeout_seconds,
            "profile": initial_profile_record,
            "reference_baseline": baseline_record,
            "reference_attempts": reference_attempts,
            "chain": {"events": [direct_event], "run": {"mh_attempts": 0}},
            "reference": [], "target": target_records,
            "target_baseline": target_baseline,
            "counts": counts,
            "ledger_path": str(ledger_path),
        }

    try:
        baseline_artifact = build_program(
            original, builder, work_dir=root / "profile",
            bare_builder=bare_builder,
        )
        if key := _build_cache_key(original, builder, bare_builder):
            artifact_cache[key] = baseline_artifact
        baseline = replace(baseline_artifact, parent_sha256=None)
    except Exception as error:
        return gap_result(
            "transport-gap",
            {"status": "transport-gap", "reason": str(error)},
        )

    try:
        # Direct reference mode defers the Target baseline until the MCMC chain
        # has closed. Other routes retain the historical synchronous baseline.
        capture_target_baseline(
            baseline, {"observation": (), "reference_identity": {}}, expected_trap,
        )
        if isinstance(target_baseline, Mapping) and target_baseline.get("status") in {
            "case-skipped", "artifact-gap",
        }:
            target_status = target_baseline.get("status")
            return gap_result(
                target_status,
                {
                    "status": target_status,
                    "reason": target_baseline.get("reason")
                    or ("unsupported-isa" if target_status == "case-skipped" else "artifact-gap"),
                },
            )
        try:
            observations = _observations(reference_runner(baseline))
        except Exception as baseline_error:
            # K1/QEMU reference 的传输或能力缺口不是 case 缺口：记入
            # reference_attempts 后继续 MCMC，Target 回放仍在链结束后进行。
            observations = ()
            reference_attempts.append({
                "status": _reference_exception_status(baseline_error),
                "reason": str(baseline_error),
                "backend": "native-rv64",
                "contract_error": str(baseline_error),
            })
        provisional_reference = {
            "observation": observations,
            "reference_identity": (
                _campaign_contract.reference_identity(observations)
                if observations else {}
            ),
        }
        # Preserve the raw reference witness for the ledger and for later
        # differential comparisons.  Profile construction has its own
        # validity view so an optional reference contract issue cannot erase
        # evidence that was actually observed.
        profile_observations = observations
        capture_target_baseline(baseline, provisional_reference, expected_trap)
        # baseline reference 失败不终止 case。MCMC 会在每个候选上独立跑
        # K1/QEMU reference，Target 回放不依赖 baseline 是否建模成功。
        for observation in observations:
            contract_error = _object_field(observation, "contract_error")
            if isinstance(contract_error, str) and contract_error:
                status = _reference_failure_status(contract_error)
                reference_attempts = [_observation_record(item) for item in observations]
                reference_evidence = []
                for item in observations:
                    evidence = {
                        name: value for name in (
                            "backend", "outcome", "exit_code", "contract_error",
                            "instruction_count",
                        )
                        if (value := _object_field(item, name)) is not None
                    }
                    executed_pcs = _object_field(item, "executed_pcs")
                    if isinstance(executed_pcs, (list, tuple)):
                        evidence["executed_pc_count"] = len(executed_pcs)
                    extra_state = _object_field(item, "extra_state")
                    if isinstance(extra_state, Mapping):
                        for name in ("native_trace_status", "runner_failure_class"):
                            if extra_state.get(name) not in (None, ""):
                                evidence[name] = extra_state[name]
                    reference_evidence.append(evidence)
                fallback_record = _reference_fallback_summary(observations)
                if not open_capability:
                    return gap_result(status, {
                        "status": status,
                        "reason": "reference-contract:" + contract_error,
                        "reference": reference_evidence,
                        "reference_attempts": reference_attempts,
                        **({
                            "reference_fallback": {
                                key: value for key, value in fallback_record.items()
                                if key != "k1_attempts"
                            }
                        } if fallback_record is not None else {}),
                    })
                reference_attempts.extend(reference_evidence)
                reference_attempts.append({
                    "status": status,
                    "reason": "reference-contract:" + contract_error,
                })
                profile_observations = ()
                break
        baseline_record = {
            "source_sha256": baseline.source_sha256,
            "executable_sha256": baseline.executable_sha256,
            "run_params": dict(baseline.run_params),
            "reference_identity": _campaign_contract.reference_identity(observations),
            "observation": observations,
        }
        if reference_feedback_mode and has_target_replay:
            queue_target_baseline(observations)
        if reference_feedback_mode:
            baseline_path_feedback = reference_path_feedback(observations, original)
            baseline_record["reference_path_feedback"] = baseline_path_feedback
            reference_feedback_scores[original_sha256] = baseline_path_feedback.get("score")
            reference_feedback_records[original_sha256] = baseline_path_feedback
        fallback_record = _reference_fallback_summary(observations)
        probe_fallback = reference_fallback_evidence
        probe_attempts = (
            probe_fallback.get("k1_attempts")
            if isinstance(probe_fallback, Mapping) else None
        )
        if (
            isinstance(fallback_record, Mapping)
            and isinstance(probe_fallback, Mapping)
            and isinstance(probe_attempts, (list, tuple))
            and probe_fallback.get("k1_attempt_digest")
            == fallback_record.get("k1_attempt_digest")
            and canonical_digest(probe_attempts) == fallback_record.get("k1_attempt_digest")
        ):
            # The first K1 request may be the model's reference-witness probe.
            # The pinned QEMU runner keeps later observations compact, so carry
            # that one original transport attempt into the formal baseline.
            fallback_record = {**fallback_record, "k1_attempts": list(probe_attempts)}
            baseline_record["observation"] = _attach_reference_fallback_evidence(
                observations, fallback_record,
            )
        if fallback_record is not None:
            fallback_summary = {
                key: value for key, value in fallback_record.items()
                if key != "k1_attempts"
            }
            baseline_record.update(
                reference_fallback=fallback_summary,
                preferred_reference_backend=fallback_record.get("preferred_backend"),
                effective_reference_backend=fallback_record.get("effective_backend"),
                fallback_reason=fallback_record.get("fallback_reason"),
            )
            if isinstance(binding, dict):
                binding.update(
                    reference_fallback=fallback_summary,
                    preferred_reference_backend=fallback_record.get("preferred_backend"),
                    effective_reference_backend=fallback_record.get("effective_backend"),
                )
        rewrite_record["reference_identity"] = dict(baseline_record["reference_identity"])
        rewrite_record["executable_sha256"] = baseline.executable_sha256
        rewrite_record["i0"] = dict(baseline_record["run_params"] or run_params)
        rewrite_record["i0_digest"] = canonical_digest(rewrite_record["i0"])
        rewrite_record["reference_observation_digest"] = canonical_digest([
            _observation_record(item) for item in observations
        ])
        rewrite_record["root_digest"] = canonical_digest({
            "raw_sha256": rewrite_record["raw_sha256"],
            "rewrite_sha256": rewrite_record["rewrite_sha256"],
            "hint": rewrite_record.get("hint"),
            "run_params": rewrite_record.get("run_params", run_params),
            "i0_digest": rewrite_record["i0_digest"],
            "reference_observation_digest": rewrite_record["reference_observation_digest"],
            "reference_identity": rewrite_record["reference_identity"],
        })
        expected = (
            binding.get("reference_expected")
            if binding is not None and isinstance(binding.get("reference_expected"), Mapping)
            else None
        )
        # Target execution is a coverage side channel.  Reference-driven Ours
        # routes defer it until after the MH chain; compatibility routes may
        # still capture a synchronous baseline here.
        capture_target_baseline(baseline, baseline_record, expected_trap)
        def structured_trap(item: object) -> bool:
            extra = _object_field(item, "extra_state")
            guest_trap = isinstance(extra, Mapping) and guest_trap_observed(dict(extra))
            boundary_trap = isinstance(extra, Mapping) and type(extra.get("trap.cause")) is int \
                and 0 <= extra["trap.cause"] < 64 \
                and type(extra.get("trap.epc")) is int \
                and 0 <= extra["trap.epc"] < 1 << 64 \
                and not extra["trap.epc"] & 1 \
                and type(extra.get("trap.tval")) is int \
                and 0 <= extra["trap.tval"] < 1 << 64
            return (
                _object_field(item, "outcome") in {"trap", "nonzero-exit"}
                and isinstance(extra, Mapping)
                and (
                    guest_trap
                    or expected_trap
                    and extra.get("sail_boundary_trap_reached") is True
                    and boundary_trap
                    or expected_trap
                    and _object_field(item, "backend") == "sail-riscv"
                    and boundary_trap
                )
            )
        if expected and not open_capability:
            wanted = expected.get("status")
            recorded_status = wanted == "recorded"
            wanted = "normal" if wanted in {"reference-valid", "completed"} else wanted
            if not recorded_status and (not isinstance(wanted, str) or not wanted.strip()):
                return profile_gap("profile-gap:reference-expected-status-invalid")
            for observation in () if recorded_status else observations:
                outcome = _object_field(observation, "outcome")
                observed_status = "normal" if outcome in {"normal", "completed"} else outcome
                matches = observed_status == wanted
                if not matches:
                    if wanted == "trap" and structured_trap(observation):
                        continue
                    if (
                        wanted == "normal"
                        and observed_status == "timeout"
                        and not _object_field(observation, "contract_error")
                    ):
                        rewrite_record["reference_expected_status_override"] = {
                            "expected": wanted, "observed": outcome,
                            "reason": "reference-timeout",
                        }
                        continue
                    if (
                        wanted == "normal"
                        and structured_trap(observation)
                        and (
                            expected_trap
                            or isinstance(binding, Mapping)
                            and binding.get("evidence_tier") == "probe"
                        )
                    ):
                        rewrite_record["reference_expected_status_override"] = {
                            "expected": wanted, "observed": outcome,
                            "reason": (
                                "single-expected-trap" if actual_single_route
                                else "program-expected-trap"
                                if expected_trap else "probe-structured-guest-trap"
                            ),
                        }
                        continue
                    if (
                        wanted == "normal"
                        and observed_status in {"trap", "nonzero-exit"}
                        and isinstance(_object_field(observation, "executed_pcs"), (list, tuple))
                        and bool(_object_field(observation, "executed_pcs"))
                        and not _object_field(observation, "contract_error")
                    ):
                        rewrite_record["reference_expected_status_override"] = {
                            "expected": wanted,
                            "observed": outcome,
                            "reason": "natural-guest-terminal",
                        }
                        continue
                    contract_error = _object_field(observation, "contract_error")
                    if isinstance(contract_error, str) and contract_error:
                        return profile_gap(
                            "profile-gap:reference-contract:" + contract_error
                        )
                    return profile_gap("profile-gap:reference-expected-status-mismatch")
            expected_identity = {
                name: expected[name] for name in (
                    "backend", "profile_id", "input_id", "tool_version",
                    "source_commit", "identity_digest", "binary_sha256",
                    "guest_elf_sha256",
                ) if expected.get(name) not in (None, "")
            }
            if _reference_fallback_matches_binding(
                binding, expected_identity, fallback_record,
            ):
                expected_identity["backend"] = fallback_record["effective_backend"]
            if expected_identity.get("backend") == "qemu":
                expected_identity["backend"] = (
                    "qemu-riscv32"
                    if str(run_params.get("isa") or run_params.get("isa_profile") or "")
                    .lower().startswith("rv32") else "qemu-riscv64"
                )
            actual_identity = dict(baseline_record["reference_identity"])
            for name in ("binary_sha256", "guest_elf_sha256"):
                actual_identity.setdefault(name, baseline.executable_sha256)
            if any(
                name == "backend"
                and isinstance(value, str)
                and isinstance(actual_identity.get(name), str)
                and not _same_execution_backend(actual_identity[name], value)
                or name != "backend" and actual_identity.get(name) != value
                for name, value in expected_identity.items()
            ):
                return profile_gap("profile-gap:reference-expected-identity-mismatch")
        elif expected:
            reference_attempts.append({
                "status": "gap",
                "reason": "profile-gap:reference-expected-contract",
            })
            baseline_record["reference_expected_contract"] = "unverified"
        if any(_object_field(item, "outcome") == "timeout" for item in observations):
            if not open_capability:
                return profile_gap("profile-gap:reference-timeout")
            reference_attempts.append({
                "status": "gap", "reason": "profile-gap:reference-timeout",
            })
            profile_observations = ()
        reference_backend = baseline_record["reference_identity"].get("backend")
        if reference_backend != "sail-riscv":
            for item in observations:
                recorded = _object_field(item, "binary_sha256")
                guest = _object_field(item, "guest_elf_sha256")
                if any(value is not None and value != baseline.executable_sha256
                       for value in (recorded, guest)):
                    if not open_capability:
                        return profile_gap("profile-gap:baseline-executable-identity-mismatch")
                    reference_attempts.append({
                        "status": "gap",
                        "reason": "profile-gap:baseline-executable-identity-mismatch",
                    })
                    profile_observations = ()
                    break
        profile_result = _campaign_contract.attach_execution_identity(
            route.build_profile(
                original,
                _profile_observations(profile_observations, baseline.executable_path),
                input_identity=_campaign_contract.profile_identity(baseline, profile_observations),
                **({"expected_trap": expected_trap} if route.name == "program" else {}),
            ),
            baseline,
            profile_observations,
        )
    except (ExecutionEnvironmentError, OSError, subprocess.SubprocessError) as error:
        capture_target_baseline(
            baseline, {"observation": (), "reference_identity": {}}, expected_trap,
        )
        if not open_capability:
            return gap_result(
                "transport-gap",
                {"status": "transport-gap", "reason": str(error)},
            )
        reference_attempts.append({
            "status": "transport-gap", "reason": str(error),
            "backend": reference_backend_label,
        })
        observations = ()
        baseline_record = {
            "source_sha256": baseline.source_sha256,
            "executable_sha256": baseline.executable_sha256,
            "run_params": dict(baseline.run_params),
            "reference_identity": {}, "observation": observations,
        }
        profile_result = None
    except Exception as error:
        capture_target_baseline(
            baseline, {"observation": (), "reference_identity": {}}, expected_trap,
        )
        status = _reference_exception_status(error)
        if not open_capability:
            if status == "profile-gap":
                return profile_gap(str(error))
            return gap_result(status, {
                "status": status, "reason": str(error),
                "reference": [{
                    "backend": reference_backend_label, "contract_error": str(error),
                }],
            })
        reference_attempts.append({
            "status": status, "reason": str(error),
            "backend": reference_backend_label, "contract_error": str(error),
        })
        observations = ()
        baseline_record = {
            "source_sha256": baseline.source_sha256,
            "executable_sha256": baseline.executable_sha256,
            "run_params": dict(baseline.run_params),
            "reference_identity": {}, "observation": observations,
        }
        profile_result = None

    profile = getattr(profile_result, "profile", profile_result)
    profile_reason = getattr(profile_result, "reason", None)
    if profile is None:
        if not open_capability:
            return profile_gap(profile_reason or "profile-gap:trusted-reference-profile-missing")
        # 扩展 reference 仍可能遇到未建模的 Q/P/H/厂商/复杂特权形式。拿不到
        # profile 时记下具名缺口；MCMC/EMI/小模型链、目标执行与覆盖率采集都
        # 不依赖该 profile，继续跑而不终止 case。
        rewrite_record["aseed"] = {
            "status": "gap",
            "build": bool(baseline_record.get("executable_sha256")),
            "reference_outcome_valid": bool(baseline_record.get("observation")),
            "realizes": rewrite_record["boundary_witness"].get("realizes") is True,
            "reason": profile_reason or "profile-gap:trusted-reference-profile-missing",
        }
        initial_profile_record = {"status": "not-required", "reason": profile_reason}
    elif getattr(profile, "source_sha256", None) != original_sha256:
        if not open_capability:
            return profile_gap("profile-gap:profile-source-mismatch")
        profile_reason = "profile-gap:profile-source-mismatch"
        profile = None
        initial_profile_record = {"status": "not-required", "reason": profile_reason}
    elif not _profile_input_matches(original, profile):
        if not open_capability:
            return profile_gap("profile-gap:profile-input-identity-mismatch")
        profile_reason = "profile-gap:profile-input-identity-mismatch"
        profile = None
        initial_profile_record = {"status": "not-required", "reason": profile_reason}
    single_route = actual_single_route
    sequence_realized = (
        _campaign_contract.profile_dynamic_realizes(profile)
        if profile is not None and (single_route or required_sequence is None)
        else _campaign_contract.profile_sequence_realizes(profile, required_sequence)
        if profile is not None
        else False
    )
    if required_sequence is not None:
        rewrite_record["boundary_witness"].update(
            required_sequence=list(required_sequence),
            sequence_realizes=sequence_realized,
        )
    intent_realized = (
        route.profile_realizes(
            original, profile, rewrite_record["hint"],
        )
        if rewrite_record["mode"] == "model-on"
        else sequence_realized
    )
    # 没有 profile（reference 域缺口）时不能据此判 seed-miss：
    # 该判定只描述 reference 是否建模了这条指令，不描述 case 是否有效。
    realized = (sequence_realized and intent_realized) if profile is not None else True
    rewrite_record["boundary_witness"].update(
        realizes=realized, status="realized" if realized else "unrealized",
    )
    if required_sequence is None:
        rewrite_record["boundary_witness"]["dynamic_observer"] = "trusted-reference-profile"
    if not realized:
        rewrite_record.update(
            status="seed-miss",
            reason=(
                "seed-miss:required-sequence-not-realized"
                if not sequence_realized
                else "seed-miss:reference-witness-not-realized"
            ),
        )
        rewrite_record["boundary_witness"]["status"] = "seed-miss"
    def outcome_valid(item: object) -> bool:
        outcome = _object_field(item, "outcome")
        exit_code = _object_field(item, "exit_code")
        return (
            outcome in {"normal", "completed"} and type(exit_code) is int and exit_code == 0
            or outcome == "nonzero-exit" and type(exit_code) is int and exit_code != 0
            or outcome == "trap" and (exit_code is None or type(exit_code) is int)
            or outcome == "timeout" and (exit_code is None or exit_code == 124)
        )
    valid_outcomes = all(outcome_valid(item) for item in observations)
    checks = {
        "build": bool(baseline_record.get("executable_sha256")),
        "reference_outcome_valid": valid_outcomes,
        "realizes": rewrite_record["boundary_witness"].get("realizes") is True,
    }
    rewrite_record["aseed"] = {
        "status": "passed" if all(
            checks[name] for name in ("build", "reference_outcome_valid", "realizes")
        ) else "gap",
        **checks,
    }
    if not all(
        checks[name] for name in ("build", "reference_outcome_valid", "realizes")
    ) and not open_capability:
        if (rewrite_record.get("status") == "seed-miss"
                and all(checks[name] for name in ("build", "reference_outcome_valid"))):
            return gap_result(
                "seed-miss",
                {"status": "seed-miss", "reason": rewrite_record.get("reason")},
            )
        return profile_gap("profile-gap:aseed")

    if profile is not None:
        initial_profile_record = {"status": "ready", **profile.to_record()}

    # The canonical Ours run executes the online EMI/MCMC proposal loop.  A
    # zero-step direct pass remains an explicit compatibility mode for callers
    # that intentionally disable the framework search modules.
    if not mcmc and steps == 0 and rewrite_hint is None:
        counts = dict(_EMPTY_COUNTS)
        reference_rows = _observations(
            baseline_record.get("observation", ())
            if isinstance(baseline_record, Mapping) else ()
        )
        reference_valid = int(
            isinstance(baseline, BuiltProgram)
            and isinstance(baseline_record, Mapping)
            and not _reference_baseline_qualification_gaps(
                baseline, baseline_record, reference_rows,
            )
        )
        counts["reference_baseline_valid"] = reference_valid
        counts["reference_valid"] = counts["reference_baseline_valid"]
        counts["reference_gap"] = int(not reference_valid)
        baseline_counts = _target_baseline_counts(target_baseline)
        counts.update(
            target_attempted=baseline_counts.get("target_attempted", 0),
            target_tested=baseline_counts.get("target_tested", 0),
            target_gap=baseline_counts.get("target_gap", 0),
            target_unsupported=baseline_counts.get("target_unsupported", 0),
            case_skipped=baseline_counts.get("case_skipped", 0),
            target_mismatch_candidate=baseline_counts.get(
                "target_mismatch_candidate", 0
            ),
            target_transport_gap=baseline_counts.get("target_transport_gap", 0),
            target_wall_clock_pending=baseline_counts.get(
                "target_wall_clock_pending", 0
            ),
        )
        comparison = (
            target_baseline.get("comparison")
            if isinstance(target_baseline, Mapping) else None
        )
        comparison_status = (
            comparison.get("status") if isinstance(comparison, Mapping) else None
        )
        comparison_qualified = (
            isinstance(target_baseline, Mapping)
            and _comparison_qualified(target_baseline)
        )
        campaign_status = (
            "target-mismatch-candidate"
            if baseline_counts.get("target_mismatch_candidate", 0) else
            "clean"
            if comparison_status == "equivalent" and comparison_qualified else
            "target-gap"
            if comparison_status in {"equivalent", "non-equivalent"}
            and not comparison_qualified else
            "case-skipped"
            if isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "case-skipped" else
            "target-tested"
            if isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "target-tested" else
            "reference-valid"
            if target_runner is None and counts["reference_valid"] else
            "target-gap"
        )
        campaign_reason = None
        if isinstance(target_baseline, Mapping) and campaign_status == "target-gap":
            value = target_baseline.get("failure_class") or target_baseline.get("reason")
            if not value and comparison_status in {"equivalent", "non-equivalent"}:
                value = ",".join(
                    item for item in target_baseline.get(
                        "comparison_qualification_gaps", (),
                    ) if isinstance(item, str)
                ) or "comparison-unqualified"
            value = value or target_baseline.get("status")
            if isinstance(value, str) and value:
                campaign_reason = value
        target_records: list[dict[str, object]] = []
        direct_event = {
            "step": 0,
            "event_type": "direct-case",
            "status": campaign_status,
            "gate": "direct-case",
            "parent_sha256": None,
            "candidate_sha256": original_sha256,
            "target_available": has_target_replay,
            "target": {"target_record_index": 0} if target_records else None,
        }
        ledger_path = write_ledger((direct_event,), counts)
        return {
            "status": campaign_status,
            **({"reason": campaign_reason} if campaign_reason else {}),
            "original_sha256": original_sha256,
            "raw_sha256": raw_sha256,
            "root_sha256": rewrite_record.get("root_sha256"),
            "root_digest": rewrite_record.get("root_digest"),
            "root": rewrite_record,
            **({"root_binding": binding} if binding is not None else {}),
            "beta": beta, "mcmc": False,
            "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
            "mcmc_chain_mode": (
                "reference-only" if reference_feedback_mode else "target-coupled"
            ),
            "seed": seed,
            "rng_scheme": _RNG_SCHEME, "search_mode": "direct",
            "target_timeout_seconds": target_timeout_seconds,
            "profile": initial_profile_record,
            "reference_baseline": baseline_record,
            "chain": {"events": [direct_event], "run": {"mh_attempts": 0}},
            "reference": [], "target": target_records,
            "target_baseline": target_baseline,
            "counts": counts,
            "ledger_path": str(ledger_path),
        }

    bound_rule = (
        binding.get("rule")
        if binding is not None and binding.get("route") != "single" else None
    )

    def record_coverage_only_target(
        result: Mapping[str, object], reference_index: int,
        candidate: ProgramVariant,
    ) -> None:
        """Keep target execution evidence when no differential is possible."""
        target = result.get("target")
        if not isinstance(target, Mapping):
            return
        target_status = target.get("status")
        status = (
            "target-tested" if target_status in {"equivalent", "non-equivalent"} else
            "case-skipped" if target_status == "case-skipped" else
            "target-gap"
        )
        artifacts = result.get("artifacts")
        target_original_artifact = (
            artifacts.get("reference_original")
            if isinstance(artifacts, Mapping) and isinstance(target, Mapping) else None
        )
        target_artifact = (
            artifacts.get("target_variant")
            if isinstance(artifacts, Mapping) else None
        )
        # 这是同步 target-feedback 兼容路径。Direct/Program-Full 的
        # reference-driven 路径不调用此 helper，而是只记录 Target 诊断。
        feedback = None
        target_records.append({
            "step": None, "reference_index": reference_index,
            "state_sha256": candidate.sha256,
            "parent_sha256": candidate.parent_sha256,
            "status": status,
            "terminal_disposition": status,
            "coverage_only": True,
            "coverage_attempted": True,
            # This helper is called only after ``run_emi_pair`` has been given
            # a target runner.  A reference rejection must not erase the fact
            # that the target process already ran; coverage is independent of
            # the differential verdict.
            "target_attempted": target_runner is not None,
            "comparison_qualified": False,
            "comparison_qualification_gaps": ["reference-not-qualified"],
            "target_result_status": result.get("status"),
            "target_feedback": feedback,
            "comparison": dict(target),
            **({"target_original_artifact": target_original_artifact}
               if isinstance(target_original_artifact, Mapping) else {}),
            "artifact": target_artifact,
        })

    def rerun_target(
        cached: Mapping[str, object], base: ProgramVariant,
        candidate: ProgramVariant, parent_sha256: str,
    ) -> dict[str, object]:
        reference = cached["reference"]
        base_sha256 = program_sha256(base)
        def cached_reference(artifact: BuiltProgram):
            side = "original" if artifact.source_sha256 == base_sha256 else "variant"
            return reference["observations"][side]
        return run_emi_pair(
            base,
            candidate,
            builder,
            cached_reference,
            target_runner,
            work_dir=root / "target" / f"{len(target_records):06d}",
            expected_parent_sha256=parent_sha256,
            reference_observations=reference["observations"]["original"],
            expected_target_identity=binding,
            bare_builder=bare_builder,
            artifact_cache=artifact_cache,
            campaign_deadline_monotonic=active_target_deadline(),
        )

    def record_target_result(
        target_result: Mapping[str, object], reference_index: int,
        candidate: ProgramVariant,
        *, target_id: str | None = None,
        records: list[dict[str, object]] | None = None,
    ) -> str:
        """落盘 Target 观察；它不参与 reference-feedback 的 MH 决策。"""
        target = target_result.get("target")
        pair_result_status = target_result.get("status", "target-gap")
        target_result_status = pair_result_status
        target_comparison_status = (
            target.get("status") if isinstance(target, Mapping) else None
        )
        target_observation_map = (
            target.get("observations") if isinstance(target, Mapping) else None
        )
        target_variant_observations = (
            target_observation_map.get("variant")
            if isinstance(target_observation_map, Mapping) else None
        )
        target_execution_completed = _target_execution_completed(
            target_variant_observations,
        )
        comparison_has_reference_gap = _comparison_has_reference_gap(target)
        pair_reference = target_result.get("reference")
        reference_qualified = (
            isinstance(pair_reference, Mapping)
            and pair_reference.get("status") == "equivalent"
            and not _record_has_qualification_gap(pair_reference)
        )
        coverage_only = (
            reference_feedback_mode
            and target_execution_completed
            and (
                pair_result_status in {
                    "reference-rejected", "reference-gap", "profile-gap", "transport-gap",
                }
                and (
                    target_comparison_status in {"equivalent", "non-equivalent"}
                    or comparison_has_reference_gap
                )
                or pair_result_status == "target-gap"
                and target_result.get("failure_class") == "target-contract-gap"
                and comparison_has_reference_gap
            )
        )
        if coverage_only:
            target_result_status = "target-tested"
        elif (
            isinstance(target, Mapping)
            and pair_result_status in {
                "reference-rejected", "reference-gap", "profile-gap", "transport-gap",
            }
        ):
            # run_emi_pair returns early when the reference is invalid, before
            # it can finish Target artifact/input/identity qualification. Keep
            # the raw Target comparison for diagnosis, but never promote it.
            target_result_status = "target-gap"
        comparison_qualified = (
            not coverage_only
            and reference_qualified
            and pair_result_status in {"clean", "target-mismatch-candidate"}
            and isinstance(target, Mapping)
            and target_comparison_status == (
                "equivalent" if pair_result_status == "clean" else "non-equivalent"
            )
            and not _record_has_qualification_gap(target)
        )
        if target_result_status in {"clean", "target-mismatch-candidate"} \
                and not comparison_qualified:
            target_result_status = "target-gap"
        target_qualification_gaps = []
        if pair_result_status in {
            "reference-rejected", "reference-gap", "profile-gap", "transport-gap",
        } or pair_result_status in {"clean", "target-mismatch-candidate"} \
                and not reference_qualified:
            target_qualification_gaps.append("reference-not-qualified")
        if isinstance(target, Mapping):
            if _record_has_qualification_gap(target):
                target_qualification_gaps.append("target-comparison-qualification-gap")
        if pair_result_status in {"clean", "target-mismatch-candidate"} \
                and not comparison_qualified:
            target_qualification_gaps.append("target-comparison-unqualified")
        staged_result = {
            **target_result,
            "status": target_result_status,
            "comparison_qualified": comparison_qualified,
        }
        artifacts = target_result.get("artifacts")
        target_original_artifact = (
            artifacts.get("reference_original")
            if isinstance(artifacts, Mapping) and isinstance(target, Mapping) else None
        )
        target_artifact = (
            artifacts.get("target_variant")
            if isinstance(artifacts, Mapping) and target is not None else
            artifacts.get("reference_variant")
            if isinstance(artifacts, Mapping) else None
        )
        target_status, target_attempted = _target_result_stage(
            staged_result, probe_mode,
        )
        comparison = target
        feedback = target_feedback_value({
            "status": target_result_status,
            "comparison": comparison,
            "coverage_only": coverage_only,
            "comparison_qualified": comparison_qualified,
            "comparison_qualification_gaps": target_qualification_gaps,
        })
        deadline_censored = not coverage_only and _campaign_deadline_exhausted(
            active_target_deadline(), target_result,
        )
        observation_reason = _target_observation_reason(comparison)
        generic_gap_reasons = {
            "target-contract-gap", "target-runner-gap",
            "target-gap", "target-probe-gap", "reference-gap",
        }
        result_reason = target_result.get("failure_class", target_result_status)
        if (
            isinstance(observation_reason, str)
            and observation_reason not in generic_gap_reasons
            and isinstance(result_reason, str)
            and result_reason in generic_gap_reasons
        ):
            result_reason = observation_reason
        failure_class = (
            "campaign-wall-clock-exhausted" if deadline_censored else
            result_reason
        )
        record = {
            "step": None, "reference_index": reference_index,
            "state_sha256": candidate.sha256,
            "parent_sha256": candidate.parent_sha256,
            **({"target_id": target_id} if target_id is not None else {}),
            "status": target_status,
            "terminal_disposition": target_status,
            "candidate": target_status == "target-mismatch-candidate",
            **({"coverage_only": True} if coverage_only else {}),
            "comparison_qualified": comparison_qualified,
            **({"comparison_qualification_gaps": list(dict.fromkeys(
                target_qualification_gaps,
            ))} if target_qualification_gaps else {}),
            **({"target_attempted": False} if not target_attempted else {}),
            "target_result_status": (
                pair_result_status if coverage_only else target_result_status
            ),
            # Keep the target comparison as diagnostic evidence.  The callback
            # passed to sample_program is reference_feedback in Direct mode.
            "target_feedback": feedback,
            **({"failure_class": failure_class}
               if target_status in {
                   "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
               } else {}),
            **({"deadline_censored": True} if deadline_censored else {}),
            "comparison": comparison,
            **({"target_original_artifact": target_original_artifact}
               if isinstance(target_original_artifact, Mapping) else {}),
            "artifact": target_artifact,
        }
        (target_records if records is None else records).append(record)
        if not target_attempted or target_status in {
            "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
        }:
            return target_status
        if feedback is not None and not reference_feedback_mode:
            target_feedback_scores[candidate.sha256] = feedback
        return target_status

    def reference_equal(base: ProgramVariant, candidate: ProgramVariant) -> bool | str:
        base_sha256 = program_sha256(base)
        key = (base_sha256, candidate.sha256)
        reference_index = reference_cache.get(key)
        cached = reference_index is not None
        if reference_index is None:
            baseline_observations = (
                baseline_record.get("observation")
                if baseline_record is not None and base_sha256 == original_sha256
                else None
            )
            result = run_emi_pair(
                base,
                candidate,
                builder,
                reference_runner,
                None if reference_feedback_mode else target_runner,
                work_dir=root / "reference" / f"{len(references):06d}",
                expected_parent_sha256=base_sha256,
                reference_observations=baseline_observations,
                expected_target_identity=binding,
                bare_builder=bare_builder,
                artifact_cache=artifact_cache,
                campaign_deadline_monotonic=active_target_deadline(),
            )
            reference_index = len(references)
            references.append(result)
            reference_cache[key] = reference_index
        else:
            result = references[reference_index]
        reference = result.get("reference", {})
        reference_status = (
            reference.get("status") if isinstance(reference, Mapping) else None
        )
        if reference_feedback_mode:
            reference_observation_map = (
                reference.get("observations") if isinstance(reference, Mapping) else None
            )
            variant_observations = (
                reference_observation_map.get("variant")
                if isinstance(reference_observation_map, Mapping) else None
            )
            path_feedback = reference_path_feedback(variant_observations, candidate)
            if reference_status == "non-equivalent":
                path_feedback = {
                    **path_feedback, "status": "semantic-rejected", "score": None,
                }
            else:
                path_feedback = reference_feedback_records.setdefault(
                    candidate.sha256, path_feedback,
                )
                reference_feedback_scores[candidate.sha256] = path_feedback.get("score")
            if isinstance(result, dict):
                result["reference_path_feedback"] = path_feedback
            if has_target_replay:
                if not isinstance(reference_observation_map, Mapping):
                    reference_observation_map = {}
                base_sha256 = program_sha256(base)
                pending_target = {
                    "base": base,
                    "candidate": candidate,
                    "reference_index": reference_index,
                    "reference_observations": reference_observation_map,
                    "base_sha256": base_sha256,
                    # The reference path already built both projections. The
                    # The campaign Target worker must consume those exact
                    # artifacts; rebuilding them in every Target consumer would
                    # merely move the old per-case cost to another process.
                    "artifact_records": result.get("artifacts", {}),
                }
                if artifact_gap := _artifact_gap_reason(
                    pending_target["artifact_records"]
                ):
                    pending_target["artifact_gap"] = artifact_gap
                pending_target_pairs.append(pending_target)
                queue_candidate_target_pair(pending_target)
        if reference_status != "equivalent":
            if reference_feedback_mode:
                # Keep semantic mismatch in the reference record, but do not
                # turn it into a search reward. A domain gap remains distinct.
                if reference_status == "non-equivalent":
                    return "reference-rejected"
                if (
                    reference_status == "reference-gap"
                    and result.get("status") == "reference-rejected"
                ):
                    return "reference-gap"
                return result["status"]
            record_coverage_only_target(result, reference_index, candidate)
            # Compatibility campaigns may keep a target-tested candidate in
            # the chain when their local reference cannot model its ISA. The
            # Direct/Program-Full reference-driven path has already returned
            # above and never uses Target feedback here.
            if open_capability and result.get("status") in {
                "reference-rejected", "profile-gap", "reference-gap",
            }:
                target = result.get("target")
                target_status = target.get("status") if isinstance(target, Mapping) else None
                if target_status not in {
                    "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
                }:
                    return "profile-gap"
            return result["status"]
        if reference_feedback_mode:
            return True
        target_result = (
            rerun_target(result, base, candidate, base_sha256)
            if cached and target_runner is not None else result
        )
        if target_runner is not None:
            target_status = record_target_result(
                target_result, reference_index, candidate,
            )
            if target_status in {
                "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
            }:
                return target_status
        return True

    def refresh(candidate: ProgramVariant) -> object:
        result = next(
            item for item in reversed(references)
            if (item["artifacts"].get("reference_variant", {}).get("source_sha256") == candidate.sha256
                and item["artifacts"].get("reference_variant", {}).get("parent_sha256") == candidate.parent_sha256)
        )
        record = result["artifacts"]["reference_variant"]
        artifact = BuiltProgram(
            Path(record["source_path"]),
            Path(record["executable_path"]),
            record["source_sha256"],
            record["executable_sha256"],
            record.get("parent_sha256"),
            record.get("run_params", {}),
            Path(record["bare_executable_path"])
            if record.get("bare_executable_path") else None,
            record.get("bare_executable_sha256"),
            (
                record.get("materialization", {})
                .get("bare_metal", {})
                .get("status", "not-requested")
                if isinstance(record.get("materialization"), Mapping)
                and isinstance(record.get("materialization", {}).get("bare_metal"), Mapping)
                else "not-requested"
            ),
            (
                record.get("materialization", {})
                .get("bare_metal", {})
                .get("reason")
                if isinstance(record.get("materialization"), Mapping)
                and isinstance(record.get("materialization", {}).get("bare_metal"), Mapping)
                else None
            ),
            Path(record["linux_executable_path"])
            if record.get("linux_executable_path") else None,
            record.get("linux_executable_sha256"),
            (
                record.get("materialization", {})
                .get("linux_user", {})
                .get("status", "not-requested")
                if isinstance(record.get("materialization"), Mapping)
                and isinstance(record.get("materialization", {}).get("linux_user"), Mapping)
                else "not-requested"
            ),
            (
                record.get("materialization", {})
                .get("linux_user", {})
                .get("reason")
                if isinstance(record.get("materialization"), Mapping)
                and isinstance(record.get("materialization", {}).get("linux_user"), Mapping)
                else None
            ),
        )
        observations = _observations(result["reference"]["observations"]["variant"])
        profile_program = candidate
        include_directory = None
        if raw_includes:
            include_directory = TemporaryDirectory()
            try:
                directory = Path(include_directory.name)
                path = directory / "candidate.S"
                path.write_bytes(candidate.source.encode("utf-8"))
                for name, content in raw_includes.items():
                    include = directory / name
                    include.parent.mkdir(parents=True, exist_ok=True)
                    include.write_bytes(
                        bytes.fromhex(content[len(_SOURCE_BINARY_PREFIX):])
                        if content.startswith(_SOURCE_BINARY_PREFIX)
                        else content.encode("utf-8")
                    )
                source_program = CaseProgram(path, artifact.run_params)
                if program_sha256(source_program) == candidate.sha256:
                    profile_program = source_program
            except (OSError, TypeError, ValueError):
                pass
        try:
            return _campaign_contract.attach_execution_identity(
                route.build_profile(
                    profile_program,
                    _profile_observations(observations, artifact.executable_path),
                    input_identity=_campaign_contract.profile_identity(artifact, observations),
                    **({"expected_trap": expected_trap} if route.name == "program" else {}),
                ),
                artifact,
                observations,
            )
        finally:
            if include_directory is not None:
                include_directory.cleanup()

    def online_profile_actions(
        current: ProgramVariant, active_profile: object,
    ) -> tuple[dict[str, object], ...]:
        actions = tuple(route.profile_actions(current, active_profile))
        toggle = (
            _model_toggle_action(
                current, active_profile, rewrite_record.get("hint"),
                raw_sha256=raw_sha256, rewritten_sha256=original_sha256,
            )
            if route.name == "program" and rewrite_record.get("mode") == "model-on"
            else None
        )
        return ((toggle,) if toggle is not None else ()) + actions

    chain = sample_program(
        original, reference_equal,
        steps=steps,
        seed=seed,
        beta=beta,
        profile=profile,
        refresh_profile=refresh if profile is not None else None,
        mcmc=mcmc,
        emi_enabled=emi_enabled,
        allowed_rule=bound_rule,
        profile_actions_fn=online_profile_actions,
        # 没有 profile 时用 route 自己的候选枚举，MCMC 才不会空池。
        enumerate_rewrites_fn=route.enumerate_rewrites,
        apply_profile_fn=route.apply_profile,
        feedback=(feedback_score if reference_feedback_mode or has_target_replay else None),
        feedback_source="reference" if reference_feedback_mode else "target",
        allow_reference_gaps=open_capability,
        max_seconds=(
            None if campaign_deadline is None
            else max(0.0, campaign_deadline - time.monotonic())
        ),
    )
    chain["feedback_source"] = "reference" if reference_feedback_mode else "target"
    chain["emi_enabled"] = emi_enabled
    chain["target_execution_mode"] = (
        "asynchronous-target-queue" if target_queue_only else
        "post-chain-replay" if reference_feedback_mode and has_target_replay
        else "reference-only" if reference_feedback_mode else "synchronous"
    )

    def link_reference_events(events: Sequence[dict[str, object]]) -> None:
        for event in events:
            event["reference_index"] = event["reference"] = None
            event["target_available"] = has_target_replay
            candidate_sha256 = event.get("candidate_sha256")
            parent_sha256 = event.get("parent_sha256")
            if candidate_sha256 is None or event.get("gate") == "parameter-gap":
                continue
            index = reference_cache.get((parent_sha256, candidate_sha256))
            if index is not None:
                event["reference_index"] = index
                event["reference"] = {"reference_index": index}

    def publish_chain_checkpoint(stage: str) -> None:
        if chain_checkpoint_path is None:
            return
        events = [dict(event) for event in chain["events"]]
        link_reference_events(events)
        primary_records = [dict(record) for record in target_records]
        target_records_snapshot = {
            target_id: [dict(record) for record in records]
            for target_id, records in target_records_by_target.items()
        }
        target_baselines_snapshot = {
            target_id: dict(record)
            for target_id, record in target_baselines_by_target.items()
        }
        for event in events:
            if target_replay_runners:
                target_links: dict[str, dict[str, int]] = {}
                target_event_fields: dict[str, dict[str, object]] = {}
                for target_id, records in target_records_snapshot.items():
                    record_index = next(
                        (index for index, item in enumerate(records)
                         if item.get("step") is None
                         and item.get("state_sha256") == event.get("candidate_sha256")
                         and item.get("parent_sha256") == event.get("parent_sha256")),
                        None,
                    )
                    if record_index is None:
                        continue
                    target_record = records[record_index]
                    target_record["step"] = event["step"]
                    target_links[target_id] = {"target_record_index": record_index}
                    target_event_fields[target_id] = _target_event_fields(target_record)
                event["targets"] = target_links
                event["target_observations"] = target_event_fields
                event["target"] = None
            else:
                record_index = next(
                    (index for index, item in enumerate(primary_records)
                     if item.get("step") is None
                     and item.get("state_sha256") == event.get("candidate_sha256")
                     and item.get("parent_sha256") == event.get("parent_sha256")),
                    None,
                )
                if record_index is None:
                    event["target"] = None
                else:
                    target_record = primary_records[record_index]
                    target_record["step"] = event["step"]
                    event["target"] = {"target_record_index": record_index}
                    event.update(_target_event_fields(target_record))

        all_target_records = (
            [record for target_id in target_replay_runners
             for record in target_records_snapshot[target_id]]
            if target_replay_runners else primary_records
        )
        counts = campaign_counts(
            events, chain["run"], all_target_records, target_baseline,
            target_baselines_snapshot,
        )
        if stage == "reference-chain-closed":
            persist_target_replay_progress(
                counts, target_progress_sequence, rebuild=True,
            )
        ledger_path = write_ledger(events, counts)
        target_ids = list(queued_target_ids)
        pairs = []
        used_events: set[int] = set()
        for pair_index, pending_target in enumerate(pending_target_pairs):
            candidate = pending_target["candidate"]
            reference_index = int(pending_target["reference_index"])
            match = next(
                (index for index, event in enumerate(events)
                 if index not in used_events
                 and event.get("candidate_sha256") == candidate.sha256
                 and event.get("parent_sha256") == pending_target["base_sha256"]
                 and event.get("reference_index") == reference_index),
                None,
            )
            if match is not None:
                used_events.add(match)
            step = events[match].get("step") if match is not None else None
            pending_target["step"] = step
            if pair_index < len(target_queue_jobs):
                target_queue_jobs[pair_index]["step"] = step
            pairs.append({
                "pair_index": pair_index,
                "reference_index": reference_index,
                "state_sha256": candidate.sha256,
                "parent_sha256": pending_target["base_sha256"],
                "step": step,
                **({"queue_job_path": pending_target["job_path"]}
                   if isinstance(pending_target.get("job_path"), str) else {}),
            })
        baseline_target_ids = (
            target_ids if deferred_target_baseline is not None else []
        )
        final_profile = chain.get("profile")
        profile_record = (
            {"status": "ready", **final_profile.to_record()}
            if final_profile is not None else {"status": "not-provided"}
        )
        checkpoint_status = (
            "target-gap" if counts.get("target_gap", 0) else
            "reference-valid" if counts.get("reference_valid", 0) else
            "reference-gap" if counts.get("reference_gap", 0) else
            "reference-rejected" if counts.get("reference_rejected", 0) else
            "profiled"
        )
        campaign_record = _json_safe({
            "status": checkpoint_status,
            "original_sha256": original_sha256,
            "raw_sha256": raw_sha256,
            "root_sha256": rewrite_record.get("root_sha256"),
            "root_digest": rewrite_record.get("root_digest"),
            "root": rewrite_record,
            **({"root_binding": binding} if binding is not None else {}),
            **({"target_bindings": target_bindings} if target_bindings else {}),
            "beta": beta, "mcmc": mcmc,
            "emi_enabled": emi_enabled,
            "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
            "reference_path_reward_version": reference_path_reward_version,
            "mcmc_chain_mode": (
                "reference-only" if reference_feedback_mode else "target-coupled"
            ),
            "target_execution_mode": (
                "asynchronous-target-queue" if target_queue_only else
                "post-chain-replay" if reference_feedback_mode and has_target_replay
                else "reference-only" if reference_feedback_mode else "synchronous"
            ),
            "target_replay": {
                "mode": "queue" if target_queue_only else
                "post-chain" if reference_feedback_mode and has_target_replay
                else "not-requested",
                "status": "queued" if stage == "reference-chain-closed" else "in-progress",
                "target_ids": target_ids,
                "pending_pair_count": len(pairs),
                "record_count": sum(map(len, target_records_snapshot.values()))
                if target_replay_runners else len(primary_records),
                "record_count_by_target": {
                    target_id: len(records)
                    for target_id, records in target_records_snapshot.items()
                },
                "pending_execution_count": (
                    None if target_queue_only else target_queue_pending_count()
                ),
                "target_replay_budget_seconds": (
                    None if target_queue_only else target_replay_budget_value
                ),
                "target_queue_deadline_policy": (
                    "per-target-timeout-only" if target_queue_only
                    else "campaign-target-replay-budget"
                ),
                "publish_gap_count": len(target_publish_gaps),
                "publish_gaps": list(target_publish_gaps),
                **({"queue_path": str(target_queue_path)}
                   if target_queue_path is not None else {}),
            },
            "reference_chain_steps": len(events),
            "step_budget": steps,
            "steps_per_case_budget": steps,
            "total_steps_completed": len(events),
            "steps_completed": len(events),
            "seed": seed,
            "rng_scheme": _RNG_SCHEME,
            "search_mode": "mcmc" if mcmc else "direct",
            "target_timeout_seconds": target_timeout_seconds,
            "profile": profile_record,
            "reference_baseline": baseline_record,
            **({"reference_attempts": reference_attempts} if reference_attempts else {}),
            "chain": _compact_chain_record({**chain, "events": events}),
            "reference": references,
            "target": primary_records if not target_replay_runners else all_target_records,
            "target_records_by_target": target_records_snapshot,
            "target_baseline": target_baseline,
            "target_baselines_by_target": target_baselines_snapshot,
            "target_publish_gaps": list(target_publish_gaps),
            "counts": counts,
            "ledger_path": str(ledger_path),
            **({"target_replay_queue_path": str(target_queue_path)}
               if target_queue_path is not None else {}),
        })
        if target_queue_path is not None:
            queue_status = (
                "chain-closed" if stage == "reference-chain-closed"
                else "target-replay-progress"
            )
            persist_target_queue(queue_status)
        atomic_write_json(chain_checkpoint_path, {
            "schema_version": "rq1-framework-chain-checkpoint-v1",
            "checkpoint_stage": stage,
            "campaign": campaign_record,
            "target_replay_plan": {
                "mode": "queued" if target_queue_only else
                "mapped" if target_replay_runners else
                "single" if target_runner is not None else "none",
                "target_ids": target_ids,
                "baseline_target_ids": baseline_target_ids,
                **({"baseline_queue_job_path": target_queue_baseline["job_path"]}
                   if target_queue_baseline is not None else {}),
                **({"queue_path": str(target_queue_path)}
                   if target_queue_path is not None else {}),
                "pairs": pairs,
            },
        })

    link_reference_events(chain["events"])
    if reference_feedback_mode and has_target_replay:
        target_replay_started = True
        if target_queue_only:
            # Queue-only consumers are independent of this producer's wall
            # clock. Their only execution limit is the Target service's job timeout;
            # a campaign-wide replay deadline would silently censor a slow
            # Target and recreate the old cross-Target coupling.
            target_replay_deadline = None
            target_replay_deadline_epoch = None
        else:
            target_replay_deadline = (
                None if target_replay_budget_value is None
                else time.monotonic() + target_replay_budget_value
            )
            target_replay_deadline_epoch = (
                None if target_replay_budget_value is None
                else time.time() + target_replay_budget_value
            )
    publish_chain_checkpoint("reference-chain-closed")
    # Reference-driven Ours 的 MH 链结束后才进入 Target 阶段。正式 queue-only 模式只发布
    # durable jobs；Target 执行不参与 proposal 选择，也不会让下一步 candidate 等待模拟器。
    # 下面的 inline 分支仅服务直接 CLI 的兼容路径。
    if not target_queue_only and reference_feedback_mode and target_runner is not None:
        single_target_id = (
            str(binding.get("target_id") or binding.get("target") or "target")
            if isinstance(binding, Mapping) else "target"
        )
        if deferred_target_baseline is not None:
            baseline_artifact, baseline_reference, baseline_expected_trap = (
                deferred_target_baseline
            )
            queue_target_state("baseline", single_target_id, "running")
            capture_target_baseline(
                baseline_artifact,
                baseline_record if baseline_record is not None else baseline_reference,
                baseline_expected_trap,
                defer_target=False,
            )
            if isinstance(target_baseline, Mapping):
                persist_target_replay_record(
                    single_target_id, None, target_baseline, baseline=True,
                )
            queue_target_state(
                "baseline", single_target_id,
                "pending" if isinstance(target_baseline, Mapping)
                and target_baseline.get("deadline_censored") else "complete",
            )

        def replay_single_target(pair_index: int, pending_target: Mapping[str, object]):
            queue_target_state(str(pending_target["job_id"]), single_target_id, "running")
            base = pending_target["base"]
            candidate = pending_target["candidate"]
            reference_observation_map = pending_target["reference_observations"]
            base_sha256 = pending_target["base_sha256"]

            def cached_reference(artifact: BuiltProgram):
                side = (
                    "original"
                    if artifact.source_sha256 == base_sha256 else "variant"
                )
                return reference_observation_map.get(side, ())

            return run_emi_pair(
                base,
                candidate,
                builder,
                cached_reference,
                target_runner,
                work_dir=root / "target-replay" / f"{pair_index:06d}",
                expected_parent_sha256=base_sha256,
                reference_observations=reference_observation_map.get("original", ()),
                expected_target_identity=binding,
                bare_builder=bare_builder,
                artifact_cache=artifact_cache,
                campaign_deadline_monotonic=active_target_deadline(),
            )

        target_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="direct-target-replay",
        )
        try:
            for pair_index, pending_target in enumerate(pending_target_pairs):
                deadline = active_target_deadline()
                if deadline is not None and time.monotonic() >= deadline:
                    break
                future = target_pool.submit(
                    replay_single_target, pair_index, pending_target,
                )
                try:
                    target_result = future.result()
                except Exception as error:
                    deadline_censored = (
                        deadline is not None and time.monotonic() >= deadline
                    )
                    target_result = {
                        "status": "target-gap",
                        "failure_class": (
                            "campaign-wall-clock-exhausted"
                            if deadline_censored else "transport-gap"
                        ),
                        "target": {
                            "status": "target-gap",
                            "differences": {"target_error": str(error)},
                        },
                        **({"deadline_censored": True} if deadline_censored else {}),
                    }
                if isinstance(target_result, Mapping):
                    record_target_result(
                        target_result,
                        int(pending_target["reference_index"]),
                        pending_target["candidate"],
                    )
                    recorded = target_records[-1]
                    persist_target_replay_record(
                        single_target_id, int(pending_target["pair_index"]), recorded,
                    )
                    queue_target_state(
                        str(pending_target["job_id"]), single_target_id,
                        "pending" if recorded.get("deadline_censored") else "complete",
                    )
                    release_pending_target_payload(pending_target)
        finally:
            target_pool.shutdown(wait=True, cancel_futures=True)
    elif not target_queue_only and reference_feedback_mode and target_replay_runners:
        baseline_artifact = (
            deferred_target_baseline[0]
            if deferred_target_baseline is not None else None
        )
        reference_baseline = (
            baseline_record
            if baseline_record is not None else
            deferred_target_baseline[1]
            if deferred_target_baseline is not None else
            {"observation": (), "reference_identity": {}}
        )

        def target_observation_key(program: object) -> tuple[str, str]:
            return (
                program.source_sha256 if isinstance(program, BuiltProgram)
                else program_sha256(program),
                canonical_digest(dict(getattr(program, "run_params", {}) or {})),
            )

        def reusable_target_observations(
            record: Mapping[str, object], observations: object,
            *, valid_statuses: tuple[str, ...],
        ) -> tuple[object, ...] | None:
            rows = _observations(observations)
            if (
                record.get("status") not in valid_statuses
                or record.get("validation_gap")
                or record.get("deadline_censored")
                or not rows
                or any(
                    _object_field(item, "outcome") in {"timeout", "unavailable"}
                    or _object_field(item, "contract_error")
                    or _object_field(item, "validation_gap")
                    for item in rows
                )
            ):
                return None
            return rows

        target_observation_cache: dict[
            str, dict[tuple[str, str], tuple[object, ...]],
        ] = {target_id: {} for target_id in target_replay_runners}
        # A cached observation is only needed while a later pair uses that
        # program as its base.  Count those uses up front so completed target
        # records can release raw simulator payloads immediately after their
        # sidecar is durable.  This keeps the cache from becoming a second
        # copy of the full campaign evidence.
        target_observation_uses = {
            target_id: Counter(
                target_observation_key(pending_target["base"])
                for pending_target in pending_target_pairs
            )
            for target_id in target_replay_runners
        }

        if baseline_artifact is not None:
            def evaluate_one_baseline(target_id: str):
                queue_target_state("baseline", target_id, "running")
                runner = target_replay_runners[target_id]
                target_binding = target_bindings[target_id]
                reference_ready = bool(_observations(reference_baseline.get("observation", ())))
                result = mark_coverage_only_baseline(
                    evaluate_target_baseline(
                        baseline_artifact, runner,
                        baseline_record=(
                            reference_baseline if reference_ready
                            else {"observation": (), "reference_identity": {}}
                        ),
                        binding=target_binding,
                        observer_fields=comparison_fields,
                        expected_trap=expected_trap,
                    )
                )
                fallback_record = reference_baseline.get("reference_fallback")
                if (
                    isinstance(fallback_record, Mapping)
                    and _same_execution_backend(
                        fallback_record.get("effective_backend"),
                        target_binding.get("target"),
                    )
                ):
                    result["reference_target_relation"] = "shared-qemu-fallback"
                if _campaign_deadline_exhausted(active_target_deadline(), result):
                    result = {
                        **result,
                        "failure_class": "campaign-wall-clock-exhausted",
                        "deadline_censored": True,
                    }
                observations = reusable_target_observations(
                    result, result.get("observations"),
                    valid_statuses=("target-tested",),
                )
                if observations is not None:
                    baseline_key = target_observation_key(baseline_artifact)
                    if target_observation_uses[target_id].get(baseline_key, 0):
                        target_observation_cache[target_id][baseline_key] = observations
                return target_id, result

        def replay_one(
            target_id: str, pair_index: int, pending_target: Mapping[str, object],
        ):
            queue_target_state(str(pending_target["job_id"]), target_id, "running")
            base = pending_target["base"]
            candidate = pending_target["candidate"]
            reference_observation_map = pending_target["reference_observations"]
            base_sha256 = pending_target["base_sha256"]

            def cached_reference(artifact: BuiltProgram):
                side = (
                    "original"
                    if artifact.source_sha256 == base_sha256 else "variant"
                )
                return reference_observation_map.get(side, ())

            cache = target_observation_cache[target_id]
            base_key = target_observation_key(base)
            remaining_uses = target_observation_uses[target_id]
            original_observations = cache.get(base_key)
            # The local variable keeps this pair's input alive.  Removing the
            # cache entry now ensures a base used for the last time cannot
            # retain its raw observations while later pairs run.
            if remaining_uses.get(base_key, 0) <= 1:
                remaining_uses.pop(base_key, None)
                cache.pop(base_key, None)
            else:
                remaining_uses[base_key] -= 1
            result = run_emi_pair(
                base,
                candidate,
                builder,
                cached_reference,
                target_replay_runners[target_id],
                work_dir=(root / "target-replay" / target_id
                          / f"{pair_index:06d}"),
                expected_parent_sha256=base_sha256,
                reference_observations=reference_observation_map.get("original", ()),
                expected_target_identity=target_bindings[target_id],
                target_original_observations=original_observations,
                bare_builder=bare_builder,
                artifact_cache=artifact_cache,
                campaign_deadline_monotonic=active_target_deadline(),
            )
            target = result.get("target")
            variant_observations = (
                target.get("observations", {}).get("variant", ())
                if isinstance(target, Mapping)
                and isinstance(target.get("observations"), Mapping)
                else ()
            )
            reusable = reusable_target_observations(
                target if isinstance(target, Mapping) else {},
                variant_observations,
                valid_statuses=("equivalent", "non-equivalent", "reference-gap", "case-skipped"),
            )
            if reusable is not None:
                candidate_key = target_observation_key(candidate)
                if remaining_uses.get(candidate_key, 0):
                    cache[candidate_key] = reusable
            return target_id, result

        target_pools = {
            target_id: ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"reference-target-{target_id}",
            )
            for target_id in target_replay_runners
        }
        pending_pair_remaining = {
            pair_index: len(target_replay_runners)
            for pair_index in range(len(pending_target_pairs))
        }
        # Keep one FIFO stream per target, but only keep one job in flight.
        # This is the small fix that gives replay backpressure and makes the
        # campaign deadline useful: there is no large executor backlog to
        # drain after the deadline.
        next_pair_by_target = {
            target_id: 0 for target_id in target_replay_runners
        }
        active_futures: dict[
            object, tuple[str, str, Mapping[str, object] | None]
        ] = {}

        def replay_open() -> bool:
            deadline = active_target_deadline()
            return deadline is None or time.monotonic() < deadline

        def submit_pair(target_id: str) -> None:
            if not replay_open():
                return
            pair_index = next_pair_by_target[target_id]
            if pair_index >= len(pending_target_pairs):
                return
            pending_target = pending_target_pairs[pair_index]
            next_pair_by_target[target_id] = pair_index + 1
            future = target_pools[target_id].submit(
                replay_one, target_id, pair_index, pending_target,
            )
            active_futures[future] = ("candidate", target_id, pending_target)

        try:
            if baseline_artifact is not None:
                for target_id, target_pool in target_pools.items():
                    if not replay_open():
                        break
                    future = target_pool.submit(evaluate_one_baseline, target_id)
                    active_futures[future] = ("baseline", target_id, None)
            else:
                for target_id in target_replay_runners:
                    submit_pair(target_id)

            while active_futures:
                done, _ = wait(
                    tuple(active_futures), return_when=FIRST_COMPLETED,
                )
                for future in done:
                    kind, target_id, pending_target = active_futures.pop(future)
                    if kind == "baseline":
                        if baseline_artifact is None:
                            continue
                        target_deadline = active_target_deadline()
                        try:
                            _, result = future.result()
                        except Exception as error:
                            deadline_censored = (
                                target_deadline is not None
                                and time.monotonic() >= target_deadline
                            )
                            result = {
                                "status": "target-gap",
                                "source_sha256": baseline_artifact.source_sha256,
                                "executable_sha256": baseline_artifact.executable_sha256,
                                "observations": [],
                                "comparison": {
                                    "status": "target-gap",
                                    "comparison_mode": "cross-backend",
                                    "differences": {"target_error": str(error)},
                                },
                                "failure_class": (
                                    "campaign-wall-clock-exhausted"
                                    if deadline_censored else "transport-gap"
                                ),
                                **({"deadline_censored": True}
                                   if deadline_censored else {}),
                            }
                        result["target_id"] = target_id
                        target_baselines_by_target[target_id] = result
                        persist_target_replay_record(
                            target_id, None, result, baseline=True,
                        )
                        queue_target_state(
                            "baseline", target_id,
                            "pending" if result.get("deadline_censored")
                            else "complete",
                        )
                        submit_pair(target_id)
                        if not replay_open():
                            for pending_future in active_futures:
                                pending_future.cancel()
                            break
                        continue

                    if pending_target is None:
                        continue
                    target_deadline = active_target_deadline()
                    try:
                        _, target_result = future.result()
                    except Exception as error:
                        deadline_censored = (
                            target_deadline is not None
                            and time.monotonic() >= target_deadline
                        )
                        target_result = {
                            "status": "target-gap",
                            "failure_class": (
                                "campaign-wall-clock-exhausted"
                                if deadline_censored else "transport-gap"
                            ),
                            "target": {
                                "status": "target-gap",
                                "differences": {"target_error": str(error)},
                            },
                            **({"deadline_censored": True} if deadline_censored else {}),
                        }
                    if isinstance(target_result, Mapping):
                        record_target_result(
                            target_result,
                            int(pending_target["reference_index"]),
                            pending_target["candidate"],
                            target_id=target_id,
                            records=target_records_by_target[target_id],
                        )
                        recorded = target_records_by_target[target_id][-1]
                        persist_target_replay_record(
                            target_id, int(pending_target["pair_index"]), recorded,
                        )
                        queue_target_state(
                            str(pending_target["job_id"]), target_id,
                            "pending" if recorded.get("deadline_censored") else "complete",
                        )
                        pair_index = int(pending_target["pair_index"])
                        pending_pair_remaining[pair_index] -= 1
                        if pending_pair_remaining[pair_index] <= 0:
                            release_pending_target_payload(pending_target)
                        submit_pair(target_id)
                if not replay_open():
                    for future in active_futures:
                        future.cancel()
                    break
        finally:
            for target_pool in target_pools.values():
                target_pool.shutdown(wait=True, cancel_futures=True)
        target_records.extend(
            record
            for target_id in target_replay_runners
            for record in target_records_by_target[target_id]
        )
    # Candidate/source/reference payloads have already been persisted in the
    # replay jobs and consumed by the per-Target queue workers. Keep only queue identity
    # fields for the final campaign summary.
    for pending_target in pending_target_pairs:
        release_pending_target_payload(pending_target)
    for event in chain["events"]:
        event["reference_index"] = event["reference"] = None
        event["target_available"] = has_target_replay
        if event.get("candidate_sha256") is not None and event.get("gate") != "parameter-gap":
            index = reference_cache.get(
                (event["parent_sha256"], event["candidate_sha256"])
            )
            if index is not None:
                event["reference_index"] = index
                event["reference"] = {"reference_index": index}
        if target_replay_runners:
            target_links = {}
            target_event_fields = {}
            for target_id, records in target_records_by_target.items():
                record_index = next(
                    (index for index, item in enumerate(records)
                     if item["step"] is None
                     and item["state_sha256"] == event.get("candidate_sha256")
                     and item.get("parent_sha256") == event.get("parent_sha256")),
                    None,
                )
                if record_index is None:
                    continue
                target_record = records[record_index]
                target_record["step"] = event["step"]
                target_links[target_id] = {"target_record_index": record_index}
                target_event_fields[target_id] = _target_event_fields(target_record)
            event["targets"] = target_links
            event["target_observations"] = target_event_fields
            event["target"] = None
        else:
            record_index = next(
                (index for index, item in enumerate(target_records)
                 if item["step"] is None
                 and item["state_sha256"] == event.get("candidate_sha256")
                 and item.get("parent_sha256") == event.get("parent_sha256")),
                None,
            )
            if record_index is None:
                event["target"] = None
            else:
                target_record = target_records[record_index]
                target_record["step"] = event["step"]
                event["target"] = {"target_record_index": record_index}
                event.update(_target_event_fields(target_record))
    events = chain["events"]
    counts = campaign_counts(
        events, chain["run"], target_records, target_baseline,
        target_baselines_by_target,
    )
    final_profile = chain.get("profile")
    profile_record = (
        {"status": "ready", **final_profile.to_record()}
        if final_profile is not None else {"status": "not-provided"}
    )
    campaign_status = (
        "proposal-gap" if counts.get("proposal_gap", 0) else
        "artifact-gap" if counts.get("artifact_gap", 0) else
        "target-gap" if counts.get("target_gap", 0) else
        "target-mismatch-candidate" if counts.get("target_mismatch_candidate", 0) else
        "case-skipped"
        if counts.get("case_skipped", 0)
        and not counts.get("accepted", 0)
        and not counts.get("clean", 0) else
        "profile-gap" if counts.get("profile_gap", 0) or counts.get("refresh_gap", 0) else
        "seed-miss" if counts.get("seed_miss", 0) else
        "target-probe-gap" if probe_mode and (
            isinstance(target_baseline, Mapping)
            and target_baseline.get("status") == "target-probe-gap"
            or any(item.get("status") == "target-probe-gap" for item in target_records)
        ) else
        "target-probe-tested" if probe_mode and isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "target-probe-tested" else
        # 生成-only 没有 target，那是执行阶段的分工，不是失败。
        # 当成状态会让 runner 丢掉整批候选；reference 有效就是 reference-valid。
        "reference-valid" if not has_target_replay and counts.get("reference_valid", 0) else
        "clean" if counts.get("clean", 0) else
        "target-tested" if isinstance(target_baseline, Mapping)
        and target_baseline.get("status") == "target-tested" else
        "reference-valid" if counts.get("reference_valid", 0) else
        "transport-gap" if counts.get("transport_gap", 0) else
        "reference-gap" if counts.get("reference_gap", 0) else
        "reference-rejected" if counts.get("reference_rejected", 0) else
        "emi-rejected" if counts.get("emi_rejected", 0) else
        "parameter-gap" if counts.get("parameter_gap", 0) else "profiled"
    )
    campaign_reason = None
    for item in target_records:
        if not isinstance(item, Mapping) or item.get("status") not in {
            "target-gap", "target-probe-gap", "case-skipped", "artifact-gap",
        } or item.get("deadline_censored") is True:
            continue
        value = item.get("failure_class") or item.get("target_result_status")
        if isinstance(value, str) and value:
            campaign_reason = value
            break
    if campaign_reason is None:
        for baseline in target_baselines_by_target.values():
            if not isinstance(baseline, Mapping) or baseline.get("status") not in {
                "target-gap", "target-probe-gap", "transport-gap",
                "case-skipped", "artifact-gap",
            }:
                continue
            if baseline.get("deadline_censored") is True:
                campaign_reason = "campaign-wall-clock-exhausted"
            else:
                value = baseline.get("failure_class") or baseline.get("reason") \
                    or baseline.get("status")
                if isinstance(value, str) and value:
                    campaign_reason = value
            if campaign_reason is not None:
                break
    if campaign_reason is None and isinstance(target_baseline, Mapping):
        if target_baseline.get("status") in {
            "target-gap", "target-probe-gap", "transport-gap",
            "case-skipped", "artifact-gap",
        }:
            if target_baseline.get("deadline_censored") is True:
                campaign_reason = "campaign-wall-clock-exhausted"
            else:
                value = target_baseline.get("failure_class") or target_baseline.get(
                    "reason"
                ) or target_baseline.get("status")
                if isinstance(value, str) and value:
                    campaign_reason = value
    if campaign_reason is None and counts.get("target_wall_clock_pending", 0):
        campaign_reason = "campaign-wall-clock-exhausted"
    queue_pending = target_queue_pending_count()
    if target_queue_path is not None and not target_queue_only:
        persist_target_queue("pending" if queue_pending else "complete")
    ledger_path = write_ledger(chain["events"], counts)
    return {
        "status": campaign_status,
        **({"reason": campaign_reason} if campaign_reason else {}),
        "original_sha256": original_sha256,
        "raw_sha256": raw_sha256,
        "root_sha256": rewrite_record.get("root_sha256"),
        "root_digest": rewrite_record.get("root_digest"),
        "root": rewrite_record,
        **({"root_binding": binding} if binding is not None else {}),
        **({"target_bindings": target_bindings} if target_bindings else {}),
        "beta": beta, "mcmc": mcmc,
        "emi_enabled": emi_enabled,
        "mcmc_feedback_source": "reference" if reference_feedback_mode else "target",
        "reference_path_reward_version": reference_path_reward_version,
        "mcmc_chain_mode": (
            "reference-only" if reference_feedback_mode else "target-coupled"
        ),
        "target_execution_mode": (
            "asynchronous-target-queue" if target_queue_only else
            "post-chain-replay" if reference_feedback_mode and has_target_replay
            else "reference-only" if reference_feedback_mode else "synchronous"
        ),
        "target_replay": {
            "mode": (
                "queue" if target_queue_only else
                "post-chain" if reference_feedback_mode and has_target_replay
                else "inline" if has_target_replay else "not-requested"
            ),
            "status": (
                "queued" if target_queue_only else
                "pending" if queue_pending else
                "complete" if target_queue_path is not None else "not-requested"
            ),
            "target_ids": list(queued_target_ids),
            "pending_pair_count": len(pending_target_pairs),
            "pending_execution_count": (
                None if target_queue_only else queue_pending
            ),
            "target_replay_budget_seconds": (
                None if target_queue_only else target_replay_budget_value
            ),
            "target_queue_deadline_policy": (
                "per-target-timeout-only" if target_queue_only
                else "campaign-target-replay-budget"
            ),
            "publish_gap_count": len(target_publish_gaps),
            "publish_gaps": list(target_publish_gaps),
            **({"queue_path": str(target_queue_path)}
               if target_queue_path is not None else {}),
            "record_count": len(target_records),
            "record_count_by_target": {
                target_id: len(records)
                for target_id, records in target_records_by_target.items()
            },
        },
        "reference_chain_steps": len(chain["events"]),
        "step_budget": steps,
        "steps_per_case_budget": steps,
        "total_steps_completed": len(chain["events"]),
        "steps_completed": len(chain["events"]),
        "seed": seed,
        "rng_scheme": _RNG_SCHEME,
        "search_mode": "mcmc" if mcmc else "direct",
        "target_timeout_seconds": target_timeout_seconds,
        "target_replay_budget_seconds": (
            None if target_queue_only else target_replay_budget_value
        ),
        "profile": profile_record,
        "reference_baseline": baseline_record,
        **({"reference_attempts": reference_attempts} if reference_attempts else {}),
        "chain": chain,
        "reference": references,
        "target": target_records,
        "target_records_by_target": target_records_by_target,
        "target_baseline": target_baseline,
        "target_baselines_by_target": target_baselines_by_target,
        "target_publish_gaps": list(target_publish_gaps),
        "counts": counts,
        "ledger_path": str(ledger_path),
        **({"target_replay_queue_path": str(target_queue_path)}
           if target_queue_path is not None else {}),
    }


__all__ = ["run_program_campaign"]

"""Comparison observation contract and differential semantics."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from framework.execution_identity import is_allowlisted_patched_identity
from framework.observability import compare_fields
from framework.reference_fallback import (
    K1_BOARD_TQEMU_FALLBACK_POLICY, K1_QEMU_FALLBACK_FAILURE_CLASSES,
)


def _valid_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and bool(re.fullmatch(r"[0-9a-fA-F]+", value))


def _valid_commit(value: object) -> bool:
    return _valid_hex(value, 40) or _valid_hex(value, 64)


def _target_identity_pinned(value: object) -> bool:
    if not isinstance(value, dict) or str(value.get("status", "")).lower() in {
        "missing", "failed", "error", "invalid", "gap",
    }:
        return False
    if (value.get("patch_lineage") or value.get("source_provenance") in {
        "patch-validated-build", "verified-git-archive+controlled-patch",
    }) and not is_allowlisted_patched_identity(value):
        return False
    if "observed" in value:
        return _target_identity_pinned(value["observed"])
    if "configured" in value:
        return False
    digests = [
        value[key] for key in ("identity_digest", "target_identity_digest")
        if value.get(key) not in (None, "")
    ]
    if digests:
        return (
            all(isinstance(item, str) for item in digests)
            and len({item.lower() for item in digests}) == 1
            and _valid_hex(digests[0], 64)
        )
    return any(str(value.get(key, "")).strip() for key in (
        "version", "commit", "source_commit",
        "path", "container_image", "container_image_digest"))


def _target_identity_record(record: dict[str, Any]) -> object:
    """Prefer an explicitly emitted Target identity, including an empty one."""
    if "target_identity" in record:
        return record["target_identity"]
    return record.get("identity")


def _target_dispatch_event(target_id: str) -> dict[str, str]:
    return {"event_id": f"{target_id}:dispatch", "target": target_id, "status": "completed"}


def _process_executed(result: dict[str, Any]) -> bool:
    if type(result.get("target_attempted")) is bool:
        return result["target_attempted"]
    if type(result.get("process_executed")) is bool:
        return result["process_executed"]
    return (
        result.get("process_started") is True
        or
        isinstance(result.get("status"), str)
        and result.get("status") in {"passed", "failed", "timeout"}
        or result.get("terminal_observed") is True
    )


def _guest_terminal_evidence(result: dict[str, Any]) -> bool:
    """Return whether a non-normal terminal came from the guest contract."""
    if result.get("terminal_observed") is not True:
        return False
    if result.get("guest_trap") == "delivered" or type(result.get("guest_cause")) is int:
        return True
    if result.get("guest_exit_channel") or (
            isinstance(result.get("semantic_outcome"), str)
            and result.get("semantic_outcome") in {
                "trap", "expected-trap", "crash", "nonzero-exit",
            }):
        return True
    if isinstance(result.get("termination"), str) and result.get("termination") in {
        "guest-tohost", "guest-terminal-after-timeout", "guest-trap",
        "guest-exception", "guest-ebreak", "guest-ecall", "guest-natural-stop",
        "simulator-fuel",
    }:
        return True
    return isinstance(result.get("reason"), str) and result.get("reason") in {
        "guest-exception", "rax-guest-trap", "guest-tohost-failure",
    }


def _result_target_outcome(result: dict[str, Any]) -> str | None:
    # ``termination=trace-limit`` is often an adapter label for a trace
    # collector that stopped after the guest already emitted its state.  Only
    # an explicit limit/budget marker without ``after_exit`` censors the
    # terminal outcome; trace completeness itself remains a coverage fact.
    trace_interrupted = (
        result.get("trace_limit_after_exit") is not True
        and (
            result.get("trace_limit_exceeded") is True
            or result.get("trace_reason") == "trace-record-budget"
        )
    )
    if trace_interrupted:
        return None
    semantic_outcome = result.get("semantic_outcome")
    if isinstance(semantic_outcome, str) and semantic_outcome in {"trap", "expected-trap"} and (
        result.get("terminal_observed") is True
        or result.get("guest_trap") == "delivered"
        or type(result.get("guest_cause")) is int
    ):
        return "trap"
    if result.get("guest_trap") == "delivered" or type(
        result.get("guest_cause", result.get("trap.cause"))
    ) is int:
        return "trap"
    if result.get("status") == "timeout":
        return "timeout"
    status = result.get("status")
    if isinstance(status, str) and status in {"failed", "error", "gap"}:
        code = result.get("returncode")
        if type(code) is int and code != 0 and _guest_terminal_evidence(result):
            return "crash" if code < 0 else "nonzero-exit"
    if result.get("status") == "passed" or (
        result.get("status") is None and result.get("execution_status") == "passed"
    ):
        return "complete_observation"
    if result.get("unsupported_isa") is True or result.get("reason_code") == "unsupported-isa":
        return "gap"
    return None


_OBSERVATION_CHANNELS = frozenset({"rvobs1-frame", "program-stdout", "terminal"})

# Differential has exactly three outcomes.  A mismatch is evidence that two
# complete observations differ; it is not a confirmed defect.
DIFFERENTIAL_MATCH = "match"
DIFFERENTIAL_MISMATCH = "mismatch_candidate"
DIFFERENTIAL_UNVERIFIED = "unverified"
DIFFERENTIAL_STATUSES = frozenset({
    DIFFERENTIAL_MATCH, DIFFERENTIAL_MISMATCH, DIFFERENTIAL_UNVERIFIED,
})


def _canonical_differential_status(value: object) -> object:
    """Read the old label once; all new evidence uses the clear label."""
    return DIFFERENTIAL_MISMATCH if value == "inconsistency" else value


def _record_is_pending(record: object) -> bool:
    """识别事件及其保留的嵌套观测中的右删失标记。"""
    if not isinstance(record, dict):
        return False
    pending_statuses = {"pending", "right-censored", "right_censored"}
    pending_outcomes = {
        "pending", "right-censored", "right_censored",
        "deadline-censored", "deadline_censored",
    }
    pending_reasons = {
        "run-wall-clock-exhausted", "execution-wall-clock-exhausted",
        "campaign-wall-clock-exhausted", "framework-target-wall-clock-exhausted",
        "external-signal",
    }
    pending_terminations = pending_reasons | {
        "run-deadline-censored", "execution-wall-clock-exhausted",
        "campaign-wall-clock-exhausted", "framework-target-wall-clock-exhausted",
    }

    def matches(value: object, allowed: set[str]) -> bool:
        # Persisted adapters are untrusted input: list/dict values must be a
        # schema gap, not a TypeError from set membership.
        return isinstance(value, str) and value in allowed

    sources = [record]
    nested = record.get("observation")
    if isinstance(nested, dict):
        sources.append(nested)
    stages = record.get("stage_records")
    if isinstance(stages, dict):
        target_stage = stages.get("target")
        if isinstance(target_stage, dict):
            sources.append(target_stage)
    for source in sources:
        if (
            matches(source.get("status"), pending_statuses)
            or matches(source.get("outcome"), pending_outcomes)
            or matches(source.get("target_outcome"), pending_outcomes)
            or source.get("right_censored") is True
            or source.get("deadline_censored") is True
            or matches(source.get("case_outcome"), pending_outcomes)
            or matches(source.get("failure_class"), pending_reasons)
            or source.get("run_deadline_censored") is True
            or matches(source.get("termination"), pending_terminations)
        ):
            return True
        reasons = (
            source.get("reason_code"), source.get("pending_reason"),
            source.get("reason"),
        )
        if (
            any(matches(reason, pending_reasons) for reason in reasons)
            and source.get("terminal_observed") is not True
            and (source.get("outcome") is None
                 or matches(source.get("outcome"), {"gap", "pending"}))
        ):
            return True
    return False


def _canonical_observation(record: dict[str, Any]) -> dict[str, Any]:
    """Read the current observation shape and the small set of legacy aliases."""
    if not isinstance(record, dict):
        record = {}
    nested = record.get("observation")
    nested = nested if isinstance(nested, dict) else {}
    sources = (nested, record)
    errors: list[str] = []

    def first(*names: str) -> Any:
        for source in sources:
            for name in names:
                value = source.get(name)
                if value is not None:
                    return value
        return None

    def integer(label: str, *names: str, upper: int = 1 << 64) -> int | None:
        value = first(*names)
        if value is None:
            return None
        if type(value) is not int or not 0 <= value < upper:
            errors.append(label)
            return None
        return value

    def digest(*names: str) -> str | None:
        value = first(*names)
        if value is None:
            return None
        if not _valid_hex(value, 64):
            errors.append(names[0])
            return None
        return value.lower()

    pending = _record_is_pending(record)
    raw_outcome = next(
        (source[name] for source in sources
         for name in ("outcome", "terminal", "target_outcome")
         if name in source),
        None,
    )
    has_explicit_outcome = any(
        name in source
        for source in sources
        for name in ("outcome", "terminal", "target_outcome")
    )
    if pending:
        raw_outcome = None
    if isinstance(raw_outcome, str):
        terminal = {
            "normal": "normal", "completed": "normal", "complete": "normal",
            "complete_observation": "normal", "trap": "trap",
            "expected-trap": "expected-trap", "crash": "crash",
            "nonzero-exit": "nonzero-exit", "timeout": "timeout", "gap": "gap",
        }.get(raw_outcome)
        if terminal is None:
            errors.append("outcome")
    else:
        terminal = None
        if raw_outcome is not None:
            errors.append("outcome")
        elif not pending and not has_explicit_outcome and first("status") == "passed":
            terminal = "normal"
        elif not pending and not has_explicit_outcome and first("status") == "timeout":
            terminal = "timeout"

    raw_returncode = first("exit_code", "returncode")
    returncode = raw_returncode
    if returncode is not None and (
        type(returncode) is not int or not -(1 << 31) <= returncode < (1 << 31)
    ):
        errors.append("returncode")
        returncode = None
    tohost_value = integer("tohost_value", "tohost_value")
    channel = first("observation_channel")
    if channel is not None and (
        not isinstance(channel, str) or channel not in _OBSERVATION_CHANNELS
    ):
        errors.append("observation_channel")
        channel = None

    frame_digest = digest("observation_frame_sha256")
    program_digest = digest("program_stdout_sha256", "stdout_sha256")
    raw_stdout_digest = digest("raw_stdout_sha256")
    if channel == "rvobs1-frame":
        stdout_digest = frame_digest or digest("stdout_sha256")
        if first("observation_frame_verified") is not True:
            stdout_digest = None
    elif channel == "program-stdout":
        stdout_digest = program_digest
    elif channel == "terminal":
        stdout_digest = None
    else:
        stdout_digest = program_digest or raw_stdout_digest or digest("stdout_sha256")

    trap_values = []
    for source in sources:
        signature = source.get("trap_signature")
        if isinstance(signature, dict):
            trap_values.append(signature)
        extra = source.get("extra_state")
        if isinstance(extra, dict):
            trap_values.append(extra)
        trap_values.append(source)
    trap_names = {
        "cause": ("cause", "trap.cause", "guest_cause", "csr.mcause", "trap_mcause"),
        "epc": ("epc", "trap.epc", "guest_epc", "csr.mepc"),
        "tval": ("tval", "trap.tval", "guest_tval", "csr.mtval"),
    }
    trap_signature: dict[str, int] = {}
    for field, aliases in trap_names.items():
        value = next(
            (source[name] for source in trap_values for name in aliases
             if name in source and source[name] is not None),
            None,
        )
        if type(value) is int and 0 <= value < 1 << 64:
            trap_signature[field] = value
        elif value is not None:
            errors.append(f"trap.{field}")

    return {
        "terminal": terminal,
        "returncode": returncode,
        "tohost_value": tohost_value,
        "stdout_sha256": stdout_digest,
        "raw_stdout_sha256": raw_stdout_digest,
        "observation_channel": channel,
        "observation_frame_sha256": frame_digest,
        "observation_frame_verified": first("observation_frame_verified") is True,
        "trap_signature": trap_signature or None,
        "schema_errors": tuple(dict.fromkeys(errors)),
    }

def _trap_signature_complete(signature: object) -> bool:
    """Cause and EPC suffice for a trap-only comparison; tval remains optional."""
    return (
        isinstance(signature, dict)
        and type(signature.get("cause")) is int
        and 0 <= signature["cause"] < 1 << 64
        and type(signature.get("epc")) is int
        and 0 <= signature["epc"] < 1 << 64
        and not signature["epc"] & 1
    )


def _trusted_observation_outcome(record: dict[str, Any]) -> str | None:
    """Read the actual outcome from one target observation."""
    if not isinstance(record, dict) or _record_is_pending(record):
        return None
    observation = _canonical_observation(record)
    terminal = observation.get("terminal")
    if terminal in (None, "gap"):
        return None
    if record.get("status") == "gap":
        nested = record.get("observation")
        has_canonical_outcome = record.get("outcome") is not None
        has_legacy_nested_outcome = isinstance(nested, dict) and (
            nested.get("outcome", nested.get("terminal")) is not None
        )
        if not has_canonical_outcome and not has_legacy_nested_outcome:
            return None
    if terminal == "expected-trap":
        nested = record.get("observation")
        extra = nested.get("extra_state") if isinstance(nested, dict) else None
        terminal_observed = any(
            source.get("terminal_observed") is True
            for source in (nested, record) if isinstance(source, dict)
        )
        delivered = any(
            isinstance(source, dict) and (
                source.get("guest_trap") == "delivered"
                or type(source.get("guest_cause")) is int
                or type(source.get("trap.cause")) is int
            )
            for source in (nested, record, extra)
        )
        if (
            not terminal_observed
            and not _trap_signature_complete(observation.get("trap_signature"))
            and not delivered
        ):
            return None
    # Old records used expected-trap as an outcome. The expectation belongs
    # to the case contract; the actual guest outcome is trap.
    return {
        "expected-trap": "trap",
        "complete_observation": "normal",
    }.get(terminal, terminal)


def _observation_record(result: dict[str, Any]) -> dict[str, Any]:
    outcome = _result_target_outcome(result)
    outcome = {
        "complete_observation": "normal",
        "expected-trap": "trap",
    }.get(outcome, outcome)
    channel = result.get("observation_channel")
    if not isinstance(channel, str):
        if outcome == "normal" and result.get("observation_frame_verified") is True:
            channel = "rvobs1-frame"
        elif outcome == "normal" and result.get("program_stdout_sha256") is not None:
            channel = "program-stdout"
        elif outcome in {"trap", "crash", "nonzero-exit", "timeout"}:
            channel = "terminal"
    observation: dict[str, Any] = {
        "schema_version": "rq1-observation-v2",
        "outcome": outcome,
    }
    values = {
        "exit_code": result.get("returncode"),
        "tohost_value": result.get("tohost_value"),
        "observation_channel": channel,
        "observation_frame_sha256": result.get("observation_frame_sha256"),
        "observation_frame_verified": result.get("observation_frame_verified"),
        "program_stdout_sha256": result.get("program_stdout_sha256"),
        "trap.cause": result.get("guest_cause"),
        "trap.epc": result.get("guest_epc"),
        "trap.tval": result.get("guest_tval"),
    }
    observation.update({name: value for name, value in values.items() if value is not None})
    return {"observation": observation}


def _record_identity_digest(record: dict[str, Any]) -> str | None:
    digests: list[Any] = []
    invalid = False
    # ``reference_identity`` identifies the board/reference, not the Target.
    # Mixing it into this digest makes a valid target look inconsistent when
    # the board happens to expose its own identity digest.
    def collect(value: object) -> None:
        nonlocal invalid
        if not isinstance(value, dict):
            return
        for field in ("identity_digest", "target_identity_digest"):
            if field not in value:
                continue
            item = value[field]
            if not isinstance(item, str) or not item:
                invalid = True
            else:
                digests.append(item)
        collect(value.get("observed"))

    for name in ("target_identity", "identity"):
        collect(record.get(name))
    for name in ("target_identity_digest", "identity_digest"):
        if name not in record:
            continue
        value = record[name]
        if not isinstance(value, str) or not value:
            invalid = True
        else:
            digests.append(value)
    normalized = {item.lower() for item in digests if isinstance(item, str)}
    if (invalid or not digests or not all(isinstance(item, str) for item in digests)
            or len(normalized) != 1):
        return None
    return next(iter(normalized))


def _record_has_identity_digest(record: dict[str, Any]) -> bool:
    def has_digest(value: object) -> bool:
        return isinstance(value, dict) and (
            any(field in value
                for field in ("identity_digest", "target_identity_digest"))
            or has_digest(value.get("observed"))
        )

    for name in ("target_identity", "identity"):
        if has_digest(record.get(name)):
            return True
    return any(name in record
               for name in ("target_identity_digest", "identity_digest"))


def _record_artifact_digest(record: dict[str, Any]) -> str | None:
    """Return one artifact digest, rejecting conflicting legacy aliases."""
    if not isinstance(record, dict):
        return None
    values = []
    invalid = False
    for name in ("artifact_sha256", "artifact_id"):
        if name not in record:
            continue
        value = record[name]
        # Artifact aliases are optional. Serializers may emit the unused
        # alias as null; that means absence, not an identity conflict.
        if value in (None, ""):
            continue
        if not isinstance(value, str) or not value:
            invalid = True
        else:
            values.append(value)
    if invalid or not values or any(not _valid_hex(value, 64) for value in values):
        return None
    normalized = {value.lower() for value in values}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _record_has_artifact_digest(record: dict[str, Any]) -> bool:
    return isinstance(record, dict) and any(
        name in record
        for name in ("artifact_sha256", "artifact_id")
    )


def _target_testable(record: dict[str, Any]) -> bool:
    """A target is tested when an attempted run produced an actual outcome."""
    outcome = _trusted_observation_outcome(record) if isinstance(record, dict) else None
    return bool(
        isinstance(record, dict)
        and record.get("target_attempted") is True
        and outcome not in (None, "timeout")
    )


def _semantic_observation_comparison(
    left_record: dict[str, Any], right_record: dict[str, Any],
) -> dict[str, Any]:
    """Compare only the semantic values present in both observations."""
    left = _canonical_observation(left_record)
    right = _canonical_observation(right_record)

    def outcome(observation: dict[str, Any]) -> str | None:
        value = observation.get("terminal")
        if value == "expected-trap":
            return "trap"
        if value == "complete_observation":
            return "normal"
        if value in {"normal", "trap", "crash", "nonzero-exit"}:
            return value
        return None

    left_outcome, right_outcome = outcome(left), outcome(right)
    left_values: dict[str, object] = {}
    right_values: dict[str, object] = {}
    fields = ["outcome"]
    if left_outcome is not None:
        left_values["outcome"] = left_outcome
    if right_outcome is not None:
        right_values["outcome"] = right_outcome

    if left_outcome == right_outcome == "trap":
        fields.extend(("trap.cause", "trap.epc"))
        for name in ("cause", "epc"):
            field = f"trap.{name}"
            for values, observation in (
                (left_values, left), (right_values, right),
            ):
                signature = observation.get("trap_signature")
                if isinstance(signature, dict) and type(signature.get(name)) is int:
                    values[field] = signature[name]
    else:
        timeout_present = "timeout" in {
            left.get("terminal"), right.get("terminal"),
        }
        fields_to_compare = [("tohost_value", "tohost_value")]
        if left_outcome is not None and right_outcome is not None and not timeout_present:
            fields_to_compare.insert(0, ("exit_code", "returncode"))
        for field, key in fields_to_compare:
            left_value, right_value = left.get(key), right.get(key)
            if type(left_value) is int and type(right_value) is int:
                fields.append(field)
                left_values[field] = left_value
                right_values[field] = right_value


    left_channel, right_channel = (
        left.get("observation_channel"), right.get("observation_channel"),
    )
    left_stdout, right_stdout = left.get("stdout_sha256"), right.get("stdout_sha256")
    if (
        left_channel == right_channel
        and left_channel in {"rvobs1-frame", "program-stdout"}
        and _valid_hex(left_stdout, 64)
        and _valid_hex(right_stdout, 64)
    ):
        fields.append("stdout_sha256")
        left_values["stdout_sha256"] = left_stdout.lower()
        right_values["stdout_sha256"] = right_stdout.lower()

    differences, missing = compare_fields(left_values, right_values, fields)
    missing = list(missing)
    if left_outcome is None or right_outcome is None:
        missing.append("outcome")
    if left_outcome == right_outcome == "trap":
        for field in ("trap.cause", "trap.epc"):
            if field not in left_values or field not in right_values:
                missing.append(field)
    elif left_outcome == right_outcome == "normal" and not any(
        field in fields for field in ("tohost_value", "stdout_sha256")
    ):
        missing.append("semantic_state_observation")
    elif left_outcome == right_outcome in {"crash", "nonzero-exit"} and not any(
        field in fields for field in ("exit_code", "tohost_value", "stdout_sha256")
    ):
        missing.append("terminal_details")

    status = (
        "non-equivalent" if differences else
        "reference-gap" if missing else
        "equivalent"
    )
    return {
        "status": status,
        "differences": differences,
        "missing_fields": list(dict.fromkeys(missing)),
        "left": left,
        "right": right,
    }


def _differential_result(
    semantic: dict[str, Any], qualification_gaps: list[str], *,
    schema_version: str, proof: dict[str, Any], reason: str | None = None,
) -> dict[str, Any]:
    semantic_missing = list(dict.fromkeys(semantic["missing_fields"]))
    qualification_gaps = list(dict.fromkeys(qualification_gaps))
    raw_mismatch = semantic["status"] == "non-equivalent"
    status = (
        DIFFERENTIAL_UNVERIFIED if semantic_missing or qualification_gaps else
        DIFFERENTIAL_MISMATCH if raw_mismatch else
        DIFFERENTIAL_MATCH if semantic["status"] == "equivalent"
        else DIFFERENTIAL_UNVERIFIED
    )
    token = hashlib.sha256(
        json.dumps(proof, sort_keys=True, default=repr).encode()
    ).hexdigest()[:16]
    if reason is None:
        if semantic_missing:
            reason = "observation-fields-missing"
        elif qualification_gaps:
            reason = "comparison-qualification-gap"
    return {
        "schema_version": schema_version,
        "status": status,
        "comparison": {
            "status": semantic["status"],
            "differences": semantic["differences"],
            "missing_fields": semantic_missing,
        },
        "basis": [
            "outcome", "exit_code", "tohost_value", "stdout_sha256",
            "observation_channel", "trap.cause", "trap.epc",
        ],
        "proof": proof,
        "mismatch_id": f"mismatch-{token}" if raw_mismatch else None,
        "reason": reason,
        "missing_fields": semantic_missing,
        "qualification_gaps": qualification_gaps,
    }


def _differential_record(reference: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    reference = reference if isinstance(reference, dict) else {}
    target = target if isinstance(target, dict) else {}
    semantic = _semantic_observation_comparison(reference, target)
    ref_observation, target_observation = semantic["left"], semantic["right"]
    qualification_gaps: list[str] = []
    for name, record in (("reference", reference), ("target", target)):
        if any(record.get(field) for field in (
            "comparison_qualification_gaps", "qualification_gaps", "missing_fields",
            "validation_gap", "contract_error", "target_execution_contract_error",
        )):
            qualification_gaps.append(f"{name}_record_qualification")

    if target.get("target_attempted") is not True:
        qualification_gaps.append("target_attempted")

    if not _trusted_reference_board(reference):
        qualification_gaps.append("reference_identity")

    target_identity = _target_identity_record(target)
    target_identity_digest = _record_identity_digest(target)
    if target_identity is None or not _target_identity_pinned(target_identity):
        qualification_gaps.append("target_identity")
    if _record_has_identity_digest(target) and not _valid_hex(target_identity_digest, 64):
        qualification_gaps.append("target_identity")

    reference_artifact = _record_artifact_digest(reference)
    target_artifact = _record_artifact_digest(target)
    if (
        not _valid_hex(reference_artifact, 64)
        or not _valid_hex(target_artifact, 64)
        or reference_artifact != target_artifact
    ):
        qualification_gaps.append("target_artifact_identity")
    if target.get("artifact_identity_ok") is False:
        qualification_gaps.append("target_artifact_identity")
    reference_input = reference.get("input_id")
    target_input = target.get("input_id")
    if reference_input not in (None, "") or target_input not in (None, ""):
        if (
            not isinstance(reference_input, str) or not reference_input.strip()
            or reference_input != target_input
        ):
            qualification_gaps.append("target_input_identity")

    proof = {
        "reference_terminal": ref_observation.get("terminal"),
        "target_terminal": target_observation.get("terminal"),
        "reference_returncode": ref_observation.get("returncode"),
        "target_returncode": target_observation.get("returncode"),
        "reference_tohost_value": ref_observation.get("tohost_value"),
        "target_tohost_value": target_observation.get("tohost_value"),
        "reference_stdout_sha256": ref_observation.get("stdout_sha256"),
        "target_stdout_sha256": target_observation.get("stdout_sha256"),
        "reference_observation_channel": ref_observation.get("observation_channel"),
        "target_observation_channel": target_observation.get("observation_channel"),
        "reference_trap_signature": ref_observation.get("trap_signature"),
        "target_trap_signature": target_observation.get("trap_signature"),
        "target_identity_digest": target_identity_digest,
        "artifact_sha256": target_artifact,
        "command_sha256": target.get("command_sha256"),
    }
    reason = (
        "reference-observation-missing"
        if ref_observation.get("terminal") in (None, "gap") else
        "target-observation-missing"
        if target_observation.get("terminal") in (None, "gap") else
        "reference-timeout" if ref_observation.get("terminal") == "timeout" else
        "target-timeout" if target_observation.get("terminal") == "timeout" else
        None
    )
    return _differential_result(
        semantic, qualification_gaps,
        schema_version="rq1-differential-v6", proof=proof, reason=reason,
    )


def _emi_pair_differential(parent: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    parent = parent if isinstance(parent, dict) else {}
    variant = variant if isinstance(variant, dict) else {}
    semantic = _semantic_observation_comparison(parent, variant)
    parent_observation, variant_observation = semantic["left"], semantic["right"]
    qualification_gaps: list[str] = []

    identities: dict[str, str | None] = {}
    for name, row, observation in (
        ("parent", parent, parent_observation),
        ("variant", variant, variant_observation),
    ):
        if any(row.get(field) for field in (
            "comparison_qualification_gaps", "qualification_gaps", "missing_fields",
            "validation_gap", "contract_error", "target_execution_contract_error",
        )):
            qualification_gaps.append(f"{name}_record_qualification")
        identity = _target_identity_record(row)
        identity_digest = _record_identity_digest(row)
        identities[name] = identity_digest
        if identity is None or not _target_identity_pinned(identity):
            qualification_gaps.append(f"{name}_target_identity")
        if not _valid_hex(identity_digest, 64):
            qualification_gaps.append(f"{name}_target_identity_digest")
        if row.get("target_attempted") is not True:
            qualification_gaps.append(f"{name}_target_attempted")

        actual_artifact = _record_artifact_digest(row)
        declared_artifact = row.get("declared_artifact_sha256")
        if (
            not _record_has_artifact_digest(row)
            or not _valid_hex(actual_artifact, 64)
            or not isinstance(declared_artifact, str)
            or not _valid_hex(declared_artifact, 64)
            or actual_artifact != declared_artifact.lower()
        ):
            qualification_gaps.append(f"{name}_artifact_identity")

    if (
        _valid_hex(identities.get("parent"), 64)
        and _valid_hex(identities.get("variant"), 64)
        and identities["parent"] != identities["variant"]
    ):
        qualification_gaps.append("target_identity_mismatch")

    proof = {
        "parent_terminal": parent_observation.get("terminal"),
        "variant_terminal": variant_observation.get("terminal"),
        "parent_returncode": parent_observation.get("returncode"),
        "variant_returncode": variant_observation.get("returncode"),
        "parent_tohost_value": parent_observation.get("tohost_value"),
        "variant_tohost_value": variant_observation.get("tohost_value"),
        "parent_stdout_sha256": parent_observation.get("stdout_sha256"),
        "variant_stdout_sha256": variant_observation.get("stdout_sha256"),
        "parent_observation_channel": parent_observation.get("observation_channel"),
        "variant_observation_channel": variant_observation.get("observation_channel"),
        "parent_trap_signature": parent_observation.get("trap_signature"),
        "variant_trap_signature": variant_observation.get("trap_signature"),
        "parent_target_identity_digest": identities["parent"],
        "variant_target_identity_digest": identities["variant"],
        "parent_artifact_sha256": _record_artifact_digest(parent),
        "parent_declared_artifact_sha256": parent.get("declared_artifact_sha256"),
        "variant_artifact_sha256": _record_artifact_digest(variant),
        "variant_declared_artifact_sha256": variant.get("declared_artifact_sha256"),
    }
    return _differential_result(
        semantic, qualification_gaps,
        schema_version="rq1-emi-differential-v5", proof=proof,
    )

def _trusted_reference_board(record: dict[str, Any]) -> bool:
    """Trust a pinned K1 state frame or a candidate-bound QEMU Target fallback."""
    if not isinstance(record, dict):
        return False
    identity = record.get("identity")
    status = record.get("status")
    if (
        (record.get("id"), record.get("engine"), record.get("backend"), record.get("target"))
        == ("R-K1-BOARD", "k1-board", "native-rv64", "T-K1-BOARD")
        and isinstance(status, str)
        and status in {"passed", "failed"}
        and record.get("observer_complete") is True
        and record.get("observation_channel") == "rvobs1-frame"
        and record.get("observation_frame_verified") is True
        and _valid_hex(record.get("observation_frame_sha256"), 64)
        and isinstance(identity, dict)
        and identity.get("id") == "R-K1-BOARD"
        and identity.get("engine") == "k1-board"
        and identity.get("backend") == "native-rv64"
        and identity.get("target") == "T-K1-BOARD"
        and isinstance(identity.get("source"), str)
        and identity.get("source") in {
            "framework.native.rv64_runner", "framework.native.k1_jtag_runner"
        }
        and identity.get("status") in {"verified", "recorded"}
    ):
        return True

    fallback = record.get("reference_fallback")
    k1_attempt = fallback.get("k1_attempt") if isinstance(fallback, dict) else None
    observation = record.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    channel = observation.get("observation_channel")
    outcome = _trusted_observation_outcome(record)
    fallback_identity_ok = (
        isinstance(identity, dict)
        and identity.get("backend") == "qemu-riscv64"
        and _target_identity_pinned(identity)
    )
    observation_ok = (
        type(record.get("returncode")) is int
        and channel in {"rvobs1-frame", "program-stdout", "terminal"}
        and (
            channel != "rvobs1-frame"
            or observation.get("observation_frame_verified") is True
            and _valid_hex(observation.get("observation_frame_sha256"), 64)
        )
        and (
            channel != "program-stdout"
            or _valid_hex(observation.get("program_stdout_sha256"), 64)
        )
    )
    return (
        (record.get("id"), record.get("engine"), record.get("backend"), record.get("target"))
        == ("R-QEMU-FALLBACK", "qemu-target-fallback", "qemu-riscv64", "T-QEMU")
        and isinstance(status, str)
        and status in {"passed", "failed"}
        and record.get("target_attempted") is True
        and isinstance(fallback, dict)
        and fallback.get("schema_version") == "rq1-reference-fallback-v1"
        and fallback.get("policy") == K1_BOARD_TQEMU_FALLBACK_POLICY
        and fallback.get("status") == "used"
        and fallback.get("preferred_backend") == "native-rv64"
        and fallback.get("effective_backend") == "qemu-riscv64"
        and fallback.get("selection_scope") == "candidate"
        and fallback.get("selection_point") == "same-candidate-t-qemu-result"
        and isinstance(fallback.get("fallback_reason"), str)
        and fallback.get("fallback_reason") in K1_QEMU_FALLBACK_FAILURE_CLASSES
        and fallback.get("source_target") == "T-QEMU"
        and isinstance(fallback.get("source_event_id"), str)
        and bool(fallback.get("source_event_id"))
        and _valid_hex(fallback.get("source_event_sha256"), 64)
        and fallback_identity_ok
        and record.get("target_identity") == identity
        and isinstance(k1_attempt, dict)
        and k1_attempt.get("failure_class") == fallback.get("fallback_reason")
        and k1_attempt.get("status") in {"gap", "timeout"}
        and _record_artifact_digest(k1_attempt) == _record_artifact_digest(record)
        and observation_ok
        and outcome in {"normal", "trap", "crash", "nonzero-exit"}
        and _valid_hex(_record_artifact_digest(record), 64)
    )

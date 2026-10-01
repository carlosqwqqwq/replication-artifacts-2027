"""Campaign 与 ledger replay 共用的程序事实。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import re

from framework._util import object_field as _object_field
from framework.rvemi.program import (
    _pc_value, _profile_source_map, _source_key, _stable_input_identity,
)
from framework.supply.model_candidate_supply import rewrite_hint_edits

_INSTRUCTION_RE = re.compile(
    r"(?P<indent>\s*)(?<![\w.$])(?P<mnemonic>[A-Za-z][\w.]*)"
    r"\s+(?P<operands>[^#;\r\n]+?)(?P<comment>\s*#.*)?"
    r"(?=\s*(?:;|$))", re.IGNORECASE,
)


def _program_source(program: object) -> str:
    source = getattr(program, "source", None)
    if isinstance(source, str):
        return source
    path = Path(getattr(program, "program", program))
    return path.read_bytes().decode("utf-8") if path.suffix.lower() == ".s" else ""


def _mnemonic_matches(row_mnemonic: object, wanted: str) -> bool:
    """Treat compressed ``c.foo`` and source ``foo`` as one instruction form."""
    left = str(row_mnemonic).lower().replace("_", ".")
    right = str(wanted).lower().replace("_", ".")
    if left == right:
        return True
    if left.startswith("c."):
        left = left[2:]
    if right.startswith("c."):
        right = right[2:]
    return left == right


def profile_realizes(
    program: object, hint: Mapping[str, object],
) -> bool:
    """编辑只需落在源码行范围内。

    改写会改变语义与执行路径，等价性由后续 reference gate 判定；这里不要求
    改写后的指令在 reference 轨迹中执行过，也不要求与原指令一致。
    """
    if not isinstance(hint, Mapping) or not rewrite_hint_edits(hint):
        return False
    line_count = len(_program_source(program).splitlines(keepends=True))
    for edit in rewrite_hint_edits(hint):
        span = edit.get("source_span") if isinstance(edit, Mapping) else None
        start = span.get("start") if isinstance(span, Mapping) else None
        end = span.get("end", start) if isinstance(span, Mapping) else None
        if type(start) is not int or type(end) is not int \
                or not 1 <= start <= end <= line_count:
            return False
    return True


def profile_sequence_realizes(profile: object, sequence: Sequence[str]) -> bool:
    provenance = getattr(profile, "provenance", {})
    execution = provenance.get("execution_trace") if isinstance(provenance, Mapping) else None
    if not isinstance(execution, Mapping) or execution.get("complete") is not True:
        return False
    expected = tuple(_source_key(item) for item in sequence)
    if any(item is None for item in expected):
        return False
    rows = tuple(getattr(profile, "instructions", ()))
    occurrences = tuple(getattr(profile, "occurrences", ()))
    if occurrences:
        by_id = {row.get("id"): row for row in rows if isinstance(row, Mapping)}
        ordered = tuple(
            by_id.get(item.get("instruction_id"))
            for item in occurrences if isinstance(item, Mapping)
        )
        if len(ordered) != len(occurrences) or any(item is None for item in ordered):
            return False
        rows = ordered
    position = 0
    for row in rows:
        if position == len(expected):
            return True
        if not isinstance(row, Mapping) or row.get("executed") is not True:
            continue
        mnemonic = row.get("mnemonic")
        operands = row.get("operands", ())
        if not isinstance(mnemonic, str) or not isinstance(operands, (list, tuple)) \
                or any(not isinstance(item, str) for item in operands):
            return False
        actual = _source_key(
            f"{mnemonic} {', '.join(operands)}"
        )
        wanted = expected[position]
        if actual is not None and wanted is not None and (
            _mnemonic_matches(actual[0], wanted[0])
            and (not wanted[1] or actual[1] == wanted[1])
        ):
            position += 1
    return position == len(expected)


def profile_dynamic_realizes(profile: object) -> bool:
    if getattr(profile, "route", None) == "single":
        return bool(getattr(profile, "dynamic_witness", False))
    rows = tuple(getattr(profile, "instructions", ()))
    occurrences = tuple(getattr(profile, "occurrences", ()))
    fields = set(getattr(profile, "observer_fields", ()))
    provenance = getattr(profile, "provenance", {})
    execution = provenance.get("execution_trace", {}) \
        if isinstance(provenance, Mapping) else {}
    ids = {row.get("id") for row in rows if isinstance(row, Mapping)}
    return bool(
        rows and occurrences and getattr(profile, "executed_ids", ())
        and {"executed_pcs", "outcome"} <= fields
        and isinstance(execution, Mapping) and execution.get("complete") is True
        and all(
            isinstance(item, Mapping)
            and item.get("instruction_id") in ids
            and _pc_value(item.get("pc")) is not None
            for item in occurrences
        )
    )


def profile_identity(artifact: object | Mapping[str, object], observations: Sequence[object]) -> dict[str, object]:
    run_params = _stable_input_identity(
        artifact.get("run_params", {}) if isinstance(artifact, Mapping) else artifact.run_params
    )
    if not observations:
        # reference 域缺口时没有 observation；返回稳定输入身份即可。
        return run_params
    observation = observations[0]
    pcs = tuple(_object_field(observation, "executed_pcs", ()))
    if not pcs:
        return run_params
    checkpoint_pc = _object_field(observation, "checkpoint_pc")
    entry = _pc_value(pcs[0])
    exit_pc = _pc_value(pcs[-1] if checkpoint_pc is None else checkpoint_pc)
    if run_params.get("harness") == "custom":
        source_map = _profile_source_map(observation)
        lines = source_map.get("lines", {}) if isinstance(source_map, Mapping) else {}
        source_pcs = tuple(
            _pc_value(pc)
            for line, values in lines.items()
            if str(line).isdigit() and int(line) < 10000
            for pc in values
        )
        source_pcs = tuple(pc for pc in source_pcs if pc is not None)
        source_pc_set = set(source_pcs)
        executed_source_pcs = tuple(pc for pc in pcs if pc in source_pc_set)
        if executed_source_pcs:
            entry, exit_pc = executed_source_pcs[0], executed_source_pcs[-1]
        if source_pcs:
            entry = entry if executed_source_pcs else min(source_pcs)
            exit_pc = exit_pc if executed_source_pcs else max(source_pcs)
    return {**run_params, "entry": entry if entry is not None else pcs[0],
            "exit": exit_pc if exit_pc is not None else (pcs[-1] if checkpoint_pc is None else checkpoint_pc)}


def reference_identity(observations: Sequence[object]) -> dict[str, object]:
    if not observations:
        # reference 域缺口时没有 observation；空身份让上层继续，不阻断目标执行。
        return {}
    observation = observations[0]
    identity = _object_field(observation, "reference_identity")
    result = dict(identity) if isinstance(identity, Mapping) else {
        name: _object_field(observation, name)
        for name in ("backend", "profile_id", "input_id", "tool_version")
        if _object_field(observation, name) not in (None, "")
    }
    evidence = _object_field(observation, "translation_evidence")
    details = _object_field(evidence, "details")
    target_identity = _object_field(details, "target_identity")
    if isinstance(target_identity, Mapping):
        result.update({
            name: target_identity[name]
            for name in ("source_commit", "identity_digest", "binary_sha256")
            if target_identity.get(name) not in (None, "")
        })
    native_identity = _object_field(details, "native_reference_identity")
    if isinstance(native_identity, Mapping):
        result.update({
            name: native_identity[name]
            for name in (
                "schema_version", "host_key_sha256", "machine", "isa", "kernel",
                "runner_sha256", "helper_sha256", "identity_digest",
            )
            if native_identity.get(name) not in (None, "")
        })
    extra_state = _object_field(observation, "extra_state")
    reference = _object_field(extra_state, "reference_identity")
    if isinstance(reference, Mapping):
        result.update({
            name: reference[name]
            for name in (
                "backend", "source_repository", "source_commit", "binary_sha256",
                "runtime_binary_sha256", "runner_sha256", "container_image_digest",
                "identity_digest", "isa",
            ) if reference.get(name) not in (None, "")
        })
    sail_digest = _object_field(extra_state, "sail_reference_identity_digest")
    if sail_digest not in (None, ""):
        result["identity_digest"] = sail_digest
    guest_elf = _object_field(observation, "guest_elf_sha256")
    if guest_elf not in (None, ""):
        result["guest_elf_sha256"] = guest_elf
    return result


def attach_execution_identity(result: object, artifact: object, observations: Sequence[object]) -> object:
    profile = getattr(result, "profile", None)
    if profile is None:
        return result
    provenance = {
        **dict(profile.provenance),
        "reference_identity": reference_identity(observations),
        "executable_sha256": artifact.executable_sha256,
    }
    profile = replace(profile, provenance=provenance)
    profile.provenance["profile_digest"] = profile.profile_digest
    return replace(result, profile=profile)


def rule_evidence_records(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "step": event.get("step"), "action_digest": event.get("action_digest"),
            "rule": action.get("rule"), "anchor": action.get("anchor"),
            "evidence": action.get("rule_evidence"),
            "evidence_level": action.get("evidence_level"),
            **({"witness_status": action["witness_status"]}
               if "witness_status" in action else {}),
            **({"witness_reason": action["witness_reason"]}
               if "witness_reason" in action else {}),
        }
        for event in events
        if isinstance(event, Mapping)
        and isinstance(action := event.get("action"), Mapping)
        and action.get("rule_evidence") is not None
        and action.get("evidence_level") == "guided"
    ]

"""Content identity and the scheduled-to-covered obligation ledger.

Identity and coverage are the same accounting question asked twice -- what is
this program, and which obligation did it discharge -- so they share one owner.
Coverage is only ever credited from the classification admission produced, never
from the intent that scheduled the work.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from ..direct_case import TestCase
from .effects import disposition_for_schema

def _program_identity(testcase: TestCase) -> str:
    """Identity of code, layout and its observation skeleton."""
    digest = hashlib.sha256()
    digest.update(testcase.code_bytes)
    digest.update(json.dumps(
        {
            "functional_tags": list(testcase.functional_tags),
            "layout_tags": list(testcase.layout_tags),
            "sail_branch_contract": testcase.dataflow_meta.get("sail_branch_contract"),
        }, sort_keys=True, separators=(",", ":")).encode())
    if testcase.observability_contract is not None:
        digest.update(
            json.dumps(
                testcase.observability_contract.canonical_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    digest.update(str(testcase.expected_path).encode())
    for item in testcase.instruction_meta:
        digest.update(
            json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        )
    for block in testcase.block_meta:
        digest.update(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _input_identity(testcase: TestCase) -> str:
    """Identity of the profile and complete initial architectural input."""
    digest = hashlib.sha256()
    digest.update(testcase.isa_profile.encode())
    realization = testcase.dataflow_meta.get("generation_realization", {})
    digest.update(
        json.dumps(
            {
                "code_address": testcase.code_address,
                "entry_checkpoint_address": testcase.entry_checkpoint_address,
                "exit_checkpoint_address": testcase.exit_checkpoint_address,
                "generation_realization": realization if isinstance(realization, dict) else {},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(",".join(str(value) for value in testcase.initial_gpr).encode())
    for region in sorted(
        testcase.initial_memory_regions,
        key=lambda item: (item.region_id, item.address, item.permissions, bytes(item.data)),
    ):
        digest.update(
            json.dumps(
                {
                    "id": region.region_id,
                    "address": region.address,
                    "permissions": region.permissions,
                    "data": bytes(region.data).hex(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _execution_path_gap(instruction_meta, expected: set[str]) -> str | None:
    risk_index = next(
        (index for index, item in enumerate(instruction_meta) if "risk" in item.tags),
        len(instruction_meta),
    )
    risks = {
        item.instruction_id for item in instruction_meta if "risk" in item.tags
    }
    if not risks <= expected:
        return "expected-execution-must-include-risk"
    required = {
        item.instruction_id for item in instruction_meta[:risk_index]
        if "producer" in item.tags
    }
    return None if required <= expected else "expected-execution-must-include-producers"


def realized_facts(
    instruction_meta,
    *,
    expected_ids,
    schema: Any | None,
    lane: str,
) -> dict[str, object]:
    if not isinstance(expected_ids, (list, tuple)):
        raise ValueError("expected instruction IDs must be an ordered sequence")
    expected_sequence = tuple(expected_ids)
    expected = set(expected_sequence)
    if len(expected_sequence) != len(expected):
        raise ValueError("expected-execution-must-not-repeat-instructions")
    if any(not isinstance(item, str) or not item for item in expected):
        raise ValueError("expected instruction IDs must be non-empty strings")
    for item in instruction_meta:
        tags = getattr(item, "tags", None)
        if (
            not isinstance(getattr(item, "instruction_id", None), str)
            or not getattr(item, "instruction_id")
            or not isinstance(tags, (list, tuple, set, frozenset))
            or any(not isinstance(tag, str) or not tag for tag in tags)
        ):
            raise ValueError("RVGEN instruction metadata is invalid")
    if lane not in {"normal-defined", "legality-expected-trap"}:
        raise ValueError("RVGEN realized facts lane is invalid")
    instruction_ids = {item.instruction_id for item in instruction_meta}
    if not expected <= instruction_ids:
        raise ValueError("expected instruction IDs must refer to instructions")
    if lane == "legality-expected-trap":
        schema = None
    if gap := _execution_path_gap(instruction_meta, expected):
        raise ValueError(gap)
    if schema is not None and any(
        not isinstance(getattr(schema, name, None), str) or not getattr(schema, name)
        for name in ("state_domain", "kind")
    ):
        raise ValueError("RVGEN effect schema is invalid")
    has_sink = any(
        "observable" in item.tags
        and item.instruction_id in expected
        for item in instruction_meta
    )
    return {
        "disposition": disposition_for_schema(schema, lane, has_sink=has_sink),
        "lane": str(lane),
        "state_domain": "trap" if schema is None else str(schema.state_domain),
        "effect_kind": "legality-only" if schema is None else str(schema.kind),
    }


def content_identity(testcase: TestCase) -> str:
    program = _program_identity(testcase)
    input_value = _input_identity(testcase)
    mask = testcase.compare_mask
    executed = testcase.dataflow_meta.get("expected_executed_instruction_ids", ())
    semantic_point = testcase.dataflow_meta.get("semantic_point")
    if not isinstance(program, str) or not isinstance(input_value, str):
        raise ValueError("content identity digests must be strings")
    try:
        payload = {
            "program": program,
            "input": input_value,
            "compare_mask": {
                name: getattr(mask, name)
                for name in (
                    "outcome", "checkpoint_pc", "gpr_indices", "memory_region_ids",
                    "signal", "fault_pc", "fault_address", "extra_state_keys",
                )
            },
            "expected_executed_instruction_ids": list(executed),
            "semantic_point": semantic_point,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("content identity fields are invalid") from exc


def identify_testcase(testcase: TestCase) -> TestCase:
    """Assign the three domain identities after admission/classification."""
    program = _program_identity(testcase)
    input_value = _input_identity(testcase)
    content = content_identity(testcase)
    form = testcase.generation_rule_id.partition(":")[0]
    return replace(
        testcase,
        base_program_id=f"rvp6-{form}-{program[:16]}",
        input_id=f"input-{input_value[:24]}",
        testcase_id=f"rvp6-{content[:24]}",
    )


@dataclass
class ObligationLedger:
    """One row per scheduled obligation, closed by an admitted classification."""

    scheduled: Counter[str] = field(default_factory=Counter)
    covered: Counter[str] = field(default_factory=Counter)
    disposition_by_obligation: dict[str, str] = field(default_factory=dict)
    case_ids_by_obligation: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stage_by_obligation: dict[str, str] = field(default_factory=dict)
    reason_by_obligation: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "rvgen-obligation-ledger-v2",
            "scheduled": dict(sorted(self.scheduled.items())),
            "covered": dict(sorted(self.covered.items())),
            "disposition_by_obligation": dict(sorted(self.disposition_by_obligation.items())),
            "case_ids_by_obligation": {
                key: list(value) for key, value in sorted(self.case_ids_by_obligation.items())
                if value or key in self.covered
            },
            "stage_by_obligation": dict(sorted(self.stage_by_obligation.items())),
            "reason_by_obligation": dict(sorted(self.reason_by_obligation.items())),
        }

"""Generation entry point: catalog/pool -> effect -> boundary -> assembly -> ledger.

The loop below iterates the frozen catalog rather than a hand-written family
list, so encoding coverage grows mechanically with upstream form inventory.
Runnable semantics still depend on the shared state/effect owners; this module
is not a complete ISA semantics owner by itself.
"""

import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cache

from ..direct_case import TestCase
from ..riscv_catalog import (
    OFFICIAL_ALL_CATALOG_BY_MNEMONIC,
    OFFICIAL_ALL_CATALOG_FORMS,
    OFFICIAL_SELECTED_SCALAR_BY_MNEMONIC,
    OfficialForm,
)
from ..spec_definedness import SemanticRealization
from .admission import admit_testcase
from .effects import (
    disposition_for_schema,
    generation_rule_for_form,
    generation_schema_for_form,
    schema_for_realization,
)
from .intents import boundary_intents
from .ledger import ObligationLedger, content_identity, identify_testcase
from .materialize import UnrealizedState, materialize
from .obligations import normative_obligation_for

_VIEW_EFFECT_KINDS = {
    "integer": {"integer-writeback", "integer-multiply"},
    "shift": {"integer-shift"},
    "float": {"fp-arith", "fp-fused", "fp-compare", "fp-classify"},
    "convert": {"fp-convert"},
    "memory": {"memory-load", "memory-store", "cache-block"},
    "atomic": {"atomic-rmw"},
    "control": {"control-compare", "control-jump"},
    "csr": {"csr-access"},
    "vector": {"vector-data", "vector-memory"},
    "privileged": {"privileged-system", "csr-access"},
}

@dataclass(frozen=True)
class RVProgramBatch:
    case_family: str
    testcases: tuple[TestCase, ...]
    ledger: ObligationLedger

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "rvgen-batch-v1",
            "case_family": self.case_family,
            "admitted_case_count": len(self.testcases),
            "case_index": [
                {
                    "testcase_id": item.testcase_id,
                    "base_program_id": item.base_program_id,
                    "input_id": item.input_id,
                    "form": item.generation_rule_id.partition(":")[0],
                    "generation_rule_id": item.generation_rule_id,
                    "isa_profile": item.isa_profile,
                    "disposition": item.dataflow_meta["realized_facts"]["disposition"],
                    "code_sha256": hashlib.sha256(item.code_bytes).hexdigest(),
                }
                for item in self.testcases
            ],
            "ledger": self.ledger.to_dict(),
        }


def _fresh_testcases(testcases: tuple[TestCase, ...]) -> tuple[TestCase, ...]:
    return tuple(replace(item, dataflow_meta=deepcopy(item.dataflow_meta)) for item in testcases)


def _forms_for_view(case_family: str) -> tuple[OfficialForm, ...]:
    if case_family == "all":
        return OFFICIAL_ALL_CATALOG_FORMS
    if case_family == "compressed":
        return tuple(f for f in OFFICIAL_ALL_CATALOG_FORMS if f.encoding_length_bytes == 2)
    if case_family == "legality-only":
        return tuple(form for form in OFFICIAL_ALL_CATALOG_FORMS if generation_schema_for_form(form) is None)
    kinds = {"encoding-only"} if case_family == "encoding-only" else _VIEW_EFFECT_KINDS[case_family]
    return tuple(
        form
        for form in OFFICIAL_ALL_CATALOG_FORMS
        if (schema := generation_schema_for_form(form)) is not None
        and (
            schema.kind in kinds
            or case_family == "vector"
            and schema.operation.startswith("vector-config")
        )
    )


@cache
def _generate_corpus(
    forms: tuple[OfficialForm, ...], xlen: int | None = None,
) -> tuple[tuple[TestCase, ...], ObligationLedger]:
    """Generate only *forms*; the default remains the complete catalog.

    Family views call the same deterministic loop with their selected forms;
    projections remain local slices.
    """
    accepted: list[TestCase] = []
    admitted_by_identity: set[str] = set()
    scheduled_obligations: set[str] = set()
    ledger = ObligationLedger()
    for rule in tuple(generation_rule_for_form(form) for form in forms):
        form, schema = rule.form, rule.schema
        for realization in rule.realizations:
            if xlen is not None and realization.xlen != xlen:
                continue
            intents = boundary_intents(form, schema, realization)
            for intent in sorted(intents, key=lambda item: (item.lane != "normal-defined", item.key())):
                obligation = intent.key()
                fallback = disposition_for_schema(schema, str(intent.lane))
                fallback = "pending" if fallback == "value-normal" else fallback
                materialization_reason = None
                try:
                    testcase = materialize(intent)
                except UnrealizedState as exc:
                    testcase = None
                    materialization_reason = str(exc)
                if testcase is not None:
                    data = testcase.dataflow_meta["generation_realization"]
                    realized = SemanticRealization.from_dict(data)
                    obligation = replace(
                        intent,
                        realization_xlen=realized.xlen,
                        realization_isa_profile=realized.isa_profile,
                        realization_extension_tokens=realized.extension_tokens,
                        realization_register_carrier=realized.register_carrier,
                        realization_privilege_class=realized.privilege_class,
                        realization_environment_profile=realized.environment_profile,
                        realization_id=str(data["realization_id"]),
                        normative_obligation=normative_obligation_for(
                            form,
                            schema_for_realization(
                                form, schema, xlen=realized.xlen,
                                register_carrier=realized.register_carrier,
                            ),
                            realized,
                            lane=intent.lane,
                            boundary_class=intent.boundary_class,
                        ),
                    ).key()
                if obligation in scheduled_obligations:
                    continue
                scheduled_obligations.add(obligation)
                ledger.scheduled[obligation] += 1
                realized_facts = testcase.dataflow_meta.get("realized_facts") if testcase is not None else None
                disposition = realized_facts.get("disposition") if isinstance(realized_facts, dict) else None
                ledger.disposition_by_obligation.setdefault(
                    obligation, str(disposition if disposition is not None else fallback),
                )
                if testcase is None:
                    ledger.stage_by_obligation[obligation] = "materialization"
                    ledger.reason_by_obligation[obligation] = (
                        materialization_reason or "materialize-returned-none"
                    )
                    continue
                result = admit_testcase(testcase, intent)
                if not result.admitted:
                    ledger.stage_by_obligation[obligation] = result.stage or "admission"
                    ledger.reason_by_obligation[obligation] = result.reason or "not-admitted"
                    continue
                classified = replace(
                    testcase,
                    dataflow_meta={
                        **testcase.dataflow_meta,
                        "semantic_point": dict(result.semantic_point.to_dict()),
                    },
                )
                identified = identify_testcase(classified)
                identity = content_identity(classified)
                if identity not in admitted_by_identity:
                    admitted_by_identity.add(identity)
                    accepted.append(identified)
                case_ids = ledger.case_ids_by_obligation.setdefault(obligation, ())
                if identified.testcase_id not in case_ids:
                    ledger.case_ids_by_obligation[obligation] = (*case_ids, identified.testcase_id)
                    ledger.covered[obligation] += 1
                ledger.stage_by_obligation[obligation] = "admission"
                ledger.reason_by_obligation[obligation] = "admitted"
    return tuple(accepted), ledger


def _resolve_form(form: OfficialForm | str) -> OfficialForm:
    if isinstance(form, OfficialForm):
        canonical = OFFICIAL_ALL_CATALOG_BY_MNEMONIC.get(form.mnemonic)
        selected = OFFICIAL_SELECTED_SCALAR_BY_MNEMONIC.get(form.mnemonic)
        if canonical is None or form not in (canonical, selected):
            raise ValueError(f"unknown catalog form: {form.mnemonic}")
        return canonical
    if isinstance(form, str):
        mnemonic = form.replace(".", "_")
        form = OFFICIAL_ALL_CATALOG_BY_MNEMONIC.get(mnemonic)
        if form is None:
            raise ValueError(f"unknown catalog form: {mnemonic}")
        return form
    raise TypeError("form must be an OfficialForm or mnemonic string")


def generate_form(form: OfficialForm | str) -> tuple[TestCase, ...]:
    """Return admitted Direct rows owned by one catalog form."""
    return _fresh_testcases(_generate_corpus((_resolve_form(form),))[0])


def generate_batch(
    case_family: str = "all", *, form: OfficialForm | str | None = None,
) -> RVProgramBatch:
    """Generate a catalog view of Direct rows."""
    requested_family = case_family
    xlen = None
    if isinstance(case_family, str) and case_family[:4] in {"rv32", "rv64"} \
            and case_family[4:5] == "-":
        xlen, case_family = int(case_family[2:4]), case_family[5:]
    if case_family not in set(_VIEW_EFFECT_KINDS) | {
        "all", "compressed", "legality-only", "encoding-only"
    }:
        raise ValueError(f"unknown case_family: {requested_family}")
    forms = _forms_for_view(case_family)
    if form is not None:
        selected = _resolve_form(form)
        if selected not in forms:
            raise ValueError(f"form is not in case_family: {selected.mnemonic}")
        forms = (selected,)
    testcases, ledger = _generate_corpus(forms, xlen)
    testcases = _fresh_testcases(testcases)
    ledger = ObligationLedger(
        Counter(ledger.scheduled), Counter(ledger.covered),
        dict(ledger.disposition_by_obligation),
        {key: tuple(value) for key, value in ledger.case_ids_by_obligation.items()},
        dict(ledger.stage_by_obligation), dict(ledger.reason_by_obligation),
    )
    return RVProgramBatch(
        case_family=requested_family,
        testcases=testcases,
        ledger=ledger,
    )

"""Static admission: the single gate between an assembled program and the corpus.

Admission never trusts the intent that produced a testcase.  It re-derives what
the program actually is -- its encoding, its state, its observability and its
lane -- and classifies it from that, so a boundary that failed to materialise
the way the formula expected is reclassified rather than mislabelled.
"""

from dataclasses import dataclass, replace
from functools import cache

from ..direct_case import (
    FFLAGS_OBSERVER_OFFSET,
    FFLAGS_OBSERVER_REGISTER,
    HARNESS_RESERVED_GPRS,
    INSTRUCTION_MEMORY_ADDRESS_REGISTER,
    INSTRUCTION_MEMORY_LINK_POISON,
    INSTRUCTION_MEMORY_LINK_OFFSET,
    INSTRUCTION_MEMORY_MARKER_REGISTER,
    INSTRUCTION_MEMORY_OBSERVER_OFFSET,
    INSTRUCTION_MEMORY_PATCH_OFFSET,
    INSTRUCTION_MEMORY_REGION_BYTES,
    INSTRUCTION_MEMORY_RETURN_REGISTER,
    INSTRUCTION_MEMORY_WORD_REGISTER,
    LOGICAL_DATA_ADDRESS,
    MEMORY_BASE_REGISTER,
    OBSERVABLE_REGION,
    PRODUCER_LOAD_OFFSET,
    PRODUCER_SOURCE_POISON,
    SemanticPoint,
    TestCase,
)
from .._translation_cell import (
    generation_realization_for_testcase,
)
from ..riscv_csrs import (
    csr_is_read_only,
    csr_is_rv32_only,
    csr_minimum_privilege,
)
from ..riscv_catalog import catalog_partition_for_form
from ..riscv_encoding import form_for_mnemonic as _emitted_form
from ..riscv_encoding import (
    branch_compare_immediate,
    decode_form,
    decoded_register_slots,
    decode_register_field_value,
    encode_instruction,
    canonical_immediate_field,
    implicit_source_roles,
    operand_groups,
    operand_kind,
    register_domain,
    sign_extend,
    source_roles,
)
from ..spec_definedness import (
    SemanticRealization,
    _compatibility_profile_matches,
    _profile_xlen_matches_form,
    enabled_extensions,
    profile_enables_form,
    profile_uses_compressed_encoding,
    rv32_xregister_writeback_compatibility_active,
    xregister_compatibility_slot_domain_for_mnemonic,
)
from .boundaries import (
    IEEE_FORMAT_BY_PRECISION,
    branch_condition_holds,
    canonical_nan_bits,
    compressed_codeword_disposition,
    declared_gpr_writeback_value,
    semantic_immediate_value,
    stack_seed_value,
    stack_shape,
    stack_slot_offsets,
    vector_state_boundary,
    xlen_mask,
    _project_gpr_result,
)
from .effects import (
    EffectSchema,
    generation_owner_contract,
    generation_rule_for_form,
    generation_schema_for_form,
    producer_chain_role,
    rv32_xregister_pair_roles,
    schema_for_realization,
    trap_outcome_kind_for_form,
    vector_dynamic_state_keys,
    vector_fault_only_first,
    vector_profile_vtype_violation,
    vector_mask_operand,
    vector_register_violation,
    disposition_for_schema,
)
from .intents import (
    CONSUMER_REGISTER,
    LANE_LEGALITY,
    LANE_NORMAL,
    MaterializationIntent,
    _csr_required_extensions_for_materialized_profile,
    _profile_enables_emitted_form,
    boundary_intents,
    per_form_prerequisite_extension_sets,
    rv32_compatibility_extension_suffix,
)
from .obligations import normative_obligation_for
from .ledger import _execution_path_gap
from ..state_contract import STATE_OBSERVER_KEYS_BY_DOMAIN, state_contract_audit
from .materialize import MEMORY_ACCESS_OFFSET

_FP_LOAD_PRECISION = {"flh": 11, "flw": 24, "fld": 53, "flq": 113}
_FP_MOVE_SOURCE_WIDTH = {"fmv.h.x": 16, "fmv.w.x": 32, "fmv.d.x": 64}
_GPR_PRODUCER_LOAD = {
    "lb": (1, True),
    "lbu": (1, False),
    "lh": (2, True),
    "lhu": (2, False),
    "lw": (4, True),
    "lwu": (4, False),
    "ld": (8, True),
}
_FP_SEED_LOAD = {11: "flh", 24: "flw", 53: "fld", 113: "flq"}
_FP_NAN_BOX_SEED_LOAD = {11: "flw", 24: "fld", 53: "flq", 113: "flq"}


@cache
def _schema_for_profile(isa_profile: str, form) -> EffectSchema | None:
    """Return the concrete effect view for a testcase/decoded profile."""
    if form is None:
        return None
    xlen = 32 if str(isa_profile).lower().startswith("rv32") else 64
    carrier = "gpr" if any(
        register_domain(form, role) == "fpr"
        and xregister_compatibility_slot_domain_for_mnemonic(
            isa_profile, form.mnemonic, role, 0
        ) == "gpr"
        for role in operand_groups(form)
    ) else "fpr"
    return schema_for_realization(
        form,
        generation_schema_for_form(form),
        xlen=xlen,
        register_carrier=carrier,
    )


def _observer_store_mnemonic(testcase: TestCase) -> str:
    """Return the XLEN-sized scalar store used by the materialized observer."""
    return "sw" if str(testcase.isa_profile).lower().startswith("rv32") else "sd"


def _compatibility_writeback_schema(
    isa_profile: str,
    form,
    schema: EffectSchema | None,
) -> EffectSchema | None:
    if schema is None or not rv32_xregister_writeback_compatibility_active(isa_profile, form):
        return schema
    return replace(schema, writeback="gpr", observer="gpr")


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    stage: str | None
    reason: str | None
    semantic_point: SemanticPoint | None

def _execution_path_audit(testcase: TestCase) -> str | None:
    raw_expected = testcase.dataflow_meta.get("expected_executed_instruction_ids")
    if raw_expected is None:
        expected = ()
    elif (
        not isinstance(raw_expected, (list, tuple))
        or any(not isinstance(item, str) or not item for item in raw_expected)
    ):
        return "expected-execution-must-be-an-ordered-sequence"
    else:
        expected = tuple(raw_expected)
    if len(expected) != len(set(expected)):
        return "expected-execution-must-not-repeat-instructions"
    required = {
        item.instruction_id
        for item in testcase.instruction_meta
        if "risk-sequence" in item.tags
    }
    if not required <= set(expected):
        return "expected-execution-must-include-risk-sequence"
    if gap := _execution_path_gap(testcase.instruction_meta, set(expected)):
        return gap
    if testcase.dataflow_meta.get("lifecycle_state_domain") == "lifecycle":
        return None
    shape = next((str(tag).removeprefix("program-shape:") for tag in testcase.layout_tags if str(tag).startswith("program-shape:")), "")
    if shape == "consumer-jump-target-store":
        selected = "1" if "jtt1" in expected else "0"
        expected_jump_path = (
            "jtmask", "jthi", "jtlo", "jtadd", "cons",
            f"jtt{selected}", f"jtobs{selected}", f"jtret{selected}", "jtskip",
        )
        try:
            jump_start = expected.index("jtmask")
        except ValueError:
            return "expected-execution-jump-target-store-path"
        return (
            None
            if expected[jump_start:] == expected_jump_path
            else "expected-execution-jump-target-store-path"
        )
    positions = {
        item.instruction_id: index
        for index, item in enumerate(testcase.instruction_meta)
    }
    actual_positions = [positions[item] for item in expected]
    if actual_positions != sorted(actual_positions):
        return "expected-execution-order-does-not-match-layout"
    return None


def _encoding_only_branch_path_audit(
    testcase: TestCase,
    risk,
    form,
    schema: EffectSchema | None,
) -> str | None:
    if (
        schema is not None and schema.kind != "encoding-only"
    ) or (
        _declared_lane(testcase) != LANE_NORMAL
        or "program-shape:single" not in testcase.layout_tags
        or risk.mnemonic not in {"beqi", "bnei"}
    ):
        return None
    word = int.from_bytes(
        testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
        "little",
    )
    decoded = decode_form(form, word)
    if "frm-write" in risk.tags:
        immediate = decoded.get("zimm5")
    else:
        field = canonical_immediate_field(form)
        immediate = None if field is None or field not in decoded else int(decoded[field])
        if immediate is not None and field == "c_imm6" and any(name.startswith("c_nzuimm6") for name in form.variable_fields):
            immediate = int(decoded[field])
        elif immediate is not None and field == "bimm12":
            immediate = sign_extend(immediate, 13)
        elif immediate is not None and field == "jimm20":
            immediate = sign_extend(immediate, 21)
        elif immediate is not None and field == "c_jimm11":
            immediate = sign_extend(immediate, 12)
        elif immediate is not None:
            immediate = semantic_immediate_value(field, immediate)
    source = _actual_gpr_source_value(testcase, risk, "rs1")
    compare = branch_compare_immediate(form, decoded)
    if immediate is None or source is None or compare is None:
        return "state-encoding-only-branch-source-unavailable"
    xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
    operation = {0x2: "beq", 0x3: "bne"}.get((word >> 12) & 0x7)
    taken = branch_condition_holds(int(source), int(compare), operation, xlen=xlen)
    if not taken:
        return None
    target = (
        int(testcase.code_address) + int(risk.byte_offset) + int(immediate)
    ) & xlen_mask(xlen)
    alignment = 2 if profile_uses_compressed_encoding(testcase.isa_profile) else 4
    if target % alignment:
        return "state-encoding-only-branch-target-misaligned"
    expected = tuple(
        testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()
    )
    if target != int(testcase.exit_checkpoint_address) or expected != (risk.instruction_id,):
        return "state-encoding-only-branch-target-unreachable"
    return None


def _risk_contract_audit(testcase: TestCase) -> str | None:
    risks = tuple(item for item in testcase.instruction_meta if "risk" in item.tags)
    contracts = testcase.dataflow_meta.get("risk_contracts")
    if not isinstance(contracts, (list, tuple)) or len(contracts) != len(risks):
        return "risk-contracts-must-match-risk-instructions"
    for item, contract in zip(risks, contracts):
        form = _emitted_form(item.mnemonic)
        if not isinstance(contract, dict) or form is None:
            return "risk-contract-is-invalid"
        if (
            contract.get("instruction_id") != item.instruction_id
            or contract.get("form") != form.mnemonic
            or contract.get("lane") != _declared_lane(testcase)
            or not contract.get("boundary_class")
        ):
            return "risk-contract-does-not-match-risk-instruction"
    return None


_OBSERVER_STORE_BYTES = {
    "sb": 1, "sh": 2, "sw": 4, "sd": 8,
    "sb.rl": 1, "sh.rl": 2, "sw.rl": 4, "sd.rl": 8,
    "fsh": 2, "fsw": 4, "fsd": 8, "fsq": 16,
    "c.sb": 1, "c.sh": 2, "c.sw": 4, "c.sd": 8,
    "c.fsw": 4, "c.fsd": 8,
    "c.swsp": 4, "c.sdsp": 8, "c.fswsp": 4, "c.fsdsp": 8,
    "amocas.b": 1, "amocas.h": 2, "amocas.w": 4, "amocas.d": 8,
    "amocas.q": 16,
}


def _observation_region_audit(
    testcase: TestCase,
    contract,
    schema: EffectSchema | None,
) -> str | None:
    if contract.sink_transform == "identity":
        return None
    region = next(
        (item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION),
        None,
    )
    if region is None:
        return "observable-region-is-required"
    if int(region.address) != LOGICAL_DATA_ADDRESS:
        return "observable-region-address-does-not-match-logical-base"
    if int(testcase.initial_gpr[MEMORY_BASE_REGISTER]) != LOGICAL_DATA_ADDRESS:
        return "observable-base-register-does-not-match-logical-address"
    shape = next((str(tag).removeprefix("program-shape:") for tag in testcase.layout_tags if str(tag).startswith("program-shape:")), "")
    address_base = next(
        (item.rd for item in testcase.instruction_meta if item.instruction_id == "addr"),
        None,
    ) if shape == "consumer-address-store" else None
    expected = set(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ())
    if schema is not None and schema.kind in {"vector-data", "vector-memory"}:
        if len(region.data) < 512:
            return "vector-observer-region-is-too-short"
    for item in testcase.instruction_meta:
        if "observable" not in item.tags or item.instruction_id not in expected:
            continue
        width = _OBSERVER_STORE_BYTES.get(item.mnemonic)
        if width is not None and "risk" not in item.tags:
            if item.rs1 != MEMORY_BASE_REGISTER and item.rs1 != address_base:
                return "observable-store-base-does-not-match-logical-base"
            offset = _observer_immediate(item)
            if offset is None or offset < 0 or offset + width > len(region.data):
                return "observable-store-does-not-fit-region"
        if "risk" in item.tags and schema is not None and schema.kind in {
            "memory-load", "memory-store", "atomic-rmw", "vector-memory",
        }:
            if (
                _declared_lane(testcase) == LANE_LEGALITY
                and str(testcase.dataflow_meta.get("legality_boundary_class", ""))
                .startswith("memory-environment:")
            ):
                continue
            source = _actual_gpr_source_value(testcase, item, "rs1")
            xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
            address = None if source is None else (
                int(source) + int(item.immediate or 0)
            ) & xlen_mask(xlen)
            width = max(int(schema.memory_width or schema.width) // 8, 1)
            if (
                address is None
                or address < int(region.address)
                or address + width > int(region.address) + len(region.data)
            ):
                return "memory-observer-does-not-fit-region"
    return None


def _state_audit(testcase: TestCase) -> str | None:
    if not testcase.initial_memory_regions:
        return "state-requires-an-initial-memory-region"
    xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
    if any(type(value) is not int or not 0 <= value < 1 << xlen for value in testcase.initial_gpr):
        return "state-initial-gpr-exceeds-xlen"
    if testcase.dataflow_meta.get("sail_branch_contract") != "rvgen-sail-branch-reconciliation-v1":
        return "state-sail-branch-contract-invalid"
    shape_tags = tuple(
        tag for tag in getattr(testcase, "layout_tags", ())
        if isinstance(tag, str) and tag.startswith("program-shape:")
    )
    shape = shape_tags[0].removeprefix("program-shape:") if len(shape_tags) == 1 else ""
    if shape not in {"single", "producer-risk"} and not shape.startswith("consumer-"):
        return "state-program-shape-invalid"
    tagged = tuple(getattr(item, "tags", ()) for item in testcase.instruction_meta)
    if shape == "single" and any(
        "sequence-producer" in tags or "risk-sequence" in tags for tags in tagged
    ):
        return "state-program-shape-mismatch"
    if shape == "producer-risk" and not any("sequence-producer" in tags for tags in tagged):
        return "state-program-shape-mismatch"
    if shape.startswith("consumer-") and not any("risk-sequence" in tags for tags in tagged):
        return "state-program-shape-mismatch"
    if len(testcase.block_meta) != 1:
        return "state-requires-one-code-block"
    block = testcase.block_meta[0]
    if (
        block.start_offset != 0
        or block.end_offset != len(testcase.code_bytes)
        or not block.executed_expected
    ):
        return "state-code-block-must-cover-program"
    raw_expected = testcase.dataflow_meta.get("expected_executed_instruction_ids")
    if not isinstance(raw_expected, (list, tuple)):
        return "state-expected-execution-must-be-an-ordered-sequence"
    expected = tuple(raw_expected)
    if not expected:
        return "state-requires-expected-execution"
    if any(not isinstance(item, str) or not item for item in expected):
        return "state-expected-execution-must-be-an-ordered-sequence"
    known = {item.instruction_id for item in testcase.instruction_meta}
    if not set(expected) <= known:
        return "expected-execution-references-unknown-instruction"
    risk_ids = tuple(
        item.instruction_id for item in testcase.instruction_meta if "risk" in item.tags
    )
    declared_risk_ids = testcase.dataflow_meta.get("risk_instruction_ids")
    if not isinstance(declared_risk_ids, (list, tuple)):
        return "state-risk-instruction-ids-required"
    if tuple(declared_risk_ids) != risk_ids:
        return "state-risk-instruction-ids-must-match"
    if not risk_ids:
        return "state-requires-risk-instruction"
    contract = testcase.observability_contract
    if contract is not None and not set(contract.sink_instruction_ids) <= set(expected):
        return "state-expected-execution-must-include-observer-sinks"
    if (reason := _execution_path_audit(testcase)) is not None:
        return reason
    if (reason := _risk_contract_audit(testcase)) is not None:
        return reason
    # RVGEN owns this static metadata gate; Sail remains the authority for
    # executed state and trace results.
    state_contract = state_contract_audit(testcase)
    if state_contract.get("status") not in {"ready", "not-required"}:
        return f"state-observer-contract-{state_contract.get('status')}"
    risk_items = tuple(item for item in testcase.instruction_meta if "risk" in item.tags)
    risk = risk_items[0]
    form = _emitted_form(risk.mnemonic)
    schema = _schema_for_profile(testcase.isa_profile, form)
    if form is None or testcase.generation_rule_id != generation_rule_for_form(form).rule_id:
        return "state-generation-rule-mismatch"
    catalog_tags = tuple(
        tag for tag in testcase.functional_tags
        if isinstance(tag, str) and tag.startswith("catalog:")
    )
    if catalog_tags != (f"catalog:{catalog_partition_for_form(form)}",):
        return "state-catalog-partition-mismatch"
    state_domain = str(testcase.dataflow_meta.get("state_domain") or "")
    observer_domain = (
        "lifecycle"
        if testcase.dataflow_meta.get("lifecycle_state_domain") == "lifecycle"
        else state_domain
    )
    observer_keys = testcase.dataflow_meta.get("state_observer_keys") or ()
    allowed_observer_keys = set(STATE_OBSERVER_KEYS_BY_DOMAIN.get(observer_domain, ()))
    if (
        schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and _declared_lane(testcase) != LANE_LEGALITY
    ):
        declared_dynamic = testcase.dataflow_meta.get("vector_dynamic_state_keys")
        expected_dynamic = vector_dynamic_state_keys(form, schema)
        if tuple(declared_dynamic or ()) != expected_dynamic:
            return "vector-dynamic-observer-keys-do-not-match-effect"
        allowed_observer_keys.update(expected_dynamic)
    if any(key not in allowed_observer_keys for key in observer_keys):
        return "state-observer-keys-contain-unknown-field"
    if (
        schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and _declared_lane(testcase) != LANE_LEGALITY
    ):
        expected_status = (
            "vector-scalar-observer-ready"
            if schema.kind == "vector-data" and schema.writeback in {"gpr", "fpr"}
            else "vector-register-trace-pending"
            if schema.kind not in {"vector-data", "vector-memory"}
            and not (
                schema.kind == "may-be-operation"
                and schema.operation.startswith("vector-config")
            )
            else "vector-csr-trace-pending"
        )
        if testcase.dataflow_meta.get("state_observer_status") != expected_status:
            return "state-observer-status-does-not-match-effect"
        if schema.kind == "vector-data" and not {
            "vector.mask", "vector.tail",
        } <= set(observer_keys):
            return "state-observer-keys-missing-vector-mask-tail"
    if schema is not None and schema.state_domain == "privileged" and _declared_lane(testcase) != LANE_LEGALITY:
        return "privileged-state-observer-pending"
    if (
        _declared_lane(testcase) == LANE_NORMAL
        and schema is not None
        and schema.kind == "vector-data"
        and schema.writeback in {"gpr", "fpr"}
        and not _vector_scalar_observer_realized(testcase, risk, schema)
    ):
        return "vector-scalar-observer-pending"
    expected_permissions = (
        "rwx"
        if _declared_lane(testcase) == LANE_NORMAL
        and schema is not None
        and schema.kind == "instruction-memory"
        else "rw"
    )
    if any(region.permissions != expected_permissions for region in testcase.initial_memory_regions):
        return "state-memory-permissions-cannot-be-realized"
    flag_risk = risk_items[-1] if schema is not None and schema.observes_fflags and len(risk_items) > 1 else risk
    if (
        _declared_lane(testcase) == LANE_NORMAL
        and schema is not None
        and schema.observes_fflags
        and not _fflags_observation_realized(testcase, flag_risk)
    ):
        return "fflags-observer-must-immediately-follow-risk"
    if (
        _declared_lane(testcase) == LANE_NORMAL
        and schema is not None
        and schema.kind == "instruction-memory"
        and not any(
            "x" in str(region.permissions)
            for region in testcase.initial_memory_regions
            if region.region_id == OBSERVABLE_REGION
        )
    ):
        return "instruction-memory-region-must-be-executable"
    if contract is not None and (reason := _observation_region_audit(testcase, contract, schema)) is not None:
        return reason
    if (reason := _encoding_only_branch_path_audit(testcase, risk, form, schema)) is not None:
        return reason
    if schema is not None and _declared_lane(testcase) != LANE_LEGALITY:
        if str(testcase.dataflow_meta.get("state_domain") or "") != str(schema.state_domain):
            return "state-domain-does-not-match-effect"
    return None


def _csr_profile_violation(csr, isa_profile: str) -> str | None:
    """Return the CSR profile-violation label for an operand, or None."""
    enabled = enabled_extensions(isa_profile)
    if csr is not None and not _csr_required_extensions_for_materialized_profile(
        int(csr), isa_profile
    ) <= enabled:
        return "extension-disabled"
    if (
        csr is not None
        and csr_is_rv32_only(int(csr))
        and not str(isa_profile).lower().startswith("rv32")
    ):
        return "csr-rv32-only"
    return None


def _definedness_audit(testcase: TestCase) -> str | None:
    legality_lane = _declared_lane(testcase) == LANE_LEGALITY
    for item in testcase.instruction_meta:
        if (
            type(item.pc_offset) is not int
            or type(item.byte_offset) is not int
            or type(item.byte_length) is not int
            or item.pc_offset != item.byte_offset
            or item.byte_offset < 0
            or item.byte_length <= 0
            or item.byte_offset + item.byte_length > len(testcase.code_bytes)
        ):
            return f"instruction-metadata-layout-mismatch:{item.instruction_id}"
        form = _emitted_form(item.mnemonic)
        if form is None:
            return f"definedness-has-no-catalog-form:{item.mnemonic}"
        word = int.from_bytes(
            testcase.code_bytes[item.byte_offset : item.byte_offset + item.byte_length],
            "little",
        )
        decoded = decode_form(form, word)
        encoding_matches = word & form.mask == form.match
        encoded_fields = {
            name: int(decoded[name])
            for name in operand_groups(form)
            if name in decoded
        }
        immediate_field = canonical_immediate_field(form)
        decoded_immediate = (
            None
            if immediate_field is None or immediate_field not in decoded
            else int(decoded[immediate_field])
            if immediate_field == "c_imm6"
            and any(name.startswith("c_nzuimm6") for name in form.variable_fields)
            else semantic_immediate_value(immediate_field, int(decoded[immediate_field]))
        )
        if (
            not encoding_matches and not (legality_lane and "risk" in item.tags)
            or dict(item.operand_fields) != encoded_fields
            or item.immediate != decoded_immediate
            or any(
            getattr(item, role) != value
            for role, value in decoded_register_slots(form, word).items()
            )
        ):
            return f"instruction-metadata-encoding-mismatch:{item.instruction_id}"
        for role, register in decoded_register_slots(form, word).items():
            if getattr(item, f"{role}_domain", None) != xregister_compatibility_slot_domain_for_mnemonic(
                testcase.isa_profile, item.mnemonic, role, register
            ):
                return f"instruction-metadata-register-domain-mismatch:{item.instruction_id}"
        if legality_lane and "risk" in item.tags:
            continue
        if not _profile_enables_emitted_form(testcase.isa_profile, form):
            return f"instruction-outside-isa-profile:{item.mnemonic}"
        csr = decoded.get("csr")
        if _csr_profile_violation(csr, testcase.isa_profile) is not None:
            return f"instruction-outside-isa-profile:{item.mnemonic}"
        # RV32E has only x0..x15.  Check every materialized instruction, not
        # just the risk, so an observer/helper accidentally left on x20 cannot
        # be admitted after the scalar sd->sw normalization.
        if str(testcase.isa_profile).lower().startswith("rv32e"):
            for register, domain in (
                (item.rd, item.rd_domain),
                (item.rs1, item.rs1_domain),
                (item.rs2, item.rs2_domain),
                (item.rs3, item.rs3_domain),
            ):
                if register is not None and register >= 16 and domain in {None, "gpr"}:
                    return "rve-register-domain"
            # Zcmp encodes s-registers in c_sreg1/c_sreg2 rather than rd/rs*;
            # decode those fields too so a non-risk helper cannot smuggle x18+
            # into an RV32E testcase.
            for field in ("c_sreg1", "c_sreg2"):
                encoded = decoded.get(field)
                if encoded is not None and decode_register_field_value(field, int(encoded)) >= 16:
                    return "rve-register-domain"
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    form = _emitted_form(risk.mnemonic)
    if _schema_for_profile(testcase.isa_profile, form) is None and testcase.compare_mask.memory_region_ids:
        return "legality-only-form-cannot-claim-a-value-observation"
    return None


def _observability_audit(testcase: TestCase) -> str | None:
    lane = _declared_lane(testcase)
    contract = testcase.observability_contract
    if contract is None:
        return "testcase-requires-an-observability-contract"
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    form = _emitted_form(risk.mnemonic)
    schema = _schema_for_profile(testcase.isa_profile, form)
    expected = set(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ())
    risk_ids = tuple(item.instruction_id for item in testcase.instruction_meta if "risk" in item.tags)
    sink_ids = tuple(
        item.instruction_id for item in testcase.instruction_meta
        if "observable" in item.tags and item.instruction_id in expected
    )
    vector_state = schema is not None and schema.state_domain in {"vector", "vector-memory"}
    vector_scalar = schema is not None and schema.kind == "vector-data" and schema.writeback in {"gpr", "fpr"}
    legality_boundary = str(testcase.dataflow_meta.get("legality_boundary_class", ""))
    if lane == LANE_LEGALITY:
        expected_kind, expected_sinks, expected_transform = "trap-outcome", risk_ids, "identity"
        expected_fields = {"outcome", "signal", "fault_pc", "trap.cause", "trap.epc", "trap.tval"}
        if "misaligned" in legality_boundary or legality_boundary == "memory-environment:access-fault":
            expected_fields.add("fault_address")
        expected_lossiness = "none"
    elif sink_ids:
        expected_kind = "vector-state" if vector_state and not vector_scalar else "destination-register"
        expected_sinks, expected_transform = sink_ids, "vector-store" if vector_state and not vector_scalar else "full-width-store"
        expected_fields = {f"memory.{OBSERVABLE_REGION}"
                          }
        expected_lossiness = (
            "partial" if vector_state and not vector_scalar
            and str(testcase.dataflow_meta.get("vector_boundary", "")).startswith(
                ("vector-vl:", "vector-tail:", "vector-mask:")
            ) else "none"
        )
    else:
        expected_kind, expected_sinks, expected_transform = "instruction-outcome", risk_ids, "identity"
        expected_fields, expected_lossiness = {"outcome", "checkpoint_pc"}, "none"
    risk_positions = [index for index, item in enumerate(testcase.instruction_meta) if "risk" in item.tags]
    sink_positions = [index for index, item in enumerate(testcase.instruction_meta)
                      if item.instruction_id in expected_sinks]
    expected_distance = (
        max(1, min(3, sink_positions[0] - risk_positions[-1]))
        if risk_positions and sink_positions and sink_positions[0] > risk_positions[-1] else 1
    )
    if contract.sink_transform != "identity" and any(
        item.instruction_id in contract.sink_instruction_ids
        and item.instruction_id != risk.instruction_id
        and item.byte_offset < risk.byte_offset + risk.byte_length
        for item in testcase.instruction_meta
    ):
        return "observer-sink-must-follow-risk"
    if (
        contract.risk_value_or_effect != expected_kind
        or tuple(contract.sink_instruction_ids) != tuple(expected_sinks)
        or contract.risk_to_sink_distance != expected_distance
        or contract.sink_transform != expected_transform
        or set(contract.final_observation_fields) != expected_fields
        or contract.lossiness != expected_lossiness
    ):
        return "observability-contract-does-not-match-case"
    if lane == LANE_LEGALITY and not testcase.compare_mask.outcome:
        return "legality-lane-must-compare-outcome"
    if lane != LANE_LEGALITY and (
        not testcase.compare_mask.outcome or not testcase.compare_mask.checkpoint_pc
    ):
        return "normal-lane-must-compare-outcome-and-checkpoint"
    expected_mask = (
            (True, False, (), (), True, True, (
                "misaligned" in legality_boundary
                or legality_boundary == "memory-environment:access-fault"
            ), ("trap.cause", "trap.epc", "trap.tval"))
        if lane == LANE_LEGALITY
        else (
            True,
            True,
            (),
            (OBSERVABLE_REGION,) if contract.sink_transform != "identity" else (),
            False,
            False,
            False,
            tuple(vector_dynamic_state_keys(form, schema))
        )
    )
    actual_mask = (
        bool(testcase.compare_mask.outcome),
        bool(testcase.compare_mask.checkpoint_pc),
        tuple(testcase.compare_mask.gpr_indices),
        tuple(testcase.compare_mask.memory_region_ids),
        bool(testcase.compare_mask.signal),
        bool(testcase.compare_mask.fault_pc),
        bool(testcase.compare_mask.fault_address),
        tuple(testcase.compare_mask.extra_state_keys),
    )
    if actual_mask != expected_mask:
        return "compare-mask-does-not-match-case"
    if lane == LANE_LEGALITY:
        if contract.sink_transform != "identity":
            return "legality-lane-must-not-claim-a-value-observer"
        if not {"outcome", "signal", "fault_pc"} <= set(
            contract.final_observation_fields
        ):
            return "legality-lane-requires-trap-outcome-fields"
    elif contract.sink_transform == "identity":
        if (
            not testcase.compare_mask.outcome
            or "outcome" not in contract.final_observation_fields
        ):
            return "outcome-only-contract-must-observe-outcome"
        if testcase.compare_mask.memory_region_ids:
            return "outcome-only-program-must-not-claim-a-region-observation"
    elif OBSERVABLE_REGION not in testcase.compare_mask.memory_region_ids:
        return "observer-must-compare-the-observable-region"
    if set(testcase.compare_mask.gpr_indices) & HARNESS_RESERVED_GPRS:
        return "observer-must-not-compare-frame-registers"
    expected = set(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ())
    sinks = {
        item.instruction_id
        for item in testcase.instruction_meta
        if item.instruction_id in expected
        and (
            "observable" in item.tags
            or (contract.sink_transform == "identity" and "risk" in item.tags)
        )
    }
    if not sinks or set(contract.sink_instruction_ids) != sinks:
        return "observability-contract-must-name-exact-sinks"
    if contract.sink_transform == "full-width-store":
        if any(
            (item := next((row for row in testcase.instruction_meta
                           if row.instruction_id == sink), None)) is None
            or item.mnemonic not in _OBSERVER_STORE_BYTES
            or item.rs2 is None
            or (
                schema is not None
                and schema.writeback == "gpr"
                and item.instruction_id != risk.instruction_id
                and item.mnemonic != _observer_store_mnemonic(testcase)
            )
            for sink in contract.sink_instruction_ids
        ):
            return "observer-full-width-store-not-realized"
    return None


def _declared_lane(testcase: TestCase) -> str:
    mask = testcase.compare_mask
    trap = bool(mask.signal or mask.fault_pc or mask.fault_address)
    return LANE_LEGALITY if trap else LANE_NORMAL


def _risk_violation_set(testcase: TestCase, risk) -> frozenset[str]:
    """The legality facts derivable from concrete bytes, state and profile."""
    form = _emitted_form(risk.mnemonic)
    if form is None:
        return frozenset({"unknown-form"})
    word = int.from_bytes(
        testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length], "little"
    )
    violations: set[str] = set()
    schema = _schema_for_profile(testcase.isa_profile, form)
    boundary = str(testcase.dataflow_meta.get("legality_boundary_class", ""))
    if word & form.mask != form.match and not boundary.startswith(
        "control-witness:reserved-shamt:"
    ):
        violations.add("encoding")
    if not _profile_xlen_matches_form(testcase.isa_profile, form):
        violations.add("xlen-mismatch")
    elif not profile_enables_form(testcase.isa_profile, form):
        violations.add("extension-disabled")
    compatibility_suffix = rv32_compatibility_extension_suffix(form, schema)
    if (
        compatibility_suffix is not None
        and not str(testcase.isa_profile).lower().startswith("rv32e")
        and _compatibility_profile_matches(
            testcase.isa_profile,
            f"rv32i_{compatibility_suffix}",
        )
    ):
        violations.add("extension-disabled")
    decoded = decode_form(form, word)
    # ``*_n0`` zero is reserved unless the descriptor marks the codeword as a
    # supported HINT lane.
    if any(
        str(field).endswith("_n0") and int(decoded.get(field, -1)) == 0
        for field in getattr(form, "variable_fields", ())
    ) and compressed_codeword_disposition(form, schema, decoded) != "hint":
        violations.add("reserved-encoding")
    if schema is not None and schema.state_domain in {"vector", "vector-memory"}:
        profile_vector_violation = vector_profile_vtype_violation(
            testcase.isa_profile,
            form,
            schema,
            str(testcase.dataflow_meta.get("vector_boundary", "")),
        )
        if profile_vector_violation is not None:
            violations.add(profile_vector_violation)
        else:
            vector_violation = vector_register_violation(
                form,
                schema,
                testcase.dataflow_meta.get("vector_boundary", ""),
                {name: int(value) for name, value in decoded.items() if isinstance(value, int)},
                testcase.isa_profile,
            )
            if vector_violation is not None:
                violations.add(vector_violation)
        if (
            testcase.dataflow_meta.get("vector_boundary") == "vector-frm:reserved"
            and "frm" in vector_dynamic_state_keys(form, schema)
        ):
            violations.add("vector-fp-reserved-frm")
    if str(testcase.isa_profile).lower().startswith("rv32"):
        pair_roles = rv32_xregister_pair_roles(
            form,
            schema,
            xlen=32,
            register_carrier="gpr",
        )
        if pair_roles and any(
            xregister_compatibility_slot_domain_for_mnemonic(
                testcase.isa_profile,
                form.mnemonic,
                role,
                6,
            )
            == "gpr"
            for role in pair_roles
        ):
            for role in pair_roles:
                if int(decoded.get(role, 0)) & 1:
                    violations.add(f"odd-paired-gpr-base-{role}")
                    break
    csr = decoded.get("csr")
    violation = _csr_profile_violation(csr, testcase.isa_profile)
    if violation is not None:
        violations.add(violation)
    if testcase.isa_profile.startswith("rv32e"):
        zcmp_high_sreg = any(
            name in {"c_sreg1", "c_sreg2"} and int(decoded.get(name, 0)) >= 2
            for name in operand_groups(form)
        )
        if zcmp_high_sreg:
            violations.add("zcmp-high-sreg-reserved")
        for register, domain in (
            (risk.rd, risk.rd_domain),
            (risk.rs1, risk.rs1_domain),
            (risk.rs2, risk.rs2_domain),
            (risk.rs3, risk.rs3_domain),
        ):
            if zcmp_high_sreg:
                break
            if domain in {None, "gpr"} and register is not None and register >= 16:
                violations.add("rve-register-domain")
                break
    trap_kind = trap_outcome_kind_for_form(form)
    if (
        trap_kind is not None
        and word & form.mask == form.match
        and profile_enables_form(testcase.isa_profile, form)
    ):
        violations.add(trap_kind)
    if int(decoded.get("rnum", 0)) > 10:
        violations.add("reserved-selector")
    if (
        schema is not None
        and schema.kind == "stack-transfer"
        and word & form.mask == form.match
        and profile_enables_form(testcase.isa_profile, form)
        and int(decoded.get("c_rlist", 4)) < 4
    ):
        violations.add("zcmp-stack-rlist-reserved")
    if (
        schema is not None
        and schema.kind == "integer-shift"
        and schema.immediate_field is not None
        and form.extension_group not in {"xtheadbb", "xtheadbs"}
    ):
        amount = decoded.get(schema.immediate_field)
        xlen_limit = 32 if testcase.isa_profile.startswith("rv32") else 64
        limit = min(int(schema.width), xlen_limit)
        if amount is not None and int(amount) >= limit:
            violations.add("reserved-shift-amount")
        if (
            form.xlen == "rv64"
            and schema.immediate_field == "shamtw"
            and word & (1 << 25)
        ):
            violations.add("reserved-shift-amount")
    if form.mnemonic == "c_addi4spn" and int(decoded.get("c_nzuimm10", 1)) == 0:
        violations.add("compressed-addi4spn-zero")
    if form.mnemonic == "c_lui" and int(decoded.get("c_nzimm18", 1)) == 0:
        violations.add("compressed-lui-zero")
    if form.mnemonic == "cm_mvsa01" and int(decoded.get("c_sreg1", -1)) == int(decoded.get("c_sreg2", -2)):
        violations.add("zcmp-equal-sreg-reserved")
    if schema is not None and schema.writeback == "paired-gpr":
        pair_roles = rv32_xregister_pair_roles(
            form,
            schema,
            xlen=32,
            register_carrier="gpr",
        )
        roles = pair_roles or (("rd", "rs2") if schema.kind == "atomic-rmw" else ())
        if any(int(decoded.get(role, 0)) & 1 for role in roles):
            for role in roles:
                if int(decoded.get(role, 0)) & 1:
                    violations.add(f"odd-paired-gpr-base-{role}")
    if schema is not None and schema.kind == "atomic-rmw" and risk.rs1 is not None:
        address = _actual_gpr_source_value(testcase, risk, "rs1")
        if address is None:
            violations.add("atomic-address-unavailable")
        elif schema.memory_width and int(address) % max(schema.memory_width // 8, 1):
            violations.add("misaligned-atomic")
    if schema is not None and schema.kind in {"memory-load", "memory-store"} and risk.rs1 is not None:
        address = _actual_gpr_source_value(testcase, risk, "rs1")
        if address is None:
            violations.add("memory-address-unavailable")
        else:
            xlen = 32 if testcase.isa_profile.startswith("rv32") else 64
            address = (int(address) + int(risk.immediate or 0)) & xlen_mask(xlen)
        if (
            address is not None
            and not boundary.startswith("memory-environment:")
            and schema.memory_width
            and address % max(schema.memory_width // 8, 1)
        ):
            violations.add("misaligned-memory")
        if boundary == "memory-environment:access-fault":
            violations.add("memory-access-fault")
    if schema is not None and schema.kind == "csr-access":
        csr = int(decoded.get("csr", 0))
        funct3 = (word >> 12) & 0x7
        source = int(decoded.get("zimm5", decoded.get("rs1", 0)))
        # Zicsr suppresses writes only for CSRRS/CSRRC when the source is zero.
        # CSRRW/CSRRWI always perform their write, including a zero source.
        writes = funct3 in {0x1, 0x5} or (
            source != 0 and funct3 in {0x2, 0x3, 0x6, 0x7}
        )
        if csr_minimum_privilege(csr) > 0 or (writes and csr_is_read_only(csr)):
            # One CSR access is one legality cause even if its address has
            # several forbidden bit properties.
            violations.add("csr-access")
    if int(decoded.get("rm", -1)) in {5, 6}:
        # FP rounding modes 5/6 are reserved encodings: executing the form
        # with one of them raises the illegal-instruction trap.  Derived from
        # the operand structure, never a mnemonic table.
        violations.add("reserved-rm")
    if (
        schema is not None
        and schema.kind == "control-jump"
        and word & form.mask == form.match
        and profile_enables_form(testcase.isa_profile, form)
        and not profile_uses_compressed_encoding(testcase.isa_profile)
        and risk.rs1 is not None
    ):
        # JALR clears only bit 0 of the computed target.  Under IALIGN=32 a
        # target whose bit 1 remains set must raise the instruction-address-
        # misaligned exception; the concrete target is the seeded source
        # plus the displacement.
        source = _actual_gpr_source_value(testcase, risk, "rs1")
        if source is not None:
            displacement = int(decoded.get(schema.immediate_field or "", 0) or 0)
            if (int(source) + displacement) & 2:
                violations.add("control-target-misaligned")
    if (
        boundary.startswith("prerequisite-gate:")
        and word & form.mask == form.match
        and profile_enables_form(testcase.isa_profile, form)
    ):
        # The per-form prerequisite gate keeps the encoding profile but
        # removes every semantic prerequisite alternative derived from the
        # form's uncompressed twin.
        alternatives = per_form_prerequisite_extension_sets(form, schema)
        enabled = enabled_extensions(testcase.isa_profile)
        if alternatives and not any(
            alternative <= enabled for alternative in alternatives
        ):
            violations.add("prerequisite-extension-disabled")
    return frozenset(violations)


def actual_violation_set(testcase: TestCase) -> frozenset[str]:
    """The legality facts derivable from every declared risk instruction."""
    violations: set[str] = set()
    for risk in testcase.instruction_meta:
        if "risk" not in risk.tags:
            continue
        violations.update(_risk_violation_set(testcase, risk))
    return frozenset(violations)


@cache
def _intent_materialization_contract(intent):
    from .materialize import materialize

    form = _emitted_form(intent.form)
    case = materialize(intent) if form is not None else None
    return None if case is None else (
        case.isa_profile,
        case.dataflow_meta.get("generation_realization"),
        case.dataflow_meta.get("legality_boundary_class"),
        actual_violation_set(case) if intent.lane == LANE_LEGALITY else frozenset(),
        case.code_bytes,
    )


def _noncanonical_codeword_outcome(testcase: TestCase) -> str | None:
    """Classify enabled descriptor codewords outside the declared form's domain."""
    for risk in testcase.instruction_meta:
        if "risk" not in risk.tags:
            continue
        form = _emitted_form(risk.mnemonic)
        if form is None or not _profile_xlen_matches_form(testcase.isa_profile, form):
            continue
        # An extension-disabled row is an ordinary extension-gate trap.  Its
        # codeword must not be reclassified using the enabled-C disposition.
        if not profile_enables_form(testcase.isa_profile, form):
            continue
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
            "little",
        )
        decoded = decode_form(form, word)
        schema = _schema_for_profile(testcase.isa_profile, form)
        disposition = compressed_codeword_disposition(form, schema, decoded)
        if disposition == "hint":
            contracts = testcase.dataflow_meta.get("risk_contracts", ())
            if any(
                isinstance(contract, dict)
                and str(contract.get("boundary_class", "")).startswith("compressed-hint:")
                for contract in contracts
            ):
                continue
            return "non-canonical-codeword:hint"
        if disposition == "non-canonical":
            return "non-canonical-codeword:canonical-alias"
    return None


def _lane_audit(testcase: TestCase) -> str | None:
    lane = _declared_lane(testcase)
    noncanonical_outcome = _noncanonical_codeword_outcome(testcase)
    if noncanonical_outcome is not None:
        return noncanonical_outcome
    violations = actual_violation_set(testcase)
    if lane != LANE_LEGALITY:
        if violations:
            return "normal-lane-has-concrete-legality-violation"
        return None
    if len(testcase.instruction_meta) != 1:
        boundary = str(testcase.dataflow_meta.get("vector_boundary", ""))
        risk_positions = [
            index
            for index, item in enumerate(testcase.instruction_meta)
            if "risk" in item.tags
        ]
        vector_trap_setup = boundary.startswith("vector-") and all(
            "vector-state-setup" in item.tags
            or "producer" in item.tags
            or "sequence-producer" in item.tags
            for item in testcase.instruction_meta[:-1]
        )
        if (
            not vector_trap_setup
            or len(risk_positions) != 1
            or risk_positions[0] != len(testcase.instruction_meta) - 1
        ):
            return "legality-lane-must-not-execute-past-the-violation"
    if "xlen-mismatch" in violations:
        boundary = str(testcase.dataflow_meta.get("legality_boundary_class", ""))
        if boundary.startswith(("rv32:xlen-gate:", "rv64:xlen-gate:")) and violations == {"xlen-mismatch"}:
            return None
        return "xlen-mismatch"
    if len(violations) != 1:
        return "legality-lane-requires-exactly-one-concrete-violation"
    return None


def _seeded_memory_value(testcase: TestCase, offset: int, width: int) -> int | None:
    region = next((item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION), None)
    if region is None or offset < 0 or offset + width > len(region.data):
        return None
    return int.from_bytes(region.data[offset : offset + width], "little")


def _instruction_by_id(testcase: TestCase, instruction_id: str):
    return next((item for item in testcase.instruction_meta if item.instruction_id == instruction_id), None)


def _fflags_observation_realized(testcase: TestCase, risk) -> bool:
    clear = _instruction_by_id(testcase, "fclr")
    suffix = str(risk.instruction_id)[1:] if str(getattr(risk, "instruction_id", "")).startswith("i") else ""
    read = _instruction_by_id(testcase, f"frd{suffix}") or _instruction_by_id(testcase, "frd")
    store = _instruction_by_id(testcase, f"fobs{suffix}") or _instruction_by_id(testcase, "fobs")
    if not all((clear, read, store)):
        return False
    if (
        clear.mnemonic != "csrrwi" or read.mnemonic != "csrrs"
        or store.mnemonic != _observer_store_mnemonic(testcase)
        or clear.byte_offset >= risk.byte_offset
        or read.byte_offset != risk.byte_offset + risk.byte_length
        or store.byte_offset != read.byte_offset + read.byte_length
        or clear.rd != 0 or read.rd != FFLAGS_OBSERVER_REGISTER or read.rs1 != 0
        or store.rs1 != MEMORY_BASE_REGISTER or store.rs2 != FFLAGS_OBSERVER_REGISTER
        or _observer_immediate(store) != FFLAGS_OBSERVER_OFFSET
    ):
        return False
    if {
        clear.instruction_id, risk.instruction_id, read.instruction_id, store.instruction_id,
    } - set(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()):
        return False
    decoded = []
    for item in (clear, read):
        form = _emitted_form(item.mnemonic)
        word = int.from_bytes(
            testcase.code_bytes[item.byte_offset : item.byte_offset + item.byte_length], "little"
        ) if form is not None else 0
        if form is None or word & form.mask != form.match:
            return False
        decoded.append(decode_form(form, word))
    return (
        int(decoded[0].get("csr", -1)) == 0x001
        and int(decoded[0].get("zimm5", -1)) == 0
        and int(decoded[1].get("csr", -1)) == 0x001
    )


def _pc_relative_target(testcase: TestCase, hi_id: str, lo_id: str, register: int) -> int | None:
    hi = _instruction_by_id(testcase, hi_id)
    lo = _instruction_by_id(testcase, lo_id)
    if hi is None or lo is None:
        return None
    if hi.mnemonic != "auipc" or lo.mnemonic != "addi":
        return None
    if hi.rd != register or lo.rd != register or lo.rs1 != register:
        return None
    hi_imm = 0 if hi.immediate is None else int(hi.immediate)
    lo_imm = 0 if lo.immediate is None else int(lo.immediate)
    return int(hi.byte_offset) + hi_imm + lo_imm


def _instruction_memory_realized(
    testcase: TestCase,
    form,
    schema,
    risk,
) -> bool:
    region = next(
        (
            item
            for item in testcase.initial_memory_regions
            if item.region_id == OBSERVABLE_REGION and "x" in str(item.permissions)
        ),
        None,
    )
    if (
        region is None
        or int(region.address) != LOGICAL_DATA_ADDRESS
        or len(region.data) < INSTRUCTION_MEMORY_REGION_BYTES
    ):
        return False
    if form is None or schema is None:
        return False
    store_mnemonic = _observer_store_mnemonic(testcase)
    target = _instruction_by_id(testcase, "fetch0")
    fetchobs = _instruction_by_id(testcase, "fetchobs")
    fetchret = _instruction_by_id(testcase, "fetchret")
    skip = _instruction_by_id(testcase, "skip")
    if (
        target is None
        or fetchobs is None
        or fetchret is None
        or skip is None
        or target.mnemonic != "addi"
        or target.rd != INSTRUCTION_MEMORY_MARKER_REGISTER
        or target.rs1 != 0
        or fetchobs.mnemonic != store_mnemonic
        or fetchobs.rs1 != MEMORY_BASE_REGISTER
        or fetchobs.rs2 != INSTRUCTION_MEMORY_MARKER_REGISTER
        or int(fetchobs.immediate or -1) != INSTRUCTION_MEMORY_OBSERVER_OFFSET
        or fetchret.mnemonic != "jalr"
        or fetchret.rd != 0
        or int(fetchret.immediate or 0) != 0
        or skip.mnemonic != "jal"
    ):
        return False
    target_offset = _pc_relative_target(
        testcase,
        "imaddrhi",
        "imaddrlo",
        INSTRUCTION_MEMORY_ADDRESS_REGISTER,
    )
    if target_offset != int(target.byte_offset):
        return False
    if schema.operation == "fence-fetch":
        patchld = _instruction_by_id(testcase, "patchld")
        patchst = _instruction_by_id(testcase, "patchst")
        fetchcall = _instruction_by_id(testcase, "fetchcall")
        expected = (
            "imaddrhi",
            "imaddrlo",
            "patchld",
            "patchst",
            "i0",
            "fetchcall",
            "fetch0",
            "fetchobs",
            "fetchret",
            "skip",
        )
        patch_word = encode_instruction(
            "addi",
            rd=INSTRUCTION_MEMORY_MARKER_REGISTER,
            rs1=0,
            immediate=2,
        )
        return (
            patchld is not None
            and patchst is not None
            and fetchcall is not None
            and patchld.mnemonic == "lw"
            and patchld.rd == INSTRUCTION_MEMORY_WORD_REGISTER
            and patchld.rs1 == MEMORY_BASE_REGISTER
            and (None if patchld.immediate is None else int(patchld.immediate)) == INSTRUCTION_MEMORY_PATCH_OFFSET
            and patchst.mnemonic == "sw"
            and patchst.rs1 == INSTRUCTION_MEMORY_ADDRESS_REGISTER
            and patchst.rs2 == INSTRUCTION_MEMORY_WORD_REGISTER
            and (None if patchst.immediate is None else int(patchst.immediate)) == 0
            and fetchcall.mnemonic == "jalr"
            and fetchcall.rd == 1
            and fetchcall.rs1 == INSTRUCTION_MEMORY_ADDRESS_REGISTER
            and (0 if fetchcall.immediate is None else int(fetchcall.immediate)) == 0
            and fetchret.rs1 == 1
            and (None if target.immediate is None else int(target.immediate)) == 1
            and region.data[
                INSTRUCTION_MEMORY_PATCH_OFFSET : INSTRUCTION_MEMORY_PATCH_OFFSET + 4
            ]
            == patch_word.to_bytes(4, "little")
            and tuple(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()) == expected
        )
    if schema.operation == "table-jump":
        jvtw = _instruction_by_id(testcase, "jvtw")
        jvtcsr = _instruction_by_id(testcase, "jvtcsr")
        jvtfence = _instruction_by_id(testcase, "jvtfence")
        retaddrhi = _instruction_by_id(testcase, "retaddrhi")
        retaddrlo = _instruction_by_id(testcase, "retaddrlo")
        raobs = _instruction_by_id(testcase, "raobs")
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
            "little",
        )
        index = int(decode_form(form, word).get("c_index", -1))
        xlen = 32 if testcase.isa_profile.startswith("rv32") else 64
        width = 4 if xlen == 32 else 8
        table_offset = index * width
        expected_prefix = (
            "imaddrhi",
            "retaddrhi",
            "retaddrlo",
            "imaddrlo",
            "jvtw",
            "jvtcsr",
            "jvtfence",
            "i0",
            "fetch0",
            "fetchobs",
            "fetchret",
            "raobs",
            "skip",
        )
        executed = tuple(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ())
        if (
            jvtw is None
            or jvtcsr is None
            or jvtfence is None
            or retaddrhi is None
            or retaddrlo is None
            or raobs is None
            or not 0 <= index < 256
            or table_offset + width > len(region.data)
            or jvtw.mnemonic != store_mnemonic
            or jvtw.rs1 != MEMORY_BASE_REGISTER
            or jvtw.rs2 != INSTRUCTION_MEMORY_ADDRESS_REGISTER
            or (None if jvtw.immediate is None else int(jvtw.immediate)) != table_offset
            or jvtcsr.mnemonic != "csrrw"
            or jvtcsr.rd != 0
            or jvtcsr.rs1 != MEMORY_BASE_REGISTER
            or jvtfence.mnemonic != "fence.i"
            or jvtfence.rd != 0
            or jvtfence.rs1 != 0
            or _observer_immediate(jvtfence) != 0
            or raobs.mnemonic != store_mnemonic
            or raobs.rs1 != MEMORY_BASE_REGISTER
            or raobs.rs2 != 1
            or (None if raobs.immediate is None else int(raobs.immediate)) != INSTRUCTION_MEMORY_LINK_OFFSET
            or (None if target.immediate is None else int(target.immediate)) != 42
            or skip.rd != 0
            or _observer_immediate(skip) != (
                int(testcase.exit_checkpoint_address)
                - int(testcase.code_address)
                - int(skip.byte_offset)
            )
            or fetchret.rs1 != INSTRUCTION_MEMORY_RETURN_REGISTER
            or _pc_relative_target(
                testcase,
                "retaddrhi",
                "retaddrlo",
                INSTRUCTION_MEMORY_RETURN_REGISTER,
            )
            != int(raobs.byte_offset)
            or int(testcase.initial_gpr[1])
            != (INSTRUCTION_MEMORY_LINK_POISON & xlen_mask(xlen))
            or region.data[table_offset : table_offset + width] != bytes(width)
            or executed[: len(expected_prefix)] != expected_prefix
        ):
            return False
        jvt_form = _emitted_form("csrrw")
        if jvt_form is None:
            return False
        csr_word = int.from_bytes(
            testcase.code_bytes[jvtcsr.byte_offset : jvtcsr.byte_offset + jvtcsr.byte_length],
            "little",
        )
        return int(decode_form(jvt_form, csr_word).get("csr", -1)) == 0x017
    return False


def _paired_state_register(
    testcase: TestCase,
    risk,
    form,
    schema: EffectSchema | None,
    role: str,
) -> int | None:
    if schema is None or not role.endswith("+1"):
        return None
    if schema.kind == "atomic-rmw" and schema.operation == "cas" and role in {"rd+1", "rs2+1"}:
        base = getattr(risk, role[:-2], None)
        if base is not None and int(base) < 31 and (role != "rd+1" or base != 0):
            return int(base) + 1
    if form is None:
        return None
    pair_roles = rv32_xregister_pair_roles(
        form,
        schema,
        xlen=32 if str(testcase.isa_profile).lower().startswith("rv32") else 64,
        register_carrier="gpr",
    )
    base_role = role[:-2]
    if base_role not in pair_roles:
        return None
    base = getattr(risk, base_role, None)
    if base is None or int(base) >= 31 or int(base) & 1:
        return None
    return int(base) + 1


def _actual_register_relation(testcase: TestCase) -> str:
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    if risk.rd is None:
        return "no-destination"
    if risk.rd == 0 and risk.rd_domain in {None, "gpr"}:
        return "rd-eq-x0"
    for role in ("rs1", "rs2"):
        register = getattr(risk, role)
        domain = getattr(risk, f"{role}_domain")
        if (
            risk.rd == register
            and risk.rd_domain is not None
            and domain is not None
            and risk.rd_domain == domain
        ):
            return f"rd-eq-{role}"
    return "distinct"


def _observer_immediate(item) -> int | None:
    return None if item.immediate is None else int(item.immediate)


def _vector_scalar_observer_realized(testcase: TestCase, risk, schema) -> bool:
    if risk.rd in (None, 0):
        return False
    observers = [item for item in testcase.instruction_meta if "observable" in item.tags]
    if not observers:
        return False
    if schema.writeback == "gpr":
        return any(
            item.rs1 == MEMORY_BASE_REGISTER and item.rs2 == risk.rd
            and _observer_immediate(item) == 0 for item in observers
        )
    helper = next((item for item in testcase.instruction_meta if item.instruction_id == "vector-helper"), None)
    form = _emitted_form(helper.mnemonic) if helper is not None else None
    word = int.from_bytes(
        testcase.code_bytes[helper.byte_offset : helper.byte_offset + helper.byte_length], "little"
    ) if helper is not None else 0
    sew = (
        {0: 8, 1: 16, 2: 32, 3: 64, 4: 128, 5: 256, 6: 512, 7: 1024}.get(
            (int(decode_form(form, word).get("zimm11", 0)) >> 3) & 7
        ) if form is not None else None
    )
    move = {16: "fmv.x.h", 32: "fmv.x.w", 64: "fmv.x.d"}.get(sew)
    if move is not None and any(
        item.mnemonic == move and item.rs1 == risk.rd and item.rd == CONSUMER_REGISTER
        for item in testcase.instruction_meta
    ):
        return any(
            item.rs1 == MEMORY_BASE_REGISTER and item.rs2 == CONSUMER_REGISTER
            and _observer_immediate(item) == 0 for item in observers
        )
    return any(
        item.mnemonic in {"fsh", "fsw", "fsd"} and item.rs1 == MEMORY_BASE_REGISTER
        and item.rs2 == risk.rd and _observer_immediate(item) == 0 for item in observers
    )


def _actual_gpr_producer_value(testcase: TestCase, producer) -> int | None:
    """Return a construction-only producer projection for static matching.

    The boundary result is a hint for checking a frozen effect edge; it is
    never a Sail/native result or an execution authority.
    """
    width_bytes, signed = _GPR_PRODUCER_LOAD.get(producer.mnemonic, (0, False))
    raw = _seeded_memory_value(testcase, PRODUCER_LOAD_OFFSET, width_bytes) if width_bytes else None
    if raw is not None:
        width_bits = width_bytes * 8
        xlen = 32 if testcase.isa_profile.startswith("rv32") else 64
        value = int(raw) & ((1 << width_bits) - 1)
        if signed and width_bits < xlen and value & (1 << (width_bits - 1)):
            value |= ((1 << (xlen - width_bits)) - 1) << width_bits
        return value & ((1 << xlen) - 1)
    form = _emitted_form(producer.mnemonic)
    schema = _schema_for_profile(testcase.isa_profile, form)
    if form is None or schema is None or schema.writeback != "gpr":
        return None
    word = int.from_bytes(
        testcase.code_bytes[producer.byte_offset : producer.byte_offset + producer.byte_length],
        "little",
    )
    if word & form.mask != form.match:
        return None
    decoded = decode_form(form, word)
    source_state = _source_state_from_bytes(testcase, producer, form)
    value = declared_gpr_writeback_value(
        form,
        schema,
        operands={name: int(value) for name, value in decoded.items()},
        source_state=source_state,
        pc=int(testcase.code_address) + int(producer.byte_offset),
    )
    return (
        None
        if value is None
        else _architectural_gpr_writeback_value(testcase, form, schema, int(value))
    )


def _actual_gpr_register_value_before(testcase: TestCase, register: int, before_offset: int) -> int:
    if register == 0:
        return 0
    previous = None
    for item in testcase.instruction_meta:
        if (
            ("producer" in item.tags or "risk-sequence" in item.tags)
            and item.rd == register
            and getattr(item, "rd_domain", None) == "gpr"
            and int(item.byte_offset) < int(before_offset)
        ):
            previous = item
    if previous is not None:
        produced = _actual_gpr_producer_value(testcase, previous)
        if produced is not None:
            return int(produced)
    return int(testcase.initial_gpr[int(register)])


def _actual_gpr_source_value(testcase: TestCase, risk, role: str) -> int | None:
    register = getattr(risk, role, None)
    if register is None:
        return None
    return _actual_gpr_register_value_before(
        testcase,
        int(register),
        int(risk.byte_offset),
    )


def _source_state_from_bytes(
    testcase: TestCase,
    risk,
    form,
) -> dict[str, int] | None:
    """Collect construction-only source-register values from the frozen bytes.

    GPR-domain sources resolve through the same projected register state used
    by the writeback evaluators; the fixed compressed rs1=x2 fallback matches
    the encoding-side implicit stack source.
    """
    source_state: dict[str, int] = {}
    for role in source_roles(form):
        register = getattr(risk, role, None)
        if getattr(risk, f"{role}_domain", None) == "gpr" and register is not None:
            actual = _actual_gpr_source_value(testcase, risk, role)
            if actual is not None:
                source_state[role] = int(actual)
        elif role in implicit_source_roles(form) and role == "rs1":
            source_state[role] = int(testcase.initial_gpr[2])
    return source_state


def _producer_realized_bits(
    testcase: TestCase,
    producer,
    schema,
    role: str,
    expected: int,
) -> bool:
    """Whether the sequence-producer's projected output equals the expected bits."""
    if role.startswith("fs"):
        expected_precision = schema.source_precision or schema.precision or 0
        expected_width = IEEE_FORMAT_BY_PRECISION.get(expected_precision, (0,))[0]
        if not expected_precision:
            image = _actual_fp_producer_image(testcase, producer)
            return image is not None and image[0] == int(expected)
        produced = (
            None
            if producer is None or expected_width <= 0
            else _actual_fp_producer_bits(testcase, producer, expected_precision)
        )
        return produced == (int(expected) & ((1 << expected_width) - 1))
    xlen = 32 if testcase.isa_profile.startswith("rv32") else 64
    produced = None if producer is None else _actual_gpr_producer_value(testcase, producer)
    return produced == (int(expected) & ((1 << xlen) - 1))


def _producer_sequence_realized(
    testcase: TestCase,
    intent: MaterializationIntent,
    risk,
    form,
    schema,
) -> bool:
    role = producer_chain_role(
        form,
        schema,
        intent.source_roles,
    )
    if role is None:
        return False
    producer = next(
        (item for item in testcase.instruction_meta if "sequence-producer" in item.tags),
        None,
    )
    if producer is None or int(producer.byte_offset) >= int(risk.byte_offset):
        return False
    source_slot = "rs" + role[2:] if role.startswith("fs") else role
    source_register = getattr(risk, source_slot, None)
    if source_register is None or (source_register == 0 and not role.startswith("fs")):
        return False
    if producer.rd != source_register or not _producer_realized_bits(
        testcase,
        producer,
        schema,
        role,
        int(intent.source_state_map[role]),
    ):
        return False
    if not role.startswith("fs"):
        width = int((_GPR_PRODUCER_LOAD.get(producer.mnemonic) or (0, False))[0])
        if width > 0:
            if producer.rs1 != MEMORY_BASE_REGISTER or _observer_immediate(producer) != PRODUCER_LOAD_OFFSET:
                return False
            if _seeded_memory_value(
                testcase,
                PRODUCER_LOAD_OFFSET,
                width,
            ) != int(intent.source_state_map[role]) & ((1 << (width * 8)) - 1):
                return False
    if role.startswith("fs"):
        return True
    return int(testcase.initial_gpr[source_register]) == (PRODUCER_SOURCE_POISON & ((1 << 64) - 1))


def _consumer_shape_realized(
    testcase: TestCase,
    shape: str,
    sequence_length: int,
    risk,
    schema,
) -> bool:
    if shape in (None, "store"):
        return True
    sequence = [
        item
        for item in testcase.instruction_meta
        if "risk" in item.tags or "risk-sequence" in item.tags
    ]
    source = risk.rd
    if risk.rd_domain == "fpr" and shape not in {"fp-classify-store", "fp-convert-store"}:
        source = CONSUMER_REGISTER
    if source in (None, 0) or len(sequence) != sequence_length:
        return False
    observers = [item for item in testcase.instruction_meta if "observable" in item.tags]

    def stores(register: int | None, *, base: int = MEMORY_BASE_REGISTER) -> bool:
        return register is not None and any(
            item.mnemonic in {"sw", "sd"}
            and item.rs1 == base
            and item.rs2 == register
            and _observer_immediate(item) == 0
            for item in observers
        )

    if shape in {"compare-store", "unsigned-compare-store"}:
        consumer = sequence[1]
        mnemonic = "slt" if shape == "compare-store" else "sltu"
        operands = (source, 0) if shape == "compare-store" else (0, source)
        return (consumer.mnemonic, consumer.rs1, consumer.rs2) == (
            mnemonic, *operands
        ) and stores(consumer.rd)
    if shape == "address-store":
        mask, address = sequence[1:]
        return (
            mask.mnemonic == "andi"
            and mask.rs1 == source
            and mask.rd is not None
            and int(mask.immediate or -1) == 8
            and address.mnemonic == "add"
            and address.rd == mask.rd
            and address.rs1 == MEMORY_BASE_REGISTER
            and address.rs2 == mask.rd
            and stores(source, base=address.rd)
        )
    if shape in {"fp-classify-store", "fp-convert-store"}:
        if schema is None:
            return False
        prefix = {"fp-classify-store": "fclass", "fp-convert-store": "fcvt.w"}[shape]
        suffix = {11: "h", 24: "s", 53: "d", 113: "q"}.get(schema.precision)
        consumer = sequence[1]
        return (
            suffix is not None
            and consumer.mnemonic == f"{prefix}.{suffix}"
            and consumer.rs1 == source
            and consumer.rs1_domain == "fpr"
            and consumer.rd is not None
            and consumer.rd_domain == "gpr"
            and stores(consumer.rd)
        )
    if shape == "branch-zero-store":
        branch = sequence[1]
        marker = next((item for item in testcase.instruction_meta if item.instruction_id == "markt"), None)
        return (
            marker is not None
            and (branch.mnemonic, branch.rs1, branch.rs2, int(branch.immediate or -1)) == (
                "bne", source, 0, 8
            )
            and marker.mnemonic == "addi"
            and marker.rd is not None
            and marker.rs1 == 0
            and int(marker.immediate or -1) == 2
            and int(testcase.initial_gpr[marker.rd]) == 1
            and stores(marker.rd)
        )
    if shape == "jump-target-store":
        mask, hi, lo, address, jump = sequence[1:]
        by_id = {item.instruction_id: item for item in testcase.instruction_meta}
        target0, target1 = by_id.get("jtt0"), by_id.get("jtt1")
        jtskip = by_id.get("jtskip")
        if (
            tuple(item.instruction_id for item in sequence[1:])
            != ("jtmask", "jthi", "jtlo", "jtadd", "cons")
            or tuple(item.mnemonic for item in sequence[1:])
            != ("andi", "auipc", "addi", "add", "jalr")
            or target0 is None
            or target1 is None
            or jtskip is None
            or target1.byte_offset - target0.byte_offset != 16
            or jtskip.mnemonic != "jal"
            or jtskip.rd != 0
            or _observer_immediate(jtskip) != (
                int(testcase.exit_checkpoint_address)
                - int(testcase.code_address)
                - int(jtskip.byte_offset)
            )
        ):
            return False
        return (
            mask.rs1 == source
            and mask.rd is not None
            and int(mask.immediate or -1) == 16
            and hi.rd is not None
            and lo.rs1 == hi.rd
            and lo.rd == hi.rd
            and address.rs1 == hi.rd
            and address.rs2 == mask.rd
            and address.rd == mask.rd
            and jump.rd == 1
            and jump.rs1 == mask.rd
            and _observer_immediate(jump) == 0
            and by_id.get("jtskip") is not None
            and by_id.get("jtobs0") is not None
            and by_id.get("jtobs1") is not None
            and target0.mnemonic == "addi"
            and target0.rd == INSTRUCTION_MEMORY_WORD_REGISTER
            and target0.rs1 == 0
            and _observer_immediate(target0) == 2
            and target1.mnemonic == "addi"
            and target1.rd == INSTRUCTION_MEMORY_WORD_REGISTER
            and target1.rs1 == 0
            and _observer_immediate(target1) == 3
            and by_id.get("jtobs0").rs2 == target0.rd
            and by_id.get("jtobs1").rs2 == target1.rd
            and stores(target0.rd)
            and stores(target1.rd)
            and _pc_relative_target(testcase, "jthi", "jtlo", int(hi.rd)) == int(target0.byte_offset)
        )
    return False


def _canonical_intent_match(
    form,
    intent: MaterializationIntent,
) -> bool:
    """Require an intent row to come from the catalog-derived boundary owner."""
    owner_schema = generation_schema_for_form(form)
    try:
        realization = SemanticRealization(
            intent.realization_xlen,
            tuple(intent.realization_extension_tokens),
            str(intent.realization_register_carrier),
            str(intent.realization_privilege_class),
            str(intent.realization_isa_profile),
            str(intent.realization_environment_profile),
        )
    except (TypeError, ValueError):
        return False
    candidates = boundary_intents(form, owner_schema, realization)
    for candidate in candidates:
        if candidate == intent:
            return True
    # Explicit legality probes may intentionally perturb a reserved encoding
    # field (for example ``*_n0``) or an extension-disabled codeword that has
    # no normal semantic boundary row.
    # Keep those rows tied to an existing catalog realization/program shape;
    # the concrete violation itself is checked after decoding below.
    if intent.lane == LANE_LEGALITY and str(intent.boundary_class).startswith(
        (
            "legality:",
            "extension-gate:",
            "rv32:compat-extension-gate:",
            "rv32:xlen-gate:",
            "rv64:xlen-gate:",
        )
    ):
        operand_fields = set(operand_groups(form))
        if not all(
            name in operand_fields and isinstance(value, int) and not isinstance(value, bool)
            for name, value in intent.operands_map.items()
        ):
            return False
        for candidate in candidates:
            boundary_prefix = str(intent.boundary_class)
            common_match = candidate == replace(
                intent,
                lane=candidate.lane,
                boundary_class=candidate.boundary_class,
                operands=candidate.operands,
            )
            if boundary_prefix.startswith("legality:"):
                matches = (
                    common_match
                    and candidate.lane == LANE_LEGALITY
                    and candidate.boundary_class == intent.boundary_class
                )
            else:
                matches = common_match and (
                    candidate.lane == LANE_NORMAL
                    or (
                        candidate.lane == LANE_LEGALITY
                        and candidate.boundary_class == intent.boundary_class
                    )
                )
            if matches:
                return True
    return False


def _control_target_realized(testcase: TestCase, risk, schema, expected_ids) -> bool:
    if schema is None or schema.kind not in {"control-compare", "control-jump"}:
        return True
    expected_index = expected_ids.index(risk.instruction_id) if risk.instruction_id in expected_ids else -1
    target_id = expected_ids[expected_index + 1] if expected_index + 1 < len(expected_ids) else None
    target = _instruction_by_id(testcase, target_id) if target_id is not None else None
    target_offset = target.byte_offset if target is not None else sum(
        item.byte_length for item in testcase.instruction_meta
    )
    immediate = int(risk.immediate or 0)
    if schema.kind == "control-compare":
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        source = _actual_gpr_source_value(testcase, risk, "rs1")
        compare = _actual_gpr_source_value(testcase, risk, "rs2") if risk.rs2 is not None else 0
        taken = (
            branch_condition_holds(source, compare, schema.operation, xlen=xlen)
            if source is not None else None
        )
        if taken is None:
            return False
        actual_target = int(risk.byte_offset) + (immediate if taken else risk.byte_length)
        target_offset = target.byte_offset if target is not None else sum(
            item.byte_length for item in testcase.instruction_meta
        )
        valid_target = target_offset == sum(
            item.byte_length for item in testcase.instruction_meta
        ) or any(item.byte_offset == target_offset for item in testcase.instruction_meta)
    elif schema.operation in {"j", "jal"}:
        actual_target = int(risk.byte_offset) + immediate
        valid_target = True
    elif risk.rs1 is None:
        return False
    else:
        actual_target = (int(testcase.initial_gpr[risk.rs1]) + immediate) & ~1
        valid_target = True
    return valid_target and actual_target == target_offset


def _obligation_matches_testcase(
    testcase: TestCase,
    intent: MaterializationIntent,
) -> bool:
    if intent.normative_obligation is None:
        return False
    risk = next((item for item in testcase.instruction_meta if "risk" in item.tags), None)
    if risk is None or _declared_lane(testcase) != intent.lane:
        return False
    form = _emitted_form(intent.form)
    if form is None:
        return False
    risk_form = _emitted_form(risk.mnemonic)
    if risk_form is None or risk_form.mnemonic != form.mnemonic:
        return False
    if not _canonical_intent_match(
        form,
        intent,
    ):
        return False
    schema = (
        None
        if intent.lane == LANE_LEGALITY
        else _compatibility_writeback_schema(
            testcase.isa_profile,
            risk_form,
            _schema_for_profile(testcase.isa_profile, risk_form),
        )
    )
    realization_data = generation_realization_for_testcase(testcase)
    if realization_data is None or not isinstance(
        testcase.dataflow_meta.get("normative_obligation"), dict
    ):
        return False
    expected_ids = testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()
    expected_facts = {
        "disposition": disposition_for_schema(
            schema, intent.lane,
            has_sink=any(
                "observable" in item.tags and item.instruction_id in expected_ids
                for item in testcase.instruction_meta
            ),
        ),
        "lane": str(intent.lane),
        "state_domain": "trap" if schema is None else str(schema.state_domain),
        "effect_kind": "legality-only" if schema is None else str(schema.kind),
    }
    facts = testcase.dataflow_meta.get("realized_facts")
    if not isinstance(facts, dict) or any(facts.get(name) != value for name, value in expected_facts.items()):
        return False
    # A shape-valid obligation id is not enough.  Rebuild the exact Manual/Sail
    # obligation for the *concrete* realization: materialization may add a
    # required observer extension (for example Zicsr for fflags), so comparing
    # against the pre-materialization intent object would reject legal rows.
    try:
        realization = (
            SemanticRealization.from_dict(realization_data)
            if realization_data is not None else None
        )
        obligation_schema = schema_for_realization(
            risk_form,
            generation_schema_for_form(risk_form),
            xlen=int(realization.xlen),
            register_carrier=str(realization.register_carrier),
        ) if realization is not None else None
        expected_obligation = (
            normative_obligation_for(
                risk_form,
                obligation_schema,
                realization,
                lane=intent.lane,
                boundary_class=str(intent.boundary_class),
            ).to_dict()
            if realization is not None
            else None
        )
    except (KeyError, TypeError, ValueError):
        return False
    if testcase.dataflow_meta.get("normative_obligation") != expected_obligation:
        return False
    boundary = str(intent.boundary_class)
    risk_contracts = testcase.dataflow_meta.get("risk_contracts")
    if (
        not isinstance(risk_contracts, (list, tuple))
        or len(risk_contracts) != 1
        or risk_contracts[0].get("boundary_class") != boundary
    ):
        return False
    for key in ("memory_boundary", "control_boundary", "vector_boundary"):
        required = (
            key == "memory_boundary" and intent.lane != LANE_LEGALITY and schema is not None and schema.kind in {
                "memory-load", "memory-store", "atomic-rmw", "vector-memory",
            }
            or key == "control_boundary" and intent.lane != LANE_LEGALITY and schema is not None and schema.kind.startswith("control")
            or key == "vector_boundary" and schema is not None and schema.state_domain in {
                "vector", "vector-memory",
            }
        )
        if required and testcase.dataflow_meta.get(key) != boundary:
            return False
    if (
        schema is not None
        and intent.lane != LANE_LEGALITY
        and schema.kind in {"memory-load", "memory-store", "atomic-rmw"}
    ):
        expected_disposition = (
            "environment-static-only"
            if boundary.startswith("memory-environment:")
            else "value-observer-pending"
        )
        if testcase.dataflow_meta.get("memory_environment_disposition") != expected_disposition:
            return False
    if (
        schema is not None
        and schema.kind == "may-be-operation"
        and schema.operation.startswith("vector-config")
        and intent.lane != LANE_LEGALITY
        and not boundary.startswith(("vector-config:", "register-domain:"))
    ):
        return False
    if (
        schema is not None
        and boundary.startswith("vector-mask:")
        and vector_mask_operand(risk_form, schema) is None
    ):
        return False
    if (
        schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and intent.lane != LANE_LEGALITY
    ):
        required_seed_registers = {
            int(value)
            for name, value in intent.operands
            if name in {"vd", "vs1", "vs2", "vs3"}
        }
        seeded_registers = {
            dict(item.operand_fields).get("vd")
            for item in testcase.instruction_meta
            if item.instruction_id.startswith("vseed")
        }
        if not required_seed_registers <= seeded_registers:
            return False
        is_vector_config = (
            schema.kind == "may-be-operation"
            and schema.operation.startswith("vector-config")
        )
        state_axis = (
            "vector-vl:", "vector-vtype:", "vector-vstart:",
            "vector-mask:", "vector-tail:", "vector-ff:", "vector-frm:",
        )
        if not is_vector_config and boundary.startswith(state_axis):
            if testcase.dataflow_meta.get("vector_boundary") != boundary:
                return False
            state = vector_state_boundary(boundary)
            if state is None and boundary not in {
                "vector-ff:element1-trim", "vector-frm:reserved",
            }:
                return False
            if state is not None and state.axis == "vstart" and int(state.vstart or 0) > 0 and not any(
                item.instruction_id == "vstart-set"
                and item.mnemonic == "csrrw"
                and dict(item.operand_fields).get("csr") == 0x008
                for item in testcase.instruction_meta
            ):
                return False
        if is_vector_config:
            base = MEMORY_BASE_REGISTER
            config_rows = (
                ("vrl", "vlobs", 0xC20, 8),
                ("vrt", "vtypeobs", 0xC21, 16),
                ("vrs", "vstartobs", 0x008, 24),
                ("vrb", "vlenbobs", 0xC22, 32),
            )
            for read_id, store_id, csr, offset in config_rows:
                read = _instruction_by_id(testcase, read_id)
                store = _instruction_by_id(testcase, store_id)
                read_form = _emitted_form(read.mnemonic) if read is not None else None
                read_word = (
                    int.from_bytes(
                        testcase.code_bytes[read.byte_offset:read.byte_offset + read.byte_length],
                        "little",
                    )
                    if read is not None else 0
                )
                if (
                    read is None or store is None or read.mnemonic != "csrrs"
                    or read.rd in (None, 0) or read.rs1 != 0
                    or read_form is None
                    or decode_form(read_form, read_word).get("csr") != csr
                    or store.mnemonic != _observer_store_mnemonic(testcase)
                    or store.rs1 != base or store.rs2 != read.rd
                    or _observer_immediate(store) != offset
                ):
                    return False
            if risk.rd not in (None, 0):
                result_store = _instruction_by_id(testcase, "vrdobs")
                if (
                    result_store is None
                    or result_store.mnemonic != _observer_store_mnemonic(testcase)
                    or result_store.rs1 != base or result_store.rs2 != risk.rd
                    or _observer_immediate(result_store) != 0
                ):
                    return False
        else:
            avl = next((item for item in testcase.instruction_meta if item.instruction_id == "vavl"), None)
            expected_avl = 0 if boundary in {"vector-vl:zero", "vector-vl:vlmax"} else None
            if (
                avl is None or avl.mnemonic != "addi" or avl.rs1 != 0
                or type(avl.immediate) is not int
                or (expected_avl == 0 and avl.immediate != 0)
                or (expected_avl is None and avl.immediate <= 0)
            ):
                return False
            vtype = next((item for item in testcase.instruction_meta if item.instruction_id == "vtype"), None)
            vtypeobs = next((item for item in testcase.instruction_meta if item.instruction_id == "vtypeobs"), None)
            if vtype is None or vtype.mnemonic != "csrrs" or vtypeobs is None:
                return False
            vtype_form = _emitted_form(vtype.mnemonic)
            vtype_word = int.from_bytes(
                testcase.code_bytes[vtype.byte_offset:vtype.byte_offset + vtype.byte_length], "little"
            )
            if vtype_form is None or decode_form(vtype_form, vtype_word).get("csr") != 0xC21:
                return False
            if vtypeobs.mnemonic != _observer_store_mnemonic(testcase) or _observer_immediate(vtypeobs) != 24:
                return False
            helper = _instruction_by_id(testcase, "vector-helper")
            if helper is None:
                return False
            helper_form = _emitted_form(helper.mnemonic)
            helper_word = int.from_bytes(
                testcase.code_bytes[helper.byte_offset:helper.byte_offset + helper.byte_length], "little"
            )
            helper_fields = decode_form(helper_form, helper_word) if helper_form is not None else {}
            vtype_field = "zimm11" if helper.mnemonic == "vsetvli" else "zimm10"
            if helper.mnemonic in {"vsetvli", "vsetivli"}:
                helper_vtype = int(helper_fields.get(vtype_field, 0))
                reserved_mask = 0x600 if helper.mnemonic == "vsetvli" else 0x200
                helper_sew = 8 << ((helper_vtype >> 3) & 7)
                if helper_vtype & reserved_mask or (
                    helper_vtype & 0x100
                    and (
                        "zvfbfa" not in enabled_extensions(testcase.isa_profile)
                        or helper_sew >= 32
                    )
                ):
                    return False
            if boundary == "vector-vl:zero" and (
                helper_fields.get("rs1") != avl.rd or helper_fields.get("rd") != 0
            ):
                return False
            if boundary == "vector-vl:vlmax" and (
                helper_fields.get("rs1") != 0 or helper_fields.get("rd") != avl.rd
            ):
                return False
    if (
        schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and intent.lane == LANE_LEGALITY
        and boundary == "vector-frm:reserved"
    ):
        frm = _instruction_by_id(testcase, "v-frm-reserved")
        frm_form = _emitted_form(frm.mnemonic) if frm is not None else None
        frm_word = (
            int.from_bytes(
                testcase.code_bytes[frm.byte_offset:frm.byte_offset + frm.byte_length],
                "little",
            )
            if frm is not None else 0
        )
        frm_fields = decode_form(frm_form, frm_word) if frm_form is not None else {}
        if (
            frm is None or frm.mnemonic != "csrrwi" or frm.byte_offset >= risk.byte_offset
            or frm_form is None or frm_fields.get("csr") != 0x002
            or frm_fields.get("zimm5") != 5
        ):
            return False
        if schema.writeback == "vector":
            result_observer = _instruction_by_id(testcase, "vobs")
            if (
                result_observer is None
                or dict(result_observer.operand_fields).get("vs3")
                != intent.operands_map.get("vd")
            ):
                return False
    seed_precision = (
        schema.source_precision or schema.precision
        if schema is not None else 0
    )
    if seed_precision in _FP_SEED_LOAD and risk.rs1_domain == "fpr":
        seed_loads = _FP_NAN_BOX_SEED_LOAD if boundary.startswith("fp:nan-box:") else _FP_SEED_LOAD
        for index, role in enumerate(intent.source_roles):
            if not role.startswith("fs"):
                continue
            seed = next((item for item in testcase.instruction_meta if item.instruction_id == f"seed{index}"), None)
            source = getattr(risk, f"rs{role[2:]}", None)
            if seed is None and intent.program_shape == "producer-risk":
                continue
            if (
                seed is None
                or seed.mnemonic != seed_loads[seed_precision]
                or seed.rd != source
                or seed.rs1 != MEMORY_BASE_REGISTER
                or _observer_immediate(seed) != 16 * (index + 1)
            ):
                return False
    word = int.from_bytes(
        testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
        "little",
    )
    if word & form.mask != form.match and intent.lane != LANE_LEGALITY:
        return False
    decoded = decode_form(form, word)
    operands = intent.operands_map
    control_bound = schema is not None and schema.kind in {"control-compare", "control-jump"}
    if control_bound and intent.lane != LANE_LEGALITY:
        if not _control_target_realized(testcase, risk, schema, expected_ids):
            return False
    if intent.consumer_shape == "jump-target-store":
        selected = "1" if "jtt1" in expected_ids else "0"
        expected_jump_path = (
            "jtmask", "jthi", "jtlo", "jtadd", "cons",
            f"jtt{selected}", f"jtobs{selected}", f"jtret{selected}", "jtskip",
        )
        try:
            jump_start = expected_ids.index("jtmask")
        except ValueError:
            return False
        if tuple(expected_ids[jump_start:]) != expected_jump_path:
            return False
    if any(
        decoded.get(name) != value
        for name, value in operands.items()
        if not (control_bound and operand_kind(form, name) == "imm")
    ):
        return False
    # A legality/trap row has no architectural value result to reconstruct.
    # Keep the structural encoding and extension-gate checks above, but do not
    # apply normal-lane source, destination, producer, or sequence contracts
    # to compressed reserved fields (for example ``c_rs2_n0``).
    gate_disabled = intent.extension_gate_disabled or str(
        intent.boundary_class
    ).startswith("rv32:compat-extension-gate:")
    if intent.lane == LANE_LEGALITY:
        expected_contract = _intent_materialization_contract(intent)
        if (
            expected_contract is None
            or testcase.isa_profile != expected_contract[0]
            or testcase.dataflow_meta.get("generation_realization")
            != expected_contract[1]
            or testcase.dataflow_meta.get("legality_boundary_class")
            != expected_contract[2]
            or testcase.code_bytes != expected_contract[4]
        ):
            return False
        violations = actual_violation_set(testcase)
        if violations != expected_contract[3]:
            return False
        if gate_disabled and violations != frozenset({"extension-disabled"}):
            return False
        return True
    expected_contract = _intent_materialization_contract(intent)
    if (
        expected_contract is None
        or testcase.isa_profile != expected_contract[0]
        or testcase.dataflow_meta.get("generation_realization") != expected_contract[1]
        or testcase.code_bytes != expected_contract[4]
    ):
        return False
    if operands.get("rm") == 7:
        frm_writes = [
            item
            for item in testcase.instruction_meta
            if "frm-write" in item.tags and item.immediate == intent.rounding_mode
        ]
        if len(frm_writes) != 1:
            return False
    if (schema is not None and schema.kind == "instruction-memory"
            and intent.lane == LANE_NORMAL
            and not _instruction_memory_realized(testcase, form, schema, risk)):
        return False
    if schema is not None and schema.kind == "stack-transfer" and intent.lane == LANE_NORMAL:
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        registers, stack_adj = stack_shape(intent.operands_map, xlen=xlen)
        width_bytes = xlen // 8
        if schema.operation != "push" and any(
            (observer := _instruction_by_id(testcase, f"obs{index}")) is None
            or observer.mnemonic != _observer_store_mnemonic(testcase)
            or observer.rs1 != MEMORY_BASE_REGISTER
            or observer.rs2 != register
            or _observer_immediate(observer) != index * width_bytes
            for index, register in enumerate(registers)
        ):
            return False
        offsets = stack_slot_offsets(registers, stack_adj, xlen=xlen)
        region = next(
            (item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION),
            None,
        )
        if region is None:
            return False
        frame_base = int(testcase.initial_gpr[2]) - int(region.address)
        stack_base = frame_base - stack_adj if schema.operation == "push" else frame_base
        if any(
            stack_base + offsets[register] < 0
            or stack_base + offsets[register] + width_bytes > len(region.data)
            for register in registers
        ):
            return False
        return_target = None
        if schema.operation in {"popret", "popretz"}:
            executed = tuple(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ())
            if len(executed) < 2:
                return False
            target = next(
                (item for item in testcase.instruction_meta if item.instruction_id == executed[1]),
                None,
            )
            if target is None or int(target.byte_offset) <= int(risk.byte_offset) + int(risk.byte_length):
                return False
            return_target = int(target.byte_offset)
            if schema.operation == "popretz":
                if int(testcase.initial_gpr[10]) == 0:
                    return False
                if not any("observable" in item.tags and item.rs2 == 10 for item in testcase.instruction_meta):
                    return False
        if schema.operation != "push":
            for index, register in enumerate(registers):
                relative = frame_base + offsets[register]
                expected = (
                    return_target
                    if register == 1 and return_target is not None
                    else stack_seed_value(index, register) & ((1 << xlen) - 1)
                )
                if _seeded_memory_value(testcase, relative, width_bytes) != int(expected):
                    return False
    if schema is not None and schema.kind == "paired-register-transfer" and intent.lane == LANE_NORMAL:
        observers = [item for item in testcase.instruction_meta if "observable" in item.tags]
        if len(observers) != 2:
            return False
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        expected = (
            (10, 11)
            if schema.operation == "move-s-to-a01"
            else (
                decode_register_field_value("c_sreg1", int(decoded.get("c_sreg1", -1))),
                decode_register_field_value("c_sreg2", int(decoded.get("c_sreg2", -1))),
            )
        )
        if tuple(item.rs2 for item in observers) != expected:
            return False
        if tuple((-1 if item.immediate is None else int(item.immediate)) for item in observers) != (0, xlen // 8):
            return False
    actual_relation = _actual_register_relation(testcase)
    structural_alias = any(
        "rd" in group and "rs1" in group for group in operand_groups(form)
    )
    if (
        risk.rd is not None
        and (schema is None or schema.writeback != "none")
        and actual_relation != intent.register_relation
        and not (
            intent.register_relation == "distinct"
            and actual_relation == "rd-eq-rs1"
            and structural_alias
        )
    ):
        return False
    if intent.destination_seed is not None:
        destination = risk.rd
        if (
            destination in (None, 0)
            or int(testcase.initial_gpr[destination])
            != int(intent.destination_seed) & ((1 << 64) - 1)
            or any(
                "sequence-producer" in item.tags and item.rd == destination
                for item in testcase.instruction_meta
                if item.byte_offset < risk.byte_offset
            )
        ):
            return False
    if (
        intent.boundary_class == "address-destination-alias"
        or (
            intent.register_relation == "rd-eq-rs1"
            and schema is not None
            and schema.kind in {"memory-load", "atomic-rmw"}
        )
    ):
        if (
            schema is None
            or schema.kind not in {"memory-load", "atomic-rmw"}
            or risk.rd in (None, 0)
            or risk.rd != risk.rs1
        ):
            return False
        width_bytes = max((schema.memory_width or schema.width) // 8, 1)
        region = next(
            (item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION),
            None,
        )
        address = _actual_gpr_source_value(testcase, risk, "rs1")
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        effective = None if address is None else (
            int(address) + int(risk.immediate or 0)
        ) & xlen_mask(xlen)
        if "memory" in intent.source_state_map and effective != LOGICAL_DATA_ADDRESS + MEMORY_ACCESS_OFFSET:
            return False
        if region is None or address is None or _seeded_memory_value(
            testcase,
            int(effective) - int(region.address) if effective is not None else 0,
            width_bytes,
        ) is None:
            return False
    actual_sequence = [
        item
        for item in testcase.instruction_meta
        if "risk" in item.tags or "risk-sequence" in item.tags
    ]
    if intent.program_shape == "producer-risk":
        if len(actual_sequence) != 1 or not _producer_sequence_realized(testcase, intent, risk, form, schema):
            return False
    elif intent.sequence_length != len(actual_sequence):
        return False
    for role, value in intent.source_state:
        if intent.program_shape == "producer-risk" and role == producer_chain_role(
            form,
            schema,
            intent.source_roles,
        ):
            producer = next(
                (item for item in testcase.instruction_meta if "sequence-producer" in item.tags),
                None,
            )
            if producer is None or not _producer_realized_bits(
                testcase, producer, schema, role, int(value)
            ):
                return False
            continue
        if role == "memory":
            width = max(((schema.memory_width if schema is not None else 64) // 8), 1)
            address = _actual_gpr_source_value(testcase, risk, "rs1")
            xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
            if address is None or (
                int(address) + int(risk.immediate or 0)
            ) & xlen_mask(xlen) != LOGICAL_DATA_ADDRESS + MEMORY_ACCESS_OFFSET:
                return False
            if _seeded_memory_value(testcase, 64, width) != int(value) & ((1 << (width * 8)) - 1):
                return False
            continue
        if role == "address-offset":
            register = risk.rs1
            immediate = int(risk.immediate or 0)
            xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
            if register is None or (
                int(testcase.initial_gpr[register]) + immediate
            ) & xlen_mask(xlen) != (64 + int(value)) & xlen_mask(xlen):
                return False
            continue
        if role == "vector-fault-element":
            if not vector_fault_only_first(form, schema) or risk.rs1 is None:
                return False
            region = next(
                (item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION),
                None,
            )
            width = max(int(schema.memory_width or schema.width) // 8, 1)
            expected = (
                int(region.address) + len(region.data) - width * int(value)
                if region is not None
                else None
            )
            xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
            if expected is None or (
                int(testcase.initial_gpr[risk.rs1]) + int(risk.immediate or 0)
            ) & xlen_mask(xlen) != int(expected) & xlen_mask(xlen):
                return False
            continue
        if role.startswith("fs"):
            index = int(role[2:]) - 1
            source_precision = schema.source_precision if schema is not None else 0
            precision = source_precision or (schema.precision if schema is not None else 0)
            width = IEEE_FORMAT_BY_PRECISION.get(precision, (0,))[0] // 8
            if _seeded_memory_value(testcase, 16 * (index + 1), width) != int(value) & ((1 << (width * 8)) - 1):
                return False
            continue
        if control_bound and role in {"rs1", "rs2"}:
            actual = _actual_gpr_source_value(testcase, risk, role)
            xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
            if actual is None or int(actual) != int(value) & xlen_mask(xlen):
                return False
            continue
        register = getattr(risk, role, None) if hasattr(risk, role) else None
        if register is None:
            register = _paired_state_register(testcase, risk, form, schema, role)
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        mask = (1 << xlen) - 1
        if (
            register is None
            or not 0 <= int(testcase.initial_gpr[register]) <= mask
            or int(testcase.initial_gpr[register]) != int(value) & mask
        ):
            return False
    if intent.sibling_state:
        from .materialize import _PROJECTION_SIBLING_RESERVED

        used = set(_PROJECTION_SIBLING_RESERVED)
        for item in testcase.instruction_meta:
            used.update(
                int(register)
                for register in (item.rd, item.rs1, item.rs2, item.rs3)
                if register is not None
            )
        sibling_register = next(
            register for register in range(32) if register not in used
        )
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        if any(
            int(testcase.initial_gpr[sibling_register]) != int(value) & ((1 << xlen) - 1)
            for _role, value in intent.sibling_state
        ):
            return False
    result_store = next(
        (
            item
            for item in testcase.instruction_meta
            if item.instruction_id == "obs"
            and item.instruction_id in expected_ids
            and "observable" in item.tags
        ),
        None,
    )
    if (
        intent.consumer_shape == "store"
        and result_store is not None
        and schema is not None
    ):
        if schema.writeback == "gpr" and result_store.rs2 != risk.rd:
            return False
        if schema.writeback == "fpr" and not (
            result_store.rs2 == risk.rd
            and result_store.rs2_domain == "fpr"
            or any(
                item.instruction_id in expected_ids
                and item.mnemonic in {"fmv.x.h", "fmv.x.w", "fmv.x.d"}
                and item.rs1 == risk.rd
                and item.rs1_domain == "fpr"
                and item.rd == result_store.rs2
                and item.rd_domain == "gpr"
                for item in testcase.instruction_meta
            )
        ):
            return False
    if (
        intent.consumer_shape == "store"
        and schema is not None
        and schema.kind == "atomic-rmw"
        and schema.writeback == "paired-gpr"
    ):
        pair_stores = [
            item
            for item in testcase.instruction_meta
            if item.instruction_id in expected_ids
            and item.instruction_id != risk.instruction_id
            and "observable" in item.tags
        ]
        xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
        expected_pair = {
            (0, risk.rd), (xlen // 8, int(risk.rd) + 1),
        }
        actual_pair = {
            (_observer_immediate(item), item.rs2) for item in pair_stores
        }
        if len(pair_stores) != 2 or actual_pair != expected_pair:
            if not (
                intent.register_relation == "rd-eq-x0"
                and risk.rd == 0
                and risk.instruction_id in expected_ids
                and "observable" in risk.tags
                and not pair_stores
            ):
                return False
    return intent.consumer_shape == "store" or _consumer_shape_realized(
        testcase, intent.consumer_shape, intent.sequence_length, risk, schema
    )


def _actual_atomic_reservation_state(testcase: TestCase, schema) -> str | None:
    if schema is None or schema.kind != "atomic-rmw" or schema.operation != "sc":
        return None
    sequence_items = []
    for item in testcase.instruction_meta:
        if "risk" in item.tags:
            sequence_items.append(item)
            break
        if "risk-sequence" in item.tags:
            sequence_items.append(item)
    sequence = tuple(sequence_items)
    if len(sequence) == 1:
        return "unpaired"
    first_form = _emitted_form(sequence[0].mnemonic)
    first_schema = _schema_for_profile(testcase.isa_profile, first_form)
    if first_schema is None or first_schema.operation != "lr":
        return "other-reservation-shape"
    if len(sequence) == 2 and sequence[0].rs1 == sequence[1].rs1:
        return "paired-same-address"
    if (
        len(sequence) == 3
        and sequence[1].mnemonic == "addi"
        and sequence[1].rd == sequence[-1].rs1
        and sequence[1].rs1 == sequence[0].rs1
        and int(sequence[1].immediate or 0) != 0
    ):
        return "address-mismatch"
    second_form = _emitted_form(sequence[1].mnemonic) if len(sequence) > 2 else None
    second_schema = _schema_for_profile(testcase.isa_profile, second_form)
    if len(sequence) == 3 and second_schema is not None and second_schema.operation == "sc":
        return "intervening-sc"
    return "other-reservation-shape"


def _actual_operation(
    testcase: TestCase,
    risk,
    form,
    schema,
    *,
    reservation_state: str | None = None,
) -> str:
    operation = schema.operation if schema is not None else "legality-only"
    if operation == "sc":
        state = reservation_state if reservation_state is not None else _actual_atomic_reservation_state(testcase, schema)
        return f"sc:{state}" if state is not None else operation
    if operation == "table-jump" and form is not None:
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
            "little",
        )
        index = int(decode_form(form, word).get("c_index", -1))
        return "table-jump-link" if index >= 32 else "table-jump-no-link"
    return operation


def _actual_boundary_class(testcase: TestCase, form, schema) -> str:
    """Describe the concrete boundary without consulting generation metadata."""
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    word = int.from_bytes(
        testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length], "little"
    )
    if form is None:
        return "encoding:outside-form"
    if _declared_lane(testcase) == LANE_LEGALITY:
        violations = sorted(actual_violation_set(testcase))
        if len(violations) == 1:
            return f"legality:{violations[0]}"
        return "legality:concrete-trap-contract"
    if word & form.mask != form.match:
        return "encoding:outside-form"
    operands = decode_form(form, word)
    if compressed_codeword_disposition(form, schema, operands) == "hint":
        return f"compressed-hint:{form.mnemonic}"
    if schema is not None and schema.kind in {
        "encoding-only",
        "privileged-system",
        "vector-data",
        "vector-memory",
    }:
        # Encoding-only and vector/system state paths either lack a scalar
        # result formula or own state fields, so every descriptor field is part
        # of the concrete boundary identity.  Do not collapse vector
        # register/mask selectors into one generic point.
        boundary_operands = [
            f"{name}={value}"
            for name, value in sorted(operands.items())
            if name in operand_groups(form)
        ]
    else:
        boundary_operands = [
            f"{name}={value}"
            for name, value in sorted(operands.items())
            if name in operand_groups(form)
            and (
                "imm" in name
                or (schema is not None and schema.immediate_field == name)
                or name
                in {
                    "aq",
                    "bs",
                    "fm",
                    "pred",
                    "rl",
                    "rm",
                    "rnum",
                    "shamt",
                    "shamtd",
                    "succ",
                    "c_sreg1",
                    "c_sreg2",
                    "zimm5",
                    "c_rlist",
                    "c_spimm",
                }
            )
        ]
    operation = _actual_operation(testcase, risk, form, schema)
    return f"{operation}:{','.join(boundary_operands)}" if boundary_operands else operation


def _actual_fp_producer_image(testcase: TestCase, producer) -> tuple[int, int] | None:
    storage_width = 128 if profile_enables_form(testcase.isa_profile, _emitted_form("flq")) else 64
    if producer.mnemonic in _FP_LOAD_PRECISION and producer.rs1 == MEMORY_BASE_REGISTER:
        region = next((item for item in testcase.initial_memory_regions if item.region_id == OBSERVABLE_REGION), None)
        if region is None:
            return None
        producer_precision = _FP_LOAD_PRECISION[producer.mnemonic]
        producer_width = IEEE_FORMAT_BY_PRECISION[producer_precision][0]
        producer_bytes = producer_width // 8
        offset = int(producer.immediate or 0)
        if offset < 0 or offset + producer_bytes > len(region.data):
            return None
        raw = int.from_bytes(region.data[offset : offset + producer_bytes], "little")
    elif producer.mnemonic in _FP_MOVE_SOURCE_WIDTH and producer.rs1 is not None:
        producer_width = _FP_MOVE_SOURCE_WIDTH[producer.mnemonic]
        raw = _actual_gpr_register_value_before(
            testcase,
            int(producer.rs1),
            int(producer.byte_offset),
        ) & ((1 << producer_width) - 1)
    else:
        return None
    if producer_width > storage_width:
        return None
    if producer_width < storage_width:
        raw |= ((1 << (storage_width - producer_width)) - 1) << producer_width
    return raw, storage_width


def _actual_fp_producer_bits(testcase: TestCase, producer, precision: int) -> int | None:
    expected_width = IEEE_FORMAT_BY_PRECISION[precision][0]
    image = _actual_fp_producer_image(testcase, producer)
    if image is None:
        return None
    raw, storage_width = image
    upper_width = storage_width - expected_width
    if upper_width and raw >> expected_width != (1 << upper_width) - 1:
        return canonical_nan_bits(precision)
    return raw & ((1 << expected_width) - 1)


def _architectural_gpr_writeback_value(testcase: TestCase, form, schema, value: int) -> int:
    """Convert an operation-width result into the architectural GPR value."""
    width = int(
        (schema.memory_width or schema.width)
        if schema.kind in {"memory-load", "atomic-rmw"}
        else schema.width
    )
    sign_extend = False
    if schema.kind == "memory-load":
        sign_extend = schema.operation != "load-unsigned"
    elif schema.kind == "atomic-rmw":
        sign_extend = schema.operation != "sc"
    elif schema.kind == "fp-convert":
        sign_extend = schema.operation in {"fp-to-int", "fp-to-int-mod"}
    elif (
        getattr(form, "xlen", None) == "rv64"
        and width == 32
        and schema.kind
        in {"integer-writeback", "integer-multiply", "integer-shift"}
    ):
        sign_extend = "unsigned-word" not in schema.operation
    xlen = 32 if testcase.isa_profile.startswith("rv32") else 64
    return _project_gpr_result(value, width, xlen, sign_extend)


def _actual_state_boundary_properties(
    testcase: TestCase,
    risk,
    form,
    schema,
) -> tuple[tuple[str, str | int | bool], ...]:
    if schema is None:
        return ()
    if schema.kind == "csr-access" and form is not None:
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
            "little",
        )
        decoded = decode_form(form, word)
        csr = decoded.get("csr")
        if csr is not None:
            category = (
                "user-counter" if 0xC00 <= int(csr) <= 0xC1F
                else "read-only" if csr_is_read_only(int(csr))
                else "machine" if csr_minimum_privilege(int(csr)) >= 3
                else "hypervisor" if csr_minimum_privilege(int(csr)) == 2
                else "supervisor" if csr_minimum_privilege(int(csr)) == 1
                else "user"
            )
            return (("csr.address", int(csr)), ("csr.class", category))
    if schema.kind == "instruction-memory":
        return ((
            "instruction_memory.second_fetch",
            "fetch0" in tuple(testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()),
        ),)
    if schema.state_domain in {"vector", "vector-memory"}:
        boundary = testcase.dataflow_meta.get("vector_boundary")
        if isinstance(boundary, str) and boundary:
            return (("vector.boundary", boundary),)
    if schema.kind.startswith("fp") and dict(risk.operand_fields).get("rm") == 7:
        for frm in testcase.instruction_meta:
            if "frm-write" not in frm.tags or frm.byte_offset >= risk.byte_offset:
                continue
            helper_form = _emitted_form(frm.mnemonic)
            word = int.from_bytes(
                testcase.code_bytes[frm.byte_offset:frm.byte_offset + frm.byte_length],
                "little",
            ) if helper_form is not None else 0
            fields = decode_form(helper_form, word) if helper_form is not None else {}
            if frm.mnemonic == "csrrwi" and fields.get("csr") == 0x002 \
                    and type(fields.get("zimm5")) is int:
                return (("fp.frm", fields["zimm5"]),)
    return ()


def classify_testcase(testcase: TestCase) -> SemanticPoint:
    """Classify from the assembled program, not from the intent that made it."""
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    form = _emitted_form(risk.mnemonic)
    schema = (
        _compatibility_writeback_schema(
            testcase.isa_profile,
            form,
            _schema_for_profile(testcase.isa_profile, form),
        )
        if form is not None
        else None
    )
    owner = testcase.dataflow_meta.get("generation_owner")
    table_index = dict(risk.operand_fields).get("c_index")
    if form is None or owner != generation_owner_contract(form, table_index=table_index):
        raise ValueError("generation-owner-schema-mismatch")
    reservation_state = _actual_atomic_reservation_state(testcase, schema)
    operation = _actual_operation(
        testcase, risk, form, schema, reservation_state=reservation_state
    )
    properties: list[tuple[str, str | int | bool]] = []
    if form is not None:
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset : risk.byte_offset + risk.byte_length],
            "little",
        )
        if word & form.mask == form.match:
            for group, value in decode_form(form, word).items():
                properties.append((f"operand.{group}", int(value)))
    if schema is not None:
        properties.append(("effect_operation", operation))
        properties.extend(_actual_state_boundary_properties(testcase, risk, form, schema))
        if reservation_state is not None:
            properties.append(("atomic_reservation_state", reservation_state))
    violations = actual_violation_set(testcase)
    if violations:
        properties.append(("violation_set", ",".join(sorted(violations))))
    executed = testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()
    return SemanticPoint(
        effect_kind=schema.kind if schema is not None else "legality-only",
        lane=_declared_lane(testcase),
        primary_boundary_class=_actual_boundary_class(testcase, form, schema),
        properties=tuple(sorted(properties)),
        observer_kind=(
            testcase.observability_contract.sink_transform
            if testcase.observability_contract is not None
            else "outcome"
        ),
        register_relation=_actual_register_relation(testcase),
        layout_shape=f"blocks-{len(testcase.block_meta)}|instructions-{len(executed)}",
        source_mode=(
            "producer"
            if any("sequence-producer" in item.tags for item in testcase.instruction_meta)
            else "initial-state"
        ),
    )


def admit_testcase(
    testcase: TestCase,
    intent: MaterializationIntent,
) -> AdmissionResult:
    for stage, audit in (
        ("state", _state_audit),
        ("definedness", _definedness_audit),
        ("observability", _observability_audit),
        ("lane", _lane_audit),
    ):
        reason = audit(testcase)
        if reason is not None:
            return AdmissionResult(False, stage, reason, None)
    try:
        point = classify_testcase(testcase)
    except ValueError as error:
        return AdmissionResult(False, "classification", str(error), None)
    realized = _obligation_matches_testcase(testcase, intent)
    if not realized:
        return AdmissionResult(False, "classification", "concrete-state-does-not-satisfy-obligation", None)
    declared_point = testcase.dataflow_meta.get("semantic_point")
    if declared_point is not None and declared_point != point.to_dict():
        return AdmissionResult(False, "classification", "semantic-point-mismatch", None)
    return AdmissionResult(
        True,
        None,
        None,
        point,
    )

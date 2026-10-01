"""Universal assembler: one intent row -> one concrete testcase.

There is no per-family template here.  What a program needs is decided by the
form's own operand slots and by the effect's writeback target, both of which
the catalog and the effect schema already carry, so a form nobody wrote code
for assembles exactly like one that has been in the corpus for months.
"""

from dataclasses import dataclass, replace
import re

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
    RVGEN_CANONICAL_TAG,
    BlockMeta,
    CompareMask,
    InstructionMeta,
    MemoryRegion,
    ObservabilityContract,
    TestCase,
)
from .._translation_cell import canonical_realization_id
from ..paths import reference_config_identity
from ..riscv_catalog import (
    OfficialForm,
    catalog_partition_for_form,
)
from ..riscv_encoding import (
    REGISTER_OPERAND_ROLES,
    U64_MASK,
    decode_form,
    assembly_mnemonic,
    decode_register_field_value,
    encode_register_field_value,
    encode_form,
    encode_instruction,
    form_for_mnemonic,
    canonical_immediate_field,
    implicit_source_roles,
    is_register_operand_group,
    operand_kind,
    operand_groups,
    operand_role,
    rewrite_instruction_operands,
)
from ..spec_definedness import (
    SemanticRealization,
    enabled_extensions,
    isa_profile_for_extensions,
    required_extension_sets_for_form,
    rv32_xregister_compatibility_profile,
    rv32_xregister_pair_compatibility_profile,
    rv32_xregister_writeback_compatibility_profile,
    xregister_compatibility_slot_domain_for_mnemonic,
)
from ..state_contract import (
    STATE_OBSERVER_KEYS_BY_DOMAIN,
)
from .boundaries import (
    CONTROL_WITNESS_PREFIX,
    FIXED_FIELD_WITNESS_PREFIX,
    IEEE_FORMAT_BY_PRECISION,
    control_witness_disabled_extensions,
    control_witness_pairs,
    declared_exact_gpr_control_consumer_value,
    fixed_field_witness_pairs,
    semantic_immediate_value,
    stack_seed_value,
    stack_shape,
    stack_slot_offsets,
    vector_state_boundary,
    vector_vtype_codes,
    branch_condition_holds,
    xlen_mask,
)
from .effects import (
    MEMORY_EFFECT_KINDS,
    EffectSchema,
    generation_owner_contract,
    generation_rule_for_form,
    generation_schema_for_form,
    producer_chain_role,
    schema_for_realization,
    rv32_xregister_pair_roles,
    vector_default_state,
    _vector_data_group_factor,
    _vector_mask_result,
    vector_crypto_violation,
    vector_dynamic_state_keys,
    vector_fault_only_first,
    vector_register_group_size,
    vector_mask_operand,
    _vector_mask_source_names,
    vector_vstart_violation,
)
from .intents import (
    CONSUMER_REGISTER,
    LANE_LEGALITY,
    MaterializationIntent,
    _csr_required_extensions_for_materialized_profile,
    _profile_enables_emitted_form,
    per_form_prerequisite_extension_sets,
)
from .ledger import realized_facts as _shared_realized_facts
from .obligations import normative_obligation_for


DESTINATION_REGISTER = 5
SOURCE_REGISTERS = (6, 7, 9)
# Keep pair bases even and away from the fixed consumer (x8), result pair
# (x10/x11), memory base (x14) and producer/CSR scratch registers.  This lets
# a D-precision x-register source survive the shared branch/address consumers
# without adding a per-testcase register parameter.
PAIR_SOURCE_REGISTERS = (6, 12, 4)
COMPRESSED_BASE_REGISTER = 8
# Memory effects address the second half of the region so the observer's own
# store at offset 0 never overlaps the line under test.  The base lives in a
# compressed-reachable register so 16-bit forms can name it.
MEMORY_SCRATCH_REGISTER = 9
FFLAGS_CSR = 0x001
FRM_CSR = 0x002
VSTART_CSR = 0x008
VXSAT_CSR = 0x009
VXRM_CSR = 0x00A
VL_CSR = 0xC20
VTYPE_CSR = 0xC21
VLENB_CSR = 0xC22
TEST_MEMORY_BYTES = 128
VECTOR_TEST_MEMORY_BYTES = 512
STORE_OFFSET = 0
MEMORY_ACCESS_OFFSET = 64
JVT_CSR = 0x017
STACK_FRAME_OFFSET = 128
STACK_MEMORY_BYTES = 288
STACK_DESTINATION_POISON = 0xDEADBEEFCAFEBABE
STACK_FALLTHROUGH_MARKER = 0x7A5
PAIRED_WRITEBACK_REGISTER = 10
PAIRED_SOURCE_REGISTER = 12
FP_PRODUCER_GPR = 11
PRODUCER_LOAD_ID = "prod0"
VECTOR_AVL_REGISTER = 18
CSR_OBSERVER_REGISTER = 13
CSR_OBSERVER_OFFSET = 32
VECTOR_OBSERVER_OFFSET = 128
# RV32E rows may still use the full-width vector encodings, but every scalar
# helper and observation address must stay inside x0..x15.
_RVE_VECTOR_BASE_REGISTER = 14
_RVE_VECTOR_AVL_REGISTER = 13


@dataclass(frozen=True)
class _Emitted:
    instruction_id: str
    mnemonic: str
    word: int
    byte_length: int
    rd: int | None = None
    rs1: int | None = None
    rs2: int | None = None
    rs3: int | None = None
    immediate: int | None = None
    tags: tuple[str, ...] = ()


def _slot_registers(
    form: OfficialForm,
    relation: str,
    *,
    memory_effect: bool = False,
    pair_roles: tuple[str, ...] = (),
) -> dict[str, int]:
    """Assign an architectural register to every register slot of a form."""
    assignment: dict[str, int] = {}
    sources = [register for register in SOURCE_REGISTERS if register != MEMORY_SCRATCH_REGISTER]
    pair_sources = [register for register in PAIR_SOURCE_REGISTERS if register != MEMORY_SCRATCH_REGISTER]
    paired = set(pair_roles)
    for name in operand_groups(form):
        if not is_register_operand_group(form, name):
            continue
        role = operand_role(name)
        if name == "c_sreg1":
            assignment[name] = 8
            continue
        if name == "c_sreg2":
            assignment[name] = 9
            continue
        if memory_effect and name.startswith("rs1"):
            assignment[name] = MEMORY_SCRATCH_REGISTER
            continue
        if role in paired:
            if role == "rd":
                register = PAIRED_WRITEBACK_REGISTER
            else:
                register = pair_sources.pop(0) if pair_sources else PAIRED_SOURCE_REGISTER
        elif role == "rd" or name == "c_sreg1":
            register = DESTINATION_REGISTER
            if relation == "rd-eq-x0":
                register = 0
            elif relation == "rd-eq-rs1":
                register = SOURCE_REGISTERS[0]
                # A combined compressed rd/rs1 slot already consumed the
                # alias.  Its independent rs2 must not accidentally reuse
                # that same register as well.
                if "rs1" in name:
                    sources = [source for source in sources if source != register]
        else:
            register = sources.pop(0) if sources else CONSUMER_REGISTER
        if (name.endswith("_p") and not name.startswith("p_")) or name in {
            "c_sreg1",
            "c_sreg2",
        }:
            register = COMPRESSED_BASE_REGISTER + (register % 8)
            if register == MEMORY_BASE_REGISTER:
                register = COMPRESSED_BASE_REGISTER
        assignment[name] = register
    if relation in {"rd-eq-rs1", "rd-eq-x0"}:
        rd_field = next((name for name in assignment if operand_role(name) == "rd"), None)
        rs1_field = next((name for name in assignment if operand_role(name) == "rs1"), None)
        if rd_field is not None:
            assignment[rd_field] = (
                0
                if relation == "rd-eq-x0"
                else assignment.get(rs1_field, assignment[rd_field])
            )
    return assignment


def _slot_for_role(registers: dict[str, int], role: str) -> str | None:
    """The register field that carries a source role, combined slots included."""
    role = "rs" + role[2:] if role.startswith("fs") else role
    for name in sorted(registers):
        if role in name or (name == "c_rs2" and role == "rs2"):
            return name
    return None


def _xregister_pair_high_register(
    form: OfficialForm,
    schema: EffectSchema | None,
    intent: MaterializationIntent,
    registers: dict[str, int],
    key: str,
) -> int | None:
    if (
        schema is not None
        and schema.kind == "atomic-rmw"
        and schema.operation == "cas"
        and schema.writeback == "paired-gpr"
        and key in {"rd+1", "rs2+1"}
    ):
        base_register = registers.get(key[:-2])
        if base_register in (None, 0) and key == "rd+1":
            return None
        if base_register is None or base_register >= 31:
            raise UnrealizedState(f"paired atomic register key has no writable base: {key}")
        return int(base_register) + 1
    if not key.endswith("+1"):
        return None
    role = key[:-2]
    pair_roles = rv32_xregister_pair_roles(
        form,
        schema,
        xlen=_intent_xlen(form, intent),
        register_carrier=str(intent.realization_register_carrier),
    )
    if role not in pair_roles:
        return None
    base_register = registers.get(role)
    if base_register is None or int(base_register) >= 31:
        raise UnrealizedState(f"paired GPR key has no writable base: {key}")
    if int(base_register) & 1:
        raise UnrealizedState(f"paired GPR base is not even: {role}")
    return int(base_register) + 1


def _implicit_registers(form: OfficialForm, schema: EffectSchema | None) -> dict[str, int]:
    """Architecture-fixed compressed operands absent from variable fields."""
    fields = operand_groups(form)
    implicit: dict[str, int] = {}
    if "rs1" in implicit_source_roles(form):
        implicit["rs1"] = 2
    if "c_nzimm10" in fields:
        implicit.update({"rs1": 2, "rd": 2})
    if (schema is not None and schema.kind == "control-jump"
            and schema.writeback == "gpr"
            and not any(name.startswith("rd") for name in fields)):
        implicit.setdefault("rd", 1)
    return implicit


def _control_witness_for_intent(
    form: OfficialForm,
    schema: EffectSchema | None,
    intent: MaterializationIntent,
):
    """Recover the derived control witness named by a witness intent.

    Mirrors the fixed-field lookup: the intent's boundary class names the
    deterministic descriptor label, and the descriptor family owns the raw
    words and the extra state/profile facts the pair needs to execute.
    """
    if not str(intent.boundary_class).startswith(CONTROL_WITNESS_PREFIX):
        return None
    label = str(intent.boundary_class).removeprefix(CONTROL_WITNESS_PREFIX)
    matches = [
        witness
        for witness in control_witness_pairs(form, schema)
        if witness.boundary.label() == label
    ]
    return matches[0] if len(matches) == 1 else None


def _risk_instructions(
    form: OfficialForm,
    intent: MaterializationIntent,
    schema: EffectSchema | None,
) -> tuple[list[_Emitted], dict[str, int]] | None:
    """Encode the intent's risk instruction(s); None when it cannot be encoded."""
    memory_effect = schema is not None and schema.kind in MEMORY_EFFECT_KINDS
    fixed_witness = None
    if intent.lane == LANE_LEGALITY and str(intent.boundary_class).startswith(
        FIXED_FIELD_WITNESS_PREFIX
    ):
        field = str(intent.boundary_class).removeprefix(FIXED_FIELD_WITNESS_PREFIX)
        matches = [
            witness
            for witness in fixed_field_witness_pairs(form)
            if witness.boundary.field == field
        ]
        if len(matches) != 1:
            return None
        fixed_witness = matches[0]
    control_witness = _control_witness_for_intent(form, schema, intent)
    operands = intent.operands_map
    pair_roles = rv32_xregister_pair_roles(
        form,
        schema,
        xlen=_intent_xlen(form, intent),
        register_carrier=str(intent.realization_register_carrier),
    )
    registers = _slot_registers(
        form,
        intent.register_relation,
        memory_effect=memory_effect,
        pair_roles=pair_roles,
    )
    for name in operand_groups(form):
        if (
            is_register_operand_group(form, name)
            and name in operands
        ):
            registers[name] = decode_register_field_value(name, int(operands[name]))
    if schema is not None and schema.kind == "atomic-rmw" and schema.writeback == "paired-gpr":
        if "rd" in registers and "rd" not in operands:
            registers["rd"] = 0 if intent.register_relation == "rd-eq-x0" else PAIRED_WRITEBACK_REGISTER
        if "rs2" in registers and "rs2" not in operands:
            registers["rs2"] = PAIRED_SOURCE_REGISTER
    if memory_effect and intent.register_relation == "rd-eq-rs1":
        address_register = (
            PAIRED_WRITEBACK_REGISTER
            if schema is not None and schema.writeback == "paired-gpr"
            else MEMORY_SCRATCH_REGISTER
        )
        if "rd" in registers and "rs1" in registers:
            registers["rd"] = address_register
            registers["rs1"] = address_register
    registers = {**registers, **_implicit_registers(form, schema)}
    if (
        form.mnemonic == "cm_jalt"
        and int(operands.get("c_index", 0)) >= 32
    ):
        registers.setdefault("rd", 1)
    if intent.uses_reduced_gpr_domain and _intent_xlen(form, intent) == 32:
        # Register-domain rows are still executable programs. Keep every
        # non-boundary GPR operand inside RV32E's x0..x15 window while
        # preserving x16/x31 on the slot that owns the legality boundary.
        boundary_parts = intent.boundary_class.split(":", 2)
        boundary_slot = (
            boundary_parts[1]
            if boundary_parts[0] == "register-domain" and len(boundary_parts) == 3
            else None
        )
        preserve_zcmp_slot = intent.boundary_class.startswith("zcmp-sreg:")
        for name, register in tuple(registers.items()):
            if (
                name == boundary_slot
                or (preserve_zcmp_slot and name in {"c_sreg1", "c_sreg2"})
                or register < 16
            ):
                continue
            role = operand_role(name)
            domain = xregister_compatibility_slot_domain_for_mnemonic(
                intent.realization_isa_profile,
                form.mnemonic,
                role,
                register,
            )
            if domain == "gpr":
                registers[name] = MEMORY_SCRATCH_REGISTER
    values: dict[str, int] = {}
    for name in operand_groups(form):
        if is_register_operand_group(form, name):
            values[name] = encode_register_field_value(name, registers[name])
            continue
        if name in operands and operands[name] is not None:
            values[name] = int(operands[name])
            continue
        values[name] = 0
    try:
        word = encode_form(form, values)
    except ValueError:
        return None
    witness = control_witness or fixed_witness
    if witness is not None:
        # The witness word is the raw encoding authority.  Rebuild register
        # metadata from its concrete bytes so provenance audits see one
        # consistent instruction.  A control witness takes precedence when
        # both descriptor routes are present, matching the old overwrite order.
        word = witness.illegal_word
        try:
            decoded = decode_form(form, word)
        except ValueError:
            return None
        registers = {
            name: decode_register_field_value(name, int(decoded[name]))
            for name in operand_groups(form)
            if is_register_operand_group(form, name) and name in decoded
        }
        registers = {**registers, **_implicit_registers(form, schema)}
        values = {**values, **{name: int(value) for name, value in decoded.items()}}
    slots: dict[str, int] = {}
    for slot_name, register in registers.items():
        role = operand_role(slot_name)
        if role in REGISTER_OPERAND_ROLES:
            slots.setdefault(role, register)
            if role == "rd" and "rs1" in slot_name:
                slots.setdefault("rs1", register)
    immediate_name = canonical_immediate_field(form)
    immediate = None
    if immediate_name is not None and values.get(immediate_name) is not None:
        immediate_value = values[immediate_name]
        immediate = (
            int(immediate_value)
            if immediate_name == "c_imm6"
            and any(name.startswith("c_nzuimm6") for name in form.variable_fields)
            else semantic_immediate_value(immediate_name, int(immediate_value))
        )
    emitted = [
        _Emitted(
            instruction_id="i0",
            mnemonic=(
                "cm.jt"
                if form.mnemonic == "cm_jalt" and int(values.get("c_index", 32)) < 32
                else assembly_mnemonic(form)
            ),
            word=word,
            byte_length=form.encoding_length_bytes,
            rd=slots.get("rd"),
            rs1=slots.get("rs1"),
            rs2=slots.get("rs2"),
            rs3=slots.get("rs3"),
            immediate=immediate,
            tags=("risk",),
        )
    ]
    return emitted, registers


def _bound_control_flow(
    form: OfficialForm,
    emitted: list[_Emitted],
    state: tuple[int, ...],
    *,
    xlen: int = 64,
    preserve_target_low_bit: bool = False,
) -> tuple[list[_Emitted], tuple[str, ...], tuple[int, ...]] | None:
    """把控制指令的落点绑定到实际布局上，并给出真实的执行序列。

    位移原本停留在 intent 声明的绝对值（常见是 0），于是分支一旦成立就跳回自己——
    那是死循环，既拿不到 observer，在真实 campaign 里也只会超时。这里改成按 PC 相对
    计算：分支跳过紧随其后的 marker（两条路径因此可区分），跳转落到下一条指令；
    ``expected_executed_instruction_ids`` 只列真正执行到的指令。

    绑定不了的（比如目标落在指令边界之外）返回 None——不materialize 一个跑不完的测例。
    """
    offsets: list[int] = []
    cursor = 0
    for item in emitted:
        offsets.append(cursor)
        cursor += item.byte_length
    control_index = next(
        (index for index, item in enumerate(emitted) if "risk" in item.tags),
        None,
    )
    if control_index is None:
        return None
    control = emitted[control_index]
    is_branch = _is_branch_form(form)
    # 分支跳过 marker，跳转落在下一条：两者都保证向前推进，不会跳回自己。
    target_index = control_index + (2 if is_branch else 1)
    if target_index > len(emitted):
        return None
    target_offset = offsets[target_index] if target_index < len(emitted) else cursor
    rebound = _rebound_control_instruction(
        form,
        control,
        offsets[control_index],
        target_offset,
        state,
        xlen=xlen,
        preserve_target_low_bit=preserve_target_low_bit,
    )
    if rebound is None:
        return None
    control, state = rebound
    emitted = [*emitted[:control_index], control, *emitted[control_index + 1 :]]
    taken = True if not is_branch else _branch_is_taken(control, state, xlen=xlen)
    if taken is None:
        return None
    executed: list[str] = []
    index = 0
    while index < len(emitted):
        executed.append(emitted[index].instruction_id)
        index = target_index if index == control_index and taken else index + 1
    return emitted, tuple(executed), state


def _is_branch_form(form: OfficialForm) -> bool:
    if form.encoding_length_bytes == 2:
        return (form.match & 0x3) == 1 and ((form.match >> 13) & 0x7) in {0b110, 0b111}
    return (form.match & 0x7F) == 0x63


def _rebound_control_instruction(
    form: OfficialForm,
    control: _Emitted,
    control_offset: int,
    target_offset: int,
    state: tuple[int, ...],
    *,
    xlen: int = 64,
    preserve_target_low_bit: bool = False,
) -> tuple[_Emitted, tuple[int, ...]] | None:
    """按落点重写控制指令。PC 相对形式改位移，间接跳转改目标（立即数或基址寄存器）。"""
    groups = operand_groups(form)
    immediate_group = next((name for name in groups if operand_kind(form, name) == "imm"), None)
    if not _is_branch_form(form) and any(
        name.startswith("rs1") or name == "c_rs1_n0"
        for name in groups
    ):
        if immediate_group is None:
            # 压缩间接跳转没有立即数槽，只能把目标放进基址寄存器。
            if control.rs1 in (None, 0):
                return None
            return control, _with_state(
                state,
                control.rs1,
                target_offset | int(preserve_target_low_bit),
                xlen=xlen,
            )
        # JALR 的立即数相对 rs1，而不是相对当前 PC。
        immediate = target_offset - int(state[control.rs1])
        immediate = (immediate & ~1) | (int(control.immediate or 0) & 1)
        word = rewrite_instruction_operands(control.mnemonic, control.word, imm=immediate)
        return replace(control, word=word, immediate=immediate), state
    if immediate_group is None:
        return None
    displacement = target_offset - control_offset
    try:
        word = rewrite_instruction_operands(control.mnemonic, control.word, imm=displacement)
    except ValueError:
        return None
    return replace(control, word=word, immediate=displacement), state


def _branch_is_taken(
    control: _Emitted,
    state: tuple[int, ...],
    *,
    xlen: int = 64,
) -> bool | None:
    """分支条件由 funct3 决定，六个形式共用一张表，不按助记符分派。"""
    if control.rs1 is None:
        return None
    if control.byte_length == 2:
        operation = "beq" if ((control.word >> 13) & 0x7) == 0b110 else "bne"
        return branch_condition_holds(state[control.rs1], 0, operation, xlen=xlen)
    condition = (control.word >> 12) & 0x7
    if control.rs2 is None:
        return None
    return branch_condition_holds(
        state[control.rs1],
        state[control.rs2],
        {
            0b000: "beq",
            0b001: "bne",
            0b100: "blt",
            0b101: "bge",
            0b110: "bltu",
            0b111: "bgeu",
        }.get(condition),
        xlen=xlen,
    )


def _with_state(
    state: tuple[int, ...], register: int, value: int, *, xlen: int = 64
) -> tuple[int, ...]:
    updated = list(state)
    updated[int(register)] = int(value) & xlen_mask(xlen)
    updated[0] = 0
    return tuple(updated)


def _store_register_observer(
    register: int,
    offset: int,
    instruction_id: str,
    *,
    base_register: int = MEMORY_BASE_REGISTER,
    xlen: int = 64,
) -> _Emitted:
    mnemonic = "sw" if xlen == 32 else "sd"
    word = encode_instruction(mnemonic, rs1=base_register, rs2=register, imm12s=offset)
    return _Emitted(
        instruction_id=instruction_id,
        mnemonic=mnemonic,
        word=word,
        byte_length=4,
        rs1=base_register,
        rs2=register,
        immediate=offset,
        tags=("consumer", "observable"),
    )


def _csr_observer_rows(
    rows: tuple[tuple[str, int, int, str], ...],
    *,
    offset: int,
    base_register: int = MEMORY_BASE_REGISTER,
    xlen: int = 64,
    state_tag: str | None = None,
) -> list[_Emitted]:
    read_tags = ("consumer",) if state_tag is None else ("consumer", state_tag)
    store_tags = (
        ("consumer", "observable")
        if state_tag is None
        else ("consumer", "observable", state_tag)
    )
    emitted: list[_Emitted] = []
    for instruction_id, register, csr, store_id in rows:
        emitted.extend(
            (
                _Emitted(
                    instruction_id=instruction_id,
                    mnemonic="csrrs",
                    word=encode_instruction("csrrs", rd=register, rs1=0, csr=csr),
                    byte_length=4,
                    rd=register,
                    rs1=0,
                    tags=read_tags,
                ),
                replace(
                    _store_register_observer(
                        register,
                        offset,
                        store_id,
                        base_register=base_register,
                        xlen=xlen,
                    ),
                    tags=store_tags,
                ),
            )
        )
        offset += 8
    return emitted


def _vector_config_observers(
    destination_register: int | None,
    *,
    base_register: int = MEMORY_BASE_REGISTER,
    xlen: int = 64,
) -> list[_Emitted]:
    emitted: list[_Emitted] = []
    offset = 0
    if destination_register not in (None, 0):
        emitted.append(
            replace(
                _store_register_observer(
                    int(destination_register),
                    offset,
                    "vrdobs",
                    base_register=base_register,
                    xlen=xlen,
                ),
                tags=("consumer", "observable", "vector-config-state"),
            )
        )
        offset += 8
    return [
        *emitted,
        *_csr_observer_rows(
            (
                ("vrl", 10, VL_CSR, "vlobs"),
                ("vrt", 11, VTYPE_CSR, "vtypeobs"),
                ("vrs", 12, VSTART_CSR, "vstartobs"),
                ("vrb", 13, VLENB_CSR, "vlenbobs"),
            ),
            offset=offset,
            base_register=base_register,
            xlen=xlen,
            state_tag="vector-config-state",
        ),
    ]


def _vector_helper_encoding(mnemonic: str, values: dict[str, int]) -> _Emitted:
    helper_form = form_for_mnemonic(mnemonic)
    if helper_form is None:
        raise UnrealizedState(f"vector helper is not in the selected catalog: {mnemonic}")
    fields = {name: int(values.get(name, 0)) for name in operand_groups(helper_form)}
    try:
        word = encode_form(helper_form, fields)
    except ValueError as exc:
        raise UnrealizedState(f"vector helper does not encode: {mnemonic}") from exc
    return _Emitted(
        instruction_id="vector-helper",
        mnemonic=assembly_mnemonic(helper_form),
        word=word,
        byte_length=helper_form.encoding_length_bytes,
        rd=fields.get("rd"),
        rs1=fields.get("rs1"),
        rs2=fields.get("rs2"),
        rs3=fields.get("rs3"),
        immediate=next(
            (
                semantic_immediate_value(name, fields[name])
                if name.startswith("simm")
                else fields[name]
                for name in ("zimm5", "simm5", "zimm11", "zimm10")
                if name in fields
            ),
            None,
        ),
        tags=("vector-state-setup",),
    )


def _vector_base_register(intent: MaterializationIntent) -> int:
    return (
        _RVE_VECTOR_BASE_REGISTER
        if intent.uses_reduced_gpr_domain
        and _intent_xlen(form_for_mnemonic(intent.form), intent) == 32
        else MEMORY_BASE_REGISTER
    )


def _validate_vector_boundary(
    form: OfficialForm,
    schema: EffectSchema,
    boundary: str,
) -> object | None:
    """Reject Vector labels that the selected descriptor path cannot realize."""
    is_config = schema.kind == "may-be-operation" and schema.operation.startswith(
        "vector-config"
    )
    is_config_boundary = boundary.startswith("vector-config:") or (
        is_config and boundary.startswith("register-domain:")
    )
    if is_config != is_config_boundary:
        raise UnrealizedState(
            "vector-config intent has a non-config boundary"
            if is_config
            else "vector-data intent has a config boundary"
        )
    if is_config:
        return None
    state = vector_state_boundary(boundary)
    if (boundary.startswith("vector-vstart:")
            and (state is None or state.vstart is None or state.vstart < 0)):
        raise UnrealizedState("vector-vstart boundary is unrecognized")
    if boundary.startswith("vector-vtype:"):
        if state is None or state.vill:
            # A vill row carries no concrete SEW/LMUL: the setup forces a
            # reserved vsew encoding so the hardware sets vtype.vill=1 and
            # clears vl/vstart (RVV 1.0 2.3.2).  The observer reads vtype
            # back through the CSR channel, no element-width carrier needed.
            return state
        if (
            state.sew is None
            or state.lmul is None
            or vector_vtype_codes(state.sew, state.lmul) is None
        ):
            # The observer has no vse128.v carrier, so unsupported VTYPE edges
            # must remain pending instead of falling back to the default SEW/LMUL.
            raise UnrealizedState("vector-vtype boundary lacks a concrete observer setup")
    if boundary.startswith("vector-mask:") and vector_mask_operand(form, schema) is None:
        raise UnrealizedState("vector-mask boundary lacks a mask operand")
    return state


def _vector_state_setup(
    intent: MaterializationIntent,
    schema: EffectSchema,
    *,
    fp_sew_override: int | None = None,
) -> list[_Emitted]:
    """Create the smallest executable V state for one descriptor-driven row."""
    boundary = str(intent.boundary_class)
    form = form_for_mnemonic(intent.form)
    if form is None:
        raise UnrealizedState(f"vector state setup form is not in the selected catalog: {intent.form}")
    state = _validate_vector_boundary(form, schema, boundary)
    vlmax = boundary == "vector-vl:vlmax"
    avl = 0 if boundary == "vector-vl:zero" or vlmax else 1 if boundary == "vector-vl:one" else 8
    tail_agnostic = boundary != "vector-tail:undisturbed"
    # Keep VMA agnostic for ordinary rows, but use undisturbed masked-off
    # elements for exact mask boundaries so the stored destination is defined.
    if state is not None and state.axis == "vtype":
        if state.vill:
            # Reserved vsew encoding (0b111 = e1024, illegal at any concrete
            # VLEN) forces vtype.vill=1 with vl=0/vstart=0 on the target.
            vtype = 0x80 | 0x40 | (0b111 << 3) | 0
        else:
            codes = vector_vtype_codes(state.sew, state.lmul)
            if codes is None:
                raise UnrealizedState("vector-vtype boundary lacks a concrete observer setup")
            sew_code, lmul_code = codes
            vtype = (
                0x80
                | (0x40 if tail_agnostic else 0)
                | (sew_code << 3)
                | lmul_code
            )
    else:
        vtype = (0 if boundary.startswith("vector-mask:") else 0x80) | (
            0x40 if tail_agnostic else 0
        )
        default_state = vector_default_state(form)
        if default_state is not None:
            sew_code, lmul_code = vector_vtype_codes(*default_state)
            vtype = (vtype & ~0x38) | (sew_code << 3) | lmul_code
    if fp_sew_override is not None:
        # vtype SEW occupies bits [5:3]; keep the row's VMA/VTA/LMUL bits and
        # replace only the element width so the scalar observer can name a
        # concrete precision that agrees with the executed vtype.
        vtype = (vtype & ~0x38) | {16: 0x08, 32: 0x10, 64: 0x18}[fp_sew_override]
    if boundary == "vector-tail:undisturbed":
        vtype &= ~0x40
    avl_helper_register = (
        _RVE_VECTOR_AVL_REGISTER
        if _vector_base_register(intent) == _RVE_VECTOR_BASE_REGISTER
        else VECTOR_AVL_REGISTER
    )
    avl_register = 0 if vlmax else avl_helper_register
    setup_word = encode_instruction(
        "addi",
        rd=avl_helper_register,
        rs1=0,
        immediate=avl,
    )
    setup = [
        _Emitted(
            instruction_id="vavl",
            mnemonic="addi",
            word=setup_word,
            byte_length=4,
            rd=avl_helper_register,
            rs1=0,
            immediate=avl,
            tags=("vector-state-setup",),
        ),
        _vector_helper_encoding(
            "vsetvli",
            {
                "rd": avl_helper_register if vlmax else 0,
                "rs1": avl_register,
                "zimm11": vtype | 0x00,
            },
        ),
    ]
    vstart_setup: list[_Emitted] = []
    if state is not None and state.axis == "vstart":
        vstart = int(state.vstart or 0)
        xlen = _intent_xlen(form, intent)
        if vstart < 0 or vstart >= (1 << xlen):
            raise UnrealizedState("vector-vstart value is outside the concrete XLEN")
        if vstart:
            producer = _exact_gpr_producer_program(
                avl_helper_register,
                vstart,
                xlen=xlen,
            )
            if producer is None:
                raise UnrealizedState("vector-vstart value lacks a compact scalar setup")
            vstart_setup.extend(
                replace(
                    item,
                    instruction_id=f"vstart-seed{index}",
                    tags=tuple(tag for tag in item.tags if tag != "sequence-producer"),
                )
                for index, item in enumerate(producer)
            )
            vstart_setup.append(
                _Emitted(
                    instruction_id="vstart-set",
                    mnemonic="csrrw",
                    word=encode_instruction(
                        "csrrw",
                        rd=0,
                        rs1=avl_helper_register,
                        csr=VSTART_CSR,
                    ),
                    byte_length=4,
                    rd=0,
                    rs1=avl_helper_register,
                    tags=("vector-state-setup", "vector-vstart-setup"),
                )
            )
    data_registers: set[int] = set()
    index_registers: set[int] = set()
    operands = intent.operands_map
    default_state = vector_default_state(form)
    lmul = (
        state.lmul if state is not None and state.lmul is not None
        else default_state[1] if default_state is not None else 1
    )
    state_lmul = int(lmul) if getattr(lmul, "denominator", 1) == 1 else 1
    indexed_memory = schema.kind == "vector-memory" and "ei" in str(form.mnemonic)
    for name, value in intent.operands:
        if name not in {"vd", "vs1", "vs2", "vs3"}:
            continue
        base = int(value) & 31
        group = vector_register_group_size(form, schema, boundary, name)
        span = group
        if schema.kind == "vector-memory" and name in {"vd", "vs3"}:
            span *= int(operands.get("nf", 0)) + 1
        # vmv.v.i observes the active VTYPE LMUL.  Seed every active group
        # start in the declared EMUL/NF span; a single seed is enough when
        # the helper itself writes the whole group.
        seeded = set(range(base, min(32, base + span), state_lmul))
        data_registers.update(seeded)
        if indexed_memory and name == "vs2":
            index_registers.update(seeded)
    mask_operand = vector_mask_operand(form, schema)
    mask_registers = {0} if mask_operand == "v0" else set()
    mask_registers.update(
        int(operands[name])
        for name in _vector_mask_source_names(form, schema)
        if name in operands
    )
    if mask_operand in operands and mask_operand != "v0":
        mask_registers.add(int(operands[mask_operand]))
    registers = set(data_registers)
    registers.update(mask_registers)
    for index, register in enumerate(sorted(registers)):
        value = (
            0
            if boundary in {"vector-mask:all-off", "vector-mask:vm:0"}
            and register in mask_registers
            else 31
            if boundary in {"vector-mask:all-on", "vector-mask:vm:1"}
            and register in mask_registers
            else 0
            if register in index_registers
            else (index % 15) + 1
        )
        seed = _vector_helper_encoding("vmv_v_i", {"vd": register, "simm5": value})
        setup.append(replace(seed, instruction_id=f"vseed{index}"))
    return [*setup, *vstart_setup]


def _vector_state_observer(
    vd: int,
    element_width: int,
    *,
    form: OfficialForm,
    schema: EffectSchema,
    intent: MaterializationIntent,
    dynamic_state_keys: tuple[str, ...] = (),
) -> list[_Emitted]:
    if element_width not in {8, 16, 32, 64}:
        raise UnrealizedState(f"unsupported vector observer element width: {element_width}")
    base_register = _vector_base_register(intent)
    xlen = _intent_xlen(form, intent)
    # Keep vector bytes away from the scalar/CSR observation slots at offsets
    # 0..39.  Preserve the logical base relationship so direct_elf remains the
    # only owner that binds the region to its runtime address.
    rebase = _Emitted(
        instruction_id="vector-observer-base",
        mnemonic="addi",
        word=encode_instruction(
            "addi",
            rd=base_register,
            rs1=base_register,
            immediate=VECTOR_OBSERVER_OFFSET,
        ),
        byte_length=4,
        rd=base_register,
        rs1=base_register,
        immediate=VECTOR_OBSERVER_OFFSET,
        tags=("consumer", "vector-state"),
    )
    store_mnemonic = f"vse{element_width}_v"
    store_values = {
        "nf": 0,
        "vm": 1,
        "rs1": base_register,
        "vs3": int(vd) & 31,
    }
    mnemonic = assembly_mnemonic(form)
    base = mnemonic.split(".", 1)[0]
    if schema.kind == "vector-data" and _vector_mask_result(form, schema):
        store_mnemonic = "vsm_v"
        store_values = {"rs1": base_register, "vs3": int(vd) & 31}
    elif (
        schema.kind == "vector-data"
        and _vector_data_group_factor(form, "vd") > 1
    ):
        if base.startswith(("vwred", "vfwred")):
            store_mnemonic = "vs1r_v"
            store_values = {"rs1": base_register, "vs3": int(vd) & 31}
        elif not base.startswith(("vred", "vfred")):
            group = vector_register_group_size(form, schema, str(intent.boundary_class), "vd")
            if group in {1, 2, 4, 8}:
                store_mnemonic = f"vs{group}r_v"
                store_values = {"rs1": base_register, "vs3": int(vd) & 31}
    token = str(form.mnemonic).split("_", 1)[0]
    if token == "vlm":
        store_mnemonic = "vsm_v"
        store_values = {"rs1": base_register, "vs3": int(vd) & 31}
    elif token.startswith("vl") and "re" in token:
        count = token[2:].split("re", 1)[0]
        if count in {"1", "2", "4", "8"}:
            store_mnemonic = f"vs{count}r_v"
            store_values = {"rs1": base_register, "vs3": int(vd) & 31}
    else:
        fields = int(intent.operands_map.get("nf", 0)) + 1
        if fields > 1:
            store_values["nf"] = fields - 1
    store = _vector_helper_encoding(store_mnemonic, store_values)
    return [
        rebase,
        replace(store, instruction_id="vobs", tags=("consumer", "observable", "vector-state")),
        replace(
            rebase,
            instruction_id="vector-csr-base",
            immediate=-VECTOR_OBSERVER_OFFSET,
            word=encode_instruction(
                "addi",
                rd=base_register,
                rs1=base_register,
                immediate=-VECTOR_OBSERVER_OFFSET,
            ),
        ),
        *_vector_csr_observers(
            base_register=base_register,
            xlen=xlen,
            dynamic_state_keys=dynamic_state_keys,
        ),
    ]


def _vector_csr_observers(
    *,
    base_register: int = MEMORY_BASE_REGISTER,
    xlen: int = 64,
    dynamic_state_keys: tuple[str, ...] = (),
) -> list[_Emitted]:
    rows = [
        ("vvl", 10, VL_CSR, "vvlobs"),
        ("vtype", 11, VTYPE_CSR, "vtypeobs"),
        ("vstart", 12, VSTART_CSR, "vstartobs"),
        ("vlenb", 13, VLENB_CSR, "vlenbobs"),
    ]
    if "vxsat" in dynamic_state_keys:
        rows.append(("vxsat", 13, VXSAT_CSR, "vxsatobs"))
    if "vxrm" in dynamic_state_keys:
        rows.append(("vxrm", 13, VXRM_CSR, "vxrmobs"))
    if "fflags" in dynamic_state_keys:
        rows.append(("fflags", 13, FFLAGS_CSR, "fflagsobs"))
    if "frm" in dynamic_state_keys:
        rows.append(("frm", 13, FRM_CSR, "frmobs"))
    return _csr_observer_rows(
        tuple(rows),
        offset=16,
        base_register=base_register,
        xlen=xlen,
        state_tag="vector-csr-state",
    )


def _vector_element_width(boundary: str, form: OfficialForm | None = None) -> int:
    state = vector_state_boundary(str(boundary))
    if (
        state is not None
        and state.axis == "vtype"
        and state.sew is not None
        and state.lmul is not None
        and vector_vtype_codes(state.sew, state.lmul) is not None
    ):
        return int(state.sew)
    default_state = vector_default_state(form) if form is not None else None
    return default_state[0] if default_state is not None else 8


def _mark_risk_observable(emitted: list[_Emitted]) -> list[_Emitted]:
    return [
        replace(item, tags=item.tags + ("observable",))
        if "risk" in item.tags and "observable" not in item.tags
        else item
        for item in emitted
    ]


def _tag_vector_scalar_observers(emitted: list[_Emitted]) -> list[_Emitted]:
    """Tag scalar observers of a vector-to-scalar result with vector-state.

    vmv.x.s-style rows observe a value that is only defined by the executed
    vector state; without the domain tag the ELF frame cannot certify the
    observation as a vector witness (observer-gap).  Bytes stay unchanged.
    """

    return [
        replace(item, tags=item.tags + ("vector-state",))
        if "observable" in item.tags and "vector-state" not in item.tags
        else item
        for item in emitted
    ]


def _paired_register_source_registers(
    schema: EffectSchema, registers: dict[str, int]
) -> tuple[int, int]:
    if schema.operation == "move-s-to-a01":
        return int(registers["c_sreg1"]), int(registers["c_sreg2"])
    return 10, 11


def _paired_register_observers(
    schema: EffectSchema,
    registers: dict[str, int],
    *,
    xlen: int = 64,
) -> list[_Emitted]:
    width_bytes = max(int(xlen) // 8, 1)
    destinations = (
        (10, 11)
        if schema.operation == "move-s-to-a01"
        else (int(registers["c_sreg1"]), int(registers["c_sreg2"]))
    )
    return [
        _store_register_observer(
            register,
            index * width_bytes,
            f"pobs{index}",
            xlen=xlen,
        )
        for index, register in enumerate(destinations)
    ]


def _pair_result_observers(
    registers: dict[str, int],
    *,
    xlen: int,
    tag_prefix: str,
    require_even: bool = False,
) -> list[_Emitted]:
    base_register = registers.get("rd")
    if (
        base_register in (None, 0)
        or int(base_register) >= 31
        or (require_even and int(base_register) & 1)
    ):
        return []
    width_bytes = max(int(xlen) // 8, 1)
    return [
        _store_register_observer(
            int(base_register) + index,
            index * width_bytes,
            f"{tag_prefix}{index}",
            xlen=xlen,
        )
        for index in range(2)
    ]


def _stack_memory_seed(
    operation: str,
    registers: tuple[int, ...],
    stack_adj: int,
    *,
    xlen: int,
    return_target: int | None = None,
) -> tuple[dict[int, int], bytes]:
    width_bytes = xlen // 8
    value_mask = (1 << xlen) - 1
    data = bytearray(STACK_MEMORY_BYTES)
    offsets = stack_slot_offsets(registers, stack_adj, xlen=xlen)
    values = {
        register: stack_seed_value(index, register) & value_mask
        for index, register in enumerate(registers)
    }
    if return_target is not None and 1 in values:
        values[1] = int(return_target) & value_mask
    if operation in {"pop", "popret", "popretz"}:
        for register, offset in offsets.items():
            data[
                STACK_FRAME_OFFSET + offset : STACK_FRAME_OFFSET + offset + width_bytes
            ] = int(values[register]).to_bytes(width_bytes, "little")
    return values, bytes(data)


def _stack_observers(
    operation: str, registers: tuple[int, ...], *, xlen: int
) -> list[_Emitted]:
    width_bytes = xlen // 8
    observers = [
        _store_register_observer(
            register, index * width_bytes, f"obs{index}", xlen=xlen
        )
        for index, register in enumerate(registers)
    ]
    next_index = len(registers)
    if operation == "popretz":
        observers.append(
            _store_register_observer(
                10, next_index * width_bytes, "a0obs", xlen=xlen
            )
        )
        next_index += 1
    observers.append(
        _store_register_observer(2, next_index * width_bytes, "spobs", xlen=xlen)
    )
    return observers


def _stack_return_layout(
    prefix: list[_Emitted],
    success: list[_Emitted],
    *,
    xlen: int = 64,
) -> tuple[list[_Emitted], int, tuple[str, ...]]:
    fallthrough = [
        _marker_observer(CONSUMER_REGISTER, STACK_FALLTHROUGH_MARKER),
        _store_register_observer(CONSUMER_REGISTER, STORE_OFFSET, "obs", xlen=xlen),
    ]
    jump_offset = sum(item.byte_length for item in [*prefix, *fallthrough])
    success_offset = jump_offset + 4
    end_offset = success_offset + sum(item.byte_length for item in success)
    skip = _Emitted(
        instruction_id="skip",
        mnemonic="jal",
        word=encode_instruction("jal", rd=0, immediate=end_offset - jump_offset),
        byte_length=4,
        rd=0,
        immediate=end_offset - jump_offset,
        tags=("control-fallthrough-guard",),
    )
    emitted = [*prefix, *fallthrough, skip, *success]
    executed = tuple(item.instruction_id for item in [*prefix, *success])
    return emitted, success_offset, executed


# A 16 byte stride holds the widest format and keeps all three seeds below the
# region a memory effect touches.
FP_SEED_OFFSETS = (16, 32, 48)

# The load that matches each format width.  Loading at the format width is also
# what boxes a narrow value into a wider register, so the seed a program sees is
# the value the obligation declared rather than a canonical NaN.
_FP_SEED_LOAD = {8: ("flh", 2), 11: ("flh", 2), 24: ("flw", 4), 53: ("fld", 8), 113: ("flq", 16)}
_FP_MOVE_FROM_GPR = {8: ("fmv_h_x", 16), 11: ("fmv_h_x", 16), 24: ("fmv_w_x", 32), 53: ("fmv_d_x", 64)}


def _fp_seed_shape(
    schema: EffectSchema, intent: MaterializationIntent, seeds: dict[str, int]
) -> tuple[str, int]:
    """Pick the narrowest seed load that carries the declared pattern intact.

    A conversion names its *result* format, so its sources are patterns in the
    wider operand format; seeding at the schema width alone would truncate them.
    Taking the widest of the two keeps a declared value from ever being cut down
    to something the obligation did not ask for.  The NaN-boxing obligations are
    the one case that declares a whole register image, so they take the register
    width outright.
    """
    if intent.boundary_class.startswith("fp:nan-box:"):
        source_precision = schema.source_precision or schema.precision or 24
        return {
            11: ("flw", 4),
            24: ("fld", 8),
            53: ("flq", 16),
        }.get(source_precision, ("fld", 8))
    source_precision = schema.source_precision or schema.precision or 53
    format_width = _FP_SEED_LOAD.get(source_precision, ("fld", 8))[1]
    declared = max((int(value).bit_length() for value in seeds.values()), default=0)
    for mnemonic, width in sorted(_FP_SEED_LOAD.values(), key=lambda item: item[1]):
        if width >= format_width and width * 8 >= declared:
            return mnemonic, width
    return ("flq", 16)


def _producer_load(
    register: int,
    *,
    schema: EffectSchema,
    xlen: int,
) -> tuple[_Emitted, int]:
    width = schema.source_width or (32 if xlen == 32 else 64)
    signed = schema.source_signed
    mnemonic = {
        (8, True): "lb",
        (8, False): "lbu",
        (16, True): "lh",
        (16, False): "lhu",
        (32, True): "lw",
        (32, False): "lwu" if xlen == 64 else "lw",
        (64, True): "ld",
        (64, False): "ld",
    }.get((width, signed))
    if mnemonic is None:
        raise UnrealizedState(f"unsupported producer-load width/sign: {width}/{signed}")
    width = {8: 1, 16: 2, 32: 4, 64: 8}[width]
    return (
        _Emitted(
            instruction_id=PRODUCER_LOAD_ID,
            mnemonic=mnemonic,
            word=encode_instruction(
                mnemonic,
                rd=register,
                rs1=MEMORY_BASE_REGISTER,
                immediate=PRODUCER_LOAD_OFFSET,
            ),
            byte_length=4,
            rd=register,
            rs1=MEMORY_BASE_REGISTER,
            immediate=PRODUCER_LOAD_OFFSET,
            tags=("producer", "sequence-producer"),
        ),
        width,
    )


def _producer_prefix(program: list[_Emitted], *, start: int = 0) -> list[_Emitted]:
    return [
        replace(item, instruction_id=f"prod{start + index}", tags=("producer",))
        for index, item in enumerate(program)
    ]


def _sign_extended_u32_chunk(value: int) -> int:
    low = int(value) & 0xFFFFFFFF
    return ((low - (1 << 32)) if (low & 0x80000000) else low) & U64_MASK


def _exact_gpr_producer_program(register: int, value: int, *, xlen: int) -> list[_Emitted] | None:
    mask = (1 << xlen) - 1
    masked = int(value) & mask
    signed = masked - (1 << xlen) if masked & (1 << (xlen - 1)) else masked
    if -2048 <= signed <= 2047:
        return [
            _Emitted(
                instruction_id=PRODUCER_LOAD_ID,
                mnemonic="addi",
                word=encode_instruction("addi", rd=register, rs1=0, immediate=signed),
                byte_length=4,
                rd=register,
                rs1=0,
                immediate=signed,
                tags=("producer", "sequence-producer"),
            )
        ]
    if xlen != 64:
        return None
    low = int(value) & 0xFFFFFFFF
    signed32 = low - (1 << 32) if low & 0x80000000 else low
    if (signed32 & U64_MASK) != (int(value) & U64_MASK):
        return None
    upper = (signed32 + 0x800) >> 12
    lower = signed32 - (upper << 12)
    if not -2048 <= lower <= 2047:
        raise UnrealizedState("u32 constant split overflowed addiw immediate")
    prefix = _Emitted(
        instruction_id="prod0",
        mnemonic="lui",
        word=encode_instruction("lui", rd=register, immediate=upper),
        byte_length=4,
        rd=register,
        immediate=semantic_immediate_value("imm20", upper),
        tags=("producer",) if lower != 0 else ("producer", "sequence-producer"),
    )
    if lower == 0:
        return [prefix]
    return [
        prefix,
        _Emitted(
            instruction_id="prod1",
            mnemonic="addiw",
            word=encode_instruction("addiw", rd=register, rs1=register, immediate=lower),
            byte_length=4,
            rd=register,
            rs1=register,
            immediate=lower,
            tags=("producer", "sequence-producer"),
        ),
    ]


def _wide_u64_gpr_producer_program(register: int, value: int, *, xlen: int) -> list[_Emitted] | None:
    if xlen != 64:
        return None
    masked = int(value) & U64_MASK
    high = (masked >> 32) & 0xFFFFFFFF
    low = masked & 0xFFFFFFFF
    high_program = _exact_gpr_producer_program(register, _sign_extended_u32_chunk(high), xlen=64)
    if high_program is None:
        return None
    prefix = _producer_prefix(high_program)
    if low == 0:
        return prefix + [
            _Emitted(
                instruction_id=f"prod{len(prefix)}",
                mnemonic="slli",
                word=encode_instruction("slli", rd=register, rs1=register, immediate=32),
                byte_length=4,
                rd=register,
                rs1=register,
                immediate=32,
                tags=("producer", "sequence-producer"),
            )
        ]
    temp = PAIRED_SOURCE_REGISTER if register != PAIRED_SOURCE_REGISTER else PAIRED_WRITEBACK_REGISTER
    low_program = _exact_gpr_producer_program(temp, _sign_extended_u32_chunk(low), xlen=64)
    if low_program is None:
        return None
    prefix.append(
        _Emitted(
            instruction_id=f"prod{len(prefix)}",
            mnemonic="slli",
            word=encode_instruction("slli", rd=register, rs1=register, immediate=32),
            byte_length=4,
            rd=register,
            rs1=register,
            immediate=32,
            tags=("producer",),
        )
    )
    low_prefix = _producer_prefix(low_program, start=len(prefix))
    low_prefix.append(
        _Emitted(
            instruction_id=f"prod{len(prefix) + len(low_prefix)}",
            mnemonic="slli",
            word=encode_instruction("slli", rd=temp, rs1=temp, immediate=32),
            byte_length=4,
            rd=temp,
            rs1=temp,
            immediate=32,
            tags=("producer",),
        )
    )
    low_prefix.append(
        _Emitted(
            instruction_id=f"prod{len(prefix) + len(low_prefix)}",
            mnemonic="srli",
            word=encode_instruction("srli", rd=temp, rs1=temp, immediate=32),
            byte_length=4,
            rd=temp,
            rs1=temp,
            immediate=32,
            tags=("producer",),
        )
    )
    return prefix + low_prefix + [
        _Emitted(
            instruction_id=f"prod{len(prefix) + len(low_prefix)}",
            mnemonic="or",
            word=encode_instruction("or", rd=register, rs1=register, rs2=temp),
            byte_length=4,
            rd=register,
            rs1=register,
            rs2=temp,
            tags=("producer", "sequence-producer"),
        )
    ]


def _gpr_producer_program(register: int, value: int, *, xlen: int) -> list[_Emitted] | None:
    program = _exact_gpr_producer_program(register, value, xlen=xlen)
    if program is not None:
        return program
    masked = int(value) & ((1 << xlen) - 1)
    if masked and not masked & (masked - 1):
        shift = masked.bit_length() - 1
        return [
            _Emitted("prod0", "addi", encode_instruction("addi", rd=register, rs1=0, immediate=1), 4,
                     rd=register, rs1=0, immediate=1, tags=("producer",)),
            _Emitted("prod1", "slli", encode_instruction("slli", rd=register, rs1=register, immediate=shift), 4,
                     rd=register, rs1=register, immediate=shift, tags=("producer", "sequence-producer")),
        ]
    return _wide_u64_gpr_producer_program(register, value, xlen=xlen)


def _producer_sequence_setup(
    form: OfficialForm,
    schema: EffectSchema | None,
    intent: MaterializationIntent,
    registers: dict[str, int],
) -> tuple[str | None, list[_Emitted], int | None, int | None, int]:
    # A ``producer-risk`` row materializes its source-state setter before the
    # risk instruction.
    producer_loaded_role = (
        producer_chain_role(
            form,
            schema,
            intent.source_roles,
        )
        if schema is not None and intent.program_shape == "producer-risk"
        else None
    )
    if producer_loaded_role is None:
        return None, [], None, None, 0
    producer_loaded_value = intent.source_state_map.get(producer_loaded_role)
    if producer_loaded_value is None:
        raise UnrealizedState("producer-risk program shape requires a concrete producer value")
    if producer_loaded_role.startswith("fs"):
        producer_program = _producer_fp_exact_move(
            schema,
            intent,
            registers,
            producer_loaded_role,
            int(producer_loaded_value),
            xlen=_intent_xlen(form, intent),
        ) or []
        if producer_program:
            return producer_loaded_role, producer_program, int(producer_loaded_value), None, 0
        producer_load, producer_loaded_width = _producer_fp_load(
            schema,
            intent,
            registers,
            producer_loaded_role,
            int(producer_loaded_value),
        )
        return producer_loaded_role, [producer_load], int(producer_loaded_value), None, producer_loaded_width
    slot = _slot_for_role(registers, producer_loaded_role)
    if slot is None:
        raise UnrealizedState("producer-loaded gpr role has no slot")
    producer_loaded_register = registers[slot]
    if producer_loaded_register in (None, 0):
        raise UnrealizedState("producer-loaded gpr role has no writable register")
    producer_program = _gpr_producer_program(
        producer_loaded_register,
        int(producer_loaded_value),
        xlen=_intent_xlen(form, intent),
    )
    producer_loaded_width = 0
    if producer_program is None:
        loaded, producer_loaded_width = _producer_load(
            producer_loaded_register,
            schema=schema,
            xlen=_intent_xlen(form, intent),
        )
        producer_program = [loaded]
    return (
        producer_loaded_role,
        producer_program,
        int(producer_loaded_value),
        producer_loaded_register,
        producer_loaded_width,
    )


def _producer_fp_load(
    schema: EffectSchema,
    intent: MaterializationIntent,
    registers: dict[str, int],
    role: str,
    value: int,
) -> tuple[_Emitted, int]:
    source_role = "rs" + role[2:]
    slot = _slot_for_role(registers, source_role)
    if slot is None:
        raise UnrealizedState(f"producer-loaded fp role has no slot: {role}")
    mnemonic, width = _fp_seed_shape(schema, intent, {role: int(value)})
    form = form_for_mnemonic(mnemonic)
    if form is None:
        raise UnrealizedState(f"producer-loaded fp seed left the catalog: {mnemonic}")
    word = encode_form(
        form,
        {
            "rd": registers[slot],
            "rs1": MEMORY_BASE_REGISTER,
            "imm12": PRODUCER_LOAD_OFFSET,
        },
    )
    return (
        _Emitted(
            instruction_id=PRODUCER_LOAD_ID,
            mnemonic=mnemonic,
            word=word,
            byte_length=4,
            rd=registers[slot],
            rs1=MEMORY_BASE_REGISTER,
            immediate=PRODUCER_LOAD_OFFSET,
            tags=("producer", "sequence-producer"),
        ),
        width,
    )


def _producer_fp_exact_move(
    schema: EffectSchema,
    intent: MaterializationIntent,
    registers: dict[str, int],
    role: str,
    value: int,
    *,
    xlen: int,
) -> list[_Emitted] | None:
    if intent.boundary_class.startswith("fp:nan-box:"):
        return None
    precision = schema.source_precision or schema.precision or 0
    # ``fmv.d.x`` is RV64-only.  RV32D still has a real FPR source, so let the
    # caller fall back to the catalog-derived ``fld`` producer instead of
    # manufacturing an instruction that the realization cannot execute.
    if xlen == 32 and precision > 24:
        return None
    move = _FP_MOVE_FROM_GPR.get(precision)
    if precision in {8, 11}:
        gpr_value = int(value) & 0xFFFF
    elif precision == 24:
        low = int(value) & 0xFFFFFFFF
        gpr_value = ((low - (1 << 32)) if low & 0x80000000 else low) & U64_MASK
    elif precision == 53:
        gpr_value = int(value) & U64_MASK
    else:
        gpr_value = None
    if move is None or gpr_value is None:
        return None
    gpr_program = _gpr_producer_program(FP_PRODUCER_GPR, gpr_value, xlen=64)
    if gpr_program is None:
        return None
    slot = _slot_for_role(registers, "rs" + role[2:])
    if slot is None:
        raise UnrealizedState(f"producer-loaded fp role has no slot: {role}")
    move_form = form_for_mnemonic(move[0])
    if move_form is None:
        raise UnrealizedState(f"fp move producer left the catalog: {move[0]}")
    word = encode_form(move_form, {"rd": registers[slot], "rs1": FP_PRODUCER_GPR})
    prefix = _producer_prefix(gpr_program)
    return prefix + [
        _Emitted(
            instruction_id=f"prod{len(prefix)}",
            mnemonic=assembly_mnemonic(move_form),
            word=word,
            byte_length=4,
            rd=registers[slot],
            rs1=FP_PRODUCER_GPR,
            tags=("producer", "sequence-producer"),
        )
    ]


def _fp_move_observer(precision: int, source_register: int, destination: int) -> _Emitted:
    """Move a floating point result into a GPR so the store can observe it."""
    if precision <= 11:
        mnemonic = "fmv_x_h"
    elif precision <= 24:
        mnemonic = "fmv_x_w"
    else:
        mnemonic = "fmv_x_d"
    form = form_for_mnemonic(mnemonic)
    word = encode_form(form, {"rd": destination, "rs1": source_register})
    return _Emitted(
        instruction_id="mov",
        mnemonic=mnemonic.replace("_", "."),
        word=word,
        byte_length=4,
        rd=destination,
        rs1=source_register,
        tags=("consumer",),
    )


def _fp_store_observer(precision: int, source_register: int) -> _Emitted:
    """Store a wide FP result without reducing its architectural bit pattern."""
    mnemonic = "fsq" if precision > 64 else "fsd"
    form = form_for_mnemonic(mnemonic)
    word = encode_form(
        form,
        {"rs1": MEMORY_BASE_REGISTER, "rs2": source_register, "imm12s": 0},
    )
    return _Emitted(
        instruction_id="obs",
        mnemonic=mnemonic.replace("_", "."),
        word=word,
        byte_length=4,
        rs1=MEMORY_BASE_REGISTER,
        rs2=source_register,
        immediate=0,
        tags=("consumer", "observable"),
    )


def _marker_observer(destination: int, value: int) -> _Emitted:
    """A value only one control path writes, so the path itself is observable."""
    form = form_for_mnemonic("addi")
    word = encode_form(form, {"rd": destination, "rs1": 0, "imm12": value & 0xFFF})
    return _Emitted(
        instruction_id="mark",
        mnemonic="addi",
        word=word,
        byte_length=4,
        rd=destination,
        rs1=0,
        immediate=value,
        tags=("consumer", "control-marker"),
    )


def _csr_write(csr: int, value: int, instruction_id: str, tag: str) -> "_Emitted":
    word = encode_instruction("csrrwi", rd=0, csr=csr, zimm5=value & 0x1F)
    return _Emitted(
        instruction_id,
        "csrrwi",
        word,
        4,
        rd=0,
        immediate=value,
        tags=("producer", tag),
    )


def _consumer_chain(
    shape: str,
    source_register: int,
    *,
    xlen: int = 64,
    consumer_register: int = CONSUMER_REGISTER,
) -> list["_Emitted"]:
    """Emit the consumer form an intent asked for.

    How a result is consumed decides the register pressure and control flow a
    translator sees, so it is a real dimension of the obligation rather than a
    cosmetic difference in the program tail.

    This function is the byte-level tail encoder for the selected shape.
    """
    simple = {
        "compare-store": ("slt", source_register, 0),
        "unsigned-compare-store": ("sltu", 0, source_register),
    }
    if shape in simple:
        mnemonic, rs1, rs2 = simple[shape]
        word = encode_instruction(
            mnemonic, rd=consumer_register, rs1=rs1, rs2=rs2
        )
        return [
            _Emitted(
                "cons",
                mnemonic,
                word,
                4,
                rd=consumer_register,
                rs1=rs1,
                rs2=rs2,
                tags=("consumer", "risk-sequence"),
            ),
            _store_register_observer(
                consumer_register, STORE_OFFSET, "obs", xlen=xlen
            ),
        ]
    if shape == "store":
        return [_store_register_observer(source_register, STORE_OFFSET, "obs", xlen=xlen)]
    if shape == "address-store":
        # Feeding the result into an address makes it part of a memory operand.
        address_register = (
            consumer_register if consumer_register != source_register
            else 10 if source_register != 10 else CONSUMER_REGISTER
        )
        mask = encode_instruction(
            "andi", rd=address_register, rs1=source_register, imm12=0x08
        )
        add = encode_instruction("add", rd=address_register, rs1=MEMORY_BASE_REGISTER, rs2=address_register)
        store_mnemonic = "sw" if xlen == 32 else "sd"
        store = encode_instruction(
            store_mnemonic,
            rs1=address_register,
            rs2=source_register,
            imm12s=0,
        )
        return [
            _Emitted(
                "cons",
                "andi",
                mask,
                4,
                rd=address_register,
                rs1=source_register,
                immediate=0x08,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "addr",
                "add",
                add,
                4,
                rd=address_register,
                rs1=MEMORY_BASE_REGISTER,
                rs2=address_register,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "obs",
                store_mnemonic,
                store,
                4,
                rs1=address_register,
                rs2=source_register,
                immediate=0,
                tags=("consumer", "observable"),
            ),
        ]
    if shape == "branch-zero-store":
        branch = encode_instruction("bne", rs1=source_register, rs2=0, immediate=8)
        marker = encode_instruction("addi", rd=consumer_register, rs1=0, immediate=2)
        return [
            _Emitted(
                "cons",
                "bne",
                branch,
                4,
                rs1=source_register,
                rs2=0,
                immediate=8,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "markt",
                "addi",
                marker,
                4,
                rd=consumer_register,
                rs1=0,
                immediate=2,
                tags=("consumer", "control-marker"),
            ),
            _store_register_observer(
                consumer_register, STORE_OFFSET, "obs", xlen=xlen
            ),
        ]
    if shape == "jump-target-store":
        target_stride = 16
        mask = encode_instruction("andi", rd=consumer_register, rs1=source_register, immediate=target_stride)
        hi = encode_instruction("auipc", rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0)
        lo = encode_instruction(
            "addi",
            rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
            rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
            immediate=0,
        )
        add = encode_instruction(
            "add",
            rd=consumer_register,
            rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
            rs2=consumer_register,
        )
        jump = encode_instruction("jalr", rd=1, rs1=consumer_register, immediate=0)
        skip = encode_instruction("jal", rd=0, immediate=0)

        def _jump_target_block(name: str, marker: int) -> list[_Emitted]:
            marker_word = encode_instruction(
                "addi",
                rd=INSTRUCTION_MEMORY_WORD_REGISTER,
                rs1=0,
                immediate=marker,
            )
            store_mnemonic = "sw" if xlen == 32 else "sd"
            store_word = encode_instruction(
                store_mnemonic,
                rs1=MEMORY_BASE_REGISTER,
                rs2=INSTRUCTION_MEMORY_WORD_REGISTER,
                imm12s=0,
            )
            ret_word = encode_instruction("jalr", rd=0, rs1=1, immediate=0)
            nop_word = encode_instruction("addi", rd=0, rs1=0, immediate=0)
            return [
                _Emitted(
                    f"jtt{name}",
                    "addi",
                    marker_word,
                    4,
                    rd=INSTRUCTION_MEMORY_WORD_REGISTER,
                    rs1=0,
                    immediate=marker,
                    tags=("consumer",),
                ),
                _Emitted(
                    f"jtobs{name}",
                    store_mnemonic,
                    store_word,
                    4,
                    rs1=MEMORY_BASE_REGISTER,
                    rs2=INSTRUCTION_MEMORY_WORD_REGISTER,
                    immediate=0,
                    tags=("consumer", "observable"),
                ),
                _Emitted(
                    f"jtret{name}",
                    "jalr",
                    ret_word,
                    4,
                    rd=0,
                    rs1=1,
                    immediate=0,
                    tags=("consumer",),
                ),
                _Emitted(
                    f"jtnop{name}",
                    "addi",
                    nop_word,
                    4,
                    rd=0,
                    rs1=0,
                    immediate=0,
                    tags=("consumer",),
                ),
            ]

        return [
            _Emitted(
                "jtmask",
                "andi",
                mask,
                4,
                rd=consumer_register,
                rs1=source_register,
                immediate=target_stride,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "jthi",
                "auipc",
                hi,
                4,
                rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                immediate=0,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "jtlo",
                "addi",
                lo,
                4,
                rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                immediate=0,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "jtadd",
                "add",
                add,
                4,
                rd=consumer_register,
                rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                rs2=consumer_register,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "cons",
                "jalr",
                jump,
                4,
                rd=1,
                rs1=consumer_register,
                immediate=0,
                tags=("consumer", "risk-sequence"),
            ),
            _Emitted(
                "jtskip",
                "jal",
                skip,
                4,
                rd=0,
                immediate=0,
                tags=("control-fallthrough-guard",),
            ),
            *_jump_target_block("0", 2),
            *_jump_target_block("1", 3),
        ]
    raise UnrealizedState(f"unsupported consumer shape: {shape}")


def _bind_jump_target_consumer(
    emitted: list[_Emitted],
    *,
    select_high_target: bool,
) -> tuple[list[_Emitted], tuple[str, ...]]:
    offsets: dict[str, int] = {}
    cursor = 0
    for item in emitted:
        offsets[item.instruction_id] = cursor
        cursor += item.byte_length
    target0 = offsets.get("jtt0")
    target1 = offsets.get("jtt1")
    if target0 is None or target1 is None or target1 - target0 != 16:
        raise UnrealizedState("jump-target consumer target layout lost its fixed stride")
    hi_field, lo_imm = _pc_relative_split(offsets["jthi"], target0)
    rewritten: list[_Emitted] = []
    for item in emitted:
        if item.instruction_id == "jthi":
            word = encode_instruction("auipc", rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=hi_field)
            rewritten.append(replace(item, word=word, immediate=hi_field << 12))
            continue
        if item.instruction_id == "jtlo":
            word = encode_instruction(
                "addi",
                rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                immediate=lo_imm,
            )
            rewritten.append(replace(item, word=word, immediate=lo_imm))
            continue
        if item.instruction_id == "jtskip":
            displacement = cursor - offsets["jtskip"]
            word = encode_instruction("jal", rd=0, immediate=displacement)
            rewritten.append(replace(item, word=word, immediate=displacement))
            continue
        rewritten.append(item)
    prefix: list[str] = []
    for item in rewritten:
        if item.instruction_id == "jtskip":
            break
        prefix.append(item.instruction_id)
    selected = ("jtt1", "jtobs1", "jtret1") if select_high_target else ("jtt0", "jtobs0", "jtret0")
    return rewritten, tuple(prefix + list(selected) + ["jtskip"])


def _intent_xlen(form: OfficialForm, intent: MaterializationIntent) -> int:
    if intent.boundary_class.startswith("rv32:"):
        return 32
    if int(intent.realization_xlen) in {32, 64}:
        return int(intent.realization_xlen)
    if form.xlen == "rv32" or intent.uses_reduced_gpr_domain:
        return 32
    return 64


def _intent_contract_meta(
    intent: MaterializationIntent,
    *,
    form: OfficialForm,
    schema: EffectSchema | None = None,
    isa_profile: str,
) -> dict[str, object]:
    realization_xlen = _intent_xlen(form, intent)
    profile = str(isa_profile)
    extension_tokens = tuple(sorted(enabled_extensions(profile)))
    obligation = intent.normative_obligation
    realization_id = str(intent.realization_id)
    compatibility_writeback = str(intent.boundary_class).startswith(
        "rv32:compat-half-sign-writeback:"
    )
    realization_carrier = (
        "gpr" if compatibility_writeback else str(intent.realization_register_carrier)
    )
    if compatibility_writeback or (
        profile != str(intent.realization_isa_profile)
        and set(extension_tokens) >= set(intent.realization_extension_tokens)
    ):
        realization = SemanticRealization(
            xlen=32 if compatibility_writeback else realization_xlen,
            extension_tokens=extension_tokens,
            register_carrier=realization_carrier,
            privilege_class=str(intent.realization_privilege_class),
            isa_profile=profile,
            environment_profile=str(intent.realization_environment_profile),
        )
        realization_id = realization.realization_id
        obligation = normative_obligation_for(
            form,
            schema,
            realization,
            lane=str(intent.lane),
            boundary_class=str(intent.boundary_class),
        )
    elif profile != str(intent.realization_isa_profile):
        # An extension-disabled point is a legality perturbation of the owning
        # realization, not a new semantic realization.  Keep the owner identity
        # while TestCase.isa_profile records the concrete disabled execution
        # profile.
        profile = str(intent.realization_isa_profile)
        extension_tokens = tuple(intent.realization_extension_tokens)
    else:
        realization_id = canonical_realization_id(
            xlen=realization_xlen,
            extension_tokens=extension_tokens,
            register_carrier=realization_carrier,
            privilege_class=str(intent.realization_privilege_class),
            isa_profile=profile,
            environment_profile=str(intent.realization_environment_profile),
        )
    config_files = () if obligation is None else tuple(
        sorted(str(path) for path in obligation.sail_config_files if str(path))
    )
    return {
        "generation_realization": {
            "xlen": realization_xlen,
            "isa_profile": profile,
            "extension_tokens": list(extension_tokens),
            "register_carrier": realization_carrier,
            "privilege_class": str(intent.realization_privilege_class),
            "environment_profile": str(intent.realization_environment_profile),
            "realization_id": realization_id,
            "reference_config_identity": reference_config_identity(config_files),
        },
        "normative_obligation": {} if obligation is None else obligation.to_dict(),
    }


def _base_state_contract_meta(
    schema: EffectSchema | None,
    intent: MaterializationIntent,
) -> dict[str, object]:
    """Persist the state edge a testcase requires, without family templates."""
    base = {
        "sail_branch_contract": "rvgen-sail-branch-reconciliation-v1",
    }
    boundary = str(intent.boundary_class)
    if schema is None or intent.lane == LANE_LEGALITY:
        meta = {
            **base,
            "state_domain": "trap",
            "state_observer_status": "trap-observer-pending",
            "state_observer_keys": list(STATE_OBSERVER_KEYS_BY_DOMAIN["trap"]),
        }
        if boundary.startswith(
            (
                CONTROL_WITNESS_PREFIX,
                "prerequisite-gate:",
                "extension-gate:",
                "rv32:compat-extension-gate:",
                "rv32:xlen-gate:",
                "rv64:xlen-gate:",
                "alignment-",
                "memory-environment:",
            )
        ):
            meta["legality_boundary_class"] = boundary
        if schema is not None and schema.state_domain in {"vector", "vector-memory"}:
            meta["vector_boundary"] = boundary
        return meta
    kind = str(schema.kind)
    if kind == "instruction-memory":
        lifecycle_keys = STATE_OBSERVER_KEYS_BY_DOMAIN["lifecycle"]
        fence = schema.operation == "fence-fetch"
        return {
            **base,
            "state_domain": str(schema.state_domain),
            "lifecycle_state_domain": "lifecycle",
            "state_observer_status": "lifecycle-observer-pending",
            "state_observer_keys": list(
                lifecycle_keys[:2] + (lifecycle_keys[2:5] if fence else lifecycle_keys[5:])
            ),
            "lifecycle_events": list(
                ("code-store", "fence.i", "second-fetch") if fence else ("instruction-fetch",)
            ),
        }
    domain = str(schema.state_domain)
    if domain in {"vector", "vector-memory"}:
        form = form_for_mnemonic(intent.form)
        dynamic_keys = (
            vector_dynamic_state_keys(form, schema)
            if form is not None
            else ()
        )
        vector_config = (
            schema.kind == "may-be-operation"
            and schema.operation.startswith("vector-config")
        )
        vector_scalar = (
            schema.kind == "vector-data"
            and schema.writeback in {"gpr", "fpr"}
        )
        vector_register_required = not vector_config and not vector_scalar and schema.kind not in {
            "vector-data",
            "vector-memory",
        }
        vector_state_keys = list(STATE_OBSERVER_KEYS_BY_DOMAIN[domain])
        if vector_config:
            vector_state_keys = [
                key
                for key in vector_state_keys
                if key in {"vector.vl", "vector.vtype", "vector.vstart", "vector.vlenb"}
            ]
        elif vector_scalar:
            vector_state_keys.remove("vector.registers")
            vector_state_keys.remove("vector.scalar-result")
        elif not vector_register_required:
            vector_state_keys = [key for key in vector_state_keys if key not in {"vector.registers", "vector.scalar-result"}]
        else:
            vector_state_keys.remove("vector.scalar-result")
        vector_memory = domain == "vector-memory"
        return {
            **base,
            "state_domain": domain,
            "state_observer_status": (
                "vector-scalar-observer-ready"
                if vector_scalar
                else (
                    "vector-register-trace-pending"
                    if vector_register_required
                    else "vector-csr-trace-pending"
                )
            ),
            "vector_boundary": boundary,
            "memory_boundary": boundary if vector_memory else None,
            "memory_environment_disposition": (
                "environment-static-only"
                if boundary.startswith(("vector-memory:pma:", "vector-memory:pmp:"))
                else "value-observer-pending"
                if schema.kind == "vector-memory"
                else None
            ),
            "vector_dynamic_state_keys": list(dynamic_keys),
            "state_observer_keys": vector_state_keys
            + list(dynamic_keys)
            + (["vector.scalar-result"] if vector_scalar else []),
        }
    if domain == "privileged":
        return {
            **base,
            "state_domain": "privileged",
            "state_observer_status": "privileged-harness-pending",
            "state_observer_keys": list(STATE_OBSERVER_KEYS_BY_DOMAIN["privileged"]),
        }
    if domain == "csr":
        return {
            **base,
            "state_domain": "csr",
            "state_observer_status": "csr-warl-observer-pending",
            "state_observer_keys": list(STATE_OBSERVER_KEYS_BY_DOMAIN["csr"]),
        }
    if kind in {"memory-load", "memory-store", "atomic-rmw", "cache-block", "fence"} or domain == "reservation":
        reservation = domain == "reservation"
        state_domain = "reservation" if reservation else "memory"
        return {
            **base,
            "state_domain": state_domain,
            "state_observer_status": (
                "reservation-observer-pending"
                if reservation
                else "pma-pmp-observer-pending"
            ),
            "memory_boundary": boundary,
            "reservation_shape": str(intent.program_shape) if reservation else None,
            "memory_environment_disposition": (
                "environment-static-only"
                if kind in {"cache-block", "fence"} or boundary.startswith("memory-environment:")
                else "value-observer-pending"
            ),
            "state_observer_keys": list(STATE_OBSERVER_KEYS_BY_DOMAIN[state_domain]),
        }
    if kind.startswith("control-"):
        return {
            **base,
            "state_domain": "control-flow",
            "state_observer_status": "control-target-observer-ready",
            "control_boundary": boundary,
            "state_observer_keys": list(STATE_OBSERVER_KEYS_BY_DOMAIN["control-flow"]),
        }
    return {
        **base,
        "state_domain": str(schema.state_domain),
        "state_observer_keys": [],
    }




def _pc_relative_split(source_offset: int, target_offset: int) -> tuple[int, int]:
    displacement = int(target_offset) - int(source_offset)
    upper = (displacement + 0x800) >> 12
    lower = displacement - (upper << 12)
    if not -2048 <= lower <= 2047:
        raise UnrealizedState("pc-relative split overflowed addi immediate")
    return upper, lower


def _instruction_memory_target_stub(
    marker: int,
    *,
    return_register: int = 1,
    xlen: int = 64,
) -> list[_Emitted]:
    """Fetch-target scaffold for the instruction-memory route."""
    set_word = encode_instruction(
        "addi",
        rd=INSTRUCTION_MEMORY_MARKER_REGISTER,
        rs1=0,
        immediate=marker,
    )
    store_mnemonic = "sw" if xlen == 32 else "sd"
    store_word = encode_instruction(
        store_mnemonic,
        rs1=MEMORY_BASE_REGISTER,
        rs2=INSTRUCTION_MEMORY_MARKER_REGISTER,
        imm12s=INSTRUCTION_MEMORY_OBSERVER_OFFSET,
    )
    ret_word = encode_instruction("jalr", rd=0, rs1=return_register, immediate=0)
    return [
        _Emitted(
            "fetch0",
            "addi",
            set_word,
            4,
            rd=INSTRUCTION_MEMORY_MARKER_REGISTER,
            rs1=0,
            immediate=marker,
            tags=("instruction-memory-target",),
        ),
        _Emitted(
            "fetchobs",
            store_mnemonic,
            store_word,
            4,
            rs1=MEMORY_BASE_REGISTER,
            rs2=INSTRUCTION_MEMORY_MARKER_REGISTER,
            immediate=INSTRUCTION_MEMORY_OBSERVER_OFFSET,
            tags=("instruction-memory-target", "observable"),
        ),
        _Emitted(
            "fetchret",
            "jalr",
            ret_word,
            4,
            rd=0,
            rs1=return_register,
            immediate=0,
            tags=("instruction-memory-target",),
        ),
    ]


def _instruction_memory_seed(
    schema: EffectSchema,
    *,
    xlen: int,
    table_index: int | None = None,
) -> bytes:
    width = 4 if xlen == 32 else 8
    table_end = 0 if table_index is None else (table_index + 1) * width
    data = bytearray(max(INSTRUCTION_MEMORY_REGION_BYTES, table_end))
    if schema.operation == "fence-fetch":
        patch_word = encode_instruction(
            "addi",
            rd=INSTRUCTION_MEMORY_MARKER_REGISTER,
            rs1=0,
            immediate=2,
        )
        data[INSTRUCTION_MEMORY_PATCH_OFFSET : INSTRUCTION_MEMORY_PATCH_OFFSET + 4] = patch_word.to_bytes(
            4, "little"
        )
    if schema.operation == "table-jump" and table_index is not None:
        table_offset = table_index * width
        data[table_offset : table_offset + width] = bytes(width)
    return bytes(data)


def _bind_instruction_memory_adapter(
    schema: EffectSchema,
    emitted: list[_Emitted],
) -> tuple[list[_Emitted], tuple[str, ...]]:
    offsets: dict[str, int] = {}
    cursor = 0
    for item in emitted:
        offsets[item.instruction_id] = cursor
        cursor += item.byte_length
    target_offset = offsets.get("fetch0")
    if target_offset is None:
        raise UnrealizedState("instruction-memory adapter has no fetch target")
    hi_field, lo_imm = _pc_relative_split(offsets["imaddrhi"], target_offset)
    rewritten: list[_Emitted] = []
    for item in emitted:
        if item.instruction_id == "imaddrhi":
            word = encode_instruction("auipc", rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=hi_field)
            rewritten.append(replace(item, word=word, immediate=hi_field << 12))
            continue
        if item.instruction_id == "imaddrlo":
            word = encode_instruction(
                "addi",
                rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                immediate=lo_imm,
            )
            rewritten.append(replace(item, word=word, immediate=lo_imm))
            continue
        if item.instruction_id == "skip":
            skip_target = cursor
            displacement = skip_target - offsets["skip"]
            word = encode_instruction("jal", rd=0, immediate=displacement)
            rewritten.append(replace(item, word=word, immediate=displacement))
            continue
        rewritten.append(item)
    if schema.operation == "fence-fetch":
        executed = (
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
        return rewritten, executed
    if schema.operation == "table-jump":
        return_hi, return_lo = _pc_relative_split(offsets["retaddrhi"], offsets["raobs"])
        rebound: list[_Emitted] = []
        for item in rewritten:
            if item.instruction_id == "retaddrhi":
                word = encode_instruction(
                    "auipc",
                    rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    immediate=return_hi,
                )
                rebound.append(replace(item, word=word, immediate=return_hi << 12))
                continue
            if item.instruction_id == "retaddrlo":
                word = encode_instruction(
                    "addi",
                    rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    rs1=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    immediate=return_lo,
                )
                rebound.append(replace(item, word=word, immediate=return_lo))
                continue
            rebound.append(item)
        executed = (
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
        return rebound, executed
    raise UnrealizedState(f"unsupported instruction-memory operation: {schema.operation}")


_INSTRUCTION_MEMORY_ADDRESS_SETTER = _Emitted(
    "imaddrlo", "addi",
    encode_instruction(
        "addi", rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
        rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0,
    ),
    4, rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
    rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0,
    tags=("producer", "instruction-memory-address"),
)
_INSTRUCTION_MEMORY_ADDRESS_BASE = _Emitted(
    "imaddrhi", "auipc",
    encode_instruction("auipc", rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0),
    4, rd=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0,
    tags=("producer", "instruction-memory-address"),
)
_INSTRUCTION_MEMORY_SKIP = _Emitted(
    "skip", "jal", encode_instruction("jal", rd=0, immediate=0), 4,
    rd=0, immediate=0, tags=("control-fallthrough-guard",),
)


def _instruction_memory_program(
    form: OfficialForm,
    schema: EffectSchema,
    emitted: list[_Emitted],
    *,
    xlen: int,
) -> tuple[list[_Emitted], bytes, tuple[str, ...]]:
    if schema.operation == "fence-fetch":
        target = _instruction_memory_target_stub(1, xlen=xlen)
        prefix = [
            _INSTRUCTION_MEMORY_ADDRESS_BASE,
            _INSTRUCTION_MEMORY_ADDRESS_SETTER,
            _Emitted(
                "patchld",
                "lw",
                encode_instruction(
                    "lw",
                    rd=INSTRUCTION_MEMORY_WORD_REGISTER,
                    rs1=MEMORY_BASE_REGISTER,
                    immediate=INSTRUCTION_MEMORY_PATCH_OFFSET,
                ),
                4,
                rd=INSTRUCTION_MEMORY_WORD_REGISTER,
                rs1=MEMORY_BASE_REGISTER,
                immediate=INSTRUCTION_MEMORY_PATCH_OFFSET,
                tags=("producer", "instruction-memory-patch"),
            ),
            _Emitted(
                "patchst",
                "sw",
                encode_instruction(
                    "sw",
                    rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                    rs2=INSTRUCTION_MEMORY_WORD_REGISTER,
                    imm12s=0,
                ),
                4,
                rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                rs2=INSTRUCTION_MEMORY_WORD_REGISTER,
                immediate=0,
                tags=("producer", "instruction-memory-patch"),
            ),
        ]
        fetchcall = _Emitted(
            "fetchcall",
            "jalr",
            encode_instruction("jalr", rd=1, rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER, immediate=0),
            4,
            rd=1,
            rs1=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
            immediate=0,
            tags=("consumer", "instruction-memory-fetch"),
        )
        skip = _INSTRUCTION_MEMORY_SKIP
        program = [*prefix, *emitted, fetchcall, skip, *target]
        bound, executed = _bind_instruction_memory_adapter(schema, program)
        return bound, _instruction_memory_seed(schema, xlen=xlen), executed
    if schema.operation == "table-jump":
        risk = next(item for item in emitted if "risk" in item.tags)
        index = int(decode_form(form, risk.word).get("c_index", -1))
        if not 0 <= index < 256:
            raise UnrealizedState("table-jump selector left its 8-bit domain")
        width = 4 if xlen == 32 else 8
        table_offset = index * width
        table_store_mnemonic = "sw" if xlen == 32 else "sd"
        target = _instruction_memory_target_stub(
            42,
            return_register=INSTRUCTION_MEMORY_RETURN_REGISTER,
            xlen=xlen,
        )
        prefix = [
            _INSTRUCTION_MEMORY_ADDRESS_BASE,
            _Emitted(
                "retaddrhi",
                "auipc",
                encode_instruction(
                    "auipc",
                    rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    immediate=0,
                ),
                4,
                rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                immediate=0,
                tags=("producer", "instruction-memory-return"),
            ),
            _Emitted(
                "retaddrlo",
                "addi",
                encode_instruction(
                    "addi",
                    rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    rs1=INSTRUCTION_MEMORY_RETURN_REGISTER,
                    immediate=0,
                ),
                4,
                rd=INSTRUCTION_MEMORY_RETURN_REGISTER,
                rs1=INSTRUCTION_MEMORY_RETURN_REGISTER,
                immediate=0,
                tags=("producer", "instruction-memory-return"),
            ),
            _INSTRUCTION_MEMORY_ADDRESS_SETTER,
            _Emitted(
                "jvtw",
                table_store_mnemonic,
                encode_instruction(
                    table_store_mnemonic,
                    rs1=MEMORY_BASE_REGISTER,
                    rs2=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                    imm12s=table_offset,
                ),
                4,
                rs1=MEMORY_BASE_REGISTER,
                rs2=INSTRUCTION_MEMORY_ADDRESS_REGISTER,
                immediate=table_offset,
                tags=("producer", "instruction-memory-table"),
            ),
            _Emitted(
                "jvtcsr",
                "csrrw",
                encode_instruction("csrrw", rd=0, rs1=MEMORY_BASE_REGISTER, csr=JVT_CSR),
                4,
                rd=0,
                rs1=MEMORY_BASE_REGISTER,
                tags=("producer", "instruction-memory-jvt"),
            ),
            _Emitted(
                "jvtfence",
                "fence.i",
                encode_instruction("fence.i", rd=0, rs1=0, immediate=0),
                4,
                rd=0,
                rs1=0,
                immediate=0,
                tags=("producer", "instruction-memory-fence"),
            ),
        ]
        raobs = _Emitted(
            "raobs",
            table_store_mnemonic,
            encode_instruction(
                table_store_mnemonic,
                rs1=MEMORY_BASE_REGISTER,
                rs2=1,
                imm12s=INSTRUCTION_MEMORY_LINK_OFFSET,
            ),
            4,
            rs1=MEMORY_BASE_REGISTER,
            rs2=1,
            immediate=INSTRUCTION_MEMORY_LINK_OFFSET,
            tags=("consumer", "observable"),
        )
        skip = _INSTRUCTION_MEMORY_SKIP
        program = [*prefix, *emitted, raobs, skip, *target]
        bound, executed = _bind_instruction_memory_adapter(schema, program)
        return bound, _instruction_memory_seed(schema, xlen=xlen, table_index=index), executed
    raise UnrealizedState(f"unsupported instruction-memory operation: {schema.operation}")


class UnrealizedState(ValueError):
    """An obligation declared state the assembler cannot place in the program."""


def _initial_state(
    intent: MaterializationIntent,
    registers: dict[str, int],
    *,
    schema: EffectSchema | None,
) -> tuple[int, ...]:
    form = form_for_mnemonic(intent.form)
    if form is None:
        raise UnrealizedState("initial state form is unavailable")
    xlen = _intent_xlen(form, intent)
    gpr_mask = (1 << xlen) - 1
    memory_effect = schema is not None and schema.kind in MEMORY_EFFECT_KINDS

    def require_gpr_value(value: int, *, key: str) -> None:
        if not 0 <= int(value) <= gpr_mask:
            raise UnrealizedState(f"{key} value does not fit RV{xlen}")

    state = [0] * 32
    state[MEMORY_BASE_REGISTER] = LOGICAL_DATA_ADDRESS & gpr_mask
    if schema is not None and schema.state_domain in {"vector", "vector-memory"}:
        state[_vector_base_register(intent)] = LOGICAL_DATA_ADDRESS
    # The address lives in whichever register the assignment gave the base slot,
    # which is the scratch register unless an obligation named another one.
    address_register = next(
        (registers[name] for name in sorted(registers) if name.startswith("rs1")),
        MEMORY_SCRATCH_REGISTER,
    )
    # Compressed forms name a combined slot (rd_rs1_p) that is both destination
    # and first source, so the source role comes from the substring, not the
    # prefix.
    def _source_role(field: str) -> str | None:
        for role in ("rs1", "rs2", "rs3"):
            if role in field:
                return role
        return "rs2" if field == "c_rs2" else None

    by_role: dict[str, int] = {}
    for name, register in registers.items():
        role = _source_role(name)
        if role is not None and role not in by_role:
            by_role[role] = register
    declared = intent.source_state_map
    realized: set[str] = set()
    for role, register in by_role.items():
        if memory_effect and register == address_register:
            continue
        if role in declared:
            require_gpr_value(declared[role], key=role)
            state[register] = int(declared[role]) & gpr_mask
            realized.add(role)
    for key, value in intent.source_state:
        if key in registers:
            if _source_role(key) in {"rs1", "rs2", "rs3"}:
                require_gpr_value(value, key=key)
            state[registers[key]] = int(value) & gpr_mask
            realized.add(key)
            continue
        paired_high = _xregister_pair_high_register(
            form,
            schema,
            intent,
            registers,
            key,
        )
        if paired_high is not None:
            require_gpr_value(value, key=key)
            state[paired_high] = int(value) & gpr_mask
            realized.add(key)
            continue
        if key in realized or key.startswith("fs"):
            if key.startswith("fs"):
                slot = _slot_for_role(registers, f"rs{key[2:]}")
                if slot is None:
                    raise UnrealizedState(f"unrealized fp source key: {key}")
            realized.add(key)
            continue
        if key == "address-offset":
            if not memory_effect:
                raise UnrealizedState("address offset without a memory effect")
            state[address_register] = (
                LOGICAL_DATA_ADDRESS + MEMORY_ACCESS_OFFSET + int(value)
            ) & gpr_mask
            realized.add(key)
            continue
        if key == "vector-fault-element":
            if not vector_fault_only_first(form, schema) or int(value) < 0:
                raise UnrealizedState("invalid vector fault-only-first element")
            width = max(int(schema.memory_width or schema.width) // 8, 1)
            state[address_register] = (
                LOGICAL_DATA_ADDRESS
                + VECTOR_TEST_MEMORY_BYTES
                - width * int(value)
            ) & gpr_mask
            realized.add(key)
            continue
        if key == "memory":
            if not memory_effect:
                raise UnrealizedState("memory contents without a memory effect")
            realized.add(key)  # placed by _seeded_memory, not by register state
            continue
        raise UnrealizedState(f"unrealized state key: {key}")
    if memory_effect and not any(
        key in {"address-offset", "vector-fault-element"}
        for key, _ in intent.source_state
    ):
        immediate = semantic_immediate_value(
            schema.immediate_field if schema is not None else None,
            intent.operands_map.get(schema.immediate_field) if schema is not None else None,
        )
        state[address_register] = (
            LOGICAL_DATA_ADDRESS + MEMORY_ACCESS_OFFSET - int(immediate or 0)
        ) & gpr_mask
    if intent.destination_seed is not None:
        require_gpr_value(intent.destination_seed, key="destination seed")
        destination = next(
            (
                register
                for name, register in registers.items()
                if name.startswith("rd") and "rs1" not in name
            ),
            None,
        )
        if destination not in (None, 0) and destination not in HARNESS_RESERVED_GPRS:
            state[destination] = int(intent.destination_seed) & gpr_mask
    if schema is not None and schema.kind == "paired-register-transfer":
        for index, register in enumerate(_paired_register_source_registers(schema, registers)):
            state[register] = (
                ((0x61 + index) << 56)
                | ((register & 0xFF) << 48)
                | (0x1112131415161718 + index)
            ) & gpr_mask
    if intent.sibling_state:
        used = {int(register) for register in registers.values()}
        used.update(_PROJECTION_SIBLING_RESERVED)
        if schema is not None:
            used.update(int(register) for register in _paired_register_source_registers(schema, registers))
        sibling_register = next(
            (register for register in range(32) if register not in used),
            None,
        )
        if sibling_register is None:
            raise UnrealizedState("no free GPR for projection sibling state")
        for role, value in intent.sibling_state:
            require_gpr_value(value, key=f"sibling {role}")
            state[sibling_register] = int(value) & gpr_mask
    state[0] = 0
    return tuple(state)


# Producer/observer harness owns these registers: a projection sibling must
# never overlap a slot the emitted program will write or read.
_PROJECTION_SIBLING_RESERVED = frozenset(
    (
        1,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        18,
        *HARNESS_RESERVED_GPRS,
    )
)


def _rv32e_profile(profile: str) -> str:
    profile = str(profile)
    if not profile.startswith("rv32"):
        return profile
    head, *tail = profile.split("_", 1)
    suffix = head[4:]
    if suffix.startswith(("i", "e")):
        head = "rv32e" + suffix[1:]
    return head if not tail else f"{head}_{tail[0]}"


def _profile_for_intent(
    form: OfficialForm,
    intent: MaterializationIntent,
    emitted: list[_Emitted],
) -> str:
    """Execution profile derived from the concrete program this intent emitted."""
    xlen = "rv32" if _intent_xlen(form, intent) == 32 else "rv64"
    base = f"{xlen}i"
    compatibility_writeback_profile = (
        rv32_xregister_writeback_compatibility_profile(form)
        if intent.boundary_class.startswith("rv32:compat-half-sign-writeback:")
        else None
    )
    if compatibility_writeback_profile is not None:
        return compatibility_writeback_profile
    compatibility_pair_profile = (
        rv32_xregister_pair_compatibility_profile(form)
        if intent.boundary_class.startswith("rv32:compat-odd-paired-gpr-base:")
        else None
    )
    if compatibility_pair_profile is not None:
        return compatibility_pair_profile
    compatibility_suffix = (
        intent.boundary_class.removeprefix("rv32:compat-extension-gate:")
        if intent.boundary_class.startswith("rv32:compat-extension-gate:")
        else None
    )
    if compatibility_suffix:
        return f"{base}_{compatibility_suffix}"
    compatibility_profile = (
        rv32_xregister_compatibility_profile(form)
        if intent.uses_reduced_gpr_domain
        else None
    )
    if compatibility_profile is not None:
        return _rv32e_profile(compatibility_profile)
    profile = intent.realization_isa_profile
    enabled = set(enabled_extensions(profile))
    prerequisites = per_form_prerequisite_extension_sets(
        form, generation_schema_for_form(form),
    )
    if intent.boundary_class.startswith("prerequisite-gate:"):
        enabled.difference_update(set().union(*prerequisites))
    elif intent.lane != LANE_LEGALITY and prerequisites:
        enabled.update(min(prerequisites, key=lambda option: (len(option), tuple(sorted(option)))))
    if str(intent.boundary_class).startswith(CONTROL_WITNESS_PREFIX):
        # The witness family owns the profile fact the pair needs.  A
        # c-disabled axis (no-C IALIGN) must not let the emitted compressed
        # risk re-add the C token through the ordinary extension loop; other
        # witness axes keep the ordinary emitted-instruction profile.
        disabled = control_witness_disabled_extensions(
            str(intent.boundary_class).removeprefix(CONTROL_WITNESS_PREFIX)
        )
        if disabled:
            enabled.difference_update(disabled)
            return isa_profile_for_extensions(xlen, frozenset(enabled))
    if intent.extension_gate_disabled:
        disabled = {
            token
            for token in re.split(
                r"[+_]",
                intent.boundary_class.removeprefix("extension-gate:"),
            )
            if token
        }
        enabled.difference_update(disabled)
        return isa_profile_for_extensions(xlen, frozenset(enabled))
    for item in emitted:
        item_form = form_for_mnemonic(item.mnemonic)
        if item_form is None:
            raise UnrealizedState(f"emitted instruction left the catalog: {item.mnemonic}")
        if not _profile_enables_emitted_form(profile, item_form):
            alternatives = required_extension_sets_for_form(item_form)
            if not alternatives:
                raise UnrealizedState(f"emitted instruction has no source extension: {item.mnemonic}")
            enabled.update(
                min(alternatives, key=lambda option: (len(option), tuple(sorted(option))))
            )
        decoded = decode_form(item_form, item.word)
        csr = decoded.get("csr")
        if csr is not None:
            enabled.update(
                _csr_required_extensions_for_materialized_profile(int(csr), profile)
            )
    profile = isa_profile_for_extensions(xlen, frozenset(enabled))
    if intent.uses_reduced_gpr_domain:
        # RV32E changes only the integer register-file cardinality; it does
        # not silently disable C/Z* extensions required by the encoded form.
        return _rv32e_profile(profile) if xlen == "rv32" else profile
    return profile


def materialize(
    intent: MaterializationIntent,
) -> TestCase | None:
    """Assemble one intent, or return None when the intent cannot be encoded."""
    form = form_for_mnemonic(intent.form)
    if form is None:
        return None
    schema = generation_schema_for_form(form)
    expected_xlen = "rv32" if _intent_xlen(form, intent) == 32 else "rv64"
    form = form if str(form.xlen) == expected_xlen else replace(form, xlen=expected_xlen)
    # Keep the owner above canonical and derive the concrete XLEN view for all
    # boundary values, seeds, observers and expected-result calculations.
    schema = schema_for_realization(
        form,
        schema,
        xlen=_intent_xlen(form, intent),
        register_carrier=str(intent.realization_register_carrier),
    )
    if (
        schema is not None
        and intent.boundary_class.startswith("rv32:compat-half-sign-writeback:")
        and rv32_xregister_writeback_compatibility_profile(form) is not None
    ):
        schema = replace(schema, writeback="gpr", observer="gpr")
    if (
        schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and schema.kind not in {"vector-data", "vector-memory"}
        and intent.lane != LANE_LEGALITY
    ):
        _validate_vector_boundary(form, schema, str(intent.boundary_class))
    built = _risk_instructions(form, intent, schema)
    if built is None:
        return None
    emitted, registers = built
    legality = intent.lane == LANE_LEGALITY
    boundary = str(intent.boundary_class)
    vector_dynamic_keys = vector_dynamic_state_keys(form, schema)
    vector_vstart_trap = vector_vstart_violation(form, schema, boundary)
    vector_crypto_trap = vector_crypto_violation(
        form, schema, boundary, intent.realization_isa_profile
    )
    if (
        legality
        and schema is not None
        and schema.state_domain in {"vector", "vector-memory"}
        and (
            boundary.startswith("vector-vtype:")
            or boundary == "vector-frm:reserved"
            or vector_vstart_trap is not None
            or vector_crypto_trap is not None
        )
    ):
        setup = [
            item
            for item in _vector_state_setup(intent, schema)
            if not item.instruction_id.startswith("vseed")
        ]
        if boundary == "vector-frm:reserved":
            setup.append(
                _csr_write(
                    FRM_CSR,
                    5 if intent.rounding_mode is None else intent.rounding_mode,
                    "v-frm-reserved",
                    "vector-frm-reserved",
                )
            )
        emitted = setup + emitted
    writeback = schema.writeback if schema is not None else "none"
    observed_register = next(
        (register for name, register in registers.items()
         if name.startswith("rd") or name == "c_sreg1"),
        None,
    )
    consumer_register = CONSUMER_REGISTER
    if intent.consumer_shape in {"branch-zero-store", "jump-target-store"}:
        used = set(registers.values()) | set(HARNESS_RESERVED_GPRS) | {
            1,
            INSTRUCTION_MEMORY_ADDRESS_REGISTER,
            INSTRUCTION_MEMORY_WORD_REGISTER,
            INSTRUCTION_MEMORY_RETURN_REGISTER,
        }
        consumer_register = next(
            register
            for register in (
                CONSUMER_REGISTER,
                10,
                13,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                3,
                4,
            )
            if register not in used
            and (not intent.uses_reduced_gpr_domain or register < 16)
        )
    observed = False
    stack_seed_values: dict[int, int] | None = None
    stack_memory: bytes | None = None
    instruction_memory: bytes | None = None
    stack_adjust = 0
    executed_ids_override: tuple[str, ...] | None = None
    (
        producer_loaded_role,
        producer_program,
        producer_loaded_value,
        producer_loaded_register,
        producer_loaded_width,
    ) = _producer_sequence_setup(form, schema, intent, registers)
    if not legality and schema is not None:
        if schema.kind == "instruction-memory":
            emitted, instruction_memory, executed_ids_override = _instruction_memory_program(
                form,
                schema,
                emitted,
                xlen=_intent_xlen(form, intent),
            )
            # The route's fetch/link stores are compatibility scaffolding, not
            # a live lifecycle observer.  Preserve their bytes and IDs while
            # removing the semantic sink tag from every instruction-memory
            # item.
            emitted = [
                replace(
                    item,
                    tags=tuple(tag for tag in item.tags if tag != "observable"),
                )
                if "observable" in item.tags
                else item
                for item in emitted
            ]
            observed = False
        elif schema.kind == "stack-transfer":
            stack_xlen = _intent_xlen(form, intent)
            stack_registers, stack_adjust = stack_shape(intent.operands_map, xlen=stack_xlen)
            if schema.operation == "push":
                stack_seed_values, stack_memory = _stack_memory_seed(
                    schema.operation,
                    stack_registers,
                    stack_adjust,
                    xlen=stack_xlen,
                )
                emitted = emitted + [
                    _store_register_observer(2, 0, "spobs", xlen=stack_xlen)
                ]
            else:
                observers = _stack_observers(
                    schema.operation,
                    stack_registers,
                    xlen=stack_xlen,
                )
                if schema.operation in {"popret", "popretz"}:
                    emitted, return_target, executed_ids_override = _stack_return_layout(
                        emitted, observers, xlen=stack_xlen
                    )
                    stack_seed_values, stack_memory = _stack_memory_seed(
                        schema.operation,
                        stack_registers,
                        stack_adjust,
                        xlen=stack_xlen,
                        return_target=return_target,
                    )
                else:
                    stack_seed_values, stack_memory = _stack_memory_seed(
                        schema.operation,
                        stack_registers,
                        stack_adjust,
                        xlen=stack_xlen,
                    )
                    emitted = emitted + observers
            observed = True
        elif schema.kind == "paired-register-transfer":
            emitted = emitted + _paired_register_observers(
                schema,
                registers,
                xlen=_intent_xlen(form, intent),
            )
            observed = True
        elif schema.writeback == "paired-gpr" and (
            rv32_xregister_pair_roles(
                form,
                schema,
                xlen=_intent_xlen(form, intent),
                register_carrier=str(intent.realization_register_carrier),
            )
            or (
                schema.kind == "memory-load"
                and _intent_xlen(form, intent) == 32
                and intent.realization_register_carrier == "gpr"
            )
        ):
            observers = _pair_result_observers(
                registers,
                xlen=_intent_xlen(form, intent),
                tag_prefix="pobs",
                require_even=True,
            )
            if not observers:
                raise UnrealizedState("paired result lacks an aligned observable destination")
            emitted = emitted + observers
            observed = True
        elif schema.kind == "atomic-rmw" and schema.operation == "cas":
            if schema.writeback == "paired-gpr":
                observers = _pair_result_observers(
                    registers,
                    xlen=_intent_xlen(form, intent),
                    tag_prefix="cobs",
                )
                emitted = emitted + observers if observers else _mark_risk_observable(emitted)
            elif observed_register in (None, 0) or observed_register in HARNESS_RESERVED_GPRS:
                emitted = _mark_risk_observable(emitted)
            # Unpaired atomic/reservation state has no consumer until its
            # state observer is modeled; do not synthesize a scalar tail.
            observed = True
        elif schema.kind == "csr-access":
            csr = intent.operands_map.get("csr")
            if csr is None:
                raise UnrealizedState("CSR materialization lacks a concrete CSR address")
            emitted = emitted + _csr_observer_rows(
                (("csr-state-read", CSR_OBSERVER_REGISTER, int(csr), "csr-state-observer"),),
                offset=CSR_OBSERVER_OFFSET,
                xlen=_intent_xlen(form, intent),
                state_tag="csr-state",
            )
            observed = True
        elif schema.kind in {"encoding-only", "privileged-system", "fence"}:
            # These descriptors own bytes/profile/definedness (or a pending
            # trap/state contract) only.  Do not attach a guessed scalar
            # consumer merely because a carrier happens to expose rd; the
            # testcase remains outcome-only and visibly non-semantic.
            observed = False
        elif schema.kind == "vector-data" and schema.writeback in {"gpr", "fpr"}:
            # Scalar V reductions/moves have a real scalar destination.  Seed
            # the vector source, then observe that destination in its own
            # register domain; do not dump v0 and call it a scalar witness.
            scalar_element_width = _vector_element_width(str(intent.boundary_class), form)
            fp_sew_override = None
            if schema.writeback == "fpr" and scalar_element_width not in {16, 32, 64}:
                # vfmv.f.s-style moves carry no SEW in the encoding.  An
                # explicit SEW=8 boundary is nevertheless not a valid scalar
                # FP observer shape; reject it instead of changing the
                # concrete VTYPE while retaining a misleading boundary label.
                if str(intent.boundary_class).startswith("vector-vtype:"):
                    raise UnrealizedState(
                        "vector scalar FP observer requires SEW16/32/64"
                    )
                # VL/tail/mask rows do not name SEW.  Use one concrete legal
                # setup for the scalar sink without changing the named state
                # boundary.
                scalar_element_width = 32
                fp_sew_override = 32
            emitted = (
                _vector_state_setup(intent, schema, fp_sew_override=fp_sew_override)
                + emitted
            )
            if observed_register in (None, 0):
                raise UnrealizedState("vector scalar result lacks an observable destination")
            if schema.writeback == "fpr":
                precision = {16: 11, 32: 24, 64: 53}[scalar_element_width]
                # Store the scalar FPR directly so the raw NaN-boxed bits are
                # preserved for every SEW.  The previous branch only emitted
                # a sink for RV32 wide values, leaving ordinary vfmv.f.s rows
                # with no observable destination at all.
                emitted.extend(
                    _tag_vector_scalar_observers(
                        [_fp_store_observer(precision, observed_register)]
                    )
                )
            else:
                emitted.extend(
                    _tag_vector_scalar_observers(
                        [
                            _store_register_observer(
                                observed_register,
                                STORE_OFFSET,
                                "vobs",
                                xlen=_intent_xlen(form, intent),
                            )
                        ]
                    )
                )
            emitted.extend(
                _vector_csr_observers(
                    base_register=_vector_base_register(intent),
                    xlen=_intent_xlen(form, intent),
                    dynamic_state_keys=vector_dynamic_keys,
                )
            )
            observed = True
        elif schema.kind in {"vector-data", "vector-memory"}:
            # Vector state is initialized and observed through V instructions;
            # no scalar GPR sink is allowed to masquerade as vector coverage.
            emitted = _vector_state_setup(intent, schema) + emitted
            vector_base_register = _vector_base_register(intent)
            if schema.kind == "vector-memory" and schema.writeback == "memory":
                emitted = _mark_risk_observable(emitted) + _vector_csr_observers(
                    base_register=vector_base_register,
                    xlen=_intent_xlen(form, intent),
                )
            else:
                vd = next(
                    (
                        int(value)
                        for name, value in intent.operands
                        if name == "vd"
                    ),
                    0,
                )
                emitted.extend(
                    _vector_state_observer(
                        vd,
                        (
                            _vector_element_width(str(intent.boundary_class), form)
                            if schema.kind == "vector-memory"
                            and schema.writeback == "vector"
                            and "ei" in str(form.mnemonic)
                            else int(schema.memory_width or schema.width)
                            if schema.kind == "vector-memory" and schema.writeback == "vector"
                            else _vector_element_width(str(intent.boundary_class), form)
                        ),
                        form=form,
                        schema=schema,
                        intent=intent,
                        dynamic_state_keys=vector_dynamic_keys,
                    )
                )
            observed = True
        elif writeback == "memory":
            # The risk store already writes the compared region.  Keep its own
            # effective address: rebasing it erased alignment and address
            # obligations before admission could verify them.
            emitted = _mark_risk_observable(emitted)
            observed = any("risk" in item.tags and item.rs1 is not None for item in emitted)
        elif writeback == "fpr":
            # FP results either feed a real second FP operation or move as raw
            # bits into the shared integer consumers.
            precision = schema.precision or (24 if schema.width <= 32 else 53)
            if intent.consumer_shape in {"fp-classify-store", "fp-convert-store"}:
                operation = intent.consumer_shape.removeprefix("fp-").removesuffix("-store")
                prefix = {"classify": "fclass", "convert": "fcvt_w"}[operation]
                mnemonic = {
                    11: f"{prefix}_h",
                    24: f"{prefix}_s",
                    53: f"{prefix}_d",
                    113: f"{prefix}_q",
                }.get(precision)
                if mnemonic is None:
                    raise UnrealizedState(f"unsupported FP consumer precision: {precision}")
                operands = {"rd": CONSUMER_REGISTER, "rs1": observed_register or 0}
                if operation == "convert":
                    operands["rm"] = 1
                emitted.extend(
                    [
                        _Emitted(
                            "cons",
                            mnemonic.replace("_", "."),
                            encode_instruction(mnemonic, **operands),
                            4,
                            rd=CONSUMER_REGISTER,
                            rs1=observed_register or 0,
                            tags=("consumer", "risk-sequence"),
                        ),
                        _store_register_observer(
                            CONSUMER_REGISTER,
                            STORE_OFFSET,
                            "obs",
                            xlen=_intent_xlen(form, intent),
                        ),
                    ]
                )
            else:
                if precision > 64 or (
                    _intent_xlen(form, intent) == 32 and precision > 24
                ):
                    emitted.append(
                        _fp_store_observer(precision, observed_register or 0)
                    )
                else:
                    emitted.append(
                        _fp_move_observer(
                            53 if (
                                _intent_xlen(form, intent) == 64
                                and intent.boundary_class.startswith(
                                    "fp:nan-box:boxed-control:"
                                )
                            ) else precision,
                            observed_register or 0,
                            CONSUMER_REGISTER,
                        )
                    )
                    emitted.extend(
                        _consumer_chain(
                            intent.consumer_shape,
                            CONSUMER_REGISTER,
                            xlen=_intent_xlen(form, intent),
                            consumer_register=consumer_register,
                        )
                    )
            observed = observed_register is not None
        elif writeback == "control":
            emitted = emitted + [
                _marker_observer(CONSUMER_REGISTER, 1),
                _store_register_observer(
                    CONSUMER_REGISTER,
                    STORE_OFFSET,
                    "obs",
                    xlen=_intent_xlen(form, intent),
                ),
            ]
            observed = True
        elif schema.kind == "may-be-operation" and schema.operation.startswith("vector-config"):
            emitted = emitted + _vector_config_observers(
                observed_register,
                base_register=_vector_base_register(intent),
                xlen=_intent_xlen(form, intent),
            )
            observed = True
        elif (
            observed_register not in (None, 0)
            and (
                observed_register not in HARNESS_RESERVED_GPRS
                or (observed_register == 2 and "rs1" in implicit_source_roles(form))
            )
        ):
            # A non-store consumer shape is an explicit dataflow obligation.
            # Do not let the shortest descriptor sink erase that request: the
            # admission owner checks for the named consumer instruction(s).
            emitted.extend(
                _consumer_chain(
                    intent.consumer_shape,
                    observed_register,
                    xlen=_intent_xlen(form, intent),
                    consumer_register=consumer_register,
                )
            )
            observed = True

    # producer_loaded_role is only populated by the ``producer-risk`` plan
    # stage (see _producer_sequence_setup), so the shape check is implied.
    if producer_loaded_role is not None:
        emitted = [*producer_program, *emitted]

    # fs-prefixed roles are floating point wherever they appear -- an fp store
    # is a memory effect, but its stored value still lives in an f-register.
    seeds = {
        name: value
        for name, value in intent.source_state
        if name.startswith("fs") and name != producer_loaded_role
    }
    seed_mnemonic, seed_width = ("fld", 8)
    if seeds and schema is not None:
        seed_mnemonic, seed_width = _fp_seed_shape(schema, intent, seeds)
        seed_form = form_for_mnemonic(seed_mnemonic)
        seed_loads = []
        nan_box_source = intent.boundary_class.rsplit(":", 1)[-1]
        seed_precision = schema.source_precision or schema.precision or 24
        nan_box_width = IEEE_FORMAT_BY_PRECISION[seed_precision][0]
        for index in range(3):
            key = f"fs{index + 1}"
            slot = _slot_for_role(registers, f"rs{index + 1}")
            if key not in seeds or slot is None:
                continue
            offset = FP_SEED_OFFSETS[index]
            seed_loads.append(
                _Emitted(
                    instruction_id=f"seed{index}",
                    mnemonic=seed_mnemonic,
                    word=encode_form(
                        seed_form,
                        {"rd": registers[slot], "rs1": MEMORY_BASE_REGISTER, "imm12": offset},
                    ),
                    byte_length=4,
                    rd=registers[slot],
                    rs1=MEMORY_BASE_REGISTER,
                    immediate=offset,
                    tags=("producer",),
                )
            )
        emitted = seed_loads + emitted
    if vector_dynamic_keys and not legality:
        vector_prologue: list[_Emitted] = []
        if "fflags" in vector_dynamic_keys:
            vector_prologue.append(
                _csr_write(FFLAGS_CSR, 0, "v-fclr", "vector-fflags-clear")
            )
        if "frm" in vector_dynamic_keys:
            vector_prologue.append(
                _csr_write(
                    FRM_CSR,
                    0 if intent.rounding_mode is None else intent.rounding_mode,
                    "v-frm",
                    "vector-frm-write",
                )
            )
        if "vxsat" in vector_dynamic_keys:
            vector_prologue.append(
                _csr_write(VXSAT_CSR, 0, "v-vxsat", "vector-vxsat-clear")
            )
        if "vxrm" in vector_dynamic_keys:
            vector_prologue.append(
                _csr_write(VXRM_CSR, 0, "v-vxrm", "vector-vxrm-write")
            )
        emitted = vector_prologue + emitted
    # An effect that declares its exception flags observable must actually show
    # them: clear before, read after, store into the compared region.
    observes_flags = schema is not None and schema.observes_fflags and not legality
    if observes_flags:
        prologue: list[_Emitted] = [_csr_write(FFLAGS_CSR, 0, "fclr", "fflags-clear")]
        if intent.rounding_mode is not None and (
            "rm" not in operand_groups(form) or intent.operands_map.get("rm") == 7
        ):
            prologue.append(_csr_write(FRM_CSR, intent.rounding_mode, "frm", "frm-write"))
        risk_index = max(
            index for index, item in enumerate(emitted) if "risk" in item.tags
        )
        emitted = (
            prologue
            + emitted[: risk_index + 1]
            + _csr_observer_rows(
                (("frd", FFLAGS_OBSERVER_REGISTER, FFLAGS_CSR, "fobs"),),
                offset=FFLAGS_OBSERVER_OFFSET,
                xlen=_intent_xlen(form, intent),
            )
            + emitted[risk_index + 1 :]
        )
        observed = True
    state = _initial_state(intent, registers, schema=schema)
    if schema is not None and schema.kind == "instruction-memory" and schema.operation == "table-jump":
        # The link sentinel is architectural GPR state.  Its high bits are
        # unobservable on RV32, so project this dedicated sentinel to the
        # concrete XLEN instead of leaving an invalid 64-bit initial value.
        state = _with_state(
            state,
            1,
            INSTRUCTION_MEMORY_LINK_POISON & xlen_mask(_intent_xlen(form, intent)),
            xlen=_intent_xlen(form, intent),
        )
    if stack_seed_values is not None and schema is not None:
        stack_state = list(state)
        stack_mask = xlen_mask(_intent_xlen(form, intent))
        if schema.operation == "push":
            stack_state[2] = (
                LOGICAL_DATA_ADDRESS + STACK_FRAME_OFFSET + stack_adjust
            ) & stack_mask
            for register, value in stack_seed_values.items():
                stack_state[register] = int(value) & stack_mask
        else:
            stack_state[2] = (LOGICAL_DATA_ADDRESS + STACK_FRAME_OFFSET) & stack_mask
            for register, value in stack_seed_values.items():
                stack_state[register] = (STACK_DESTINATION_POISON ^ int(value)) & stack_mask
            if schema.operation == "popretz":
                stack_state[10] = STACK_DESTINATION_POISON & stack_mask
        state = tuple(stack_state)
    if producer_loaded_register is not None:
        seeded = list(state)
        seeded[producer_loaded_register] = PRODUCER_SOURCE_POISON & xlen_mask(
            _intent_xlen(form, intent)
        )
        state = tuple(seeded)
    execution_state = state
    if producer_loaded_register is not None and producer_loaded_value is not None:
        execution_state = _with_state(
            execution_state,
            producer_loaded_register,
            int(producer_loaded_value),
            xlen=_intent_xlen(form, intent),
        )
    # Branch/jump consumers belong to the plan's "gpr-observer" stage; the
    # consumer-shape label only projects the concrete control path
    # (which instructions actually execute) from the declared value.
    if intent.consumer_shape in {"branch-zero-store", "jump-target-store"}:
        state = _with_state(state, consumer_register, 1, xlen=_intent_xlen(form, intent))
        if intent.consumer_shape == "jump-target-store":
            execution_state = _with_state(
                execution_state,
                consumer_register,
                1,
                xlen=_intent_xlen(form, intent),
            )
    risk_pc = 0
    for item in emitted:
        if "risk" in item.tags:
            break
        risk_pc += item.byte_length
    else:
        return None
    control_witness = _control_witness_for_intent(form, schema, intent)
    if control_witness is not None and control_witness.pc_delta_seed:
        # The target-bearing axes seed a source register relative to the
        # risk instruction's own address (for example JALR target = PC+2 for
        # the no-C bit1 misaligned edge).  The witness word already names the
        # register slot; only the concrete value depends on the layout.
        for role, delta in control_witness.pc_delta_seed:
            register = next(
                (
                    register
                    for name, register in registers.items()
                    if role in name and name in operand_groups(form)
                ),
                None,
            )
            if register is None or int(register) == 0:
                return None
            state = _with_state(
                state,
                register,
                int(risk_pc) + int(delta),
                xlen=_intent_xlen(form, intent),
            )

    def declared_control_value() -> int | None:
        return declared_exact_gpr_control_consumer_value(
            form,
            schema,
            operands=intent.operands_map,
            source_state=intent.source_state_map,
            pc=risk_pc,
            rounding_mode=intent.rounding_mode,
        )

    if intent.consumer_shape == "branch-zero-store" and schema is not None:
        # Path projection for the plan's gpr-observer stage: whether the
        # branch falls through the marker is a concrete control-value
        # projection, not a separate semantic path.
        result = declared_control_value()
        if result is None:
            return None
        if schema.kind == "instruction-memory" and schema.operation == "table-jump":
            executed_ids = tuple(executed_ids_override or ())
        else:
            skipped = {"markt"} if int(result) != 0 else set()
            executed_ids = tuple(
                item.instruction_id
                for item in emitted
                if item.instruction_id not in skipped
            )
    elif intent.consumer_shape == "jump-target-store" and schema is not None:
        # Bind the concrete jump-target block selected by the declared value.
        if not {
            "jtt0",
            "jtt1",
        } <= {item.instruction_id for item in emitted}:
            return None
        result = declared_control_value()
        if result is None:
            return None
        emitted, executed_ids = _bind_jump_target_consumer(
            emitted,
            select_high_target=bool(int(result) & 0x10),
        )
        if schema.kind == "instruction-memory" and schema.operation == "table-jump":
            selected = ("jtt1", "jtobs1", "jtret1") if int(result) & 0x10 else ("jtt0", "jtobs0", "jtret0")
            executed_ids = tuple(executed_ids_override or ()) + (
                "jtmask",
                "jthi",
                "jtlo",
                "jtadd",
                "cons",
                *selected,
                "jtskip",
            )
    else:
        executed_ids = executed_ids_override or tuple(item.instruction_id for item in emitted)
    if schema is not None and schema.kind.startswith("control") and not legality:
        bound = _bound_control_flow(
            form,
            emitted,
            execution_state,
            xlen=_intent_xlen(form, intent),
            preserve_target_low_bit=boundary == "jump-target:odd",
        )
        if bound is None:
            return None
        emitted, executed_ids, bound_state = bound
        merged_state = list(state)
        for index, (before, after) in enumerate(zip(execution_state, bound_state)):
            if before != after:
                merged_state[index] = after
        state = tuple(merged_state)
    if _intent_xlen(form, intent) == 32:
        for index, item in enumerate(emitted):
            if item.mnemonic != "sd" or "observable" not in item.tags or "risk" in item.tags:
                continue
            if item.rs1 is None or item.rs2 is None:
                raise UnrealizedState("RV32 observer store lacks register metadata")
            emitted[index] = replace(
                item,
                mnemonic="sw",
                word=encode_instruction(
                    "sw",
                    rs1=int(item.rs1),
                    rs2=int(item.rs2),
                    imm12s=int(item.immediate or 0),
                ),
            )
    code = b"".join(item.word.to_bytes(item.byte_length, "little") for item in emitted)
    offset = 0
    isa_profile = _profile_for_intent(form, intent, emitted)
    instruction_meta = []
    for item in emitted:
        item_form = form_for_mnemonic(item.mnemonic.replace(".", "_"))
        operand_fields = ()
        if item_form is not None:
            decoded_fields = decode_form(item_form, item.word)
            operand_fields = tuple(
                (str(name), int(decoded_fields[name]))
                for name in operand_groups(item_form)
                if name in decoded_fields
            )
        registers = {name: getattr(item, name) for name in ("rd", "rs1", "rs2", "rs3")}
        registers.update(
            {
                f"{name}_domain": xregister_compatibility_slot_domain_for_mnemonic(
                    isa_profile, item.mnemonic, name, registers[name]
                )
                for name in ("rd", "rs1", "rs2", "rs3")
            }
        )
        instruction_meta.append(
            InstructionMeta(
                instruction_id=item.instruction_id,
                pc_offset=offset,
                mnemonic=item.mnemonic,
                byte_offset=offset,
                byte_length=item.byte_length,
                **registers,
                immediate=item.immediate,
                operand_fields=operand_fields,
                tags=item.tags,
            )
        )
        offset += item.byte_length

    # A program that cannot show its result observes only its outcome; claiming
    # a value observer without a sink would be an unbacked coverage claim.
    # Sinks are the observable instructions on the concrete expected path;
    # alternate branch targets are not executed sinks for this testcase.
    executed_id_set = set(executed_ids)
    sink_ids = tuple(
        item.instruction_id
        for item in emitted
        if "observable" in item.tags and item.instruction_id in executed_id_set
    )
    risk_ids = tuple(item.instruction_id for item in emitted if "risk" in item.tags)
    risk_positions = [index for index, item in enumerate(emitted) if "risk" in item.tags]
    sink_positions = [
        index
        for index, item in enumerate(emitted)
        if "observable" in item.tags and item.instruction_id in executed_id_set
    ]
    risk_to_sink_distance = (
        max(1, min(3, sink_positions[0] - risk_positions[-1]))
        if risk_positions and sink_positions and sink_positions[0] > risk_positions[-1]
        else 1
    )
    observed = observed and bool(sink_ids)
    address_fault = legality and (
        "misaligned" in str(intent.boundary_class)
        or str(intent.boundary_class) == "memory-environment:access-fault"
    )
    compare_mask = (
        CompareMask(
            outcome=True,
            checkpoint_pc=False,
            signal=True,
            fault_pc=True,
            fault_address=address_fault,
            extra_state_keys=("trap.cause", "trap.epc", "trap.tval"),
        )
        if legality
        else CompareMask(
            outcome=True,
            checkpoint_pc=True,
            memory_region_ids=(OBSERVABLE_REGION,) if observed else (),
            extra_state_keys=tuple(vector_dynamic_keys),
        )
    )
    if legality:
        observation_kind = "trap-outcome"
        observation_sinks = risk_ids
        observation_distance = 1
        observation_transform = "identity"
        observation_fields = (
            "outcome", "signal", "fault_pc",
            *(("fault_address",) if address_fault else ()),
            "trap.cause", "trap.epc", "trap.tval",
        )
        observation_lossiness = "none"
    elif observed:
        vector_state = schema is not None and schema.state_domain in {"vector", "vector-memory"}
        vector_scalar = (
            schema is not None
            and schema.kind == "vector-data"
            and schema.writeback in {"gpr", "fpr"}
        )
        vector_lossiness = (
            "partial"
            if (
                vector_state
                and not vector_scalar
                and str(intent.boundary_class).startswith(
                    ("vector-vl:", "vector-tail:", "vector-mask:")
                )
                )
            else "none"
        )
        observation_kind = "vector-state" if vector_state and not vector_scalar else "destination-register"
        observation_sinks = sink_ids
        observation_distance = risk_to_sink_distance
        observation_transform = "vector-store" if vector_state and not vector_scalar else "full-width-store"
        observation_fields = (f"memory.{OBSERVABLE_REGION}",)
        observation_lossiness = vector_lossiness
    else:
        observation_kind = "instruction-outcome"
        observation_sinks = risk_ids
        observation_distance = 1
        observation_transform = "identity"
        observation_fields = ("outcome", "checkpoint_pc")
        observation_lossiness = "none"
    observability = ObservabilityContract(
        schema_version="observability-contract-v1",
        risk_value_or_effect=observation_kind,
        sink_instruction_ids=observation_sinks,
        risk_to_sink_distance=observation_distance,
        sink_transform=observation_transform,
        final_observation_fields=observation_fields,
        lossiness=observation_lossiness,
    )
    risk_contracts: list[dict[str, object]] = []
    for risk_item in (item for item in emitted if "risk" in item.tags):
        risk_form = form_for_mnemonic(str(risk_item.mnemonic).replace(".", "_"))
        if risk_form is not None:
            risk_contracts.append(
                {
                    "instruction_id": risk_item.instruction_id,
                    "form": risk_form.mnemonic,
                    "lane": str(intent.lane),
                    "boundary_class": str(intent.boundary_class),
                }
            )
    realized_facts = _shared_realized_facts(
        instruction_meta,
        expected_ids=executed_ids,
        schema=schema,
        lane=str(intent.lane),
    )
    if instruction_memory is not None:
        memory_data = instruction_memory
    elif stack_memory is not None:
        memory_data = stack_memory
    else:
        memory = bytearray(
            VECTOR_TEST_MEMORY_BYTES
            if schema is not None and schema.state_domain in {"vector", "vector-memory"}
            else TEST_MEMORY_BYTES
        )
        for index in range(3):
            key = f"fs{index + 1}"
            if key in seeds:
                seed_offset = FP_SEED_OFFSETS[index]
                mask = (1 << (seed_width * 8)) - 1
                value = int(seeds[key])
                if intent.boundary_class.startswith("fp:nan-box:") and key != nan_box_source:
                    value = (
                        ((1 << (seed_width * 8 - nan_box_width)) - 1) << nan_box_width
                    ) | (value & ((1 << nan_box_width) - 1))
                memory[seed_offset : seed_offset + seed_width] = (
                    value & mask
                ).to_bytes(seed_width, "little")
        memory_value = intent.source_state_map.get("memory")
        if memory_value is not None:
            memory_width = schema.memory_width if schema is not None else 64
            if schema is not None and schema.kind == "vector-memory" and "ei" in str(form.mnemonic):
                memory_width = _vector_element_width(str(intent.boundary_class), form)
            width = max(int(memory_width) // 8, 1)
            mask = (1 << (width * 8)) - 1
            memory[MEMORY_ACCESS_OFFSET : MEMORY_ACCESS_OFFSET + width] = (
                int(memory_value) & mask
            ).to_bytes(width, "little")
        if producer_loaded_role is not None and producer_loaded_width > 0:
            value = intent.source_state_map.get(producer_loaded_role)
            mask = (1 << (producer_loaded_width * 8)) - 1
            memory[PRODUCER_LOAD_OFFSET : PRODUCER_LOAD_OFFSET + producer_loaded_width] = (
                int(value) & mask
            ).to_bytes(producer_loaded_width, "little")
        memory_data = bytes(memory)
    if (
        not legality
        and schema is not None
        and schema.kind in {"memory-load", "memory-store", "atomic-rmw"}
    ):
        risk = next(item for item in instruction_meta if "risk" in item.tags)
        width = max(schema.memory_width // 8, 1)
        address = (
            (int(state[risk.rs1]) + int(risk.immediate or 0))
            & xlen_mask(_intent_xlen(form, intent))
            if risk.rs1 is not None
            else -1
        )
        if (
            risk.rs1 is None
            or address < LOGICAL_DATA_ADDRESS
            or address + width > LOGICAL_DATA_ADDRESS + len(memory_data)
            or address % width
        ):
            return None
    catalog_tags = (f"catalog:{catalog_partition_for_form(form)}",)
    return TestCase(
        testcase_id="rvp6-pending",
        base_program_id="rvp6-pending",
        input_id="input-pending",
        generation_rule_id=generation_rule_for_form(form).rule_id,
        isa_profile=isa_profile,
        code_bytes=code,
        code_address=0,
        entry_checkpoint_address=0,
        exit_checkpoint_address=offset,
        initial_gpr=state,
        initial_memory_regions=(
            MemoryRegion(
                region_id=OBSERVABLE_REGION,
                address=LOGICAL_DATA_ADDRESS,
                data=memory_data,
                permissions="rwx" if instruction_memory is not None else "rw",
            ),
        ),
        compare_mask=compare_mask,
        functional_tags=(
            RVGEN_CANONICAL_TAG,
            f"effect:{schema.kind if schema else 'legality-only'}",
            *catalog_tags,
        ),
        layout_tags=(f"program-shape:{intent.program_shape}",),
        instruction_meta=tuple(instruction_meta),
        block_meta=(
            BlockMeta(block_id="b0", start_offset=0, end_offset=offset, executed_expected=True),
        ),
        dataflow_meta={
            "expected_executed_instruction_ids": executed_ids,
            "risk_instruction_ids": risk_ids,
            "risk_contracts": risk_contracts,
            "realized_facts": realized_facts,
            "generation_owner": generation_owner_contract(
                form,
                table_index=(
                    int(intent.operands_map.get("c_index", 0))
                    if form.mnemonic == "cm_jalt" else None
                ),
            ),
            **_intent_contract_meta(
                intent,
                form=form,
                schema=schema,
                isa_profile=isa_profile,
            ),
            **_base_state_contract_meta(schema, intent),
        },
        observability_contract=observability,
    )

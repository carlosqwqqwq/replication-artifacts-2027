"""Boundary intents: EffectSchema parameters -> concrete obligations.

Every value here is computed from effect parameters (width, signedness,
precision, field width) or from the frozen catalog structure.  No value is
attached to a mnemonic, so a form the author has never seen gets the same
boundary treatment as one they have.
"""

from dataclasses import dataclass, replace
from functools import cache
from itertools import product

from ..riscv_catalog import (
    OFFICIAL_ALL_CATALOG_FORMS,
    OFFICIAL_FIELD_BIT_RANGES,
    OfficialForm,
)
from ..riscv_encoding import (
    SCATTERED_IMMEDIATE_LAYOUT,
    decode_form,
    encode_register_field_value,
    immediate_fields,
    implicit_source_roles,
    is_register_operand_group,
    operand_groups,
    register_domain as operand_register_domain,
    source_roles,
)
from ..spec_definedness import (
    SemanticRealization,
    csr_required_extensions_for_profile,
    enabled_extensions,
    isa_profile_for_extensions,
    profile_enables_form,
    rv32_xregister_compatibility_profile,
    _rv32_xregister_profile,
    required_extension_sets_for_form,
    semantic_realizations_for_form,
    rv32_xregister_writeback_compatibility_profile,
)
from .boundaries import (
    CONTROL_WITNESS_PREFIX,
    IEEE_FORMAT_BY_PRECISION,
    ROUNDING_MODES,
    STACK_ADJ_RV64,
    STACK_RLIST_REGISTERS_RV64,
    declared_exact_gpr_control_consumer_value,
    integer_operation_boundary_pairs,
    projection_writeback_pairs,
    comparison_boundary_pairs,
    control_witness_pairs,
    division_exception_pairs,
    FIXED_FIELD_WITNESS_PREFIX,
    fixed_field_witness_pairs,
    ieee_bit_classes,
    ieee_operation_operand_rows,
    fp_to_fp_rounding_rows,
    integer_boundary_values,
    integer_source_boundary_values,
    atomic_sign_pair_boundaries,
    compressed_codeword_disposition,
    integer_to_fp_boundaries,
    nan_box_carriers,
    natural_alignment_offsets,
    fp_to_int_exact_negative_minimum_bits,
    fp_to_int_exact_positive_limit_bits,
    shift_amount_boundaries,
    vector_state_boundary,
    vector_vtype_codes,
)
from .effects import (
    EffectSchema,
    generation_rule_for_form,
    generation_schema_for_form,
    producer_chain_role,
    register_fields,
    schema_for_realization,
    trap_outcome_kind_for_form,
    rv32_xregister_pair_roles,
    vector_profile_vtype_violation,
    vector_default_state,
    vector_crypto_default_state,
    vector_crypto_overlap_restricted,
    vector_dynamic_state_keys,
    vector_fault_only_first,
    vector_mask_operand,
    vector_register_group_size,
    vector_register_violation,
)
from .obligations import NormativeObligation, normative_obligation_for


def _profile_enables_emitted_form(isa_profile: str, form: OfficialForm) -> bool:
    """Accept generic V helpers under an embedded Zve profile.

    The pinned catalog tags shared vector helpers (``vsetvli``, ``vmv`` and
    vector loads/stores) with ``rv_v`` even when a Zv* form is realizable via
    Zve*.  The profile owner already expands the Zv* risk form to V/Zve
    alternatives; keep the same projection for helper instructions instead of
    upgrading the materialized testcase to full V.
    """
    if profile_enables_form(isa_profile, form):
        return True
    enabled = enabled_extensions(isa_profile)
    if "v" in enabled or not any(token.startswith("zve") for token in enabled):
        return False
    # Only the setup/observer instructions emitted by this materializer are
    # shared by embedded Zve profiles.  Do not turn every catalog V form into
    # a Zve form merely because its source extension is ``rv_v``.
    return form.mnemonic in {
        "vsetvli",
        "vsetivli",
        "vsetvl",
        "vmv_v_i",
        "vse8_v",
        "vse16_v",
        "vse32_v",
        "vse64_v",
        "vs1r_v",
        "vs2r_v",
        "vs4r_v",
        "vs8r_v",
    }


def _csr_required_extensions_for_materialized_profile(
    csr: int,
    isa_profile: str,
) -> frozenset[str]:
    """Keep vector CSR helpers on an embedded Zve profile."""
    required = csr_required_extensions_for_profile(int(csr), isa_profile)
    enabled = enabled_extensions(isa_profile)
    if required == frozenset({"v"}) and "v" not in enabled and any(
        token.startswith("zve") for token in enabled
    ):
        return frozenset()
    return required


LANE_NORMAL = "normal-defined"
LANE_LEGALITY = "legality-expected-trap"
CONSUMER_REGISTER = 8
# The direct-ELF observer frame reserves x2 (sp), x29 (saved base) and x30
# (memory register) before the risk instruction; a GPR slot seeded onto one
# of them cannot be observed, so field axes substitute x27 (same high-index
# boundary character) instead.
_OBSERVER_RESERVED_GPRS = frozenset({2, 29, 30})
_GPR_AXIS_HIGH_SUBSTITUTE = 27
PAIRED_SREG_SELECTOR_ROWS = (
    ("s0-s1", 0, 1),
    ("s1-s0", 1, 0),
    ("s6-s7", 6, 7),
    ("s7-s6", 7, 6),
)
_RV32_SHARED_SHIFT_OPERATIONS = frozenset(
    {"shift-left", "shift-right-logical", "shift-right-arithmetic", "rotate-right"}
)
_RV32_SHARED_BIT_INDEX_OPERATIONS = frozenset(
    {"bit-set", "bit-clear", "bit-invert", "bit-extract"}
)
def rv32_compatibility_extension_suffix(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> str | None:
    """Current minimal x-register compatibility slice, owned by shared structure.

    We only restore the old RV32 transfer-illegal surface where the compatibility
    rule is unambiguous from current structure facts:

    - RV32/shared scalar FP move forms
    - one GPR endpoint and one FPR endpoint
    - the minimal required extension set has a current x-register twin

    This keeps the prototype honest: no synthetic runtime support, no per-bug
    template, and no broad claim that every F-profile form has a current
    x-register twin.
    """
    if (
        schema is None
        or schema.kind != "fp-convert"
        or schema.operation != "move"
        or form.encoding_length_bytes != 4
        or str(form.xlen) not in {"shared", "rv32"}
    ):
        return None
    # The transfer-illegal gate is the complement of the compatibility
    # profile: ``rv32_xregister_compatibility_profile`` excludes FMV transfers
    # because the x-register profiles reuse the F/D/H encodings for computation
    # only, while the gate row must name the profile under which the transfer
    # form itself is illegal.
    profile = _rv32_xregister_profile(form)
    if profile is None:
        return None
    domains = {
        operand_register_domain(form, role)
        for role in register_fields(form)
    }
    if domains != {"gpr", "fpr"}:
        return None
    _head, *tail = profile.split("_", 1)
    return tail[0] if tail else None


@dataclass(frozen=True)
class MaterializationIntent:
    """One obligation: a form, a lane, operand values and structural axes."""

    form: str
    lane: str
    boundary_class: str
    operands: tuple[tuple[str, int], ...] = ()
    register_relation: str = "distinct"
    program_shape: str = "single"
    source_state: tuple[tuple[str, int], ...] = ()
    sibling_state: tuple[tuple[str, int], ...] = ()
    rounding_mode: int | None = None
    destination_seed: int | None = None
    realization_xlen: int = 64
    realization_isa_profile: str = "rv64i"
    realization_extension_tokens: tuple[str, ...] = ()
    realization_register_carrier: str = "gpr"
    realization_privilege_class: str = "user"
    realization_environment_profile: str = "base"
    realization_id: str = ""
    normative_obligation: NormativeObligation | None = None


    @property
    def sequence_length(self) -> int:
        return _PROGRAM_SHAPES[self.program_shape][0]

    @property
    def consumer_shape(self) -> str:
        return _PROGRAM_SHAPES[self.program_shape][1]

    @property
    def extension_gate_disabled(self) -> bool:
        return self.boundary_class.startswith("extension-gate:")

    @property
    def uses_reduced_gpr_domain(self) -> bool:
        return self.boundary_class.startswith("register-domain:") or (
            self.boundary_class.startswith("zcmp-sreg:")
            and self.realization_xlen == 32
            and self.realization_isa_profile.startswith("rv32e")
        )

    @property
    def operands_map(self) -> dict[str, int]:
        return dict(self.operands)

    @property
    def source_state_map(self) -> dict[str, int]:
        return dict(self.source_state)

    @property
    def source_roles(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.source_state)

    def key(self) -> str:
        parts = [self.form, self.lane, self.boundary_class, self.register_relation]
        parts.extend(f"{name}={value}" for name, value in self.operands)
        parts.extend(f"{name}:{value}" for name, value in self.source_state)
        parts.extend(f"sibling:{name}:{value}" for name, value in self.sibling_state)
        if self.rounding_mode is not None:
            parts.append(f"rm={self.rounding_mode}")
        if self.destination_seed is not None:
            parts.append(f"rd0={self.destination_seed:#x}")
        if self.consumer_shape != "store":
            parts.append(f"consumer={self.consumer_shape}")
        parts.append(f"k={self.sequence_length}")
        if self.program_shape != "single":
            parts.append(f"sequence={self.program_shape}")
        parts.append(f"realization=rv{self.realization_xlen}:{self.realization_isa_profile}")
        parts.append(f"carrier={self.realization_register_carrier}")
        if self.realization_environment_profile != "base":
            parts.append(f"environment={self.realization_environment_profile}")
        if self.normative_obligation is not None:
            parts.append(f"obligation={self.normative_obligation.obligation_id}")
        return "|".join(parts)


def _realization_for_intent(
    form: OfficialForm,
    intent: MaterializationIntent,
    realization: SemanticRealization,
) -> SemanticRealization:
    """Align reduced-RV32 register rows with the materialized profile.

    ``materialize`` derives an ``rv32e`` base whenever a concrete
    ``register-domain`` row, or an explicit RV32E ``zcmp-sreg`` row, uses the
    reduced integer register file.  Keep that existing profile rule in the
    intent owner as well, so the normative obligation is built from the same
    realization that admission and the Sail contract later reconstruct.
    """
    if (
        int(realization.xlen) == 32
        and intent.boundary_class.startswith("rv64:xlen-gate:")
    ):
        profile = isa_profile_for_extensions(
            "rv64",
            frozenset(
                token for token in enabled_extensions(str(realization.isa_profile))
                if token != "e"
            ),
        )
        return replace(
            realization,
            xlen=64,
            extension_tokens=tuple(sorted(enabled_extensions(profile))),
            isa_profile=profile,
        )
    if (
        int(realization.xlen) == 64
        and intent.boundary_class.startswith("rv32:xlen-gate:")
    ):
        profile = isa_profile_for_extensions(
            "rv32",
            frozenset(enabled_extensions(str(realization.isa_profile))),
        )
        return replace(
            realization,
            xlen=32,
            extension_tokens=tuple(sorted(enabled_extensions(profile))),
            isa_profile=profile,
        )
    if int(realization.xlen) != 32 or not intent.uses_reduced_gpr_domain:
        return realization
    compatibility_profile = rv32_xregister_compatibility_profile(form)
    source_profile = compatibility_profile or str(realization.isa_profile)
    required = set(enabled_extensions(source_profile))
    required.add("e")
    profile = isa_profile_for_extensions("rv32", frozenset(required))
    carrier = "gpr" if compatibility_profile is not None else realization.register_carrier
    if profile == realization.isa_profile and carrier == realization.register_carrier:
        return realization
    extension_tokens = tuple(sorted(enabled_extensions(profile)))
    return replace(
        realization,
        extension_tokens=extension_tokens,
        isa_profile=profile,
        register_carrier=carrier,
    )


def _apply_realization_to_intent(
    form: OfficialForm,
    schema: EffectSchema | None,
    intent: MaterializationIntent,
    realization: SemanticRealization,
) -> MaterializationIntent:
    realization = _realization_for_intent(form, intent, realization)
    concrete_schema = schema_for_realization(
        form,
        schema,
        xlen=int(realization.xlen),
        register_carrier=str(realization.register_carrier),
    )
    pair_roles = rv32_xregister_pair_roles(
        form,
        concrete_schema,
        xlen=int(realization.xlen),
        register_carrier=str(realization.register_carrier),
    )
    if realization.register_carrier != "gpr":
        realized_source_state = intent.source_state
    else:
        paired = set(pair_roles)
        realized: list[tuple[str, int]] = []
        for name, value in intent.source_state:
            role = ("rs" + name[2:]) if name.startswith("fs") else name
            if role in paired:
                raw = int(value) & ((1 << 64) - 1)
                realized.extend(((role, raw & 0xFFFFFFFF), (role + "+1", raw >> 32)))
            else:
                realized.append((role, value))
        realized_source_state = tuple(realized)
    if (
        realization.register_carrier == "gpr"
        and intent.boundary_class == "source-suppression:rs1-x0"
    ):
        # x0 is architecturally hard-wired zero and cannot be seeded.  The
        # suppressed source (rs1 after carrier realization) therefore leaves
        # source_state: materialization already binds the slot to x0, and the
        # obligation check would otherwise demand a value that can never
        # appear in the GPR file.
        realized_source_state = tuple(
            (role, value)
            for role, value in realized_source_state
            if role != "rs1" and not role.startswith("rs1+")
        )
    obligation = normative_obligation_for(
        form,
        concrete_schema,
        realization,
        lane=intent.lane,
        boundary_class=intent.boundary_class,
    )
    return replace(
        intent,
        source_state=realized_source_state,
        realization_xlen=int(realization.xlen),
        realization_isa_profile=str(realization.isa_profile),
        realization_extension_tokens=tuple(str(token) for token in realization.extension_tokens),
        realization_register_carrier=str(realization.register_carrier),
        realization_privilege_class=str(realization.privilege_class),
        realization_environment_profile=str(realization.environment_profile),
        realization_id=str(realization.realization_id),
        normative_obligation=obligation,
    )


def source_state_for(form: OfficialForm, *values: int) -> tuple[tuple[str, int], ...]:
    """Bind every source role to an explicit concrete value."""
    roles = source_roles(form)
    if not roles:
        return ()
    if len(values) == 1 and len(roles) > 1:
        values *= len(roles)
    if len(values) != len(roles):
        raise ValueError(
            f"source-state arity mismatch for {form.mnemonic}: "
            f"{len(roles)} roles, {len(values)} values"
        )
    return tuple(zip(roles, values))


def field_value_width(name: str) -> int:
    """How many value bits a field carries, scattered or not."""
    if name in SCATTERED_IMMEDIATE_LAYOUT:
        return max(msb for msb, _, _ in SCATTERED_IMMEDIATE_LAYOUT[name]) + 1
    msb, lsb = OFFICIAL_FIELD_BIT_RANGES[name]
    return msb - lsb + 1


def _realizable_immediate_values(name: str) -> tuple[int, ...]:
    """Boundary values whose logical bits the form can actually encode."""
    width = field_value_width(name)
    if width <= 0:
        return ()
    span = 1 << width
    boundaries = {0, 1, span - 1}
    if width > 1:
        boundaries.update((span >> 1, (span >> 1) - 1, span - 2))
    layout = SCATTERED_IMMEDIATE_LAYOUT.get(name)
    if layout is None:
        values = boundaries
    else:
        mask = 0
        for msb, lsb, _instruction_lsb in layout:
            mask |= ((1 << (msb - lsb + 1)) - 1) << lsb
        low = mask & -mask
        values = {0, low, mask, mask ^ low}
        values.update(value & mask for value in boundaries)
    if name.startswith("c_nz"):
        values.discard(0)
    return tuple(sorted(values))


# Consumer shapes decide how a result is used, which is what varies register
# allocation and dataflow inside a translated block.  Control-edge consumers
# are admitted only when the exact post-risk value can already be proved; the
# rest stay out rather than declaring path obligations the current pipeline
# cannot justify end-to-end.
_PROGRAM_SHAPES = {
    **dict.fromkeys(("single", "producer-risk"), (1, "store")),
    "consumer-compare-store": (2, "compare-store"),
    "consumer-unsigned-compare-store": (2, "unsigned-compare-store"),
    "consumer-address-store": (3, "address-store"),
    "consumer-fp-classify-store": (2, "fp-classify-store"),
    "consumer-fp-convert-store": (2, "fp-convert-store"),
    "consumer-branch-zero-store": (2, "branch-zero-store"),
    "consumer-jump-target-store": (6, "jump-target-store"),
}

def _consumer_axis_eligible(form: OfficialForm, schema: EffectSchema | None) -> bool:
    """Single owner of whether a risk result can enter the shared consumers."""
    if schema is None:
        return False
    if schema.writeback == "gpr" and not (
        any(name.startswith("rd") or name == "c_sreg1" for name in register_fields(form))
        or "c_nzimm10" in operand_groups(form)
    ):
        return False
    return (
        schema.kind in {"integer-writeback", "integer-multiply", "integer-shift"}
        and schema.writeback == schema.observer == "gpr"
        or schema.kind == "memory-load"
        and (
            schema.writeback == schema.observer == "gpr"
            or schema.writeback == "fpr"
            and schema.observer in {"rawbits", "rawbits-plus-fflags"}
        )
        or schema.kind in {"fp-arith", "fp-fused", "fp-compare", "fp-convert", "fp-classify"}
        and schema.writeback in {"gpr", "fpr"}
        and schema.observer in {"gpr", "gpr-plus-fflags", "rawbits", "rawbits-plus-fflags"}
    )


def _consumer_axis_intents(
    form: OfficialForm,
    schema: EffectSchema,
    seed_row: MaterializationIntent,
) -> tuple[MaterializationIntent, ...]:
    """The same boundary observed through each way a result can be consumed."""
    if not _consumer_axis_eligible(form, schema):
        return ()
    if (
        seed_row.lane != LANE_NORMAL
        or schema.state_domain in {
            "privileged", "vector", "vector-memory", "csr",
            "reservation", "instruction-memory", "lifecycle",
        }
        or (
            seed_row.register_relation == "rd-eq-x0"
            and schema.writeback == "gpr"
        )
    ):
        return ()
    shapes = []
    if schema.writeback == "gpr" or (
        schema.writeback == "fpr"
        and (schema.precision or 0) <= 64
        # RV32 FPR results wider than one scalar word stay on the raw FP
        # store/classification observers; a GPR compare/address consumer
        # cannot carry the complete value without inventing a lossy projection.
        and not (
            str(form.xlen) == "rv32"
            and (schema.precision or 0) > 24
        )
    ):
        shapes.extend(("compare-store", "unsigned-compare-store", "address-store"))
    if schema.writeback == "fpr" and schema.precision in {11, 24, 53, 113}:
        shapes.append("fp-classify-store")
        if form.xlen != "rv32":
            shapes.append("fp-convert-store")
    exact_control_value = declared_exact_gpr_control_consumer_value(
        form,
        schema,
        operands=dict(seed_row.operands),
        source_state=dict(seed_row.source_state),
        rounding_mode=seed_row.rounding_mode,
    )
    if exact_control_value is not None:
        shapes.extend(("branch-zero-store", "jump-target-store"))
    return tuple(
        replace(
            seed_row,
            program_shape="consumer-" + shape,
        )
        for shape in shapes
    )


_SEMANTIC_SEED_LIMIT = 12
_DESTINATION_SEED_BUDGET = 2
_SEMANTIC_RARE_BOUNDARY_MARKERS = (
    "exception",
    "misaligned",
    "nan",
    "overflow",
    "underflow",
    "saturate",
    "reservation",
)

def _seed_is_rare_boundary(candidate: MaterializationIntent) -> bool:
    boundary = str(candidate.boundary_class).lower()
    return any(marker in boundary for marker in _SEMANTIC_RARE_BOUNDARY_MARKERS)


def _bounded_semantic_seed_rows(
    schema: EffectSchema,
    rows: tuple[MaterializationIntent, ...],
    *,
    preferred: MaterializationIntent | None = None,
    eligible=None,
) -> tuple[MaterializationIntent, ...]:
    """Select a finite semantic basis without forming an axis product.

    The first row remains the canonical semantic seed.  A small budget is
    added per independent descriptor axis: two destination representatives
    (the first projected seed and the all-ones extreme) plus one row each for
    register relation, source projection, ordinary source state, and a rare
    boundary marker.
    """
    normal_rows = tuple(
        row
        for row in rows
        if row.lane == LANE_NORMAL
        and (eligible is None or eligible(row))
    )
    preferred_row = (
        preferred
        if preferred in normal_rows
        else (normal_rows[0] if normal_rows else None)
    )
    if preferred_row is None:
        return ()

    candidates: list[MaterializationIntent] = []

    def add(row: MaterializationIntent | None) -> None:
        if row is None or row in candidates:
            return
        if len(candidates) >= _SEMANTIC_SEED_LIMIT:
            return
        candidates.append(row)

    # A preferred row may carry an already-declared axis (for example an
    # explicit rounding mode).  It is still a valid single representative;
    # derived consumer rows replace that axis with their own consumer edge,
    # while the remaining seed selectors only inspect axis-free rows.
    add(preferred_row)
    canonical = candidates[0]
    destination_added = 0
    for row in (item for item in normal_rows if item.destination_seed is not None):
        if destination_added >= _DESTINATION_SEED_BUDGET:
            break
        before = len(candidates)
        add(row)
        if len(candidates) > before:
            destination_added += 1
    selectors = (
        lambda row: row.register_relation != "distinct",
        lambda row: (
            row.boundary_class != canonical.boundary_class
            and row.boundary_class.startswith("projection:")
        )
        or _is_projection_axis_variant(canonical, row, schema),
        lambda row: (
            row.source_state != canonical.source_state
            and row.destination_seed is None
            and row.register_relation == "distinct"
            and not (
                row.boundary_class != canonical.boundary_class
                and row.boundary_class.startswith("projection:")
            )
            and not _seed_is_rare_boundary(row)
        ),
        lambda row: _seed_is_rare_boundary(row),
        # A dirty-high-half projection variant is its own translation-
        # sensitive representative.  The projection selector above may be
        # claimed by a plain (undirtied) projection boundary class, so this
        # trailing selector guarantees the dirty source-projection row also
        # reaches sequence/consumer/producer seeds when one exists.
        lambda row: _is_projection_axis_variant(canonical, row, schema),
    )
    for selector in selectors:
        add(next((row for row in normal_rows if selector(row)), None))
    return tuple(candidates)


def _consumer_axis_seed_rows(
    form: OfficialForm,
    schema: EffectSchema | None,
    rows: tuple[MaterializationIntent, ...],
) -> tuple[MaterializationIntent, ...]:
    if not _consumer_axis_eligible(form, schema):
        return ()
    normal_rows = tuple(row for row in rows if row.lane == LANE_NORMAL)
    if not normal_rows:
        return ()
    if schema.kind == "memory-load":
        return tuple(row for row in normal_rows if row.boundary_class == "memory-value")
    exact_row = next(
        (
            row
            for row in normal_rows
            if (
                declared_exact_gpr_control_consumer_value(
                    form,
                    schema,
                    operands=row.operands_map,
                    source_state=row.source_state_map,
                    rounding_mode=row.rounding_mode,
                )
                is not None
            )
        ),
        normal_rows[0],
    )
    return _bounded_semantic_seed_rows(schema, rows, preferred=exact_row)


@cache
def _producer_axis_seed_rows(
    form: OfficialForm,
    schema: EffectSchema | None,
    rows: tuple[MaterializationIntent, ...],
) -> tuple[MaterializationIntent, ...]:
    if schema is None:
        return ()
    normal_rows = tuple(row for row in rows if row.lane == LANE_NORMAL)
    if not normal_rows:
        return ()
    chosen = next(
        (
            row
            for row in normal_rows
            if producer_chain_role(
                form,
                schema,
                row.source_roles,
            )
            is not None
        ),
        None,
    )
    if chosen is None:
        return ()

    def producer_seed_eligible(row: MaterializationIntent) -> bool:
        if producer_chain_role(
            form,
            schema,
            row.source_roles,
        ) is None:
            return False
        if row is chosen:
            return True
        if row.destination_seed is not None or row.register_relation != "distinct":
            return True
        if _is_projection_axis_variant(chosen, row, schema):
            return True
        # A scalar GPR exception row has a defined writeback contract and can
        # be carried by the existing producer sink.  FP register-image
        # boundaries (notably NaN boxing) require a raw-FPR observer instead;
        # do not route them through the scalar producer chain.
        return schema.writeback == "gpr" and _seed_is_rare_boundary(row)

    return _bounded_semantic_seed_rows(
        schema,
        rows,
        preferred=chosen,
        eligible=producer_seed_eligible,
    )


def _is_projection_axis_variant(
    base: MaterializationIntent,
    candidate: MaterializationIntent,
    schema: EffectSchema,
) -> bool:
    projection_width = schema.source_width or schema.width
    if (
        projection_width >= 64
        or candidate.boundary_class != base.boundary_class
        or candidate.destination_seed is not None
        or candidate.register_relation != "distinct"
    ):
        return False
    base_state = dict(base.source_state)
    mask = (1 << projection_width) - 1
    for role, value in candidate.source_state:
        if not role.startswith("rs") or role not in base_state:
            continue
        if (int(value) & mask) == (int(base_state[role]) & mask) and int(value) >> projection_width:
            return True
    return False


# The base integer ISA is required by every executable harness.  M is not
# implicit: a multiply/divide form must get a real RV{32,64}I gate control.
ALWAYS_PRESENT_EXTENSIONS = frozenset({"i"})

# RVE halves the integer register file.  The encodings are unchanged, so naming
# x16-x31 is a reserved operand rather than a different instruction.
RVE_REGISTER_LIMIT = 16
_RVE_HELPER_REGISTER = 9


@cache
def _extension_gate_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    realization: SemanticRealization,
) -> tuple[MaterializationIntent, ...]:
    """The same encoding with its own extension switched off must be illegal.

    Derived from ``extension_group``, so the obligation exists for every
    extension in the catalog including ones added later.  It covers both gate
    polarities: the enabled profile is the ordinary corpus, this is the other
    side.
    """
    group = form.extension_group
    if not group or group in ALWAYS_PRESENT_EXTENSIONS:
        return ()
    # ``extension_group`` is a compact catalog label (for example ``c_d`` or
    # ``d_zfa``), not the canonical extension set itself.  System rows carry
    # an environment tag rather than an ISA requirement; manufacturing an
    # ``extension-gate:system`` intent would create an impossible gate and
    # compete with the privileged-state contract.
    if not any(required_extension_sets_for_form(form)):
        return ()
    operands: tuple[tuple[str, int], ...] = ()
    if schema is not None and schema.kind == "stack-transfer":
        operands = (("c_rlist", 4), ("c_spimm", 0))
    elif "c_nzuimm10" in operand_groups(form):
        operands = (("c_nzuimm10", 4),)
    elif "c_nzimm18" in operand_groups(form):
        operands = (("c_nzimm18", 4096),)
    elif "c_nzimm10" in operand_groups(form):
        operands = (("c_nzimm10", 16),)
    group_tokens = tuple(token for token in str(group).split("+") if token)
    enabled = enabled_extensions(str(realization.isa_profile))
    if len(group_tokens) > 1:
        disabled_tokens = {token for token in group_tokens if token in enabled}
        if not disabled_tokens:
            disabled_tokens = set(group_tokens)
    else:
        disabled_tokens = set(group_tokens[0].split("_") if group_tokens else ())
    # A Zfinx/Zdinx/Zhinx realization reuses the scalar FP encoding through
    # the GPR file.  Disabling only the catalog F/D/H extension leaves that
    # compatibility profile enabled, so the supposed gate remains legal.
    # Include every non-base token of the active compatibility profile in the
    # same descriptor-derived gate; FPR realizations retain the original gate.
    if int(realization.xlen) == 32 and str(realization.register_carrier) == "gpr":
        compatibility_profile = rv32_xregister_compatibility_profile(form)
        if compatibility_profile is not None:
            disabled_tokens.update(
                token
                for token in enabled_extensions(compatibility_profile)
                if token not in ALWAYS_PRESENT_EXTENSIONS
            )
    gate = "+".join(sorted(disabled_tokens))
    rows = [
        _base(
            form,
            f"extension-gate:{gate}",
            lane=LANE_LEGALITY,
            operands=operands,
        )
    ]
    rows.extend(_per_form_prerequisite_gate_intents(form, schema, operands))
    return tuple(rows)


def _register_domain_signature(form: OfficialForm) -> tuple[tuple[str, str], ...]:
    fields = register_fields(form)
    implicit = set(implicit_source_roles(form))
    return tuple(
        (role, operand_register_domain(form, role))
        for role in ("rd", "rs1", "rs2", "rs3")
        if role in implicit or any(role in field for field in fields)
    )


@cache
def _semantic_twin_forms(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> tuple[OfficialForm, ...]:
    """32-bit official forms with the same effect schema (uncompressed twins).

    A compressed form's encoding extension (for example Zcb) does not carry
    the semantic extension its uncompressed twin needs: C.MUL executes MUL
    (needs M), C.SEXT.B executes the Zbb sign-extension.  The twin is found
    structurally from the frozen catalog by matching the effect schema, so
    every compressed form gets the same treatment without a per-instruction
    table.
    """
    if (
        int(getattr(form, "encoding_length_bytes", 0)) != 2
        or schema is None
    ):
        return ()
    # The width is compared at the generation (XLEN-independent) level: the
    # concrete realization view narrows it to the profile XLEN, which must
    # not hide the uncompressed twin from the RV32 realization.
    owner_width = int(getattr(generation_schema_for_form(form), "width", 64) or 64)
    key = (
        schema.kind,
        schema.operation,
        owner_width,
        int(getattr(schema, "source_width", 0) or 0),
        bool(getattr(schema, "source_signed", True)),
        _register_domain_signature(form),
    )
    form_xlen = str(getattr(form, "xlen", "shared"))

    def matches(candidate: OfficialForm) -> bool:
        candidate_xlen = str(getattr(candidate, "xlen", "shared"))
        if not (
            candidate_xlen == form_xlen
            or (form_xlen == "shared" and candidate_xlen in {"rv32", "rv64"})
            or (candidate_xlen == "shared" and form_xlen in {"rv32", "rv64"})
        ):
            return False
        candidate_schema = generation_schema_for_form(candidate)
        if form.mnemonic == "c_zext_w" and candidate.mnemonic == "add_uw":
            return candidate_schema is not None and (
                candidate_schema.width == schema.width
                and candidate_schema.source_width == schema.source_width
                and candidate_schema.source_signed == schema.source_signed
            )
        return (
            candidate_schema is not None
            and (
                candidate_schema.kind,
                candidate_schema.operation,
                int(candidate_schema.width or 64),
                int(getattr(candidate_schema, "source_width", 0) or 0),
                bool(getattr(candidate_schema, "source_signed", True)),
                _register_domain_signature(candidate),
            )
            == key
        )

    return tuple(
        candidate
        for candidate in OFFICIAL_ALL_CATALOG_FORMS
        if candidate.encoding_length_bytes == 4 and matches(candidate)
    )


def per_form_prerequisite_extension_sets(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> tuple[frozenset[str], ...]:
    """Semantic prerequisite extension alternatives for one form.

    Derived from the pinned catalog: the uncompressed twin's required
    extension alternatives minus the form's own encoding requirements.  A
    form is legal only while the encoding requirements *and* at least one
    prerequisite alternative are enabled; removing every alternative while
    keeping the encoding profile is the per-form prerequisite negative
    (compressed prerequisite boundary: keep the form's own extension
    requirements instead of borrowing an uncompressed twin).
    """
    twins = _semantic_twin_forms(form, schema)
    if not twins:
        return ()
    own = frozenset().union(*required_extension_sets_for_form(form))
    alternatives: list[frozenset[str]] = []
    for twin in twins:
        for required in required_extension_sets_for_form(twin):
            prerequisite = frozenset(
                token
                for token in required
                if token not in own and token not in ALWAYS_PRESENT_EXTENSIONS
            )
            if prerequisite and prerequisite not in alternatives:
                alternatives.append(prerequisite)
    return tuple(alternatives)


@cache
def _per_form_prerequisite_gate_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    operands: tuple[tuple[str, int], ...],
) -> tuple[MaterializationIntent, ...]:
    """One legality row per form that carries a derived prerequisite.

    The negative witness runs the same encoding under the realization profile
    with every prerequisite alternative removed (the profile keeps the
    encoding extension, so ``extension-gate`` would re-enable it): the form
    must go from legal to illegal.  The pair's legal half is the same word
    under the profile upgraded with one alternative, which the ordinary
    corpus rows own.
    """
    alternatives = per_form_prerequisite_extension_sets(form, schema)
    if not alternatives:
        return ()
    disabled = frozenset().union(*alternatives)
    return (
        _base(
            form,
            f"prerequisite-gate:{'+'.join(sorted(disabled))}",
            lane=LANE_LEGALITY,
            operands=operands,
        ),
    )


def _xlen_gate_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    realization: SemanticRealization,
) -> tuple[MaterializationIntent, ...]:
    """Exercise an encoding under the opposite XLEN profile."""
    if int(realization.xlen) not in {32, 64}:
        return ()
    if str(form.xlen) == "rv32":
        profile = (
            isa_profile_for_extensions(
                "rv64",
                frozenset(
                    token for token in enabled_extensions(str(realization.isa_profile))
                    if token != "e"
                ),
            )
            if int(realization.xlen) == 32
            else str(realization.isa_profile)
        )
        return (_base(
            form,
            "rv64:xlen-gate:rv32-only",
            lane=LANE_LEGALITY,
            realization_xlen=64,
            realization_isa_profile=profile,
            realization_extension_tokens=tuple(sorted(enabled_extensions(profile))),
            realization_register_carrier=str(realization.register_carrier),
            realization_privilege_class=str(realization.privilege_class),
            realization_environment_profile=str(realization.environment_profile),
        ),)
    if any(
        int(item.xlen) == 32
        for item in semantic_realizations_for_form(
            form,
            writeback=None if schema is None else schema.writeback,
        )
    ):
        return ()
    return (_base(form, "rv32:xlen-gate:rv64-only", lane=LANE_LEGALITY),)


def _rv32_compatibility_pair_legality_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    realization: SemanticRealization,
) -> tuple[MaterializationIntent, ...]:
    if int(realization.xlen) != 32 or str(realization.register_carrier) != "gpr":
        return ()
    pair_roles = rv32_xregister_pair_roles(
        form,
        schema,
        xlen=32,
        register_carrier="gpr",
    )
    if not pair_roles:
        return ()
    fields = set(operand_groups(form))
    return tuple(
        _base(
            form,
            f"rv32:compat-odd-paired-gpr-base:{role}",
            lane=LANE_LEGALITY,
            operands=tuple(
                sorted(
                    {
                        **({role: 11} if role in fields else {}),
                        **({"rm": 0} if "rm" in fields else {}),
                    }.items()
                )
            ),
        )
        for role in pair_roles
    )


def _rv32_compatibility_half_sign_writeback_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> tuple[MaterializationIntent, ...]:
    if rv32_xregister_writeback_compatibility_profile(form) is None:
        return ()
    row = {
        "copy-sign": (
            "control",
            (("rd", 8), ("rs1", 6), ("rs2", 7)),
            (("rs1", 0x00003C00), ("rs2", 0x00000000)),
        ),
        "invert-sign": (
            "invert-sign",
            (("rd", 10), ("rs1", 6), ("rs2", 7)),
            (("rs1", 0x00003C00), ("rs2", 0x00000000)),
        ),
        "xor-sign": (
            "xor-sign",
            (("rd", 12), ("rs1", 6), ("rs2", 7)),
            (("rs1", 0x00003C00), ("rs2", 0xFFFF8000)),
        ),
    }.get(None if schema is None else schema.operation)
    if row is None:
        return ()
    suffix, operands, source_state = row
    return (
        _base(
            form,
            f"rv32:compat-half-sign-writeback:{suffix}",
            lane=LANE_NORMAL,
            operands=operands,
            source_state=source_state,
        ),
    )


def _register_domain_intents(
    form: OfficialForm,
    base: MaterializationIntent | None,
    schema: EffectSchema | None = None,
) -> tuple[MaterializationIntent, ...]:
    """Naming a register RVE removed must be illegal, one obligation per slot.

    The boundary is the register number itself: x15 is the last legal one and
    x16 the first reserved one, so both sides come from the field width rather
    than from a list of instructions.
    """
    compatibility_profile = rv32_xregister_compatibility_profile(form)
    if rv32_xregister_pair_roles(
        form,
        schema,
        xlen=32,
        register_carrier="gpr",
    ):
        return ()
    slots = [
        name
        for name in register_fields(form)
        if field_value_width(name) >= 5
        and (
            operand_register_domain(form, name) == "gpr"
            or (
                compatibility_profile is not None
                and operand_register_domain(form, name) == "fpr"
            )
        )
    ]
    # RVE is RV32E.  Applying its x16 boundary to shared/RV64 encodings made a
    # normal RV64 program claim a trap that the selected profile cannot have.
    if not slots or base is None or form.xlen not in {"shared", "rv32"}:
        return ()
    out: list[MaterializationIntent] = []
    register_rows = (
        ((RVE_REGISTER_LIMIT, False), (31, False))
        if compatibility_profile is not None
        else ((RVE_REGISTER_LIMIT - 1, True), (RVE_REGISTER_LIMIT, False), (31, False))
    )
    paired_base = schema is not None and schema.writeback == "paired-gpr"
    for slot in slots:
        for register, legal in register_rows:
            # A wide RV32 result/compare value consumes an aligned pair.  The
            # last RVE register (x15) is encodable but cannot name x15/x16;
            # keep that boundary in the legality lane instead of generating a
            # normal testcase whose helper necessarily leaves the RVE domain.
            paired_slot = slot.startswith(("rd", "rs2")) or slot == "c_rs2"
            if paired_base and paired_slot and register == 31:
                # x31 would combine the RVE-domain violation with an odd pair
                # base, so it cannot be a single-concrete-violation row.
                continue
            row_legal = legal and not (paired_base and paired_slot and int(register) & 1)
            operands = dict(base.operands) if base is not None else {}
            # Register-domain rows vary only the register slot.  Keep the
            # concrete operands that make the base semantic row realizable
            # (notably the CSR address) instead of constructing an otherwise
            # under-specified instruction.
            operands[slot] = encode_register_field_value(slot, register)
            # A compressed HINT immediate can hide this independent register
            # boundary.  Pick the first encodable non-HINT value from the
            # descriptor so the RVE row remains an actual reserved-register
            # witness instead of being filtered as an unmodelled codeword.
            if compressed_codeword_disposition(form, schema, operands) == "hint":
                for field in immediate_fields(form):
                    replacement = next(
                        (
                            value
                            for value in _realizable_immediate_values(field)
                            if compressed_codeword_disposition(
                                form,
                                schema,
                                {**operands, field: value},
                            )
                            != "hint"
                        ),
                        None,
                    )
                    if replacement is not None:
                        operands[field] = int(replacement)
                        break
            for name, value in tuple(operands.items()):
                if name == slot or int(value) < RVE_REGISTER_LIMIT:
                    continue
                if is_register_operand_group(form, name) and operand_register_domain(form, name) == "gpr":
                    operands[name] = encode_register_field_value(
                        name,
                        _RVE_HELPER_REGISTER,
                    )
            out.append(
                _base(
                    form,
                    f"register-domain:{slot}:x{register}",
                    lane=LANE_NORMAL if row_legal else LANE_LEGALITY,
                    operands=tuple(sorted(operands.items())),
                    source_state=base.source_state if row_legal and base is not None else (),
                    rounding_mode=base.rounding_mode if row_legal and base is not None else None,
                )
            )
    return tuple(out)


def _compressed_legality_intents(
    form: OfficialForm,
    realization: SemanticRealization,
) -> tuple[MaterializationIntent, ...]:
    """Compressed-only legality boundaries that current direct lane explicitly owns."""
    if form.encoding_length_bytes != 2:
        return ()
    if form.mnemonic == "c_addi4spn":
        return (
            _base(
                form,
                "legality:compressed-addi4spn-zero",
                lane=LANE_LEGALITY,
                operands=(("c_nzuimm10", 0), ("rd_p", 1)),
            ),
        )
    if form.mnemonic == "c_add":
        return (
            _base(
                form,
                "compressed-hint:rd-x0",
                register_relation="rd-eq-x0",
                operands=(("rd_rs1_n0", 0), ("c_rs2_n0", 6)),
            ),
        )
    if form.mnemonic == "c_lui":
        return (
            _base(
                form,
                "compressed-hint:rd-x0",
                register_relation="rd-eq-x0",
                operands=(("c_nzimm18", 4096), ("rd_n2", 0)),
            ),
            _base(
                form,
                "legality:compressed-lui-zero",
                lane=LANE_LEGALITY,
                operands=(("c_nzimm18", 0), ("rd_n2", 0)),
            ),
        )
    if not {"c_sreg1", "c_sreg2"} <= set(operand_groups(form)):
        return ()
    rows = [
        _base(
            form,
            "zcmp-sreg:s0-s1",
            operands=(("c_sreg1", encode_register_field_value("c_sreg1", 8)),),
        )
    ]
    # Zcmp's s-register selectors name x8..x19.  Only RV32E removes x16+
    # from the architectural GPR file; on RV64 (and ordinary RV32I) these
    # selectors are legal and must not be emitted as trap obligations.
    if int(realization.xlen) == 32 and str(realization.isa_profile).startswith("rv32e"):
        rows.extend(
            (
                _base(
                    form,
                    "zcmp-sreg:c_sreg1:s2",
                    lane=LANE_LEGALITY,
                    operands=(("c_sreg1", encode_register_field_value("c_sreg1", 18)),),
                ),
                _base(
                    form,
                    "zcmp-sreg:c_sreg2:s3",
                    lane=LANE_LEGALITY,
                    operands=(("c_sreg2", encode_register_field_value("c_sreg2", 19)),),
                ),
            )
        )
    return tuple(rows)


def _apply_vector_profile_lanes(
    form: OfficialForm,
    schema: EffectSchema | None,
    rows: tuple[MaterializationIntent, ...],
    isa_profile: str,
) -> tuple[MaterializationIntent, ...]:
    if schema is None or schema.state_domain not in {"vector", "vector-memory"}:
        return rows
    projected: list[MaterializationIntent] = []
    for row in rows:
        violation = vector_profile_vtype_violation(
            isa_profile,
            form,
            schema,
            row.boundary_class,
        )
        projected.append(
            replace(row, lane=LANE_LEGALITY)
            if row.lane == LANE_NORMAL and violation is not None
            else row
        )
    return tuple(projected)


@cache
def boundary_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    realization: SemanticRealization | None = None,
) -> tuple[MaterializationIntent, ...]:
    """Every obligation this form owns, derived from structure and effect.

    The semantic rows come from the effect kind; the sequence, encoding and
    profile obligations ride on the common path so every kind -- including ones
    added later -- owes them without anyone wiring them up.
    """
    if not isinstance(form, OfficialForm):
        raise ValueError("boundary_intents form must be an OfficialForm")
    # The full opcode inventory contains unratified P rows that share bytes
    # with a selected Z* semantic alias.  GenerationRule already canonicalizes
    # that owner; boundary_intents must use the same form so direct callers do
    # not manufacture a second extension/provenance lane.
    form = generation_rule_for_form(form).form
    if schema is not None and not isinstance(schema, EffectSchema):
        raise ValueError("boundary_intents schema must be an EffectSchema or None")
    owner_schema = generation_schema_for_form(form)
    if schema != owner_schema:
        raise ValueError("boundary_intents schema must match the form generation schema")
    if realization is not None and not isinstance(realization, SemanticRealization):
        raise ValueError("boundary_intents realization must be a SemanticRealization or None")
    active_realization = realization or semantic_realizations_for_form(
        form,
        writeback=None if schema is None else schema.writeback,
    )[-1]
    expected_xlen = f"rv{int(active_realization.xlen)}"
    effective_form = (
        form if str(form.xlen) == expected_xlen else replace(form, xlen=expected_xlen)
    )
    concrete_schema = schema_for_realization(
        effective_form,
        schema,
        xlen=int(active_realization.xlen),
        register_carrier=str(active_realization.register_carrier),
    )
    if schema is None:
        trap_kind = trap_outcome_kind_for_form(effective_form)
        semantic = () if trap_kind is None else (
            _base(effective_form, f"trap-outcome:{trap_kind}", lane=LANE_LEGALITY),
        )
    else:
        semantic = _apply_vector_profile_lanes(
            effective_form,
            concrete_schema,
            _semantic_kind_intents(
                effective_form,
                concrete_schema,
                active_realization.isa_profile,
            ),
            active_realization.isa_profile,
        )
    consumers = tuple(
        consumer
        for seed in _consumer_axis_seed_rows(effective_form, concrete_schema, semantic)
        for consumer in _consumer_axis_intents(effective_form, concrete_schema, seed)
    )
    producer_seeds = _producer_axis_seed_rows(
        effective_form,
        concrete_schema,
        semantic,
    )
    if producer_seeds:
        producers = tuple(
            replace(row, program_shape="producer-risk")
            for row in producer_seeds
        )
    else:
        producers = ()
    compatibility_suffix = rv32_compatibility_extension_suffix(effective_form, schema)
    rows = (
        semantic
        + consumers
        + producers
        + tuple(
            _base(
                effective_form,
                f"{FIXED_FIELD_WITNESS_PREFIX}{witness.boundary.field}",
                lane=LANE_LEGALITY,
            )
            for witness in fixed_field_witness_pairs(effective_form)
        )
        + _control_witness_intents(effective_form, concrete_schema)
        + _extension_gate_intents(effective_form, schema, active_realization)
        + _xlen_gate_intents(form, schema, active_realization)
        + (() if compatibility_suffix is None else (
            _base(
                effective_form,
                f"rv32:compat-extension-gate:{compatibility_suffix}",
                lane=LANE_LEGALITY,
            ),
        ))
        + _rv32_compatibility_pair_legality_intents(
            effective_form,
            concrete_schema,
            active_realization,
        )
        + _rv32_compatibility_half_sign_writeback_intents(effective_form, schema)
        + _compressed_legality_intents(effective_form, active_realization)
        + _register_domain_intents(
            effective_form,
            next((row for row in semantic if row.lane == LANE_NORMAL), None),
            concrete_schema,
        )
    )
    unique: list[MaterializationIntent] = []
    seen: set[str] = set()
    for intent in rows:
        if (
            compressed_codeword_disposition(
                effective_form, concrete_schema, intent.operands_map
            ) == "hint"
            and not intent.boundary_class.startswith("compressed-hint:")
        ):
            continue
        if (
            active_realization.register_carrier == "gpr"
            and active_realization.xlen == 32
            and concrete_schema is not None
            and rv32_xregister_pair_roles(
                effective_form,
                concrete_schema,
                xlen=32,
                register_carrier="gpr",
            )
            and intent.register_relation != "distinct"
        ):
            # The fixed fflags observer uses a scalar scratch register; an
            # aliased pair destination would clobber one half before the pair
            # sink.  Keep unsupported pair relations out of the value-normal
            # slice instead of claiming a lossy observer.
            continue
        # NaN boxing is an FPR register-image property.  X-register floating
        # extensions (Zfinx/Zdinx/Zhinx) carry the value in GPRs, so those
        # realizations must not inherit an FPR-only boundary row.
        if (
            active_realization.register_carrier == "gpr"
            and intent.boundary_class.startswith("fp:nan-box:")
        ):
            continue
        realized = _apply_realization_to_intent(
            effective_form,
            schema,
            intent,
            active_realization,
        )
        key = realized.key()
        if key not in seen:
            seen.add(key)
            unique.append(realized)
    return tuple(unique)


@cache
def _control_witness_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> tuple[MaterializationIntent, ...]:
    """Legality rows for trap-side control/definedness witnesses.

    Axes whose trap side is already owned by an existing boundary class
    (reserved shamt, read-only CSR writes) or whose edge has no trap side
    (odd-target bit0, link/source alias, far jump, FENCE.I second-fetch
    shape) stay in their owning rows; this function adds only the derived
    rows that would otherwise be missing.  Every added row carries the
    witness label so materialize can recover the raw word from the same
    descriptor family as fixed-field witnesses.
    """
    if schema is None:
        return ()
    rows: list[MaterializationIntent] = []
    for witness in control_witness_pairs(form, schema):
        axis = witness.boundary.axis
        if axis in {"reserved-shamt", "jalr-target:bit0", "link-source-alias", "far-jump", "fence-i-refetch"} and not (
            axis == "reserved-shamt" and form.xlen == "rv64"
        ):
            # Owned by existing rows: shift-amount-reserved /
            # rv32:shift-amount-reserved, jump-target:odd,
            # jump-link:source-alias, and the instruction-memory:code-patch
            # store->FENCE.I->second-fetch route.
            continue
        if axis == "csr-warl":
            # Read-only writes are owned by the per-class csr:*:write rows.
            # WARL readback is different: keep the raw witness, and derive
            # its concrete CSR operand from the same official word so the
            # existing CSR observer can materialize the post-write state.
            if witness.boundary.label() != "csr-warl:reserved-frm":
                continue
            try:
                decoded = decode_form(form, witness.illegal_word)
                csr = decoded.get("csr")
            except (KeyError, TypeError, ValueError):
                continue
            if csr is None:
                continue
            operands = (("csr", int(csr)),)
        else:
            operands = ()
        lane = LANE_NORMAL if witness.expected_outcome == "normal" else LANE_LEGALITY
        rows.append(
            _base(
                form,
                f"{CONTROL_WITNESS_PREFIX}{witness.boundary.label()}",
                lane=lane,
                operands=operands,
                source_state=witness.seed,
                register_relation="rd-eq-x0" if axis == "csr-warl" else "distinct",
            )
        )
    return tuple(rows)


@cache
def _semantic_kind_intents(
    form: OfficialForm,
    schema: EffectSchema | None,
    isa_profile: str,
) -> tuple[MaterializationIntent, ...]:
    if schema is None:
        return ()
    kind = schema.kind
    if kind in {"integer-writeback", "integer-multiply"}:
        return _integer_intents(form, schema)
    if kind == "integer-shift":
        return _shift_intents(form, schema)
    if kind in {"fp-arith", "fp-fused", "fp-compare"}:
        return _fp_intents(form, schema)
    if kind == "fp-convert":
        return _fp_convert_intents(form, schema)
    if kind in {"memory-load", "memory-store", "atomic-rmw"}:
        return _memory_intents(form, schema)
    if kind == "stack-transfer":
        return _stack_intents(form, schema)
    if kind == "paired-register-transfer":
        return _paired_register_intents(form, schema, isa_profile)
    if kind == "control-compare":
        return _control_intents(form, schema)
    if kind == "control-jump":
        return _control_jump_intents(form, schema)
    if kind == "csr-access":
        return _csr_intents(form, schema, isa_profile)
    if kind == "instruction-memory":
        return _instruction_memory_intents(form, schema)
    if kind == "fp-classify":
        return _fp_classify_intents(form, schema)
    if kind == "fence":
        return _fence_intents(form)
    if kind == "cache-block":
        # CBO has no value writeback.  A single structural outcome row is the
        # useful contract; permission/cache-line state remains an execution
        # concern and is never guessed here.
        return (_base(form, "cache-block-outcome"),)
    if kind == "may-be-operation" and schema.operation.startswith("vector-config"):
        return _vector_config_intents(form)
    if kind in {"vector-data", "vector-memory"}:
        return _vector_data_intents(form, schema, isa_profile)
    if kind in {"encoding-only", "privileged-system"}:
        return _encoding_intents(form, schema)
    if kind == "may-be-operation":
        return _may_be_operation_intents(form, schema)
    # Effects such as FENCE deliberately have no value writeback, but still
    # need one original direct testcase whose outcome is observable.  Leaving
    # them with legality-only neighbours silently removed the legal half of
    # their ISA contract.
    return (_base(form, "effect-outcome"),)


@cache


def _paired_register_intents(
    form: OfficialForm,
    schema: EffectSchema,
    isa_profile: str,
) -> tuple[MaterializationIntent, ...]:
    selector_rows = PAIRED_SREG_SELECTOR_ROWS
    if str(form.xlen) == "rv32" and str(isa_profile).startswith("rv32e"):
        selector_rows = selector_rows[:2]
    out = [
        _base(
            form,
            f"paired-register:{name}",
            operands=(("c_sreg1", left), ("c_sreg2", right)),
        )
        for name, left, right in selector_rows
    ]
    if schema.operation == "move-a01-to-s":
        out.append(
            _base(
                form,
                "paired-register:equal-sreg-reserved",
                lane=LANE_LEGALITY,
                operands=(("c_sreg1", 0), ("c_sreg2", 0)),
            )
        )
    return tuple(out)


def _base(form: OfficialForm, boundary_class: str, **kwargs) -> MaterializationIntent:
    lane = str(kwargs.pop("lane", LANE_NORMAL))
    if "register_relation" not in kwargs and "c_nzimm10" in operand_groups(form):
        kwargs["register_relation"] = "rd-eq-rs1"
    return MaterializationIntent(
        form=form.mnemonic,
        lane=lane,
        boundary_class=boundary_class,
        **kwargs,
    )


def _integer_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out: list[MaterializationIntent] = []
    register_relation = "rd-eq-rs1" if form.mnemonic == "c_addi16sp" else "distinct"
    immediates = immediate_fields(form)
    # Ask the form for its source roles rather than reading them off the field
    # prefix: a compressed form names one combined slot, so a unary compressed
    # operation has a source that no prefix test can see.
    source_slots = source_roles(form)
    if immediates:
        name = immediates[0]
        encoded_values = _realizable_immediate_values(name)
        source_values = integer_source_boundary_values(schema.width, schema.source_width)
        for value in source_values:
            for encoded in encoded_values:
                selector_reserved = name == "rnum" and encoded > 10
                out.append(
                    _base(
                        form,
                        "selector-reserved" if selector_reserved else "integer-immediate",
                        lane=LANE_LEGALITY if selector_reserved else LANE_NORMAL,
                        operands=((name, encoded),),
                        source_state=source_state_for(form, value),
                        register_relation=register_relation,
                    )
                )
            break  # immediate sweep is independent of the source sweep
        source_encoded = (
            max(value for value in encoded_values if value <= 10)
            if name == "rnum"
            else encoded_values[-1]
        )
        for value in source_values:
            out.append(
                _base(
                    form,
                    "integer-source",
                    operands=((name, source_encoded),),
                    source_state=source_state_for(form, value),
                    register_relation=register_relation,
                )
            )
    if schema.width < 64 and len(source_slots) >= 2:
        # The projection boundary: results either side of the bit the narrower
        # effect must sign-extend from.
        for relation, rows in projection_writeback_pairs(schema.width).items():
            for left, right in rows:
                out.append(
                    _base(
                        form,
                        f"projection:{relation}",
                        source_state=source_state_for(form, left, right),
                    )
                )
    if len(source_slots) >= 2:
        for left, right in integer_operation_boundary_pairs(
            width=schema.width, operation=schema.operation, signed=schema.signed
        ):
            out.append(
                _base(
                    form,
                    "integer-operand-pair",
                    source_state=(("rs1", left), ("rs2", right)),
                )
            )
        # DIV/REM have architectural zero-divisor and signed-overflow rules;
        # MUL/MULH do not.  They share an opcode family but not an exception
        # formula, so the schema operation is the dispatch boundary.
        if schema.operation in {"div", "divu", "rem", "remu"}:
            for left, right in division_exception_pairs(schema.width, schema.signed):
                out.append(
                    _base(
                        form,
                        "integer-exception-pair",
                        source_state=source_state_for(form, left, right),
                    )
                )
    elif not immediates and source_slots:
        for value in integer_source_boundary_values(schema.width, schema.source_width):
            out.append(_base(form, "integer-source", source_state=source_state_for(form, value)))
    return tuple(out) + _structural_axis_intents(form, schema, out)


def _shift_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out: list[MaterializationIntent] = []
    immediates = immediate_fields(form)
    xlen_limit = 32 if str(form.xlen) == "rv32" else 64
    effective_width = min(int(schema.width or xlen_limit), xlen_limit)
    amounts = shift_amount_boundaries(effective_width)
    interior = (effective_width // 2,) if effective_width >= 4 else ()
    if immediates:
        name = immediates[0]
        width = field_value_width(name)
        vendor_high_amount = (
            form.xlen == "rv32" and form.extension_group in {"xtheadbb", "xtheadbs"}
        )
        for amount in tuple(dict.fromkeys(amounts + interior)):
            legal = amount < effective_width or vendor_high_amount
            if amount >= (1 << width):
                continue
            boundary = (
                "vendor-shift-high"
                if vendor_high_amount and amount >= effective_width
                else "shift-amount" if legal else "shift-amount-reserved"
            )
            out.append(
                _base(
                    form,
                    boundary,
                    lane=LANE_NORMAL if legal else LANE_LEGALITY,
                    operands=((name, amount),),
                    source_state=source_state_for(form, 1 << min(amount, effective_width - 1)),
                )
            )
        if form.xlen == "rv32" and name in {"shamtd", "c_imm6"}:
            if schema.operation in _RV32_SHARED_SHIFT_OPERATIONS:
                reserved_amount = 32 if name == "c_imm6" else 63
                rv32_rows = (
                    ("rv32:shift-amount-control", 31, 31, LANE_NORMAL),
                    ("rv32:shift-amount-reserved", reserved_amount, 31, LANE_LEGALITY),
                )
            elif schema.operation in _RV32_SHARED_BIT_INDEX_OPERATIONS:
                rv32_rows = (
                    ("rv32:shift-amount-control", 1, 1, LANE_NORMAL),
                    ("rv32:shift-amount-reserved", 33, 1, LANE_LEGALITY),
                )
            else:
                rv32_rows = ()
            for boundary_class, amount, seed_amount, lane in rv32_rows:
                out.append(
                    _base(
                        form,
                        boundary_class,
                        lane=lane,
                        operands=((name, amount),),
                        source_state=source_state_for(form, 1 << seed_amount),
                    )
                )
    else:
        for amount in tuple(dict.fromkeys(amounts + interior)):
            out.append(
                _base(
                    form,
                    "shift-amount",
                    source_state=source_state_for(form, 1 << min(amount, effective_width - 1), amount),
                )
            )
    if schema.source_width and schema.source_width < schema.width:
        for value in integer_source_boundary_values(schema.width, schema.source_width):
            out.append(
                _base(
                    form,
                    "shift-source-projection",
                    source_state=source_state_for(form, value),
                )
            )
    return tuple(out) + _structural_axis_intents(form, schema, out)


def _fp_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out: list[MaterializationIntent] = []
    precision = schema.precision or 24
    sources = max(
        sum(1 for name in operand_groups(form) if name.startswith("rs") and "imm" not in name),
        1,
    )
    modes = ROUNDING_MODES if "rm" in operand_groups(form) else (("none", None),)
    for name, carriers in ieee_operation_operand_rows(precision, sources, schema.operation):
        state = tuple((f"fs{index + 1}", int(bits)) for index, bits in enumerate(carriers))
        for _mode_name, encoded_mode in modes:
            frm_mode = 0 if encoded_mode == 7 else encoded_mode
            out.append(
                _base(
                    form,
                    f"fp:{name}",
                    operands=(("rm", encoded_mode),) if encoded_mode is not None else (),
                    source_state=state,
                    rounding_mode=frm_mode,
                )
            )
    for name, bits in nan_box_carriers(precision):
        boxed = (
            ((1 << IEEE_FORMAT_BY_PRECISION[precision][0]) - 1)
            << IEEE_FORMAT_BY_PRECISION[precision][0]
        ) | (int(bits) & ((1 << IEEE_FORMAT_BY_PRECISION[precision][0]) - 1))
        for index in range(sources):
            state = []
            for slot in range(sources):
                value = int(bits) if slot == index else boxed
                state.append((f"fs{slot + 1}", value))
            out.append(
                _base(
                    form,
                    f"fp:nan-box:{name}:fs{index + 1}",
                    source_state=tuple(state),
                )
            )
    return tuple(out) + _structural_axis_intents(form, schema, out)


def _fp_convert_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    """State rows target the register file the form actually reads.

    A convert reads one integer register or one floating point register, never
    both.  Emitting both groups regardless of direction had left half of every
    convert form's obligations seeding a register the instruction ignores --
    the program changed, the ledger stayed clean, and the conversion ran on
    zero instead of the declared boundary.
    """
    out: list[MaterializationIntent] = []

    def rounding_contexts() -> tuple[tuple[int | None, int | None], ...]:
        if "rm" not in operand_groups(form):
            return ((None, None),)
        if schema.operation != "fp-to-int":
            return tuple(
                (encoded_mode, 0 if encoded_mode == 7 else encoded_mode)
                for _mode_name, encoded_mode in ROUNDING_MODES
            )
        concrete_modes = tuple(
            encoded_mode
            for _mode_name, encoded_mode in ROUNDING_MODES
            if encoded_mode is not None and encoded_mode != 7
        )
        return tuple(
            (encoded_mode, frm_mode)
            for encoded_mode, frm_mode in (
                *((mode, mode) for mode in concrete_modes),
                *((7, mode) for mode in concrete_modes),
            )
        )

    if schema.operation == "literal":
        selector_field = schema.immediate_field or "rs1"
        return tuple(
            _base(
                form,
                f"literal-selector:{selector}",
                operands=((selector_field, selector),),
            )
            for selector in range(32)
        )
    if schema.operation in {"move", "move-pair"}:
        if schema.operation == "move-pair":
            return tuple(
                _base(
                    form,
                    "move:raw-gpr-pair",
                    source_state=(("rs1", left), ("rs2", right)),
                )
                for left, right in integer_operation_boundary_pairs(
                    width=schema.source_width or 64,
                    operation="generic",
                    signed=False,
                )
            )
        if operand_register_domain(form, "rs1") == "gpr":
            values = integer_boundary_values(schema.source_width or 32)
            return tuple(
                _base(
                    form,
                    f"move:raw-gpr:{index}",
                    source_state=(("rs1", int(value)),),
                )
                for index, value in enumerate(values)
            )
        source_precision = schema.source_precision or schema.precision or 24
        return tuple(
            _base(
                form,
                f"move:raw-fpr:{name}",
                source_state=(("fs1", int(value)),),
            )
            for name, value in ieee_bit_classes(source_precision).items()
        )
    if operand_register_domain(form, "rs1") == "gpr":
        precision = schema.precision or 24
        source_width = schema.source_width or schema.width
        concrete_xlen = 32 if str(form.xlen) == "rv32" else 64
        for row in integer_to_fp_boundaries(source_width, schema.source_signed, precision):
            # ``integer_to_fp_boundaries`` exposes a canonical 64-bit carrier
            # (including sign extension for a signed 32-bit source).  The
            # carrier must be projected to the concrete GPR width before it
            # becomes initial state: RV32 accepts the 32-bit bit pattern, not
            # the already sign-extended RV64 spelling.  A source wider than
            # concrete XLEN has no representable GPR carrier and is omitted
            # rather than silently truncating a semantic boundary.
            if source_width > concrete_xlen:
                continue
            encoded_source = int(row.encoded_u64) & ((1 << concrete_xlen) - 1)
            for encoded_mode, frm_mode in rounding_contexts():
                out.append(
                    _base(
                        form,
                        f"convert:{row.rounding_relation}:{row.boundary_class}",
                        operands=(("rm", encoded_mode),) if encoded_mode is not None else (),
                        source_state=(("rs1", encoded_source),),
                        rounding_mode=frm_mode,
                    )
                )
        return tuple(out)
    source_precision = schema.source_precision or schema.precision or 24
    bits = ieee_bit_classes(source_precision)
    rounding_rows = rounding_contexts()

    def add_fpr_rows(rows) -> None:
        for boundary_class, value in rows:
            for encoded_mode, frm_mode in rounding_rows:
                out.append(
                    _base(
                        form,
                        boundary_class,
                        operands=(("rm", encoded_mode),) if encoded_mode is not None else (),
                        source_state=(("fs1", int(value)),),
                        rounding_mode=frm_mode,
                    )
                )

    add_fpr_rows(
        (f"convert:saturate:{name}", bits[name])
        for name in (
            "pos-inf", "neg-inf", "qnan", "qnan-payload",
            "qnan-payload-negative", "snan", "snan-negative", "pos-zero",
            "neg-zero", "half-below", "half", "half-above", "one-and-half",
            "max-finite", "min-normal", "max-subnormal", "min-subnormal",
        )
    )
    if schema.operation == "fp-to-int":
        if schema.signed:
            exact_minimum = fp_to_int_exact_negative_minimum_bits(
                source_precision,
                schema.width,
            )
            if exact_minimum is not None:
                add_fpr_rows((("convert:exact-negative-minimum", exact_minimum),))
        exact_limit = fp_to_int_exact_positive_limit_bits(
            source_precision,
            schema.width,
            signed=schema.signed,
        )
        if exact_limit is not None:
            add_fpr_rows((("convert:positive-clip-at-limit", exact_limit),))
    if schema.operation == "fp-to-int":
        sign_bit = int(bits["neg-zero"])
        add_fpr_rows(
            (f"convert:rounding-active:{name}", sign_bit | int(bits[source]))
            for name, source in (
                ("negative-half-below", "half-below"),
                ("negative-half", "half"),
                ("negative-half-above", "half-above"),
                ("negative-one-and-half", "one-and-half"),
            )
        )
    if schema.operation == "fp-to-fp":
        add_fpr_rows(
            (f"convert:{name}", value)
            for name, value in fp_to_fp_rounding_rows(source_precision, schema.precision)
        )
    return tuple(out)


def _distinct_integer_value(value: int, width: int) -> int:
    mask = (1 << width) - 1
    alternate = (int(value) ^ 1) & mask
    return 1 if alternate == int(value) else alternate


def _cas_source_state(
    schema: EffectSchema,
    *,
    compare_value: int | None,
    swap_value: int,
    memory_value: int,
    concrete_xlen: int = 64,
) -> tuple[tuple[str, int], ...]:
    state: list[tuple[str, int]] = [("memory", int(memory_value))]
    if schema.writeback == "paired-gpr":
        if concrete_xlen not in {32, 64}:
            raise ValueError("paired atomic concrete XLEN must be 32 or 64")
        half_mask = (1 << concrete_xlen) - 1
        if compare_value is not None:
            state.extend(
                (
                    ("rd", int(compare_value) & half_mask),
                    ("rd+1", (int(compare_value) >> concrete_xlen) & half_mask),
                )
            )
        state.extend(
            (
                ("rs2", int(swap_value) & half_mask),
                ("rs2+1", (int(swap_value) >> concrete_xlen) & half_mask),
            )
        )
        return tuple(state)
    if compare_value is not None:
        state.append(("rd", int(compare_value)))
    state.append(("rs2", int(swap_value)))
    return tuple(state)


def _atomic_cas_intents(
    form: OfficialForm, schema: EffectSchema
) -> tuple[MaterializationIntent, ...]:
    width = schema.memory_width or schema.width
    concrete_xlen = (
        int(schema.width) // 2
        if schema.writeback == "paired-gpr"
        else (32 if str(form.xlen) == "rv32" else 64)
    )
    values = (
        integer_boundary_values(width)
        if width <= 64
        else tuple(
            dict.fromkeys(
                (
                    0,
                    1,
                    (1 << 64) - 1,
                    1 << 64,
                    (1 << 64) + 1,
                    (1 << (width - 1)) - 1,
                    1 << (width - 1),
                    (1 << (width - 1)) + 1,
                    (1 << width) - 1,
                )
            )
        )
    )
    out: list[MaterializationIntent] = []
    for compare in values:
        swap = _distinct_integer_value(compare, width)
        out.append(
            _base(
                form,
                "cas-compare:match",
                source_state=_cas_source_state(
                    schema,
                    compare_value=compare,
                    swap_value=swap,
                    memory_value=compare,
                    concrete_xlen=concrete_xlen,
                ),
            )
        )
        out.append(
            _base(
                form,
                "cas-compare:mismatch",
                source_state=_cas_source_state(
                    schema,
                    compare_value=_distinct_integer_value(compare, width),
                    swap_value=swap,
                    memory_value=compare,
                    concrete_xlen=concrete_xlen,
                ),
            )
        )
    for swap in values:
        out.append(
            _base(
                form,
                "cas-swap:match",
                source_state=_cas_source_state(
                    schema,
                    compare_value=0,
                    swap_value=swap,
                    memory_value=0,
                    concrete_xlen=concrete_xlen,
                ),
            )
        )
    out.append(
        _base(
            form,
            "cas-zero-compare-discard",
            register_relation="rd-eq-x0",
            source_state=_cas_source_state(
                schema,
                compare_value=None,
                swap_value=1,
                memory_value=0,
                concrete_xlen=concrete_xlen,
            ),
        )
    )
    if schema.writeback == "paired-gpr":
        out.extend(
            (
                _base(
                    form,
                    "paired-register-base:rd-odd",
                    lane=LANE_LEGALITY,
                    operands=(("rd", 11), ("rs2", 12)),
                ),
                _base(
                    form,
                    "paired-register-base:rs2-odd",
                    lane=LANE_LEGALITY,
                    operands=(("rd", 10), ("rs2", 13)),
                ),
            )
        )
    return tuple(out)


def _memory_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out: list[MaterializationIntent] = []
    width_bytes = max(schema.memory_width // 8, 1)
    if schema.operation == "cas":
        out.extend(_atomic_cas_intents(form, schema))
    else:
        values = (
            tuple(ieee_bit_classes(schema.precision).values())
            if schema.precision
            else integer_boundary_values(schema.memory_width or schema.width)
        )
        stored = tuple(name for name in register_fields(form) if name.startswith("rs2") or name == "c_rs2")
        if stored:
            # The operand an atomic or store contributes decides the update, so it
            # needs the same boundary sweep the destination gets.  The role names
            # the register file the form reads: a floating point store takes its
            # value from an f-register, so seeding the x-file would never reach it.
            stored_role = "fs2" if operand_register_domain(form, "rs2") == "fpr" else "rs2"
            for value in values:
                out.append(
                    _base(
                        form,
                        "memory-operand",
                        source_state=((stored_role, value), ("memory", 1)) if "rs2" in source_roles(form) else (("memory", value),),
                    )
                )
        if schema.kind != "memory-store":
            for value in values:
                out.append(_base(form, "memory-value", source_state=(("memory", value),)))
    if out:
        base = out[0]
        for field in immediate_fields(form):
            for encoded in _realizable_immediate_values(field):
                operands = dict(base.operands)
                operands[field] = int(encoded)
                out.append(
                    replace(
                        base,
                        boundary_class=f"memory-immediate:{field}={int(encoded)}",
                        operands=tuple(sorted(operands.items())),
                    )
                )
        if (
            schema.kind == "atomic-rmw"
            and schema.operation in {"min", "max"}
            and schema.signed
            and "rs2" in source_roles(form)
        ):
            existing_states = {tuple(item.source_state) for item in out}
            for memory_value, source_value in atomic_sign_pair_boundaries(
                schema.memory_width or schema.width
            ):
                pair_state = (("rs2", source_value), ("memory", memory_value))
                if pair_state in existing_states:
                    continue
                out.append(
                    _base(
                        form,
                        "atomic-sign-pair",
                        source_state=pair_state,
                    )
                )
                out.append(
                    _base(
                        form,
                        "atomic-sign-pair:rd-eq-rs1",
                        register_relation="rd-eq-rs1",
                        source_state=pair_state,
                    )
                )
                existing_states.add(pair_state)
    for offset in natural_alignment_offsets(width_bytes):
        aligned = offset % width_bytes == 0
        lane = LANE_NORMAL if aligned else LANE_LEGALITY
        out.append(
            _base(
                form,
                "alignment-natural" if aligned else "alignment-misaligned",
                lane=lane,
                source_state=(("address-offset", offset),),
            )
        )
    ordering_fields = tuple(name for name in ("aq", "rl") if name in operand_groups(form))
    if schema.kind == "atomic-rmw" and ordering_fields:
        source_state = [("memory", 1)]
        if "rs2" in source_roles(form):
            source_state.insert(0, ("rs2", 1))
        if schema.writeback == "paired-gpr":
            source_state.append(("rs2+1", 0))
        for encoded in range(1 << len(ordering_fields)):
            ordering = tuple(
                (name, (encoded >> index) & 1)
                for index, name in enumerate(ordering_fields)
            )
            out.append(
                _base(
                    form,
                    "atomic-ordering",
                    operands=ordering,
                    source_state=tuple(source_state),
                )
            )
    fields = set(operand_groups(form))
    if schema.kind == "memory-load" and schema.writeback == "gpr" and {"rd", "rs1"} <= fields:
        out.append(
            _base(
                form,
                "destination-suppression:rd-x0",
                register_relation="rd-eq-x0",
                source_state=(("memory", 1),),
            )
        )
        out.append(
            _base(
                form,
                "memory-environment:access-fault",
                lane=LANE_LEGALITY,
                register_relation="rd-eq-x0",
                source_state=(("address-offset", -65),),
            )
        )
    if (
        schema.writeback in {"gpr", "paired-gpr"}
        and {"rd", "rs1"} <= fields
        and schema.kind in {"memory-load", "atomic-rmw"}
    ):
        alias_state: list[tuple[str, int]] = [("memory", 1)]
        if "rs2" in source_roles(form):
            alias_state.insert(0, ("rs2", 2))
        if schema.writeback == "paired-gpr":
            alias_state.append(("rs2+1", 0))
        out.append(
            _base(
                form,
                "address-destination-alias",
                register_relation="rd-eq-rs1",
                source_state=tuple(alias_state),
            )
        )
        aligned = next(
            (
                row
                for row in out
                if row.lane == LANE_NORMAL
                and row.boundary_class == "alignment-natural"
            ),
            None,
        )
        if aligned is not None:
            existing_roles = {name for name, _value in aligned.source_state}
            out.append(
                replace(
                    aligned,
                    register_relation="rd-eq-rs1",
                    source_state=aligned.source_state
                    + tuple(
                        item for item in alias_state
                        if item[0] not in existing_roles
                    ),
                )
            )
    return tuple(out)


def _stack_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out = [
        _base(
            form,
            "stack-rlist-reserved",
            lane=LANE_LEGALITY,
            operands=(("c_rlist", rlist), ("c_spimm", 0)),
        )
        for rlist in range(4)
    ]
    out.extend(
        _base(
            form,
            f"stack:{schema.operation}",
            operands=(("c_rlist", rlist), ("c_spimm", spimm)),
        )
        for rlist in STACK_RLIST_REGISTERS_RV64
        for spimm in range(len(STACK_ADJ_RV64[rlist]))
    )
    return tuple(out)


def _control_intents(form: OfficialForm, schema: EffectSchema) -> tuple[MaterializationIntent, ...]:
    out: list[MaterializationIntent] = []
    width = schema.width if schema.width in (32, 64) else 64
    relations = {
        "beq": ("equal", "not-equal"),
        "bne": ("not-equal", "equal"),
        "blt": ("less", "not-less"),
        "bge": ("not-less", "less"),
        "bltu": ("less", "not-less"),
        "bgeu": ("not-less", "less"),
    }.get(schema.operation, ("less", "not-less", "equal", "not-equal"))
    unary = len(source_roles(form)) == 1
    mask = (1 << width) - 1
    immediate = schema.immediate_field
    forward_immediate = (
        _control_flow_immediate_value(form, schema)
        if immediate in immediate_fields(form)
        else None
    )
    operands = (
        ()
        if immediate is None or forward_immediate is None
        else ((immediate, forward_immediate),)
    )
    for relation in relations:
        if unary:
            values = {"equal": (0,), "not-equal": (1, mask)}.get(relation, ())
            pairs = ((value,) for value in values)
        else:
            pairs = iter(
                comparison_boundary_pairs(
                    width=width, signed=schema.signed, relation=relation
                ) + _control_extreme_pairs(
                    width=width, signed=schema.signed, relation=relation
                )
            )
        for values in pairs:
            out.append(
                _base(
                    form,
                    f"control:{relation}",
                    operands=operands,
                    source_state=source_state_for(form, *values),
                )
            )
    return tuple(out)


def _control_extreme_pairs(
    *,
    width: int,
    signed: bool,
    relation: str,
) -> tuple[tuple[int, int], ...]:
    """Sign/zero-boundary source pairs that keep the declared relation true."""
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    if relation == "equal":
        return ((sign, sign), (sign - 1, sign - 1))
    if relation == "not-equal":
        return ((sign, sign - 1), (sign - 1, sign))
    if relation == "less":
        return (
            ((sign, sign - 1), (sign, 0))
            if signed
            else ((sign - 1, sign), (0, mask))
        )
    if relation == "not-less":
        return (
            ((sign - 1, sign), (0, sign))
            if signed
            else ((sign, sign - 1), (mask, 0))
        )
    return ()


def _control_flow_immediate_value(
    form: OfficialForm,
    schema: EffectSchema,
) -> int | None:
    """Return the layout owner's legal displacement in encoded field units."""
    field = schema.immediate_field
    if field is None or field not in immediate_fields(form):
        return None
    # The shared control layout lands a forward edge immediately after the risk
    # (plus a four-byte marker for branches).  Encode that concrete offset
    # through the catalog field width instead of leaving the descriptor field
    # implicit or attaching a mnemonic-specific displacement.
    displacement = form.encoding_length_bytes + (
        4 if schema.kind == "control-compare" else 0
    )
    return int(displacement) & ((1 << field_value_width(field)) - 1)


def _control_jump_intents(form, schema):
    """Only keep jump axes the current layout binder preserves concretely."""
    immediate = schema.immediate_field
    forward_immediate = (
        _control_flow_immediate_value(form, schema)
        if immediate in immediate_fields(form)
        else None
    )
    out = [
        _base(
            form,
            "jump-target:layout-bound",
            operands=()
            if immediate is None or forward_immediate is None
            else ((immediate, forward_immediate),),
        )
    ]
    base_state = ()
    if immediate == "imm12":
        for low_bit in (0, 1):
            # JALR clears bit 0 of the computed target.  The layout binder
            # rewrites the displacement but preserves this declared low-bit
            # axis, so it is a real semantic obligation rather than a dropped
            # pre-layout placeholder.
            out.append(
                _base(
                    form,
                    "jump-target:" + ("odd" if low_bit else "even"),
                    operands=((immediate, int(forward_immediate or 0) | low_bit),),
                    source_state=base_state,
                )
            )
    if (
        immediate is None
        and int(form.encoding_length_bytes) == 2
        and schema.operation in {"jr", "jalr"}
    ):
        out.append(_base(form, "jump-target:odd", source_state=base_state))
    fields = register_fields(form)
    has_explicit_destination = any(name.startswith("rd") for name in fields)
    has_implicit_link = schema.writeback == "gpr" and not has_explicit_destination
    relations = (
        ("rd-eq-x0", "distinct")
        if has_explicit_destination
        else ("distinct",)
        if has_implicit_link
        else ()
    )
    for relation in relations:
        out.append(
            _base(
                form,
                "jump-link",
                operands=((immediate, int(forward_immediate or 0)),)
                if immediate and forward_immediate is not None
                else (),
                source_state=base_state,
                register_relation=relation,
            )
        )
    # A link-form JALR reads its source before writing the link register.  The
    # compressed encoding has no explicit ``rd`` field, so the implicit link
    # register must be represented by the source slot itself; ordinary JALR
    # has the same descriptor-backed alias when its explicit rd/rs1 slots are
    # both present.  Keep this as one structural row rather than a mnemonic
    # selector.  The concrete x1 spelling is encoded through the catalog field
    # width, and the explicit source/destination spelling makes the
    # source-before-write edge observable without adding a producer tail or a
    # public generator parameter.
    source_field = next(
        (
            name
            for name in operand_groups(form)
            if is_register_operand_group(form, name) and "rs1" in name
        ),
        None,
    )
    # The compressed link destination is implicit; ordinary JALR's explicit
    # rd/rs1 alias remains outside this seed so existing scalar golden rows do
    # not acquire a second control contract unexpectedly.
    alias_link = bool(
        schema.writeback == "gpr"
        and source_field is not None
        and not has_explicit_destination
    )
    if alias_link:
        out.append(
            _base(
                form,
                "jump-link:source-alias",
                operands=((source_field, encode_register_field_value(source_field, 1)),),
                register_relation="rd-eq-rs1",
            )
        )
    return tuple(out) + _structural_axis_intents(form, schema, out)


# CSR address classes, chosen from the frozen directory rather than written by
# hand: the address bits themselves give writability and privilege, so a
# representative per class covers the whole space without enumerating it.
CSR_CLASS_ORDER = (
    "user-fp",
    "user-counter",
    "user-counter-time",
    "user-hpmcounter",
    "user-other",
    "supervisor",
    "hypervisor",
    "machine",
    "read-only-machine",
    "high-half",
    "unallocated",
)

_CSR_WITNESS_ADDRESSES = {
    "senvcfg": 0x10A,
    "sip": 0x144,
    "jvt": 0x017,
    "ssp": 0x011,
    "menvcfg": 0x30A,
    "medeleg": 0x302,
    "mepc": 0x341,
    "misa": 0x301,
    "mtvec": 0x305,
    "mnstatus": 0x744,
    "henvcfg": 0x60A,
    "mstatush": 0x310,
}


def _csr_class(address: int, name: str) -> str:
    from ..riscv_csrs import (
        csr_is_read_only,
        csr_is_rv32_only,
        csr_minimum_privilege,
    )

    if name is None:
        return "unallocated"
    # Catalog names are quoted (``"mstatush"``), so checking the raw value
    # misses every high-half CSR outside the counter bank.  Normalize once and
    # require the RV32-only address bank before applying the ``*h`` suffix;
    # ordinary names such as ``sscratch``/``mscratch`` and the non-high-half
    # ``mnscratch`` CSR are deliberately excluded.
    normalized_name = str(name).strip('"')
    if (
        0xC80 <= address <= 0xCBF
        or (
            csr_is_rv32_only(address)
            and normalized_name.endswith("h")
            and normalized_name != "mnscratch"
        )
    ):
        return "high-half"
    privilege = csr_minimum_privilege(address)
    if privilege == 0:
        if address <= 0x003:
            return "user-fp"
        if 0xC00 <= address <= 0xC02:
            return "user-counter"
        if 0xC03 <= address <= 0xC1F:
            return "user-hpmcounter"
        return "user-other"
    if privilege == 1:
        return "supervisor"
    if privilege == 2:
        return "hypervisor"
    return "read-only-machine" if csr_is_read_only(address) else "machine"


def csr_class_representatives(csr_table: dict[int, str]) -> tuple[tuple[str, int], ...]:
    """One current representative per CSR access class.

    The representative is a static address-dispatch witness, not a promise
    that every execution environment implements the CSR.  Keeping
    ``user-other`` in the directory is important: its address-specific
    dispatch/WARL behavior is exactly the boundary this lane must retain.
    Dynamic availability remains fail-closed at the state observer boundary.
    """
    chosen: dict[str, int] = {}
    for address, name in sorted(csr_table.items()):
        group = _csr_class(address, name)
        chosen.setdefault(group, address)
    if "user-fp" in chosen:
        chosen["user-fp"] = 0x001
    if "high-half" in chosen:
        # Prefer the user-visible counter high-half when present.  Lower
        # address ``*h`` CSRs (for example ``sieh``) are supervisor/machine
        # accesses; using one as the sole representative would erase the
        # legal RV32 high-half read edge behind a privilege trap row.
        chosen["high-half"] = next(
            (
                address
                for address, name in sorted(csr_table.items())
                if 0xC80 <= address <= 0xCBF
                and _csr_class(address, name) == "high-half"
            ),
            chosen["high-half"],
        )
    chosen.setdefault("unallocated", 0x7C0)
    rows = [
        (group, chosen[group])
        for group in CSR_CLASS_ORDER
        if group in chosen
    ]
    if 0xC01 in csr_table:
        rows.append(("user-counter-time", 0xC01))
    return tuple(rows)


def _csr_intents(form, schema, isa_profile):
    """Every CSR form against a representative of each access class."""
    from ..riscv_csrs import csr_is_read_only, csr_minimum_privilege
    from ..riscv_csrs import OFFICIAL_CSRS, OFFICIAL_RV32_ONLY_CSRS

    out = []
    groups = operand_groups(form)
    immediate_form = "zimm5" in groups
    xlen = 32 if str(form.xlen) == "rv32" else 64
    full_mask = (1 << xlen) - 1
    value_boundaries = (0, 1, 0x1F, 0x15) if immediate_form else (
        0,
        1,
        full_mask,
        full_mask // 3,
    )
    value_names = {
        0: "zero",
        1: "one",
        value_boundaries[2]: "all-ones",
        value_boundaries[3]: "alternating",
    }
    # The high-half CSR bank is architecturally present only on RV32.  Keep
    # those addresses in the RV32 realization so the address-dispatch edge is
    # generated, but never let the same representative leak into RV64.
    csr_catalog = dict(OFFICIAL_CSRS)
    if str(form.xlen) == "rv32":
        csr_catalog.update(OFFICIAL_RV32_ONLY_CSRS)
    representatives = list(csr_class_representatives(csr_catalog))
    selected = {address for _name, address in representatives}
    representatives.extend(
        (name, address)
        for name, address in _CSR_WITNESS_ADDRESSES.items()
        if address in csr_catalog and address not in selected
        and (name != "mstatush" or xlen == 32)
    )
    for name, address in representatives:
        for argument in value_boundaries:
            # CSRRS/CSRRC (and their immediate forms) suppress the write when
            # the source is zero.  CSRRW/CSRRWI still write zero, so their zero
            # boundary must remain a write/legality witness.
            pure_read = argument == 0 and schema.operation in {
                "set",
                "clear",
                "set-immediate",
                "clear-immediate",
            }
            illegal = (not pure_read and csr_is_read_only(address)) or csr_minimum_privilege(address) > 0
            operands = {"csr": address}
            if immediate_form:
                operands["zimm5"] = argument
            elif pure_read:
                operands["rs1"] = 0
            boundary = "csr:" + name + (":read" if pure_read else ":write")
            if not pure_read:
                boundary += ":value-" + value_names[argument]
            intent = MaterializationIntent(
                form=form.mnemonic,
                lane=LANE_LEGALITY if illegal else LANE_NORMAL,
                boundary_class=boundary,
                operands=tuple(sorted(operands.items())),
                source_state=() if immediate_form or pure_read else source_state_for(form, argument),
            )
            out.append(intent)
            if (
                name == "user-counter-time"
                and not immediate_form
                and schema.operation in {"set", "clear"}
                and argument == 1
            ):
                out.append(
                    MaterializationIntent(
                        form=form.mnemonic,
                        lane=LANE_LEGALITY,
                        boundary_class=f"csr:{name}:write:source-zero",
                        operands=tuple(sorted(operands.items())),
                        source_state=source_state_for(form, 0),
                    )
                )
            missing = csr_required_extensions_for_profile(address, isa_profile) - enabled_extensions(isa_profile)
            if missing and not illegal:
                out.append(
                    replace(
                        intent,
                        lane=LANE_LEGALITY,
                        boundary_class=f"extension-gate:{'+'.join(sorted(missing))}",
                    )
                )
    return tuple(out)


def _fence_intents(form: OfficialForm) -> tuple[MaterializationIntent, ...]:
    """Ordering-set boundaries stay in the current outcome-only model."""
    out = [_base(form, "effect-outcome")]
    for pred, succ in ((0x0, 0x0), (0x1, 0x1), (0x1, 0x8), (0x8, 0x1), (0xF, 0xF)):
        out.append(
            _base(
                form,
                f"fence:pred{pred:x}-succ{succ:x}",
                operands=(("pred", pred), ("succ", succ)),
            )
        )
    for fm in _realizable_immediate_values("fm"):
        out.append(
            _base(
                form,
                f"fence:fm={fm:x}",
                operands=(("fm", fm), ("pred", 0xF), ("succ", 0xF)),
            )
        )
    return tuple(out)


def _instruction_memory_intents(
    form: OfficialForm,
    schema: EffectSchema,
) -> tuple[MaterializationIntent, ...]:
    if schema.operation == "fence-fetch":
        return (
            _base(
                form,
                "instruction-memory:code-patch",
                operands=(("imm12", 0), ("rs1", 0), ("rd", 0)),
            ),
        )
    if schema.operation == "table-jump":
        return tuple(
            _base(
                form,
                "instruction-memory:" + boundary,
                operands=(("c_index", index),),
            )
            for boundary, index in (
                ("jt-index-low", 0),
                ("jt-index-high", 31),
                ("jalt-index-low", 32),
                ("jalt-index-high", 255),
            )
        )
    return (_base(form, "instruction-memory"),)


def _may_be_operation_intents(
    form: OfficialForm, schema: EffectSchema
) -> tuple[MaterializationIntent, ...]:
    selectors = immediate_fields(form)
    rows = [_base(form, schema.operation)]
    if selectors:
        rows = [
            _base(
                form,
                schema.operation,
                operands=tuple(zip(selectors, values)),
            )
            for values in product(
                *(range(1 << field_value_width(name)) for name in selectors)
            )
        ]
    if schema.operation == "write-zero":
        return tuple(rows) + _structural_axis_intents(form, schema, rows)
    return tuple(rows)


def _encoding_intents(
    form: OfficialForm, schema: EffectSchema
) -> tuple[MaterializationIntent, ...]:
    """Bounded encoding-neighborhood rows for forms without a value model.

    Every operand slot comes from the frozen opcode descriptor.  We vary one
    slot at a time over zero/one/high values and keep all other slots explicit;
    this exercises rare encodings without multiplying a cartesian template or
    claiming a scalar value effect for a vector/system form.
    """
    groups = operand_groups(form)
    if not groups:
        return (
            _base(
                form,
                "privileged-system:fixed"
                if schema.kind == "privileged-system"
                else "encoding:fixed",
            ),
        )
    baseline = {name: 0 for name in groups}
    prefix = "privileged-system" if schema.kind == "privileged-system" else "encoding"
    rows: list[MaterializationIntent] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for name in groups:
        width = field_value_width(name)
        if width <= 0:
            continue
        span = 1 << width
        values = _realizable_immediate_values(name)
        for value in values:
            encoded = int(value) & (span - 1)
            operands = dict(baseline)
            operands[name] = encoded
            row = tuple((key, int(operands[key])) for key in groups)
            if row in seen:
                continue
            seen.add(row)
            reserved = name == "rnum" and encoded > 10
            source_state = ()
            if form.mnemonic == "bnei":
                operands["rs1"] = encoded if name == "rs1" else 1
                compare = operands["imm5"] or -1
                source_state = (("rs1", compare & ((1 << (32 if form.xlen == "rv32" else 64)) - 1)),)
                row = tuple((key, int(operands[key])) for key in groups)
            rows.append(
                _base(
                    form,
                    "selector-reserved" if reserved else f"{prefix}:{name}:{encoded}",
                    lane=LANE_LEGALITY if reserved else LANE_NORMAL,
                    operands=row,
                    source_state=source_state,
                )
            )
    return tuple(rows)


def _vector_data_intents(
    form: OfficialForm, schema: EffectSchema, isa_profile: str
) -> tuple[MaterializationIntent, ...]:
    """Bounded V state edges shared by arithmetic, mask and memory forms.

    The opcode descriptor supplies every field.  One field changes per row;
    the state axis (mask, tail, VL and address/alignment) is named explicitly
    so a future vector observer can replay the same rows without adding a
    mnemonic-specific generator.
    """
    groups = operand_groups(form)
    mask_operand = vector_mask_operand(form, schema)
    baseline = {
        name: (
            20
            if schema.kind == "vector-memory" and name == "rs1"
            else 1
            if schema.kind == "vector-memory" and name == "rs2"
            else 1
            if name == "vd"
            else 1
            if name == "rd"
            else 1
            if name == "vm"
            else 2
            if name == "vs1"
            else 3
            if name == "vs2"
            else 4
            if name == "vs3"
            else 0
        )
        for name in groups
    }
    baseline_boundary = "vector-memory:baseline" if schema.kind == "vector-memory" else "vector:baseline"

    def vector_source_state(
        boundary: str, operands: dict[str, int]
    ) -> tuple[tuple[str, int], ...]:
        state: list[tuple[str, int]] = []
        if schema.kind == "vector-data" and "rs1" in groups:
            if operand_register_domain(form, "rs1") == "fpr":
                edge = vector_state_boundary(boundary)
                default = vector_default_state(form)
                sew = (
                    edge.sew
                    if edge is not None and edge.sew is not None
                    else default[0]
                    if default is not None
                    else 32
                )
                state.append(("fs1", {
                    16: 0xFFFFFFFFFFFF3C00,
                    32: 0xFFFFFFFF3F800000,
                    64: 0x3FF0000000000000,
                }.get(int(sew), 0xFFFFFFFF3F800000)))
            else:
                if operands.get("rs1") != 0:
                    state.append(("rs1", 1))
        if (
            schema.kind == "vector-memory"
            and "rs2" in groups
            and operands.get("rs2") != 0
            and operands.get("rs2") != operands.get("rs1")
        ):
            state.append(("rs2", max(1, int(schema.memory_width or 8) // 8)))
        return tuple(state)

    def aligned_register(value: int, group: int, boundary: str) -> int:
        suffix = boundary.rsplit("-m", 1)[-1] if "-m" in boundary else ""
        active_lmul = int(suffix) if suffix.isdigit() else 1
        return ((int(value) + max(group, active_lmul) - 1) // max(group, active_lmul)) * max(group, active_lmul)

    def place_vector_registers(operands: dict[str, int], boundary: str) -> bool:
        cursor = 1
        for name in ("vd", "vs1", "vs2", "vs3"):
            if name not in groups:
                continue
            group = vector_register_group_size(form, schema, boundary, name)
            cursor = aligned_register(cursor, group, boundary)
            if cursor + group > 32:
                return False
            operands[name] = cursor
            cursor += group
        return True

    if schema.kind == "vector-data":
        place_vector_registers(baseline, baseline_boundary)
    else:
        for name in groups:
            if name not in {"vd", "vs1", "vs2", "vs3"}:
                continue
            group = vector_register_group_size(form, schema, baseline_boundary, name)
            aligned = aligned_register(baseline[name], group, baseline_boundary)
            baseline[name] = aligned if aligned + group <= 32 else 0
    rows: list[MaterializationIntent] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for name in groups:
        width = field_value_width(name)
        if width <= 0:
            continue
        span = 1 << width
        values = (0, 1, span - 1, span - 2 if span > 1 else 0)
        if name == "rd" and "vd" not in groups:
            # x0 suppresses the scalar result and has no current value sink;
            # keep this lane execution-ready only for writable rd.
            values = (1, 2, span - 1, span - 2 if span > 2 else 1)
        if name in {"rs1", "rs2"} or (name == "rd" and "vd" not in groups):
            # GPR slots must avoid the direct-ELF observer frame registers
            # (x2/x29/x30); x27 keeps the high-index boundary character.
            values = tuple(
                _GPR_AXIS_HIGH_SUBSTITUTE
                if value in _OBSERVER_RESERVED_GPRS
                else value
                for value in values
            )
        for value in values:
            operands = dict(baseline)
            operands[name] = int(value) & (span - 1)
            row = tuple((key, int(operands[key])) for key in groups)
            if row in seen:
                continue
            seen.add(row)
            axis = "vector-field"
            if name == "vm":
                axis = "vector-mask"
            elif name in {"vd", "vs1", "vs2", "vs3"}:
                axis = "vector-register"
            elif name in {"nf", "rs1", "rs2"} and schema.kind == "vector-memory":
                axis = "vector-memory-address"
            rows.append(
                _base(
                    form,
                    f"{axis}:{name}:{int(value) & (span - 1)}",
                    operands=row,
                    lane=(
                        LANE_LEGALITY
                        if vector_register_violation(
                            form,
                            schema,
                            f"{axis}:{name}:{int(value) & (span - 1)}",
                            dict(row),
                            isa_profile,
                        )
                        else LANE_NORMAL
                    ),
                )
            )
    # Keep the V-1.0 legal widening/narrowing overlap as a relation axis.
    # The legality helper already distinguishes the permitted high/low part;
    # generate one concrete row for it instead of hard-coding instruction
    # families or leaving overlap coverage trap-only.
    if "vd" in groups:
        boundary = "vector-register-overlap"
        destination = int(baseline["vd"])
        destination_group = vector_register_group_size(
            form, schema, boundary, "vd"
        )
        destination_set = set(
            range(destination, destination + destination_group)
        )
        if (
            schema.kind == "vector-data"
            and schema.operation == "vector-unary:funct6-14"
            and vector_mask_operand(form, schema) == "v0"
            and "vs2" in groups
        ):
            state_boundary = "vector-vtype:sew64-m8"
            state_destination_group = vector_register_group_size(
                form, schema, state_boundary, "vd"
            )
            if state_destination_group > 1:
                operands = dict(baseline)
                operands.update({"vd": 8, "vs2": 15, "vm": 1})
                row = tuple((key, int(operands[key])) for key in groups)
                if vector_register_violation(
                    form, schema, state_boundary, dict(row), isa_profile
                ) == "vector-register-overlap":
                    rows.append(
                        _base(
                            form,
                            state_boundary,
                            operands=row,
                            lane=LANE_LEGALITY,
                        )
                    )
        state_boundary = "vector-vtype:sew32-m1"
        mask = vector_mask_operand(form, schema)
        if (
            schema.kind == "vector-data"
            and "vd" in groups
            and mask == "v0"
            and schema.operation not in {
                "vector-to-scalar:funct6-10",
                "vector-unary:funct6-14",
            }
        ):
            data_source = next(
                (name for name in ("vs2", "vs1", "vs3") if name in groups),
                None,
            )
            if data_source is not None:
                operands = dict(baseline)
                operands.update({"vd": 16, data_source: 0, "vm": 0})
                row = tuple((key, int(operands[key])) for key in groups)
                if vector_register_violation(
                    form, schema, state_boundary, dict(row), isa_profile
                ) == "vector-source-eew-overlap":
                    rows.append(
                        _base(
                            form,
                            state_boundary,
                            operands=row,
                            lane=LANE_LEGALITY,
                        )
                    )
        if (
            schema.kind == "vector-data"
            and "vd" in groups
            and mask == "vs1"
            and schema.operation == "mask-or-carry:funct6-17"
            and "vs2" in groups
        ):
            operands = dict(baseline)
            operands.update({"vd": 16, "vs1": 2, "vs2": 2})
            row = tuple((key, int(operands[key])) for key in groups)
            if vector_register_violation(
                form, schema, state_boundary, dict(row), isa_profile
            ) == "vector-source-eew-overlap":
                rows.append(
                    _base(
                        form,
                        state_boundary,
                        operands=row,
                        lane=LANE_LEGALITY,
                    )
                )
        source_names = [
            name for name in ("vs1", "vs2", "vs3") if name in groups
        ]
        if "vd" in groups and len(source_names) > 1:
            source_groups = [
                vector_register_group_size(
                    form, schema, "vector-vtype:sew32-m1", name
                )
                for name in source_names
            ]
            if len(set(source_groups)) > 1:
                state_boundary = "vector-vtype:sew32-m1"
                operands = dict(baseline)
                destination_group = vector_register_group_size(
                    form, schema, state_boundary, "vd"
                )
                operands["vd"] = 16 - (16 % destination_group)
                left_name, right_name = source_names[:2]
                left_group = vector_register_group_size(
                    form, schema, state_boundary, left_name
                )
                right_group = vector_register_group_size(
                    form, schema, state_boundary, right_name
                )
                operands[left_name] = 2
                operands[right_name] = (
                    2 + left_group - right_group
                    if left_group > right_group
                    else 2
                )
                row = tuple((key, int(operands[key])) for key in groups)
                if vector_register_violation(
                    form, schema, state_boundary, dict(row), isa_profile
                ) == "vector-source-eew-overlap":
                    rows.append(
                        _base(
                            form,
                            state_boundary,
                            operands=row,
                            lane=LANE_LEGALITY,
                        )
                    )
        for source_name in ("vs1", "vs2", "vs3"):
            if source_name not in groups or not vector_crypto_overlap_restricted(
                form, source_name
            ):
                continue
            source_group = vector_register_group_size(
                form, schema, boundary, source_name
            )
            source = (
                destination + destination_group - source_group
                if destination_group > source_group
                else destination
            )
            if source < 0 or source + source_group > 32:
                continue
            operands = dict(baseline)
            operands[source_name] = source
            row = tuple((key, int(operands[key])) for key in groups)
            if vector_register_violation(
                form, schema, boundary, dict(row), isa_profile
            ) == "vector-register-overlap":
                rows.append(
                    _base(
                        form,
                        f"vector-register-overlap:{source_name}-crypto",
                        operands=row,
                        lane=LANE_LEGALITY,
                    )
                )
        for source_name in ("vs1", "vs2", "vs3"):
            if source_name not in groups:
                continue
            source_group = vector_register_group_size(
                form, schema, boundary, source_name
            )
            if source_group == destination_group:
                continue
            source = (
                destination + destination_group - source_group
                if destination_group > source_group
                else destination
            )
            source_set = set(range(source, source + source_group))
            if source < 0 or source + source_group > 32:
                continue
            operands = dict(baseline)
            operands["vm"] = 1
            occupied = destination_set | source_set
            valid = True
            for other_name in ("vs1", "vs2", "vs3"):
                if other_name not in groups or other_name == source_name:
                    continue
                other_group = vector_register_group_size(
                    form, schema, boundary, other_name
                )
                other = int(operands[other_name])
                other_set = set(range(other, other + other_group))
                if other_set & occupied:
                    other = next(
                        (
                            value
                            for value in range(32)
                            if value % other_group == 0
                            and value + other_group <= 32
                            and not set(range(value, value + other_group)) & occupied
                        ),
                        None,
                    )
                    if other is None:
                        valid = False
                        break
                    operands[other_name] = other
                    other_set = set(range(other, other + other_group))
                occupied |= other_set
            if not valid:
                continue
            operands[source_name] = source
            row = tuple((key, int(operands[key])) for key in groups)
            if vector_register_violation(
                form, schema, boundary, dict(row), isa_profile
            ) is not None:
                continue
            relation = "high" if destination_group > source_group else "low"
            rows.append(
                _base(
                    form,
                    f"vector-register-overlap:{source_name}-{relation}",
                    operands=row,
                    lane=LANE_NORMAL,
                )
            )
    # State edges cannot be represented by a scalar source_state tuple.  They
    # remain explicit boundary classes and are materialized into the compact
    # vector-state descriptor by the shared materializer.
    boundaries = [
        "vector-vl:zero",
        "vector-vl:one",
        "vector-vl:vlmax",
        "vector-vtype:sew8-m1",
        "vector-vtype:sew16-mf2",
        "vector-vtype:sew32-m1",
        "vector-vtype:sew64-m8",
        # The vill state edge forces a reserved vsew so the target sets
        # vtype.vill=1.  Descriptor-backed vtype-dependent forms are emitted
        # on the legality lane; only vset* and whole-register memory forms
        # remain normal-defined because they do not depend on element VTYPE.
        "vector-vtype:vill",
        "vector-vstart:one",
        # A second nonzero vstart value sharpens the restart
        # observation (parser already accepts any decimal; keep one concrete
        # multi-element distance so the vstart-set sequence is nontrivial).
        "vector-vstart:2",
        "vector-vstart:4",
        "vector-tail:agnostic",
        "vector-tail:undisturbed",
    ]
    if vector_crypto_default_state(form) is not None:
        boundaries.insert(5, "vector-vtype:sew32-m2")
    if "frm" in vector_dynamic_state_keys(form, schema) and vector_profile_vtype_violation(
        isa_profile, form, schema, "vector:baseline"
    ) is None:
        boundaries.append("vector-frm:reserved")
    default_state = vector_default_state(form) or (8, 1)
    boundaries = [
        boundary
        for boundary in boundaries
        if not (
            boundary.startswith("vector-vtype:")
            and vector_crypto_default_state(form) is None
            and (edge := vector_state_boundary(boundary)) is not None
            and edge.sew == default_state[0]
            and edge.lmul == default_state[1]
        )
    ]
    # A mask is an operand only when the encoding exposes vm.  The two
    # descriptor forms whose operation is mask-or-carry use v0 implicitly;
    # keep mask edges for those forms without inventing a mnemonic table.
    if mask_operand is not None and "vm" not in groups:
        # ``vm=0`` is the only mask bit the instruction encoding can carry.
        # Keep the explicit all-off boundary (which remains a useful state
        # label for the current observer), but do not emit an ``all-on`` row:
        # it would carry the same vm=0 operand and therefore materialize the
        # exact same bytes/state as all-off while claiming a different mask
        # state that has no descriptor-backed carrier.
        boundaries.append("vector-mask:all-off")
    fallback_registers = {
        "vector-vtype:sew64-m8": (("vd", 8), ("vs1", 16), ("vs2", 24), ("vs3", 8)),
        "vector-vtype:sew32-m2": (("vd", 2), ("vs1", 4), ("vs2", 6), ("vs3", 8)),
    }
    for boundary in boundaries:
        state_operands = dict(baseline)
        fallback = fallback_registers.get(boundary)
        if fallback is not None and (
            schema.kind != "vector-data" or not place_vector_registers(
                state_operands, boundary
            )
        ):
            for name, value in fallback:
                if name in state_operands:
                    state_operands[name] = value
        if boundary == "vector-mask:all-off" and "vm" in state_operands:
            state_operands["vm"] = 0
        state_row = tuple((name, state_operands[name]) for name in groups)
        rows.append(
            _base(
                form,
                boundary,
                operands=state_row,
                lane=(
                    LANE_LEGALITY
                    if boundary == "vector-frm:reserved"
                    or vector_register_violation(
                        form, schema, boundary, dict(state_row), isa_profile
                    )
                    else LANE_NORMAL
                ),
                rounding_mode=5 if boundary == "vector-frm:reserved" else None,
            )
        )
    if vector_fault_only_first(form, schema):
        rows.append(
            _base(
                form,
                "vector-ff:element1-trim",
                operands=tuple((name, baseline[name]) for name in groups),
                source_state=(("vector-fault-element", 1),),
            )
        )
    if schema.kind == "vector-memory" and schema.operation == "vector-load":
        memory_width = 8 if "ei" in str(form.mnemonic) else int(schema.memory_width or 8)
        for label, value in (
            ("zero", 0),
            ("one", 1),
            ("all-ones", (1 << min(memory_width, 64)) - 1),
            ("sign-bit", 1 << (min(memory_width, 64) - 1)),
        ):
            rows.append(
                _base(
                    form,
                    f"vector-memory:value:{label}",
                    operands=tuple((name, baseline[name]) for name in groups),
                    source_state=(("memory", value),),
                )
            )
    return tuple(
        replace(
            row,
            source_state=vector_source_state(row.boundary_class, row.operands_map)
            + tuple(
                pair for pair in row.source_state
                if pair[0] not in dict(
                    vector_source_state(row.boundary_class, row.operands_map)
                )
            ),
        )
        for row in rows
    )


def _vector_vtype_value(
    *,
    sew_bits: int,
    lmul: str,
    ta: int = 1,
    ma: int = 1,
    altfmt: int = 0,
    reserved_bits: int = 0,
) -> int:
    codes = vector_vtype_codes(sew_bits, lmul)
    if codes is None:
        raise ValueError("unsupported VTYPE SEW/LMUL")
    sew_code, lmul_code = codes
    return (
        ((int(reserved_bits) & 0x3) << 9)
        | ((int(altfmt) & 0x1) << 8)
        | ((int(ma) & 0x1) << 7)
        | ((int(ta) & 0x1) << 6)
        | ((sew_code & 0x7) << 3)
        | (lmul_code & 0x7)
    )


def _vector_config_intents(
    form: OfficialForm,
) -> tuple[MaterializationIntent, ...]:
    valid_small = _vector_vtype_value(sew_bits=8, lmul="m1")
    valid_wide = _vector_vtype_value(sew_bits=64, lmul="m8")
    altfmt = _vector_vtype_value(sew_bits=8, lmul="m1", altfmt=1)
    reserved = _vector_vtype_value(sew_bits=8, lmul="m1", reserved_bits=0b01)
    fields = frozenset(form.variable_fields)
    # Unsupported vtype settings complete normally with vill=1 and vl=0;
    # the config-state observers compare that readback instead of expecting a trap.
    if "zimm11" in fields:
        rows = (
            ("avl-zero", (("zimm11", valid_small),), source_state_for(form, 0), LANE_NORMAL),
            ("avl-one", (("zimm11", valid_small),), source_state_for(form, 1), LANE_NORMAL),
            ("avl-vlmax", (("rs1", 0), ("zimm11", valid_wide)), (), LANE_NORMAL),
            ("altfmt-vtype", (("zimm11", altfmt),), source_state_for(form, 3), LANE_NORMAL),
            ("reserved-vtype", (("zimm11", reserved),), source_state_for(form, 3), LANE_NORMAL),
        )
    elif "zimm10" in fields:
        rows = (
            ("avl-zero", (("zimm10", valid_small), ("zimm5", 0)), (), LANE_NORMAL),
            ("avl-one", (("zimm10", valid_small), ("zimm5", 1)), (), LANE_NORMAL),
            ("avl-vlmax", (("zimm10", valid_wide), ("zimm5", 31)), (), LANE_NORMAL),
            ("altfmt-vtype", (("zimm10", altfmt), ("zimm5", 3)), (), LANE_NORMAL),
            ("reserved-vtype", (("zimm10", reserved), ("zimm5", 3)), (), LANE_NORMAL),
        )
    else:
        rows = (
            ("avl-zero", (), (("rs1", 0), ("rs2", valid_small)), LANE_NORMAL),
            ("avl-one", (), (("rs1", 1), ("rs2", valid_small)), LANE_NORMAL),
            ("avl-vlmax", (("rs1", 0),), (("rs2", valid_wide),), LANE_NORMAL),
            ("altfmt-vtype", (), (("rs1", 3), ("rs2", altfmt)), LANE_NORMAL),
            ("reserved-vtype", (), (("rs1", 3), ("rs2", reserved)), LANE_NORMAL),
        )
    return tuple(
        _base(
            form,
            f"vector-config:{name}",
            operands=operands,
            source_state=source_state,
            lane=lane,
        )
        for name, operands, source_state, lane in rows
    )


def _fp_classify_intents(form, schema):
    """One row per IEEE category: the result must be exactly one class bit."""
    precision = schema.precision or 24
    rows = [
        _base(
            form,
            "classify:" + name,
            source_state=(("fs1", int(value)),),
        )
        for name, value in ieee_bit_classes(precision).items()
    ]
    # Classification consumes the same architectural narrower-value rule as
    # arithmetic: on RV64, a non-NaN-boxed .S/.H source is canonical qNaN,
    # not an ordinary low-width value.  Keep this structural and shared so a
    # new classify form inherits the boundary without a mnemonic table.
    rows.extend(
        _base(
            form,
            f"fp:nan-box:{name}:fs1",
            source_state=(("fs1", int(bits)),),
        )
        for name, bits in nan_box_carriers(precision)
    )
    return tuple(rows)


# Destination seeds are a bounded Cartesian product of architectural
# register representatives and extreme 64-bit masks.  It stays a generator so
# the 1479 catalog forms never pre-expand the full grid; each form consumes
# at most _DESTINATION_SEED_AXIS_LIMIT cells.
DESTINATION_REGISTER_REPRESENTATIVES = (0, 1, 2, 3, 5, 8, 28, 31)
DESTINATION_EXTREME_VALUES = (
    0x0,
    0xFFFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x7FFFFFFFFFFFFFFF,
    0x0000000000000001,
    0x8000000000000001,
    0xAAAAAAAA55555555,
    0x55555555AAAAAAAA,
    0xDEADBEEFCAFEBABE,
)
# x0 is hardwired and x2/x31 are harness-reserved, so a seed there can never
# be realized; those registers stay in the representative grid but the seed
# axis skips them (x0's boundary is the rd-eq-x0 relation row instead).
_UNSEEDABLE_DESTINATION_REGISTERS = frozenset({0, 2, 31})
_DESTINATION_SEED_AXIS_LIMIT = 9  # 3 registers x 3 extreme values per form


def _structural_axis_intents(
    form: OfficialForm,
    schema: EffectSchema,
    base_rows: list[MaterializationIntent],
) -> tuple[MaterializationIntent, ...]:
    """Structural axes are dimensions of an intent row, not a separate subsystem."""
    if not base_rows:
        return ()
    seed_row = base_rows[0]
    concrete_xlen = 32 if str(form.xlen) == "rv32" else 64
    concrete_mask = (1 << concrete_xlen) - 1
    registers = register_fields(form)
    destination_fields = tuple(
        name
        for name in registers
        if name.startswith("rd") and "rs1" not in name
    )
    destination_rows: list[MaterializationIntent] = []
    if destination_fields:
        # Compressed pair slots (rd_p/rd_n0) cannot name an arbitrary
        # architectural register; their destination axis stays value-only so
        # the seed sweep remains materializable for every catalog form.
        seed_field = next(
            (
                name
                for name in destination_fields
                if not name.endswith("_p") and "_n0" not in name
            ),
            None,
        )
        seen_destination_seeds: set[tuple[int, int] | int] = set()
        destination_domain = operand_register_domain(
            form, seed_field or destination_fields[0]
        )
        seed_registers = DESTINATION_REGISTER_REPRESENTATIVES if destination_domain == "gpr" else ()
        for register in seed_registers:
            for seed in DESTINATION_EXTREME_VALUES:
                if register in _UNSEEDABLE_DESTINATION_REGISTERS:
                    continue
                projected_seed = int(seed) & concrete_mask
                dedup_key = (
                    (register, projected_seed)
                    if seed_field is not None
                    else projected_seed
                )
                if dedup_key in seen_destination_seeds:
                    continue
                seen_destination_seeds.add(dedup_key)
                operands = dict(seed_row.operands)
                if seed_field is not None:
                    operands[seed_field] = encode_register_field_value(
                        seed_field, register
                    )
                destination_rows.append(
                    replace(
                        seed_row,
                        destination_seed=projected_seed,
                        operands=tuple(sorted(operands.items())),
                    )
                )
                if len(destination_rows) >= _DESTINATION_SEED_AXIS_LIMIT:
                    break
    relation_rows: list[MaterializationIntent] = []
    source_fields = tuple(name for name in registers if "rs1" in name)
    same_register_domain = bool(source_fields) and (
        operand_register_domain(form, "rd")
        == operand_register_domain(form, "rs1")
    )
    relation_seed_row = next(
        (
            row for row in reversed(base_rows)
            if sum(name.startswith("rs") for name, _ in row.source_state) >= 2
            and all(
                int(value) != 0
                for name, value in row.source_state
                if name.startswith("rs")
            )
        ),
        seed_row,
    )
    if destination_fields and same_register_domain:
        relations = ["rd-eq-rs1"]
        if (
            operand_register_domain(form, "rd") == "gpr"
            and any(not name.endswith("_p") and "_n0" not in name for name in destination_fields)
        ):
            relations.append("rd-eq-x0")
        for relation in relations:
            relation_rows.append(
                replace(
                    relation_seed_row,
                    register_relation=relation,
                )
            )
    # x0 is the architecturally hard-wired zero source: a translator that
    # reads it as a normal register (or folds it away) is observable only
    # when the source slot itself is bound to x0.  One suppressed-source row
    # per form keeps this translation-sensitive axis on every integer lane
    # without a per-mnemonic rule.
    source_suppression_rows: list[MaterializationIntent] = []
    if (
        source_fields
        and operand_register_domain(form, "rs1") == "gpr"
        and seed_row.source_state
        and any(
            not name.endswith("_p") and "_n0" not in name
            for name in source_fields
        )
    ):
        suppressed_operands = dict(seed_row.operands)
        suppressed_operands["rs1"] = encode_register_field_value("rs1", 0)
        # x0 is architecturally zero and needs no seeding, so the suppressed
        # source leaves source_state (which also keeps the row out of the
        # rs1 stateflow consumer basis: a predecessor rd cannot write x0).
        source_suppression_rows.append(
            replace(
                seed_row,
                boundary_class="source-suppression:rs1-x0",
                operands=tuple(sorted(suppressed_operands.items())),
                source_state=tuple(
                    (role, value)
                    for role, value in seed_row.source_state
                    if role != "rs1"
                ),
            )
        )
    projection_rows: list[MaterializationIntent] = []
    projection_width = schema.source_width or schema.width
    if projection_width < concrete_xlen:
        # A narrower effect must ignore the high half of its source; dirtying it
        # is how a missing canonicalisation becomes observable.
        low_mask = (1 << projection_width) - 1
        high_mask = concrete_mask ^ low_mask
        seed_by_role = dict(seed_row.source_state)
        for pattern in (
            high_mask,
            0xDEADBEEFDEADBEEF & high_mask,
        ):
            dirty = tuple(
                (name, (int(value) & low_mask) | pattern) if name.startswith("rs") else (name, value)
                for name, value in seed_row.source_state
            )
            if dirty == seed_row.source_state:
                continue
            # 投影兄弟寄存器：同一投影值、不同完整值。RVEMI R2 的
            # projected-alias 候选要求窗口里存在"low32 相同、high 不同"
            # 的寄存器对；对 W 形式（低 32 位投影）来说，兄弟值 = 干净
            # high 的同一 low32。这样 R2 变体把脏源换成干净兄弟，正确
            # 实现结果不变，而误用完整 64 位源的目标会产生 mismatch。
            first_rs = next(
                (name for name, _value in dirty if name.startswith("rs")),
                None,
            )
            sibling = ()
            if first_rs is not None:
                sibling_value = int(seed_by_role[first_rs]) & low_mask
                # 零值兄弟天然存在（空闲寄存器初始为 0），无需显式播种。
                if sibling_value:
                    sibling = ((first_rs, sibling_value),)
            projection_rows.append(
                replace(
                    seed_row,
                    source_state=dirty,
                    sibling_state=sibling,
                )
            )
    return tuple(
        destination_rows
        + relation_rows
        + source_suppression_rows
        + projection_rows
    )

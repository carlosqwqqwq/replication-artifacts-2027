from functools import cache

U64_MASK = (1 << 64) - 1
SYSTEM_OPCODE = 0x73

# Official vtype CSR field encodings (RISC-V V spec, vtype layout):
# vsew[2:0] and vlmul[2:0] bit patterns inside the vtype CSR.
VECTOR_VTYPE_SEW_ENCODING = {8: 0b000, 16: 0b001, 32: 0b010, 64: 0b011}
VECTOR_VTYPE_LMUL_ENCODING = {
    "mf8": 0b101,
    "mf4": 0b110,
    "mf2": 0b111,
    "m1": 0b000,
    "m2": 0b001,
    "m4": 0b010,
    "m8": 0b011,
}


def sign_extend(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


# --- generic form encoding driven by the frozen riscv-opcodes field table ---
#
# Every instruction bit an operand occupies comes from OFFICIAL_FIELD_BIT_RANGES
# (the frozen upstream arg_lut).  For most fields the operand value sits in that
# span unchanged.  The B/J formats and the compressed formats scatter the value's
# bits across the span; that scattering is a property of the *field name*, which
# riscv-opcodes reuses across every extension, so one table below covers the whole
# ISA -- present and future -- rather than one encoder per instruction format.
#
# Each entry maps a logical immediate to chunks of (value_msb, value_lsb, inst_lsb).
# Layouts are the immediate encodings of the RISC-V ISA Manual (base integer
# formats B/J; compressed formats CI/CSS/CIW/CL/CS/CB/CJ).

SCATTERED_IMMEDIATE_LAYOUT: dict[str, tuple[tuple[int, int, int], ...]] = {
    # B-type branch: imm[12|10:5] in 31:25, imm[4:1|11] in 11:7
    "bimm12": ((12, 12, 31), (10, 5, 25), (4, 1, 8), (11, 11, 7)),
    # J-type jump: imm[20|10:1|11|19:12] in 31:12
    "jimm20": ((20, 20, 31), (10, 1, 21), (11, 11, 20), (19, 12, 12)),
    # S-type store: imm[11:5] in 31:25, imm[4:0] in 11:7
    "imm12s": ((11, 5, 25), (4, 0, 7)),
    # CI 6-bit immediates: imm[5] in 12, imm[4:0] in 6:2
    "c_imm6": ((5, 5, 12), (4, 0, 2)),
    # C.LUI: nzimm[17] in 12, nzimm[16:12] in 6:2
    "c_nzimm18": ((17, 17, 12), (16, 12, 2)),
    # C.ADDI16SP: nzimm[9] in 12, [4] in 6, [6] in 5, [8:7] in 4:3, [5] in 2
    "c_nzimm10": ((9, 9, 12), (4, 4, 6), (6, 6, 5), (8, 7, 3), (5, 5, 2)),
    # C.ADDI4SPN: nzuimm[5:4] in 12:11, [9:6] in 10:7, [2] in 6, [3] in 5
    "c_nzuimm10": ((5, 4, 11), (9, 6, 7), (2, 2, 6), (3, 3, 5)),
    # CL/CS word: uimm[5:3] in 12:10, [2] in 6, [6] in 5
    "c_uimm7": ((5, 3, 10), (2, 2, 6), (6, 6, 5)),
    # Zcb halfword CL/CS: the architectural offset is uimm[1], so the
    # encoded bit is already a two-byte-aligned logical immediate.
    "c_uimm1": ((1, 1, 5),),
    # Zcb byte CL/CS: uimm[1] is in bit 5 and uimm[0] is in bit 6.
    "c_uimm2": ((0, 0, 6), (1, 1, 5)),
    # CL/CS doubleword: uimm[5:3] in 12:10, [7:6] in 6:5
    "c_uimm8": ((5, 3, 10), (7, 6, 5)),
    # C.LWSP: uimm[5] in 12, [4:2] in 6:4, [7:6] in 3:2
    "c_uimm8sp": ((5, 5, 12), (4, 2, 4), (7, 6, 2)),
    # C.LDSP: uimm[5] in 12, [4:3] in 6:5, [8:6] in 4:2
    "c_uimm9sp": ((5, 5, 12), (4, 3, 5), (8, 6, 2)),
    # C.SWSP: uimm[5:2] in 12:9, [7:6] in 8:7
    "c_uimm8sp_s": ((5, 2, 9), (7, 6, 7)),
    # C.SDSP: uimm[5:3] in 12:10, [8:6] in 9:7
    "c_uimm9sp_s": ((5, 3, 10), (8, 6, 7)),
    # CJ: imm[11] in 12, [4] in 11, [9:8] in 10:9, [10] in 8, [6] in 7,
    #     [7] in 6, [3:1] in 5:3, [5] in 2
    "c_imm12": (
        (11, 11, 12),
        (4, 4, 11),
        (9, 8, 9),
        (10, 10, 8),
        (6, 6, 7),
        (7, 7, 6),
        (3, 1, 3),
        (5, 5, 2),
    ),
    # CB branch: imm[8] in 12, [4:3] in 11:10, [7:6] in 6:5, [2:1] in 4:3, [5] in 2
    "c_bimm9": ((8, 8, 12), (4, 3, 10), (7, 6, 5), (2, 1, 3), (5, 5, 2)),
    # Vector rotate-immediate: zimm[5] in 26, zimm[4:0] in 19:15.
    "zimm6": ((5, 5, 26), (4, 0, 15)),
}

_FORM_ALIASES = {"cm_jt": "cm_jalt"}
_CM_INDEX_BOUNDS = {"cm_jt": (0, 32), "cm_jalt": (32, 256)}

# Field names that carry one logical immediate split over a hi/lo pair, plus the
# split-free spellings.  Mapping a field name to its logical immediate is what
# lets a form be encoded without knowing which instruction it is.
_LOGICAL_IMMEDIATE_BY_FIELD = {
    **dict.fromkeys(("bimm12hi", "bimm12lo"), "bimm12"),
    **dict.fromkeys(("imm12hi", "imm12lo"), "imm12s"),
    "jimm20": "jimm20",
    **dict.fromkeys(
        ("c_imm6hi", "c_imm6lo", "c_nzimm6hi", "c_nzimm6lo", "c_nzuimm6hi", "c_nzuimm6lo"),
        "c_imm6",
    ),
    **dict.fromkeys(("c_nzimm18hi", "c_nzimm18lo"), "c_nzimm18"),
    **dict.fromkeys(("c_nzimm10hi", "c_nzimm10lo"), "c_nzimm10"),
    "c_nzuimm10": "c_nzuimm10",
    **dict.fromkeys(("c_uimm7hi", "c_uimm7lo"), "c_uimm7"),
    **dict.fromkeys(("c_uimm8hi", "c_uimm8lo"), "c_uimm8"),
    **dict.fromkeys(("c_uimm8sphi", "c_uimm8splo"), "c_uimm8sp"),
    **dict.fromkeys(("c_uimm9sphi", "c_uimm9splo"), "c_uimm9sp"),
    "c_uimm8sp_s": "c_uimm8sp_s",
    "c_uimm1": "c_uimm1",
    "c_uimm2": "c_uimm2",
    "c_uimm9sp_s": "c_uimm9sp_s",
    "c_imm12": "c_imm12",
    **dict.fromkeys(("c_bimm9hi", "c_bimm9lo"), "c_bimm9"),
    **dict.fromkeys(("zimm6hi", "zimm6lo"), "zimm6"),
}

# Zibi compare-immediate branches use a B-type displacement even though the
# pinned opcode row spells the two carrier fields ``imm12hi/lo``.  Correct the
# logical owner at this boundary without adding a mnemonic-specific generator.
_FORM_LOGICAL_IMMEDIATE_OVERRIDES = dict.fromkeys(("beqi", "bnei"), {"imm12hi": "bimm12", "imm12lo": "bimm12"})


@cache
def logical_immediate_for_form_field(form, field: str) -> str | None:
    """Return the logical immediate owner after form-specific ISA correction."""
    override = _FORM_LOGICAL_IMMEDIATE_OVERRIDES.get(
        str(getattr(form, "mnemonic", "")),
        {},
    ).get(field)
    return override or _LOGICAL_IMMEDIATE_BY_FIELD.get(field)


def operand_group(field: str) -> str:
    """Classify a variable field without knowing which instruction owns it."""
    return _LOGICAL_IMMEDIATE_BY_FIELD.get(field, field)


def encode_scattered_immediate(logical_name: str, value: int) -> int:
    layout = SCATTERED_IMMEDIATE_LAYOUT[logical_name]
    value = _require_operand_int(value, f"{logical_name}:immediate")
    width = max(value_msb for value_msb, _value_lsb, _inst_lsb in layout) + 1
    if value < 0 or value >= (1 << width):
        raise ValueError(f"{logical_name} value {value} does not fit {width} bits")
    low_bit = min(value_lsb for _value_msb, value_lsb, _inst_lsb in layout)
    if low_bit and value % (1 << low_bit):
        raise ValueError(
            f"{logical_name} value {value} is not aligned to {1 << low_bit}"
        )
    word = 0
    for value_msb, value_lsb, inst_lsb in layout:
        width = value_msb - value_lsb + 1
        chunk = (int(value) >> value_lsb) & ((1 << width) - 1)
        word |= chunk << inst_lsb
    return word


def decode_scattered_immediate(logical_name: str, word: int) -> int:
    layout = SCATTERED_IMMEDIATE_LAYOUT[logical_name]
    encoded = _require_encoded_word(word, 32, f"{logical_name}:word")
    value = 0
    for value_msb, value_lsb, inst_lsb in layout:
        width = value_msb - value_lsb + 1
        chunk = (encoded >> inst_lsb) & ((1 << width) - 1)
        value |= chunk << value_lsb
    return value


def scattered_immediate_mask(logical_name: str) -> int:
    mask = 0
    for value_msb, value_lsb, inst_lsb in SCATTERED_IMMEDIATE_LAYOUT[logical_name]:
        width = value_msb - value_lsb + 1
        mask |= ((1 << width) - 1) << inst_lsb
    return mask


def _field_instruction_mask_for_form(form, field: str) -> int:
    from .riscv_catalog import OFFICIAL_FIELD_BIT_RANGES

    msb, lsb = OFFICIAL_FIELD_BIT_RANGES[field]
    span = ((1 << (msb - lsb + 1)) - 1) << lsb
    logical = logical_immediate_for_form_field(form, field)
    return span if logical is None else scattered_immediate_mask(logical) & span


def _require_operand_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_encoded_word(value: object, bits: int, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    encoded = value
    if encoded < 0 or encoded >= (1 << bits):
        raise ValueError(f"{name} is outside the {bits}-bit encoding range")
    return encoded


def encode_form(form, values: dict[str, int]) -> int:
    """Encode any catalog form from its frozen structure.

    ``values`` is keyed by operand group: register fields by their own name,
    scattered immediates by the logical immediate name (``bimm12``, ``c_imm6``
    ...), so a caller never needs to know how a format splits its immediate.
    Forms this function has never seen encode exactly like the ones it has.
    """
    from .riscv_catalog import OFFICIAL_FIELD_BIT_RANGES

    word = int(form.match)
    applied: set[str] = set()
    for field in form.variable_fields:
        logical = logical_immediate_for_form_field(form, field)
        group = logical or operand_group(field)
        if group not in values:
            raise ValueError(f"{form.mnemonic}: missing operand {group!r}")
        value = _require_operand_int(values[group], f"{form.mnemonic}:{group}")
        if logical is None:
            msb, lsb = OFFICIAL_FIELD_BIT_RANGES[field]
            width = msb - lsb + 1
            if value < 0 or value >= (1 << width):
                raise ValueError(f"{form.mnemonic}: {field} value {value} does not fit {width} bits")
            word |= value << lsb
            applied.add(group)
            continue
        if logical in applied:
            continue
        # A scattered immediate is written once, across every field carrying it.
        whole = 0
        for sibling in form.variable_fields:
            if logical_immediate_for_form_field(form, sibling) == logical:
                whole |= _field_instruction_mask_for_form(form, sibling)
        word |= encode_scattered_immediate(logical, value) & whole
        applied.add(logical)
    unexpected = set(values) - applied
    if unexpected:
        raise ValueError(f"{form.mnemonic}: unexpected operands {sorted(unexpected)}")
    return word


def decode_form(form, word: int) -> dict[str, int]:
    """Recover operand-group values from an encoded word of a known form."""
    from .riscv_catalog import OFFICIAL_FIELD_BIT_RANGES

    bits = int(form.encoding_length_bytes) * 8
    if bits not in {16, 32}:
        raise ValueError(f"{form.mnemonic}: unsupported encoding length")
    word = _require_encoded_word(word, bits, f"{form.mnemonic}:word")
    values: dict[str, int] = {}
    for field in form.variable_fields:
        logical = logical_immediate_for_form_field(form, field)
        if logical is None:
            msb, lsb = OFFICIAL_FIELD_BIT_RANGES[field]
            values[field] = (int(word) >> lsb) & ((1 << (msb - lsb + 1)) - 1)
            continue
        if logical not in values:
            values[logical] = decode_scattered_immediate(logical, int(word))
    return values


@cache
def operand_groups(form) -> tuple[str, ...]:
    """Distinct operand groups a form takes, in catalog field order."""
    seen: list[str] = []
    for field in form.variable_fields:
        logical = logical_immediate_for_form_field(form, field)
        group = logical or operand_group(field)
        if group not in seen:
            seen.append(group)
    return tuple(seen)


def branch_compare_immediate(form, decoded: dict[str, int]) -> int | None:
    """Decode the immediate comparison constant of a Zibi branch."""
    if "bimm12" not in operand_groups(form) or "imm5" not in operand_groups(form):
        return None
    value = decoded.get("imm5")
    return None if value is None else -1 if int(value) == 0 else int(value)


REGISTER_OPERAND_ROLES = ("rd", "rs1", "rs2", "rs3")

_FORM_OPERAND_KIND_OVERRIDES = {
    **dict.fromkeys(("fli_d", "fli_h", "fli_q", "fli_s"), {"rs1": "imm"}),
    "cm_jalt": {"c_index": "imm"},
}


def operand_role(field_group: str) -> str:
    """把 catalog 的字段组名归一成操作数槽名。

    压缩编码的 ``rd_rs1_p`` / ``c_rs2_n0`` 与 32 位编码的 ``imm12`` / ``bimm12`` /
    ``shamtd`` 在这里收敛成同一套槽名，上层因此不必认识具体字段拼写，新增编码形式
    也不需要改这里。不属于操作数槽的模式字段（``rm``/``aq``/``rl``/``csr``）原样返回。
    """
    name = str(field_group).removeprefix("c_").removeprefix("p_")
    for role in REGISTER_OPERAND_ROLES:
        if name.startswith(role):
            return role
    if "imm" in name or name.startswith("shamt"):
        return "imm"
    return name


@cache
def operand_kind(form, field_group: str) -> str:
    """Form-aware operand kind.

    Most ISA forms can classify a field group by name alone, but a small number
    of encodings reuse a register bit span as a literal selector.  Keeping that
    override here prevents every consumer from growing its own ad hoc exception.
    """
    mnemonic = str(getattr(form, "mnemonic", "")).replace(".", "_")
    override = _FORM_OPERAND_KIND_OVERRIDES.get(mnemonic, {}).get(str(field_group))
    return override or operand_role(field_group)


def is_register_operand_group(form, field_group: str) -> bool:
    """Whether this form/group encodes a register number.

    Semantic operand kind and register encoding carrier are not the same thing:
    ``fli.*`` reuses ``rs1`` bits as a literal selector, while Zcmp ``c_sreg*``
    fields still encode registers even though they are not ordinary
    ``rd/rs1/rs2/rs3`` roles.
    """
    group = str(field_group)
    if operand_kind(form, group) == "imm":
        return False
    return operand_role(group) in REGISTER_OPERAND_ROLES or group in {"c_sreg1", "c_sreg2"}


def _register_roles_for_group(form, field_group: str) -> tuple[str, ...]:
    """Canonical register roles carried by one named operand group."""
    if not is_register_operand_group(form, field_group):
        return ()
    name = str(field_group)
    return tuple(role for role in REGISTER_OPERAND_ROLES if role in name)


def implicit_source_roles(form) -> tuple[str, ...]:
    """Architecture-fixed source roles absent from variable operand fields."""
    fields = operand_groups(form)
    if getattr(form, "encoding_length_bytes", 4) != 2:
        return ()
    if any("sp" in name for name in fields) or "c_nzimm10" in fields or "c_nzuimm10" in fields:
        return ("rs1",)
    return ()


def source_roles(form) -> tuple[str, ...]:
    """Source roles the form actually reads, in operand order."""
    roles: list[str] = []
    for name in operand_groups(form):
        if operand_kind(form, name) not in REGISTER_OPERAND_ROLES:
            continue
        for role in ("rs1", "rs2", "rs3"):
            if role in name and role not in roles:
                roles.append(role)
    for role in implicit_source_roles(form):
        if role not in roles:
            roles.append(role)
    return tuple(roles)


def immediate_fields(form) -> tuple[str, ...]:
    """Immediate-like operand groups in catalog field order."""
    return tuple(name for name in operand_groups(form) if not is_register_operand_group(form, name))


def canonical_immediate_field(form) -> str | None:
    """Return the immediate field used by the instruction's public API."""
    fields = tuple(
        name for name in immediate_fields(form) if operand_kind(form, name) == "imm"
    )
    if not fields:
        return None
    return "zimm5" if form.mnemonic == "vsetivli" else fields[0]


def encoded_immediate_for_metadata(form, value: int | None) -> int | None:
    """Convert an ``InstructionMeta.immediate`` value to named-encoder units.

    U-type metadata records the architectural value (the raw 20-bit field
    shifted left by 12), while ``encode_instruction`` accepts the raw field.
    Keep that conversion at the single RVEMI/rewriter boundary so a same-form
    rewrite cannot accidentally move an AUIPC/LUI immediate.
    """
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{form.mnemonic}:metadata immediate must be an integer")
    semantic = value
    if "imm20" not in operand_groups(form):
        return semantic
    if semantic % (1 << 12):
        raise ValueError(
            f"{form.mnemonic}: metadata imm20 value {semantic} is not 4096-byte aligned"
        )
    raw = semantic // (1 << 12)
    if raw < -(1 << 19) or raw > (1 << 20) - 1:
        raise ValueError(f"{form.mnemonic}: metadata imm20 value {semantic} is out of range")
    return raw


@cache
def _cached_form_for_mnemonic(normalized: str):
    from .riscv_catalog import official_form

    canonical = _FORM_ALIASES.get(normalized, normalized)
    return official_form(canonical)


def form_for_mnemonic(mnemonic: str):
    """The catalog form behind a generation mnemonic, or None."""
    if type(mnemonic) is not str:
        return None
    return _cached_form_for_mnemonic(mnemonic.replace(".", "_"))


def assembly_mnemonic(form) -> str:
    return str(form.mnemonic).replace("_", ".")


def _cm_index_valid(mnemonic: str, value: object) -> bool:
    bounds = _CM_INDEX_BOUNDS.get(mnemonic.replace(".", "_").lower())
    return bounds is None or type(value) is int and bounds[0] <= value < bounds[1]


def encode_instruction(
    mnemonic: str,
    *,
    rd: int | None = None,
    rs1: int | None = None,
    rs2: int | None = None,
    rs3: int | None = None,
    immediate: int | None = None,
    **extra: int,
) -> int:
    """Encode a named instruction through the frozen catalog structure.

    Callers name registers and one immediate; which fields a form actually has,
    and where their bits live, comes from the catalog, so this works for forms
    the caller has never heard of.
    """
    form = form_for_mnemonic(mnemonic)
    if form is None:
        raise ValueError(f"unknown form: {mnemonic!r}")
    normalized = mnemonic.replace(".", "_").lower()
    if normalized in _CM_INDEX_BOUNDS:
        selector = extra.get("c_index", immediate)
        if selector is None:
            selector = 32 if normalized == "cm_jalt" else 0
        if not _cm_index_valid(mnemonic, selector):
            raise ValueError(f"{mnemonic}: c_index {selector} is out of range")
        if "c_index" not in extra and immediate is None:
            extra["c_index"] = selector
    # FLI reuses the architectural ``rs1`` bit span as a literal selector.
    # Keep the generic call signature compatible, but do not silently discard
    # an explicitly supplied selector when the canonical ``immediate`` alias
    # is also present.
    if operand_kind(form, "rs1") == "imm" and rs1 is not None:
        rs1_value = _require_operand_int(rs1, f"{form.mnemonic}:rs1")
        if immediate is not None and rs1_value != _require_operand_int(
            immediate, f"{form.mnemonic}:immediate"
        ):
            raise ValueError(
                f"{form.mnemonic}: rs1 selector conflicts with 'immediate'"
            )
        immediate = rs1_value if immediate is None else immediate
        rs1 = None
    groups = operand_groups(form)
    unexpected = set(extra) - set(groups)
    if unexpected:
        raise ValueError(f"{form.mnemonic}: unexpected operands {sorted(unexpected)}")
    if immediate is not None and not any(operand_kind(form, group) == "imm" for group in groups):
        raise ValueError(f"{form.mnemonic}: instruction has no immediate operand")
    supplied = {"rd": rd, "rs1": rs1, "rs2": rs2, "rs3": rs3}
    for role, value in supplied.items():
        if value is not None:
            _require_operand_int(value, f"{form.mnemonic}:{role}")
    register_roles = {
        role
        for group in groups
        for role in _register_roles_for_group(form, group)
    }
    for role, value in supplied.items():
        if value is not None and role not in register_roles:
            raise ValueError(f"{form.mnemonic}: instruction has no {role} register operand")
    for group in groups:
        roles = _register_roles_for_group(form, group)
        if group in extra and any(supplied[role] is not None for role in roles):
            raise ValueError(
                f"{form.mnemonic}: supply register group {group!r} either by its group or canonical roles"
            )
    immediate_groups = tuple(
        group for group in groups if operand_kind(form, group) == "imm"
    )
    primary_immediate_group = canonical_immediate_field(form)
    if (
        immediate is not None
        and primary_immediate_group is not None
        and primary_immediate_group in extra
    ):
        raise ValueError(
            f"{form.mnemonic}: supply the primary immediate as either "
            f"'immediate' or '{primary_immediate_group}', not both"
        )
    secondary_immediate_groups = tuple(
        group for group in immediate_groups if group != primary_immediate_group
    )
    if (
        immediate is not None
        and len(immediate_groups) > 1
        and not set(secondary_immediate_groups).issubset(extra)
    ):
        raise ValueError(
            f"{form.mnemonic}: immediate is ambiguous; supply "
            f"{list(secondary_immediate_groups)} explicitly"
        )
    values: dict[str, int] = {}
    for group in groups:
        if group in extra:
            raw_value = _require_operand_int(extra[group], f"{form.mnemonic}:{group}")
            if operand_kind(form, group) == "imm":
                values[group] = _encode_instruction_immediate(form, group, raw_value)
            elif is_register_operand_group(form, group):
                values[group] = encode_register_field_value(group, raw_value)
            else:
                mask = _group_value_mask(group)
                if raw_value < 0 or raw_value > mask:
                    raise ValueError(
                        f"{form.mnemonic}: {group} value {raw_value} exceeds field range"
                    )
                values[group] = raw_value
            continue
        # 槽名要按角色归一后再匹配：压缩编码把它们拼写成 rd_p / rs1_p / c_rs2_n0，
        # 按字面名匹配会让整条压缩指令的寄存器字段全部落到 0。
        roles = _register_roles_for_group(form, group)
        if roles:
            provided = [supplied[role] for role in roles if supplied[role] is not None]
            if len(set(provided)) > 1:
                raise ValueError(
                    f"{form.mnemonic}: register group {group!r} received conflicting canonical roles"
                )
            values[group] = (
                0
                if not provided
                else encode_register_field_value(group, provided[0])
            )
        elif operand_kind(form, group) == "imm":
            # The named API has one canonical ``immediate`` slot.  A handful of
            # forms (notably Zibi beqi/bnei) carry a second literal selector;
            # that selector must be supplied through its explicit group instead
            # of receiving the branch displacement by accident.
            value = (
                0
                if immediate is None or group != primary_immediate_group
                else _require_operand_int(immediate, f"{form.mnemonic}:immediate")
            )
            values[group] = _encode_instruction_immediate(form, group, value)
        else:
            # rm / aq / rl / bs / csr 这类模式字段没被指名就取 0，绝不拿立即数去填。
            values[group] = 0
    return encode_form(form, values)


_SIGNED_IMMEDIATE_BITS = {
    **dict.fromkeys(("imm12", "imm12s", "c_imm12"), 12),
    "bimm12": 13,
    "jimm20": 21,
    **dict.fromkeys(("c_imm6", "c_nzimm6"), 6),
    "c_nzimm10": 10,
    "c_nzimm18": 18,
    "c_bimm9": 9,
    "simm5": 5,
    "p_imm10": 10,
}


def _encode_instruction_immediate(form, group: str, value: int) -> int:
    """Validate and encode the base ISA immediate groups used by generators.

    ``encode_form`` remains the low-level field encoder (and intentionally keeps
    its raw-bit behavior).  The named-instruction API, however, receives
    semantic immediates, so signed fields and branch/jump alignment must be
    checked before the value is converted to field bits.
    """
    if group == "c_imm6" and any(
        name.startswith("c_nzuimm6") for name in form.variable_fields
    ):
        # C.SLLI/C.SRLI/C.SRAI reuse the c_imm6 carrier but it is an unsigned
        # shift amount.  The high bit is reserved in RV32 and defined in RV64.
        limit = 1 << (5 if str(getattr(form, "xlen", "shared")) == "rv32" else 6)
        if value < 0 or value >= limit:
            raise ValueError(
                f"{form.mnemonic}: compressed shift amount {value} is out of range [0, {limit})"
            )
        return value
    layout = SCATTERED_IMMEDIATE_LAYOUT.get(group)
    if layout:
        low_bit = min(value_lsb for _value_msb, value_lsb, _inst_lsb in layout)
        if low_bit and value % (1 << low_bit):
            raise ValueError(
                f"{form.mnemonic}: {group} value {value} is not aligned to {1 << low_bit}"
            )
    bits = _SIGNED_IMMEDIATE_BITS.get(group)
    if bits is not None:
        if group in {"bimm12", "jimm20", "c_bimm9", "c_imm12"} and value % 2:
            message = (
                "branch offset"
                if group in {"bimm12", "c_bimm9"}
                else "jump offset"
            )
            raise ValueError(f"{message} must be 2-byte aligned")
        if not -(1 << (bits - 1)) <= value <= (1 << (bits - 1)) - 1:
            raise ValueError(f"immediate {value} does not fit signed {bits}-bit field")
        return value & ((1 << bits) - 1)

    if group == "imm20":
        # U-type callers use both the encoded unsigned field (e.g. 0xfffff)
        # and its signed alias (e.g. -1); reject values outside either 20-bit
        # representation rather than silently truncating arbitrary integers.
        if value < -(1 << 19) or value > (1 << 20) - 1:
            raise ValueError(f"immediate {value} does not fit U-type 20-bit field")
        return value & ((1 << 20) - 1)

    if group == "imm5" and "bimm12" in operand_groups(form):
        # Zibi-style branch forms encode a uimm5 compare selector; -1 is the
        # assembler's signed spelling for all ones.  Derive this from the
        # catalog's paired branch/immediate fields, not a mnemonic allowlist.
        # Zero remains representable for a reserved legality row, but values
        # outside the selector field are never aliases.
        if value < -1 or value > 31:
            raise ValueError(f"{form.mnemonic}: imm5 selector {value} is out of range")
        return 0 if value == -1 else value

    if form.mnemonic in {"fli_d", "fli_h", "fli_q", "fli_s"} and group == "rs1":
        if value < 0 or value > 31:
            raise ValueError(f"{form.mnemonic}: literal selector {value} is out of range")
        return value

    if group.startswith("shamt"):
        if group == "shamtd":
            xlen = str(getattr(form, "xlen", "shared"))
            limit = 1 << (5 if xlen == "rv32" else 6)
        elif group == "shamtw":
            limit = 1 << 5
        else:
            limit = 1 << (_group_value_mask(group).bit_length())
        if value < 0 or value >= limit:
            raise ValueError(
                f"{form.mnemonic}: {group} shift amount {value} is out of range [0, {limit})"
            )
        return value

    # Extension/compressed immediates still need a representability check even
    # when their exact signedness is form-specific.  Accept the union of the
    # signed semantic range and the raw unsigned field range, but never an
    # arbitrary integer that would be silently truncated.  Explicitly unsigned
    # fields stay non-negative.  A representable zero is deliberately allowed:
    # reserved-zero encodings are useful legality boundaries and belong to the
    # admission layer, not to this bit-field encoder.
    mask = _group_value_mask(group)
    if group.startswith(("zimm", "uimm", "c_uimm", "c_nzuimm", "p_imm8", "p_w_uimm")) or group in {"c_index", "c_spimm"}:
        if value < 0 or value > mask:
            raise ValueError(f"{form.mnemonic}: {group} value {value} exceeds field range")
    else:
        bits = mask.bit_length()
        if value < -(1 << (bits - 1)) or value > mask:
            raise ValueError(f"{form.mnemonic}: {group} value {value} exceeds field range")
    return value & mask


COMPRESSED_REGISTER_RANGE = (8, 15)
COMPRESSED_SREG_REGISTERS = (8, 9, 18, 19, 20, 21, 22, 23)


def encode_register_field_value(group: str, register: int) -> int:
    """寄存器号在该字段里的编码值。

    压缩编码的 3 位寄存器字段只能表示 x8--x15，超出范围时报错而不是回绕：
    悄悄绕成另一个寄存器会产出一条语义不同、却看起来合法的指令。
    """
    register = _require_operand_int(register, f"{group}:register")
    if str(group).startswith("p_"):
        if not 0 <= register < 16:
            raise ValueError(f"packed register field {group} cannot name x{register}")
        return register
    # Every catalog register field is five bits wide.  The named API is a
    # semantic operand boundary, so an out-of-range register must be rejected
    # instead of being masked into a different architectural register.
    if not 0 <= register < 32:
        raise ValueError(f"register field {group} cannot name x{register}")
    if not _is_compressed_register_field(group):
        return register
    if group in {"c_sreg1", "c_sreg2"}:
        if register not in COMPRESSED_SREG_REGISTERS:
            raise ValueError(f"compressed s-register field {group} cannot name x{register}")
        return COMPRESSED_SREG_REGISTERS.index(register)
    if not COMPRESSED_REGISTER_RANGE[0] <= register <= COMPRESSED_REGISTER_RANGE[1]:
        raise ValueError(f"compressed register field {group} cannot name x{register}")
    return register - COMPRESSED_REGISTER_RANGE[0]


def decode_register_field_value(group: str, encoded: int) -> int:
    """Recover the architectural register a field encoding names."""
    value = _require_operand_int(encoded, f"{group}:encoded register")
    if str(group).startswith("p_") and not 0 <= value < 16:
        raise ValueError(f"packed register field {group} cannot decode {encoded}")
    if not 0 <= value < 32:
        raise ValueError(f"register field {group} cannot decode {encoded}")
    if not _is_compressed_register_field(group):
        return value
    if group in {"c_sreg1", "c_sreg2"}:
        if not 0 <= value < len(COMPRESSED_SREG_REGISTERS):
            raise ValueError(f"compressed s-register field {group} cannot decode {encoded}")
        return COMPRESSED_SREG_REGISTERS[value]
    register = value + COMPRESSED_REGISTER_RANGE[0]
    if not COMPRESSED_REGISTER_RANGE[0] <= register <= COMPRESSED_REGISTER_RANGE[1]:
        raise ValueError(f"compressed register field {group} cannot decode {encoded}")
    return register


def decoded_register_slots(form, word: int) -> dict[str, int]:
    """Return canonical register slots decoded from one catalog form.

    The catalog names compressed and combined fields differently from the
    public ``rd``/``rs1``/``rs2`` metadata slots.  Keeping this normalization
    beside the encoder prevents definedness checks from accepting bytes that
    disagree with their declared register operands.
    """
    decoded = decode_form(form, int(word))
    slots: dict[str, int] = {}
    for group in operand_groups(form):
        if not is_register_operand_group(form, group):
            continue
        encoded = decoded.get(group)
        if encoded is None:
            continue
        register = decode_register_field_value(group, int(encoded))
        for role in _register_roles_for_group(form, group):
            slots.setdefault(role, register)
    for role in implicit_source_roles(form):
        slots.setdefault(role, 2)
    if "c_nzimm10" in operand_groups(form):
        slots.update({"rd": 2, "rs1": 2})
    if str(getattr(form, "mnemonic", "")) in {"c_jal", "c_jalr"} or (
        str(getattr(form, "mnemonic", "")) == "cm_jalt"
        and int(decoded.get("c_index", 0)) >= 32
    ):
        slots.setdefault("rd", 1)
    return slots


def _is_compressed_register_field(group: str) -> bool:
    # ``p_rd_p``/``p_rs*_p`` are packed-extension four-bit fields, not the
    # compressed x8..x15 carrier.  Treating them as C fields silently adds x8
    # and corrupts P-form metadata.
    return (group.endswith("_p") and not group.startswith("p_")) or group in {
        "c_sreg1",
        "c_sreg2",
    }


def rewrite_instruction_operands(mnemonic: str, word: int, **operands: int | None) -> int:
    """在原编码上只改指定的操作数槽，其余位原样保留。

    从零重编码会把 ``rm`` / ``aq`` / ``rl`` / ``bs`` 这些调用方没提到的字段清成 0，
    等于顺手改掉舍入模式或字节选择——那不再是等价改写。这里因此只把**被指名的槽**
    所占的位从重编码结果里取出来盖上去，立即数散列的细节仍由 `encode_instruction()`
    统一负责，不在这里复制第二份。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None:
        raise ValueError(f"unknown form: {mnemonic!r}")
    word = _require_encoded_word(
        word,
        int(form.encoding_length_bytes) * 8,
        f"{form.mnemonic}:word",
    )
    groups = operand_groups(form)
    immediate_groups = tuple(
        group for group in groups if operand_kind(form, group) == "imm"
    )
    allowed_aliases = {"rd", "rs1", "rs2", "rs3", "imm"}
    unknown = set(operands) - set(groups) - allowed_aliases
    if unknown:
        raise ValueError(f"{form.mnemonic}: unexpected operands {sorted(unknown)}")
    if "imm" in operands and operands["imm"] is not None and not immediate_groups:
        raise ValueError(f"{form.mnemonic}: instruction has no immediate operand")
    register_roles = {
        role
        for group in groups
        for role in _register_roles_for_group(form, group)
    }
    for role in REGISTER_OPERAND_ROLES:
        if role not in operands:
            continue
        if operands[role] is None:
            continue
        if role == "rs1" and operand_kind(form, "rs1") == "imm":
            continue
        if role not in register_roles:
            raise ValueError(f"{form.mnemonic}: instruction has no {role} register operand")
    for group in groups:
        roles = _register_roles_for_group(form, group)
        provided = [
            int(operands[role])
            for role in roles
            if role in operands and operands[role] is not None
        ]
        if len(set(provided)) > 1:
            raise ValueError(
                f"{form.mnemonic}: register group {group!r} received conflicting canonical roles"
            )
        # A group whose name already is the canonical role (e.g. ``rd``) is
        # supplied through that same key; only non-canonical group spellings
        # (e.g. compressed ``rd_p`` / ``c_sreg1``) can collide with a
        # canonical-role alias, so check ambiguity only for those.
        if (
            group not in REGISTER_OPERAND_ROLES
            and group in operands
            and operands[group] is not None
            and provided
        ):
            raise ValueError(
                f"{form.mnemonic}: supply register group {group!r} either by its group or canonical roles"
            )
    named = {role: value for role, value in operands.items() if value is not None}
    # Match encode_instruction(): FLI's rs1 spelling is a literal selector,
    # not a GPR operand.  Normalize the alias before changed-bit selection so
    # rewrite_instruction_operands cannot quietly replace it with selector 0.
    if operand_kind(form, "rs1") == "imm" and "rs1" in named:
        selector = int(named.pop("rs1"))
        if "imm" in named and int(named["imm"]) != selector:
            raise ValueError(
                f"{form.mnemonic}: rs1 selector conflicts with 'imm'"
            )
        named.setdefault("imm", selector)
    if mnemonic.replace(".", "_").lower() in _CM_INDEX_BOUNDS:
        selector = named.get("c_index", named.get("imm"))
        if selector is not None and not _cm_index_valid(mnemonic, selector):
            raise ValueError(f"{mnemonic}: c_index {selector} is out of range")
    primary_immediate_group = canonical_immediate_field(form)
    if primary_immediate_group == "imm20" and "imm" in named:
        named["imm"] = encoded_immediate_for_metadata(form, named["imm"])
    named_noncanonical = set(named) - {"imm"}
    changed = 0
    for field in form.variable_fields:
        logical = logical_immediate_for_form_field(form, field)
        group = logical or operand_group(field)
        register_roles = _register_roles_for_group(form, group)
        if group in named or (
            group == primary_immediate_group and "imm" in named
        ) or operand_kind(form, group) in named_noncanonical or any(
            role in named_noncanonical for role in register_roles
        ):
            changed |= _field_instruction_mask_for_form(form, field)
    if not changed:
        return word
    extra = {
        key: int(value)
        for key, value in named.items()
        if key not in {"rd", "rs1", "rs2", "rs3", "imm"}
    }
    # Preserve any secondary immediate fields (e.g. Zibi's ``imm5`` selector)
    # from the source word when only the canonical immediate is rewritten.
    original = decode_form(form, word)
    for group in immediate_groups:
        if group == primary_immediate_group:
            continue
        extra.setdefault(group, int(original[group]))
    encoded = encode_instruction(
        mnemonic,
        rd=named.get("rd"),
        rs1=named.get("rs1"),
        rs2=named.get("rs2"),
        rs3=named.get("rs3"),
        immediate=named.get("imm"),
        **extra,
    )
    return (word & ~changed) | (encoded & changed)


def _group_value_mask(group: str) -> int:
    from .riscv_catalog import OFFICIAL_FIELD_BIT_RANGES

    if group in SCATTERED_IMMEDIATE_LAYOUT:
        width = max(msb for msb, _, _ in SCATTERED_IMMEDIATE_LAYOUT[group]) + 1
    else:
        msb, lsb = OFFICIAL_FIELD_BIT_RANGES[group]
        width = msb - lsb + 1
    return (1 << width) - 1


# 操作数槽落在哪个寄存器堆，是纯编码事实，必须只有一个 owner：
# LOAD-FP / STORE-FP 的基址在 GPR、数据在 FPR；MADD/MSUB/NMSUB/NMADD 四个槽全在 FPR；
# OP-FP 由 funct5 决定方向。沿用这些 opcode 的新指令族不需要在这里增加条目。
_FP_MAJOR_OPCODES = frozenset({0x07, 0x27, 0x43, 0x47, 0x4B, 0x4F, 0x53})
_OP_FP_GPR_DESTINATION_FUNCT5 = frozenset({0b10100, 0b11000, 0b11100})  # 比较 / 浮点→整数 / fmv.x、fclass
_OP_FP_GPR_SOURCE_FUNCT5 = frozenset({0b11010, 0b11110})  # 整数→浮点 / fmv.f.x
_OP_FP_GPR_PAIR_SOURCE_FUNCT5 = frozenset({0b10110})  # fmvp.d.x / fmvp.q.x


def register_domain(form, role: str) -> str | None:
    """操作数槽所在的寄存器域（``gpr`` / ``fpr``），按官方编码字段判定。

    这是全仓库判断「读写哪个寄存器堆」的唯一事实来源：生成侧用它填
    ``InstructionMeta`` 的域字段与选 observer，RVEMI 用它取等值见证，
    效果 schema 用它定 writeback。任何一处自己推一遍都会与另外两处失配。
    """
    if operand_kind(form, role) == "imm":
        return None
    opcode = form.match & 0x7F
    if form.encoding_length_bytes == 2:
        return _compressed_register_domain(form, role)
    if opcode not in _FP_MAJOR_OPCODES:
        # OP-V scalar move forms reuse the integer-looking ``rd``/``rs1``
        # fields for FPRs.  Their fixed funct3 and operand shape distinguish
        # them without a mnemonic table; all other OP-V carriers stay GPR.
        if opcode == 0x57:
            funct3 = (form.match >> 12) & 0x7
            fields = set(operand_groups(form))
            if role == "rd" and funct3 == 0x1 and "vs2" in fields and "vd" not in fields:
                return "fpr"
            if role == "rs1" and funct3 == 0x5 and "vd" in fields:
                return "fpr"
        return "gpr"
    if opcode == 0x07:
        return "fpr" if role == "rd" else "gpr"
    if opcode == 0x27:
        if "vs3" in operand_groups(form):
            return "gpr"
        return "fpr" if role == "rs2" else "gpr"
    if opcode != 0x53:
        return "fpr"
    funct5 = (form.match >> 27) & 0x1F
    if role == "rd":
        return "gpr" if funct5 in _OP_FP_GPR_DESTINATION_FUNCT5 else "fpr"
    if role == "rs1":
        return (
            "gpr"
            if funct5 in _OP_FP_GPR_SOURCE_FUNCT5 | _OP_FP_GPR_PAIR_SOURCE_FUNCT5
            else "fpr"
        )
    if role == "rs2" and funct5 in _OP_FP_GPR_PAIR_SOURCE_FUNCT5:
        return "gpr"
    return "fpr"


def _compressed_register_domain(form, role: str) -> str:
    """压缩象限 0/2 的浮点访存：funct3 与 xlen 决定数据槽的寄存器堆。

    c.fld/c.fsd（rv32 另有 c.flw/c.fsw）的数据在 FPR、基址在 GPR；其余压缩
    form 全在 GPR。地址角色（rs1）永远是 GPR，所以只有数据槽需要判定。
    """
    logical_role = (
        "rs2"
        if role == "c_rs2" or "rs2" in role
        else "rd"
        if "rd" in role
        else "rs1"
        if "rs1" in role
        else role
    )
    quadrant = form.match & 0x3
    funct3 = (form.match >> 13) & 0x7
    if quadrant not in (0x0, 0x2) or logical_role == "rs1":
        return "gpr"
    if funct3 == 0x1 and logical_role == "rd":  # c.fld / c.fldsp
        return "fpr"
    if funct3 == 0x5 and logical_role == "rs2":  # c.fsd / c.fsdsp
        return "fpr"
    if form.xlen == "rv32":
        if funct3 == 0x3 and logical_role == "rd":  # c.flw / c.flwsp
            return "fpr"
        if funct3 == 0x7 and logical_role == "rs2":  # c.fsw / c.fswsp
            return "fpr"
    return "gpr"

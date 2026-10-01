from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import isqrt
import re
from typing import Mapping

from ..riscv_encoding import (
    VECTOR_VTYPE_LMUL_ENCODING,
    VECTOR_VTYPE_SEW_ENCODING,
    encode_form,
    encode_register_field_value,
    immediate_fields,
    is_register_operand_group,
    operand_groups,
    source_roles,
)
from ..riscv_catalog import OFFICIAL_ALL_CATALOG_FORMS, OFFICIAL_FIELD_BIT_RANGES


ROUNDING_MODES: tuple[tuple[str, int], ...] = (
    ("rne", 0),
    ("rtz", 1),
    ("rdn", 2),
    ("rup", 3),
    ("rmm", 4),
    ("dyn-rne", 7),
)

_STACK_RLIST_REGISTERS = (1, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
STACK_RLIST_REGISTERS_RV64 = {
    **{rlist: _STACK_RLIST_REGISTERS[:rlist - 3] for rlist in range(4, 15)},
    15: _STACK_RLIST_REGISTERS,
}

STACK_ADJ_RV64 = {
    **dict.fromkeys(range(4, 6), (16, 32, 48, 64)),
    **dict.fromkeys(range(6, 8), (32, 48, 64, 80)),
    **dict.fromkeys(range(8, 10), (48, 64, 80, 96)),
    **dict.fromkeys(range(10, 12), (64, 80, 96, 112)),
    **dict.fromkeys(range(12, 14), (80, 96, 112, 128)),
    14: (96, 112, 128, 144),
    15: (112, 128, 144, 160),
}
STACK_ADJ_RV32 = {
    **dict.fromkeys(range(4, 8), (16, 32, 48, 64)),
    **dict.fromkeys(range(8, 12), (32, 48, 64, 80)),
    **dict.fromkeys(range(12, 15), (48, 64, 80, 96)),
    15: (64, 80, 96, 112),
}

_ROUNDING_MODE_BY_ENCODING = {encoding: name for name, encoding in ROUNDING_MODES if encoding is not None and encoding != 7}


def branch_condition_holds(
    left: int,
    right: int,
    operation: str | None,
    *,
    xlen: int = 64,
) -> bool | None:
    """Evaluate a decoded branch predicate for the concrete XLEN."""

    if type(xlen) is not int or xlen not in {32, 64}:
        raise ValueError("branch XLEN must be 32 or 64")
    width = xlen
    mask = (1 << width) - 1
    left_u = int(left) & mask
    right_u = int(right) & mask
    left_s = left_u - (1 << width) if left_u & (1 << (width - 1)) else left_u
    right_s = right_u - (1 << width) if right_u & (1 << (width - 1)) else right_u
    return {
        "beq": left_u == right_u,
        "bne": left_u != right_u,
        "blt": left_s < right_s,
        "bge": left_s >= right_s,
        "bltu": left_u < right_u,
        "bgeu": left_u >= right_u,
    }.get(operation)


def stack_slot_offsets(
    registers: tuple[int, ...], stack_adj: int, *, xlen: int = 64
) -> dict[int, int]:
    if type(xlen) is not int or xlen not in {32, 64}:
        raise ValueError("stack-transfer XLEN must be 32 or 64")
    width = xlen // 8
    base = stack_adj - (len(registers) * width)
    return {
        register: base + (index * width)
        for index, register in enumerate(registers)
    }


def stack_seed_value(index: int, register: int) -> int:
    return ((0x51 + index) << 56) | ((register & 0xFF) << 48) | (0x0102030405060708 + index)


def stack_shape(operands: Mapping[str, int], *, xlen: int = 64) -> tuple[tuple[int, ...], int]:
    rlist = operands.get("c_rlist", -1)
    spimm = operands.get("c_spimm", -1)
    if type(rlist) is not int or type(spimm) is not int or type(xlen) is not int or xlen not in {32, 64}:
        raise ValueError("stack-transfer operands must be native integers")
    registers = STACK_RLIST_REGISTERS_RV64.get(rlist)
    adjustments = (STACK_ADJ_RV32 if xlen == 32 else STACK_ADJ_RV64).get(rlist)
    if registers is None or adjustments is None or spimm not in range(len(adjustments)):
        raise ValueError("stack-transfer operands name an unsupported rlist/spimm pair")
    return registers, int(adjustments[spimm])


@dataclass(frozen=True)
class IntegerToFpBoundary:
    encoded_u64: int
    boundary_class: str
    rounding_relation: str


# ``rs3`` shares the funct7 high bits in the 32-bit encoding and is therefore
# not a recoverable fixed register slot from an encoding string alone.  The
# lower canonical slots are unambiguous (notably atomic ``rs2=0``); ``vm`` is
# the one vector control bit that also has fixed reserved encodings.
_REGISTER_FIELD_NAMES = ("rd", "rs1", "rs2", "vm", "nf", "mew")
_FIXED_FIELD_BIT_RANGES = {"mew": (28, 28)}


def _fixed_field_range(field: str) -> tuple[int, int]:
    return _FIXED_FIELD_BIT_RANGES[field] if field in _FIXED_FIELD_BIT_RANGES else OFFICIAL_FIELD_BIT_RANGES[field]

FIXED_FIELD_WITNESS_PREFIX = "fixed-field:"


@cache
def fixed_field_witness_pairs(form) -> tuple["ControlWitness", ...]:
    """Return executable raw-word witnesses for each fixed register slot.

    The legal word is a plain encoding of the form; the illegal word uses the
    first alternate field value that does not collide with an official form.
    A small register assignment sweep avoids an alias at the all-zero
    baseline.  A colliding mutation is a legal different instruction, not a
    reserved encoding, so it is skipped.  Only frozen encoding facts are used;
    no issue ledger, defect ID or tool name is consulted.
    """

    try:
        width = int(form.encoding_length_bytes) * 8
        encoding = form.encoding
        if width not in {16, 32} or len(encoding) < width:
            return ()
        bits = encoding[-width:]
        groups = tuple(operand_groups(form))
        variable_roles = {
            role
            for group in groups
            if is_register_operand_group(form, group)
            for role in _REGISTER_FIELD_NAMES
            if role in str(group)
        }
        boundaries = []
        for field in _REGISTER_FIELD_NAMES:
            if field in variable_roles or field in groups:
                continue
            if field == "vm" and (int(form.match) & 0x7F) not in {0x07, 0x27, 0x57, 0x77}:
                continue
            if field in {"nf", "mew"} and (int(form.match) & 0x7F) not in {0x07, 0x27}:
                continue
            msb, lsb = _fixed_field_range(field)
            if msb >= width or lsb < 0:
                continue
            positions = tuple(bits[width - 1 - bit] for bit in range(msb, lsb - 1, -1))
            if positions and all(bit in "01" for bit in positions):
                boundaries.append(ControlWitnessBoundary("fixed-field", field, int("".join(positions), 2)))
        boundaries = tuple(boundaries)
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    if not boundaries:
        return ()
    try:
        width = int(form.encoding_length_bytes) * 8
        if width not in {16, 32}:
            return ()
        baseline = {name: 0 for name in operand_groups(form)}
        assignments = [baseline]
        for name in operand_groups(form):
            if not is_register_operand_group(form, name) and name not in {
                "vd", "vs1", "vs2", "vs3"
            }:
                continue
            for value in (1, 31):
                candidate = dict(baseline)
                candidate[name] = value
                try:
                    encode_form(form, candidate)
                except (KeyError, TypeError, ValueError):
                    continue
                assignments.append(candidate)
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    witnesses: list[ControlWitness] = []
    for boundary in boundaries:
        msb, lsb = _fixed_field_range(boundary.field)
        span = msb - lsb + 1
        field_mask = ((1 << span) - 1) << lsb
        for values in assignments:
            legal_word = encode_form(form, values)
            for illegal_value in range(1 << span):
                if illegal_value == boundary.value:
                    continue
                illegal_word = (legal_word & ~field_mask) | (illegal_value << lsb)
                if not 0 <= illegal_word < (1 << width) or any(
                    candidate.encoding_length_bytes * 8 == width
                    and (
                        candidate.xlen == form.xlen
                        or "shared" in {candidate.xlen, form.xlen}
                    )
                    and (illegal_word & candidate.mask) == candidate.match
                    for candidate in OFFICIAL_ALL_CATALOG_FORMS
                ):
                    continue
                witnesses.append(ControlWitness(boundary, illegal_word))
                break
            if witnesses and witnesses[-1].boundary == boundary:
                break
    return tuple(witnesses)


CONTROL_WITNESS_PREFIX = "control-witness:"


def control_witness_disabled_extensions(label: str) -> frozenset[str]:
    """Extensions the execution profile must drop for one witness label."""
    label = str(label)
    if any(
        label == axis or label.startswith(axis + ":")
        for axis in ("no-c-ialign", "jalr-target:bit1")
    ):
        return frozenset({"c"})
    return frozenset()


@dataclass(frozen=True)
class ControlWitnessBoundary:
    """One derived control/definedness axis for a catalog form.

    ``axis`` names the architectural edge (``jalr-target:bit0``,
    ``jalr-target:bit1``, ``no-c-ialign``, ``link-source-alias``, ``far-jump``,
    ``fence-i-refetch``, ``csr-warl``, ``reserved-shamt``, ``reserved-rm``,
    ``illegal-encoding``);
    ``field``/``value`` carry the concrete operand that realizes it.  The
    label is deterministic so materialize can look the witness back up from
    the intent's boundary class.
    """

    axis: str
    field: str = ""
    value: int = 0

    def label(self) -> str:
        if self.field:
            return f"{self.axis}:{self.field}"
        if self.value:
            return f"{self.axis}:{self.value}"
        return self.axis


@dataclass(frozen=True)
class ControlWitness:
    """One executable raw-word legality pair for a control/definedness axis.

    ``illegal_word`` is the perturbed encoding whose expected outcome is the
    trap named by ``expected_outcome``.  The extra facts a
    control pair needs to become executable are carried explicitly:
    ``seed`` names register roles seeded relative to the risk instruction's
    own address (delta values).
    """

    boundary: ControlWitnessBoundary
    illegal_word: int
    expected_outcome: str = "illegal-instruction"
    seed: tuple[tuple[str, int], ...] = ()
    pc_delta_seed: tuple[tuple[str, int], ...] = ()


def _witness_word(form, values: dict[str, int]) -> int | None:
    """Build the baseline raw word for one pinned catalog form."""
    try:
        width = int(form.encoding_length_bytes) * 8
        if width not in {16, 32}:
            return None
        complete = {name: 0 for name in operand_groups(form)}
        complete.update(values)
        return encode_form(form, complete)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _witness_register_values(form) -> dict[str, int]:
    """Working register seeds for the witness's own operand slots.

    Register groups (including compressed ``*_n0`` slots that reserve zero)
    are seeded with a concrete non-zero GPR so the raw word stays inside the
    form's own encoding space; the materializer overrides the seeded source
    slot for the target-bearing axes.
    """
    values: dict[str, int] = {}
    for name in operand_groups(form):
        if not is_register_operand_group(form, name):
            continue
        if name == "rd":
            values[name] = 0
        elif name == "c_rs1_n0":
            values[name] = encode_register_field_value(name, 2)
        elif "rs1" in name or name == "c_rs2":
            values[name] = encode_register_field_value(name, 8)
        else:
            values[name] = encode_register_field_value(name, 9)
    return values


def _control_jump_family(form) -> str | None:
    """Structural control-jump family (``jalr`` or ``jal``) from encoding bits."""
    try:
        width = int(getattr(form, "encoding_length_bytes")) * 8
        match = int(getattr(form, "match", 0))
    except (AttributeError, TypeError, ValueError):
        return None
    if width == 32:
        if match & 0x7F == 0x67:
            return "jalr"
        if match & 0x7F == 0x6F:
            return "jal"
        return None
    if width == 16:
        funct3 = (match >> 13) & 0x7
        if funct3 == 0b100:
            return "jalr"
        if funct3 == 0b001:
            return "jal"
    return None


def _jalr_target_bit_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """JALR target bit0/bit1 raw-word pair, derived from the encoding layout.

    bit0=1 is the architectural clear edge: the odd-target seed is realized
    by the layout binder's preserved low bit
    (``jump-target:odd`` owns the normal lane).  bit1=1 under IALIGN=32 is
    the no-C misaligned trap: JALR clears only bit 0, so a
    target whose bit 1 remains set must raise the instruction-address-
    misaligned exception.
    """
    if (
        schema is None
        or schema.kind != "control-jump"
        or schema.operation not in {"jalr", "jr"}
        or int(getattr(form, "encoding_length_bytes", 0)) != 4
        or _control_jump_family(form) != "jalr"
    ):
        return ()
    values = _witness_register_values(form)
    legal_word = _witness_word(form, values)
    if legal_word is None:
        return ()
    # imm12 bit 0 is the target low-bit carrier for the 32-bit JALR layout.
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("jalr-target:bit0", "imm12", 1),
            illegal_word=legal_word | (1 << 20),
            expected_outcome="normal",
        ),
        ControlWitness(
            boundary=ControlWitnessBoundary("jalr-target:bit1", "imm12", 2),
            illegal_word=legal_word,
            expected_outcome="instruction-address-misaligned",
            pc_delta_seed=(("rs1", 2),),
        ),
    )


def _jalr_funct3_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Return the reserved JALR funct3 encoding used by translated paths."""
    if (
        schema is None
        or schema.kind != "control-jump"
        or schema.operation != "jalr"
        or int(getattr(form, "encoding_length_bytes", 0)) != 4
        or _control_jump_family(form) != "jalr"
    ):
        return ()
    legal_word = _witness_word(form, {"rd": 1, "rs1": 0, "imm12": 0})
    if legal_word is None:
        return ()
    illegal_word = (legal_word & ~(0x7 << 12)) | (0x1 << 12)
    if any(
        illegal_word & candidate.mask == candidate.match
        for candidate in OFFICIAL_ALL_CATALOG_FORMS
        if candidate.encoding_length_bytes == 4
    ):
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("illegal-encoding", "jalr-funct3", 1),
            illegal_word=illegal_word,
        ),
    )


def _cbo_funct7_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Return the reserved CBO funct7 encoding used by translated paths."""
    if (
        schema is None
        or schema.kind != "cache-block"
        or not str(form.mnemonic).startswith("cbo_")
        or int(getattr(form, "encoding_length_bytes", 0)) != 4
    ):
        return ()
    legal_word = _witness_word(form, {"rs1": 10})
    if legal_word is None:
        return ()
    illegal_word = legal_word | (0x3 << 25)
    if any(
        illegal_word & candidate.mask == candidate.match
        for candidate in OFFICIAL_ALL_CATALOG_FORMS
        if candidate.encoding_length_bytes == 4
    ):
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("illegal-encoding", "cbo-funct7", 3),
            illegal_word=illegal_word,
        ),
    )


def _no_c_ialign_pair_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Compressed/uncompressed double witness for the same control edge.

    With the C extension disabled (IALIGN=32), the 16-bit form is an illegal
    instruction while its 32-bit twin stays legal.  The twin is found
    structurally: the 32-bit official form in the same encoding family
    (``jalr`` for C.JALR/C.JR/C.RET, ``jal`` for C.JAL).  ``legal_word`` is
    the twin's baseline and ``illegal_word`` the compressed baseline.
    """
    if (
        schema is None
        or schema.kind != "control-jump"
        or int(getattr(form, "encoding_length_bytes", 0)) != 2
    ):
        return ()
    family = _control_jump_family(form)
    if family is None:
        return ()
    twin = next(
        (
            candidate
            for candidate in OFFICIAL_ALL_CATALOG_FORMS
            if candidate.encoding_length_bytes == 4
            and _control_jump_family(candidate) == family
            and str(getattr(candidate, "xlen", "shared"))
            in {"shared", str(getattr(form, "xlen", "shared"))}
        ),
        None,
    )
    if twin is None:
        return ()
    legal_word = _witness_word(twin, _witness_register_values(twin))
    illegal_word = _witness_word(form, _witness_register_values(form))
    if legal_word is None or illegal_word is None:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("no-c-ialign", form.mnemonic, 0),
            illegal_word=illegal_word,
        ),
    )


def _link_source_alias_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Link/source alias pair: rd==x1 overlapping rs1.

    A link-form jump reads its source before writing the link register; the
    alias spelling (rs1=x1, or the compressed implicit link with rs1=x1) is
    the source-before-write edge.  The pair has no trap side -- both spellings
    are legal -- so the normal lane is owned by ``jump-link:source-alias`` and
    this descriptor documents the derived alias word.
    """
    if schema is None or schema.kind != "control-jump" or schema.writeback != "gpr":
        return ()
    values = _witness_register_values(form)
    if "rd" in values and "rs1" in values:
        # Explicit JALR: rd=x1 with rs1=x1 vs a distinct rs1.
        values["rd"] = encode_register_field_value("rd", 1)
        legal = dict(values)
        legal["rs1"] = encode_register_field_value("rs1", 2)
        illegal = dict(values)
        illegal["rs1"] = encode_register_field_value("rs1", 1)
    else:
        # Compressed implicit link (C.JALR): rs1=x1 is the alias spelling.
        source = next(
            (name for name in operand_groups(form) if "rs1" in name),
            None,
        )
        if source is None:
            return ()
        legal = dict(values)
        legal[source] = encode_register_field_value(source, 2)
        illegal = dict(values)
        illegal[source] = encode_register_field_value(source, 1)
    legal_word = _witness_word(form, legal)
    illegal_word = _witness_word(form, illegal)
    if legal_word is None or illegal_word is None or legal_word == illegal_word:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("link-source-alias"),
            illegal_word=illegal_word,
            expected_outcome="normal",
        ),
    )


def _far_jump_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Far-jump edge: a target beyond the imm12 range (``>4KB``).

    A JALR carries a signed 12-bit displacement, so a far target is not
    expressible in one indirect jump -- the pair fails closed for imm12
    layouts.  JAL's 20-bit displacement reaches it; the legal word encodes
    displacement 4096.  The layout that actually places a
    target that far away is a materializer concern, not a word fact.
    """
    if schema is None or schema.kind != "control-jump":
        return ()
    if "jimm20" in immediate_fields(form):
        legal_word = _witness_word(form, {"jimm20": 4096})
        if legal_word is None:
            return ()
        return (
            ControlWitness(
                boundary=ControlWitnessBoundary("far-jump", "jimm20", 4096),
                illegal_word=legal_word,
                expected_outcome="normal",
            ),
        )
    return ()


def _fence_i_refetch_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """FENCE.I store -> fence -> second-fetch program shape (T-7 input).

    The instruction-fetch cache contract is exercised by a program that stores
    into instruction memory, executes FENCE.I, then fetches the modified
    location again.  The descriptor names the fence word;
    the executable shape is the existing ``instruction-memory:code-patch``
    route, which already owns the store/second-fetch scaffold.
    """
    if (
        schema is None
        or schema.kind != "instruction-memory"
        or schema.operation != "fence-fetch"
    ):
        return ()
    legal_word = _witness_word(form, {name: 0 for name in operand_groups(form)})
    if legal_word is None:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("fence-i-refetch"),
            illegal_word=legal_word,
            expected_outcome="normal",
        ),
    )


def _csr_warl_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """CSR/WARL legal-set pair, derived from the frozen CSR directory.

    The legal set comes from the CSR directory structure: writable addresses
    accept a write, read-only addresses (top two address bits 11) reject one
    with an illegal-instruction trap, and the FRM/FCSR legal set excludes the
    reserved rounding modes 5/6.  No per-CSR table is consulted.
    """
    if schema is None or schema.kind != "csr-access":
        return ()
    from ..riscv_csrs import csr_is_read_only
    from ..riscv_csrs import OFFICIAL_CSRS

    values = _witness_register_values(form)
    groups = set(operand_groups(form))
    immediate_form = "zimm5" in groups
    by_name = {
        str(name).strip('"'): address
        for address, name in OFFICIAL_CSRS.items()
    }
    frm = by_name.get("frm")
    writable = next(
        (
            address
            for address, _name in sorted(OFFICIAL_CSRS.items())
            if not csr_is_read_only(address)
        ),
        None,
    )
    read_only = next(
        (
            address
            for address, _name in sorted(OFFICIAL_CSRS.items())
            if csr_is_read_only(address)
        ),
        None,
    )
    if writable is None or read_only is None:
        return ()

    def word_for(address: int, argument: int) -> int | None:
        row = dict(values)
        row["csr"] = address
        if immediate_form:
            row["zimm5"] = argument
        elif "rs1" in row:
            row["rs1"] = argument
        return _witness_word(form, row)

    def source_seed(argument: int) -> tuple[tuple[str, int], ...]:
        # Register-form CSR writes encode a register number, not its value.
        # Keep the same concrete value in initial state; otherwise the raw
        # witness says ``rs1=xN`` while execution silently supplies xN=0.
        return () if immediate_form else (("rs1", int(argument)),)

    legal_word = word_for(writable, 1)
    illegal_word = word_for(read_only, 1)
    if legal_word is None or illegal_word is None or legal_word == illegal_word:
        return ()
    rows = [
        ControlWitness(
            boundary=ControlWitnessBoundary("csr-warl", "read-only-write", 1),
            illegal_word=illegal_word,
            seed=source_seed(1),
        ),
    ]
    if frm is not None:
        frm_legal = word_for(frm, 0)
        reserved_word = word_for(frm, 5)
        if frm_legal is not None and reserved_word is not None and reserved_word != frm_legal:
            rows.append(
                ControlWitness(
                    boundary=ControlWitnessBoundary("csr-warl", "reserved-frm", 5),
                    illegal_word=reserved_word,
                    expected_outcome="normal",
                    seed=source_seed(5),
                )
            )
        if form.mnemonic == "csrrw":
            fcsr = by_name.get("fcsr")
            fcsr_word = word_for(fcsr, 5) if fcsr is not None else None
            if fcsr_word is not None:
                rows.append(
                    ControlWitness(
                        boundary=ControlWitnessBoundary("csr-warl", "reserved-frm", 5),
                        illegal_word=fcsr_word,
                        expected_outcome="normal",
                        seed=(("rs1", (5 << 5) | 0x1F),),
                    )
                )
    return tuple(rows)


def _reserved_shamt_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Return the raw reserved word-shift encoding when the form owns it."""
    if (
        schema is None
        or schema.kind != "integer-shift"
        or form.extension_group in {"xtheadbb", "xtheadbs"}
    ):
        return ()
    fields = immediate_fields(form)
    if not fields:
        return ()
    legal_word = _witness_word(form, {fields[0]: 1})
    if form.xlen == "rv64" and fields[0] == "shamtw":
        illegal_word = None if legal_word is None else legal_word | (1 << 25)
    elif int(getattr(schema, "width", 64) or 64) == 32:
        illegal_word = _witness_word(form, {fields[0]: 32})
    else:
        return ()
    if legal_word is None or illegal_word is None or legal_word == illegal_word:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("reserved-shamt", fields[0], 32),
            illegal_word=illegal_word,
        ),
    )


def _reserved_rm_witnesses(form) -> tuple[ControlWitness, ...]:
    """Reserved rounding-mode pair: rm=5/6 is an illegal FP encoding."""
    if "rm" not in set(operand_groups(form)):
        return ()
    legal_word = _witness_word(form, {"rm": 0})
    illegal_word = _witness_word(form, {"rm": 5})
    if legal_word is None or illegal_word is None or legal_word == illegal_word:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("reserved-rm", "rm", 5),
            illegal_word=illegal_word,
        ),
    )


def _rd_x0_illegal_witnesses(form, schema) -> tuple[ControlWitness, ...]:
    """Derive the canonical OP-family ``rd=x0`` predecode witness.

    The four layouts share the same architectural edge: validation must run
    before the hard-wired-zero destination is discarded.  Use the smallest
    catalog owner for each opcode shape and mutate only the reserved decode
    bits, leaving the instruction's register fields at x0.
    """
    if schema is None or schema.kind not in {"integer-writeback", "integer-shift"}:
        return ()
    groups = operand_groups(form)
    opcode = int(form.match) & 0x7F
    legal_word = None
    illegal_word = None
    if (
        schema.kind == "integer-writeback"
        and schema.operation == "add"
        and groups == ("rd", "rs1", "rs2")
        and int(form.match) == opcode
        and opcode in {0x33, 0x3B}
    ):
        legal_word = _witness_word(form, {})
        if legal_word is not None:
            illegal_word = (legal_word & ~((0x7F << 25) | (0x7 << 12))) | (2 << 25) | (7 << 12)
    elif (
        schema.kind == "integer-shift"
        and schema.operation == "shift-left"
        and groups == ("rd", "rs1", "shamtd")
        and int(form.match) == 0x1013
    ):
        legal_word = _witness_word(form, {"shamtd": 1})
        if legal_word is not None:
            illegal_word = legal_word | (1 << 26)
    elif (
        schema.kind == "integer-writeback"
        and schema.operation == "add"
        and groups == ("rd", "rs1", "imm12")
        and int(form.match) == opcode == 0x1B
    ):
        legal_word = _witness_word(form, {"imm12": 1, "rs1": 10})
        if legal_word is not None:
            illegal_word = (legal_word & ~(0x7 << 12)) | (0x7 << 12)
    if legal_word is None or illegal_word is None or legal_word == illegal_word:
        return ()
    return (
        ControlWitness(
            boundary=ControlWitnessBoundary("illegal-encoding", "rd-x0"),
            illegal_word=illegal_word,
        ),
    )


@cache
def control_witness_pairs(form, schema=None) -> tuple[ControlWitness, ...]:
    """Every derived control/definedness raw-word witness pair for one form.

    All axes fail closed with an empty tuple when the form cannot carry them;
    only the frozen catalog structure and the effect schema are consulted (no
    per-instruction table, no issue ledger).  ``legal_word`` is always the
    baseline encoding and ``illegal_word`` the perturbed one, matching the
    :func:`fixed_field_witness_pairs` contract so the same materialize path
    can consume both families.
    """
    if schema is None:
        return ()
    return (
        _jalr_target_bit_witnesses(form, schema)
        + _jalr_funct3_witnesses(form, schema)
        + _cbo_funct7_witnesses(form, schema)
        + _no_c_ialign_pair_witnesses(form, schema)
        + _link_source_alias_witnesses(form, schema)
        + _far_jump_witnesses(form, schema)
        + _fence_i_refetch_witnesses(form, schema)
        + _csr_warl_witnesses(form, schema)
        + _reserved_shamt_witnesses(form, schema)
        + _reserved_rm_witnesses(form)
        + _rd_x0_illegal_witnesses(form, schema)
    )


@dataclass(frozen=True)
class VectorStateBoundary:
    """Descriptor-derived classification of one Vector state edge.

    This is deliberately only a *boundary* description.  In particular,
    ``vstart`` is a CSR obligation and not evidence that a restart or a
    partial-trap transition was observed.  Unknown/reserved spellings are
    rejected by :func:`vector_state_boundary` instead of being normalised to a
    convenient default.
    """

    axis: str
    sew: int | None = None
    lmul: Fraction | None = None
    vstart: int | None = None
    # A vill row names the vtype.vill state edge (reserved
    # vsew encoding forced by the setup, vl/vstart cleared by hardware).
    # vill is a *state* edge, not a concrete SEW/LMUL; sew/lmul stay None.
    vill: bool = False
    value: str | None = None


_VECTOR_VTYPE_BOUNDARY_RE = re.compile(
    r"^sew(?P<sew>8|16|32|64|128)-(?P<lmul>mf8|mf4|mf2|m1|m2|m4|m8)$"
)


def vector_state_boundary(boundary_class: str) -> VectorStateBoundary | None:
    """Parse the small Vector state boundary vocabulary used by RVGEN.

    The parser consumes only the boundary label emitted by the descriptor
    path; it does not infer a VLEN, an active-element trace, or an execution
    result.  ``vector-vl:vlmax`` therefore carries ``vl=None`` (the concrete
    VLMAX depends on the configured VLEN/SEW/LMUL), while retaining ``value``
    so callers can distinguish it from an unrecognised label.
    """

    if not isinstance(boundary_class, str):
        return None
    text = boundary_class
    if text.startswith("vector-vl:"):
        value = text.partition(":")[2]
        if value in {"zero", "one", "vlmax"}:
            return VectorStateBoundary("vl", value=value)
        return None
    if text.startswith("vector-vtype:"):
        value = text.partition(":")[2]
        if value == "vill":
            return VectorStateBoundary("vtype", vill=True)
        match = _VECTOR_VTYPE_BOUNDARY_RE.fullmatch(value)
        if match is None:
            return None
        lmul_text = match.group("lmul")
        lmul = (
            Fraction(1, int(lmul_text[2:]))
            if lmul_text.startswith("mf")
            else Fraction(int(lmul_text[1:]), 1)
        )
        return VectorStateBoundary(
            "vtype",
            sew=int(match.group("sew")),
            lmul=lmul,
        )
    if text.startswith("vector-vstart:"):
        value = text.partition(":")[2]
        if value in {"zero", "one"}:
            return VectorStateBoundary("vstart", vstart=int(value == "one"))
        if value.isascii() and value.isdecimal() and value == str(int(value)):
            return VectorStateBoundary("vstart", vstart=int(value))
        return None
    if text.startswith("vector-mask:"):
        value = text.partition(":")[2]
        if value in {"all-off", "all-on"}:
            return VectorStateBoundary("mask")
        # ``vm`` is the encoding bit, not an all-ones mask guarantee: vm=1
        # means unmasked, while vm=0 selects the mask carrier.
        if value == "vm:0":
            return VectorStateBoundary("mask")
        if value == "vm:1":
            return VectorStateBoundary("mask")
        return None
    if text.startswith("vector-tail:"):
        value = text.partition(":")[2]
        if value in {"agnostic", "undisturbed"}:
            return VectorStateBoundary("tail")
    return None


def compressed_codeword_disposition(
    form,
    schema,
    operands: Mapping[str, int],
) -> str | None:
    """Classify descriptor-local Zca HINT/non-canonical code points.

    ``riscv-opcodes`` records the carrier fields, while the Manual/Sail
    legality predicates constrain a few compressed values inside those
    carriers.  This helper keeps that small structural disposition in one
    place for both intent filtering and admission.  It does not classify
    reserved values (for example ``c.lui`` immediate zero) as traps: those
    remain explicit legality obligations owned by admission.
    """
    if (
        form is None
        or schema is None
        or int(getattr(form, "encoding_length_bytes", 0)) != 2
        or not isinstance(operands, Mapping)
    ):
        return None
    groups = set(operand_groups(form))
    if form.mnemonic == "c_add" and operands.get("rd_rs1_n0") == 0:
        return "hint"
    immediate = operands.get("c_imm6")
    if immediate is not None:
        immediate = int(immediate)
        if any(
            str(field).startswith("c_nzuimm6")
            for field in getattr(form, "variable_fields", ())
        ):
            # Zca shift amounts of zero are HINTs.  RV32's high shift bit is
            # a separate reserved/custom-extension boundary handled elsewhere.
            if str(getattr(schema, "kind", "")) == "integer-shift" and immediate == 0:
                return "hint"
        elif any(
            str(field).startswith("c_nzimm6")
            for field in getattr(form, "variable_fields", ())
        ) and str(getattr(schema, "kind", "")) == "integer-writeback":
            # C.NOP is the zero spelling; its non-zero siblings are HINTs.
            # Other c_nzimm6 integer forms use zero as their HINT spelling.
            operation = str(getattr(schema, "operation", ""))
            if operation == "nop" and immediate != 0:
                return "hint"
            if operation != "nop" and immediate == 0:
                return "hint"
    if {"rd_n2", "c_nzimm18"} <= groups:
        immediate = operands.get("c_nzimm18")
        destination = operands.get("rd_n2")
        if immediate is not None and destination is not None and int(immediate) != 0:
            if int(destination) == 0:
                return "hint"
            if int(destination) == 2:
                return "non-canonical"
    return None


def vector_vtype_codes(
    sew_bits: int,
    lmul: str | Fraction,
) -> tuple[int, int] | None:
    """Return the encoded SEW/LMUL fields for one concrete VTYPE edge."""
    if not isinstance(sew_bits, int) or isinstance(sew_bits, bool):
        return None
    sew_code = VECTOR_VTYPE_SEW_ENCODING.get(sew_bits)
    if sew_code is None:
        return None
    if isinstance(lmul, Fraction):
        def as_fraction(name: str) -> Fraction:
            return (
                Fraction(1, int(name[2:]))
                if name.startswith("mf")
                else Fraction(int(name[1:]), 1)
            )

        lmul_name = next(
            (name for name in VECTOR_VTYPE_LMUL_ENCODING if as_fraction(name) == lmul),
            None,
        )
    elif isinstance(lmul, str):
        lmul_name = lmul
    else:
        return None
    lmul_code = VECTOR_VTYPE_LMUL_ENCODING.get(lmul_name or "")
    return None if lmul_code is None else (sew_code, lmul_code)


def xlen_mask(xlen: int) -> int:
    """Return the architectural GPR mask for one concrete XLEN."""
    if not isinstance(xlen, int) or isinstance(xlen, bool):
        raise ValueError("XLEN must be an integer 32 or 64")
    if xlen not in {32, 64}:
        raise ValueError("XLEN must be 32 or 64")
    return (1 << xlen) - 1


def _bit_width(width: int, *, name: str) -> int:
    """Validate a native bit width shared by scalar boundary formulae."""
    if type(width) is not int or not 0 < width <= 64:
        raise ValueError(f"{name} must be a native integer between 1 and 64")
    return width


def integer_boundary_values(width: int) -> tuple[int, ...]:
    width = _bit_width(width, name="integer width")
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    return tuple(
        dict.fromkeys(
            value & mask
            for value in (
                0,
                1,
                -1,
                sign - 1,
                sign,
                sign + 1,
            )
        )
    )


def integer_source_boundary_values(width: int, source_width: int = 0) -> tuple[int, ...]:
    """Operation-neutral bit-pattern boundaries for one integer source."""
    width = _bit_width(width, name="integer width")
    if type(source_width) is not int or source_width < 0:
        raise ValueError("integer source width must be a non-negative native integer")
    mask = (1 << width) - 1
    alternating = sum(1 << bit for bit in range(0, width, 2))
    values = list(integer_boundary_values(width))
    values.extend((alternating, mask ^ alternating))
    for bit in (7, 8, 15, 16, 31, 32):
        if bit < width:
            values.extend(((1 << bit) - 1, 1 << bit, (1 << bit) + 1))
    if 0 < source_width < width:
        edge = 1 << source_width
        sign = 1 << (source_width - 1)
        values.extend((sign - 1, sign, edge - 1, edge, edge + 1))
    return tuple(dict.fromkeys(value & mask for value in values))


def atomic_sign_pair_boundaries(width: int) -> tuple[tuple[int, int], ...]:
    """Return one row for each signed memory/``rs2`` sign combination.

    The first value is the stored memory element and the second is the AMO
    source operand.  Zero and the signed minimum keep the memory sign edge
    explicit; one and all-ones do the same for ``rs2`` while retaining the
    exact ``0/-1`` and ``signed-min/1`` witnesses for signed min/max.
    """
    width = _bit_width(width, name="atomic sign-pair width")
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    return tuple(
        dict.fromkeys(
            (memory, source)
            for memory in (0, sign)
            for source in (1, mask)
        )
    )


def semantic_immediate_value(field: str | None, value: int | None) -> int | None:
    if field is None or value is None:
        return None
    if type(field) is not str or type(value) is not int:
        return None
    encoded = value
    if field in {"imm12", "imm12s"}:
        return _signed(encoded, 12)
    if field == "imm20":
        return _signed(encoded, 20) << 12
    if field == "bimm12":
        return _signed(encoded, 13)
    if field == "jimm20":
        return _signed(encoded, 21)
    if field == "simm5":
        return _signed(encoded, 5)
    if field in {"c_imm6", "c_nzimm10", "c_imm12", "c_bimm9"}:
        return _signed(encoded, {"c_imm6": 6, "c_nzimm10": 10, "c_imm12": 12, "c_bimm9": 9}[field])
    if field == "c_nzimm18":
        return _signed(encoded, 18)
    return encoded


def declared_gpr_writeback_value(
    form,
    schema,
    *,
    operands: Mapping[str, int],
    source_state: Mapping[str, int],
    pc: int = 0,
    rounding_mode: int | None = None,
) -> int | None:
    """Semantic GPR result derived from a declared form/operand/state row."""
    if getattr(schema, "writeback", None) != "gpr":
        return None
    declared_sources = tuple(
        int(value)
        for role in source_roles(form)
        for value in (_declared_source_value(source_state, role),)
        if value is not None
    )
    if schema.kind in {"fp-arith", "fp-fused"}:
        required_sources = 3 if schema.kind == "fp-fused" else 2
        declared_rm = _declared_rounding_mode(operands.get("rm"), rounding_mode)
        if len(declared_sources) < required_sources or declared_rm is None:
            return None
        return evaluate_fp_arithmetic(
            schema.precision,
            schema.operation,
            declared_sources,
            rounding_mode=declared_rm,
        )
    if schema.kind == "fp-compare":
        if len(declared_sources) < 2:
            return None
        return evaluate_fp_arithmetic(
            schema.precision,
            schema.operation,
            declared_sources,
            rounding_mode="rne",
        )
    if schema.kind == "fp-classify":
        return (
            evaluate_fp_classify(schema.precision, declared_sources[0])
            if len(declared_sources) == 1
            else None
        )
    immediate_field = getattr(schema, "immediate_field", None)
    if immediate_field is None:
        candidates = immediate_fields(form)
        immediate_field = candidates[0] if len(candidates) == 1 else None
    if schema.kind == "instruction-memory" and schema.operation == "table-jump":
        index = semantic_immediate_value(
            immediate_field,
            operands.get(immediate_field or ""),
        )
        if index is None or int(index) < 32:
            return None
        return (int(pc) + int(form.encoding_length_bytes)) & _mask(schema.width or 64)
    if schema.kind == "fp-convert":
        declared_rm = _declared_rounding_mode(
            operands.get("rm"),
            rounding_mode,
        )
        source_precision = schema.source_precision or schema.precision
        if schema.operation == "fp-to-int":
            if len(declared_sources) != 1 or declared_rm is None:
                return None
            result = evaluate_fp_to_int(
                source_precision,
                declared_sources[0],
                width=schema.width,
                signed=schema.signed,
                rounding_mode=declared_rm,
            )
            return _architectural_declared_gpr_result(form, schema, result)
        if schema.operation == "fp-to-int-mod":
            if len(declared_sources) != 1:
                return None
            return _architectural_declared_gpr_result(
                form,
                schema,
                evaluate_fp_to_int_mod_w_d(declared_sources[0]),
            )
        if schema.operation == "move" and len(declared_sources) == 1:
            if getattr(form, "xlen", "rv64") == "rv32" and schema.precision == 53:
                return (int(declared_sources[0]) >> 32) & 0xFFFFFFFF
            if getattr(form, "xlen", "") == "rv64" and schema.source_precision == 113:
                return (int(declared_sources[0]) >> 64) & _mask(64)
            width = IEEE_FORMAT_BY_PRECISION[source_precision][0]
            result = int(declared_sources[0]) & _mask(min(width, 64))
            if (
                schema.writeback == "gpr"
                and bool(getattr(schema, "signed", False))
                and width < (32 if getattr(form, "xlen", "") == "rv32" else 64)
            ):
                target_width = 32 if getattr(form, "xlen", "") == "rv32" else 64
                return _signed(result, width) & _mask(target_width)
            return result
    result = evaluate_gpr_writeback(
        kind=schema.kind,
        operation=schema.operation,
        width=schema.width,
        sources=declared_sources,
        immediate=semantic_immediate_value(
            immediate_field,
            operands.get(immediate_field or ""),
        ),
        memory_value=source_state.get("memory"),
        source_width=schema.source_width,
        pc=pc,
    )
    return (
        None
        if result is None
        else _architectural_declared_gpr_result(form, schema, int(result))
    )


def declared_exact_gpr_control_consumer_value(
    form,
    schema,
    *,
    operands: Mapping[str, int],
    source_state: Mapping[str, int],
    pc: int = 0,
    rounding_mode: int | None = None,
) -> int | None:
    """Declared GPR result that is stable enough to drive exact control consumers."""
    result = declared_gpr_writeback_value(
        form,
        schema,
        operands=operands,
        source_state=source_state,
        pc=pc,
        rounding_mode=rounding_mode,
    )
    if result is None:
        return None
    if schema.kind == "integer-writeback" and schema.operation == "auipc":
        return None
    return result


def _trunc_division(left: int, right: int) -> int:
    quotient = abs(int(left)) // abs(int(right))
    return -quotient if (left < 0) ^ (right < 0) else quotient


def _declared_source_value(source_state: Mapping[str, int], role: str) -> int | None:
    if role in source_state:
        value = int(source_state[role])
        # RV32 Zdinx represents one 64-bit FP source as adjacent GPR halves.
        # Reconstruct that pair before evaluating conversions/comparisons so
        # control-consumer selection uses the same concrete value as the
        # materializer and admission, instead of silently using only the low
        # half.
        high = source_state.get(f"{role}+1")
        if high is not None:
            return (value & 0xFFFFFFFF) | ((int(high) & 0xFFFFFFFF) << 32)
        return value
    if role.startswith("rs"):
        fp_role = f"fs{role[2:]}"
        if fp_role in source_state:
            return int(source_state[fp_role])
    return None


def _architectural_declared_gpr_result(form, schema, value: int) -> int:
    """Project a narrow scalar result to the concrete RV64 GPR width.

    The pure boundary evaluator returns an operation-width bit pattern.  RV64
    word operations (including PACKW) and signed narrow loads/atomics expose
    that pattern sign-extended in ``rd``.  Admission already applies this
    projection to an executed testcase; keeping the declaration path in sync
    is required before branch/address consumers are selected.
    """
    form_xlen = str(getattr(form, "xlen", ""))
    if form_xlen not in {"rv32", "rv64", "shared"}:
        return int(value)
    kind = str(getattr(schema, "kind", ""))
    operation = str(getattr(schema, "operation", ""))
    target_xlen = 32 if (
        form_xlen == "rv32"
        or form_xlen == "shared" and operation in {"lui", "auipc"}
        and int(getattr(schema, "width", 0) or 0) == 32
    ) else 64
    if kind in {"memory-load", "atomic-rmw"}:
        width = int(getattr(schema, "memory_width", 0) or getattr(schema, "width", 0) or 0)
        sign_extend = operation != "load-unsigned" if kind == "memory-load" else operation != "sc"
    elif kind == "fp-convert":
        width = int(getattr(schema, "width", 0) or 0)
        sign_extend = operation in {"fp-to-int", "fp-to-int-mod"}
    elif kind in {"integer-writeback", "integer-multiply", "integer-shift"}:
        width = int(getattr(schema, "width", 0) or 0)
        sign_extend = "unsigned-word" not in operation
    else:
        return int(value)
    if width <= 0 or width >= target_xlen or not sign_extend:
        return int(value)
    return _project_gpr_result(value, width, target_xlen, True)


def evaluate_gpr_writeback(
    *,
    kind: str,
    operation: str,
    width: int,
    sources: tuple[int, ...] = (),
    immediate: int | None = None,
    memory_value: int | None = None,
    source_width: int = 0,
    pc: int = 0,
) -> int | None:
    """Pure semantic result for exactly modelled GPR-producing operations.

    ``None`` means the current schema is still too coarse to support an exact
    control-edge obligation without guessing.
    """
    mask = _mask(width)
    values = tuple(_unsigned(int(value), width) for value in sources)
    left = values[0] if values else None
    immediate = None if immediate is None else _unsigned(immediate, width)
    right = values[1] if len(values) >= 2 else immediate
    operation_family, _, operation_variant = operation.partition(":")
    if kind == "memory-load":
        if memory_value is None:
            return None
        raw = _unsigned(memory_value, width)
        if operation == "load-unsigned":
            return raw
        if operation in {"load-signed", "load"}:
            return _signed(raw, width) & mask
        return None
    if kind == "atomic-rmw":
        if operation == "sc" or memory_value is None:
            return None
        raw = _unsigned(memory_value, width)
        # LR/AMO/CAS all feed their destination from the pre-risk memory word.
        # Current consumer axes only need the architecturally exact value flow,
        # not a second owner for reservation success/failure.
        return _signed(raw, width) & mask if width < 64 else raw
    if kind == "csr-access":
        return 0
    if kind == "may-be-operation":
        return 0 if operation == "write-zero" else None
    if kind == "integer-writeback":
        if operation in {
            "carry-less-multiply",
            "carry-less-multiply-reverse",
            "carry-less-multiply-high",
        }:
            if left is None or right is None:
                return None
            product = 0
            for bit in range(width):
                if right & (1 << bit):
                    product ^= left << bit
            shift = (
                width if operation.endswith("-high")
                else width - 1 if operation.endswith("-reverse")
                else 0
            )
            return (product >> shift) & mask
        if operation == "add":
            return None if left is None or right is None else (left + right) & mask
        if operation == "add-unsigned-word":
            return None if left is None or right is None else (_unsigned(left, 32) + right) & mask
        if operation == "sub":
            return None if left is None or right is None else (left - right) & mask
        if operation == "and":
            return None if left is None or right is None else left & right
        if operation == "andn":
            return None if left is None or right is None else left & (~right & mask)
        if operation == "or":
            return None if left is None or right is None else left | right
        if operation == "orn":
            return None if left is None or right is None else left | (~right & mask)
        if operation == "xor":
            return None if left is None or right is None else left ^ right
        if operation == "xnor":
            return None if left is None or right is None else (~(left ^ right)) & mask
        if operation == "set-less-than":
            return None if left is None or right is None else int(_signed(left, width) < _signed(right, width))
        if operation == "set-less-than-unsigned":
            return None if left is None or right is None else int(left < right)
        if operation == "min":
            return None if left is None or right is None else (_signed(left, width) if _signed(left, width) <= _signed(right, width) else _signed(right, width)) & mask
        if operation == "minu":
            return None if left is None or right is None else min(left, right)
        if operation == "max":
            return None if left is None or right is None else (_signed(left, width) if _signed(left, width) >= _signed(right, width) else _signed(right, width)) & mask
        if operation == "maxu":
            return None if left is None or right is None else max(left, right)
        if operation == "move":
            if left is not None:
                return left
            return None if immediate is None else int(immediate) & mask
        if operation == "lui":
            return None if immediate is None else int(immediate) & mask
        if operation == "auipc":
            return None if immediate is None else (int(pc) + int(immediate)) & mask
        if operation == "not":
            return None if left is None else (~left) & mask
        if operation == "byte-or-combine":
            if left is None:
                return None
            return sum(
                ((0xFF if ((left >> shift) & 0xFF) else 0) << shift)
                for shift in range(0, width, 8)
            ) & mask
        if operation_family == "bit-count":
            if left is None:
                return None
            left &= mask
            if operation_variant == "leading-zeros":
                return width if left == 0 else width - left.bit_length()
            if operation_variant == "trailing-zeros":
                return width if left == 0 else (left & -left).bit_length() - 1
            return left.bit_count() if operation_variant == "population" else None
        if operation_family == "byte-permute":
            if left is None:
                return None
            if operation_variant == "reverse-bits-in-bytes":
                result = 0
                for offset in range(0, width, 8):
                    value = (left >> offset) & 0xFF
                    value = ((value & 0x55) << 1) | ((value >> 1) & 0x55)
                    value = ((value & 0x33) << 2) | ((value >> 2) & 0x33)
                    result |= (((value << 4) | (value >> 4)) & 0xFF) << offset
                return result
            if operation_variant in {"reverse-bytes", "reverse-four-bytes"}:
                byte_width = 4 if operation_variant == "reverse-four-bytes" else width // 8
                return int.from_bytes(
                    int(left & _mask(byte_width * 8)).to_bytes(byte_width, "little")[::-1],
                    "little",
                )
            if operation_variant in {"interleave-halves", "deinterleave-halves"}:
                left &= _mask(32)
                if operation_variant == "interleave-halves":
                    return sum(
                        (((left >> bit) & 1) << (2 * bit))
                        | (((left >> (bit + 16)) & 1) << ((2 * bit) + 1))
                        for bit in range(16)
                    )
                return sum(
                    (((left >> (2 * bit)) & 1) << bit)
                    | (((left >> ((2 * bit) + 1)) & 1) << (bit + 16))
                    for bit in range(16)
                )
            return None
        if operation_family == "pack":
            if left is None or right is None:
                return None
            if operation_variant == "halves":
                half = width // 2
                return (left & _mask(half)) | ((right & _mask(half)) << half)
            if operation_variant == "bytes":
                return (left & 0xFF) | ((right & 0xFF) << 8)
            if operation_variant == "words":
                return (left & 0xFFFF) | ((right & 0xFFFF) << 16)
            return None
        if operation_family == "crossbar-permute":
            if left is None or right is None:
                return None
            element_width = {"4": 4, "8": 8}.get(operation_variant)
            if element_width is None:
                return None
            element_mask = _mask(element_width)
            element_count = width // element_width
            result = 0
            for offset in range(0, width, element_width):
                index = (right >> offset) & element_mask
                if index < element_count:
                    result |= ((left >> (index * element_width)) & element_mask) << offset
            return result
        if operation == "sign-extend":
            if left is None or source_width <= 0:
                return None
            return _signed(left, source_width) & mask
        if operation == "zero-extend":
            if left is None or source_width <= 0:
                return None
            return _unsigned(left, source_width)
        if operation_family == "shift-add":
            if left is None or right is None:
                return None
            return (right + (left << int(operation_variant))) & mask
        if operation_family == "shift-add-unsigned-word":
            if left is None or right is None:
                return None
            return (right + (_unsigned(left, 32) << int(operation_variant))) & mask
        if operation_family == "conditional-zero":
            if left is None or right is None:
                return None
            take = right == 0 if operation_variant == "eqz" else right != 0
            return 0 if take else left
        return None
    if kind == "integer-shift":
        if left is None:
            return None
        amount = values[1] if len(values) >= 2 else immediate
        if amount is None:
            return None
        amount = int(amount)
        if operation == "bit-extract" and amount >= width:
            return 0
        amount &= width - 1
        if operation == "shift-left":
            return (left << amount) & mask
        if operation == "shift-right-logical":
            return left >> amount
        if operation == "shift-right-arithmetic":
            return (_signed(left, width) >> amount) & mask
        if operation == "rotate-left":
            return ((left << amount) | (left >> ((width - amount) & (width - 1)))) & mask
        if operation == "rotate-right":
            return ((left >> amount) | (left << ((width - amount) & (width - 1)))) & mask
        if operation == "bit-set":
            return left | (1 << amount)
        if operation == "bit-clear":
            return left & ~(1 << amount) & mask
        if operation == "bit-invert":
            return left ^ (1 << amount)
        if operation == "bit-extract":
            return (left >> amount) & 0x1
        if operation == "shift-left-unsigned-word":
            return (_unsigned(left, 32) << amount) & mask
        return None
    if kind == "integer-multiply":
        if left is None or right is None:
            return None
        if operation == "mul":
            return (left * right) & mask
        wide_mask = _mask(width * 2)
        if operation == "mulh":
            return ((_signed(left, width) * _signed(right, width)) & wide_mask) >> width
        if operation == "mulhu":
            return ((left * right) & wide_mask) >> width
        if operation == "mulhsu":
            return ((_signed(left, width) * right) & wide_mask) >> width
        if operation == "div":
            divisor = _signed(right, width)
            dividend = _signed(left, width)
            minimum = -(1 << (width - 1))
            if divisor == 0:
                return mask
            if dividend == minimum and divisor == -1:
                return _unsigned(dividend, width)
            return _unsigned(_trunc_division(dividend, divisor), width)
        if operation == "divu":
            return mask if right == 0 else left // right
        if operation == "rem":
            divisor = _signed(right, width)
            dividend = _signed(left, width)
            minimum = -(1 << (width - 1))
            if divisor == 0:
                return _unsigned(dividend, width)
            if dividend == minimum and divisor == -1:
                return 0
            return _unsigned(dividend - (_trunc_division(dividend, divisor) * divisor), width)
        if operation == "remu":
            return left if right == 0 else left % right
    return None


def _declared_rounding_mode(
    encoded_mode: int | None,
    dynamic_mode: int | None,
) -> str | None:
    if type(encoded_mode) is not int:
        return None
    if encoded_mode == 7:
        encoded_mode = dynamic_mode
        if type(encoded_mode) is not int:
            return None
    return _ROUNDING_MODE_BY_ENCODING.get(encoded_mode)


def projection_writeback_pairs(width: int) -> dict[str, tuple[tuple[int, int], ...]]:
    if width <= 1 or width > 64:
        raise ValueError("projection width must be between 2 and 64")
    sign = 1 << (width - 1)
    return {
        "sign-clear": tuple((sign - delta, delta - 2) for delta in (2, 3, 4)),
        "sign-set": tuple((sign - delta, delta) for delta in (1, 2, 3)),
    }


def comparison_boundary_pairs(
    *,
    width: int,
    signed: bool,
    relation: str,
) -> tuple[tuple[int, int], ...]:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    if relation == "less":
        rows = (
            ((-1, 0), (-5, -1), (0, 1), (sign, 0))
            if signed
            else ((0, 1), (1, 2), (mask - 1, mask))
        )
    elif relation == "not-less":
        rows = (
            ((1, 0), (5, 5), (0, -1), (0, sign))
            if signed
            else ((1, 0), (5, 5), (mask, mask - 1))
        )
    elif relation == "equal":
        rows = ((0, 0), (5, 5), (mask, mask))
    elif relation == "not-equal":
        rows = ((0, 1), (5, 4), (mask, 0))
    else:
        raise ValueError(f"unsupported comparison relation: {relation}")
    return tuple((left & mask, right & mask) for left, right in rows)


def integer_operation_boundary_pairs(
    *,
    width: int,
    operation: str,
    signed: bool,
) -> tuple[tuple[int, int], ...]:
    """Boundary rows chosen by the arithmetic relation an operation owns."""
    operation = operation.partition(":")[0]
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    alt0 = sum(1 << bit for bit in range(0, width, 2))
    alt1 = mask ^ alt0
    if operation == "add":
        rows = (
            (0, 0),
            (0, 1),
            (mask, 1),
            (sign - 1, 1),
            (sign, sign),
            (mask, mask),
        )
    elif operation == "sub":
        rows = (
            (0, 0),
            (0, 1),
            (1, 0),
            (0, mask),
            (sign, 1),
            (sign - 1, mask),
        )
    elif operation in {"set-less-than", "set-less-than-unsigned", "min", "minu", "max", "maxu"}:
        less = comparison_boundary_pairs(width=width, signed=signed, relation="less")
        equal = comparison_boundary_pairs(width=width, signed=signed, relation="equal")
        not_less = comparison_boundary_pairs(width=width, signed=signed, relation="not-less")
        rows = less + equal[:1] + not_less
    elif operation in {"and", "andn", "or", "orn", "xor", "xnor"}:
        rows = (
            (0, 0),
            (0, mask),
            (mask, 0),
            (mask, mask),
            (alt0, alt1),
            (alt0, alt0),
        )
    else:
        rows = tuple(
            (left, right)
            for left in (0, 1, sign - 1, sign, mask)
            for right in (0, 1, sign - 1, sign, mask)
        ) + (
            (alt0, alt1),
            (alt1, alt0),
            (alt0, alt0),
            (1, sign),
            (sign, 1),
        )
    return tuple(dict.fromkeys((left & mask, right & mask) for left, right in rows))


def shift_amount_boundaries(xlen: int) -> tuple[int, ...]:
    if xlen not in {32, 64}:
        raise ValueError("xlen must be 32 or 64")
    return (0, 1, xlen - 1, xlen, xlen + 1, (2 * xlen) - 1)


def natural_alignment_offsets(width_bytes: int) -> tuple[int, ...]:
    if width_bytes not in {1, 2, 4, 8, 16}:
        raise ValueError("unsupported natural alignment width")
    return tuple(dict.fromkeys((0, width_bytes - 1, width_bytes, width_bytes + 1)))


def division_exception_pairs(width: int, signed: bool) -> tuple[tuple[int, int], ...]:
    mask = (1 << width) - 1
    minimum = 1 << (width - 1)
    rows = [(0, 0), (1, 0), (mask, 0)]
    if signed:
        rows.append((minimum, mask))
    return tuple((left & mask, right & mask) for left, right in rows)


def integer_to_fp_rounding_relation(value: int, precision: int) -> str:
    magnitude = abs(int(value))
    if magnitude == 0 or magnitude.bit_length() <= precision:
        return "exact"
    discarded_width = magnitude.bit_length() - precision
    discarded = magnitude & ((1 << discarded_width) - 1)
    if discarded == 0:
        return "exact"
    halfway = 1 << (discarded_width - 1)
    if discarded < halfway:
        return "below-halfway"
    if discarded == halfway:
        return "halfway"
    return "above-halfway"


def integer_to_fp_boundaries(
    source_width: int,
    signed: bool,
    destination_precision: int,
) -> tuple[IntegerToFpBoundary, ...]:
    if source_width not in {32, 64}:
        raise ValueError("source_width must be 32 or 64")
    if destination_precision not in IEEE_FORMAT_BY_PRECISION:
        raise ValueError("unsupported destination precision")
    unsigned_max = (1 << source_width) - 1
    signed_min = -(1 << (source_width - 1))
    signed_max = (1 << (source_width - 1)) - 1
    edge = 1 << destination_precision
    wider_edge = 1 << (destination_precision + 1)
    rows: list[tuple[int, str]] = [
        (0, "zero"),
        (1, "one"),
        (edge - 1, "precision-minus-one"),
        (edge, "precision-edge"),
        (edge + 1, "precision-plus-one"),
        (edge + 2, "precision-plus-two"),
        (wider_edge + 1, "rounding-just-below"),
        (wider_edge + 2, "rounding-halfway"),
        (wider_edge + 3, "rounding-just-above"),
        (signed_max if signed else unsigned_max, "source-maximum"),
    ]
    if source_width == 64 and destination_precision == 24:
        rows.append((1 << 40, "source-upper-power-of-two"))
    if signed:
        rows.extend(
            (
                (-1, "negative-one"),
                (-edge - 1, "negative-precision-plus-one"),
                (-wider_edge - 1, "negative-rounding-just-below"),
                (-wider_edge - 2, "negative-rounding-halfway"),
                (-wider_edge - 3, "negative-rounding-just-above"),
                (signed_min, "source-minimum"),
            )
        )
    unique: dict[int, IntegerToFpBoundary] = {}
    source_mask = (1 << source_width) - 1
    lower = signed_min if signed else 0
    upper = signed_max if signed else unsigned_max
    for value, boundary_class in rows:
        if value < lower or value > upper:
            continue
        encoded = int(value) & source_mask
        if signed and source_width < 64 and encoded & (1 << (source_width - 1)):
            encoded |= ((1 << 64) - 1) ^ source_mask
        unique.setdefault(
            value,
            IntegerToFpBoundary(
                encoded_u64=encoded & ((1 << 64) - 1),
                boundary_class=boundary_class,
                rounding_relation=integer_to_fp_rounding_relation(value, destination_precision),
            ),
        )
    return tuple(unique[value] for value in sorted(unique))


IEEE_FORMAT_BY_PRECISION = {
    8: (16, 8, 7),
    11: (16, 5, 10),
    24: (32, 8, 23),
    53: (64, 11, 52),
    113: (128, 15, 112),
}


def ieee_bit_classes(precision: int) -> dict[str, int]:
    """Every IEEE category as raw bits, from the format parameters alone.

    Categories -- not per-instruction boundary names -- are what a floating
    point effect is defined over, so one table serves every current and future
    form of a given precision.
    """
    try:
        width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    except KeyError:
        raise ValueError(f"unsupported fp precision: {precision}") from None
    sign = 1 << (width - 1)
    exponent = ((1 << exponent_bits) - 1) << fraction_bits
    quiet = 1 << (fraction_bits - 1)
    one = ((1 << (exponent_bits - 1)) - 1) << fraction_bits
    return {
        "pos-zero": 0,
        "neg-zero": sign,
        "pos-subnormal": quiet,
        "neg-subnormal": sign | quiet,
        "min-subnormal": 1,
        "second-min-subnormal": 2,
        "max-subnormal": (1 << fraction_bits) - 1,
        "min-normal": 1 << fraction_bits,
        "one": one,
        "two": one + (1 << fraction_bits),
        "neg-one": sign | one,
        "max-finite": exponent - 1,
        "pos-inf": exponent,
        "neg-inf": sign | exponent,
        "qnan": exponent | quiet | 1,
        "snan": exponent | (quiet >> 1) | 1,
        "snan-negative": sign | exponent | (quiet >> 1) | 1,
        "qnan-negative": sign | exponent | quiet | 1,
        "qnan-payload": exponent | quiet | (quiet >> 2) | 0x2B,
        "qnan-payload-negative": sign | exponent | quiet | (quiet >> 2) | 0x2B,
        "half": one - (1 << fraction_bits),
        "half-below": one - (1 << fraction_bits) - 1,
        "half-above": one - (1 << fraction_bits) + 1,
        "one-and-half": one | (1 << (fraction_bits - 1)),
    }


def _fp_fields(precision: int, value: int) -> tuple[bool, int, int]:
    width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    value &= (1 << width) - 1
    return (
        bool(value & (1 << (width - 1))),
        (value >> fraction_bits) & ((1 << exponent_bits) - 1),
        value & ((1 << fraction_bits) - 1),
    )


def _fp_pack(precision: int, negative: bool, exponent: int, fraction: int) -> int:
    width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    del exponent_bits
    return (
        (int(bool(negative)) << (width - 1))
        | (int(exponent) << fraction_bits)
        | (int(fraction) & ((1 << fraction_bits) - 1))
    )


def _fp_sign(precision: int, value: int) -> bool:
    return _fp_fields(precision, value)[0]


def _fp_is_zero(precision: int, value: int) -> bool:
    _negative, exponent, fraction = _fp_fields(precision, value)
    return exponent == 0 and fraction == 0


def _fp_is_inf(precision: int, value: int) -> bool:
    _negative, exponent, fraction = _fp_fields(precision, value)
    exponent_bits = IEEE_FORMAT_BY_PRECISION[precision][1]
    return exponent == (1 << exponent_bits) - 1 and fraction == 0


def _fp_is_nan(precision: int, value: int) -> bool:
    _negative, exponent, fraction = _fp_fields(precision, value)
    exponent_bits = IEEE_FORMAT_BY_PRECISION[precision][1]
    return exponent == (1 << exponent_bits) - 1 and fraction != 0


def canonical_nan_bits(precision: int) -> int:
    """The RISC-V default NaN, independent of host FP payload handling."""
    _width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    return ((1 << exponent_bits) - 1) << fraction_bits | (1 << (fraction_bits - 1))


def _infinite_bits(precision: int, negative: bool) -> int:
    _width, exponent_bits, _fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    return _fp_pack(precision, negative, (1 << exponent_bits) - 1, 0)


def _round_fraction_to_bits(
    precision: int,
    value: Fraction,
    *,
    rounding_mode: str,
    zero_negative: bool = False,
) -> int:
    """Round an exact finite rational to one IEEE binary interchange format."""
    if value == 0:
        return _fp_pack(precision, zero_negative, 0, 0)
    _width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    significand_bits = fraction_bits + 1
    bias = (1 << (exponent_bits - 1)) - 1
    min_normal_exponent = 1 - bias
    max_normal_exponent = bias
    negative = value < 0
    magnitude = -value if negative else value
    exponent = _floor_log2_fraction(magnitude)
    numerator = int(magnitude.numerator)
    denominator = int(magnitude.denominator)
    if exponent >= min_normal_exponent:
        shift = significand_bits - 1 - exponent
        scaled_numerator = numerator << shift if shift >= 0 else numerator
        scaled_denominator = denominator if shift >= 0 else denominator << (-shift)
        significand, _ = _round_nonnegative_ratio(
            scaled_numerator,
            scaled_denominator,
            rounding_mode=rounding_mode,
            negative=negative,
        )
        if significand == (1 << significand_bits):
            significand >>= 1
            exponent += 1
        if exponent > max_normal_exponent:
            to_infinity = rounding_mode in {"rne", "rmm"} or (
                rounding_mode == "rup" and not negative
            ) or (rounding_mode == "rdn" and negative)
            return (
                _infinite_bits(precision, negative)
                if to_infinity
                else _fp_pack(precision, negative, (1 << exponent_bits) - 2, (1 << fraction_bits) - 1)
            )
        if exponent >= min_normal_exponent:
            return _fp_pack(
                precision,
                negative,
                exponent + bias,
                significand - (1 << (significand_bits - 1)),
            )
    # The normal rounding path can fall into the subnormal range after a
    # directed rounding.  Round directly against the subnormal unit.
    shift = fraction_bits - min_normal_exponent
    scaled_numerator = numerator << shift if shift >= 0 else numerator
    scaled_denominator = denominator if shift >= 0 else denominator << (-shift)
    fraction, _ = _round_nonnegative_ratio(
        scaled_numerator,
        scaled_denominator,
        rounding_mode=rounding_mode,
        negative=negative,
    )
    if fraction >= (1 << fraction_bits):
        return _fp_pack(precision, negative, 1, 0)
    return _fp_pack(precision, negative, 0, fraction)


def _select_minmax(precision: int, operation: str, operands: tuple[int, ...]) -> int:
    left, right = operands[:2]
    left_nan, right_nan = _fp_is_nan(precision, left), _fp_is_nan(precision, right)
    if operation in {"minimum-number", "maximum-number"} and (left_nan or right_nan):
        return canonical_nan_bits(precision)
    if left_nan and right_nan:
        return canonical_nan_bits(precision)
    if left_nan:
        return right
    if right_nan:
        return left
    left_value = ieee_finite_fraction(precision, left)
    right_value = ieee_finite_fraction(precision, right)
    if left_value is None or right_value is None:
        minimum = operation in {"minimum", "minimum-number"}
        if _fp_is_inf(precision, left) and _fp_is_inf(precision, right):
            if _fp_sign(precision, left) == _fp_sign(precision, right):
                return left
            choose_left = _fp_sign(precision, left) if minimum else not _fp_sign(precision, left)
        elif _fp_is_inf(precision, left):
            choose_left = _fp_sign(precision, left) if minimum else not _fp_sign(precision, left)
        elif _fp_is_inf(precision, right):
            choose_left = not _fp_sign(precision, right) if minimum else _fp_sign(precision, right)
        else:
            raise ValueError("minmax requires finite or infinite operands")
        return left if choose_left else right
    if left_value == right_value:
        if _fp_is_zero(precision, left) and _fp_is_zero(precision, right):
            want_negative = operation in {"minimum", "minimum-number"}
            if _fp_sign(precision, left) == want_negative:
                return left
            return right
        return left
    minimum = operation in {"minimum", "minimum-number"}
    choose_left = left_value < right_value if minimum else left_value > right_value
    return left if choose_left else right


def _compare_fp(precision: int, operation: str, operands: tuple[int, ...]) -> int:
    left, right = operands[:2]
    if _fp_is_nan(precision, left) or _fp_is_nan(precision, right):
        return 0
    left_value = ieee_finite_fraction(precision, left)
    right_value = ieee_finite_fraction(precision, right)
    if left_value is None or right_value is None:
        if _fp_is_inf(precision, left) and _fp_is_inf(precision, right):
            comparison = (_fp_sign(precision, right) > _fp_sign(precision, left)) - (
                _fp_sign(precision, right) < _fp_sign(precision, left)
            )
        elif _fp_is_inf(precision, left):
            comparison = -1 if _fp_sign(precision, left) else 1
        elif _fp_is_inf(precision, right):
            comparison = 1 if _fp_sign(precision, right) else -1
        else:
            raise ValueError("comparison requires finite or infinite operands")
    else:
        comparison = (left_value > right_value) - (left_value < right_value)
    if operation in {"equal"}:
        result = int(comparison == 0)
    elif operation in {"less-than", "less-than-quiet"}:
        result = int(comparison < 0)
    elif operation in {"less-equal", "less-equal-quiet"}:
        result = int(comparison <= 0)
    else:
        raise ValueError(f"unsupported FP compare operation: {operation}")
    return result


def _sqrt_fraction_to_bits(precision: int, value: Fraction, *, rounding_mode: str) -> int:
    """Round sqrt(value) without routing through host binary floating point."""
    if value <= 0:
        raise ValueError("sqrt evaluator requires a positive finite value")
    _width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    exponent = _floor_log2_fraction(value) // 2
    significand_bits = fraction_bits + 1
    shift = significand_bits - 1 - exponent
    numerator = int(value.numerator)
    denominator = int(value.denominator)
    if shift >= 0:
        numerator <<= 2 * shift
    else:
        denominator <<= -2 * shift
    whole = numerator // denominator
    significand = isqrt(whole)
    while (significand + 1) * (significand + 1) * denominator <= numerator:
        significand += 1
    while significand * significand * denominator > numerator:
        significand -= 1
    exact = significand * significand * denominator == numerator
    if not exact:
        compare_half = (4 * numerator > denominator * (2 * significand + 1) ** 2) - (
            4 * numerator < denominator * (2 * significand + 1) ** 2
        )
        increment = (
            rounding_mode == "rup"
            or (rounding_mode == "rne" and (compare_half > 0 or (compare_half == 0 and bool(significand & 1))))
            or (rounding_mode == "rmm" and compare_half >= 0)
        )
        if increment:
            significand += 1
    if significand == (1 << significand_bits):
        significand >>= 1
        exponent += 1
    bias = (1 << (exponent_bits - 1)) - 1
    return _fp_pack(precision, False, exponent + bias, significand - (1 << (significand_bits - 1)))


def evaluate_fp_arithmetic(
    precision: int,
    operation: str,
    operands: tuple[int, ...],
    *,
    rounding_mode: str,
) -> int:
    """Exact result bits for scalar arithmetic, compare, sign and min/max."""
    if not operands:
        raise ValueError("FP operation requires operands")
    if operation in {"copy-sign", "invert-sign", "xor-sign"}:
        left, right = operands[:2]
        width = IEEE_FORMAT_BY_PRECISION[precision][0]
        sign = _fp_sign(precision, right)
        if operation == "invert-sign":
            sign = not sign
        elif operation == "xor-sign":
            sign = _fp_sign(precision, left) ^ sign
        return (left & ((1 << (width - 1)) - 1)) | (int(sign) << (width - 1))
    if operation in {"minimum", "maximum", "minimum-number", "maximum-number"}:
        return _select_minmax(precision, operation, operands)
    if operation in {"equal", "less-than", "less-equal", "less-than-quiet", "less-equal-quiet"}:
        return _compare_fp(precision, operation, operands)
    if operation == "sqrt":
        source = operands[0]
        if _fp_is_nan(precision, source):
            return canonical_nan_bits(precision)
        if _fp_is_inf(precision, source):
            return source if not _fp_sign(precision, source) else canonical_nan_bits(precision)
        if _fp_is_zero(precision, source):
            return source
        value = ieee_finite_fraction(precision, source)
        if value is None or value < 0:
            return canonical_nan_bits(precision)
        return _sqrt_fraction_to_bits(precision, value, rounding_mode=rounding_mode)
    if operation not in {"add", "sub", "mul", "div", "madd", "msub", "nmsub", "nmadd"}:
        raise ValueError(f"unsupported FP arithmetic operation: {operation}")
    if operation in {"madd", "msub", "nmsub", "nmadd"}:
        first, second, third = operands[:3]
        zero_times_inf = (_fp_is_zero(precision, first) and _fp_is_inf(precision, second)) or (
            _fp_is_inf(precision, first) and _fp_is_zero(precision, second)
        )
        if zero_times_inf:
            return canonical_nan_bits(precision)
    if any(_fp_is_nan(precision, value) for value in operands):
        return canonical_nan_bits(precision)
    first, second = operands[:2]
    if operation in {"add", "sub"}:
        second_negative = _fp_sign(precision, second) ^ (operation == "sub")
        if _fp_is_inf(precision, first) and _fp_is_inf(precision, second):
            if _fp_sign(precision, first) != second_negative:
                return canonical_nan_bits(precision)
            return _infinite_bits(precision, _fp_sign(precision, first))
        if _fp_is_inf(precision, first):
            return first
        if _fp_is_inf(precision, second):
            return _infinite_bits(precision, second_negative)
        left = ieee_finite_fraction(precision, first)
        right = ieee_finite_fraction(precision, second)
        assert left is not None and right is not None
        exact = left + (-right if operation == "sub" else right)
        zero_negative = rounding_mode == "rdn" if exact == 0 else False
        if (
            _fp_is_zero(precision, first)
            and _fp_is_zero(precision, second)
            and _fp_sign(precision, first) == second_negative
        ):
            zero_negative = _fp_sign(precision, first)
        return _round_fraction_to_bits(precision, exact, rounding_mode=rounding_mode, zero_negative=zero_negative)
    if operation == "mul":
        if (_fp_is_zero(precision, first) and _fp_is_inf(precision, second)) or (_fp_is_inf(precision, first) and _fp_is_zero(precision, second)):
            return canonical_nan_bits(precision)
        negative = _fp_sign(precision, first) ^ _fp_sign(precision, second)
        if _fp_is_inf(precision, first) or _fp_is_inf(precision, second):
            return _infinite_bits(precision, negative)
        left = ieee_finite_fraction(precision, first)
        right = ieee_finite_fraction(precision, second)
        assert left is not None and right is not None
        return _round_fraction_to_bits(precision, left * right, rounding_mode=rounding_mode, zero_negative=negative)
    if operation == "div":
        negative = _fp_sign(precision, first) ^ _fp_sign(precision, second)
        if (_fp_is_zero(precision, first) and _fp_is_zero(precision, second)) or (_fp_is_inf(precision, first) and _fp_is_inf(precision, second)):
            return canonical_nan_bits(precision)
        if _fp_is_zero(precision, second):
            return _infinite_bits(precision, negative)
        if _fp_is_inf(precision, first):
            return _infinite_bits(precision, negative)
        if _fp_is_inf(precision, second):
            return _fp_pack(precision, negative, 0, 0)
        left = ieee_finite_fraction(precision, first)
        right = ieee_finite_fraction(precision, second)
        assert left is not None and right is not None
        return _round_fraction_to_bits(precision, left / right, rounding_mode=rounding_mode, zero_negative=negative)
    # Fused operations above have excluded NaNs and invalid 0*inf products.
    third = operands[2]
    product_negative = _fp_sign(precision, first) ^ _fp_sign(precision, second)
    if operation in {"nmsub", "nmadd"}:
        product_negative = not product_negative
    addend_negative = _fp_sign(precision, third) ^ (operation in {"msub", "nmsub"})
    product_inf = _fp_is_inf(precision, first) or _fp_is_inf(precision, second)
    if product_inf and _fp_is_inf(precision, third) and product_negative != addend_negative:
        return canonical_nan_bits(precision)
    if product_inf:
        return _infinite_bits(precision, product_negative)
    if _fp_is_inf(precision, third):
        return _infinite_bits(precision, addend_negative)
    left = ieee_finite_fraction(precision, first)
    right = ieee_finite_fraction(precision, second)
    addend = ieee_finite_fraction(precision, third)
    assert left is not None and right is not None and addend is not None
    product = left * right
    if operation in {"nmsub", "nmadd"}:
        product = -product
    if operation in {"msub", "nmsub"}:
        addend = -addend
    exact = product + addend
    zero_negative = rounding_mode == "rdn" if exact == 0 else False
    if (
        (_fp_is_zero(precision, first) or _fp_is_zero(precision, second))
        and _fp_is_zero(precision, third)
        and product_negative == addend_negative
    ):
        zero_negative = product_negative
    return _round_fraction_to_bits(
        precision,
        exact,
        rounding_mode=rounding_mode,
        zero_negative=zero_negative,
    )




def evaluate_fp_to_int(
    precision: int,
    value: int,
    *,
    width: int,
    signed: bool,
    rounding_mode: str,
) -> int:
    """Architectural FCVT.int result, including the specified clipping result."""
    minimum = -(1 << (width - 1)) if signed else 0
    maximum = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    if _fp_is_nan(precision, value) or (_fp_is_inf(precision, value) and not _fp_sign(precision, value)):
        return maximum & ((1 << width) - 1)
    if _fp_is_inf(precision, value):
        return minimum & ((1 << width) - 1)
    components = _ieee_finite_components(precision, value)
    if components is None:
        raise ValueError("finite conversion source did not decode")
    rounded, _ = _rounded_integer(*components, rounding_mode)
    if rounded < minimum:
        return minimum & ((1 << width) - 1)
    if rounded > maximum:
        return maximum & ((1 << width) - 1)
    return rounded & ((1 << width) - 1)


def fp_to_int_exact_negative_minimum_bits(precision: int, width: int) -> int | None:
    """Encode the exact signed minimum of an integer conversion target.

    ``FCVT.W*`` and ``FCVT.L*`` have target minima ``-2**31`` and
    ``-2**63`` respectively.  Constructing the source bits from the target
    width keeps this partition independent of a mnemonic or defect label and
    lets the ordinary IEEE rounding helper provide the format projection.
    """
    if width not in {32, 64}:
        raise ValueError("fp-to-int target width must be 32 or 64")
    encoded = _round_fraction_to_bits(
        precision, Fraction(-(1 << (width - 1)), 1), rounding_mode="rne"
    )
    # A narrow source format (for example H) cannot represent the target
    # minimum at all; its rounded image is -inf and is already covered by the
    # ordinary saturation partition, not by an exact-minimum row.
    return None if _fp_is_inf(precision, encoded) else int(encoded)


def fp_to_int_exact_positive_limit_bits(
    precision: int, width: int, *, signed: bool = True
) -> int | None:
    """Encode the first positive integer-conversion clip boundary."""
    if width not in {32, 64}:
        raise ValueError("fp-to-int target width must be 32 or 64")
    encoded = _round_fraction_to_bits(
        precision, Fraction(1 << (width - 1 if signed else width), 1), rounding_mode="rne"
    )
    return None if _fp_is_inf(precision, encoded) else int(encoded)


def evaluate_fp_to_int_mod_w_d(value: int) -> int:
    """Exact Zfa FCVTMOD.W.D low-word result."""
    precision = 53
    if _fp_is_nan(precision, value) or _fp_is_inf(precision, value):
        return 0
    components = _ieee_finite_components(precision, value)
    if components is None:
        raise ValueError("finite FCVTMOD source did not decode")
    rounded, _ = _rounded_integer(*components, "rtz")
    return rounded & 0xFFFFFFFF


def evaluate_fp_classify(precision: int, value: int) -> int:
    negative, exponent, fraction = _fp_fields(precision, value)
    exponent_bits = IEEE_FORMAT_BY_PRECISION[precision][1]
    fraction_bits = IEEE_FORMAT_BY_PRECISION[precision][2]
    if exponent == (1 << exponent_bits) - 1:
        if fraction == 0:
            bit = 0 if negative else 7
        else:
            bit = 8 if not (fraction & (1 << (fraction_bits - 1))) else 9
    elif exponent == 0:
        bit = (3 if negative else 4) if fraction == 0 else 2 if negative else 5
    else:
        bit = 1 if negative else 6
    return 1 << bit


def _mask(width: int) -> int:
    return (1 << width) - 1


def _unsigned(value: int, width: int) -> int:
    return int(value) & _mask(width)


def _signed(value: int, width: int) -> int:
    value = _unsigned(value, width)
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def _project_gpr_result(value: int, width: int, xlen: int, sign_extend: bool) -> int:
    value = int(value) & _mask(width)
    if sign_extend and width < xlen and value & (1 << (width - 1)):
        value |= (1 << (xlen - width)) - 1 << width
    return value & _mask(xlen)


def _ieee_finite_components(precision: int, value: int) -> tuple[int, int, int] | None:
    """Return sign, significand and binary shift for one finite IEEE value."""
    negative, exponent, fraction = _fp_fields(precision, value)
    _width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
    if exponent == (1 << exponent_bits) - 1:
        return None
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == 0:
        if fraction == 0:
            return (-1 if negative else 1), 0, 0
        return (-1 if negative else 1), fraction, 1 - bias - fraction_bits
    return (-1 if negative else 1), (1 << fraction_bits) | fraction, exponent - bias - fraction_bits


def ieee_finite_fraction(precision: int, value: int) -> Fraction | None:
    components = _ieee_finite_components(precision, value)
    if components is None:
        return None
    sign, significand, shift = components
    if significand == 0:
        return Fraction(0, 1)
    magnitude = Fraction(significand << shift, 1) if shift >= 0 else Fraction(significand, 1 << (-shift))
    return magnitude if sign > 0 else -magnitude


def _compare_ratio_to_power_of_two(numerator: int, denominator: int, exponent: int) -> int:
    if exponent >= 0:
        left, right = numerator, denominator << exponent
    else:
        left, right = numerator << (-exponent), denominator
    return (left > right) - (left < right)


def _floor_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 is defined only for positive values")
    numerator = int(value.numerator)
    denominator = int(value.denominator)
    exponent = numerator.bit_length() - denominator.bit_length()
    if _compare_ratio_to_power_of_two(numerator, denominator, exponent) < 0:
        exponent -= 1
    while _compare_ratio_to_power_of_two(numerator, denominator, exponent + 1) >= 0:
        exponent += 1
    return exponent


def _round_nonnegative_ratio(
    numerator: int,
    denominator: int,
    *,
    rounding_mode: str,
    negative: bool,
) -> tuple[int, bool]:
    quotient, remainder = divmod(numerator, denominator)
    if remainder == 0:
        return quotient, True
    increment = False
    doubled = remainder * 2
    if rounding_mode == "rtz":
        increment = False
    elif rounding_mode == "rdn":
        increment = negative
    elif rounding_mode == "rup":
        increment = not negative
    elif rounding_mode == "rne":
        increment = doubled > denominator or (doubled == denominator and bool(quotient & 1))
    elif rounding_mode == "rmm":
        increment = doubled >= denominator
    else:
        raise ValueError(f"unsupported rounding mode: {rounding_mode}")
    return quotient + (1 if increment else 0), False


def _rounded_integer(sign: int, significand: int, shift: int, rounding_mode: str) -> tuple[int, bool]:
    """Round an exact binary rational to an integer under one IEEE mode."""
    if shift >= 0:
        return sign * (significand << shift), True
    denominator_shift = -shift
    integer_part = significand >> denominator_shift
    remainder = significand & ((1 << denominator_shift) - 1)
    trunc = integer_part if sign > 0 else -integer_part
    if remainder == 0:
        return trunc, True
    floor = trunc if sign > 0 else -(integer_part + 1)
    ceil = integer_part + 1 if sign > 0 else trunc
    if rounding_mode == "rtz":
        return trunc, False
    if rounding_mode == "rdn":
        return floor, False
    if rounding_mode == "rup":
        return ceil, False
    half = 1 << (denominator_shift - 1)
    away = ceil if sign > 0 else floor
    if remainder < half:
        return trunc, False
    if remainder > half:
        return away, False
    if rounding_mode == "rne":
        return (trunc if trunc % 2 == 0 else away), False
    if rounding_mode == "rmm":
        return away, False
    raise ValueError(f"unsupported rounding mode: {rounding_mode}")


def fp_to_fp_rounding_rows(
    source_precision: int,
    destination_precision: int,
) -> tuple[tuple[str, int], ...]:
    """Source-format encodings around one destination-format half-ULP."""
    if source_precision <= destination_precision:
        return ()
    _source_width, source_exponent_bits, source_fraction_bits = IEEE_FORMAT_BY_PRECISION[source_precision]
    _destination_width, _destination_exponent_bits, destination_fraction_bits = IEEE_FORMAT_BY_PRECISION[
        destination_precision
    ]
    source_bias = (1 << (source_exponent_bits - 1)) - 1
    one = source_bias << source_fraction_bits
    delta = 1 << (source_fraction_bits - destination_fraction_bits - 1)
    return (
        ("rounding-just-below", one + delta - 1),
        ("rounding-halfway", one + delta),
        ("rounding-just-above", one + delta + 1),
    )


def ieee_operation_operand_rows(
    precision: int, sources: int, operation: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """IEEE boundary coordinates selected by an effect, not by a mnemonic.

    A row only promises the architectural input state that is materialised.  It
    deliberately does not predict a result here: that would duplicate
    the execution oracle.  The operation decides which exceptional and
    discontinuity coordinates are meaningful, while format parameters create
    their actual bit patterns for H/S/D/Q alike.
    """
    bits = ieee_bit_classes(precision)

    def row(name: str, *values: int) -> tuple[str, tuple[int, ...]]:
        carried = values[:sources]
        return name, carried + (bits["pos-zero"],) * max(sources - len(carried), 0)

    if sources <= 1:
        unary = {
            "sqrt": (
                "snan",
                "snan-negative",
                "qnan",
                "qnan-payload-negative",
                "neg-one",
                "neg-zero",
                "pos-inf",
                "min-subnormal",
                "min-normal",
                "two",
            ),
        }.get(operation)
        if unary is not None:
            return tuple(row(name, bits[name]) for name in unary)
        return ()
    snan_one = ("snan-one", bits["snan"], bits["one"])
    qnan_one = ("qnan-one", bits["qnan"], bits["one"])
    signed_zero = ("signed-zero", bits["neg-zero"], bits["pos-zero"])
    ordered_comparison_rows = (
        snan_one,
        qnan_one,
        ("ordered-less", bits["min-normal"], bits["max-finite"]),
        ("ordered-greater", bits["max-finite"], bits["min-normal"]),
        ("ordered-equal", bits["one"], bits["one"]),
        ("signed-zero-equal", bits["neg-zero"], bits["pos-zero"]),
    )
    sign_rows = (
        ("rs2-positive-sign", bits["qnan-payload"], bits["one"]),
        ("rs2-negative-sign", bits["qnan-payload"], bits["neg-one"]),
        signed_zero,
    )
    minmax_common = (
        snan_one,
        qnan_one,
        ("same-snan-payload", bits["snan"], bits["snan"]),
        ("same-qnan-payload", bits["qnan-payload"], bits["qnan-payload"]),
        signed_zero,
    )
    minmax_number_common = (
        snan_one,
        qnan_one,
        signed_zero,
    )
    inexact_quotient = (
        "inexact-quotient",
        bits["one"],
        bits["two"] + (1 << (IEEE_FORMAT_BY_PRECISION[precision][2] - 1)),
    )
    rows: dict[str, tuple[tuple[str, int, int], ...]] = {
        "add": (
            snan_one,
            ("qnan-payload", bits["qnan-payload"], bits["one"]),
            ("opposite-inf", bits["pos-inf"], bits["neg-inf"]),
            ("exact-cancel", bits["one"], bits["neg-one"]),
            signed_zero,
            ("min-subnormal", bits["min-subnormal"], bits["min-subnormal"]),
            ("max-finite", bits["max-finite"], bits["max-finite"]),
        ),
        "sub": (
            snan_one,
            ("qnan-payload", bits["qnan-payload"], bits["one"]),
            ("same-inf", bits["pos-inf"], bits["pos-inf"]),
            ("exact-cancel", bits["one"], bits["one"]),
            signed_zero,
            (
                "opposite-max-finite",
                bits["max-finite"],
                bits["max-finite"] | (bits["neg-inf"] ^ bits["pos-inf"]),
            ),
            ("min-subnormal", bits["min-subnormal"], bits["min-subnormal"]),
        ),
        "mul": (
            snan_one,
            ("zero-inf", bits["pos-zero"], bits["pos-inf"]),
            ("inf-one", bits["pos-inf"], bits["one"]),
            ("signed-zero", bits["neg-zero"], bits["one"]),
            ("max-finite", bits["max-finite"], bits["max-finite"]),
            ("min-normal", bits["min-normal"], bits["min-normal"]),
        ),
        "div": (
            snan_one,
            inexact_quotient,
            ("zero-zero", bits["pos-zero"], bits["pos-zero"]),
            ("inf-inf", bits["pos-inf"], bits["pos-inf"]),
            ("finite-zero", bits["one"], bits["pos-zero"]),
            ("finite-inf", bits["one"], bits["pos-inf"]),
            ("signed-zero", bits["neg-zero"], bits["one"]),
            ("overflow", bits["max-finite"], bits["min-normal"]),
            ("underflow", bits["min-normal"], bits["max-finite"]),
        ),
        **dict.fromkeys(("copy-sign", "invert-sign"), sign_rows),
        "xor-sign": (
            ("same-sign", bits["qnan-negative"], bits["neg-one"]),
            ("opposite-sign", bits["qnan-negative"], bits["one"]),
            signed_zero,
        ),
        "minimum": minmax_common + (
            ("ordered-left-min", bits["min-normal"], bits["max-finite"]),
            ("ordered-right-min", bits["max-finite"], bits["min-normal"]),
        ),
        "maximum": minmax_common + (
            ("ordered-left-max", bits["max-finite"], bits["min-normal"]),
            ("ordered-right-max", bits["min-normal"], bits["max-finite"]),
        ),
        "minimum-number": minmax_number_common + (
            ("ordered-left-min", bits["min-normal"], bits["max-finite"]),
            ("ordered-right-min", bits["max-finite"], bits["min-normal"]),
        ),
        "maximum-number": minmax_number_common + (
            ("ordered-left-max", bits["max-finite"], bits["min-normal"]),
            ("ordered-right-max", bits["min-normal"], bits["max-finite"]),
        ),
        "equal": (
            snan_one,
            qnan_one,
            ("ordered-equal", bits["one"], bits["one"]),
            ("ordered-not-equal", bits["min-normal"], bits["max-finite"]),
            ("signed-zero-equal", bits["neg-zero"], bits["pos-zero"]),
        ),
        **dict.fromkeys(
            ("less-than", "less-equal", "less-than-quiet", "less-equal-quiet"),
            ordered_comparison_rows,
        ),
    }
    selected = rows.get(operation)
    if selected is None:
        if sources >= 3 and operation in {"fused", "madd", "msub", "nmsub", "nmadd"}:
            fusion_addend = (
                bits["neg-one"] if operation in {"madd", "nmsub"}
                else bits["one"]
            )
            return (
                row("snan", bits["snan"], bits["one"], bits["pos-zero"]),
                row("snan-negative", bits["snan-negative"], bits["one"], bits["pos-zero"]),
                row("qnan-payload-negative", bits["qnan-payload-negative"], bits["one"], bits["pos-zero"]),
                row("zero-inf", bits["pos-zero"], bits["pos-inf"], bits["one"]),
                row("qnan-zero-inf", bits["qnan"], bits["pos-zero"], bits["pos-inf"]),
                row("zero-inf-qnan", bits["pos-zero"], bits["pos-inf"], bits["qnan"]),
                row(
                    "exact-cancel", bits["one"], bits["one"],
                    bits["neg-one"] if operation in {"madd", "nmsub"}
                    else bits["one"],
                ),
                row("max-finite", bits["max-finite"], bits["max-finite"], bits["neg-one"]),
                row("signed-zero", bits["neg-zero"], bits["one"], bits["pos-zero"]),
                row("min-subnormal", bits["min-subnormal"], bits["one"], bits["pos-zero"]),
                row("underflow", bits["min-normal"], bits["min-normal"], bits["pos-zero"]),
                row("fusion-difference", bits["one"] + 1, bits["one"] - 2, fusion_addend),
            )
        return ()
    selected_rows = list(selected)
    if operation in {"add", "sub"}:
        width, exponent_bits, fraction_bits = IEEE_FORMAT_BY_PRECISION[precision]
        bias = (1 << (exponent_bits - 1)) - 1
        half_ulp = (bias - fraction_bits - 1) << fraction_bits
        sign = 1 << (width - 1)
        selected_rows.extend(
            (name, bits["one"], value | sign if operation == "sub" else value)
            for name, value in (
                ("rounding-just-below", half_ulp - 1),
                ("rounding-halfway", half_ulp),
                ("rounding-just-above", half_ulp + 1),
            )
        )
    selected_rows.extend(
        (
            ("snan-negative", bits["snan-negative"], bits["one"]),
            ("qnan-payload-negative", bits["qnan-payload-negative"], bits["one"]),
            ("subnormal-normal", bits["min-subnormal"], bits["one"]),
        )
    )
    return tuple(row(name, left, right) for name, left, right in selected_rows)


def nan_box_carriers(precision: int) -> tuple[tuple[str, int], ...]:
    """Single-precision payloads that are not NaN-boxed into a wider register."""
    if precision not in {11, 24, 53}:
        return ()
    bits = ieee_bit_classes(precision)
    width = IEEE_FORMAT_BY_PRECISION[precision][0]
    upper_mask = (1 << width) - 1
    return (
        ("nonboxed-zero-upper", bits["one"]),
        ("nonboxed-partial-upper", ((upper_mask - 1) << width) | bits["one"]),
        ("boxed-control", (upper_mask << width) | bits["one"]),
    )

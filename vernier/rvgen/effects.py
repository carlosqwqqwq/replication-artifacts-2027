"""Effect projection: structural pattern -> EffectSchema.

The generator must not grow when the ISA grows.  A form's *effect* is therefore
derived from structural facts the frozen catalog already carries -- major
opcode, funct fields, format suffix, operand shape -- rather than from a
per-instruction table.  Extensions that reuse an existing shape (Zbb rotates,
Zbkb pack, Zknh hashes, Zfa arithmetic ...) cost nothing to support.

A form that matches no value-semantic class is not an error: it still owns a
conservative encoding-only descriptor and encoding-legality obligations built
from structure alone.  This is the surface where translators forget to
implement or gate rare instructions, without pretending that a missing state
observer proves the instruction's value semantics.
"""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache
import re

from ..riscv_catalog import (
    OFFICIAL_ALL_CATALOG_BY_MNEMONIC,
    OFFICIAL_SELECTED_SCALAR_BY_MNEMONIC,
    OfficialForm,
)
from ..riscv_encoding import (
    assembly_mnemonic,
    is_register_operand_group,
    operand_groups,
    operand_kind,
    register_domain,
)
from ..spec_definedness import (
    _PRIVILEGED_STATE_EXTENSIONS,
    SemanticRealization,
    enabled_extensions,
    semantic_realizations_for_form,
)
from .boundaries import vector_state_boundary

MAJOR_OPCODE_MASK = 0x7F
FUNCT3_MASK = 0x7000
FUNCT7_MASK = 0xFE000000
LOAD = 0x03
OP_IMM = 0x13
AUIPC = 0x17
OP_IMM_32 = 0x1B
STORE = 0x23
AMO = 0x2F
OP = 0x33
LUI = 0x37
OP_32 = 0x3B
MADD = 0x43
MSUB = 0x47
NMSUB = 0x4B
NMADD = 0x4F
OP_FP = 0x53
BRANCH = 0x63
JALR = 0x67
JAL = 0x6F
SYSTEM = 0x73
MISC_MEM = 0x0F


_BRANCH_OPERATION_BY_FUNCT3 = {
    0x0: "beq",
    0x1: "bne",
    0x4: "blt",
    0x5: "bge",
    0x6: "bltu",
    0x7: "bgeu",
}

_FP_COMPARE_OPERATION_BY_FUNCT3 = {
    0x0: "less-equal",
    0x1: "less-than",
    0x2: "equal",
    0x4: "less-equal-quiet",
    0x5: "less-than-quiet",
}

_FP_SIGN_OPERATION_BY_FUNCT3 = {
    0x0: "copy-sign",
    0x1: "invert-sign",
    0x2: "xor-sign",
}

_FP_MINMAX_OPERATION_BY_FUNCT3 = {
    0x0: "minimum",
    0x1: "maximum",
    0x2: "minimum-number",
    0x3: "maximum-number",
}

_CSR_OPERATION_BY_FUNCT3 = {
    0x1: "write",
    0x2: "set",
    0x3: "clear",
    0x5: "write-immediate",
    0x6: "set-immediate",
    0x7: "clear-immediate",
}

_FLAGLESS_FP_OPERATIONS = frozenset(
    {"literal", "move", "move-pair", "copy-sign", "invert-sign", "xor-sign"}
)


@dataclass(frozen=True)
class EffectSchema:
    """Parameters of what a form does, never which form it is."""

    kind: str
    width: int = 64
    signed: bool = True
    precision: int = 0
    # Projection and conversion forms can read a narrower domain than they
    # write.  Keeping both facts here prevents boundary formulae and seeders
    # from guessing from a mnemonic or from the result format.
    source_width: int = 0
    source_signed: bool = True
    source_precision: int = 0
    writeback: str = "gpr"
    observer: str = "gpr"
    immediate_field: str | None = None
    memory_width: int = 0
    # The operation is an ISA fact used where boundary formulae really differ
    # inside one broad effect kind (for example MUL versus DIV).  It is not an
    # instruction-name dispatch table.
    operation: str = "generic"
    # Architectural state owner.  This is a small discriminator used by the
    # shared intent/materializer path; it is not a new family-specific plugin.
    state_domain: str = "scalar"

    @property
    def observes_fflags(self) -> bool:
        return self.observer in {"rawbits-plus-fflags", "gpr-plus-fflags"}


def disposition_for_schema(
    schema: EffectSchema | None,
    lane: str,
    *,
    has_sink: bool = True,
) -> str:
    """Return the one structural disposition used by all RVGEN projections."""

    if schema is None:
        return "definedness-outcome"
    kind = str(schema.kind)
    if kind == "encoding-only":
        return "encoding-only"
    if str(lane) == "legality-expected-trap":
        return "definedness-outcome"
    if kind in {"privileged-system", "instruction-memory", "atomic-rmw", "cache-block", "fence"}:
        return "state-environment-pending"
    if schema.state_domain in {"vector", "vector-memory", "csr", "reservation"}:
        return "state-environment-pending"
    if kind == "may-be-operation" and schema.operation == "write-zero" and has_sink:
        return "value-normal"
    if kind == "may-be-operation" or schema.operation == "nop":
        return "state-ready"
    if has_sink and schema.writeback != "none":
        return "value-normal"
    return "state-ready"


def rv32_xregister_pair_roles(
    form: OfficialForm,
    schema: EffectSchema | None,
    *,
    xlen: int,
    register_carrier: str,
) -> tuple[str, ...]:
    """Return FP operand roles carried by an RV32 x-register pair.

    Zdinx keeps the ordinary OP-FP encoding but maps a 64-bit FP value to an
    aligned ``xN/xN+1`` pair.  The catalog/register-domain facts identify the
    FP roles; precision is the only width discriminator, so this predicate is
    shared by intents, materialization and admission instead of growing a
    mnemonic list.  ``D`` is the only currently modelled paired precision
    (53-bit significand / 64-bit storage); unsupported wider precision stays
    fail-closed at the caller.
    """
    if (
        schema is None
        or type(xlen) is not int
        or xlen != 32
        or type(register_carrier) is not str
        or register_carrier != "gpr"
        or schema.kind not in {
            "fp-arith",
            "fp-fused",
            "fp-compare",
            "fp-classify",
            "fp-convert",
        }
        or 53 not in {int(schema.precision or 0), int(schema.source_precision or 0)}
    ):
        return ()
    source_precision = int(schema.source_precision or schema.precision or 0)
    if schema.kind == "fp-convert" and schema.operation == "int-to-fp":
        source_precision = 0
    return tuple(
        role
        for role in ("rd", "rs1", "rs2", "rs3")
        if (
            role in operand_groups(form)
            and register_domain(form, role) == "fpr"
            and (
                int(schema.precision or 0) == 53
                if role == "rd"
                else source_precision == 53
            )
        )
    )


@cache
def schema_for_realization(
    form: OfficialForm,
    schema: EffectSchema | None,
    *,
    xlen: int,
    register_carrier: str,
) -> EffectSchema | None:
    """Project a type-level schema onto one concrete register realization."""
    if schema is None or type(register_carrier) is not str or register_carrier not in {"gpr", "fpr"}:
        return None
    if type(xlen) is not int or xlen not in {32, 64}:
        return None
    if any(
        type(value := getattr(schema, name)) is not int or value < 0
        for name in ("width", "source_width", "memory_width", "precision", "source_precision")
    ):
        return None
    if schema.kind == "atomic-rmw" and schema.operation == "cas":
        width = int(schema.width or xlen)
        if width > xlen and width != 2 * xlen:
            return None
        concrete = replace(
            schema,
            writeback="paired-gpr" if width > xlen else "gpr",
            observer="gpr",
        )
    elif schema.kind == "stack-transfer":
        concrete = replace(schema, width=xlen, memory_width=xlen)
    elif schema.kind in {
        "integer-writeback", "integer-multiply", "integer-shift", "control-compare",
    }:
        width = int(schema.width or xlen)
        concrete = replace(schema, width=xlen if schema.kind == "control-compare" or width == 64 else width)
    else:
        concrete = schema
    if (
        xlen == 32
        and register_carrier == "gpr"
        and concrete.kind.startswith("fp")
        and max(int(concrete.precision or 0), int(concrete.source_precision or 0)) > 53
    ):
        return None
    pair_roles = rv32_xregister_pair_roles(
        form,
        concrete,
        xlen=xlen,
        register_carrier=register_carrier,
    )
    if concrete.writeback == "fpr" and (
        "rd" in pair_roles
        or register_carrier == "gpr" and concrete.kind.startswith("fp")
    ):
        return replace(
            concrete,
            writeback="paired-gpr" if "rd" in pair_roles else "gpr",
            observer="gpr-plus-fflags" if concrete.observes_fflags else "gpr",
        )
    return concrete


def vector_mask_operand(form: OfficialForm, schema: EffectSchema | None = None) -> str | None:
    """Return the mask carrier described by OP-V/OP-VE encoding facts."""
    # Vector loads/stores use the separate 0x07/0x27 major opcodes, but their
    # descriptor still exposes the same ``vm`` mask bit.  Keep the check
    # structural so ``vector-mask:vm:*`` rows are not rejected merely because
    # the risk form is a memory form rather than OP-V/OP-VE.
    if schema is None or _major(form) not in {0x07, 0x27, 0x57, 0x77}:
        return None
    groups = operand_groups(form)
    if "vm" in groups:
        return "v0"
    funct6 = (int(form.match) >> 26) & 0x3F
    if (
        funct6 == 0x17
        and form.mask & (1 << 25)
        and ((int(form.match) >> 25) & 1) == 0
    ):
        return "v0"
    # vadc/vmadc/vmsbc/vsbc encode the carry-in in v0 when vm=0.  Their
    # immediate and scalar forms have no ``vs1`` field, so the operation
    # shape alone cannot expose this state edge.  vm=1 is the no-carry form;
    # it must not be mistaken for a ``vs1`` mask source.
    if funct6 in range(0x10, 0x14):
        if not (form.mask & (1 << 25)):
            return None
        return "v0" if ((form.match >> 25) & 1) == 0 else None
    if not schema.operation.startswith("mask-or-carry") or not (form.mask & (1 << 25)):
        return None
    return "v0" if ((form.match >> 25) & 1) == 0 else "vs1"


def _vector_mask_result(form: OfficialForm, schema: EffectSchema | None) -> bool:
    mnemonic = assembly_mnemonic(form)
    if mnemonic.endswith(".mm") or mnemonic.startswith(
        _VECTOR_MASK_DEST_PREFIXES + _VECTOR_MASK_COMPARE_PREFIXES
    ):
        return True
    if schema is None or not schema.operation.startswith("mask-or-carry:"):
        return False
    funct6 = (int(form.match) >> 26) & 0x3F
    return funct6 in {0x11, 0x13} or 0x18 <= funct6 <= 0x1F


def _vector_mask_source_names(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> frozenset[str]:
    mnemonic = assembly_mnemonic(form)
    if mnemonic.endswith(".mm"):
        return frozenset({"vs1", "vs2"})
    if vector_mask_operand(form, schema) == "vs1":
        return frozenset({"vs1"})
    if (
        schema is not None
        and schema.operation in {
            "vector-to-scalar:funct6-10",
            "vector-unary:funct6-14",
        }
        and vector_mask_operand(form, schema) == "v0"
    ):
        return frozenset({"vs2"})
    return frozenset()


_VECTOR_INDEX_EEW_RE = re.compile(r"ei(?P<eew>8|16|32|64)")
_VECTOR_DATA_EEW_RE = re.compile(r"e(?P<eew>8|16|32|64|128)")
# Whole-register loads encode EEW after ``r`` (``vl8re8.v``), while stores
# use the shorter ``vs8r.v`` spelling.  NFIELDS, not EEW/SEW, owns the
# register-group size for both forms.
_VECTOR_WHOLE_REGISTER_RE = re.compile(
    r"v[ls](?P<count>[1248])r(?:e(?P<eew>8|16|32|64|128))?(?:\.|_)"
)
_VECTOR_WHOLE_MOVE_RE = re.compile(r"vmv(?P<count>[1248])r(?:\.|_)")
_VECTOR_EXTENSION_RE = re.compile(r"v[zs]ext\.vf(?P<factor>[248])")
_VECTOR_MASK_OR_CARRY_FUNCT6 = frozenset(
    (*range(0x10, 0x14), *range(0x17, 0x20))
)
_VECTOR_OVERLAP_RESTRICTED_PREFIXES = (
    "vslideup.",
    "vslide1up.",
    "vfslide1up.",
    "vrgather.",
    "vrgatherei16.",
    "vmsbf.",
    "vmsif.",
    "vmsof.",
    "viota.",
)
_VECTOR_MASK_DEST_PREFIXES = ("vmsbf.", "vmsif.", "vmsof.")
_VECTOR_MASK_COMPARE_PREFIXES = (
    "vmfeq.", "vmfge.", "vmfgt.", "vmfle.", "vmflt.", "vmfne.",
    "vmseq.", "vmsbc.", "vmsgt.", "vmsgtu.", "vmsle.", "vmsleu.",
    "vmslt.", "vmsltu.", "vmsne.",
)
_VECTOR_VLEN_BITS = 256


def _vector_crypto_family(form: OfficialForm) -> str | None:
    mnemonic = assembly_mnemonic(form)
    if mnemonic.startswith("vaes"):
        return "aes"
    if mnemonic.startswith("vsha2"):
        return "sha2"
    if mnemonic.startswith("vsm3"):
        return "sm3"
    if mnemonic.startswith("vsm4"):
        return "sm4"
    if mnemonic.startswith(("vgh", "vgm")):
        return "gcm"
    return None


def vector_crypto_default_state(form: OfficialForm) -> tuple[int, Fraction] | None:
    family = _vector_crypto_family(form)
    if family is None:
        return None
    return 32, Fraction(2 if family == "sm3" else 1, 1)


def vector_crypto_overlap_restricted(form: OfficialForm, source_name: str) -> bool:
    family = _vector_crypto_family(form)
    mnemonic = assembly_mnemonic(form)
    if family == "sha2":
        return source_name in {"vs1", "vs2"}
    if family == "sm3":
        return source_name == "vs2"
    return (
        family in {"aes", "sm4"}
        and mnemonic.endswith(".vs")
        and source_name == "vs2"
    )


def vector_default_state(form: OfficialForm) -> tuple[int, Fraction] | None:
    crypto = vector_crypto_default_state(form)
    if crypto is not None:
        return crypto
    mnemonic = assembly_mnemonic(form)
    if mnemonic.startswith(("vclmul.", "vclmulh.")):
        return 64, Fraction(1, 1)
    if mnemonic == "vfncvtbf16.sat.f.f.w":
        return 8, Fraction(1, 1)
    if "bf16" in mnemonic:
        return 16, Fraction(1, 1)
    extension = _VECTOR_EXTENSION_RE.fullmatch(mnemonic)
    if extension is not None:
        return 8 * int(extension.group("factor")), Fraction(1, 1)
    if (
        mnemonic.startswith(("vf", "vfw", "vfn", "vfcvt", "vfred", "vmf"))
        and not mnemonic.startswith("vfirst.")
    ):
        return 32, Fraction(1, 1)
    if _major(form) in {0x07, 0x27} and "v" in _extension_tokens(form):
        if (
            _VECTOR_WHOLE_REGISTER_RE.search(mnemonic) is None
            and _VECTOR_INDEX_EEW_RE.search(mnemonic) is None
            and mnemonic not in {"vlm.v", "vsm.v"}
        ):
            data = _VECTOR_DATA_EEW_RE.search(mnemonic)
            if data is not None and int(data.group("eew")) <= 64:
                return int(data.group("eew")), Fraction(1, 1)
    return None


def _vector_fp_form(form: OfficialForm, schema: EffectSchema | None) -> bool:
    if schema is None or schema.kind != "vector-data":
        return False
    mnemonic = assembly_mnemonic(form)
    return (
        mnemonic.startswith(("vf", "vfw", "vfn", "vfcvt", "vfred", "vmf"))
        and not mnemonic.startswith("vfirst.")
    )


_VECTOR_FP_STATELESS_PREFIXES = (
    "vfclass.", "vfmerge.", "vfmv.", "vfsgnj.", "vfsgnjn.",
    "vfsgnjx.", "vfslide1down.", "vfslide1up.",
)
_VECTOR_VXSAT_PREFIXES = ("vsadd.", "vsaddu.", "vssub.", "vssubu.")
_VECTOR_VXRM_PREFIXES = (
    "vaadd.", "vaaddu.", "vasub.", "vasubu.", "vssrl.", "vssra.",
)
_VECTOR_VXSAT_VXRM_PREFIXES = ("vsmul.", "vnclip.", "vnclipu.")
_VECTOR_ZVE64_EEW64_RESTRICTED_PREFIXES = (
    "vmulh.", "vmulhu.", "vmulhsu.", "vsmul.",
)


def vector_dynamic_state_keys(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> tuple[str, ...]:
    """Return vector CSR state written or consumed by one concrete form."""

    mnemonic = assembly_mnemonic(form)
    if _vector_fp_form(form, schema):
        return ("frm",) if mnemonic.startswith(_VECTOR_FP_STATELESS_PREFIXES) else ("fflags", "frm")
    if schema is None or schema.kind != "vector-data":
        return ()
    if mnemonic.startswith(_VECTOR_VXSAT_VXRM_PREFIXES):
        return ("vxsat", "vxrm")
    if mnemonic.startswith(_VECTOR_VXSAT_PREFIXES):
        return ("vxsat",)
    if mnemonic.startswith(_VECTOR_VXRM_PREFIXES):
        return ("vxrm",)
    return ()


def vector_fault_only_first(
    form: OfficialForm,
    schema: EffectSchema | None,
) -> bool:
    """Identify the unit-stride fault-only-first memory forms."""

    return (
        schema is not None
        and schema.kind == "vector-memory"
        and assembly_mnemonic(form) in {
            "vle8ff.v", "vle16ff.v", "vle32ff.v", "vle64ff.v",
        }
    )


def vector_extension_violation(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
) -> str | None:
    if schema is None or schema.kind != "vector-data":
        return None
    mnemonic = assembly_mnemonic(form)
    extension = _VECTOR_EXTENSION_RE.fullmatch(mnemonic)
    state = vector_state_boundary(boundary_class)
    default = vector_default_state(form)
    if extension is None or state is None or state.vill or default is None:
        return None
    sew, lmul = default
    if state.axis == "vtype":
        sew = int(state.sew or 0)
        lmul = state.lmul or Fraction(1, 1)
    factor = int(extension.group("factor"))
    if sew // factor < 8:
        return "vector-extension-source-eew"
    if lmul / factor < Fraction(1, 8):
        return "vector-extension-source-emul"
    return None


def vector_crypto_violation(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
    isa_profile: str | None = None,
) -> str | None:
    if schema is None or schema.kind != "vector-data":
        return None
    family = _vector_crypto_family(form)
    state = vector_state_boundary(boundary_class)
    default = vector_crypto_default_state(form)
    if family is None or state is None or state.vill or default is None:
        return None
    sew, lmul = default
    if state.axis == "vtype":
        sew = int(state.sew or 0)
        lmul = state.lmul or Fraction(1, 1)
        allowed_sew = (
            (32, 64)
            if family == "sha2"
            and (
                not isa_profile
                or enabled_extensions(str(isa_profile or ""))
                & {"zvkn", "zvknc", "zvkng", "zvknhb"}
            )
            else (32,)
        )
        if sew not in allowed_sew:
            return "vector-crypto-sew-reserved"
    egs = 8 if family == "sm3" else 4
    egw = 256 if family == "sm3" or (family == "sha2" and sew == 64) else 128
    if lmul * _VECTOR_VLEN_BITS < egw:
        return "vector-crypto-lmul-egw"
    if state.axis == "vstart" and int(state.vstart or 0) % egs:
        return "vector-crypto-vstart-reserved"
    vlmax = int(Fraction(_VECTOR_VLEN_BITS, sew) * lmul)
    if boundary_class == "vector-vl:zero":
        vl = 0
    elif boundary_class == "vector-vl:one":
        vl = 1
    elif boundary_class == "vector-vl:vlmax":
        vl = vlmax
    elif boundary_class.startswith("vector"):
        vl = min(8, vlmax)
    else:
        return None
    return "vector-crypto-vl-reserved" if vl % egs else None


def _vector_vtype_violation(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
) -> str | None:
    if schema is None or not boundary_class.startswith("vector-vtype:"):
        return None
    state = vector_state_boundary(boundary_class)
    if state is None:
        return "vector-state-boundary-unrecognized"
    mnemonic = assembly_mnemonic(form)
    if state.vill:
        whole_register = _VECTOR_WHOLE_REGISTER_RE.search(mnemonic) is not None
        config = schema.kind == "may-be-operation" and schema.operation.startswith(
            "vector-config"
        )
        # vill makes vtype-dependent vector operations illegal.  Whole-register
        # loads/stores and vset* are the architectural exceptions.
        if not config and not whole_register:
            return "vector-vill-requires-illegal"
    if mnemonic.startswith(("vclmul.", "vclmulh.")) and state.sew != 64:
        return "vector-clmul-sew-reserved"
    if mnemonic == "vfncvtbf16.sat.f.f.w":
        if state.sew != 8:
            return "vector-ofp8-sew-reserved"
    elif "bf16" in mnemonic and state.sew != 16:
        return "vector-bf16-sew-reserved"
    if state.sew == 8 and schema.kind == "vector-data" and mnemonic.startswith(
        ("vf", "vfw", "vfn", "vfcvt", "vfred", "vmf")
    ) and not mnemonic.startswith("vfirst.") and mnemonic != "vfncvtbf16.sat.f.f.w":
        return "vector-fp-sew8-requires-illegal"
    return None


def vector_profile_vtype_violation(
    isa_profile: str,
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
) -> str | None:
    """Classify a VTYPE width that the selected profile cannot support."""

    if schema is None or schema.state_domain not in {"vector", "vector-memory"}:
        return None
    if not boundary_class.startswith("vector"):
        return None
    state = vector_state_boundary(boundary_class)
    enabled = enabled_extensions(str(isa_profile or "").lower())
    max_elen = 64 if "v" in enabled or enabled & {"zve64x", "zve64f", "zve64d"} else (
        32 if enabled & {"zve32x", "zve32f"} else None
    )
    if max_elen is None or (state is not None and state.vill):
        return None
    default_state = vector_default_state(form)
    sew = int(
        state.sew
        if state is not None and state.sew is not None
        else default_state[0]
        if default_state is not None
        else 8
    )
    mnemonic = assembly_mnemonic(form)
    whole_register = _VECTOR_WHOLE_REGISTER_RE.search(mnemonic) is not None
    config = schema.kind == "may-be-operation" and schema.operation.startswith(
        "vector-config"
    )
    if (
        "v" not in enabled
        and "zve64x" in enabled
        and sew == 64
        and mnemonic.startswith(_VECTOR_ZVE64_EEW64_RESTRICTED_PREFIXES)
    ):
        return "vector-vtype-elen-limit"
    if not config and not whole_register:
        effective_eews = _vector_effective_eews(form, schema, sew)
        if sew > max_elen or any(width > max_elen for width in effective_eews):
            return "vector-vtype-elen-limit"
        if (
            _vector_fp_form(form, schema)
            and "zve64f" in enabled
            and "zve64d" not in enabled
            and any(width > 32 for width in effective_eews)
        ):
            return "vector-fp-sew-unsupported"
    if (
        not config
        and not whole_register
        and sew == 16
        and _vector_fp_form(form, schema)
        and "bf16" not in mnemonic
        and not enabled & {"zvfh", "zvfhmin"}
    ):
        return "vector-fp-sew-unsupported"
    return None


def _vector_group_count(emul: Fraction) -> int:
    """Return the architectural register count for a descriptor EMUL.

    Fractional LMUL/EMUL still names one physical register at minimum.  The
    descriptor parser keeps the exact Fraction so widening/narrowing checks can
    reject EMUL>8 without truncating it to an integer first.
    """
    numerator = int(emul.numerator)
    denominator = int(emul.denominator)
    return max(1, (numerator + denominator - 1) // denominator)


def _vector_data_group_factor(form: OfficialForm, field: str) -> int | Fraction:
    """Return the integer EMUL multiplier for a vector-data operand.

    ``1`` is the ordinary LMUL, ``2`` is a twice-wide operand and ``4`` is the
    Q-to-narrow conversion source.  These are format properties encoded by the
    operation suffix, not a per-mnemonic semantic table.
    """
    if field not in {"vd", "vs1", "vs2"}:
        return 1
    name = str(form.mnemonic)
    base, *suffix = name.split("_")
    if field == "vd" and (
        base.startswith(("vw", "vfw"))
        and not base.startswith("vfn")
    ):
        return 2
    if field != "vs2":
        return 1
    extension = _VECTOR_EXTENSION_RE.fullmatch(name.replace("_", "."))
    if extension is not None:
        return Fraction(1, int(extension.group("factor")))
    if any(part in {"wv", "wx", "wi", "wf"} for part in suffix):
        return 2
    if base.startswith("vfncvt") and suffix:
        if suffix[-1] == "q":
            return 4
        if suffix[-1] == "w":
            return 2
    return 1


def _vector_effective_eews(
    form: OfficialForm,
    schema: EffectSchema,
    sew: int,
) -> tuple[Fraction, ...]:
    mnemonic = assembly_mnemonic(form)
    if schema.kind == "vector-data":
        widths = []
        scalar_reduction = mnemonic.split(".", 1)[0] in {"vwredsum", "vwredsumu"}
        for operand in ("vd", "vs1", "vs2"):
            if operand not in operand_groups(form):
                continue
            if scalar_reduction and operand == "vd":
                continue
            if operand == "vs1" and mnemonic.startswith("vrgatherei16."):
                widths.append(Fraction(16))
            else:
                widths.append(Fraction(sew) * _vector_data_group_factor(form, operand))
        return tuple(widths)
    if schema.kind == "vector-memory":
        if _VECTOR_WHOLE_REGISTER_RE.search(mnemonic) is not None:
            return ()
        indexed = _VECTOR_INDEX_EEW_RE.search(mnemonic)
        if indexed is not None:
            return (Fraction(sew), Fraction(int(indexed.group("eew"))))
        data = _VECTOR_DATA_EEW_RE.search(mnemonic)
        return (Fraction(int(data.group("eew")) if data is not None else sew),)
    return ()


def vector_register_group_size(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
    field: str,
) -> int:
    """Return the architectural register-group alignment for one V field.

    This is deliberately a small structural legality fact.  It covers the
    register-group rules that can be derived from the emitted mnemonic, VTYPE
    edge and catalog fields; it does not attempt to model the whole V spec.
    """
    if schema is None or schema.state_domain not in {"vector", "vector-memory"}:
        return 1
    state = vector_state_boundary(boundary_class)
    default_state = vector_default_state(form)
    sew = (
        int(state.sew)
        if state is not None and state.sew is not None
        else default_state[0]
        if default_state is not None
        else 8
    )
    lmul = (
        state.lmul
        if state is not None and state.lmul is not None
        else default_state[1]
        if default_state is not None
        else Fraction(1, 1)
    )
    mnemonic = assembly_mnemonic(form)
    if mnemonic in {"vlm.v", "vsm.v"} and field in {"vd", "vs3"}:
        return 1
    if field == "vs2" and mnemonic.endswith(".vs") and _vector_crypto_family(form) in {"aes", "sm4"}:
        return 1
    if (
        schema.kind == "vector-data"
        and field == "vs1"
        and mnemonic.startswith("vrgatherei16.")
    ):
        return _vector_group_count(lmul * 16 / sew)
    whole_move = _VECTOR_WHOLE_MOVE_RE.search(mnemonic)
    if whole_move and field in {"vd", "vs2"}:
        return int(whole_move.group("count"))
    if (
        schema.kind == "vector-data"
        and field in {"vd", "vs2"}
        and schema.operation in {
            "vector-to-scalar:funct6-10",
            "vector-scalar:funct6-10",
        }
        and vector_mask_operand(form, schema) is None
    ):
        return 1
    if (
        schema.kind == "vector-data"
        and field == "vs2"
        and schema.operation in {
            "vector-to-scalar:funct6-10",
            "vector-unary:funct6-14",
        }
        and vector_mask_operand(form, schema) == "v0"
    ):
        return 1
    if mnemonic.startswith(_VECTOR_MASK_DEST_PREFIXES) and field in {"vd", "vs2"}:
        return 1
    if schema.kind == "vector-data" and field == "vd" and _vector_mask_result(form, schema):
        return 1
    whole = _VECTOR_WHOLE_REGISTER_RE.search(mnemonic)
    if whole and field in {"vd", "vs3"}:
        return int(whole.group("count"))
    if schema.kind == "vector-memory" and field == "vs2":
        indexed = _VECTOR_INDEX_EEW_RE.search(mnemonic)
        if indexed is not None:
            return _vector_group_count(lmul * int(indexed.group("eew")) / sew)
    if schema.kind == "vector-memory" and field not in {"vd", "vs3"}:
        return 1
    if schema.kind == "vector-data":
        base = mnemonic.split(".", 1)[0]
        if base.startswith(("vred", "vfred", "vfwred", "vwred")) and field in {"vd", "vs1"}:
            return 1
        if schema.operation.startswith("mask-or-carry:"):
            funct6 = (int(form.match) >> 26) & 0x3F
            if mnemonic.endswith(".mm"):
                return 1
            if field == "vd" and funct6 not in {0x10, 0x12, 0x17}:
                return 1
            if funct6 == 0x17 and field == "vs1" and vector_mask_operand(form, schema) == "vs1":
                return 1
        factor = _vector_data_group_factor(form, field)
        return _vector_group_count(lmul * factor)
    data = _VECTOR_DATA_EEW_RE.search(mnemonic) if schema.kind == "vector-memory" else None
    eew = int(data.group("eew")) if data is not None else sew
    numerator = lmul * eew
    # Segment forms use NFIELDS to describe the number of fields, not the
    # alignment of each field's register group.  The V manual defines EMUL
    # per field as (EEW / SEW) * LMUL; multiplying by NFIELDS here would reject
    # legal encodings such as vle32.v with nf=1 (vlseg2e32.v).
    return _vector_group_count(numerator / sew)


def vector_vstart_violation(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
) -> str | None:
    state = vector_state_boundary(boundary_class)
    if (
        state is None
        or state.axis != "vstart"
        or int(state.vstart or 0) == 0
        or schema is None
    ):
        return None
    mnemonic = assembly_mnemonic(form)
    whole = _VECTOR_WHOLE_REGISTER_RE.search(mnemonic)
    whole_move = _VECTOR_WHOLE_MOVE_RE.search(mnemonic)
    if whole is not None or whole_move is not None:
        default = vector_default_state(form)
        sew = default[0] if default is not None else 8
        if state.sew is not None:
            sew = int(state.sew)
        evl = (
            _VECTOR_VLEN_BITS * int(whole.group("count")) // int(whole.group("eew") or 8)
            if whole is not None
            else _VECTOR_VLEN_BITS * int(whole_move.group("count")) // int(sew)
        )
        return "vector-vstart-nonzero-illegal" if int(state.vstart) >= evl else None
    if mnemonic in {"vlm.v", "vsm.v"}:
        default = vector_default_state(form)
        sew = default[0] if default is not None else 8
        lmul = default[1] if default is not None else Fraction(1, 1)
        vlmax = int(Fraction(_VECTOR_VLEN_BITS, sew) * lmul)
        evl = (min(8, vlmax) + 7) // 8
        return "vector-vstart-nonzero-illegal" if int(state.vstart) >= evl else None
    if schema.kind not in {"vector-data", "vector-memory"}:
        return None
    default = vector_default_state(form)
    sew = default[0] if default is not None else 8
    lmul = default[1] if default is not None else Fraction(1, 1)
    if state.sew is not None:
        sew = int(state.sew)
        lmul = state.lmul or Fraction(1, 1)
    if int(state.vstart) >= int(Fraction(_VECTOR_VLEN_BITS, sew) * lmul):
        return "vector-vstart-nonzero-illegal"
    if schema.kind != "vector-data":
        return None
    if (
        mnemonic == "vcompress.vm"
        or mnemonic.startswith(("vred", "vfred", "vwred", "vfwred"))
        or mnemonic.startswith(("vmsbf.", "vmsif.", "vmsof.", "viota."))
        or (
            schema.operation == "vector-to-scalar:funct6-10"
            and "vm" in operand_groups(form)
        )
    ):
        return "vector-vstart-nonzero-illegal"
    return None


def vector_register_violation(
    form: OfficialForm,
    schema: EffectSchema | None,
    boundary_class: str,
    operands: dict[str, int],
    isa_profile: str | None = None,
) -> str | None:
    """Return the first concrete V register-group legality violation."""
    if type(boundary_class) is not str:
        return "vector-state-boundary-unrecognized"
    boundary = str(boundary_class)
    state = vector_state_boundary(boundary_class)
    if state is None and boundary.startswith(
        ("vector-vl:", "vector-vtype:", "vector-vstart:", "vector-mask:", "vector-tail:")
    ):
        return "vector-state-boundary-unrecognized"
    if (extension_violation := vector_extension_violation(
        form, schema, boundary
    )) is not None:
        return extension_violation
    if (crypto_violation := vector_crypto_violation(
        form, schema, boundary, isa_profile
    )) is not None:
        return crypto_violation
    if (vstart_violation := vector_vstart_violation(form, schema, boundary)) is not None:
        return vstart_violation
    if (boundary_violation := _vector_vtype_violation(form, schema, boundary)) is not None:
        return boundary_violation
    default_state = vector_default_state(form)
    lmul = (
        state.lmul
        if state is not None and state.lmul is not None
        else default_state[1]
        if default_state is not None
        else Fraction(1, 1)
    )
    sew = (
        int(state.sew)
        if state is not None and state.sew is not None
        else default_state[0]
        if default_state is not None
        else 8
    )
    nf = operands.get("nf", 0)
    if type(nf) is not int or not 0 <= nf <= 7:
        return "vector-segment-field-count-invalid"
    mnemonic = assembly_mnemonic(form)
    for operand in operand_groups(form):
        if operand not in {"vd", "vs1", "vs2", "vs3"} or operand not in operands:
            continue
        value = operands[operand]
        if type(value) is not int or not 0 <= value < 32:
            return f"vector-register-value-invalid:{operand}"
        if schema is not None and schema.kind == "vector-memory":
            whole = _VECTOR_WHOLE_REGISTER_RE.search(mnemonic)
            emul = None
            if whole is None and operand in {"vd", "vs3"}:
                data = _VECTOR_DATA_EEW_RE.search(mnemonic)
                emul = lmul * (int(data.group("eew")) if data is not None else sew) / sew
            elif whole is None and operand == "vs2":
                indexed = _VECTOR_INDEX_EEW_RE.search(mnemonic)
                if indexed is not None:
                    emul = lmul * int(indexed.group("eew")) / sew
            if emul is not None and not Fraction(1, 8) <= emul <= 8:
                return f"vector-register-emul:{operand}"
        group = vector_register_group_size(form, schema, boundary_class, operand)
        if schema is not None and schema.kind == "vector-data":
            if operand == "vs1" and mnemonic.startswith("vrgatherei16."):
                emul = lmul * 16 / sew
                if not Fraction(1, 8) <= emul <= 8:
                    return "vector-register-emul:vs1"
            factor = _vector_data_group_factor(form, operand)
            if factor > 1 and lmul * factor > 8:
                if operand == "vd" and factor == 2:
                    return "vector-widening-destination-emul"
                return f"vector-register-emul:{operand}"
        if value + group > 32:
            return f"vector-register-group-range:{operand}"
        if group > 1 and value % group:
            return f"vector-register-group-alignment:{operand}"
        if schema is not None and schema.kind == "vector-memory" and operand in {"vd", "vs3"}:
            fields = nf + 1
            if fields > 1 and group * fields > 8:
                return "vector-segment-group-span"
            if fields > 1 and value + group * fields > 32:
                return f"vector-segment-group-range:{operand}"
    if schema is not None and schema.kind == "vector-data":
        source_names = [
            name for name in ("vs1", "vs2", "vs3") if name in operands
        ]
        mask_sources = _vector_mask_source_names(form, schema)
        if vector_mask_operand(form, schema) == "v0" and (
            "vm" not in operand_groups(form) or operands.get("vm") == 0
        ):
            source_names.append("v0")
        source_sets: dict[str, set[int]] = {}
        source_eew: dict[str, Fraction] = {}
        for name in source_names:
            if name == "v0":
                source_sets[name] = {0}
                source_eew[name] = Fraction(1, 1)
                continue
            group = vector_register_group_size(form, schema, boundary_class, name)
            source_sets[name] = set(
                range(int(operands[name]), min(32, int(operands[name]) + group))
            )
            if name in mask_sources:
                source_eew[name] = Fraction(1, 1)
            elif (
                name == "vs1"
            and assembly_mnemonic(form).startswith("vrgatherei16.")
            ):
                source_eew[name] = Fraction(16)
            else:
                source_eew[name] = Fraction(sew) * _vector_data_group_factor(form, name)
        for index, left_name in enumerate(source_names):
            for right_name in source_names[index + 1 :]:
                if source_eew[left_name] == source_eew[right_name]:
                    continue
                if source_sets[left_name].intersection(source_sets[right_name]):
                    return "vector-source-eew-overlap"
    if schema is not None and schema.kind == "vector-data" and "vd" in operands:
        destination_group = vector_register_group_size(form, schema, boundary_class, "vd")
        destination = set(range(int(operands["vd"]), min(32, int(operands["vd"]) + destination_group)))
        for source_name in ("vs1", "vs2", "vs3"):
            if source_name not in operands:
                continue
            source_group = vector_register_group_size(form, schema, boundary_class, source_name)
            source = set(
                range(
                    int(operands[source_name]),
                    min(32, int(operands[source_name]) + source_group),
                )
            )
            if not destination.intersection(source):
                continue
            mnemonic = assembly_mnemonic(form)
            destination_factor = _vector_data_group_factor(form, "vd")
            source_factor = _vector_data_group_factor(form, source_name)
            if destination_factor > source_factor and lmul * source_factor < 1:
                return "vector-register-overlap"
            if mnemonic.startswith(_VECTOR_OVERLAP_RESTRICTED_PREFIXES) or vector_crypto_overlap_restricted(
                form, source_name
            ):
                return "vector-register-overlap"
            if str(form.mnemonic) == "vcompress_vm":
                return "vector-register-overlap"
            if destination_group > source_group and max(destination) != max(source):
                return "vector-register-overlap"
            if destination_group < source_group and min(destination) != min(source):
                return "vector-register-overlap"
    if schema is not None and schema.kind == "vector-memory" and "vs2" in operands:
        destination_name = "vd" if schema.operation == "vector-load" else "vs3"
        if destination_name in operands:
            destination_group = vector_register_group_size(
                form, schema, boundary_class, destination_name
            )
            index_group = vector_register_group_size(
                form, schema, boundary_class, "vs2"
            )
            indexed = _VECTOR_INDEX_EEW_RE.search(assembly_mnemonic(form))
            fields = nf + 1
            destination = set(
                range(
                    int(operands[destination_name]),
                    min(32, int(operands[destination_name]) + destination_group * fields),
                )
            )
            index = set(
                range(
                    int(operands["vs2"]),
                    min(32, int(operands["vs2"]) + index_group),
                )
            )
            if destination.intersection(index):
                if indexed is None:
                    return "vector-register-overlap"
                data_eew = Fraction(sew)
                index_eew = Fraction(int(indexed.group("eew")))
                if schema.operation == "vector-load" and fields > 1:
                    return "vector-register-overlap"
                if schema.operation != "vector-load" and data_eew != index_eew:
                    return "vector-source-eew-overlap"
                if schema.operation == "vector-load" and data_eew > index_eew:
                    index_emul = lmul * index_eew / sew
                    if index_emul < 1 or max(destination) != max(index):
                        return "vector-register-overlap"
                elif schema.operation == "vector-load" and data_eew < index_eew:
                    if min(destination) != min(index):
                        return "vector-register-overlap"
    if schema is not None and schema.kind == "vector-memory" and operands.get("vm") == 0:
        source_names = ["vs2"]
        if schema.operation != "vector-load":
            source_names.append("vs3")
        for name in source_names:
            if name not in operands:
                continue
            group = vector_register_group_size(form, schema, boundary_class, name)
            if int(operands[name]) <= 0 < int(operands[name]) + group:
                return (
                    "vector-mask-destination-overlap"
                    if name == "vs3"
                    else "vector-source-eew-overlap"
                )
    if schema is not None and schema.state_domain in {"vector", "vector-memory"}:
        mnemonic = assembly_mnemonic(form)
        destination = operands.get("vd")
        if destination is not None:
            masked = operands.get("vm") == 0
            mask_result = _vector_mask_result(form, schema)
            # Only an implicit v0 carry/mask reserves vd=0.  vcompress.vm
            # carries its mask in an explicit vs1 field, so vd=0 is legal
            # whenever it does not overlap that source group.
            implicit_mask = (
                vector_mask_operand(form, schema) == "v0"
                and "vm" not in operand_groups(form)
            )
            if (implicit_mask or masked) and not mask_result and int(destination) == 0:
                return "vector-mask-destination-overlap"
    return None



@dataclass(frozen=True)
class GenerationRule:
    """The generated registration binding for one frozen catalog form.

    Catalog owns ISA facts; this value only records which generic formula
    family receives those facts.  It is deliberately computed, never a second
    hand-maintained form list.
    """

    form: OfficialForm
    schema: EffectSchema | None
    realizations: tuple[SemanticRealization, ...] = field(default_factory=tuple)

    @property
    def rule_id(self) -> str:
        return f"{self.form.mnemonic}:{self.schema.kind if self.schema is not None else 'legality-only'}"


def trap_outcome_kind_for_form(form: OfficialForm) -> str | None:
    """environment-call/breakpoint from the fixed SYSTEM funct12 slots.

    ecall/ebreak are the SYSTEM rows whose funct12 field is fully pinned by
    the catalog mask (0x000/0x001), which CSR forms never satisfy because
    their csr field keeps bits 31-20 variable.  c_ebreak is the fully pinned
    16-bit 0x9002 codeword.  No mnemonic list is needed.
    """
    if form.encoding_length_bytes == 2:
        return "breakpoint" if form.mask == 0xFFFF and form.match == 0x9002 else None
    if (form.match & MAJOR_OPCODE_MASK) != SYSTEM or (form.mask & 0xFFF00000) != 0xFFF00000:
        return None
    return {0x000: "environment-call", 0x001: "breakpoint"}.get(
        (form.match >> 20) & 0xFFF
    )


def _major(form: OfficialForm) -> int | None:
    if form.encoding_length_bytes != 4:
        return None
    return form.match & MAJOR_OPCODE_MASK


def _fp_format(form: OfficialForm) -> int | None:
    """The fmt field of an OP-FP style form (0=S, 1=D, 2=H, 3=Q)."""
    if form.encoding_length_bytes != 4:
        return None
    if not (form.mask & (0x3 << 25)):
        return None
    return (form.match >> 25) & 0x3


def _register_shift_operation(funct7: int, funct3: int | None) -> str | None:
    """Register-carried shift/rotate/bit-index effects from encoding fields."""
    return {
        (0x00, 0x1): "shift-left",
        (0x00, 0x5): "shift-right-logical",
        (0x20, 0x5): "shift-right-arithmetic",
        (0x30, 0x1): "rotate-left",
        (0x30, 0x5): "rotate-right",
        (0x14, 0x1): "bit-set",
        (0x24, 0x1): "bit-clear",
        (0x34, 0x1): "bit-invert",
        (0x24, 0x5): "bit-extract",
    }.get((funct7, funct3))


def _extension_tokens(form: OfficialForm) -> frozenset[str]:
    """Normalize catalog extension provenance without introducing a form table."""
    tokens: set[str] = set()
    for source in form.source_extensions:
        name = str(source).removeprefix("rv32_").removeprefix("rv64_").removeprefix("rv_")
        tokens.add(name)
        tokens.update(part for part in name.split("_") if part)
    return frozenset(tokens)


def _scalar_crypto_width(
    form: OfficialForm,
    *,
    major: int | None,
    funct3: int | None,
    fixed_function: int,
    funct7: int,
    fields: frozenset[str],
    extensions: frozenset[str],
) -> int | None:
    """Recognize scalar crypto encodings whose value formulas are implemented."""
    if major in {OP, OP_32} and funct3 == 0:
        if "bs" in fields:
            if (
                extensions & {"zkne", "zknd"}
                and funct7 in {0x11, 0x13, 0x15, 0x17}
            ) or (
                "zksed" in extensions and funct7 in {0x18, 0x1A}
            ):
                return 32
        if (
            str(form.xlen) == "rv64"
            and extensions & {"zkne", "zknd"}
            and funct7 in {0x19, 0x1B, 0x1D, 0x1F, 0x3F}
        ):
            return 64
        if (
            str(form.xlen) == "rv32"
            and "zknh" in extensions
            and funct7 in {0x28, 0x29, 0x2A, 0x2B, 0x2E, 0x2F}
        ):
            return 32
    if major == OP_IMM and funct3 == 0x1:
        if "zknh" in extensions and fixed_function in {
            0x100, 0x101, 0x102, 0x103,
        }:
            return 32
        if (
            "zknh" in extensions
            and str(form.xlen) == "rv64"
            and fixed_function in {0x104, 0x105, 0x106, 0x107}
        ):
            return 64
        if "zksh" in extensions and fixed_function in {0x108, 0x109}:
            return 32
        if (
            str(form.xlen) == "rv64"
            and "zknd" in extensions
            and fixed_function == 0x300
        ):
            return 64
        if (
            str(form.xlen) == "rv64"
            and extensions & {"zkne", "zknd"}
            and fixed_function == 0x310
        ):
            return 64
    return None


def _rare_integer_operation(
    form: OfficialForm,
    *,
    major: int | None,
    funct3: int | None,
    fields: frozenset[str],
) -> str | None:
    """Classify rare scalar effects by shared ISA extension semantics.

    The catalog already owns the exact form and field layout.  This helper
    adds only the small number of formula classes whose input discontinuities
    differ from ordinary ALU operations; it deliberately does not name every
    instruction or create a template per extension member.
    """
    extensions = _extension_tokens(form)
    source_count = sum(1 for field in fields if field.startswith("rs"))
    fixed_function = (form.match >> 20) & 0xFFF
    funct7 = (form.match >> 25) & 0x7F
    if "zba" in extensions and major in {OP, OP_32}:
        if funct7 == 0x04 and major == OP_32 and funct3 == 0x0:
            return "add-unsigned-word"
        if funct7 == 0x10 and funct3 in {0x2, 0x4, 0x6}:
            shift = {0x2: 1, 0x4: 2, 0x6: 3}[funct3]
            return (
                f"shift-add-unsigned-word:{shift}"
                if major == OP_32 else f"shift-add:{shift}"
            )
    if "zicond" in extensions and major == OP and funct7 == 0x07:
        return {0x5: "conditional-zero:eqz", 0x7: "conditional-zero:nez"}.get(funct3)
    if "zbc" in extensions and major == OP and (form.match & FUNCT7_MASK) >> 25 == 0x05:
        return {
            0x1: "carry-less-multiply",
            0x2: "carry-less-multiply-reverse",
            0x3: "carry-less-multiply-high",
        }.get(funct3)
    if "zbkx" in extensions:
        element_width = {0x2: 4, 0x4: 8}.get(funct3)
        return None if element_width is None else f"crossbar-permute:{element_width}"
    # ponytail: unwitnessed scalar extensions stay encoding-only.
    if "zbkb" in extensions:
        if source_count >= 2:
            variant = "bytes" if funct3 == 0x7 else ("words" if major == OP_32 else "halves")
            return f"pack:{variant}"
        variant = {
            (0x687, 0x5): "reverse-bits-in-bytes",
            (0x698, 0x5): "reverse-four-bytes",
            (0x08F, 0x1): "interleave-halves",
            (0x08F, 0x5): "deinterleave-halves",
        }.get((fixed_function, funct3))
        if variant is not None:
            return f"byte-permute:{variant}"
        # Some catalog rows carry both Zbb and Zbkb provenance.  An unmatched
        # unary Zbkb shape must continue to the shared Zbb classifier below;
        # returning here would incorrectly demote e.g. ``rev8`` to
        # encoding-only despite its ratified Zbb value effect.
    if "zbb" in extensions and source_count == 1:
        if fixed_function == 0x287:
            return "byte-or-combine"
        variant = {
            0x600: "leading-zeros",
            0x601: "trailing-zeros",
            0x602: "population",
        }.get(fixed_function)
        if variant is not None:
            return f"bit-count:{variant}"
        return "byte-permute:reverse-bytes" if fixed_function == 0x6B8 else None
    return None


def _compressed_writeback_schema(
    form: OfficialForm,
    fields: frozenset[str],
    funct3: int,
    quadrant: int,
) -> EffectSchema | None:
    if "c_nzuimm10" in fields or "c_nzimm10" in fields:
        return EffectSchema(kind="integer-writeback", operation="add")
    if quadrant == 1 and funct3 == 0x0:
        operation = "add" if "rd_rs1" in "".join(fields) else "nop"
        return EffectSchema(
            kind="integer-writeback",
            operation=operation,
            immediate_field="c_imm6",
        )
    if quadrant == 1 and funct3 == 0x1:
        return EffectSchema(
            kind="integer-writeback",
            width=32,
            operation="add",
            immediate_field="c_imm6",
        )
    if quadrant == 1 and funct3 == 0x2:
        return EffectSchema(
            kind="integer-writeback",
            operation="move",
            immediate_field="c_imm6",
        )
    if quadrant == 1 and funct3 == 0x3 and "c_nzimm18" in fields:
        return EffectSchema(
            kind="integer-writeback",
            operation="lui",
            immediate_field="c_nzimm18",
        )
    if quadrant == 1 and funct3 == 0x4 and "c_imm6" in fields and (form.match >> 10) & 0x3 == 0x2:
        return EffectSchema(
            kind="integer-writeback",
            operation="and",
            immediate_field="c_imm6",
        )
    if quadrant == 2 and funct3 == 0x4 and "c_rs2_n0" in fields:
        return EffectSchema(
            kind="integer-writeback",
            operation="add" if any("rs1" in name for name in fields) else "move",
        )
    if quadrant == 1 and funct3 == 0x4 and "rs2_p" in fields:
        bits6_5 = (form.match >> 5) & 0x3
        bit12 = (form.match >> 12) & 0x1
        if bit12 == 0:
            operation = {0x0: "sub", 0x1: "xor", 0x2: "or", 0x3: "and"}.get(bits6_5)
            if operation is not None:
                return EffectSchema(
                    kind="integer-writeback",
                    operation=operation,
                )
        if bit12 == 1:
            if bits6_5 == 0x0:
                return EffectSchema(
                    kind="integer-writeback",
                    width=32,
                    operation="sub",
                )
            if bits6_5 == 0x1:
                return EffectSchema(
                    kind="integer-writeback",
                    width=32,
                    operation="add",
                )
            if bits6_5 == 0x2:
                return EffectSchema(
                    kind="integer-multiply",
                    operation="mul",
                )
    if quadrant == 1 and funct3 == 0x4 and "rd_rs1_p" in fields:
        subfunction = (form.match >> 2) & 0x1F
        operation, source_width, source_signed = {
            0x18: ("zero-extend", 8, False),
            0x19: ("sign-extend", 8, True),
            0x1A: ("zero-extend", 16, False),
            0x1B: ("sign-extend", 16, True),
            0x1C: ("zero-extend", 32, False),
            0x1D: ("not", 0, False),
        }.get(subfunction, ("generic", 0, True))
        return EffectSchema(
            kind="integer-writeback",
            operation=operation,
            source_width=source_width,
            source_signed=source_signed,
        )
    return None


_FP_PRECISION_BY_FORMAT = {0: 24, 1: 53, 2: 11, 3: 113}
_FP_MEMORY_FORMAT = {1: (16, 11), 2: (32, 24), 3: (64, 53), 4: (128, 113)}
# Integer load/store width and signedness follow funct3.
_ACCESS_WIDTH = {0: 8, 1: 16, 2: 32, 3: 64}
_ATOMIC_ACCESS_WIDTH = {**_ACCESS_WIDTH, 4: 128}

# Effect kinds that read or write the seeded memory line.  An effect-kind fact,
# shared by intent construction and the assembler.
MEMORY_EFFECT_KINDS = frozenset(
    {"memory-load", "memory-store", "atomic-rmw", "vector-memory", "cache-block"}
)


def register_fields(form: OfficialForm) -> tuple[str, ...]:
    return tuple(
        name
        for name in operand_groups(form)
        if is_register_operand_group(form, name)
    )


def producer_chain_role(
    form: OfficialForm,
    schema: EffectSchema | None,
    available_roles: tuple[str, ...],
) -> str | None:
    """Return the explicit source role a pre-risk producer may feed."""
    if schema is None or schema.kind in {
        "memory-load",
        "memory-store",
        "atomic-rmw",
        "stack-transfer",
        "paired-register-transfer",
        "instruction-memory",
        "fence",
        "may-be-operation",
    }:
        return None
    if form.xlen == "rv32":
        return None
    registers = register_fields(form)
    for role in available_roles:
        source_slot = "rs" + role[2:] if role.startswith("fs") else role
        if role.startswith(("rs", "fs")) and (
            any(source_slot in name for name in registers)
            or source_slot == "rs2" and "c_rs2" in registers
        ):
            return role
    return None


# Compressed memory forms carry their access width in the immediate field name,
# so width follows from structure rather than from the instruction.
_COMPRESSED_ACCESS_WIDTH = {
    "c_uimm1": 16,
    "c_uimm2": 8,
    "c_uimm7": 32,
    "c_uimm8sp": 32,
    "c_uimm8sp_s": 32,
    "c_uimm8": 64,
    "c_uimm9sp": 64,
    "c_uimm9sp_s": 64,
}
_COMPRESSED_IMMEDIATE_WRITEBACK = ("c_imm6", "c_nzimm18", "c_nzimm10", "c_nzuimm10")
_COMPRESSED_DESTINATION_FIELDS = ("rd", "rd_p", "rd_n0", "rd_n2", "rd_rs1_p", "rd_rs1_n0")


def _compressed_schema(form: OfficialForm, fields: frozenset[str]) -> EffectSchema | None:
    """Classify a 16-bit form by operand shape alone."""
    if {"c_rlist", "c_spimm"} <= fields:
        # The push/pop/popret/popretz split is the pinned 16-bit encoding:
        # bits 9-8 of the fixed funct field select 0/2/4/6, so the operation
        # comes from the codeword, not from a mnemonic table.
        operation = {
            0x0: "push",
            0x2: "pop",
            0x4: "popretz",
            0x6: "popret",
        }.get((form.match >> 8) & 0x7)
        if operation is not None:
            return EffectSchema(
                kind="stack-transfer",
                width=64,
                signed=False,
                memory_width=64,
                writeback="stack",
                observer="memory",
                operation=operation,
            )
        return None
    if fields == {"c_sreg1", "c_sreg2"}:
        # The two paired-register transfers are the c_sreg1/c_sreg2 rows of
        # the catalog; their direction is the fixed bit 6 of the codeword.
        return EffectSchema(
            kind="paired-register-transfer",
            width=64,
            signed=False,
            writeback="paired-gpr",
            observer="memory",
            operation=(
                "move-s-to-a01" if form.match & 0x40 else "move-a01-to-s"
            ),
        )
    if fields == {"c_index"}:
        return EffectSchema(
            kind="instruction-memory",
            width=64,
            signed=False,
            writeback="gpr",
            observer="memory",
            operation="table-jump",
            immediate_field="c_index",
        )
    access = next((name for name in fields if name in _COMPRESSED_ACCESS_WIDTH), None)
    if access is not None:
        width = _COMPRESSED_ACCESS_WIDTH[access]
        unsigned = form.mnemonic in {"c_lbu", "c_lhu"}
        writes = any(name in fields for name in _COMPRESSED_DESTINATION_FIELDS)
        if writes:
            # c.fld/c.fldsp land in the floating point file; the register
            # domain is the one owner of that fact, here as everywhere.
            destination_domain = register_domain(form, "rd")
            destination_is_fp = destination_domain == "fpr"
            return EffectSchema(
                kind="memory-load",
                width=width,
                signed=not destination_is_fp and not unsigned,
                precision={32: 24, 64: 53}.get(width, 0) if destination_is_fp else 0,
                memory_width=width,
                writeback=destination_domain,
                observer="rawbits" if destination_is_fp else "gpr",
                operation="load-unsigned" if unsigned else "load",
                immediate_field=access,
                state_domain="memory",
            )
        source_domain = register_domain(form, "rs2")
        return EffectSchema(
            kind="memory-store",
            width=width,
            precision={32: 24, 64: 53}.get(width, 0) if source_domain == "fpr" else 0,
            memory_width=width,
            writeback="memory",
            observer="memory",
            operation="store",
            immediate_field=access,
            state_domain="memory",
        )
    if "c_bimm9" in fields:
        operation = "bne" if ((form.match >> 13) & 0x7) == 0x7 else "beq"
        return EffectSchema(
            kind="control-compare",
            operation=operation,
            signed=False,
            writeback="control",
            observer="control-path",
            immediate_field="c_bimm9",
            state_domain="control-flow",
        )
    if "c_imm12" in fields:
        is_link = ((form.match >> 13) & 0x7) == 0x1
        return EffectSchema(
            kind="control-jump",
            writeback="gpr" if is_link else "control",
            observer="control-path",
            immediate_field="c_imm12",
            operation="jal" if is_link else "j",
            state_domain="control-flow",
        )
    funct3 = (form.match >> 13) & 0x7
    quadrant = form.match & 0x3
    # c.jr and c.jalr share quadrant-2/funct3=4 with c.mv/c.add.  They are the
    # forms where rs2 is fixed to zero, hence must be recognised before the
    # generic compressed writeback fallback below.
    has_rs1 = any("rs1" in name for name in fields)
    has_rs2 = any("rs2" in name for name in fields)
    if quadrant == 2 and funct3 == 0x4 and has_rs1 and not has_rs2:
        is_jalr = bool((form.match >> 12) & 0x1)
        return EffectSchema(
            kind="control-jump",
            writeback="gpr" if is_jalr else "control",
            observer="control-path",
            operation="jalr" if is_jalr else "jr",
            state_domain="control-flow",
        )
    # Quadrant 1 funct3=4 excludes value 2, whose c_imm6 form is a mask rather
    # than a shift amount.
    if (
        quadrant == 2 and funct3 == 0x0
        or quadrant == 1
        and funct3 == 0x4
        and "c_imm6" in fields
        and (form.match >> 10) & 0x3 != 0x2
    ):
        return EffectSchema(
            kind="integer-shift",
            immediate_field="c_imm6",
            operation=(
                "shift-left"
                if quadrant == 2 else {
                    0x0: "shift-right-logical",
                    0x1: "shift-right-arithmetic",
                }.get((form.match >> 10) & 0x3, "generic")
            ),
        )
    writeback = _compressed_writeback_schema(form, fields, funct3, quadrant)
    if writeback is not None:
        return writeback
    immediate = next((name for name in _COMPRESSED_IMMEDIATE_WRITEBACK if name in fields), None)
    if immediate is not None:
        return EffectSchema(
            kind="integer-writeback",
            immediate_field=immediate,
        )
    if any(name in fields for name in _COMPRESSED_DESTINATION_FIELDS):
        return EffectSchema(kind="integer-writeback")
    return None


def _classify_form(form: OfficialForm) -> EffectSchema | None:
    """Classify a form structurally, or return None for legality-only forms."""
    if trap_outcome_kind_for_form(form) is not None:
        return None
    major = _major(form)
    funct3 = (
        (form.match & FUNCT3_MASK) >> 12
        if form.encoding_length_bytes == 4 and form.mask & FUNCT3_MASK
        else None
    )
    if major == MISC_MEM and funct3 == 0x2:
        return EffectSchema(
            kind="cache-block",
            writeback="none",
            observer="outcome",
            operation="cache-block",
            state_domain="memory",
        )
    fields = frozenset(operand_groups(form))
    extensions = _extension_tokens(form)

    if form.extension_group in {"xtheadbb", "xtheadbs"}:
        return EffectSchema(
            kind="integer-shift",
            width=32,
            immediate_field="p_w_uimm6",
            operation="rotate-right" if form.extension_group == "xtheadbb" else "bit-extract",
        )

    # The three vector-configuration forms are the OP-V funct3=0x7 rows; the
    # immediate-vs-register split follows the operand fields of the pinned
    # catalog (zimm11/zimm5 vs rs2), not a mnemonic table.
    vector_config = None
    if major == 0x57 and funct3 == 0x7:
        if "zimm11" in fields:
            vector_config = ("zimm11", "vector-config-immediate")
        elif "zimm5" in fields:
            vector_config = ("zimm5", "vector-config-immediate-avl")
        elif "rs2" in fields:
            vector_config = (None, "vector-config-register")
    if vector_config is not None:
        immediate_field, operation = vector_config
        return EffectSchema(
            kind="may-be-operation",
            writeback="gpr",
            observer="gpr",
            immediate_field=immediate_field,
            operation=operation,
            state_domain="vector",
        )

    reservation_wait = (
        {
            0x00D: "reservation-wait-nto",
            0x01D: "reservation-wait-sto",
        }.get((form.match >> 20) & 0xFFF)
        if form.encoding_length_bytes == 4
        and (form.match & MAJOR_OPCODE_MASK) == SYSTEM
        and (form.mask & 0xFFF00000) == 0xFFF00000
        else None
    )
    if reservation_wait is not None:
        return EffectSchema(
            kind="may-be-operation",
            writeback="none",
            observer="outcome",
            operation=reservation_wait,
            state_domain="reservation",
        )

    # Fixed/structured SYSTEM forms are admitted as a privileged-state slice,
    # not as ordinary user-mode scalar effects.  The descriptor still drives
    # every field boundary; privilege transition/trap results remain an
    # outcome-only contract until a privileged harness is attached.
    if (
        major == SYSTEM
        and ("system" in extensions or "s" in extensions or extensions & _PRIVILEGED_STATE_EXTENSIONS)
        and "csr" not in fields
    ) or "zicfiss" in extensions:
        return EffectSchema(
            kind="privileged-system",
            writeback="none",
            observer="outcome",
            operation="hidden-state-transition"
            if "zicfiss" in extensions and major != SYSTEM
            else "privileged-transition",
            state_domain="privileged",
        )

    # Vector data forms share one descriptor-driven state path.  Ratified Vector
    # Crypto uses OP-VE (0x77), while ordinary V/Zv bitmanip uses OP-V (0x57);
    # both carry the same vector register/CSR observer obligations.  Keep the
    # opcode distinction here instead of adding one generator per crypto family.
    if major == 0x57 or (
        major == 0x77 and any(token.startswith("zv") for token in extensions)
    ):
        mnemonic = assembly_mnemonic(form)
        scalar_destination = (
            "rd" in fields
            and "vd" not in fields
            and register_domain(form, "rd") in {"gpr", "fpr"}
        )
        scalar_writeback = (
            register_domain(form, "rd") if scalar_destination else "vector"
        )
        vector_funct6 = (int(form.match) >> 26) & 0x3F
        if "rd" in fields and "vd" not in fields:
            vector_shape = "vector-to-scalar"
        elif (
            "vs1" in fields
            and vector_funct6 in _VECTOR_MASK_OR_CARRY_FUNCT6
            and not mnemonic.startswith(("vabd.", "vabdu."))
            and not (vector_funct6 == 0x17 and "vs2" not in fields)
        ):
            vector_shape = "mask-or-carry"
        elif "simm5" in fields or "zimm5" in fields or "zimm6" in fields:
            vector_shape = "vector-immediate"
        elif "rs1" in fields:
            vector_shape = "vector-scalar"
        elif "vs1" in fields:
            vector_shape = "vector-vector"
        else:
            vector_shape = "vector-unary"
        return EffectSchema(
            kind="vector-data",
            writeback=scalar_writeback,
            observer=(
                "rawbits"
                if scalar_writeback == "fpr"
                else "gpr"
                if scalar_writeback == "gpr"
                else "vector-state"
            ),
            operation=f"{vector_shape}:funct6-{vector_funct6:02x}",
            state_domain="vector",
        )

    if major in {0x07, 0x27} and "v" in extensions:
        width = {0: 8, 5: 16, 6: 32, 7: 64}.get(funct3 or 0, 8)
        return EffectSchema(
            kind="vector-memory",
            width=width,
            memory_width=width,
            writeback="vector" if major == 0x07 else "memory",
            observer="vector-state" if major == 0x07 else "memory",
            operation="vector-load" if major == 0x07 else "vector-store",
            state_domain="vector-memory",
        )

    if "zimop" in extensions:
        return EffectSchema(
            kind="may-be-operation",
            writeback="gpr",
            observer="gpr",
            operation="write-zero",
        )
    if "zcmop" in extensions:
        return EffectSchema(
            kind="may-be-operation",
            writeback="none",
            observer="outcome",
            operation="no-write",
        )

    if form.encoding_length_bytes == 2:
        return _compressed_schema(form, fields)

    if major in (MADD, MSUB, NMSUB, NMADD):
        precision = _FP_PRECISION_BY_FORMAT.get(_fp_format(form) or 0, 24)
        return EffectSchema(
            kind="fp-fused",
            precision=precision,
            writeback="fpr",
            observer="rawbits-plus-fflags",
            operation={MADD: "madd", MSUB: "msub", NMSUB: "nmsub", NMADD: "nmadd"}[major],
        )

    if major == OP_FP:
        precision = _FP_PRECISION_BY_FORMAT.get(_fp_format(form) or 0, 24)
        function = (form.match >> 27) & 0x1F
        source_code = (form.match >> 20) & 0x1F
        bf16_convert = function == 0x08 and "zfbfmin" in extensions and source_code in {6, 8}
        if bf16_convert and source_code == 8:
            precision = 8
        # 写回落在哪个寄存器堆由 register_domain() 统一判定，不在这里再推一遍 funct5。
        writes_gpr = "rd" in fields and register_domain(form, "rd") == "gpr"
        if function == 0x1C and funct3 == 0x1:
            kind = "fp-classify"
            operation = "classify"
        elif function == 0x0B:
            kind = "fp-arith"
            operation = "sqrt"
        elif function in {0x08, 0x18, 0x1A}:
            kind = "fp-convert"
            operation = {0x08: "fp-to-fp", 0x18: "fp-to-int", 0x1A: "int-to-fp"}[function]
        elif function == 0x14:
            # funct5=10100 owns both the base compares and the quiet Zfa
            # variants.  Their input/output relation differs by funct3, so one
            # broad "compare" owner would collapse distinct IEEE boundaries.
            kind = "fp-compare"
            operation = _FP_COMPARE_OPERATION_BY_FUNCT3.get(funct3 or 0x0, "compare")
        elif function in {0x1C, 0x1E}:
            kind = "fp-convert"
            operation = "literal" if operand_kind(form, "rs1") == "imm" else "move"
        elif function == 0x16:
            kind = "fp-convert"
            operation = "move-pair"
        else:
            kind = "fp-arith"
            operation = {
                0x00: "add",
                0x01: "sub",
                0x02: "mul",
                0x03: "div",
                0x04: _FP_SIGN_OPERATION_BY_FUNCT3.get(funct3 or 0x0, "sign-inject"),
                0x05: _FP_MINMAX_OPERATION_BY_FUNCT3.get(funct3 or 0x0, "minmax"),
            }.get(function, "generic")
        if function == 0x08 and "zfa" in extensions and source_code == 5:
            kind, operation = "fp-convert", "round-to-integral-nx"
        elif function == 0x08 and "zfa" in extensions and source_code == 4:
            kind, operation = "fp-convert", "round-to-integral"
        elif function == 0x18 and "zfa" in extensions and source_code == 8 and funct3 == 0x1:
            kind, operation = "fp-convert", "fp-to-int-mod"
        writeback = "gpr" if writes_gpr else "fpr"
        observes_flags = kind not in {"fp-classify"} and operation not in _FLAGLESS_FP_OPERATIONS
        observer = (
            ("gpr-plus-fflags" if writes_gpr else "rawbits-plus-fflags")
            if observes_flags
            else ("gpr" if writes_gpr else "rawbits")
        )
        conversion = {}
        if operation in {"int-to-fp", "fp-to-int", "fp-to-int-mod"}:
            # FCVT.{S,D,H,Q}.{W,WU,L,LU} and its reverse encode the integer
            # domain in fixed rs2: 0=W, 1=WU, 2=L, 3=LU.
            integer_kind = (form.match >> 20) & 0x3
            integer_width = 32 if integer_kind < 2 else 64
            integer_signed = integer_kind in {0, 2}
            conversion = {
                "width": integer_width,
                "signed": integer_signed,
                "source_width": integer_width if operation == "int-to-fp" else 0,
                "source_signed": integer_signed,
                "source_precision": precision if operation in {"fp-to-int", "fp-to-int-mod"} else 0,
            }
        elif operation == "fp-to-fp":
            # The destination format is fmt (bits 26:25); fixed rs2 names the
            # source format.  This is an ISA encoding fact, not a name map.
            conversion = {
                "source_precision": (
                    24
                    if bf16_convert and source_code == 8
                    else 8
                    if bf16_convert and source_code == 6
                    else _FP_PRECISION_BY_FORMAT.get((form.match >> 20) & 0x3, precision)
                )
            }
        elif operation in {"round-to-integral", "round-to-integral-nx"}:
            conversion = {"source_precision": precision}
        elif operation == "move":
            source_is_gpr = register_domain(form, "rs1") == "gpr"
            conversion = {
                "source_width": ({11: 16, 24: 32, 53: 64}.get(precision, 0) if source_is_gpr else 0),
                "source_precision": 0 if source_is_gpr else precision,
            }
        elif operation == "literal":
            conversion = {
                "immediate_field": "rs1"
                if operand_kind(form, "rs1") == "imm"
                else None
            }
        elif operation == "move-pair":
            conversion = {
                "source_width": 32 if form.xlen == "rv32" else 64,
                "source_signed": False,
            }
        return EffectSchema(
            kind=kind,
            precision=precision,
            writeback=writeback,
            observer=observer,
            operation=operation,
            **conversion,
        )

    if major == AMO:
        width = _ATOMIC_ACCESS_WIDTH.get(2 if funct3 is None else funct3, 32)
        function = (form.match >> 27) & 0x1F
        if width > 64:
            if function != 0x05:
                return None
            return EffectSchema(
                kind="atomic-rmw",
                width=width,
                memory_width=width,
                writeback="paired-gpr",
                operation="cas",
                state_domain="memory",
            )
        if function == 0x06:
            return EffectSchema(
                kind="atomic-rmw",
                width=width,
                memory_width=width,
                operation="load-acquire",
                state_domain="memory",
            )
        if function == 0x07:
            return EffectSchema(
                kind="atomic-rmw",
                width=width,
                memory_width=width,
                writeback="memory",
                observer="memory",
                operation="store-release",
                state_domain="memory",
            )
        operation = {
            0x00: "add",
            0x01: "swap",
            0x02: "lr",
            0x03: "sc",
            0x04: "xor",
            0x05: "cas",
            0x08: "or",
            0x0C: "and",
            0x10: "min",
            0x14: "max",
            0x18: "minu",
            0x1C: "maxu",
        }.get(function, "generic")
        return EffectSchema(
            kind="atomic-rmw",
            width=width,
            memory_width=width,
            operation=operation,
            state_domain="reservation" if operation in {"lr", "sc"} else "memory",
        )

    if major == LOAD:
        code = funct3 or 0
        width = _ACCESS_WIDTH.get(code & 0x3, 64)
        return EffectSchema(
            kind="memory-load",
            width=width,
            signed=not (code & 0x4),
            memory_width=width,
            writeback="gpr",
            operation="load-signed" if not (code & 0x4) else "load-unsigned",
            immediate_field="imm12",
            state_domain="memory",
        )

    if major in {0x07, 0x27}:  # FP load/store share the LOAD-FP major opcodes
        fp_memory = _FP_MEMORY_FORMAT.get(funct3 or -1)
        if fp_memory is None:
            return None
        width, precision = fp_memory
        is_store = major == 0x27
        return EffectSchema(
            kind="memory-store" if is_store else "memory-load",
            width=width,
            signed=is_store,
            precision=precision,
            memory_width=width,
            writeback="memory" if is_store else "fpr",
            observer="memory" if is_store else "rawbits",
            operation="store" if is_store else "load",
            immediate_field="imm12s" if is_store else "imm12",
            state_domain="memory",
        )

    if major == STORE:
        code = 3 if funct3 is None else funct3
        width = _ACCESS_WIDTH.get(code & 0x3, 64)
        return EffectSchema(
            kind="memory-store",
            width=width,
            memory_width=width,
            writeback="memory",
            observer="memory",
            operation="store",
            immediate_field="imm12s",
            state_domain="memory",
        )

    if major == BRANCH:
        operation = _BRANCH_OPERATION_BY_FUNCT3.get(funct3 or 0x0, "branch")
        return EffectSchema(
            kind="control-compare",
            operation=operation,
            signed=operation in {"blt", "bge"},
            writeback="control",
            observer="control-path",
            immediate_field="bimm12",
            state_domain="control-flow",
        )

    if major in (JAL, JALR):
        return EffectSchema(
            kind="control-jump",
            writeback="gpr",
            observer="control-path",
            operation="jalr" if major == JALR else "jal",
            immediate_field="jimm20" if major == JAL else "imm12",
            state_domain="control-flow",
        )

    if major == SYSTEM and funct3 not in (None, 0x0, 0x4):
        return EffectSchema(
            kind="csr-access",
            writeback="gpr",
            observer="gpr",
            operation=_CSR_OPERATION_BY_FUNCT3.get(funct3 or 0x1, "generic"),
            state_domain="csr",
        )

    if major in (LUI, AUIPC):
        return EffectSchema(
            kind="integer-writeback",
            immediate_field="imm20",
            operation="lui" if major == LUI else "auipc",
        )

    if major in (OP, OP_32):
        width = 64 if major == OP else 32
        funct7 = (form.match & FUNCT7_MASK) >> 25
        if funct7 == 0x01:
            operation = {
                0x0: "mul",
                0x1: "mulh",
                0x2: "mulhsu",
                0x3: "mulhu",
                0x4: "div",
                0x5: "divu",
                0x6: "rem",
                0x7: "remu",
            }.get(funct3, "generic")
            return EffectSchema(
                kind="integer-multiply",
                width=width,
                signed=operation not in {"mulhu", "divu", "remu"},
                operation=operation,
            )
        shift_operation = _register_shift_operation(funct7, funct3)
        if shift_operation is not None and "rs2" in fields:
            return EffectSchema(
                kind="integer-shift",
                width=width,
                operation=shift_operation,
            )
        operation = {
            (0x00, 0x0): "add",
            (0x20, 0x0): "sub",
            (0x00, 0x2): "set-less-than",
            (0x00, 0x3): "set-less-than-unsigned",
            (0x00, 0x4): "xor",
            (0x20, 0x4): "xnor",
            (0x00, 0x6): "or",
            (0x20, 0x6): "orn",
            (0x00, 0x7): "and",
            (0x20, 0x7): "andn",
            (0x05, 0x4): "min",
            (0x05, 0x5): "minu",
            (0x05, 0x6): "max",
            (0x05, 0x7): "maxu",
        }.get((funct7, funct3), "generic")
        if operation == "generic":
            crypto_width = _scalar_crypto_width(
                form,
                major=major,
                funct3=funct3,
                fixed_function=(form.match >> 20) & 0xFFF,
                funct7=(form.match >> 25) & 0x7F,
                fields=fields,
                extensions=extensions,
            )
            rare_operation = _rare_integer_operation(
                form, major=major, funct3=funct3, fields=fields,
            )
            operation = "scalar-crypto" if crypto_width is not None else rare_operation or operation
        else:
            crypto_width = None
        # ZEXT.H is the unary OP-32 encoding with funct3=100, a fully fixed
        # rs2/funct7 field, and no rs2 operand.  Derive it from those frozen
        # encoding facts so aliases and catalog rebuilds use the same owner.
        zero_extend_halfword = (
            major == OP_32
            and funct3 == 0x4
            and "rs2" not in fields
            and (form.mask & 0xFFF00000) == 0xFFF00000
            and ((form.match >> 20) & 0xFFF) == 0x080
        )
        if zero_extend_halfword:
            operation = "zero-extend"
        unsigned_word_operation = (
            operation == "add-unsigned-word"
            or operation.startswith("shift-add-unsigned-word:")
        )
        return EffectSchema(
            kind="integer-writeback",
            width=(
                64 if zero_extend_halfword or unsigned_word_operation
                else crypto_width if crypto_width is not None
                else width
            ),
            signed=(
                False
                if zero_extend_halfword or unsigned_word_operation
                else operation not in {"set-less-than-unsigned", "minu", "maxu"}
            ),
            source_width=(
                16 if zero_extend_halfword
                else 32 if unsigned_word_operation
                else crypto_width if crypto_width is not None
                else 0
            ),
            source_signed=not (zero_extend_halfword or unsigned_word_operation),
            operation=operation,
        )

    if major in (OP_IMM, OP_IMM_32):
        width = 64 if major == OP_IMM else 32
        # The shift funct3 values are also the doorway to the Zbb/Zbkb/Zknh
        # unary operations, which spend the shift-amount bits on a function
        # code.  Only a form that still exposes those bits as an operand is a
        # shift; the rest are ordinary register-immediate writebacks.
        shift_field = next((f for f in fields if f.startswith("shamt")), None)
        if shift_field is not None:
            funct7 = (form.match & FUNCT7_MASK) >> 25
            operation = _register_shift_operation(funct7, funct3) or (
                "shift-left-unsigned-word" if (funct7, funct3) == (0x04, 0x1) else "generic"
            )
            return EffectSchema(
                kind="integer-shift",
                width=64 if operation == "shift-left-unsigned-word" else width,
                source_width=32 if operation == "shift-left-unsigned-word" else 0,
                source_signed=operation != "shift-left-unsigned-word",
                immediate_field=shift_field or "shamtd",
                operation=operation,
            )
        fixed_function = (form.match >> 20) & 0xFFF
        projection = {
            0x604: ("sign-extend", 8, True),
            0x605: ("sign-extend", 16, True),
        }.get(fixed_function)
        operation = {
            0x0: "add",
            0x2: "set-less-than",
            0x3: "set-less-than-unsigned",
            0x4: "xor",
            0x6: "or",
            0x7: "and",
        }.get(funct3, "generic")
        if operation == "generic":
            crypto_width = _scalar_crypto_width(
                form,
                major=major,
                funct3=funct3,
                fixed_function=fixed_function,
                funct7=(form.match >> 25) & 0x7F,
                fields=fields,
                extensions=extensions,
            )
            rare_operation = _rare_integer_operation(
                form, major=major, funct3=funct3, fields=fields,
            )
            operation = "scalar-crypto" if crypto_width is not None else rare_operation or operation
        else:
            crypto_width = None
        if projection is not None:
            operation, source_width, source_signed = projection
        else:
            source_width, source_signed = 0, operation != "set-less-than-unsigned"
        return EffectSchema(
            kind="integer-writeback",
            width=crypto_width if crypto_width is not None else width,
            source_width=crypto_width if crypto_width is not None else source_width,
            source_signed=source_signed,
            immediate_field="imm12",
            signed=operation != "set-less-than-unsigned",
            operation=operation,
        )

    if major == MISC_MEM:
        # FENCE.I is the MISC-MEM funct3=001 form; use the encoding facts so
        # the instruction-memory owner is not tied to one catalog spelling.
        if funct3 == 0x1:
            return EffectSchema(
                kind="instruction-memory",
                writeback="none",
                observer="memory",
                operation="fence-fetch",
            )
        return EffectSchema(
            kind="fence",
            writeback="none",
            observer="outcome",
            operation="fence",
            state_domain="memory",
        )

    # A catalog form without a currently modelled scalar effect still owns a
    # real, useful obligation: its exact encoding, extension gate and profile
    # disposition.  Keep this descriptor-driven fallback separate from the
    # explicit trap/cache forms above so unsupported execution semantics are
    # visible as encoding-only rather than silently disappearing.
    return EffectSchema(
        kind="encoding-only",
        writeback="none",
        observer="outcome",
        operation="encoding-neighborhood",
    )

@cache
def generation_rule_for_form(form: OfficialForm) -> GenerationRule:
    """Return the sole generic generation binding for *form*."""
    # A small set of catalog mnemonics are duplicated by an unratified P row
    # and a ratified Z* semantic alias with identical encoding fields.  The
    # alias is the canonical value owner; otherwise materialization resolves the
    # mnemonic back to the selected row and admission sees two schemas for the
    # same bytes.  Unique P rows remain in the full catalog as encoding-only
    # forms, so this does not add an extension-specific generator.
    if not isinstance(form, OfficialForm):
        raise TypeError("catalog form must be an OfficialForm")
    candidate = OFFICIAL_ALL_CATALOG_BY_MNEMONIC.get(form.mnemonic)
    selected = OFFICIAL_SELECTED_SCALAR_BY_MNEMONIC.get(form.mnemonic, candidate)
    fields = (
        "mnemonic", "source_extensions", "extension_group",
        "encoding_length_bytes", "mask", "match", "variable_fields", "encoding",
    )
    if candidate is None or not any(
        all(getattr(owner, field) == getattr(form, field) for field in fields)
        for owner in (candidate, selected)
        if owner is not None
    ):
        raise ValueError(f"unknown catalog form: {form.mnemonic}")
    form = selected
    schema = _classify_form(form)
    if schema is not None and schema.immediate_field is not None \
            and schema.immediate_field not in operand_groups(form):
        schema = replace(schema, immediate_field=None)
    # 白名单降级已删除：catalog 里每条 form 都有自己的 encoding 与语义 schema，
    # 能不能被某个模拟器执行是目标侧事实，不再由生成侧的白名单决定。
    if schema is not None and (
        schema.operation in {"", "generic"}
        or (
            schema.kind == "control-compare"
            and form.encoding_length_bytes == 4
            and "rs2" not in operand_groups(form)
        )
    ):
        # The pinned opcode catalog is larger than the currently executable
        # semantic slice.  Keep such forms in the generation denominator as a
        # descriptor-driven encoding lane instead of silently dropping them or
        # inventing a mnemonic-specific value formula.  The admission and
        # reference layers continue to treat this lane as outcome/encoding
        # coverage only.
        schema = replace(
            schema,
            kind="encoding-only",
            width=0,
            signed=False,
            precision=0,
            source_width=0,
            source_precision=0,
            writeback="none",
            observer="outcome",
            immediate_field=None,
            memory_width=0,
            operation="encoding-neighborhood",
            state_domain="scalar",
        )
    realizations = semantic_realizations_for_form(
        form,
        writeback=None if schema is None else schema.writeback,
    )
    return GenerationRule(
        form=form,
        schema=schema,
        realizations=realizations,
    )


def generation_schema_for_form(form: OfficialForm) -> EffectSchema | None:
    """Return the schema owned by generation, including encoding-only fallback."""
    return generation_rule_for_form(form).schema


def generation_owner_contract(
    form: OfficialForm, *, table_index: int | None = None,
) -> dict[str, object]:
    """Return the exact effect owner persisted with one generated testcase."""
    schema = generation_rule_for_form(form).schema
    if (
        schema is not None
        and form.mnemonic == "cm_jalt"
        and table_index is not None
        and table_index < 32
    ):
        schema = replace(schema, writeback="none")
    effect_kind = "legality-only" if schema is None else str(schema.kind)
    return {
        "contract": "rvgen-generation-owner-v1",
        "effect_kind": effect_kind,
        "operation": None if schema is None else str(schema.operation),
        "state_domain": None if schema is None else str(schema.state_domain),
        "schema": (
            None
            if schema is None
            else {"writeback": str(schema.writeback), "observer": str(schema.observer)}
        ),
    }

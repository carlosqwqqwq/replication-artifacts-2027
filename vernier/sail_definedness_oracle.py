"""Sail/Manual definedness second reference oracle (C2e).

This module statically adjudicates encoding legality and expected outcome for
profile-legality rows.  It deliberately does not launch the Sail simulator:
the pinned Sail profile/config contract decides whether a profile is
adjudicable at all (``sail-only`` bookkeeping), while the encoding facts come
from the frozen catalog and the RISC-V ISA Manual (HINT/reserved/illegal
spellings).  Native reference is never implied, so ``sail-only`` rows must not
enter native coverage.

The oracle is issue-agnostic: every decision derives from encoding facts and
the requested ISA profile, never from a defect ID or tool name.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from .reference_contract import sail_profile_support
from .riscv_encoding import (
    decode_form,
    decoded_register_slots,
    register_domain,
    form_for_mnemonic,
    logical_immediate_for_form_field,
)
from .riscv_csrs import (
    csr_is_read_only,
    csr_is_rv32_only,
    csr_minimum_privilege,
)
from .riscv_csrs import csr_name
from .rvgen.boundaries import compressed_codeword_disposition
from .rvgen.effects import generation_schema_for_form
from .spec_definedness import (
    _profile_xlen_matches_form,
    csr_required_extensions_for_profile,
    enabled_extensions,
    is_canonical_isa_profile,
    profile_enables_form,
    profile_uses_compressed_encoding,
    required_extension_sets_for_form,
    x_register_fp_extensions,
)


@dataclass(frozen=True)
class SailDefinednessDecision:
    """One static legality/definedness ruling for a concrete encoding."""

    disposition: str
    expected_outcome: str
    decision_source: str
    reason: str
    expected_mcause: int | None = None
    sail_supported: bool = False
    sail_reason: str = ""
    config_identity: str = ""


_RESERVED_MCAUSE = 2  # illegal instruction
_PRIVILEGED_MCAUSE = 3  # breakpoint (ebreak)
_ENVIRONMENT_MCAUSE = 8  # environment call from user mode (ecall)


@cache
def _profile_form_support(isa_profile: str, form) -> bool:
    """Canonical-form profile gate without rejecting non-canonical profiles.

    ``profile_enables_form`` refuses non-canonical profile spellings (for
    example ``rv64imafdc_zca``), while this oracle must rule on whatever
    profile string the testcase carries.  The gate is: XLEN matches the
    form, no x-register FP conflict, and at least one required extension
    set is enabled.
    """
    if is_canonical_isa_profile(isa_profile):
        return profile_enables_form(isa_profile, form)
    if x_register_fp_extensions(isa_profile):
        return False
    if not _profile_xlen_matches_form(isa_profile, form):
        return False
    enabled = enabled_extensions(isa_profile)
    return any(
        required <= enabled
        for required in required_extension_sets_for_form(form)
    )


def _structural_legality(
    form,
    word: int,
    isa_profile: str,
    operands: Mapping[str, int],
) -> str | None:
    """Return a structural reserved/illegal reason, or None when legal."""
    mnemonic = str(form.mnemonic).replace(".", "_")
    enabled = enabled_extensions(isa_profile)
    xlen = 32 if str(isa_profile).lower().startswith("rv32") else 64
    csr = operands.get("csr")
    if csr is not None and csr_name(int(csr)) is None:
        return "csr-unimplemented"
    if csr is not None and not csr_required_extensions_for_profile(int(csr), isa_profile) <= enabled:
        return "extension-disabled"
    schema = generation_schema_for_form(form)
    compressed_disposition = _compressed_definedness_disposition(
        form, schema, operands
    )
    # ``*_n0`` / ``*_n2`` variable fields are catalog legality constraints:
    # the zero/two spellings are reserved, not ordinary register values.
    # ``c_lui``'s ``rd_n2`` is handled below by the Manual disposition: rd=0
    # with imm!=0 is a HINT and rd=2 with imm!=0 is non-canonical, neither is
    # a reserved-encoding by itself.
    for field in getattr(form, "variable_fields", ()):
        if field not in operands:
            continue
        value = int(operands[field])
        if mnemonic == "c_lui" and field == "rd_n2":
            continue
        if field.endswith("_n0") and value == 0 and compressed_disposition not in {"hint", "normal"}:
            return "reserved-encoding"
        if field.endswith("_n2") and value in {0, 2}:
            return "reserved-encoding"
    if mnemonic == "c_addi4spn" and int(operands.get("c_nzuimm10", 1)) == 0:
        return "reserved-encoding"
    if mnemonic == "c_addi16sp" and int(operands.get("c_nzimm10", 1)) == 0:
        return "reserved-encoding"
    if mnemonic == "c_lui" and int(operands.get("c_nzimm18", 1)) == 0:
        return "reserved-encoding"
    # Shift legality comes from the effect schema and catalog fields.  This
    # keeps a newly catalogued shift/rotate form on the same path without a
    # mnemonic-specific list.
    if getattr(schema, "kind", None) == "integer-shift":
        shift_keys = []
        for field in getattr(form, "variable_fields", ()):
            logical = logical_immediate_for_form_field(form, field)
            if "shamt" in field or (logical is not None and "imm" in logical):
                shift_keys.append(logical or field)
        encoded_operands = decode_form(form, word)
        amount = None
        for key in shift_keys:
            amount = operands.get(key, encoded_operands.get(key))
            if amount is not None:
                break
        limit = min(xlen, int(getattr(schema, "width", xlen)))
        if amount is not None and int(amount) >= limit:
            return "reserved-shift-amount"
        if getattr(schema, "immediate_field", None) == "shamtw" and word & (1 << 25):
            return "reserved-shift-amount"
    if str(isa_profile).lower().startswith("rv32e") and any(
        value >= 16
        for role, value in decoded_register_slots(form, word).items()
        if register_domain(form, role) == "gpr"
    ):
        return "rve-register-domain"
    if getattr(schema, "kind", None) == "csr-access" and csr is not None:
        csr = int(csr)
        if str(isa_profile).lower().startswith("rv64") and csr_is_rv32_only(csr):
            return "rv32-only-csr"
        funct3 = (word >> 12) & 0x7
        source = int(operands.get("zimm5", operands.get("rs1", 0)))
        writes = funct3 in {0x1, 0x5} or (
            source != 0 and funct3 in {0x2, 0x3, 0x6, 0x7}
        )
        if csr_minimum_privilege(csr) > 0 or (writes and csr_is_read_only(csr)):
            return "csr-access"
    if int(operands.get("rm", -1)) in {5, 6}:
        return "reserved-rm"
    if getattr(schema, "kind", None) == "privileged-system":
        return "privileged-instruction-user-mode"
    if int(operands.get("rnum", 0)) > 10:
        return "reserved-selector"
    if enabled.isdisjoint({"a", "zacas", "zalrsc"}) and mnemonic in {
        "lr_w",
        "lr_d",
        "lr_b",
        "lr_h",
        "sc_w",
        "sc_d",
        "sc_b",
        "sc_h",
    }:
        return "extension-disabled"
    return None


def _compressed_definedness_disposition(form, schema, operands: Mapping[str, int]) -> str | None:
    disposition = compressed_codeword_disposition(form, schema, operands)
    mnemonic = str(getattr(form, "mnemonic", "")).replace(".", "_")
    if mnemonic == "c_addi" and operands.get("rd_rs1_n0") == 0:
        return "hint" if operands.get("c_imm6", 0) else "normal"
    if (
        mnemonic == "c_li" and operands.get("rd_n0") == 0
        or mnemonic == "c_mv"
        and operands.get("rd_n0") == 0
        and operands.get("c_rs2_n0") != 0
        or mnemonic == "c_slli" and operands.get("rd_rs1_n0") == 0
    ):
        return "hint"
    return disposition


def definedness_for_encoding(
    mnemonic: str,
    word: int,
    *,
    isa_profile: str,
    operands: Mapping[str, int] | None = None,
) -> SailDefinednessDecision:
    """Adjudicate one concrete encoding under one ISA profile.

    The ruling is a static legal/illegal fact plus the expected user-mode
    outcome.  ``decision_source`` names the authority: ``manual`` for Manual
    derived HINT/reserved facts, ``structural-encoding`` for pure catalog
    facts, and ``sail-definedness`` for Sail-endorsed normal rows.
    """
    profile = str(isa_profile or "").lower()
    sail_supported, sail_reason, config_identity = sail_profile_support(profile)

    def decision(
        disposition: str,
        outcome: str,
        source: str,
        reason: str,
        mcause: int | None = None,
    ) -> SailDefinednessDecision:
        return SailDefinednessDecision(
            disposition, outcome, source, reason, mcause,
            sail_supported, sail_reason, config_identity,
        )

    if type(word) is not int:
        raise ValueError("word must be a native integer")
    form = form_for_mnemonic(mnemonic)
    if operands is not None and not isinstance(operands, Mapping):
        raise ValueError("operands must be a mapping")
    if form is not None:
        width = int(getattr(form, "encoding_length_bytes", 0)) * 8
        if width not in {16, 32} or not 0 <= word < (1 << width):
            form = None
    if form is None:
        return decision(
            "illegal-encoding", "illegal-instruction", "structural-encoding",
            "unknown-or-outside-form", _RESERVED_MCAUSE,
        )
    if word & form.mask != form.match:
        schema = generation_schema_for_form(form)
        if (
            getattr(schema, "kind", None) == "integer-shift"
            and getattr(schema, "immediate_field", None) == "shamtw"
            and word & (1 << 25)
            and not ((word ^ form.match) & (form.mask & ~(1 << 25)))
        ):
            return decision(
                "illegal-reserved", "illegal-instruction", "manual",
                "reserved-shift-amount", _RESERVED_MCAUSE,
            )
        return decision(
            "illegal-encoding", "illegal-instruction", "structural-encoding",
            "outside-form-match", _RESERVED_MCAUSE,
        )
    decoded = decode_form(form, word)
    if operands is not None:
        supplied = dict(operands)
        if (
            set(supplied) - set(decoded)
            or any(type(value) is not int or decoded.get(name) != value for name, value in supplied.items())
        ):
            raise ValueError("operands do not match encoding")
    if not _profile_form_support(profile, form):
        return decision(
            "illegal-extension-disabled", "illegal-instruction", "manual",
            "extension-disabled", _RESERVED_MCAUSE,
        )
    structural = _structural_legality(form, word, profile, decoded)
    if structural is not None:
        return decision(
            "illegal-extension-disabled"
            if structural == "extension-disabled" else "illegal-reserved",
            "illegal-instruction", "manual", structural, _RESERVED_MCAUSE,
        )
    if not sail_supported:
        return decision(
            "blocked", "", "sail-definedness", sail_reason,
        )
    schema = generation_schema_for_form(form)
    disposition = _compressed_definedness_disposition(form, schema, decoded)
    if disposition == "hint":
        return decision(
            "legal-hint", "normal", "manual", "hint-encoding",
        )
    if disposition == "non-canonical":
        return decision(
            "illegal-reserved", "illegal-instruction", "manual",
            "non-canonical-compressed", _RESERVED_MCAUSE,
        )
    if str(form.mnemonic).replace(".", "_") == "ecall":
        return decision(
            "trap-expected", "environment-call", "manual",
            "environment-call", _ENVIRONMENT_MCAUSE,
        )
    if str(form.mnemonic).replace(".", "_") in {"ebreak", "c_ebreak"}:
        return decision(
            "trap-expected", "breakpoint", "manual", "breakpoint",
            _PRIVILEGED_MCAUSE,
        )
    return decision(
        "legal-normal", "normal", "sail-definedness", "normal-defined",
    )


def definedness_outcome_for_testcase(testcase: Any) -> SailDefinednessDecision | None:
    """Adjudicate the single risk instruction of a legality-lane testcase."""
    from .rvgen.admission import _declared_lane, _emitted_form
    from .rvgen.intents import LANE_LEGALITY

    try:
        if _declared_lane(testcase) != LANE_LEGALITY:
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    instruction_meta = getattr(testcase, "instruction_meta", ())
    if not isinstance(instruction_meta, (list, tuple)):
        return None
    risks = [
        item
        for item in instruction_meta
        if isinstance(getattr(item, "tags", ()), (list, tuple, set, frozenset))
        and "risk" in getattr(item, "tags", ())
    ]
    if len(risks) != 1:
        return None
    risk = risks[0]
    mnemonic = getattr(risk, "mnemonic", None)
    form = _emitted_form(mnemonic)
    if form is None:
        return None
    code_bytes = getattr(testcase, "code_bytes", None)
    offset = getattr(risk, "byte_offset", None)
    length = getattr(risk, "byte_length", None)
    alignment = 2 if profile_uses_compressed_encoding(getattr(testcase, "isa_profile", "")) else 4
    if (
        not isinstance(code_bytes, (bytes, bytearray, memoryview))
        or type(offset) is not int
        or type(length) is not int
        or offset < 0
        or offset % alignment
        or length != int(form.encoding_length_bytes)
        or offset + length > len(code_bytes)
        or not is_canonical_isa_profile(getattr(testcase, "isa_profile", None))
    ):
        return None
    word = int.from_bytes(
        code_bytes[offset : offset + length],
        "little",
    )
    return definedness_for_encoding(
        mnemonic,
        word,
        isa_profile=testcase.isa_profile,
    )

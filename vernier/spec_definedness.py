from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from collections.abc import Mapping
from typing import Any

from .riscv_catalog import (
    OFFICIAL_ALL_CATALOG_FORMS,
    OFFICIAL_SELECTED_SCALAR_FORMS,
    OfficialForm,
)
from .riscv_csrs import csr_required_extensions

from .riscv_encoding import (
    REGISTER_OPERAND_ROLES as _REGISTER_OPERAND_ROLES,
    SYSTEM_OPCODE,
    encoded_immediate_for_metadata,
    encode_instruction,
    form_for_mnemonic,
    implicit_source_roles,
    operand_kind,
    operand_groups,
    register_domain,
)


def compare_mask_requests_outcome_boundary(compare_mask: Any) -> bool:
    """Return whether a compare mask requests an outcome/trap boundary.

    The predicate only reads the compare-mask projection, so it belongs to the
    specification layer rather than the direct-case DTO.
    """
    return any(
        bool(getattr(compare_mask, field, False))
        for field in ("signal", "fault_pc", "fault_address")
    )


# 中立 ISA catalog 的对外面：RVEMI 只从这里读官方编码/字段/扩展事实与编码器。
# 执行语义不由 riscv-opcodes 定义；真实语义所有权仍在 ISA / Sail / native reference。

# Keep the standard single-letter order, including V.  V remains a base-profile
# extension (not an underscore suffix); its required F/D/Zicsr prerequisites
# are expanded below while the ``v`` letter stays in canonical base order.
_BASE_ORDER = "imafdqcvphs"
_NON_ISA_SOURCE_TOKENS = frozenset({"system"})
_X_REGISTER_FP_EXTENSIONS = frozenset({"zfinx", "zdinx", "zhinx", "zhinxmin"})
# Structural privileged forms are classified from the same extension facts that
# drive the effect owner.  Keeping this set here gives realization identity and
# effect classification one source of truth without a mnemonic table.
_PRIVILEGED_STATE_EXTENSIONS = frozenset(
    {"h", "svinval", "svinval_h", "sdext", "smrnmi", "ssctr", "zicfiss"}
)
_RV32_XREGISTER_COMPATIBILITY_BY_REQUIRED = {
    frozenset({"f"}): "rv32i_zfinx",
    frozenset({"d", "f"}): "rv32i_zdinx_zfinx",
    frozenset({"f", "zfhmin"}): "rv32i_zfinx_zhinxmin",
    frozenset({"f", "zfh"}): "rv32i_zfinx_zhinx",
    frozenset({"d", "f", "zfhmin"}): "rv32i_zdinx_zfinx_zhinxmin",
}
_RV32_XREGISTER_WRITEBACK_COMPATIBILITY = dict.fromkeys(
    ("fsgnj_h", "fsgnjn_h", "fsgnjx_h"), "rv32i_zfinx_zhinx"
)
_RV32_XREGISTER_PAIR_COMPATIBILITY = {
    "fcvt_d_h": ("rv32i_zdinx_zfinx_zhinxmin", "rd"),
    "fcvt_h_d": ("rv32i_zdinx_zfinx_zhinxmin", "rs1"),
}
# These catalog rows use the RV64 immediate field width, but the operation is
# shared with RV32; RV32's high shift/index bit is reserved and audited later.
_RV32_SHARED_XLEN_MNEMONICS = frozenset(
    {
        "slli",
        "srli",
        "srai",
        "rori",
        "bseti",
        "bclri",
        "binvi",
        "bexti",
        "c_slli",
        "c_srli",
        "c_srai",
    }
)
# ISA 手册级的扩展蕴含事实（例如 ``D`` 蕴含 ``F``、``Zcb`` 蕴含 ``C``）。
# pinned riscv-opcodes catalog 只记录每条指令的最小扩展要求，不表达扩展之间
# 的蕴含：``fld`` 的 source 是 ``('rv_d',)``、``c_lbu`` 是 ``('rv_zcb',)``，
# 单独出现的 token 证明 catalog 无法派生出 ``d -> f`` / ``zcb -> c`` 这类
# 蕴含（组合 source 如 ``rv_c_d`` 只说明单条指令同时需要多个扩展）。因此这
# 里保留最小映射，来源是 RISC-V ISA 手册的扩展依赖章节，而不是助记符清单。
_IMPLIED_EXTENSIONS = {
    # F/D/Q and the integer-register FP families all use Zicsr for their
    # control/status registers (ISA naming conventions and Zfinx chapter).
    "f": frozenset({"f", "zicsr"}),
    "d": frozenset({"f", "d", "zicsr"}),
    "q": frozenset({"f", "d", "q", "zicsr"}),
    "zfinx": frozenset({"zfinx", "zicsr"}),
    "zdinx": frozenset({"zfinx", "zdinx", "zicsr"}),
    "zhinx": frozenset({"zfinx", "zhinx", "zicsr"}),
    "zhinxmin": frozenset({"zfinx", "zhinxmin", "zicsr"}),
    # The application vector extension depends on F, D and Zicsr.  Keep the
    # vector letter in the base spelling; only the architectural prerequisites
    # are expanded into the profile identity.
    "v": frozenset({"v", "d", "f", "zicsr"}),
    "zabha": frozenset({"a", "zabha"}),
    "zabhlrsc": frozenset({"zalrsc", "zabhlrsc"}),
    "zacas": frozenset({"a", "zacas"}),
    "zalasr": frozenset({"a", "zalasr"}),
    "zawrs": frozenset({"a", "zawrs"}),
    "zcb": frozenset({"c", "zcb"}),
    "zcmp": frozenset({"c", "zcmp"}),
    "zcmt": frozenset({"c", "zcmt", "zicsr"}),
    "zfh": frozenset({"f", "zfhmin", "zfh", "zicsr"}),
    "zfhmin": frozenset({"f", "zfhmin", "zicsr"}),
    "zfbfmin": frozenset({"f", "zfbfmin", "zicsr"}),
    "zfa": frozenset({"f", "zfa", "zicsr"}),
    "zicntr": frozenset({"zicntr", "zicsr"}),
    "zihpm": frozenset({"zihpm", "zicsr"}),
    "zicfiss": frozenset({"zicfiss", "zaamo", "zicsr", "zimop"}),
    "zicfilp": frozenset({"zicfilp", "zicsr"}),
    # Embedded vector profiles are transitive dependencies, not aliases for
    # the full V extension.  Keep the closure explicit so profile identity
    # and Sail admission see the same Zve base.
    "zve32x": frozenset({"zve32x", "zicsr"}),
    "zve32f": frozenset({"zve32f", "zve32x", "f", "zicsr"}),
    "zve64x": frozenset({"zve64x", "zve32x", "zicsr"}),
    "zve64f": frozenset({"zve64f", "zve64x", "zve32f", "f", "zicsr"}),
    "zve64d": frozenset({"zve64d", "zve64f", "zve64x", "zve32f", "f", "d", "zicsr"}),
    # Zvfh is the vector half-precision family.  A full V profile already
    # carries the F/D/Zicsr prerequisites; spelling them here also lets the
    # canonical validator reason about an explicit ``..._zvfh`` suffix.
    "zvfh": frozenset({"zvfh", "v", "d", "f", "zicsr"}),
    # Zvfbfa v0.9 draft: BF16 vector compute uses the altfmt VTYPE bit and
    # requires scalar BF16 support; vector-base alternatives are handled below.
    "zvfbfa": frozenset({"zvfbfa", "zfbfmin", "f", "zicsr"}),
}


@dataclass(frozen=True)
class SemanticRealization:
    """Minimal read-only realization facts for one catalog form."""

    xlen: int
    extension_tokens: tuple[str, ...]
    register_carrier: str
    privilege_class: str
    isa_profile: str
    environment_profile: str = "base"

    def __post_init__(self) -> None:
        if not isinstance(self.xlen, int) or isinstance(self.xlen, bool) or self.xlen not in {32, 64}:
            raise ValueError("semantic realization xlen must be 32 or 64")
        if type(self.register_carrier) is not str or self.register_carrier not in {"gpr", "fpr"}:
            raise ValueError("semantic realization register carrier must be gpr or fpr")
        if type(self.privilege_class) is not str or not self.privilege_class:
            raise ValueError("semantic realization privilege_class is required")
        if type(self.isa_profile) is not str or not self.isa_profile:
            raise ValueError("semantic realization isa_profile is required")
        if not is_canonical_isa_profile(self.isa_profile, xlen=self.xlen):
            raise ValueError("semantic realization isa_profile is not canonical for xlen")
        if not isinstance(self.environment_profile, str) or not self.environment_profile:
            raise ValueError("semantic realization environment_profile is required")
        if (
            type(self.extension_tokens) is not tuple
            or
            not self.extension_tokens
            or any(type(token) is not str or not token for token in self.extension_tokens)
            or tuple(self.extension_tokens) != tuple(sorted(set(self.extension_tokens)))
        ):
            raise ValueError("semantic realization extension_tokens must be sorted and non-empty")
        if frozenset(self.extension_tokens) != enabled_extensions(self.isa_profile):
            raise ValueError(
                "semantic realization extension_tokens must match isa_profile"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "xlen": self.xlen,
            "extension_tokens": list(self.extension_tokens),
            "register_carrier": self.register_carrier,
            "privilege_class": self.privilege_class,
            "isa_profile": self.isa_profile,
            "environment_profile": self.environment_profile,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SemanticRealization":
        return cls(
            xlen=int(data["xlen"]),
            extension_tokens=tuple(data["extension_tokens"]),
            register_carrier=str(data["register_carrier"]),
            privilege_class=str(data["privilege_class"]),
            isa_profile=str(data["isa_profile"]),
            environment_profile=str(data["environment_profile"]),
        )

    @property
    def realization_id(self) -> str:
        from ._translation_cell import canonical_realization_id

        return canonical_realization_id(
            xlen=self.xlen,
            extension_tokens=self.extension_tokens,
            register_carrier=self.register_carrier,
            privilege_class=self.privilege_class,
            isa_profile=self.isa_profile,
            environment_profile=self.environment_profile,
        )


def _source_extension_tokens(source_extension: str) -> tuple[str, ...]:
    source = str(source_extension).lower().removeprefix("unratified/")
    for prefix in ("rv32_", "rv64_", "rv_"):
        if source.startswith(prefix):
            source = source.removeprefix(prefix)
            break
    return tuple(token for token in source.split("_") if token)


# Ratified vector profile tokens (``zve*``/``zvl*``).  The base letter ``v``
# stays in the single-letter base spelling, while these suffix tokens are the
# legal vector profile spellings the native ISA map must recognize.
_VECTOR_PROFILE_TOKENS = frozenset(
    {
        "zve32x",
        "zve32f",
        "zve64x",
        "zve64f",
        "zve64d",
        "zvl32b",
        "zvl64b",
        "zvl128b",
        "zvl256b",
        "zvl512b",
        "zvl1024b",
    }
)

# Vector ISA source rows name the Zv* member but omit the vector base profile.
# Keep the dependency at extension level so the same rule covers new forms and
# does not grow a mnemonic table.  V is the application base; Zve32x/Zve64x
# are the embedded alternatives.  The 64-bit crypto group and vector FP/BF16
# groups have the narrower base/FP prerequisites stated by the ISA Manual.
_VECTOR_ZVE64X_EXTENSIONS = frozenset(
    {
        "zvbc",
        "zvkn",
        "zvknc",
        "zvkng",
        "zvknhb",
        "zvksc",
    }
)


@cache
def _known_profile_suffix_tokens() -> frozenset[str]:
    """Extension tokens admitted by the pinned opcode catalog."""
    tokens = set(_IMPLIED_EXTENSIONS)
    tokens.update(_X_REGISTER_FP_EXTENSIONS)
    tokens.update({
        "zaamo", "zalrsc", "zicbom", "zicboz", "zicntr", "zihpm",
        "zicsr", "zihintpause", "zkr",
        # 标准扩展 token，pinned catalog 的 source_extensions 里没有直接出现，
        # 但 GCC 与模拟器都接受，且会被 isa_profile_for_extensions 产出。
        "zca", "zcd", "zicbop", "zkt", "zvkb", "zvkt",
    })
    tokens.update((*_BASE_ORDER, "e"))
    tokens.update(_PRIVILEGED_STATE_EXTENSIONS)
    tokens.update(_VECTOR_PROFILE_TOKENS)
    for form in OFFICIAL_ALL_CATALOG_FORMS:
        for source in getattr(form, "source_extensions", ()):
            for token in _source_extension_tokens(str(source)):
                tokens.update(
                    part
                    for part in token.split("+")
                    if part and part not in _NON_ISA_SOURCE_TOKENS
                )
    return frozenset(tokens)


@cache
def _expanded_extensions(tokens: tuple[str, ...]) -> frozenset[str]:
    result: set[str] = set()
    pending = list(tokens)
    while pending:
        token = pending.pop()
        if token in _NON_ISA_SOURCE_TOKENS or token in result:
            continue
        result.add(token)
        pending.extend(_IMPLIED_EXTENSIONS.get(token, ()))
    return frozenset(result)


def expanded_extensions(tokens: tuple[str, ...]) -> frozenset[str]:
    if type(tokens) is not tuple or any(type(token) is not str or not token for token in tokens):
        raise ValueError("extension tokens must be a tuple of non-empty native strings")
    return _expanded_extensions(tokens)


def _normalized_required_extensions(required: frozenset[str]) -> frozenset[str]:
    normalized = set(required)
    # Zicsr is an implicit prerequisite for several extension families.  It
    # remains in the generated profile, but it must not alter the semantic
    # family key used by x-register compatibility and form admission.
    if len(normalized) > 1:
        normalized.discard("zicsr")
    if "zfh" in normalized and "zfhmin" in normalized:
        normalized.discard("zfhmin")
    return frozenset(normalized)


def _vector_required_extension_sets(
    required: frozenset[str],
) -> tuple[frozenset[str], ...]:
    """Add the official vector-base alternatives to one Zv* requirement."""

    vector_tokens = frozenset(
        token
        for token in required
        if token.startswith("zv") and not token.startswith(("zve", "zvl"))
    )
    if not vector_tokens:
        return (required,)

    common = set(required)
    # The ratified vector BF16 forms require an F-capable base.  The widening
    # BF16 multiply-accumulate also depends on both BF16 extensions.
    has_bf16 = bool(vector_tokens & {"zvfbfmin", "zvfbfwma", "zvfbfa"})
    if has_bf16:
        common.update({"f"})
    if "zvfbfwma" in vector_tokens:
        common.update({"zfbfmin", "zvfbfmin"})

    embedded_base = (
        "zve64x"
        if vector_tokens & _VECTOR_ZVE64X_EXTENSIONS
        else "zve32f"
        if has_bf16
        else "zve32x"
    )
    return (
        frozenset((*common, "v")),
        frozenset((*common, embedded_base)),
    )


@cache
def _required_extension_sets_for_form(form: OfficialForm) -> tuple[frozenset[str], ...]:
    alternatives: list[frozenset[str]] = []
    for source in getattr(form, "source_extensions", ()):
        required = _normalized_required_extensions(
            expanded_extensions(_source_extension_tokens(source))
        )
        for candidate in _vector_required_extension_sets(required):
            if candidate and candidate not in alternatives:
                alternatives.append(candidate)
    if not alternatives:
        # Privileged/system catalog rows carry an environment tag, not an
        # ISA extension suffix.  They still use the base profile identity.
        alternatives.append(frozenset())
    return tuple(alternatives)


def required_extension_sets_for_form(form) -> tuple[frozenset[str], ...]:
    if not isinstance(form, OfficialForm):
        raise ValueError("catalog form must be an OfficialForm")
    return _required_extension_sets_for_form(form)


def isa_profile_for_extensions(xlen: str, required_extensions: frozenset[str]) -> str:
    if type(xlen) is not str:
        raise ValueError("RISC-V XLEN must be a native string")
    if type(required_extensions) is not frozenset or any(
        type(token) is not str or not token for token in required_extensions
    ):
        raise ValueError("required extensions must be a frozenset of non-empty native strings")
    if xlen not in {"rv32", "rv64"}:
        raise ValueError(f"unsupported RISC-V XLEN: {xlen!r}")
    # Build from the architectural closure, not the spelling of one source
    # row.  ``D``/``Q`` and extension bundles such as ``Zcmp`` imply other
    # extensions; omitting those implications produces profiles (for example
    # ``rv64id``) that this module's own canonical validator rejects.
    required = set(expanded_extensions(tuple(required_extensions)))
    vector_tokens = {
        token for token in required
        if token.startswith("zv") and not token.startswith(("zve", "zvl"))
    }
    if vector_tokens and "v" not in required and not any(
        token.startswith("zve") for token in required
    ):
        required.update(min(
            _vector_required_extension_sets(frozenset(required)),
            key=lambda item: (len(item), tuple(sorted(item))),
        ))
        required = set(expanded_extensions(tuple(required)))
    if {"zcmp", "zcmt"} & required and "d" in required:
        raise ValueError("Zcmp/Zcmt cannot be combined with D")
    had_g_alias = "g" in required
    if had_g_alias:
        # G is an ISA spelling alias, not an independent extension token.
        # Expand it before rebuilding a canonical profile so an enabled G
        # profile cannot round-trip to an unbuildable ``...g`` suffix.
        required.discard("g")
        required.update({"i", "m", "a", "f", "d"})
        required = set(expanded_extensions(tuple(required)))
    rv32e = xlen == "rv32" and "e" in required
    if "e" in required and (xlen != "rv32" or had_g_alias):
        raise ValueError("RISC-V E base is only valid for RV32 profiles")
    if rv32e:
        # E is an alternative base ISA, not an I suffix.  Keeping both letters
        # would produce the non-canonical (and generally rejected) ``rv32ie``
        # toolchain identity.
        required.difference_update({"e", "i"})
        base = "rv32e"
        base_order = "mafdqc"
    else:
        required.add("i")
        base = "rv32" if xlen == "rv32" else "rv64"
        base_order = _BASE_ORDER
    if "zfh" in required:
        required.discard("zfhmin")
    base += "".join(extension for extension in base_order if extension in required)
    base += "".join(
        extension
        for extension in sorted(required)
        if len(extension) == 1 and extension not in base_order
    )
    suffixes = sorted(extension for extension in required if len(extension) > 1)
    return base if not suffixes else f"{base}_{'_'.join(suffixes)}"


@cache
def _is_canonical_isa_profile(isa_profile: str, *, xlen: int | None = None) -> bool:
    """Accept only the lowercase profile spelling emitted by RVGEN.

    The profile is an execution identity, not a free-form extension bag.  In
    particular, RV32E is an alternative base (``rv32e``), never ``rv32ie``;
    keeping this check beside profile construction prevents metadata and ELF
    identity from diverging before any backend is invoked.
    """
    if type(isa_profile) is not str:
        return False
    if xlen is not None and type(xlen) is not int:
        return False
    profile = str(isa_profile)
    if profile != profile.lower():
        return False
    head, *suffixes = profile.split("_", 1)
    if head.startswith("rv32"):
        profile_xlen = 32
    elif head.startswith("rv64"):
        profile_xlen = 64
    else:
        return False
    if xlen is not None:
        requested_xlen = xlen
        if requested_xlen != profile_xlen:
            return False
    base = head[4:]
    if not base or (
        "e" in base
        and (profile_xlen != 32 or "i" in base or "g" in base)
    ):
        return False
    if any(character not in _BASE_ORDER + "eg" for character in base):
        return False
    # ``g`` expands IMAFD, but it may still be followed by other canonical
    # single-letter extensions (for example ``rv64gcv``).  Reject only
    # duplicate/reordered base spellings such as ``rv64ig`` or ``rv64gm``;
    # those cannot be reproduced by the profile builder without ambiguity.
    if "g" in base:
        if not base.startswith("g"):
            return False
        g_tail = base[1:]
        canonical_g_tail = "".join(
            extension
            for extension in _BASE_ORDER
            if extension not in "imafd" and extension in g_tail
        )
        if g_tail != canonical_g_tail:
            return False
    suffix_tokens = tuple(suffixes[0].split("_")) if suffixes else ()
    if suffix_tokens != tuple(sorted(set(suffix_tokens))):
        return False
    if any(token not in _known_profile_suffix_tokens() for token in suffix_tokens):
        return False
    if {"zcmp", "zcmt"} & set(suffix_tokens) and ("d" in base or "g" in base):
        return False
    has_vector_base = "v" in base or any(
        token.startswith("zve") for token in suffix_tokens
    )
    if (
        any(token.startswith("zv") and not token.startswith(("zve", "zvl")) for token in suffix_tokens)
        or any(token.startswith("zvl") for token in suffix_tokens)
    ) and not has_vector_base:
        return False
    required = set(
        expanded_extensions(tuple(base) + tuple(token for token in suffix_tokens if token))
    )
    vector_tokens = {
        token for token in required
        if token.startswith("zv") and not token.startswith(("zve", "zvl"))
    }
    if vector_tokens and not any(
        set(candidate) <= required
        for candidate in _vector_required_extension_sets(frozenset(required))
    ):
        return False
    # Keep standard G aliases (including ``gcv``) accepted for concrete
    # witnesses; new RVGEN realizations still use the expanded canonical
    # form produced by isa_profile_for_extensions().
    if "g" in base:
        # Historical G/GC aliases remain accepted, but their suffixes still
        # have to come from the pinned catalog.  Otherwise ``rv64g_foo``
        # would pass identity validation and leak an unbuildable -march.
        return not any(len(token) == 1 for token in suffix_tokens)
    # RVGEN emits the fully expanded spelling, but the ISA naming convention
    # permits Zicsr to remain implicit in F/D/Q and dependent profiles.  Keep
    # accepting that standard alias while retaining one explicit generated
    # identity for new realizations.
    try:
        canonical = isa_profile_for_extensions(
            f"rv{profile_xlen}",
            frozenset(required),
        )
    except ValueError:
        return False
    if profile == canonical:
        return True
    # Zicsr is the one dependency routinely omitted by standard ISA aliases
    # (for example RV64IF is equivalent to RV64IFZicsr).  Keep that alias
    # accepted while requiring all other prerequisite extensions to be
    # explicit in the canonical RVGEN identity.
    canonical_head, *canonical_suffixes = canonical.split("_")
    if "zicsr" not in canonical_suffixes or "zicsr" in suffix_tokens:
        return any(token.startswith("zve") for token in suffix_tokens) and (
            frozenset(required) == expanded_extensions(
                tuple(base) + tuple(token for token in suffix_tokens if token)
            )
        )
    alias_suffixes = tuple(token for token in canonical_suffixes if token != "zicsr")
    alias = canonical_head if not alias_suffixes else "_".join((canonical_head, *alias_suffixes))
    if profile == alias:
        return True
    return any(token.startswith("zve") for token in suffix_tokens) and (
        frozenset(required) == expanded_extensions(
            tuple(base) + tuple(token for token in suffix_tokens if token)
        )
    )


def is_canonical_isa_profile(isa_profile: str, *, xlen: int | None = None) -> bool:
    if type(isa_profile) is not str or (xlen is not None and type(xlen) is not int):
        return False
    return _is_canonical_isa_profile(isa_profile, xlen=xlen)


@cache
def _enabled_extensions(isa_profile: str) -> frozenset[str]:
    profile = isa_profile.lower()
    head, *suffixes = profile.split("_")
    parsed = set(head[4:])
    if "g" in parsed:
        parsed.update({"i", "m", "a", "f", "d"})
    if "e" in parsed:
        parsed.add("i")
    parsed.update(token for token in suffixes if token)
    return _expanded_extensions(tuple(parsed))


def enabled_extensions(isa_profile: str) -> frozenset[str]:
    if type(isa_profile) is not str:
        return frozenset()
    profile = isa_profile.lower()
    if not profile.startswith(("rv32", "rv64")):
        return frozenset()
    return _enabled_extensions(isa_profile)


def mabi_matches_isa(isa_profile: str, mabi: str) -> bool:
    if not isinstance(isa_profile, str) or not isinstance(mabi, str):
        return False
    isa = isa_profile.lower()
    abi = mabi.lower()
    base = "ilp32" if isa.startswith("rv32") else "lp64" if isa.startswith("rv64") else ""
    if not base or not abi.startswith(base):
        return False
    suffix = abi[len(base):]
    if suffix == "e":
        return isa.startswith("rv32e")
    if isa.startswith("rv32e") or suffix == "e":
        return False
    return suffix in {"", "f", "d", "q"} and (
        not suffix or suffix in enabled_extensions(isa)
    )


def isa_base_extensions(isa_profile: str) -> str:
    profile = str(isa_profile or "").lower()
    return profile.split("_", 1)[0][4:] if profile.startswith(("rv32", "rv64")) else ""


def profile_uses_compressed_encoding(isa_profile: str) -> bool:
    """Return whether a profile permits 16-bit instruction alignment."""

    extensions = enabled_extensions(isa_profile)
    return "c" in extensions or any(token.startswith("zc") for token in extensions)


def x_register_fp_extensions(isa_profile: str) -> tuple[str, ...]:
    return tuple(sorted(enabled_extensions(isa_profile) & _X_REGISTER_FP_EXTENSIONS))


def profile_requires_fp_state(isa_profile: str) -> bool:
    """Return whether a profile requires final architectural FP state."""

    profile = str(isa_profile or "").lower()
    if not profile.startswith("rv"):
        return False
    enabled = enabled_extensions(profile)
    return bool(enabled & {"f", "d", "q", "g"}) or bool(
        x_register_fp_extensions(profile)
    )


def csr_required_extensions_for_profile(
    address: int,
    isa_profile: str,
) -> frozenset[str]:
    """Return CSR extension requirements under the concrete register view.

    ``fflags``/``frm``/``fcsr`` are shared architectural state.  The plain CSR
    table reports their conventional ``f`` owner, while Zfinx/Zdinx/Zhinx keep
    the same state with FP operands carried in GPRs.  Centralizing this small
    profile exception prevents materialization, definedness, and admission from
    independently inventing incompatible profile upgrades.
    """
    if type(address) is not int or isinstance(address, bool) or type(isa_profile) is not str:
        return frozenset()
    required = csr_required_extensions(address)
    if required == frozenset({"f"}) and x_register_fp_extensions(isa_profile):
        return frozenset()
    return required


def _compatibility_profile_matches(
    isa_profile: str,
    compatibility_profile: str,
) -> bool:
    """Match a compatibility view while allowing required profile suffixes.

    The compatibility table names the minimum x-register extension view (for
    example ``rv32i_zfinx``).  Materialization may add independent contracts
    such as ``zicsr`` for an observer; those additions must not switch the
    source/destination register domain back to FPR.
    """
    actual_xlen = _profile_xlen(isa_profile)
    required_xlen = _profile_xlen(compatibility_profile)
    if actual_xlen is None or actual_xlen != required_xlen:
        return False
    return enabled_extensions(compatibility_profile) <= enabled_extensions(isa_profile)


def _xregister_transfer_form(form) -> bool:
    """Return forms excluded by the Zfinx-family transfer rules.

    The x-register extensions reuse the F/D/H encodings for computation, but
    explicitly exclude floating-point memory transfers and FMV transfers.
    Derive that boundary from opcode/register facts so new precision forms do
    not require a mnemonic allow-list.
    """
    domains = tuple(
        register_domain(form, role)
        for role in operand_groups(form)
        if register_domain(form, role) is not None
    )
    if "fpr" not in domains:
        return False
    if int(getattr(form, "encoding_length_bytes", 0)) == 2:
        return True
    major = int(getattr(form, "match", 0)) & 0x7F
    if major in {0x07, 0x27}:
        return True
    if major != 0x53:
        return False
    funct5 = (int(getattr(form, "match", 0)) >> 27) & 0x1F
    funct3 = (int(getattr(form, "match", 0)) >> 12) & 0x7
    # FMV.X.{H,S,D} and FMV.{H,S,D}.X use rs1=0 in the transfer encoding;
    # FLI uses rs1=1 and remains a literal-producing FP operation.
    return funct3 == 0 and funct5 in {0x1C, 0x1E} and ((int(getattr(form, "match", 0)) >> 20) & 0x1F) == 0


def _rv32_xregister_profile(form, exclude_transfers: bool = False) -> str | None:
    if str(getattr(form, "xlen", "shared")) not in {"shared", "rv32"}:
        return None
    if not any(register_domain(form, role) == "fpr" for role in operand_groups(form)):
        return None
    if exclude_transfers and _xregister_transfer_form(form):
        return None
    profiles = (
        profile
        for required in required_extension_sets_for_form(form)
        if (
            profile := _RV32_XREGISTER_COMPATIBILITY_BY_REQUIRED.get(
                _normalized_required_extensions(required)
            )
        ) is not None
    )
    return min(profiles, default=None, key=lambda item: (len(item), item))


def rv32_xregister_compatibility_profile(form) -> str | None:
    return _rv32_xregister_profile(form, exclude_transfers=True)


def rv32_xregister_pair_compatibility_profile(form) -> str | None:
    if str(getattr(form, "xlen", "shared")) not in {"shared", "rv32"}:
        return None
    if not any(
        _normalized_required_extensions(required) == frozenset({"f", "d", "zfhmin"})
        for required in required_extension_sets_for_form(form)
    ):
        return None
    row = _RV32_XREGISTER_PAIR_COMPATIBILITY.get(str(getattr(form, "mnemonic", "")))
    return None if row is None else str(row[0])


def rv32_xregister_writeback_compatibility_profile(form) -> str | None:
    if (
        str(getattr(form, "xlen", "shared")) not in {"shared", "rv32"}
        or not any(
            frozenset({"f", "zfh"}) <= required
            for required in required_extension_sets_for_form(form)
        )
    ):
        return None
    profile = _RV32_XREGISTER_WRITEBACK_COMPATIBILITY.get(str(getattr(form, "mnemonic", "")))
    return None if profile is None else str(profile)


def rv32_xregister_writeback_compatibility_active(
    isa_profile: str,
    form,
) -> bool:
    profile = rv32_xregister_writeback_compatibility_profile(form)
    return profile is not None and _compatibility_profile_matches(isa_profile, profile)


def xregister_compatibility_slot_domain_for_mnemonic(
    isa_profile: str | None,
    mnemonic: str,
    role: str,
    register: int | None,
) -> str | None:
    if register is None:
        return None
    form = form_for_mnemonic(mnemonic)
    default_domain = "gpr" if form is None else register_domain(form, role)
    if not isa_profile:
        return default_domain
    if form is None:
        return default_domain
    return (
        "gpr"
        if default_domain == "fpr"
        and x_register_fp_extensions(isa_profile)
        and not _xregister_transfer_form(form)
        else default_domain
    )


def _profile_enables_required_extensions(enabled: frozenset[str], required: frozenset[str]) -> bool:
    required = _normalized_required_extensions(required)
    if required == {"f"} and "zfinx" in enabled:
        return True
    # The x-register precision extensions are layered: Zdinx, Zhinx and
    # Zhinxmin all depend on Zfinx.  Check the complete dependency here,
    # rather than accepting a bare suffix that cannot form a valid
    # compatibility profile.
    if required == {"f", "d"} and {"zfinx", "zdinx"} <= enabled:
        return True
    if required == {"f", "zfh"} and {"zfinx", "zhinx"} <= enabled:
        return True
    if required == {"f", "zfhmin"} and {"zfinx", "zhinxmin"} <= enabled:
        return True
    return required <= enabled


def _atomic_subextension_form_enabled(
    enabled: frozenset[str],
    form: OfficialForm,
) -> bool:
    if not {"zaamo", "zalrsc"} & enabled:
        return False
    if not any(
        _normalized_required_extensions(required) == frozenset({"a"})
        for required in required_extension_sets_for_form(form)
    ) or int(getattr(form, "match", 0)) & 0x7F != 0x2F:
        return False
    function = (int(getattr(form, "match", 0)) >> 27) & 0x1F
    return (
        "zalrsc" in enabled and function in {2, 3}
    ) or (
        "zaamo" in enabled and function in {0, 1, 4, 8, 12, 16, 20, 24, 28}
    )


def _embedded_vector_form_enabled(
    enabled: frozenset[str],
    form: OfficialForm,
) -> bool:
    if "v" in enabled or not any(token.startswith("zve") for token in enabled):
        return False
    if not str(form.mnemonic).replace("_", ".").startswith("v"):
        return False
    max_elen = 64 if enabled & {"zve64x", "zve64f", "zve64d"} else 32
    match = int(getattr(form, "match", 0))
    if match & 0x7F in {0x07, 0x27}:
        return {0: 8, 5: 16, 6: 32, 7: 64}.get((match >> 12) & 0x7, 0) <= max_elen
    mnemonic = str(form.mnemonic).replace("_", ".")
    if mnemonic.startswith(("vf", "vmf")):
        if "f" not in enabled:
            return False
        parts = mnemonic.split(".")
        if mnemonic.startswith("vfncvt.") and len(parts) > 2 and parts[2] == "f":
            return "d" in enabled
        if mnemonic.startswith("vfwcvt.") and len(parts) > 1 and parts[1] == "f":
            return "d" in enabled
        if mnemonic.startswith("vfw") and not mnemonic.startswith("vfwcvt.x"):
            return "d" in enabled
    return True


def _profile_xlen(isa_profile: str) -> str | None:
    if type(isa_profile) is not str:
        return None
    profile = isa_profile.lower()
    if profile.startswith("rv32"):
        return "rv32"
    if profile.startswith("rv64"):
        return "rv64"
    return None


def _form_requires_rv64(form) -> bool:
    """Return structural RV64-only facts missing from a shared catalog row."""
    if str(getattr(form, "xlen", "shared")) == "rv64":
        return True
    source_tokens = {
        token
        for source in getattr(form, "source_extensions", ())
        for token in _source_extension_tokens(str(source))
    }
    # Zalasr's 64-bit load/store encodings are shared in riscv-opcodes, while
    # the Manual constrains the ld.aq/sd.rl forms to RV64.  The width is a
    # fixed encoding fact, so keep the correction structural and mnemonic-free.
    if "zalasr" in source_tokens and ((int(getattr(form, "match", 0)) >> 12) & 0x7) == 0x3:
        return True
    # RV32 V excludes the indexed EEW=64 encodings.  The index EEW and
    # addressing mode are fixed encoding facts; ordinary EEW=64 loads/stores
    # remain legal on RV32.
    if (
        "v" in source_tokens
        and (int(getattr(form, "match", 0)) & 0x7F) in {0x07, 0x27}
        and "vs2" in operand_groups(form)
        and ((int(getattr(form, "match", 0)) >> 26) & 0x3) in {0x1, 0x3}
        and ((int(getattr(form, "match", 0)) >> 12) & 0x7) == 0x7
    ):
        return True
    # C.FLD/C.FSD use the compressed 64-bit offset carriers while the opcode
    # catalog keeps their XLEN tag shared.  The carrier width is an encoding
    # fact; deriving the rule from it avoids a mnemonic allowlist and prevents
    # these RV64-only forms from entering an RV32 profile with D enabled.
    return (
        {"c", "d"} <= source_tokens
        and any(
            str(field).startswith(("c_uimm8", "c_uimm9"))
            for field in getattr(form, "variable_fields", ())
        )
    )


def _profile_xlen_matches_form(isa_profile: str, form) -> bool:
    profile_xlen = _profile_xlen(isa_profile)
    form_xlen = str(getattr(form, "xlen", "shared"))
    if profile_xlen is None:
        return False
    if form_xlen == "shared" and _form_requires_rv64(form):
        return profile_xlen == "rv64"
    if form_xlen in {"shared", profile_xlen}:
        return True
    return (
        profile_xlen == "rv32"
        and form_xlen == "rv64"
        and str(getattr(form, "mnemonic", "")) in _RV32_SHARED_XLEN_MNEMONICS
    )


@cache
def _profile_enables_form(isa_profile: str, form: OfficialForm) -> bool:
    if not is_canonical_isa_profile(isa_profile):
        return False
    if not _profile_xlen_matches_form(isa_profile, form):
        return False
    if _xregister_transfer_form(form) and x_register_fp_extensions(isa_profile):
        return False
    if rv32_xregister_writeback_compatibility_active(isa_profile, form):
        return True
    compatibility_profile = rv32_xregister_pair_compatibility_profile(form)
    if compatibility_profile is not None and _compatibility_profile_matches(
        isa_profile,
        compatibility_profile,
    ):
        return True
    enabled = enabled_extensions(isa_profile)
    if _atomic_subextension_form_enabled(enabled, form):
        return True
    if any(
        _profile_enables_required_extensions(enabled, required)
        for required in required_extension_sets_for_form(form)
    ):
        return True
    return any(
        set(required) <= {"v", "f", "d"}
        and _embedded_vector_form_enabled(enabled, form)
        for required in required_extension_sets_for_form(form)
    )


def profile_enables_form(isa_profile: str, form) -> bool:
    if type(isa_profile) is not str or not isinstance(form, OfficialForm):
        return False
    return _profile_enables_form(isa_profile, form)


@cache
def semantic_realizations_for_form(
    form,
    *,
    writeback: str | None = None,
) -> tuple[SemanticRealization, ...]:
    if not isinstance(form, OfficialForm):
        raise ValueError("catalog form must be an OfficialForm")
    xlen_tag = str(getattr(form, "xlen", "shared"))
    mnemonic = str(getattr(form, "mnemonic", ""))
    if xlen_tag == "shared" and _form_requires_rv64(form):
        xlens = (64,)
    elif xlen_tag == "shared":
        xlens = (32, 64)
    elif xlen_tag == "rv32":
        xlens = (32,)
    elif xlen_tag == "rv64" and mnemonic in _RV32_SHARED_XLEN_MNEMONICS:
        # A few upstream rows carry the RV64 field width even though the
        # operation exists at both XLENs; retain one catalog row and emit both
        # concrete realizations.  RV32 reserved shift/index bits are audited
        # from the encoded boundary, not by inventing a second form.
        xlens = (32, 64)
    else:
        xlens = (64,)
    register_carrier = (
        "fpr"
        if writeback == "fpr"
        or any(register_domain(form, role) == "fpr" for role in operand_groups(form))
        else "gpr"
    )
    privilege_class = "user"
    source_tokens = {
        token
        for source in getattr(form, "source_extensions", ())
        for token in _source_extension_tokens(str(source))
    }
    if getattr(form, "mnemonic", "") in {"ecall", "ebreak", "c_ebreak"}:
        privilege_class = "trap"
    elif (
        (
            int(getattr(form, "encoding_length_bytes", 0)) == 4
            and ((int(getattr(form, "match", 0)) & 0x7F) == SYSTEM_OPCODE)
            and (source_tokens & {"system", "s"})
        )
        or source_tokens & _PRIVILEGED_STATE_EXTENSIONS
    ):
        privilege_class = "system"
    required_sets = required_extension_sets_for_form(form)
    environment_profile = "base"
    vector_form = (
        mnemonic.startswith("v")
        or "v" in source_tokens
        or any(token.startswith("zv") for token in source_tokens)
    )
    if vector_form:
        # VLEN/ELEN are harness configuration, not instruction operands.  Keep
        # them in realization identity so a vector case cannot be mistaken for
        # a scalar case with the same bytes.  The observer may later replace
        # this bounded profile with a measured target configuration.
        environment_profile = "vector-vlen256-elen64"
    elif privilege_class == "system":
        if mnemonic == "mret" or source_tokens & {"smrnmi"}:
            environment_profile = "machine"
        elif source_tokens & {"h", "svinval_h"}:
            environment_profile = "hypervisor"
        elif (
            mnemonic in {"sret", "sfence_vma"}
            or source_tokens & {"s", "svinval", "ssctr"}
        ):
            environment_profile = "supervisor"
        elif source_tokens & {"sdext"}:
            environment_profile = "debug"
        else:
            environment_profile = "privileged"
    realizations: list[SemanticRealization] = []
    for xlen in xlens:
        for required in required_sets:
            isa_profile = isa_profile_for_extensions(f"rv{xlen}", required)
            realizations.append(
                SemanticRealization(
                    xlen=xlen,
                    extension_tokens=tuple(sorted(enabled_extensions(isa_profile))),
                    register_carrier=register_carrier,
                    privilege_class=privilege_class,
                    isa_profile=isa_profile,
                    environment_profile=environment_profile,
                )
            )
    compatibility_profiles = {
        profile
        for profile in (
            rv32_xregister_compatibility_profile(form),
            rv32_xregister_pair_compatibility_profile(form),
            rv32_xregister_writeback_compatibility_profile(form),
        )
        if isinstance(profile, str) and profile
    }
    if 32 in xlens:
        for isa_profile in sorted(compatibility_profiles):
            realizations.append(
                SemanticRealization(
                    xlen=32,
                    extension_tokens=tuple(sorted(enabled_extensions(isa_profile))),
                    register_carrier="gpr",
                    privilege_class=privilege_class,
                    isa_profile=isa_profile,
                    environment_profile=environment_profile,
                )
            )
        if "zcmp" in source_tokens and {"c_sreg1", "c_sreg2"} <= set(operand_groups(form)):
            isa_profile = isa_profile_for_extensions("rv32", frozenset({"e", "c", "zcmp"}))
            realizations.append(
                SemanticRealization(
                    xlen=32,
                    extension_tokens=tuple(sorted(enabled_extensions(isa_profile))),
                    register_carrier="gpr",
                    privilege_class=privilege_class,
                    isa_profile=isa_profile,
                    environment_profile=environment_profile,
                )
            )
    deduplicated = {
        (
            realization.xlen,
            realization.register_carrier,
            realization.privilege_class,
            realization.isa_profile,
            realization.environment_profile,
        ): realization
        for realization in realizations
    }
    return tuple(
        deduplicated[key]
        for key in sorted(
            deduplicated,
            key=lambda item: (
                int(item[0]),
                0 if item[1] == "gpr" else 1,
                str(item[2]),
                str(item[3]),
            ),
        )
    )


class GenerationSpecView:
    """A read-only view of a catalog form in the shape RVEMI expects."""

    __slots__ = ("form",)

    def __init__(self, form) -> None:
        self.form = form

    @property
    def mnemonic(self) -> str:
        return self.form.mnemonic

    @property
    def mask(self) -> int:
        return int(self.form.mask)

    @property
    def match(self) -> int:
        return int(self.form.match)

    @property
    def encoding_format(self) -> str:
        return "c" if self.form.encoding_length_bytes == 2 else "i"

    @property
    def xlen(self) -> str:
        return str(getattr(self.form, "xlen", "shared"))

    @property
    def operand_kinds(self) -> tuple[str, ...]:
        return tuple(operand_kind(self.form, name) for name in operand_groups(self.form))

    @property
    def operand_domains(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (role, register_domain(self.form, role))
            for role in self.operand_kinds
            if role in _REGISTER_OPERAND_ROLES
        )

def instruction_spec_for_generation(mnemonic: str) -> GenerationSpecView | None:
    alias = {"zext.b": "c.zext.b"}.get(str(mnemonic).lower().replace("_", "."), mnemonic)
    form = form_for_mnemonic(mnemonic) or form_for_mnemonic(alias)
    return None if form is None else GenerationSpecView(form)


def encode_instruction_for_generation(mnemonic: str, **operands) -> int:
    alias = {"zext.b": "c.zext.b"}.get(
        str(mnemonic).lower().replace("_", "."),
        mnemonic,
    )
    form = form_for_mnemonic(mnemonic) or form_for_mnemonic(alias)
    if form is not None and "immediate" in operands and "imm20" in operand_groups(form):
        operands = dict(operands)
        operands["immediate"] = encoded_immediate_for_metadata(
            form,
            operands["immediate"],
        )
    return encode_instruction(alias, **operands)


def csr_state_access_for_generation(
    mnemonic: str,
    *,
    rd: int | None,
    rs1: int | None,
) -> tuple[bool, bool] | None:
    """Return whether one CSR instruction reads and writes its CSR state."""
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return None
    if form.encoding_length_bytes != 4 or (form.match & 0x7F) != SYSTEM_OPCODE:
        return None
    funct3 = (form.match >> 12) & 0x7
    if funct3 in {0, 4}:
        return None
    source_nonzero = int(rs1 or 0) != 0
    if funct3 in {1, 5}:  # CSRRW/CSRRWI: rd=x0 suppresses the CSR read.
        return (int(rd or 0) != 0, True)
    if funct3 in {2, 3, 6, 7}:  # CSRRS/CSRRC (+ immediate): x0/uimm=0 suppresses write.
        return (True, source_nonzero)
    return None


def is_control_semantic_mnemonic_for_generation(mnemonic: str) -> bool:
    return instruction_effect_class(mnemonic) in {"branch", "indirect-jump", "direct-jump"}


@dataclass(frozen=True)
class EffectViewSibling:
    """一个由编码结构导出的、可能在固定输入下等价的同长度 sibling。

    这不是 RVEMI 的规则表：它只说明 candidate 改了哪一种语义 view；具体输入分区、
    profile proof 和 native gate 仍由 RVEMI 负责。新增同结构指令只要 catalog 给出 form，
    即可复用该事实。
    """

    mnemonic: str
    kind: str
    swap_source_operands: bool = False


@dataclass(frozen=True)
class LiteralIdentitySibling:
    """同长度 literal/register form 的固定 identity-law sibling。

    这类边不是“同 funct 字段的普通 OP-IMM ⟷ OP sibling”，而是只有当 literal 落在
    一个很小的 identity 集合里时才成立的表示互换，例如 ``addi rs1, 0`` 与
    ``or/xor/sub rs1, x0``。规则层仍需逐 occurrence 证明 carrier 的实际投影值满足
    该 literal 义务。
    """

    mnemonic: str
    allowed_literals: tuple[int, ...]


_SEMANTIC_OPERATION_BY_MNEMONIC = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    **dict.fromkeys(("pack", "packh", "packw"), "pack"),
    **dict.fromkeys(("slt", "slti"), "set-less-than"),
    **dict.fromkeys(("sltu", "sltiu"), "set-less-than-unsigned"),
}
_SOURCE_WIDTH_BY_SUFFIX = {"b": 8, "h": 16, "w": 32, "wu": 32, "l": 64, "lu": 64}
def _mnemonic_suffix_width(mnemonic: str) -> int | None:
    suffix = str(mnemonic).rsplit("_", 1)[-1]
    width = _SOURCE_WIDTH_BY_SUFFIX.get(suffix)
    return None if width is None else int(width)


def operand_permutation_pairs_for_generation(mnemonic: str) -> tuple[tuple[str, str], ...]:
    """返回同一指令内可交换的 source pair。

    这是编码/架构语义事实，不是 RVEMI 的规则表。算术交换与 ``BEQ/BNE`` 的 predicate
    symmetry 走同一描述；规则层只消费 pair，不为 branch 维护单独 rewrite。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes != 4:
        return ()
    major = form.match & 0x7F
    funct3 = (form.match >> 12) & 0x7
    funct7 = (form.match >> 25) & 0x7F
    if major in {0x33, 0x3B} and (
        funct7 == 0 and funct3 in {0x0, 0x4, 0x6, 0x7}
        or funct7 == 0x05 and funct3 in {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7}
        or funct7 == 0x20 and funct3 == 0x4
    ):
        return (("rs1", "rs2"),)
    if major in {0x33, 0x3B} and funct7 == 1 and funct3 in {0x0, 0x1, 0x3}:
        return (("rs1", "rs2"),) if funct3 != 0x2 else ()
    if major == 0x53:
        return (("rs1", "rs2"),) if (form.match >> 27) & 0x1F in {0x00, 0x02} else ()
    if major == 0x63 and funct3 in {0x0, 0x1}:
        return (("rs1", "rs2"),)
    return ()


# OP-IMM ⟷ OP、OP-IMM-32 ⟷ OP-32：立即数形式与寄存器形式互为等价编码，两个方向
# 都是同一个结构关系，不需要各写一张表。
_SIBLING_MAJOR_OPCODE = {0x13: 0x33, 0x1B: 0x3B, 0x33: 0x13, 0x3B: 0x1B}


def sibling_forms_for_generation(mnemonic: str) -> tuple[str, ...]:
    """产生同一状态转换的等价编码形式，按结构匹配，双向。

    立即数形式与寄存器形式共享 funct3 与 funct7，只有 major opcode 不同。
    立即数形式可以改写成持有该值的寄存器形式；寄存器形式在其 rs2 值落进目标立即数域
    时，也可以折回立即数形式。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes != 4:
        return ()
    sibling_major = _SIBLING_MAJOR_OPCODE.get(form.match & 0x7F)
    if sibling_major is None:
        return ()
    funct3 = (form.match >> 12) & 0x7
    funct7 = (form.match >> 25) & 0x7F
    source_roles = tuple(GenerationSpecView(form).operand_kinds)
    out = []
    for candidate in OFFICIAL_SELECTED_SCALAR_FORMS:
        if candidate.encoding_length_bytes != 4:
            continue
        if candidate.match & 0x7F != sibling_major:
            continue
        if (candidate.match >> 12) & 0x7 != funct3:
            continue
        if (candidate.match >> 25) & 0x7F != funct7:
            continue
        target_roles = tuple(GenerationSpecView(candidate).operand_kinds)
        source_only = tuple(role for role in source_roles if role not in target_roles)
        target_only = tuple(role for role in target_roles if role not in source_roles)
        if not (
            source_only == ("imm",)
            and len(target_only) == 1
            and target_only[0] in {"rs1", "rs2", "rs3"}
            and register_domain(candidate, target_only[0]) == "gpr"
            or target_only == ("imm",)
            and len(source_only) == 1
            and source_only[0] in {"rs1", "rs2", "rs3"}
            and register_domain(form, source_only[0]) == "gpr"
        ):
            continue
        out.append(candidate.mnemonic)
    return tuple(out)


def literal_identity_siblings_for_generation(mnemonic: str) -> tuple[LiteralIdentitySibling, ...]:
    """返回仅在固定 identity literal 下成立的 same-length form sibling。"""
    return {
        "addi": (
            LiteralIdentitySibling("or", (0,)),
            LiteralIdentitySibling("xor", (0,)),
            LiteralIdentitySibling("sub", (0,)),
        ),
        "addiw": (
            LiteralIdentitySibling("subw", (0,)),
        ),
        "or": (
            LiteralIdentitySibling("addi", (0,)),
        ),
        "xor": (
            LiteralIdentitySibling("addi", (0,)),
        ),
        "sub": (
            LiteralIdentitySibling("addi", (0,)),
        ),
        "subw": (
            LiteralIdentitySibling("addiw", (0,)),
        ),
        "sraiw": (
            LiteralIdentitySibling("roriw", (0,)),
        ),
        "roriw": (
            LiteralIdentitySibling("sraiw", (0,)),
        ),
    }.get(mnemonic, ())


def effect_view_siblings_for_generation(mnemonic: str) -> tuple[EffectViewSibling, ...]:
    """返回由 opcode/funct 字段表达的 fixed-input effect-view sibling。

    只列出两个形式具有相同长度、相同 operand layout 的结构关系：branch predicate
    dual、load sign/zero extension、整数 signed/unsigned compare、逻辑/算术右移，以及
    少数可由位级条件闭合的 full/W 或 M-extension signedness view。
    不在这里宣称它们已等价；rules.py 必须逐 occurrence 验充分条件。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes not in {2, 4}:
        return ()
    siblings = [
        EffectViewSibling(target, "load-extension-view")
        for target in _load_extension_siblings_for_generation(form)
    ]
    if form.encoding_length_bytes != 4:
        return tuple(siblings)
    major = int(form.match) & 0x7F
    funct3 = (int(form.match) >> 12) & 0x7
    funct7 = (int(form.match) >> 25) & 0x7F
    operation = _semantic_operation(form)

    def add(
        kind: str,
        *,
        target_major: int | None = None,
        target_funct3: int | None = None,
        target_funct7: int | None = None,
        swap: bool = False,
    ) -> None:
        target = _same_layout_view_sibling(
            form,
            major=target_major,
            funct3=target_funct3,
            funct7=target_funct7,
        )
        if target is not None:
            siblings.append(EffectViewSibling(target, kind, swap))

    if major == 0x63:
        dual = {0x4: 0x5, 0x5: 0x4, 0x6: 0x7, 0x7: 0x6}.get(funct3)
        if dual is not None:
            add("branch-predicate-dual", target_funct3=dual, swap=True)
        signedness = {0x4: 0x6, 0x6: 0x4, 0x5: 0x7, 0x7: 0x5}.get(funct3)
        if signedness is not None:
            add("signedness-compare", target_funct3=signedness)
    elif major in {0x13, 0x1B, 0x33, 0x3B}:
        upper = (int(form.match) >> 20) & 0xFFF
        if funct3 == 0x1 and upper in {0x601, 0x602}:
            add(
                "unary-full-w-ctz" if upper == 0x601 else "unary-full-w-cpop",
                target_major=0x1B if major == 0x13 else 0x13,
            )
        if operation in {"set-less-than", "set-less-than-unsigned"}:
            signedness = {0x2: 0x3, 0x3: 0x2}.get(funct3)
            if signedness is not None:
                add("signedness-compare", target_funct3=signedness)
        if major in {0x13, 0x1B, 0x33, 0x3B} and funct3 == 0x5 and funct7 in {0x00, 0x20}:
            add("signedness-shift", target_funct7=0x20 if funct7 == 0x00 else 0x00)
        if major in {0x33, 0x3B} and funct7 == 0x01:
            for target_funct3 in {0x1: (0x2, 0x3), 0x2: (0x1, 0x3), 0x3: (0x1, 0x2)}.get(funct3, ()):
                add("high-multiply-signedness", target_funct3=target_funct3)
            target_funct3 = {0x4: 0x5, 0x5: 0x4, 0x6: 0x7, 0x7: 0x6}.get(funct3)
            if target_funct3 is not None:
                add("division-remainder-signedness", target_funct3=target_funct3)
        if major == 0x33 and operation in {"add", "sub", "mul"}:
            add("full-to-w-result", target_major=0x3B)
    return tuple(siblings)


def _load_extension_siblings_for_generation(form) -> tuple[str, ...]:
    """返回同长度、同布局、同宽度且 signedness 相反的整数 load form。"""
    mnemonic = str(form.mnemonic).replace("_", ".")
    width = load_width_bytes(mnemonic)
    signedness = _signedness_from_encoding(mnemonic)
    if width is None or signedness not in {"signed", "unsigned"}:
        return ()
    source = GenerationSpecView(form)
    return tuple(
        str(candidate.mnemonic).replace("_", ".")
        for candidate in OFFICIAL_SELECTED_SCALAR_FORMS
        if candidate.mnemonic != form.mnemonic
        and candidate.encoding_length_bytes == form.encoding_length_bytes
        and candidate.mask == form.mask
        and GenerationSpecView(candidate).operand_kinds == source.operand_kinds
        and GenerationSpecView(candidate).operand_domains == source.operand_domains
        and load_width_bytes(str(candidate.mnemonic).replace("_", ".")) == width
        and {signedness, _signedness_from_encoding(str(candidate.mnemonic).replace("_", "."))}
        == {"signed", "unsigned"}
        and ((int(form.match) ^ int(candidate.match)) & ~int(form.mask)) == 0
    )


def _semantic_operation(form) -> str:
    """Return the neutral operation fact needed by projection/sibling helpers."""
    mnemonic = str(getattr(form, "mnemonic", ""))
    operation = _SEMANTIC_OPERATION_BY_MNEMONIC.get(mnemonic)
    if operation is not None:
        return operation
    if mnemonic.startswith("fcvt_") and _mnemonic_suffix_width(mnemonic) in {32, 64}:
        return "int-to-fp"
    if mnemonic.startswith(("sext_", "c_sext_", "psext_")) and _mnemonic_suffix_width(mnemonic) in {8, 16, 32}:
        return "sign-extend"
    if mnemonic.startswith(("zext_", "c_zext_")) and _mnemonic_suffix_width(mnemonic) in {8, 16, 32}:
        return "zero-extend"
    return ""


def _same_layout_view_sibling(
    form,
    *,
    major: int | None = None,
    funct3: int | None = None,
    funct7: int | None = None,
) -> str | None:
    """在冻结 catalog 中查找只替换指定 funct 字段的同布局 form。"""
    changed_mask = 0
    expected = int(form.match)
    if major is not None:
        changed_mask |= 0x7F
        expected = (expected & ~0x7F) | (int(major) & 0x7F)
    if funct3 is not None:
        changed_mask |= 0x7 << 12
        expected = (expected & ~(0x7 << 12)) | ((int(funct3) & 0x7) << 12)
    if funct7 is not None:
        changed_mask |= 0x7F << 25
        expected = (expected & ~(0x7F << 25)) | ((int(funct7) & 0x7F) << 25)
    if changed_mask == 0:
        return None
    for candidate in OFFICIAL_SELECTED_SCALAR_FORMS:
        if candidate.encoding_length_bytes != form.encoding_length_bytes:
            continue
        if int(candidate.mask) != int(form.mask):
            continue
        if (int(candidate.match) & ~changed_mask) != (int(form.match) & ~changed_mask):
            continue
        if (int(candidate.match) & changed_mask) != (expected & changed_mask):
            continue
        return str(candidate.mnemonic)
    return None


def control_no_effect_target_for_generation(mnemonic: str) -> str | None:
    """返回可在 fallthrough-equivalent 分区物化的同宽 control NOP target。"""
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return None
    if instruction_effect_class(mnemonic) not in {"branch", "direct-jump"}:
        return None
    if form.encoding_length_bytes == 2:
        return "c.nop"
    if form.encoding_length_bytes == 4:
        return "addi"
    return None


def direct_jump_target_for_branch(mnemonic: str) -> str | None:
    """返回条件分支在 taken-equivalent 分区可物化的同宽直接跳转 target。

    由编码形状（长度/effect-class）推导，不依赖助记符名单：

    - 2 字节 branch -> c.j（压缩直接跳转，无 rd 槽）
    - 4 字节 branch -> jal（带 rd 槽，link 语义）
    """
    form = form_for_mnemonic(mnemonic)
    if form is None or instruction_effect_class(mnemonic) != "branch":
        return None
    if form.encoding_length_bytes == 2:
        return "c.j"
    if form.encoding_length_bytes == 4:
        return "jal"
    return None


def high_multiply_signed_source_operands(mnemonic: str) -> frozenset[str] | None:
    """M 扩展高位乘法的 signed 输入源操作数集合，None 表示不是 high-multiply 形式。"""
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return None
    match = int(form.match)
    if (match & 0x7F) != 0x33 or ((match >> 25) & 0x7F) != 0x01:
        return None
    return {
        0x1: frozenset({"rs1", "rs2"}),
        0x2: frozenset({"rs1"}),
        0x3: frozenset(),
    }.get((match >> 12) & 0x7)


def gpr_result_width_bits(mnemonic: str, xlen: int | None = None) -> int | None:
    """RV64 下该 GPR 形式的结果视图宽度：W 形式为 32，其余为 64。

    32 位立即数/寄存器形式由 major opcode 的 ``0x1B/0x3B`` 编码字段决定；
    RV64 的压缩 C.ADDIW/C.ADDW/C.SUBW 也产生 32 位结果；
    Zba 的 ``*.UW`` 与 Zbb 的 ``ZEXT.H`` 虽落在 OP-32，也产生 XLEN 结果；
    RV32 下所有 GPR 视图都是 32，那是 profile 事实，由规则层结合 isa_profile 判定。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return None
    normalized = str(form.mnemonic).replace("_", ".")
    if xlen == 32:
        return 32
    if normalized.endswith(".uw") or normalized == "zext.h":
        return 64
    if _is_w_result_form(form):
        return 32
    if form.encoding_length_bytes == 4 and (form.match & 0x7F) in {0x1B, 0x3B}:
        return 32
    return 64


def zero_immediate_targets_for_generation(mnemonic: str) -> tuple[str, ...]:
    """Return the GPR identity target for an encoding with immediate zero."""
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes != 4:
        return ()
    view = GenerationSpecView(form)
    if view.operand_kinds != ("rd", "rs1", "imm"):
        return ()
    match = int(form.match)
    major = match & 0x7F
    if major not in {0x13, 0x1B} or (match >> 12) & 0x7 not in {1, 5}:
        return ()
    if (match >> 20) & 0xFFF not in {0, 0x400, 0x600}:
        return ()
    return ("addi",) if major == 0x13 else ("addiw",)


def full_w_view_guard(kind: str) -> str | None:
    """unary-full-w 视图的证明义务类别，由 sibling 的 kind 推导。"""
    return {
        "unary-full-w-ctz": "low32-nonzero",
        "unary-full-w-cpop": "high32-zero",
    }.get(kind)


_FMA_MAJOR_OPCODES = frozenset({0x43, 0x47, 0x4B, 0x4F})


def fp_form_family_siblings(mnemonic: str) -> tuple[str, ...]:
    """同一 FP 形式族里的其它形式：编码形状与格式都相同，只有选择具体运算的字段不同。

    两族由编码结构自然划出，不需要列助记符：

    - MADD/MSUB/NMSUB/NMADD 共用一套操作数，只有 major opcode 的两个 bit 不同，
      它们分别表示「取负乘积」与「取负加数」；
    - OP-FP 里 funct5 相同、funct3 不同的形式（fsgnj/fsgnjn/fsgnjx、fmin/fmax）。

    ``mask`` 相同保证两个形式的变长字段布局逐位一致，因此源指令的操作数位可以整体搬到
    目标形式上；``fmt`` 相同把 ``.S ↔ .D`` 挡在外面——那两个形式消费的源位宽不同，
    同一组寄存器并不表示同一组值。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes != 4:
        return ()
    opcode = form.match & 0x7F
    fmt = (form.match >> 25) & 0x3
    if opcode in _FMA_MAJOR_OPCODES:
        def same_family(candidate) -> bool:
            return (candidate.match & 0x7F) in _FMA_MAJOR_OPCODES
    elif opcode == 0x53:
        funct5 = (form.match >> 27) & 0x1F

        def same_family(candidate) -> bool:
            return (candidate.match & 0x7F) == 0x53 and (candidate.match >> 27) & 0x1F == funct5
    else:
        return ()
    return tuple(
        candidate.mnemonic
        for candidate in OFFICIAL_SELECTED_SCALAR_FORMS
        if candidate.encoding_length_bytes == 4
        and candidate.mask == form.mask
        and candidate.match != form.match
        and (candidate.match >> 25) & 0x3 == fmt
        and same_family(candidate)
    )


def fma_sign_selectors(mnemonic: str) -> tuple[bool, bool] | None:
    """FMA 形式的 (取负乘积, 取负加数)。非 FMA 形式返回 None。

    官方把这两位放在 major opcode 的 bit 3 与 bit 2 上，所以不需要按助记符登记。
    """
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return None
    opcode = form.match & 0x7F
    if opcode not in _FMA_MAJOR_OPCODES:
        return None
    selector = (opcode >> 2) & 0x3
    return bool(selector & 0x2), bool(selector & 0x1)


# R8 的等价条件：一族形式互换在什么位级前提下产生同一状态转换。
# 名字由规则层按 profile 里的 rawbits 求值，这里只负责从编码推出「该验哪一条」。
FP_ZERO_PRODUCT = "zero-product"
FP_ZERO_ADDEND = "zero-addend"
FP_POSITIVE_RS1 = "positive-rs1"
FP_NEGATIVE_RS1 = "negative-rs1"
FP_EQUAL_SOURCES = "equal-sources"

_SGNJ_FUNCT5 = 0b00100
_MINMAX_FUNCT5 = 0b00101


def fp_family_equivalence_condition(source_mnemonic: str, target_mnemonic: str) -> str | None:
    """两个同族 FP 形式在什么条件下产生同一状态转换。``None`` 表示不构成等价关系。

    条件都是位级的，可以直接在 profile 的 rawbits 上验，不需要 FP 语义模拟器：

    - FMA 只差乘积符号 → 乘积为零（任一乘数为零）；只差加数符号 → 加数为零；
      两者都差时 signed-zero、NaN payload 和 flag 需要更强的语义证明，当前不生成；
    - ``fsgnj``/``fsgnjx`` 的符号来源在 ``rs1`` 为正时一致，``fsgnjn``/``fsgnjx`` 在为负时一致；
    - ``fmin``/``fmax`` 在两个源持有同一个值时一致。
    """
    source_signs = fma_sign_selectors(source_mnemonic)
    target_signs = fma_sign_selectors(target_mnemonic)
    if source_signs is not None and target_signs is not None:
        product_differs = source_signs[0] != target_signs[0]
        addend_differs = source_signs[1] != target_signs[1]
        if product_differs and addend_differs:
            # ``p + a`` 与 ``-p - a`` 即使都为数值零也可能在 IEEE signed zero、
            # NaN payload 或 flag 上不同；当前 one-window guard 没有小而通用的充分条件。
            return None
        if product_differs:
            return FP_ZERO_PRODUCT
        if addend_differs:
            return FP_ZERO_ADDEND
        return None
    source = form_for_mnemonic(source_mnemonic)
    target = form_for_mnemonic(target_mnemonic)
    if source is None or target is None:
        return None
    if (source.match & 0x7F) != 0x53 or (target.match & 0x7F) != 0x53:
        return None
    funct5 = (source.match >> 27) & 0x1F
    pair = {(source.match >> 12) & 0x7, (target.match >> 12) & 0x7}
    if funct5 == _SGNJ_FUNCT5:
        if pair == {0, 2}:
            return FP_POSITIVE_RS1
        if pair == {1, 2}:
            return FP_NEGATIVE_RS1
        return None
    if funct5 == _MINMAX_FUNCT5 and pair == {0, 1}:
        return FP_EQUAL_SOURCES
    return None


def fp_operand_width_bits(mnemonic: str) -> int | None:
    """FP 形式的数据位宽。当前 RVEMI profile 能精确表示 H/S/D，Q 返回 ``None``。"""
    form = form_for_mnemonic(mnemonic)
    if form is None or form.encoding_length_bytes != 4:
        return None
    if (form.match & 0x7F) not in {*_FMA_MAJOR_OPCODES, 0x53}:
        return None
    return {0x0: 32, 0x1: 64, 0x2: 16}.get((form.match >> 25) & 0x3)


def _fp_source_width_bits(form) -> int | None:
    """返回 FPR source 的编码宽度；Q 退化为不可区分的 XLEN 视图。"""
    if form.encoding_length_bytes != 4:
        return None
    match = int(form.match)
    if (match & 0x7F) not in {*_FMA_MAJOR_OPCODES, 0x53}:
        return None
    code = (match >> 20) & 0x1F if (match >> 27) & 0x1F == 0x08 else (match >> 25) & 0x3
    return {0: 32, 1: 64, 2: 16, 3: 64, 6: 16, 8: 32}.get(code)


FPR16_BOXED_OR_CANONICAL_NAN = "fpr16-boxed-or-canonical-nan"
FPR32_BOXED_OR_CANONICAL_NAN = "fpr32-boxed-or-canonical-nan"
FPR16_RAW = "fpr16-raw"
FPR32_RAW = "fpr32-raw"


def _register_index_width(form) -> int | None:
    """返回 R 型 shift/rotate/bit-index 的 rs2 实际消费宽度。

    这些形式共用 OP/OP-32 的 register index 编码；M、CLMUL、min/max、czero 等虽落在
    相同 opcode/funct3 槽位，却由 funct7 排除。新增同一编码族的 index form 只需补该
    encoding fact，不涉及 R2。
    """
    if form.encoding_length_bytes != 4 or (form.match & 0x7F) not in {0x33, 0x3B}:
        return None
    funct3 = (form.match >> 12) & 0x7
    funct7 = (form.match >> 25) & 0x7F
    if funct3 not in {0x1, 0x5} or funct7 not in {0x00, 0x14, 0x20, 0x24, 0x30, 0x34}:
        return None
    return 5 if (form.match & 0x7F) == 0x3B else 6


def _gpr_source_projection(
    mnemonic: str,
    form,
    role: str,
    index_width: int | None,
    xlen: int | None = None,
) -> str:
    major = form.match & 0x7F if form.encoding_length_bytes == 4 else None
    operation = _semantic_operation(form)
    if operation == "int-to-fp" and role == "rs1":
        width = _semantic_source_width(form, 32)
        return f"low{width}" if width < 64 else "xlen"
    if operation == "sign-extend" and role in {"rd", "rs1"}:
        width = _semantic_source_width(form, 0)
        if width in {8, 16, 32}:
            return f"low{width}"
    if operation == "zero-extend" and role in {"rd", "rs1"}:
        width = _semantic_source_width(form, 0)
        if width in {8, 16, 32}:
            return f"low{width}"
    if (role in {"rd", "rs2"}
            and any(str(field).startswith("rd_rs1_")
                    for field in getattr(form, "variable_fields", ()))
            and str(getattr(form, "mnemonic", "")).rsplit("_", 1)[-1].endswith("w")):
        return "low32"
    if operation == "pack" and role in {"rs1", "rs2"}:
        if mnemonic == "packh":
            return "low8"
        if mnemonic == "packw":
            return "low16"
        return {32: "low16", 64: "low32"}.get(xlen, "xlen")
    if role == "rs2" and index_width is not None:
        return f"low{5 if xlen == 32 and index_width == 6 else index_width}"
    if mnemonic.endswith(".uw"):
        return "low32" if role == "rs1" else "xlen"
    if major in {0x1B, 0x3B}:
        return "low32"
    return "xlen"


def _semantic_source_width(form, default: int) -> int:
    width = _mnemonic_suffix_width(str(getattr(form, "mnemonic", "")))
    return default if width is None else int(width)


def source_projections_for_mnemonic(
    mnemonic: str, xlen: int | None = None,
) -> tuple[tuple[str, str], ...]:
    """返回每个语义 source 的实际 consumed projection。

    返回完整 source 集合而不是只报告非默认投影，缺槽由调用者 fail closed。投影是逐
    operand 事实：例如 ``SLLW`` 的 rs1 是 ``low32``，rs2 是 ``low5``；``SLL`` 的
    rs2 是 ``low6``；H/S FP 再区分 raw-bit consumer 与 NaN-box consumer。
    """
    if type(mnemonic) is not str:
        return ()
    mnemonic = mnemonic.lower().replace("_", ".")
    form = form_for_mnemonic(mnemonic)
    if form is None:
        return ()
    if (
        form.encoding_length_bytes == 4
        and (form.match & 0x7F) == 0x53
        and ((form.match >> 27) & 0x1F) == 0x1E
        and ((form.match >> 20) & 0x1F) == 0x1
    ):
        return ()
    major = form.match & 0x7F if form.encoding_length_bytes == 4 else None
    fp_width = fp_operand_width_bits(mnemonic)
    fp_source_width = _fp_source_width_bits(form)
    index_width = _register_index_width(form)
    implicit_rd_source = any(
        str(field).startswith("rd_rs1_")
        for field in getattr(form, "variable_fields", ())
    )
    operand_rows = list(GenerationSpecView(form).operand_domains)
    rows = []
    for role, domain in operand_rows:
        if role == "rd" and not implicit_rd_source:
            continue
        if role not in {"rd", "rs1", "rs2", "rs3"}:
            continue
        projection = "xlen"
        if (
            domain == "gpr"
            and major == 0x53
            and fp_width is not None
            and ((form.match >> 27) & 0x1F) == 0x1E
            and ((form.match >> 12) & 0x7) == 0
        ):
            projection = f"low{fp_width}" if fp_width < 64 else "xlen"
        elif domain == "gpr":
            projection = _gpr_source_projection(mnemonic, form, role, index_width, xlen)
        elif domain == "fpr" and fp_width is not None:
            width = fp_source_width if fp_source_width is not None else fp_width
            raw_bits = str(form.mnemonic).replace("_", ".").startswith("fmv.x.")
            projection = {
                16: FPR16_RAW if raw_bits else FPR16_BOXED_OR_CANONICAL_NAN,
                32: FPR32_RAW if raw_bits else FPR32_BOXED_OR_CANONICAL_NAN,
            }.get(width, "xlen")
        rows.append((role, projection))
    roles = {role for role, _projection in rows}
    rows.extend((role, "xlen") for role in implicit_source_roles(form) if role not in roles)
    return tuple(rows)

# --------------------------------------------------------------------------
# 助记符分类事实改为从 pinned catalog 的编码位段派生，不再维护按助记符的
# 手写分类表。以下每类事实都能从 form 的 (match, mask, xlen) 直接读出：
#   * 32 位整数 load/store 的 funct3 低两位是宽度的 log2；
#   * 0x07/0x27 的 OP-FP load/store 用 funct3 作格式选择器（含向量元素宽）；
#   * LR.W/LR.D 是 0x2F 主 opcode 下 funct5=00010 的唯一形式，宽度看 funct3；
#   * 压缩 load/store 由 (quadrant, funct3) 决定；Zcb 的 C.LBU/C.LH/C.LHU 与
#     C.SB/C.SH 共享 funct3=100，但 bit11（load/store）与 bit10（字节/半字）
#     是固定编码位，逐 form 可区分（旧注释认为不可表达，已由 match/mask 证伪）；
#   * signedness 落在 funct3/funct7 位段：整数 load 的 bit2、branch 的 bit1、
#     SLT/SLTU 与 M 家族（MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU 及 W 版本）；
#   * W 形式（结果 32 位）由 0x1B/0x3B 主 opcode 加 funct3/funct7 组合判定，
#     Zba 的 ADD.UW/SLLI.UW 与 ZEXT.H（funct7=0x04）因 funct7 不同不落入；
#   * shamt 语义的移位由 funct3/funct7/主 opcode 判定：RORW 与 CLZW 共享
#     funct3=1/funct7=0x30，但主 opcode 0x3B/0x1B 区分二者。
# --------------------------------------------------------------------------

_FP_FORMAT_WIDTH = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 2, 6: 4, 7: 8}


def _is_lr_encoding(match: int) -> bool:
    """0x2F 主 opcode 下 funct5=00010 是 LR.W/LR.D 的唯一编码。"""
    return (match & 0x7F) == 0x2F and ((match >> 27) & 0x1F) == 0x02


def _is_zcb_form(match: int, bit11: int) -> bool:
    """Zcb 压缩整数 memory form：funct3=100 且 bit11 固定。"""
    return (
        (match & 0x3) == 0
        and ((match >> 13) & 0x7) == 0x4
        and ((match >> 11) & 0x1) == bit11
    )


def load_width_bytes(mnemonic: str) -> int | None:
    """按 pinned catalog 编码位段推导 load 宽度。"""
    spec = instruction_spec_for_generation(mnemonic)
    if spec is None:
        return None
    match = int(spec.match)
    if spec.encoding_format != "c":
        funct3 = (match >> 12) & 0x7
        opcode = match & 0x7F
        if opcode == 0x03:
            # 整数 load：funct3 低两位是宽度的 log2。
            return 1 << (funct3 & 0x3)
        if opcode == 0x07:
            # OP-FP loads use funct3 as a format selector rather than the
            # integer-load log2 width encoding (FLH/FLW/FLD/FLQ).  Vector
            # loads occupy the remaining funct3 values and retain their
            # element-width mapping.
            return _FP_FORMAT_WIDTH.get(funct3)
        if _is_lr_encoding(match):
            # LR.W/LR.D 宽度是 funct3 的编码事实。
            return {0x2: 4, 0x3: 8}.get(funct3)
        return None
    funct3 = (match >> 13) & 0x7
    if (match & 0x3) in {0, 2}:
        if funct3 in {0x1, 0x2, 0x3}:
            if funct3 == 0x3 and spec.xlen == "rv32":
                # C.FLW/C.FLWSP share the C.LD funct3 slot on RV64, but
                # carry a 32-bit FP value on RV32.
                return 4
            return {0x1: 8, 0x2: 4, 0x3: 8}.get(funct3)
        if _is_zcb_form(match, 0):
            # C.LBU/C.LH/C.LHU：bit10 区分字节（0）与半字（1）。
            return 1 if ((match >> 10) & 0x1) == 0 else 2
    return None


def _signedness_from_encoding(mnemonic: str) -> str:
    """按 pinned catalog 编码位段推导整数符号规则。

    signedness 是 funct3/funct7 位段事实：整数 load 的 bit2、branch 的 bit1、
    SLT/SLTU 与 M 家族的 funct3、W 形式的 funct3/funct7 组合。不是语义名单。
    """
    spec = instruction_spec_for_generation(mnemonic)
    if spec is None:
        return "none"
    match = int(spec.match)
    if spec.encoding_format == "c":
        funct3 = (match >> 13) & 0x7
        if (match & 0x3) in {0, 2}:
            if funct3 == 0x2:
                return "signed"  # C.LW/C.LWSP
            if _is_zcb_form(match, 0):
                if ((match >> 10) & 0x1) == 0:
                    return "unsigned"  # C.LBU
                # C.LH 与 C.LHU 由固定 bit6 区分；C.LBU 的 bit6 是可变位移。
                if ((match >> 6) & 0x1) and ((int(spec.mask) >> 6) & 0x1):
                    return "signed"  # C.LH
                return "unsigned"  # C.LHU
        return "none"
    opcode = match & 0x7F
    funct3 = (match >> 12) & 0x7
    funct7 = (match >> 25) & 0x7F
    if opcode == 0x03:
        if funct3 == 0x3:
            return "none"  # LD：64 位 load 没有符号规则声明
        return "unsigned" if funct3 & 0x4 else "signed"
    if _is_lr_encoding(match):
        return "signed" if funct3 == 0x2 else "none"  # LR.W/LR.D
    if opcode == 0x63:
        if funct3 not in {0x4, 0x5, 0x6, 0x7}:
            # BEQ/BNE 与未评级立即数分支（BEQI/BNEI 占用 funct3=2/3）。
            return "none"
        return "unsigned" if funct3 & 0x2 else "signed"  # BLTU/BGEU vs BLT/BGE
    if opcode in {0x13, 0x33} and funct7 == 0x00 and funct3 in {0x2, 0x3}:
        return "signed" if funct3 == 0x2 else "unsigned"  # SLTI/SLT vs SLTIU/SLTU
    if (
        opcode in {0x33, 0x3B}
        and funct7 == 0x01
        and funct3 in {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7}
    ):
        # M 家族（MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU 与 W 版本）。
        if funct3 in {0x1, 0x4, 0x6}:
            return "signed"  # MULH / DIV / REM
        if funct3 == 0x2:
            return "mixed"  # MULHSU
        return "unsigned" if funct3 in {0x3, 0x5, 0x7} else "none"
    if (opcode in {0x1B, 0x3B} and spec.xlen == "rv64"
            and ((funct7 in {0x00, 0x20} and funct3 in {0x00, 0x01, 0x05}) or (
            funct7 == 0x30 and funct3 == 0x05
        ) or (funct7 == 0x30 and funct3 == 0x01 and (match & 0x7F) == 0x3B))):
            # ADDIW/ADDW/SUBW 与移位/旋转 W 形式。CLZW/CTZW/CPOPW/ABSW
            # （funct3=1/funct7=0x30/0x1B）不落入，保持无符号规则声明。
            return "signed"
    return "none"


def signedness_mode_for_generation(mnemonic: str) -> str:
    """Return the encoding-derived signedness mode, or ``none``."""
    return _signedness_from_encoding(mnemonic)


def _is_w_result_form(form) -> bool:
    """W 形式（结果 32 位）由编码字段判定。

    Zba 的 ADD.UW/SLLI.UW 与 ZBB 的 ZEXT.H 同样占用 0x1B/0x3B 但结果是 64 位
    （funct7=0x04），以及 RV32-only 的 p 扩展行（xlen 不是 rv64），都不落入；
    RV64 的 C.ADDIW/C.ADDW/C.SUBW 是压缩 W 形式。
    """
    if (
        getattr(form, "encoding_length_bytes", 0) == 2
        and str(getattr(form, "xlen", "shared")) == "rv64"
        and str(getattr(form, "mnemonic", "")).replace("_", ".")
        in {"c.addiw", "c.addw", "c.subw"}
    ):
        return True
    if (
        getattr(form, "encoding_length_bytes", 0) != 4
        or str(getattr(form, "xlen", "shared")) != "rv64"
    ):
        return False
    match = int(form.match)
    if (match & 0x7F) not in {0x1B, 0x3B}:
        return False
    funct3 = (match >> 12) & 0x7
    funct7 = (match >> 25) & 0x7F
    if funct7 in {0x00, 0x20} and funct3 in {0x00, 0x01, 0x05}:
        return True
    if funct7 == 0x30 and funct3 == 0x05:
        return True  # RORIW/RORW
    if funct7 == 0x30 and funct3 == 0x01 and (match & 0x7F) == 0x3B:
        return True  # ROLW（0x1B 的 CLZW/ABSW 家族不是 W 结果口径）
    if (
        funct7 == 0x04
        and funct3 == 0x04
        and (match & 0x7F) == 0x3B
        and str(getattr(form, "mnemonic", "")).replace("_", ".") == "packw"
    ):
        return True  # PACKW
    return funct7 == 0x01 and funct3 in {0x00, 0x04, 0x05, 0x06, 0x07}


# RISC-V 官方 opcode 到架构效果类别的映射。这是编码事实而不是助记符清单：
# 沿用标准 opcode 的新指令族不需要在这里增加条目。
_EFFECT_CLASS_BY_OPCODE = {
    **dict.fromkeys((0x03, 0x07), "load"),
    0x0F: "system",  # MISC-MEM：fence / fence.i / pause，内存序不可改写
    **dict.fromkeys((0x23, 0x27), "store"),
    0x2F: "atomic",
    **dict.fromkeys((0x43, 0x47, 0x4B, 0x4F, 0x53), "fp-op"),
    0x63: "branch",
    0x67: "indirect-jump",
    0x6F: "direct-jump",
    0x73: "system",
}
# 压缩编码的 (quadrant, funct3)。两处需要额外的编码判据，见下面的函数：
# quadrant 2 / funct3 4 按固定字段区分 C.EBREAK、C.JR/C.JALR 与 C.MV/C.ADD；
# quadrant 1 / funct3 1 在 RV32 是 C.JAL、在 RV64 是 C.ADDIW。
_COMPRESSED_EFFECT_CLASS = {
    **dict.fromkeys(((0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)), "load"),
    **dict.fromkeys(((0, 5), (0, 6), (0, 7), (2, 5), (2, 6), (2, 7)), "store"),
    (1, 5): "direct-jump",
    **dict.fromkeys(((1, 6), (1, 7)), "branch"),
}
_COMPRESSED_EXACT_EFFECT_CLASS = {
    **dict.fromkeys(("c.lbu", "c.lh", "c.lhu"), "load"),
    **dict.fromkeys(("c.sb", "c.sh"), "store"),
    "cm.jt": "indirect-jump", "cm.jalt": "indirect-jump",
    **dict.fromkeys(("cm.mva01s", "cm.mvsa01"), "op"),
    "cm.push": "store",
    "cm.pop": "load",
    **dict.fromkeys(("cm.popret", "cm.popretz"), "indirect-jump"),
}


def instruction_effect_class(mnemonic: str) -> str | None:
    """按官方编码字段判定指令的架构效果类别。

    这是 RVEMI 的控制流、原子、CSR 和 load 判据的唯一来源，取代按助记符维护的
    多张清单。返回 ``None`` 表示 catalog 不认识该助记符。
    """
    spec = instruction_spec_for_generation(mnemonic)
    if spec is None:
        return None
    exact = _COMPRESSED_EXACT_EFFECT_CLASS.get(
        mnemonic
    ) or _COMPRESSED_EXACT_EFFECT_CLASS.get(mnemonic.replace("_", "."))
    if exact is not None:
        return exact
    if (spec.match & 0x3) == 0x3:
        return _EFFECT_CLASS_BY_OPCODE.get(spec.match & 0x7F, "op")
    quadrant = spec.match & 0x3
    funct3 = (spec.match >> 13) & 0x7
    if quadrant == 2 and funct3 == 4:
        if ((spec.mask >> 7) & 0x1F) == 0x1F:
            return "system"
        return "indirect-jump" if ((spec.mask >> 2) & 0x1F) == 0x1F else "op"
    if quadrant == 1 and funct3 == 1:
        return "direct-jump" if spec.xlen == "rv32" else "op"
    return _COMPRESSED_EFFECT_CLASS.get((quadrant, funct3), "op")

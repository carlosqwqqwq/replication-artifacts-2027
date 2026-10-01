from dataclasses import dataclass
from functools import cache

# 中立层负责 catalog 查找、助记符命名归一与编码，RVEMI 不直接触碰 catalog 结构。
from ..riscv_encoding import operand_groups
from ..riscv_catalog import official_form
from ..spec_definedness import (
    EffectViewSibling,
    GenerationSpecView,
    LiteralIdentitySibling,
    encode_instruction_for_generation,
    effect_view_siblings_for_generation,
    instruction_effect_class,
    instruction_spec_for_generation,
    literal_identity_siblings_for_generation,
    operand_permutation_pairs_for_generation,
    semantic_realizations_for_form,
    sibling_forms_for_generation,
    source_projections_for_mnemonic,
    signedness_mode_for_generation,
    zero_immediate_targets_for_generation,
)

GPR_EFFECT = "gpr"
FPR_EFFECT = "fpr"
GPR_FROM_FPR_EFFECT = "gpr-from-fpr"
PRIVATE_LOAD_EFFECT = "private-load"
CONTROL_EFFECT = "control"
_DATA_OPERAND_ROLES = ("rd", "rs1", "rs2", "rs3", "imm")


@dataclass(frozen=True)
class InstructionSpec:
    shared_spec: GenerationSpecView
    effect_kind: str
    operand_permutation_pairs: tuple[tuple[str, str], ...]
    equivalent_targets: tuple[str, ...]
    literal_identity_siblings: tuple[LiteralIdentitySibling, ...]
    effect_view_siblings: tuple[EffectViewSibling, ...]
    requested_mnemonic: str = ""
    generation_mnemonic: str = ""
    xlen: int | None = None

    @property
    def mnemonic(self) -> str:
        return self.requested_mnemonic or str(self.shared_spec.mnemonic).replace("_", ".")

    @property
    def operand_roles(self) -> tuple[str, ...]:
        return tuple(role for role in _DATA_OPERAND_ROLES if role in self.shared_spec.operand_kinds)

    @property
    def source_projections(self) -> tuple[tuple[str, str], ...]:
        return source_projections_for_mnemonic(self.mnemonic)

    def source_projections_for_xlen(self, xlen: int) -> tuple[tuple[str, str], ...]:
        return source_projections_for_mnemonic(self.mnemonic, xlen)

    @property
    def signedness(self) -> str:
        return signedness_mode_for_generation(self.mnemonic)

    @property
    def zero_immediate_targets(self) -> tuple[str, ...]:
        return zero_immediate_targets_for_generation(self.mnemonic)

    @property
    def control_shape(self) -> str | None:
        return {
            "branch": "conditional",
            "direct-jump": "direct",
            "indirect-jump": "indirect",
        }.get(instruction_effect_class(self.mnemonic))

    def encode(
        self,
        rd: int | None,
        rs1: int | None,
        rs2: int | None,
        rs3: int | None,
        immediate: int | None,
    ) -> int:
        encoded = encode_instruction_for_generation(
            self.generation_mnemonic or self.mnemonic,
            rd=rd,
            rs1=rs1,
            rs2=rs2,
            rs3=rs3,
            immediate=immediate,
        )
        if self.xlen == 32 and type(immediate) is int and immediate >= 32:
            groups = operand_groups(self.shared_spec.form)
            if "shamtd" in groups or "c_imm6" in groups:
                raise ValueError(f"{self.mnemonic}: shift amount {immediate} is out of range for RV32")
        return encoded


def runtime_instruction_spec(mnemonic: str, xlen: int | None = None) -> InstructionSpec | None:
    if not isinstance(mnemonic, str) or not mnemonic:
        return None
    xlen = xlen if type(xlen) is int and xlen in (32, 64) else None
    return _runtime_instruction_spec(
        mnemonic.lower().replace("_", "."),
        xlen,
    )


@cache
def _runtime_instruction_spec(mnemonic: str, xlen: int | None = None) -> InstructionSpec | None:
    generation_mnemonic = "rev8.rv32" if mnemonic == "rev8" and xlen == 32 else mnemonic
    generation = instruction_spec_for_generation(generation_mnemonic)
    if generation is None:
        return None
    form = official_form(generation.form.mnemonic)
    if form is None:
        return None
    if xlen is not None and xlen not in {
        realization.xlen for realization in semantic_realizations_for_form(form)
    }:
        return None
    spec = GenerationSpecView(form)
    if not any(role in spec.operand_kinds for role in _DATA_OPERAND_ROLES):
        return None
    if mnemonic.startswith("v"):
        return None
    effect_class = instruction_effect_class(mnemonic)
    if effect_class in {"atomic", "system", "store"}:
        return None
    if effect_class == "load":
        effect_kind = PRIVATE_LOAD_EFFECT
    elif effect_class in {"direct-jump", "indirect-jump", "branch"}:
        effect_kind = CONTROL_EFFECT
    else:
        operand_domains = dict(spec.operand_domains)
        rd_domain = operand_domains.get("rd", "gpr")
        effect_kind = (
            FPR_EFFECT if rd_domain == "fpr"
            else GPR_FROM_FPR_EFFECT
            if rd_domain == "gpr" and any(
                operand_domains.get(role) == "fpr" for role in ("rs1", "rs2", "rs3")
            )
            else GPR_EFFECT
        )
    return InstructionSpec(
        spec,
        effect_kind,
        operand_permutation_pairs_for_generation(mnemonic),
        sibling_forms_for_generation(mnemonic),
        literal_identity_siblings_for_generation(mnemonic),
        effect_view_siblings_for_generation(mnemonic),
        mnemonic,
        generation_mnemonic,
        xlen,
    )

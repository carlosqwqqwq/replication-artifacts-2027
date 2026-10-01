"""可选的 RVGEN 本地架构语义模型。

它只消费冻结在 TestCase 中的指令元数据、初始 GPR 和内存；FPR、向量寄存器和
CSR 使用模型定义的初态并由指令更新。它不读取 Target 结果，也不计算模拟器源码
覆盖率。生产调用只在显式选择 rvgen-semantic reference，或单指令变异需要保守判定
expected-trap 时使用；它不是生成、MCMC/EMI 或 Target 执行的门禁。

形式/状态无法严格建模时抛出 SemanticUnsupported。reference adapter 将其记录为
reference-gap；变异分类将 trap 预期留为未知。两种情况都保留 case，供真实 reference
或 Target 执行。只有逐项核准的语义才可返回架构状态，不能静默填入猜测值。
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .boundaries import (
    STACK_ADJ_RV32,
    STACK_ADJ_RV64,
    STACK_RLIST_REGISTERS_RV64,
    branch_condition_holds,
    semantic_immediate_value,
    stack_slot_offsets,
)
from ..spec_definedness import (
    csr_required_extensions_for_profile,
    enabled_extensions,
    x_register_fp_extensions,
)
from ..direct_case import (
    p_form_uses_vxsat,
    testcase_uses_fp_instruction,
    testcase_uses_vector_instruction,
)
from ..riscv_catalog import OFFICIAL_ALL_CATALOG_BY_MNEMONIC
from ..riscv_csrs import (
    OFFICIAL_CSRS,
    OFFICIAL_RV32_ONLY_CSRS,
    csr_is_read_only,
    csr_is_rv32_only,
    csr_minimum_privilege,
)
from .effects import generation_schema_for_form, rv32_xregister_pair_roles


class SemanticUnsupported(ValueError):
    """该形式或其状态观察尚未被本地语义机严格建模。"""


class SemanticTrap(ValueError):
    """测试用例明确要求的架构 trap。"""

    def __init__(self, cause: int, pc_offset: int, tval: int = 0) -> None:
        super().__init__(f"semantic trap cause={cause} pc={pc_offset}")
        self.cause = int(cause)
        self.pc_offset = int(pc_offset)
        self.tval = int(tval)


@dataclass(frozen=True)
class SemanticResult:
    gpr: tuple[int, ...]
    memory: dict[str, str]
    executed_offsets: tuple[int, ...]
    extra_state: dict[str, Any]
    outcome: str = "normal"
    trap_cause: int | None = None
    trap_pc_offset: int | None = None
    trap_tval: int | None = None


_CSR_NAMES = {
    address: name.strip('"')
    for address, name in (OFFICIAL_CSRS | OFFICIAL_RV32_ONLY_CSRS).items()
}
_UNINITIALIZED_RW_CSRS = frozenset({"mscratch", "sscratch"})


def _misa_value_for_profile(isa_profile: str) -> int:
    profile = str(isa_profile or "").lower()
    if profile.startswith("rv32"):
        value = 1 << 30
    elif profile.startswith("rv64"):
        value = 2 << 62
    else:
        raise SemanticUnsupported(f"misa requires an RV32/RV64 profile: {isa_profile!r}")
    base = profile.split("_", 1)[0][4:].upper()
    letters = set(base)
    if "G" in letters:
        letters.remove("G")
        letters.update({"I", "M", "A", "F", "D"})
    if "E" in letters:
        letters.discard("I")
    else:
        letters.add("I")
    for letter in letters:
        if "A" <= letter <= "Z":
            value |= 1 << (ord(letter) - ord("A"))
    return value


_VTYPE_ALTFMT_BIT = 8
_P_PACKED_BASIC_FORMS = frozenset(
    f"{operation}.{size}"
    for operation in (
        "padd", "psub", "psadd", "psaddu", "pssub", "pssubu",
        "pmin", "pminu", "pmax", "pmaxu",
    )
    for size in ("b", "h", "w")
) | frozenset({
    "pabd.b", "pabd.h", "pabdu.b", "pabdu.h", "psabs.b", "psabs.h",
})
_VECTOR_SEGMENT_MEMORY_RE = re.compile(
    r"^(?P<op>vl|vs)(?P<mode>ux|ox|s)?seg(?P<nf>[2-8])(?P<width>ei|e)(?P<bits>8|16|32|64)\.v$"
)
_VECTOR_WHOLE_LOAD_RE = re.compile(r"^vl(?P<count>[1248])r(?:e(?P<bits>8|16|32|64))?\.v$")
_VECTOR_WHOLE_STORE_RE = re.compile(r"^vs(?P<count>[1248])r\.v$")
_VECTOR_WIDENING_FP_ARITHMETIC_FORMS = frozenset(
    f"{operation}.{suffix}"
    for operation in ("vfwadd", "vfwsub", "vfwmul")
    for suffix in ("vv", "vf", "wv", "wf")
    if not (operation == "vfwmul" and suffix in {"wv", "wf"})
)
_VECTOR_FP_REDUCTION_FORMS = frozenset({
    "vfredosum.vs", "vfredmax.vs", "vfredmin.vs", "vfwredosum.vs",
})


def _is_vector_fp_form(form: str) -> bool:
    return form.startswith("vmf") or (form.startswith("vf") and form != "vfirst.m")


def _signed(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def _sext(value: int, bits: int, width: int) -> int:
    return _signed(value, bits) & ((1 << width) - 1)


def _bits(value: int, width: int) -> int:
    return int(value) & ((1 << width) - 1)


def _float_from_bits(value: int, width: int) -> float:
    if width == 16:
        return struct.unpack("<e", int(value & 0xFFFF).to_bytes(2, "little"))[0]
    if width == 32:
        return struct.unpack("<f", int(value & 0xFFFFFFFF).to_bytes(4, "little"))[0]
    if width == 64:
        return struct.unpack("<d", int(value & ((1 << 64) - 1)).to_bytes(8, "little"))[0]
    raise SemanticUnsupported(f"floating width {width} is not supported")


def _float_to_bits(value: float, width: int) -> int:
    if math.isnan(value):
        return _canonical_nan_bits(width)
    try:
        if width == 16:
            return int.from_bytes(struct.pack("<e", float(value)), "little")
        if width == 32:
            return int.from_bytes(struct.pack("<f", float(value)), "little")
        if width == 64:
            return int.from_bytes(struct.pack("<d", float(value)), "little")
    except (OverflowError, struct.error) as exc:
        # IEEE overflow is an architectural infinity for the supported
        # binary formats; struct.pack rejects a finite Python value that is
        # above the format's range, so retry with the correctly signed inf.
        if width in {16, 32, 64} and math.isfinite(value):
            infinity = math.copysign(math.inf, value)
            if width == 16:
                return int.from_bytes(struct.pack("<e", infinity), "little")
            if width == 32:
                return int.from_bytes(struct.pack("<f", infinity), "little")
            return int.from_bytes(struct.pack("<d", infinity), "little")
        raise SemanticUnsupported(f"floating result cannot be represented at f{width}") from exc
    raise SemanticUnsupported(f"floating width {width} is not supported")


def _canonical_nan_bits(width: int) -> int:
    try:
        return {
            16: 0x7E00,
            32: 0x7FC00000,
            64: 0x7FF8000000000000,
            128: 0x7FFF8000000000000000000000000000,
        }[width]
    except KeyError as exc:
        raise SemanticUnsupported(f"canonical NaN width {width}") from exc


def _binary128_fraction(raw: int) -> Fraction | None:
    """Decode finite IEEE binary128 bits without losing significand precision."""
    raw = _bits(raw, 128)
    exponent = (raw >> 112) & 0x7FFF
    fraction = raw & ((1 << 112) - 1)
    if exponent == 0x7FFF:
        return None
    significand = fraction if exponent == 0 else (1 << 112) | fraction
    power = (1 - 16383 if exponent == 0 else exponent - 16383) - 112
    value = (
        Fraction(significand << power)
        if power >= 0 else Fraction(significand, 1 << -power)
    )
    return -value if raw >> 127 else value


def _binary128_kind(raw: int) -> str:
    exponent = (raw >> 112) & 0x7FFF
    fraction = raw & ((1 << 112) - 1)
    if exponent != 0x7FFF:
        return "finite"
    if fraction == 0:
        return "infinity"
    return "quiet_nan" if fraction & (1 << 111) else "signaling_nan"


def _binary128_compare(left_raw: int, right_raw: int) -> int:
    """Compare non-NaN binary128 values and return -1, 0, or 1."""
    left_kind, right_kind = _binary128_kind(left_raw), _binary128_kind(right_raw)
    left_sign, right_sign = left_raw >> 127, right_raw >> 127
    left_inf, right_inf = left_kind == "infinity", right_kind == "infinity"
    if left_inf and right_inf:
        left = -1 if left_sign else 1
        right = -1 if right_sign else 1
        return (left > right) - (left < right)
    if left_inf:
        return -1 if left_sign else 1
    if right_inf:
        return 1 if right_sign else -1
    left, right = _binary128_fraction(left_raw), _binary128_fraction(right_raw)
    assert left is not None and right is not None
    return (left > right) - (left < right)


def _binary128_infinity(sign: int) -> int:
    return (int(bool(sign)) << 127) | (0x7FFF << 112)


def _is_signaling_nan_bits(raw: int, width: int) -> bool:
    fraction_bits, exponent_bits = {
        16: (10, 5), 32: (23, 8), 64: (52, 11), 128: (112, 15),
    }[width]
    exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = raw & ((1 << fraction_bits) - 1)
    return (
        exponent == (1 << exponent_bits) - 1
        and fraction != 0
        and not fraction & (1 << (fraction_bits - 1))
    )


def _float_kind(raw: int, width: int) -> str:
    fraction_bits, exponent_bits = {
        16: (10, 5), 32: (23, 8), 64: (52, 11),
    }[width]
    exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = raw & ((1 << fraction_bits) - 1)
    if exponent != (1 << exponent_bits) - 1:
        return "finite"
    if fraction == 0:
        return "infinity"
    return "signaling_nan" if _is_signaling_nan_bits(raw, width) else "quiet_nan"


def _float_infinity_bits(sign: int, width: int) -> int:
    fraction_bits, exponent_bits = {
        16: (10, 5), 32: (23, 8), 64: (52, 11),
    }[width]
    return (int(bool(sign)) << (width - 1)) | (
        ((1 << exponent_bits) - 1) << fraction_bits
    )


def _fp_format_kind(raw: int, width: int) -> str:
    return _binary128_kind(raw) if width == 128 else _float_kind(raw, width)


def _fp_format_fraction(raw: int, width: int) -> Fraction | None:
    if width == 128:
        return _binary128_fraction(raw)
    if _float_kind(raw, width) != "finite":
        return None
    return Fraction.from_float(_float_from_bits(raw, width))


def _fp_infinity_bits(sign: int, width: int) -> int:
    return _binary128_infinity(sign) if width == 128 else _float_infinity_bits(sign, width)


def _round_binary128_sqrt(raw: int, rounding_mode: int) -> tuple[int, int]:
    """Return correctly rounded binary128 sqrt bits and raised IEEE flags."""
    raw = _bits(raw, 128)
    sign = raw >> 127
    exponent = (raw >> 112) & 0x7FFF
    fraction = raw & ((1 << 112) - 1)
    if exponent == 0x7FFF:
        if fraction:
            invalid = not fraction & (1 << 111)
            return _canonical_nan_bits(128), 1 << 4 if invalid else 0
        return (
            (raw, 0) if not sign else (_canonical_nan_bits(128), 1 << 4)
        )

    value = _binary128_fraction(raw)
    assert value is not None
    if value == 0:
        return raw, 0
    if value < 0:
        return _canonical_nan_bits(128), 1 << 4

    max_finite = (0x7FFE << 112) | ((1 << 112) - 1)
    lower_raw, upper_raw, floor_raw = 0, max_finite, 0
    while lower_raw <= upper_raw:
        candidate_raw = (lower_raw + upper_raw) // 2
        candidate = _binary128_fraction(candidate_raw)
        assert candidate is not None
        if candidate * candidate <= value:
            floor_raw = candidate_raw
            lower_raw = candidate_raw + 1
        else:
            upper_raw = candidate_raw - 1

    floor_value = _binary128_fraction(floor_raw)
    assert floor_value is not None
    if floor_value * floor_value == value:
        return floor_raw, 0

    ceil_raw = floor_raw + 1
    ceil_value = _binary128_fraction(ceil_raw)
    assert ceil_value is not None
    if rounding_mode in {1, 2}:
        result = floor_raw
    elif rounding_mode == 3:
        result = ceil_raw
    else:
        midpoint = (floor_value + ceil_value) / 2
        midpoint_square = midpoint * midpoint
        if value < midpoint_square:
            result = floor_raw
        elif value > midpoint_square:
            result = ceil_raw
        elif rounding_mode == 4 or floor_raw & 1:
            result = ceil_raw
        else:
            result = floor_raw
    return result, 1


def _round_fraction_to_integer(value: Fraction, rounding_mode: int) -> int:
    negative = value < 0
    numerator, denominator = abs(value.numerator), value.denominator
    integer, remainder = divmod(numerator, denominator)
    if not remainder:
        return -integer if negative else integer
    if rounding_mode == 0:
        increment = (
            remainder * 2 > denominator
            or remainder * 2 == denominator and bool(integer & 1)
        )
    elif rounding_mode == 4:
        increment = remainder * 2 >= denominator
    elif rounding_mode == 2:
        increment = negative
    elif rounding_mode == 3:
        increment = not negative
    elif rounding_mode == 1:
        increment = False
    else:
        raise SemanticUnsupported(f"reserved rounding mode {rounding_mode}")
    result = integer + int(increment)
    return -result if negative else result


def _round_fraction_to_bits(
    value: Fraction, width: int, rounding_mode: int, *, negative_zero: bool = False,
) -> int:
    exponent_bits, fraction_bits, bias = {
        16: (5, 10, 15), 32: (8, 23, 127), 64: (11, 52, 1023),
        128: (15, 112, 16383),
    }[width]
    sign = int(value < 0)
    value = abs(value)
    if value == 0:
        return int(negative_zero) << (width - 1)
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    emin, emax = 1 - bias, ((1 << exponent_bits) - 2) - bias

    def rounded_quotient(n: int, d: int) -> int:
        quotient, remainder = divmod(n, d)
        if not remainder:
            return quotient
        if rounding_mode == 5:  # ROD: round toward zero, then jam to odd.
            return quotient | 1
        if rounding_mode == 0:
            increment = remainder * 2 > d or remainder * 2 == d and bool(quotient & 1)
        elif rounding_mode == 4:
            increment = remainder * 2 >= d
        elif rounding_mode == 2:
            increment = bool(sign)
        elif rounding_mode == 3:
            increment = not sign
        else:
            increment = False
        return quotient + int(increment)

    if exponent >= emin:
        shift = fraction_bits - exponent
        significand = rounded_quotient(
            numerator << shift if shift >= 0 else numerator,
            denominator if shift >= 0 else denominator << -shift,
        )
        if significand >= 1 << (fraction_bits + 1):
            significand >>= 1
            exponent += 1
        if exponent > emax:
            to_infinity = rounding_mode in {0, 4} or (
                (rounding_mode == 2 and bool(sign))
                or (rounding_mode == 3 and not sign)
            )
            exponent_field = (1 << exponent_bits) - 1 if to_infinity else (1 << exponent_bits) - 2
            fraction_field = 0 if to_infinity else (1 << fraction_bits) - 1
        else:
            exponent_field = exponent + bias
            fraction_field = significand - (1 << fraction_bits)
    else:
        shift = fraction_bits - emin
        significand = rounded_quotient(numerator << shift, denominator)
        if significand >= 1 << fraction_bits:
            exponent_field, fraction_field = 1, 0
        else:
            exponent_field, fraction_field = 0, significand
    return (sign << (width - 1)) | (exponent_field << fraction_bits) | fraction_field


def _round_fraction_to_precision(value: Fraction, width: int, rounding_mode: int) -> Fraction:
    """Round to the format's precision with an unbounded exponent range."""
    fraction_bits = {16: 10, 32: 23, 64: 52, 128: 112}[width]
    if value == 0:
        return Fraction(0)
    negative = value < 0
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    shift = fraction_bits - exponent
    scaled_numerator = numerator << shift if shift >= 0 else numerator
    scaled_denominator = denominator if shift >= 0 else denominator << -shift
    significand, remainder = divmod(scaled_numerator, scaled_denominator)
    if remainder:
        if rounding_mode == 0:
            increment = (
                remainder * 2 > scaled_denominator
                or remainder * 2 == scaled_denominator and bool(significand & 1)
            )
        elif rounding_mode == 4:
            increment = remainder * 2 >= scaled_denominator
        elif rounding_mode == 2:
            increment = negative
        elif rounding_mode == 3:
            increment = not negative
        elif rounding_mode == 1:
            increment = False
        else:
            raise SemanticUnsupported(f"reserved rounding mode {rounding_mode}")
        significand += int(increment)
    if significand >= 1 << (fraction_bits + 1):
        significand >>= 1
        exponent += 1
    shift = exponent - fraction_bits
    rounded = (
        Fraction(significand << shift)
        if shift >= 0 else Fraction(significand, 1 << -shift)
    )
    return -rounded if negative else rounded


def _round_finite_fraction_with_flags(
    value: Fraction, width: int, rounding_mode: int, *, negative_zero: bool = False,
) -> tuple[int, int]:
    """Round one finite exact result and return its precise IEEE fflags."""
    bits = _round_fraction_to_bits(
        value, width, rounding_mode, negative_zero=negative_zero,
    )
    if width == 128:
        rounded = _binary128_fraction(bits)
    else:
        rounded_value = _float_from_bits(bits, width)
        rounded = Fraction.from_float(rounded_value) if math.isfinite(rounded_value) else None
    inexact = rounded is None or rounded != value
    unbounded = _round_fraction_to_precision(value, width, rounding_mode)
    exponent_bits, fraction_bits, bias = {
        16: (5, 10, 15), 32: (8, 23, 127), 64: (11, 52, 1023),
        128: (15, 112, 16383),
    }[width]
    emin = 1 - bias
    emax = ((1 << exponent_bits) - 2) - bias
    max_significand = (1 << (fraction_bits + 1)) - 1
    max_finite = (
        Fraction(max_significand << (emax - fraction_bits))
        if emax >= fraction_bits
        else Fraction(max_significand, 1 << (fraction_bits - emax))
    )
    overflow = abs(unbounded) > max_finite
    min_normal = (
        Fraction(1 << emin) if emin >= 0 else Fraction(1, 1 << -emin)
    )
    tiny_after_rounding = abs(unbounded) < min_normal
    flags = int(inexact)  # NX
    if overflow:
        flags |= 1 << 2  # OF implies NX.
        flags |= 1
    elif tiny_after_rounding and inexact:
        flags |= 1 << 1  # RISC-V detects tininess after rounding.
    return bits, flags


def _round_sqrt_to_bits(value: float, width: int, rounding_mode: int) -> int:
    if value < 0:
        return _float_to_bits(math.nan, width)
    if value == 0 or not math.isfinite(value):
        return _float_to_bits(math.sqrt(value), width)
    exact = Fraction.from_float(value)
    nearest = _float_to_bits(math.sqrt(value), width)
    rounded = Fraction.from_float(_float_from_bits(nearest, width))
    square = rounded * rounded
    if square == exact:
        return nearest
    if rounding_mode in {1, 2}:
        return nearest - 1 if square > exact else nearest
    if rounding_mode == 3:
        return nearest if square > exact else nearest + 1
    lower, upper = (nearest - 1, nearest) if square > exact else (nearest, nearest + 1)
    lower_value = Fraction.from_float(_float_from_bits(lower, width))
    upper_value = Fraction.from_float(_float_from_bits(upper, width))
    midpoint_square = ((lower_value + upper_value) / 2) ** 2
    if exact < midpoint_square:
        return lower
    if exact > midpoint_square:
        return upper
    return upper if rounding_mode == 4 or lower & 1 else lower


def _round_integral(value: float, rounding_mode: int) -> int:
    if rounding_mode == 0:
        return round(value)
    if rounding_mode == 1:
        return math.trunc(value)
    if rounding_mode == 2:
        return math.floor(value)
    if rounding_mode == 3:
        return math.ceil(value)
    if rounding_mode == 4:
        return int(math.copysign(math.floor(abs(value) + 0.5), value))
    raise SemanticUnsupported(f"reserved floating rounding mode {rounding_mode}")


def _float_to_integer(value: float, width: int, *, unsigned: bool, rounding_mode: int) -> int:
    lower = 0 if unsigned else -(1 << (width - 1))
    upper = (1 << width) - 1 if unsigned else (1 << (width - 1)) - 1
    if math.isnan(value):
        return upper
    if math.isinf(value):
        return lower if value < 0 else upper
    rounded = int(_round_integral(value, rounding_mode))
    return max(lower, min(upper, rounded))


def _roundoff_fixed(value: int, shift: int, rounding_mode: int) -> int:
    """Apply RVV vxrm rounding to a signed/unsigned integer right shift."""

    shift = int(shift)
    if shift <= 0:
        return int(value) << -shift
    value = int(value)
    quotient = value >> shift
    round_bit = (value >> (shift - 1)) & 1
    sticky = bool(value & ((1 << (shift - 1)) - 1))
    if rounding_mode == 0:  # RNU: nearest, ties toward +infinity.
        return quotient + round_bit
    if rounding_mode == 1:  # RNE: nearest, ties to even.
        return quotient + int(bool(round_bit and (sticky or (quotient & 1))))
    if rounding_mode == 2:  # RDN: truncate discarded bits.
        return quotient
    if rounding_mode == 3:  # ROD: jam any discarded bit into the LSB.
        return quotient | int(bool(round_bit or sticky))
    raise SemanticUnsupported(f"reserved vxrm rounding mode {rounding_mode}")


def _ieee_divide(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if (numerator == 0 and denominator == 0) or (
        math.isinf(numerator) and math.isinf(denominator)
    ):
        return math.nan
    if denominator == 0:
        sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
        return math.copysign(math.inf, sign)
    return numerator / denominator


def _fp_minmax(left: float, right: float, *, minimum: bool, canonical_if_any_nan: bool = False) -> float:
    if canonical_if_any_nan and (math.isnan(left) or math.isnan(right)):
        return math.nan
    if math.isnan(left) and math.isnan(right):
        return math.nan
    if math.isnan(left):
        return right
    if math.isnan(right):
        return left
    if left == 0 and right == 0:
        left_negative = math.copysign(1.0, left) < 0
        right_negative = math.copysign(1.0, right) < 0
        if minimum:
            return -0.0 if left_negative or right_negative else 0.0
        return 0.0 if not left_negative or not right_negative else -0.0
    return min(left, right) if minimum else max(left, right)


def _round_f32_to_bf16(raw: int, rounding_mode: int) -> int:
    raw = int(raw) & 0xFFFFFFFF
    upper, lower = raw >> 16, raw & 0xFFFF
    exponent, fraction = (raw >> 23) & 0xFF, raw & 0x7FFFFF
    if exponent == 0xFF and fraction:
        return 0x7FC0
    increment = (
        rounding_mode == 0 and (lower > 0x8000 or (lower == 0x8000 and bool(upper & 1)))
        or rounding_mode == 2 and bool(raw >> 31)
        or rounding_mode == 3 and not raw >> 31
        or rounding_mode == 4 and lower >= 0x8000
    )
    return (upper + int(bool(lower and increment))) & 0xFFFF


def _float_width(form: str) -> int | None:
    if form.endswith(".q") or form.endswith("_q"):
        return 128
    if form.endswith(".s") or form.endswith("_s"):
        return 32
    if form.endswith(".d") or form.endswith("_d"):
        return 64
    if form.endswith(".h") or form.endswith("_h"):
        return 16
    return None


def _normal_form(value: object) -> str:
    return str(value or "").lower().replace("_", ".")


def _field(item: object, name: str, default: int | None = None) -> int | None:
    fields = dict(getattr(item, "operand_fields", ()) or ())
    value = fields.get(name)
    if value is None:
        value = getattr(item, name, None)
    return value if isinstance(value, int) else default


def _immediate(item: object, default: int = 0) -> int:
    value = getattr(item, "immediate", None)
    if isinstance(value, int):
        return value
    fields = dict(getattr(item, "operand_fields", ()) or ())
    for name in (
        "imm12s", "imm12", "simm5", "zimm11", "zimm10", "zimm5",
        "p_w_uimm6", "p_w_uimm3", "p_imm8", "p_imm9", "p_imm10",
        "p_imm11", "uimm", "shamt",
    ):
        if isinstance(fields.get(name), int):
            return int(fields[name])
    for name in ("imm20", "c_nzimm18"):
        if isinstance(fields.get(name), int):
            value = semantic_immediate_value(name, int(fields[name]))
            if value is not None:
                return value
    # The catalog carries a few extension-specific immediate names.  They are
    # still literal immediates, so accepting a field whose name explicitly
    # contains ``imm`` is safer than dropping the form into a reference gap.
    for name, value in fields.items():
        if "imm" in str(name).lower() and isinstance(value, int):
            return int(value)
    return default


def _width_from_suffix(form: str, default: int) -> int:
    for suffix, width in ((".q", 16), (".d", 8), (".w", 4), (".h", 2), (".b", 1)):
        if suffix in form:
            return width
    return default


def _gf8_multiply(left: int, right: int) -> int:
    product = 0
    for _ in range(8):
        if right & 1:
            product ^= left
        left = ((left << 1) ^ (0x11B if left & 0x80 else 0)) & 0xFF
        right >>= 1
    return product


def _aes_sbox(value: int) -> int:
    inverse = 0
    if value:
        inverse = 1
        base, exponent = value, 254
        while exponent:
            if exponent & 1:
                inverse = _gf8_multiply(inverse, base)
            base = _gf8_multiply(base, base)
            exponent >>= 1
    rotate = lambda amount: ((inverse << amount) | (inverse >> (8 - amount))) & 0xFF
    return inverse ^ rotate(1) ^ rotate(2) ^ rotate(3) ^ rotate(4) ^ 0x63


_AES_SBOX = tuple(_aes_sbox(value) for value in range(256))
_AES_INV_SBOX = [0] * 256
for _aes_index, _aes_value in enumerate(_AES_SBOX):
    _AES_INV_SBOX[_aes_value] = _aes_index
_AES_INV_SBOX = tuple(_AES_INV_SBOX)
_BIT_REVERSE_BYTE = tuple(int(f"{value:08b}"[::-1], 2) for value in range(256))
_SM4_SBOX = bytes.fromhex("""
    d6 90 e9 fe cc e1 3d b7 16 b6 14 c2 28 fb 2c 05
    2b 67 9a 76 2a be 04 c3 aa 44 13 26 49 86 06 99
    9c 42 50 f4 91 ef 98 7a 33 54 0b 43 ed cf ac 62
    e4 b3 1c a9 c9 08 e8 95 80 df 94 fa 75 8f 3f a6
    47 07 a7 fc f3 73 17 ba 83 59 3c 19 e6 85 4f a8
    68 6b 81 b2 71 64 da 8b f8 eb 0f 4b 70 56 9d 35
    1e 24 0e 5e 63 58 d1 a2 25 22 7c 3b 01 21 78 87
    d4 00 46 57 9f d3 27 52 4c 36 02 e7 a0 c4 c8 9e
    ea bf 8a d2 40 c7 38 b5 a3 f7 f2 ce f9 61 15 a1
    e0 ae 5d a4 9b 34 1a 55 ad 93 32 30 f5 8c b1 e3
    1d f6 e2 2e 82 66 ca 60 c0 29 23 ab 0d 53 4e 6f
    d5 db 37 45 de fd 8e 2f 03 ff 6a 72 6d 6c 5b 51
    8d 1b af 92 bb dd bc 7f 11 d9 5c 41 1f 10 5a d8
    0a c1 31 88 a5 cd 7b bd 2d 74 d0 12 b8 e5 b4 b0
    89 69 97 4a 0c 96 77 7e 65 b9 f1 09 c5 6e c6 84
    18 f0 7d ec 3a dc 4d 20 79 ee 5f 3e d7 cb 39 48
""")
_SM4_CK = (
    0x00070E15, 0x1C232A31, 0x383F464D, 0x545B6269,
    0x70777E85, 0x8C939AA1, 0xA8AFB6BD, 0xC4CBD2D9,
    0xE0E7EEF5, 0xFC030A11, 0x181F262D, 0x343B4249,
    0x50575E65, 0x6C737A81, 0x888F969D, 0xA4ABB2B9,
    0xC0C7CED5, 0xDCE3EAF1, 0xF8FF060D, 0x141B2229,
    0x30373E45, 0x4C535A61, 0x686F767D, 0x848B9299,
    0xA0A7AEB5, 0xBCC3CAD1, 0xD8DFE6ED, 0xF4FB0209,
    0x10171E25, 0x2C333A41, 0x484F565D, 0x646B7279,
)


def _rotate_left(value: int, amount: int, width: int) -> int:
    amount %= width
    mask = (1 << width) - 1
    return ((value << amount) | (value >> ((width - amount) % width))) & mask


def _rotate_right(value: int, amount: int, width: int) -> int:
    return _rotate_left(value, -amount, width)


def _reverse_bits_each_byte(value: int, width: int) -> int:
    return sum(_BIT_REVERSE_BYTE[(value >> shift) & 0xFF] << shift for shift in range(0, width, 8))


def _vector_group(values: list[int], start: int, count: int, width: int) -> list[int]:
    mask = (1 << width) - 1
    return [int(values[start + index]) & mask for index in range(count)]


def _pack_vector_group(values: list[int], width: int) -> int:
    return sum(int(value) << (index * width) for index, value in enumerate(values))


def _aes_subword(word: int, inverse: bool = False) -> int:
    table = _AES_INV_SBOX if inverse else _AES_SBOX
    return sum(table[(word >> shift) & 0xFF] << shift for shift in (0, 8, 16, 24))


def _aes_rotword(word: int) -> int:
    a0, a1, a2, a3 = ((word >> shift) & 0xFF for shift in (0, 8, 16, 24))
    return (a0 << 24) | (a3 << 16) | (a2 << 8) | a1


def _aes_shift_rows(words: list[int], inverse: bool = False) -> list[int]:
    direction = -1 if inverse else 1
    return [
        sum(((words[(column + direction * row) % 4] >> (8 * row)) & 0xFF) << (8 * row)
            for row in range(4))
        for column in range(4)
    ]


def _aes_mix_columns(words: list[int], inverse: bool = False) -> list[int]:
    coefficients = (
        ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))
        if inverse else ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
    )
    result = []
    for word in words:
        column = [(word >> (8 * index)) & 0xFF for index in range(4)]
        output = [
            _xor(_gf8_multiply(column[index], coefficient) for index, coefficient in enumerate(row))
            for row in coefficients
        ]
        result.append(sum(value << (8 * index) for index, value in enumerate(output)))
    return result


def _aes_mixcolumn_word(word: int, inverse: bool = False) -> int:
    return _aes_mix_columns([int(word) & 0xFFFFFFFF], inverse)[0]


def _aes_mixcolumn_byte(value: int, inverse: bool = False) -> int:
    coefficients = (14, 9, 13, 11) if inverse else (2, 1, 1, 3)
    return sum(
        _gf8_multiply(int(value) & 0xFF, coefficient) << (8 * index)
        for index, coefficient in enumerate(coefficients)
    )


def _aes_rv64_shiftrows(rs2: int, rs1: int, *, inverse: bool = False) -> int:
    rs1, rs2 = _bits(rs1, 64), _bits(rs2, 64)
    indices = (
        (("rs2", 3), ("rs2", 6), ("rs1", 1), ("rs1", 4), ("rs1", 7), ("rs2", 2), ("rs2", 5), ("rs1", 0))
        if inverse else
        (("rs1", 3), ("rs2", 6), ("rs2", 1), ("rs1", 4), ("rs2", 7), ("rs2", 2), ("rs1", 5), ("rs1", 0))
    )
    sources = {"rs1": rs1, "rs2": rs2}
    result = 0
    for position, (source, byte_index) in enumerate(indices):
        result |= ((sources[source] >> (byte_index * 8)) & 0xFF) << ((7 - position) * 8)
    return result


def _xor(values: object) -> int:
    result = 0
    for value in values:
        result ^= int(value)
    return result


def _aes_vector_round(state: list[int], key: list[int], form: str) -> list[int]:
    if form == "vaesz.vs":
        return [left ^ right for left, right in zip(state, key)]
    inverse = form.startswith("vaesd")
    table = _AES_INV_SBOX if inverse else _AES_SBOX
    words = [sum(table[(word >> shift) & 0xFF] << shift for shift in (0, 8, 16, 24)) for word in state]
    words = _aes_shift_rows(words, inverse)
    if form.startswith(("vaesem", "vaesdm")):
        words = _aes_mix_columns(words, inverse)
    return [left ^ right for left, right in zip(words, key)]


def _aes_key_schedule(current: list[int], previous: list[int] | None, immediate: int, form: str) -> list[int]:
    rnd = immediate & 0xF
    rcon = (1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36, 0, 0, 0, 0, 0, 0)
    if form == "vaeskf1.vi":
        if rnd == 0 or rnd > 10:
            rnd ^= 8
        result = [_aes_subword(_aes_rotword(current[3])) ^ rcon[(rnd - 1) & 0xF]]
        for word in current[1:]:
            result.append(result[-1] ^ word)
        return result
    if previous is None:
        raise SemanticUnsupported("vaeskf2.vi requires previous AES-256 round key")
    if rnd < 2 or rnd > 14:
        rnd ^= 8
    if rnd & 1:
        transformed = _aes_subword(current[3])
    else:
        transformed = _aes_subword(_aes_rotword(current[3])) ^ rcon[((rnd >> 1) - 1) & 0xF]
    result = [transformed ^ previous[0]]
    for word in previous[1:]:
        result.append(result[-1] ^ word)
    return result


def _sm4_subword(word: int) -> int:
    return sum(_SM4_SBOX[(word >> shift) & 0xFF] << shift for shift in (0, 8, 16, 24))


def _sm4_linear(value: int) -> int:
    return value ^ _rotate_left(value, 2, 32) ^ _rotate_left(value, 10, 32) \
        ^ _rotate_left(value, 18, 32) ^ _rotate_left(value, 24, 32)


def _sm4_key_linear(value: int) -> int:
    return value ^ _rotate_left(value, 13, 32) ^ _rotate_left(value, 23, 32)


def _sm4_rounds(state: list[int], keys: list[int]) -> list[int]:
    x0, x1, x2, x3 = state
    rk0, rk1, rk2, rk3 = keys
    x4 = x0 ^ _sm4_linear(_sm4_subword(x1 ^ x2 ^ x3 ^ rk0))
    x5 = x1 ^ _sm4_linear(_sm4_subword(x2 ^ x3 ^ x4 ^ rk1))
    x6 = x2 ^ _sm4_linear(_sm4_subword(x3 ^ x4 ^ x5 ^ rk2))
    x7 = x3 ^ _sm4_linear(_sm4_subword(x4 ^ x5 ^ x6 ^ rk3))
    return [x4, x5, x6, x7]


def _sm4_key_rounds(keys: list[int], immediate: int) -> list[int]:
    rk0, rk1, rk2, rk3 = keys
    base = 4 * (immediate & 7)
    rk4 = rk0 ^ _sm4_key_linear(_sm4_subword(rk1 ^ rk2 ^ rk3 ^ _SM4_CK[base]))
    rk5 = rk1 ^ _sm4_key_linear(_sm4_subword(rk2 ^ rk3 ^ rk4 ^ _SM4_CK[base + 1]))
    rk6 = rk2 ^ _sm4_key_linear(_sm4_subword(rk3 ^ rk4 ^ rk5 ^ _SM4_CK[base + 2]))
    rk7 = rk3 ^ _sm4_key_linear(_sm4_subword(rk4 ^ rk5 ^ rk6 ^ _SM4_CK[base + 3]))
    return [rk4, rk5, rk6, rk7]


def _ghash_multiply(left: int, right: int) -> int:
    product = 0
    for _ in range(128):
        if left & 1:
            product ^= right
        left >>= 1
        carry = right >> 127
        right = (right << 1) & ((1 << 128) - 1)
        if carry:
            right ^= 0x87
    return product


def _sha2_schedule(old: list[int], middle: list[int], recent: list[int], width: int) -> list[int]:
    mask = (1 << width) - 1
    words = [0] * 20
    words[:4] = old
    words[4], words[9], words[10], words[11] = middle
    words[12:16] = recent
    if width == 32:
        s0, s1 = (7, 18, 3), (17, 19, 10)
    else:
        s0, s1 = (1, 8, 7), (19, 61, 6)

    def rotr(value: int, amount: int) -> int:
        return ((value >> amount) | (value << (width - amount))) & mask

    for index in range(16, 20):
        small0 = rotr(words[index - 15], s0[0]) ^ rotr(words[index - 15], s0[1]) ^ (words[index - 15] >> s0[2])
        small1 = rotr(words[index - 2], s1[0]) ^ rotr(words[index - 2], s1[1]) ^ (words[index - 2] >> s1[2])
        words[index] = (small1 + words[index - 7] + small0 + words[index - 16]) & mask
    return words[16:20]


def _sha2_compress_pair(vs2: list[int], vd: list[int], schedule: list[int], width: int, high: bool) -> list[int]:
    mask = (1 << width) - 1
    a, b, e, f = reversed(vs2)
    c, d, g, h = reversed(vd)
    w0, w1 = schedule[2:4] if high else schedule[:2]
    rotations = (2, 13, 22, 6, 11, 25) if width == 32 else (28, 34, 39, 14, 18, 41)

    def rotr(value: int, amount: int) -> int:
        return ((value >> amount) | (value << (width - amount))) & mask

    def step(message: int) -> None:
        nonlocal a, b, c, d, e, f, g, h
        sigma1 = rotr(e, rotations[3]) ^ rotr(e, rotations[4]) ^ rotr(e, rotations[5])
        choose = (e & f) ^ ((~e) & g)
        t1 = (h + sigma1 + choose + message) & mask
        sigma0 = rotr(a, rotations[0]) ^ rotr(a, rotations[1]) ^ rotr(a, rotations[2])
        majority = (a & b) ^ (a & c) ^ (b & c)
        t2 = (sigma0 + majority) & mask
        a, b, c, d, e, f, g, h = (t1 + t2) & mask, a, b, c, (d + t1) & mask, e, f, g

    step(w0)
    step(w1)
    return [f, e, b, a]


def _sm3_message_schedule(vs1: list[int], vs2: list[int]) -> list[int]:
    swap = lambda value: int.from_bytes(value.to_bytes(4, "little"), "big")
    words = [swap(value) for value in (*vs1, *vs2)]
    for index in range(16, 24):
        mixed = words[index - 16] ^ words[index - 9] ^ _rotate_left(words[index - 3], 15, 32)
        p1 = mixed ^ _rotate_left(mixed, 15, 32) ^ _rotate_left(mixed, 23, 32)
        words.append((p1 ^ _rotate_left(words[index - 13], 7, 32) ^ words[index - 6]) & 0xFFFFFFFF)
    return [swap(value) for value in words[16:24]]


def _sm3_compress_pair(state: list[int], message: list[int], immediate: int) -> list[int]:
    swap = lambda value: int.from_bytes(value.to_bytes(4, "little"), "big")
    a, b, c, d, e, f, g, h = (swap(value) for value in state)
    w = [swap(value) for value in message]

    def step(index: int, message_word: int) -> None:
        nonlocal a, b, c, d, e, f, g, h
        constant = 0x79CC4519 if index <= 15 else 0x7A879D8A
        a12 = _rotate_left(a, 12, 32)
        ss1 = _rotate_left((a12 + e + _rotate_left(constant, index, 32)) & 0xFFFFFFFF, 7, 32)
        ss2 = ss1 ^ a12
        ff = a ^ b ^ c if index <= 15 else (a & b) | (a & c) | (b & c)
        gg = e ^ f ^ g if index <= 15 else (e & f) | ((~e) & g)
        tt1 = (ff + d + ss2 + message_word) & 0xFFFFFFFF
        tt2 = (gg + h + ss1 + w[index & 1]) & 0xFFFFFFFF
        d, c, b, a = c, _rotate_left(b, 9, 32), a, tt1
        h, g, f, e = g, _rotate_left(f, 19, 32), e, tt2 ^ _rotate_left(tt2, 9, 32) ^ _rotate_left(tt2, 17, 32)

    step(2 * immediate, w[0] ^ w[4])
    step(2 * immediate + 1, w[1] ^ w[5])
    return [swap(value) for value in (a, b, c, d, e, f, g, h)]


def _instruction_word(testcase: object, item: object) -> int:
    raw = getattr(testcase, "code_bytes", b"")
    offset = int(getattr(item, "byte_offset", 0))
    length = int(getattr(item, "byte_length", 0))
    if not isinstance(raw, (bytes, bytearray)) or length not in {2, 4}:
        return 0
    return int.from_bytes(raw[offset:offset + length], "little")


def _environment_call_cause(testcase: object) -> int:
    dataflow = getattr(testcase, "dataflow_meta", {})
    realization = dataflow.get("generation_realization", {}) if isinstance(dataflow, dict) else {}
    if isinstance(realization, dict):
        mode = str(
            realization.get("privilege_mode")
            or realization.get("privilege_class")
            or realization.get("environment_profile")
            or ""
        ).lower()
        cause = {
            "user": 8, "u": 8,
            "supervisor": 9, "s": 9,
            "hypervisor": 10, "h": 10, "virtual-supervisor": 10, "vs": 10,
            "machine": 11, "m": 11, "base": 11,
        }.get(mode)
        if cause is not None:
            return cause
    return 8


class SemanticMachine:
    """执行一个冻结 RVGEN testcase 的架构状态。"""

    def __init__(self, testcase: object) -> None:
        self.testcase = testcase
        profile = str(getattr(testcase, "isa_profile", "rv64i")).lower()
        self.xlen = 32 if profile.startswith("rv32") else 64
        self.mask = (1 << self.xlen) - 1
        self.extensions = enabled_extensions(profile)
        self.x_register_fp = bool(x_register_fp_extensions(profile))
        self.flen = (
            128 if "q" in self.extensions else
            64 if "d" in self.extensions else
            32 if "f" in self.extensions else 0
        )
        self.gpr = [_bits(value, self.xlen) for value in getattr(testcase, "initial_gpr", (0,) * 32)]
        self.gpr += [0] * (32 - len(self.gpr))
        self.gpr = self.gpr[:32]
        self.fpr = [0] * 32
        # Each entry is one architectural vector register, stored as VLEN bits.
        self.vreg = [0] * 32
        self.csrs: dict[str, int] = {
            "fflags": 0, "frm": 0, "fcsr": 0, "vstart": 0,
            "vxsat": 0, "vxrm": 0, "vcsr": 0,
            "mstatus": (
                (1 << 13 if testcase_uses_fp_instruction(testcase) else 0)
                | (1 << 9 if testcase_uses_vector_instruction(testcase) else 0)
            ),
            "vl": 0, "vtype": 0, "vlenb": 32,
        }
        if profile.startswith(("rv32", "rv64")):
            self.csrs["misa"] = _misa_value_for_profile(profile)
        self.regions = {
            region.region_id: bytearray(region.data)
            for region in getattr(testcase, "initial_memory_regions", ())
        }
        self.region_meta = {
            region.region_id: region
            for region in getattr(testcase, "initial_memory_regions", ())
        }
        self.reservation: tuple[int, int] | None = None
        self.executed: list[int] = []
        self.fp_used = False
        self.fp_flags_exact = True
        self.vector_used = False
        self.csr_used = False
        self._by_offset = {
            int(item.pc_offset): index
            for index, item in enumerate(getattr(testcase, "instruction_meta", ()))
        }

    # ----- common register/memory helpers ---------------------------------

    def _read_gpr(self, index: int | None) -> int:
        if index is None or not 0 <= int(index) < 32:
            raise SemanticUnsupported("missing GPR operand")
        return self.gpr[int(index)]

    def _write_gpr(self, index: int | None, value: int) -> None:
        if index is not None and int(index) != 0:
            self.gpr[int(index)] = _bits(value, self.xlen)
        self.gpr[0] = 0

    def _read_operand(self, item: object, name: str) -> int:
        index = getattr(item, name, None)
        domain = getattr(item, f"{name}_domain", None)
        if index is None:
            index = _field(item, name)
        if index is None:
            raise SemanticUnsupported(f"missing {name} operand")
        if domain == "fpr":
            self.fp_used = True
            return self.fpr[int(index)]
        return self._read_gpr(int(index))

    def _xregister_fp_width_supported(self, width: int) -> bool:
        required = {
            16: {"zhinx", "zhinxmin"},
            32: {"zfinx"},
            64: {"zdinx"},
        }.get(width)
        return required is not None and not self.extensions.isdisjoint(required)

    def _rv32_xregister_fp_pair_role(self, item: object, name: str, width: int) -> bool:
        if self.xlen != 32 or width != 64 or not self.x_register_fp:
            return False
        mnemonic = str(getattr(item, "mnemonic", "")).lower().replace(".", "_")
        form = OFFICIAL_ALL_CATALOG_BY_MNEMONIC.get(mnemonic)
        return form is not None and name in rv32_xregister_pair_roles(
            form,
            generation_schema_for_form(form),
            xlen=self.xlen,
            register_carrier="gpr",
        )

    def _rv32_xregister_pair_base(self, item: object, name: str) -> int:
        index = getattr(item, name, None)
        if index is None:
            index = _field(item, name)
        if index is None:
            raise SemanticUnsupported(f"missing {name} operand")
        index = int(index)
        if not 0 <= index < 32:
            raise SemanticUnsupported(
                f"RV32 Zdinx {name}=x{index} is outside the GPR register file"
            )
        if index & 1:
            raise SemanticUnsupported(
                f"RV32 Zdinx {name}=x{index} is a reserved misaligned register pair"
            )
        return index

    def _read_fp_bits(self, item: object, name: str, width: int) -> int:
        domain = getattr(item, f"{name}_domain", None)
        if domain == "gpr" and self.x_register_fp:
            if not self._xregister_fp_width_supported(width):
                raise SemanticUnsupported(
                    f"f{width} X-register operands are not enabled by the ISA profile"
                )
            if width > self.xlen:
                if self._rv32_xregister_fp_pair_role(item, name, width):
                    base = self._rv32_xregister_pair_base(item, name)
                    if base == 0:
                        return 0
                    return self._read_gpr(base) | (self._read_gpr(base + 1) << 32)
                raise SemanticUnsupported(
                    f"f{width} X-register operand exceeds XLEN={self.xlen}"
                )
            return _bits(self._read_operand(item, name), width)
        if not self.flen or width > self.flen:
            raise SemanticUnsupported(
                f"f{width} register state exceeds profile FLEN={self.flen}"
            )
        raw = self._read_operand(item, name)
        if width < self.flen and raw >> width != (1 << (self.flen - width)) - 1:
            return _canonical_nan_bits(width)
        return _bits(raw, width)

    def _write_operand(self, item: object, name: str, value: int, *, width: int | None = None) -> None:
        index = getattr(item, name, None)
        domain = getattr(item, f"{name}_domain", None)
        if index is None:
            index = _field(item, name)
        if index is None:
            return
        if width is not None:
            value = _bits(value, width)
            mnemonic = str(getattr(item, "mnemonic", "")).lower()
            if (
                name == "rd"
                and domain == "gpr"
                and self.x_register_fp
                and mnemonic.startswith(("f", "c.f"))
            ):
                if not self._xregister_fp_width_supported(width):
                    raise SemanticUnsupported(
                        f"f{width} X-register results are not enabled by the ISA profile"
                    )
                if width > self.xlen:
                    if self._rv32_xregister_fp_pair_role(item, name, width):
                        base = self._rv32_xregister_pair_base(item, name)
                        if base != 0:
                            self._write_gpr(base, value)
                            self._write_gpr(base + 1, value >> 32)
                        self.gpr[0] = 0
                        return
                    raise SemanticUnsupported(
                        f"f{width} X-register result exceeds XLEN={self.xlen}"
                    )
                value = _sext(value, width, self.xlen)
        if domain == "fpr":
            if not self.flen or width is not None and width > self.flen:
                raise SemanticUnsupported(
                    f"FPR write exceeds profile FLEN={self.flen}"
                )
            self.fp_used = True
            if width is not None and width < self.flen:
                value = _bits(value, width) | (((1 << (self.flen - width)) - 1) << width)
            self.fpr[int(index)] = _bits(value, self.flen)
        else:
            self._write_gpr(int(index), value)

    def _locate(self, address: int, width: int) -> tuple[str, int]:
        address = _bits(address, self.xlen)
        for region_id, region in self.region_meta.items():
            offset = address - int(region.address)
            if 0 <= offset <= len(region.data) - width:
                return region_id, offset
        raise SemanticUnsupported(f"memory access outside testcase regions: 0x{address:x}/{width}")

    def _load(self, address: int, width: int, *, signed: bool = False) -> int:
        region_id, offset = self._locate(address, width)
        value = int.from_bytes(self.regions[region_id][offset:offset + width], "little")
        return _signed(value, width * 8) if signed else value

    def _store(self, address: int, width: int, value: int) -> None:
        region_id, offset = self._locate(address, width)
        self.regions[region_id][offset:offset + width] = _bits(value, width * 8).to_bytes(width, "little")
        if self.reservation is not None:
            reserved_address, reserved_width = self.reservation
            if address < reserved_address + reserved_width and reserved_address < address + width:
                self.reservation = None

    def _vector_status_implemented(self) -> bool:
        return "v" in self.extensions or any(
            extension.startswith("zve") for extension in self.extensions
        )

    def _check_vxsat_access(self, item: object, form: str) -> None:
        if self._vector_status_implemented() and not (
            int(self.csrs.get("mstatus", 0)) & (3 << 9)
        ):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

    def _set_vxsat(self, item: object, form: str) -> None:
        self._check_vxsat_access(item, form)
        self.csrs["vxsat"] = 1
        if self._vector_status_implemented():
            self.csrs["vcsr"] = (int(self.csrs.get("vxrm", 0)) & 3) | (1 << 2)
            self.csrs["mstatus"] = (
                (int(self.csrs.get("mstatus", 0)) & ~(3 << 9)) | (3 << 9)
            )
        self.csr_used = True

    def _require_atomic_read_write(self, address: int, width: int, form: str) -> None:
        region_id, _ = self._locate(address, width)
        permissions = set(self.region_meta[region_id].permissions)
        if not {"r", "w"}.issubset(permissions):
            raise SemanticUnsupported(
                f"{form} requires readable and writable memory; "
                f"region {region_id!r} has permissions {''.join(sorted(permissions))!r}"
            )

    def _check_vector_index_width(self, item: object, index_width: int) -> None:
        if self.xlen == 32 and index_width == 64:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

    def _address(self, item: object) -> int:
        base = self._read_operand(item, "rs1")
        return _bits(base + _immediate(item), self.xlen)

    def _csr_name(self, item: object) -> str:
        csr = _field(item, "csr")
        return _CSR_NAMES.get(int(csr)) if csr is not None else "unknown"

    def _packed_lanes(self, value: int, lane_bits: int) -> list[int]:
        """Split an XLEN value into little-endian packed lanes."""

        if lane_bits not in {8, 16, 32, 64} or lane_bits > self.xlen:
            raise SemanticUnsupported(f"packed lane width {lane_bits}")
        mask = (1 << lane_bits) - 1
        return [(_bits(value, self.xlen) >> offset) & mask
                for offset in range(0, self.xlen, lane_bits)]

    def _pack_lanes(self, lanes: list[int], lane_bits: int) -> int:
        mask = (1 << lane_bits) - 1
        value = 0
        for index, lane in enumerate(lanes[: max(1, self.xlen // lane_bits)]):
            value |= (int(lane) & mask) << (index * lane_bits)
        return _bits(value, self.xlen)

    def _packed_binary(
        self,
        left: int,
        right: int,
        lane_bits: int,
        operation: str,
        *,
        signed: bool = False,
        saturating: bool = False,
    ) -> int:
        """Apply a P-style operation independently to packed subwords."""

        left_lanes = self._packed_lanes(left, lane_bits)
        right_lanes = self._packed_lanes(right, lane_bits)
        mask = (1 << lane_bits) - 1
        result: list[int] = []
        lo = -(1 << (lane_bits - 1))
        hi = (1 << (lane_bits - 1)) - 1
        for lhs, rhs in zip(left_lanes, right_lanes):
            lhs_value = _signed(lhs, lane_bits) if signed else lhs
            rhs_value = _signed(rhs, lane_bits) if signed else rhs
            if operation == "add":
                value = lhs_value + rhs_value
            elif operation == "sub":
                value = lhs_value - rhs_value
            elif operation == "min":
                value = min(lhs_value, rhs_value)
            elif operation == "max":
                value = max(lhs_value, rhs_value)
            elif operation == "sll":
                value = lhs << (rhs & (lane_bits - 1))
            elif operation == "srl":
                value = lhs >> (rhs & (lane_bits - 1))
            elif operation == "sra":
                value = lhs_value >> (rhs & (lane_bits - 1))
            elif operation == "xor":
                value = lhs ^ rhs
            elif operation == "or":
                value = lhs | rhs
            elif operation == "and":
                value = lhs & rhs
            elif operation == "mul":
                value = lhs_value * rhs_value if signed else lhs * rhs
            else:
                raise SemanticUnsupported(f"packed operation {operation}")
            if saturating:
                value = max(lo, min(hi, value)) if signed else max(0, min(mask, value))
            result.append(value & mask)
        return self._pack_lanes(result, lane_bits)

    def _packed_shift(self, value: int, amount: int, lane_bits: int, operation: str) -> int:
        lanes = self._packed_lanes(value, lane_bits)
        mask = (1 << lane_bits) - 1
        shift = int(amount) & (lane_bits - 1)
        if operation == "sll":
            result = [(lane << shift) & mask for lane in lanes]
        elif operation == "srl":
            result = [lane >> shift for lane in lanes]
        elif operation == "sra":
            result = [_signed(lane, lane_bits) >> shift for lane in lanes]
        else:
            raise SemanticUnsupported(f"packed shift {operation}")
        return self._pack_lanes(result, lane_bits)

    def _stack_registers(self, rlist: int) -> tuple[int, ...]:
        """Return the Zcmp register list used by the generated witnesses."""

        registers = STACK_RLIST_REGISTERS_RV64.get(int(rlist))
        if registers is None:
            raise SemanticUnsupported(f"Zcmp register list {rlist}")
        return tuple(registers)

    def _stack_transfer(self, item: object, form: str) -> bool:
        rlist = _field(item, "c_rlist")
        spimm = _field(item, "c_spimm")
        if rlist is None or spimm is None:
            raise SemanticUnsupported(f"Zcmp operands for {form}")
        registers = self._stack_registers(int(rlist))
        adjustments = STACK_ADJ_RV32 if self.xlen == 32 else STACK_ADJ_RV64
        choices = adjustments.get(int(rlist))
        if choices is None or not 0 <= int(spimm) < len(choices):
            raise SemanticUnsupported(f"Zcmp stack shape {rlist}/{spimm}")
        stack_adj = int(choices[int(spimm)])
        old_sp = self._read_gpr(2)
        width = self.xlen // 8
        if form == "cm.push":
            new_sp = _bits(old_sp - stack_adj, self.xlen)
            offsets = stack_slot_offsets(registers, stack_adj, xlen=self.xlen)
            for register in registers:
                self._store(new_sp + offsets[register], width, self._read_gpr(register))
        else:
            new_sp = _bits(old_sp + stack_adj, self.xlen)
            offsets = stack_slot_offsets(registers, stack_adj, xlen=self.xlen)
            for register in registers:
                self._write_gpr(register, self._load(old_sp + offsets[register], width))
            if form == "cm.popretz":
                self._write_gpr(1, 0)
        self._write_gpr(2, new_sp)
        return True

    # ----- integer and CSR operations -------------------------------------

    def _integer(self, item: object, form: str) -> bool:
        if p_form_uses_vxsat(form):
            self._check_vxsat_access(item, form)
        rd = getattr(item, "rd", None)
        rs1 = self._read_gpr(getattr(item, "rs1", None)) if getattr(item, "rs1", None) is not None else 0
        rs2 = self._read_gpr(getattr(item, "rs2", None)) if getattr(item, "rs2", None) is not None else 0
        imm = _immediate(item)
        word = form in {
            "addiw", "c.addiw", "addw", "subw", "sllw", "srlw", "sraw",
            "slliw", "srliw", "sraiw", "mulw", "divw", "divuw",
            "remw", "remuw", "clzw", "ctzw", "cpopw",
            "rolw", "rorw", "roriw", "absw", "c.addw", "c.subw",
        }
        width = 32 if word else self.xlen
        mask = (1 << width) - 1
        signed_rs1, signed_rs2 = _signed(rs1, width), _signed(rs2, width)
        result: int | None = None
        if form in {"cm.mva01s", "cm.mvsa01"}:
            sreg1 = int(_field(item, "c_sreg1", 0) or 0)
            sreg2 = int(_field(item, "c_sreg2", 1) or 1)
            if form == "cm.mva01s":
                self._write_gpr(10, self._read_gpr(8 + sreg1))
                self._write_gpr(11, self._read_gpr(8 + sreg2))
            else:
                self._write_gpr(8 + sreg1, self._read_gpr(10))
                self._write_gpr(8 + sreg2, self._read_gpr(11))
            return True
        elif form in {"cm.push", "cm.pop", "cm.popret", "cm.popretz"}:
            return self._stack_transfer(item, form)
        elif form.startswith(("mop.r.", "mop.rr.")):
            # Zimop explicitly permits an implementation to treat these
            # encodings as a no-op.  The generator's value lane uses the
            # architecturally defined write-zero choice.
            result = 0
        elif form in {"pli.b", "pli.h", "pli.w", "pli.d", "pli.db", "pli.dh", "pli.dw"}:
            bits = 8 if ".b" in form else 16 if ".h" in form else 32 if ".w" in form else self.xlen
            result = _signed(imm, bits)
        elif form in {"plui.w", "plui.h", "plui.dh"}:
            bits = 16 if form.endswith(".h") or form.endswith(".dh") else 32
            result = _bits(imm, bits) << (bits // 2)
        elif form in {
            "nclip", "nclipi", "nclipiu", "nclipr", "nclipri", "nclipriu", "nclipru", "nclipu",
            "pnclip.bs", "pnclip.hs", "pnclipi.b", "pnclipi.h",
            "pnclipiu.b", "pnclipiu.h", "pnclipr.bs", "pnclipr.hs",
            "pnclipri.b", "pnclipri.h", "pnclipriu.b", "pnclipriu.h",
            "pnclipru.bs", "pnclipru.hs", "pnclipu.bs", "pnclipu.hs",
            "pnclipp.b", "pnclipp.h", "pnclipp.w",
            "pnclipup.b", "pnclipup.h", "pnclipup.w",
        }:
            operation, _, suffix = form.partition(".")
            pair_input = operation.startswith("nclip") or operation in {
                "pnclip", "pnclipi", "pnclipiu", "pnclipr", "pnclipri", "pnclipriu", "pnclipru", "pnclipu",
            }
            wide_pair_input = operation in {"pnclipp", "pnclipup"}
            unsigned = operation in {
                "nclipiu", "nclipriu", "nclipru", "nclipu",
                "pnclipiu", "pnclipriu", "pnclipru", "pnclipu", "pnclipup",
            }
            rounded = operation in {
                "nclipr", "nclipri", "nclipriu", "nclipru",
                "pnclipr", "pnclipri", "pnclipriu", "pnclipru",
            }
            immediate_shift = operation in {
                "nclipi", "nclipiu", "nclipri", "nclipriu",
                "pnclipi", "pnclipiu", "pnclipri", "pnclipriu",
            }

            if operation.startswith("nclip"):
                narrow_bits = 32
                source_bits = 64
            else:
                suffix_code = suffix[0] if suffix else ""
                narrow_bits = {"b": 8, "h": 16, "w": 32}[suffix_code]
                source_bits = narrow_bits * 2

            if wide_pair_input:
                if self.xlen != 64:
                    raise SemanticUnsupported(f"{form} requires RV64 128-bit register-pair input")
                source = (rs2 << 64) | rs1
                total_bits = 128
            elif pair_input:
                if self.xlen != 32:
                    raise SemanticUnsupported(f"{form} requires RV32 register-pair input")
                pair_index = _field(item, "p_rs1_p", getattr(item, "rs1", None))
                if pair_index is None or not 0 <= int(pair_index) < 16:
                    raise SemanticUnsupported(f"missing P register-pair source for {form}")
                pair_index = int(pair_index)
                source = 0 if pair_index == 0 else (
                    self._read_gpr(pair_index * 2)
                    | (self._read_gpr(pair_index * 2 + 1) << 32)
                )
                total_bits = 64
            else:
                raise SemanticUnsupported(f"invalid P narrowing clip source for {form}")

            if operation in {
                "nclip", "nclipu", "nclipr", "nclipru",
                "pnclip", "pnclipu", "pnclipr", "pnclipru",
            }:
                shift_bits = 6 if operation.startswith("nclip") else 5
                shift = (rs2 & ((1 << shift_bits) - 1))
            elif immediate_shift:
                shift_bits = 6 if operation.startswith("nclip") else (4 if narrow_bits == 8 else 5)
                field_name = f"p_w_uimm{shift_bits}"
                shift = _field(item, field_name)
                if shift is None:
                    shift = getattr(item, "immediate", None)
                if shift is None:
                    raise SemanticUnsupported(f"missing shift immediate for {form}")
                if not 0 <= int(shift) < (1 << shift_bits):
                    raise SemanticUnsupported(f"invalid shift immediate for {form}")
                shift = int(shift)
            elif wide_pair_input:
                shift = 0
            else:
                raise SemanticUnsupported(f"missing shift form for {form}")

            output = 0
            saturated = False
            lane_mask = (1 << narrow_bits) - 1
            lower = 0 if unsigned else -(1 << (narrow_bits - 1))
            upper = lane_mask if unsigned else (1 << (narrow_bits - 1)) - 1
            for index, offset in enumerate(range(0, total_bits, source_bits)):
                source_lane = (source >> offset) & ((1 << source_bits) - 1)
                value = source_lane if unsigned else _signed(source_lane, source_bits)
                if rounded and shift:
                    # The pinned P proposal specifies round-to-nearest, ties up
                    # for these forms; unlike RVV vnclip, they do not read VXRM.
                    value = (value + (1 << (shift - 1))) >> shift
                elif shift:
                    value = value >> shift
                clipped = max(lower, min(upper, value))
                saturated |= clipped != value
                output |= (clipped & lane_mask) << (index * narrow_bits)

            if saturated:
                self._set_vxsat(item, form)
            result = output
        elif form.startswith(("nsra", "nsrl", "pnsra", "pnsrl")):
            if form.startswith(("nsrar", "nsrlr", "pnsrar", "pnsrlr")):
                raise SemanticUnsupported(f"rounded packed shift semantics are not modeled for {form}")
            packed = form.startswith("pns")
            arithmetic = form.startswith(("nsra", "pnsra"))
            shift = imm if "i" in form else rs2
            if packed:
                lane_bits = 8 if any(token in form for token in (".b", ".bs", ".db")) else 16 if any(token in form for token in (".h", ".hs", ".dh")) else 32
                result = self._packed_shift(rs1, shift, lane_bits, "sra" if arithmetic else "srl")
            else:
                value = _signed(rs1, self.xlen) if arithmetic else rs1
                result = value >> (int(shift) & (self.xlen - 1))
        elif form.startswith(("psll", "psrl", "psra")):
            operation = "sll" if form.startswith("psll") else "srl" if form.startswith("psrl") else "sra"
            lane_bits = 8 if any(token in form for token in (".b", ".bs", ".db")) else 16 if any(token in form for token in (".h", ".hs", ".dh")) else 32 if any(token in form for token in (".w", ".ws", ".dw")) else 64
            shift = imm if "i" in form else rs2
            result = self._packed_shift(rs1, shift, lane_bits, operation)
        elif form == "orc.b":
            result = 0
            for offset in range(0, self.xlen, 8):
                byte = (rs1 >> offset) & 0xFF
                result |= (0xFF if byte else 0) << offset
        elif form in {"abs", "absw"}:
            result = abs(signed_rs1)
        elif form in {
            "psh1add.h", "psh1add.w", "psh1add.dh", "psh1add.dw",
            "pssh1sadd.h", "pssh1sadd.w", "pssh1sadd.dh", "pssh1sadd.dw",
            "psslai.h", "psslai.w", "psslai.dh", "psslai.dw",
        }:
            operation, suffix = form.rsplit(".", 1)
            paired = suffix.startswith("d")
            lane_bits = {"h": 16, "w": 32}[suffix[-1]]
            total_bits = 64 if paired else self.xlen
            lane_mask = (1 << lane_bits) - 1

            def pair_value(register: int | None) -> int:
                if register is None or not 0 <= int(register) < 16:
                    raise SemanticUnsupported(f"missing P register-pair operand for {form}")
                pair = int(register)
                if pair == 0:
                    return 0
                return (self._read_gpr(pair * 2 + 1) << 32) | self._read_gpr(pair * 2)

            left = pair_value(getattr(item, "rs1", None)) if paired else rs1
            left_lanes = [
                (left >> offset) & lane_mask
                for offset in range(0, total_bits, lane_bits)
            ]
            saturated = False
            if operation == "psslai":
                immediate_bits = 4 if lane_bits == 16 else 5
                shift = _field(item, f"p_w_uimm{immediate_bits}")
                if shift is None:
                    shift = getattr(item, "immediate", None)
                if shift is None or not 0 <= int(shift) < (1 << immediate_bits):
                    raise SemanticUnsupported(f"invalid packed shift immediate for {form}")
                shifted = []
                lower = -(1 << (lane_bits - 1))
                upper = (1 << (lane_bits - 1)) - 1
                for lane in left_lanes:
                    raw = _signed(lane, lane_bits) << int(shift)
                    value = max(lower, min(upper, raw))
                    saturated |= value != raw
                    shifted.append(value & lane_mask)
                output = sum(value << (index * lane_bits) for index, value in enumerate(shifted))
            else:
                right = pair_value(getattr(item, "rs2", None)) if paired else rs2
                right_lanes = [
                    (right >> offset) & lane_mask
                    for offset in range(0, total_bits, lane_bits)
                ]
                output = 0
                lower = -(1 << (lane_bits - 1))
                upper = (1 << (lane_bits - 1)) - 1
                for index, (lhs, rhs) in enumerate(zip(left_lanes, right_lanes)):
                    if operation == "psh1add":
                        value = ((lhs << 1) + rhs) & lane_mask
                    else:
                        shifted = _signed(lhs, lane_bits) << 1
                        clamped_shift = max(lower, min(upper, shifted))
                        saturated |= clamped_shift != shifted
                        added = clamped_shift + _signed(rhs, lane_bits)
                        value = max(lower, min(upper, added))
                        saturated |= value != added
                        value &= lane_mask
                    output |= value << (index * lane_bits)

            if saturated:
                self._set_vxsat(item, form)
            if paired:
                destination = getattr(item, "rd", None)
                if destination is None or not 0 <= int(destination) < 16:
                    raise SemanticUnsupported(f"missing P register-pair destination for {form}")
                if int(destination) != 0:
                    pair = int(destination)
                    self._write_gpr(pair * 2, output)
                    self._write_gpr(pair * 2 + 1, output >> 32)
                return True
            result = output
        elif form in {
            "pmulh.h", "pmulh.h.b0", "pmulh.h.b1",
            "pmulh.w", "pmulh.w.h0", "pmulh.w.h1",
            "pmulhsu.h", "pmulhsu.h.b0", "pmulhsu.h.b1",
            "pmulhsu.w", "pmulhsu.w.h0", "pmulhsu.w.h1",
            "pmulhu.h", "pmulhu.w",
        }:
            parts = form.split(".")
            operation, lane_code = parts[0], parts[1]
            lane_bits = {"h": 16, "w": 32}[lane_code]
            selector = parts[2] if len(parts) == 3 else None
            if selector is None:
                source_bits = lane_bits
                source_offset = 0
            elif selector in {"b0", "b1"} and lane_bits == 16:
                source_bits = 8
                source_offset = 8 if selector == "b1" else 0
            elif selector in {"h0", "h1"} and lane_bits == 32:
                source_bits = 16
                source_offset = 16 if selector == "h1" else 0
            else:
                raise SemanticUnsupported(f"mixed-width multiply-high selector {form}")

            left_signed = operation != "pmulhu"
            right_signed = operation == "pmulh"
            lane_mask = (1 << lane_bits) - 1
            source_mask = (1 << source_bits) - 1
            output = 0
            for index, offset in enumerate(range(0, self.xlen, lane_bits)):
                left_lane = (rs1 >> offset) & lane_mask
                right_lane = (rs2 >> offset) & lane_mask
                right_part = (right_lane >> source_offset) & source_mask
                left_value = _signed(left_lane, lane_bits) if left_signed else left_lane
                right_value = _signed(right_part, source_bits) if right_signed else right_part
                product = left_value * right_value
                output_lane = (product >> source_bits) & lane_mask
                output |= output_lane << (index * lane_bits)
            result = output
        elif form in {
            "wadd", "wadda", "waddau", "waddu",
            "wsub", "wsuba", "wsubau", "wsubu",
        } or (
            "." in form
            and form.split(".", 1)[0] in {
                "pwadd", "pwadda", "pwaddu", "pwaddau",
                "pwsub", "pwsuba", "pwsubu", "pwsubau",
            }
        ):
            operation, _, suffix = form.partition(".")

            def destination_pair_value(pair: int | None) -> int:
                if pair is None or not 0 <= int(pair) < 16:
                    raise SemanticUnsupported(f"missing P widening destination pair for {form}")
                pair = int(pair)
                if pair == 0:
                    return 0
                return (self._read_gpr(pair * 2 + 1) << 32) | self._read_gpr(pair * 2)

            destination = getattr(item, "rd", None)
            accumulate = operation in {
                "wadda", "waddau", "wsuba", "wsubau",
                "pwadda", "pwaddau", "pwsuba", "pwsubau",
            }
            accumulator = destination_pair_value(destination) if accumulate else 0
            if operation.startswith("pw"):
                if suffix not in {"b", "h"}:
                    raise SemanticUnsupported(f"packed widening suffix {form}")
                source_bits = {"b": 8, "h": 16}[suffix]
                result_bits = source_bits * 2
                source_mask = (1 << source_bits) - 1
                result_mask = (1 << result_bits) - 1
                output = 0
                unsigned = operation in {"pwaddu", "pwaddau", "pwsubu", "pwsubau"}
                subtract = operation in {"pwsub", "pwsuba", "pwsubu", "pwsubau"}
                for index in range(32 // source_bits):
                    lhs = (rs1 >> (index * source_bits)) & source_mask
                    rhs = (rs2 >> (index * source_bits)) & source_mask
                    if unsigned:
                        value = lhs - rhs if subtract else lhs + rhs
                    else:
                        left_value = _signed(lhs, source_bits)
                        right_value = _signed(rhs, source_bits)
                        value = left_value - right_value if subtract else left_value + right_value
                    value &= result_mask
                    if accumulate:
                        value += (accumulator >> (index * result_bits)) & result_mask
                    output |= (value & result_mask) << (index * result_bits)
            else:
                unsigned = operation in {"waddu", "waddau", "wsubu", "wsubau"}
                subtract = operation.startswith("wsub")
                left = _bits(rs1, 32) if unsigned else _signed(rs1, 32)
                right = _bits(rs2, 32) if unsigned else _signed(rs2, 32)
                raw = left - right if subtract else left + right
                output = _bits(raw + accumulator if accumulate else raw, 64)

            if destination is None or not 0 <= int(destination) < 16:
                raise SemanticUnsupported(f"missing P widening destination pair for {form}")
            if int(destination) != 0:
                pair = int(destination)
                self._write_gpr(pair * 2, output)
                self._write_gpr(pair * 2 + 1, output >> 32)
            return True
        elif "." in form and form.split(".", 1)[0] in {
            "padd", "psub", "psadd", "psaddu", "pssub", "pssubu",
            "paadd", "paaddu", "pasub", "pasubu",
        }:
            operation, suffix = form.split(".", 1)
            if operation == "padd":
                valid_suffixes = {
                    "b", "h", "w", "db", "dh", "dw",
                    "bs", "hs", "ws", "dbs", "dhs", "dws",
                }
            else:
                valid_suffixes = {"b", "h", "w", "db", "dh", "dw"}
            if suffix not in valid_suffixes:
                raise SemanticUnsupported(f"packed arithmetic suffix {form}")

            paired = suffix.startswith("d")
            scalar = suffix.endswith("s")
            if paired:
                lane_code = suffix[1:-1] if scalar else suffix[1:]
            else:
                lane_code = suffix[:-1] if scalar else suffix
            lane_bits = {"b": 8, "h": 16, "w": 32}[lane_code]
            lane_mask = (1 << lane_bits) - 1
            total_bits = 64 if paired else self.xlen

            def pair_value(register: int | None) -> int:
                if register is None or not 0 <= int(register) < 16:
                    raise SemanticUnsupported(f"missing P register-pair operand for {form}")
                pair = int(register)
                if pair == 0:
                    return 0
                return (self._read_gpr(pair * 2 + 1) << 32) | self._read_gpr(pair * 2)

            left = pair_value(getattr(item, "rs1", None)) if paired else rs1
            if scalar:
                scalar_register = getattr(item, "rs2", None)
                if scalar_register is None:
                    raise SemanticUnsupported(f"missing scalar source for {form}")
                scalar_lane = self._read_gpr(int(scalar_register)) & lane_mask
                right_lanes = [scalar_lane] * (total_bits // lane_bits)
            else:
                right = pair_value(getattr(item, "rs2", None)) if paired else rs2
                right_lanes = [
                    (right >> offset) & lane_mask
                    for offset in range(0, total_bits, lane_bits)
                ]

            output = 0
            saturated = False
            lower_signed = -(1 << (lane_bits - 1))
            upper_signed = (1 << (lane_bits - 1)) - 1
            left_lanes = [
                (left >> offset) & lane_mask
                for offset in range(0, total_bits, lane_bits)
            ]
            for index, (lhs, rhs) in enumerate(zip(left_lanes, right_lanes)):
                signed_lhs, signed_rhs = _signed(lhs, lane_bits), _signed(rhs, lane_bits)
                if operation == "padd":
                    value = lhs + rhs
                elif operation == "psub":
                    value = lhs - rhs
                elif operation == "paadd":
                    value = (signed_lhs + signed_rhs) >> 1
                elif operation == "paaddu":
                    value = (lhs + rhs) >> 1
                elif operation == "pasub":
                    value = (signed_lhs - signed_rhs) >> 1
                elif operation == "pasubu":
                    value = (lhs - rhs) >> 1
                elif operation == "psadd":
                    raw = signed_lhs + signed_rhs
                    value = max(lower_signed, min(upper_signed, raw))
                    saturated |= value != raw
                elif operation == "psaddu":
                    raw = lhs + rhs
                    value = min(lane_mask, raw)
                    saturated |= value != raw
                elif operation == "pssub":
                    raw = signed_lhs - signed_rhs
                    value = max(lower_signed, min(upper_signed, raw))
                    saturated |= value != raw
                else:  # PSSUBU clamps an unsigned underflow to zero.
                    raw = lhs - rhs
                    value = max(0, raw)
                    saturated |= value != raw
                output |= (value & lane_mask) << (index * lane_bits)

            if saturated:
                self._set_vxsat(item, form)
            if paired:
                destination = getattr(item, "rd", None)
                if destination is None or not 0 <= int(destination) < 16:
                    raise SemanticUnsupported(f"missing P register-pair destination for {form}")
                if int(destination) != 0:
                    pair = int(destination)
                    self._write_gpr(pair * 2, output)
                    self._write_gpr(pair * 2 + 1, output >> 32)
                return True
            result = output
        elif form.startswith(("pmseq.", "pmslt.", "pmsltu.")) and form.rsplit(".", 1)[1] in {"b", "h", "w"}:
            lane_bits = {"b": 8, "h": 16, "w": 32}[form.rsplit(".", 1)[1]]
            left_lanes = self._packed_lanes(rs1, lane_bits)
            right_lanes = self._packed_lanes(rs2, lane_bits)
            lane_mask = (1 << lane_bits) - 1
            if form.startswith("pmseq."):
                result = self._pack_lanes(
                    [lane_mask if lhs == rhs else 0 for lhs, rhs in zip(left_lanes, right_lanes)],
                    lane_bits,
                )
            elif form.startswith("pmslt."):
                result = self._pack_lanes(
                    [lane_mask if _signed(lhs, lane_bits) < _signed(rhs, lane_bits) else 0
                     for lhs, rhs in zip(left_lanes, right_lanes)],
                    lane_bits,
                )
            else:
                result = self._pack_lanes(
                    [lane_mask if lhs < rhs else 0 for lhs, rhs in zip(left_lanes, right_lanes)],
                    lane_bits,
                )
        elif "." in form and form.split(".", 1)[0] in {
            "pmin", "pminu", "pmax", "pmaxu", "pmseq", "pmslt", "pmsltu",
            "pabd", "pabdu", "psabs",
        } and form.rsplit(".", 1)[1] in {"db", "dh", "dw"}:
            operation, suffix = form.split(".", 1)
            if suffix == "dw" and operation in {"pabd", "pabdu", "psabs"}:
                raise SemanticUnsupported(f"packed pair operation {form}")
            lane_bits = {"db": 8, "dh": 16, "dw": 32}[suffix]

            def pair_value(register: int | None) -> int:
                if register is None or not 0 <= int(register) < 16:
                    raise SemanticUnsupported(f"missing P register-pair operand for {form}")
                pair = int(register)
                if pair == 0:
                    return 0
                return (self._read_gpr(pair * 2 + 1) << 32) | self._read_gpr(pair * 2)

            left = pair_value(getattr(item, "rs1", None))
            right = pair_value(getattr(item, "rs2", None)) if operation != "psabs" else 0
            lane_mask = (1 << lane_bits) - 1
            output = 0
            for offset in range(0, 64, lane_bits):
                lhs = (left >> offset) & lane_mask
                rhs = (right >> offset) & lane_mask
                signed_lhs = _signed(lhs, lane_bits)
                signed_rhs = _signed(rhs, lane_bits)
                if operation == "pmseq":
                    value = lane_mask if lhs == rhs else 0
                elif operation == "pmslt":
                    value = lane_mask if signed_lhs < signed_rhs else 0
                elif operation == "pmsltu":
                    value = lane_mask if lhs < rhs else 0
                elif operation in {"pmin", "pmax", "pminu", "pmaxu"}:
                    unsigned = operation.endswith("u")
                    a, b = (lhs, rhs) if unsigned else (signed_lhs, signed_rhs)
                    value = min(a, b) if operation.startswith("pmin") else max(a, b)
                elif operation in {"pabd", "pabdu"}:
                    a, b = (lhs, rhs) if operation == "pabdu" else (signed_lhs, signed_rhs)
                    value = abs(a - b)
                else:  # PSABS saturates abs(min_signed) to max_signed.
                    minimum = -(1 << (lane_bits - 1))
                    if signed_lhs == minimum:
                        self._set_vxsat(item, form)
                    value = min((1 << (lane_bits - 1)) - 1, abs(signed_lhs))
                output |= (value & lane_mask) << offset

            destination = getattr(item, "rd", None)
            if destination is None or not 0 <= int(destination) < 16:
                raise SemanticUnsupported(f"missing P register-pair destination for {form}")
            if int(destination) != 0:
                pair = int(destination)
                self._write_gpr(pair * 2, output)
                self._write_gpr(pair * 2 + 1, output >> 32)
            return True
        elif form in _P_PACKED_BASIC_FORMS:
            lane_bits = {"b": 8, "h": 16, "w": 32}[form.rsplit(".", 1)[1]]
            if form.startswith("psabs"):
                lanes = self._packed_lanes(rs1, lane_bits)
                maximum = (1 << (lane_bits - 1)) - 1
                if any(_signed(value, lane_bits) == -(1 << (lane_bits - 1)) for value in lanes):
                    self._set_vxsat(item, form)
                result = self._pack_lanes(
                    [min(maximum, abs(_signed(value, lane_bits))) for value in lanes],
                    lane_bits,
                )
            elif form.startswith(("pabd", "pabdu")):
                left_lanes = self._packed_lanes(rs1, lane_bits)
                right_lanes = self._packed_lanes(rs2, lane_bits)
                result = self._pack_lanes([
                    abs(lhs - rhs) if form.startswith("pabdu") else abs(_signed(lhs, lane_bits) - _signed(rhs, lane_bits))
                    for lhs, rhs in zip(left_lanes, right_lanes)
                ], lane_bits)
            else:
                operation = (
                    "add" if form.startswith(("padd", "psadd", "psaddu"))
                    else "sub" if form.startswith(("psub", "pssub", "pssubu"))
                    else "min" if form.startswith("pmin") else "max"
                )
                unsigned = "u" in form.split(".", 1)[0]
                result = self._packed_binary(
                    rs1, rs2, lane_bits, operation, signed=not unsigned,
                    saturating=form.startswith(("psadd", "psaddu", "pssub", "pssubu")),
                )
        elif form.startswith("psext."):
            parts = form.split(".")[-2:]
            source_bits = {"b": 8, "h": 16, "w": 32}.get(parts[1])
            target_bits = {"b": 8, "h": 16, "w": 32, "d": 64, "dh": 32, "dw": 64}.get(parts[0])
            if source_bits is None or target_bits is None or source_bits >= target_bits:
                raise SemanticUnsupported(f"packed extension {form}")
            source = _signed(rs1, source_bits)
            result = _bits(source, target_bits)
        elif form.startswith(("predsum", "predsumu")):
            lane_bits = 8 if ".b" in form else 16 if ".h" in form else 32
            lanes = self._packed_lanes(rs1, lane_bits)
            total = sum(lanes if form.startswith("predsumu") else (_signed(value, lane_bits) for value in lanes))
            result = _bits(total + rs2, self.xlen)
        elif form.startswith(("psati", "pusati", "sati", "usati")):
            shift = int(imm) & (31 if self.xlen == 32 else 63)
            unsigned = form.startswith(("pusa", "usa"))
            limit = (1 << shift) - 1 if unsigned else (1 << max(0, shift - 1)) - 1
            lower = 0 if unsigned else -(1 << max(0, shift - 1))
            result = max(lower, min(limit, _signed(rs1, self.xlen)))
        elif form in {"zip", "unzip"}:
            half = self.xlen // 2
            if form == "zip":
                result = sum(((rs1 >> index) & 1) << (2 * index) for index in range(half))
                result |= sum(((rs1 >> (half + index)) & 1) << (2 * index + 1) for index in range(half))
            else:
                result = sum(((rs1 >> (2 * index)) & 1) << index for index in range(half))
                result |= sum(((rs1 >> (2 * index + 1)) & 1) << (half + index) for index in range(half))
        elif form in {"sha", "shar"}:
            raise SemanticUnsupported(f"experimental P form {form} is not modeled")
        elif form in {"add", "c.add"}:
            result = rs1 + rs2
        elif form in {"addi", "c.addi", "c.addi16sp", "c.addi4spn"}:
            result = rs1 + imm
        elif form in {"addiw", "c.addiw", "addw", "c.addw"}:
            result = _signed(
                rs1 + (imm if form in {"addiw", "c.addiw"} else rs2), 32,
            )
        elif form in {"sub", "c.sub", "subw", "c.subw"}:
            value = rs1 - rs2
            result = _signed(value, 32) if word else value
        elif form in {"and", "c.and", "andi", "c.andi"}:
            result = rs1 & (imm if form.endswith("i") else rs2)
        elif form in {"or", "c.or", "ori"}:
            result = rs1 | (imm if form == "ori" else rs2)
        elif form in {"xor", "c.xor", "xori"}:
            result = rs1 ^ (imm if form == "xori" else rs2)
        elif form in {"sll", "c.slli", "slli", "sllw", "slliw", "c.sll"}:
            result = rs1 << ((imm if "i" in form else rs2) & (width - 1))
        elif form in {"srl", "c.srli", "srli", "srlw", "srliw"}:
            result = (rs1 & mask) >> ((imm if "i" in form else rs2) & (width - 1))
        elif form in {"sra", "c.srai", "srai", "sraw", "sraiw"}:
            result = signed_rs1 >> ((imm if "i" in form else rs2) & (width - 1))
        elif form in {"slt", "c.slt"}:
            result = int(signed_rs1 < signed_rs2)
        elif form == "slti":
            result = int(signed_rs1 < imm)
        elif form == "sltu":
            result = int(rs1 < rs2)
        elif form == "sltiu":
            result = int(rs1 < _bits(imm, self.xlen))
        elif form in {"lui", "c.lui"}:
            # InstructionMeta.immediate is already the semantic U-immediate
            # (sign-extended imm20 << 12 or the scattered C.LUI immediate).
            result = imm
        elif form == "auipc":
            result = int(getattr(self.testcase, "code_address", 0)) + int(item.pc_offset) + imm
        elif form in {"mv", "c.mv", "c.li", "c.nop", "nop"}:
            result = (
                rs2 if form == "c.mv"
                else rs1 if form == "mv"
                else imm if form == "c.li"
                else None
            )
        elif form in {"clz", "clzw"}:
            value = rs1 & mask
            result = width if value == 0 else width - value.bit_length()
        elif form in {"ctz", "ctzw"}:
            value = rs1 & mask
            result = width if value == 0 else (value & -value).bit_length() - 1
        elif form in {"cpop", "cpopw"}:
            result = (rs1 & mask).bit_count()
        elif form in {"min", "minu", "max", "maxu"}:
            result = min(signed_rs1, signed_rs2) if form == "min" else max(signed_rs1, signed_rs2) if form == "max" else min(rs1, rs2) if form == "minu" else max(rs1, rs2)
        elif form in {"andn", "orn", "xnor"}:
            result = rs1 & ~rs2 if form == "andn" else rs1 | ~rs2 if form == "orn" else ~(rs1 ^ rs2)
        elif form in {"c.not", "c.zext.b", "c.zext.h", "c.zext.w"}:
            bits = 8 if form.endswith(".b") else 16 if form.endswith(".h") else 32
            result = ~rs1 if form == "c.not" else rs1 & ((1 << bits) - 1)
        elif form in {"czero.eqz", "czero.nez"}:
            zero = rs2 == 0
            result = rs1 if zero == form.endswith(".eqz") else 0
        elif form.startswith("aes32"):
            byte_select = _field(item, "bs")
            if byte_select is None or not 0 <= int(byte_select) < 4:
                raise SemanticUnsupported(f"{form} byte selector")
            shift = int(byte_select) * 8
            source_byte = (rs2 >> shift) & 0xFF
            inverse = "d" in form[5:7]
            sbox = _AES_INV_SBOX if inverse else _AES_SBOX
            substituted = sbox[source_byte]
            mixed = (
                _aes_mixcolumn_byte(substituted, inverse)
                if form.endswith("smi") else substituted
            )
            result = _signed(
                (rs1 & 0xFFFFFFFF) ^ _rotate_left(mixed, shift, 32),
                32,
            )
        elif form in {"aes64es", "aes64esm", "aes64ds", "aes64dsm"}:
            inverse = form.startswith("aes64d")
            shifted = _aes_rv64_shiftrows(rs2, rs1, inverse=inverse)
            sbox = _AES_INV_SBOX if inverse else _AES_SBOX
            substituted = sum(
                sbox[(shifted >> offset) & 0xFF] << offset
                for offset in range(0, 64, 8)
            )
            if form.endswith(("esm", "dsm")):
                low = _aes_mixcolumn_word(substituted & 0xFFFFFFFF, inverse)
                high = _aes_mixcolumn_word(substituted >> 32, inverse)
                result = (high << 32) | low
            else:
                result = substituted
        elif form == "aes64im":
            low = _aes_mixcolumn_word(rs1 & 0xFFFFFFFF, True)
            high = _aes_mixcolumn_word(rs1 >> 32, True)
            result = (high << 32) | low
        elif form == "aes64ks1i":
            round_number = int(_field(item, "rnum", _immediate(item)))
            if not 0 <= round_number <= 10:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            rcon = (1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36, 0)
            high_word = (rs1 >> 32) & 0xFFFFFFFF
            rotated = high_word if round_number == 10 else _rotate_right(high_word, 8, 32)
            word = _aes_subword(rotated) ^ rcon[round_number]
            result = (word << 32) | word
        elif form == "aes64ks2":
            low_word = ((rs1 >> 32) ^ rs2) & 0xFFFFFFFF
            high_word = low_word ^ ((rs2 >> 32) & 0xFFFFFFFF)
            result = (high_word << 32) | low_word
        elif form in {"sm4ed", "sm4ks"}:
            byte_select = _field(item, "bs")
            if byte_select is None or not 0 <= int(byte_select) < 4:
                raise SemanticUnsupported(f"{form} byte selector")
            shift = int(byte_select) * 8
            source_byte = (rs2 >> shift) & 0xFF
            substituted = _SM4_SBOX[source_byte]
            if form == "sm4ed":
                mixed = (
                    substituted ^ (substituted << 8) ^ (substituted << 2)
                    ^ (substituted << 18) ^ ((substituted & 0x3F) << 26)
                    ^ ((substituted & 0xC0) << 10)
                )
            else:
                mixed = (
                    substituted ^ ((substituted & 0x7) << 29)
                    ^ ((substituted & 0xFE) << 7) ^ ((substituted & 1) << 23)
                    ^ ((substituted & 0xF8) << 13)
                )
            result = _signed(
                (rs1 & 0xFFFFFFFF) ^ _rotate_left(mixed, shift, 32),
                32,
            )
        elif form in {
            "sha256sig0", "sha256sig1", "sha256sum0", "sha256sum1",
            "sha512sig0h", "sha512sig0l", "sha512sig1h", "sha512sig1l",
            "sha512sum0r", "sha512sum1r", "sha512sig0", "sha512sig1",
            "sha512sum0", "sha512sum1", "sm3p0", "sm3p1",
        }:
            value = rs1 & 0xFFFFFFFF
            if form == "sha256sig0":
                result = _signed(_rotate_right(value, 7, 32) ^ _rotate_right(value, 18, 32) ^ (value >> 3), 32)
            elif form == "sha256sig1":
                result = _signed(_rotate_right(value, 17, 32) ^ _rotate_right(value, 19, 32) ^ (value >> 10), 32)
            elif form == "sha256sum0":
                result = _signed(_rotate_right(value, 2, 32) ^ _rotate_right(value, 13, 32) ^ _rotate_right(value, 22, 32), 32)
            elif form == "sha256sum1":
                result = _signed(_rotate_right(value, 6, 32) ^ _rotate_right(value, 11, 32) ^ _rotate_right(value, 25, 32), 32)
            elif form == "sm3p0":
                result = _signed(value ^ _rotate_left(value, 9, 32) ^ _rotate_left(value, 17, 32), 32)
            elif form == "sm3p1":
                result = _signed(value ^ _rotate_left(value, 15, 32) ^ _rotate_left(value, 23, 32), 32)
            elif form in {"sha512sig0h", "sha512sig0l", "sha512sig1h", "sha512sig1l", "sha512sum0r", "sha512sum1r"}:
                other = rs2 & 0xFFFFFFFF
                if form == "sha512sig0h":
                    result = (value >> 1) ^ (value >> 7) ^ (value >> 8) ^ (other << 31) ^ (other << 24)
                elif form == "sha512sig0l":
                    result = (value >> 1) ^ (value >> 7) ^ (value >> 8) ^ (other << 31) ^ (other << 25) ^ (other << 24)
                elif form == "sha512sig1h":
                    result = (value << 3) ^ (value >> 6) ^ (value >> 19) ^ (other >> 29) ^ (other << 13)
                elif form == "sha512sig1l":
                    result = (value << 3) ^ (value >> 6) ^ (value >> 19) ^ (other >> 29) ^ (other << 26) ^ (other << 13)
                elif form == "sha512sum0r":
                    result = (value << 25) ^ (value << 30) ^ (value >> 28) ^ (other >> 7) ^ (other >> 2) ^ (other << 4)
                else:
                    result = (value << 23) ^ (value >> 14) ^ (value >> 18) ^ (other >> 9) ^ (other << 18) ^ (other << 14)
                result = _signed(result, 32)
            else:
                value64 = rs1 & 0xFFFFFFFFFFFFFFFF
                if form == "sha512sig0":
                    result = _rotate_right(value64, 1, 64) ^ _rotate_right(value64, 8, 64) ^ (value64 >> 7)
                elif form == "sha512sig1":
                    result = _rotate_right(value64, 19, 64) ^ _rotate_right(value64, 61, 64) ^ (value64 >> 6)
                elif form == "sha512sum0":
                    result = _rotate_right(value64, 28, 64) ^ _rotate_right(value64, 34, 64) ^ _rotate_right(value64, 39, 64)
                else:
                    result = _rotate_right(value64, 14, 64) ^ _rotate_right(value64, 18, 64) ^ _rotate_right(value64, 41, 64)
        elif form in {"sext.b", "sext.h", "zext.b", "zext.h", "c.sext.b", "c.sext.h", "c.zext.b", "c.zext.h", "c.zext.w"}:
            bits = 8 if form.endswith(".b") else 16 if form.endswith(".h") else 32
            result = _signed(rs1, bits) if form.startswith("sext") or ".sext" in form else rs1 & ((1 << bits) - 1)
        elif form in {"sh1add", "sh2add", "sh3add", "sh1add.uw", "sh2add.uw", "sh3add.uw"}:
            shift = int(form[2])
            left = rs1 & 0xFFFFFFFF if form.endswith(".uw") else rs1
            result = (left << shift) + rs2
        elif form in {"add.uw", "slli.uw"}:
            result = (
                (rs1 & 0xFFFFFFFF) + rs2
                if form == "add.uw"
                else (rs1 & 0xFFFFFFFF) << (imm & (self.xlen - 1))
            )
        elif form in {"mul", "c.mul", "mulw", "mulh", "mulhsu", "mulhu", "div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}:
            if form.startswith("mul"):
                if form == "mulh":
                    result = (_signed(rs1, width) * _signed(rs2, width)) >> width
                elif form == "mulhsu":
                    result = (_signed(rs1, width) * rs2) >> width
                elif form == "mulhu":
                    result = (rs1 * rs2) >> width
                else:
                    result = rs1 * rs2
            else:
                unsigned = form.startswith(("divu", "remu"))
                dividend = _bits(rs1, width) if unsigned else _signed(rs1, width)
                divisor = _bits(rs2, width) if unsigned else _signed(rs2, width)
                is_remainder = form.startswith("rem")
                if divisor == 0:
                    result = (dividend if is_remainder else mask)
                elif dividend == -(1 << (width - 1)) and divisor == -1:
                    result = 0 if is_remainder else dividend
                else:
                    quotient = abs(dividend) // abs(divisor)
                    if (dividend < 0) != (divisor < 0):
                        quotient = -quotient
                    result = dividend - quotient * divisor if is_remainder else quotient
            if form.endswith("w"):
                result = _signed(result, 32)
        elif form in {"bset", "bseti", "bclr", "bclri", "binv", "binvi", "bext", "bexti"}:
            bit = (imm if form.endswith("i") else rs2) & (width - 1)
            if form.startswith("bset"):
                result = rs1 | (1 << bit)
            elif form.startswith("bclr"):
                result = rs1 & ~(1 << bit)
            elif form.startswith("binv"):
                result = rs1 ^ (1 << bit)
            else:
                result = (rs1 >> bit) & 1
        elif form in {"rol", "ror", "rori", "rolw", "rorw", "roriw"}:
            amount = (imm if "i" in form else rs2) & (width - 1)
            lhs = rs1 & mask
            result = (
                (lhs << amount) | (lhs >> ((width - amount) & (width - 1)))
                if form.startswith("rol") else
                (lhs >> amount) | (lhs << ((width - amount) & (width - 1)))
            )
        elif form in {"rev8", "rev8.rv32"}:
            result = int.from_bytes((rs1 & mask).to_bytes(width // 8, "little"), "big")
        elif form == "brev8":
            result = sum(
                _BIT_REVERSE_BYTE[(rs1 >> offset) & 0xFF] << offset
                for offset in range(0, width, 8)
            )
        elif form == "packw":
            packed = (rs1 & 0xFFFF) | ((rs2 & 0xFFFF) << 16)
            result = _signed(packed, 32)
        elif form in {"pack", "packh"}:
            half = width // 2
            if form == "packh":
                result = ((rs1 & 0xFF) | ((rs2 & 0xFF) << 8))
            else:
                result = (rs1 & ((1 << half) - 1)) | ((rs2 & ((1 << half) - 1)) << half)
        elif form in {"clmul", "clmulh", "clmulr"}:
            product = 0
            for bit in range(width):
                if (rs2 >> bit) & 1:
                    product ^= rs1 << bit
            result = product if form == "clmul" else product >> (width if form == "clmulh" else width - 1)
        elif form in {"xperm4", "xperm8", "xperm16", "xperm32"}:
            chunk = int(form.removeprefix("xperm"))
            slots = width // chunk
            selector_mask = (1 << chunk) - 1
            result = 0
            for index in range(slots):
                selector = (rs2 >> (index * chunk)) & selector_mask
                if selector < slots:
                    result |= ((rs1 >> (selector * chunk)) & selector_mask) << (index * chunk)
        else:
            return False
        if result is not None:
            self._write_gpr(rd, result if not word else _signed(result, 32))
        return True

    def _memory_or_atomic(self, item: object, form: str) -> bool:
        if form == "amocas.q" or (form == "amocas.d" and self.xlen == 32):
            instruction = _instruction_word(self.testcase, item)
            pair_width = 16 if form == "amocas.q" else 8
            required_xlen = 64 if form == "amocas.q" else 32
            if self.xlen != required_xlen or "zacas" not in self.extensions:
                raise SemanticTrap(2, int(item.pc_offset), instruction)
            rd = _field(item, "rd")
            rs2 = _field(item, "rs2")
            if rd is None or rs2 is None:
                raise SemanticUnsupported("AMOCAS.Q requires register-pair operands")
            rd, rs2 = int(rd), int(rs2)
            if rd & 1 or rs2 & 1:
                raise SemanticTrap(2, int(item.pc_offset), instruction)

            address = self._address(item)
            if address % pair_width:
                raise SemanticTrap(6, int(item.pc_offset), address)
            self._require_atomic_read_write(address, pair_width, form)

            # Zacas uses rd/rd+1 and rs2/rs2+1 for AMOCAS.Q on RV64 and
            # AMOCAS.D on RV32. x0 is a zero-valued source pair and discards
            # both result halves when it is the destination base.
            compare = (
                (self._read_gpr(rd + 1) << self.xlen) | self._read_gpr(rd)
                if rd != 0 else 0
            )
            replacement = (
                (self._read_gpr(rs2 + 1) << self.xlen) | self._read_gpr(rs2)
                if rs2 != 0 else 0
            )
            old = self._load(address, pair_width)
            if old == compare:
                self._store(address, pair_width, replacement)
            # A failed CAS is modeled as no store, one outcome permitted by Zacas.
            if rd != 0:
                self._write_gpr(rd, old & self.mask)
                self._write_gpr(rd + 1, old >> self.xlen)
            return True
        if form.endswith((".aq", ".rl")) and form.split(".", 1)[0] in {
            "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "sb", "sh", "sw", "sd",
        }:
            return self._memory_or_atomic(item, form.split(".", 1)[0])
        if form in {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.lbu", "c.lh", "c.lhu", "c.lw", "c.lwsp", "c.ld", "c.ldsp"}:
            width = 1 if form.endswith("b") or form.endswith("bu") else 2 if form.endswith("h") else 4 if form.endswith("w") else 8
            value = self._load(self._address(item), width, signed=form in {"lb", "lh", "lw", "c.lh", "c.lw", "c.lwsp"})
            self._write_gpr(getattr(item, "rd", None), value)
            return True
        if form in {"sb", "sh", "sw", "sd", "c.sb", "c.sh", "c.sw", "c.swsp", "c.sd", "c.sdsp"}:
            width = 1 if form.endswith("b") else 2 if form.endswith("h") else 4 if form.endswith("w") else 8
            self._store(self._address(item), width, self._read_operand(item, "rs2"))
            return True
        if form.startswith("f") and form[1:3] in {"lw", "ld", "lh", "lq"}:
            width = {"flw": 4, "fld": 8, "flh": 2, "flq": 16}[form]
            self._write_operand(item, "rd", self._load(self._address(item), width), width=width * 8)
            self.fp_used = True
            return True
        if form in {"fsw", "fsd", "fsh", "fsq"}:
            width = {"fsw": 4, "fsd": 8, "fsh": 2, "fsq": 16}[form]
            self._store(self._address(item), width, self._read_operand(item, "rs2"))
            self.fp_used = True
            return True
        if form.startswith("lr.") or form.startswith("sc.") or form.startswith("amo") or form.startswith("amocas."):
            width = _width_from_suffix(form, 4)
            address = self._address(item)
            if address % width:
                raise SemanticTrap(4 if form.startswith("lr.") else 6, int(item.pc_offset), address)
            if form.startswith("amocas."):
                self._require_atomic_read_write(address, width, form)
            old = self._load(address, width, signed=width in {1, 2, 4})
            if form.startswith("lr."):
                self.reservation = (address, width)
                self._write_gpr(getattr(item, "rd", None), old)
            elif form.startswith("sc."):
                success = int(self.reservation != (address, width))
                if not success:
                    self._store(address, width, self._read_operand(item, "rs2"))
                self._write_gpr(getattr(item, "rd", None), success)
                self.reservation = None
            elif form.startswith("amocas."):
                rd = getattr(item, "rd", None)
                compare = self._read_gpr(rd)
                replacement = self._read_operand(item, "rs2")
                self._write_gpr(rd, old)
                if _bits(old, width * 8) == _bits(compare, width * 8):
                    self._store(address, width, replacement)
            else:
                source = self._read_operand(item, "rs2")
                op = form.split(".")[0][3:]
                old_u = _bits(old, width * 8)
                src_u = _bits(source, width * 8)
                old_s, src_s = _signed(old_u, width * 8), _signed(src_u, width * 8)
                new = {
                    "add": old_u + src_u, "and": old_u & src_u, "or": old_u | src_u,
                    "xor": old_u ^ src_u, "swap": src_u,
                    "max": max(old_s, src_s), "min": min(old_s, src_s),
                    "maxu": max(old_u, src_u), "minu": min(old_u, src_u),
                }.get(op)
                if new is None:
                    raise SemanticUnsupported(f"atomic operation {form}")
                self._store(address, width, new)
                self._write_gpr(getattr(item, "rd", None), old)
            return True
        return False

    def _csr(self, item: object, form: str) -> bool:
        if form not in {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"}:
            return False
        csr = _field(item, "csr")
        if csr is None or not 0 <= int(csr) <= 0xFFF:
            raise SemanticUnsupported(f"invalid CSR address: {csr!r}")
        csr = int(csr)
        immediate_form = form.endswith("i")
        source_index = _field(item, "rs1")
        zimm = int(_field(item, "zimm5", 0) or 0)
        write_requested = form.startswith("csrrw") or (
            form.startswith(("csrrs", "csrrc"))
            and (zimm != 0 if immediate_form else int(source_index or 0) != 0)
        )

        dataflow = getattr(self.testcase, "dataflow_meta", {})
        realization = dataflow.get("generation_realization", {}) if isinstance(dataflow, dict) else {}
        profile = str(getattr(self.testcase, "isa_profile", ""))
        extensions = enabled_extensions(profile)
        privilege_class = (
            str(
                realization.get("privilege_class")
                or realization.get("privilege_mode")
                or "user"
            ).lower()
            if isinstance(realization, dict) else "user"
        )
        current_privilege = {
            "user": 0, "u": 0,
            "supervisor": 1, "s": 1, "virtual-supervisor": 1, "vs": 1,
            "hypervisor": 2, "h": 2,
            "machine": 3, "m": 3,
        }.get(privilege_class)
        if current_privilege is None:
            raise SemanticUnsupported(f"unknown CSR privilege class: {privilege_class}")
        if "zicsr" not in extensions:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        minimum_privilege = csr_minimum_privilege(csr)
        if minimum_privilege == 1 and "s" not in extensions:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if current_privilege < csr_minimum_privilege(csr):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if csr_is_rv32_only(csr) and self.xlen != 32:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if not csr_required_extensions_for_profile(csr, profile).issubset(extensions):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if write_requested and csr_is_read_only(csr):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        name = self._csr_name(item)
        if name == "unknown":
            raise SemanticUnsupported(f"unallocated CSR 0x{csr:03x}")
        if name not in self.csrs and name not in _UNINITIALIZED_RW_CSRS:
            raise SemanticUnsupported(f"CSR {name} (0x{csr:03x}) has no modeled value semantics")
        source = zimm if immediate_form else self._read_gpr(getattr(item, "rs1", None))
        source = int(source or 0)
        old_is_known = name in self.csrs
        write_without_read = form.startswith("csrrw") and int(_field(item, "rd", 0) or 0) == 0
        if not old_is_known and not write_without_read:
            raise SemanticUnsupported(f"CSR {name} (0x{csr:03x}) read before modeled write")
        old = int(self.csrs.get(name, 0))
        if name == "vcsr":
            old = (int(self.csrs["vxrm"]) & 3) | ((int(self.csrs["vxsat"]) & 1) << 2)
        if name in {"vl", "vtype", "vlenb"} and write_requested:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if form.startswith("csrrw"):
            new = source
        elif form.startswith("csrrs"):
            new = old | source
        else:
            new = old & ~source
        if name == "misa" and write_requested:
            if new != old:
                raise SemanticUnsupported(
                    "misa write changes the profile-defined ISA set"
                )
        csr_width = {
            "fflags": 5, "frm": 3, "fcsr": 8,
            "vxsat": 1, "vxrm": 2, "vcsr": 3,
        }.get(name)
        if csr_width is not None:
            new = _bits(new, csr_width)
        if write_requested:
            self.csrs[name] = _bits(new, self.xlen)
        if name == "vcsr" and write_requested:
            self.csrs["vxsat"] = (new >> 2) & 1
            self.csrs["vxrm"] = new & 3
            self.csrs["vcsr"] = _bits(new, 3)
        elif name in {"vxsat", "vxrm"} and write_requested:
            self.csrs["vcsr"] = (int(self.csrs["vxrm"]) & 3) | ((int(self.csrs["vxsat"]) & 1) << 2)
        if name == "fcsr":
            self.csrs["fflags"] = self.csrs[name] & 0x1F
            self.csrs["frm"] = (self.csrs[name] >> 5) & 7
        if name in {"fflags", "frm"}:
            self.csrs["fcsr"] = self.csrs["fflags"] | (self.csrs["frm"] << 5)
        self._write_gpr(getattr(item, "rd", None), old)
        self.csr_used = True
        return True

    # ----- floating point --------------------------------------------------

    def _fp_value(self, item: object, name: str, width: int) -> float:
        raw = self._read_fp_bits(item, name, width)
        return _float_from_bits(raw, width)

    def _accrue_fp_flags(self, flags: int) -> None:
        self.csrs["fflags"] = (int(self.csrs.get("fflags", 0)) | flags) & 0x1F
        self.csrs["fcsr"] = self.csrs["fflags"] | ((int(self.csrs.get("frm", 0)) & 7) << 5)

    def _fp(self, item: object, form: str) -> bool:
        if not (form.startswith("f") or form.startswith("c.f")):
            return False
        if form.startswith("fli"):
            width = {"fli.s": 32, "fli.d": 64, "fli.h": 16, "fli.q": 128}.get(form)
            if width is None:
                raise SemanticUnsupported(f"floating immediate {form}")
            literal_index = (_instruction_word(self.testcase, item) >> 20) & 0x1F
            biases = {16: 15, 32: 127, 64: 1023, 128: 16383}
            literals = {
                0: Fraction(-1),
                1: Fraction(1, 1 << (biases[width] - 1)),
                2: Fraction(1, 1 << 16),
                3: Fraction(1, 1 << 15),
                4: Fraction(1, 1 << 8),
                5: Fraction(1, 1 << 7),
                6: Fraction(1, 16),
                7: Fraction(1, 8),
                8: Fraction(1, 4),
                9: Fraction(5, 16),
                10: Fraction(3, 8),
                11: Fraction(7, 16),
                12: Fraction(1, 2),
                13: Fraction(5, 8),
                14: Fraction(3, 4),
                15: Fraction(7, 8),
                16: Fraction(1),
                17: Fraction(5, 4),
                18: Fraction(3, 2),
                19: Fraction(7, 4),
                20: Fraction(2),
                21: Fraction(5, 2),
                22: Fraction(3),
                23: Fraction(4),
                24: Fraction(8),
                25: Fraction(16),
                26: Fraction(128),
                27: Fraction(256),
                28: Fraction(1 << 15),
                29: Fraction(1 << 16),
            }
            if literal_index == 31:
                result_bits = _canonical_nan_bits(width)
            elif literal_index == 30 or (width == 16 and literal_index == 29):
                result_bits = _fp_infinity_bits(0, width)
            elif literal_index in literals:
                result_bits = _round_fraction_to_bits(
                    literals[literal_index], width, 0,
                )
            else:
                raise SemanticUnsupported(f"floating literal index {literal_index}")
            self._write_operand(item, "rd", result_bits, width=width)
            self.fp_used = True
            return True
        if form.startswith("c.f"):
            form = form[2:]
        form = {
            "flwsp": "flw", "fldsp": "fld", "fswsp": "fsw", "fsdsp": "fsd",
        }.get(form, form)
        if form in {"flw", "fld", "flh", "flq", "fsw", "fsd", "fsh", "fsq"}:
            return self._memory_or_atomic(item, form)
        finite_flag_forms = {
            f"{operation}.{suffix}"
            for operation in (
                "fadd", "fsub", "fmul", "fdiv", "fsqrt", "fmadd", "fmsub",
                "fnmsub", "fnmadd", "feq", "flt", "fle", "fmin", "fmax",
                "fminm", "fmaxm", "fleq", "fltq", "fcvtmod.w.d",
            )
            for suffix in ("h", "s", "d", "q")
        }
        finite_flag_forms.add("fcvtmod.w.d")
        if form not in finite_flag_forms and not form.startswith(
            ("fmv.", "fmvh.", "fmvp.", "fclass.", "fsgnj", "fround", "fli.", "fcvt.")
        ):
            self.fp_flags_exact = False
        rounded_forms = (
            "fadd", "fsub", "fmul", "fdiv", "fsqrt", "fmadd", "fnm", "fmsub", "fcvt.",
        )
        rounding_mode = 0
        if form.startswith(rounded_forms) and form != "fcvtmod.w.d":
            rounding_mode = int(_field(item, "rm", 0) or 0)
            if rounding_mode == 7:
                rounding_mode = int(self.csrs.get("frm", 0))
            if rounding_mode not in {0, 1, 2, 3, 4}:
                raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
        if form.startswith("fmv.w.x") or form.startswith("fmv.d.x") or form.startswith("fmv.h.x"):
            width = {"fmv.w.x": 32, "fmv.d.x": 64, "fmv.h.x": 16}[form]
            self._write_operand(item, "rd", self._read_gpr(getattr(item, "rs1", None)), width=width)
            self.fp_used = True
            return True
        if form.startswith("fmv.x.w") or form.startswith("fmv.x.d") or form.startswith("fmv.x.h"):
            width = {"fmv.x.w": 32, "fmv.x.d": 64, "fmv.x.h": 16}[form]
            raw = self._read_operand(item, "rs1") & ((1 << width) - 1)
            self._write_gpr(getattr(item, "rd", None), _sext(raw, width, self.xlen) if width < self.xlen else raw)
            self.fp_used = True
            return True
        if form in {"fmvh.x.d", "fmvh.x.q"}:
            raw = self._read_operand(item, "rs1")
            shift = 32 if form.endswith(".d") else 64
            self._write_gpr(getattr(item, "rd", None), raw >> shift)
            self.fp_used = True
            return True
        if form in {"fmvp.d.x", "fmvp.q.x"}:
            low = self._read_gpr(getattr(item, "rs1", None))
            high = self._read_gpr(getattr(item, "rs2", None))
            width = 64 if form.endswith(".d.x") else 128
            self._write_operand(item, "rd", low | (high << (32 if width == 64 else 64)), width=width)
            self.fp_used = True
            return True
        if form.startswith("fcvt."):
            return self._fp_convert(item, form, _float_width(form) or 64)
        if form == "fcvtmod.w.d":
            raw = self._read_fp_bits(item, "rs1", 64)
            kind = _float_kind(raw, 64)
            if kind != "finite":
                integer = 0
                self._accrue_fp_flags(1 << 4)
            else:
                value = Fraction.from_float(_float_from_bits(raw, 64))
                integer = _round_fraction_to_integer(value, 1)  # Zfa fixes RTZ.
                if not -(1 << 31) <= integer <= (1 << 31) - 1:
                    self._accrue_fp_flags(1 << 4)  # FCVT.W.D would be invalid.
                elif Fraction(integer) != value:
                    self._accrue_fp_flags(1)  # NX, unless NV was raised.
            self._write_gpr(getattr(item, "rd", None), _signed(integer & 0xFFFFFFFF, 32))
            self.fp_used = True
            return True
        width = _float_width(form)
        if width is None:
            return False
        if form.startswith("fclass"):
            raw = self._read_fp_bits(item, "rs1", width)
            sign = bool(raw & (1 << (width - 1)))
            fraction_bits = 112 if width == 128 else 10 if width == 16 else 23 if width == 32 else 52
            exponent_bits = width - fraction_bits - 1
            exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
            fraction = raw & ((1 << fraction_bits) - 1)
            exponent_max = (1 << exponent_bits) - 1
            if exponent == exponent_max and fraction:
                quiet_bit = 1 << (fraction_bits - 1)
                result = 1 << (9 if fraction & quiet_bit else 8)
            elif exponent == exponent_max: result = 1 << (0 if sign else 7)
            elif exponent == 0 and fraction == 0: result = 1 << (3 if sign else 4)
            elif exponent == 0: result = 1 << (2 if sign else 5)
            elif sign: result = 1 << 1
            else: result = 1 << 6
            self._write_gpr(getattr(item, "rd", None), result)
            self.fp_used = True
            return True
        if form.startswith("fround"):
            raw = self._read_fp_bits(item, "rs1", width)
            rm = _field(item, "rm", 0) or 0
            if rm == 7:
                rm = int(self.csrs.get("frm", 0))
            if rm not in {0, 1, 2, 3, 4}:
                raise SemanticUnsupported(f"{form} rounding mode {rm}")
            if width == 128:
                sign = raw >> 127
                exponent = (raw >> 112) & 0x7FFF
                fraction = raw & ((1 << 112) - 1)
                if exponent == 0x7FFF:
                    rounded_bits = (
                        raw if not fraction else _canonical_nan_bits(128)
                    )
                    if fraction and not fraction & (1 << 111):
                        self.csrs["fflags"] |= 1 << 4
                        self.csrs["fcsr"] = self.csrs["fflags"] | (self.csrs["frm"] << 5)
                else:
                    exact = _binary128_fraction(raw)
                    assert exact is not None
                    rounded = _round_fraction_to_integer(exact, rm)
                    rounded_fraction = Fraction(rounded)
                    rounded_bits = _round_fraction_to_bits(
                        rounded_fraction, 128, 0,
                        negative_zero=bool(sign and rounded == 0),
                    )
                    if form.startswith("froundnx") and exact != rounded_fraction:
                        self.csrs["fflags"] |= 1
                        self.csrs["fcsr"] = self.csrs["fflags"] | (self.csrs["frm"] << 5)
                self._write_operand(item, "rd", rounded_bits, width=128)
                self.fp_used = True
                return True
            value = _float_from_bits(raw, width)
            if math.isnan(value):
                rounded_bits = _float_to_bits(math.nan, width)
                fraction_bits = 10 if width == 16 else 23 if width == 32 else 52
                quiet_bit = 1 << (fraction_bits - 1)
                fraction = raw & ((1 << fraction_bits) - 1)
                if fraction and not fraction & quiet_bit:
                    self.csrs["fflags"] |= 1 << 4
                    self.csrs["fcsr"] = self.csrs["fflags"] | (self.csrs["frm"] << 5)
            elif math.isinf(value) or value == 0:
                rounded_bits = raw
                rounded = value
            else:
                if rm == 1: rounded = math.trunc(value)
                elif rm == 2: rounded = math.floor(value)
                elif rm == 3: rounded = math.ceil(value)
                elif rm == 4: rounded = math.copysign(math.floor(abs(value) + 0.5), value)
                else: rounded = round(value)
                rounded_bits = (
                    1 << (width - 1)
                    if rounded == 0 and raw & (1 << (width - 1))
                    else _float_to_bits(float(rounded), width)
                )
            self._write_operand(item, "rd", rounded_bits, width=width)
            if form.startswith("froundnx") and math.isfinite(value) and rounded != value:
                self.csrs["fflags"] |= 1
                self.csrs["fcsr"] = self.csrs["fflags"] | (self.csrs["frm"] << 5)
            self.fp_used = True
            return True
        if form.startswith("fsgnj"):
            left = self._read_fp_bits(item, "rs1", width)
            right = self._read_fp_bits(item, "rs2", width)
            sign = 1 << (width - 1)
            magnitude = left & (sign - 1)
            rhs_sign = right & sign
            if form.startswith("fsgnjn"):
                rhs_sign ^= sign
            elif form.startswith("fsgnjx"):
                rhs_sign = (left ^ right) & sign
            self._write_operand(item, "rd", magnitude | rhs_sign, width=width)
            self.fp_used = True
            return True
        if width == 128:
            if form not in {
                "fadd.q", "fsub.q", "fmul.q", "fdiv.q", "feq.q", "flt.q", "fle.q",
                "fltq.q", "fleq.q",
                "fmadd.q", "fmsub.q", "fnmsub.q", "fnmadd.q",
                "fmin.q", "fmax.q", "fminm.q", "fmaxm.q",
                "fsqrt.q",
            }:
                raise SemanticUnsupported(f"binary128 semantics are not modeled for {form}")
            left_raw = self._read_fp_bits(item, "rs1", 128)
            if form == "fsqrt.q":
                result, raised_flags = _round_binary128_sqrt(left_raw, rounding_mode)
                self._accrue_fp_flags(raised_flags)
                self._write_operand(item, "rd", result, width=128)
                self.fp_used = True
                return True
            right_raw = self._read_fp_bits(item, "rs2", 128)
            left_kind, right_kind = _binary128_kind(left_raw), _binary128_kind(right_raw)
            left_nan = left_kind.endswith("nan")
            right_nan = right_kind.endswith("nan")
            invalid_nan = left_kind == "signaling_nan" or right_kind == "signaling_nan"
            left = _binary128_fraction(left_raw)
            right = _binary128_fraction(right_raw)
            operation = form.split(".", 1)[0]
            is_fma = operation in {"fmadd", "fmsub", "fnmsub", "fnmadd"}
            third_raw = self._read_fp_bits(item, "rs3", 128) if is_fma else None
            third_kind = _binary128_kind(third_raw) if third_raw is not None else "finite"
            third_nan = third_kind.endswith("nan")
            invalid_nan = invalid_nan or third_kind == "signaling_nan"
            third = _binary128_fraction(third_raw) if third_raw is not None else None
            if form == "feq.q":
                if left_nan or right_nan:
                    result = 0
                    self._accrue_fp_flags(int(invalid_nan) << 4)
                else:
                    result = int(_binary128_compare(left_raw, right_raw) == 0)
            elif form == "flt.q":
                if left_nan or right_nan:
                    result = 0
                    self._accrue_fp_flags(1 << 4)
                else:
                    result = int(_binary128_compare(left_raw, right_raw) < 0)
            elif form == "fle.q":
                if left_nan or right_nan:
                    result = 0
                    self._accrue_fp_flags(1 << 4)
                else:
                    result = int(_binary128_compare(left_raw, right_raw) <= 0)
            elif form in {"fltq.q", "fleq.q"}:
                if left_nan or right_nan:
                    result = 0
                    self._accrue_fp_flags(int(invalid_nan) << 4)
                else:
                    comparison = _binary128_compare(left_raw, right_raw)
                    result = int(
                        comparison < 0 if form == "fltq.q" else comparison <= 0
                    )
            elif form.startswith(("fmin", "fmax")):
                minimum = form.startswith("fmin")
                if form.endswith("minm.q") or form.endswith("maxm.q"):
                    result = _canonical_nan_bits(128) if left_nan or right_nan else 0
                    choose_numeric = not left_nan and not right_nan
                elif left_nan and right_nan:
                    result = _canonical_nan_bits(128)
                    choose_numeric = False
                elif left_nan:
                    result = right_raw
                    choose_numeric = False
                elif right_nan:
                    result = left_raw
                    choose_numeric = False
                else:
                    result = 0
                    choose_numeric = True
                if choose_numeric:
                    comparison = _binary128_compare(left_raw, right_raw)
                    if comparison == 0 and left == 0 and right == 0:
                        left_sign, right_sign = left_raw >> 127, right_raw >> 127
                        sign = (left_sign | right_sign) if minimum else (left_sign & right_sign)
                        result = sign << 127
                    else:
                        choose_left = comparison <= 0 if minimum else comparison >= 0
                        result = left_raw if choose_left else right_raw
                self._accrue_fp_flags(int(invalid_nan) << 4)
                self._write_operand(item, "rd", result, width=128)
                self.fp_used = True
                return True
            else:
                rounding_mode = int(_field(item, "rm", 0) or 0)
                if rounding_mode == 7:
                    rounding_mode = int(self.csrs.get("frm", 0))
                if rounding_mode not in {0, 1, 2, 3, 4}:
                    raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                if is_fma:
                    left_zero = left_kind == "finite" and left == 0
                    right_zero = right_kind == "finite" and right == 0
                    invalid_product = (
                        left_kind == "infinity" and right_zero
                        or right_kind == "infinity" and left_zero
                    )
                    if invalid_product or invalid_nan:
                        self._accrue_fp_flags(1 << 4)
                        self._write_operand(item, "rd", _canonical_nan_bits(128), width=128)
                        self.fp_used = True
                        return True
                    if left_nan or right_nan or third_nan:
                        self._write_operand(item, "rd", _canonical_nan_bits(128), width=128)
                        self.fp_used = True
                        return True
                    negative_product = operation in {"fnmsub", "fnmadd"}
                    negative_addend = operation in {"fmsub", "fnmadd"}
                    if left_kind == "infinity" or right_kind == "infinity":
                        product_sign = (left_raw >> 127) ^ (right_raw >> 127) ^ int(negative_product)
                        if third_kind == "infinity":
                            addend_sign = (third_raw >> 127) ^ int(negative_addend)
                            if product_sign != addend_sign:
                                self._accrue_fp_flags(1 << 4)
                                result = _canonical_nan_bits(128)
                            else:
                                result = _binary128_infinity(product_sign)
                        else:
                            result = _binary128_infinity(product_sign)
                        self._write_operand(item, "rd", result, width=128)
                        self.fp_used = True
                        return True
                    if third_kind == "infinity":
                        addend_sign = (third_raw >> 127) ^ int(negative_addend)
                        self._write_operand(item, "rd", _binary128_infinity(addend_sign), width=128)
                        self.fp_used = True
                        return True
                else:
                    if invalid_nan:
                        self._accrue_fp_flags(1 << 4)
                        self._write_operand(item, "rd", _canonical_nan_bits(128), width=128)
                        self.fp_used = True
                        return True
                    if left_nan or right_nan:
                        self._write_operand(item, "rd", _canonical_nan_bits(128), width=128)
                        self.fp_used = True
                        return True
                    left_sign, right_sign = left_raw >> 127, right_raw >> 127
                    left_inf, right_inf = left_kind == "infinity", right_kind == "infinity"
                    left_zero = left_kind == "finite" and left == 0
                    right_zero = right_kind == "finite" and right == 0
                    if operation in {"fadd", "fsub"} and (left_inf or right_inf):
                        rhs_sign = right_sign ^ int(operation == "fsub")
                        if left_inf and right_inf and left_sign != rhs_sign:
                            self._accrue_fp_flags(1 << 4)
                            result = _canonical_nan_bits(128)
                        else:
                            result = _binary128_infinity(left_sign if left_inf else rhs_sign)
                        self._write_operand(item, "rd", result, width=128)
                        self.fp_used = True
                        return True
                    if operation == "fmul" and (left_inf or right_inf):
                        if left_zero or right_zero:
                            self._accrue_fp_flags(1 << 4)
                            result = _canonical_nan_bits(128)
                        else:
                            result = _binary128_infinity(left_sign ^ right_sign)
                        self._write_operand(item, "rd", result, width=128)
                        self.fp_used = True
                        return True
                    if operation == "fdiv":
                        sign = left_sign ^ right_sign
                        if left_inf and right_inf or left_zero and right_zero:
                            self._accrue_fp_flags(1 << 4)
                            result = _canonical_nan_bits(128)
                            self._write_operand(item, "rd", result, width=128)
                            self.fp_used = True
                            return True
                        if right_zero:
                            if not left_inf:
                                self._accrue_fp_flags(1 << 3)
                            result = _binary128_infinity(sign)
                            self._write_operand(item, "rd", result, width=128)
                            self.fp_used = True
                            return True
                        if left_inf:
                            self._write_operand(item, "rd", _binary128_infinity(sign), width=128)
                            self.fp_used = True
                            return True
                        if right_inf:
                            self._write_operand(item, "rd", sign << 127, width=128)
                            self.fp_used = True
                            return True
                if form == "fadd.q":
                    exact_result = left + right
                elif form == "fsub.q":
                    exact_result = left - right
                elif form == "fmul.q":
                    exact_result = left * right
                elif is_fma:
                    product = left * right
                    negative_product = operation in {"fnmsub", "fnmadd"}
                    negative_addend = operation in {"fmsub", "fnmadd"}
                    exact_result = (-product if negative_product else product) + (
                        -third if negative_addend else third
                    )
                else:
                    exact_result = left / right
                negative_zero = False
                if exact_result == 0:
                    left_sign = bool(left_raw >> 127)
                    right_sign = bool(right_raw >> 127)
                    if is_fma:
                        product_sign = left_sign ^ right_sign ^ negative_product
                        addend_sign = bool(third_raw >> 127) ^ negative_addend
                        negative_zero = (
                            product_sign
                            if (left == 0 or right == 0) and third == 0 and product_sign == addend_sign
                            else rounding_mode == 2
                        )
                    elif form in {"fmul.q", "fdiv.q"}:
                        negative_zero = left_sign ^ right_sign
                    else:
                        if form == "fsub.q":
                            right_sign = not right_sign
                        both_zero = left == 0 and right == 0
                        negative_zero = (
                            left_sign if both_zero and left_sign == right_sign
                            else rounding_mode == 2
                        )
                result, raised_flags = _round_finite_fraction_with_flags(
                    exact_result, 128, rounding_mode, negative_zero=negative_zero,
                )
                self._accrue_fp_flags(raised_flags)
                self._write_operand(item, "rd", result, width=128)
                self.fp_used = True
                return True
            self._write_gpr(getattr(item, "rd", None), result)
            self.fp_used = True
            return True
        left = self._fp_value(item, "rs1", width)
        right = self._fp_value(item, "rs2", width) if getattr(item, "rs2", None) is not None else 0.0
        left_raw = self._read_fp_bits(item, "rs1", width)
        right_raw = (
            self._read_fp_bits(item, "rs2", width)
            if getattr(item, "rs2", None) is not None else 0
        )
        left_kind, right_kind = _float_kind(left_raw, width), _float_kind(right_raw, width)
        left_nan, right_nan = left_kind.endswith("nan"), right_kind.endswith("nan")
        invalid_nan = (
            left_kind == "signaling_nan" or right_kind == "signaling_nan"
        )
        operation = form.split(".", 1)[0]
        is_fma = operation in {"fmadd", "fmsub", "fnmsub", "fnmadd"}
        third = self._fp_value(item, "rs3", width) if is_fma else 0.0
        third_raw = self._read_fp_bits(item, "rs3", width) if is_fma else 0
        third_kind = _float_kind(third_raw, width) if is_fma else "finite"
        third_nan = third_kind.endswith("nan")
        invalid_nan = invalid_nan or third_kind == "signaling_nan"

        if operation == "fsqrt":
            if left_kind == "signaling_nan":
                self._accrue_fp_flags(1 << 4)
                self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                self.fp_used = True
                return True
            if left_kind == "quiet_nan":
                self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                self.fp_used = True
                return True
            if left_kind == "infinity":
                if left_raw >> (width - 1):
                    self._accrue_fp_flags(1 << 4)
                    self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                else:
                    self._write_operand(item, "rd", left_raw, width=width)
                self.fp_used = True
                return True
            if left < 0:
                self._accrue_fp_flags(1 << 4)
                self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                self.fp_used = True
                return True

        if operation in {"feq", "flt", "fle", "fltq", "fleq"} and (left_nan or right_nan):
            raised_flags = int(
                invalid_nan or operation in {"flt", "fle"}
            ) << 4
            self._accrue_fp_flags(raised_flags)
            self._write_gpr(getattr(item, "rd", None), 0)
            self.fp_used = True
            return True

        if operation in {"fmin", "fmax", "fminm", "fmaxm"} and (left_nan or right_nan):
            self._accrue_fp_flags(int(invalid_nan) << 4)
            if operation in {"fminm", "fmaxm"} or left_nan and right_nan:
                result_bits = _canonical_nan_bits(width)
            else:
                result_bits = right_raw if left_nan else left_raw
            self._write_operand(item, "rd", result_bits, width=width)
            self.fp_used = True
            return True

        if is_fma:
            left_zero = left_kind == "finite" and left == 0
            right_zero = right_kind == "finite" and right == 0
            invalid_product = (
                left_kind == "infinity" and right_zero
                or right_kind == "infinity" and left_zero
            )
            if invalid_product or invalid_nan:
                self._accrue_fp_flags(1 << 4)
                self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                self.fp_used = True
                return True
            if left_nan or right_nan or third_nan:
                self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                self.fp_used = True
                return True
            negative_product = operation in {"fnmsub", "fnmadd"}
            negative_addend = operation in {"fmsub", "fnmadd"}
            left_inf, right_inf = left_kind == "infinity", right_kind == "infinity"
            if left_inf or right_inf:
                product_sign = (
                    (left_raw >> (width - 1)) ^ (right_raw >> (width - 1))
                    ^ int(negative_product)
                )
                if third_kind == "infinity":
                    addend_sign = (third_raw >> (width - 1)) ^ int(negative_addend)
                    if product_sign != addend_sign:
                        self._accrue_fp_flags(1 << 4)
                        result_bits = _canonical_nan_bits(width)
                    else:
                        result_bits = _float_infinity_bits(product_sign, width)
                else:
                    result_bits = _float_infinity_bits(product_sign, width)
                self._write_operand(item, "rd", result_bits, width=width)
                self.fp_used = True
                return True
            if third_kind == "infinity":
                addend_sign = (third_raw >> (width - 1)) ^ int(negative_addend)
                self._write_operand(
                    item, "rd", _float_infinity_bits(addend_sign, width), width=width,
                )
                self.fp_used = True
                return True
        elif left_nan or right_nan:
            self._accrue_fp_flags(int(invalid_nan) << 4)
            self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
            self.fp_used = True
            return True
        else:
            left_sign, right_sign = left_raw >> (width - 1), right_raw >> (width - 1)
            left_inf, right_inf = left_kind == "infinity", right_kind == "infinity"
            left_zero = left_kind == "finite" and left == 0
            right_zero = right_kind == "finite" and right == 0
            if operation in {"fadd", "fsub"} and (left_inf or right_inf):
                rhs_sign = right_sign ^ int(operation == "fsub")
                if left_inf and right_inf and left_sign != rhs_sign:
                    self._accrue_fp_flags(1 << 4)
                    result_bits = _canonical_nan_bits(width)
                else:
                    result_bits = _float_infinity_bits(
                        left_sign if left_inf else rhs_sign, width,
                    )
                self._write_operand(item, "rd", result_bits, width=width)
                self.fp_used = True
                return True
            if operation == "fmul" and (left_inf or right_inf):
                if left_zero or right_zero:
                    self._accrue_fp_flags(1 << 4)
                    result_bits = _canonical_nan_bits(width)
                else:
                    result_bits = _float_infinity_bits(left_sign ^ right_sign, width)
                self._write_operand(item, "rd", result_bits, width=width)
                self.fp_used = True
                return True
            if operation == "fdiv":
                sign = left_sign ^ right_sign
                if (left_inf and right_inf) or (left_zero and right_zero):
                    self._accrue_fp_flags(1 << 4)
                    self._write_operand(item, "rd", _canonical_nan_bits(width), width=width)
                    self.fp_used = True
                    return True
                if right_zero:
                    if not left_inf:
                        self._accrue_fp_flags(1 << 3)
                    self._write_operand(
                        item, "rd", _float_infinity_bits(sign, width), width=width,
                    )
                    self.fp_used = True
                    return True
                if left_inf:
                    self._write_operand(
                        item, "rd", _float_infinity_bits(sign, width), width=width,
                    )
                    self.fp_used = True
                    return True
                if right_inf:
                    self._write_operand(item, "rd", sign << (width - 1), width=width)
                    self.fp_used = True
                    return True

        exact_result = None
        left_fraction = Fraction.from_float(left) if math.isfinite(left) else None
        right_fraction = Fraction.from_float(right) if math.isfinite(right) else None
        if form.startswith("fmadd") or form.startswith("fnmadd") or form.startswith("fmsub") or form.startswith("fnmsub"):
            third = self._fp_value(item, "rs3", width)
            third_fraction = Fraction.from_float(third) if math.isfinite(third) else None
            operation = form.split(".", 1)[0]
            negative_product = operation in {"fnmsub", "fnmadd"}
            negative_addend = operation in {"fmsub", "fnmadd"}
            if left_fraction is not None and right_fraction is not None and third_fraction is not None:
                product = left_fraction * right_fraction
                exact_result = (-product if negative_product else product) + (
                    -third_fraction if negative_addend else third_fraction
                )
                value = 0.0
            else:
                product = left * right
                value = (-product if negative_product else product) + (
                    -third if negative_addend else third
                )
        elif form.startswith("fadd"):
            exact_result = left_fraction + right_fraction if left_fraction is not None and right_fraction is not None else None
            value = left + right
        elif form.startswith("fsub"):
            exact_result = left_fraction - right_fraction if left_fraction is not None and right_fraction is not None else None
            value = left - right
        elif form.startswith("fmul"):
            exact_result = left_fraction * right_fraction if left_fraction is not None and right_fraction is not None else None
            value = left * right
        elif form.startswith("fdiv"):
            exact_result = (
                left_fraction / right_fraction
                if left_fraction is not None and right_fraction not in (None, 0) else None
            )
            value = _ieee_divide(left, right)
        elif form.startswith("fsqrt"):
            value = math.sqrt(left) if left >= 0 else math.nan
            if math.isfinite(left) and left >= 0:
                result_bits = _round_sqrt_to_bits(left, width, rounding_mode)
                rounded_root = Fraction.from_float(_float_from_bits(result_bits, width))
                self._accrue_fp_flags(int(rounded_root * rounded_root != left_fraction))
                self._write_operand(item, "rd", result_bits, width=width)
                self.fp_used = True
                return True
        elif form.startswith("fmin"):
            value = _fp_minmax(
                left, right, minimum=True,
                canonical_if_any_nan=form.startswith("fminm"),
            )
        elif form.startswith("fmax"):
            value = _fp_minmax(
                left, right, minimum=False,
                canonical_if_any_nan=form.startswith("fmaxm"),
            )
        elif form.startswith("feq"):
            self._write_gpr(getattr(item, "rd", None), int(not math.isnan(left) and not math.isnan(right) and left == right))
            self.fp_used = True
            return True
        elif form.startswith(("fleq", "fltq")):
            left_raw = self._read_fp_bits(item, "rs1", width)
            right_raw = self._read_fp_bits(item, "rs2", width)
            if math.isnan(left) or math.isnan(right):
                raised_flags = int(
                    _is_signaling_nan_bits(left_raw, width)
                    or _is_signaling_nan_bits(right_raw, width)
                ) << 4
                self._accrue_fp_flags(raised_flags)
                result = 0
            elif form.startswith("fleq"):
                result = int(left <= right)
            else:
                result = int(left < right)
            self._write_gpr(getattr(item, "rd", None), result)
            self.fp_used = True
            return True
        elif form.startswith("fle"):
            self._write_gpr(getattr(item, "rd", None), int(not math.isnan(left) and not math.isnan(right) and left <= right))
            self.fp_used = True
            return True
        elif form.startswith("flt"):
            self._write_gpr(getattr(item, "rd", None), int(not math.isnan(left) and not math.isnan(right) and left < right))
            self.fp_used = True
            return True
        else:
            raise SemanticUnsupported(f"floating form {form}")
        negative_zero = False
        if exact_result == 0:
            if form.startswith("fadd") or form.startswith("fsub"):
                right_sign = math.copysign(1.0, right) < 0
                if form.startswith("fsub"):
                    right_sign = not right_sign
                left_sign = math.copysign(1.0, left) < 0
                negative_zero = (
                    left_sign if left == 0 and right == 0 and left_sign == right_sign
                    else rounding_mode == 2
                )
            elif form.startswith(("fmadd", "fnmadd", "fmsub", "fnmsub")):
                product_sign = (
                    (math.copysign(1.0, left) < 0)
                    ^ (math.copysign(1.0, right) < 0)
                    ^ negative_product
                )
                addend_sign = (
                    (math.copysign(1.0, third) < 0)
                    ^ negative_addend
                )
                both_zero = (left == 0 or right == 0) and third == 0
                negative_zero = (
                    product_sign if both_zero and product_sign == addend_sign
                    else rounding_mode == 2
                )
            else:
                negative_zero = math.copysign(1.0, value) < 0
        if exact_result is not None:
            result_bits, raised_flags = _round_finite_fraction_with_flags(
                exact_result, width, rounding_mode, negative_zero=negative_zero,
            )
            self._accrue_fp_flags(raised_flags)
        elif form.startswith(("fmin", "fmax")):
            result_bits = _float_to_bits(value, width)
        else:
            finite_inputs = left_fraction is not None and right_fraction is not None
            if form.startswith("fdiv") and finite_inputs and right_fraction == 0:
                self._accrue_fp_flags((1 << 4) if left_fraction == 0 else (1 << 3))
            elif form.startswith("fsqrt") and left_fraction is not None and left_fraction < 0:
                self._accrue_fp_flags(1 << 4)
            else:
                self.fp_flags_exact = False
            result_bits = _float_to_bits(value, width)
        self._write_operand(item, "rd", result_bits, width=width)
        self.fp_used = True
        return True

    def _fp_convert(self, item: object, form: str, width: int) -> bool:
        flags_exact_before = self.fp_flags_exact
        self.fp_flags_exact = False
        rounding_mode = int(_field(item, "rm", 0) or 0)
        if rounding_mode == 7:
            rounding_mode = int(self.csrs.get("frm", 0))
        if rounding_mode not in {0, 1, 2, 3, 4}:
            raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
        parts = form.split(".")
        if len(parts) < 3:
            raise SemanticUnsupported(f"floating conversion {form}")
        dst, src = parts[1], parts[2]
        if dst == "bf16" and src == "s":
            raw = self._read_fp_bits(item, "rs1", 32)
            self._write_operand(item, "rd", _round_f32_to_bf16(raw, rounding_mode), width=16)
            self.fp_used = True
            return True
        if dst == "s" and src == "bf16":
            raw = self._read_fp_bits(item, "rs1", 16) << 16
            self._write_operand(item, "rd", raw, width=32)
            self.fp_used = True
            return True
        src_width = {"h": 16, "s": 32, "d": 64, "q": 128}.get(src)
        dst_width = {"h": 16, "s": 32, "d": 64, "q": 128}.get(dst)
        integer_formats = {"w", "wu", "l", "lu"}
        if src in integer_formats and dst_width is not None:
            source_width = 32 if src in {"w", "wu"} else self.xlen
            raw_integer = self._read_operand(item, "rs1")
            integer = (
                _bits(raw_integer, source_width)
                if src.endswith("u") else _signed(raw_integer, source_width)
            )
            result, raised_flags = _round_finite_fraction_with_flags(
                Fraction(integer), dst_width, rounding_mode,
            )
            self._accrue_fp_flags(raised_flags)
            self.fp_flags_exact = flags_exact_before
            self._write_operand(item, "rd", result, width=dst_width)
            self.fp_used = True
            return True

        if src_width is not None and dst in integer_formats:
            raw = self._read_fp_bits(item, "rs1", src_width)
            kind = _fp_format_kind(raw, src_width)
            value = _fp_format_fraction(raw, src_width)
            target_width = 32 if dst in {"w", "wu"} else self.xlen
            lower, upper = (
                (0, (1 << target_width) - 1)
                if dst.endswith("u") else
                (-(1 << (target_width - 1)), (1 << (target_width - 1)) - 1)
            )
            if kind != "finite":
                if kind == "infinity" and raw >> (src_width - 1):
                    integer = lower
                else:
                    integer = upper
                self._accrue_fp_flags(1 << 4)  # NV suppresses NX.
            else:
                assert value is not None
                integer = _round_fraction_to_integer(value, rounding_mode)
                if integer < lower or integer > upper:
                    integer = lower if integer < lower else upper
                    self._accrue_fp_flags(1 << 4)  # NV suppresses NX.
                else:
                    self._accrue_fp_flags(int(Fraction(integer) != value))
            self.fp_flags_exact = flags_exact_before
            if target_width == 32:
                integer = _signed(integer, 32)
            self._write_gpr(getattr(item, "rd", None), integer)
            self.fp_used = True
            return True

        if src_width is not None and dst_width is not None:
            raw = self._read_fp_bits(item, "rs1", src_width)
            kind = _fp_format_kind(raw, src_width)
            value = _fp_format_fraction(raw, src_width)
            if kind == "infinity":
                result = _fp_infinity_bits(raw >> (src_width - 1), dst_width)
                raised_flags = 0
            elif kind == "quiet_nan":
                result = _canonical_nan_bits(dst_width)
                raised_flags = 0
            elif kind == "signaling_nan":
                result = _canonical_nan_bits(dst_width)
                raised_flags = 1 << 4
            else:
                assert value is not None
                result, raised_flags = _round_finite_fraction_with_flags(
                    value, dst_width, rounding_mode,
                    negative_zero=bool(raw >> (src_width - 1)) and value == 0,
                )
            self._accrue_fp_flags(raised_flags)
            self.fp_flags_exact = flags_exact_before
            self._write_operand(item, "rd", result, width=dst_width)
            self.fp_used = True
            return True

        raise SemanticUnsupported(f"floating conversion {form} is not modeled")

    # ----- vector operations -----------------------------------------------

    def _vector_vlmax(self) -> tuple[int, int]:
        zimm = int(self.csrs.get("vtype", 0))
        if zimm & (1 << (self.xlen - 1)):
            raise SemanticUnsupported("vtype.vill is set")
        sew = 8 << ((zimm >> 3) & 7)
        if sew > 64:
            raise SemanticUnsupported(f"SEW={sew} exceeds modeled ELEN=64")
        lmul_code = zimm & 7
        lmul = {0: 1, 1: 2, 2: 4, 3: 8, 5: 0.125, 6: 0.25, 7: 0.5}.get(lmul_code)
        if lmul is None:
            raise SemanticUnsupported(f"reserved LMUL encoding {lmul_code}")
        max_lanes = int((int(self.csrs.get("vlenb", 32)) * 8 * lmul) // sew)
        if max_lanes < 1:
            raise SemanticUnsupported(f"SEW={sew} is too wide for LMUL={lmul}")
        return sew, max_lanes

    def _vector_shape(self) -> tuple[int, int]:
        sew, max_lanes = self._vector_vlmax()
        return sew, min(int(self.csrs.get("vl", max_lanes)), max_lanes)

    def _vector_get(self, register: int, index: int, sew: int) -> int:
        if not 0 <= register < 32 or index < 0 or sew <= 0:
            raise SemanticUnsupported("invalid vector element address")
        vlen = int(self.csrs.get("vlenb", 32)) * 8
        bit_offset = index * sew
        consumed = 0
        result = 0
        while consumed < sew:
            reg_offset, in_register = divmod(bit_offset, vlen)
            physical_register = register + reg_offset
            if physical_register >= 32:
                raise SemanticUnsupported("vector register group exceeds v31")
            width = min(sew - consumed, vlen - in_register)
            chunk = (self.vreg[physical_register] >> in_register) & ((1 << width) - 1)
            result |= chunk << consumed
            consumed += width
            bit_offset += width
        return result

    def _vector_set(self, register: int, index: int, value: int, sew: int) -> None:
        if not 0 <= register < 32 or index < 0 or sew <= 0:
            raise SemanticUnsupported("invalid vector element address")
        vlen = int(self.csrs.get("vlenb", 32)) * 8
        bit_offset = index * sew
        consumed = 0
        value = _bits(value, sew)
        while consumed < sew:
            reg_offset, in_register = divmod(bit_offset, vlen)
            physical_register = register + reg_offset
            if physical_register >= 32:
                raise SemanticUnsupported("vector register group exceeds v31")
            width = min(sew - consumed, vlen - in_register)
            chunk_mask = (1 << width) - 1
            shifted_mask = chunk_mask << in_register
            chunk = ((value >> consumed) & chunk_mask) << in_register
            self.vreg[physical_register] = (
                (self.vreg[physical_register] & ~shifted_mask) | chunk
            )
            consumed += width
            bit_offset += width

    def _vector_mask_get(self, register: int, index: int) -> bool:
        return bool(self._vector_get(register, index, 1))

    def _vector_mask_set(self, register: int, index: int, value: bool) -> None:
        self._vector_set(register, index, int(value), 1)

    def _vector_vm_enabled(self, item: object) -> bool:
        if int(getattr(item, "byte_length", 0)) == 4:
            return bool((_instruction_word(self.testcase, item) >> 25) & 1)
        return bool(_field(item, "vm", 1))

    def _vector_masked(self, item: object, index: int) -> bool:
        return self._vector_vm_enabled(item) or self._vector_mask_get(0, index)

    def _vector_crypto(self, item: object, form: str, fields: dict[str, int], sew: int, vl: int) -> bool:
        if form in {
            "vqwbdotau.vv", "vqwbdotas.vv", "vfwdota.vv", "vfqwdota.vv",
            "vfqwdota.alt.vv", "vfwbdota.vv", "vfqwbdota.vv",
            "vfqwbdota.alt.vv", "vfbdota.vv",
        }:
            raise SemanticUnsupported(f"{form} dot-product format is not modeled")
        crypto_forms = (
            "vaesdf.vs", "vaesdf.vv", "vaesdm.vs", "vaesdm.vv",
            "vaesef.vs", "vaesef.vv", "vaesem.vs", "vaesem.vv",
            "vaeskf1.vi", "vaeskf2.vi", "vaesz.vs",
            "vghsh.vv", "vgmul.vv", "vsha2ch.vv", "vsha2cl.vv", "vsha2ms.vv",
            "vsm3c.vi", "vsm3me.vv", "vsm4k.vi", "vsm4r.vs", "vsm4r.vv",
            "vqwdotas.vv", "vqwdotau.vv",
        )
        if form not in crypto_forms:
            return False

        vtype = int(self.csrs.get("vtype", 0))
        lmul_code = vtype & 7
        lmul_by_code = {0: 1, 1: 2, 2: 4, 3: 8, 5: 0.125, 6: 0.25, 7: 0.5}
        if lmul_code not in lmul_by_code:
            raise SemanticUnsupported(f"{form} reserved LMUL encoding {lmul_code}")
        lmul = lmul_by_code[lmul_code]

        def register(name: str) -> int:
            value = fields.get(name)
            if not isinstance(value, int) or not 0 <= value < 32:
                raise SemanticUnsupported(f"{form} missing or invalid {name}")
            return value

        def register_group(base: int, emul: float) -> set[int]:
            count = max(1, math.ceil(emul))
            if base + count > 32 or (count > 1 and base % count):
                raise SemanticUnsupported(f"{form} invalid register group v{base}/EMUL={emul}")
            return set(range(base, base + count))

        if form in {"vqwdotas.vv", "vqwdotau.vv"}:
            if sew != 8:
                raise SemanticUnsupported(f"{form} requires SEW=8, got {sew}")
            if int(self.csrs.get("vstart", 0)) != 0:
                raise SemanticUnsupported(f"{form} requires vstart=0")
            vd = register("vd")
            vs2 = register("vs2")
            vs1 = register("vs1")
            vd_group = register_group(vd, 1)
            if vd_group & register_group(vs2, lmul) or vd_group & register_group(vs1, lmul):
                raise SemanticUnsupported(f"{form} destination overlaps a source group")
            signed_vs2 = form == "vqwdotas.vv"
            signed_vs1 = bool(
                int(self.csrs.get("vtype", 0)) & (1 << _VTYPE_ALTFMT_BIT)
            )
            accumulator = self._vector_get(vd, 0, 32)
            for index in range(vl):
                if not self._vector_masked(item, index):
                    continue
                left = self._vector_get(vs2, index, 8)
                right = self._vector_get(vs1, index, 8)
                if signed_vs2:
                    left = _signed(left, 8)
                if signed_vs1:
                    right = _signed(right, 8)
                accumulator = (accumulator + left * right) & 0xFFFFFFFF
            self._vector_set(vd, 0, accumulator, 32)
            self.csrs["vstart"] = 0
            return True

        group_size = 8 if form.startswith("vsm3") else 4
        if form.startswith("vsha2"):
            if sew not in {32, 64}:
                raise SemanticUnsupported(f"{form} SEW={sew}")
        elif sew != 32:
            raise SemanticUnsupported(f"{form} SEW={sew}")
        start = int(self.csrs.get("vstart", 0))
        if vl % group_size or start % group_size or start > vl:
            raise SemanticUnsupported(f"{form} requires vl and vstart multiples of {group_size}")

        element_group_width = (
            256 if form.startswith("vsm3") or (form.startswith("vsha2") and sew == 64)
            else 128
        )
        if int(self.csrs.get("vlenb", 32)) * 8 * lmul < element_group_width:
            raise SemanticUnsupported(f"{form} LMUL*VLEN is smaller than {element_group_width} bits")

        vd = register("vd")
        vs2 = register("vs2")
        uses_vs1 = form in {
            "vghsh.vv", "vsha2ch.vv", "vsha2cl.vv", "vsha2ms.vv", "vsm3me.vv",
        }
        vs1 = register("vs1") if uses_vs1 else 0
        vd_group = register_group(vd, lmul)
        vs2_emul = max(1, element_group_width / (int(self.csrs.get("vlenb", 32)) * 8)) \
            if form.endswith(".vs") else lmul
        vs2_group = register_group(vs2, vs2_emul)
        if form.endswith(".vs") and form.startswith(("vaes", "vsm4r")):
            if vd_group & vs2_group:
                raise SemanticUnsupported(f"{form} destination overlaps its scalar source")
        if form.startswith("vsha2"):
            vs1_group = register_group(vs1, lmul)
            if vd_group & vs2_group or vd_group & vs1_group:
                raise SemanticUnsupported(f"{form} destination overlaps a source group")
        elif form.startswith("vsm3") and vd_group & vs2_group:
            raise SemanticUnsupported(f"{form} destination overlaps vs2")
        immediate = int(fields.get("zimm5", _immediate(item)))
        mask = (1 << sew) - 1

        def read(register: int, base: int, count: int = group_size) -> list[int]:
            return [self._vector_get(register, base + offset, sew) for offset in range(count)]

        def write(base: int, values: list[int]) -> None:
            for offset, value in enumerate(values):
                self._vector_set(vd, base + offset, int(value) & mask, sew)

        for base in range(start, vl, group_size):
            if form.startswith("vaes"):
                state = read(vd, base)
                key = read(vs2, 0 if form.endswith(".vs") else base)
                if form in {"vaeskf1.vi", "vaeskf2.vi"}:
                    previous = state if form == "vaeskf2.vi" else None
                    current = read(vs2, base)
                    write(base, _aes_key_schedule(current, previous, immediate, form))
                else:
                    write(base, _aes_vector_round(state, key, form))
            elif form in {"vghsh.vv", "vgmul.vv"}:
                left = read(vd, base)
                right = read(vs2, base)
                left_bits = _pack_vector_group(left, 32)
                right_bits = _pack_vector_group(right, 32)
                y = _reverse_bits_each_byte(left_bits, 128)
                h = _reverse_bits_each_byte(right_bits, 128)
                if form == "vghsh.vv":
                    x = _reverse_bits_each_byte(_pack_vector_group(read(vs1, base), 32), 128)
                    y ^= x
                result = _reverse_bits_each_byte(_ghash_multiply(y, h), 128)
                write(base, _vector_group([result >> (32 * index) for index in range(4)], 0, 4, 32))
            elif form == "vsha2ms.vv":
                old = read(vd, base)
                middle = read(vs2, base)
                recent = read(vs1, base)
                write(base, _sha2_schedule(old, middle, recent, sew))
            elif form in {"vsha2ch.vv", "vsha2cl.vv"}:
                result = _sha2_compress_pair(
                    read(vs2, base), read(vd, base), read(vs1, base), sew,
                    high=form == "vsha2ch.vv",
                )
                write(base, result)
            elif form == "vsm3me.vv":
                write(base, _sm3_message_schedule(read(vs1, base), read(vs2, base)))
            elif form == "vsm3c.vi":
                write(base, _sm3_compress_pair(read(vd, base), read(vs2, base), immediate & 0x1F))
            elif form == "vsm4k.vi":
                write(base, _sm4_key_rounds(read(vs2, base), immediate))
            else:
                write(base, _sm4_rounds(read(vd, base), read(vs2, 0 if form.endswith(".vs") else base)))

        self.csrs["vstart"] = 0
        return True

    def _vector(self, item: object, form: str) -> bool:
        if not (form.startswith("v") or form.startswith("vm")):
            return False
        if form in {"vfrsub.vv", "vfrdiv.vv"}:
            raise SemanticUnsupported(f"{form} has no ratified vector form")
        self.vector_used = True
        if not self._vector_status_implemented() or not (
            int(self.csrs.get("mstatus", 0)) & (3 << 9)
        ):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if _is_vector_fp_form(form):
            self.fp_used = True
        vector_fp_flags_modeled = (
            form in _VECTOR_FP_REDUCTION_FORMS
            or form in {"vfcvt.f.x.v", "vfcvt.f.xu.v"}
            or form.startswith((
                "vfadd.", "vfsub.", "vfrsub.", "vfmul.", "vfdiv.", "vfrdiv.",
                "vfmin.", "vfmax.", "vfsqrt.",
                "vmfeq.", "vmfne.", "vmflt.", "vmfle.", "vmfgt.", "vmfge.",
                "vfmacc.", "vfnmacc.", "vfmsac.", "vfnmsac.",
                "vfmadd.", "vfnmadd.", "vfmsub.", "vfnmsub.",
                "vfwmacc.", "vfwnmacc.", "vfwmsac.", "vfwnmsac.",
                "vfwadd.", "vfwsub.", "vfwmul.",
            )))
        whole_load = _VECTOR_WHOLE_LOAD_RE.fullmatch(form)
        whole_store = _VECTOR_WHOLE_STORE_RE.fullmatch(form)
        whole_register_transfer = whole_load is not None or whole_store is not None
        if form not in {"vsetvli", "vsetvl", "vsetivli"} and not whole_register_transfer \
                and int(self.csrs.get("vtype", 0)) & (1 << (self.xlen - 1)):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if _is_vector_fp_form(form) and not form.startswith(
            ("vfclass", "vfsgnj", "vfmv", "vfmerge", "vfslide")
        ) and not vector_fp_flags_modeled:
            self.fp_flags_exact = False
        fields = dict(getattr(item, "operand_fields", ()) or ())
        if form in {"vsetvli", "vsetvl", "vsetivli"}:
            rs1 = getattr(item, "rs1", None)
            rs1 = fields.get("rs1") if rs1 is None else rs1
            rs2 = getattr(item, "rs2", None)
            rs2 = fields.get("rs2") if rs2 is None else rs2
            rd = getattr(item, "rd", None)
            rd = fields.get("rd") if rd is None else rd
            zimm = (
                self._read_gpr(rs2)
                if form == "vsetvl"
                else int(fields.get("zimm11", fields.get("zimm10", 0)))
            )
            vill_bit = 1 << (self.xlen - 1)
            requested_vill = form == "vsetvl" and bool(zimm & vill_bit)
            reserved_vtype = (
                bool(zimm & (3 << 9))
                if form in {"vsetvli", "vsetivli"}
                else bool(zimm & ((vill_bit - 1) & ~0x1FF))
            )
            if requested_vill:
                reserved_vtype = reserved_vtype or zimm != vill_bit
            sew = 8 << ((zimm >> 3) & 7)
            altfmt = bool(zimm & (1 << _VTYPE_ALTFMT_BIT))
            altfmt_supported = "zvfbfa" in enabled_extensions(
                str(getattr(self.testcase, "isa_profile", ""))
            )
            reserved_vtype = reserved_vtype or (
                altfmt and (sew >= 32 or not altfmt_supported)
            )
            lmul_code = zimm & 7
            if requested_vill and not reserved_vtype:
                self.csrs["vtype"] = vill_bit
                self.csrs["vl"] = 0
                self.csrs["vlenb"] = 32
                self.csrs["vstart"] = 0
                self._write_gpr(rd, 0)
                return True
            if reserved_vtype or sew > 64 or lmul_code == 4:
                self.csrs["vtype"] = vill_bit
                self.csrs["vl"] = 0
                self.csrs["vlenb"] = 32
                self.csrs["vstart"] = 0
                self._write_gpr(rd, 0)
                return True
            previous_vtype = int(self.csrs.get("vtype", 0))
            previous_vl = int(self.csrs.get("vl", 0))
            previous_vlmax = (
                None if previous_vtype & vill_bit
                else self._vector_vlmax()[1]
            )
            self.csrs["vtype"] = zimm
            self.csrs["vlenb"] = 32
            sew, max_lanes = self._vector_vlmax()
            if form == "vsetivli":
                avl = int(fields.get("zimm5", fields.get("uimm", 0)))
            elif rs1 == 0 and rd == 0:
                if previous_vlmax is None or previous_vlmax != max_lanes:
                    raise SemanticUnsupported(
                        f"{form} rs1=rd=x0 with changed VLMAX is reserved"
                    )
                avl = previous_vl
            elif rs1 == 0:
                avl = max_lanes
            else:
                avl = self._read_gpr(rs1)
            self.csrs["vl"] = min(avl, max_lanes)
            self.csrs["vstart"] = 0
            self._write_gpr(rd, self.csrs["vl"])
            return True

        if whole_register_transfer:
            if not self._vector_vm_enabled(item):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            count = int((whole_load or whole_store).group("count"))
            eew_bytes = int(whole_load.group("bits") or 8) // 8 if whole_load else 1
            vlenb = int(self.csrs.get("vlenb", 32))
            register = int(
                fields.get("vd", 0) if whole_load
                else fields.get("vs3", fields.get("vd", 0))
            )
            if register % count or register + count > 32:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            total_bytes = count * vlenb
            effective_vl = total_bytes // eew_bytes
            vector_start = int(self.csrs.get("vstart", 0))
            if vector_start >= effective_vl:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            address = self._read_operand(item, "rs1")
            first_byte = vector_start * eew_bytes
            for byte_offset in range(first_byte, total_bytes, eew_bytes):
                physical_register = register + byte_offset // vlenb
                bit_offset = (byte_offset % vlenb) * 8
                width_mask = (1 << (eew_bytes * 8)) - 1
                if whole_load:
                    value = self._load(address + byte_offset, eew_bytes)
                    shifted_mask = width_mask << bit_offset
                    self.vreg[physical_register] = (
                        (self.vreg[physical_register] & ~shifted_mask)
                        | ((value & width_mask) << bit_offset)
                    )
                else:
                    value = (self.vreg[physical_register] >> bit_offset) & width_mask
                    self._store(address + byte_offset, eew_bytes, value)
            self.csrs["vstart"] = 0
            return True

        if int(self.csrs.get("vtype", 0)) & (1 << (self.xlen - 1)):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        sew, vl = self._vector_shape()
        vector_fma_signs = {
            "vfmacc": (False, False, True),
            "vfnmacc": (True, True, True),
            "vfmsac": (False, True, True),
            "vfnmsac": (True, False, True),
            "vfmadd": (False, False, False),
            "vfnmadd": (True, True, False),
            "vfmsub": (False, True, False),
            "vfnmsub": (True, False, False),
        }
        vector_widening_fma_signs = {
            "vfwmacc": (False, False),
            "vfwnmacc": (True, True),
            "vfwmsac": (False, True),
            "vfwnmsac": (True, False),
        }
        fma_operation, _, fma_suffix = form.partition(".")
        vector_fma = fma_operation in vector_fma_signs and fma_suffix in {"vv", "vf"}
        vector_widening_fma = (
            fma_operation in vector_widening_fma_signs and fma_suffix in {"vv", "vf"}
        )
        vector_widening_fp = form in _VECTOR_WIDENING_FP_ARITHMETIC_FORMS
        vector_fp_reduction = form in _VECTOR_FP_REDUCTION_FORMS
        vector_widening_fp_reduction = form == "vfwredosum.vs"
        if fma_operation in vector_fma_signs and not vector_fma:
            raise SemanticUnsupported(f"{form} vector FMA operand form is not modeled")
        if fma_operation in vector_widening_fma_signs and not vector_widening_fma:
            raise SemanticUnsupported(f"{form} vector widening FMA operand form is not modeled")
        if vector_fma and sew not in {16, 32, 64}:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if vector_fp_reduction and sew not in {16, 32, 64}:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if (
            vector_widening_fma or vector_widening_fp or vector_widening_fp_reduction
        ) and sew not in {16, 32}:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if vector_fma or vector_widening_fma or vector_widening_fp or vector_fp_reduction:
            extensions = enabled_extensions(
                str(getattr(self.testcase, "isa_profile", "")),
            )
            if sew == 16:
                vector_fp_supported = "zvfh" in extensions
            elif vector_widening_fma or vector_widening_fp_reduction \
                    or vector_widening_fp or sew == 64:
                vector_fp_supported = bool(extensions & {"v", "zve64d"})
            else:
                vector_fp_supported = bool(extensions & {"v", "zve32f"})
            if not vector_fp_supported:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        altfmt_independent_bf16 = (
            form in {"vfwcvtbf16.f.f.v", "vfncvtbf16.f.f.w"}
            and "zvfbfmin" in enabled_extensions(
                str(getattr(self.testcase, "isa_profile", ""))
            )
        )
        if (
            int(self.csrs.get("vtype", 0)) & (1 << _VTYPE_ALTFMT_BIT)
            and _is_vector_fp_form(form)
            and not altfmt_independent_bf16
        ):
            raise SemanticUnsupported(
                f"{form} Zvfbfa v0.9 alternate-format floating semantics are not modeled"
            )
        if form.startswith(("vfred", "vfwred")) and not vector_fp_reduction:
            raise SemanticUnsupported(f"{form} floating reduction order is not modeled")
        if form.startswith(("vfrec7", "vfrsqrt7")):
            raise SemanticUnsupported(f"{form} floating semantics are not modeled exactly")
        vector_start = int(self.csrs.get("vstart", 0))
        vlmax = self._vector_vlmax()[1]
        if vector_start and form.startswith(("vred", "vwred", "vfred", "vfwred")):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if vector_start > vlmax:
            raise SemanticUnsupported(f"vstart={vector_start} exceeds VLMAX={vlmax}")
        initial_v0 = self.vreg[0]
        vm_enabled = self._vector_vm_enabled(item)

        # The implicit mask reads v0 at EEW=1.  Reusing v0 as a data source
        # with another EEW in the same masked instruction is reserved.
        if not vm_enabled and form not in {
            "vmsbf.m", "vmsof.m", "vmsif.m", "viota.m", "vcpop.m", "vfirst.m",
        } and any(fields.get(name) == 0 for name in ("vs1", "vs2", "vs3")):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        vector_destination = fields.get("vd", getattr(item, "vd", None))
        mask_destination = form.startswith((
            "vmseq", "vmsne", "vmslt", "vmsle", "vmsgt",
            "vmfeq", "vmfne", "vmflt", "vmfle", "vmfgt", "vmfge",
            "vmadc", "vmsbc", "vmsbf", "vmsof", "vmsif",
        ))
        reduction_destination = form.startswith(("vred", "vwred", "vfred", "vfwred"))
        if (
            not vm_enabled and vector_destination is not None
            and int(vector_destination) == 0
            and not mask_destination and not reduction_destination
        ):
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        unmasked_only_forms = {
            "vmv.v.i", "vmv.v.v", "vmv.v.x", "vmv.s.x", "vmv.x.s",
            "vmv1r.v", "vmv2r.v", "vmv4r.v", "vmv8r.v",
            "vfmv.v.f", "vfmv.s.f", "vfmv.f.s", "vlm.v", "vsm.v",
        }
        if form in unmasked_only_forms and not vm_enabled:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if form.startswith(("vmerge.", "vfmerge.")) and vm_enabled:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        if form.startswith(("vadc", "vsbc")) and vm_enabled:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        if form in {
            "vmand.mm", "vmandn.mm", "vmor.mm", "vmorn.mm",
            "vmxor.mm", "vmnand.mm", "vmnor.mm", "vmxnor.mm",
        } and not vm_enabled:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        def active(index: int) -> bool:
            return vm_enabled or bool((initial_v0 >> index) & 1)

        def fp_sign(raw: int, width: int) -> int:
            return (int(raw) >> (width - 1)) & 1

        def fp_kind(raw: int, width: int) -> str:
            return _fp_format_kind(raw, width)

        def fp_fraction(raw: int, width: int) -> Fraction | None:
            return _fp_format_fraction(raw, width)

        def fp_compare(left: int, left_width: int, right: int, right_width: int) -> int:
            left_kind, right_kind = fp_kind(left, left_width), fp_kind(right, right_width)
            left_sign, right_sign = fp_sign(left, left_width), fp_sign(right, right_width)
            if left_kind == "infinity":
                if right_kind == "infinity":
                    return 0 if left_sign == right_sign else -1 if left_sign else 1
                return -1 if left_sign else 1
            if right_kind == "infinity":
                return 1 if right_sign else -1
            left_fraction = fp_fraction(left, left_width)
            right_fraction = fp_fraction(right, right_width)
            assert left_fraction is not None and right_fraction is not None
            return (left_fraction > right_fraction) - (left_fraction < right_fraction)

        def fp_round(
            exact: Fraction, width: int, rounding_mode: int,
            *, negative_zero: bool = False,
        ) -> tuple[int, int]:
            return _round_finite_fraction_with_flags(
                exact, width, rounding_mode, negative_zero=negative_zero,
            )

        def fp_binary(
            operation: str,
            left: int,
            left_width: int,
            right: int,
            right_width: int,
            result_width: int,
            rounding_mode: int,
        ) -> tuple[int, int]:
            left_kind, right_kind = fp_kind(left, left_width), fp_kind(right, right_width)
            left_sign, right_sign = fp_sign(left, left_width), fp_sign(right, right_width)
            left_fraction = fp_fraction(left, left_width)
            right_fraction = fp_fraction(right, right_width)
            if left_kind.endswith("nan") or right_kind.endswith("nan"):
                invalid = int(
                    left_kind == "signaling_nan" or right_kind == "signaling_nan"
                ) << 4
                return _canonical_nan_bits(result_width), invalid

            if operation in {"add", "sub"}:
                effective_right_sign = right_sign ^ int(operation == "sub")
                if left_kind == "infinity" or right_kind == "infinity":
                    if (
                        left_kind == "infinity" and right_kind == "infinity"
                        and left_sign != effective_right_sign
                    ):
                        return _canonical_nan_bits(result_width), 1 << 4
                    sign = left_sign if left_kind == "infinity" else effective_right_sign
                    return _fp_infinity_bits(sign, result_width), 0
                assert left_fraction is not None and right_fraction is not None
                exact = (
                    left_fraction + right_fraction
                    if operation == "add" else left_fraction - right_fraction
                )
                negative_zero = False
                if exact == 0:
                    negative_zero = (
                        left_sign
                        if left_fraction == 0 and right_fraction == 0
                        and left_sign == effective_right_sign
                        else rounding_mode == 2
                    )
                return fp_round(exact, result_width, rounding_mode, negative_zero=negative_zero)

            if operation == "mul":
                left_zero = left_kind == "finite" and left_fraction == 0
                right_zero = right_kind == "finite" and right_fraction == 0
                if (
                    left_kind == "infinity" and right_zero
                    or right_kind == "infinity" and left_zero
                ):
                    return _canonical_nan_bits(result_width), 1 << 4
                sign = left_sign ^ right_sign
                if left_kind == "infinity" or right_kind == "infinity":
                    return _fp_infinity_bits(sign, result_width), 0
                assert left_fraction is not None and right_fraction is not None
                exact = left_fraction * right_fraction
                return fp_round(
                    exact, result_width, rounding_mode,
                    negative_zero=bool(sign) if exact == 0 else False,
                )

            if operation != "div":
                raise SemanticUnsupported(f"vector floating binary operation {operation}")
            left_zero = left_kind == "finite" and left_fraction == 0
            right_zero = right_kind == "finite" and right_fraction == 0
            if (
                left_kind == right_kind == "infinity"
                or left_zero and right_zero
            ):
                return _canonical_nan_bits(result_width), 1 << 4
            sign = left_sign ^ right_sign
            if right_zero:
                flags = 1 << 3 if left_kind == "finite" and not left_zero else 0
                return _fp_infinity_bits(sign, result_width), flags
            if left_kind == "infinity":
                return _fp_infinity_bits(sign, result_width), 0
            if right_kind == "infinity":
                return int(bool(sign)) << (result_width - 1), 0
            assert left_fraction is not None and right_fraction is not None
            exact = left_fraction / right_fraction
            return fp_round(
                exact, result_width, rounding_mode,
                negative_zero=bool(sign) if exact == 0 else False,
            )

        def fp_minmax(
            left: int, left_width: int, right: int, right_width: int,
            *, minimum: bool,
        ) -> tuple[int, int]:
            left_kind, right_kind = fp_kind(left, left_width), fp_kind(right, right_width)
            invalid = int(
                left_kind == "signaling_nan" or right_kind == "signaling_nan"
            ) << 4
            left_nan, right_nan = left_kind.endswith("nan"), right_kind.endswith("nan")
            if left_nan and right_nan:
                return _canonical_nan_bits(left_width), invalid
            if left_nan:
                return _bits(right, left_width), invalid
            if right_nan:
                return _bits(left, left_width), invalid
            comparison = fp_compare(left, left_width, right, right_width)
            if comparison == 0:
                left_fraction = fp_fraction(left, left_width)
                if left_fraction == 0:
                    sign = (
                        fp_sign(left, left_width) | fp_sign(right, right_width)
                        if minimum else
                        fp_sign(left, left_width) & fp_sign(right, right_width)
                    )
                    return sign << (left_width - 1), 0
            choose_left = comparison <= 0 if minimum else comparison >= 0
            return _bits(left if choose_left else right, left_width), 0

        def fp_fma(
            first: int,
            first_width: int,
            second: int,
            second_width: int,
            addend: int,
            addend_width: int,
            result_width: int,
            rounding_mode: int,
            *,
            negate_product: bool = False,
            negate_addend: bool = False,
        ) -> tuple[int, int]:
            first_kind, second_kind = fp_kind(first, first_width), fp_kind(second, second_width)
            addend_kind = fp_kind(addend, addend_width)
            first_fraction = fp_fraction(first, first_width)
            second_fraction = fp_fraction(second, second_width)
            addend_fraction = fp_fraction(addend, addend_width)
            signaling_nan = any(
                kind == "signaling_nan" for kind in (first_kind, second_kind, addend_kind)
            )
            invalid_product = (
                first_kind == "infinity" and second_kind == "finite" and second_fraction == 0
                or second_kind == "infinity" and first_kind == "finite" and first_fraction == 0
            )
            if signaling_nan or invalid_product:
                return _canonical_nan_bits(result_width), 1 << 4
            if any(kind == "quiet_nan" for kind in (first_kind, second_kind, addend_kind)):
                return _canonical_nan_bits(result_width), 0

            product_sign = (
                fp_sign(first, first_width) ^ fp_sign(second, second_width)
                ^ int(negate_product)
            )
            addend_sign = fp_sign(addend, addend_width) ^ int(negate_addend)
            product_infinite = (
                first_kind == "infinity" or second_kind == "infinity"
            )
            addend_infinite = addend_kind == "infinity"
            if product_infinite:
                if addend_infinite and product_sign != addend_sign:
                    return _canonical_nan_bits(result_width), 1 << 4
                return _fp_infinity_bits(product_sign, result_width), 0
            if addend_infinite:
                return _fp_infinity_bits(addend_sign, result_width), 0

            assert first_fraction is not None and second_fraction is not None
            assert addend_fraction is not None
            product = first_fraction * second_fraction
            if negate_product:
                product = -product
            exact = product + (
                -addend_fraction if negate_addend else addend_fraction
            )
            negative_zero = False
            if exact == 0:
                negative_zero = (
                    bool(product_sign)
                    if product == 0 and addend_fraction == 0
                    and product_sign == addend_sign
                    else rounding_mode == 2
                )
            return fp_round(exact, result_width, rounding_mode, negative_zero=negative_zero)

        width_mask = (1 << sew) - 1

        def widened_width() -> int:
            if sew > 32:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            return sew * 2

        lmul = {
            0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0,
            5: 0.125, 6: 0.25, 7: 0.5,
        }.get(int(self.csrs.get("vtype", 0)) & 7)
        if lmul is None:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))

        def register_group(base: int, emul: float) -> set[int]:
            count = max(1, math.ceil(emul))
            if (
                emul < 0.125 or emul > 8
                or not 0 <= base < 32
                or (count > 1 and base % count)
                or base + count > 32
            ):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            return set(range(base, base + count))

        def require_disjoint(destination: set[int], *sources: set[int]) -> None:
            if any(destination & source for source in sources):
                raise SemanticUnsupported(f"{form} overlapping vector groups are not modeled")

        def require_source_eew_compatibility(
            *operands: tuple[set[int], int],
        ) -> None:
            for index, (first_group, first_eew) in enumerate(operands):
                for second_group, second_eew in operands[index + 1:]:
                    if first_eew != second_eew and first_group & second_group:
                        raise SemanticTrap(
                            2, int(item.pc_offset), _instruction_word(self.testcase, item),
                        )

        def require_widening_source_overlap(
            destination: set[int], source: set[int], source_emul: float,
        ) -> None:
            overlap = destination & source
            if not overlap:
                return
            destination_last = max(destination)
            legal_high_overlap = (
                source_emul >= 1
                and source == set(range(
                    destination_last - len(source) + 1,
                    destination_last + 1,
                ))
            )
            if not legal_high_overlap:
                raise SemanticTrap(
                    2, int(item.pc_offset), _instruction_word(self.testcase, item),
                )

        def fault_only_first_load(
            field_address: int, index: int, element_bytes: int,
        ) -> int | None:
            try:
                return self._load(field_address, element_bytes)
            except SemanticUnsupported as error:
                if (
                    not self.region_meta
                    or not str(error).startswith("memory access outside testcase regions:")
                ):
                    raise
                if index == 0:
                    self.csrs["vstart"] = 0
                    raise SemanticTrap(5, int(item.pc_offset), field_address) from error
                self.csrs["vl"] = index
                self.csrs["vstart"] = 0
                return None

        if self._vector_crypto(item, form, fields, sew, vl):
            return True
        if vector_start and form in {
            "vmv.s.x", "vmv.x.s", "vfmv.s.f", "vfmv.f.s",
            "vmv1r.v", "vmv2r.v", "vmv4r.v", "vmv8r.v",
            "vcompress.vm", "vcpop.m", "vfirst.m", "viota.m",
            "vmsbf.m", "vmsof.m", "vmsif.m",
        }:
            raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
        segment = _VECTOR_SEGMENT_MEMORY_RE.fullmatch(form)
        indexed_segment = re.fullmatch(
            r"vs(?P<mode>ux|ox)ei(?P<bits>8|16|32|64)\.v", form,
        )
        encoded_nf = _field(item, "nf", 0)
        catalog_indexed_segment = indexed_segment is not None and encoded_nf > 0
        if catalog_indexed_segment or "seg" in form and form.startswith(("vl", "vs")):
            if segment is None and not catalog_indexed_segment:
                raise SemanticUnsupported(f"{form} segment memory encoding is not modeled")
            if catalog_indexed_segment:
                assert indexed_segment is not None
                operation, mode = "vs", indexed_segment.group("mode")
                field_count = encoded_nf + 1
                indexed = True
                data_width = sew
                index_width = int(indexed_segment.group("bits"))
            else:
                assert segment is not None
                operation, mode = segment.group("op"), segment.group("mode") or "unit"
                field_count = int(segment.group("nf"))
                indexed = segment.group("width") == "ei"
                data_width = sew if indexed else int(segment.group("bits"))
                index_width = int(segment.group("bits")) if indexed else None
            if indexed:
                assert index_width is not None
                self._check_vector_index_width(item, index_width)
            emul = lmul * data_width / sew
            registers_per_field = max(1, math.ceil(emul))
            base_register = int(
                fields.get("vd", 0) if operation == "vl"
                else fields.get("vs3", fields.get("vd", 0))
            )
            if (
                field_count * emul > 8
                or base_register + field_count * registers_per_field > 32
            ):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            data_groups = [
                register_group(base_register + field * registers_per_field, emul)
                for field in range(field_count)
            ]
            index_register = int(fields.get("vs2", 0)) if indexed else None
            if indexed:
                assert index_width is not None and index_register is not None
                index_group = register_group(index_register, lmul * index_width / sew)
                if operation == "vs":
                    for group in data_groups:
                        require_source_eew_compatibility(
                            (index_group, index_width), (group, data_width),
                        )
                elif any(index_group & group for group in data_groups):
                    raise SemanticUnsupported(f"{form} overlapping index/data groups are not modeled")
            address = self._read_operand(item, "rs1")
            stride = (
                _signed(self._read_operand(item, "rs2"), self.xlen)
                if mode == "s" else field_count * (data_width // 8)
            )
            element_bytes = data_width // 8
            stores_by_lane: list[tuple[int, list[tuple[int, int]]]] = []
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                if indexed:
                    assert index_register is not None and index_width is not None
                    offset = self._vector_get(index_register, index, index_width)
                else:
                    offset = index * stride
                lane_stores: list[tuple[int, int]] = []
                for field in range(field_count):
                    field_register = base_register + field * registers_per_field
                    field_address = _bits(
                        address + offset + field * element_bytes, self.xlen,
                    )
                    if operation == "vl":
                        value = self._load(field_address, element_bytes)
                        self._vector_set(field_register, index, value, data_width)
                    else:
                        value = self._vector_get(field_register, index, data_width)
                        lane_stores.append((field_address, value))
                if lane_stores:
                    stores_by_lane.append((index, lane_stores))
            if operation == "vs":
                if indexed and mode == "ox":
                    # Segment order is fixed, but fields inside the faulting
                    # segment are unordered by the V spec; see the fault gap.
                    for index, lane_stores in stores_by_lane:
                        try:
                            for store_address, _ in lane_stores:
                                self._locate(store_address, element_bytes)
                        except SemanticUnsupported as error:
                            if (
                                not self.region_meta
                                or not str(error).startswith("memory access outside testcase regions:")
                            ):
                                raise
                            self.csrs["vstart"] = index
                            raise SemanticUnsupported(
                                f"{form} faulting segment may partially commit fields"
                            ) from error
                        for store_address, value in lane_stores:
                            self._store(store_address, element_bytes, value)
                    self.csrs["vstart"] = 0
                elif indexed and mode == "ux":
                    stores = [store for _, lane_stores in stores_by_lane for store in lane_stores]
                    byte_values: dict[int, int] = {}
                    for store_address, value in stores:
                        for byte_index in range(element_bytes):
                            byte_address = store_address + byte_index
                            byte_value = (value >> (byte_index * 8)) & 0xFF
                            previous = byte_values.get(byte_address)
                            if previous is not None and previous != byte_value:
                                raise SemanticUnsupported(
                                    f"{form} conflicting overlapping stores have no unique result"
                                )
                            byte_values[byte_address] = byte_value
                    for store_address, _ in stores:
                        self._locate(store_address, element_bytes)
                    for store_address, value in stores:
                        self._store(store_address, element_bytes, value)
                    self.csrs["vstart"] = 0
                else:
                    stores = [store for _, lane_stores in stores_by_lane for store in lane_stores]
                    intervals = sorted(
                        (store_address, store_address + element_bytes)
                        for store_address, _ in stores
                    )
                    if any(
                        current_start < previous_end
                        for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
                    ):
                        raise SemanticUnsupported(f"{form} overlapping stores have order-dependent results")
                    for store_address, value in stores:
                        self._store(store_address, element_bytes, value)
            return True

        if form == "vmv.v.i":
            vd = int(fields.get("vd", 0)); value = _signed(int(fields.get("simm5", _immediate(item))), 5)
            for index in range(vector_start, vl): self._vector_set(vd, index, value, sew)
            return True
        if form == "vmv.v.v":
            vd, vs1 = int(fields.get("vd", 0)), int(fields.get("vs1", 0))
            for index in range(vector_start, vl):
                self._vector_set(vd, index, self._vector_get(vs1, index, sew), sew)
            return True
        if form == "vmv.v.x":
            vd = int(fields.get("vd", 0))
            value = _bits(_signed(self._read_operand(item, "rs1"), self.xlen), sew)
            for index in range(vector_start, vl): self._vector_set(vd, index, value, sew)
            return True
        if form == "vmv.s.x":
            if vector_start < vl:
                value = _bits(_signed(self._read_operand(item, "rs1"), self.xlen), sew)
                self._vector_set(int(fields.get("vd", 0)), 0, value, sew)
            return True
        if form == "vmv.x.s":
            value = self._vector_get(int(fields.get("vs2", 0)), 0, sew)
            self._write_operand(item, "rd", _signed(value, sew) if sew < self.xlen else value)
            return True
        if form in {"vmv1r.v", "vmv2r.v", "vmv4r.v", "vmv8r.v"}:
            vd, vs2 = int(fields.get("vd", 0)), int(fields.get("vs2", 0))
            groups = int(form[3])
            if vd % groups or vs2 % groups or vd + groups > 32 or vs2 + groups > 32:
                raise SemanticUnsupported(f"{form} invalid register group")
            self.vreg[vd:vd + groups] = self.vreg[vs2:vs2 + groups]
            return True
        if form == "vid.v":
            vd = int(fields.get("vd", 0))
            for index in range(vector_start, vl):
                if active(index):
                    self._vector_set(vd, index, index, sew)
            return True
        if form == "viota.m":
            vd, vs2 = int(fields.get("vd", 0)), int(fields.get("vs2", 0))
            destination_group = register_group(vd, lmul)
            if destination_group & register_group(vs2, 1.0):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            if not vm_enabled and 0 in destination_group:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            source_bits = [self._vector_mask_get(vs2, index) for index in range(vl)]
            count = sum(
                int(source_bits[index] and active(index))
                for index in range(vector_start)
            )
            for index in range(vector_start, vl):
                if active(index):
                    self._vector_set(vd, index, count, sew)
                    count += int(source_bits[index])
            return True
        if form == "vfmv.v.f":
            vd = int(fields.get("vd", 0))
            scalar = self._read_fp_bits(item, "rs1", sew)
            for index in range(vector_start, vl):
                self._vector_set(vd, index, scalar, sew)
            self.fp_used = True
            return True
        if form == "vfmv.s.f":
            vd = int(fields.get("vd", 0))
            if vector_start < vl:
                self._vector_set(vd, 0, self._read_fp_bits(item, "rs1", sew), sew)
            self.fp_used = True
            return True
        if form == "vfmv.f.s":
            vs2 = int(fields.get("vs2", 0))
            self._write_operand(item, "rd", self._vector_get(vs2, 0, sew), width=sew)
            self.fp_used = True
            return True
        if form == "vfclass.v":
            vd, vs2 = int(fields.get("vd", 0)), int(fields.get("vs2", 0))
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                raw = self._vector_get(vs2, index, sew)
                value = _float_from_bits(raw, sew)
                sign = bool(raw & (1 << (sew - 1)))
                fraction_bits = 10 if sew == 16 else 23 if sew == 32 else 52
                exponent_bits = sew - fraction_bits - 1
                exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
                fraction = raw & ((1 << fraction_bits) - 1)
                if math.isnan(value):
                    value = 1 << (9 if fraction & (1 << (fraction_bits - 1)) else 8)
                elif math.isinf(value): value = 1 << (0 if sign else 7)
                elif value == 0: value = 1 << (3 if sign else 4)
                elif exponent == 0: value = 1 << (2 if sign else 5)
                elif value < 0: value = 1 << 1
                else: value = 1 << 6
                self._vector_set(vd, index, int(value), sew)
            self.fp_used = True
            return True
        if form.startswith("vl"):
            if form == "vlm.v":
                vd = int(fields.get("vd", 0)); address = self._read_operand(item, "rs1")
                effective_vl = (vl + 7) // 8
                for byte_index in range(vector_start, effective_vl):
                    byte = self._load(address + byte_index, 1)
                    self._vector_set(vd, byte_index, byte, 8)
                return True
            fault_only_first = re.fullmatch(r"vle(?:8|16|32|64)ff\.v", form) is not None
            if "ff" in form and not fault_only_first:
                raise SemanticUnsupported(f"{form} fault-only-first memory form is not modeled")
            width = (
                sew if "ei" in form
                else next((int(part) for part in ("8", "16", "32", "64") if part in form), sew)
            )
            vd = int(fields.get("vd", 0)); address = self._read_operand(item, "rs1")
            destination_group = register_group(vd, lmul * width / sew)
            index_register = int(fields.get("vs2", 0)) if "ei" in form else None
            if index_register is not None:
                index_width = next(
                    (int(part) for part in ("8", "16", "32", "64") if f"ei{part}" in form),
                    sew,
                )
                self._check_vector_index_width(item, index_width)
                index_group = register_group(index_register, lmul * index_width / sew)
                if destination_group & index_group:
                    raise SemanticUnsupported(f"{form} overlapping index/data groups are not modeled")
            stride = width // 8
            if "se" in form:
                stride = _signed(self._read_operand(item, "rs2"), self.xlen)
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                offset = index * stride
                if "uxei" in form or "oxei" in form:
                    index_width = next(
                        (int(part) for part in ("8", "16", "32", "64") if f"ei{part}" in form),
                        sew,
                    )
                    offset = self._vector_get(index_register, index, index_width)
                field_address = _bits(address + offset, self.xlen)
                value = (
                    fault_only_first_load(field_address, index, width // 8)
                    if fault_only_first else self._load(field_address, width // 8)
                )
                if value is None:
                    return True
                self._vector_set(vd, index, value, width)
            if fault_only_first:
                self.csrs["vstart"] = 0
            return True
        ordered_indexed_store = re.fullmatch(r"vsoxei(?:8|16|32|64)\.v", form) is not None
        unordered_indexed_store = re.fullmatch(r"vsuxei(?:8|16|32|64)\.v", form) is not None
        if form.startswith(("vsoxei", "vsuxei")) and not (
            ordered_indexed_store or unordered_indexed_store
        ):
            raise SemanticUnsupported(f"{form} indexed store form is not modeled")
        if form.startswith((
            "vse8.", "vse16.", "vse32.", "vse64.",
            "vsse8.", "vsse16.", "vsse32.", "vsse64.",
            "vsuxei", "vsoxei", "vsm.", "vs1r.", "vs2r.", "vs4r.", "vs8r.",
        )):
            width = (
                sew if "ei" in form
                else next((int(part) for part in ("8", "16", "32", "64") if part in form), sew)
            )
            vs3 = int(fields.get("vs3", fields.get("vd", 0))); address = self._read_operand(item, "rs1")
            if form != "vsm.v":
                data_group = register_group(vs3, lmul * width / sew)
                index_register = int(fields.get("vs2", 0)) if "ei" in form else None
                if index_register is not None:
                    index_width = next(
                        (int(part) for part in ("8", "16", "32", "64") if f"ei{part}" in form),
                        sew,
                    )
                    self._check_vector_index_width(item, index_width)
                    index_group = register_group(index_register, lmul * index_width / sew)
                    require_source_eew_compatibility(
                        (data_group, width), (index_group, index_width),
                    )
            stride = width // 8
            if "sse" in form:
                stride = _signed(self._read_operand(item, "rs2"), self.xlen)
            if form == "vsm.v":
                effective_vl = (vl + 7) // 8
                for byte_index in range(vector_start, effective_vl):
                    self._store(
                        address + byte_index, 1,
                        self._vector_get(vs3, byte_index, 8),
                    )
            else:
                stores: list[tuple[int, int]] = []
                for index in range(vector_start, vl):
                    if not active(index):
                        continue
                    offset = index * stride
                    if "uxei" in form or "oxei" in form:
                        offset = self._vector_get(index_register, index, index_width)
                    store_address = _bits(address + offset, self.xlen)
                    value = self._vector_get(vs3, index, width)
                    if ordered_indexed_store:
                        try:
                            self._store(store_address, width // 8, value)
                        except SemanticUnsupported as error:
                            if (
                                not self.region_meta
                                or not str(error).startswith("memory access outside testcase regions:")
                            ):
                                raise
                            self.csrs["vstart"] = index
                            raise SemanticTrap(
                                7, int(item.pc_offset), store_address,
                            ) from error
                    else:
                        stores.append((store_address, value))
                if unordered_indexed_store:
                    byte_values: dict[int, int] = {}
                    for store_address, value in stores:
                        for byte_index in range(width // 8):
                            byte_address = store_address + byte_index
                            byte_value = (value >> (byte_index * 8)) & 0xFF
                            previous = byte_values.get(byte_address)
                            if previous is not None and previous != byte_value:
                                raise SemanticUnsupported(
                                    f"{form} conflicting overlapping stores have no unique result"
                                )
                            byte_values[byte_address] = byte_value
                    # Unordered indexed stores may fault in an implementation-
                    # dependent element order. Validate all active destinations
                    # before committing so a reference gap never leaves partial
                    # memory changes behind.
                    for store_address, _ in stores:
                        self._locate(store_address, width // 8)
                elif not ordered_indexed_store:
                    intervals = sorted(
                        (store_address, store_address + width // 8)
                        for store_address, _ in stores
                    )
                    if any(
                        current_start < previous_end
                        for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
                    ):
                        raise SemanticUnsupported(
                            f"{form} overlapping stores have order-dependent results"
                        )
                for store_address, value in stores:
                    self._store(store_address, width // 8, value)
            if ordered_indexed_store or unordered_indexed_store:
                self.csrs["vstart"] = 0
            return True
        vd = int(fields.get("vd", 0)); vs2 = int(fields.get("vs2", 0)); vs1 = int(fields.get("vs1", 0))
        scalar = (
            (
                self._read_fp_bits(item, "rs1", sew)
                if form.endswith((".vf", ".wf")) or form.startswith("vfmerge")
                else self._read_operand(item, "rs1")
            )
            if ".vx" in form or ".wx" in form or form.endswith((".vf", ".wf")) or form.startswith("vfmerge")
            else None
        )
        if vector_fma:
            register_group(vd, lmul)
            register_group(vs2, lmul)
            if fma_suffix == "vv":
                register_group(vs1, lmul)
        if vector_widening_fma:
            destination_group = register_group(vd, lmul * 2)
            source_groups = [register_group(vs2, lmul)]
            if fma_suffix == "vv":
                source_groups.append(register_group(vs1, lmul))
            for source_group in source_groups:
                require_widening_source_overlap(
                    destination_group, source_group, lmul,
                )
        if vector_widening_fp:
            destination_group = register_group(vd, lmul * 2)
            wide_source = form.endswith((".wv", ".wf"))
            vs2_emul = lmul * 2 if wide_source else lmul
            vs2_group = register_group(vs2, vs2_emul)
            if not wide_source:
                require_widening_source_overlap(destination_group, vs2_group, lmul)
            if form.endswith((".vv", ".wv")):
                vs1_group = register_group(vs1, lmul)
                if wide_source:
                    require_source_eew_compatibility(
                        (vs2_group, sew * 2), (vs1_group, sew),
                    )
                require_widening_source_overlap(destination_group, vs1_group, lmul)
        unsigned_immediate = form.startswith((
            "vrgather", "vslide", "vnsra", "vnsrl", "vssra", "vssrl",
            "vsll", "vsrl", "vsra", "vror", "vwsll", "vnclip",
        ))
        if ".vi" in form:
            raw_immediate = int(fields.get("simm5", fields.get("zimm5", _immediate(item))))
            immediate = (
                _bits(raw_immediate, 5) if unsigned_immediate
                else _signed(raw_immediate, 5)
            )
        else:
            immediate = None
        slide_source = (
            [self._vector_get(vs2, index, sew) for index in range(vlmax)]
            if form.startswith(("vslide", "vfslide")) else None
        )
        if form == "vcompress.vm":
            if not vm_enabled:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            destination_group = register_group(int(fields.get("vd", 0)), lmul)
            source_group = register_group(int(fields.get("vs2", 0)), lmul)
            source_mask = int(fields.get("vs1", 0))
            require_source_eew_compatibility(
                (source_group, sew),
                (register_group(source_mask, 1.0), 1),
            )
            if destination_group & source_group or source_mask in destination_group:
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            write = 0
            for index in range(vl):
                if self._vector_mask_get(vs1, index):
                    self._vector_set(vd, write, self._vector_get(vs2, index, sew), sew); write += 1
            return True
        if form in {"vslide1up.vx", "vslide1down.vx", "vfslide1up.vf", "vfslide1down.vf"}:
            scalar_value = (
                self._read_fp_bits(item, "rs1", sew)
                if form.endswith(".vf")
                else _bits(_signed(self._read_operand(item, "rs1"), self.xlen), sew)
            )
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                if form in {"vslide1up.vx", "vfslide1up.vf"}:
                    value = scalar_value if index == 0 else slide_source[index - 1]
                else:
                    value = slide_source[index + 1] if index + 1 < vl else scalar_value
                self._vector_set(vd, index, value, sew)
            return True
        if form == "vcpop.v":
            for index in range(vector_start, vl):
                if active(index):
                    value = self._vector_get(vs2, index, sew).bit_count()
                    self._vector_set(vd, index, value, sew)
            return True
        if form in {"vcpop.m", "vfirst.m"}:
            bits = [
                bool(self._vector_get(vs2, index, 1)) and active(index)
                for index in range(vl)
            ]
            value = sum(bits) if form == "vcpop.m" else next((index for index, bit in enumerate(bits) if bit), -1)
            rd = getattr(item, "rd", None)
            self._write_gpr(fields.get("rd") if rd is None else rd, value)
            return True
        if form in {"vmsbf.m", "vmsof.m", "vmsif.m"}:
            vd, vs2 = int(fields.get("vd", 0)), int(fields.get("vs2", 0))
            if vd == vs2 or (not vm_enabled and vd == 0):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            source = [self._vector_mask_get(vs2, index) for index in range(vl)]
            first = next(
                (index for index, bit in enumerate(source) if bit and active(index)),
                vl,
            )
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                value = index < first if form == "vmsbf.m" else index == first if form == "vmsof.m" else index <= first
                self._vector_mask_set(vd, index, value)
            return True
        if form in {"vmand.mm", "vmandn.mm", "vmor.mm", "vmorn.mm", "vmxor.mm", "vmnand.mm", "vmnor.mm", "vmxnor.mm"}:
            left_mask = [self._vector_mask_get(vs2, index) for index in range(vl)]
            right_mask = [self._vector_mask_get(vs1, index) for index in range(vl)]
            for index in range(vector_start, vl):
                left_bit, right_bit = left_mask[index], right_mask[index]
                value = (
                    left_bit and right_bit if form == "vmand.mm"
                    else left_bit and not right_bit if form == "vmandn.mm"
                    else left_bit or right_bit if form == "vmor.mm"
                    else left_bit or not right_bit if form == "vmorn.mm"
                    else left_bit != right_bit if form == "vmxor.mm"
                    else not (left_bit and right_bit) if form == "vmnand.mm"
                    else not (left_bit or right_bit) if form == "vmnor.mm"
                    else left_bit == right_bit
                )
                self._vector_mask_set(vd, index, bool(value))
            return True
        if vector_fp_reduction:
            rounding_mode = int(self.csrs.get("frm", 0))
            if form in {"vfredosum.vs", "vfwredosum.vs"} \
                    and rounding_mode not in {0, 1, 2, 3, 4}:
                raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
            widening = vector_widening_fp_reduction
            result_width = sew * 2 if widening else sew
            input_group = register_group(vs2, lmul)
            accumulator_group = register_group(vs1, 1.0)
            if widening:
                require_source_eew_compatibility(
                    (input_group, sew), (accumulator_group, result_width),
                )
            result_bits = self._vector_get(vs1, 0, result_width)
            if vl == 0:
                self._vector_set(vd, 0, result_bits, result_width)
                self.fp_used = True
                self.csrs["vstart"] = 0
                return True
            if form in {"vfredmax.vs", "vfredmin.vs"}:
                minimum = form == "vfredmin.vs"
                for index in range(vl):
                    if not active(index):
                        continue
                    value_bits = self._vector_get(vs2, index, sew)
                    result_bits, flags = fp_minmax(
                        result_bits, result_width, value_bits, sew, minimum=minimum,
                    )
                    self._accrue_fp_flags(flags)
            else:
                for index in range(vl):
                    if not active(index):
                        continue
                    result_bits, flags = fp_binary(
                        "add", result_bits, result_width,
                        self._vector_get(vs2, index, sew), sew,
                        result_width, rounding_mode,
                    )
                    self._accrue_fp_flags(flags)
            self._vector_set(vd, 0, result_bits, result_width)
            self.fp_used = True
            self.csrs["vstart"] = 0
            return True
        if form.startswith(("vwredsum",)):
            result_width = widened_width()
            accumulator = self._vector_get(vs1, 0, result_width)
            require_source_eew_compatibility(
                (register_group(vs2, lmul), sew),
                (register_group(vs1, 1.0), result_width),
            )
            total = _signed(accumulator, result_width) if not form.startswith("vwredsumu") else accumulator
            for index in range(vl):
                if not active(index):
                    continue
                element = self._vector_get(vs2, index, sew)
                total += element if form.startswith("vwredsumu") else _signed(element, sew)
            self._vector_set(vd, 0, total, result_width)
            return True
        if vector_widening_fp:
            rounding_mode = int(self.csrs.get("frm", 0))
            if rounding_mode not in {0, 1, 2, 3, 4}:
                raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
            wide_source = form.endswith((".wv", ".wf"))
            scalar_source = form.endswith((".vf", ".wf"))
            subtract = form.startswith("vfwsub")
            operation = "mul" if form.startswith("vfwmul") else "sub" if subtract else "add"
            results: list[tuple[int, int, int]] = []
            for index in range(vector_start, vl):
                if not active(index):
                    continue
                left_width = sew * 2 if wide_source else sew
                left_bits = self._vector_get(vs2, index, left_width)
                right_bits = (
                    int(scalar) if scalar_source
                    else self._vector_get(vs1, index, sew)
                )
                result, flags = fp_binary(
                    operation, left_bits, left_width, right_bits, sew,
                    sew * 2, rounding_mode,
                )
                results.append((index, result, flags))
            for index, result, flags in results:
                self._vector_set(vd, index, result, sew * 2)
                self._accrue_fp_flags(flags)
            self.fp_used = True
            self.csrs["vstart"] = 0
            return True
        if form in {
            "vzip.vv", "vunzipe.v", "vunzipo.v", "vpaire.vv", "vpairo.vv",
        }:
            raise SemanticUnsupported(f"{form} Zvzip draft semantics are not modeled")
        if form.startswith("vdot4") and form not in {
            "vdot4a.vv", "vdot4asu.vv", "vdot4au.vv",
        }:
            raise SemanticUnsupported(f"{form} dot-product operand form is not modeled")
        if form.startswith("vred"):
            accumulator = self._vector_get(vs1, 0, sew)
            if form.startswith(("vredmax", "vredmin")):
                unsigned = form.startswith(("vredmaxu", "vredminu"))
                compare = (lambda value: value) if unsigned else (lambda value: _signed(value, sew))
                accumulator = compare(accumulator)
                for index in range(vl):
                    if not active(index):
                        continue
                    element = compare(self._vector_get(vs2, index, sew))
                    accumulator = max(accumulator, element) if "max" in form else min(accumulator, element)
            elif form.startswith("vredand"):
                for index in range(vl):
                    if active(index): accumulator &= self._vector_get(vs2, index, sew)
            elif form.startswith("vredor"):
                for index in range(vl):
                    if active(index): accumulator |= self._vector_get(vs2, index, sew)
            elif form.startswith("vredxor"):
                for index in range(vl):
                    if active(index): accumulator ^= self._vector_get(vs2, index, sew)
            else:
                for index in range(vl):
                    if active(index):
                        accumulator = (accumulator + self._vector_get(vs2, index, sew)) & width_mask
            self._vector_set(vd, 0, accumulator, sew)
            return True
        merge_operation = form.startswith(("vmerge", "vfmerge"))
        carry_operation = form.startswith(("vadc", "vmadc", "vsbc", "vmsbc"))
        fma_results: list[tuple[int, int, int]] = []
        widening_fma_results: list[tuple[int, int, int]] = []
        fma_rounding_mode = int(self.csrs.get("frm", 0))
        if (vector_fma or vector_widening_fma) and fma_rounding_mode not in {0, 1, 2, 3, 4}:
            raise SemanticUnsupported(f"{form} reserved rounding mode {fma_rounding_mode}")
        for index in range(vector_start, vl):
            if not merge_operation and not carry_operation and not active(index):
                continue
            left = self._vector_get(vs2, index, sew)
            right = (
                immediate if immediate is not None
                else _signed(scalar, self.xlen) if scalar is not None and (".vx" in form or ".wx" in form)
                else scalar if scalar is not None
                else self._vector_get(vs1, index, sew)
            )
            right = _bits(right, sew)
            signed_left, signed_right = _signed(left, sew), _signed(right, sew)
            result_width = sew
            write_mask = False
            if vector_widening_fma:
                negate_product, negate_addend = vector_widening_fma_signs[fma_operation]
                first_bits, second_bits = (
                    (scalar, left) if fma_suffix == "vf"
                    else (right, left)
                )
                addend_bits = self._vector_get(vd, index, sew * 2)
                result, flags = fp_fma(
                    first_bits, sew, second_bits, sew,
                    addend_bits, sew * 2, sew * 2, fma_rounding_mode,
                    negate_product=negate_product,
                    negate_addend=negate_addend,
                )
                widening_fma_results.append((
                    index, result, flags,
                ))
                continue
            if vector_fma:
                negate_product, negate_addend, accumulates_into_vd = vector_fma_signs[fma_operation]
                if accumulates_into_vd:
                    first_bits, second_bits, addend_bits = left, right, self._vector_get(vd, index, sew)
                else:
                    first_bits, second_bits, addend_bits = self._vector_get(vd, index, sew), right, left
                result, flags = fp_fma(
                    first_bits, sew, second_bits, sew,
                    addend_bits, sew, sew, fma_rounding_mode,
                    negate_product=negate_product,
                    negate_addend=negate_addend,
                )
                fma_results.append((
                    index, result, flags,
                ))
                continue
            if form.startswith("vmf"):
                left_kind, right_kind = fp_kind(left, sew), fp_kind(right, sew)
                unordered = left_kind.endswith("nan") or right_kind.endswith("nan")
                quiet_comparison = form.startswith(("vmfeq", "vmfne"))
                flags = int(
                    left_kind == "signaling_nan" or right_kind == "signaling_nan"
                    if quiet_comparison else unordered
                ) << 4
                if unordered:
                    value = int(form.startswith("vmfne"))
                else:
                    comparison = fp_compare(left, sew, right, sew)
                    value = int(
                        comparison == 0 if form.startswith("vmfeq")
                        else comparison != 0 if form.startswith("vmfne")
                        else comparison <= 0 if form.startswith("vmfle")
                        else comparison < 0 if form.startswith("vmflt")
                        else comparison >= 0 if form.startswith("vmfge")
                        else comparison > 0
                    )
                self._vector_mask_set(vd, index, bool(value))
                self._accrue_fp_flags(flags)
                self.fp_used = True
                continue
            elif form.startswith(("vf", "vfw", "vfn")) and not form.startswith(
                ("vfcvt", "vfncvt", "vfwcvt", "vfclass", "vfmerge", "vfext")
            ):
                if form.startswith("vfsgnj"):
                    sign = 1 << (sew - 1)
                    raw = left & (sign - 1)
                    source_sign = right & sign
                    if form.startswith("vfsgnjn"):
                        source_sign ^= sign
                    elif form.startswith("vfsgnjx"):
                        source_sign = (left ^ right) & sign
                    self._vector_set(vd, index, raw | source_sign, sew)
                    self.fp_used = True
                    continue
                if form.startswith(("vfadd.", "vfsub.", "vfrsub.", "vfmul.", "vfdiv.", "vfrdiv.")):
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(
                            f"{form} reserved rounding mode {rounding_mode}"
                        )
                    operation = (
                        "add" if form.startswith("vfadd")
                        else "sub" if form.startswith(("vfsub", "vfrsub"))
                        else "mul" if form.startswith("vfmul")
                        else "div"
                    )
                    left_bits, right_bits = left, right
                    if form.startswith(("vfrsub", "vfrdiv")):
                        left_bits, right_bits = right, left
                    result, flags = fp_binary(
                        operation, left_bits, sew, right_bits, sew,
                        sew, rounding_mode,
                    )
                    self._vector_set(vd, index, result, sew)
                    self._accrue_fp_flags(flags)
                    continue
                if form.startswith(("vfmin.", "vfmax.")):
                    result, flags = fp_minmax(
                        left, sew, right, sew,
                        minimum=form.startswith("vfmin"),
                    )
                    self._vector_set(vd, index, result, sew)
                    self._accrue_fp_flags(flags)
                    continue
                if form.startswith("vfsqrt."):
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(
                            f"{form} reserved rounding mode {rounding_mode}"
                        )
                    kind = fp_kind(left, sew)
                    sign = fp_sign(left, sew)
                    if kind == "signaling_nan":
                        result, flags = _canonical_nan_bits(sew), 1 << 4
                    elif kind == "quiet_nan":
                        result, flags = _canonical_nan_bits(sew), 0
                    elif kind == "infinity":
                        result, flags = (
                            (_canonical_nan_bits(sew), 1 << 4)
                            if sign else (left, 0)
                        )
                    else:
                        exact_input = fp_fraction(left, sew)
                        assert exact_input is not None
                        if exact_input < 0:
                            result, flags = _canonical_nan_bits(sew), 1 << 4
                        elif exact_input == 0:
                            result, flags = left, 0
                        else:
                            result = _round_sqrt_to_bits(
                                _float_from_bits(left, sew), sew, rounding_mode,
                            )
                            rounded_root = fp_fraction(result, sew)
                            assert rounded_root is not None
                            flags = int(rounded_root * rounded_root != exact_input)
                    self._vector_set(vd, index, result, sew)
                    self._accrue_fp_flags(flags)
                    continue
                left_float = _float_from_bits(left, sew)
                right_float = _float_from_bits(right, sew)
                left_fraction = Fraction.from_float(left_float) if math.isfinite(left_float) else None
                right_fraction = Fraction.from_float(right_float) if math.isfinite(right_float) else None
                exact_result = None
                negative_zero = False
                if form.startswith(("vfmacc", "vfmadd", "vfmsac", "vfmsub", "vfnmacc", "vfnmadd", "vfnmsac", "vfnmsub", "vfwmacc", "vfwmsac", "vfwnmacc", "vfwnmsac")):
                    accumulator = _float_from_bits(self._vector_get(vd, index, sew), sew)
                    value = left_float * right_float + accumulator
                    if "fmsub" in form or "fmsac" in form:
                        value = left_float * right_float - accumulator
                    if form.startswith("vfnm"):
                        value = -value
                elif form.startswith("vfadd"):
                    exact_result = (
                        left_fraction + right_fraction
                        if left_fraction is not None and right_fraction is not None else None
                    )
                    value = left_float + right_float
                    if exact_result == 0:
                        right_sign = math.copysign(1.0, right_float) < 0
                        left_sign = math.copysign(1.0, left_float) < 0
                        negative_zero = (
                            left_sign if left_float == 0 and right_float == 0 and left_sign == right_sign
                            else int(self.csrs.get("frm", 0)) == 2
                        )
                elif form.startswith("vfrsub"):
                    exact_result = (
                        right_fraction - left_fraction
                        if left_fraction is not None and right_fraction is not None else None
                    )
                    value = right_float - left_float
                    if exact_result == 0:
                        left_sign = math.copysign(1.0, right_float) < 0
                        right_sign = not (math.copysign(1.0, left_float) < 0)
                        negative_zero = (
                            left_sign if right_float == 0 and left_float == 0 and left_sign == right_sign
                            else int(self.csrs.get("frm", 0)) == 2
                        )
                elif form.startswith("vfsub"):
                    exact_result = (
                        left_fraction - right_fraction
                        if left_fraction is not None and right_fraction is not None else None
                    )
                    value = left_float - right_float
                    if exact_result == 0:
                        right_sign = not (math.copysign(1.0, right_float) < 0)
                        left_sign = math.copysign(1.0, left_float) < 0
                        negative_zero = (
                            left_sign if left_float == 0 and right_float == 0 and left_sign == right_sign
                            else int(self.csrs.get("frm", 0)) == 2
                        )
                elif form.startswith("vfmul"):
                    exact_result = (
                        left_fraction * right_fraction
                        if left_fraction is not None and right_fraction is not None else None
                    )
                    value = left_float * right_float
                    negative_zero = (
                        (math.copysign(1.0, left_float) < 0)
                        ^ (math.copysign(1.0, right_float) < 0)
                    )
                elif form.startswith("vfrdiv"):
                    exact_result = (
                        right_fraction / left_fraction
                        if right_fraction is not None and left_fraction not in (None, 0) else None
                    )
                    value = _ieee_divide(right_float, left_float)
                    negative_zero = (
                        (math.copysign(1.0, right_float) < 0)
                        ^ (math.copysign(1.0, left_float) < 0)
                    )
                elif form.startswith("vfdiv"):
                    exact_result = (
                        left_fraction / right_fraction
                        if left_fraction is not None and right_fraction not in (None, 0) else None
                    )
                    value = _ieee_divide(left_float, right_float)
                    negative_zero = (
                        (math.copysign(1.0, left_float) < 0)
                        ^ (math.copysign(1.0, right_float) < 0)
                    )
                elif form.startswith("vfmin"):
                    value = _fp_minmax(left_float, right_float, minimum=True)
                elif form.startswith("vfmax"):
                    value = _fp_minmax(left_float, right_float, minimum=False)
                elif form.startswith("vfrec7"):
                    value = math.copysign(math.inf, left_float) if left_float == 0 else 1.0 / left_float
                elif form.startswith("vfrsqrt7"):
                    value = math.nan if left_float < 0 else (math.inf if left_float == 0 else 1.0 / math.sqrt(left_float))
                elif form.startswith(("vfbdota", "vfqwbdota", "vfqwdota")):
                    accumulator = _float_from_bits(self._vector_get(vd, index, sew), sew)
                    value = accumulator + left_float * right_float
                elif form.startswith("vfsqrt"):
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    if math.isnan(left_float):
                        result = _canonical_nan_bits(sew)
                    elif left_float < 0:
                        result = _canonical_nan_bits(sew)
                    else:
                        result = _round_sqrt_to_bits(left_float, sew, rounding_mode)
                    self._vector_set(vd, index, result, sew)
                    self.fp_used = True
                    continue
                else:
                    raise SemanticUnsupported(f"vector floating form {form}")
                result_bits = (
                    _round_fraction_to_bits(
                        exact_result, sew, int(self.csrs.get("frm", 0)),
                        negative_zero=negative_zero,
                    )
                    if exact_result is not None else _float_to_bits(value, sew)
                )
                self._vector_set(vd, index, result_bits, result_width)
                self.fp_used = True
                continue
            if form.startswith(("vfcvt.x.f", "vfcvt.xu.f", "vfcvt.rtz.x.f", "vfcvt.rtz.xu.f")):
                converted = _float_from_bits(left, sew)
                rounding_mode = 1 if ".rtz." in form else int(self.csrs.get("frm", 0))
                if rounding_mode not in {0, 1, 2, 3, 4}:
                    raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                unsigned = ".xu." in form
                result = _float_to_integer(
                    converted, sew, unsigned=unsigned, rounding_mode=rounding_mode,
                )
                self._vector_set(vd, index, result, sew)
                self.fp_used = True
                continue
            if form.startswith(("vfcvt.f.x", "vfcvt.f.xu", "vfcvt.rtz.f.x", "vfcvt.rtz.f.xu")):
                integer = left if ".xu." in form else signed_left
                rounding_mode = 1 if ".rtz." in form else int(self.csrs.get("frm", 0))
                if rounding_mode not in {0, 1, 2, 3, 4}:
                    raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                value, flags = _round_finite_fraction_with_flags(
                    Fraction(integer), sew, rounding_mode,
                )
                self._accrue_fp_flags(flags)
                self._vector_set(vd, index, value, sew)
                self.fp_used = True
                continue
            if form.startswith("vfwcvtbf16"):
                if sew != 16:
                    raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, 16)
                self._vector_set(vd, index, source << 16, 32)
                self.fp_used = True
                continue
            if form.startswith("vfwcvt"):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, sew)
                if ".xu.f." in form or ".x.f." in form:
                    converted = _float_from_bits(source, sew)
                    rounding_mode = 1 if ".rtz." in form else int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    value = _float_to_integer(
                        converted, result_width, unsigned=".xu." in form,
                        rounding_mode=rounding_mode,
                    )
                elif ".f.xu." in form or ".f.x." in form:
                    integer = source if ".xu." in form else _signed(source, sew)
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    value = _round_fraction_to_bits(Fraction(integer), result_width, rounding_mode)
                elif ".f.f." in form:
                    converted = _float_from_bits(source, sew)
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    value = (
                        _round_fraction_to_bits(
                            Fraction.from_float(converted), result_width, rounding_mode,
                            negative_zero=math.copysign(1.0, converted) < 0,
                        )
                        if math.isfinite(converted) else _float_to_bits(converted, result_width)
                    )
                else:
                    raise SemanticUnsupported(f"vector widening conversion {form}")
                self._vector_set(vd, index, value, result_width)
                self.fp_used = True
                continue
            if form.startswith("vfext"):
                if sew not in {32, 64}:
                    raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                source_width = sew // 2
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul * source_width / sew)
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, source_width)
                converted = _float_from_bits(source, source_width)
                value = _float_to_bits(converted, sew)
                self._vector_set(vd, index, value, sew)
                self.fp_used = True
                continue
            if form.startswith(("vfncvt",)):
                if form in {
                    "vfncvtbf16.sat.f.f.w",
                    "vfncvt.f.f.q",
                    "vfncvt.sat.f.f.q",
                }:
                    raise SemanticUnsupported(f"{form} draft OFP8 conversion is not modeled")
                if form.startswith("vfncvtbf16") and sew != 16:
                    raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                source_width = widened_width()
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul * 2)
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, source_width)
                if form.startswith("vfncvtbf16"):
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    value = _round_f32_to_bf16(source, rounding_mode)
                elif ".f.f." in form:
                    converted = _float_from_bits(source, source_width)
                    if ".rod." in form:
                        if math.isfinite(converted):
                            exact = Fraction.from_float(converted)
                            value = _round_fraction_to_bits(
                                exact, sew, 5,
                                negative_zero=math.copysign(1.0, converted) < 0,
                            )
                        else:
                            value = _float_to_bits(converted, sew)
                    else:
                        rounding_mode = int(self.csrs.get("frm", 0))
                        if rounding_mode not in {0, 1, 2, 3, 4}:
                            raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                        value = (
                            _round_fraction_to_bits(
                                Fraction.from_float(converted), sew, rounding_mode,
                                negative_zero=math.copysign(1.0, converted) < 0,
                            )
                            if math.isfinite(converted) else _float_to_bits(converted, sew)
                        )
                elif ".f.x." in form or ".f.xu." in form:
                    integer = source if ".xu." in form else _signed(source, source_width)
                    rounding_mode = int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    value = _round_fraction_to_bits(Fraction(integer), sew, rounding_mode)
                elif ".x.f." in form or ".xu.f." in form or ".rtz.x.f." in form or ".rtz.xu.f." in form:
                    converted = _float_from_bits(source, source_width)
                    rounding_mode = 1 if ".rtz." in form else int(self.csrs.get("frm", 0))
                    if rounding_mode not in {0, 1, 2, 3, 4}:
                        raise SemanticUnsupported(f"{form} reserved rounding mode {rounding_mode}")
                    unsigned = ".xu." in form
                    value = _float_to_integer(
                        converted, sew, unsigned=unsigned, rounding_mode=rounding_mode,
                    )
                else:
                    raise SemanticUnsupported(f"vector narrowing conversion {form}")
                self._vector_set(vd, index, value, sew)
                self.fp_used = True
                continue
            if form.startswith(("vwabda", "vwabdau")):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                value = (
                    abs(left - right) if form.startswith("vwabdau")
                    else abs(signed_left - signed_right)
                )
            elif form in {"vdot4a.vv", "vdot4asu.vv", "vdot4au.vv"}:
                if sew != 32:
                    raise SemanticUnsupported(f"{form} requires SEW=32")
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul)
                second_group = register_group(vs1, lmul)
                require_disjoint(destination_group, source_group, second_group)
                lhs_signed = form != "vdot4au.vv"
                rhs_signed = form == "vdot4a.vv"
                total = self._vector_get(vd, index, sew)
                for part in range(4):
                    lhs = self._vector_get(vs2, index * 4 + part, 8)
                    rhs = self._vector_get(vs1, index * 4 + part, 8)
                    total += (_signed(lhs, 8) if lhs_signed else lhs) * (_signed(rhs, 8) if rhs_signed else rhs)
                value = total
            elif form.startswith(("vsext", "vzext")):
                factor = 2 if form.endswith("vf2") else 4 if form.endswith("vf4") else 8
                source_width = sew // factor
                if source_width not in {8, 16, 32, 64}:
                    raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul * source_width / sew)
                require_disjoint(destination_group, source_group)
                source_value = self._vector_get(vs2, index, source_width)
                value = _signed(source_value, source_width) if form.startswith("vsext") else source_value
                result_width = sew
            elif form.startswith(("vnsra", "vnsrl")):
                source_width = widened_width()
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul * 2)
                if ".wv" in form:
                    require_source_eew_compatibility(
                        (source_group, source_width),
                        (register_group(vs1, lmul), sew),
                    )
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, source_width)
                shift = immediate if immediate is not None else scalar if scalar is not None else self._vector_get(vs1, index, sew)
                amount = int(shift) & (source_width - 1)
                value = _signed(source, source_width) >> amount if form.startswith("vnsra") else source >> amount
                result_width = sew
            elif form.startswith(("vssra", "vssrl")):
                shift = immediate if immediate is not None else scalar if scalar is not None else right
                amount = int(shift) & (sew - 1)
                raw = signed_left if form.startswith("vssra") else left
                value = _roundoff_fixed(raw, amount, int(self.csrs.get("vxrm", 0)))
            elif form.startswith("vwsll"):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                value = left << (right & (result_width - 1))
            elif form.startswith("vwmacc"):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                if ".vv" in form:
                    require_disjoint(destination_group, register_group(vs1, lmul))
                if form.startswith("vwmaccus"):
                    product = signed_left * right
                elif form.startswith("vwmaccsu"):
                    product = left * signed_right
                elif form.startswith("vwmaccu"):
                    product = left * right
                else:
                    product = signed_left * signed_right
                value = self._vector_get(vd, index, result_width) + product
            elif form.startswith("vwmul"):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(vs2, lmul)
                require_disjoint(destination_group, source_group)
                if ".vv" in form:
                    require_disjoint(destination_group, register_group(vs1, lmul))
                if form.startswith("vwmulu"):
                    value = left * right
                elif form.startswith("vwmulsu"):
                    value = signed_left * right
                else:
                    value = signed_left * signed_right
            elif form.startswith(("vwadd", "vwaddu", "vwsub", "vwsubu")):
                result_width = widened_width()
                destination_group = register_group(vd, lmul * 2)
                source_group = register_group(
                    vs2, lmul * 2 if (".wv" in form or ".wx" in form) else lmul,
                )
                narrow_source_group = None
                if ".wv" in form:
                    narrow_source_group = register_group(vs1, lmul)
                    require_source_eew_compatibility(
                        (source_group, result_width),
                        (narrow_source_group, sew),
                    )
                require_disjoint(destination_group, source_group)
                if narrow_source_group is not None:
                    require_disjoint(destination_group, narrow_source_group)
                unsigned = form.startswith(("vwaddu", "vwsubu"))
                subtract = form.startswith(("vwsub", "vwsubu"))
                if ".wv" in form or ".wx" in form:
                    wide_left = self._vector_get(vs2, index, result_width)
                    lhs = wide_left if unsigned else _signed(wide_left, result_width)
                    rhs = right if unsigned else signed_right
                else:
                    lhs = left if unsigned else signed_left
                    rhs = right if unsigned else signed_right
                value = lhs - rhs if subtract else lhs + rhs
            elif form.startswith(("vmul", "vwmul")):
                if form.startswith("vmulh"):
                    value = (signed_left * signed_right) >> sew
                elif form.startswith("vmulhu"):
                    value = (left * right) >> sew
                elif form.startswith("vmulhsu"):
                    value = (signed_left * right) >> sew
                else:
                    value = left * right
            elif form.startswith(("vdiv", "vrem")):
                unsigned = form.startswith(("vdivu", "vremu"))
                dividend = left if unsigned else signed_left
                divisor = right if unsigned else signed_right
                if divisor == 0:
                    value = width_mask if form.startswith("vdiv") else dividend
                elif form.startswith("vdiv") and not unsigned and dividend == -(1 << (sew - 1)) and divisor == -1:
                    value = dividend
                else:
                    quotient = abs(dividend) // abs(divisor)
                    if (dividend < 0) != (divisor < 0): quotient = -quotient
                    value = dividend - quotient * divisor if form.startswith("vrem") else quotient
            elif form.startswith(("vaadd", "vaaddu", "vasub", "vasubu")):
                unsigned = form.startswith(("vaaddu", "vasubu"))
                subtract = form.startswith(("vasub", "vasubu"))
                lhs, rhs = (left, right) if unsigned else (signed_left, signed_right)
                raw = lhs - rhs if subtract else lhs + rhs
                value = _roundoff_fixed(raw, 1, int(self.csrs.get("vxrm", 0)))
            elif form.startswith(("vabd", "vabdu")):
                value = abs(signed_left - signed_right) if not form.startswith("vabdu") else abs(left - right)
            elif form.startswith("vadc"):
                value = left + right + int(bool((initial_v0 >> index) & 1))
            elif form.startswith(("vadd", "vwadd", "vwaddu")):
                value = left + right
            elif form.startswith("vrsub"):
                value = right - left
            elif form.startswith(("vsub", "vwsub", "vwsubu")):
                value = left - right
            elif form.startswith("vandn"):
                value = left & ~right
            elif form.startswith("vand"):
                value = left & right
            elif form.startswith("vor"):
                value = left | right
            elif form.startswith("vxor"):
                value = left ^ right
            elif form.startswith("vsll"):
                value = left << (right & (sew - 1))
            elif form.startswith("vsrl"):
                value = left >> (right & (sew - 1))
            elif form.startswith("vsra"):
                value = signed_left >> (right & (sew - 1))
            elif form.startswith(("vmin", "vmax")):
                value = min(signed_left, signed_right) if form.startswith("vmin") else max(signed_left, signed_right)
                if "u" in form.split(".")[0]: value = min(left, right) if form.startswith("vmin") else max(left, right)
            elif form.startswith(("vmseq", "vmsne", "vmslt", "vmsle", "vmsgt")):
                unsigned = "u" in form.split(".")[0]
                lhs, rhs = (left, right) if unsigned else (signed_left, signed_right)
                if form.startswith("vmseq"): value = int(lhs == rhs)
                elif form.startswith("vmsne"): value = int(lhs != rhs)
                elif form.startswith("vmslt"): value = int(lhs < rhs)
                elif form.startswith("vmsle"): value = int(lhs <= rhs)
                else: value = int(lhs > rhs)
                write_mask = True
            elif form.startswith("vabs"):
                value = abs(signed_left)
            elif form.startswith("vneg"):
                value = -signed_left
            elif form.startswith("vnot"):
                value = ~left
            elif form.startswith(("vclz", "vctz")):
                value = sew if left == 0 else sew - left.bit_length() if form.startswith("vclz") else (left & -left).bit_length() - 1
            elif form.startswith("vbrev8"):
                value = sum(
                    _BIT_REVERSE_BYTE[(left >> offset) & 0xFF] << offset
                    for offset in range(0, sew, 8)
                )
            elif form.startswith("vbrev"):
                value = int(f"{left:0{sew}b}"[::-1], 2)
            elif form.startswith("vrev8"):
                value = int.from_bytes(left.to_bytes(max(1, sew // 8), "little"), "big")
            elif form.startswith(("vrol", "vror")):
                amount = right & (sew - 1)
                value = ((left << amount) | (left >> ((sew - amount) & (sew - 1)))) if form.startswith("vrol") else ((left >> amount) | (left << ((sew - amount) & (sew - 1))))
            elif form.startswith(("vclmul", "vclmulh")):
                product = 0
                for bit in range(sew):
                    if (right >> bit) & 1: product ^= left << bit
                value = product >> (sew if form.startswith("vclmulh") else 0)
            elif form.startswith(("vmadc", "vmsbc")):
                carry = int(form.endswith("m") and bool((initial_v0 >> index) & 1))
                total = left + right + carry if form.startswith("vmadc") else left - right - carry
                value = int(total >> sew != 0) if form.startswith("vmadc") else int(total < 0)
                write_mask = True
            elif form.startswith("vsbc"):
                value = left - right - int(bool((initial_v0 >> index) & 1))
            elif form.startswith(("vmacc", "vmadd", "vnmsac", "vnmsub")):
                accumulator = self._vector_get(vd, index, sew)
                product = left * right
                if form.startswith("vmacc"): value = accumulator + product
                elif form.startswith("vnmsac"): value = accumulator - product
                elif form.startswith("vmadd"): value = right * accumulator + left
                else: value = left - right * accumulator
            elif form.startswith("vrgather"):
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul)
                if destination_group & source_group:
                    raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                if form.endswith(".vv"):
                    index_width = 16 if form.startswith("vrgatherei16") else sew
                    index_group = register_group(vs1, lmul * index_width / sew)
                    require_source_eew_compatibility(
                        (source_group, sew), (index_group, index_width),
                    )
                    if destination_group & index_group:
                        raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
                    gather_index = self._vector_get(vs1, index, index_width)
                elif form.endswith(".vi"):
                    gather_index = int(immediate or 0)
                else:
                    gather_index = int(scalar or 0) & self.mask
                vlmax = self._vector_vlmax()[1]
                value = self._vector_get(vs2, gather_index, sew) if gather_index < vlmax else 0
            elif form.startswith("vsmul"):
                raw = _roundoff_fixed(
                    signed_left * signed_right, sew - 1,
                    int(self.csrs.get("vxrm", 0)),
                )
                lo, hi = -(1 << (sew - 1)), (1 << (sew - 1)) - 1
                value = max(lo, min(hi, raw))
                if value != raw:
                    self._set_vxsat(item, form)
            elif form.startswith(("vsaddu", "vssubu")):
                raw = left + right if form.startswith("vsaddu") else left - right
                value = max(0, min(width_mask, raw))
                if value != raw:
                    self._set_vxsat(item, form)
            elif form.startswith(("vsadd", "vssub")):
                raw = signed_left + signed_right if form.startswith("vsadd") else signed_left - signed_right
                lo, hi = -(1 << (sew - 1)), (1 << (sew - 1)) - 1
                value = max(lo, min(hi, raw))
                if value != raw:
                    self._set_vxsat(item, form)
            elif form.startswith("vmseq"):
                value = int(left == right)
            elif form.startswith("vmsne"):
                value = int(left != right)
            elif form.startswith("vmslt"):
                value = int(signed_left < signed_right)
            elif form.startswith("vnclip"):
                source_width = widened_width()
                destination_group = register_group(vd, lmul)
                source_group = register_group(vs2, lmul * 2)
                if ".wv" in form:
                    require_source_eew_compatibility(
                        (source_group, source_width),
                        (register_group(vs1, lmul), sew),
                    )
                require_disjoint(destination_group, source_group)
                source = self._vector_get(vs2, index, source_width)
                unsigned = form.startswith("vnclipu")
                source_value = source if unsigned else _signed(source, source_width)
                shift = immediate if immediate is not None else scalar if scalar is not None else right
                amount = int(shift) & (source_width - 1)
                rounded = _roundoff_fixed(
                    source_value, amount, int(self.csrs.get("vxrm", 0)),
                )
                lo, hi = (
                    (0, width_mask) if unsigned else
                    (-(1 << (sew - 1)), (1 << (sew - 1)) - 1)
                )
                value = max(lo, min(hi, rounded))
                if value != rounded:
                    self._set_vxsat(item, form)
            elif form.startswith("vslideup"):
                offset = int(immediate if immediate is not None else scalar or 0) & self.mask
                if index < offset:
                    continue
                value = slide_source[index - offset]
            elif form.startswith("vslidedown"):
                offset = int(immediate if immediate is not None else scalar or 0) & self.mask
                source_index = index + offset
                value = slide_source[source_index] if source_index < vlmax else 0
            elif form.startswith(("vmerge", "vfmerge")):
                selected = right if immediate is not None or scalar is not None else self._vector_get(vs1, index, sew)
                value = selected if (initial_v0 >> index) & 1 else self._vector_get(vs2, index, sew)
            else:
                raise SemanticUnsupported(f"vector form {form}")
            if write_mask:
                self._vector_mask_set(vd, index, bool(value))
            else:
                self._vector_set(vd, index, value, result_width)
        if vector_fma:
            for index, value, flags in fma_results:
                self._vector_set(vd, index, value, sew)
                self._accrue_fp_flags(flags)
        if vector_widening_fma:
            for index, value, flags in widening_fma_results:
                self._vector_set(vd, index, value, sew * 2)
                self._accrue_fp_flags(flags)
        if vector_fma or vector_widening_fma:
            self.fp_used = True
            self.csrs["vstart"] = 0
        return True

    # ----- top-level -------------------------------------------------------

    def run(self) -> SemanticResult:
        items = tuple(getattr(self.testcase, "instruction_meta", ()))
        index = 0
        visited: set[int] = set()
        while index < len(items):
            if index in visited:
                raise SemanticUnsupported("control-flow loop")
            visited.add(index)
            item = items[index]
            self.executed.append(int(item.pc_offset))
            form = _normal_form(getattr(item, "mnemonic", ""))
            if self.testcase.dataflow_meta.get("candidate_illegal_encoding") is True \
                    and "risk" in getattr(item, "tags", ()):
                raise SemanticTrap(2, int(item.pc_offset), _instruction_word(self.testcase, item))
            if form in {"ebreak", "c.ebreak"}:
                raise SemanticTrap(3, int(item.pc_offset))
            if form == "ecall":
                raise SemanticTrap(_environment_call_cause(self.testcase), int(item.pc_offset))
            if form == "cbo.zero":
                raise SemanticUnsupported("CBO.ZERO block size is not present in the testcase profile")
            if form in {
                "fence", "fence.i", "c.nop", "nop", "c.mop.n", "wfi",
                "wrs.nto", "wrs.sto", "cbo.clean", "cbo.flush", "cbo.inval",
            }:
                index += 1; continue
            if form in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}:
                left = self._read_gpr(getattr(item, "rs1", None)); right = self._read_gpr(getattr(item, "rs2", None)) if getattr(item, "rs2", None) is not None else 0
                op = (
                    "beq" if form == "c.beqz"
                    else "bne" if form == "c.bnez"
                    else form
                )
                taken = branch_condition_holds(left, right, op, xlen=self.xlen)
                target = int(item.pc_offset) + _immediate(item)
                end_offset = sum(int(x.byte_length) for x in items)
                if taken and target not in self._by_offset and target != end_offset:
                    raise SemanticUnsupported(f"invalid branch target: {form}")
                index = self._by_offset.get(target, len(items)) if taken else index + 1
                continue
            if form in {"jal", "c.j", "c.jal"}:
                target = int(item.pc_offset) + _immediate(item)
                if target not in self._by_offset and target != sum(int(x.byte_length) for x in items):
                    raise SemanticUnsupported(f"invalid jump target: {form}")
                self._write_gpr(getattr(item, "rd", None) if form == "jal" else 1 if form == "c.jal" else 0, int(getattr(self.testcase, "code_address", 0)) + int(item.pc_offset) + int(item.byte_length))
                index = self._by_offset.get(target, len(items)); continue
            if form in {"jalr", "c.jr", "c.jalr", "jr", "ret"}:
                base = self._read_gpr(getattr(item, "rs1", None)) if getattr(item, "rs1", None) is not None else self.gpr[1]
                target = (
                    _bits(base + _immediate(item), self.xlen) & ~1
                ) - int(getattr(self.testcase, "code_address", 0))
                self._write_gpr(1 if form == "c.jalr" else getattr(item, "rd", None) if form == "jalr" else 0, int(getattr(self.testcase, "code_address", 0)) + int(item.pc_offset) + int(item.byte_length))
                if target not in self._by_offset and target != sum(int(x.byte_length) for x in items):
                    raise SemanticUnsupported(f"invalid indirect jump target: {form}")
                index = self._by_offset.get(target, len(items)); continue
            handled = (
                self._csr(item, form)
                or self._memory_or_atomic(item, form)
                or self._fp(item, form)
                or self._vector(item, form)
                or self._integer(item, form)
            )
            if not handled:
                raise SemanticUnsupported(f"unsupported form: {form}")
            if form.startswith(("v", "vm")):
                self.csrs["vstart"] = 0
            index += 1
        memory = {region_id: bytes(data).hex() for region_id, data in self.regions.items()}
        extra: dict[str, Any] = {
            "observer_fields": ["executed_pcs", "outcome", "gpr", *[f"memory.{key}" for key in memory]],
        }
        if self.fp_used:
            extra.update({
                "fflags": int(self.csrs["fflags"]), "frm": int(self.csrs["frm"]),
                "fp_observer": "rvgen-semantic",
                "fflags_observer": (
                    "rvgen-semantic" if self.fp_flags_exact
                    else "gap: floating exception flags are not fully modeled"
                ),
            })
            if not self.x_register_fp:
                extra["fpr_rawbits"] = list(self.fpr)
                extra["csr.mstatus.fs"] = 3
        if self.vector_used:
            extra.update({
                "vxsat": int(self.csrs["vxsat"]), "vxrm": int(self.csrs["vxrm"]),
                "vector_csrs": {name: int(self.csrs[name]) for name in ("vl", "vtype", "vstart", "vlenb", "vxsat", "vxrm")},
                "vector_registers": {
                    f"v{index}": value
                    for index, value in enumerate(self.vreg)
                },
                "vector_observer_status": "rvgen-semantic",
            })
            boundary = str(getattr(self.testcase, "dataflow_meta", {}).get("vector_boundary", ""))
            if boundary.startswith("vector-mask:"):
                extra["vector.mask"] = boundary.removeprefix("vector-mask:")
            if boundary.startswith("vector-tail:"):
                extra["vector.tail"] = boundary.removeprefix("vector-tail:")
        if self.csr_used:
            extra["csr_values"] = dict(self.csrs)
            extra["csr_observer_status"] = (
                "gap: floating CSR flags are incomplete"
                if self.fp_used and not self.fp_flags_exact
                else "rvgen-semantic"
            )
            risk = next(
                (item for item in getattr(self.testcase, "instruction_meta", ()) if "risk" in item.tags),
                None,
            )
            extra["csr.address"] = _field(risk, "csr") if risk is not None else None
            dataflow = getattr(self.testcase, "dataflow_meta", {})
            realization = dataflow.get("generation_realization", {}) if isinstance(dataflow, dict) else {}
            privilege = (
                realization.get("privilege_class") or realization.get("privilege_mode") or "user"
                if isinstance(realization, dict) else "user"
            )
            extra["csr.privilege"] = str(
                privilege
            )
            extra["csr.warl.readback"] = dict(self.csrs)
            extra["csr.reserved-fields"] = 0
        return SemanticResult(tuple(self.gpr), memory, tuple(self.executed), extra)


def execute_semantic_case(testcase: object) -> SemanticResult:
    """执行扩展语义机；保留一个稳定的公共入口供 reference adapter 调用。"""

    return SemanticMachine(testcase).run()


__all__ = ["SemanticResult", "SemanticUnsupported", "SemanticTrap", "SemanticMachine", "execute_semantic_case"]

"""将 RVGEN TestCase 接入程序级 rewrite、RVEMI 和 MH。"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from contextlib import nullcontext

from framework._util import _compare_mask_fields, canonical_digest, is_sha256_digest
from framework.adapters.contracts import CAPSULE_BACKENDS, capsule_cpu_profile
from framework.adapters.runner import (
    CAPSULE_MAILBOX_ENV, CAPSULE_MEMORY_SIZE_ENV, CAPSULE_OBSERVATION_SIZE_ENV,
    CAPSULE_TEST_MEMORY_ENV, guest_trap_observed, observed_state_fields,
)
from framework.direct_case import (
    HARNESS_RESERVED_GPRS, OBSERVATION_HEADER_SIZE, CompareMask,
    ObservabilityContract, TestCase,
    _runner_contract_gap_observation,
    expected_trap_from_dataflow, final_fp_state_required_for_testcase,
)
from framework.riscv_catalog import OFFICIAL_ALL_CATALOG_FORMS, OFFICIAL_FIELD_BIT_RANGES
from framework.direct_elf import build_elf, symbol_offsets_from_elf
from framework.framework_coverage import run_backend_with_coverage
from framework.riscv_encoding import (
    canonical_immediate_field, decode_form, decode_register_field_value,
    decoded_register_slots, encode_instruction, encoded_immediate_for_metadata,
    form_for_mnemonic, is_register_operand_group, operand_groups, operand_kind,
    register_domain, rewrite_instruction_operands,
)
from framework.rvgen.boundaries import semantic_immediate_value, stack_shape
from framework.rvgen.ledger import identify_testcase
from framework.spec_definedness import (
    instruction_effect_class, xregister_compatibility_slot_domain_for_mnemonic,
)


_FRAGMENT_CACHE = {}


def _case(program: object) -> TestCase:
    params = getattr(program, "run_params", {})
    data = params.get("single_case") if isinstance(params, Mapping) else None
    if not isinstance(data, Mapping):
        raise ValueError("single program case is missing")
    testcase = _require_rvgen_case(TestCase.from_dict(dict(data)))
    expected = hashlib.sha256(testcase.code_bytes).hexdigest()
    if params.get("route") != "single" or params.get("single_case_sha256") != expected \
            or data.get("code_sha256") not in (None, expected):
        raise ValueError("single program case identity is invalid")
    for name, expected_value in (
        ("isa", testcase.isa_profile), ("isa_profile", testcase.isa_profile),
        ("input_id", testcase.input_id),
        ("rvgen_testcase_id", testcase.testcase_id),
        ("rvgen_base_program_id", testcase.base_program_id),
    ):
        value = params.get(name)
        if value is not None and str(value).lower() != str(expected_value).lower():
            raise ValueError("single program case identity is invalid")
    return testcase


def _require_rvgen_case(testcase: TestCase) -> TestCase:
    owner = testcase.dataflow_meta.get("generation_owner")
    if not isinstance(owner, Mapping) or owner.get("contract") != "rvgen-generation-owner-v1":
        raise ValueError("single route requires an RVGEN TestCase")
    return testcase


def _renode_preflight_gap(artifact: object, testcase: TestCase, backend: str):
    if backend != "renode-riscv64":
        return None
    profile = str(testcase.isa_profile)
    if capsule_cpu_profile(profile) is not None:
        return None
    params = artifact.run_params if isinstance(artifact.run_params, Mapping) else {}
    return (_runner_contract_gap_observation(
        backend,
        f"target-unsupported-isa-profile:{profile}",
        binary_sha256=getattr(artifact, "executable_sha256", None),
        profile_id=profile,
        input_id=params.get("input_id") if isinstance(params.get("input_id"), str) else None,
        extra_state={
            "target_unsupported": f"isa-profile:{profile}",
            "target_preflight": "renode-capability-contract-v1",
        },
    ),)


def is_single_program(program: object) -> bool:
    params = getattr(program, "run_params", {})
    if not isinstance(params, Mapping):
        return False
    case = params.get("single_case")
    owner = case.get("dataflow_meta", {}).get("generation_owner") if isinstance(case, Mapping) else None
    return params.get("route") == "single" and isinstance(case, Mapping) and isinstance(owner, Mapping) \
        and owner.get("contract") == "rvgen-generation-owner-v1"


def _register(value: int | None, domain: str | None) -> str | None:
    if value is None:
        return None
    prefix = {"fpr": "f", "vpr": "v"}.get(domain, "x")
    return f"{prefix}{value}"


def _assembly_mnemonic(item: object) -> str:
    mnemonic = str(item.mnemonic).replace("_", ".")
    form = form_for_mnemonic(mnemonic)
    fields = dict(getattr(item, "operand_fields", ()))
    if form is not None and form.mnemonic == "c_mop_N":
        return f"c.mop.{2 * int(fields['c_mop_t']) + 1}"
    if form is not None and form.mnemonic == "mop_r_N":
        number = (
            (int(fields["mop_r_t_30"]) << 4)
            | (int(fields["mop_r_t_27_26"]) << 2)
            | int(fields["mop_r_t_21_20"])
        )
        return f"mop.r.{number}"
    if form is not None and form.mnemonic == "mop_rr_N":
        number = (int(fields["mop_rr_t_30"]) << 2) | int(fields["mop_rr_t_27_26"])
        return f"mop.rr.{number}"
    return mnemonic


def _vector_type_operands(value: int) -> list[str]:
    sew = {0: 8, 1: 16, 2: 32, 3: 64}.get((value >> 3) & 7)
    lmul = {
        0: "m1", 1: "m2", 2: "m4", 3: "m8",
        5: "mf8", 6: "mf4", 7: "mf2",
    }.get(value & 7)
    return [] if sew is None or lmul is None else [
        f"e{sew}", lmul, "ta" if value & 0x40 else "tu", "ma" if value & 0x80 else "mu",
    ]


_ROUNDING_NAMES = ("rne", "rtz", "rdn", "rup", "rmm", None, None, "dyn")
_FLI_LITERALS = {
    0: "-1.0", 1: "min", 2: "0x1p-16", 3: "0x1p-15", 4: "0x1p-8", 5: "0x1p-7",
    6: "0x1p-4", 7: "0x1p-3", 8: "0x1p-2", 9: "0.3125", 10: "0.375",
    11: "0.4375", 12: "0.5", 13: "0.625", 14: "0.75", 15: "0.875",
    16: "1.0", 17: "1.25", 18: "1.5", 19: "1.75", 20: "2.0", 21: "2.5",
    22: "3.0", 23: "4.0", 24: "8.0", 25: "16.0", 26: "128.0",
    27: "256.0", 28: "32768.0", 29: "65536.0", 30: "inf", 31: "nan",
}
_FENCE_PREDICATES = ("0", "i", "o", "io", "r", "ir", "or", "ior", "w", "iw", "ow", "iow", "rw", "irw", "orw", "iorw")
_IMMEDIATE_DELTAS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16, -4096, 4096)
_SINGLE_FORM_SIBLINGS = {
    "ecall": ("ebreak",), "ebreak": ("ecall",),
    "wrs.nto": ("wrs.sto",), "wrs.sto": ("wrs.nto",),
    "c.ebreak": ("c.addi",), "c.fld": ("c.flw",),
}
_SINGLE_CANONICAL_ALIASES = {"c.addi": ("c.nop",)}


def _operands(item: object, xlen: int = 64) -> list[str]:
    fields = getattr(item, "operand_fields", ())
    if fields:
        form = form_for_mnemonic(str(item.mnemonic).replace(".", "_"))
        field_map = dict(fields)
        if form is not None and form.mnemonic == "c_nop" and int(field_map.get("c_imm6", -1)) == 0:
            return []
        groups = operand_groups(form) if form is not None else ()
        stack = None
        from framework.rvemi.program import _rule_facts
        if form is not None and "c_rlist" in field_map and "c_spimm" in field_map:
            stack = stack_shape(field_map, xlen=xlen)
            adjustment = stack[1] if form.mnemonic != "cm_push" else -stack[1]
            return [
                "{" + ",".join(f"x{value}" for value in stack[0]) + "}",
                str(adjustment),
            ]
        if form is not None and form.mnemonic in {"c_mop_N", "mop_r_N", "mop_rr_N"}:
            if form.mnemonic == "c_mop_N":
                return []
            roles = ("rd", "rs1") if form.mnemonic == "mop_r_N" else ("rd", "rs1", "rs2")
            return [_register(getattr(item, role), "gpr") for role in roles]
        effect = _rule_facts(item.mnemonic, xlen).get("effect")
        from framework.spec_definedness import instruction_effect_class
        effect_class = instruction_effect_class(item.mnemonic)
        if form is not None and form.mnemonic.startswith("v"):
            def vreg(role):
                return f"v{int(field_map[role])}"
            def xreg(role):
                return f"x{int(field_map[role])}"
            def sreg(role):
                domain = getattr(item, f"{role}_domain", None) or register_domain(form, role)
                return _register(int(field_map[role]), domain)
            mask = ["v0.t"] if field_map.get("vm") == 0 else []
            if form.mnemonic == "vsetvli":
                return [xreg("rd"), xreg("rs1"), str(field_map["zimm11"])]
            if form.mnemonic == "vsetivli":
                return [
                    xreg("rd"), str(field_map["zimm5"]),
                    *_vector_type_operands(int(field_map["zimm10"])),
                ]
            if form.mnemonic == "vsetvl":
                return [xreg("rd"), xreg("rs1"), xreg("rs2")]
            if "vs3" in field_map:
                values = [vreg("vs3"), f"0({xreg('rs1')})"]
                if "vs2" in field_map:
                    values.append(vreg("vs2"))
                elif "rs2" in field_map:
                    values.append(xreg("rs2"))
                return values + mask
            if form.mnemonic.startswith("vl") and "vd" in field_map:
                values = [vreg("vd"), f"0({xreg('rs1')})"]
                if "vs2" in field_map:
                    values.append(vreg("vs2"))
                elif "rs2" in field_map:
                    values.append(xreg("rs2"))
                return values + mask
            if "vs2" in field_map and "rd" in field_map:
                return [sreg("rd"), vreg("vs2")] + mask
            if "vd" in field_map:
                values = [vreg("vd")]
                scalar_first = "rs1" in field_map and form.mnemonic.startswith((
                    "vmacc_", "vnmsac_", "vmadd_", "vnmsub_", "vwmacc",
                    "vfmacc_", "vfnmacc_", "vfmsac_", "vfnmsac_",
                    "vfmadd_", "vfnmadd_", "vfmsub_", "vfnmsub_",
                    "vfwmacc", "vfwnmacc", "vfwmsac", "vfwnmsac",
                ))
                if scalar_first:
                    values.extend((sreg("rs1"), vreg("vs2")))
                elif "vs2" in field_map:
                    values.append(vreg("vs2"))
                if "vs1" in field_map:
                    values.append(vreg("vs1"))
                elif "rs1" in field_map and not scalar_first:
                    values.append(sreg("rs1"))
                elif any(field in field_map for field in ("simm5", "zimm5", "zimm6")):
                    immediate_field = next(
                        field for field in ("simm5", "zimm5", "zimm6") if field in field_map
                    )
                    values.append(str(getattr(item, "immediate", field_map[immediate_field])))
                return values + (["v0"] if form.mnemonic.endswith(("vim", "vxm", "vvm", "vfm")) else mask)
        if form is not None and form.mnemonic.startswith("cbo_"):
            return [f"0(x{int(field_map['rs1'])})"]
        if form is not None and form.mnemonic.startswith(("hfence_", "hinval_", "hlv_", "hlvx_", "hsv_")):
            def reg(role):
                return _register(decode_register_field_value(role, int(field_map[role])), "gpr")
            if form.mnemonic.startswith(("hlv_", "hlvx_")):
                return [reg("rd"), f"0({reg('rs1')})"]
            if form.mnemonic.startswith("hsv_"):
                return [reg("rs2"), f"0({reg('rs1')})"]
            return [reg("rs1"), reg("rs2")]
        if effect_class == "atomic":
            memory = f"0({_register(getattr(item, 'rs1'), 'gpr')})"
            if "rd" not in groups:
                return [_register(getattr(item, "rs2"), "gpr"), memory]
            values = [_register(getattr(item, "rd"), "gpr")]
            if "rs2" in groups:
                values.append(_register(getattr(item, "rs2"), "gpr"))
            return values + [memory]
        if form is not None and form.mnemonic.startswith("csrr"):
            source = (
                str(next(value for role, value in fields if role == "zimm5"))
                if "zimm5" in field_map
                else _register(getattr(item, "rs1"), "gpr")
            )
            return [_register(getattr(item, "rd"), "gpr"), str(field_map["csr"]), source]
        if effect == "private-load" and "c_rlist" not in field_map:
            base = _register(getattr(item, "rs1", None), getattr(item, "rs1_domain", "gpr"))
            value = getattr(item, "immediate", None)
            if base is not None and value is not None:
                return [
                    _register(getattr(item, "rd", None), getattr(item, "rd_domain", "gpr")),
                    f"{value}({base})",
                ]
        if effect == "control":
            return [
                str(getattr(item, "immediate")) if role == "imm"
                else _register(getattr(item, role), getattr(item, f"{role}_domain", "gpr"))
                for role in _rule_facts(item.mnemonic, xlen).get("operand_roles", ())
            ]
        if form is not None and form.mnemonic in {"fence_i", "fence_tso"}:
            return []
        if form is not None and form.mnemonic == "fence":
            return [_FENCE_PREDICATES[int(field_map["pred"])], _FENCE_PREDICATES[int(field_map["succ"])]]
        if form is not None and form.mnemonic in {
            "c_lbu", "c_lh", "c_lhu", "c_lw", "c_ld", "c_flw", "c_fld",
            "c_sb", "c_sh", "c_sw", "c_sd", "c_fsw", "c_fsd",
            "c_lwsp", "c_ldsp", "c_flwsp", "c_fldsp", "c_swsp", "c_sdsp",
            "c_fswsp", "c_fsdsp",
        }:
            load = form.mnemonic.startswith(("c_l", "c_fl"))
            store = form.mnemonic.startswith(("c_s", "c_fs"))
            data_group = next(
                group for group in groups
                if (load and "rd" in group) or (store and "rs2" in group)
            )
            immediate_group = next(group for group in groups if operand_kind(form, group) == "imm")
            register = decode_register_field_value(data_group, int(field_map[data_group]))
            canonical = "rs2" if "rs2" in data_group else "rd"
            domain = getattr(item, f"{canonical}_domain", None) or register_domain(form, canonical)
            base_group = next((group for group in groups if "rs1" in group), None)
            base = 2 if base_group is None else decode_register_field_value(base_group, int(field_map[base_group]))
            return [_register(register, domain), f"{field_map[immediate_group]}(x{base})"]
        base_group = next((group for group in groups if "rs1" in group), None)
        immediate_group = next(
            (group for group in groups if operand_kind(form, group) == "imm"), None,
        )
        data_group = next(
            (group for group in groups if group.startswith(("rs2", "fs2", "vs3"))), None,
        )
        if form is not None and not data_group and form.mnemonic.startswith("vl"):
            data_group = next(
                (group for group in groups if group.startswith(("rd", "fd", "vd"))), None,
            )
        if form is not None and base_group and immediate_group and data_group:
            data = decode_register_field_value(data_group, int(field_map[data_group]))
            base = decode_register_field_value(base_group, int(field_map[base_group]))
            domain = getattr(item, f"{data_group}_domain", None) or register_domain(form, data_group)
            immediate = getattr(item, "immediate", None)
            if immediate is None:
                immediate = int(field_map[immediate_group])
            return [_register(data, domain), f"{immediate}(x{base})"]
        values = []
        masked = False
        for role, raw in fields:
            if role in {"aq", "rl", "nf"}:
                continue
            if role == "vm":
                masked |= int(raw) == 0
                continue
            if role == "c_rlist":
                if stack is None:
                    return []
                values.append("{" + ",".join(f"x{value}" for value in stack[0]) + "}")
            elif role == "c_spimm":
                if stack is None:
                    return []
                adjustment = stack[1] if form.mnemonic != "cm_push" else -stack[1]
                values.append(str(adjustment))
            elif role == "rm":
                mode = _ROUNDING_NAMES[int(raw)] if 0 <= int(raw) < len(_ROUNDING_NAMES) else None
                if mode is None:
                    return []
                values.append(mode)
            elif form is not None and form.mnemonic.startswith("fli_") and role == "rs1":
                literal = "inf" if form.mnemonic == "fli_h" and int(raw) == 29 else _FLI_LITERALS.get(int(raw))
                if literal is None:
                    return []
                values.append(literal)
            elif role in {"csr", "bs", "rnum"} or operand_kind(form, role) == "imm":
                value = getattr(item, "immediate", None) if (
                    form is not None and role == canonical_immediate_field(form)
                    and role != "imm20"
                ) else int(raw)
                if form is not None and form.mnemonic == "c_lui" and role == "c_nzimm18":
                    value //= 4096
                    if value < 0:
                        value += 1 << 20
                values.append(str(value))
            elif role in {"vd", "vs1", "vs2", "vs3"}:
                values.append(f"v{raw}")
            elif role in {"fd", "fs1", "fs2", "fs3"}:
                values.append(f"f{raw}")
            else:
                value = decode_register_field_value(role, int(raw)) if form is not None else int(raw)
                canonical = next((name for name in ("rd", "rs1", "rs2", "rs3") if name in role), role)
                domain = getattr(item, f"{canonical}_domain", None)
                if domain is None and form is not None:
                    domain = register_domain(form, canonical)
                if value is not None:
                    values.append(_register(value, domain or ("fpr" if role.startswith("f") else "x")))
        if masked:
            values.append("v0.t")
        if values:
            return values
    values = [
        _register(getattr(item, role), getattr(item, f"{role}_domain"))
        for role in ("rd", "rs1", "rs2", "rs3")
    ]
    values = [value for value in values if value is not None]
    immediate = getattr(item, "immediate", None)
    if immediate is not None:
        values.append(str(immediate))
    return values[:4]


def _encoded_group_value(form: object, group: str, value: int) -> int:
    if group == "imm20":
        return int(value)
    if group == "c_imm6" and any(
        name.startswith("c_nzuimm6") for name in form.variable_fields
    ):
        return int(value)
    return int(semantic_immediate_value(group, int(value)))


def _metadata_immediate(form: object, decoded: Mapping[str, int]) -> int | None:
    group = canonical_immediate_field(form)
    value = decoded.get(group) if group is not None else None
    if value is None:
        return None
    if group == "c_imm6" and any(
        name.startswith("c_nzuimm6") for name in form.variable_fields
    ):
        return int(value)
    return semantic_immediate_value(group, int(value))


def _target_values(item: object, word: int, target_form: object) -> dict[str, int] | None:
    source_form = form_for_mnemonic(str(item.mnemonic).replace("_", "."))
    if source_form is None:
        return None
    decoded = decode_form(source_form, word)
    slots = decoded_register_slots(source_form, word)
    values = {}
    for group in operand_groups(target_form):
        if is_register_operand_group(target_form, group):
            roles = tuple(role for role in ("rd", "rs1", "rs2", "rs3") if role in group)
            value = next((slots.get(role) for role in roles if slots.get(role) is not None), None)
            if value is None and group in decoded:
                value = decode_register_field_value(group, int(decoded[group]))
            values[group] = 0 if value is None else int(value)
        elif group in decoded:
            values[group] = _encoded_group_value(source_form, group, int(decoded[group]))
        elif group == canonical_immediate_field(target_form) and item.immediate is not None:
            values[group] = (
                encoded_immediate_for_metadata(target_form, int(item.immediate))
                if group == "imm20" else int(item.immediate)
            )
        else:
            values[group] = 0
    return values


def _field_change_encodable(
    testcase: TestCase, item: object, changes: Mapping[str, int],
) -> bool:
    word = int.from_bytes(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length], "little"
    )
    try:
        return rewrite_instruction_operands(item.mnemonic, word, **dict(changes)) != word
    except (KeyError, TypeError, ValueError):
        return False


def _raw_fixed_field_changes(testcase: TestCase, item: object) -> tuple[dict[str, int], ...]:
    if "risk" not in getattr(item, "tags", ()) \
            or getattr(item, "operand_fields", ()) \
            or any(getattr(item, role, None) is not None for role in ("rd", "rs1", "rs2", "rs3")):
        return ()
    width = item.byte_length * 8
    word = int.from_bytes(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length], "little"
    )
    xlen = "rv32" if testcase.isa_profile.lower().startswith("rv32") else "rv64"
    result = []
    seen = set()
    for contract in testcase.dataflow_meta.get("risk_contracts", ()):
        boundary = str(contract.get("boundary_class", ""))
        if not boundary.startswith("fixed-field:"):
            continue
        field = boundary.split(":", 1)[1]
        msb, lsb = (28, 28) if field == "mew" else OFFICIAL_FIELD_BIT_RANGES.get(field, (-1, -1))
        if msb < lsb or msb >= width:
            continue
        field_mask = ((1 << (msb - lsb + 1)) - 1) << lsb
        current = (word & field_mask) >> lsb
        for value in range(1 << (msb - lsb + 1)):
            if value == current:
                continue
            candidate = (word & ~field_mask) | (value << lsb)
            if candidate in seen or any(
                form.encoding_length_bytes * 8 == width
                and form.xlen in {"shared", xlen}
                and candidate & form.mask == form.match
                for form in OFFICIAL_ALL_CATALOG_FORMS
            ):
                continue
            seen.add(candidate)
            result.append({"raw_word": candidate})
            break
    return tuple(result)


def _sibling_case(testcase: TestCase, mnemonic: str, code_bytes: bytes) -> TestCase | None:
    from framework.rvgen import generate_form

    for name in (mnemonic, *_SINGLE_CANONICAL_ALIASES.get(mnemonic, ())):
        try:
            cases = generate_form(name)
        except (TypeError, ValueError):
            continue
        candidate = next(
            (
                candidate for candidate in cases
                if candidate.isa_profile == testcase.isa_profile
                and candidate.code_bytes == code_bytes
            ),
            None,
        )
        if candidate is not None:
            return candidate
    return None


def _field_changes(testcase: TestCase, item: object) -> tuple[dict[str, int], ...]:
    form = form_for_mnemonic(str(item.mnemonic).replace("_", "."))
    if form is None:
        return ()
    fields = dict(getattr(item, "operand_fields", ()))
    canonical = canonical_immediate_field(form)
    result = []
    hidden = {}
    for group, raw in fields.items():
        if (group == canonical and isinstance(item.immediate, int)) or group == "rm":
            continue
        if group in {"rd", "rs1", "rs2", "rs3", "vd", "vs1", "vs2", "vs3"}:
            continue
        if is_register_operand_group(form, group):
            try:
                current = decode_register_field_value(group, int(raw))
            except (TypeError, ValueError):
                continue
            roles = tuple(role for role in ("rd", "rs1", "rs2", "rs3") if role in group)
            domain = next(
                (getattr(item, f"{role}_domain", None) for role in roles
                 if getattr(item, f"{role}_domain", None)),
                "gpr",
            )
            limit = 16 if domain == "gpr" and testcase.isa_profile.lower().startswith("rv32e") else 32
            reserved = HARNESS_RESERVED_GPRS if domain == "gpr" else ()
            alternatives = (
                (current + offset) % limit
                for offset in range(1, limit + 1)
            )
        else:
            limit = {
                "aq": 2, "rl": 2, "vm": 2, "nf": 8, "bs": 4,
                "rnum": 11, "c_mop_t": 8, "mop_r_t_30": 2,
                "mop_r_t_27_26": 4, "mop_r_t_21_20": 4,
                "mop_rr_t_30": 2, "mop_rr_t_27_26": 4,
                "zimm5": 32, "zimm10": 1024, "zimm11": 2048,
                "c_index": 256, "csr": 4096, "fm": 16, "pred": 16,
                "succ": 16, "c_rlist": 16,
            }.get(group)
            if limit is None:
                alternatives = (int(raw) + 1, int(raw) - 1, 0)
            else:
                alternatives = ((int(raw) + offset) % limit for offset in range(1, limit + 1))
        for value in alternatives:
            if value == int(raw) or (
                is_register_operand_group(form, group) and value in reserved
            ):
                continue
            if group == "c_index" and str(item.mnemonic).replace("_", ".") == "cm.jalt" \
                    and value < 32:
                continue
            if _field_change_encodable(testcase, item, {group: int(value)}):
                if group in {"aq", "rl", "nf", "fm"}:
                    hidden[group] = int(value)
                else:
                    result.append({group: int(value)})
                break
    if hidden and _field_change_encodable(testcase, item, hidden):
        result.append(hidden)
    return tuple(result)


def single_program_source(testcase: TestCase) -> str:
    lines = [
        ".section .text",
        ".globl _start",
        "_start:",
        f"  # RVGEN testcase {testcase.testcase_id}",
    ]
    for item in sorted(testcase.instruction_meta, key=lambda value: value.byte_offset):
        raw = testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length]
        if item.byte_length == 2:
            body = f".2byte 0x{int.from_bytes(raw, 'little'):04x}"
        elif item.byte_length == 4:
            body = f".word 0x{int.from_bytes(raw, 'little'):08x}"
        else:
            body = ".byte " + ", ".join(f"0x{value:02x}" for value in raw)
        lines.append(f"  {body}  # {item.instruction_id}:{item.mnemonic}")
    return "\n".join(lines) + "\n"


def _risks(testcase: TestCase) -> tuple[tuple[int, object], ...]:
    return tuple(
        (index, item) for index, item in enumerate(testcase.instruction_meta)
        if "risk" in item.tags
    )


def _register_written_before(
    testcase: TestCase, index: int, register: int, domain: str,
) -> bool:
    for item in testcase.instruction_meta[:index]:
        if getattr(item, "rd_domain", None) == domain and getattr(item, "rd", None) == register:
            return True
        fields = dict(getattr(item, "operand_fields", ()))
        destination = {"fpr": "fd", "vpr": "vd"}.get(domain)
        if destination is not None and fields.get(destination) == register:
            return True
    return False


def _registers_written_before(
    testcase: TestCase, index: int, domain: str,
) -> set[int]:
    return {
        register for register in range(32)
        if _register_written_before(testcase, index, register, domain)
    }


def _rewrite_case_instruction(
    testcase: TestCase, index: int, changes: Mapping[str, int],
) -> TestCase | None:
    item = testcase.instruction_meta[index]
    form = form_for_mnemonic(str(item.mnemonic).replace("_", "."))
    if form is None:
        return None
    word = int.from_bytes(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length], "little",
    )
    try:
        updated_word = rewrite_instruction_operands(item.mnemonic, word, **dict(changes))
        decoded = decode_form(form, updated_word)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    raw = bytearray(testcase.code_bytes)
    raw[item.byte_offset:item.byte_offset + item.byte_length] = updated_word.to_bytes(
        item.byte_length, "little",
    )
    slots = decoded_register_slots(form, updated_word)
    registers = {
        name: slots.get(name)
        for name in ("rd", "rs1", "rs2", "rs3")
    }
    registers.update({
        f"{name}_domain": xregister_compatibility_slot_domain_for_mnemonic(
            testcase.isa_profile, item.mnemonic, name, registers[name]
        ) if registers[name] is not None else None
        for name in ("rd", "rs1", "rs2", "rs3")
    })
    fields = tuple(
        (name, int(decoded[name]))
        for name in operand_groups(form) if name in decoded
    )
    updated_item = replace(
        item,
        **registers,
        immediate=_metadata_immediate(form, decoded),
        operand_fields=fields,
    )
    instruction_meta = list(testcase.instruction_meta)
    instruction_meta[index] = updated_item
    return replace(testcase, code_bytes=bytes(raw), instruction_meta=tuple(instruction_meta))


def _propagate_adjacent_producer_alias(
    testcase: TestCase, producer_index: int, old_register: int, new_register: int,
    domain: str,
) -> TestCase:
    """Keep the immediately following risk/consumer on a rewritten value.

    Producer-risk RVGEN rows intentionally form a short dataflow chain.  A
    register rewrite on the producer must move the consumer's encoded source
    operand with it; otherwise the candidate reads an unrelated FPR/VPR even
    though its GPR state is valid.
    """
    consumer_index = producer_index + 1
    if consumer_index >= len(testcase.instruction_meta):
        return testcase
    consumer = testcase.instruction_meta[consumer_index]
    if not set(consumer.tags).intersection({"risk", "consumer"}):
        return testcase
    role = next(
        (
            candidate for candidate in ("rs1", "rs2", "rs3")
            if getattr(consumer, candidate) == old_register
            and getattr(consumer, f"{candidate}_domain") == domain
        ),
        None,
    )
    if role is None:
        return testcase
    return _rewrite_case_instruction(testcase, consumer_index, {role: new_register}) or testcase


def _propagate_observation_alias(
    testcase: TestCase, old_register: int, new_register: int, domain: str,
) -> tuple[TestCase, bool]:
    contract = testcase.observability_contract
    sink_ids = set(contract.sink_instruction_ids) if contract is not None else set()
    changed = testcase
    rewrote_sink = False
    for index, item in enumerate(changed.instruction_meta):
        if item.instruction_id not in sink_ids:
            continue
        aliases = {
            role: new_register
            for role in ("rs1", "rs2", "rs3")
            if getattr(item, role, None) == old_register
            and getattr(item, f"{role}_domain", None) == domain
        }
        aliases.update({
            role: new_register
            for role, value in dict(item.operand_fields).items()
            if role in {"fs1", "fs2", "fs3", "vs1", "vs2", "vs3"}
            and value == old_register
            and ("fpr" if role.startswith("f") else "vpr") == domain
        })
        if aliases:
            rewritten = _rewrite_case_instruction(changed, index, aliases)
            if rewritten is None:
                return testcase, False
            changed = rewritten
            rewrote_sink = True

    if rewrote_sink:
        return changed, True
    if domain == "gpr" and old_register in testcase.compare_mask.gpr_indices:
        indices = tuple(
            new_register if value == old_register else value
            for value in testcase.compare_mask.gpr_indices
        )
        return replace(
            testcase, compare_mask=replace(testcase.compare_mask, gpr_indices=tuple(dict.fromkeys(indices))),
        ), True
    observed_fields = set(testcase.dataflow_meta.get("state_observer_keys", ()))
    observed_fields.update(testcase.compare_mask.extra_state_keys)
    if domain == "fpr" and "fpr_rawbits" in observed_fields:
        return testcase, True
    if domain == "vpr" and "vector.registers" in observed_fields:
        return testcase, True
    disposition = testcase.dataflow_meta.get("realized_facts", {}).get("disposition")
    if disposition in {"encoding-only", "state-ready"}:
        return testcase, True
    return testcase, False


def _memory_access_width(mnemonic: str) -> int | None:
    form = str(mnemonic).replace("_", ".").lower()
    widths = {
        "lb": 1, "lbu": 1, "sb": 1,
        "lh": 2, "lhu": 2, "sh": 2, "flh": 2, "fsh": 2,
        "lw": 4, "lwu": 4, "sw": 4, "flw": 4, "fsw": 4,
        "ld": 8, "sd": 8, "fld": 8, "fsd": 8,
        "flq": 16, "fsq": 16,
        "c.lw": 4, "c.sw": 4, "c.lwsp": 4, "c.swsp": 4,
        "c.ld": 8, "c.sd": 8, "c.ldsp": 8, "c.sdsp": 8,
        "c.lq": 16, "c.sq": 16, "c.lqsp": 16, "c.sqsp": 16,
    }
    if form in widths:
        return widths[form]
    if form.startswith(("amo", "lr.", "sc.")):
        return {"w": 4, "d": 8}.get(form.rsplit(".", 1)[-1])
    return None


def _memory_immediate_in_bounds(
    testcase: TestCase, index: int, item: object, immediate: int,
    changes: Mapping[str, object],
) -> bool:
    width = _memory_access_width(str(item.mnemonic))
    if width is None or not isinstance(getattr(item, "rs1", None), int):
        return True
    base_register = int(changes.get("rs1", item.rs1))
    if _register_written_before(testcase, index, int(item.rs1), "gpr") \
            or _register_written_before(testcase, index, base_register, "gpr"):
        return False
    base_value = int(testcase.initial_gpr[base_register])
    xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
    address = (base_value + int(immediate)) & ((1 << xlen) - 1)
    return any(
        int(region.address) <= address
        and address + width <= int(region.address) + len(region.data)
        for region in testcase.initial_memory_regions
    )


def _replace_candidate(testcase: TestCase, **changes: object) -> TestCase | None:
    try:
        return replace(testcase, **changes)
    except ValueError as error:
        if str(error) == "code bytes must be instruction-aligned":
            return None
        raise


def _mutate_at(testcase: TestCase, index: int, changes: Mapping[str, object]) -> TestCase | None:
    if not 0 <= index < len(testcase.instruction_meta):
        return None
    item = testcase.instruction_meta[index]
    if "imm" in changes and not _memory_immediate_in_bounds(
        testcase, index, item, int(changes["imm"]), changes,
    ):
        return None
    word = int.from_bytes(
        testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length], "little"
    )
    if "raw_word" in changes:
        raw_word = changes["raw_word"]
        if type(raw_word) is not int or not 0 <= raw_word < 1 << (8 * item.byte_length) \
                or raw_word == word:
            return None
        raw = bytearray(testcase.code_bytes)
        raw[item.byte_offset:item.byte_offset + item.byte_length] = raw_word.to_bytes(
            item.byte_length, "little"
        )
        dataflow = dict(testcase.dataflow_meta)
        dataflow["candidate_mutated"] = True
        dataflow["candidate_illegal_encoding"] = True
        changed = _replace_candidate(
            testcase, code_bytes=bytes(raw), dataflow_meta=dataflow,
        )
        return None if changed is None else identify_testcase(changed)
    target_mnemonic = str(changes.get("mnemonic", item.mnemonic))
    target_form = form_for_mnemonic(target_mnemonic)
    if target_form is None or target_form.encoding_length_bytes != item.byte_length:
        return None
    try:
        if "mnemonic" in changes:
            values = _target_values(item, word, target_form)
            if values is None:
                return None
            updated_word = encode_instruction(target_mnemonic, **values)
        else:
            updated_word = rewrite_instruction_operands(item.mnemonic, word, **dict(changes))
        decoded = decode_form(target_form, updated_word)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    raw = bytearray(testcase.code_bytes)
    raw[item.byte_offset:item.byte_offset + item.byte_length] = updated_word.to_bytes(
        item.byte_length, "little"
    )
    fields = dict(item.operand_fields)
    fields = {
        str(name): int(decoded[name])
        for name in operand_groups(target_form) if name in decoded
    }
    slots = decoded_register_slots(target_form, updated_word)
    registers = {
        role: slots.get(role)
        for role in ("rd", "rs1", "rs2", "rs3")
    }
    registers.update({
        f"{role}_domain": xregister_compatibility_slot_domain_for_mnemonic(
            testcase.isa_profile, target_mnemonic, role, registers[role]
        )
        if registers[role] is not None else None
        for role in ("rd", "rs1", "rs2", "rs3")
    })
    meta = replace(
        item,
        mnemonic=target_mnemonic,
        **registers,
        immediate=_metadata_immediate(target_form, decoded),
        operand_fields=tuple(fields.items()),
    )
    instruction_meta = list(testcase.instruction_meta)
    instruction_meta[index] = meta
    # Operand rewrites change the physical source register.  Keep the
    # generated state attached to the value rather than to the old register;
    # otherwise a control or memory rewrite can silently use zero/garbage.
    initial_gpr = list(testcase.initial_gpr)
    if "risk" in item.tags:
        for role in ("rs1", "rs2", "rs3"):
            old_register = getattr(item, role)
            new_register = getattr(meta, role)
            old_domain = getattr(item, f"{role}_domain")
            new_domain = getattr(meta, f"{role}_domain")
            if (
                old_domain == "gpr" and new_domain == "gpr"
                and old_register is not None and new_register is not None
                and new_register != 0
                and old_register != new_register
            ):
                if _register_written_before(testcase, index, old_register, "gpr") \
                        or _register_written_before(testcase, index, new_register, "gpr"):
                    return None
                initial_gpr[new_register] = initial_gpr[old_register]
    changed = replace(
        testcase,
        code_bytes=bytes(raw),
        instruction_meta=tuple(instruction_meta),
        initial_gpr=tuple(initial_gpr),
    )
    if (
        "producer" in item.tags
        and item.rd is not None
        and meta.rd is not None
        and item.rd_domain == meta.rd_domain
        and item.rd != meta.rd
    ):
        changed = _propagate_adjacent_producer_alias(
            changed, index, item.rd, meta.rd, item.rd_domain,
        )
    if "risk" in item.tags:
        old_destination = None
        new_destination = None
        destination_domain = None
        if item.rd is not None and meta.rd is not None:
            old_destination, new_destination = item.rd, meta.rd
            destination_domain = item.rd_domain
        else:
            old_fields, new_fields = dict(item.operand_fields), dict(meta.operand_fields)
            destination = next((name for name in ("vd", "fd") if name in old_fields), None)
            if destination is not None and destination in new_fields:
                old_destination, new_destination = old_fields[destination], new_fields[destination]
                destination_domain = "vpr" if destination == "vd" else "fpr"
        if (
            old_destination is not None and new_destination is not None
            and old_destination != new_destination and destination_domain in {"gpr", "fpr", "vpr"}
        ):
            changed, observed = _propagate_observation_alias(
                changed, old_destination, new_destination, destination_domain,
            )
            if not observed:
                return None
    sibling = (
        _sibling_case(changed, target_mnemonic, changed.code_bytes)
        if "mnemonic" in changes
        and target_mnemonic in _SINGLE_FORM_SIBLINGS.get(
            str(item.mnemonic).replace("_", "."), ()
        )
        else None
    )
    if sibling is not None:
        changed = replace(
            changed,
            generation_rule_id=sibling.generation_rule_id,
            instruction_meta=sibling.instruction_meta,
            dataflow_meta=sibling.dataflow_meta,
            initial_gpr=sibling.initial_gpr,
            initial_memory_regions=sibling.initial_memory_regions,
        )
    else:
        dataflow = dict(changed.dataflow_meta)
        dataflow["candidate_mutated"] = True
        changed = _replace_candidate(changed, dataflow_meta=dataflow)
        if changed is None:
            return None
    return identify_testcase(changed)


def _changes_for_item(testcase: TestCase, item: object) -> tuple[dict[str, object], ...]:
    changes: list[dict[str, object]] = []
    index = next((i for i, candidate in enumerate(testcase.instruction_meta) if candidate is item), -1)
    seeded = {
        domain: _registers_written_before(testcase, index, domain)
        for domain in ("fpr", "vpr")
    } if index >= 0 else {"fpr": set(), "vpr": set()}
    for role in ("rd", "rs1", "rs2", "rs3"):
        value = getattr(item, role)
        domain = getattr(item, f"{role}_domain")
        if value is not None and domain in {"gpr", "fpr", "vpr"}:
            limit = 16 if domain == "gpr" and testcase.isa_profile.lower().startswith("rv32e") else 32
            reserved = HARNESS_RESERVED_GPRS if domain == "gpr" else ()
            source = role != "rd"
            if source and domain == "gpr" and value in reserved \
                    and not (value == 0 and "risk" in getattr(item, "tags", ())):
                continue
            if source and "risk" in getattr(item, "tags", ()):
                if domain == "gpr" and index >= 0 and _register_written_before(
                    testcase, index, value, domain,
                ):
                    continue
                if domain in {"fpr", "vpr"} and value not in seeded[domain]:
                    continue
            source_risk = source and "risk" in getattr(item, "tags", ())
            eligible = tuple(
                candidate for candidate in range(limit)
                if (
                    candidate not in reserved
                    or source_risk and domain == "gpr" and candidate == 0
                )
                and (
                    not source_risk
                    or domain != "gpr" or index < 0
                    or not _register_written_before(testcase, index, candidate, domain)
                )
                and (
                    not source_risk or domain == "gpr"
                    or candidate in seeded[domain]
                )
            )
            if value in eligible and len(eligible) > 1:
                position = eligible.index(value)
                for offset in (-1, 1):
                    changes.append({role: eligible[(position + offset) % len(eligible)]})
    for role, value in getattr(item, "operand_fields", ()):
        if role == "c_mop_t":
            changes.append({role: (int(value) + 1) % 8})
            continue
        if role == "c_rlist":
            if int(value) < 4:
                changes.append({role: 4})
            continue
        if role in {
            "mop_r_t_30", "mop_r_t_27_26", "mop_r_t_21_20",
            "mop_rr_t_30", "mop_rr_t_27_26",
        }:
            limit = 2 if role.endswith("_30") else 4
            changes.append({role: (int(value) + 1) % limit})
            continue
        if role == "rm":
            changes.append({
                role: next(
                    candidate for candidate in (0, 1, 2, 3, 4, 7)
                    if candidate != int(value)
                ),
            })
            continue
        if role not in {"vd", "vs1", "vs2", "vs3", "fd", "fs1", "fs2", "fs3"}:
            continue
        limit = 32
        domain = "vpr" if role.startswith("v") else "fpr"
        source = role in {"vs1", "vs2", "vs3", "fs1", "fs2", "fs3"}
        candidates = (
            (int(value) + offset) % limit for offset in range(1, limit)
        )
        next_value = next((
            candidate for candidate in candidates
            if not source or index < 0 or candidate in seeded[domain]
        ), None)
        if next_value is None:
            continue
        changes.append({role: next_value})
    if isinstance(item.immediate, int):
        changes.extend({"imm": item.immediate + delta} for delta in _IMMEDIATE_DELTAS)
    from framework.rvemi.program import _rule_facts
    xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
    facts = _rule_facts(item.mnemonic, xlen)
    effect = facts.get("effect")
    if effect in {"gpr", "gpr-from-fpr"}:
        for role in ("rs1", "rs2", "rs3"):
            value = getattr(item, role, None)
            if effect == "gpr" and isinstance(value, int) and value != 0:
                changes.append({role: 0})
        if isinstance(getattr(item, "rd", None), int) and item.rd != 0:
            changes.append({"rd": 0})
        if effect == "gpr" and isinstance(item.immediate, int) and item.immediate != 0:
            changes.append({"imm": 0})
    for target in facts.get("equivalent_targets", ()):
        if str(target).replace("_", ".") != str(item.mnemonic).replace("_", "."):
            changes.append({"mnemonic": str(target)})
    for sibling in facts.get("literal_identity_siblings", ()):
        if isinstance(sibling, Mapping):
            target = sibling.get("mnemonic")
            allowed = sibling.get("allowed_literals", ())
            if (
                target
                and isinstance(allowed, (list, tuple))
                and item.immediate in allowed
                and str(target).replace("_", ".") != str(item.mnemonic).replace("_", ".")
            ):
                changes.append({"mnemonic": str(target)})
    for sibling in facts.get("effect_view_siblings", ()):
        if isinstance(sibling, Mapping):
            target = sibling.get("mnemonic")
            if target and str(target).replace("_", ".") != str(item.mnemonic).replace("_", "."):
                changes.append({"mnemonic": str(target)})
    for target in facts.get("zero_immediate_targets", ()):
        if isinstance(item.immediate, int) and item.immediate == 0:
            changes.append({"mnemonic": str(target)})
    if facts.get("effect") == "fpr" and not facts.get("has_rm"):
        changes.extend({"mnemonic": sibling} for sibling in facts.get("fp_form_family_siblings", ()))
    changes.extend(
        {"mnemonic": sibling}
        for sibling in _SINGLE_FORM_SIBLINGS.get(
            str(item.mnemonic).replace("_", "."), ()
        )
    )
    changes.extend(_r3_changes(testcase, item))
    changes.extend(_field_changes(testcase, item))
    changes.extend(_raw_fixed_field_changes(testcase, item))
    return tuple(changes)


def _r3_changes(testcase: TestCase, item: object) -> tuple[dict[str, int], ...]:
    from framework.rvemi.program import _rule_facts

    xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
    facts = _rule_facts(item.mnemonic, xlen)
    if facts.get("effect") != "gpr":
        return ()
    result = []
    for left, right in facts.get("permutation_pairs", ()):
        left_value, right_value = getattr(item, left, None), getattr(item, right, None)
        if (
            isinstance(left_value, int) and isinstance(right_value, int)
            and getattr(item, f"{left}_domain", None) == "gpr"
            and getattr(item, f"{right}_domain", None) == "gpr"
            and left_value != right_value
        ):
            result.append({left: right_value, right: left_value})
    return tuple(result)


def _single_changes(testcase: TestCase) -> tuple[tuple[int, dict[str, object]], ...]:
    risks = _risks(testcase)
    risk_indices = {index for index, _ in risks}
    items = (*risks, *tuple(
        (index, item) for index, item in enumerate(testcase.instruction_meta)
        if index not in risk_indices and _single_item_rewriteable(item)
    ))
    return tuple(
        (index, change)
        for index, item in items
        for change in _changes_for_item(testcase, item)
    )


_SINGLE_PROTECTED_TAGS = {
    "observable", "control-marker", "control-fallthrough-guard",
    "fflags-clear", "vector-vxsat-clear",
}


def _single_item_rewriteable(item: object) -> bool:
    tags = set(getattr(item, "tags", ()))
    return "risk" in tags or (
        not tags.intersection(_SINGLE_PROTECTED_TAGS)
        and not any(tag.endswith("-setup") for tag in tags)
    )


def _candidate(
    testcase: TestCase, index: int, changes: Mapping[str, object],
) -> dict[str, object] | None:
    changed = _mutate_at(testcase, index, changes)
    if changed is None or changed.code_bytes == testcase.code_bytes:
        return None
    if _candidate_observer_gap(changed) is not None:
        return None
    item = testcase.instruction_meta[index]
    line = 5 + index
    changed_item = changed.instruction_meta[index]
    raw = testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length]
    source_text = (
        f".2byte 0x{int.from_bytes(raw, 'little'):04x}"
        if item.byte_length == 2 else
        f".word 0x{int.from_bytes(raw, 'little'):08x}"
        if item.byte_length == 4 else
        ".byte " + ", ".join(f"0x{value:02x}" for value in raw)
    )
    try:
        payload = {
            "mnemonic": _assembly_mnemonic(changed_item),
            "operands": _operands(changed_item, 32 if testcase.isa_profile.lower().startswith("rv32") else 64),
        }
    except ValueError:
        return None
    if "raw_word" in changes:
        payload["raw_word_hex"] = f"0x{int(changes['raw_word']):0{item.byte_length * 2}x}"
    else:
        from framework.rvemi.program import _profile_operands_valid, _text_encodable
        from framework.rvemi.runtime_spec import runtime_instruction_spec
        xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
        spec = runtime_instruction_spec(payload["mnemonic"], xlen)
        if spec is not None and not changed_item.mnemonic.startswith("cm.pop") \
                and not _text_encodable(
                    spec, spec.operand_roles, payload["operands"], isa_profile=testcase.isa_profile,
                ):
            return None
        if not _profile_operands_valid(
            payload["mnemonic"], payload["operands"],
            xlen=xlen, isa_profile=testcase.isa_profile,
        ):
            return None
    base = {
        "source_span": {"start": line, "end": line, "start_col": 2, "end_col": 2 + len(source_text)},
        "operator": "replace",
        "payload": payload,
    }
    return {
        **base,
        "source_text": source_text,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "candidate_id": canonical_digest(base),
    }


def _candidate_observer_gap(testcase: TestCase) -> str | None:
    """Reject normal candidates that would overwrite the Direct observer."""
    if expected_trap_from_dataflow(testcase.dataflow_meta):
        return None
    from framework.direct_elf import _observer_register_contract
    from framework.rvgen.admission import actual_violation_set

    # A statically illegal candidate takes the trap frame and does not execute
    # the ordinary observer, so its operands may use observer scratch GPRs.
    violations = actual_violation_set(testcase)
    if any(str(value) != "unknown-form" for value in violations):
        return None
    try:
        _observer_register_contract(testcase)
    except ValueError as error:
        reason = str(error)
        if reason.startswith("observer-gap:observer-register-interference:"):
            return reason
    return None


def _fragment_candidates(
    testcase: TestCase,
) -> tuple[tuple[dict[str, object], tuple[int, ...], tuple[dict[str, object], ...]], ...]:
    cache_key = canonical_digest(testcase.to_dict())
    if cache_key in _FRAGMENT_CACHE:
        return _FRAGMENT_CACHE[cache_key]
    risk_indices = {index for index, _item in _risks(testcase)}
    if len(testcase.instruction_meta) < 2:
        return ()
    source_lines = single_program_source(testcase).splitlines(keepends=True)
    changes = {}
    for index, item in enumerate(testcase.instruction_meta):
        if not _single_item_rewriteable(item):
            continue
        changes[index] = tuple(
            change for change in _changes_for_item(testcase, item)
            if _candidate(testcase, index, change) is not None
        )
    result = []
    for size in (2, 3):
        for start in range(len(testcase.instruction_meta) - size + 1):
            end = start + size - 1
            if not risk_indices.intersection(range(start, end + 1)):
                continue
            items = testcase.instruction_meta[start:end + 1]
            if any(
                not changes.get(index)
                for index in range(start, end + 1)
            ) or any(
                left.byte_offset + left.byte_length != right.byte_offset
                for left, right in zip(items, items[1:])
            ):
                continue
            pivot = next(index for index in range(start, end + 1) if index in risk_indices)
            # ponytail：片段只变化一个风险枢轴；独立编辑覆盖完整组合，避免组合爆炸。
            for pivot_change in changes[pivot]:
                operations = tuple(
                    (index, pivot_change if index == pivot else changes[index][0])
                    for index in range(start, end + 1)
                )
                changed = testcase
                for index, change in reversed(operations):
                    changed = _mutate_at(changed, index, change)
                    if changed is None:
                        break
                if changed is None or changed.code_bytes == testcase.code_bytes:
                    continue
                start_line, end_line = 5 + start, 5 + end
                source_block = "".join(source_lines[start_line - 1:end_line])
                source_text = source_block.rstrip("\r\n")
                changed_lines = single_program_source(changed).splitlines(keepends=True)
                base = {
                    "source_span": {
                        "start": start_line, "end": end_line, "start_col": 2,
                        "end_col": len(source_lines[end_line - 1].rstrip("\r\n")),
                    },
                    "operator": "replace_fragment",
                    "payload": {
                        "lines": [
                            line.rstrip("\r\n")
                            for line in changed_lines[start_line - 1:end_line]
                        ],
                    },
                }
                result.append((
                    {
                        **base,
                        "source_text": source_text,
                        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                        "candidate_id": canonical_digest(base),
                    },
                    tuple(index for index, _change in operations),
                    tuple(change for _index, change in operations),
                ))
    result = tuple(result)
    # ponytail：候选池只做进程内复用，容量到顶清空；避免长批次的无界缓存。
    if len(_FRAGMENT_CACHE) >= 256:
        _FRAGMENT_CACHE.clear()
    _FRAGMENT_CACHE[cache_key] = result
    return result


def enumerate_single_rewrites(program: object) -> tuple[dict[str, object], ...]:
    testcase = _case(program)
    if not _risks(testcase):
        return ()
    result = []
    seen = set()
    for index, change in _single_changes(testcase):
        entry = _candidate(testcase, index, change)
        if entry is None or (digest := entry["candidate_id"]) in seen:
            continue
        seen.add(digest)
        result.append(entry)
    for entry, _indices, _changes in _fragment_candidates(testcase):
        if entry["candidate_id"] not in seen:
            seen.add(entry["candidate_id"])
            result.append(entry)
    return tuple(result)


def _candidate_matches(entry: Mapping[str, object], wanted: Mapping[str, object]) -> bool:
    return (
        all(entry.get(name) == wanted.get(name) for name in ("source_span", "operator", "payload"))
        and wanted.get("candidate_id") in (None, entry.get("candidate_id"))
        and all(wanted.get(name) in (None, entry.get(name)) for name in ("source_text", "source_sha256"))
    )


def _select_candidate(program: object, wanted: Mapping[str, object]):
    testcase = _case(program)
    if wanted.get("operator") == "replace_fragment":
        return next(
            (
                (entry, indices, changes)
                for entry, indices, changes in _fragment_candidates(testcase)
                if _candidate_matches(entry, wanted)
            ),
            None,
        )
    span = wanted.get("source_span")
    start = span.get("start") if isinstance(span, Mapping) else None
    if type(start) is int:
        index = start - 5
        changes = (
            ((index, change) for change in _changes_for_item(testcase, testcase.instruction_meta[index]))
            if 0 <= index < len(testcase.instruction_meta) else ()
        )
    else:
        changes = _single_changes(testcase)
    for index, change in changes:
        entry = _candidate(testcase, index, change)
        if entry is None:
            continue
        if _candidate_matches(entry, wanted):
            return entry, index, change
    return None


def _variant(program: object, testcase: TestCase, action: Mapping[str, object] | None = None):
    from framework.rvemi.program import ProgramVariant, program_sha256

    source = single_program_source(testcase)
    params = dict(getattr(program, "run_params", {}))
    params.update({
        "route": "single", "isa": testcase.isa_profile,
        "isa_profile": testcase.isa_profile, "input_id": testcase.input_id,
        "rvgen_testcase_id": testcase.testcase_id,
        "rvgen_base_program_id": testcase.base_program_id,
        "single_case": testcase.to_dict(),
        "single_case_sha256": hashlib.sha256(testcase.code_bytes).hexdigest(),
        "observer_fields": list(dict.fromkeys((
            "executed_pcs", *_compare_mask_fields(testcase.compare_mask),
        ))),
    })
    params.setdefault("harness", "rvgen-single")
    parent = program_sha256(program) if program is not None else hashlib.sha256(source.encode()).hexdigest()
    return ProgramVariant(
        source,
        parent,
        "{}" if action is None else json.dumps(dict(action), sort_keys=True, separators=(",", ":")),
        params,
    )


def program_from_testcase(testcase: TestCase):
    return _variant(None, _require_rvgen_case(testcase))


def apply_single_hint(program: object, hint: Mapping[str, object]):
    edits = hint.get("edits") if isinstance(hint, Mapping) else None
    if not isinstance(edits, list) or not edits:
        raise ValueError("single route requires rewrite candidates")
    testcase = _case(program)
    selected = []
    operations = []
    seen = set()
    for wanted in edits:
        if not isinstance(wanted, Mapping):
            raise ValueError("single rewrite candidate is invalid")
        match = _select_candidate(program, wanted)
        if match is None:
            raise ValueError("seed-miss:single candidate not in pool")
        candidate, indices, changes = match
        if isinstance(indices, int):
            indices, changes = (indices,), (changes,)
        if any(index in seen for index in indices):
            raise ValueError("generation-gap:single edits overlap")
        seen.update(indices)
        selected.append(candidate)
        operations.extend(zip(indices, changes))
    changed = testcase
    for index, changes in sorted(operations, key=lambda item: item[0], reverse=True):
        changed = _mutate_at(changed, index, changes)
        if changed is None:
            raise ValueError("seed-miss:single candidate cannot be encoded")
    changed = _classify_mutated_case(changed)
    if changed is None:
        raise ValueError("seed-miss:single candidate has invalid control flow")
    if (reason := _candidate_observer_gap(changed)) is not None:
        raise ValueError("candidate-observer-register-gap:" + reason)
    changed = identify_testcase(changed)
    return _variant(program, changed, {"edits": selected})


def single_rewrite_realizes(before: object, after: object, hint: Mapping[str, object]) -> bool:
    if not isinstance(hint, Mapping) or not hint.get("edits"):
        return False
    try:
        expected = _case(apply_single_hint(before, hint))
        actual = _case(after)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        actual.code_bytes == expected.code_bytes
        and actual.code_bytes != _case(before).code_bytes
        and after.run_params.get("single_case_sha256") == hashlib.sha256(actual.code_bytes).hexdigest()
    )


def build_single_profile(program: object, observation: object, *, input_identity=None):
    from framework.rvemi.program import (
        ProfileResult, ProgramProfile, _pc_value, _stable_input_identity, program_sha256,
    )

    observations = tuple(observation) if isinstance(observation, (list, tuple)) else (observation,)
    observations = observations[:1]
    if not observations:
        # 扩展 reference 对常用 I/M/A/B/C/F/D/Zicsr/RVV 路径直接建模；仍未
        # 严格建模的形式返回具名缺口而不是崩溃。MCMC 会在每个候选上独立
        # 跑 reference，目标执行与覆盖率不受影响。
        return ProfileResult(None, "profile-gap:single-reference-unavailable")
    def value(item: object, name: str):
        return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)

    def expected_output_matches(item: object, expected: Mapping[str, object]) -> bool:
        output_ranges = []
        for instruction in testcase.instruction_meta:
            if "observable" not in instruction.tags or instruction_effect_class(instruction.mnemonic) not in {"store", "atomic"}:
                continue
            fields = dict(instruction.operand_fields)
            base = fields.get("rs1")
            offset = next((fields.get(name) for name in ("imm12s", "imm12", "offset") if fields.get(name) is not None), None)
            if type(base) is not int or type(offset) is not int or not 0 <= base < len(testcase.initial_gpr):
                continue
            mnemonic = str(instruction.mnemonic).lower().replace("_", ".")
            width = next(
                (size for suffix, size in (("q", 16), ("d", 8), ("w", 4), ("h", 2), ("b", 1))
                 if mnemonic.endswith(suffix)), 4,
            )
            address = testcase.initial_gpr[base] + offset
            region = next(
                (region for region in testcase.initial_memory_regions
                 if region.address <= address < region.address + len(region.data)),
                None,
            )
            if region is not None:
                output_ranges.append((region.region_id, address - region.address, width))
        if not output_ranges:
            return False
        output_ranges = output_ranges[-1:]
        blobs = []
        for name in ("memory_snapshot", "memory_delta"):
            raw = value(item, name)
            entries = raw.items() if isinstance(raw, Mapping) else ((None, raw),)
            for region_id, blob in entries:
                if isinstance(blob, (bytes, bytearray)):
                    data = bytes(blob)
                elif isinstance(blob, str):
                    try:
                        data = bytes.fromhex(blob.removeprefix("0x"))
                    except ValueError:
                        return False
                else:
                    continue
                for expected_region, offset, width in output_ranges:
                    if region_id in (None, expected_region):
                        blobs.append(data[offset:offset + width])
        if not blobs:
            return False
        for name, raw in expected.items():
            if name == "expected_upper_word":
                try:
                    expected_upper = int(str(raw).strip(), 0)
                except (TypeError, ValueError):
                    return False
                if not 0 <= expected_upper <= 0xFFFFFFFF or not any(
                    len(blob) >= 8 and int.from_bytes(blob[:8], "little") >> 32 == expected_upper
                    for blob in blobs
                ):
                    return False
                continue
            text = str(raw).strip().lower().removeprefix("0x")
            if not text or len(text) % 2 or any(char not in "0123456789abcdef" for char in text):
                return False
            needle = int(text, 16).to_bytes(len(text) // 2, "little")
            if len(needle) not in {4, 8, 16} or not any(
                len(blob) >= len(needle) and blob[:len(needle)] == needle
                for blob in blobs
            ):
                return False
        return True

    if input_identity is not None and not isinstance(input_identity, Mapping):
        return ProfileResult(None, "profile-gap:single-input-identity-invalid")
    try:
        testcase = _case(program)
    except (KeyError, TypeError, ValueError):
        return ProfileResult(None, "profile-gap:single-case-invalid")
    params = getattr(program, "run_params", {})
    raw_identity = {**dict(params), **dict(input_identity or {})}
    if any(
        raw_identity.get(name) is not None
        and str(raw_identity[name]).lower() != str(expected).lower()
        for name, expected in (
            ("isa", testcase.isa_profile), ("isa_profile", testcase.isa_profile),
            ("input_id", testcase.input_id),
        )
    ):
        return ProfileResult(None, "profile-gap:single-input-identity-mismatch")
    try:
        identity = _stable_input_identity(raw_identity)
    except (TypeError, ValueError):
        return ProfileResult(None, "profile-gap:single-input-identity-invalid")
    xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
    expected_output = testcase.dataflow_meta.get("recipe_expected", {})
    expected_output = expected_output if isinstance(expected_output, Mapping) else {}

    def window_checkpoints(item: object) -> tuple[dict[str, object], ...]:
        evidence = value(item, "translation_evidence")
        details = evidence.get("details") if isinstance(evidence, Mapping) else getattr(
            evidence, "details", None
        )
        rows = details.get("native_window_checkpoints") if isinstance(details, Mapping) else None
        digest = details.get("native_window_checkpoint_digest") if isinstance(details, Mapping) else None
        if (
            not isinstance(rows, (list, tuple)) or not rows
            or digest != canonical_digest({"native_window_checkpoints": rows})
        ):
            return ()
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                return ()
            pc, after_pc = _pc_value(row.get("pc")), _pc_value(row.get("after_pc"))
            if (
                pc is None or after_pc is None
                or pc & 1 or after_pc & 1
                or pc >= 1 << xlen or after_pc >= 1 << xlen
            ):
                return ()
            normalized = {"pc": pc, "after_pc": after_pc}
            for name in (
                "before_gpr", "after_gpr", "before_fpr_rawbits", "after_fpr_rawbits",
            ):
                raw = row.get(name)
                if raw is None:
                    continue
                if not isinstance(raw, (list, tuple)) or len(raw) != 32:
                    return ()
                values = tuple(_pc_value(value) for value in raw)
                limit = 1 << (xlen if name.endswith("gpr") else 64)
                if any(value is None or value < 0 or value >= limit for value in values) \
                        or name.endswith("gpr") and values[0] != 0:
                    return ()
                normalized[name] = list(values)
            for name in ("before_fflags", "after_fflags", "before_frm", "after_frm"):
                if name in row:
                    parsed = _pc_value(row[name])
                    limit = 0x1F if name.endswith("fflags") else 0x7
                    if parsed is None or parsed > limit:
                        return ()
                    normalized[name] = parsed
            result.append(normalized)
        if any(left["after_pc"] != right["pc"] for left, right in zip(result, result[1:])):
            return ()
        if any(
            left.get("after_gpr") is not None and right.get("before_gpr") is not None
            and left["after_gpr"] != right["before_gpr"]
            for left, right in zip(result, result[1:])
        ):
            return ()
        for before, after in (
            ("before_fpr_rawbits", "after_fpr_rawbits"),
            ("before_fflags", "after_fflags"),
            ("before_frm", "after_frm"),
        ):
            if any(
                left.get(after) is not None and right.get(before) is not None
                and left[after] != right[before]
                for left, right in zip(result, result[1:])
            ):
                return ()
        return tuple(result)

    windows = tuple(window_checkpoints(item) for item in observations)
    native_window_checkpoints = (
        windows[0] if windows and windows[0]
        else ()
    )
    reference_identity = None
    normalized = []
    for item in observations:
        outcome = value(item, "outcome")
        if not isinstance(outcome, str):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        extra = value(item, "extra_state")
        trap = (
            outcome in {"trap", "nonzero-exit"}
            and isinstance(extra, Mapping)
            and (
                guest_trap_observed(dict(extra))
                or type(extra.get("trap.cause")) is int
                and 0 <= extra.get("trap.cause") < 64
                and type(extra.get("trap.epc")) is int
                and 0 <= extra.get("trap.epc") < 1 << xlen
                and not extra.get("trap.epc") & 1
            )
        )
        raw_pcs = value(item, "executed_pcs")
        if (
            not isinstance(raw_pcs, (list, tuple)) or not raw_pcs
            or outcome not in {"normal", "completed"} and not trap
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        signal = value(item, "signal")
        signal_code = value(item, "signal_code")
        if (
            signal is not None and (not isinstance(signal, str) or not signal.strip())
            or signal_code is not None and (type(signal_code) is not int or signal_code <= 0)
            or outcome in {"normal", "completed"} and (signal is not None or signal_code is not None)
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        raw_reference = value(item, "reference_identity")
        if raw_reference is not None and not isinstance(raw_reference, Mapping):
            return ProfileResult(None, "profile-gap:single-reference-identity-invalid")
        reference = dict(raw_reference) if isinstance(raw_reference, Mapping) else {
            name: value(item, name)
            for name in (
                "backend", "profile_id", "input_id", "tool_version",
                "binary_sha256", "guest_elf_sha256",
            )
            if value(item, name) not in (None, "")
        }
        identity_fields = (
            "backend", "profile_id", "input_id", "tool_version",
            "binary_sha256", "guest_elf_sha256",
        )
        if (
            not isinstance(reference.get("backend"), str)
            or not reference["backend"].strip()
            or any(value in (None, "") for value in reference.values())
            or any(
                name in reference
                and (not isinstance(reference[name], str) or not reference[name].strip())
                for name in identity_fields
            )
            or any(
                not is_sha256_digest(reference[name])
                for name in ("binary_sha256", "guest_elf_sha256")
                if name in reference
            )
            or any(
                value(item, name) not in (None, reference.get(name))
                for name in (
                    "backend", "profile_id", "input_id", "tool_version",
                    "binary_sha256", "guest_elf_sha256",
                )
                if value(item, name) is not None and name in reference
            )
        ):
            return ProfileResult(None, "profile-gap:single-reference-identity-invalid")
        try:
            canonical_digest(reference)
        except (TypeError, ValueError):
            return ProfileResult(None, "profile-gap:single-reference-identity-invalid")
        reference_identity = reference
        try:
            pcs = tuple(_pc_value(pc) for pc in raw_pcs)
            checkpoint = value(item, "checkpoint_pc")
            checkpoint = None if checkpoint is None else _pc_value(checkpoint)
        except (TypeError, ValueError):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        if (
            any(pc is None or pc & 1 or pc >= 1 << xlen for pc in pcs)
            or checkpoint is not None
            and (checkpoint & 1 or checkpoint >= 1 << xlen)
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        for name in ("fault_pc", "fault_address"):
            raw_fault = value(item, name)
            if raw_fault is not None:
                fault = _pc_value(raw_fault)
                if fault is None or fault >= 1 << xlen or name == "fault_pc" and fault & 1:
                    return ProfileResult(None, "profile-gap:single-reference-invalid")
        observed_input = value(item, "input_id")
        if observed_input not in (None, testcase.input_id) \
                or reference.get("input_id") not in (None, testcase.input_id):
            return ProfileResult(None, "profile-gap:single-input-identity-mismatch")
        exit_code = value(item, "exit_code")
        if (
            exit_code is not None and type(exit_code) is not int
            or outcome in {"normal", "completed"} and exit_code not in (None, 0)
            or outcome == "nonzero-exit" and (type(exit_code) is not int or exit_code == 0)
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        instruction_count = value(item, "instruction_count")
        if instruction_count is not None and (
            type(instruction_count) is not int or instruction_count < 0
            or instruction_count != len(pcs)
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        trace_complete = value(item, "trace_complete")
        if trace_complete is not None and trace_complete is not True:
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        if expected_output and not expected_output_matches(item, expected_output):
            return ProfileResult(None, "profile-gap:expected-result-mismatch")
        gpr = value(item, "gpr")
        if gpr is not None and (
            not isinstance(gpr, (list, tuple)) or len(gpr) != 32
            or any(type(register) is not int or not 0 <= register < 1 << xlen for register in gpr)
            or gpr[0] != 0
        ):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        memory_digest = value(item, "memory_digest")
        if memory_digest is not None and not is_sha256_digest(memory_digest):
            return ProfileResult(None, "profile-gap:single-reference-invalid")
        for name in ("memory_snapshot", "memory_delta"):
            raw_memory = value(item, name)
            entries = raw_memory.items() if isinstance(raw_memory, Mapping) else ((None, raw_memory),)
            for _region, blob in entries:
                if isinstance(blob, (bytes, bytearray)):
                    memory_bytes = bytes(blob)
                elif isinstance(blob, str):
                    try:
                        memory_bytes = bytes.fromhex(blob.removeprefix("0x"))
                    except ValueError:
                        return ProfileResult(None, "profile-gap:single-reference-invalid")
                elif isinstance(blob, Mapping) and "test-memory" in blob:
                    try:
                        memory_bytes = bytes.fromhex(str(blob["test-memory"]).removeprefix("0x"))
                    except ValueError:
                        return ProfileResult(None, "profile-gap:single-reference-invalid")
                else:
                    continue
                if memory_digest is not None and hashlib.sha256(memory_bytes).hexdigest() != memory_digest:
                    return ProfileResult(None, "profile-gap:single-reference-invalid")
        normalized_gpr = tuple(gpr) if gpr is not None else None
        normalized.append((outcome, pcs, checkpoint, observed_input, normalized_gpr))
    if not observations:
        return ProfileResult(None, "profile-gap:single-reference-invalid")
    if identity.get("route") != "single":
        return ProfileResult(None, "profile-gap:single-input-identity-mismatch")
    source_sha = program_sha256(program)
    entry = _pc_value(identity.get("entry"))
    base_pc = testcase.code_address if entry is None else entry - testcase.code_address
    risk_offsets = tuple(
        item.pc_offset for item in testcase.instruction_meta if "risk" in item.tags
    )

    def _observed_risk_pcs(item: object, pcs: tuple[int, ...]) -> tuple[int, ...]:
        extra = value(item, "extra_state")
        marker = extra.get("rvgen_risk_execution") if isinstance(extra, Mapping) else None
        raw = marker.get("risk_pcs") if isinstance(marker, Mapping) \
            and marker.get("status") == "observed" else None
        if raw is None:
            reconciliation = extra.get("sail_risk_contract_reconciliation") \
                if isinstance(extra, Mapping) else None
            records = reconciliation.get("records") if isinstance(reconciliation, Mapping) \
                and reconciliation.get("status") == "passed" else ()
            raw = [record.get("expected_pc") for record in records
                   if isinstance(record, Mapping) and record.get("observed") is True]
        try:
            observed = tuple(_pc_value(pc) for pc in raw or ())
        except (TypeError, ValueError):
            return ()
        return observed if all(
            pc is not None and not pc & 1 and pc in pcs for pc in observed
        ) else ()

    def observed_risk_base(item: object, pcs: tuple[int, ...]) -> int | None:
        observed = _observed_risk_pcs(item, pcs)
        if len(observed) != len(risk_offsets) or not observed:
            return None
        candidate = observed[0] - risk_offsets[0]
        return candidate if all(
            pc == candidate + offset for pc, offset in zip(observed, risk_offsets)
        ) else None

    source_bases = {base_pc}
    execution_truncated = False
    for source_observation, (outcome, pcs, checkpoint, _input, _gpr) in zip(
        observations, normalized,
    ):
        observed_base = observed_risk_base(source_observation, pcs)
        effective_base = base_pc if observed_base is None else observed_base
        source_bases.add(effective_base)
        source_pcs = {
            candidate_base + instruction.pc_offset
            for instruction in testcase.instruction_meta
            for candidate_base in source_bases
        }
        observed_risk = _observed_risk_pcs(source_observation, pcs)
        risk_index = next(
            (index for index, pc in enumerate(pcs) if pc in observed_risk),
            None,
        )
        if risk_index is not None and risk_offsets and any(
            instruction.pc_offset >= min(risk_offsets)
            for pc in pcs[:risk_index]
            for instruction in testcase.instruction_meta
            if effective_base + instruction.pc_offset == pc
        ):
            return ProfileResult(None, "profile-gap:single-execution-order-invalid")
        checked_pcs = pcs[risk_index:] if risk_index is not None else pcs
        if any(pc not in source_pcs for pc in checked_pcs):
            return ProfileResult(None, "profile-gap:single-executed-pc-unmapped")
        expected_ids = testcase.dataflow_meta.get("expected_executed_instruction_ids", ())
        first_risk = next(
            (index for index, instruction in enumerate(testcase.instruction_meta)
             if "risk" in instruction.tags), None,
        )
        if isinstance(expected_ids, (list, tuple)) and first_risk is not None \
                and (checkpoint is not None or instruction_count is not None or trace_complete is True):
            risk_id = testcase.instruction_meta[first_risk].instruction_id
            start = next((index for index, item in enumerate(expected_ids) if item == risk_id), None)
            instruction_offsets = {
                item.instruction_id: item.pc_offset for item in testcase.instruction_meta
            }
            required = tuple(
                instruction_offsets[item]
                for item in expected_ids[start:]
                if start is not None and item in instruction_offsets
            )
            expected_pcs = tuple(effective_base + offset for offset in required)
            cursor = 0
            for expected_pc in expected_pcs:
                cursor = next((index + 1 for index in range(cursor, len(checked_pcs))
                               if checked_pcs[index] == expected_pc), -1)
                if cursor < 0:
                    if instruction_count is not None or trace_complete is True:
                        return ProfileResult(None, "profile-gap:single-execution-truncated")
                    execution_truncated = True
                    break
        if checkpoint is None:
            continue
        last_pc = pcs[-1]
        next_pc = next(
            (
                last_pc + instruction.byte_length
                for instruction in testcase.instruction_meta
                if effective_base + instruction.pc_offset == last_pc
            ),
            None,
        )
        expected_exit = effective_base + testcase.exit_checkpoint_address
        if checkpoint not in {last_pc, next_pc, expected_exit}:
            return ProfileResult(None, "profile-gap:single-checkpoint-mismatch")
    source_pcs = {
        candidate_base + item.pc_offset
        for item in testcase.instruction_meta
        for candidate_base in source_bases
    }
    if native_window_checkpoints and any(
        row["pc"] not in source_pcs for row in native_window_checkpoints
    ):
        native_window_checkpoints = ()
    def risk_witness(item: object, pcs: tuple[int, ...]) -> tuple[int, ...]:
        observed = _observed_risk_pcs(item, pcs)
        extra = value(item, "extra_state")
        direct = isinstance(extra, Mapping) and isinstance(
            extra.get("rvgen_risk_execution"), Mapping
        ) and extra["rvgen_risk_execution"].get("status") == "observed"
        if (len(observed) != len(risk_offsets) or not observed
                or not direct and observed != tuple(base_pc + offset for offset in risk_offsets)
                or any(pc is None or pc & 1 or pc not in pcs for pc in observed)):
            return ()
        witness_base = observed[0] - risk_offsets[0]
        if (witness_base < 0 or witness_base & 1
                or witness_base >= 1 << xlen or witness_base not in pcs):
            return ()
        return observed if all(
            pc == witness_base + offset for pc, offset in zip(observed, risk_offsets)
        ) else ()
    risk_witnesses = tuple(
        risk_witness(item, pcs)
        for item, (_, pcs, *_rest) in zip(observations, normalized)
    )
    observed_risk_pcs = risk_witnesses[0] if risk_witnesses \
        and risk_witnesses[0] \
        and all(item == risk_witnesses[0] for item in risk_witnesses) else ()
    complete_gpr = all(
        isinstance(gpr := value(item, "gpr"), (list, tuple))
        and len(gpr) == 32
        and all(type(register) is int for register in gpr)
        and gpr[0] == 0
        for item in observations
    )
    facts = testcase.dataflow_meta.get("realized_facts", {})
    state_domain = facts.get("state_domain") if isinstance(facts, Mapping) else "gpr"
    schema = testcase.dataflow_meta.get("generation_owner", {}).get("schema", {})
    if state_domain == "scalar" and isinstance(schema, Mapping) and schema.get("writeback") == "fpr":
        state_domain = "fpr"
    def semantic_witness(item: object) -> bool:
        extra = value(item, "extra_state")
        if not isinstance(extra, Mapping):
            return state_domain not in {"fpr", "vector", "csr"}
        observed = set(observed_state_fields(dict(extra)))
        if state_domain == "fpr":
            return bool(observed & {"fpr_rawbits", "fflags", "frm"})
        if state_domain == "vector":
            vector_fields = {
                name for name in observed if name.startswith("vector.")
            } | (observed & {"vxsat", "vxrm"})
            return len(vector_fields) >= 2
        if state_domain == "vector-memory":
            vector_fields = {
                name for name in observed if name.startswith("vector.")
            } | (observed & {"vxsat", "vxrm"})
            memory = extra.get("memory_reads") or extra.get("memory_writes")
            return len(vector_fields) >= 2 and bool(memory or value(item, "memory_snapshot"))
        if state_domain == "control-flow":
            return any(
                extra.get(name) not in (None, "")
                for name in ("control.target", "branch_outcome", "after_pc")
            )
        if state_domain == "csr":
            if isinstance(schema, Mapping) and schema.get("observer") == "gpr":
                expected = set(testcase.dataflow_meta.get(
                    "expected_executed_instruction_ids", ()
                ))
                return {"csr-state-read", "csr-state-observer"} <= expected
            return any(name.startswith("csr.") for name in observed) or (
                extra.get("csr_observer_status") == "observed"
                and bool(extra.get("csr_values") or extra.get("csr_trace_events"))
            )
        return True
    dynamic_witness = complete_gpr and all(
        isinstance(extra := value(item, "extra_state"), Mapping)
        and (
            value(item, "outcome") in {"normal", "completed"}
            or value(item, "outcome") in {"trap", "nonzero-exit"}
            and (
                guest_trap_observed(dict(extra))
                or type(extra.get("trap.cause")) is int
                and 0 <= extra["trap.cause"] < 64
                and type(extra.get("trap.epc")) is int
                and 0 <= extra["trap.epc"] < 1 << xlen
                and not extra["trap.epc"] & 1
            )
        )
        and risk_witnesses[index]
        and semantic_witness(item)
        for index, item in enumerate(observations)
    ) and not execution_truncated
    fields = {"outcome", "executed_pcs"}
    if complete_gpr:
        fields.add("gpr")
    if any(value(item, "checkpoint_pc") is not None for item in observations):
        fields.add("checkpoint_pc")
    if any(value(item, "instruction_count") is not None for item in observations):
        fields.add("instruction_count")
    if any(value(item, "trace_complete") is not None for item in observations):
        fields.add("trace_complete")
    if any(value(item, "memory_snapshot") is not None for item in observations):
        fields.add("memory_snapshot")
    observation_digest = canonical_digest({
        "route": "single", "source_sha256": source_sha,
        "testcase_id": testcase.testcase_id, "dynamic_witness": dynamic_witness,
        "observations": [
            {
                "outcome": row[0], "executed_pcs": list(row[1]),
                "checkpoint_pc": row[2], "input_id": row[3],
            } | ({"gpr": value(item, "gpr")} if complete_gpr else {})
            for row, item in zip(normalized, observations)
        ],
    })
    profile = ProgramProfile(
        source_sha, identity,
        (), (), tuple(sorted(fields)),
        {
            "execution_trace": {
                "complete": True, "count": len(normalized[0][1]),
                **({"truncated": True} if execution_truncated else {}),
                "pc_digest": canonical_digest(list(normalized[0][1])),
            },
            "observation_digest": observation_digest,
            "testcase_id": testcase.testcase_id,
            "reference_identity": reference_identity,
            **({
                "risk_pcs": list(observed_risk_pcs),
                "risk_base_pc": observed_risk_pcs[0] - risk_offsets[0],
            } if observed_risk_pcs else {}),
            **({"native_window_checkpoints": list(native_window_checkpoints)}
               if native_window_checkpoints else {}),
        },
        route="single", dynamic_witness=dynamic_witness,
    )
    profile.provenance["profile_digest"] = profile.profile_digest
    return ProfileResult(profile)


def _single_rule_witness(profile: object, pc: int, fields: tuple[str, ...]) -> bool:
    rows = getattr(profile, "provenance", {}).get("native_window_checkpoints", ())
    return any(
        isinstance(row, Mapping) and row.get("pc") == pc
        and all(field in row for field in fields)
        for row in rows
    )


def _r3_action(
    testcase: TestCase, index: int, change: Mapping[str, object], profile: object,
) -> dict[str, object] | None:
    candidate = _candidate(testcase, index, change)
    if candidate is None:
        return None
    item = testcase.instruction_meta[index]
    xlen = 32 if testcase.isa_profile.lower().startswith("rv32") else 64
    from framework.rvemi.program import _pc_value, _rule_facts

    facts = _rule_facts(item.mnemonic, xlen)
    effect = facts.get("effect")
    target = str(change.get("mnemonic", "")).replace("_", ".")
    if "raw_word" in change:
        rule = "R2"
        required_trace_fields = ["outcome"]
    elif effect == "fpr":
        if facts.get("has_rm") and "rm" in change:
            rule = "R7"
        elif change.get("mnemonic") in facts.get("fp_form_family_siblings", ()):
            rule = "R8"
        elif set(change) & {"rd", "rs1", "rs2", "rs3"}:
            rule = "R2"
        else:
            return None
        required_trace_fields = [
            "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags",
            "after_fflags", "before_frm", "after_frm",
        ]
    elif effect == "private-load":
        siblings = tuple(
            str(row.get("mnemonic", "")).replace("_", ".")
            for row in facts.get("effect_view_siblings", ())
            if isinstance(row, Mapping)
        )
        if target in siblings:
            rule = "R6"
        elif set(change) & {"rs1", "imm"}:
            rule = "R4"
        else:
            return None
        required_trace_fields = ["before_gpr", "memory_reads", "after_gpr"]
    elif facts.get("control_shape") in {"conditional", "direct", "indirect"}:
        rule = "R9"
        required_trace_fields = ["after_pc", "branch_outcome", "link", "ialign"]
    elif effect == "gpr-from-fpr":
        if target in tuple(str(row).replace("_", ".") for row in facts.get("equivalent_targets", ())):
            rule = "R5"
        elif any(
            role in change and getattr(item, f"{role}_domain", None) == "fpr"
            for role in ("rs1", "rs2", "rs3")
        ):
            rule = "R2"
        elif change.get("rd") == 0:
            rule = "R10"
        else:
            return None
        required_trace_fields = [
            "before_fpr_rawbits", "after_fpr_rawbits", "before_gpr", "after_gpr",
        ]
    elif effect == "gpr":
        permutation_changes = _r3_changes(testcase, item)
        if change in permutation_changes:
            rule = "R3"
        elif target in tuple(str(row).replace("_", ".") for row in facts.get("equivalent_targets", ())):
            rule = "R5"
        elif any(
            isinstance(row, Mapping)
            and target == str(row.get("mnemonic", "")).replace("_", ".")
            and item.immediate in row.get("allowed_literals", ())
            for row in facts.get("literal_identity_siblings", ())
        ):
            rule = "R5"
        elif target in tuple(
            str(row.get("mnemonic", "")).replace("_", ".")
            for row in facts.get("effect_view_siblings", ()) if isinstance(row, Mapping)
        ) or target in tuple(str(row) for row in facts.get("zero_immediate_targets", ())):
            rule = "R11"
        elif change.get("rd") == 0:
            rule = "R10"
        elif any(role in change and change[role] == 0 for role in ("rs1", "rs2", "rs3")):
            rule = "R1"
        elif set(change) & {"rs1", "rs2", "rs3", "imm"}:
            rule = "R2"
        else:
            return None
        required_trace_fields = ["before_gpr", "after_gpr"]
    elif "mnemonic" in change:
        rule = "R2"
        required_trace_fields = ["outcome"]
    elif set(change) & {
        "rd", "rs1", "rs2", "rs3", "vd", "vs1", "vs2", "vs3", "vm", "imm",
    }:
        rule = "R2"
        required_trace_fields = ["outcome"]
    else:
        return None
    raw = testcase.code_bytes[item.byte_offset:item.byte_offset + item.byte_length]
    source_line = single_program_source(testcase).splitlines()[5 + index - 1]
    entry = _pc_value(profile.input_identity.get("entry"))
    risk_base_pc = getattr(profile, "provenance", {}).get("risk_base_pc")
    base_pc = risk_base_pc if isinstance(risk_base_pc, int) else (
        testcase.code_address if entry is None else entry - testcase.code_address
    )
    required = tuple(required_trace_fields)
    guided = _single_rule_witness(profile, base_pc + item.pc_offset, required)
    return {
        **candidate,
        "kind": "local-rewrite",
        "anchor": {
            "instruction_id": item.instruction_id,
            "occurrence_id": item.instruction_id,
            "pc": base_pc + item.pc_offset,
            "source_start": 0,
            "source_end": len(source_line),
            "final_bytes": list(raw),
        },
        "occurrence_scope": "single",
        "evidence_level": "guided" if guided else "probe",
        **({} if guided else {"witness_reason": f"single-{rule.lower()}-trace-unavailable"}),
        "source_identity": profile.source_sha256, "rule": rule,
        "rule_evidence": {
            "definedness": facts.get("definedness"), "effect": facts.get("effect"),
            "operand_roles": list(facts.get("operand_roles", ())),
            "source_projections": [list(value) for value in facts.get("source_projections", ())],
            "xlen": xlen, "result_width": facts.get("result_width"),
            "signedness": facts.get("signedness"),
            "required_trace_fields": required_trace_fields,
            "occurrence_scope": "single",
            "permutation_pairs": [list(value) for value in facts.get("permutation_pairs", ())],
        },
        "profile_digest": profile.profile_digest,
        "source_cell": f"scalar:{item.mnemonic}",
        "target_cell": f"scalar:{candidate['payload']['mnemonic']}",
    }


def single_profile_actions(program: object, profile: object) -> tuple[dict[str, object], ...]:
    from framework.rvemi.program import _profile_input_matches, program_sha256

    if not getattr(profile, "complete", False) \
            or not is_single_program(program) \
            or program_sha256(program) != getattr(profile, "source_sha256", None) \
            or not _profile_input_matches(program, profile):
        return ()
    testcase = _case(program)
    return tuple(
        action
        for index, item in _risks(testcase)
        for change in _changes_for_item(testcase, item)
        for action in (_r3_action(testcase, index, change, profile),)
        if action is not None and action["rule"] in {f"R{i}" for i in range(1, 12)}
    )


def _classify_mutated_case(testcase: TestCase) -> TestCase | None:
    if not testcase.dataflow_meta.get("candidate_mutated"):
        return testcase
    dataflow = dict(testcase.dataflow_meta)
    dataflow.pop("candidate_trap_classification_gap", None)
    dataflow.pop("candidate_observer_contract_gap", None)
    root_was_trap = (
        dataflow.get("state_domain") == "trap"
        or isinstance(dataflow.get("realized_facts"), Mapping)
        and dataflow["realized_facts"].get("state_domain") == "trap"
    )
    from framework.rvgen.semantic_machine import (
        SemanticTrap, execute_semantic_case,
    )

    def trap_contract(
        *, source: str, violations: Sequence[str] = (), trap: object | None = None,
    ) -> TestCase:
        dataflow["candidate_expected_trap"] = True
        dataflow["candidate_trap_classification_source"] = source
        if violations:
            dataflow["candidate_trap_violations"] = list(violations)
        else:
            dataflow.pop("candidate_trap_violations", None)
        if trap is not None:
            dataflow["candidate_trap_witness"] = {
                "cause": int(trap.cause),
                "pc_offset": int(trap.pc_offset),
                "tval": int(trap.tval),
            }
        else:
            dataflow.pop("candidate_trap_witness", None)
        dataflow.update({
            "state_domain": "trap",
            "state_observer_status": "trap-observer-pending",
            "state_observer_keys": ["trap.cause", "trap.epc", "trap.tval"],
        })
        risk_ids = tuple(
            item.instruction_id for item in testcase.instruction_meta
            if "risk" in item.tags
        )
        trap_fields = (
            "outcome", "signal", "fault_pc",
            "trap.cause", "trap.epc", "trap.tval",
        )
        return replace(
            testcase,
            dataflow_meta=dataflow,
            compare_mask=CompareMask(
                outcome=True, checkpoint_pc=False, signal=True,
                fault_pc=True, fault_address=False,
                extra_state_keys=("trap.cause", "trap.epc", "trap.tval"),
            ),
            observability_contract=ObservabilityContract(
                schema_version="observability-contract-v1",
                risk_value_or_effect="trap-outcome",
                sink_instruction_ids=risk_ids,
                risk_to_sink_distance=1,
                sink_transform="identity",
                final_observation_fields=trap_fields,
                lossiness="none",
            ),
        )

    # The semantic machine models instruction behavior, not every profile and
    # encoding legality rule. Admission owns those concrete legality facts and
    # must run before semantic execution.
    try:
        from framework.rvgen.admission import actual_violation_set

        violations = tuple(sorted(actual_violation_set(testcase)))
    except (TypeError, ValueError, KeyError):
        violations = ()
    concrete_violations = tuple(
        value for value in violations if str(value) != "unknown-form"
    )
    if concrete_violations:
        return trap_contract(source="static-legality", violations=concrete_violations)

    try:
        execute_semantic_case(testcase)
    except SemanticTrap as trap:
        return trap_contract(source="semantic", trap=trap)
    except Exception as error:
        # 无法证明 candidate 是 trap 或普通可观察执行时，保持 fail-closed，
        # 但把原因交给 frame/admission 层的具名 gap，而不是污染成 trap frame。
        dataflow["candidate_expected_trap"] = None
        dataflow["candidate_trap_classification_gap"] = (
            f"{type(error).__name__}: {str(error)[:500]}"
        )
    else:
        dataflow["candidate_expected_trap"] = False
        dataflow.pop("candidate_trap_witness", None)
        dataflow.pop("candidate_trap_classification_source", None)
        dataflow.pop("candidate_trap_violations", None)
        if root_was_trap:
            owner = dataflow.get("generation_owner")
            owner = owner if isinstance(owner, Mapping) else {}
            schema = owner.get("schema")
            schema = schema if isinstance(schema, Mapping) else {}
            normal_sinks = tuple(
                item.instruction_id for item in testcase.instruction_meta
                if "observable" in item.tags and "risk" not in item.tags
            )
            observer = schema.get("observer")
            if normal_sinks or observer not in {None, "outcome"}:
                dataflow["candidate_observer_contract_gap"] = (
                    "normal-candidate-observer-not-materialized"
                )
                return replace(testcase, dataflow_meta=dataflow)
            state_domain = str(owner.get("state_domain") or "scalar")
            if state_domain == "scalar" and schema.get("writeback") == "fpr":
                state_domain = "fpr"
            dataflow.update({
                "state_domain": state_domain,
                "state_observer_status": None,
                "state_observer_keys": [],
            })
            facts = dict(
                dataflow.get("realized_facts")
                if isinstance(dataflow.get("realized_facts"), Mapping) else {}
            )
            facts.update({
                "lane": "normal-defined",
                "state_domain": state_domain,
                "effect_kind": owner.get("effect_kind") or facts.get(
                    "effect_kind", "encoding-only"
                ),
                "disposition": "encoding-only",
            })
            dataflow["realized_facts"] = facts
            semantic_point = dataflow.get("semantic_point")
            if isinstance(semantic_point, Mapping):
                dataflow["semantic_point"] = {
                    **dict(semantic_point), "lane": "normal-defined",
                }
            risk_ids = tuple(
                item.instruction_id for item in testcase.instruction_meta
                if "risk" in item.tags
            )
            return replace(
                testcase,
                dataflow_meta=dataflow,
                compare_mask=CompareMask(
                    outcome=True, checkpoint_pc=True,
                    signal=False, fault_pc=False, fault_address=False,
                    extra_state_keys=(),
                ),
                observability_contract=ObservabilityContract(
                    schema_version="observability-contract-v1",
                    risk_value_or_effect="instruction-outcome",
                    sink_instruction_ids=risk_ids,
                    risk_to_sink_distance=1,
                    sink_transform="identity",
                    final_observation_fields=("outcome", "checkpoint_pc"),
                    lossiness="none",
                ),
            )
    return replace(testcase, dataflow_meta=dataflow)


def apply_single_profile(program: object, profile: object, action: Mapping[str, object], *, _action_pool=None):
    from framework.rvemi.program import program_sha256

    if not isinstance(action, Mapping):
        return None
    # profile 缺失（reference 域外 root）时跳过身份校验：EMI 动作只依赖
    # case 本身，MCMC 仍要能物化候选。
    if profile is not None and (
        not getattr(profile, "complete", False)
        or program_sha256(program) != profile.source_sha256
    ):
        return None
    digest = canonical_digest(dict(action))
    if isinstance(_action_pool, Mapping):
        selected = _action_pool.get(digest)
        if selected is None:
            return None
    else:
        authorized = (
            tuple(single_profile_actions(program, profile))
            if profile is not None else
            tuple(enumerate_single_rewrites(program))
        )
        if not authorized:
            authorized = tuple(enumerate_single_rewrites(program))
        pool = tuple(_action_pool) if _action_pool is not None else authorized
        selected = next((item for item in pool if canonical_digest(dict(item)) == digest), None)
        if selected is None or not any(
            canonical_digest(dict(item)) == digest for item in authorized
        ):
            return None
    selected_candidate = _select_candidate(program, selected)
    if selected_candidate is None:
        return None
    testcase = _case(program)
    # replace_fragment 携带的是多个 (index, change)，逐条反向应用；
    # 早期实现把索引元组当单索引传给 _mutate_at，片段候选永远物化失败，
    # MCMC 唯一能做的结构变异因此全部丢失。
    indices, changes = selected_candidate[1], selected_candidate[2]
    operations = (
        tuple(zip(indices, changes))
        if isinstance(indices, (tuple, list)) else ((indices, changes),)
    )
    changed = testcase
    for index, change in reversed(operations):
        changed = _mutate_at(changed, index, change)
        if changed is None:
            return None
    changed = _classify_mutated_case(changed)
    if changed is None:
        return None
    if (reason := _candidate_observer_gap(changed)) is not None:
        raise ValueError("candidate-observer-register-gap:" + reason)
    changed = identify_testcase(changed)
    return _variant(program, changed, selected)


def _build_single_program(
    _source_path: Path, output_path: Path, run_params: Mapping[str, object],
    *, bare_metal: bool = False,
) -> Path:
    testcase = _require_rvgen_case(TestCase.from_dict(dict(run_params["single_case"])))
    temporary = output_path.with_name(f".{output_path.stem}.rvgen.elf")
    try:
        built = build_elf(testcase, temporary, bare_metal=bare_metal)
    except Exception:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(".S").unlink(missing_ok=True)
        raise
    try:
        if hasattr(built, "ok") and not built.ok:
            raise RuntimeError(
                f"single build failed: {getattr(built, 'stderr', '') or built.exit_code}"
            )
        shutil.copyfile(built.path, output_path)
    finally:
        for path in (getattr(built, "path", None), getattr(built, "source_path", None)):
            if path is not None and Path(path).resolve() != output_path.resolve():
                Path(path).unlink(missing_ok=True)
    return output_path


def build_single_program(source_path: Path, output_path: Path, run_params: Mapping[str, object]) -> Path:
    return _build_single_program(source_path, output_path, run_params)


def build_single_bare_program(
    source_path: Path, output_path: Path, run_params: Mapping[str, object],
) -> Path:
    return _build_single_program(source_path, output_path, run_params, bare_metal=True)


def _single_runner_env(
    testcase: TestCase, backend: str, *, rvvm_jit: bool | None = None,
    observer_fields: Sequence[str] = (),
) -> dict[str, str]:
    final_fp_required = final_fp_state_required_for_testcase(testcase) or any(
        str(field).removeprefix("extra.") in {
            "fpr_rawbits", "before_fpr_rawbits", "after_fpr_rawbits",
            "fflags", "before_fflags", "after_fflags", "csr.fflags",
            "frm", "before_frm", "after_frm", "csr.frm",
        }
        for field in observer_fields
    )
    runner_env = {
        "RV_TESTCASE_ISA_PROFILE_NAME": testcase.isa_profile,
        "RV_TESTCASE_REQUIRE_FINAL_FP_STATE": "1" if final_fp_required else "0",
    }
    disabled = set()
    for contract in testcase.dataflow_meta.get("risk_contracts", ()):
        boundary = contract.get("boundary_class") if isinstance(contract, Mapping) else None
        if isinstance(boundary, str) and boundary.startswith("extension-gate:"):
            disabled.update(boundary.removeprefix("extension-gate:").split("+"))
    if disabled:
        runner_env["RV_TESTCASE_DISABLED_EXTENSIONS"] = ",".join(sorted(disabled))
    if backend in CAPSULE_BACKENDS:
        runner_env["RVVM_NO_JIT"] = (
            "0" if rvvm_jit else "1"
        ) if rvvm_jit is not None else "1"
    if backend == "rvvm-riscv64":
        runner_env["RVVM_BARE_METAL_CAPSULE"] = "1"
    if expected_trap_from_dataflow(testcase.dataflow_meta):
        runner_env["RV_TESTCASE_EXPECTED_TRAP"] = "1"
    return runner_env


def _risk_pcs(testcase: TestCase, executable_path: Path) -> tuple[int, ...]:
    try:
        _, addresses = symbol_offsets_from_elf(executable_path, ("test_start",))
        start = int(addresses["test_start"], 0)
    except (KeyError, TypeError, ValueError):
        return ()
    return tuple(start + item.pc_offset for item in testcase.instruction_meta if "risk" in item.tags)


def _runtime_temporary_directory(prefix: str):
    """Use the run volume for capsules when the container root is read-only."""
    run_root = os.environ.get("RQ1_RUN")
    if run_root and Path(run_root).is_dir():
        return tempfile.TemporaryDirectory(prefix=prefix, dir=run_root)
    return tempfile.TemporaryDirectory(prefix=prefix)


def run_single_backend(
    artifact: object, backend: str, backend_binary_path: str | None = None,
    *, rvvm_jit: bool | None = None, timeout_seconds: float | None = None,
    campaign_deadline_monotonic: float | None = None,
):
    testcase = TestCase.from_dict(dict(artifact.run_params["single_case"]))
    preflight_gap = _renode_preflight_gap(artifact, testcase, backend)
    if preflight_gap is not None:
        return preflight_gap
    coverage_config = artifact.run_params.get("framework_coverage")
    if not isinstance(coverage_config, Mapping) or coverage_config.get("backend") != backend:
        coverage_config = None
    observer_fields = artifact.run_params.get("observer_fields", ())
    observer_fields = (
        tuple(observer_fields)
        if isinstance(observer_fields, (list, tuple))
        and all(isinstance(item, str) for item in observer_fields)
        else ()
    )
    runner_env = _single_runner_env(
        testcase, backend, rvvm_jit=rvvm_jit,
        observer_fields=observer_fields,
    )
    executable_path = Path(artifact.executable_path)
    artifact_bare_path = getattr(artifact, "bare_executable_path", None)
    if backend == "rax-riscv64" and not os.environ.get("RAX_SOURCE"):
        rax_source = Path(os.environ.get("RQ1_DEPS", "/path/to/deps")) / "simulator-sources" / "rax"
        if rax_source.is_dir():
            runner_env["RAX_SOURCE"] = str(rax_source)
    temporary = nullcontext()
    if backend in CAPSULE_BACKENDS and artifact_bare_path is not None:
        executable_path = Path(artifact_bare_path)
    elif backend == "rvvm-riscv64":
        # 兼容直接调用该函数的旧调用方；正式 campaign 会在 BuiltProgram
        # 中物化并校验双 ELF，优先使用上面的 companion。
        temporary = _runtime_temporary_directory(prefix=".rvvm-capsule-")
    elif backend in CAPSULE_BACKENDS:
        raise RuntimeError(f"bare-metal-artifact-missing:{backend}")
    with temporary as temp_dir:
        if backend == "rvvm-riscv64" and artifact_bare_path is None:
            executable_path = Path(temp_dir) / "program.elf"
            built = build_elf(testcase, executable_path, bare_metal=True)
            if not built.ok:
                raise RuntimeError(f"RVVM capsule build failed: {built.stderr or built.exit_code}")
        risk_pcs = _risk_pcs(testcase, executable_path)
        if backend.startswith("qemu-riscv") and risk_pcs:
            runner_env["RV_TESTCASE_RISK_PCS"] = ",".join(hex(pc) for pc in risk_pcs)
        if backend in CAPSULE_BACKENDS:
            _, addresses = symbol_offsets_from_elf(
                executable_path,
                ("obs_buf", "test_memory_start", "test_memory_end"),
                allow_before_start=True,
            )
            if all(name in addresses for name in ("obs_buf", "test_memory_start", "test_memory_end")):
                memory_start = int(addresses["test_memory_start"], 0)
                memory_end = int(addresses["test_memory_end"], 0)
                memory_size = memory_end - memory_start
                if memory_size < 0:
                    raise RuntimeError("single capsule memory symbols are reversed")
                runner_env.update({
                    CAPSULE_MAILBOX_ENV: addresses["obs_buf"],
                    CAPSULE_TEST_MEMORY_ENV: addresses["test_memory_start"],
                    CAPSULE_OBSERVATION_SIZE_ENV: str(OBSERVATION_HEADER_SIZE + memory_size),
                    CAPSULE_MEMORY_SIZE_ENV: str(memory_size),
                })
            elif backend in CAPSULE_BACKENDS:
                # Compatibility for direct callers whose capsule ELF
                # predates the common obs_buf/test_memory_* aliases.  All
                # capsule backends consume the same mailbox contract; RVVM
                # was previously the only backend allowed to use it.
                _, addresses = symbol_offsets_from_elf(
                    executable_path, ("result_buffer", "test_memory"),
                )
                memory_size = sum(len(region.data) for region in testcase.initial_memory_regions)
                if all(name in addresses for name in ("result_buffer", "test_memory")):
                    runner_env.update({
                        CAPSULE_MAILBOX_ENV: addresses["result_buffer"],
                        CAPSULE_TEST_MEMORY_ENV: addresses["test_memory"],
                        CAPSULE_OBSERVATION_SIZE_ENV: str(OBSERVATION_HEADER_SIZE + memory_size),
                        CAPSULE_MEMORY_SIZE_ENV: str(memory_size),
                    })
        observations = run_backend_with_coverage(
            backend, executable_path,
            profile_id=f"{testcase.input_id}/{backend}", input_id=testcase.input_id,
            runner_env=runner_env, backend_binary_path=backend_binary_path,
            coverage_config=coverage_config,
            timeout_seconds=timeout_seconds,
            campaign_deadline_monotonic=campaign_deadline_monotonic,
        )
        if not risk_pcs:
            return observations
        marked = []
        for observation in observations:
            extra = dict(getattr(observation, "extra_state", {}) or {})
            pcs = tuple(getattr(observation, "executed_pcs", ()) or ())
            evidence = getattr(observation, "translation_evidence", None)
            details = getattr(evidence, "details", {})
            no_pc_channel = not pcs and (
                backend == "rax-riscv64"
                and details.get("trace_mechanism") is None
                or backend == "rvvm-riscv64"
                and details.get("rvvm_trace_mechanism") == "gdb-rsp-mailbox-only"
            )
            if no_pc_channel:
                # 只在后端明确没有启用 PC trace 通道时记录路径不可用。
                extra["rvgen_risk_execution"] = {
                    "status": "unavailable",
                    "reason": "backend-path-unavailable",
                    "risk_pcs": list(risk_pcs),
                }
                marked.append(replace(observation, extra_state=extra))
                continue
            extra["rvgen_risk_execution"] = {
                "status": "observed" if all(
                    pc in pcs for pc in risk_pcs
                ) else "missing",
                "risk_pcs": list(risk_pcs),
            }
            marked.append(replace(observation, extra_state=extra))
        return tuple(marked)


__all__ = [
    "apply_single_hint", "apply_single_profile", "build_single_profile",
    "build_single_program", "enumerate_single_rewrites", "is_single_program",
    "program_from_testcase", "run_single_backend",
    "single_profile_actions", "single_program_source", "single_rewrite_realizes",
]

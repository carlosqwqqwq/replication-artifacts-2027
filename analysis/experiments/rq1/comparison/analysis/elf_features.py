#!/usr/bin/env python3
"""从最终 RISC-V ELF 生成覆盖率注册表、静态特征和 reference 行为签名。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from framework.riscv_catalog import (
        OFFICIAL_ALL_CATALOG_FORMS,
        catalog_partition_for_form,
    )
    from framework.riscv_encoding import decode_form
    from framework.spec_definedness import (
        is_canonical_isa_profile,
        profile_enables_form,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from framework.riscv_catalog import (
        OFFICIAL_ALL_CATALOG_FORMS,
        catalog_partition_for_form,
    )
    from framework.riscv_encoding import decode_form
    from framework.spec_definedness import (
        is_canonical_isa_profile,
        profile_enables_form,
    )

DECODER_VERSION = "riscv-opcodes-aeb94df-coverage-v4"
SIGNATURE_VERSION = "rq1-reference-behavior-signature-v1"
CATALOG_SOURCE = "riscv-opcodes@aeb94df5dd272d1893d43af270dd8d1b2fef699b"
ISA_SPEC_RELEASE = "v20260120"
TAXONOMY_VERSION = "rq1-instruction-classes-v2"
REGISTRY_SCHEMA = "rq1-isa-coverage-registry-v2"
# 放开 ISA/扩展族：不再用静态白名单限制生成范围。生成器可以产出自身支持的任意
# 扩展；模拟器不支持的指令在执行期报错，按目标侧终态记录后继续下一个 case。
# 分母仍由实际 profile 的合法 form 集合派生（见 _eligible_forms），不随命中收缩。
COMMON_PROFILES = (
    "rv64i", "rv64i_zicsr", "rv64i_zifencei", "rv64i_zicsr_zifencei",
    "rv64im", "rv64im_zicsr", "rv64im_zifencei", "rv64im_zicsr_zifencei",
    "rv64imc", "rv64imc_zicsr", "rv64imc_zifencei", "rv64imc_zicsr_zifencei",
)
DEFAULT_PROFILES = COMMON_PROFILES + (
    "rv64imafd_zicsr", "rv64imafdc_zicsr", "rv64imafdc_zicsr_zifencei",
)
DEFAULT_PRIVILEGE_MODES = ("user", "machine")
_SEALED_CACHE: dict[tuple[int, str], bool] = {}

_BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu", "c_beqz", "c_bnez"})
_LOAD_SIZES = {
    "lb": "b", "lbu": "b", "lh": "h", "lhu": "h", "lw": "w", "lwu": "w", "ld": "d",
    "c_lw": "w", "c_ld": "d", "c_lwsp": "w", "c_ldsp": "d",
}
_STORE_SIZES = {
    "sb": "b", "sh": "h", "sw": "w", "sd": "d",
    "c_sw": "w", "c_sd": "d", "c_swsp": "w", "c_sdsp": "d",
}
_MULTIPLY = frozenset({"mul", "mulh", "mulhsu", "mulhu", "mulw"})
_DIVIDE = frozenset({"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"})
_INTEGER_REGISTER = frozenset({
    "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    "c_add", "c_mv", "c_sub", "c_xor", "c_or", "c_and",
})
_INTEGER_WORD = frozenset({
    "addiw", "slliw", "srliw", "sraiw", "addw", "subw", "sllw", "srlw", "sraw",
    "c_addiw", "c_addw", "c_subw",
})
_INTEGER_IMMEDIATE = frozenset({
    "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai",
    "c_addi4spn", "c_addi", "c_li", "c_addi16sp", "c_andi", "c_srli", "c_srai", "c_slli",
})


def _form_name(form: Any) -> str:
    return str(form.mnemonic).replace("_", ".")


def _form_key(form: Any) -> str:
    return str(form.mnemonic).lower()


_FORM_BY_KEY = {_form_key(form): form for form in OFFICIAL_ALL_CATALOG_FORMS}


def _semantic_type(mnemonic: str, extension: str) -> str:
    name = mnemonic.replace(".", "_").lower()
    if name in _BRANCHES:
        return "control-conditional"
    if name in _LOAD_SIZES:
        return "memory-load"
    if name in _STORE_SIZES:
        return "memory-store"
    if name in {"jal", "jalr", "c_j", "c_jr"}:
        return "control-jump"
    if name in {"c_jalr"}:
        return "control-call"
    if name == "mret":
        return "control-return"
    if name == "wfi":
        return "system-wait"
    if name in {"ecall", "ebreak", "c_ebreak"}:
        return "system-environment"
    if name.startswith("csrr"):
        return "system-csr"
    if extension == "a":
        return "atomic"
    if extension in {"f", "d", "c_f", "c_d"}:
        return "floating-point"
    if name in {"fence", "fence_i"}:
        return "system-fence"
    if name in {"lui", "auipc", "c_lui"}:
        return "integer-upper"
    if name in _INTEGER_WORD:
        return "integer-word"
    if extension == "m" and name in _MULTIPLY | _DIVIDE:
        return "integer-register"
    if name in _INTEGER_IMMEDIATE or name in {"c_nop"}:
        return "integer-immediate"
    if name in _INTEGER_REGISTER:
        return "integer-register"
    form = _FORM_BY_KEY.get(name)
    if form is not None and (int(form.match) & 0x7f) in {0x33, 0x3b}:
        return "integer-register"
    if extension == "c":
        return "integer-compressed"
    return "integer-other"


def _type_ids(mnemonic: str, semantic: str) -> list[str]:
    name = mnemonic.replace(".", "_").lower()
    ids = {f"type:{semantic}"}
    if name.startswith("c_"):
        ids.add("type:integer-compressed")
    if name in _MULTIPLY:
        ids.add("type:integer-multiply")
    if name in _DIVIDE:
        ids.add("type:integer-divide")
    if semantic == "atomic":
        ids.add("type:atomic")
    if semantic == "floating-point":
        ids.add("type:floating-point")
    return sorted(ids)


def _type_id(semantic: str) -> str:
    return f"type:{semantic}"


def _configuration_bin(mnemonic: str, semantic: str) -> str:
    name = mnemonic.replace(".", "_").lower()
    if name in _LOAD_SIZES:
        return f"cfg:memory:load:{_LOAD_SIZES[name]}"
    if name in _STORE_SIZES:
        return f"cfg:memory:store:{_STORE_SIZES[name]}"
    if semantic.startswith("control-"):
        return f"cfg:control:{semantic[8:]}"
    if semantic.startswith("system-"):
        return f"cfg:system:{semantic[7:]}"
    if semantic == "atomic":
        return "cfg:atomic"
    if semantic == "floating-point":
        return "cfg:floating-point"
    if semantic == "integer-immediate":
        return "cfg:integer:immediate"
    if semantic == "integer-register":
        return "cfg:integer:register"
    if semantic == "integer-upper":
        return "cfg:integer:upper"
    if semantic == "integer-word":
        return "cfg:integer:word"
    return "cfg:integer:other"


def _configuration_bins(mnemonic: str, semantic: str) -> list[str]:
    name = mnemonic.replace(".", "_").lower()
    bins = {_configuration_bin(mnemonic, semantic)}
    if name.startswith("c_"):
        bins.add("cfg:integer:compressed")
    if name in _MULTIPLY:
        bins.add("cfg:integer:multiply")
    if name in _DIVIDE:
        bins.add("cfg:integer:divide")
    return sorted(bins)


def _profile_from_lane(lane: object) -> str | None:
    if not isinstance(lane, str) or not lane.strip():
        return None
    profile = lane.split("/", 1)[0].strip().lower()
    return profile if is_canonical_isa_profile(profile) else None


def _validate_coverage_profile(profile: str, privilege_mode: str) -> None:
    if not is_canonical_isa_profile(profile):
        raise ValueError("isa-profile-not-canonical")
    if privilege_mode not in {"user", "machine"}:
        raise ValueError("privilege-mode-not-supported")


def _form_enabled(form: Any, profile: str, privilege_mode: str) -> bool:
    if not profile_enables_form(profile, form):
        return False
    if catalog_partition_for_form(form) != "ratified":
        return False
    if form.extension_group == "system" and privilege_mode != "machine":
        return False
    return True


def _eligible_forms(profile: str, privilege_mode: str) -> tuple[Any, ...]:
    _validate_coverage_profile(profile, privilege_mode)
    return tuple(
        form for form in OFFICIAL_ALL_CATALOG_FORMS
        if _form_enabled(form, profile, privilege_mode)
    )


def _unit_id(mnemonic: str, spec: dict[str, Any]) -> str:
    extension = str(spec["extension"]).lower()
    return f"enc:{spec['length_bits']}:rv64{extension}:{mnemonic}"


def _form_unit_id(form: Any) -> str:
    name = _form_name(form)
    return _unit_id(name, {
        "length_bits": int(form.encoding_length_bytes) * 8,
        "extension": str(form.extension_group),
    })


def _legal_form_encoding(form: Any, raw: int) -> bool:
    """Reject reserved operands while retaining architected HINT code points."""
    name = _form_key(form)
    try:
        operands = decode_form(form, raw)
    except (TypeError, ValueError):
        return False
    if name == "c_addi4spn":
        return operands.get("c_nzuimm10") != 0
    if name == "c_addi16sp":
        return operands.get("c_nzimm10") != 0
    if name == "c_lui":
        return operands.get("rd_n2") != 2 and operands.get("c_nzimm18") != 0
    if name == "c_addiw":
        return operands.get("rd_rs1_n0") != 0
    if name in {"c_lwsp", "c_ldsp"}:
        return operands.get("rd_n0") != 0
    if name == "c_jr":
        return operands.get("rs1_n0") != 0
    if name in {"c_mv", "c_add"}:
        return operands.get("c_rs2_n0") != 0
    if name == "fence_i":
        return (operands.get("imm12") == 0 and operands.get("rs1") == 0
                and operands.get("rd") == 0)
    if name == "fence":
        fm = operands.get("fm")
        pred = operands.get("pred")
        succ = operands.get("succ")
        rs1 = operands.get("rs1")
        rd = operands.get("rd")
        if fm != 0 or pred is None or succ is None or rs1 is None or rd is None:
            # Ztso's FENCE.TSO mode is outside the current RQ1 profile set.
            return False
        if rs1 == 0 and rd == 0:
            return True
        # RV64I reserves the remaining FENCE encodings except these HINTs.
        return (pred == 0 or succ == 0) and ((rs1 == 0) != (rd == 0))
    return True


def _build_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for form in OFFICIAL_ALL_CATALOG_FORMS:
        mnemonic = _form_name(form)
        specs[mnemonic] = {
            "length_bits": int(form.encoding_length_bytes) * 8,
            "mask": hex(int(form.mask)),
            "match": hex(int(form.match)),
            "extension": str(form.extension_group),
            "semantic_type": _semantic_type(mnemonic, str(form.extension_group)),
            "form": form,
        }
    return specs


SPECS = _build_specs()
_MASK_KEYS = {
    width: sorted(
        ((int(spec["mask"], 16), int(spec["match"], 16), name, spec["form"])
         for name, spec in SPECS.items() if spec["length_bits"] == width),
        key=lambda item: (-int(item[0]).bit_count(), item[1], item[2]),
    )
    for width in (16, 32)
}


def _decode_word(
    width: int, raw: int, isa_profile: str = "rv64imc_zicsr_zifencei",
    privilege_mode: str = "machine",
) -> str | None:
    try:
        _validate_coverage_profile(isa_profile, privilege_mode)
    except ValueError:
        return None
    for mask, match, name, form in _MASK_KEYS.get(width, ()):
        if raw & mask == match and _form_enabled(form, isa_profile, privilege_mode) \
                and _legal_form_encoding(form, raw):
            return name
    return None


def _profile_registry_body(profile: str, privilege_mode: str) -> dict[str, Any]:
    forms = _eligible_forms(profile, privilege_mode)
    encoding = sorted(_form_unit_id(form) for form in forms)
    types = sorted({
        type_id for form in forms
        for type_id in _type_ids(_form_name(form), _semantic_type(_form_name(form), form.extension_group))
    })
    configurations = sorted({
        configuration for form in forms
        for configuration in _configuration_bins(
            _form_name(form), _semantic_type(_form_name(form), form.extension_group)
        )
    })
    metrics = {
        "ICov-encoding": encoding,
        "ICov-type": types,
        "CCov": configurations,
    }
    body = {
        "schema_version": REGISTRY_SCHEMA,
        "decoder_version": DECODER_VERSION,
        "catalog_source": CATALOG_SOURCE,
        "isa_spec_release": ISA_SPEC_RELEASE,
        "taxonomy_version": TAXONOMY_VERSION,
        "isa_profile": profile,
        "privilege_mode": privilege_mode,
        "eligible_form_count": len(forms),
        "metrics": metrics,
    }
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {**body, "registry_sha256": digest,
            "metric_profile_id": f"{profile}|{privilege_mode}|{digest[:16]}"}


@lru_cache(maxsize=64)
def metric_registry_for_profile(profile: str, privilege_mode: str = "user") -> dict[str, Any]:
    return _profile_registry_body(profile.strip().lower(), privilege_mode)


def seal_entry(
    kind: str, units: list[dict[str, Any]],
    eligible_bins_by_profile: dict[str, list[str]],
) -> dict[str, Any]:
    unit_fields = {
        "encoding": ["id", "length_bits", "mask", "match", "extension", "mnemonic"],
        "type": ["id", "semantic_type"],
        "configuration": ["id", "configuration"],
    }[kind]
    body = {
        "kind": kind, "decoder_version": DECODER_VERSION,
        "catalog_source": CATALOG_SOURCE, "isa_spec_release": ISA_SPEC_RELEASE,
        "taxonomy_version": TAXONOMY_VERSION, "unit_fields": unit_fields,
        "units": units, "eligible_bins_by_profile": eligible_bins_by_profile,
    }
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {**body, "sealed": True, "registry_sha256": digest,
            "seal": {"schema_version": "rq1-coverage-registry-seal-v2",
                     "algorithm": "sha256", "canonical_sha256": digest}}


def verify_sealed_entry(entry: Any, kind: str) -> bool:
    fields = (
        "kind", "decoder_version", "catalog_source", "isa_spec_release",
        "taxonomy_version", "unit_fields", "units", "eligible_bins_by_profile",
    )
    if not isinstance(entry, dict) or entry.get("sealed") is not True or any(
        field not in entry for field in fields
    ):
        return False
    body = {field: entry[field] for field in fields}
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    cache_key = (id(entry), digest)
    if cache_key in _SEALED_CACHE:
        return _SEALED_CACHE[cache_key]
    seal = entry.get("seal")
    result = (
        body["kind"] == kind and entry.get("registry_sha256") == digest
        and isinstance(seal, dict)
        and seal.get("schema_version") == "rq1-coverage-registry-seal-v2"
        and seal.get("algorithm") == "sha256"
        and seal.get("canonical_sha256") == digest
    )
    _SEALED_CACHE[cache_key] = result
    return result


def metric_registry_fragment() -> dict[str, Any]:
    profiles = {
        f"{profile}@{mode}": metric_registry_for_profile(profile, mode)
        for profile in DEFAULT_PROFILES for mode in DEFAULT_PRIVILEGE_MODES
    }
    encoding_forms = {
        _form_unit_id(form): form
        for profile in DEFAULT_PROFILES for mode in DEFAULT_PRIVILEGE_MODES
        for form in _eligible_forms(profile, mode)
    }
    encoding_units = [{
        "id": unit_id, "length_bits": form.encoding_length_bytes * 8,
        "mask": hex(form.mask), "match": hex(form.match),
        "extension": form.extension_group, "mnemonic": _form_name(form),
    } for unit_id, form in sorted(encoding_forms.items())]
    type_units = sorted({
        type_id for registry in profiles.values()
        for type_id in registry["metrics"]["ICov-type"]
    })
    configuration_units = sorted({
        configuration for registry in profiles.values()
        for configuration in registry["metrics"]["CCov"]
    })
    by_profile = {
        profile_key: registry["metrics"]["ICov-encoding"]
        for profile_key, registry in profiles.items()
    }
    type_by_profile = {
        profile_key: registry["metrics"]["ICov-type"]
        for profile_key, registry in profiles.items()
    }
    configuration_by_profile = {
        profile_key: registry["metrics"]["CCov"]
        for profile_key, registry in profiles.items()
    }
    return {
        "encoding": seal_entry("encoding", encoding_units, by_profile),
        "type": seal_entry("type", [
            {"id": value, "semantic_type": value.removeprefix("type:")}
            for value in type_units
        ], type_by_profile),
        "configuration": seal_entry("configuration", [
            {"id": value, "configuration": value.removeprefix("cfg:")}
            for value in configuration_units
        ], configuration_by_profile),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _objdump_line(line: str) -> tuple[int, int, int, str, str] | None:
    parts = line.split()
    if not parts or not re.fullmatch(r"[0-9a-fA-F]+:", parts[0]):
        return None
    address = int(parts[0][:-1], 16)
    raw = []
    index = 1
    while index < len(parts) and re.fullmatch(r"[0-9a-fA-F]{4,8}", parts[index]):
        raw.append(parts[index])
        index += 1
    if not raw or index >= len(parts):
        return None
    width = sum(len(item) for item in raw) * 4
    if width not in {16, 32, 48, 64}:
        return None
    return address, width, int("".join(raw), 16), parts[index].lower(), " ".join(parts[index + 1:])


def _target(operands: str) -> int | None:
    values = re.findall(r"(?:0x)?[0-9a-fA-F]{4,}", operands)
    return int(values[-1], 16) if values else None


def _cfg(instructions: list[dict[str, Any]]) -> list[str]:
    addresses = {item["address"]: index for index, item in enumerate(instructions)}
    conditional = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
    jumps = {"jal", "c.j"}
    result = set()
    for index, item in enumerate(instructions):
        mnemonic = item["mnemonic"]
        if mnemonic == "c.unimp":
            continue
        target = _target(item["operands"]) if mnemonic in conditional | jumps else None
        if target is not None:
            destination = addresses.get(target, f"external:0x{target:x}")
            result.add(f"cfg:{index}->{destination}")
        elif mnemonic in {"jalr", "c.jr", "c.jalr"}:
            result.add(f"cfg:{index}->indirect")
        elif index + 1 < len(instructions):
            result.add(f"cfg:{index}->{index + 1}")
        if mnemonic in conditional and index + 1 < len(instructions):
            result.add(f"cfg:{index}->{index + 1}")
        if mnemonic == "jal" and item["operands"].split(",", 1)[0].strip() not in {"x0", "zero"} \
                and index + 1 < len(instructions):
            result.add(f"cfg:{index}->{index + 1}")
    return sorted(result)


def decode_elf(
    elf: Path, lane: str, objdump: str | None = None,
    privilege_mode: str = "user",
) -> dict[str, Any]:
    elf = elf.resolve()
    profile = _profile_from_lane(lane)
    if profile is None:
        return {"schema_version": "rq1-artifact-features-v2", "status": "gap",
                "reason": "isa-profile-not-canonical", "lane": lane}
    if not elf.is_file():
        return {"schema_version": "rq1-artifact-features-v2", "status": "gap",
                "reason": "elf-missing", "lane": lane,
                "isa_profile": profile, "privilege_mode": privilege_mode,
                "elf_path": str(elf)}
    try:
        registry = metric_registry_for_profile(profile, privilege_mode)
    except ValueError as exc:
        return {"schema_version": "rq1-artifact-features-v2", "status": "gap",
                "reason": str(exc), "lane": lane, "isa_profile": profile,
                "privilege_mode": privilege_mode}
    objdump = objdump or shutil.which("riscv64-linux-gnu-objdump") or "riscv64-linux-gnu-objdump"
    try:
        # ``-d`` treats symbols such as RVGEN's entry/exit checkpoints as data
        # and emits their instruction bytes as .word/.short.  ``-D -j .text``
        # keeps the scope identical while decoding the actual variable-width
        # instruction stream, so dynamic compressed PCs can map back to ELF.
        result = subprocess.run([objdump, "-D", "-j", ".text", "-M", "no-aliases", str(elf)],
                                capture_output=True, text=True, check=False,
                                timeout=30)
    except subprocess.TimeoutExpired:
        return {"schema_version": "rq1-artifact-features-v2", "status": "gap",
                "reason": "objdump-timeout", "lane": lane,
                "isa_profile": profile, "privilege_mode": privilege_mode}
    except OSError as exc:
        return {"schema_version": "rq1-artifact-features-v2", "status": "gap",
                "reason": f"objdump-unavailable:{type(exc).__name__}",
                "lane": lane, "isa_profile": profile, "privilege_mode": privilege_mode}
    instructions = []
    unknown = []
    reserved_encodings = []
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    for line in stdout.splitlines():
        parsed = _objdump_line(line)
        if parsed is None:
            continue
        address, width, raw, disassembled_mnemonic, operands = parsed
        mnemonic = _decode_word(width, raw, profile, privilege_mode)
        privileged_stub = False
        if mnemonic is None and disassembled_mnemonic in SPECS:
            privileged_mnemonic = _decode_word(width, raw, profile, "machine")
            if privileged_mnemonic == disassembled_mnemonic:
                # User ELFs may contain legal but unreachable privileged stubs.
                mnemonic = privileged_mnemonic
                privileged_stub = True
        spec = SPECS.get(mnemonic) if mnemonic else None
        if spec is None and width == 16 and raw == 0 and disassembled_mnemonic in {
            "c.unimp", ".insn",
        }:
            reserved_encodings.append({
                "address": address, "width_bits": width, "raw": "0x0",
                "mnemonic": disassembled_mnemonic,
            })
            extension, semantic = "reserved", "reserved"
            encoding_id, type_id = "reserved:16:c.unimp", "type:reserved"
            type_ids, configuration_ids = [type_id], ["cfg:reserved"]
            mnemonic = "c.unimp"
        elif spec is None:
            unknown.append(disassembled_mnemonic)
            extension, semantic = "unknown", "unknown"
            encoding_id, type_ids = f"unknown:{width}:{disassembled_mnemonic}", ["type:unknown"]
            configuration_ids = ["cfg:unknown"]
            mnemonic = disassembled_mnemonic
            type_id = "type:unknown"
        else:
            extension = str(spec["extension"])
            semantic = str(spec["semantic_type"])
            encoding_id = _unit_id(mnemonic, spec)
            type_ids = _type_ids(mnemonic, semantic)
            type_id = _type_id(semantic)
            configuration_ids = _configuration_bins(mnemonic, semantic)
            if mnemonic != disassembled_mnemonic:
                # The raw codepoint selected a more-specific legal form or a
                # canonical alias; keep operands only when objdump named it.
                operands = ""
        instruction = {
            "index": len(instructions), "address": address, "width_bits": width,
            "raw": f"0x{raw:x}", "mnemonic": mnemonic, "operands": operands,
            "encoding_id": encoding_id, "type_id": type_id, "type_ids": type_ids,
            "extension": extension, "semantic_type": semantic,
            "configuration_id": configuration_ids[0],
            "configuration_ids": configuration_ids,
        }
        if privileged_stub:
            # Keep the instruction in the PC map so a trace can be diagnosed,
            # but do not count an unreachable machine-only stub in a user-mode
            # generated/executed coverage denominator.
            instruction["coverage_excluded_metrics"] = [
                "GenCov", "ExecCov", "OpcodeCov",
            ]
            instruction["coverage_exclusion_reason"] = "privileged-stub-in-user-elf"
        elif encoding_id == "reserved:16:c.unimp":
            # RISC-V objdump renders zero padding in an executable text
            # section as c.unimp. Keep its PC for trace diagnostics, but it is
            # a reserved encoding, not an RQ1 instruction-universe member.
            instruction["coverage_excluded_metrics"] = [
                "GenCov", "ExecCov", "OpcodeCov",
            ]
            instruction["coverage_exclusion_reason"] = (
                "reserved-encoding-outside-rq1-instruction-universe"
            )
        instructions.append(instruction)
    status = "ok" if result.returncode == 0 and instructions and not unknown else "gap"
    reason = None if status == "ok" else (
        "unknown-or-reserved-instruction-encoding" if unknown else "elf-disassembly-empty"
    )
    configuration_bins = {configuration_id for item in instructions
                          for configuration_id in item["configuration_ids"]}
    return {
        "schema_version": "rq1-artifact-features-v2", "status": status, "reason": reason,
        "decoder_version": DECODER_VERSION, "elf_path": str(elf), "elf_sha256": _sha256(elf),
        "lane": lane, "isa_profile": profile, "privilege_mode": privilege_mode,
        "coverage_profile_id": registry["metric_profile_id"],
        "coverage_registry_sha256": registry["registry_sha256"],
        "coverage_registry": {
            name: hashlib.sha256(json.dumps(units, separators=(",", ":")).encode()).hexdigest()
            for name, units in registry["metrics"].items()
        },
        "machine_code_scope": "text", "instruction_count": len(instructions),
        "unknown_mnemonics": sorted(set(unknown)), "reserved_encodings": reserved_encodings,
        "instructions": instructions,
        "machine_code_sequence_tokens": [item["encoding_id"] for item in instructions],
        "instruction_encoding_tokens": [item["encoding_id"] for item in instructions],
        "instruction_type_tokens": [type_id for item in instructions for type_id in item["type_ids"]],
        "encoding_bins": sorted({item["encoding_id"] for item in instructions}),
        "type_bins": sorted({type_id for item in instructions for type_id in item["type_ids"]}),
        "configuration_bins": sorted(configuration_bins),
        "cfg_repr_version": DECODER_VERSION, "cfg_decode_status": status,
        "cfg_edge_tokens": _cfg(instructions), "cfg_edges": _cfg(instructions),
        "machine_code_repr_version": DECODER_VERSION,
    }


def _pc_values(value: Any) -> list[int]:
    if not isinstance(value, dict):
        return []
    for key in ("trace_pcs", "pc_trace", "pcs", "observed_pcs"):
        raw = value.get(key)
        if not isinstance(raw, list):
            continue
        result = []
        for item in raw:
            if isinstance(item, bool):
                return []
            if isinstance(item, float) and (
                    not math.isfinite(item) or not item.is_integer()):
                return []
            try:
                parsed = int(item, 0) if isinstance(item, str) else int(item)
            except (TypeError, ValueError, OverflowError):
                return []
            if not 0 <= parsed < 1 << 64:
                return []
            result.append(parsed)
        return result
    return []


def reference_coverage(feature: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Map a trusted PC trace to the pinned, profile-specific ISA registry."""
    if isinstance(feature, dict) and feature.get("decoder_version") != DECODER_VERSION:
        return {"schema_version": "rq1-trusted-reference-coverage-v2",
                "decoder_version": feature.get("decoder_version"), "status": "gap",
                "reason": "static-feature-map-stale"}
    if isinstance(feature, dict) and feature.get("status", "ok") != "ok":
        return {"schema_version": "rq1-trusted-reference-coverage-v2",
                "decoder_version": feature.get("decoder_version"), "status": "gap",
                "reason": "static-feature-map-invalid"}
    profile = (feature.get("isa_profile") if isinstance(feature, dict) else None) \
        or _profile_from_lane(feature.get("lane") if isinstance(feature, dict) else None)
    privilege_mode = str(feature.get("privilege_mode") or "user") if isinstance(feature, dict) else "user"
    try:
        registry = metric_registry_for_profile(str(profile or ""), privilege_mode)
    except ValueError as exc:
        return {"schema_version": "rq1-trusted-reference-coverage-v2",
                "decoder_version": feature.get("decoder_version") if isinstance(feature, dict) else None,
                "status": "gap", "reason": str(exc)}
    instructions = feature.get("instructions") if isinstance(feature, dict) else None
    static_map_reason = None
    if not isinstance(instructions, list) or not instructions:
        static_map_reason = "static-feature-map-incomplete"
        instructions = []
    elif any(
        not isinstance(item, dict)
        or type(item.get("address")) is not int
        or not 0 <= item.get("address") < 1 << 64
        or not isinstance(item.get("encoding_id"), str)
        or not item.get("encoding_id")
        for item in instructions
    ):
        static_map_reason = "static-feature-map-incomplete"
        instructions = []
    elif len({item["address"] for item in instructions}) != len(instructions):
        static_map_reason = "static-feature-map-duplicate-pc"
        instructions = []
    by_pc = {item["address"]: item for item in instructions}
    pcs = _pc_values(observation)
    bounds = (min(by_pc), max(by_pc)) if by_pc else (None, None)
    scoped = [pc for pc in pcs if bounds[0] is not None and bounds[0] <= pc <= bounds[1]]
    mapped = [by_pc[pc] for pc in scoped if pc in by_pc]
    unknown = len(scoped) - len(mapped)
    complete = (
        static_map_reason is None
        and bool(pcs) and bool(by_pc) and bool(mapped) and not unknown
    )
    observed = [
        item for item in mapped
        if not item.get("coverage_excluded_metrics")
    ] if complete else []
    instruction_encoding_bins = sorted({item["encoding_id"] for item in observed})
    instruction_type_bins = sorted({
        type_id for item in observed
        for type_id in (item.get("type_ids") or
                        ([item["type_id"]] if item.get("type_id") else []))
    })
    configuration_bins = sorted({
        configuration_id for item in observed
        for configuration_id in (item.get("configuration_ids") or
                                 ([item["configuration_id"]]
                                  if item.get("configuration_id") else []))
    })
    metric_inputs = {
        "ICov-encoding": instruction_encoding_bins,
        "ICov-type": instruction_type_bins,
        "CCov": configuration_bins,
    }
    metrics = {}
    profile_mismatch = False
    registry_hashes = {}
    for name, values in metric_inputs.items():
        eligible = set(registry["metrics"][name])
        out_of_profile = set(values) - eligible
        profile_mismatch |= bool(out_of_profile)
        covered = set(values) & eligible if complete else set()
        metric_status = "gap" if out_of_profile else (
            "observed" if complete and eligible else "gap" if not complete else "NA"
        )
        registry_hashes[name] = hashlib.sha256(json.dumps(
            sorted(eligible), separators=(",", ":")
        ).encode()).hexdigest()
        metrics[name] = {
            "covered": len(covered), "eligible": len(eligible),
            "value": round(len(covered) / len(eligible), 6)
            if metric_status == "observed" else None,
            "status": metric_status,
            "reason": None if metric_status == "observed" else (
                "feature-unit-outside-isa-profile" if out_of_profile else
                "reference-trace-or-pc-map-incomplete" if not complete else
                "isa-profile-registry-empty"
            ),
            "covered_units": sorted(covered), "eligible_units": sorted(eligible),
            "registry_sha256": registry_hashes[name],
        }
    return {
        "schema_version": "rq1-trusted-reference-coverage-v2",
        "decoder_version": feature.get("decoder_version") if isinstance(feature, dict) else None,
        "isa_profile": profile, "privilege_mode": privilege_mode,
        "coverage_profile_id": registry["metric_profile_id"],
        "coverage_registry_sha256": registry["registry_sha256"],
        "trace_records": len(pcs), "scoped_trace_records": len(scoped),
        "mapped_records": len(mapped), "unknown_pc_records": unknown,
        "ignored_pc_records": len(pcs) - len(scoped),
        "scope_start": f"0x{bounds[0]:x}" if bounds[0] is not None else None,
        "scope_end": f"0x{bounds[1]:x}" if bounds[1] is not None else None,
        "status": "gap" if profile_mismatch else "ok" if complete else "gap",
        "reason": "feature-unit-outside-isa-profile" if profile_mismatch else
        static_map_reason if static_map_reason else
        None if complete else "reference-trace-or-pc-map-incomplete",
        "instruction_encoding_bins": instruction_encoding_bins,
        "instruction_type_bins": instruction_type_bins,
        "configuration_bins": configuration_bins,
        "metrics": metrics, "coverage_registry": registry_hashes,
    }


def behavior_signature(observation: dict[str, Any]) -> dict[str, Any]:
    pcs = _pc_values(observation)
    outcome = observation.get("outcome") or observation.get("target_outcome") or observation.get("status")
    outcome = str(outcome).strip().lower() if outcome not in (None, "") else None
    exit_class = "timeout" if outcome == "timeout" else "trap" if outcome in {"trap", "crash"} else (
        "nonzero" if observation.get("exit_code") not in (None, 0, "0") else "normal")
    transitions = [f"edge:{index}->{index + 1}" for index in range(max(0, len(pcs) - 1))]
    deltas = [pcs[index + 1] - pcs[index] for index in range(max(0, len(pcs) - 1))]
    repeats = Counter(pcs)
    repeat_bins = [f"repeat:{pc:x}:{'1' if count == 1 else '2-3' if count <= 3 else '4+'}"
                   for pc, count in sorted(repeats.items())]
    bins = sorted(set(transitions + repeat_bins + [f"exit:{exit_class}"] +
                      [f"pc-delta:{delta:+d}" for delta in deltas]))
    complete = bool(pcs) and outcome is not None
    return {"schema_version": "rq1-reference-behavior-signature-v1",
            "signature_version": SIGNATURE_VERSION, "version": SIGNATURE_VERSION,
            "complete": complete, "exit_class": exit_class, "outcome": outcome,
            "trace_records": len(pcs), "pc_trace": [f"0x{pc:x}" for pc in pcs],
            "pc_deltas": deltas, "transition_bins": transitions, "repeat_bins": repeat_bins,
            "dynamic_bins": bins, "behavior_bins": bins,
            **({"missing_reason": "reference-trace-or-outcome-missing"} if not complete else {})}


def self_check() -> dict[str, Any]:
    fragment = metric_registry_fragment()
    assert set(fragment) == {"encoding", "type", "configuration"}
    assert all(verify_sealed_entry(fragment[kind], kind) for kind in fragment)
    assert {"c.srli", "c.srai", "sraw", "c.nop"} <= SPECS.keys()
    assert SPECS["csrrw"]["extension"] == "zicsr"
    assert SPECS["fence.i"]["extension"] == "zifencei"
    assert SPECS["mret"]["extension"] == "system"
    assert "unimp" not in SPECS and "c.unimp" not in SPECS
    assert _configuration_bins("c.addi", "integer-immediate") == [
        "cfg:integer:compressed", "cfg:integer:immediate"]
    assert _configuration_bin("lbu", "memory-load") == "cfg:memory:load:b"
    assert "cfg:integer:multiply" in _configuration_bins("mul", "integer-register")
    assert "cfg:integer:divide" in _configuration_bins("div", "integer-register")
    assert _decode_word(16, 0x0004) is None
    assert _decode_word(16, 0x8002) is None
    assert _decode_word(16, 0x0001) == "c.nop"

    base = metric_registry_for_profile("rv64im", "user")
    machine = metric_registry_for_profile("rv64im", "machine")
    assert len(base["metrics"]["ICov-encoding"]) == 65
    assert len(machine["metrics"]["ICov-encoding"]) == 67
    assert base["metric_profile_id"] != machine["metric_profile_id"]
    assert len(metric_registry_for_profile(
        "rv64imc_zicsr", "user")["metrics"]["ICov-encoding"]) == 104
    compressed = metric_registry_for_profile("rv64imc_zicsr_zifencei", "machine")
    assert len(compressed["metrics"]["ICov-encoding"]) == 107
    comparison_profiles = {
        mode: metric_registry_for_profile("rv64imc_zicsr_zifencei", mode)
        for mode in DEFAULT_PRIVILEGE_MODES
    }
    assert len(comparison_profiles["user"]["metrics"]["ICov-encoding"]) == 105
    assert {
        name: len(units)
        for name, units in comparison_profiles["user"]["metrics"].items()
    } == {
        "ICov-encoding": 105, "ICov-type": 15, "CCov": 21,
    }
    assert {
        name: len(units)
        for name, units in comparison_profiles["machine"]["metrics"].items()
    } == {
        "ICov-encoding": 107, "ICov-type": 17, "CCov": 23,
    }
    for profile in COMMON_PROFILES:
        for mode in DEFAULT_PRIVILEGE_MODES:
            local = metric_registry_for_profile(profile, mode)["metrics"]
            common = comparison_profiles[mode]["metrics"]
            assert all(set(local[name]) <= set(common[name]) for name in common)
    assert "rv64imc_zicsr@user" in fragment["encoding"]["eligible_bins_by_profile"]
    assert any("csrrw" in unit for unit in metric_registry_for_profile(
        "rv64im_zicsr", "user")["metrics"]["ICov-encoding"])
    assert all(
        _type_ids(_form_name(form), _semantic_type(_form_name(form), form.extension_group))
        and _configuration_bins(
            _form_name(form), _semantic_type(_form_name(form), form.extension_group)
        )
        for form in _eligible_forms("rv64imc_zicsr_zifencei", "machine")
    )
    assert all(
        _semantic_type(_form_name(form), form.extension_group) != "integer-other"
        for profile in DEFAULT_PROFILES for mode in DEFAULT_PRIVILEGE_MODES
        for form in _eligible_forms(profile, mode)
    )

    signature = behavior_signature({
        "trace_pcs": ["0x1000", "0x1004", "0x1004"],
        "outcome": "exit", "exit_code": 0,
    })
    assert signature["complete"] and "exit:normal" in signature["dynamic_bins"]
    coverage = reference_coverage({
        "decoder_version": DECODER_VERSION, "lane": "rv64i/lp64",
        "isa_profile": "rv64i", "privilege_mode": "user",
        "instructions": [{
            "address": 0x1000, "encoding_id": "enc:32:rv64i:add",
            "type_id": "type:integer-register", "type_ids": ["type:integer-register"],
            "configuration_id": "cfg:integer:register",
            "configuration_ids": ["cfg:integer:register"],
        }],
    }, {"trace_pcs": ["0x1000", "0x2000"], "outcome": "exit", "exit_code": 0})
    assert coverage["status"] == "ok"
    assert coverage["ignored_pc_records"] == 1 and coverage["unknown_pc_records"] == 0
    assert coverage["metrics"]["ICov-encoding"]["eligible"] == 52
    assert coverage["metrics"]["CCov"]["status"] == "observed"
    csr_coverage = reference_coverage({
        "decoder_version": DECODER_VERSION, "lane": "rv64im_zicsr/lp64",
        "isa_profile": "rv64im_zicsr", "privilege_mode": "user",
        "instructions": [{
            "address": 0x1000, "encoding_id": "enc:32:rv64i:add",
            "type_id": "type:integer-register", "type_ids": ["type:integer-register"],
            "configuration_id": "cfg:integer:register",
            "configuration_ids": ["cfg:integer:register"],
        }],
    }, {"trace_pcs": [0x1000], "outcome": "exit"})
    assert csr_coverage["coverage_profile_id"] != coverage["coverage_profile_id"]
    assert csr_coverage["metrics"]["ICov-encoding"]["eligible"] == 71
    mismatch = reference_coverage({
        "decoder_version": DECODER_VERSION, "isa_profile": "rv64i",
        "privilege_mode": "user", "instructions": [{
            "address": 0x1000, "encoding_id": "enc:32:rv64zicsr:csrrw",
            "type_id": "type:system-csr", "type_ids": ["type:system-csr"],
            "configuration_id": "cfg:system:csr", "configuration_ids": ["cfg:system:csr"],
        }],
    }, {"trace_pcs": [0x1000], "outcome": "exit"})
    assert mismatch["status"] == "gap"
    assert reference_coverage({
        "decoder_version": DECODER_VERSION, "isa_profile": "rv64i",
        "instructions": [{"address": 0x1000, "encoding_id": "e", "type_id": "t",
                          "configuration_id": "c"}],
    }, {"trace_pcs": [0x2000], "outcome": "exit"})["status"] == "gap"
    return {
        "status": "ok", "decoder_version": DECODER_VERSION,
        "catalog_source": CATALOG_SOURCE,
        "rv64im_user_encoding_units": len(base["metrics"]["ICov-encoding"]),
        "rv64im_machine_encoding_units": len(machine["metrics"]["ICov-encoding"]),
        "rv64imc_zicsr_zifencei_machine_encoding_units": len(
            compressed["metrics"]["ICov-encoding"]),
        "encoding_registry_sha256": fragment["encoding"]["registry_sha256"],
        "type_registry_sha256": fragment["type"]["registry_sha256"],
        "configuration_registry_sha256": fragment["configuration"]["registry_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("registry", "signature", "elf", "self-check"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--lane", default="rv64i/lp64")
    parser.add_argument("--privilege-mode", choices=("user", "machine"), default="user")
    parser.add_argument("--observation", type=Path)
    args = parser.parse_args(argv)
    if args.command == "self-check":
        value = self_check()
    elif args.command == "registry":
        value = metric_registry_fragment()
    elif args.command == "signature":
        value = behavior_signature(json.loads(args.observation.read_text(encoding="utf-8")))
    else:
        value = decode_elf(args.elf, args.lane, privilege_mode=args.privilege_mode)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if value.get("status") in {None, "ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from functools import partial
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .direct_case import (
    OBSERVABLE_REGION,
    Observation,
    TestCase,
    TranslationEvidence,
    _runner_contract_gap_observation,
)
from ._translation_cell import (
    generation_normative_obligation_for_testcase,
    generation_realization_for_testcase,
)
from .direct_elf import build_elf, symbol_offsets_from_elf
from .execution_environment import (
    EM_X86_64,
    elf_machine,
    require_execution_plane,
)
from .paths import (
    SAIL_MEMORY_OVERRIDE_CONFIG_PATH,
    _reference_config_label,
    reference_config_identity,
)
from .riscv_encoding import (
    U64_MASK,
    VECTOR_VTYPE_LMUL_ENCODING,
    VECTOR_VTYPE_SEW_ENCODING,
)
from .rvgen.boundaries import branch_condition_holds
from .rvgen.semantic_machine import (
    SemanticTrap,
    SemanticUnsupported,
    execute_semantic_case,
)
from .reference_contract import REFERENCE_LOCK_CONFIG
from .spec_definedness import (
    enabled_extensions,
    isa_base_extensions,
    is_canonical_isa_profile,
    is_control_semantic_mnemonic_for_generation,
    compare_mask_requests_outcome_boundary,
    x_register_fp_extensions,
)
from .state_contract import state_contract_audit as _state_contract_audit
from ._util import canonical_digest, elf_writable_loads, sha256_file


RISCV_SAIL_SOURCE_CONFIG_RELATIVE_PATH = "config"
_RVGEN_VENDOR_REFERENCE_IDENTITY = canonical_digest({
    "backend": "rvgen-semantic",
    "contract": "rvgen-vendor-reference-v2",
    "forms": ("th_srri", "th_tst", "vt_maskc_root", "vt_maskcn_root"),
})
_RVGEN_SCALAR_REFERENCE_FORMS = frozenset({
    # Fast path for Direct's baseline scalar roots. The dispatcher routes
    # every other instruction form to the extended semantic machine below.
    "add", "addi", "addiw", "addw", "and", "andi", "auipc",
    "beq", "bge", "bgeu", "blt", "bltu", "bne", "ebreak", "ecall", "fence",
    "jal", "jalr", "lb", "lbu", "ld", "lh", "lhu", "lui", "lw", "lwu",
    "or", "ori", "sb", "sd", "sh", "sll", "slli", "slliw", "sllw",
    "slt", "slti", "sltiu", "sltu", "sra", "srai", "sraiw", "sraw",
    "srl", "srli", "srliw", "srlw", "sub", "subw", "sw", "xor", "xori",
})
_RVGEN_SCALAR_REFERENCE_IDENTITY = canonical_digest({
    "backend": "rvgen-semantic",
    "contract": "rvgen-scalar-reference-v4",
    "forms": tuple(sorted(_RVGEN_SCALAR_REFERENCE_FORMS)),
    "control_flow": ("jal", "jalr"),
})
_RVGEN_EXTENDED_REFERENCE_FAMILIES = (
    "I", "M", "A", "B", "C", "Zcmp", "Zcmop", "Zicond",
    "F", "D", "Q-bit-operations", "Q-finite-arithmetic-comparison-fma",
    "Q-finite-format-integer-conversions", "Q-correctly-rounded-sqrt",
    "scalar-fp-word-result-sign-extension",
    "Zfh", "Zfbfmin", "Zfa", "Zicsr", "Zicsr-access-legality", "Zk-scalar-crypto",
    "P-basic-packed", "P-packed-multiply-high", "P-narrowing-clip-fixed-ties-up",
    "P-vxsat-vcsr-vs",
    "Zacas-AMOCAS-D-RV32-pair", "Zacas-AMOCAS-Q-RV64-pair",
    "V", "Zvfh", "V-crypto", "Zvbb", "Zvkb", "RVV-fixed-point-vxrm",
    "RVV-width-conversions", "RVV-segmented-memory", "RVV-indexed-segment-stores",
    "RVV-RV32-ei64-index-illegal", "RVV-VS-off-illegal", "FPR-NaN-boxing",
    "scalar-FLEN-dependent-FPR-width",
    "RVV-same-width-finite-FMA", "RVV-widening-finite-FMA",
    "RVV-widening-finite-arithmetic", "RVV-ordered-fp-reductions",
    "RVV-reduction-zero-length-register-validation", "Scalar-FMA-sign-conventions",
    "Scalar-finite-fp-exception-flags",
    "Zfa-quiet-comparisons", "Scalar-nonfinite-ieee-semantics",
    "Q-nonfinite-ieee-semantics", "RVV-fault-only-first-loads",
    "Scalar-IEEE-format-and-integer-conversions", "Zfa-fcvtmod.w.d",
    "Zfinx-X-register-scalar-FP", "Zfinx-profile-aware-FCSR",
    "Zvfbfa-v0.9-vtype-altfmt", "canonical-NaN", "Zvqwdota8i-draft",
)
_RVGEN_EXTENDED_REFERENCE_IDENTITY = canonical_digest({
    "backend": "rvgen-semantic",
    "contract": "rvgen-extended-reference-v22",
    "families": _RVGEN_EXTENDED_REFERENCE_FAMILIES,
    "observer": (
        "gpr", "memory", "fpr_rawbits", "fflags", "frm",
        "vector_csrs", "vector_registers", "vector.mask", "vector.tail",
        "vxsat", "vxrm", "csr_values", "csr.address", "csr.privilege",
        "csr.warl.readback", "csr.reserved-fields",
    ),
})
_SAIL_CSRW_MISA_RAW = "  .word 0x30109073"
_SAIL_ENABLE_FS_RAW = "  .word 0x3000a073"
_SAIL_ENABLE_VS_RAW = "  .word 0x3000a073"
_SAIL_TRACE_INSTRUCTION_RE = re.compile(
    r"^\[(?P<index>[0-9]+)\]\s+\[[^\]]+\]:\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<word>[0-9a-fA-F]+)\)\s+(?P<body>.+)$"
)
_SAIL_TRACE_GPR_RE = re.compile(r"^x(?P<index>[0-9]+)\s+<-\s+0x(?P<value>[0-9a-fA-F]+)$")
_SAIL_TRACE_FPR_RE = re.compile(r"^f(?P<index>[0-9]+)\s+<-\s+0x(?P<value>[0-9a-fA-F]+)$")
_SAIL_TRACE_VECTOR_RE = re.compile(
    r"^v(?P<index>[0-9]+)(?:\[(?P<element>[0-9]+)\])?\s+<-\s+0x(?P<value>[0-9a-fA-F]+)$"
)
_SAIL_TRACE_CSR_RE = re.compile(
    r"^CSR\s+(?P<name>[a-zA-Z0-9_]+)\s+\(0x(?P<address>[0-9a-fA-F]+)\)\s+(?P<direction><-|->)\s+0x(?P<value>[0-9a-fA-F]+)$"
)
_SAIL_TRACE_MEM_WRITE_RE = re.compile(
    # AMO read-modify-write events are emitted as mem[RW,...] by the pinned
    # Sail trace producer; a writeback happens for both W and RW forms.
    r"^mem\[(?:W|RW),0x(?P<address>[0-9a-fA-F]+)\]\s+<-\s+0x(?P<value>[0-9a-fA-F]+)$"
)
_SAIL_TRAP_SIGNAL_BY_MCAUSE = {
    2: "SIGILL",
    4: "SIGBUS",
    5: "SIGSEGV",
    6: "SIGBUS",
    7: "SIGSEGV",
    12: "SIGSEGV",
    13: "SIGSEGV",
    15: "SIGSEGV",
}
_VECTOR_CSR_ADDRESSES = {
    "fflags": 0x001,
    "frm": 0x002,
    "fcsr": 0x003,
    "vstart": 0x008,
    "vxsat": 0x009,
    "vxrm": 0x00A,
    "vcsr": 0x00F,
    "vl": 0xC20,
    "vtype": 0xC21,
    "vlenb": 0xC22,
}
_TRAP_CSR_ADDRESSES = {
    "sepc": 0x141,
    "scause": 0x142,
    "stval": 0x143,
    "mepc": 0x341,
    "mcause": 0x342,
    "mtval": 0x343,
}
_STATE_CSR_ADDRESSES = {"mstatus": 0x300}
_VECTOR_VTYPE_RE = re.compile(
    r"^vector-vtype:sew(?P<sew>8|16|32|64|128)-(?P<lmul>mf8|mf4|mf2|m1|m2|m4|m8)$"
)


def _reference_facts(testcase):
    dataflow = getattr(testcase, "dataflow_meta", {})
    if not isinstance(dataflow, dict):
        return {
            "realization": None,
            "obligation": None,
            "semantic_point": None,
            "risk_contracts": None,
            "violations": (),
        }
    if dataflow.get("candidate_mutated"):
        return {
            "realization": None,
            "obligation": None,
            "semantic_point": None,
            "risk_contracts": None,
            "violations": (),
        }
    realization = dataflow.get("generation_realization")
    obligation = dataflow.get("normative_obligation")
    semantic_point = dataflow.get("semantic_point")
    raw_contracts = dataflow.get("risk_contracts")
    risk_contracts = (
        tuple(raw_contracts)
        if isinstance(raw_contracts, (list, tuple))
        and all(isinstance(item, dict) for item in raw_contracts)
        else None
    )
    properties = semantic_point.get("properties") if isinstance(semantic_point, dict) else None
    raw_violations = properties.get("violation_set") if isinstance(properties, dict) else None
    violations = (
        tuple(sorted({item for item in raw_violations.split(",") if item}))
        if isinstance(raw_violations, str) and raw_violations
        else ()
    )
    return {
        "realization": realization,
        "obligation": obligation,
        "semantic_point": semantic_point,
        "risk_contracts": risk_contracts,
        "violations": violations,
    }


def _expected_trap_mcause(facts):
    violations = set(facts["violations"])
    if "breakpoint" in violations:
        return 3
    if "environment-call" in violations:
        realization = facts.get("realization") if isinstance(facts, dict) else None
        return _environment_call_cause(realization)
    if "misaligned-atomic" in violations:
        properties = (facts["semantic_point"] or {}).get("properties")
        operation = str(properties.get("effect_operation") or "").lower() if isinstance(properties, dict) else ""
        operation = operation or str((facts["obligation"] or {}).get("operation") or "").lower()
        return 4 if operation.startswith("lr") else 6
    if "misaligned-memory" in violations:
        properties = (facts["semantic_point"] or {}).get("properties")
        operation = str(properties.get("effect_operation") or "").lower() if isinstance(properties, dict) else ""
        operation = operation or str((facts["obligation"] or {}).get("operation") or "").lower()
        return 4 if operation.startswith("load") else 6 if operation.startswith("store") else 2
    if "memory-access-fault" in violations:
        properties = (facts["semantic_point"] or {}).get("properties")
        operation = str(properties.get("effect_operation") or "").lower() if isinstance(properties, dict) else ""
        operation = operation or str((facts["obligation"] or {}).get("operation") or "").lower()
        return 5 if operation.startswith("load") else 7 if operation.startswith("store") else 5
    if "control-target-misaligned" in violations:
        return 0
    return 2 if violations else None


def _environment_call_cause(realization) -> int:
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


def _rvgen_expected_trap_observation(
    testcase: TestCase, executable_path: str | Path, form: str, facts: Mapping,
) -> Observation:
    """Materialize a legality witness as a comparable reference trap."""
    _, addresses = symbol_offsets_from_elf(Path(executable_path), ("test_start",))
    test_start = int(addresses["test_start"], 16)
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    risk_pc = test_start + int(risk.pc_offset)
    word = int.from_bytes(
        testcase.code_bytes[risk.byte_offset:risk.byte_offset + risk.byte_length],
        "little",
    )
    expected_cause = _expected_trap_mcause(facts)
    cause = 2 if expected_cause is None else int(expected_cause)
    signal = _SAIL_TRAP_SIGNAL_BY_MCAUSE.get(cause, "SIGILL")
    signal_code = {"SIGILL": 4, "SIGTRAP": 5, "SIGBUS": 7, "SIGSEGV": 11}.get(signal)
    identity = {
        "contract": "rvgen-extended-reference-v22",
        "backend": "rvgen-semantic",
        "identity_digest": _RVGEN_EXTENDED_REFERENCE_IDENTITY,
        "form": form,
        "mode": "legality-expected-trap",
    }
    return Observation(
        backend="rvgen-semantic", outcome="trap", exit_code=1,
        checkpoint_pc=None, gpr=tuple(testcase.initial_gpr), memory_delta={},
        memory_digest=None, signal=signal, signal_code=signal_code,
        fault_pc=risk_pc, fault_address=word, executed_pcs=(risk_pc,),
        instruction_count=1, translation_evidence=None,
        profile_id=testcase.isa_profile, input_id=testcase.input_id,
        raw_stdout="", raw_stderr="", binary_sha256=sha256_file(Path(executable_path)),
        tool_version="rvgen-semantic-extended-v22",
        extra_state={
            "reference_identity": identity,
            "observer_fields": [
                "executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval",
            ],
            "guest_trap": "delivered", "guest_cause": cause,
            "guest_epc": risk_pc, "guest_tval": word,
            "trap.cause": cause, "trap.epc": risk_pc, "trap.tval": word,
            "trap_observer": "rvgen-semantic-legality-contract",
            "rvgen_risk_execution": {"status": "observed", "risk_pcs": [risk_pc]},
        },
        contract_error=None,
    )


def _vector_boundary_facts(boundary):
    if not isinstance(boundary, str):
        return None
    prefix, _, value = boundary.partition(":")
    if prefix == "vector-vl":
        return ("vl", None, None, None) if value in {"zero", "one", "vlmax"} else None
    if prefix == "vector-vstart":
        if value in {"zero", "one"}:
            return "vstart", int(value == "one"), None, None
        if value.isascii() and value.isdecimal() and value == str(int(value)):
            return "vstart", int(value), None, None
        return None
    match = _VECTOR_VTYPE_RE.fullmatch(boundary)
    if match is not None:
        sew = int(match.group("sew"))
        lmul = match.group("lmul")
        if sew not in VECTOR_VTYPE_SEW_ENCODING:
            return None
        return "vtype", None, VECTOR_VTYPE_SEW_ENCODING[sew], VECTOR_VTYPE_LMUL_ENCODING[lmul]
    axis = (
        "tail" if prefix == "vector-tail" and value in {"agnostic", "undisturbed"}
        else "mask" if prefix == "vector-mask" and value in {"all-off", "all-on", "vm:0", "vm:1"}
        else None
    )
    return (axis, None, None, None) if axis else None


def _locked_sail_candidate() -> dict[str, object] | None:
    candidates = (
        (
            "binary-release",
            REFERENCE_LOCK_CONFIG.binary_release_root / "bin" / "sail_riscv_sim",
            REFERENCE_LOCK_CONFIG.binary_release_root / "share" / "sail-riscv" / "config",
        ),
        (
            "source-build",
            RISCV_SAIL_MODEL_LOCK.checkout_path / "build" / "c_emulator" / "sail_riscv_sim",
            RISCV_SAIL_MODEL_LOCK.checkout_path / RISCV_SAIL_SOURCE_CONFIG_RELATIVE_PATH,
        ),
    )
    for kind, path, config_root in candidates:
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        try:
            if elf_machine(path) != EM_X86_64:
                continue
        except RuntimeError:
            continue
        return {
            "candidate_kind": kind,
            "path": str(path),
            "config_root": str(config_root),
            "sha256": sha256_file(path),
        }
    return None


def _misa_value_for_profile(isa_profile: str, *, include_user: bool = False) -> int | None:
    profile = str(isa_profile or "").lower()
    if profile.startswith("rv32"):
        value = 1 << 30
    elif profile.startswith("rv64"):
        value = 2 << 62
    else:
        return None
    base_extensions = isa_base_extensions(profile)
    letters = {letter.upper() for letter in base_extensions if letter.isalpha()}
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
    if include_user:
        value |= 1 << 20
    return value


def _sail_profile_prologue_lines(
    isa_profile: str, *, user_mode: bool = False,
) -> tuple[str, ...]:
    lines: list[str] = []
    misa_value = _misa_value_for_profile(isa_profile, include_user=user_mode)
    if misa_value is not None:
        lines.extend((f"  li x1, 0x{misa_value:x}", _SAIL_CSRW_MISA_RAW))
    if {"f", "d"} & enabled_extensions(isa_profile):
        lines.extend(("  lui x1, 0x6", _SAIL_ENABLE_FS_RAW))
    if _sail_vector_config(isa_profile) is not None:
        lines.extend(("  li x1, 0x600", _SAIL_ENABLE_VS_RAW))
    if user_mode:
        lines.extend((
            "  li x1, -1",
            "  csrw pmpaddr0, x1",
            "  li x1, 0xf",
            "  csrw pmpcfg0, x1",
        ))
    return tuple(lines)


def _sail_profile_plan(
    candidate: dict[str, object], isa_profile: str, *, form: str | None = None,
    user_mode: bool = False,
) -> dict[str, object]:
    profile = str(isa_profile or "").lower()
    if not is_canonical_isa_profile(profile):
        return {
            "supported": False,
            "reason": f"unsupported-isa-profile:{isa_profile}",
        }
    from .sail_definedness_oracle import sail_profile_support
    supported, reason, _identity = sail_profile_support(profile)
    if not supported:
        return {"supported": False, "reason": reason}
    if x_register_fp_extensions(profile):
        return {
            "supported": False,
            "reason": f"unsupported-sail-floating-register-model:{isa_profile}",
        }
    base_extensions = isa_base_extensions(profile)
    if "h" in base_extensions:
        return {
            "supported": False,
            "reason": f"unsupported-sail-h-extension:{isa_profile}",
        }
    if "q" in base_extensions:
        return {
            "supported": False,
            "reason": f"unsupported-sail-q-precision-sample-config:{isa_profile}",
        }
    config_args: list[str] = []
    config_files: list[str] = []
    prologue_lines = _sail_profile_prologue_lines(profile, user_mode=user_mode)
    vector_config = _sail_vector_config(profile)
    if vector_config is not None or {"f", "d"} & enabled_extensions(profile):
        config_root = Path(str(candidate["config_root"]))
        if vector_config is not None:
            sample_name = (
                "rv32d_v128_e64.json" if profile.startswith("rv32") else "rv64d_v128_e64.json"
            )
        else:
            sample_name = "rv32d_v64_e64.json" if profile.startswith("rv32") else "rv64d_v64_e64.json"
        sample_path = config_root / sample_name
        if not sample_path.is_file():
            return {
                "supported": False,
                "reason": f"missing-sail-sample-config:{sample_name}",
            }
        config_args.extend(("--config", str(sample_path)))
        config_files.append(str(sample_path))
    elif profile.startswith("rv32"):
        # The release simulator defaults to RV64.  A plain RV32/I/E/Z* ELF
        # has no sample config to select, so bind the execution plane
        # explicitly; otherwise Sail exits before producing a trace.
        config_args.append("--rv32")
    if not SAIL_MEMORY_OVERRIDE_CONFIG_PATH.is_file():
        return {
            "supported": False,
            "reason": f"missing-sail-memory-override:{SAIL_MEMORY_OVERRIDE_CONFIG_PATH}",
        }
    config_args.extend(("--config-override", str(SAIL_MEMORY_OVERRIDE_CONFIG_PATH)))
    config_files.append(str(SAIL_MEMORY_OVERRIDE_CONFIG_PATH))
    extension_override = _sail_profile_extension_override(profile, form=form)
    if extension_override is not None:
        digest = canonical_digest(extension_override)[:16]
        override_path = Path(tempfile.gettempdir()) / f"sail-rvgen-profile-{digest}.json"
        if not override_path.is_file():
            override_path.write_text(
                json.dumps(extension_override, indent=2) + "\n",
                encoding="utf-8",
            )
        config_args.extend(("--config-override", str(override_path)))
        config_files.append(str(override_path))
    return {
        "supported": True,
        "reason": "supported",
        "config_args": tuple(config_args),
        "config_files": tuple(config_files),
        "prologue_lines": prologue_lines,
    }


_SAIL_SCHEMA_BASE_KEYS = {
    "a": "A",
    "b": "B",
    "d": "D",
    "f": "F",
    "m": "M",
    "s": "S",
    "u": "U",
    "v": "V",
}


def _sail_vector_config(isa_profile: str) -> dict[str, object] | None:
    """Map V/Zve profile facts to Sail's single V configuration object."""

    enabled = enabled_extensions(str(isa_profile or "").lower())
    if "v" in enabled:
        support_level = "Full"
        minimum_elen = 6
        minimum_vlen = 8
    elif "zve64d" in enabled:
        support_level = "Float_double"
        minimum_elen = minimum_vlen = 6
    elif "zve64f" in enabled:
        support_level = "Float_single"
        minimum_elen = minimum_vlen = 6
    elif "zve64x" in enabled:
        support_level = "Integer"
        minimum_elen = minimum_vlen = 6
    elif "zve32f" in enabled:
        support_level = "Float_single"
        minimum_elen = minimum_vlen = 5
    elif "zve32x" in enabled:
        support_level = "Integer"
        minimum_elen = minimum_vlen = 5
    else:
        return None

    # Zvl is a minimum VLEN declaration; choose the largest declared minimum
    # while keeping the config valid for the selected embedded profile.
    vlen_exp = minimum_vlen
    for token in enabled:
        if token.startswith("zvl") and token.endswith("b"):
            try:
                bits = int(token[3:-1])
                if bits > 0:
                    vlen_exp = max(vlen_exp, (bits - 1).bit_length())
            except ValueError:
                continue
    return {
        "support_level": support_level,
        "vlen_exp": vlen_exp,
        "elen_exp": minimum_elen,
        "vl_use_ceil": False,
    }


def _sail_profile_extension_override(
    isa_profile: str, *, form: str | None = None,
) -> dict[str, object] | None:
    """Enable the profile's declared extensions on the Sail base config.

    The pinned Sail base/default configuration is narrower than many RVGEN
    profiles (for example C/Zca is not enabled by default), so a legality row
    would decode its risk encoding as illegal purely because of the Sail
    machine, not the profile.  This override only *enables* the declared
    extensions (schema key names), never disables: disabling breaks Sail's
    dependent-extension validation, and an over-supporting Sail machine keeps
    every expectation fail-closed (extension-disabled rows become reference
    gaps instead of false verdicts).  Counter extensions are explicit because
    the sample config enables them by default.
    """
    profile = str(isa_profile or "").lower()
    declared = enabled_extensions(profile)
    rv32 = profile.startswith("rv32")
    extension_values: dict[str, object] = {}
    for ext in declared:
        if ext in {"i", "e", "h", "v"} or ext.startswith(("zve", "zvl")):
            continue
        if ext == "c":
            extension_values["Zca"] = {"supported": True}
            if rv32:
                if "f" in declared:
                    extension_values["Zcf"] = {"supported": True}
            elif "d" in declared or "f" in declared:
                extension_values["Zcd"] = {"supported": True}
        elif ext in _SAIL_SCHEMA_BASE_KEYS:
            extension_values[_SAIL_SCHEMA_BASE_KEYS[ext]] = {"supported": True}
        else:
            extension_values[ext.capitalize()] = {"supported": True}
    vector_config = _sail_vector_config(profile)
    if vector_config is not None:
        extension_values["V"] = {"supported": True, **vector_config}
    for ext in ("zicntr", "zihpm"):
        extension_values[ext.capitalize()] = {"supported": ext in declared}
    if str(form or "").startswith("cbo_"):
        enabled = "zicbo" in declared
        for ext in ("zicbom", "zicbop", "zicboz"):
            extension_values[ext.capitalize()] = {"supported": enabled}
    if not extension_values:
        return None
    return {"extensions": dict(sorted(extension_values.items()))}


def _sail_tool_version(simulator_path: Path) -> str | None:
    try:
        proc = subprocess.run(
            [str(simulator_path), "--version"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return text.splitlines()[0] if text else None


RISCV_SAIL_MODEL_LOCK = REFERENCE_LOCK_CONFIG.sail


_sail_gap_observation = partial(_runner_contract_gap_observation, "sail-riscv")


def _rvgen_scalar_reference_observation(
    testcase: TestCase, executable_path: str | Path, form: str,
) -> Observation:
    """用小型、确定性的 RV64I 模型给 Direct MCMC 提供独立 reference。"""
    _, addresses = symbol_offsets_from_elf(
        Path(executable_path), ("test_start", "exit_checkpoint"),
    )
    test_start = int(addresses["test_start"], 16)
    checkpoint = int(addresses["exit_checkpoint"], 16)
    xlen = 32 if str(testcase.isa_profile).lower().startswith("rv32") else 64
    mask = (1 << xlen) - 1
    sign_bit = 1 << (xlen - 1)
    registers = [int(value) & mask for value in testcase.initial_gpr]
    registers[0] = 0
    regions = {
        region.region_id: bytearray(region.data)
        for region in testcase.initial_memory_regions
    }

    def register(index: int | None) -> int:
        if index is None or not 0 <= int(index) < len(registers):
            raise ValueError(f"rvgen-semantic missing register for {form}")
        return registers[int(index)]

    def signed(value: int, width: int = xlen) -> int:
        value &= (1 << width) - 1
        return value - (1 << width) if value & (1 << (width - 1)) else value

    def immediate(item: object, bits: int = 12) -> int:
        value = getattr(item, "immediate", None)
        if not isinstance(value, int):
            raise ValueError(f"rvgen-semantic missing immediate for {form}")
        if value < 0:
            return value
        return value - (1 << bits) if value & (1 << (bits - 1)) else value

    def upper_immediate(item: object) -> int:
        value = getattr(item, "immediate", None)
        if isinstance(value, int):
            return value
        fields = dict(getattr(item, "operand_fields", ()) or ())
        raw = fields.get("imm20")
        if isinstance(raw, int):
            return signed(raw, 20) << 12
        raise ValueError(f"rvgen-semantic missing U-immediate for {form}")

    def write_register(index: int | None, value: int, width: int = xlen) -> None:
        if index is not None and int(index) != 0:
            value &= (1 << width) - 1
            registers[int(index)] = value
        registers[0] = 0

    def memory_read(address: int, width: int) -> int:
        for region in testcase.initial_memory_regions:
            offset = address - int(region.address)
            if 0 <= offset <= len(region.data) - width:
                return int.from_bytes(regions[region.region_id][offset:offset + width], "little")
        raise ValueError("rvgen-semantic memory read is outside testcase memory")

    def memory_write(address: int, width: int, value: int) -> None:
        for region in testcase.initial_memory_regions:
            offset = address - int(region.address)
            if 0 <= offset <= len(region.data) - width:
                regions[region.region_id][offset:offset + width] = (
                    int(value) & ((1 << (width * 8)) - 1)
                ).to_bytes(width, "little")
                return
        raise ValueError("rvgen-semantic memory write is outside testcase memory")

    reference_facts = _reference_facts(testcase)
    memory_trap_violations = {
        "memory-access-fault", "misaligned-memory", "misaligned-atomic",
    }

    def memory_trap_observation(item: object, address: int) -> Observation | None:
        """Close an explicitly generated memory-legality row as a trap.

        The Direct root pool intentionally includes access-fault and
        misalignment witnesses.  They are observations, not errors in the
        semantic model; only an unplanned out-of-bounds access remains a
        reference gap.
        """
        violations = set(reference_facts.get("violations") or ())
        if not violations & memory_trap_violations:
            return None
        cause = _expected_trap_mcause(reference_facts)
        if cause is None:
            return None
        risk_pc = test_start + int(item.pc_offset)
        signal = _SAIL_TRAP_SIGNAL_BY_MCAUSE.get(cause)
        signal_code = {"SIGILL": 4, "SIGBUS": 7, "SIGSEGV": 11}.get(signal)
        identity = {
            "contract": "rvgen-scalar-reference-identity-v4",
            "backend": "rvgen-semantic",
            "identity_digest": _RVGEN_SCALAR_REFERENCE_IDENTITY,
            "form": form,
        }
        return Observation(
            backend="rvgen-semantic", outcome="trap", exit_code=1,
            checkpoint_pc=None, gpr=tuple(registers), memory_delta={},
            memory_digest=None, signal=signal, signal_code=signal_code,
            fault_pc=risk_pc, fault_address=address,
            executed_pcs=tuple(test_start + current.pc_offset for current in executed_items),
            instruction_count=len(executed_items), translation_evidence=None,
            profile_id=testcase.isa_profile, input_id=testcase.input_id,
            raw_stdout="", raw_stderr="", binary_sha256=sha256_file(Path(executable_path)),
            tool_version="rvgen-semantic-scalar-v4",
            extra_state={
                "reference_identity": identity,
                "observer_fields": [
                    "executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval",
                ],
                "guest_trap": "delivered", "guest_cause": cause,
                "guest_epc": risk_pc, "guest_tval": address,
                "trap.cause": cause, "trap.epc": risk_pc, "trap.tval": address,
                "trap_observer": "rvgen-semantic",
                "rvgen_risk_execution": {
                    "status": "observed", "risk_pcs": [
                        test_start + current.pc_offset for current in executed_items
                        if "risk" in current.tags
                    ],
                },
            },
            contract_error=None,
        )

    items = tuple(testcase.instruction_meta)
    by_offset = {int(item.pc_offset): index for index, item in enumerate(items)}
    end_offset = sum(int(item.byte_length) for item in items)
    executed_items: list[object] = []
    control_trace: dict[str, object] | None = None
    index = 0
    visited: set[int] = set()
    while index < len(items):
        if index in visited:
            raise ValueError("rvgen-semantic control-flow loop")
        visited.add(index)
        item = items[index]
        executed_items.append(item)
        mnemonic = str(item.mnemonic).replace("_", ".").lower()
        left = register(getattr(item, "rs1", None)) if getattr(item, "rs1", None) is not None else 0
        right = register(getattr(item, "rs2", None)) if getattr(item, "rs2", None) is not None else 0
        result = None
        if mnemonic in {"add", "addi"}:
            result = left + (immediate(item) if mnemonic == "addi" else right)
        elif mnemonic in {"addiw"}:
            result = signed(left + immediate(item), 32)
            write_register(item.rd, result)
            index += 1
            continue
        elif mnemonic in {"addw", "subw"}:
            result = left + right if mnemonic == "addw" else left - right
            write_register(item.rd, signed(result, 32))
            index += 1
            continue
        elif mnemonic in {"sllw", "srlw", "sraw"}:
            shift = right & 0x1F
            word = left & 0xFFFFFFFF
            if mnemonic == "sllw":
                result = word << shift
            elif mnemonic == "srlw":
                result = word >> shift
            else:
                result = signed(word, 32) >> shift
            write_register(item.rd, signed(result, 32))
            index += 1
            continue
        elif mnemonic in {"srliw", "sraiw"}:
            shift = immediate(item, 5) & 0x1F
            word = left & 0xFFFFFFFF
            result = word >> shift if mnemonic == "srliw" else signed(word, 32) >> shift
            write_register(item.rd, signed(result, 32))
            index += 1
            continue
        elif mnemonic in {"sub"}:
            result = left - right
        elif mnemonic in {"and", "andi"}:
            result = left & (immediate(item) if mnemonic == "andi" else right)
        elif mnemonic in {"or", "ori"}:
            result = left | (immediate(item) if mnemonic == "ori" else right)
        elif mnemonic in {"xor", "xori"}:
            result = left ^ (immediate(item) if mnemonic == "xori" else right)
        elif mnemonic in {"sll", "slli"}:
            shift = immediate(item, 6) if mnemonic == "slli" else right
            result = left << (shift & (xlen - 1))
        elif mnemonic in {"slliw"}:
            result = (left & 0xFFFFFFFF) << (immediate(item, 5) & 31)
            write_register(item.rd, signed(result, 32))
            index += 1
            continue
        elif mnemonic in {"srl", "srli"}:
            shift = immediate(item, 6) if mnemonic == "srli" else right
            result = left >> (shift & (xlen - 1))
        elif mnemonic in {"sra", "srai"}:
            shift = immediate(item, 6) if mnemonic == "srai" else right
            result = signed(left) >> (shift & (xlen - 1))
        elif mnemonic in {"slt", "slti"}:
            result = int(signed(left) < signed(immediate(item) if mnemonic == "slti" else right))
        elif mnemonic in {"sltu", "sltiu"}:
            result = int(left < (immediate(item) & mask) if mnemonic == "sltiu" else left < right)
        elif mnemonic == "lui":
            result = upper_immediate(item)
        elif mnemonic == "auipc":
            result = int(testcase.code_address) + int(item.pc_offset) + upper_immediate(item)
        elif mnemonic in {"mv", "c.mv"}:
            result = right if mnemonic == "c.mv" else left
        elif mnemonic in {"nop", "c.nop"}:
            index += 1
            continue
        elif mnemonic in {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"}:
            width = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2,
                     "lw": 4, "lwu": 4, "ld": 8}[mnemonic]
            address = (left + int(item.immediate or 0)) & mask
            if address % width:
                trap = memory_trap_observation(item, address)
                if trap is not None:
                    return trap
            try:
                loaded = memory_read(address, width)
            except ValueError:
                trap = memory_trap_observation(item, address)
                if trap is not None:
                    return trap
                raise
            result = signed(loaded, width * 8) if mnemonic in {"lb", "lh", "lw"} else loaded
        elif mnemonic in {"sb", "sh", "sw", "sd"}:
            width = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}[mnemonic]
            address = (left + int(item.immediate or 0)) & mask
            if address % width:
                trap = memory_trap_observation(item, address)
                if trap is not None:
                    return trap
            try:
                memory_write(address, width, right)
            except ValueError:
                trap = memory_trap_observation(item, address)
                if trap is not None:
                    return trap
                raise
            index += 1
            continue
        elif mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}:
            operation = mnemonic
            if mnemonic in {"beqz", "bnez"}:
                operation = "beq" if mnemonic == "beqz" else "bne"
                right = 0
            taken = branch_condition_holds(left, right, operation, xlen=xlen)
            target_offset = int(item.pc_offset) + int(item.immediate or 0)
            target_index = by_offset.get(target_offset)
            if taken is None or (taken and target_index is None and target_offset != end_offset):
                raise ValueError(f"rvgen-semantic invalid branch target: {mnemonic}")
            control_trace = {
                "branch_outcome": "taken" if taken else "not-taken",
                "after_pc": test_start + (
                    target_offset if taken else int(item.pc_offset) + int(item.byte_length)
                ),
                "control.target": test_start + target_offset,
            }
            index = (target_index if target_index is not None else len(items)) if taken else index + 1
            continue
        elif mnemonic in {"jal", "j"}:
            target_offset = int(item.pc_offset) + int(item.immediate or 0)
            target_index = by_offset.get(target_offset)
            if target_index is None and target_offset != end_offset:
                raise ValueError(f"rvgen-semantic invalid jump target: {mnemonic}")
            link = int(testcase.code_address) + int(item.pc_offset) + int(item.byte_length)
            write_register(getattr(item, "rd", None) if mnemonic == "jal" else 0, link)
            control_trace = {
                "after_pc": test_start + target_offset,
                "control.target": test_start + target_offset,
                "link": link,
            }
            index = target_index if target_index is not None else len(items)
            continue
        elif mnemonic in {"jalr", "jr", "ret", "c.jalr", "c.jr"}:
            target_address = (
                left + int(item.immediate or 0)
            ) & mask
            target_offset = (target_address & ~1) - int(testcase.code_address)
            target_index = by_offset.get(target_offset)
            if target_index is None and target_offset != end_offset:
                raise ValueError(f"rvgen-semantic invalid indirect jump target: {mnemonic}")
            link = int(testcase.code_address) + int(item.pc_offset) + int(item.byte_length)
            link_register = (
                1 if mnemonic == "c.jalr"
                else 0 if mnemonic in {"jr", "ret", "c.jr"}
                else getattr(item, "rd", None)
            )
            write_register(link_register, link)
            control_trace = {
                "after_pc": test_start + target_offset,
                "control.target": test_start + target_offset,
                "link": link,
            }
            index = target_index if target_index is not None else len(items)
            continue
        elif mnemonic == "fence":
            # Fence has no architectural value in the isolated single-case
            # memory model; execution itself is the observable fact.
            control_trace = {
                "after_pc": test_start + int(item.pc_offset) + int(item.byte_length),
            }
            index += 1
            continue
        else:
            raise ValueError(f"rvgen-semantic unsupported scalar instruction: {mnemonic}")
        write_register(getattr(item, "rd", None), int(result))
        index += 1

    memory = {
        region_id: bytes(data).hex()
        for region_id, data in regions.items()
    }
    risk_pcs = tuple(
        test_start + item.pc_offset
        for item in executed_items if "risk" in item.tags
    )
    identity = {
        "contract": "rvgen-scalar-reference-identity-v4",
        "backend": "rvgen-semantic",
        "identity_digest": _RVGEN_SCALAR_REFERENCE_IDENTITY,
        "form": form,
    }
    test_memory = memory.get("test-memory", "")
    observer_fields = ["executed_pcs", "outcome", "gpr", "memory.test-memory"]
    extra_state = {
        "reference_identity": identity,
        "observer_fields": observer_fields,
        "rvgen_risk_execution": {"status": "observed", "risk_pcs": list(risk_pcs)},
    }
    if control_trace is not None:
        extra_state.update(control_trace)
        observer_fields.extend(
            field for field in control_trace if field not in observer_fields
        )
    return Observation(
        backend="rvgen-semantic", outcome="normal", exit_code=0,
        checkpoint_pc=checkpoint, gpr=tuple(registers), memory_delta=memory,
        memory_digest=hashlib.sha256(bytes.fromhex(test_memory)).hexdigest()
        if test_memory else None,
        signal=None, signal_code=None, fault_pc=None, fault_address=None,
        executed_pcs=tuple(test_start + item.pc_offset for item in executed_items),
        instruction_count=len(executed_items), translation_evidence=None,
        profile_id=testcase.isa_profile, input_id=testcase.input_id,
        raw_stdout="", raw_stderr="", binary_sha256=sha256_file(Path(executable_path)),
        tool_version="rvgen-semantic-scalar-v4",
        extra_state=extra_state,
        contract_error=None,
    )


def _rvgen_extended_reference_observation(
    testcase: TestCase, executable_path: str | Path, form: str,
) -> Observation:
    """Run the expanded local semantic machine for non-RV64I forms.

    The old scalar interpreter remains the compatibility path for the
    original RV64I contract.  This adapter owns the wider instruction-family
    contract and deliberately records the machine identity so a result cannot
    be mistaken for a Sail/native observation.
    """

    _, addresses = symbol_offsets_from_elf(
        Path(executable_path), ("test_start", "exit_checkpoint"),
    )
    test_start = int(addresses["test_start"], 16)
    checkpoint = int(addresses["exit_checkpoint"], 16)

    def reference_gap(reason: str, *, executed_offsets: tuple[int, ...] = ()) -> Observation:
        risk_pcs = tuple(
            test_start + int(item.pc_offset)
            for item in testcase.instruction_meta if "risk" in item.tags
        )
        executed_pcs = tuple(test_start + int(offset) for offset in executed_offsets)
        return _runner_contract_gap_observation(
            "rvgen-semantic", reason,
            binary_sha256=sha256_file(Path(executable_path)),
            profile_id=testcase.isa_profile,
            input_id=testcase.input_id,
            executed_pcs=executed_pcs,
            instruction_count=len(executed_offsets) if executed_offsets else None,
            tool_version="rvgen-semantic-extended-v22",
            extra_state={
                "reference_identity": {
                    "contract": "rvgen-extended-reference-v22",
                    "backend": "rvgen-semantic",
                    "identity_digest": _RVGEN_EXTENDED_REFERENCE_IDENTITY,
                    "form": form,
                },
                "observer_fields": ["outcome"],
                "semantic_machine": "reference-gap",
                "rvgen_risk_execution": {
                    "status": "observed" if any(pc in executed_pcs for pc in risk_pcs) else "unobserved",
                    "risk_pcs": list(risk_pcs),
                },
            },
        )

    def has_compared_fflags_memory_sink() -> bool:
        if OBSERVABLE_REGION not in testcase.compare_mask.memory_region_ids:
            return False
        items = tuple(testcase.instruction_meta)
        risk_index = next(
            (index for index, item in enumerate(items) if "risk" in item.tags),
            None,
        )
        if risk_index is None:
            return False
        expected = set(testcase.dataflow_meta.get("expected_executed_instruction_ids", ()))
        for index in range(risk_index + 1, len(items) - 1):
            item, sink = items[index], items[index + 1]
            if expected and (item.instruction_id not in expected or sink.instruction_id not in expected):
                continue
            fields = dict(item.operand_fields)
            sink_fields = dict(sink.operand_fields)
            if (
                item.mnemonic == "csrrs"
                and fields.get("csr") == 0x001
                and sink.mnemonic in {"sw", "sd"}
                and "observable" in sink.tags
                and sink_fields.get("rs2") == fields.get("rd")
            ):
                return True
        return False

    # Encoding-only and state-ready rows intentionally observe only
    # execution/outcome.  They do not make a value claim, so a normal
    # architectural completion is the useful reference baseline; an actual
    # illegal-instruction result from a target remains visible as a target
    # observation.  ``state-ready`` is the generator's explicit no-sink
    # disposition (for example a one-instruction P/Z* witness), not a hidden
    # ISA allow-list.  Keeping it at the observation boundary lets open
    # generation reach the target while reserving value semantics for rows
    # that actually have a state sink.
    dataflow = testcase.dataflow_meta
    facts = (
        dataflow.get("realized_facts", {})
        if not dataflow.get("candidate_mutated") else {}
    )
    reference_facts = _reference_facts(testcase)
    if isinstance(facts, Mapping) and facts.get("lane") == "legality-expected-trap":
        return _rvgen_expected_trap_observation(
            testcase, executable_path, form, reference_facts,
        )
    violations = set(
        str(item) for item in str(
            (dataflow.get("semantic_point", {}) or {}).get("properties", {}).get("violation_set", "")
        ).split(",") if item
    ) if isinstance(dataflow, Mapping) and not dataflow.get("candidate_mutated") else set()
    if (
        isinstance(facts, Mapping)
        and facts.get("disposition") in {"encoding-only", "state-ready"}
        and (
            facts.get("disposition") == "encoding-only"
            or not testcase.dataflow_meta.get("state_observer_keys")
        )
        and facts.get("lane") != "legality-expected-trap"
        and not violations
    ):
        memory = {
            region.region_id: bytes(region.data).hex()
            for region in testcase.initial_memory_regions
        }
        risk_pcs = tuple(
            test_start + int(item.pc_offset)
            for item in testcase.instruction_meta if "risk" in item.tags
        )
        identity = {
            "contract": "rvgen-extended-reference-v22",
            "backend": "rvgen-semantic",
            "identity_digest": _RVGEN_EXTENDED_REFERENCE_IDENTITY,
            "form": form,
            "mode": str(facts.get("disposition")),
        }
        return Observation(
            backend="rvgen-semantic", outcome="normal", exit_code=0,
            checkpoint_pc=checkpoint, gpr=tuple(testcase.initial_gpr),
            memory_delta=memory,
            memory_digest=hashlib.sha256(
                bytes.fromhex(memory.get("test-memory", ""))
            ).hexdigest() if memory.get("test-memory") else None,
            signal=None, signal_code=None, fault_pc=None, fault_address=None,
            executed_pcs=tuple(test_start + int(item.pc_offset) for item in testcase.instruction_meta),
            instruction_count=len(testcase.instruction_meta),
            translation_evidence=None, profile_id=testcase.isa_profile,
            input_id=testcase.input_id, raw_stdout="", raw_stderr="",
            binary_sha256=sha256_file(Path(executable_path)),
            tool_version="rvgen-semantic-extended-v22",
            extra_state={
                "reference_identity": identity,
                "semantic_machine": f"{facts.get('disposition')}-observation",
                "observer_fields": ["outcome"],
                "rvgen_risk_execution": {"status": "observed", "risk_pcs": list(risk_pcs)},
            },
            contract_error=None,
        )
    try:
        if (
            isinstance(facts, Mapping)
            and facts.get("lane") == "legality-expected-trap"
            and facts.get("disposition") == "definedness-outcome"
        ):
            risk_item = next(
                (item for item in testcase.instruction_meta if "risk" in item.tags),
                testcase.instruction_meta[0],
            )
            raise SemanticTrap(2, int(risk_item.pc_offset))
        result = execute_semantic_case(testcase)
    except SemanticUnsupported as error:
        return reference_gap(f"rvgen-semantic unsupported form: {form}: {error}")
    except SemanticTrap as trap:
        identity = {
            "contract": "rvgen-extended-reference-v22",
            "backend": "rvgen-semantic",
            "identity_digest": _RVGEN_EXTENDED_REFERENCE_IDENTITY,
            "form": form,
        }
        trap_pc = test_start + int(trap.pc_offset)
        cause_signal = _SAIL_TRAP_SIGNAL_BY_MCAUSE.get(int(trap.cause), "SIGILL")
        return Observation(
            backend="rvgen-semantic", outcome="trap", exit_code=1,
            checkpoint_pc=None, gpr=tuple(getattr(testcase, "initial_gpr", (0,) * 32)),
            memory_delta={}, memory_digest=None,
            signal=cause_signal, signal_code={"SIGILL": 4, "SIGTRAP": 5, "SIGSEGV": 11}.get(cause_signal),
            fault_pc=trap_pc, fault_address=int(trap.tval),
            executed_pcs=(trap_pc,), instruction_count=1,
            translation_evidence=None, profile_id=testcase.isa_profile,
            input_id=testcase.input_id, raw_stdout="", raw_stderr="",
            binary_sha256=sha256_file(Path(executable_path)),
            tool_version="rvgen-semantic-extended-v22",
            extra_state={
                "reference_identity": identity,
                "observer_fields": ["executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval"],
                "guest_trap": "delivered", "guest_cause": int(trap.cause),
                "guest_epc": trap_pc, "guest_tval": int(trap.tval),
                "trap.cause": int(trap.cause), "trap.epc": trap_pc, "trap.tval": int(trap.tval),
                "trap_observer": "rvgen-semantic-extended",
                "rvgen_risk_execution": {"status": "observed", "risk_pcs": [trap_pc]},
            },
            contract_error=None,
        )

    if (
        str(result.extra_state.get("fflags_observer", "")).startswith("gap:")
        and has_compared_fflags_memory_sink()
    ):
        return reference_gap(
            "rvgen-semantic fflags observer is incomplete for the compared memory sink",
            executed_offsets=result.executed_offsets,
        )

    executed_pcs = tuple(test_start + int(offset) for offset in result.executed_offsets)
    risk_pcs = tuple(
        test_start + int(item.pc_offset)
        for item in testcase.instruction_meta
        if "risk" in item.tags
    )
    extra_state = dict(result.extra_state)
    identity = {
        "contract": "rvgen-extended-reference-v22",
        "backend": "rvgen-semantic",
        "identity_digest": _RVGEN_EXTENDED_REFERENCE_IDENTITY,
        "form": form,
        "families": list(_RVGEN_EXTENDED_REFERENCE_FAMILIES),
    }
    extra_state.update({
        "reference_identity": identity,
        "rvgen_risk_execution": {"status": "observed", "risk_pcs": list(risk_pcs)},
        "semantic_machine": "rvgen-extended-v22",
    })
    observer_fields = list(extra_state.get("observer_fields", ()))
    for field in (
        "vxsat", "vxrm", "fflags", "frm", "csr.mstatus.fs",
        "vector.mask", "vector.tail", "csr.address", "csr.privilege",
        "csr.warl.readback", "csr.reserved-fields",
    ):
        if field in extra_state and field not in observer_fields:
            observer_fields.append(field)
    extra_state["observer_fields"] = observer_fields
    memory_hex = result.memory.get("test-memory", "")
    return Observation(
        backend="rvgen-semantic", outcome=result.outcome, exit_code=0,
        checkpoint_pc=checkpoint, gpr=tuple(result.gpr),
        memory_delta=dict(result.memory),
        memory_digest=hashlib.sha256(bytes.fromhex(memory_hex)).hexdigest()
        if memory_hex else None,
        signal=None, signal_code=None, fault_pc=None, fault_address=None,
        executed_pcs=executed_pcs, instruction_count=len(executed_pcs),
        translation_evidence=None, profile_id=testcase.isa_profile,
        input_id=testcase.input_id, raw_stdout="", raw_stderr="",
        binary_sha256=sha256_file(Path(executable_path)),
        tool_version="rvgen-semantic-extended-v22",
        extra_state=extra_state, contract_error=None,
    )


def rvgen_vendor_reference_observation(
    testcase: TestCase, executable_path: str | Path,
) -> Observation:
    """独立计算 RVGEN scalar 与 encoding 见证；不依赖 Sail/QEMU。"""
    risk = next(item for item in testcase.instruction_meta if "risk" in item.tags)
    form = str(getattr(risk, "mnemonic", "")).replace(".", "_")
    dataflow_meta = getattr(testcase, "dataflow_meta", {})
    if (isinstance(dataflow_meta, Mapping)
            and dataflow_meta.get("candidate_illegal_encoding") is True):
        try:
            return _rvgen_extended_reference_observation(testcase, executable_path, form)
        except SemanticUnsupported as error:
            raise ValueError(f"rvgen-semantic unsupported form: {form}") from error
    if form in {"ebreak", "ecall"}:
        _, addresses = symbol_offsets_from_elf(
            Path(executable_path), ("test_start", "exit_checkpoint"),
        )
        test_start = int(addresses["test_start"], 16)
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset:risk.byte_offset + risk.byte_length],
            "little",
        )
        # The catalog's legality rows deliberately mutate the fixed ``rd``
        # field, so their words (0x001000f3/0x000000f3) are illegal encodings,
        # not canonical ebreak/ecall instructions.  Model the actual word;
        # only a canonical encoding reaches the architectural breakpoint or
        # environment-call cause.
        canonical = (
            form == "ebreak" and word == 0x00100073
        ) or (
            form == "ecall" and word == 0x00000073
        )
        cause = (
            3 if form == "ebreak" and canonical else
            _environment_call_cause(_reference_facts(testcase).get("realization"))
            if form == "ecall" and canonical else
            2
        )
        trap_tval = 0 if cause == 3 else word
        identity = {
            "contract": "rvgen-scalar-reference-identity-v4",
            "backend": "rvgen-semantic",
            "identity_digest": _RVGEN_SCALAR_REFERENCE_IDENTITY,
            "form": form,
        }
        return Observation(
            backend="rvgen-semantic", outcome="trap", exit_code=1,
            checkpoint_pc=None, gpr=tuple(testcase.initial_gpr), memory_delta={},
            memory_digest=None,
            signal="SIGTRAP" if cause == 3 else "SIGILL",
            signal_code=5 if cause == 3 else 4, fault_pc=test_start,
            fault_address=word, executed_pcs=(test_start,), instruction_count=1,
            translation_evidence=None, profile_id=testcase.isa_profile,
            input_id=testcase.input_id, raw_stdout="", raw_stderr="",
            binary_sha256=sha256_file(Path(executable_path)),
            tool_version="rvgen-semantic-scalar-v4",
            extra_state={
                "reference_identity": identity,
                "observer_fields": [
                    "executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval",
                ],
                "guest_trap": "delivered", "guest_cause": cause,
                "guest_epc": test_start, "guest_tval": trap_tval,
                "trap.cause": cause, "trap.epc": test_start, "trap.tval": trap_tval,
                "trap_observer": "rvgen-semantic",
                "rvgen_risk_execution": {"status": "observed", "risk_pcs": [test_start]},
            },
            contract_error=None,
        )
    if form in _RVGEN_SCALAR_REFERENCE_FORMS:
        return _rvgen_scalar_reference_observation(testcase, executable_path, form)
    if form in {"vt_maskc_root", "vt_maskcn_root"}:
        _, addresses = symbol_offsets_from_elf(
            Path(executable_path), ("test_start", "exit_checkpoint"),
        )
        test_start = int(addresses["test_start"], 16)
        risk_pc = test_start + int(risk.pc_offset)
        word = int.from_bytes(
            testcase.code_bytes[risk.byte_offset:risk.byte_offset + risk.byte_length], "little",
        )
        identity = {
            "contract": "rvgen-vendor-reference-identity-v2",
            "backend": "rvgen-semantic",
            "identity_digest": _RVGEN_VENDOR_REFERENCE_IDENTITY,
            "form": form,
        }
        return Observation(
            backend="rvgen-semantic", outcome="trap", exit_code=1,
            checkpoint_pc=None, gpr=tuple(testcase.initial_gpr), memory_delta={},
            memory_digest=None, signal=None, signal_code=None, fault_pc=risk_pc,
            fault_address=word, executed_pcs=(risk_pc,), instruction_count=1,
            translation_evidence=None, profile_id=testcase.isa_profile,
            input_id=testcase.input_id, raw_stdout="", raw_stderr="",
            binary_sha256=sha256_file(Path(executable_path)),
            tool_version="rvgen-semantic-v2",
            extra_state={
                "reference_identity": identity,
                "observer_fields": ["executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval"],
                "trap.cause": 2, "trap.epc": risk_pc, "trap.tval": word,
                "rvgen_risk_execution": {"status": "observed", "risk_pcs": [risk_pc]},
            },
            contract_error=None,
        )
    if form not in {"th_srri", "th_tst"}:
        try:
            return _rvgen_extended_reference_observation(testcase, executable_path, form)
        except SemanticUnsupported as error:
            raise ValueError(f"rvgen-semantic unsupported form: {form}") from error
    mask = (1 << 32) - 1
    operands = dict(risk.operand_fields)
    rd, rs1 = int(operands["rd"]), int(operands["rs1"])
    immediate = int(getattr(risk, "immediate", operands.get("p_w_uimm6")))
    source = int(testcase.initial_gpr[rs1]) & mask
    if form == "th_srri":
        amount = immediate & 31
        result = ((source >> amount) | (source << ((32 - amount) & 31))) & mask
    else:
        result = 0 if immediate >= 32 else (source >> immediate) & 1
    gpr = list(testcase.initial_gpr)
    gpr[rd] = result
    gpr[0] = 0
    regions = [bytearray(region.data) for region in testcase.initial_memory_regions]
    for item in testcase.instruction_meta[1:]:
        if "observable" not in item.tags or str(item.mnemonic) != "sw":
            continue
        fields = dict(item.operand_fields)
        address = (int(gpr[int(fields["rs1"])]) + int(fields.get("imm12s", 0))) & mask
        for source_region, region in zip(testcase.initial_memory_regions, regions):
            offset = address - int(source_region.address)
            if 0 <= offset <= len(region) - 4:
                region[offset:offset + 4] = result.to_bytes(4, "little")
    _, addresses = symbol_offsets_from_elf(
        Path(executable_path), ("test_start", "exit_checkpoint"),
    )
    test_start = int(addresses["test_start"], 16)
    checkpoint = int(addresses["exit_checkpoint"], 16)
    risk_pcs = tuple(test_start + item.pc_offset for item in testcase.instruction_meta if "risk" in item.tags)
    memory = bytes(regions[0]) if regions else b""
    identity = {
        "contract": "rvgen-vendor-reference-identity-v1",
        "backend": "rvgen-semantic",
        "identity_digest": _RVGEN_VENDOR_REFERENCE_IDENTITY,
        "form": form,
    }
    return Observation(
        backend="rvgen-semantic", outcome="normal", exit_code=0,
        checkpoint_pc=checkpoint, gpr=tuple(gpr),
        memory_delta={"test-memory": memory.hex()} if memory else {},
        memory_digest=hashlib.sha256(memory).hexdigest() if memory else None,
        signal=None, signal_code=None, fault_pc=None, fault_address=None,
        executed_pcs=tuple(test_start + item.pc_offset for item in testcase.instruction_meta),
        instruction_count=len(testcase.instruction_meta), translation_evidence=None,
        profile_id=testcase.isa_profile, input_id=testcase.input_id,
        raw_stdout="", raw_stderr="", binary_sha256=sha256_file(Path(executable_path)),
        tool_version="rvgen-semantic-v1",
        extra_state={
            "reference_identity": identity,
            "observer_fields": ["executed_pcs", "outcome", "gpr", "memory.test-memory"],
            "rvgen_risk_execution": {"status": "observed", "risk_pcs": list(risk_pcs)},
        },
        contract_error=None,
    )


def _valid_sail_profile_override(path: str) -> bool:
    match = re.fullmatch(r"sail-rvgen-profile-(?P<digest>[0-9a-f]{16})\.json", Path(path).name)
    if match is None:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    digest = canonical_digest(payload)[:16]
    return digest == match.group("digest")


def _sail_config_identity_reconciliation(
    testcase: TestCase,
    config_files: tuple[str, ...],
) -> dict[str, object]:
    """Bind the runtime Sail config to the testcase's frozen realization."""
    dataflow = testcase.dataflow_meta
    realization = dataflow.get("generation_realization")
    obligation = dataflow.get("normative_obligation")
    actual_files = tuple(sorted(str(path) for path in config_files if str(path)))
    recorded_files = tuple(
        sorted(
            str(path)
            for path in obligation.get("sail_config_files", ())
            if isinstance(path, str) and path
        )
        if isinstance(obligation, dict)
        else ()
    )
    # Freeze payloads record absolute config paths from the plane that
    # produced them; a clean-copy execution root relocates the repository.
    # Compare the portable repo-relative labels instead of raw path strings.
    actual_labels = tuple(sorted(_reference_config_label(path) for path in actual_files))
    recorded_labels = tuple(sorted(_reference_config_label(path) for path in recorded_files))
    recorded_identity = (
        str(realization.get("reference_config_identity") or "")
        if isinstance(realization, dict)
        else ""
    )
    owner_identity = reference_config_identity(recorded_files)
    owner_profile = str(realization.get("isa_profile") or "") if isinstance(realization, dict) else ""
    concrete_profile = str(testcase.isa_profile)
    owner_config_matches = len(recorded_identity) == 64 and recorded_identity == owner_identity
    added_files = tuple(
        path for path in actual_files if _reference_config_label(path) not in recorded_labels
    )
    removed_files = tuple(
        path for path in recorded_files if _reference_config_label(path) not in actual_labels
    )
    runtime_base_files = tuple(
        path for path in actual_files if _reference_config_label(path) in recorded_labels
    )
    runtime_base_identity = reference_config_identity(runtime_base_files)
    base_config_matches = recorded_identity == runtime_base_identity
    base_labels_match = tuple(
        sorted(label for label in actual_labels if label in recorded_labels)
    ) == recorded_labels
    # The runtime profile plan appends a deterministic extension override file
    # (content-addressed ``sail-rvgen-profile-<digest>.json``) that the older
    # frozen realization did not record.  Such a difference is an addition
    # only, never a removal, so the frozen owner identity stays authoritative
    # and the runtime machine is a strict superset (enable-only override).
    profile_override = (
        bool(added_files)
        and not removed_files
        and base_config_matches
        and base_labels_match
        and all(
            _valid_sail_profile_override(path)
            for path in added_files
        )
    )
    actual_identity = reference_config_identity(actual_files)
    if not isinstance(realization, dict) or not isinstance(obligation, dict):
        status = "missing-sail-config-contract"
    elif not recorded_files:
        status = "missing-sail-config-files"
    elif not owner_config_matches:
        status = "sail-owner-config-identity-mismatch"
    elif profile_override:
        status = "passed-owner-config-diff"
    elif recorded_labels != actual_labels:
        status = "sail-config-files-mismatch"
    elif len(recorded_identity) != 64 or recorded_identity != actual_identity:
        status = "sail-config-identity-mismatch"
    else:
        status = "passed"
    return {
        "contract": "rvgen-sail-config-identity-reconciliation-v1",
        "status": status,
        "recorded_config_files": list(recorded_files),
        "runtime_config_files": list(actual_files),
        "recorded_config_identity": recorded_identity,
        "owner_config_identity": owner_identity,
        "runtime_config_identity": actual_identity,
        "runtime_base_config_identity": runtime_base_identity,
        "base_config_matches": base_config_matches,
        "base_labels_match": base_labels_match,
        "owner_profile": owner_profile,
        "concrete_profile": concrete_profile,
        "profile_override": profile_override,
        "added_config_files": list(added_files),
        "removed_config_files": list(removed_files),
    }


def _vector_memory_write_width(testcase: TestCase, mnemonic: str) -> int | None:
    normalized = mnemonic.replace("_", ".")
    boundary = str(testcase.dataflow_meta.get("vector_boundary") or "")
    state = re.search(r"sew(?P<sew>8|16|32|64|128)", boundary)
    sew = int(state.group("sew")) if state else 8
    if normalized.startswith("vs") and re.search(r"vs[1248]r\.v$", normalized):
        # Sail emits one callback per byte for whole-register stores.  This
        # helper returns the width of one trace callback, not the footprint of
        # the complete register group.
        return 1
    if normalized == "vsm.v":
        # vsm.v uses the same byte-granular callback path in Sail.
        return 1
    if not normalized.startswith("vs") or not normalized.endswith(".v"):
        return None
    indexed = re.search(r"ei(?P<eew>8|16|32|64)", normalized)
    if indexed is not None:
        # eiN is the index EEW; data EEW is VTYPE.SEW.
        return max(1, sew // 8)
    data = re.search(r"e(?P<eew>8|16|32|64|128)(?:\.v)$", normalized)
    if data is None:
        return None
    return int(data.group("eew")) // 8


def _sail_memory_write_width(
    mnemonic: str | None,
    testcase: TestCase | None = None,
) -> int | None:
    if not isinstance(mnemonic, str) or not mnemonic:
        return None
    direct_widths = {
        "sb": 1,
        "sh": 2,
        "sw": 4,
        "sd": 8,
        "c.sb": 1,
        "c.sh": 2,
        "c.sw": 4,
        "c.swsp": 4,
        "c.sd": 8,
        "c.sdsp": 8,
        "fsw": 4,
        "fsd": 8,
        "fsh": 2,
        "c.fsw": 4,
        "c.fswsp": 4,
        "c.fsd": 8,
        "c.fsdsp": 8,
    }
    width = direct_widths.get(mnemonic)
    if width is not None:
        return width
    if testcase is not None and mnemonic.startswith("vs") and mnemonic.endswith(".v"):
        return _vector_memory_write_width(testcase, mnemonic)
    if mnemonic.startswith(("amo", "sc.", "amocas.")):
        widths = {
            "b": 1,
            "h": 2,
            "w": 4,
            "d": 8,
            "q": 16,
        }
        return next((widths[item] for item in mnemonic.split(".")[1:] if item in widths), None)
    return None


def _sail_trace_state(
    testcase: TestCase,
    trace_text: str,
    *,
    test_memory_address: int,
    test_start_address: int | None = None,
    test_end_address: int | None = None,
    allowed_memory_ranges: tuple[tuple[int, int], ...] = (),
) -> tuple[dict[str, object] | None, str | None]:
    memory = bytearray(b"".join(region.data for region in testcase.initial_memory_regions))
    memory_limit = test_memory_address + len(memory)
    gpr = list(testcase.initial_gpr)
    fpr = [0] * 32
    vector_registers: dict[str, int] = {}
    vector_trace_events: list[dict[str, object]] = []
    csr_values: dict[str, int] = {"fflags": 0, "frm": 0}
    csr_trace_names: set[str] = set()
    csr_trace_addresses: set[int] = set()
    csr_trace_events: list[dict[str, object]] = []
    memory_write_events: list[dict[str, object]] = []
    executed_pcs: list[int] = []
    executed_instruction_words: list[int | None] = []
    executed_mnemonics: list[str] = []
    current_mnemonic: str | None = None
    current_pc: int | None = None
    gpr_mask = (1 << (32 if str(testcase.isa_profile).startswith("rv32") else 64)) - 1
    fpr_mask = (1 << (64 if str(testcase.isa_profile).startswith("rv64") or "d" in enabled_extensions(testcase.isa_profile) else 32)) - 1
    csr_mask = gpr_mask
    vector_config = _sail_vector_config(testcase.isa_profile) or {}
    vector_register_bits = 1 << int(vector_config.get("vlen_exp", 6))
    for line in trace_text.splitlines():
        instruction_match = _SAIL_TRACE_INSTRUCTION_RE.match(line)
        if instruction_match is not None:
            current_pc = int(instruction_match.group("pc"), 16)
            if current_pc > U64_MASK:
                return None, "sail-pc-out-of-range"
            executed_pcs.append(current_pc)
            instruction_word = int(instruction_match.group("word"), 16)
            if instruction_word > 0xFFFFFFFF:
                return None, "sail-instruction-word-out-of-range"
            executed_instruction_words.append(instruction_word)
            body = instruction_match.group("body").strip()
            current_mnemonic = body.split()[0] if body else None
            executed_mnemonics.append(str(current_mnemonic or ""))
            continue
        gpr_match = _SAIL_TRACE_GPR_RE.match(line)
        if gpr_match is not None:
            index = int(gpr_match.group("index"))
            value = int(gpr_match.group("value"), 16)
            if index >= 32:
                return None, "sail-gpr-index-out-of-range"
            if value > gpr_mask:
                return None, "sail-gpr-value-out-of-range"
            if index == 0 and value != 0:
                return None, "sail-gpr-x0-nonzero"
            gpr[index] = value
            continue
        fpr_match = _SAIL_TRACE_FPR_RE.match(line)
        if fpr_match is not None:
            index = int(fpr_match.group("index"))
            if index >= 32:
                return None, "sail-fpr-index-out-of-range"
            value = int(fpr_match.group("value"), 16)
            if value > fpr_mask:
                return None, "unsupported-sail-fpr-width"
            fpr[index] = value
            continue
        vector_match = _SAIL_TRACE_VECTOR_RE.match(line)
        if vector_match is not None:
            index = int(vector_match.group("index"))
            if index >= 32:
                return None, "sail-vector-index-out-of-range"
            element = vector_match.group("element")
            element_index = None if element is None else int(element)
            if element_index is not None and element_index >= vector_register_bits // 8:
                return None, "sail-vector-element-out-of-range"
            key = f"v{index}" if element_index is None else f"v{index}[{element_index}]"
            value = int(vector_match.group("value"), 16)
            if value.bit_length() > (64 if element_index is not None else vector_register_bits):
                return None, "sail-vector-value-out-of-range"
            vector_registers[key] = value
            vector_trace_events.append(
                {
                    "index": index,
                    "element": element_index,
                    "value": value,
                    "mnemonic": current_mnemonic,
                    "pc": current_pc,
                }
            )
            continue
        csr_match = _SAIL_TRACE_CSR_RE.match(line)
        if csr_match is not None:
            name = csr_match.group("name")
            address = int(csr_match.group("address"), 16)
            direction = csr_match.group("direction")
            if address > 0xFFF:
                return None, "sail-csr-address-out-of-range"
            expected_address = _VECTOR_CSR_ADDRESSES.get(
                name, _TRAP_CSR_ADDRESSES.get(name, _STATE_CSR_ADDRESSES.get(name))
            )
            if expected_address is not None and address != expected_address:
                return None, "sail-csr-name-address-mismatch"
            csr_trace_names.add(name)
            csr_trace_addresses.add(address)
            value = int(csr_match.group("value"), 16)
            if value > csr_mask:
                return None, "sail-csr-value-out-of-range"
            csr_trace_events.append(
                {
                    "name": name,
                    "address": address,
                    "direction": direction,
                    "value": value,
                    "mnemonic": current_mnemonic,
                    "pc": current_pc,
                }
            )
            if name in {"mepc", "sepc", "mtval", "stval"}:
                if direction == "<-" and name not in csr_values:
                    csr_values[name] = value
            elif name in {"mcause", "scause"}:
                # Keep the first delivered cause: the pinned Sail emulator
                # re-enters its trap loop after delivery and re-writes the
                # cause at pc 0, which would otherwise overwrite the real one.
                if direction == "<-" and name not in csr_values:
                    csr_values[name] = value
            else:
                csr_values[name] = value
            if name == "fcsr":
                csr_values["fflags"] = value & 0x1F
                csr_values["frm"] = (value >> 5) & 0x7
            continue
        mem_write_match = _SAIL_TRACE_MEM_WRITE_RE.match(line)
        if mem_write_match is None:
            continue
        if (
            test_start_address is not None and test_end_address is not None
            and current_pc is not None
            and not test_start_address <= current_pc < test_end_address
        ):
            continue
        address = int(mem_write_match.group("address"), 16)
        width = _sail_memory_write_width(current_mnemonic, testcase)
        if width is None:
            return None, f"unsupported-sail-memory-write-width:{current_mnemonic or 'unknown'}"
        inside_test_memory = test_memory_address <= address and address + width <= memory_limit
        inside_allowed_range = any(
            start <= address and address + width <= end
            for start, end in allowed_memory_ranges
        )
        if not inside_test_memory and not inside_allowed_range:
            return None, f"out-of-bounds-sail-memory-write:0x{address:x}"
        value_text = mem_write_match.group("value")
        if len(value_text) % 2 or len(value_text) // 2 != width:
            return None, (
                "sail-memory-write-width-mismatch:"
                f"{current_mnemonic or 'unknown'}:expected={width}:trace={len(value_text) // 2}"
        )
        value = int(value_text, 16)
        memory_write_events.append(
            {
                "address": address,
                "value": value,
                "width": width,
                "direction": "write",
                "mnemonic": current_mnemonic,
                "pc": current_pc,
            }
        )
        offset = address - test_memory_address
        for index in range(width):
            if 0 <= offset + index < len(memory):
                memory[offset + index] = (value >> (8 * index)) & 0xFF
    vector_vtype = csr_values.get("vtype")
    vector_sew = (
        {0: 8, 1: 16, 2: 32, 3: 64}.get((int(vector_vtype) >> 3) & 0x7)
        if isinstance(vector_vtype, int)
        else None
    )
    if vector_sew is not None and any(
        event["element"] is not None and int(event["value"]).bit_length() > vector_sew
        for event in vector_trace_events
    ):
        return None, "sail-vector-value-out-of-range"
    while (
        executed_pcs
        and executed_pcs[-1] == 0
        and test_start_address is not None
        and test_end_address is not None
        and not test_start_address <= executed_pcs[-1] < test_end_address
    ):
        executed_pcs.pop()
        executed_instruction_words.pop()
        executed_mnemonics.pop()
    if test_start_address is not None and test_end_address is not None:
        selected = [
            (pc, mnemonic)
            for pc, mnemonic in zip(executed_pcs, executed_mnemonics)
            if int(test_start_address) <= int(pc) < int(test_end_address)
        ]
        branch_pcs = tuple(pc for pc, _mnemonic in selected)
        branch_mnemonics = tuple(mnemonic for _pc, mnemonic in selected)
    else:
        branch_pcs = tuple(executed_pcs)
        branch_mnemonics = tuple(executed_mnemonics)
    return {
        "gpr": tuple(gpr),
        "fpr_rawbits": tuple(fpr),
        "vector_registers": dict(sorted(vector_registers.items())),
        "vector_trace_events": tuple(vector_trace_events),
        "vector_trace_observed": bool(vector_registers),
        "vector_csrs": {
            name: int(value)
            for name, value in csr_values.items()
            if name in {"vl", "vtype", "vstart", "vxsat", "vxrm", "vcsr", "vlenb"}
        },
        "csr_values": {
            str(name): int(value)
            for name, value in sorted(csr_values.items())
            if isinstance(value, int)
        },
        "csr_trace_names": tuple(sorted(csr_trace_names)),
        "csr_trace_addresses": tuple(sorted(csr_trace_addresses)),
        "csr_trace_events": tuple(csr_trace_events),
        "csr_trace_observed": bool(csr_trace_names),
        "memory_write_events": tuple(memory_write_events),
        "fflags": int(csr_values.get("fflags", 0)) & 0x1F,
        "frm": int(csr_values.get("frm", 0)) & 0x7,
        "mcause": csr_values.get("mcause"),
        "mepc": csr_values.get("mepc"),
        "mtval": csr_values.get("mtval"),
        "sepc": csr_values.get("sepc"),
        "stval": csr_values.get("stval"),
        "memory_hex": memory.hex(),
        "memory_digest": hashlib.sha256(memory).hexdigest() if memory else None,
        "executed_pcs": tuple(executed_pcs),
        "executed_instruction_words": tuple(executed_instruction_words),
        "executed_mnemonics": tuple(executed_mnemonics),
        "test_region_pcs": branch_pcs,
        "test_region_mnemonics": branch_mnemonics,
        "test_start_address": test_start_address,
        "instruction_count": len(executed_pcs),
        "trace_sha256": hashlib.sha256(trace_text.encode("utf-8")).hexdigest(),
    }, None


def _expected_instruction_ids(testcase: TestCase) -> tuple[bool, tuple[str, ...]]:
    raw = testcase.dataflow_meta.get("expected_executed_instruction_ids", ())
    valid = (
        isinstance(raw, (list, tuple))
        and all(isinstance(item, str) and item for item in raw)
        and len(raw) == len(set(raw))
    )
    values = raw if isinstance(raw, (list, tuple)) else ()
    return valid, tuple(item for item in values if isinstance(item, str))


def _canonical_sail_mnemonic(value: object) -> str:
    parts = str(value).split()
    normalized = (parts[0] if parts else "").replace("_", ".")
    base, separator, suffix = normalized.rpartition(".")
    if separator and (suffix in {"N", "n"} or suffix.isdigit()):
        number = int(suffix) if suffix.isdigit() else -1
        if base == "mop.r" and (suffix in {"N", "n"} or number in range(32)):
            return "mop.r.N"
        if base == "mop.rr" and (suffix in {"N", "n"} or number in range(8)):
            return "mop.rr.N"
        if base == "c.mop" and (suffix in {"N", "n"} or number in range(1, 16, 2)):
            return "c.mop.N"
    return re.sub(r"\.(?:aqrl|aq|rl)$", "", normalized)


_VECTOR_SEGMENT_MNEMONIC_RE = re.compile(
    r"^(?P<prefix>vlux|vlox|vsux|vsox|vls|vss|vl|vs)"
    r"(?P<tail>e(?:i)?(?:8|16|32|64|128)(?:ff)?)\.v$"
)


def _sail_mnemonic_aliases(
    value: object, operand_fields: tuple[tuple[str, int], ...] = ()
) -> tuple[str, ...]:
    mnemonic = _canonical_sail_mnemonic(value)
    try:
        fields = int(dict(operand_fields).get("nf", 0)) + 1
    except (TypeError, ValueError):
        return (mnemonic,)
    if fields <= 1:
        return (mnemonic,)
    match = _VECTOR_SEGMENT_MNEMONIC_RE.fullmatch(mnemonic)
    if match is None:
        return (mnemonic,)
    return (
        mnemonic,
        f"{match.group('prefix')}seg{fields}{match.group('tail')}.v",
    )


def sail_branch_reconciliation(
    testcase: TestCase,
    *,
    executed_mnemonics: tuple[str, ...] | list[str] = (),
    executed_pcs: tuple[int, ...] | list[int] = (),
    test_start_address: int | None = None,
) -> dict[str, object]:
    """Compare the frozen expected instruction path with Sail's trace path.

    This is deliberately a path identity check, not a claim that a mnemonic
    list is a complete Sail proof.  When PCs are supplied, their offsets must
    also match the frozen instruction metadata.  A missing trace or a path
    mismatch is reported as a distinct disposition and never silently counted
    as a pass.
    """
    expected_ids_valid, expected_ids = _expected_instruction_ids(testcase)
    by_id = {str(item.instruction_id): item for item in testcase.instruction_meta}

    expected_items = tuple(by_id[item] for item in expected_ids if item in by_id)
    expected = tuple(_canonical_sail_mnemonic(item.mnemonic) for item in expected_items)
    expected_aliases = tuple(
        _sail_mnemonic_aliases(item.mnemonic, tuple(item.operand_fields))
        for item in expected_items
    )
    expected_offsets = tuple(int(item.byte_offset) for item in expected_items)
    first_risk_index = next(
        (index for index, item in enumerate(testcase.instruction_meta) if "risk" in item.tags),
        None,
    )
    expected_prefix_complete = (
        first_risk_index is None
        or expected_ids[: first_risk_index + 1]
        == tuple(
            str(item.instruction_id)
            for item in testcase.instruction_meta[: first_risk_index + 1]
        )
    )
    non_contiguous_after_non_control = any(
        current != previous + int(previous_item.byte_length)
        and not is_control_semantic_mnemonic_for_generation(str(previous_item.mnemonic))
        for previous_item, previous, current in zip(
            expected_items,
            expected_offsets,
            expected_offsets[1:],
        )
    )
    observed_contract_valid = isinstance(executed_mnemonics, (tuple, list))
    observed = tuple(
        _canonical_sail_mnemonic(item)
        for item in executed_mnemonics
        if isinstance(item, str) and item.strip()
    ) if observed_contract_valid else ()
    if observed_contract_valid and any(
        not isinstance(item, str) or not item.strip() for item in executed_mnemonics
    ):
        observed_contract_valid = False
    expected_digest = canonical_digest(list(expected))
    observed_digest = canonical_digest(list(observed))
    observed_pc_values: tuple[int, ...] = ()
    if not isinstance(executed_pcs, (tuple, list)):
        observed_contract_valid = False
    else:
        parsed_pcs: list[int] = []
        for pc in executed_pcs:
            if isinstance(pc, bool) or not isinstance(pc, int) or not 0 <= pc <= U64_MASK:
                observed_contract_valid = False
                continue
            parsed_pcs.append(pc)
        observed_pc_values = tuple(parsed_pcs)
    if test_start_address is not None and (
        isinstance(test_start_address, bool)
        or not isinstance(test_start_address, int)
        or not 0 <= test_start_address <= U64_MASK
    ):
        observed_contract_valid = False
    observed_offsets = (
        tuple(pc - test_start_address for pc in observed_pc_values)
        if observed_pc_values and test_start_address is not None
        else ()
    )
    boundary_mode = compare_mask_requests_outcome_boundary(getattr(testcase, "compare_mask", None))
    boundary_prefix_match = (
        boundary_mode
        and bool(observed_offsets)
        and len(observed_offsets) <= len(expected_offsets)
        and observed_offsets == expected_offsets[: len(observed_offsets)]
        and all(
            first_risk_index is not None and index >= first_risk_index
            or mnemonic in expected_aliases[index]
            for index, mnemonic in enumerate(observed)
            if index < len(expected_aliases)
        )
    )
    if not expected_ids_valid or len(expected_ids) != len(expected):
        status = "invalid-expected-branch-contract"
    elif not expected_prefix_complete:
        status = "invalid-expected-branch-contract"
    elif non_contiguous_after_non_control:
        status = "invalid-expected-branch-contract"
    elif not observed_contract_valid:
        status = "invalid-observed-branch-contract"
    elif not observed:
        status = "missing-sail-branch-trace"
    elif len(expected) == len(observed) and all(
        mnemonic in aliases
        for aliases, mnemonic in zip(expected_aliases, observed)
    ):
        status = (
            "passed"
            if not observed_pc_values or observed_offsets == expected_offsets
            else "branch-pc-trace-mismatch"
        )
    elif boundary_prefix_match:
        # A legality/expected-trap row terminates in the delivered exception,
        # so the Sail trace is a prefix of the frozen program.  The runtime
        # decoder also names an illegal encoding by its decode form (for
        # example ``c.illegal`` for a disabled compressed instruction), so the
        # executed PC path is the contract, not the mnemonic spelling.
        status = "passed-boundary-path"
    else:
        status = "branch-trace-mismatch"
    return {
        "contract": "rvgen-sail-branch-reconciliation-v1",
        "status": status,
        "expected_instruction_count": len(expected),
        "observed_instruction_count": len(observed),
        "expected_mnemonics": list(expected),
        "observed_mnemonics": list(observed),
        "expected_instruction_ids": list(expected_ids),
        "expected_byte_offsets": list(expected_offsets),
        "observed_pcs": list(observed_pc_values),
        "observed_relative_byte_offsets": list(observed_offsets),
        "expected_trace_sha256": expected_digest,
        "observed_trace_sha256": observed_digest,
    }


def sail_branch_contract_audit(testcase: TestCase) -> dict[str, object]:
    """Validate the frozen per-testcase branch contract before execution.

    This is intentionally cheaper than launching Sail.  It proves that a
    testcase names one ordered instruction identity sequence that the runtime
    reconciler can compare against trace mnemonics and PCs.  A ``ready`` result
    is a prerequisite, never an execution pass.
    """
    expected_ids_valid, expected_ids = _expected_instruction_ids(testcase)
    contract = str(testcase.dataflow_meta.get("sail_branch_contract") or "")
    instruction_ids = tuple(str(item.instruction_id) for item in testcase.instruction_meta)
    duplicate_instruction_meta_ids = tuple(
        sorted(
            {
                item
                for item in instruction_ids
                if instruction_ids.count(item) > 1
            }
        )
    )
    by_id = {str(item.instruction_id): item for item in testcase.instruction_meta}
    missing_ids = tuple(item for item in expected_ids if item not in by_id)
    expected_items = tuple(by_id[item] for item in expected_ids if item in by_id)
    expected_offsets = tuple(int(item.byte_offset) for item in expected_items)
    first_risk_index = next(
        (index for index, item in enumerate(testcase.instruction_meta) if "risk" in item.tags),
        None,
    )
    expected_prefix_complete = (
        first_risk_index is None
        or expected_ids[: first_risk_index + 1] == instruction_ids[: first_risk_index + 1]
    )
    risk_ids = tuple(
        str(item.instruction_id)
        for item in testcase.instruction_meta
        if "risk" in getattr(item, "tags", ())
    )
    missing_risk_ids = tuple(item for item in risk_ids if item not in expected_ids)
    contract_sink_ids = tuple(
        str(item)
        for item in getattr(testcase, "observability_contract", None).sink_instruction_ids
    ) if getattr(testcase, "observability_contract", None) is not None else ()
    missing_sink_ids = tuple(item for item in contract_sink_ids if item not in expected_ids)
    # Control-flow paths may skip a physical block only after a control
    # instruction; an ordinary instruction's successor is fixed by layout.
    non_monotonic_after_non_control = any(
        current != previous + int(previous_item.byte_length)
        and not is_control_semantic_mnemonic_for_generation(str(previous_item.mnemonic))
        for previous_item, previous, current in zip(
            expected_items,
            expected_offsets,
            expected_offsets[1:],
        )
    )
    if not expected_ids_valid:
        status = "invalid-expected-branch"
    elif duplicate_instruction_meta_ids:
        status = "duplicate-instruction-meta-id"
    elif not contract:
        status = "missing-branch-contract"
    elif contract != "rvgen-sail-branch-reconciliation-v1":
        status = "unsupported-branch-contract"
    elif not expected_ids:
        status = "missing-expected-branch"
    elif missing_ids:
        status = "unknown-expected-instruction-id"
    elif not expected_prefix_complete:
        status = "missing-expected-instruction-prefix"
    elif missing_risk_ids:
        status = "missing-expected-risk-instruction-id"
    elif missing_sink_ids:
        status = "missing-expected-observer-sink-id"
    elif non_monotonic_after_non_control:
        status = "non-monotonic-expected-byte-offset"
    else:
        status = "ready"
    return {
        "contract": contract or "rvgen-sail-branch-reconciliation-v1",
        "status": status,
        "expected_instruction_ids": list(expected_ids),
        "expected_mnemonics": [str(item.mnemonic) for item in expected_items],
        "expected_byte_offsets": [int(item.byte_offset) for item in expected_items],
        "risk_instruction_ids": list(risk_ids),
        "missing_risk_instruction_ids": list(missing_risk_ids),
        "missing_observer_sink_instruction_ids": list(missing_sink_ids),
        "non_monotonic_after_non_control": non_monotonic_after_non_control,
        "duplicate_instruction_meta_ids": list(duplicate_instruction_meta_ids),
        "missing_instruction_ids": list(missing_ids),
    }


def _sail_risk_contract_obligation_error(
    testcase: TestCase,
    risk,
    contract: object,
    *,
    require_lane_boundary: bool,
) -> str | None:
    """Validate one risk contract against frozen canonical testcase facts.

    Reference reconciliation consumes the contract produced by RVGEN/admission;
    it must not call RVGEN to rebuild a schema or obligation.  Independent
    provenance checks remain owned by the admission gate, while this function
    rejects malformed or stale metadata before a Sail comparison.
    """
    if not isinstance(contract, dict):
        return "missing-risk-contract"
    expected_mnemonic = _canonical_sail_mnemonic(getattr(risk, "mnemonic", ""))
    if contract.get("instruction_id") != getattr(risk, "instruction_id", None):
        return "risk-instruction-id-mismatch"
    if _canonical_sail_mnemonic(contract.get("form") or "") != expected_mnemonic:
        return "risk-form-mismatch"
    lane = contract.get("lane")
    boundary_class = contract.get("boundary_class")
    if require_lane_boundary and (
        not isinstance(lane, str)
        or not lane
        or not isinstance(boundary_class, str)
        or not boundary_class
    ):
        return "risk-obligation-context-missing"
    dataflow = testcase.dataflow_meta
    realization_data = contract.get("generation_realization", dataflow.get("generation_realization"))
    obligation_data = contract.get("normative_obligation", dataflow.get("normative_obligation"))
    if not isinstance(realization_data, dict):
        return "missing-risk-realization"
    if not isinstance(obligation_data, dict):
        return "missing-risk-obligation"
    facts = _reference_facts(testcase)
    canonical_realization = generation_realization_for_testcase(testcase)
    canonical_obligation = generation_normative_obligation_for_testcase(testcase)
    if len(getattr(risk, "tags", ())) == 0:
        return "risk-instruction-metadata-missing"
    if (canonical_realization is not None and len(facts["risk_contracts"] or ()) <= 1
            and realization_data != canonical_realization):
        return "risk-realization-mismatch"
    if (canonical_obligation is not None and len(facts["risk_contracts"] or ()) <= 1
            and obligation_data != canonical_obligation):
        return "risk-obligation-mismatch"
    if not isinstance(realization_data.get("realization_id"), str):
        return "invalid-risk-realization-or-obligation"
    if not isinstance(obligation_data.get("obligation_id"), str):
        return "invalid-risk-realization-or-obligation"
    obligation_id = obligation_data.get("obligation_id")
    if (
        not isinstance(obligation_id, str)
        or len(obligation_id) != 28
        or not obligation_id.startswith("obl-")
        or any(character not in "0123456789abcdef" for character in obligation_id[4:])
    ):
        return "invalid-risk-obligation-id"
    return None


def _sail_single_risk_contract(testcase: TestCase, risk) -> tuple[dict[str, object] | None, str | None]:
    """Lift a single-risk testcase payload into the common risk contract.

    Older materializations keep lane/boundary only in the intent that produced
    them.  Recover that context from the locked obligation (or from the
    explicit generation fields when present) so a forged but format-valid
    obligation id cannot be credited as a Sail reference.
    """
    dataflow = testcase.dataflow_meta
    raw_contracts = dataflow.get("risk_contracts")
    if raw_contracts is not None:
        if (
            not isinstance(raw_contracts, (list, tuple))
            or len(raw_contracts) != 1
            or not isinstance(raw_contracts[0], dict)
        ):
            return None, "single-risk-contract-mismatch"
        persisted = raw_contracts[0]
        if (
            persisted.get("instruction_id") != getattr(risk, "instruction_id", None)
            or _canonical_sail_mnemonic(persisted.get("form") or "")
            != _canonical_sail_mnemonic(getattr(risk, "mnemonic", ""))
        ):
            return None, "single-risk-contract-mismatch"
        if any(
            name in persisted and persisted.get(name) != dataflow.get(name)
            for name in (
                "generation_realization",
                "normative_obligation",
            )
        ):
            return None, "single-risk-obligation-context-unresolved"
        persisted = {
            **persisted,
            "generation_realization": persisted.get(
                "generation_realization", dataflow.get("generation_realization")
            ),
            "normative_obligation": persisted.get(
                "normative_obligation", dataflow.get("normative_obligation")
            ),
        }
        error = _sail_risk_contract_obligation_error(
            testcase,
            risk,
            persisted,
            require_lane_boundary=True,
        )
        return (persisted, None) if error is None else (None, error)
    contract: dict[str, object] = {
        "instruction_id": risk.instruction_id,
        "form": risk.mnemonic,
        "generation_realization": dataflow.get("generation_realization"),
        "normative_obligation": dataflow.get("normative_obligation"),
    }
    facts = _reference_facts(testcase)
    canonical_obligation = generation_normative_obligation_for_testcase(testcase)
    canonical_realization = generation_realization_for_testcase(testcase)
    if canonical_obligation is None or canonical_realization is None:
        return None, "single-risk-obligation-context-unresolved"
    lane = dataflow.get("generation_lane")
    boundary_class = dataflow.get("generation_boundary_class")
    if not isinstance(lane, str) or not lane:
        point = dataflow.get("semantic_point")
        lane = point.get("lane") if isinstance(point, dict) else None
    if not isinstance(boundary_class, str) or not boundary_class:
        point = dataflow.get("semantic_point")
        boundary_class = point.get("primary_boundary_class") if isinstance(point, dict) else None
    if isinstance(lane, str) and lane and isinstance(boundary_class, str) and boundary_class:
        contract["lane"] = lane
        contract["boundary_class"] = boundary_class
        if _sail_risk_contract_obligation_error(
            testcase,
            risk,
            contract,
            require_lane_boundary=True,
        ) is None:
            return contract, None

    obligation_data = facts["obligation"]
    realization_data = facts["realization"]
    if not isinstance(obligation_data, dict) or not isinstance(realization_data, dict):
        return None, "missing-single-risk-obligation-context"
    if obligation_data != canonical_obligation or realization_data != canonical_realization:
        return None, "single-risk-obligation-context-unresolved"
    if not isinstance(lane, str) or not lane:
        point = facts["semantic_point"]
        lane = point.get("lane") if isinstance(point, dict) else None
    if not isinstance(boundary_class, str) or not boundary_class:
        point = facts["semantic_point"]
        boundary_class = point.get("primary_boundary_class") if isinstance(point, dict) else None
    if (
        not isinstance(lane, str)
        or not lane
        or not isinstance(boundary_class, str)
        or not boundary_class
    ):
        return None, "single-risk-obligation-context-unresolved"
    contract["generation_realization"] = canonical_realization
    contract["normative_obligation"] = canonical_obligation
    contract["lane"] = lane
    contract["boundary_class"] = boundary_class
    error = _sail_risk_contract_obligation_error(
        testcase,
        risk,
        contract,
        require_lane_boundary=True,
    )
    return (contract, None) if error is None else (None, error)


def sail_risk_contract_reconciliation(
    testcase: TestCase,
    trace_state: dict[str, object],
) -> dict[str, object]:
    """Reconcile every frozen risk contract with one observed Sail trace.

    Freeze validation owns canonical metadata.  This runtime check only binds
    that metadata to the executed risk PC and mnemonic, so a primary-only
    observation cannot silently stand in for a multi-risk program.
    """
    risks = [item for item in testcase.instruction_meta if "risk" in item.tags]
    raw_contracts = testcase.dataflow_meta.get("risk_contracts")
    if len(risks) == 0:
        return {
            "contract": "rvgen-sail-risk-contract-reconciliation-v1",
            "status": "not-required",
            "risk_count": 0,
            "records": [],
        }
    single_contract_error = None
    if len(risks) == 1:
        # Single-risk materializations persist the realization/obligation at
        # testcase scope rather than creating a one-element risk_contracts
        # list.  They still need a runtime PC binding and strict shape checks.
        single_contract, single_contract_error = _sail_single_risk_contract(testcase, risks[0])
        if (single_contract_error is None and isinstance(raw_contracts, (list, tuple))
                and (len(raw_contracts) != 1 or any(
                    raw_contracts[0].get(key) != single_contract.get(key)
                    for key in ("instruction_id", "form", "lane", "boundary_class")
                ))):
            single_contract_error = "single-risk-contract-mismatch"
        raw_contracts = [] if single_contract is None else [single_contract]
    if single_contract_error is not None:
        return {
            "contract": "rvgen-sail-risk-contract-reconciliation-v1",
            "status": single_contract_error,
            "risk_count": len(risks),
            "records": [],
        }
    if not isinstance(raw_contracts, (list, tuple)) or len(raw_contracts) != len(risks):
        return {
            "contract": "rvgen-sail-risk-contract-reconciliation-v1",
            "status": "missing-risk-contracts",
            "risk_count": len(risks),
            "records": [],
        }
    test_start = trace_state.get("test_start_address")
    if not isinstance(test_start, int):
        return {
            "contract": "rvgen-sail-risk-contract-reconciliation-v1",
            "status": "missing-test-start-address",
            "risk_count": len(risks),
            "records": [],
        }
    executed_pcs = trace_state.get("executed_pcs", ())
    executed_mnemonics = trace_state.get("executed_mnemonics", ())
    if (
        not isinstance(executed_pcs, (list, tuple))
        or not isinstance(executed_mnemonics, (list, tuple))
        or len(executed_pcs) != len(executed_mnemonics)
        or any(
            isinstance(pc, bool) or not isinstance(pc, int) or not 0 <= pc <= U64_MASK
            for pc in executed_pcs
        )
        or any(not isinstance(mnemonic, str) or not mnemonic.strip() for mnemonic in executed_mnemonics)
    ):
        return {
            "contract": "rvgen-sail-risk-contract-reconciliation-v1",
            "status": "invalid-observed-risk-trace",
            "risk_count": len(risks),
            "records": [],
        }
    observed = {
        (pc, _canonical_sail_mnemonic(mnemonic))
        for pc, mnemonic in zip(executed_pcs, executed_mnemonics)
    }
    observed_words = {
        pc: int(word)
        for pc, word in zip(
            executed_pcs,
            trace_state.get("executed_instruction_words", ()),
        )
        if isinstance(word, int)
    }
    records: list[dict[str, object]] = []
    for risk, raw_contract in zip(risks, raw_contracts):
        contract = raw_contract if isinstance(raw_contract, dict) else {}
        obligation = contract.get("normative_obligation")
        realization = contract.get("generation_realization")
        expected_pc = int(test_start) + int(risk.pc_offset)
        expected_mnemonics = _sail_mnemonic_aliases(
            risk.mnemonic, tuple(getattr(risk, "operand_fields", ()))
        )
        observed_match = any(
            (expected_pc, mnemonic) in observed for mnemonic in expected_mnemonics
        )
        if not observed_match and compare_mask_requests_outcome_boundary(getattr(testcase, "compare_mask", None)):
            # A legality/expected-trap row terminates in the exception; the
            # runtime decoder names the encoding by its illegal-decode form
            # (for example ``c.illegal`` for a compressed encoding whose
            # extension is disabled).  The executed PC binding is the
            # contract; the trap cause is cross-checked separately by the
            # trap state contract.
            observed_match = any(pc == expected_pc for pc, _mnemonic in observed)
        record = {
            "instruction_id": str(risk.instruction_id),
            "form": str(contract.get("form") or ""),
            "expected_pc": expected_pc,
            "observed": observed_match,
            "realization_id": (
                str(realization.get("realization_id") or "")
                if isinstance(realization, dict)
                else ""
            ),
            "obligation_id": (
                str(obligation.get("obligation_id") or "")
                if isinstance(obligation, dict)
                else ""
            ),
        }
        if observed_match and observed_words:
            try:
                expected_word = int.from_bytes(
                    testcase.code_bytes[
                        int(risk.byte_offset) : int(risk.byte_offset) + int(risk.byte_length)
                    ],
                    "little",
                )
            except (AttributeError, TypeError, ValueError):
                expected_word = None
            if expected_word is None or observed_words.get(expected_pc) != expected_word:
                record["status"] = "risk-instruction-word-mismatch"
                return {
                    "contract": "rvgen-sail-risk-contract-reconciliation-v1",
                    "status": "risk-instruction-word-mismatch",
                    "risk_count": len(risks),
                    "records": [*records, record],
                }
        records.append(record)
        contract_error = _sail_risk_contract_obligation_error(
            testcase,
            risk,
            contract,
            require_lane_boundary=len(risks) > 1,
        )
        record["obligation_reconciliation"] = "passed" if contract_error is None else contract_error
        if contract_error is not None or not observed_match:
            if contract_error is not None:
                record["status"] = contract_error
            elif not observed_match:
                record["status"] = "risk-pc-or-mnemonic-not-observed"
            return {
                "contract": "rvgen-sail-risk-contract-reconciliation-v1",
                "status": contract_error or "risk-contract-mismatch",
                "risk_count": len(risks),
                "records": records,
            }
    return {
        "contract": "rvgen-sail-risk-contract-reconciliation-v1",
        "status": "passed",
        "risk_count": len(risks),
        "records": records,
    }


def _state_trace_contract_error(
    testcase: TestCase,
    trace_state: dict[str, object],
    *,
    state_domain: str,
    state_observer_keys: set[str],
) -> str | None:
    """Return a fail-closed error for stateful obligations."""
    observability = testcase.observability_contract
    sink_transform = None if observability is None else str(observability.sink_transform)
    test_start = trace_state.get("test_start_address")
    if sink_transform in {"full-width-store", "raw-bits-store", "vector-store"}:
        events = trace_state.get("memory_write_events")
        if not isinstance(events, (list, tuple)) or not events:
            return "missing-sail-memory-sink"
        if sink_transform == "vector-store" and not any(
            str(event.get("mnemonic") or "").replace("_", ".").startswith("vs")
            and str(event.get("mnemonic") or "").replace("_", ".").endswith(".v")
            for event in events
            if isinstance(event, dict)
        ):
            return "missing-sail-vector-store-sink"
        if sink_transform in {"full-width-store", "raw-bits-store"}:
            if not isinstance(test_start, int) or isinstance(test_start, bool):
                return "missing-sail-memory-sink"
            expected_sinks = tuple(
                (
                    _canonical_sail_mnemonic(item.mnemonic),
                    int(test_start) + int(item.pc_offset),
                    _sail_memory_write_width(item.mnemonic, testcase),
                )
                for item in testcase.instruction_meta
                if observability is not None
                and item.instruction_id in observability.sink_instruction_ids
            )
            if (
                not expected_sinks
                or not all(
                    any(
                        isinstance(event.get("pc"), int)
                        and not isinstance(event.get("pc"), bool)
                        and int(event["pc"]) == expected_pc
                        and _canonical_sail_mnemonic(event.get("mnemonic")) == expected_mnemonic
                        and event.get("direction") == "write"
                        and event.get("width") == expected_width
                        and isinstance(event.get("address"), int)
                        and not isinstance(event.get("address"), bool)
                        and expected_width is not None
                        and int(event["address"]) % int(expected_width) == 0
                        for event in events
                        if isinstance(event, dict)
                    )
                    for expected_mnemonic, expected_pc, expected_width in expected_sinks
                )
            ):
                return "missing-sail-memory-sink"
    if state_domain == "lifecycle":
        executed_pcs = trace_state.get("executed_pcs", ())
        executed_mnemonics = trace_state.get("executed_mnemonics", ())
        if (
            not isinstance(executed_pcs, (list, tuple))
            or not isinstance(executed_mnemonics, (list, tuple))
            or len(executed_pcs) != len(executed_mnemonics)
        ):
            return "missing-sail-lifecycle-trace"
        if any(
            isinstance(pc, bool)
            or not isinstance(pc, int)
            or not 0 <= pc <= U64_MASK
            or not isinstance(mnemonic, str)
            or not mnemonic.strip()
            for pc, mnemonic in zip(executed_pcs, executed_mnemonics)
        ):
            return "invalid-sail-lifecycle-trace"
        observed = {
            (int(pc), _canonical_sail_mnemonic(mnemonic))
            for pc, mnemonic in zip(executed_pcs, executed_mnemonics)
        }
        expected_items = testcase.instruction_meta
        if not isinstance(test_start, int) or isinstance(test_start, bool):
            return "missing-sail-test-start-address"

        def lifecycle_tag_observed(tag: str) -> bool:
            return any(
                tag in getattr(item, "tags", ())
                and (int(test_start) + int(item.pc_offset), _canonical_sail_mnemonic(item.mnemonic))
                in observed
                for item in expected_items
            )

        lifecycle_events = testcase.dataflow_meta.get("lifecycle_events", ())
        if "code-store" in lifecycle_events and not lifecycle_tag_observed(
            "instruction-memory-patch"
        ):
            return "missing-sail-lifecycle-code-store"
        if "fence.i" in lifecycle_events and not any(
            _canonical_sail_mnemonic(item.mnemonic) == "fence.i"
            and (int(test_start) + int(item.pc_offset), "fence.i") in observed
            for item in expected_items
        ):
            return "missing-sail-lifecycle-fence-i"
        if "second-fetch" in lifecycle_events and not lifecycle_tag_observed(
            "instruction-memory-target"
        ):
            return "missing-sail-lifecycle-second-fetch"
        if "instruction-fetch" in lifecycle_events and not observed:
            return "missing-sail-lifecycle-instruction-fetch"
    if state_domain in {"vector", "vector-memory"}:
        if isinstance(test_start, bool) or not isinstance(test_start, int):
            return "missing-sail-test-start-address"

        # Bind every vector memory callback to the emitted observer instruction
        # and its architectural single-callback width.  Merely seeing *some*
        # ``vs*`` write (for example the risk store) is not sufficient.
        if sink_transform == "vector-store":
            expected_sinks = tuple(
                (
                    item,
                    int(test_start) + int(item.pc_offset),
                    _sail_mnemonic_aliases(
                        item.mnemonic, tuple(getattr(item, "operand_fields", ()))
                    ),
                )
                for item in testcase.instruction_meta
                if observability is not None
                and item.instruction_id in observability.sink_instruction_ids
            )
            if not expected_sinks:
                return "missing-sail-vector-sink-contract"
            observed_sinks = [
                event
                for event in trace_state.get("memory_write_events", ())
                if isinstance(event, dict)
            ]
            for item, expected_pc, expected_mnemonics in expected_sinks:
                matching = [
                    event
                    for event in observed_sinks
                    if _canonical_sail_mnemonic(event.get("mnemonic")) in expected_mnemonics
                    and isinstance(event.get("pc"), int)
                    and int(event["pc"]) == expected_pc
                    and event.get("direction") == "write"
                ]
                if not matching:
                    return "missing-sail-vector-sink-event"
                expected_width = _sail_memory_write_width(item.mnemonic, testcase)
                if expected_width is None:
                    return f"unsupported-sail-memory-write-width:{item.mnemonic}"
                if any(int(event.get("width", -1)) != expected_width for event in matching):
                    return "sail-vector-callback-width-mismatch"
                if any(
                    not isinstance(event.get("address"), int)
                    or int(event["address"]) % max(int(expected_width), 1) != 0
                    for event in matching
                ):
                    return "sail-vector-memory-address-misaligned"
        elif sink_transform in {"full-width-store", "raw-bits-store"}:
            # Scalar writebacks from vector extracts (for example vmv.x.s and
            # vfmv.f.s) still need a concrete architectural sink.  A random
            # store in the trace must not satisfy the scalar observer merely
            # because VL/VTYPE/VSTART were present.
            if "vector.scalar-result" not in state_observer_keys:
                return "missing-sail-vector-scalar-sink-contract"
            expected_sinks = {
                (
                    _canonical_sail_mnemonic(item.mnemonic),
                    int(test_start) + int(item.pc_offset),
                ): item
                for item in testcase.instruction_meta
                if observability is not None
                and item.instruction_id in observability.sink_instruction_ids
            }
            if not expected_sinks:
                return "missing-sail-vector-scalar-sink-contract"
            observed_sinks = [
                event
                for event in trace_state.get("memory_write_events", ())
                if isinstance(event, dict)
            ]
            for (expected_mnemonic, expected_pc), item in expected_sinks.items():
                matching = [
                    event
                    for event in observed_sinks
                    if _canonical_sail_mnemonic(event.get("mnemonic"))
                    == expected_mnemonic
                    and isinstance(event.get("pc"), int)
                    and not isinstance(event.get("pc"), bool)
                    and int(event["pc"]) == expected_pc
                ]
                if not matching:
                    return "missing-sail-vector-scalar-sink-event"
                if any(event.get("direction") != "write" for event in matching):
                    return "sail-vector-scalar-sink-direction-mismatch"
                expected_width = _sail_memory_write_width(item.mnemonic, testcase)
                if expected_width is None:
                    return f"unsupported-sail-memory-write-width:{item.mnemonic}"
                if any(int(event.get("width", -1)) != expected_width for event in matching):
                    return "sail-vector-scalar-callback-width-mismatch"

        vector_csrs = set(trace_state.get("vector_csrs", {}))
        required = {"vl", "vtype", "vstart", "vlenb"}
        if not required <= vector_csrs:
            return "missing-sail-vector-csr-trace"
        vector_boundary = str(testcase.dataflow_meta.get("vector_boundary") or "")
        vector_values = trace_state.get("vector_csrs")
        if not isinstance(vector_values, dict):
            return "missing-sail-vector-csr-trace"
        dynamic_keys = {
            str(key)
            for key in state_observer_keys
            if str(key) in {"fflags", "frm", "vxsat", "vxrm"}
        }
        csr_values = trace_state.get("csr_values")
        csr_values = csr_values if isinstance(csr_values, dict) else {}
        trace_names = set(trace_state.get("csr_trace_names", ()))
        dynamic_limits = {"fflags": 0x1F, "frm": 0x7, "vxsat": 0x1, "vxrm": 0x3}
        for key in sorted(dynamic_keys):
            if key not in trace_names:
                return f"missing-sail-vector-dynamic-csr-trace:{key}"
            values = vector_values if key in {"vxsat", "vxrm"} else csr_values
            if (
                key not in values
                or not isinstance(values[key], int)
                or int(values[key]) < 0
                or int(values[key]) > dynamic_limits[key]
            ):
                return f"invalid-sail-vector-dynamic-csr-trace:{key}"
        try:
            vl = vector_values["vl"]
            vtype = vector_values["vtype"]
            vstart = vector_values["vstart"]
            vlenb = vector_values["vlenb"]
        except KeyError:
            return "invalid-sail-vector-csr-trace"
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (vl, vtype, vstart, vlenb)
        ):
            return "invalid-sail-vector-csr-trace"
        if vlenb <= 0:
            return "invalid-sail-vector-csr-trace"
        xlen_bits = 32 if str(testcase.isa_profile).startswith("rv32") else 64
        if any(value < 0 or value >= (1 << xlen_bits) for value in (vtype, vstart, vlenb)):
            return "invalid-sail-vector-csr-trace"
        vector_config = _sail_vector_config(testcase.isa_profile)
        if (
            vector_config is None
            or vlenb != (1 << int(vector_config["vlen_exp"])) // 8
        ):
            return "invalid-sail-vector-csr-trace"
        vill_bit = 1 << (xlen_bits - 1)
        altfmt = "zvfbfa" in enabled_extensions(testcase.isa_profile)
        config_vill_expected = (
            vector_boundary == "vector-config:reserved-vtype"
            or (
                vector_boundary == "vector-config:altfmt-vtype"
                and not altfmt
            )
        )
        if vtype & vill_bit:
            if (
                not config_vill_expected
                or vtype != vill_bit
                or vl != 0
                or vstart != 0
            ):
                return "invalid-sail-vector-csr-trace"
        elif config_vill_expected:
            return "sail-vector-config-vill-mismatch"
        sew = {0: 8, 1: 16, 2: 32, 3: 64}.get((vtype >> 3) & 0x7)
        lmul = {
            0: (1, 1), 1: (2, 1), 2: (4, 1), 3: (8, 1),
            5: (1, 8), 6: (1, 4), 7: (1, 2),
        }.get(vtype & 0x7)
        if not vtype & vill_bit:
            reserved_mask = ((1 << (xlen_bits - 1)) - 1) & ~0x1FF
            if (
                vtype & reserved_mask
                or sew is None
                or lmul is None
                or (bool(vtype & 0x100) and (not altfmt or sew >= 32))
                or sew > (1 << int(vector_config["elen_exp"]))
            ):
                return "invalid-sail-vector-csr-trace"
            if vector_boundary == "vector-config:altfmt-vtype" and not vtype & 0x100:
                return "sail-vector-config-altfmt-mismatch"
            if vector_boundary.startswith("vector-config:") and vstart != 0:
                return "sail-vector-config-vstart-mismatch"
            vlmax = (vlenb * 8 * lmul[0]) // (sew * lmul[1])
            if vl > vlmax:
                return "invalid-sail-vector-csr-trace"
        if vector_boundary == "vector-vl:zero" and vl != 0:
            return "sail-vector-vl-boundary-mismatch"
        if vector_boundary == "vector-vl:one" and vl != 1:
            return "sail-vector-vl-boundary-mismatch"
        if vector_boundary == "vector-vl:vlmax" and vl <= 0:
            return "sail-vector-vl-boundary-mismatch"
        if vector_boundary == "vector-vl:vlmax":
            expected_vlmax = (vlenb * 8 * lmul[0]) // (sew * lmul[1])
            if vl != expected_vlmax:
                return "sail-vector-vl-boundary-mismatch"
        if vector_boundary == "vector-ff:element1-trim" and vl != 1:
            return "sail-vector-fault-only-first-vl-mismatch"
        boundary_axis, boundary_vstart, expected_sew, expected_lmul = (
            _vector_boundary_facts(vector_boundary) or (None, None, None, None)
        )
        if (
            boundary_axis == "vstart"
            and boundary_vstart is not None
            and vstart != boundary_vstart
        ):
            return "sail-vector-vstart-boundary-mismatch"
        if boundary_axis == "vtype":
            if expected_sew is None or expected_lmul is None:
                return "unsupported-sail-vector-vtype-boundary"
            if ((vtype >> 3) & 0x7) != expected_sew or (vtype & 0x7) != expected_lmul:
                return "sail-vector-vtype-boundary-mismatch"
        expected_csr_observers = tuple(
            (name, int(address), int(test_start) + int(item.pc_offset))
            for item in testcase.instruction_meta
            if {"vector-csr-state", "vector-config-state"} & set(getattr(item, "tags", ()))
            for name, address in _VECTOR_CSR_ADDRESSES.items()
            if any(
                field == "csr" and int(value) == address
                for field, value in getattr(item, "operand_fields", ())
            )
        )
        if any(
            not any(
                event.get("name") == name
                and event.get("address") == address
                and event.get("pc") == pc
                and event.get("direction") == "->"
                for event in trace_state.get("csr_trace_events", ())
                if isinstance(event, dict)
            )
            for name, address, pc in expected_csr_observers
        ):
            return "missing-sail-vector-csr-observer"
        if vector_boundary and not vector_boundary.startswith("vector-config:"):
            if bool(vtype & 0x40) != (vector_boundary != "vector-tail:undisturbed"):
                return "sail-vector-tail-policy-mismatch"
            if bool(vtype & 0x80) != (not vector_boundary.startswith("vector-mask:")):
                return "sail-vector-mask-policy-mismatch"
        if "vector.registers" in state_observer_keys and not trace_state.get(
            "vector_trace_observed"
        ):
            return "missing-sail-vector-register-trace"
        if "vector.registers" in state_observer_keys:
            expected_destinations = {
                (
                    str(item.mnemonic).replace("_", "."),
                    int(value),
                    int(test_start) + int(item.pc_offset),
                )
                for item in testcase.instruction_meta
                if "risk" in item.tags
                for name, value in item.operand_fields
                if name == "vd" and isinstance(test_start, int)
            }
            events = {
                (
                    str(event.get("mnemonic") or "").replace("_", "."),
                    int(event.get("index", -1)),
                    int(event["pc"]) if isinstance(event.get("pc"), int) else -1,
                )
                for event in trace_state.get("vector_trace_events", ())
                if isinstance(event, dict)
            }
            if not expected_destinations or not expected_destinations <= events:
                return "missing-sail-risk-vector-register-trace"
            mask_required = any(
                any(name == "vm" and int(value) == 0 for name, value in item.operand_fields)
                for item in testcase.instruction_meta
                if "risk" in item.tags
            )
            mask_seed = next(
                (
                    item
                    for item in testcase.instruction_meta
                    if "vector-state-setup" in item.tags
                    and item.mnemonic == "vmv.v.i"
                    and any(name == "vd" and int(value) == 0 for name, value in item.operand_fields)
                ),
                None,
            )
            if mask_required:
                if mask_seed is None:
                    return "missing-sail-vector-mask-contract"
                mask_pc = int(test_start) + int(mask_seed.pc_offset)
                mask_events = [
                    event
                    for event in trace_state.get("vector_trace_events", ())
                    if isinstance(event, dict)
                    and int(event.get("index", -1)) == 0
                    and isinstance(event.get("pc"), int)
                    and int(event["pc"]) == mask_pc
                ]
                if not mask_events:
                    # The parser has no element-level mask/tail callback.  A
                    # missing mask carrier is therefore a contract gap, not a
                    # silently complete vector observation.
                    return "missing-sail-vector-mask-trace"
                boundary = str(testcase.dataflow_meta.get("vector_boundary") or "")
                expected_mask_value = (
                    0
                    if boundary.endswith("all-off") or boundary.endswith("vm:0")
                    else 31
                    if boundary.endswith("all-on")
                    else None
                )
                expected_mask_values = (
                    {expected_mask_value}
                    if expected_mask_value is not None
                    else set()
                )
                if expected_mask_value == 31:
                    # ``vmv.v.i vd, 31`` encodes the -1 immediate.  The
                    # architectural element trace therefore reports all bits
                    # set for the active SEW (for example 0xff at SEW=8), not
                    # the five-bit immediate value itself.  Keep accepting 31
                    # for compact trace producers, but derive the concrete
                    # all-ones element from the observed VTYPE as well.
                    vtype = trace_state.get("vector_csrs", {}).get("vtype")
                    if isinstance(vtype, int):
                        sew_code = (int(vtype) >> 3) & 0x7
                        if sew_code <= 3:
                            expected_mask_values.add((1 << (8 << sew_code)) - 1)
                if expected_mask_values and not any(
                    isinstance(event.get("value"), int)
                    and int(event["value"]) in expected_mask_values
                    for event in mask_events
                    if isinstance(event, dict)
                ):
                    return "sail-vector-mask-boundary-mismatch"
    if state_domain == "csr":
        expected = {
            int(value)
            for item in testcase.instruction_meta
            if "risk" in item.tags
            for name, value in item.operand_fields
            if name == "csr"
        }
        observed = {int(value) for value in trace_state.get("csr_trace_addresses", ())}
        if not trace_state.get("csr_trace_observed"):
            return "missing-sail-csr-observer"
        if expected and not expected <= observed:
            return "missing-sail-risk-csr-trace"
        if expected:
            expected_risk_events = {
                (
                    int(value),
                    int(test_start) + int(item.pc_offset),
                )
                for item in testcase.instruction_meta
                if "risk" in item.tags
                for name, value in item.operand_fields
                if name == "csr" and isinstance(test_start, int)
            }
            events = [
                event
                for event in trace_state.get("csr_trace_events", ())
                if isinstance(event, dict) and int(event.get("address", -1)) in expected
            ]
            if not expected_risk_events <= {
                (
                    int(event.get("address", -1)),
                    int(event["pc"]) if isinstance(event.get("pc"), int) else -1,
                )
                for event in events
            }:
                return "missing-sail-csr-risk-event"
            for address, risk_pc in expected_risk_events:
                risk = next(
                    (
                        item
                        for item in testcase.instruction_meta
                        if "risk" in item.tags
                        and isinstance(test_start, int)
                        and int(test_start) + int(item.pc_offset) == risk_pc
                        and any(
                            name == "csr" and int(value) == address
                            for name, value in item.operand_fields
                        )
                    ),
                    None,
                )
                if risk is None:
                    return "missing-sail-csr-risk-event"
                try:
                    from .riscv_encoding import decode_form, form_for_mnemonic
                    from .spec_definedness import csr_state_access_for_generation

                    mnemonic = str(risk.mnemonic).replace(".", "_")
                    form = form_for_mnemonic(mnemonic)
                    if hasattr(testcase, "code_bytes"):
                        word = int.from_bytes(
                            testcase.code_bytes[
                                int(risk.byte_offset) : int(risk.byte_offset) + int(risk.byte_length)
                            ],
                            "little",
                        )
                        decoded = decode_form(form, word) if form is not None else {}
                        source = int(decoded.get("zimm5", decoded.get("rs1", 0)))
                    else:
                        fields = dict(getattr(risk, "operand_fields", ()))
                        source = int(fields.get("zimm5", fields.get("rs1", 0)))
                    access = csr_state_access_for_generation(
                        mnemonic,
                        rd=risk.rd,
                        rs1=source,
                    )
                except (KeyError, TypeError, ValueError):
                    access = None
                if access is None:
                    return "unsupported-sail-csr-access-contract"
                reads, writes = access
                risk_events = [
                    event
                    for event in events
                    if int(event.get("address", -1)) == address
                    and isinstance(event.get("pc"), int)
                    and int(event["pc"]) == risk_pc
                ]
                try:
                    from .riscv_csrs import csr_name

                    expected_name = str(csr_name(address) or "").strip('"')
                except (TypeError, ValueError):
                    expected_name = ""
                if expected_name and any("name" in event for event in risk_events) and not any(
                    str(event.get("name") or "").strip('"') == expected_name
                    for event in risk_events
                ):
                    return "sail-csr-name-mismatch"
                expected_directions = {
                    direction
                    for enabled, direction in ((reads, "->"), (writes, "<-"))
                    if enabled
                }
                if not expected_directions <= {
                    str(event.get("direction") or "") for event in risk_events
                }:
                    return "sail-csr-risk-direction-mismatch"
                readback_items = tuple(
                    item
                    for item in testcase.instruction_meta
                    if "risk" not in item.tags
                    and "csr-state" in item.tags
                    and any(
                        name == "csr" and int(value) == address
                        for name, value in item.operand_fields
                    )
                )
                if readback_items and not isinstance(test_start, int):
                    return "missing-sail-test-start-address"
                post_risk_events = [
                    event
                    for event in events
                    if event.get("address") == address
                    and event.get("direction") == "->"
                    and isinstance(event.get("pc"), int)
                    and int(event["pc"]) > risk_pc
                ]
                if readback_items:
                    if not all(
                        any(
                            isinstance(event.get("pc"), int)
                            and int(event["pc"])
                            == int(test_start) + int(item.pc_offset)
                            and (
                                "mnemonic" not in event
                                or _canonical_sail_mnemonic(event.get("mnemonic"))
                                == _canonical_sail_mnemonic(item.mnemonic)
                            )
                            for event in post_risk_events
                        )
                        for item in readback_items
                    ):
                        return "missing-sail-csr-post-risk-readback"
                elif not post_risk_events:
                    return "missing-sail-csr-post-risk-readback"
    if state_domain == "privileged":
        return "unsupported-sail-privileged-observer"
    if state_domain == "trap":
        names = {str(name) for name in trace_state.get("csr_trace_names", ())}
        if not {"mcause", "mepc", "mtval"} <= names:
            return "missing-sail-trap-observer"
        mcause = trace_state.get("mcause")
        expected_mcause = _expected_trap_mcause(_reference_facts(testcase))
        if not isinstance(mcause, int):
            return "missing-sail-trap-cause"
        if expected_mcause is not None and int(mcause) != expected_mcause:
            return "sail-trap-cause-mismatch"
        mepc = trace_state.get("mepc")
        mtval = trace_state.get("mtval")
        if not isinstance(mepc, int) or not isinstance(mtval, int):
            return "missing-sail-trap-payload"
        risk_pcs = {
            int(test_start) + int(item.pc_offset)
            for item in testcase.instruction_meta
            if "risk" in item.tags
            and isinstance(test_start, int)
        }
        if not risk_pcs:
            return "missing-sail-test-start-address"
        if mepc not in risk_pcs:
            return "sail-trap-epc-mismatch"
        trap_events = {
            (
                str(event.get("name") or ""),
                int(event["pc"]) if isinstance(event.get("pc"), int) else -1,
                str(event.get("direction") or ""),
            )
            for event in trace_state.get("csr_trace_events", ())
            if isinstance(event, dict)
        }
        trap_event_pcs = {
            required: {
                pc for name, pc, direction in trap_events
                if name == required and pc in risk_pcs and direction == "<-"
            }
            for required in ("mcause", "mepc", "mtval")
        }
        if not all(trap_event_pcs.values()) or not (
            trap_event_pcs["mcause"]
            & trap_event_pcs["mepc"]
            & trap_event_pcs["mtval"]
        ):
            return "sail-trap-event-direction-mismatch"
        trap_event_pc = next(iter(
            trap_event_pcs["mcause"]
            & trap_event_pcs["mepc"]
            & trap_event_pcs["mtval"]
        ))
        if mepc != trap_event_pc:
            return "sail-trap-event-pc-mismatch"
        trap_values = {"mcause": mcause, "mepc": mepc, "mtval": mtval}
        if any(
            event.get("name") in trap_values
            and isinstance(event.get("pc"), int)
            and int(event["pc"]) in risk_pcs
            and event.get("direction") == "<-"
            and event.get("value") != trap_values[event["name"]]
            for event in trace_state.get("csr_trace_events", ())
            if isinstance(event, dict)
        ):
            return "sail-trap-event-value-mismatch"
    if state_domain == "control-flow" and str(
        testcase.dataflow_meta.get("control_boundary", "")
    ).startswith("control-cfi:"):
        return "unsupported-sail-cfi-observer"
    if state_domain == "reservation":
        return "unsupported-sail-reservation-observer"
    return None


def _sail_checkpoint_reached(
    testcase: TestCase,
    executed_pcs: tuple[int, ...],
    *,
    exit_address: int,
    test_start_address: int | None,
) -> bool:
    last_instruction = (
        sorted(testcase.instruction_meta, key=lambda item: item.byte_offset)[-1]
        if testcase.instruction_meta
        else None
    )
    physical_predecessor = (
        exit_address - int(last_instruction.byte_length)
        if last_instruction is not None
        else None
    )
    expected_ids = tuple(
        testcase.dataflow_meta.get("expected_executed_instruction_ids") or ()
    )
    expected_by_id = {
        str(item.instruction_id): item for item in testcase.instruction_meta
    }
    terminal = expected_by_id.get(expected_ids[-1]) if expected_ids else None
    terminal_pc = (
        int(test_start_address) + int(terminal.pc_offset)
        if terminal is not None and test_start_address is not None
        else None
    )
    return exit_address in executed_pcs or (
        bool(executed_pcs)
        and (
            physical_predecessor is not None
            and executed_pcs[-1] == physical_predecessor
            or "control-fallthrough-guard" in getattr(terminal, "tags", ())
            and terminal_pc is not None
            and executed_pcs[-1] == terminal_pc
        )
    )


def sail_reference_observation(
    testcase: TestCase,
    *,
    profile_id: str | None = None,
    input_id: str | None = None,
) -> Observation:
    testcase_profile = str(testcase.isa_profile)
    testcase_input = str(testcase.input_id)
    if profile_id is not None and str(profile_id) != testcase_profile:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=str(profile_id),
            input_id=input_id,
            contract_error="sail-profile-id-mismatch",
        )
    if input_id is not None and str(input_id) != testcase_input:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id or testcase_profile,
            input_id=str(input_id),
            contract_error="sail-input-id-mismatch",
        )
    profile_id = testcase_profile
    input_id = testcase_input
    normative_obligation = testcase.dataflow_meta.get("normative_obligation")
    normative_obligation_id = (
        str(normative_obligation.get("obligation_id") or "")
        if isinstance(normative_obligation, dict)
        else ""
    )
    branch_contract_audit = sail_branch_contract_audit(testcase)
    if branch_contract_audit.get("status") != "ready":
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id,
            input_id=input_id,
            contract_error=(
                "sail-branch-contract:"
                + str(branch_contract_audit.get("status") or "invalid")
            ),
            extra_state={"sail_branch_contract_audit": branch_contract_audit},
        )
    state_contract_audit = _state_contract_audit(testcase)
    if state_contract_audit.get("status") not in {"ready", "not-required"}:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id,
            input_id=input_id,
            contract_error=(
                "sail-state-contract:"
                + str(state_contract_audit.get("status") or "invalid")
            ),
            extra_state={
                "sail_branch_contract_audit": branch_contract_audit,
                "sail_state_contract_audit": state_contract_audit,
                "sail_testcase_id": str(testcase.testcase_id),
                "sail_base_program_id": str(testcase.base_program_id),
                "sail_normative_obligation_id": normative_obligation_id,
            },
        )
    require_execution_plane()
    selected_candidate = _locked_sail_candidate()
    if selected_candidate is None:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id,
            input_id=input_id,
            contract_error="no-runnable-sail-simulator-candidate",
        )
    realization = testcase.dataflow_meta.get("generation_realization", {})
    user_mode = isinstance(realization, dict) and str(
        realization.get("privilege_mode")
        or realization.get("privilege_class")
        or realization.get("environment_profile")
        or ""
    ).lower() in {"user", "u"}
    profile_plan = _sail_profile_plan(
        selected_candidate,
        testcase.isa_profile,
        form=str(testcase.generation_rule_id).partition(":")[0],
        user_mode=user_mode,
    )
    simulator_path = Path(str(selected_candidate["path"]))
    tool_version = _sail_tool_version(simulator_path)
    if profile_plan.get("supported") is not True:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id,
            input_id=input_id,
            contract_error=str(profile_plan.get("reason") or "unsupported-sail-profile-plan"),
            tool_version=tool_version,
        )
    config_identity_audit = _sail_config_identity_reconciliation(
        testcase,
        tuple(str(path) for path in profile_plan.get("config_files", ()) if str(path)),
    )
    if config_identity_audit.get("status") not in {"passed", "passed-owner-config-diff"}:
        return _sail_gap_observation(
            binary_sha256=None,
            profile_id=profile_id,
            input_id=input_id,
            contract_error=(
                "sail-config-identity:"
                + str(config_identity_audit.get("status") or "invalid")
            ),
            tool_version=tool_version,
            extra_state={"sail_config_identity_reconciliation": config_identity_audit},
        )
    with TemporaryDirectory(prefix="sail-reference-") as temp_dir:
        workdir = Path(temp_dir)
        built = build_elf(
            testcase,
            workdir / "case.elf",
            prologue_lines=tuple(profile_plan["prologue_lines"]),
        )
        if not built.ok:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=f"sail-reference-build-failed:{built.exit_code}",
                raw_stdout=built.stdout,
                raw_stderr=built.stderr,
                exit_code=built.exit_code,
                tool_version=tool_version,
            )
        _, addresses = symbol_offsets_from_elf(
            Path(built.path),
            ("entry_checkpoint", "test_start", "test_end", "exit_checkpoint", "test_memory"),
        )
        exit_pc = addresses.get("exit_checkpoint")
        test_start_pc = addresses.get("test_start")
        test_end_pc = addresses.get("test_end")
        test_memory = addresses.get("test_memory")
        if (
            exit_pc is None
            or test_memory is None
            or test_start_pc is None
            or test_end_pc is None
        ):
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="missing-sail-symbol-addresses",
                tool_version=tool_version,
            )
        trace_path = workdir / "case.trace"
        command = [
            str(simulator_path),
            *profile_plan["config_args"],
            "--inst-limit",
            "4000",
            "--stop-at-pc",
            exit_pc,
            "--trace-output",
            str(trace_path),
            "--trace-instr",
            "--trace-gpr",
            "--trace-csr",
            "--trace-mem",
            str(Path(built.path)),
        ]
        if {"f", "d"} & enabled_extensions(testcase.isa_profile):
            command.insert(-1, "--trace-fpr")
        if _sail_vector_config(testcase.isa_profile) is not None:
            command.insert(-1, "--trace-vreg")
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="sail-reference-timeout",
                tool_version=tool_version,
            )
        except OSError as exc:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=f"sail-reference-launch-failed:{exc}",
                tool_version=tool_version,
            )
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace") if trace_path.is_file() else ""
        if not trace_text:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="missing-sail-trace-output",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
            )
        trace_state, trace_error = _sail_trace_state(
            testcase,
            trace_text,
            test_memory_address=int(test_memory, 16),
            test_start_address=(int(test_start_pc, 16) if test_start_pc is not None else None),
            test_end_address=(int(test_end_pc, 16) if test_end_pc is not None else int(exit_pc, 16)),
        )
        if trace_error is not None or trace_state is None:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=str(trace_error or "invalid-sail-trace-state"),
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
            )
        trace_state["sail_config_files"] = tuple(config_identity_audit["runtime_config_files"])
        trace_state["sail_config_identity"] = str(
            config_identity_audit["runtime_config_identity"]
        )
        trace_state["sail_candidate_sha256"] = selected_candidate.get("sha256")
        sail_reference_identity = {
            "contract": "rvgen-sail-reference-identity-v1",
            "backend": "sail-riscv",
            "isa_profile": str(testcase.isa_profile),
            "tool_version": tool_version,
            "config_identity": str(config_identity_audit["runtime_config_identity"]),
        }
        sail_reference_identity_digest = canonical_digest(sail_reference_identity)
        executed_pcs = tuple(int(pc) for pc in trace_state["executed_pcs"])
        exit_address = int(exit_pc, 16)
        checkpoint_reached = _sail_checkpoint_reached(
            testcase,
            executed_pcs,
            exit_address=exit_address,
            test_start_address=(int(test_start_pc, 16) if test_start_pc is not None else None),
        )
        # A legality/expected-trap row terminates in the delivered exception
        # before the exit checkpoint: the risk instruction executed and the
        # simulator exited nonzero.  The trap cause itself is cross-checked
        # against the frozen violation projection by the trap state contract.
        boundary_trap_reached = (
            compare_mask_requests_outcome_boundary(getattr(testcase, "compare_mask", None))
            and proc.returncode != 0
            and test_start_pc is not None
            and int(test_start_pc, 16) in executed_pcs
        )
        if not checkpoint_reached and not boundary_trap_reached:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="sail-exit-checkpoint-not-executed",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state={
                    "sail_exit_checkpoint": exit_address,
                    "sail_test_start": (
                        int(test_start_pc, 16) if test_start_pc is not None else None
                    ),
                    "sail_test_end": (
                        int(test_end_pc, 16) if test_end_pc is not None else None
                    ),
                    "executed_pcs": list(executed_pcs),
                    "executed_mnemonics": list(trace_state.get("executed_mnemonics", ())),
                },
            )
        if proc.returncode != 0 and not boundary_trap_reached:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="sail-unexpected-nonzero-exit",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state={
                    "sail_exit_checkpoint": exit_address,
                    "executed_pcs": list(executed_pcs),
                },
            )
        risk_contract_reconciliation = sail_risk_contract_reconciliation(
            testcase,
            trace_state,
        )
        risk_records = tuple(
            item for item in risk_contract_reconciliation.get("records", ())
            if isinstance(item, Mapping)
        )
        risk_pcs = tuple(
            int(item["expected_pc"])
            for item in risk_records
            if type(item.get("expected_pc")) is int
        )
        branch_reconciliation = sail_branch_reconciliation(
            testcase,
            executed_mnemonics=tuple(trace_state.get("test_region_mnemonics", ())),
            executed_pcs=tuple(trace_state.get("test_region_pcs", ())),
            test_start_address=(int(test_start_pc, 16) if test_start_pc is not None else None),
        )
        state_domain = str(state_contract_audit.get("state_domain") or "")
        state_observer_keys = {
            str(item)
            for item in testcase.dataflow_meta.get("state_observer_keys", ())
            if isinstance(item, str)
        }
        requires_vector_register_trace = "vector.registers" in state_observer_keys
        vector_csr_trace_complete = {"vl", "vtype", "vstart", "vlenb"} <= set(
            trace_state.get("vector_csrs", {})
        )
        vector_trace_complete = bool(trace_state.get("vector_trace_observed"))
        vector_observer_complete = vector_csr_trace_complete and (
            vector_trace_complete if requires_vector_register_trace else True
        )
        extra_state = {
            "fpr_rawbits": list(trace_state["fpr_rawbits"]),
            "fflags": int(trace_state["fflags"]),
            "frm": int(trace_state["frm"]),
            "fp_observer": "observed",
            "sail_trace_sha256": str(trace_state["trace_sha256"]),
            "sail_trace_pc_digest": canonical_digest(list(executed_pcs)),
            "sail_reference_identity": sail_reference_identity,
            "sail_reference_identity_digest": sail_reference_identity_digest,
            "sail_exit_checkpoint_reached": bool(checkpoint_reached),
            "sail_boundary_trap_reached": bool(boundary_trap_reached),
            "sail_stop_at_pc_semantics": "pre-execution",
            "sail_branch_contract_audit": branch_contract_audit,
            "sail_state_contract_audit": state_contract_audit,
            "sail_risk_contract_reconciliation": risk_contract_reconciliation,
            "rvgen_risk_execution": {
                "status": (
                    "observed"
                    if risk_pcs and all(item.get("observed") is True for item in risk_records)
                    else "missing"
                ),
                "risk_pcs": list(risk_pcs),
            },
            "sail_testcase_id": str(testcase.testcase_id),
            "sail_base_program_id": str(testcase.base_program_id),
            "sail_normative_obligation_id": normative_obligation_id,
            "sail_candidate_kind": str(selected_candidate["candidate_kind"]),
            "sail_config_files": list(profile_plan["config_files"]),
            "sail_config_identity_reconciliation": config_identity_audit,
            "sail_candidate_sha256": selected_candidate.get("sha256"),
            "sail_branch_reconciliation": branch_reconciliation,
            "vector_registers": dict(trace_state.get("vector_registers", {})),
            "vector_trace_events": list(trace_state.get("vector_trace_events", ())),
            "vector_csrs": dict(trace_state.get("vector_csrs", {})),
            "vector_observer_status": (
                "observed"
                if vector_observer_complete
                else "missing-vector-register-trace"
                if requires_vector_register_trace and not vector_trace_complete
                else "missing-vector-csr-trace"
            ),
            "csr_trace_names": list(trace_state.get("csr_trace_names", ())),
            "csr_trace_addresses": list(trace_state.get("csr_trace_addresses", ())),
            "csr_trace_events": list(trace_state.get("csr_trace_events", ())),
            "memory_write_events": list(trace_state.get("memory_write_events", ())),
            "csr_values": dict(trace_state.get("csr_values", {})),
            "csr_observer_status": (
                "observed" if bool(trace_state.get("csr_trace_observed")) else "missing-csr-trace"
            ),
        }
        for key in ("fflags", "frm"):
            if key in state_observer_keys:
                extra_state[f"{key}_observer"] = "observed"
        for key in ("vxsat", "vxrm"):
            if key in state_observer_keys and key in extra_state["vector_csrs"]:
                extra_state[key] = int(extra_state["vector_csrs"][key])
                extra_state[f"{key}_observer"] = "observed"
        mstatus = trace_state.get("csr_values", {}).get("mstatus")
        if isinstance(mstatus, int):
            extra_state["csr.mstatus.fs"] = (int(mstatus) >> 13) & 0x3
        nontrap_required_extra_state = {
            str(key)
            for key in getattr(testcase.compare_mask, "extra_state_keys", ())
            if str(key) not in {"trap.cause", "trap.epc", "trap.tval"}
        }
        missing_extra_state = sorted(
            key for key in nontrap_required_extra_state if key not in extra_state
        )
        if missing_extra_state:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=(
                    "missing-sail-extra-state-observer:"
                    + ",".join(missing_extra_state)
                ),
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        if risk_contract_reconciliation.get("status") not in {"passed", "not-required"}:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=(
                    "sail-risk-contract:"
                    + str(risk_contract_reconciliation.get("status") or "invalid")
                ),
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        if state_domain in {"vector", "vector-memory"} and not vector_observer_complete:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="missing-sail-vector-state-observer",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        if branch_reconciliation.get("status") not in {"passed", "passed-boundary-path"}:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=f"sail-branch-reconciliation:{branch_reconciliation.get('status')}",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        expected_csr_addresses = {
            int(value)
            for item in testcase.instruction_meta
            if "risk" in item.tags
            for name, value in item.operand_fields
            if name == "csr"
        }
        observed_csr_addresses = {
            int(value) for value in trace_state.get("csr_trace_addresses", ())
        }
        csr_observer_complete = bool(trace_state.get("csr_trace_observed")) and (
            not expected_csr_addresses
            or expected_csr_addresses <= observed_csr_addresses
        )
        if state_domain == "csr" and not csr_observer_complete:
            # CSR trace parsing is intentionally strict; a scalar GPR result is
            # not a WARL/readback witness.
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=(
                    "missing-sail-csr-observer"
                    if not trace_state.get("csr_trace_observed")
                    else "missing-sail-risk-csr-trace"
                ),
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        state_contract_error = _state_trace_contract_error(
            testcase,
            trace_state,
            state_domain=state_domain,
            state_observer_keys=state_observer_keys,
        )
        if state_contract_error is not None:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=state_contract_error,
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state={
                    **extra_state,
                    "sail_state_contract_error": state_contract_error,
                    "sail_state_domain": state_domain,
                },
            )
        if compare_mask_requests_outcome_boundary(getattr(testcase, "compare_mask", None)) and proc.returncode == 0:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error="sail-expected-trap-not-observed",
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state={
                    **extra_state,
                    "sail_expected_trap": True,
                    "sail_expected_trap_mcause": _expected_trap_mcause(_reference_facts(testcase)),
                },
            )
        memory_hex = str(trace_state["memory_hex"])
        mcause = trace_state.get("mcause")
        if proc.returncode == 0:
            return Observation(
                backend="sail-riscv",
                outcome="normal",
                exit_code=proc.returncode,
                checkpoint_pc=int(exit_pc, 16),
                gpr=tuple(trace_state["gpr"]),
                memory_delta={OBSERVABLE_REGION: memory_hex} if memory_hex else {},
                memory_digest=str(trace_state["memory_digest"]) if trace_state["memory_digest"] else None,
                signal=None,
                signal_code=None,
                fault_pc=None,
                fault_address=None,
                executed_pcs=executed_pcs,
                instruction_count=int(trace_state["instruction_count"]),
                translation_evidence=None,
                profile_id=profile_id,
                input_id=input_id,
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                binary_sha256=built.sha256,
                tool_version=tool_version,
                extra_state=extra_state,
                contract_error=None,
            )
        fault_pc = (
            trace_state.get("mepc")
            if isinstance(trace_state.get("mepc"), int)
            else trace_state.get("sepc")
        )
        fault_address = (
            trace_state.get("mtval")
            if isinstance(trace_state.get("mtval"), int)
            else trace_state.get("stval")
        )
        signal_name = _SAIL_TRAP_SIGNAL_BY_MCAUSE.get(int(mcause)) if isinstance(mcause, int) else None
        if isinstance(mcause, int):
            extra_state["trap_mcause"] = int(mcause)
            extra_state["trap.cause"] = int(mcause)
        if isinstance(fault_pc, int):
            extra_state["trap.epc"] = int(fault_pc)
        if isinstance(fault_address, int):
            extra_state["trap.tval"] = int(fault_address)
        if isinstance(mcause, int) and int(mcause) not in (0, 1, 4, 5, 6, 7, 12, 13, 15):
            fault_address = None
        missing_trap_state = sorted(
            str(key)
            for key in getattr(testcase.compare_mask, "extra_state_keys", ())
            if str(key) not in extra_state
        )
        if missing_trap_state:
            return _sail_gap_observation(
                binary_sha256=built.sha256,
                profile_id=profile_id,
                input_id=input_id,
                contract_error=(
                    "missing-sail-extra-state-observer:"
                    + ",".join(missing_trap_state)
                ),
                raw_stdout=proc.stdout,
                raw_stderr=proc.stderr,
                exit_code=proc.returncode,
                tool_version=tool_version,
                extra_state=extra_state,
            )
        return Observation(
            backend="sail-riscv",
            outcome="nonzero-exit",
            exit_code=proc.returncode,
            checkpoint_pc=None,
            gpr=tuple(trace_state["gpr"]),
            memory_delta={OBSERVABLE_REGION: memory_hex} if memory_hex else {},
            memory_digest=str(trace_state["memory_digest"]) if trace_state["memory_digest"] else None,
            signal=signal_name,
            signal_code=None,
            fault_pc=int(fault_pc) if isinstance(fault_pc, int) else None,
            fault_address=int(fault_address) if isinstance(fault_address, int) else None,
            executed_pcs=executed_pcs,
            instruction_count=int(trace_state["instruction_count"]),
            translation_evidence=None,
            profile_id=profile_id,
            input_id=input_id,
            raw_stdout=proc.stdout,
            raw_stderr=proc.stderr,
            binary_sha256=built.sha256,
            tool_version=tool_version,
            extra_state=extra_state,
            contract_error=None,
        )


def _sail_program_checkpoint_reached(
    executed_pcs: tuple[int, ...], exit_address: int,
) -> bool:
    if exit_address in executed_pcs:
        return True
    if not executed_pcs:
        return False
    last = executed_pcs[-1]
    return last in {exit_address - 2, exit_address - 4} or exit_address <= last <= exit_address + 4


def sail_program_observations(
    source_path: str | Path,
    *,
    isa_profile: str,
    mabi: str | None = None,
    profile_id: str | None = None,
    input_id: str | None = None,
    run_params: Mapping[str, object] | None = None,
    expected_trap: bool = False,
) -> tuple[Observation, ...]:
    """Run a materialized program through the locked Sail reference.

    Program routes do not carry RVGEN ``TestCase`` metadata, so this keeps the
    existing trace parser and observes only the executable's generic boundary:
    GPRs, executed PCs, and trap CSRs when a trap is expected.
    """
    profile = str(isa_profile or "").lower()
    source = Path(source_path).resolve()
    binary_sha256 = None

    def gaps(reason: str, **kwargs: object) -> tuple[Observation, ...]:
        return (_sail_gap_observation(
            binary_sha256=binary_sha256,
            profile_id=profile_id or profile,
            input_id=input_id,
            contract_error=reason,
            **kwargs,
        ),)

    require_execution_plane()
    if not source.is_file():
        return gaps("materialized-program-source-missing")
    selected_candidate = _locked_sail_candidate()
    if selected_candidate is None:
        return gaps("no-runnable-sail-simulator-candidate")
    profile_plan = _sail_profile_plan(selected_candidate, profile)
    simulator_path = Path(str(selected_candidate["path"]))
    tool_version = _sail_tool_version(simulator_path)
    if profile_plan.get("supported") is not True:
        return gaps(str(profile_plan.get("reason") or "unsupported-sail-profile-plan"), tool_version=tool_version)

    from .adapters.program_observer import build_linux_user

    with TemporaryDirectory(prefix="sail-program-reference-") as temp_dir:
        workdir = Path(temp_dir)
        build_params = dict(run_params or {})
        build_params.update({
            "isa": profile,
            "text_address": "0x10000",
            "sail_prologue_lines": tuple(profile_plan["prologue_lines"]),
        })
        if mabi:
            build_params["mabi"] = mabi
        try:
            build_linux_user(source, workdir / "program.elf", build_params)
        except (OSError, RuntimeError, ValueError) as exc:
            return gaps(f"sail-program-build-failed:{exc}", tool_version=tool_version)
        elf_path = workdir / "program.elf"
        binary_sha256 = sha256_file(elf_path)
        _, addresses = symbol_offsets_from_elf(
            elf_path,
            (
                "test_start", "exit_checkpoint", "test_memory_start", "test_memory_end",
                "rvgen_program_body_start", "rvgen_program_body_end",
            ),
        )
        required = ("test_start", "exit_checkpoint", "test_memory_start", "test_memory_end")
        if any(name not in addresses for name in required):
            return gaps("missing-sail-program-symbol-addresses", tool_version=tool_version)
        test_start = int(addresses["test_start"], 16)
        exit_address = int(addresses["exit_checkpoint"], 16)
        memory_start = int(addresses["test_memory_start"], 16)
        memory_end = int(addresses["test_memory_end"], 16)
        body_start = addresses.get("rvgen_program_body_start")
        body_end = addresses.get("rvgen_program_body_end")
        body_start = int(body_start, 16) if body_start is not None else None
        body_end = int(body_end, 16) if body_end is not None else None
        memory_size = memory_end - memory_start
        if memory_size < 0 or memory_size > 1 << 20:
            return gaps("invalid-sail-program-memory-range", tool_version=tool_version)
        try:
            writable_memory_ranges = tuple(
                (start, start + size) for start, size in elf_writable_loads(elf_path)
            )
        except (OSError, ValueError):
            return gaps("invalid-sail-program-memory-layout", tool_version=tool_version)
        source_pc_lines: dict[int, list[int]] = {}
        try:
            debug_lines = subprocess.run(
                ["riscv64-linux-gnu-objdump", "--dwarf=decodedline", str(elf_path)],
                capture_output=True, check=False, encoding="utf-8", errors="replace", timeout=10,
            ).stdout
            for line in debug_lines.splitlines():
                match = re.match(r"\s*program\.S\s+(\d+)\s+0x([0-9a-fA-F]+)", line)
                if match:
                    source_pc_lines.setdefault(int(match[1]), []).append(int(match[2], 16))
        except (OSError, subprocess.SubprocessError):
            source_pc_lines = {}
        if source_pc_lines:
            first_line = min(source_pc_lines)
            source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
            previous_pc = min(source_pc_lines[first_line])
            widths = {".byte": 1, ".hword": 2, ".2byte": 2, ".word": 4, ".4byte": 4,
                      ".dword": 8, ".8byte": 8}
            for line_number in range(first_line - 1, 0, -1):
                match = re.match(r"\s*(\.\w+)", source_lines[line_number - 1])
                width = widths.get(match[1]) if match else None
                if width is None:
                    break
                previous_pc -= width
                source_pc_lines.setdefault(line_number, []).append(previous_pc)
        alignment = 2 if "c" in enabled_extensions(profile) else 4
        source_pc_lines = {
            line: [pc for pc in pcs if pc % alignment == 0]
            for line, pcs in source_pc_lines.items()
            if any(pc % alignment == 0 for pc in pcs)
        }
        translation_evidence = (
            TranslationEvidence(
                backend="sail-riscv", expected_path="either",
                details={
                    "source_pc_map": {
                        "binary_sha256": binary_sha256,
                        "lines": {str(line): sorted(set(pcs)) for line, pcs in source_pc_lines.items()},
                    }
                },
            )
            if source_pc_lines else None
        )
        testcase = SimpleNamespace(
            isa_profile=profile,
            initial_gpr=(0,) * 32,
            initial_memory_regions=(SimpleNamespace(data=bytes(memory_size)),),
            dataflow_meta={},
            instruction_meta=(),
        )
        config_files = tuple(str(path) for path in profile_plan["config_files"])
        reference_identity = {
            "contract": "program-sail-reference-identity-v1",
            "backend": "sail-riscv",
            "isa_profile": profile,
            "tool_version": tool_version,
            "config_identity": reference_config_identity(config_files),
            "source_sha256": sha256_file(source),
        }
        reference_identity_digest = canonical_digest(reference_identity)
        observations: list[Observation] = []
        trace_path = workdir / "program.trace"
        command = [
            str(simulator_path),
            *profile_plan["config_args"],
            "--inst-limit", "4000",
            "--stop-at-pc", hex(exit_address),
            "--trace-output", str(trace_path),
            "--trace-instr", "--trace-gpr", "--trace-csr", "--trace-mem",
            str(elf_path),
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, check=False,
                encoding="utf-8", timeout=60,
            )
        except subprocess.TimeoutExpired:
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error="sail-program-timeout",
                tool_version=tool_version,
            ))
            return tuple(observations)
        except OSError as exc:
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error=f"sail-program-launch-failed:{exc}",
                tool_version=tool_version,
            ))
            return tuple(observations)
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace") if trace_path.is_file() else ""
        if not trace_text:
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error="missing-sail-program-trace",
                raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                exit_code=proc.returncode, tool_version=tool_version,
            ))
            return tuple(observations)
        trace_state, trace_error = _sail_trace_state(
            testcase, trace_text,
            test_memory_address=memory_start,
            test_start_address=test_start,
            test_end_address=exit_address,
            allowed_memory_ranges=writable_memory_ranges,
        )
        if trace_error is not None or trace_state is None:
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error=str(trace_error or "invalid-sail-program-trace"),
                raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                exit_code=proc.returncode, tool_version=tool_version,
            ))
            return tuple(observations)
        executed_pcs = tuple(int(pc) for pc in trace_state["executed_pcs"])
        checkpoint_reached = _sail_program_checkpoint_reached(executed_pcs, exit_address)
        mcause = trace_state.get("mcause")
        mepc = trace_state.get("mepc")
        mtval = trace_state.get("mtval")
        if expected_trap and (proc.returncode == 0 or not isinstance(mcause, int)):
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error="sail-expected-trap-not-observed",
                raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                exit_code=proc.returncode, executed_pcs=executed_pcs,
                instruction_count=len(executed_pcs), tool_version=tool_version,
            ))
            return tuple(observations)
        if expected_trap and (
            not isinstance(mepc, int) or not test_start <= mepc < exit_address
        ):
            observations.append(_sail_gap_observation(
                binary_sha256=binary_sha256, profile_id=profile_id or profile,
                input_id=input_id, contract_error="sail-program-trap-epc-outside-program",
                raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                exit_code=proc.returncode, executed_pcs=executed_pcs,
                instruction_count=len(executed_pcs), tool_version=tool_version,
            ))
            return tuple(observations)
        extra_state = {
            "sail_reference_identity": reference_identity,
            "sail_reference_identity_digest": reference_identity_digest,
            "sail_exit_checkpoint_reached": checkpoint_reached,
            "sail_stop_at_pc_semantics": "pre-execution",
            "csr_trace_names": list(trace_state.get("csr_trace_names", ())),
            "csr_trace_addresses": list(trace_state.get("csr_trace_addresses", ())),
            "csr_trace_events": list(trace_state.get("csr_trace_events", ())),
            "sail_program_test_start": test_start,
            "sail_program_exit_checkpoint": exit_address,
        }
        if body_start is not None and body_end is not None and body_start < body_end:
            body = tuple(pc for pc in executed_pcs if body_start <= pc < body_end)
            extra_state["program_test_path"] = {
                "contract": "program-test-path-v1",
                "status": "observed" if body else "gap",
                "start_pc": body_start,
                "end_pc": body_end,
                "executed_pcs": list(body),
            }
        if isinstance(mcause, int):
            extra_state["trap.cause"] = int(mcause)
        if isinstance(mepc, int):
            extra_state["trap.epc"] = int(mepc)
        if isinstance(mtval, int):
            extra_state["trap.tval"] = int(mtval)
        if proc.returncode == 0:
            if not checkpoint_reached:
                observations.append(_sail_gap_observation(
                    binary_sha256=binary_sha256, profile_id=profile_id or profile,
                    input_id=input_id, contract_error="sail-exit-checkpoint-not-executed",
                    raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                    exit_code=proc.returncode, executed_pcs=executed_pcs,
                    instruction_count=len(executed_pcs), tool_version=tool_version,
                ))
                return tuple(observations)
            observations.append(Observation(
                backend="sail-riscv", outcome="normal", exit_code=0,
                checkpoint_pc=exit_address, gpr=tuple(trace_state["gpr"]),
                memory_delta={"test-memory": str(trace_state["memory_hex"])}
                if trace_state["memory_hex"] else {},
                memory_digest=trace_state["memory_digest"], signal=None, signal_code=None,
                fault_pc=None, fault_address=None, executed_pcs=executed_pcs,
                instruction_count=len(executed_pcs), translation_evidence=translation_evidence,
                profile_id=profile_id or profile, input_id=input_id,
                raw_stdout=proc.stdout, raw_stderr=proc.stderr,
                binary_sha256=binary_sha256, tool_version=tool_version,
                extra_state=extra_state,
            ))
            return tuple(observations)
        fault_pc = mepc if isinstance(mepc, int) else trace_state.get("sepc")
        fault_address = mtval if isinstance(mtval, int) else trace_state.get("stval")
        signal_name = _SAIL_TRAP_SIGNAL_BY_MCAUSE.get(int(mcause)) if isinstance(mcause, int) else None
        observations.append(Observation(
            backend="sail-riscv", outcome="nonzero-exit", exit_code=proc.returncode,
            checkpoint_pc=None, gpr=tuple(trace_state["gpr"]),
            memory_delta={"test-memory": str(trace_state["memory_hex"])}
            if trace_state["memory_hex"] else {},
            memory_digest=trace_state["memory_digest"], signal=signal_name,
            signal_code=None, fault_pc=int(fault_pc) if isinstance(fault_pc, int) else None,
            fault_address=(int(fault_address) if isinstance(fault_address, int)
                           and isinstance(mcause, int) and int(mcause) in (0, 1, 4, 5, 6, 7, 12, 13, 15)
                           else None),
            executed_pcs=executed_pcs, instruction_count=len(executed_pcs),
            translation_evidence=translation_evidence, profile_id=profile_id or profile,
            input_id=input_id, raw_stdout=proc.stdout, raw_stderr=proc.stderr,
            binary_sha256=binary_sha256, tool_version=tool_version,
            extra_state=extra_state,
        ))
        return tuple(observations)

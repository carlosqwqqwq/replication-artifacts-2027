"""完整 program 到 builder/runner 的最小适配入口。"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import posixpath
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import TypeAlias

from framework._util import (
    _PROTECTED_ENTRY_LABELS as _PROTECTED_REWRITE_LABELS,
    _TIMING_ONLY_FIELDS as _VOLATILE_EXTRA_STATE_FIELDS,
    _compare_mask_fields, _path_in_root, canonical_digest, is_riscv_elf, is_sha256_digest,
    json_safe as _json_safe,
    object_field as _observation_value, pc_path, sha256_file,
    strip_c_comments as _strip_c_comments,
)
from framework.adapters.contracts import (
    CAPSULE_BACKENDS,
    PATH_IDENTITY_WITNESS_CONTRACT,
    _STRUCTURED_PATH_BY_BACKEND,
    _observer_value_is_non_observed_marker,
    _observer_value_is_valid,
    canonical_execution_backend,
    is_execution_backend,
    same_execution_backend as _same_execution_backend,
)
from framework.direct_case import (
    _SIGNAL_CODE_BY_NAME, CompareMask, expected_trap_from_dataflow,
)
from framework.adapters.runner import guest_trap_observed, terminal_ebreak_observed
from framework.execution_environment import ExecutionEnvironmentError
from framework.execution_identity import TargetBinaryIdentity, _target_identity_parts
from framework.reference_fallback import (
    K1_QEMU_FALLBACK_FAILURE_CLASSES,
    qemu_fallback_policy_matches,
)
from framework.observability import compare_fields
from framework.observability import (
    _is_observation_side_channel_field,
    _strip_observation_side_channels,
)
from framework.rvemi.program import (
    ProgramVariant, _pc_value, _profile_memory_valid, _profile_operands_valid,
    _register_index,
    _stable_input_identity, _text_encodable,
    _SOURCE_INCBIN_RE,
    _split_source_statements,
    _trace_entries,
    _strip_comments, program_sha256,
)
from framework.rvemi.runtime_spec import runtime_instruction_spec
from framework.rvgen.provider import CaseProgram
from framework.spec_definedness import (
    effect_view_siblings_for_generation, fp_form_family_siblings,
    instruction_effect_class, instruction_spec_for_generation,
    profile_enables_form,
)
from framework.supply.model_candidate_supply import rewrite_candidate_id, rewrite_hint_edits


Program: TypeAlias = CaseProgram | ProgramVariant
BuilderCallback: TypeAlias = Callable[[Path, Path, Mapping[str, object]], str | Path]
_INCLUDE_RE = re.compile(
    r'^[ \t]*(?:(?:[A-Za-z_.$][\w.$]*|\d+)[ \t]*:[ \t]*)*'
    r'(?:\.include|#\s*include)[ \t]+"([^"]+)"[ \t]*(?:#.*|//.*)?\r?$',
    re.IGNORECASE,
)
_SCALAR_IMM = frozenset({
    "addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai",
    "addiw", "slliw", "srliw", "sraiw", "rori", "roriw",
})
_ROUNDING_MODES = ("rne", "rtz", "rdn", "rup", "rmm", "dyn")
_IMMEDIATE_DELTAS = (-2, -1, 1, 2)
_INSTRUCTION_RE = re.compile(
    r"(?P<indent>\s*)(?<![\w.$])(?P<mnemonic>[A-Za-z][\w.]*)"
    r"\s+(?P<operands>[^#;\r\n]+?)(?P<comment>\s*#.*)?"
    r"(?=\s*(?:;|$))", re.IGNORECASE,
)
_SCALAR_REWRITES = {
    "add": "sub", "sub": "add", "and": "or", "or": "and", "xor": "and",
    "sll": "srl", "srl": "sra", "sra": "sll", "slt": "sltu", "sltu": "slt",
    "addw": "subw", "subw": "addw", "sllw": "srlw", "srlw": "sraw", "sraw": "sllw",
    "mul": "mulh", "mulh": "mulhsu", "mulhsu": "mulhu", "mulhu": "mul",
    "div": "divu", "divu": "rem", "rem": "remu", "remu": "div",
    "mulw": "divw", "divw": "divuw", "divuw": "remw", "remw": "remuw", "remuw": "mulw",
    "addi": "andi", "andi": "ori", "ori": "xori", "xori": "addi",
    "slti": "sltiu", "sltiu": "slti",
    "slli": "srli", "srli": "srai", "srai": "slli",
    "addiw": "slliw", "slliw": "srliw", "srliw": "sraiw", "sraiw": "addiw",
}
_COMMUTATIVE_SCALARS = frozenset({"add", "and", "or", "xor", "mul", "mulw"})
_CONTROL_REWRITES = {
    "beq": "bne", "bne": "beq", "blt": "bge", "bge": "blt",
    "bltu": "bgeu", "bgeu": "bltu",
    "beqz": "bnez", "bnez": "beqz", "bltz": "bgez", "bgez": "bltz",
    "blez": "bgtz", "bgtz": "blez",
    "c.beqz": "c.bnez", "c.bnez": "c.beqz",
}
_JUMP_MNEMONICS = frozenset({"jal", "jalr", "c.j", "c.jal", "c.jr", "c.jalr"})
_NON_FALLTHROUGH_MNEMONICS = frozenset(
    _CONTROL_REWRITES.keys()
    | _JUMP_MNEMONICS
    | {
        "j", "jr", "ret", "call", "tail", "ecall", "ebreak", "c.ebreak",
        "scall", "sbreak", "mret", "sret", "uret", "wfi", "unimp", "c.unimp",
    }
)


def _fragment_is_fallthrough(lines: Sequence[str]) -> bool:
    """A multi-line rewrite must stay in one dynamic basic block.

    A control transfer is allowed as the final instruction.  Putting one in
    the middle makes the remaining source lines conditional on the new edge,
    so a source-contiguous fragment is no longer a single witnessed sequence.
    """
    instruction_positions = []
    control_positions = []
    for index, line in enumerate(lines):
        if not isinstance(line, str):
            return False
        match = _INSTRUCTION_RE.search(line.rstrip("\r\n"))
        if match is None:
            continue
        instruction_positions.append(index)
        if match.group("mnemonic").lower() in _NON_FALLTHROUGH_MNEMONICS:
            control_positions.append(index)
    return bool(instruction_positions) and all(
        position == instruction_positions[-1] for position in control_positions
    )


_MEMORY_REWRITES = {
    "lb": "lbu", "lbu": "lb", "lh": "lhu", "lhu": "lh",
    "lw": "lwu", "lwu": "lw",
}
_SPECIAL_REWRITES = {
    "c.addi": "c.andi", "c.andi": "c.addi",
    "c.srli": "c.srai", "c.srai": "c.srli",
    "c.sub": "c.add",
    "c.mv": "c.add", "c.add": "c.mv",
    "fadd.s": "fsub.s", "fsub.s": "fadd.s",
    "fmul.s": "fdiv.s", "fdiv.s": "fmul.s",
    "fadd.d": "fsub.d", "fsub.d": "fadd.d",
    "fmul.d": "fdiv.d", "fdiv.d": "fmul.d",
    "fmin.s": "fmax.s", "fmax.s": "fmin.s",
    "fmin.d": "fmax.d", "fmax.d": "fmin.d",
    "fsgnj.s": "fsgnjn.s", "fsgnjn.s": "fsgnj.s",
    "fsgnj.d": "fsgnjn.d", "fsgnjn.d": "fsgnj.d",
    "fmadd.s": "fmsub.s", "fmsub.s": "fmadd.s",
    "fmadd.d": "fmsub.d", "fmsub.d": "fmadd.d",
    "vadd.vv": "vsub.vv", "vsub.vv": "vadd.vv",
    "vmax.vv": "vmin.vv", "vmin.vv": "vmax.vv",
    "vmseq.vv": "vmsne.vv", "vmsne.vv": "vmseq.vv",
    "csrrw": "csrrs", "csrrs": "csrrc", "csrrc": "csrrw",
    "csrrwi": "csrrsi", "csrrsi": "csrrci", "csrrci": "csrrwi",
    "amoadd.w": "amoswap.w", "amoswap.w": "amoadd.w",
    "amoadd.d": "amoswap.d", "amoswap.d": "amoadd.d",
    "amoxor.w": "amoand.w", "amoand.w": "amoor.w", "amoor.w": "amoxor.w",
    "lr.w": "lr.d", "lr.d": "lr.w", "sc.w": "sc.d", "sc.d": "sc.w",
}


def _available_pairs(pairs, suffixes):
    return {
        f"{left}.{suffix}": f"{right}.{suffix}"
        for suffix in suffixes
        for pair in pairs
        for left, right in (pair, pair[::-1])
        if instruction_spec_for_generation(f"{left}.{suffix}") is not None
        and instruction_spec_for_generation(f"{right}.{suffix}") is not None
    }


_SPECIAL_REWRITES.update(_available_pairs(
    (("fadd", "fsub"), ("fmul", "fdiv"), ("fmin", "fmax"),
     ("fsgnj", "fsgnjn"), ("fmadd", "fmsub")),
    ("h", "s", "d", "q"),
))
_SPECIAL_REWRITES.update(_available_pairs(
    (("vadd", "vsub"), ("vmax", "vmin"), ("vmseq", "vmsne")),
    ("vv", "vx", "vi"),
))
_SPECIAL_REWRITES.update(_available_pairs(
    (("amoadd", "amoswap"), ("amoxor", "amoand"), ("amoand", "amoor")),
    ("b", "h", "w", "d"),
))
_SPECIAL_REWRITES.update(_available_pairs(
    (("amomin", "amomax"), ("amominu", "amomaxu")),
    ("b", "h", "w", "d"),
))
_CSR_PSEUDO_REWRITES = {
    "csrr": ("csrrs", (0, 1, "x0")),
    "csrw": ("csrrw", ("x0", 0, 1)),
    "csrs": ("csrrs", ("x0", 0, 1)),
    "csrc": ("csrrc", ("x0", 0, 1)),
    "csrwi": ("csrrwi", ("x0", 0, 1)),
    "csrsi": ("csrrsi", ("x0", 0, 1)),
    "csrci": ("csrrci", ("x0", 0, 1)),
}
_SPECIAL_OPERAND_RE = re.compile(r"^[A-Za-z0-9_.$+\-()]+$", re.IGNORECASE)
_MEMORY_MNEMONICS = frozenset({
    "lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
    "sb", "sh", "sw", "sd",
})

# Keep deterministic enumeration results once per CLI process.  The source,
# witness-line filter, and fragment budget all participate in the cache key.
_CASE_REWRITE_CACHE: dict[tuple[object, ...], tuple[dict[str, object], ...]] = {}
_CASE_REWRITE_CACHE_LIMIT = 8
_PROTECTED_REWRITE_REGISTERS = frozenset({"x1", "x2", "x10", "a0", "ra", "sp"})
_UNSUPPORTED_REWRITE_DIRECTIVES = frozenset({
    ".subsection", ".rept", ".irp", ".irpc", ".endr", ".macro", ".endm",
    ".if", ".ifdef", ".ifndef", ".else", ".elseif", ".endif",
})
_REGISTER_RE = re.compile(
    r"^(?:x(?:[0-9]|[12][0-9]|3[01])|zero|ra|sp|gp|tp|fp|t[0-6]|s(?:[0-9]|1[01])|a[0-7])$",
    re.IGNORECASE,
)
_FP_REGISTER_RE = re.compile(
    r"^(?:f(?:[0-9]|[12][0-9]|3[01])|ft(?:[0-7]|1[01])|fs(?:[0-9]|1[01])|fa[0-7])$",
    re.IGNORECASE,
)
_IMMEDIATE_RE = re.compile(r"^[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|[0-9]+)$")
_LABEL_OR_IMMEDIATE_RE = re.compile(
    r"^(?:[A-Za-z_.$][\w.$]*|\d+[fb]?|[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|[0-9]+))$"
)
_MEMORY_OPERAND_RE = re.compile(
    r"^\s*(?P<offset>[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|[0-9]+))?"
    r"\s*\(\s*(?P<base>[^()\s]+)\s*\)\s*$", re.IGNORECASE,
)


def _immediate_value(value: str) -> int:
    return int(value, 0) if re.match(r"^[+-]?0[bBoOxX]", value) else int(value, 10)


def _copy_includes(source: Path, source_dir: Path, destination_dir: Path) -> None:
    source_dir = _path_in_root(source_dir, source_dir, "source-dependency")
    destination_dir = _path_in_root(
        destination_dir, destination_dir, "source-dependency"
    )
    source = _path_in_root(source, destination_dir, "source-dependency")
    seen: set[Path] = set()

    def copy_binary(name: str, base: Path) -> None:
        relative = Path(name)
        if relative.is_absolute():
            raise ValueError("source-dependency-outside-root")
        dependency = _path_in_root(base / relative, source_dir, "source-dependency")
        if dependency in seen:
            return
        if not dependency.is_file():
            raise ValueError("source-dependency-missing")
        seen.add(dependency)
        destination = _path_in_root(
            destination_dir / dependency.relative_to(source_dir),
            destination_dir, "source-dependency",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(dependency.read_bytes())

    def copy(name: str, base: Path, stack: tuple[Path, ...] = ()) -> None:
        relative = Path(name)
        if relative.is_absolute():
            raise ValueError("source-dependency-outside-root")
        dependency = _path_in_root(
            base / relative, source_dir, "source-dependency"
        )
        if dependency in stack:
            raise ValueError("source-dependency-cyclic")
        if dependency in seen:
            return
        if not dependency.is_file():
            raise ValueError("source-dependency-missing")
        seen.add(dependency)
        destination = _path_in_root(
            destination_dir / dependency.relative_to(source_dir),
            destination_dir, "source-dependency",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if dependency != destination:
            destination.write_bytes(dependency.read_bytes())
        for _, _, line in _logical_source_entries(_strip_c_comments(
            dependency.read_bytes().decode("utf-8")
        ).splitlines(keepends=True)):
            for statement in _split_source_statements(line):
                match = _INCLUDE_RE.fullmatch(statement.rstrip("\r\n"))
                if match:
                    copy(match[1], dependency.parent, (*stack, dependency))
                match = _SOURCE_INCBIN_RE.fullmatch(statement.rstrip("\r\n"))
                if match:
                    copy_binary(match[2], dependency.parent)

    for _, _, line in _logical_source_entries(_strip_c_comments(
        source.read_bytes().decode("utf-8")
    ).splitlines(keepends=True)):
        for statement in _split_source_statements(line):
            match = _INCLUDE_RE.fullmatch(statement.rstrip("\r\n"))
            if match:
                copy(match[1], source_dir, (source.resolve(),))
            match = _SOURCE_INCBIN_RE.fullmatch(statement.rstrip("\r\n"))
            if match:
                copy_binary(match[2], source_dir)


@dataclass(frozen=True)
class BuiltProgram:
    source_path: Path
    executable_path: Path
    source_sha256: str
    executable_sha256: str
    parent_sha256: str | None
    run_params: Mapping[str, object] = field(default_factory=dict)
    bare_executable_path: Path | None = None
    bare_executable_sha256: str | None = None
    bare_materialization_status: str = "not-requested"
    bare_materialization_reason: str | None = None
    # The default executable remains the historical primary artifact.  These
    # explicit Linux fields let a successful capsule-only fallback stay
    # executable by capsule targets without accidentally feeding that ELF to
    # a Linux-user target.
    linux_executable_path: Path | None = None
    linux_executable_sha256: str | None = None
    linux_materialization_status: str = "not-requested"
    linux_materialization_reason: str | None = None


def _artifact_sha256_for_backend(
    artifact: BuiltProgram, backend: object,
) -> str | None:
    if backend in CAPSULE_BACKENDS:
        return artifact.bare_executable_sha256
    if (
        artifact.linux_executable_path is None
        and artifact.linux_materialization_status == "not-requested"
    ):
        return artifact.executable_sha256
    return artifact.linux_executable_sha256


def _program_params(program: object) -> dict[str, object]:
    return dict(getattr(program, "run_params", {}) or {})


def _logical_source_entries(source_lines: Sequence[str]):
    entries, index = [], 0
    while index < len(source_lines):
        end, logical = index, source_lines[index]
        while logical.rstrip().endswith("\\") and end + 1 < len(source_lines):
            ending = "\r\n" if logical.endswith("\r\n") else "\n" if logical.endswith("\n") else ""
            logical = logical[:-len(ending) - 1] + " " * (len(ending) + 1) + source_lines[end + 1]
            end += 1
        entries.append((index, end, logical))
        index = end + 1
    return entries


def build_program(
    program: Program,
    builder: BuilderCallback,
    *,
    work_dir: str | Path,
    source_dir: str | Path | None = None,
    bare_builder: BuilderCallback | None = None,
) -> BuiltProgram:
    work_dir = _path_in_root(work_dir, work_dir, "build-root")
    work_dir.mkdir(parents=True, exist_ok=True)
    run_params = _program_params(program)
    if isinstance(program, ProgramVariant):
        source_path = _path_in_root(
            work_dir / "program.S", work_dir, "build-artifact"
        )
        if source_path.exists() or source_path.is_symlink():
            raise ValueError("build-artifact already exists")
        source_path.write_bytes(program.source.encode("utf-8"))
        if source_dir is not None:
            _copy_includes(source_path, Path(source_dir), source_path.parent)
        else:
            for name, data in program._include_context:
                target = _path_in_root(source_path.parent / name, source_path.parent, "source-dependency")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        parent_sha256 = program.parent_sha256
    else:
        source_path = Path(getattr(program, "program", program)).resolve()
        parent_sha256 = None
    source_bytes = source_path.read_bytes()
    source_sha256 = program_sha256(program)
    raw_source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    def restore_source_snapshot() -> None:
        """Keep a failed projection builder from poisoning the next one."""
        try:
            if source_path.read_bytes() != source_bytes:
                source_path.write_bytes(source_bytes)
        except OSError:
            source_path.write_bytes(source_bytes)

    executable_path: Path | None = None
    executable_sha256: str | None = None
    linux_executable_path: Path | None = None
    linux_executable_sha256: str | None = None
    linux_materialization_status = "attempted"
    linux_materialization_reason = None
    primary_error: Exception | None = None
    try:
        if is_riscv_elf(source_path):
            executable_path = source_path
            if run_params.get("bare_metal") is True:
                linux_executable_path = None
                linux_executable_sha256 = None
                linux_materialization_status = "gap"
                linux_materialization_reason = "input ELF is declared bare-metal"
            else:
                linux_executable_path = source_path
                linux_executable_sha256 = sha256_file(source_path)
                linux_materialization_status = "built"
        else:
            output_path = _path_in_root(
                work_dir / "program.elf", work_dir, "builder-output"
            )
            if output_path.exists() or output_path.is_symlink():
                raise ValueError("builder-output already exists")
            executable_path = Path(builder(source_path, output_path, run_params))
        try:
            source_unchanged = sha256_file(source_path) == raw_source_sha256
        except OSError:
            source_unchanged = False
        if not source_unchanged:
            source_path.write_bytes(source_bytes)
            raise ValueError("builder changed program source")
        if not is_riscv_elf(source_path):
            executable_path = _path_in_root(
                executable_path if executable_path.is_absolute()
                else work_dir / executable_path,
                work_dir, "builder-output",
            )
        if not executable_path.is_file() or not is_riscv_elf(executable_path):
            raise ValueError("builder must return an ELF executable")
        executable_sha256 = sha256_file(executable_path)
        if run_params.get("bare_metal") is not True:
            linux_executable_path = executable_path
            linux_executable_sha256 = executable_sha256
            linux_materialization_status = "built"
        elif linux_materialization_status != "gap":
            linux_executable_path = None
            linux_executable_sha256 = None
            linux_materialization_status = "gap"
            linux_materialization_reason = "primary ELF is declared bare-metal"
    except Exception as error:
        restore_source_snapshot()
        primary_error = error
        executable_path = None
        executable_sha256 = None
        linux_materialization_status = "gap"
        linux_materialization_reason = f"{type(error).__name__}: {str(error)[:500]}"
    bare_executable_path = None
    bare_executable_sha256 = None
    bare_materialization_status = "not-requested"
    bare_materialization_reason = None
    if bare_builder is not None:
        bare_materialization_status = "attempted"
        if is_riscv_elf(source_path):
            if run_params.get("bare_metal") is True:
                try:
                    bare_output = _path_in_root(
                        work_dir / "program-bare.elf", work_dir, "bare-builder-output"
                    )
                    if bare_output.exists() or bare_output.is_symlink():
                        raise ValueError("bare-builder-output already exists")
                    shutil.copy2(source_path, bare_output)
                    if not is_riscv_elf(bare_output):
                        raise ValueError("reused bare artifact is not an ELF executable")
                    bare_executable_path = bare_output
                    bare_executable_sha256 = sha256_file(bare_output)
                    bare_materialization_status = "reused"
                except Exception as error:
                    bare_materialization_status = "gap"
                    bare_materialization_reason = (
                        f"{type(error).__name__}: {str(error)[:500]}"
                    )
            else:
                bare_materialization_status = "gap"
                bare_materialization_reason = (
                    "bare builder requires source text; binary ELF has no safe "
                    "cross-runtime projection"
                )
        else:
            # A capsule projection is best effort.  Keep the primary artifact
            # and let the capsule target record `artifact-gap` when this
            # projection cannot be materialized; a failed optional conversion
            # must not erase the generated source or Linux-user projection.
            try:
                restore_source_snapshot()
                bare_output = _path_in_root(
                    work_dir / "program-bare.elf", work_dir, "bare-builder-output"
                )
                if bare_output.exists() or bare_output.is_symlink():
                    raise ValueError("bare-builder-output already exists")
                bare_executable_path = Path(bare_builder(
                    source_path, bare_output, {**run_params, "bare_metal": True},
                ))
                if not bare_executable_path.is_absolute():
                    bare_executable_path = work_dir / bare_executable_path
                bare_executable_path = _path_in_root(
                    bare_executable_path, work_dir, "bare-builder-output"
                )
                if not bare_executable_path.is_file() or not is_riscv_elf(bare_executable_path):
                    raise ValueError("bare builder must return an ELF executable")
                if sha256_file(source_path) != raw_source_sha256:
                    raise ValueError("bare builder changed program source")
                bare_executable_sha256 = sha256_file(bare_executable_path)
                bare_materialization_status = "built"
            except Exception as error:
                restore_source_snapshot()
                bare_executable_path = None
                bare_executable_sha256 = None
                bare_materialization_status = "gap"
                bare_materialization_reason = (
                    f"{type(error).__name__}: {str(error)[:500]}"
                )
    if executable_path is None:
        # Keep a valid capsule-only result usable by capsule targets.  The
        # Linux projection remains explicitly absent, so backend selection
        # cannot accidentally run a bare ELF through QEMU/libriscv.
        if bare_executable_path is None or bare_executable_sha256 is None:
            if primary_error is not None:
                raise primary_error
            raise ValueError("no executable projection was materialized")
        executable_path = bare_executable_path
        executable_sha256 = bare_executable_sha256
    return BuiltProgram(
        source_path, executable_path, source_sha256,
        executable_sha256, parent_sha256, run_params,
        bare_executable_path, bare_executable_sha256,
        bare_materialization_status, bare_materialization_reason,
        linux_executable_path, linux_executable_sha256,
        linux_materialization_status, linux_materialization_reason,
    )


def _rewrite_source(program: CaseProgram | str | Path) -> tuple[Path, list[str]]:
    if isinstance(program, ProgramVariant):
        # MCMC states are in-memory ProgramVariant objects.  The source-only
        # fallback enumerator must be able to inspect them after a reference
        # profile gap; no ELF or on-disk source path is required for ordinary
        # single-file cases.
        return Path("__rq1_program_variant__.S"), program.source.splitlines(keepends=True)
    path = Path(getattr(program, "program", program)).resolve()
    if path.suffix.lower() != ".s":
        raise ValueError("case rewrite requires assembly source")
    return path, path.read_bytes().decode("utf-8").splitlines(keepends=True)


def _same_shape_targets(mnemonic: str, targets) -> tuple[str, ...]:
    source = instruction_spec_for_generation(mnemonic)
    if source is None:
        return ()
    shape = (tuple(source.operand_kinds), tuple(source.operand_domains))
    result = []
    for target in targets:
        target = str(target).replace("_", ".")
        target_spec = instruction_spec_for_generation(target)
        if target_spec is not None and (
            tuple(target_spec.operand_kinds), tuple(target_spec.operand_domains)
        ) == shape:
            result.append(target)
    return tuple(dict.fromkeys(result))


def _scalar_targets(mnemonic: str, operands: Sequence[str]) -> tuple[str, ...]:
    result = [_SCALAR_REWRITES[mnemonic]] if mnemonic in _SCALAR_REWRITES else []
    if mnemonic in {"rori", "roriw"} and len(operands) == 3:
        result.append(mnemonic)
    if all(_REGISTER_RE.fullmatch(item) for item in operands):
        result.extend(_same_shape_targets(
            mnemonic,
            (item.mnemonic for item in effect_view_siblings_for_generation(mnemonic)),
        ))
    return tuple(dict.fromkeys(result))


def enumerate_case_rewrites(
    program: CaseProgram | str | Path,
    *,
    source_line_filter: Sequence[int] | set[int] | frozenset[int] | None = None,
    source_location_filter: (
        Sequence[tuple[str, int]] | set[tuple[str, int]]
        | frozenset[tuple[str, int]] | None
    ) = None,
) -> tuple[dict[str, object], ...]:
    """枚举作用范围内全部可物化候选；可按 reference witness 源位置收窄。"""
    path, lines = _rewrite_source(program)
    run_params = _program_params(program)
    virtual_includes = {
        posixpath.normpath(str(name)): data
        for name, data in (getattr(program, "_include_context", ()) or ())
        if isinstance(name, str) and isinstance(data, bytes)
    }

    def source_lines_for(source_name: str | None) -> list[str]:
        if not source_name:
            return lines
        virtual = virtual_includes.get(posixpath.normpath(source_name))
        if virtual is not None:
            return virtual.decode("utf-8").splitlines(keepends=True)
        return (path.parent / source_name).read_bytes().decode("utf-8").splitlines(keepends=True)

    allowed_lines = (
        None if source_line_filter is None
        else frozenset(value for value in source_line_filter if type(value) is int)
    )
    allowed_locations = (
        None if source_location_filter is None else frozenset(
            (str(Path(source_file).resolve()), line)
            for source_file, line in source_location_filter
            if isinstance(source_file, str) and source_file
            and type(line) is int and line > 0
        )
    )
    cache_key = (
        str(path),
        hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
        canonical_digest(run_params),
        None if allowed_lines is None else tuple(sorted(allowed_lines)),
        None if allowed_locations is None else tuple(sorted(allowed_locations)),
    )
    cached = _CASE_REWRITE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dv_allowed_ranges = None
    if run_params.get("harness") == "riscv-dv":
        def label_line(name: str) -> int | None:
            pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s*(?:$|\S)", re.IGNORECASE)
            return next(
                (index for index, line in enumerate(lines) if pattern.match(line.rstrip("\r\n"))),
                None,
            )

        labels = {
            name: label_line(name)
            for name in (
                "_start", "init", "h0_start", "main", "test_done", "write_tohost",
            )
        }
        init = labels["init"] if labels["init"] is not None else labels["h0_start"]
        body = labels["main"] if labels["main"] is not None else init
        if (
            labels["_start"] is not None and body is not None
            and labels["test_done"] is not None and labels["write_tohost"] is not None
            and labels["_start"] < body <= labels["test_done"] < labels["write_tohost"]
        ):
            subprogram_start = next(
                (
                    index for index in range(labels["test_done"] + 1, labels["write_tohost"])
                    if re.match(r"^\s*sub_[0-9]+\s*:", lines[index])
                ),
                labels["write_tohost"],
            )
            # linux_user_harness 的程序路径从 main 开始；旧样本没有 main
            # 时才回退到 init。其余启动/初始化行不会进入最终可执行 profile。
            dv_allowed_ranges = [(body + 1, labels["test_done"] + 2)]
            if subprogram_start < labels["write_tohost"]:
                dv_allowed_ranges.append(
                    (subprogram_start + 1, labels["write_tohost"] + 1)
                )
    isa = run_params.get("isa") or run_params.get("isa_profile")
    rv32 = isinstance(isa, str) and isa.lower().startswith("rv32")
    initial_state = run_params.get("initial_state")
    custom_entry_label = (
        str(initial_state.get("entry", "_start")).lower()
        if run_params.get("harness") == "custom" and isinstance(initial_state, Mapping)
        else None
    )
    protected_labels = _PROTECTED_REWRITE_LABELS | (
        {"exit_checkpoint", "emit_observation", "rvobs_copy_loop", "rvobs_copy_done"}
        if custom_entry_label is not None else set()
    )
    if run_params.get("harness") == "riscv-dv":
        # init/h0_start is copied into the Linux harness and its setup body is
        # executable witness code; only the stack-sensitive operands remain
        # protected by the normal register checks below.
        protected_labels -= {"init", "h0_start"}
    in_text, protected, previous_text, previous_protected, section_stack = True, False, None, False, []
    opaque_depth = 0
    candidates = []
    include_gap = False
    include_protected = False
    opaque_starts = frozenset({"if", "ifdef", "ifndef", "macro", "rept", "irp", "irpc"})
    opaque_ends = frozenset({"endif", "endm", "endr"})

    def opaque_directive(value: str) -> str | None:
        match = re.match(r"^\s*(?:#\s*)?(\.?[A-Za-z][\w.]*)\b", value)
        return match[1].lower().lstrip(".") if match else None

    def advance_section(section_code: str) -> bool:
        nonlocal in_text, protected, previous_text, previous_protected
        if re.match(r"^\s*\.text(?:[.\s]|$)", section_code, re.IGNORECASE):
            previous_text, previous_protected = in_text, protected
            in_text, protected = True, False
            return True
        if re.match(r"^\s*\.section(?:\s|$)", section_code, re.IGNORECASE):
            previous_text, previous_protected = in_text, protected
            protected = False
            flags = re.search(r',\s*"([^"\n]*)"', section_code)
            in_text = "x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", section_code, re.IGNORECASE))
            return True
        if re.match(r"^\s*\.pushsection(?:\s|$)", section_code, re.IGNORECASE):
            section_stack.append((in_text, previous_text, protected, previous_protected))
            previous_text, previous_protected = in_text, protected
            protected = False
            flags = re.search(r',\s*"([^"\n]*)"', section_code)
            in_text = "x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", section_code, re.IGNORECASE))
            return True
        if re.match(r"^\s*\.popsection(?:\s|$)", section_code, re.IGNORECASE):
            if len(section_code.split()) != 1 or not section_stack:
                return True
            in_text, previous_text, protected, previous_protected = section_stack.pop()
            return True
        if re.match(r"^\s*\.(?:data|bss|rodata|sdata|sbss|tdata|tbss)(?:\s|$)", section_code, re.IGNORECASE):
            previous_text, previous_protected = in_text, protected
            in_text, protected = False, False
            return True
        if re.match(r"^\s*\.previous(?:\s|$)", section_code, re.IGNORECASE):
            if previous_text is None:
                return True
            in_text, previous_text = previous_text, in_text
            protected, previous_protected = previous_protected, protected
            return True
        return False

    def scalar_parts(parts):
        label_prefix, instruction, match, mnemonic, operands = parts
        targets = ("li",) if mnemonic == "li" else _scalar_targets(mnemonic, operands)
        if not targets or any(
            item.lower() in _PROTECTED_REWRITE_REGISTERS for item in operands
        ):
            return None
        if mnemonic == "li":
            valid = (
                len(operands) == 2
                and _REGISTER_RE.fullmatch(operands[0])
                and _IMMEDIATE_RE.fullmatch(operands[1])
            )
        elif mnemonic in _SCALAR_IMM:
            valid = (
                len(operands) == 3
                and all(_REGISTER_RE.fullmatch(item) for item in operands[:2])
                and _IMMEDIATE_RE.fullmatch(operands[2])
            )
        elif mnemonic in _SCALAR_REWRITES:
            valid = len(operands) == 3 and all(
                _REGISTER_RE.fullmatch(item) for item in operands
            )
        else:
            spec = instruction_spec_for_generation(mnemonic)
            valid = (
                spec is not None
                and len(operands) == len(spec.operand_kinds)
                and all(_REGISTER_RE.fullmatch(item) for item in operands)
            )
        return parts if valid else None

    def memory_variant(operands, address):
        result = list(operands)
        value = _immediate_value(address["offset"] or "0")
        result[1] = f'{"1" if value == 0 else "0"}({address["base"]})'
        return result

    def instruction_parts(code: str, active: bool):
        stripped = code.strip()
        if not active or not stripped or stripped.startswith((".", "#", ";")):
            return None
        label_prefix = re.match(
            r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", code
        ).group(0)
        instruction = code[len(label_prefix):]
        match = _INSTRUCTION_RE.fullmatch(instruction)
        if match is None:
            return None
        return (
            label_prefix, instruction, match, match["mnemonic"].lower(),
            tuple(item.strip() for item in match["operands"].split(",")),
        )

    def control_parts(parts):
        label_prefix, instruction, match, mnemonic, operands = parts
        if mnemonic in _CONTROL_REWRITES and len(operands) == 3 and (
            all(_REGISTER_RE.fullmatch(item) for item in operands[:2])
            and _LABEL_OR_IMMEDIATE_RE.fullmatch(operands[2])
            and not any(item.lower() in _PROTECTED_REWRITE_REGISTERS for item in operands[:2])
        ):
            return label_prefix, instruction, match, mnemonic, operands
        if mnemonic in {"beqz", "bnez", "c.beqz", "c.bnez", "bltz", "bgez", "blez", "bgtz"} and len(operands) == 2 and (
            _REGISTER_RE.fullmatch(operands[0])
            and _LABEL_OR_IMMEDIATE_RE.fullmatch(operands[1])
            and operands[0].lower() not in _PROTECTED_REWRITE_REGISTERS
        ):
            return label_prefix, instruction, match, mnemonic, operands
        return None

    def jump_parts(parts):
        label_prefix, instruction, match, mnemonic, operands = parts
        if mnemonic in {"jal", "jalr"}:
            if mnemonic == "jal":
                if len(operands) == 1:
                    rd, target, implicit = "ra", _LABEL_OR_IMMEDIATE_RE.fullmatch(operands[0]), True
                elif len(operands) == 2 and _REGISTER_RE.fullmatch(operands[0]):
                    rd, target, implicit = operands[0], _LABEL_OR_IMMEDIATE_RE.fullmatch(operands[1]), False
                else:
                    return None
            elif len(operands) == 2 and _REGISTER_RE.fullmatch(operands[0]):
                rd, target, implicit = operands[0], _MEMORY_OPERAND_RE.fullmatch(operands[1]), False
            elif len(operands) == 3 and all(_REGISTER_RE.fullmatch(item) for item in operands[:2]) and _IMMEDIATE_RE.fullmatch(operands[2]):
                rd, target, implicit = operands[0], {"base": operands[1], "offset": operands[2]}, False
            else:
                return None
            if target is None or not implicit and rd.lower() in _PROTECTED_REWRITE_REGISTERS:
                return None
            if mnemonic == "jalr" and (
                not _REGISTER_RE.fullmatch(target["base"])
                or target["base"].lower() in _PROTECTED_REWRITE_REGISTERS
            ):
                return None
            return label_prefix, instruction, match, mnemonic, operands, target, rd
        if mnemonic in {"c.j", "c.jal"} and len(operands) == 1 and (
            _LABEL_OR_IMMEDIATE_RE.fullmatch(operands[0])
        ) and (mnemonic != "c.jal" or rv32):
            return label_prefix, instruction, match, mnemonic, operands, True, (
                "ra" if mnemonic == "c.jal" else "x0"
            )
        if mnemonic in {"c.jr", "c.jalr"} and len(operands) == 1 and (
            _REGISTER_RE.fullmatch(operands[0])
            and operands[0].lower() not in _PROTECTED_REWRITE_REGISTERS
        ):
            return label_prefix, instruction, match, mnemonic, operands, True, (
                "ra" if mnemonic == "c.jalr" else "x0"
            )
        return None

    def memory_parts(parts):
        label_prefix, instruction, match, mnemonic, operands = parts
        if instruction_effect_class(mnemonic) not in {"load", "store"} \
                or len(operands) < 2:
            return None
        address = _MEMORY_OPERAND_RE.fullmatch(operands[1])
        if address is None or not (
            _REGISTER_RE.fullmatch(operands[0])
            or _FP_REGISTER_RE.fullmatch(operands[0])
            or re.fullmatch(r"v(?:[0-9]|[12][0-9]|3[01])", operands[0], re.IGNORECASE)
        ):
            return None
        if (
            operands[0].lower() in _PROTECTED_REWRITE_REGISTERS
            or address["base"].lower() in _PROTECTED_REWRITE_REGISTERS - {"sp"}
            or not _REGISTER_RE.fullmatch(address["base"])
        ):
            return None
        return label_prefix, instruction, match, mnemonic, operands, address

    def special_edits(parts):
        _, _, _, mnemonic, operands = parts
        pseudo = _CSR_PSEUDO_REWRITES.get(mnemonic)
        if pseudo is not None:
            target, order = pseudo
            return ((target, tuple(
                operands[item] if isinstance(item, int) else item
                for item in order
            )),) if len(operands) == 2 else ()
        result = [_SPECIAL_REWRITES[mnemonic]] if mnemonic in _SPECIAL_REWRITES else []
        result.extend(_same_shape_targets(mnemonic, fp_form_family_siblings(mnemonic)))
        return tuple((_target, operands) for _target in dict.fromkeys(result))

    def generic_edits(parts):
        _, _, _, mnemonic, operands = parts
        if mnemonic in {"li", "rori", "roriw"}:
            return ()
        spec = instruction_spec_for_generation(mnemonic)
        if spec is None or len(spec.operand_kinds) < len(operands):
            return ()
        rv32e = isinstance(isa, str) and isa.lower().startswith("rv32e")
        reserved = {
            _register_index(name) for name in _PROTECTED_REWRITE_REGISTERS
            if _register_index(name) is not None
        }
        result = []
        for index, operand in enumerate(operands):
            if operand.lower() in _PROTECTED_REWRITE_REGISTERS:
                continue
            prefix = (
                "x" if _REGISTER_RE.fullmatch(operand)
                else "f" if _FP_REGISTER_RE.fullmatch(operand)
                else "v" if re.fullmatch(r"v(?:[0-9]|[12][0-9]|3[01])", operand, re.IGNORECASE)
                else None
            )
            if prefix is not None:
                current = (
                    _register_index(operand)
                    if prefix == "x"
                    else int(re.search(r"\d+", operand)[0])
                )
                limit = 16 if prefix == "x" and rv32e else 32
                if current is not None:
                    domain = tuple(
                        value for value in range(limit)
                        if prefix != "x" or value not in reserved
                    )
                    position = domain.index(current) if current in domain else None
                    if position is not None:
                        for offset in (-1, 1):
                            next_value = domain[(position + offset) % len(domain)]
                            changed = list(operands)
                            changed[index] = f"{prefix}{next_value}"
                            result.append((mnemonic, tuple(changed)))
                continue
            kind = spec.operand_kinds[index] if index < len(spec.operand_kinds) else None
            if kind == "rm" and operand.lower() in _ROUNDING_MODES:
                changed = list(operands)
                changed[index] = _ROUNDING_MODES[
                    (_ROUNDING_MODES.index(operand.lower()) + 1) % len(_ROUNDING_MODES)
                ]
                result.append((mnemonic, tuple(changed)))
                continue
            if "imm" in spec.operand_kinds and _IMMEDIATE_RE.fullmatch(operand):
                value = _immediate_value(operand)
                for updated in (value + delta for delta in _IMMEDIATE_DELTAS):
                    changed = list(operands)
                    changed[index] = str(updated)
                    result.append((mnemonic, tuple(changed)))
        return tuple(result)

    def candidate_for(
        line_no, cursor, parts, mnemonic, operands, source_path=None, fragment=None,
    ):
        label_prefix, instruction, match = parts[:3]
        if not _profile_operands_valid(
            mnemonic, operands, xlen=32 if rv32 else 64, isa_profile=isa,
        ):
            return None
        spec = runtime_instruction_spec(mnemonic, 32 if rv32 else 64)
        if spec is not None and not _text_encodable(
            spec, spec.operand_roles, operands, isa_profile=isa,
        ):
            return None
        start_col = cursor + len(label_prefix) + match.start()
        if fragment is not None:
            start_line, end_line, first_raw, last_raw, source_text = fragment
            first_code = first_raw.rstrip("\r\n")
            last_code = last_raw.rstrip("\r\n")
            comment = re.search(r"\s*(?:#|//).*$", last_code)
            replacement = f"{first_code[:start_col]}{mnemonic} {', '.join(operands)}"
            if comment:
                replacement += comment.group(0)
            payload_lines = [replacement]
            payload_lines.extend("" for _ in range(end_line - start_line - 1))
            payload_lines.append("")
            candidate = {
                "source_span": {
                    "start": start_line, "end": end_line,
                    "start_col": start_col, "end_col": len(last_code),
                },
                "operator": "replace_fragment",
                "payload": {"lines": payload_lines},
                "source_text": source_text.rstrip("\r\n"),
                "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            }
        else:
            candidate = {
                "source_span": {
                    "start": line_no, "end": line_no,
                    "start_col": start_col,
                    "end_col": cursor + len(label_prefix) + len(instruction.rstrip()),
                },
                "operator": "replace",
                "payload": {"mnemonic": mnemonic, "operands": list(operands)},
                "source_text": instruction.rstrip(),
                "source_sha256": hashlib.sha256(instruction.rstrip().encode()).hexdigest(),
            }
        if source_path is not None:
            candidate["source_path"] = source_path
        return {**candidate, "candidate_id": rewrite_candidate_id(candidate)}

    def replaced_line(candidate: Mapping[str, object]) -> str | None:
        span = candidate["source_span"]
        source_name = candidate.get("source_path")
        try:
            source_lines = source_lines_for(source_name if isinstance(source_name, str) else None)
        except (OSError, UnicodeError):
            return None
        line = source_lines[span["start"] - 1].rstrip("\r\n")
        segment = line[span["start_col"]:span["end_col"]]
        parsed = _INSTRUCTION_RE.fullmatch(segment)
        payload = candidate.get("payload")
        if parsed is None or not isinstance(payload, Mapping):
            return None
        if not isinstance(payload.get("mnemonic"), str) or not isinstance(payload.get("operands"), list):
            return None
        replacement = f"{parsed.group('indent')}{payload['mnemonic']} {', '.join(payload['operands'])}"
        return line[:span["start_col"]] + replacement + (parsed.group("comment") or "") + line[span["end_col"]:]

    def collect_candidates(line_no, cursor, parts, source_path=None, fragment=None):
        if parts is None or (
            dv_allowed_ranges is not None and source_path is None
            and not any(start <= line_no < end for start, end in dv_allowed_ranges)
        ):
            return []
        def make_candidate(*args):
            return candidate_for(*args, fragment=fragment)
        found = []
        scalar = scalar_parts(parts)
        if scalar is None:
            control = control_parts(parts)
            jump = jump_parts(parts)
            memory = memory_parts(parts)
            operands = parts[4]
            special = parts if (
                special_edits(parts)
                and operands
                and all(_SPECIAL_OPERAND_RE.fullmatch(item) for item in operands)
                and not any(
                    re.search(rf"(?<!\w){re.escape(name)}(?!\w)", item, re.IGNORECASE)
                    for item in operands for name in _PROTECTED_REWRITE_REGISTERS
                )
            ) else None
            if control is not None:
                _, _, _, mnemonic, operands = control
                found.append(make_candidate(
                    line_no, cursor, control, _CONTROL_REWRITES[mnemonic], operands, source_path,
                ))
                if len(operands) == 3 and _register_index(operands[0]) != _register_index(operands[1]):
                    found.append(make_candidate(
                        line_no, cursor, control, mnemonic,
                        (operands[1], operands[0], operands[2]), source_path,
                    ))
            elif jump is not None and jump[6].lower() != "x0":
                mnemonic, operands = jump[3], jump[4]
                if mnemonic in {"jal", "jalr"}:
                    replacement = ("x0", *operands[1:]) if mnemonic == "jalr" and len(operands) == 3 else ("x0", operands[-1])
                    found.append(make_candidate(line_no, cursor, jump, mnemonic, replacement, source_path))
                else:
                    found.append(make_candidate(
                        line_no, cursor, jump,
                        "c.j" if mnemonic == "c.jal" else "c.jr", (operands[-1],), source_path,
                    ))
            elif jump is not None and jump[3] == "c.j" and rv32:
                found.append(make_candidate(line_no, cursor, jump, "c.jal", jump[4], source_path))
            elif memory is not None:
                _, _, _, mnemonic, operands, address = memory
                if mnemonic in _MEMORY_REWRITES and not (rv32 and mnemonic == "lw"):
                    found.append(make_candidate(
                        line_no, cursor, memory, _MEMORY_REWRITES[mnemonic], operands, source_path,
                    ))
                found.append(make_candidate(
                    line_no, cursor, memory, mnemonic,
                    memory_variant(operands, address), source_path,
                ))
            elif special is not None:
                mnemonic = special[3]
                found.extend(
                    make_candidate(line_no, cursor, special, target, operands, source_path)
                    for target, operands in special_edits(special)
                )
            found.extend(
                make_candidate(line_no, cursor, parts, target, operands, source_path)
                for target, operands in generic_edits(parts)
            )
            return found
        _, _, _, mnemonic, operands = scalar
        found.extend(
            make_candidate(line_no, cursor, scalar, target, operands, source_path)
            for target in _scalar_targets(mnemonic, operands)
        )
        if mnemonic in _COMMUTATIVE_SCALARS \
                and _register_index(operands[1]) != _register_index(operands[2]):
            found.append(make_candidate(
                line_no, cursor, scalar, mnemonic,
                (operands[0], operands[2], operands[1]), source_path,
            ))
        elif mnemonic == "li":
            value = _immediate_value(operands[1])
            found.append(make_candidate(
                line_no, cursor, scalar, mnemonic,
                (operands[0], "1" if value == 0 else "0"), source_path,
            ))
        elif mnemonic in _SCALAR_IMM:
            value = _immediate_value(operands[2])
            found.append(make_candidate(
                line_no, cursor, scalar, mnemonic,
                (*operands[:2], "1" if value == 0 else "0"), source_path,
            ))
        found.extend(
            make_candidate(line_no, cursor, parts, target, operands, source_path)
            for target, operands in generic_edits(parts)
        )
        return found

    def scan_include(value: str, base: Path, stack_paths: tuple[Path, ...] = ()) -> bool:
        nonlocal in_text, protected, previous_text, previous_protected, include_gap, include_protected
        match = re.fullmatch(
            r'\s*(?:\.include|#\s*include)\s+"([^"]+)"[ \t]*(?:#.*|//.*)?',
            value, re.IGNORECASE,
        )
        virtual_name = None
        if match:
            try:
                base_name = base.relative_to(path.parent).as_posix()
            except ValueError:
                base_name = ""
            virtual_name = posixpath.normpath(
                posixpath.join(base_name, match[1])
            )
            if virtual_name not in virtual_includes:
                virtual_name = None
        try:
            dependency = (
                path.parent / virtual_name
                if virtual_name is not None else
                _path_in_root(base / match[1], path.parent, "source-dependency")
                if match else None
            )
        except ValueError:
            include_gap = True
            return False
        if dependency is None or (
            virtual_name is None and not dependency.is_file()
        ) or dependency in stack_paths:
            include_gap = True
            return False
        found = False
        try:
            dependency_source = (
                virtual_includes[virtual_name].decode("utf-8")
                if virtual_name is not None else
                dependency.read_bytes().decode("utf-8")
            )
        except (OSError, UnicodeError):
            include_gap = True
            return False
        parent_state = (
            in_text, protected, previous_text, previous_protected,
            tuple(section_stack), include_protected,
        )
        source_name = virtual_name or dependency.relative_to(path.parent).as_posix()
        included_protected = False
        included_opaque_depth = 0
        included_lines = _strip_c_comments(dependency_source).splitlines(keepends=True)
        for included_start, included_end, included_line in _logical_source_entries(included_lines):
            included_line_no = included_start + 1
            included_statements = _split_source_statements(included_line)
            included_fragment = (
                included_line_no, included_end + 1, included_lines[included_start],
                included_lines[included_end], "".join(included_lines[included_start:included_end + 1]),
            ) if included_end > included_start and len(included_statements) == 1 else None
            included_cursor = 0
            for included_part in included_statements:
                included_code = included_part.strip()
                directive = opaque_directive(included_code)
                if directive in opaque_starts:
                    included_opaque_depth += 1
                    continue
                if directive in opaque_ends:
                    if included_opaque_depth:
                        included_opaque_depth -= 1
                    continue
                if included_opaque_depth:
                    continue
                if not re.match(
                    r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*#\s*include\b",
                    included_code, re.IGNORECASE,
                ):
                    included_code = _strip_comments(included_code)
                labels = re.findall(
                    r"[A-Za-z_.$][\w.$]*|\d+",
                    re.match(
                        r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*",
                        included_code,
                    ).group(0),
                )
                if labels:
                    protected = any(
                        name.lower() in protected_labels and name.lower() != "main"
                        for name in labels
                    )
                    if custom_entry_label is not None and any(
                        name.lower() in {custom_entry_label, "init", "main"}
                        for name in labels
                    ):
                        protected = False
                    included_protected = protected
                included_section = re.sub(
                    r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", "",
                    included_code, count=1,
                ).strip()
                if re.fullmatch(
                    r'(?:\.include|#\s*include)\s+"([^"]+)"[ \t]*(?:#.*|//.*)?',
                    included_section, re.IGNORECASE,
                ):
                    found = scan_include(
                        included_section, dependency.parent, (*stack_paths, dependency)
                    ) or found
                    included_protected = include_protected
                    continue
                if advance_section(included_section):
                    included_protected = False
                    continue
                words = included_section.lower().split()
                if in_text and included_section and (
                    not included_section.startswith((".", "#"))
                    or words[0] not in {
                        ".option", ".include", "#include", ".equ", ".set",
                        ".globl", ".global", ".type", ".size", ".attribute",
                        ".file", ".loc", ".ident",
                    }
                    and not words[0].startswith((".cfi_", ".debug_"))
                ):
                    found = True
                    parts = instruction_parts(
                        _strip_comments(included_part.rstrip("\r\n")),
                        in_text and not protected,
                    )
                    candidates.extend(collect_candidates(
                        included_line_no, included_cursor, parts, source_name, included_fragment,
                    ))
                included_cursor += len(included_part)
        included_in_text = in_text
        (
            in_text, protected, previous_text, previous_protected,
            parent_stack, parent_include_protected,
        ) = parent_state
        section_stack[:] = parent_stack
        include_protected = included_protected if included_in_text else parent_include_protected
        protected = include_protected if included_in_text else protected
        return found

    parsed_lines = _strip_c_comments("".join(lines)).splitlines(keepends=True)
    if any(re.search(r";\s*(?:\.include|#\s*include)\b", line, re.IGNORECASE)
           for line in parsed_lines):
        return ()
    for start_index, end_index, parsed_line in _logical_source_entries(parsed_lines):
        line_no = start_index + 1
        statements = _split_source_statements(parsed_line)
        fragment = (
            line_no, end_index + 1, lines[start_index], lines[end_index],
            "".join(lines[start_index:end_index + 1]),
        ) if end_index > start_index and len(statements) == 1 else None
        cursor = 0
        for part in statements:
            raw = part.rstrip("\r\n")
            code = _strip_comments(raw)
            section_code = re.sub(
                r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", "", code,
                count=1,
            ).strip()
            include_code = re.sub(
                r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", "", raw,
                count=1,
            ).strip()
            include = re.fullmatch(
                r'\s*(?:\.include|#\s*include)\s+"([^"]+)"[ \t]*(?:#.*|//.*)?',
                include_code, re.IGNORECASE,
            )
            if include:
                scan_include(include_code, path.parent)
                cursor += len(part)
                continue
            directive = opaque_directive(section_code)
            if directive in opaque_starts:
                opaque_depth += 1
                cursor += len(part)
                continue
            if directive in opaque_ends:
                if opaque_depth:
                    opaque_depth -= 1
                cursor += len(part)
                continue
            if opaque_depth:
                cursor += len(part)
                continue
            if advance_section(section_code):
                include_protected = False
                cursor += len(part)
                continue
            if (in_text and section_code
                    and section_code.split(None, 1)[0].lower()
                    in _UNSUPPORTED_REWRITE_DIRECTIVES):
                cursor += len(part)
                continue
            label_prefix = re.match(
                r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", code
            ).group(0)
            labels = re.findall(r"[A-Za-z_.$][\w.$]*|\d+", label_prefix)
            if labels:
                protected = any(
                    name.lower() in protected_labels and name.lower() != "main"
                    for name in labels
                )
                if (
                    custom_entry_label is not None
                    and any(
                        name.lower() in {custom_entry_label, "init", "main"}
                        for name in labels
                    )
                ):
                    protected = False
                include_protected = protected
            else:
                protected = include_protected
            parts = instruction_parts(code, in_text and not protected)
            candidates.extend(collect_candidates(line_no, cursor, parts, fragment=fragment))
            cursor += len(part)
    candidates = [item for item in candidates if item is not None]
    if allowed_locations is not None:
        def candidate_locations(candidate: Mapping[str, object]) -> frozenset[tuple[str, int]]:
            span = candidate.get("source_span")
            if not isinstance(span, Mapping) or type(span.get("start")) is not int:
                return frozenset()
            source_name = candidate.get("source_path")
            source_file = (
                (path.parent / source_name).resolve()
                if isinstance(source_name, str) else path.resolve()
            )
            payload = candidate.get("payload")
            payload_lines = payload.get("lines") if isinstance(payload, Mapping) else None
            lines = (
                tuple(
                    span["start"] + offset
                    for offset, value in enumerate(payload_lines)
                    if isinstance(value, str) and value.strip()
                ) if isinstance(payload_lines, list) else (span["start"],)
            )
            return frozenset((str(source_file), line) for line in lines)

        candidates = [
            item for item in candidates
            if (locations := candidate_locations(item))
            and locations <= allowed_locations
        ]
    if allowed_lines is not None:
        def candidate_lines(candidate: Mapping[str, object]) -> frozenset[int]:
            span = candidate.get("source_span")
            if not isinstance(span, Mapping) or type(span.get("start")) is not int:
                return frozenset()
            payload = candidate.get("payload")
            payload_lines = payload.get("lines") if isinstance(payload, Mapping) else None
            if isinstance(payload_lines, list):
                return frozenset(
                    span["start"] + offset
                    for offset, value in enumerate(payload_lines)
                    if isinstance(value, str) and value.strip()
                )
            return frozenset({span["start"]})

        # Included files have their own source-line coordinate system.  Keep
        # them here and let the witness filter decide whether their mapped
        # lines are usable; root-source candidates can be pruned before
        # fragment windows are expanded.
        candidates = [
            item for item in candidates
            if item.get("source_path") or (
                candidate_lines(item) and candidate_lines(item) <= allowed_lines
            )
        ]
    if isinstance(isa, str):
        candidates = [
            item for item in candidates
            if (
                (spec := instruction_spec_for_generation(
                    item.get("payload", {}).get("mnemonic", "")
                )) is None
                or profile_enables_form(isa, spec.form)
            )
        ]
    if include_gap:
        return ()
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["candidate_id"], candidate)
    candidates = list(unique.values())
    by_line = {}
    for candidate in candidates:
        span = candidate["source_span"]
        by_line.setdefault((
            candidate.get("source_path") or "", span["start"],
            span["start_col"], span["end_col"],
        ), []).append(candidate)
    ordered = [
        (key, tuple(by_line[key]))
        for key in sorted(by_line, key=lambda item: (item[0], item[1]))
    ]
    grouped = {}
    for key, alternatives in ordered:
        grouped.setdefault(key[0], []).append((key, alternatives))
    for source_name, source_candidates in grouped.items():
        source_lines = source_lines_for(source_name or None)
        for size in (2, 3):
            for offset in range(len(source_candidates) - size + 1):
                window = source_candidates[offset:offset + size]
                if any(
                    item.get("operator") != "replace"
                    for _key, alternatives in window for item in alternatives
                ):
                    continue
                spans = [alternatives[0]["source_span"] for _key, alternatives in window]
                if any(span["start"] != span["end"] for span in spans):
                    continue
                if any(right["start"] != left["start"] + 1 for left, right in zip(spans, spans[1:])):
                    continue
                start, end = spans[0]["start"], spans[-1]["end"]
                source = "".join(source_lines[start - 1:end])
                alternatives = tuple(alternatives for _key, alternatives in window)
                # A fragment is one sequence-level mutation.  Change one
                # pivot at a time and use the first valid rewrite for the
                # other lines; a Cartesian product creates combinations that
                # have no additional witness meaning and grows multiplicatively.
                base = tuple(items[0] for items in alternatives)
                combinations = []
                for pivot, choices in enumerate(alternatives):
                    for choice in choices:
                        combination = list(base)
                        combination[pivot] = choice
                        combinations.append(tuple(combination))
                for combination in combinations:
                    block_lines = [replaced_line(item) for item in combination]
                    if any(line is None for line in block_lines):
                        continue
                    if not _fragment_is_fallthrough(block_lines):
                        continue
                    fragment = {
                        **({"source_path": source_name} if source_name else {}),
                        "source_span": {
                            "start": start, "end": end,
                            "start_col": spans[0]["start_col"], "end_col": spans[-1]["end_col"],
                        },
                        "operator": "replace_fragment",
                        "payload": {"lines": block_lines},
                        "source_text": source.rstrip("\r\n"),
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    }
                    candidates.append({**fragment, "candidate_id": rewrite_candidate_id(fragment)})
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["candidate_id"], candidate)
    result = tuple(unique.values())
    _CASE_REWRITE_CACHE[cache_key] = result
    if len(_CASE_REWRITE_CACHE) > _CASE_REWRITE_CACHE_LIMIT:
        _CASE_REWRITE_CACHE.pop(next(iter(_CASE_REWRITE_CACHE)))
    return result


def executed_source_locations(
    executable: str | Path, program: CaseProgram | str | Path,
    observations: Sequence[object],
    *, timeout_seconds: float | None = 2.0,
) -> frozenset[tuple[str, int]] | None:
    """Map executed PCs to exact source-file/line locations using DWARF rows."""
    timeout = 2.0
    if timeout_seconds is not None:
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            return None
        timeout = min(timeout, timeout_seconds)
    pcs = {
        parsed
        for item in observations
        for values in (_observation_value(item, "executed_pcs"),)
        if isinstance(values, (list, tuple))
        for value in values
        if (parsed := _pc_value(value)) is not None
    }
    if not pcs:
        return None
    source = Path(getattr(program, "program", program)).resolve()
    try:
        result = subprocess.run(
            ["riscv64-linux-gnu-objdump", "-dS", "--line-numbers", str(executable)],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    source_file: Path | None = None
    source_line: int | None = None
    locations: set[tuple[str, int]] = set()
    marker_re = re.compile(r"\s*(?P<file>.+):(?P<line>[0-9]+)\s*")
    instruction_re = re.compile(r"\s*([0-9a-f]+):\s*[0-9a-f]+\s+", re.IGNORECASE)
    for raw in result.stdout.splitlines():
        marker = marker_re.fullmatch(raw)
        if marker:
            marker_path = Path(marker["file"].replace("\\", "/"))
            if marker_path.is_absolute():
                source_file = marker_path.resolve()
            else:
                rooted = (source.parent / marker_path).resolve()
                cwd_path = marker_path.resolve()
                if rooted.exists():
                    source_file = rooted
                elif marker_path == Path("program.S"):
                    # reference probe 编译时使用的临时文件名，行号仍对应生成输入。
                    source_file = source
                elif cwd_path.exists():
                    source_file = cwd_path
                else:
                    source_file = rooted
            source_line = int(marker["line"])
            continue
        instruction = instruction_re.match(raw)
        if instruction is not None and source_file is not None and source_line is not None:
            pc = int(instruction[1], 16)
            if pc in pcs:
                locations.add((str(source_file), source_line))
    return frozenset(locations)


def prioritize_case_rewrites_for_model(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """稳定地把模型首屏候选分散到不同源码位置。"""
    groups: dict[tuple[object, object], list[Mapping[str, object]]] = {}
    for candidate in candidates:
        span = candidate.get("source_span")
        span = span if isinstance(span, Mapping) else {}
        key = (candidate.get("source_path"), span.get("start"))
        groups.setdefault(key, []).append(candidate)
    prioritized = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        prioritized.extend(
            group[index] for group in groups.values() if index < len(group)
        )
    return tuple(prioritized)


def witness_safe_case_rewrites_for_model(
    candidates: Sequence[Mapping[str, object]],
    *, executed_source_lines: Sequence[int] | set[int] | frozenset[int] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """保留 reference path 上可共同到达的单指令和多指令候选。"""
    prioritized = prioritize_case_rewrites_for_model(candidates)

    def source_lines(candidate: Mapping[str, object]) -> frozenset[int]:
        span = candidate.get("source_span")
        if not isinstance(span, Mapping) or type(span.get("start")) is not int:
            return frozenset()
        payload = candidate.get("payload")
        lines = payload.get("lines") if isinstance(payload, Mapping) else None
        if isinstance(lines, list):
            return frozenset(
                span["start"] + offset
                for offset, line in enumerate(lines)
                if isinstance(line, str) and line.strip()
            )
        return frozenset({span["start"]})

    if executed_source_lines is not None:
        executed = frozenset(executed_source_lines)
        prioritized = tuple(
            candidate for candidate in prioritized
            if source_lines(candidate) and source_lines(candidate) <= executed
        )
        if not prioritized:
            return ()

    def fragment_is_path_safe(candidate: Mapping[str, object]) -> bool:
        if candidate.get("operator") != "replace_fragment":
            return True
        payload = candidate.get("payload")
        lines = payload.get("lines") if isinstance(payload, Mapping) else None
        return isinstance(lines, list) and _fragment_is_fallthrough(lines)

    prioritized = tuple(
        candidate for candidate in prioritized if fragment_is_path_safe(candidate)
    )

    return prioritized


def _semantic_witness_contract(
    sequence: Sequence[str], description: str | None = None,
) -> dict[str, object] | None:
    names = []
    text = (description or "").lower().replace("_", ".")
    for item in sequence:
        if not isinstance(item, str):
            continue
        name = item.strip().split(None, 1)[0].lower().replace("_", ".")
        if re.fullmatch(r"[a-z][\w.]*", name) and (len(sequence) == 1 or name in text):
            names.append(name)
    if not names:
        return None
    mnemonic = names[0] if len(sequence) == 1 else names[-1]
    fields: list[str] = []
    if mnemonic in _CONTROL_REWRITES and (mnemonic.startswith("b") or mnemonic.startswith("c.b")):
        fields = ["branch_outcome", "after_pc"]
    elif mnemonic in _JUMP_MNEMONICS:
        fields = ["after_pc"] if mnemonic in {"c.j", "c.jr"} else ["after_pc", "link"]
    elif mnemonic in _MEMORY_MNEMONICS:
        fields = ["memory_reads"]
    elif mnemonic.startswith(("amo", "lr.", "sc.")):
        fields = ["memory_reads"]
    elif mnemonic == "fence.i":
        fields = ["after_pc", "final_bytes"]
    elif mnemonic in {"fence", "fence.tso", "sfence.vma"}:
        fields = ["after_pc"]
    elif mnemonic.startswith("f"):
        fields = ["after_gpr"] if mnemonic.startswith(("fclass", "fmv.x", "fcvt.w", "fcvt.l")) else [
            "after_fpr_rawbits", "after_fflags", "after_frm",
        ]
    elif mnemonic.startswith("v"):
        fields = ["vector.vstart", "vector.vl", "vector.vtype"]
    elif mnemonic.startswith(("csr", "csrr", "csrw", "csrs", "csrc", "ecall", "ebreak", "mret", "sret", "uret", "sfence")):
        fields = ["fault", "fault_pc"] if any(
            word in text for word in ("trap", "illegal", "fault")
        ) else ["after_gpr"]
    elif instruction_effect_class(mnemonic) == "op":
        fields = ["after_gpr"]
    return {"mnemonic": mnemonic, "fields": fields, "values": {}}


def _witness_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {
            name: _observation_value(value, name)
            for name in (
                "outcome", "executed_pcs", "gpr", "extra_state", "after_pc",
                "before_gpr", "after_gpr", "before_fpr_rawbits", "after_fpr_rawbits",
                "before_fflags", "after_fflags", "before_frm", "after_frm",
                "memory_reads", "branch_outcome", "fault_pc", "fault_address",
                "fault", "link", "final_bytes",
            )
            if _observation_value(value, name) is not None
        }
    extra = result.get("extra_state")
    if isinstance(extra, Mapping):
        def observed(key: str, item: object) -> bool:
            markers = {f"{key}_observer", f"{key}_observer_status"}
            if key.startswith("vector.") or key in {"vxsat", "vxrm"}:
                markers.update({"vector_observer_status", "vector_observer"})
            if key == "csr.mstatus.fs":
                markers.add("mstatus_fs_observer")
            if key in {"fflags", "frm", "csr.fflags", "csr.frm"}:
                markers.add("fp_observer")
            return item is not None and not any(
                name in extra and _observer_value_is_non_observed_marker(extra[name])
                for name in markers
            )

        result.update({
            key: item for key, item in extra.items() if observed(key, item)
        })
        result.update({
            f"extra.{key}": item for key, item in extra.items()
            if observed(key, item)
        })
        csrs = extra.get("vector_csrs")
        if isinstance(csrs, Mapping):
            result.update({
                f"vector.{key}": value for key, value in csrs.items()
                if observed(f"vector.{key}", value)
            })
            result.update({
                f"csr.{key}": value for key, value in csrs.items()
                if observed(f"csr.{key}", value)
            })
    return result


def _witness_value(values: Mapping[str, object], field: str) -> object:
    aliases = [field]
    if field.startswith("extra."):
        aliases.append(field[6:])
    if "." in field:
        aliases.append(field.split(".", 1)[1])
    if alias := {
        "after_fpr_rawbits": "fpr_rawbits", "after_fflags": "fflags",
        "after_frm": "frm", "after_gpr": "gpr",
    }.get(field):
        aliases.append(alias)
    return next((values[name] for name in dict.fromkeys(aliases) if name in values and values[name] is not None), None)


def _observer_field_value(observation: object, field: str) -> object:
    if not isinstance(field, str):
        return None
    lookup_field = field.removeprefix("extra.")
    extra = _observation_value(observation, "extra_state")
    if isinstance(extra, Mapping):
        marker_fields = {
            f"{field}_observer", f"{field}_observer_status",
            f"{lookup_field}_observer", f"{lookup_field}_observer_status",
        }
        if field.startswith("extra."):
            marker_fields.update(
                f"{field[6:]}_{suffix}" for suffix in ("observer", "observer_status")
            )
        if lookup_field.startswith("vector.") or lookup_field in {"vxsat", "vxrm"}:
            marker_fields.update({"vector_observer_status", "vector_observer"})
        if lookup_field == "csr.mstatus.fs":
            marker_fields.update({"mstatus_fs_observer", "fs_observer"})
        if lookup_field == "privilege.mode":
            marker_fields.add("csr.priv_observer")
        if lookup_field in {"fflags", "frm", "csr.fflags", "csr.frm"}:
            marker_fields.add("fp_observer")
        if lookup_field.startswith("lifecycle."):
            marker_fields.update({"lifecycle_observer", "state_observer_status"})
        if any(
            name in extra and _observer_value_is_non_observed_marker(extra[name])
            for name in marker_fields
        ):
            return None
        for name in (field, lookup_field):
            if name in extra:
                return extra[name]
    if lookup_field.startswith("memory."):
        region = lookup_field.removeprefix("memory.")
        for name in ("memory_snapshot", "memory_delta"):
            memory = _observation_value(observation, name)
            if isinstance(memory, Mapping) and region in memory:
                return memory[region]
            if region == "test-memory" and memory is not None and not isinstance(memory, Mapping):
                return memory
    values = _witness_mapping(observation)
    aliases = {
        "csr.mstatus.fs": ("mstatus_fs",),
        "privilege.mode": ("csr.priv",),
        "vxsat": ("vector.vxsat", "csr.vxsat"),
        "vxrm": ("vector.vxrm", "csr.vxrm"),
    }
    value = next(
        (values[name] for name in (*aliases.get(lookup_field, ()), lookup_field)
         if name in values and values[name] is not None),
        None,
    )
    if value is not None:
        return value
    if lookup_field.startswith(("gpr.", "gpr[")):
        match = re.fullmatch(r"gpr(?:\.x|\[)(\d+)(?:\])?", lookup_field)
        gpr = _observation_value(observation, "gpr")
        if match and isinstance(gpr, (list, tuple)) and int(match[1]) < len(gpr):
            index = int(match[1])
            return None if index in _volatile_gpr_indices(observation) else gpr[index]
    if lookup_field in {"vector.mask", "vector.tail"} and isinstance(extra, Mapping):
        if lookup_field == "vector.tail":
            vtype = extra.get("vector.vtype")
            vector_csrs = extra.get("vector_csrs")
            if vtype is None and isinstance(vector_csrs, Mapping):
                vtype = vector_csrs.get("vtype")
            return (int(vtype) >> 6) & 1 if isinstance(vtype, int) and not isinstance(vtype, bool) else None
        registers = extra.get("vector.registers", extra.get("vector_registers"))
        vector_csrs = extra.get("vector_csrs")
        if registers is None and isinstance(vector_csrs, Mapping):
            registers = vector_csrs.get("registers", vector_csrs.get("vector.registers"))
        return registers[0] if isinstance(registers, (list, tuple)) and registers else (
            registers.get("v0") if isinstance(registers, Mapping) else None
        )
    if lookup_field.startswith("lifecycle."):
        evidence = _observation_value(observation, "translation_evidence")
        details = evidence.get("details") if isinstance(evidence, Mapping) else getattr(evidence, "details", None)
        lifecycle = details.get("lifecycle") if isinstance(details, Mapping) else None
        if isinstance(lifecycle, Mapping):
            suffix = lookup_field.removeprefix("lifecycle.")
            return next((lifecycle[name] for name in (field, suffix, suffix.replace("-", "_"))
                         if name in lifecycle and lifecycle[name] is not None), None)
    if lookup_field == "translation.path_identity":
        evidence = _observation_value(observation, "translation_evidence")
        details = evidence.get("details") if isinstance(evidence, Mapping) else getattr(evidence, "details", None)
        return details.get("path_identity") if isinstance(details, Mapping) else None
    direct = _observation_value(observation, lookup_field, None)
    if direct is not None:
        return direct
    return None


def _observer_validation_field(field: str) -> str:
    field = field.removeprefix("extra.")
    match = re.fullmatch(r"gpr(?:\.x|\[)(\d+)(?:\])?", field)
    if match and int(match[1]) < 32:
        return f"gpr.x{int(match[1])}"
    return field


# RISC-V 压缩与非压缩指令是同一语义的两种编码：rv64imc 下源码里的 `li`
# 会被汇编器压成 `c.li`，witness 只比字面助记符就永远找不到已执行的行。
def _mnemonic_matches(row_mnemonic: object, wanted: str) -> bool:
    left = str(row_mnemonic).lower().replace("_", ".")
    right = str(wanted).lower().replace("_", ".")
    if left == right:
        return True
    if left.startswith("c."):
        left = left[2:]
    if right.startswith("c."):
        right = right[2:]
    return left == right


def _semantic_witness_realizes(
    profile: object, witness: object, observations: Sequence[object] = (),
) -> bool:
    if witness is None or isinstance(witness, str):
        return True
    if not isinstance(witness, Mapping):
        return False
    mnemonic = witness.get("mnemonic")
    fields = witness.get("fields", ())
    values = witness.get("values", {})
    if (not isinstance(mnemonic, str) or not mnemonic.strip()
            or not isinstance(fields, (list, tuple))
            or any(not isinstance(field, str) or not field.strip() for field in fields)
            or not isinstance(values, Mapping)
            or any(not isinstance(field, str) for field in values)):
        return False
    rows = tuple(getattr(profile, "instructions", ()))
    occurrences = tuple(getattr(profile, "occurrences", ()))
    observation_values = tuple(_witness_mapping(item) for item in observations)
    vector_fields = {"vector.vstart", "vector.vl", "vector.vtype"}
    terminal_fields = {
        "after_gpr", "after_fpr_rawbits", "after_fflags", "after_frm", "final_bytes",
    }
    fp_terminal_fields = {"after_fpr_rawbits", "after_fflags", "after_frm"}
    def writes_fp_control(row: Mapping[str, object]) -> bool:
        mnemonic = str(row.get("mnemonic", "")).lower().replace("_", ".")
        operands = tuple(str(item).lower() for item in row.get("operands", ()))
        if mnemonic.startswith(("fsrm", "fsflags")):
            return True
        if mnemonic in {"csrw", "csrs", "csrc", "csrwi", "csrsi", "csrci"}:
            return True
        if mnemonic in {"csrrw", "csrrs", "csrrc"}:
            return len(operands) > 2 and operands[2] not in {"x0", "zero"}
        return False

    def final_fp_state_allowed(index: int) -> bool:
        return not any(
            isinstance(item, Mapping)
            and item.get("executed") is True
            and (
                (item.get("rule_facts", {}).get("effect") == "fpr"
                 if isinstance(item.get("rule_facts"), Mapping) else False)
                or (
                    not isinstance(item.get("rule_facts"), Mapping)
                    and str(item.get("mnemonic", "")).lower().startswith("f")
                )
                or writes_fp_control(item)
            )
            for item in rows[index + 1:]
        )

    wanted = mnemonic.lower().replace("_", ".")
    for index, row in enumerate(rows):
        if (not isinstance(row, Mapping) or row.get("executed") is not True
                or not _mnemonic_matches(row.get("mnemonic", ""), wanted)):
            continue
        global_fields = vector_fields | {
            "executed_pcs", "fault_pc", "fault_address",
            "trap.cause", "trap.epc", "trap.tval",
        } | (
            terminal_fields
            if not any(
                isinstance(item, Mapping) and item.get("executed") is True
                for item in rows[index + 1:]
            )
            else fp_terminal_fields
            if final_fp_state_allowed(index) else set()
        )
        observed = _witness_mapping(row)
        observed["executed"] = True
        row_occurrences = tuple(
            item for item in occurrences
            if isinstance(item, Mapping) and item.get("instruction_id") == row.get("id")
        )
        for occurrence in row_occurrences or (None,):
            current = dict(observed)
            if occurrence is not None:
                current.update({key: value for key, value in occurrence.items() if value is not None})
            if not (all(_witness_value(current, field) is not None or field in global_fields and any(
                _witness_value(item, field) is not None for item in observation_values
            ) for field in fields) and all(
                next((value for item in (current, *observation_values)
                      if item is current or field in global_fields
                      if (value := _witness_value(item, field)) is not None), None) == expected
                for field, expected in values.items()
            )):
                break
        else:
            return True
    return False


def rewrite_program(
    program: CaseProgram,
    hint: Mapping[str, object],
    output_path: str | Path,
    *,
    candidates: Sequence[Mapping[str, object]] | None = None,
) -> CaseProgram:
    """把一份已枚举的结构化改写计划确定性地落到完整 P0 源码。"""
    if not isinstance(hint, Mapping):
        raise ValueError("generation-gap: rewrite hint must be an object")
    path, lines = _rewrite_source(program)
    output = _path_in_root(output_path, Path(output_path).parent, "rewrite-output")
    if output == path:
        raise ValueError("generation-gap: rewrite output equals source")
    candidates = (
        tuple(enumerate_case_rewrites(program))
        if candidates is None else tuple(candidates)
    )
    if any(
        not isinstance(candidate, Mapping)
        or not isinstance(candidate.get("candidate_id"), str)
        or candidate.get("candidate_id") != rewrite_candidate_id(candidate)
        for candidate in candidates
    ):
        raise ValueError("generation-gap: rewrite candidate pool is invalid")
    candidate_ids = tuple(candidate["candidate_id"] for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("generation-gap: rewrite candidate pool has duplicate IDs")
    provenance = hint.get("provenance")
    if (not isinstance(provenance, Mapping)
            or provenance.get("raw_sha256") != program_sha256(program)
            or provenance.get("candidate_pool_digest") != canonical_digest(list(candidates))):
        raise ValueError("generation-gap: rewrite provenance does not match source pool")
    edits = rewrite_hint_edits(hint)
    if not edits:
        raise ValueError("generation-gap: rewrite hint has no edits")
    if any(not isinstance(edit, Mapping) for edit in edits):
        raise ValueError("generation-gap: rewrite edit must be an object")
    matches = []
    for edit in edits:
        core = {key: edit.get(key) for key in ("source_span", "operator", "payload")}
        if "source_path" in edit:
            core["source_path"] = edit["source_path"]
        candidate_id = rewrite_candidate_id(core)
        match = next(
            (
                item for item in candidates
                if (
                    rewrite_candidate_id(item) == candidate_id
                    and edit.get("candidate_id") in (None, candidate_id)
                )
            ),
            None,
        )
        if match is None:
            raise ValueError("generation-gap: rewrite hint is not an enumerated candidate")
        for name in ("source_text", "source_sha256"):
            if name in edit and edit[name] != match.get(name):
                raise ValueError("generation-gap: rewrite source identity mismatch")
        matches.append(match)
    ordered = sorted(matches, key=lambda item: (
        item.get("source_path", ""),
        item["source_span"]["start"], item["source_span"]["start_col"],
    ))
    if any(
        (right["source_span"]["start"], right["source_span"]["start_col"])
        < (left["source_span"]["end"], left["source_span"]["end_col"])
        for left, right in zip(ordered, ordered[1:])
        if right.get("source_path") == left.get("source_path")
    ):
        raise ValueError("generation-gap: rewrite edits overlap")
    changed: dict[str, list[str]] = {}

    def edit_lines(match):
        source_name = match.get("source_path")
        if not isinstance(source_name, str) or not source_name:
            return lines, None
        target = _path_in_root(path.parent / source_name, path.parent, "source-dependency")
        if source_name not in changed:
            changed[source_name] = target.read_bytes().decode("utf-8").splitlines(keepends=True)
        return changed[source_name], source_name

    for match in reversed(ordered):
        span = match["source_span"]
        payload = match["payload"]
        target_lines, source_name = edit_lines(match)
        if match.get("operator") == "replace_fragment":
            if (not isinstance(span, Mapping) or not isinstance(payload, Mapping)
                    or type(span.get("start")) is not int or type(span.get("end")) is not int
                    or span["end"] < span["start"]
                    or not 1 <= span["start"] <= span["end"] <= len(target_lines)
                    or not isinstance(payload.get("lines"), list)
                    or len(payload["lines"]) != span["end"] - span["start"] + 1
                    or any(not isinstance(line, str) for line in payload["lines"])):
                raise ValueError("generation-gap: invalid rewrite fragment")
            original_block = target_lines[span["start"] - 1:span["end"]]
            if match.get("source_sha256") not in (
                None, hashlib.sha256("".join(original_block).encode()).hexdigest(),
            ):
                raise ValueError("generation-gap: rewrite source identity mismatch")
            target_lines[span["start"] - 1:span["end"]] = [
                value + original[len(original.rstrip("\r\n")):]
                for value, original in zip(payload["lines"], original_block)
            ]
            continue
        if (not isinstance(span, Mapping) or not isinstance(payload, Mapping)
                or span.get("start") != span.get("end")
                or not isinstance(span.get("start"), int)):
            raise ValueError("generation-gap: invalid rewrite span")
        index = span["start"] - 1
        if not 0 <= index < len(target_lines):
            raise ValueError("generation-gap: rewrite span is out of range")
        original = target_lines[index]
        source_line = original.rstrip("\r\n")
        start_col = span.get("start_col", 0)
        end_col = span.get("end_col", len(source_line))
        if (type(start_col) is not int or type(end_col) is not int
                or not 0 <= start_col < end_col <= len(source_line)):
            raise ValueError("generation-gap: rewrite span is out of range")
        segment = source_line[start_col:end_col]
        if match.get("source_sha256") not in (None, hashlib.sha256(segment.encode()).hexdigest()):
            raise ValueError("generation-gap: rewrite source identity mismatch")
        parsed = _INSTRUCTION_RE.fullmatch(segment)
        if parsed is None:
            raise ValueError("generation-gap: rewrite source span is not instruction")
        newline = original[len(source_line):]
        replacement = f"{parsed.group('indent')}{payload['mnemonic']} {', '.join(payload['operands'])}"
        target_lines[index] = (
            source_line[:start_col] + replacement + (parsed.group("comment") or "")
            + source_line[end_col:] + newline
        )
    if changed and output.parent.resolve() == path.parent.resolve():
        raise ValueError("generation-gap: rewrite output overlaps source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes("".join(lines).encode("utf-8"))
    _copy_includes(output, path.parent, output.parent)
    for source_name, source_lines in changed.items():
        target = output.parent / source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes("".join(source_lines).encode("utf-8"))
    return CaseProgram(output, _program_params(program))


_OBSERVATION_FIELDS = (
    "outcome", "exit_code", "checkpoint_pc", "gpr", "memory_snapshot",
    "memory_digest", "signal", "signal_code", "fault_pc", "fault_address",
    "extra_state", "state_trace", "memory_delta", "executed_pcs", "instruction_count",
    "trace_complete",
)
_SEMANTIC_OBSERVATION_FIELDS = frozenset(_OBSERVATION_FIELDS) - {
    # PC/trace 字段证明执行完整性；等价插入可能改变它们。
    "checkpoint_pc", "state_trace", "executed_pcs", "extra_state",
    "instruction_count", "trace_complete",
}
_OBSERVED_OUTCOMES = frozenset({"normal", "completed", "trap", "nonzero-exit"})
_REFERENCE_IDENTITY_FIELDS = ("backend", "profile_id", "input_id", "tool_version")
_REFERENCE_PROVENANCE_FIELDS = frozenset({
    "backend", "source_repository", "source_commit", "binary_sha256",
    "runtime_binary_sha256", "runner_sha256", "container_image_digest",
    "identity_digest", "isa", "machine", "kernel", "host_key_sha256",
    "helper_sha256", "schema_version", "profile_id", "input_id", "tool_version",
})


def _reference_provenance(observation: object) -> dict[str, object]:
    """Collect the reference identity regardless of its transport location."""
    result: dict[str, object] = {}
    conflict = False

    def merge(value: object) -> None:
        nonlocal conflict
        if not isinstance(value, Mapping):
            return
        for name in _REFERENCE_PROVENANCE_FIELDS:
            item = value.get(name)
            if item in (None, ""):
                continue
            if name in result and result[name] != item:
                conflict = True
            result[name] = item

    merge(_observation_value(observation, "reference_identity"))
    extra = _observation_value(observation, "extra_state")
    if isinstance(extra, Mapping):
        merge(extra.get("reference_identity"))
    evidence = _observation_value(observation, "translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else getattr(
        evidence, "details", None,
    )
    if isinstance(details, Mapping):
        merge(details.get("native_reference_identity"))
    else:
        merge(getattr(details, "native_reference_identity", None))
    if conflict:
        result["__identity_conflict__"] = True
    return result


def _is_declared_qemu_reference_fallback(observation: object) -> bool:
    extra = _observation_value(observation, "extra_state")
    fallback = extra.get("reference_fallback") if isinstance(extra, Mapping) else None
    fallback_reason = fallback.get("fallback_reason") if isinstance(fallback, Mapping) else None
    return (
        isinstance(fallback, Mapping)
        and fallback.get("schema_version") == "rq1-reference-fallback-v1"
        and qemu_fallback_policy_matches(
            fallback.get("policy"), fallback_reason,
            K1_QEMU_FALLBACK_FAILURE_CLASSES,
        )
        and fallback.get("preferred_backend") == "native-rv64"
        and fallback.get("effective_backend") == "qemu-riscv64"
        and fallback.get("selection_scope") == "whole-campaign"
        and fallback.get("target_id") == "T-QEMU"
        and is_sha256_digest(fallback.get("k1_attempt_digest"))
        and _observation_value(observation, "backend") == "qemu-riscv64"
    )


def _gpr_index(field: str) -> int | None:
    field = field.removeprefix("extra.")
    match = re.fullmatch(r"gpr(?:\.x|\[)(\d+)(?:\])?", field)
    if match is None:
        return None
    index = int(match[1])
    return index if index < 32 else None


def _observation_has(observation: object, name: str) -> bool:
    return name in observation if isinstance(observation, Mapping) else hasattr(observation, name)


def _result_extra_state(observation: object) -> dict[str, object]:
    raw = _observation_value(observation, "extra_state", {})
    if raw is not None and not isinstance(raw, Mapping):
        return {}
    raw = _strip_observation_side_channels(raw or {})
    return {
        key: _comparable(value)
        for key, value in dict(raw or {}).items()
        if isinstance(key, str)
        if key != "observer_fields"
        and key != "state_trace"
        and not any(
            key == field or key.startswith(f"{field}_")
            for field in _VOLATILE_EXTRA_STATE_FIELDS
        )
        and not key.startswith("sail_")
        and not key.endswith(("_observer", "_observer_status", "_gap"))
    }


def _volatile_gpr_indices(observation: object) -> frozenset[int]:
    extra = _observation_value(observation, "extra_state", {})
    values = extra.get("volatile_gpr_indices", ()) if isinstance(extra, Mapping) else ()
    return frozenset(value for value in values if type(value) is int and 0 <= value < 32)


def _normalized_gpr(observation: object, value: object) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    volatile = _volatile_gpr_indices(observation)
    return tuple(None if index in volatile else item for index, item in enumerate(value))


def _normalized_trace(observation: object) -> object:
    trace = _trace_entries(observation)
    volatile = _volatile_gpr_indices(observation)
    if not volatile:
        return trace
    result = []
    for item in trace:
        row = dict(item)
        for name in ("before_gpr", "after_gpr"):
            value = row.get(name)
            if isinstance(value, (list, tuple)):
                row[name] = tuple(None if index in volatile else value[index] for index in range(len(value)))
        result.append(row)
    return tuple(result)


def _comparable(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    return tuple(_comparable(item) for item in value) if isinstance(value, (list, tuple)) else value


def _comparison_value(name: str, value: object) -> object:
    value = _comparable(value)
    if name == "outcome" and value in {"normal", "completed"}:
        return "normal"
    if name in {"checkpoint_pc", "fault_pc", "fault_address"}:
        return _pc_value(value) if value is not None else None
    if name == "executed_pcs" and isinstance(value, tuple):
        return tuple(_pc_value(item) for item in value)
    if name in {"memory_snapshot", "memory_delta"}:
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, Mapping):
            return {
                key: item.lower() if isinstance(item, str) else item
                for key, item in value.items()
            }
    return value


def _comparison_outcome(observation: object) -> object:
    outcome = _observation_value(observation, "outcome")
    extra = _observation_value(observation, "extra_state")
    return "normal" if outcome == "trap" and isinstance(extra, Mapping) \
        and terminal_ebreak_observed(dict(extra)) else outcome


def _cross_backend_executed_pcs(observation: object) -> object:
    extra = _observation_value(observation, "extra_state")
    marker = extra.get("rvgen_risk_execution") if isinstance(extra, Mapping) else None
    if isinstance(marker, Mapping):
        pcs = marker.get("risk_pcs")
        actual = tuple(
            _pc_value(item)
            for item in (_observation_value(observation, "executed_pcs") or ())
        )
        risk = tuple(_pc_value(item) for item in pcs) if isinstance(pcs, (list, tuple)) else ()
        if (
            marker.get("status") != "observed"
            or not risk
            or len(set(risk)) != len(risk)
            or any(item is None for item in risk)
            or not all(item in actual for item in risk)
        ):
            return None
        return ("rvgen-risk-pc", "observed", len(risk))
    pcs = tuple(_pc_value(item) for item in (_observation_value(observation, "executed_pcs") or ()))
    if not pcs or any(item is None for item in pcs):
        return pcs
    marker = extra.get("program_test_path") if isinstance(extra, Mapping) else None
    if isinstance(marker, Mapping):
        start, end = marker.get("start_pc"), marker.get("end_pc")
        body = tuple(item for item in pcs if type(start) is int and type(end) is int and start <= item < end)
        declared = tuple(_pc_value(item) for item in marker.get("executed_pcs", ()))
        if (
            marker.get("contract") != "program-test-path-v1"
            or marker.get("status") != "observed"
            or type(start) is not int or type(end) is not int or not start < end
            or not body or body != declared
        ):
            return None
        return tuple(item - start for item in body)
    # ponytail: 跨后端只比较路径形状；首地址属于各自装载布局，未来若需绝对地址再由显式 witness 约束。
    return tuple(item - pcs[0] for item in pcs)


def _program_path_unavailable(observation: object) -> bool:
    """Return whether this backend explicitly lacks a program PC trace."""
    if _observation_value(observation, "backend") not in {
        "rax-riscv64", "rvvm-riscv64",
    }:
        return False
    evidence = _observation_value(observation, "translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    path = details.get("path_identity") if isinstance(details, Mapping) else None
    return (
        isinstance(path, Mapping)
        and path.get("contract") == PATH_IDENTITY_WITNESS_CONTRACT
        and path.get("status") == "unavailable"
    )


def _reference_gap(differences: dict[str, object]) -> dict[str, object]:
    return {"status": "reference-gap", "differences": differences}


def _reference_failure_status(contract_error: object) -> str:
    return (
        "transport-gap"
        if contract_error == "native-reference-lock-timeout"
        else "reference-gap"
    )


def _reference_exception_status(error: Exception) -> str:
    # 本地参考机无法建模某个形式或控制流时，这是 reference gap，不是
    # 语义不等价；开放能力链继续搜索，并在链结束后回放候选到 Target。
    message = str(error)
    if isinstance(error, ValueError) and message.startswith((
        "rvgen-semantic invalid branch target:",
        "rvgen-semantic invalid jump target:",
        "rvgen-semantic invalid indirect jump target:",
        "rvgen-semantic control-flow loop",
        "rvgen-semantic unsupported form:",
    )):
        return "reference-gap"
    return "profile-gap"


def _pair_result(status, reference, target=None, artifacts=None, failure_class=None):
    result = {"status": status, "reference": reference, "target": target, "artifacts": artifacts or {}}
    if failure_class is not None:
        result["failure_class"] = failure_class
    return result


def _record_has_qualification_gap(record: Mapping[str, object]) -> bool:
    return any(record.get(name) for name in (
        "comparison_qualification_gaps", "qualification_gaps", "missing_fields",
        "validation_gap", "contract_error", "target_execution_contract_error",
    ))


def _comparison_qualified(
    record: Mapping[str, object], expected_status: str | None = None,
) -> bool:
    comparison = record.get("comparison")
    status = comparison.get("status") if isinstance(comparison, Mapping) else None
    candidate = record.get("candidate")
    record_status = record.get("status")
    candidate_status = isinstance(record_status, str) and record_status in {
        "target-mismatch-candidate", "target-state-mismatch-candidate",
    }
    return bool(
        record.get("comparison_qualified") is True
        and (
            candidate is None
            or type(candidate) is bool and candidate is candidate_status
        )
        and not _record_has_qualification_gap(record)
        and isinstance(comparison, Mapping)
        and isinstance(status, str)
        and status in {"equivalent", "non-equivalent"}
        and (expected_status is None or status == expected_status)
        and not _record_has_qualification_gap(comparison)
    )


def _execution_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("reference_identity")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    result.update({
        name: value[name]
        for name in ("backend", "tool_version", "target_binary_sha256", "target_identity_digest")
        if value.get(name) not in (None, "")
    })
    containers, payloads = _target_identity_parts(value)
    for container in containers:
        for name in ("target_binary_sha256", "target_identity_digest"):
            if container.get(name) not in (None, ""):
                result[name] = container[name]
    for target in payloads:
        for source, destination in (
            ("binary_sha256", "target_binary_sha256"),
            ("identity_digest", "target_identity_digest"),
        ):
            if target.get(source) not in (None, ""):
                result[destination] = target[source]
    return result


def compare_observations(
    original: list[object] | tuple[object, ...],
    variant: list[object] | tuple[object, ...],
    observer_fields: Sequence[str] | None = None,
    *,
    cross_backend: bool = False,
    observer_fields_by_side: tuple[Sequence[str], Sequence[str]] | None = None,
) -> dict[str, object]:
    if type(cross_backend) is not bool:
        return _reference_gap({"cross_backend": "invalid"})
    if not original or not variant or len(original) != len(variant):
        return _reference_gap({"count": (len(original), len(variant))})

    if observer_fields_by_side is not None:
        if (
            observer_fields is not None
            or not isinstance(observer_fields_by_side, (list, tuple))
            or len(observer_fields_by_side) != 2
        ):
            return _reference_gap({"observer_fields_by_side": "invalid"})
        if any(
            not isinstance(fields, Sequence)
            or isinstance(fields, (str, bytes))
            or any(not isinstance(field, str) or not field.strip() for field in fields)
            for fields in observer_fields_by_side
        ):
            return _reference_gap({"observer_fields_by_side": "invalid"})
        side_fields = tuple(
            tuple(dict.fromkeys(
                field for field in fields
                if not _is_observation_side_channel_field(field)
            ))
            for fields in observer_fields_by_side
        )
        observer_fields = tuple(dict.fromkeys((
            "outcome", *side_fields[0], *side_fields[1],
        )))
    else:
        side_fields = None
        if observer_fields is not None and (
            not isinstance(observer_fields, Sequence)
            or isinstance(observer_fields, (str, bytes))
            or any(not isinstance(field, str) or not field.strip() for field in observer_fields)
        ):
            return _reference_gap({"observer_fields": "invalid"})

    requested_fields = None if observer_fields is None else tuple(dict.fromkeys(
        field for field in observer_fields
        if not _is_observation_side_channel_field(field)
    ))
    if requested_fields == ():
        return _reference_gap({"observer_fields": "empty"})

    def identity(item: object) -> dict[str, object]:
        result = {
            name: _observation_value(item, name)
            for name in _REFERENCE_IDENTITY_FIELDS
            if _observation_has(item, name) and _observation_value(item, name) is not None
        }
        provenance = _reference_provenance(item)
        if provenance.get("__identity_conflict__"):
            result["__identity_conflict__"] = True
        result.update({
            name: value for name, value in provenance.items()
            if name != "__identity_conflict__"
        })
        nested = _observation_value(item, "reference_identity")
        if isinstance(nested, Mapping):
            nested = {
                name: value for name, value in nested.items()
                if value is not None
                and name not in {"binary_sha256", "guest_elf_sha256", "executable_sha256"}
                and not _is_observation_side_channel_field(name)
            }
            if any(
                name in result and name in nested and result[name] != nested[name]
                for name in _REFERENCE_IDENTITY_FIELDS
            ):
                result["__identity_conflict__"] = True
            result.update(nested)
        return result

    def has(item: object, name: str) -> bool:
        return _observation_has(item, name) or (
            name in {"memory_snapshot", "memory_delta"}
            and (
                _observation_has(item, "memory_snapshot")
                or _observation_has(item, "memory_delta")
            )
        )

    def raw_value(item: object, name: str) -> object:
        if name == "outcome":
            return _comparison_outcome(item)
        if name == "executed_pcs" and cross_backend:
            return _cross_backend_executed_pcs(item)
        if name == "state_trace":
            return _normalized_trace(item) if _observation_value(item, name) is not None else None
        if name in {"memory_snapshot", "memory_delta"}:
            return (
                _observation_value(item, "memory_snapshot")
                if _observation_has(item, "memory_snapshot")
                else _observation_value(item, "memory_delta")
            )
        if name.startswith("memory.") or name not in _OBSERVATION_FIELDS:
            return _observer_field_value(item, name)
        if name == "gpr":
            return _normalized_gpr(item, _observation_value(item, name))
        return _observation_value(item, name)

    def valid_value(name: str, value: object) -> bool:
        validation_field = _observer_validation_field(name)
        if name == "outcome":
            return isinstance(value, str) and value in _OBSERVED_OUTCOMES
        if name in {"memory_snapshot", "memory_delta"}:
            return _profile_memory_valid(value)
        if name == "memory_digest":
            return is_sha256_digest(value)
        if name == "state_trace":
            return isinstance(value, (list, tuple, Mapping))
        return _observer_value_is_valid(validation_field, value)

    differences: list[dict[str, object]] = []
    missing_fields: list[dict[str, object]] = []
    qualification_gaps: list[dict[str, object]] = []
    for index, (left, right) in enumerate(zip(original, variant)):
        left_identity, right_identity = identity(left), identity(right)
        if (
            not left_identity.get("backend")
            or left_identity.get("__identity_conflict__")
            or not right_identity.get("backend")
            or right_identity.get("__identity_conflict__")
        ):
            qualification_gaps.append({"index": index, "identity": "backend-missing"})
        elif not cross_backend and left_identity != right_identity:
            qualification_gaps.append({"index": index, "identity": "mismatch"})
        if cross_backend:
            left_input = _observation_value(left, "input_id")
            right_input = _observation_value(right, "input_id")
            if not isinstance(left_input, str) or not left_input.strip():
                qualification_gaps.append({"index": index, "identity": "input-missing"})
            elif left_input != right_input:
                qualification_gaps.append({"index": index, "identity": "input-mismatch"})

        fields = requested_fields
        if fields is None:
            fields = tuple(
                sorted(
                    name for name in _SEMANTIC_OBSERVATION_FIELDS
                    if has(left, name) and has(right, name)
                )
            )
        if not fields:
            missing_fields.append({"index": index, "missing_fields": "no-common-result-field"})
            continue

        left_values: dict[str, object] = {}
        right_values: dict[str, object] = {}
        invalid = []
        compared_fields = []
        for name in fields:
            gpr_index = _gpr_index(name)
            if (
                gpr_index is not None
                and gpr_index in _volatile_gpr_indices(left)
                and gpr_index in _volatile_gpr_indices(right)
            ):
                continue
            compared_fields.append(name)
            left_value = raw_value(left, name)
            right_value = raw_value(right, name)
            left_present = left_value is not None and valid_value(name, left_value)
            right_present = right_value is not None and valid_value(name, right_value)
            if left_present:
                left_values[name] = _comparison_value(name, left_value)
            if right_present:
                right_values[name] = _comparison_value(name, right_value)
            if not left_present or not right_present:
                invalid.append(name)

        changed, missing = compare_fields(left_values, right_values, compared_fields)
        missing = tuple(dict.fromkeys((*missing, *invalid)))
        if changed:
            differences.append({"index": index, "fields": changed})
        if missing:
            missing_fields.append({"index": index, "missing_fields": missing})

    status = (
        "non-equivalent" if differences else
        "reference-gap" if missing_fields else
        "equivalent"
    )
    result: dict[str, object] = {
        "status": status,
        "differences": differences,
    }
    if missing_fields:
        result["missing_fields"] = missing_fields
    if qualification_gaps:
        result["qualification_gaps"] = qualification_gaps
    if cross_backend:
        result["comparison_mode"] = "cross-backend"
    return result

def _observations(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


_TARGET_UNSUPPORTED_TEXT = (
    "unsupported isa", "unsupported-isa", "unsupported extension",
    "unsupported instruction", "unsupported-instruction", "unsupported opcode",
    "unimplemented instruction", "unimplemented-instruction", "illegal opcode",
    "illegal instruction", "illegal-instruction", "unknown instruction",
    "unknown-instruction",
)
_TARGET_UNSUPPORTED_KEYS = frozenset({
    "target_unsupported", "unsupported_isa", "unsupported_extension",
    "unsupported_instruction", "guest_zbb_unsupported",
})


def _target_unsupported_reason(
    value: object, *, expected_trap: bool = False,
) -> str | None:
    """Recognize a pre-execution refusal without hiding a guest result."""
    if isinstance(value, BaseException):
        text = str(value).strip().lower()
        if not expected_trap and any(marker in text for marker in _TARGET_UNSUPPORTED_TEXT):
            return text or type(value).__name__
        return None
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:
            value = None
    if not isinstance(value, Mapping):
        return None
    if expected_trap:
        return None
    root = value
    nested = root.get("observation")
    observation = nested if isinstance(nested, Mapping) else root
    extra = observation.get("extra_state")
    outcome = observation.get(
        "outcome", observation.get("terminal", observation.get("semantic_outcome"))
    )
    if isinstance(outcome, str) and outcome in {
        "normal", "completed", "complete_observation", "trap",
        "expected-trap", "nonzero-exit", "crash", "timeout",
    }:
        return None
    if isinstance(extra, Mapping) and (
        extra.get("guest_trap") == "delivered"
        or type(extra.get("guest_cause")) is int
        or type(extra.get("trap.cause")) is int
    ):
        return None
    if (
        observation.get("guest_trap") == "delivered"
        or type(observation.get("guest_cause")) is int
        or type(observation.get("trap.cause")) is int
    ):
        return None
    root_extra = root.get("extra_state")
    for source in (extra, root_extra, observation, root):
        if not isinstance(source, Mapping):
            continue
        for key in _TARGET_UNSUPPORTED_KEYS:
            marker = source.get(key)
            if marker not in (False, None, "", 0):
                return f"{key}:{marker}" if marker is not True else key
    reason = observation.get("reason_code", root.get("reason_code"))
    if reason == "unsupported-isa":
        return str(reason)
    return None


def _observation_outcome_valid(observation: object) -> bool:
    outcome, exit_code = _observation_value(observation, "outcome"), _observation_value(
        observation, "exit_code"
    )
    return (
        outcome in {"normal", "completed"} and type(exit_code) is int and exit_code == 0
        or outcome == "nonzero-exit" and type(exit_code) is int and exit_code != 0
        or outcome == "trap" and (exit_code is None or type(exit_code) is int)
        or outcome in {"timeout", "unavailable"}
        and (exit_code is None or type(exit_code) is int)
    )


def _observation_record(observation: object) -> dict[str, object]:
    if hasattr(observation, "to_dict"):
        value = observation.to_dict()
    elif isinstance(observation, Mapping):
        value = dict(observation)
    else:
        value = {
            name: getattr(observation, name)
            for name in _OBSERVATION_FIELDS + (
                "backend", "checkpoint_pc", "executed_pcs", "instruction_count",
                "contract_error", "profile_id", "input_id", "tool_version",
                "reference_identity", "binary_sha256", "translation_evidence", "state_trace",
                "memory_delta", "guest_elf_sha256",
            )
            if hasattr(observation, name)
        }
    if "memory_snapshot" not in value and "memory_delta" in value:
        value["memory_snapshot"] = value["memory_delta"]
    for name in ("memory_snapshot", "memory_delta"):
        if isinstance(value.get(name), (bytes, bytearray)):
            value[name] = bytes(value[name]).hex()
    value["translation_evidence"] = _evidence_record(value.get("translation_evidence"))
    safe = _json_safe(value)
    return safe if isinstance(safe, dict) else {}


def _evidence_record(evidence: object) -> object:
    if hasattr(evidence, "to_dict"):
        evidence = evidence.to_dict()
    if isinstance(evidence, Mapping):
        return {str(key): _evidence_record(value) for key, value in evidence.items()}
    if isinstance(evidence, (list, tuple)):
        return [_evidence_record(value) for value in evidence]
    return {
        str(key): _evidence_record(value)
        for key, value in vars(evidence).items()
    } if hasattr(evidence, "__dict__") else evidence


def _artifact_record(artifact: BuiltProgram) -> dict[str, object]:
    linux_path = artifact.linux_executable_path
    linux_sha256 = artifact.linux_executable_sha256
    linux_status = artifact.linux_materialization_status
    linux_reason = artifact.linux_materialization_reason
    if linux_path is None and linux_status == "not-requested":
        # Preserve the legacy six-argument BuiltProgram contract used by
        # older fixtures and replay records: executable_path was the
        # Linux-user projection before the explicit field was added.
        linux_path = artifact.executable_path
        linux_sha256 = artifact.executable_sha256
        linux_status = "built"
    strategy = (
        "reuse-existing-compatible-elf"
        if artifact.bare_materialization_status == "reused"
        else "rebuild-from-source-with-target-harness"
    )
    return {
        "source_path": str(artifact.source_path),
        "executable_path": str(artifact.executable_path),
        "source_sha256": artifact.source_sha256,
        "executable_sha256": artifact.executable_sha256,
        "linux_executable_path": (
            str(linux_path) if linux_path is not None else None
        ),
        "linux_executable_sha256": linux_sha256,
        "bare_executable_path": (
            str(artifact.bare_executable_path)
            if artifact.bare_executable_path is not None else None
        ),
        "bare_executable_sha256": artifact.bare_executable_sha256,
        "materialization": {
            "contract": "rq1-source-projection-v1",
            "strategy": strategy,
            "source_sha256": artifact.source_sha256,
            "linux_user": {
                "status": linux_status,
                "path": (
                    str(linux_path) if linux_path is not None else None
                ),
                "sha256": linux_sha256,
                **({"reason": linux_reason} if linux_reason else {}),
            },
            "bare_metal": {
                "status": artifact.bare_materialization_status,
                "path": (
                    str(artifact.bare_executable_path)
                    if artifact.bare_executable_path is not None else None
                ),
                "sha256": artifact.bare_executable_sha256,
                **({"reason": artifact.bare_materialization_reason}
                   if artifact.bare_materialization_reason else {}),
            },
        },
        "parent_sha256": artifact.parent_sha256,
        "run_params": dict(artifact.run_params),
    }


def _build_cache_key(
    program: Program, builder: BuilderCallback,
    bare_builder: BuilderCallback | None,
) -> tuple[object, ...] | None:
    try:
        return (
            program_sha256(program),
            getattr(program, "parent_sha256", None),
            canonical_digest(_program_params(program)),
            id(builder), id(bare_builder),
        )
    except (OSError, TypeError, ValueError):
        return None


def target_risk_execution_contract_error(
    record: Mapping[str, object], expected_identity: Mapping[str, object] | None = None,
) -> str | None:
    if not isinstance(expected_identity, Mapping) or expected_identity.get("route") != "single":
        return None
    extra = record.get("extra_state")
    marker = extra.get("rvgen_risk_execution") if isinstance(extra, Mapping) else None
    evidence = record.get("translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    path_identity = details.get("path_identity") if isinstance(details, Mapping) else None
    path_status = path_identity.get("status") if isinstance(path_identity, Mapping) else None
    rax_without_trace = (
        record.get("backend") == "rax-riscv64"
        and isinstance(details, Mapping)
        and "trace_mechanism" in details
        and details.get("trace_mechanism") is None
    )
    rvvm_mailbox_only = (
        record.get("backend") == "rvvm-riscv64"
        and isinstance(details, Mapping)
        and details.get("rvvm_trace_mechanism") == "gdb-rsp-mailbox-only"
    )
    no_pc_channel = (
        not record.get("executed_pcs")
        and isinstance(details, Mapping)
        and details.get("trace_available") is False
        and path_status == "unavailable"
        and (rax_without_trace or rvvm_mailbox_only)
    )
    path_unavailable = (
        no_pc_channel
        and isinstance(marker, Mapping)
        and marker.get("status") == "unavailable"
        and marker.get("reason") == "backend-path-unavailable"
    )
    if not isinstance(marker, Mapping):
        return "risk-execution-not-observed"
    try:
        risk_pcs = tuple(_pc_value(value) for value in marker.get("risk_pcs", ()))
    except (TypeError, ValueError):
        return "invalid-risk-execution-pcs"

    def expected_trap_pc_error() -> str | None:
        if expected_identity.get("expected_trap") is not True:
            return None
        if terminal_ebreak_observed(extra, risk_pcs=risk_pcs):
            return None
        if _observation_value(record, "outcome") != "trap" \
                and not guest_trap_observed(extra):
            return None
        if not risk_pcs or any(value is None or value & 1 for value in risk_pcs) \
                or len(set(risk_pcs)) != len(risk_pcs):
            return "invalid-risk-execution-pcs"
        try:
            trap_epc = _pc_value(_observer_field_value(record, "trap.epc"))
        except (TypeError, ValueError):
            trap_epc = None
        if trap_epc is None:
            return "expected-trap-epc-missing"
        if trap_epc not in risk_pcs:
            return "expected-trap-epc-not-risk-pc"
        return None

    if path_unavailable:
        if expected_identity.get("expected_trap") is True:
            if (
                _observation_value(record, "outcome") != "trap"
                and not guest_trap_observed(extra)
                or terminal_ebreak_observed(extra, risk_pcs=risk_pcs)
            ):
                return "expected-trap-risk-execution-unobserved"
            return expected_trap_pc_error()
        return None
    if marker.get("status") != "observed":
        return "risk-execution-not-observed"
    try:
        executed = tuple(_pc_value(value) for value in record.get("executed_pcs", ()))
        if record.get("backend") in {"rax-riscv64", "rvvm-riscv64"}:
            path_pcs = tuple(
                _pc_value(value)
                for value in path_identity.get("executed_pcs", ())
            ) if isinstance(path_identity, Mapping) else ()
            if (
                not isinstance(details, Mapping)
                or details.get("trace_available") is not True
                or path_status != "observed"
                or path_pcs != executed
            ):
                return "risk-trace-evidence-mismatch"
    except (TypeError, ValueError):
        return "invalid-risk-execution-pcs"
    if (
        not risk_pcs
        or any(value is None or value & 1 for value in risk_pcs)
        or len(set(risk_pcs)) != len(risk_pcs)
        or any(value not in executed for value in risk_pcs)
    ):
        return "risk-pc-not-present-in-executed-pcs"
    return expected_trap_pc_error()


def target_baseline_contract_valid(
    record: Mapping[str, object], expected_identity: Mapping[str, object] | None = None,
    *, expected_trap: bool = False,
) -> bool:
    if isinstance(expected_identity, Mapping) and expected_identity.get("route") == "single":
        if target_risk_execution_contract_error(record, expected_identity) is not None:
            return False
    backend = record.get("backend")
    if not is_execution_backend(backend):
        return False
    outcome, exit_code = record.get("outcome"), record.get("exit_code")
    if not (
        outcome in {"normal", "completed"} and type(exit_code) is int and exit_code == 0
        or outcome == "nonzero-exit" and type(exit_code) is int and exit_code != 0
        or outcome == "trap" and (exit_code is None or type(exit_code) is int)
    ):
        return False
    extra = record.get("extra_state")
    signal = record.get("signal")
    signal_observed = isinstance(signal, str) and bool(signal.strip())
    signal_observed = signal_observed or type(record.get("signal_code")) is int \
        and record["signal_code"] > 0
    if (expected_trap or outcome == "nonzero-exit") and outcome in {"trap", "nonzero-exit"} and not (
        signal_observed
        or guest_trap_observed(extra)
        or isinstance(extra, Mapping) and extra.get("target_stop_observed") is True
    ):
        return False
    if isinstance(extra, Mapping) and extra.get("host_abort") is not None and (
        extra.get("host_abort") != "unhandled-cpu-exception"
        or extra.get("target_host_abort_observed") is not True
    ):
        return False
    evidence = record.get("translation_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("backend") != backend:
        return False
    expected_path = _STRUCTURED_PATH_BY_BACKEND.get(canonical_execution_backend(backend))
    if expected_path is not None and evidence.get("expected_path") != expected_path:
        return False
    details = evidence.get("details")
    configuration = details.get("configuration_identity") if isinstance(details, Mapping) else None
    if not isinstance(details, Mapping) or not isinstance(configuration, Mapping):
        return False
    if isinstance(extra, Mapping) and extra.get("host_abort") is not None \
            and details.get("host_abort_path_observed") is not True:
        return False
    if (
        configuration.get("target_identity_status") != "verified"
        or not is_sha256_digest(configuration.get("target_binary_sha256"))
        or not is_sha256_digest(configuration.get("target_identity_digest"))
    ):
        return False
    target_identity = details.get("target_identity")
    if not isinstance(target_identity, Mapping):
        return False
    try:
        target_identity = TargetBinaryIdentity.from_dict(dict(target_identity))
    except (KeyError, TypeError, ValueError):
        return False
    if target_identity.binary_sha256 != configuration["target_binary_sha256"] \
            or target_identity.identity_digest != configuration["target_identity_digest"] \
            or not _same_execution_backend(target_identity.backend, backend):
        return False
    if isinstance(expected_identity, Mapping):
        if expected_identity.get("target") not in (None, "") \
                and not _same_execution_backend(backend, expected_identity["target"]):
            return False
        if expected_identity.get("target_binary_sha256") not in (None, "") \
                and configuration.get("target_binary_sha256") != expected_identity["target_binary_sha256"]:
            return False
        if expected_identity.get("target_identity_digest") not in (None, "") \
                and configuration.get("target_identity_digest") != expected_identity["target_identity_digest"]:
            return False
    try:
        pcs = tuple(_pc_value(value) for value in record.get("executed_pcs", ()))
    except (TypeError, ValueError):
        pcs = ()
    terminal_trap = guest_trap_observed(extra)
    trace_observed = bool(pcs)
    path_required = True
    if isinstance(expected_identity, Mapping) and "path_required" in expected_identity:
        path_required = expected_identity.get("path_required") is True
    if not terminal_trap and any(value is None for value in pcs):
        return False
    path = details.get("path_identity")
    if path_required and trace_observed:
        try:
            path_pcs = tuple(_pc_value(value) for value in path.get("executed_pcs", ()))
        except (AttributeError, TypeError, ValueError):
            return False
        digest = pc_path(pcs)[1]
        witness = path.get("executed_pc") if isinstance(path, Mapping) else None
        if (
            not isinstance(path, Mapping)
            or path.get("contract") != PATH_IDENTITY_WITNESS_CONTRACT
            or path.get("status") != "observed"
            or path_pcs != pcs
            or path.get("digest") != digest
            or not isinstance(witness, Mapping)
            or witness.get("observed") is not True
            or witness.get("count") != len(pcs)
            or witness.get("digest") != digest
        ):
            return False
    elif path_required and isinstance(path, Mapping) and path.get("status") not in {None, "unavailable", "gap"}:
        return False
    counts = tuple(evidence.get(name) for name in (
        "translated_count", "execution_count", "interpreter_count", "fallback_count",
    ))
    if isinstance(details, Mapping) and any(
        name in details and details.get(name) != evidence.get(name)
        for name in ("translated_count", "execution_count", "interpreter_count", "fallback_count")
    ):
        return False
    return (
        all(type(value) is int and value >= 0 for value in counts)
        and (not trace_observed or counts[1] == len(pcs))
        and (evidence.get("tested_pc_executed") is not True or counts[1] > 0)
        and (evidence.get("tested_pc_translated") is not True or counts[0] > 0)
    )


def _target_execution_contract_reason(
    record: Mapping[str, object],
    expected_identity: Mapping[str, object] | None = None,
) -> str:
    if _observation_value(record, "outcome") == "timeout":
        return "target-timeout"
    evidence = _observation_value(record, "translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    configuration = details.get("configuration_identity") if isinstance(details, Mapping) else None
    if (
        isinstance(expected_identity, Mapping)
        and expected_identity.get("route") == "single"
        and target_risk_execution_contract_error(record, expected_identity) is not None
    ):
        return "rvgen-risk-execution-missing"
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("target_identity_status") != "verified"
    ):
        return "target-identity-unverified"
    return "invalid"


def _target_observation_summary(comparison: Mapping[str, object]) -> list[dict[str, object]]:
    """保留 Target gap 的原因摘要，不把完整 trace 塞进 campaign ledger。"""
    observations = comparison.get("observations")
    if not isinstance(observations, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for side in ("original", "variant"):
        values = observations.get(side)
        if not isinstance(values, (list, tuple)):
            continue
        for index, item in enumerate(values):
            def field(name: str, default: object = None) -> object:
                value = _observation_value(item, name)
                if value is None and isinstance(item, Mapping):
                    nested = item.get("observation")
                    value = _observation_value(nested, name, default)
                return default if value is None else value

            extra = field("extra_state")
            evidence = field("translation_evidence")
            details = evidence.get("details") if isinstance(evidence, Mapping) else None
            executed_pcs = field("executed_pcs")
            row: dict[str, object] = {
                "side": side,
                "index": index,
                "backend": field("backend"),
                "outcome": field("outcome"),
                "exit_code": field("exit_code"),
                "instruction_count": field("instruction_count"),
                "executed_pc_count": (
                    len(executed_pcs) if isinstance(executed_pcs, (list, tuple)) else None
                ),
                "contract_error": field("contract_error"),
            }
            for name in (
                "reason", "termination", "trace_reason",
                "trace_limit_exceeded", "stdout_truncated",
            ):
                value = field(name)
                if value is not None:
                    row[name] = value
            if isinstance(extra, Mapping):
                for name in (
                    "target_timeout_origin", "campaign_wall_clock_exhausted",
                    "target_host_abort_observed",
                ):
                    if name in extra:
                        row[name] = extra[name]
            if isinstance(details, Mapping):
                for name in (
                    "trace_returncode", "trace_failed", "granularity",
                    "trace_checkpoint_error",
                ):
                    if name in details:
                        row[name] = details[name]
            rows.append(row)
    return rows


def run_emi_pair(
    original: Program,
    variant: ProgramVariant,
    builder: BuilderCallback,
    reference_runner: Callable[[BuiltProgram], object],
    target_runner: Callable[[BuiltProgram], object] | None,
    *,
    work_dir: str | Path,
    expected_parent_sha256: str | None = None,
    reference_observations: object | None = None,
    expected_target_identity: Mapping[str, object] | None = None,
    target_original_observations: object | None = None,
    bare_builder: BuilderCallback | None = None,
    artifact_cache: dict[tuple[object, ...], BuiltProgram] | None = None,
    artifact_overrides: Mapping[str, BuiltProgram] | None = None,
    campaign_deadline_monotonic: float | None = None,
) -> dict[str, object]:
    """执行可选 Target，并用 reference 做差分比较。"""
    observer_fields = (
        expected_target_identity.get(
            "comparison_fields", expected_target_identity.get("observer_fields")
        )
        if isinstance(expected_target_identity, Mapping) else None
    )
    if expected_parent_sha256 is not None and not is_sha256_digest(expected_parent_sha256):
        return _pair_result("reference-rejected", _reference_gap({
            "expected_parent_sha256": "invalid",
        }))
    expected_parent = program_sha256(original) if expected_parent_sha256 is None else expected_parent_sha256
    original_params = _program_params(original)
    variant_params = dict(variant.run_params)
    compared_original_params, compared_variant_params = original_params, variant_params
    if original_params.get("route") == "single":
        compared_original_params = _stable_input_identity(original_params)
        compared_variant_params = _stable_input_identity(variant_params)
    if variant.parent_sha256 != expected_parent or compared_variant_params != compared_original_params:
        differences = {}
        if variant.parent_sha256 != expected_parent:
            differences["parent_sha256"] = (expected_parent, variant.parent_sha256)
        if compared_variant_params != compared_original_params:
            differences["run_params"] = (original_params, variant_params)
        return _pair_result("reference-rejected", _reference_gap(differences))
    root = Path(work_dir)

    def deadline_reached() -> bool:
        return (
            target_runner is not None
            and campaign_deadline_monotonic is not None
            and time.monotonic() >= campaign_deadline_monotonic
        )

    def deadline_result(stage: str, artifacts: Mapping[str, object] | None = None):
        return _pair_result(
            "target-gap",
            _reference_gap({"target_deadline": stage}),
            {
                "status": "target-gap",
                "failure_class": "campaign-wall-clock-exhausted",
                "deadline_censored": True,
            },
            dict(artifacts or {}),
            failure_class="campaign-wall-clock-exhausted",
        )

    def build_cached(
        program: Program, *, work_dir: Path, source_dir: Path | None = None,
    ) -> BuiltProgram:
        key = _build_cache_key(program, builder, bare_builder) \
            if artifact_cache is not None else None
        cached = artifact_cache.get(key) if key is not None else None
        if cached is not None:
            return cached
        artifact = build_program(
            program, builder, work_dir=work_dir, source_dir=source_dir,
            bare_builder=bare_builder,
        )
        if key is not None:
            artifact_cache[key] = artifact
        return artifact

    if artifact_overrides is not None:
        supplied_original = artifact_overrides.get("original")
        supplied_variant = artifact_overrides.get("variant")
        if not isinstance(supplied_original, BuiltProgram) or not isinstance(
            supplied_variant, BuiltProgram
        ):
            return _pair_result(
                "transport-gap",
                _reference_gap({"artifact_overrides": "original-and-variant-required"}),
            )
        original_artifact = replace(supplied_original, parent_sha256=None)
        variant_artifact = supplied_variant
    else:
        if deadline_reached():
            return deadline_result("before-build")
        try:
            original_artifact = replace(
                build_cached(original, work_dir=root / "original"),
                parent_sha256=None,
            )
            if deadline_reached():
                return deadline_result(
                    "after-original-build",
                    {"reference_original": _artifact_record(original_artifact)},
                )
            source_dir = getattr(original, "program", None)
            source_dir = Path(source_dir).parent if source_dir is not None else None
            variant_artifact = build_cached(
                variant, work_dir=root / "variant", source_dir=source_dir,
            )
            if deadline_reached():
                return deadline_result(
                    "after-variant-build",
                    {
                        "reference_original": _artifact_record(original_artifact),
                        "reference_variant": _artifact_record(variant_artifact),
                    },
                )
        except Exception as error:
            return _pair_result("transport-gap", _reference_gap({"build_error": str(error)}))
    artifacts = {
        "reference_original": _artifact_record(original_artifact),
        "reference_variant": _artifact_record(variant_artifact),
    }
    artifact_pairs = (("original", original_artifact), ("variant", variant_artifact))
    def artifact_case(artifact: BuiltProgram) -> Mapping[str, object] | None:
        params = artifact.run_params
        case = params.get("single_case")
        return case if params.get("route") == "single" and isinstance(case, Mapping) else None

    def artifact_expected_identity(artifact: BuiltProgram) -> dict[str, object]:
        identity = dict(expected_target_identity or {})
        case = artifact_case(artifact)
        if case is not None:
            dataflow = case.get("dataflow_meta")
            identity["route"] = "single"
            identity["expected_trap"] = expected_trap_from_dataflow(
                dict(dataflow) if isinstance(dataflow, Mapping) else {},
            )
            raw_mask = case.get("compare_mask")
            if isinstance(raw_mask, Mapping):
                fields = _compare_mask_fields(CompareMask.from_dict(dict(raw_mask)))
                identity["comparison_fields"] = list(fields)
                identity["observer_fields"] = list(fields)
        return identity

    def artifact_comparison_fields(
        artifact: BuiltProgram,
        observations: Sequence[object] = (),
        *,
        reference_side: bool = False,
    ):
        case = artifact_case(artifact)
        raw_mask = case.get("compare_mask") if case is not None else None
        fields = (
            _compare_mask_fields(CompareMask.from_dict(dict(raw_mask)))
            if isinstance(raw_mask, Mapping) else observer_fields
        )
        if (
            not reference_side
            or not observations
            or artifact_expected_identity(artifact).get("expected_trap") is not True
            or fields is None
        ):
            return fields
        for item in observations:
            extra = _observation_value(item, "extra_state", {})
            declared = extra.get("observer_fields") if isinstance(extra, Mapping) else None
            executed_pcs = _observation_value(item, "executed_pcs", ())
            fault_pc = _observation_value(item, "fault_pc")
            if not (
                _observation_value(item, "backend") == "native-rv64"
                and _observation_value(item, "outcome") == "trap"
                and _observation_value(item, "signal") == "SIGILL"
                and type(fault_pc) is int
                and fault_pc in executed_pcs
                and isinstance(extra, Mapping)
                and extra.get("target_stop_observed") is True
                and extra.get("target_stop_observer") == "native-ptrace-terminal-signal"
                and isinstance(declared, (list, tuple))
                and {"outcome", "signal", "fault_pc"}.issubset(declared)
            ):
                return fields
        # Native ptrace supplies the terminal signal and PC, but no mcause,
        # mepc, or mtval. Compare the fields it actually observes.
        unavailable_fields = {
            "extra.trap.cause", "extra.trap.epc", "extra.trap.tval",
        }
        return tuple(field for field in fields if field not in unavailable_fields)

    def pair_comparison_fields(
        left: BuiltProgram,
        right: BuiltProgram,
        rows: tuple[Sequence[object], Sequence[object]],
        *,
        reference_side: bool = False,
    ):
        left_fields = artifact_comparison_fields(
            left, rows[0], reference_side=reference_side,
        )
        right_fields = artifact_comparison_fields(
            right, rows[1], reference_side=reference_side,
        )
        return (
            (observer_fields, observer_fields)
            if left_fields is None or right_fields is None
            else (left_fields, right_fields)
        )

    reference_identity_gap = None
    if reference_observations is not None:
        reference_rows = _observations(reference_observations)
        for item in reference_rows:
            backend = _observation_value(item, "backend")
            if backend == "sail-riscv":
                continue
            expected_sha256 = _artifact_sha256_for_backend(original_artifact, backend)
            recorded = _observation_value(item, "binary_sha256")
            guest = _observation_value(item, "guest_elf_sha256")
            if any(value is not None and value != expected_sha256
                   for value in (recorded, guest)):
                reference_identity_gap = _reference_gap({
                        "baseline_observation_binary_sha256": recorded,
                        "baseline_observation_guest_elf_sha256": guest,
                        "actual_executable_sha256": expected_sha256,
                    })
                break

    def compare_pair(
        runner, left, right, cached=None,
        *, stop_on_unsupported: bool = False, reference_side: bool = False,
    ):
        left_expected_trap = artifact_expected_identity(left).get("expected_trap") is True
        right_expected_trap = artifact_expected_identity(right).get("expected_trap") is True
        left_rows = (
            _observations(cached) if cached is not None else _observations(runner(left))
        )
        if stop_on_unsupported:
            reason = next(
                (
                value for item in left_rows
                if (value := _target_unsupported_reason(
                    item, expected_trap=left_expected_trap,
                )) is not None
                ),
                None,
            )
            if reason is not None:
                return {
                    "status": "case-skipped",
                    "comparison_mode": "target-only",
                    "differences": {"target": reason},
                    "observations": {
                        "original": [_observation_record(item) for item in left_rows],
                        "variant": [],
                    },
                }
        right_rows = _observations(runner(right))
        if stop_on_unsupported:
            reason = next(
                (
                value for item in right_rows
                if (value := _target_unsupported_reason(
                    item, expected_trap=right_expected_trap,
                )) is not None
                ),
                None,
            )
            if reason is not None:
                return {
                    "status": "case-skipped",
                    "comparison_mode": "target-only",
                    "differences": {"target": reason},
                    "observations": {
                        "original": [_observation_record(item) for item in left_rows],
                        "variant": [_observation_record(item) for item in right_rows],
                    },
                }
        rows = (left_rows, right_rows)
        left_fields, right_fields = pair_comparison_fields(
            left, right, rows, reference_side=reference_side,
        )
        comparison = (
            compare_observations(*rows, left_fields)
            if left_fields == right_fields else
            compare_observations(
                *rows,
                observer_fields_by_side=(left_fields, right_fields),
            )
        )
        comparison["observations"] = {
            name: [_observation_record(item) for item in row]
            for name, row in zip(("original", "variant"), rows)
        }
        return comparison

    def artifact_identity_gap(comparison, pairs):
        for side, artifact in pairs:
            for item in comparison.get("observations", {}).get(side, ()):
                backend = _observation_value(item, "backend")
                recorded = _observation_value(item, "binary_sha256")
                guest = _observation_value(item, "guest_elf_sha256")
                expected_sha256 = _artifact_sha256_for_backend(artifact, backend)
                if is_execution_backend(backend) and expected_sha256 is None:
                    return _reference_gap({
                        f"{side}_observation_artifact_identity": "selected-artifact-missing",
                        f"{side}_backend": backend,
                    })
                if is_execution_backend(backend) and recorded is None and guest is None:
                    return _reference_gap({
                        f"{side}_observation_artifact_identity": "missing",
                        f"{side}_executable_sha256": expected_sha256,
                    })
                if backend != "sail-riscv" and any(
                    value is not None and value != expected_sha256
                    for value in (recorded, guest)
                ):
                    evidence = _observation_value(item, "translation_evidence")
                    details = evidence.get("details") if isinstance(evidence, Mapping) else None
                    capsule_sha = details.get("capsule_sha256") if isinstance(details, Mapping) else None
                    if capsule_sha is None and isinstance(details, Mapping):
                        configuration = details.get("configuration_identity")
                        capsule_sha = (
                            configuration.get("capsule_sha256")
                            if isinstance(configuration, Mapping) else None
                        )
                    if (
                        backend == "rvvm-riscv64"
                        and isinstance(details, Mapping)
                        and details.get("capsule_mode") == "bare-metal-no-ecall"
                        and capsule_sha in (recorded, guest)
                    ):
                        continue
                    return _reference_gap({
                        f"{side}_observation_binary_sha256": recorded,
                        f"{side}_observation_guest_elf_sha256": guest,
                        f"{side}_executable_sha256": expected_sha256,
                    })
        return None

    def target_reference_identity_gap(reference, target):
        reference_observations = reference.get("observations", {})
        target_observations = target.get("observations", {})
        for side in ("original", "variant"):
            reference_rows = reference_observations.get(side, ()) \
                if isinstance(reference_observations, Mapping) else ()
            target_rows = target_observations.get(side, ()) \
                if isinstance(target_observations, Mapping) else ()
            if not isinstance(reference_rows, list) or not isinstance(target_rows, list):
                continue
            for index, (reference_item, target_item) in enumerate(zip(reference_rows, target_rows)):
                reference_backend = _observation_value(reference_item, "backend")
                target_backend = _observation_value(target_item, "backend")
                same_backend = (
                    isinstance(reference_backend, str) and isinstance(target_backend, str)
                    and _same_execution_backend(reference_backend, target_backend)
                )
                shared_qemu_fallback = (
                    same_backend and _is_declared_qemu_reference_fallback(reference_item)
                )
                if same_backend and not shared_qemu_fallback:
                    reference_identity = _execution_identity(reference_item)
                    target_identity = _execution_identity(target_item)
                    if not any(
                        reference_identity.get(name) not in (None, "")
                        and target_identity.get(name) not in (None, "")
                        and reference_identity[name] != target_identity[name]
                        for name in ("target_binary_sha256", "target_identity_digest")
                    ):
                        return _reference_gap({
                            f"target_identity:{side}:{index}": "same-backend-as-reference",
                        })
                identities = []
                for item, label in ((reference_item, "reference"), (target_item, "target")):
                    payloads = _target_identity_parts(item)[1]
                    if payloads:
                        try:
                            identities.append(TargetBinaryIdentity.from_dict(payloads[0]))
                        except (KeyError, TypeError, ValueError):
                            return _reference_gap({
                                f"{label}_target_identity:{side}:{index}": "invalid",
                            })
                if len(identities) == 2 and (
                    identities[0].identity_digest == identities[1].identity_digest
                    or identities[0].binary_sha256 == identities[1].binary_sha256
                ) and not shared_qemu_fallback:
                    return _reference_gap({
                        f"target_identity:{side}:{index}": "same-binary-as-reference",
                    })
        return None

    def target_execution_contract_gap(comparison):
        identities = []
        artifacts_by_side = dict(artifact_pairs)
        for side, items in comparison.get("observations", {}).items():
            expected_identity = artifact_expected_identity(artifacts_by_side[side])
            for item in items:
                raw_backend = _observation_value(item, "backend")
                if not isinstance(raw_backend, str) or not is_execution_backend(raw_backend):
                    return _reference_gap({f"{side}_backend": "unknown-or-invalid"})
                expected_backend = (
                    expected_identity.get("target")
                    if isinstance(expected_identity, Mapping) else None
                )
                if isinstance(expected_backend, str) and not _same_execution_backend(
                    raw_backend, expected_backend
                ):
                    return _reference_gap({
                        f"{side}_backend": "root-binding-mismatch",
                        "expected_target": expected_backend,
                        "actual_target": raw_backend,
                    })
                if not target_baseline_contract_valid(
                    item, expected_identity,
                    expected_trap=isinstance(expected_identity, Mapping)
                    and expected_identity.get("expected_trap") is True,
                ):
                    reason = _target_execution_contract_reason(item, expected_identity)
                    return _reference_gap({f"{side}_target_execution_contract": reason})
                evidence = _observation_value(item, "translation_evidence")
                details = evidence.get("details") if isinstance(evidence, Mapping) else None
                configuration = details.get("configuration_identity") if isinstance(details, Mapping) else None
                if not isinstance(configuration, Mapping):
                    return _reference_gap({f"{side}_target_execution_contract": "target-identity-unverified"})
                identities.append((
                    configuration["target_binary_sha256"],
                    configuration["target_identity_digest"],
                ))
        if identities and any(item != identities[0] for item in identities[1:]):
            return _reference_gap({"target_identity": "pair-drift"})
        return None

    def target_input_identity_gap(comparison, pairs, reference=None):
        def input_id(item):
            value = _observation_value(item, "input_id")
            nested = _observation_value(item, "reference_identity")
            return value if value not in (None, "") else (
                nested.get("input_id") if isinstance(nested, Mapping) else None
            )

        for side, artifact in pairs:
            expected = artifact.run_params.get("input_id")
            if expected in (None, "") and isinstance(reference, Mapping):
                reference_values = [
                    input_id(item)
                    for item in reference.get("observations", {}).get(side, ())
                ]
                reference_values = [value for value in reference_values if value not in (None, "")]
                if reference_values:
                    expected = reference_values[0]
            if expected in (None, ""):
                continue
            for item in comparison.get("observations", {}).get(side, ()):
                actual = input_id(item)
                if actual != expected:
                    return _reference_gap({
                        f"{side}_observation_input_id": actual,
                        f"{side}_expected_input_id": expected,
                    })
        return None

    target = None
    target_failure_class = None

    def target_unsupported_reason(comparison: object) -> str | None:
        if isinstance(comparison, Mapping):
            observations = comparison.get("observations")
            if isinstance(observations, Mapping):
                for side in ("original", "variant"):
                    side_rows = observations.get(side, ())
                    if not isinstance(side_rows, (list, tuple)):
                        continue
                    expected_trap = artifact_expected_identity(
                        dict(artifact_pairs)[side],
                    ).get("expected_trap") is True
                    for item in side_rows:
                        reason = _target_unsupported_reason(
                            item, expected_trap=expected_trap,
                        )
                        if reason is not None:
                            return reason
            differences = comparison.get("differences")
            reason = _target_unsupported_reason(
                differences,
                expected_trap=all(
                    artifact_expected_identity(artifact).get("expected_trap") is True
                    for _side, artifact in artifact_pairs
                ),
            )
            if reason is not None:
                return reason
        return _target_unsupported_reason(comparison)

    if target_runner is not None:
        if deadline_reached():
            return deadline_result(
                "before-target",
                {**artifacts, "target_variant": _artifact_record(variant_artifact)},
            )
        try:
            # Target execution is independent evidence.  It runs before the
            # reference verdict so rejected/unverified references still leave
            # simulator coverage and raw observations behind.
            target = compare_pair(
                target_runner, original_artifact, variant_artifact,
                cached=target_original_observations,
                stop_on_unsupported=True,
            )
        except (OSError, subprocess.SubprocessError, ExecutionEnvironmentError) as error:
            target = _reference_gap({"target_error": str(error)})
            target_failure_class = "transport-gap"
        except (TypeError, ValueError) as error:
            target = _reference_gap({"target_error": str(error)})
            target_failure_class = "target-contract-gap"
        except Exception as error:
            target = _reference_gap({"target_error": str(error)})

        if deadline_reached():
            return deadline_result(
                "after-target",
                {**artifacts, "target_variant": _artifact_record(variant_artifact)},
            )

        unsupported_reason = target_unsupported_reason(target)
        if unsupported_reason is not None:
            target = {
                **(dict(target) if isinstance(target, Mapping) else {}),
                "status": "case-skipped",
                "skip_reason": unsupported_reason,
                "failure_class": "unsupported-isa",
            }
            return _pair_result(
                "case-skipped",
                _reference_gap({"target": unsupported_reason}),
                target,
                {
                    **artifacts,
                    "target_variant": _artifact_record(variant_artifact),
                },
                failure_class="unsupported-isa",
            )

    try:
        reference_observations_for_pair = reference_observations
        reference = compare_pair(
            reference_runner, original_artifact, variant_artifact,
            reference_observations_for_pair,
            reference_side=True,
        )
    except (OSError, subprocess.SubprocessError, ExecutionEnvironmentError) as error:
        reference = _reference_gap({"reference_error": str(error)})
        reference_failure_class = "transport-gap"
    except Exception as error:
        reference = _reference_gap({"reference_error": str(error)})
        reference_failure_class = _reference_exception_status(error)
    else:
        reference_failure_class = None
    if reference_identity_gap is not None:
        reference = reference_identity_gap
    reference_qualification_gap = _record_has_qualification_gap(reference)
    if reference["status"] != "equivalent" or reference_qualification_gap:
        differences = reference.get("differences")
        contract_error = (
            differences.get("contract_error")
            if isinstance(differences, Mapping) else None
        )
        status = (
            "reference-rejected" if reference_identity_gap is not None else
            reference_failure_class
            if reference_failure_class in {
                "profile-gap", "reference-rejected", "transport-gap",
            } else "reference-gap" if reference_qualification_gap else
            _reference_failure_status(contract_error)
            if reference["status"] == "reference-gap" else "reference-rejected"
        )
        return _pair_result(
            status, reference, target,
            {**artifacts, **({"target_variant": _artifact_record(variant_artifact)}
                             if target_runner is not None else {})},
            failure_class=reference_failure_class or target_failure_class,
        )
    # Compatibility calls may capture Target evidence before the reference
    # verdict; each input contributes its single observed execution.
    if identity_gap := artifact_identity_gap(
        reference, artifact_pairs
    ):
        return _pair_result(
            "reference-rejected", identity_gap, target,
            {**artifacts, "target_variant": _artifact_record(variant_artifact)},
        )
    if identity_gap := target_input_identity_gap(
        reference, artifact_pairs
    ):
        return _pair_result(
            "profile-gap", identity_gap, target,
            {**artifacts, "target_variant": _artifact_record(variant_artifact)},
        )
    if target_runner is None:
        return _pair_result(
            "target-unavailable", reference,
            _reference_gap({"target_runner": "unavailable"}), artifacts,
        )
    target_timeout = any(
        _observation_value(item, "outcome") == "timeout"
        for items in target.get("observations", {}).values()
        if isinstance(items, (list, tuple))
        for item in items
    ) if isinstance(target.get("observations"), Mapping) else False
    target_observation_summary = _target_observation_summary(target)
    target_comparison_qualification_gap = _reference_gap({
        "target_comparison_qualification": {
            name: target[name]
            for name in (
                "comparison_qualification_gaps", "qualification_gaps",
                "missing_fields", "validation_gap", "contract_error",
                "target_execution_contract_error",
            )
            if target.get(name)
        },
    }) if _record_has_qualification_gap(target) else None
    validation_gap = (
        target_comparison_qualification_gap
        or target_reference_identity_gap(reference, target)
        or target_execution_contract_gap(target)
        or artifact_identity_gap(target, artifact_pairs)
        or target_input_identity_gap(target, artifact_pairs, reference)
    )
    if validation_gap:
        if target_observation_summary:
            validation_gap["target_observation_summary"] = target_observation_summary
        # Keep the target observations and simulator trace available to the
        # coverage side channel.  The validation result controls differential
        # eligibility only; it must not erase execution evidence.
        target = {**target, "validation_gap": validation_gap}
    raw_target_status = target.get("status")
    pair_status = (
        "target-mismatch-candidate"
        if raw_target_status == "non-equivalent" and not validation_gap else
        "clean" if raw_target_status == "equivalent" and not validation_gap else
        "target-gap"
    )
    return _pair_result(
        pair_status,
        reference,
        target,
        {**artifacts, "target_variant": _artifact_record(variant_artifact)},
        failure_class=(
            "target-timeout" if target_timeout else
            "target-contract-gap" if validation_gap else
            "target-runner-gap"
            if raw_target_status == "reference-gap"
            and "target_error" in target.get("differences", {}) else None
        ),
    )


__all__ = [
    "BuiltProgram", "BuilderCallback", "Program",
    "build_program", "enumerate_case_rewrites", "prioritize_case_rewrites_for_model",
    "rewrite_program",
    "compare_observations", "run_emi_pair",
    "target_baseline_contract_valid", "target_risk_execution_contract_error",
]

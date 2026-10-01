"""以完整程序为状态的最小 MH 入口。"""

import math
import posixpath
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

from framework._util import (
    canonical_digest, is_sha256_digest, strip_c_comments as _strip_c_comments,
)
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from framework.mcmc.sampler import MHProposal, MHRun, REJECTED, mh_chain
from framework.rvemi.program import (
    ProgramVariant, _data_directive_size, _float_literal_valid, _insn_size, _instruction_size, _int_value,
    _register_index,
    _split_operands, _split_source_statements, _string_size, _strip_comments,
    _source as _text, _pseudo_enabled, _leb128_size,
    _fp_pseudo_enabled,
    _relocation_neutral, _relocation_valid, _resolve_memory_alias, _resolve_size_operand, _layout_int,
    _collect_equ_aliases, _normalize_target_expression,
    _source_include_files, _SOURCE_INCBIN_RE,
    _BASE_EXPANDING_PSEUDOS, _DATA_DIRECTIVE_WIDTHS, _FP_PSEUDOS,
    _FLOAT_DIRECTIVE_WIDTHS,
    _PROFILE_CONTROL_ARITIES, _PROFILE_PSEUDO_ARITIES,
    _text_encodable, _insn_operand_syntax_valid,
    _stable_input_identity, apply_profile, profile_actions, program_sha256,
)


from framework.rvemi.program import (
    _profile_operands_valid, _source_mnemonic, _zcmp_stack_operands_valid,
)
from framework.rvemi.runtime_spec import runtime_instruction_spec
from framework.spec_definedness import (
    enabled_extensions, instruction_spec_for_generation, isa_profile_for_extensions,
    is_canonical_isa_profile, profile_enables_form, profile_uses_compressed_encoding,
)

_ACTION_POOL_AUDIT_LIMIT = 256

_RNG_SCHEME = "python-random-mt19937-seedx100-v1"
_LABEL_RE = re.compile(r"^\s*([A-Za-z_.$][\w.$]*|\d+):(?:\s*(.*))?$")
_REGISTER_RE = re.compile(
    r"(?<![A-Za-z0-9_.$])(?:(?:ft|fs|fa|x|f|v|a|t|s)\d+|"
    r"(?:zero|ra|sp|gp|tp|fp))(?![A-Za-z0-9_.$])"
)
_HEX_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?0[xX][0-9A-Fa-f]+")
_REGISTER_ALIASES = {
    "zero": "x0", "ra": "x1", "sp": "x2", "gp": "x3", "tp": "x4", "fp": "x8",
    **dict(zip((f"t{i}" for i in range(3)), (f"x{i + 5}" for i in range(3)))),
    **dict(zip((f"t{i}" for i in range(3, 7)), (f"x{i + 25}" for i in range(3, 7)))),
    **dict(zip((f"s{i}" for i in range(2)), (f"x{i + 8}" for i in range(2)))),
    **dict(zip((f"s{i}" for i in range(2, 12)), (f"x{i + 16}" for i in range(2, 12)))),
    **dict(zip((f"a{i}" for i in range(8)), (f"x{i + 10}" for i in range(8)))),
    **dict(zip((f"ft{i}" for i in range(8)), (f"f{i}" for i in range(8)))),
    **dict(zip((f"ft{i}" for i in range(8, 12)), (f"f{i + 20}" for i in range(8, 12)))),
    "fs0": "f8", "fs1": "f9",
    **dict(zip((f"fs{i}" for i in range(2, 12)), (f"f{i + 16}" for i in range(2, 12)))),
    **dict(zip((f"fa{i}" for i in range(8)), (f"f{i + 10}" for i in range(8)))),
}
_CONTROL_OPS = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez",
    "bltzal", "bgezal", "bgt", "ble", "bgtz", "blez", "bgtu", "bleu", "beqi", "bnei", "j", "jal",
    "jalr", "jr", "ret", "call", "tail", "c.beqz", "c.bnez", "c.j", "c.jal",
    "c.jr", "c.jalr", "c.ebreak", "cm.jt", "cm.jalt", "cm.popret", "cm.popretz",
    "ecall", "ebreak", "scall", "sbreak", "mret", "sret", "wfi", "unimp", "c.unimp",
    ".insn.b", ".insn.j", ".insn.i",
}
_CODE_DIRECTIVES = frozenset({
    ".byte", ".2byte", ".4byte", ".8byte", ".half", ".short", ".word",
    ".long", ".dword", ".quad", ".insn",
})
_UNSUPPORTED_DIRECTIVES = {".rept", ".irp", ".irpc", ".endr", ".macro", ".endm", ".if", ".ifdef", ".ifndef", ".else", ".elseif", ".endif"}
_DATA_SECTIONS = frozenset({".data", ".bss", ".rodata", ".sdata", ".sbss", ".tdata", ".tbss"})
_REGISTER_JUMP_OPS = {
    "jr", "jalr", "c.jr", "c.jalr", "c.ebreak", "cm.jt", "cm.jalt", "cm.popret", "cm.popretz",
    "ret", "ecall", "ebreak", "scall", "sbreak", "mret", "sret", "wfi",
}
_LAYOUT_DIRECTIVES = frozenset({
    ".ascii", ".asciz", ".string", ".balign", ".balignw", ".balignl",
    ".align", ".p2align", ".p2alignw", ".p2alignl", ".space", ".zero", ".skip", ".fill", ".org", ".incbin",
    ".float", ".single", ".double", ".uleb128", ".sleb128",
})
_KNOWN_DIRECTIVES = frozenset({
    ".text", ".section", ".pushsection", ".popsection", ".previous",
    ".subsection", ".data", ".bss", ".rodata", ".sdata", ".sbss", ".tdata",
    ".tbss", ".include", ".option", ".equ", ".set", ".globl", ".global",
    ".type", ".size", ".attribute", ".file", ".loc", ".ident", ".hidden",
    ".weak", ".local", ".comm", ".common", ".lcomm", ".extern", ".end",
    ".symver", ".uleb128", ".sleb128", ".float", ".single", ".double",
    ".cfi_startproc", ".cfi_endproc", ".cfi_def_cfa", ".cfi_offset",
    ".cfi_restore", ".cfi_remember_state", ".cfi_restore_state", ".cfi_sections",
    ".cfi_escape", ".cfi_signal_frame", ".riscv.attributes",
}) | _CODE_DIRECTIVES | _LAYOUT_DIRECTIVES | _DATA_SECTIONS | _UNSUPPORTED_DIRECTIVES
_PSEUDO_OPS = frozenset({
    "beqz", "bnez", "bltz", "bgez", "bltzal", "bgezal", "bgt", "ble", "bgtz",
    "blez", "bgtu", "bleu", "j", "jr", "ret", "call", "tail", "la", "li", "mv",
    "nop", "not", "neg", "negw", "sext.b", "sext.h", "sext.w", "zext.b", "zext.h", "zext.w",
    "fmv.s", "fabs.s", "fneg.s", "fmv.d", "fabs.d", "fneg.d", "fmv.h", "fabs.h", "fneg.h",
    "seqz", "snez", "sltz", "sgtz",
    "csrr", "csrw", "csrs", "csrc", "csrwi", "csrsi", "csrci", "rdcycle", "rdtime",
    "rdinstret", "rdcycleh", "rdtimeh", "rdinstreth", "fmv.s.x", "fmv.x.s", "frcsr", "fscsr", "frflags", "frrm", "fsrm", "fsrmi", "fence.tso", "scall", "sbreak", "unimp", "c.unimp", "pause", "lla", "lga",
})
_COMPRESSED_REL_OPS = frozenset({"j", "jal", "beqz", "bnez", "beq", "bne", "c.j", "c.jal", "c.beqz", "c.bnez"})
_BRANCH_IMMEDIATE_OPS = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqi", "bnei",
    "beqz", "bnez", "bltz", "bgez", "bltzal", "bgezal", "bgt", "ble", "bgtz",
    "blez", "bgtu", "bleu", "c.beqz", "c.bnez",
})
_JUMP_IMMEDIATE_OPS = frozenset({"j", "jal", "c.j", "c.jal"})
_DIRECT_TARGET_OPS = frozenset({"call", "tail"})
_CONDITIONAL_DIRECTIVES = frozenset({".if", ".ifdef", ".ifndef", ".else", ".elseif", ".endif"})
_DIRECTIVES_WITH_PREFIXES = (".cfi_", ".debug_")


def _registers(text: str) -> tuple[str, ...]:
    text = re.sub(
        r"\bs(\d+)\s*-\s*s(\d+)\b",
        lambda match: " ".join(f"s{i}" for i in range(int(match[1]), int(match[2]) + 1)),
        text.lower(),
    )
    text = re.sub(r"(?<![A-Za-z0-9_.$])v(\d+)\.t(?![A-Za-z0-9_.$])", r"v\1", text)
    return tuple(
        _REGISTER_ALIASES.get(item, item)
        for item in _REGISTER_RE.findall(_HEX_RE.sub(" ", text))
    )


def _branch_state(op: str, operands: Sequence[str]) -> bool | None:
    if op in {"beqi", "bnei"} and len(operands) >= 3:
        register = _registers(operands[0])
        immediate = _int_value(operands[1])
        if register[:1] == ("x0",) and immediate is not None:
            return (immediate == 0) if op == "beqi" else (immediate != 0)
    values = operands if op in _REGISTER_JUMP_OPS else operands[:-1]
    registers = _registers(" ".join(values))
    if op in {"beq", "bge", "bgeu", "ble", "bleu"}:
        return True if len(registers) >= 2 and registers[0] == registers[1] else None
    if op in {"bne", "blt", "bltu", "bgt", "bgtu"}:
        return False if len(registers) >= 2 and registers[0] == registers[1] else None
    if op in {"beqz", "bgez", "blez", "bgezal", "c.beqz"}:
        return True if registers[:1] == ("x0",) else None
    if op in {"bnez", "bltz", "bgtz", "bltzal", "c.bnez"}:
        return False if registers[:1] == ("x0",) else None
    return None


def _dataflow(op: str, operands: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    used = _registers(
        " ".join(operands[:-1] if op in _CONTROL_OPS - _REGISTER_JUMP_OPS else operands)
    )
    if op in {"call", "c.jal", "c.jalr", "cm.jalt", "cm.popret", "cm.popretz", "ret"} or op == "jal" and len(operands) == 1:
        used += ("x1",)
    elif op == "tail":
        used += ("x6",)
    if op in {"cm.push", "cm.pop", "cm.popret", "cm.popretz"}:
        used += ("x2",)
    if op in {"cm.mva01s", "cm.mvsa01"}:
        used += ("x10", "x11")
    return op, tuple(dict.fromkeys(used))


def _split_line(raw: str) -> tuple[tuple[str, ...], str]:
    value = raw.strip()
    if not re.match(
        r"^(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*#\s*include\b",
        value, re.IGNORECASE,
    ):
        value = _strip_comments(raw).strip()
    labels = []
    while match := _LABEL_RE.match(value):
        labels.append(match[1])
        value = (match[2] or "").strip()
    return tuple(labels), value


def _logical_source_lines(source: str) -> tuple[str, ...]:
    lines, pending = [], ""
    for raw in source.splitlines():
        value = raw.rstrip()
        if value.endswith("\\"):
            pending += value[:-1]
            continue
        lines.append(pending + raw)
        pending = ""
    return (*lines, pending) if pending else tuple(lines)


def _validate_beta(beta: object) -> None:
    try:
        valid = not isinstance(beta, bool) and isinstance(beta, (int, float)) and math.isfinite(beta)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("beta must be finite")


def _layout_size(
    op: str, operands: Sequence[str], pc: int,
    aliases: Mapping[str, object] | None = None,
) -> int | None:
    if op in {".ascii", ".asciz", ".string"}:
        if not operands:
            return None
        return _string_size(f"{op} {','.join(operands)}") if operands else None
    if op in _FLOAT_DIRECTIVE_WIDTHS:
        if not operands:
            return None
        return (
            _FLOAT_DIRECTIVE_WIDTHS[op] * len(operands)
            if operands and all(_float_literal_valid(value) for value in operands)
            else None
        )
    if op in {".uleb128", ".sleb128"}:
        if not operands:
            return None
        return _leb128_size(operands, op == ".sleb128", aliases)
    if op in {".space", ".zero", ".skip"}:
        value = _layout_int(operands[0], aliases) if len(operands) in {1, 2} else None
        fill = len(operands) == 2 and operands[1].strip()
        if fill and _layout_int(operands[1], aliases) is None:
            return None
        return value if value is not None and value >= 0 else None
    if op == ".fill":
        if len(operands) not in {1, 2, 3}:
            return None
        repeat = _layout_int(operands[0], aliases)
        width = _layout_int(operands[1], aliases) if len(operands) > 1 and operands[1].strip() else 1
        fill = len(operands) == 3 and operands[2].strip()
        if fill and _layout_int(operands[2], aliases) is None:
            return None
        if repeat is None or width is None or repeat < 0 or width < 0:
            return None
        return repeat * width
    if op == ".org":
        value = _layout_int(operands[0], aliases) if len(operands) in {1, 2} else None
        fill = len(operands) == 2 and operands[1].strip()
        if fill and _layout_int(operands[1], aliases) is None:
            return None
        return value if value is not None and value >= 0 else None
    if op in {".align", ".p2align", ".p2alignw", ".p2alignl", ".balign", ".balignw", ".balignl"}:
        if not operands:
            return None
        if not 1 <= len(operands) <= 3:
            return None
        value = _layout_int(operands[0], aliases)
        fill = len(operands) > 1 and operands[1].strip()
        if fill and _layout_int(operands[1], aliases) is None:
            return None
        max_skip = _layout_int(operands[2], aliases) if len(operands) == 3 and operands[2].strip() else None
        if value is None or value < 0 or len(operands) == 3 and operands[2].strip() and (max_skip is None or max_skip < 0):
            return None
        alignment = value if op.startswith(".balign") else 1 << value
        if op.startswith(".balign") and value and value & (value - 1):
            return None
        if alignment < 0:
            return None
        alignment = alignment or 1
        padding = (-pc) % alignment
        if (max_skip is None or padding <= max_skip) and (
            op in {".balignw", ".p2alignw"} and padding % 2
            or op in {".balignl", ".p2alignl"} and padding % 4
        ):
            return None
        return padding if max_skip is None or padding <= max_skip else 0
    return 0


def _facts(
    source: str, source_path: object = None, run_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(source, str):
        raise ValueError("program source must be text")
    if run_params is not None and not isinstance(run_params, Mapping):
        raise ValueError("run_params must be an object")
    ops, labels, control_ops, dataflow, edges = [], [], [], [], []
    label_blocks, label_sections, branch_targets, label_positions = {}, {}, [], []
    all_labels: set[str] = set()
    block_order, last_ops, block_sections = [], {}, {}
    items, op_items, op_blocks = [], [], []
    run_params = {} if run_params is None else run_params
    isa_value, isa_profile_value = run_params.get("isa"), run_params.get("isa_profile")
    isa = isa_value if isa_value is not None else isa_profile_value
    active_isa = isa.lower() if isinstance(isa, str) else None
    xlen = 32 if active_isa and active_isa.startswith("rv32") else 64
    configured_xlen = xlen if active_isa is not None else None
    compressed = profile_uses_compressed_encoding(active_isa) if active_isa else False
    option_stack = []
    source_lines = _logical_source_lines(_strip_c_comments(source))
    reserved_labels = {
        label for line in source_lines for label in _split_line(line)[0]
    }
    section, previous, stack, block = ".text", None, [], "entry"
    section_executable, block_restored = True, False
    section_transitions, unsupported = [], []
    if isa_value is not None and isa_profile_value is not None and str(isa_value).lower() != str(isa_profile_value).lower():
        unsupported.append("isa-conflict")
    if isa is not None and not isinstance(isa, str):
        unsupported.append("isa-invalid")
    elif active_isa is not None and not is_canonical_isa_profile(active_isa):
        unsupported.append("isa-invalid")

    symbols: dict[str, int | str] = _collect_equ_aliases(source_lines)
    data_pcs: dict[str, int] = {}
    ended = False

    def resolve_symbol(value: str, alias_map: Mapping[str, int | str] | None = None) -> int | str:
        alias_map = symbols if alias_map is None else alias_map
        seen = set()
        while isinstance(value, str) and value in alias_map and value not in seen:
            seen.add(value)
            value = alias_map[value]
        return value

    def define_symbol(body: str) -> None:
        parts = body.split(None, 1)
        if len(parts) != 2 or parts[0].lower() not in {".equ", ".set"}:
            return
        operands = _split_operands(parts[1])
        if len(operands) == 2:
            name, value = operands[0].strip(), operands[1].strip()
            symbols[name] = _int_value(value) if _int_value(value) is not None else value

    def section_info(body: str) -> tuple[str, bool] | None:
        parts = body.split(None, 1)
        if len(parts) != 2:
            return None
        raw = parts[1].strip()
        name = raw.split(",", 1)[0].strip().strip('"')
        if not name:
            return None
        flags = re.search(r',\s*"([^"]*)"', raw)
        executable = "x" in flags[1] if flags else name == ".text" or name.startswith(".text.")
        return name, executable

    def advance_section(value: str) -> bool:
        nonlocal section, previous, section_executable, block, block_restored
        before = (section, section_executable)
        block_restored = False
        _, body = _split_line(value)
        words = body.lower().split()
        directive = words[0] if words else None
        if directive == ".text" or directive and directive.startswith(".text."):
            previous, section = (section, section_executable), ".text"
            if directive != ".text":
                section = directive
            section_executable = True
        elif directive == ".section":
            parsed = section_info(body)
            if parsed is None:
                unsupported.append("section-invalid")
                return False
            previous, (section, section_executable) = (section, section_executable), parsed
        elif directive == ".pushsection":
            parsed = section_info(body)
            if parsed is None:
                unsupported.append("section-invalid")
                return False
            stack.append((section, previous, block, section_executable))
            previous, (section, section_executable) = (section, section_executable), parsed
        elif directive == ".popsection":
            if len(words) != 1 or not stack:
                unsupported.append("section-invalid")
                return False
            section, previous, block, section_executable = stack.pop()
            block_restored = True
        elif directive == ".previous":
            if previous is None:
                unsupported.append("section-invalid")
                return False
            current = (section, section_executable)
            section, section_executable = previous
            previous = current
        elif directive in _DATA_SECTIONS:
            previous, section, section_executable = (
                (section, section_executable), directive, False,
            )
        else:
            return False
        if section_executable and not (section == ".text" or section.startswith(".text.")):
            unsupported.append("executable-section-layout")
        section_transitions.append((
            directive, before[0], section, before[1], section_executable,
            previous[0] if previous is not None else None, len(stack),
        ))
        return True

    def fresh_block() -> str:
        index = len(block_order)
        while (candidate := f"b{index}") in block_order or candidate in reserved_labels:
            index += 1
        return candidate

    include_context = tuple(getattr(source_path, "_include_context", ()) or ())
    path_value = source_path if isinstance(source_path, (str, Path)) else getattr(source_path, "program", None)
    try:
        include_base = Path(path_value).resolve().parent if path_value is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("source_path must be path-like") from exc
    alias_sources = include_context
    if not alias_sources and path_value is not None:
        try:
            alias_sources = tuple(_source_include_files(Path(path_value)).items())
        except (OSError, TypeError, ValueError, UnicodeError):
            alias_sources = ()
    binary_names = set()
    source_payloads = (("", source), *alias_sources)
    for name, payload in source_payloads:
        try:
            text = payload if isinstance(payload, str) else payload.decode("utf-8")
        except (AttributeError, UnicodeError):
            continue
        base = posixpath.dirname(str(name).replace("\\", "/"))
        for line in _logical_source_lines(_strip_c_comments(text)):
            for statement in _split_source_statements(line):
                match = _SOURCE_INCBIN_RE.fullmatch(statement.rstrip("\r\n"))
                if match:
                    binary_names.add(posixpath.normpath(posixpath.join(base, match[2])))
    for name, payload in alias_sources:
        normalized = posixpath.normpath(str(name).replace("\\", "/"))
        if normalized in binary_names:
            continue
        try:
            text = payload.decode("utf-8")
        except (AttributeError, UnicodeError):
            continue
        for alias, value in _collect_equ_aliases(
            _logical_source_lines(_strip_c_comments(text))
        ).items():
            symbols.setdefault(alias, value)

    def set_isa(value: str) -> None:
        nonlocal active_isa, compressed, xlen
        active_isa = value.lower()
        xlen = 32 if active_isa.startswith("rv32") else 64
        compressed = profile_uses_compressed_encoding(active_isa)
        if configured_xlen is not None and xlen != configured_xlen:
            unsupported.append("isa-invalid")

    def update_option(body: str) -> None:
        nonlocal active_isa, compressed, xlen
        normalized = re.sub(r"\s+", " ", body.strip().lower())
        if normalized == ".option rvc":
            if active_isa:
                enabled = set(enabled_extensions(active_isa))
                enabled.add("c")
                set_isa(isa_profile_for_extensions(
                    "rv32" if active_isa.startswith("rv32") else "rv64",
                    frozenset(enabled),
                ))
            compressed = True
            return
        if normalized == ".option norvc":
            compressed = False
            return
        if normalized == ".option push":
            option_stack.append((active_isa, compressed, xlen))
            return
        if normalized == ".option pop":
            if option_stack:
                active_isa, compressed, xlen = option_stack.pop()
            else:
                unsupported.append("option-invalid")
            return
        if normalized in {".option relax", ".option norelax", ".option pic", ".option nopic"}:
            return
        match = re.fullmatch(r"\.option\s+arch\s*,\s*(.+)", normalized)
        if match is None:
            unsupported.append("option-invalid")
            return
        value = match[1]
        if re.fullmatch(r"rv(?:32|64)[a-z0-9_]+", value) and is_canonical_isa_profile(value):
            set_isa(value)
            return
        deltas = tuple(item.strip() for item in value.split(","))
        if not active_isa or not deltas or any(
            re.fullmatch(r"[+-][a-z][a-z0-9]*", item) is None for item in deltas
        ):
            unsupported.append("option-invalid")
            return
        enabled = set(enabled_extensions(active_isa))
        for delta in deltas:
            (enabled.add if delta[0] == "+" else enabled.discard)(delta[1:])
        try:
            candidate = isa_profile_for_extensions(
                "rv32" if active_isa.startswith("rv32") else "rv64",
                frozenset(enabled),
            )
            if not is_canonical_isa_profile(candidate):
                raise ValueError("non-canonical ISA")
            previous_compressed = compressed
            compressed_added = any(
                delta[0] == "+" and (
                    delta[1:] == "c" or delta[1:].startswith("zc")
                )
                for delta in deltas
            )
            set_isa(candidate)
            if not compressed_added:
                compressed = previous_compressed and profile_uses_compressed_encoding(candidate)
        except ValueError:
            unsupported.append("option-invalid")

    def incbin_size(operands: Sequence[str], base: Path | None = None) -> int | None:
        if not 1 <= len(operands) <= 3:
            return None
        name = operands[0].strip()
        if len(name) < 2 or name[0] != name[-1] or name[0] not in {'"', "'"}:
            return None
        if include_context:
            dependency_name = posixpath.normpath(
                ((base or Path(".")) / name[1:-1]).as_posix()
            )
            data = next(
                (data for path, data in include_context
                 if Path(path).as_posix() == dependency_name),
                None,
            )
            if data is None:
                return None
            total = len(data)
        else:
            root = base or include_base
            if root is None:
                return None
            dependency = (root / name[1:-1]).resolve()
            if not dependency.is_file():
                return None
            total = dependency.stat().st_size
        skip = _layout_int(operands[1], symbols) if len(operands) > 1 and operands[1].strip() else 0
        count = _layout_int(operands[2], symbols) if len(operands) > 2 and operands[2].strip() else total - (skip or 0)
        return count if skip is not None and count is not None and 0 <= skip <= total and 0 <= count <= total - skip else None

    def known_directive(name: str) -> bool:
        return name in _KNOWN_DIRECTIVES or name.startswith(_DIRECTIVES_WITH_PREFIXES) or name.startswith(".text.")

    def scan_include(value: str, base: Path, stack_paths: tuple[Path, ...] = ()) -> None:
        nonlocal block, ended
        _, value = _split_line(value)
        if not re.match(r"^\s*(?:\.include|#\s*include)\b", value, re.IGNORECASE):
            return
        match = re.fullmatch(
            r'(?:\.include|#\s*include)\s+"([^"]+)"[ \t]*(?:#.*|//.*)?',
            value, re.IGNORECASE,
        )
        if match is None:
            unsupported.append("include-source-invalid")
            return
        if include_context:
            dependency = Path(posixpath.normpath(((base or Path(".")) / match[1]).as_posix()))
            if dependency in stack_paths:
                unsupported.append("include-source-missing")
                return
            payload = next(
                (data for name, data in include_context
                 if Path(name).as_posix() == dependency.as_posix()),
                None,
            )
            if payload is None:
                unsupported.append("include-source-missing")
                return
            try:
                included_source = payload.decode("utf-8")
            except UnicodeError:
                unsupported.append("include-source-invalid")
                return
            next_base = Path(dependency).parent
        else:
            dependency = (base / match[1]).resolve()
            if not dependency.is_file() or dependency in stack_paths:
                unsupported.append("include-source-missing")
                return
            try:
                included_source = dependency.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                unsupported.append("include-source-invalid")
                return
            next_base = dependency.parent
        for included_line in _logical_source_lines(_strip_c_comments(included_source)):
            if ended:
                return
            for included_part in _split_source_statements(included_line):
                if ended:
                    return
                part = included_part.strip()
                if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", part, re.IGNORECASE):
                    unsupported.append("preprocessor-condition")
                    continue
                scan_include(part, next_base, (*stack_paths, dependency))
                if ended:
                    return
                if advance_section(part):
                    if not block_restored:
                        block = fresh_block()
                    continue
                included_labels, body = _split_line(part)
                if any(
                    label in all_labels or included_labels.count(label) > 1
                    for label in included_labels if not label.isdigit()
                ):
                    unsupported.append("label-duplicate")
                all_labels.update(label for label in included_labels if not label.isdigit())
                if included_labels and section_executable:
                    if any(
                        label in label_blocks or included_labels.count(label) > 1
                        for label in included_labels if not label.isdigit()
                    ):
                        unsupported.append("label-duplicate")
                    reserved_labels.update(included_labels)
                    labels.extend(included_labels)
                    if (
                        not label_positions
                        or label_positions[-1][1] != len(items)
                        or label_positions[-1][3] != section
                    ):
                        candidate = f"{section}:{included_labels[0]}" if included_labels[0].isdigit() else included_labels[0]
                        block = candidate if candidate not in block_order else fresh_block()
                    for label in included_labels:
                        label_blocks[label] = block
                        label_sections[label] = section
                        label_positions.append((label, len(items), block, section))
                included_words = body.lower().split()
                if included_words and included_words[0] == ".end":
                    ended = True
                    return
                if not section_executable and included_words and included_words[0] == ".incbin":
                    data_operands = _split_operands(
                        body.split(None, 1)[1]
                    ) if len(included_words) > 1 else ()
                    if incbin_size(data_operands, next_base) is None:
                        unsupported.append("incbin-source-missing")
                        return
                if included_words and included_words[0] in {".equ", ".set"}:
                    define_symbol(body)
                if body.lower().startswith(".option"):
                    update_option(body)
                elif included_words and included_words[0] == ".attribute" and re.match(
                    r"\.attribute\s+arch\b", body, re.IGNORECASE,
                ):
                    match = re.fullmatch(
                        r'\.attribute\s+arch\s*,\s*"([^"]+)"',
                        body, re.IGNORECASE,
                    )
                    if match is None or not is_canonical_isa_profile(match[1].lower()):
                        unsupported.append("option-invalid")
                included_directive = included_words[0] if included_words else None
                if included_directive in _CONDITIONAL_DIRECTIVES:
                    unsupported.append("include-code-unresolved")
                if included_directive == ".subsection" and section_executable:
                    unsupported.append("include-code-unresolved")
                if section_executable and body and (
                    not body.startswith((".", "#"))
                    or included_directive in _CODE_DIRECTIVES | _LAYOUT_DIRECTIVES
                    or included_directive in _UNSUPPORTED_DIRECTIVES
                    or not known_directive(included_directive)
                ):
                    unsupported.append("include-code-unresolved")

    for source_line in source_lines:
        if ended:
            break
        for raw in _split_source_statements(source_line):
            if ended:
                break
            if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", raw, re.IGNORECASE):
                unsupported.append("preprocessor-condition")
                continue
            line_labels, line = _split_line(raw)
            line = re.sub(r"^#\s*include\b", "#include", line, flags=re.IGNORECASE)
            words = line.lower().split()
            directive = words[0] if words else None
            if re.match(r'^#include(?=\s|<|"|$)', line, re.IGNORECASE):
                directive = "#include"
            if directive in {".include", "#include"}:
                if include_base is None and not include_context:
                    unsupported.append("include-source-missing")
                else:
                    scan_include(line, None if include_context else include_base)
                line = ""
                if ended:
                    break
            if advance_section(line):
                if not block_restored:
                    block = fresh_block()
            in_text = section_executable
            if directive in {".equ", ".set"}:
                define_symbol(line)
            if directive in _CONDITIONAL_DIRECTIVES:
                unsupported.append("preprocessor-condition")
                continue
            if directive == ".subsection":
                if in_text:
                    unsupported.append("section-invalid")
                continue
            if directive in _UNSUPPORTED_DIRECTIVES and in_text:
                unsupported.append(directive)
                continue
            if directive == ".option":
                update_option(line)
            elif directive == ".attribute":
                match = re.fullmatch(r'\.attribute\s+arch\s*,\s*"([^"]+)"', line, re.IGNORECASE)
                if re.match(r"\.attribute\s+arch\b", line, re.IGNORECASE) and (
                    match is None or not is_canonical_isa_profile(match[1].lower())
                ):
                    unsupported.append("option-invalid")
            if directive and directive.startswith(".") and not known_directive(directive):
                unsupported.append("directive-unknown")
                continue
            if any(
                label in all_labels or line_labels.count(label) > 1
                for label in line_labels if not label.isdigit()
            ):
                unsupported.append("label-duplicate")
            all_labels.update(label for label in line_labels if not label.isdigit())
            if line_labels and in_text:
                if any(
                    label in label_blocks or line_labels.count(label) > 1
                    for label in line_labels if not label.isdigit()
                ):
                    unsupported.append("label-duplicate")
                labels.extend(line_labels)
                if (
                    not label_positions
                    or label_positions[-1][1] != len(items)
                    or label_positions[-1][3] != section
                ):
                    block = f"{section}:{line_labels[0]}" if line_labels[0].isdigit() else line_labels[0]
                for label in line_labels:
                    label_blocks[label] = block
                    label_sections[label] = section
                    label_positions.append((label, len(items), block, section))
            if directive == ".end":
                ended = True
                break
            if not in_text and line:
                data_operands = _split_operands(line.split(None, 1)[1]) if len(words) > 1 else ()
                data_pc = data_pcs.get(section, 0)
                if directive in set(_DATA_DIRECTIVE_WIDTHS) | {".insn"} and not data_operands:
                    unsupported.append("data-directive-invalid")
                elif directive in _DATA_DIRECTIVE_WIDTHS and data_operands and _data_directive_size(
                    1 if directive == ".byte" else _DATA_DIRECTIVE_WIDTHS[directive],
                    data_operands, symbols,
                ) is None:
                    unsupported.append("data-directive-invalid")
                elif directive in _DATA_DIRECTIVE_WIDTHS and data_operands:
                    data_pcs[section] = data_pc + _data_directive_size(
                        1 if directive == ".byte" else _DATA_DIRECTIVE_WIDTHS[directive],
                        data_operands, symbols,
                    )
                elif directive == ".insn" and (
                    _insn_size(data_operands, symbols) is None
                    or len(words) > 1 and not _insn_operand_syntax_valid(words[1])
                ):
                    unsupported.append("data-directive-invalid")
                elif directive == ".insn":
                    data_pcs[section] = data_pc + _insn_size(data_operands, symbols)
                elif directive in _LAYOUT_DIRECTIVES:
                    size = incbin_size(data_operands) if directive == ".incbin" else _layout_size(
                        directive, data_operands, data_pc, symbols,
                    )
                    if size is None:
                        unsupported.append(
                            "incbin-source-missing"
                            if directive == ".incbin" else "layout-directive-invalid"
                        )
                    elif directive == ".org":
                        address = _layout_int(data_operands[0], symbols) if data_operands else None
                        if address is None or address < data_pc:
                            unsupported.append("layout-directive-invalid")
                        else:
                            data_pcs[section] = address
                    else:
                        data_pcs[section] = data_pc + size
            if not in_text or not line:
                continue
            if (not line_labels and block in last_ops
                    and last_ops[block][0] in _CONTROL_OPS):
                block = fresh_block()
            parts = line.split(None, 1)
            op = parts[0].lower().replace("_", ".").rstrip(";")
            operands = _split_operands(parts[1]) if len(parts) > 1 else ()
            if op == ".insn" and len(parts) > 1 \
                    and not _insn_operand_syntax_valid(parts[1]):
                unsupported.append("data-directive-invalid")
            insn_control = None
            if op == ".insn" and operands:
                control = operands[0].split(None, 1)
                control_format = {"sb": "b", "uj": "j"}.get(
                    control[0].lower(), control[0].lower(),
                )
                insn_control = {
                    ("b", 0x63): ".insn.b",
                    ("j", 0x6F): ".insn.j",
                    ("i", 0x67): ".insn.i",
                }.get((control_format, _layout_int(control[1], symbols) if len(control) == 2 else None))
            operands = tuple(_resolve_memory_alias(operand, symbols) for operand in operands)
            if active_isa and active_isa.startswith("rv32e") and any(
                value.startswith("x") and value[1:].isdigit() and int(value[1:]) >= 16
                for value in _registers(" ".join(operands))
            ):
                unsupported.append("register-disabled-for-isa")
            if any(not operand for operand in operands) and op not in {
                ".align", ".p2align", ".p2alignw", ".p2alignl",
                ".balign", ".balignw", ".balignl",
                ".fill", ".space", ".zero", ".skip", ".incbin",
            }:
                unsupported.append("instruction-operands-invalid")
            if op == "fence" and len(parts) > 1 and parts[1].strip() and not operands:
                unsupported.append("instruction-operands-invalid")
            arity = _PROFILE_PSEUDO_ARITIES.get(op, _PROFILE_CONTROL_ARITIES.get(op))
            if arity is not None and len(operands) != arity:
                unsupported.append("instruction-operands-invalid")
            if op == "zext.b" and (
                len(operands) != 2 or any(_register_index(operand) is None for operand in operands)
            ):
                unsupported.append("instruction-operands-invalid")
            if op != ".insn" and not _relocation_valid(op, operands):
                unsupported.append("instruction-operands-invalid")
            runtime = runtime_instruction_spec(op, xlen)
            encodable_operands = (
                operands if op in _CONTROL_OPS and op not in {"jalr", "jr"}
                else tuple(_relocation_neutral(value) for value in operands)
            )
            if not _profile_operands_valid(op, encodable_operands, symbols, xlen=xlen, isa_profile=active_isa):
                unsupported.append("instruction-operands-invalid")
            if op in {"cm.push", "cm.pop", "cm.popret", "cm.popretz"}:
                if not _zcmp_stack_operands_valid(op, operands, xlen, symbols):
                    unsupported.append("instruction-operands-invalid")
            elif runtime is not None and op not in _BASE_EXPANDING_PSEUDOS and not _text_encodable(
                runtime, runtime.operand_roles, encodable_operands, symbols, isa_profile=active_isa
            ):
                target = operands[-1] if operands else ""
                valid_target_alias = operands and op not in {"jalr", "jr"} \
                    and op in _CONTROL_OPS and not _registers(target)
                if not valid_target_alias or not _text_encodable(
                    runtime, runtime.operand_roles, (*operands[:-1], "0"), symbols, isa_profile=active_isa
                ):
                    unsupported.append("instruction-operands-invalid")
            if op in {
                ".align", ".p2align", ".p2alignw", ".p2alignl",
                ".balign", ".balignw", ".balignl",
            } and len(parts) > 1:
                operands = tuple(item.strip() for item in parts[1].split(","))
            if op.startswith(".") and op not in _CODE_DIRECTIVES:
                if op in _LAYOUT_DIRECTIVES:
                    if op == ".incbin":
                        size = incbin_size(operands)
                        if size is None:
                            unsupported.append("incbin-source-missing")
                        items.append({"kind": "layout", "op": op, "operands": operands, "size": size or 0, "section": section})
                    else:
                        if _layout_size(op, operands, 0, symbols) is None:
                            unsupported.append("layout-directive-invalid")
                        items.append({
                            "kind": "layout", "op": op, "operands": operands,
                            "size": _layout_size(op, operands, 0, symbols) or 0,
                            "aliases": dict(symbols),
                            "section": section,
                        })
                continue
            if op in _CODE_DIRECTIVES - {".insn"} and not operands:
                unsupported.append("data-directive-invalid")
                continue
            if block not in block_order:
                block_order.append(block)
                block_sections[block] = section
            source_op = _source_mnemonic(op)
            spec = instruction_spec_for_generation(
                "rev8.rv32" if source_op == "rev8" and xlen == 32 else source_op
            )
            if not _pseudo_enabled(op, active_isa):
                unsupported.append("instruction-disabled-for-isa")
            if op in {"beqi", "bnei"} and active_isa is not None \
                    and "zibi" not in enabled_extensions(active_isa):
                unsupported.append("instruction-disabled-for-isa")
            if op not in _CODE_DIRECTIVES and spec is None and op not in _PSEUDO_OPS and op not in _CONTROL_OPS:
                unsupported.append("instruction-unknown")
            if spec is not None and active_isa is not None and (
                not profile_enables_form(active_isa, spec.form)
                or op.startswith(("c.", "cm.")) and not compressed
            ) and op not in _BASE_EXPANDING_PSEUDOS:
                unsupported.append("instruction-disabled-for-isa")
            if op in _FP_PSEUDOS and not _fp_pseudo_enabled(op, active_isa):
                unsupported.append("instruction-disabled-for-isa")
            if op == "c.unimp" and not compressed:
                unsupported.append("instruction-disabled-for-isa")
            size_operands = operands
            if op == "li" and len(operands) == 2:
                value = _layout_int(operands[1], symbols)
                if isinstance(value, int):
                    size_operands = (operands[0], str(value))
                elif operands[1].strip().startswith("%") and _relocation_valid(op, operands):
                    size_operands = (operands[0], "0")
                else:
                    unsupported.append("layout-unresolved")
            elif op == "li":
                unsupported.append("layout-unresolved")
            size_operands = tuple(_resolve_size_operand(value, symbols) for value in size_operands)
            size = (
                _insn_size(size_operands, symbols)
                if op == ".insn"
                else _data_directive_size(
                    1 if op == ".byte" else _DATA_DIRECTIVE_WIDTHS[op],
                    operands, symbols,
                )
            ) if op in _CODE_DIRECTIVES else _instruction_size(
                op, size_operands, compressed, xlen, active_isa,
            )
            if size is None:
                unsupported.append("data-directive-invalid")
                size = 0
            if op == ".insn" and size == 2 and active_isa and not compressed:
                unsupported.append("instruction-disabled-for-isa")
            item = {
                "kind": "op", "op": op, "operands": operands, "size": size,
                "compressed": compressed, "isa": active_isa, "xlen": xlen,
                "ordinal": len(ops), "item_index": len(items), "section": section,
            }
            items.append(item)
            op_items.append(item)
            op_blocks.append(block)
            ops.append(op)
            dataflow.append(_dataflow(op, operands))
            state_operands = tuple(_resolve_size_operand(value, symbols) for value in operands)
            last_ops[block] = (insn_control or op, state_operands)
            if op in _CONTROL_OPS or insn_control is not None:
                control_ops.append(insn_control or op)
                if operands and op not in _REGISTER_JUMP_OPS and (
                    op in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS
                    or op in _DIRECT_TARGET_OPS
                    or insn_control in {".insn.b", ".insn.j"}
                ):
                    target = operands[-1]
                    branch_targets.append((
                        block, target, len(ops) - 1, insn_control or op, state_operands,
                        item["item_index"], dict(symbols),
                    ))

    labels_at: dict[int, list[tuple[str, str, str]]] = {}
    for label, position, label_block, label_section in label_positions:
        labels_at.setdefault(position, []).append((label, label_block, label_section))

    def rebuild_layout() -> tuple[list[int], dict[str, int], dict[tuple[str, str], list[tuple[int, str, int]]], dict[tuple[str, int], str]]:
        op_pcs, label_pcs, local_points, pc_blocks = [0] * len(op_items), {}, {}, {}
        label_sections.clear()
        section_pcs = {}
        for index in range(len(items) + 1):
            for label, label_block, label_section in labels_at.get(index, ()):
                pc = section_pcs.get(label_section, 0)
                label_pcs[label] = pc
                label_sections[label] = label_section
                if label.isdigit():
                    local_points.setdefault((label_section, label), []).append((index, label_block, pc))
            if index == len(items):
                break
            item = items[index]
            section = item["section"]
            pc = section_pcs.get(section, 0)
            item["pc"] = pc
            if item["kind"] == "op":
                op_pcs[item["ordinal"]] = pc
                pc_blocks[(section, pc)] = op_blocks[item["ordinal"]]
                pc += item["size"]
            elif item["op"] == ".org":
                aliases = item.get("aliases", symbols)
                value = _layout_int(item["operands"][0], aliases) if item["operands"] else None
                if value is not None and value >= pc:
                    item["size"] = value - pc
                    pc = value
                elif value is not None:
                    item["size"] = 0
                    unsupported.append("layout-directive-invalid")
            else:
                size = item["size"] if item["op"] == ".incbin" else _layout_size(
                    item["op"], item["operands"], pc, item.get("aliases", symbols),
                )
                if size is None:
                    unsupported.append("layout-directive-invalid")
                    size = 0
                item["size"] = size
                pc += item["size"]
            section_pcs[section] = pc
        return op_pcs, label_pcs, local_points, pc_blocks

    def target_parts(
        target: str, seen: frozenset[str] = frozenset(),
        aliases: Mapping[str, int | str] | None = None,
    ) -> tuple[str | None, int]:
        target = _normalize_target_expression(target)
        if target.strip() == ".":
            return ".", 0
        relative = re.fullmatch(
            r"\.\s*([+-])\s*(.+)",
            target.strip(),
        )
        if relative:
            value = _layout_int(relative[2], aliases)
            if value is None:
                return target.strip(), 0
            return ".", value if relative[1] == "+" else -value
        if (value := _layout_int(target, aliases)) is not None:
            return None, value
        match = re.fullmatch(
            r"([A-Za-z_.$][\w.$]*|\d+[fb])\s*([+-])\s*"
            r"(.+)",
            target.strip(),
        )
        if match:
            value = _layout_int(match[3], aliases)
            if value is None:
                return target.strip(), 0
            base, offset = match[1], value if match[2] == "+" else -value
        else:
            base, offset = target.strip(), 0
        if base in seen:
            return base, offset
        resolved = resolve_symbol(base, aliases)
        if isinstance(resolved, str) and resolved != base:
            nested_base, nested_offset = target_parts(resolved, seen | {base}, aliases)
            return nested_base, nested_offset + offset
        return (None, resolved + offset) if isinstance(resolved, int) else (resolved, offset)

    def target_pc(
        target: str, item_index: int, source_pc: int,
        label_pcs: Mapping[str, int], local_points: Mapping[tuple[str, str], Sequence[tuple[int, str, int]]],
        aliases: Mapping[str, int | str] | None = None,
        relative: bool = False,
    ) -> int | None:
        base, offset = target_parts(target, aliases=aliases)
        if base is None:
            return source_pc + offset if relative else offset
        if base == ".":
            return source_pc + offset
        match = re.fullmatch(r"(\d+)([fb])", base.lower())
        if match:
            points = local_points.get((items[item_index]["section"], match[1]), ())
            points = [point for point in points if point[0] > item_index] if match[2] == "f" else [
                point for point in points if point[0] <= item_index
            ]
            return (points[0] if match[2] == "f" else points[-1])[2] + offset if points else None
        return label_pcs.get(base, None) + offset if base in label_pcs else None

    def target_block(
        target: str, item_index: int, source_pc: int,
        label_pcs: Mapping[str, int], local_points: Mapping[tuple[str, str], Sequence[tuple[int, str, int]]],
        pc_blocks: Mapping[tuple[str, int], str], section: str,
        aliases: Mapping[str, int | str] | None = None,
        relative: bool = False,
    ) -> str | None:
        base, offset = target_parts(target, aliases=aliases)
        if base is None:
            if relative:
                offset += source_pc
            return pc_blocks.get((section, offset))
        if base == ".":
            return pc_blocks.get((section, source_pc + offset))
        match = re.fullmatch(r"(\d+)([fb])", base.lower())
        if match:
            points = local_points.get((section, match[1]), ())
            points = [point for point in points if point[0] > item_index] if match[2] == "f" else [
                point for point in points if point[0] <= item_index
            ]
            if points:
                point = points[0] if match[2] == "f" else points[-1]
                if offset == 0:
                    return point[1]
                return pc_blocks.get((section, point[2] + offset))
        if base not in label_pcs:
            return None
        if offset == 0:
            target = label_blocks.get(base)
            return target if target in block_order else None
        target_section = label_sections.get(base, section)
        return pc_blocks.get((target_section, label_pcs[base] + offset))

    for _ in range(len(op_items) + 1):
        op_pcs, label_pcs, local_points, pc_blocks = rebuild_layout()
        for _, target, ordinal, op, _, item_index, aliases in branch_targets:
            if op not in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS | {".insn.b", ".insn.j"}:
                continue
            base, _ = target_parts(target, aliases=aliases)
            if base not in (None, "."):
                continue
            address = target_pc(
                target, item_index, op_pcs[ordinal], label_pcs, local_points, aliases,
                True,
            )
            if address is None:
                continue
            delta = address - op_pcs[ordinal]
            if op in {"c.beqz", "c.bnez"}:
                valid = -256 <= delta <= 254
                reason = "compressed-control-out-of-range"
            elif op in {"c.j", "c.jal"}:
                valid = -2048 <= delta <= 2046
                reason = "compressed-control-out-of-range"
            elif op in _BRANCH_IMMEDIATE_OPS or op == ".insn.b":
                valid = -4096 <= delta <= 4094
                reason = "branch-immediate-out-of-range"
            else:
                valid = -(1 << 20) <= delta <= (1 << 20) - 2
                reason = "jump-immediate-out-of-range"
            if not valid:
                unsupported.append(reason)
        widened = False
        for _, target, ordinal, op, _, item_index, aliases in branch_targets:
            item = op_items[ordinal]
            if op not in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS:
                continue
            size = item["size"]
            if size not in {2, 4, 6}:
                continue
            address = target_pc(
                target, item_index, op_pcs[ordinal], label_pcs, local_points, aliases,
                op in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS | {".insn.b", ".insn.j"},
            )
            if address is None:
                if size == 2:
                    unsupported.append("compressed-control-target-missing")
                continue
            if op in _JUMP_IMMEDIATE_OPS:
                if size != 2:
                    continue
                delta = address - op_pcs[ordinal]
                if -2048 <= delta <= 2046:
                    continue
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    unsupported.append("compressed-control-out-of-range")
                    continue
                item["size"] = 4
                widened = True
                continue
            if size == 2:
                delta = address - op_pcs[ordinal]
                if -256 <= delta <= 254:
                    continue
                if -4096 <= delta <= 4094:
                    new_size = 4
                elif -(1 << 20) <= delta <= (1 << 20) - 2:
                    new_size = 6
                else:
                    unsupported.append("compressed-control-out-of-range")
                    continue
            elif size == 4:
                delta = address - op_pcs[ordinal]
                if -4096 <= delta <= 4094:
                    continue
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    unsupported.append("branch-immediate-out-of-range")
                    continue
                new_size = 8
            else:
                delta = address - (op_pcs[ordinal] + 2)
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    unsupported.append("branch-immediate-out-of-range")
                    continue
                continue
            item["size"] = new_size
            widened = True
        if not widened:
            break
    for _, target, ordinal, op, _, item_index, aliases in branch_targets:
        base, _ = target_parts(target, aliases=aliases)
        if base is None:
            address = target_pc(
                target, item_index, op_pcs[ordinal],
                label_pcs, local_points, aliases,
                op in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS | {".insn.b", ".insn.j"},
            )
            source_pc = op_pcs[ordinal]
            if op in _BRANCH_IMMEDIATE_OPS or op == ".insn.b":
                source_pc += 4 if op_items[ordinal]["size"] >= 8 else 0
            if address is not None and (
                    address % 2 or not -1048576 <= address - source_pc <= 1048574
            ):
                unsupported.append(
                    "branch-immediate-out-of-range"
                    if op in _BRANCH_IMMEDIATE_OPS or op == ".insn.b"
                    else "jump-immediate-out-of-range"
                )
    op_pcs, label_pcs, local_points, pc_blocks = rebuild_layout()
    for item in op_items:
        if item["op"] in _CODE_DIRECTIVES - {".insn"}:
            continue
        if item["pc"] % (2 if profile_uses_compressed_encoding(item["isa"]) else 4):
            unsupported.append("instruction-pc-misaligned")
    for _, target, ordinal, op, _, item_index, aliases in branch_targets:
        if op not in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS \
                and op not in {".insn.b", ".insn.j"}:
            continue
        if op_items[ordinal]["size"] == 2:
            continue
        base, _ = target_parts(target, aliases=aliases)
        if base is None:
            continue
        address = target_pc(
            target, item_index, op_pcs[ordinal], label_pcs, local_points, aliases,
            op in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS | {".insn.b", ".insn.j"},
        )
        if address is None:
            continue
        source_pc = op_pcs[ordinal]
        if op in _BRANCH_IMMEDIATE_OPS or op == ".insn.b":
            size = op_items[ordinal]["size"]
            limit = 1 << 20 if size == 6 or size >= 8 else 4096
            source_pc += 2 if size == 6 else 4 if size >= 8 else 0
        else:
            limit = 1 << 20
        delta = address - source_pc
        if delta & 1 or not -limit <= delta <= limit - 2:
            unsupported.append(
                "branch-immediate-out-of-range"
                if op in _BRANCH_IMMEDIATE_OPS or op == ".insn.b"
                else "jump-immediate-out-of-range"
            )
    for source_block, target, ordinal, op, operands, item_index, aliases in branch_targets:
        resolved_block = target_block(
            target, item_index, op_pcs[ordinal], label_pcs, local_points, pc_blocks,
            op_items[ordinal]["section"], aliases,
            op in _BRANCH_IMMEDIATE_OPS | _JUMP_IMMEDIATE_OPS | {".insn.b", ".insn.j"},
        )
        if resolved_block is None:
            base, _ = target_parts(target, aliases=aliases)
            if base not in (None, "."):
                unsupported.append("control-target-unresolved")
        elif op in {".insn.b", ".insn.j"} or _branch_state(op, operands) is not False:
            edges.append((source_block, resolved_block))
    direct_ops = {"j", "jal", "jalr", "jr", "ret", "call", "tail",
                  "c.j", "c.jal", "c.jr", "c.jalr", "c.ebreak", "cm.jt", "cm.jalt", "cm.popret",
                  "cm.popretz", "ecall", "ebreak", "scall", "sbreak", "unimp", "c.unimp",
                  "mret", "sret", "wfi", ".insn.j", ".insn.i"}
    for current_section in dict.fromkeys(block_sections.values()):
        section_blocks = [
            name for name in block_order if block_sections.get(name) == current_section
        ]
        for left, right in zip(section_blocks, section_blocks[1:]):
            op, operands = last_ops[left]
            if op not in direct_ops and _branch_state(op, operands) is not True:
                edges.append((left, right))
    unique_edges = tuple(sorted(dict.fromkeys(edges)))
    return {
        "ops": tuple(ops), "length": len(ops), "labels": tuple(labels),
        "control_ops": tuple(control_ops), "dataflow": tuple(dataflow),
        "edges": unique_edges, "cfg_shape": (len(block_order), len(unique_edges)),
        "section_transitions": tuple(section_transitions),
        "unsupported": tuple(dict.fromkeys(unsupported)),
    }


def _distance(left: Sequence[object], right: Sequence[object]) -> float:
    return 1.0 - SequenceMatcher(None, tuple(left), tuple(right)).ratio()


def _canonical_action_pool(actions: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    unique = {}
    try:
        iterator = iter(actions)
    except TypeError:
        return ()
    for action in iterator:
        if not isinstance(action, Mapping):
            continue
        try:
            value = dict(action)
            digest = canonical_digest(value)
        except (TypeError, ValueError):
            continue
        unique.setdefault(digest, value)
    return tuple(unique[digest] for digest in sorted(unique))


def _profile_facts(base: dict[str, object], owner: object, profile: object | None) -> dict[str, object]:
    if profile is None or getattr(profile, "source_sha256", None) != program_sha256(owner):
        return base
    raw_rows, raw_blocks, raw_edges = (
        getattr(profile, "instructions", ()),
        getattr(profile, "blocks", ()),
        getattr(profile, "edges", ()),
    )
    if (
        not isinstance(raw_rows, (list, tuple))
        or not isinstance(raw_blocks, (list, tuple))
        or not isinstance(raw_edges, (list, tuple))
    ):
        raise ValueError("MCMC profile facts invalid")
    rows, blocks, edges = tuple(raw_rows), tuple(raw_blocks), tuple(raw_edges)
    if (
        not all(isinstance(row, Mapping) for row in rows)
        or not all(isinstance(block, str) and block for block in blocks)
        or not all(
            isinstance(edge, (list, tuple)) and len(edge) == 2
            and all(isinstance(block, str) and block for block in edge)
            for edge in edges
        )
    ):
        raise ValueError("MCMC profile facts invalid")
    labeled_blocks = {row.get("block") for row in rows if row.get("labels")}
    profile_edges = tuple(sorted(
        (next(
            (str(label) for row in rows if row.get("block") == left
             for label in row.get("labels", ()) if label),
            left,
        ), next(
            (str(label) for row in rows if row.get("block") == right
             for label in row.get("labels", ()) if label),
            right,
        ))
        for left, right in edges
    ))
    profile_cfg = bool(edges) and len(labeled_blocks) == len(blocks) \
        and profile_edges == tuple(sorted(base["edges"])) and not any(
        isinstance(facts := row.get("rule_facts"), Mapping)
        and (
            facts.get("control_shape") == "indirect"
            or facts.get("control_shape") in {"conditional", "direct"}
            and _int_value((row.get("operands") or ("",))[-1]) is not None
        )
        for row in rows
    )
    if profile_cfg:
        base["edges"] = profile_edges
    if blocks and profile_cfg:
        base["cfg_shape"] = (len(blocks), len(edges))
    if rows:
        base["dataflow"] = tuple(
            _dataflow(str(row.get("mnemonic", "")), tuple(row.get("operands", ())))
            for row in rows
        )
    return base


def _single_boundary_score(program: object, original: object) -> float:
    """单指令候选的连续结构分数：按风险指令的操作数域偏离程度打分。

    只返回 0/1 两档会让所有候选密度相同、alpha 恒为 1.0，MCMC 退化成
    等概率随机游走。风险指令的 mnemonic/寄存器/立即数/掩码位才是候选真正
    改变的边界语义，这里按「改变了几个字段」给出可分档的分数。
    """
    def fields(value: object) -> tuple[tuple[object, ...], ...]:
        params = getattr(value, "run_params", {}) or {}
        case = params.get("single_case") or {}
        rows = []
        for item in case.get("instruction_meta") or ():
            if not isinstance(item, Mapping) or "risk" not in (item.get("tags") or ()):
                continue
            rows.append((
                str(item.get("mnemonic")), item.get("rd"), item.get("rs1"),
                item.get("rs2"), item.get("rs3"), item.get("immediate"),
                tuple(tuple(pair) for pair in (item.get("operand_fields") or ())),
            ))
        return tuple(rows)

    current, base = fields(program), fields(original)
    if current == base:
        return 0.0
    if len(current) != len(base):
        return 2.0
    changed = sum(
        1 for left, right in zip(current, base)
        if left != right
    )
    return min(2.0, float(changed))

def structure_score(program: object, original: object, profile: object | None = None) -> float:
    if (getattr(program, "run_params", {}).get("route") == "single"
            or getattr(original, "run_params", {}).get("route") == "single"):
        try:
            return _single_boundary_score(program, original)
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0.0 if program_sha256(program) == program_sha256(original) else 1.0
    base = _profile_facts(
        _facts(_text(original), original, getattr(original, "run_params", None)),
        original, profile,
    )
    current_source_path = program
    if isinstance(program, ProgramVariant) and getattr(original, "program", None) is not None:
        current_source_path = SimpleNamespace(
            program=original.program,
            _include_context=getattr(program, "_include_context", ()),
        )
    current = _profile_facts(
        _facts(_text(program), current_source_path,
               getattr(program, "run_params", None)),
        program, profile,
    )
    unsupported = tuple(dict.fromkeys((*base["unsupported"], *current["unsupported"])))
    if unsupported:
        return 0.0  # 目录外 form 无静态事实模型，平坦密度让链走完预算
    control = sum(_distance(base[name], current[name]) for name in ("control_ops", "labels", "edges", "cfg_shape", "section_transitions")) / 5.0
    length_cost = abs(base["length"] - current["length"]) / max(1, base["length"])
    return max(0.0, 0.25 * _distance(base["ops"], current["ops"])
               + 0.25 * control + 0.25 * _distance(base["dataflow"], current["dataflow"])
               - 0.25 * length_cost)


def _log_density(
    program: object, original: object, beta: float, profile: object | None = None,
    feedback_score: float = 0.0,
) -> float:
    _validate_beta(beta)
    value = beta * (structure_score(program, original, profile) + feedback_score)
    if not math.isfinite(value):
        raise ValueError("beta produces an unrepresentable log density")
    return value


def sample_program(
    original: object,
    reference_equal: Callable[[object, ProgramVariant], bool | str],
    *, steps: int = FRAMEWORK_CANONICAL_STEPS,
    seed: int = 1,
    beta: float = 1.0,
    profile: object | None = None,
    refresh_profile: Callable[[ProgramVariant], object] | None = None,
    mcmc: bool = True,
    emi_enabled: bool = True,
    allowed_rule: str | None = None,
    profile_actions_fn: Callable[[object, object], Sequence[Mapping[str, object]]] = profile_actions,
    enumerate_rewrites_fn: Callable[[object], Sequence[Mapping[str, object]]] = lambda _p: (),
    apply_profile_fn: Callable[..., object] = apply_profile,
    feedback: Callable[[ProgramVariant], float | None] | None = None,
    target_feedback: Callable[[ProgramVariant], float | None] | None = None,
    feedback_source: str = "target",
    max_seconds: float | None = None,
    allow_reference_gaps: bool = False,
) -> dict[str, object]:
    """在完整程序状态上做 EMI proposal；有观测时反馈分数参与 MH 密度。"""
    if (
        type(seed) is not int
        or type(steps) is not int
        or steps < 0
        or type(emi_enabled) is not bool
        or type(mcmc) is not bool
        or type(allow_reference_gaps) is not bool
    ):
        raise ValueError("invalid MCMC arguments")
    if not emi_enabled and mcmc:
        raise ValueError("MCMC requires EMI proposals; disable both modules together")
    _validate_beta(beta)
    if not callable(reference_equal):
        raise ValueError("reference_equal must be callable")
    if refresh_profile is not None and not callable(refresh_profile):
        raise ValueError("refresh_profile must be callable")
    if feedback is not None and target_feedback is not None:
        raise ValueError("feedback and target_feedback are mutually exclusive")
    feedback_fn = feedback if feedback is not None else target_feedback
    if feedback_fn is not None and not callable(feedback_fn):
        raise ValueError("feedback must be callable")
    if feedback_source not in {"target", "reference"}:
        raise ValueError("feedback_source must be target or reference")
    raw_run_params = getattr(original, "run_params", {})
    if not isinstance(raw_run_params, Mapping):
        raise ValueError("run_params must be an object")
    run_params = dict(raw_run_params)
    if profile is None and refresh_profile is not None:
        raise ValueError("refresh_profile requires profile")
    if profile is not None and steps > 1 and refresh_profile is None:
        raise ValueError("profile-gap:refresh_profile-required")
    initial = (
        original
        if isinstance(original, ProgramVariant)
        else ProgramVariant(
            _text(original), program_sha256(original), "{}",
            run_params,
        )
    )
    fixed_bounds = {name for name in ("entry", "exit") if name in run_params}

    def stable_identity(value: object) -> dict[str, object]:
        result = _stable_input_identity(value)
        for name in ("entry", "exit"):
            if name not in fixed_bounds:
                result.pop(name, None)
        return result

    input_identity = getattr(profile, "input_identity", run_params)
    if profile is not None:
        profile_identity, current_identity = stable_identity(input_identity), stable_identity(run_params)
        if (getattr(profile, "source_sha256", None) != program_sha256(original)
                or any(profile_identity.get(k) != v for k, v in current_identity.items())
                or any(k not in current_identity and k not in {"entry", "exit", "harness", "result_channel"} for k in profile_identity)):
            raise ValueError("profile input identity mismatch")
    input_digest = canonical_digest(stable_identity(input_identity))
    profile_state, stopped, events = profile, False, []
    pending: dict[str, object] = {"parent_sha256": initial.sha256}
    rng = random.Random(seed * 100)
    action_pool_cache: dict[tuple[str, str | None], tuple] = {}

    def feedback_value(state: ProgramVariant) -> float | None:
        value = 0.0 if feedback_fn is None else feedback_fn(state)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("feedback must be finite")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("feedback must be finite") from error
        if not math.isfinite(value):
            raise ValueError("feedback must be finite")
        return value

    def action_pool(
        current: ProgramVariant, active: object | None, *, materialize: bool,
    ) -> tuple[
        tuple[Mapping[str, object], ...],
        tuple[str | None, ...], Mapping[str, Mapping[str, object]],
        str, bool, bool,
    ]:
        profile_digest = getattr(active, "profile_digest", None)
        cache_key = (
            current.sha256,
            profile_digest if isinstance(profile_digest, str) else None,
        )
        if materialize and cache_key in action_pool_cache:
            cached = action_pool_cache.pop(cache_key)
            action_pool_cache[cache_key] = cached
            return cached

        def route_pool() -> tuple[Mapping[str, object], ...]:
            try:
                return _canonical_action_pool(enumerate_rewrites_fn(current))
            except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                return ()

        # profile 缺失（reference 域缺口）时不再直接空池：EMI 动作只依赖
        # case 本身，MCMC 仍要能提出候选。
        if active is not None:
            try:
                all_actions = _canonical_action_pool(
                    profile_actions_fn(current, active),
                )
            except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                # A stale/incomplete reference profile must not make the
                # proposal pool disappear.  The route enumerator only needs
                # the case itself and is the authoritative fallback.
                all_actions = route_pool()
            if not all_actions:
                # A complete profile may still have no executed witness for
                # this route (for example a target-side trap or a backend with
                # no PC stream).  Keep the trusted case rewrite pool live so
                # one missing witness cannot stop the whole chain.
                all_actions = route_pool()
        else:
            all_actions = route_pool()
        bound_actions = tuple(
            action for action in all_actions
            if allowed_rule is None or action.get("rule") == allowed_rule
        )
        growth_actions = tuple(
            action for action in all_actions
            if (
                action.get("operator") == "insert"
                and action.get("mutation_kind") == "instruction-insert"
            ) or (
                action.get("operator") == "delete_line"
                and action.get("mutation_kind") == "instruction-delete"
            )
        )
        rule_fallback = bool(
            allowed_rule is not None and not bound_actions and all_actions
        )
        actions = (
            all_actions if rule_fallback else
            (*bound_actions, *growth_actions) if allowed_rule is not None else
            all_actions
        )
        actions = _canonical_action_pool(actions)
        action_index = {
            canonical_digest(dict(action)): action for action in actions
        }
        pool_digest = canonical_digest(list(actions))
        if not materialize:
            return actions, (), action_index, pool_digest, False, rule_fallback
        candidate_hashes = []
        for action in actions:
            action_profile = active if isinstance(action.get("profile_digest"), str) else None
            try:
                candidate = apply_profile_fn(
                    current, action_profile, action, _action_pool=action_index,
                )
            except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                candidate = None
            candidate_hashes.append(
                candidate.sha256 if isinstance(candidate, ProgramVariant) else None
            )
        result = (
            actions, tuple(candidate_hashes), action_index, pool_digest,
            True, rule_fallback,
        )
        action_pool_cache[cache_key] = result
        # ponytail: two-state cache reuses reverse q as next forward q; keeping more pools
        # costs memory without helping a single chain's adjacent-state traversal.
        if len(action_pool_cache) > 2:
            action_pool_cache.pop(next(iter(action_pool_cache)))
        return result

    def proposal_weights(pool: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
        if not pool:
            return ()
        growth = {
            index for index, action in enumerate(pool)
            if (
                action.get("operator") == "insert"
                and action.get("mutation_kind") == "instruction-insert"
            ) or (
                action.get("operator") == "delete_line"
                and action.get("mutation_kind") == "instruction-delete"
            )
        }
        if not growth:
            return tuple(1.0 / len(pool) for _ in pool)
        return tuple(
            0.5 / len(pool) + (0.5 / len(growth) if index in growth else 0.0)
            for index in range(len(pool))
        )

    def sample_index(rng: random.Random, weights: Sequence[float]) -> int:
        draw = rng.random()
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw < cumulative:
                return index
        return len(weights) - 1

    def transition_hashes(
        base: ProgramVariant, hashes: Sequence[str | None],
    ) -> list[str]:
        # EMI failures and no-op rewrites are self-loops in the proposal kernel.
        return [value if value is not None else base.sha256 for value in hashes]

    def pool_record(
        pool: Sequence[Mapping[str, object]], active: object | None,
        candidate_hashes: Sequence[str | None] | None = None,
        pool_digest: str | None = None,
    ) -> dict[str, object]:
        audit_pool = len(pool) <= _ACTION_POOL_AUDIT_LIMIT
        items = [dict(item) for item in pool] if audit_pool else None
        return {
            # Keep the full pool in memory for proposal selection and exact
            # digesting.  Persisting every large pool in every event made a
            # ten-step case consume tens of MB without adding replay facts.
            "action_pool": items,
            "action_pool_elided": not audit_pool,
            "action_pool_size": len(pool),
            "action_pool_digest": pool_digest or canonical_digest(list(pool)),
            "action_candidate_sha256": (
                list(candidate_hashes) if candidate_hashes is not None else []
            ) if audit_pool else None,
            "profile_digest": getattr(active, "profile_digest", None),
        }

    def usable(value: object, candidate: ProgramVariant) -> object | None:
        next_profile = getattr(value, "profile", value)
        if not (getattr(next_profile, "complete", False) and getattr(next_profile, "source_sha256", None) == candidate.sha256):
            return None
        def reference_key(item: object) -> dict[str, object]:
            value = getattr(item, "provenance", {}).get("reference_identity", {})
            return {
                name: field for name, field in value.items()
                if name not in {"binary_sha256", "guest_elf_sha256"}
            } if isinstance(value, Mapping) else {}
        if reference_key(next_profile) != reference_key(profile):
            return None
        return next_profile if canonical_digest(stable_identity(getattr(next_profile, "input_identity", {}))) == input_digest else None

    def prepare_refresh(candidate: ProgramVariant) -> object | None:
        try:
            value = refresh_profile(candidate)
            next_profile = usable(value, candidate)
        except Exception as error:
            next_profile = None
            value = error
        if next_profile is None:
            pending["profile_refresh"] = {
                "status": "gap",
                "reason": getattr(value, "reason", None)
                or f"profile-gap:{value or 'refresh-unavailable'}",
            }
            return None
        pending["candidate_profile"] = next_profile
        return next_profile

    def install_refresh(candidate: ProgramVariant) -> bool:
        nonlocal profile_state
        next_profile = pending.pop("candidate_profile", None)
        if next_profile is None:
            refresh = pending.get("profile_refresh")
            if allow_reference_gaps and pending.pop("profile_gap_fallback", False):
                # The candidate already passed the configured observation
                # stage. When the reference cannot model its ISA, keep the
                # accepted feedback-driven state and switch the next proposal
                # to the case-only route enumerator until a profile is usable.
                profile_state = None
                pending["profile_refresh"] = refresh if isinstance(refresh, Mapping) else {
                    "status": "gap", "reason": "profile-gap:incomplete",
                }
                return True
            pending["profile_refresh"] = refresh if isinstance(refresh, Mapping) else {
                "status": "gap", "reason": "profile-gap:incomplete",
            }
            return False
        pending["profile_refresh"] = {
            "status": "ready", "source_sha256": candidate.sha256,
            "digest": next_profile.profile_digest,
        }
        profile_state = next_profile
        return True

    def propose(rng: random.Random, current: ProgramVariant) -> MHProposal:
        nonlocal profile_state, stopped
        (
            pool, forward_hashes, action_index, forward_pool_digest,
            forward_complete, rule_fallback,
        ) = action_pool(
            current, profile_state, materialize=mcmc,
        )
        q_status = "exact" if mcmc else "off"
        if not pool:
            bound_rule_miss = allowed_rule is not None and not events and not rule_fallback
            pending.clear()
            pending.update({
                "action": None, "candidate": None, "parent_sha256": current.sha256,
                **pool_record(
                    pool, profile_state, forward_hashes, forward_pool_digest,
                ),
                "action_digest": None, "proposal_index": None,
                "gate": (
                    "seed-miss:bound-rule-inapplicable"
                    if bound_rule_miss else "proposal-gap:action-pool-empty"
                ),
                "q_status": q_status,
            })
            if rule_fallback:
                pending["proposal_rule_fallback"] = {
                    "requested": allowed_rule, "applied": "any-applicable",
                }
            if bound_rule_miss:
                pending["proposal_reason"] = f"no-action-for-bound-rule:{allowed_rule}"
            stopped = True
            return MHProposal(REJECTED, q_status=pending["q_status"])
        weights = proposal_weights(pool) if mcmc else ()
        index = sample_index(rng, weights) if mcmc else rng.randrange(len(pool))
        action = pool[index]
        action_profile = (
            profile_state
            if isinstance(action.get("profile_digest"), str)
            else None
        )
        candidate_recomputation_gap = False
        if mcmc:
            expected_candidate_sha256 = forward_hashes[index]
            if expected_candidate_sha256 is None:
                candidate = None
            else:
                try:
                    candidate = apply_profile_fn(
                        current, action_profile, action, _action_pool=action_index,
                    )
                except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                    candidate = None
                if (
                    not isinstance(candidate, ProgramVariant)
                    or candidate.sha256 != expected_candidate_sha256
                ):
                    candidate = None
                    candidate_recomputation_gap = True
        else:
            try:
                candidate = apply_profile_fn(
                    current, action_profile, action,
                    _action_pool={
                        canonical_digest(dict(item)): item for item in pool
                    },
                )
            except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                candidate = None
        pending.clear()
        pending.update({
            "action": dict(action), "candidate": candidate,
            "action_digest": canonical_digest(dict(action)), "proposal_index": index,
            "parent_sha256": current.sha256,
            **pool_record(
                pool, profile_state,
                candidate_hashes=forward_hashes if mcmc else None,
                pool_digest=forward_pool_digest,
            ),
        })
        if rule_fallback:
            pending["proposal_rule_fallback"] = {
                "requested": allowed_rule, "applied": "any-applicable",
            }
        if candidate is None:
            pending["gate"] = (
                "proposal-gap:candidate-recomputation-mismatch"
                if candidate_recomputation_gap else "emi-rejected"
            )
            if candidate_recomputation_gap:
                pending["proposal_reason"] = "forward-candidate-digest-mismatch"
            pending["q_status"] = q_status
            return MHProposal(REJECTED, q_status=q_status)
        if candidate.sha256 == current.sha256:
            pending["gate"] = "proposal-self-loop"
            pending["q_status"] = q_status
            return MHProposal(REJECTED, q_status=q_status)
        try:
            result = reference_equal(current, candidate)
        except Exception as error:
            # reference 回调失败只影响当前候选。开放能力链把它记成
            # reference gap，继续使用配置的 feedback/结构密度做 MH。
            pending["reference_gap"] = {
                "status": "gap", "reason": f"{type(error).__name__}: {error}",
            }
            if not allow_reference_gaps:
                pending["gate"], pending["q_status"] = "reference-rejected", q_status
                return MHProposal(REJECTED, q_status=q_status)
            result = "profile-gap"
        if result is False or result == "reference-rejected":
            # 语义不等价是正确性拒绝，不是开放能力缺口或搜索奖励。
            if feedback_source == "reference":
                pending["reference_feedback"] = None
                pending["reference_feedback_status"] = "semantic-rejected"
            pending["gate"], pending["q_status"] = "reference-rejected", q_status
            return MHProposal(REJECTED, q_status=q_status)
        if not allow_reference_gaps and isinstance(result, str) and result in {
            "reference-rejected", "reference-gap", "profile-gap",
        }:
            pending["gate"], pending["q_status"] = str(result), q_status
            return MHProposal(REJECTED, q_status=q_status)
        valid_gaps = {
            "reference-gap", "profile-gap",
            "transport-gap", "target-gap", "target-probe-gap",
            "case-skipped", "artifact-gap", "unsupported-isa",
        }
        if result is not True and (
                not isinstance(result, str) or result not in valid_gaps
        ):
            if not allow_reference_gaps:
                pending["gate"], pending["q_status"] = "reference-rejected", q_status
                return MHProposal(REJECTED, q_status=q_status)
            pending["reference_gap"] = {
                "status": "gap", "reason": f"unexpected-result:{result}",
            }
            result = "profile-gap"
        if result is not True:
            if allow_reference_gaps and result == "reference-gap":
                pending["reference_gap"] = {"status": "gap", "reason": result}
                result = "profile-gap"
            elif allow_reference_gaps and result == "profile-gap":
                pending["reference_gap"] = {"status": "gap", "reason": result}
            pending["gate"] = result
            pending["q_status"] = q_status
            # 参考域缺口可以开放继续；执行失败、Target gap 和非法输入保持拒绝。
            if pending["gate"] != "profile-gap":
                return MHProposal(REJECTED, q_status=q_status)
        candidate_feedback = feedback_value(candidate)
        if feedback_fn is not None:
            pending[f"{feedback_source}_feedback"] = candidate_feedback
            pending[f"{feedback_source}_feedback_status"] = (
                "observed" if candidate_feedback is not None else "unavailable"
            )
        candidate_profile = None
        if refresh_profile is not None:
            candidate_profile = prepare_refresh(candidate)
            if candidate_profile is None:
                if not allow_reference_gaps:
                    pending["gate"] = "reference-passed"
                    pending["q_status"] = q_status
                    if mcmc:
                        pending["proposal_reason"] = "reverse-q-unavailable"
                    return MHProposal(REJECTED, q_status=pending["q_status"])
                # Keep the reference-observed candidate eligible for MH. The next
                # pool falls back to route.enumerate_rewrites until a usable
                # profile is available again.
                pending["profile_gap_fallback"] = True
                pending["proposal_reason"] = "profile-refresh-gap:route-fallback"
        if mcmc:
            try:
                pending["candidate_log_density"] = _log_density(
                    candidate, original, beta, profile,
                    0.0 if candidate_feedback is None else candidate_feedback,
                )
            except ValueError as error:
                # 结构事实缺失退化为平坦密度；其余 ValueError 仍是真错误。
                if not str(error).startswith("MCMC static facts unsupported:"):
                    raise
                pending["candidate_log_density"] = 0.0
        if mcmc:
            (
                reverse_pool, reverse_hashes, _reverse_action_index,
                reverse_pool_digest, reverse_complete, reverse_rule_fallback,
            ) = action_pool(
                candidate, candidate_profile, materialize=True,
            )
            if reverse_rule_fallback:
                pending["reverse_proposal_rule_fallback"] = {
                    "requested": allowed_rule, "applied": "any-applicable",
                }
            if not forward_complete or not reverse_complete:
                pending.update({
                    "gate": "proposal-gap:incomplete-q",
                    "proposal_reason": "complete-action-probability-unavailable",
                    "q_status": q_status,
                })
                return MHProposal(REJECTED, q_status=q_status)
            forward_transition_hashes = transition_hashes(current, forward_hashes)
            reverse_transition_hashes = transition_hashes(candidate, reverse_hashes)
            reverse_weights = proposal_weights(reverse_pool)
            q_forward = sum(
                weight for digest, weight in zip(forward_transition_hashes, weights)
                if digest == candidate.sha256
            )
            q_reverse = sum(
                weight for digest, weight in zip(reverse_transition_hashes, reverse_weights)
                if digest == current.sha256
            )
            if q_forward <= 0.0:
                pending.update({
                    "gate": "proposal-gap:forward-q-zero",
                    "proposal_reason": "selected-candidate-absent-from-forward-pool",
                    "q_status": q_status,
                })
                return MHProposal(REJECTED, q_status=q_status)
            pending.update({
                "action_candidate_sha256": (
                    forward_hashes if len(pool) <= _ACTION_POOL_AUDIT_LIMIT else None
                ),
                "reverse_action_pool": (
                    [dict(item) for item in reverse_pool]
                    if len(reverse_pool) <= _ACTION_POOL_AUDIT_LIMIT else None
                ),
                "reverse_action_pool_elided": (
                    len(reverse_pool) > _ACTION_POOL_AUDIT_LIMIT
                ),
                "reverse_action_pool_size": len(reverse_pool),
                "reverse_action_pool_digest": reverse_pool_digest,
                "reverse_candidate_sha256": (
                    reverse_hashes
                    if len(reverse_pool) <= _ACTION_POOL_AUDIT_LIMIT else None
                ),
                "log_q_forward": math.log(q_forward),
                "log_q_reverse": math.log(q_reverse) if q_reverse > 0.0 else -math.inf,
                "proposal_reason": "reverse-q-zero" if q_reverse == 0.0 else None,
                "q_status": q_status,
            })
        if pending.get("gate") is None:
            pending["gate"] = "reference-passed"
        return MHProposal(
            candidate, pending.get("log_q_forward"), pending.get("log_q_reverse"), q_status,
        )

    def refresh_state(candidate: ProgramVariant) -> bool | float:
        try:
            return install_refresh(candidate)
        except Exception as error:
            reason = str(error)
            pending["profile_refresh"] = {"status": "gap", "reason": reason if reason.startswith("profile-gap:") else f"profile-gap:{reason}"}
            return False

    def trace(step: dict[str, object]) -> bool:
        # ponytail: events keep transition facts; proposal diagnostics are not runtime state.
        candidate = pending.get("candidate")
        accepted = bool(step["accepted"])
        gate = pending.get("gate")
        event = {
            "step": step["step"], "action": pending.get("action"),
            "parent_sha256": pending.get("parent_sha256"),
            "candidate_sha256": candidate.sha256 if isinstance(candidate, ProgramVariant) else None,
            "state_sha256": candidate.sha256 if accepted and isinstance(candidate, ProgramVariant) else pending.get("parent_sha256"),
            "action_pool": pending.get("action_pool"),
            "action_pool_elided": pending.get("action_pool_elided", False),
            "action_pool_size": pending.get("action_pool_size"),
            "action_pool_digest": pending.get("action_pool_digest"),
            "action_candidate_sha256": pending.get("action_candidate_sha256"),
            "proposal_index": pending.get("proposal_index"),
            "action_digest": pending.get("action_digest"),
            "profile_digest": pending.get(
                "profile_digest", getattr(profile_state, "profile_digest", None)
            ),
            "rho_s_digest": getattr(profile, "profile_digest", None),
            "reverse_action_pool": pending.get("reverse_action_pool"),
            "reverse_action_pool_elided": pending.get(
                "reverse_action_pool_elided", False,
            ),
            "reverse_action_pool_size": pending.get("reverse_action_pool_size"),
            "reverse_action_pool_digest": pending.get("reverse_action_pool_digest"),
            "reverse_candidate_sha256": pending.get("reverse_candidate_sha256"),
            "gate": gate, "status": None,
            "feedback_source": feedback_source,
            "target_feedback": pending.get("target_feedback"),
            "reference_feedback": pending.get("reference_feedback"),
            "reference_path_score": (
                pending.get("reference_feedback")
                if feedback_source == "reference" else None
            ),
            "reference_reward_status": (
                pending.get("reference_feedback_status")
                if feedback_source == "reference" else None
            ),
            "emi_enabled": emi_enabled,
            "mh_accepted": step.get("mh_accepted", accepted) if mcmc else None,
            **({"state_accepted": step["state_accepted"]}
               if mcmc and step.get("mh_attempt") is True else {}),
            "alpha": step["alpha"], "log_q_forward": step["log_q_forward"],
            "log_q_reverse": step["log_q_reverse"], "q_status": step["q_status"],
            "log_pi_current": step.get("log_pi_current"),
            "log_pi_candidate": step.get("log_pi_candidate"),
            "log_alpha": step.get("log_alpha"),
            "mh_attempt": step.get("mh_attempt", False),
            "accept_draw": step.get("accept_draw"),
        }
        refresh_gap = isinstance(pending.get("profile_refresh"), Mapping) and pending["profile_refresh"].get("status") == "gap"
        refresh_failed = refresh_gap
        # reference 域只建模 rv64i 标量子集；域外候选的 reference 判定是诊断
        # 字段而非终态（合同 §reference 域不阻断 case）。MH 已经尝试推进，
        # status 必须反映真实接受结果，否则链跑通也被读成 reference-rejected。
        mh_attempted = step.get("mh_attempt") is True
        event["status"] = (
            "accepted" if accepted else
            "mh-rejected" if mh_attempted else
            "profile-gap" if gate == "reference-passed" and refresh_failed else
            "seed-miss" if isinstance(gate, str) and gate.startswith("seed-miss:") else
            "proposal-gap" if isinstance(gate, str) and gate.startswith("proposal-gap:") else
            gate
        )
        if pending.get("proposal_reason"):
            event["proposal_reason"] = pending["proposal_reason"]
        if pending.get("reference_gap"):
            event["reference_gap"] = pending["reference_gap"]
        if pending.get("profile_gap_fallback"):
            event["profile_gap_fallback"] = True
        if pending.get("proposal_rule_fallback"):
            event["proposal_rule_fallback"] = pending["proposal_rule_fallback"]
        if pending.get("reverse_proposal_rule_fallback"):
            event["reverse_proposal_rule_fallback"] = pending["reverse_proposal_rule_fallback"]
        if pending.get("profile_refresh"):
            event["profile_refresh"] = pending["profile_refresh"]
        events.append(event)
        return stopped

    def density(state: ProgramVariant) -> float:
        if state is pending.get("candidate") and "candidate_log_density" in pending:
            return pending["candidate_log_density"]
        score = feedback_value(state)
        return _log_density(
            state, original, beta, profile, 0.0 if score is None else score,
        )

    if not emi_enabled:
        run = MHRun((), 0)
    elif mcmc:
        # 结构事实缺失只影响密度取值，不再终止链。
        run = mh_chain(
            rng, initial, propose, chain_len=steps, on_step=trace,
            on_accept=refresh_state if refresh_profile is not None else None,
            log_density=density, max_seconds=max_seconds,
        )
    else:
        current, samples = initial, []
        started = time.monotonic()
        for step in range(steps):
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
            proposed = propose(rng, current)
            candidate = proposed.candidate
            step_accepted = candidate is not REJECTED
            if step_accepted:
                previous = current
                if refresh_profile is not None and refresh_state(candidate) is False:
                    step_accepted, current = False, previous
                else:
                    current = candidate
            trace({
                "step": step, "accepted": step_accepted, "alpha": None,
                "log_pi_current": None, "log_pi_candidate": None, "log_alpha": None,
                "log_q_forward": None, "log_q_reverse": None, "q_status": "off", "mh_attempt": False,
            })
            samples.append(current)
            if stopped:
                break
        run = MHRun(tuple(samples), 0)
    final = run.samples[-1] if run.samples else initial
    if profile_state is not None and profile_state.source_sha256 != program_sha256(final):
        profile_state = None
    return {
        "run": run, "events": events, "final": final, "profile": profile_state,
        "step_budget": steps, "steps_completed": len(events),
        "seed": seed, "rng_scheme": _RNG_SCHEME,
        "emi_enabled": emi_enabled,
    }


__all__ = ["sample_program", "structure_score"]

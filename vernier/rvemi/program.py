"""完整程序的最小 EMI 变异入口。"""

import ast
import hashlib
import json
import os
import posixpath
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

from .._util import (
    _PROTECTED_ENTRY_LABELS as _M5_PROTECTED_LABELS,
    _path_in_root, canonical_digest, is_sha256_digest, object_field as _profile_get,
    strip_c_comments as _strip_c_comments,
)
from ..direct_case import native_trace_stdout_mismatch_is_compatible
from ..riscv_encoding import implicit_source_roles, source_roles as _encoding_source_roles
from ..riscv_encoding import operand_groups
from ..adapters.runner import canonical_trap_state, guest_trap_observed
from ..spec_definedness import (
    enabled_extensions,
    control_no_effect_target_for_generation,
    direct_jump_target_for_branch,
    FP_EQUAL_SOURCES,
    FP_NEGATIVE_RS1,
    FP_POSITIVE_RS1,
    FP_ZERO_ADDEND,
    FP_ZERO_PRODUCT,
    fma_sign_selectors,
    full_w_view_guard,
    fp_family_equivalence_condition,
    fp_form_family_siblings,
    fp_operand_width_bits,
    gpr_result_width_bits,
    high_multiply_signed_source_operands,
    instruction_effect_class,
    instruction_spec_for_generation,
    encode_instruction_for_generation,
    isa_profile_for_extensions,
    is_canonical_isa_profile,
    load_width_bytes,
    profile_enables_form,
    profile_uses_compressed_encoding,
    signedness_mode_for_generation,
    x_register_fp_extensions,
    xregister_compatibility_slot_domain_for_mnemonic,
)
from .runtime_spec import runtime_instruction_spec


_MEMORY_OPERAND_RE = re.compile(
    r"\s*([+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|[0-9]+))"
    r"\s*\(\s*([^()]+?)\s*\)\s*"
)
_PROGRAM_GROWTH_MARKER = "rq1-mcmc-growth"
_MEMORY_BASE_RE = re.compile(
    r"\s*(?:[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|[0-9]+))?"
    r"\s*\(\s*([^()]+?)\s*\)\s*"
)


def _source_mnemonic(mnemonic: str) -> str:
    value = mnemonic.lower()
    base, separator, suffix = value.rpartition(".")
    if separator and suffix in {"aq", "rl", "aqrl"} and base.startswith(("amo", "lr.", "sc.")):
        return base
    if separator and suffix == "aqrl":
        if base in {"lb", "lh", "lw", "ld"}:
            return f"{base}.aq"
        if base in {"sb", "sh", "sw", "sd"}:
            return f"{base}.rl"
    if separator and suffix.isdigit():
        number = int(suffix)
        if base == "mop.r" and number in range(32):
            return "mop.r.N"
        if base == "mop.rr" and number in range(8):
            return "mop.rr.N"
        if base == "c.mop" and number in range(1, 16, 2):
            return "c.mop.N"
    return value

@dataclass(frozen=True)
class ProgramProfile:
    """供 program EMI 使用的一条扁平 profile 记录。"""

    source_sha256: str
    input_identity: Mapping[str, object]
    instructions: tuple[Mapping[str, object], ...]
    executed_ids: tuple[str, ...]
    observer_fields: tuple[str, ...]
    provenance: Mapping[str, object]
    blocks: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()
    occurrences: tuple[Mapping[str, object], ...] = ()
    route: str = "program"
    dynamic_witness: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.input_identity, Mapping) or not isinstance(self.provenance, Mapping):
            raise ValueError("profile identities must be objects")
        if self.route not in {"program", "single"} or type(self.dynamic_witness) is not bool:
            raise ValueError("profile route is invalid")
        if not isinstance(self.observer_fields, (list, tuple)) or any(
            not isinstance(field, str) or not field for field in self.observer_fields
        ):
            raise ValueError("observer_fields must be non-empty strings")
        try:
            json.dumps(self.input_identity, allow_nan=False)
            json.dumps(self.provenance, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("profile identities must contain JSON values") from exc
        for name in ("instructions", "occurrences", "blocks", "edges"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise ValueError(f"profile {name} must be a sequence")
        if any(not isinstance(item, Mapping) for item in (*self.instructions, *self.occurrences)):
            raise ValueError("profile instruction and occurrence rows must be objects")
        if any(
            not isinstance(edge, (list, tuple)) or len(edge) != 2
            for edge in self.edges
        ):
            raise ValueError("profile edges must be pairs")
        if not isinstance(self.executed_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in self.executed_ids
        ):
            raise ValueError("executed_ids must be a sequence of non-empty strings")
        object.__setattr__(self, "executed_ids", tuple(self.executed_ids))
        object.__setattr__(self, "observer_fields", tuple(self.observer_fields))
        object.__setattr__(self, "input_identity", dict(self.input_identity))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "instructions", tuple(dict(item) for item in self.instructions))
        object.__setattr__(self, "occurrences", tuple(dict(item) for item in self.occurrences))

    def _payload(self) -> dict[str, object]:
        instructions = []
        for row in self.instructions:
            item = dict(row)
            if not item.get("_aliases"):
                item.pop("_aliases", None)
            instructions.append(item)
        return {
            "source_sha256": self.source_sha256,
            "input_identity": dict(self.input_identity),
            "instructions": instructions,
            "executed_ids": list(self.executed_ids),
            "observer_fields": list(self.observer_fields),
            "provenance": {
                key: value for key, value in self.provenance.items()
                if key != "profile_digest"
            },
            "blocks": list(self.blocks),
            "edges": [list(edge) for edge in self.edges],
            "occurrences": list(self.occurrences),
        }

    @property
    def profile_digest(self) -> str:
        if self.route == "single":
            return canonical_digest({
                "contract": "rvemi-single-profile-v2",
                "source_sha256": self.source_sha256,
                "input_identity": dict(self.input_identity),
                "observer_fields": list(self.observer_fields),
                "provenance": {
                    key: value for key, value in self.provenance.items()
                    if key != "profile_digest"
                },
                "dynamic_witness": self.dynamic_witness,
            })
        return canonical_digest({"contract": "rvemi-program-profile-v1", **self._payload()})

    def to_record(self) -> dict[str, object]:
        if self.route == "single":
            provenance = {
                key: value for key, value in self.provenance.items()
                if key != "profile_digest"
            }
            return {
                "route": "single", "source_sha256": self.source_sha256,
                "profile_digest": self.profile_digest,
                "input_identity": dict(self.input_identity),
                "observer_fields": list(self.observer_fields),
                "provenance": provenance,
                "dynamic_witness": self.dynamic_witness,
            }
        return {"digest": self.profile_digest, **self._payload()}

    @property
    def complete(self) -> bool:
        recorded_digest = self.provenance.get("profile_digest")
        if self.route == "single":
            if recorded_digest is not None and recorded_digest != self.profile_digest:
                return False
            execution = self.provenance.get("execution_trace")
            reference = self.provenance.get("reference_identity")
            return (
                is_sha256_digest(self.source_sha256)
                and is_sha256_digest(self.profile_digest)
                and {"outcome", "executed_pcs"} <= set(self.observer_fields)
                and isinstance(execution, Mapping)
                and execution.get("complete") is True
                and type(execution.get("count")) is int
                and execution["count"] > 0
                and is_sha256_digest(execution.get("pc_digest"))
                and is_sha256_digest(self.provenance.get("observation_digest"))
                and isinstance(self.provenance.get("testcase_id"), str)
                and bool(self.provenance["testcase_id"].strip())
                and isinstance(reference, Mapping)
                and isinstance(reference.get("backend"), str)
                and bool(reference["backend"].strip())
                and all(
                    field in {
                        "outcome", "executed_pcs", "instruction_count", "trace_complete", "gpr", "pc",
                        "checkpoint_pc", "memory_snapshot", "memory_digest", "extra_state", "state_trace",
                        *_TRACE_WITNESS_FIELDS,
                    }
                    for field in self.observer_fields
                )
            )
        if not is_sha256_digest(recorded_digest) or recorded_digest != self.profile_digest:
            return False
        rows = self.instructions
        isa = self.input_identity.get("isa") or self.input_identity.get("isa_profile")
        if (
            self.input_identity.get("isa") is not None
            and self.input_identity.get("isa_profile") is not None
            and str(self.input_identity["isa"]).lower()
            != str(self.input_identity["isa_profile"]).lower()
        ):
            return False
        base_xlen = 32 if isinstance(isa, str) and isa.lower().startswith("rv32") else 64
        alignment = 2 if _isa_has_extension(str(isa), "c") or any(
            row.get("compressed") for row in rows
        ) else 4
        if not (is_sha256_digest(self.source_sha256)
                and rows and self.executed_ids and self.occurrences):
            return False
        if not all(
            isinstance(row.get("id"), str) and row["id"]
            and isinstance(row.get("block"), str) and row["block"]
            and isinstance(row.get("labels", ()), (list, tuple))
            and all(isinstance(label, str) and label for label in row.get("labels", ()))
            and isinstance(row.get("operands", ()), (list, tuple))
            and all(isinstance(operand, str) for operand in row.get("operands", ()))
            for row in rows
        ):
            return False
        if not all(
            isinstance(edge[0], str) and edge[0]
            and isinstance(edge[1], str) and edge[1]
            for edge in self.edges
        ):
            return False
        if not all(
            isinstance(row.get("id"), str) and row["id"]
            and isinstance(row.get("instruction_id"), str) and row["instruction_id"]
            and isinstance(row.get("block"), str) and row["block"]
            and type(row.get("ordinal")) is int
            for row in self.occurrences
        ):
            return False
        ids = tuple(row.get("id") for row in rows)
        blocks = {row.get("block") for row in rows}
        occurrence_keys = tuple(item.get("id") for item in self.occurrences)
        occurrence_ids = tuple(item.get("instruction_id") for item in self.occurrences)
        reference = self.provenance.get("reference_identity")
        by_id = {row.get("id"): row for row in rows}
        execution = self.provenance.get("execution_trace")
        source_spans = tuple((row.get("start"), row.get("end")) for row in rows)
        aliases = tuple(row.get("_aliases", {}) for row in rows)
        def valid_transition(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
            if self.provenance.get("execution_path") == "program-test-path-relaxed":
                return True
            left_row, right_row = by_id.get(left.get("instruction_id")), by_id.get(right.get("instruction_id"))
            left_pc, right_pc = _pc_value(left.get("pc")), _pc_value(right.get("pc"))
            if left.get("instruction_id") == right.get("instruction_id") \
                    and left_pc == right_pc:
                return False
            return left_row is not None and right_row is not None \
                and left_pc is not None and right_pc is not None \
                and not _trace_transition_invalid(
                    left_pc, left_row, right_pc, right_row, rows,
                    left_row.get("_aliases"),
                )
        dynamic_edges = tuple(dict.fromkeys(
            (left.get("block"), right.get("block"))
            for left, right in zip(self.occurrences, self.occurrences[1:])
            if left.get("block") != right.get("block")
            or left.get("mnemonic") in {"jr", "jalr", "c.jr", "c.jalr", "cm.jt", "cm.jalt"}
        ))
        return (
            isinstance(isa, str)
            and is_canonical_isa_profile(isa.lower())
            and all(isinstance(item, str) and item for item in ids)
            and len(set(ids)) == len(ids)
            and self.blocks == tuple(dict.fromkeys(row.get("block") for row in rows))
            and all(isinstance(item, str) and item for item in blocks)
            and self.executed_ids == tuple(
                row["id"] for row in rows if row.get("executed") is True
            )
            and all(isinstance(item, str) and item for item in occurrence_keys)
            and len(set(occurrence_keys)) == len(occurrence_keys)
            and all(
                isinstance(row.get("block"), str)
                and row["block"] in blocks
                and type(row.get("line")) is int
                and (pc := _pc_value(row.get("pc"))) is not None
                and (pc_end := _pc_value(row.get("pc_end"))) is not None
                and pc_end > pc
                and isinstance(row.get("mnemonic"), str)
                and isinstance(row.get("isa"), str)
                and is_canonical_isa_profile(row["isa"].lower())
                and isinstance(row.get("section"), str) and row["section"]
                and isinstance(row.get("translation_cell"), str) and row["translation_cell"]
                and type(row.get("logical_line")) is int and row["logical_line"] > 0
                and type(row.get("compressed")) is bool
                and (not row["compressed"] or profile_uses_compressed_encoding(row["isa"]))
                and type(row.get("executed")) is bool
                and (
                    (spec := runtime_instruction_spec(row["mnemonic"], row.get("xlen"))) is None
                    or row["mnemonic"] in _BASE_EXPANDING_PSEUDOS
                    or profile_enables_form(row["isa"], spec.shared_spec.form)
                )
                and type(row.get("xlen")) is int and row["xlen"] == base_xlen
                and isinstance(row.get("rule_facts"), Mapping)
                and {
                    key: value for key, value in row["rule_facts"].items()
                    if key != "relations"
                } == _rule_facts(row["mnemonic"], row["xlen"])
                and (
                    row.get("mnemonic") in _DATA_DIRECTIVE_WIDTHS
                    or pc % alignment == 0
                )
                for row in rows
            )
            and all(
                type(start) is int and type(end) is int and 0 <= start < end
                for start, end in source_spans
            )
            and all(isinstance(item, Mapping) for item in aliases)
            and source_spans == tuple(sorted(source_spans))
            and all(left[1] <= right[0] for left, right in zip(source_spans, source_spans[1:]))
            and all(
                left.get("section") != right.get("section")
                or _pc_value(left.get("pc_end")) <= _pc_value(right.get("pc"))
                for left, right in zip(rows, rows[1:])
            )
            and {"outcome", "executed_pcs"} <= set(self.observer_fields)
            and self.input_identity.get("route") in (None, self.route)
            and isinstance(execution, Mapping)
            and type(execution.get("complete")) is bool
            and type(execution.get("count")) is int
            and (
                execution.get("complete") is True
                or execution.get("source") == "unattested"
            )
            and execution.get("count") == len(self.occurrences)
            and set(occurrence_ids) == set(self.executed_ids)
            and all(
                valid_transition(left, right)
                for left, right in zip(self.occurrences, self.occurrences[1:])
            )
            and tuple(item.get("ordinal") for item in self.occurrences) == tuple(range(len(self.occurrences)))
            and all(
                isinstance(item.get("instruction_id"), str)
                and item.get("instruction_id") in ids
                and item.get("block") == by_id[item["instruction_id"]].get("block")
                and item.get("mnemonic") == by_id[item["instruction_id"]].get("mnemonic")
                and (pc := _pc_value(item.get("pc"))) is not None
                and pc % alignment == 0
                and (
                    pc == _pc_value(by_id[item["instruction_id"]].get("pc"))
                    or _final_layout_valid(by_id[item["instruction_id"]])
                    and pc in tuple(_pc_value(value) for value in by_id[item["instruction_id"]]["final_pcs"])
                    or _pc_value(by_id[item["instruction_id"]].get("pc")) <= pc
                    < _pc_value(by_id[item["instruction_id"]].get("pc_end"))
                )
                and _pc_value(by_id[item["instruction_id"]].get("pc")) <= pc
                < _pc_value(by_id[item["instruction_id"]].get("pc_end"))
                for item in self.occurrences
            )
            and all(
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and edge[0] in blocks and edge[1] in blocks
                for edge in self.edges
            )
            and set(self.edges) == set((*_profile_edges(
                tuple(rows), self.blocks, self.occurrences,
                rows[0].get("_aliases", {}) if rows else {},
            ), *dynamic_edges))
            and all(
                field in {
                    "outcome", "executed_pcs", "instruction_count", "trace_complete", "gpr", "pc",
                    "checkpoint_pc", "memory_snapshot", "memory_digest", "extra_state", "state_trace",
                    *_TRACE_WITNESS_FIELDS,
                }
                for field in self.observer_fields
            )
            and _pc_value(self.input_identity.get("entry")) == _pc_value(self.occurrences[0].get("pc"))
            and _pc_value(self.input_identity.get("exit")) in {
                _pc_value(self.occurrences[-1].get("pc")),
                _pc_value(by_id[self.occurrences[-1]["instruction_id"]].get("pc_end")),
            }
            and isinstance(reference, Mapping)
            and isinstance(reference.get("backend"), str)
            and bool(reference["backend"].strip())
            and all(value not in (None, "") for value in reference.values())
            and all(
                is_sha256_digest(reference[name])
                for name in ("binary_sha256", "guest_elf_sha256")
                if reference.get(name)
            )
        )


@dataclass(frozen=True)
class ProfileResult:
    profile: ProgramProfile | None
    reason: str | None = None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.fullmatch(
            r"([+-]?)(0[bB][01]+|0[xX][0-9a-fA-F]+|0[0-7]+|[1-9][0-9]*|0)",
            value.strip(),
        )
        if match is None:
            return None
        digits = match[2]
        base = 2 if digits.lower().startswith("0b") else (
            16 if digits.lower().startswith("0x") else
            8 if len(digits) > 1 and digits.startswith("0") else 10
        )
        return (-1 if match[1] == "-" else 1) * int(digits, base)
    return value if isinstance(value, int) else None


def _integer_expression(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    if '"' in value:
        return None
    text = re.sub(
        r"(?<![A-Za-z0-9_.$])0([0-7]+)(?![A-Za-z0-9_.$])", r"0o\1",
        re.sub(r"!(?!=)", " not ", value.strip().replace("&&", " and ").replace("||", " or ")).strip(),
    )
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    def divide(left: int, right: int) -> int:
        if right == 0:
            raise ValueError("division by zero")
        return (-1 if (left < 0) != (right < 0) else 1) * (abs(left) // abs(right))

    def remainder(left: int, right: int) -> int:
        return left - divide(left, right) * right

    binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: divide,
        ast.Mod: remainder,
        ast.LShift: lambda left, right: left << right,
        ast.RShift: lambda left, right: left >> right,
        ast.BitOr: lambda left, right: left | right,
        ast.BitAnd: lambda left, right: left & right,
        ast.BitXor: lambda left, right: left ^ right,
    }

    def evaluate(item):
        if isinstance(item, ast.Constant) and type(item.value) is int:
            return item.value
        if isinstance(item, ast.Constant) and type(item.value) is str and len(item.value) == 1:
            return ord(item.value)
        if isinstance(item, ast.BoolOp) and type(item.op) in {ast.And, ast.Or}:
            values = tuple(evaluate(value) for value in item.values)
            if any(value is None for value in values):
                return None
            return int(all(values) if type(item.op) is ast.And else any(values))
        if isinstance(item, ast.Compare) and len(item.ops) == 1:
            left, right = evaluate(item.left), evaluate(item.comparators[0])
            if left is None or right is None:
                return None
            comparison = {
                ast.Eq: lambda: left == right,
                ast.NotEq: lambda: left != right,
                ast.Lt: lambda: left < right,
                ast.LtE: lambda: left <= right,
                ast.Gt: lambda: left > right,
                ast.GtE: lambda: left >= right,
            }.get(type(item.ops[0]))
            return int(comparison()) if comparison is not None else None
        if isinstance(item, ast.UnaryOp) and type(item.op) in {ast.UAdd, ast.USub, ast.Invert, ast.Not}:
            operand = evaluate(item.operand)
            return None if operand is None else {
                ast.UAdd: lambda value: value,
                ast.USub: lambda value: -value,
                ast.Invert: lambda value: ~value,
                ast.Not: lambda value: int(not value),
            }[type(item.op)](operand)
        if isinstance(item, ast.BinOp) and type(item.op) in binary:
            left, right = evaluate(item.left), evaluate(item.right)
            if left is None or right is None or type(right) is bool:
                return None
            try:
                return binary[type(item.op)](left, right)
            except (ValueError, OverflowError):
                return None
        return None

    return evaluate(node)


def _memory_parts(value: str) -> tuple[int | None, int | None] | None:
    match = _MEMORY_OPERAND_RE.fullmatch(value)
    if match:
        return _int_value(match[1]), _register_index(match[2].strip())
    match = _MEMORY_BASE_RE.fullmatch(value)
    return (0, _register_index(match[1].strip())) if match else None


def _isa_has_extension(isa: object, extension: str) -> bool:
    return extension in enabled_extensions(isa)


def _gpr_state(value: object, xlen: int = 64) -> tuple[int, ...] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 32
            or xlen not in {32, 64}
            or any(_int_value(item) is None for item in value)):
        return None
    result = tuple(_int_value(item) for item in value)
    return result if result[0] == 0 and all(0 <= item < (1 << xlen) for item in result) else None


def _profile_memory_valid(value: object) -> bool:
    if value is None or isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, str):
        return len(value) % 2 == 0 and all(char in "0123456789abcdefABCDEF" for char in value)
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and key and isinstance(item, str)
        and len(item) % 2 == 0
        and all(char in "0123456789abcdefABCDEF" for char in item)
        for key, item in value.items()
    )


_TRACE_WITNESS_FIELDS = (
    "pc", "after_pc", "before_gpr", "after_gpr", "before_fpr_rawbits",
    "after_fpr_rawbits", "before_fflags", "after_fflags", "before_frm",
    "after_frm", "memory_reads", "branch_outcome", "fault_pc",
    "fault_address", "fault", "consumer", "link", "ialign",
)
_TRACE_STATE_FIELDS = frozenset(_TRACE_WITNESS_FIELDS) - {"pc"}
_RULE_TRACE_FIELDS = {
    "R1": {"before_gpr", "after_gpr", "fault_pc", "fault_address", "fault"},
    "R2": {"before_gpr"},
    "R3": {"before_gpr", "after_gpr"},
    "R4": {"before_gpr", "memory_reads", "after_gpr", "fault_pc", "fault_address", "fault"},
    "R6": {"before_gpr", "memory_reads", "after_gpr", "fault_pc", "fault_address", "fault"},
    "R5": {"before_gpr", "after_gpr", "after_pc", "fault_pc", "fault_address", "fault"},
    "R7": {
        "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags", "after_fflags",
        "before_frm", "after_frm", "fault_pc", "fault_address", "fault",
    },
    "R8": {
        "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags", "after_fflags",
        "before_frm", "after_frm", "fault_pc", "fault_address", "fault",
    },
    "R9": {"after_pc", "branch_outcome", "link", "ialign", "fault_pc", "fault_address", "fault"},
    "R10": {"before_gpr", "after_gpr", "after_pc", "consumer", "fault_pc", "fault_address", "fault"},
    "R11": {"before_gpr", "after_gpr", "after_pc", "consumer", "fault_pc", "fault_address", "fault"},
}
_RULE_WITNESS_DOMAINS = {
    **dict.fromkeys(
        ("R1", "R2", "R3", "R5", "R10", "R11"),
        frozenset({
            "before_gpr", "after_gpr", "after_pc", "consumer",
            "fault_pc", "fault_address", "fault",
        }),
    ),
    "R4": frozenset({"before_gpr", "after_gpr", "after_pc", "memory_reads", "fault_pc", "fault_address", "fault"}),
    "R6": frozenset({"before_gpr", "after_gpr", "after_pc", "memory_reads", "fault_pc", "fault_address", "fault"}),
    "R7": frozenset({
        "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags", "after_fflags",
        "before_frm", "after_frm", "after_pc", "fault_pc", "fault_address", "fault",
    }),
    "R8": frozenset({
        "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags", "after_fflags",
        "before_frm", "after_frm", "after_pc", "fault_pc", "fault_address", "fault",
    }),
    "R9": frozenset({
        "before_gpr", "after_pc", "branch_outcome", "link", "ialign",
        "fault_pc", "fault_address", "fault",
    }),
}
_RULE_OBSERVER_GAP_REASONS = {
    "R1": "value-witness-required",
    "R3": "value-witness-required",
    "R5": "effect-witness-required",
    "R4": "private-memory-witness-required",
    "R6": "load-extension-witness-required",
    "R7": "rounding-state-witness-required",
    "R8": "fp-format-witness-required",
    "R9": "control-successor-witness-required",
    "R10": "consumer-witness-required",
    "R11": "width-signedness-witness-required",
}
_OPTIONAL_TRACE_FIELDS = frozenset({"fault_pc", "fault_address", "fault"})


def _trace_entries(observation: object) -> tuple[dict[str, object], ...]:
    trace = _profile_get(observation, "state_trace")
    extra = _profile_get(observation, "extra_state", {})
    volatile = frozenset(
        value for value in extra.get("volatile_gpr_indices", ())
        if type(value) is int and 0 <= value < 32
    ) if isinstance(extra, Mapping) else frozenset()
    if trace is None:
        if not native_trace_stdout_mismatch_is_compatible(extra):
            return ()
        trace = extra.get("state_trace") if isinstance(extra, Mapping) else None
    if trace is None:
        evidence = _profile_get(observation, "translation_evidence")
        details = evidence.get("details", {}) if isinstance(evidence, Mapping) else getattr(evidence, "details", {})
        native = details.get("native_window_checkpoints") if isinstance(details, Mapping) else None
        digest = details.get("native_window_checkpoint_digest") if isinstance(details, Mapping) else None
        if isinstance(native, (list, tuple)) and digest == canonical_digest({"native_window_checkpoints": native}):
            trace = native
        elif native is not None:
            return ({"pc": None, "_invalid": True},)
    # native_window_checkpoints 可能含每次启动都变化的寄存器状态。
    if isinstance(trace, Mapping):
        trace = tuple(trace.values())
    if not isinstance(trace, (list, tuple)):
        return ()
    result = []
    for item in trace:
        data = dict(item) if isinstance(item, Mapping) else {
            name: getattr(item, name) for name in _TRACE_WITNESS_FIELDS
            if hasattr(item, name)
        }
        pc = _pc_value(data.get("pc"))
        if pc is None:
            result.append({"pc": None, "_invalid": True})
            continue
        data["pc"] = pc
        invalid_fields = []
        for name in ("after_pc", "fault_pc", "fault_address"):
            if name in data and data[name] is not None:
                data[name] = _pc_value(data[name])
                if data[name] is None or name != "fault_address" and data[name] & 1:
                    data[name] = None
                    invalid_fields.append(name)
        for name in ("before_fflags", "after_fflags", "before_frm", "after_frm"):
            if name in data and data[name] is not None:
                value = _int_value(data[name])
                valid = value is not None and (
                    name.endswith("fflags") and 0 <= value <= 31
                    or name.endswith("frm") and 0 <= value <= 7
                )
                data[name] = value if valid else None
                if not valid:
                    invalid_fields.append(name)
        for name in ("before_gpr", "after_gpr", "before_fpr_rawbits", "after_fpr_rawbits"):
            if name in data and data[name] is not None:
                values = data[name]
                if isinstance(values, (list, tuple)):
                    parsed = tuple(_int_value(value) for value in values)
                    if name.endswith("gpr") and volatile:
                        parsed = tuple(0 if index in volatile else value
                                       for index, value in enumerate(parsed))
                    valid = not (
                        any(
                            value is None or not 0 <= value < (1 << 64)
                            for value in parsed
                        )
                        or len(parsed) != 32
                    )
                    if name.endswith("gpr"):
                        valid &= _gpr_state(values) is not None
                    data[name] = parsed if valid else None
                    if not valid:
                        invalid_fields.append(name)
                else:
                    data[name] = None
                    invalid_fields.append(name)
        branch_outcome = data.get("branch_outcome")
        if branch_outcome is not None and (
            not isinstance(branch_outcome, str)
            or branch_outcome not in {"taken", "not-taken"}
        ):
            data["branch_outcome"] = None
            invalid_fields.append("branch_outcome")
        if "memory_reads" in data and (
            data["memory_reads"] is None
            or not isinstance(data["memory_reads"], (list, tuple))
            or any(not isinstance(read, Mapping) for read in data["memory_reads"])
        ):
            invalid_fields.append("memory_reads")
        if data.get("fault") is not None:
            try:
                json.dumps(data["fault"], allow_nan=False)
            except (TypeError, ValueError):
                data["fault"] = None
                invalid_fields.append("fault")
        if invalid_fields:
            invalid_fields = tuple(dict.fromkeys(invalid_fields))
            for name in invalid_fields:
                data.pop(name, None)
            data["_invalid_trace_fields"] = invalid_fields
        result.append(data)
    return tuple(result)


def _pc_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    parsed = None
    try:
        parsed = value if isinstance(value, int) else int(value, 0) if isinstance(value, str) else None
    except (TypeError, ValueError):
        if isinstance(value, str) and re.fullmatch(r"0[0-9a-fA-F]+", value):
            parsed = int(value, 16)
    return parsed if parsed is not None and 0 <= parsed < (1 << 64) else None


def _stable_input_identity(value: object) -> dict[str, object]:
    result = dict(value or {})
    if result.get("route") == "single":
        for name in (
            "single_case", "single_case_sha256", "rvgen_testcase_id", "rvgen_base_program_id",
            "observer_fields",
        ):
            result.pop(name, None)
    if "isa" in result and "isa_profile" in result:
        if str(result["isa"]).lower() != str(result["isa_profile"]).lower():
            raise ValueError("conflicting ISA identity")
        result.pop("isa_profile")
    elif "isa_profile" in result:
        result["isa"] = result.pop("isa_profile")
    if isinstance(result.get("isa"), str):
        result["isa"] = result["isa"].lower()
    for name in ("entry", "exit"):
        parsed = _pc_value(result.get(name))
        if parsed is not None:
            result[name] = parsed
    return result


def _li_instruction_sizes(
    value: int, xlen: int = 64, compressed: bool = False, rd: int | None = None,
) -> tuple[int, ...]:
    raw_value = value
    if xlen == 32 and compressed and raw_value >= 1 << 31 \
            and (raw_value & 0xffffffff) >= 0xffffffe0:
        return (4,)
    value &= (1 << xlen) - 1
    if value & (1 << (xlen - 1)):
        value -= 1 << xlen
    low = value & 0xfff
    low -= 0x1000 if low & 0x800 else 0
    upper = value - low

    if xlen > 32 and not -(1 << 31) <= value < (1 << 31):
        shift = 12
        while ((upper >> shift) & 1) == 0:
            shift += 1
        result = _li_instruction_sizes(upper >> shift, xlen, compressed, rd)
        if result and result[0] == 2:
            result = (4, *result[1:])
        result += (2 if compressed and rd not in (None, 0) and shift < xlen else 4,)
        low_size = 2 if compressed and rd not in (None, 0) and (
            -32 <= low <= 31 or rd == 2 and -512 <= low <= 496 and low % 16 == 0
        ) else 4
        result += (low_size,) if low else ()
        return result
    result = ()
    if upper:
        immediate = upper >> 12
        lui_compressed = compressed and rd not in (None, 0, 2) and immediate and -32 <= immediate <= 31
        result += (2 if lui_compressed else 4,)
    if low or not upper:
        result += (2 if compressed and rd not in (None, 0)
                   and -32 <= low <= 31 else 4,)
    return result


def _strip_comments(raw: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(raw):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"\"", "'"}:
            quote = char
        elif char == "#" or raw.startswith("//", index):
            return raw[:index]
    return raw


def _split_source_statements(raw: str) -> tuple[str, ...]:
    code = _strip_comments(raw)
    if ";" not in code:
        return (raw,)
    comment = raw[len(code):]
    parts, start, quote, escaped = [], 0, None, False
    for index, char in enumerate(code):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"\"", "'"}:
            quote = char
        elif char == ";":
            parts.append(code[start:index] + " ")
            start = index + 1
    parts.append(code[start:] + comment)
    return tuple(parts)


def _split_operands(value: str) -> tuple[str, ...]:
    result, start, quote, escaped, braces = [], 0, None, False, 0
    for index, char in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"\"", "'"}:
            quote = char
        elif char == "{":
            braces += 1
        elif char == "}" and braces:
            braces -= 1
        elif char == "," and not braces:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return tuple(result)


def _expand_asm_macros(source: str, *, line_directives: bool = True) -> str:
    raw_lines = source.splitlines()
    lines, line_numbers = [], []
    index = 0
    while index < len(raw_lines):
        start = index + 1
        line = raw_lines[index]
        while line.rstrip().endswith("\\") and index + 1 < len(raw_lines):
            line = line.rstrip()[:-1] + " " + raw_lines[index + 1].lstrip()
            index += 1
        lines.append(line)
        line_numbers.append(start)
        index += 1
    macros, result, index = {}, [], 0
    while index < len(lines):
        match = re.match(r"^\s*\.macro\s+([A-Za-z_.$][\w.$]*)(?:\s+(.*?))?\s*$", lines[index], re.I)
        if match:
            body, index = [], index + 1
            while index < len(lines) and not re.match(r"^\s*\.endm\b", lines[index], re.I):
                body.append(lines[index])
                index += 1
            macros[match[1].lower()] = _split_operands(match[2] or ""), tuple(body)
            index += 1
            continue
        call = re.match(r"^(\s*)([A-Za-z_.$][\w.$]*)(?:\s+(.*?))?\s*$", lines[index])
        macro = macros.get(call[2].lower()) if call else None
        if macro:
            for body_line in macro[1]:
                expanded = body_line
                for name, value in zip(macro[0], _split_operands(call[3] or "")):
                    expanded = re.sub(rf"\\{re.escape(name)}(?![\w.$])", value, expanded)
                result.extend((f"#line {line_numbers[index]}", expanded) if line_directives else (expanded,))
        else:
            result.extend((f"#line {line_numbers[index]}", lines[index]) if line_directives else (lines[index],))
        index += 1
    return "\n".join(result) + ("\n" if source.endswith(("\n", "\r")) else "")


def _instruction_size(
    mnemonic: str, operands: tuple[str, ...], compressed: bool = False,
    xlen: int = 64, isa: str | None = None, pc: int = 0,
) -> int:
    target_value = _int_value(operands[-1]) if operands else None
    target_delta = target_value - pc if target_value is not None else None
    if mnemonic.startswith(("c.", "cm.")):
        return 2
    if mnemonic in {"sb", "sh", "sw", "sd", "fsw", "fsd"} and len(operands) == 3:
        return 8
    if compressed:
        if mnemonic in {"nop", "ret", "ebreak", "unimp"}:
            return 2
        if mnemonic == "j" and operands:
            immediate = target_delta
            return 2 if immediate is None or (
                immediate % 2 == 0 and -2048 <= immediate <= 2046
            ) else 4
        if mnemonic == "jal" and operands:
            rd = 1 if len(operands) == 1 else None
            immediate = target_delta
            if rd == 1 and xlen == 32:
                if immediate is None or (
                    immediate % 2 == 0 and -2048 <= immediate <= 2046
                ):
                    return 2
        if mnemonic == "li" and len(operands) == 2:
            immediate = _int_value(operands[1])
            if _register_index(operands[0]) not in (None, 0) \
                    and immediate is not None and -32 <= immediate <= 31:
                return 2
        registers = tuple(_register_index(item) for item in operands)
        if mnemonic == "addi" and len(registers) == 3:
            immediate = _int_value(operands[2])
            if registers[:2] == (0, 0) and immediate == 0:
                return 2
            if registers[0] not in (None, 0) and immediate is not None \
                    and -32 <= immediate <= 31 \
                    and registers[1] == 0:
                return 2
            if registers[0] not in (None, 0) and registers[1] not in (None, 0) \
                    and immediate == 0:
                return 2
        if _isa_has_extension(isa, "zcb"):
            if mnemonic in {"lbu", "lh", "lhu", "sb", "sh"} and len(operands) == 2:
                memory = _memory_parts(operands[1])
                if memory:
                    offset, base = memory
                    data = _register_index(operands[0])
                    scale = 1 if mnemonic in {"lbu", "sb"} else 2
                    if (data in range(8, 16) and base in range(8, 16)
                            and offset is not None and 0 <= offset < 4 and offset % scale == 0):
                        return 2
            if (mnemonic in {"not", "zext.b"}
                    or mnemonic in {"sext.b", "sext.h", "zext.h"}
                    and _isa_has_extension(isa, "zbb")) \
                    and len(registers) == 2 and registers[0] == registers[1] in range(8, 16):
                return 2
            if (mnemonic == "zext.w" and _isa_has_extension(isa, "zba")
                    and xlen == 64 and len(registers) == 2
                    and registers[0] == registers[1] in range(8, 16)):
                return 2
        if mnemonic == "mul" and len(registers) == 3 \
                    and registers[0] in range(8, 16) \
                    and registers[1] in range(8, 16) \
                    and registers[2] in range(8, 16) \
                    and registers[0] == registers[1] \
                    and _isa_has_extension(isa, "zcb") \
                    and (_isa_has_extension(isa, "m") or _isa_has_extension(isa, "zmmul")):
                return 2
        if mnemonic in {"add", "addw", "and", "or", "xor"} \
                and len(registers) == 3 \
                and registers[0] in {registers[1], registers[2]} \
                and (
                    mnemonic == "add" and all(item not in (None, 0) for item in registers)
                    or all(item in range(8, 16) for item in registers)
                    and (mnemonic != "addw" or xlen == 64)
                ):
            return 2
        if mnemonic == "add" and len(registers) == 3 \
                and registers[0] not in (None, 0) and registers[1] == 0 \
                and registers[2] not in (None, 0):
            return 2
        if mnemonic == "zext.w" and xlen == 64 and len(registers) == 2 \
                and not _isa_has_extension(isa, "zba"):
            first = _instruction_size(
                "slli", (operands[0], operands[1], "32"), True, xlen, isa,
            )
            second = _instruction_size(
                "srli", (operands[0], operands[0], "32"), True, xlen, isa,
            )
            return first + second
        if mnemonic == "lui" and len(registers) == 2:
            immediate = _int_value(operands[1])
            if immediate is not None:
                immediate = immediate & ((1 << 20) - 1)
                immediate -= 1 << 20 if immediate & (1 << 19) else 0
                if registers[0] not in (None, 0, 2) and immediate and -32 <= immediate <= 31:
                    return 2
        if mnemonic == "addiw" and xlen == 64 and len(registers) == 3 \
                and registers[0] == registers[1] != 0:
            immediate = _int_value(operands[2])
            if immediate is not None and -32 <= immediate <= 31:
                return 2
        if mnemonic in {"srli", "srai"} and len(registers) == 3 \
                and registers[0] == registers[1] in range(8, 16):
            shift = _int_value(operands[2])
            if shift is not None and 0 < shift < xlen:
                return 2
        if mnemonic == "andi" and len(registers) == 3 \
                and registers[0] == registers[1] in range(8, 16):
            immediate = _int_value(operands[2])
            if immediate is not None and -32 <= immediate <= 31:
                return 2
        if mnemonic in {"lw", "sw", "ld", "sd", "flw", "fsw", "fld", "fsd"} and len(operands) == 2:
            memory = _memory_parts(operands[1])
            if memory:
                offset, base = memory
                floating = mnemonic in {"flw", "fsw", "fld", "fsd"}
                data = _fp_index(operands[0]) if floating else _register_index(operands[0])
                double = mnemonic in {"ld", "sd", "fld", "fsd"}
                single = mnemonic in {"lw", "sw", "flw", "fsw"}
                if base == 2:
                    limit, alignment = (512, 8) if double else (256, 4)
                    stack_data = data is not None and (floating or not mnemonic.startswith("l") or data != 0)
                    if (stack_data and offset is not None
                            and (floating or not double or xlen == 64)
                            and (not floating or not single or xlen == 32)
                            and 0 <= offset < limit and offset % alignment == 0):
                        return 2
                elif (data in range(8, 16) and base in range(8, 16)
                      and offset is not None
                      and (floating or not double or xlen == 64)
                      and (not floating or not single or xlen == 32)
                      and 0 <= offset < (256 if double else 128)
                      and offset % (8 if double else 4) == 0):
                    return 2
        if mnemonic in {"beqz", "bnez"} and len(registers) == 2 \
                and registers[0] in range(8, 16) \
                or mnemonic in {"beq", "bne"} and len(registers) == 3 \
                and registers[0] in range(8, 16) and registers[1] == 0:
            immediate = target_delta
            if immediate is None or immediate % 2 == 0 and -256 <= immediate <= 254:
                return 2
            if immediate % 2 == 0 and -4096 <= immediate <= 4094:
                return 4
            return 8
        if mnemonic == "addi" and len(registers) == 3 \
                and registers[0] in range(8, 16) and registers[1] == 2:
            immediate = _int_value(operands[2])
            if immediate is not None and 4 <= immediate <= 1020 and immediate % 4 == 0:
                return 2
        if mnemonic == "jr" and len(registers) == 1 and registers[0] not in (None, 0):
            return 2
        if mnemonic == "jalr":
            if len(registers) == 1 and registers[0] not in (None, 0):
                return 2
        if mnemonic == "mv" and len(registers) == 2 \
                and all(register not in (None, 0) for register in registers):
            return 2
        if mnemonic == "sext.w" and xlen == 64 and len(registers) == 2 \
                and registers[0] == registers[1] != 0:
            return 2
        if mnemonic in {"sub", "subw"} and len(registers) == 3 \
                and (mnemonic == "sub" or xlen == 64) \
                and registers[0] == registers[1] \
                and all(item is not None and 8 <= item <= 15 for item in registers):
            return 2
        if mnemonic == "addi" and len(registers) == 3 and registers[0] == registers[1] and registers[0] != 0:
            immediate = _int_value(operands[2])
            if immediate is not None and (
                    -32 <= immediate <= 31
                    or registers[0] == 2 and -512 <= immediate <= 496 and immediate % 16 == 0):
                return 2
        if mnemonic == "slli" and len(registers) == 3 and registers[0] == registers[1] and registers[0] != 0:
            shift = _int_value(operands[2])
            if shift is not None and 0 < shift < (64 if xlen == 64 else 32):
                return 2
    if target_value is not None and mnemonic in {
        "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
        "bltz", "bgez", "bltzal", "bgezal", "bgt", "ble", "bgtz", "blez",
        "bgtu", "bleu", "beqi", "bnei",
    }:
        return 4 if target_delta % 2 == 0 and -4096 <= target_delta <= 4094 else 8
    if mnemonic == "zext.b" and len(operands) == 2:
        return 4
    if mnemonic in {"sext.b", "sext.h", "zext.h"} and len(operands) == 2 \
            and not _isa_has_extension(isa, "zbb"):
        shift = {"sext.b": xlen - 8, "sext.h": xlen - 16, "zext.h": xlen - 16}[mnemonic]
        narrow = "srai" if mnemonic.startswith("sext") else "srli"
        return _instruction_size(
            "slli", (operands[0], operands[1], str(shift)), compressed, xlen, isa,
        ) + _instruction_size(
            narrow, (operands[0], operands[0], str(shift)), compressed, xlen, isa,
        )
    if mnemonic in {"la", "lla", "lga", "call", "tail"}:
        return 8
    if mnemonic == "li":
        value = _int_value(operands[-1]) if operands else None
        if value is None:
            if len(operands) == 2 and operands[1].strip().startswith("%") \
                    and _relocation_valid(mnemonic, operands):
                return 4
            return 8
        return sum(_li_instruction_sizes(value, xlen, compressed, _register_index(operands[0])))
    if mnemonic == "zext.w" and xlen == 64 and len(operands) == 2 \
            and not _isa_has_extension(isa, "zba"):
        return _instruction_size(
            "slli", (operands[0], operands[1], "32"), compressed, xlen, isa,
        ) + _instruction_size(
            "srli", (operands[0], operands[0], "32"), compressed, xlen, isa,
        )
    return 4


def _string_size(body: str) -> int | None:
    match = re.match(r"^(\.ascii|\.asciz|\.string)(?:\s+(.*))?$", body, re.IGNORECASE)
    if match is None:
        return None
    if not match[2] or not match[2].strip():
        return 0
    literals = []
    for operand in _split_operands(match[2]):
        if re.fullmatch(
            r'\s*"(?:\\.|[^"\\])*"(?:\s*"(?:\\.|[^"\\])*")*\s*', operand
        ) is None:
            return None
        literals.extend(re.findall(r'"(?:\\.|[^"\\])*"', operand))
    if not literals:
        return None
    try:
        size = 0
        for literal in literals:
            if not isinstance(ast.literal_eval(literal), str):
                return None
            body, index = literal[1:-1], 0
            while index < len(body):
                if body[index] != "\\":
                    size += len(body[index].encode("utf-8"))
                    index += 1
                    continue
                index += 1
                if index == len(body):
                    return None
                if body[index] in "xX":
                    index += 1
                    start = index
                    while index < len(body) and body[index] in "0123456789abcdefABCDEF":
                        index += 1
                    if index == start:
                        return None
                elif body[index] in "01234567":
                    index += 1
                    for _ in range(2):
                        if index >= len(body) or body[index] not in "01234567":
                            break
                        index += 1
                else:
                    index += 1
                size += 1
    except (SyntaxError, ValueError):
        return None
    return size + len(literals) if match[1].lower() != ".ascii" else size


def _data_directive_size(
    width: int, operands: Sequence[str], aliases: Mapping[str, str] | None = None,
) -> int | None:
    if not operands:
        return 0
    size = 0
    for operand in operands:
        value = operand.strip()
        base, resolved = _resolve_target(value, aliases)
        if base is None:
            value = str(resolved)
        if len(value) >= 2 and value[0] == value[-1] == "'":
            try:
                literal = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                literal = None
            if not isinstance(literal, str) or len(literal) != 1:
                return None
            size += width
            continue
        if _int_value(value) is None:
            if (width >= 4 or len(operands) > 1) and len(value) >= 2 \
                    and value[0] == value[-1] == '"':
                try:
                    literal = ast.literal_eval(value)
                    if not isinstance(literal, str):
                        return None
                except (SyntaxError, ValueError):
                    return None
                size += len(literal.encode()) if width == 1 else width
                continue
            elif width < 4 or not _target_expression_valid(value):
                return None
        size += 1 if width == 1 else width
    return size


def _float_literal_valid(value: str) -> bool:
    value = value.strip()
    if _int_value(value) is not None:
        return True
    if re.fullmatch(r"[+-]?(?:inf|infinity|nan)", value, re.IGNORECASE):
        return True
    return re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value
    ) is not None


_INSN_FORMATS = frozenset({
    "r", "r4", "i", "s", "sb", "b", "u", "uj", "j", "cr", "ci", "css", "ciw", "cl", "cs", "ca", "cb", "cj",
})
_COMPRESSED_INSN_FORMATS = frozenset({"cr", "ci", "css", "ciw", "cl", "cs", "ca", "cb", "cj", "c"})
_VALID_INSN_SIZES = frozenset({2, 4, 6, 8, 10, 12, 14, 16})


def _insn_operand_syntax_valid(value: str) -> bool:
    return re.match(
        r"^(?:r|r4|i|s|sb|b|u|uj|j|cr|ci|css|ciw|cl|cs|ca|cb|cj)\s*,",
        value.strip(), re.IGNORECASE,
    ) is None


def _insn_size(
    operands: tuple[str, ...], aliases: Mapping[str, object] | None = None,
) -> int | None:
    if not operands:
        return None
    first_parts = operands[0].strip().split(None, 1)
    first_token = first_parts[0]
    first = {"sb": "b", "uj": "j"}.get(first_token.lower(), first_token.lower())
    fields = ((*first_parts[1:], *operands[1:]))
    def valid_int(value, limit):
        parsed = _layout_int(value, aliases)
        return parsed is not None and 0 <= parsed < limit

    def valid_imm(value, lower, upper, align=1):
        parsed = _layout_int(value, aliases)
        return parsed is not None and lower <= parsed <= upper and parsed % align == 0

    def valid_gpr(value):
        return _register_index(value) in range(32)
    def valid_reg(value):
        value = value.strip()
        return (
            _register_index(value) in range(32)
            or _fp_index(value) in range(32)
            or re.fullmatch(r"v(?:[0-9]|[12][0-9]|3[01])", value) is not None
        )
    def valid_compact_reg(value):
        value = value.strip()
        return (
            _register_index(value) in range(8, 16)
            or _fp_index(value) in range(8, 16)
            or re.fullmatch(r"v(?:[89]|1[0-5])", value) is not None
        )
    def valid_compact_gpr(value):
        return _register_index(value) in range(8, 16)
    def valid_symbol(value):
        return _layout_int(value, aliases) is None and _target_expression_valid(value)
    def valid_low_relocation(value):
        return re.fullmatch(r"%(?:lo|pcrel_lo|tprel_lo)\([^()]+\)", value.strip()) is not None
    def valid_high_relocation(value):
        return re.fullmatch(r"%(?:hi|pcrel_hi|tprel_hi)\([^()]+\)", value.strip()) is not None
    if first in _INSN_FORMATS:
        if first == "s" and not (
            len(fields) == 4 and valid_int(fields[0], 128) and valid_int(fields[1], 8)
            and valid_reg(fields[2]) and (
                (memory := _memory_parts(fields[3])) is not None
                and memory[0] in range(-2048, 2048)
                and memory[1] is not None
                or (relocation := re.fullmatch(
                    r"%(?:lo|pcrel_lo|tprel_lo)\([^()]+\)\s*\(\s*([^()]+)\s*\)",
                    fields[3],
                )) is not None and valid_gpr(relocation[1])
            )
        ):
            return None
        if first == "r" and not (
            len(fields) == 6 and valid_int(fields[0], 128) and valid_int(fields[1], 8)
            and valid_int(fields[2], 128) and all(valid_reg(value) for value in fields[3:])
        ):
            return None
        if first == "r4" and not (
            len(fields) == 7 and valid_int(fields[0], 128) and valid_int(fields[1], 4)
            and valid_int(fields[2], 8) and all(valid_reg(value) for value in fields[3:])
        ):
            return None
        if first == "i" and not (
            len(fields) == 5 and valid_int(fields[0], 128) and valid_int(fields[1], 8)
            and valid_reg(fields[2]) and valid_reg(fields[3])
            and (valid_imm(fields[4], -2048, 2047) or valid_low_relocation(fields[4]))
        ):
            return None
        if first == "b" and not (
            len(fields) == 5 and valid_int(fields[0], 128) and valid_int(fields[1], 8)
            and valid_gpr(fields[2]) and valid_gpr(fields[3])
            and (
                (target := _layout_int(fields[4], aliases)) is not None
                and -4096 <= target <= 4094 and target % 2 == 0
                or valid_symbol(fields[4])
            )
        ):
            return None
        if first == "u" and not (
            len(fields) == 3 and valid_int(fields[0], 128) and valid_gpr(fields[1])
            and (valid_imm(fields[2], 0, (1 << 20) - 1) or valid_high_relocation(fields[2]))
        ):
            return None
        if first == "j" and not (
            len(fields) == 3 and valid_int(fields[0], 128) and valid_gpr(fields[1])
            and (valid_imm(fields[2], -(1 << 20), (1 << 20) - 2, 2) or valid_symbol(fields[2]))
        ):
            return None
        if first == "ci" and not (
            len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
            and valid_gpr(fields[2]) and valid_imm(fields[3], -32, 31)
        ):
            return None
        if first == "cr" and not (
            len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 16)
            and valid_gpr(fields[2]) and valid_gpr(fields[3])
        ):
            return None
        if first == "css" and not (
            len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
            and valid_compact_reg(fields[2]) and valid_imm(fields[3], 0, 63)
        ):
            return None
        if first in {"cl", "cs"}:
            memory = _memory_parts(fields[3]) if len(fields) == 4 else None
            if not (
                len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
                and valid_compact_reg(fields[2]) and memory is not None
                and memory[0] in range(32) and valid_compact_gpr(f"x{memory[1]}")
            ):
                return None
        if first == "ca" and not (
            len(fields) == 5 and valid_int(fields[0], 3) and valid_int(fields[1], 64)
            and valid_int(fields[2], 4) and valid_compact_reg(fields[3])
            and valid_compact_reg(fields[4])
            ):
                return None
        if first == "cj":
            target = _layout_int(fields[2], aliases) if len(fields) == 3 else None
            if not (
                len(fields) == 3 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
                and (
                    valid_imm(fields[2], -2048, 2046, 2)
                    or target is None and valid_symbol(fields[2])
                )
            ):
                return None
        if first == "ciw" and not (
            len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
            and valid_compact_gpr(fields[2]) and valid_imm(fields[3], 0, 255)
        ):
            return None
        if first == "cb" and not (
            len(fields) == 4 and valid_int(fields[0], 3) and valid_int(fields[1], 8)
            and valid_compact_gpr(fields[2])
            and (valid_imm(fields[3], -256, 254, 2) or valid_symbol(fields[3]))
        ):
            return None
        return None if len(operands) < 2 else 2 if first in _COMPRESSED_INSN_FORMATS else 4
    if len(operands) > 1:
        length = _layout_int(first_token, aliases)
        value = _layout_int(operands[1], aliases) if len(operands) == 2 else None
        return length if length in _VALID_INSN_SIZES and value is not None and 0 <= value < (1 << (8 * length)) else None
    value = _layout_int(first, aliases)
    size = None if value is None else 2 if value & 3 != 3 else 4
    return size if size is not None and 0 <= value < (1 << (8 * size)) else None


_BRANCH_PSEUDO_FORMS = {
    "beqz": ("beq", ("rs1", "imm")),
    "bnez": ("bne", ("rs1", "imm")),
    "bltz": ("blt", ("rs1", "imm")),
    "bgez": ("bge", ("rs1", "imm")),
    "bgtz": ("blt", ("rs1", "imm")),
    "blez": ("bge", ("rs1", "imm")),
    "bgt": ("blt", ("rs1", "rs2", "imm")),
    "ble": ("bge", ("rs1", "rs2", "imm")),
    "bgtu": ("bltu", ("rs1", "rs2", "imm")),
    "bleu": ("bgeu", ("rs1", "rs2", "imm")),
}


def _rule_facts(mnemonic: str, xlen: int | None = None) -> dict[str, object]:
    normalized = mnemonic.lower().replace("_", ".")
    if normalized in _BRANCH_PSEUDO_FORMS:
        base, roles = _BRANCH_PSEUDO_FORMS[normalized]
        facts = _rule_facts(base, xlen)
        facts.update({
            "operand_roles": list(roles),
            "source_projections": [[role, "xlen"] for role in roles if role != "imm"],
            "permutation_pairs": [], "equivalent_targets": [],
            "literal_identity_siblings": [], "effect_view_siblings": [],
        })
        return facts
    if normalized == "zext.b":
        facts = _rule_facts("zext.h", xlen)
        facts["source_projections"] = [["rs1", "low8"]]
        return facts
    spec = runtime_instruction_spec(mnemonic, xlen)
    if spec is None:
        return {"definedness": "unresolved"}
    projections = (
        spec.source_projections_for_xlen(xlen)
        if xlen in {32, 64} else spec.source_projections
    )
    facts = {
        "definedness": "catalog-known",
        "effect": spec.effect_kind,
        "control_shape": spec.control_shape,
        "operand_roles": list(spec.operand_roles),
        "signedness": spec.signedness,
        "result_width": (
            gpr_result_width_bits(mnemonic, xlen)
            if spec.effect_kind in {"gpr", "private-load", "gpr-from-fpr"}
            else fp_operand_width_bits(mnemonic)
            if spec.effect_kind == "fpr" else None
        ),
        "zero_immediate_targets": list(spec.zero_immediate_targets),
        "source_projections": [list(item) for item in projections],
        "permutation_pairs": [list(item) for item in spec.operand_permutation_pairs],
        "equivalent_targets": list(spec.equivalent_targets),
        "literal_identity_siblings": [
            {"mnemonic": item.mnemonic, "allowed_literals": list(item.allowed_literals)}
            for item in spec.literal_identity_siblings
        ],
        "effect_view_siblings": [
            {"mnemonic": item.mnemonic, "kind": item.kind,
             "swap_source_operands": item.swap_source_operands}
            for item in spec.effect_view_siblings
        ],
        "has_rm": "rm" in spec.shared_spec.operand_kinds,
        "fp_form_family_siblings": [
            str(sibling).replace("_", ".") for sibling in fp_form_family_siblings(mnemonic)
            if fp_family_equivalence_condition(mnemonic, sibling) is not None
        ],
    }
    return facts


def _target_enabled(row: Mapping[str, object], mnemonic: str) -> bool:
    isa = _row_isa(row)
    normalized = mnemonic.lower().replace("_", ".")
    normalized = _BRANCH_PSEUDO_FORMS.get(normalized, (normalized, ()))[0]
    if normalized in _BASE_EXPANDING_PSEUDOS:
        return True
    if not isa:
        return True
    profile = isa
    xlen = 32 if profile.startswith("rv32") else 64 if profile.startswith("rv64") else None
    spec = runtime_instruction_spec(normalized, xlen)
    return spec is not None and profile_enables_form(profile, spec.shared_spec.form)


def _r9_branch_target(mnemonic: str) -> str | None:
    base = _BRANCH_PSEUDO_FORMS.get(mnemonic.lower().replace("_", "."), (mnemonic, ()))[0]
    return direct_jump_target_for_branch(base)


def _r9_control_nop(mnemonic: str) -> str | None:
    base = _BRANCH_PSEUDO_FORMS.get(mnemonic.lower().replace("_", "."), (mnemonic, ()))[0]
    return control_no_effect_target_for_generation(base)


def _row_isa(row: Mapping[str, object]) -> str | None:
    value = row.get("isa") or row.get("isa_profile")
    return str(value).lower() if isinstance(value, str) else None


def _r3_swap_text(row: Mapping[str, object]) -> str | None:
    facts = row.get("rule_facts")
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    pair = next((tuple(item) for item in facts.get("permutation_pairs", ())
                 if isinstance(item, (list, tuple)) and len(item) == 2
                 and set(item) == {"rs1", "rs2"}), None) if isinstance(facts, Mapping) else None
    if pair is None or len(roles) != len(operands):
        return None
    left, right = (roles.index(role) for role in pair)
    left_register, right_register = _register_index(operands[left]), _register_index(operands[right])
    if operands[left] == operands[right] or (
        left_register is not None and left_register == right_register
    ):
        return None
    rewritten = list(operands)
    rewritten[left], rewritten[right] = rewritten[right], rewritten[left]
    spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
    if spec is None or not _text_encodable(spec, roles, rewritten, row.get("_aliases"), isa_profile=_row_isa(row)):
        return None
    return f"{row.get('mnemonic')} {', '.join(rewritten)}"


def _profile_layout_ready(profile: object | None) -> bool:
    provenance = getattr(profile, "provenance", {})
    return not isinstance(provenance, Mapping) or provenance.get("layout") != "unresolved"


def _final_layout_valid(row: Mapping[str, object]) -> bool:
    values = tuple(row.get(name) for name in ("final_pcs", "final_sizes", "final_bytes"))
    if not all(isinstance(value, (list, tuple)) and value for value in values):
        return False
    if len({len(value) for value in values}) != 1:
        return False
    pcs, sizes, encoded = values
    parsed_pcs = tuple(_pc_value(value) for value in pcs)
    xlen, isa = row.get("xlen"), row.get("isa")
    if xlen not in {32, 64} or not isinstance(isa, str):
        return False
    ialign, pc_limit = (2 if _isa_has_extension(isa, "c") else 4), 1 << xlen
    return (
        all(value is not None for value in parsed_pcs)
        and parsed_pcs == tuple(sorted(parsed_pcs))
        and len(set(parsed_pcs)) == len(parsed_pcs)
        and all(value % ialign == 0 and value < pc_limit for value in parsed_pcs)
        and all(
            type(value) is int and value in _VALID_INSN_SIZES
            and (value != 2 or _isa_has_extension(isa, "c"))
            for value in sizes
        )
        and all(
            isinstance(value, str) and len(value) == size * 2
            and all(char in "0123456789abcdefABCDEF" for char in value)
            for value, size in zip(encoded, sizes)
        )
        and all(left + size <= right for left, size, right in zip(
            parsed_pcs, sizes, parsed_pcs[1:]
        ))
        and all(pc + size <= pc_limit for pc, size in zip(parsed_pcs, sizes))
    )


def _final_pcs_valid(row: Mapping[str, object]) -> bool:
    values = row.get("final_pcs")
    if not isinstance(values, (list, tuple)) or not values:
        return False
    parsed = tuple(_pc_value(value) for value in values)
    xlen, isa = row.get("xlen"), row.get("isa")
    pc, pc_end = _pc_value(row.get("pc")), _pc_value(row.get("pc_end"))
    if xlen not in {32, 64} or not isinstance(isa, str) or pc is None or pc_end is None:
        return False
    ialign, pc_limit = (2 if _isa_has_extension(isa, "c") else 4), 1 << xlen
    return all(value is not None for value in parsed) and parsed == tuple(sorted(parsed)) \
        and len(set(parsed)) == len(parsed) \
        and pc < pc_end and parsed[0] == pc \
        and all(pc <= value < pc_end and value % ialign == 0 and value < pc_limit for value in parsed)


def _profile_layout_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_digest([
        {
            "id": row.get("id"),
            "pc": row.get("pc"), "pc_end": row.get("pc_end"),
            "final_pcs": list(row.get("final_pcs") or ()),
            "final_sizes": list(row.get("final_sizes") or ()),
            "final_bytes": list(row.get("final_bytes") or ()),
        }
        for row in rows if isinstance(row, Mapping)
    ])


def _candidate_size_matches(row: Mapping[str, object], text: object) -> bool:
    if not isinstance(text, str):
        return False
    parts = text.strip().split(None, 1)
    if not parts or type(row.get("xlen")) is not int:
        return False
    operands = tuple(
        _resolve_size_operand(value, row.get("_aliases"))
        for value in (_split_operands(parts[1]) if len(parts) > 1 else ())
    )
    try:
        size = _instruction_size(
            parts[0].lower().replace("_", "."), operands,
            row.get("compressed") is True, row["xlen"], str(row.get("isa", "")),
            pc=int(row["pc"]),
        )
        return size == int(row["pc_end"]) - int(row["pc"])
    except (KeyError, TypeError, ValueError):
        return False


def _r3_anchor_ready(row: Mapping[str, object], profile: object | None = None) -> bool:
    if not _profile_layout_ready(profile):
        return False
    anchors = tuple(row.get(name) for name in ("final_pcs", "final_sizes", "final_bytes"))
    if any(value is not None for value in anchors[1:]) and not _final_layout_valid(row):
        return False
    return _final_layout_valid(row) if row.get("compressed") is True else (
        not any(value is not None for value in anchors)
        or _final_pcs_valid(row)
    )


def _row_final_pcs(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.get("final_pcs") or (row.get("pc"),))


def _rule_form_possible(row: Mapping[str, object], rule: str) -> bool:
    facts = row.get("rule_facts", {})
    if not isinstance(facts, Mapping):
        return False
    effect = facts.get("effect")
    operands = tuple(row.get("operands", ()))
    if rule == "R1":
        return effect == "gpr" and bool(
            set(facts.get("operand_roles", ())) & {"rs1", "rs2", "rs3"}
        )
    if rule == "R4":
        return effect == "private-load" and {"rs1", "imm"} <= set(
            facts.get("operand_roles", ())
        )
    if rule == "R6":
        roles = tuple(facts.get("operand_roles", ()))
        if "rd" not in roles or roles.index("rd") >= len(operands):
            return False
        rd = _register_index(operands[roles.index("rd")])
        return effect == "private-load" and any(
            isinstance(item, Mapping)
            and item.get("kind") == "load-extension-view"
            and _target_enabled(row, str(item.get("mnemonic", "")))
            for item in facts.get("effect_view_siblings", ())
        ) and rd not in (None, 0)
    if rule == "R7":
        return effect == "fpr" and facts.get("has_rm") is True and fp_operand_width_bits(
            str(row.get("mnemonic", ""))
        ) is not None
    if rule == "R8":
        return effect == "fpr" and fp_operand_width_bits(
            str(row.get("mnemonic", ""))
        ) is not None and any(
            _target_enabled(row, str(sibling))
            for sibling in facts.get("fp_form_family_siblings", ())
        )
    if rule == "R9":
        if facts.get("control_shape") != "conditional":
            return False
        return any(
            _target_enabled(row, target)
            for target in (
                _r9_branch_target(str(row.get("mnemonic", ""))),
                _r9_control_nop(str(row.get("mnemonic", ""))),
            )
            if target is not None
        )
    if rule == "R5":
        targets = tuple(facts.get("equivalent_targets", ())) + tuple(
            item.get("mnemonic") for item in facts.get("literal_identity_siblings", ())
            if isinstance(item, Mapping)
        )
        return effect == "gpr" and len(operands) == 3 and any(
            _target_enabled(row, str(target)) for target in targets
        )
    if rule == "R10":
        if effect != "gpr" or not operands:
            return False
        roles = tuple(facts.get("operand_roles", ()))
        if len(roles) != len(operands) or "rd" not in roles:
            return False
        destination = _register_index(operands[roles.index("rd")])
        if destination in (None, 0):
            return False
        spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
        rewritten = list(operands)
        rewritten[roles.index("rd")] = "x0"
        return spec is not None and _text_encodable(spec, roles, rewritten, row.get("_aliases"), isa_profile=_row_isa(row))
    if rule == "R11":
        zero_targets = facts.get("zero_immediate_targets", ())
        roles = tuple(facts.get("operand_roles", ()))
        zero = len(roles) == len(operands) and "imm" in roles and _layout_int(
            operands[roles.index("imm")], row.get("_aliases")
        ) == 0
        return effect == "gpr" and (
            zero and any(_target_enabled(row, str(target)) for target in zero_targets)
            or any(
                isinstance(item, Mapping)
                and _target_enabled(row, str(item.get("mnemonic", "")))
                for item in facts.get("effect_view_siblings", ())
            )
        )
    return False


def _source_text(mnemonic: str, operands: Sequence[str]) -> str:
    return mnemonic + (f" {', '.join(operands)}" if operands else "")


def _probe_register_text(row: Mapping[str, object], role: str) -> str | None:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = list(row.get("operands", ()))
    aliases = row.get("_aliases")
    spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))

    def render(values: Sequence[str]) -> str | None:
        return _source_text(str(row["mnemonic"]), values) \
            if spec is None or _text_encodable(spec, roles, values, aliases, isa_profile=_row_isa(row)) else None

    register_count = 16 if str(row.get("isa", "")).lower().startswith("rv32e") else 32
    index = roles.index(role) if role in roles and len(roles) == len(operands) else None
    for position, operand in enumerate(operands):
        memory = _memory_parts(_resolve_memory_alias(operand, aliases))
        if role == "rs1" and memory:
            offset, source = memory
            if source is not None:
                for candidate in range(register_count):
                    if candidate == source:
                        continue
                    candidate_operands = operands.copy()
                    candidate_operands[position] = f"{offset}(x{candidate})"
                    if rendered := render(candidate_operands):
                        return rendered
        if index is None or position != index:
            continue
        source = _register_index(operand)
        if source is None:
            return None
        for candidate in range(register_count):
            if candidate == source:
                continue
            candidate_operands = operands.copy()
            candidate_operands[index] = f"x{candidate}"
            if rendered := render(candidate_operands):
                return rendered
    return None


def _address_alias_texts(row: Mapping[str, object]) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    aliases = row.get("_aliases")
    operands = tuple(_resolve_memory_alias(str(item), aliases) for item in row.get("operands", ()))
    if (not isinstance(facts, Mapping)
            or tuple(facts.get("operand_roles", ())) != ("rd", "rs1", "imm")
            or len(operands) != 2):
        return ()
    memory = _memory_parts(operands[1])
    source = memory[1] if memory else None
    states = _r2_before_states(row)
    if source is None or not states or any(state is None for state in states):
        return ()
    spec = runtime_instruction_spec(str(row["mnemonic"]), row.get("xlen"))
    if spec is None:
        return ()
    register_count = 16 if str(row.get("isa", "")).lower().startswith("rv32e") else 32
    mask = (1 << int(row.get("xlen") or 64)) - 1
    result = []
    for candidate in range(register_count):
        if candidate == source or not all(
            state[candidate] & mask == state[source] & mask for state in states
        ):
            continue
        rewritten = list(operands)
        rewritten[1] = f"{memory[0]}(x{candidate})"
        if _text_encodable(spec, facts["operand_roles"], rewritten, aliases, isa_profile=_row_isa(row)):
            result.append(_source_text(str(row["mnemonic"]), rewritten))
    return tuple(dict.fromkeys(result))


_FLI_LITERAL_SELECTORS = {
    -1.0: 0, 2.0 ** -16: 2, 2.0 ** -15: 3, 2.0 ** -8: 4,
    2.0 ** -7: 5, 2.0 ** -4: 6, 2.0 ** -3: 7, 2.0 ** -2: 8,
    0.3125: 9, 0.375: 10, 0.4375: 11, 0.5: 12, 0.625: 13, 0.75: 14,
    0.875: 15, 1.0: 16, 1.25: 17, 1.5: 18, 1.75: 19, 2.0: 20,
    2.5: 21, 3.0: 22, 4.0: 23, 8.0: 24, 16.0: 25, 128.0: 26,
    256.0: 27, 32768.0: 28, 65536.0: 29,
}


def _text_encodable(
    spec: object, roles: Sequence[str], operands: Sequence[str],
    aliases: Mapping[str, str] | None = None, *, isa_profile: str | None = None,
) -> bool:
    values = tuple(_resolve_memory_alias(value, aliases) for value in operands)
    if spec.mnemonic == "c.nop" and values:
        immediate = _layout_int(values[0], aliases) if len(values) == 1 else None
        if immediate is None or immediate == 0:
            return False
    if spec.mnemonic in {"beqi", "bnei"} and len(values) == 3:
        immediate = _layout_int(values[1], aliases)
        return (
            _register_index(values[0]) is not None
            and immediate is not None and -1 <= immediate <= 31
            and _register_index(values[2]) is None
            and _fp_index(values[2]) is None
            and re.fullmatch(r"v\d+(?:\.\w+)?", values[2]) is None
        )
    if spec.mnemonic == "c.zext.b" and tuple(roles) == ("rd", "rs1"):
        return len(values) == 2 and all(_register_index(value) is not None for value in values)
    if getattr(spec, "effect_kind", None) == "private-load" and len(values) != 2:
        return False
    if not values and spec.mnemonic == "c.nop":
        values = ("0",)
    if spec.mnemonic in {"fcvtmod.w.d", "fcvtmod.w.s"} and len(values) == len(roles) + 1:
        if values[-1].lower() != "rtz":
            return False
        values = values[:-1]
    if len(values) == len(roles) + 1 and "rm" in getattr(
        getattr(spec, "shared_spec", None), "operand_kinds", ()
    ):
        mode = values[-1].lower()
        if mode not in {"rne", "rtz", "rdn", "rup", "rmm", "dyn"}:
            return False
        values = values[:len(roles)]
    if spec.mnemonic == "jal" and len(values) == 1:
        values = ("x1", values[0])
    elif spec.mnemonic == "jalr":
        if len(values) == 1 and _register_index(values[0]) is not None:
            values = (values[0], values[0], "0")
        elif len(values) == 2 and _register_index(values[1]) is not None:
            values = (values[0], values[1], "0")
    elif spec.mnemonic == "c.addi16sp" and len(values) == 2 \
            and _register_index(values[0]) == 2:
        values = (values[1],)
    elif spec.mnemonic == "c.addi4spn" and len(values) == 3 \
            and _register_index(values[1]) == 2:
        values = (values[0], values[2])
    if len(values) == 2 and tuple(roles) == ("rd", "rs1", "imm"):
        match = _MEMORY_OPERAND_RE.fullmatch(values[1])
        if match:
            if getattr(spec, "effect_kind", None) not in {"private-load", "control"}:
                return False
            values = (values[0], match[2], match[1])
        elif (base_match := _MEMORY_BASE_RE.fullmatch(values[1])):
            values = (values[0], base_match[1], "0")
        elif (
            getattr(spec, "effect_kind", None) == "private-load"
            and _target_expression_valid(values[1])
            and _integer_expression(values[1]) is None
        ):
            values = (values[0], "x0", "0")
    form = spec.shared_spec.form
    operand_kinds = tuple(getattr(spec.shared_spec, "operand_kinds", ()))
    if (len(operand_kinds) > len(roles) and len(values) == len(roles)
            and "rm" not in operand_kinds):
        return False
    if implicit_source_roles(form) and "rlist" not in operand_kinds \
            and spec.mnemonic not in {"cm.jt", "cm.jalt", "c.addi4spn", "c.addi16sp"}:
        if len(values) == len(roles) + 1:
            implicit_index = 0 if roles == ("imm",) else 1
            if _register_index(values[implicit_index]) != 2:
                return False
            values = values[:implicit_index] + values[implicit_index + 1:]
        elif len(values) == len(roles) and not any(
            _MEMORY_OPERAND_RE.fullmatch(value) or _MEMORY_BASE_RE.fullmatch(value)
            for value in values
        ):
            return False
        if len(values) == len(roles):
            for index, value in enumerate(values):
                match = _MEMORY_OPERAND_RE.fullmatch(value)
                if match and _register_index(match[2].strip()) == 2:
                    values = (*values[:index], match[1], *values[index + 1:])
                    break
                if (base_match := _MEMORY_BASE_RE.fullmatch(value)) \
                        and _register_index(base_match[1].strip()) == 2:
                    values = (*values[:index], "0", *values[index + 1:])
                    break
    if len(values) != len(roles):
        if len(values) != len(operand_kinds):
            return False
        if tuple(kind for kind in operand_kinds if kind in roles) != tuple(roles):
            return False
        for kind, value in zip(operand_kinds, values):
            if kind in roles:
                continue
            selector = _int_value(value)
            if kind == "bs" and selector in range(4):
                continue
            if kind == "rnum" and selector in range(11):
                continue
            return False
        values = tuple(value for kind, value in zip(operand_kinds, values) if kind in roles)
    if len(values) != len(roles):
        return False
    by_role = dict(zip(roles, values))
    def role_domain(role: str) -> str | None:
        domain = dict(getattr(spec.shared_spec, "operand_domains", ())).get(role)
        return xregister_compatibility_slot_domain_for_mnemonic(
            isa_profile, spec.mnemonic, role, 0
        ) if domain == "fpr" and isa_profile else domain

    def xregister_pair_aligned(role: str, value: object) -> bool:
        domain = dict(getattr(spec.shared_spec, "operand_domains", ())).get(role)
        if not (
            domain == "fpr" and role_domain(role) == "gpr"
            and isa_profile and isa_profile.startswith("rv32")
            and "zdinx" in enabled_extensions(isa_profile)
            and fp_operand_width_bits(spec.mnemonic) == 64
        ):
            return True
        index = _register_index(value)
        return index is not None and index % 2 == 0

    if "imm" in by_role and (
        _register_index(by_role["imm"]) is not None
        or _fp_index(by_role["imm"]) is not None
        or re.fullmatch(r"v\d+(?:\.\w+)?", str(by_role["imm"])) is not None
    ):
        return False
    for role, domain in getattr(spec.shared_spec, "operand_domains", ()):
        value = by_role.get(role)
        domain = role_domain(role)
        if domain == "gpr" and _register_index(value) is None \
                or domain == "fpr" and _fp_index(value) is None \
                or not xregister_pair_aligned(role, value):
            return False

    for group in getattr(spec.shared_spec.form, "variable_fields", ()):
        group = str(group)
        group_roles = tuple(role for role in ("rd", "rs1", "rs2", "rs3") if role in group)
        registers = tuple(by_role[role] for role in group_roles if role in by_role)
        if group.endswith("_n0") and any(_register_index(value) == 0 for value in registers) \
                and not (spec.mnemonic in {"c.add", "c.mv", "c.li", "c.slli", "c.addi"}
                         and _register_index(by_role.get("rd")) == 0):
            return False
        if group.endswith("_n2") and any(_register_index(value) == 2 for value in registers):
            return False
        if group.endswith("_p") and not group.startswith("p_") and any(
            (_fp_index(value) if role_domain(role) == "fpr" else _register_index(value)) not in range(8, 16)
            for role, value in zip(group_roles, registers)
        ):
            return False

    def register(value: object) -> int | None:
        return _register_index(value) if _register_index(value) is not None else _fp_index(value)

    if "imm" in by_role and spec.mnemonic in {
        "fli.s", "fli.d", "fli.h", "fli.q",
    }:
        raw = str(by_role["imm"]).strip()
        if not raw:
            return False
        normalized = raw.lstrip("+-").lower()
        if normalized == "inf":
            immediate = 30
        elif normalized == "nan":
            immediate = 31
        elif normalized == "min":
            immediate = 1
        else:
            try:
                number = float.fromhex(raw) if normalized.startswith("0x") else float(raw)
            except ValueError:
                return False
            immediate = _FLI_LITERAL_SELECTORS.get(number)
            if immediate is None:
                return False
        if immediate is None:
            return False
    else:
        immediate = _int_value(by_role.get("imm"))
    if spec.mnemonic in {"c.addi16sp", "c.addi4spn", "c.lui"} and immediate == 0:
        return False
    if spec.mnemonic == "c.lui" and immediate is not None:
        if 0xfffe0 <= immediate <= 0xfffff:
            immediate -= 1 << 20
        elif not 1 <= immediate <= 31:
            return False
    if immediate is None and "imm" in by_role and spec.control_shape is None:
        raw_immediate = str(by_role["imm"]).strip()
        if not raw_immediate.startswith("%"):
            base, resolved = _resolve_target(raw_immediate, aliases)
            if base is not None:
                return False
            immediate = resolved
    spec_xlen = 32 if str(isa_profile or "").lower().startswith("rv32") else (
        getattr(spec, "xlen", None) or getattr(spec.shared_spec, "xlen", None)
    )
    if immediate is not None and spec_xlen == 32:
        groups = set(operand_groups(form))
        if "shamtd" in groups \
                and not 0 <= immediate < 32:
            return False
    if immediate is not None and spec.mnemonic in {"slli", "srli", "srai", "slliw", "srliw", "sraiw"}:
        limit = 32 if spec.mnemonic.endswith("iw") or spec_xlen == 32 else 64
        if not 0 <= immediate < limit:
            return False
    groups = set(operand_groups(form))
    if immediate is not None and spec.mnemonic in {"auipc", "lui"} and "imm20" in groups \
            and not 0 <= immediate < (1 << 20):
        return False
    if immediate is not None and groups & {"imm20", "c_nzimm18"}:
        immediate <<= 12
    try:
        spec.encode(
            register(by_role.get("rd")), register(by_role.get("rs1")),
            register(by_role.get("rs2")), register(by_role.get("rs3")),
            immediate,
        )
    except (TypeError, ValueError):
        return False
    return True


def _fp_condition_holds_row(row: Mapping[str, object], target: str) -> bool:
    condition = fp_family_equivalence_condition(str(row.get("mnemonic", "")), target)
    width = fp_operand_width_bits(str(row.get("mnemonic", "")))
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    witnesses = _row_witnesses(row)
    if condition is None or width is None or not witnesses or len(roles) != len(operands):
        return False
    indices = {
        role: _fp_index(operands[index])
        for index, role in enumerate(roles)
        if role in {"rd", "rs1", "rs2", "rs3"}
    }
    if any(value is None for value in indices.values()):
        return False
    rd = indices.get("rd")
    if rd is None:
        return False

    try:
        extensions = enabled_extensions(str(row.get("isa", ""))) if row.get("isa") else ()
    except (TypeError, ValueError):
        return False
    storage_width = (
        128 if "q" in extensions else 64 if "d" in extensions
        else 32 if "f" in extensions else 16
    )
    if storage_width < width:
        return False

    def payload(value: object) -> int | None:
        raw = _int_value(value)
        if raw is None or not 0 <= raw < (1 << 64):
            return None
        if storage_width > width:
            upper = ((1 << (storage_width - width)) - 1) << width
            if raw & upper != upper:
                return None
        return raw & ((1 << width) - 1)

    def zero(value: object) -> bool:
        raw = payload(value)
        if raw is None:
            return False
        exponent_bits, fraction_bits = {16: (5, 10), 32: (8, 23), 64: (11, 52)}.get(width, (0, 0))
        return exponent_bits > 0 and raw & ((1 << (width - 1)) - 1) == 0

    def finite_nonzero(value: object) -> bool:
        raw = payload(value)
        if raw is None:
            return False
        exponent_bits, fraction_bits = {16: (5, 10), 32: (8, 23), 64: (11, 52)}.get(width, (0, 0))
        exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
        return exponent != (1 << exponent_bits) - 1 and raw & ((1 << (width - 1)) - 1) != 0

    for witness in witnesses:
        before = witness.get("before_fpr_rawbits")
        after = witness.get("after_fpr_rawbits")
        if (not isinstance(before, (list, tuple)) or not isinstance(after, (list, tuple))
                or len(before) != 32 or len(after) != 32
                or any(payload(value) is None for value in (*before, *after))
                or witness.get("before_fflags") != witness.get("after_fflags")
                or witness.get("before_frm") != witness.get("after_frm")
                or any(before[index] != after[index] for index in range(32) if index != rd)):
            return False
        if any(payload(before[indices[role]]) is None for role in ("rs1", "rs2", "rs3") if role in indices):
            return False
        if condition == FP_EQUAL_SOURCES:
            if payload(before[indices["rs1"]]) != payload(before[indices["rs2"]]):
                return False
        elif condition in {FP_POSITIVE_RS1, FP_NEGATIVE_RS1}:
            negative = bool(payload(before[indices["rs1"]]) & (1 << (width - 1)))
            if negative != (condition == FP_NEGATIVE_RS1):
                return False
        elif condition == FP_ZERO_ADDEND:
            if (
                not zero(before[indices["rs3"]])
                or not finite_nonzero(before[indices["rs1"]])
                or not finite_nonzero(before[indices["rs2"]])
                or zero(after[rd])
            ):
                return False
        elif condition == FP_ZERO_PRODUCT:
            left, right = before[indices["rs1"]], before[indices["rs2"]]
            product_zero = zero(left) and finite_nonzero(right) or zero(right) and finite_nonzero(left)
            signs = fma_sign_selectors(str(row.get("mnemonic", "")))
            expected = payload(before[indices["rs3"]])
            if (not product_zero or not finite_nonzero(before[indices["rs3"]])
                    or signs is None or signs[1] and expected is None
                    or payload(after[rd]) != expected ^ ((1 << (width - 1)) if signs[1] else 0)):
                return False
        else:
            return False
    return True


def _same_shape_texts(
    row: Mapping[str, object], targets: Sequence[str], *, fp_guard: bool = False,
) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    result = []
    for target in targets:
        spec = runtime_instruction_spec(str(target), row.get("xlen"))
        valid_shape = (
            len(operands) == len(roles)
            or facts.get("has_rm") is True and len(operands) == len(roles) + 1
            or facts.get("effect") == "private-load" and len(operands) == 2
        )
        if (spec is None or not _target_enabled(row, str(target))
                or tuple(spec.operand_roles) != roles or not valid_shape
                or not _text_encodable(spec, roles, operands, row.get("_aliases"), isa_profile=_row_isa(row))
                or fp_guard and not _fp_condition_holds_row(row, str(target))):
            continue
        result.append(_source_text(str(target).replace("_", "."), operands))
    return tuple(dict.fromkeys(result))


def _r5_texts(row: Mapping[str, object], *, guided: bool = False) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    if not isinstance(facts, Mapping):
        return ()
    roles = tuple(facts.get("operand_roles", ()))
    operands = tuple(row.get("operands", ()))
    aliases = row.get("_aliases")
    literal = _layout_int(operands[-1], aliases) if operands else None
    states = _r2_before_states(row)
    xlen = int(row.get("xlen") or 64)
    mask = (1 << xlen) - 1
    if literal is None and operands and (source := _register_index(operands[-1])) is not None:
        values_seen = (
            {0 if source == 0 else state[source] & mask for state in states}
            if states and not any(state is None for state in states)
            else ({0} if source == 0 else set())
        )
        literal = values_seen.pop() if len(values_seen) == 1 else None
    targets = list(facts.get("equivalent_targets", ()))
    targets.extend(
        item.get("mnemonic") for item in facts.get("literal_identity_siblings", ())
        if isinstance(item, Mapping)
        and literal is not None
        and literal in tuple(item.get("allowed_literals", ()))
    )
    result = []

    def projection_mask(projection: object) -> int | None:
        if projection == "xlen":
            return mask
        match = re.fullmatch(r"low([0-9]+)", str(projection))
        width = int(match[1]) if match else None
        return (1 << width) - 1 if width and width <= xlen else None

    for target in targets:
        target = str(target).replace("_", ".")
        spec = runtime_instruction_spec(target, row.get("xlen"))
        if (spec is None or spec.effect_kind != facts.get("effect")
                or not _target_enabled(row, target)):
            continue
        target_roles = tuple(spec.operand_roles)
        if target_roles == roles and len(operands) == len(roles):
            if _text_encodable(spec, target_roles, operands, aliases, isa_profile=_row_isa(row)):
                result.append(_source_text(target, operands))
            continue
        if not states or any(state is None for state in states):
            if guided:
                continue
            if roles == ("rd", "rs1", "imm") and target_roles != roles:
                target_only = next((role for role in target_roles if role not in roles), None)
                candidate_values = list(operands)
                if target_only is not None:
                    candidate_values[target_roles.index(target_only)] = "x0"
            elif roles != target_roles and "imm" in target_roles:
                candidate_values = list(operands)
                candidate_values[target_roles.index("imm")] = "0"
            else:
                continue
            if _text_encodable(spec, target_roles, candidate_values, aliases, isa_profile=_row_isa(row)):
                result.append(_source_text(target, candidate_values))
            continue
        if roles == ("rd", "rs1", "imm") and len(operands) == len(roles):
            literal = _layout_int(operands[-1], aliases)
            target_only = next((role for role in target_roles if role not in roles), None)
            target_mask = projection_mask(dict(
                spec.source_projections_for_xlen(xlen)
            ).get(target_only))
            if literal is not None and target_only is not None and target_mask is not None:
                register_count = 16 if str(row.get("isa", "")).lower().startswith("rv32e") else 32
                for candidate in range(register_count):
                    if all((state[candidate] & target_mask) == literal & target_mask for state in states):
                        candidate_values = list(operands)
                        candidate_values[target_roles.index(target_only)] = f"x{candidate}"
                        if _text_encodable(spec, target_roles, candidate_values, aliases, isa_profile=_row_isa(row)):
                            result.append(_source_text(target, candidate_values))
        elif roles and target_roles and len(operands) == len(roles):
            source_only = next((role for role in roles if role not in target_roles), None)
            if source_only is not None and "imm" in target_roles:
                source_index = roles.index(source_only)
                source_register = _register_index(operands[source_index])
                source_mask = projection_mask(dict(facts.get("source_projections", ())).get(source_only))
                values_seen = {
                    state[source_register] & source_mask for state in states
                } if source_register is not None and source_mask is not None else set()
                if len(values_seen) == 1:
                    value = values_seen.pop()
                    candidate_values = list(operands)
                    source_width = source_mask.bit_length()
                    values = [value]
                    if value & (1 << (source_width - 1)):
                        values.append(value - (1 << source_width))
                    for value in dict.fromkeys(values):
                        candidate_values[target_roles.index("imm")] = str(value)
                        if _text_encodable(spec, target_roles, candidate_values, aliases, isa_profile=_row_isa(row)):
                            result.append(_source_text(target, candidate_values))
                            break
    return tuple(dict.fromkeys(result))


def _r7_texts(row: Mapping[str, object]) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    operands = list(row.get("operands", ()))
    if not isinstance(facts, Mapping) or facts.get("has_rm") is not True:
        return ()
    index = len(operands) if len(operands) == len(facts.get("operand_roles", ())) else len(operands) - 1
    current = operands[index].lower() if index < len(operands) else "dyn"
    numeric_modes = ("rne", "rtz", "rdn", "rup", "rmm")
    if current.isdigit() and int(current) in range(5):
        current = numeric_modes[int(current)]
    if current not in (*numeric_modes, "dyn"):
        return ()
    modes = {
        witness.get("before_frm") for witness in _row_witnesses(row)
        if witness.get("before_frm") is not None
    }
    after_modes = {
        witness.get("after_frm") for witness in _row_witnesses(row)
        if witness.get("after_frm") is not None
    }
    if modes or after_modes:
        if len(modes) != 1 or modes != after_modes or next(iter(modes)) not in range(5):
            return ()
        frm = next(iter(modes))
        mode = numeric_modes[frm] if current == "dyn" else "dyn" if numeric_modes[frm] == current else None
    else:
        mode = next((value for value in (*numeric_modes, "dyn") if value != current), None)
    if mode is None:
        return ()
    if index == len(operands):
        operands.append(mode)
    else:
        operands[index] = mode
    spec = runtime_instruction_spec(str(row["mnemonic"]), row.get("xlen"))
    return (_source_text(str(row["mnemonic"]), operands),) if spec is not None and (
        _target_enabled(row, str(row["mnemonic"]))
        and _text_encodable(spec, spec.operand_roles, operands, row.get("_aliases"), isa_profile=_row_isa(row))
    ) else ()


def _r9_predicate_dual_texts(
    row: Mapping[str, object], *, probe: bool,
) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    if len(roles) != len(operands) or not {"rs1", "rs2"} <= set(roles):
        return ()
    left, right = roles.index("rs1"), roles.index("rs2")
    left_register, right_register = _register_index(operands[left]), _register_index(operands[right])
    if left_register is None or right_register is None:
        return ()
    states = _r2_before_states(row)
    xlen = row.get("xlen")
    if xlen not in {32, 64}:
        return ()
    mask = (1 << xlen) - 1
    result = []
    for item in facts.get("effect_view_siblings", ()):
        if (not isinstance(item, Mapping)
                or item.get("kind") != "branch-predicate-dual"):
            continue
        if not probe and (not states or any(state is None for state in states)):
            continue
        if not probe and any(
            (state[left_register] & mask) == (state[right_register] & mask)
            for state in states
        ):
            continue
        target = str(item.get("mnemonic", "")).replace("_", ".")
        spec = runtime_instruction_spec(target, xlen)
        if spec is None or not _target_enabled(row, target):
            continue
        values = list(operands)
        if item.get("swap_source_operands"):
            values[left], values[right] = values[right], values[left]
        if _text_encodable(spec, spec.operand_roles, values, row.get("_aliases"), isa_profile=_row_isa(row)):
            result.append(_source_text(target, values))
    return tuple(dict.fromkeys(result))


def _r9_texts(row: Mapping[str, object], *, probe: bool = False) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    operands = tuple(row.get("operands", ()))
    if not isinstance(facts, Mapping):
        return ()
    result = []
    if (swapped := _r3_swap_text(row)) is not None:
        result.append(swapped)
    outcomes = {
        value for value in (row.get("branch_outcome"),)
        if value in {"taken", "not-taken"}
    }
    outcomes.update(
        witness.get("branch_outcome") for witness in _row_witnesses(row)
        if witness.get("branch_outcome") in {"taken", "not-taken"}
    )
    outcome = True if outcomes == {"taken"} else False if outcomes == {"not-taken"} else None
    label = operands[-1] if operands else None
    pc, pc_end = _pc_value(row.get("pc")), _pc_value(row.get("pc_end"))
    source_width = pc_end - pc if pc is not None and pc_end is not None else None
    result.extend(_r9_predicate_dual_texts(row, probe=probe))
    target = _r9_branch_target(str(row.get("mnemonic", "")))
    compressed_jump = source_width == 2 and _isa_has_extension(row.get("isa"), "c")
    if compressed_jump and target == "jal":
        target = "c.j"
    reachable = True
    if target in {"jal", "c.j"}:
        pc, target_pc = _pc_value(row.get("pc")), _pc_value(row.get("target_pc"))
        reachable = pc is not None and target_pc is not None and (
            not target_pc & ((2 if _isa_has_extension(row.get("isa"), "c") else 4) - 1)
            and -(1 << 20 if target == "jal" else 2048) <= target_pc - pc
            <= (1 << 20 if target == "jal" else 2048) - 2
        )
    pc, pc_end = _pc_value(row.get("pc")), _pc_value(row.get("pc_end"))
    source_width = pc_end - pc if pc is not None and pc_end is not None else None
    if (outcome is True or probe and outcome is None) and target is not None and label is not None \
            and _target_enabled(row, target) and reachable:
        values = ("x0", label) if target == "jal" else (label,)
        spec = runtime_instruction_spec(target, row.get("xlen"))
        if spec is not None and _text_encodable(spec, spec.operand_roles, values, row.get("_aliases"), isa_profile=_row_isa(row)):
            result.append(
                _source_text("j", (label,)) if target == "c.j"
                else _source_text(target, values)
            )
    target = _r9_control_nop(str(row.get("mnemonic", "")))
    if compressed_jump and target == "addi":
        target = "c.nop"
    if (outcome is False or probe and outcome is None) and target is not None and _target_enabled(row, target):
        values = ("x0", "x0", "0") if target == "addi" else ()
        spec = runtime_instruction_spec(target, row.get("xlen"))
        if spec is not None and _text_encodable(spec, spec.operand_roles, values, row.get("_aliases"), isa_profile=_row_isa(row)):
            result.append(_source_text(target, values))
    return tuple(dict.fromkeys(result))


def _rule_candidate_texts(row: Mapping[str, object], rule: str, *, probe: bool) -> tuple[str, ...]:
    if rule == "R1":
        return _source_alias_texts(row, include_xlen=True) if not probe else tuple(
            text for role in ("rs1", "rs2", "rs3")
            if (text := _probe_register_text(row, role)) is not None
        )[:1]
    if rule == "R2":
        return _source_alias_texts(row) if not probe else tuple(
            text for role in ("rs1", "rs2", "rs3")
            if (text := _probe_register_text(row, role)) is not None
        )[:1]
    if rule == "R3":
        return (_r3_swap_text(row),) if _r3_swap_text(row) is not None else ()
    if rule == "R4":
        return _address_alias_texts(row) if not probe else tuple(
            text for text in (_probe_register_text(row, "rs1"),) if text is not None
        )
    if rule == "R5":
        return _r5_texts(row, guided=not probe)
    if rule == "R6":
        return _same_shape_texts(
            row, tuple(item.get("mnemonic") for item in row["rule_facts"].get("effect_view_siblings", ())
                      if isinstance(item, Mapping) and item.get("kind") == "load-extension-view")
        )
    if rule == "R7":
        return _r7_texts(row)
    if rule == "R8":
        return _same_shape_texts(
            row, tuple(row["rule_facts"].get("fp_form_family_siblings", ())),
            fp_guard=not probe,
        )
    if rule == "R9":
        return _r9_texts(row, probe=probe)
    if rule == "R10":
        facts = row.get("rule_facts", {})
        roles, operands = tuple(facts.get("operand_roles", ())), list(row.get("operands", ()))
        if roles and len(roles) == len(operands) and "rd" in roles:
            index = roles.index("rd")
            if _register_index(operands[index]) in (None, 0):
                return ()
            operands[index] = "x0"
            spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
            if spec is not None and _text_encodable(spec, roles, operands, row.get("_aliases"), isa_profile=_row_isa(row)):
                return (_source_text(str(row["mnemonic"]), operands),)
        return ()
    if rule == "R11":
        facts = row.get("rule_facts", {})
        operands = tuple(row.get("operands", ()))
        roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
        result = []
        if (isinstance(facts, Mapping) and len(roles) == len(operands) == 3
                and "imm" in roles and _layout_int(
                    operands[roles.index("imm")], row.get("_aliases")
                ) == 0):
            for target in facts.get("zero_immediate_targets", ()):
                spec = runtime_instruction_spec(str(target), row.get("xlen"))
                target_roles = spec.operand_roles if spec is not None else ()
                values = (operands[0], operands[1], "0")
                if (spec is not None and _target_enabled(row, str(target))
                        and _text_encodable(spec, target_roles, values, row.get("_aliases"), isa_profile=_row_isa(row))):
                    result.append(_source_text(str(target), values))
        siblings = tuple(
            str(item.get("mnemonic")) for item in facts.get("effect_view_siblings", ())
            if isinstance(item, Mapping)
            and (probe or _r11_target_condition_holds(row, str(item.get("mnemonic", ""))))
        )
        result.extend(_same_shape_texts(row, siblings))
        return tuple(dict.fromkeys(result))
    return ()


def _load_value(value: int, width: int, xlen: int, signedness: str) -> int:
    if signedness == "signed" and value & (1 << (width * 8 - 1)):
        value -= 1 << (width * 8)
    return value & ((1 << xlen) - 1)


def _pure_gpr_witness(row: Mapping[str, object], witness: Mapping[str, object], rd: int) -> bool:
    before, after = _gpr_state(witness.get("before_gpr")), _gpr_state(witness.get("after_gpr"))
    if before is None or after is None or row.get("xlen") not in {32, 64}:
        return False
    after_pc, expected_pc = _pc_value(witness.get("after_pc")), _pc_value(row.get("pc_end"))
    if after_pc is None or expected_pc is not None and after_pc != expected_pc:
        return False
    reads = witness.get("memory_reads")
    invalid_fields = witness.get("_invalid_trace_fields", ())
    if not isinstance(invalid_fields, (list, tuple, set)):
        invalid_fields = ()
    if (("memory_reads" in witness and
         (not isinstance(reads, (list, tuple)) or reads))
            or "memory_reads" in invalid_fields
            or witness.get("branch_outcome") is not None
            or witness.get("fault") is not None
            or witness.get("fault_pc") is not None
            or witness.get("fault_address") is not None
            or any(witness.get(name) is not None for name in (
                "before_fpr_rawbits", "after_fpr_rawbits", "before_fflags",
                "after_fflags", "before_frm", "after_frm", "link", "ialign",
            ))):
        return False
    mask = (1 << row["xlen"]) - 1
    return all(
        (before[index] & mask) == (after[index] & mask)
        for index in range(32) if index != rd
    )


def _r11_target_condition_holds(row: Mapping[str, object], target: str) -> bool:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    if len(roles) != len(operands):
        return False
    target = str(target).replace("_", ".")
    zero_targets = tuple(facts.get("zero_immediate_targets", ())) \
        if isinstance(facts, Mapping) else ()
    if target in zero_targets:
        return _layout_int(
            operands[roles.index("imm")], row.get("_aliases")
        ) == 0 if "imm" in roles else False
    edge = next(
        (
            item for item in facts.get("effect_view_siblings", ())
            if isinstance(item, Mapping)
            and str(item.get("mnemonic", "")).replace("_", ".") == target
        ),
        None,
    ) if isinstance(facts, Mapping) else None
    source_spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
    target_spec = runtime_instruction_spec(target, row.get("xlen"))
    witnesses = _row_witnesses(row)
    if (edge is None or source_spec is None or target_spec is None
            or source_spec.effect_kind != "gpr" or target_spec.effect_kind != "gpr"
            or not _target_enabled(row, target) or not witnesses):
        return False
    index = {role: roles.index(role) for role in roles}
    registers = {
        role: _register_index(operands[position])
        for role, position in index.items()
        if role in {"rd", "rs1", "rs2", "rs3"}
    }
    if any(value is None for value in registers.values()):
        return False
    width = gpr_result_width_bits(str(row.get("mnemonic", "")), row.get("xlen"))
    width = 32 if row.get("xlen") == 32 else width
    if width not in {32, 64}:
        return False
    mask, sign_bit = (1 << width) - 1, 1 << (width - 1)
    kind = str(edge.get("kind", ""))
    for witness in witnesses:
        before = _gpr_state(witness.get("before_gpr"))
        after = _gpr_state(witness.get("after_gpr"))
        if before is None or after is None:
            return False
        if kind == "signedness-compare":
            left = before[registers["rs1"]] & mask
            if "rs2" in registers:
                right = before[registers["rs2"]] & mask
            elif "imm" in index:
                immediate = _layout_int(
                    operands[index["imm"]], row.get("_aliases")
                )
                if immediate is None:
                    return False
                right = immediate & mask
            else:
                return False
            if bool(left & sign_bit) != bool(right & sign_bit):
                return False
        elif kind == "signedness-shift":
            source = before[registers["rs1"]] & mask
            shift_mask = (1 << (5 if width == 32 else 6)) - 1
            if "rs2" in registers:
                shift = before[registers["rs2"]] & shift_mask
            elif "imm" in index:
                immediate = _layout_int(
                    operands[index["imm"]], row.get("_aliases")
                )
                if immediate is None:
                    return False
                shift = immediate & shift_mask
            else:
                return False
            if source & sign_bit and shift:
                return False
        elif kind in {"unary-full-w-ctz", "unary-full-w-cpop"}:
            if row.get("xlen") != 64 or "rs1" not in registers:
                return False
            guard = full_w_view_guard(kind)
            if guard == "low32-nonzero" and before[registers["rs1"]] & 0xFFFFFFFF == 0:
                return False
            if guard == "high32-zero" and before[registers["rs1"]] >> 32:
                return False
        elif kind == "high-multiply-signedness":
            source_signed = high_multiply_signed_source_operands(source_spec.mnemonic)
            target_signed = high_multiply_signed_source_operands(target_spec.mnemonic)
            if source_signed is None or target_signed is None:
                return False
            changed = source_signed ^ target_signed
            if any(role not in registers or before[registers[role]] & sign_bit for role in changed):
                return False
        elif kind == "division-remainder-signedness":
            if "rs1" not in registers or "rs2" not in registers:
                return False
            divisor = before[registers["rs2"]] & mask
            if divisor and (
                before[registers["rs1"]] & sign_bit or divisor & sign_bit
            ):
                return False
        elif kind == "full-to-w-result":
            if row.get("xlen") != 64 or "rd" not in registers:
                return False
            value = after[registers["rd"]]
            low = value & 0xFFFFFFFF
            expected = low | (mask ^ 0xFFFFFFFF) if low & 0x80000000 else low
            if value != expected:
                return False
        else:
            return False
    return True


def _r11_consumer_witness(
    witness: Mapping[str, object], rd: int, xlen: int,
    *, row: Mapping[str, object] | None = None, profile: object | None = None,
) -> bool:
    consumer = witness.get("consumer")
    if not isinstance(consumer, Mapping) or consumer.get("kind") not in {"gpr", "register"}:
        return False
    register = consumer.get("register")
    register = _register_index(register) if isinstance(register, str) else _int_value(register)
    width = _int_value(consumer.get("width"))
    if (register != rd or width not in {32, 64} or width > xlen
            or consumer.get("signedness") not in {"signed", "unsigned"}
            or not all(isinstance(consumer.get(name), str) and consumer[name]
                       for name in ("instruction_id", "occurrence_id"))
            or _pc_value(consumer.get("pc")) is None):
        return False
    if row is None or profile is None:
        return True
    occurrences = tuple(getattr(profile, "occurrences", ()))
    rows = {
        item.get("id"): item for item in getattr(profile, "instructions", ())
        if isinstance(item, Mapping)
    }
    producer_index = next(
        (index for index, item in enumerate(occurrences)
         if item.get("id") == witness.get("id")), None,
    )
    sink_index = next(
        (index for index, item in enumerate(occurrences)
         if item.get("id") == consumer.get("occurrence_id")), None,
    )
    sink = rows.get(consumer.get("instruction_id"))
    if (producer_index is None or sink_index is None or sink_index <= producer_index
            or not isinstance(sink, Mapping)):
        return False
    sink_spec = runtime_instruction_spec(
        str(sink.get("mnemonic", "")), sink.get("xlen"),
    )
    sink_width = gpr_result_width_bits(
        str(sink.get("mnemonic", "")), sink.get("xlen"),
    )
    sink_signedness = signedness_mode_for_generation(str(sink.get("mnemonic", "")))
    if (
        sink_spec is None
        or sink_spec.effect_kind not in {"gpr", "private-load", "gpr-from-fpr"}
        or width != sink_width
        or sink_signedness == "mixed"
        or sink_signedness in {"signed", "unsigned"}
        and consumer.get("signedness") != sink_signedness
    ):
        return False
    sink_occurrence = occurrences[sink_index]
    if (sink_occurrence.get("instruction_id") != consumer.get("instruction_id")
            or _pc_value(sink_occurrence.get("pc")) != _pc_value(consumer.get("pc"))):
        return False
    for occurrence in occurrences[producer_index + 1:sink_index]:
        candidate = rows.get(occurrence.get("instruction_id"))
        if not isinstance(candidate, Mapping):
            return False
        facts = candidate.get("rule_facts", {})
        roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
        operands = tuple(candidate.get("operands", ()))
        if len(roles) != len(operands):
            return False
        if any(
            role != "rd" and (
                _register_index(operand) == rd
                or (_memory_parts(str(operand)) or (None, None))[1] == rd
            )
            for role, operand in zip(roles, operands)
        ) or any(
            role == "rd" and _register_index(operand) == rd
            for role, operand in zip(roles, operands)
        ):
            return False
    facts = sink.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(sink.get("operands", ()))
    if len(roles) != len(operands):
        return False
    return any(
        role != "rd" and (
            _register_index(operand) == rd
            or (_memory_parts(str(operand)) or (None, None))[1] == rd
        )
        for role, operand in zip(roles, operands)
    )


def _rule_witness_present(
    row: Mapping[str, object], rule: str, *, strict: bool = True,
    profile: object | None = None,
) -> bool:
    facts = row.get("rule_facts", {})
    if not isinstance(facts, Mapping):
        return False
    witnesses = _row_witnesses(row)
    fields = set(_RULE_TRACE_FIELDS.get(rule, ())) - _OPTIONAL_TRACE_FIELDS
    if not witnesses or any(
        not all(_witness_field_valid(name, witness.get(name)) for name in fields)
        for witness in witnesses
    ):
        return False
    if any(
        isinstance(invalid := witness.get("_invalid_trace_fields"), (list, tuple, set))
        and any(name in _RULE_TRACE_FIELDS.get(rule, ()) for name in invalid)
        for witness in witnesses
    ):
        return False
    allowed_fields = _RULE_WITNESS_DOMAINS.get(rule, frozenset())
    if strict and any(
        any(witness.get(name) is not None for name in _TRACE_STATE_FIELDS - allowed_fields)
        for witness in witnesses
    ):
        return False
    if any(
        any(witness.get(name) is not None for name in _OPTIONAL_TRACE_FIELDS)
        and any(
            witness.get(name) is None
            or not _witness_field_valid(name, witness.get(name))
            for name in _OPTIONAL_TRACE_FIELDS
        )
        for witness in witnesses
    ):
        return False
    if rule in {"R4", "R6", "R7", "R8", "R9"} and any(
        any(witness.get(name) is not None for name in _OPTIONAL_TRACE_FIELDS)
        for witness in witnesses
    ):
        return False
    if rule in {"R1", "R3", "R5"} and facts.get("effect") == "gpr":
        roles = tuple(facts.get("operand_roles", ()))
        operands = tuple(row.get("operands", ()))
        rd = _register_index(operands[roles.index("rd")]) \
            if "rd" in roles and len(roles) == len(operands) else None
        mask = (1 << row["xlen"]) - 1 if row.get("xlen") in {32, 64} else 0
        if rd is None or any(
            (before := _gpr_state(witness.get("before_gpr"))) is None
            or (after := _gpr_state(witness.get("after_gpr"))) is None
            or any((before[index] & mask) != (after[index] & mask)
                   for index in range(32) if index != rd)
            for witness in witnesses
        ):
            return False
    if rule == "R7":
        roles = tuple(facts.get("operand_roles", ()))
        operands = tuple(row.get("operands", ()))
        rd = _fp_index(operands[roles.index("rd")]) \
            if "rd" in roles and len(roles) == len(operands) else None
        if (
            not _r7_texts(row)
            or rd is None
            or any(
                witness.get("before_frm") != witness.get("after_frm")
                or witness.get("before_fflags") != witness.get("after_fflags")
                or any(
                    witness["before_fpr_rawbits"][index]
                    != witness["after_fpr_rawbits"][index]
                    for index in range(32) if index != rd
                )
                for witness in witnesses
            )
        ):
            return False
    if rule == "R8":
        siblings = tuple(
            item for item in facts.get("fp_form_family_siblings", ())
            if isinstance(item, str)
        )
        if not any(_fp_condition_holds_row(row, target) for target in siblings):
            return False
    if rule in {"R4", "R6"}:
        operands = tuple(row.get("operands", ()))
        roles = tuple(facts.get("operand_roles", ()))
        memory = _memory_parts(str(operands[1])) \
            if len(operands) == 2 and {"rd", "rs1", "imm"} <= set(roles) else None
        base, offset, xlen = (
            memory[1], memory[0], row.get("xlen")
        ) if memory else (None, None, None)
        if base is None or offset is None or xlen not in {32, 64}:
            return False
        width = load_width_bytes(str(row.get("mnemonic", "")).replace("_", "."))
        signedness = facts.get("signedness")
        if width is None or rule == "R6" and signedness not in {"signed", "unsigned"}:
            return False
        rd = _register_index(operands[0]) if "rd" in roles and operands else None
        if rule == "R6" and rd in (None, 0):
            return False
        mask = (1 << xlen) - 1
        for witness in witnesses:
            reads = witness.get("memory_reads", ())
            before = _gpr_state(witness.get("before_gpr"))
            after = _gpr_state(witness.get("after_gpr"))
            if (not isinstance(reads, (list, tuple)) or len(reads) != 1
                    or before is None or after is None
                    or any(
                        (before[index] & mask) != (after[index] & mask)
                        for index in range(32) if index != rd
                    )):
                return False
            read = reads[0]
            if (not isinstance(read, Mapping)
                    or read.get("memory_scope") != "private"
                    or _int_value(read.get("address")) != (before[base] + offset) & ((1 << xlen) - 1)
                    or _int_value(read.get("width")) != width):
                return False
            raw_value = _int_value(read.get("value"))
            if (raw_value is None or not 0 <= raw_value < 1 << (width * 8)
                    or rd is None
                    or rd != 0 and after[rd] & ((1 << (width * 8)) - 1) != raw_value):
                return False
            if rule == "R6":
                roles = tuple(facts.get("operand_roles", ()))
                rd = _register_index(operands[roles.index("rd")]) if "rd" in roles else None
                after = _gpr_state(witness.get("after_gpr"))
                if rd is None or after is None:
                    return False
                if rd != 0 and _load_value(
                        raw_value, width, xlen, facts.get("signedness")
                ) != after[rd]:
                    return False
                if rd != 0 and not any(
                    _load_value(
                        raw_value, width, xlen,
                        signedness_mode_for_generation(str(sibling.get("mnemonic", ""))),
                    )
                    == after[rd]
                    for sibling in facts.get("effect_view_siblings", ())
                    if isinstance(sibling, Mapping)
                ):
                    return False
    if rule == "R9" and (
        not _final_layout_valid(row)
        or any(not _r9_witness_consistent(row, witness) for witness in witnesses)
        or any(
            _pc_value(witness.get("link")) is None
            or _int_value(witness.get("ialign")) not in {2, 4}
            for witness in witnesses
        )
    ):
        return False
    if rule == "R5" and any(
        _pc_value(witness.get("after_pc")) != _pc_value(row.get("pc_end"))
        for witness in witnesses
    ):
        return False
    if rule in {"R10", "R11"}:
        roles = tuple(facts.get("operand_roles", ()))
        operands = tuple(row.get("operands", ()))
        rd = _register_index(operands[roles.index("rd")]) \
            if "rd" in roles and len(roles) == len(operands) else None
        if rd is None or any(not _pure_gpr_witness(row, witness, rd) for witness in witnesses):
            return False
    if rule == "R11" and any(
        not _r11_consumer_witness(witness, rd, row["xlen"], row=row, profile=profile)
        for witness in witnesses
    ):
        return False
    if rule == "R11" and not _rule_candidate_texts(row, "R11", probe=False):
        return False
    if rule == "R10":
        facts = row.get("rule_facts", {})
        roles, operands = tuple(facts.get("operand_roles", ())), tuple(row.get("operands", ()))
        rd = _register_index(operands[roles.index("rd")]) if "rd" in roles and len(roles) == len(operands) else None
        mask = (1 << row["xlen"]) - 1 if row.get("xlen") in {32, 64} else 0
        if rd is None or any(
            (after := _gpr_state(witness.get("after_gpr"))) is None or after[rd] & mask
            for witness in witnesses
        ):
            return False
    return rule != "R10" or all(
        isinstance(witness.get("consumer"), Mapping)
        and witness["consumer"].get("kind") == "none"
        and witness["consumer"].get("proof") == "dynamic-exhaustive"
        for witness in witnesses
    )


def _r9_witness_consistent(
    row: Mapping[str, object], witness: Mapping[str, object],
) -> bool:
    outcome = witness.get("branch_outcome")
    if _pc_value(witness.get("link")) != 0:
        return False
    ialign = _int_value(witness.get("ialign"))
    if ialign not in {2, 4}:
        return False
    isa = row.get("isa")
    if isinstance(isa, str) and isa and ialign != (2 if _isa_has_extension(isa, "c") else 4):
        return False
    before = _gpr_state(witness.get("before_gpr"))
    if before is not None:
        facts = row.get("rule_facts", {})
        roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
        operands = tuple(row.get("operands", ()))
        registers = {
            role: _register_index(operands[roles.index(role)])
            for role in ("rs1", "rs2")
            if role in roles and roles.index(role) < len(operands)
        }
        if any(value is None for value in registers.values()):
            return False
        xlen = row.get("xlen")
        if xlen not in {32, 64}:
            return False
        mask, sign = (1 << xlen) - 1, 1 << (xlen - 1)
        left = before[registers["rs1"]] & mask if "rs1" in registers else None
        right = before[registers["rs2"]] & mask if "rs2" in registers else 0
        mnemonic = str(row.get("mnemonic", "")).replace("_", ".")
        if mnemonic in {"beqz", "c.beqz"}:
            expected = left == 0
        elif mnemonic in {"bnez", "c.bnez"}:
            expected = left != 0
        elif mnemonic in {"bltz", "c.bltz"}:
            expected = left - (1 << xlen) if left & sign else left
            expected = expected < 0
        elif mnemonic in {"bgez", "c.bgez"}:
            expected = left - (1 << xlen) if left & sign else left
            expected = expected >= 0
        elif mnemonic == "bgtz":
            expected = (left - (1 << xlen) if left & sign else left) > 0
        elif mnemonic == "blez":
            expected = (left - (1 << xlen) if left & sign else left) <= 0
        elif mnemonic in {"bgt", "ble", "bgtu", "bleu"}:
            if right is None:
                return False
            if mnemonic in {"bgtu", "bleu"}:
                expected = left > right if mnemonic == "bgtu" else left <= right
            else:
                signed_left = left - (1 << xlen) if left & sign else left
                signed_right = right - (1 << xlen) if right & sign else right
                expected = signed_left > signed_right if mnemonic == "bgt" else signed_left <= signed_right
        elif mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}:
            if mnemonic == "beq":
                expected = left == right
            elif mnemonic == "bne":
                expected = left != right
            elif mnemonic == "bltu":
                expected = left < right
            elif mnemonic == "bgeu":
                expected = left >= right
            else:
                signed_left = left - (1 << xlen) if left & sign else left
                signed_right = right - (1 << xlen) if right & sign else right
                expected = signed_left < signed_right if mnemonic == "blt" else signed_left >= signed_right
        elif mnemonic in {"beqi", "bnei"}:
            if "imm" not in roles or roles.index("imm") >= len(operands):
                return False
            immediate = _layout_int(
                operands[roles.index("imm")], row.get("_aliases")
            )
            if immediate is None:
                return False
            expected = left == (immediate & mask) if mnemonic == "beqi" else left != (immediate & mask)
        else:
            expected = None
        if expected is not None and (outcome == "taken") != expected:
            return False
    target = _pc_value(row.get("target_pc"))
    after = _pc_value(witness.get("after_pc"))
    fallthrough = _pc_value(row.get("pc_end"))
    if not isinstance(isa, str) or not isa or target is None or after is None or fallthrough is None:
        return False
    if target & (ialign - 1) or after & (ialign - 1) or fallthrough & (ialign - 1):
        return False
    return after == (target if outcome == "taken" else fallthrough)


def _r2_before_states(row: Mapping[str, object]) -> tuple[tuple[int, ...] | None, ...]:
    return tuple(_gpr_state(item.get("before_gpr")) for item in _row_witnesses(row))


def _source_alias_texts(
    row: Mapping[str, object], *, include_xlen: bool = False,
) -> tuple[str, ...]:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
    states = _r2_before_states(row)
    if (spec is None or not states or any(state is None for state in states)
            or len(roles) != len(operands)):
        return ()
    projections = dict(facts.get("source_projections", ())) if isinstance(facts, Mapping) else {}
    register_count = 16 if str(row.get("isa", "")).lower().startswith("rv32e") else 32
    result = []
    for role, projection in projections.items():
        if (role not in {"rs1", "rs2", "rs3"}
                or role not in roles
                or (projection == "xlen") != include_xlen):
            continue
        name = str(projection)
        width = row.get("xlen") if name == "xlen" else int(name[3:]) if name.startswith("low") and name[3:].isdigit() else None
        if not isinstance(width, int) or not 0 < width <= 64:
            continue
        index = roles.index(role)
        source = _register_index(operands[index])
        if source is None:
            continue
        mask = (1 << width) - 1
        for candidate in range(register_count):
            if candidate == source or not all(
                (state[source] & mask) == (state[candidate] & mask) for state in states
            ):
                continue
            rewritten = list(operands)
            rewritten[index] = f"x{candidate}"
            if not _text_encodable(spec, roles, rewritten, row.get("_aliases"), isa_profile=_row_isa(row)):
                continue
            result.append(f"{row.get('mnemonic')} {', '.join(rewritten)}")
    return tuple(dict.fromkeys(result))


def _program_relation_facts(
    row: Mapping[str, object], profile: object | None = None, *, strict: bool = True,
    occurrence_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    facts = row.get("rule_facts", {})
    reason = {
        "R1": "no-zero-immediate-form",
        "R2": "no-non-xlen-source-projection",
        "R3": "no-distinct-permutation-pair",
        "R4": "no-address-coordinate-form",
        "R5": "no-effect-equivalent-sibling",
        "R6": "no-load-extension-form",
        "R7": "no-rounding-mode-form",
        "R8": "no-fp-form-family",
        "R9": "no-control-successor-form",
        "R10": "destination-suppression-not-open",
        "R11": "no-width-signedness-form",
    }
    occurrence_ids = tuple(
        occurrence_ids
        if occurrence_ids is not None else (
            item.get("id") for item in getattr(profile, "occurrences", ())
            if isinstance(item, Mapping)
            and item.get("instruction_id") == row.get("id")
            and isinstance(item.get("id"), str)
        )
    )
    if not isinstance(facts, Mapping) or facts.get("definedness") != "catalog-known":
        return {
            rule: {
                "status": "definedness-gap", "route": "direct",
                "reason": "instruction-descriptor-unresolved",
                "applicable": False, "ready": False,
            }
            for rule in reason
        }
    if not _target_enabled(row, str(row.get("mnemonic", ""))):
        return {
            rule: {
                "status": "isa-gap", "route": "profile",
                "reason": "instruction-disabled-for-isa",
                "applicable": False, "ready": False,
            }
            for rule in reason
        }
    result = {
        rule: {"status": "not-applicable", "route": "direct", "reason": reason[rule]}
        for rule in reason
    }
    effect = facts.get("effect")
    if effect == "gpr":
        projections = dict(facts.get("source_projections", ()))
        source_roles = {
            role for role in facts.get("operand_roles", ())
            if role in {"rs1", "rs2", "rs3"}
        }
        missing_projections = source_roles - projections.keys()
        if missing_projections:
            result["R2"] = {
                "status": "definedness-gap", "route": "direct",
                "reason": "source-projection-fact-incomplete",
            }
        elif any(
            role in {"rs1", "rs2", "rs3"} and projection not in (None, "xlen")
            for role, projection in projections.items()
        ):
            states = _r2_before_states(row)
            result["R2"] = (
                {"status": "available", "route": "program"}
                if occurrence_ids and len(states) == len(occurrence_ids)
                and states and all(state is not None for state in states)
                and _r3_anchor_ready(row, profile)
                else {"status": "observer-gap", "route": "profile",
                      "reason": "all-occurrence-before-gpr-required"
                      if not states or any(state is None for state in states)
                      else "final-layout-anchor-required"}
            )
        swapped = _r3_swap_text(row)
        r3_witnesses_ready = (
            bool(occurrence_ids) and len(_row_witnesses(row)) == len(occurrence_ids)
            and _rule_witness_present(row, "R3", strict=strict, profile=profile)
        )
        facts = row.get("rule_facts", {})
        roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
        operands = tuple(row.get("operands", ()))
        registers_defined = len(roles) == len(operands) and all(
            _register_index(value) is not None
            for role, value in zip(roles, operands)
            if role in {"rd", "rs1", "rs2", "rs3"}
        )
        if swapped is not None and registers_defined:
            result["R3"] = (
                {"status": "available", "route": "program"}
                if r3_witnesses_ready and _r3_anchor_ready(row, profile)
                else {"status": "observer-gap", "route": "profile",
                      "reason": "final-layout-anchor-required"
                      if not _r3_anchor_ready(row, profile)
                      else "execution-occurrence-witness-required"}
            )
    for rule, gap_reason in _RULE_OBSERVER_GAP_REASONS.items():
        if (result[rule]["status"] == "not-applicable"
                and _rule_form_possible(row, rule)):
            result[rule] = (
                {"status": "available", "route": "program"}
                if _rule_witness_present(row, rule, strict=strict, profile=profile)
                else {"status": "observer-gap", "route": "profile", "reason": gap_reason}
            )
    for relation in result.values():
        status = relation.get("status")
        relation["applicable"] = status in {"available", "observer-gap"}
        relation["ready"] = status == "available"
    return result


# 这里只保留 catalog 没有的汇编伪指令；真实编码统一走 runtime_spec。
_PROFILE_CONTROL = frozenset({
    "beqz", "bnez", "bltz", "bgez", "bltzal", "bgezal", "bgt", "ble",
    "bgtz", "blez", "bgtu", "bleu", "beqi", "bnei", "j", "jr", "ret", "call", "tail",
})
_PROFILE_BRANCH_IMMEDIATE_OPS = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez",
    "bltzal", "bgezal", "bgt", "ble", "bgtz", "blez", "bgtu", "bleu", "beqi", "bnei",
    "c.beqz", "c.bnez", ".insn.b",
})
_PROFILE_JUMP_IMMEDIATE_OPS = frozenset({"j", "jal", "c.j", "c.jal", ".insn.j"})
_PROFILE_DIRECT_CONTROL = frozenset({
    "j", "jr", "ret", "call", "tail", "ecall", "ebreak", "mret", "sret",
    "scall", "sbreak", "wfi", "unimp", "c.unimp", "c.ebreak", "cm.popret", "cm.popretz",
})
_PROFILE_INDIRECT_CONTROL = frozenset({"cm.jt", "cm.jalt"})
_PROFILE_CONTROL_ARITIES = {
    **dict.fromkeys(
        ("beqz", "bnez", "bltz", "bgez", "bltzal", "bgezal", "bgtz", "blez"),
        2,
    ),
    **dict.fromkeys(("bgt", "ble", "bgtu", "bleu"), 3),
    **dict.fromkeys(("beqi", "bnei"), 3),
    **dict.fromkeys(("j", "call", "tail", "jr", "c.jr", "c.jalr", "cm.jt", "cm.jalt"), 1),
    **dict.fromkeys(
        ("ret", "ecall", "ebreak", "scall", "sbreak", "mret", "sret", "wfi", "unimp", "c.unimp", "c.ebreak"),
        0,
    ),
}
_PROFILE_PSEUDO_ARITIES = {
    **dict.fromkeys((
        "la", "lla", "lga", "li", "mv", "not", "neg", "negw", "sext.b",
        "sext.h", "sext.w", "zext.b", "zext.h", "zext.w", "fmv.s", "fabs.s", "fneg.s",
        "fmv.d", "fabs.d", "fneg.d", "fmv.h", "fabs.h", "fneg.h", "seqz", "snez", "sltz", "sgtz", "csrr",
        "csrw", "csrs", "csrc", "csrwi", "csrsi", "csrci",
    ), 2),
    **dict.fromkeys(("nop", "pause"), 0),
    **dict.fromkeys(("rdcycle", "rdtime", "rdinstret", "rdcycleh", "rdtimeh", "rdinstreth"), 1),
    **dict.fromkeys(("fmv.s.x", "fmv.x.s"), 2),
    **dict.fromkeys(("frcsr", "frflags", "frrm"), 1),
    **dict.fromkeys(("fsrm",), 2),
}
_PSEUDO_REQUIRED_EXTENSIONS = {
    "c.ebreak": "c", "c.unimp": "c", "cm.jt": "zcmt", "cm.jalt": "zcmt",
    "cm.popret": "zcmp", "cm.popretz": "zcmp",
    "frcsr": "f", "fscsr": "f", "frflags": "f", "frrm": "f", "fsrm": "f", "fsrmi": "f",
    "pause": "zihintpause", "fmv.s.x": "f", "fmv.x.s": "f",
}


def _pseudo_enabled(mnemonic: str, isa: str | None) -> bool:
    if mnemonic in {"negw", "sext.w", "zext.w"} and isa is not None \
            and not isa.startswith("rv64"):
        return False
    if mnemonic in {"rdcycleh", "rdtimeh", "rdinstreth"} and isa is not None \
            and not isa.startswith("rv32"):
        return False
    extension = _PSEUDO_REQUIRED_EXTENSIONS.get(mnemonic)
    return isa is None or extension is None or extension in enabled_extensions(isa) \
        or extension == "f" and bool(x_register_fp_extensions(isa))


def _fp_pseudo_enabled(mnemonic: str, isa: str | None) -> bool:
    if isa is None:
        return True
    enabled = enabled_extensions(isa)
    if mnemonic.endswith(".h"):
        return "zfh" in enabled or "zhinx" in enabled
    return (
        "d" in enabled if mnemonic.endswith(".d")
        else "f" in enabled or bool(x_register_fp_extensions(isa))
    ) or mnemonic.endswith(".d") and {"zfinx", "zdinx"} <= enabled


def _control_shape_for_mnemonic(mnemonic: str) -> str | None:
    if mnemonic == ".insn.i":
        return "indirect"
    if mnemonic == ".insn.j":
        return "direct"
    if mnemonic == ".insn.b":
        return "conditional"
    if mnemonic in _PROFILE_INDIRECT_CONTROL:
        return "indirect"
    spec = runtime_instruction_spec(mnemonic)
    return spec.control_shape if spec is not None else (
        "direct" if mnemonic in _PROFILE_DIRECT_CONTROL
        else "conditional" if mnemonic in _PROFILE_CONTROL else None
    )


def _control_shape(row: Mapping[str, object]) -> str | None:
    mnemonic = str(row.get("control_mnemonic") or row.get("mnemonic", ""))
    operands = row.get("operands") or ()
    if mnemonic == "jr" and operands and _register_index(str(operands[0])) is None:
        return "indirect"
    facts = row.get("rule_facts")
    shape = facts.get("control_shape") if isinstance(facts, Mapping) else None
    return shape or _control_shape_for_mnemonic(mnemonic)


def _static_branch_outcome(row: Mapping[str, object], aliases=None) -> bool | None:
    mnemonic = row.get("mnemonic")
    operands = tuple(row.get("operands") or ())
    aliases = row.get("_aliases", aliases)
    if row.get("control_mnemonic") == ".insn.b" and len(operands) >= 5:
        funct3 = _layout_int(operands[1], aliases)
        left, right = _register_index(operands[2]), _register_index(operands[3])
        if left is not None and left == right and funct3 in {0, 1}:
            return funct3 == 0
    if mnemonic in {"beqi", "bnei"} and len(operands) >= 3:
        register, immediate = _register_index(operands[0]), _layout_int(operands[1], aliases)
        if register == 0 and immediate is not None:
            return immediate == 0 if mnemonic == "beqi" else immediate != 0
    if mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "bgt", "ble", "bgtu", "bleu"} and len(operands) >= 3:
        left, right = _register_index(operands[0]), _register_index(operands[1])
        if left is not None and left == right:
            return mnemonic in {"beq", "bge", "bgeu", "ble", "bleu"}
    if mnemonic in {"beqz", "bnez", "bltz", "bgez", "bgtz", "blez", "bltzal", "bgezal", "c.beqz", "c.bnez"} and operands:
        if _register_index(operands[0]) == 0:
            return mnemonic in {"beqz", "bgez", "blez", "bgezal", "c.beqz"}
    return None


def _insn_control_mnemonic(
    operands: Sequence[str], aliases: Mapping[str, object] | None = None,
) -> str | None:
    if not operands:
        return None
    fields = operands[0].split(None, 1)
    if len(fields) != 2:
        return None
    kind = {"sb": "b", "uj": "j"}.get(fields[0].lower(), fields[0].lower())
    opcode = _layout_int(fields[1], aliases)
    return {
        ("b", 0x63): ".insn.b",
        ("j", 0x6F): ".insn.j",
        ("i", 0x67): ".insn.i",
    }.get((kind, opcode))
_PROFILE_PSEUDOS = frozenset({
    "la", "lla", "lga", "li", "mv", "nop", "not", "neg", "negw", "sext.b", "sext.h",
    "sext.w", "zext.b", "zext.h", "zext.w", "fmv.s", "fabs.s", "fneg.s", "fmv.d", "fabs.d", "fneg.d",
    "fmv.h", "fabs.h", "fneg.h",
    "seqz", "snez", "sltz", "sgtz", "csrr", "csrw", "csrs", "csrc", "csrwi", "csrsi",
    "csrci", "rdcycle", "rdtime", "rdinstret", "frcsr", "fscsr",
    "frflags", "frrm", "fsrm", "fsrmi", "pause",
    "rdcycleh", "rdtimeh", "rdinstreth", "fmv.s.x", "fmv.x.s",
    "fence.tso", "scall", "sbreak",
})
_BASE_EXPANDING_PSEUDOS = frozenset({"sext.b", "sext.h", "zext.b", "zext.h"})
_FP_PSEUDOS = frozenset({
    "fmv.s", "fabs.s", "fneg.s", "fmv.d", "fabs.d", "fneg.d", "fmv.h", "fabs.h", "fneg.h",
})
_FLOAT_DIRECTIVE_WIDTHS = {".float": 4, ".single": 4, ".double": 8}
_DATA_DIRECTIVE_WIDTHS = {
    ".byte": 1, ".2byte": 2, ".half": 2, ".short": 2,
    ".word": 4, ".4byte": 4, ".long": 4,
    ".dword": 8, ".8byte": 8, ".quad": 8,
}
_KNOWN_PROFILE_DIRECTIVES = frozenset({
    ".text", ".section", ".pushsection", ".popsection", ".previous",
    ".subsection", ".data", ".bss", ".rodata", ".sdata", ".sbss",
    ".tdata", ".tbss", ".include", ".option", ".equ", ".set", ".globl",
    ".global", ".type", ".size", ".attribute", ".file", ".loc", ".ident",
    ".hidden", ".weak", ".local", ".comm", ".common", ".lcomm", ".extern",
    ".end", ".symver", ".uleb128", ".sleb128", ".float", ".single", ".double",
    ".cfi_startproc", ".cfi_endproc", ".cfi_def_cfa", ".cfi_offset",
    ".cfi_restore", ".cfi_remember_state", ".cfi_restore_state", ".cfi_sections",
    ".cfi_escape", ".cfi_signal_frame", ".riscv.attributes",
    ".ascii", ".asciz", ".string", ".balign", ".balignw", ".balignl",
    ".align", ".p2align", ".p2alignw", ".p2alignl", ".space", ".zero", ".skip", ".fill", ".org", ".incbin",
}) | frozenset(_DATA_DIRECTIVE_WIDTHS) | {".insn"}
_PROFILE_DIRECTIVE_PREFIXES = (".cfi_", ".debug_")


def _incbin_size(
    source_path: object, operands: Sequence[str], aliases: Mapping[str, object] | None = None,
    include_context: Sequence[tuple[str, bytes]] = (), base: Path | None = None,
) -> int | None:
    if not 1 <= len(operands) <= 3:
        return None
    name = operands[0].strip()
    if len(name) < 2 or name[0] != name[-1] or name[0] not in {'"', "'"}:
        return None
    if include_context:
        dependency_name = Path(posixpath.normpath(
            ((base or Path(".")) / name[1:-1]).as_posix()
        )).as_posix()
        data = next(
            (data for path, data in include_context
             if Path(path).as_posix() == dependency_name),
            None,
        )
        if data is None:
            return None
        total = len(data)
    else:
        if source_path is None:
            return None
        root = Path(base) if base is not None else Path(source_path).resolve().parent
        path = (root / name[1:-1]).resolve()
        if not path.is_file():
            return None
        total = path.stat().st_size
    skip = _layout_int(operands[1], aliases) if len(operands) > 1 and operands[1].strip() else 0
    count = _layout_int(operands[2], aliases) if len(operands) > 2 and operands[2].strip() else total - (skip or 0)
    return count if skip is not None and count is not None and 0 <= skip <= total and 0 <= count <= total - skip else None


def _leb128_size(
    operands: Sequence[str], signed: bool, aliases: Mapping[str, object] | None = None,
) -> int | None:
    if not operands:
        return 0
    total = 0
    for operand in operands:
        value = _layout_int(operand, aliases)
        if value is None or not -(1 << 64) < value <= (1 << 64) - 1:
            return None
        value &= (1 << 64) - 1
        if signed and value >= 1 << 63:
            value -= 1 << 64
        size = 1
        if signed:
            while not -64 <= value <= 63:
                value >>= 7
                size += 1
        else:
            while value >= 0x80:
                value >>= 7
                size += 1
        total += size
    return total


def _profile_edges(
    rows: tuple[Mapping[str, object], ...], blocks: tuple[str, ...],
    occurrences: Sequence[Mapping[str, object]] = (),
    aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[str, str], ...]:
    labels: dict[str, list[Mapping[str, object]]] = {}
    local_labels: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        for label in row.get("labels", (row.get("label"),)):
            if label:
                (local_labels if str(label).isdigit() else labels).setdefault(
                    (str(row.get("section", ".text")), str(label))
                    if str(label).isdigit() else str(label),
                    [],
                ).append(row)
    rows_by_pc = {
        (str(row.get("section", ".text")), int(row["pc"])): row for row in rows
    }

    def target_block(row: Mapping[str, object]) -> str | None:
        target = str((row.get("operands") or ("",))[-1])
        if _register_index(target) is not None:
            return None
        offset = _int_value(target)
        if offset is not None:
            if _control_shape(row) in {"conditional", "direct"}:
                offset += int(row["pc"])
            return rows_by_pc.get(
                (str(row.get("section", ".text")), offset), {}
            ).get("block")
        base, offset = _resolve_target(target, row.get("_aliases", aliases))
        if base is None:
            return rows_by_pc.get(
                (str(row.get("section", ".text")), offset), {}
            ).get("block")
        if base == ".":
            return rows_by_pc.get(
                (str(row.get("section", ".text")), int(row["pc"]) + offset), {}
            ).get("block")
        candidates = local_labels.get(
            (str(row.get("section", ".text")), base), ()
        ) if base.isdigit() else labels.get(base, ())
        match = re.fullmatch(r"(\d+)([fb])", base.lower())
        if match:
            candidates = tuple(
                item for item in local_labels.get(
                    (str(row.get("section", ".text")), match[1]), ()
                )
                if (item["start"] > row["start"] if match[2] == "f" else item["start"] <= row["start"])
            )
            if not candidates:
                return None
            base_row = candidates[0] if match[2] == "f" else candidates[-1]
        else:
            base_row = candidates[0] if candidates else None
        if base_row is None:
            return None
        target_section = str(base_row.get("section", row.get("section", ".text")))
        return rows_by_pc.get(
            (target_section, int(base_row["pc"]) + offset), {}
        ).get("block")

    occurrence_by_instruction: dict[str, list[Mapping[str, object]]] = {}
    for occurrence in occurrences:
        occurrence_by_instruction.setdefault(
            str(occurrence.get("instruction_id")), []
        ).append(occurrence)

    def branch_outcome(row: Mapping[str, object]) -> bool | None:
        static = _static_branch_outcome(row, aliases)
        if static is not None:
            return static
        observed = row.get("branch_outcome")
        if observed in {"taken", "not-taken"}:
            return observed == "taken"
        items = occurrence_by_instruction.get(str(row["id"]), ())
        outcomes = tuple(item.get("branch_outcome") for item in items)
        if (outcomes and all(outcome in {"taken", "not-taken"} for outcome in outcomes)
                and len(set(outcomes)) == 1):
            return outcomes[0] == "taken"
        return None

    edges = []
    rows_by_block = {}
    for row in rows:
        rows_by_block.setdefault(row.get("block"), []).append(row)
    for index, block in enumerate(blocks):
        block_rows = tuple(rows_by_block.get(block, ()))
        no_fallthrough = any(
            _control_shape(row) in {"direct", "indirect"} or branch_outcome(row) is True
            for row in block_rows
        )
        if index + 1 < len(blocks) and not no_fallthrough:
            edges.append((block, blocks[index + 1]))
        for row in block_rows:
            if _control_shape(row) not in {None, "indirect"} and row.get("operands"):
                target = target_block(row)
                if branch_outcome(row) is not False and target is not None:
                    edges.append((block, target))
    return tuple(dict.fromkeys(edges))


def _branch_target_pc(
    row: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    label_points: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> int | None:
    aliases = row.get("_aliases", aliases)
    target = str((row.get("operands") or ("",))[-1])
    if _register_index(target) is not None:
        return None
    base, offset = _resolve_target(target, aliases)
    if base is None:
        if _control_shape(row) in {"conditional", "direct"}:
            return int(row["pc"]) + offset
        return offset
    if base == ".":
        return int(row["pc"]) + offset
    section = str(row.get("section", ".text"))
    def points_for(label_name: str) -> tuple[tuple[int, int], ...]:
        points = {}
        for item in rows:
            for label in item.get("labels", ()):
                if (str(label) == label_name
                        and (not label_name.isdigit()
                             or str(item.get("section", ".text")) == section)):
                    points[int(item["start"])] = int(item["pc"])
        if points:
            return tuple(sorted(points.items()))
        scoped = (label_points or {}).get((section, label_name), ())
        if not scoped:
            scoped = (label_points or {}).get(label_name, ())
        for start, pc in scoped:
            points.setdefault(int(start), int(pc))
        return tuple(sorted(points.items()))

    points = points_for(base)
    match = re.fullmatch(r"(\d+)([fb])", base.lower())
    if match:
        local_points = points_for(match[1])
        candidates = [
            item for item in local_points
            if (
                item[0] > row["start"]
                if match[2] == "f" else item[0] <= row["start"]
            )
        ]
        return (candidates[0][1] if match[2] == "f" else candidates[-1][1]) + offset if candidates else None
    return points[0][1] + offset if points else None


def _control_target_invalid(
    row: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    label_points: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> bool:
    if _control_shape(row) not in {"conditional", "direct"} or not row.get("operands"):
        return False
    if str(row.get("mnemonic", "")) in {"j", "c.j"} and str(row["operands"][-1]).strip() == ".":
        return False
    target = _branch_target_pc(row, rows, label_points, aliases)
    if target is None:
        return _control_shape(row) == "direct" and _register_index(
            str(row["operands"][-1])
        ) is None
    alignment = 2 if _isa_has_extension(row.get("isa"), "c") else 4
    if target % alignment:
        return True
    mnemonic = str(row.get("mnemonic", ""))
    delta = target - int(row["pc"])
    size = int(row["pc_end"]) - int(row["pc"])
    if mnemonic in {"j", "jal", "c.j", "c.jal"}:
        limit = 2048 if size == 2 else 1 << 20
    elif _control_shape(row) == "conditional":
        limit = 256 if size == 2 else 4096 if size == 4 else 1 << 20
        if size == 6:
            delta = target - (int(row["pc"]) + 2)
        elif size >= 8:
            delta = target - (int(row["pc"]) + 4)
    else:
        return False
    return not -limit <= delta <= limit - 2


def _trace_transition_invalid(
    left_pc: int, left: Mapping[str, object], right_pc: int,
    right: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    aliases: Mapping[str, object] | None = None,
) -> bool:
    if left["id"] == right["id"]:
        return left_pc != right_pc and not left["pc"] <= right_pc < left["pc_end"]
    shape = _control_shape(left)
    if shape == "indirect":
        return False
    target = _branch_target_pc(left, rows, aliases=aliases) \
        if shape in {"conditional", "direct"} else None
    if shape == "conditional":
        outcome = left.get("branch_outcome")
        static_outcome = _static_branch_outcome(left, aliases)
        outcome = outcome if outcome in {"taken", "not-taken"} else (
            "taken" if static_outcome is True else
            "not-taken" if static_outcome is False else None
        )
        if outcome == "taken":
            return target is None or right_pc != target
        if outcome == "not-taken":
            return right_pc != left["pc_end"]
        return right_pc not in {left["pc_end"], target}
    if shape == "direct":
        return left.get("mnemonic") in {"j", "c.j", ".insn.j"} \
            and target is not None and right_pc != target
    return right_pc != left["pc_end"]


def _normalize_target_expression(value: object) -> str:
    text = str(value).strip()
    simple = re.fullmatch(r"\(\s*([A-Za-z_.$][\w.$]*|\d+[fb])\s*\)", text)
    if simple:
        return simple[1]
    difference = re.fullmatch(
        r"\(?\s*([A-Za-z_.$][\w.$]*|\d+[fb])\s*\)?\s*-\s*"
        r"\(?\s*([A-Za-z_.$][\w.$]*|\d+[fb])\s*\)?", text,
    )
    if difference and difference[1] == difference[2]:
        return "0"
    leading_plus = re.fullmatch(r"\+\s*([A-Za-z_.$][\w.$]*|\d+[fb])", text)
    if leading_plus:
        return leading_plus[1]
    leading_number = re.fullmatch(
        r"([+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+))\s*\+\s*"
        r"([A-Za-z_.$][\w.$]*|\d+[fb])", text,
    )
    return f"{leading_number[2]}+{leading_number[1]}" if leading_number else text


def _resolve_target(
    target: str, aliases: Mapping[str, str] | None = None,
) -> tuple[str | None, int]:
    adjustment = 0
    seen = set()
    while True:
        target = _normalize_target_expression(target)
        value = _int_value(target)
        if value is None:
            value = _integer_expression(target)
        if value is None and aliases:
            expanded = str(target)
            for _ in range(len(aliases) + 1):
                previous = expanded
                for name, mapped in aliases.items():
                    expanded = re.sub(
                        rf"(?<![A-Za-z0-9_.$]){re.escape(str(name))}(?![A-Za-z0-9_.$])",
                        f"({mapped})", expanded,
                    )
                if expanded == previous:
                    break
            if expanded != str(target):
                value = _integer_expression(expanded)
        if value is not None:
            return None, value + adjustment
        match = re.fullmatch(
            r"([A-Za-z_.$][\w.$]*|\d+[fb]?)\s*([+-])\s*"
            r"(.+)",
            target,
        )
        offset = _int_value(match[3]) if match is not None else None
        if offset is None and match is not None:
            offset_base, resolved_offset = _resolve_target(match[3], aliases)
            if offset_base is None:
                offset = resolved_offset
        base, offset = (
            (match[1], offset if match[2] == "+" else -offset)
            if match is not None and offset is not None else (target, 0)
        )
        adjustment += offset
        mapped = aliases.get(base) if aliases is not None else None
        if mapped is None or base in seen:
            return base, adjustment
        seen.add(base)
        target = str(mapped).strip()


def _layout_int(value: object, aliases: Mapping[str, object] | None = None) -> int | None:
    text = str(value).strip()
    base, resolved = _resolve_target(text, aliases)
    if base is None:
        return resolved
    if len(base) >= 2 and base[0] == base[-1] == "'":
        try:
            literal = ast.literal_eval(base)
        except (SyntaxError, ValueError):
            literal = None
        if isinstance(literal, str) and len(literal) == 1:
            return ord(literal) + resolved
    return None


def _collect_equ_aliases(lines) -> dict[str, str]:
    aliases = {}
    for raw in lines:
        body = _strip_comments(str(raw)).strip()
        while match := re.match(r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body):
            body = match[2].strip()
        parts = body.split(None, 1)
        if parts and parts[0].lower() in {".equ", ".set"} and len(parts) == 2:
            operands = _split_operands(parts[1])
            if len(operands) == 2 and operands[0].strip() not in aliases:
                aliases[operands[0].strip()] = operands[1].strip()
    return aliases


def _resolve_memory_alias(
    value: str, aliases: Mapping[str, object] | None = None,
) -> str:
    match = re.fullmatch(r"\s*([A-Za-z_.$][\w.$]*)\s*\(\s*([^()]+?)\s*\)\s*", value)
    if match is None:
        return value
    base, offset = _resolve_target(match[1], aliases)
    return f"{offset}({match[2]})" if base is None else value


def _resolve_size_operand(value: str, aliases: Mapping[str, object] | None = None) -> str:
    value = _resolve_memory_alias(value, aliases)
    base, resolved = _resolve_target(value, aliases)
    return str(resolved) if base is None else value


def _target_expression_valid(value: str) -> bool:
    text = str(value).strip()
    if not text or text == ".":
        return text == "."
    normalized = re.sub(
        r"(?<![A-Za-z0-9_.$])(?:\d+[fb]|[A-Za-z_.$][\w.$]*)",
        "1", text,
    )
    return _integer_expression(normalized) is not None


def _relocation_neutral(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(
        r"%(?:lo|pcrel_lo|tprel_lo)\([^()]+\)(?:\(([^()]+)\))?", value
    )
    if match and match[1]:
        return f"0({match[1]})"
    return "0" if re.fullmatch(
        r"%(?:lo|hi|pcrel_lo|pcrel_hi|tprel_lo)\([^()]+\)", value
    ) else value


_RELOCATION_LOW = frozenset({"lo", "pcrel_lo", "tprel_lo"})
_RELOCATION_HIGH = frozenset({"hi", "pcrel_hi", "tprel_hi"})


def _relocation_valid(mnemonic: str, operands: Sequence[str]) -> bool:
    spec = instruction_spec_for_generation(mnemonic)
    groups = operand_groups(spec.form) if spec is not None else ()
    for operand in operands:
        match = re.fullmatch(
            r"%(lo|hi|pcrel_lo|pcrel_hi|tprel_lo|tprel_hi)\([^()]+\)(?:\(([^()]+)\))?",
            str(operand).strip(),
        )
        if match is None:
            continue
        kind, base = match[1], match[2]
        if base is not None:
            return kind in _RELOCATION_LOW
        if kind in _RELOCATION_LOW:
            return mnemonic == "li" or "imm12" in groups
        return kind in _RELOCATION_HIGH and "imm20" in groups
    return True


def _profile_source_map(observation: object) -> Mapping[str, object] | None:
    evidence = _profile_get(observation, "translation_evidence")
    details = evidence.get("details", {}) if isinstance(evidence, Mapping) else getattr(evidence, "details", {})
    return details.get("source_pc_map") if isinstance(details, Mapping) else None


def _profile_source_pc_map(observation: object) -> dict[int, tuple[int, ...]] | None:
    raw = _profile_source_map(observation)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return {}
    binary_sha256 = raw.get("binary_sha256")
    observed_hashes = tuple(
        value for value in (
            _profile_get(observation, "binary_sha256"),
            _profile_get(observation, "guest_elf_sha256"),
        ) if value is not None
    )
    if (not is_sha256_digest(binary_sha256) or not observed_hashes
            or any(value != binary_sha256 for value in observed_hashes)):
        return {}
    lines = raw.get("lines")
    if not isinstance(lines, Mapping):
        return {}
    result = {}
    for line, values in lines.items():
        try:
            line = int(line)
        except (TypeError, ValueError):
            return {}
        if line < 1 or not isinstance(values, (list, tuple)):
            return {}
        if line in result:
            return {}
        pcs = tuple(_pc_value(value) for value in values)
        if not pcs or any(value is None for value in pcs) or tuple(sorted(pcs)) != pcs:
            return {}
        result[line] = pcs
    mapped_pcs = tuple(pc for values in result.values() for pc in values)
    if len(set(mapped_pcs)) != len(mapped_pcs):
        return {}
    return result


def _profile_final_instruction_map(observation: object) -> dict[int, dict[str, object]] | None:
    raw = _profile_source_map(observation)
    if not isinstance(raw, Mapping) or "instructions" not in raw:
        return None
    items = raw.get("instructions")
    if not isinstance(items, (list, tuple)) or not items:
        return {}
    result = {}
    for item in items:
        if not isinstance(item, Mapping):
            return {}
        pc = _pc_value(item.get("pc"))
        size = _int_value(item.get("size"))
        value = item.get("bytes")
        if (pc is None or size not in {2, 4, 6, 8, 10, 12, 14, 16}
                or not isinstance(value, str) or len(value) != size * 2
                or any(char not in "0123456789abcdefABCDEF" for char in value)
                or pc in result):
            return {}
        result[pc] = {"size": size, "bytes": value.lower()}
    starts = sorted(result)
    if any(left + result[left]["size"] > right for left, right in zip(starts, starts[1:])):
        return {}
    return result


def _logical_source_lines(source_lines: Sequence[str]) -> tuple[int | None, ...]:
    logical, result = 1, []
    for raw in source_lines:
        directive = re.fullmatch(r"#\s*(?:line\s+)?(\d+)(?:\s+.*)?", raw.strip(), re.I)
        if directive:
            result.append(None)
            logical = int(directive[1])
        else:
            result.append(logical)
            logical += 1
    return tuple(result)


def _profile_arch_options(
    body: str, active_isa: str, compressed: bool,
) -> tuple[str, bool]:
    normalized = body.lower()
    words = normalized.split()
    full_arch = re.fullmatch(r"\.option\s+arch\s*,\s*(rv(?:32|64)\S+)", normalized)
    arch_delta = re.fullmatch(
        r"\.option\s+arch\s*,\s*((?:[+-][a-z][a-z0-9]*)(?:\s*,\s*[+-][a-z][a-z0-9]*)*)",
        normalized,
    )
    if arch_delta:
        enabled = set(enabled_extensions(active_isa))
        deltas = tuple(re.split(r"\s*,\s*", arch_delta[1]))
        for delta in deltas:
            (enabled.add if delta[0] == "+" else enabled.discard)(delta[1:])
        previous_compressed = compressed
        active_isa = isa_profile_for_extensions(
            "rv32" if active_isa.lower().startswith("rv32") else "rv64",
            frozenset(enabled),
        )
        if not is_canonical_isa_profile(active_isa):
            raise ValueError("invalid ISA option")
        compressed = (
            profile_uses_compressed_encoding(active_isa)
            if any(
                delta[0] == "+" and (
                    delta[1:] == "c" or delta[1:].startswith("zc")
                )
                for delta in deltas
            )
            else previous_compressed and profile_uses_compressed_encoding(active_isa)
        )
    elif full_arch:
        if not is_canonical_isa_profile(full_arch[1]):
            raise ValueError("invalid ISA option")
        active_isa = full_arch[1]
        compressed = profile_uses_compressed_encoding(active_isa)
    elif normalized == ".option norvc":
        compressed = False
    elif normalized == ".option rvc":
        enabled = set(enabled_extensions(active_isa))
        enabled.add("c")
        active_isa = isa_profile_for_extensions(
            "rv32" if active_isa.lower().startswith("rv32") else "rv64",
            frozenset(enabled),
        )
        compressed = True
    elif words and words[0] == ".option" and tuple(words[1:]) not in {
        ("push",), ("pop",), ("relax",), ("norelax",),
        ("pic",), ("nopic",),
    }:
        raise ValueError("invalid ISA option")
    return active_isa, compressed


def _profile_declared_arch(source_lines: Sequence[str]) -> str | None:
    for raw in source_lines:
        for statement in _split_source_statements(_strip_comments(raw)):
            match = re.fullmatch(
                r"\s*\.attribute\s+arch\s*,\s*([\"'])([^\"']+)\1\s*",
                statement, re.IGNORECASE,
            )
            if match:
                return match[2].lower()
    return None


def build_profile(
    program: object,
    observation: object,
    *,
    input_identity: Mapping[str, object] | None = None,
    expected_trap: bool = False,
) -> ProfileResult:
    """从 trusted reference 的执行轨迹建立源码 profile。"""
    if getattr(program, "run_params", {}).get("route") == "single":
        from ..rvgen.single_program import build_single_profile
        return build_single_profile(program, observation, input_identity=input_identity)
    if isinstance(observation, (list, tuple)):
        observations = tuple(observation)
        observation = observations[0] if observations else None
    if observation is None:
        return ProfileResult(None, "profile-gap:trusted-reference-missing")
    if input_identity is not None and not isinstance(input_identity, Mapping):
        return ProfileResult(None, "profile-gap:input-identity-invalid")
    pcs = _profile_get(observation, "executed_pcs", ())
    if not isinstance(pcs, (list, tuple)) or not pcs:
        return ProfileResult(None, "profile-gap:dynamic-trace-missing")
    try:
        pcs = tuple(_pc_value(item) for item in pcs)
    except (TypeError, ValueError):
        return ProfileResult(None, "profile-gap:dynamic-trace-invalid")
    if any(pc is None for pc in pcs):
        return ProfileResult(None, "profile-gap:dynamic-trace-invalid")
    identity = dict(getattr(program, "run_params", {}))
    identity.update(dict(input_identity or {}))
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError):
        return ProfileResult(None, "profile-gap:input-identity-invalid")
    isa, isa_profile = identity.get("isa"), identity.get("isa_profile")
    if isa is not None and isa_profile is not None and str(isa).lower() != str(isa_profile).lower():
        return ProfileResult(None, "profile-gap:input-identity-mismatch")
    identity["isa"] = isa if isa is not None else isa_profile
    identity.pop("isa_profile", None)
    if not isinstance(identity["isa"], str) or not is_canonical_isa_profile(
        identity["isa"].lower()
    ):
        return ProfileResult(None, "profile-gap:isa-profile-invalid")
    identity["isa"] = identity["isa"].lower()
    xlen = 32 if identity["isa"].lower().startswith("rv32") else 64
    pc_limit = 1 << xlen
    if any(pc >= pc_limit for pc in pcs):
        return ProfileResult(None, "profile-gap:dynamic-trace-invalid")
    entry_fixed, exit_fixed = "entry" in identity, "exit" in identity
    exit_declared = exit_fixed or _profile_get(observation, "checkpoint_pc") is not None
    observed_checkpoint = _profile_get(observation, "checkpoint_pc")
    checkpoint_value = _pc_value(observed_checkpoint)
    if observed_checkpoint is not None and (
        checkpoint_value is None or checkpoint_value >= pc_limit
    ):
        return ProfileResult(None, "profile-gap:checkpoint-pc-invalid")
    for name in ("fault_pc", "fault_address"):
        value = _profile_get(observation, name)
        if value is not None:
            parsed = _pc_value(value)
            if parsed is None or parsed >= pc_limit or name == "fault_pc" and parsed & 1:
                return ProfileResult(None, "profile-gap:fault-address-invalid")
    if any(
        name in identity
        and ((_pc_value(identity[name]) is None) or _pc_value(identity[name]) >= pc_limit)
        for name in ("entry", "exit")
    ):
        return ProfileResult(None, "profile-gap:input-identity-missing")
    identity.setdefault("entry", pcs[0])
    identity.setdefault("exit", pcs[-1] if checkpoint_value is None else checkpoint_value)
    if not identity.get("isa") or any(identity.get(name) in (None, "") for name in ("entry", "exit")):
        return ProfileResult(None, "profile-gap:input-identity-missing")
    if identity.get("input_id") not in (None, "") and _profile_get(
        observation, "input_id"
    ) != identity["input_id"]:
        return ProfileResult(None, "profile-gap:input-identity-mismatch")
    if ("outcome" not in observation if isinstance(observation, Mapping)
            else not hasattr(observation, "outcome")):
        return ProfileResult(None, "observer-gap:missing-fields")
    outcome = _profile_get(observation, "outcome")
    contract_error = _profile_get(observation, "contract_error")
    if contract_error is not None:
        return ProfileResult(None, "profile-gap:trusted-reference-contract-error")
    extra_state = _profile_get(observation, "extra_state")
    if _profile_get(observation, "backend") == "sail-riscv" \
            and isinstance(extra_state, Mapping) \
            and isinstance(path := extra_state.get("program_test_path"), Mapping) \
            and path.get("status") != "observed":
        return ProfileResult(None, "profile-gap:sail-program-test-path")
    sail_trap = canonical_trap_state(dict(extra_state)) if (
        expected_trap and _profile_get(observation, "backend") == "sail-riscv"
        and isinstance(extra_state, Mapping)
    ) else {}
    signal_observed = isinstance(signal := _profile_get(observation, "signal"), str) and bool(signal.strip())
    trap_observed = (
        isinstance(outcome, str)
        and outcome in {"trap", "nonzero-exit"}
        and (
            signal_observed
            or isinstance(extra_state, Mapping)
            and (
                guest_trap_observed(dict(extra_state))
                or type(sail_trap.get("trap.cause")) is int
                and 0 <= sail_trap["trap.cause"] < 64
                and type(sail_trap.get("trap.epc")) is int
                and 0 <= sail_trap["trap.epc"] < pc_limit
                and not sail_trap["trap.epc"] & 1
                and type(sail_trap.get("trap.tval")) is int
                and 0 <= sail_trap["trap.tval"] < pc_limit
            )
        )
    )
    natural_terminal = (
        isinstance(outcome, str)
        and outcome in {"trap", "nonzero-exit"}
        and bool(pcs)
        and contract_error is None
    )
    trap_observed = trap_observed or natural_terminal
    if not isinstance(outcome, str) or (outcome not in {"normal", "completed"} and not trap_observed):
        return ProfileResult(None, "profile-gap:trusted-reference-not-normal")
    if checkpoint_value == 0:
        return ProfileResult(None, "profile-gap:checkpoint-pc-invalid")
    exit_code = _profile_get(observation, "exit_code")
    if exit_code is not None and type(exit_code) is not int:
        return ProfileResult(None, "profile-gap:trusted-reference-exit-code")
    if exit_code not in (None, 0) and not trap_observed:
        return ProfileResult(None, "profile-gap:trusted-reference-exit-code")
    has_gpr = "gpr" in observation if isinstance(observation, Mapping) else hasattr(observation, "gpr")
    if has_gpr and _gpr_state(_profile_get(observation, "gpr"), xlen) is None and not trap_observed:
        return ProfileResult(None, "profile-gap:trusted-reference-gpr-state")
    for name in ("memory_snapshot", "memory_delta"):
        if ((name in observation) if isinstance(observation, Mapping) else hasattr(observation, name)) \
                and not _profile_memory_valid(_profile_get(observation, name)):
            return ProfileResult(None, f"profile-gap:trusted-reference-{name}")
    memory_digest = _profile_get(observation, "memory_digest")
    if memory_digest is not None and not is_sha256_digest(memory_digest):
        return ProfileResult(None, "profile-gap:trusted-reference-memory-digest")
    artifact_hashes = tuple(
        _profile_get(observation, name)
        for name in ("binary_sha256", "guest_elf_sha256")
        if _profile_get(observation, name) is not None
    )
    if any(not is_sha256_digest(value) for value in artifact_hashes):
        return ProfileResult(None, "profile-gap:trusted-reference-artifact-identity")
    if len(set(artifact_hashes)) > 1:
        return ProfileResult(None, "profile-gap:trusted-reference-artifact-mismatch")
    if isinstance(observation, Mapping) and "extra_state" in observation \
            or not isinstance(observation, Mapping) and hasattr(observation, "extra_state"):
        extra_state = _profile_get(observation, "extra_state")
        if not isinstance(extra_state, Mapping) or any(
            not isinstance(key, str) for key in extra_state
        ):
            return ProfileResult(None, "profile-gap:trusted-reference-extra-state")
        for name in ("guest_epc", "guest_tval"):
            if name in extra_state and extra_state[name] is not None:
                value = _pc_value(extra_state[name])
                if value is None or value >= pc_limit or name == "guest_epc" and value & 1:
                    return ProfileResult(None, "profile-gap:trusted-reference-trap-state")
    instruction_count = _profile_get(observation, "instruction_count")
    if instruction_count is not None and (
        type(instruction_count) is not int
        or instruction_count < 0
        or instruction_count != len(pcs)
    ):
        return ProfileResult(None, "profile-gap:trusted-reference-instruction-count")
    trace_complete = _profile_get(observation, "trace_complete")
    if trace_complete is not None and type(trace_complete) is not bool:
        return ProfileResult(None, "profile-gap:trusted-reference-trace-completeness")
    execution_trace_complete = trace_complete is True or (
        trace_complete is None
        and instruction_count is not None
        and instruction_count == len(pcs)
    )
    program_path = None
    if isinstance(extra_state, Mapping):
        marker = extra_state.get("program_test_path")
        if identity.get("harness") == "riscv-dv" and isinstance(marker, Mapping):
            start, end = marker.get("start_pc"), marker.get("end_pc")
            body = tuple(pc for pc in pcs if type(start) is int and type(end) is int and start <= pc < end)
            declared = tuple(_pc_value(item) for item in marker.get("executed_pcs", ()))
            if (
                marker.get("contract") != "program-test-path-v1"
                or marker.get("status") != "observed"
                or type(start) is not int or type(end) is not int or not start < end
                or not body or body != declared
            ):
                return ProfileResult(None, "profile-gap:program-test-path-invalid")
            program_path = marker
            pcs = body
            identity["entry"], identity["exit"] = body[0], end
    has_gpr = has_gpr and _gpr_state(_profile_get(observation, "gpr"), xlen) is not None
    try:
        source = _source(program)
    except (OSError, TypeError, ValueError):
        return ProfileResult(None, "profile-gap:source-mapping-missing")
    generation = getattr(program, "provenance", None)
    if isinstance(generation, Mapping) and generation.get("program_sha256") not in (
        None, program_sha256(program)
    ):
        return ProfileResult(None, "profile-gap:generation-identity-mismatch")
    source = _strip_c_comments(source)
    if identity.get("harness") == "custom":
        source = _expand_asm_macros(source)
    raw_source_lines = source.splitlines(keepends=True)
    source_lines = []
    index = 0
    while index < len(raw_source_lines):
        parts = [raw_source_lines[index]]
        continuations = 0
        while parts[-1].rstrip("\r\n").endswith("\\") and index + 1 < len(raw_source_lines):
            ending = "\r\n" if parts[-1].endswith("\r\n") else "\n" if parts[-1].endswith("\n") else ""
            parts[-1] = parts[-1][:-len(ending) - 1] + "  "
            index += 1
            parts.append(raw_source_lines[index])
            continuations += 1
        source_lines.append("".join(parts))
        source_lines.extend("" for _ in range(continuations))
        index += 1
    logical_lines = _logical_source_lines(source_lines)
    source_offsets = []
    source_offset = 0
    for raw in source_lines:
        source_offsets.append(source_offset)
        source_offset += len(raw)
    declared_isa = _profile_declared_arch(source_lines)
    if declared_isa is not None:
        if not is_canonical_isa_profile(declared_isa):
            return ProfileResult(None, "profile-gap:isa-option-invalid")
        if declared_isa[:4] != identity["isa"][:4] \
                or not set(enabled_extensions(declared_isa)) <= set(enabled_extensions(identity["isa"])):
            return ProfileResult(None, "profile-gap:isa-profile-mismatch")
    profile_lines = tuple(
        (line, part)
        for line, raw in enumerate(source_lines)
        for part in _split_source_statements(raw)
    )
    # The line table may attach the materialized subprogram tail to the wrong
    # source line.  For the RVDV program-test path, use the source parser for
    # row ownership and retain only the disassembler's instruction map as the
    # final-layout witness.
    raw_source_pc_map = _profile_source_map(observation)
    source_pc_map = _profile_source_pc_map(observation)
    if raw_source_pc_map is not None and source_pc_map == {}:
        # RVDV's disassembler witness can be instruction-only.  The final
        # instruction map plus the source parser is enough to build the
        # profile; an absent line table is not an execution mismatch.
        source_pc_map = None if _profile_final_instruction_map(observation) else {}
        if source_pc_map == {}:
            return ProfileResult(None, "profile-gap:source-mapping-invalid")
    final_instruction_map = _profile_final_instruction_map(observation)
    if final_instruction_map == {}:
        return ProfileResult(None, "profile-gap:final-instruction-map-invalid")
    try:
        compressed_hint = profile_uses_compressed_encoding(str(identity["isa"])) or any(
            _profile_arch_options(_strip_comments(raw).strip(), str(identity["isa"]), False)[1]
            for raw in source.splitlines()
        )
    except ValueError:
        return ProfileResult(None, "profile-gap:isa-option-invalid")
    pc_alignment = 2 if compressed_hint else 4
    if (re.search(r'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"user_init\.s"', source)
            and source_pc_map is None):
        return ProfileResult(None, "profile-gap:user-init-expansion")
    source_path = getattr(program, "program", None)
    include_context = tuple(getattr(program, "_include_context", ()) or ())
    if source_path is None and not include_context and re.search(
            r'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"[^"]+"', source):
        return ProfileResult(None, "profile-gap:include-source-missing")
    if source_pc_map is not None and any(
        pc >= pc_limit for values in source_pc_map.values() for pc in values
    ):
        return ProfileResult(None, "profile-gap:dynamic-trace-invalid")
    if final_instruction_map is not None and any(
        pc >= pc_limit for pc in final_instruction_map
    ):
        return ProfileResult(None, "profile-gap:dynamic-trace-invalid")
    source_map_pcs = sorted({pc for values in source_pc_map.values() for pc in values}) if source_pc_map is not None else ()
    if identity.get("harness") == "riscv-dv" and source_pc_map is not None:
        # Keep the exact source map; drop only observed harness PCs with no source row.
        mapped = set(source_map_pcs)
        pcs = tuple(pc for pc in pcs if pc in mapped)
    if identity.get("harness") == "custom" and source_pc_map is not None:
        source_pc_map = dict(source_pc_map)
        source_map_pcs = sorted({pc for values in source_pc_map.values() for pc in values})
        mapped_source_pcs = set(source_map_pcs)
        prefix_rows = []
        if final_instruction_map is not None and source_map_pcs:
            first_source_line = min(source_pc_map)
            for line, raw in profile_lines:
                logical = logical_lines[line]
                if logical is None or logical >= first_source_line:
                    continue
                body = _strip_comments(raw).strip()
                while match := re.match(r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body):
                    body = match[2].strip()
                words = body.lower().split()
                width = _DATA_DIRECTIVE_WIDTHS.get(words[0]) if words else None
                if width is not None and len(words) > 1:
                    prefix_rows.append((logical, width * len(_split_operands(body.split(None, 1)[1]))))
            prefix_size = sum(size for _line, size in prefix_rows)
            prefix_start = source_map_pcs[0] - prefix_size
            prefix_items = [
                (pc, item) for pc, item in sorted(final_instruction_map.items())
                if prefix_start <= pc < source_map_pcs[0]
            ]
            cursor = prefix_start
            prefix_map = {}
            valid_prefix = prefix_size > 0 and prefix_items and prefix_items[0][0] == prefix_start
            for line, size in prefix_rows:
                row_pcs = []
                consumed = 0
                while valid_prefix and consumed < size:
                    if not prefix_items or prefix_items[0][0] != cursor:
                        valid_prefix = False
                        break
                    pc, item = prefix_items.pop(0)
                    row_pcs.append(pc)
                    consumed += item["size"]
                    cursor += item["size"]
                if not valid_prefix or consumed != size:
                    valid_prefix = False
                    break
                prefix_map[line] = (*prefix_map.get(line, ()), *row_pcs)
            if valid_prefix and not prefix_items and cursor == source_map_pcs[0]:
                prefix_map = {
                    line: tuple(pcs) for line, pcs in prefix_map.items()
                    if not set(pcs) & mapped_source_pcs
                }
                source_pc_map.update(prefix_map)
                source_map_pcs = sorted({pc for values in source_pc_map.values() for pc in values})
        custom_pcs = {pc for values in source_pc_map.values() for pc in values}
        pcs = tuple(pc for pc in pcs if pc in custom_pcs)
        if not pcs:
            return ProfileResult(None, "profile-gap:custom-source-map-empty")
        if prefix_rows and pcs[0] not in mapped_source_pcs:
            identity["entry"], identity["exit"] = pcs[0], pcs[-1]
        elif pcs[0] not in mapped_source_pcs:
            identity["entry"] = pcs[0]
        checkpoint_value = None
    observed_terminal = _profile_get(observation, "checkpoint_pc")
    terminal_pc = _pc_value(observed_terminal) if observed_terminal is not None else (
        _pc_value(identity.get("exit")) if exit_declared else None
    )
    terminal_boundary = (
        exit_declared and terminal_pc is not None and pcs[-1] == terminal_pc
        and final_instruction_map is not None and terminal_pc not in final_instruction_map
        and any(pc + item["size"] == terminal_pc for pc, item in final_instruction_map.items())
    )
    missing_source_pcs = tuple(pc for pc in pcs if pc not in source_map_pcs)
    if source_pc_map is not None and missing_source_pcs and not (
        terminal_boundary and missing_source_pcs == (terminal_pc,)
    ):
        return ProfileResult(None, "profile-gap:source-mapping-invalid")
    if final_instruction_map is not None and (
            any(pc not in final_instruction_map for pc in source_map_pcs)
            or any(
                pc not in final_instruction_map
                and not (terminal_boundary and pc == terminal_pc)
                for pc in pcs
            )
    ):
        return ProfileResult(None, "profile-gap:source-mapping-invalid")
    next_map_pc = dict(zip(source_map_pcs, source_map_pcs[1:]))
    profile_start = 0
    checkpoint_pcs: set[int] = set()
    if identity.get("harness") == "riscv-dv":
        def has_label(raw: str, names: set[str]) -> bool:
            for statement in _split_source_statements(raw):
                body = _strip_comments(statement).strip()
                while match := re.match(
                    r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body
                ):
                    if match[1] in names:
                        return True
                    body = match[2].strip()
            return False

        start = next((i for i, raw in enumerate(source_lines)
                      if has_label(raw, {"init"})), None)
        if start is None:
            start = next((i for i, raw in enumerate(source_lines)
                          if has_label(raw, {"h0_start"})), None)
        if program_path is not None:
            # The marker starts before the generated setup instruction(s),
            # not necessarily at ``main``.  Keep the source cursor anchored to
            # the marker label so its first dynamic PC is not assigned to the
            # following source instruction.
            marker_start = next(
                (
                    i for i, raw in enumerate(source_lines)
                    if has_label(raw, {"rvgen_program_body_start"})
                ),
                None,
            )
            if marker_start is not None:
                start = marker_start
            else:
                # build_linux_user places the marker before its setup slice.
                # For the original RVDV source, mirror that slice boundary:
                # an exact ``main:`` starts it; an inline main label means the
                # stack setup line immediately before it is the real boundary.
                exact_main = next(
                    (i for i, raw in enumerate(source_lines) if raw.strip() == "main:"),
                    None,
                )
                if exact_main is not None:
                    start = exact_main
                else:
                    stack_line = next(
                        (
                            i for i, raw in enumerate(source_lines[start:], start)
                            if re.match(r"^\s*la\s+(?:x[0-9]+|sp)\s*,\s*user_stack_end\s*$", raw)
                        ),
                        None,
                    )
                    if stack_line is not None:
                        start = stack_line + 1
        end = next((i for i, raw in enumerate(source_lines[start:], start)
                    if has_label(raw, {"test_done"})), None) if start is not None else None
        checkpoint = _pc_value(_profile_get(observation, "checkpoint_pc"))
        if program_path is not None:
            checkpoint = _pc_value(program_path.get("end_pc"))
        checkpoint_index = (
            next((i for i, pc in enumerate(pcs) if pc == checkpoint), len(pcs))
            if checkpoint is not None else None
        )
        if start is not None and end is not None:
            profile_ranges = [(start, end + 1)]
            if program_path is not None:
                subprogram_start = next(
                    (
                        i for i, raw in enumerate(source_lines[end + 1:], end + 1)
                        if any(
                            re.match(r"^sub_[0-9]+\s*:", _strip_comments(statement).strip())
                            for statement in _split_source_statements(raw)
                        )
                    ),
                    None,
                )
                tohost = next(
                    (
                        i for i, raw in enumerate(source_lines[end + 1:], end + 1)
                        if has_label(raw, {"write_tohost"})
                    ),
                    None,
                )
                if (
                    subprogram_start is not None
                    and tohost is not None
                    and subprogram_start < tohost
                ):
                    # linux_user_harness materializes every sub_N routine after
                    # test_done and before the body-end marker. Keep that same
                    # source range in the profile so direct calls resolve.
                    profile_ranges.append((subprogram_start, tohost))
            if (source_pc_map is not None and end + 1 < len(source_lines)
                    and source_pc_map.get(logical_lines[end + 1] or end + 2)):
                profile_ranges[0] = (start, end + 2)
            if source_pc_map is not None:
                body_starts = sorted({
                    pc
                    for range_start, range_end in profile_ranges
                    for line in range(range_start, range_end)
                    for logical in (logical_lines[line] or line + 1,)
                    for pc in source_pc_map.get(logical, ())
                })
                entry_body = _strip_comments(source_lines[start]).strip()
                while match := re.match(r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", entry_body):
                    entry_body = match[2].strip()
                if not entry_body and has_label(source_lines[start], {"h0_start"}):
                    body_starts = [
                        pc for pc in body_starts
                        if pc not in source_pc_map.get(logical_lines[start] or start + 1, ())
                    ]
                pcs = tuple(
                    pc for pc in pcs
                    if any(
                        left <= pc < next_map_pc.get(
                            left,
                            checkpoint if checkpoint is not None else (
                                left + final_instruction_map[left]["size"]
                                if final_instruction_map is not None
                                and left in final_instruction_map else left + (
                                    2 if profile_uses_compressed_encoding(str(identity["isa"])) else 4
                                )
                            ),
                        )
                        for left in body_starts
                    )
                )
                if not pcs:
                    return ProfileResult(None, "profile-gap:executed-pc-unmapped")
            else:
                if program_path is None:
                    pcs = pcs[2:checkpoint_index]
                if not pcs:
                    return ProfileResult(None, "profile-gap:executed-pc-unmapped")
            if pcs:
                identity["entry"], identity["exit"] = pcs[0], pcs[-1]
                profile_lines = []
                for range_start, range_end in profile_ranges:
                    for line, raw in enumerate(source_lines[range_start:range_end], range_start):
                        for part in _split_source_statements(raw):
                            if line == end and has_label(part, {"test_done"}):
                                # 保留终止标签占用的源码长度；否则后续 li 等指令的
                                # source span 会整体左移，动作池会被误判为不可用。
                                part = re.sub(r"[^\r\n]", " ", part)
                            profile_lines.append((line, part))
                profile_lines = tuple(profile_lines)
                profile_start = start
    observed_checkpoint = _profile_get(observation, "checkpoint_pc")
    checkpoint = checkpoint_value
    trace_entries = _trace_entries(observation)
    if any(item.get("_invalid") for item in trace_entries):
        return ProfileResult(None, "profile-gap:state-trace-invalid")
    state_limit = 1 << xlen
    for item in trace_entries:
        invalid_fields = set(item.get("_invalid_trace_fields", ()))
        for name in ("before_gpr", "after_gpr"):
            if name in item and _gpr_state(item[name], xlen) is None:
                item.pop(name)
                invalid_fields.add(name)
        if invalid_fields:
            item["_invalid_trace_fields"] = tuple(sorted(invalid_fields))
    if any(
        any(
            name in item and item[name] is not None
            and name in {"pc", "after_pc", "fault_pc", "fault_address"}
            and (_pc_value(item[name]) is None or _pc_value(item[name]) >= state_limit)
            for name in ("before_gpr", "after_gpr", "pc", "after_pc", "fault_pc", "fault_address")
        )
        for item in trace_entries
    ):
        return ProfileResult(None, "profile-gap:state-trace-xlen")
    trace_pcs = tuple(item["pc"] for item in trace_entries)
    if identity.get("harness") == "riscv-dv" and trace_entries and trace_pcs != pcs:
        projected, cursor = [], 0
        for item in trace_entries:
            if cursor < len(pcs) and item["pc"] == pcs[cursor]:
                projected.append(item)
                cursor += 1
        if cursor == len(pcs):
            trace_entries, trace_pcs = tuple(projected), pcs
    terminal_trace_prefix = (
        bool(trace_entries) and len(pcs) > 1 and trace_pcs == pcs[:-1]
        and exit_declared and pcs[-1] == _pc_value(identity.get("exit"))
    )
    if trace_entries and trace_pcs != pcs and not terminal_trace_prefix:
        trace_entries = ()
    elif trace_pcs == pcs or terminal_trace_prefix:
        expected_after = pcs[1:]
        if trace_entries[-1].get("after_pc") is not None:
            terminal_pc = checkpoint if checkpoint is not None else (
                _pc_value(identity["exit"]) if exit_fixed else None
            )
            if terminal_pc is not None:
                expected_after += (terminal_pc,)
            elif (
                natural_terminal
                and trace_entries[-1].get("after_pc") == trace_entries[-1].get("pc")
            ):
                # A native ptrace trace ends with the faulting instruction
                # pointing at itself; keep its state witness for the selected
                # program path instead of discarding the whole trace.
                expected_after += (trace_entries[-1]["after_pc"],)
        if any(
            item.get("after_pc") is not None
            and (index >= len(expected_after) or item["after_pc"] != expected_after[index])
            for index, item in enumerate(trace_entries)
        ):
            trace_entries = ()
    before_by_pc = {
        item["pc"]: value
        for item in trace_entries
        if (value := _gpr_state(item.get("before_gpr"), xlen)) is not None
    }
    entry = _pc_value(identity["entry"])
    exit_pc = _pc_value(identity["exit"])
    if entry is None or exit_pc is None:
        return ProfileResult(None, "profile-gap:input-identity-missing")
    identity["entry"], identity["exit"] = entry, exit_pc
    if entry != pcs[0]:
        return ProfileResult(None, "profile-gap:entry-pc-mismatch")
    if identity.get("harness") != "riscv-dv" and checkpoint is None and exit_fixed \
            and exit_pc != pcs[-1]:
        trace_details = _profile_get(
            _profile_get(observation, "translation_evidence"), "details", {}
        )
        if _profile_get(trace_details, "terminal_ebreak") is not True:
            return ProfileResult(None, "profile-gap:checkpoint-pc-mismatch")
    if identity.get("harness") != "riscv-dv" and checkpoint is not None and exit_pc != checkpoint:
        trace_details = _profile_get(
            _profile_get(observation, "translation_evidence"), "details", {}
        )
        if _profile_get(trace_details, "terminal_ebreak") is not True:
            return ProfileResult(None, "profile-gap:checkpoint-pc-mismatch")
    compressed = profile_uses_compressed_encoding(str(identity["isa"]))
    cursor = sum(len(raw) for raw in source_lines[:profile_start])
    initial_pc = entry if source_pc_map is not None else (
        _pc_value(program_path.get("start_pc"))
        if isinstance(program_path, Mapping) else 0
    )
    prefix_layout = False
    aliases: dict[str, str] = _collect_equ_aliases(raw for _, raw in profile_lines)
    if not initial_pc:
        pending_entry_label = False
        for raw in source_lines[profile_start:]:
            body = _strip_comments(raw).strip()
            labels = []
            while match := re.match(r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body):
                labels.append(match[1])
                body = match[2].strip()
            pending_entry_label |= bool(set(labels) & {"_start", "main", "init", "h0_start"})
            if not body or body.startswith((".", "#")):
                parts = body.lower().split()
                if parts and parts[0] in {
                    ".ascii", ".asciz", ".string", ".balign", ".balignw", ".balignl",
                    ".align", ".p2align", ".p2alignw", ".p2alignl", ".space", ".zero", ".skip", ".fill",
                    ".incbin", ".org", ".float", ".single", ".double", ".uleb128", ".sleb128",
                    *set(_DATA_DIRECTIVE_WIDTHS),
                }:
                    operation = parts[0]
                    operands = _split_operands(body.split(None, 1)[1]) if len(parts) > 1 else ()
                    zero_layout = operation in {
                        ".balign", ".balignw", ".balignl", ".align", ".p2align",
                        ".p2alignw", ".p2alignl",
                    } or operation in {".byte", ".2byte", ".half", ".short", ".word", ".4byte",
                                        ".long", ".dword", ".8byte", ".quad", ".float", ".single",
                                        ".double", ".uleb128", ".sleb128"} and not operands
                    if operation in {".space", ".zero", ".skip", ".org"} and operands:
                        zero_layout = _layout_int(operands[0], aliases) == 0
                    elif operation == ".fill" and operands:
                        zero_layout = _layout_int(operands[0], aliases) == 0
                    elif operation in {".ascii", ".asciz", ".string"}:
                        zero_layout = _string_size(body) == 0
                    elif operation == ".incbin":
                        zero_layout = _incbin_size(
                            source_path, operands, aliases, include_context,
                        ) == 0
                    prefix_layout |= not zero_layout
                if pending_entry_label and parts and parts[0] in (
                    set(_DATA_DIRECTIVE_WIDTHS)
                    | {".ascii", ".asciz", ".string", ".balign", ".balignw", ".balignl",
                       ".align", ".p2align", ".p2alignw", ".p2alignl", ".space", ".zero", ".skip", ".fill", ".incbin",
                       ".org", ".float", ".single", ".double", ".uleb128", ".sleb128"}
                ):
                    pending_entry_label = False
                continue
            if pending_entry_label and not prefix_layout:
                initial_pc = entry
            break
    if not initial_pc and entry_fixed and not prefix_layout:
        initial_pc = entry
    rows, pc, block, block_label, pending_labels = [], initial_pc, "b0", None, []
    label_points: dict[tuple[str, str], list[tuple[int, int]]] = {}
    named_labels: set[str] = set()
    active_isa = str(identity["isa"])
    base_xlen = xlen
    compressed_stack, isa_stack, section_stack = [], [], []
    section_gap = None
    previous_text = None
    previous_section = None
    section_name = ".text"
    in_text = True
    include_code = False
    include_gap = None
    ended = False
    data_pcs: dict[str, int] = {}

    def active_isa_valid() -> bool:
        return xlen == base_xlen

    def source_body(raw: str) -> str:
        value = raw.strip()
        return value if re.match(
            r"^(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*#\s*include\b",
            value, re.IGNORECASE,
        ) else _strip_comments(raw).strip()

    def section_value(body: str) -> str | None:
        parts = body.split(None, 1)
        if len(parts) < 2:
            return None
        values = _split_operands(parts[1])
        return values[0].strip().strip('"') if values else None

    def advance_section(body: str) -> bool:
        nonlocal in_text, previous_text, previous_section, section_name
        nonlocal block, block_label, section_gap
        changed = False
        words = body.lower().split()
        if words and words[0] == ".pushsection":
            name = section_value(body)
            if name is None:
                section_gap = "profile-gap:section-invalid"
                return True
            section_stack.append((in_text, previous_text, section_name, previous_section))
            previous_text, previous_section = in_text, section_name
            section_name = name
            flags = re.search(r',\s*"([^"\n]*)"', body, re.IGNORECASE)
            in_text = "x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", body, re.IGNORECASE))
            if in_text and not (section_name == ".text" or section_name.startswith(".text.")):
                section_gap = "profile-gap:executable-section-layout"
            changed = True
        elif words and words[0] == ".popsection":
            if len(words) != 1 or not section_stack:
                section_gap = "profile-gap:section-invalid"
                return True
            in_text, previous_text, section_name, previous_section = section_stack.pop()
            changed = True
        elif words and words[0] in {".text", ".section"}:
            name = ".text" if words[0] == ".text" else section_value(body)
            if name is None:
                section_gap = "profile-gap:section-invalid"
                return True
            previous_text, previous_section = in_text, section_name
            section_name = name
            flags = re.search(r',\s*"([^"\n]*)"', body, re.IGNORECASE)
            in_text = words[0] == ".text" or ("x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", body, re.IGNORECASE)))
            if in_text and not (section_name == ".text" or section_name.startswith(".text.")):
                section_gap = "profile-gap:executable-section-layout"
            changed = True
        elif words and words[0] in {".data", ".bss", ".rodata", ".sdata", ".sbss", ".tdata", ".tbss"}:
            previous_text, previous_section = in_text, section_name
            section_name, in_text = words[0], False
            changed = True
        elif words and words[0] == ".previous":
            if previous_text is None or previous_section is None:
                section_gap = "profile-gap:section-invalid"
                return True
            in_text, previous_text = previous_text, in_text
            section_name, previous_section = previous_section, section_name
            changed = True
        else:
            return False
        if changed:
            block = f"b{len({row['block'] for row in rows})}"
            block_label = None
        return True

    def scan_include(
        body: str, base: Path | None = None, stack: tuple[Path, ...] = (),
    ) -> None:
        nonlocal active_isa, compressed, compressed_hint, include_code, include_gap, xlen, ended, block, block_label
        match = re.fullmatch(
            r'(?:\.include|#\s*include)\s+"([^"]+)"[ \t]*(?:#.*|//.*)?',
            body, re.IGNORECASE,
        )
        if match is None:
            return
        if source_path is None:
            dependency = Path(posixpath.normpath(((base or Path(".")) / match[1]).as_posix()))
            if dependency in stack:
                include_gap = "profile-gap:include-source-missing"
                return
            payload = next(
                (data for name, data in include_context
                 if Path(name).as_posix() == dependency.as_posix()),
                None,
            )
            if payload is None:
                include_gap = "profile-gap:include-source-missing"
                return
            try:
                included_source = payload.decode("utf-8")
            except UnicodeError:
                include_gap = "profile-gap:include-source-invalid"
                return
        else:
            dependency = ((base or Path(source_path).parent) / match[1]).resolve()
            if not dependency.is_file() or dependency in stack:
                include_gap = "profile-gap:include-source-missing"
                return
            try:
                included_source = dependency.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                include_gap = "profile-gap:include-source-invalid"
                return
        for included_line in _joined_source_lines(_strip_c_comments(included_source)):
            if ended:
                return
            for included_part in _split_source_statements(included_line):
                if ended:
                    return
                part = source_body(included_part)
                included_labels = []
                while match := re.match(
                    r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", part
                ):
                    included_labels.append(match[1])
                    part = match[2].strip()
                if in_text and included_labels:
                    for label in included_labels:
                        key = (section_name, label)
                        if not label.isdigit() and key in label_points:
                            include_gap = "profile-gap:label-duplicate"
                            return
                        if not label.isdigit():
                            named_labels.add(label)
                        label_points.setdefault(key, []).append((cursor, pc))
                    if not pending_labels:
                        if rows:
                            block = f"b{len({row['block'] for row in rows})}"
                        block_label = included_labels[0]
                    pending_labels.extend(included_labels)
                try:
                    active_isa, compressed = _profile_arch_options(part, active_isa, compressed)
                except ValueError:
                    include_gap = "profile-gap:isa-option-invalid"
                    return
                compressed_hint = compressed_hint or compressed
                words = part.lower().split()
                if words and words[0] == ".attribute" and re.match(
                    r"\.attribute\s+arch\b", part, re.IGNORECASE,
                ):
                    attribute_match = re.fullmatch(
                        r'\.attribute\s+arch\s*,\s*"([^"]+)"',
                        part, re.IGNORECASE,
                    )
                    if attribute_match is None or not is_canonical_isa_profile(
                        attribute_match[1].lower()
                    ):
                        include_gap = "profile-gap:isa-option-invalid"
                        return
                if words[:2] == [".option", "push"]:
                    compressed_stack.append(compressed)
                    isa_stack.append(active_isa)
                elif words[:2] == [".option", "pop"]:
                    if not compressed_stack:
                        include_gap = "profile-gap:option-invalid"
                        return
                    compressed = compressed_stack.pop()
                    active_isa = isa_stack.pop()
                xlen = 32 if active_isa.lower().startswith("rv32") else 64
                if not active_isa_valid():
                    include_gap = "profile-gap:isa-profile-mismatch"
                    return
                scan_include(part, dependency.parent, (*stack, dependency))
                if ended:
                    return
                section_changed = advance_section(part)
                if section_gap:
                    return
                words = part.lower().split()
                if not in_text and words and words[0] == ".incbin":
                    data_operands = _split_operands(
                        part.split(None, 1)[1]
                    ) if len(words) > 1 else ()
                    if _incbin_size(
                        source_path, data_operands, aliases, include_context, base=dependency.parent,
                    ) is None:
                        include_gap = "profile-gap:incbin-source-missing"
                        return
                if in_text and part and not section_changed and not re.fullmatch(
                    r'(?:\.include|#\s*include)\s+"[^"]+"[ \t]*(?:#.*|//.*)?',
                    part, re.IGNORECASE,
                ) and words[0] not in {
                    ".option", ".include", "#include", ".equ", ".set", ".globl",
                    ".global", ".type", ".size", ".attribute", ".file", ".loc", ".ident", ".end",
                }:
                    include_code = True
                if words and words[0] == ".end":
                    ended = True
                    return
                if words and words[0] in {".equ", ".set"}:
                    args = _split_operands(part.split(None, 1)[1]) if len(words) > 1 else ()
                    if len(args) >= 2:
                        aliases[args[0]] = args[1]

    for line, raw in profile_lines:
        # RVDV's profile ranges skip the original test-done epilogue before
        # entering materialized sub_N routines.  Re-anchor each row to its
        # source line so source spans do not slide across that gap.
        if 0 <= line < len(source_offsets):
            cursor = source_offsets[line]
        if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", raw, re.IGNORECASE):
            return ProfileResult(None, "profile-gap:conditional-unsupported")
        body = source_body(raw)
        line_labels = []
        while True:
            match = re.match(r"^\s*([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body)
            if match is None:
                break
            line_labels.append(match[1])
            body = match[2].strip()
        words = body.lower().split()
        if words and words[0] == ".end":
            break
        try:
            active_isa, compressed = _profile_arch_options(body, active_isa, compressed)
        except ValueError:
            return ProfileResult(None, "profile-gap:isa-option-invalid")
        if words and words[0] == ".attribute" and (
            match := re.fullmatch(r'\.attribute\s+arch\s*,\s*"([^"]+)"', body, re.IGNORECASE)
        ) is not None and not is_canonical_isa_profile(match[1].lower()):
            return ProfileResult(None, "profile-gap:isa-option-invalid")
        if words and words[0] == ".attribute" and re.match(
            r"\.attribute\s+arch\b", body, re.IGNORECASE,
        ) and re.fullmatch(r'\.attribute\s+arch\s*,\s*"[^"]+"', body, re.IGNORECASE) is None:
            return ProfileResult(None, "profile-gap:isa-option-invalid")
        if words[:2] == [".option", "push"]:
            compressed_stack.append(compressed)
            isa_stack.append(active_isa)
        elif words[:2] == [".option", "pop"]:
            if not compressed_stack:
                return ProfileResult(None, "profile-gap:option-invalid")
            compressed = compressed_stack.pop()
            active_isa = isa_stack.pop()
        xlen = 32 if active_isa.lower().startswith("rv32") else 64
        if not active_isa_valid():
            return ProfileResult(None, "profile-gap:isa-profile-mismatch")
        scan_include(body)
        if include_gap:
            return ProfileResult(None, include_gap)
        if include_code:
            return ProfileResult(None, "profile-gap:include-code-unresolved")
        if ended:
            break
        if re.fullmatch(
            r'(?:\.include|#\s*include)\s+"[^"]+"[ \t]*(?:#.*|//.*)?',
            body, re.IGNORECASE,
        ):
            cursor += len(raw)
            continue
        if advance_section(body):
            pending_labels = []
        if section_gap:
            return ProfileResult(None, section_gap)
        if in_text and words and words[0] in {
            ".subsection", ".rept", ".irp", ".irpc", ".endr",
            ".macro", ".endm", ".if", ".ifdef", ".ifndef", ".else",
            ".elseif", ".endif",
        }:
            return ProfileResult(None, "profile-gap:conditional-unsupported")
        if not in_text and body:
            data_op = words[0]
            data_operands = _split_operands(body.split(None, 1)[1]) if len(words) > 1 else ()
            data_pc = data_pcs.get(section_name, 0)
            if (
                not data_operands and (
                    data_op in _DATA_DIRECTIVE_WIDTHS
                    or data_op in _FLOAT_DIRECTIVE_WIDTHS
                    or data_op in {".ascii", ".asciz", ".string", ".insn", ".uleb128", ".sleb128"}
                )
            ):
                return ProfileResult(None, "profile-gap:data-directive-invalid")
            if data_op in _DATA_DIRECTIVE_WIDTHS and data_operands:
                size = _data_directive_size(_DATA_DIRECTIVE_WIDTHS[data_op], data_operands, aliases)
                if size is None:
                    return ProfileResult(None, "profile-gap:data-directive-invalid")
                data_pcs[section_name] = data_pc + size
            elif data_op == ".insn":
                size = _insn_size(data_operands, aliases)
                if size is None or len(words) > 1 and not _insn_operand_syntax_valid(words[1]):
                    return ProfileResult(None, "profile-gap:data-directive-invalid")
                data_pcs[section_name] = data_pc + size
            elif data_op in {".ascii", ".asciz", ".string"}:
                size = _string_size(body)
                if size is None:
                    return ProfileResult(None, "profile-gap:string-directive-invalid")
                data_pcs[section_name] = data_pc + size
            if data_op in _FLOAT_DIRECTIVE_WIDTHS and any(
                not _float_literal_valid(value)
                for value in data_operands
            ):
                return ProfileResult(None, "profile-gap:layout-directive-invalid")
            if data_op in {".uleb128", ".sleb128"} and _leb128_size(
                data_operands, data_op == ".sleb128", aliases,
            ) is None:
                return ProfileResult(None, "profile-gap:layout-directive-invalid")
            if data_op in _FLOAT_DIRECTIVE_WIDTHS:
                data_pcs[section_name] = data_pc + _FLOAT_DIRECTIVE_WIDTHS[data_op] * len(data_operands)
            if data_op in {".uleb128", ".sleb128"}:
                data_pcs[section_name] = data_pc + _leb128_size(
                    data_operands, data_op == ".sleb128", aliases,
                )
            if data_op in {
                ".balign", ".balignw", ".balignl", ".align", ".p2align",
                ".p2alignw", ".p2alignl",
            }:
                value = _layout_int(data_operands[0], aliases) if data_operands else None
                fill = data_operands[1].strip() if len(data_operands) > 1 else ""
                max_skip = data_operands[2].strip() if len(data_operands) > 2 else ""
                if (
                    not 1 <= len(data_operands) <= 3
                    or value is None or value < 0
                    or fill and _layout_int(fill, aliases) is None
                    or max_skip and (
                        _layout_int(max_skip, aliases) is None
                        or _layout_int(max_skip, aliases) < 0
                    )
                    or data_op.startswith(".balign") and value and value & (value - 1)
                ):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                alignment = value if data_op.startswith(".balign") else 1 << value
                alignment = alignment or 1
                padding = (-data_pc) % alignment
                limit = _layout_int(max_skip, aliases) if max_skip else None
                if limit is None or padding <= limit:
                    if (
                        data_op in {".balignw", ".p2alignw"} and padding % 2
                        or data_op in {".balignl", ".p2alignl"} and padding % 4
                    ):
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    data_pcs[section_name] = data_pc + padding
                else:
                    data_pcs[section_name] = data_pc
            if data_op in {".space", ".zero", ".skip"}:
                value = _layout_int(data_operands[0], aliases) if data_operands else None
                if (
                    len(data_operands) not in {1, 2}
                    or value is None or value < 0
                    or len(data_operands) == 2 and data_operands[1].strip()
                    and _layout_int(data_operands[1], aliases) is None
                ):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                data_pcs[section_name] = data_pc + value
            if data_op == ".fill":
                repeat = _layout_int(data_operands[0], aliases) if data_operands else None
                width = _layout_int(data_operands[1], aliases) if len(data_operands) > 1 and data_operands[1].strip() else 1
                fill = data_operands[2].strip() if len(data_operands) > 2 else ""
                if (
                    not 1 <= len(data_operands) <= 3
                    or repeat is None or width is None or repeat < 0 or width < 0
                    or fill and _layout_int(fill, aliases) is None
                ):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                data_pcs[section_name] = data_pc + repeat * width
            if data_op == ".org":
                address = _layout_int(data_operands[0], aliases) if data_operands else None
                if (
                    len(data_operands) not in {1, 2}
                    or address is None or address < data_pc
                    or len(data_operands) == 2 and data_operands[1].strip()
                    and _layout_int(data_operands[1], aliases) is None
                ):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                data_pcs[section_name] = address
            if data_op == ".incbin" and _incbin_size(
                source_path, data_operands, aliases, include_context,
            ) is None:
                return ProfileResult(None, "profile-gap:incbin-source-missing")
            if data_op == ".incbin":
                data_pcs[section_name] = data_pcs.get(section_name, 0) + _incbin_size(
                    source_path, data_operands, aliases, include_context,
                )
        for label in line_labels:
            if not label.isdigit() and label in named_labels:
                return ProfileResult(None, "profile-gap:label-duplicate")
            if not label.isdigit():
                named_labels.add(label)
        if not in_text:
            cursor += len(raw)
            continue
        for label in line_labels:
            label_points.setdefault((section_name, label), []).append((cursor, pc))
            if not pending_labels and rows:
                block = f"b{len({row['block'] for row in rows})}"
                block_label = label
            elif not rows and block_label is None:
                block_label = label
            pending_labels.append(label)
        parts = body.lower().split()
        string_directive = parts and parts[0] in {".ascii", ".asciz", ".string"}
        string_size = _string_size(body) if string_directive else None
        directive_width = _DATA_DIRECTIVE_WIDTHS.get(parts[0]) if parts else None
        directive_args = _split_operands(body.split(None, 1)[1]) if len(parts) > 1 else ()
        if (
            parts and not directive_args and (
                directive_width is not None
                or parts[0] in _FLOAT_DIRECTIVE_WIDTHS
                or parts[0] in {".ascii", ".asciz", ".string", ".insn", ".uleb128", ".sleb128"}
            )
        ):
            return ProfileResult(None, "profile-gap:data-directive-invalid")
        if parts and not directive_args and parts[0] in {
            ".balign", ".balignw", ".balignl", ".align",
            ".p2align", ".p2alignw", ".p2alignl",
        }:
            return ProfileResult(None, "profile-gap:layout-directive-invalid")
        code_directive = directive_width is not None and bool(directive_args) \
            or bool(parts and parts[0] == ".insn")
        if parts and parts[0] == ".insn" and len(parts) > 1 \
                and not _insn_operand_syntax_valid(body.split(None, 1)[1]):
            return ProfileResult(None, "profile-gap:data-directive-invalid")
        if (parts and parts[0].startswith(".")
                and parts[0] not in _KNOWN_PROFILE_DIRECTIVES
                and not parts[0].startswith(_PROFILE_DIRECTIVE_PREFIXES)
                and not parts[0].startswith(".text.")):
            return ProfileResult(None, "profile-gap:directive-unknown")
        if not body or (body.startswith(".") and not code_directive):
            if parts and parts[0] in {".equ", ".set"} and len(directive_args) >= 2:
                aliases[directive_args[0]] = directive_args[1]
            if string_directive:
                if string_size is None:
                    return ProfileResult(None, "profile-gap:string-directive-invalid")
                pc += string_size
            elif parts and parts[0] in {
                ".balign", ".balignw", ".balignl", ".align", ".p2align", ".p2alignw", ".p2alignl",
            }:
                directive_args = tuple(
                    item.strip() for item in body.split(None, 1)[1].split(",")
                ) if len(parts) > 1 else ()
                alignment = _layout_int(directive_args[0], aliases) if directive_args else None
                if not directive_args:
                    cursor += len(raw)
                    continue
                if len(directive_args) > 3 or alignment is None or alignment < 0:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if parts[0].startswith(".balign") and alignment and alignment & (alignment - 1):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if len(directive_args) > 1 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if alignment == 0:
                    alignment = 1
                elif parts[0] in {".align", ".p2align", ".p2alignw", ".p2alignl"}:
                    alignment = 1 << alignment
                padding = (-pc) % alignment
                max_skip = _layout_int(directive_args[2], aliases) if len(directive_args) > 2 and directive_args[2].strip() else None
                if len(directive_args) > 2 and directive_args[2].strip() and (max_skip is None or max_skip < 0):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if max_skip is None or padding <= max_skip:
                    if (
                        parts[0] in {".balignw", ".p2alignw"} and padding % 2
                        or parts[0] in {".balignl", ".p2alignl"} and padding % 4
                    ):
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    pc += padding
            elif parts and parts[0] in {".space", ".zero", ".skip"}:
                count = _layout_int(directive_args[0], aliases) if directive_args else None
                if len(directive_args) not in {1, 2} or count is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if len(directive_args) == 2 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                pc += max(count, 0)
            elif parts and parts[0] == ".fill":
                if not 1 <= len(directive_args) <= 3:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                repeat = _layout_int(directive_args[0], aliases)
                width = _layout_int(directive_args[1], aliases) if len(directive_args) > 1 and directive_args[1].strip() else 1
                if repeat is None or width is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if len(directive_args) == 3 and directive_args[2].strip() and _layout_int(directive_args[2], aliases) is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                pc += max(repeat, 0) * min(max(width, 0), 8)
            elif parts and parts[0] == ".incbin":
                size = _incbin_size(
                    source_path, directive_args, aliases, include_context,
                )
                if size is None:
                    return ProfileResult(None, "profile-gap:incbin-source-missing")
                pc += size
            elif parts and parts[0] == ".org":
                address = _layout_int(directive_args[0], aliases) if directive_args else None
                if len(directive_args) not in {1, 2} or address is None or address < 0 or address < pc:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                if len(directive_args) == 2 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                pc = address
            elif parts and parts[0] in _FLOAT_DIRECTIVE_WIDTHS:
                if any(
                        not _float_literal_valid(value) for value in directive_args
                ):
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                pc += _FLOAT_DIRECTIVE_WIDTHS[parts[0]] * len(directive_args)
            elif parts and parts[0] in {".uleb128", ".sleb128"}:
                size = _leb128_size(directive_args, parts[0] == ".sleb128", aliases)
                if size is None:
                    return ProfileResult(None, "profile-gap:layout-directive-invalid")
                pc += size
            cursor += len(raw)
            continue
        if rows and not pending_labels and _control_shape(rows[-1]) is not None:
            block = f"b{len({row['block'] for row in rows})}"
            block_label = None
        parts = body.split(None, 1)
        mnemonic = parts[0].lower().replace("_", ".")
        operands = _split_operands(parts[1]) if len(parts) > 1 else ()
        insn_control = _insn_control_mnemonic(operands, aliases) if mnemonic == ".insn" else None
        if any(not operand for operand in operands):
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        if mnemonic == "fence" and len(parts) > 1 and parts[1].strip() and not operands:
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        source_mnemonic = _source_mnemonic(mnemonic)
        spec = instruction_spec_for_generation(
            "rev8.rv32" if source_mnemonic == "rev8" and xlen == 32 else source_mnemonic
        )
        if declared_isa is not None and spec is not None \
                and not profile_enables_form(declared_isa, spec.form):
            return ProfileResult(None, "profile-gap:isa-profile-mismatch")
        if not _pseudo_enabled(mnemonic, active_isa):
            return ProfileResult(None, "profile-gap:instruction-disabled-for-isa")
        if mnemonic in {"beqi", "bnei"} and "zibi" not in enabled_extensions(active_isa):
            return ProfileResult(None, "profile-gap:instruction-disabled-for-isa")
        if (not code_directive and spec is None
                and mnemonic not in _PROFILE_PSEUDOS
                and _control_shape_for_mnemonic(mnemonic) is None
                and (source_pc_map is None or final_instruction_map is None)):
            return ProfileResult(None, "profile-gap:instruction-unknown")
        if spec is not None and mnemonic not in _BASE_EXPANDING_PSEUDOS and (
                not profile_enables_form(active_isa, spec.form)
                or mnemonic.startswith(("c.", "cm.")) and not compressed
        ):
            return ProfileResult(None, "profile-gap:instruction-disabled-for-isa")
        runtime_spec = runtime_instruction_spec(mnemonic, xlen)
        arity = _PROFILE_PSEUDO_ARITIES.get(mnemonic, _PROFILE_CONTROL_ARITIES.get(mnemonic))
        if arity is not None and len(operands) != arity:
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        if mnemonic == "zext.b" and (
            len(operands) != 2 or any(_register_index(operand) is None for operand in operands)
        ):
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        if mnemonic != ".insn" and not _relocation_valid(mnemonic, operands):
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        encodable_operands = (
            operands if _control_shape_for_mnemonic(mnemonic) is not None
            and mnemonic not in {"jalr", "jr"}
            else tuple(_relocation_neutral(value) for value in operands)
        )
        if not _profile_operands_valid(mnemonic, encodable_operands, aliases, xlen=xlen, isa_profile=active_isa):
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        if mnemonic in {"cm.push", "cm.pop", "cm.popret", "cm.popretz"}:
            if not _zcmp_stack_operands_valid(mnemonic, operands, xlen, aliases):
                return ProfileResult(None, "profile-gap:instruction-operands-invalid")
            runtime_spec = None
        operands_valid = runtime_spec is None or mnemonic in _BASE_EXPANDING_PSEUDOS or _profile_operands_valid(
            mnemonic, operands, aliases, xlen=xlen, isa_profile=active_isa
        )
        if (operands_valid and runtime_spec is not None
                and mnemonic not in _BASE_EXPANDING_PSEUDOS
                and not _text_encodable(
                    runtime_spec, runtime_spec.operand_roles, encodable_operands,
                    aliases, isa_profile=active_isa,
                )):
            target = operands[-1] if operands else ""
            valid_target_alias = (
                operands and mnemonic not in {"jalr", "jr"}
                and _control_shape_for_mnemonic(mnemonic) in {"conditional", "direct"}
                and _register_index(str(target)) is None
            )
            operands_valid = valid_target_alias and _text_encodable(
                runtime_spec, runtime_spec.operand_roles,
                (*encodable_operands[:-1], "0"), aliases,
                isa_profile=active_isa,
            )
        if (
            not operands_valid
            and runtime_spec is not None
            and operands
            and mnemonic not in {"jalr", "jr"}
            and _control_shape_for_mnemonic(mnemonic) in {"conditional", "direct"}
            and _register_index(str(operands[-1])) is None
        ):
            operands_valid = _text_encodable(
                runtime_spec, runtime_spec.operand_roles, (*encodable_operands[:-1], "0"), aliases, isa_profile=active_isa
            )
        if not operands_valid:
            return ProfileResult(None, "profile-gap:instruction-operands-invalid")
        if mnemonic in _FP_PSEUDOS and not _fp_pseudo_enabled(mnemonic, active_isa):
            return ProfileResult(None, "profile-gap:instruction-disabled-for-isa")
        if active_isa.startswith("rv32e") and (
            any((_register_index(operand) or -1) >= 16 for operand in operands)
            or any(
                (memory[1] or -1) >= 16
                for operand in operands
                for memory in (_memory_parts(str(operand)),)
                if memory is not None
            )
        ):
            return ProfileResult(None, "profile-gap:register-disabled-for-isa")
        directive_size = (
            _data_directive_size(directive_width, operands, aliases)
            if directive_width is not None else None
            if code_directive else None
        )
        if code_directive and mnemonic != ".insn" and directive_size is None:
            return ProfileResult(None, "profile-gap:data-directive-invalid")
        li_size_operands = operands
        li_value = None
        if mnemonic == "li" and len(operands) == 2:
            li_base, li_value = _resolve_target(operands[1], aliases)
            if li_base is None:
                li_size_operands = (operands[0], str(li_value))
        if (mnemonic == "li" and source_pc_map is None
                and (not operands or li_value is None)):
            return ProfileResult(None, "profile-gap:layout-unresolved")
        if mnemonic == ".insn":
            size = _insn_size(operands, aliases)
            if size is None:
                return ProfileResult(None, "profile-gap:data-directive-invalid")
            if size == 2 and not profile_uses_compressed_encoding(active_isa):
                return ProfileResult(None, "profile-gap:compressed-insn-without-c")
        else:
            size = directive_size if code_directive else _instruction_size(
                mnemonic,
                tuple(_resolve_size_operand(value, aliases) for value in li_size_operands),
                compressed, xlen, active_isa, pc=pc,
            )
        source_start = cursor + raw.find(body)
        row = {
            "id": f"i-{pc:x}-{source_start:x}", "line": line, "start": source_start,
            "end": source_start + len(body),
            "pc": pc, "pc_end": pc + size, "mnemonic": mnemonic,
            "operands": operands, "block": block, "label": block_label,
            "labels": tuple(pending_labels),
            "_aliases": dict(aliases),
            "executed": pc in pcs, "compressed": compressed, "xlen": xlen,
            "isa": active_isa, "section": section_name,
            "logical_line": logical_lines[line] or line + 1,
            "translation_cell": f"scalar:{mnemonic}",
            "rule_facts": _rule_facts(mnemonic, xlen),
        }
        if insn_control is not None:
            row["control_mnemonic"] = insn_control
        if pc in before_by_pc:
            row["before_gpr"] = before_by_pc[pc]
        rows.append(row)
        pending_labels = []
        pc += size
        cursor += len(raw)

    if program_path is not None and source_pc_map is None and len(profile_ranges) > 1:
        # linux_user_harness inserts ``j exit_checkpoint`` between test_done
        # and the materialized subprograms.  The raw source is intentionally
        # parsed without that harness line; recover its exact size from the
        # observed body-end marker before resolving branch targets.
        subprogram_line = profile_ranges[1][0]
        body_end = _pc_value(program_path.get("end_pc"))
        layout_gap = body_end - pc if body_end is not None else 0
        if layout_gap < 0 or layout_gap % (2 if compressed_hint else 4):
            return ProfileResult(None, "profile-gap:program-test-layout-mismatch")
        if layout_gap:
            boundary = source_offsets[subprogram_line]
            for row in rows:
                if row["line"] >= subprogram_line:
                    row["pc"] += layout_gap
                    row["pc_end"] += layout_gap
            for label, points in label_points.items():
                label_points[label] = [
                    (start, point_pc + layout_gap if start >= boundary else point_pc)
                    for start, point_pc in points
                ]
            pc += layout_gap

    if source_pc_map is not None:
        mapped_lines = sorted(
            line for line in source_pc_map
            if any(row["logical_line"] == line for row in rows)
        )
        if mapped_lines:
            anchor_line = mapped_lines[0]
            anchor_row = next(row for row in rows if row["logical_line"] == anchor_line)
            shift = source_pc_map[anchor_line][0] - anchor_row["pc"]
            if shift:
                for row in rows:
                    if row["logical_line"] < anchor_line:
                        row["pc"] += shift
                        row["pc_end"] += shift
                for label, points in label_points.items():
                    label_points[label] = [
                        (start, point_pc + shift if start < anchor_row["start"] else point_pc)
                        for start, point_pc in points
                    ]

    pc_alignment = 2 if compressed_hint else 4
    if any(pc % pc_alignment for pc in pcs):
        return ProfileResult(None, "profile-gap:dynamic-trace-alignment")
    if source_pc_map is not None and any(
        pc % pc_alignment for values in source_pc_map.values() for pc in values
    ):
        return ProfileResult(None, "profile-gap:source-mapping-invalid")
    if final_instruction_map is not None and any(
        pc % pc_alignment for pc in final_instruction_map
    ):
        return ProfileResult(None, "profile-gap:final-instruction-alignment")

    while True:
        widened = False
        for row_index, row in enumerate(rows):
            mnemonic = str(row.get("control_mnemonic") or row["mnemonic"])
            if mnemonic not in _PROFILE_BRANCH_IMMEDIATE_OPS | _PROFILE_JUMP_IMMEDIATE_OPS:
                continue
            size = row["pc_end"] - row["pc"]
            if size not in {2, 4, 6}:
                continue
            target_pc = _branch_target_pc(row, rows, label_points, aliases)
            if target_pc is None:
                # A static RVDV profile intentionally stops before the
                # harness-only exit tail.  Its dead compressed jump may point
                # to a label outside the selected program ranges; the raw
                # preflight already proved the emitted source is assemblable.
                # An executed missing target remains a real profile gap.
                if size == 2 and row.get("executed"):
                    return ProfileResult(None, "profile-gap:compressed-control-target-missing")
                continue
            if mnemonic in _PROFILE_JUMP_IMMEDIATE_OPS:
                if size != 2:
                    if mnemonic == ".insn.j":
                        delta = target_pc - row["pc"]
                        if not -(1 << 20) <= delta <= (1 << 20) - 2:
                            return ProfileResult(None, "profile-gap:jump-immediate-out-of-range")
                    continue
                delta = target_pc - row["pc"]
                if -2048 <= delta <= 2046:
                    continue
                if mnemonic in {"c.j", "c.jal"}:
                    return ProfileResult(None, "profile-gap:compressed-control-out-of-range")
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    return ProfileResult(None, "profile-gap:compressed-control-out-of-range")
                new_size = 4
            elif size == 2:
                delta = target_pc - row["pc"]
                if -256 <= delta <= 254:
                    continue
                if -4096 <= delta <= 4094:
                    new_size = 4
                elif -(1 << 20) <= delta <= (1 << 20) - 2:
                    new_size = 6
                else:
                    return ProfileResult(None, "profile-gap:compressed-control-out-of-range")
            elif size == 4:
                delta = target_pc - row["pc"]
                if mnemonic == ".insn.b":
                    if not -4096 <= delta <= 4094:
                        return ProfileResult(None, "profile-gap:branch-immediate-out-of-range")
                    continue
                if -4096 <= delta <= 4094:
                    continue
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    return ProfileResult(None, "profile-gap:branch-immediate-out-of-range")
                new_size = 8
            else:
                delta = target_pc - (row["pc"] + 2)
                if not -(1 << 20) <= delta <= (1 << 20) - 2:
                    return ProfileResult(None, "profile-gap:branch-immediate-out-of-range")
                continue
            growth = new_size - size
            row["pc_end"] += growth
            for later in rows[row_index + 1:]:
                later["pc"] += growth
                later["pc_end"] += growth
            for label, points in label_points.items():
                label_points[label] = [
                    (start, point_pc + growth if start > row["start"] else point_pc)
                    for start, point_pc in points
                ]
            widened = True
        if not widened:
            break
    if program_path is not None and source_pc_map is None and final_instruction_map is not None:
        # Attach the real encoded layout without trusting its DWARF line
        # ownership.  A source row is eligible only when its static span is
        # covered exactly by contiguous disassembled instructions.
        for row in rows:
            items = tuple(
                (pc, item) for pc, item in sorted(final_instruction_map.items())
                if row["pc"] <= pc < row["pc_end"]
            )
            if (
                items
                and items[0][0] == row["pc"]
                and items[-1][0] + items[-1][1]["size"] == row["pc_end"]
                and sum(item["size"] for _, item in items) == row["pc_end"] - row["pc"]
            ):
                row["final_pcs"] = tuple(pc for pc, _ in items)
                row["final_sizes"] = tuple(item["size"] for _, item in items)
                row["final_bytes"] = tuple(item["bytes"] for _, item in items)
    if identity.get("harness") != "riscv-dv" and checkpoint is not None:
        last_row = next(
            (row for row in rows if row["pc"] <= pcs[-1] < row["pc_end"]), None
        )
        trace_after = trace_entries[-1].get("after_pc") if trace_entries else None
        if last_row is not None and checkpoint not in {pcs[-1], last_row["pc_end"], trace_after}:
            trace_details = _profile_get(
                _profile_get(observation, "translation_evidence"), "details", {}
            )
            if _profile_get(trace_details, "terminal_ebreak") is not True:
                return ProfileResult(None, "profile-gap:checkpoint-pc-mismatch")
    if any(
        _control_shape(row) == "conditional" and row.get("operands")
        and _register_index(str(row["operands"][-1])) is None
        and _branch_target_pc(row, rows, label_points, aliases) is None
        for row in rows if row.get("executed")
    ):
        return ProfileResult(None, "profile-gap:branch-target-missing")
    if any(
        _control_shape(row) == "conditional" and row.get("operands")
        and (target := _branch_target_pc(row, rows, label_points, aliases)) is not None
        and target & 1
        for row in rows if row.get("executed")
    ):
        return ProfileResult(None, "profile-gap:branch-target-invalid")
    if any(
        _control_target_invalid(row, rows, label_points, aliases)
        for row in rows if row.get("executed")
    ):
        return ProfileResult(None, "profile-gap:control-target-invalid")
    if (
        identity.get("harness") == "riscv-dv"
        and source_pc_map is not None
        and final_instruction_map is not None
        and any(
            line not in {row.get("logical_line") for row in rows}
            for line in source_pc_map
        )
    ):
        # RVDV line tables can point at a label-only line (for example a
        # harness boundary) while the disassembled instruction map is exact.
        # In that case use the parsed instruction layout as the authoritative
        # executable mapping instead of rejecting an otherwise complete trace.
        source_pc_map = None
        source_map_pcs = []
    if source_pc_map is None:
        terminal = _pc_value(identity.get("exit"))
        if (
            exit_declared and len(pcs) > 1 and terminal == pcs[-1]
            and not any(row["pc"] == terminal for row in rows)
            and any(
                row["pc_end"] == terminal
                and row["pc"] <= pcs[-2] < row["pc_end"]
                for row in rows
            )
        ):
            pcs = pcs[:-1]
            if trace_entries and trace_entries[-1].get("pc") == terminal:
                trace_entries = trace_entries[:-1]
        for row in rows:
            row["executed"] = any(row["pc"] <= value < row["pc_end"] for value in pcs)
        if any(
            not any(row["pc"] <= value < row["pc_end"] for row in rows)
            for value in pcs
        ):
            return ProfileResult(None, "profile-gap:executed-pc-unmapped")

    mapped_by_pc = {}
    if source_pc_map is not None:
        executed_pcs = set(pcs)
        rows_by_line = {}
        for row in rows:
            rows_by_line.setdefault(row["logical_line"], []).append(row)
        mapped_rows = []
        for line, line_rows in rows_by_line.items():
            starts = source_pc_map.get(line, ())
            if not starts:
                continue
            if len(line_rows) == 1:
                assignments = ((line_rows[0], starts),)
            elif final_instruction_map is not None and len(starts) == len(line_rows):
                assignments = tuple((row, (pc,)) for row, pc in zip(line_rows, starts))
            elif final_instruction_map is not None:
                assignments = []
                cursor = 0
                for row in line_rows:
                    if cursor >= len(starts):
                        break
                    end = cursor + 1
                    expected = row["pc_end"] - row["pc"]
                    size = final_instruction_map[starts[cursor]]["size"]
                    while size < expected and end < len(starts):
                        size += final_instruction_map[starts[end]]["size"]
                        end += 1
                    assignments.append((row, tuple(starts[cursor:end])))
                    cursor = end
                assignments = tuple(assignments)
                if len(assignments) != len(line_rows) or cursor != len(starts):
                    return ProfileResult(None, "profile-gap:source-mapping-ambiguous")
            else:
                assignments = tuple(
                    (row, tuple(pc for pc in starts if row["pc"] <= pc < row["pc_end"]))
                    for row in line_rows
                )
                if any(not row_starts for _row, row_starts in assignments) \
                        or sum(len(row_starts) for _row, row_starts in assignments) != len(starts):
                    return ProfileResult(None, "profile-gap:source-mapping-ambiguous")
            for row, row_starts in assignments:
                source_pc = row["pc"]
                row["pc"], row["final_pcs"] = row_starts[0], row_starts
                if final_instruction_map is not None:
                    row["final_sizes"] = tuple(final_instruction_map[pc]["size"] for pc in row_starts)
                    row["final_bytes"] = tuple(final_instruction_map[pc]["bytes"] for pc in row_starts)
                    row_end = row_starts[-1] + row["final_sizes"][-1]
                else:
                    row_end = next_map_pc.get(
                        row_starts[-1], row["pc_end"] + row_starts[0] - source_pc
                    )
                if row_end <= row["pc"]:
                    return ProfileResult(None, "profile-gap:source-mapping-invalid")
                row["pc_end"] = row_end
                mapped_rows.append(row)
        if final_instruction_map is not None:
            for row in rows:
                width = _DATA_DIRECTIVE_WIDTHS.get(row["mnemonic"])
                if row in mapped_rows or width is None or len(row["operands"]) != 1:
                    continue
                value = _layout_int(row["operands"][0], row.get("_aliases"))
                if value is None:
                    continue
                encoded = f"{value & ((1 << (width * 8)) - 1):0{width * 2}x}"
                matches = tuple(
                    pc for pc, item in final_instruction_map.items()
                    if item["bytes"] == encoded and abs(pc - row["pc"]) <= 16
                )
                owners = tuple(
                    owner for owner in mapped_rows
                    if any(pc in owner.get("final_pcs", ()) for pc in matches)
                )
                if len(matches) != 1 or len(owners) != 1:
                    continue
                actual = matches[0]
                owner = owners[0]
                kept = tuple(pc for pc in owner["final_pcs"] if pc != actual)
                if not kept:
                    continue
                owner["final_pcs"] = kept
                owner["final_sizes"] = tuple(final_instruction_map[pc]["size"] for pc in kept)
                owner["final_bytes"] = tuple(final_instruction_map[pc]["bytes"] for pc in kept)
                owner["pc"] = kept[0]
                owner["pc_end"] = kept[-1] + owner["final_sizes"][-1]
                row["pc"] = actual
                row["pc_end"] = actual + width
                row["final_pcs"] = (actual,)
                row["final_sizes"] = (width,)
                row["final_bytes"] = (final_instruction_map[actual]["bytes"],)
                mapped_rows.append(row)
        mapped_rows.sort(key=lambda item: item["pc"])
        if any(left["pc_end"] > right["pc"] for left, right in zip(mapped_rows, mapped_rows[1:])):
            return ProfileResult(None, "profile-gap:source-mapping-ambiguous")
        for row in rows:
            row["executed"] = False
        starts = [row["pc"] for row in mapped_rows]
        for pc in sorted(executed_pcs):
            index = bisect_left(starts, pc + 1) - 1
            if index >= 0 and pc < mapped_rows[index]["pc_end"]:
                row = mapped_rows[index]
                mapped_by_pc[pc] = row
                row["executed"] = True
        missing = tuple(pc for pc in pcs if pc not in mapped_by_pc)
        if missing:
            if identity.get("harness") == "custom":
                mapped_indices = [index for index, pc in enumerate(pcs) if pc in mapped_by_pc]
                edge = mapped_indices == list(range(mapped_indices[0], mapped_indices[-1] + 1)) if mapped_indices else False
                mapped_pcs = tuple(pcs[mapped_indices[0]:mapped_indices[-1] + 1]) if edge else ()
                if edge and mapped_pcs:
                    # ponytail: 丢弃 custom harness 注入的首尾边界指令；升级路径是保留独立 harness 源映射。
                    pcs = mapped_pcs
                    executed_pcs = set(pcs)
                    identity["entry"], identity["exit"] = pcs[0], pcs[-1]
                    missing = ()
                else:
                    return ProfileResult(None, "profile-gap:executed-pc-unmapped")
            else:
                terminal = _pc_value(identity.get("exit"))
                previous = mapped_by_pc.get(pcs[-2]) if len(pcs) > 1 else None
                if (
                    missing == (terminal,)
                    and pcs[-1] == terminal
                    and isinstance(previous, Mapping)
                    and previous["pc_end"] == terminal
                ):
                    pcs = pcs[:-1]
                    executed_pcs.discard(terminal)
                    if trace_entries and trace_entries[-1].get("pc") == terminal:
                        trace_entries = trace_entries[:-1]
                else:
                    return ProfileResult(None, "profile-gap:executed-pc-unmapped")
        if not mapped_by_pc:
            return ProfileResult(None, "profile-gap:executed-pc-unmapped")

    for row in rows:
        if _control_shape(row) == "conditional":
            target_pc = _branch_target_pc(row, rows, label_points, aliases)
            if target_pc is not None:
                row["target_pc"] = target_pc

    pc_starts = [row["pc"] for row in rows]
    def row_for_pc(value: int) -> Mapping[str, object] | None:
        if source_pc_map is not None:
            return mapped_by_pc.get(value)
        index = bisect_left(pc_starts, value)
        if index == len(pc_starts) or pc_starts[index] != value:
            index -= 1
        if index < 0:
            return None
        row = rows[index]
        size = row["pc_end"] - row["pc"]
        compressed_pair = (
            row.get("compressed") and row.get("mnemonic") == "zext.w"
            and row.get("xlen") == 64 and size in {4, 6, 8}
            and not _isa_has_extension(row.get("isa"), "zba")
        )
        long_compressed_control = (
            size == 6 and _control_shape(row) == "conditional"
        )
        wide_li = False
        li_sizes = ()
        if row.get("compressed") and row.get("mnemonic") == "li":
            operands = tuple(row.get("operands", ()))
            literal = _layout_int(
                operands[-1], row.get("_aliases")
            ) if operands else None
            if literal is not None:
                li_sizes = _li_instruction_sizes(
                    literal, row["xlen"], True,
                    _register_index(operands[0]) if operands else None,
                )
                wide_li = len(li_sizes) > 1
        starts = {0}
        data_width = _DATA_DIRECTIVE_WIDTHS.get(str(row.get("mnemonic", "")))
        if data_width is not None and len(row.get("operands", ())) > 1:
            starts.update(range(0, size, data_width))
        if wide_li:
            offset = 0
            for width in li_sizes[:-1]:
                offset += width
                starts.add(offset)
        elif compressed_pair:
            operands = tuple(row.get("operands", ()))
            starts.add(_instruction_size(
                "slli", (operands[0], operands[1], "32"), True,
                row["xlen"], row.get("isa"),
            ))
        elif long_compressed_control:
            starts.add(2)
        offset = value - row["pc"]
        valid_offset = (
            offset in starts if wide_li or compressed_pair or long_compressed_control
            or data_width is not None and len(row.get("operands", ())) > 1
            else offset == 0 or offset % 4 == 0 and size > 4
        )
        return row if value < row["pc_end"] and (program_path is not None or valid_offset) else None

    if identity.get("harness") == "riscv-dv" and start is not None and end is not None and checkpoint is not None:
        test_done_pc = checkpoint
        checkpoint_pcs = {test_done_pc} if test_done_pc is not None else set()
        next_compressed = compressed
        next_isa = active_isa
        next_xlen = xlen
        post_compressed_stack, post_isa_stack = [], []
        post_section_stack, post_in_text, post_previous = [], True, None
        post_pc = test_done_pc
        post_lines = []
        terminal_seen = False

        def terminal_body(statement: str) -> str | None:
            body = _strip_comments(statement).strip()
            while match := re.match(
                r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body
            ):
                body = match[2].strip()
                if match[1] == "test_done":
                    return body
            return None

        for statement in _split_source_statements(source_lines[end]):
            tail = terminal_body(statement)
            if tail is not None:
                terminal_seen = True
                if tail:
                    post_lines.append(tail)
                continue
            if terminal_seen:
                post_lines.append(statement)
        post_lines.extend(source_lines[end + 1:])
        post_ended = False
        for raw in post_lines:
            if post_ended:
                break
            for statement in _split_source_statements(raw):
                if post_ended:
                    break
                body = _strip_comments(statement).strip()
                while (match := re.match(r"^([A-Za-z_.$][\w.$]*|\d+)\s*:\s*(.*)$", body)):
                    body = match[2].strip()
                normalized = body.lower()
                words = normalized.split()
                if words and words[0] == ".end":
                    post_ended = True
                    break
                try:
                    next_isa, next_compressed = _profile_arch_options(
                        normalized, next_isa, next_compressed
                    )
                except ValueError:
                    return ProfileResult(None, "profile-gap:isa-option-invalid")
                next_xlen = 32 if next_isa.lower().startswith("rv32") else 64
                if next_xlen != base_xlen:
                    return ProfileResult(None, "profile-gap:isa-profile-mismatch")
                if words[:2] == [".option", "push"]:
                    post_compressed_stack.append(next_compressed)
                    post_isa_stack.append(next_isa)
                    continue
                if words[:2] == [".option", "pop"]:
                    if not post_compressed_stack:
                        return ProfileResult(None, "profile-gap:option-invalid")
                    next_compressed = post_compressed_stack.pop()
                    next_isa = post_isa_stack.pop()
                    next_xlen = 32 if next_isa.lower().startswith("rv32") else 64
                    if next_xlen != base_xlen:
                        return ProfileResult(None, "profile-gap:isa-profile-mismatch")
                    continue
                if words and words[0] == ".pushsection":
                    post_section_stack.append((post_in_text, post_previous))
                    post_previous = post_in_text
                    flags = re.search(r',\s*"([^"\n]*)"', normalized)
                    post_in_text = "x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", normalized))
                    continue
                if words and words[0] == ".popsection":
                    if len(words) != 1 or not post_section_stack:
                        return ProfileResult(None, "profile-gap:section-invalid")
                    post_in_text, post_previous = post_section_stack.pop()
                    continue
                if words and words[0] in {".text", ".section"}:
                    post_previous = post_in_text
                    flags = re.search(r',\s*"([^"\n]*)"', normalized)
                    post_in_text = words[0] == ".text" or ("x" in flags[1] if flags else bool(re.search(r"\.text(?:[.\s]|$)", normalized)))
                    continue
                if words and words[0] in {".data", ".bss", ".rodata", ".sdata", ".sbss", ".tdata", ".tbss"}:
                    post_previous = post_in_text
                    post_in_text = False
                    continue
                if words and words[0] == ".previous" and post_previous is not None:
                    post_in_text, post_previous = post_previous, post_in_text
                    continue
                if words and words[0] == ".previous":
                    return ProfileResult(None, "profile-gap:section-invalid")
                if words and words[0] in {".equ", ".set"}:
                    args = _split_operands(body.split(None, 1)[1]) if len(words) > 1 else ()
                    if len(args) >= 2:
                        aliases[args[0]] = args[1]
                    continue
                if not post_in_text:
                    continue
                if (words and words[0].startswith(".")
                        and words[0] not in _KNOWN_PROFILE_DIRECTIVES
                        and not words[0].startswith(_PROFILE_DIRECTIVE_PREFIXES)
                        and not words[0].startswith(".text.")):
                    return ProfileResult(None, "profile-gap:directive-unknown")
                directive_args = _split_operands(
                    body.split(None, 1)[1]
                ) if len(words) > 1 else ()
                if words and words[0] == ".insn" and len(words) > 1 \
                        and not _insn_operand_syntax_valid(body.split(None, 1)[1]):
                    return ProfileResult(None, "profile-gap:data-directive-invalid")
                if words and words[0] in {
                    ".balign", ".balignw", ".balignl", ".align", ".p2align", ".p2alignw", ".p2alignl",
                }:
                    directive_args = tuple(
                        item.strip() for item in body.split(None, 1)[1].split(",")
                    ) if len(words) > 1 else ()
                    alignment = _layout_int(directive_args[0], aliases) if directive_args else None
                    if not directive_args:
                        continue
                    if len(directive_args) > 3 or alignment is None or alignment < 0 or post_pc is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if words[0].startswith(".balign") and alignment and alignment & (alignment - 1):
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if len(directive_args) > 1 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    alignment = 1 if alignment == 0 else 1 << alignment if words[0] in {
                        ".align", ".p2align", ".p2alignw", ".p2alignl",
                    } else alignment
                    padding = (-post_pc) % alignment
                    max_skip = _layout_int(directive_args[2], aliases) if len(directive_args) > 2 and directive_args[2].strip() else None
                    if len(directive_args) > 2 and directive_args[2].strip() and (max_skip is None or max_skip < 0):
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if max_skip is None or padding <= max_skip:
                        if (
                            words[0] in {".balignw", ".p2alignw"} and padding % 2
                            or words[0] in {".balignl", ".p2alignl"} and padding % 4
                        ):
                            return ProfileResult(None, "profile-gap:layout-directive-invalid")
                        post_pc += padding
                    continue
                if words and words[0] in {".space", ".zero", ".skip"}:
                    count = _layout_int(directive_args[0], aliases) if directive_args else None
                    if len(directive_args) not in {1, 2} or count is None or count < 0 or post_pc is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if len(directive_args) == 2 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += count
                    continue
                if words and words[0] == ".fill":
                    if not 1 <= len(directive_args) <= 3 or post_pc is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    repeat = _layout_int(directive_args[0], aliases)
                    width = _layout_int(directive_args[1], aliases) if len(directive_args) > 1 and directive_args[1].strip() else 1
                    if repeat is None or width is None or repeat < 0 or width < 0:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if len(directive_args) == 3 and directive_args[2].strip() and _layout_int(directive_args[2], aliases) is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += repeat * width
                    continue
                if words and words[0] == ".incbin":
                    size = _incbin_size(
                        source_path, directive_args, aliases, include_context,
                    )
                    if size is None:
                        return ProfileResult(None, "profile-gap:incbin-source-missing")
                    post_pc += size
                    checkpoint_pcs.add(post_pc)
                    continue
                if words and words[0] == ".org":
                    address = _layout_int(directive_args[0], aliases) if directive_args else None
                    if len(directive_args) not in {1, 2} or address is None or address < 0 or post_pc is None or address < post_pc:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    if len(directive_args) == 2 and directive_args[1].strip() and _layout_int(directive_args[1], aliases) is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc = address
                    continue
                if words and words[0] in _FLOAT_DIRECTIVE_WIDTHS:
                    if any(
                        not _float_literal_valid(value) for value in directive_args
                    ):
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += _FLOAT_DIRECTIVE_WIDTHS[words[0]] * len(directive_args)
                    if directive_args:
                        checkpoint_pcs.add(post_pc)
                    continue
                if words and words[0] in {".uleb128", ".sleb128"}:
                    size = _leb128_size(directive_args, words[0] == ".sleb128", aliases)
                    if size is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += size
                    if directive_args:
                        checkpoint_pcs.add(post_pc)
                    continue
                raw_width = _DATA_DIRECTIVE_WIDTHS.get(words[0]) if words else None
                if words and words[0] in {".ascii", ".asciz", ".string"}:
                    size = _string_size(body)
                    if size is None or post_pc is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += size
                    if size:
                        checkpoint_pcs.add(post_pc)
                    continue
                if raw_width is not None:
                    size = _data_directive_size(raw_width, directive_args, aliases)
                    if size is None:
                        return ProfileResult(None, "profile-gap:layout-directive-invalid")
                    post_pc += size
                    if directive_args:
                        checkpoint_pcs.add(post_pc)
                    continue
                if not body or (body.startswith(".") and not normalized.startswith(".insn")):
                    continue
                parts = body.split(None, 1)
                mnemonic = parts[0].lower().replace("_", ".")
                operands = _split_operands(parts[1]) if len(parts) > 1 else ()
                if mnemonic == ".insn":
                    size = _insn_size(operands, aliases)
                    if size is None:
                        return ProfileResult(None, "profile-gap:data-directive-invalid")
                else:
                    size = _instruction_size(
                        mnemonic,
                        tuple(_resolve_size_operand(value, aliases) for value in operands),
                        next_compressed, next_xlen, next_isa, pc=post_pc,
                    )
                post_pc += size
                checkpoint_pcs.add(post_pc)
                continue
            else:
                continue
        if checkpoint not in checkpoint_pcs:
            trace_details = _profile_get(
                _profile_get(observation, "translation_evidence"), "details", {}
            )
            terminal_ebreak = _profile_get(trace_details, "terminal_ebreak") is True
            if not terminal_ebreak:
                return ProfileResult(None, "profile-gap:checkpoint-pc-mismatch")
            checkpoint_pcs.add(checkpoint)
    if checkpoint is not None and checkpoint not in pcs and checkpoint not in checkpoint_pcs:
        last_row = row_for_pc(pcs[-1]) if pcs else None
        if last_row is None or checkpoint != last_row["pc_end"]:
            return ProfileResult(None, "profile-gap:checkpoint-pc-unmapped")
    trace_rows = [(pc, row_for_pc(pc)) for pc in pcs]
    removed_exit = False
    if (
        exit_declared and trace_rows and trace_rows[-1][1] is None
        and trace_rows[-1][0] == _pc_value(identity["exit"])
        and (
            trace_rows[-1][0] in checkpoint_pcs
            or len(trace_rows) > 1
            and trace_rows[-2][1] is not None
            and trace_rows[-1][0] == trace_rows[-2][1]["pc_end"]
        )
    ):
        trace_rows.pop()
        removed_exit = True
    if not rows or not any(row["executed"] for row in rows):
        return ProfileResult(None, "profile-gap:executed-pc-unmapped")
    if len(trace_rows) != len(pcs) - int(removed_exit):
        return ProfileResult(None, "profile-gap:executed-pc-unmapped")
    if any(row is None for _, row in trace_rows):
        return ProfileResult(None, "profile-gap:executed-pc-unmapped")
    if expected_trap and identity.get("harness") == "custom":
        rows = rows[:max(index for index, row in enumerate(rows) if row["executed"]) + 1]

    trace_by_pc: dict[int, list[dict[str, object]]] = {}
    for item in trace_entries:
        trace_by_pc.setdefault(item["pc"], []).append(item)
    instruction_counts: dict[str, int] = {}
    for _, row in trace_rows:
        instruction_id = str(row["id"])
        instruction_counts[instruction_id] = instruction_counts.get(instruction_id, 0) + 1
    occurrences = []
    for ordinal, (pc, row) in enumerate(trace_rows):
        occurrence = {
            "id": f"o-{row['id']}-{ordinal}", "instruction_id": row["id"],
            "pc": pc, "ordinal": ordinal, "block": row["block"],
            "mnemonic": row["mnemonic"],
        }
        if final_instruction_map is not None:
            occurrence["final_bytes"] = final_instruction_map[pc]["bytes"]
        witness = trace_by_pc.get(pc, [])
        witness = witness.pop(0) if witness else {}
        if witness.get("_invalid_trace_fields"):
            occurrence["_invalid_trace_fields"] = tuple(witness["_invalid_trace_fields"])
        before = _gpr_state(witness.get("before_gpr"), xlen)
        if before is None and instruction_counts[str(row["id"])] == 1:
            before = _gpr_state(row.get("before_gpr"), xlen)
        if before is not None:
            occurrence["before_gpr"] = before
        for name in ("after_gpr", "before_fpr_rawbits", "after_fpr_rawbits"):
            value = witness.get(name)
            if isinstance(value, (list, tuple)) and all(_int_value(item) is not None for item in value):
                occurrence[name] = tuple(_int_value(item) for item in value)
        for name in (
            "after_pc", "before_fflags", "after_fflags", "before_frm",
            "after_frm", "fault_pc", "fault_address", "fault", "branch_outcome",
            "consumer", "link", "ialign",
        ):
            if name in witness:
                occurrence[name] = witness[name]
        if isinstance(witness.get("memory_reads"), (list, tuple)):
            occurrence["memory_reads"] = [
                dict(item) for item in witness["memory_reads"] if isinstance(item, Mapping)
            ]
        occurrences.append(occurrence)
    occurrences = tuple(occurrences)
    by_instruction: dict[str, list[Mapping[str, object]]] = {}
    for occurrence in occurrences:
        instruction_id = str(occurrence["instruction_id"])
        by_instruction.setdefault(instruction_id, []).append(occurrence)
    profile_view = SimpleNamespace(instructions=tuple(rows), occurrences=occurrences)
    for row in rows:
        row["witnesses"] = tuple(by_instruction.get(str(row["id"]), ()))
    for row_index, row in enumerate(rows):
        row_occurrence_ids = tuple(
            item["id"] for item in by_instruction.get(str(row["id"]), ())
        )
        if len(row["witnesses"]) == 1:
            row.update({
                name: row["witnesses"][0][name]
                for name in (
                    "before_gpr", "after_gpr", "before_fpr_rawbits", "after_fpr_rawbits",
                    "before_fflags", "after_fflags", "before_frm", "after_frm",
                    "memory_reads", "branch_outcome", "fault_pc", "fault_address", "fault",
                    "consumer", "link", "ialign",
                )
                if name in row["witnesses"][0]
            })
        facts = dict(row["rule_facts"])
        facts["relations"] = _program_relation_facts(
            row, profile_view, strict=False, occurrence_ids=row_occurrence_ids,
        )
        row["rule_facts"] = facts
        keep = set().union(*(
            _RULE_TRACE_FIELDS.get(rule, ())
            for rule, relation in facts["relations"].items()
            if relation.get("status") == "available"
        ))
        r9_relation = facts["relations"].get("R9")
        if (isinstance(r9_relation, Mapping) and r9_relation.get("status") == "available"
                and any("before_gpr" in item
                        for item in by_instruction.get(str(row["id"]), ()))):
            keep.add("before_gpr")
        row["witnesses"] = tuple(
            {name: value for name, value in occurrence.items()
             if name not in _TRACE_STATE_FIELDS or name in keep}
            for occurrence in row["witnesses"]
        )
        for name in _TRACE_STATE_FIELDS:
            if name not in keep:
                row.pop(name, None)
        if len(row["witnesses"]) == 1:
            row.update({
                name: row["witnesses"][0][name]
                for name in keep
                if name in row["witnesses"][0]
            })
        facts["relations"] = _program_relation_facts(
            row, profile_view, occurrence_ids=row_occurrence_ids,
        )
        row["rule_facts"] = facts
    unsupported = tuple(dict.fromkeys(
        row["mnemonic"] for row in rows if row["executed"]
        and (row["mnemonic"].startswith(("f", "v", "csr")) or row["mnemonic"] in {"ecall", "ebreak"})
    ))
    raw_reference_identity = _profile_get(observation, "reference_identity")
    if raw_reference_identity is not None and not isinstance(raw_reference_identity, Mapping):
        return ProfileResult(None, "profile-gap:reference-identity-invalid")
    if isinstance(raw_reference_identity, Mapping) and any(
        isinstance(_profile_get(observation, name), str)
        and isinstance(raw_reference_identity.get(name), str)
        and _profile_get(observation, name) != raw_reference_identity.get(name)
        for name in (
            "backend", "profile_id", "input_id", "tool_version",
            "binary_sha256", "guest_elf_sha256",
        )
    ):
        return ProfileResult(None, "profile-gap:reference-identity-mismatch")
    ref = raw_reference_identity
    if ref is None:
        ref = {
            name: _profile_get(observation, name)
            for name in (
                "backend", "profile_id", "input_id", "tool_version",
                "binary_sha256", "guest_elf_sha256",
            )
            if _profile_get(observation, name) not in (None, "")
        }
    ref = {name: value for name, value in ref.items() if value not in (None, "", {}, [], ())}
    if (
        not ref
        or not isinstance(ref.get("backend"), str)
        or not ref["backend"].strip()
        or any(
            name in ref and (not isinstance(ref[name], str) or not ref[name].strip())
            for name in (
                "profile_id", "input_id", "tool_version",
                "binary_sha256", "guest_elf_sha256",
            )
        )
        or any(
            name in ref and not is_sha256_digest(ref[name])
            for name in ("binary_sha256", "guest_elf_sha256")
        )
    ):
        return ProfileResult(None, "profile-gap:reference-identity-missing")
    text_sections = {
        value or ".text"
        for value in re.findall(
            r'(?m)^\s*(?:\.text(?:\s|$)|(?:\.section|\.pushsection)\s+([.]text(?:[.\w$]*))(?=\s*$|\s*,\s*"[^"\n]*x[^"\n]*"))',
            source,
        )
    }
    layout_unresolved = source_pc_map is None and bool(
        any(row["mnemonic"] in {"la", "lla", "lga", "call", "tail"} for row in rows)
        or re.search(r"(?m)^\s*\.option\s+relax\b", source)
        or len(text_sections) > 1
    )
    provenance = {"reference_identity": dict(ref), "direct_only": unsupported}
    if program_path is not None:
        provenance["execution_path"] = "program-test-path-relaxed"
    if program_path is not None:
        # The linux-user program-test slice is laid out by the same static
        # parser that validated its raw ELF.  The compiler line table is not a
        # reliable row map for the materialized subprogram tail, but that does
        # not make the parser's per-row PCs unusable for size-preserving rules.
        provenance["layout"] = "static"
    elif layout_unresolved:
        provenance["layout"] = "unresolved"
    if source_pc_map is not None:
        provenance["mapping"] = {
            "kind": "objdump-source-line+instruction-bytes"
            if final_instruction_map is not None else "objdump-source-line",
            "line_count": len(source_pc_map),
        }
        if final_instruction_map is not None:
            provenance["mapping"]["instruction_count"] = len(final_instruction_map)
            provenance["mapping"]["instruction_digest"] = canonical_digest(
                [{"pc": pc, **final_instruction_map[pc]} for pc in sorted(final_instruction_map)]
            )
        provenance["mapping"]["row_layout_digest"] = _profile_layout_digest(rows)
    elif program_path is not None and final_instruction_map is not None:
        provenance["mapping"] = {
            "kind": "static-source+instruction-bytes",
            "instruction_count": len(final_instruction_map),
            "instruction_digest": canonical_digest(
                [{"pc": pc, **final_instruction_map[pc]} for pc in sorted(final_instruction_map)]
            ),
            "row_layout_digest": _profile_layout_digest(rows),
        }
    if isinstance(generation, Mapping):
        provenance["generation"] = dict(generation)
    blocks = tuple(dict.fromkeys(row["block"] for row in rows))
    if program_path is None and any(
        _trace_transition_invalid(
            left[0], left[1], right[0], right[1], rows, aliases,
        ) and not (left[0] == right[0] and left[1]["id"] == right[1]["id"])
        for left, right in zip(trace_rows, trace_rows[1:])
    ):
        return ProfileResult(None, "profile-gap:execution-path-invalid")
    dynamic_edges = tuple(dict.fromkeys(
        (left["block"], right["block"])
        for (_, left), (_, right) in zip(trace_rows, trace_rows[1:])
        if left["block"] != right["block"]
        or left.get("mnemonic") in {"jr", "jalr", "c.jr", "c.jalr", "cm.jt", "cm.jalt"}
    ))
    static_edges = _profile_edges(tuple(rows), blocks, occurrences, aliases)
    edges = tuple(dict.fromkeys((*static_edges, *dynamic_edges)))
    provenance["executed_blocks"] = list(dict.fromkeys(row["block"] for _, row in trace_rows))
    provenance["edge_source"] = "static+trace"
    provenance["execution_trace"] = {
        "complete": execution_trace_complete,
        "count": len(pcs),
        "source": (
            "explicit" if trace_complete is not None
            else "instruction-count" if instruction_count is not None
            else "unattested"
        ),
    }
    executed_ids = tuple(dict.fromkeys(row["id"] for row in rows if row["executed"]))
    observer_fields = ["outcome", "executed_pcs"]
    if instruction_count is not None:
        observer_fields.append("instruction_count")
    if trace_complete is not None:
        observer_fields.append("trace_complete")
    if has_gpr:
        observer_fields.append("gpr")
    for name in ("checkpoint_pc", "memory_snapshot", "memory_digest", "extra_state", "state_trace"):
        if _profile_get(observation, name) is not None and (name != "state_trace" or trace_entries):
            observer_fields.append(name)
    for name in _TRACE_WITNESS_FIELDS:
        if any(name in occurrence for occurrence in occurrences):
            observer_fields.append(name)
    try:
        profile = ProgramProfile(
            program_sha256(program), identity, tuple(rows), executed_ids,
            tuple(dict.fromkeys(observer_fields)), provenance, blocks, edges, occurrences,
        )
        profile.provenance["profile_digest"] = profile.profile_digest
    except (TypeError, ValueError):
        return ProfileResult(None, "profile-gap:profile-record-invalid")
    return ProfileResult(profile) if profile.complete else ProfileResult(None, "profile-gap:incomplete")





@dataclass(frozen=True)
class ProgramVariant:
    """内存中的完整程序变体；执行上下文由 Runner 保存。"""

    source: str
    parent_sha256: str
    action_json: str
    run_params: Mapping[str, object] = field(default_factory=dict)
    _include_context: tuple[tuple[str, bytes], ...] = field(
        default=(), init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not isinstance(self.parent_sha256, str) \
                or not isinstance(self.action_json, str):
            raise ValueError("program variant fields must be strings")
        if not is_sha256_digest(self.parent_sha256):
            raise ValueError("program variant parent_sha256 must be a SHA-256 digest")
        if not isinstance(self.run_params, Mapping):
            raise ValueError("program variant run_params must be an object")
        object.__setattr__(self, "run_params", dict(self.run_params))
        object.__setattr__(
            self, "_include_context",
            _PROGRAM_INCLUDE_CONTEXT.get(self.parent_sha256, ()),
        )
        if self.action_json != "{}" and not self._include_context and any(
            _SOURCE_INCLUDE_RE.fullmatch(statement) or _SOURCE_INCBIN_RE.fullmatch(statement)
            for line in _joined_source_lines(_strip_c_comments(self.source))
            for statement in _split_source_statements(line)
        ):
            raise ValueError("program variant include identity unavailable")

    @property
    def sha256(self) -> str:
        return program_sha256(self)


def _source(program: object) -> str:
    if isinstance(program, ProgramVariant):
        return program.source
    path = getattr(program, "program", None)
    if path is not None:
        path = Path(path)
        if path.suffix.lower() != ".s":
            raise ValueError("program mutation currently requires a .S source")
        return path.read_bytes().decode("utf-8")
    raise TypeError("program must provide a .S path or be ProgramVariant")


def _action_source(program: object) -> str:
    source = _source(program)
    return (
        _expand_asm_macros(source)
        if getattr(program, "run_params", {}).get("harness") == "custom"
        else source
    )


_SOURCE_INCLUDE_RE = re.compile(
    r'^[ \t]*(?:(?:[A-Za-z_.$][\w.$]*|\d+)[ \t]*:[ \t]*)*(?:\.include|#\s*include)[ \t]+"([^"]+)"[ \t]*(?:#.*|//.*)?$',
    re.IGNORECASE,
)
_SOURCE_INCBIN_RE = re.compile(
    r'''^[ \t]*(?:(?:[A-Za-z_.$][\w.$]*|\d+)[ \t]*:[ \t]*)*\.incbin[ \t]+(["'])([^"']+)\1(?:[ \t]*,[^\r\n]*)?[ \t]*(?:#.*|//.*)?$''',
    re.IGNORECASE,
)
_SOURCE_BINARY_PREFIX = "__bytes__:"


def _joined_source_lines(source: str) -> tuple[str, ...]:
    lines, pending = [], ""
    for raw in source.splitlines(keepends=True):
        value = raw.rstrip("\r\n")
        if value.rstrip().endswith("\\"):
            pending += value.rstrip()[:-1] + " "
            continue
        lines.append(pending + raw.rstrip("\r\n"))
        pending = ""
    return (*lines, pending) if pending else tuple(lines)


def _source_include_files(path: Path) -> dict[str, bytes]:
    root = path.parent.resolve()
    files: dict[str, bytes] = {}
    seen: set[Path] = set()

    def visit(current: Path, stack: tuple[Path, ...] = (), scan_source: bool = True) -> None:
        current = _path_in_root(current, root, "source-dependency")
        if current in stack:
            raise ValueError("source-dependency-cyclic")
        if current in seen:
            return
        seen.add(current)
        data = current.read_bytes()
        try:
            name = current.relative_to(root).as_posix()
        except ValueError:
            name = Path(os.path.relpath(current, root)).as_posix()
        files[name] = data
        if not scan_source:
            return
        for line in _joined_source_lines(_strip_c_comments(data.decode("utf-8"))):
            for statement in _split_source_statements(line):
                match = _SOURCE_INCLUDE_RE.fullmatch(statement)
                if match:
                    visit(current.parent / match[1], (*stack, current))
                match = _SOURCE_INCBIN_RE.fullmatch(statement)
                if match:
                    visit(current.parent / match[2], (*stack, current), False)

    visit(path)
    return files


_PROGRAM_INCLUDE_CONTEXT: dict[str, tuple[tuple[str, bytes], ...]] = {}
_PROGRAM_SOURCE_CONTEXT: dict[str, bytes] = {}


def _program_identity_digest(
    raw: bytes, includes: Sequence[tuple[str, bytes]],
) -> str:
    if not includes:
        return hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(raw)
    for name, data in includes:
        digest.update(b"\0include:")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def program_sha256(program: object) -> str:
    if isinstance(program, ProgramVariant):
        raw = program.source.encode("utf-8")
        if program.action_json == "{}":
            includes = program._include_context or _PROGRAM_INCLUDE_CONTEXT.get(
                program.parent_sha256, ()
            )
            if not includes and any(
                _SOURCE_INCLUDE_RE.fullmatch(statement) or _SOURCE_INCBIN_RE.fullmatch(statement)
                for line in _joined_source_lines(_strip_c_comments(program.source))
                for statement in _split_source_statements(line)
            ):
                # 旧账本把未改写的 P0 按源码 raw SHA 记录；允许 verifier
                # 用这个精确身份重建旧 profile。任何改写变体仍必须带
                # include context，避免把未知依赖误当成同一程序。
                legacy_raw = hashlib.sha256(raw).hexdigest()
                if legacy_raw != program.parent_sha256:
                    raise ValueError("program variant include identity unavailable")
                return legacy_raw
            return _program_identity_digest(raw, includes)
        includes = program._include_context or _PROGRAM_INCLUDE_CONTEXT.get(
            program.parent_sha256, ()
        )
        digest = _program_identity_digest(raw, includes)
        if includes and digest != program.parent_sha256:
            _PROGRAM_INCLUDE_CONTEXT.setdefault(digest, includes)
        _PROGRAM_SOURCE_CONTEXT.setdefault(digest, raw)
        return digest
    else:
        path = getattr(program, "program", None)
        if path is None:
            raise TypeError("program must provide a path or be ProgramVariant")
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix.lower() == ".s":
            files = _source_include_files(path)
            includes = tuple(
                (name, data) for name, data in sorted(files.items())
                if name != path.name
            )
            digest = _program_identity_digest(raw, includes)
            if includes:
                _PROGRAM_INCLUDE_CONTEXT.setdefault(digest, includes)
            _PROGRAM_SOURCE_CONTEXT.setdefault(digest, raw)
            return digest
    return hashlib.sha256(raw).hexdigest()


def _label_prefix(raw: str) -> str:
    return re.match(r"^\s*(?:(?:[A-Za-z_.$][\w.$]*|\d+)\s*:\s*)*", raw).group(0)
_REGISTER_ALIASES = dict(zip(
    ("zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
     "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4",
     "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"),
    range(32),
))
_REGISTER_ALIASES["fp"] = 8


def _register_index(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    if value in _REGISTER_ALIASES:
        return _REGISTER_ALIASES[value]
    if not value.startswith("x") or not value[1:].isdigit():
        return None
    index = int(value[1:])
    return index if index < 32 else None


_FP_ALIASES = dict(zip(
    ("ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1",
     "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3",
     "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9",
     "ft10", "ft11"),
    range(32),
))


def _row_witnesses(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    values = row.get("witnesses", ())
    if isinstance(values, Mapping):
        values = (values,)
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(value, Mapping) for value in values)):
        return ()
    return tuple(values)


def _witness_field_valid(name: str, value: object) -> bool:
    if name in {"before_gpr", "after_gpr"}:
        return _gpr_state(value) is not None
    if name in {"before_fpr_rawbits", "after_fpr_rawbits"}:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 32
            and all(
                (parsed := _int_value(item)) is not None and 0 <= parsed < (1 << 64)
                for item in value
            )
        )
    if name in {"before_fflags", "after_fflags"}:
        return _int_value(value) in range(0x20)
    if name in {"before_frm", "after_frm"}:
        return _int_value(value) in range(0x8)
    if name == "after_pc":
        pc = _pc_value(value)
        return pc is not None and not pc & 1
    if name == "branch_outcome":
        return isinstance(value, str) and value in {"taken", "not-taken"}
    if name == "fault_pc":
        pc = _pc_value(value)
        return pc is not None and not pc & 1
    if name == "fault_address":
        return _pc_value(value) is not None
    if name == "fault":
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return False
        return True
    if name == "memory_reads":
        return isinstance(value, (list, tuple)) and bool(value) and all(
            isinstance(read, Mapping)
            and _pc_value(read.get("address")) is not None
            and _int_value(read.get("width")) is not None
            and _int_value(read["width"]) > 0
            for read in value
        )
    if name == "consumer":
        return (
            isinstance(value, Mapping)
            and isinstance(value.get("kind"), str)
            and bool(value["kind"].strip())
        )
    return value is not None


def _fp_index(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    if value in _FP_ALIASES:
        return _FP_ALIASES[value]
    return int(value[1:]) if value.startswith("f") and value[1:].isdigit() and int(value[1:]) < 32 else None


def _zcmp_stack_operands_valid(
    mnemonic: str, operands: Sequence[str], xlen: int,
    aliases: Mapping[str, object] | None = None,
) -> bool:
    if mnemonic not in {"cm.push", "cm.pop", "cm.popret", "cm.popretz"}:
        return True
    if len(operands) != 2 or type(xlen) is not int:
        return False
    match = re.fullmatch(r"\{([^{}]+)\}", str(operands[0]).strip())
    if match is None:
        return False
    registers = []
    for item in match[1].split(","):
        parts = item.strip().split("-")
        if len(parts) == 1:
            index = _register_index(parts[0])
            if index is None:
                return False
            registers.append(index)
            continue
        if len(parts) != 2:
            return False
        start_name, end_name = (part.strip().lower() for part in parts)
        alias_range = re.fullmatch(r"([sx])(\d+)", start_name), re.fullmatch(
            r"([sx])(\d+)", end_name
        )
        if not alias_range[0] or not alias_range[1] or alias_range[0][1] != alias_range[1][1]:
            return False
        start, end = int(alias_range[0][2]), int(alias_range[1][2])
        if end < start:
            return False
        registers.extend(_register_index(f"{start_name[0]}{index}") for index in range(start, end + 1))
    expected = (1, 8, 9, *range(18, 28))
    size = len(registers)
    if size not in (*range(1, 12), 13) or tuple(registers) != expected[:size]:
        return False
    base, adjustment = _resolve_target(str(operands[1]).strip(), aliases)
    if base is not None:
        return False
    if mnemonic == "cm.push":
        if adjustment >= 0:
            return False
        adjustment = -adjustment
    elif adjustment <= 0:
        return False
    if xlen == 32:
        minimum = 16 if size <= 4 else 32 if size <= 8 else 48 if size <= 11 else 64
    elif xlen == 64:
        minimum = (
            16 if size <= 2 else 32 if size <= 4 else 48 if size <= 6 else
            64 if size <= 8 else 80 if size <= 10 else 96 if size == 11 else 112
        )
    else:
        return False
    return adjustment in range(minimum, minimum + 64, 16)


def _profile_operands_valid(
    mnemonic: str, operands: Sequence[str], aliases: Mapping[str, object] | None = None,
    *, xlen: int | None = None, isa_profile: str | None = None,
) -> bool:
    values = tuple(
        _resolve_memory_alias(str(item).strip(), aliases) for item in operands
    )
    mnemonic = _source_mnemonic(mnemonic)
    def register(value: str, fp: bool = False) -> bool:
        return (_fp_index(value) if fp else _register_index(value)) is not None

    def target(value: str) -> bool:
        return not (
            register(value) or register(value, True)
            or re.fullmatch(r"v\d+(?:\.\w+)?", value)
        )
    def target_expression(value: str) -> bool:
        return target(value) and _target_expression_valid(value)
    def integer(value: str) -> int | None:
        base, resolved = _resolve_target(value, aliases)
        return resolved if base is None else None
    def vector(value: str) -> bool:
        return re.fullmatch(r"v(?:[0-9]|[12][0-9]|3[01])", value) is not None
    def source_gpr(value: str) -> bool:
        return _register_index(value) is not None

    def domain_register(value: str, role: str) -> bool:
        if spec is None:
            return source_gpr(value)
        raw_domain = domain = dict(spec.operand_domains).get(role)
        if domain == "fpr" and isa_profile:
            domain = xregister_compatibility_slot_domain_for_mnemonic(
                isa_profile, mnemonic, role, 0
            )
        index = _register_index(value)
        return register(value, domain == "fpr") and not (
            raw_domain == "fpr" and domain == "gpr"
            and isa_profile and isa_profile.startswith("rv32")
            and "zdinx" in enabled_extensions(isa_profile)
            and fp_operand_width_bits(mnemonic) == 64
            and (index is None or index % 2)
        )

    def scalar_rs1(value: str) -> bool:
        return domain_register(value, "rs1")

    def memory_base(value: str) -> bool:
        match = _MEMORY_BASE_RE.fullmatch(value)
        return match is not None and source_gpr(match[1].strip())

    def zero_memory_base(value: str) -> bool:
        offset = _MEMORY_OPERAND_RE.fullmatch(value)
        return memory_base(value) and (offset is None or integer(offset[1]) == 0)

    def memory_offset(value: str) -> int | None:
        match = _MEMORY_OPERAND_RE.fullmatch(value)
        return integer(match[1]) if match else 0 if _MEMORY_BASE_RE.fullmatch(value) else None

    def vector_type(values: Sequence[str]) -> bool:
        return (
            1 <= len(values) <= 4
            and re.fullmatch(r"e(?:8|16|32|64)", values[0], re.IGNORECASE)
            and (len(values) == 1 or re.fullmatch(r"m(?:1|2|4|8|f2|f4|f8)", values[1], re.IGNORECASE))
            and (len(values) <= 2 or values[2].lower() in {"ta", "tu"})
            and (len(values) <= 3 or values[3].lower() in {"ma", "mu"})
        )

    def compact(value: str, fp: bool = False) -> bool:
        index = _fp_index(value) if fp else _register_index(value)
        return index in range(8, 16)

    spec = instruction_spec_for_generation(mnemonic)
    if spec is not None and mnemonic.startswith("c.") \
            and instruction_effect_class(mnemonic) == "load":
        runtime = runtime_instruction_spec(mnemonic, xlen)
        memory = _memory_parts(values[-1]) if values else None
        if runtime is None or memory is None or memory[0] is None or memory[1] is None:
            return False
        if len(runtime.operand_roles) == 3:
            return _text_encodable(
                runtime, runtime.operand_roles, values,
                aliases, isa_profile=isa_profile,
            )
        if memory[1] != 2:
            return False
        rd = _fp_index(values[0]) if dict(spec.operand_domains).get("rd") == "fpr" \
            else _register_index(values[0])
        if rd is None or any(
            str(field).endswith("_n0") and rd == 0
            for field in spec.form.variable_fields
        ):
            return False
        try:
            runtime.encode(rd, None, None, None, memory[0])
        except (TypeError, ValueError):
            return False
        return True
    kinds = tuple(spec.operand_kinds) if spec is not None else ()
    vector_vd_disjoint = mnemonic.startswith((
        "vwadd.", "vwaddu.", "vwsub.", "vwsubu.", "vwmul.", "vwmulu.",
        "vwmulsu.", "vwmacc.", "vwmaccu.", "vwmaccsu.", "vwmaccus.",
        "vwsll.", "vwabda.", "vwabdau.", "vfwadd.", "vfwsub.",
        "vfwmul.", "vfwmacc.", "vfwnmacc.", "vfwmsac.", "vfwnmsac.",
        "vfwcvt.", "vslideup.", "vslide1up.", "vrgather.",
    ))
    if mnemonic == "c.nop":
        immediate = integer(values[0]) if len(values) == 1 else None
        return not values or immediate in range(-32, 32) and immediate != 0
    if xlen == 32 and mnemonic in {
        "slli", "srli", "srai", "rori", "bseti", "bclri", "binvi", "bexti",
        "c.slli", "c.srli", "c.srai",
    }:
        immediate = integer(values[-1]) if values else None
        if immediate is not None and immediate not in range(32):
            return False
    if mnemonic in {"bltzal", "bgezal"}:
        return False
    if spec is not None and not kinds:
        return not values
    if mnemonic == "fence":
        return not values or len(values) == 2 and all(
            value and re.fullmatch(r"(?:0|i?o?r?w?)", value, re.IGNORECASE)
            for value in values
        )
    if mnemonic == "fence.i":
        return not values
    if mnemonic == "fence.tso":
        return not values
    if spec is not None and spec.form.extension_group in {"h", "svinval_h"}:
        if mnemonic.startswith(("hfence.", "hinval.")):
            return len(values) <= 2 and all(source_gpr(value) for value in values)
        if mnemonic.startswith(("hlv.", "hlvx.", "hsv.")):
            return len(values) == 2 and source_gpr(values[0]) and zero_memory_base(values[1])
    if mnemonic == "jalr":
        if len(values) == 2 and register(values[0]):
            if register(values[1]):
                return True
            offset = memory_offset(values[1])
            return offset is not None and -2048 <= offset < 2048 and memory_base(values[1])
        return len(values) == 3 and register(values[0]) and register(values[1]) \
            and integer(values[2]) in range(-2048, 2048)
    if mnemonic in {"fmv.s.x", "fmv.x.s"}:
        return len(values) == 2 and (
            register(values[0], True) and source_gpr(values[1])
            if mnemonic == "fmv.s.x" else source_gpr(values[0]) and register(values[1], True)
        )
    if mnemonic in {"csrrw", "csrrs", "csrrc"}:
        return (
            len(values) == 3 and source_gpr(values[0])
            and _csr_operand_valid(values[1], aliases, xlen)
            and source_gpr(values[2])
        )
    if mnemonic in {"csrrwi", "csrrsi", "csrrci"}:
        base, immediate = _resolve_target(values[2], aliases) if len(values) == 3 else (None, -1)
        return (
            len(values) == 3 and source_gpr(values[0])
            and _csr_operand_valid(values[1], aliases, xlen)
            and base is None and 0 <= immediate < 32
        )
    if mnemonic == "sfence.vma":
        return len(values) <= 2 and all(source_gpr(value) for value in values)
    if mnemonic.startswith("cbo."):
        return len(values) == 1 and zero_memory_base(values[0])
    if spec is not None and spec.form.extension_group == "zalasr":
        return len(values) == 2 and source_gpr(values[0]) and zero_memory_base(values[1])
    if spec is not None and spec.form.mnemonic == "mop_r_N":
        return len(values) == 2 and all(source_gpr(value) for value in values)
    if spec is not None and spec.form.mnemonic == "mop_rr_N":
        return len(values) == 3 and all(source_gpr(value) for value in values)
    if spec is not None and spec.form.mnemonic == "c_mop_N":
        return not values
    if mnemonic in {"sb", "sh", "sw", "sd", "fsw", "fsd"} and len(values) == 3:
        return register(values[0], mnemonic.startswith("f")) \
            and target_expression(values[1]) and _integer_expression(values[1]) is None \
            and register(values[2])
    if spec is not None and mnemonic.startswith(("amo", "lr.", "sc.")):
        memory_index = 1 if mnemonic.startswith("lr.") else 2
        return len(values) == memory_index + 1 and zero_memory_base(values[memory_index]) \
            and all(source_gpr(value) for value in values[:memory_index])
    if match := re.fullmatch(r"vl([1248])re(?:8|16|32|64)\.v", mnemonic):
        register = _int_value(values[0][1:]) if len(values) == 2 and vector(values[0]) else None
        return register is not None and register % int(match[1]) == 0 and zero_memory_base(values[1])
    if match := re.fullmatch(r"vs([1248])r\.v", mnemonic):
        register = _int_value(values[0][1:]) if len(values) == 2 and vector(values[0]) else None
        return register is not None and register % int(match[1]) == 0 and zero_memory_base(values[1])
    if mnemonic in {"vlm.v", "vsm.v"}:
        return len(values) == 2 and vector(values[0]) and zero_memory_base(values[1])
    if mnemonic in {"vmadc.vv", "vmsbc.vv"}:
        return len(values) == 3 and all(vector(value) for value in values)
    if mnemonic in {"vadc.vvm", "vsbc.vvm", "vmadc.vvm", "vmsbc.vvm"}:
        return len(values) == 4 and all(vector(value) for value in values[:3]) and values[3].lower() == "v0"
    if mnemonic in {"vmand.mm", "vmandn.mm", "vmnand.mm", "vmnor.mm", "vmor.mm", "vmorn.mm", "vmxnor.mm", "vmxor.mm"}:
        return len(values) == 3 and all(vector(value) for value in values)
    if mnemonic.startswith(("vaes", "vghsh.", "vgmul.", "vsha2", "vsm3", "vsm4")) \
            and all(kind in {"vs1", "vs2", "vd"} for kind in kinds):
        return len(values) == len(kinds) and all(vector(value) for value in values)
    if kinds == ("vm", "vs2", "vs1", "vd"):
        return len(values) in (3, 4) and all(vector(value) for value in values[:3]) \
            and (not vector_vd_disjoint or values[0] not in values[1:3]) \
            and (len(values) == 3 or values[3].lower() == "v0.t")
    if kinds == ("vm", "vs2", "rs1", "vd"):
        rs1_first = mnemonic in {
            "vmacc.vx", "vnmsac.vx", "vmadd.vx", "vnmsub.vx",
            "vwmacc.vx", "vwmaccu.vx", "vwmaccsu.vx", "vwmaccus.vx",
            "vfmacc.vf", "vfnmacc.vf", "vfmsac.vf", "vfnmsac.vf",
            "vfmadd.vf", "vfnmadd.vf", "vfmsub.vf", "vfnmsub.vf",
            "vfwmacc.vf", "vfwmaccbf16.vf", "vfwnmacc.vf", "vfwmsac.vf", "vfwnmsac.vf",
        }
        left, right = (scalar_rs1, vector) if rs1_first else (vector, scalar_rs1)
        return len(values) in (3, 4) and vector(values[0]) and left(values[1]) \
            and right(values[2]) \
            and (not vector_vd_disjoint or values[0] not in values[1:3]) \
            and (len(values) == 3 or values[3].lower() == "v0.t")
    if kinds == ("vs2", "rs1", "vd"):
        masked = mnemonic.endswith(("vxm", "vfm"))
        return len(values) == 3 + masked and vector(values[0]) and vector(values[1]) \
            and scalar_rs1(values[2]) and (not masked or values[3].lower() == "v0")
    if mnemonic.startswith(("vaes", "vsm3", "vsm4")) and kinds == ("vs2", "imm", "vd"):
        immediate = integer(values[1]) if len(values) >= 3 else None
        groups = set(operand_groups(spec.form))
        valid_imm = immediate in range(32) if "zimm5" in groups else immediate in range(-16, 16)
        return len(values) == 3 and vector(values[0]) and valid_imm and vector(values[2])
    if kinds == ("vs2", "imm", "vd"):
        immediate = integer(values[2]) if len(values) >= 3 else None
        groups = set(operand_groups(spec.form))
        valid_imm = immediate in range(32) if "zimm5" in groups else immediate in range(-16, 16)
        masked = mnemonic.endswith("vim")
        return len(values) == 3 + masked and vector(values[0]) and vector(values[1]) \
            and valid_imm and (not masked or values[3].lower() == "v0")
    if kinds == ("vm", "vs2", "vd"):
        return len(values) in (2, 3) and vector(values[0]) and vector(values[1]) \
            and (not vector_vd_disjoint or values[0] != values[1]) \
            and (len(values) == 2 or values[2].lower() == "v0.t")
    if kinds == ("vs1", "vd"):
        return len(values) == 2 and vector(values[0]) and vector(values[1])
    if kinds == ("vs2", "rd"):
        return len(values) == 2 and domain_register(values[0], "rd") and vector(values[1])
    if kinds == ("rs1", "vd") and mnemonic in {"vmv.s.x", "vmv.v.x", "vfmv.s.f", "vfmv.v.f"}:
        return len(values) == 2 and vector(values[0]) and scalar_rs1(values[1])
    if kinds == ("vm", "vs2", "rd"):
        return len(values) in (2, 3) and domain_register(values[0], "rd") \
            and vector(values[1]) and (len(values) == 2 or values[2].lower() == "v0.t")
    if kinds == ("vm", "vd"):
        return len(values) in (1, 2) and vector(values[0]) \
            and (len(values) == 1 or values[1].lower() == "v0.t")
    if match := re.fullmatch(r"vmv([1248])r\.v", mnemonic):
        count = int(match[1])
        registers = tuple(_int_value(value[1:]) if vector(value) else None for value in values)
        return len(registers) == 2 and all(value is not None and value % count == 0 for value in registers)
    if kinds in {("vm", "vs2", "imm", "vd"), ("imm", "vm", "vs2", "vd")}:
        immediate = integer(values[2]) if len(values) >= 3 else None
        groups = set(operand_groups(spec.form))
        valid_imm = immediate in range(64) if "zimm6" in groups or {"zimm6hi", "zimm6lo"} <= groups \
            else immediate in range(-16, 16) if "simm5" in groups else immediate in range(32)
        return len(values) in (3, 4) and vector(values[0]) and vector(values[1]) \
            and valid_imm and (not vector_vd_disjoint or values[0] != values[1]) \
            and (len(values) == 3 or values[3].lower() == "v0.t")
    if mnemonic == "vcompress.vm":
        return len(values) == 3 and all(vector(value) for value in values) and values[0] not in values[1:]
    if mnemonic == "vmerge.vvm":
        return len(values) == 4 and all(vector(value) for value in values[:3]) and values[3].lower() == "v0"
    if mnemonic == "vmv.v.i":
        immediate = integer(values[1]) if len(values) == 2 else None
        return len(values) == 2 and vector(values[0]) and immediate in range(-16, 16)
    if mnemonic.startswith(("vluxei", "vloxei", "vsuxei", "vsoxei")):
        return len(values) in (3, 4) and vector(values[0]) and zero_memory_base(values[1]) \
            and vector(values[2]) and (len(values) == 3 or values[3].lower() == "v0.t")
    if kinds in {
        ("nf", "vm", "rs1", "vd"), ("nf", "vm", "rs1", "vs3"),
    }:
        return len(values) in (2, 3) and vector(values[0]) and zero_memory_base(values[1]) \
            and (len(values) == 2 or values[2].lower() == "v0.t")
    if kinds in {
        ("nf", "vm", "rs2", "rs1", "vd"),
        ("nf", "vm", "rs2", "rs1", "vs3"),
    }:
        return len(values) in (3, 4) and vector(values[0]) and zero_memory_base(values[1]) \
            and source_gpr(values[2]) and (len(values) == 3 or values[3].lower() == "v0.t")
    if mnemonic == "vsetvl":
        return len(values) == 3 and all(source_gpr(value) for value in values)
    if mnemonic == "vsetvli":
        return len(values) >= 3 and source_gpr(values[0]) and source_gpr(values[1]) and (
            len(values[2:]) == 1 and integer(values[2]) in range(2048)
            or vector_type(values[2:])
        )
    if mnemonic == "vsetivli":
        return len(values) >= 3 and source_gpr(values[0]) and integer(values[1]) in range(32) \
            and vector_type(values[2:])
    if spec is not None and instruction_effect_class(mnemonic) == "store" and mnemonic != "cm.push":
        if len(values) != 2:
            return False
        domains = dict(spec.operand_domains)
        is_fp = domains.get("rs2") == "fpr" and not (
            isa_profile and xregister_compatibility_slot_domain_for_mnemonic(
                isa_profile, mnemonic, "rs2", 0
            ) == "gpr"
        )
        base = _MEMORY_BASE_RE.fullmatch(values[1])
        if base is None or not source_gpr(base[1].strip()):
            return False
        offset = memory_offset(values[1])
        if offset is None or not -2048 <= offset < 2048:
            return False
        compressed_limit = {
            "c.sw": 128, "c.swsp": 256, "c.sd": 256, "c.sdsp": 512,
            "c.fsw": 128, "c.fswsp": 256, "c.fsd": 256, "c.fsdsp": 512,
        }.get(mnemonic)
        if compressed_limit is not None:
            width = 8 if "sd" in mnemonic else 4
            if not 0 <= offset < compressed_limit or offset % width:
                return False
        data = _fp_index(values[0]) if is_fp else _register_index(values[0])
        if data is None:
            return False
        if mnemonic.startswith("c."):
            try:
                encode_instruction_for_generation(
                    mnemonic,
                    **{
                        **({"rs1": _register_index(base[1].strip())}
                           if "rs1" in spec.operand_kinds else {}),
                        **({"rs2": data} if "rs2" in spec.operand_kinds else {}),
                        "immediate": offset,
                    },
                )
            except (TypeError, ValueError):
                return False
        if implicit_source_roles(spec.form):
            return _register_index(base[1].strip()) == 2
        if any(str(field).endswith("_p") for field in spec.form.variable_fields):
            return compact(values[0], is_fp) and compact(base[1].strip())
        return True
    if mnemonic in {"cm.mva01s", "cm.mvsa01"}:
        values = tuple(_register_index(value) for value in values)
        allowed = {8, 9, *range(18, 24)}
        return len(values) == 2 and all(value in allowed for value in values) and (
            mnemonic == "cm.mva01s" or values[0] != values[1]
        )
    if mnemonic in {"cm.jt", "cm.jalt"}:
        base, index = _resolve_target(values[0], aliases) if len(values) == 1 else (None, -1)
        return len(values) == 1 and base is None and (
            0 <= index < 32 if mnemonic == "cm.jt" else 32 <= index < 256
        )
    if mnemonic in {"frcsr", "frflags", "frrm"}:
        return len(values) == 1 and register(values[0])
    if mnemonic == "fscsr":
        return len(values) in {1, 2} and all(register(value) for value in values)
    if mnemonic == "fsrm":
        return len(values) == 2 and register(values[0]) and register(values[1])
    if mnemonic == "fsrmi":
        return (
            len(values) == 1 and integer(values[0]) in range(32)
            or len(values) == 2 and register(values[0]) and integer(values[1]) in range(32)
        )
    if mnemonic in _FP_PSEUDOS:
        fp = not (isa_profile and bool(x_register_fp_extensions(isa_profile)))
        return len(values) == 2 and all(register(value, fp) for value in values) and (
            fp or not (
                mnemonic.endswith(".d") and isa_profile and isa_profile.startswith("rv32")
                and "zdinx" in enabled_extensions(isa_profile)
            ) or all(
                (index := _register_index(value)) is not None and index % 2 == 0
                for value in values
            )
        )
    if mnemonic in {"la", "lla", "lga"}:
        return len(values) == 2 and register(values[0]) and target_expression(values[1])
    if mnemonic == "li":
        base, immediate = _resolve_target(values[1], aliases) if len(values) == 2 else ("", 0)
        return len(values) == 2 and register(values[0]) and base is None and (
            xlen not in {32, 64}
            or xlen == 32 and -(1 << 31) <= immediate < (1 << 32)
            or xlen == 64 and -(1 << 64) < immediate < (1 << 64)
        )
    if mnemonic in {
        "mv", "not", "neg", "negw", "sext.b", "sext.h", "sext.w", "zext.b", "zext.h",
        "zext.w", "seqz", "snez", "sltz", "sgtz",
    }:
        return len(values) == 2 and all(register(value) for value in values)
    if mnemonic in {"rdcycle", "rdtime", "rdinstret"}:
        return len(values) == 1 and register(values[0])
    if mnemonic == "csrr":
        return len(values) == 2 and register(values[0]) and _csr_operand_valid(values[1], aliases, xlen)
    if mnemonic in {"csrw", "csrs", "csrc"}:
        return len(values) == 2 and _csr_operand_valid(values[0], aliases, xlen) and register(values[1])
    if mnemonic in {"csrwi", "csrsi", "csrci"}:
        base, immediate = _resolve_target(values[1], aliases) if len(values) == 2 else (None, -1)
        return len(values) == 2 and _csr_operand_valid(values[0], aliases, xlen) and base is None and 0 <= immediate < 32
    if mnemonic in {"beqz", "bnez", "bltz", "bgez", "bltzal", "bgezal", "bgtz", "blez"}:
        return len(values) == 2 and register(values[0]) and target(values[1])
    if mnemonic in {"bgt", "ble", "bgtu", "bleu"}:
        return len(values) == 3 and register(values[0]) and register(values[1]) and target(values[2])
    if mnemonic in {"beqi", "bnei"}:
        base, immediate = _resolve_target(values[1], aliases) if len(values) == 3 else (None, -1)
        return (
            len(values) == 3 and register(values[0]) and base is None
            and -1 <= immediate <= 31 and target(values[2])
        )
    if mnemonic == "jr":
        if len(values) != 1:
            return False
        if register(values[0]):
            return True
        memory = _MEMORY_OPERAND_RE.fullmatch(values[0])
        if memory:
            immediate = _int_value(memory[1])
            return source_gpr(memory[2].strip()) and immediate is not None and -2048 <= immediate < 2048
        base = _MEMORY_BASE_RE.fullmatch(values[0])
        return base is not None and source_gpr(base[1].strip())
    if mnemonic in {"j", "call", "tail"}:
        return len(values) == 1 and target_expression(values[0])
    if spec is not None and mnemonic.startswith("v"):
        return False
    return True


def _csr_operand_valid(
    value: str, aliases: Mapping[str, object] | None = None, xlen: int | None = None,
) -> bool:
    base, resolved = _resolve_target(value, aliases)
    if base is None:
        return 0 <= resolved <= 0xFFF
    from ..riscv_csrs import OFFICIAL_CSRS, OFFICIAL_RV32_ONLY_CSRS
    names = OFFICIAL_CSRS.values()
    if xlen != 64:
        names = (*names, *OFFICIAL_RV32_ONLY_CSRS.values())
    return resolved == 0 and base.lower() in {
        str(name).strip('"').lower() for name in names
    }


def _profile_action(
    row: Mapping[str, object], profile: object, rule: str, text: str,
    occurrence_id: str | None = None, occurrence_ids: Sequence[str] = (),
    *, probe: bool = False, witness_reason: str | None = None,
    profile_digest: str | None = None,
) -> dict[str, object]:
    if probe and (not isinstance(witness_reason, str) or not witness_reason.strip()):
        raise ValueError("probe action requires witness_reason")
    if not probe:
        witness_reason = None
    if occurrence_id is None:
        occurrence_id = next((item.get("id") for item in getattr(profile, "occurrences", ())
                              if item.get("instruction_id") == row["id"]), None)
    anchor = {
        "instruction_id": row["id"], "occurrence_id": occurrence_id,
        "block": row["block"], "source_start": row["start"],
        "source_end": row["end"], "pc": row["pc"],
    }
    if row.get("final_pcs") is not None:
        anchor["final_pcs"] = tuple(row["final_pcs"])
    if row.get("final_sizes") is not None:
        anchor["final_sizes"] = tuple(row["final_sizes"])
    if row.get("final_bytes") is not None:
        anchor["final_bytes"] = tuple(row["final_bytes"])
    if rule == "R2" or len(occurrence_ids) > 1:
        anchor.update({"occurrence_scope": "all", "occurrence_ids": list(occurrence_ids)})
    facts = row.get("rule_facts", {})
    target = text.split(None, 1)[0].lower().replace("_", ".")
    result_width = facts.get("result_width")
    if (facts.get("effect") in {"gpr", "private-load", "gpr-from-fpr"}
            and row.get("xlen") == 32 and result_width == 64):
        result_width = 32
    occurrence_scope = "all" if rule == "R2" or len(occurrence_ids) > 1 else "single"
    evidence = {
        "definedness": facts.get("definedness"),
        "effect": facts.get("effect"),
        "operand_roles": list(facts.get("operand_roles", ())),
        "source_projections": [list(item) for item in facts.get("source_projections", ())],
        "xlen": row.get("xlen"),
        "result_width": result_width,
        "signedness": facts.get("signedness"),
        "required_trace_fields": sorted(_RULE_TRACE_FIELDS.get(rule, ())),
        "occurrence_scope": occurrence_scope,
    }
    if rule == "R3":
        evidence["permutation_pairs"] = [list(item) for item in facts.get("permutation_pairs", ())]
    if rule == "R9":
        evidence["successors"] = [
            {
                "occurrence_id": witness.get("id"),
                "pc": witness.get("pc"),
                "after_pc": witness.get("after_pc"),
                "branch_outcome": witness.get("branch_outcome"),
                "link": witness.get("link"),
                "ialign": witness.get("ialign"),
                "fault": witness.get("fault"),
            }
            for witness in _row_witnesses(row)
        ]
    if rule == "R11":
        evidence["consumers"] = [
            dict(witness["consumer"])
            for witness in _row_witnesses(row)
            if isinstance(witness.get("consumer"), Mapping)
        ]
    return {
        "kind": "local-rewrite",
        "anchor": anchor,
        "rule": rule,
        "occurrence_scope": occurrence_scope,
        "applicable": True,
        "ready": not probe,
        "witness_status": "probe" if probe else "ready",
        "evidence_level": "probe" if probe else "guided",
        "source_identity": profile.source_sha256,
        "payload": {"source": text},
        "rule_evidence": evidence,
        "profile_digest": profile_digest or profile.profile_digest,
        "source_cell": row.get("translation_cell", f"scalar:{row['mnemonic']}"),
        "target_cell": f"scalar:{target}",
        **({"witness_reason": witness_reason} if witness_reason else {}),
    }


def _dead_block_action(
    rows: tuple[Mapping[str, object], ...], profile: object,
    *, profile_digest: str | None = None,
) -> dict[str, object]:
    block = rows[0]["block"]
    anchor = {
        "block": block,
        "instruction_ids": tuple(row["id"] for row in rows),
        "source_start": min(row["start"] for row in rows),
        "source_end": max(row["end"] for row in rows),
    }
    for name in ("final_pcs", "final_sizes", "final_bytes"):
        values = tuple(value for row in rows for value in row.get(name, ()))
        if values:
            anchor[name] = values
    return {
        # M5 is still materialized by the dedicated dead-block path below,
        # but it is an EMI local rewrite at the campaign-contract boundary.
        # Keep the concrete mutation kind separately so the chain auditor can
        # require ``action.kind=local-rewrite`` without losing M5 semantics.
        "kind": "local-rewrite",
        "mutation_kind": "block-delete",
        "anchor": anchor,
        "rule": "M5",
        "occurrence_scope": "block",
        "evidence_level": "guided",
        "source_identity": profile.source_sha256,
        "rule_evidence": {
            "definedness": "profile-known",
            "effect": "dead-block",
            "operand_roles": [],
            "source_projections": [],
            "xlen": profile.input_identity.get("xlen"),
            "result_width": None,
            "signedness": None,
            "required_trace_fields": [],
            "required_profile_fields": [
                "executed_ids", "blocks", "occurrences", "input_identity",
                "execution_trace",
            ],
            "occurrence_scope": "block",
            "witness": {
                "all_unexecuted": True,
                "no_observed_incoming_edge": True,
                "entry_exit_protected": True,
                "trace_complete": True,
                "layout_preserving": True,
            },
        },
        "payload": {"source": ""},
        "profile_digest": profile_digest or profile.profile_digest,
        "source_cell": "block:dead",
        "target_cell": "block:deleted",
    }


def _program_growth_action(
    row: Mapping[str, object], profile: object, *, profile_digest: str | None = None,
) -> dict[str, object]:
    """Insert an unreachable architectural no-op into a complete program.

    The anchor is an unexecuted row in a block with no observed incoming edge,
    so the reference path and its observed state stay unchanged while the
    program state gains a real instruction.  Subsequent profiles can choose a
    new anchor and continue the chain.
    """
    return {
        # The campaign contract records every EMI transition as a
        # ``local-rewrite``; the mutation kind carries the structural growth.
        "kind": "local-rewrite",
        "operator": "insert",
        "mutation_kind": "instruction-insert",
        "anchor": {
            "instruction_id": row["id"],
            "block": row["block"],
            "line": row["line"],
            "source_start": row["start"],
            "source_end": row["end"],
            "pc": row["pc"],
        },
        "rule": "GROW",
        "occurrence_scope": "none",
        "evidence_level": "guided",
        "source_identity": profile.source_sha256,
        "rule_evidence": {
            "definedness": "profile-known",
            "effect": "structural-growth",
            "operand_roles": [],
            "source_projections": [],
            "xlen": profile.input_identity.get("xlen"),
            "result_width": None,
            "signedness": None,
            "required_trace_fields": [],
            "required_profile_fields": [
                "instructions", "blocks", "edges", "execution_trace",
            ],
            "occurrence_scope": "none",
            "witness": {
                "anchor_executed": False,
                "block_has_incoming_edge": False,
                "semantic_effect": "addi x0, x0, 0",
            },
        },
        "profile_digest": profile_digest or profile.profile_digest,
        "payload": {"source": f"addi x0, x0, 0 # {_PROGRAM_GROWTH_MARKER}"},
        "source_cell": "program:dead-code",
        "target_cell": "program:dead-code+noop",
    }


def _program_growth_delete_action(
    row: Mapping[str, object], source_line: str, profile: object,
    *, profile_digest: str | None = None,
) -> dict[str, object]:
    line = row["line"]
    return {
        "kind": "local-rewrite",
        "operator": "delete_line",
        "mutation_kind": "instruction-delete",
        "source_span": {
            "start": int(line) + 1, "end": int(line) + 1,
            "start_col": 0,
            "end_col": len(source_line.rstrip("\r\n")),
        },
        "rule": "GROW",
        "occurrence_scope": "none",
        "evidence_level": "guided",
        "source_identity": profile.source_sha256,
        "rule_evidence": {
            "definedness": "profile-known",
            "effect": "structural-growth-undo",
            "operand_roles": [], "source_projections": [],
            "xlen": profile.input_identity.get("xlen"),
            "result_width": None, "signedness": None,
            "required_trace_fields": [],
            "required_profile_fields": ["instructions", "execution_trace"],
            "occurrence_scope": "none",
            "witness": {"inserted_noop_marker": _PROGRAM_GROWTH_MARKER},
        },
        "profile_digest": profile_digest or profile.profile_digest,
        "payload": {"marker": _PROGRAM_GROWTH_MARKER},
        "source_cell": "program:dead-code+noop",
        "target_cell": "program:dead-code",
    }


def _profile_incoming_edges(profile: object) -> set[tuple[object, object]]:
    occurrences = tuple(getattr(profile, "occurrences", ()))
    dynamic = {
        (left.get("block"), right.get("block"))
        for left, right in zip(occurrences, occurrences[1:])
        if left.get("block") != right.get("block")
    }
    rows = tuple(getattr(profile, "instructions", ()))
    def static_target_survives(block: object) -> bool:
        target_rows = tuple(row for row in rows if row.get("block") == block)
        if not target_rows or not any(row.get("labels") for row in target_rows):
            return False
        section = target_rows[0].get("section")
        end = max(_pc_value(row.get("pc_end")) or 0 for row in target_rows)
        return any(
            row.get("block") != block and row.get("section") == section
            and (_pc_value(row.get("pc")) or 0) >= end
            for row in rows
        )
    return dynamic | {
        tuple(edge) for edge in getattr(profile, "edges", ())
        if tuple(edge) not in dynamic and not static_target_survives(tuple(edge)[1])
    }


def _m5_trace_complete(profile: object) -> bool:
    execution = getattr(profile, "provenance", {}).get("execution_trace", {})
    return isinstance(execution, Mapping) and execution.get("complete") is True


def _m5_layout_safe(profile: object, rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows or len({row.get("section") for row in rows}) != 1:
        return False
    section = rows[0].get("section")

    def span(row: Mapping[str, object]) -> tuple[int, int]:
        pcs, sizes = row.get("final_pcs"), row.get("final_sizes")
        if (isinstance(pcs, (list, tuple)) and isinstance(sizes, (list, tuple))
                and len(pcs) == len(sizes) and pcs):
            parsed = tuple(_pc_value(pc) for pc in pcs)
            if all(pc is not None and type(size) is int for pc, size in zip(parsed, sizes)):
                return min(parsed), max(pc + size for pc, size in zip(parsed, sizes))
        return int(row["pc"]), int(row["pc_end"])

    dead_start = min(span(row)[0] for row in rows)
    executed_end = max(
        (span(row)[1] for row in profile.instructions
         if row.get("executed") and row.get("section") == section),
        default=-1,
    )
    return dead_start >= executed_end


def _profile_input_matches(program: object, profile: object) -> bool:
    try:
        program_identity = _stable_input_identity(getattr(program, "run_params", {}))
        profile_identity = _stable_input_identity(profile.input_identity)
    except (TypeError, ValueError):
        return False
    for name in ("entry", "exit", "harness", "result_channel"):
        if name not in program_identity:
            profile_identity.pop(name, None)
    return program_identity == profile_identity


def _profile_source_spans_valid(program: object, profile: object) -> bool:
    try:
        raw_source = _strip_c_comments(_source(program))
    except (OSError, TypeError, ValueError, UnicodeError):
        return False
    expanded = (
        _expand_asm_macros(raw_source)
        if getattr(program, "run_params", {}).get("harness") == "custom"
        else raw_source
    )
    source = expanded
    lines = expanded.splitlines(keepends=True)
    starts, offset = [], 0
    for raw in lines:
        starts.append(offset)
        offset += len(raw)
    aliases = {}

    def collect_aliases(source_lines: Sequence[str]) -> None:
        for raw in source_lines:
            parts = raw.strip().split(None, 1)
            if parts and parts[0].lower() in {".equ", ".set"} and len(parts) > 1:
                operands = _split_operands(parts[1])
                if len(operands) >= 2:
                    aliases[operands[0]] = operands[1]

    collect_aliases(lines)
    include_context = tuple(getattr(program, "_include_context", ()) or ())
    if not include_context and getattr(program, "program", None) is not None:
        try:
            include_context = _PROGRAM_INCLUDE_CONTEXT.get(program_sha256(program), ())
        except (OSError, TypeError, ValueError):
            include_context = ()
    for _name, data in include_context:
        try:
            collect_aliases(_strip_c_comments(data.decode("utf-8")).splitlines())
        except UnicodeError:
            continue
    for row in profile.instructions:
        line, start, end = row.get("line"), row.get("start"), row.get("end")
        if (type(line) is not int or type(start) is not int or type(end) is not int
                or line < 0 or line >= len(lines) or start < starts[line]
                or end <= start or end > offset):
            return False
        statement = re.sub(r"\\(?:\r\n|\n)", " ", source[start:end]).strip()
        parts = statement.split(None, 1)
        if not parts:
            return False
        operands = _split_operands(parts[1]) if len(parts) > 1 else ()
        matches = parts[0].lower().replace("_", ".") == row.get("mnemonic") \
            and tuple(operands) == tuple(row.get("operands", ()))
        if not matches and expanded != source:
            matches = any(
                (expanded_parts := line.strip().split(None, 1))
                and expanded_parts[0].lower().replace("_", ".") == row.get("mnemonic")
                and tuple(_split_operands(expanded_parts[1]))
                == tuple(row.get("operands", ()))
                for line in expanded.splitlines()
            )
        if not matches:
            return False
    for row in profile.instructions:
        facts = row.get("rule_facts", {})
        if (isinstance(facts, Mapping) and facts.get("control_shape") == "conditional"
                and "target_pc" in row
                and _pc_value(row["target_pc"]) != _branch_target_pc(
                    row, profile.instructions, aliases=row.get("_aliases", aliases)
                )):
            return False
    return True


def _gpr_use_def(row: Mapping[str, object]) -> tuple[set[int], set[int]] | None:
    facts = row.get("rule_facts", {})
    if not isinstance(facts, Mapping) or facts.get("definedness") != "catalog-known":
        return None
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    uses, defines = set(), set()
    for role, operand in zip(roles, operands):
        if role == "rd":
            if (register := _register_index(operand)) is not None:
                defines.add(register)
            continue
        memory = _memory_parts(str(operand))
        register = memory[1] if memory else _register_index(operand)
        if register is not None:
            uses.add(register)
    spec = runtime_instruction_spec(str(row.get("mnemonic", "")), row.get("xlen"))
    implicit = implicit_source_roles(spec.shared_spec.form) if spec is not None else ()
    if "rs1" in implicit and "rs1" not in roles:
        uses.add(2)
    elif spec is not None and "rs1" in _encoding_source_roles(spec.shared_spec.form) \
            and "rs1" not in roles and "rd" in roles:
        register = _register_index(operands[roles.index("rd")])
        if register is not None:
            uses.add(register)
    if str(row.get("mnemonic", "")).lower().replace("_", ".") in {
        "c.jal", "c.jalr", "cm.jalt", "call",
    } or str(row.get("mnemonic", "")).lower().replace("_", ".") == "jal" \
            and len(operands) == 1:
        defines.add(1)
    return uses, defines


def _profile_gpr_consumer_cache(
    rows: Sequence[Mapping[str, object]], occurrences: Sequence[Mapping[str, object]],
) -> dict[str, bool] | None:
    rows_by_id = {
        row.get("id"): row for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    if len(rows_by_id) != len(rows) or any(
        not isinstance(item, Mapping) or item.get("instruction_id") not in rows_by_id
        for item in occurrences
    ):
        return None
    if any(
        not isinstance(row.get("rule_facts"), Mapping)
        or row["rule_facts"].get("definedness") != "catalog-known"
        for row in rows
    ):
        return None
    future = [None] * len(occurrences)
    next_use: list[bool | None] = [None] * 32
    for index in range(len(occurrences) - 1, -1, -1):
        future[index] = tuple(next_use)
        uses, defines = _gpr_use_def(rows_by_id[occurrences[index]["instruction_id"]])
        for register in uses:
            next_use[register] = True
        for register in defines - uses:
            next_use[register] = False
    result: dict[str, bool] = {}
    for index, occurrence in enumerate(occurrences):
        row = rows_by_id[occurrence["instruction_id"]]
        facts = row.get("rule_facts", {})
        roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
        operands = tuple(row.get("operands", ()))
        if "rd" not in roles or len(roles) != len(operands):
            continue
        rd = _register_index(operands[roles.index("rd")])
        if rd is not None:
            result[row["id"]] = result.get(row["id"], False) or future[index][rd] is True
    return result


def _has_static_gpr_consumer(
    row: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    profile: object | None = None, consumer_cache: Mapping[str, bool] | None = None,
) -> bool:
    facts = row.get("rule_facts", {})
    roles = tuple(facts.get("operand_roles", ())) if isinstance(facts, Mapping) else ()
    operands = tuple(row.get("operands", ()))
    if "rd" not in roles or len(roles) != len(operands):
        return False
    rd = _register_index(operands[roles.index("rd")])
    if rd is None:
        return False
    if consumer_cache is not None and isinstance(row.get("id"), str):
        return bool(consumer_cache.get(row["id"], False))

    def uses_or_defines(candidate: Mapping[str, object]) -> tuple[bool, bool]:
        candidate_facts = candidate.get("rule_facts", {})
        mnemonic = str(candidate.get("mnemonic", "")).lower().replace("_", ".")
        if (not isinstance(candidate_facts, Mapping)
                or candidate_facts.get("definedness") != "catalog-known"):
            operands = tuple(candidate.get("operands", ()))
            uses, defines = set(), set()

            def add_use(value: object) -> None:
                if (register := _register_index(value)) is not None:
                    uses.add(register)

            def add_define(value: object) -> None:
                if (register := _register_index(value)) is not None:
                    defines.add(register)

            if mnemonic in {
                "beqz", "bnez", "bltz", "bgez", "bgtz", "blez",
                "bltzal", "bgezal",
            }:
                add_use(operands[0] if operands else None)
                if mnemonic in {"bltzal", "bgezal"}:
                    defines.add(1)
            elif mnemonic in {"bgt", "ble", "bgtu", "bleu"}:
                for value in operands[:2]:
                    add_use(value)
            elif mnemonic in {"beqi", "bnei", "jr"}:
                add_use(operands[0] if operands else None)
            elif mnemonic == "ret":
                uses.add(1)
            elif mnemonic == "call":
                if rd != 1:
                    return True, False
                defines.add(1)
            elif mnemonic == "tail":
                if rd != 6:
                    return True, False
                defines.add(6)
            elif mnemonic in {"li", "la", "lla", "lga"}:
                add_define(operands[0] if operands else None)
            elif mnemonic in {
                "mv", "not", "neg", "negw", "sext.b", "sext.h", "sext.w",
                "zext.b", "zext.h", "zext.w", "seqz", "snez", "sltz", "sgtz",
            }:
                add_define(operands[0] if operands else None)
                add_use(operands[1] if len(operands) > 1 else None)
            elif mnemonic in {"rdcycle", "rdtime", "rdinstret", "rdcycleh", "rdtimeh", "rdinstreth"}:
                add_define(operands[0] if operands else None)
            elif mnemonic == "csrr":
                add_define(operands[0] if operands else None)
            elif mnemonic in {"csrw", "csrs", "csrc"}:
                add_use(operands[1] if len(operands) > 1 else None)
            else:
                return True, False
            return rd in uses, rd in defines
        candidate_roles = tuple(
            candidate_facts.get("operand_roles", ())
        ) if isinstance(candidate_facts, Mapping) else ()
        candidate_operands = tuple(candidate.get("operands", ()))
        for role, operand in zip(candidate_roles, candidate_operands):
            if role == "rd":
                continue
            memory = _memory_parts(str(operand))
            value = memory[1] if memory else _register_index(operand)
            if value == rd:
                return True, False
        return False, any(
            role == "rd" and _register_index(operand) == rd
            for role, operand in zip(candidate_roles, candidate_operands)
        )

    occurrences = tuple(getattr(profile, "occurrences", ())) if profile is not None else ()
    if occurrences:
        rows_by_id = {
            item.get("id"): item for item in rows if isinstance(item, Mapping)
        }
        producer_indices = [
            index for index, item in enumerate(occurrences)
            if item.get("instruction_id") == row.get("id")
        ]
        if producer_indices:
            for start in producer_indices:
                for occurrence in occurrences[start + 1:]:
                    candidate = rows_by_id.get(occurrence.get("instruction_id"))
                    if candidate is None:
                        return True
                    used, defined = uses_or_defines(candidate)
                    if used:
                        return True
                    if defined:
                        break
            return False
    try:
        start = next(index for index, item in enumerate(rows) if item.get("id") == row.get("id")) + 1
    except StopIteration:
        return False
    for candidate in rows[start:]:
        if candidate.get("executed", True) is False:
            continue
        used, defined = uses_or_defines(candidate)
        if used:
            return True
        if defined:
            return False
    return False


def _profile_actions(program: object, profile: object) -> tuple[dict[str, object], ...]:
    """由 ProgramProfile 生成 canonical guided/probe actions。"""
    if (not getattr(profile, "complete", False)
            or program_sha256(program) != profile.source_sha256
            or not _profile_input_matches(program, profile)
            or not _profile_source_spans_valid(program, profile)):
        return ()
    if any(
        not isinstance(row, Mapping)
        or any(
            field not in row or type(row[field]) is not int
            for field in ("line", "start", "end")
        )
        or not isinstance(row.get("mnemonic"), str)
        or not isinstance(row.get("operands"), (list, tuple))
        or row["line"] < 0 or row["start"] < 0 or row["end"] <= row["start"]
        or not (
            isinstance((facts := row.get("rule_facts")), Mapping)
            and {key: value for key, value in facts.items() if key != "relations"}
            == _rule_facts(str(row.get("mnemonic", "")), row.get("xlen"))
        )
        for row in profile.instructions
    ):
        return ()
    result = []
    profile_digest = profile.profile_digest
    layout_ready = _profile_layout_ready(profile)
    occurrence_ids = {}
    for item in profile.occurrences:
        instruction_id = item.get("instruction_id")
        occurrence_id = item.get("id")
        if isinstance(instruction_id, str) and isinstance(occurrence_id, str) and occurrence_id:
            occurrence_ids.setdefault(instruction_id, []).append(occurrence_id)
    for row in profile.instructions:
        if not row.get("executed"):
            continue
        facts = row.get("rule_facts", {})
        if (not isinstance(facts, Mapping)
                or facts.get("definedness") != "catalog-known"):
            continue
        row_occurrence_ids = tuple(occurrence_ids.get(row["id"], ()))
        relations = _program_relation_facts(
            row, profile, occurrence_ids=row_occurrence_ids,
        )
        occurrence_id = row_occurrence_ids[0] if len(row_occurrence_ids) == 1 else None
        r3_text = _r3_swap_text(row)
        r3_relation = relations.get("R3", {})
        r2_relation = relations.get("R2", {})
        if (r3_text is not None and row_occurrence_ids
                and r3_relation.get("status") == "available"
                and _r3_anchor_ready(row, profile)
                and _candidate_size_matches(row, r3_text)):
            result.append(_profile_action(
                row, profile, "R3", r3_text, occurrence_id, row_occurrence_ids,
                profile_digest=profile_digest,
            ))
        if (row_occurrence_ids and r2_relation.get("status") == "available"
                and _r3_anchor_ready(row, profile)):
            result.extend(
                _profile_action(
                    row, profile, "R2", text, occurrence_id, row_occurrence_ids,
                    profile_digest=profile_digest,
                )
                for text in _source_alias_texts(row)
                if _candidate_size_matches(row, text)
            )
    rows_by_block = {}
    for row in profile.instructions:
        rows_by_block.setdefault(row.get("block"), []).append(row)
    incoming_blocks = {right for _left, right in _profile_incoming_edges(profile)}
    consumer_cache = _profile_gpr_consumer_cache(
        profile.instructions, profile.occurrences,
    )
    protected = {_pc_value(profile.input_identity.get("entry"))}
    for block in profile.blocks:
        block_rows = tuple(rows_by_block.get(block, ()))
        block_labels = {
            str(name) for row in block_rows for name in row.get("labels", ())
            if isinstance(name, str) and name
        }
        if (layout_ready and _m5_trace_complete(profile) and block_rows
                and _m5_layout_safe(profile, block_rows)
                and all(not row.get("executed") for row in block_rows)
                and not any(
                    any(_pc_value(pc) in protected for pc in _row_final_pcs(row))
                    for row in block_rows
                )
                and not block_labels & _M5_PROTECTED_LABELS
                and block not in incoming_blocks
                and profile.input_identity.get("result_channel") not in (None, "")
                and str(profile.input_identity.get("result_channel")) not in block_labels
        ):
            result.append(_dead_block_action(block_rows, profile, profile_digest=profile_digest))
    for row in profile.instructions:
        if not row.get("executed") or not _r3_anchor_ready(row, profile):
            continue
        facts = row.get("rule_facts", {})
        if not isinstance(facts, Mapping):
            continue
        row_occurrence_ids = tuple(occurrence_ids.get(row.get("id"), ()))
        relations = _program_relation_facts(
            row, profile, occurrence_ids=row_occurrence_ids,
        )
        if not row_occurrence_ids:
            continue
        for rule in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"):
            relation = relations.get(rule, {})
            status = relation.get("status")
            if status not in {"available", "observer-gap"}:
                continue
            if status == "available" and rule in {"R2", "R3"}:
                continue
            if rule == "R10" and _has_static_gpr_consumer(
                row, profile.instructions, profile, consumer_cache
            ):
                continue
            probe = status != "available"
            if probe and rule == "R2" and len(row_occurrence_ids) > 1:
                continue
            texts = tuple(
                text for text in _rule_candidate_texts(row, rule, probe=probe)
                if _candidate_size_matches(row, text)
            )
            occurrence_id = row_occurrence_ids[0] if len(row_occurrence_ids) == 1 else None
            result.extend(
                _profile_action(
                    row, profile, rule, text, occurrence_id, row_occurrence_ids,
                    probe=probe,
                    witness_reason=relation.get("reason") if probe else None,
                    profile_digest=profile_digest,
                )
                for text in texts
            )
    # Keep one structural action per unreachable block.  Rewriting an
    # executed instruction changes the observed PC path and is rejected by
    # the reference comparator; an unreachable no-op grows the full program
    # while preserving the reference witness.  The MCMC layer gives these
    # actions precedence on the first proposal, then samples them normally.
    # ponytail: 只在执行路径和固定 exit checkpoint 之后的不可达块增长；
    # 要支持可达插入，需要把 reference comparator 从绝对 PC 比较升级为路径映射。
    growth_blocks: set[object] = set()
    executed_end = max(
        (
            _pc_value(row.get("pc_end")) or 0
            for row in profile.instructions
            if row.get("executed") is True
        ),
        default=0,
    )
    fixed_exit = _pc_value(profile.input_identity.get("exit")) or 0
    growth_floor = max(executed_end, fixed_exit)
    source_lines = _action_source(program).splitlines(keepends=True)
    for row in profile.instructions:
        block = row.get("block")
        row_pc = _pc_value(row.get("pc"))
        line = row.get("line")
        raw_line = (
            source_lines[line]
            if type(line) is int and 0 <= line < len(source_lines)
            else ""
        )
        safe_growth_row = (
            row.get("executed") is False
            and row.get("source_path") in (None, "")
            and isinstance(block, str)
            and block not in incoming_blocks
            and row_pc is not None
            and row_pc >= growth_floor
            and isinstance(row.get("section"), str)
            and (row["section"] == "text" or row["section"].startswith(".text"))
            and type(line) is int
            and type(row.get("start")) is int
            and type(row.get("end")) is int
        )
        if safe_growth_row and _PROGRAM_GROWTH_MARKER in raw_line:
            result.append(_program_growth_delete_action(
                row, raw_line, profile, profile_digest=profile_digest,
            ))
        if (
            safe_growth_row
            and block not in growth_blocks
        ):
            result.append(_program_growth_action(row, profile, profile_digest=profile_digest))
            growth_blocks.add(block)
    unique = {}
    for action in result:
        unique.setdefault(canonical_digest(dict(action)), action)
    return tuple(unique.values())


def profile_actions(program: object, profile: object) -> tuple[dict[str, object], ...]:
    if getattr(program, "run_params", {}).get("route") == "single":
        from ..rvgen.single_program import single_profile_actions
        return single_profile_actions(program, profile)
    try:
        return _profile_actions(program, profile)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, StopIteration):
        return ()


def _source_key(value: object) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(value, str):
        return None
    body = _strip_comments(value).strip()
    parts = body[len(_label_prefix(body)):].strip().split(None, 1)
    if not parts:
        return None
    normalized = []
    for operand in parts[1].split(",") if len(parts) > 1 else ():
        operand = operand.strip()
        memory = _memory_parts(operand)
        if memory and (base := memory[1]) is not None:
            offset = memory[0]
            normalized.append(f"{offset}(x{base})" if offset is not None else operand.lower())
            continue
        register = _register_index(operand)
        fp_register = _fp_index(operand)
        number = _int_value(operand)
        normalized.append(
            f"x{register}" if register is not None
            else f"f{fp_register}" if fp_register is not None
            else str(number) if number is not None
            else operand.lower().replace("_", ".")
        )
    return parts[0].lower().replace("_", "."), tuple(normalized)


def _apply_dead_block(program: object, profile: object, action: Mapping[str, object]) -> ProgramVariant | None:
    anchor = action.get("anchor")
    if (action.get("rule") != "M5" or action.get("source_cell") != "block:dead"
            or action.get("target_cell") != "block:deleted" or not isinstance(anchor, Mapping)):
        return None
    rows = tuple(row for row in profile.instructions if row.get("block") == anchor.get("block"))
    protected = {_pc_value(profile.input_identity.get("entry"))}
    if (not rows or any(row.get("executed") for row in rows)
            or not _m5_trace_complete(profile)
            or not _m5_layout_safe(profile, rows)
            or any(
                _pc_value(pc) in protected
                for row in rows for pc in _row_final_pcs(row)
            )
            or any(right == anchor.get("block") for _left, right in _profile_incoming_edges(profile))
            or tuple(row["id"] for row in rows) != tuple(anchor.get("instruction_ids", ()))
            or anchor.get("source_start") != min(row["start"] for row in rows)
            or anchor.get("source_end") != max(row["end"] for row in rows)
            or any(
                name in anchor
                and tuple(anchor[name]) != tuple(value for row in rows for value in row.get(name, ()))
                for name in ("final_pcs", "final_sizes", "final_bytes")
        )):
        return None
    source = _action_source(program)
    changed = source
    for row in sorted(rows, key=lambda item: int(item["start"]), reverse=True):
        start, end = int(row["start"]), int(row["end"])
        changed = changed[:start] + _label_prefix(changed[start:end]) + changed[end:]
    if changed == source:
        return None
    return ProgramVariant(changed, program_sha256(program), json.dumps(dict(action), sort_keys=True, separators=(",", ":")), dict(getattr(program, "run_params", {})))


def _rewrite_instruction_line(
    raw: str, text: str, source_start: int = 0, source_end: int | None = None,
) -> str:
    newline = "\r\n" if raw.endswith("\r\n") else "\n" if raw.endswith("\n") else ""
    code = raw[:-len(newline)] if newline else raw
    if source_end is not None:
        separator = ";" if source_end and code[source_end - 1] == ";" else ""
        end = source_end - bool(separator)
        replacement = _rewrite_instruction_line(code[source_start:end], text)
        return code[:source_start] + replacement + separator + code[source_end:] + newline
    cut = min((index for index in (code.find("#"), code.find("//")) if index >= 0), default=len(code))
    statement, comment = code[:cut], code[cut:]
    prefix = _label_prefix(statement)
    body = statement[len(prefix):]
    indent = body[:len(body) - len(body.lstrip())]
    old = body[len(indent):]
    old_match = re.match(r"\S+(.*)", old)
    new_parts = text.split(None, 1)
    if old_match is None or not new_parts:
        return prefix + indent + text.strip() + comment + newline
    old_tail = old_match.group(1)
    gap = re.match(r"\s*", old_tail).group(0)
    old_operands = old_tail[len(gap):]
    trailing = re.search(r"\s*$", old_operands).group(0)
    operand_body = old_operands[:-len(trailing)] if trailing else old_operands
    separators = re.findall(r",\s*", operand_body)
    new_operands = [item.strip() for item in new_parts[1].split(",")] if len(new_parts) > 1 else []
    if new_operands:
        if separators:
            operand_text = "".join(
                item + separators[min(index, len(separators) - 1)]
                for index, item in enumerate(new_operands[:-1])
            ) + new_operands[-1]
        else:
            operand_text = ", ".join(new_operands)
        replacement = new_parts[0] + gap + operand_text + trailing
    else:
        replacement = new_parts[0]
    return prefix + indent + replacement + comment + newline


def _materialize_profile_action(
    program: object, profile: object, action: Mapping[str, object],
) -> ProgramVariant | None:
    if (
        action.get("operator") == "delete_line"
        and action.get("mutation_kind") == "instruction-delete"
    ):
        span = action.get("source_span")
        payload = action.get("payload")
        if not isinstance(span, Mapping) or not isinstance(payload, Mapping) \
                or payload.get("marker") != _PROGRAM_GROWTH_MARKER:
            return None
        start, end = span.get("start"), span.get("end")
        source = _action_source(program)
        lines = source.splitlines(keepends=True)
        if type(start) is not int or start != end or not 1 <= start <= len(lines):
            return None
        line = lines[start - 1]
        if _PROGRAM_GROWTH_MARKER not in line:
            return None
        changed = "".join((*lines[:start - 1], *lines[start:]))
        if changed == source:
            return None
        return ProgramVariant(
            changed, program_sha256(program),
            json.dumps(dict(action), sort_keys=True, separators=(",", ":")),
            dict(getattr(program, "run_params", {})),
        )
    if (
        action.get("operator") == "insert"
        and action.get("mutation_kind") == "instruction-insert"
    ):
        anchor = action.get("anchor")
        payload = action.get("payload")
        if not isinstance(anchor, Mapping) or not isinstance(payload, Mapping):
            return None
        source_text = payload.get("source")
        if not isinstance(source_text, str) or not source_text.strip():
            return None
        row = next(
            (item for item in profile.instructions
             if item.get("id") == anchor.get("instruction_id")),
            None,
        )
        if row is None or row.get("block") != anchor.get("block") \
                or row.get("executed") is True:
            return None
        source = _action_source(program)
        lines = source.splitlines(keepends=True)
        line = row.get("line")
        if type(line) is not int or not 0 <= line < len(lines):
            return None
        line_start = sum(len(item) for item in lines[:line])
        if row.get("start") != anchor.get("source_start") \
                or row.get("end") != anchor.get("source_end") \
                or not line_start <= row["start"] < row["end"] <= line_start + len(lines[line]):
            return None
        raw = lines[line]
        newline = raw[len(raw.rstrip("\r\n")):] or "\n"
        indent = raw[:len(raw) - len(raw.lstrip())]
        lines.insert(line + 1, f"{indent}{source_text.strip()}{newline}")
        changed = "".join(lines)
        return ProgramVariant(
            changed, program_sha256(program),
            json.dumps(dict(action), sort_keys=True, separators=(",", ":")),
            dict(getattr(program, "run_params", {})),
        )
    if (
        action.get("rule") == "M5"
        and action.get("source_cell") == "block:dead"
        and action.get("target_cell") == "block:deleted"
    ):
        return _apply_dead_block(program, profile, action)
    anchor = action["anchor"]
    row = next(item for item in profile.instructions if item["id"] == anchor["instruction_id"])
    text = action["payload"]["source"]
    source = _action_source(program)
    lines = source.splitlines(keepends=True)
    line = int(row["line"])
    raw = lines[line]
    line_start = sum(len(item) for item in lines[:line])
    if row["end"] > line_start + len(raw):
        changed = source[:row["start"]] + text + source[row["end"]:]
    else:
        changed = "".join(lines[:line] + [_rewrite_instruction_line(
            raw, text, row["start"] - line_start, row["end"] - line_start,
        )] + lines[line + 1:])
    return ProgramVariant(
        changed, program_sha256(program),
        json.dumps(dict(action), sort_keys=True, separators=(",", ":")),
        dict(getattr(program, "run_params", {})),
    )


def apply_profile(
    program: object, profile: object, action: Mapping[str, object], *,
    _action_pool: (
        Sequence[Mapping[str, object]]
        | Mapping[str, Mapping[str, object]] | None
    ) = None,
) -> ProgramVariant | None:
    """按 ProgramProfile 的稳定 anchor 应用一个 scalar action。"""
    if getattr(program, "run_params", {}).get("route") == "single":
        from ..rvgen.single_program import apply_single_profile
        return apply_single_profile(program, profile, action, _action_pool=_action_pool)
    if profile is None:
        # Open-capability campaigns can lose the semantic reference profile
        # for an extension that rvgen-semantic does not model.  The route
        # enumerator still supplies source-only EMI candidates; materialize
        # those candidates directly so MCMC does not turn into an empty
        # proposal loop while waiting for a profile that cannot exist.
        if not isinstance(action, Mapping):
            return None
        digest = canonical_digest(dict(action))
        if isinstance(_action_pool, Mapping):
            if digest not in _action_pool:
                return None
        else:
            pool = tuple(_action_pool or ())
            if pool and not any(
                isinstance(item, Mapping) and canonical_digest(dict(item)) == digest
                for item in pool
            ):
                return None
        span = action.get("source_span")
        payload = action.get("payload")
        if not isinstance(span, Mapping) or not isinstance(payload, Mapping):
            return None
        start, end = span.get("start"), span.get("end")
        start_col, end_col = span.get("start_col"), span.get("end_col")
        if not all(type(value) is int for value in (start, end, start_col, end_col)):
            return None
        source = _action_source(program)
        lines = source.splitlines(keepends=True)
        if not 1 <= start <= end <= len(lines) or start_col < 0 or end_col < start_col:
            return None
        newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
        if action.get("operator") == "replace":
            mnemonic = payload.get("mnemonic")
            operands = payload.get("operands")
            if not isinstance(mnemonic, str) or not isinstance(operands, list) \
                    or not all(isinstance(item, str) for item in operands):
                return None
            line = lines[start - 1]
            code = line.rstrip("\r\n")
            if end_col > len(code):
                return None
            replacement = f"{mnemonic} {', '.join(operands)}" if operands else mnemonic
            changed_lines = [
                *lines[:start - 1],
                code[:start_col] + replacement + code[end_col:] + (line[len(code):] or ""),
                *lines[start:],
            ]
        elif action.get("operator") == "replace_fragment":
            fragment = payload.get("lines")
            if not isinstance(fragment, list) or not all(isinstance(item, str) for item in fragment):
                return None
            replacement = "".join(
                item.rstrip("\r\n") + newline for item in fragment
            )
            changed_lines = [
                *lines[:start - 1], replacement, *lines[end:]
            ]
        else:
            return None
        changed = "".join(changed_lines)
        if changed == source:
            return None
        return ProgramVariant(
            changed, program_sha256(program),
            json.dumps(dict(action), sort_keys=True, separators=(",", ":")),
            dict(getattr(program, "run_params", {})),
        )
    if not getattr(profile, "complete", False) or not isinstance(action, Mapping):
        return None
    actual_source_sha256 = program_sha256(program)
    if (actual_source_sha256 != profile.source_sha256
            or not _profile_input_matches(program, profile)
            or action.get("profile_digest") != profile.profile_digest):
        return None
    try:
        action_digest = canonical_digest(dict(action))
        # 调用方已经把 action pool 传进来时直接复用。否则每个 action 都要重跑一次
        # profile_actions，而它内部又是「每个 action 一次 canonical_digest」，
        # 整步就退化成 O(n^2)：实测一个 case 的 MCMC 被拖到 900s 超时。
        if _action_pool is not None:
            if isinstance(_action_pool, Mapping):
                selected = _action_pool.get(action_digest)
            else:
                selected = {
                    canonical_digest(dict(item)): item
                    for item in _action_pool if isinstance(item, Mapping)
                }.get(action_digest)
            return (
                _materialize_profile_action(program, profile, selected)
                if selected is not None else None
            )
        authorized_pool = profile_actions(program, profile)
        selected = {
            canonical_digest(dict(item)): item
            for item in authorized_pool if isinstance(item, Mapping)
        }.get(action_digest)
        return (
            _materialize_profile_action(program, profile, selected)
            if selected is not None else None
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, StopIteration):
        return None


__all__ = [
    "ProfileResult", "ProgramProfile", "ProgramVariant",
    "apply_profile", "build_profile",
    "profile_actions", "program_sha256",
]

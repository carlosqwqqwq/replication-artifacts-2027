import hashlib
import json
import struct
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._util import (
    _non_empty_string,
    canonical_digest,
    is_sha256_digest,
    pc_int as _trace_int,
    strict_bool,
)
from ._translation_cell import _optional_int, _tuple_str


U64_MASK = (1 << 64) - 1
OBSERVATION_MAGIC = b"RVOBS1\x00\x00"
OBSERVATION_HEADER_SIZE = 8 + 8 + (32 * 8) + 8
NATIVE_OBSERVER_TEMP_GPRS = frozenset({29, 30, 31})
MAX_OBSERVATION_MEMORY_SIZE = 1 << 24
INLINE_EXECUTED_PC_LIMIT = 100_000
_OBSERVATION_MEMORY_SIZE_OFFSET = 16 + (32 * 8)
OBSERVATION_CONTRACT_VERSION = "RVOBS1"
_TRANSLATION_PATHS = frozenset(
    {"reference", "translated", "tcg", "interpreter", "jit", "either"}
)
_TRAP_FIELD_ALIASES = {
    "trap.cause": ("trap.cause", "guest_cause", "csr.mcause", "trap_mcause"),
    "trap.epc": ("trap.epc", "guest_epc", "csr.mepc"),
    "trap.tval": ("trap.tval", "guest_tval", "csr.mtval"),
}
_TRAP_COMPARISON_ALIASES = {
    canonical.replace(".", "_"): canonical for canonical in _TRAP_FIELD_ALIASES
}


def native_trace_stdout_mismatch_is_compatible(extra_state: object) -> bool:
    if not isinstance(extra_state, Mapping):
        return True
    mismatch = extra_state.get("native_trace_stdout_mismatch")
    if mismatch is None:
        return True
    if not isinstance(mismatch, Mapping):
        return False
    diff_indices = mismatch.get("gpr_diff_indices")
    volatile_indices = extra_state.get("volatile_gpr_indices")
    if not isinstance(diff_indices, (list, tuple)) or not diff_indices:
        return False
    if not isinstance(volatile_indices, (list, tuple)):
        return False
    if any(type(index) is not int for index in (*diff_indices, *volatile_indices)):
        return False
    return (
        len(set(diff_indices)) == len(diff_indices)
        and set(diff_indices).issubset(volatile_indices)
        and set(diff_indices).issubset(NATIVE_OBSERVER_TEMP_GPRS)
        and mismatch.get("memory_equal") is True
        and mismatch.get("checkpoint_pc_equal") is True
        and mismatch.get("other_bytes_equal") is True
    )
_P_VXSAT_FORM_PREFIXES = (
    "nclip", "pnclip", "psadd.", "psaddu.", "pssub.", "pssubu.",
    "psabs.", "psslai.", "pssh1sadd.",
)
OBSERVATION_CONTRACT = {
    "version": OBSERVATION_CONTRACT_VERSION,
    "checkpoint": "exit_checkpoint before RVOBS1 dump suffix",
    "gpr_snapshot": "architectural GPR state at checkpoint; observer scratch registers are saved before clobber (RV32E uses x14/x15, full register profiles use x30/x31); x29 is used only after the GPR snapshot",
    "memory_snapshot": "test_memory bytes copied before write/exit syscalls",
    "obligation_state": "optional runner- or testcase-defined signal/fault/extra-state may be reported out-of-band",
    "syscall_state": "write/exit syscall register state is outside the compared snapshot",
}


def _observation_frame_end(data: object, offset: int) -> int | None:
    """Return the end declared by one RVOBS1 header, if that header exists."""
    if (
        not isinstance(data, (bytes, bytearray, memoryview))
        or type(offset) is not int
        or offset < 0
        or len(data) < offset + OBSERVATION_HEADER_SIZE
    ):
        return None
    memory_size = struct.unpack_from(
        "<Q", data, offset + _OBSERVATION_MEMORY_SIZE_OFFSET,
    )[0]
    if memory_size > MAX_OBSERVATION_MEMORY_SIZE:
        return None
    return offset + OBSERVATION_HEADER_SIZE + memory_size


def _valid_window_checkpoint(item: dict) -> bool:
    try:
        pc = _trace_int(item.get("pc"))
        after_pc = _trace_int(item.get("after_pc"))
        if pc <= 0 or after_pc <= 0 or pc & 1 or after_pc & 1:
            return False
    except (TypeError, ValueError):
        return False
    for key, required in (
        ("before_gpr", True), ("after_gpr", True),
        ("before_fpr_rawbits", False), ("after_fpr_rawbits", False),
    ):
        values = item.get(key)
        if values is None:
            if required:
                return False
            continue
        if not isinstance(values, list) or len(values) != 32:
            return False
        try:
            if required and _trace_int(values[0]) != 0:
                return False
            tuple(_trace_int(value) for value in values)
        except (TypeError, ValueError):
            return False
    if (item.get("before_fpr_rawbits") is None) != (item.get("after_fpr_rawbits") is None):
        return False
    for name in ("fflags", "frm"):
        if (item.get(f"before_{name}") is None) != (item.get(f"after_{name}") is None):
            return False
    for key, mask in (("before_fflags", 0x1F), ("after_fflags", 0x1F), ("before_frm", 0x7), ("after_frm", 0x7)):
        value = item.get(key)
        if value is None:
            continue
        try:
            if _trace_int(value) > mask:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _checkpoint_state_chain_valid(checkpoints: list[dict]) -> bool:
    for left, right in zip(checkpoints, checkpoints[1:]):
        for name in ("gpr", "fpr_rawbits"):
            left_values = left.get(f"after_{name}")
            right_values = right.get(f"before_{name}")
            if (left_values is None) != (right_values is None):
                return False
            if left_values is not None and tuple(_trace_int(value) for value in left_values) != tuple(
                _trace_int(value) for value in right_values
            ):
                return False
        for name in ("fflags", "frm"):
            left_value = left.get(f"after_{name}")
            right_value = right.get(f"before_{name}")
            if (left_value is None) != (right_value is None):
                return False
            if left_value is not None and _trace_int(left_value) != _trace_int(right_value):
                return False
    return True
_SIGNAL_CODE_BY_NAME = {
    "SIGHUP": 1, "SIGINT": 2, "SIGQUIT": 3, "SIGILL": 4,
    "SIGTRAP": 5, "SIGABRT": 6, "SIGBUS": 7, "SIGFPE": 8,
    "SIGKILL": 9, "SIGUSR1": 10, "SIGSEGV": 11, "SIGUSR2": 12,
    "SIGPIPE": 13, "SIGALRM": 14, "SIGTERM": 15,
}
_SIGNAL_NAME_BY_TEXT = {
    "segmentation fault": "SIGSEGV",
    "bus error": "SIGBUS",
    "illegal instruction": "SIGILL",
    "floating point exception": "SIGFPE",
    "trace/breakpoint trap": "SIGTRAP",
    "aborted": "SIGABRT",
    "killed": "SIGKILL",
    "terminated": "SIGTERM",
    "hangup": "SIGHUP",
    "quit": "SIGQUIT",
    "alarm clock": "SIGALRM",
    "broken pipe": "SIGPIPE",
}


def observation_blob(
    *,
    checkpoint_pc: int,
    gpr: tuple[int, ...],
    memory: bytes,
) -> bytes:
    if len(gpr) != 32:
        raise ValueError("observation blob requires 32 GPR values")
    if len(memory) > MAX_OBSERVATION_MEMORY_SIZE:
        raise ValueError("observation memory snapshot exceeds the RVOBS1 limit")
    payload = bytearray(OBSERVATION_MAGIC)
    payload.extend(struct.pack("<Q", _as_u64(checkpoint_pc)))
    for value in gpr:
        payload.extend(struct.pack("<Q", _as_u64(value)))
    payload.extend(struct.pack("<Q", len(memory)))
    payload.extend(memory)
    return bytes(payload)
STRICT_EMI_GPR_INDICES = tuple(range(32))
DIRECT_FULL_STATE_COMPARE_POLICY = "direct-full-state-compare-v1"
OBLIGATION_STATE_COMPARE_POLICY = "obligation-state-compare-v1"
RVGEN_CANONICAL_TAG = "rvgen-canonical"

# Shared program layout contract.  Generation places state and observers at
# these logical coordinates; direct_elf binds them to real addresses and RVEMI
# reads them back from the native profile, so they belong to the testcase
# contract rather than to any one producer.  Keep helper registers in x8..x15:
# the same fixed layout is then executable for both RV32I/RV64 and RV32E.
LOGICAL_DATA_ADDRESS = 0
OBSERVABLE_REGION = "test-memory"
MEMORY_BASE_REGISTER = 14
FFLAGS_OBSERVER_REGISTER = 6
FFLAGS_OBSERVER_OFFSET = 16
INSTRUCTION_MEMORY_PATCH_OFFSET = 0
INSTRUCTION_MEMORY_OBSERVER_OFFSET = 16
INSTRUCTION_MEMORY_LINK_OFFSET = 24
INSTRUCTION_MEMORY_REGION_BYTES = 384
INSTRUCTION_MEMORY_ADDRESS_REGISTER = 11
INSTRUCTION_MEMORY_WORD_REGISTER = 12
INSTRUCTION_MEMORY_MARKER_REGISTER = 13
INSTRUCTION_MEMORY_RETURN_REGISTER = 15
INSTRUCTION_MEMORY_LINK_POISON = 0xBADC0FFEE0DDF00D
PRODUCER_LOAD_OFFSET = 80
PRODUCER_SOURCE_POISON = 0xBAD0BAD0BAD0BAD0
HARNESS_RESERVED_GPRS = frozenset({0, 2, MEMORY_BASE_REGISTER, 29, 30, 31})
ADDRESS_MODEL = "logical-offsets-bound-by-direct-elf"

VERDICT_RUNNER_CONTRACT_GAP = "runner-contract-gap"
OBSERVATION_OUTCOMES = frozenset(
    {
        "normal",
        "completed",
        "trap",
        "nonzero-exit",
        "timeout",
        "unavailable",
        VERDICT_RUNNER_CONTRACT_GAP,
    }
)

SEMANTIC_POINT_CONTRACT = "rvgen-semantic-point-v2"
_SEMANTIC_POINT_STRING_FIELDS = (
    "effect_kind",
    "lane",
    "primary_boundary_class",
    "observer_kind",
    "register_relation",
    "layout_shape",
    "source_mode",
)


@dataclass(frozen=True)
class SemanticPoint:
    """The classified semantic identity of one concrete testcase.

    Generation produces it from actual state, campaign persists it, and RVEMI
    consumes it, so the value object and its strict parser are part of the
    shared testcase contract rather than of the generator.
    """

    effect_kind: str
    lane: str
    primary_boundary_class: str
    properties: tuple[tuple[str, str | int | bool], ...]
    observer_kind: str
    register_relation: str
    layout_shape: str
    source_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": SEMANTIC_POINT_CONTRACT,
            "effect_kind": self.effect_kind,
            "lane": self.lane,
            "primary_boundary_class": self.primary_boundary_class,
            "properties": {name: value for name, value in self.properties},
            "observer_kind": self.observer_kind,
            "register_relation": self.register_relation,
            "layout_shape": self.layout_shape,
            "source_mode": self.source_mode,
        }


def parse_current_semantic_point(data: object) -> SemanticPoint | None:
    """Parse the only semantic-point contract accepted by current discovery.

    A recognised contract name without a complete typed payload is not
    provenance that can identify a prospective variant.
    """
    if not isinstance(data, dict) or data.get("contract") != SEMANTIC_POINT_CONTRACT:
        return None
    fields = {name: data.get(name) for name in _SEMANTIC_POINT_STRING_FIELDS}
    if not all(isinstance(value, str) and value for value in fields.values()):
        return None
    properties = data.get("properties")
    if not isinstance(properties, dict):
        return None
    normalized_properties: list[tuple[str, str | int | bool]] = []
    for name, value in properties.items():
        if not isinstance(name, str) or not name or not isinstance(value, (str, int, bool)):
            return None
        normalized_properties.append((name, value))
    return SemanticPoint(
        effect_kind=str(fields["effect_kind"]),
        lane=str(fields["lane"]),
        primary_boundary_class=str(fields["primary_boundary_class"]),
        properties=tuple(sorted(normalized_properties)),
        observer_kind=str(fields["observer_kind"]),
        register_relation=str(fields["register_relation"]),
        layout_shape=str(fields["layout_shape"]),
        source_mode=str(fields["source_mode"]),
    )


def expected_trap_from_dataflow(dataflow_meta: object) -> bool:
    if not isinstance(dataflow_meta, dict):
        return False
    if dataflow_meta.get("candidate_mutated"):
        return dataflow_meta.get("candidate_expected_trap") is True
    facts = dataflow_meta.get("realized_facts", {})
    return isinstance(facts, dict) and (
        facts.get("state_domain") == "trap"
        or facts.get("lane") == "legality-expected-trap"
    )


def testcase_uses_fp_instruction(testcase: Any) -> bool:
    from .spec_definedness import x_register_fp_extensions

    if x_register_fp_extensions(str(getattr(testcase, "isa_profile", ""))):
        return False
    for item in getattr(testcase, "instruction_meta", ()):
        mnemonic = str(getattr(item, "mnemonic", "")).lower()
        if (
            mnemonic.startswith("f") and not mnemonic.startswith("fence")
        ) or mnemonic.startswith("c.f"):
            return True
        if (
            mnemonic.startswith(("vf", "vmf"))
            and not mnemonic.startswith("vfirst")
        ):
            return True
        # FS also gates the three scalar floating-point state CSRs.
        if any(
            name == "csr" and address in {0x001, 0x002, 0x003}
            for name, address in getattr(item, "operand_fields", ())
        ):
            return True
    return False


def testcase_uses_vector_instruction(testcase: Any) -> bool:
    from .riscv_csrs import csr_required_extensions
    from .spec_definedness import enabled_extensions

    extensions = enabled_extensions(str(getattr(testcase, "isa_profile", "")))
    vector_status_exists = "v" in extensions or any(
        extension.startswith("zve") for extension in extensions
    )

    for item in getattr(testcase, "instruction_meta", ()):
        mnemonic = str(getattr(item, "mnemonic", "")).lower()
        if mnemonic.startswith(("v", "vm")):
            return True
        if vector_status_exists and p_form_uses_vxsat(mnemonic):
            return True
        if any(
            name == "csr" and "v" in csr_required_extensions(address)
            for name, address in getattr(item, "operand_fields", ())
        ):
            return True
    return False


def p_form_uses_vxsat(mnemonic: str) -> bool:
    """Whether this modeled P form can access the sticky saturation flag."""

    return str(mnemonic).lower().startswith(_P_VXSAT_FORM_PREFIXES)


def final_fp_state_required_for_testcase(testcase: Any) -> bool:
    dataflow_meta = getattr(testcase, "dataflow_meta", {})
    if expected_trap_from_dataflow(dataflow_meta):
        return False
    owner = dataflow_meta.get("generation_owner") if isinstance(dataflow_meta, dict) else None
    if not isinstance(owner, dict):
        from .spec_definedness import profile_requires_fp_state

        return profile_requires_fp_state(str(getattr(testcase, "isa_profile", "")))
    schema = owner.get("schema")
    observer = schema.get("observer") if isinstance(schema, dict) else None
    fields = (
        list(dataflow_meta.get("state_observer_keys", ()))
        + list(getattr(getattr(testcase, "compare_mask", None), "extra_state_keys", ()))
    )
    if any(
        field in {"fpr_rawbits", "fflags", "frm", "csr.fflags", "csr.frm"}
        for field in fields
    ):
        return True
    if _lossless_memory_observation(testcase):
        return False
    from .spec_definedness import profile_requires_fp_state

    if not profile_requires_fp_state(str(getattr(testcase, "isa_profile", ""))):
        return False
    return observer not in {None, "", "gpr"}


def generation_semantic_point_for_testcase(testcase: Any) -> dict[str, Any] | None:
    dataflow_meta = getattr(testcase, "dataflow_meta", None)
    semantic_point = dataflow_meta.get("semantic_point") if isinstance(dataflow_meta, dict) else None
    parsed = parse_current_semantic_point(semantic_point)
    return None if parsed is None else dict(parsed.to_dict())


def _lossless_memory_observation(testcase: object) -> bool:
    contract = getattr(testcase, "observability_contract", None)
    return (
        contract is not None
        and getattr(contract, "sink_transform", None) == "full-width-store"
        and getattr(contract, "lossiness", None) == "none"
        and any(
            isinstance(field, str) and field.startswith("memory.")
            for field in getattr(contract, "final_observation_fields", ())
        )
    )


def _as_u64(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("u64 field must be an integer")
    if value < 0 or value > U64_MASK:
        raise ValueError("u64 field is out of range")
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _operand_fields(value: Any) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("operand_fields must be a list of pairs")
    fields: list[tuple[str, int]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            or item[1] < 0
        ):
            raise ValueError("operand_fields must contain non-negative integer pairs")
        fields.append((item[0], item[1]))
    return tuple(fields)


def _bytes_from_hex(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("hex bytes must be a string")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid hex bytes") from exc


def _tuple_int(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("expected a list of integers")
    result = tuple(values)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in result):
        raise ValueError("expected a list of integers")
    return result


def _sequence_field(data: dict[str, Any], field_name: str) -> list[Any] | tuple[Any, ...]:
    value = data.get(field_name, ())
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    return value


def _object_field(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{field_name} must be an object")
    return data


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain JSON values") from error
    return value


def _optional_register_domain(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in {"gpr", "fpr"}:
        raise ValueError(f"{field_name} must be 'gpr', 'fpr', or null")
    return value


@dataclass(frozen=True)
class ObservabilityContract:
    schema_version: str
    risk_value_or_effect: str
    sink_instruction_ids: tuple[str, ...]
    risk_to_sink_distance: int
    sink_transform: str
    final_observation_fields: tuple[str, ...]
    lossiness: str

    def __post_init__(self) -> None:
        if self.schema_version != "observability-contract-v1":
            raise ValueError("unsupported observability contract schema")
        _non_empty_string(self.risk_value_or_effect, "risk_value_or_effect")
        _non_empty_string(self.sink_transform, "sink_transform")
        if not self.sink_instruction_ids:
            raise ValueError("observability contract requires sinks")
        if (
            not isinstance(self.sink_instruction_ids, (list, tuple))
            or any(not isinstance(item, str) or not item.strip() for item in self.sink_instruction_ids)
        ):
            raise ValueError("observability contract sink IDs must be non-empty strings")
        if len(self.sink_instruction_ids) != len(set(self.sink_instruction_ids)):
            raise ValueError("observability contract sink IDs must be unique")
        if not self.final_observation_fields:
            raise ValueError("observability contract requires observation fields")
        if (
            not isinstance(self.final_observation_fields, (list, tuple))
            or any(not isinstance(item, str) or not item.strip() for item in self.final_observation_fields)
        ):
            raise ValueError("observability contract fields must be non-empty strings")
        if len(self.final_observation_fields) != len(set(self.final_observation_fields)):
            raise ValueError("observability contract fields must be unique")
        if (
            not isinstance(self.risk_to_sink_distance, int)
            or isinstance(self.risk_to_sink_distance, bool)
            or self.risk_to_sink_distance < 1
            or self.risk_to_sink_distance > 3
        ):
            raise ValueError("fresh observability sink distance must be between 1 and 3")
        if self.lossiness not in {"none", "partial"}:
            raise ValueError("unsupported observability lossiness")
        if self.sink_transform not in {
            "identity",
            "injective-full-width-arithmetic",
            "full-width-store",
            "control-merge",
            "nonzero-reduction",
            "raw-bits-store",
            "raw-bits-plus-fflags",
            "vector-store",
            "warl-readback",
            "side-effect-readback",
        }:
            raise ValueError("unsupported observability sink transform")
        object.__setattr__(self, "sink_instruction_ids", tuple(self.sink_instruction_ids))
        object.__setattr__(self, "final_observation_fields", tuple(self.final_observation_fields))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "risk_value_or_effect": self.risk_value_or_effect,
            "sink_instruction_ids": list(self.sink_instruction_ids),
            "risk_to_sink_distance": self.risk_to_sink_distance,
            "sink_transform": self.sink_transform,
            "final_observation_fields": list(self.final_observation_fields),
            "lossiness": self.lossiness,
        }

    @property
    def contract_id(self) -> str:
        return "oc-" + canonical_digest({
            "contract": self.schema_version, **self.canonical_dict()
        })[:24]

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "contract_id": self.contract_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservabilityContract":
        data = _object_field(data, "observability contract")
        contract = cls(
            schema_version=data.get("schema_version"),
            risk_value_or_effect=data.get("risk_value_or_effect"),
            sink_instruction_ids=_tuple_str(data.get("sink_instruction_ids", ())),
            risk_to_sink_distance=_strict_nonnegative_int(
                data.get("risk_to_sink_distance"), "risk_to_sink_distance"
            ),
            sink_transform=data.get("sink_transform"),
            final_observation_fields=_tuple_str(data.get("final_observation_fields", ())),
            lossiness=data.get("lossiness"),
        )
        recorded_id = data.get("contract_id")
        if recorded_id is not None and recorded_id != contract.contract_id:
            raise ValueError("observability contract id does not match canonical fields")
        return contract


@dataclass(frozen=True)
class MemoryRegion:
    region_id: str
    address: int
    data: bytes
    permissions: str = "rw"

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("memory region_id is required")
        if not isinstance(self.permissions, str) or not self.permissions:
            raise ValueError("memory permissions are required")
        if (
            any(character not in "rwx" for character in self.permissions)
            or len(set(self.permissions)) != len(self.permissions)
        ):
            raise ValueError("memory permissions must contain unique r/w/x characters")
        if type(self.data) is not bytes:
            raise ValueError("memory region data must be bytes")
        address = _as_u64(self.address)
        if len(self.data) > U64_MASK + 1 - address:
            raise ValueError("memory region range overflows u64")
        object.__setattr__(self, "address", address)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "address": self.address,
            "data_hex": self.data.hex(),
            "permissions": self.permissions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRegion":
        data = _object_field(data, "memory region")
        return cls(
            region_id=data.get("region_id"),
            address=data.get("address"),
            data=_bytes_from_hex(data.get("data_hex")),
            permissions=data.get("permissions", "rw"),
        )


def _memory_region_layout(regions):
    return [
        {"region_id": region.region_id, "size": len(region.data),
         "sha256": hashlib.sha256(region.data).hexdigest(),
         "permissions": region.permissions}
        for region in regions
    ]


@dataclass(frozen=True)
class CompareMask:
    outcome: bool = True
    checkpoint_pc: bool = True
    gpr_indices: tuple[int, ...] = ()
    memory_region_ids: tuple[str, ...] = ()
    signal: bool = False
    fault_pc: bool = False
    fault_address: bool = False
    extra_state_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "outcome",
            "checkpoint_pc",
            "signal",
            "fault_pc",
            "fault_address",
        ):
            strict_bool(getattr(self, name), name)
        if (
            not isinstance(self.gpr_indices, (list, tuple))
            or any(type(index) is not int for index in self.gpr_indices)
        ):
            raise ValueError("gpr compare indices must be integers")
        if (
            not isinstance(self.memory_region_ids, (list, tuple))
            or any(not isinstance(region_id, str) or not region_id for region_id in self.memory_region_ids)
        ):
            raise ValueError("memory compare region IDs must be non-empty strings")
        if (
            not isinstance(self.extra_state_keys, (list, tuple))
            or any(not isinstance(key, str) or not key for key in self.extra_state_keys)
        ):
            raise ValueError("extra_state_keys must contain non-empty strings")
        if len(self.gpr_indices) != len(set(self.gpr_indices)):
            raise ValueError("gpr compare indices must be unique")
        if len(self.memory_region_ids) != len(set(self.memory_region_ids)):
            raise ValueError("memory compare region IDs must be unique")
        if len(self.extra_state_keys) != len(set(self.extra_state_keys)):
            raise ValueError("extra state keys must be unique")
        for index in self.gpr_indices:
            if index < 0 or index > 31:
                raise ValueError("gpr compare index out of range")
        object.__setattr__(self, "gpr_indices", tuple(self.gpr_indices))
        object.__setattr__(self, "memory_region_ids", tuple(self.memory_region_ids))
        object.__setattr__(self, "extra_state_keys", tuple(self.extra_state_keys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "checkpoint_pc": self.checkpoint_pc,
            "gpr_indices": list(self.gpr_indices),
            "memory_region_ids": list(self.memory_region_ids),
            "signal": self.signal,
            "fault_pc": self.fault_pc,
            "fault_address": self.fault_address,
            "extra_state_keys": list(self.extra_state_keys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompareMask":
        data = _object_field(data, "compare mask")
        return cls(
            outcome=strict_bool(data.get("outcome", True), "outcome"),
            checkpoint_pc=strict_bool(data.get("checkpoint_pc", True), "checkpoint_pc"),
            gpr_indices=_tuple_int(data.get("gpr_indices", ())),
            memory_region_ids=_tuple_str(data.get("memory_region_ids", ())),
            signal=strict_bool(data.get("signal", False), "signal"),
            fault_pc=strict_bool(data.get("fault_pc", False), "fault_pc"),
            fault_address=strict_bool(data.get("fault_address", False), "fault_address"),
            extra_state_keys=_tuple_str(data.get("extra_state_keys", ())),
        )


@dataclass(frozen=True)
class InstructionMeta:
    instruction_id: str
    pc_offset: int
    mnemonic: str
    byte_offset: int
    byte_length: int
    rd: int | None = None
    rd_domain: str | None = None
    rs1: int | None = None
    rs1_domain: str | None = None
    rs2: int | None = None
    rs2_domain: str | None = None
    rs3: int | None = None
    rs3_domain: str | None = None
    immediate: int | None = None
    operand_fields: tuple[tuple[str, int], ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.instruction_id, str)
            or not self.instruction_id.strip()
            or not isinstance(self.mnemonic, str)
            or not self.mnemonic.strip()
        ):
            raise ValueError("instruction_id and mnemonic are required")
        _strict_nonnegative_int(self.pc_offset, "pc_offset")
        _strict_nonnegative_int(self.byte_offset, "byte_offset")
        _strict_nonnegative_int(self.byte_length, "byte_length")
        if self.byte_length not in {2, 4, 6, 8, 10, 12, 14, 16}:
            raise ValueError("instruction byte length is unsupported")
        for register in (self.rd, self.rs1, self.rs2, self.rs3):
            if register is not None:
                _strict_nonnegative_int(register, "register index")
            if register is not None and register > 31:
                raise ValueError("instruction register index out of range")
        if self.immediate is not None and (
            not isinstance(self.immediate, int) or isinstance(self.immediate, bool)
        ):
            raise ValueError("immediate must be an integer")
        for field_name in ("rd_domain", "rs1_domain", "rs2_domain", "rs3_domain"):
            _optional_register_domain(getattr(self, field_name), field_name)
        if (
            not isinstance(self.operand_fields, (list, tuple))
            or any(
                not isinstance(item, (list, tuple)) or len(item) != 2
                for item in self.operand_fields
            )
        ):
            raise ValueError("instruction operand fields must be pairs")
        if (
            not isinstance(self.tags, (list, tuple))
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("instruction tags must be non-empty strings")
        if any(
            not isinstance(name, str)
            or not name
            or type(value) is not int
            or value < 0
            for name, value in self.operand_fields
        ):
            raise ValueError("instruction operand fields must be non-negative integer pairs")
        field_names = [name for name, _value in self.operand_fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("instruction operand field names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionMeta":
        data = _object_field(data, "instruction metadata")
        return cls(
            instruction_id=data.get("instruction_id"),
            pc_offset=data.get("pc_offset"),
            mnemonic=data.get("mnemonic"),
            byte_offset=data.get("byte_offset"),
            byte_length=data.get("byte_length"),
            rd=_optional_int(data.get("rd")),
            rd_domain=_optional_register_domain(data.get("rd_domain"), "rd_domain"),
            rs1=_optional_int(data.get("rs1")),
            rs1_domain=_optional_register_domain(data.get("rs1_domain"), "rs1_domain"),
            rs2=_optional_int(data.get("rs2")),
            rs2_domain=_optional_register_domain(data.get("rs2_domain"), "rs2_domain"),
            rs3=_optional_int(data.get("rs3")),
            rs3_domain=_optional_register_domain(data.get("rs3_domain"), "rs3_domain"),
            immediate=_optional_int(data.get("immediate")),
            operand_fields=_operand_fields(data.get("operand_fields", ())),
            tags=_tuple_str(data.get("tags", ())),
        )


@dataclass(frozen=True)
class BlockMeta:
    block_id: str
    start_offset: int
    end_offset: int
    executed_expected: bool
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("block_id is required")
        _strict_nonnegative_int(self.start_offset, "start_offset")
        _strict_nonnegative_int(self.end_offset, "end_offset")
        if self.end_offset <= self.start_offset:
            raise ValueError("block offsets must be non-empty")
        strict_bool(self.executed_expected, "executed_expected")
        if (
            not isinstance(self.tags, (list, tuple))
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("block tags must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockMeta":
        data = _object_field(data, "block metadata")
        return cls(
            block_id=data.get("block_id"),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            executed_expected=strict_bool(data.get("executed_expected"), "executed_expected"),
            tags=_tuple_str(data.get("tags", ())),
        )


@dataclass(frozen=True)
class TranslationEvidence:
    backend: str
    expected_path: str
    tested_pc_seen: bool = False
    tested_pc_translated: bool = False
    tested_pc_executed: bool = False
    translated_count: int = 0
    execution_count: int = 0
    interpreter_count: int = 0
    fallback_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("translation evidence backend is required")
        if not isinstance(self.expected_path, str) or not self.expected_path:
            raise ValueError("translation evidence expected_path is required")
        if self.expected_path not in _TRANSLATION_PATHS:
            raise ValueError("translation evidence expected_path is unsupported")
        for name in ("tested_pc_seen", "tested_pc_translated", "tested_pc_executed"):
            strict_bool(getattr(self, name), name)
        for name in ("translated_count", "execution_count", "interpreter_count", "fallback_count"):
            _strict_nonnegative_int(getattr(self, name), name)
        if not isinstance(self.details, dict):
            raise ValueError("translation evidence details must be an object")
        _json_object(self.details, "translation evidence details")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TranslationEvidence | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("translation evidence must be an object")
        return cls(
            backend=data.get("backend"),
            expected_path=data.get("expected_path", "either"),
            tested_pc_seen=strict_bool(data.get("tested_pc_seen", False), "tested_pc_seen"),
            tested_pc_translated=strict_bool(
                data.get("tested_pc_translated", False), "tested_pc_translated"
            ),
            tested_pc_executed=strict_bool(
                data.get("tested_pc_executed", False), "tested_pc_executed"
            ),
            translated_count=_strict_nonnegative_int(
                data.get("translated_count", 0), "translated_count"
            ),
            execution_count=_strict_nonnegative_int(
                data.get("execution_count", 0), "execution_count"
            ),
            interpreter_count=_strict_nonnegative_int(
                data.get("interpreter_count", 0), "interpreter_count"
            ),
            fallback_count=_strict_nonnegative_int(
                data.get("fallback_count", 0), "fallback_count"
            ),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class TestCase:
    testcase_id: str
    base_program_id: str
    input_id: str
    generation_rule_id: str
    isa_profile: str
    code_bytes: bytes
    code_address: int
    entry_checkpoint_address: int
    exit_checkpoint_address: int
    initial_gpr: tuple[int, ...]
    initial_memory_regions: tuple[MemoryRegion, ...]
    compare_mask: CompareMask
    functional_tags: tuple[str, ...]
    layout_tags: tuple[str, ...]
    instruction_meta: tuple[InstructionMeta, ...]
    block_meta: tuple[BlockMeta, ...]
    dataflow_meta: dict[str, Any]
    expected_path: str = "either"
    observability_contract: ObservabilityContract | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "testcase_id",
            "base_program_id",
            "input_id",
            "generation_rule_id",
            "isa_profile",
            "expected_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        if self.expected_path not in _TRANSLATION_PATHS:
            raise ValueError("expected_path is unsupported")
        from .spec_definedness import (
            is_canonical_isa_profile,
            profile_uses_compressed_encoding,
        )

        if not is_canonical_isa_profile(self.isa_profile):
            raise ValueError("isa_profile must be a canonical RISC-V profile")
        if type(self.code_bytes) is not bytes or not self.code_bytes:
            raise ValueError("code_bytes must be non-empty bytes")
        if not isinstance(self.initial_gpr, (list, tuple)):
            raise ValueError("initial_gpr must be a list")
        if len(self.initial_gpr) != 32:
            raise ValueError("initial_gpr must contain 32 registers")
        for field_name in (
            "initial_memory_regions",
            "instruction_meta",
            "block_meta",
        ):
            if not isinstance(getattr(self, field_name), (list, tuple)):
                raise ValueError(f"{field_name} must be a list")
        for field_name, item_type in (
            ("initial_memory_regions", MemoryRegion),
            ("instruction_meta", InstructionMeta),
            ("block_meta", BlockMeta),
        ):
            if any(not isinstance(item, item_type) for item in getattr(self, field_name)):
                raise ValueError(f"{field_name} contains an invalid item")
        for field_name in ("functional_tags", "layout_tags"):
            values = getattr(self, field_name)
            if (
                not isinstance(values, (list, tuple))
                or any(not isinstance(item, str) or not item.strip() for item in values)
            ):
                raise ValueError(f"{field_name} must be a list of non-empty strings")
            object.__setattr__(self, field_name, tuple(values))
        if self.observability_contract is not None and not isinstance(
            self.observability_contract, ObservabilityContract
        ):
            raise ValueError("observability_contract has an invalid type")
        if not isinstance(self.compare_mask, CompareMask):
            raise ValueError("compare_mask has an invalid type")
        _json_object(self.dataflow_meta, "dataflow_meta")
        normalized = tuple(_as_u64(value) for value in self.initial_gpr)
        xlen = 32 if self.isa_profile.startswith("rv32") else 64
        if any(value >= 1 << xlen for value in normalized):
            raise ValueError("initial_gpr value exceeds testcase XLEN")
        if normalized[0] != 0:
            raise ValueError("x0 initial value must be zero")
        code_address = _as_u64(self.code_address)
        alignment = 2 if profile_uses_compressed_encoding(self.isa_profile) else 4
        if code_address % alignment:
            raise ValueError("code address must be instruction-aligned")
        if len(self.code_bytes) % alignment and not expected_trap_from_dataflow(self.dataflow_meta):
            raise ValueError("code bytes must be instruction-aligned")
        entry_address = _as_u64(self.entry_checkpoint_address)
        exit_address = _as_u64(self.exit_checkpoint_address)
        if code_address >= 1 << xlen or exit_address > (1 << xlen) - 1:
            raise ValueError("testcase address exceeds XLEN")
        if entry_address != code_address or exit_address != code_address + len(self.code_bytes):
            raise ValueError("checkpoint addresses must bound code_bytes")
        object.__setattr__(self, "initial_gpr", normalized)
        object.__setattr__(self, "code_address", code_address)
        object.__setattr__(self, "entry_checkpoint_address", entry_address)
        object.__setattr__(self, "exit_checkpoint_address", exit_address)
        region_ids = tuple(region.region_id for region in self.initial_memory_regions)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("initial memory region IDs must be unique")
        regions = sorted(
            (region.address, region.address + len(region.data))
            for region in self.initial_memory_regions
            if region.data
        )
        if any(right[0] < left[1] for left, right in zip(regions, regions[1:])):
            raise ValueError("initial memory regions must not overlap")
        if any(
            region.address >= 1 << xlen
            or region.address + len(region.data) > 1 << xlen
            for region in self.initial_memory_regions
        ):
            raise ValueError("initial memory region address exceeds testcase XLEN")
        block_ids = tuple(block.block_id for block in self.block_meta)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block IDs must be unique")
        _validate_instruction_layout(self.code_bytes, self.instruction_meta)
        if any(block.end_offset > len(self.code_bytes) for block in self.block_meta):
            raise ValueError("block metadata range exceeds code_bytes length")
        object.__setattr__(self, "initial_memory_regions", tuple(self.initial_memory_regions))
        object.__setattr__(self, "instruction_meta", tuple(self.instruction_meta))
        object.__setattr__(self, "block_meta", tuple(self.block_meta))
    def to_dict(self) -> dict[str, Any]:
        return {
            "testcase_id": self.testcase_id,
            "base_program_id": self.base_program_id,
            "input_id": self.input_id,
            "generation_rule_id": self.generation_rule_id,
            "isa_profile": self.isa_profile,
            "code_hex": self.code_bytes.hex(),
            "code_address": self.code_address,
            "entry_checkpoint_address": self.entry_checkpoint_address,
            "exit_checkpoint_address": self.exit_checkpoint_address,
            "initial_gpr": list(self.initial_gpr),
            "initial_memory_regions": [region.to_dict() for region in self.initial_memory_regions],
            "compare_mask": self.compare_mask.to_dict(),
            "functional_tags": list(self.functional_tags),
            "layout_tags": list(self.layout_tags),
            "instruction_meta": [item.to_dict() for item in self.instruction_meta],
            "block_meta": [item.to_dict() for item in self.block_meta],
            "dataflow_meta": deepcopy(self.dataflow_meta),
            "expected_path": self.expected_path,
            "observability_contract": (
                self.observability_contract.to_dict() if self.observability_contract is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        data = _object_field(data, "testcase")
        return cls(
            testcase_id=data.get("testcase_id"),
            base_program_id=data.get("base_program_id"),
            input_id=data.get("input_id"),
            generation_rule_id=data.get("generation_rule_id"),
            isa_profile=data.get("isa_profile"),
            code_bytes=_bytes_from_hex(data.get("code_hex")),
            code_address=data.get("code_address"),
            entry_checkpoint_address=data.get("entry_checkpoint_address"),
            exit_checkpoint_address=data.get("exit_checkpoint_address"),
            initial_gpr=tuple(_sequence_field(data, "initial_gpr")),
            initial_memory_regions=tuple(
                MemoryRegion.from_dict(item)
                for item in _sequence_field(data, "initial_memory_regions")
            ),
            compare_mask=CompareMask.from_dict(data.get("compare_mask")),
            functional_tags=_tuple_str(data.get("functional_tags", ())),
            layout_tags=_tuple_str(data.get("layout_tags", ())),
            instruction_meta=tuple(
                InstructionMeta.from_dict(item)
                for item in _sequence_field(data, "instruction_meta")
            ),
            block_meta=tuple(
                BlockMeta.from_dict(item)
                for item in _sequence_field(data, "block_meta")
            ),
            dataflow_meta=data.get("dataflow_meta", {}),
            expected_path=data.get("expected_path", "either"),
            observability_contract=(
                ObservabilityContract.from_dict(data["observability_contract"])
                if data.get("observability_contract") is not None
                else None
            ),
        )


def _compact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    """Replace a large file-backed PC path with a verified trace reference."""
    if record.get("backend") not in {
        "renode-riscv64", "rax-riscv64", "rvvm-riscv64",
    }:
        return record
    executed_pcs = record.get("executed_pcs")
    if not isinstance(executed_pcs, list) or len(executed_pcs) <= INLINE_EXECUTED_PC_LIMIT:
        return record
    evidence = record.get("translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    artifact = details.get("renode_trace_artifact") if isinstance(details, Mapping) else None
    if isinstance(artifact, Mapping):
        trace_reference = dict(artifact)
    else:
        trace_path = details.get("trace_path") if isinstance(details, Mapping) else None
        if not isinstance(trace_path, str) or not trace_path:
            return record
        trace = Path(trace_path)
        if not trace.is_file():
            return record
        trace_reference = {
            "status": "recorded",
            "absolute_path": str(trace.resolve()),
            "bytes": trace.stat().st_size,
        }
        trace_sha = details.get("trace_sha256") if isinstance(details, Mapping) else None
        if isinstance(trace_sha, str) and trace_sha:
            trace_reference["sha256"] = trace_sha
    trace_reference["format"] = "text-pc"
    trace_reference["pc_count"] = len(executed_pcs)
    digest = None
    if isinstance(details, Mapping):
        digest = (
            details.get("renode_executed_pc_digest")
            or details.get("rvvm_executed_pc_digest")
        )
        path_identity = details.get("path_identity")
        if not digest and isinstance(path_identity, Mapping):
            digest = path_identity.get("digest")
    if isinstance(digest, str) and digest:
        trace_reference["pc_digest"] = digest
    compact = dict(record)
    compact["executed_pcs"] = []
    compact["instruction_count"] = None
    compact["executed_pc_count"] = len(executed_pcs)
    compact["executed_pc_trace"] = trace_reference
    raw_stderr = record.get("raw_stderr")
    if isinstance(raw_stderr, str):
        compact_stderr = "\n".join(
            "RV_EXECUTED_PCS_FILE=" + json.dumps(
                trace_reference, separators=(",", ":"),
            ) if line.startswith("RV_EXECUTED_PCS=") else line
            for line in raw_stderr.splitlines()
        )
        if raw_stderr.endswith("\n"):
            compact_stderr += "\n"
        compact["raw_stderr"] = compact_stderr
    extra_state = record.get("extra_state")
    compact["extra_state"] = {
        **(dict(extra_state) if isinstance(extra_state, Mapping) else {}),
        "executed_pc_trace": trace_reference,
    }
    return compact


@dataclass(frozen=True)
class Observation:
    backend: str
    outcome: str
    exit_code: int | None
    checkpoint_pc: int | None
    gpr: tuple[int | None, ...]
    memory_delta: dict[str, str]
    memory_digest: str | None
    signal: str | None
    signal_code: int | None
    fault_pc: int | None
    fault_address: int | None
    executed_pcs: tuple[int, ...]
    instruction_count: int | None
    translation_evidence: TranslationEvidence | None
    profile_id: str | None
    input_id: str | None
    raw_stdout: str
    raw_stderr: str
    binary_sha256: str | None
    tool_version: str | None
    extra_state: dict[str, Any] = field(default_factory=dict)
    contract_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("observation backend and outcome are required")
        if not isinstance(self.outcome, str) or self.outcome not in OBSERVATION_OUTCOMES:
            raise ValueError("observation outcome is not supported")
        if self.outcome in {"normal", "completed"} and self.exit_code != 0:
            raise ValueError("normal observation must have exit_code 0")
        if (
            self.outcome in {"normal", "completed"}
            and (
                self.signal is not None or self.signal_code is not None
            )
        ):
            raise ValueError("normal observation cannot carry an OS signal")
        if self.outcome == "nonzero-exit" and self.exit_code in (None, 0):
            raise ValueError("nonzero-exit observation must have a nonzero exit_code")
        if self.translation_evidence is not None and (
            not isinstance(self.translation_evidence, TranslationEvidence)
            or self.translation_evidence.backend != self.backend
        ):
            raise ValueError("observation translation evidence must match backend")
        if self.contract_error is not None and not isinstance(self.contract_error, str):
            raise ValueError("observation contract_error must be a string or null")
        if self.instruction_count is not None and (
            type(self.instruction_count) is not int or self.instruction_count < 0
        ):
            raise ValueError("observation instruction_count must be non-negative or null")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("observation exit_code must be an integer or null")
        if self.signal is not None and (
            not isinstance(self.signal, str) or not self.signal.strip()
        ):
            raise ValueError("observation signal must be a string or null")
        if self.signal_code is not None and (
            type(self.signal_code) is not int or self.signal_code <= 0
        ):
            raise ValueError("observation signal_code must be an integer or null")
        if (
            self.signal is not None
            and self.signal_code is not None
            and self.signal in _SIGNAL_CODE_BY_NAME
            and _SIGNAL_CODE_BY_NAME[self.signal] != self.signal_code
        ):
            raise ValueError("observation signal does not match signal_code")
        if self.memory_digest is not None and not is_sha256_digest(self.memory_digest):
            raise ValueError("observation memory_digest must be a SHA-256 digest or null")
        if self.binary_sha256 is not None and not is_sha256_digest(self.binary_sha256):
            raise ValueError("observation binary_sha256 must be a SHA-256 digest or null")
        if not isinstance(self.memory_delta, dict) or any(
            not isinstance(region_id, str)
            or not region_id
            or not isinstance(value, str)
            or len(value) % 2
            or any(char not in "0123456789abcdefABCDEF" for char in value)
            for region_id, value in self.memory_delta.items()
        ):
            raise ValueError("observation memory_delta must contain hex strings")
        if self.memory_digest is not None and "test-memory" in self.memory_delta:
            if hashlib.sha256(bytes.fromhex(self.memory_delta["test-memory"])).hexdigest() != self.memory_digest:
                raise ValueError("observation memory_digest does not match test-memory")
        if not isinstance(self.extra_state, dict):
            raise ValueError("observation extra_state must be an object")
        _json_object(self.extra_state, "observation extra_state")
        if not isinstance(self.raw_stdout, str) or not isinstance(self.raw_stderr, str):
            raise ValueError("observation raw output must be strings")
        for field_name in ("profile_id", "input_id", "tool_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"observation {field_name} must be a non-empty string or null")
        if not isinstance(self.gpr, (list, tuple)) or len(self.gpr) != 32:
            raise ValueError("observation gpr must contain 32 entries")
        for value in self.gpr:
            if value is not None:
                _as_u64(value)
        if self.gpr[0] is not None and self.gpr[0] != 0:
            raise ValueError("observation x0 must be zero")
        if not isinstance(self.executed_pcs, (list, tuple)):
            raise ValueError("observation executed_pcs must be a sequence")
        if self.instruction_count is not None \
                and self.instruction_count != len(self.executed_pcs):
            raise ValueError("observation instruction_count does not match executed_pcs")
        for field_name in ("checkpoint_pc", "fault_pc", "fault_address"):
            value = getattr(self, field_name)
            if value is not None:
                parsed = _as_u64(value)
                if field_name != "fault_address" and parsed & 1:
                    raise ValueError(f"observation {field_name} must be instruction-aligned")
        for value in self.executed_pcs:
            if _as_u64(value) & 1:
                raise ValueError("observation executed_pcs must be instruction-aligned")
        object.__setattr__(self, "gpr", tuple(self.gpr))
        object.__setattr__(self, "executed_pcs", tuple(self.executed_pcs))
        object.__setattr__(self, "memory_delta", dict(self.memory_delta))
        object.__setattr__(self, "extra_state", dict(self.extra_state))

    @property
    def memory_snapshot(self) -> dict[str, str]:
        return self.memory_delta

    @property
    def guest_elf_sha256(self) -> str | None:
        return self.binary_sha256

    def to_dict(self, *, compact_trace: bool = False) -> dict[str, Any]:
        record = {
            "backend": self.backend,
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "checkpoint_pc": self.checkpoint_pc,
            "gpr": list(self.gpr),
            "memory_delta": dict(self.memory_delta),
            "memory_snapshot": dict(self.memory_snapshot),
            "memory_digest": self.memory_digest,
            "signal": self.signal,
            "signal_code": self.signal_code,
            "fault_pc": self.fault_pc,
            "fault_address": self.fault_address,
            "executed_pcs": list(self.executed_pcs),
            "instruction_count": self.instruction_count,
            "translation_evidence": (
                self.translation_evidence.to_dict()
                if self.translation_evidence is not None
                else None
            ),
            "profile_id": self.profile_id,
            "input_id": self.input_id,
            "raw_stdout": self.raw_stdout,
            "raw_stderr": self.raw_stderr,
            "guest_elf_sha256": self.guest_elf_sha256,
            "binary_sha256": self.binary_sha256,
            "tool_version": self.tool_version,
            "extra_state": dict(self.extra_state),
            "contract_error": self.contract_error,
        }
        return _compact_trace_record(record) if compact_trace else record

def _runner_contract_gap_observation(
    backend: str,
    contract_error: str,
    raw_stderr: str = "",
    binary_sha256: str | None = None,
    profile_id: str | None = None,
    input_id: str | None = None,
    *,
    exit_code: int | None = None,
    raw_stdout: str = "",
    executed_pcs: tuple[int, ...] = (),
    instruction_count: int | None = None,
    translation_evidence: TranslationEvidence | None = None,
    tool_version: str | None = None,
    extra_state: dict[str, object] | None = None,
) -> Observation:
    return Observation(
        backend=backend,
        outcome=VERDICT_RUNNER_CONTRACT_GAP,
        exit_code=exit_code,
        checkpoint_pc=None,
        gpr=(None,) * 32,
        memory_delta={},
        memory_digest=None,
        signal=None,
        signal_code=None,
        fault_pc=None,
        fault_address=None,
        executed_pcs=executed_pcs,
        instruction_count=instruction_count,
        translation_evidence=translation_evidence,
        profile_id=profile_id,
        input_id=input_id,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        binary_sha256=binary_sha256,
        tool_version=tool_version,
        extra_state=dict(extra_state or {}),
        contract_error=contract_error,
    )


def _validate_instruction_layout(code_bytes: bytes, instruction_meta: tuple[InstructionMeta, ...]) -> None:
    if not instruction_meta:
        return
    from .riscv_encoding import form_for_mnemonic

    expected_offset = 0
    instruction_ids: set[str] = set()
    for item in instruction_meta:
        if item.instruction_id in instruction_ids:
            raise ValueError("instruction_meta instruction_id must be unique")
        instruction_ids.add(item.instruction_id)
        if item.pc_offset != item.byte_offset:
            raise ValueError("instruction_meta pc_offset must match byte_offset")
        form = form_for_mnemonic(item.mnemonic)
        if form is not None and item.byte_length != form.encoding_length_bytes:
            raise ValueError("instruction_meta byte_length must match catalog encoding length")
        if item.byte_offset != expected_offset:
            raise ValueError("instruction_meta must form a contiguous byte layout")
        end = item.byte_offset + item.byte_length
        if end > len(code_bytes):
            raise ValueError("instruction_meta byte range exceeds code_bytes length")
        expected_offset = end
    if expected_offset != len(code_bytes):
        raise ValueError("instruction_meta does not cover the full code_bytes payload")

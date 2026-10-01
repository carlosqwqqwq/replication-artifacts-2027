"""RISC-V 指令形式覆盖率；底层执行器只负责提供 PC 观测。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from analysis.elf_features import (
        DECODER_VERSION,
        SPECS,
        _eligible_forms,
        _legal_form_encoding,
        _form_unit_id,
        metric_registry_for_profile,
    )
    from framework.spec_definedness import (
        is_canonical_isa_profile,
    )
    from framework.riscv_catalog import (
        CATALOG_PARTITION_EXPERIMENTAL_UNRATIFIED,
        CATALOG_PARTITION_RATIFIED,
        OFFICIAL_ALL_CATALOG_FORMS,
        catalog_partition_for_form,
    )
except ModuleNotFoundError:
    from elf_features import (  # type: ignore[no-redef]
        DECODER_VERSION,
        SPECS,
        _eligible_forms,
        _legal_form_encoding,
        _form_unit_id,
        metric_registry_for_profile,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from framework.spec_definedness import (  # type: ignore[no-redef]
        is_canonical_isa_profile,
    )
    from framework.riscv_catalog import (
        CATALOG_PARTITION_EXPERIMENTAL_UNRATIFIED,
        CATALOG_PARTITION_RATIFIED,
        OFFICIAL_ALL_CATALOG_FORMS,
        catalog_partition_for_form,
    )


SCHEMA = "rq1-rv-instruction-coverage-v2"
BATCH_SCHEMA = "rq1-simulator-coverage-batch-v6"
METRICS = ("GenCov", "ExecCov", "OpcodeCov")
RV_INSTRUCTION_METRICS = METRICS
# Fallback denominator used only when a run carries no usable ISA profile.
# 放开 ISA/扩展族后，整轮分母由实际执行到的 profile 并集派生（见
# _union_profile）；生成器可以产出自身支持的任何扩展，模拟器不支持的指令
# 在执行期报错并按目标侧终态记录，不影响源码覆盖率分母。
EXPERIMENT_COVERAGE_PROFILE = "rv64imc_zicsr_zifencei"
EXPERIMENT_PRIVILEGE_MODE = "user"
_DYNAMIC_METRICS = frozenset({"ExecCov", "OpcodeCov"})
_STATUSES = frozenset({"observed", "partial", "gap", "NA"})
OPCODE_CATALOG_SCHEMA = "rq1-rv-opcode-catalog-coverage-v1"
OPCODE_CATALOG_ID = (
    "riscv-opcodes@aeb94df5dd272d1893d43af270dd8d1b2fef699b"
    "+rq1-local-overlay-v1"
)
OPCODE_KEY_SCHEMA = "full-fixed-mask-match-v1"
OPCODE_CATALOG_PARTITION_VENDOR_OVERLAY = "local-vendor-overlay"


def _digest(values: Iterable[str]) -> str:
    canonical = sorted({str(value) for value in values})
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _field(form: Any, name: str) -> Any:
    if isinstance(form, Mapping):
        if name in form:
            return form.get(name)
        if name == "encoding_length_bytes":
            length_bits = form.get("length_bits")
            try:
                bits = _integer(length_bits)
            except (TypeError, ValueError, OverflowError):
                return None
            return bits // 8 if bits > 0 and bits % 8 == 0 else None
        return None
    return getattr(form, name, None)


def _integer(value: Any) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


def _fixed_pattern(match: int, mask: int, high: int, low: int) -> str:
    """Keep every fixed decoder bit and wildcard operand bits."""
    return "".join(
        str((match >> bit) & 1) if (mask >> bit) & 1 else "*"
        for bit in range(high, low - 1, -1)
    )


def _opcode_id(form: Any) -> str:
    width_value = _field(form, "encoding_length_bytes")
    width = (
        _integer(form["length_bits"])
        if isinstance(form, Mapping) and "length_bits" in form
        else _integer(width_value) * 8
    )
    mask = _integer(_field(form, "mask"))
    match = _integer(_field(form, "match"))
    if width == 32:
        # Keep the major opcode plus every fixed decoder bit. Operand bits are
        # wildcards, so immediate/register values do not create fake classes.
        major = f"0x{match & 0x7f:02x}"
        fixed = _fixed_pattern(match, mask, 31, 7)
        return f"opcode:32:o={major}:fixed={fixed}"
    if width == 16:
        quadrant = (
            f"0x{match & 0x3:x}" if mask & 0x3 == 0x3
            else _fixed_pattern(match, mask, 1, 0)
        )
        funct3 = (
            f"0x{(match >> 13) & 0x7:x}" if mask & 0xe000 == 0xe000
            else _fixed_pattern(match, mask, 15, 13)
        )
        return f"opcode:16:q={quadrant}:f3={funct3}"
    return f"opcode:{width}:match=0x{match:x}"


def _catalog_opcode_id(form: Any) -> str:
    """Return the exact fixed-bit pattern for any catalog instruction width."""
    width = _integer(_field(form, "encoding_length_bytes")) * 8
    mask = _integer(_field(form, "mask"))
    match = _integer(_field(form, "match"))
    return f"opcode:{width}:fixed={_fixed_pattern(match, mask, width - 1, 0)}"


def _catalog_partition(form: Any) -> str:
    if any(
        str(extension).startswith(("rv_x", "rv32_x", "rv64_x"))
        for extension in getattr(form, "source_extensions", ())
    ):
        return OPCODE_CATALOG_PARTITION_VENDOR_OVERLAY
    return catalog_partition_for_form(form)


OPCODE_CATALOG_FORM_TO_KEY = {
    _form_unit_id(form): _catalog_opcode_id(form)
    for form in OFFICIAL_ALL_CATALOG_FORMS
}
OPCODE_CATALOG_FORM_BY_ID = {
    _form_unit_id(form): form for form in OFFICIAL_ALL_CATALOG_FORMS
}
OPCODE_CATALOG_KEYS = frozenset(OPCODE_CATALOG_FORM_TO_KEY.values())
OPCODE_CATALOG_FORMS_BY_WIDTH = {
    width: tuple(sorted(
        (form for form in OFFICIAL_ALL_CATALOG_FORMS
         if int(form.encoding_length_bytes) * 8 == width),
        key=lambda form: (-int(form.mask).bit_count(), int(form.match), form.mnemonic),
    ))
    for width in sorted({
        int(form.encoding_length_bytes) * 8
        for form in OFFICIAL_ALL_CATALOG_FORMS
    })
}
OPCODE_CATALOG_PARTITION_KEYS = {
    partition: frozenset(
        _catalog_opcode_id(form) for form in OFFICIAL_ALL_CATALOG_FORMS
        if _catalog_partition(form) == partition
    )
    for partition in (
        CATALOG_PARTITION_RATIFIED,
        CATALOG_PARTITION_EXPERIMENTAL_UNRATIFIED,
        OPCODE_CATALOG_PARTITION_VENDOR_OVERLAY,
    )
}
OPCODE_CATALOG_DENOMINATOR = len(OPCODE_CATALOG_KEYS)
OPCODE_CATALOG_REGISTRY_SHA256 = _digest(OPCODE_CATALOG_KEYS)
OPCODE_CATALOG_FORM_COUNT = len(OFFICIAL_ALL_CATALOG_FORMS)


def opcode_unit_from_encoding(width_bits: int, raw: int) -> str | None:
    """Legacy raw-only fallback; valid rows use the form-derived decoder key."""
    if width_bits == 32:
        return f"opcode:32:0x{raw & 0x7f:02x}"
    if width_bits == 16:
        value = (((raw >> 13) & 0x7) << 2) | (raw & 0x3)
        return f"opcode:16:0x{value:02x}"
    return None


def registry_for_profile(
    profile: str, privilege_mode: str = "user",
) -> dict[str, Any]:
    """Return the form and opcode universes for one profile or XLEN union."""

    components = sorted(set(str(profile).split("+")))
    if len(components) > 1:
        if any(not is_canonical_isa_profile(item) for item in components):
            raise ValueError(f"invalid ISA profile union: {profile}")
        registries = [
            registry_for_profile(item, privilege_mode) for item in components
        ]
        metrics = {
            name: sorted({unit for item in registries
                          for unit in item["metrics"][name]})
            for name in registries[0]["metrics"]
        }
        form_units = sorted({unit for item in registries for unit in item["form_units"]})
        opcode_units = sorted({unit for item in registries for unit in item["opcode_units"]})
        form_to_opcode = {
            form: opcode
            for item in registries for form, opcode in item["form_to_opcode"].items()
        }
        profile_id = f"{'+'.join(components)}|{privilege_mode}|profile-union-v1"
        registry_digest = hashlib.sha256(json.dumps({
            "profile_id": profile_id,
            "component_registries": sorted(
                (item["coverage_profile_id"], item["coverage_registry_sha256"])
                for item in registries
            ),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        base = {
            **registries[0],
            "isa_profile": "+".join(components),
            "privilege_mode": privilege_mode,
            "eligible_form_count": len(form_units),
            "metrics": metrics,
            "registry_sha256": registry_digest,
            "metric_profile_id": profile_id,
        }
        return {
            **base,
            "form_units": form_units,
            "opcode_units": opcode_units,
            "form_to_opcode": form_to_opcode,
            "coverage_profile_id": profile_id,
            "coverage_registry_sha256": registry_digest,
            "form_registry_sha256": _digest(form_units),
            "opcode_registry_sha256": _digest(opcode_units),
        }

    base = metric_registry_for_profile(profile, privilege_mode)
    form_to_opcode = {
        _form_unit_id(form): _opcode_id(form)
        for form in _eligible_forms(profile, privilege_mode)
    }
    opcode_units = sorted(set(form_to_opcode.values()))
    form_units = list(base["metrics"]["ICov-encoding"])
    return {
        **base,
        "form_units": form_units,
        "opcode_units": opcode_units,
        "form_to_opcode": form_to_opcode,
        "coverage_profile_id": base["metric_profile_id"],
        "coverage_registry_sha256": base["registry_sha256"],
        "form_registry_sha256": _digest(form_units),
        "opcode_registry_sha256": _digest(opcode_units),
    }


def _profile(
    feature: Mapping[str, Any], coverage: Mapping[str, Any],
) -> tuple[str | None, str]:
    guest = coverage.get("guest")
    guest = guest if isinstance(guest, Mapping) else {}
    raw = (
        feature.get("isa_profile")
        or feature.get("lane")
        or coverage.get("isa_profile")
        or coverage.get("lane")
        or guest.get("isa_profile")
        or guest.get("coverage_profile_id")
    )
    if not raw:
        basis = coverage.get("coverage_basis")
        if isinstance(basis, Mapping):
            raw = basis.get("rv_coverage_profile_id") or basis.get(
                "coverage_profile_id",
            )
    value = str(raw or "")
    parts = value.split("|", 1)[0].split("/", 1)
    profile = parts[0].strip().lower()
    mode = str(
        feature.get("privilege_mode")
        or coverage.get("privilege_mode")
        or guest.get("privilege_mode")
        or (value.split("|")[1] if "|" in value else "")
        or "user"
    ).strip().lower()
    return (profile or None), mode


def _union_profile(profiles: Iterable[str]) -> str:
    """Derive the run denominator from the profiles that actually executed.

    放开 ISA/扩展族后，整轮分母跟随实际生成/执行的 profile 并集；混合
    XLEN 时保留每档 XLEN 各自的指令形式，不按频次丢弃较少出现的一档。
    """
    parsed = set()
    for value in profiles:
        profile = str(value or "").split("|", 1)[0].split("/", 1)[0].strip().lower()
        parsed.update(
            component for component in profile.split("+")
            if is_canonical_isa_profile(component)
        )
    if not parsed:
        return EXPERIMENT_COVERAGE_PROFILE
    return "+".join(sorted(parsed, key=lambda profile: (profile[:4], profile)))


def _union_profile_registry(
    profiles: Iterable[tuple[str, str]],
) -> tuple[str, str, dict[str, Any]]:
    """Build the exact denominator union for observed ISA/privilege pairs."""
    pairs = sorted({
        (str(profile).strip().lower(), str(mode).strip().lower())
        for profile, mode in profiles
        if str(mode).strip().lower() in {"user", "machine"}
        and all(is_canonical_isa_profile(part) for part in str(profile).split("+"))
    })
    if not pairs:
        profile, mode = EXPERIMENT_COVERAGE_PROFILE, EXPERIMENT_PRIVILEGE_MODE
        return profile, mode, registry_for_profile(profile, mode)
    profile = _union_profile(f"{item[0]}|{item[1]}" for item in pairs)
    modes = {mode for _, mode in pairs}
    if len(modes) == 1:
        mode = next(iter(modes))
        return profile, mode, registry_for_profile(profile, mode)

    registries = [registry_for_profile(*item) for item in pairs]
    metrics = {
        name: sorted({unit for item in registries for unit in item["metrics"][name]})
        for name in registries[0]["metrics"]
    }
    form_units = sorted({unit for item in registries for unit in item["form_units"]})
    opcode_units = sorted({unit for item in registries for unit in item["opcode_units"]})
    form_to_opcode = {
        form: opcode
        for item in registries for form, opcode in item["form_to_opcode"].items()
    }
    privilege_mode = "mixed"
    profile_id = f"{profile}|{privilege_mode}|profile-pair-union-v1"
    component_registries = sorted(
        (pair, item["coverage_profile_id"], item["coverage_registry_sha256"])
        for pair, item in zip(pairs, registries)
    )
    registry_digest = hashlib.sha256(json.dumps({
        "profile_id": profile_id,
        "component_registries": component_registries,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return profile, privilege_mode, {
        **registries[0],
        "isa_profile": profile,
        "privilege_mode": privilege_mode,
        "eligible_form_count": len(form_units),
        "metrics": metrics,
        "registry_sha256": registry_digest,
        "metric_profile_id": profile_id,
        "form_units": form_units,
        "opcode_units": opcode_units,
        "form_to_opcode": form_to_opcode,
        "coverage_profile_id": profile_id,
        "coverage_registry_sha256": registry_digest,
        "form_registry_sha256": _digest(form_units),
        "opcode_registry_sha256": _digest(opcode_units),
    }


def _status(value: object) -> str:
    return value if value in _STATUSES else "gap"


def _declared_case_count(value: object) -> tuple[int, bool]:
    """Parse a persisted case count without truncating malformed numbers."""
    if value is None:
        return 0, True
    if type(value) is int:
        return (value, value >= 0)
    if isinstance(value, float):
        return (
            (int(value), True)
            if math.isfinite(value) and value.is_integer() and value >= 0
            else (0, False)
        )
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip()), True
    return 0, False


def _catalog_entry_case_count(entry: Mapping[str, Any]) -> tuple[int, bool]:
    if "case_count" not in entry:
        return 1, True
    return _declared_case_count(entry.get("case_count"))


def _metric(
    covered: set[str],
    eligible: set[str],
    status: str,
    aggregation: str,
    reason: str | None = None,
) -> dict[str, Any]:
    status = _status(status)
    result: dict[str, Any] = {
        "covered": len(covered),
        "eligible": len(eligible),
        "value": (
            round(len(covered) / len(eligible), 6)
            if eligible and status == "observed"
            else None
        ),
        "status": status,
        "covered_units": sorted(covered),
        "eligible_units": sorted(eligible),
        "registry_sha256": _digest(eligible),
        "aggregation": aggregation,
    }
    if reason:
        result["reason"] = reason
    return result


def _registry(feature: Mapping[str, Any]) -> dict[str, Any] | None:
    value = str(feature.get("isa_profile") or feature.get("lane") or "")
    profile = value.split("|", 1)[0].split("/", 1)[0].strip().lower()
    if not profile:
        return None
    try:
        return registry_for_profile(
            profile, str(feature.get("privilege_mode") or "user"),
        )
    except ValueError:
        return None


def _opcode_for(
    item: Mapping[str, Any], form_to_opcode: Mapping[str, str] | None = None,
) -> str | None:
    explicit = item.get("opcode_unit_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    form = item.get("encoding_id")
    if form_to_opcode is not None:
        mapped = form_to_opcode.get(str(form)) if form else None
        if mapped:
            return mapped
    mnemonic = str(item.get("mnemonic") or "")
    if not mnemonic:
        encoding = item.get("encoding_id")
        if isinstance(encoding, str) and encoding.startswith("enc:"):
            mnemonic = encoding.rsplit(":", 1)[-1]
    form_spec = SPECS.get(mnemonic) or SPECS.get(mnemonic.replace("_", "."))
    if form_spec is not None:
        return _opcode_id(form_spec)
    try:
        width = int(item.get("width_bits"))
        raw_value = item.get("raw")
        raw = int(raw_value, 0) if isinstance(raw_value, str) else int(raw_value)
        return opcode_unit_from_encoding(width, raw)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _form_opcode_consistent(
    item: Mapping[str, Any], form_to_opcode: Mapping[str, str],
) -> bool:
    """Reject a feature row whose raw encoding disagrees with its form id."""
    form = item.get("encoding_id")
    expected = form_to_opcode.get(str(form)) if form else None
    if expected is None:
        return True
    explicit = item.get("opcode_unit_id")
    if explicit not in (None, ""):
        if not isinstance(explicit, str) or not explicit or explicit != expected:
            return False
    raw_value = item.get("raw")
    width_value = item.get("width_bits")
    if (raw_value is None) != (width_value is None):
        return False
    if raw_value is None:
        return True
    if isinstance(raw_value, bool) or isinstance(width_value, bool):
        return False
    try:
        width = int(width_value)
        raw = int(raw_value, 0) if isinstance(raw_value, str) else int(raw_value)
    except (TypeError, ValueError, OverflowError):
        return False
    if width not in {16, 32} or raw < 0 or raw >= 1 << width:
        return False
    form_name = str(form).rsplit(":", 1)[-1]
    spec = SPECS.get(form_name) or SPECS.get(form_name.replace("_", "."))
    if spec is None:
        return True
    try:
        expected_width = int(spec["length_bits"])
        mask = int(str(spec["mask"]), 0)
        match = int(str(spec["match"]), 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return width == expected_width and raw & mask == match


def _catalog_opcode_from_raw(width: int, raw: int) -> tuple[str | None, bool]:
    matches = []
    for form in OPCODE_CATALOG_FORMS_BY_WIDTH.get(width, ()):
        mask = int(form.mask)
        if raw & mask == int(form.match) and _legal_form_encoding(form, raw):
            matches.append((mask.bit_count(), _catalog_opcode_id(form)))
    if not matches:
        return None, False
    specificity = max(item[0] for item in matches)
    keys = {key for bits, key in matches if bits == specificity}
    return (next(iter(keys)), False) if len(keys) == 1 else (None, True)


def _catalog_opcode_for(
    item: Mapping[str, Any], *, trust_decoded_form_id: bool = False,
) -> tuple[str | None, bool]:
    """Map one instruction to the catalog, validating fallback decodes against raw bits."""
    form_id = item.get("encoding_id")
    form_id = str(form_id) if form_id else ""
    key = OPCODE_CATALOG_FORM_TO_KEY.get(form_id)
    form = OPCODE_CATALOG_FORM_BY_ID.get(form_id) if form_id else None
    form_id_known = form is not None
    if form_id_known and trust_decoded_form_id:
        explicit = item.get("opcode_unit_id")
        if explicit not in (None, "") and explicit != key:
            return None, True
        return key, False
    if form is None:
        mnemonic = str(item.get("mnemonic") or "")
        if not mnemonic and isinstance(form_id, str) and form_id.startswith("enc:"):
            mnemonic = form_id.rsplit(":", 1)[-1]
        form_spec = SPECS.get(mnemonic) or SPECS.get(mnemonic.replace("_", "."))
        form = form_spec.get("form") if isinstance(form_spec, Mapping) else None

    raw_value = item.get("raw")
    width_value = item.get("width_bits")
    has_raw = raw_value is not None or width_value is not None
    if has_raw:
        if raw_value is None or width_value is None or isinstance(raw_value, bool) \
                or isinstance(width_value, bool):
            return None, True
        try:
            width = int(width_value)
            raw = int(raw_value, 0) if isinstance(raw_value, str) else int(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None, True
        # objdump recognizes 48/64-bit instruction encodings as well. The
        # pinned catalog currently has only 16/32-bit forms, so wider
        # encodings are out-of-catalog evidence, not malformed decodes.
        if width not in {16, 32, 48, 64} or raw < 0 or raw >= 1 << width:
            return None, True
        if form is not None:
            key = key or OPCODE_CATALOG_FORM_TO_KEY.get(_form_unit_id(form))
            explicit = item.get("opcode_unit_id")
            if (key is None or width != int(form.encoding_length_bytes) * 8
                    or raw & int(form.mask) != int(form.match)
                    or (not form_id_known and not _legal_form_encoding(form, raw))
                    or explicit not in (None, "") and explicit != key):
                return None, True
            return key, False
        raw_key, ambiguous = _catalog_opcode_from_raw(width, raw)
        if ambiguous:
            return None, True
        explicit = item.get("opcode_unit_id")
        if isinstance(explicit, str) and explicit in OPCODE_CATALOG_KEYS:
            return (explicit, False) if raw_key == explicit else (None, True)
        return raw_key, False

    if form is not None:
        key = key or OPCODE_CATALOG_FORM_TO_KEY.get(_form_unit_id(form))
        explicit = item.get("opcode_unit_id")
        if explicit not in (None, "") and explicit != key:
            return None, True
        return key, False
    explicit = item.get("opcode_unit_id")
    return explicit if isinstance(explicit, str) and explicit in OPCODE_CATALOG_KEYS else None, False


def _catalog_metric(
    covered: set[str], status: str, aggregation: str,
    reason: str | None = None, out_of_catalog: Iterable[str] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "covered": len(covered),
        "eligible": OPCODE_CATALOG_DENOMINATOR,
        "value": None,
        "status": _status(status),
        "covered_units": sorted(covered),
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "aggregation": aggregation,
    }
    excluded = sorted(set(out_of_catalog))
    if excluded:
        result["out_of_catalog_units"] = excluded
    if reason:
        result["reason"] = reason
    return result


def _catalog_case_result(
    generated: set[str], generated_status: str, generated_reason: str | None,
    executed: set[str], executed_status: str, executed_reason: str | None,
    generated_outside: Iterable[str] = (), executed_outside: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "case-evidence",
        "catalog_id": OPCODE_CATALOG_ID,
        "key_schema": OPCODE_KEY_SCHEMA,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "metrics": {
            "GenOpcodeCov": _catalog_metric(
                generated, generated_status, "generated-final-elf-text",
                generated_reason, generated_outside,
            ),
            "ExecOpcodeCov": _catalog_metric(
                executed, executed_status, "dynamic-executed-pc-union",
                executed_reason, executed_outside,
            ),
        },
    }


def _catalog_dynamic_metric(
    coverage: Mapping[str, Any], raw_values: list[Any] | None, pcs: list[int],
    mapped: list[Mapping[str, Any]], executed: set[str], outside: set[str],
    mismatch: bool, *, feature_problem: str | None = None,
) -> dict[str, Any]:
    guest = coverage.get("guest")
    guest_metrics = guest.get("metrics") if isinstance(guest, Mapping) else None
    pc_metric = guest_metrics.get("PCov") if isinstance(guest_metrics, Mapping) else None
    pc_status_value = pc_metric.get("status") if isinstance(pc_metric, Mapping) else None
    pc_status = _status(pc_status_value) if pc_status_value is not None else None
    coverage_status_value = coverage.get("status")
    coverage_status = (
        _status(coverage_status_value) if coverage_status_value is not None else None
    )
    events = coverage.get("simulator_events")
    trace = events.get("trace") if isinstance(events, Mapping) else None
    trace_complete = bool(
        isinstance(trace, Mapping)
        and trace.get("truncated") is False
        and isinstance(trace.get("errors"), list)
        and not trace.get("errors")
        and type(trace.get("ignored_records")) is int
        and trace.get("ignored_records") == 0
    )
    if raw_values is None:
        status = "NA" if "NA" in {pc_status, coverage_status} else "gap"
        reason = "guest-pc-trace-missing"
    elif len(pcs) != len(raw_values):
        status, reason = "gap", "guest-pc-record-invalid"
    elif len(mapped) != len(pcs):
        status, reason = "gap", "guest-pc-map-mismatch"
    elif mismatch:
        status, reason = "gap", "dynamic-opcode-encoding-mismatch"
    elif isinstance(trace, Mapping) and trace.get("truncated") is True:
        status = "partial" if executed or outside else "gap"
        reason = "dynamic-trace-incomplete"
    elif isinstance(trace, Mapping) and trace.get("errors"):
        status = "partial" if executed or outside else "gap"
        reason = "dynamic-trace-error"
    elif not pcs:
        status = "NA" if "NA" in {pc_status, coverage_status} else "gap"
        reason = "executed-opcode-evidence-empty"
    elif feature_problem:
        status = "partial" if executed or outside else "gap"
        reason = feature_problem
    elif trace_complete:
        status, reason = "observed", None
    elif pc_status in {"gap", "partial", "NA"}:
        status = pc_status
        reason = str(
            (pc_metric or {}).get("reason")
            or coverage.get("reason")
            or "dynamic-trace-incomplete"
        )
    elif coverage_status in {"gap", "partial", "NA"}:
        status = coverage_status
        reason = str(coverage.get("reason") or "dynamic-trace-incomplete")
    else:
        status, reason = "observed", None
    return _catalog_metric(
        executed, status, "dynamic-executed-pc-union", reason, outside,
    )


def _empty(
    profile: str | None,
    mode: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "aggregation_scope": "case-evidence",
        "status": _status(status),
        "reason": reason,
        "isa_profile": profile,
        "privilege_mode": mode,
        "dynamic_trace": {
            "records": 0, "valid_records": 0, "mapped_records": 0,
            "invalid_records": 0, "unmapped_records": 0, "complete": False,
        },
        "metrics": {},
    }
    if profile is None:
        return result
    try:
        registry = registry_for_profile(profile, mode)
    except ValueError:
        return result
    result.update({
        "coverage_profile_id": registry["coverage_profile_id"],
        "coverage_registry_sha256": registry["coverage_registry_sha256"],
        "form_registry_sha256": registry["form_registry_sha256"],
        "opcode_registry_sha256": registry["opcode_registry_sha256"],
    })
    result["metrics"] = {
        "GenCov": _metric(
            set(), set(registry["form_units"]), status,
            "generated-elf-text", reason,
        ),
        "ExecCov": _metric(
            set(), set(registry["form_units"]), status,
            "dynamic-executed-instruction-trace", reason,
        ),
        "OpcodeCov": _metric(
            set(), set(registry["opcode_units"]), status,
            "dynamic-decoder-opcode-class", reason,
        ),
    }
    return result


def _coverage_pc_values(coverage: Mapping[str, Any]) -> list[Any] | None:
    guest = coverage.get("guest")
    metrics = guest.get("metrics") if isinstance(guest, Mapping) else None
    pc_metric = metrics.get("PCov") if isinstance(metrics, Mapping) else None
    if not isinstance(pc_metric, Mapping):
        pc_metric = guest.get("PCov") if isinstance(guest, Mapping) else None
    values = pc_metric.get("covered_units") if isinstance(pc_metric, Mapping) else None
    if not isinstance(values, list):
        values = coverage.get("trace_pcs")
    if not isinstance(values, list):
        return None
    return values


def _coverage_pcs(coverage: Mapping[str, Any]) -> list[int]:
    values = _coverage_pc_values(coverage)
    if values is None:
        return []
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            continue
        try:
            parsed = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= parsed < 1 << 64:
            result.append(parsed)
    return result


def _metric_item_eligible(item: Mapping[str, Any], metric: str) -> bool:
    """Honor coverage-only exclusions for expected trap scaffolding."""
    excluded = item.get("coverage_excluded_metrics", ())
    return not (
        item.get("coverage_excluded") is True
        or isinstance(excluded, (list, tuple, set, frozenset))
        and metric in excluded
    )


def _dynamic(
    feature: Mapping[str, Any],
    coverage: Mapping[str, Any],
    registry: Mapping[str, Any],
    catalog_by_pc: Mapping[int, tuple[str | None, bool]] | None = None,
) -> tuple[set[str], set[str], str, str | None, dict[str, Any], dict[str, Any]]:
    instructions = feature.get("instructions")
    by_pc = {
        item.get("address"): item
        for item in instructions if isinstance(item, Mapping)
        and type(item.get("address")) is int
    } if isinstance(instructions, list) else {}
    raw_values = _coverage_pc_values(coverage)
    pcs = _coverage_pcs(coverage)
    mapped = [by_pc[pc] for pc in pcs if pc in by_pc]
    raw_forms = {
        str(item.get("encoding_id")) for item in mapped
        if _metric_item_eligible(item, "ExecCov")
        if item.get("encoding_id")
    }
    forms = set(raw_forms)
    opcode_form_mismatch = False
    opcodes: set[str] = set()
    catalog_opcodes: set[str] = set()
    catalog_outside: set[str] = set()
    catalog_opcode_mismatch = False
    for item in mapped:
        if not _metric_item_eligible(item, "OpcodeCov"):
            continue
        opcode_form_mismatch |= not _form_opcode_consistent(
            item, registry["form_to_opcode"],
        )
        opcode = _opcode_for(item, registry["form_to_opcode"])
        if opcode:
            opcodes.add(opcode)
        catalog_opcode, catalog_mismatch = (
            catalog_by_pc.get(item["address"], (None, True))
            if catalog_by_pc is not None else _catalog_opcode_for(
                item, trust_decoded_form_id=(
                    feature.get("decoder_version") == DECODER_VERSION
                ),
            )
        )
        catalog_opcode_mismatch |= catalog_mismatch
        if catalog_opcode in OPCODE_CATALOG_KEYS:
            catalog_opcodes.add(catalog_opcode)
        else:
            catalog_outside.add(str(
                item.get("encoding_id") or item.get("mnemonic") or "unknown",
            ))
    opcodes.update(
        registry["form_to_opcode"].get(form)
        for form in forms
        if registry["form_to_opcode"].get(form)
    )
    guest = coverage.get("guest")
    guest_metrics = guest.get("metrics") if isinstance(guest, Mapping) else None
    pc_metric = (
        guest_metrics.get("PCov")
        if isinstance(guest_metrics, Mapping)
        else None
    )
    pc_raw_status = pc_metric.get("status") if isinstance(
        pc_metric, Mapping,
    ) else None
    pc_status = (
        _status(pc_raw_status)
        if pc_raw_status is not None
        else _status(coverage.get("status"))
    )
    coverage_status = (
        _status(coverage.get("status"))
        if coverage.get("status") is not None else pc_status
    )
    if "gap" in {coverage_status, pc_status}:
        status = "gap"
    elif "partial" in {coverage_status, pc_status}:
        status = "partial"
    elif "NA" in {coverage_status, pc_status}:
        status = "NA"
    else:
        status = "observed"
    reason = None
    if raw_values is None:
        reason = "guest-pc-trace-missing"
        if status == "observed":
            status = "gap"
    elif len(pcs) != len(raw_values):
        status, reason = "gap", "guest-pc-record-invalid"
    elif len(mapped) != len(pcs):
        status, reason = "gap", "guest-pc-map-mismatch"
    out_of_profile = raw_forms - set(registry["form_units"])
    reserved_out_of_profile = {
        form for form in out_of_profile if form.startswith("reserved:")
    }
    unknown_out_of_profile = out_of_profile - reserved_out_of_profile
    if unknown_out_of_profile:
        status = "gap"
        reason = "executed-form-outside-profile"
    elif reserved_out_of_profile and status != "gap":
        # Reserved encodings are retained in the PC map for diagnostics, but
        # they are not legal RQ1 instruction-universe members.  A dynamic hit
        # makes the execution channel partial; it must not invalidate the
        # ordinary generated/executed forms around the reserved padding.
        status = "partial"
        reason = "reserved-form-observed"
    elif opcode_form_mismatch:
        status = "gap"
        reason = "dynamic-opcode-form-mismatch"
    elif status == "observed" and not forms:
        status, reason = "gap", "executed-form-evidence-empty"
    elif status != "observed" and reason is None:
        reason = str(coverage.get("reason") or "dynamic-trace-incomplete")
    forms &= set(registry["form_units"])
    opcodes &= set(registry["opcode_units"])
    trace_evidence = {
        "records": len(raw_values) if raw_values is not None else 0,
        "valid_records": len(pcs),
        "mapped_records": len(mapped),
        "invalid_records": max(
            0, len(raw_values) - len(pcs),
        ) if raw_values is not None else 0,
        "unmapped_records": max(0, len(pcs) - len(mapped)),
        "complete": bool(
            raw_values is not None
            and len(raw_values) == len(pcs) == len(mapped)
            and status == "observed"
        ),
    }
    catalog_metric = _catalog_dynamic_metric(
        coverage, raw_values, pcs, mapped,
        catalog_opcodes, catalog_outside, catalog_opcode_mismatch,
    )
    return forms, opcodes, status, reason, trace_evidence, catalog_metric


def _legacy_case_metrics(
    feature: Mapping[str, Any] | None,
    simulator_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project one static ELF feature map and one dynamic PC trace."""

    feature = feature if isinstance(feature, Mapping) else {}
    coverage = (
        simulator_coverage
        if isinstance(simulator_coverage, Mapping)
        else {}
    )
    profile, mode = _profile(feature, coverage)
    if not profile:
        return _empty(profile, mode, "gap", "isa-profile-missing")
    try:
        registry = registry_for_profile(profile, mode)
    except (TypeError, ValueError) as exc:
        return _empty(
            profile, mode, "gap", f"isa-profile-unsupported:{exc}",
        )
    guest = coverage.get("guest")
    if isinstance(guest, Mapping):
        declared_profile = guest.get("coverage_profile_id")
        declared_registry = guest.get("coverage_registry_sha256")
        if (declared_profile not in (None, registry["coverage_profile_id"])
                or declared_registry not in (None, registry["coverage_registry_sha256"])):
            return _empty(profile, mode, "gap", "coverage-registry-mismatch")
    if not feature:
        return _empty(profile, mode, "gap", "static-feature-map-missing")
    if feature.get("decoder_version") != DECODER_VERSION:
        return _empty(profile, mode, "gap", "static-feature-map-stale")
    instructions = feature.get("instructions")
    if feature.get("status", "ok") != "ok" or not isinstance(
        instructions, list,
    ) or not instructions:
        return _empty(profile, mode, "gap", "static-feature-map-invalid")
    if any(
        not isinstance(item, Mapping)
        or type(item.get("address")) is not int
        or not 0 <= item.get("address") < 1 << 64
        or not isinstance(item.get("encoding_id"), str)
        or not item.get("encoding_id")
        for item in instructions
    ):
        return _empty(profile, mode, "gap", "static-feature-map-incomplete")
    addresses = [item["address"] for item in instructions]
    if len(set(addresses)) != len(addresses):
        return _empty(profile, mode, "gap", "static-feature-map-duplicate-pc")

    if any(
        not _form_opcode_consistent(item, registry["form_to_opcode"])
        for item in instructions
    ):
        return _empty(profile, mode, "gap", "static-feature-map-encoding-mismatch")

    form_units = set(registry["form_units"])
    raw_generated: set[str] = set()
    generated_catalog_units: set[str] = set()
    generated_catalog_outside: set[str] = set()
    generated_catalog_mismatch = False
    catalog_by_pc: dict[int, tuple[str | None, bool]] = {}
    for item in instructions:
        catalog_opcode, catalog_mismatch = _catalog_opcode_for(
            item,
            trust_decoded_form_id=(
                item.get("encoding_id") in registry["form_to_opcode"]
            ),
        )
        catalog_by_pc[item["address"]] = (catalog_opcode, catalog_mismatch)
        if not _metric_item_eligible(item, "GenCov"):
            continue
        form = str(item.get("encoding_id") or "")
        if form:
            raw_generated.add(form)
        generated_catalog_mismatch |= catalog_mismatch
        if catalog_opcode in OPCODE_CATALOG_KEYS:
            generated_catalog_units.add(catalog_opcode)
        else:
            generated_catalog_outside.add(str(
                item.get("encoding_id") or item.get("mnemonic") or "unknown",
            ))
    generated_hit = raw_generated & form_units
    generated_native_excluded = raw_generated - form_units
    generated_reserved = {
        form for form in generated_native_excluded if form.startswith("reserved:")
    }
    # 放开 ISA/扩展族后不再引用固定 profile：分母就是本 case 自己声明的
    # profile 注册表。落在它之外的合法编码（catalog 已登记，只是该 profile
    # 未启用）只作为审计证据保留；只有 catalog 完全无法识别的编码才是 gap。
    generated_unknown = {
        form for form in generated_native_excluded
        if form.startswith("unknown:")
    }
    (
        dynamic_forms, dynamic_opcodes, dynamic_status, dynamic_reason,
        dynamic_trace, dynamic_catalog,
    ) = _dynamic(
        feature, coverage, registry, catalog_by_pc,
    )
    # A native generator may legally emit a form outside its declared lane
    # while the form is still inside the fixed RQ1 comparison universe (for
    # example, a compressed form in a wider tool lane).  Exclude that unit
    # from the native case denominator and retain it as audit evidence; only
    # an unknown form makes the static metric a real gap.
    generated_status = "observed" if not generated_unknown else "gap"
    generated_reason = (
        None if generated_status == "observed"
        else "generated-form-outside-experiment-profile"
    )
    if generated_status == "gap" or dynamic_status == "gap":
        overall = "gap"
    elif dynamic_status == "partial":
        overall = "partial"
    else:
        overall = "observed"
    reason = generated_reason or dynamic_reason
    generated_metric = _metric(
        generated_hit, form_units, generated_status,
        "generated-elf-text", generated_reason,
    )
    if generated_native_excluded:
        generated_metric["excluded_units"] = sorted(generated_native_excluded)
        generated_metric["excluded_reason"] = "native-profile-outside-fixed-scope"
    result = {
        "schema_version": SCHEMA,
        "aggregation_scope": "case-evidence",
        "status": overall,
        "reason": reason,
        "isa_profile": registry["isa_profile"],
        "privilege_mode": registry["privilege_mode"],
        "coverage_profile_id": registry["coverage_profile_id"],
        "coverage_registry_sha256": registry["coverage_registry_sha256"],
        "form_registry_sha256": registry["form_registry_sha256"],
        "opcode_registry_sha256": registry["opcode_registry_sha256"],
        "static_instruction_count": len(instructions),
        "dynamic_trace": dynamic_trace,
        "metrics": {
            "GenCov": generated_metric,
            "ExecCov": _metric(
                dynamic_forms, form_units, dynamic_status,
                "dynamic-executed-instruction-trace", dynamic_reason,
            ),
            "OpcodeCov": _metric(
                dynamic_opcodes, set(registry["opcode_units"]), dynamic_status,
                "dynamic-decoder-opcode-class", dynamic_reason,
            ),
        },
    }
    if "opcode_catalog_static_instructions" not in feature:
        result["_rv_opcode_catalog_case"] = _catalog_case_result(
            generated_catalog_units,
            "gap" if generated_catalog_mismatch else "observed",
            "static-opcode-encoding-mismatch" if generated_catalog_mismatch else None,
            dynamic_catalog["covered_units"],
            dynamic_catalog["status"],
            dynamic_catalog.get("reason"),
            generated_catalog_outside,
            dynamic_catalog.get("out_of_catalog_units", ()),
        )
    return result


def opcode_catalog_case_coverage(
    feature_value: Mapping[str, Any] | None,
    coverage_value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compute catalog metrics independently of the narrower legacy ISA profile."""
    feature = feature_value if isinstance(feature_value, Mapping) else {}
    coverage = coverage_value if isinstance(coverage_value, Mapping) else {}
    instructions = feature.get("instructions")
    execution_instructions = feature.get(
        "opcode_catalog_execution_instructions", instructions,
    )
    static_instructions = feature.get(
        "opcode_catalog_static_instructions", instructions,
    )
    static_units: set[str] = set()
    static_outside: set[str] = set()
    static_mismatch = False
    malformed_static = (
        not isinstance(static_instructions, list) or not static_instructions
    )
    by_pc: dict[int, Mapping[str, Any]] = {}
    catalog_by_pc: dict[int, tuple[str | None, bool]] = {}
    trust_decoded_form_id = feature.get("decoder_version") == DECODER_VERSION
    if isinstance(execution_instructions, list):
        for item in execution_instructions:
            if (isinstance(item, Mapping)
                    and type(item.get("address")) is int
                    and 0 <= item.get("address") < 1 << 64):
                by_pc[item["address"]] = item

    if isinstance(static_instructions, list):
        for item in static_instructions:
            if (not isinstance(item, Mapping)
                    or type(item.get("address")) is not int
                    or not 0 <= item.get("address") < 1 << 64
                    or not isinstance(item.get("encoding_id"), str)
                    or not item.get("encoding_id")):
                malformed_static = True
                continue
            address = item["address"]
            if address in catalog_by_pc:
                malformed_static = True
            catalog_by_pc[address] = _catalog_opcode_for(
                item, trust_decoded_form_id=trust_decoded_form_id,
            )
            if not _metric_item_eligible(item, "GenCov"):
                continue
            opcode, mismatch = catalog_by_pc[address]
            static_mismatch |= mismatch
            if opcode in OPCODE_CATALOG_KEYS:
                static_units.add(opcode)
            else:
                static_outside.add(str(
                    item.get("encoding_id") or item.get("mnemonic") or "unknown",
                ))

    feature_status = feature.get("status", "ok")
    feature_reason = feature.get("reason")
    instruction_count = feature.get("instruction_count")
    count_matches = (
        instruction_count is None
        or type(instruction_count) is int
        and instruction_count == len(static_instructions or ())
    )
    feature_recognizes_all_text = (
        feature_status == "ok"
        or feature_status == "gap"
        and feature_reason == "unknown-or-reserved-instruction-encoding"
        and count_matches
    )
    if feature.get("decoder_version") != DECODER_VERSION:
        gen_status, gen_reason = "gap", "static-feature-map-stale"
        feature_problem = "static-feature-map-stale"
    elif malformed_static or not count_matches:
        gen_status, gen_reason = "gap", "static-feature-map-incomplete"
        feature_problem = "static-feature-map-incomplete"
    elif static_mismatch:
        gen_status, gen_reason = "gap", "static-opcode-encoding-mismatch"
        feature_problem = "static-opcode-encoding-mismatch"
    elif feature_recognizes_all_text:
        # Unknown/out-of-catalog forms are explicitly reported and do not alter
        # coverage of the pinned catalog; known catalog forms are decoded from
        # the full form table even when the artifact's declared lane omits them.
        gen_status, gen_reason = "observed", None
        feature_problem = None
    else:
        gen_status = "partial" if static_units else "gap"
        gen_reason = str(feature_reason or "static-feature-map-invalid")
        feature_problem = gen_reason

    if "_rv_opcode_catalog_trace_pcs" in coverage:
        raw_values = coverage.get("_rv_opcode_catalog_trace_pcs")
        raw_values = raw_values if isinstance(raw_values, list) else None
        pcs = _coverage_pcs({"trace_pcs": raw_values}) if raw_values is not None else []
    else:
        raw_values = _coverage_pc_values(coverage)
        pcs = _coverage_pcs(coverage)
    executed_units: set[str] = set()
    executed_outside: set[str] = set()
    exec_mismatch = False
    mapped = [by_pc[pc] for pc in pcs if pc in by_pc]
    for item in mapped:
        if not _metric_item_eligible(item, "OpcodeCov"):
            continue
        opcode, mismatch = catalog_by_pc.get(
            item["address"], (None, True),
        )
        exec_mismatch |= mismatch
        if opcode in OPCODE_CATALOG_KEYS:
            executed_units.add(opcode)
        else:
            executed_outside.add(str(
                item.get("encoding_id") or item.get("mnemonic") or "unknown",
            ))
    exec_metric = _catalog_dynamic_metric(
        coverage, raw_values, pcs, mapped, executed_units, executed_outside,
        exec_mismatch, feature_problem=feature_problem,
    )
    return {
        "schema_version": OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "case-evidence",
        "catalog_id": OPCODE_CATALOG_ID,
        "key_schema": OPCODE_KEY_SCHEMA,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "metrics": {
            "GenOpcodeCov": _catalog_metric(
                static_units, gen_status, "generated-final-elf-text",
                gen_reason, static_outside,
            ),
            "ExecOpcodeCov": exec_metric,
        },
    }


def case_metrics(
    feature: Mapping[str, Any] | None,
    simulator_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return historical ISA metrics plus the independent full-catalog metric."""
    result = _legacy_case_metrics(feature, simulator_coverage)
    catalog = result.pop("_rv_opcode_catalog_case", None)
    result["rv_opcode_catalog_coverage"] = (
        catalog if isinstance(catalog, Mapping) else
        opcode_catalog_case_coverage(feature, simulator_coverage)
    )
    return result


def attach_case_metrics(
    feature: Mapping[str, Any] | None,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    result = case_metrics(feature, coverage)
    catalog_result = result.pop("rv_opcode_catalog_coverage", None)
    coverage["rv_instruction_coverage"] = result
    if isinstance(catalog_result, Mapping):
        coverage["rv_opcode_catalog_coverage"] = dict(catalog_result)
    guest = coverage.get("guest")
    if isinstance(guest, dict):
        metrics = guest.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics.update(result["metrics"])
        scope = guest.setdefault("metric_scope", {})
        if isinstance(scope, dict):
            scope.update({name: "case-evidence" for name in METRICS})
    return coverage


def _profile_from_metric(
    metric: Mapping[str, Any],
) -> tuple[str, str] | None:
    value = (
        metric.get("coverage_profile_id")
        or metric.get("rv_coverage_profile_id")
    )
    if not isinstance(value, str) or not value:
        return None
    parts = value.split("|")
    if len(parts) < 2:
        return None
    return parts[0].strip().lower(), parts[1].strip().lower() or "user"


def _merge_metrics(
    items: list[Mapping[str, Any]],
    profile: str,
    mode: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    registry = registry if isinstance(registry, Mapping) else registry_for_profile(profile, mode)
    expected = {
        "GenCov": set(registry["form_units"]),
        "ExecCov": set(registry["form_units"]),
        "OpcodeCov": set(registry["opcode_units"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        covered: set[str] = set()
        statuses: list[str] = []
        invalid = False
        for item in items:
            metric = item.get(name)
            if not isinstance(metric, Mapping):
                statuses.append("gap")
                invalid = True
                continue
            statuses.append(_status(metric.get("status")))
            profile_info = item.get("__rv_profile")
            item_expected = expected[name]
            mode_mismatch = False
            item_registry = None
            profile_pairs = item.get("__rv_profile_pairs")
            if isinstance(profile_pairs, (list, tuple)) and profile_pairs:
                try:
                    _, _, item_registry = _union_profile_registry(
                        pair for pair in profile_pairs
                        if isinstance(pair, tuple) and len(pair) == 2
                    )
                    item_expected = set(
                        item_registry["form_units"]
                        if name in {"GenCov", "ExecCov"}
                        else item_registry["opcode_units"]
                    )
                except (TypeError, ValueError):
                    item_expected = set()
            elif isinstance(profile_info, tuple) and len(profile_info) == 2:
                source_mode = str(profile_info[1]).lower()
                mode_mismatch = source_mode not in {"user", "machine"}
                # Machine-mode wrappers can also execute unprivileged forms.
                # Project their valid units into the selected aggregate
                # registry instead of discarding the whole source row.
            if not profile_pairs and isinstance(profile_info, tuple) and len(profile_info) == 2:
                try:
                    item_registry = registry_for_profile(
                        str(profile_info[0]), str(profile_info[1]),
                    )
                    item_expected = set(
                        item_registry["form_units"]
                        if name in {"GenCov", "ExecCov"}
                        else item_registry["opcode_units"]
                    )
                except (TypeError, ValueError):
                    item_expected = set()
            eligible = metric.get("eligible_units")
            units = metric.get("covered_units")
            # v1 persisted only 7-bit major-opcode units. Rebuild the new
            # decoder-key numerator from already validated ExecCov forms;
            # compressed forms may intentionally share one decoder key.
            if (name == "OpcodeCov" and item_registry is not None
                    and isinstance(eligible, list)
                    and all(isinstance(value, str) for value in eligible)
                    and set(eligible) != item_expected):
                form_metric = item.get("ExecCov")
                form_eligible = (
                    form_metric.get("eligible_units")
                    if isinstance(form_metric, Mapping) else None
                )
                form_covered = (
                    form_metric.get("covered_units")
                    if isinstance(form_metric, Mapping) else None
                )
                form_expected = set(item_registry["form_units"])
                form_status = (
                    _status(form_metric.get("status"))
                    if isinstance(form_metric, Mapping) else "gap"
                )
                form_valid = (
                    isinstance(form_eligible, list)
                    and isinstance(form_covered, list)
                    and all(isinstance(value, str)
                            for value in [*form_eligible, *form_covered])
                    and len(set(form_eligible)) == len(form_eligible)
                    and len(set(form_covered)) == len(form_covered)
                    and set(form_eligible) == form_expected
                    and set(form_covered) <= form_expected
                    and type(form_metric.get("eligible")) is int
                    and form_metric.get("eligible") == len(form_eligible)
                    and type(form_metric.get("covered")) is int
                    and form_metric.get("covered") == len(form_covered)
                    and form_metric.get("registry_sha256") == _digest(form_expected)
                    and (
                        form_status != "observed"
                        or form_metric.get("value") == round(
                            len(form_covered) / len(form_expected), 6
                        )
                    )
                    and (
                        form_status == "observed"
                        or form_metric.get("value") is None
                    )
                )
                if form_valid:
                    migrated_units = {
                        item_registry["form_to_opcode"][value]
                        for value in form_covered
                    }
                    eligible = sorted(item_expected)
                    units = sorted(migrated_units)
                    migrated_status = (
                        "gap" if _status(metric.get("status")) == "gap"
                        or form_status == "gap" else
                        "partial" if _status(metric.get("status")) == "partial"
                        or form_status in {"partial", "NA"} else
                        "NA" if _status(metric.get("status")) == "NA"
                        else "observed"
                    )
                    metric = {
                        **metric,
                        "eligible_units": eligible,
                        "covered_units": units,
                        "eligible": len(eligible),
                        "covered": len(units),
                        "registry_sha256": _digest(item_expected),
                        "value": (
                            round(len(units) / len(eligible), 6)
                            if migrated_status == "observed" else None
                        ),
                        "status": migrated_status,
                    }
                    statuses[-1] = _status(metric.get("status"))
            valid = (
                isinstance(eligible, list)
                and all(isinstance(value, str) for value in eligible)
                and len(set(eligible)) == len(eligible)
                and isinstance(units, list)
                and all(isinstance(value, str) for value in units)
                and len(set(units)) == len(units)
                and set(eligible) == item_expected
                and set(units) <= item_expected
                and type(metric.get("eligible")) is int
                and metric.get("eligible") == len(eligible)
                and type(metric.get("covered")) is int
                and metric.get("covered") == len(units)
                and metric.get("registry_sha256") == _digest(item_expected)
                and not mode_mismatch
            )
            structural_valid = valid
            if valid:
                metric_status = _status(metric.get("status"))
                value = metric.get("value")
                if metric_status == "observed":
                    if type(value) not in (int, float) or isinstance(value, bool):
                        valid = False
                    else:
                        expected_value = (
                            round(len(units) / len(item_expected), 6)
                            if item_expected else None
                        )
                        valid = value == expected_value
                elif value is not None:
                    valid = False
            if structural_valid:
                # Unit identity/counts remain usable numerator evidence even
                # when a persisted ratio is stale; the merged status is gap.
                # A native lane can be a superset of this aggregate registry;
                # project valid hits instead of turning them into a false zero.
                covered.update(set(units) & expected[name])
            if not valid:
                invalid = True
        if not statuses or all(value == "NA" for value in statuses):
            status, reason = "NA", "rv-instruction-metric-missing"
        elif invalid or "gap" in statuses:
            status, reason = "gap", "rv-instruction-observation-gap"
        elif all(value == "observed" for value in statuses):
            status, reason = "observed", None
        else:
            status, reason = "partial", "rv-instruction-observation-incomplete"
        result[name] = _metric(
            covered & expected[name], expected[name], status,
            "run-union-observed-profile-pair-registry", reason,
        )
    return result


def _missing_metric_entry(
    profile_info: tuple[str, str] | None,
) -> dict[str, Any]:
    """Represent one declared case for which RV evidence was not recorded."""
    result: dict[str, Any] = {"__rv_profile": profile_info}
    if profile_info is None:
        return result
    try:
        registry = registry_for_profile(*profile_info)
    except (TypeError, ValueError):
        return result
    result.update({
        name: _metric(
            set(),
            set(registry["form_units"] if name in {"GenCov", "ExecCov"}
                else registry["opcode_units"]),
            "gap", "case-evidence", "rv-instruction-metric-missing",
        )
        for name in METRICS
    })
    return result


def _metric_container(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rv = value.get("rv_instruction_coverage")
    if isinstance(rv, Mapping) and isinstance(rv.get("metrics"), Mapping):
        return rv["metrics"]
    guest = value.get("guest")
    if isinstance(guest, Mapping) and isinstance(guest.get("metrics"), Mapping):
        metrics = guest["metrics"]
        if any(name in metrics for name in METRICS):
            return metrics
    if isinstance(value.get("metrics"), Mapping) and any(
        name in value["metrics"] for name in METRICS
    ):
        return value["metrics"]
    return None


def _entry_profile(value: Mapping[str, Any]) -> tuple[str, str] | None:
    for container in (
        value.get("rv_instruction_coverage"),
        value.get("simulator_coverage"),
    ):
        if isinstance(container, Mapping):
            rv = container.get("rv_instruction_coverage")
            rv = rv if isinstance(rv, Mapping) else container
            parsed = _profile_from_metric(rv)
            if parsed:
                return parsed
    basis = value.get("coverage_basis")
    if isinstance(basis, Mapping):
        parsed = _profile_from_metric(basis)
        if parsed:
            return parsed
    parsed = _profile_from_metric(value)
    if parsed:
        return parsed
    lane = value.get("lane") or value.get("isa_profile")
    if isinstance(lane, str) and lane:
        return lane.split("/", 1)[0].split("|", 1)[0].lower(), str(
            value.get("privilege_mode") or "user",
        ).lower()
    return None


def _rv_metric_validation(
    metric: Mapping[str, Any], expected: set[str], name: str,
) -> tuple[set[str], list[str]]:
    """Validate one persisted union metric before using it for comparison."""
    reasons: list[str] = []
    covered_raw = metric.get("covered_units")
    eligible_raw = metric.get("eligible_units")
    covered = {
        value for value in covered_raw
        if isinstance(value, str)
    } if isinstance(covered_raw, list) else set()
    eligible = {
        value for value in eligible_raw
        if isinstance(value, str)
    } if isinstance(eligible_raw, list) else set()
    if not isinstance(covered_raw, list) or not all(
        isinstance(value, str) for value in covered_raw
    ):
        reasons.append(f"{name}-covered-units-invalid")
    elif len(covered) != len(covered_raw):
        reasons.append(f"{name}-covered-units-duplicate")
    if not isinstance(eligible_raw, list) or not all(
        isinstance(value, str) for value in eligible_raw
    ):
        reasons.append(f"{name}-eligible-units-invalid")
    elif len(eligible) != len(eligible_raw) or eligible != expected:
        reasons.append(f"{name}-denominator-invalid")
    if not covered <= expected or (isinstance(eligible_raw, list) and not covered <= eligible):
        reasons.append(f"{name}-covered-unit-invalid")
    covered_len = len(covered_raw) if isinstance(covered_raw, list) else -1
    eligible_len = len(eligible_raw) if isinstance(eligible_raw, list) else -1
    if type(metric.get("covered")) is not int or metric.get("covered") != covered_len:
        reasons.append(f"{name}-covered-count-invalid")
    if type(metric.get("eligible")) is not int or metric.get("eligible") != eligible_len:
        reasons.append(f"{name}-eligible-count-invalid")
    if metric.get("registry_sha256") != _digest(expected):
        reasons.append(f"{name}-registry-digest-invalid")
    status = metric.get("status")
    if status not in _STATUSES:
        reasons.append(f"{name}-status-invalid")
    value = metric.get("value")
    if status == "observed":
        expected_value = round(len(covered) / len(expected), 6) if expected else None
        if type(value) not in (int, float) or isinstance(value, bool):
            reasons.append(f"{name}-value-invalid")
        elif value != expected_value:
            reasons.append(f"{name}-value-mismatch")
    elif value is not None:
        reasons.append(f"{name}-value-invalid")
    return covered, reasons


def aggregate_summary(
    summary: dict[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add observed-profile-union RV instruction metrics to a batch summary."""

    entries_by_target: dict[
        tuple[str, str, str, str, str],
        list[tuple[Mapping[str, Any], tuple[str, str] | None]],
    ] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        coverage = record.get("simulator_coverage")
        container = coverage if isinstance(coverage, Mapping) else record
        metrics = _metric_container(container)
        if metrics is not None:
            entry_profile = _entry_profile(record)
            key = tuple(str(record.get(name) or "") for name in (
                "method", "route", "lane", "stratum",
            )) + (str(record.get("target") or record.get("id") or ""),)
            entries_by_target.setdefault(key, []).append((metrics, entry_profile))

    targets = summary.get("targets")
    targets = targets if isinstance(targets, list) else []
    fixed_profile = EXPERIMENT_COVERAGE_PROFILE
    fixed_mode = EXPERIMENT_PRIVILEGE_MODE
    fixed_registry = registry_for_profile(fixed_profile, fixed_mode)

    def _row_scope(
        profile_info: tuple[str, str] | None,
        selected: list[tuple[Mapping[str, Any], tuple[str, str] | None]],
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Pick one row's denominator from the profiles it actually ran.

        放开 ISA/扩展族后，逐 target 分母是该 target 实际执行到的 profile
        并集；拿不到任何 profile 时退回默认分母。
        """
        candidates = [
            info for _, info in selected if isinstance(info, tuple)
        ]
        if not candidates and isinstance(profile_info, tuple):
            candidates.append(profile_info)
        if not candidates:
            return None
        try:
            return _union_profile_registry(candidates)
        except (TypeError, ValueError):
            return None
    rv_evidence_present = bool(entries_by_target) or any(
        isinstance(row, Mapping)
        and isinstance(row.get("rv_instruction_coverage"), Mapping)
        for row in targets
    )
    if isinstance(targets, list):
        for row in targets:
            if not isinstance(row, dict):
                continue
            key = tuple(str(row.get(name) or "") for name in (
                "method", "route", "lane", "stratum", "target",
            ))
            selected = entries_by_target.get(key, [])
            profile_info = _entry_profile(row) or (
                selected[0][1] if selected else None
            )
            framework = summary.get("framework")
            framework_case_count = (
                framework.get("coverage_summary_case_count")
                if isinstance(framework, Mapping) else None
            )
            # A framework target row may count its three internal observations
            # (original/variant/reference), while ``records`` has one entry
            # per framework case.  Use the framework's case count only for
            # this aggregation layer; ordinary batch summaries keep their
            # declared target-row case count and its mismatch gate.
            declared_value = (
                framework_case_count
                if type(framework_case_count) is int
                and framework_case_count >= 0
                else row.get("cases")
            )
            declared_cases, case_count_valid = _declared_case_count(
                declared_value
            )
            cases_declared = declared_value is not None
            row_rv = row.get("rv_instruction_coverage")
            if not selected and isinstance(row_rv, Mapping) \
                    and row_rv.get("aggregation_scope") == "case-evidence":
                row_metrics = _metric_container(row)
                if row_metrics is not None:
                    selected = [(row_metrics, profile_info)]
            selected_case_count = len(selected)
            case_count_mismatch = bool(
                cases_declared
                and (
                    not case_count_valid
                    or declared_cases != selected_case_count
                )
            )
            if selected and case_count_valid and declared_cases > selected_case_count:
                selected = [*selected, *(
                    (_missing_metric_entry(profile_info), None)
                    for _ in range(declared_cases - selected_case_count)
                )]
            elif not selected and not isinstance(
                row.get("rv_instruction_coverage"), Mapping
            ) and declared_cases and rv_evidence_present:
                selected = [(_missing_metric_entry(profile_info), profile_info)]
            if selected:
                scope = _row_scope(profile_info, selected)
                row_profile, row_mode, row_registry = (
                    scope if scope else (fixed_profile, fixed_mode, fixed_registry)
                )
                if profile_info is not None:
                    try:
                        merged_metrics = _merge_metrics(
                            [
                                {
                                    **item,
                                    "__rv_profile": entry_profile or profile_info,
                                }
                                for item, entry_profile in selected
                            ],
                            row_profile, row_mode, registry=row_registry,
                        )
                    except (TypeError, ValueError):
                        merged_metrics = {}
                    if merged_metrics:
                        if not case_count_valid or case_count_mismatch:
                            for name, metric in merged_metrics.items():
                                if name in _DYNAMIC_METRICS:
                                    metric.update(
                                        status="gap", value=None,
                                        reason=(
                                            "case-count-invalid"
                                            if not case_count_valid
                                            else "case-count-mismatch"
                                        ),
                                    )
                        if row.get("status") == "gap":
                            for name, metric in merged_metrics.items():
                                if name in _DYNAMIC_METRICS:
                                    metric.update(
                                        status="gap", value=None,
                                        reason="batch-observation-gap",
                                    )
                        elif row.get("status") in {"partial", "NA"}:
                            for name, metric in merged_metrics.items():
                                if (name in _DYNAMIC_METRICS
                                        and metric.get("status") != "gap"):
                                    metric.update(status="partial", value=None,
                                                  reason="batch-observation-incomplete")
                        row_guest = row.setdefault("guest", {})
                        if isinstance(row_guest, dict):
                            row_guest.update(merged_metrics)
                            old_metrics = row_guest.get("metrics")
                            row_guest["metrics"] = (
                                {**old_metrics, **merged_metrics}
                                if isinstance(old_metrics, Mapping)
                                else dict(merged_metrics)
                            )
                            row_guest["metric_scope"] = {
                                **(
                                    row_guest.get("metric_scope", {})
                                    if isinstance(row_guest.get("metric_scope"), Mapping)
                                    else {}
                                ),
                                **{
                                    name: "target-run-fixed-profile-union"
                                    for name in METRICS
                                },
                            }
                        source_profile_ids = sorted({
                            f"{item_profile[0]}|{item_profile[1]}"
                            for _, item_profile in selected
                            if isinstance(item_profile, tuple)
                            and len(item_profile) == 2
                        } or {
                            f"{profile_info[0]}|{profile_info[1]}"
                        })
                        row["rv_instruction_coverage"] = {
                            "schema_version": SCHEMA,
                            "aggregation_scope": "target-run-fixed-profile-union",
                            "status": (
                                "gap" if any(
                                    metric.get("status") == "gap"
                                    for metric in merged_metrics.values()
                                ) else "partial" if any(
                                    metric.get("status") == "partial"
                                    for metric in merged_metrics.values()
                                ) else "observed"
                            ),
                            **({
                                "reason": (
                                    "case-count-invalid"
                                    if not case_count_valid
                                    else "case-count-mismatch"
                                )}
                               if not case_count_valid or case_count_mismatch
                               else {}),
                            "denominator": "observed-profile-pair-union-registry",
                            "isa_profile": row_profile,
                            "privilege_mode": row_mode,
                            "source_profile_ids": source_profile_ids,
                            "coverage_profile_id": row_registry[
                                "coverage_profile_id"
                            ],
                            "coverage_registry_sha256": row_registry[
                                "coverage_registry_sha256"
                            ],
                            "form_registry_sha256": row_registry[
                                "form_registry_sha256"
                            ],
                            "opcode_registry_sha256": row_registry[
                                "opcode_registry_sha256"
                            ],
                            "metrics": merged_metrics,
                        }
                else:
                    missing = _empty(
                        fixed_profile, fixed_mode, "gap", "isa-profile-missing",
                    )
                    missing.update({
                        "aggregation_scope": "target-run-fixed-profile-union",
                        "denominator": "fallback-profile-registry",
                        "source_profile_ids": [],
                    })
                    row["rv_instruction_coverage"] = missing

    unions = summary.get("experiment_unions")
    if isinstance(unions, list):
        for union in unions:
            if not isinstance(union, dict):
                continue
            profile = fixed_profile
            method_rows = [
                row for row in targets
                if isinstance(row, Mapping)
                and str(row.get("method") or "") == str(union.get("method") or "")
                and isinstance(row.get("rv_instruction_coverage"), Mapping)
            ]
            method_entries = [row["rv_instruction_coverage"] for row in method_rows]
            union_profiles = []
            for entry in method_entries:
                source_profiles = entry.get("source_profile_ids")
                if isinstance(source_profiles, list):
                    union_profiles.extend(
                        info for info in (
                            _profile_from_metric({"coverage_profile_id": item})
                            for item in source_profiles
                        ) if info is not None
                    )
                else:
                    info = _entry_profile(entry)
                    if info is not None:
                        union_profiles.append(info)
            union_profiles = sorted(set(union_profiles))
            if union_profiles:
                profile, mode, union_registry = _union_profile_registry(union_profiles)
            else:
                mode = fixed_mode
                profile, union_registry = fixed_profile, fixed_registry
            selected = [
                {
                    **entry["metrics"],
                    "__rv_profile": _entry_profile(entry),
                    "__rv_profile_pairs": [
                        info for info in (
                            _profile_from_metric({"coverage_profile_id": value})
                            for value in entry.get("source_profile_ids", ())
                        ) if info is not None
                    ],
                }
                for entry in method_entries
                if isinstance(entry.get("metrics"), Mapping)
            ]
            union_guest = union.setdefault("guest", {})
            if not isinstance(union_guest, dict):
                union_guest = {}
                union["guest"] = union_guest
            if selected:
                merged_metrics = _merge_metrics(
                    selected, profile, mode, registry=union_registry,
                )
                if union.get("status") == "gap":
                    for name, metric in merged_metrics.items():
                        if name in _DYNAMIC_METRICS:
                            metric.update(
                                status="gap", value=None,
                                reason="batch-observation-gap",
                            )
                elif union.get("status") in {"partial", "NA"}:
                    for name, metric in merged_metrics.items():
                        if (name in _DYNAMIC_METRICS
                                and metric.get("status") != "gap"):
                            metric.update(status="partial", value=None,
                                          reason="batch-observation-incomplete")
                # Batch consumers historically read these metrics directly
                # below the guest object; case consumers read guest.metrics.
                union_guest.update(merged_metrics)
                old_metrics = union_guest.get("metrics")
                union_guest["metrics"] = (
                    {**old_metrics, **merged_metrics}
                    if isinstance(old_metrics, Mapping)
                    else dict(merged_metrics)
                )
                union["rv_instruction_coverage"] = {
                    "schema_version": SCHEMA,
                    "aggregation_scope": "experiment-wide-union",
                    "status": (
                        "gap" if any(
                            metric.get("status") == "gap"
                            for metric in merged_metrics.values()
                        ) else "partial" if any(
                            metric.get("status") == "partial"
                            for metric in merged_metrics.values()
                        ) else "observed"
                    ),
                    "isa_profile": profile,
                    "privilege_mode": mode,
                    "coverage_profile_id": union_registry[
                        "coverage_profile_id"
                    ],
                    "coverage_registry_sha256": union_registry[
                        "coverage_registry_sha256"
                    ],
                    "source_profile_ids": sorted({
                        f"{item[0]}|{item[1]}" for item in union_profiles
                    }),
                    "form_registry_sha256": union_registry[
                        "form_registry_sha256"
                    ],
                    "opcode_registry_sha256": union_registry[
                        "opcode_registry_sha256"
                    ],
                    "metrics": merged_metrics,
                }
            union_guest["metric_scope"] = {
                name: "experiment-wide-union" for name in METRICS
            }

    summary["schema_version"] = BATCH_SCHEMA
    # 顶层块反映本次 run 实际执行到的 profile 并集。
    top_profile = fixed_profile
    top_mode = fixed_mode
    top_registry = fixed_registry
    top_profiles = []
    for row in targets:
        if not isinstance(row, Mapping):
            continue
        rv = row.get("rv_instruction_coverage")
        source_profile_ids = rv.get("source_profile_ids") if isinstance(rv, Mapping) else None
        if isinstance(source_profile_ids, list):
            top_profiles.extend(
                info for info in (
                    _profile_from_metric({"coverage_profile_id": value})
                    for value in source_profile_ids
                ) if info is not None
            )
        else:
            info = _entry_profile(row)
            if isinstance(info, tuple):
                top_profiles.append(info)
    top_profiles = sorted(set(top_profiles))
    if top_profiles:
        try:
            top_profile, top_mode, top_registry = _union_profile_registry(top_profiles)
        except (TypeError, ValueError):
            top_profile, top_mode, top_registry = (
                fixed_profile, fixed_mode, fixed_registry
            )
    summary["rv_instruction_coverage"] = {
        "schema_version": SCHEMA,
        "aggregation_scope": "experiment-wide-union",
        "metrics": list(METRICS),
        "aggregation": "whole-method-run-all-cases-and-targets",
        "denominator": (
            "observed-profile-pair-union-registry"
            if top_profiles else "fallback-profile-registry"
        ),
        "isa_profile": top_profile,
        "privilege_mode": top_mode,
        "coverage_profile_id": top_registry["coverage_profile_id"],
        "coverage_registry_sha256": top_registry[
            "coverage_registry_sha256"
        ],
        "source_profile_ids": sorted({
            f"{item[0]}|{item[1]}" for item in top_profiles
        }),
        "form_registry_sha256": top_registry["form_registry_sha256"],
        "opcode_registry_sha256": top_registry["opcode_registry_sha256"],
        "eligible": {
            "GenCov": len(top_registry["form_units"]),
            "ExecCov": len(top_registry["form_units"]),
            "OpcodeCov": len(top_registry["opcode_units"]),
        },
    }
    return summary


def _catalog_entry(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    coverage = record.get("simulator_coverage")
    container = coverage if isinstance(coverage, Mapping) else record
    entry = container.get("rv_opcode_catalog_coverage")
    return entry if isinstance(entry, Mapping) else None


def _merge_catalog_metric(
    entries: Iterable[Mapping[str, Any]], name: str, *, force_gap: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    covered: set[str] = set()
    outside: set[str] = set()
    statuses: list[str] = []
    reason_counts: dict[str, int] = {}
    invalid = False
    for entry in entries:
        if (entry.get("schema_version") != OPCODE_CATALOG_SCHEMA
                or entry.get("catalog_id") != OPCODE_CATALOG_ID
                or entry.get("key_schema") != OPCODE_KEY_SCHEMA
                or entry.get("registry_sha256") != OPCODE_CATALOG_REGISTRY_SHA256):
            statuses.append("gap")
            invalid = True
            continue
        metrics = entry.get("metrics")
        metric = metrics.get(name) if isinstance(metrics, Mapping) else None
        if not isinstance(metric, Mapping):
            statuses.append("gap")
            invalid = True
            reason_counts["opcode-catalog-metric-missing"] = (
                reason_counts.get("opcode-catalog-metric-missing", 0) + 1
            )
            continue
        units = metric.get("covered_units")
        outside_units = metric.get("out_of_catalog_units", ())
        valid = (
            isinstance(units, list)
            and all(isinstance(unit, str) for unit in units)
            and len(set(units)) == len(units)
            and set(units) <= OPCODE_CATALOG_KEYS
            and type(metric.get("covered")) is int
            and metric.get("covered") == len(units)
            and type(metric.get("eligible")) is int
            and metric.get("eligible") == OPCODE_CATALOG_DENOMINATOR
            and metric.get("registry_sha256") == OPCODE_CATALOG_REGISTRY_SHA256
            and isinstance(outside_units, (list, tuple))
            and all(isinstance(unit, str) for unit in outside_units)
        )
        if not valid:
            statuses.append("gap")
            invalid = True
            entry_reason = metric.get("reason")
            key = entry_reason if isinstance(entry_reason, str) and entry_reason \
                else "opcode-catalog-metric-invalid"
            reason_counts[key] = reason_counts.get(key, 0) + 1
            continue
        covered.update(units)
        outside.update(outside_units)
        metric_status = _status(metric.get("status"))
        statuses.append(metric_status)
        metric_reason = metric.get("reason")
        if metric_status in {"gap", "partial"} and isinstance(metric_reason, str) \
                and metric_reason:
            reason_counts[metric_reason] = reason_counts.get(metric_reason, 0) + 1

    if force_gap or invalid or not statuses or "gap" in statuses:
        status = "gap"
    elif all(item == "NA" for item in statuses):
        status = "NA"
    elif "partial" in statuses or "NA" in statuses:
        status = "partial"
    else:
        status = "observed"
    result: dict[str, Any] = {
        "covered": len(covered),
        "eligible": OPCODE_CATALOG_DENOMINATOR,
        "value": (
            round(len(covered) / OPCODE_CATALOG_DENOMINATOR, 6)
            if status == "observed" else None
        ),
        "status": status,
        "covered_units": sorted(covered),
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "aggregation": "unique-opcode-set-union",
        "partition_coverage": {
            partition: {
                "covered": len(covered & units),
                "eligible": len(units),
                "value": (
                    round(len(covered & units) / len(units), 6)
                    if units and status == "observed" else None
                ),
            }
            for partition, units in OPCODE_CATALOG_PARTITION_KEYS.items()
        },
    }
    if outside:
        result["out_of_catalog_units"] = sorted(outside)
    if reason_counts:
        result["reason_counts"] = dict(sorted(reason_counts.items()))
    if reason or force_gap or invalid:
        result["reason"] = reason or "opcode-catalog-evidence-incomplete"
    elif status == "partial":
        result["reason"] = (
            next(iter(reason_counts)) if len(reason_counts) == 1
            else "opcode-catalog-evidence-partial"
        )
    elif status == "gap":
        result["reason"] = (
            next(iter(reason_counts)) if len(reason_counts) == 1
            else "opcode-catalog-evidence-incomplete"
        )
    return result


def summarize_opcode_catalog_target_rows(
    summary: dict[str, Any], *,
    expected_target_scopes: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Build method-wide generation and method-target execution unions."""
    targets = summary.get("targets")
    targets = targets if isinstance(targets, list) else []
    expected_scopes = {
        (str(target), str(stratum))
        for target, stratum in expected_target_scopes
        if target and stratum
    }
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    by_method_target: dict[
        tuple[str, str, str], list[tuple[str, str, Mapping[str, Any]]],
    ] = {}
    expected_target_rows: dict[tuple[str, str, str], int] = {}
    for row in targets:
        if not isinstance(row, Mapping):
            continue
        method = str(row.get("method") or "")
        target_key = (
            method, str(row.get("stratum") or ""), str(row.get("target") or ""),
        )
        expected_target_rows[target_key] = expected_target_rows.get(target_key, 0) + 1
        entry = row.get("rv_opcode_catalog_coverage")
        if not isinstance(entry, Mapping):
            continue
        by_method.setdefault(method, []).append(entry)
        by_method_target.setdefault(target_key, []).append((
            str(row.get("route") or ""), str(row.get("lane") or ""), entry,
        ))
    methods = {
        str(row.get("method") or "")
        for row in targets if isinstance(row, Mapping)
    }
    if not methods:
        unions = summary.get("experiment_unions")
        methods = {
            str(row.get("method") or "")
            for row in unions if isinstance(row, Mapping)
        } if isinstance(unions, list) else set()
    for method in methods:
        by_method.setdefault(method, [])

    if not by_method_target and not (expected_scopes and methods):
        return summary

    target_runs = []
    for (method, stratum, target), scopes in sorted(by_method_target.items()):
        entries = [entry for _route, _lane, entry in scopes]
        count_values = [_catalog_entry_case_count(entry) for entry in entries]
        case_count = (
            sum(count for count, valid in count_values if valid)
            if all(valid for _count, valid in count_values) else None
        )
        metrics = {
            name: _merge_catalog_metric(
                entries, name,
                force_gap=len(entries) != expected_target_rows.get(
                    (method, stratum, target), 0,
                ),
                reason=("target-opcode-evidence-missing"
                        if len(entries) != expected_target_rows.get(
                            (method, stratum, target), 0,
                        ) else None),
            )
            for name in ("GenOpcodeCov", "ExecOpcodeCov")
        }
        target_runs.append({
            "method": method or None,
            "stratum": stratum or None,
            "target": target or None,
            "case_count": case_count,
            "aggregation_scope": "method-target-run",
            "input_scopes": [
                {"route": route or None, "lane": lane or None}
                for route, lane in sorted({(route, lane) for route, lane, _ in scopes})
            ],
            "metrics": metrics,
        })

    for method in sorted(methods):
        for target, stratum in sorted(expected_scopes):
            target_key = (method, stratum, target)
            if target_key in by_method_target:
                continue
            target_runs.append({
                "method": method or None,
                "stratum": stratum,
                "target": target,
                "case_count": None,
                "aggregation_scope": "method-target-run",
                "input_scopes": [],
                "metrics": {
                    name: _merge_catalog_metric(
                        (), name, force_gap=True,
                        reason="target-opcode-evidence-missing",
                    )
                    for name in ("GenOpcodeCov", "ExecOpcodeCov")
                },
            })

    method_generated_unions = []
    for method, entries in sorted(by_method.items()):
        target_row_count = sum(
            1 for row in targets
            if isinstance(row, Mapping)
            and str(row.get("method") or "") == method
        )
        missing_expected_scope = any(
            (method, stratum, target) not in by_method_target
            for target, stratum in expected_scopes
        )
        generated = _merge_catalog_metric(
            entries, "GenOpcodeCov",
            force_gap=len(entries) != target_row_count or missing_expected_scope,
            reason=("method-target-opcode-evidence-missing"
                    if len(entries) != target_row_count or missing_expected_scope
                    else None),
        )
        method_generated_unions.append({
            "method": method or None,
            "aggregation_scope": "method-generated-elf-union",
            "metrics": {"GenOpcodeCov": generated},
        })

    summary["rv_opcode_catalog_coverage"] = {
        "schema_version": OPCODE_CATALOG_SCHEMA,
        "aggregation_scope": "catalog-wide-comparison",
        "catalog_id": OPCODE_CATALOG_ID,
        "key_schema": OPCODE_KEY_SCHEMA,
        "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
        "catalog_form_count": OPCODE_CATALOG_FORM_COUNT,
        "opcode_denominator": OPCODE_CATALOG_DENOMINATOR,
        "partitions": {
            partition: {"opcode_denominator": len(units)}
            for partition, units in OPCODE_CATALOG_PARTITION_KEYS.items()
        },
        "aggregation": "unique-opcode-set-union; no-case-averaging",
        "target_runs": target_runs,
        "method_generated_unions": method_generated_unions,
    }
    return summary


def aggregate_opcode_catalog_summary(
    summary: dict[str, Any], records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Union per-case full-catalog opcode evidence by method and target."""
    entries_by_target: dict[
        tuple[str, str, str, str, str], list[Mapping[str, Any]],
    ] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        entry = _catalog_entry(record)
        if entry is None:
            continue
        key = tuple(str(record.get(name) or "") for name in (
            "method", "route", "lane", "stratum",
        )) + (str(record.get("target") or record.get("id") or ""),)
        entries_by_target.setdefault(key, []).append(entry)

    targets = summary.get("targets")
    targets = targets if isinstance(targets, list) else []
    evidence_present = bool(entries_by_target) or any(
        isinstance(row, Mapping)
        and isinstance(row.get("rv_opcode_catalog_coverage"), Mapping)
        for row in targets
    )
    if not evidence_present:
        return summary

    for row in targets:
        if not isinstance(row, dict):
            continue
        key = tuple(str(row.get(name) or "") for name in (
            "method", "route", "lane", "stratum", "target",
        ))
        selected = entries_by_target.get(key, [])
        if not selected and isinstance(
            row.get("rv_opcode_catalog_coverage"), Mapping,
        ):
            continue
        declared, valid_count = _declared_case_count(row.get("cases"))
        represented_counts = [_catalog_entry_case_count(entry) for entry in selected]
        represented_valid = all(valid for _count, valid in represented_counts)
        represented_count = sum(
            count for count, valid in represented_counts if valid
        )
        case_count_invalid = (
            row.get("cases") is None or not valid_count or not represented_valid
        )
        mismatch = bool(
            case_count_invalid or declared != represented_count
        )
        metrics = {
            name: _merge_catalog_metric(
                selected, name,
                force_gap=mismatch or not selected,
                reason=("case-count-invalid" if case_count_invalid else
                        "case-count-mismatch" if mismatch else
                        "opcode-catalog-case-evidence-missing" if not selected
                        else None),
            )
            for name in ("GenOpcodeCov", "ExecOpcodeCov")
        }
        row["rv_opcode_catalog_coverage"] = {
            "schema_version": OPCODE_CATALOG_SCHEMA,
            "aggregation_scope": "method-target-run",
            "catalog_id": OPCODE_CATALOG_ID,
            "key_schema": OPCODE_KEY_SCHEMA,
            "registry_sha256": OPCODE_CATALOG_REGISTRY_SHA256,
            "case_count": represented_count if represented_valid else None,
            "metrics": metrics,
            "status": (
                "gap" if any(item["status"] == "gap" for item in metrics.values())
                else "partial" if any(item["status"] == "partial" for item in metrics.values())
                else "NA" if all(item["status"] == "NA" for item in metrics.values())
                else "observed"
            ),
        }
    return summarize_opcode_catalog_target_rows(summary)


def compare_experiment_unions(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare the three RV metric unions using their recorded denominators."""
    left = first.get("rv_instruction_coverage") if isinstance(first, Mapping) else None
    right = second.get("rv_instruction_coverage") if isinstance(second, Mapping) else None
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {"status": "not-recorded", "reasons": ["rv-instruction-coverage-missing"]}
    results, reasons = {}, []
    left_profile = _profile_from_metric(left)
    right_profile = _profile_from_metric(right)
    registry = None
    if left_profile is None or right_profile is None:
        reasons.append("coverage-profile-invalid")
    elif left_profile != right_profile:
        reasons.append("coverage-profile-mismatch")
    else:
        try:
            source_profile_ids = left.get("source_profile_ids")
            profile_pairs = [
                info for info in (
                    _profile_from_metric({"coverage_profile_id": value})
                    for value in source_profile_ids
                ) if info is not None
            ] if isinstance(source_profile_ids, list) else []
            if not profile_pairs and left_profile[1] in {"user", "machine"}:
                profile_pairs = [left_profile]
            if not profile_pairs:
                raise ValueError("coverage-profile-pairs-missing")
            expected_profile, expected_mode, registry = _union_profile_registry(profile_pairs)
            if (expected_profile, expected_mode) != left_profile:
                reasons.append("coverage-profile-pairs-mismatch")
        except (TypeError, ValueError):
            reasons.append("coverage-profile-invalid")
    if left.get("coverage_profile_id") != right.get("coverage_profile_id"):
        reasons.append("coverage-profile-mismatch")
    if registry is not None:
        expected_profile_id = registry["coverage_profile_id"]
        expected_registry_digest = registry["coverage_registry_sha256"]
        for side, value in (("baseline", left), ("candidate", right)):
            if value.get("coverage_profile_id") != expected_profile_id:
                reasons.append(f"{side}-coverage-profile-invalid")
            if value.get("coverage_registry_sha256") != expected_registry_digest:
                reasons.append(f"{side}-coverage-registry-invalid")
    elif left.get("coverage_registry_sha256") != right.get("coverage_registry_sha256"):
        reasons.append("coverage-registry-mismatch")
    elif left.get("coverage_registry_sha256") in (None, ""):
        reasons.append("coverage-registry-invalid")
    for name in METRICS:
        lm = left.get("metrics", {}).get(name) if isinstance(left.get("metrics"), Mapping) else None
        rm = right.get("metrics", {}).get(name) if isinstance(right.get("metrics"), Mapping) else None
        if not isinstance(lm, Mapping) or not isinstance(rm, Mapping):
            reason = f"{name}-missing"
            results[name] = {"status": "not-comparable", "reasons": [reason]}
            reasons.append(reason)
            continue
        expected = (
            set(registry["form_units"])
            if registry is not None and name in {"GenCov", "ExecCov"}
            else set(registry["opcode_units"])
            if registry is not None
            else set()
        )
        lc, local_left = _rv_metric_validation(lm, expected, f"baseline-{name}")
        rc, local_right = _rv_metric_validation(rm, expected, f"candidate-{name}")
        local = []
        local.extend(local_left)
        local.extend(local_right)
        le = set(lm.get("eligible_units")) if isinstance(lm.get("eligible_units"), list) and all(
            isinstance(value, str) for value in lm.get("eligible_units", ())
        ) else set()
        re = set(rm.get("eligible_units")) if isinstance(rm.get("eligible_units"), list) and all(
            isinstance(value, str) for value in rm.get("eligible_units", ())
        ) else set()
        if lm.get("status") != "observed":
            local.append(f"baseline-{name}-incomplete")
        if rm.get("status") != "observed":
            local.append(f"candidate-{name}-incomplete")
        if le != re:
            local.append(f"{name}-denominator-mismatch")
        if lm.get("registry_sha256") != _digest(le):
            local.append(f"baseline-{name}-registry-digest-mismatch")
        if rm.get("registry_sha256") != _digest(re):
            local.append(f"candidate-{name}-registry-digest-mismatch")
        if not lc <= le or not rc <= re:
            local.append(f"{name}-covered-unit-invalid")
        common, union = lc & rc, lc | rc
        results[name] = {
            "status": "comparable" if not local else "not-comparable",
            "reasons": sorted(set(local)),
            "baseline_covered": len(lc), "candidate_covered": len(rc),
            "eligible": len(le) if le == re else None,
            "baseline_percent": round(100 * len(lc) / len(le), 4) if not local and le else None,
            "candidate_percent": round(100 * len(rc) / len(re), 4) if not local and re else None,
            "delta_percentage_points": round(
                100 * (len(rc) / len(re) - len(lc) / len(le)), 4,
            ) if not local and le else None,
            "covered_set_jaccard": round(len(common) / len(union), 6) if union else None,
            "denominator_sha256": _digest(le) if le == re else None,
        }
        reasons.extend(local)
    return {
        "status": "comparable" if not reasons else "not-comparable",
        "reasons": sorted(set(reasons)),
        "scope": "whole-method-run-all-cases-and-targets",
        "metrics": results,
    }


aggregate_rv_instruction_coverage = aggregate_summary
attach_rv_instruction_metrics = attach_case_metrics

__all__ = [
    "BATCH_SCHEMA",
    "EXPERIMENT_COVERAGE_PROFILE",
    "EXPERIMENT_PRIVILEGE_MODE",
    "METRICS",
    "OPCODE_CATALOG_DENOMINATOR",
    "OPCODE_CATALOG_FORM_COUNT",
    "OPCODE_CATALOG_ID",
    "OPCODE_CATALOG_PARTITION_KEYS",
    "OPCODE_CATALOG_PARTITION_VENDOR_OVERLAY",
    "OPCODE_CATALOG_REGISTRY_SHA256",
    "OPCODE_CATALOG_SCHEMA",
    "OPCODE_KEY_SCHEMA",
    "RV_INSTRUCTION_METRICS",
    "SCHEMA",
    "aggregate_opcode_catalog_summary",
    "aggregate_summary",
    "aggregate_rv_instruction_coverage",
    "attach_case_metrics",
    "attach_rv_instruction_metrics",
    "case_metrics",
    "compare_experiment_unions",
    "opcode_catalog_case_coverage",
    "registry_for_profile",
    "summarize_opcode_catalog_target_rows",
]

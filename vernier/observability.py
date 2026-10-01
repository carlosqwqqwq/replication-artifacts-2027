from collections.abc import Mapping, Sequence
from typing import Any


# Experiment bookkeeping never describes guest state.
_OBSERVATION_SIDE_CHANNEL_KEYS = frozenset({
    "coverage", "coverage_summary", "coverage_status", "coverage_gap",
    "ledger", "ledger_path", "ledger_status", "ledger_events",
    "event_count", "formal_eligible", "observer_gaps",
    "adapter_execution_error", "execution_status", "trace_status",
    "trace_records", "trace_available", "trace_gap", "trace_reason",
    "trace_truncated", "mailbox_status", "fpr_available",
    "fp_observer", "state_observer", "volatile_gpr_indices",
    "configuration_identity", "native_reference_identity",
})


def _is_observation_side_channel_field(field: object) -> bool:
    if not isinstance(field, str):
        return False
    field = field.removeprefix("extra.")
    return (
        field in _OBSERVATION_SIDE_CHANNEL_KEYS
        or field.startswith(("coverage_", "coverage.", "ledger_", "ledger."))
    )


def _strip_observation_side_channels(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_observation_side_channels(item)
            for key, item in value.items()
            if not _is_observation_side_channel_field(key)
        }
    if isinstance(value, list):
        return [_strip_observation_side_channels(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_observation_side_channels(item) for item in value)
    return value


def compare_fields(
    left: Mapping[str, object], right: Mapping[str, object], fields: Sequence[str],
) -> tuple[dict[str, tuple[object, object]], tuple[str, ...]]:
    """Compare requested values and report absent values separately."""
    differences: dict[str, tuple[object, object]] = {}
    missing = []
    for field in fields:
        if field not in left or field not in right:
            missing.append(field)
        elif left[field] != right[field]:
            differences[field] = (left[field], right[field])
    return differences, tuple(missing)

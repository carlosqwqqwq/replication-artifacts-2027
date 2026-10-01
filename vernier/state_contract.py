"""Shared static state-observer contract validation.

This module is deliberately independent of RVGEN and reference execution.
It validates only the metadata needed to describe a stateful witness; Sail or
another reference remains authoritative for executed state and trace results.
"""

from typing import Any


STATE_DOMAINS = frozenset({
    "scalar", "memory", "reservation", "control-flow", "privileged", "csr",
    "vector", "vector-memory", "trap", "instruction-memory", "lifecycle",
})
_MEMORY_OBSERVER_KEYS = ("memory.alignment", "memory.pma", "memory.pmp")
_TRAP_OBSERVER_KEYS = ("trap.cause", "trap.epc", "trap.tval")
_VECTOR_OBSERVER_KEYS = (
    "vector.vl",
    "vector.vtype",
    "vector.vstart",
    "vector.vlenb",
    "vector.mask",
    "vector.tail",
    "vector.registers",
    "vector.scalar-result",
)
STATE_OBSERVER_KEYS_BY_DOMAIN = {
    # Tuples keep one canonical spelling and stable serialization order.  A
    # consumer that needs membership can still convert the tuple to a set.
    "scalar": (),
    "memory": _MEMORY_OBSERVER_KEYS,
    "reservation": (
        *_MEMORY_OBSERVER_KEYS,
        "reservation.valid",
        "reservation.address",
        "reservation.ordering",
        "reservation.result",
    ),
    "control-flow": ("control.target",),
    "privileged": ("privilege.mode", *_TRAP_OBSERVER_KEYS),
    "csr": (
        "csr.address",
        "csr.privilege",
        "csr.warl.readback",
        "csr.reserved-fields",
    ),
    "vector": _VECTOR_OBSERVER_KEYS,
    "vector-memory": _VECTOR_OBSERVER_KEYS + _MEMORY_OBSERVER_KEYS,
    "trap": _TRAP_OBSERVER_KEYS,
    "instruction-memory": (),
    "lifecycle": (
        "lifecycle.executed-pc",
        "lifecycle.path-identity",
        "lifecycle.code-store",
        "lifecycle.fence-i",
        "lifecycle.second-fetch",
        "lifecycle.instruction-fetch",
    ),
}


_CONTRACT = "rvgen-sail-state-contract-audit-v1"

def state_contract_audit(testcase: Any) -> dict[str, object]:
    """Validate one testcase's static state observer declaration."""

    dataflow_meta = getattr(testcase, "dataflow_meta", {})
    if not isinstance(dataflow_meta, dict):
        dataflow_meta = {}
    status = str(dataflow_meta.get("state_observer_status") or "")

    def result(status_name: str, state_domain: str | None, missing=()) -> dict[str, object]:
        return {
            "contract": _CONTRACT,
            "status": status_name,
            "state_domain": state_domain,
            "missing_observer_keys": list(missing),
            "observer_status": status,
        }

    raw_domain = dataflow_meta.get("state_domain")
    if not isinstance(raw_domain, str) or not raw_domain:
        return result("missing-state-domain", None)
    domain = raw_domain
    lifecycle_domain = str(dataflow_meta.get("lifecycle_state_domain") or "")
    raw_declared_keys = dataflow_meta.get("state_observer_keys", ())
    lifecycle_signals = bool(dataflow_meta.get("lifecycle_events")) or status.startswith(
        "lifecycle-"
    ) or (
        isinstance(raw_declared_keys, (list, tuple))
        and any(
            isinstance(key, str) and key.startswith("lifecycle.")
            for key in raw_declared_keys
        )
    )
    if domain == "scalar" and lifecycle_signals and lifecycle_domain != "lifecycle":
        return result("unsupported-lifecycle-state-domain", domain)
    if lifecycle_domain or domain == "lifecycle":
        if lifecycle_domain not in {"", "lifecycle"} or domain not in {"scalar", "lifecycle"}:
            return result("unsupported-lifecycle-state-domain", domain)
        domain = "lifecycle"
    if domain not in STATE_DOMAINS:
        return result("unsupported-state-domain", domain)
    if domain == "lifecycle":
        raw_events = dataflow_meta.get("lifecycle_events", ())
        if not isinstance(raw_events, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_events
        ):
            return result("unsupported-lifecycle-events", domain)
        base = STATE_OBSERVER_KEYS_BY_DOMAIN["lifecycle"][:2]
        required_values = {
            ("instruction-fetch",): base + STATE_OBSERVER_KEYS_BY_DOMAIN["lifecycle"][5:6],
            ("code-store", "fence.i", "second-fetch"):
                base + STATE_OBSERVER_KEYS_BY_DOMAIN["lifecycle"][2:5],
        }.get(tuple(raw_events))
        if required_values is None:
            return result("unsupported-lifecycle-events", domain)
    else:
        required_values = STATE_OBSERVER_KEYS_BY_DOMAIN.get(domain, ())
        if domain == "vector":
            required_values = required_values[:4]
        elif domain == "vector-memory":
            required_values = (
                required_values[:4]
                + ("vector.mask", "vector.tail")
                + STATE_OBSERVER_KEYS_BY_DOMAIN["memory"]
            )
    allowed = set(STATE_OBSERVER_KEYS_BY_DOMAIN.get(domain, ()))
    raw_keys = dataflow_meta.get("state_observer_keys", ())
    keys_valid = isinstance(raw_keys, (list, tuple)) and all(
        isinstance(item, str) and item for item in raw_keys
    )
    keys = set(raw_keys) if keys_valid else set()
    if not keys_valid:
        return result("invalid-observer-keys", domain, sorted(set(required_values)))
    required = set(required_values)
    functional_tags = getattr(testcase, "functional_tags", ())
    if not isinstance(functional_tags, (list, tuple)):
        functional_tags = ()
    effect_kind = next(
        (
            str(tag).removeprefix("effect:")
            for tag in functional_tags
            if str(tag).startswith("effect:")
        ),
        "",
    )
    instruction_meta = getattr(testcase, "instruction_meta", ())
    if not isinstance(instruction_meta, (list, tuple)):
        instruction_meta = ()
    if domain in {"vector", "vector-memory"} and effect_kind in {"vector-data", "vector-memory"}:
        observability = getattr(testcase, "observability_contract", None)
        vector_store_sink = getattr(observability, "sink_transform", None) == "vector-store"
        raw_dynamic = dataflow_meta.get("vector_dynamic_state_keys", ())
        if not isinstance(raw_dynamic, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in raw_dynamic
        ):
            return result("invalid-observer-keys", domain, sorted(required))
        if any(item not in {"fflags", "frm", "vxsat", "vxrm"} for item in raw_dynamic):
            return result("invalid-observer-keys", domain, sorted(required))
        allowed.update(raw_dynamic)
        if any(
            not hasattr(item, "tags") or not isinstance(item.tags, (list, tuple))
            for item in instruction_meta
        ):
            return result("invalid-instruction-meta", domain, sorted(required))
        required.update(("vector.mask", "vector.tail"))
        required.update(raw_dynamic)
        if any(
            "observable" in item.tags
            and "vector-state" in item.tags
            and (
                (mnemonic := str(getattr(item, "mnemonic", "")).replace("_", ".")).startswith("vse")
                or (
                    len(mnemonic) >= 5
                    and mnemonic.startswith("vs")
                    and mnemonic[2] in "1248"
                    and mnemonic.endswith("r.v")
                )
                or mnemonic == "vsm.v"
            )
            for item in instruction_meta
        ) and not vector_store_sink:
            required.add("vector.registers")
    if domain != "scalar" and any(item not in allowed for item in keys):
        return result("invalid-observer-keys", domain, sorted(required))
    if domain == "scalar":
        return {"contract": _CONTRACT, "status": "not-required", "state_domain": domain}
    allowed_statuses = {
        **dict.fromkeys(("vector", "vector-memory"), frozenset({
            "vector-register-trace-pending",
            "vector-csr-trace-pending",
            "vector-scalar-observer-ready",
        })),
        "privileged": {"privileged-harness-pending"},
        "csr": {"csr-warl-observer-pending"},
        "memory": {"pma-pmp-observer-pending"},
        "reservation": {"reservation-observer-pending"},
        "control-flow": {"control-target-observer-ready"},
        "trap": {"trap-observer-pending"},
        "instruction-memory": {"instruction-memory-observer-ready"},
        "lifecycle": {"lifecycle-observer-pending", "lifecycle-observer-ready"},
    }.get(domain, set())
    if not str(dataflow_meta.get("sail_branch_contract") or ""):
        status_name = "missing-branch-contract"
    elif not status:
        status_name = "missing-observer-status"
    elif allowed_statuses and status not in allowed_statuses:
        status_name = "unsupported-observer-status"
    else:
        missing = sorted(required - keys)
        status_name = "missing-observer-keys" if missing else "ready"
    return result(status_name, domain, sorted(required - keys))

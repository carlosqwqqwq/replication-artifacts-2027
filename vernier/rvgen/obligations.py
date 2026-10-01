"""生成义务的最小身份合同。

生成只需要 catalog/effect/realization 和 Sail 配置身份；手册 YAML 匹配属于
离线证据，不进入原型准入链。
"""

import hashlib
import json
from dataclasses import dataclass

from ..reference_contract import REFERENCE_LOCK_CONFIG, sail_profile_support


@dataclass(frozen=True)
class NormativeObligation:
    obligation_id: str
    sail_support_status: str
    sail_support_reason: str
    sail_candidate_kind: str
    sail_config_files: tuple[str, ...]
    sail_reference_commit: str
    sail_reference_release_tag: str

    def to_dict(self) -> dict[str, object]:
        result = vars(self).copy()
        result["sail_config_files"] = list(result["sail_config_files"])
        return result


def normative_obligation_for(
    form, schema, realization, *, lane: str, boundary_class: str,
) -> NormativeObligation:
    sail = REFERENCE_LOCK_CONFIG.sail_profile_contract(str(realization.isa_profile))
    status = (
        "supported" if sail.supported else "unsupported"
        if sail.reason.startswith((
            "unsupported-isa-profile:",
            "unsupported-sail-floating-register-model:",
            "unsupported-sail-h-extension:",
            "unsupported-sail-q-precision-sample-config:",
        )) else "config-blocked"
    )
    reason = sail.reason
    if type(REFERENCE_LOCK_CONFIG).__name__ == "ReferenceLockConfig":
        exact_supported, exact_reason, _identity = sail_profile_support(
            str(realization.isa_profile)
        )
        if not exact_supported:
            status, reason = "config-blocked", exact_reason
    if schema is not None and getattr(schema, "kind", "") == "encoding-only":
        status, reason = (
            "unsupported", "encoding-only-value-observer-pending",
        )
    config_files = tuple(str(path) for path in sail.config_files)
    identity = {
        "form": str(getattr(form, "mnemonic", "")),
        "effect_kind": None if schema is None else str(getattr(schema, "kind", "")),
        "operation": None if schema is None else str(getattr(schema, "operation", "")),
        "lane": str(lane), "boundary_class": str(boundary_class),
        "realization_id": str(getattr(realization, "realization_id", "")),
        "sail_support_status": status, "sail_support_reason": reason,
        "sail_candidate_kind": str(sail.candidate_kind),
        "sail_config_files": list(config_files),
        "sail_reference_commit": REFERENCE_LOCK_CONFIG.sail.commit,
        "sail_reference_release_tag": REFERENCE_LOCK_CONFIG.sail.release_tag,
    }
    obligation_id = "obl-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return NormativeObligation(
        obligation_id, status, reason, str(sail.candidate_kind), config_files,
        REFERENCE_LOCK_CONFIG.sail.commit,
        REFERENCE_LOCK_CONFIG.sail.release_tag,
    )

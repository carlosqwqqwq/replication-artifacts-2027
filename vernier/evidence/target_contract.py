"""Target coverage 的纯身份、状态和路线谓词。"""

from collections.abc import Mapping

from framework._util import canonical_digest
from framework.adapters.riscv_dv import _normalize_generation_profile, _portable_generation_profile
from framework.adapters.runner import terminal_ebreak_observed


_RECIPE_METADATA_FIELDS = frozenset({"route_attempt"})


def _normal_outcome(value: object, extra_state: object = None) -> object:
    return "normal" if value == "completed" or (
        value == "trap" and terminal_ebreak_observed(extra_state)
    ) else value


def _positive_count(value: object) -> bool:
    return type(value) is int and value > 0


def _generation_profile_digest(profile: Mapping[str, object]) -> str:
    profile = dict(profile)
    profile.setdefault("iterations", 1)
    return canonical_digest(_portable_generation_profile(_normalize_generation_profile(profile)))


def _recipe_digest(recipe: Mapping[str, object]) -> str:
    return canonical_digest({
        key: value for key, value in recipe.items()
        if key not in _RECIPE_METADATA_FIELDS
    })


def _recipe_target_identity(recipe: Mapping[str, object]) -> tuple[object, object]:
    target = recipe.get("target")
    if not isinstance(target, Mapping):
        return None, None
    expected = target.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    return (
        target.get("identity_digest") or expected.get("identity_digest"),
        target.get("binary_sha256") or expected.get("binary_sha256"),
    )


def _target_claimed(record: Mapping[str, object]) -> bool:
    counts = record.get("counts")
    target = record.get("target")
    target_baseline = record.get("target_baseline")
    return (
        isinstance(counts, Mapping)
        and any(_positive_count(counts.get(name, 0)) for name in (
            "target_attempted", "target_tested", "target_gap",
            "target_raw_recorded", "target_comparison_pending",
            "target_unsupported", "case_skipped", "target_mismatch_candidate",
            "proposal_gap", "clean",
        ))
    ) or bool(record.get("targets") or record.get("target_records")
             or isinstance(target, (Mapping, list, tuple)) and target
             or isinstance(target_baseline, Mapping)
             and target_baseline.get("status") is not None)


def _target_status_coherence(target: Mapping[str, object]) -> str | None:
    candidate = target.get("candidate")
    if candidate is None:
        return None
    if type(candidate) is not bool:
        return "target-candidate-invalid"
    return None if candidate is (
        target.get("status") in {
            "target-state-mismatch-candidate", "target-mismatch-candidate",
        }
    ) else "target-candidate-status-mismatch"

"""Program route 的 target probe 准入谓词。"""

from collections.abc import Mapping, Sequence

from framework.evidence.root_route_matrix import _execution_recipe_gaps


def target_probe_allowed(
    enabled: bool, execution_gaps: Sequence[str], recipe: Mapping[str, object] | None = None,
) -> bool:
    if not enabled:
        return False
    if recipe is None:
        return bool(execution_gaps)
    target = recipe.get("target")
    expected = target.get("expected") if isinstance(target, Mapping) else None
    if not isinstance(target, Mapping) or not isinstance(expected, Mapping):
        return False
    return not _execution_recipe_gaps({
        **recipe,
        "target": {
            **target,
            "expected": {
                key: value for key, value in expected.items()
                if key not in {"status", "reason"}
            },
        },
    })

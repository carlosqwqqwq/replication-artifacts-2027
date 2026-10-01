"""Shared policy values for the K1-to-QEMU reference fallback."""

K1_QEMU_FALLBACK_FAILURE_CLASSES = frozenset({
    "k1-timeout",
    "k1-transport-unavailable",
})
K1_QEMU_FALLBACK_POLICY = "native-rv64-to-qemu-on-timeout-or-transport-unavailable"
K1_BOARD_TQEMU_FALLBACK_POLICY = (
    "native-rv64-to-existing-t-qemu-on-timeout-or-transport-unavailable"
)
LEGACY_K1_QEMU_FALLBACK_POLICY = "native-rv64-to-qemu-on-transport-unavailable"


def configured_k1_qemu_fallback_classes(value: object) -> frozenset[str] | None:
    """Read current policy lists and the historical transport-only value."""
    if value == "k1-transport-unavailable":
        return frozenset({"k1-transport-unavailable"})
    if not isinstance(value, (list, tuple)):
        return None
    if not value or any(
        not isinstance(item, str) or item not in K1_QEMU_FALLBACK_FAILURE_CLASSES
        for item in value
    ):
        return None
    if len(value) != len(set(value)):
        return None
    return frozenset(value)


def qemu_fallback_policy_matches(
    policy: object, reason: object, allowed_classes: frozenset[str] | None,
) -> bool:
    if (
        allowed_classes is None
        or not isinstance(reason, str)
        or reason not in allowed_classes
    ):
        return False
    if policy == K1_QEMU_FALLBACK_POLICY:
        return True
    return (
        policy == LEGACY_K1_QEMU_FALLBACK_POLICY
        and reason == "k1-transport-unavailable"
    )

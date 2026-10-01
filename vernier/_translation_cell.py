"""Canonical translation-cell value object shared by direct and spec layers."""

from typing import Any

from .paths import reference_config_identity
from ._util import canonical_digest


def _tuple_str(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("expected a list of strings")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("expected a list of non-empty strings")
    return tuple(values)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("optional integer field must be an integer or null")
    return value


def canonical_realization_id(
    *,
    xlen: int,
    extension_tokens: tuple[str, ...] | list[str],
    register_carrier: str,
    privilege_class: str,
    isa_profile: str,
    environment_profile: str = "base",
) -> str:
    """Return the stable identity derived from one execution realization."""
    payload = {
        "xlen": int(xlen),
        "extension_tokens": [str(token) for token in extension_tokens],
        "register_carrier": str(register_carrier),
        "privilege_class": str(privilege_class),
        "isa_profile": str(isa_profile),
    }
    # Keep the historical scalar identity stable; only realizations that
    # actually select a non-default execution environment pay for the extra
    # identity dimension.  This avoids a corpus-wide churn while making V/
    # privileged/PMP slices impossible to alias with the user-mode baseline.
    if str(environment_profile) != "base":
        payload["environment_profile"] = str(environment_profile)
    return "real-" + canonical_digest(payload)[:24]


def generation_realization_for_testcase(testcase: Any) -> dict[str, Any] | None:
    dataflow_meta = getattr(testcase, "dataflow_meta", None)
    realization = (
        dataflow_meta.get("generation_realization")
        if isinstance(dataflow_meta, dict)
        else None
    )
    if not isinstance(realization, dict):
        return None
    xlen = realization.get("xlen")
    isa_profile = realization.get("isa_profile")
    extension_tokens = realization.get("extension_tokens")
    register_carrier = realization.get("register_carrier")
    privilege_class = realization.get("privilege_class")
    environment_profile = realization.get("environment_profile", "base")
    realization_id = realization.get("realization_id")
    config_identity = realization.get("reference_config_identity")
    actual_profile = getattr(testcase, "isa_profile", None)
    if (
        not isinstance(xlen, int)
        or xlen not in {32, 64}
        or not isinstance(isa_profile, str)
        or not isa_profile
        or not isinstance(actual_profile, str)
        or not isinstance(register_carrier, str)
        or register_carrier not in {"gpr", "fpr"}
        or not isinstance(privilege_class, str)
        or not privilege_class
        or not isinstance(environment_profile, str)
        or not environment_profile
        or not isinstance(extension_tokens, list)
        or not all(isinstance(token, str) and token for token in extension_tokens)
        or not isinstance(realization_id, str)
        or len(realization_id) != 29
        or not realization_id.startswith("real-")
        or any(character not in "0123456789abcdef" for character in realization_id[5:])
        or not isinstance(config_identity, str)
        or len(config_identity) != 64
        or any(character not in "0123456789abcdef" for character in config_identity)
    ):
        return None
    if tuple(extension_tokens) != tuple(sorted(set(extension_tokens))):
        return None
    # The profile is the source of truth for enabled ISA extensions.  Do not
    # let a caller pair a valid realization id with a different token set.
    from .spec_definedness import enabled_extensions, is_canonical_isa_profile

    if frozenset(extension_tokens) != enabled_extensions(isa_profile):
        return None
    if not is_canonical_isa_profile(isa_profile, xlen=xlen):
        return None
    if actual_profile != isa_profile:
        boundary = dataflow_meta.get("legality_boundary_class")
        owner_extensions = enabled_extensions(isa_profile)
        actual_extensions = enabled_extensions(actual_profile)
        if isinstance(boundary, str) and boundary.startswith("extension-gate:"):
            disabled = frozenset(
                token for token in boundary.removeprefix("extension-gate:")
                .replace("+", "_").split("_") if token
            )
            profile_ok = (
                bool(disabled & owner_extensions)
                and actual_extensions == owner_extensions - (disabled & owner_extensions)
                and actual_extensions != owner_extensions
            )
        elif isinstance(boundary, str) and boundary.startswith("rv32:compat-extension-gate:"):
            suffix = frozenset(
                token for token in boundary.removeprefix("rv32:compat-extension-gate:")
                .split("_") if token
            )
            compat_profile = "rv32i_" + "_".join(sorted(suffix))
            profile_ok = bool(suffix) and actual_extensions == enabled_extensions(compat_profile)
        elif isinstance(boundary, str) and boundary.startswith("control-witness:"):
            from .rvgen.boundaries import control_witness_disabled_extensions

            disabled = control_witness_disabled_extensions(
                boundary.removeprefix("control-witness:")
            )
            profile_ok = (
                bool(disabled & owner_extensions)
                and actual_extensions == owner_extensions - (disabled & owner_extensions)
                and actual_extensions != owner_extensions
            )
        else:
            profile_ok = False
        if (
            not profile_ok
            or not is_canonical_isa_profile(actual_profile, xlen=xlen)
        ):
            return None
    if realization_id != canonical_realization_id(
        xlen=xlen,
        extension_tokens=extension_tokens,
        register_carrier=str(register_carrier),
        privilege_class=privilege_class,
        isa_profile=isa_profile,
        environment_profile=environment_profile,
    ):
        return None
    normative = dataflow_meta.get("normative_obligation")
    config_files = (
        tuple(sorted(str(path) for path in normative.get("sail_config_files", ()) if isinstance(path, str) and path))
        if isinstance(normative, dict)
        else ()
    )
    if config_identity != reference_config_identity(config_files):
        return None
    payload = {
        "xlen": int(xlen),
        "isa_profile": str(isa_profile),
        "register_carrier": str(register_carrier),
        "privilege_class": str(privilege_class),
        "environment_profile": str(environment_profile),
    }
    payload["extension_tokens"] = [str(token) for token in extension_tokens]
    payload["realization_id"] = str(realization_id)
    payload["reference_config_identity"] = str(config_identity)
    return payload


def generation_normative_obligation_for_testcase(testcase: Any) -> dict[str, Any] | None:
    obligation = getattr(testcase, "dataflow_meta", {}).get("normative_obligation")
    if not isinstance(obligation, dict):
        return None
    value = obligation.get("obligation_id")
    if (
        not isinstance(value, str) or len(value) != 28
        or not value.startswith("obl-")
        or any(character not in "0123456789abcdef" for character in value[4:])
    ):
        return None
    return dict(obligation)

"""Root recipe checks shared by route and coverage views."""

from collections.abc import Mapping
from pathlib import Path
import re

from .._util import is_sha256_digest
from ..adapters.contracts import same_execution_backend
from ..riscv_catalog import official_form
from ..spec_definedness import is_canonical_isa_profile
from ..reference_fallback import configured_k1_qemu_fallback_classes


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ROOT = _ROOT.with_name(_ROOT.name + ".local")
_RECIPE_FIELDS = ("generation", "reference", "target", "observer", "path")
_RECIPE_SECTIONS = {
    "reference": ("backend", "expected"),
    "target": ("backend", "expected"),
    "observer": ("fields",),
}
_FORMAL_TARGET_STATUSES = frozenset({
    "clean", "target-state-mismatch-candidate", "target-mismatch-candidate",
})
_REFERENCE_EXPECTED_STATUSES = frozenset({
    "normal", "completed", "trap", "nonzero-exit", "timeout", "unavailable",
    "reference-valid", "recorded",
})
_TARGET_EXPECTED_STATUSES = _FORMAL_TARGET_STATUSES | frozenset({
    "normal", "completed", "target-tested", "target-gap",
    "target-unavailable", "transport-gap", "runner-contract-gap",
    "provider-replay-tested",
})
_SINGLE_ROUTE_NAMES = frozenset({
    "single", "single-instruction", "direct", "custom", "direct/custom", "direct-custom",
})
_PROGRAM_ROUTE_NAMES = frozenset({"program", "multi-instruction"})


def _route_alias(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value in _SINGLE_ROUTE_NAMES:
        return "single"
    if value in _PROGRAM_ROUTE_NAMES:
        return "program"
    return None


def _recipe_gaps(route: Mapping[str, object]) -> list[str]:
    gaps = []
    for field in _RECIPE_FIELDS:
        value = route.get(field)
        if (field == "path" and (not isinstance(value, str) or not value.strip())) \
                or (field != "path" and (not isinstance(value, Mapping) or not value)):
            gaps.append(field)
    generation = route.get("generation")
    generation_name = generation.get("route") if isinstance(generation, Mapping) else None
    explicit_name = route.get("route")
    generation_route = _route_alias(generation_name)
    explicit_route = _route_alias(explicit_name)
    if explicit_name not in (None, "") and explicit_route is None:
        gaps.append("route")
    if (explicit_route is not None and generation_route is not None
            and explicit_route != generation_route):
        gaps.append("route-conflict")
    if isinstance(generation, Mapping) and "generation" not in gaps:
        if generation_route is None:
            gaps.append("generation.route")
        if not any(
            isinstance(generation.get(name), str) and generation[name].strip()
            for name in ("source", "test", "directed_stream", "direct")
        ):
            gaps.append("generation.source-or-test")
        if not isinstance(generation.get("isa"), str) or not is_canonical_isa_profile(
            generation["isa"].lower()
        ):
            gaps.append("generation.isa")
        if not isinstance(generation.get("mabi"), str) or not generation["mabi"].strip():
            gaps.append("generation.mabi")
        if (not isinstance(generation.get("sequence"), (list, tuple))
                or not generation["sequence"]
                or any(not isinstance(item, str) or not item.strip() for item in generation["sequence"])):
            gaps.append("generation.sequence")
        initial_state = generation.get("initial_state")
        if not isinstance(initial_state, Mapping) or not initial_state:
            gaps.append("generation.initial_state")
        generation_profile = generation.get("generation_profile")
        # ponytail: 生成选项交给生成器校验，recipe 只保留路由和身份必需字段。
        if generation_route == "program" and (
            not isinstance(generation_profile, Mapping) or not generation_profile
        ):
            gaps.append("generation.generation_profile")
        elif generation_route == "program":
            for name in ("target", "test", "isa", "mabi"):
                profile_value, generation_value = (
                    generation_profile.get(name), generation.get(name)
                )
                if profile_value not in (None, "") and generation_value not in (None, ""):
                    left = str(profile_value).lower() if name in {"isa", "mabi"} else profile_value
                    right = str(generation_value).lower() if name in {"isa", "mabi"} else generation_value
                    if left != right:
                        gaps.append(f"generation.generation_profile.{name}-conflict")
            source = generation.get("source")
            asm_test = generation_profile.get("asm_test")
            asm_path = Path(str(asm_test).replace(
                "/work/migration_Emprical_Study.local", str(_LOCAL_ROOT)
            ).replace("/work/migration_Emprical_Study", str(_ROOT))) if isinstance(asm_test, str) else None
            if (isinstance(source, str) and source.strip().lower() == "custom-program"
                    and (not isinstance(asm_test, str) or not asm_test.strip()
                         or asm_path is None or not asm_path.is_file())):
                gaps.append("generation.custom-program-source")
        generation_identity_digest = generation.get("generation_identity_digest")
        if (generation_route == "program"
                and generation_identity_digest != "runtime"
                and not is_sha256_digest(generation_identity_digest)):
            gaps.append("generation.generation_identity_digest")
        if generation_route == "single":
            source = generation.get("source")
            form = source.split(":", 1)[1].strip() if (
                isinstance(source, str) and source.startswith("framework/rvgen:")
            ) else ""
            if not form or official_form(form) is None:
                gaps.append("generation.source-form")
            if generation.get("direct") != f"generate_form({form})":
                gaps.append("generation.direct-entry")
    for section, fields in _RECIPE_SECTIONS.items():
        value = route.get(section)
        if not isinstance(value, Mapping) or section in gaps:
            continue
        for name in fields:
            if not value.get(name):
                gaps.append(f"{section}.{name}")
        if section in {"reference", "target"}:
            if not isinstance(value.get("backend"), str):
                gaps.append(f"{section}.backend")
            if not isinstance(value.get("expected"), Mapping):
                gaps.append(f"{section}.expected")
            expected = value.get("expected")
            if isinstance(expected, Mapping):
                expected_backend = expected.get("backend")
                actual_backend = value.get("backend")
                if (isinstance(expected_backend, str)
                        and isinstance(actual_backend, str)
                        and not same_execution_backend(expected_backend, actual_backend)):
                    gaps.append(f"{section}.backend-conflict")
                for name in ("binary_sha256", "identity_digest"):
                    actual = value.get(name)
                    if name in expected and not is_sha256_digest(expected[name]):
                        gaps.append(f"{section}.expected.{name}")
                    if actual not in (None, "") and not is_sha256_digest(actual):
                        gaps.append(f"{section}.{name}")
                    if (actual not in (None, "") and expected.get(name) not in (None, "")
                            and actual != expected[name]):
                        gaps.append(f"{section}.{name}-conflict")
            expected_status = expected.get("status") if isinstance(expected, Mapping) else None
            allowed_statuses = (
                _REFERENCE_EXPECTED_STATUSES if section == "reference"
                else _TARGET_EXPECTED_STATUSES
            )
            if section == "reference" and isinstance(expected, Mapping) and "status" not in expected:
                gaps.append("reference.expected.status")
            if isinstance(expected, Mapping) and "status" in expected and expected_status not in allowed_statuses:
                gaps.append(f"{section}.expected.status")
        if section == "reference" and value.get("fallback") is not None:
            fallback = value.get("fallback")
            if (
                not isinstance(fallback, Mapping)
                or value.get("backend") != "native-rv64"
                or fallback.get("backend") != "qemu-riscv64"
                or not isinstance(fallback.get("target_id"), str)
                or not fallback.get("target_id", "").strip()
                or configured_k1_qemu_fallback_classes(fallback.get("on")) is None
                or fallback.get("selection_scope") != "whole-campaign"
            ):
                gaps.append("reference.fallback.invalid")
            elif (
                isinstance(generation, Mapping)
                and isinstance(generation.get("isa"), str)
                and not generation["isa"].lower().startswith("rv64")
            ):
                gaps.append("reference.fallback.isa-mismatch")
        if section == "observer":
            fields_value = value.get("fields")
            if (not isinstance(fields_value, (list, tuple)) or not fields_value
                    or any(not isinstance(item, str) or not item.strip() for item in fields_value)):
                gaps.append("observer.fields")
            witness = value.get("semantic_witness")
            structured_witness = (
                isinstance(witness, Mapping)
                and isinstance(witness.get("mnemonic"), str)
                and witness["mnemonic"].strip()
                and isinstance(witness.get("fields", ()), (list, tuple))
                and witness.get("fields")
                and all(isinstance(field, str) and field.strip() for field in witness.get("fields", ()))
                and isinstance(witness.get("values", {}), Mapping)
            )
            if not (isinstance(witness, str) and witness.strip() or structured_witness):
                gaps.append("observer.semantic_witness")
            elif (generation_route == "program"
                  and value.get("evidence_tier") == "guided"
                  and not structured_witness):
                gaps.append("observer.semantic_witness")
    return list(dict.fromkeys(gaps))


def _recipe_route_name(recipe: Mapping[str, object]) -> str | None:
    generation = recipe.get("generation")
    generation_route = _route_alias(
        generation.get("route") if isinstance(generation, Mapping) else None
    )
    return generation_route or _route_alias(recipe.get("route"))


def _recipe_route(recipe: Mapping[str, object]) -> str:
    route = _recipe_route_name(recipe)
    if route is None:
        raise ValueError("root recipe has invalid route")
    return route


def _recipe_contains_trap_marker(value: object) -> bool:
    return any(
        token == marker or token.startswith(f"{marker}-")
        for token in re.findall(r"[a-z0-9-]+", str(value).lower())
        for marker in ("trap", "illegal", "illegal-instruction")
    )

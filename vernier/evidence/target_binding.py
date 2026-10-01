"""Target coverage 使用的 root recipe 解析与 binding 构造。"""

from collections.abc import Mapping

from framework._util import is_sha256_digest
from framework.adapters.contracts import (
    canonical_execution_backend, is_execution_backend, same_execution_backend,
)
from framework.evidence.recipe_contract import (
    _recipe_contains_trap_marker, _recipe_gaps, _recipe_route,
    _recipe_route_name, _route_alias,
)
from framework.evidence.target_contract import (
    _generation_profile_digest, _recipe_digest, _recipe_target_identity,
)


METHODS = ("B0", "B1", "B2", "B3")
EXPERIMENT_METHODS = ("S0", "S1", "S2", "S3", "S4", "S5")
RULES = ("M5", *(f"R{i}" for i in range(1, 12)))

# ``observer.fields`` historically doubled as both the execution contract and
# the semantic comparison contract.  That made a path witness (usually
# ``executed_pcs``) accidentally become the only thing compared for a GPR
# case.  Keep the declared fields for execution/provenance, but derive a small
# state projection for comparisons.  This is deliberately a projection, not a
# new admission gate: unavailable optional fields remain ordinary observation
# gaps and are handled by the existing comparison code.
_PATH_FIELDS = frozenset({
    "executed_pcs", "instruction_count", "trace_complete", "trace",
    "translation_evidence", "path_identity", "path", "program.stdout",
})
_STATE_FIELDS = {
    "gpr": ("gpr",),
    "register": ("gpr",),
    "registers": ("gpr",),
    "memory": ("memory_snapshot",),
    "memory_state": ("memory_snapshot",),
    "fp": ("fpr_rawbits", "fflags", "frm"),
    "fpr": ("fpr_rawbits", "fflags", "frm"),
    "floating_point": ("fpr_rawbits", "fflags", "frm"),
}


def _observer_fields(fields: object) -> tuple[str, ...]:
    if not isinstance(fields, (list, tuple)):
        return ()
    aliases = {
        "after_gpr": "gpr", "after_fpr_rawbits": "fpr_rawbits",
        "after_fflags": "fflags", "after_frm": "frm",
    }
    return tuple(dict.fromkeys(
        aliases.get(field, field)
        for field in fields
        if isinstance(field, str) and field.strip()
    ))


def _comparison_fields(state: object, fields: tuple[str, ...]) -> tuple[str, ...]:
    result = [
        field for field in fields
        if field not in _PATH_FIELDS
        and not field.startswith(("extra.", "translation.", "target."))
    ]
    state_name = state.strip().lower() if isinstance(state, str) else ""
    result.extend(_STATE_FIELDS.get(state_name, ()))
    # Outcome is architectural termination state and must remain in every
    # projection even when an old recipe only declared its path witness.
    if "outcome" not in result:
        result.append("outcome")
    return tuple(dict.fromkeys(result))


def _concrete_sequence(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        from ..riscv_catalog import official_form
    except ImportError:
        return None
    sequence = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        mnemonic = item.strip().split(None, 1)[0].lower()
        if mnemonic.startswith(".") or "/" in mnemonic:
            return None
        if official_form(mnemonic.replace(".", "_")) is None and mnemonic not in {
            "li", "la", "csrr", "fsflags", "mv",
        }:
            return None
        sequence.append(item)
    return sequence


def resolve_root_recipe(
    ledger: Mapping[str, object], root_key: str, lineage_key: str, *,
    route: str | None = None,
) -> dict[str, object]:
    if route not in (None, "") and _route_alias(route) is None:
        raise ValueError("root recipe has invalid route")
    route_filter = _route_alias(route) if isinstance(route, str) and route else route
    routes = tuple(
        record for record in ledger.get("route_records", ())
        if isinstance(record, Mapping)
        and record.get("target_scope_included") is True
        and record.get("root_key") == root_key
        and record.get("lineage_key") == lineage_key
        and (route_filter in (None, "") or _recipe_route_name(record) == route_filter)
    )
    ready = tuple(route for route in routes if not _recipe_gaps(route))
    label = (root_key, lineage_key) if route in (None, "") else (root_key, lineage_key, route)
    if len(routes) > 1 or len(ready) > 1:
        raise ValueError(f"root recipe is ambiguous: {label}")
    if len(routes) != 1 or len(ready) != 1:
        gaps = ", ".join(_recipe_gaps(routes[0])) if len(routes) == 1 else ""
        raise ValueError(f"root recipe is incomplete: {label}" + (f" ({gaps})" if gaps else ""))
    return dict(ready[0])


def root_binding_for_recipe(
    recipe: Mapping[str, object], *, method: str, rule: str, state: str,
    observer: str, target: str, evidence_path: str | None = None,
    validate_target: bool = True,
) -> dict[str, object]:
    if not isinstance(recipe, Mapping) or _recipe_gaps(recipe):
        raise ValueError("root recipe is incomplete")
    fields = (method, rule, state, observer, target)
    if (method not in (*METHODS, *EXPERIMENT_METHODS) or rule not in RULES
            or any(not isinstance(value, str) or not value.strip() for value in fields)):
        raise ValueError("root binding is invalid")
    recipe_target = recipe["target"]["backend"]
    generation = recipe.get("generation")
    if validate_target:
        if canonical_execution_backend(target) != canonical_execution_backend(recipe_target):
            raise ValueError("root binding target does not match recipe")
        if not same_execution_backend(target, recipe_target):
            raise ValueError("root binding target does not match recipe ISA")
        if not is_execution_backend(target):
            raise ValueError("root binding target is unknown")
    target_identity_digest, target_binary_sha256 = _recipe_target_identity(recipe)
    if target_identity_digest not in (None, "") and not is_sha256_digest(target_identity_digest):
        raise ValueError("root recipe target identity is invalid")
    if target_binary_sha256 not in (None, "") and not is_sha256_digest(target_binary_sha256):
        raise ValueError("root recipe target binary identity is invalid")
    root_key, lineage_key = recipe.get("root_key"), recipe.get("lineage_key")
    if any(not isinstance(value, str) or not value.strip() for value in (root_key, lineage_key)):
        raise ValueError("root recipe is missing root/lineage key")
    binding = {
        "root_key": root_key, "lineage_key": lineage_key, "method": method,
        "rule": rule, "state": state, "observer": observer, "target": target,
        "route": _recipe_route(recipe),
        "root_recipe_digest": _recipe_digest(recipe),
        "reference_backend": recipe["reference"]["backend"],
        "harness": (
            "rvgen-single" if _recipe_route(recipe) == "single"
            else "custom" if "custom" in str(generation.get("source", "")).lower()
            else "riscv-dv"
        ),
    }
    # Both Ours routes use the trusted recipe reference for the MH loop.  The
    # Target is an observation/coverage side channel and is replayed after the
    # reference-driven chain closes.  Keep this in the binding, rather than in
    # a CLI-only branch, so direct library callers and both route CLIs share
    # the same feedback contract.
    if binding["route"] in {"single", "program"}:
        binding["mcmc_feedback_source"] = "reference"
    reference = recipe.get("reference")
    expected_reference = reference.get("expected") if isinstance(reference, Mapping) else None
    if isinstance(expected_reference, Mapping):
        reference_backend = reference.get("backend") if isinstance(reference, Mapping) else None
        expected_backend = expected_reference.get("backend")
        if (expected_backend not in (None, "")
                and not same_execution_backend(expected_backend, reference_backend)):
            raise ValueError("root recipe reference identity is invalid")
        binding["reference_expected"] = {
            **dict(expected_reference),
            **({"backend": reference_backend} if isinstance(reference_backend, str) else {}),
        }
    if isinstance(reference, Mapping) and reference.get("identity_resolution") == "runtime":
        binding["reference_identity_resolution"] = "runtime"
    reference_fallback = reference.get("fallback") if isinstance(reference, Mapping) else None
    if isinstance(reference_fallback, Mapping):
        binding["reference_fallback_policy"] = dict(reference_fallback)
    target_spec = recipe.get("target")
    if isinstance(target_spec, Mapping) and target_spec.get("identity_resolution") == "runtime":
        binding["target_identity_resolution"] = "runtime"
    observer_contract = recipe.get("observer")
    semantic_witness = observer_contract.get("semantic_witness") if isinstance(observer_contract, Mapping) else None
    fields = observer_contract.get("fields") if isinstance(observer_contract, Mapping) else None
    normalized_fields = _observer_fields(fields)
    state_fields = _comparison_fields(state, normalized_fields)
    if normalized_fields or state_fields:
        # Keep path fields in observer_fields because runners still need them,
        # and add the state fields required by the declared route state.  The
        # separate comparison_fields projection is what prevents path
        # identity from silently standing in for architectural state.
        binding["observer_fields"] = list(dict.fromkeys((*normalized_fields, *state_fields)))
        binding["comparison_fields"] = list(state_fields)
    evidence_tier = observer_contract.get("evidence_tier") if isinstance(observer_contract, Mapping) else None
    if isinstance(evidence_tier, str) and evidence_tier.strip():
        binding["evidence_tier"] = evidence_tier
    if (
        isinstance(semantic_witness, str) and semantic_witness.strip()
        or isinstance(semantic_witness, Mapping)
    ):
        binding["semantic_witness"] = dict(semantic_witness) if isinstance(semantic_witness, Mapping) else semantic_witness
    if isinstance(generation, Mapping):
        sequence = _concrete_sequence(generation["sequence"])
        if _recipe_route(recipe) == "program" and sequence is None:
            binding["sequence_mode"] = "dynamic"
        else:
            binding["required_sequence"] = list(generation["sequence"])
        if any(_recipe_contains_trap_marker(item)
               for item in (*generation.get("sequence", ()), generation.get("initial_state"))):
            binding["expected_trap"] = True
    if is_sha256_digest(target_identity_digest):
        binding["target_identity_digest"] = target_identity_digest
    if is_sha256_digest(target_binary_sha256):
        binding["target_binary_sha256"] = target_binary_sha256
    if isinstance(generation, Mapping) and isinstance(generation.get("generation_profile"), Mapping):
        binding["generation_profile_digest"] = _generation_profile_digest(generation["generation_profile"])
    if isinstance(generation, Mapping) and generation.get("generation_identity_digest"):
        if generation["generation_identity_digest"] != "runtime" and not is_sha256_digest(generation["generation_identity_digest"]):
            raise ValueError("root recipe generation identity is invalid")
        if generation["generation_identity_digest"] != "runtime":
            binding["generation_identity_digest"] = generation["generation_identity_digest"]
    if evidence_path is not None:
        if not isinstance(evidence_path, str) or not evidence_path.strip():
            raise ValueError("root evidence path is invalid")
        binding["evidence_path"] = evidence_path
    return binding

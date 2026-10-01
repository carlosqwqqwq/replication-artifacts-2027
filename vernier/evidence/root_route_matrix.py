"""把 canonical root 展开为独立的 single/program route 记录。"""

from collections.abc import Iterable, Mapping
from pathlib import Path

from .._util import is_sha256_digest, read_json_object, sha256_file
from ..adapters.contracts import canonical_execution_backend, is_execution_backend
from ..riscv_catalog import official_form
from ..spec_definedness import enabled_extensions, mabi_matches_isa
from .recipe_contract import _recipe_gaps, _recipe_route_name


ROUTES = ("single", "program")
_REFERENCE_BACKENDS = frozenset(canonical_execution_backend(item) for item in (
    "native-rv64", "sail-riscv", "rvgen-semantic", "qemu", "qemu-riscv32", "qemu-riscv64",
))
_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ROOT = _ROOT.with_name(_ROOT.name + ".local")


def _materialized_source(recipe: Mapping[str, object]) -> bool:
    return _materialized_source_path(recipe) is not None


def _materialized_source_path(recipe: Mapping[str, object]) -> Path | None:
    generation = recipe.get("generation")
    profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
    contract = profile.get("provider_contract") if isinstance(profile, Mapping) else None
    source_sha = (
        contract.get("source_sha256") if isinstance(contract, Mapping) else None
    ) or (profile.get("source_sha256") if isinstance(profile, Mapping) else None)

    def paths(value: object) -> tuple[Path, ...]:
        if not isinstance(value, str) or not value.strip():
            return ()
        text = value.replace("\\", "/")
        path = Path(value)
        for prefix, base in (
            ("/work/migration_Emprical_Study.local/", _LOCAL_ROOT),
            ("/work/migration_Emprical_Study/", _ROOT),
        ):
            if text.startswith(prefix):
                path = base / text.removeprefix(prefix)
                break
        if path.is_file():
            return (path,) if path.suffix.lower() in {".s", ".asm"} else ()
        if not path.is_dir():
            return ()
        return tuple(
            candidate
            for directory in (path / "generation" / "asm_test", path / "asm_test", path)
            for candidate in sorted((*directory.glob("*.S"), *directory.glob("*.s")))
        )

    for value in (
        profile.get("asm_test") if isinstance(profile, Mapping) else None,
        recipe.get("source_locator"),
    ):
        for candidate in paths(value):
            if not is_sha256_digest(source_sha) or sha256_file(candidate) == source_sha:
                return candidate
    root_key = recipe.get("root_key")
    if isinstance(root_key, str) and is_sha256_digest(source_sha):
        base = _LOCAL_ROOT / "framework-experiments" / root_key
        candidates = sorted(base.glob("S*/*/generation/asm_test/*.S"))
        candidates += sorted(base.glob("**/*.S")) if not candidates else []
        return next((candidate for candidate in candidates if sha256_file(candidate) == source_sha), None)
    return None


def _enrich_direct_copy_source_identity(recipe: Mapping[str, object]) -> dict[str, object]:
    value = dict(recipe)
    generation = value.get("generation")
    profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
    if not isinstance(generation, Mapping) or not isinstance(profile, Mapping):
        return value
    if (profile.get("gen_opts") or generation.get("gen_opts")) != "direct-copy" \
            or not _materialized_source(value):
        return value
    if is_sha256_digest(profile.get("source_sha256")):
        return value
    source = _materialized_source_path(value)
    if source is None:
        return value
    updated_generation = dict(generation)
    updated_profile = dict(profile)
    updated_profile["source_sha256"] = sha256_file(source)
    updated_generation["generation_profile"] = updated_profile
    value["generation"] = updated_generation
    return value


def _target_probe_binding_current(
    recipe: Mapping[str, object], binding: Mapping[str, object],
) -> bool:
    target = recipe.get("target")
    expected = target.get("expected") if isinstance(target, Mapping) else None
    return (
        binding.get("binding_contract") == "program-target-probe-binding-v2"
        and isinstance(target, Mapping)
        and isinstance(expected, Mapping)
        and binding.get("target_backend") == target.get("backend")
        and is_sha256_digest(expected.get("binary_sha256"))
        and is_sha256_digest(expected.get("identity_digest"))
        and is_sha256_digest(binding.get("target_binary_sha256"))
        and is_sha256_digest(binding.get("target_identity_digest"))
        and binding.get("target_binary_sha256") == expected.get("binary_sha256")
        and binding.get("target_identity_digest") == expected.get("identity_digest")
        and (
            not binding.get("reference_backend_override")
            or (
                binding.get("reference_backend") == binding.get("reference_backend_override")
                and is_sha256_digest(binding.get("reference_binary_sha256"))
                and is_sha256_digest(binding.get("reference_identity_digest"))
            )
        )
    )


def _sequence_extension_gaps(
    sequence: object, isa: object, qemu_cpu: object = None,
) -> tuple[str, ...]:
    if not isinstance(sequence, (list, tuple)) or not isinstance(isa, str):
        return ()
    enabled = set(enabled_extensions(isa))
    if isinstance(qemu_cpu, str):
        enabled.update(
            token.split("=", 1)[0].lower()
            for token in qemu_cpu.split(",")[1:]
            if token.lower().endswith("=true")
        )
    missing = set()
    for item in sequence:
        if not isinstance(item, str) or not item.strip():
            continue
        mnemonic = item.split()[0].lower().replace(".", "_")
        form = official_form(mnemonic)
        if form is None:
            continue
        groups = []
        for source in form.source_extensions:
            tokens = source.lower().split("_")
            if tokens and tokens[0] in {"rv", "rv32", "rv64"}:
                tokens = tokens[1:]
            if tokens == ["zicbo"]:
                groups.extend(({"zicbom"}, {"zicboz"}))
                continue
            tokens = {
                token for token in tokens
                if token in {"i", "e", "m", "a", "f", "d", "q", "c", "v"}
                or token.startswith(("z", "x"))
            }
            if tokens:
                groups.append(tokens)
        if groups and not any(group <= enabled for group in groups):
            missing.update(sorted(groups[0] - enabled))
    return tuple(sorted(missing))


def _provider_replay_binding_gaps(
    recipe: Mapping[str, object], binding: Mapping[str, object],
) -> list[str]:
    if not isinstance(binding.get("binding_contract"), str) \
            or not binding["binding_contract"].strip():
        return ["provider-replay.binding-contract-invalid"]
    evidence_value = binding.get("evidence_path")
    if not isinstance(evidence_value, str) or not evidence_value.strip():
        return ["provider-replay.evidence-missing"]
    try:
        evidence = read_json_object(Path(evidence_value))
    except (OSError, ValueError):
        return ["provider-replay.evidence-missing"]
    contract = binding.get("binding_contract")
    if not isinstance(contract, str) or evidence.get("schema") != contract:
        return ["provider-replay.binding-contract-mismatch"]
    if evidence.get("status") != "complete":
        return ["provider-replay.evidence-incomplete"]
    cases = evidence.get("cases")
    case_name = binding.get("case")
    case = next(
        (
            item for item in cases if isinstance(item, Mapping)
            and item.get("root_key") == recipe.get("root_key")
            and item.get("lineage_key") == recipe.get("lineage_key")
            and item.get("case") == case_name
        ),
        None,
    ) if isinstance(cases, list) else None
    if case is None or case.get("status") != "provider-replay-tested":
        return ["provider-replay.case-missing"]
    if not isinstance(case.get("execution_plane"), str) or not case["execution_plane"].strip():
        return ["provider-replay.execution-plane-missing"]
    if case.get("runner_script_exit_code") not in (None, 0):
        return ["provider-replay.runner-failed"]
    if case.get("stable") is False or case.get("target_legs_stable") is False:
        return ["provider-replay.unstable"]
    log_path = binding.get("log_path")
    log_sha = binding.get("log_sha256")
    case_log_path = case.get("log_path")
    case_log_sha = case.get("log_sha256")
    if not isinstance(log_path, str) or not isinstance(case_log_path, str) \
            or Path(log_path).resolve() != Path(case_log_path).resolve():
        return ["provider-replay.log-mismatch"]
    if not is_sha256_digest(log_sha) or log_sha != case_log_sha:
        return ["provider-replay.log-hash-mismatch"]
    try:
        if not Path(log_path).is_file() or sha256_file(Path(log_path)) != log_sha:
            return ["provider-replay.log-invalid"]
    except OSError:
        return ["provider-replay.log-invalid"]
    for field in ("source_locator", "source_sha256", "runner_locator", "runner_sha256"):
        if field in binding and binding.get(field) != case.get(field):
            return [f"provider-replay.{field}-mismatch"]
    source_sha = case.get("source_sha256")
    runner_sha = case.get("runner_sha256")
    if not is_sha256_digest(source_sha) or not is_sha256_digest(runner_sha):
        return ["provider-replay.source-runner-hash-invalid"]
    generation = recipe.get("generation")
    profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
    provider_contract = profile.get("provider_contract") if isinstance(profile, Mapping) else None
    if not isinstance(provider_contract, Mapping):
        return ["provider-replay.provider-contract-missing"]
    source_locator = provider_contract.get("source_locator") or recipe.get("source_locator")
    if source_locator != case.get("source_locator"):
        return ["provider-replay.source-locator-mismatch"]
    if provider_contract.get("source_sha256") != source_sha:
        return ["provider-replay.source-hash-mismatch"]
    for field in ("runner_locator", "runner_sha256"):
        if field in provider_contract and provider_contract.get(field) != case.get(field):
            return [f"provider-replay.{field}-mismatch"]
    return []


def _execution_recipe_gaps(recipe: Mapping[str, object] | None) -> list[str]:
    if not isinstance(recipe, Mapping):
        return ["recipe-not-ready"]
    generation = recipe.get("generation")
    target = recipe.get("target")
    isa = generation.get("isa") if isinstance(generation, Mapping) else None
    backend = target.get("backend") if isinstance(target, Mapping) else None
    # Target ISA/extension capability is a runtime observation.  Do not reject
    # a generated case here merely because a backend has no static allowlist.
    provider_replay = recipe.get("provider_replay_binding")
    if isinstance(provider_replay, Mapping) and provider_replay.get("status") == "provider-replay-tested":
        return _provider_replay_binding_gaps(recipe, provider_replay)
    execution = recipe.get("execution_contract")
    if isinstance(execution, Mapping) and execution.get("status") not in (None, "ready"):
        if not (execution.get("status") == "not-run" and _materialized_source(recipe)):
            return [f"execution-contract:{execution.get('status') or 'gap'}"]
    observer = recipe.get("observer")
    reference = recipe.get("reference")
    reference_backend = reference.get("backend") if isinstance(reference, Mapping) else None
    binding = recipe.get("target_probe_binding")
    if (
        isinstance(binding, Mapping)
        and isinstance(reference, Mapping)
        and binding.get("status") == "promoted"
        and _target_probe_binding_current(recipe, binding)
        and binding.get("reference_backend_override")
    ):
        reference_backend = binding.get("reference_backend")
        reference_expected = {
            "status": binding.get("reference_observed_status"),
            "binary_sha256": binding.get("reference_binary_sha256"),
            "identity_digest": binding.get("reference_identity_digest"),
        }
        reference = {**dict(reference), "backend": reference_backend, "expected": reference_expected}
    reference_available = (
        isinstance(reference_backend, str)
        and bool(reference_backend.strip())
        and canonical_execution_backend(reference_backend) in _REFERENCE_BACKENDS
    )
    if not reference_available:
        reference_backend = None
    reference_expected = reference.get("expected") if isinstance(reference, Mapping) else None
    if isinstance(reference_expected, Mapping) and reference_expected.get("status") == "unavailable":
        reference_expected = None
    if not isinstance(backend, str) or not is_execution_backend(backend):
        return ["target.backend-unavailable"]
    if _recipe_gaps(recipe):
        return ["recipe-not-ready"]
    # A shared backend weakens differential evidence but does not stop target
    # execution or source coverage.
    target_expected = target.get("expected") if isinstance(target, Mapping) else None
    target_status = target_expected.get("status") if isinstance(target_expected, Mapping) else None
    mabi = generation.get("mabi") if isinstance(generation, Mapping) else None
    profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
    if isinstance(isa, str) and isinstance(mabi, str):
        isa_xlen = isa.lower()[:4]
        abi_xlen = (
            "rv32" if mabi.lower().startswith("ilp32")
            else "rv64" if mabi.lower().startswith("lp64") else None
        )
        if (isa_xlen in {"rv32", "rv64"} and abi_xlen != isa_xlen):
            return ["generation.mabi-isa-mismatch"]
        if not mabi_matches_isa(isa, mabi):
            return ["generation.mabi-isa-mismatch"]
    source = generation.get("source") if isinstance(generation, Mapping) else None
    if (
        _recipe_route_name(recipe) == "program"
        and isinstance(profile, Mapping)
        and isinstance(source, str)
        and source.lower() in {"riscv-dv", "riscv_dv"}
    ):
        for name in ("target", "test", "isa", "mabi"):
            if not isinstance(profile.get(name), str) or not profile[name].strip():
                return [f"generation.profile-{name}-missing"]
        for name in ("target", "test"):
            profile_value = profile.get(name)
            generation_value = generation.get(name)
            value = profile_value or generation_value
            if not isinstance(value, str) or not value.strip():
                return [f"generation.program-{name}-missing"]
            if (
                isinstance(profile_value, str) and profile_value.strip()
                and isinstance(generation_value, str) and generation_value.strip()
                and profile_value != generation_value
            ):
                return [f"generation.profile-{name}-conflict"]
        # The generator profile describes the tool's own capability.  It is
        # not compared with a shared experiment lane; the adapter may emit a
        # wider or narrower profile and the target decides at execution time.
    # Sequence extensions and target status are runtime facts.  Keep them in
    # the per-case evidence and let the simulator advance to the next case;
    # neither is a campaign-wide admission gate.
    # Keep the reference identity in the result; do not use backend identity
    # as a target launch gate.
    route_name = _recipe_route_name(recipe)
    if route_name == "program" or isinstance(recipe.get("route_key"), str):
        def identity_digest(section: object) -> object:
            if not isinstance(section, Mapping):
                return None
            expected = section.get("expected")
            return section.get("identity_digest") or (
                expected.get("identity_digest")
                if isinstance(expected, Mapping) else None
            )

        reference_identity = identity_digest(reference)
        target_identity = identity_digest(target)
        if route_name in {"single", "program"}:
            runtime_reference = (
                isinstance(reference, Mapping)
                and reference.get("identity_resolution") == "runtime"
            )
            runtime_target = (
                isinstance(target, Mapping)
                and target.get("identity_resolution") == "runtime"
            )
            if not is_sha256_digest(reference_identity) and not runtime_reference:
                return ["reference.expected.identity_digest-missing"]
            if not is_sha256_digest(target_identity) and not runtime_target:
                return ["target.expected.identity_digest-missing"]
        elif any(
            value is not None and not is_sha256_digest(value)
            for value in (reference_identity, target_identity)
        ):
            return ["route.expected.identity_digest-invalid"]
        if reference_identity is not None and target_identity is not None \
                and reference_identity == target_identity:
            return ["reference-target.identity-not-distinct"]
    # Target ISA and extension support are runtime facts.  The generated ELF
    # must reach the simulator so it can report an unsupported instruction;
    # that result is recorded per case and the caller advances the case loop.
    return []


def _gap_reason(
    recipe_status: str,
    recipe_gaps: Iterable[object],
    execution_gaps: Iterable[object],
    route_gap: Mapping[str, object] | None = None,
) -> str | None:
    if isinstance(route_gap, Mapping):
        value = route_gap.get("reason") or route_gap.get("status")
        return str(value) if value else None
    gaps = recipe_gaps if recipe_status != "recipe-ready" else execution_gaps
    return next((str(value) for value in gaps if value), None)


def _validate_canonical_root_declaration(
    ledger: Mapping[str, object],
    groups: Mapping[tuple[str, str], list[Mapping[str, object]]],
) -> None:
    counts = ledger.get("counts")
    if not isinstance(counts, Mapping) or counts.get("canonical_root_count_frozen") is not True:
        return
    declared = ledger.get("canonical_root_keys")
    if not isinstance(declared, (list, tuple)) or any(
        not isinstance(value, str) or not value.strip() for value in declared
    ):
        raise ValueError("frozen canonical root declaration is invalid")
    lineages = ledger.get("canonical_lineage_keys")
    if not isinstance(lineages, (list, tuple)) or any(
        not isinstance(value, str) or not value.strip() for value in lineages
    ):
        raise ValueError("frozen canonical lineage declaration is invalid")
    if len(set(lineages)) != len(lineages):
        raise ValueError("frozen canonical lineage declaration is duplicated")
    expected = set(declared)
    if len(expected) != len(declared):
        raise ValueError("frozen canonical root declaration is duplicated")
    if counts.get("canonical_target_root_count") != len(expected):
        raise ValueError("frozen canonical root count does not match declaration")
    lineage_count = counts.get("canonical_target_lineage_count")
    if lineage_count is not None and lineage_count != len(set(lineages)):
        raise ValueError("frozen canonical lineage count does not match declaration")
    actual_roots = {root_key for root_key, _lineage_key in groups}
    if actual_roots != expected:
        raise ValueError("eligible roots do not match frozen canonical root declaration")
    actual_lineages = {lineage_key for _root_key, lineage_key in groups}
    if len(lineages) == len(declared):
        if set(groups) != set(zip(declared, lineages)):
            raise ValueError("eligible roots do not match frozen canonical root declaration")
    elif actual_lineages != set(lineages):
        raise ValueError("eligible lineages do not match frozen canonical lineage declaration")
    root_lineages = {}
    for root_key, lineage_key in groups:
        root_lineages.setdefault(root_key, set()).add(lineage_key)
    if any(len(values) != 1 for values in root_lineages.values()):
        raise ValueError("canonical root maps to multiple lineages")


def build_root_route_matrix(
    ledger: Mapping[str, object], *, routes: Iterable[str] = ROUTES,
) -> dict[str, object]:
    selected_routes = tuple(dict.fromkeys(routes))
    if not selected_routes or any(route not in ROUTES for route in selected_routes):
        raise ValueError("root route scope is invalid")
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    source_ledger = ledger
    source_path = ledger.get("source_ledger")
    if not ledger.get("root_records") and isinstance(source_path, str):
        path = Path(source_path)
        path = path if path.is_absolute() else _ROOT / path
        if path.is_file():
            source_ledger = read_json_object(path)
    root_records = {
        (record.get("root_key"), record.get("lineage_key")): record
        for record in source_ledger.get("root_records", ())
        if isinstance(record, Mapping)
        and isinstance(record.get("root_key"), str)
        and isinstance(record.get("lineage_key"), str)
    }
    source_groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in source_ledger.get("route_records", ()):
        key = (record.get("root_key"), record.get("lineage_key"))
        if isinstance(record, Mapping) and all(isinstance(value, str) for value in key):
            source_groups.setdefault(key, []).append(record)
    invalid_records = 0
    route_records = ledger.get("route_records", ())
    if not isinstance(route_records, (list, tuple)):
        raise ValueError("root ledger route_records must be a list")
    declared_counts = ledger.get("counts")
    expected_root_count = (
        declared_counts.get("canonical_target_root_count")
        if isinstance(declared_counts, Mapping)
        and declared_counts.get("canonical_root_count_frozen") is True
        else None
    )
    for record in route_records:
        if not isinstance(record, Mapping) or record.get("target_scope_included") is not True:
            continue
        key = (record.get("root_key"), record.get("lineage_key"))
        if any(not isinstance(value, str) or not value.strip() for value in key):
            invalid_records += 1
            continue
        groups.setdefault(key, []).append(record)
    if expected_root_count is not None:
        declared_roots = ledger.get("canonical_root_keys")
        declared_lineages = ledger.get("canonical_lineage_keys")
        if (
            isinstance(declared_roots, (list, tuple))
            and isinstance(declared_lineages, (list, tuple))
            and len(declared_roots) == len(declared_lineages)
        ):
            groups.update(
                ((root, lineage), [])
                for root, lineage in zip(declared_roots, declared_lineages)
                if (root, lineage) not in groups
            )
        _validate_canonical_root_declaration(ledger, groups)
    elif groups:
        _validate_canonical_root_declaration(ledger, groups)

    rows = []
    for (root_key, lineage_key), records in sorted(groups.items()):
        root_record = root_records.get((root_key, lineage_key), {})
        provenance = [root_record, *source_groups.get((root_key, lineage_key), ()), *records]
        provenance.extend(
            value.get("source_provenance")
            for value in records
            if isinstance(value.get("source_provenance"), Mapping)
        )
        origins = set()
        if any(record.get("route_kind") in {"external-submission", "submitted-or-disposed"} for record in provenance):
            origins.add("community-submitted")
        if any(record.get("route_kind") in {"pending-submission", "independent-root-projection"} for record in provenance):
            origins.add("self-found")
        classified = bool(origins)
        refs = {
            ref for record in provenance
            for field in ("source_refs", "source_locator", "issue", "pr")
            for ref in (record.get(field),)
            if isinstance(ref, str) and ref.strip()
        }
        if not classified:
            for ref in refs:
                if "#/external_submissions/" in ref or ref.startswith("community-submissions/"):
                    origins.add("community-submitted")
                elif "#/local_posting_ready/" in ref:
                    origins.add("self-found")
        origin = "both" if len(origins) > 1 else next(iter(origins), "unknown")
        root_refs = {
            ref for ref in root_record.get("source_refs", ())
            if isinstance(ref, str) and ref.strip()
        }
        source_refs = sorted(root_refs | refs)
        preferred = (
            "community-submissions/", "#/external_submissions/", "http"
        ) if origin in {"community-submitted", "both"} else (
            "framework/evidence/qemu-program-sources/", "community-submissions/",
            "framework-experiments/", "#/local_posting_ready/", "framework/evidence/"
        )
        source_ref = next(
            (ref for marker in preferred for ref in source_refs if marker in ref),
            source_refs[0] if source_refs else None,
        )
        defect_ids = sorted({
            defect_id for record in provenance
            for defect_id in (record.get("defect_ids", ()) if isinstance(record.get("defect_ids"), (list, tuple)) else ())
            if isinstance(defect_id, str) and defect_id.strip()
        })
        for route in selected_routes:
            candidates = [record for record in records if _recipe_route_name(record) == route]
            gap_record = (
                candidates[0].get("route_gap")
                if len(candidates) == 1 and isinstance(candidates[0].get("route_gap"), Mapping)
                else None
            )
            if len(candidates) > 1:
                recipe = None
                gaps = ["recipe-ambiguous"]
                status = "route-gap"
                source_keys = [record.get("route_key") for record in candidates]
            elif gap_record is not None:
                recipe = None
                gaps = [str(gap_record.get("status") or f"{route}-route-gap")]
                status = "route-gap"
                source_keys = [candidates[0].get("route_key")]
            elif len(candidates) == 1:
                recipe = dict(candidates[0])
                recipe = _enrich_direct_copy_source_identity(recipe)
                gaps = _recipe_gaps(recipe)
                status = "recipe-ready" if not gaps else "recipe-gap"
                route_key = recipe.get("route_key")
                source_keys = (
                    [route_key] if not gaps and isinstance(route_key, str) and route_key
                    else [] if not gaps else [record.get("route_key") for record in candidates]
                )
            else:
                recipe = None
                gaps = [f"{route}-route-gap"]
                status = "route-gap"
                source_keys = [record.get("route_key") for record in candidates]
            route_attempt = candidates[0].get("route_attempt") if len(candidates) == 1 else None
            attempted = bool(candidates) and (
                isinstance(route_attempt, Mapping)
                and route_attempt.get("status") == "attempted"
            ) and (gap_record is None or gap_record.get("attempt_status") == "attempted")
            execution_gaps = _execution_recipe_gaps(recipe)
            row = {
                "root_key": root_key,
                "lineage_key": lineage_key,
                "route": route,
                "status": status,
                "recipe_status": "recipe-ready" if status == "recipe-ready" else "recipe-gap",
                "attempt_status": "attempted" if attempted else "unattempted",
                "recipe_gaps": gaps,
                "execution_status": (
                    "executable-ready"
                    if status == "recipe-ready" and not execution_gaps
                    else "execution-gap"
                ),
                "evidence_status": "unobserved",
                "execution_gaps": execution_gaps,
                "gap_reason": _gap_reason(status, gaps, execution_gaps, gap_record),
                "source_record_count": len(candidates),
                "source_route_keys": source_keys,
                "source_locators": sorted({
                    record["source_locator"] for record in candidates
                    if isinstance(record.get("source_locator"), str)
                    and record["source_locator"].strip()
                }),
                "origin": origin,
                "source_ref": source_ref,
                "source_refs": source_refs,
                "defect_ids": defect_ids,
                "recipe": recipe,
            }
            if len(candidates) > 1:
                row["source_candidates"] = [
                    {
                        name: dict(value) if isinstance(value, Mapping) else value
                        for name in (
                            "route_key", "source_locator", "route_gap",
                            "source_provenance", "route_attempt",
                        )
                        if (value := record.get(name)) not in (None, "")
                    }
                    for record in candidates
                ]
            rows.append(row)
            if len(candidates) == 1:
                attempt = candidates[0].get("route_attempt")
                if isinstance(attempt, Mapping):
                    rows[-1]["route_attempt"] = dict(attempt)
            if gap_record is not None:
                rows[-1]["route_gap"] = dict(gap_record)
                provenance = candidates[0].get("source_provenance")
                if isinstance(provenance, Mapping):
                    rows[-1]["source_provenance"] = dict(provenance)
    scoped_record_counts = tuple(
        sum(_recipe_route_name(record) in selected_routes for record in records)
        for records in groups.values()
    )
    complete = bool(groups) and all(
        count == len(selected_routes)
        for count in scoped_record_counts
    ) and all(
        {
            _recipe_route_name(record)
            for record in records
            if _recipe_route_name(record) in selected_routes
        } == set(selected_routes)
        for records in groups.values()
    )
    counts = {
        "root_count": len(groups),
        "source_route_record_count": sum(scoped_record_counts),
        "route_record_count": len(rows),
        "route_slot_complete": bool(groups) and len(rows) == len(groups) * len(selected_routes),
        # 只有已经具备 recipe 的路线才有资格进入执行尝试审计；缺失路线
        # 由 route-slots/route-assignment/route-recipes 单独报告。
        "route_attempt_eligible_count": sum(
            row["status"] == "recipe-ready" for row in rows
        ),
        "route_attempt_complete": all(
            row["attempt_status"] == "attempted"
            for row in rows if row["status"] == "recipe-ready"
        ),
        "route_unattempted": sum(
            row["attempt_status"] == "unattempted"
            for row in rows if row["status"] == "recipe-ready"
        ),
        "source_route_assignment_complete": complete,
        "source_route_record_assigned_count": sum(scoped_record_counts),
        "source_route_record_unassigned_count": sum(
            _recipe_route_name(record) is None
            for records in groups.values() for record in records
        ),
        "single_recipe_ready": sum(
            row["route"] == "single" and row["status"] == "recipe-ready" for row in rows
        ),
        "single_recipe_gap": sum(
            row["route"] == "single" and row["status"] != "recipe-ready" for row in rows
        ),
        "program_recipe_ready": sum(
            row["route"] == "program" and row["status"] == "recipe-ready" for row in rows
        ),
        "program_recipe_gap": sum(
            row["route"] == "program" and row["status"] != "recipe-ready" for row in rows
        ),
        "single_executable_ready": sum(
            row["route"] == "single" and row["execution_status"] == "executable-ready"
            for row in rows
        ),
        "program_executable_ready": sum(
            row["route"] == "program" and row["execution_status"] == "executable-ready"
            for row in rows
        ),
    }
    audit_gaps = []
    if expected_root_count is not None and len(groups) != expected_root_count:
        audit_gaps.append("canonical-root-coverage")
    if not counts["route_slot_complete"]:
        audit_gaps.append("route-slots")
    if not counts["source_route_assignment_complete"]:
        audit_gaps.append("route-assignment")
    if not counts["route_attempt_complete"]:
        audit_gaps.append("route-attempt")
    if invalid_records:
        audit_gaps.append("route-record-invalid")
    return {
        "contract": "rvgen-root-route-matrix-v1",
        "source_schema": ledger.get("schema"),
        "routes": list(selected_routes),
        "counts": counts,
        "route_audit_status": "complete" if not audit_gaps else "incomplete",
        "route_audit_gaps": audit_gaps,
        "rows": rows,
    }

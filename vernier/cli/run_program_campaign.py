"""把一条 program recipe 接入生成器、可选 ELF 转换和模拟器。"""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework._util import atomic_write_json, canonical_digest, json_safe, read_json_object
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from framework.adapters.program import (
    _observations, enumerate_case_rewrites, executed_source_locations,
    witness_safe_case_rewrites_for_model,
)
from framework.adapters.program_observer import build_linux_user
from framework.adapters.riscv_dv import _normalize_generation_profile, generate_program
from framework.evidence.root_route_matrix import _execution_recipe_gaps
from framework.evidence.recipe_contract import _recipe_gaps
from framework.evidence.target_coverage import (
    _recipe_route_name,
    resolve_root_recipe,
    root_binding_for_recipe,
    summarize_target_coverage_by_target,
)
from framework.framework_coverage import (
    coverage_observation_identity, finalize_framework_coverage,
    wait_for_dotnet_coverage,
)
from framework.program_campaign import (
    _compact_chain_record, _generation_gap_result,
    reference_witness_source_lines, run_program_campaign,
)
from framework.target_evidence import (
    iter_materialized_records, materialize_record,
)
from framework.program_runner import (
    backend_runner as _backend_runner,
    identity_binary_path,
)
from framework.supply.model_candidate_supply import (
    _candidate_bucket, _model_candidate_limit, _model_candidate_record,
    _model_candidate_view, request_case_rewrite,
)
_identity_binary_path = identity_binary_path
from framework.program_route import target_probe_allowed as _target_probe_allowed
from framework.paths import LOCAL_ROOT
from framework.rvgen.provider import CaseProgram
from framework.rvemi.program import _register_index, program_sha256 as canonical_program_sha256


def _campaign_disk_record(result: Mapping[str, object]) -> object:
    """Return a JSON-safe view while reusing already-safe trace containers."""
    record = dict(result)
    chain = record.get("chain")
    chain_run = chain.get("run") if isinstance(chain, Mapping) else None
    if isinstance(chain, Mapping) and not (
        isinstance(chain_run, Mapping)
        and type(chain_run.get("mh_attempts")) is int
        and type(chain_run.get("sample_count")) is int
    ):
        record["chain"] = _compact_chain_record(chain)
    return _campaign_json_safe(record)


def _materialize_campaign_target_evidence(
    root: Path, campaign: Mapping[str, object],
) -> dict[str, object]:
    """Hydrate compact Target records only for an inline coverage collector."""
    result = dict(campaign)
    records = result.get("target")
    if isinstance(records, (list, tuple)):
        result["target"] = list(iter_materialized_records(root, records))
    baselines = result.get("target_baselines_by_target")
    if isinstance(baselines, Mapping):
        result["target_baselines_by_target"] = {
            target_id: materialize_record(root, record)
            if isinstance(record, Mapping) else record
            for target_id, record in baselines.items()
        }
    baseline = result.get("target_baseline")
    if isinstance(baseline, Mapping):
        result["target_baseline"] = materialize_record(root, baseline)
    records_by_target = result.get("target_records_by_target")
    if isinstance(records_by_target, Mapping):
        result["target_records_by_target"] = {
            target_id: list(iter_materialized_records(root, records))
            if isinstance(records, (list, tuple)) else records
            for target_id, records in records_by_target.items()
        }
    return result


def _campaign_json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else json_safe(value)
    if isinstance(value, (Path, bytes, bytearray)):
        return json_safe(value)
    if isinstance(value, Mapping):
        converted: dict[str, object] | None = None
        for key, item in value.items():
            safe_key = key if isinstance(key, str) else str(key)
            safe_item = _campaign_json_safe(item)
            if safe_key != key or safe_item is not item:
                if converted is None:
                    converted = dict(value)
                if safe_key != key:
                    converted.pop(key, None)
                converted[safe_key] = safe_item
            elif converted is not None:
                converted[safe_key] = safe_item
        return value if converted is None else converted
    if isinstance(value, list):
        converted: list[object] | None = None
        for index, item in enumerate(value):
            safe_item = _campaign_json_safe(item)
            if converted is None and safe_item is not item:
                converted = list(value[:index])
                converted.append(safe_item)
            elif converted is not None:
                converted.append(safe_item)
        return value if converted is None else converted
    if isinstance(value, (tuple, set, frozenset)):
        return [_campaign_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _campaign_json_safe(value.to_dict())
    if hasattr(value, "to_record"):
        return _campaign_json_safe(value.to_record())
    return json_safe(value)


def _atomic_write_campaign_result(path: Path, result: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _campaign_disk_record(result), handle,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _coverage_ledger(recipe: Mapping[str, object]) -> dict[str, object]:
    source_ref = recipe.get("source_ledger")
    if not isinstance(source_ref, (str, Path)):
        return dict(recipe)
    source_file = Path(source_ref)
    if not source_file.is_absolute() and not source_file.exists():
        source_file = Path(__file__).resolve().parents[2] / source_file
    source = read_json_object(source_file)
    source_records = source.get("route_records", ())
    recipe_records = recipe.get("route_records", ())
    if not isinstance(source_records, (list, tuple)) or not isinstance(recipe_records, (list, tuple)):
        raise ValueError("coverage ledger route_records must be lists")
    recipe_keys = {
        (item.get("root_key"), item.get("lineage_key"), _recipe_route_name(item))
        for item in recipe_records
        if isinstance(item, Mapping) and _recipe_route_name(item) is not None
    }
    merged = [
        item for item in source_records
        if not isinstance(item, Mapping)
        or (item.get("root_key"), item.get("lineage_key"), _recipe_route_name(item))
        not in recipe_keys
    ]
    return {**source, "route_records": [*merged, *recipe_records]}


def _materialized_source(
    recipe: Mapping[str, object], test: str, *, include_recipe_source: bool = True,
) -> Path | None:
    generation = recipe.get("generation")
    profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
    locators = (
        ((recipe.get("source_locator"),) if include_recipe_source else ())
        + ((profile.get("asm_test"),) if isinstance(profile, Mapping) else ())
    )
    for locator in locators:
        if not isinstance(locator, str) or not locator or "#" in locator:
            continue
        text = locator.replace("\\", "/")
        raw = Path(locator)
        roots = [raw] if raw.is_absolute() else [LOCAL_ROOT / raw, Path(__file__).resolve().parents[2] / raw]
        for prefix, base in (
            ("/work/migration_Emprical_Study.local/", LOCAL_ROOT),
            ("/work/migration_Emprical_Study/", Path(__file__).resolve().parents[2]),
        ):
            if text.startswith(prefix):
                roots.append(base / text.removeprefix(prefix))
        for root in dict.fromkeys(roots):
            if root.is_file() and root.suffix.lower() == ".s":
                return root.resolve()
            for directory in (root / "generation" / "asm_test", root / "asm_test", root):
                if not directory.is_dir():
                    continue
                preferred = directory / f"{test}_0.S"
                if preferred.is_file():
                    return preferred.resolve()
                sources = sorted((*directory.glob("*.S"), *directory.glob("*.s")))
                if len(sources) == 1:
                    return sources[0].resolve()
    return None


def run_program_recipe(
    ledger: Mapping[str, object],
    *,
    root_key: str,
    lineage_key: str,
    checkout: str | Path,
    output_dir: str | Path,
    method: str,
    rule: str,
    state: str,
    observer: str,
    steps: int = FRAMEWORK_CANONICAL_STEPS,
    seed: int = 1,
    beta: float = 1.0,
    mcmc: bool = True,
    target_probe: bool = False,
    target_backend: str | None = None,
    target_id: str | None = None,
    generation_only: bool = False,
    rewrite_hint: object | None = None,
    pyflow_pythonpath: str | Path | None = None,
    max_seconds: float | None = None,
    target_timeout_seconds: float | None = None,
    target_replay_budget_seconds: float | None = None,
    reference_path_reward_version: str = "path-v2",
    target_binary_path: str | Path | None = None,
    coverage_config: Mapping[str, object] | None = None,
    target_configs: Mapping[str, Mapping[str, object]] | None = None,
    small_model: str | None = None,
    small_model_enabled: bool | None = None,
    emi_enabled: bool = True,
    open_capability: bool = False,
    chain_checkpoint_path: str | Path | None = None,
    seed_case: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """按 recipe 完成 ``generate_program → run_program_campaign`` 交接。"""
    if small_model_enabled is None:
        small_model_enabled = mcmc
    if type(small_model_enabled) is not bool or type(emi_enabled) is not bool:
        raise ValueError("invalid framework module toggles")
    if not emi_enabled and mcmc:
        raise ValueError("EMI-off runs require MCMC to be off")
    target_queue_only_requested = os.environ.get("RQ1_TARGET_QUEUE_ONLY", "0") == "1"
    recipe = resolve_root_recipe(ledger, root_key, lineage_key, route="program")
    startup_gaps = _recipe_gaps(recipe)
    if startup_gaps:
        raise ValueError("program recipe startup validation failed: " + ",".join(startup_gaps))
    execution_gaps = _execution_recipe_gaps(recipe)
    generation = recipe["generation"]
    if not isinstance(generation, Mapping):
        raise ValueError("program recipe has no generation section")
    initial_state = generation.get("initial_state")
    if initial_state is not None and not isinstance(initial_state, Mapping):
        raise ValueError("generation.initial_state must be an object")
    generation_profile = generation.get("generation_profile")
    if not isinstance(generation_profile, Mapping) or not generation_profile:
        raise ValueError("program recipe has no generation profile")
    checkout = Path(checkout).expanduser().resolve()
    profile = _normalize_generation_profile(generation_profile)
    if profile.get("asm_test") not in (None, ""):
        asm_test = Path(str(profile["asm_test"])).expanduser()
        profile["asm_test"] = str(
            (checkout / asm_test if not asm_test.is_absolute() else asm_test).resolve()
        )
    profile.setdefault("iterations", 1)
    provider = str(generation.get("source") or "").strip().lower()
    if provider == "dossier-derived-custom-program":
        provider = "custom-program"
    target = recipe["target"]["backend"]
    reference = recipe["reference"]["backend"]
    reference_binary = recipe["reference"].get("binary_path")
    target_binary = recipe["target"].get("binary_path")
    isa = str(profile.get("isa") or generation.get("isa") or "").lower()
    target_name = profile.get("target") or generation.get("target") or (
        "rv32imc" if provider == "custom-program" and isa.startswith("rv32")
        else "rv64imc" if provider == "custom-program" else None
    )
    test_name = profile.get("test") or generation.get("test") or (
        "riscv_arithmetic_basic_test" if provider == "custom-program" else None
    )
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("program recipe generation profile has no target")
    if not isinstance(test_name, str) or not test_name.strip():
        raise ValueError("program recipe generation profile has no test")
    # ``path`` in the program ledger is evidence for the route, not an input
    # program.  A RISC-V-DV recipe must reach the generator unless the recipe
    # explicitly supplies ``generation_profile.asm_test``.  Treating the
    # evidence file as asm_test silently turns a full program route into the
    # three-instruction witness kept in the ledger.
    source = _materialized_source(
        recipe, test_name, include_recipe_source=provider == "custom-program",
    )
    if seed_case is not None:
        if provider not in {"riscv-dv", "riscv_dv"}:
            raise ValueError("pre-generated seed cases require the RISC-V-DV program provider")
        source_path = Path(str(seed_case.get("source_path", ""))).resolve(strict=True)
        expected_sha = str(seed_case.get("source_sha256", "")).lower()
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            source_path.suffix.lower() != ".s"
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or actual_sha != expected_sha
        ):
            raise ValueError("pre-generated RVDV seed source identity mismatch")
        profile["asm_test"] = str(source_path)
        profile["harness"] = "riscv-dv"
        artifact_isa = seed_case.get("artifact_isa")
        if isinstance(artifact_isa, str) and artifact_isa.strip():
            profile["isa"] = artifact_isa.strip()
    elif source is not None:
        profile["asm_test"] = str(source)
    output_dir = Path(output_dir).resolve()
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError("program campaign output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_deadline = (
        time.monotonic() + float(max_seconds)
        if max_seconds is not None else None
    )

    def remaining_budget() -> float | None:
        if campaign_deadline is None:
            return None
        return max(0.0, campaign_deadline - time.monotonic())

    def model_budget() -> float | None:
        remaining = remaining_budget()
        if remaining is None:
            return None
        try:
            configured = float(os.environ.get("RQ1_SMALL_MODEL_BUDGET", "120"))
        except (TypeError, ValueError):
            configured = 120.0
        return max(0.0, min(remaining * 0.5, configured))

    def build_linux_user_with_budget(source_path, output_path, params):
        remaining = remaining_budget()
        if remaining is None:
            return build_linux_user(source_path, output_path, params)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(["linux-user-build", str(source_path)], 0)
        if not isinstance(params, Mapping):
            return build_linux_user(source_path, output_path, params)
        try:
            configured = float(params.get("timeout", 120))
        except (TypeError, ValueError):
            return build_linux_user(source_path, output_path, params)
        if not math.isfinite(configured) or configured <= 0:
            return build_linux_user(source_path, output_path, params)
        return build_linux_user(
            source_path, output_path,
            {**dict(params), "timeout": min(configured, remaining)},
        )

    if isinstance(rewrite_hint, Mapping) and isinstance(rewrite_hint.get("hint"), Mapping):
        rewrite_hint = rewrite_hint["hint"]
    # Expected target status is an observation from an earlier run, not a
    # launch allowlist.  The current target process gets its own attempt.
    target_status_gap = False
    probe_gaps = _target_probe_allowed(target_probe, execution_gaps, recipe)
    blocking_execution_gaps = [] if probe_gaps else execution_gaps
    target_validation_ready = not target_status_gap
    binding = root_binding_for_recipe(
        recipe, method=method, rule=rule, state=state, observer=observer,
        target=target, evidence_path=str(output_dir / "candidate-ledger.json"),
        validate_target=not blocking_execution_gaps and target_validation_ready,
    )
    binding["module_toggles"] = {
        "small_model": small_model_enabled,
        "emi": emi_enabled,
        "mcmc": mcmc,
    }
    binding["path_required"] = bool(
        isinstance(coverage_config, Mapping)
        and coverage_config.get("enabled", True)
    )
    if seed_case is not None:
        binding["seed_case"] = dict(seed_case)
    # The recipe's original observer list often contains only the transport
    # witness (executed PCs and outcome).  Binding derives the state projection
    # from ``--state``; feed that projection back to the runner so FP/GPR
    # capture is requested exactly when the route declares it.
    runner_recipe = dict(recipe)
    recipe_observer = recipe.get("observer")
    if isinstance(recipe_observer, Mapping):
        runner_recipe["observer"] = {
            **dict(recipe_observer),
            "fields": list(binding.get(
                "observer_fields", recipe_observer.get("fields", ())
            )),
        }
    if probe_gaps:
        binding["target_execution_mode"] = "probe"
    if provider not in {"riscv-dv", "riscv_dv", "custom-program"}:
        return _generation_gap_result(
            work_dir=output_dir, root_binding=binding, status="generation-gap",
            reason=f"generation-gap: unsupported-program-provider:{provider or 'missing'}",
            seed=seed, beta=beta, mcmc=mcmc,
        )
    try:
        generated = generate_program(
            checkout, output_dir / "generation", target=target_name, test=test_name,
            seed=seed, pyflow_pythonpath=pyflow_pythonpath,
            generation_profile=profile, initial_state=initial_state,
            total_budget_sec=remaining_budget(), execute=subprocess.run,
        )
        try:
            manifest = read_json_object(output_dir / "generation" / "program-manifest.json")
        except (OSError, ValueError) as error:
            raise ValueError("generation-gap: generation manifest is missing or invalid") from error
        if manifest.get("steps") not in {"gen", "direct-copy"}:
            raise ValueError("generation-gap: unsupported generation steps")
        if (
            provider in {"riscv-dv", "riscv_dv"}
            and profile.get("asm_test") in (None, "")
            and manifest.get("steps") != "gen"
        ):
            raise ValueError(
                "generation-gap: riscv-dv program route did not run the generator"
            )
        manifest_profile = manifest.get("generation_profile")
        if not isinstance(manifest.get("resolved_options"), Mapping):
            raise ValueError("generation-gap: resolved generation options are missing")
        profile_digest = manifest.get("generation_profile_digest")
        if not isinstance(manifest_profile, Mapping):
            raise ValueError("generation-gap: generated profile is missing")
        if (manifest.get("status") != "generated" or not isinstance(profile_digest, str)
                or profile_digest != canonical_digest(dict(manifest_profile))):
            raise ValueError("generation-gap: generation manifest identity is invalid")
        for name in ("target", "test", "isa", "mabi"):
            expected = profile.get(name) or generation.get(name)
            actual = manifest_profile.get(name, manifest.get(name))
            if expected not in (None, "") and actual != expected:
                raise ValueError(f"generation-gap: generated profile {name} does not match recipe")
        if (
            not isinstance(manifest.get("program"), str)
            or Path(manifest["program"]).resolve() != generated.program.resolve()
            or manifest.get("program_sha256") != canonical_program_sha256(generated)
        ):
            raise ValueError("generation-gap: generated program identity does not match manifest")
        binding["generation_profile_digest"] = profile_digest
        identity = manifest.get("generation_identity")
        identity_digest = manifest.get("generation_identity_digest")
        if (not isinstance(identity, Mapping) or not isinstance(identity_digest, str)
                or identity_digest != canonical_digest(dict(identity))):
            raise ValueError("generation-gap: generation identity is invalid")
        expected_identity = generation.get("generation_identity_digest")
        expected_identity_record = generation.get("generation_identity")
        derived_identity = (
            isinstance(expected_identity_record, Mapping)
            and expected_identity_record.get("contract") == "derived-program-route-identity-v1"
        )
        if expected_identity not in (None, "", "runtime"):
            if not derived_identity and identity_digest != expected_identity:
                raise ValueError("generation-gap: generation identity does not match recipe")
            if derived_identity and (
                manifest.get("steps") != "direct-copy"
                or expected_identity_record.get("source_root") not in (None, root_key)
                or expected_identity_record.get("initial_state_digest") not in (None, identity.get("initial_state_digest"))
            ):
                raise ValueError("generation-gap: derived generation identity does not match recipe")
        expected_initial_state = (
            canonical_digest(dict(initial_state)) if initial_state is not None else None
        )
        if identity.get("initial_state_digest") != expected_initial_state:
            raise ValueError("generation-gap: generation identity initial state mismatch")
        binding["generation_identity_digest"] = identity_digest
    except Exception as error:
        manifest_status = None
        with suppress(OSError, ValueError):
            manifest_status = read_json_object(
                output_dir / "generation" / "program-manifest.json"
            ).get("status")
        status = manifest_status if manifest_status in {"generation-gap", "transport-gap"} else (
            "transport-gap"
            if isinstance(error, OSError) or str(error).startswith("transport-gap:")
            else "generation-gap"
        )
        return _generation_gap_result(
            work_dir=output_dir, root_binding=binding, status=status,
            reason=str(error), seed=seed, beta=beta, mcmc=mcmc,
        )
    run_params = dict(generated.run_params)
    if provider in {"riscv-dv", "riscv_dv"}:
        run_params.setdefault("text_address", "0x80001000")
    runtime_initial_state = dict(initial_state) if isinstance(initial_state, Mapping) else None
    if provider == "custom-program" and runtime_initial_state is not None:
        runtime_initial_state = {
            key: value for key, value in runtime_initial_state.items()
            if key in {"entry", "privilege", "registers"}
            or _register_index(str(key).lower()) is not None
        }
    if (
        runtime_initial_state
        and runtime_initial_state.get("privilege") == "machine"
    ):
        normalize_privilege = provider in {"riscv-dv", "riscv_dv"}
        if provider == "custom-program" and profile.get("asm_test") not in (None, ""):
            normalize_privilege = not re.search(
                r"(?mi)^\s*(?:mret|sret|wfi|sfence|hfence)(?=\s|$)"
                r"|^\s*csr[a-z.]*\s+(?:[^,\r\n]*,\s*)?"
                r"(?:0x[0-9a-f]+|m[a-z][a-z0-9]*|s(?![0-9p])[a-z][a-z0-9]*|h[a-z][a-z0-9]*)\b",
                Path(str(profile["asm_test"])).read_text(encoding="utf-8"),
            )
        if normalize_privilege:
            runtime_initial_state.pop("privilege")
    if provider == "custom-program":
        run_params["harness"] = "riscv-dv" if profile.get("harness") == "riscv-dv" else "custom"
    if runtime_initial_state is not None:
        run_params["initial_state"] = runtime_initial_state
    if isinstance(coverage_config, Mapping):
        run_params["framework_coverage"] = dict(coverage_config)
    if run_params != generated.run_params:
        generated = CaseProgram(
            generated.program,
            run_params,
        )
    if provider in {"riscv-dv", "riscv_dv"}:
        # RISC-V-DV can report a successful generation while its emitted
        # numeric-local-label graph is not assemblable.  Validate the raw
        # program before asking the model for a rewrite so this failure stays
        # in the generation stage and cannot be misreported as transport or
        # profile failure after a candidate is selected.
        raw_preflight_dir = output_dir / "generation" / "raw-preflight"
        raw_preflight_path = raw_preflight_dir / "program.elf"
        try:
            build_linux_user_with_budget(generated.program, raw_preflight_path, run_params)
        except Exception as error:
            timed_out = (
                isinstance(error, subprocess.TimeoutExpired)
                and remaining_budget() == 0
            )
            return _generation_gap_result(
                work_dir=output_dir, root_binding=binding,
                status="transport-gap" if timed_out else "generation-gap",
                reason=(
                    f"campaign-wall-clock-exhausted:raw-program-build:{error}"
                    if timed_out else f"generation-gap:raw-program-build:{error}"
                ),
                seed=seed, beta=beta, mcmc=mcmc,
            )
        atomic_write_json(output_dir / "generation" / "raw-preflight.json", {
            "status": "passed",
            "program": str(Path(generated.program).resolve()),
            "program_sha256": canonical_program_sha256(generated),
            "executable": str(raw_preflight_path.resolve()),
            "run_params_digest": canonical_digest(run_params),
        })
    reference_runner = _backend_runner(
        reference, reference_binary, recipe.get("reference"), runner_recipe,
        # Framework source coverage belongs to the observation Target.  The
        # K1/QEMU reference path must not inherit the Target coverage config.
        coverage_enabled=False,
    )

    def campaign_reference_runner(artifact, *, timeout_seconds=None):
        remaining = remaining_budget()
        if timeout_seconds is not None:
            remaining = (
                float(timeout_seconds) if remaining is None
                else min(remaining, float(timeout_seconds))
            )
        return reference_runner(artifact, timeout_seconds=remaining)

    # The probe is the first reference request of this campaign.  Reuse the
    # exact runner instance so a K1 transport failure pins the QEMU fallback
    # for the subsequent MCMC chain instead of silently retrying K1.
    reference_probe_runner = reference_runner
    reference_fallback_evidence = None
    if target_backend:
        target = target_backend
        target_binary = target_binary_path or recipe.get("target", {}).get("binary_path")
        target_status_gap = False
        blocking_execution_gaps = []
        target_validation_ready = True
        # root recipe 的默认 target 可能是 qemu；framework-run 允许声明
        # 其它兼容的 Linux-user Target，binding 必须同步实际 Target 后端。
        binding["target"] = target
        binding["target_id"] = str(
            target_id or binding.get("target_id") or target
        )
        binding["target_backend_override"] = True
    # The target is the measurement endpoint.  Reference/ISA capability
    # gaps are recorded in the case ledger, but they must not suppress a
    # target process that can still consume the generated artifact.  A
    # target status gap remains the only static reason to omit the runner.
    target_runner = None if target_queue_only_requested or generation_only or target_configs or (
        not open_capability and not (target_validation_ready or probe_gaps)
    ) else _backend_runner(
        target, target_binary, recipe.get("target"), runner_recipe
    )
    candidate_pool = None
    model = str(
        small_model or os.environ.get("RQ1_SMALL_MODEL", "qwen3.8:latest")
    ).strip()
    model_request_mode = os.environ.get(
        "RQ1_SMALL_MODEL_REQUEST_MODE", "ollama",
    ).strip().lower()
    model_request_name = (
        os.environ.get("RQ1_SMALL_MODEL_OPENAI_MODEL", "").strip()
        if model_request_mode == "openai-compatible" else
        model.removeprefix("ollama/").removeprefix("openai-compatible/")
    )
    if small_model_enabled and model:
        probe_path = output_dir / "generation" / "reference-witness-probe.json"
        probe_observations = []

        def capture_probe_observations(artifact, **kwargs):
            observations = _observations(reference_probe_runner(artifact, **kwargs))
            probe_observations.extend(observations)
            return observations

        _executed_lines, probe_evidence = reference_witness_source_lines(
            generated,
            build_linux_user_with_budget,
            capture_probe_observations,
            work_dir=output_dir / "generation" / "reference-witness-probe",
            timeout_seconds=remaining_budget(),
        )
        source_locations = (
            executed_source_locations(
                str(probe_evidence["executable"]), generated, probe_observations,
                timeout_seconds=remaining_budget(),
            )
            if probe_evidence.get("status") == "recorded"
            and isinstance(probe_evidence.get("executable"), str)
            else None
        )
        probe_fallback = probe_evidence.get("reference_fallback")
        if isinstance(probe_fallback, Mapping):
            reference_fallback_evidence = dict(probe_fallback)
        if probe_evidence.get("status") == "recorded" and source_locations:
            # The reference path is known before optional legacy model work.  Restrict
            # fragment expansion to those source locations so unreachable code
            # never consumes the model budget.
            candidates = enumerate_case_rewrites(
                generated, source_location_filter=source_locations,
            )
            model_candidates = witness_safe_case_rewrites_for_model(candidates)
            candidate_scope = "reference-path"
            source_root = Path(generated.program).resolve().parent
            probe_evidence["executed_source_locations"] = [
                {
                    "source_path": (
                        Path(source_file).relative_to(source_root).as_posix()
                        if Path(source_file).is_relative_to(source_root)
                        else source_file
                    ),
                    "line": line,
                }
                for source_file, line in sorted(source_locations)
            ]
            probe_evidence["source_location_status"] = "recorded"
        else:
            # A failed reference probe expands over the full source; the model
            # still receives its bounded semantic candidate view below.
            candidates = enumerate_case_rewrites(
                generated,
            )
            model_candidates = witness_safe_case_rewrites_for_model(candidates)
            candidate_scope = "full-source"
            probe_evidence["source_location_status"] = "gap"
        path_pool_digest = canonical_digest(list(candidates))
        supplied_digest = (
            rewrite_hint.get("provenance", {}).get("candidate_pool_digest")
            if isinstance(rewrite_hint, Mapping)
            and isinstance(rewrite_hint.get("provenance"), Mapping)
            else None
        )
        if rewrite_hint is None or supplied_digest == path_pool_digest:
            candidate_pool = tuple(candidates)
        model_limit = _model_candidate_limit()
        model_view = _model_candidate_view(model_candidates, model_limit)
        model_view_records = [
            _model_candidate_record(candidate) for candidate in model_view
        ]
        model_view_count = len(model_view_records)
        model_bucket_counts = {}
        for candidate in model_candidates:
            bucket = _candidate_bucket(candidate)
            if bucket is not None:
                name = f"{bucket[0]}:{bucket[1]}"
                model_bucket_counts[name] = model_bucket_counts.get(name, 0) + 1
        probe_evidence.update({
            "candidate_count_after_probe": len(candidates),
            "model_candidate_pool_count": len(model_candidates),
            "model_candidate_count": model_view_count,
            "model_candidate_limit": model_limit,
            "model_candidate_bucket_counts": model_bucket_counts,
            "model_candidate_view": model_view_records,
            "candidate_pool_digest": path_pool_digest,
            "candidate_pool_scope": candidate_scope,
        })
        atomic_write_json(probe_path, probe_evidence)
        binding["model_witness_probe"] = str(probe_path.resolve())
        if rewrite_hint is None:
            try:
                model_response = request_case_rewrite(
                    model_candidates, model=model,
                    timeout_seconds=model_budget(),
                    cache_key=canonical_digest({
                        "program_sha256": canonical_program_sha256(generated),
                        "candidate_pool_digest": path_pool_digest,
                        "candidate_scope": candidate_scope,
                    }),
                    cache_dir=os.environ.get(
                        "RQ1_SMALL_MODEL_CACHE_DIR", "/tmp/rq1-small-model-cache-v1",
                    ),
                )
            except (OSError, RuntimeError, ValueError) as error:
                # The model is an optional rewrite selector.  Preserve the
                # named failure in the probe artifact and let the campaign
                # ledger record a normal supply gap instead of aborting the
                # whole matrix worker.
                probe_evidence["model_request"] = {
                    "status": "gap", "reason": str(error),
                    "request_mode": model_request_mode,
                    "model": model_request_name,
                }
                atomic_write_json(probe_path, probe_evidence)
                model_response = None
            # ``None`` means the selector did not produce a usable ID.  The
            # model is an optional proposal source: keep the trusted raw
            # program and let the normal bounded MCMC route continue.  This
            # preserves target evidence without turning model capacity or
            # transport into a generation gate.
            if model_response is not None:
                rewrite_hint = model_response
                probe_evidence["model_request"] = {
                    "status": "selected", "response": model_response,
                    "request_mode": model_request_mode,
                    "model": model_request_name,
                }
            else:
                binding["model_request_status"] = "gap"
                if not isinstance(probe_evidence.get("model_request"), Mapping):
                    probe_evidence["model_request"] = {
                        "status": "gap", "reason": "no-usable-candidate-id",
                        "request_mode": model_request_mode,
                        "model": model_request_name,
                    }
            atomic_write_json(probe_path, probe_evidence)
    # Materialize the capsule projection for every Program-Full case, even
    # when the observation Target is a Linux-user simulator.  The source is the
    # portable boundary; Linux-user and capsule ELFs need different harnesses
    # and cannot be converted safely with objcopy alone.
    def build_bare_metal(
        source_path: str | Path,
        output_path: str | Path,
        params: Mapping[str, object],
    ) -> Path:
        bare_params = {
            **dict(params),
            "target_backend": (
                target if target in {
                    "unicorn-riscv64", "renode-riscv64",
                    "rax-riscv64", "rvvm-riscv64",
                } else "rvvm-riscv64"
            ),
            # All capsule adapters consume the same RVOBS1 mailbox/tohost
            # contract.  Keep the first LOAD segment page-aligned so RVVM
            # does not inherit the overlapping 0x7ffff000 layout.
            "transport": "tohost",
            "text_address": "0x80000000",
            "text_segment": "0x80000000",
            "bare_metal": True,
        }
        if target == "rax-riscv64":
            # RAX's loader uses the established Linux-user-style entry frame
            # while still consuming the bare tohost/mailbox projection.
            bare_params["bare_metal"] = False
        return build_linux_user_with_budget(source_path, output_path, bare_params)

    bare_builder = build_bare_metal
    target_replay_runners = {}
    target_bindings = {}
    target_job_configs = {}
    if target_configs and not generation_only:
        for target_id, target_config in target_configs.items():
            backend = target_config.get("backend")
            binary_path = target_config.get("binary_path")
            identity_section = target_config.get("identity_section")
            target_coverage = target_config.get("coverage_config")
            target_timeout = target_config.get("target_timeout_seconds")
            if (
                not isinstance(target_id, str) or not target_id
                or not isinstance(backend, str) or not backend
                or not isinstance(identity_section, Mapping)
            ):
                raise ValueError("target matrix config is invalid")
            target_recipe = {
                **runner_recipe,
                "target": {
                    **dict(recipe.get("target", {})),
                    **dict(identity_section),
                    "id": target_id,
                    "backend": backend,
                },
            }
            backend_target_runner = None
            if not target_queue_only_requested:
                backend_target_runner = _backend_runner(
                    backend, binary_path, identity_section, target_recipe,
                    coverage_config=(
                        target_coverage if isinstance(target_coverage, Mapping) else None
                    ),
                )

                def run_target(
                    artifact, *, timeout_seconds=None,
                    campaign_deadline_monotonic=None,
                    _runner=backend_target_runner, _target_timeout=target_timeout,
                ):
                    budget = _target_timeout
                    if timeout_seconds is not None:
                        budget = float(timeout_seconds) if budget is None else min(
                            float(budget), float(timeout_seconds),
                        )
                    if campaign_deadline_monotonic is not None:
                        remaining = max(
                            0.0, campaign_deadline_monotonic - time.monotonic(),
                        )
                        budget = remaining if budget is None else min(float(budget), remaining)
                    return _runner(artifact, timeout_seconds=budget)

                target_replay_runners[target_id] = run_target
            target_binding = {
                **binding,
                "target": backend,
                "target_id": target_id,
                "target_identity_resolution": "runtime",
                **({"target_backend_override": True}
                   if backend != recipe.get("target", {}).get("backend") else {}),
            }
            runtime_identity = None
            if (
                not target_queue_only_requested
                and
                isinstance(target_coverage, Mapping)
                and target_coverage.get("enabled", True)
                and isinstance(target_coverage.get("binary_path"), str)
                and isinstance(target_coverage.get("identity_binary_path"), str)
            ):
                try:
                    runtime_identity = coverage_observation_identity(target_coverage)
                except (OSError, TypeError, ValueError):
                    pass
            for name in ("target_identity_digest", "target_binary_sha256"):
                target_binding.pop(name, None)
                value = (
                    runtime_identity.identity_digest
                    if runtime_identity is not None and name == "target_identity_digest"
                    else runtime_identity.binary_sha256
                    if runtime_identity is not None
                    else target_config.get(
                        "identity_digest" if name == "target_identity_digest"
                        else "binary_sha256"
                    )
                )
                if isinstance(value, str):
                    target_binding[name] = value
            target_bindings[target_id] = target_binding
            if not target_queue_only_requested:
                target_job_configs[target_id] = {
                    "backend": backend,
                    "binary_path": binary_path,
                    "identity_section": dict(identity_section),
                    "session_command": target_config.get("session_command"),
                    "coverage_config": (
                        dict(target_coverage) if isinstance(target_coverage, Mapping) else None
                    ),
                    "target_timeout_seconds": target_timeout,
                    "target_binding": dict(target_binding),
                    "target_recipe": target_recipe,
                }

    if target_runner is not None and not target_replay_runners:
        selected_target_id = str(
            binding.get("target_id") or target_id or target or "target"
        )
        selected_binary = (
            str(target_binary_path) if target_binary_path is not None
            else str(target_binary) if target_binary is not None else None
        )
        identity_section = {
            "backend": target,
            "binary_path": selected_binary,
            "identity_resolution": "runtime",
            "expected": {
                "status": "normal",
                "backend": target,
            },
        }
        target_recipe = {
            **runner_recipe,
            "target": {
                **dict(binding),
                **identity_section,
                "id": selected_target_id,
                "backend": target,
            },
        }
        target_job_configs[selected_target_id] = {
            "backend": target,
            "binary_path": selected_binary,
            "identity_section": identity_section,
            "session_command": (
                coverage_config.get("session_command")
                if isinstance(coverage_config, Mapping) else None
            ),
            "coverage_config": (
                dict(coverage_config) if isinstance(coverage_config, Mapping) else None
            ),
            "target_timeout_seconds": target_timeout_seconds,
            "target_binding": {
                **binding,
                "target": target,
                "target_id": selected_target_id,
            },
            "target_recipe": target_recipe,
        }
    return run_program_campaign(
        generated, build_linux_user_with_budget, campaign_reference_runner, target_runner,
        work_dir=output_dir, steps=steps, seed=seed, beta=beta, mcmc=mcmc,
        rewrite_hint=rewrite_hint, root_binding=binding,
        # Generation, witness probing, and model selection share the same
        # case budget; do not restart the wall clock at the MCMC boundary.
        max_seconds=remaining_budget(),
        target_timeout_seconds=target_timeout_seconds,
        target_replay_budget_seconds=target_replay_budget_seconds,
        reference_path_reward_version=reference_path_reward_version,
        bare_builder=bare_builder, candidate_pool=candidate_pool,
        target_replay_runners=target_replay_runners,
        target_bindings=target_bindings,
        target_job_configs=target_job_configs,
        target_ids=(
            tuple(target_configs)
            if target_configs and target_queue_only_requested else
            (str(target_id or target_backend),)
            if target_queue_only_requested and target_backend else None
        ),
        open_capability=open_capability,
        emi_enabled=emi_enabled,
        reference_fallback_evidence=reference_fallback_evidence,
        chain_checkpoint_path=chain_checkpoint_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one root-bound program recipe")
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--root-key", required=True)
    parser.add_argument("--lineage-key", required=True)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--observer", required=True)
    parser.add_argument("--steps", type=int, default=FRAMEWORK_CANONICAL_STEPS)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--target-timeout-seconds", type=float, default=None)
    parser.add_argument("--target-replay-budget-seconds", type=float, default=None)
    parser.add_argument(
        "--reference-path-reward-version", choices=("pc-edge-v1", "path-v2"),
        default="path-v2",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no-mcmc", action="store_true")
    parser.add_argument("--target-probe", action="store_true")
    parser.add_argument("--target-backend")
    parser.add_argument("--target-id")
    parser.add_argument("--target-binary", type=Path)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--rewrite-hint", type=Path)
    parser.add_argument("--pyflow-pythonpath", type=Path)
    parser.add_argument("--coverage-config", type=Path)
    parser.add_argument("--target-configs", type=Path)
    parser.add_argument("--small-model")
    small_model_group = parser.add_mutually_exclusive_group()
    small_model_group.add_argument(
        "--use-small-model", dest="small_model_enabled", action="store_true",
    )
    small_model_group.add_argument(
        "--no-small-model", dest="small_model_enabled", action="store_false",
    )
    parser.set_defaults(small_model_enabled=None)
    parser.add_argument("--no-emi", action="store_true")
    parser.add_argument("--open-capability", action="store_true")
    parser.add_argument("--chain-checkpoint", type=Path)
    parser.add_argument("--seed-source", type=Path)
    parser.add_argument("--seed-source-sha256")
    parser.add_argument("--seed-case-id")
    parser.add_argument("--seed-case-number", type=int)
    parser.add_argument("--seed-sequence-index", type=int)
    parser.add_argument("--seed-catalog-id")
    parser.add_argument("--seed-catalog-revision", type=int)
    parser.add_argument("--seed-catalog-manifest-sha256")
    parser.add_argument("--seed-artifact-isa")
    args = parser.parse_args(argv)
    try:
        seed_fields = (
            args.seed_source_sha256, args.seed_case_id, args.seed_case_number,
            args.seed_sequence_index, args.seed_catalog_id,
            args.seed_catalog_revision, args.seed_catalog_manifest_sha256,
        )
        if args.seed_source is None:
            if any(value is not None for value in seed_fields) or args.seed_artifact_isa:
                raise ValueError("seed case metadata requires --seed-source")
            seed_case = None
        else:
            if any(value is None for value in seed_fields):
                raise ValueError("pre-generated seed case metadata is incomplete")
            if (
                not args.seed_case_id
                or not args.seed_catalog_id
                or args.seed_case_number < 1
                or args.seed_sequence_index < 0
                or args.seed_catalog_revision < 1
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(args.seed_catalog_manifest_sha256).lower(),
                ) is None
            ):
                raise ValueError("pre-generated seed case identity is invalid")
            seed_case = {
                "source_path": str(args.seed_source),
                "source_sha256": args.seed_source_sha256,
                "case_id": args.seed_case_id,
                "case_number": args.seed_case_number,
                "sequence_index": args.seed_sequence_index,
                "catalog_id": args.seed_catalog_id,
                "catalog_revision": args.seed_catalog_revision,
                "catalog_manifest_sha256": args.seed_catalog_manifest_sha256.lower(),
                "artifact_isa": args.seed_artifact_isa,
            }
        coverage_output = (args.output / "target-coverage.json").resolve()
        input_paths = {args.ledger.resolve()}
        if args.rewrite_hint is not None:
            input_paths.add(args.rewrite_hint.resolve())
        generated_paths = {
            (args.output / name).resolve()
            for name in (
                "candidate-ledger.json",
                "campaign-result.json", "generation/program-manifest.json",
            )
        }
        if coverage_output in input_paths | generated_paths:
            raise ValueError("coverage output path collides with an input or campaign artifact")
        rewrite_hint = read_json_object(args.rewrite_hint) if args.rewrite_hint else None
        if isinstance(rewrite_hint, Mapping) and isinstance(rewrite_hint.get("hint"), Mapping):
            rewrite_hint = rewrite_hint["hint"]
        coverage_config = (
            read_json_object(args.coverage_config) if args.coverage_config else None
        )
        target_config_document = (
            read_json_object(args.target_configs) if args.target_configs else None
        )
        target_configs = (
            target_config_document.get("targets")
            if isinstance(target_config_document, Mapping) else None
        )
        if target_configs is not None and not isinstance(target_configs, Mapping):
            raise ValueError("target configs file must contain a targets object")
        result = run_program_recipe(
            read_json_object(args.ledger), root_key=args.root_key,
            lineage_key=args.lineage_key, checkout=args.checkout,
            output_dir=args.output, method=args.method, rule=args.rule,
            state=args.state, observer=args.observer, steps=args.steps,
            seed=args.seed, beta=args.beta, mcmc=not args.no_mcmc,
            target_probe=args.target_probe,
            target_backend=args.target_backend,
            target_id=args.target_id,
            generation_only=args.generation_only,
            rewrite_hint=rewrite_hint,
            pyflow_pythonpath=args.pyflow_pythonpath,
            max_seconds=args.max_seconds,
            target_timeout_seconds=args.target_timeout_seconds,
            target_replay_budget_seconds=args.target_replay_budget_seconds,
            reference_path_reward_version=args.reference_path_reward_version,
            target_binary_path=args.target_binary,
            coverage_config=coverage_config,
            target_configs=target_configs,
            small_model=args.small_model,
            small_model_enabled=args.small_model_enabled,
            emi_enabled=not args.no_emi,
            open_capability=args.open_capability,
            chain_checkpoint_path=(
                args.chain_checkpoint or os.environ.get("RQ1_CHAIN_CHECKPOINT_PATH")
            ),
            seed_case=seed_case,
        )
        # Persist the core online campaign before the potentially slow
        # coverage collector.  A watchdog interruption during finalization
        # must retain the completed generator/reference/Target/MCMC record as a
        # partial, auditable campaign instead of looking like a missing case.
        target_coverage_enabled = bool(target_configs) and any(
            isinstance(item, Mapping)
            and isinstance(item.get("coverage_config"), Mapping)
            and item["coverage_config"].get("enabled", True)
            for item in target_configs.values()
        )
        coverage_enabled = target_coverage_enabled or bool(
            isinstance(coverage_config, Mapping) and coverage_config.get("enabled", True)
        )
        raw_coverage_only = os.environ.get("RQ1_RAW_COVERAGE_ONLY") == "1"
        if raw_coverage_only:
            result["raw_coverage_only"] = True
            result["target_coverage"] = {
                "schema": "target-coverage-matrix-v1",
                "status": "deferred",
                "reason": "raw-coverage-only",
            }
        else:
            root_ledger = read_json_object(args.ledger)
            campaign_ledger = read_json_object(Path(result["ledger_path"]))
            coverage = summarize_target_coverage_by_target(
                _coverage_ledger(root_ledger), campaign_ledger,
                **({"evidence_root": args.output}
                   if os.environ.get("RQ1_COMPACT_TARGET_EVIDENCE") == "1"
                   else {}),
            )
            atomic_write_json(coverage_output, coverage)
            result["target_coverage"] = {
                "output": str(coverage_output.resolve()), "counts": coverage["counts"],
            }
        result["finalization"] = {
            "status": "complete" if raw_coverage_only else "pending",
            "coverage_enabled": coverage_enabled,
        }
        _atomic_write_campaign_result(args.output / "campaign-result.json", result)
        defer_finalization = not raw_coverage_only and coverage_enabled and (
            os.environ.get("RQ1_DEFER_FINALIZE") == "1" or bool(target_configs)
        )
        coverage_wait_timeout = False
        if not raw_coverage_only and coverage_enabled and not target_configs:
            # .NET source collection is a side task of target execution.  It
            # must finish before a single-target inline conversion. The shared
            # Target fanout is finalized by the outer case finalizer so this
            # child can return and the producer lane can continue.
            if isinstance(coverage_config, Mapping):
                coverage_wait_timeout = wait_for_dotnet_coverage(
                    coverage_config,
                ).get("wait_timeout") is True
        if raw_coverage_only:
            result["framework_coverage"] = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": "deferred" if coverage_enabled else "disabled",
                "artifact_complete": False,
                "reason": "raw-coverage-only" if coverage_enabled else "coverage-disabled",
            }
        elif coverage_enabled and not defer_finalization:
            if coverage_wait_timeout:
                result["framework_coverage"] = {
                    "schema_version": "rq1-framework-coverage-v1",
                    "status": "gap", "artifact_complete": False,
                    "reason": "dotnet-coverage-task-wait-timeout",
                }
            else:
                try:
                    coverage_campaign = (
                        _materialize_campaign_target_evidence(args.output, result)
                        if os.environ.get("RQ1_COMPACT_TARGET_EVIDENCE") == "1"
                        else result
                    )
                    result["framework_coverage"] = finalize_framework_coverage(
                        args.output, coverage_campaign, coverage_config,
                    )
                except (OSError, TypeError, ValueError, RuntimeError) as error:
                    result["framework_coverage"] = {
                        "schema_version": "rq1-framework-coverage-v1",
                        "status": "gap", "artifact_complete": False,
                        "reason": f"framework-coverage-error:{type(error).__name__}",
                    }
        elif coverage_enabled:
            result["framework_coverage"] = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": "deferred", "artifact_complete": False,
                "reason": "coverage-finalization-deferred",
            }
        else:
            result["framework_coverage"] = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": "disabled", "artifact_complete": False,
                "reason": "coverage-disabled",
        }
        result["finalization"] = {
            "status": "deferred" if defer_finalization else "complete",
            "coverage_enabled": coverage_enabled,
            "framework_coverage_status": (
                result.get("framework_coverage", {}).get("status")
                if isinstance(result.get("framework_coverage"), Mapping) else None
            ),
            "reason": (
                "raw-coverage-only" if raw_coverage_only and coverage_enabled else
                "coverage-finalization-deferred" if defer_finalization else None
            ),
        }
        _atomic_write_campaign_result(args.output / "campaign-result.json", result)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_program_recipe"]

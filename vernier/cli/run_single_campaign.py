"""按 single recipe 运行 RVGEN TestCase → RVEMI/MH → 可选 ELF → target。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework._util import atomic_write_json, json_safe, read_json_object
from framework.step_policy import FRAMEWORK_CANONICAL_STEPS
from framework.adapters.contracts import is_execution_backend
from framework.execution_environment import load_target_binary_manifest
from framework.paths import LOCAL_ROOT
from framework.evidence.target_coverage import (
    _recipe_digest, _recipe_route_name,
    resolve_root_recipe, root_binding_for_recipe,
    summarize_target_coverage_by_target,
)
from framework.evidence.root_route_matrix import _execution_recipe_gaps
from framework.framework_coverage import finalize_framework_coverage, wait_for_dotnet_coverage
from framework.rvgen.single_campaign import (
    _catalog_root_case, _catalog_root_forms, _rewrite_hint, _testcase_state,
    run_single_campaign,
    testcase_from_recipe,
)
from framework.program_campaign import _generation_gap_result
from framework.spec_definedness import enabled_extensions


def _mabi_for_isa(isa: str) -> str:
    """Choose the freestanding ABI used by the selected TestCase profile."""
    profile = str(isa).lower()
    xlen = "ilp32" if profile.startswith("rv32") else "lp64"
    enabled = enabled_extensions(profile)
    if xlen == "ilp32" and "e" in enabled:
        return "ilp32e"
    if "q" in enabled or "d" in enabled:
        return f"{xlen}d"
    if "f" in enabled:
        return f"{xlen}f"
    return xlen


def _target_binary_path(target: Mapping[str, object]) -> str | None:
    path = target.get("binary_path")
    if isinstance(path, str) and path.strip():
        return path
    backend = target.get("backend")
    expected = target.get("expected")
    digest = target.get("identity_digest")
    if isinstance(expected, Mapping) and not isinstance(digest, str):
        digest = expected.get("identity_digest")
    if backend != "rvvm-riscv64" or not isinstance(digest, str) or not digest:
        return None
    manifest = LOCAL_ROOT / "target-store" / backend / digest / "target-identity.json"
    try:
        return str(load_target_binary_manifest(manifest, expected_backend=backend).binary_path)
    except Exception:
        return None


def _rv32_qemu_binary(path: str | None) -> str | None:
    """Resolve the 32-bit QEMU executable for an RV32 testcase."""
    candidates = []
    if isinstance(path, str) and path.strip():
        original = Path(path)
        if "qemu-riscv64" in original.name:
            candidates.append(
                original.with_name(original.name.replace("qemu-riscv64", "qemu-riscv32"))
            )
        elif "qemu-riscv32" in original.name:
            candidates.append(original)
    if (system := shutil.which("qemu-riscv32")):
        candidates.insert(0, Path(system))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def run_single_recipe(
    ledger: Mapping[str, object], *, root_key: str, lineage_key: str,
    output_dir: str | Path, method: str, rule: str, state: str | None, observer: str,
    steps: int = FRAMEWORK_CANONICAL_STEPS, seed: int = 1, beta: float = 1.0,
    mcmc: bool = True,
    rewrite_hint: object | None = None, target_enabled: bool = True,
    target_backend_override: str | None = None,
    target_id: str | None = None,
    target_binary: str | Path | None = None,
    # Direct opens the full frozen catalog by default; callers may explicitly
    # request the recipe-bound root with ``open_catalog=False``.
    open_catalog: bool = True,
    max_seconds: float | None = None,
    target_timeout_seconds: float | None = None,
    target_replay_budget_seconds: float | None = None,
    reference_path_reward_version: str = "path-v2",
    case_index: int = 0,
    coverage_config: Mapping[str, object] | None = None,
    target_configs: Mapping[str, Mapping[str, object]] | None = None,
    small_model: str | None = None,
    small_model_enabled: bool | None = None,
    emi_enabled: bool = True,
    open_capability: bool = True,
    chain_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    if small_model_enabled is not None and type(small_model_enabled) is not bool:
        raise ValueError("invalid small-model toggle")
    if type(emi_enabled) is not bool:
        raise ValueError("invalid EMI toggle")
    if not emi_enabled and mcmc:
        raise ValueError("EMI-off runs require MCMC to be off")
    recipe_gap = None
    try:
        recipe = resolve_root_recipe(ledger, root_key, lineage_key, route="single")
    except ValueError as error:
        if not str(error).startswith("root recipe is incomplete:"):
            raise
        routes = tuple(
            row for row in ledger.get("route_records", ())
            if isinstance(row, Mapping)
            and row.get("target_scope_included") is True
            and row.get("root_key") == root_key
            and row.get("lineage_key") == lineage_key
            and _recipe_route_name(row) == "single"
        )
        if len(routes) != 1:
            raise
        recipe, recipe_gap = dict(routes[0]), str(error)
    recipe_digest = _recipe_digest(recipe)
    output = Path(output_dir).resolve()
    if output.is_dir() and any(output.iterdir()):
        raise ValueError("single campaign output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)

    def finish(result):
        target_coverage_enabled = bool(target_configs) and any(
            isinstance(item, Mapping)
            and isinstance(item.get("coverage_config"), Mapping)
            and item["coverage_config"].get("enabled", True)
            for item in target_configs.values()
        )
        coverage_enabled = target_coverage_enabled or bool(
            isinstance(coverage_config, Mapping) and coverage_config.get("enabled", True)
        )
        if os.environ.get("RQ1_RAW_COVERAGE_ONLY") == "1":
            coverage_status = "deferred" if coverage_enabled else "disabled"
            result["raw_coverage_only"] = True
            result["target_coverage"] = {
                "schema": "target-coverage-matrix-v1",
                "status": "deferred",
                "reason": "raw-coverage-only",
            }
            result["framework_coverage"] = {
                "schema_version": "rq1-framework-coverage-v1",
                "status": coverage_status,
                "artifact_complete": False,
                "reason": (
                    "raw-coverage-only" if coverage_enabled else "coverage-disabled"
                ),
            }
            result["finalization"] = {
                "status": "complete",
                "coverage_enabled": coverage_enabled,
                "framework_coverage_status": coverage_status,
                "reason": "raw-coverage-only" if coverage_enabled else None,
            }
            atomic_write_json(output / "campaign-result.json", json_safe(result))
            return result
        coverage_output = output / "target-coverage.json"
        campaign_ledger = read_json_object(Path(result["ledger_path"]))
        coverage = summarize_target_coverage_by_target(ledger, campaign_ledger)
        result["target_coverage"] = {
            "output": str(coverage_output), "counts": coverage["counts"],
        }
        atomic_write_json(coverage_output, coverage)
        # 先保存在线链，再做可能较慢的源码覆盖率收尾；截止或采集失败时，
        # 调度器仍能区分“已执行但 coverage 未收尾”和“case 未落盘”。
        result["finalization"] = {
            "status": "pending",
            "coverage_enabled": coverage_enabled,
        }
        atomic_write_json(output / "campaign-result.json", json_safe(result))
        defer_finalization = coverage_enabled and (
            os.environ.get("RQ1_DEFER_FINALIZE") == "1" or bool(target_configs)
        )
        coverage_wait_timeout = False
        if coverage_enabled and not target_configs:
            # .NET source collection is a side task of target execution.  It
            # must finish before a single-target inline conversion. The shared
            # Target fanout is finalized by the outer case finalizer so this
            # child can return and the producer lane can continue.
            if isinstance(coverage_config, Mapping):
                coverage_wait_timeout = wait_for_dotnet_coverage(
                    coverage_config,
                ).get("wait_timeout") is True
        if coverage_enabled and not defer_finalization:
            if coverage_wait_timeout:
                result["framework_coverage"] = {
                    "schema_version": "rq1-framework-coverage-v1",
                    "status": "gap", "artifact_complete": False,
                    "reason": "dotnet-coverage-task-wait-timeout",
                }
            else:
                try:
                    result["framework_coverage"] = finalize_framework_coverage(
                        output, result, coverage_config,
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
            "reason": "coverage-finalization-deferred"
            if defer_finalization else None,
        }
        atomic_write_json(output / "campaign-result.json", json_safe(result))
        return result

    if recipe_gap is not None:
        generation = recipe.get("generation", {})
        source = generation.get("source") if isinstance(generation, Mapping) else None
        route_gap = recipe.get("route_gap")
        if isinstance(route_gap, Mapping) and isinstance(route_gap.get("reason"), str):
            reason = f"generation-gap:{route_gap['reason']}"
            missing = route_gap.get("missing_fields")
            if isinstance(missing, (list, tuple)) and missing:
                reason += ":" + ",".join(map(str, missing))
        elif not isinstance(source, str) or not source.startswith("framework/rvgen:"):
            reason = "generation-gap:single-recipe-requires-rvgen-form"
        else:
            reason = f"generation-gap:single-recipe-incomplete:{recipe_gap}"
        target_spec = recipe.get("target")
        target = (
            target_spec.get("backend")
            if isinstance(target_spec, Mapping) and isinstance(target_spec.get("backend"), str)
            else "unavailable"
        )
        binding = {
            "root_key": recipe.get("root_key", root_key),
            "lineage_key": recipe.get("lineage_key", lineage_key),
            "route": "single", "method": method, "rule": rule,
            "state": state or "generation", "observer": observer, "target": target,
            "root_recipe_digest": recipe_digest,
            "evidence_path": str(output / "candidate-ledger.json"),
        }
        if isinstance(route_gap, Mapping):
            binding["route_gap"] = dict(route_gap)
        for name, binding_name in (
            ("testcase_id", "generation_testcase_id"),
            ("base_program_id", "generation_base_program_id"),
            ("input_id", "generation_input_id"),
        ):
            value = generation.get(name) if isinstance(generation, Mapping) else None
            if isinstance(value, str) and value.strip():
                binding[binding_name] = value
        return finish(_generation_gap_result(
            work_dir=output, root_binding=binding, status="generation-gap",
            reason=reason, seed=seed, beta=beta, mcmc=mcmc,
        ))

    generation = recipe.get("generation", {})
    source = generation.get("source") if isinstance(generation, Mapping) else None
    reference = recipe.get("reference")
    if isinstance(source, str) and source.startswith("framework/rvgen:") \
            and isinstance(reference, Mapping):
        expected = reference.get("expected")
        if isinstance(expected, Mapping) and expected.get("identity_digest"):
            recipe = dict(recipe)
            reference = dict(reference)
            reference["expected"] = {
                key: value for key, value in expected.items()
                if key != "identity_digest"
            }
            reference["identity_resolution"] = "runtime"
            recipe["reference"] = reference

    target_spec = recipe["target"]
    target = target_spec["backend"]
    # 调用方声明的靶二进制优先于按镜像匹配的解析结果。
    target_binary_path = str(target_binary) if target_binary else _target_binary_path(target_spec)
    execution_gaps = _execution_recipe_gaps(recipe)
    # ISA/extension support is deliberately deferred to the target process.
    # Missing binaries and malformed route recipes remain execution gaps, but
    # a static backend allowlist must never prevent the case from starting.
    target_ready = target_enabled and is_execution_backend(target)
    if open_capability:
        # The target decides whether the generated instruction is supported.
        execution_gaps = []
    if target_backend_override:
        target = target_backend_override
        execution_gaps = []
        target_ready = target_enabled and is_execution_backend(target)
    target_jit = None
    if target == "rvvm-riscv64":
        path = target_spec.get("translation_path")
        if path is None:
            path = "jit" if "-jit" in str(root_key) else "interpreter"
        if path not in {"jit", "interpreter"}:
            raise ValueError("single target translation path is invalid")
        target_jit = path == "jit"
    binding_recipe = recipe
    if target_backend_override:
        binding_recipe = {
            **recipe,
            "target": {**dict(recipe["target"]), "backend": target_backend_override},
        }
    try:
        # The recipe keeps the route contract, while Direct samples the full
        # enabled RVGEN catalog.  The historical addi root remains a valid
        # fallback for old ledgers that cannot materialize the open catalog.
        generation_profile = recipe.get("generation")
        open_isa = (
            generation_profile.get("isa")
            if isinstance(generation_profile, Mapping) else None
        )
        catalog_mode = (
            open_catalog and method.startswith("S") and isinstance(open_isa, str)
        )
        testcase = _catalog_root_case(open_isa, case_index, root_seed=seed) \
            if catalog_mode else None
        if catalog_mode and testcase is None:
            catalog_forms = _catalog_root_forms(open_isa)
            requested_index = (
                (case_index + seed) % len(catalog_forms)
                if catalog_forms else None
            )
            requested_form = (
                catalog_forms[requested_index]
                if requested_index is not None else "unavailable"
            )
            binding = root_binding_for_recipe(
                binding_recipe, method=method, rule=rule, state=state or "generation",
                observer=observer, target=target,
                evidence_path=str(output / "candidate-ledger.json"),
                validate_target=target_ready,
            )
            binding.update({
                "root_case_index": case_index,
                "root_recipe_digest": recipe_digest,
                "catalog_size": len(catalog_forms),
                "catalog_requested_form_index": requested_index,
                "catalog_form_index": requested_index,
                "catalog_form": requested_form,
                "catalog_skipped_form_count": 0,
            })
            return finish(_generation_gap_result(
                work_dir=output, root_binding=binding, status="generation-gap",
                reason=f"generation-gap:catalog-form-unmaterialized:{requested_form}",
                seed=seed, beta=beta, mcmc=mcmc,
            ))
        if testcase is None:
            testcase = testcase_from_recipe(
                recipe, case_index=case_index, root_pool=method.startswith("S"),
                root_seed=seed,
            )
    except ValueError as error:
        reason = str(error)
        if not reason.startswith("generation-gap:"):
            raise
        binding = root_binding_for_recipe(
            binding_recipe, method=method, rule=rule, state=state or "generation",
            observer=observer, target=target,
            evidence_path=str(output / "candidate-ledger.json"),
            validate_target=target_ready,
        )
        if target_backend_override:
            binding["target_backend_override"] = True
        binding["root_recipe_digest"] = recipe_digest
        if target == "rvvm-riscv64":
            binding["target_translation_path"] = "jit" if target_jit else "interpreter"
        if target_enabled and execution_gaps:
            binding["target_execution_gap"] = execution_gaps[0]
        for name, binding_name in (
            ("testcase_id", "generation_testcase_id"),
            ("base_program_id", "generation_base_program_id"),
            ("input_id", "generation_input_id"),
        ):
            value = generation.get(name) if isinstance(generation, Mapping) else None
            if isinstance(value, str) and value.strip():
                binding[binding_name] = value
        return finish(_generation_gap_result(
            work_dir=output, root_binding=binding, status="generation-gap",
            reason=reason, seed=seed, beta=beta, mcmc=mcmc,
        ))
    # QEMU has separate Linux-user binaries for RV32 and RV64.  The open
    # catalog deliberately contains both XLENs, so bind the actual target
    # backend after RVGEN selects the TestCase instead of inheriting the
    # recipe's RV64 target for every root.  Other targets keep their declared
    # backend and report unsupported RV32 cases at execution time.
    testcase_isa = str(getattr(testcase, "isa_profile", "") or "").lower()
    if testcase_isa.startswith("rv32") and target in {"qemu", "qemu-riscv64"}:
        target = "qemu-riscv32"
        target_binary_path = _rv32_qemu_binary(target_binary_path)
        binding_recipe = {
            **recipe,
            "target": {**dict(recipe["target"]), "backend": target},
        }
    elif target_backend_override:
        binding_recipe = {
            **recipe,
            "target": {**dict(recipe["target"]), "backend": target_backend_override},
        }
    state = (
        _testcase_state(testcase)
        if method.startswith("S") else state or _testcase_state(testcase)
    )
    if method.startswith("S"):
        # ledger 的 root_key 是路线身份；Direct 的真实单指令身份必须来自本次 catalog testcase。
        root_form = testcase.generation_rule_id.partition(":")[0]
        testcase_isa = str(
            getattr(testcase, "isa_profile", None)
            or generation.get("isa")
            or "rv64i"
        )
        facts = testcase.dataflow_meta.get("realized_facts", {})
        expected_trap = (
            state == "trap"
            or isinstance(facts, Mapping)
            and facts.get("lane") == "legality-expected-trap"
        )
        dynamic_generation = {
            **dict(generation),
            "source": f"framework/rvgen:{root_form}",
            "direct": f"generate_form({root_form})",
            "isa": testcase_isa,
            "mabi": _mabi_for_isa(testcase_isa),
            "sequence": [root_form],
            "initial_state": {
                "entry": "_start",
                **({"expected": "trap"} if expected_trap else {}),
            },
        }
        dynamic_observer = dict(binding_recipe.get("observer", {}))
        control_fields = {
            "beq": ("branch_outcome", "after_pc", "control.target"),
            "bge": ("branch_outcome", "after_pc", "control.target"),
            "bgeu": ("branch_outcome", "after_pc", "control.target"),
            "blt": ("branch_outcome", "after_pc", "control.target"),
            "bltu": ("branch_outcome", "after_pc", "control.target"),
            "bne": ("branch_outcome", "after_pc", "control.target"),
            "jal": ("after_pc", "control.target", "link"),
            "jalr": ("after_pc", "control.target", "link"),
            "fence": ("after_pc",),
        }.get(root_form, ())
        if control_fields:
            existing_fields = dynamic_observer.get("fields", ())
            dynamic_observer["fields"] = list(dict.fromkeys(
                [*existing_fields, *control_fields]
                if isinstance(existing_fields, (list, tuple)) else control_fields
            ))
        if expected_trap:
            # The canonical recipe is a normal addi witness.  Once the root
            # pool selects a trap form, its observer must follow the selected
            # TestCase; retaining memory.test-memory makes the semantic trap
            # reference fail ASeed before any target is executed.
            dynamic_observer["fields"] = [
                "executed_pcs", "outcome", "trap.cause", "trap.epc", "trap.tval",
            ]
        binding_recipe = {
            **binding_recipe,
            "generation": dynamic_generation,
            "observer": dynamic_observer,
        }
    binding = root_binding_for_recipe(
        binding_recipe, method=method, rule=rule, state=state, observer=observer,
        target=target, evidence_path=str(output / "candidate-ledger.json"),
        validate_target=target_ready,
    )
    if target_backend_override:
        binding["target_backend_override"] = True
    binding["root_case_index"] = case_index
    binding["root_recipe_digest"] = recipe_digest
    if open_catalog and isinstance(open_isa, str):
        catalog_forms = _catalog_root_forms(open_isa)
        requested_index = (
            (case_index + seed) % len(catalog_forms) if catalog_forms else None
        )
        actual_form = testcase.generation_rule_id.partition(":")[0]
        actual_index = (
            catalog_forms.index(actual_form) if actual_form in catalog_forms else None
        )
        binding["catalog_size"] = len(catalog_forms)
        binding["catalog_requested_form_index"] = requested_index
        binding["catalog_form_index"] = actual_index
        binding["catalog_form"] = actual_form
        binding["catalog_skipped_form_count"] = (
            (actual_index - requested_index) % len(catalog_forms)
            if actual_index is not None and requested_index is not None else None
        )
    binding.update({
        "generation_testcase_id": testcase.testcase_id,
        "generation_base_program_id": testcase.base_program_id,
        "generation_input_id": testcase.input_id,
    })
    binding["path_required"] = bool(
        isinstance(coverage_config, Mapping) and coverage_config.get("enabled", True)
    )
    if target == "rvvm-riscv64":
        binding["target_translation_path"] = "jit" if target_jit else "interpreter"
    if target_enabled and execution_gaps:
        binding["target_execution_gap"] = execution_gaps[0]
    rewrite_hint = _rewrite_hint(rewrite_hint)
    result = run_single_campaign(
        testcase, work_dir=output, root_binding=binding,
        reference_backend=recipe["reference"]["backend"],
        target_backend=target if target_ready else None,
        target_id=target_id,
        target_binary_path=target_binary_path,
        rewrite_hint=rewrite_hint, steps=steps, seed=seed, beta=beta, mcmc=mcmc,
        max_seconds=max_seconds,
        target_timeout_seconds=target_timeout_seconds,
        target_replay_budget_seconds=target_replay_budget_seconds,
        reference_path_reward_version=reference_path_reward_version,
        target_jit=target_jit,
        coverage_config=coverage_config,
        target_configs=target_configs,
        small_model=small_model,
        small_model_enabled=small_model_enabled,
        emi_enabled=emi_enabled,
        open_capability=open_capability,
        chain_checkpoint_path=chain_checkpoint_path,
    )
    return finish(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one root-bound RVGEN single recipe")
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--root-key", required=True)
    parser.add_argument("--lineage-key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="S1")
    parser.add_argument("--rule", default="R3")
    parser.add_argument("--state")
    parser.add_argument("--observer", default="RVOBS1")
    parser.add_argument("--steps", type=int, default=FRAMEWORK_CANONICAL_STEPS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no-mcmc", action="store_true")
    parser.add_argument("--no-target", action="store_true")
    parser.add_argument("--target-backend")
    parser.add_argument("--target-id")
    parser.add_argument("--target-binary", type=Path)
    parser.add_argument("--open-catalog", action="store_true")
    parser.add_argument("--rewrite-hint", type=Path)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--target-timeout-seconds", type=float, default=None)
    parser.add_argument("--target-replay-budget-seconds", type=float, default=None)
    parser.add_argument(
        "--reference-path-reward-version", choices=("pc-edge-v1", "path-v2"),
        default="path-v2",
    )
    parser.add_argument("--case-index", type=int, default=0)
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
    parser.set_defaults(open_catalog=True, open_capability=True)
    args = parser.parse_args(argv)
    try:
        result = run_single_recipe(
            read_json_object(args.ledger), root_key=args.root_key,
            lineage_key=args.lineage_key, output_dir=args.output,
            method=args.method, rule=args.rule, state=args.state,
            observer=args.observer, steps=args.steps, seed=args.seed,
            beta=args.beta, mcmc=not args.no_mcmc,
            rewrite_hint=(read_json_object(args.rewrite_hint) if args.rewrite_hint else None),
            target_enabled=not args.no_target,
            target_backend_override=args.target_backend,
            target_id=args.target_id,
            target_binary=args.target_binary,
            open_catalog=args.open_catalog,
            max_seconds=args.max_seconds,
            target_timeout_seconds=args.target_timeout_seconds,
            target_replay_budget_seconds=args.target_replay_budget_seconds,
            reference_path_reward_version=args.reference_path_reward_version,
            case_index=args.case_index,
        coverage_config=(read_json_object(args.coverage_config)
                         if args.coverage_config else None),
        target_configs=(
            read_json_object(args.target_configs).get("targets")
            if args.target_configs else None
        ),
            small_model=args.small_model,
            small_model_enabled=args.small_model_enabled,
            emi_enabled=not args.no_emi,
            open_capability=args.open_capability,
            chain_checkpoint_path=(
                args.chain_checkpoint or os.environ.get("RQ1_CHAIN_CHECKPOINT_PATH")
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_single_recipe"]

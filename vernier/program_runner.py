"""Program 后端 runner 的唯一装配点。"""

from collections.abc import Callable, Mapping
from dataclasses import replace
import os
from pathlib import Path
import re

from framework._util import (
    _compare_mask_fields, canonical_digest, is_sha256_digest, json_safe,
    read_json_object, sha256_file,
)
from framework.adapters.contracts import CAPSULE_BACKENDS, canonical_execution_backend
from framework.adapters.runner import (
    CAPSULE_MAILBOX_ENV,
    CAPSULE_MEMORY_SIZE_ENV,
    CAPSULE_OBSERVATION_SIZE_ENV,
    CAPSULE_TEST_MEMORY_ENV,
)
from framework.direct_case import (
    OBSERVATION_HEADER_SIZE, CompareMask, TestCase, _runner_contract_gap_observation,
    expected_trap_from_dataflow,
)
from framework.direct_elf import symbol_offsets_from_elf
from framework.execution_environment import ExecutionEnvironmentError, load_target_binary_manifest
from framework.execution_identity import _target_binary_name
from framework.framework_coverage import run_backend_with_coverage
from framework.paths import LOCAL_ROOT, REPOSITORY_ROOT
from framework.reference_fallback import (
    K1_QEMU_FALLBACK_POLICY,
    configured_k1_qemu_fallback_classes,
)


def _artifact_executable(artifact: object, backend: str) -> Path:
    """Select the ELF matching the backend's execution model."""
    if backend in CAPSULE_BACKENDS:
        value = getattr(artifact, "bare_executable_path", None)
        if value is None:
            raise ExecutionEnvironmentError(
                f"bare-metal-artifact-missing:{backend}"
            )
    else:
        linux_status = getattr(artifact, "linux_materialization_status", "not-requested")
        value = getattr(artifact, "linux_executable_path", None)
        if value is None and linux_status != "gap":
            # Legacy BuiltProgram records predate the explicit projection
            # field and use executable_path for the Linux-user artifact.
            value = getattr(artifact, "executable_path", None)
    if value is None:
        raise ExecutionEnvironmentError(f"artifact-missing:{backend}")
    return Path(value)


def identity_binary_path(section: object, backend: str) -> str | None:
    if not isinstance(section, Mapping):
        return None
    explicit = section.get("binary_path")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    expected = section.get("expected")
    digest = section.get("identity_digest") or (
        expected.get("identity_digest") if isinstance(expected, Mapping) else None
    )
    backend = section.get("backend") if section.get("backend") in {
        "qemu-riscv32", "qemu-riscv64",
    } else canonical_execution_backend(backend)
    name = _target_binary_name(backend)
    if not isinstance(digest, str) or name is None:
        return None
    return str(LOCAL_ROOT / "target-store" / backend / digest / name)


def _attach_program_test_path(observations: object, executable: Path) -> tuple[object, ...]:
    rows = tuple(observations) if isinstance(observations, (list, tuple)) else ()
    _, addresses = symbol_offsets_from_elf(
        executable, ("rvgen_program_body_start", "rvgen_program_body_end"),
    )
    try:
        start = int(addresses["rvgen_program_body_start"], 16)
        end = int(addresses["rvgen_program_body_end"], 16)
    except (KeyError, TypeError, ValueError):
        return rows
    if not start < end:
        return rows
    result = []
    for observation in rows:
        extra = dict(getattr(observation, "extra_state", {}) or {})
        pcs = tuple(getattr(observation, "executed_pcs", ()) or ())
        body = tuple(pc for pc in pcs if type(pc) is int and start <= pc < end)
        extra["program_test_path"] = {
            "contract": "program-test-path-v1",
            "status": "observed" if body else "gap",
            "start_pc": start,
            "end_pc": end,
            "executed_pcs": list(body),
        }
        result.append(replace(observation, extra_state=extra))
    return tuple(result)


def _qemu_fallback_configuration(
    reference: Mapping[str, object],
) -> tuple[str, str, dict[str, object], str]:
    fallback = reference.get("fallback")
    if not isinstance(fallback, Mapping):
        raise ExecutionEnvironmentError("QEMU reference fallback is not configured")
    fallback_classes = configured_k1_qemu_fallback_classes(fallback.get("on"))
    backend = fallback.get("backend")
    target_id = fallback.get("target_id")
    if (
        backend != "qemu-riscv64"
        or fallback_classes is None
        or fallback.get("selection_scope") != "whole-campaign"
        or not isinstance(target_id, str)
        or not target_id.strip()
    ):
        raise ExecutionEnvironmentError("QEMU reference fallback policy is invalid")
    config_path = REPOSITORY_ROOT / "config" / "rq1-comparison-isolated-v1.json"
    config = read_json_object(config_path)
    targets = config.get("targets")
    target = next(
        (
            item for item in targets
            if isinstance(item, Mapping) and item.get("id") == target_id
        ),
        None,
    ) if isinstance(targets, (list, tuple)) else None
    if (
        not isinstance(target, Mapping)
        or target.get("kind") != "qemu"
        or target.get("execution_model") != "linux-user"
        or not is_sha256_digest(target.get("identity_digest"))
        or not is_sha256_digest(target.get("binary_sha256"))
    ):
        raise ExecutionEnvironmentError(
            f"configured QEMU reference target identity is unavailable: {target_id}"
        )
    identity_section: dict[str, object] = {
        "backend": backend,
        "target": target_id,
        "identity_resolution": "runtime",
        "expected": {
            "status": "recorded",
            "backend": backend,
            "identity_digest": target["identity_digest"],
            "binary_sha256": target["binary_sha256"],
        },
    }
    binary = identity_binary_path(identity_section, backend)
    if binary is None:
        raise ExecutionEnvironmentError(
            f"configured QEMU reference binary path is unavailable: {target_id}"
        )
    return backend, binary, identity_section, target_id


def _k1_fallback_reason(
    observations: object, fallback_classes: frozenset[str] | None,
) -> str | None:
    if fallback_classes is None:
        return None
    rows = tuple(observations) if isinstance(observations, (list, tuple)) else ()
    for item in rows:
        extra = getattr(item, "extra_state", None)
        failure_class = (
            extra.get("runner_failure_class") if isinstance(extra, Mapping) else None
        )
        if isinstance(failure_class, str) and failure_class in fallback_classes:
            return str(failure_class)
    return None


def _observation_record(observation: object) -> dict[str, object]:
    value = observation.to_dict() if hasattr(observation, "to_dict") else observation
    safe = json_safe(value)
    return dict(safe) if isinstance(safe, Mapping) else {"value": safe}


def _attach_reference_fallback(
    observations: object, fallback: Mapping[str, object],
) -> tuple[object, ...]:
    rows = tuple(observations) if isinstance(observations, (list, tuple)) else ()
    result = []
    for item in rows:
        if isinstance(item, Mapping):
            updated = dict(item)
            extra = updated.get("extra_state")
            updated["extra_state"] = {
                **(dict(extra) if isinstance(extra, Mapping) else {}),
                "reference_fallback": dict(fallback),
            }
            result.append(updated)
            continue
        extra = getattr(item, "extra_state", {})
        result.append(replace(
            item,
            extra_state={
                **(dict(extra) if isinstance(extra, Mapping) else {}),
                "reference_fallback": dict(fallback),
            },
        ))
    return tuple(result)


def backend_runner(
    backend: str, binary_path: object = None,
    identity_section: object = None, recipe: Mapping[str, object] | None = None,
    *, coverage_enabled: bool = True,
    coverage_config: Mapping[str, object] | None = None,
) -> Callable[[object], object]:
    """把 recipe 的 backend/harness/target identity 接到统一执行入口。"""
    def _run_once(
        artifact: object, *, timeout_seconds: float | None = None,
        campaign_deadline_monotonic: float | None = None,
    ) -> object:
        params = getattr(artifact, "run_params", {})
        params = params if isinstance(params, Mapping) else {}
        executable_path = _artifact_executable(artifact, backend)
        requested_coverage = (
            coverage_config if isinstance(coverage_config, Mapping)
            else params.get("framework_coverage")
        )
        if isinstance(requested_coverage, Mapping) and not coverage_enabled:
            # Keep the framework's exclusive Target queue even when the
            # optional coverage side-channel is disabled.
            requested_coverage = {**requested_coverage, "enabled": False}
        generation = recipe.get("generation") if isinstance(recipe, Mapping) else None
        profile = generation.get("generation_profile") if isinstance(generation, Mapping) else None
        sequence = generation.get("sequence") if isinstance(generation, Mapping) else ()
        initial_state = generation.get("initial_state") if isinstance(generation, Mapping) else None
        expected_text = list(sequence) if isinstance(sequence, (list, tuple)) else []
        if isinstance(initial_state, Mapping):
            expected_text.append(initial_state.get("expected", ""))
        expected = identity_section.get("expected") if isinstance(identity_section, Mapping) else None
        expected_trap = any(
            set(re.findall(r"[a-z0-9-]+", str(item).lower().replace("_", "-")))
            & {"trap", "illegal", "illegal-instruction"}
            for item in expected_text
        ) or isinstance(expected, Mapping) and expected.get("status") == "trap"
        single_case = params.get("single_case")
        if params.get("route") == "single" and isinstance(single_case, Mapping):
            dataflow = single_case.get("dataflow_meta")
            expected_trap = expected_trap_from_dataflow(
                dict(dataflow) if isinstance(dataflow, Mapping) else {},
            )
        if backend == "sail-riscv":
            from framework.riscv_semantic_references import sail_program_observations

            isa_profile = profile.get("isa") if isinstance(profile, Mapping) else None
            isa_profile = isa_profile or generation.get("isa") if isinstance(generation, Mapping) else isa_profile
            mabi = profile.get("mabi") if isinstance(profile, Mapping) else None
            mabi = mabi or generation.get("mabi") if isinstance(generation, Mapping) else mabi
            if not isinstance(isa_profile, str) or not isa_profile.strip():
                return sail_program_observations(
                    getattr(artifact, "source_path", ""), isa_profile="",
                )
            return sail_program_observations(
                getattr(artifact, "source_path", ""),
                isa_profile=isa_profile, mabi=mabi if isinstance(mabi, str) else None,
                profile_id=isa_profile,
                input_id=params.get("input_id") if isinstance(params.get("input_id"), str) else None,
                run_params=params, expected_trap=expected_trap,
            )

        selected_binary = params.get("backend_binary_path") or binary_path
        if selected_binary is None and backend == "unicorn-riscv64":
            selected_binary = identity_binary_path(identity_section, backend)
        runner_env = None
        if backend in CAPSULE_BACKENDS:
            _, addresses = symbol_offsets_from_elf(
                executable_path,
                ("obs_buf", "test_memory_start", "test_memory_end"),
                allow_before_start=True,
            )
            if "obs_buf" in addresses:
                runner_env = {
                    CAPSULE_MAILBOX_ENV: addresses["obs_buf"],
                    CAPSULE_OBSERVATION_SIZE_ENV: str(OBSERVATION_HEADER_SIZE),
                }
            if all(name in addresses for name in ("obs_buf", "test_memory_start", "test_memory_end")):
                start, end = (int(addresses[name], 0) for name in ("test_memory_start", "test_memory_end"))
                runner_env.update({
                    CAPSULE_OBSERVATION_SIZE_ENV: str(OBSERVATION_HEADER_SIZE + end - start),
                    CAPSULE_TEST_MEMORY_ENV: addresses["test_memory_start"],
                    CAPSULE_MEMORY_SIZE_ENV: str(end - start),
                })
        if backend == "rax-riscv64" and not runner_env:
            runner_env = {CAPSULE_OBSERVATION_SIZE_ENV: str(OBSERVATION_HEADER_SIZE)}
        if backend == "rvvm-riscv64" and isinstance(selected_binary, str):
            try:
                target = read_json_object(Path(selected_binary).with_name("target-identity.json"))
            except (OSError, ValueError):
                target = {}
            source = target.get("source_repository")
            if isinstance(source, str) and source.strip():
                runner_env = runner_env or {}
                runner_env["RVVM_SOURCE"] = source
            runner_env = runner_env or {}
            runner_env["LD_LIBRARY_PATH"] = str(Path(selected_binary).parent)
        if backend == "unicorn-riscv64" and isinstance(selected_binary, str):
            library = Path(selected_binary).resolve()
            runner_env = runner_env or {}
            runner_env["LIBUNICORN_PATH"] = str(library.parent)
            identity = library.with_name("target-identity.json")
            if identity.is_file():
                runner_env["RV_UNICORN_TARGET_IDENTITY_PATH"] = str(identity)
                target_identity = read_json_object(identity)
                if isinstance(target_identity.get("source_commit"), str):
                    runner_env["RV_UNICORN_SOURCE_COMMIT"] = target_identity["source_commit"]
            runner_env["LD_LIBRARY_PATH"] = str(library.parent)
        if backend in {"qemu", "qemu-riscv32", "qemu-riscv64"} and params.get("qemu_cpu"):
            runner_env = runner_env or {}
            runner_env["RV_QEMU_CPU"] = str(params["qemu_cpu"])
        runtime_backend = backend
        isa = str(params.get("isa") or params.get("isa_profile") or "")
        observer = recipe.get("observer") if isinstance(recipe, Mapping) else None
        fields = observer.get("fields", ()) if isinstance(observer, Mapping) else ()
        if (
            params.get("route") == "single"
            and isinstance(single_case, Mapping)
            and isinstance(single_case.get("compare_mask"), Mapping)
        ):
            fields = tuple(dict.fromkeys((
                *fields,
                *_compare_mask_fields(CompareMask.from_dict(
                    dict(single_case["compare_mask"]),
                )),
            )))
        needs_fp = any(str(field).removeprefix("extra.") in {"before_fpr_rawbits", "after_fpr_rawbits", "fpr_rawbits", "before_fflags", "after_fflags", "fflags", "csr.fflags", "before_frm", "after_frm", "frm", "csr.frm", "mstatus_fs", "csr.mstatus.fs"} for field in fields)
        runner_env = runner_env or {}
        if expected_trap:
            runner_env["RV_TESTCASE_EXPECTED_TRAP"] = "1"
        if backend == "rvvm-riscv64":
            runner_env["RVVM_BARE_METAL_CAPSULE"] = "1"
        runner_env["RV_TESTCASE_REQUIRE_FINAL_FP_STATE"] = "1" if needs_fp else "0"
        if backend in {"rax-riscv64", "renode-riscv64", "rvvm-riscv64"} and isa:
            runner_env["RV_TESTCASE_ISA_PROFILE"] = isa
            runner_env["RV_TESTCASE_ISA_PROFILE_NAME"] = isa
        if runtime_backend == "qemu":
            runtime_backend = "qemu-riscv32" if isa.lower().startswith("rv32") else "qemu-riscv64"
        elif runtime_backend == "qemu-riscv64" and isa.lower().startswith("rv32"):
            runtime_backend = "qemu-riscv32"
        if params.get("route") == "single" and isinstance(single_case, Mapping):
            dataflow = single_case.get("dataflow_meta")
            contracts = (
                dataflow.get("risk_contracts", ())
                if isinstance(dataflow, Mapping) else ()
            )
            disabled = sorted({
                token
                for contract in contracts
                if isinstance(contract, Mapping)
                and isinstance((boundary := contract.get("boundary_class")), str)
                and boundary.startswith("extension-gate:")
                for token in boundary.removeprefix("extension-gate:").split("+")
                if token
            })
            if disabled:
                runner_env["RV_TESTCASE_DISABLED_EXTENSIONS"] = ",".join(disabled)
            if runtime_backend in {"qemu-riscv32", "qemu-riscv64"}:
                try:
                    testcase = TestCase.from_dict(dict(single_case))
                    _, addresses = symbol_offsets_from_elf(
                        executable_path, ("test_start",),
                    )
                    start = int(addresses["test_start"], 0)
                    risk_pcs = tuple(
                        start + item.pc_offset
                        for item in testcase.instruction_meta
                        if "risk" in item.tags
                    )
                except (KeyError, TypeError, ValueError, OSError):
                    risk_pcs = ()
                if risk_pcs:
                    runner_env["RV_TESTCASE_RISK_PCS"] = ",".join(
                        hex(pc) for pc in risk_pcs
                    )
        if runtime_backend.startswith(("qemu-", "libriscv-")) and isinstance(fields, (list, tuple)):
            requested_gprs: set[int] = set()
            for field in fields:
                name = str(field).removeprefix("extra.")
                if name == "gpr":
                    requested_gprs.update(range(32))
                    break
                match = re.fullmatch(r"gpr(?:\.x|\[)(\d+)\]?", name)
                if match and int(match.group(1)) < 32:
                    requested_gprs.add(int(match.group(1)))
            volatile_gprs = sorted(set(range(32)) - requested_gprs)
            if volatile_gprs:
                runner_env["RV_VOLATILE_GPR_INDICES"] = ",".join(
                    str(index) for index in volatile_gprs
                )
        selected_coverage_config = (
            requested_coverage
            if isinstance(requested_coverage, Mapping)
            and requested_coverage.get("backend") == runtime_backend
            else None
        )
        if runtime_backend != backend and backend in {"qemu", "qemu-riscv32", "qemu-riscv64"}:
            selected_binary = None
        if selected_binary is None:
            lookup = identity_section
            if runtime_backend != backend and runtime_backend in {"qemu-riscv32", "qemu-riscv64"} and isinstance(identity_section, Mapping):
                lookup = {key: value for key, value in identity_section.items() if key != "binary_path"}
                lookup["backend"] = runtime_backend
            selected_binary = identity_binary_path(lookup, runtime_backend)
        if selected_binary is not None and runtime_backend in {
            "qemu-riscv32", "qemu-riscv64", "libriscv-translated",
            "rvvm-riscv64", "unicorn-riscv64", "renode-riscv64", "rax-riscv64",
        }:
            if runtime_backend == "rax-riscv64":
                try:
                    rax_identity = read_json_object(Path(selected_binary).with_name("target-identity.json"))
                except (OSError, ValueError):
                    rax_identity = {}
                source = rax_identity.get("source_repository")
                commit = rax_identity.get("source_commit")
                local_source = (
                    LOCAL_ROOT / "external" / "formal-git" / f"rax-{commit[:4]}"
                    if isinstance(commit, str) else None
                )
                if local_source is None or not local_source.is_dir():
                    # The execution image mounts the checked-out compatibility
                    # source under simulator-sources; target identities retain
                    # /input/rax as the reproducible build path.
                    candidate = LOCAL_ROOT / "simulator-sources" / "rax"
                    local_source = candidate if candidate.is_dir() else local_source
                if local_source is None or not local_source.is_dir():
                    deps_root = Path(os.environ.get("RQ1_DEPS", "/path/to/deps"))
                    candidate = deps_root / "simulator-sources" / "rax"
                    local_source = candidate if candidate.is_dir() else local_source
                if local_source is not None and local_source.is_dir():
                    source = str(local_source)
                if isinstance(source, str) and source.strip():
                    runner_env = runner_env or {}
                    runner_env["RAX_SOURCE"] = source
            identity_backend = (
                runtime_backend
                if runtime_backend in {"qemu-riscv32", "qemu-riscv64"}
                else canonical_execution_backend(runtime_backend)
            )
            installed = load_target_binary_manifest(
                Path(selected_binary).with_name("target-identity.json"),
                expected_backend=identity_backend,
            )
            expected = identity_section if isinstance(identity_section, Mapping) else {}
            expected = expected.get("expected") if isinstance(expected.get("expected"), Mapping) else expected
            for field in ("identity_digest", "binary_sha256"):
                wanted = expected.get(field)
                if wanted not in (None, "") and wanted != getattr(installed.identity, field):
                    raise ExecutionEnvironmentError(f"target identity {field} does not match recipe")
            selected_binary = str(installed.binary_path)
        return _attach_program_test_path(run_backend_with_coverage(
            runtime_backend,
            executable_path,
            profile_id=(params.get("isa") or params.get("isa_profile")),
            input_id=params.get("input_id"),
            runner_env=runner_env,
            backend_binary_path=selected_binary,
            coverage_config=selected_coverage_config,
            timeout_seconds=timeout_seconds,
            campaign_deadline_monotonic=campaign_deadline_monotonic,
        ), executable_path)

    fallback_spec = (
        identity_section.get("fallback")
        if backend == "native-rv64" and isinstance(identity_section, Mapping) else None
    )
    fallback_runner = None
    fallback_context: dict[str, object] | None = None
    reference_selection_pinned = False

    def run(artifact: object, *, timeout_seconds: float | None = None) -> object:
        nonlocal fallback_runner, fallback_context, reference_selection_pinned
        if fallback_context is not None:
            try:
                rows = fallback_runner(artifact, timeout_seconds=timeout_seconds)
            except Exception as error:
                rows = _qemu_fallback_gap(artifact, error)
            return _attach_reference_fallback(rows, fallback_context)

        observations = _run_once(artifact, timeout_seconds=timeout_seconds)
        if reference_selection_pinned:
            return observations
        reference_selection_pinned = True
        fallback_classes = (
            configured_k1_qemu_fallback_classes(fallback_spec.get("on"))
            if isinstance(fallback_spec, Mapping) else None
        )
        fallback_reason = _k1_fallback_reason(observations, fallback_classes)
        if not isinstance(fallback_spec, Mapping) or fallback_reason is None:
            return observations

        attempts = [
            _observation_record(item)
            for item in (observations if isinstance(observations, (list, tuple)) else ())
        ]
        backend_name = str(fallback_spec.get("backend", ""))
        target_id = fallback_spec.get("target_id")
        fallback_context = {
            "schema_version": "rq1-reference-fallback-v1",
            "policy": K1_QEMU_FALLBACK_POLICY,
            "preferred_backend": "native-rv64",
            "effective_backend": backend_name,
            "fallback_reason": fallback_reason,
            "selection_scope": fallback_spec.get("selection_scope"),
            "selection_point": "first-reference-request",
            "target_id": target_id,
            "k1_attempt_digest": canonical_digest(attempts),
        }
        try:
            fallback_backend, fallback_binary, fallback_identity, resolved_target_id = (
                _qemu_fallback_configuration(identity_section)
            )
            if fallback_backend != backend_name or resolved_target_id != target_id:
                raise ExecutionEnvironmentError("resolved QEMU fallback does not match recipe")
            fallback_runner = backend_runner(
                fallback_backend, fallback_binary, fallback_identity, recipe,
                coverage_enabled=False,
            )
            fallback_observations = fallback_runner(
                artifact, timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            fallback_observations = _qemu_fallback_gap(artifact, error)
        return _attach_reference_fallback(
            fallback_observations,
            {**fallback_context, "k1_attempts": attempts},
        )

    return run


def _qemu_fallback_gap(artifact: object, error: Exception) -> tuple[object, ...]:
    params = getattr(artifact, "run_params", {})
    params = params if isinstance(params, Mapping) else {}
    executable = getattr(artifact, "executable_path", None)
    digest = getattr(artifact, "executable_sha256", None)
    if not is_sha256_digest(digest) and isinstance(executable, (str, Path)):
        try:
            digest = sha256_file(Path(executable))
        except OSError:
            digest = None
    return (_runner_contract_gap_observation(
        "qemu-riscv64",
        f"QEMU reference fallback failed: {type(error).__name__}: {error}",
        binary_sha256=digest if is_sha256_digest(digest) else None,
        profile_id=(params.get("isa") or params.get("isa_profile"))
        if isinstance(params.get("isa") or params.get("isa_profile"), str) else None,
        input_id=params.get("input_id") if isinstance(params.get("input_id"), str) else None,
    ),)

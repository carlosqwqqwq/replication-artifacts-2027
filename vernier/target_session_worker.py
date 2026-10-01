"""Per-Target resident JSONL worker.

The worker process is started once for one Target.  Each ``run`` request
creates a clean guest machine, loads the case ELF, executes it, and returns
the observation plus coverage metadata.  The backend library stays loaded in
this process; no Target request calls the framework's one-shot process runner.

Unicorn uses its in-process adapter.  The other backends reuse the existing
``backend_runner`` from this same resident service process; their native
binary remains loaded/configured by the adapter and is called for each reset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import time


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROTOCOL = "rq1-target-session-v1"
EMBEDDED_BACKENDS = frozenset({"unicorn-riscv64"})


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


class ResidentTargetWorker:
    """One backend library lifetime and one request queue."""

    def __init__(
        self, backend: str, metadata: Mapping[str, object] | None = None,
        *, binary_path: str | None = None,
    ):
        self.backend = backend
        self.metadata = dict(metadata or {})
        self.binary_path = Path(binary_path).resolve() if binary_path else None
        if backend not in EMBEDDED_BACKENDS and self.binary_path is None:
            raise RuntimeError(
                f"{backend or '<unknown>'} resident worker needs a target binary"
            )
        self._identity: dict[str, object] = {}
        if self.binary_path is not None:
            try:
                from framework._util import read_json_object
                identity = read_json_object(
                    self.binary_path.with_name("target-identity.json")
                )
                if isinstance(identity, Mapping):
                    self._identity = dict(identity)
            except (OSError, TypeError, ValueError):
                self._identity = {}
        # Import and capability discovery happen once, before the queue loop.
        # A later run only resets/loads a guest machine through this library.
        from framework.adapters import capsule_unicorn
        import unicorn

        self._run_capsule = capsule_unicorn._run_capsule
        self._unicorn_version = str(getattr(unicorn, "__version__", "unknown"))
        self.session_pid = os.getpid()
        self.run_count = 0
        self.coverage_warm = True
        self._dotnet_coverage_servers: dict[str, object] = {}

    def hello(self) -> dict[str, object]:
        return {
            "status": "ready",
            "protocol": PROTOCOL,
            "backend": self.backend,
            "session_pid": self.session_pid,
            "backend_session_scope": "per-target-session",
            "library_lifetime": "worker-process",
            "coverage_warm": self.coverage_warm,
        }

    @staticmethod
    def _artifact_path(artifact: Mapping[str, object]) -> Path:
        value = artifact.get("bare_executable_path") or artifact.get("executable_path")
        if not isinstance(value, str) or not value:
            raise ValueError("resident worker artifact executable is missing")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    @staticmethod
    def _case_options(
        artifact: Mapping[str, object], metadata: Mapping[str, object],
    ) -> dict[str, object]:
        params = artifact.get("run_params")
        params = dict(params) if isinstance(params, Mapping) else {}
        options = {}
        for source in (metadata, params):
            for name in (
                "mailbox", "observation_size", "test_memory", "memory_size",
                "runner_env", "isa_profile", "isa", "profile", "profile_id",
            ):
                if name in source:
                    options[name] = source[name]
        # Program-Full artifacts identify the same execution profile as
        # ``isa`` while Direct/RVGEN records use ``isa_profile``.  Normalize
        # both forms before the capsule runner checks its environment; a
        # resident service must not reject an otherwise valid case merely
        # because the route used a different field name.
        if options.get("isa_profile") is None:
            for name in ("isa", "profile", "profile_id"):
                if options.get(name) is not None:
                    options["isa_profile"] = options[name]
                    break
        coverage = metadata.get("coverage_config")
        if isinstance(coverage, Mapping):
            options.setdefault("isa_profile", coverage.get("isa_profile"))
        case = params.get("single_case")
        if isinstance(case, Mapping):
            options.setdefault("isa_profile", case.get("isa_profile"))
            options.setdefault("input_id", case.get("input_id"))
        return options

    @staticmethod
    def _capsule_layout(
        elf: Path, options: Mapping[str, object],
    ) -> tuple[int, int, int | None, int | None]:
        """Resolve the common capsule mailbox from request metadata/ELF symbols."""
        values = {name: options.get(name) for name in (
            "mailbox", "observation_size", "test_memory", "memory_size",
        )}
        if any(values[name] is None for name in ("mailbox", "observation_size")):
            from framework.direct_elf import symbol_offsets_from_elf
            _, symbols = symbol_offsets_from_elf(
                elf, ("obs_buf", "test_memory_start", "test_memory_end"),
                allow_before_start=True,
            )
            if not all(name in symbols for name in (
                "obs_buf", "test_memory_start", "test_memory_end",
            )):
                _, symbols = symbol_offsets_from_elf(
                    elf, ("result_buffer", "test_memory"), allow_before_start=True,
                )
                if "obs_buf" not in symbols and "result_buffer" in symbols:
                    symbols["obs_buf"] = symbols["result_buffer"]
                if "test_memory_start" not in symbols and "test_memory" in symbols:
                    symbols["test_memory_start"] = symbols["test_memory"]
            if values["mailbox"] is None:
                values["mailbox"] = symbols.get("obs_buf")
            if values["test_memory"] is None:
                values["test_memory"] = symbols.get("test_memory_start")
            if values["memory_size"] is None:
                try:
                    values["memory_size"] = int(symbols["test_memory_end"], 0) - int(
                        symbols["test_memory_start"], 0,
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            if values["observation_size"] is None and values["memory_size"] is not None:
                from framework.direct_case import OBSERVATION_HEADER_SIZE
                values["observation_size"] = (
                    OBSERVATION_HEADER_SIZE + int(values["memory_size"])
                )
        try:
            mailbox = int(str(values["mailbox"]), 0)
            observation_size = int(str(values["observation_size"]), 0)
        except (TypeError, ValueError):
            raise ValueError(
                "resident Unicorn request needs mailbox and observation_size"
            ) from None
        test_memory = values["test_memory"]
        memory_size = values["memory_size"]
        return (
            mailbox, observation_size,
            int(str(test_memory), 0) if test_memory is not None else None,
            int(str(memory_size), 0) if memory_size is not None else None,
        )

    @staticmethod
    def _coverage_config(metadata: Mapping[str, object]) -> dict[str, object]:
        value = metadata.get("coverage_config")
        result = dict(value) if isinstance(value, Mapping) else {}
        for name, flag in (
            ("rv_instruction_coverage", "rv_instruction_coverage_enabled"),
            ("rv_opcode_catalog_coverage", "rv_opcode_catalog_coverage_enabled"),
        ):
            config = metadata.get(name)
            if isinstance(config, Mapping) and config.get("enabled") is True:
                result[flag] = True
        return result

    def _ensure_dotnet_coverage_server(
        self, config: Mapping[str, object],
    ) -> dict[str, object]:
        """Reuse v1's batch collector while keeping case status separate."""
        source = config.get("source_coverage")
        if self.backend != "renode-riscv64" or not isinstance(source, Mapping) \
                or source.get("collector") != "dotnet":
            return dict(config)
        batch_raw = config.get("coverage_batch_raw_dir") or config.get("raw_dir")
        if not isinstance(batch_raw, str) or not batch_raw:
            return dict(config)
        server = self._dotnet_coverage_servers.get(batch_raw)
        if server is None:
            from framework.framework_coverage import DotnetCoverageServer
            server = DotnetCoverageServer(config)
            server.start()
            self._dotnet_coverage_servers[batch_raw] = server
        session_id = getattr(server, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("dotnet coverage server session id is missing")
        return {
            **dict(config),
            "coverage_batch_session_id": session_id,
            "dotnet_coverage_session_mode": "server",
        }

    def close(self) -> None:
        servers = self._dotnet_coverage_servers
        self._dotnet_coverage_servers = {}
        for server in servers.values():
            try:
                server.stop()
            except Exception:
                # The server writes its own shutdown marker; keep worker exit
                # independent from collector cleanup.
                pass

    @staticmethod
    def _add_rv_metrics(
        elf: Path, profile: object, coverage_config: Mapping[str, object],
        coverage: dict[str, object], pcs: list[int],
    ) -> None:
        if not any(
            coverage_config.get(name) is True
            for name in (
                "rv_instruction_coverage_enabled",
                "rv_opcode_catalog_coverage_enabled",
            )
        ):
            return
        try:
            from analysis.elf_features import decode_elf
            from analysis.rv_instruction_coverage import case_metrics

            feature = decode_elf(elf, str(profile or ""))
            metric_input = {
                **coverage,
                "status": (
                    "observed" if coverage.get("status") in {
                        "recorded", "raw-trace-recorded",
                    } else "NA" if coverage.get("status") == "target-unsupported"
                    else "gap"
                ),
                "trace_pcs": list(pcs),
            }
            metrics = case_metrics(feature, metric_input)
            for name in (
                "rv_instruction_coverage", "rv_opcode_catalog_coverage",
            ):
                flag = name + "_enabled"
                if coverage_config.get(flag) is True:
                    coverage[name] = metrics.get(name, {
                        "status": "gap", "reason": name + "-metric-missing",
                    })
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            for name in (
                "rv_instruction_coverage", "rv_opcode_catalog_coverage",
            ):
                if coverage_config.get(name + "_enabled") is True:
                    coverage[name] = {"status": "gap", "reason": reason}

    @staticmethod
    def _source_profile_status(
        coverage_config: Mapping[str, object],
    ) -> dict[str, object]:
        source = coverage_config.get("source_coverage")
        raw_value = coverage_config.get("raw_dir")
        if not isinstance(source, Mapping) or not raw_value:
            return {"status": "not-configured"}
        raw_dir = Path(str(raw_value))
        collector = str(source.get("collector") or "")
        suffixes = {
            "gcov": (".gcda",), "lcov": (".gcda",),
            "llvm": (".profraw",),
            "dotnet": (".cobertura.xml", ".cobertura.xml.gz", ".coverage"),
        }.get(collector, ())
        files = [
            path for suffix in suffixes for path in raw_dir.rglob("*" + suffix)
            if path.is_file() and path.stat().st_size > 0
        ] if raw_dir.is_dir() else []
        return {
            "status": "recorded" if files else "gap",
            "collector": collector,
            "raw_dir": str(raw_dir),
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }

    def _run_unicorn_source_coverage(
        self, elf: Path, options: Mapping[str, object],
        coverage_config: Mapping[str, object],
    ) -> dict[str, object]:
        """Run the instrumented Unicorn once, in a short-lived child.

        The resident worker keeps the normal Unicorn binding warm.  The
        instrumented library cannot be swapped into that already-imported
        process, so a child process is the smallest reliable way to flush
        GCC counters while preserving the resident Target protocol.
        """
        source = coverage_config.get("source_coverage")
        if not isinstance(source, Mapping) or source.get("collector") not in {
            "gcov", "lcov",
        }:
            return {"status": "not-configured"}
        binary_value = (
            coverage_config.get("coverage_binary")
            or coverage_config.get("binary_path")
        )
        raw_value = coverage_config.get("raw_dir")
        if not isinstance(binary_value, str) or not binary_value:
            return {"status": "gap", "reason": "coverage-binary-missing"}
        if not isinstance(raw_value, str) or not raw_value:
            return {"status": "gap", "reason": "coverage-raw-dir-missing"}
        binary = Path(binary_value)
        raw_dir = Path(raw_value)
        if not binary.is_file():
            return {"status": "gap", "reason": "coverage-binary-missing"}
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = raw_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            mailbox, observation_size, test_memory, memory_size = (
                self._capsule_layout(elf, options)
            )
            profile = options.get("isa_profile")
            env = os.environ.copy()
            env.update({
                "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
                    str(binary.parent), env.get("LD_LIBRARY_PATH", ""),
                ))),
                "GCOV_PREFIX": str(raw_dir),
                "GCOV_PREFIX_STRIP": str(source.get("gcov_prefix_strip", 2)),
                "TMPDIR": str(tmp_dir),
            })
            if isinstance(profile, str) and profile:
                env["RV_TESTCASE_ISA_PROFILE"] = profile
                env["RV_TESTCASE_ISA_PROFILE_NAME"] = profile
            package_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                package_root, env.get("PYTHONPATH", ""),
            )))
            command = [
                sys.executable, "-u", "-m",
                "framework.adapters.capsule_dispatch", "--unicorn-bin",
                str(binary), str(elf), "--mailbox", hex(mailbox),
                "--observation-size", str(observation_size),
            ]
            if test_memory is not None:
                command.extend(("--test-memory", hex(test_memory)))
            if memory_size is not None:
                command.extend(("--memory-size", str(memory_size)))
            timeout_value = self.metadata.get("timeout_seconds")
            timeout = (
                max(1.0, float(timeout_value))
                if isinstance(timeout_value, (int, float)) else 180.0
            )
            completed = subprocess.run(
                command, env=env, cwd=package_root,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "gap", "reason": "coverage-run-timeout"}
        except (OSError, TypeError, ValueError) as error:
            return {"status": "gap", "reason": f"coverage-run:{type(error).__name__}"}
        profile_status = self._source_profile_status(coverage_config)
        if completed.returncode != 0 and profile_status.get("status") != "recorded":
            return {
                **profile_status,
                "status": "gap",
                "reason": "coverage-run-failed",
                "returncode": completed.returncode,
            }
        return {**profile_status, "returncode": completed.returncode}

    def _run_unicorn(
        self, artifact: Mapping[str, object], metadata: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        elf = self._artifact_path(artifact)
        options = self._case_options(artifact, metadata)
        mailbox, observation_size, test_memory, memory_size = self._capsule_layout(
            elf, options,
        )
        env = options.get("runner_env")
        env = dict(env) if isinstance(env, Mapping) else {}
        profile = options.get("isa_profile") or env.get("RV_TESTCASE_ISA_PROFILE")
        if isinstance(profile, str) and profile:
            env.setdefault("RV_TESTCASE_ISA_PROFILE", profile)
            env.setdefault("RV_TESTCASE_ISA_PROFILE_NAME", profile)
        previous = {name: os.environ.get(name) for name in env}
        try:
            os.environ.update({str(name): str(value) for name, value in env.items()})
            started = time.monotonic()
            stdout, pcs, details, _state, observation_state = self._run_capsule(
                elf, mailbox, observation_size,
                test_memory=test_memory, memory_size=memory_size,
            )
            elapsed = round(max(0.0, time.monotonic() - started), 6)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        stderr_lines = [
            "RV_OBSERVATION_STATE=" + json.dumps(observation_state, separators=(",", ":")),
            "RV_EXECUTED_PCS=" + json.dumps([hex(pc) for pc in pcs], separators=(",", ":")),
            f"RV_INSTRUCTION_COUNT={len(pcs)}",
            f"RV_TOTAL_GUEST_INSTRUCTION_COUNT={len(pcs)}",
            "RV_TOOL_VERSION=unicorn " + self._unicorn_version,
            "RV_TRANSLATION_EVIDENCE=" + json.dumps({
                "backend": self.backend,
                "expected_path": "either",
                "tested_pc_seen": False,
                "tested_pc_translated": False,
                "tested_pc_executed": False,
                "execution_count": len(pcs),
                "details": details,
            }, separators=(",", ":")),
        ]
        from framework.direct_runner import parse_observation_stdout
        from framework._util import sha256_file

        params = artifact.get("run_params")
        params = params if isinstance(params, Mapping) else {}
        input_id = params.get("input_id")
        if not isinstance(input_id, str):
            case = params.get("single_case")
            input_id = case.get("input_id") if isinstance(case, Mapping) else None
        observation = parse_observation_stdout(
            self.backend, 0, stdout, ("\n".join(stderr_lines) + "\n").encode(),
            str(artifact.get("bare_executable_sha256")
                or artifact.get("executable_sha256") or sha256_file(elf)),
            "unicorn " + self._unicorn_version,
            str(profile) if profile else None,
            input_id if isinstance(input_id, str) else None,
        )
        coverage_config = self._coverage_config(metadata)
        guest_trace_status = (
            "raw-trace-recorded" if details.get("trace_available") else "gap"
        )
        coverage = {
            # The resident worker records guest PCs; the instrumented child
            # below flushes simulator source counters for offline merging.
            "status": guest_trace_status,
            "guest_trace_status": guest_trace_status,
            "simulator_source_profile_status": "not-collected-by-resident-worker",
            "collector": "unicorn-hooks",
            "backend": self.backend,
            "session_pid": self.session_pid,
            "job_id": metadata.get("job_id"),
            "pc_count": len(pcs),
            "config": _jsonable(coverage_config) if isinstance(coverage_config, Mapping) else None,
        }
        source_profile = self._run_unicorn_source_coverage(
            elf, options, coverage_config,
        )
        coverage["simulator_source_profile"] = source_profile
        coverage["simulator_source_profile_status"] = source_profile.get("status")
        self._add_rv_metrics(elf, profile, coverage_config, coverage, list(pcs))
        observation_record = observation.to_dict()
        extra = observation_record.get("extra_state")
        observation_record["extra_state"] = {
            **(dict(extra) if isinstance(extra, Mapping) else {}),
            "resident_session": {
                "pid": self.session_pid,
                "scope": "per-target-session",
                "library_lifetime": "worker-process",
            },
            "simulator_coverage": coverage,
        }
        timing = {
            "backend_total_seconds": elapsed,
            "reset_load_seconds": None,
            "guest_execution_seconds": elapsed,
            "trace_flush_read_seconds": 0.0,
            "coverage_setup_seconds": 0.0,
            "session_pid": self.session_pid,
            "backend_session_scope": "per-target-session",
        }
        return [observation_record], timing, coverage

    def _run_external(
        self, artifact: Mapping[str, object], metadata: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        """Run a non-Unicorn backend through the existing adapter seam."""
        from framework.adapters.program import _observation_record
        from framework.program_runner import backend_runner
        from framework.target_queue_service import _built_program

        if self.binary_path is None:
            raise RuntimeError("resident worker target binary is missing")
        started = time.monotonic()
        identity = {
            **self._identity,
            "backend": self.backend,
            "binary_path": str(self.binary_path),
        }
        coverage_config = self._coverage_config(metadata)
        if coverage_config:
            # The container runtime manifest keeps the short Target-facing
            # fields, while the coverage adapter needs the complete identity
            # contract. Fill only the missing fields here so old manifests
            # remain executable and the resident path matches the one-shot
            # path.
            coverage_config = {
                **coverage_config,
                "backend": coverage_config.get("backend") or self.backend,
                "binary_path": (
                    coverage_config.get("binary_path")
                    or coverage_config.get("coverage_binary")
                    or str(self.binary_path)
                ),
                "identity_binary_path": (
                    coverage_config.get("identity_binary_path")
                    or str(self.binary_path)
                ),
            }
        raw_root = metadata.get("raw_root")
        if isinstance(raw_root, str) and raw_root and not coverage_config.get("raw_dir"):
            # Coverage adapters need a case-local directory for their trace
            # side channel.  Without it RVVM falls back to per-instruction
            # GDB stepping and long Program-Full capsules time out.
            coverage_config = {**coverage_config, "raw_dir": raw_root}
        coverage_config = self._ensure_dotnet_coverage_server(coverage_config)
        recipe = metadata.get("target_recipe")
        recipe = dict(recipe) if isinstance(recipe, Mapping) else {}
        runner = backend_runner(
            self.backend, str(self.binary_path), identity, recipe,
            coverage_enabled=True, coverage_config=coverage_config,
        )
        built = _built_program(artifact)
        result = runner(
            built,
            timeout_seconds=(
                float(metadata["timeout_seconds"])
                if isinstance(metadata.get("timeout_seconds"), (int, float))
                else None
            ),
        )
        source_profile = {"status": "not-configured"}
        source = coverage_config.get("source_coverage")
        if isinstance(source, Mapping) and source.get("collector") == "dotnet":
            from framework.framework_coverage import wait_for_dotnet_coverage
            source_profile = wait_for_dotnet_coverage(coverage_config)
        rows = (
            list(result) if isinstance(result, (list, tuple)) else [result]
        )
        observations = [
            _observation_record(item) for item in rows if item is not None
        ]
        elapsed = round(max(0.0, time.monotonic() - started), 6)
        execution_gaps = []
        target_unsupported = []
        options = self._case_options(artifact, metadata)
        profile = options.get("isa_profile")
        unsupported_profile = False
        unsupported_extensions: tuple[str, ...] = ()
        if self.backend in {"renode-riscv64", "rvvm-riscv64"}:
            from framework.adapters.contracts import capsule_cpu_profile
            unsupported_profile = (
                isinstance(profile, str) and bool(profile)
                and capsule_cpu_profile(profile) is None
            )
        elif self.backend == "rax-riscv64" and isinstance(profile, str) and profile:
            from framework.spec_definedness import enabled_extensions
            # The pinned RAX build exposes this fixed ISA surface.  Unsupported
            # generated extensions are a target capability result, not a
            # resident-session/coverage failure.
            rax_extensions = frozenset({
                "i", "m", "a", "f", "d", "q", "c", "g", "zicsr",
                "zifencei", "zihintpause", "zihintntl", "zacas", "zawrs",
                "zicbom", "zicboz", "zicbop", "zba", "zbb", "zbc", "zbs",
                "zicond", "zfa", "zbkb", "zfh", "zbkx", "zknh", "zksh",
                "zksed", "zkne", "zknd", "zcb", "zcmp", "zcmt", "zclsd",
                "zilsd", "h", "svinval", "v", "xandes", "xthead", "xhazard3",
                "xida_sltw",
            })
            try:
                unsupported_extensions = tuple(sorted(
                    set(enabled_extensions(profile)) - rax_extensions
                ))
            except (TypeError, ValueError):
                unsupported_extensions = ()
            unsupported_profile = bool(unsupported_extensions)
        for row in observations:
            contract_error = row.get("contract_error")
            outcome = str(row.get("outcome") or "").strip().lower()
            extra = row.get("extra_state")
            extra = dict(extra) if isinstance(extra, Mapping) else {}
            unsupported_reason = extra.get("target_unsupported")
            if unsupported_reason is None and unsupported_profile and contract_error:
                unsupported_reason = (
                    f"isa-profile:{profile}"
                    if not unsupported_extensions else
                    "unsupported-extensions:" + ",".join(unsupported_extensions)
                )
            if unsupported_reason is not None:
                reason = str(unsupported_reason)
                extra["target_unsupported"] = reason
                extra.setdefault("target_preflight", "simulator-capability-contract-v1")
                row["extra_state"] = extra
                target_unsupported.append(reason)
            elif contract_error:
                execution_gaps.append(str(contract_error))
            elif outcome in {
                "runner-contract-gap", "transport-gap", "timeout",
                "unavailable", "failed", "error",
            }:
                execution_gaps.append(outcome)
        successful_observations = (
            len(observations) - len(execution_gaps) - len(target_unsupported)
        )
        observed_coverage = []
        rv_metrics: dict[str, object] = {}
        for row in observations:
            extra = row.get("extra_state")
            extra = extra if isinstance(extra, Mapping) else {}
            coverage = extra.get("simulator_coverage")
            if isinstance(coverage, Mapping):
                observed_coverage.append(dict(coverage))
            for name in ("rv_instruction_coverage", "rv_opcode_catalog_coverage"):
                value = row.get(name) or extra.get(name)
                if isinstance(value, Mapping):
                    rv_metrics[name] = dict(value)
        coverage = {
            "status": (
                "recorded" if successful_observations else
                "target-unsupported" if target_unsupported else "gap"
            ),
            "collector": "resident-backend-runner",
            "backend": self.backend,
            "session_pid": self.session_pid,
            "job_id": metadata.get("job_id"),
            "observations": len(observations),
            "successful_observations": successful_observations,
        }
        if execution_gaps:
            coverage["execution_gaps"] = execution_gaps
        if target_unsupported:
            coverage["target_unsupported"] = sorted(set(target_unsupported))
        if observed_coverage:
            coverage["simulator_coverage"] = observed_coverage
        if target_unsupported and not successful_observations:
            source_profile = {
                "status": "not-applicable",
                "reason": "target-unsupported",
            }
        if source_profile.get("status") not in {"not-configured", "not-started"}:
            coverage["simulator_source_profile"] = source_profile
            coverage["simulator_source_profile_status"] = source_profile.get("status")
        elif source_profile.get("status") == "not-configured":
            source_profile = self._source_profile_status(coverage_config)
            coverage["simulator_source_profile"] = source_profile
            coverage["simulator_source_profile_status"] = source_profile.get("status")
        coverage.update(rv_metrics)
        elf_value = (
            artifact.get("bare_executable_path")
            if self.backend in {"rvvm-riscv64", "renode-riscv64", "rax-riscv64"}
            else artifact.get("linux_executable_path") or artifact.get("executable_path")
        )
        if isinstance(elf_value, str):
            pcs = [
                int(value) for row in observations
                if isinstance(row.get("executed_pcs"), (list, tuple))
                for value in row["executed_pcs"]
                if isinstance(value, int)
            ]
            self._add_rv_metrics(
                Path(elf_value), profile, coverage_config, coverage, pcs,
            )
        # RV metrics need the complete in-memory path.  Compact the persisted
        # Renode response only after coverage has consumed that path.
        from framework.direct_case import _compact_trace_record
        observations = [_compact_trace_record(row) for row in observations]
        timing = {
            "backend_total_seconds": elapsed,
            "reset_load_seconds": None,
            "guest_execution_seconds": elapsed,
            "trace_flush_read_seconds": 0.0,
            "coverage_setup_seconds": 0.0,
            "session_pid": self.session_pid,
            "backend_session_scope": "per-target-session",
        }
        return observations, timing, coverage

    def run(self, request: Mapping[str, object]) -> dict[str, object]:
        if not all(request.get(name) is True for name in ("reset", "load", "collect")):
            raise ValueError("resident Target run requires reset/load/collect=true")
        artifact = request.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("resident Target run artifact is missing")
        metadata = {
            **self.metadata,
            **(dict(request.get("metadata"))
               if isinstance(request.get("metadata"), Mapping) else {}),
            "job_id": request.get("job_id"),
        }
        self.run_count += 1
        result, timing, coverage = (
            self._run_unicorn(artifact, metadata)
            if self.backend in EMBEDDED_BACKENDS else
            self._run_external(artifact, metadata)
        )
        response: dict[str, object] = {
            "status": "ok",
            "result": result,
            "timing": timing,
            "coverage": coverage,
            "session_pid": self.session_pid,
            "backend_session_scope": "per-target-session",
        }
        # Keep RV metrics explicit even when this resident adapter has only
        # collected the raw guest-PC trace.  A named gap is auditable and can
        # be repaired offline; omitting the field makes a configured metric
        # indistinguishable from a broken transport.
        coverage_config = self._coverage_config(metadata)
        if coverage_config:
            for name in ("rv_instruction_coverage", "rv_opcode_catalog_coverage"):
                if coverage_config.get(name + "_enabled") is True:
                    response[name] = coverage.get(name) or {
                        "status": "gap",
                        "reason": name + "-metric-missing",
                    }
        return response

    def serve(self, stream_in, stream_out) -> int:
        try:
            for raw in stream_in:
                try:
                    request = json.loads(raw)
                    if not isinstance(request, Mapping):
                        raise ValueError("request must be an object")
                    operation = request.get("op")
                    if operation == "hello":
                        metadata = request.get("metadata")
                        if isinstance(metadata, Mapping):
                            self.metadata.update(metadata)
                        response = self.hello()
                    elif operation == "run":
                        response = self.run(request)
                    elif operation == "close":
                        return 0
                    else:
                        raise ValueError(f"unknown resident Target operation: {operation}")
                except Exception as error:  # one bad job must become one result
                    response = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "session_pid": self.session_pid,
                    }
                stream_out.write(json.dumps(_jsonable(response), separators=(",", ":")) + "\n")
                stream_out.flush()
            return 0
        finally:
            self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ1 resident Target JSONL worker")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--binary")
    args = parser.parse_args(argv)
    try:
        worker = ResidentTargetWorker(args.backend, binary_path=args.binary)
    except Exception as error:
        sys.stdout.write(json.dumps({
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }) + "\n")
        sys.stdout.flush()
        return 2
    return worker.serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())

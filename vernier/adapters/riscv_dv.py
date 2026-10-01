"""RISC-V-DV 多指令程序的最小入口。"""

import math
import os
import posixpath
import re
import shlex
import signal
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from framework._util import (
    atomic_write_json, canonical_digest, git_head, sha256_file,
    strip_c_comments as _strip_c_comments,
)
from framework.adapters.program import _logical_source_entries, _split_source_statements
from framework.execution_environment import ExecutionEnvironmentError, require_execution_plane
from framework.rvgen.provider import CaseProgram
from framework.rvemi.program import (
    _expand_asm_macros, program_sha256 as canonical_program_sha256,
)
from framework.spec_definedness import enabled_extensions, is_canonical_isa_profile

_TARGET_ISA = {
    "rv32imc": "rv32imc_zicsr_zifencei",
    "rv32i": "rv32i_zicsr_zifencei",
    "rv32imafdc": "rv32imafdc_zicsr_zifencei",
    "rv32imcb": "rv32imc_zba_zbb_zbc_zbkb_zbs_zicsr_zifencei",
    "multi_harts": "rv32gc_zicsr_zifencei",
    "rv32imc_sv32": "rv32imc_zicsr_zifencei",
    "rv64imc": "rv64imc_zicsr_zifencei",
    # RISC-V-DV names this target ``imcb``; the executable ISA identity uses
    # the ratified B subsets that the catalog can canonicalize.
    "rv64imcb": "rv64imc_zba_zbb_zbc_zbkb_zbs_zicsr_zifencei",
    "rv64gc": "rv64gc_zicsr_zifencei",
    "rv64gcv": "rv64gcv_zicsr_zifencei",
    "ml": "rv64imc_zicsr_zifencei",
    "rv64imafdc": "rv64imafdc_zicsr_zifencei",
}

_TARGET_MABI = {target: "ilp32" if target.startswith("rv32") or target == "multi_harts" else "lp64" for target in _TARGET_ISA}
_SIMULATOR_TOOLS = {"vcs": "vcs", "verilator": "verilator", "questa": "vsim", "xrun": "xrun"}
_PROFILE_PATH_FIELDS = frozenset({
    "testlist", "custom_target", "core_setting_dir", "user_extension_dir",
    "csr_yaml", "simulator_yaml", "iss_yaml", "asm_test", "c_test",
})


def _toolchain_identity(timeout_sec: float | None = None) -> dict[str, object]:
    compiler = os.environ.get("RISCV_GCC") or os.environ.get("RISCV_TOOLCHAIN") or os.environ.get("CROSS_COMPILE") or "riscv64-linux-gnu-gcc"
    if compiler.endswith("-"):
        compiler += "gcc"
    path = shutil.which(compiler)
    if path is None:
        return {"status": "unavailable", "compiler": compiler}
    if timeout_sec is not None and timeout_sec <= 0:
        return {"status": "unavailable", "compiler": compiler}
    try:
        result = subprocess.run(
            (path, "--version"), capture_output=True, text=True, check=False,
            **({"timeout": timeout_sec} if timeout_sec is not None else {}),
        )
        if result.returncode != 0:
            return {"status": "unavailable", "compiler": compiler}
        version = (result.stdout or result.stderr).splitlines()[0] if result.stdout or result.stderr else ""
        digest = sha256_file(Path(path))
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "compiler": compiler}
    return {
        "status": "ready", "compiler": compiler, "path": path,
        "version": version, "binary_sha256": digest,
    }


def _option_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("RISC-V-DV options must be strings")
        text = " ".join(value)
    elif isinstance(value, str):
        text = value
    else:
        raise ValueError("RISC-V-DV options must be a string or string sequence")
    return " ".join(text.split())


def _normalize_generation_profile(
    generation_profile: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        name: str(Path(os.path.expanduser(str(value))))
        if name in _PROFILE_PATH_FIELDS and value not in (None, "") else value
        for name, value in dict(generation_profile or {}).items()
    }


def _portable_generation_profile(profile: Mapping[str, object]) -> dict[str, object]:
    return {
        name: str(value).replace("\\", "/")
        if name in _PROFILE_PATH_FIELDS and isinstance(value, str) else value
        for name, value in dict(profile).items()
    }


def _merge_options(*values: object) -> str:
    result, positions = [], {}
    for value in values:
        for token in shlex.split(_option_text(value)):
            key = "-O" if re.fullmatch(r"-O\d+", token) else token.split("=", 1)[0] if "=" in token else None
            if key is None:
                result.append(token)
                continue
            if key in positions:
                result[positions[key]] = token
            else:
                positions[key] = len(result)
                result.append(token)
    return " ".join(shlex.quote(token) for token in result)


def riscv_dv_command(
    checkout: str | Path,
    output_dir: str | Path,
    *,
    target: str = "rv64imc",
    test: str = "riscv_arithmetic_basic_test",
    seed: int = 1,
    testlist: str | Path | None = None,
    generation_profile: Mapping[str, object] | None = None,
    compat_runner: str | Path | None = None,
) -> tuple[str, ...]:
    if generation_profile is not None and not isinstance(generation_profile, Mapping):
        raise ValueError("generation_profile must be an object")
    checkout = Path(os.path.expanduser(str(checkout))).resolve()
    output_dir = Path(os.path.expanduser(str(output_dir))).resolve()
    profile = _normalize_generation_profile(generation_profile)
    target = profile.get("target") or target
    test = profile.get("test") or test
    simulator = profile.get("simulator") or "pyflow"
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    if not isinstance(test, str) or not test or test == "all" or "," in test or Path(test).name != test:
        raise ValueError("test must be one RISC-V-DV test")
    if not isinstance(simulator, str) or not simulator:
        raise ValueError("simulator must be a non-empty string")
    testlist = profile.get("testlist", testlist)
    custom_target = profile.get("custom_target")
    if testlist is None and custom_target:
        testlist = Path(os.path.expanduser(str(custom_target))) / "testlist.yaml"
    seed = profile.get("seed", seed)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    iterations = profile.get("iterations", 1)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be positive")
    simulator = simulator.lower()
    sim_opts = _option_text(profile.get("sim_opts"))
    if compat_runner is not None:
        compat_runner = Path(os.path.expanduser(str(compat_runner))).resolve()
        if not compat_runner.is_file():
            raise ValueError("compat_runner must point to an existing file")
        launcher = ("python3", str(compat_runner), str(checkout))
    else:
        launcher = ("python3", str(checkout / "run.py"))
    command = [
        *launcher,
        "--target", target, "--test", test, "--iterations", str(iterations),
        "--start_seed" if iterations > 1 else "--seed", str(seed),
        "--simulator", simulator,
        "--steps", "gen", "--output", str(output_dir),
    ]
    for name, flag in (
        ("isa", "--isa"), ("mabi", "--mabi"), ("custom_target", "--custom_target"),
        ("core_setting_dir", "--core_setting_dir"), ("user_extension_dir", "--user_extension_dir"),
        ("cmp_opts", "--cmp_opts"), ("gcc_opts", "--gcc_opts"),
        ("csr_yaml", "--csr_yaml"), ("priv", "--priv"), ("iss", "--iss"),
        ("iss_opts", "--iss_opts"), ("simulator_yaml", "--simulator_yaml"),
        ("iss_yaml", "--iss_yaml"), ("end_signature_addr", "--end_signature_addr"),
        ("asm_test", "--asm_test"), ("c_test", "--c_test"),
    ):
        value = profile.get(name)
        if name == "mabi" and not value:
            value = _TARGET_MABI.get(target)
        if name in {"cmp_opts", "gcc_opts", "iss_opts"} and isinstance(value, (list, tuple)):
            value = _option_text(value)
        if value not in (None, ""):
            if name == "mabi":
                value = str(value).lower()
            command.extend((flag, str(value)))
    if sim_opts:
        index = command.index("--steps")
        command[index:index] = (f"--sim_opts={sim_opts}",)
    testlist_path = (
        _testlist_path(checkout, target, testlist, custom_target=custom_target)
        if testlist else None
    )
    return tuple(command) + (
        ("--testlist", str(testlist_path)) if testlist_path else ()
    )


def _compat_runner_from_pythonpath(
    pyflow_pythonpath: str | Path | None,
) -> Path | None:
    if pyflow_pythonpath not in (None, ""):
        for item in str(pyflow_pythonpath).split(os.pathsep):
            if not item:
                continue
            candidate = Path(os.path.expanduser(item)).resolve() / "run_compat.py"
            if candidate.is_file():
                return candidate
    bundled = Path(__file__).resolve().parents[2] / "container" / "assets" / "rvdv" / "run_compat.py"
    return bundled if bundled.is_file() else None


def _testlist_path(
    checkout: Path, target: str, testlist: str | Path | None, *,
    custom_target: str | Path | None = None,
) -> Path:
    path = Path(os.path.expanduser(str(testlist))) if testlist else (
        Path(os.path.expanduser(str(custom_target))) / "testlist.yaml" if custom_target
        else Path("target") / target / "testlist.yaml"
    )
    if path.is_absolute() and not path.is_file():
        text = path.as_posix()
        local_root = Path(__file__).resolve().parents[2]
        for prefix in ("/opt/rq1/comparison/", "/experiments/rq1/comparison/"):
            if text.startswith(prefix):
                path = local_root / text.removeprefix(prefix)
                break
    return (path if path.is_absolute() else checkout / path).resolve()


def _testlist_tree(
    path: Path, checkout: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[Mapping[str, object], ...]]:
    files, gaps, entries, seen, active = [], [], [], set(), set()

    def walk(current: Path) -> None:
        current = current.resolve()
        label = str(current.relative_to(checkout)) if current.is_relative_to(checkout) else str(current)
        if current in active:
            gaps.append(f"testlist-import-cyclic:{label}")
            return
        if current in seen:
            return
        if not current.is_file():
            gaps.append(f"testlist-import-missing:{label}")
            return
        seen.add(current)
        files.append(current)
        active.add(current)
        try:
            try:
                raw = yaml.safe_load(current.read_text(encoding="utf-8")) or ()
            except (OSError, UnicodeError, yaml.YAMLError):
                gaps.append(f"testlist-import-invalid:{label}")
                return
            if not isinstance(raw, list):
                gaps.append(f"testlist-import-invalid:{label}")
                return
            for entry in raw:
                if not isinstance(entry, Mapping):
                    continue
                imported = entry.get("import")
                if imported:
                    imported_path = Path(str(imported).replace(
                        "<riscv_dv_root>", str(checkout)
                    ))
                    walk(imported_path if imported_path.is_absolute() else current.parent / imported_path)
                else:
                    entries.append(entry)
        finally:
            active.remove(current)

    walk(path)
    return tuple(files), tuple(dict.fromkeys(gaps)), tuple(entries)


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        **(
            {"sha256": sha256_file(path)} if path.is_file()
            else {"status": "directory" if path.is_dir() else "missing"}
        ),
    }


def _directory_identity(path: Path) -> dict[str, object]:
    if not path.is_dir():
        return _file_identity(path)
    return {
        "path": str(path), "status": "directory",
        "sha256": canonical_digest([
            (item.relative_to(path).as_posix(), sha256_file(item))
            for item in sorted(path.rglob("*")) if item.is_file()
        ]),
    }


def _without_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _without_paths(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_paths(item) for item in value)
    return value


def _run_with_timeout(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess:
    timeout = kwargs.pop("timeout")
    kwargs.pop("check", False)
    if kwargs.pop("capture_output", False):
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.Popen(
        command, start_new_session=os.name != "nt", **kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        pids = {process.pid}
        pending = [process.pid]
        while pending and os.name != "nt":
            parent = pending.pop()
            try:
                children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
            except OSError:
                children = ()
            for child in (int(value) for value in children):
                if child not in pids:
                    pids.add(child)
                    pending.append(child)
        if os.name == "nt":
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        else:
            groups = {os.getpgid(pid) for pid in pids if os.path.exists(f"/proc/{pid}")}
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # Killing the process group does not guarantee that every descendant
        # has closed the generator's pipes. A plain ``communicate()`` here
        # could therefore block forever while the caller is already handling
        # a bounded generation timeout. Give cleanup a short window, then
        # close streams and reap the direct child best-effort.
        try:
            process.communicate(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def preflight_generation_route(
    checkout: str | Path,
    *,
    target: str = "rv64imc",
    isa: str | None = None,
    mabi: str | None = None,
    custom_target: str | Path | None = None,
    test: str = "riscv_arithmetic_basic_test",
    testlist: str | Path | None = None,
    route: str = "multi-instruction",
    simulator: str = "pyflow",
    requirements: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """检查生成表达能力，并把环境/执行资源缺口单独记录。"""
    requirements_valid = requirements is None or isinstance(requirements, Mapping)
    target_valid = isinstance(target, str) and bool(target)
    test_valid = isinstance(test, str)
    route_valid = isinstance(route, str)
    target = target if target_valid else ""
    test = test if test_valid else ""
    route = {"single": "single-instruction", "program": "multi-instruction"}.get(
        route, route
    ) if route_valid else ""
    checkout = Path(os.path.expanduser(str(checkout))).resolve()
    requirements = dict(requirements) if requirements_valid and requirements is not None else {}
    explicit_isa, explicit_mabi = isa not in (None, ""), mabi not in (None, "")
    isa = str(isa if isa not in (None, "") else _TARGET_ISA.get(target, target))
    abi = str(mabi or _TARGET_MABI.get(target) or "").lower()
    abi_match = re.fullmatch(r"(?:ilp32|lp64)([efdq]?)", abi)
    abi_suffix = abi_match[1] if abi_match is not None else ""
    isa_extensions = enabled_extensions(isa) if is_canonical_isa_profile(isa) else ()
    simulator_valid = isinstance(simulator, str) and simulator.lower() in {
        "pyflow", *_SIMULATOR_TOOLS
    }
    simulator = simulator.lower() if simulator_valid else ""
    custom_target_path = Path(os.path.expanduser(str(custom_target))) if custom_target else None
    if custom_target_path and not custom_target_path.is_absolute():
        custom_target_path = checkout / custom_target_path
    testlist_path = _testlist_path(checkout, target, testlist, custom_target=custom_target)
    target_module_path = checkout / "pygen" / "pygen_src" / "target" / target / "riscv_core_setting.py"
    checkout_ready = (checkout / "run.py").is_file()
    checks = {
        "checkout": route == "direct" or checkout_ready,
        "target_module": (
            not checkout_ready or route == "direct" or custom_target_path is not None
            or simulator not in {"vcs", "verilator", "questa", "xrun"}
            and target_module_path.is_file()
            or simulator in {"vcs", "verilator", "questa", "xrun"}
            and (checkout / "target" / target / "riscv_core_setting.sv").is_file()
        ),
        "target_isa": is_canonical_isa_profile(isa),
        "mabi_isa": (
            not abi
            or (abi_match is not None
                and ((isa.lower().startswith("rv32") and abi.startswith("ilp32"))
                     or (isa.lower().startswith("rv64") and abi.startswith("lp64")))
                and (not abi_suffix or abi_suffix in isa_extensions)
                and (abi != "ilp32e" or isa.lower().startswith("rv32e"))
        )),
        "custom_target": custom_target_path is None or custom_target_path.is_dir(),
        "custom_target_isa": custom_target_path is None or explicit_isa,
        "custom_target_mabi": custom_target_path is None or explicit_mabi,
        "custom_target_settings": (
            custom_target_path is None
            or (custom_target_path / "riscv_core_setting.sv").is_file()
        ),
        "test": test_valid and bool(test) and test != "all" and "," not in test and Path(test).name == test,
        "route": route_valid and route in {"single-instruction", "multi-instruction", "direct", "custom", "direct/custom"},
        "target": target_valid,
    }
    transport_gaps = []
    execution_gaps = []
    observer_gaps = []

    def add_gap(gaps: list[str], name: str) -> None:
        if name not in gaps:
            gaps.append(name)

    transport_checks = {
        "checkout": "checkout-missing",
        "target_module": "target-module-missing",
        "custom_target": "custom-target-missing",
        "custom_target_settings": "custom-target-settings-missing",
    }
    for name, gap in transport_checks.items():
        if not checks[name]:
            add_gap(transport_gaps, f"transport-gap:{gap}")
    reasons = [
        name.replace("_", "-")
        for name, ready in checks.items()
        if not ready and name not in transport_checks
    ]
    if not requirements_valid:
        reasons.append("requirements-invalid")
    declared_simulator = requirements.get("simulator")
    if isinstance(declared_simulator, str) and declared_simulator.lower() != simulator:
        reasons.append("simulator-conflict")
    elif declared_simulator is not None and not isinstance(
        declared_simulator, (bool, Mapping, str)
    ):
        reasons.append("simulator-invalid")
    if not simulator_valid:
        reasons.append("simulator")

    testlist_required = requirements.get(
        "testlist_required", testlist not in (None, "") or custom_target_path is not None,
    )
    if "testlist_required" in requirements and type(testlist_required) is not bool:
        reasons.append("testlist-required-invalid")
        testlist_required = False
    if (testlist_required is True
            or requirements.get("test_exists") is True) and not testlist_path.is_file():
        add_gap(transport_gaps, "transport-gap:testlist-missing")
    testlist_entry = None
    if testlist_path.is_file():
        _, import_gaps, entries = _testlist_tree(testlist_path, checkout)
        testlist_entry = next((entry for entry in entries if entry.get("test") == test), None)
        reasons.extend(import_gaps)
    if (
        (requirements.get("test_exists") is True
         or requirements.get("test_entry_conflict") is True
         or testlist_required is True)
        and testlist_path.is_file()
    ):
        if testlist_entry is None:
            reasons.append("test-missing")
        elif requirements.get("test_entry_conflict") is True:
            reasons.append("test-entry-conflict")
        elif not (
            testlist_entry.get("asm_test") not in (None, "")
            or testlist_entry.get("c_test") not in (None, "")
            or isinstance(testlist_entry.get("gen_test"), str)
            and bool(testlist_entry["gen_test"].strip())
        ):
            reasons.append("gen-test-missing")
        elif type(testlist_entry.get("iterations")) is int and testlist_entry.get("iterations") == 0:
            reasons.append("test-disabled")
        elif "iterations" in testlist_entry and (
                type(testlist_entry["iterations"]) is not int
                or testlist_entry["iterations"] < 1
        ):
            reasons.append("iterations-invalid")
    expected_gen_opts = requirements.get("gen_opts")
    if expected_gen_opts not in (None, "") and testlist_path.is_file():
        if testlist_entry is None or _option_text(testlist_entry.get("gen_opts")) != _option_text(expected_gen_opts):
            reasons.append("gen-opts-mismatch")
    if requirements.get("complete_source") is False:
        reasons.append("directed-test-no-complete-source")
    iterations = requirements.get("iterations")
    if iterations is not None and (type(iterations) is not int or iterations < 1):
        reasons.append("iterations-invalid")
    if "seed" in requirements and (
            type(requirements["seed"]) is not int or requirements["seed"] < 0
    ):
        reasons.append("seed-invalid")
    for name, reason in (
        ("test_exists", "test-missing"),
        ("directed_asm_exists", "directed-asm-missing"),
        ("directed_stream_exists", "directed-stream-missing"),
        ("initial_state_ready", "initial-state-not-ready"),
        ("harness_ready", "harness-not-ready"),
        ("entry_ready", "entry-not-ready"),
        ("observer_ready", "observer-not-ready"),
        ("csr_yaml_exists", "csr-yaml-missing"),
        ("simulator_yaml_exists", "simulator-yaml-missing"),
        ("iss_yaml_exists", "iss-yaml-missing"),
        ("asm_test_exists", "asm-test-missing"),
        ("c_test_exists", "c-test-missing"),
        ("core_setting_dir_exists", "core-setting-dir-missing"),
        ("user_extension_dir_exists", "user-extension-dir-missing"),
    ):
        if name in requirements and requirements[name] is not True:
            if name == "observer_ready":
                add_gap(observer_gaps, "observer-gap:observer-not-ready")
            elif name in {"initial_state_ready", "harness_ready", "entry_ready"}:
                execution_gaps.append(f"execution-gap:{reason}")
            elif name == "test_exists" and not testlist_path.is_file():
                continue
            elif name in {
                "directed_asm_exists", "directed_stream_exists", "csr_yaml_exists",
                "simulator_yaml_exists", "iss_yaml_exists", "asm_test_exists",
                "c_test_exists", "core_setting_dir_exists", "user_extension_dir_exists",
            }:
                add_gap(transport_gaps, f"transport-gap:{reason}")
            else:
                reasons.append(reason)

    if "toolchain_ready" in requirements and requirements["toolchain_ready"] is not True:
        add_gap(transport_gaps, "transport-gap:toolchain-missing")
    simulator_missing = (
        "simulator_ready" in requirements
        and requirements["simulator_ready"] is not True
    ) or (
        simulator != "pyflow"
        and shutil.which(_SIMULATOR_TOOLS.get(simulator, simulator)) is None
    )
    checks["simulator"] = simulator_valid and not simulator_missing
    if simulator_missing:
        add_gap(transport_gaps, "transport-gap:simulator-missing")

    conflicts = requirements.get("configuration_conflicts")
    if conflicts not in (None, ""):
        if not isinstance(conflicts, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in conflicts
        ):
            reasons.append("configuration-conflicts-invalid")
        else:
            reasons.extend(f"configuration-conflict:{name}" for name in conflicts)

    required_extensions = requirements.get("required_extensions", ())
    if isinstance(required_extensions, str):
        required_extensions = (required_extensions,)
    elif not isinstance(required_extensions, (list, tuple)) or any(
        not isinstance(extension, str) or not extension for extension in required_extensions
    ):
        reasons.append("required-extensions-invalid")
        required_extensions = ()
    for extension in required_extensions:
        if extension not in enabled_extensions(isa):
            reasons.append(f"extension-missing:{extension}")
    if route == "single-instruction" and requirements.get("directed_asm_exists") is not True:
        reasons.append("single-route-requires-directed-source")
    if route == "direct" and requirements.get("directed_asm_exists") is not True:
        reasons.append("direct-route-requires-directed-source")
    if route == "multi-instruction" and requirements.get("directed_asm_exists") is True:
        reasons.append("multi-route-directed-source-conflict")
    # The generator profile is allowed to exceed the historical target
    # default.  If the eventual simulator cannot execute an extension, its
    # process result becomes a per-case gap; generation must still proceed.
    generation_transport_gaps = [
        gap for gap in transport_gaps
        if gap in {
            "transport-gap:checkout-missing",
            "transport-gap:target-module-missing",
            "transport-gap:custom-target-missing",
            "transport-gap:custom-target-settings-missing",
            "transport-gap:testlist-missing",
        }
    ]
    resource_gaps = [*transport_gaps, *execution_gaps, *observer_gaps]
    generation_ready = not reasons and not generation_transport_gaps
    status = "generation-gap" if reasons else "transport-gap" if resource_gaps else "ready"
    return {
        "contract": "generation-route-preflight-v1",
        "status": status,
        "checkout": str(checkout),
        "target": target,
        "custom_target": str(custom_target_path) if custom_target_path else None,
        "isa": isa,
        "mabi": abi,
        "test": test,
        "testlist": str(testlist_path),
        "route": route,
        "simulator": simulator,
        "requirements": requirements,
        "checks": checks,
        "reasons": reasons,
        "transport_gaps": transport_gaps,
        "execution_gaps": execution_gaps,
        "observer_gaps": observer_gaps,
        "resource_gaps": resource_gaps,
        "generation_gaps": [*reasons, *generation_transport_gaps],
        "generation_ready": generation_ready,
        "reason": (
            "zero-probability" if reasons
            else resource_gaps[0] if resource_gaps
            else None
        ),
        "details": [*reasons, *resource_gaps],
    }


def generate_program(
    checkout: str | Path,
    output_dir: str | Path,
    *,
    target: str = "rv64imc",
    test: str = "riscv_arithmetic_basic_test",
    seed: int = 1,
    testlist: str | Path | None = None,
    pyflow_pythonpath: str | Path | None = None,
    generation_profile: Mapping[str, object] | None = None,
    initial_state: Mapping[str, object] | None = None,
    total_budget_sec: float | None = None,
    execute: Callable[..., object] = subprocess.run,
) -> CaseProgram:
    """运行官方生成阶段；真实执行由外层 Linux 平面提供。"""
    total_budget_started = time.monotonic() if total_budget_sec is not None else None
    if generation_profile is not None and not isinstance(generation_profile, Mapping):
        raise ValueError("generation_profile must be an object")
    if generation_profile is not None and "requirements" in generation_profile and not isinstance(
        generation_profile["requirements"], Mapping
    ):
        raise ValueError("generation_profile.requirements must be an object")
    if initial_state is not None and not isinstance(initial_state, Mapping):
        raise ValueError("initial_state must be an object")
    checkout = Path(os.path.expanduser(str(checkout))).resolve()
    output_dir = Path(os.path.expanduser(str(output_dir))).resolve()
    generation_profile = _normalize_generation_profile(generation_profile)
    for name in ("target", "test", "simulator"):
        if name in generation_profile and (
            not isinstance(generation_profile[name], str) or not generation_profile[name]
        ):
            raise ValueError(f"generation_profile.{name} must be a non-empty string")
    target = str(generation_profile.get("target") or target)
    test = str(generation_profile.get("test") or test)
    profile_isa = generation_profile.get("isa")
    isa = str(profile_isa if profile_isa not in (None, "") else _TARGET_ISA.get(target, target))
    mabi = str(generation_profile.get("mabi") or _TARGET_MABI.get(target, "")).lower()
    custom_target = generation_profile.get("custom_target")
    seed = generation_profile.get("seed", seed)
    iterations = generation_profile.get("iterations", 1)
    simulator = str(generation_profile.get("simulator") or "pyflow").lower()
    harness = str(generation_profile.get("harness") or (
        "custom" if generation_profile.get("asm_test") not in (None, "") else "riscv-dv"
    ))
    if harness not in {"custom", "riscv-dv"}:
        raise ValueError("generation_profile.harness must be custom or riscv-dv")
    testlist = generation_profile.get("testlist", testlist)
    directed_source = generation_profile.get("asm_test")
    if directed_source not in (None, ""):
        directed_source = Path(os.path.expanduser(str(directed_source)))
        if not directed_source.is_absolute():
            directed_source = checkout / directed_source
        directed_source = directed_source.resolve()
        generation_profile["asm_test"] = str(directed_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    testlist = _testlist_path(checkout, target, testlist, custom_target=custom_target)
    testlist_files, testlist_import_gaps, testlist_entries = _testlist_tree(
        testlist, checkout,
    )
    testlist_entry = next(
        (entry for entry in testlist_entries if entry.get("test") == test), None,
    )
    test_entry_conflict = (
        directed_source is None
        and isinstance(testlist_entry, Mapping)
        and (
            testlist_entry.get("asm_test") not in (None, "")
            or testlist_entry.get("c_test") not in (None, "")
        )
        and testlist_entry.get("gen_test") not in (None, "")
    )
    entry_iterations = (
        testlist_entry.get("iterations", 1)
        if isinstance(testlist_entry, Mapping) else 1
    )
    if (directed_source is None and isinstance(testlist_entry, Mapping)
            and testlist_entry.get("asm_test") not in (None, "")
            and type(entry_iterations) is int and entry_iterations > 0):
        directed_source = Path(os.path.expanduser(str(testlist_entry["asm_test"])))
        if not directed_source.is_absolute():
            directed_source = checkout / directed_source
        directed_source = directed_source.resolve()
        generation_profile["asm_test"] = str(directed_source)
    if (directed_source is None and "iterations" not in generation_profile
            and isinstance(testlist_entry, Mapping)
            and "iterations" in testlist_entry
            and type(entry_iterations) is int and entry_iterations > 0):
        iterations = entry_iterations
        generation_profile["iterations"] = iterations
    testlist_source = testlist
    testlist_options = ("gen_opts", "sim_opts", "cmp_opts", "gcc_opts")
    source_options = {
        name: _option_text(testlist_entry.get(name))
        if isinstance(testlist_entry, Mapping) else ""
        for name in testlist_options
    }
    effective_options = {
        name: _merge_options(
            source_options[name],
            generation_profile.get(name),
            generation_profile.get("directed_stream") if name == "gen_opts" else None,
        )
        for name in testlist_options
    }
    gcc_opts = effective_options["gcc_opts"]
    overlay_options = isinstance(testlist_entry, Mapping) and (
        any(effective_options[name] != source_options[name] for name in testlist_options)
        or any(
            generation_profile.get(name) not in (None, "")
            for name in ("sim_opts", "cmp_opts", "gcc_opts")
        )
    )
    if directed_source is None and overlay_options:
        testlist = output_dir / "testlist.yaml"
        entry = {**dict(testlist_entry), "test": test}
        for name, value in effective_options.items():
            if value:
                entry[name] = value
        testlist.write_text(yaml.safe_dump([entry], sort_keys=False), encoding="utf-8")
    requirements = {
        "testlist_required": directed_source is None,
        "iterations": iterations,
        "seed": seed,
        "simulator": simulator,
        "test_entry_conflict": test_entry_conflict,
    }
    if generation_profile.get("c_test") not in (None, "") or (
        isinstance(testlist_entry, Mapping) and "c_test" in testlist_entry
    ):
        requirements["complete_source"] = False
    if directed_source is not None:
        requirements["directed_asm_exists"] = (
            directed_source.is_file() and directed_source.suffix.lower() == ".s"
        )
    if directed_source is None and testlist.is_file():
        requirements["test_exists"] = True
    if directed_source is None and effective_options["gen_opts"]:
        requirements["gen_opts"] = effective_options["gen_opts"]
    profile_requirements = generation_profile.get("requirements")
    configuration_conflicts = []
    if profile_requirements is not None:
        for name, value in profile_requirements.items():
            if name in {
                "testlist_required", "iterations", "seed", "simulator",
                "test_entry_conflict", "complete_source",
            } and name in requirements and requirements[name] != value:
                configuration_conflicts.append(name)
                continue
            requirements[name] = value
    if configuration_conflicts:
        requirements["configuration_conflicts"] = tuple(dict.fromkeys(configuration_conflicts))
    if isinstance(generation_profile.get("route"), str):
        requirements["route"] = generation_profile["route"]
    profile_files = {"testlist": _file_identity(testlist)}
    if testlist != testlist_source:
        profile_files["testlist_source"] = _file_identity(testlist_source)
    profile_files["testlist_imports"] = {
        "status": (
            "missing" if any(gap.startswith("testlist-import-missing:") for gap in testlist_import_gaps)
            else "invalid" if testlist_import_gaps else "ready" if testlist_files else "missing"
        ),
        "gaps": list(testlist_import_gaps),
        "sha256": canonical_digest([
            (
                str(path.relative_to(checkout).as_posix())
                if path.is_relative_to(checkout) else str(path),
                sha256_file(path),
            )
            for path in testlist_files
        ]) if testlist_files else None,
    }
    compat_runner = _compat_runner_from_pythonpath(pyflow_pythonpath)
    generator_source_items = {
        "run": _file_identity(checkout / "run.py"),
        "pygen_src": _directory_identity(checkout / "pygen" / "pygen_src"),
        "scripts": _directory_identity(checkout / "scripts"),
    }
    if compat_runner is not None:
        generator_source_items["compat_runner"] = _file_identity(compat_runner)
    profile_files["generator_sources"] = {
        "status": "ready" if all(item.get("sha256") for item in generator_source_items.values()) else "missing",
        "sha256": canonical_digest(generator_source_items),
    }
    if pyflow_pythonpath:
        pyflow_paths = tuple(
            Path(os.path.expanduser(item)).resolve()
            for item in str(pyflow_pythonpath).split(os.pathsep) if item
        )
        profile_files["pyflow_pythonpath"] = {
            "status": "ready" if all(path.is_dir() for path in pyflow_paths) else "missing",
            "entries": [_directory_identity(path) for path in pyflow_paths],
        }
    for name in (
        "core_setting_dir", "user_extension_dir",
        "csr_yaml", "simulator_yaml", "iss_yaml", "asm_test", "c_test",
    ):
        value = generation_profile.get(name)
        if value not in (None, ""):
            path = Path(os.path.expanduser(str(value)))
            identity = _directory_identity if name.endswith("_dir") else _file_identity
            profile_files[name] = identity(
                (checkout / path if not path.is_absolute() else path).resolve()
            )
            if profile_files[name].get("status") == "missing":
                requirements[f"{name}_exists"] = False
    if custom_target not in (None, ""):
        path = Path(os.path.expanduser(str(custom_target)))
        custom_target_path = (checkout / path if not path.is_absolute() else path).resolve()
        profile_files["custom_target"] = _directory_identity(custom_target_path)
        profile_files["custom_target_settings"] = _file_identity(
            custom_target_path / "riscv_core_setting.sv"
        )
    identity_profile_digest = canonical_digest({
        name: value for name, value in generation_profile.items()
        if name not in _PROFILE_PATH_FIELDS
    })
    command = ("direct-copy", str(directed_source)) if directed_source is not None else ()
    toolchain_timeout = None
    if total_budget_sec is not None:
        try:
            if isinstance(total_budget_sec, bool):
                raise ValueError("timeout must be numeric")
            total_budget = float(total_budget_sec)
            toolchain_timeout = (
                max(0.0, total_budget - (time.monotonic() - total_budget_started))
                if math.isfinite(total_budget) else 0.0
            )
        except (OverflowError, TypeError, ValueError):
            # The normal deadline validation below records invalid budgets.
            toolchain_timeout = 0.0
    toolchain_identity = _toolchain_identity(toolchain_timeout)
    if execute is subprocess.run and directed_source is None:
        requirements.setdefault(
            "toolchain_ready", toolchain_identity.get("status") == "ready"
        )
        if simulator != "pyflow":
            requirements.setdefault(
                "simulator_ready",
                shutil.which(_SIMULATOR_TOOLS.get(simulator, simulator)) is not None,
            )
    route = (requirements.get("route") if isinstance(requirements.get("route"), str)
             else "direct" if directed_source is not None else "multi-instruction")
    if directed_source is not None and route == "program":
        route = "direct"
    route_preflight = preflight_generation_route(
        checkout, target=target,
        isa=generation_profile.get("isa") if custom_target else isa,
        mabi=generation_profile.get("mabi") if custom_target else mabi,
        custom_target=custom_target,
        test=test, testlist=testlist,
        route=route,
        simulator=simulator,
        requirements=requirements,
    )
    generation_ready = route_preflight["generation_ready"]
    command_profile = dict(generation_profile)
    if overlay_options:
        for name in ("sim_opts", "cmp_opts", "gcc_opts"):
            command_profile.pop(name, None)
    if generation_ready and directed_source is None:
        command = riscv_dv_command(
            checkout, output_dir, target=target, test=test, seed=seed,
            testlist=testlist, generation_profile=command_profile,
            compat_runner=compat_runner,
        ) if type(iterations) is int and iterations > 0 else ()
    generation_profile_digest = canonical_digest(_portable_generation_profile(generation_profile))
    manifest_base = {
        "contract": "riscv-dv-program-v1",
        "target": target,
        "isa": isa,
        "mabi": mabi,
        "test": test,
        "testlist": str(testlist),
        "command": list(command),
        "seed": seed,
        "iterations": iterations,
        "simulator": simulator,
        "generation_profile": generation_profile,
        "resolved_options": {
            name: value for name, value in effective_options.items() if value
        },
        "generation_profile_digest": generation_profile_digest,
        "generation_profile_files": profile_files,
        "execution_plane": os.environ.get("MIGRATION_EXECUTION_PLANE"),
        "container_image": os.environ.get("MIGRATION_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("MIGRATION_CONTAINER_IMAGE_DIGEST"),
        "toolchain_identity": toolchain_identity,
        "route_preflight": route_preflight,
    }

    def write_gap(reason: str, *, status: str = "generation-gap", **details: object) -> None:
        atomic_write_json(output_dir / "program-manifest.json", {
            **manifest_base, "status": status, "reason": reason, **details,
        })

    if type(iterations) is not int or iterations < 1:
        write_gap("iterations-invalid")
        raise RuntimeError("generation-gap: iterations-invalid")
    if not generation_ready:
        reason = route_preflight["reason"] or route_preflight["status"]
        write_gap(reason, status=route_preflight["status"])
        detail = route_preflight["reasons"] or route_preflight["resource_gaps"] or [reason]
        raise RuntimeError(f"{route_preflight['status']}: {reason} ({detail[0]})")
    if execute is subprocess.run and directed_source is None:
        try:
            if os.name == "nt":
                raise RuntimeError("RISC-V-DV generation must run in the Linux execution plane")
            require_execution_plane()
        except (ExecutionEnvironmentError, RuntimeError) as error:
            write_gap(
                "transport-gap:execution-plane-missing", status="transport-gap",
                error=str(error),
            )
            raise
    program = output_dir / "asm_test" / f"{test}_0.S"
    if program.exists() and (directed_source is None or program.resolve() != directed_source):
        program.unlink()
    if directed_source is None or directed_source.parent != program.parent:
        for name in ("user_init.s", "user_define.h"):
            dependency_output = program.with_name(name)
            if dependency_output.exists() or dependency_output.is_symlink():
                dependency_output.unlink()
    env = os.environ.copy()
    pythonpath = [
        str(Path(os.path.expanduser(item)).resolve())
        for item in (
            str(pyflow_pythonpath).split(os.pathsep)
            if pyflow_pythonpath else ()
        )
        if item
    ]
    pythonpath.append(str((checkout / "pygen").resolve()))
    if env.get("PYTHONPATH"):
        pythonpath.extend(item for item in env["PYTHONPATH"].split(os.pathsep) if item)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if isinstance(toolchain_identity.get("path"), str):
        env.setdefault("RISCV_GCC", toolchain_identity["path"])
    objcopy = os.environ.get("RISCV_OBJCOPY") or shutil.which("riscv64-linux-gnu-objcopy")
    if objcopy:
        env.setdefault("RISCV_OBJCOPY", objcopy)
    run_kwargs = {"cwd": str(checkout), "capture_output": True, "text": True, "check": False, "env": env}
    budget_exhausted = False
    if generation_profile.get("timeout_sec") not in (None, ""):
        try:
            raw_timeout = generation_profile["timeout_sec"]
            if isinstance(raw_timeout, bool):
                raise ValueError("timeout must be numeric")
            timeout = float(raw_timeout)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite")
            run_kwargs["timeout"] = timeout
        except (OverflowError, TypeError, ValueError) as error:
            write_gap("timeout-invalid", command=list(command), error=str(error))
            raise ValueError("generation-gap: timeout-invalid") from error
    if total_budget_sec is not None:
        try:
            if isinstance(total_budget_sec, bool):
                raise ValueError("timeout must be numeric")
            total_budget = float(total_budget_sec)
            if not math.isfinite(total_budget):
                raise ValueError("timeout must be finite")
            remaining_budget = total_budget - (time.monotonic() - total_budget_started)
            if remaining_budget <= 0:
                budget_exhausted = True
            else:
                run_kwargs["timeout"] = min(
                    float(run_kwargs.get("timeout", remaining_budget)), remaining_budget,
                )
        except (OverflowError, TypeError, ValueError) as error:
            write_gap("timeout-invalid", command=list(command), error=str(error))
            raise ValueError("generation-gap: timeout-invalid") from error
    if directed_source is None:
        try:
            if budget_exhausted:
                raise subprocess.TimeoutExpired(command, timeout=0)
            runner = _run_with_timeout if execute is subprocess.run and "timeout" in run_kwargs else execute
            result = runner(command, **run_kwargs)
        except subprocess.TimeoutExpired as error:
            write_gap("generator-timeout", command=list(command), error=str(error))
            raise
        except (OSError, ExecutionEnvironmentError) as error:
            write_gap(
                "generator-transport", status="transport-gap",
                command=list(command), error=str(error),
            )
            raise
        except Exception as error:
            write_gap("generator-exception", command=list(command), error=str(error))
            raise
        if getattr(result, "returncode", 1) != 0:
            error = str(getattr(result, "stderr", "") or getattr(result, "stdout", ""))
            transport = (
                re.search(
                    r"(?:command not found|(?:exec:\s+)?\S+:\s+not found)",
                    error, re.IGNORECASE,
                ) is not None
            )
            write_gap(
                "generator-transport" if transport else "generator-failed",
                status="transport-gap" if transport else "generation-gap",
                command=list(command), error=error,
            )
            raise RuntimeError(
                ("transport-gap: generator unavailable: " if transport
                 else "RISC-V-DV generation failed: ") + error
            )
    if directed_source is not None and directed_source.is_file():
        program.parent.mkdir(parents=True, exist_ok=True)
        if program.resolve() != directed_source:
            shutil.copyfile(directed_source, program)
    if not program.is_file() or not program.read_bytes().strip():
        write_gap("program-missing", command=list(command), program=str(program))
        raise RuntimeError(f"RISC-V-DV generation did not produce the complete program: {program}")
    program = program.resolve()
    commit = git_head(checkout)
    if not isinstance(commit, str) or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ):
        write_gap("source-identity-missing", command=list(command), program=str(program))
        raise RuntimeError("generation-gap: RISC-V-DV checkout has no Git source identity")
    source_dependencies = {}
    try:
        source_text = program.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        write_gap(
            "program-source-invalid", command=list(command), program=str(program),
            error=str(error),
        )
        raise RuntimeError("generation-gap: invalid generated source") from error
    expanded_user_init = False
    include_pattern = re.compile(
        r'(?m)^[ \t]*(?P<labels>(?:(?:[A-Za-z_.$][\w.$]*|\d+)[ \t]*:[ \t]*)*)'
        r'(?:\.include|#\s*include)[ \t]+"(?P<name>[^"]+)"'
        r'(?P<tail>[^\r\n]*)(?:\r?\n|$)'
    )

    def expand_includes(text: str, resolver: Callable[[re.Match[str], int], str | None]) -> str:
        lines, result = text.splitlines(keepends=True), []
        for start, end, logical in _logical_source_entries(lines):
            line_no = start + 1
            if end > start:
                parsed_line = _strip_c_comments(logical)
                statements = _split_source_statements(parsed_line)
                if len(statements) != 1 or not (match := include_pattern.fullmatch(statements[0])):
                    result.extend(lines[start:end + 1])
                    continue
                replacement = resolver(match, line_no)
                if replacement is None:
                    result.extend(lines[start:end + 1])
                    continue
                newline = lines[end][len(lines[end].rstrip("\r\n")):]
                result.append(replacement + newline)
                continue
            raw_line = lines[start]
            parsed_line = _strip_c_comments(raw_line)
            replacements, cursor = [], 0
            for statement in _split_source_statements(parsed_line):
                end = cursor + len(statement)
                match = include_pattern.fullmatch(statement)
                replacement = resolver(match, line_no) if match else None
                if replacement is not None:
                    replacements.append((cursor, end, replacement))
                cursor = end
            for start, end, replacement in reversed(replacements):
                raw_line = raw_line[:start] + replacement + raw_line[end:]
            result.append(raw_line)
        return "".join(result)

    def expand_named_includes(text: str, wanted: str, content: str) -> str:
        def replace(match: re.Match[str], line_no: int) -> str | None:
            if match.group("name") != wanted:
                return None
            labels, tail = match.group("labels").strip(), match.group("tail")
            return (
                (f"{labels}\n" if labels else "")
                + f"#line {line_no}\n{content.rstrip(chr(10) + chr(13))}\n"
                + f"#line {line_no + 1}\n"
                + (f"{tail}\n" if tail else "")
            )
        return expand_includes(text, replace)

    def include_matches(text: str):
        for _, _, line in _logical_source_entries(
            _strip_c_comments(text).splitlines(keepends=True)
        ):
            for statement in _split_source_statements(line):
                if match := include_pattern.fullmatch(statement):
                    yield match

    scanned_dependencies = set()
    extension_dir = generation_profile.get("user_extension_dir")
    if extension_dir not in (None, ""):
        extension_dir = Path(os.path.expanduser(str(extension_dir)))
        if not extension_dir.is_absolute():
            extension_dir = checkout / extension_dir
        extension_dir = extension_dir.resolve()
    direct_source_dir = directed_source.parent if directed_source is not None else None

    def find_dependency(name: str, origin: Path) -> Path | None:
        try:
            origin.resolve().relative_to(output_dir)
        except ValueError:
            origin_dirs = (origin.parent,)
        else:
            origin_dirs = ()
        directories = (
            *origin_dirs,
            *((extension_dir,) if extension_dir else ()),
            *((direct_source_dir,) if direct_source_dir else ()),
            checkout / "user_extension", checkout, checkout / "target" / target,
            *((origin.parent,) if not origin_dirs else ()),
        )
        candidates = tuple(directory / name for directory in directories)
        return next((path for path in candidates if path.is_file()), None)

    def scan_dependency(
        name: str, dependency: Path | None, destination_dir: Path | None = None,
        stack: tuple[Path, ...] = (),
    ) -> None:
        destination_dir = program.parent if destination_dir is None else destination_dir
        destination = destination_dir / name
        try:
            destination.resolve().relative_to(output_dir)
        except ValueError as error:
            write_gap(
                "source-dependency-outside-output", command=list(command),
                program=str(program), source_dependencies=source_dependencies,
                dependency=name,
            )
            raise RuntimeError("generation-gap: source dependency outside output") from error
        try:
            dependency_key = destination.resolve().relative_to(program.parent.resolve()).as_posix()
        except ValueError:
            dependency_key = name
        if dependency is None:
            source_dependencies[dependency_key] = {"status": "missing", "sha256": None}
            if destination.is_file() or destination.is_symlink():
                destination.unlink()
            return
        dependency = dependency.resolve()
        if dependency in stack:
            source_dependencies[dependency_key] = {"status": "cyclic", "sha256": None}
            return
        dependency_sha256 = sha256_file(dependency)
        existing = source_dependencies.get(dependency_key)
        if (isinstance(existing, Mapping)
                and existing.get("sha256") not in (None, dependency_sha256)):
            write_gap(
                "source-dependency-collision", command=list(command),
                program=str(program), source_dependencies=source_dependencies,
                dependency=dependency_key,
            )
            raise RuntimeError(
                "generation-gap: source dependency basename collision: "
                + dependency_key
            )
        source_dependencies[dependency_key] = {
            "status": "ready", "sha256": dependency_sha256,
        }
        if dependency.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dependency, destination)
        location = (dependency, destination.parent.resolve())
        if location in scanned_dependencies:
            return
        scanned_dependencies.add(location)
        try:
            dependency_source = dependency.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as error:
            write_gap(
                "source-dependency-invalid", command=list(command),
                program=str(program), source_dependencies=source_dependencies,
                error=str(error),
            )
            raise RuntimeError("generation-gap: invalid source dependency") from error
        for match in include_matches(dependency_source):
            scan_dependency(
                match.group("name"), find_dependency(match.group("name"), dependency),
                destination.parent, (*stack, dependency),
            )

    for name in dict.fromkeys(
        match.group("name") for match in include_matches(source_text)
    ):
        dependency = find_dependency(name, program)
        scan_dependency(name, dependency)
        if posixpath.normpath(name) == "user_init.s" and dependency is not None:
            def expand_nested(text: str, origin: Path, stack: tuple[Path, ...]) -> str:
                def expand(match: re.Match[str], line_no: int) -> str | None:
                    nested = match.group("name")
                    nested_path = find_dependency(nested, origin)
                    scan_dependency(nested, nested_path)
                    if nested_path is None or not nested.lower().endswith(".s"):
                        return None
                    nested_path = nested_path.resolve()
                    if nested_path in stack:
                        raise RuntimeError("generation-gap: cyclic source dependency")
                    nested_text = expand_nested(
                        nested_path.read_bytes().decode("utf-8"),
                        nested_path, (*stack, nested_path),
                    ).rstrip("\r\n")
                    labels = match.group("labels").strip()
                    tail = match.group("tail")
                    return (f"{labels}\n" if labels else "") + f"#line {line_no}\n{nested_text}\n#line {line_no + 1}\n" + (
                        f"{tail}\n" if tail else ""
                    )
                return expand_includes(text, expand)

            try:
                content = expand_nested(
                    dependency.read_text(encoding="utf-8"), dependency, (dependency.resolve(),)
                )
            except Exception as error:
                write_gap(
                    "source-dependency-failed", command=list(command),
                    program=str(program), error=str(error),
                )
                raise
            source_text = expand_named_includes(source_text, name, content)
            expanded_user_init = True
    if expanded_user_init:
        program.write_text(source_text, encoding="utf-8")
    if directed_source is not None and re.search(r"(?m)^\s*\.macro\b", source_text):
        source_text = _expand_asm_macros(source_text, line_directives=False)
        program.write_text(source_text, encoding="utf-8")
    missing_dependencies = tuple(
        name for name, item in source_dependencies.items()
        if item.get("status") == "missing"
    )
    cyclic_dependencies = tuple(
        name for name, item in source_dependencies.items()
        if item.get("status") == "cyclic"
    )
    if missing_dependencies:
        write_gap(
            "source-dependency-missing", command=list(command), program=str(program),
            source_dependencies=source_dependencies,
            missing_dependencies=missing_dependencies,
        )
        raise RuntimeError(
            "generation-gap: missing source dependency: "
            + ", ".join(missing_dependencies)
        )
    if cyclic_dependencies:
        write_gap(
            "source-dependency-cyclic", command=list(command), program=str(program),
            source_dependencies=source_dependencies,
            cyclic_dependencies=cyclic_dependencies,
        )
        raise RuntimeError(
            "generation-gap: cyclic source dependency: "
            + ", ".join(cyclic_dependencies)
        )
    program_params = {
        "isa": isa,
        **({"mabi": mabi} if mabi else {}),
        "target": target,
        **({"qemu_cpu": generation_profile["qemu_cpu"]}
           if generation_profile.get("qemu_cpu") else {}),
        "harness": harness,
        "result_channel": "RVOBS1",
        **({"compiler": toolchain_identity["path"]}
           if isinstance(toolchain_identity.get("path"), str) else {}),
        **({"gcc_opts": gcc_opts} if gcc_opts else {}),
        **({"initial_state": dict(initial_state)} if initial_state is not None else {}),
    }
    program_identity = CaseProgram(program, program_params)
    program_sha256 = canonical_program_sha256(program_identity)
    generation_identity = {
        "contract": "riscv-dv-generation-identity-v1",
        "commit": commit, "target": target, "isa": isa, "mabi": mabi,
        "test": test, "seed": seed, "iterations": iterations,
        "simulator": simulator, "generation_profile_digest": identity_profile_digest,
        "profile_files": _without_paths(profile_files),
        "toolchain": _without_paths(toolchain_identity),
        "program_sha256": program_sha256,
        "source_dependencies": _without_paths(source_dependencies),
        "execution_plane": os.environ.get("MIGRATION_EXECUTION_PLANE"),
        "container_image": os.environ.get("MIGRATION_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("MIGRATION_CONTAINER_IMAGE_DIGEST"),
        **({"initial_state_digest": canonical_digest(dict(initial_state))}
           if initial_state is not None else {}),
    }
    provenance = {
        "contract": "riscv-dv-program-v1",
        "status": "generated",
        "checkout": str(checkout),
        "commit": commit,
        "command": list(command),
        "target": target,
        "isa": isa,
        "mabi": mabi,
        "custom_target": str(custom_target) if custom_target else None,
        "testlist": str(testlist),
        "test": test,
        "seed": seed,
        "iterations": iterations,
        "simulator": simulator,
        "steps": "direct-copy" if directed_source is not None else "gen",
        "pyflow_pythonpath": env["PYTHONPATH"],
        "program": str(program),
        "program_sha256": program_sha256,
        "testlist_sha256": sha256_file(testlist) if testlist.is_file() else None,
        "generation_profile": generation_profile,
        "resolved_options": {
            name: value for name, value in effective_options.items() if value
        },
        "generation_profile_digest": generation_profile_digest,
        "generation_profile_files": profile_files,
        "generation_identity": generation_identity,
        "generation_identity_digest": canonical_digest(generation_identity),
        "initial_state_digest": generation_identity.get("initial_state_digest"),
        "harness": harness,
        "result_channel": "RVOBS1",
        "toolchain_identity": toolchain_identity,
        "manifest_path": str((output_dir / "program-manifest.json").resolve()),
        "execution_plane": os.environ.get("MIGRATION_EXECUTION_PLANE"),
        "container_image": os.environ.get("MIGRATION_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("MIGRATION_CONTAINER_IMAGE_DIGEST"),
        "source_dependencies": source_dependencies,
        "route_preflight": route_preflight,
    }
    atomic_write_json(output_dir / "program-manifest.json", provenance)
    return program_identity


__all__ = [
    "CaseProgram", "generate_program", "preflight_generation_route",
    "riscv_dv_command",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil

from .._util import is_sha256_digest
from ..spec_definedness import (
    enabled_extensions,
    isa_profile_for_extensions,
    is_canonical_isa_profile,
    profile_requires_fp_state,
)


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    fp_rawbits: str
    runner_entrypoint: str
    runner_entrypoint_kind: str
    runner_mode: str | None
    runner_binary_option: str | None
    runner_pass_backend: bool
    dependency_family: str = ""
    execution_model: str = ""
    trap_channel: str = ""


EXECUTION_SPECS = (
    BackendSpec("qemu-riscv64", "observed", "qemu_trace_runner.py", "script", None, "--qemu-bin", True, "qemu", "linux-user", "signal"),
    BackendSpec("libriscv-translated", "observed", "libriscv_trace_runner.py", "script", None, "--rvlinux-bin", False, "libriscv", "linux-user", "signal"),
    BackendSpec("rvvm-riscv64", "observed", "framework.adapters.rvvm_runner", "module", None, "--rvvm", False, "rvvm", "bare-metal-capsule", "gdb-rsp"),
    BackendSpec("unicorn-riscv64", "observed", "framework.adapters.capsule_dispatch", "module", None, "--unicorn-bin", False, "unicorn", "bare-metal-capsule", "capsule-mailbox"),
    BackendSpec("renode-riscv64", "observed", "framework.adapters.capsule_dispatch", "module", None, "--renode-bin", False, "renode", "bare-metal-capsule", "capsule-mailbox"),
    # The RAX compatibility build emits the shared RVOBS1 mailbox at guest
    # shutdown.  Its executed-PC stream is still unavailable, but architectural
    # state and the terminal frame are complete for the comparison contract.
    BackendSpec("rax-riscv64", "observed", "framework.adapters.capsule_dispatch", "module", None, "--rax-bin", False, "rax", "bare-metal-capsule", "capsule-mailbox"),
)
EXECUTION_BACKENDS = tuple(spec.backend for spec in EXECUTION_SPECS)
CAPSULE_BACKENDS = ("rvvm-riscv64", "unicorn-riscv64", "renode-riscv64", "rax-riscv64")
_BY_BACKEND = {spec.backend: spec for spec in EXECUTION_SPECS}
_AUXILIARY_SPECS = {
    "libriscv-interpreter": BackendSpec(
        "libriscv-interpreter", "observed", "libriscv_trace_runner.py", "script",
        "interpreter", "--rvlinux-bin", False, "libriscv", "linux-user", "signal",
    ),
}
# Translation path expected by the program adapter's evidence projection.
_STRUCTURED_PATH_BY_BACKEND = {
    "qemu-riscv64": "tcg", "libriscv-translated": "translated",
}
_EXECUTION_BACKEND_ALIASES = {"qemu-riscv32": "qemu-riscv64"}
_QEMU_BACKENDS = frozenset(_EXECUTION_BACKEND_ALIASES) | {"qemu-riscv64"}
_REFERENCE_BACKENDS = frozenset({
    "native-rv64", "sail-riscv", "rvgen-semantic", "qemu", "qemu-riscv32", "qemu-riscv64",
})

def canonical_execution_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise ValueError(f"unknown execution backend: {backend}")
    return _EXECUTION_BACKEND_ALIASES.get(backend, backend)


def same_execution_backend(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if left in _QEMU_BACKENDS or right in _QEMU_BACKENDS:
        return left == right
    return canonical_execution_backend(left) == canonical_execution_backend(right)


def is_execution_backend(backend: object) -> bool:
    return isinstance(backend, str) and (
        backend in EXECUTION_BACKENDS
        or backend in _EXECUTION_BACKEND_ALIASES
        or backend in _AUXILIARY_SPECS
    )


def backend_spec(backend: str) -> BackendSpec:
    if not isinstance(backend, str):
        raise ValueError(f"unknown execution backend: {backend}")
    try:
        canonical = canonical_execution_backend(backend)
        return _BY_BACKEND.get(canonical) or _AUXILIARY_SPECS[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown execution backend: {backend}") from exc


# Capability contract for the pinned Renode target used by this experiment.
# Rejecting an unsupported CPU extension before launch turns an illegal guest
# into a target gap instead of spending the complete target timeout.
_RENODE_SUPPORTED_EXTENSIONS = frozenset({
    "e", "i", "m", "a", "f", "d", "c", "s", "u", "v", "b", "g",
    "zba", "zbb", "zbc", "zbs", "zicsr", "zifencei", "zfh", "zvfh",
    "zve32x", "zve32f", "zve64x", "zve64f", "zve64d", "zacas",
    "smepmp", "sscofpmf",
})


def capsule_cpu_profile(isa_profile: str) -> str | None:
    profile = str(isa_profile or "").strip().lower()
    if not profile.startswith(("rv32", "rv64")):
        return None
    enabled = set(enabled_extensions(profile))
    if "g" in enabled:
        enabled.discard("g")
        enabled.update(("i", "m", "a", "f", "d"))
        if "_zicsr" not in profile:
            enabled.discard("zicsr")
    if "zfhmin" in enabled:
        enabled.discard("zfhmin")
        enabled.add("zfh")
    if not enabled <= _RENODE_SUPPORTED_EXTENSIONS:
        return None
    if "i" not in enabled:
        return None
    try:
        return isa_profile_for_extensions(profile[:4], frozenset(enabled))
    except ValueError:
        return None


PATH_IDENTITY_WITNESS_CONTRACT = "rax-path-identity-witness-v1"
PATH_IDENTITY_WITNESS_STATUS_UNAVAILABLE = "unavailable"
PATH_IDENTITY_WITNESS_STATUS_OBSERVED = "observed"
_OBSERVER_NON_OBSERVED_MARKERS = (
    "gap", "unstable", "missing", "unavailable", "timeout", "pending", "running",
    "not-required", "not-observed", "page-base-only", "unpopulated",
)


def _observer_value_is_non_observed_marker(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        for separator in ("_", ":", ".", "/"):
            normalized = normalized.replace(separator, "-")
        normalized = "-".join(normalized.split())
        return any(
            normalized == marker
            or normalized.startswith(f"{marker}-")
            or normalized.endswith(f"-{marker}")
            or f"-{marker}-" in normalized
            for marker in _OBSERVER_NON_OBSERVED_MARKERS
        )
    if isinstance(value, Mapping):
        return _observer_value_is_non_observed_marker(value.get("status"))
    if isinstance(value, (list, tuple)):
        return any(_observer_value_is_non_observed_marker(item) for item in value)
    return False


def _observer_value_is_valid(field: str, value: object) -> bool:
    if not isinstance(field, str) or not field or value is None:
        return False
    if _observer_value_is_non_observed_marker(value):
        return False

    def u64(item: object) -> bool:
        return isinstance(item, int) and not isinstance(item, bool) and 0 <= item < 1 << 64

    def rawbits(item: object) -> bool:
        if isinstance(item, int) and not isinstance(item, bool):
            return 0 <= item < 1 << 64
        return isinstance(item, str) and item.startswith("0x") and len(item) == 18 \
            and all(char in "0123456789abcdefABCDEF" for char in item[2:])

    if field == "reservation.valid":
        return isinstance(value, bool)
    if field == "gpr":
        return isinstance(value, (list, tuple)) and len(value) == 32 and all(
            item is None
            or type(item) is int and 0 <= item < 1 << 64
            for item in value
        ) and value[0] == 0
    if field == "executed_pcs":
        return isinstance(value, (list, tuple)) and bool(value) and all(
            type(item) is int and 0 <= item < 1 << 64 and not item & 1
            for item in value
        )
    if field == "fpr_rawbits":
        return isinstance(value, (list, tuple)) and len(value) == 32 and all(rawbits(item) for item in value)
    if field == "vector.registers":
        if isinstance(value, Mapping):
            if tuple(value) != tuple(f"v{index}" for index in range(32)):
                return False
            value = tuple(value[f"v{index}"] for index in range(32))
        return isinstance(value, (list, tuple)) and len(value) == 32 and all(
            (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
            or (isinstance(item, str) and item.startswith("0x") and len(item[2:]) > 0
                and len(item[2:]) % 2 == 0
                and all(char in "0123456789abcdefABCDEF" for char in item[2:]))
            for item in value
        )
    if field == "vector.mask":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if field in {
        "reservation.address", "reservation.result", "gpr.writeback", "fault_pc",
        "executed_pc", "control.next_pc", "control.target", "after_pc", "link",
        "fault_address",
    }:
        return u64(value)
    if field == "branch_outcome":
        return value in {"taken", "not-taken"}
    if field == "csr.mstatus.fs":
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3
    if field in {"csr.fflags", "fflags"}:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0x1F
    if field in {"csr.frm", "frm"}:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 7
    if field in {"vxsat", "vector.vxsat"}:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1
    if field in {"vxrm", "vector.vxrm"}:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3
    if field == "privilege.mode":
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3
    if field.startswith("gpr.x"):
        register = field.removeprefix("gpr.x")
        if not register.isdigit() or str(int(register)) != register or int(register) > 31:
            return False
        return type(value) is int and (value == 0 if int(register) == 0 else u64(value))
    if field.startswith("gpr["):
        return False
    if field == "trap.cause":
        return type(value) is int and 0 <= value < 64
    if field in {"trap.epc", "csr.mepc"}:
        return u64(value) and not value & 1
    if field.startswith(("trap.", "csr.", "privilege.")):
        return u64(value)
    if field in {"reservation.ordering", "memory.ordering", "legality.outcome"}:
        return isinstance(value, str) and bool(value)
    if field == "outcome":
        return isinstance(value, str) and value in {
            "normal", "completed", "trap", "nonzero-exit", "timeout", "unavailable", "runner-contract-gap",
        }
    if field == "exit_code":
        return isinstance(value, int) and not isinstance(value, bool)
    if field == "checkpoint_pc":
        return u64(value)
    if field == "memory.size":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if field == "signal":
        return isinstance(value, str) and bool(value)
    if field == "signal_code":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if field == "signal.si_code":
        return isinstance(value, int) and not isinstance(value, bool) and -(1 << 31) <= value < 1 << 31
    if field == "signal.si_code_name":
        return isinstance(value, str) and bool(value.strip())
    if field == "signal.si_addr":
        try:
            return 0 <= int(value, 0) < 1 << 64 if isinstance(value, str) else u64(value)
        except ValueError:
            return False
    if field == "translation.path_identity":
        return is_sha256_digest(value) if isinstance(value, str) else (
            isinstance(value, Mapping) and value.get("status") == PATH_IDENTITY_WITNESS_STATUS_OBSERVED
        )
    if field == "profile.extension-gate":
        return isinstance(value, (bool, str, Mapping)) and bool(value)
    if field in {"profile.pair", "sail.profile", "sail.reference", "sail.outcome"}:
        return isinstance(value, (str, Mapping)) and bool(value)
    if field.startswith(("memory.", "lifecycle.", "state.")) or field == "fetch.second":
        if isinstance(value, Mapping):
            return bool(value)
        return not isinstance(value, (str, bytes, list, tuple)) or bool(value)
    if field.startswith("vector."):
        return u64(value)
    if field == "profile.xlen":
        return type(value) is int and value in {32, 64}
    if field == "profile.extensions":
        return isinstance(value, str) and bool(value.strip())
    if field == "profile.privilege":
        return isinstance(value, str) and value.strip().lower() in {"m", "s", "u", "h"}
    if field == "profile.ialign":
        return type(value) is int and value in {2, 4}
    return isinstance(value, (str, int, float, bool, bytes, bytearray, Mapping, list, tuple))


def backend_support_reason(backend: str, isa_profile: str) -> str | None:
    """Return only malformed backend/profile errors.

    ISA capability is deliberately not checked here.  A generator may emit
    an extension that a target lacks; the target process must be allowed to
    report its illegal-instruction/unsupported-ISA result.  The caller can
    still reject an unknown backend or a malformed XLEN profile before trying
    to spawn a process.
    """
    profile = str(isa_profile or "").strip().lower()
    if not is_execution_backend(backend):
        return f"{backend} is an unknown execution backend"
    if not profile.startswith(("rv32", "rv64")):
        return "ISA profile is malformed"
    return None


_HARNESS_NAMES = frozenset({"riscv-dv", "custom", "rvgen-single"})
def backend_capability_gap(
    backend: object, *, harness: object = None,
    binary_path: object = None, runner_env: Mapping[str, object] | None = None,
) -> str | None:
    """检查 runner、harness 和外部依赖是否真的可用。"""
    if not is_execution_backend(backend):
        return f"unknown-backend:{backend}"
    canonical = canonical_execution_backend(str(backend))
    spec = backend_spec(str(backend))
    repository = Path(__file__).resolve().parents[2]
    entrypoint = (
        repository / "framework" / "cli" / spec.runner_entrypoint
        if spec.runner_entrypoint_kind == "script" else
        repository / Path(*spec.runner_entrypoint.split(".")).with_suffix(".py")
    )
    environment = runner_env if isinstance(runner_env, Mapping) else {}
    if not entrypoint.is_file():
        return f"runner-entrypoint-missing:{spec.runner_entrypoint}"
    if harness not in (None, "") and harness not in _HARNESS_NAMES:
        return f"harness-unsupported:{harness}"
    if binary_path not in (None, ""):
        binary = str(binary_path)
        if not Path(binary).is_file() and shutil.which(binary) is None:
            return f"backend-binary-missing:{binary}"
    dependency_env = {
        "rvvm-riscv64": ("RVVM_BINARY",),
        "renode-riscv64": ("RENODE_DLL",),
        "rax-riscv64": ("RAX_BINARY",),
    }.get(canonical, ())
    for name in dependency_env:
        value = environment.get(name)
        if value not in (None, "") and not Path(str(value)).is_file() and shutil.which(str(value)) is None:
            return f"{name.lower().replace('_', '-')}-missing:{value}"
    library_path = environment.get("LIBUNICORN_PATH") if canonical == "unicorn-riscv64" else None
    if library_path not in (None, "") and not Path(str(library_path)).is_dir():
        return f"libunicorn-path-missing:{library_path}"
    return None


def route_admission_gap(
    root: Mapping[str, object], route: str, reference_backend: object,
    target_backend: object, isa_profile: object, harness: object,
    observer_fields: Sequence[str] = (),
) -> str | None:
    """检查正式路线是否值得进入构建器和执行器。"""
    if route not in {"single", "program"}:
        return "route-invalid"
    if any(not isinstance(root.get(name), str) or not root[name].strip()
           for name in ("root_key", "lineage_key", "observer")):
        return "root-identity-incomplete"
    if not isinstance(observer_fields, Sequence) or isinstance(observer_fields, (str, bytes)) \
            or any(not isinstance(field, str) or not field.strip() for field in observer_fields):
        return "observer-fields-invalid"
    admission_env = dict(root.get("runner_env")) if isinstance(root.get("runner_env"), Mapping) else {}
    if not isinstance(isa_profile, str) or not is_canonical_isa_profile(isa_profile.lower()):
        return "isa-profile-invalid"
    if not isinstance(harness, str) or not harness.strip():
        return "harness-missing"
    # Reference execution is optional in the open-capability run.  A missing
    # reference only makes the observation comparator emit reference-gap.
    if not isinstance(reference_backend, str) or not reference_backend.strip():
        reference_backend = None
    reference_runtime = reference_backend
    if reference_backend == "qemu":
        reference_runtime = (
            "qemu-riscv32" if isa_profile.lower().startswith("rv32") else "qemu-riscv64"
        )
    if reference_runtime in {"qemu-riscv32", "qemu-riscv64"}:
        if reason := backend_support_reason(
            reference_runtime, isa_profile,
        ):
            return f"reference-backend-unsupported:{reason}"
    elif reference_backend is not None and reference_backend not in _REFERENCE_BACKENDS:
        if not is_execution_backend(reference_backend):
            return f"unknown-reference-backend:{reference_backend}"
        if reason := backend_support_reason(
            reference_backend, isa_profile,
        ):
            return f"reference-backend-unsupported:{reason}"
        spec = backend_spec(reference_backend)
        if not spec.runner_entrypoint or not spec.execution_model:
            return "reference-runner-contract-missing"
    if isinstance(reference_backend, str) and is_execution_backend(reference_backend):
        if reason := backend_capability_gap(
            reference_backend,
            harness=harness,
            binary_path=root.get("reference_binary_path"),
            runner_env=admission_env,
        ):
            return f"reference-backend-capability-gap:{reason}"
    if target_backend not in (None, "", "unavailable", "unused", "not-run"):
        # A shared backend may still execute for source coverage; it only
        # limits the strength of a differential claim.
        if not is_execution_backend(target_backend):
            return f"unknown-target-backend:{target_backend}"
        if reason := backend_support_reason(
            target_backend, isa_profile,
        ):
            return f"target-backend-unsupported:{reason}"
        spec = backend_spec(target_backend)
        if not spec.runner_entrypoint or not spec.execution_model:
            return "target-runner-contract-missing"
        if reason := backend_capability_gap(
            target_backend,
            harness=harness,
            binary_path=root.get("target_binary_path"),
            runner_env=admission_env,
        ):
            return f"target-backend-capability-gap:{reason}"
    return None

import base64
import hashlib
import json
import os
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path

from .execution_identity import (
    InstalledTargetBinary,
    NativeReferenceIdentity,
    TargetBinaryIdentity,
    _target_binary_names,
    is_allowlisted_patched_identity,
)
from .paths import CURRENT_OFFICIAL_EVIDENCE_INDEX, LOCAL_ROOT, REPOSITORY_ROOT
from ._util import is_sha256_digest

DOCKER_EXECUTION_PLANE = "docker-x86_64-lab"
EXECUTION_PLANE_ENV = "MIGRATION_EXECUTION_PLANE"
CONTAINER_IMAGE_ENV = "MIGRATION_CONTAINER_IMAGE"
CONTAINER_IMAGE_DIGEST_ENV = "MIGRATION_CONTAINER_IMAGE_DIGEST"
EM_X86_64 = 62
EM_RISCV = 243
DEFAULT_NATIVE_REFERENCE_HOST = "133.133.137.252"


class ExecutionEnvironmentError(RuntimeError):
    pass


def require_execution_plane(
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    current_environment = os.environ if environment is None else environment
    plane = current_environment.get(EXECUTION_PLANE_ENV)
    if (
        platform.system() != "Linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
        or plane != DOCKER_EXECUTION_PLANE
        or not Path("/.dockerenv").is_file()
    ):
        raise ExecutionEnvironmentError(
            "direct RV64 execution is supported only inside the Docker x86_64 lab; "
            "use framework/container/x86_64-lab/run-x86-lab.ps1"
        )


def elf_machine(path: Path) -> int:
    try:
        header = path.read_bytes()[:20]
    except OSError as exc:
        raise ExecutionEnvironmentError(f"cannot read executable {path}: {exc}") from exc
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise ExecutionEnvironmentError(f"not an ELF executable: {path}")
    if header[5] not in {1, 2}:
        raise ExecutionEnvironmentError(f"unsupported ELF byte order: {path}")
    byte_order = "little" if header[5] == 1 else "big"
    return int.from_bytes(header[18:20], byte_order)


def validate_elf_machine(path: Path, expected_machine: int, label: str) -> None:
    actual_machine = elf_machine(path)
    if actual_machine != expected_machine:
        raise ExecutionEnvironmentError(
            f"{label} has ELF machine {actual_machine}, expected {expected_machine}: {path}"
        )


def resolve_x86_64_runner(command: str, label: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        candidate = Path(command)
        if not candidate.is_file():
            raise ExecutionEnvironmentError(f"missing {label}: {command}")
        resolved = str(candidate.resolve())
    path = Path(resolved)
    validate_elf_machine(path, EM_X86_64, label)
    if not os.access(path, os.X_OK):
        raise ExecutionEnvironmentError(f"{label} is not executable: {path}")
    return str(path)


def load_target_binary_manifest(
    manifest_path: Path,
    *,
    expected_backend: str,
    require_source_verified: bool = False,
    require_build_attested: bool = False,
) -> InstalledTargetBinary:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ExecutionEnvironmentError(f"missing target identity manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not is_sha256_digest(payload.get("identity_digest")):
            raise ValueError("target identity digest is missing")
        identity = TargetBinaryIdentity.from_dict(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ExecutionEnvironmentError(f"invalid target identity manifest: {manifest_path}") from exc
    if identity.backend != expected_backend:
        raise ExecutionEnvironmentError(
            f"target manifest backend is {identity.backend}, expected {expected_backend}"
        )
    if (identity.patch_lineage or identity.source_provenance in {
        "patch-validated-build", "verified-git-archive+controlled-patch",
    }) and not is_allowlisted_patched_identity(identity):
        raise ExecutionEnvironmentError(
            f"simulator-source-patches are not permitted: {manifest_path}"
        )
    if require_source_verified and not identity.is_source_verified:
        raise ExecutionEnvironmentError(
            f"target manifest is not source verified: {manifest_path}"
        )
    if require_build_attested and not identity.is_build_attested:
        raise ExecutionEnvironmentError(
            f"target manifest is not a build-attested replayable target: {manifest_path}"
        )
    binary_path = next(
        (
            manifest_path.parent / name
            for name in _target_binary_names(expected_backend)
            if (manifest_path.parent / name).is_file()
        ),
        None,
    )
    if binary_path is None:
        raise ExecutionEnvironmentError(
            f"target manifest has no source-verified binary beside it for {expected_backend}: {manifest_path}"
        )
    if expected_backend != "renode-riscv64":
        validate_elf_machine(binary_path, EM_X86_64, expected_backend)
    observed = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    if observed != identity.binary_sha256:
        raise ExecutionEnvironmentError(
            f"target binary hash does not match manifest: {binary_path}"
        )
    return InstalledTargetBinary(binary_path, manifest_path, identity)


def _read_target_identity_manifest(
    manifest_path: Path,
    *,
    expected_backend: str,
) -> TargetBinaryIdentity | None:
    try:
        identity = TargetBinaryIdentity.from_dict(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if identity.backend != expected_backend:
        return None
    return identity


def _current_source_verified_matches(
    *,
    expected_backend: str,
    environment: Mapping[str, str] | None = None,
    store_root: Path | None = None,
) -> tuple[tuple[Path, TargetBinaryIdentity], ...]:
    current_environment = os.environ if environment is None else environment
    container_image = current_environment.get(CONTAINER_IMAGE_ENV)
    container_image_digest = current_environment.get(CONTAINER_IMAGE_DIGEST_ENV)
    backend_root = (store_root or (LOCAL_ROOT / "target-store")).resolve() / expected_backend
    if not container_image or not container_image_digest or not backend_root.is_dir():
        return ()
    return tuple(
        (path, identity)
        for path in sorted(backend_root.glob("*/target-identity.json"))
        if (identity := _read_target_identity_manifest(path, expected_backend=expected_backend))
        is not None
        and identity.is_source_verified
        and identity.container_image == container_image
        and identity.container_image_digest == container_image_digest
    )

def _current_build_attested_manifests(
    *,
    expected_backend: str,
    container_image: str,
    container_image_digest: str,
    matched_manifests: tuple[Path, ...],
    store_root: Path,
) -> tuple[Path, ...]:
    """Use the existing current-container build provenance to disambiguate targets."""
    provenance_root = LOCAL_ROOT / "calibration-builds" / expected_backend
    if not provenance_root.is_dir():
        return ()
    matched = {path.resolve() for path in matched_manifests}
    selected: set[Path] = set()
    for provenance_path in sorted(
        provenance_root.glob("*/current-container-baseline/build-provenance.json")
    ):
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            identity = TargetBinaryIdentity.from_dict(
                dict(payload["installed_binary"]["identity"])
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if (
            identity.backend != expected_backend
            or not identity.is_source_verified
            or identity.container_image != container_image
            or identity.container_image_digest != container_image_digest
        ):
            continue
        manifest_path = (
            store_root.resolve()
            / expected_backend
            / identity.identity_digest
            / "target-identity.json"
        ).resolve()
        if manifest_path in matched:
            selected.add(manifest_path)
    return tuple(sorted(selected))


def resolve_current_container_source_verified_manifest(
    *,
    expected_backend: str,
    environment: Mapping[str, str] | None = None,
    store_root: Path | None = None,
) -> Path | None:
    matches = list(
        _current_source_verified_matches(
            expected_backend=expected_backend,
            environment=environment,
            store_root=store_root,
        )
    )
    matched_manifests = [path for path, _identity in matches]
    if not matched_manifests:
        return None
    official_index = (
        CURRENT_OFFICIAL_EVIDENCE_INDEX
        if CURRENT_OFFICIAL_EVIDENCE_INDEX.is_file()
        else REPOSITORY_ROOT / "framework" / "evidence" / "official-evidence-index.json"
    )
    try:
        official = json.loads(official_index.read_text(encoding="utf-8"))
        preferred_digest = (
            official.get("current_code_identity", {})
            .get("target_identity", {})
            .get(expected_backend)
        )
    except (OSError, AttributeError, TypeError, ValueError):
        preferred_digest = None
    if isinstance(preferred_digest, str) and preferred_digest:
        matched_manifests = [
            path
            for path, identity in matches
            if identity.identity_digest == preferred_digest
        ]
        if not matched_manifests:
            raise ExecutionEnvironmentError(
                f"official current target identity is missing for {expected_backend}"
            )
    if len(matched_manifests) > 1:
        build_manifests = _current_build_attested_manifests(
            expected_backend=expected_backend,
            container_image=matches[0][1].container_image,
            container_image_digest=matches[0][1].container_image_digest,
            matched_manifests=tuple(matched_manifests),
            store_root=(store_root or (LOCAL_ROOT / "target-store")).resolve(),
        )
        if len(build_manifests) == 1:
            matched_manifests = list(build_manifests)
    if len(matched_manifests) != 1:
        raise ExecutionEnvironmentError(
            f"multiple current-container source-verified manifests for {expected_backend}"
        )
    return load_target_binary_manifest(
        matched_manifests[0],
        expected_backend=expected_backend,
        require_source_verified=True,
    ).manifest_path


def resolve_preferred_libriscv_translated_manifest(
    *,
    environment: Mapping[str, str] | None = None,
    store_root: Path | None = None,
) -> Path | None:
    matches = list(
        _current_source_verified_matches(
            expected_backend="libriscv-translated",
            environment=environment,
            store_root=store_root,
        )
    )
    matched_manifests = [path for path, _identity in matches]
    if not matched_manifests:
        return None
    fcsr_on_manifests = [
        path for path, identity in matches if "-DRISCV_FCSR=ON" in identity.build_flags
    ]
    if fcsr_on_manifests:
        if len(fcsr_on_manifests) != 1:
            raise ExecutionEnvironmentError(
                "multiple current-container source-verified FCSR=ON manifests for libriscv-translated"
            )
        matched_manifests = fcsr_on_manifests
    elif len(matched_manifests) != 1:
        raise ExecutionEnvironmentError(
            "multiple current-container source-verified manifests for libriscv-translated"
        )
    return load_target_binary_manifest(
        matched_manifests[0],
        expected_backend="libriscv-translated",
        require_source_verified=True,
    ).manifest_path


def _sha256_path(path: Path, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ExecutionEnvironmentError(f"missing {label}: {resolved}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _known_host_key_sha256(known_hosts_path: Path, host: str) -> str:
    resolved = known_hosts_path.resolve()
    if not resolved.is_file():
        raise ExecutionEnvironmentError(f"missing native known_hosts file: {resolved}")
    for line in resolved.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record or record.startswith("#"):
            continue
        parts = record.split()
        if len(parts) < 3:
            continue
        if host not in parts[0].split(","):
            continue
        try:
            key_bytes = base64.b64decode(parts[2].encode("ascii"), validate=True)
        except ValueError as exc:
            raise ExecutionEnvironmentError(
                f"invalid base64 key for native host {host}: {resolved}"
            ) from exc
        return hashlib.sha256(key_bytes).hexdigest()
    raise ExecutionEnvironmentError(f"native host {host} is missing from known_hosts: {resolved}")


def resolve_native_reference_identity(
    machine: str,
    kernel: str,
    isa: str,
    *,
    known_hosts_path: Path | None = None,
    runner_path: Path | None = None,
    helper_path: Path | None = None,
) -> NativeReferenceIdentity:
    if not machine or not kernel or not isa:
        raise ExecutionEnvironmentError("native machine, kernel, and ISA are required")
    return NativeReferenceIdentity(
        schema_version="native-reference-identity-v1",
        host_key_sha256=_known_host_key_sha256(
            known_hosts_path
            or REPOSITORY_ROOT
            / ".codex"
            / "skills"
            / "use-native-riscv-environment"
            / "scripts"
            / "riscv-known-hosts",
            DEFAULT_NATIVE_REFERENCE_HOST,
        ),
        machine=machine,
        isa=isa,
        kernel=kernel,
        runner_sha256=_sha256_path(
            runner_path or REPOSITORY_ROOT / "framework" / "native" / "rv64_runner.py",
            "native RV64 runner",
        ),
        helper_sha256=_sha256_path(
            helper_path
            or REPOSITORY_ROOT / "framework" / "native_trace" / "ptrace_trace.c",
            "native trace helper",
        ),
    )

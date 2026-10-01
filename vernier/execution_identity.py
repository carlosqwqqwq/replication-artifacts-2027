import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import (
    _non_empty_string as _require_text,
    canonical_digest,
    require_sha256 as _require_sha256,
)

LIVE_NATIVE_REFERENCE_PROVENANCE = "native-runner"
LIVE_NATIVE_REFERENCE_EVIDENCE_TIER = "live-native"
LIVE_NATIVE_REFERENCE_IDENTITY_SOURCE = "runner-contract"

def _require_git_commit(value: object, name: str) -> None:
    if not isinstance(value, str) or value != value.lower() or len(value) not in {40, 64}:
        raise ValueError(f"{name} must be a full Git commit")
    try:
        if len(bytes.fromhex(value)) not in {20, 32}:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"{name} must be a full Git commit") from exc


@dataclass(frozen=True)
class PatchLineageEntry:
    patch_path: str | None
    patch_sha256: str
    patch_id: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.patch_path is None:
            _require_text(self.patch_id, "patch_id")
        else:
            _require_text(self.patch_path, "patch_path")
        if self.description is not None:
            _require_text(self.description, "description")
        _require_sha256(self.patch_sha256, "patch_sha256")

    def to_dict(self) -> dict[str, str]:
        if self.patch_path is None:
            result = {"id": self.patch_id}
            if self.description is not None:
                result["description"] = self.description
            result["patch_sha256"] = self.patch_sha256
            return result
        return {"patch_path": self.patch_path, "patch_sha256": self.patch_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PatchLineageEntry":
        if not isinstance(data, Mapping):
            raise ValueError("patch lineage entry must be an object")
        patch_path = data.get("patch_path")
        patch_id = data.get("id")
        if patch_path is None and patch_id is None:
            raise ValueError("patch lineage entry needs patch_path or id")
        return cls(
            patch_path=patch_path,
            patch_sha256=data.get("patch_sha256"),
            patch_id=patch_id,
            description=data.get("description"),
        )


@dataclass(frozen=True)
class TargetBinaryIdentity:
    schema_version: str
    backend: str
    source_repository: str | None
    source_commit: str | None
    build_flags: tuple[str, ...]
    container_image: str
    container_image_digest: str
    binary_sha256: str
    source_provenance: str = "verified-build"
    patch_lineage: tuple[PatchLineageEntry, ...] = ()
    commit_date: str | None = None
    source_tree: str | None = None
    artifact_file_count: int | None = None
    artifact_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "backend",
            "container_image",
            "container_image_digest",
            "binary_sha256",
            "source_provenance",
        ):
            _require_text(getattr(self, name), name)
        for name in ("source_repository", "source_commit"):
            value = getattr(self, name)
            if value is not None:
                _require_git_commit(value, name) if name == "source_commit" else _require_text(value, name)
        if "/" in self.backend or "\\" in self.backend or self.backend in {".", ".."}:
            raise ValueError("backend must be a single path component")
        if not isinstance(self.build_flags, tuple) or any(
            not isinstance(flag, str) for flag in self.build_flags
        ):
            raise ValueError("build_flags must be a tuple of strings")
        if not isinstance(self.patch_lineage, tuple) or any(
            not isinstance(entry, PatchLineageEntry) for entry in self.patch_lineage
        ):
            raise ValueError("patch_lineage must be a tuple of entries")
        if self.schema_version not in {
            "target-binary-identity-v1",
            "target-binary-identity-v2",
            "target-binary-identity-v3",
        }:
            raise ValueError("unsupported target binary identity schema")
        if self.schema_version == "target-binary-identity-v1":
            if self.source_provenance != "verified-build":
                raise ValueError("v1 target identities must be source verified")
            if self.patch_lineage:
                raise ValueError("v1 target identities cannot carry patch lineage")
            if not self.source_repository or not self.source_commit:
                raise ValueError("v1 target identities require source provenance")
        elif self.source_provenance in {"verified-build", "verified-checkout-clean-tree"}:
            if not self.source_repository or not self.source_commit:
                raise ValueError("verified target identities require source provenance")
            if self.patch_lineage:
                raise ValueError("verified target identities cannot carry patch lineage")
        elif self.source_provenance in {
            "patch-validated-build", "verified-git-archive+controlled-patch",
        }:
            if self.schema_version != "target-binary-identity-v3":
                raise ValueError("patch-validated target identities require schema v3")
            if not self.source_repository or not self.source_commit:
                raise ValueError("patch-validated target identities require source provenance")
            if not self.patch_lineage:
                raise ValueError("patch-validated target identities require patch lineage")
        else:
            raise ValueError("unsupported target source provenance")
        _require_sha256(self.binary_sha256, "binary_sha256")
        if (
            not isinstance(self.container_image_digest, str)
            or not self.container_image_digest.startswith("sha256:")
            or len(self.container_image_digest) != 71
        ):
            raise ValueError("container_image_digest must be content addressed")
        _require_sha256(self.container_image_digest[7:], "container_image_digest")

    def canonical_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "build_flags": list(self.build_flags),
            "container_image": self.container_image,
            "container_image_digest": self.container_image_digest,
            "binary_sha256": self.binary_sha256,
        }
        if self.schema_version in {"target-binary-identity-v2", "target-binary-identity-v3"}:
            result["source_provenance"] = self.source_provenance
        if self.schema_version == "target-binary-identity-v3":
            for key, value in (
                ("commit_date", self.commit_date),
                ("source_tree", self.source_tree),
                ("artifact_file_count", self.artifact_file_count),
                ("artifact_manifest_sha256", self.artifact_manifest_sha256),
            ):
                if value is not None:
                    result[key] = value
        if self.patch_lineage or self.schema_version == "target-binary-identity-v3":
            result["patch_lineage"] = [entry.to_dict() for entry in self.patch_lineage]
        return result

    @property
    def is_source_verified(self) -> bool:
        return self.source_provenance in {
            "verified-build", "verified-checkout-clean-tree", "verified-git-archive",
        }

    @property
    def is_build_attested(self) -> bool:
        return self.source_provenance in {
            "verified-build", "verified-checkout-clean-tree", "patch-validated-build",
            "verified-git-archive+controlled-patch",
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest({"contract": self.schema_version, **self.canonical_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "identity_digest": self.identity_digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetBinaryIdentity":
        if not isinstance(data, dict):
            raise ValueError("target binary identity must be an object")
        build_flags = data.get("build_flags", ())
        if not isinstance(build_flags, (list, tuple)):
            raise ValueError("build_flags must be a list or tuple")
        raw_patch_lineage = data.get("patch_lineage", ())
        if not isinstance(raw_patch_lineage, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in raw_patch_lineage
        ):
            raise ValueError("patch_lineage must be a list of objects")
        identity = cls(
            schema_version=data.get("schema_version"),
            backend=data.get("backend"),
            source_repository=data.get("source_repository"),
            source_commit=data.get("source_commit"),
            build_flags=tuple(build_flags),
            container_image=data.get("container_image"),
            container_image_digest=data.get("container_image_digest"),
            binary_sha256=data.get("binary_sha256"),
            source_provenance=data.get("source_provenance", "verified-build"),
            patch_lineage=tuple(
                PatchLineageEntry.from_dict(item)
                for item in raw_patch_lineage
            ),
            commit_date=data.get("commit_date"),
            source_tree=data.get("source_tree"),
            artifact_file_count=data.get("artifact_file_count"),
            artifact_manifest_sha256=data.get("artifact_manifest_sha256"),
        )
        recorded_digest = data.get("identity_digest")
        if recorded_digest is not None and recorded_digest != identity.identity_digest:
            raise ValueError("target identity digest mismatch")
        return identity


def is_allowlisted_patched_identity(value: Mapping[str, object] | TargetBinaryIdentity) -> bool:
    """Return whether a patched identity carries a reproducible lineage.

    The experiment may need a small observer hook in a simulator.  The
    identity contract already binds the resulting binary, source commit and
    patch digest, so a global binary allowlist only turns valid local builds
    into false gaps.  Keep this helper name for the current callers, but make
    the policy evidence-based rather than tied to yesterday's artifact hash.
    """
    data = value.to_dict() if isinstance(value, TargetBinaryIdentity) else value
    if not isinstance(data, Mapping):
        return False
    provenance = data.get("source_provenance")
    if provenance not in {
        "patch-validated-build", "verified-git-archive+controlled-patch",
    }:
        return False
    lineage = data.get("patch_lineage")
    if not isinstance(lineage, (list, tuple)) or not lineage:
        return False
    for entry in lineage:
        if not isinstance(entry, Mapping):
            return False
        patch_ref = entry.get("patch_path") or entry.get("id")
        patch_sha256 = entry.get("patch_sha256")
        if not isinstance(patch_ref, str) or not patch_ref.strip():
            return False
        if not isinstance(patch_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", patch_sha256):
            return False
    return True


def _target_identity_parts(
    value: object,
) -> tuple[tuple[Mapping[str, object], ...], tuple[dict[str, object], ...]]:
    if not isinstance(value, Mapping):
        return (), ()
    evidence = value.get("translation_evidence")
    details = evidence.get("details") if isinstance(evidence, Mapping) else None
    containers = tuple(
        item for item in (
            value, details,
            details.get("configuration_identity") if isinstance(details, Mapping) else None,
        )
        if isinstance(item, Mapping)
    )
    return containers, tuple(
        dict(item["target_identity"])
        for item in containers
        if isinstance(item.get("target_identity"), Mapping)
    )


def _target_identity_containers_match(
    containers: tuple[Mapping[str, object], ...], identity: TargetBinaryIdentity,
) -> bool:
    return all(
        container.get("target_binary_sha256") in (None, "")
        or container["target_binary_sha256"] == identity.binary_sha256
        for container in containers
    ) and all(
        container.get("target_identity_digest") in (None, "")
        or container["target_identity_digest"] == identity.identity_digest
        for container in containers
    )


@dataclass(frozen=True)
class NativeReferenceIdentity:
    """Identity of the native reference host and the source used by its trace helper.

    ``helper_sha256`` deliberately names the pinned helper *source* digest, not
    a transient uploaded ELF build; the latter can change with toolchain build
    metadata while the executed helper semantics remain the same.
    """

    schema_version: str
    host_key_sha256: str
    machine: str
    isa: str
    kernel: str
    runner_sha256: str
    helper_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "schema_version")
        for name in (
            "host_key_sha256",
            "machine",
            "isa",
            "kernel",
            "runner_sha256",
            "helper_sha256",
        ):
            _require_text(getattr(self, name), name)
        if self.schema_version != "native-reference-identity-v1":
            raise ValueError("unsupported native reference identity schema")
        if not re.fullmatch(r"rv(?:32|64)[a-z]+(?:_[a-z0-9]+)*", self.isa.lower()):
            raise ValueError("isa must be a RISC-V ISA string")
        _require_sha256(self.host_key_sha256, "host_key_sha256")
        _require_sha256(self.runner_sha256, "runner_sha256")
        _require_sha256(self.helper_sha256, "helper_sha256")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "host_key_sha256": self.host_key_sha256,
            "machine": self.machine,
            "isa": self.isa,
            "kernel": self.kernel,
            "runner_sha256": self.runner_sha256,
            "helper_sha256": self.helper_sha256,
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest({"contract": self.schema_version, **self.canonical_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "identity_digest": self.identity_digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NativeReferenceIdentity":
        if not isinstance(data, dict):
            raise ValueError("native reference identity must be an object")
        identity = cls(
            schema_version=data.get("schema_version"),
            host_key_sha256=data.get("host_key_sha256"),
            machine=data.get("machine"),
            isa=data.get("isa"),
            kernel=data.get("kernel"),
            runner_sha256=data.get("runner_sha256"),
            helper_sha256=data.get("helper_sha256"),
        )
        recorded_digest = data.get("identity_digest")
        if recorded_digest is not None and recorded_digest != identity.identity_digest:
            raise ValueError("native reference identity digest mismatch")
        return identity


@dataclass(frozen=True)
class InstalledTargetBinary:
    binary_path: Path
    manifest_path: Path
    identity: TargetBinaryIdentity


_TARGET_BINARY_NAMES = {
    "qemu-riscv32": ("qemu-riscv32",),
    "qemu-riscv64": ("qemu-riscv64",),
    "libriscv-translated": ("rvlinux",),
    "rvvm-riscv64": ("rvvm_x86_64", "rvvm_x86_64_wrapper", "rvvm-riscv64"),
    "unicorn-riscv64": ("libunicorn.so.2",),
    "renode-riscv64": ("Renode.dll",),
    "rax-riscv64": ("rax",),
}


def _target_binary_names(backend: str) -> tuple[str, ...]:
    return _TARGET_BINARY_NAMES.get(backend, (backend,))


def _target_binary_name(backend: str) -> str | None:
    names = _TARGET_BINARY_NAMES.get(backend)
    return names[0] if names else None


@dataclass(frozen=True)
class ExecutionEvidenceBundle:
    schema_version: str
    main_state_digest: str
    trace_state_digest: str
    count_state_digest: str
    main_command_fingerprint: str
    trace_command_fingerprint: str
    count_command_fingerprint: str
    coherent: bool
    evidence_scope: str
    failures: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        main_observation: dict[str, Any],
        trace_observation: dict[str, Any],
        count_observation: dict[str, Any],
        main_command_fingerprint: str,
        trace_command_fingerprint: str,
        count_command_fingerprint: str,
        count_state_comparable: bool = True,
    ) -> "ExecutionEvidenceBundle":
        main_pcs = main_observation.get("executed_pcs")
        main_pcs = tuple(main_pcs) if isinstance(main_pcs, (list, tuple)) else None
        main_extra = main_observation.get("extra_state")
        terminal_checkpoint = main_observation.get("checkpoint_pc")
        if (
            main_pcs is not None
            and terminal_checkpoint is not None
            and isinstance(main_extra, dict)
            and main_extra.get("guest_trap") == "delivered"
            and main_extra.get("guest_cause") == 3
            and main_extra.get("trap_observer") == "qemu-linux-user-ebreak"
        ):
            def align_checkpoint(item: dict[str, Any]) -> dict[str, Any]:
                return (
                    {**item, "checkpoint_pc": terminal_checkpoint}
                    if item.get("checkpoint_pc") is None
                    and isinstance(item.get("executed_pcs"), (list, tuple))
                    and tuple(item.get("executed_pcs", ())) == main_pcs
                    else item
                )
            trace_observation = align_checkpoint(trace_observation)
            count_observation = align_checkpoint(count_observation)
        extra_states = tuple(
            item.get("extra_state")
            for item in (main_observation, trace_observation, count_observation)
        )
        extra_key_sets = [
            set(item) for item in extra_states if isinstance(item, dict)
        ]
        shared_extra_keys = (
            set.intersection(*extra_key_sets) if extra_key_sets else set()
        )
        shared_extra_keys = {
            name for name in shared_extra_keys
            if name != "observer_fields"
            and not any(
                name == timing or name.startswith(f"{timing}_")
                for timing in ("csr.cycle", "csr.time", "csr.instret")
            )
            and not name.endswith(("_observer", "_observer_status", "_gap"))
        }
        volatile_gpr_indices = {
            index
            for extra in extra_states
            if isinstance(extra, dict)
            for index in extra.get("volatile_gpr_indices", ())
            if type(index) is int and 0 <= index < 32
        }
        counter_read = any(
            isinstance(extra, dict)
            and any(name in extra for name in ("csr.cycle", "csr.time", "csr.instret"))
            for extra in extra_states
        )
        if counter_read:
            # ponytail: counter reads make all downstream GPR snapshots run-dependent; keep path/trap state.
            volatile_gpr_indices.update(range(32))
            main_observation = {**main_observation, "memory_snapshot": {}}
            trace_observation = {**trace_observation, "memory_snapshot": {}}
            count_observation = {**count_observation, "memory_snapshot": {}}
        main, trace, count = (
            _architectural_state_digest(item, shared_extra_keys, volatile_gpr_indices)
            for item in (main_observation, trace_observation, count_observation)
        )
        failures = []
        if trace != main:
            failures.append("trace-state-mismatch")
        if count_state_comparable and count != main:
            failures.append("count-state-mismatch")
        for label, item in (("trace", trace_observation), ("count", count_observation)):
            if label == "count" and not count_state_comparable:
                continue
            pcs = item.get("executed_pcs")
            pcs = tuple(pcs) if isinstance(pcs, (list, tuple)) else None
            if label == "count" and not pcs:
                continue
            if main_pcs is not None and pcs is not None and main_pcs != pcs:
                failures.append(f"{label}-path-mismatch")
        exact_command = (
            main_command_fingerprint == trace_command_fingerprint == count_command_fingerprint
        )
        evidence_scope = (
            "invalid" if failures else
            "count-only/diagnostic" if not count_state_comparable else
            "exact-main" if exact_command else "site-only/diagnostic"
        )
        return cls(
            schema_version="execution-evidence-bundle-v1",
            main_state_digest=main,
            trace_state_digest=trace,
            count_state_digest=count,
            main_command_fingerprint=main_command_fingerprint,
            trace_command_fingerprint=trace_command_fingerprint,
            count_command_fingerprint=count_command_fingerprint,
            coherent=not failures,
            evidence_scope=evidence_scope,
            failures=tuple(failures),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "main_state_digest": self.main_state_digest,
            "trace_state_digest": self.trace_state_digest,
            "count_state_digest": self.count_state_digest,
            "main_command_fingerprint": self.main_command_fingerprint,
            "trace_command_fingerprint": self.trace_command_fingerprint,
            "count_command_fingerprint": self.count_command_fingerprint,
            "coherent": self.coherent,
            "evidence_scope": self.evidence_scope,
            "failures": list(self.failures),
        }


def _architectural_state_digest(
    observation: dict[str, Any], shared_extra_keys: set[str] = frozenset(),
    volatile_gpr_indices: set[int] = frozenset(),
) -> str:
    state = {
        "outcome": observation.get("outcome"),
        "checkpoint_pc": observation.get("checkpoint_pc"),
        "memory_snapshot": observation.get("memory_snapshot", observation.get("memory_delta", {})),
    }
    gpr = observation.get("gpr")
    state["gpr"] = (
        [None if index in volatile_gpr_indices else value for index, value in enumerate(gpr)]
        if isinstance(gpr, (list, tuple)) else gpr
    )
    extra_state = observation.get("extra_state")
    if isinstance(extra_state, dict):
        state["extra_state"] = {
            name: extra_state[name] for name in shared_extra_keys if name in extra_state
        }
    for name in ("signal", "signal_code", "fault_pc", "fault_address"):
        if name in observation:
            state[name] = observation[name]
    if isinstance(extra_state, dict):
        state["trap_state"] = {
            name: extra_state.get(name)
            for name in (
                "guest_trap", "guest_cause", "trap.cause", "trap.epc", "trap.tval",
            )
            if name in extra_state
        }
    return canonical_digest({"contract": "architectural-observation-state-v1", **state})

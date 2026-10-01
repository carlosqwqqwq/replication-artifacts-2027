"""Fixed semantic-reference lock/config contract for RVGEN owners.

This module carries only pinned lock metadata and the narrow Sail profile
configuration contract needed to stamp a generation obligation.  It does not
launch Sail, inspect traces, or evaluate instruction results.
"""

from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path

from .paths import EXTERNAL_ROOT, REPOSITORY_ROOT, SAIL_MEMORY_OVERRIDE_CONFIG_PATH
from .spec_definedness import (
    enabled_extensions,
    isa_base_extensions,
    is_canonical_isa_profile,
    x_register_fp_extensions,
)


_REFERENCE_METADATA = {
    "sail-riscv": {
        "release_tag": "2026-07-27-9901550",
    },
}

_LOCK_PATH = REPOSITORY_ROOT / "external" / "repositories.lock.json"

@dataclass(frozen=True)
class ReferenceLock:
    """Pinned source identity used by generation provenance."""

    name: str
    origin: str
    relative_path: str
    commit: str
    release_tag: str

    def __post_init__(self) -> None:
        for name in ("name", "origin", "release_tag"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"reference lock {name} is required")
        if (
            type(self.relative_path) is not str
            or not self.relative_path.strip()
            or not (path := Path(self.relative_path)).parts
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("reference lock relative_path must be safe and relative")
        if (
            type(self.commit) is not str
            or len(self.commit) != 40
            or any(character not in "0123456789abcdef" for character in self.commit)
        ):
            raise ValueError("reference lock commit must be a lowercase 40-digit SHA-1")

    @property
    def checkout_path(self) -> Path:
        return EXTERNAL_ROOT / Path(self.relative_path)


@dataclass(frozen=True)
class SailProfileContract:
    """Static profile/config disposition; no Sail execution is performed."""

    supported: bool
    reason: str
    config_files: tuple[str, ...]
    candidate_kind: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise ValueError("sail profile supported must be a boolean")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("sail profile reason is required")
        if (
            not isinstance(self.config_files, (list, tuple))
            or any(type(path) is not str or not path.strip() for path in self.config_files)
            or len(self.config_files) != len(set(self.config_files))
        ):
            raise ValueError("sail profile config_files must be unique strings")
        if type(self.candidate_kind) is not str:
            raise ValueError("sail profile candidate_kind must be a string")


@dataclass(frozen=True)
class ReferenceLockConfig:
    """Neutral lock/config DTO shared by generation-side owners."""

    sail: ReferenceLock
    binary_release_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.sail, ReferenceLock):
            raise ValueError("reference lock config requires a sail lock")
        if not isinstance(self.binary_release_root, Path):
            raise ValueError("reference lock binary_release_root must be a path")

    def sail_profile_contract(self, isa_profile: str) -> SailProfileContract:
        """Return the fixed profile/config disposition used in provenance.

        The contract only selects the pinned config file names and checks that
        the declared files exist.  It intentionally does not inspect a Sail
        trace or reproduce architectural semantics.
        """
        profile = isa_profile if isinstance(isa_profile, str) else ""
        if not is_canonical_isa_profile(isa_profile):
            return SailProfileContract(
                False,
                f"unsupported-isa-profile:{isa_profile}",
                (),
                "",
            )
        if x_register_fp_extensions(profile):
            return SailProfileContract(
                False,
                f"unsupported-sail-floating-register-model:{isa_profile}",
                (),
                "",
            )
        if "h" in isa_base_extensions(profile):
            return SailProfileContract(
                False,
                f"unsupported-sail-h-extension:{isa_profile}",
                (),
                "",
            )
        enabled = enabled_extensions(profile)
        if "q" in enabled:
            return SailProfileContract(
                False,
                f"unsupported-sail-q-precision-sample-config:{isa_profile}",
                (),
                "",
            )
        if "v" in enabled or any(token.startswith("zv") for token in enabled):
            sample_name = (
                "rv32d_v128_e64.json"
                if profile.startswith("rv32")
                else "rv64d_v128_e64.json"
            )
        elif enabled & {"f", "d"}:
            sample_name = (
                "rv32d_v64_e64.json"
                if profile.startswith("rv32")
                else "rv64d_v64_e64.json"
            )
        else:
            sample_name = None
        candidates = (
            (
                "binary-release",
                self.binary_release_root / "bin" / "sail_riscv_sim",
                self.binary_release_root / "share" / "sail-riscv" / "config",
            ),
            (
                "source-build",
                self.sail.checkout_path / "build" / "c_emulator" / "sail_riscv_sim",
                self.sail.checkout_path / "config",
            ),
        )
        reason = "locked-checkout-present-without-runnable-sail-bundle"
        for candidate_kind, simulator, config_root in candidates:
            if not simulator.is_file():
                continue
            if sample_name is not None:
                sample_path = config_root / sample_name
                if not sample_path.is_file():
                    reason = f"missing-sail-sample-config:{sample_name}"
                    continue
            if not SAIL_MEMORY_OVERRIDE_CONFIG_PATH.is_file():
                reason = f"missing-sail-memory-override:{SAIL_MEMORY_OVERRIDE_CONFIG_PATH}"
                continue
            config_files = () if sample_name is None else (str(sample_path),)
            return SailProfileContract(
                True,
                "supported",
                (*config_files, str(SAIL_MEMORY_OVERRIDE_CONFIG_PATH)),
                candidate_kind,
            )
        return SailProfileContract(False, reason, (), "")

def _locked_reference(name: str) -> ReferenceLock:
    metadata = _REFERENCE_METADATA.get(name)
    if metadata is None:
        raise ValueError(f"unknown locked reference: {name}")
    try:
        payload = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("repositories.lock.json cannot be read") from exc
    if not isinstance(payload, dict):
        raise ValueError("repositories.lock.json must be a JSON object")
    repositories = payload.get("repositories")
    if not isinstance(repositories, list):
        raise ValueError("repositories.lock.json is missing repositories")
    matches = [
        entry
        for entry in repositories
        if isinstance(entry, dict) and entry.get("name") == name
    ]
    if not matches:
        raise ValueError(f"repository {name} is missing from repositories.lock.json")
    if len(matches) != 1:
        raise ValueError(f"repository {name} is duplicated in repositories.lock.json")
    entry = matches[0]
    origin = entry.get("origin")
    relative_path = entry.get("relative_path")
    commit = entry.get("commit")
    return ReferenceLock(
        name=name,
        origin=origin,
        relative_path=relative_path,
        commit=commit,
        release_tag=str(metadata["release_tag"]),
    )


_SAIL_LOCK = _locked_reference("sail-riscv")
REFERENCE_LOCK_CONFIG = ReferenceLockConfig(
    sail=_SAIL_LOCK,
    binary_release_root=(
        EXTERNAL_ROOT
        / "isa-references"
        / f"sail-riscv-{_SAIL_LOCK.release_tag}-Linux-x86_64"
    ),
)


@cache
def _sail_config_extension_support(isa_profile: str) -> dict[str, bool] | None:
    """Return extension support from the pinned Sail sample config."""
    sample_profile = "rv32ifd" if str(isa_profile).lower().startswith("rv32") else "rv64ifd"
    contract = REFERENCE_LOCK_CONFIG.sail_profile_contract(sample_profile)
    if not contract.supported:
        return None
    import re

    try:
        config_path = Path(contract.config_files[0])
        stripped = re.sub(
            r"^\s*//.*$", "", config_path.read_text(encoding="utf-8"), flags=re.MULTILINE,
        )
        payload = json.loads(stripped)
    except (IndexError, OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    extensions = payload.get("extensions") or {}
    if not isinstance(extensions, dict):
        return None
    support: dict[str, bool] = {}
    for name, body in extensions.items():
        if not isinstance(name, str) or not isinstance(body, dict):
            continue
        supported = body.get("supported", True)
        if type(supported) is not bool:
            return None
        support[name.lower()] = supported
    return support


@cache
def sail_profile_support(isa_profile: str) -> tuple[bool, str, str]:
    """Return exact Sail adjudicability for one declared ISA profile."""
    contract = REFERENCE_LOCK_CONFIG.sail_profile_contract(isa_profile)
    identity = "|".join(contract.config_files) if contract.config_files else ""
    if not contract.supported:
        return False, contract.reason, identity
    support = _sail_config_extension_support(isa_profile)
    if support is None:
        return False, "missing-sail-sample-config", identity
    declared = enabled_extensions(isa_profile) - {"g"}
    if "h" in declared:
        return False, "sail-config-extension-disabled:h", identity
    rv32 = str(isa_profile).lower().startswith("rv32")
    embedded_vector = "v" in declared or any(ext.startswith("zve") for ext in declared)
    sail_keys: list[str] = []
    for ext in declared:
        if ext in {"i", "e", "h", "v"} or ext.startswith(("zve", "zvl")):
            continue
        if ext == "c":
            sail_keys.append("zca")
            if rv32:
                if "f" in declared:
                    sail_keys.append("zcf")
            elif "d" in declared or "f" in declared:
                sail_keys.append("zcd")
        elif ext == "zicbo":
            sail_keys.extend(("zicbom", "zicbop", "zicboz"))
        else:
            sail_keys.append(ext)
    if embedded_vector:
        sail_keys.append("v")
    unknown = sorted(key for key in sail_keys if key not in support)
    if unknown:
        return False, "sail-config-unknown-extension:" + ",".join(unknown), identity
    disabled = sorted(key for key in sail_keys if support[key] is not True)
    if disabled:
        return False, "sail-config-extension-disabled:" + ",".join(disabled), identity
    return True, "supported", identity

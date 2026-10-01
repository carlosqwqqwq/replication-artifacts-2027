import hashlib
import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

_local_root = os.environ.get("RQ1_FRAMEWORK_LOCAL_ROOT")
LOCAL_ROOT = Path(_local_root).resolve() if _local_root else next(
    (
        ancestor.parent
        for ancestor in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
        if ancestor.name == "worktrees"
        and ancestor.parent.name.endswith(".local")
        and (
            ancestor.parent.parent / ancestor.parent.name.removesuffix(".local") / ".git"
        ).exists()
    ),
    REPOSITORY_ROOT.parent / f"{REPOSITORY_ROOT.name}.local",
)
EXTERNAL_ROOT = LOCAL_ROOT / "external"
SAIL_MEMORY_OVERRIDE_CONFIG_PATH = REPOSITORY_ROOT / "framework" / "configs" / "sail-rvgen-user.json"
FRAMEWORK_RUNS_ROOT = Path(
    os.environ.get("RQ1_FRAMEWORK_RUNS_ROOT")
    or LOCAL_ROOT / "framework-experiments" / "runs"
).resolve()
CURRENT_OFFICIAL_EVIDENCE_INDEX = (
    LOCAL_ROOT / "framework-experiments" / "current-official-evidence-index.json"
)
REFERENCE_CONFIG_IDENTITY_CONTRACT = "rvgen-reference-config-identity-v1"


def _reference_config_path(raw_path: Path | str) -> Path:
    """Resolve a recorded config path on the current execution plane.

    Freeze payloads are portable between the Windows control plane and the
    Linux lab.  A payload may therefore contain a Windows absolute path even
    when it is being validated inside the lab.  Use the repository/local-root
    basename as the stable boundary and resolve the suffix against the roots
    available on this plane.
    """
    raw = str(raw_path)
    normalized = raw.replace("\\", "/").removeprefix("//?/")
    components = [component for component in normalized.split("/") if component]
    # Freeze payloads may be produced from a nested clean worktree while the
    # execution plane exposes only its repository root.  Anchor stable
    # repository paths at ``framework`` before interpreting host-specific
    # root names; otherwise ``.../.local/m0-clean-*/framework/...`` cannot be
    # resolved inside Docker.
    if "framework" in components:
        framework_index = components.index("framework")
        return REPOSITORY_ROOT.joinpath(*components[framework_index:])
    if "external" in components:
        external_index = components.index("external")
        return EXTERNAL_ROOT.joinpath(*components[external_index + 1 :])
    for root, label in ((LOCAL_ROOT, LOCAL_ROOT.name), (REPOSITORY_ROOT, REPOSITORY_ROOT.name)):
        try:
            index = components.index(label)
        except ValueError:
            continue
        suffix = components[index + 1 :]
        return root.joinpath(*suffix)
    candidate = Path(raw_path)
    # Normalize relative local payload paths before the Windows extended-path
    # wrapper sees ``..`` components (nested clean snapshots use them).
    return candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()


def _reference_config_label(path: Path | str) -> str:
    resolved = _reference_config_path(path).resolve()
    for root, label in ((REPOSITORY_ROOT, "repo"), (LOCAL_ROOT, "local")):
        try:
            return f"{label}/{resolved.relative_to(root.resolve()).as_posix()}"
        except ValueError:
            continue
    return resolved.as_posix()


def reference_config_identity(paths: tuple[str, ...]) -> str:
    """Hash the exact reference-config bytes named by one realization."""
    records: list[dict[str, str]] = []
    for raw_path in sorted({str(path) for path in paths if str(path)}):
        path = _reference_config_path(raw_path)
        record = {"path": _reference_config_label(raw_path)}
        try:
            record["sha256"] = hashlib.sha256(
                windows_extended_length_path(path).read_bytes()
            ).hexdigest()
        except OSError as exc:
            record["error"] = type(exc).__name__
        records.append(record)
    payload = json.dumps(
        {"contract": REFERENCE_CONFIG_IDENTITY_CONTRACT, "files": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def windows_extended_length_path(path: Path | str) -> Path:
    candidate = Path(path)
    if os.name != "nt":
        return candidate
    text = str(candidate)
    if text.startswith("\\\\?\\"):
        return candidate
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text.lstrip("\\"))
    if candidate.is_absolute():
        return Path("\\\\?\\" + text)
    return candidate

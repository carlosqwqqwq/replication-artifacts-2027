"""独立 capsule 执行器的轻量分发入口。"""

from __future__ import annotations

from importlib import import_module
import json
import os
import sys
from pathlib import Path


_DISPATCH = {
    "--unicorn-bin": ("unicorn", "LIBUNICORN_PATH"),
    "--renode-bin": ("renode", "RENODE_DLL"),
    "--rax-bin": ("rax", "RAX_BINARY"),
}


def _run(name: str, argv: list[str], option: str, env_name: str) -> int:
    forwarded = []
    binary = None
    index = 0
    while index < len(argv):
        if argv[index] == option and index + 1 < len(argv):
            binary = argv[index + 1]
            index += 2
            continue
        forwarded.append(argv[index])
        index += 1

    previous = os.environ.get(env_name)
    previous_identity = os.environ.get("RV_UNICORN_TARGET_IDENTITY_PATH")
    previous_source_commit = os.environ.get("RV_UNICORN_SOURCE_COMMIT")
    if env_name == "LIBUNICORN_PATH":
        os.environ.pop("RV_UNICORN_TARGET_IDENTITY_PATH", None)
        os.environ.pop("RV_UNICORN_SOURCE_COMMIT", None)
    if binary:
        path = Path(binary).resolve()
        os.environ[env_name] = str(path.parent if env_name == "LIBUNICORN_PATH" else path)
        if env_name == "LIBUNICORN_PATH":
            identity = path.with_name("target-identity.json")
            if identity.is_file():
                os.environ["RV_UNICORN_TARGET_IDENTITY_PATH"] = str(identity)
                try:
                    source_commit = json.loads(identity.read_text(encoding="utf-8")).get("source_commit")
                except (OSError, json.JSONDecodeError, TypeError):
                    source_commit = None
                if isinstance(source_commit, str) and source_commit:
                    os.environ["RV_UNICORN_SOURCE_COMMIT"] = source_commit
    try:
        module = import_module(f"framework.adapters.capsule_{name}")
        return int(module.main(forwarded))
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
        if previous_identity is None:
            os.environ.pop("RV_UNICORN_TARGET_IDENTITY_PATH", None)
        else:
            os.environ["RV_UNICORN_TARGET_IDENTITY_PATH"] = previous_identity
        if previous_source_commit is None:
            os.environ.pop("RV_UNICORN_SOURCE_COMMIT", None)
        else:
            os.environ["RV_UNICORN_SOURCE_COMMIT"] = previous_source_commit


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for option, (name, env_name) in _DISPATCH.items():
        if option in argv:
            return _run(name, argv, option, env_name)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

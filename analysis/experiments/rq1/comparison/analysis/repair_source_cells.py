#!/usr/bin/env python3
"""Recollect selected source-coverage cells with the current collectors.

This is used after a source-only run when a collector dependency or a bounded
collector path was corrected.  It never edits a raw run root; repaired profiles
are written to a new derived directory and then copied into the existing
source-only output with the cell metadata replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis import coverage_offline


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe(value: str) -> str:
    return coverage_offline._safe(value)


def _summary_status(cells: list[dict[str, Any]]) -> str:
    source = [str(cell.get("status") or "gap") for cell in cells]
    unique = [
        str((cell.get("unique_coverage") or {}).get("status") or "gap")
        for cell in cells
    ]
    statuses = source + unique
    if all(item == "observed" for item in statuses):
        return "recorded"
    if any(item in {"observed", "partial"} for item in statuses):
        return "partial"
    return "gap"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="定向重算 source-coverage cell。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args(argv)

    source_output = args.source_output.resolve(strict=True)
    summary_path = source_output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    previous_repairs = summary.get("source_repairs")
    previous_repairs = previous_repairs if isinstance(previous_repairs, dict) else {}
    previous_cells = previous_repairs.get("cells")
    previous_cells = [str(item) for item in previous_cells] \
        if isinstance(previous_cells, list) else []
    previous_outputs = previous_repairs.get("repair_outputs")
    previous_outputs = [str(item) for item in previous_outputs] \
        if isinstance(previous_outputs, list) else []
    previous_output = previous_repairs.get("repair_output")
    if isinstance(previous_output, str) and previous_output:
        previous_outputs.append(previous_output)
    methods, targets, experiments = coverage_offline._load_launches(
        [args.manifest.resolve(strict=True)]
    )
    requested_methods = [method for method in args.methods if method in methods]
    requested_targets = [target for target in args.targets if target in targets]
    if not requested_methods or not requested_targets:
        raise SystemExit("requested source repair cell is not in the 7×7 manifest")
    runs = [run for experiment in experiments for run in experiment["runs"]]
    specs = coverage_offline._target_specs(experiments, targets)
    repair_output = args.repair_output.resolve()
    if repair_output.exists():
        raise SystemExit(f"repair output already exists: {repair_output}")
    repair_output.mkdir(parents=True, exist_ok=False)
    # The host ships the required runtime with the dependency tree.  Keep the
    # default here as a guard for direct invocation outside the pipeline.
    manifest_root = args.manifest.resolve().parents[3]
    deps = Path(os.environ.get("RQ1_DEPS", str(manifest_root / "deps")))
    os.environ.setdefault(
        "RQ1_DOTNET_RUNTIME", str(deps / "toolchains" / "dotnet-8")
    )
    repaired: list[dict[str, Any]] = []
    for method in requested_methods:
        method_runs = [run for run in runs if run["method"] == method]
        for target_id in requested_targets:
            target = specs.get((method, target_id), {"id": target_id})
            print(f"[source-repair] start {method} × {target_id}", flush=True)
            result = coverage_offline._collect_cell(
                repair_output, method, target_id, target, method_runs, None,
                include_guest_metrics=False,
            )
            repaired.append(result)
            _write_json(
                repair_output / "cells" / f"{_safe(method)}-{_safe(target_id)}.json",
                result,
            )
            print(
                f"[source-repair] done {method} × {target_id}: "
                f"SimSrcCov={result.get('status')}",
                flush=True,
            )

    by_key = {(cell.get("method"), cell.get("target")): cell
              for cell in summary.get("cells", [])
              if isinstance(cell, dict)}
    for result in repaired:
        key = (result.get("method"), result.get("target"))
        old = by_key.get(key)
        if old is None:
            raise RuntimeError(f"source summary is missing repair cell: {key}")
        profile_root = repair_output / "profiles" / _safe(str(key[0])) / _safe(str(key[1]))
        destination = source_output / "profiles" / _safe(str(key[0])) / _safe(str(key[1]))
        if not profile_root.is_dir():
            raise RuntimeError(f"repaired profile directory missing: {profile_root}")
        shutil.copytree(profile_root, destination, dirs_exist_ok=True)
        preserved_rv = old.get("rv_opcode_catalog_coverage")
        old_unique = old.get("unique_coverage")
        replacement = dict(result)
        if preserved_rv is not None:
            replacement["rv_opcode_catalog_coverage"] = preserved_rv
        if old_unique is not None:
            replacement["unique_coverage"] = old_unique
        by_key[key] = replacement

    cells = [by_key[(method, target)] for method in methods for target in targets]
    coverage_offline._add_unique_coverage(
        source_output, methods, targets, cells, specs,
    )
    summary["cells"] = cells
    summary["status"] = _summary_status(cells)
    repair_cells = [f"{method}×{target}" for method in requested_methods
                    for target in requested_targets]
    summary["source_repairs"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_output": str(repair_output),
        "repair_outputs": sorted(set([*previous_outputs, str(repair_output)])),
        "cells": sorted(set([*previous_cells, *repair_cells])),
        "dotnet_runtime": os.environ.get("RQ1_DOTNET_RUNTIME"),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary["source_repairs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

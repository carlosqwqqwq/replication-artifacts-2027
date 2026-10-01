#!/usr/bin/env python3
"""Index one raw-only run without reading trace/profile payloads.

The output is JSONL: one record per target/case/job raw record.  Paths remain
relative to the run root, so the index can be moved with the published data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def _relative(path: str | Path, root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _walk_trace_refs(value: Any, root: Path, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        path = value.get("trace_path") or value.get("absolute_path")
        if isinstance(path, str) and path:
            item = {"path": _relative(path, root) or path}
            for key in ("sha256", "bytes", "pc_count", "pc_digest", "format"):
                if key in value:
                    item[key] = value[key]
            if item not in out:
                out.append(item)
        for child in value.values():
            _walk_trace_refs(child, root, out)
    elif isinstance(value, list):
        for child in value:
            _walk_trace_refs(child, root, out)


def _raw_rows(run_root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    base = run_root / "framework/Ours-RVGEN-Direct/target-queues"
    if not base.is_dir():
        return
    for target_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        raw = target_dir / "raw/cases"
        if not raw.is_dir():
            continue
        for path in sorted(raw.rglob("*.json")):
            if path.name.endswith(".raw-artifact-manifest.json"):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict) or value.get("schema_version") != "rq1-target-raw-execution-v1":
                continue
            yield path, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for path, value in _raw_rows(root):
            if args.max_records and count >= args.max_records:
                break
            traces: list[dict[str, Any]] = []
            _walk_trace_refs(value.get("executions", {}), root, traces)
            coverage = value.get("simulator_coverage")
            manifest = value.get("raw_artifact_manifest")
            row = {
                "schema": "vernier-per-test-artifact-index-v1",
                "run_id": root.name,
                "target_id": value.get("target_id"),
                "case_root": value.get("case_root"),
                "job_id": value.get("job_id"),
                "raw_record": _relative(path, root),
                "coverage_raw_dir": coverage.get("raw_dir") if isinstance(coverage, dict) else None,
                "coverage_summary": coverage.get("summary_path") if isinstance(coverage, dict) else None,
                "coverage_profile": coverage.get("source_coverage", {}).get("profile") if isinstance(coverage, dict) and isinstance(coverage.get("source_coverage"), dict) else None,
                "raw_artifact_manifest": manifest.get("path") if isinstance(manifest, dict) else None,
                "raw_artifact_files": manifest.get("files", []) if isinstance(manifest, dict) else [],
                "instruction_traces": traces,
                "execution_status": value.get("executions", {}),
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(json.dumps({"records": count, "output": str(args.output), "run_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

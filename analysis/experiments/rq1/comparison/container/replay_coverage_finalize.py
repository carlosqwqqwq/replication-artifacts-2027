#!/usr/bin/env python3
"""Run deferred full-batch coverage after five-tool replay execution ends."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "as_of_utc", "data_age_seconds", "run_id", "run_status", "method", "target", "status",
    "coverage_sample_status", "sample_phase", "collector_status", "source_status",
    "attempted_cases", "profiles_with_data",
    "function_covered", "function_eligible", "function_value", "function_provisional_value",
    "line_covered", "line_eligible", "line_value", "line_provisional_value",
    "branch_covered", "branch_eligible", "branch_value", "branch_provisional_value",
    "gen_opcode_covered", "gen_opcode_eligible", "gen_opcode_value",
    "gen_opcode_provisional_value", "gen_opcode_status",
    "exec_opcode_covered", "exec_opcode_eligible", "exec_opcode_value",
    "exec_opcode_provisional_value", "exec_opcode_status", "legacy_rv_instruction_status",
    "target_outcomes", "target_statuses", "generated_candidates", "raw_candidates",
    "artifact_gaps", "target_attempts_total", "target_tested", "target_gaps",
]


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_progress_rows(path: Path) -> dict[str, dict]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return {row.get("target", ""): row for row in csv.DictReader(stream)}
    except (OSError, csv.Error):
        return {}


def summarize_status(values: list[str]) -> str:
    observed = [value for value in values if value]
    if not observed:
        return "missing"
    if all(value == "observed" for value in observed):
        return "observed"
    if all(value == "NA" for value in observed):
        return "NA"
    if all(value == "gap" for value in observed):
        return "gap"
    return "partial"


def summarize_metric_statuses(rows: dict[str, dict]) -> dict[str, str]:
    return {
        "SimSrcCov": summarize_status([row.get("source_status", "") for row in rows.values()]),
        "GenOpcodeCov": summarize_status([row.get("gen_opcode_status", "") for row in rows.values()]),
        "ExecOpcodeCov": summarize_status([row.get("exec_opcode_status", "") for row in rows.values()]),
    }


def build_matrix(launch_dir: Path, output_root: Path, run_prefix: str,
                 methods: list[str], targets: list[str]) -> dict:
    now = datetime.now(timezone.utc)
    rows = []
    for method in methods:
        run_id = f"{run_prefix}-{method}"
        run_root = output_root / "runs" / run_id
        progress_root = run_root / "coverage-progress"
        progress = read_json(progress_root / "latest.json")
        target_rows = read_progress_rows(progress_root / "latest.csv")
        wrapper = read_json(run_root / "wrapper-result.json")
        run_status = wrapper.get("status")
        if run_status not in {"completed", "partial", "failed"}:
            run_status = "failed" if (launch_dir / "completed" / f"{method}.done").is_file() else "running"
        for target in targets:
            item = target_rows.get(target, {})
            as_of = item.get("as_of_utc") or None
            age = None
            if as_of:
                try:
                    sampled = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                    if sampled.tzinfo is None:
                        sampled = sampled.replace(tzinfo=timezone.utc)
                    age = max(0, int((now - sampled.astimezone(timezone.utc)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    as_of = None
            row = {name: item.get(name) or None for name in FIELDS}
            row.update(
                as_of_utc=as_of, data_age_seconds=age, run_id=run_id,
                run_status=run_status, method=method, target=target,
                status=item.get("collector_status") or item.get("source_status")
                or item.get("coverage_sample_status") or "waiting",
                sample_phase=item.get("sample_phase") or progress.get("sample_phase"),
            )
            rows.append(row)
    expected = len(methods) * len(targets)
    reported = sum(row.get("as_of_utc") is not None for row in rows)
    metric_statuses = {
        "SimSrcCov": summarize_status([row.get("source_status") or "" for row in rows]),
        "GenOpcodeCov": summarize_status([row.get("gen_opcode_status") or "" for row in rows]),
        "ExecOpcodeCov": summarize_status([row.get("exec_opcode_status") or "" for row in rows]),
    }
    payload = {
        "schema_version": "rq1-existing-case-coverage-matrix-v1",
        "as_of_utc": now.isoformat(), "run_prefix": run_prefix,
        "methods": methods, "targets": targets,
        "expected_cells": expected, "reported_cells": reported,
        "status": (
            "provisional"
            if expected and reported == expected
            and all(status == "observed" for status in metric_statuses.values())
            else "partial"
        ),
        "metrics": ["SimSrcCov.function", "SimSrcCov.line", "SimSrcCov.branch",
                    "GenOpcodeCov", "ExecOpcodeCov"],
        "rv_opcode_catalog_denominator": 1480,
        "metric_statuses": metric_statuses,
        "rows": rows,
    }
    json_path = launch_dir / "coverage-matrix-latest.json"
    write_json(json_path, payload)
    csv_path = launch_dir / "coverage-matrix-latest.csv"
    temporary = csv_path.with_name(csv_path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    with (launch_dir / "coverage-matrix-checkpoints.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return payload


def sample_method(method: str, args) -> dict:
    run_id = f"{args.run_prefix}-{method}"
    run_root = args.output_root / "runs" / run_id
    snapshot = args.output_root / "source-snapshots" / run_id
    sampler = snapshot / "coverage_progress.py"
    if not (run_root / "execution-manifest.json").is_file() or not sampler.is_file():
        return {"status": "failed", "error": "run-or-source-snapshot-missing"}
    log_path = args.launch_dir / f"{method}.coverage-progress.log"
    environment = os.environ.copy()
    environment.update({
        "RQ1_DEPS": str(args.dependency_root),
        "RQ1_RUN_ROOT": str(run_root),
        "RQ1_COVERAGE_JOBS": "1",
        "RQ1_DOTNET_RUNTIME": str(args.dependency_root / "toolchains" / "dotnet-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = [
        sys.executable, str(sampler), "--run-root", str(run_root), "--method", method,
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{datetime.now(timezone.utc).isoformat()}] phase=post-execution-full-batch\n")
        log.flush()
        try:
            result = subprocess.run(command, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        except OSError as error:
            return {"status": "failed", "error": str(error)}
    progress = read_json(run_root / "coverage-progress" / "latest.json")
    progress_rows = read_progress_rows(run_root / "coverage-progress" / "latest.csv")
    metric_statuses = summarize_metric_statuses(progress_rows)
    missing_targets = sorted(set(args.targets) - set(progress_rows))
    unexpected_targets = sorted(set(progress_rows) - set(args.targets))
    if missing_targets or unexpected_targets:
        metric_statuses = {
            name: "partial" if status != "missing" else "missing"
            for name, status in metric_statuses.items()
        }
    valid = (
        result.returncode == 0
        and progress.get("status") == "provisional"
        and progress.get("sample_phase") == "post-execution-full-batch"
        and progress.get("run_id") == run_id
        and progress.get("method") == method
    )
    post_execution_coverage_status = (
        "provisional"
        if valid and metric_statuses
        and all(value == "observed" for value in metric_statuses.values())
        else "partial" if valid else "failed"
    )
    return {
        "status": "completed" if valid else "failed",
        "post_execution_coverage_status": post_execution_coverage_status,
        "return_code": result.returncode,
        "coverage_progress_status": progress.get("status"),
        "sample_phase": progress.get("sample_phase"),
        "sampled_at_utc": progress.get("as_of_utc"),
        "metric_statuses": metric_statuses,
        "missing_targets": missing_targets,
        "unexpected_targets": unexpected_targets,
        **({} if valid else {"error": "post-execution-full-batch-sample-failed"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()
    args.launch_dir = args.launch_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.dependency_root = args.dependency_root.resolve()
    if not args.launch_dir.is_dir() or not args.output_root.is_dir():
        parser.error("launch and output directories must already exist")

    final_rows = {method: {"status": "pending"} for method in args.methods}
    status_path = args.launch_dir / "coverage-final-sampling.json"

    def publish(status: str) -> None:
        write_json(status_path, {
            "schema_version": "rq1-replay-final-coverage-sampling-v1",
            "run_prefix": args.run_prefix, "status": status,
            "sample_phase": "post-execution-full-batch",
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "methods": final_rows,
        })

    publish("sampling")
    build_matrix(args.launch_dir, args.output_root, args.run_prefix, args.methods, args.targets)
    with ThreadPoolExecutor(max_workers=len(args.methods)) as pool:
        futures = {pool.submit(sample_method, method, args): method for method in args.methods}
        for future in as_completed(futures):
            method = futures[future]
            try:
                final_rows[method] = future.result()
            except Exception as error:
                final_rows[method] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            publish("sampling")
            build_matrix(args.launch_dir, args.output_root, args.run_prefix,
                         args.methods, args.targets)
    failures = sum(row.get("status") != "completed" for row in final_rows.values())
    publish("failed" if failures == len(args.methods)
            else "partial" if failures else "completed")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

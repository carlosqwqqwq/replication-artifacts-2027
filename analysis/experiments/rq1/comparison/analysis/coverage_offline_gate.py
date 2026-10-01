#!/usr/bin/env python3
"""Validate an offline 7×7 coverage result before it is reported as final.

The collector deliberately preserves ``partial`` and ``gap`` results.  This
small gate makes that distinction explicit: a summary can be structurally
valid and still be *not ready* for a final table.  A source-only summary marks
``guest_metrics_enabled=false``; in that mode RV opcode evidence is diagnostic
and does not block the SimSrcCov gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


METHODS = (
    "B-RVDV", "B-TORTURE", "B-CSMITH", "B-GEMI", "Fuzz4All",
    "Ours-RVGEN-Direct", "Ours-Program-Full",
)
TARGETS = (
    "T-QEMU", "T-LRSV-INT", "T-LRSV-TRANS", "T-UNICORN", "T-RENODE",
    "T-RAX", "T-RVVM",
)
SOURCE_METRICS = ("function", "line", "branch")
RV_METRICS = ("GenOpcodeCov", "ExecOpcodeCov")
ALLOWED_STATUS = {"observed", "partial", "gap", "NA"}
RV_DENOMINATOR = 1480


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _metric_status(
    metric: Mapping[str, Any] | None, label: str, structural: list[str],
) -> str:
    if not isinstance(metric, Mapping):
        structural.append(f"{label}:metric-missing")
        return "gap"
    status = metric.get("status")
    if status not in ALLOWED_STATUS:
        structural.append(f"{label}:invalid-status:{status}")
        return "gap"
    covered = metric.get("covered")
    eligible = metric.get("eligible")
    if covered is not None and (type(covered) is not int or covered < 0):
        structural.append(f"{label}:invalid-covered")
    if eligible is not None and (type(eligible) is not int or eligible < 0):
        structural.append(f"{label}:invalid-eligible")
    if type(covered) is int and type(eligible) is int and covered > eligible:
        structural.append(f"{label}:covered-exceeds-eligible")
    if status == "observed" and not isinstance(metric.get("value"), (int, float)):
        structural.append(f"{label}:observed-value-missing")
    return str(status)


def _check_report_files(
    report_dir: Path, *, require_growth: bool, require_rv: bool,
    structural: list[str],
) -> dict[str, Any]:
    files = {
        "final": report_dir / "final-coverage.csv",
        "unique": report_dir / "unique-coverage.csv",
        "rv_growth": report_dir / "rv-opcode-growth-exponential.csv",
        "simsrc_growth": report_dir / "simsrc-growth-exponential.csv",
    }
    rows: dict[str, int] = {}
    required = ("final", "unique", "rv_growth") if require_rv else ("final", "unique")
    for name in required:
        path = files[name]
        if not path.is_file():
            structural.append(f"report:{path.name}:missing")
            continue
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows[name] = max(0, sum(1 for _ in csv.DictReader(stream)))
        except (OSError, csv.Error, UnicodeError):
            structural.append(f"report:{path.name}:unreadable")
    if rows.get("final") != 49:
        structural.append(f"report:final-coverage.csv:rows={rows.get('final', 0)}")
    if rows.get("unique", 0) < 49:
        structural.append(f"report:unique-coverage.csv:rows={rows.get('unique', 0)}")
    if require_rv and rows.get("rv_growth", 0) == 0:
        structural.append("report:rv-opcode-growth-exponential.csv:empty")
    if require_growth:
        path = files["simsrc_growth"]
        if not path.is_file():
            structural.append(f"report:{path.name}:missing")
        else:
            try:
                with path.open(newline="", encoding="utf-8") as stream:
                    rows["simsrc_growth"] = max(0, sum(1 for _ in csv.DictReader(stream)))
            except (OSError, csv.Error, UnicodeError):
                structural.append(f"report:{path.name}:unreadable")
            if rows.get("simsrc_growth", 0) == 0:
                structural.append(f"report:{path.name}:empty")
    return {"directory": str(report_dir), "rows": rows}


def validate(
    summary: Mapping[str, Any], report_dir: Path | None = None, *,
    require_complete: bool = False, require_growth: bool = False,
) -> dict[str, Any]:
    structural: list[str] = []
    incomplete: list[str] = []
    guest_metrics_enabled = summary.get("guest_metrics_enabled", True) is not False
    if summary.get("status") not in {"recorded", "partial", "gap"}:
        structural.append(f"summary:invalid-status:{summary.get('status')}")
    methods = summary.get("methods")
    targets = summary.get("targets")
    if not isinstance(methods, list) or len(methods) != 7 \
            or len(set(methods)) != 7 or set(methods) != set(METHODS):
        structural.append("summary:methods-do-not-match-7-method-contract")
    if not isinstance(targets, list) or len(targets) != 7 \
            or len(set(targets)) != 7 or set(targets) != set(TARGETS):
        structural.append("summary:targets-do-not-match-7-target-contract")
    if summary.get("matrix_cells_expected") != 49:
        structural.append("summary:matrix_cells_expected-not-49")
    cells = summary.get("cells")
    if not isinstance(cells, list):
        cells = []
        structural.append("summary:cells-missing")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            structural.append("cell:not-object")
            continue
        key = (str(cell.get("method")), str(cell.get("target")))
        if key in by_key:
            structural.append(f"cell:duplicate:{key[0]}×{key[1]}")
        by_key[key] = cell
    if len(cells) != 49 or len(by_key) != 49:
        structural.append(f"summary:matrix_cells_written={len(cells)}")
    if summary.get("matrix_cells_written") != 49:
        structural.append(
            f"summary:matrix_cells_written-field={summary.get('matrix_cells_written')}"
        )

    status_counts: dict[str, Counter[str]] = {
        "cell": Counter(), "SimSrcCov": Counter(), "RV": Counter(),
        "unique": Counter(),
    }
    for method in METHODS:
        for target in TARGETS:
            label = f"{method}×{target}"
            cell = by_key.get((method, target))
            if cell is None:
                structural.append(f"cell:{label}:missing")
                continue
            cell_status = str(cell.get("status") or "gap")
            status_counts["cell"][cell_status] += 1
            if cell_status not in ALLOWED_STATUS | {"recorded"}:
                structural.append(f"cell:{label}:invalid-status:{cell_status}")
            if cell_status != "observed":
                incomplete.append(f"{label}:cell={cell_status}")

            source = cell.get("source_coverage")
            if not isinstance(source, Mapping):
                structural.append(f"{label}:SimSrcCov:missing")
                source = {}
            source_status = str(source.get("status") or "gap")
            status_counts["SimSrcCov"][source_status] += 1
            if source_status != "observed":
                incomplete.append(f"{label}:SimSrcCov={source_status}")
            if report_dir is not None and source_status in {"observed", "partial"}:
                profile_path = source.get("profile_path")
                if not isinstance(profile_path, str) or not profile_path:
                    structural.append(f"{label}:SimSrcCov:profile_path-missing")
                elif not (report_dir / profile_path).is_file():
                    structural.append(f"{label}:SimSrcCov:profile-missing:{profile_path}")
            source_metrics = source.get("metrics")
            if not isinstance(source_metrics, Mapping):
                structural.append(f"{label}:SimSrcCov:metrics-missing")
                source_metrics = {}
            for name in SOURCE_METRICS:
                status = _metric_status(
                    source_metrics.get(name), f"{label}:SimSrcCov:{name}", structural,
                )
                if status != "observed":
                    incomplete.append(f"{label}:SimSrcCov-{name}={status}")

            if guest_metrics_enabled:
                rv = cell.get("rv_opcode_catalog_coverage")
                if not isinstance(rv, Mapping):
                    structural.append(f"{label}:RV:missing")
                    rv = {}
                rv_status = str(rv.get("status") or "gap")
                status_counts["RV"][rv_status] += 1
                if rv_status != "observed":
                    incomplete.append(f"{label}:RV={rv_status}")
                if rv.get("denominator") != RV_DENOMINATOR:
                    structural.append(f"{label}:RV:denominator={rv.get('denominator')}")
                missing_cases = rv.get("missing_case_count")
                if type(missing_cases) is not int or missing_cases < 0:
                    structural.append(f"{label}:RV:missing_case_count-invalid")
                elif missing_cases:
                    incomplete.append(f"{label}:RV:missing_cases={missing_cases}")
                rv_metrics = rv.get("metrics")
                if not isinstance(rv_metrics, Mapping):
                    structural.append(f"{label}:RV:metrics-missing")
                    rv_metrics = {}
                for name in RV_METRICS:
                    status = _metric_status(
                        rv_metrics.get(name), f"{label}:RV:{name}", structural,
                    )
                    metric = rv_metrics.get(name)
                    if isinstance(metric, Mapping) and metric.get("eligible") != RV_DENOMINATOR:
                        structural.append(
                            f"{label}:RV:{name}:eligible={metric.get('eligible')}"
                        )
                    if status != "observed":
                        incomplete.append(f"{label}:RV-{name}={status}")
            else:
                status_counts["RV"]["disabled"] += 1

            unique = cell.get("unique_coverage")
            if not isinstance(unique, Mapping):
                structural.append(f"{label}:unique:missing")
                unique_status = "gap"
            else:
                unique_status = str(unique.get("status") or "gap")
                if unique_status not in ALLOWED_STATUS:
                    structural.append(f"{label}:unique:invalid-status:{unique_status}")
            status_counts["unique"][unique_status] += 1
            if unique_status != "observed":
                incomplete.append(f"{label}:unique={unique_status}")

    report = None
    if report_dir is not None:
        report = _check_report_files(
            report_dir, require_growth=require_growth,
            require_rv=guest_metrics_enabled, structural=structural,
        )
    if require_complete and incomplete:
        structural.extend(f"final-gate:{item}" for item in incomplete[:200])
    status = "invalid" if structural else "ready" if not incomplete else "partial"
    return {
        "schema_version": "rq1-offline-coverage-gate-v1",
        "guest_metrics_enabled": guest_metrics_enabled,
        "status": status,
        "final_ready": status == "ready",
        "structural_issue_count": len(structural),
        "incomplete_metric_count": len(incomplete),
        "status_counts": {key: dict(sorted(value.items())) for key, value in status_counts.items()},
        "issues": structural[:200],
        "incomplete": incomplete[:200],
        "report": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 RQ1 7×7 离线覆盖率结果是否可封存。")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-source-growth", action="store_true")
    args = parser.parse_args(argv)
    summary = _read_json(args.summary)
    result = validate(
        summary, args.report_dir,
        require_complete=args.require_complete,
        require_growth=args.require_source_growth,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

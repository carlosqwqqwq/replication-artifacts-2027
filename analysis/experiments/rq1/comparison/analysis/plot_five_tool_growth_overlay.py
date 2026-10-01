#!/usr/bin/env python3
"""Replace five-tool terminal reference lines with within-run case curves."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


REPO = Path("/path/to/rq1-comparison")
HERE = REPO / "results/coverage-reanalysis-20260921/convergence-7x7"
FIGURES = REPO / "results/coverage-aggregates/mcmc-8h-cellwise-v4/figures"
METHODS = ("B-RVDV", "B-TORTURE", "B-CSMITH", "B-GEMI", "Fuzz4All")
TARGETS = ("T-QEMU", "T-LRSV-INT", "T-LRSV-TRANS", "T-UNICORN", "T-RENODE", "T-RAX", "T-RVVM")
METRICS = ("Function", "Line", "Branch", "GenOpcodeCov", "ExecOpcodeCov")
COLORS = {
    "B-RVDV": "#009E73",
    "B-TORTURE": "#CC79A7",
    "B-CSMITH": "#E69F00",
    "B-GEMI": "#56B4E9",
    "Fuzz4All": "#444444",
}
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def load_growth_helper():
    path = HERE / "coverage_growth_7x7.py"
    spec = importlib.util.spec_from_file_location("rq1_growth_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-growth-helper:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoints(total: int) -> list[int]:
    values = []
    n = 1
    while n <= total:
        values.append(n)
        n *= 2
    if total and (not values or values[-1] != total):
        values.append(total)
    return values


def load_opcode_growth() -> list[dict]:
    helper = load_growth_helper()
    merged = json.loads((REPO / "results/coverage-reanalysis-20260921/merged-two-batches/coverage-7x7-merged.json").read_text())
    batch = next(item for item in merged["batches"] if item["batch"] == "35d486")
    run_ids = {item["method"]: item["run_id"] for item in batch["runs"]}
    rows: list[dict] = []

    for method in METHODS:
        root = REPO / "results/runs" / run_ids[method]
        ledger = root / "ledger/events.partial.jsonl"
        if not ledger.is_file():
            ledger = root / "ledger/events.jsonl"
        by_target: dict[str, list[dict]] = defaultdict(list)
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("target_attempted") is True and event.get("target") in TARGETS:
                by_target[event["target"]].append(event)

        for target in TARGETS:
            events = sorted(by_target[target], key=lambda item: item.get("stream_index", -1))
            indexes = [item.get("stream_index") for item in events]
            if any(type(index) is not int or index < 0 for index in indexes) or len(indexes) != len(set(indexes)):
                raise ValueError(f"invalid-case-order:{method}:{target}")
            cached = []
            for event, index in zip(events, indexes):
                candidate = index
                if candidate is None:
                    match = re.fullmatch(r"candidate-(\d+)", str(event.get("case_id") or ""))
                    candidate = int(match.group(1)) if match else None
                cached.append(helper.read_catalog(root, method, target, event, candidate))

            cumulative = {"GenOpcodeCov": set(), "ExecOpcodeCov": set()}
            metric_statuses = {name: [] for name in cumulative}
            evidence = 0
            rows.append({
                "family": "RV Opcode", "method": method, "target": target,
                "cases": 0, "evidence_cases": 0,
                "metric": "GenOpcodeCov", "covered": 0, "eligible": 1480,
                "coverage_pct": 0.0, "status": "waiting", "batch": "35d486",
            })
            rows.append({
                "family": "RV Opcode", "method": method, "target": target,
                "cases": 0, "evidence_cases": 0,
                "metric": "ExecOpcodeCov", "covered": 0, "eligible": 1480,
                "coverage_pct": 0.0, "status": "waiting", "batch": "35d486",
            })

            prior = 0
            for count in checkpoints(len(events)):
                for case in cached[prior:count]:
                    if case is None:
                        continue
                    evidence += 1
                    for metric in cumulative:
                        cumulative[metric].update(case[metric]["units"])
                        metric_statuses[metric].append(case[metric]["status"])
                for metric, covered_units in cumulative.items():
                    status = (
                        "gap" if evidence == 0
                        else "observed" if evidence == count and all(s == "observed" for s in metric_statuses[metric])
                        else "partial"
                    )
                    rows.append({
                        "family": "RV Opcode", "method": method, "target": target,
                        "cases": count, "evidence_cases": evidence, "metric": metric,
                        "covered": len(covered_units), "eligible": 1480,
                        "coverage_pct": 100 * len(covered_units) / 1480,
                        "status": status, "batch": "35d486",
                    })
                prior = count

    # The two previously audited targets must agree with their saved per-case curves.
    existing = {}
    with (HERE / "coverage-growth-within-run-7x7-all-cases.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            existing[(row["method"], row["target"], int(row["cases"]))] = row
    for row in rows:
        if row["family"] != "RV Opcode" or row["cases"] == 0 or row["target"] not in {"T-QEMU", "T-LRSV-TRANS"}:
            continue
        expected = existing[(row["method"], row["target"], row["cases"])][f"{row['metric']}_covered"]
        if row["covered"] != int(expected):
            raise ValueError(f"opcode-growth-mismatch:{row['method']}:{row['target']}:{row['metric']}:{row['cases']}")
    return rows


def load_source_growth() -> list[dict]:
    rows: list[dict] = []
    source_path = HERE / "coverage-growth-within-run-7x7-all-cases.csv"
    source_cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with source_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["method"] not in METHODS or row["target"] not in {"T-QEMU", "T-LRSV-TRANS"}:
                continue
            source_cells[(row["method"], row["target"])].append(row)
            for metric in ("Function", "Line", "Branch"):
                key = metric.lower()
                covered = int(row[f"{key}_covered"])
                eligible = int(row[f"{key}_eligible"])
                rows.append({
                    "family": "SimSrcCov", "method": row["method"], "target": row["target"],
                    "cases": int(row["cases"]), "evidence_cases": int(row["source_profiles_with_data"]),
                    "metric": metric, "covered": covered, "eligible": eligible,
                    "coverage_pct": 100 * covered / eligible, "status": row[f"{key}_status"],
                    "batch": "35d486",
                })
    for (method, target), cell_rows in source_cells.items():
        terminal = max(cell_rows, key=lambda row: int(row["cases"]))
        if terminal["source_endpoint_matches_saved_summary"] != "True":
            raise ValueError(f"source-growth-endpoint-not-validated:{method}:{target}")

    # Reuse any complete, endpoint-matching RAX source cells already collected.
    merged = json.loads((REPO / "results/coverage-reanalysis-20260921/merged-two-batches/coverage-7x7-merged.json").read_text())
    helper = load_growth_helper()
    import sys

    sys.path.insert(0, str(REPO / "source/rq1-comparison/experiments/rq1/comparison"))
    from analysis import coverage_offline as co

    manifest = REPO / "results/launch-logs/rq1-7x7-20260920154727-35d486/launch-manifest.json"
    _, targets, experiments = co._load_launches([manifest.resolve()])
    target_specs = co._target_specs(experiments, targets)
    for method in METHODS:
        cell = HERE / "all-target-growth-35d486" / method / "T-RAX"
        marker = cell / "cell-complete.json"
        result = cell / f"{method}-T-RAX.json"
        if not marker.is_file() or not result.is_file():
            continue
        metadata = json.loads(marker.read_text())
        if metadata.get("result_sha256") != hashlib.sha256(result.read_bytes()).hexdigest():
            raise ValueError(f"source-growth-result-hash-mismatch:{method}:T-RAX")
        cell_rows = json.loads(result.read_text())
        expected_cell = next(r for r in merged["rows"] if r["method"] == method and r["target"] == "T-RAX")
        source_input = next(i for i in expected_cell["simulator_source_coverage"]["inputs"] if i.get("batch") == "35d486")
        expected_units = helper.lcov_units(Path(source_input["profile_path"]), target_specs[(method, "T-RAX")])
        for metric in ("Function", "Line", "Branch"):
            key = metric.lower()
            endpoint = cell_rows[-1]
            expected_covered = len(expected_units[key]["covered"])
            expected_eligible = len(expected_units[key]["eligible"])
            if endpoint[f"SimSrcCov_{key}_covered"] != expected_covered or endpoint[f"SimSrcCov_{key}_eligible"] != expected_eligible:
                raise ValueError(f"source-growth-endpoint-mismatch:{method}:T-RAX:{metric}")
            for item in cell_rows:
                eligible = item.get(f"SimSrcCov_{key}_eligible")
                covered = item.get(f"SimSrcCov_{key}_covered")
                if eligible is None:
                    continue
                rows.append({
                    "family": "SimSrcCov", "method": method, "target": "T-RAX",
                    "cases": item["case_checkpoint"], "evidence_cases": item["represented_cases"],
                    "metric": metric, "covered": covered, "eligible": eligible,
                    "coverage_pct": 100 * covered / eligible,
                    "status": item[f"SimSrcCov_{key}_status"], "batch": "35d486",
                })
    return rows


def save_data(rows: list[dict]) -> None:
    path = HERE / "external-case-growth-35d486.csv"
    fields = ("family", "method", "target", "cases", "evidence_cases", "metric", "covered", "eligible", "coverage_pct", "status", "batch")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["family"], r["target"], r["metric"], METHODS.index(r["method"]), r["cases"])))


def text_of(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def panel_layout(root: ET.Element) -> dict[str, dict]:
    texts = list(root.iter(f"{{{SVG}}}text"))
    headers = [node for node in texts if text_of(node).startswith("T-") and "(" in text_of(node)]
    result = {}
    for header in headers:
        title = text_of(header)
        target = title.split(" ", 1)[0]
        header_y = float(header.attrib["y"])
        tick_y = "347" if header_y < 300 else "697"
        center = float(header.attrib["x"])
        x_ticks = {}
        for node in texts:
            if node.attrib.get("y") == tick_y and abs(float(node.attrib["x"]) - center) <= 150:
                label = text_of(node).replace(",", "")
                if label.isdigit():
                    x_ticks[int(label)] = float(node.attrib["x"])
        if 0 not in x_ticks or 1 not in x_ticks or 2 not in x_ticks:
            continue
        scale = x_ticks[2] - x_ticks[1]
        ymax_match = re.search(r"\(0[–-]([0-9.]+)%\)", title)
        ymax = float(ymax_match.group(1)) if ymax_match else None
        bottom = (float(tick_y) - 17) if tick_y == "347" else (float(tick_y) - 17)
        result[target] = {
            "left": x_ticks[0], "offset": x_ticks[1] - x_ticks[0],
            "scale": scale, "top": bottom - 180, "bottom": bottom,
            "ymax": ymax,
        }
    return result


def point_xy(panel: dict, cases: int, percentage: float) -> tuple[float, float]:
    x = panel["left"] if cases == 0 else panel["left"] + panel["offset"] + panel["scale"] * math.log2(cases)
    y = panel["bottom"] - percentage / panel["ymax"] * (panel["bottom"] - panel["top"])
    return x, y


def add_series(root: ET.Element, panel: dict, method: str, target: str, metric: str, rows: list[dict]) -> None:
    data = [row for row in rows if row["method"] == method and row["target"] == target and row["metric"] == metric]
    data.sort(key=lambda row: row["cases"])
    if not data or not any(row["evidence_cases"] for row in data):
        return
    values = []
    first_observed = next((row for row in data if row["evidence_cases"] and row["cases"] > 0), None)
    if first_observed is None:
        return
    if first_observed["cases"] == 1:
        values.append((0, 0.0, "waiting", 0, 1480))
    for row in data:
        if row["cases"] == 0 or row["evidence_cases"] == 0:
            continue
        values.append((row["cases"], row["coverage_pct"], row["status"], row["covered"], row["eligible"]))
    if not values:
        return
    xy = [point_xy(panel, cases, pct) for cases, pct, *_ in values]
    group = ET.SubElement(root, f"{{{SVG}}}g", {
        "class": "external-growth-series", "data-method": method,
        "data-target": target, "data-metric": metric,
    })
    d = " ".join(("M" if i == 0 else "L") + f" {x:.2f},{y:.2f}" for i, (x, y) in enumerate(xy))
    path = ET.SubElement(group, f"{{{SVG}}}path", {
        "class": "external-growth-line", "d": d, "fill": "none",
        "stroke": COLORS[method], "stroke-width": "1.65",
        "stroke-dasharray": "4 2.4", "stroke-linecap": "round",
        "stroke-linejoin": "round", "opacity": "0.95",
    })
    title = ET.SubElement(path, f"{{{SVG}}}title")
    title.text = f"{method} / {target} / {metric}; cumulative 35d486 case-prefix coverage"
    for (cases, pct, status, covered, eligible), (x, y) in zip(values, xy):
        marker = ET.SubElement(group, f"{{{SVG}}}circle", {
            "class": "external-growth-point", "cx": f"{x:.2f}", "cy": f"{y:.2f}",
            "r": "2.35", "fill": "white" if status == "partial" else COLORS[method],
            "stroke": COLORS[method], "stroke-width": "1.1",
        })
        marker_title = ET.SubElement(marker, f"{{{SVG}}}title")
        marker_title.text = f"{method}, {target}, cases={cases}, {covered}/{eligible} ({pct:.2f}%), {status}"


def update_chart(metric: str, rows: list[dict]) -> None:
    filename = {
        "Function": "simsrc-function-growth-with-five-tools.svg",
        "Line": "simsrc-line-growth-with-five-tools.svg",
        "Branch": "simsrc-branch-growth-with-five-tools.svg",
        "GenOpcodeCov": "rv-genopcodecov-growth-with-five-tools.svg",
        "ExecOpcodeCov": "rv-execopcodecov-growth-with-five-tools.svg",
    }[metric]
    path = FIGURES / filename
    tree = ET.parse(path)
    root = tree.getroot()
    panels = panel_layout(root)
    metric_family = "SimSrcCov" if metric in {"Function", "Line", "Branch"} else "RV Opcode"
    for element in list(root.iter(f"{{{SVG}}}g")):
        if element.attrib.get("class") == "external-growth-series":
            parent = next(node for node in root.iter() if element in list(node))
            parent.remove(element)
    for element in list(root.iter(f"{{{SVG}}}line")):
        if element.attrib.get("opacity") == "0.92" and element.attrib.get("stroke") in COLORS.values():
            parent = next(node for node in root.iter() if element in list(node))
            parent.remove(element)

    for header in list(root.iter(f"{{{SVG}}}text")):
        label = text_of(header)
        if label == "Run-case prefix (log₂; unique case_root; zero baseline)":
            header.text = "Run-case prefix (log₂; zero baseline)"
            for child in list(header):
                header.remove(child)
        elif label == "Each Target uses a local x/y scale; all x values count unique completed case_root prefixes.":
            header.text = "Ours counts completed case_root prefixes; comparators count target-attempted cases. Target panels use local log₂ scales."
            for child in list(header):
                header.remove(child)
        if label == "T-RENODE (baseline endpoints)" and metric_family == "SimSrcCov":
            header.text = "T-RENODE (source growth unavailable)"
            for child in list(header):
                header.remove(child)
        elif label in {
            "Legacy LCOV baselines only; Ours uses corrected scope",
            "Renode SimSrc has no comparable Ours case-prefix series",
        }:
            parent = next(node for node in root.iter() if header in list(node))
            parent.remove(header)
        elif label.startswith(("Dashed lines: historical five-tool final coverage union", "Comparators:")):
            if metric_family == "SimSrcCov":
                header.text = "Comparators: 35d486 cumulative case prefixes; SimSrc data: QEMU/LRSV-TRANS/RAX (five tools); hollow markers show incomplete profiles."
            else:
                header.text = "Comparators: 35d486 cumulative case prefixes from validated catalogs; T-UNICORN has no catalog data; hollow markers show incomplete evidence."
            for child in list(header):
                header.remove(child)

    description = next((node for node in root.iter(f"{{{SVG}}}desc") if node.attrib.get("id") == "figdesc"), None)
    if description is not None:
        description.text = (
            "Cumulative case-prefix coverage. Ours counts completed case_root prefixes; "
            "the five comparators count target-attempted cases in the independent 35d486 run. "
            + (
                "SimSrc comparator trajectories are available for QEMU, LRSV-TRANS, and RAX; "
                "Renode Ours has session-level scope-corrected coverage only. "
                if metric_family == "SimSrcCov" else
                "Opcode comparator trajectories use identity-validated per-case catalogs; "
                "T-UNICORN has no per-case catalog evidence. "
            )
            + "Hollow markers indicate incomplete prefix evidence. Each Target panel uses local log2 x and y scales."
        )

    for target, panel in panels.items():
        if panel["ymax"] is None:
            continue
        for method in METHODS:
            add_series(root, panel, method, target, metric, rows)

    tree.write(path, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    rows = load_source_growth() + load_opcode_growth()
    save_data(rows)
    for metric in METRICS:
        update_chart(metric, rows)
    print(f"wrote {len(rows)} case-prefix rows and updated five growth charts")


if __name__ == "__main__":
    main()

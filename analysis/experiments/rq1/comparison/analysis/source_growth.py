#!/usr/bin/env python3
"""Build a SimSrcCov growth curve with one pass over ordered case profiles.

Each case profile is collected once.  The collector output is reduced to
stable LCOV unit sets, those sets are accumulated in memory, and exponential
checkpoints only snapshot the accumulator.  Raw experiment profiles are never
modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import sys

COMPARISON = Path("/path/to/rq1-comparison/source/rq1-comparison/experiments/rq1/comparison")
sys.path.insert(0, str(COMPARISON))
from analysis import coverage_offline as co  # noqa: E402
from analysis import simulator_coverage  # noqa: E402
from analysis import source_coverage  # noqa: E402


METRICS = ("function", "line", "branch")
_GCOV_CREATING = re.compile(r"Creating '([^']+)'", re.IGNORECASE)


def checkpoints(maximum: int) -> list[int]:
    values = [0]
    value = 1
    while value < maximum:
        values.append(value)
        value *= 2
    if maximum and values[-1] != maximum:
        values.append(maximum)
    return values


def ordered_roots(run_inputs: list[dict[str, Any]]) -> list[Path]:
    ordered: list[Path] = []
    for item in run_inputs:
        roots = [
            Path(value) for value in (
                item.get("profile_data_roots")
                or item.get("profile_roots")
                or ()
            )
        ]
        metadata = item.get("profile_case_metadata") or {}
        roots.sort(key=lambda path: (
            metadata.get(path, {}).get("stream_index")
            if isinstance(metadata.get(path), dict)
            and type(metadata.get(path, {}).get("stream_index")) is int
            else 1 << 60,
            str(path),
        ))
        ordered.extend(path for path in roots if path.is_dir())
    return ordered


def _empty_metrics(status: str, reason: str | None = None) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "covered": 0,
            "eligible": None,
            "value": None,
            "status": status,
            **({"reason": reason} if reason else {}),
        }
        for name in METRICS
    }


def _unit_tuple(value: object) -> tuple[Any, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    return tuple(value)


def _gcov_source_path(value: object, source_root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return str(path.resolve())
    if path.is_absolute() and source_root.name in path.parts:
        suffix = path.parts[path.parts.index(source_root.name) + 1:]
        candidate = source_root.joinpath(*suffix)
        if candidate.is_file():
            return str(candidate.resolve())
    for candidate in (source_root / path, source_root.parent / path):
        if candidate.is_file():
            return str(candidate.resolve())
    return str((source_root / path).resolve())


def _gcov_note_index(build_root: Path) -> tuple[
    dict[tuple[str, ...], list[Path]], dict[str, list[Path]]
]:
    by_suffix: dict[tuple[str, ...], list[Path]] = {}
    by_name: dict[str, list[Path]] = {}
    for note in build_root.rglob("*.gcno"):
        by_name.setdefault(note.name, []).append(note)
        relative = note.relative_to(build_root)
        parts = relative.parts
        for length in range(1, len(parts) + 1):
            by_suffix.setdefault(parts[-length:], []).append(note)
    return by_suffix, by_name


def _match_gcov_note(
    root: Path, gcda: Path, build_root: Path,
    by_suffix: dict[tuple[str, ...], list[Path]],
    by_name: dict[str, list[Path]],
) -> Path | None:
    relative = gcda.relative_to(root)
    direct = build_root / relative.with_suffix(".gcno")
    if direct.is_file():
        return direct
    matches = by_name.get(gcda.stem + ".gcno", ())
    if len(matches) == 1:
        return matches[0]
    parts = relative.with_suffix(".gcno").parts
    for length in range(1, len(parts) + 1):
        candidates = tuple(dict.fromkeys(by_suffix.get(parts[-length:], ())))
        if len(candidates) == 1:
            return candidates[0]
    return None


def _attach_gcov_notes(
    roots: list[Path], spec: dict[str, Any], stage_dir: Path,
) -> tuple[list[Path], list[bool], dict[str, Any]]:
    """Attach one NVMe-cached .gcno per staged .gcda and return inputs by case."""
    build_root = co._dependency_path(spec.get("build_root")) or Path()
    notes_dir = stage_dir / "gcno-cache"
    notes_dir.mkdir(parents=True, exist_ok=True)
    if not build_root.is_dir():
        return [], [False] * len(roots), {"gcno_status": "build-root-missing"}
    by_suffix, by_name = _gcov_note_index(build_root)
    note_cache: dict[Path, Path] = {}
    inputs: list[Path] = []
    case_ok = [True] * len(roots)
    case_inputs: list[list[Path]] = [[] for _ in roots]
    source_bytes = 0
    for index, root in enumerate(roots):
        gcda_paths = sorted(
            path for path in root.rglob("*.gcda")
            if not path.name.endswith(".tmp.gcda") and path.is_file()
        )
        if not gcda_paths:
            case_ok[index] = False
        if any(root.rglob(source_coverage.GCOV_FLUSH_INCOMPLETE_MARKER)):
            case_ok[index] = False
        for gcda in gcda_paths:
            if gcda.stat().st_size == 0:
                case_ok[index] = False
                continue
            note = _match_gcov_note(root, gcda, build_root, by_suffix, by_name)
            if note is None:
                case_ok[index] = False
                continue
            cached = note_cache.get(note)
            if cached is None:
                cached = notes_dir / f"note-{len(note_cache):08d}.gcno"
                shutil.copyfile(note, cached)
                os.utime(cached, None)
                note_cache[note] = cached
                source_bytes += cached.stat().st_size
            paired = gcda.with_suffix(".gcno")
            if not paired.exists():
                paired.symlink_to(cached)
            case_inputs[index].append(gcda)
            inputs.append(gcda)
    for index, values in enumerate(case_inputs):
        if not values:
            case_ok[index] = False
    return inputs, case_ok, {
        "gcno_status": "observed",
        "gcno_count": len(note_cache),
        "gcno_bytes": source_bytes,
        "gcda_count": len(inputs),
    }


def _note_gcov_json(
    report: Path,
    case_index: int,
    source_root: Path,
    scopes: list[str],
    excludes: list[str],
    first_seen: dict[str, dict[tuple[Any, ...], int]],
    eligible: dict[str, set[tuple[Any, ...]]],
    eligible_first_seen: dict[str, dict[tuple[Any, ...], int]],
) -> bool:
    """Read one GCC JSON report and retain only earliest-hit unit indices."""
    try:
        with gzip.open(report, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return False
    records = 0
    source_occurrences: dict[str, int] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        source = _gcov_source_path(item.get("file"), source_root)
        if not source:
            continue
        source_key = simulator_coverage._source_key(source, str(source_root))
        in_scope = simulator_coverage._in_scope(source, scopes, str(source_root), excludes)
        source_occurrences[source_key] = source_occurrences.get(source_key, 0) + 1
        records += 1
        for function in item.get("functions", ()):
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            line = source_coverage._safe_count(function.get("start_line", 0))
            unit = (source_key, f"{line}:{function['name']}")
            if in_scope:
                eligible["function"].add(unit)
                eligible_first_seen["function"].setdefault(unit, case_index)
                if source_coverage._safe_count(function.get("execution_count", 0)) > 0:
                    first_seen["function"].setdefault(unit, case_index)
        for line_record in item.get("lines", ()):
            if not isinstance(line_record, dict):
                continue
            number = source_coverage._safe_count(line_record.get("line_number", 0))
            line_unit = (source_key, str(number))
            if in_scope:
                eligible["line"].add(line_unit)
                eligible_first_seen["line"].setdefault(line_unit, case_index)
                count = line_record.get("count", 0)
                if isinstance(count, (int, float)) and count < 0:
                    count = 0
                if isinstance(count, str) and count.strip().startswith("-"):
                    count = 0
                if source_coverage._safe_count(count) > 0:
                    first_seen["line"].setdefault(line_unit, case_index)
            for branch_index, branch in enumerate(line_record.get("branches", ())):
                if not isinstance(branch, dict):
                    continue
                branch_unit = (source_key, f"{number},-,{branch_index}")
                if in_scope:
                    eligible["branch"].add(branch_unit)
                    eligible_first_seen["branch"].setdefault(branch_unit, case_index)
                    count = branch.get("count", 0)
                    if isinstance(count, (int, float)) and count < 0:
                        count = 0
                    if isinstance(count, str) and count.strip().startswith("-"):
                        count = 0
                    if source_coverage._safe_count(count) > 0:
                        first_seen["branch"].setdefault(branch_unit, case_index)
    return records > 0


def _read_case(
    output: Path,
    temporary_parent: Path,
    target: dict[str, Any],
    binary: Path | None,
    root: Path,
    index: int,
    identity_cache: dict[str, dict[str, Any]] | None = None,
    identity_cache_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    """Collect and reduce one case, leaving only compact unit sets in memory."""
    spec = dict(target.get("source_coverage") or {})
    scopes = [str(item) for item in spec.get("source_scope", []) if str(item).strip()]
    excludes = [
        str(item) for item in spec.get("source_scope_exclude", []) if str(item).strip()
    ]
    source_root = str(co._dependency_path(spec.get("source_root")) or spec.get("source_root") or "")
    collector = str(spec.get("collector", "gcov"))
    # Temporary LCOV and identity files are derived data.  The original case
    # profile remains in the staged cell and is cleaned only by the staging
    # context after the whole cell is complete.
    with tempfile.TemporaryDirectory(
        prefix=f"src-growth-case-{index:08d}-",
        dir=str(temporary_parent),
    ) as temporary:
        case_dir = Path(temporary)
        local_spec = dict(spec)
        local_spec["profile"] = str(case_dir / "lcov.info")
        local_spec["identity"] = str(case_dir / "identity.json")
        local_target = {**target, "source_coverage": local_spec}
        try:
            collection = source_coverage.collect_source_coverage_batch(
                case_dir,
                local_target,
                binary,
                root,
                expected_coverage_cases=1,
                identity_cache=identity_cache,
                identity_cache_lock=identity_cache_lock,
                compute_input_fingerprint=False,
            )
        except Exception as error:  # keep one malformed case from aborting a cell
            return {
                "index": index,
                "status": "gap",
                "reason": f"one-pass-collector:{type(error).__name__}",
                "metrics": _empty_metrics("gap", f"one-pass-collector:{type(error).__name__}"),
                "units": {name: {"eligible": set(), "covered": set()} for name in METRICS},
            }
        collection_status = str(collection.get("status") or "gap")
        collection_reason = collection.get("reason")
        profile = case_dir / "lcov.info"
        if collection_status not in {"observed", "partial"} or not profile.is_file():
            reason = str(collection_reason or f"source-coverage-collection-{collection_status}")
            return {
                "index": index,
                "status": "gap",
                "reason": reason,
                "metrics": _empty_metrics("gap", reason),
                "units": {name: {"eligible": set(), "covered": set()} for name in METRICS},
            }
        parsed = simulator_coverage.summarize_lcov(
            profile,
            scopes,
            source_root,
            excludes,
            include_units=True,
            units_only=True,
        )
        raw_units = parsed.get("unit_sets") if isinstance(parsed, dict) else None
        units = {name: {"eligible": set(), "covered": set()} for name in METRICS}
        if isinstance(raw_units, dict):
            for name in METRICS:
                item = raw_units.get(name)
                if not isinstance(item, dict):
                    continue
                for key in ("eligible", "covered"):
                    values = item.get(key)
                    if isinstance(values, list):
                        units[name][key].update(
                            unit for value in values
                            if (unit := _unit_tuple(value)) is not None
                        )
        status = str(parsed.get("status") or collection_status)
        if collection_status == "partial" and status == "observed":
            status = "partial"
        reason = parsed.get("reason") or collection_reason
        return {
            "index": index,
            "status": status,
            "reason": reason,
            "metrics": parsed.get("metrics") if isinstance(parsed.get("metrics"), dict) else _empty_metrics(status, reason),
            "units": units,
        }


def _stage_prefix_inputs(
    roots: list[Path], collector: str, checkpoint: int, destination: Path,
) -> list[int]:
    """Expose one ordered case prefix to a batch collector using symlinks.

    The raw files were already copied to the NVMe cell stage by
    ``_stage_cell_inputs``.  Prefix directories therefore add only directory
    entries and never copy the large profile payloads again.
    """
    counts: list[int] = []
    for index, root in enumerate(roots[:checkpoint], 1):
        files = [
            path for path in source_coverage.coverage_input_files(root, collector)
            if path.is_file()
        ]
        counts.append(len(files))
        for source in files:
            try:
                relative = source.relative_to(root)
                resolved = source.resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                continue
            target = destination / f"case-{index:08d}" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(resolved)
            except OSError:
                # A same-filesystem hard link is cheap and keeps the staging
                # path usable on filesystems that reject symlinks.
                try:
                    os.link(resolved, target)
                except OSError:
                    shutil.copy2(resolved, target)
    return counts


def _prefix_unit_sets(
    profile: Path, spec: dict[str, Any],
) -> tuple[str, str | None, dict[str, set[tuple[Any, ...]]]]:
    """Parse a checkpoint LCOV once and return compact eligible/covered sets."""
    scopes = [str(item) for item in spec.get("source_scope", []) if str(item).strip()]
    excludes = [
        str(item) for item in spec.get("source_scope_exclude", []) if str(item).strip()
    ]
    source_root = str(
        co._dependency_path(spec.get("source_root"))
        or spec.get("source_root") or ""
    )
    parsed = simulator_coverage.summarize_lcov(
        profile, scopes, source_root, excludes, include_units=True, units_only=True,
    )
    units = {name: {"eligible": set(), "covered": set()} for name in METRICS}
    raw_units = parsed.get("unit_sets") if isinstance(parsed, dict) else None
    if isinstance(raw_units, dict):
        for name in METRICS:
            item = raw_units.get(name)
            if not isinstance(item, dict):
                continue
            for key in ("eligible", "covered"):
                values = item.get(key)
                if isinstance(values, list):
                    units[name][key].update(
                        unit for value in values
                        if (unit := _unit_tuple(value)) is not None
                    )
    status = str(parsed.get("status") or "gap") if isinstance(parsed, dict) else "gap"
    reason = parsed.get("reason") if isinstance(parsed, dict) else "source-growth-profile-missing"
    return status, reason, units


def _mark_branch_universe(rows: list[dict[str, Any]], universes: dict[int, set[tuple[Any, ...]]]) -> None:
    """Reject growth branch ratios when a collector changes branch IDs."""
    checkpoints_seen = [key for key in universes if key > 0]
    full = universes.get(max(checkpoints_seen)) if checkpoints_seen else None
    stable = full is not None and all(universes[key] == full for key in checkpoints_seen)
    reason = None if stable else "source-growth-branch-universe-unstable"
    for row in rows:
        row["branch_universe_status"] = "stable" if stable else "unstable"
        if reason and row.get("case_checkpoint", 0):
            row["SimSrcCov_branch_covered"] = None
            row["SimSrcCov_branch_eligible"] = None
            row["SimSrcCov_branch_value"] = None
            row["SimSrcCov_branch_status"] = "gap"
            row["SimSrcCov_branch_reason"] = reason


def _collect_checkpoint_prefix(
    method: str,
    target_id: str,
    target: dict[str, Any],
    roots: list[Path],
    stage_info: dict[str, Any],
    output: Path,
    binary: Path | None,
) -> list[dict[str, Any]]:
    """Collect LLVM/Dotnet growth at exponential prefixes only.

    This is the bounded conversion path for collectors that cannot expose a
    per-case unit stream.  It performs O(log N) collector invocations instead
    of starting a converter for every case while preserving exact prefix
    semantics.  The final whole-cell collector remains independent.
    """
    spec = dict(target.get("source_coverage") or {})
    collector = str(spec.get("collector", "gcov"))
    expected = len(roots)
    checkpoint_set = set(checkpoints(expected))
    temporary_parent = Path(stage_info.get("stage_dir") or output)
    status_path = output / "case-status.jsonl"
    rows: list[dict[str, Any]] = []
    universes: dict[int, set[tuple[Any, ...]]] = {}
    rows.append(_snapshot(
        method, target_id, 0, expected, 0, 0,
        {name: set() for name in METRICS},
        {name: set() for name in METRICS}, None, stage_info,
    ))
    rows[0].update({
        "one_pass": False,
        "growth_mode": "checkpoint-prefix",
        "collector": collector,
        "collector_invocations": 0,
    })
    collector_calls = 0
    identity_cache: dict[str, dict[str, Any]] = {}
    identity_cache_lock = threading.Lock()
    with status_path.open("w", encoding="utf-8") as status_stream:
        status_stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
        # Case status is only an input-presence audit in this mode; coverage
        # conversion happens at the sparse checkpoints below.
        for index, root in enumerate(roots, 1):
            count = sum(
                1 for path in source_coverage.coverage_input_files(root, collector)
                if path.is_file()
            )
            status_stream.write(json.dumps({
                "case_index": index,
                "status": "observed" if count else "gap",
                "input_files": count,
                "reason": None if count else "source-growth-input-missing",
            }, ensure_ascii=False) + "\n")

        for checkpoint in sorted(checkpoint_set):
            if checkpoint == 0:
                continue
            represented = 0
            bad_cases = 0
            collection_status = "gap"
            collection_reason: str | None = None
            units = {name: {"eligible": set(), "covered": set()} for name in METRICS}
            with tempfile.TemporaryDirectory(
                prefix=f"src-growth-prefix-{checkpoint:08d}-",
                dir=temporary_parent,
            ) as temporary:
                prefix_dir = Path(temporary) / "raw"
                counts = _stage_prefix_inputs(roots, collector, checkpoint, prefix_dir)
                represented = sum(count > 0 for count in counts)
                bad_cases = checkpoint - represented
                local_spec = dict(spec)
                local_spec["profile"] = str(Path(temporary) / "lcov.info")
                local_spec["identity"] = str(Path(temporary) / "identity.json")
                local_target = {**target, "source_coverage": local_spec}
                try:
                    collection = source_coverage.collect_source_coverage_batch(
                        Path(temporary), local_target, binary, prefix_dir,
                        expected_coverage_cases=checkpoint,
                        identity_cache=identity_cache,
                        identity_cache_lock=identity_cache_lock,
                        compute_input_fingerprint=False,
                    )
                except Exception as error:  # keep one bad prefix auditable
                    collection = {
                        "status": "gap",
                        "reason": f"checkpoint-collector:{type(error).__name__}",
                    }
                collector_calls += 1
                collection_status = str(collection.get("status") or "gap")
                collection_reason = collection.get("reason")
                profile = Path(temporary) / "lcov.info"
                if collection_status in {"observed", "partial"} and profile.is_file():
                    parsed_status, parsed_reason, units = _prefix_unit_sets(profile, local_spec)
                    if parsed_status == "gap":
                        collection_status = "gap"
                    elif parsed_status == "partial" and collection_status == "observed":
                        collection_status = "partial"
                    collection_reason = parsed_reason or collection_reason
            universes[checkpoint] = set(units["branch"]["eligible"])
            status_override = "gap" if collection_status == "gap" else (
                "partial" if collection_status == "partial" or bad_cases else None
            )
            row = _snapshot(
                method, target_id, checkpoint, expected, represented, bad_cases,
                {name: units[name]["eligible"] for name in METRICS},
                {name: units[name]["covered"] for name in METRICS},
                str(collection_reason) if collection_reason else (
                    "source-growth-input-missing" if bad_cases else None
                ),
                stage_info,
                status_override=status_override,
            )
            row.update({
                "one_pass": False,
                "growth_mode": "checkpoint-prefix",
                "collector": collector,
                "collector_invocations": collector_calls,
                "prefix_input_cases": checkpoint,
                "prefix_input_files": sum(counts),
                "prefix_represented_cases": represented,
            })
            rows.append(row)
            status_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _mark_branch_universe(rows, universes)
    for row in rows:
        row["collector_invocations"] = collector_calls
        row["ssd_stage"] = stage_info
    return rows


def _collect_gcov_one_pass(
    method: str,
    target_id: str,
    target: dict[str, Any],
    roots: list[Path],
    stage_info: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    """Convert all staged GCDA files in bounded GCC JSON batches.

    ``gcov`` accepts many independent ``.gcda`` inputs in one invocation.  Its
    JSON output names are unique for each input path, so the output stream can
    be mapped back to the case index.  This removes prefix profile merges and
    avoids retaining one large accumulator per case: only the earliest case
    that covers each source unit is kept.
    """
    spec = dict(target.get("source_coverage") or {})
    scopes = [str(item) for item in spec.get("source_scope", []) if str(item).strip()]
    excludes = [
        str(item) for item in spec.get("source_scope_exclude", []) if str(item).strip()
    ]
    source_root = co._dependency_path(spec.get("source_root")) or Path()
    expected = len(roots)
    checkpoint_set = set(checkpoints(expected))
    work = Path(stage_info.get("stage_dir") or output) / "gcov-one-pass"
    work.mkdir(parents=True, exist_ok=True)
    inputs, case_ok, note_info = _attach_gcov_notes(roots, spec, work)
    stage_info.update({f"one_pass_{key}": value for key, value in note_info.items()})
    case_for_input = {
        path: index
        for index, root in enumerate(roots)
        for path in root.rglob("*.gcda")
        if path.is_file() and not path.name.endswith(".tmp.gcda")
        and path.with_suffix(".gcno").is_file()
    }
    first_seen = {name: {} for name in METRICS}
    eligible = {name: set() for name in METRICS}
    eligible_first_seen = {name: {} for name in METRICS}
    case_reports = [0] * expected
    case_parse_errors = [False] * expected
    default_batch_size = 512 if "LRSV" in target_id else 4096
    try:
        batch_size = max(256, min(8192, int(os.environ.get("RQ1_GCOV_JSON_BATCH", str(default_batch_size)))))
    except (TypeError, ValueError, OverflowError):
        batch_size = 4096
    gcov = shutil.which(str(co._dependency_path(spec.get("gcov_tool", "gcov")) or "gcov")) or "gcov"
    def run_batch(batch_index: int, batch: list[Path]) -> tuple[int, list[Path], int, str, Path]:
        """Run one independent gcov batch in its own output directory."""
        batch_dir = work / f"json-{batch_index:08d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [gcov, "-j", "-l", "-b", "-c", "-p", "-x", *map(str, batch)],
            cwd=batch_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        return batch_index, batch, completed.returncode, completed.stdout or "", batch_dir

    def consume_batch(result: tuple[int, list[Path], int, str, Path]) -> None:
        """Reduce a completed batch in input order, preserving earliest cases."""
        _batch_index, batch, returncode, stdout, batch_dir = result
        if returncode:
            for path in batch:
                index = case_for_input.get(path)
                if index is not None:
                    case_parse_errors[index] = True
        created = [batch_dir / name for name in _GCOV_CREATING.findall(stdout)]
        reports_by_hash = {
            path.name.split("##", 1)[1].split(".", 1)[0]: path
            for path in created
            if "##" in path.name
        }
        if len(created) != len(batch):
            # Do not guess a case mapping when gcov omitted a report.  The
            # affected cases remain partial/gap, while complete batches keep
            # their exact evidence.
            for path in batch:
                index = case_for_input.get(path)
                if index is not None:
                    case_parse_errors[index] = True
            shutil.rmtree(batch_dir, ignore_errors=True)
            return
        for input_path in batch:
            index = case_for_input.get(input_path)
            digest = hashlib.md5(str(input_path).encode()).hexdigest()
            report = reports_by_hash.get(digest)
            if index is None or report is None or not report.is_file():
                if index is not None:
                    case_parse_errors[index] = True
                continue
            case_reports[index] += 1
            if not _note_gcov_json(
                report, index + 1, source_root, scopes, excludes,
                first_seen, eligible, eligible_first_seen,
            ):
                case_parse_errors[index] = True
        shutil.rmtree(batch_dir, ignore_errors=True)

    try:
        pair_jobs = max(1, min(4, int(os.environ.get("RQ1_GCOV_PAIR_JOBS", "1"))))
    except (TypeError, ValueError, OverflowError):
        pair_jobs = 1
    if pair_jobs == 1:
        for batch_index in range(0, len(inputs), batch_size):
            consume_batch(run_batch(batch_index, inputs[batch_index:batch_index + batch_size]))
    else:
        # Keep only ``pair_jobs`` batches in flight.  Results are consumed in
        # ascending case order so ``setdefault`` always records the earliest
        # case even though gcov itself runs concurrently.
        with ThreadPoolExecutor(max_workers=pair_jobs, thread_name_prefix="gcov-batch") as pool:
            pending: dict[int, Any] = {}
            next_start = 0
            next_consume = 0
            while next_start < len(inputs) or pending:
                while next_start < len(inputs) and len(pending) < pair_jobs:
                    batch = inputs[next_start:next_start + batch_size]
                    pending[next_start] = pool.submit(run_batch, next_start, batch)
                    next_start += batch_size
                future = pending.pop(next_consume)
                consume_batch(future.result())
                next_consume += batch_size

    status_path = output / "case-status.jsonl"
    rows = [_snapshot(
        method, target_id, 0, expected, 0, 0,
        {name: set() for name in METRICS},
        {name: set() for name in METRICS},
        None, stage_info,
    )]
    reasons: list[str] = []
    with status_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
        for index in range(expected):
            if not case_ok[index]:
                reasons.append(f"case-{index + 1:08d}:gcov-input-incomplete")
            if case_parse_errors[index]:
                reasons.append(f"case-{index + 1:08d}:gcov-json-parse-failed")
            stream.write(json.dumps({
                "case_index": index + 1,
                "status": "observed" if case_ok[index] and not case_parse_errors[index] else "partial",
                "reports": case_reports[index],
                "reason": reasons[-1] if reasons and reasons[-1].startswith(f"case-{index + 1:08d}:") else None,
            }, ensure_ascii=False) + "\n")
        for checkpoint in sorted(checkpoint_set):
            if checkpoint == 0:
                continue
            represented = sum(
                1 for index in range(checkpoint)
                if case_ok[index] and not case_parse_errors[index]
            )
            bad = checkpoint - represented
            checkpoint_eligible = {
                name: {
                    unit for unit, first in eligible_first_seen[name].items()
                    if first <= checkpoint
                }
                for name in METRICS
            }
            checkpoint_covered = {
                name: {
                    unit for unit, first in first_seen[name].items()
                    if first <= checkpoint and unit in checkpoint_eligible[name]
                }
                for name in METRICS
            }
            row = _snapshot(
                method, target_id, checkpoint, expected, represented, bad,
                checkpoint_eligible, checkpoint_covered,
                ";".join(sorted(set(reasons))) or None, stage_info,
            )
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _snapshot(
    method: str,
    target: str,
    checkpoint: int,
    expected: int,
    represented: int,
    bad_cases: int,
    eligible: dict[str, set[tuple[Any, ...]]],
    covered: dict[str, set[tuple[Any, ...]]],
    reason: str | None,
    stage_info: dict[str, Any],
    *,
    status_override: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "target": target,
        "case_checkpoint": checkpoint,
        "expected_cases": expected,
        "represented_cases": represented,
        "one_pass": True,
        "ssd_stage": stage_info,
    }
    if checkpoint == 0:
        row.update({
            "source_status": "waiting",
            "collection_status": "waiting",
            "collection_reason": None,
        })
        metrics = _empty_metrics("waiting")
    else:
        status = status_override if status_override in {"observed", "partial", "gap"} else (
            "gap" if represented == 0 else "partial" if bad_cases else "observed"
        )
        row.update({
            "source_status": status,
            "collection_status": status,
            "collection_reason": reason,
        })
        metrics = {}
        for name in METRICS:
            eligible_units = eligible[name]
            covered_units = covered[name] & eligible_units
            metric_status = status
            metric_reason = reason
            if not eligible_units:
                metric_status = "gap"
                metric_reason = metric_reason or "source-coverage-scope-empty"
            metrics[name] = {
                "covered": len(covered_units),
                "eligible": len(eligible_units) if eligible_units else None,
                "value": (
                    round(len(covered_units) / len(eligible_units), 6)
                    if eligible_units and metric_status in {"observed", "partial"}
                    else None
                ),
                "status": metric_status,
                **({"reason": metric_reason} if metric_reason else {}),
            }
    for name in METRICS:
        metric = metrics[name]
        row.update({
            f"SimSrcCov_{name}_covered": metric.get("covered"),
            f"SimSrcCov_{name}_eligible": metric.get("eligible"),
            f"SimSrcCov_{name}_value": metric.get("value"),
            f"SimSrcCov_{name}_status": metric.get("status"),
        })
    return row


def collect(manifest: Path, method: str, target_id: str, output: Path) -> list[dict[str, Any]]:
    methods, targets, experiments = co._load_launches([manifest.resolve()])
    specs = co._target_specs(experiments, targets)
    target = specs[(method, target_id)]
    runs = [
        run for experiment in experiments
        for run in experiment["runs"]
        if run["method"] == method
    ]
    collector = str((target.get("source_coverage") or {}).get("collector") or "gcov")
    if collector not in {"gcov", "lcov", "llvm", "dotnet"}:
        return [{
            "method": method,
            "target": target_id,
            "case_checkpoint": 0,
            "source_status": "gap",
            "collection_status": "gap",
            "collection_reason": f"growth-collector-unsupported:{collector}",
            "one_pass": True,
        }]
    inputs = [co._target_run_inputs(run, target_id, collector) for run in runs]
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "case-status.jsonl"
    binary = co._dependency_path(target.get("coverage_binary"))
    with co._stage_cell_inputs(inputs, collector) as (staged_inputs, stage_info):
        roots = ordered_roots(staged_inputs)
        if collector in {"gcov", "lcov"}:
            return _collect_gcov_one_pass(
                method, target_id, target, roots, stage_info, output,
            )
        growth_mode = os.environ.get("RQ1_GROWTH_MODE", "checkpoint-prefix").strip().lower()
        if collector in {"llvm", "dotnet"} and growth_mode == "checkpoint-prefix":
            return _collect_checkpoint_prefix(
                method, target_id, target, roots, stage_info, output, binary,
            )
        expected = len(roots)
        checkpoint_set = set(checkpoints(expected))
        temporary_parent = Path(stage_info.get("stage_dir") or output)
        rows: list[dict[str, Any]] = []
        eligible = {name: set() for name in METRICS}
        covered = {name: set() for name in METRICS}
        rows.append(_snapshot(
            method, target_id, 0, expected, 0, 0,
            eligible, covered, None, stage_info,
        ))
        represented = 0
        bad_cases = 0
        reasons: list[str] = []
        # Source, binary, and collector identity are immutable inputs for a
        # staged cell.  Reuse their evidence across cases; the final offline
        # gate still performs the complete input fingerprint and identity
        # audit on the raw cell.
        identity_cache: dict[str, dict[str, Any]] = {}
        identity_cache_lock = threading.Lock()
        try:
            case_jobs = max(1, min(6, int(os.environ.get("RQ1_GROWTH_CASE_JOBS", "1"))))
        except (TypeError, ValueError, OverflowError):
            case_jobs = 1
        # Each Renode case starts a dotnet coverage/reporting process.  The
        # worker releases completed Future objects below, so the bounded heap
        # setting in ``main`` permits the configured six-way limit without
        # retaining every case's unit set.  GCOV batches remain independently
        # parallel.
        if collector == "dotnet":
            case_jobs = max(1, min(6, case_jobs))

        def report_progress(processed: int) -> None:
            # Exponential snapshots are intentionally sparse.  Emit a cheap
            # operational heartbeat so a long Renode cell is not mistaken for
            # a stalled worker; this is progress logging, not coverage sampling.
            if processed == 1 or processed % 32 == 0 or processed == expected:
                print(
                    f"source-growth progress method={method} target={target_id} "
                    f"cases={processed}/{expected} represented={represented} gaps={bad_cases}",
                    flush=True,
                )
        with status_path.open("w", encoding="utf-8") as status_stream:
            status_stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
            if case_jobs == 1:
                case_results = (
                    _read_case(
                        output, temporary_parent, target, binary, root, index,
                        identity_cache, identity_cache_lock,
                    )
                    for index, root in enumerate(roots, 1)
                )
                for processed, result in enumerate(case_results, 1):
                    represented, bad_cases, reasons = _consume_case(
                        result, represented, bad_cases, reasons,
                        eligible, covered, status_stream,
                    )
                    report_progress(processed)
                    checkpoint = processed
                    if checkpoint in checkpoint_set:
                        rows.append(_snapshot(
                            method, target_id, checkpoint, expected, represented,
                            bad_cases, eligible, covered,
                            ";".join(sorted(set(reasons))) or None, stage_info,
                        ))
                        status_stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            else:
                with ThreadPoolExecutor(max_workers=case_jobs, thread_name_prefix="src-growth-case") as pool:
                    futures = [
                        pool.submit(
                            _read_case, output, temporary_parent,
                            target, binary, root, index,
                            identity_cache, identity_cache_lock,
                        )
                        for index, root in enumerate(roots, 1)
                    ]
                    for processed, future in enumerate(futures, 1):
                        try:
                            result = future.result()
                        finally:
                            # A result contains the per-case eligible/covered
                            # sets.  The cumulative union is kept in
                            # ``eligible``/``covered``; retaining completed
                            # Future objects would duplicate all of those sets
                            # until the cell finishes.
                            futures[processed - 1] = None
                        represented, bad_cases, reasons = _consume_case(
                            result, represented, bad_cases, reasons,
                            eligible, covered, status_stream,
                        )
                        report_progress(processed)
                        checkpoint = processed
                        if checkpoint in checkpoint_set:
                            rows.append(_snapshot(
                                method, target_id, checkpoint, expected, represented,
                                bad_cases, eligible, covered,
                                ";".join(sorted(set(reasons))) or None, stage_info,
                            ))
                            status_stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                        del result
        # The maximum checkpoint is always emitted, including an empty input cell.
        if expected and rows[-1].get("case_checkpoint") != expected:
            rows.append(_snapshot(
                method, target_id, expected, expected, represented, bad_cases,
                eligible, covered, ";".join(sorted(set(reasons))) or None, stage_info,
            ))
        for row in rows:
            row["ssd_stage"] = stage_info
        return rows


def _consume_case(
    result: dict[str, Any],
    represented: int,
    bad_cases: int,
    reasons: list[str],
    eligible: dict[str, set[tuple[Any, ...]]],
    covered: dict[str, set[tuple[Any, ...]]],
    status_stream,
) -> tuple[int, int, list[str]]:
    status = str(result.get("status") or "gap")
    if status in {"observed", "partial"}:
        represented += 1
        for name in METRICS:
            units = result.get("units", {}).get(name, {})
            eligible[name].update(units.get("eligible") or ())
            covered[name].update(units.get("covered") or ())
    if status not in {"observed"}:
        bad_cases += 1
    reason = result.get("reason")
    if reason:
        reasons.append(str(reason))
    status_stream.write(json.dumps({
        "case_index": result.get("index"),
        "status": status,
        "reason": reason,
    }, ensure_ascii=False) + "\n")
    status_stream.flush()
    return represented, bad_cases, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-gib", type=float, default=9.5)
    parser.add_argument(
        "--growth-mode", choices=("checkpoint-prefix", "case-single-pass"),
        default=None,
        help="增长实现；LLVM/Renode 默认使用指数前缀归并。",
    )
    args = parser.parse_args()
    if args.growth_mode:
        os.environ["RQ1_GROWTH_MODE"] = args.growth_mode
    # CoreCLR refuses to initialize when RLIMIT_AS is finite, even when the
    # requested address space is tens of GiB.  Renode's ReportGenerator is a
    # bounded single-case process, so leave virtual-address accounting
    # unlimited and cap the managed heap instead.  Other collectors retain
    # the per-worker address-space guard.
    if "RENODE" in args.target.upper():
        os.environ.setdefault("DOTNET_GCHeapHardLimit", str(4 * 1024 ** 3))
    else:
        resource.setrlimit(resource.RLIMIT_AS, (int(args.memory_limit_gib * (1024 ** 3)),) * 2)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.manifest, args.method, args.target, args.output)
    path = args.output / f"{co._safe(args.method)}-{co._safe(args.target)}.json"
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    metadata = args.output / "cell-metadata.json"
    metadata_tmp = metadata.with_name(metadata.name + ".tmp")
    metadata_tmp.write_text(
        json.dumps({
            "schema_version": "rq1-growth-cell-metadata-v1",
            "method": args.method,
            "target": args.target,
            "row_count": len(rows),
            "growth_mode": os.environ.get("RQ1_GROWTH_MODE", "checkpoint-prefix"),
            "ssd_stage": rows[0].get("ssd_stage") if rows else None,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_tmp, metadata)
    marker = args.output / "cell-complete.json"
    marker_payload = {
        "schema_version": "rq1-growth-cell-complete-v1",
        "method": args.method,
        "target": args.target,
        "result": path.name,
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "growth_mode": os.environ.get("RQ1_GROWTH_MODE", "checkpoint-prefix"),
        "row_count": len(rows),
    }
    marker_tmp = marker.with_name(marker.name + ".tmp")
    marker_tmp.write_text(
        json.dumps(marker_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

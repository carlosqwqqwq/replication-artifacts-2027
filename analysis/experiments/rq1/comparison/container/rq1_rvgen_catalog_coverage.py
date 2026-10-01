#!/usr/bin/env python3
"""Run a random, time-boxed RVGEN catalog campaign inside the RQ1 container."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import random
import re
import resource
import secrets
import signal
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUN_ROOT = Path(os.environ["RQ1_RUN_ROOT"]).resolve()
CODE_ROOT = Path(os.environ.get("RQ1_COMPARISON_ROOT", "/opt/rq1/comparison")).resolve()
CONFIG_PATH = Path(os.environ.get(
    "RQ1_CONFIG_PATH", str(CODE_ROOT / "config" / "rq1-comparison-isolated-v1.json")
)).resolve()
os.environ.setdefault("RQ1_DEPS", "/path/to/deps")
os.environ.setdefault("RQ1_ENABLE_COVERAGE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(CODE_ROOT))

from comparison_events import _join_reference, _make_event  # noqa: E402
from decoupled_pipeline import _k1_fallback_class, _qemu_fallback_reference  # noqa: E402
from framework.direct_case import TestCase  # noqa: E402
from framework.direct_elf import _toolchain_profile, build_elf  # noqa: E402
from framework.execution_environment import require_execution_plane  # noqa: E402
from framework.direct_runner import MAX_NATIVE_REFERENCE_PARALLELISM  # noqa: E402
from framework.rvgen.core import _generate_corpus, generate_form  # noqa: E402
from framework.riscv_catalog import OFFICIAL_ALL_CATALOG_FORMS  # noqa: E402
from framework.reference_fallback import K1_BOARD_TQEMU_FALLBACK_POLICY  # noqa: E402
from runner import (  # noqa: E402
    BOARD_LINUX_USER_ONLY,
    _board_reference_gap,
    _board_reference_record,
    _configured_target_timeout,
    _identity,
    _materialize_dotnet_runtime,
    _provenance,
    dep,
    load_json,
    run_targets,
    validate_config,
)

METHOD = "Ours-RVGEN-Direct"
ROUTE = "single"
LANE = "rvgen-catalog-mixed-isa"
PREVIOUSLY_REPORTED_CATALOG_ROWS = 114_377
DEFAULT_DURATION_SECONDS = 72 * 60 * 60

START_MONO = 0.0
PROCESS_STARTED_MONO = time.monotonic()
START_UTC = ""
TARGET_DEADLINE = 0.0
SEED = 0
CONFIG: dict = {}
TARGETS_BY_ID: dict[str, dict] = {}
PROVENANCE: dict = {}
_STATE_LOCK = threading.RLock()
_APPEND_LOCK = threading.Lock()
_CASE_LOCKS_LOCK = threading.Lock()
_CASE_LOCKS: dict[tuple[str, int], threading.Lock] = {}


def _record_stop_request(
    stop_requested: threading.Event, stop_state: dict, signum: int,
) -> None:
    if not stop_requested.is_set():
        stop_state["signal_name"] = signal.Signals(signum).name
    stop_requested.set()


def _campaign_outcome(signal_name: str | None, all_targets_exhausted: bool) -> dict:
    if signal_name:
        return {
            "status": "partial",
            "reason_code": "external-signal",
            "stop_signal": signal_name,
        }
    return {
        "status": "catalog-exhausted" if all_targets_exhausted else "incomplete",
        "reason_code": None if all_targets_exhausted else "runner-stopped-before-catalog-exhaustion",
        "stop_signal": None,
    }


def _case_lock(scope: str, sequence: int) -> threading.Lock:
    key = (scope, sequence)
    with _CASE_LOCKS_LOCK:
        return _CASE_LOCKS.setdefault(key, threading.Lock())


def _bump_counter(counter: Counter, key: str, amount: int | bool = 1) -> None:
    with _STATE_LOCK:
        counter[key] += int(amount)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, values: list[dict]) -> None:
    if not values:
        return
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for value in values
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def case_record(item: dict, sequence: int, *, include_testcase: bool = True) -> dict:
    testcase = item["testcase"]
    return {
        "sequence": sequence,
        "candidate_id": item["candidate_id"],
        "form": item["form"],
        "form_index": item["form_index"],
        "form_case_index": item["form_case_index"],
        "testcase_id": testcase.testcase_id,
        "base_program_id": testcase.base_program_id,
        "input_id": testcase.input_id,
        "isa_profile": testcase.isa_profile,
        "generation_rule_id": testcase.generation_rule_id,
        "instruction_count": len(testcase.instruction_meta),
        "code_hex": testcase.code_bytes.hex(),
        **({"testcase": testcase.to_dict()} if include_testcase else {}),
    }


def _record_from_payload(item: dict, sequence: int) -> dict:
    testcase = item["testcase"]
    return {
        "sequence": sequence,
        "candidate_id": item["candidate_id"],
        "form": item["form"],
        "form_index": item["form_index"],
        "form_case_index": item["form_case_index"],
        "testcase_id": testcase["testcase_id"],
        "base_program_id": testcase["base_program_id"],
        "input_id": testcase["input_id"],
        "isa_profile": testcase["isa_profile"],
        "generation_rule_id": testcase["generation_rule_id"],
        "instruction_count": len(testcase["instruction_meta"]),
        "code_hex": testcase["code_hex"],
        "testcase": testcase,
    }


def catalog_random_queue(selection_path: Path, seed: int) -> dict:
    """Write every generated catalog case once in seeded random order."""
    rng = random.Random(seed)
    database_path = RUN_ROOT / "catalog-random.sqlite"
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE cases (random_key BLOB NOT NULL, source_sequence INTEGER NOT NULL, "
        "payload TEXT NOT NULL, PRIMARY KEY(random_key, source_sequence)) WITHOUT ROWID"
    )

    started = time.monotonic()
    form_counts: dict[str, int] = {}
    errors: list[dict] = []
    total = 0
    forms = OFFICIAL_ALL_CATALOG_FORMS
    for form_index, form in enumerate(forms):
        name = str(form.mnemonic)
        try:
            cases = generate_form(form)
        except Exception as exc:
            errors.append({"form": name, "error": f"{type(exc).__name__}: {exc}"[:500]})
            _generate_corpus.cache_clear()
            continue
        form_counts[name] = len(cases)
        batch = []
        for case_index, testcase in enumerate(cases):
            payload = {
                "form": name,
                "form_index": form_index,
                "form_case_index": case_index,
                "candidate_id": f"{name}:{testcase.testcase_id}:{case_index}",
                "testcase": testcase.to_dict(),
            }
            batch.append((
                rng.getrandbits(128).to_bytes(16, "big"),
                total,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ))
            total += 1
            if len(batch) >= 1_024:
                connection.executemany("INSERT INTO cases VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO cases VALUES (?, ?, ?)", batch)
        connection.commit()
        _generate_corpus.cache_clear()
        if (form_index + 1) % 100 == 0:
            print(json.dumps({
                "event": "catalog-generation-progress",
                "forms": form_index + 1,
                "total_forms": len(forms),
                "catalog_rows": total,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "peak_rss_mb": peak_rss_mb(),
            }), flush=True)

    if total == 0:
        connection.close()
        raise RuntimeError("catalog generation produced no cases")

    queued_forms: Counter[str] = Counter()
    queued_profiles: Counter[str] = Counter()
    with selection_path.open("w", encoding="utf-8") as stream:
        query = connection.execute(
            "SELECT payload FROM cases ORDER BY random_key, source_sequence"
        )
        for sequence, (payload,) in enumerate(query):
            item = json.loads(payload)
            row = _record_from_payload(item, sequence)
            stream.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ) + "\n")
            queued_forms[str(item["form"])] += 1
            queued_profiles[str(item["testcase"]["isa_profile"])] += 1
    connection.close()

    database_bytes = database_path.stat().st_size if database_path.exists() else 0
    database_path.unlink(missing_ok=True)
    if total != sum(queued_forms.values()):
        raise RuntimeError(
            f"random queue row mismatch: generated={total}, queued={sum(queued_forms.values())}"
        )
    for suffix in ("-wal", "-shm"):
        database_path.with_name(database_path.name + suffix).unlink(missing_ok=True)
    return {
        "catalog_form_count": len(forms),
        "generated_catalog_rows": total,
        "previously_reported_catalog_rows": PREVIOUSLY_REPORTED_CATALOG_ROWS,
        "queued_rows": sum(queued_forms.values()),
        "selection": "full-catalog-uniform-random-key-permutation-without-replacement",
        "seed": seed,
        "generation_errors": errors,
        "generation_error_count": len(errors),
        "generation_seconds": round(time.monotonic() - started, 3),
        "peak_rss_mb_during_generation": peak_rss_mb(),
        "random_queue_database_bytes_before_cleanup": database_bytes,
        "queued_isa_profile_count": len(queued_profiles),
        "queued_form_count": len(queued_forms),
        "queued_isa_profile_counts": dict(sorted(queued_profiles.items())),
        "queued_form_counts": dict(sorted(queued_forms.items())),
        "generated_rows_by_form": form_counts,
        "queue_sha256": sha256_file(selection_path),
    }


def build_target_queues(
    selection_path: Path, queue_root: Path, seed: int,
    target_ids: list[str], expected_rows: int,
) -> tuple[list[tuple[int, int]], dict[str, dict]]:
    """Index the shared case file and persist one seeded permutation per target."""
    offsets: list[tuple[int, int]] = []
    with selection_path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            offsets.append((offset, len(line)))
    if len(offsets) != expected_rows:
        raise RuntimeError(
            f"target queue source row mismatch: expected={expected_rows}, got={len(offsets)}"
        )

    queue_root.mkdir(parents=True, exist_ok=True)
    queues = {}
    for target_id in target_ids:
        seed_material = f"rvgen-target-queue-v1\0{seed}\0{target_id}".encode()
        queue_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        order = list(range(expected_rows))
        random.Random(queue_seed).shuffle(order)
        queue_path = queue_root / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', target_id)}.jsonl"
        with queue_path.open("w", encoding="ascii") as stream:
            stream.writelines(f"{sequence}\n" for sequence in order)
        queues[target_id] = {
            "path": str(queue_path.relative_to(RUN_ROOT)),
            "seed": queue_seed,
            "rows": len(order),
            "sha256": sha256_file(queue_path),
        }
    return offsets, queues


def _read_queued_case(
    selection_fd: int, offsets: list[tuple[int, int]], sequence: int,
) -> dict:
    offset, length = offsets[sequence]
    payload = os.pread(selection_fd, length, offset)
    if len(payload) != length:
        raise RuntimeError(f"random catalog row {sequence} could not be read completely")
    item = json.loads(payload)
    if item.get("sequence") != sequence:
        raise RuntimeError(
            f"random catalog sequence mismatch: expected={sequence}, got={item.get('sequence')}"
        )
    return item


def _target_stratum(target: dict) -> str:
    return (
        "linux-user"
        if str(target.get("execution_model", "")).startswith("linux-user")
        else "bare-metal"
    )


def _case_assets_locked(
    sequence: int, item: dict, required_strata: set[str],
) -> tuple[Path, TestCase, dict, dict[str, Path], dict[str, str]]:
    testcase = TestCase.from_dict(item["testcase"])
    form = re.sub(r"[^A-Za-z0-9._-]+", "_", str(item["form"]))
    case_dir = RUN_ROOT / "cases" / f"{sequence:06d}-{form}-{testcase.testcase_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    identity = case_record({**item, "testcase": testcase}, sequence, include_testcase=False)
    profile = str(testcase.isa_profile)
    _march, mabi, dropped = _toolchain_profile(profile)
    builds_path = case_dir / "builds.json"
    existing = read_json(builds_path)
    builds = existing.get("builds") if isinstance(existing.get("builds"), dict) else {}
    artifacts: dict[str, Path] = {}
    changed = not builds_path.is_file()
    for name, bare in (("linux-user", False), ("bare-metal", True)):
        if name in builds or name not in required_strata:
            continue
        output = case_dir / "artifacts" / f"{name}.elf"
        try:
            built = build_elf(
                testcase, output, bare_metal=bare,
            )
            builds[name] = built.to_dict()
        except Exception as exc:
            builds[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1_000]}
        changed = True
    for name in ("linux-user", "bare-metal"):
        output = case_dir / "artifacts" / f"{name}.elf"
        if builds.get(name, {}).get("ok") is True and output.is_file():
            artifacts[name] = output
    if changed:
        dump_json(builds_path, {
            "isa_profile": profile,
            "mabi": mabi,
            "toolchain_dropped_extensions": list(dropped),
            "builds": builds,
        })
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    return case_dir, testcase, identity, artifacts, hashes


def _case_assets(
    sequence: int, item: dict, required_strata: set[str],
) -> tuple[Path, TestCase, dict, dict[str, Path], dict[str, str]]:
    with _case_lock("assets", sequence):
        return _case_assets_locked(sequence, item, required_strata)


def _run_one_target(
    target_id: str, sequence: int, item: dict, case_dir: Path,
    testcase: TestCase, identity: dict, artifacts: dict[str, Path],
    artifact_hashes: dict[str, str], references: dict[str, dict], *,
    skip_reference: bool = False,
) -> tuple[dict, dict]:
    target = TARGETS_BY_ID[target_id]
    worker_root = case_dir / "target-workers" / target_id
    profile = str(testcase.isa_profile)
    _march, mabi, _dropped = _toolchain_profile(profile)
    provenance = {
        **PROVENANCE,
        "method": METHOD,
        "route": ROUTE,
        "lane": LANE,
        "case_id": item["candidate_id"],
        "testcase_id": testcase.testcase_id,
        "random_seed": SEED,
        "random_submission_sequence": sequence,
        "catalog_form": item["form"],
    }
    try:
        result = run_targets(
            CONFIG,
            worker_root,
            provenance=provenance,
            artifacts=artifacts,
            bare_isa=f"{profile}/{mabi}",
            target_ids={target_id},
            run_deadline=None,
            coverage_root=RUN_ROOT / "coverage-raw",
            feature_cache_root=RUN_ROOT / "feature-cache",
            collect_source=False,
            skip_reference=skip_reference,
            reference_override=references,
            include_guest_metrics=False,
            coverage_side_on_timeout=True,
        )
    except Exception as exc:
        result = {
            "schema_version": "rq1-comparison-target-run-v1",
            "status": "gap",
            "reason": f"run-targets-error:{type(exc).__name__}",
            "error": str(exc)[:1_000],
            "targets": [],
        }
    row = next((value for value in result.get("targets", [])
                if isinstance(value, dict) and value.get("id") == target_id), None)
    if row is None:
        row = {
            "id": target_id,
            "kind": target.get("kind"),
            "status": "gap",
            "reason": result.get("reason") or "target-worker-returned-no-result",
            "target_attempted": False,
            "target_dispatch": {"event_id": f"{target_id}:dispatch", "status": "completed"},
        }
    row.update({
        "method": METHOD,
        "route": ROUTE,
        "lane": LANE,
        "case_id": item["candidate_id"],
        "random_submission_sequence": sequence,
        "catalog_form": item["form"],
        "isa_profile": profile,
    })
    for key in (
        "simulator_coverage", "simulator_coverage_deferred", "coverage_status",
        "simulator_source_coverage", "source_coverage_collection",
        "coverage_runtime_enabled", "coverage_gap", "execution_binary_kind",
    ):
        row.pop(key, None)
    for key in ("coverage", "coverage_timing", "coverage_status"):
        result.pop(key, None)
    result["targets"] = [row]
    result["input"] = {key: identity[key] for key in (
        "sequence", "candidate_id", "form", "testcase_id", "isa_profile",
        "generation_rule_id", "instruction_count", "code_hex",
    )}
    result["artifact_sha256_by_stratum"] = artifact_hashes
    return result, row


def _event_entry(sequence: int, item: dict, profile: str, artifact_hashes: dict[str, str]) -> dict:
    return {
        "method": METHOD,
        "route": ROUTE,
        "lane": LANE,
        "campaign_id": RUN_ROOT.name,
        "seed": SEED,
        "candidate": {
            "index": sequence,
            "artifact_status": "ready" if artifact_hashes else "gap",
            "linux_elf_sha256": artifact_hashes.get("linux-user"),
            "elf_sha256": artifact_hashes.get("bare-metal"),
            "linux_artifact_isa": profile,
            "artifact_isa": profile,
            "generation_rule_id": item.get("generation_rule_id"),
        },
    }


def _persist_target_impl(
    target_id: str, sequence: int, item: dict, case_dir: Path,
    testcase: TestCase, identity: dict, artifact_hashes: dict[str, str],
    target_run: dict, row: dict, reference: dict,
    target_progress: dict, completed_bits: dict[str, bytearray],
) -> None:
    stratum = _target_stratum(TARGETS_BY_ID[target_id])
    event = _make_event(
        method={"id": METHOD, "route": ROUTE, "lane": LANE},
        target_id=target_id,
        entry=_event_entry(sequence, item, str(testcase.isa_profile), artifact_hashes),
        result=target_run,
        config=CONFIG,
        registry_sha="",
        registry_sealed=False,
        target_attempted=row.get("target_attempted") is True,
    )
    target_reference = _reference_map(reference).get(stratum, {}) if reference else {}
    _join_reference(event, {stratum: target_reference} if target_reference else {})
    row.update({
        "reference_id": event.get("reference_id"),
        "reference_status": event.get("reference_status"),
        "reference_identity": event.get("reference_identity"),
        "reference_join_status": event.get("reference_join_status"),
        "reference_fallback": event.get("reference_fallback"),
        "differential": event.get("differential"),
    })
    target_run["targets"] = [row]
    target_run["references"] = _reference_map(reference) if reference else {}
    worker_root = case_dir / "target-workers" / target_id
    dump_json(worker_root / "targets-run.json", target_run)

    case_path = case_dir / "targets-run.json"
    case_result = read_json(case_path)
    prior_rows = case_result.get("targets") if isinstance(case_result.get("targets"), list) else []
    rows_by_id = {
        str(value.get("id")): value for value in prior_rows
        if isinstance(value, dict) and value.get("id")
    }
    rows_by_id[target_id] = row
    case_result.update({
        "schema_version": "rq1-rvgen-catalog-case-v1",
        "input": {key: identity[key] for key in (
            "sequence", "candidate_id", "form", "testcase_id", "isa_profile",
            "generation_rule_id", "instruction_count", "code_hex",
        )},
        "references": _reference_map(reference) if reference else {},
        "targets": [rows_by_id[key] for key in TARGETS_BY_ID if key in rows_by_id],
        "builds": read_json(case_dir / "builds.json").get("builds", {}),
    })
    dump_json(case_path, case_result)
    append_jsonl(RUN_ROOT / "ledger" / "events.partial.jsonl", [event])
    append_jsonl(
        RUN_ROOT / "target-completions" / f"{target_id}.jsonl",
        [{
            "sequence": sequence,
            "candidate_id": item["candidate_id"],
            "form": item["form"],
            "case_dir": str(case_dir.relative_to(RUN_ROOT)),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "target_attempted": row.get("target_attempted") is True,
        }],
    )
    with _STATE_LOCK:
        target_progress[target_id]["attempted_cases"] += row.get("target_attempted") is True
        if completed_bits[target_id][sequence] == 0:
            completed_bits[target_id][sequence] = 1
            target_progress[target_id]["completed_cases"] += 1
        target_progress[target_id]["last_completed_sequence"] = sequence
        target_progress[target_id]["last_status"] = row.get("status")
        target_progress[target_id]["execution_seconds"] += float(row.get("elapsed_s") or 0)


def _persist_target(
    target_id: str, sequence: int, item: dict, case_dir: Path,
    testcase: TestCase, identity: dict, artifact_hashes: dict[str, str],
    target_run: dict, row: dict, reference: dict,
    target_progress: dict, completed_bits: dict[str, bytearray],
) -> None:
    with _case_lock("case-results", sequence):
        _persist_target_impl(
            target_id, sequence, item, case_dir, testcase, identity,
            artifact_hashes, target_run, row, reference,
            target_progress, completed_bits,
        )


def _reference_map(reference: dict) -> dict[str, dict]:
    return {
        "linux-user": reference,
        "bare-metal": _board_reference_gap("bare-metal", BOARD_LINUX_USER_ONLY),
    }


def _unavailable_fallback(k1_record: dict, qemu_event: dict, fallback_reason: str,
                          reason: str) -> dict:
    return {
        "schema_version": "rq1-reference-fallback-v1",
        "policy": K1_BOARD_TQEMU_FALLBACK_POLICY,
        "status": "unavailable",
        "preferred_backend": "native-rv64",
        "fallback_backend": "qemu-riscv64",
        "fallback_reason": fallback_reason,
        "selection_scope": "candidate",
        "selection_point": "same-candidate-t-qemu-result",
        "source_target": "T-QEMU",
        "source_event_id": qemu_event.get("event_id"),
        "reason": reason,
        "k1_attempt": {
            "id": k1_record.get("id"),
            "status": k1_record.get("status"),
            "reason": k1_record.get("reason"),
            "failure_class": fallback_reason,
            "returncode": k1_record.get("returncode"),
            "artifact_sha256": k1_record.get("artifact_sha256"),
        },
    }


def _ensure_reference(
    sequence: int, item: dict, testcase: TestCase, identity: dict,
    case_dir: Path, artifacts: dict[str, Path], artifact_hashes: dict[str, str],
    target_progress: dict, completed_bits: dict[str, bytearray],
    reference_counts: Counter, progress_callback,
) -> dict:
    reference_path = case_dir / "references.json"
    saved = read_json(reference_path)
    if saved.get("linux-user"):
        return saved["linux-user"]

    user_elf = artifacts.get("linux-user")
    profile = str(testcase.isa_profile).lower()
    if not user_elf or not user_elf.is_file():
        k1_record = _board_reference_gap("linux-user", "reference-artifact-missing")
        _bump_counter(reference_counts, "k1_not_applicable")
    elif not profile.startswith("rv64"):
        k1_record = _board_reference_gap("linux-user", "k1-reference-isa-width-unsupported")
        k1_record["artifact_sha256"] = artifact_hashes.get("linux-user")
        _bump_counter(reference_counts, "k1_not_applicable")
    else:
        _bump_counter(reference_counts, "k1_attempted")
        progress_callback("k1-reference", "T-K1-BOARD", sequence, item)
        timeout = _configured_target_timeout(CONFIG, None, TARGETS_BY_ID["T-QEMU"])
        try:
            k1_record = _board_reference_record(
                case_dir, "linux-user", user_elf, timeout,
                run_deadline=None,
            )
        except Exception as exc:
            k1_record = _board_reference_gap(
                "linux-user", f"k1-board-reference-error:{type(exc).__name__}",
            )
        k1_record.setdefault("artifact_sha256", artifact_hashes.get("linux-user"))

    fallback_reason = _k1_fallback_class(k1_record)
    if fallback_reason is None:
        _bump_counter(reference_counts, "k1_trusted_or_terminal", (
            k1_record.get("status") in {"passed", "failed"}
            and k1_record.get("terminal_observed") is True
        ))
        dump_json(reference_path, _reference_map(k1_record))
        return k1_record

    _bump_counter(reference_counts, "qemu_fallback_attempted")
    progress_callback("qemu-fallback", "T-QEMU", sequence, item)
    with _case_lock("target-T-QEMU", sequence):
        with _STATE_LOCK:
            qemu_already_completed = bool(completed_bits["T-QEMU"][sequence])
        qemu_run = read_json(
            case_dir / "target-workers" / "T-QEMU" / "targets-run.json"
        ) if qemu_already_completed else {}
        qemu_row = next((value for value in qemu_run.get("targets", [])
                         if isinstance(value, dict) and value.get("id") == "T-QEMU"), None)
        if qemu_row is None and qemu_already_completed:
            qemu_row = {
                "id": "T-QEMU", "status": "gap",
                "reason": "qemu-case-completed-result-missing",
                "target_attempted": False,
            }
            qemu_run = {"schema_version": "rq1-comparison-target-run-v1", "targets": [qemu_row]}
        elif qemu_row is None:
            qemu_run, qemu_row = _run_one_target(
                "T-QEMU", sequence, item, case_dir, testcase, identity,
                artifacts, artifact_hashes, {}, skip_reference=True,
            )

        qemu_event = _make_event(
            method={"id": METHOD, "route": ROUTE, "lane": LANE},
            target_id="T-QEMU",
            entry=_event_entry(sequence, item, str(testcase.isa_profile), artifact_hashes),
            result=qemu_run,
            config=CONFIG,
            registry_sha="",
            registry_sealed=False,
            target_attempted=qemu_row.get("target_attempted") is True,
        )
        fallback_reference, unavailable_reason = _qemu_fallback_reference(
            k1_record, qemu_event, fallback_reason,
        )
        if fallback_reference is None:
            _bump_counter(reference_counts, "qemu_fallback_unavailable")
            k1_record["reference_fallback"] = _unavailable_fallback(
                k1_record, qemu_event, fallback_reason,
                unavailable_reason or "qemu-target-reference-unavailable",
            )
            selected_reference = k1_record
        else:
            _bump_counter(reference_counts, "qemu_fallback_used")
            selected_reference = fallback_reference
        references = _reference_map(selected_reference)
        dump_json(reference_path, references)
        if not qemu_already_completed:
            _persist_target(
                "T-QEMU", sequence, item, case_dir, testcase, identity,
                artifact_hashes, qemu_run, qemu_row, selected_reference,
                target_progress, completed_bits,
            )
    return references["linux-user"]


def join_partial_event_references(target_ids: list[str]) -> dict[str, int]:
    events_path = RUN_ROOT / "ledger" / "events.partial.jsonl"
    case_dirs = {}
    for target_id in target_ids:
        completion_path = RUN_ROOT / "target-completions" / f"{target_id}.jsonl"
        try:
            with completion_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        case_dir = record.get("case_dir")
                        sequence = record.get("sequence")
                        if isinstance(case_dir, str) and type(sequence) is int:
                            case_dirs[(target_id, sequence)] = RUN_ROOT / case_dir
                    except (TypeError, ValueError):
                        continue
        except OSError:
            continue

    join_counts: Counter[str] = Counter()
    if not events_path.is_file():
        return {}
    temporary = events_path.with_name(
        events_path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with events_path.open("r", encoding="utf-8", errors="replace") as source, \
            temporary.open("w", encoding="utf-8") as output:
        for line in source:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            sequence = event.get("stream_index")
            case_dir = case_dirs.get((str(event.get("target") or ""), sequence))
            references = read_json(case_dir / "references.json") if case_dir else {}
            _join_reference(event, references)
            join_counts[str(event.get("reference_join_status") or "unknown")] += 1
            output.write(json.dumps(
                event, ensure_ascii=False, sort_keys=True, default=str,
                separators=(",", ":"),
            ) + "\n")
    os.replace(temporary, events_path)
    return dict(sorted(join_counts.items()))


def write_raw_artifact_index() -> None:
    def ref(relative: str) -> dict:
        path = RUN_ROOT / relative
        return {
            "path": relative,
            "exists": path.exists(),
            "kind": "directory" if path.is_dir() else "file",
        }

    case_root = ref("cases")
    target_runs = []
    profile_roots = []
    for target_id in sorted(TARGETS_BY_ID):
        target_runs.append({
            "target": target_id,
            "root": case_root,
            "target_run_pattern": f"*/target-workers/{target_id}/targets-run.json",
        })
        profile_roots.append({
            "target": target_id,
            **ref(f"coverage-raw/{target_id}/attempts"),
        })

    payload = {
        "schema_version": "rq1-rvgen-raw-artifact-index-v1",
        "created_at_utc": utc_now(),
        "run_id": RUN_ROOT.name,
        "coverage_mode": "raw-profiles-and-execution-traces",
        "run_root": ref("."),
        "execution_manifest": ref("execution-manifest.json"),
        "experiment_manifest": ref("experiment-manifest.json"),
        "runtime_progress": ref("progress.json"),
        "control": {
            "request": ref("control/request.json"),
            "status": ref("control/status.json"),
        },
        "case_catalog": {
            "selection": ref("catalog-selection.json"),
            "selected_cases": ref("selected-cases.jsonl"),
            "target_queues_manifest": ref("target-queues.json"),
            "target_queues_root": ref("target-queues"),
        },
        "execution_artifacts": {
            "cases_root": case_root,
            "target_runs": target_runs,
            "reference_records_pattern": "*/references.json",
        },
        "trace_artifacts": {
            "root": case_root,
            "path_pattern": "*/target-workers/*/traces",
            "paths_recorded_in": "*/target-workers/*/targets-run.json: targets[].trace_path",
        },
        "coverage_raw_profiles": {
            "root": ref("coverage-raw"),
            "target_attempt_roots": profile_roots,
        },
        "ledger": {
            "root": ref("ledger"),
            "complete_events": ref("ledger/events.jsonl"),
            "partial_events": ref("ledger/events.partial.jsonl"),
            "target_completions_root": ref("target-completions"),
            "reference_join_summary": ref("reference-join-summary.json"),
        },
        "interruption_records": {
            "unsubmitted_inputs": ref("input-window-unsubmitted.jsonl"),
            "unprocessed_references": ref("reference-queue-unprocessed.jsonl"),
            "orchestrator_errors": ref("orchestrator-errors.jsonl"),
            "reference_errors": ref("reference-errors.jsonl"),
        },
        "artifact_features": ref("feature-cache"),
    }
    dump_json(RUN_ROOT / "raw-artifact-index.json", payload)


class _SegmentPause:
    """Keep all simulator/reference workers at case boundaries between run segments."""

    def __init__(self, run_id: str, duration_seconds: int, participants: set[str]) -> None:
        self.run_id = run_id
        self.participants = set(participants)
        self.control_dir = RUN_ROOT / "control"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.control_dir / "request.json"
        self.status_path = self.control_dir / "status.json"
        self.condition = threading.Condition()
        self.cursors: dict[str, dict] = {}
        self.arrived: dict[str, dict] = {}
        self.finished: set[str] = set()
        self.state = "running"
        self.pause_reason: str | None = None
        self.pause_started: float | None = None
        self.pause_started_utc: str | None = None
        self.paused_seconds = 0.0
        self.active_seconds = 0.0
        self.segment_index = 1
        self.segment_duration_seconds = duration_seconds
        self.segment_started = time.monotonic()
        self.segment_deadline = self.segment_started + duration_seconds
        self.segment_deadline_utc = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + duration_seconds,
            tz=timezone.utc,
        ).isoformat()
        global TARGET_DEADLINE
        TARGET_DEADLINE = self.segment_deadline
        self.last_request_sequence = 0
        self.segments = [{
            "index": 1,
            "duration_seconds": duration_seconds,
            "started_at_utc": utc_now(),
            "deadline_utc": self.segment_deadline_utc,
        }]
        self.control_error: str | None = None
        self._write_status()

    def _read_request(self) -> tuple[int, str, int | None]:
        request = read_json(self.request_path)
        sequence = request.get("sequence", 0)
        state = request.get("desired_state", "running")
        duration = request.get("segment_duration_seconds")
        if type(sequence) is not int or sequence < 0 or state not in {"running", "pause"}:
            raise ValueError("control request must have a nonnegative sequence and desired_state")
        if duration is not None and (type(duration) is not int or duration < 1):
            raise ValueError("segment_duration_seconds must be a positive integer")
        return sequence, state, duration

    def _write_status(self) -> None:
        try:
            request_sequence, desired_state, _duration = self._read_request()
        except ValueError as exc:
            request_sequence, desired_state = self.last_request_sequence, "pause"
            self.control_error = str(exc)
        workers = {
            participant: {
                "state": (
                    "finished" if participant in self.finished
                    else "waiting" if participant in self.arrived
                    else "running"
                ),
                **self.cursors.get(participant, {}),
            }
            for participant in sorted(self.participants)
        }
        if self.state in {"pausing", "paused"}:
            desired_state = "pause"
        now = time.monotonic()
        active_elapsed = self.active_seconds + (
            max(0.0, now - self.segment_started)
            if self.state in {"running", "pausing"} else 0.0
        )
        dump_json(self.status_path, {
            "schema_version": "rq1-case-boundary-control-v1",
            "run_id": self.run_id,
            "method_filter": METHOD,
            "state": self.state,
            "desired_state": desired_state,
            "request_sequence": max(request_sequence, self.last_request_sequence),
            "pause_reason": self.pause_reason,
            "updated_at_utc": utc_now(),
            "pause_started_at_utc": self.pause_started_utc,
            "paused_seconds": round(self.paused_seconds, 6),
            "active_elapsed_seconds": round(active_elapsed, 6),
            "segment_index": self.segment_index,
            "segment_duration_seconds": self.segment_duration_seconds,
            "segment_deadline_utc": self.segment_deadline_utc,
            "segment_remaining_seconds": round(
                max(0.0, self.segment_deadline - now)
                if self.state not in {"paused", "completed", "interrupted"} else 0.0,
                1,
            ),
            "control_error": self.control_error,
            "workers": workers,
        })

    def _resume(self, duration_seconds: int | None) -> None:
        now = time.monotonic()
        if self.pause_started is not None:
            self.paused_seconds += max(0.0, now - self.pause_started)
            self.pause_started = None
            self.pause_started_utc = None
        else:
            self.active_seconds += max(0.0, now - self.segment_started)
            if self.segments:
                self.segments[-1]["ended_at_utc"] = utc_now()
                self.segments[-1]["active_elapsed_seconds"] = round(
                    now - self.segment_started, 6,
                )
        self.segment_index += 1
        self.segment_duration_seconds = duration_seconds or self.segment_duration_seconds
        self.segment_started = now
        self.segment_deadline = now + self.segment_duration_seconds
        self.segment_deadline_utc = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + self.segment_duration_seconds,
            tz=timezone.utc,
        ).isoformat()
        self.segments.append({
            "index": self.segment_index,
            "duration_seconds": self.segment_duration_seconds,
            "started_at_utc": utc_now(),
            "deadline_utc": self.segment_deadline_utc,
        })
        self.state = "running"
        self.pause_reason = None
        self.arrived.clear()
        global TARGET_DEADLINE
        TARGET_DEADLINE = self.segment_deadline
        self._write_status()
        self.condition.notify_all()

    def _consume_request(self) -> None:
        try:
            sequence, desired_state, duration = self._read_request()
            self.control_error = None
        except ValueError as exc:
            self.control_error = str(exc)
            return
        if sequence <= self.last_request_sequence:
            return
        self.last_request_sequence = sequence
        if desired_state == "pause" and self.state == "running":
            self.state = "pausing"
            self.pause_reason = "operator-request"
        elif desired_state == "running" and self.state == "paused":
            self._resume(duration)
        elif desired_state == "running" and self.state == "pausing":
            self._resume(duration)

    def _start_pause_if_ready(self) -> None:
        active = self.participants - self.finished
        if active and active.issubset(self.arrived) and self.state == "pausing":
            now = time.monotonic()
            elapsed = max(0.0, now - self.segment_started)
            self.active_seconds += elapsed
            if self.segments:
                self.segments[-1]["ended_at_utc"] = utc_now()
                self.segments[-1]["active_elapsed_seconds"] = round(elapsed, 6)
            self.pause_started = now
            self.pause_started_utc = utc_now()
            self.state = "paused"
            self._write_status()
            write_raw_artifact_index()
            self.condition.notify_all()

    def checkpoint(
        self, participant: str, cursor: dict, stop_requested: threading.Event,
    ) -> bool:
        with self.condition:
            self.cursors[participant] = dict(cursor)
            self._consume_request()
            if self.state == "running" and time.monotonic() >= self.segment_deadline:
                self.state = "pausing"
                self.pause_reason = "segment-budget-ended"
            if self.state != "pausing":
                self.arrived.pop(participant, None)
                return not stop_requested.is_set()
            self.arrived[participant] = dict(cursor)
            self._write_status()
            self._start_pause_if_ready()
            while not stop_requested.is_set():
                self._consume_request()
                self._start_pause_if_ready()
                if self.state == "running":
                    self.arrived.pop(participant, None)
                    return True
                if self.state in {"completed", "interrupted"}:
                    return False
                self.condition.wait(timeout=0.5)
            if self.pause_started is not None:
                self.paused_seconds += max(0.0, time.monotonic() - self.pause_started)
                self.pause_started = None
                self.pause_started_utc = None
            else:
                elapsed = max(0.0, time.monotonic() - self.segment_started)
                self.active_seconds += elapsed
                if self.segments:
                    self.segments[-1]["ended_at_utc"] = utc_now()
                    self.segments[-1]["active_elapsed_seconds"] = round(elapsed, 6)
            self.state = "interrupted"
            self._write_status()
            self.condition.notify_all()
            return False

    def finish(self, participant: str, cursor: dict, interrupted: bool) -> None:
        with self.condition:
            self.cursors[participant] = dict(cursor)
            self.arrived.pop(participant, None)
            self.finished.add(participant)
            self._consume_request()
            self._start_pause_if_ready()
            if self.finished == self.participants:
                if self.state in {"running", "pausing"}:
                    elapsed = max(0.0, time.monotonic() - self.segment_started)
                    self.active_seconds += elapsed
                    if self.segments:
                        self.segments[-1]["ended_at_utc"] = utc_now()
                        self.segments[-1]["active_elapsed_seconds"] = round(elapsed, 6)
                self.state = "interrupted" if interrupted else "completed"
            self._write_status()
            self.condition.notify_all()

    def snapshot(self) -> dict:
        with self.condition:
            return {
                "state": self.state,
                "segment_index": self.segment_index,
                "segment_duration_seconds": self.segment_duration_seconds,
                "remaining_seconds": round(max(0.0, self.segment_deadline - time.monotonic())
                                            if self.state not in {"paused", "completed", "interrupted"}
                                            else 0.0, 1),
                "active_elapsed_seconds": round(
                    self.active_seconds + (
                        max(0.0, time.monotonic() - self.segment_started)
                        if self.state in {"running", "pausing"} else 0.0
                    ), 3,
                ),
                "paused_seconds": round(self.paused_seconds, 3),
            }

    def refresh_status(self) -> None:
        with self.condition:
            self._write_status()

    def poll(self) -> None:
        with self.condition:
            old_state = self.state
            old_sequence = self.last_request_sequence
            self._consume_request()
            if self.state == "running" and time.monotonic() >= self.segment_deadline:
                self.state = "pausing"
                self.pause_reason = "segment-budget-ended"
            self._start_pause_if_ready()
            if self.state != old_state or self.last_request_sequence != old_sequence:
                self._write_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=lambda value: int(value, 0))
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    args = parser.parse_args()
    if args.duration_seconds < 1:
        parser.error("--duration-seconds must be a positive integer")

    stop_requested = threading.Event()
    stop_state = {"signal_name": None}

    def request_stop(signum, _frame) -> None:
        _record_stop_request(stop_requested, stop_state, signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    global CONFIG, TARGETS_BY_ID, PROVENANCE, START_MONO, START_UTC
    global TARGET_DEADLINE, SEED
    CONFIG = load_json(CONFIG_PATH)
    TARGETS_BY_ID = {
        str(item.get("id")): item for item in CONFIG.get("targets", [])
        if isinstance(item, dict) and item.get("id")
    }
    expected_target_ids = {
        "T-QEMU", "T-LRSV-INT", "T-LRSV-TRANS", "T-UNICORN",
        "T-RENODE", "T-RAX", "T-RVVM",
    }
    if set(TARGETS_BY_ID) != expected_target_ids:
        raise RuntimeError(
            f"configured simulator targets differ from the seven-target run: "
            f"{sorted(TARGETS_BY_ID)}"
        )
    config_gaps = validate_config(CONFIG)
    if config_gaps:
        raise RuntimeError("comparison config is invalid: " + ",".join(config_gaps))
    require_execution_plane()
    opcode_contract = CONFIG.get("rv_opcode_catalog_coverage")
    expected_form_count = (
        opcode_contract.get("form_count")
        if isinstance(opcode_contract, dict) else None
    )
    if expected_form_count != len(OFFICIAL_ALL_CATALOG_FORMS):
        raise RuntimeError(
            "RV opcode catalog form count mismatch: "
            f"config={expected_form_count}, generator={len(OFFICIAL_ALL_CATALOG_FORMS)}"
        )
    if not RUN_ROOT.is_dir() or not os.access(RUN_ROOT, os.W_OK):
        raise RuntimeError(f"run root is not writable: {RUN_ROOT}")
    ssh_config = Path(os.environ.get("RQ1_NATIVE_SSH_CONFIG", ""))
    known_hosts = Path(os.environ.get("RQ1_NATIVE_KNOWN_HOSTS", ""))
    if not ssh_config.is_file() or not os.access(ssh_config, os.R_OK):
        raise RuntimeError("K1 SSH config is unavailable")
    if not known_hosts.is_file() or not os.access(known_hosts, os.R_OK):
        raise RuntimeError("K1 known_hosts is unavailable")
    if not shutil.which("ssh"):
        raise RuntimeError("K1 SSH client is unavailable")
    stop_flush = Path(os.environ.get("RQ1_RVVM_STOP_FLUSH", ""))
    if not stop_flush.is_file():
        raise RuntimeError("RVVM source coverage requires RQ1_RVVM_STOP_FLUSH")
    expected_stop_flush_sha = os.environ.get("RQ1_RVVM_STOP_FLUSH_SHA256", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_stop_flush_sha):
        raise RuntimeError("RVVM stop-flush helper SHA-256 is missing or invalid")
    if sha256_file(stop_flush) != expected_stop_flush_sha:
        raise RuntimeError("RVVM stop-flush helper identity mismatch")
    for target_id, target in TARGETS_BY_ID.items():
        binary = dep(target.get("binary"))
        identity_path = dep(target.get("identity"))
        coverage_binary = dep(target.get("coverage_binary"))
        if not binary or not binary.is_file():
            raise RuntimeError(f"{target_id} binary is missing")
        if not identity_path or not identity_path.is_file():
            raise RuntimeError(f"{target_id} identity is missing")
        if not coverage_binary or not coverage_binary.is_file():
            raise RuntimeError(f"{target_id} coverage_binary is missing")
        identity = _identity(identity_path, binary, target)
        if identity.get("status") != "verified":
            raise RuntimeError(
                f"{target_id} target identity preflight failed: "
                f"{identity.get('reason_code') or identity.get('status')}"
            )
        expected_sha = target.get("coverage_binary_sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha.lower()):
            raise RuntimeError(f"{target_id} coverage binary SHA-256 is not pinned")
        if sha256_file(coverage_binary) != expected_sha.lower():
            raise RuntimeError(f"{target_id} coverage binary identity mismatch")

    PROVENANCE = _provenance(CONFIG, CONFIG_PATH)
    config_sha = sha256_file(CONFIG_PATH)
    SEED = args.seed if args.seed is not None else secrets.randbits(64)
    sample_path = RUN_ROOT / "selected-cases.jsonl"

    renode = TARGETS_BY_ID["T-RENODE"]
    renode_runtime = dep(renode.get("coverage_binary"))
    runtime_cache = _materialize_dotnet_runtime(RUN_ROOT, "T-RENODE", renode_runtime)
    dump_json(RUN_ROOT / "runtime-warmup.json", {
        "status": "ready",
        "path": str(runtime_cache.relative_to(RUN_ROOT)),
    })
    catalog = catalog_random_queue(sample_path, SEED)
    target_ids = list(TARGETS_BY_ID)
    try:
        k1_reference_parallelism = int(
            os.environ.get("RQ1_NATIVE_REFERENCE_PARALLELISM", "8")
        )
    except ValueError as exc:
        raise RuntimeError("RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer") from exc
    if k1_reference_parallelism < 1:
        raise RuntimeError("RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer")
    reference_worker_count = min(
        MAX_NATIVE_REFERENCE_PARALLELISM, k1_reference_parallelism,
    )
    selection_offsets, target_queues = build_target_queues(
        sample_path, RUN_ROOT / "target-queues", SEED,
        target_ids, catalog["queued_rows"],
    )
    dump_json(RUN_ROOT / "target-queues.json", target_queues)
    dump_json(RUN_ROOT / "catalog-selection.json", catalog)
    print(json.dumps({
        "event": "random-catalog-ready",
        "catalog_rows": catalog["generated_catalog_rows"],
        "forms": catalog["catalog_form_count"],
        "queue_sha256": catalog["queue_sha256"],
        "target_queue_sha256": {
            target_id: row["sha256"] for target_id, row in target_queues.items()
        },
        "generation_seconds": catalog["generation_seconds"],
        "seed": SEED,
    }), flush=True)

    initial_duration_seconds = args.duration_seconds
    resource_profile = {}
    try:
        resource_profile = json.loads(os.environ.get("RQ1_RESOURCE_PROFILE_JSON", "{}"))
    except (TypeError, ValueError):
        resource_profile = {}
    expected_resource_profile = (
        CONFIG.get("execution_policy", {})
        .get("resource_profiles", {})
        .get("rvgen_catalog_72h")
    )
    profile_limits = resource_profile.get("limits")
    if (
        not isinstance(expected_resource_profile, dict)
        or resource_profile != expected_resource_profile
        or not isinstance(profile_limits, dict)
    ):
        raise RuntimeError("RVGEN catalog resource profile is missing or invalid")
    # Queue/catalog preparation and runtime warm-up do not consume the first
    # active segment budget.
    START_MONO = time.monotonic()
    START_UTC = utc_now()
    TARGET_DEADLINE = START_MONO + initial_duration_seconds
    deadline_utc = datetime.fromtimestamp(
        datetime.fromisoformat(START_UTC).timestamp() + initial_duration_seconds,
        tz=timezone.utc,
    ).isoformat()
    pause_gate = _SegmentPause(
        RUN_ROOT.name,
        initial_duration_seconds,
        set(target_ids) | {
            f"reference-{worker_id:02d}" for worker_id in range(reference_worker_count)
        },
    )
    execution_manifest = {
        "schema_version": "rq1-direct-rvgen-catalog-execution-manifest-v4",
        "run_id": RUN_ROOT.name,
        "method_filter": METHOD,
        "action": "run",
        "started_at_utc": START_UTC,
        "duration_seconds": initial_duration_seconds,
        "deadline_utc": deadline_utc,
        "duration_semantics": "active segment budget; pause at target-case boundaries and wait for resume",
        "source_commit": os.environ.get("RQ1_SOURCE_COMMIT"),
        "source_dirty": os.environ.get("RQ1_SOURCE_DIRTY", "false").lower() == "true",
        "source_snapshot_sha256": os.environ.get("RQ1_SOURCE_SNAPSHOT_SHA256"),
        "config_sha256": config_sha,
        "config_snapshot": "config.snapshot.json",
        "queue_sha256": catalog["queue_sha256"],
        "random_seed": SEED,
    }
    dump_json(RUN_ROOT / "execution-manifest.json", execution_manifest)
    manifest = {
        "schema_version": "rq1-direct-rvgen-catalog-run-v5",
        "status": "running",
        **execution_manifest,
        "initial_segment_duration_seconds": initial_duration_seconds,
        "input_population": "sum(generate_form(form) for OFFICIAL_ALL_CATALOG_FORMS)",
        "catalog_form_count": catalog["catalog_form_count"],
        "generated_catalog_rows": catalog["generated_catalog_rows"],
        "previously_reported_catalog_rows": PREVIOUSLY_REPORTED_CATALOG_ROWS,
        "selection_algorithm": catalog["selection"],
        "selection_seed": SEED,
        "parallel_simulator_workers": len(target_ids),
        "parallel_reference_workers": reference_worker_count,
        "target_timeout_policy": "configured per-target timeout; segment budget is enforced at target-case boundaries",
        "target_queue_policy": (
            "one Python worker per target with independently seeded random permutation; "
            "simulator_session_mode=adapter-callback"
        ),
        "target_queues": target_queues,
        "target_reference": "independent K1 reference queue; RV64 Linux-user timeout/transport failure uses same-candidate T-QEMU observation",
        "targets": list(TARGETS_BY_ID),
        "source_execution": PROVENANCE,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "random_queue_path": "selected-cases.jsonl",
        "random_queue_sha256": catalog["queue_sha256"],
        "resource_profile": resource_profile,
        "resource_limit": {
            **profile_limits,
            "parallel_containers": resource_profile["parallel_containers"],
            "simultaneous_simulator_processes": resource_profile[
                "simultaneous_simulator_processes"
            ],
            "tmpfs": resource_profile["tmpfs"],
        },
        "trace_policy": "retain full trace through normal completion or a detected repeated out-of-ELF invalid PC",
        "coverage_mode": "raw-profiles-and-execution-traces",
        "raw_artifact_index": "raw-artifact-index.json",
        "raw_profiles": "retain simulator-produced raw profile files under coverage-raw/",
        "raw_traces": "retain complete execution traces under each case target worker",
    }
    dump_json(RUN_ROOT / "experiment-manifest.json", manifest)
    print(json.dumps({
        "event": "simulator-input-window-started",
        "started_at_utc": START_UTC,
        "deadline_utc": deadline_utc,
        "duration_seconds": initial_duration_seconds,
        "segment_budget_seconds": initial_duration_seconds,
        "random_seed": SEED,
        "target_scheduler": "independent-per-target-queue-workers",
        "target_worker_count": len(target_ids),
        "reference_worker_count": reference_worker_count,
        "resource_note": "all workers share the single configured 2-CPU/14-GiB container limit",
        "targets": list(TARGETS_BY_ID),
    }), flush=True)

    target_progress = {
        target_id: {
            "queue_position": 0,
            "queue_rows": catalog["queued_rows"],
            "queue_seed": target_queues[target_id]["seed"],
            "queue_sha256": target_queues[target_id]["sha256"],
            "completed_cases": 0,
            "attempted_cases": 0,
            "service_seconds": 0.0,
            "execution_seconds": 0.0,
            "last_completed_sequence": None,
            "last_status": None,
            "worker_status": "queued",
            "exhausted": False,
        }
        for target_id in target_ids
    }
    completed_bits = {target_id: bytearray(catalog["queued_rows"]) for target_id in target_ids}
    reference_counts = Counter()
    reference_queue: queue.Queue[int | None] = queue.Queue()
    reference_requested: set[int] = set()
    reference_request_lock = threading.Lock()
    reference_worker_done = threading.Event()
    target_workers_stopped = threading.Event()
    target_queues_closed = threading.Event()
    worker_activity = {
        target_id: {"status": "queued", "sequence": None, "stage": None}
        for target_id in target_ids
    }
    reference_activity = {
        worker_id: {"status": "queued", "sequence": None, "stage": None}
        for worker_id in range(reference_worker_count)
    }
    reference_workers_remaining = [reference_worker_count]
    worker_activity["reference"] = {
        "status": "queued", "sequence": None, "stage": None,
        "active_worker_count": 0, "worker_count": reference_worker_count,
    }
    progress_path = RUN_ROOT / "progress.json"
    selection_fd = os.open(sample_path, os.O_RDONLY)

    def state_snapshot() -> tuple[dict, dict, dict]:
        with _STATE_LOCK:
            return (
                copy.deepcopy(target_progress),
                copy.deepcopy(worker_activity),
                dict(reference_counts),
            )

    def write_runtime_progress(stage: str) -> None:
        now = time.monotonic()
        progress, activity, reference_count_snapshot = state_snapshot()
        control = pause_gate.snapshot()
        pause_gate.refresh_status()
        dump_json(progress_path, {
            "updated_at_utc": utc_now(),
            "elapsed_seconds": control["active_elapsed_seconds"],
            "input_window_remaining_seconds": control["remaining_seconds"],
            "segment_index": control["segment_index"],
            "case_boundary_state": control["state"],
            "catalog_rows": catalog["generated_catalog_rows"],
            "queued_rows_per_target": catalog["queued_rows"],
            "stage": "paused" if control["state"] == "paused" else stage,
            "target_workers": activity,
            "reference_queue_pending": reference_queue.qsize(),
            "reference_counts": reference_count_snapshot,
            "target_progress": progress,
        })

    def update_reference_activity(
        worker_id: int, status: str, stage: str | None = None,
        sequence: int | None = None, item: dict | None = None,
    ) -> None:
        with _STATE_LOCK:
            reference_activity[worker_id] = {
                "worker_id": worker_id,
                "status": status,
                "stage": stage,
                "sequence": sequence,
                "candidate_id": item.get("candidate_id") if item else None,
            }
            active = [
                activity for activity in reference_activity.values()
                if activity["status"] == "running"
            ]
            latest = active[-1] if active else {}
            worker_activity["reference"] = {
                "status": "running" if active else (
                    "stopped" if all(
                        activity["status"] == "stopped"
                        for activity in reference_activity.values()
                    ) else "idle"
                ),
                "stage": latest.get("stage"),
                "sequence": latest.get("sequence"),
                "candidate_id": latest.get("candidate_id"),
                "active_worker_count": len(active),
                "worker_count": reference_worker_count,
                "active_workers": [dict(activity) for activity in active],
            }

    def progress_callback(
        worker_id: int, stage: str, _target_id: str, sequence: int, item: dict,
    ) -> None:
        update_reference_activity(worker_id, "running", stage, sequence, item)

    def record_unsubmitted(
        target_id: str, sequence: int, item: dict, reason: str,
    ) -> None:
        append_jsonl(RUN_ROOT / "input-window-unsubmitted.jsonl", [{
            "at_utc": utc_now(),
            "target": target_id,
            "sequence": sequence,
            "candidate_id": item.get("candidate_id"),
            "reason": reason,
        }])
        print(json.dumps({
            "event": (
                "target-case-unsubmitted-at-interruption"
                if reason.startswith("external-interruption")
                else "target-case-unsubmitted-at-deadline"
            ),
            "target": target_id,
            "sequence": sequence,
            "candidate_id": item.get("candidate_id"),
            "reason": reason,
        }), flush=True)

    def request_reference(sequence: int) -> None:
        with reference_request_lock:
            if sequence not in reference_requested:
                reference_requested.add(sequence)
                reference_queue.put(sequence)

    def record_unprocessed_references(first: int | None, reason: str) -> None:
        pending = [first] if first is not None else []
        sentinels = 0
        while True:
            try:
                queued = reference_queue.get_nowait()
            except queue.Empty:
                break
            if queued is None:
                sentinels += 1
            else:
                pending.append(queued)
        pending_log = RUN_ROOT / "reference-queue-unprocessed.jsonl"
        for start in range(0, len(pending), 1_000):
            append_jsonl(pending_log, [{
                "sequence": value, "reason": reason,
            } for value in pending[start:start + 1_000]])
        for _ in range(sentinels):
            reference_queue.put(None)

    def reference_worker(worker_id: int) -> None:
        participant = f"reference-{worker_id:02d}"
        cursor = {"stage": "reference-queue", "sequence": None}
        current_sequence = None
        try:
            while True:
                try:
                    sequence = reference_queue.get(timeout=0.5)
                except queue.Empty:
                    if target_queues_closed.is_set():
                        return
                    cursor = {"stage": "reference-queue", "sequence": None}
                    if not pause_gate.checkpoint(
                        participant, cursor, target_workers_stopped,
                    ):
                        record_unprocessed_references(None, "external-interruption")
                        return
                    continue
                if sequence is None:
                    return
                current_sequence = sequence
                cursor = {"stage": "case-materialization", "sequence": sequence}
                if not pause_gate.checkpoint(participant, cursor, target_workers_stopped):
                    record_unprocessed_references(sequence, "external-interruption")
                    return
                if stop_requested.is_set() and target_workers_stopped.is_set():
                    record_unprocessed_references(sequence, "external-interruption")
                    return
                case_dir = None
                item = {}
                update_reference_activity(
                    worker_id, "running", "case-materialization", sequence,
                )
                try:
                    item = _read_queued_case(selection_fd, selection_offsets, sequence)
                    profile = str(item["testcase"].get("isa_profile", "")).lower()
                    required = {"linux-user"} if profile.startswith("rv64") else set()
                    case_dir, testcase, identity, artifacts, hashes = _case_assets(
                        sequence, item, required,
                    )
                    _ensure_reference(
                        sequence, item, testcase, identity, case_dir,
                        artifacts, hashes, target_progress, completed_bits,
                        reference_counts,
                        lambda stage, target_id, case_sequence, case_item: progress_callback(
                            worker_id, stage, target_id, case_sequence, case_item,
                        ),
                    )
                except Exception as exc:
                    _bump_counter(reference_counts, "reference_worker_errors")
                    append_jsonl(RUN_ROOT / "reference-errors.jsonl", [{
                        "at_utc": utc_now(),
                        "sequence": sequence,
                        "candidate_id": item.get("candidate_id"),
                        "error": f"{type(exc).__name__}: {exc}"[:1_000],
                        "traceback": traceback.format_exc()[-8_000:],
                    }])
                    if case_dir is not None and not (case_dir / "references.json").is_file():
                        gap = _board_reference_gap(
                            "linux-user", f"reference-worker-error:{type(exc).__name__}",
                        )
                        dump_json(case_dir / "references.json", _reference_map(gap))
                finally:
                    current_sequence = None
                    update_reference_activity(worker_id, "idle")
        finally:
            update_reference_activity(worker_id, "stopped")
            pause_gate.finish(
                participant,
                {"stage": "stopped", "sequence": current_sequence},
                stop_requested.is_set(),
            )
            with _STATE_LOCK:
                reference_workers_remaining[0] -= 1
                if reference_workers_remaining[0] == 0:
                    reference_worker_done.set()

    def record_orchestrator_gap(
        target_id: str, sequence: int, item: dict, case_dir: Path | None,
        exc: Exception,
    ) -> None:
        reason = f"orchestrator-error:{type(exc).__name__}"
        append_jsonl(RUN_ROOT / "orchestrator-errors.jsonl", [{
            "at_utc": utc_now(),
            "target": target_id,
            "sequence": sequence,
            "candidate_id": item.get("candidate_id"),
            "error": f"{type(exc).__name__}: {exc}"[:1_000],
            "traceback": traceback.format_exc()[-8_000:],
        }])
        with _case_lock(f"target-{target_id}", sequence):
            with _STATE_LOCK:
                if completed_bits[target_id][sequence]:
                    return
                gap_row = {
                    "id": target_id,
                    "kind": TARGETS_BY_ID[target_id].get("kind"),
                    "status": "gap",
                    "reason": reason,
                    "target_attempted": False,
                }
                completed_bits[target_id][sequence] = 1
                target_progress[target_id]["completed_cases"] += 1
                target_progress[target_id]["last_completed_sequence"] = sequence
                target_progress[target_id]["last_status"] = "orchestrator-error"
            append_jsonl(
                RUN_ROOT / "target-completions" / f"{target_id}.jsonl",
                [{
                    "sequence": sequence,
                    "candidate_id": item.get("candidate_id"),
                    "form": item.get("form"),
                    "case_dir": (
                        str(case_dir.relative_to(RUN_ROOT))
                        if case_dir is not None else None
                    ),
                    "status": "gap",
                    "reason": reason,
                    "target_attempted": False,
                    "error_record": "orchestrator-errors.jsonl",
                }],
            )
        print(json.dumps({
            "event": "orchestrator-gap",
            "target": target_id,
            "sequence": sequence,
            "candidate_id": item.get("candidate_id"),
            "reason": reason,
        }), flush=True)

    def target_worker(target_id: str) -> None:
        queue_path = RUN_ROOT / target_queues[target_id]["path"]
        cursor = {"queue_position": 0, "current_sequence": None, "stage": "queue-ready"}
        current_sequence = None
        try:
            with queue_path.open("r", encoding="ascii") as stream:
                with _STATE_LOCK:
                    target_progress[target_id]["worker_status"] = "running"
                    worker_activity[target_id] = {
                        "status": "running", "stage": "queue-ready", "sequence": None,
                    }
                while True:
                    line = stream.readline()
                    if not line:
                        with _STATE_LOCK:
                            target_progress[target_id]["exhausted"] = True
                            target_progress[target_id]["worker_status"] = "queue-exhausted"
                            worker_activity[target_id] = {
                                "status": "queue-exhausted", "stage": None, "sequence": None,
                            }
                        return
                    sequence = int(line)
                    current_sequence = sequence
                    cursor = {
                        "queue_position": target_progress[target_id]["queue_position"],
                        "last_completed_sequence": target_progress[target_id]["last_completed_sequence"],
                        "current_sequence": sequence,
                        "stage": "case-boundary",
                    }
                    if not pause_gate.checkpoint(target_id, cursor, stop_requested):
                        item = _read_queued_case(selection_fd, selection_offsets, sequence)
                        record_unsubmitted(
                            target_id, sequence, item,
                            "external-interruption-before-target-dispatch",
                        )
                        with _STATE_LOCK:
                            target_progress[target_id]["worker_status"] = "interrupted"
                            worker_activity[target_id] = {
                                "status": "interrupted", "stage": None, "sequence": sequence,
                            }
                        return
                    with _STATE_LOCK:
                        target_progress[target_id]["queue_position"] += 1
                        if completed_bits[target_id][sequence]:
                            current_sequence = None
                            continue
                    item = _read_queued_case(selection_fd, selection_offsets, sequence)
                    task_started = time.monotonic()
                    paused_seconds_before = pause_gate.snapshot()["paused_seconds"]
                    case_dir = None
                    cursor = {
                        "queue_position": target_progress[target_id]["queue_position"],
                        "last_completed_sequence": target_progress[target_id]["last_completed_sequence"],
                        "current_sequence": sequence,
                        "candidate_id": item.get("candidate_id"),
                        "stage": "case-materialization",
                    }
                    with _STATE_LOCK:
                        worker_activity[target_id] = {
                            "status": "running", "stage": "case-materialization",
                            "sequence": sequence, "candidate_id": item.get("candidate_id"),
                        }
                    try:
                        required_strata = {_target_stratum(TARGETS_BY_ID[target_id])}
                        if str(item["testcase"].get("isa_profile", "")).lower().startswith("rv64"):
                            required_strata.add("linux-user")
                        case_dir, testcase, identity, artifacts, hashes = _case_assets(
                            sequence, item, required_strata,
                        )
                        cursor["stage"] = "target-dispatch"
                        if not pause_gate.checkpoint(target_id, cursor, stop_requested):
                            record_unsubmitted(
                                target_id, sequence, item,
                                "external-interruption-before-target-dispatch",
                            )
                            with _STATE_LOCK:
                                target_progress[target_id]["worker_status"] = "interrupted"
                            return
                        request_reference(sequence)
                        with _case_lock(f"target-{target_id}", sequence):
                            with _STATE_LOCK:
                                already_completed = bool(completed_bits[target_id][sequence])
                            if not already_completed:
                                with _STATE_LOCK:
                                    worker_activity[target_id]["stage"] = "target-execution"
                                target_run, row = _run_one_target(
                                    target_id, sequence, item, case_dir, testcase,
                                    identity, artifacts, hashes, {}, skip_reference=True,
                                )
                                saved_references = read_json(case_dir / "references.json")
                                reference = saved_references.get("linux-user", {})
                                _persist_target(
                                    target_id, sequence, item, case_dir, testcase,
                                    identity, hashes, target_run, row, reference,
                                    target_progress, completed_bits,
                                )
                    except Exception as exc:
                        record_orchestrator_gap(target_id, sequence, item, case_dir, exc)
                    finally:
                        elapsed = max(
                            0.0,
                            time.monotonic() - task_started
                            - (pause_gate.snapshot()["paused_seconds"] - paused_seconds_before),
                        )
                        with _STATE_LOCK:
                            target_progress[target_id]["service_seconds"] += elapsed
                            if target_progress[target_id]["worker_status"] != "interrupted":
                                target_progress[target_id]["worker_status"] = "running"
                                worker_activity[target_id] = {
                                    "status": "running", "stage": "queue-wait",
                                    "sequence": None,
                                }
                            elif target_progress[target_id]["worker_status"] == "interrupted":
                                worker_activity[target_id] = {
                                    "status": "interrupted", "stage": None,
                                    "sequence": sequence,
                                }
                        current_sequence = None
                        cursor = {
                            "queue_position": target_progress[target_id]["queue_position"],
                            "last_completed_sequence": target_progress[target_id]["last_completed_sequence"],
                            "current_sequence": None,
                            "stage": "case-boundary",
                        }
        except Exception as exc:
            append_jsonl(RUN_ROOT / "orchestrator-errors.jsonl", [{
                "at_utc": utc_now(), "target": target_id,
                "error": f"target-worker-fatal:{type(exc).__name__}: {exc}"[:1_000],
                "traceback": traceback.format_exc()[-8_000:],
            }])
            with _STATE_LOCK:
                target_progress[target_id]["worker_status"] = "worker-error"
                worker_activity[target_id] = {
                    "status": "worker-error", "stage": None, "sequence": None,
                }
        finally:
            pause_gate.finish(
                target_id,
                {**cursor, "current_sequence": current_sequence},
                stop_requested.is_set(),
            )

    write_runtime_progress(stage="ready")
    reference_threads = [
        threading.Thread(
            target=reference_worker, args=(worker_id,),
            name=f"rvgen-reference-{worker_id:02d}", daemon=False,
        )
        for worker_id in range(reference_worker_count)
    ]
    for thread in reference_threads:
        thread.start()
    next_progress_at = START_MONO + 60
    sentinel_sent = False
    try:
        with ThreadPoolExecutor(
            max_workers=len(target_ids), thread_name_prefix="rvgen-target",
        ) as pool:
            target_futures = [pool.submit(target_worker, target_id) for target_id in target_ids]
            while True:
                now = time.monotonic()
                pause_gate.poll()
                targets_done = all(future.done() for future in target_futures)
                if targets_done and stop_requested.is_set():
                    target_workers_stopped.set()
                if targets_done and not sentinel_sent:
                    for _ in range(reference_worker_count):
                        reference_queue.put(None)
                    sentinel_sent = True
                    target_queues_closed.set()
                if targets_done and reference_worker_done.is_set():
                    break
                if now >= next_progress_at:
                    write_runtime_progress(stage="running")
                    with _STATE_LOCK:
                        progress_snapshot = copy.deepcopy(target_progress)
                        reference_count_snapshot = dict(reference_counts)
                    active_elapsed = pause_gate.snapshot()["active_elapsed_seconds"]
                    elapsed_hours = max(active_elapsed / 3600, 1e-9)
                    target_progress_lines = ",".join(
                        f"{tid}:{progress_snapshot[tid]['completed_cases']}/"
                        f"{progress_snapshot[tid]['queue_position']}@"
                        f"{progress_snapshot[tid]['completed_cases'] / elapsed_hours:.2f}cases/h"
                        for tid in target_ids
                    )
                    print(json.dumps({
                        "event": "target-queue-progress",
                        "elapsed_seconds": active_elapsed,
                        "remaining_seconds": pause_gate.snapshot()["remaining_seconds"],
                        "segment_index": pause_gate.snapshot()["segment_index"],
                        "case_boundary_state": pause_gate.snapshot()["state"],
                        "target_progress": target_progress_lines,
                        "reference_queue_pending": reference_queue.qsize(),
                        "reference_counts": reference_count_snapshot,
                    }), flush=True)
                    next_progress_at = now + 60
                time.sleep(0.5)
            for future in target_futures:
                future.result()
    finally:
        if not sentinel_sent:
            for _ in range(reference_worker_count):
                reference_queue.put(None)
        for thread in reference_threads:
            thread.join()
        os.close(selection_fd)

    join_counts = join_partial_event_references(target_ids)
    dump_json(RUN_ROOT / "reference-join-summary.json", {
        "joined_at_utc": utc_now(),
        "event_count_by_join_status": join_counts,
        "reference_queue_unprocessed": sum(
            1 for _ in (RUN_ROOT / "reference-queue-unprocessed.jsonl").open(
                "r", encoding="utf-8",
            )
        ) if (RUN_ROOT / "reference-queue-unprocessed.jsonl").is_file() else 0,
    })

    input_stop_mono = time.monotonic()
    pause_snapshot = pause_gate.snapshot()
    input_elapsed_seconds = pause_snapshot["active_elapsed_seconds"]
    final_input_stop = utc_now()
    for target_id in target_ids:
        if target_progress[target_id]["completed_cases"] >= catalog["queued_rows"]:
            target_progress[target_id]["exhausted"] = True
            target_progress[target_id]["queue_position"] = catalog["queued_rows"]
    unscheduled_by_target = {
        target_id: max(0, catalog["queued_rows"] - target_progress[target_id]["completed_cases"])
        for target_id in target_ids
    }
    write_runtime_progress(stage=(
        "external-interruption-partial" if stop_requested.is_set()
        else "catalog-exhausted" if all(
            target_progress[target_id]["exhausted"] for target_id in target_ids
        ) else "simulator-input-stopped"
    ))
    print(json.dumps({
        "event": "simulator-run-finished",
        "reason_code": "external-signal" if stop_requested.is_set() else None,
        "stop_signal": stop_state.get("signal_name"),
        "stopped_at_utc": final_input_stop,
        "active_elapsed_seconds": input_elapsed_seconds,
        "total_elapsed_seconds": round(input_stop_mono - START_MONO, 3),
        "paused_seconds": pause_snapshot["paused_seconds"],
        "segment_index": pause_snapshot["segment_index"],
        "target_cases_completed": {
            target_id: target_progress[target_id]["completed_cases"]
            for target_id in target_ids
        },
        "unscheduled_by_target": unscheduled_by_target,
        "reference_counts": dict(reference_counts),
    }), flush=True)

    finished_at = utc_now()
    all_targets_exhausted = all(target_progress[target_id]["exhausted"] for target_id in target_ids)
    outcome = _campaign_outcome(
        stop_state.get("signal_name"), all_targets_exhausted,
    )
    manifest.update({
        "schema_version": "rq1-direct-rvgen-catalog-run-v5",
        **outcome,
        "started_at_utc": START_UTC,
        "simulator_input_stopped_at_utc": final_input_stop,
        "finished_at_utc": finished_at,
        "initial_segment_duration_seconds": initial_duration_seconds,
        "input_elapsed_seconds": input_elapsed_seconds,
        "total_elapsed_seconds": round(time.monotonic() - START_MONO, 3),
        "paused_seconds": pause_snapshot["paused_seconds"],
        "pause_resume_segments": pause_gate.segments,
        "last_segment_deadline_utc": pause_gate.segment_deadline_utc,
        "random_seed": SEED,
        "selection": catalog,
        "target_progress": target_progress,
        "unscheduled_by_target": unscheduled_by_target,
        "reference_counts": dict(reference_counts),
    })
    dump_json(RUN_ROOT / "experiment-manifest.json", manifest)
    write_raw_artifact_index()
    print(json.dumps({
        "event": "experiment-finished",
        "active_elapsed_seconds": input_elapsed_seconds,
        "paused_seconds": pause_snapshot["paused_seconds"],
        "target_cases_completed": {
            target_id: target_progress[target_id]["completed_cases"]
            for target_id in target_ids
        },
        "raw_artifact_index": str(RUN_ROOT / "raw-artifact-index.json"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        try:
            dump_json(RUN_ROOT / "experiment-failure.json", {
                "at_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - (START_MONO or PROCESS_STARTED_MONO), 3,
                ),
                "input_window_started": START_MONO > 0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        try:
            write_raw_artifact_index()
        except Exception:
            pass
        raise

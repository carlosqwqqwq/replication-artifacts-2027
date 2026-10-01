#!/usr/bin/env python3
"""Stage one trace manifest on NVMe, scan it, verify, and clean up."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _mount_source(path: Path) -> str:
    result = subprocess.run(
        ["findmnt", "-T", str(path), "-o", "SOURCE", "-n"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot identify filesystem for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def _manifest_rows(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        candidate, separator, raw_path = line.partition("\t")
        if not separator or not raw_path:
            raise ValueError(f"invalid manifest line {line_number}: {line!r}")
        source = Path(raw_path)
        if not source.is_file():
            raise FileNotFoundError(f"trace missing at line {line_number}: {source}")
        rows.append((candidate, source))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _copy_rows(rows: list[tuple[str, Path]], stage: Path) -> tuple[Path, int, int]:
    trace_root = stage / "traces"
    trace_root.mkdir()
    manifest = stage / "trace-manifest.tsv"
    source_bytes = 0
    staged_bytes = 0
    with manifest.open("w", encoding="utf-8") as stream:
        for index, (candidate, source) in enumerate(rows):
            destination = trace_root / f"{index:06d}-{source.name}"
            shutil.copyfile(source, destination)
            source_size = source.stat().st_size
            staged_size = destination.stat().st_size
            if source_size != staged_size:
                raise IOError(
                    f"size mismatch after copy: {source} ({source_size}) -> "
                    f"{destination} ({staged_size})"
                )
            stream.write(f"{candidate}\t{destination}\n")
            source_bytes += source_size
            staged_bytes += staged_size
    return manifest, source_bytes, staged_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy one trace manifest to NVMe and benchmark trace_scan."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scanner", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--stage-parent", type=Path, default=Path("/var/tmp"))
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = args.manifest.resolve(strict=True)
    scanner = args.scanner.resolve(strict=True)
    if not scanner.is_file() or not scanner.stat().st_mode & 0o111:
        raise ValueError(f"scanner is not executable: {scanner}")
    stage_parent = args.stage_parent.resolve(strict=True)
    device = _mount_source(stage_parent)
    if "nvme" not in device.lower():
        raise RuntimeError(
            f"stage parent is not on an NVMe device: {stage_parent} -> {device}"
        )

    rows = _manifest_rows(manifest)
    expected_bytes = sum(source.stat().st_size for _, source in rows)
    free_bytes = shutil.disk_usage(stage_parent).free
    if free_bytes < expected_bytes * 2:
        raise RuntimeError(
            f"insufficient NVMe space: free={free_bytes} expected={expected_bytes}"
        )

    result: dict[str, object] = {
        "manifest": str(manifest),
        "scanner": str(scanner),
        "kind": args.kind,
        "filesystem": device,
        "trace_count": len(rows),
        "source_bytes": expected_bytes,
        "cleanup": "pending",
    }
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(
            prefix="rq1-ssd-trace-", dir=str(stage_parent)
        ) as temporary:
            stage = Path(temporary)
            staged_manifest, source_bytes, staged_bytes = _copy_rows(rows, stage)
            result["copy_seconds"] = round(time.monotonic() - started, 6)
            result["staged_bytes"] = staged_bytes
            if source_bytes != staged_bytes:
                raise IOError(
                    f"batch byte mismatch: source={source_bytes} staged={staged_bytes}"
                )

            scan_output = stage / "scan.pc.tsv"
            scan_started = time.monotonic()
            completed = subprocess.run(
                [
                    str(scanner),
                    "--input",
                    str(staged_manifest),
                    "--kind",
                    args.kind,
                    "--output",
                    str(scan_output),
                ],
                check=False,
            )
            result["scan_seconds"] = round(time.monotonic() - scan_started, 6)
            result["scan_returncode"] = completed.returncode
            result["scan_output_bytes"] = scan_output.stat().st_size if scan_output.is_file() else 0
            if completed.returncode != 0 or not scan_output.is_file():
                raise RuntimeError("SSD trace scan failed")

            if args.reference_output:
                reference = args.reference_output.resolve(strict=True)
                result["reference_output"] = str(reference)
                result["reference_match"] = filecmp.cmp(
                    scan_output, reference, shallow=False
                )
                if not result["reference_match"]:
                    raise RuntimeError("SSD scan output differs from reference output")
        result["cleanup"] = "complete"
    except Exception:
        result["cleanup"] = "complete"
        raise
    finally:
        if args.result:
            output = args.result.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

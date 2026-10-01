"""Per-cell NVMe staging for offline trace and coverage processing.

The run roots stay on the HDD.  A caller stages one cell's stable inputs into
``/var/tmp`` (NVMe on the experiment host), consumes the returned paths, and
lets this module remove the temporary copy when the context exits.
"""

from __future__ import annotations

import os
import stat
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator


def _stage_parent(value: Path | str | None) -> Path:
    parent = Path(value or os.environ.get("RQ1_SSD_STAGE_PARENT", "/var/tmp"))
    parent = parent.resolve(strict=True)
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise RuntimeError(f"SSD stage parent is not writable: {parent}")
    return parent


@lru_cache(maxsize=8)
def _mount_source(parent: str) -> str:
    result = subprocess.run(
        ["findmnt", "-T", parent, "-o", "SOURCE", "-n"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot identify filesystem for SSD stage parent {parent}: "
            f"{result.stderr.strip()}"
        )
    device = result.stdout.strip()
    if "nvme" not in device.lower():
        raise RuntimeError(
            f"SSD stage parent is not on NVMe: {parent} -> {device}"
        )
    return device


def _prepare_parent(value: Path | str | None) -> Path:
    parent = _stage_parent(value)
    _mount_source(str(parent))
    return parent


def _copy_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_size = source.stat().st_size
    staged_size = destination.stat().st_size
    if source_size != staged_size:
        raise IOError(
            f"SSD staged byte mismatch: {source} ({source_size}) -> "
            f"{destination} ({staged_size})"
        )
    return staged_size


def _check_space(parent: Path, expected_bytes: int) -> None:
    free_bytes = shutil.disk_usage(parent).free
    # Keep one input-sized reserve for temporary scanner/collector output and
    # unexpected filesystem overhead.
    if free_bytes < expected_bytes * 2:
        raise RuntimeError(
            f"insufficient NVMe space: free={free_bytes} expected={expected_bytes}"
        )


@contextmanager
def stage_paths(
    paths: Iterable[Path], *, stage_parent: Path | str | None = None,
    prefix: str = "rq1-ssd-trace-",
) -> Iterator[tuple[dict[Path, Path], dict[str, object]]]:
    """Stage individual files and yield ``source -> SSD copy`` mappings."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for value in paths:
        source = Path(value).resolve(strict=True)
        if not source.is_file():
            raise FileNotFoundError(f"SSD stage input is not a file: {source}")
        if source not in seen:
            seen.add(source)
            unique.append(source)
    if not unique:
        yield {}, {"file_count": 0, "source_bytes": 0, "staged_bytes": 0}
        return
    parent = _prepare_parent(stage_parent)
    source_bytes = sum(path.stat().st_size for path in unique)
    _check_space(parent, source_bytes)
    temporary = Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    mapping: dict[Path, Path] = {}
    staged_bytes = 0
    metadata = {
        "stage_dir": str(temporary), "filesystem": _mount_source(str(parent)),
        "file_count": len(unique), "source_bytes": source_bytes,
        "staged_bytes": staged_bytes,
        "cleanup": "pending",
    }
    started = time.monotonic()
    try:
        destination_root = temporary / "traces"
        for index, source in enumerate(unique):
            destination = destination_root / f"{index:08d}-{source.name}"
            staged_bytes += _copy_file(source, destination)
            mapping[source] = destination
        if staged_bytes != source_bytes:
            raise IOError(
                f"SSD stage batch byte mismatch: source={source_bytes} "
                f"staged={staged_bytes}"
            )
        metadata["staged_bytes"] = staged_bytes
        metadata["stage_seconds"] = round(time.monotonic() - started, 6)
        yield mapping, metadata
    finally:
        try:
            shutil.rmtree(temporary)
        except OSError:
            metadata["cleanup"] = "failed"
            raise
        metadata["cleanup"] = "complete"


def _iter_files(root: Path, predicate: Callable[[Path], bool]) -> Iterator[Path]:
    for directory, _subdirectories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in filenames:
            path = base / name
            # ``filenames`` already excludes directories.  Deferring the
            # regular-file check until the canonical ``stat`` below avoids a
            # second HDD metadata lookup for every ordinary profile file.
            if predicate(path):
                yield path


def _stage_tar_chunk(
    tar: str, common: Path, extracted: Path, archive_list: Path,
) -> None:
    """Stream one file list from HDD into the shared NVMe root."""
    producer = subprocess.Popen(
        [tar, "-C", str(common), "--null", "--no-recursion", "--dereference",
         "--files-from", str(archive_list), "-cf", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert producer.stdout is not None
    consumer = None
    try:
        consumer = subprocess.Popen(
            [tar, "-C", str(extracted), "--no-overwrite-dir", "-xf", "-"],
            stdin=producer.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        producer.stdout.close()
        _consumer_stdout, consumer_stderr = consumer.communicate()
        _producer_stdout, producer_stderr = producer.communicate()
        if producer.returncode or consumer.returncode:
            detail = (producer_stderr or consumer_stderr or b"")[-500:]
            raise IOError(
                f"SSD tar stage failed: producer={producer.returncode} "
                f"consumer={consumer.returncode} {detail.decode(errors='replace')}"
            )
    except BaseException:
        for process in (consumer, producer):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (consumer, producer):
            if process is None:
                continue
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    finally:
        if producer.stdout is not None:
            producer.stdout.close()


@contextmanager
def stage_roots(
    roots: Iterable[Path], *, predicate: Callable[[Path], bool],
    stage_parent: Path | str | None = None,
    prefix: str = "rq1-ssd-profile-",
) -> Iterator[tuple[dict[Path, Path], dict[str, object]]]:
    """Stage files under directory roots while preserving each root layout.

    The returned mapping is keyed by the resolved original root.  Duplicate
    source files shared by overlapping roots are represented by SSD symlinks,
    so one cell is never copied twice merely because two metadata views point
    at the same file.
    """
    original_roots: list[Path] = []
    seen_roots: set[Path] = set()
    files_by_root: dict[Path, list[tuple[Path, Path]]] = {}
    source_sizes: dict[Path, int] = {}
    metadata: dict[str, object] | None = None
    for value in roots:
        root = Path(value).resolve()
        if not root.is_dir() or root in seen_roots:
            continue
        seen_roots.add(root)
        original_roots.append(root)
        files: list[tuple[Path, Path]] = []
        for path in _iter_files(root, predicate):
            resolved = path.resolve(strict=True)
            file_stat = resolved.stat()
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            files.append((path, resolved))
            source_sizes.setdefault(resolved, file_stat.st_size)
        files_by_root[root] = files
    unique_files = list(source_sizes)
    if not unique_files:
        yield {}, {"root_count": len(original_roots), "file_count": 0,
                   "source_bytes": 0, "staged_bytes": 0}
        return
    parent = _prepare_parent(stage_parent)
    # Source sizes were captured during the initial manifest walk.  After the
    # tar stream completes, re-stat only the NVMe copies; re-resolving every
    # HDD path a second time turns validation into another full seek workload.
    source_bytes = sum(source_sizes.values())
    _check_space(parent, source_bytes)
    temporary = Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    started = time.monotonic()
    mapping: dict[Path, Path] = {}
    archive_paths: dict[Path, Path] = {}
    aliases_by_file: dict[Path, list[Path]] = {}
    for root in original_roots:
        for source, resolved in files_by_root[root]:
            archive_paths.setdefault(resolved, source)
            aliases_by_file.setdefault(resolved, []).append(source)
    try:
        common = Path(os.path.commonpath([str(root) for root in original_roots]))
        extracted = temporary / "root-data"
        extracted.mkdir(parents=True, exist_ok=True)
        archive_values = list(archive_paths.values())
        tar = shutil.which("tar")
        if tar:
            archive_list = temporary / "files.list"
            with archive_list.open("wb") as stream:
                for source in archive_values:
                    stream.write(str(source.relative_to(common)).encode("utf-8"))
                    stream.write(b"\0")
            _stage_tar_chunk(tar, common, extracted, archive_list)
        else:
            # Keep a portable fallback for hosts without GNU tar.  The normal
            # experiment host uses tar because it is much faster for the
            # hundreds of thousands of small gcda files in one cell.
            for source in archive_paths.values():
                _copy_file(source.resolve(strict=True), extracted / source.relative_to(common))
        staged_bytes = 0
        for resolved, source in archive_paths.items():
            destination = extracted / source.relative_to(common)
            if not destination.is_file():
                raise IOError(f"SSD tar stage missing file: {destination}")
            source_size = source_sizes[resolved]
            if destination.stat().st_size != source_size:
                raise IOError(f"SSD tar stage size mismatch: {source}")
            staged_bytes += source_size
            # A symlink or hardlink may expose the same underlying file from
            # more than one input root.  The tar stream contains one canonical
            # copy, so recreate the other lexical paths on NVMe; otherwise a
            # root mapped below can silently point at a missing file.
            for alias in aliases_by_file[resolved]:
                alias_destination = extracted / alias.relative_to(common)
                if alias_destination == destination:
                    continue
                alias_destination.parent.mkdir(parents=True, exist_ok=True)
                if not alias_destination.exists() and not alias_destination.is_symlink():
                    alias_destination.symlink_to(
                        os.path.relpath(destination, alias_destination.parent)
                    )
        for root in original_roots:
            staged_root = extracted / root.relative_to(common)
            staged_root.mkdir(parents=True, exist_ok=True)
            mapping[root] = staged_root
        if staged_bytes != source_bytes:
            raise IOError(
                f"SSD profile batch byte mismatch: source={source_bytes} "
                f"staged={staged_bytes}"
            )
        metadata = {
            "stage_dir": str(temporary), "filesystem": _mount_source(str(parent)),
            "root_count": len(original_roots), "file_count": len(unique_files),
            "source_bytes": source_bytes, "staged_bytes": staged_bytes,
            "stage_jobs": 1,
            "stage_seconds": round(time.monotonic() - started, 6),
            "cleanup": "pending",
        }
        yield mapping, metadata
    finally:
        try:
            shutil.rmtree(temporary)
        except OSError:
            if metadata is not None:
                metadata["cleanup"] = "failed"
            raise
        if metadata is not None:
            metadata["cleanup"] = "complete"


def coverage_file_predicate(collector: str) -> Callable[[Path], bool]:
    collector = str(collector)
    if collector == "dotnet":
        return lambda path: path.name.endswith(
            (".coverage", ".cobertura.xml", ".cobertura.xml.gz")
        )
    if collector == "llvm":
        return lambda path: path.name.endswith(".profraw")
    if collector in {"gcov", "lcov"}:
        return lambda path: (
            path.name.endswith(".gcda") and not path.name.endswith(".tmp.gcda")
        ) or path.name == ".rq1-gcov-flush-incomplete.json"
    return lambda _path: False

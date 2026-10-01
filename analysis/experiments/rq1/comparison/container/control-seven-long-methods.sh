#!/usr/bin/env bash
set -euo pipefail

output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
action=${1:-}
shift || true
run_prefix=
method_filter=
duration_seconds=
methods=(B-RVDV B-TORTURE B-CSMITH B-GEMI Fuzz4All Ours-RVGEN-Direct Ours-Program-Full)

usage() {
  cat <<'EOF'
用法：
  container/control-seven-long-methods.sh {pause|resume|status} --run-prefix PREFIX [--method METHOD] [--duration-seconds N]

pause 等各活动运行完成当前 case 后暂停；容器和 runner 保持存活。
resume 在同一 run 和 case 游标继续；可用 --duration-seconds 指定新的活跃运行段。
不传时，resume 沿用上一段长度。status 只查看本 7×7 run 的状态和游标。
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --run-prefix) run_prefix=${2:?--run-prefix requires a value}; shift 2;;
    --method) method_filter=${2:?--method requires a value}; shift 2;;
    --duration-seconds) duration_seconds=${2:?--duration-seconds requires a value}; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ "$action" == pause || "$action" == resume || "$action" == status ]] || {
  usage >&2; exit 2;
}
[[ "$run_prefix" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid run-prefix' >&2; exit 2; }
[[ -z "$duration_seconds" || "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--duration-seconds must be a positive integer' >&2; exit 2;
}
if [[ "$action" != resume && -n "$duration_seconds" ]]; then
  echo '--duration-seconds only applies to resume' >&2; exit 2;
fi
if [[ -n "$method_filter" ]] && [[ ! " ${methods[*]} " == *" $method_filter "* ]]; then
  echo "unknown method: $method_filter" >&2; exit 2;
fi
if [[ -n "$method_filter" ]]; then methods=("$method_filter"); fi
[[ "$output_root" == /path/to/rq1-comparison/results ]] || {
  echo 'output-root must be exactly /path/to/rq1-comparison/results' >&2; exit 2;
}

launch_dir="$output_root/launch-logs/$run_prefix"
[[ -d "$launch_dir" ]] || { echo "unknown 7×7 run prefix: $run_prefix" >&2; exit 2; }

python3 - "$action" "$output_root" "$launch_dir" "$run_prefix" "$duration_seconds" "${methods[@]}" <<'PY'
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

action, output_root, launch_dir, prefix, duration, *methods = sys.argv[1:]
output_root = Path(output_root)
launch_dir = Path(launch_dir)

def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}

known_external_methods = {"B-RVDV", "B-TORTURE", "B-CSMITH", "B-GEMI", "Fuzz4All"}
external_methods = known_external_methods

if action == "resume" and any(method in known_external_methods for method in methods):
    launch_manifest = read(launch_dir / "launch-manifest.json")
    if not launch_manifest:
        raise SystemExit(f"launch manifest is unavailable: {launch_dir / 'launch-manifest.json'}")
    catalog = launch_manifest.get("case_catalog")
    if not isinstance(catalog, dict):
        raise SystemExit("launch manifest has no case_catalog")
    external_methods_value = catalog.get("external_methods")
    if not isinstance(external_methods_value, list) \
            or any(not isinstance(item, str) for item in external_methods_value):
        raise SystemExit("launch manifest external_methods is invalid")
    if set(external_methods_value) - known_external_methods:
        raise SystemExit("launch manifest external_methods contains an unknown method")
    external_methods = set(external_methods_value)
    if any(method in known_external_methods and method not in external_methods for method in methods):
        raise SystemExit("selected external method is absent from launch manifest")

def root_for(method):
    return output_root / "runs" / f"{prefix}-{method}"

def unpause_container(method):
    """Make a manually Docker-paused run runnable before sending resume."""
    container_name = f"rq1cmp-{root_for(method).name}"
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Paused}}", container_name],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return
    if result.stdout.strip().lower() != "true":
        return
    resumed = subprocess.run(
        ["docker", "unpause", container_name],
        capture_output=True, text=True, check=False,
    )
    if resumed.returncode != 0:
        detail = (resumed.stderr or resumed.stdout or "docker unpause failed").strip()
        raise SystemExit(f"cannot unpause {container_name}: {detail}")

def is_terminal(method, root):
    status = read(root / "control" / "status.json").get("state")
    if status in {"completed", "failed", "stopped"}:
        return True
    wrapper = read(root / "wrapper-result.json").get("status")
    if wrapper in {"completed", "partial", "failed"}:
        return True
    if not (launch_dir / "completed" / f"{method}.done").is_file():
        return False
    container_name = f"rq1cmp-{root.name}"
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
        capture_output=True, text=True, check=False,
    )
    return result.returncode != 0 or result.stdout.strip() != "running"

def write_request(root, desired_state, segment_duration=None):
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    request_path = control / "request.json"
    current = read(request_path)
    sequence = current.get("sequence", 0)
    if type(sequence) is not int or sequence < 0:
        raise SystemExit(f"invalid control sequence: {request_path}")
    request = {
        "schema_version": "rq1-case-boundary-request-v1",
        "sequence": sequence + 1,
        "desired_state": desired_state,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if segment_duration is not None:
        request["segment_duration_seconds"] = int(segment_duration)
    temporary = request_path.with_name(request_path.name + ".tmp")
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, request_path)

def snapshot(method):
    root = root_for(method)
    control = read(root / "control" / "status.json")
    container_name = f"rq1cmp-{root.name}"
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
            capture_output=True, text=True, check=False,
        )
        container_state = result.stdout.strip() if result.returncode == 0 else "missing"
    except OSError:
        container_state = "unavailable"
    return {
        "method": method,
        "run_id": root.name,
        "state": control.get("state", "starting"),
        "desired_state": control.get("desired_state"),
        "pause_reason": control.get("pause_reason"),
        "request_sequence": control.get("request_sequence"),
        "segment_index": control.get("segment_index"),
        "segment_duration_seconds": control.get("segment_duration_seconds"),
        "segment_remaining_seconds": control.get("segment_remaining_seconds"),
        "paused_seconds": control.get("paused_seconds"),
        "workers": control.get("workers", {}),
        "container": container_name,
        "container_state": container_state,
        "terminal": is_terminal(method, root),
    }

if action in {"pause", "resume"}:
    desired = "pause" if action == "pause" else "running"
    waiting_for_roots = set(methods)
    sent = set()
    while waiting_for_roots:
        for method in list(waiting_for_roots):
            root = root_for(method)
            if not root.exists():
                failure = launch_dir / "completed" / f"{method}.failure.tsv"
                if failure.is_file():
                    waiting_for_roots.remove(method)
                continue
            if is_terminal(method, root):
                waiting_for_roots.remove(method)
                continue
            control = read(root / "control" / "status.json")
            state = control.get("state")
            request = read(root / "control" / "request.json")
            request_state = request.get("desired_state")
            wanted = "paused" if action == "pause" else "running"
            if action == "resume" and state == "paused" \
                    and control.get("resume_rejection_sequence") == request.get("sequence"):
                raise SystemExit(
                    f"resume rejected for {method}: "
                    + str(control.get("resume_rejection_reason") or "unknown reason")
                )
            if state == wanted and request_state == desired:
                waiting_for_roots.remove(method)
                continue
            if method not in sent:
                if action == "resume":
                    unpause_container(method)
                write_request(
                    root, desired,
                    int(duration) if action == "resume" and duration else None,
                )
                sent.add(method)
                request = read(root / "control" / "request.json")
                request_state = request.get("desired_state")
                if action == "resume" and state == "paused" \
                        and control.get("resume_rejection_sequence") == request.get("sequence"):
                    raise SystemExit(
                        f"resume rejected for {method}: "
                        + str(control.get("resume_rejection_reason") or "unknown reason")
                    )
        if waiting_for_roots:
            time.sleep(0.5)

for method in methods:
    print(json.dumps(snapshot(method), ensure_ascii=False))

index_path = launch_dir / "raw-artifact-index.json"
index = read(index_path)
if isinstance(index.get("runs"), list):
    for row in index["runs"]:
        method = row.get("method") if isinstance(row, dict) else None
        if method not in methods:
            continue
        root = root_for(method)
        control = read(root / "control" / "status.json")
        wrapper = read(root / "wrapper-result.json")
        row.update(
            runner_state=control.get("state", "starting"),
            wrapper_status=wrapper.get("status"),
            wrapper_exit_code=wrapper.get("exit_code"),
            container_oom_killed=wrapper.get("container_oom_killed"),
            next_case_cursors=control.get("workers", {}),
        )
        for key in (
            "execution_manifest", "run_result", "wrapper_result", "container_inspect",
            "failure_record", "control_request", "control_status", "ledger_events",
            "partial_ledger_events", "coverage_registry",
        ):
            ref = row.get(key)
            if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                ref["exists"] = (output_root / ref["path"]).exists()
        roots = row.get("raw_artifact_roots")
        if isinstance(roots, dict):
            if method.startswith("Ours-"):
                for key, relative in {
                    "framework_target_replays": Path("framework") / method / "lanes",
                    "framework_experiments": Path("framework-experiments"),
                    "framework_coverage_configs": Path("framework-coverage-configs") / method,
                    "simulator_traces": Path("traces"),
                }.items():
                    path = root / relative
                    roots[key] = {
                        "path": path.relative_to(output_root).as_posix(),
                        "exists": path.exists(),
                    }
            for ref in roots.values():
                if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                    ref["exists"] = (output_root / ref["path"]).exists()
    index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = index_path.with_name(index_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, index_path)
PY

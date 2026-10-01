#!/usr/bin/env bash
set -euo pipefail

output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
methods=(B-RVDV B-TORTURE B-CSMITH B-GEMI Fuzz4All)
action=
run_prefix=
method_filter=
duration_seconds=

usage() {
  cat <<'EOF'
用法：
  container/control-five-tool-replay.sh {pause|resume|status} --run-prefix PREFIX [--method METHOD] [--duration-seconds N]

pause 在当前 case 完成后暂停全部 Target 和 K1；resume 在同一容器中继续。
  resume 可用 --duration-seconds 指定下一段预算；省略时沿用当前段长。
无 --method 时，已完成或失败的方法作为终态跳过，其余活动方法仍须确认到达 case 边界。
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
action=$1
shift
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --run-prefix) run_prefix=$2; shift 2;;
    --method) method_filter=$2; shift 2;;
    --duration-seconds) duration_seconds=$2; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$action" == pause || "$action" == resume || "$action" == status ]] || {
  echo 'action must be pause, resume, or status' >&2; exit 2;
}
[[ "$run_prefix" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo 'invalid run-prefix' >&2; exit 2;
}
[[ -z "$duration_seconds" || "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--duration-seconds must be a positive integer' >&2; exit 2;
}
if [[ "$action" != resume && -n "$duration_seconds" ]]; then
  echo '--duration-seconds only applies to resume' >&2; exit 2;
fi
if [[ -n "$method_filter" ]]; then
  [[ " ${methods[*]} " == *" $method_filter "* ]] || {
    echo "unknown method: $method_filter" >&2; exit 2;
  }
  methods=("$method_filter")
fi
[[ "$output_root" == /path/to/rq1-comparison/results ]] || {
  echo 'output-root must be exactly /path/to/rq1-comparison/results' >&2; exit 2;
}

launch_dir="$output_root/launch-logs/$run_prefix"
[[ -d "$launch_dir" ]] || {
  echo "unknown replay prefix: $run_prefix" >&2; exit 2;
}
manifest_methods=$(python3 - "$launch_dir/replay-batch-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError) as error:
    raise SystemExit(f"replay manifest is unavailable: {error}")
methods = manifest.get("methods") if isinstance(manifest, dict) else None
if not isinstance(methods, dict) or not methods:
    raise SystemExit("replay manifest has no methods")
print("\n".join(methods))
PY
)
mapfile -t methods <<<"$manifest_methods"
if [[ -n "$method_filter" ]] &&
   [[ ! " ${methods[*]} " == *" $method_filter "* ]]; then
  echo "method is not part of replay prefix: $method_filter" >&2; exit 2;
fi
if [[ -n "$method_filter" ]]; then methods=("$method_filter"); fi
run_roots=()
for method in "${methods[@]}"; do
  run_roots+=("$output_root/runs/$run_prefix-$method")
done

python3 - "$action" "$launch_dir" "$run_prefix" "$duration_seconds" \
  "${run_roots[@]}" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

action, launch_dir, run_prefix, duration_seconds, *roots = sys.argv[1:]
launch_dir = Path(launch_dir)

def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return default

def method_for(root):
    prefix = run_prefix + "-"
    name = Path(root).name
    return name[len(prefix):] if name.startswith(prefix) else name

def terminal(root):
    root = Path(root)
    method = method_for(root)
    wrapper = read(root / "wrapper-result.json", {})
    status = wrapper.get("status") if isinstance(wrapper, dict) else None
    done = (launch_dir / "completed" / f"{method}.done").is_file()
    control_status = read(root / "control" / "status.json", {})
    control_state = control_status.get("state") if isinstance(control_status, dict) else None
    finished = done or status in {"completed", "partial", "failed"} \
        or control_state == "completed"
    return finished, status, done

def write_request(root, state, segment_duration=None):
    control = Path(root) / "control"
    path = control / "request.json"
    current = read(path, {})
    sequence = current.get("sequence", 0) if isinstance(current, dict) else 0
    if type(sequence) is not int or sequence < 0:
        raise SystemExit(f"invalid request sequence: {path}")
    sequence += 1
    temporary = control / "request.json.tmp"
    request = {
        "schema_version": "rq1-case-boundary-request-v1",
        "sequence": sequence,
        "desired_state": state,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if segment_duration is not None:
        request["segment_duration_seconds"] = int(segment_duration)
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return sequence

if action in {"pause", "resume"}:
    desired = "pause" if action == "pause" else "running"
    requested = set()
    while True:
        waiting = []
        for root in roots:
            if terminal(root)[0]:
                continue
            control = Path(root) / "control"
            if not control.is_dir():
                waiting.append(root)
                continue
            status = read(control / "status.json", {})
            if not isinstance(status, dict):
                status = {}
            state = status.get("state") if isinstance(status, dict) else None
            request = read(control / "request.json", {})
            request_state = request.get("desired_state") if isinstance(request, dict) else None
            if action == "pause" and state == "paused" and request_state == "pause":
                continue
            if action == "resume" and state == "running" and request_state == "running":
                continue
            if root not in requested:
                write_request(
                    root, desired,
                    int(duration_seconds)
                    if action == "resume" and duration_seconds else None,
                )
                requested.add(root)
                request = read(control / "request.json", {})
                request_state = (
                    request.get("desired_state") if isinstance(request, dict) else None
                )
            wanted = "paused" if action == "pause" else "running"
            if action == "resume" and state == "paused" \
                    and status.get("resume_rejection_sequence") == request.get("sequence"):
                raise SystemExit(
                    "resume rejected: "
                    + str(status.get("resume_rejection_reason") or "unknown reason")
                )
            if state != wanted or request_state != desired:
                waiting.append(root)
        if not waiting:
            break
        time.sleep(1)

for root in roots:
    control = Path(root) / "control"
    status = read(control / "status.json", {})
    if not isinstance(status, dict):
        status = {}
    wrapper = read(Path(root) / "wrapper-result.json", {})
    if not isinstance(wrapper, dict):
        wrapper = {}
    is_terminal, wrapper_status, launch_done = terminal(root)
    method = method_for(root)
    run_root = Path(root)
    execution_root = run_root / "executions"
    profile_root = run_root / "coverage-raw"
    raw_index = launch_dir / "raw-artifact-index.json"
    print(json.dumps({
        "run_prefix": run_prefix,
        "run_id": status.get("run_id", f"{run_prefix}-{method}"),
        "method": status.get("method_filter", method),
        "state": status.get("state", "terminal" if is_terminal else "starting"),
        "boundary_confirmed": status.get("state") == "paused",
        "wrapper_status": wrapper_status,
        "launcher_done": launch_done,
        "execution_complete": wrapper.get("execution_complete"),
        "raw_profile_root": {"path": str(profile_root), "exists": profile_root.exists()},
        "raw_artifact_index": {"path": str(raw_index), "exists": raw_index.exists()},
        "execution_artifact_root": {"path": str(execution_root), "exists": execution_root.exists()},
        "trace_root": {
            "path": str(execution_root),
            "exists": execution_root.exists(),
            "recorded_in": "**/target-run.json: targets[].trace_path",
        },
        "desired_state": status.get("desired_state"),
        "request_sequence": status.get("request_sequence"),
        "pause_reason": status.get("pause_reason"),
        "segment_index": status.get("segment_index"),
        "segment_duration_seconds": status.get("segment_duration_seconds"),
        "paused_seconds": status.get("paused_seconds"),
        "workers": status.get("workers", {}),
    }, ensure_ascii=False))
PY

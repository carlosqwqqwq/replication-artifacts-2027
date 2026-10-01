#!/usr/bin/env bash
set -Eeuo pipefail

comparison_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}
action=${1:-}
run_id=
duration_seconds=

usage() {
  cat <<'EOF'
用法：
  container/control-rvgen-catalog-72h.sh {pause|resume|status} --run-id ID [--duration-seconds N]

pause 在所有 target 与 reference worker 到达 case 边界后返回；runner 和容器继续存活。
resume 在同一 run、同一 runner 进程和队列游标上继续；可为下一段设置新的活跃秒数。
EOF
}

[[ -n "$action" ]] || { usage >&2; exit 2; }
shift
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --run-id) run_id=${2:?--run-id requires a value}; shift 2;;
    --duration-seconds) duration_seconds=${2:?--duration-seconds requires a value}; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$action" == pause || "$action" == resume || "$action" == status ]] || {
  echo 'action must be pause, resume, or status' >&2; exit 2;
}
[[ "$run_id" =~ ^[A-Za-z0-9_.-]{1,55}$ ]] || { echo 'invalid run-id' >&2; exit 2; }
[[ -z "$duration_seconds" || "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--duration-seconds must be a positive integer' >&2; exit 2;
}
if [[ "$action" != resume && -n "$duration_seconds" ]]; then
  echo '--duration-seconds only applies to resume' >&2; exit 2;
fi
[[ "$output_root" == /path/to/rq1-comparison/results ]] || {
  echo 'output-root must be exactly /path/to/rq1-comparison/results' >&2; exit 2;
}

run_root="$output_root/runs/$run_id"
container_name="rq1cmp-$run_id"
[[ -d "$run_root" ]] || { echo "unknown run: $run_id" >&2; exit 2; }
docker_state=missing
if docker container inspect "$container_name" >/dev/null 2>&1; then
  docker_state=$(docker inspect --format '{{.State.Status}}' "$container_name")
fi
if [[ "$action" != status && "$docker_state" != running ]]; then
  echo "container is not running (state=$docker_state); same-process resume is unavailable" >&2
  exit 1
fi

python3 - "$action" "$run_id" "$run_root" "$duration_seconds" "$docker_state" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

action, run_id, root_arg, duration_arg, docker_state = sys.argv[1:]
root = Path(root_arg)
control = root / "control"
request_path = control / "request.json"
status_path = control / "status.json"

def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}

def write_request(desired_state):
    control.mkdir(parents=True, exist_ok=True)
    current = read(request_path)
    sequence = current.get("sequence", 0)
    if type(sequence) is not int or sequence < 0:
        raise SystemExit(f"invalid control request sequence: {request_path}")
    request = {
        "schema_version": "rq1-case-boundary-request-v1",
        "sequence": sequence + 1,
        "desired_state": desired_state,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if desired_state == "running" and duration_arg:
        request["segment_duration_seconds"] = int(duration_arg)
    temporary = control / "request.json.tmp"
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, request_path)

if action in {"pause", "resume"}:
    wanted = "paused" if action == "pause" else "running"
    current = read(status_path)
    if current.get("state") != wanted:
        write_request("pause" if action == "pause" else "running")
    while True:
        current = read(status_path)
        state = current.get("state")
        if state == wanted:
            break
        if state in {"completed", "interrupted"}:
            raise SystemExit(f"run is terminal: state={state}")
        time.sleep(1)

status = read(status_path)
progress = read(root / "progress.json")
print(json.dumps({
    "run_id": run_id,
    "container": f"rq1cmp-{run_id}",
    "container_state": docker_state,
    "state": status.get("state", "starting"),
    "desired_state": status.get("desired_state"),
    "request_sequence": status.get("request_sequence"),
    "pause_reason": status.get("pause_reason"),
    "segment_index": status.get("segment_index"),
    "segment_duration_seconds": status.get("segment_duration_seconds"),
    "paused_seconds": status.get("paused_seconds"),
    "active_elapsed_seconds": status.get("active_elapsed_seconds"),
    "remaining_seconds": status.get("segment_remaining_seconds",
                                   progress.get("input_window_remaining_seconds")),
    "workers": status.get("workers", {}),
    "raw_artifact_index": str(root / "raw-artifact-index.json"),
}, ensure_ascii=False))
PY

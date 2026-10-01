#!/usr/bin/env bash
set -euo pipefail
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
catalog_root="${RQ1_CASE_CATALOG_ROOT:-/path/to/rq1-comparison/results/cases/rq1-case-library}"
action=${1:-}
shift || true
run_id=
duration=
usage() { echo 'control-run.sh {pause|resume|stop|status|import-checkpoint|watch-checkpoints} --run-id ID [--duration-seconds N] [--catalog-root DIR]'; }
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --run-id) run_id=${2:?}; shift 2 ;;
    --duration-seconds) duration=${2:?}; shift 2 ;;
    --catalog-root) catalog_root=${2:?}; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$action" == pause || "$action" == resume || "$action" == stop || "$action" == status ||
   "$action" == import-checkpoint || "$action" == watch-checkpoints ]] || { usage >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid run-id' >&2; exit 2; }
[[ -z "$duration" || "$duration" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid duration' >&2; exit 2; }
if [[ "$action" != resume && -n "$duration" ]]; then echo 'duration only applies to resume' >&2; exit 2; fi
[[ "$output_root" == /path/to/rq1-comparison/results ]] || { echo 'invalid output-root' >&2; exit 2; }
catalog_root=$(realpath -m -- "$catalog_root")
[[ "$catalog_root" == "$output_root"/cases/* ]] || {
  echo 'catalog-root must be below /path/to/rq1-comparison/results/cases' >&2; exit 2;
}
run_root="$output_root/runs/$run_id"
[[ -d "$run_root" ]] || { echo "unknown run-id: $run_id" >&2; exit 2; }

if [[ "$action" == import-checkpoint ]]; then
  exec python3 "$source_root/case_catalog.py" import-generation-checkpoint \
    --catalog-root "$catalog_root" --run-root "$run_root"
fi

if [[ "$action" == watch-checkpoints ]]; then
  last_attempt=
  while [[ -d "$run_root" ]]; do
    read -r state request_sequence pause_started wrapper_done < <(
      python3 - "$run_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}
status = read(root / "control" / "status.json")
print(status.get("state", "starting"), status.get("request_sequence", -1),
      status.get("pause_started_at_utc") or "-",
      str((root / "wrapper-result.json").is_file()).lower())
PY
    )
    [[ "$wrapper_done" == true ]] && break
    case "$state" in
      completed|failed|stopped) break;;
      paused)
        signature="$request_sequence:$pause_started"
        if [[ "$signature" != "$last_attempt" ]]; then
          if python3 "$source_root/case_catalog.py" import-generation-checkpoint \
              --catalog-root "$catalog_root" --run-root "$run_root"; then
            last_attempt=$signature
          else
            last_attempt=$signature
            echo "checkpoint import failed for $run_id; use control-run.sh import-checkpoint to retry" >&2
          fi
        fi
        ;;
    esac
    sleep 5
  done
  exit 0
fi

python3 - "$action" "$run_root" "$duration" <<'PY'
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
action, root_value, duration = sys.argv[1:]
root = Path(root_value)
control = root / "control"

def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}

if action in {"pause", "resume", "stop"}:
    manifest = read(root / "execution-manifest.json")
    if manifest.get("action") == "run":
        raise SystemExit(
            "the short diagnostic action 'run' does not support same-process pause/resume"
        )

def request(state):
    control.mkdir(parents=True, exist_ok=True)
    path = control / "request.json"
    old = read(path)
    sequence = old.get("sequence", 0)
    if type(sequence) is not int or sequence < 0:
        raise SystemExit(f"invalid sequence: {path}")
    payload = {
        "schema_version": "rq1-case-boundary-request-v1",
        "sequence": sequence + 1,
        "desired_state": state,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if action == "resume" and duration:
        payload["segment_duration_seconds"] = int(duration)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)

name = f"rq1cmp-{root.name}"

def container_state():
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", name],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "missing"

def effective_container_state():
    """Treat a stale Docker state as paused when cgroup v2 is frozen."""
    state = container_state()
    if state != "running":
        return state
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", name],
            capture_output=True, text=True, check=False,
        )
        container_id = result.stdout.strip()
        freeze = (
            Path("/sys/fs/cgroup/system.slice")
            / f"docker-{container_id}.scope" / "cgroup.freeze"
        )
        if container_id and freeze.read_text(encoding="ascii").strip() == "1":
            return "paused"
    except (OSError, UnicodeError):
        pass
    return state

def start_for_resume():
    state = effective_container_state()
    if state in {"exited", "created", "dead"}:
        result = subprocess.run(["docker", "start", name],
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"cannot resume container {name}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    elif state == "paused":
        result = subprocess.run(["docker", "unpause", name],
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"cannot unpause container {name}: {result.stderr.strip()}"
            )
    elif state in {"missing", "unavailable"}:
        raise SystemExit(f"cannot resume: container {name} is {state}")

if action in {"pause", "resume", "stop"}:
    desired, wanted = (
        ("pause", "paused") if action == "pause" else
        ("stop", "stopped") if action == "stop" else
        ("running", "running")
    )
    if action in {"resume", "stop"}:
        # A paused experiment is a stopped, retained Docker container. Write
        # the new request before starting it so the new supervisor sees the
        # same queue/cursors and does not re-enter the old pause boundary.
        if read(control / "status.json").get("state") not in {"completed", "failed"}:
            request(desired)
            start_for_resume()
    while True:
        status = read(control / "status.json")
        current = read(control / "request.json")
        if action == "resume" and status.get("state") == "paused" \
                and status.get("resume_rejection_sequence") == current.get("sequence"):
            raise SystemExit(
                "resume rejected: "
                + str(status.get("resume_rejection_reason") or "unknown reason")
            )
        if status.get("state") in {"completed", "failed"}:
            break
        if action == "pause" and status.get("state") == "stopped":
            break
        if status.get("state") == wanted and current.get("desired_state") == desired:
            break
        if current.get("desired_state") != desired:
            request(desired)
        time.sleep(0.5)
status = read(control / "status.json")
if action == "pause" and status.get("state") == "paused":
    # Freeze the same container after the producer acknowledged the case
    # boundary.  This preserves resident simulator sessions, Target cursors
    # and the supervisor's resume state; no Target drain is requested.
    subprocess.run(["docker", "pause", name],
                   capture_output=True, text=True, check=False)
try:
    result = subprocess.run(["docker", "inspect", "--format", "{{.State.Status}}", name],
                            capture_output=True, text=True, check=False)
    container_state = effective_container_state() if result.returncode == 0 else "missing"
except OSError:
    container_state = "unavailable"
print(json.dumps({
    "run_id": root.name,
    "state": status.get("state", "starting"),
    "desired_state": status.get("desired_state"),
    "pause_reason": status.get("pause_reason"),
    "paused_seconds": status.get("paused_seconds"),
    "workers": status.get("workers", {}),
    "container": name,
    "container_state": container_state,
    "raw_profile_root": str(root / "coverage-raw"),
    "execution_trace_root": str(root / "executions"),
}, ensure_ascii=False))
PY

if [[ "$action" == pause ]]; then
  run_action=$(python3 - "$run_root/execution-manifest.json" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, TypeError, ValueError):
    value = {}
print(value.get("action", ""))
PY
  )
  if [[ "$run_action" == framework-run ]]; then
    # Keep resident simulator sessions and Target cursors alive.  The
    # supervisor/producer resume path is awakened by docker unpause.
    docker pause "rq1cmp-$run_id" >/dev/null 2>&1 || true
  elif [[ "$run_action" == generate ]]; then
    python3 "$source_root/case_catalog.py" import-generation-checkpoint \
      --catalog-root "$catalog_root" --run-root "$run_root"
  fi
fi

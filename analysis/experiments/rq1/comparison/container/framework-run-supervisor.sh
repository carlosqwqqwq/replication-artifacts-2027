#!/usr/bin/env bash
set -Eeuo pipefail

# 一个 framework-run 只对应一个 Docker 容器。Target service 是本容器内的
# simulator-side sibling：每个 Target 一个独立进程、queue、cursor 和 resident
# session；它不是 producer 的 child，也不向 producer 返回 admission/完成信号。
# producer 到达时间边界后退出；Target service 完成当前 job 后关闭 session，supervisor
# 保留 cursor 等待 resume。只有 producer 正常完成并关闭 publication 时才进入完整 drain。
# 显式 pause/TERM 仍保留 cursor 供 resume。
run_root=
method_dir=
config=
method=
feedback_target=all
owner=framework
duration_seconds=
drain_timeout_seconds="${RQ1_DRAIN_TIMEOUT_SECONDS:-300}"
runner_command=()
target_pids=()
target_ids=()
drain_failure_reason=
target_services_stopped_at_boundary=false

while (($#)); do
  case "$1" in
    --run-root) run_root=$2; shift 2;;
    --method-dir) method_dir=$2; shift 2;;
    --config) config=$2; shift 2;;
    --method) method=$2; shift 2;;
    --feedback-target) feedback_target=$2; shift 2;;
    --owner) owner=$2; shift 2;;
    --duration-seconds) duration_seconds=$2; shift 2;;
    --) shift; runner_command=("$@"); break;;
    *) echo "unknown supervisor option: $1" >&2; exit 2;;
  esac
done

[[ -n "$run_root" && -n "$method_dir" && -n "$config" && -n "$method" \
  && ${#runner_command[@]} -gt 0 ]] || {
  echo 'run-root, method-dir, config, method and a producer command are required' >&2
  exit 2
}
if [[ -n "$duration_seconds" && ! "$duration_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo 'duration-seconds must be a positive integer' >&2
  exit 2
fi
[[ "$drain_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || drain_timeout_seconds=300

log_root="$run_root/logs"
mkdir -p "$log_root"
producer_pid=
duration_timer_pid=
closing=false
runtime_config="$method_dir/target-runtime-configs.json"
producer_start_time=

process_start_time() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

process_alive() {
  local pid=$1 state
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [[ "$state" != Z ]]
}

producer_alive() {
  local state current stat
  stat=$(awk '{print $3 " " $22}' "/proc/$producer_pid/stat" 2>/dev/null) || return 1
  read -r state current <<<"$stat"
  [[ "$state" != Z && -n "$producer_start_time" && "$current" == "$producer_start_time" ]]
}

stop_duration_timer() {
  if [[ "$duration_timer_pid" =~ ^[0-9]+$ ]]; then
    kill "$duration_timer_pid" 2>/dev/null || true
    wait "$duration_timer_pid" 2>/dev/null || true
  fi
  duration_timer_pid=
}

current_segment_duration() {
  python3 - "$run_root/control/request.json" "${duration_seconds:-0}" <<'PY'
import json
import sys
from pathlib import Path

try:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    request = {}
duration = request.get("segment_duration_seconds") if isinstance(request, dict) else None
if type(duration) is int and duration > 0:
    print(duration)
else:
    print(sys.argv[2])
PY
}

start_duration_timer() {
  [[ -n "$duration_seconds" ]] || return 0
  local segment_duration
  segment_duration=$(current_segment_duration)
  [[ "$segment_duration" =~ ^[1-9][0-9]*$ ]] || segment_duration=$duration_seconds
  (
    sleep "$segment_duration"
    kill -TERM "$$" 2>/dev/null || true
  ) &
  duration_timer_pid=$!
}

load_target_ids() {
  mapfile -t target_ids < <(python3 - "$runtime_config" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    value = {}
targets = value.get("targets") if isinstance(value, dict) else {}
if isinstance(targets, dict):
    for target_id in targets:
        if isinstance(target_id, str) and target_id.strip():
            print(target_id)
PY
  )
}

start_target_services() {
  [[ -s "$runtime_config" ]] || {
    echo "missing Target runtime config: $runtime_config" >&2
    exit 2
  }
  load_target_ids
  ((${#target_ids[@]} > 0)) || {
    echo "Target runtime config declares no Target services: $runtime_config" >&2
    exit 2
  }
  local target_id safe_target log_dir pid
  log_dir="$log_root/targets"
  mkdir -p "$log_dir"
  for target_id in "${target_ids[@]}"; do
    safe_target=$(printf '%s' "$target_id" | sed 's/[^A-Za-z0-9._-]/_/g')
    python3 /opt/rq1/comparison/container/resident_target_service.py \
      --method-dir "$method_dir" --target "$target_id" \
      --runtime-config "$runtime_config" \
      >"$log_dir/$safe_target.stdout.log" \
      2>"$log_dir/$safe_target.stderr.log" &
    pid=$!
    target_pids+=("$pid")
  done
}

stop_target_services() {
  local stop_signal=${1:-TERM}
  local pid alive deadline
  for pid in "${target_pids[@]}"; do
    kill -"$stop_signal" "$pid" 2>/dev/null || true
  done
  # A backend may be inside a native request while its resident worker is
  # closing coverage. Keep the bounded shutdown, but never let one Target
  # hold the whole container forever after the producer boundary.
  # Resident Renode closes dotnet-coverage with a bounded 60s IPC shutdown;
  # leave enough time for the worker and its child to serialize Cobertura.
  deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${target_pids[@]}"; do
      if process_alive "$pid"; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && break
    sleep 0.1
  done
  for pid in "${target_pids[@]}"; do
    if process_alive "$pid"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${target_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  target_pids=()
}

write_drain_status() {
  local state=$1
  local reason=${2:-}
  python3 - "$method_dir/target-queues/drain-status.json" "$run_root/drain-status.json" "$state" "$reason" "$runtime_config" "$method_dir" <<'PY'
import json
import os
import sys
import tempfile
import time
from pathlib import Path

method_path = Path(sys.argv[1])
root_path = Path(sys.argv[2])
status = sys.argv[3]
reason = sys.argv[4] or None
runtime_path = Path(sys.argv[5])
method_dir = Path(sys.argv[6])

def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}

def safe(value):
    return "".join(
        char if char.isalnum() or char in ".-_" else "_"
        for char in str(value)
    ) or "unknown"

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent), text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass

runtime = read(runtime_path)
declared = runtime.get("targets")
target_ids = list(declared) if isinstance(declared, dict) else []
targets = {}
for target_id in target_ids:
    queue_root = method_dir / "target-queues" / safe(target_id)
    queue_state = read(queue_root / "state.json")
    worker = read(queue_root / "worker-status.json")
    session = read(method_dir / "target-sessions" / f"{safe(target_id)}.json")
    published = queue_state.get("published_count")
    completed = worker.get("completed_count")
    published = published if type(published) is int and published >= 0 else 0
    completed = completed if type(completed) is int and completed >= 0 else 0
    worker_status = worker.get("status")
    session_status = session.get("worker_status")
    session_flushed = (
        worker_status in {"stopped", "finished", "drained"}
        and session_status in {"stopped", "finished", "drained"}
    )
    targets[str(target_id)] = {
        "published_count": published,
        "completed_count": completed,
        "pending_count": max(0, published - completed),
        "worker_status": worker_status,
        "error_count": worker.get("error_count", 0),
        "session_status": session_status,
        "session_flushed": session_flushed,
    }
producer_closed = (method_dir / "target-queues" / "producer-closed.json").is_file()
sessions_flushed = bool(targets) and all(
    value["session_flushed"] for value in targets.values()
)
updated = time.time()
atomic(method_path, {
    "schema_version": "rq1-target-queue-drain-v1",
    "status": status,
    "reason": reason,
    "updated_at_epoch": updated,
})
atomic(root_path, {
    "schema_version": "rq1-comparison-v2-drain-status-v1",
    "status": status,
    "producer_closed": producer_closed,
    "sessions_flushed": sessions_flushed,
    "targets": targets,
    "reason": reason,
    "updated_at_epoch": updated,
})
PY
}

mark_publication_closed() {
  python3 - "$method_dir/target-queues" "$runtime_config" <<'PY'
import json
import os
import sys
import tempfile
import time
from pathlib import Path

queue_root = Path(sys.argv[1])
try:
    runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    runtime = {}
targets = runtime.get("targets") if isinstance(runtime, dict) else {}
target_ids = list(targets) if isinstance(targets, dict) else []
published = {}
for target_id in target_ids:
    safe = "".join(
        char if char.isalnum() or char in ".-_" else "_"
        for char in str(target_id)
    ) or "unknown"
    try:
        value = json.loads(
            (queue_root / safe / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError):
        value = {}
    count = value.get("published_count") if isinstance(value, dict) else 0
    published[str(target_id)] = count if type(count) is int and count >= 0 else 0
payload = {
    "schema_version": "rq1-target-queue-producer-closed-v1",
    "target_ids": [str(value) for value in target_ids],
    "published_counts": published,
    "closed_at_epoch": time.time(),
}
path = queue_root / "producer-closed.json"
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=path.name + ".", dir=str(path.parent), text=True,
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

write_control_state() {
  local state=$1
  python3 - "$run_root/control/status.json" "$state" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    payload = {}
if not isinstance(payload, dict):
    payload = {}
payload["state"] = sys.argv[2]
payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
if sys.argv[2] == "stopped":
    payload["desired_state"] = "stop"
    payload["stop_reason"] = "control-stop"
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=path.name + ".", dir=str(path.parent), text=True,
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

target_drain_state() {
  python3 - "$runtime_config" "$method_dir/target-queues" <<'PY'
import json
import sys
from pathlib import Path

runtime_path = Path(sys.argv[1])
queue_root = Path(sys.argv[2])
try:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    runtime = {}
targets = runtime.get("targets") if isinstance(runtime, dict) else {}
target_ids = list(targets) if isinstance(targets, dict) else []
pending = 0
missing = []
errors = []
terminal = 0
for target_id in target_ids:
    safe = "".join(
        char if char.isalnum() or char in ".-_" else "_"
        for char in str(target_id)
    ) or "unknown"
    root = queue_root / safe
    def read(name):
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    state = read("state.json")
    worker = read("worker-status.json")
    published = state.get("published_count")
    completed = worker.get("completed_count")
    if not isinstance(published, int) or not isinstance(completed, int):
        missing.append(target_id)
        continue
    pending += max(0, published - completed)
    status = worker.get("status")
    if status in {"error", "missing-session-command", "fatal"} \
            or (isinstance(worker.get("error_count"), int) and worker.get("error_count", 0) > 0):
        errors.append(target_id)
    if status in {"error", "missing-session-command", "fatal", "stopped", "finished", "drained"} \
            or (status == "running" and published == completed):
        terminal += 1
if errors:
    # A failed Target is terminal for that Target only.  Let healthy Targets
    # finish and preserve their evidence for the offline analyzer.
    print(
        "partial" if terminal == len(target_ids)
        else "target-error"
    )
elif missing:
    print("pending")
elif pending == 0:
    print("drained")
else:
    print("pending")
PY
}

wait_for_target_drain() {
  local pid deadline dead_pid drain_state
  drain_failure_reason=
  deadline=$((SECONDS + drain_timeout_seconds))
  write_drain_status draining
  while true; do
    dead_pid=0
    for pid in "${target_pids[@]}"; do
      if ! process_alive "$pid"; then
        dead_pid=1
      fi
    done
    drain_state=$(target_drain_state)
    case "$drain_state" in
      drained)
        write_drain_status drained
        return 0
        ;;
      partial)
        drain_failure_reason=target-error
        write_drain_status partial
        return 2
        ;;
      target-error)
        # Keep healthy Target services draining while the failed Target is
        # recorded as a local error.
        ;;
    esac
    # A dead Target with no durable terminal status is still a coordinator
    # error; do not silently call it a completed queue.
    if (( dead_pid )) && [[ "$drain_state" != target-error ]]; then
      drain_failure_reason=target-worker-dead
      write_drain_status error
      return 1
    fi
    if (( SECONDS >= deadline )); then
      drain_failure_reason=target-drain-timeout
      write_drain_status partial "$drain_failure_reason"
      return 2
    fi
    sleep 1
  done
}

write_raw_seal() {
  # Online framework runs finish after Target raw evidence is flushed.  Source,
  # RV and mismatch aggregation is an explicit offline action and must not be
  # part of this close barrier.
  python3 - "$run_root" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
try:
    drain = json.loads((root / "drain-status.json").read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    drain = {}
targets = drain.get("targets") if isinstance(drain, dict) else {}
targets = targets if isinstance(targets, dict) else {}
def integer(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0
pending = sum(
    integer(item.get("pending_count", 0))
    for item in targets.values() if isinstance(item, dict)
)
errors = any(
    integer(item.get("error_count", 0)) > 0
    for item in targets.values() if isinstance(item, dict)
)
sealed = bool(
    drain.get("status") == "drained"
    and drain.get("producer_closed") is True
    and drain.get("sessions_flushed") is True
    and pending == 0
    and not errors
)
reasons = []
if drain.get("producer_closed") is not True:
    reasons.append("producer-not-closed")
if drain.get("status") != "drained":
    reasons.append("target-drain-incomplete")
if drain.get("sessions_flushed") is not True:
    reasons.append("target-session-not-flushed")
if pending:
    reasons.append("target-backlog")
if errors:
    reasons.append("target-error")
path = root / "derived" / "seal.json"
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": "rq1-comparison-v2-raw-seal-v1",
    "run_id": root.name,
    "sealed": sealed,
    "raw_only": True,
    "raw_sealed": sealed,
    "formal_ready": False,
    "reason": None if sealed else sorted(set(reasons)),
    "drain_status": drain.get("status"),
    "producer_closed": drain.get("producer_closed") is True,
    "sessions_flushed": drain.get("sessions_flushed") is True,
    "pending_count": pending,
    "target_error": errors,
    "coverage_status": "deferred",
    "coverage_finalized": False,
    "finalizer_status": "deferred",
}
fd, temporary_name = tempfile.mkstemp(
    prefix=path.name + ".", dir=str(path.parent), text=True,
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

request_boundary_pause() {
  local signal_name=$1
  python3 - "$run_root/control/request.json" "$signal_name" <<'PY'
import json
import os
import sys
import tempfile
import time
from pathlib import Path

path = Path(sys.argv[1])
try:
    current = json.loads(path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    current = {}
if not isinstance(current, dict):
    current = {}
sequence = current.get("sequence", 0)
if type(sequence) is not int or sequence < 0:
    sequence = 0
payload = {
    **current,
    "schema_version": "rq1-case-boundary-request-v1",
    "sequence": sequence + 1,
    "desired_state": "pause",
    "reason": "supervisor-signal:" + str(sys.argv[2]),
    "requested_at_epoch": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=path.name + ".", dir=str(path.parent), text=True,
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

boundary_is_paused() {
  python3 - "$run_root/control/status.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    payload = {}
raise SystemExit(
    0 if isinstance(payload, dict) and payload.get("state") == "paused" else 1
)
PY
}

finish_closed_run() {
  [[ "$closing" == true ]] && return 0
  closing=true
  local reason=${1:-producer-closed}
  local drain_code=0
  local drain_status=drained
  local drain_reason=$reason
  mark_publication_closed
  if wait_for_target_drain; then
    :
  else
    drain_code=$?
    if [[ "$drain_code" -eq 2 ]]; then
      drain_status=partial
      drain_reason=${drain_failure_reason:-$reason}
    else
      drain_status=error
      drain_reason=${drain_failure_reason:-$reason}
    fi
  fi
  stop_target_services
  write_drain_status "$drain_status" "$drain_reason"
  write_raw_seal
  [[ "$drain_code" -ne 0 ]] && return "$drain_code"
  return 0
}

forward_signal() {
  local signal_name=$1
  [[ "$closing" == true ]] && return 0
  stop_duration_timer
  request_boundary_pause "$signal_name" || true
  if [[ "$producer_pid" =~ ^[0-9]+$ ]]; then
    # A segment limit stops new cases. The producer owns the current case
    # checkpoint, which can take longer than the segment when reference or
    # simulator calls are still in flight; never kill it at an arbitrary time.
    while producer_alive && ! boundary_is_paused; do
      sleep 0.05
    done
  fi
  if boundary_is_paused; then
    # Let each Target finish its current queue item before the host wrapper
    # freezes the container. Pending queue items remain resumable.
    stop_target_services USR1 || true
    target_services_stopped_at_boundary=true
    return 0
  fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

start_target_services

while true; do
  start_duration_timer
  "${runner_command[@]}" \
    >>"$log_root/framework-producer.stdout.log" \
    2>>"$log_root/framework-producer.stderr.log" &
  producer_pid=$!
  producer_start_time=$(process_start_time "$producer_pid" || true)

  producer_status=0
  # Poll first and reap second.  Calling wait before checking the process
  # state can keep PID 1 in an uninterruptible-looking wait after the Python
  # producer has become a zombie; kill -0 also treats that zombie as alive.
  # Signals remain deliverable during the short sleep and the existing trap
  # still owns the case-boundary handoff.
  while producer_alive; do
    sleep 0.1
  done
  set +e
  wait "$producer_pid"
  producer_status=$?
  set -e
  printf '%s\n' "$producer_status" >"$run_root/framework-producer.exit-code"
  producer_pid=
  producer_start_time=
  stop_duration_timer

  state=$(python3 - "$run_root/control/status.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    value = {}
print(value.get("state", "failed") if isinstance(value, dict) else "failed")
PY
  )
  if [[ "$state" == completed && "$producer_status" -eq 0 ]]; then
    # A successful producer has closed publication. Normal completion owns
    # the drain barrier; it never feeds back into producer admission.
    if finish_closed_run "producer-complete"; then
      exit 0
    else
      drain_code=$?
      # A failed Target is a durable partial experiment; coordinator failure
      # is still reported after preserving all completed evidence.
      [[ "$drain_code" -eq 2 ]] && exit 0
      exit 1
    fi
  fi
  if [[ "$state" == failed ]] || [[ "$producer_status" -ne 0 && "$state" != paused ]]; then
    # A producer failure/TERM closes the already-published suffix as a
    # durable partial run. Only an acknowledged pause remains resumable;
    # leaving failed workers alive would lose the raw close marker.
    finish_closed_run "producer-failed" || true
    write_control_state stopped || true
    if [[ "$producer_status" =~ ^[1-9][0-9]*$ ]]; then
      exit "$producer_status"
    fi
    exit 1
  fi

  # A time-boxed producer has reached a durable case checkpoint. Target
  # services have finished their current items and wait for control-run.sh
  # resume; this is not a Target completion gate.
  while true; do
    desired=$(python3 - "$run_root/control/request.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    value = {}
print(value.get("desired_state", "pause") if isinstance(value, dict) else "pause")
PY
    )
    case "$desired" in
      running)
        if [[ "$target_services_stopped_at_boundary" == true ]]; then
          start_target_services
          target_services_stopped_at_boundary=false
        fi
        break
        ;;
      stop)
        # Stop closes publication, then drains already-published work. It does
        # not turn a stopped worker into a falsely completed queue.
        finish_closed_run "control-stop" || true
        write_control_state stopped
        exit 0
        ;;
    esac
    sleep 0.2
  done
done

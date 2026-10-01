#!/usr/bin/env bash
set -Eeuo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
entry="$source_root/container/run-linux.sh"
control_entry="$source_root/container/control-seven-long-methods.sh"
dependency_root="${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}"
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
case_catalog_root="${RQ1_CASE_CATALOG_ROOT:-/path/to/rq1-comparison/results/cases/rq1-case-library}"
run_prefix="${RQ1_RUN_PREFIX:-rq1-7x7-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
duration_seconds=259200
target_timeout_seconds=180
with_board=true
dry_run=false
resource_mode=seven-method-parallel
skip_doctor=false
# Keep the host wrapper alive until run-linux has completed its 900s Docker stop.
run_wrapper_stop_grace_seconds=905
declare -a children=() child_methods=() active_children=() remaining_active_pids=()
methods=(B-RVDV B-TORTURE B-CSMITH B-GEMI Fuzz4All Ours-RVGEN-Direct Ours-Program-Full)
targets=(T-QEMU T-LRSV-INT T-LRSV-TRANS T-UNICORN T-RENODE T-RAX T-RVVM)

usage() {
  cat <<'EOF'
用法：
  container/run-seven-long-methods.sh [选项]

七种方法各启动一个独立运行，每个运行覆盖配置中的全部七个 Target。
五个外部工具回放固定 case catalog；RVGEN-Direct 在线生成根 case，Program-Full 使用同一
B-RVDV catalog 快照中的有序源码作为根 case，并继续在线运行小模型、EMI 和 MCMC。
指定时长是一个活跃运行段；到期后在 case 边界暂停并保留 runner 和容器，之后可在同一 run 中继续。
运行期间只保留原始 profile、trace 和账本；覆盖率归并由离线流程完成。K1 reference 保持启用。
选项：
  --hours N                     本次运行段时长，默认 72 小时
  --duration-seconds N          以秒指定本次运行段时长
  --case-dir DIR                 外部回放与 Program-Full seed 使用的 case catalog
  --run-prefix PREFIX           唯一运行前缀，默认带 UTC 时间和进程号
  --target-timeout-seconds N    单次 Target 执行时限，默认 180
  --dependency-root DIR         依赖根
  --output-root DIR             结果根，只允许 /path/to/rq1-comparison/results
  --resource-mode MODE         seven-method-parallel
  --skip-doctor                 复用已完成的只读预检
  --with-board                  显式确认启用 K1（默认已启用）
  --dry-run                     打印七个命令，不运行 doctor 或启动容器
  --help                        显示帮助
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --hours)
      [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo '--hours must be positive' >&2; exit 2; }
      duration_seconds=$((${2} * 3600))
      shift 2
      ;;
    --duration-seconds) duration_seconds=${2:?--duration-seconds requires a value}; shift 2;;
    --case-dir|--case-catalog) case_catalog_root=$2; shift 2;;
    --run-prefix) run_prefix=$2; shift 2;;
    --target-timeout-seconds) target_timeout_seconds=$2; shift 2;;
    --dependency-root) dependency_root=$2; shift 2;;
    --output-root) output_root=$2; shift 2;;
    --resource-mode) resource_mode=$2; shift 2;;
    --skip-doctor) skip_doctor=true; shift;;
    --with-board) shift;;
    --without-board) echo 'K1 is required for the 7×7 long-run matrix' >&2; exit 2;;
    --dry-run) dry_run=true; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ "$target_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--target-timeout-seconds must be a positive integer' >&2; exit 2;
}
[[ "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--duration-seconds must be a positive integer' >&2; exit 2;
}
[[ "$run_prefix" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid run-prefix' >&2; exit 2; }
case "$resource_mode" in
  seven|seven-parallel|seven-method-parallel) resource_mode=seven-method-parallel;;
  *) echo 'the 7×7 launcher uses seven-method-parallel resources' >&2; exit 2;;
esac
[[ -x "$entry" ]] || { echo "missing executable entry: $entry" >&2; exit 2; }

canonical_config="$source_root/config/rq1-comparison-isolated-v1.json"
if [[ -n "${RQ1_CONFIG_FILE:-}" ]]; then
  [[ "$(realpath -- "$RQ1_CONFIG_FILE")" == "$canonical_config" ]] || {
    echo 'the 7x7 launcher uses the repository-pinned configuration only' >&2; exit 2;
  }
fi
output_root=$(realpath -m -- "$output_root")
canonical_output_root=/path/to/rq1-comparison/results
[[ "$output_root" == "$canonical_output_root" ]] || {
  echo "output-root must be exactly $canonical_output_root" >&2; exit 2;
}
hours=$(awk -v seconds="$duration_seconds" 'BEGIN {printf "%.6f", seconds / 3600}')
launch_dir="$output_root/launch-logs/$run_prefix"
completion_dir="$launch_dir/completed"
canonical_case_catalog=/path/to/rq1-comparison/results/cases
case_catalog_root=$(realpath -e -- "$case_catalog_root")
case "$case_catalog_root" in
  "$canonical_case_catalog"/*) ;;
  *) echo "case directory must be below $canonical_case_catalog" >&2; exit 2;;
esac
catalog_validation=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source_root" \
  python3 - "$case_catalog_root" "$canonical_config" "${methods[@]:0:5}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

catalog_root = Path(sys.argv[1]).resolve()
execution_config_path = Path(sys.argv[2]).resolve(strict=True)
methods = sys.argv[3:]
manifest_path = catalog_root / "catalog-manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit("case catalog manifest must be a regular file")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
artifact_root = catalog_root.parent.resolve(strict=True)
allowed_root = Path("/path/to/rq1-comparison/results/cases").resolve(strict=True)
config_path = None
for batch in manifest.get("batches", []):
    if not isinstance(batch, dict) or not isinstance(batch.get("index_path"), str):
        continue
    index_path = (catalog_root / batch["index_path"]).resolve()
    if not index_path.is_relative_to(catalog_root) or not index_path.is_file():
        continue
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    corpus_rel = index.get("corpus_root")
    if not isinstance(corpus_rel, str):
        continue
    batch_root = (artifact_root / corpus_rel).resolve()
    if not batch_root.is_relative_to(allowed_root):
        continue
    for method in methods:
        row = index.get("methods", {}).get(method)
        if not isinstance(row, dict):
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            continue
        candidate = batch_root / "runs" / run_id / "config.snapshot.json"
        if candidate.is_symlink() or not candidate.is_file():
            continue
        config_path = candidate.resolve()
        if not config_path.is_relative_to(batch_root):
            config_path = None
            continue
        break
    if config_path is not None:
        break
config = json.loads(execution_config_path.read_text(encoding="utf-8"))
from case_catalog import load_catalog_queues

queues, rows, source, _ = load_catalog_queues(catalog_root, config)
targets = [str(row["id"]) for row in config.get("targets", [])
           if isinstance(row, dict) and row.get("id")]
if len(targets) != 7:
    raise SystemExit("pinned config must contain seven targets")
for method in methods:
    rows.setdefault(method, {"method": method, "status": "partial", "candidate_count": 0})
manifest_path = catalog_root / "catalog-manifest.json"
manifest_bytes = manifest_path.read_bytes()
print(str(catalog_root))
print(source["catalog_revision"])
print(source["catalog_sha256"])
print(hashlib.sha256(manifest_bytes).hexdigest())
print(str(config_path) if config_path is not None else "")
print(hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path is not None else "")
PY
)
mapfile -t catalog_values <<<"$catalog_validation"
(( ${#catalog_values[@]} == 6 )) || {
  echo 'case catalog validation returned an incomplete result' >&2; exit 2;
}
case_catalog_root=${catalog_values[0]}
catalog_revision=${catalog_values[1]}
catalog_sha256=${catalog_values[2]}
catalog_manifest_sha256=${catalog_values[3]}
catalog_config_path=${catalog_values[4]}
catalog_config_sha256=${catalog_values[5]}

python3 - "$canonical_config" "${methods[@]}" "${targets[@]}" <<'PY'
import json
import sys

methods, targets = sys.argv[2:9], sys.argv[9:16]
config = json.load(open(sys.argv[1], encoding="utf-8"))
declared = {
    str(item.get("id"))
    for group in ("methods", "framework_methods")
    for item in config.get(group, []) if isinstance(item, dict)
}
actual_targets = [str(item.get("id")) for item in config.get("targets", [])
                  if isinstance(item, dict) and item.get("id")]
if declared != set(methods) or len(actual_targets) != 7 or set(actual_targets) != set(targets):
    raise SystemExit("current config does not match the fixed 7×7 method/Target matrix")
for method in config.get("framework_methods", []):
    if isinstance(method, dict) and method.get("id") in methods:
        if set(method.get("feedback_targets", [])) != set(targets):
            raise SystemExit(
                f"{method.get('id')}: feedback_targets must include all seven Targets"
            )
PY

config_sha256=$(sha256sum "$canonical_config" | awk '{print $1}')
resource_profile_json=$(python3 "$source_root/resource_policy.py" \
  --config "$canonical_config" --mode "$resource_mode" --format json)
if [[ "$dry_run" == true ]]; then
  printf '7×7 matrix; segment=%s seconds (%s hours); coverage=raw-only; K1 enabled\n' \
    "$duration_seconds" "$hours"
  printf 'case catalog=%s revision=%s sha256=%s\n' "$case_catalog_root" "$catalog_revision" "$catalog_sha256"
  printf 'simulator execution config=%s sha256=%s; catalog generation config=%s sha256=%s\n' \
    "$canonical_config" "$config_sha256" "$catalog_config_path" "$catalog_config_sha256"
  printf 'external K1 fallback: timeout/transport -> same-candidate T-QEMU result\n'
  printf 'resource profile: %s\n' "$resource_profile_json"
  for method in "${methods[@]}"; do
    run_id="${run_prefix}-${method}"
    if [[ "$method" == Ours-* ]]; then
      command=(env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$canonical_config" bash "$entry"
        --action framework-run --method "$method"
        --feedback-target all --run-id "$run_id")
      if [[ "$method" == Ours-Program-Full ]]; then
        command+=(--rvdv-seed-catalog "$case_catalog_root"
          --rvdv-seed-manifest "$launch_dir/catalog-manifest.snapshot.json")
      fi
    else
      command=(env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$canonical_config" RQ1_DEFER_FINALIZE=1 bash "$entry"
        --action execute-existing-queues --method "$method" --run-id "$run_id"
        --case-catalog "$case_catalog_root"
        --case-catalog-manifest "$launch_dir/catalog-manifest.snapshot.json")
    fi
    command+=(--duration-seconds "$duration_seconds"
      --target-timeout-seconds "$target_timeout_seconds"
      --resource-mode "$resource_mode"
      --dependency-root "$dependency_root" --output-root "$output_root")
    command+=(--with-board)
    printf '%q ' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

small_model_request_mode=${RQ1_SMALL_MODEL_REQUEST_MODE:-openai-compatible}
case "$small_model_request_mode" in
  openai-compatible)
    [[ -n "${RQ1_SMALL_MODEL_OPENAI_BASE_URL:-}" &&
       -n "${RQ1_SMALL_MODEL_OPENAI_MODEL:-}" &&
       -n "${RQ1_SMALL_MODEL_OPENAI_API_KEY:-}" ]] || {
      echo 'openai-compatible 7×7 runs require the small-model base URL, model, and API key in the environment' >&2
      exit 125
    }
    ;;
  ollama) ;;
  *)
    echo 'RQ1_SMALL_MODEL_REQUEST_MODE must be openai-compatible or ollama' >&2
    exit 125
    ;;
esac

output_parent=$(dirname -- "$output_root")
if [[ -e "$output_root" ]]; then
  [[ -d "$output_root" ]] && test -w "$output_root" || {
    echo "output root is not a writable directory: $output_root" >&2; exit 2;
  }
else
  [[ -d "$output_parent" ]] && test -w "$output_parent" || {
    echo "output parent is not writable: $output_parent" >&2; exit 2;
  }
fi
df -hT / /mnt/data >&2
docker system df >&2
mkdir -p "$output_root/runs" "$output_root/launch-logs"
test -w "$output_root/runs" || { echo "not writable: $output_root/runs" >&2; exit 2; }
[[ ! -e "$launch_dir" ]] || { echo "launch prefix already exists: $launch_dir" >&2; exit 2; }
doctor_id="${run_prefix}-doctor"
[[ ! -e "$output_root/runs/$doctor_id" ]] || { echo "doctor run already exists: $doctor_id" >&2; exit 2; }
for method in "${methods[@]}"; do
  run_id="${run_prefix}-${method}"
  [[ ! -e "$output_root/runs/$run_id" ]] || { echo "run already exists: $run_id" >&2; exit 2; }
  [[ ! -e "$output_root/source-snapshots/$run_id" ]] || {
    echo "source snapshot already exists: $run_id" >&2; exit 2;
  }
done

mkdir "$launch_dir" "$completion_dir"
catalog_manifest_snapshot="$launch_dir/catalog-manifest.snapshot.json"
cp -- "$case_catalog_root/catalog-manifest.json" "$catalog_manifest_snapshot"
chmod 0444 "$catalog_manifest_snapshot"
[[ "$(sha256sum -- "$catalog_manifest_snapshot" | awk '{print $1}')" == "$catalog_manifest_sha256" ]] || {
  echo 'case catalog manifest changed while pinning this launch' >&2
  exit 125
}
preflight_log="$launch_dir/preflight.log"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  df -hT / /mnt/data
  free -h
  uptime
  nproc
  docker ps --no-trunc
  docker system df
} >"$preflight_log" 2>&1
cat "$preflight_log" >&2

if [[ "$skip_doctor" != true ]]; then
  echo "[doctor] $doctor_id" >&2
  if ! bash "$entry" --action doctor --run-id "$doctor_id" \
    --dependency-root "$dependency_root" --output-root "$output_root" \
    >"$launch_dir/doctor.stdout.log" 2>"$launch_dir/doctor.stderr.log"; then
    echo "doctor failed; see $launch_dir/doctor.stderr.log" >&2
    exit 125
  fi
fi

tree_sha256() {
  tar --sort=name --mtime='@0' --mode='a+rwX' --owner=0 --group=0 --numeric-owner \
    -cf - -C "$1" . | sha256sum | awk '{print $1}'
}

source_commit=$(git -C /path/to/rq1-comparison/source/rq1-comparison rev-parse HEAD)
source_dirty=false
[[ -z "$(git -C /path/to/rq1-comparison/source/rq1-comparison status --porcelain=v1 --untracked-files=all)" ]] || source_dirty=true
working_tree_sha256=$(tree_sha256 "$source_root")

wait_pids_bounded() {
  local grace_seconds=$1 completed_pid='' pid watchdog
  shift
  local -a pending=() remaining=()
  for pid in "$@"; do
    [[ -n "$pid" ]] && pending+=("$pid")
  done
  ((${#pending[@]})) || return 0
  sleep "$grace_seconds" &
  watchdog=$!
  while ((${#pending[@]})); do
    completed_pid=
    wait -n -p completed_pid "${pending[@]}" "$watchdog" 2>/dev/null || true
    if [[ -z "$completed_pid" || "$completed_pid" == "$watchdog" ]]; then
      for pid in "${pending[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
      kill -TERM "$watchdog" 2>/dev/null || true
      wait "$watchdog" 2>/dev/null || true
      return 124
    fi
    remaining=()
    for pid in "${pending[@]}"; do
      [[ "$pid" == "$completed_pid" ]] || remaining+=("$pid")
    done
    pending=("${remaining[@]}")
  done
  kill -TERM "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
}

stop_children() {
  local pid
  # Each run-linux wrapper owns its container and records the stop provenance.
  # Signal the wrapper only; stopping Docker here races its graceful-stop trap.
  for pid in "${active_children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
stop_method_child() {
  local method=$1 index
  for index in "${!child_methods[@]}"; do
    [[ "${child_methods[$index]}" == "$method" ]] || continue
    kill -TERM "${children[$index]}" 2>/dev/null || true
    return 0
  done
  return 1
}
pause_requested=false
launch_complete=false
pending_signal=
handle_signal() {
  local signal_name=${1:-UNKNOWN}
  if [[ "$launch_complete" != true ]]; then
    pending_signal=$signal_name
    echo "[$signal_name] will request pause after all seven runners launch" >&2
    return 0
  fi
  [[ "$pause_requested" == false ]] || return 0
  pause_requested=true
  printf '{"event":"supervisor-pause-request","received_at_utc":"%s","signal":"%s","supervisor_pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$signal_name" "$BASHPID" \
    >>"$launch_dir/supervisor-events.jsonl"
  echo "[$signal_name] requesting a case-boundary pause for this 7×7 run" >&2
  if [[ -x "$control_entry" ]]; then
    setsid bash "$control_entry" pause --run-prefix "$run_prefix" >&2 || \
      echo "pause request was not fully acknowledged; check $launch_dir/supervisor-events.jsonl" >&2
  else
    echo "missing control entry: $control_entry" >&2
  fi
  echo "7×7 runners remain alive; use control-seven-long-methods.sh resume --run-prefix $run_prefix to continue" >&2
  pause_requested=false
}
trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM

cleanup_abnormal_exit() {
  local code=$?
  if ((code != 0)); then
    stop_children
    wait_pids_bounded "$run_wrapper_stop_grace_seconds" "${active_children[@]}" || true
  fi
}
trap cleanup_abnormal_exit EXIT

record_method_failure() {
  local method=$1 code=$2 reason=$3
  local failure_file="$completion_dir/$method.failure.tsv" temporary_file
  if [[ ! -e "$failure_file" ]]; then
    temporary_file="$failure_file.tmp.$$"
    if printf '%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$code" "$reason" \
        >"$temporary_file" && mv -- "$temporary_file" "$failure_file"; then
      printf '{"event":"method-failed","recorded_at_utc":"%s","method":"%s","exit_code":%s,"reason":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$code" "$reason" \
        >>"$launch_dir/supervisor-events.jsonl" || \
        echo "could not append method-failed event for $method" >&2
    else
      rm -f -- "$temporary_file" || true
      echo "could not persist method failure for $method" >&2
    fi
  fi
  touch "$completion_dir/$method.done" || \
    echo "could not mark $method complete" >&2
}

for method in "${methods[@]}"; do
  run_id="${run_prefix}-${method}"
  if [[ "$method" == Ours-* ]]; then
    launch_command=(env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$canonical_config" bash "$entry"
      --action framework-run --method "$method"
      --feedback-target all --run-id "$run_id")
    if [[ "$method" == Ours-Program-Full ]]; then
      launch_command+=(--rvdv-seed-catalog "$case_catalog_root"
        --rvdv-seed-manifest "$catalog_manifest_snapshot")
    fi
  else
    launch_command=(env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$canonical_config" RQ1_DEFER_FINALIZE=1 bash "$entry"
      --action execute-existing-queues --method "$method" --run-id "$run_id"
      --case-catalog "$case_catalog_root"
      --case-catalog-manifest "$catalog_manifest_snapshot")
  fi
  launch_command+=(--duration-seconds "$duration_seconds"
    --target-timeout-seconds "$target_timeout_seconds"
    --resource-mode "$resource_mode"
    --dependency-root "$dependency_root" --output-root "$output_root")
  launch_command+=(--with-board)
  echo "[start] $method -> $run_id" >&2
  setsid "${launch_command[@]}" >"$launch_dir/$method.stdout.log" \
    2>"$launch_dir/$method.stderr.log" &
  children+=("$!")
  active_children+=("$!")
  child_methods+=("$method")
done
launch_complete=true
if [[ -n "$pending_signal" ]]; then
  handle_signal "$pending_signal"
fi

manifest_deadline=$((SECONDS + 300))
manifests_resolved=false
while ((SECONDS < manifest_deadline)); do
  resolved_count=0
  for index in "${!children[@]}"; do
    method=${child_methods[$index]}
    run_id="${run_prefix}-${method}"
    if [[ -f "$output_root/runs/$run_id/execution-manifest.json" ]]; then
      resolved_count=$((resolved_count + 1))
      continue
    fi
    if [[ -e "$completion_dir/$method.failure.tsv" ]]; then
      resolved_count=$((resolved_count + 1))
      continue
    fi
    process_state=$(ps -o stat= -p "${children[$index]}" 2>/dev/null | tr -d ' ' || true)
    if [[ -z "$process_state" || "$process_state" == Z* ]]; then
      echo "[$method] exited before writing execution-manifest.json; continuing other methods" >&2
      record_method_failure "$method" 125 exited-before-manifest
      resolved_count=$((resolved_count + 1))
    fi
  done
  if ((resolved_count == ${#methods[@]})); then
    manifests_resolved=true
    break
  fi
  sleep 1
done

if [[ "$manifests_resolved" != true ]]; then
  for index in "${!children[@]}"; do
    method=${child_methods[$index]}
    run_id="${run_prefix}-${method}"
    [[ -f "$output_root/runs/$run_id/execution-manifest.json" ]] && continue
    [[ -e "$completion_dir/$method.failure.tsv" ]] && continue
    echo "[$method] did not write execution-manifest.json within five minutes; stopping only this method" >&2
    record_method_failure "$method" 125 manifest-timeout
    stop_method_child "$method" || true
  done
fi

snapshot_index="$launch_dir/snapshots.tsv"
: >"$snapshot_index"
for method in "${methods[@]}"; do
  if [[ -e "$completion_dir/$method.failure.tsv" ]]; then
    continue
  fi
  run_id="${run_prefix}-${method}"
  expected_config_sha256=$config_sha256
  action=execute-existing-queues
  [[ "$method" == Ours-* ]] && action=framework-run
  manifest="$output_root/runs/$run_id/execution-manifest.json"
  if ! snapshot=$(python3 - "$manifest" "$run_id" "$method" "$action" "$duration_seconds" \
    "$source_commit" "$source_dirty" "$expected_config_sha256" "$with_board" \
    "${targets[*]}" "$resource_profile_json" "$target_timeout_seconds" <<'PY'
import json
import sys

path, run_id, method, action, duration = sys.argv[1:6]
commit, dirty, config_sha, board, targets = sys.argv[6:11]
expected_resources = json.loads(sys.argv[11]).get("limits", {})
target_timeout = int(sys.argv[12])
manifest = json.load(open(path, encoding="utf-8"))
if manifest.get("run_id") != run_id or manifest.get("method_filter") != method:
    raise SystemExit(f"{method}: run identity mismatch")
if manifest.get("action") != action or manifest.get("duration_seconds") != int(duration):
    raise SystemExit(f"{method}: action or duration mismatch")
if manifest.get("coverage_enabled") is not True:
    raise SystemExit(f"{method}: coverage is not enabled")
if manifest.get("source_commit") != commit or manifest.get("source_dirty") is not (dirty == "true"):
    raise SystemExit(f"{method}: source Git identity mismatch")
if manifest.get("config_sha256") != config_sha:
    raise SystemExit(f"{method}: config digest mismatch")
if manifest.get("board_requested") is not (board == "true"):
    raise SystemExit(f"{method}: K1 reference setting mismatch")
profile = manifest.get("resource_profile", {})
resources = manifest.get("resources", {})
if profile.get("mode") != "seven-method-parallel" or profile.get("limits") != expected_resources:
    raise SystemExit(f"{method}: resource profile mismatch")
if any(resources.get(name) != value for name, value in expected_resources.items()):
    raise SystemExit(f"{method}: enforced resource limits mismatch")
deadlines = manifest.get("deadlines", {})
if deadlines.get("target_seconds") != target_timeout or any(
    value != target_timeout for value in deadlines.get("target_seconds_by_target", {}).values()
):
    raise SystemExit(f"{method}: target timeout mismatch")
snapshot = manifest.get("source_snapshot_host_root")
if not isinstance(snapshot, str) or not snapshot:
    raise SystemExit(f"{method}: host source snapshot path is missing")
print(snapshot)
PY
  ); then
    echo "[$method] execution manifest verification failed; continuing other methods" >&2
    record_method_failure "$method" 125 invalid-execution-manifest
    stop_method_child "$method" || true
    continue
  fi
  if [[ "$snapshot" != "$output_root/source-snapshots/$run_id" ]]; then
    echo "[$method] source snapshot path is outside its run-owned directory; continuing other methods" >&2
    record_method_failure "$method" 125 invalid-snapshot-path
    stop_method_child "$method" || true
    continue
  fi
  if ! snapshot_sha256=$(tree_sha256 "$snapshot"); then
    echo "[$method] could not hash its source snapshot; continuing other methods" >&2
    record_method_failure "$method" 125 snapshot-hash-failed
    stop_method_child "$method" || true
    continue
  fi
  if [[ "$snapshot_sha256" != "$working_tree_sha256" ]]; then
    echo "[$method] source snapshot differs from the pre-launch code tree; continuing other methods" >&2
    record_method_failure "$method" 125 snapshot-digest-mismatch
    stop_method_child "$method" || true
    continue
  fi
  snapshot=$(realpath -- "$snapshot")
  if ! chmod -R a-w -- "$snapshot"; then
    echo "[$method] could not mark its source snapshot read-only; continuing other methods" >&2
    record_method_failure "$method" 125 snapshot-readonly-failed
    stop_method_child "$method" || true
    continue
  fi
  printf '%s\t%s\t%s\n' "$method" "$snapshot" "$snapshot_sha256" >>"$snapshot_index"
done

python3 - "$launch_dir/launch-manifest.json" "$snapshot_index" \
  "$run_prefix" "$hours" "$duration_seconds" "$target_timeout_seconds" \
  "$with_board" "$source_commit" "$source_dirty" "$working_tree_sha256" \
  "$config_sha256" "$catalog_config_path" "$catalog_config_sha256" \
  "$resource_profile_json" "$case_catalog_root" \
  "$catalog_revision" "$catalog_sha256" "$catalog_manifest_sha256" \
  "${methods[@]}" "${targets[@]}" <<'PY'
import json
import sys
from pathlib import Path

output, index, prefix, hours, duration, timeout, board = sys.argv[1:8]
commit, dirty, tree_sha, config_sha, catalog_config_path, catalog_config_sha = sys.argv[8:14]
resource = json.loads(sys.argv[14])
catalog_root, catalog_revision, catalog_sha, catalog_manifest_sha = sys.argv[15:19]
methods, targets = sys.argv[19:26], sys.argv[26:33]
runs = []
snapshots = {}
for line in Path(index).read_text(encoding="utf-8").splitlines():
    method, snapshot, digest = line.split("\t")
    snapshots[method] = (snapshot, digest)
for method in methods:
    run_id = f"{prefix}-{method}"
    if method not in snapshots:
        failure_path = Path(index).parent / "completed" / f"{method}.failure.tsv"
        try:
            recorded_at, failure_method, exit_code, reason = failure_path.read_text(
                encoding="utf-8"
            ).rstrip("\n").split("\t", 3)
            if failure_method != method:
                raise ValueError("failure record method mismatch")
            failure = {
                "recorded_at_utc": recorded_at,
                "exit_code": int(exit_code),
                "reason": reason,
            }
        except (OSError, ValueError):
            failure = {"reason": "snapshot-not-verified"}
        runs.append({
            "run_id": run_id,
            "method": method,
            "status": "failed",
            "failure": failure,
        })
        continue
    snapshot, digest = snapshots[method]
    manifest_path = Path(snapshot).parents[1] / "runs" / run_id / "execution-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs.append({
        "run_id": run_id,
        "method": method,
        "status": "snapshot-verified",
        "action": manifest.get("action"),
        "config_sha256": manifest.get("config_sha256"),
        "source_snapshot_host_root": snapshot,
        "source_snapshot_sha256": digest,
        "resource_limits": manifest.get("resources"),
    })
payload = {
    "schema_version": "rq1-long-run-launch-v1",
    "run_prefix": prefix,
    "hours_per_segment": float(hours),
    "duration_seconds_per_segment": int(duration),
    "target_timeout_seconds": int(timeout),
    "board_requested": board == "true",
    "ours_reference_fallback": {
        "preferred_backend": "native-rv64",
        "fallback_backend": "qemu-riscv64",
        "on": ["k1-timeout", "k1-transport-unavailable"],
        "selection_scope": "whole-campaign",
        "selection_point": "first-reference-request",
    },
    "external_board_reference_fallback": {
        "preferred_backend": "native-rv64",
        "fallback_backend": "qemu-riscv64",
        "fallback_source": "same-candidate-T-QEMU-result",
        "on": ["k1-timeout", "k1-transport-unavailable"],
        "selection_scope": "candidate",
        "shared_target_cell": "T-QEMU",
    },
    "methods": methods,
    "targets": targets,
    "matrix_cells": len(methods) * len(targets),
    "verified_matrix_cells": len(snapshots) * len(targets),
    "launch_status": "ready" if len(snapshots) == len(methods) else "partial",
    "source_commit": commit,
    "source_dirty": dirty == "true",
    "working_tree_sha256": tree_sha,
    "source_tree_digest_algorithm": "sha256 of sorted tar stream with normalized mtime, owner/group, and write bits",
    "config_sha256": config_sha,
    "external_replay_config_sha256": catalog_config_sha,
    "case_catalog": {
      "root": catalog_root,
      "revision": catalog_revision,
      "catalog_sha256": catalog_sha,
      "config_snapshot": catalog_config_path,
      "config_sha256": catalog_config_sha,
      "pinned_manifest": f"launch-logs/{prefix}/catalog-manifest.snapshot.json",
        "pinned_manifest_file_sha256": catalog_manifest_sha,
        "external_methods": methods[:5],
    },
    "resource_mode": "seven-method-parallel",
    "resource_profile": resource,
    "snapshots_read_only_on_host": True,
    "coverage_collection": "raw-artifacts-only",
    "coverage_aggregation": "offline-post-processing",
    "raw_artifact_index": f"launch-logs/{prefix}/raw-artifact-index.json",
    "runs": runs,
}
path = Path(output)
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(
    f"verified {payload['verified_matrix_cells']}/{payload['matrix_cells']} "
    f"method×Target cells; source SHA256 {tree_sha}"
)
PY

write_raw_artifact_index() {
  python3 - "$launch_dir/raw-artifact-index.json" "$output_root" \
    "$run_prefix" "${methods[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

index_path, output_root, prefix, *methods = sys.argv[1:]
output_root = Path(output_root).resolve()

def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}

def ref(path):
    path = Path(path)
    return {"path": path.relative_to(output_root).as_posix(), "exists": path.exists()}

runs = []
for method in methods:
    run_id = f"{prefix}-{method}"
    root = output_root / "runs" / run_id
    control = read(root / "control" / "status.json")
    wrapper = read(root / "wrapper-result.json")
    raw_paths = {
        "executions": root / "executions",
        "coverage_raw": root / "coverage-raw",
        "artifact_features": root / "artifact-features",
        "ledger": root / "ledger",
        "trace_references": root / "executions",
        "wrapper_logs": root / "wrapper-logs",
    }
    if method.startswith("Ours-"):
        raw_paths.update({
            "framework_cases": root / "framework" / method,
            "framework_target_replays": root / "framework" / method / "lanes",
            "framework_experiments": root / "framework-experiments",
            "framework_coverage_configs": root / "framework-coverage-configs" / method,
            "simulator_traces": root / "traces",
        })
    runs.append({
        "method": method,
        "run_id": run_id,
        "runner_state": control.get("state", "starting"),
        "wrapper_status": wrapper.get("status"),
        "wrapper_exit_code": wrapper.get("exit_code"),
        "container_oom_killed": wrapper.get("container_oom_killed"),
        "next_case_cursors": control.get("workers", {}),
        "execution_manifest": ref(root / "execution-manifest.json"),
        "run_result": ref(root / "run-result.json"),
        "wrapper_result": ref(root / "wrapper-result.json"),
        "container_inspect": ref(root / "container-inspect.json"),
        "failure_record": ref(
            Path(index_path).parent / "completed" / f"{method}.failure.tsv"
        ),
        "control_request": ref(root / "control" / "request.json"),
        "control_status": ref(root / "control" / "status.json"),
        "ledger_events": ref(root / "ledger" / "events.jsonl"),
        "partial_ledger_events": ref(root / "ledger" / "events.partial.jsonl"),
        "coverage_registry": ref(root / "coverage-registry.json"),
        "raw_artifact_roots": {name: ref(path) for name, path in raw_paths.items()},
        "container_name": f"rq1cmp-{run_id}",
    })
payload = {
    "schema_version": "rq1-7x7-raw-artifact-index-v1",
    "run_prefix": prefix,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "coverage_processing": "raw-artifacts-only; aggregate offline",
    "case_catalog_manifest": ref(
        Path(index_path).parent / "catalog-manifest.snapshot.json"
    ),
    "runs": runs,
}
path = Path(index_path)
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
os.replace(temporary, path)
PY
}

write_raw_artifact_index

running_pids=("${children[@]}")
status=0
while ((${#running_pids[@]})); do
  unset finished_pid
  if wait -n -p finished_pid "${running_pids[@]}"; then code=0; else code=$?; fi
  [[ -n "${finished_pid:-}" ]] || continue
  remaining_active_pids=()
  for pid in "${active_children[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || remaining_active_pids+=("$pid")
  done
  active_children=("${remaining_active_pids[@]}")
  for index in "${!children[@]}"; do
    if [[ "${children[$index]}" == "$finished_pid" ]]; then
      method=${child_methods[$index]}
      touch "$completion_dir/$method.done" || \
        echo "could not mark $method complete" >&2
      if ((code == 0)) && [[ ! -e "$completion_dir/$method.failure.tsv" ]]; then
        echo "[done] $method" >&2
      else
        echo "[failed] $method wrapper-exit=$code; continuing the remaining matrix runs" >&2
        if ((code != 0)); then
          failure_reason=run-wrapper-exit
          wrapper_result="$output_root/runs/${run_prefix}-${method}/wrapper-result.json"
          if [[ -s "$wrapper_result" ]] && python3 - "$wrapper_result" <<'PY'
import json
import sys

try:
    result = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, TypeError, ValueError):
    result = {}
raise SystemExit(0 if isinstance(result, dict) and (
    result.get("container_oom_killed") is True
    or result.get("stop_provenance") == "resource-limit"
) else 1)
PY
          then
            failure_reason=container-oom-killed
            echo "[$method] container OOM recorded; other method containers continue" >&2
          fi
          record_method_failure "$method" "$code" "$failure_reason"
        fi
        status=1
      fi
      break
    fi
  done
  remaining_pids=()
  for pid in "${running_pids[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || remaining_pids+=("$pid")
  done
  running_pids=("${remaining_pids[@]}")
  write_raw_artifact_index
done

for method in "${methods[@]}"; do
  run_id="$run_prefix-$method"
  run_root="$output_root/runs/$run_id"
  result_path="runs/$run_id/run-result.json"
  result_status=present
  [[ -s "$run_root/run-result.json" ]] || result_status=missing
  if [[ "$result_status" == present ]]; then
    echo "[run-result] $method: $output_root/$result_path" >&2
  else
    echo "[missing-run-result] $method: $output_root/$result_path" >&2
    status=1
  fi
done
write_raw_artifact_index
trap - INT TERM EXIT
echo "launch manifest: $launch_dir/launch-manifest.json" >&2
echo "raw artifact index: $launch_dir/raw-artifact-index.json" >&2
exit "$status"

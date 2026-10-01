#!/usr/bin/env bash
set -Eeuo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
entry="$source_root/container/run-linux.sh"
control_entry="$source_root/container/control-run.sh"
config="${RQ1_CONFIG_FILE:-$source_root/config/rq1-comparison-isolated-v1.json}"
dependency_root="${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}"
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
case_catalog_root="${RQ1_CASE_CATALOG_ROOT:-/path/to/rq1-comparison/results/cases/rq1-case-library}"
model_env_file="${RQ1_MODEL_ENV_FILE:-/path/to/user/.config/rq1/wangyang25-qwen38.env}"
run_prefix="${RQ1_RUN_PREFIX:-rq1-five-tool-generation-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
resource_mode=seven-method-parallel
duration_seconds=21600
dry_run=false
continue_from=
methods=(B-RVDV B-TORTURE B-CSMITH B-GEMI Fuzz4All)
children=()
declare -A child_pid=()
declare -A cont_parent_run=()
declare -A cont_parent_batch=()
declare -A cont_parent_source_sha=()
declare -A cont_start_index=()
declare -A cont_start_batch=()
declare -A cont_attempt_offset=()
declare -A cont_candidate_offset=()
declare -A cont_fuzz_seed_source=()
declare -A cont_fuzz_parent_candidate=()
watcher_pids=()
interrupted_signal=
pause_dispatched=false

usage() {
  cat <<'EOF'
用法：
  container/run-five-tool-generation.sh [选项]

并行启动五个外部工具，各自先运行六小时；到段尾在 case 边界暂停，保留同一 runner 和容器，等待后续继续；不执行模拟器。
选项：
  --hours N                 生成运行段时长（小时）
  --run-prefix PREFIX       唯一批次前缀
  --dependency-root DIR     依赖根
  --output-root DIR         结果根，只允许 /path/to/rq1-comparison/results
  --continue-from BATCH     从 case catalog 中已有批次续接；默认选择每个工具的最新批次
  --dry-run                 展开命令，不检查 Docker 或启动容器
  --help                    显示帮助
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --hours)
      [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo 'hours must be a positive integer' >&2; exit 2; }
      duration_seconds=$((10#${2} * 3600))
      shift 2
      ;;
    --run-prefix) run_prefix=$2; shift 2;;
    --dependency-root) dependency_root=$2; shift 2;;
    --output-root) output_root=$2; shift 2;;
    --continue-from) continue_from=$2; shift 2;;
    --dry-run) dry_run=true; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ "$run_prefix" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid run-prefix' >&2; exit 2; }
[[ -x "$entry" ]] || { echo "missing executable entry: $entry" >&2; exit 2; }
output_root=$(realpath -m -- "$output_root")
canonical_output_root=/path/to/rq1-comparison/results
[[ "$output_root" == "$canonical_output_root" ]] || {
  echo "output-root must be exactly $canonical_output_root" >&2; exit 2;
}
case_catalog_root=$(realpath -m -- "$case_catalog_root")
[[ "$case_catalog_root" == "$output_root"/cases/* ]] || {
  echo 'case catalog must be below /path/to/rq1-comparison/results/cases' >&2; exit 2;
}
config=$(realpath -- "$config")
[[ -f "$config" ]] || { echo "missing generation config: $config" >&2; exit 2; }

python3 - "$config" "${methods[@]}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
methods = sys.argv[2:]
declared = [str(item.get("id")) for item in config.get("methods", [])
            if isinstance(item, dict)]
targets = [str(item.get("id")) for item in config.get("targets", [])
           if isinstance(item, dict) and item.get("id")]
if declared != methods or len(targets) != 7:
    raise SystemExit("pinned config must declare the five external methods and seven Targets")
PY

if printf '%s\n' "${methods[@]}" | grep -qx Fuzz4All; then
  [[ -f "$model_env_file" && ! -L "$model_env_file" && -r "$model_env_file" ]] || {
    echo "Fuzz4All model environment file is missing or unreadable: $model_env_file" >&2
    exit 2
  }
  model_env_mode=$(stat -c '%a' "$model_env_file")
  [[ "$model_env_mode" == 600 ]] || {
    echo "Fuzz4All model environment file must have mode 600: $model_env_file" >&2
    exit 2
  }
  set -a
  # shellcheck disable=SC1090
  source "$model_env_file"
  set +a
  [[ "${RQ1_SMALL_MODEL_OPENAI_BASE_URL:-}" =~ ^https?://[^[:space:]]+$ ]] || {
    echo 'RQ1_SMALL_MODEL_OPENAI_BASE_URL must be an HTTP URL' >&2
    exit 2
  }
  [[ -n "${RQ1_SMALL_MODEL_OPENAI_MODEL:-}" && -n "${RQ1_SMALL_MODEL_OPENAI_API_KEY:-}" ]] || {
    echo 'Fuzz4All OpenAI-compatible model name and API key are required' >&2
    exit 2
  }
fi

continuation_plan_json=
if [[ -n "$continue_from" ]]; then
  [[ -f "$case_catalog_root/catalog-manifest.json" ]] || {
    echo "continuation catalog manifest is missing: $case_catalog_root/catalog-manifest.json" >&2
    exit 2
  }
  continuation_plan_json=$(python3 "$source_root/generation_continuation.py" \
    --catalog-root "$case_catalog_root" --anchor-batch "$continue_from")
  while IFS=$'\t' read -r method parent_run parent_batch parent_source start_index start_batch attempt_offset candidate_offset fuzz_seed fuzz_parent; do
    cont_parent_run["$method"]=$parent_run
    cont_parent_batch["$method"]=$parent_batch
    cont_parent_source_sha["$method"]=$parent_source
    cont_start_index["$method"]=$start_index
    cont_start_batch["$method"]=$start_batch
    cont_attempt_offset["$method"]=$attempt_offset
    cont_candidate_offset["$method"]=$candidate_offset
    cont_fuzz_seed_source["$method"]=$fuzz_seed
    cont_fuzz_parent_candidate["$method"]=$fuzz_parent
  done < <(python3 - "$continuation_plan_json" <<'PY'
import json
import sys
plan = json.loads(sys.argv[1])
numeric = {"start_index", "start_batch", "attempt_offset",
           "candidate_index_offset", "fuzz4all_parent_candidate_index"}
for method, row in plan["methods"].items():
    keys = (
        "method", "parent_run_id", "parent_batch_id", "parent_source_sha256",
        "start_index", "start_batch", "attempt_offset", "candidate_index_offset",
        "fuzz4all_seed_source", "fuzz4all_parent_candidate_index",
    )
    values = []
    for key in keys:
        value = row.get(key)
        values.append(str(value) if value is not None else ("0" if key in numeric else "-"))
    print("\t".join(values))
PY
  )
fi

continuation_env=()
build_continuation_env() {
  local method=$1
  continuation_env=()
  [[ -n "$continue_from" ]] || return 0
  local start_index= start_batch= attempt_offset=
  case "$method" in
    B-RVDV) start_batch=${cont_start_batch[$method]:-0};;
    B-TORTURE|B-CSMITH) start_index=${cont_start_index[$method]:-0};;
    B-GEMI) attempt_offset=${cont_attempt_offset[$method]:-0};;
  esac
  continuation_env=(
    "RQ1_GENERATION_START_INDEX=$start_index"
    "RQ1_GENERATION_START_BATCH=$start_batch"
    "RQ1_GENERATION_ATTEMPT_OFFSET=$attempt_offset"
    "RQ1_GENERATION_CANDIDATE_OFFSET=${cont_candidate_offset[$method]:-}"
    "RQ1_GENERATION_PARENT_RUN_ID=${cont_parent_run[$method]:-}"
    "RQ1_GENERATION_PARENT_BATCH_ID=${cont_parent_batch[$method]:-}"
    "RQ1_GENERATION_PARENT_SOURCE_SHA256=${cont_parent_source_sha[$method]:-}"
  )
  if [[ "$method" == Fuzz4All ]]; then
    continuation_env+=(
      "RQ1_FUZZ4ALL_SEED_SOURCE=${cont_fuzz_seed_source[$method]:-}"
      "RQ1_FUZZ4ALL_PARENT_RUN_ID=${cont_parent_run[$method]:-}"
      "RQ1_FUZZ4ALL_PARENT_CANDIDATE_INDEX=${cont_fuzz_parent_candidate[$method]:-}"
    )
  fi
}

resource_profile=$(python3 "$source_root/resource_policy.py" \
  --config "$source_root/config/rq1-comparison-isolated-v1.json" \
  --mode "$resource_mode" --format json)
if [[ "$dry_run" == true ]]; then
  printf 'five external generators; segment=%s hours (%s seconds); simulator_execution=false; K1=disabled\n' \
    "$((duration_seconds / 3600))" "$duration_seconds"
  printf 'config=%s\ncase_catalog=%s\n' "$config" "$case_catalog_root"
  printf 'resource profile: %s\n' "$resource_profile"
  if [[ -n "$continue_from" ]]; then
    printf 'continuation plan: %s\n' "$continuation_plan_json"
  fi
  for method in "${methods[@]}"; do
    build_continuation_env "$method"
    command=(env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$config" "${continuation_env[@]}" bash "$entry" --action generate
      --method "$method" --run-id "${run_prefix}-${method}"
      --duration-seconds "$duration_seconds" --resource-mode "$resource_mode"
      --dependency-root "$dependency_root" --output-root "$output_root")
    printf '%q ' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

output_parent=$(dirname -- "$output_root")
if [[ -e "$output_root" ]]; then
  [[ -d "$output_root" ]] && test -w "$output_root" || {
    echo "output root is not a writable directory: $output_root" >&2; exit 2;
  }
else
  [[ -d "$output_parent" ]] && test -w "$output_parent" || {
    echo "output parent is not a writable directory: $output_parent" >&2; exit 2;
  }
fi
mkdir -p "$output_root/cases"
if [[ -d "$case_catalog_root" ]]; then
  test -w "$case_catalog_root" || {
    echo "case catalog is not writable: $case_catalog_root" >&2; exit 2;
  }
  if [[ -d "$case_catalog_root/batches" ]]; then
    test -w "$case_catalog_root/batches" || {
      echo "case catalog batch directory is not writable: $case_catalog_root/batches" >&2; exit 2;
    }
  fi
else
  case_catalog_parent=$(dirname -- "$case_catalog_root")
  [[ -d "$case_catalog_parent" ]] && test -w "$case_catalog_parent" || {
    echo "case catalog parent is not writable: $case_catalog_parent" >&2; exit 2;
  }
fi
df -hT / /mnt/data >&2
docker system df >&2
docker ps --no-trunc >&2
mkdir -p "$output_root/runs" "$output_root/source-snapshots" "$output_root/launch-logs" "$case_catalog_root"
test -w "$output_root/runs" && test -w "$output_root/source-snapshots" || {
  echo 'runs and source-snapshots must be writable' >&2; exit 2;
}
launch_dir="$output_root/launch-logs/$run_prefix"
completion_dir="$launch_dir/completed"
[[ ! -e "$launch_dir" ]] || { echo "launch prefix already exists: $launch_dir" >&2; exit 2; }
for method in "${methods[@]}"; do
  run_id="${run_prefix}-${method}"
  [[ ! -e "$output_root/runs/$run_id" ]] || { echo "run already exists: $run_id" >&2; exit 2; }
  [[ ! -e "$output_root/source-snapshots/$run_id" ]] || {
    echo "source snapshot already exists: $run_id" >&2; exit 2;
  }
done
mkdir "$launch_dir" "$completion_dir"
if [[ -n "$continue_from" ]]; then
  printf '%s\n' "$continuation_plan_json" >"$launch_dir/continuation-plan.json"
fi

tree_sha256() {
  tar --sort=name --mtime='@0' --mode='a+rwX' --owner=0 --group=0 --numeric-owner \
    -cf - -C "$1" . | sha256sum | awk '{print $1}'
}
source_tree_sha=$(tree_sha256 "$source_root")
source_commit=$(git -C "$source_root" rev-parse HEAD 2>/dev/null || true)
source_dirty=false
[[ -z "$(git -C "$source_root" status --porcelain=v1 --untracked-files=all 2>/dev/null || true)" ]] || source_dirty=true
config_sha=$(sha256sum "$config" | awk '{print $1}')
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  printf '%s\n' "$started_at"
  printf 'source_tree_sha256=%s\nsource_commit=%s\nsource_dirty=%s\n' \
    "$source_tree_sha" "$source_commit" "$source_dirty"
  df -hT / /mnt/data
  free -h
  uptime
  nproc
  docker ps --no-trunc
  docker system df
} >"$launch_dir/preflight.log" 2>&1

stop_owned_children() {
  [[ -n "$interrupted_signal" ]] || interrupted_signal=${1:-SIGTERM}
}
trap 'stop_owned_children SIGINT' INT
trap 'stop_owned_children SIGTERM' TERM
for method in "${methods[@]}"; do
  [[ -z "$interrupted_signal" ]] || break
  run_id="${run_prefix}-${method}"
  echo "[start] $method ($run_id)" >&2
  build_continuation_env "$method"
  setsid env RQ1_ENABLE_COVERAGE=1 RQ1_CONFIG_FILE="$config" "${continuation_env[@]}" bash "$entry" --action generate \
    --method "$method" --run-id "$run_id" \
    --duration-seconds "$duration_seconds" --resource-mode "$resource_mode" \
    --dependency-root "$dependency_root" --output-root "$output_root" \
    >"$launch_dir/$method.wrapper.log" 2>&1 &
  child_pid["$method"]=$!
  children+=("${child_pid[$method]}")
done

snapshot_hash_set=false
for method in "${methods[@]}"; do
  run_id="${run_prefix}-${method}"
  run_root="$output_root/runs/$run_id"
  pid=${child_pid[$method]:-}
  while [[ ! -d "$run_root" && -n "$pid" && -z "$interrupted_signal" ]] && \
        kill -0 "$pid" 2>/dev/null; do
    sleep 0.2
  done
  [[ -d "$run_root" ]] || continue
  if [[ "$snapshot_hash_set" == false && -d "$output_root/source-snapshots/$run_id" ]]; then
    source_tree_sha=$(tree_sha256 "$output_root/source-snapshots/$run_id")
    snapshot_hash_set=true
  fi
  setsid env RQ1_OUTPUT_ROOT="$output_root" RQ1_CASE_CATALOG_ROOT="$case_catalog_root" \
    bash "$control_entry" watch-checkpoints --run-id "$run_id" \
    >"$launch_dir/$method.checkpoint-watch.log" 2>&1 &
  watcher_pids+=("$!")
done

pause_generation_runs() {
  [[ "$pause_dispatched" == false ]] || return 0
  pause_dispatched=true
  local received_signal=$interrupted_signal
  local method run_id run_root pid
  local -a controls=()
  echo "[$received_signal] requesting case-boundary pause for this generation batch" >&2
  for method in "${methods[@]}"; do
    run_id="${run_prefix}-${method}"
    run_root="$output_root/runs/$run_id"
    pid=${child_pid[$method]:-}
    while [[ ! -d "$run_root" && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; do
      sleep 0.2
    done
    [[ -d "$run_root" ]] || continue
    setsid env RQ1_OUTPUT_ROOT="$output_root" RQ1_CASE_CATALOG_ROOT="$case_catalog_root" \
      bash "$control_entry" pause --run-id "$run_id" --catalog-root "$case_catalog_root" \
      >"$launch_dir/$method.pause.log" 2>&1 &
    controls+=("$!")
  done
  local control_pid
  for control_pid in "${controls[@]}"; do
    wait "$control_pid" || echo "pause acknowledgement failed for control pid $control_pid" >&2
  done
  echo "generation runners remain alive; each paused prefix is imported to the case catalog; resume with control-run.sh" >&2
  interrupted_signal=
  pause_dispatched=false
}

[[ -z "$interrupted_signal" ]] || pause_generation_runs
for method in "${methods[@]}"; do
  pid=${child_pid[$method]:-}
  [[ -n "$pid" ]] || continue
  while true; do
    [[ -z "$interrupted_signal" ]] || pause_generation_runs
    if wait "$pid"; then code=0; break; else code=$?; fi
    kill -0 "$pid" 2>/dev/null || break
  done
  printf '%s\n' "$code" >"$completion_dir/$method.exit-code"
  echo "[done] $method wrapper_exit=$code" >&2
done
for watcher_pid in "${watcher_pids[@]}"; do
  wait "$watcher_pid" || true
done
trap - INT TERM

python3 - "$launch_dir/generation-batch-manifest.json" "$output_root" \
  "$source_tree_sha" "$duration_seconds" "$completion_dir" "$config" \
  "$source_commit" "$source_dirty" "$started_at" "$resource_profile" \
  "$continue_from" "$continuation_plan_json" \
  "${methods[@]}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(manifest_path, output_root, expected_source_sha, duration, completion_dir,
 config_path, source_commit, source_dirty, started_at, resource_profile,
 continuation_anchor, continuation_plan_json) = sys.argv[1:13]
methods = sys.argv[13:]
output_root = Path(output_root)
completion_dir = Path(completion_dir)
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
continuation_plan = json.loads(continuation_plan_json) if continuation_plan_json else None

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tree_sha256(path):
    command = ["tar", "--sort=name", "--mtime=@0", "--mode=a+rwX", "--owner=0",
               "--group=0", "--numeric-owner", "-cf", "-", "-C", str(path), "."]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"could not hash source tree: {path}")
    return digest.hexdigest()

targets = [item["id"] for item in config.get("targets", [])
           if isinstance(item, dict) and item.get("id")]
batch = {
    "schema_version": "rq1-generation-batch-v1",
    "status": "running",
    "phase": "generation-only",
    "duration_seconds": int(duration),
    "simulator_execution": False,
    "board_requested": False,
    "config_sha256": file_sha256(Path(config_path)),
    "source": {"commit": source_commit or None, "dirty": source_dirty == "true",
                "tree_sha256": expected_source_sha},
    "resource_profile": json.loads(resource_profile),
    "targets": targets,
    "format_spec": "docs/EXTERNAL_CASE_CORPUS_FORMAT.md",
    "started_at_utc": started_at,
    "continuation": {
        "enabled": bool(continuation_anchor),
        "anchor_batch_id": continuation_anchor or None,
        "plan": continuation_plan,
    },
    "methods": [],
}
usable_count = 0
completed_count = 0
for method in methods:
    run_id = f"{Path(manifest_path).parent.name}-{method}"
    run_root = output_root / "runs" / run_id
    snapshot = output_root / "source-snapshots" / run_id
    generation_path = run_root / "generation-result.json"
    wrapper_path = run_root / "wrapper-result.json"
    execution_path = run_root / "execution-manifest.json"
    queue_path = run_root / "generation" / "candidate-queues.json"
    generation = load(generation_path)
    wrapper = load(wrapper_path)
    execution = load(execution_path)
    queues = load(queue_path)
    try:
        wrapper_exit = int((completion_dir / f"{method}.exit-code").read_text().strip())
    except (OSError, ValueError):
        wrapper_exit = None
    snapshot_sha = tree_sha256(snapshot) if snapshot.is_dir() else None
    target_queues = {}
    queued_entries = 0
    queued_candidates = set()
    for target in targets:
        path = run_root / "queues" / f"{target}.json"
        payload = load(path)
        entries = payload.get("entries")
        entries = entries if isinstance(entries, list) else []
        queued_entries += len(entries)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate = entry.get("candidate")
            if isinstance(candidate, dict):
                queued_candidates.add((
                    entry.get("method"), entry.get("route"), entry.get("lane"),
                    candidate.get("case_id", candidate.get("index")),
                ))
        target_queues[target] = {
            "path": str(path),
            "entry_count": len(entries),
            "sha256": file_sha256(path) if path.is_file() else None,
        }
    method_rows = generation.get("methods", [])
    method_result = next((row for row in method_rows
                          if isinstance(row, dict) and row.get("method") == method), {})
    sealed = generation.get("generation_sealed") is True
    snapshot_matches = snapshot_sha == expected_source_sha
    deadlines = execution.get("deadlines")
    identity_ok = (
        execution.get("action") == "generate"
        and execution.get("method_filter") == method
        and execution.get("duration_seconds") == int(duration)
        and isinstance(deadlines, dict)
        and deadlines.get("generation_seconds") == int(duration)
        and execution.get("target_execution_enabled") is False
        and execution.get("board_requested") is False
    )
    wrapper_ok = (
        wrapper_exit == 0 and wrapper.get("exit_code") == 0
        and wrapper.get("status") in {"completed", "partial"}
        and wrapper.get("method_filter") == method
        and wrapper.get("dependency_identity_ok") is True
        and wrapper.get("dependency_identity_stable") is True
    )
    usable = run_root.is_dir() and queued_entries > 0
    generation_status = generation.get("status") or "missing"
    status = (generation_status if generation_status in {"completed", "partial", "failed"}
              else "partial" if usable else "failed")
    usable_count += int(usable)
    completed_count += int(status == "completed")
    candidate_count = method_result.get("accepted_candidate_count")
    if not isinstance(candidate_count, int):
        candidate_count = method_result.get("candidate_count")
    if not isinstance(candidate_count, int):
        candidate_count = len(queued_candidates)
    batch["methods"].append({
        "method": method,
        "run_id": run_id,
        "status": status,
        "usable_for_replay": usable,
        "wrapper_exit_code": wrapper_exit,
        "wrapper_status": wrapper.get("status"),
        "wrapper_reason_code": wrapper.get("reason_code"),
        "generation_status": generation_status,
        "generation_sealed": sealed,
        "execution_manifest_matches_request": identity_ok,
        "wrapper_identity_ok": wrapper_ok,
        "queued_entry_count": queued_entries,
        "observed_seconds": generation.get("observed_seconds"),
        "candidate_count": candidate_count,
        "artifact_gap_count": method_result.get("artifact_gap_count"),
        "run_root": str(run_root),
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": snapshot_sha,
        "source_snapshot_matches_batch": snapshot_matches,
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": file_sha256(queue_path) if queue_path.is_file() else None,
        "target_queues": target_queues,
    })
    if usable:
        subprocess.run(["chmod", "-R", "a-w", str(run_root)], check=True)
        control_root = run_root / "control"
        if control_root.is_dir():
            subprocess.run(["chmod", "u+w", str(control_root)], check=True)
            for control_file in (
                control_root / "generation-catalog-import.lock",
                control_root / "generation-catalog-import.json",
            ):
                if control_file.exists():
                    subprocess.run(["chmod", "u+w", str(control_file)], check=True)

batch["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
batch["usable_method_count"] = usable_count
batch["failed_method_count"] = len(methods) - usable_count
batch["status"] = (
    "completed" if usable_count == len(methods) and completed_count == len(methods)
    else "partial" if usable_count else "failed"
)
path = Path(manifest_path)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
os.replace(temporary, path)
print(f"batch_status={batch['status']} usable_methods={usable_count}/{len(methods)}")
for row in batch["methods"]:
    print(f"{row['method']}: status={row['status']} candidates={row['candidate_count']} "
          f"generation_sealed={str(row['generation_sealed']).lower()}")
PY

manifest="$launch_dir/generation-batch-manifest.json"
echo "batch manifest: $manifest" >&2
usable_count=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["usable_method_count"])' "$manifest")
if [[ -n "$interrupted_signal" ]]; then exit 130; fi
if ((usable_count > 0)); then
  import_failed=false
  while IFS=$'\t' read -r method usable; do
    [[ "$usable" == true ]] || continue
    run_id="${run_prefix}-${method}"
    if ! python3 "$source_root/case_catalog.py" import-generation-checkpoint \
        --catalog-root "$case_catalog_root" --run-root "$output_root/runs/$run_id" \
        >"$launch_dir/$method.final-checkpoint-import.json" 2>"$launch_dir/$method.final-checkpoint-import.stderr"; then
      import_failed=true
      echo "[catalog-import-failed] $method; details: $launch_dir/$method.final-checkpoint-import.stderr" >&2
    fi
  done < <(python3 - "$manifest" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8")).get("methods", []):
    usable = row.get("usable_for_replay") is True
    print(row.get("method", ""), str(usable).lower(), sep="\t")
PY
  )
  [[ "$import_failed" == false ]] || exit 1
  echo "appended new generation prefixes to $case_catalog_root; per-method results: $launch_dir/*.final-checkpoint-import.json" >&2
fi
[[ "$usable_count" == "${#methods[@]}" ]] || exit 1

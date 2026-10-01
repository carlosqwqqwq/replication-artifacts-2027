#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/path/to/rq1-comparison/source/rq1-comparison
source_root="$project_root/experiments/rq1/comparison"
entry="$source_root/container/run-linux.sh"
dependency_root="${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}"
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
corpus_root=/path/to/rq1-comparison/results/cases/rq1-case-library
run_prefix="rq1-five-tool-replay-$(date -u +%Y%m%dT%H%M%SZ)-$$"
duration_seconds=259200
resource_mode=seven-method-parallel
method_filter=
dry_run=false
methods=(B-RVDV B-TORTURE B-CSMITH B-GEMI Fuzz4All)
declare -A child_pid=()
pending_signal=
all_children_started=false
pause_request_inflight=false

usage() {
  cat <<'EOF'
用法：
  container/run-five-tool-replay.sh [选项]

把已生成语料按原队列顺序输入七个 Target；每个方法当前段预算默认 72 小时，到点后在 case 边界暂停并等待续跑。
选项：
  --case-dir DIR          append-only case catalog 根目录
  --corpus-root DIR       --case-dir 的兼容名称
  --run-prefix PREFIX     本轮唯一前缀
  --duration-seconds N    当前段预算，默认 259200 秒；到点后在 case 边界暂停
  --method METHOD         只回放一个外部方法
  --dependency-root DIR   依赖目录
  --output-root DIR       结果根，只允许 /path/to/rq1-comparison/results
  --resource-mode MODE    默认 seven-method-parallel
  --dry-run               校验语料并展开命令，不启动容器
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --case-dir|--corpus-root) corpus_root=$2; shift 2;;
    --run-prefix) run_prefix=$2; shift 2;;
    --duration-seconds) duration_seconds=$2; shift 2;;
    --method) method_filter=$2; shift 2;;
    --dependency-root) dependency_root=$2; shift 2;;
    --output-root) output_root=$2; shift 2;;
    --resource-mode) resource_mode=$2; shift 2;;
    --dry-run) dry_run=true; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ "$run_prefix" =~ ^[A-Za-z0-9_.-]+$ &&
   "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid run-prefix or duration' >&2; exit 2;
}
if [[ -n "$method_filter" ]] &&
   [[ ! " ${methods[*]} " =~ " $method_filter " ]]; then
  echo "unknown method: $method_filter" >&2
  exit 2
fi
if [[ -n "$method_filter" ]]; then methods=("$method_filter"); fi
output_root=$(realpath -m -- "$output_root")
[[ "$output_root" == /path/to/rq1-comparison/results ]] || {
  echo 'output-root must be exactly /path/to/rq1-comparison/results' >&2; exit 2;
}
corpus_root=$(realpath -- "$corpus_root")
launch_dir="$output_root/launch-logs/$run_prefix"

validation_output=$(PYTHONDONTWRITEBYTECODE=1 python3 - "$source_root" "$corpus_root" \
  "$run_prefix" "$duration_seconds" "$method_filter" "${methods[@]}" <<'PY'
import json
import sys
from pathlib import Path

source_root, corpus_arg, run_prefix, duration, selected, *methods = sys.argv[1:]
source_root = Path(source_root).resolve()
corpus = Path(corpus_arg).resolve()
sys.path.insert(0, str(source_root))
from case_catalog import load_catalog_queues

catalog = json.loads((corpus / "catalog-manifest.json").read_text(encoding="utf-8"))
execution_config = source_root / "config" / "rq1-comparison-isolated-v1.json"
if not execution_config.is_file():
    raise SystemExit(f"simulator execution config is missing: {execution_config}")
config = json.loads(execution_config.read_text(encoding="utf-8"))
method_filter = methods[0] if len(methods) == 1 else None
queues, rows, source, _ = load_catalog_queues(corpus, config, method_filter)
targets = [row["id"] for row in config.get("targets", [])
           if isinstance(row, dict) and row.get("id")]
for method in methods:
    rows.setdefault(method, {"method": method, "status": "partial", "candidate_count": 0})
if len(targets) != 7:
    raise SystemExit("execution config must contain seven Targets")
print(execution_config)
print(corpus)
print(source["catalog_revision"])
print(source["catalog_sha256"])
PY
)
mapfile -t validated_paths <<<"$validation_output"
if ((${#validated_paths[@]} != 4)); then
  echo 'case catalog validation returned an incomplete result' >&2
  exit 2
fi
config_path=${validated_paths[0]}
corpus_root=${validated_paths[1]}
catalog_revision=${validated_paths[2]}
catalog_digest=${validated_paths[3]}
mapfile -t targets < <(python3 - "$config_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(str(row["id"]) for row in config.get("targets", [])
                  if isinstance(row, dict) and row.get("id")))
PY
)

if [[ "$dry_run" == true ]]; then
  printf 'corpus=%s\n' "$corpus_root"
  printf 'simulator-execution-config=%s\n' "$config_path"
  python3 - "$corpus_root/catalog-manifest.json" "${methods[@]}" <<'PY'
import json
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"catalog revision={catalog['revision']} sha256={catalog['catalog_sha256']}")
for method in sys.argv[2:]:
    row = catalog["methods"].get(method)
    count = row.get("case_count", 0) if isinstance(row, dict) else 0
    next_number = row.get("next_case_number") if isinstance(row, dict) else None
    print(f"{method}: cases={count} next_case_number={next_number}")
PY
  printf 'segment-budget=%s seconds per method; targets=7; K1 reference=enabled; coverage=raw-artifacts-only\n' \
    "$duration_seconds"
  for method in "${methods[@]}"; do
    run_id="$run_prefix-$method"
    printf 'execute-existing-queues method=%s case-catalog=%s output-run=%s\n' \
      "$method" "$corpus_root" "$run_id"
  done
  exit 0
fi

[[ -x "$entry" ]] || { echo "missing run entry: $entry" >&2; exit 2; }
[[ -d "$dependency_root/external" && -d "$dependency_root/toolchains" &&
   -d "$dependency_root/python" && -d "$dependency_root/target-store" ]] || {
  echo 'dependency root is incomplete' >&2; exit 2;
}
[[ -d "$output_root/runs" && -w "$output_root/runs" &&
   -d "$output_root/launch-logs" && -w "$output_root/launch-logs" ]] || {
  echo 'results run and launch-log directories must be writable' >&2; exit 2;
}
df -hT / /mnt/data >&2
docker system df >&2
for method in "${methods[@]}"; do
  [[ ! -e "$output_root/runs/$run_prefix-$method" ]] || {
    echo "run already exists: $run_prefix-$method" >&2; exit 2;
  }
done

launcher_path=$(readlink -f -- "${BASH_SOURCE[0]}")
tree_sha256() {
  tar --sort=name --mtime='@0' --mode='a+rwX' --owner=0 --group=0 \
    --numeric-owner -cf - -C "$1" . | sha256sum | awk '{print $1}'
}
expected_source_tree_sha=$(tree_sha256 "$source_root")
if [[ "${RQ1_REPLAY_FROZEN_ENTRY:-0}" == 1 ]]; then
  [[ -d "$launch_dir" ]] || { echo "missing launch directory: $launch_dir" >&2; exit 2; }
  expected_launcher_sha=${RQ1_REPLAY_FROZEN_SHA256:-}
  actual_launcher_sha=$(sha256sum -- "$launcher_path" | awk '{print $1}')
  [[ "$expected_launcher_sha" =~ ^[0-9a-f]{64}$ &&
     "$actual_launcher_sha" == "$expected_launcher_sha" ]] || {
    echo 'replay launcher snapshot digest mismatch' >&2; exit 125;
  }
else
  [[ ! -e "$launch_dir" ]] || { echo "launch prefix already exists: $launch_dir" >&2; exit 2; }
  mkdir -p "$launch_dir"
  launcher_snapshot="$launch_dir/run-five-tool-replay.sh"
  cp -- "$launcher_path" "$launcher_snapshot"
  chmod a-w -- "$launcher_snapshot"
  launcher_sha=$(sha256sum -- "$launcher_snapshot" | awk '{print $1}')
  normalized_args=(--corpus-root "$corpus_root" --run-prefix "$run_prefix"
    --duration-seconds "$duration_seconds"
    --resource-mode "$resource_mode"
    --dependency-root "$dependency_root" --output-root "$output_root")
  [[ -z "$method_filter" ]] || normalized_args+=(--method "$method_filter")
  exec env RQ1_REPLAY_FROZEN_ENTRY=1 RQ1_REPLAY_FROZEN_SHA256="$launcher_sha" \
    bash "$launcher_snapshot" "${normalized_args[@]}"
fi

mkdir -p "$launch_dir/completed"
entry_snapshot="$launch_dir/run-linux.sh"
cp -- "$entry" "$entry_snapshot"
chmod a-w -- "$entry_snapshot"
entry_sha=$(sha256sum -- "$entry_snapshot" | awk '{print $1}')
catalog_manifest_file_sha=$(sha256sum -- "$corpus_root/catalog-manifest.json" | awk '{print $1}')
catalog_manifest_snapshot="$launch_dir/catalog-manifest.snapshot.json"
cp -- "$corpus_root/catalog-manifest.json" "$catalog_manifest_snapshot"
chmod a-w -- "$catalog_manifest_snapshot"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$launch_dir/replay-batch-manifest.json" "$corpus_root" "$catalog_manifest_snapshot" \
  "$catalog_manifest_file_sha" "$catalog_revision" "$catalog_digest" \
  "$run_prefix" "$duration_seconds" "$resource_mode" \
  "$entry_sha" "$expected_source_tree_sha" "${RQ1_REPLAY_FROZEN_SHA256:-}" \
  "$started_at" "${methods[@]}" <<'PY'
import json
import sys
from pathlib import Path

path, corpus, catalog_snapshot, catalog_file_sha, catalog_revision, catalog_sha, prefix, duration, resource_mode, entry_sha, source_tree_sha, launcher_sha, started, *methods = sys.argv[1:]
catalog_root = Path(corpus)
catalog_path = Path(catalog_snapshot)
catalog_bytes = catalog_path.read_bytes()
catalog = json.loads(catalog_bytes)
import hashlib
catalog_file_sha = hashlib.sha256(catalog_bytes).hexdigest()
catalog_revision = str(catalog.get("revision", catalog_revision))
catalog_sha = catalog.get("catalog_sha256", catalog_sha)
batch_indexes = []
for item in catalog.get("batches", []):
    try:
        batch_indexes.append(
            json.loads((catalog_root / item["index_path"]).read_text(encoding="utf-8"))
        )
    except (OSError, KeyError, TypeError, ValueError):
        continue
doc = {
    "schema_version": "rq1-existing-case-replay-batch-v1",
    "status": "running",
    "case_catalog_root": corpus,
    "case_catalog_revision": int(catalog_revision) if str(catalog_revision).isdigit() else 0,
    "case_catalog_sha256": catalog_sha,
    "case_catalog_manifest_file_sha256": catalog_file_sha,
    "case_catalog_manifest_snapshot": "catalog-manifest.snapshot.json",
    "case_counts": {
        method: catalog.get("methods", {}).get(method, {}).get("case_count", 0)
        for method in methods
    },
    "catalog_batches": [
        {"batch_id": item["batch_id"],
         "corpus_manifest_sha256": item["corpus_manifest_sha256"],
         "generation_batch_manifest_sha256": item["generation_batch_manifest_sha256"],
         "original_batch_status": item.get("original_batch_status")}
        for item in batch_indexes
    ],
    "replay_basis": "case catalog queues available at launch",
    "replay_action": "execute-existing-queues",
    "duration_seconds": int(duration),
    "segment_budget_mode": "case-boundary-pause-resume",
    "duration_accounting": "case-boundary-segment; paused time excluded",
    "resource_mode": resource_mode,
    "coverage_processing": "raw-artifacts-only; aggregate offline",
    "raw_artifact_index": "raw-artifact-index.json",
    "board_requested": True,
    "launcher_snapshot_sha256": launcher_sha,
    "run_linux_snapshot_sha256": entry_sha,
    "source_tree_sha256": source_tree_sha,
    "started_at_utc": started,
    "methods": {method: {"run_id": f"{prefix}-{method}", "status": "pending"}
                for method in methods},
}
Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY

write_raw_artifact_index() {
  python3 - "$launch_dir/replay-batch-manifest.json" "$output_root" \
    "$run_prefix" "${#methods[@]}" "${methods[@]}" \
    "${#targets[@]}" "${targets[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2]).resolve()
run_prefix = sys.argv[3]
offset = 4
method_count = int(sys.argv[offset]); offset += 1
methods = sys.argv[offset:offset + method_count]; offset += method_count
target_count = int(sys.argv[offset]); offset += 1
targets = sys.argv[offset:offset + target_count]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def reference(path):
    path = Path(path)
    return {
        "path": path.relative_to(output_root).as_posix(),
        "exists": path.exists(),
    }

items = []
for method in methods:
    row = manifest.get("methods", {}).get(method, {})
    run_id = row.get("run_id") or f"{run_prefix}-{method}"
    run_root = output_root / "runs" / run_id
    executions = run_root / "executions"
    profiles = run_root / "coverage-raw"
    items.append({
        "method": method,
        "run_id": run_id,
        "status": row.get("status", "pending"),
        "run_root": reference(run_root),
        "execution_manifest": reference(run_root / "execution-manifest.json"),
        "run_result": reference(run_root / "run-result.json"),
        "wrapper_result": reference(run_root / "wrapper-result.json"),
        "ledger": {
            "root": reference(run_root / "ledger"),
            "events": reference(run_root / "ledger" / "events.jsonl"),
            "partial_events": reference(run_root / "ledger" / "events.partial.jsonl"),
            "seal": reference(run_root / "ledger" / "seal.json"),
        },
        "execution_artifacts": {
            "root": reference(executions),
            "target_run_records_pattern": "**/target-run.json",
        },
        "trace_artifacts": {
            "root": reference(executions),
            "paths_recorded_in": "**/target-run.json: targets[].trace_path",
        },
        "case_boundary_control": {
            "request": reference(run_root / "control" / "request.json"),
            "status_and_cursors": reference(run_root / "control" / "status.json"),
        },
        "coverage_raw": {
            "root": reference(profiles),
            "profile_roots": [
                {"target": target,
                 **reference(profiles / target / "attempts")}
                for target in targets
            ],
        },
    })

index_path = manifest_path.parent / "raw-artifact-index.json"
temporary = index_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps({
    "schema_version": "rq1-external-replay-raw-artifact-index-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_prefix": run_prefix,
    "methods": items,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, index_path)
PY
}

pause_on_signal() {
  local signal=${1:-SIGTERM}
  if [[ "$all_children_started" != true ]]; then
    pending_signal=$signal
    return 0
  fi
  [[ "$pause_request_inflight" == false ]] || return 0
  pause_request_inflight=true
  bash "$source_root/container/control-five-tool-replay.sh" pause \
    --run-prefix "$run_prefix" || echo 'case-boundary pause request failed' >&2
  write_raw_artifact_index
  pause_request_inflight=false
}
trap 'pause_on_signal SIGINT' INT
trap 'pause_on_signal SIGTERM' TERM

write_raw_artifact_index

for method in "${methods[@]}"; do
  run_id="$run_prefix-$method"
  method_config="$config_path"
  echo "[start] $method ($run_id)"
  (
    set +e
    env RQ1_ENABLE_COVERAGE=1 RQ1_DEFER_FINALIZE=1 RQ1_CONFIG_FILE="$method_config" \
      RQ1_EXPECTED_SOURCE_TREE_SHA256="$expected_source_tree_sha" \
      RQ1_ALLOW_FROZEN_ENTRY=1 RQ1_FROZEN_ENTRY_SHA256="$entry_sha" \
      bash "$entry_snapshot" --action execute-existing-queues --method "$method" \
      --case-catalog "$corpus_root" --run-id "$run_id" \
      --case-catalog-manifest "$catalog_manifest_snapshot" \
        --duration-seconds "$duration_seconds" \
        --resource-mode "$resource_mode" \
        --with-board --dependency-root "$dependency_root" --output-root "$output_root"
    code=$?
    printf '%s\n' "$code" > "$launch_dir/completed/$method.exit-code"
    touch "$launch_dir/completed/$method.done"
    exit "$code"
  ) >"$launch_dir/$method.wrapper.log" 2>&1 &
  child_pid["$method"]=$!
done
all_children_started=true
if [[ -n "$pending_signal" ]]; then
  signal=$pending_signal
  pending_signal=
  pause_on_signal "$signal"
fi
failed_methods=0
partial_methods=0
for method in "${methods[@]}"; do
  pid=${child_pid[$method]}
  while :; do
    if wait "$pid"; then code=0; else code=$?; fi
    [[ -f "$launch_dir/completed/$method.done" ]] && break
    kill -0 "$pid" 2>/dev/null || break
  done
  if [[ -f "$launch_dir/completed/$method.exit-code" ]]; then
    read -r code < "$launch_dir/completed/$method.exit-code"
  fi
  status=$(python3 - "$output_root/runs/$run_prefix-$method/wrapper-result.json" \
    "$code" <<'PY'
import json
import sys
from pathlib import Path

try:
    result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    result = {}
status = result.get("status")
if status not in {"completed", "partial", "failed"}:
    status = "failed"
print(status)
PY
  )
  if [[ "$status" == failed ]]; then ((failed_methods+=1)); fi
  if [[ "$status" == partial ]]; then ((partial_methods+=1)); fi
  python3 - "$launch_dir/replay-batch-manifest.json" "$method" "$status" "$code" \
    "$output_root/runs/$run_prefix-$method" <<'PY'
import json
import os
import sys
from pathlib import Path

path, method, status, code, run_root = sys.argv[1:]
manifest = json.loads(Path(path).read_text(encoding="utf-8"))
run_root = Path(run_root)
def load(relative):
    try:
        return json.loads((run_root / relative).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
wrapper = load("wrapper-result.json")
manifest["methods"][method].update(
    status=status,
    wrapper_exit_code=int(code),
    execution_complete=wrapper.get("execution_complete") is True,
)
manifest["finished_methods"] = sum(
    row.get("status") in {"completed", "partial", "failed"}
    for row in manifest["methods"].values()
)
temporary = Path(path).with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY
  echo "[done] $method wrapper_exit=$code"
done
trap - INT TERM

python3 - "$launch_dir/replay-batch-manifest.json" "$failed_methods" "$partial_methods" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
failed = int(sys.argv[2])
partial = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "failed" if failed else "partial" if partial else "completed"
manifest["execution_status"] = manifest["status"]
manifest["coverage_processing"] = "raw-artifacts-only; aggregate offline"
manifest["raw_artifact_index"] = "raw-artifact-index.json"
manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
os.replace(temporary, manifest_path)
PY
write_raw_artifact_index
[[ "$failed_methods" -eq 0 ]]

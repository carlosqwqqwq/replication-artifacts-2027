#!/usr/bin/env bash
set -euo pipefail

action=run
run_id=
method_filter=
feedback_target=
dependency_root="${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}"
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
image_ref="${RQ1_IMAGE_REF:-migration-rq1-compare-jdk17:rq1-comparison}"
duration_seconds="${RQ1_DURATION_SECONDS:-1800}"
mcmc_steps="${RQ1_MCMC_STEPS:-}"
small_model_mode=on
emi_mode=on
mcmc_mode=on
module_toggles_requested=false
torture_cache="${RQ1_TORTURE_CACHE_ROOT:-/path/to/rq1-comparison/deps/torture-cache}"
target_timeout_seconds=
drain_timeout_seconds="${RQ1_DRAIN_TIMEOUT_SECONDS:-300}"
finalize_timeout_seconds="${RQ1_FINALIZE_TIMEOUT_SECONDS:-900}"
generation_root=
case_catalog_root=
case_catalog_manifest_host=
rvdv_seed_catalog_root=
rvdv_seed_catalog_manifest_host=
fuzz4all_seed_source="${RQ1_FUZZ4ALL_SEED_SOURCE:-}"
fuzz4all_parent_run_id="${RQ1_FUZZ4ALL_PARENT_RUN_ID:-}"
fuzz4all_parent_candidate_index="${RQ1_FUZZ4ALL_PARENT_CANDIDATE_INDEX:-}"
generation_start_index="${RQ1_GENERATION_START_INDEX:-}"
generation_start_batch="${RQ1_GENERATION_START_BATCH:-}"
generation_attempt_offset="${RQ1_GENERATION_ATTEMPT_OFFSET:-}"
generation_candidate_offset="${RQ1_GENERATION_CANDIDATE_OFFSET:-}"
generation_parent_run_id="${RQ1_GENERATION_PARENT_RUN_ID:-}"
generation_parent_batch_id="${RQ1_GENERATION_PARENT_BATCH_ID:-}"
generation_parent_source_sha256="${RQ1_GENERATION_PARENT_SOURCE_SHA256:-}"
with_board=false
without_board=false
resource_mode="${RQ1_RESOURCE_MODE:-single-method}"


usage() {
  cat <<'EOF'
用法：
  run-linux.sh --action {doctor|run|execute-existing-queues|framework-run|generate} --run-id ID [选项]

选项：
  --dependency-root DIR   依赖目录，默认 /path/to/rq1-comparison/deps
  --output-root DIR       结果目录，默认 /path/to/rq1-comparison/results
  --duration-seconds N    当前运行段时长，默认 1800 秒
  --mcmc-steps N          framework-run 覆盖两个 Ours 方法的每 case MCMC 步数
  --small-model on|off    framework-run 的根 small-model 改写开关
  --emi on|off            framework-run 的 EMI proposal 开关
  --mcmc on|off           framework-run 的 MCMC 接受/拒绝开关
  --method METHOD         comparison 每个容器只运行一个外部工具；framework-run 选择一个 Ours 方法
  --generation-root DIR   execute-existing-queues 使用的只读生成语料 run root
  --case-catalog DIR     execute-existing-queues 使用的只读追加语料目录
  --case-catalog-manifest FILE  固定回放启动时的 catalog revision 快照
  --rvdv-seed-catalog DIR Program-Full 使用的 B-RVDV 初始 case catalog
  --rvdv-seed-manifest FILE 固定本组 Program-Full run 共用的 catalog revision
  Fuzz4All 续生成可设置 RQ1_FUZZ4ALL_SEED_SOURCE、RQ1_FUZZ4ALL_PARENT_RUN_ID 和 RQ1_FUZZ4ALL_PARENT_CANDIDATE_INDEX
  续生成游标由 RQ1_GENERATION_START_INDEX、RQ1_GENERATION_START_BATCH、RQ1_GENERATION_ATTEMPT_OFFSET 和 RQ1_GENERATION_CANDIDATE_OFFSET 注入
  --image IMAGE           Docker 镜像，可用 RQ1_IMAGE_REF 覆盖
  --torture-cache-root DIR
  --target-timeout-seconds N
  --feedback-target TARGET  framework-run 的在线反馈 Target；默认 all，逐 Target 独立闭环
  --resource-mode MODE      single-method、two-method-parallel、four-method-parallel 或 seven-method-parallel；默认 single-method
  --with-board           comparison 附加 K1；两个 Ours framework-run 默认已启用 K1
  --without-board        comparison 不使用 K1；Ours framework-run 禁用 K1 挂载
EOF
}

set_toggle() {
  local name=$1 value=$2
  [[ "$value" == on || "$value" == off ]] || {
    echo "$name must be on or off" >&2
    exit 2
  }
  printf -v "$name" '%s' "$value"
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --action) action=$2; shift 2;;
    --run-id) run_id=$2; shift 2;;
    --dependency-root) dependency_root=$2; shift 2;;
    --output-root) output_root=$2; shift 2;;
    --image) image_ref=$2; shift 2;;
    --duration-seconds) duration_seconds=$2; shift 2;;
    --mcmc-steps) mcmc_steps=$2; shift 2;;
    --small-model) set_toggle small_model_mode "$2"; module_toggles_requested=true; shift 2;;
    --emi) set_toggle emi_mode "$2"; module_toggles_requested=true; shift 2;;
    --mcmc) set_toggle mcmc_mode "$2"; module_toggles_requested=true; shift 2;;
    --method) method_filter=$2; shift 2;;
    --feedback-target) feedback_target=$2; shift 2;;
    --torture-cache-root) torture_cache=$2; shift 2;;
    --with-board) with_board=true; shift;;
    --without-board) without_board=true; shift;;
    --target-timeout-seconds) target_timeout_seconds=$2; shift 2;;
    --generation-root) generation_root=$2; shift 2;;
    --case-catalog) case_catalog_root=$2; shift 2;;
    --case-catalog-manifest) case_catalog_manifest_host=$2; shift 2;;
    --rvdv-seed-catalog) rvdv_seed_catalog_root=$2; shift 2;;
    --rvdv-seed-manifest) rvdv_seed_catalog_manifest_host=$2; shift 2;;
    --resource-mode) resource_mode=$2; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$run_id" && -n "$dependency_root" && -n "$output_root" ]] || {
  echo 'run-id, dependency-root and output-root are required' >&2; exit 2;
}
[[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ && "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid run-id or duration' >&2; exit 2;
}
if [[ -n "$mcmc_steps" ]]; then
  [[ "$action" == "framework-run" && "$mcmc_steps" =~ ^[1-9][0-9]*$ ]] || {
    echo '--mcmc-steps requires framework-run and a positive integer' >&2; exit 2;
  }
fi
case "$action" in
  run|doctor|execute-existing-queues|framework-run|generate) ;;
  *) echo "unsupported action: $action" >&2; exit 2;;
esac
if [[ "$module_toggles_requested" == true && "$action" != "framework-run" ]]; then
  echo 'small-model, EMI and MCMC toggles only apply to framework-run' >&2
  exit 2
fi
if [[ "$action" == "framework-run" && "$module_toggles_requested" == true &&
      "$method_filter" != "Ours-RVGEN-Direct" &&
      "$method_filter" != "Ours-Program-Full" ]]; then
  echo 'module toggles only apply to Ours framework methods' >&2
  exit 2
fi
if [[ "$action" == "framework-run" && "$emi_mode" == off && "$mcmc_mode" == on ]]; then
  echo 'EMI off requires MCMC off because MCMC uses EMI proposals' >&2
  exit 2
fi
export RQ1_FRAMEWORK_SMALL_MODEL_MODE="$small_model_mode"
export RQ1_FRAMEWORK_EMI_MODE="$emi_mode"
export RQ1_FRAMEWORK_MCMC_MODE="$mcmc_mode"
comparison_execution_action=false
if [[ "$action" == "run" || "$action" == "execute-existing-queues" ]]; then
  comparison_execution_action=true
fi
case "$resource_mode" in
  single|single-method) resource_mode=single-method;;
  two|two-method-parallel) resource_mode=two-method-parallel;;
  four|four-method-parallel) resource_mode=four-method-parallel;;
  seven|seven-parallel|seven-method-parallel) resource_mode=seven-method-parallel;;
  *) echo 'resource-mode must be single-method, two-method-parallel, four-method-parallel, or seven-method-parallel' >&2; exit 2;;
esac
if [[ "$action" == "framework-run" && -z "$feedback_target" ]]; then
  feedback_target=all
fi
if [[ "$action" == "framework-run" && -z "$method_filter" ]]; then
  echo 'framework-run requires --method Ours-*' >&2
  exit 2
fi
if [[ ( "$action" == "run" || "$action" == "generate" ||
        "$action" == "execute-existing-queues" ) && -z "$method_filter" ]]; then
  echo "$action requires --method for one-method-per-container execution" >&2
  exit 2
fi
fuzz4all_seed_sha256=
if [[ -n "$fuzz4all_seed_source" || -n "$fuzz4all_parent_run_id" ||
      -n "$fuzz4all_parent_candidate_index" ]]; then
  [[ "$action" == "generate" && "$method_filter" == "Fuzz4All" ]] || {
    echo 'Fuzz4All continuation inputs only apply to generate --method Fuzz4All' >&2
    exit 2
  }
  [[ -n "$fuzz4all_seed_source" &&
     "$fuzz4all_parent_run_id" =~ ^[A-Za-z0-9_.-]+$ &&
     "$fuzz4all_parent_candidate_index" =~ ^[0-9]+$ ]] || {
    echo 'Fuzz4All continuation seed, parent run id, and candidate index are required' >&2
    exit 2
  }
  [[ -f "$fuzz4all_seed_source" && ! -L "$fuzz4all_seed_source" && -r "$fuzz4all_seed_source" ]] || {
    echo "invalid Fuzz4All continuation seed: $fuzz4all_seed_source" >&2
    exit 2
  }
  fuzz4all_seed_source=$(realpath -e -- "$fuzz4all_seed_source")
  [[ -s "$fuzz4all_seed_source" ]] || {
    echo 'Fuzz4All continuation seed is empty' >&2
    exit 2
  }
  fuzz4all_seed_sha256=$(sha256sum -- "$fuzz4all_seed_source" | awk '{print $1}')
fi
if [[ -n "$generation_start_index" || -n "$generation_start_batch" ||
      -n "$generation_attempt_offset" || -n "$generation_candidate_offset" ||
      -n "$generation_parent_run_id" || -n "$generation_parent_batch_id" ||
      -n "$generation_parent_source_sha256" ]]; then
  [[ "$action" == "generate" && -n "$method_filter" ]] || {
    echo 'generation continuation inputs only apply to generate --method' >&2
    exit 2
  }
  for value in "$generation_start_index" "$generation_start_batch" \
               "$generation_attempt_offset" "$generation_candidate_offset"; do
    [[ -z "$value" || "$value" =~ ^[0-9]+$ ]] || {
      echo 'generation continuation offsets must be non-negative integers' >&2
      exit 2
    }
  done
  [[ -z "$generation_parent_run_id" || "$generation_parent_run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo 'invalid generation parent run id' >&2; exit 2;
  }
  [[ -z "$generation_parent_batch_id" || "$generation_parent_batch_id" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo 'invalid generation parent batch id' >&2; exit 2;
  }
  [[ -z "$generation_parent_source_sha256" || "$generation_parent_source_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo 'invalid generation parent source sha256' >&2; exit 2;
  }
  case "$method_filter" in
    B-RVDV) [[ -z "$generation_start_index" && -z "$generation_attempt_offset" ]] || {
      echo 'B-RVDV continuation uses start-batch, not start-index/attempt-offset' >&2; exit 2;
    };;
    B-TORTURE|B-CSMITH) [[ -z "$generation_start_batch" && -z "$generation_attempt_offset" ]] || {
      echo "$method_filter continuation uses start-index, not start-batch/attempt-offset" >&2; exit 2;
    };;
    B-GEMI) [[ -z "$generation_start_index" && -z "$generation_start_batch" ]] || {
      echo 'B-GEMI continuation uses attempt-offset, not start-index/start-batch' >&2; exit 2;
    };;
    Fuzz4All) [[ -z "$generation_start_index" && -z "$generation_start_batch" &&
                 -z "$generation_attempt_offset" ]] || {
      echo 'Fuzz4All continuation uses its seed inputs, not generic start offsets' >&2; exit 2;
    };;
    *) echo "unsupported continuation method: $method_filter" >&2; exit 2;;
  esac
fi
canonical_repo_root=/path/to/rq1-comparison/source/rq1-comparison
canonical_entry="$canonical_repo_root/experiments/rq1/comparison/container/run-linux.sh"
entry_path=$(readlink -f -- "${BASH_SOURCE[0]}")
if [[ "$entry_path" != "$canonical_entry" ]]; then
  expected_entry_sha=${RQ1_FROZEN_ENTRY_SHA256:-}
  [[ "${RQ1_ALLOW_FROZEN_ENTRY:-0}" == 1 &&
     "$expected_entry_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "formal entry must use canonical source: $canonical_entry" >&2
    exit 125
  }
  actual_entry_sha=$(sha256sum -- "$entry_path" | awk '{print $1}')
  [[ "$actual_entry_sha" == "$expected_entry_sha" ]] || {
    echo "frozen entry digest mismatch: $entry_path" >&2
    exit 125
  }
fi
source_root="$canonical_repo_root/experiments/rq1/comparison"
repo_root="$canonical_repo_root"
[[ -d "$source_root" ]] || { echo "missing canonical source: $source_root" >&2; exit 2; }
if [[ "$action" == "execute-existing-queues" ]]; then
  [[ -n "$generation_root" || -n "$case_catalog_root" ]] || {
    echo 'execute-existing-queues requires --generation-root or --case-catalog' >&2; exit 2;
  }
  [[ -z "$generation_root" || -z "$case_catalog_root" ]] || {
    echo 'choose one of --generation-root or --case-catalog' >&2; exit 2;
  }
  if [[ -n "$case_catalog_manifest_host" && -z "$case_catalog_root" ]]; then
    echo '--case-catalog-manifest requires --case-catalog' >&2; exit 2;
  fi
  if [[ -n "$case_catalog_root" ]]; then
    case_catalog_root=$(realpath -- "$case_catalog_root")
    [[ -d "$case_catalog_root" && -f "$case_catalog_root/catalog-manifest.json" ]] || {
      echo "invalid case catalog: $case_catalog_root" >&2; exit 2;
    }
    generation_root=$(dirname -- "$case_catalog_root")
  else
    [[ -d "$generation_root" ]] || {
      echo "missing generation root: $generation_root" >&2; exit 2;
    }
    generation_root=$(realpath -- "$generation_root")
  fi
elif [[ -n "$generation_root" || -n "$case_catalog_root" ]]; then
  echo '--generation-root and --case-catalog only apply to execute-existing-queues' >&2
  exit 2
fi
if [[ "$action" != "execute-existing-queues" && -n "$case_catalog_manifest_host" ]]; then
  echo '--case-catalog-manifest only applies to execute-existing-queues' >&2
  exit 2
fi
config_file="${RQ1_CONFIG_FILE:-$source_root/config/rq1-comparison-isolated-v1.json}"
config_file=$(realpath -- "$config_file")
if [[ -n "$mcmc_steps" ]]; then
  config_override_file=$(mktemp /tmp/rq1-mcmc-config.XXXXXX.json)
  python3 - "$config_file" "$config_override_file" "$mcmc_steps" <<'PY'
import json
import sys
from pathlib import Path

source, destination, raw_steps = sys.argv[1:]
steps = int(raw_steps)
config = json.loads(Path(source).read_text(encoding="utf-8"))
updated = 0
for method in config.get("framework_methods", ()):
    if not isinstance(method, dict) or method.get("id") not in {
        "Ours-RVGEN-Direct", "Ours-Program-Full",
    }:
        continue
    profile = method.get("generator_profile")
    if not isinstance(profile, dict):
        raise SystemExit(f"missing generator profile: {method.get('id')}")
    profile["steps"] = steps
    updated += 1
if updated != 2:
    raise SystemExit("framework config must contain both Ours methods")
Path(destination).write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)
PY
  config_file=$(realpath -- "$config_override_file")
fi
canonical_output_root=/path/to/rq1-comparison/results
output_root=$(realpath -m -- "$output_root")
case "$output_root" in
  "$canonical_output_root") ;;
  *)
    echo "output-root must be exactly $canonical_output_root (large results belong on /mnt/data)" >&2
    exit 2
    ;;
  esac
if [[ -n "$case_catalog_root" ]]; then
  case "$case_catalog_root" in
    "$canonical_output_root"/cases/*) ;;
    *) echo 'case-catalog must be below /path/to/rq1-comparison/results/cases' >&2; exit 2;;
  esac
fi
rvdv_seed_artifact_root=
rvdv_seed_catalog_relative=
rvdv_seed_manifest_source=
rvdv_seed_catalog_manifest_sha256=
if [[ "$action" == "framework-run" && "$method_filter" == "Ours-Program-Full" ]]; then
  rvdv_seed_catalog_root=${rvdv_seed_catalog_root:-$canonical_output_root/cases/rq1-case-library}
  rvdv_seed_catalog_root=$(realpath -- "$rvdv_seed_catalog_root")
  case "$rvdv_seed_catalog_root" in
    "$canonical_output_root"/cases/*) ;;
    *) echo 'RVDV seed catalog must be below /path/to/rq1-comparison/results/cases' >&2; exit 2;;
  esac
  [[ -d "$rvdv_seed_catalog_root" && -f "$rvdv_seed_catalog_root/catalog-manifest.json" ]] || {
    echo "invalid RVDV seed catalog: $rvdv_seed_catalog_root" >&2; exit 2;
  }
  rvdv_seed_artifact_root=$(dirname -- "$rvdv_seed_catalog_root")
  rvdv_seed_catalog_relative=${rvdv_seed_catalog_root#"$rvdv_seed_artifact_root"/}
  if [[ -n "$rvdv_seed_catalog_manifest_host" ]]; then
    [[ ! -L "$rvdv_seed_catalog_manifest_host" && -f "$rvdv_seed_catalog_manifest_host" ]] || {
      echo "invalid pinned RVDV seed manifest: $rvdv_seed_catalog_manifest_host" >&2; exit 2;
    }
    rvdv_seed_catalog_manifest_host=$(realpath -- "$rvdv_seed_catalog_manifest_host")
    case "$rvdv_seed_catalog_manifest_host" in
      "$output_root"/launch-logs/*/rvdv-seed-catalog-manifest.snapshot.json|\
      "$output_root"/launch-logs/*/catalog-manifest.snapshot.json) ;;
      *) echo 'pinned RVDV seed manifest must be inside its launch log directory' >&2; exit 2;;
    esac
    rvdv_seed_manifest_source=$rvdv_seed_catalog_manifest_host
  else
    rvdv_seed_manifest_source="$rvdv_seed_catalog_root/catalog-manifest.json"
  fi
  rvdv_seed_catalog_manifest_sha256=$(sha256sum -- "$rvdv_seed_manifest_source" | awk '{print $1}')
  python3 - "$rvdv_seed_manifest_source" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = value.get("methods", {}).get("B-RVDV", {})
if (
    value.get("schema_version") != "rq1-case-catalog-v1"
    or type(value.get("revision")) is not int
    or type(row.get("case_count")) is not int
    or row["case_count"] < 1
):
    raise SystemExit("pinned case catalog has no B-RVDV sequence")
PY
elif [[ -n "$rvdv_seed_catalog_root" || -n "$rvdv_seed_catalog_manifest_host" ]]; then
  echo '--rvdv-seed-catalog only applies to framework-run --method Ours-Program-Full' >&2
  exit 2
fi
case_catalog_manifest_sha256=
if [[ -n "$case_catalog_manifest_host" ]]; then
  [[ ! -L "$case_catalog_manifest_host" && -f "$case_catalog_manifest_host" ]] || {
    echo "invalid pinned case catalog manifest: $case_catalog_manifest_host" >&2; exit 2;
  }
  case_catalog_manifest_host=$(realpath -- "$case_catalog_manifest_host")
  case "$case_catalog_manifest_host" in
    "$output_root"/launch-logs/*/catalog-manifest.snapshot.json) ;;
    *) echo 'pinned catalog manifest must be inside its replay launch directory' >&2; exit 2;;
  esac
  case_catalog_manifest_sha256=$(sha256sum -- "$case_catalog_manifest_host" | awk '{print $1}')
fi
python3 - "$config_file" <<'PY'
import json
import math
import sys

config = json.load(open(sys.argv[1]))
limits = config.get("execution_policy", {}).get("resource_limits")
if not isinstance(limits, dict):
    raise SystemExit("execution_policy.resource_limits is missing")
cpus = limits.get("cpus")
if type(cpus) not in (int, float) or not math.isfinite(cpus) or cpus <= 0:
    raise SystemExit("execution_policy.resource_limits.cpus is invalid")
memory = limits.get("memory")
memory_swap = limits.get("memory_swap")
pids = limits.get("pids")
if (not isinstance(memory, str) or not memory.strip()
        or not isinstance(memory_swap, str) or not memory_swap.strip()
        or type(pids) is not int or pids <= 0):
    raise SystemExit("execution_policy.resource_limits is invalid")
PY
resource_profile_json=$(python3 "$source_root/resource_policy.py" \
  --config "$config_file" \
  --mode "$resource_mode" --format json)
read -r container_cpus container_memory container_memory_swap container_pids container_tmpfs \
  <<<"$(python3 "$source_root/resource_policy.py" \
    --config "$config_file" \
    --mode "$resource_mode" --format line)"
run_root="$output_root/runs/$run_id"
log_root="$run_root/wrapper-logs"
run_user=${RQ1_RUN_USER:-rvdata}
# comparison 需要 K1 时挂载 SSH；两个 Ours 方法默认使用 K1 reference。
framework_ours_method=false
if [[ "$action" == "framework-run" &&
      ( "$method_filter" == "Ours-RVGEN-Direct" ||
        "$method_filter" == "Ours-Program-Full" ) ]]; then
  framework_ours_method=true
fi
if [[ ( "$action" == "run" || "$action" == "execute-existing-queues" ||
        "$framework_ours_method" == true ) &&
      "$without_board" == false ]]; then
  with_board=true
fi
if [[ "$with_board" == true && "$without_board" == true ]]; then
  echo "--with-board and --without-board are mutually exclusive" >&2; exit 2
fi
coverage_enabled="${RQ1_ENABLE_COVERAGE:-1}"
case "$coverage_enabled" in 1|true|yes|0|false|no) ;; *) echo "invalid RQ1_ENABLE_COVERAGE" >&2; exit 2;; esac
defer_external_finalize=false
if [[ ( "$action" == "run" || "$action" == "execute-existing-queues" ) &&
      "$coverage_enabled" =~ ^(1|true|yes)$ ]]; then
  defer_external_finalize=true
fi
if [[ -z "${RQ1_IMAGE_REF:-}" && "$coverage_enabled" =~ ^(1|true|yes)$ ]]; then
  image_ref=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution_policy"]["coverage_container_image"])' "$config_file")
fi
[[ -z "$target_timeout_seconds" || "$target_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid target timeout' >&2; exit 2;
}
[[ "$finalize_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid finalizer timeout' >&2; exit 2;
}
[[ "$drain_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid drain timeout' >&2; exit 2;
}
[[ "$action" == "run" || "$action" == "execute-existing-queues" ||
   "$framework_ours_method" == true ||
   "$with_board" == false ]] || {
  echo '--with-board only applies to comparison execution' >&2; exit 2;
}
# comparison 面只有 5 个外部工具，K1 是可选证据臂（--without-board 合法）。
# 两个 Ours 方法的参考由在线框架处理：首个 K1 请求超时或传输不可用时固定回退到 T-QEMU。
# `--without-board` 只跳过 K1 挂载；实际参考来源仍记录在框架 campaign 账本。
run_uid=$(id -u "$run_user")
run_gid=$(id -g "$run_user")
[[ -d "$dependency_root/external" && -d "$dependency_root/toolchains" &&
   -d "$dependency_root/python" && -d "$dependency_root/target-store" ]] || {
  echo 'dependency root is incomplete' >&2; exit 2;
}
coverage_dependency_gaps=()
if [[ "$coverage_enabled" =~ ^(1|true|yes)$ ]]; then
  for coverage_dir in simulator-sources simulator-coverage-sources simulator-coverage-builds simulator-coverage-artifacts simulator-coverage-tools; do
    if [[ ! -d "$dependency_root/$coverage_dir" ]]; then
      coverage_dependency_gaps+=("$coverage_dir")
    fi
  done
fi
read -r needs_torture_cache < <(python3 - "$action" "$method_filter" <<'PY'
import sys

action, selected = sys.argv[1:]
need_torture = action in {"run", "generate"} and selected == "B-TORTURE"
print(str(need_torture).lower())
PY
)
execution_dependency_gaps=()
if [[ "$needs_torture_cache" == true && ! -d "$torture_cache" ]]; then
  execution_dependency_gaps+=("torture-cache")
  torture_cache=
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image_ref")
expected_image_id=$(RQ1_COVERAGE_ENABLED="$coverage_enabled" python3 -c 'import json,os,sys; p=json.load(open(sys.argv[1])); k="coverage_container_image_digest" if os.environ["RQ1_COVERAGE_ENABLED"].lower() in {"1","true","yes"} else "container_image_digest"; print(p["execution_policy"][k])' "$config_file")
[[ "$image_id" == "$expected_image_id" ]] || {
  echo "image identity mismatch: expected $expected_image_id got $image_id" >&2; exit 125;
}
network=$(python3 - "$config_file" "$method_filter" <<'PY'
import json, sys

config = json.load(open(sys.argv[1]))
ordinary = config.get("execution_policy", {}).get("network", "none")
selected = sys.argv[2]
if selected:
    declared = list(config.get("methods", [])) + list(config.get("framework_methods", []))
    method = next((item for item in declared
                   if isinstance(item, dict) and item.get("id") == selected), None)
    if method is None:
        raise SystemExit(f"unknown method: {selected}")
    print(method.get("network") or ordinary)
else:
    print(ordinary)
PY
)
method_count=$(python3 - "$config_file" "$action" "$method_filter" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
action, selected = sys.argv[2:]
pool = config.get("framework_methods", []) if action == "framework-run" else config.get("methods", [])
if action == "framework-run":
    if not selected or not any(item.get("id") == selected for item in pool if isinstance(item, dict)):
        raise SystemExit("framework-run requires one declared framework method")
    print(1)
elif selected:
    if not any(item.get("id") == selected for item in pool if isinstance(item, dict)):
        raise SystemExit("comparison action requires one declared external method")
    print(1)
else:
    print(len(pool))
PY
)
method_filter_json=null
if [[ -n "$method_filter" ]]; then method_filter_json="\"$method_filter\""; fi
case "$network" in
  bridge|none) ;;
  *) echo "unsupported ordinary-container network: $network" >&2; exit 2;;
esac
source_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
fuzz4all_commit=$(git -C "$dependency_root/external/fuzz4all" rev-parse HEAD 2>/dev/null || true)
dependency_identity_json=$(RQ1_DEPENDENCY_ACTION="$action" RQ1_DEPENDENCY_METHOD="$method_filter" python3 - "$config_file" "$dependency_root" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
action = os.environ.get("RQ1_DEPENDENCY_ACTION", "comparison")
selected_method = os.environ.get("RQ1_DEPENDENCY_METHOD", "")
method_pool = config.get("framework_methods", []) if action == "framework-run" else config.get("methods", [])
if selected_method:
    method_pool = [item for item in method_pool
                   if isinstance(item, dict) and item.get("id") == selected_method]
required = {
    name
    for method in method_pool
    if isinstance(method, dict)
    for name in method.get("requires", [])
    if isinstance(name, str)
}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

rows = {}
for name, pin in sorted(config.get("dependency_pins", {}).items()):
    if name not in required:
        continue
    kind = pin.get("kind")
    path = root / str(pin.get("path", ""))
    row = {"kind": kind, "path": str(path), "expected_commit": pin.get("commit"),
           "expected_sha256": pin.get("binary_sha256"), "head": None,
           "actual_sha256": None, "status": "unavailable", "dirty": False,
           "dirty_paths": [], "head_match": None, "digest_match": None,
           "identity_ok": False, "evidence": "wrapper-dependency-attestation-v1"}
    if kind == "git" and path.is_dir():
        try:
            head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False, timeout=10)
            status = subprocess.run(["git", "-C", str(path), "status", "--porcelain=v1",
                                     "--untracked-files=all"],
                                    capture_output=True, text=True, check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
            head = status = None
        if head is not None and status is not None and head.returncode == 0 and status.returncode == 0:
            dirty_paths = status.stdout.splitlines()
            head_value = head.stdout.strip() or None
            row.update(head=head_value, dirty=bool(dirty_paths), dirty_paths=dirty_paths[:20],
                       head_match=bool(pin.get("commit")) and head_value == pin.get("commit"),
                       status="dirty" if dirty_paths else ("clean" if head_value else "unavailable"))
            # dirty 只作 provenance；依赖身份由 pin/head 判定。
            row["identity_ok"] = bool(head_value and row["head_match"])
    elif kind == "binary" and path.is_file():
        try:
            actual = digest(path)
        except OSError:
            actual = None
        row.update(actual_sha256=actual, digest_match=actual == pin.get("binary_sha256"),
                   status="clean" if actual else "unavailable")
        row["identity_ok"] = bool(actual and row["digest_match"])
    rows[name] = row

payload = {"schema_version": "rq1-dependency-identity-v1", "root": str(root),
           "strict_clean": False, "dirty_is_telemetry": True,
           "scope": {"action": action, "method": selected_method or None,
                                             "required": sorted(required)},
           "dependencies": rows,
           "identity_ok": all(row["identity_ok"] for row in rows.values())}
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)
dependency_identity_ok=$(python3 -c 'import json,sys; print(str(bool(json.load(sys.stdin).get("identity_ok"))).lower())' <<<"$dependency_identity_json")
source_dirty=false
[[ -z "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]] || source_dirty=true
source_status_sha=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all \
  | sha256sum | awk '{print $1}')
config_sha=$(sha256sum "$config_file" | awk '{print $1}')
if [[ "$action" != "doctor" && "$action" != "framework-run" &&
      "$dependency_identity_ok" != true ]]; then
  echo "dependency identity is not pinned/available; refusing $action" >&2
  echo "$dependency_identity_json" >&2
  exit 125
fi
mkdir -p "$output_root/runs"
container_name="rq1cmp-$run_id"
container_owned=false
container_running=false
pause_signal_requested=false
pause_signal=
source_snapshot=
source_snapshot_ready=false
config_override_file=
rvvm_stop_flush_dir=
cleanup() {
  rm -f "$run_root/.rq1-run.lock"
  if [[ "$source_snapshot_ready" != true && -n "$source_snapshot" ]]; then
    chmod -R u+rwX -- "$source_snapshot" 2>/dev/null || true
    rm -rf -- "$source_snapshot"
  fi
  [[ -z "$rvvm_stop_flush_dir" ]] || rm -rf -- "$rvvm_stop_flush_dir"
  [[ -z "$config_override_file" ]] || rm -f -- "$config_override_file"
}
trap cleanup EXIT
write_pause_request() {
  [[ -d "$run_root" ]] || return 0
  python3 - "$run_root" "$pause_signal" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]) / "control"
root.mkdir(parents=True, exist_ok=True)
path = root / "request.json"
try:
    current = json.loads(path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    current = {}
sequence = current.get("sequence", 0) if isinstance(current, dict) else 0
if type(sequence) is not int or sequence < 0:
    raise SystemExit(f"invalid control sequence: {path}")
payload = {
    "schema_version": "rq1-case-boundary-request-v1",
    "sequence": sequence + 1,
    "desired_state": "pause",
    "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_signal": sys.argv[2],
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
os.replace(temporary, path)
PY
}
request_case_boundary_pause() {
  pause_signal_requested=true
  pause_signal=${1:-SIGTERM}
  if [[ "$container_owned" == true ]]; then
    write_pause_request
    echo "[$pause_signal] case-boundary pause requested; container will stop after the boundary" >&2
  fi
}
trap 'request_case_boundary_pause SIGINT' INT
trap 'request_case_boundary_pause SIGTERM' TERM
trap 'request_case_boundary_pause SIGHUP' HUP
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "container name already exists: $container_name" >&2
  exit 2
fi
source_snapshot_root="$output_root/source-snapshots"
mkdir -p "$source_snapshot_root"
source_snapshot="$source_snapshot_root/$run_id"
mkdir "$source_snapshot"
# The host worktree is shared with other tasks and may change while a run is
# alive. Mount a one-time control-plane snapshot into Docker instead.
cp -a --reflink=auto "$source_root"/. "$source_snapshot"/
expected_source_tree_sha=${RQ1_EXPECTED_SOURCE_TREE_SHA256:-}
if [[ -n "$expected_source_tree_sha" ]]; then
  [[ "$expected_source_tree_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'invalid expected source tree digest' >&2; exit 2;
  }
fi
source_after_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
source_after_status_sha=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all \
  | sha256sum | awk '{print $1}')
[[ "$source_after_commit" == "$source_commit" &&
   "$source_after_status_sha" == "$source_status_sha" ]] || {
  echo "source changed while creating the run snapshot" >&2
  exit 125
}
snapshot_config_file="$source_snapshot/config/rq1-comparison-isolated-v1.json"
snapshot_config_sha=$(sha256sum "$snapshot_config_file" | awk '{print $1}')
if [[ "$config_file" != "$snapshot_config_file" ]]; then
  # cp -a preserves the read-only control-plane modes used by the snapshot.
  # Make only the overridden file writable before replacing its contents.
  chmod u+w -- "$snapshot_config_file"
  cp -- "$config_file" "$snapshot_config_file"
fi
snapshot_config_sha=$(sha256sum "$snapshot_config_file" | awk '{print $1}')
[[ "$snapshot_config_sha" == "$config_sha" ]] || {
  echo "source/config changed while creating the run snapshot" >&2
  exit 125
}
actual_source_tree_sha=$(tar --sort=name --mtime='@0' --mode='a+rwX' \
  --owner=0 --group=0 --numeric-owner -cf - -C "$source_snapshot" . \
  | sha256sum | awk '{print $1}')
if [[ -n "$expected_source_tree_sha" ]]; then
  [[ "$actual_source_tree_sha" == "$expected_source_tree_sha" ]] || {
    echo 'source snapshot does not match the replay batch source digest' >&2
    exit 125
  }
fi
source_snapshot_ready=true
mkdir "$run_root" || { echo "run already exists: $run_root" >&2; exit 2; }
if [[ "$action" == "execute-existing-queues" ]]; then
  mkdir "$run_root/control"
  python3 - "$run_root/control/request.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "rq1-case-boundary-request-v1",
    "sequence": 0,
    "desired_state": "running",
}, indent=2) + "\n", encoding="utf-8")
PY
fi
if [[ -n "$fuzz4all_seed_source" ]]; then
  mkdir "$run_root/continuation-seed"
  cp -- "$fuzz4all_seed_source" "$run_root/continuation-seed/source.c"
  chmod 0444 "$run_root/continuation-seed/source.c"
  [[ "$(sha256sum "$run_root/continuation-seed/source.c" | awk '{print $1}')" == "$fuzz4all_seed_sha256" ]] || {
    echo 'Fuzz4All continuation seed changed while copying' >&2
    exit 125
  }
fi
if [[ -n "$rvdv_seed_manifest_source" ]]; then
  mkdir "$run_root/inputs"
  seed_manifest_snapshot="$run_root/inputs/rvdv-seed-catalog-manifest.snapshot.json"
  cp -- "$rvdv_seed_manifest_source" "$seed_manifest_snapshot"
  chmod 0444 "$seed_manifest_snapshot"
  [[ "$(sha256sum -- "$seed_manifest_snapshot" | awk '{print $1}')" == "$rvdv_seed_catalog_manifest_sha256" ]] || {
    echo 'RVDV seed catalog manifest changed while snapshotting' >&2
    exit 125
  }
fi
mkdir "$log_root"
tmp_root="$run_root/tmp"
mkdir "$tmp_root"
chown "$run_uid:$run_gid" "$run_root" "$log_root" "$tmp_root"
if [[ -d "$run_root/control" ]]; then
  chown "$run_uid:$run_gid" "$run_root/control" "$run_root/control/request.json"
fi
config_snapshot="$run_root/config.snapshot.json"
rvvm_stop_flush=
cp -- "$snapshot_config_file" "$config_snapshot"
chown "$run_uid:$run_gid" "$config_snapshot"
chmod -R a-w -- "$source_snapshot"
if [[ "$coverage_enabled" =~ ^(1|true|yes)$ ]]; then
  rvvm_stop_flush_dir=$(mktemp -d "/tmp/rq1-rvvm-stop-$run_id.XXXXXX")
  rvvm_stop_flush="$rvvm_stop_flush_dir/rvvm-stop-flush.so"
  if ! cc -O2 -fPIC -shared -pthread "$source_root/container/rvvm-stop-flush.c" -o "$rvvm_stop_flush"; then
    rm -f "$rvvm_stop_flush"
    rvvm_stop_flush=
  fi
fi
started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

PROGRAM_SEED_ARTIFACT_ROOT="$rvdv_seed_artifact_root" PROGRAM_SEED_CATALOG_RELATIVE="$rvdv_seed_catalog_relative" PROGRAM_SEED_MANIFEST_SHA256="$rvdv_seed_catalog_manifest_sha256" CASE_CATALOG_MANIFEST_HOST="$case_catalog_manifest_host" CASE_CATALOG_MANIFEST_SHA256="$case_catalog_manifest_sha256" SOURCE_ROOT="$source_root" SOURCE_SNAPSHOT="$source_snapshot" CONFIG_SNAPSHOT="$config_snapshot" DEPENDENCY_ROOT="$dependency_root" RUN_ROOT="$run_root" GENERATION_ROOT="$generation_root" SOURCE_TREE_SHA256="$actual_source_tree_sha" RUN_ID="$run_id" ACTION="$action" IMAGE_REF="$image_ref" IMAGE_ID="$image_id" CONTAINER_NAME="$container_name" SOURCE_COMMIT="$source_commit" SOURCE_DIRTY="$source_dirty" EXECUTION_DEPENDENCY_GAPS="${execution_dependency_gaps[*]}" STARTED_AT="$started_at_utc" DURATION_SECONDS="$duration_seconds" TARGET_TIMEOUT_SECONDS="$target_timeout_seconds" TORTURE_CACHE="$torture_cache" WITH_BOARD="$with_board" COVERAGE_ENABLED="$coverage_enabled" COVERAGE_DEPENDENCY_GAPS="${coverage_dependency_gaps[*]}" CONFIG_SHA256="$config_sha" NETWORK="$network" METHOD_COUNT="$method_count" METHOD_FILTER="$method_filter" FEEDBACK_TARGET="$feedback_target" MCMC_STEPS="$mcmc_steps" FUZZ4ALL_SEED_SOURCE="$fuzz4all_seed_source" FUZZ4ALL_SEED_SHA256="$fuzz4all_seed_sha256" FUZZ4ALL_PARENT_RUN_ID="$fuzz4all_parent_run_id" FUZZ4ALL_PARENT_CANDIDATE_INDEX="$fuzz4all_parent_candidate_index" GENERATION_START_INDEX="$generation_start_index" GENERATION_START_BATCH="$generation_start_batch" GENERATION_ATTEMPT_OFFSET="$generation_attempt_offset" GENERATION_CANDIDATE_OFFSET="$generation_candidate_offset" GENERATION_PARENT_RUN_ID="$generation_parent_run_id" GENERATION_PARENT_BATCH_ID="$generation_parent_batch_id" GENERATION_PARENT_SOURCE_SHA256="$generation_parent_source_sha256" DEPENDENCY_IDENTITY_JSON="$dependency_identity_json" RESOURCE_PROFILE_JSON="$resource_profile_json" python3 - "$run_root/execution-manifest.json" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
source = Path(os.environ["SOURCE_ROOT"])
source_host = source
source = Path(os.environ["SOURCE_SNAPSHOT"])
deps = Path(os.environ["DEPENDENCY_ROOT"])
config = json.loads(Path(os.environ["CONFIG_SNAPSHOT"]).read_text())
dependency_identity = json.loads(os.environ["DEPENDENCY_IDENTITY_JSON"])
applicability = config.get("applicability", {})
ordinary_network = config.get("execution_policy", {}).get("network", "none")

def source_identity(spec):
    container_root = spec.get("source_root")
    row = {
        "source_root": container_root,
        "expected_commit": spec.get("source_commit"),
        "host_source_root": None,
        "head": None,
        "status": "unavailable",
        "source_dirty": False,
        "dirty_path_count": 0,
        "non_generated_dirty_paths": [],
        "generated_untracked_path_count": 0,
        "evidence": "wrapper-host-git-attestation-v1",
    }
    if not isinstance(container_root, str) or not container_root.startswith("/path/to/deps/"):
        return row
    host_root = deps / container_root.removeprefix("/path/to/deps/")
    row["host_source_root"] = str(host_root)
    if not host_root.is_dir():
        return row
    try:
        head = subprocess.run(
            ["git", "-C", str(host_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", str(host_root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return row
    if head.returncode != 0 or status.returncode != 0:
        return row
    entries = [(line[:2], line[3:]) for line in status.stdout.splitlines() if len(line) >= 4]
    def generated_untracked(code, relative_path):
        if code != "??":
            return False
        relative = Path(relative_path.split(" -> ")[-1])
        if relative.name.endswith(".profraw"):
            return True
        if relative == Path("Cargo.lock.git-source"):
            try:
                with (host_root / relative).open(encoding="utf-8") as stream:
                    return stream.readline().rstrip("\r\n") == (
                        "# This file is automatically @generated by Cargo."
                    )
            except (OSError, UnicodeError):
                return False
        return False

    generated = [path for code, path in entries if generated_untracked(code, path)]
    generated_paths = set(generated)
    non_generated = [path for code, path in entries
                     if not (code == "??" and path in generated_paths)]
    row.update(
        head=head.stdout.strip() or None,
        status="clean" if head.stdout.strip() and not non_generated else "dirty",
        source_dirty=bool(non_generated),
        dirty_path_count=len(entries),
        non_generated_dirty_paths=non_generated[:20],
        generated_untracked_path_count=len(generated),
    )
    return row

target_specs = [target for target in config.get("targets", []) if isinstance(target, dict)]
if os.environ.get("ACTION") == "framework-run":
    feedback_target = os.environ.get("FEEDBACK_TARGET")
    if feedback_target != "all":
        target_specs = [target for target in target_specs
                        if target.get("id") == feedback_target]

coverage_source_identities = []
if os.environ.get("COVERAGE_ENABLED", "0").lower() in {"1", "true", "yes"}:
    coverage_source_identities = [
        source_identity(target.get("source_coverage") or {})
        for target in target_specs
    ]

method_pool = config.get("framework_methods", []) \
    if os.environ.get("ACTION") == "framework-run" else config.get("methods", [])
if os.environ.get("METHOD_FILTER"):
    method_pool = [method for method in method_pool
                   if isinstance(method, dict)
                   and method.get("id") == os.environ.get("METHOD_FILTER")]
methods = [{"id": method["id"], "route": method["route"], "lane": method["lane"],
            "network": method.get("network") or ordinary_network, "model": method.get("model"),
            "timeout_seconds": method.get("timeout_seconds", int(os.environ["DURATION_SECONDS"])),
            "generator_profile": method.get("generator_profile"),
            "feedback_targets": method.get("feedback_targets", []),
            "targets": (applicability.get(method["id"], {}).get("targets", [])
                        if os.environ.get("ACTION") != "framework-run"
                        or os.environ.get("FEEDBACK_TARGET") == "all"
                        else [os.environ.get("FEEDBACK_TARGET")])}
           for method in method_pool]
method_networks = {item["id"]: item["network"] for item in methods}
configured_network_shape = (
    "mixed" if len(set(method_networks.values())) > 1 else "uniform"
)
network_shape = "selected" if os.environ.get("METHOD_FILTER") else configured_network_shape
framework_reference_policy = None
if (
    os.environ.get("ACTION") == "framework-run"
    and os.environ.get("METHOD_FILTER") == "Ours-Program-Full"
):
    recipe_path = source / "framework" / "evidence" / "rvgen-rq1-program-recipes.json"
    recipe_ledger = json.loads(recipe_path.read_text(encoding="utf-8"))
    route_records = recipe_ledger.get("route_records", [])
    program_recipe = next(
        (row for row in route_records
         if isinstance(row, dict) and row.get("route_id") == "RQ1-RVGEN-PROGRAM"),
        None,
    )
    reference = program_recipe.get("reference") if isinstance(program_recipe, dict) else None
    fallback = reference.get("fallback") if isinstance(reference, dict) else None
    fallback_target = next(
        (row for row in config.get("targets", [])
         if isinstance(row, dict) and isinstance(fallback, dict)
         and row.get("id") == fallback.get("target_id")),
        None,
    )
    if not isinstance(reference, dict) or not isinstance(fallback, dict) \
            or not isinstance(fallback_target, dict):
        raise SystemExit("Program-Full reference fallback policy is incomplete")
    framework_reference_policy = {
        "preferred_backend": reference.get("backend"),
        "fallback_backend": fallback.get("backend"),
        "fallback_target_id": fallback.get("target_id"),
        "fallback_target_binary_sha256": fallback_target.get("binary_sha256"),
        "fallback_target_identity_digest": fallback_target.get("identity_digest"),
        "fallback_on": fallback.get("on"),
        "selection_scope": fallback.get("selection_scope"),
        "actual_selection_record": "framework/Ours-Program-Full/lanes/lane-*/cases/case-*/campaign-result.json",
    }
targets = [{"id": target["id"], "kind": target["kind"],
            "execution_model": target["execution_model"],
            "translation_mode": target.get("translation_mode"),
            "timeout_seconds": target.get("timeout_seconds"),
            "commit": target.get("commit"),
            "binary": target.get("binary"),
            "binary_sha256": target.get("binary_sha256"),
            "coverage_binary": target.get("coverage_binary"),
            "coverage_binary_sha256": target.get("coverage_binary_sha256"),
            "simulator_coverage": target.get("simulator_coverage"),
            "identity": target.get("identity"),
            "identity_digest": target.get("identity_digest")}
           for target in target_specs]

def mount(container, host, mode):
    return {"container": container, "host": str(host), "mode": mode}
mounts = [
    mount("/opt/rq1/comparison", source, "ro"),
    mount("/opt/runs/" + os.environ["RUN_ID"], Path(os.environ["RUN_ROOT"]), "rw"),
    mount("/path/to/deps/external", deps / "external", "ro"),
    mount("/path/to/deps/toolchains", deps / "toolchains", "ro"),
    mount("/path/to/deps/python", deps / "python", "ro"),
    mount("/path/to/deps/target-store", deps / "target-store", "ro"),
]
generation_root = os.environ.get("GENERATION_ROOT") or None
if generation_root:
    mounts.append(mount("/opt/generation", Path(generation_root), "ro"))
catalog_manifest_host = os.environ.get("CASE_CATALOG_MANIFEST_HOST") or None
catalog_manifest = None
if catalog_manifest_host:
    catalog_manifest = json.loads(Path(catalog_manifest_host).read_text(encoding="utf-8"))
    mounts.append(mount(
        "/opt/case-catalog-manifest.snapshot.json",
        Path(catalog_manifest_host), "ro",
    ))
program_seed_sequence_input = None
program_seed_artifact_root = os.environ.get("PROGRAM_SEED_ARTIFACT_ROOT") or None
program_seed_catalog_relative = os.environ.get("PROGRAM_SEED_CATALOG_RELATIVE") or None
if program_seed_artifact_root and program_seed_catalog_relative:
    program_seed_artifact_root = Path(program_seed_artifact_root).resolve(strict=True)
    program_seed_catalog_path = (
        program_seed_artifact_root / program_seed_catalog_relative
    ).resolve(strict=True)
    if not program_seed_catalog_path.is_relative_to(program_seed_artifact_root):
        raise SystemExit("Program-Full seed catalog escapes its artifact root")
    mounts.append(mount("/opt/case-artifacts", program_seed_artifact_root, "ro"))
    seed_snapshot = Path(os.environ["RUN_ROOT"]) / "inputs" / (
        "rvdv-seed-catalog-manifest.snapshot.json"
    )
    seed_manifest = json.loads(seed_snapshot.read_text(encoding="utf-8"))
    program_seed_sequence_input = {
        "method": "B-RVDV",
        "catalog_id": seed_manifest.get("catalog_id"),
        "catalog_revision": seed_manifest.get("revision"),
        "catalog_path": str(program_seed_catalog_path),
        "container_catalog_path": (
            "/opt/case-artifacts/" + program_seed_catalog_relative
        ),
        "manifest_snapshot": "inputs/rvdv-seed-catalog-manifest.snapshot.json",
        "catalog_manifest_sha256": os.environ.get("PROGRAM_SEED_MANIFEST_SHA256"),
        "case_count": seed_manifest.get("methods", {}).get("B-RVDV", {}).get("case_count"),
        "sequence_artifact": "framework/Ours-Program-Full/program-seed-sequence.json",
    }
fuzz4all_continuation_seed = None
if os.environ.get("FUZZ4ALL_SEED_SOURCE"):
    fuzz4all_continuation_seed = {
        "parent_run_id": os.environ["FUZZ4ALL_PARENT_RUN_ID"],
        "parent_candidate_index": int(os.environ["FUZZ4ALL_PARENT_CANDIDATE_INDEX"]),
        "source_sha256": os.environ["FUZZ4ALL_SEED_SHA256"],
        "run_artifact": "continuation-seed/source.c",
    }
generation_continuation = {
    "schema_version": "rq1-generation-continuation-v1",
    "parent_run_id": os.environ.get("GENERATION_PARENT_RUN_ID") or None,
    "parent_batch_id": os.environ.get("GENERATION_PARENT_BATCH_ID") or None,
    "parent_source_sha256": os.environ.get("GENERATION_PARENT_SOURCE_SHA256") or None,
    "start_index": int(os.environ["GENERATION_START_INDEX"])
    if os.environ.get("GENERATION_START_INDEX") else 0,
    "start_batch": int(os.environ["GENERATION_START_BATCH"])
    if os.environ.get("GENERATION_START_BATCH") else 0,
    "attempt_offset": int(os.environ["GENERATION_ATTEMPT_OFFSET"])
    if os.environ.get("GENERATION_ATTEMPT_OFFSET") else 0,
    "candidate_index_offset": int(os.environ["GENERATION_CANDIDATE_OFFSET"])
    if os.environ.get("GENERATION_CANDIDATE_OFFSET") else 0,
}
generation_continuation["enabled"] = any(
    generation_continuation[key] not in (None, 0)
    for key in ("parent_run_id", "parent_batch_id", "parent_source_sha256",
                "start_index", "start_batch", "attempt_offset",
                "candidate_index_offset")
)
if os.environ.get("COVERAGE_ENABLED", "0").lower() in {"1", "true", "yes"}:
    for name in ("simulator-sources", "simulator-coverage-sources",
                 "simulator-coverage-builds", "simulator-coverage-artifacts",
                 "simulator-coverage-tools"):
        mounts.append(mount("/path/to/deps/" + name, deps / name, "ro"))
    mounts.append(mount("/work", Path(os.environ["RUN_ROOT"]), "rw"))
if os.environ.get("TORTURE_CACHE"):
    mounts.append(mount("/opt/torture-cache", Path(os.environ["TORTURE_CACHE"]), "ro"))
if os.environ.get("WITH_BOARD") == "true":
    mounts.append(mount("/path/to/user/.ssh", Path("/path/to/user/.ssh"), "ro"))
    mounts.extend(mount(path, Path(path), "ro") for path in ("/etc/passwd", "/etc/group"))
try:
    resource_profile = json.loads(os.environ.get("RESOURCE_PROFILE_JSON", "{}"))
except (TypeError, ValueError):
    resource_profile = {}
profile_limits = resource_profile.get("limits") if isinstance(resource_profile, dict) else None
resource_limits = profile_limits if isinstance(profile_limits, dict) else config["execution_policy"]["resource_limits"]
target_timeout_override = os.environ.get("TARGET_TIMEOUT_SECONDS")
default_target_timeout = int(
    target_timeout_override or config["limits"]["target_timeout_seconds"]
)
target_deadlines = {
    str(target["id"]): default_target_timeout if target_timeout_override else int(
        target.get("timeout_seconds") or default_target_timeout
    )
    for target in target_specs
    if isinstance(target.get("id"), str)
}
doc = {
    "schema_version": "rq1-comparison-run-manifest-v1",
    "run_id": os.environ["RUN_ID"],
    "action": os.environ["ACTION"],
    "experiment_face": ("framework" if os.environ["ACTION"] == "framework-run" else "comparison"),
    "feedback_target": os.environ.get("FEEDBACK_TARGET") or None,
    "mcmc_steps_override": (
        int(os.environ["MCMC_STEPS"])
        if os.environ.get("MCMC_STEPS") else None
    ),
    "module_toggles": (
        {
            "small_model": os.environ.get("RQ1_FRAMEWORK_SMALL_MODEL_MODE") == "on",
            "emi": os.environ.get("RQ1_FRAMEWORK_EMI_MODE") == "on",
            "mcmc": os.environ.get("RQ1_FRAMEWORK_MCMC_MODE") == "on",
        }
        if os.environ.get("ACTION") == "framework-run" else None
    ),
    "framework_reference_policy": framework_reference_policy,
    "method_filter": os.environ.get("METHOD_FILTER") or None,
    "execution_plane": "docker-isolated",
    "method_isolation": {
        "mode": "one-method-per-container" if os.environ.get("METHOD_FILTER") else "diagnostic",
        "method": os.environ.get("METHOD_FILTER") or None,
        "method_count": len(methods),
    },
    "platform": {"host": "server-51", "container": "linux"},
    "source_root": str(source_host),
    "source_snapshot": "source-snapshot",
    "source_snapshot_host_root": os.environ.get("SOURCE_SNAPSHOT"),
    "source_tree_sha256": os.environ.get("SOURCE_TREE_SHA256") or None,
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_dirty": os.environ["SOURCE_DIRTY"] == "true",
    "experiment_seed": int(config.get("experiment_seed", 303)),
    "config_sha256": os.environ["CONFIG_SHA256"],
    "config_snapshot": "config.snapshot.json",
    "dependency_root": str(deps),
    "dependency_identity": dependency_identity,
    "dependency_identity_ok": dependency_identity.get("identity_ok") is True,
    "image_ref": os.environ["IMAGE_REF"],
    "image_id": os.environ["IMAGE_ID"],
    "container_name": os.environ["CONTAINER_NAME"],
    "network": os.environ["NETWORK"],
    "network_shape": network_shape,
    "configured_network_shape": configured_network_shape,
    "mounts": mounts,
    "isolation": {"rootfs_read_only": True, "ipc": "private",
                  "no_new_privileges": True, "cap_drop": ["ALL"],
                  "cap_add": ["DAC_OVERRIDE", "SETUID", "SETGID", "KILL"]},
    "resources": {
        "cpus": resource_limits["cpus"],
        "memory": resource_limits["memory"],
        "memory_swap": resource_limits["memory_swap"],
        "pids": resource_limits["pids"],
        "enforcement": "docker-cgroup",
    },
    "target_worker_count": int((config.get("execution_policy") or {}).get("target_worker_count") or 0),
    "resource_profile": resource_profile,
    "board_requested": os.environ.get("WITH_BOARD") == "true",
    "trusted_reference": {"id": "R-K1-BOARD", "engine": "k1-board",
                          "backend": "native-rv64", "execution_model": "ssh-linux-user+jtag-bare-metal"},
    "deadlines": {
        "target_seconds": default_target_timeout,
        "target_seconds_by_target": target_deadlines,
    },
    "target_execution_enabled": os.environ["ACTION"] != "generate",
    "generation_root": "/opt/generation" if generation_root else None,
    "generation_run_id": Path(generation_root).name if generation_root else None,
    "case_catalog_manifest": (
        {
            "container_path": "/opt/case-catalog-manifest.snapshot.json",
            "file_sha256": os.environ.get("CASE_CATALOG_MANIFEST_SHA256"),
            "catalog_revision": catalog_manifest.get("revision"),
            "catalog_sha256": catalog_manifest.get("catalog_sha256"),
        }
        if catalog_manifest is not None else None
    ),
    "program_seed_sequence_input": program_seed_sequence_input,
    "fuzz4all_continuation_seed": fuzz4all_continuation_seed,
    "generation_continuation": generation_continuation,
    "duration_seconds": int(os.environ["DURATION_SECONDS"]),
    "method_count": int(os.environ["METHOD_COUNT"]),
    "coverage_enabled": os.environ.get("COVERAGE_ENABLED", "1").lower() in {"1", "true", "yes"},
    "coverage_processing": (
        "raw-artifacts-only; aggregate offline"
        if os.environ.get("ACTION") in {"execute-existing-queues", "framework-run"}
        and os.environ.get("COVERAGE_ENABLED", "1").lower() in {"1", "true", "yes"}
        else "disabled-or-not-applicable"
    ),
    "rv_instruction_coverage": config.get("rv_instruction_coverage"),
    "rv_opcode_catalog_coverage": config.get("rv_opcode_catalog_coverage"),
    "execution_dependency_gaps": [item for item in os.environ.get("EXECUTION_DEPENDENCY_GAPS", "").split() if item],
    "coverage_dependency_gaps": [item for item in os.environ.get("COVERAGE_DEPENDENCY_GAPS", "").split() if item],
    "coverage_source_identities": coverage_source_identities,
    "methods": methods,
    "targets": targets,
    "started_at_utc": os.environ["STARTED_AT"],
}
if os.environ["ACTION"] == "generate":
    doc["deadlines"]["generation_seconds"] = int(os.environ["DURATION_SECONDS"])
Path(sys.argv[1]).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY

if [[ "$action" == "framework-run" ]]; then
  # Target runtime is a container-side concern.  The producer only needs the
  # queue root; each sibling service reads this immutable launch description
  # and owns its resident session.  A missing session_command is recorded as a
  # deployment gap by the service; it never becomes a per-job fallback.
  runtime_config_host="$run_root/framework/${method_filter//\//-}/target-runtime-configs.json"
  python3 - "$config_file" "$runtime_config_host" "$method_filter" "$feedback_target" "$run_id" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

config_path, output_path, method_id, feedback_target, run_id = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
method = next(
    (item for item in config.get("framework_methods", ())
     if isinstance(item, dict) and item.get("id") == method_id),
    {},
)
declared = method.get("feedback_targets", ())
declared = [str(item) for item in declared if isinstance(item, str)]
target_ids = declared if feedback_target == "all" else [feedback_target]
target_rows = {
    str(item.get("id")): item for item in config.get("targets", ())
    if isinstance(item, dict) and item.get("id") in target_ids
}

def command(value):
    if isinstance(value, (list, tuple)) and value:
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return None

def backend(row):
    kind = str(row.get("kind") or "")
    target_id = str(row.get("id") or "")
    return {
        "qemu": "qemu-riscv64",
        "libriscv": "libriscv-interpreter" if target_id.endswith("INT") else "libriscv-translated",
        "unicorn": "unicorn-riscv64",
        "renode": "renode-riscv64",
        "rax": "rax-riscv64",
        "rvvm": "rvvm-riscv64",
    }.get(kind, kind)

def service_command(target_id, row):
    service = row.get("service")
    value = row.get("session_command") or row.get("service_command")
    if isinstance(service, dict):
        value = value or service.get("command") or service.get("session_command")
    try:
        command_map = json.loads(os.environ.get("RQ1_TARGET_SESSION_COMMANDS_JSON", "{}"))
    except (TypeError, ValueError):
        command_map = {}
    if isinstance(command_map, dict):
        value = value or command_map.get(target_id)
    env_name = "RQ1_TARGET_SESSION_COMMAND_" + "".join(
        char if char.isalnum() else "_" for char in target_id.upper()
    )
    value = value or os.environ.get(env_name)
    parsed = command(value)
    if parsed is None and target_id == "T-UNICORN":
        parsed = [
            "python3", "-u", "/opt/rq1/comparison/framework/target_session_worker.py",
            "--backend", "unicorn-riscv64",
        ]
    if parsed is None and isinstance(row.get("binary"), str):
        # Keep one resident JSONL service per Target.  The worker reuses the
        # existing backend adapter for the six native binaries; no producer
        # callback or per-case service process is created here.
        parsed = [
            "python3", "-u", "/opt/rq1/comparison/framework/target_session_worker.py",
            "--backend", backend(row), "--binary", "{binary}",
        ]
    if parsed is None:
        return None
    binary = row.get("binary")
    binary_path = "/path/to/deps/" + str(binary) if isinstance(binary, str) else ""
    replacements = {
        "{target_id}": target_id,
        "{binary}": binary_path,
        "{run_root}": "/opt/runs/" + run_id,
        "{method_dir}": "/opt/runs/" + run_id + "/framework/" + method_id.replace("/", "-"),
    }
    return [replacements.get(item, item) for item in parsed]

targets = {}
for target_id in target_ids:
    row = target_rows.get(target_id, {})
    session_command = service_command(target_id, row)
    target_backend = backend(row)
    target_binary = (
        "/path/to/deps/" + str(row.get("binary"))
        if isinstance(row.get("binary"), str) else None
    )
    target_coverage_binary = (
        "/path/to/deps/" + str(row.get("coverage_binary"))
        if isinstance(row.get("coverage_binary"), str) else None
    )
    targets[target_id] = {
        "target_id": target_id,
        "backend": target_backend,
        "kind": row.get("kind"),
        "execution_model": row.get("execution_model"),
        "binary_path": target_binary,
        "identity": row.get("identity_digest"),
        "identity_path": "/path/to/deps/" + str(row.get("identity"))
        if isinstance(row.get("identity"), str) else None,
        "session_command": session_command,
        "timeout_seconds": row.get("timeout_seconds", 180),
        "coverage_config": {
            "enabled": os.environ.get("COVERAGE_ENABLED", "1").lower()
            in {"1", "true", "yes"},
            "backend": target_backend,
            "binary_path": target_coverage_binary or target_binary,
            "identity_binary_path": target_binary,
            "coverage_binary": target_coverage_binary,
            "coverage_binary_sha256": row.get("coverage_binary_sha256"),
            "coverage_binary_cache": (
                "/opt/runs/" + run_id + "/coverage-binaries/" + target_id
                if target_coverage_binary else None
            ),
            "simulator_coverage": row.get("simulator_coverage"),
        },
        "service_status": "configured"
        if session_command is not None else "missing-session-command",
    }
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps({
    "schema_version": "rq1-target-runtime-config-v1",
    "method": method_id,
    "feedback_target": feedback_target,
    "session_protocol": "rq1-target-session-v1",
    "rv_instruction_coverage": config.get("rv_instruction_coverage"),
    "rv_opcode_catalog_coverage": config.get("rv_opcode_catalog_coverage"),
    "targets": targets,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

args=(python3 /opt/rq1/comparison/runner.py "$action"   --config "/opt/runs/$run_id/config.snapshot.json"   --output "/opt/runs/$run_id")
# comparison run reads its shared deadline from the wrapper-written execution-manifest;
# framework-run receives the same explicit duration directly on the runner CLI.
if [[ ("$action" == "run" || "$action" == "execute-existing-queues" ||
      "$action" == "framework-run") &&
      -n "$target_timeout_seconds" ]]; then
  args+=(--target-timeout-seconds "$target_timeout_seconds")
fi
if [[ ( "$action" == "run" || "$action" == "execute-existing-queues" ) &&
      -n "$method_filter" ]]; then
  args+=(--method "$method_filter")
fi
if [[ "$action" == "execute-existing-queues" ]]; then
  if [[ -n "$case_catalog_root" ]]; then
    args+=(--case-catalog "/opt/generation/$(basename -- "$case_catalog_root")")
    if [[ -n "$case_catalog_manifest_host" ]]; then
      args+=(--case-catalog-manifest /opt/case-catalog-manifest.snapshot.json)
    fi
  else
    args+=(--generation-root /opt/generation)
  fi
fi
if [[ "$action" == "generate" ]]; then
  args+=(--method "$method_filter" --duration-seconds "$duration_seconds")
fi
if [[ "$action" == "framework-run" ]]; then
  args+=(--method "$method_filter" --duration-seconds "$duration_seconds")
  [[ -n "$feedback_target" ]] && args+=(--feedback-target "$feedback_target")
  [[ "$small_model_mode" == on ]] || args+=(--no-small-model)
  [[ "$emi_mode" == on ]] || args+=(--no-emi)
  [[ "$mcmc_mode" == on ]] || args+=(--no-mcmc)
  if [[ "$method_filter" == "Ours-Program-Full" ]]; then
    args+=(
      --program-seed-catalog "/opt/case-artifacts/$rvdv_seed_catalog_relative"
      --program-seed-catalog-manifest "/opt/runs/$run_id/inputs/rvdv-seed-catalog-manifest.snapshot.json"
    )
  fi
  producer_args=("${args[@]}")
  args=(
    bash /opt/rq1/comparison/container/framework-run-supervisor.sh
    --run-root "/opt/runs/$run_id"
    --method-dir "/opt/runs/$run_id/framework/${method_filter//\//-}"
    --config "/opt/runs/$run_id/config.snapshot.json"
    --method "$method_filter"
    --feedback-target "${feedback_target:-all}"
    --duration-seconds "$duration_seconds"
    --owner "framework-$run_id"
    -- "${producer_args[@]}"
  )
fi
mounts=(-v "$source_snapshot:/opt/rq1/comparison:ro"
  -v "$run_root:/opt/runs/$run_id:rw"
  -v "$dependency_root/external:/path/to/deps/external:ro"
  -v "$dependency_root/toolchains:/path/to/deps/toolchains:ro"
  -v "$dependency_root/python:/path/to/deps/python:ro"
  -v "$dependency_root/target-store:/path/to/deps/target-store:ro")
if [[ "$action" == "execute-existing-queues" ]]; then
  mounts+=(-v "$generation_root:/opt/generation:ro")
  if [[ -n "$case_catalog_manifest_host" ]]; then
    mounts+=(-v "$case_catalog_manifest_host:/opt/case-catalog-manifest.snapshot.json:ro")
  fi
fi
if [[ -n "$rvdv_seed_artifact_root" ]]; then
  mounts+=(-v "$rvdv_seed_artifact_root:/opt/case-artifacts:ro")
fi
# RAX validates the pinned upstream checkout at runtime even when the
# optional coverage side-channel is disabled; RVVM uses the same source path
# for its target identity.  Keep this small source mount in both modes.
if [[ -d "$dependency_root/simulator-sources" ]]; then
  mounts+=(-v "$dependency_root/simulator-sources:/path/to/deps/simulator-sources:ro")
fi
small_model_request_mode=${RQ1_SMALL_MODEL_REQUEST_MODE:-openai-compatible}
if [[ "$action" == "framework-run" && -z "${RQ1_SMALL_MODEL_REQUEST_MODE:-}" ]]; then
  small_model_request_mode=ollama
fi
extra_env=(-e RQ1_DEPS=/path/to/deps -e RQ1_RUN="/opt/runs/$run_id"
  -e RQ1_RESOURCE_MODE="$resource_mode" -e RQ1_RESOURCE_PROFILE_JSON="$resource_profile_json"
  -e TMPDIR="/opt/runs/$run_id/tmp"
  -e RQ1_EXECUTION_PLANE=rq1-comparison-docker -e RQ1_NETWORK="$network"
  -e OLLAMA_HOST="${OLLAMA_HOST:-http://133.133.135.123:11434}"
  -e RQ1_SMALL_MODEL="${RQ1_SMALL_MODEL-qwen3.8:latest}"
  -e RQ1_SMALL_MODEL_REQUEST_MODE="$small_model_request_mode"
  -e RQ1_SMALL_MODEL_OPENAI_BASE_URL -e RQ1_SMALL_MODEL_OPENAI_MODEL
  -e RQ1_SMALL_MODEL_OPENAI_API_KEY
  -e RQ1_SMALL_MODEL_DIGEST -e RQ1_SMALL_MODEL_TIMEOUT
  -e RQ1_SMALL_MODEL_BUDGET -e RQ1_SMALL_MODEL_MAX_CANDIDATES
  -e RQ1_SMALL_MODEL_RETRIES
  -e RQ1_SMALL_MODEL_NUM_CTX -e RQ1_SMALL_MODEL_NUM_PREDICT
  -e RQ1_SMALL_MODEL_TEMPERATURE
  -e RQ1_GENERATION_START_INDEX="$generation_start_index"
  -e RQ1_GENERATION_START_BATCH="$generation_start_batch"
  -e RQ1_GENERATION_ATTEMPT_OFFSET="$generation_attempt_offset"
  -e RQ1_GENERATION_CANDIDATE_OFFSET="$generation_candidate_offset"
  -e RQ1_GENERATION_PARENT_RUN_ID="$generation_parent_run_id"
  -e RQ1_GENERATION_PARENT_BATCH_ID="$generation_parent_batch_id"
  -e RQ1_GENERATION_PARENT_SOURCE_SHA256="$generation_parent_source_sha256"
  -e RQ1_SOURCE_COMMIT="$source_commit" -e RQ1_FUZZ4ALL_COMMIT="$fuzz4all_commit"
  -e RQ1_RUNTIME_MODE=docker-isolated -e RQ1_FRAMEWORK_LOCAL_ROOT=/path/to/deps
  -e MIGRATION_EXECUTION_PLANE=docker-x86_64-lab -e MIGRATION_CONTAINER_IMAGE="$image_ref"
  -e MIGRATION_CONTAINER_IMAGE_DIGEST="$image_id" -e RQ1_IMAGE_DIGEST="$image_id"
  -e RQ1_DOTNET_RUNTIME=/path/to/deps/toolchains/dotnet-8
  -e RQ1_DRAIN_TIMEOUT_SECONDS="$drain_timeout_seconds"
  -e PYTHONPATH=/opt/rq1/comparison/adapters/fuzz4all-remote:/path/to/deps/python:/opt/pymysql:/opt/rvdv-python:/opt/unicorn-2.1.3
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e DOTNET_CLI_HOME=/tmp
  -e RQ1_FRAMEWORK_RUNS_ROOT="/opt/runs/$run_id/framework-experiments/runs"
  -e DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 -e PYTHONDONTWRITEBYTECODE=1)
if [[ -n "$fuzz4all_seed_source" ]]; then
  extra_env+=(
    -e "RQ1_FUZZ4ALL_SEED_SOURCE=/opt/runs/$run_id/continuation-seed/source.c"
    -e "RQ1_FUZZ4ALL_PARENT_RUN_ID=$fuzz4all_parent_run_id"
    -e "RQ1_FUZZ4ALL_PARENT_CANDIDATE_INDEX=$fuzz4all_parent_candidate_index"
    -e "RQ1_FUZZ4ALL_SEED_SHA256=$fuzz4all_seed_sha256"
  )
fi
if [[ -n "$rvvm_stop_flush" ]]; then
  mounts+=(-v "$rvvm_stop_flush:/opt/rvvm-stop-flush.so:ro")
  extra_env+=(-e RQ1_RVVM_STOP_FLUSH=/opt/rvvm-stop-flush.so)
fi
if [[ -n "$torture_cache" ]]; then
  mounts+=(-v "$torture_cache:/opt/torture-cache:ro")
  extra_env+=(-e RQ1_TORTURE_CACHE=/opt/torture-cache)
fi
if [[ "$coverage_enabled" =~ ^(1|true|yes)$ ]]; then
  for coverage_dir in simulator-coverage-sources simulator-coverage-builds simulator-coverage-artifacts simulator-coverage-tools; do
    if [[ -d "$dependency_root/$coverage_dir" ]]; then
      mounts+=(-v "$dependency_root/$coverage_dir:/path/to/deps/$coverage_dir:ro")
    fi
  done
  mounts+=(-v "$run_root:/work:rw")
fi
if [[ "$coverage_enabled" == "1" || "$coverage_enabled" == "true" || "$coverage_enabled" == "yes" ]]; then
  extra_env+=(-e RQ1_ENABLE_COVERAGE=1)
else
  extra_env+=(-e RQ1_ENABLE_COVERAGE=0)
fi
if [[ "$action" == "framework-run" ]]; then
  # Ours 运行时仅保留原始 profile/trace，由离线流程统一归并。
  extra_env+=(-e RQ1_RAW_COVERAGE_ONLY=1)
fi
# Ours reference 优先连 K1；coverage 基础镜像不含 openssh-client，
# 只有启用 K1 时挂载宿主客户端；首个请求超时或传输不可用时由框架记录并执行 QEMU 回退。
if [[ "$with_board" == true ]]; then
  k1_reference_parallelism=${RQ1_NATIVE_REFERENCE_PARALLELISM:-8}
  [[ "$k1_reference_parallelism" =~ ^[1-9][0-9]*$ ]] || {
    echo 'RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer' >&2; exit 2;
  }
  if ((${#k1_reference_parallelism} > 1)) || ((k1_reference_parallelism > 8)); then
    k1_reference_parallelism=8
  fi
  native_riscv_sysroot=/usr/riscv64-linux-gnu
  native_riscv_gcc_support=/usr/lib/gcc-cross/riscv64-linux-gnu
  [[ -d "$native_riscv_sysroot" && -d "$native_riscv_gcc_support" ]] || {
    echo 'RISC-V cross compiler sysroot is unavailable for the K1 trace helper' >&2; exit 2;
  }
  source "$(dirname "${BASH_SOURCE[0]}")/k1-reference-locks.sh"
  k1_reference_lock_root="$output_root/.k1-reference-locks"
  k1_reference_lock_prepare "$output_root" "$run_uid" "$run_gid"
  mounts+=(
    -v /path/to/user/.ssh:/path/to/user/.ssh:ro
    -v /etc/passwd:/etc/passwd:ro
    -v /etc/group:/etc/group:ro
    -v /usr/bin/ssh:/usr/bin/ssh:ro
    -v /usr/lib/openssh:/usr/lib/openssh:ro
    -v "$native_riscv_sysroot:$native_riscv_sysroot:ro"
    -v "$native_riscv_gcc_support:$native_riscv_gcc_support:ro"
    -v "$k1_reference_lock_root:/run/rq1-native-reference:rw"
  )
  extra_env+=(
    -e RQ1_NATIVE_SSH_CONFIG=/path/to/ssh/config
    -e RQ1_NATIVE_KNOWN_HOSTS=/path/to/user/.ssh/known_hosts
    -e RQ1_NATIVE_SSH_ALIAS=k1-board
    -e RQ1_NATIVE_REMOTE_ROOT="/mnt/build/wangyang/runs/rq1-$run_id"
    -e RQ1_NATIVE_REFERENCE_LOCK_PATH=/run/rq1-native-reference/native-rv64.lock
    -e "RQ1_NATIVE_REFERENCE_PARALLELISM=$k1_reference_parallelism"
    -e RQ1_NATIVE_TRACE_HELPER_CACHE=1
  )
fi

wait_code=0
set +e
defer_finalize_value=0
if [[ "$defer_external_finalize" == true ]]; then defer_finalize_value=1; fi
docker run -d --name "$container_name" --user "$run_uid:$run_gid"   --cpus "$container_cpus" --memory "$container_memory" --memory-swap "$container_memory_swap" --pids-limit "$container_pids"   --workdir /opt/rq1/comparison --read-only --entrypoint ""   --tmpfs "/tmp:rw,nosuid,nodev,exec,size=$container_tmpfs" --network "$network" --ipc private   --log-driver json-file --log-opt max-size=50m --log-opt max-file=3   --security-opt no-new-privileges --cap-drop ALL --cap-add DAC_OVERRIDE   --cap-add SETUID --cap-add SETGID --cap-add KILL "${mounts[@]}" "${extra_env[@]}" -e "RQ1_DEFER_FINALIZE=$defer_finalize_value"   "$image_ref" "${args[@]}" > "$log_root/container.id"   2> "$log_root/container-launch.stderr"
launch_code=$?
code=$launch_code
if [[ "$launch_code" -eq 0 ]]; then
  container_owned=true
  # A signal can arrive while `docker run -d` is returning. Preserve it as a
  # control-file pause request once this wrapper owns the new container.
  if [[ "$pause_signal_requested" == true ]]; then
    write_pause_request
  fi
  if [[ "$action" == "framework-run" ]]; then
    # producer exit-code 只表示 producer 已经结束。暂停可以在这个 checkpoint
    # 冻结同一个容器；正常完成/stop 必须继续等待 supervisor 退出，因为它还要
    # drain Target queue、flush session、写 raw seal；coverage/mismatch 留给离线命令。
    producer_status="$run_root/framework-producer.exit-code"
    while [[ ! -s "$producer_status" ]]; do
      running=$(docker inspect --format '{{.State.Running}}' "$container_name" \
        2>/dev/null || printf 'false')
      if [[ "$running" != true ]]; then
        break
      fi
      sleep 1
    done
    if [[ -s "$producer_status" ]]; then
      final_pause=$(python3 - "$run_root" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / "control"
try:
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    request = {}
try:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    status = {}
print(str(
    isinstance(request, dict) and request.get("desired_state") == "pause"
    and isinstance(status, dict) and status.get("state") == "paused"
).lower())
PY
      )
      if [[ "$final_pause" == true ]]; then
        # Freeze only after the producer/supervisor handshake is durable. The
        # resident Target sessions and their cursors remain in the same
        # container; control-run.sh resume performs docker unpause.
        docker pause "$container_name" >/dev/null 2>&1 || true
        raw_code=$(tr -d '\r\n' < "$producer_status")
        if [[ "$raw_code" =~ ^-?[0-9]+$ ]]; then code=$raw_code; else code=125; fi
      else
        # The supervisor writes producer-status before its close barrier. Wait
        # for the container, not just that intermediate file, so the host never
        # races Target shutdown/finalization.
        docker wait "$container_name" > "$log_root/container.exit-code" &
        docker_wait_pid=$!
        while true; do
          if wait "$docker_wait_pid"; then wait_code=0; else wait_code=$?; fi
          kill -0 "$docker_wait_pid" 2>/dev/null || break
        done
        docker_wait_pid=
        if [[ -s "$log_root/container.exit-code" ]]; then
          raw_code=$(tr -d '\r\n' < "$log_root/container.exit-code")
          if [[ "$raw_code" =~ ^-?[0-9]+$ ]]; then code=$raw_code; else code=125; fi
        else
          code=125
        fi
      fi
    else
      code=125
    fi
    docker logs "$container_name" > "$log_root/container.stdout.log" \
      2> "$log_root/container.stderr.log" || true
    container_running=$(docker inspect --format '{{.State.Running}}' "$container_name" \
      2>/dev/null || printf 'false')
  else
    docker wait "$container_name" > "$log_root/container.exit-code" &
    docker_wait_pid=$!
    while true; do
      if wait "$docker_wait_pid"; then wait_code=0; else wait_code=$?; fi
      # The host signal handler may interrupt wait while the Docker wait process
      # is still alive; resume waiting instead of returning during a pause.
      kill -0 "$docker_wait_pid" 2>/dev/null || break
    done
    docker_wait_pid=
    if [[ -s "$log_root/container.exit-code" ]]; then
      raw_code=$(tr -d '\r\n' < "$log_root/container.exit-code")
      if [[ "$raw_code" =~ ^-?[0-9]+$ ]]; then code=$raw_code; else code=125; fi
    elif [[ "$wait_code" -eq 0 ]]; then
      code=125
    else
      code=$wait_code
    fi
    docker logs "$container_name" > "$log_root/container.stdout.log" \
      2> "$log_root/container.stderr.log" || true
    container_running=false
  fi
fi
set -e
if [[ "$container_owned" == true ]]; then
  docker inspect --format '{{json .State}}' "$container_name" \
    > "$run_root/container-inspect.json" 2>/dev/null || true
  read -r container_oom_killed container_state_exit_code container_running < <(python3 - "$run_root/container-inspect.json" <<'PY'
import json
import sys

try:
    state = json.load(open(sys.argv[1]))
    state = state if isinstance(state, dict) else {}
except (OSError, TypeError, ValueError):
    state = {}
print(
    str(state.get("OOMKilled") is True).lower(),
    state.get("ExitCode", ""),
    str(state.get("Running") is True).lower(),
)
PY
  )
  echo "container retained: $container_name" >&2
fi
container_oom_killed=${container_oom_killed:-false}
container_state_exit_code=${container_state_exit_code:-$code}
[[ "$container_state_exit_code" =~ ^-?[0-9]+$ ]] || container_state_exit_code=$code
sync_framework_failure_status() {
  [[ "$action" == framework-run ]] || return 0
  [[ "$container_running" == false ]] || return 0
  [[ "$container_state_exit_code" =~ ^[1-9][0-9]*$ ]] || return 0
  python3 - "$run_root" "$container_state_exit_code" "$container_oom_killed" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
path = root / "control" / "status.json"
try:
    status = json.loads(path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    status = {}
if not isinstance(status, dict) or status.get("state") in {"completed", "failed"}:
    raise SystemExit(0)
status.update(
    state="failed",
    failure_class="resource-limit" if sys.argv[3] == "true" else "container-exit",
    failure_reason="container-exit:%s" % sys.argv[2],
    container_exit_code=int(sys.argv[2]),
    container_oom_killed=sys.argv[3] == "true",
    updated_at_utc=datetime.now(timezone.utc).isoformat(),
)
temporary = path.with_name("." + path.name + ".failure.%s.tmp" % os.getpid())
temporary.write_text(
    json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}
sync_framework_failure_status
container_elapsed_seconds=$(python3 - "$run_root/container-inspect.json" <<'PY'
from datetime import datetime
import json
import sys

try:
    state = json.load(open(sys.argv[1]))
    started = datetime.fromisoformat(str(state["StartedAt"]).replace("Z", "+00:00"))
    finished = datetime.fromisoformat(str(state["FinishedAt"]).replace("Z", "+00:00"))
    print(max(0, int((finished - started).total_seconds())))
except (KeyError, OSError, TypeError, ValueError):
    print(0)
PY
)
[[ "$container_elapsed_seconds" =~ ^[0-9]+$ ]] || container_elapsed_seconds=0
board_status=independent-worker
board_result_file=board/
if [[ "$with_board" == false ]]; then board_status=disabled; fi
source_end_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
source_end_dirty=false
[[ -z "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]] || source_end_dirty=true
dependency_start_identity_ok="$dependency_identity_ok"
dependency_end_identity_json=$(DEPENDENCY_START_JSON="$dependency_identity_json" python3 - "$dependency_root" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

start = json.loads(os.environ["DEPENDENCY_START_JSON"])
rows = {}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

for name, original in start.get("dependencies", {}).items():
    row = dict(original) if isinstance(original, dict) else {}
    kind = row.get("kind")
    path = Path(str(row.get("path") or ""))
    row.update(head=None, actual_sha256=None, dirty=False, dirty_paths=[],
               head_match=None, digest_match=None, identity_ok=False,
               status="unavailable")
    if kind == "git" and path.is_dir():
        try:
            head = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False, timeout=10,
            )
            status = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain=v1",
                 "--untracked-files=all"],
                capture_output=True, text=True, check=False, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            head = status = None
        if head is not None and status is not None and head.returncode == 0 and status.returncode == 0:
            dirty_paths = status.stdout.splitlines()
            head_value = head.stdout.strip() or None
            row.update(
                head=head_value, dirty=bool(dirty_paths), dirty_paths=dirty_paths[:20],
                head_match=bool(row.get("expected_commit"))
                and head_value == row.get("expected_commit"),
                status="dirty" if dirty_paths else ("clean" if head_value else "unavailable"),
            )
            # dirty 只作 provenance；依赖身份由 pin/head 判定。
            row["identity_ok"] = bool(head_value and row["head_match"])
    elif kind == "binary" and path.is_file():
        try:
            actual = digest(path)
        except OSError:
            actual = None
        row.update(
            actual_sha256=actual,
            digest_match=actual == row.get("expected_sha256"),
            status="clean" if actual else "unavailable",
        )
        row["identity_ok"] = bool(actual and row["digest_match"])
    rows[name] = row

payload = dict(start)
payload.update(
    phase="end", dependencies=rows,
    identity_ok=all(row.get("identity_ok") is True for row in rows.values()),
)
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)
dependency_end_identity_ok=$(python3 -c 'import json,sys; print(str(bool(json.load(sys.stdin).get("identity_ok"))).lower())' <<<"$dependency_end_identity_json")
dependency_identity_stable=$(python3 - "$dependency_identity_json" "$dependency_end_identity_json" <<'PY'
import json
import sys

def comparable(payload):
    rows = {}
    for name, row in payload.get("dependencies", {}).items():
        if not isinstance(row, dict):
            rows[name] = row
            continue
        rows[name] = {
            key: row.get(key)
            for key in ("kind", "path", "expected_commit", "expected_sha256",
                        "head", "actual_sha256", "head_match", "digest_match")
        }
    return rows

start, end = map(json.loads, sys.argv[1:3])
print(str(comparable(start) == comparable(end)).lower())
PY
)
dependency_identity_ok="$dependency_start_identity_ok"
python3 - "$run_root/execution-manifest.json" "$source_end_commit" "$source_end_dirty" "$dependency_end_identity_json" "$dependency_end_identity_ok" "$dependency_identity_stable" "$dependency_identity_ok" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.update(
    source_end_commit=sys.argv[2],
    source_end_dirty=sys.argv[3] == "true",
    dependency_identity_end=json.loads(sys.argv[4]),
    dependency_identity_end_ok=sys.argv[5] == "true",
    dependency_identity_stable=sys.argv[6] == "true",
    dependency_identity_ok=sys.argv[7] == "true",
)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY
case "$action" in
  run) result_file=run-result.json;;
  execute-existing-queues) result_file=run-result.json;;
  framework-run) result_file=run-result.json;;
  generate) result_file=generation-result.json;;
  doctor) result_file=doctor.json;;
esac
framework_partial_reason() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def under_root(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("missing result path")
    path = (root / relative).resolve()
    path.relative_to(root)
    return path

try:
    control_status = load(root / "control" / "status.json")
    control_request = load(root / "control" / "request.json")
except (OSError, TypeError, ValueError):
    control_status = control_request = {}

if (
    isinstance(control_status, dict)
    and control_status.get("state") == "paused"
    and isinstance(control_request, dict)
    and control_request.get("desired_state") == "pause"
):
    print("framework-paused")
    raise SystemExit(0)

try:
    result = load(root / "run-result.json")
    if not isinstance(result, dict):
        raise ValueError("invalid framework result")
    chain = result.get("chain")
    if isinstance(chain, dict) and chain.get("mode") == "framework-target-matrix":
        target_results = result.get("target_results")
        if not isinstance(target_results, dict) or not target_results:
            raise ValueError("missing framework target matrix")
        stop_reasons = tuple(
            row.get("stop_reason") for row in target_results.values()
            if isinstance(row, dict)
        )
        pending = result.get("target_wall_clock_pending")
        if type(pending) is not int:
            pending = 0
            for row in target_results.values():
                if not isinstance(row, dict):
                    continue
                row_pending = row.get("target_wall_clock_pending")
                if type(row_pending) is int:
                    pending += max(0, row_pending)
                    continue
                try:
                    child = load(under_root(row.get("run_result")))
                except (OSError, TypeError, ValueError, KeyError):
                    child = {}
                if isinstance(child, dict) and type(child.get("target_wall_clock_pending")) is int:
                    pending += max(0, child["target_wall_clock_pending"])
        timebox_lane_stop = (
            type(pending) is int and pending > 0
            and stop_reasons
            and all(reason in {"deadline", "case-count"} for reason in stop_reasons)
            and not any(
                isinstance(row, dict) and row.get("status") == "gap"
                for row in target_results.values()
            )
        )
        if timebox_lane_stop or (
            stop_reasons and all(reason == "deadline" for reason in stop_reasons)
        ):
            print("framework-wall-clock-exhausted")
        elif any(str(reason).startswith("scheduler-error:") for reason in stop_reasons):
            print("framework-scheduler-error")
        elif any(isinstance(row, dict) and row.get("status") == "gap"
                 for row in target_results.values()):
            print("framework-case-failure")
        else:
            print("framework-lane-stop")
        raise SystemExit(0)
    if not isinstance(chain, dict) or chain.get("mode") != "parallel-lanes":
        raise ValueError("missing parallel chain")
    integrity_path = under_root(chain.get("integrity"))
    if not integrity_path.is_file():
        raise ValueError("missing parallel integrity record")
    integrity_record = load(integrity_path)
    if not isinstance(integrity_record, dict):
        raise ValueError("invalid parallel integrity record")
    parallel_path = integrity_path.parent / "parallel-result.json"
    parallel = load(parallel_path)
    if not isinstance(parallel, dict) or parallel.get("schema_version") != "rq1-framework-parallel-v1":
        raise ValueError("invalid parallel result")
    lanes = parallel.get("lanes")
    lane_count = parallel.get("parallel_lanes")
    if (
        type(lane_count) is not int
        or lane_count < 1
        or not isinstance(lanes, list)
        or len(lanes) != lane_count
    ):
        raise ValueError("incomplete lane list")

    parallel_chain = parallel.get("chain")
    if (
        not isinstance(parallel_chain, dict)
        or parallel_chain.get("mode") != "parallel-lanes"
        or parallel_chain.get("integrity") != chain.get("integrity")
        or type(parallel_chain.get("case_count")) is not int
        or type(parallel_chain.get("sealed")) is not bool
        or type(parallel_chain.get("integrity_verified")) is not bool
        or type(chain.get("integrity_verified")) is not bool
    ):
        raise ValueError("incomplete parallel chain record")

    lane_ids = set()
    stop_reasons = []
    total_cases = 0
    for lane in lanes:
        if not isinstance(lane, dict):
            raise ValueError("invalid lane record")
        lane_id = lane.get("lane_id")
        cases = lane.get("cases")
        stop_reason = lane.get("stop_reason")
        completed_cases = lane.get("completed_cases", 0)
        if (
            type(lane_id) is not int
            or lane_id in lane_ids
            or not isinstance(cases, list)
            or lane.get("status") != ("completed" if cases else "gap")
            or type(completed_cases) is not int
            or completed_cases != len(cases)
            or not isinstance(stop_reason, str)
            or not stop_reason
        ):
            raise ValueError("incomplete lane record")
        lane_ids.add(lane_id)
        stop_reasons.append(stop_reason)
        for case_index, case in enumerate(cases):
            if (
                not isinstance(case, dict)
                or type(case.get("lane_id")) is not int
                or case["lane_id"] != lane_id
                or type(case.get("case_index")) is not int
                or case["case_index"] != case_index
            ):
                raise ValueError("invalid case record")
            case_dir = under_root(case.get("case_dir"))
            case_result = under_root(case.get("case_result"))
            if not case_dir.is_dir() or not case_result.is_file() or case_result.parent != case_dir:
                raise ValueError("missing case record")
            if load(case_result) != case:
                raise ValueError("case record mismatch")
        total_cases += len(cases)

    expected_lane_ids = set(range(lane_count))
    if lane_ids != expected_lane_ids or type(parallel.get("case_count")) is not int:
        raise ValueError("incomplete parallel record")
    if parallel["case_count"] != total_cases or parallel_chain["case_count"] != total_cases:
        raise ValueError("parallel case count mismatch")

    pending = parallel.get("target_wall_clock_pending")
    cases = [
        case for lane in lanes for case in lane.get("cases", ())
        if isinstance(case, dict)
    ]
    has_right_censored_case = any(
        case.get("right_censored") is True
        or case.get("case_outcome") == "right-censored"
        for case in cases
    )
    has_observed_gap = any(
        case.get("case_outcome") in {"case-timeout", "case-failure"}
        or case.get("status") in {"gap", "failed"}
        or (
            isinstance(case.get("counts"), dict)
            and type(case["counts"].get("target_gap")) is int
            and case["counts"].get("target_gap") > 0
        )
        for case in cases
    )
    timebox_lane_stop = (
        type(pending) is int and pending > 0
        and stop_reasons
        and all(reason in {"deadline", "case-count"} for reason in stop_reasons)
        and has_right_censored_case
        and not has_observed_gap
    )
    if timebox_lane_stop or all(reason == "deadline" for reason in stop_reasons):
        if (
            chain["integrity_verified"] is True
            and parallel_chain["integrity_verified"] is True
            and integrity_record.get("status") == "verified"
            and type(integrity_record.get("case_count")) is int
            and integrity_record.get("case_count") == total_cases
        ):
            print("framework-wall-clock-exhausted")
        else:
            print("framework-record-gap")
    elif any(reason.startswith("scheduler-error:") for reason in stop_reasons):
        print("framework-scheduler-error")
    elif "case-failure" in stop_reasons:
        print("framework-case-failure")
    elif "case-timeout" in stop_reasons:
        print("framework-case-timeout")
    else:
        print("framework-lane-stop")
except (OSError, RuntimeError, TypeError, ValueError, KeyError):
    print("framework-record-gap")
PY
}
wrapper_code=$code
reason_code=run-complete
phase_status=unknown
phase_sealed=false
coverage_status=unknown
if [[ -f "$run_root/$result_file" ]]; then
  read -r phase_status phase_sealed coverage_status < <(python3 - "$run_root/$result_file" <<'PY'
import json, sys
try:
    payload = json.loads(open(sys.argv[1]).read())
except (OSError, TypeError, ValueError):
    payload = {}
sealed = payload.get("execution_complete") is True
if payload.get("phase") == "generation-only":
    sealed = payload.get("generation_sealed") is True
elif payload.get("experiment_face") == "framework":
    chain = payload.get("chain")
    sealed = (
        isinstance(chain, dict)
        and chain.get("sealed") is True
        and chain.get("integrity_verified") is True
    )
print(
    payload.get("status") or "invalid", str(sealed).lower(),
    payload.get("coverage_status") or "unknown",
)
PY
  )
fi
# comparison 的固定时长结束可能由 runner 以 125 返回；只要它已经留下
# 可重算的 partial 账本，这属于 right-censor，不是容器崩溃。保留这些
# 证据供同一主 Docker 的收尾逻辑和宿主 wrapper 继续审计。
expected_partial=false
external_interruption=false
framework_partial_classified=false
framework_stop_reason_code=
partial_artifacts=false
framework_matrix_artifacts=false
framework_single_artifacts=false
framework_shared_artifacts=false
framework_case_artifacts=false
framework_pause_boundary=false
framework_has_case_dirs() {
  local case_dir
  for case_dir in "$run_root/framework/$method_filter/lanes"/lane-*/cases/case-*; do
    [[ -d "$case_dir" ]] && return 0
  done
  return 1
}
if [[ "$action" == "framework-run" && "$feedback_target" == "all" &&
      -d "$run_root/framework-targets/$method_filter" ]] &&
   compgen -G "$run_root/framework-targets/$method_filter/*/execution-manifest.json" >/dev/null; then
  framework_matrix_artifacts=true
fi
if [[ "$action" == "framework-run" &&
      -d "$run_root/framework/$method_filter/lanes" ]] &&
   framework_has_case_dirs; then
  framework_case_artifacts=true
fi
if [[ "$action" == "framework-run" && "$feedback_target" != "all" &&
      -d "$run_root/framework/$method_filter/lanes" ]] &&
   { compgen -G "$run_root/framework/$method_filter/lanes/*/lane-result.json" >/dev/null ||
     [[ "$framework_case_artifacts" == true ]]; }; then
  # 单 Target 运行也可能只留下孤立 case 目录；它足以触发宿主恢复。
  framework_single_artifacts=true
fi
if [[ "$action" == "framework-run" && "$feedback_target" == "all" &&
      -d "$run_root/framework/$method_filter/lanes" ]] &&
   { compgen -G "$run_root/framework/$method_filter/lanes/*/lane-result.json" >/dev/null ||
     [[ "$framework_case_artifacts" == true ]]; }; then
  framework_shared_artifacts=true
fi
if [[ -f "$run_root/run-result.json" || -f "$run_root/execution-result.partial.json" ||
      -f "$run_root/generation-result.json" || -f "$run_root/generation-result.partial.json" ||
      -f "$run_root/generation/candidate-queues.partial.json" ||
      -f "$run_root/ledger/events.partial.jsonl" ||
      "$framework_matrix_artifacts" == true ||
      "$framework_single_artifacts" == true ||
      "$framework_shared_artifacts" == true ||
      "$framework_case_artifacts" == true ]]; then
  partial_artifacts=true
fi
if [[ "$action" == "framework-run" &&
      "$(framework_partial_reason "$run_root" 2>/dev/null || true)" == "framework-paused" ]]; then
  # Docker may report the outer TERM as 143 even though the producer already
  # reached its checkpoint.  That is a requested framework pause, not an
  # external interruption or a failed container.
  framework_pause_boundary=true
fi
if [[ "$container_oom_killed" != true &&
        "$wrapper_code" =~ ^(130|131|137|143)$ &&
      "$partial_artifacts" == true &&
      "$framework_pause_boundary" != true ]]; then
  # A signal outside the wrapper watchdog is not evidence of a deadline.  Keep
  # it as a separate recovery provenance and let the host finalizer close the
  # admitted queue suffix as pending.
  external_interruption=true
  expected_partial=true
  reason_code=external-signal
  python3 - "$run_root" "$wrapper_code" "$container_oom_killed" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
code = int(sys.argv[2])
payload = {
    "schema_version": "rq1-external-interruption-v1",
    "status": "partial",
    "reason_code": "external-signal",
    "container_exit_code": code,
    "signal": code - 128 if code >= 128 else None,
    "oom_killed": sys.argv[3] == "true",
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}
(root / "external-interruption.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
fi
if [[ "$comparison_execution_action" == true ]] &&
   [[ "$container_oom_killed" == true && "$partial_artifacts" == true ]]; then
  # OOM 是资源边界，不是外部 signal；若 partial 账本存在，允许恢复已落盘
  # 的观察并把未闭合队列保留为 pending，但绝不提升为 completed。
  expected_partial=true
  reason_code=resource-limit
fi
if [[ "$comparison_execution_action" == true &&
      "$phase_status" == partial &&
      ("$wrapper_code" -eq 0 || "$wrapper_code" -eq 125) &&
      ( -f "$run_root/run-result.json" || -f "$run_root/execution-result.partial.json" ||
        -f "$run_root/ledger/events.partial.jsonl" ) ]]; then
  expected_partial=true
fi
# framework 的 chain 可以已经完整封存，而统一 simulator source coverage
# 仍是具名 gap。这个状态是可审计的部分结果，不应被 wrapper 当作容器失败
# 再把 chain/integrity/seal 改写成 gap。
if [[ "$action" == "framework-run" && "$phase_status" == partial &&
      ("$wrapper_code" -eq 0 || "$wrapper_code" -eq 125 ||
       "$framework_pause_boundary" == true) &&
      "$container_oom_killed" != true && "$external_interruption" != true &&
      -f "$run_root/run-result.json" ]]; then
  # 只有完整 parallel-result 中每条 lane 都明确因共享 deadline 停止，
  # 才把 partial 作为预期删失；单 case timeout、case failure 和调度错误保留原因。
  framework_partial_classified=true
  framework_stop_reason_code=$(framework_partial_reason "$run_root")
  reason_code="$framework_stop_reason_code"
  if [[ "$framework_stop_reason_code" == framework-wall-clock-exhausted ||
        "$framework_stop_reason_code" == framework-paused ]]; then
    expected_partial=true
  fi
  # A normal framework timebox/pause is resumable partial progress, not a
  # failed container.  Keep the wrapper successful so callers can distinguish
  # it from a crashed producer; the result status remains ``partial``.
  if [[ "$expected_partial" == true ]]; then
    wrapper_code=0
  else
    wrapper_code=125
  fi
fi
if [[ "$action" == "framework-run" && "$phase_status" == partial && "$phase_sealed" == true &&
      -f "$run_root/run-result.json" ]]; then
  read -r framework_coverage_status framework_coverage_gap < <(python3 - "$run_root/run-result.json" <<'PY'
import json, sys
try:
    payload = json.loads(open(sys.argv[1]).read())
except (OSError, TypeError, ValueError):
    payload = {}
print(payload.get("coverage_status") or "", payload.get("coverage_gap") or "")
PY
  )
  if [[ "$framework_coverage_status" == gap && -n "$framework_coverage_gap" ]]; then
    expected_partial=true
    reason_code=framework-coverage-gap
  fi
fi
if [[ "$wrapper_code" -eq 0 && "$action" == "generate" ]]; then
  if [[ ( "$phase_status" == "completed" || "$phase_status" == "partial" ) &&
        "$phase_sealed" == true ]]; then
    if [[ "$phase_status" == "partial" ]]; then
      expected_partial=true
      reason_code=generation-partial
    else
      reason_code=generation-complete
    fi
  else
    wrapper_code=125
    reason_code=generation-unsealed
  fi
fi
if [[ "$wrapper_code" -eq 0 &&
      "$comparison_execution_action" == true &&
      ("$phase_status" == "unknown" || "$phase_status" == "invalid") ]]; then
  wrapper_code=125
  if [[ "$phase_status" == "unknown" ]]; then
    reason_code=missing-run-result
  else
    reason_code=invalid-run-result
  fi
fi
if [[ "$wrapper_code" -eq 0 && "$comparison_execution_action" == true && "$phase_status" == "partial" ]]; then
  wrapper_code=125
  reason_code=partial-execution
fi
if [[ "$wrapper_code" -eq 0 && "$action" == "framework-run" && "$expected_partial" != true &&
      ("$phase_status" != "passed" || "$phase_sealed" != true) ]]; then
  reason_code=partial-framework
  wrapper_code=125
fi
if [[ "$wrapper_code" -eq 0 && "$action" == "framework-run" &&
      "$phase_status" == "passed" && -f "$run_root/run-result.json" ]]; then
  read -r framework_execution_complete framework_coverage_status framework_coverage_complete framework_raw_only framework_raw_only_reason < <(python3 - "$run_root/run-result.json" <<'PY'
import json, sys

try:
    payload = json.load(open(sys.argv[1]))
except (OSError, TypeError, ValueError):
    payload = {}
framework_coverage = payload.get("framework_coverage")
if not isinstance(framework_coverage, dict):
    framework_coverage = {}
print(
    str(payload.get("execution_complete") is True).lower(),
    str(payload.get("coverage_status") or "").lower(),
    str(payload.get("coverage_artifact_complete") is True).lower(),
    str(payload.get("raw_coverage_only") is True).lower(),
    str(framework_coverage.get("reason") or ""),
)
PY
  )
  if [[ "$framework_execution_complete" != true ]]; then
    wrapper_code=125
    reason_code=partial-framework
  elif [[ "$coverage_enabled" =~ ^(1|true|yes)$ &&
          "$framework_coverage_complete" != true &&
          ( "$framework_raw_only" != true ||
            "$framework_coverage_status" != deferred ||
            "$framework_raw_only_reason" != raw-coverage-only ) ]]; then
    wrapper_code=125
    reason_code=framework-coverage-gap
  elif [[ "$coverage_enabled" =~ ^(0|false|no)$ ]]; then
    reason_code=coverage-disabled
  else
    reason_code=run-complete
  fi
fi
if [[ "$wrapper_code" -ne 0 && "$expected_partial" != true ]]; then
  if [[ "$reason_code" == "run-complete" ]]; then
    reason_code=container-run-failed
    if [[ "$wrapper_code" -eq 137 ]]; then reason_code=container-signal-killed; fi
  fi
  if [[ "$container_state_exit_code" -eq 137 && "$container_oom_killed" != true ]]; then
    reason_code=container-signal-killed
  fi
  if [[ "$comparison_execution_action" == true && -f "$run_root/derived/campaign-complete.json" ]] &&
     grep -q '"status"[[:space:]]*:[[:space:]]*"gap"' "$run_root/derived/campaign-complete.json"; then
    reason_code=campaign-seal-gap
  fi
fi
# 所有非正常停止都写一份统一的来源证据。external marker 保留
# 更细的旧 schema；该文件用于 runner 在 OOM、容器故障或 runner 自身 partial
# 返回时选择正确的恢复原因，避免把不同根因折叠成 external-interruption。
stop_reason_code=
if [[ "$container_oom_killed" == true ]]; then
  stop_reason_code=resource-limit
elif [[ "$external_interruption" == true ]]; then
  stop_reason_code=external-signal
elif [[ "$comparison_execution_action" == true &&
       "$framework_partial_classified" != true ]] &&
     [[ "$phase_status" == partial && "$partial_artifacts" == true ]]; then
  if [[ "$wrapper_code" -ne 125 ]]; then
    stop_reason_code=container-failure
  else
    stop_reason_code=$(python3 - "$run_root" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
for relative in ("run-result.json", "execution-result.partial.json"):
    path = root / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        continue
    for field in ("partial_reason", "reason_code", "stop_reason"):
        value = payload.get(field)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:-]+", value):
            print(value)
            raise SystemExit(0)
print("partial-execution")
PY
    )
  fi
elif [[ "$action" == "generate" && "$phase_status" == partial &&
        "$phase_sealed" == true ]]; then
  stop_reason_code=generation-partial
elif [[ "$action" == "framework-run" && "$framework_partial_classified" == true ]]; then
  stop_reason_code="${framework_stop_reason_code:-$reason_code}"
elif [[ "$wrapper_code" -ne 0 ]]; then
  stop_reason_code=container-failure
fi
if [[ -n "$stop_reason_code" &&
      ( "$expected_partial" == true || "$wrapper_code" -ne 0 ||
        "$container_oom_killed" == true ) &&
      ! -f "$run_root/stop-provenance.json" ]]; then
  python3 - "$run_root" "$stop_reason_code" "$wrapper_code" "$container_state_exit_code" \
      "$container_oom_killed" "$external_interruption" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": "rq1-stop-provenance-v1",
    "status": "partial" if (
        (root / "run-result.json").is_file()
        or (root / "generation-result.json").is_file()
        or (root / "generation-result.partial.json").is_file()
        or (root / "execution-result.partial.json").is_file()
        or (root / "ledger/events.partial.jsonl").is_file()
        or (root / "wall-clock-exhaustion.json").is_file()
    ) else "failed",
    "reason_code": sys.argv[2],
    "wrapper_exit_code": int(sys.argv[3]),
    "container_exit_code": (
        int(sys.argv[4]) if sys.argv[4].lstrip("-").isdigit() else None
    ),
    "oom_killed": sys.argv[5] == "true",
    "external_interruption": sys.argv[6] == "true",
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}
(root / "stop-provenance.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)
PY
fi
execution_complete=false
formal_ready=false
if [[ "$defer_external_finalize" == true && -f "$run_root/run-result.json" ]]; then
  read -r execution_complete coverage_status < <(python3 - "$run_root/run-result.json" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, TypeError, ValueError):
    payload = {}
print(
    str(payload.get("execution_complete") is True).lower(),
    payload.get("coverage_status") or "unknown",
)
PY
  )
  if [[ "$wrapper_code" -eq 0 && "$execution_complete" == true &&
        "$coverage_status" == deferred ]]; then
    reason_code=coverage-deferred
  fi
fi
if [[ "$action" == "framework-run" && -f "$run_root/run-result.json" ]]; then
  read -r execution_complete formal_ready < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("execution_complete") is True).lower(), str(p.get("formal_ready", False)).lower())' "$run_root/run-result.json")
fi
if [[ "$comparison_execution_action" == true && -f "$run_root/derived/campaign-complete.json" ]]; then
  read -r execution_complete formal_ready < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("execution_complete", p.get("status") == "complete")).lower(), str(p.get("formal_ready", False)).lower())' "$run_root/derived/campaign-complete.json")
  if [[ "$wrapper_code" -eq 0 && "$execution_complete" == true && "$formal_ready" != true ]]; then
    reason_code=reference-gap
  elif [[ "$wrapper_code" -eq 0 && "$execution_complete" != true ]]; then
    reason_code=partial-execution
  fi
fi
if [[ "$dependency_identity_ok" != true && "$action" != "framework-run" ]]; then
  wrapper_code=125
  reason_code=dependency-identity-drift
fi
if [[ "$wrapper_code" -ne 0 && "$expected_partial" != true ]]; then
  execution_complete=false
  formal_ready=false
fi

write_wrapper_result() {
  local wrapper_status=completed
  local wrapper_reason_code="$reason_code"
  if [[ -f "$run_root/run-result.json" ]]; then
    coverage_status=$(python3 - "$run_root/run-result.json" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, TypeError, ValueError):
    payload = {}
print(payload.get("coverage_status") or "unknown")
PY
    )
  fi
  case "$coverage_status" in
    recorded|partial|deferred|disabled|pending|gap) ;;
    *) coverage_status=unknown;;
  esac
  if [[ "$expected_partial" == true && "$dependency_identity_ok" == true ]]; then
    wrapper_status=partial
  elif [[ $wrapper_code -ne 0 ]]; then
    wrapper_status=failed
  fi
  if [[ -f "$run_root/stop-provenance.json" ]]; then
    local normalized_reason_code
    normalized_reason_code=$(python3 - "$run_root/stop-provenance.json" <<'PY'
import json
import re
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    payload = {}
value = payload.get("reason_code") if isinstance(payload, dict) else None
if (
    isinstance(payload, dict)
    and payload.get("schema_version") == "rq1-stop-provenance-v1"
    and isinstance(value, str)
    and re.fullmatch(r"[A-Za-z0-9._:-]+", value)
):
    print(value)
PY
    )
    if [[ -n "$normalized_reason_code" ]]; then
      wrapper_reason_code="$normalized_reason_code"
    fi
  fi
  local stop_provenance=normal
  case "$wrapper_reason_code" in
    resource-limit) stop_provenance=resource-limit;;
    external-signal) stop_provenance=external-signal;;
    framework-wall-clock-exhausted) stop_provenance=framework-wall-clock;;
    framework-paused) stop_provenance=framework-paused;;
    run-wall-clock-exhausted|execution-wall-clock-exhausted) stop_provenance=run-wall-clock;;
    framework-coverage-gap) stop_provenance=framework-coverage-gap;;
    coverage-gap) stop_provenance=coverage-gap;;
    generation-partial) stop_provenance=generation-budget;;
    framework-case-timeout|case-timeout) stop_provenance=framework-case-timeout;;
    framework-case-failure|case-failure) stop_provenance=framework-case-failure;;
    framework-scheduler-error|scheduler-error|scheduler-error:*) stop_provenance=framework-scheduler-error;;
    source-commit-drift) stop_provenance=source-identity-change;;
    source-identity-unavailable) stop_provenance=source-identity-gap;;
    framework-record-gap|framework-lane-stop) stop_provenance=framework-gap;;
    *)
      if [[ "$container_oom_killed" == true ]]; then
        stop_provenance=resource-limit
      elif [[ "$external_interruption" == true ]]; then
        stop_provenance=external-signal
      elif [[ "$container_state_exit_code" -eq 137 ]]; then
        stop_provenance=container-signal-killed
      elif [[ "$wrapper_code" -ne 0 ]]; then
        stop_provenance=container-failure
      fi
      ;;
  esac
  printf '{"schema_version":"rq1-comparison-wrapper-result-v1","run_id":"%s","result_file":"%s","status":"%s","exit_code":%d,"container_name":"%s","image_id":"%s","config_sha256":"%s","phase_status":"%s","phase_sealed":%s,"duration_seconds":%d,"method_filter":%s,"method_count":%d,"source_commit":"%s","source_end_commit":"%s","source_dirty":%s,"source_end_dirty":%s,"dependency_identity_ok":%s,"dependency_identity_end_ok":%s,"dependency_identity_stable":%s,"board_requested":%s,"board_status":"%s","board_result_file":"%s","execution_complete":%s,"coverage_status":"%s","formal_ready":%s,"reason_code":"%s","stop_provenance":"%s","container_exit_code":%d,"container_oom_killed":%s,"container_running":%s,"finished_at_utc":"%s"}\n' \
    "$run_id" "$result_file" "$wrapper_status" "$wrapper_code" "$container_name" "$image_id" "$config_sha" "$phase_status" "$phase_sealed" "$duration_seconds" "$method_filter_json" "$method_count" "$source_commit" "$source_end_commit" "$source_dirty" "$source_end_dirty" "$dependency_identity_ok" "$dependency_end_identity_ok" "$dependency_identity_stable" "$with_board" "$board_status" "$board_result_file" "$execution_complete" "$coverage_status" "$formal_ready" "$wrapper_reason_code" "$stop_provenance" "$container_state_exit_code" "$container_oom_killed" "$container_running" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_root/wrapper-result.json"
}

sync_stop_provenance_status() {
  [[ "$external_interruption" == true && -f "$run_root/stop-provenance.json" ]] || return 0
  if python3 - "$run_root" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
provenance_path = root / "stop-provenance.json"
try:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    raise SystemExit(1)
if not isinstance(provenance, dict) or provenance.get("reason_code") != "external-signal":
    raise SystemExit(1)
try:
    result = json.loads((root / "run-result.json").read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    result = {}
is_partial = (
    isinstance(result, dict)
    and (result.get("status") == "partial" or result.get("execution_complete") is False)
) or (root / "execution-result.partial.json").is_file()
if not is_partial:
    raise SystemExit(1)
provenance["status"] = "partial"
temporary = provenance_path.with_name(provenance_path.name + ".tmp")
temporary.write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, provenance_path)
PY
  then
    expected_partial=true
  fi
}

# comparison execution 只保留原始 profile 和 trace；覆盖率汇总由离线流程完成。
finalization_attempted=false
finalization_code=0
if [[ "$defer_external_finalize" != true &&
      "$comparison_execution_action" == true &&
      ( "$phase_status" != completed || "$phase_sealed" != true ||
        "$wrapper_code" -ne 0 ) &&
      ( -f "$run_root/run-result.json" || -f "$run_root/execution-result.partial.json" ||
        -f "$run_root/ledger/events.partial.jsonl" ||
        "$framework_matrix_artifacts" == true ||
        "$framework_single_artifacts" == true ||
        "$framework_shared_artifacts" == true ||
        "$framework_case_artifacts" == true ) ]]; then
  finalization_attempted=true
  finalizer_source_root="$source_snapshot"
  host_finalize_attempted=false
  host_finalize_code=0
  needs_host_finalize=false
  if [[ ! -f "$run_root/seal.json" || ! -f "$run_root/derived/campaign-complete.json" ]] || \
       ! python3 - "$run_root" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    seal = json.loads((root / "seal.json").read_text(encoding="utf-8"))
    campaign = json.loads((root / "derived/campaign-complete.json").read_text(encoding="utf-8"))
except (OSError, TypeError, ValueError):
    raise SystemExit(1)
if not isinstance(seal, dict) or not isinstance(campaign, dict):
    raise SystemExit(1)
if seal.get("schema_version") != "rq1-campaign-seal-v1":
    raise SystemExit(1)
if campaign.get("schema_version") != "rq1-campaign-complete-v1":
    raise SystemExit(1)
if seal.get("run_id") != root.name or campaign.get("run_id") != root.name:
    raise SystemExit(1)
if (
    seal.get("sealed") is not (campaign.get("execution_complete") is True)
    or seal.get("execution_complete") is not (campaign.get("execution_complete") is True)
    or seal.get("formal_ready") is not (campaign.get("formal_ready") is True)
):
    raise SystemExit(1)
declared = seal.get("seal_sha256")
basis = dict(seal)
basis.pop("seal_sha256", None)
actual = hashlib.sha256(json.dumps(
    basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
if not isinstance(declared, str) or declared != actual:
    raise SystemExit(1)
def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
event_path = root / "ledger/events.jsonl"
if not event_path.is_file():
    event_path = root / "ledger/events.partial.jsonl"
for field, path in (
    ("campaign_complete_sha256", root / "derived/campaign-complete.json"),
    ("integrity_sha256", root / "derived/integrity.json"),
    ("event_ledger_sha256", event_path),
    ("ledger_seal_sha256", root / "ledger/seal.json"),
):
    if not path.is_file() or seal.get(field) != sha(path):
        raise SystemExit(1)
registry_path = root / "coverage-registry.json"
if (not registry_path.is_file()
        or seal.get("coverage_registry_file_sha256") != sha(registry_path)):
    raise SystemExit(1)
if seal.get("coverage_summary_sha256") is not None:
    coverage_path = root / "coverage/summary.json"
    if not coverage_path.is_file() or seal.get("coverage_summary_sha256") != sha(coverage_path):
        raise SystemExit(1)
integrity = json.loads((root / "derived/integrity.json").read_text(encoding="utf-8"))
files = integrity.get("files") if isinstance(integrity, dict) else None
if not isinstance(files, dict):
    raise SystemExit(1)
root_resolved = root.resolve()
for relative, expected in files.items():
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root_resolved) or not path.is_file() or sha(path) != expected:
        raise SystemExit(1)
PY
  then
    # The container runner is the single owner of finalization.  A second
    # host-side recovery pass can downgrade an already sealed run when its
    # duplicate hash check sees a transient/stale file view.  Recover only
    # when the container did not report a sealed finalization at all.
    if [[ -f "$run_root/finalization-result.json" ]] && \
       python3 - "$run_root/finalization-result.json" <<'PY'
import json, sys
try:
    result = json.load(open(sys.argv[1]))
except (OSError, TypeError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if (
    isinstance(result, dict)
    and result.get("status") == "sealed"
    and result.get("exit_code") == 0
    and result.get("finalized_in_container") is True
) else 1)
PY
    then
      :
    else
      needs_host_finalize=true
    fi
  fi
  if [[ "$needs_host_finalize" == true ]]; then
    # The main container may be killed after writing the partial ledger but
    # before its in-container finalizer runs.  Recompute a gap/seal locally so
    # the run remains auditable; this does not promote incomplete evidence.
    set +e
    host_finalize_attempted=true
    env RQ1_DEPS="$dependency_root" RQ1_RUN_ROOT="$run_root" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$finalizer_source_root" \
      timeout --signal=TERM --kill-after=30s "${finalize_timeout_seconds}s" \
      python3 "$finalizer_source_root/runner.py" finalize \
        --allow-missing-final-identity \
        --recovery-partial \
        --config "$run_root/config.snapshot.json" --output "$run_root" \
      > "$log_root/host-finalize.stdout.log" 2> "$log_root/host-finalize.stderr.log"
    host_finalize_code=$?
    set -e
  fi
  if [[ "$host_finalize_attempted" == true ]]; then
    finalization_code=$host_finalize_code
  elif [[ -f "$run_root/finalization-result.json" ]]; then
    finalization_code=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("exit_code", 125)))' "$run_root/finalization-result.json" 2>/dev/null || printf '125')
  elif [[ -f "$run_root/seal.json" ]]; then
    finalization_code=$(python3 -c 'import json,sys; print(0 if json.load(open(sys.argv[1])).get("sealed") is True else 125)' "$run_root/seal.json" 2>/dev/null || printf '125')
  else
    finalization_code=125
  fi
  if [[ ! "$finalization_code" =~ ^[0-9]+$ ]]; then
    finalization_code=125
  fi
  if [[ "$wrapper_code" -ne 0 && "$expected_partial" == true &&
        "$reason_code" == run-complete ]]; then
    if [[ "$wrapper_code" -eq 137 ]]; then
      reason_code=container-signal-killed
    elif [[ "$host_finalize_code" -ne 0 && ! -f "$run_root/seal.json" ]]; then
      reason_code=finalization-incomplete
    else
      reason_code=partial-execution
    fi
  fi
fi

if [[ "$finalization_attempted" == true ]]; then
  if [[ "$comparison_execution_action" == true &&
        -f "$run_root/derived/campaign-complete.json" ]]; then
    read -r execution_complete formal_ready formal_reason < <(python3 - "$run_root/derived/campaign-complete.json" <<'PY'
import json, sys

path = sys.argv[1]
try:
    payload = json.load(open(path))
except (OSError, TypeError, ValueError):
    payload = {}
execution_complete = payload.get("execution_complete", payload.get("status") == "complete")
formal_ready = payload.get("formal_ready", False)
if formal_ready:
    reason = "run-complete"
elif payload.get("coverage_enabled") is True and payload.get("coverage_artifact_complete") is not True:
    reason = "coverage-gap"
elif payload.get("coverage_enabled") is False:
    reason = "coverage-disabled"
elif payload.get("reference_execution_complete") is not True \
        or payload.get("differential_evidence_complete") is not True:
    reason = "reference-gap"
else:
    reason = "formal-evidence-gap"
print(str(bool(execution_complete)).lower(), str(bool(formal_ready)).lower(), reason)
PY
    )
    if [[ "$finalization_code" -eq 0 && "$execution_complete" == true ]]; then
      wrapper_code=0
      reason_code="$formal_reason"
    fi
  fi
  if [[ "$finalization_code" -ne 0 && "$wrapper_code" -eq 0 ]]; then
    wrapper_code=125
    reason_code=finalization-incomplete
  fi
  # 主 Docker 已经写下 seal；此处只让 wrapper-result 反映最终的
  # execution/formal 状态。
  sync_stop_provenance_status
  write_wrapper_result
else
  sync_stop_provenance_status
  write_wrapper_result
fi
exit "$wrapper_code"

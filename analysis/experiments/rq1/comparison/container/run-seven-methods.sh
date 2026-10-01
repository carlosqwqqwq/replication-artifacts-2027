#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
entry="$source_root/container/run-seven-long-methods.sh"
duration_seconds="${RQ1_DURATION_SECONDS:-259200}"
target_timeout_seconds="${RQ1_TARGET_TIMEOUT_SECONDS:-180}"
dependency_root="${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}"
output_root="${RQ1_OUTPUT_ROOT:-/path/to/rq1-comparison/results}"
case_catalog_root="${RQ1_CASE_CATALOG_ROOT:-/path/to/rq1-comparison/results/cases/rq1-case-library}"
run_prefix="${RQ1_RUN_PREFIX:-rq1-seven-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
resource_mode=seven-method-parallel
skip_doctor=false
dry_run=false

usage() {
  cat <<'EOF'
用法：
  container/run-seven-methods.sh [选项]

七方法×七 Target 的运行入口，默认运行 72 小时。五个外部工具回放固定 case catalog；
RVGEN-Direct 在线生成根 case，Program-Full 使用相同启动快照中的 B-RVDV 有序 `.S` 序列，
并继续在线运行 small-model、EMI、MCMC 和 reference 链。到时在 case 边界暂停，同一 run 可继续。

选项：
  --run-prefix PREFIX          本轮唯一前缀
  --duration-seconds N         单段运行时间，默认 259200
  --target-timeout-seconds N   单次 Target 执行时限，默认 180
  --case-dir DIR               外部回放与 Program-Full seed 使用的 case catalog
  --dependency-root DIR        依赖根
  --output-root DIR            结果根
  --resource-mode MODE         seven-method-parallel
  --skip-doctor                跳过 doctor 预检
  --dry-run                    只展开命令
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --run-prefix) run_prefix=${2:?--run-prefix requires a value}; shift 2;;
    --duration-seconds) duration_seconds=${2:?--duration-seconds requires a value}; shift 2;;
    --target-timeout-seconds) target_timeout_seconds=${2:?--target-timeout-seconds requires a value}; shift 2;;
    --case-dir|--case-catalog) case_catalog_root=${2:?--case-dir requires a value}; shift 2;;
    --dependency-root) dependency_root=${2:?--dependency-root requires a value}; shift 2;;
    --output-root) output_root=${2:?--output-root requires a value}; shift 2;;
    --resource-mode) resource_mode=${2:?--resource-mode requires a value}; shift 2;;
    --skip-doctor) skip_doctor=true; shift;;
    --dry-run) dry_run=true; shift;;
    --without-board) echo 'the 7×7 RQ1 matrix requires K1 reference evidence' >&2; exit 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

args=(--duration-seconds "$duration_seconds"
  --target-timeout-seconds "$target_timeout_seconds"
  --case-dir "$case_catalog_root"
  --run-prefix "$run_prefix"
  --resource-mode "$resource_mode"
  --dependency-root "$dependency_root"
  --output-root "$output_root")
[[ "$skip_doctor" == true ]] && args+=(--skip-doctor)
[[ "$dry_run" == true ]] && args+=(--dry-run)
exec bash "$entry" "${args[@]}"

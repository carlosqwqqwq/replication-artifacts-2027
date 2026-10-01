#!/usr/bin/env bash
set -Eeuo pipefail

comparison_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(git -C "$comparison_root" rev-parse --show-toplevel)
entry="$comparison_root/container/rq1_rvgen_catalog_coverage.py"
config_file="$comparison_root/config/rq1-comparison-isolated-v1.json"
dependency_root=$(realpath -- "${RQ1_DEPENDENCY_ROOT:-/path/to/rq1-comparison/deps}")
output_root=/path/to/rq1-comparison/results
ssh_root=${RQ1_SSH_ROOT:-/path/to/user/.ssh}
run_user=${RQ1_RUN_USER:-rvdata}
run_id="rvgen-catalog-72h-$(date -u +%Y%m%dT%H%M%SZ)-$$"
seed=
dry_run=false

duration_seconds=259200

usage() {
  cat <<'EOF'
用法：
  container/run-rvgen-catalog-coverage-72h.sh [选项]

启动随机 RVGEN 全 catalog 执行长跑；完整随机队列准备完成后计时 72 小时。
七个 target 各自消费独立随机队列；一个容器总限额为 2 CPU、14 GiB 内存。
模拟器仅产出并保留原始 profile 和完整 trace，汇总由后续离线流程完成。

选项：
  --run-id ID                 唯一运行 ID
  --seed N                    可复现随机种子；默认由启动器生成
  --duration-seconds N        本段活跃时长；默认 259200 秒（72 小时）
  --dependency-root DIR       依赖目录，默认 /path/to/rq1-comparison/deps
  --dry-run                   执行只读预检并打印资源与启动计划
  --help                      显示帮助

默认运行段长为 72 小时；target timeout 读配置；容器资源读配置
`execution_policy.resource_profiles.rvgen_catalog_72h`。正式启动使用只读源码快照。
到达本段时长后 runner 在 case 边界暂停并留在容器中；可用 control-rvgen-catalog-72h.sh resume 接续。
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --run-id) run_id=${2:?--run-id requires a value}; shift 2;;
    --seed) seed=${2:?--seed requires a value}; shift 2;;
    --duration-seconds)
      duration_seconds=${2:?--duration-seconds requires a value}
      shift 2
      ;;
    --dependency-root)
      dependency_root=$(realpath -- "${2:?--dependency-root requires a value}")
      shift 2
      ;;
    --dry-run) dry_run=true; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9_.-]{1,55}$ ]] || { echo 'invalid run-id (1-55 safe characters)' >&2; exit 2; }
[[ "$duration_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo '--duration-seconds must be a positive integer' >&2; exit 2;
}
k1_reference_parallelism=${RQ1_NATIVE_REFERENCE_PARALLELISM:-8}
[[ "$k1_reference_parallelism" =~ ^[1-9][0-9]*$ ]] || {
  echo 'RQ1_NATIVE_REFERENCE_PARALLELISM must be a positive integer' >&2; exit 2;
}
if ((${#k1_reference_parallelism} > 1)) || ((k1_reference_parallelism > 8)); then
  k1_reference_parallelism=8
fi
reference_worker_count=$k1_reference_parallelism
if [[ -z "$seed" ]]; then
  seed=$(python3 -c 'import secrets; print(secrets.randbits(64))')
else
  [[ "$seed" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]] || {
    echo 'seed must be a decimal or hexadecimal integer' >&2
    exit 2
  }
  seed=$(python3 - "$seed" <<'PY'
import sys
value = sys.argv[1]
print(int(value, 0) if value.lower().startswith("0x") else int(value, 10))
PY
)
fi

[[ -f "$entry" && -f "$config_file" ]] || { echo 'experiment source or config missing' >&2; exit 2; }
[[ -d "$dependency_root" && -r "$dependency_root" ]] || {
  echo "dependency root is unavailable: $dependency_root" >&2; exit 2;
}
for path in "$ssh_root/config" "$ssh_root/known_hosts" /usr/bin/ssh /usr/lib/openssh; do
  [[ -e "$path" && -r "$path" ]] || { echo "K1 transport input is unavailable: $path" >&2; exit 2; }
done
for path in /usr/riscv64-linux-gnu /usr/lib/gcc-cross/riscv64-linux-gnu; do
  [[ -d "$path" && -r "$path" ]] || {
    echo "RISC-V cross compiler sysroot is unavailable: $path" >&2; exit 2;
  }
done
for coverage_dir in simulator-coverage-sources simulator-coverage-builds \
    simulator-coverage-artifacts simulator-coverage-tools; do
  [[ -d "$dependency_root/$coverage_dir" ]] || {
    echo "coverage dependency is missing: $dependency_root/$coverage_dir" >&2
    exit 2
  }
done

mapfile -t resource_settings < <(python3 - "$config_file" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
profile = config.get("execution_policy", {}).get("resource_profiles", {}).get(
    "rvgen_catalog_72h"
)
expected_limits = {
    "cpus": 2.0, "memory": "14g", "memory_swap": "14g", "pids": 512,
}
if (
    not isinstance(profile, dict)
    or profile.get("mode") != "rvgen-catalog-72h"
    or profile.get("parallel_containers") != 1
    or profile.get("simultaneous_simulator_processes") != 15
    or profile.get("limits") != expected_limits
    or profile.get("tmpfs") != {"/tmp": "2g"}
):
    raise SystemExit("rvgen_catalog_72h resource profile violates the run contract")
print(json.dumps(profile, sort_keys=True, separators=(",", ":")))
print(expected_limits["cpus"])
print(expected_limits["memory"])
print(expected_limits["memory_swap"])
print(expected_limits["pids"])
print(profile["tmpfs"]["/tmp"])
PY
)
(( ${#resource_settings[@]} == 6 )) || {
  echo 'could not read the pinned RVGEN catalog resource profile' >&2
  exit 2
}
resource_profile_json=${resource_settings[0]}
container_cpus=${resource_settings[1]}
container_memory=${resource_settings[2]}
container_memory_swap=${resource_settings[3]}
container_pids=${resource_settings[4]}
container_tmpfs=${resource_settings[5]}

mapfile -t image_settings < <(python3 - "$config_file" "$dependency_root" <<'PY'
import json
import re
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
targets = config.get("targets")
if not isinstance(targets, list) or len(targets) != 7:
    raise SystemExit("config must declare all seven simulator targets")
expected_ids = {
    "T-QEMU", "T-LRSV-INT", "T-LRSV-TRANS", "T-UNICORN",
    "T-RENODE", "T-RAX", "T-RVVM",
}
if {str(target.get("id")) for target in targets} != expected_ids:
    raise SystemExit("config target IDs do not match the RVGEN catalog run")
for target in targets:
    for key in ("binary", "coverage_binary", "identity"):
        value = target.get(key)
        if not isinstance(value, str) or not (root / value).is_file():
            raise SystemExit(f"{target.get('id')}: missing dependency {key}: {value}")
    if not isinstance(target.get("coverage_binary_sha256"), str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", target["coverage_binary_sha256"],
    ):
        raise SystemExit(f"{target.get('id')}: coverage binary SHA-256 is not pinned")
policy = config.get("execution_policy", {})
print(policy["coverage_container_image"])
print(policy["coverage_container_image_digest"])
PY
)
(( ${#image_settings[@]} == 2 )) || { echo 'could not read pinned coverage image' >&2; exit 2; }
image_ref=${image_settings[0]}
expected_image_id=${image_settings[1]}

command -v docker >/dev/null || { echo 'docker CLI is unavailable' >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo 'Docker daemon is unavailable' >&2; exit 125; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'expected x86_64 host' >&2; exit 2; }
[[ -w "$output_root" ]] || { echo "results root is not writable: $output_root" >&2; exit 2; }
docker_image_id=$(docker image inspect --format '{{.Id}}' "$image_ref")
[[ "$docker_image_id" == "$expected_image_id" ]] || {
  echo "image identity mismatch: expected $expected_image_id, got $docker_image_id" >&2
  exit 125
}
if [[ "$dry_run" == true ]]; then
  df -hT / /mnt/data >&2
  docker system df >&2
  printf 'run=%s seed=%s segment_duration=%ss coverage=raw-profiles-and-traces\n' \
    "$run_id" "$seed" "$duration_seconds"
  printf 'targets: T-QEMU T-LRSV-INT T-LRSV-TRANS T-UNICORN T-RENODE T-RAX T-RVVM\n'
  printf 'scheduler: seven target workers plus %s K1/QEMU reference workers; K1 slots=%s\n' \
    "$reference_worker_count" "$k1_reference_parallelism"
  printf 'references: K1 per RV64 Linux-user candidate; timeout/transport failure -> same-candidate T-QEMU\n'
  printf 'container: image=%s cpus=%s memory=%s swap=%s pids=%s network=bridge\n' \
    "$image_ref" "$container_cpus" "$container_memory" "$container_memory_swap" "$container_pids"
  printf 'tmpfs: /tmp=%s\n' "$container_tmpfs"
  printf 'source and config snapshot: %s/source-snapshots/%s (read-only bind mounts)\n' "$output_root" "$run_id"
  printf 'run root: %s/runs/%s\n' "$output_root" "$run_id"
  exit 0
fi

df -hT / /mnt/data >&2
docker system df >&2
run_uid=$(id -u "$run_user")
run_gid=$(id -g "$run_user")
for path in "$output_root" "$output_root/runs" "$output_root/source-snapshots"; do
  [[ ! -L "$path" ]] || { echo "results path must not be a symlink: $path" >&2; exit 2; }
done
mkdir -p "$output_root/runs" "$output_root/source-snapshots"
test -w "$output_root/runs" && test -w "$output_root/source-snapshots" || {
  echo "results directories are not writable under $output_root" >&2
  exit 2
}
[[ "$(realpath -m -- "$output_root/runs")" == "$output_root/runs" &&
   "$(realpath -m -- "$output_root/source-snapshots")" == "$output_root/source-snapshots" ]] || {
  echo 'results directories resolve outside the configured data path' >&2
  exit 2
}
run_root="$output_root/runs/$run_id"
source_snapshot="$output_root/source-snapshots/$run_id"
container_name="rq1cmp-$run_id"
[[ ! -e "$run_root" && ! -L "$run_root" &&
   ! -e "$source_snapshot" && ! -L "$source_snapshot" ]] || {
  echo 'run ID already exists' >&2
  exit 2
}
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "container name already exists: $container_name" >&2
  exit 2
fi
tree_sha256() {
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -cf - -C "$1" . | sha256sum | awk '{print $1}'
}
source_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
source_dirty=false
[[ -z "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]] || source_dirty=true
source_tree_before=$(tree_sha256 "$comparison_root")

mkdir "$run_root" "$source_snapshot"
chown "$run_uid:$run_gid" "$run_root"
framework_local_root="$run_root/framework-local"
framework_runs_root="$run_root/framework-runs"
tmp_root="$run_root/tmp"
mkdir -p "$framework_local_root" "$framework_runs_root" "$tmp_root"
chown "$run_uid:$run_gid" "$framework_local_root" "$framework_runs_root" "$tmp_root"
cp -a --reflink=auto "$comparison_root"/. "$source_snapshot"/
chmod --reference="$comparison_root" "$source_snapshot"
cp -- "$source_snapshot/config/rq1-comparison-isolated-v1.json" "$run_root/config.snapshot.json"
chmod 0444 "$run_root/config.snapshot.json"
chown "$run_uid:$run_gid" "$run_root/config.snapshot.json"
source_snapshot_tree_sha256=$(tree_sha256 "$source_snapshot")
source_tree_after=$(tree_sha256 "$comparison_root")
source_commit_after=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
source_dirty_after=false
[[ -z "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]] || source_dirty_after=true
[[ "$source_tree_before" == "$source_snapshot_tree_sha256" &&
   "$source_tree_before" == "$source_tree_after" &&
   "$source_commit" == "$source_commit_after" &&
   "$source_dirty" == "$source_dirty_after" ]] || {
  echo 'worktree changed while creating the code snapshot; no experiment was started' >&2
  exit 125
}
chmod -R a-w -- "$source_snapshot"

flush_helper="$run_root/rvvm-stop-flush.so"
cc -O2 -fPIC -shared -pthread "$source_snapshot/container/rvvm-stop-flush.c" -o "$flush_helper"
chmod 0644 "$flush_helper"
chown "$run_uid:$run_gid" "$flush_helper"
flush_helper_sha256=$(sha256sum "$flush_helper" | awk '{print $1}')

config_sha256=$(sha256sum "$run_root/config.snapshot.json" | awk '{print $1}')
source_snapshot_sha256=$(tree_sha256 "$source_snapshot")

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  df -hT / /mnt/data
  free -h
  uptime
  nproc
  docker system df
  printf 'image_id=%s\n' "$docker_image_id"
  printf 'resource_profile=%s\n' "$resource_profile_json"
  printf 'source_commit=%s\nsource_dirty=%s\n' "$source_commit" "$source_dirty"
  printf 'source_tree_sha256=%s\nsource_snapshot_sha256=%s\nconfig_sha256=%s\n' \
    "$source_tree_before" "$source_snapshot_sha256" "$config_sha256"
} > "$run_root/host-preflight.txt" 2>&1

RQ1_RUN_ID="$run_id" RQ1_SEED="$seed" RQ1_IMAGE_ID="$docker_image_id" \
  RQ1_DURATION_SECONDS="$duration_seconds" \
  RQ1_RVVM_STOP_FLUSH_SHA256="$flush_helper_sha256" \
  RQ1_SOURCE_COMMIT="$source_commit" RQ1_SOURCE_DIRTY="$source_dirty" \
  RQ1_SOURCE_TREE_SHA256="$source_tree_before" \
  RQ1_SOURCE_SNAPSHOT_SHA256="$source_snapshot_sha256" RQ1_CONFIG_SHA256="$config_sha256" \
  RQ1_REFERENCE_WORKERS="$reference_worker_count" \
  RQ1_RESOURCE_PROFILE_JSON="$resource_profile_json" \
  python3 - "$run_root/host-manifest.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "rq1-rvgen-catalog-host-manifest-v4",
    "run_id": os.environ["RQ1_RUN_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "duration_seconds": int(os.environ["RQ1_DURATION_SECONDS"]),
    "duration_semantics": "active segment; pause and wait at target-case boundaries",
    "coverage_mode": "raw-profiles-and-execution-traces",
    "raw_artifact_index": "raw-artifact-index.json",
    "seed": int(os.environ["RQ1_SEED"]),
    "container_image_id": os.environ["RQ1_IMAGE_ID"],
    "source_commit": os.environ.get("RQ1_SOURCE_COMMIT"),
    "source_dirty": os.environ.get("RQ1_SOURCE_DIRTY") == "true",
    "source_worktree_sha256": os.environ["RQ1_SOURCE_TREE_SHA256"],
    "source_snapshot_sha256": os.environ["RQ1_SOURCE_SNAPSHOT_SHA256"],
    "config_sha256": os.environ["RQ1_CONFIG_SHA256"],
    "rvvm_stop_flush_helper_sha256": os.environ["RQ1_RVVM_STOP_FLUSH_SHA256"],
    "resource_profile": json.loads(os.environ["RQ1_RESOURCE_PROFILE_JSON"]),
    "source_snapshot_mount": "read-only",
    "config_snapshot_mount": "read-only",
    "target_workers": 7,
    "reference_workers": int(os.environ["RQ1_REFERENCE_WORKERS"]),
    "simulator_tasks_at_once": 7 + int(os.environ["RQ1_REFERENCE_WORKERS"]),
}, ensure_ascii=False, indent=2) + "\n")
PY

command=(python3 /opt/rq1/comparison/container/rq1_rvgen_catalog_coverage.py
  --seed "$seed" --duration-seconds "$duration_seconds")
source "$(dirname "${BASH_SOURCE[0]}")/k1-reference-locks.sh"
k1_reference_lock_root="$output_root/.k1-reference-locks"
k1_reference_lock_prepare "$output_root" "$run_uid" "$run_gid"
mounts=(
  -v "$source_snapshot:/opt/rq1/comparison:ro"
  -v "$run_root:/opt/runs/$run_id:rw"
  -v "$source_snapshot/config/rq1-comparison-isolated-v1.json:/opt/runs/$run_id/config.snapshot.json:ro"
  -v "$dependency_root:/path/to/deps:ro"
  -v "$flush_helper:/opt/rvvm-stop-flush.so:ro"
  -v "$ssh_root:/path/to/user/.ssh:ro"
  -v /etc/passwd:/etc/passwd:ro
  -v /etc/group:/etc/group:ro
  -v /usr/bin/ssh:/usr/bin/ssh:ro
  -v /usr/lib/openssh:/usr/lib/openssh:ro
  -v /usr/riscv64-linux-gnu:/usr/riscv64-linux-gnu:ro
  -v /usr/lib/gcc-cross/riscv64-linux-gnu:/usr/lib/gcc-cross/riscv64-linux-gnu:ro
  -v "$k1_reference_lock_root:/run/rq1-native-reference:rw"
)
environment=(
  -e RQ1_RUN_ROOT="/opt/runs/$run_id"
  -e RQ1_RUN="/opt/runs/$run_id"
  -e RQ1_COMPARISON_ROOT=/opt/rq1/comparison
  -e RQ1_CONFIG_PATH="/opt/runs/$run_id/config.snapshot.json"
  -e RQ1_FRAMEWORK_LOCAL_ROOT="/opt/runs/$run_id/framework-local"
  -e RQ1_FRAMEWORK_RUNS_ROOT="/opt/runs/$run_id/framework-runs"
  -e RQ1_DEPS=/path/to/deps
  -e RQ1_ENABLE_COVERAGE=1
  -e RQ1_RVVM_STOP_FLUSH=/opt/rvvm-stop-flush.so
  -e RQ1_RVVM_STOP_FLUSH_SHA256="$flush_helper_sha256"
  -e RQ1_SOURCE_COMMIT="$source_commit"
  -e RQ1_SOURCE_DIRTY="$source_dirty"
  -e RQ1_SOURCE_SNAPSHOT_SHA256="$source_snapshot_sha256"
  -e RQ1_RESOURCE_MODE=rvgen-catalog-72h
  -e RQ1_RESOURCE_PROFILE_JSON="$resource_profile_json"
  -e RQ1_NETWORK=bridge
  -e RQ1_EXECUTION_PLANE=rq1-comparison-docker
  -e MIGRATION_EXECUTION_PLANE=docker-x86_64-lab
  -e MIGRATION_CONTAINER_IMAGE="$image_ref"
  -e MIGRATION_CONTAINER_IMAGE_DIGEST="$docker_image_id"
  -e RQ1_IMAGE_DIGEST="$docker_image_id"
  -e RQ1_DOTNET_RUNTIME=/path/to/deps/toolchains/dotnet-8
  -e RQ1_NATIVE_SSH_CONFIG=/path/to/ssh/config
  -e RQ1_NATIVE_KNOWN_HOSTS=/path/to/user/.ssh/known_hosts
  -e RQ1_NATIVE_SSH_ALIAS=k1-board
  -e RQ1_NATIVE_REMOTE_ROOT="/mnt/build/wangyang/runs/rq1-$run_id"
  -e RQ1_NATIVE_REFERENCE_LOCK_PATH=/run/rq1-native-reference/native-rv64.lock
  -e "RQ1_NATIVE_REFERENCE_PARALLELISM=$k1_reference_parallelism"
  -e RQ1_NATIVE_TRACE_HELPER_CACHE=1
  -e DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
  -e DOTNET_CLI_HOME=/tmp
  -e HOME=/tmp
  -e TMPDIR="/opt/runs/$run_id/tmp"
  -e XDG_CACHE_HOME=/tmp/.cache
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONPATH=/opt/rq1/comparison/adapters/fuzz4all-remote:/path/to/deps/python:/opt/pymysql:/opt/rvdv-python:/opt/unicorn-2.1.3
)

if ! docker run -d --init --name "$container_name" \
  --user "$run_uid:$run_gid" \
  --cpus "$container_cpus" --memory "$container_memory" \
  --memory-swap "$container_memory_swap" --pids-limit "$container_pids" \
  --workdir /opt/rq1/comparison --read-only --entrypoint "" \
  --tmpfs "/tmp:rw,nosuid,nodev,exec,size=$container_tmpfs" \
  --network bridge --ipc private \
  --log-driver json-file --log-opt max-size=50m --log-opt max-file=3 \
  --security-opt no-new-privileges --cap-drop ALL \
  --cap-add DAC_OVERRIDE --cap-add SETUID --cap-add SETGID --cap-add KILL \
  "${mounts[@]}" "${environment[@]}" \
  "$image_ref" "${command[@]}" > "$run_root/container-id.txt" \
  2> "$run_root/container-launch.stderr"; then
  cat "$run_root/container-launch.stderr" >&2
  exit 125
fi

container_id=$(cat "$run_root/container-id.txt")
docker inspect --format '{{json .State}}' "$container_id" > "$run_root/container-inspect.json"
printf 'started run=%s seed=%s container=%s\n' "$run_id" "$seed" "$container_name"
printf 'progress=%s\n' "$run_root/progress.json"
printf 'raw_artifact_index=%s/raw-artifact-index.json (updated at pause and runner exit)\n' "$run_root"
printf 'raw profiles=%s/coverage-raw\n' "$run_root"
printf 'execution traces=%s/cases/*/target-workers/*/traces\n' "$run_root"
printf 'control=%s/container/control-rvgen-catalog-72h.sh {pause|resume|status} --run-id %s\n' \
  "$comparison_root" "$run_id"
printf 'logs: docker logs -f %s\n' "$container_name"

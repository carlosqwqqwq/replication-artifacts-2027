k1_reference_lock_prepare() {
  local output_root=$1 run_uid=$2 run_gid=$3
  local lock_root="$output_root/.k1-reference-locks"
  local setup_lock="$output_root/.k1-reference-locks.setup.lock"
  local setup_fd lock_root_created=false lock_root_owner lock_file lock_file_owner invalid_entry

  [[ ! -L "$lock_root" ]] || {
    echo "K1 reference lock directory must not be a symlink: $lock_root" >&2; exit 2;
  }
  [[ ! -L "$setup_lock" ]] || {
    echo "K1 reference setup lock must not be a symlink: $setup_lock" >&2; exit 2;
  }
  [[ ! -e "$setup_lock" || -f "$setup_lock" ]] || {
    echo "K1 reference setup lock must be a regular file: $setup_lock" >&2; exit 2;
  }
  command -v flock >/dev/null || {
    echo 'flock is required to initialize the shared K1 reference gate' >&2; exit 2;
  }
  exec {setup_fd}>"$setup_lock"
  flock -x "$setup_fd"
  if [[ ! -e "$lock_root" ]]; then
    mkdir -m 700 "$lock_root"
    lock_root_created=true
  fi
  if [[ "$lock_root_created" == true ]]; then
    if ! chown "$run_uid:$run_gid" "$lock_root"; then
      echo "cannot set K1 reference lock ownership for container user $run_uid:$run_gid" >&2
      exit 2
    fi
  else
    [[ -d "$lock_root" ]] || {
      echo "K1 reference lock path must be a directory: $lock_root" >&2; exit 2;
    }
    lock_root_owner=$(stat -c '%u:%g' "$lock_root")
    if [[ "$lock_root_owner" != "$run_uid:$run_gid" ]]; then
      echo "K1 runs sharing $lock_root must use one RQ1_RUN_USER; owner is $lock_root_owner, requested $run_uid:$run_gid" >&2
      exit 2
    fi
  fi
  invalid_entry=$(find "$lock_root" -maxdepth 1 -name 'native-rv64.lock*' ! -type f -print -quit)
  [[ -z "$invalid_entry" ]] || {
    echo "K1 reference lock entries must be regular files: $invalid_entry" >&2; exit 2;
  }
  while IFS= read -r -d '' lock_file; do
    lock_file_owner=$(stat -c '%u:%g' "$lock_file")
    if [[ "$lock_file_owner" != "$run_uid:$run_gid" ]]; then
      echo "K1 reference lock file $lock_file belongs to $lock_file_owner; requested $run_uid:$run_gid" >&2
      exit 2
    fi
  done < <(find "$lock_root" -maxdepth 1 -type f -name 'native-rv64.lock*' -print0)
  chmod 700 "$lock_root"
  flock -u "$setup_fd"
  exec {setup_fd}>&-
}

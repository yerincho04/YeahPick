#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CONDA_ROOT="${CONDA_ROOT:-$(conda info --base)}"
export CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$HOME/envs}"

child_pid=""
stop_all() {
  trap - INT TERM
  if [[ -n "$child_pid" ]]; then
    kill -KILL -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 130
}
trap stop_all INT TERM

for condition in knowledge explicit_object explicit_spatial; do
  for start_id in 0 5; do
    echo "START condition=$condition questions=$start_id..$((start_id + 4))"
    setsid env \
      ASSETS=public_info_v1 \
      START_ID="$start_id" COUNT=5 BUFFER_INFERBATCH=5 \
      EVAL_GPU="${EVAL_GPU:-0}" SEED="${SEED:-0}" \
      INSTRUCTION_CONDITION="$condition" \
      bash "$REPO_ROOT/scripts/eval_openvla.sh" &
    child_pid=$!
    if wait "$child_pid"; then
      child_pid=""
    else
      status=$?
      child_pid=""
      exit "$status"
    fi
  done
done

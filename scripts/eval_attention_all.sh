#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

# Discover every runnable asset directly under carrot at execution time.
asset_root="$REPO_ROOT/ManiSkill/mani_skill/assets/carrot"
start_id="${START_ID:-0}"
failures=()
for asset_dir in "$asset_root"/*/; do
  [[ -d "$asset_dir" ]] || continue
  asset="${asset_dir%/}"
  asset="${asset##*/}"
  if [[ ! -f "$asset_dir/pairs.json" ]]; then
    echo "SKIP_ATTENTION_ASSET assets=$asset reason=no_pairs_json"
    continue
  fi
  echo "START_ATTENTION_ASSET assets=$asset questions=$start_id..$((start_id + 4)) seed=${SEED:-0}"
  setsid env ASSETS="$asset" bash "$REPO_ROOT/scripts/eval_attention_openvla.sh" &
  child_pid=$!
  if wait "$child_pid"; then
    child_pid=""
  else
    status=$?
    child_pid=""
    failures+=("$asset:$status")
    echo "FAILED_ATTENTION_ASSET assets=$asset exit=$status"
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "DONE_ATTENTION_ALL failed=${failures[*]}"
  exit 1
fi
echo "DONE_ATTENTION_ALL all_runnable_assets_succeeded"

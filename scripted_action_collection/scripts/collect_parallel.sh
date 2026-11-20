#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECT="$ROOT/scripts/collect.sh"
LOG_DIR="$ROOT/output_tile_22cm_v4/parallel_logs/$(date +%Y%m%d_%H%M%S)_$$"

# Edit this list to select the categories to collect.
CATEGORIES=(
  counting
  symmetry
  celebrity
  living_world
  traffic
  state
  attribute
  emotion
  public_info
  time_clock
)
# One GPU per concurrent slot. Set GPUS=(1) and MAX_PARALLEL=1 to use only GPU 1.
GPUS=(0 1)
MAX_PARALLEL=2

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
if ((${#CATEGORIES[@]} == 0)); then
  echo "CATEGORIES is empty" >&2
  exit 2
fi
if ((${#GPUS[@]} < MAX_PARALLEL)); then
  echo "GPUS must have at least MAX_PARALLEL entries" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU number: $gpu" >&2
    exit 2
  fi
done

declare -A running=()
declare -A seen=()
active_pids=()
active_categories=()
failed=()

stop_all() {
  local code="$1" pid
  trap - EXIT
  trap '' INT TERM
  for pid in "${!running[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${!running[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  for pid in "${!running[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$code"
}
trap 'stop_all 130' INT
trap 'stop_all 143' TERM
trap 'stop_all "$?"' EXIT

wait_batch() {
  local index pid category status
  for index in "${!active_pids[@]}"; do
    pid="${active_pids[$index]}"
    category="${active_categories[$index]}"
    if wait "$pid"; then
      echo "DONE $category"
    else
      status=$?
      failed+=("$category:$status")
      echo "FAILED $category exit=$status log=$LOG_DIR/$category.log" >&2
    fi
    unset 'running[$pid]'
  done
  active_pids=()
  active_categories=()
}

mkdir -p "$LOG_DIR"
for category in "${CATEGORIES[@]}"; do
  if [[ -n "${seen[$category]+x}" ]]; then
    echo "Duplicate category: $category" >&2
    exit 2
  fi
  seen[$category]=1
  gpu="${GPUS[${#active_pids[@]}]}"
  # The child becomes its own process group. collect.sh then execs Python,
  # allowing Ctrl-C to terminate the entire group, including descendants.
  EVAL_GPU="$gpu" python3 -c 'import os, sys; os.setsid(); os.execv("/bin/bash", ["bash", *sys.argv[1:]])' \
    "$COLLECT" --category "$category" "$@" >"$LOG_DIR/$category.log" 2>&1 &
  pid=$!
  running[$pid]="$category"
  active_pids+=("$pid")
  active_categories+=("$category")
  echo "START $category gpu=$gpu pid=$pid log=$LOG_DIR/$category.log"
  if ((${#active_pids[@]} >= MAX_PARALLEL)); then
    wait_batch
  fi
done
if ((${#active_pids[@]} > 0)); then
  wait_batch
fi

trap - INT TERM EXIT
if ((${#failed[@]} > 0)); then
  printf 'COLLECTION_FAILED %s\n' "${failed[*]}" >&2
  exit 1
fi
echo "COLLECTION_DONE_ALL"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"
conda activate "${CONDA_ENVS_DIR}/openvla_rl4vla"

assets=(counting_color_v2 symmetry_color_v2)
if [[ -n "${ASSET:-}" ]]; then
  case "$ASSET" in
    counting_color_v2|symmetry_color_v2) assets=("$ASSET") ;;
    *) echo "ASSET must be counting_color_v2 or symmetry_color_v2" >&2; exit 2 ;;
  esac
fi

read -r -a seeds <<< "${SEEDS:-0 1 2}"
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done
for asset in "${assets[@]}"; do
  total=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "ManiSkill/mani_skill/assets/carrot/$asset/pairs.json")
  for seed in "${seeds[@]}"; do
  for condition in knowledge explicit_object explicit_spatial; do
    for ((start=0; start<total; start+=5)); do
      count=$((total-start))
      if ((count>5)); then count=5; fi
      ENABLE_DIAGNOSTICS=1 ASSETS="$asset" START_ID="$start" COUNT="$count" \
        BUFFER_INFERBATCH="$count" EVAL_GPU="${EVAL_GPU:-0}" SEED="$seed" \
        INSTRUCTION_CONDITION="$condition" bash scripts/eval_openvla.sh
    done
  done
  done
done

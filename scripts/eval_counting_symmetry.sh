#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"
conda activate "${CONDA_ENVS_DIR}/openvla_rl4vla"

assets=(counting_pilot_v1 symmetry_pilot_v1)
if [[ -n "${ASSET:-}" ]]; then
  case "$ASSET" in
    counting_pilot_v1|symmetry_pilot_v1) assets=("$ASSET") ;;
    *) echo "ASSET must be counting_pilot_v1 or symmetry_pilot_v1" >&2; exit 2 ;;
  esac
fi

for asset in "${assets[@]}"; do
  total=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "ManiSkill/mani_skill/assets/carrot/$asset/pairs.json")
  for condition in knowledge explicit_spatial; do
    for ((start=0; start<total; start+=5)); do
      count=$((total-start))
      if ((count>5)); then count=5; fi
      ENABLE_DIAGNOSTICS=1 ASSETS="$asset" START_ID="$start" COUNT="$count" \
        BUFFER_INFERBATCH="$count" EVAL_GPU="${EVAL_GPU:-0}" SEED="${SEED:-0}" \
        INSTRUCTION_CONDITION="$condition" bash scripts/eval_openvla.sh
    done
  done
done

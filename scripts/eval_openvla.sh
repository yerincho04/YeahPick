#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-test_colors}
COUNT=${COUNT:-6}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
VLA_PATH=${VLA_PATH:-gen-robot/openvla-7b-rlvla-sft_16k}
UNNORM=${UNNORM:-sft}
SEED=${SEED:-0}
INSTRUCTION_CONDITION=${INSTRUCTION_CONDITION:-knowledge}
END_ID=$((START_ID + COUNT - 1))
LOG=${A2A_LOG_DIR}/openvla_${ASSETS}_${INSTRUCTION_CONDITION}_q${START_ID}-${END_ID}_seed${SEED}_eval.log

diag_extra=()
[ "${ENABLE_DIAGNOSTICS:-0}" = "1" ] && diag_extra=(--enable-diagnostics)
hidden_extra=()
# Hidden states are captured for the probe conditions only (knowledge + the neutral control).
case "$INSTRUCTION_CONDITION" in knowledge|neutral) ;; *) hidden_extra=(--disable-diagnostic-hidden-states) ;; esac

conda activate "${CONDA_ENVS_DIR}/openvla_rl4vla"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${REPO_ROOT}/openvla:${PYTHONPATH:-}"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_OPENVLA_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU vla=$VLA_PATH unnorm=$UNNORM condition=$INSTRUCTION_CONDITION seed=$SEED"
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_OPENVLA ${swap_arg} $(date -u)"
  cuda_env=(env CUDA_VISIBLE_DEVICES="$EVAL_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false)
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Slurm already exposes exactly the allocated GPU. Overwriting its physical device
    # mapping with "0" can hide the GPU on heterogeneous nodes.
    cuda_env=(env XLA_PYTHON_CLIENT_PREALLOCATE=false)
  fi
  "${cuda_env[@]}" python3 -u -m simpler_env.eval \
      --vla openvla --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" \
      --vla-path "$VLA_PATH" --vla-unnorm-key "$UNNORM" \
      --instruction-condition "$INSTRUCTION_CONDITION" --seed "$SEED" \
      "${extra[@]}" "${diag_extra[@]}" "${hidden_extra[@]}"
done
echo "DONE_OPENVLA_EVAL $(date -u)"

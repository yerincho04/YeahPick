#!/usr/bin/env bash
# One evaluation wrapper for every in-process model: runs noswap + swap for one asset slice and one
# instruction condition, with episode diagnostics. Model-specific settings live in the case block.
#   VLA=magma ASSETS=photo_state_v1 START_ID=0 COUNT=5 INSTRUCTION_CONDITION=knowledge bash scripts/eval_vla.sh
# Optional: VLA_PATH (checkpoint), UNNORM (OpenVLA un-normalisation key), RUN_PREFIX (output dir prefix),
# CAPTURE_ATTENTION=1 (attention maps; slower).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

: "${VLA:?set VLA=openvla|magma|spatialvla|pi0}"
ASSETS=${ASSETS:-test_colors}
COUNT=${COUNT:-6}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-0}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
SEED=${SEED:-0}
INSTRUCTION_CONDITION=${INSTRUCTION_CONDITION:-knowledge}
RUN_PREFIX=${RUN_PREFIX:-$VLA}
END_ID=$((START_ID + COUNT - 1))
LOG=${A2A_LOG_DIR}/${RUN_PREFIX}_${ASSETS}_${INSTRUCTION_CONDITION}_q${START_ID}-${END_ID}_seed${SEED}_eval.log

export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"
case "$VLA" in
  openvla)    ENV_NAME=openvla_rl4vla; export PYTHONPATH="${REPO_ROOT}/openvla:${PYTHONPATH}"; UNNORM=${UNNORM:-sft} ;;
  magma)      ENV_NAME=magma_act2answer; BUFFER_INFERBATCH=1 ;;  # its batched prompts are not padded
  spatialvla) ENV_NAME=spatialvla_act2answer ;;
  pi0)        ENV_NAME=pi0_act2answer; BUFFER_INFERBATCH=1
              export VLA_DATA_DIR="${PI0_DEPS_ROOT}" LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 \
                     MASTER_PORT="${MASTER_PORT:-29555}" PYTHONPATH="${PI0_DEPS_ROOT}:${PYTHONPATH}" ;;
  *) echo "unsupported VLA=$VLA (server-based models use their own scripts)"; exit 1 ;;
esac
conda activate "${CONDA_ENVS_DIR}/${ENV_NAME}"

extra_args=(--instruction-condition "$INSTRUCTION_CONDITION" --seed "$SEED" --run-prefix "$RUN_PREFIX")
[ -n "${VLA_PATH:-}" ] && extra_args+=(--vla-path "$VLA_PATH")
[ -n "${UNNORM:-}" ] && extra_args+=(--vla-unnorm-key "$UNNORM")
[ "${ENABLE_DIAGNOSTICS:-1}" = "1" ] && extra_args+=(--enable-diagnostics)
# CAPTURE_ATTENTION=1: per-step image attention maps, videos and CSVs under <run>/attention/
[ "${CAPTURE_ATTENTION:-0}" = "1" ] && extra_args+=(--capture-attention)
# STATIC_QA=1: answer Left/Right about the first camera image instead of acting (outputs under <run>/static_qa/)
[ "${STATIC_QA:-0}" = "1" ] && extra_args+=(--static-qa --disable-diagnostic-hidden-states)
# hidden states only for the knowledge condition (and only where the policy supports capture)
[ "$INSTRUCTION_CONDITION" != "knowledge" ] && extra_args+=(--disable-diagnostic-hidden-states)

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "START_EVAL $(date -u) vla=$VLA prefix=$RUN_PREFIX path=${VLA_PATH:-default} assets=$ASSETS q=$START_ID-$END_ID condition=$INSTRUCTION_CONDITION seed=$SEED"
for swap_arg in noswap swap; do
  swap=(); [ "$swap_arg" = swap ] && swap=(--do-swap)
  cuda_env=(env XLA_PYTHON_CLIENT_PREALLOCATE=false)
  [ -z "${SLURM_JOB_ID:-}" ] && cuda_env+=(CUDA_VISIBLE_DEVICES="$EVAL_GPU")
  "${cuda_env[@]}" python3 -u -m simpler_env.eval --vla "$VLA" --start-id "$START_ID" --count "$COUNT" \
      --assets "$ASSETS" --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" \
      "${extra_args[@]}" "${swap[@]}"
done
echo "DONE_EVAL $(date -u)"

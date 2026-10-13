#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export CONDA_ROOT="${CONDA_ROOT:-$(conda info --base)}"
export CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$HOME/envs}"
source "$REPO_ROOT/scripts/env.sh"

conda activate "${CONDA_ENVS_DIR}/openvla_rl4vla"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${REPO_ROOT}/openvla:${REPO_ROOT}:${PYTHONPATH:-}"

ASSETS="${ASSETS:-public_info_v1}"
START_ID="${START_ID:-0}"
SEED="${SEED:-0}"
EVAL_GPU="${EVAL_GPU:-0}"
VLA_PATH="${VLA_PATH:-gen-robot/openvla-7b-rlvla-sft_16k}"

echo "START_ATTENTION assets=$ASSETS questions=$START_ID..$((START_ID + 4)) seed=$SEED gpu=$EVAL_GPU"
cuda_env=(env CUDA_VISIBLE_DEVICES="$EVAL_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cuda_env=(env XLA_PYTHON_CLIENT_PREALLOCATE=false)
fi
"${cuda_env[@]}" python3 -u -m simpler_env.eval \
  --vla openvla --assets "$ASSETS" --start-id "$START_ID" --count 5 \
  --buffer-inferbatch 1 --seed "$SEED" \
  --vla-path "$VLA_PATH" --vla-unnorm-key "${UNNORM:-sft}" \
  --instruction-condition explicit_object --capture-attention

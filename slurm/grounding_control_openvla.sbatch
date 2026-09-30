#!/usr/bin/env bash
#SBATCH --job-name=a2a_grounding
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_grounding_%A_%a.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_grounding_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/home/yerincho04/YeahPick/Act2Answer
WORKSPACE_ROOT=/home/yerincho04/YeahPick
cd "$PROJECT_ROOT"
mkdir -p logs outputs "$WORKSPACE_ROOT/hf-cache" "$WORKSPACE_ROOT/tmp"
eval "$("$WORKSPACE_ROOT/miniconda3.partial-20260916/_conda" shell.posix hook)"
export CONDA_ENVS_DIR="$WORKSPACE_ROOT/envs" CONDA_ROOT="$WORKSPACE_ROOT/miniconda3.partial-20260916"
export HF_HOME="$WORKSPACE_ROOT/hf-cache" HUGGINGFACE_HUB_CACHE="$WORKSPACE_ROOT/hf-cache/hub"
export TMPDIR="$WORKSPACE_ROOT/tmp" PIP_CACHE_DIR="$WORKSPACE_ROOT/pip-cache" PYTHONNOUSERSITE=1
module load cuda/12.1
echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CONDITIONS=(spatial tile_object explicit_object visual_description)
if [ -n "${SMOKE_COUNT:-}" ]; then
  START_ID=${START_ID:-0}; COUNT=$SMOKE_COUNT
  for condition in "${CONDITIONS[@]}"; do
    ENABLE_DIAGNOSTICS=1 ASSETS=grounding_control_v1 START_ID="$START_ID" COUNT="$COUNT" \
      BUFFER_INFERBATCH="$COUNT" EVAL_GPU=0 INSTRUCTION_CONDITION="$condition" SEED="${SEED:-0}" \
      bash scripts/eval_openvla.sh
  done
else
  task=${SLURM_ARRAY_TASK_ID:?Run as an array with indices 0-15}
  condition=${CONDITIONS[$((task / 4))]}
  START_ID=$((task % 4 * 5))
  ENABLE_DIAGNOSTICS=1 ASSETS=grounding_control_v1 START_ID="$START_ID" COUNT=5 \
    BUFFER_INFERBATCH=5 EVAL_GPU=0 INSTRUCTION_CONDITION="$condition" SEED="${SEED:-0}" \
    bash scripts/eval_openvla.sh
fi

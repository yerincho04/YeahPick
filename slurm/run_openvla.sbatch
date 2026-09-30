#!/usr/bin/env bash
#SBATCH --job-name=a2a_openvla
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_openvla_%A_%a.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_openvla_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/yerincho04/YeahPick/Act2Answer
WORKSPACE_ROOT=/home/yerincho04/YeahPick
cd "$PROJECT_ROOT"
mkdir -p logs outputs "$WORKSPACE_ROOT/hf-cache" "$WORKSPACE_ROOT/tmp"

eval "$("$WORKSPACE_ROOT/miniconda3.partial-20260916/_conda" shell.posix hook)"
export CONDA_ENVS_DIR="$WORKSPACE_ROOT/envs"
export CONDA_ROOT="$WORKSPACE_ROOT/miniconda3.partial-20260916"
export HF_HOME="$WORKSPACE_ROOT/hf-cache"
export HUGGINGFACE_HUB_CACHE="$WORKSPACE_ROOT/hf-cache/hub"
export TMPDIR="$WORKSPACE_ROOT/tmp"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/pip-cache"
export PYTHONNOUSERSITE=1
module load cuda/12.1
echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CONDITIONS=(knowledge explicit_object explicit_spatial)
ASSETS=${ASSETS:?Set ASSETS=<asset folder name under ManiSkill/mani_skill/assets/carrot/>}
SHARD=${SHARD:-5}   # questions per job; keep small enough for one GPU (5 fits a 24GB card)
N=$(python3 -c "import json;print(len(json.load(open('ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json'))))")
SHARDS=$(( (N + SHARD - 1) / SHARD ))
task=${SLURM_ARRAY_TASK_ID:?Run as an array with indices 0..3*ceil(N/SHARD)-1 (see README)}
condition=${CONDITIONS[$((task / SHARDS))]}
START_ID=$(( task % SHARDS * SHARD ))
COUNT=$(( N - START_ID < SHARD ? N - START_ID : SHARD ))
ENABLE_DIAGNOSTICS=1 ASSETS="$ASSETS" START_ID="$START_ID" COUNT="$COUNT" \
  BUFFER_INFERBATCH="$COUNT" EVAL_GPU=0 INSTRUCTION_CONDITION="$condition" SEED="${SEED:-0}" \
  bash scripts/eval_openvla.sh

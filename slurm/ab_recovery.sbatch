#!/usr/bin/env bash
#SBATCH --job-name=ab_recovery
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_ab_recovery_%j.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_ab_recovery_%j.err

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


# K test (text only, two named options, A/B by next-token log-probability): diagnostics/ab_knowledge.py
#   sbatch --export=ALL,SPEC=hf:meta-llama/Llama-2-7b-hf,FMT=raw,NAME=llama2-7b,ENV=openvla_rl4vla slurm/ab_knowledge.sbatch
export HF_TOKEN="${HF_TOKEN:-$(cat "$WORKSPACE_ROOT/hf-cache/token" 2>/dev/null)}"
conda activate "$CONDA_ENVS_DIR/${ENV:?set ENV}"
export PYTHONPATH="$PROJECT_ROOT/openvla:$PROJECT_ROOT/SimplerEnv:$PROJECT_ROOT/ManiSkill:$PROJECT_ROOT:${PYTHONPATH:-}" OMP_NUM_THREADS=4
python -m diagnostics.ab_recovery --model "${SPEC:?set SPEC}" --name "${NAME:?set NAME}" --epochs "${EPOCHS:-6}" --lr "${LR:-2e-4}"

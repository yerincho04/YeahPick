#!/usr/bin/env bash
#SBATCH --job-name=a2a_text_only
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_text_only_%j.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_text_only_%j.err

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

# Text-only knowledge control: question text through the VLA's language model, no image.
#   sbatch --export=ALL,MODEL=magma slurm/text_only_knowledge.sbatch
# MODEL=openvla|magma|spatialvla|pi0; optional VLA_PATH (checkpoint) and OUT (output dir).
MODEL=${MODEL:-openvla}
case "$MODEL" in
  openvla)    ENV_NAME=openvla_rl4vla ;;
  magma)      ENV_NAME=magma_act2answer ;;
  spatialvla) ENV_NAME=spatialvla_act2answer ;;
  pi0)        ENV_NAME=pi0_act2answer; export PYTHONPATH="$WORKSPACE_ROOT/pi0_deps:${PYTHONPATH:-}" ;;
esac
conda activate "$CONDA_ENVS_DIR/$ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/openvla:$PROJECT_ROOT:${PYTHONPATH:-}" OMP_NUM_THREADS=4
python -m diagnostics.text_only_knowledge --model "$MODEL" ${VLA_PATH:+--vla-path "$VLA_PATH"} \
    --out-dir "${OUT:-outputs/text_only_$MODEL}"

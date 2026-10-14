#!/usr/bin/env bash
#SBATCH --job-name=a2a_setup
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_setup_%x_%j.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_setup_%x_%j.err

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

# Build one model's conda env with the repo's own setup script, then pre-download its weights.
#   sbatch --job-name=setup_magma --export=ALL,MODEL=magma,CKPT=microsoft/Magma-8B slurm/setup_env.sbatch
: "${MODEL:?set MODEL=magma|spatialvla|pi0}"
export A2A_EXTERNAL_DIR="$WORKSPACE_ROOT"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt CONDA_SSL_VERIFY=/etc/pki/tls/certs/ca-bundle.crt
export MAX_JOBS=8
set +u
bash scripts/setup/setup_${MODEL}_env.sh
if [ -n "${CKPT:-}" ]; then
  conda activate "$CONDA_ENVS_DIR/${MODEL}_act2answer"
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CKPT'))"
fi
echo "SETUP_DONE $MODEL"

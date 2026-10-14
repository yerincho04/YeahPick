#!/usr/bin/env bash
#SBATCH --job-name=a2a_photo_build
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_photo_build_%j.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_photo_build_%j.err

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

# ---- 1. build the three real-photo asset sets (Visual Genome photos, VAW labels, SigLIP filter) ----
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt   # cluster proxy; conda's CA bundle rejects it
conda activate "$CONDA_ENVS_DIR/openvla_rl4vla"
ASSETS_LIST=(photo_state_v1 photo_attribute_v1 photo_emotion_v1)
(cd diagnostics/dataset_build && python build_vaw_photo_sets.py \
   --vaw-dir "$WORKSPACE_ROOT/data/vaw" --cache-dir "$WORKSPACE_ROOT/data/vg_images") || true

# ---- 2. submit the 4-condition evaluation for every asset that built, then the analysis ----
job_ids=()
for asset in "${ASSETS_LIST[@]}"; do
  pairs="ManiSkill/mani_skill/assets/carrot/$asset/pairs.json"
  [ -f "$pairs" ] || { echo "SKIP $asset (not built)"; continue; }
  n=$(python -c "import json;print(len(json.load(open('$pairs'))))")
  last=$(( 4 * ((n + 4) / 5) - 1 ))
  jid=$(sbatch --parsable --array=0-$last \
        --export=ALL,ASSETS=$asset,SEED=0,CONDITIONS="knowledge tile_object explicit_spatial neutral" \
        slurm/run_openvla.sbatch)
  echo "SUBMITTED $asset n=$n array=0-$last job=$jid"
  job_ids+=("$jid")
done
if [ ${#job_ids[@]} -gt 0 ]; then
  dep=$(IFS=:; echo "${job_ids[*]}")
  sbatch --dependency=afterany:$dep slurm/photo_analysis.sbatch
fi

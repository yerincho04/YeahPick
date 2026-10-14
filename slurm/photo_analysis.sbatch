#!/usr/bin/env bash
#SBATCH --job-name=a2a_photo_analysis
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_photo_analysis_%j.out
#SBATCH --error=/home/yerincho04/YeahPick/Act2Answer/logs/slurm_photo_analysis_%j.err

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

# ---- analyse every photo asset with both fold schemes; write one combined summary ----
conda activate "$CONDA_ENVS_DIR/openvla_rl4vla"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=outputs/photo_hypothesis
mkdir -p "$OUT"
for asset in photo_state_v1 photo_attribute_v1 photo_emotion_v1; do
  ls -d outputs/openvla-$asset-* >/dev/null 2>&1 || { echo "SKIP $asset (no rollouts)"; continue; }
  mkdir -p "$OUT/$asset"
  for folds in formulation image_pair; do
    python -m diagnostics.hypothesis_pilot --asset $asset --fold-by $folds --out-dir "$OUT/$asset/$folds" \
      > "$OUT/$asset/$folds.log" 2>&1 || echo "ANALYSIS FAILED $asset $folds"
  done
done
python - <<'PY'
import json, pathlib
out = pathlib.Path("outputs/photo_hypothesis")
lines = ["# Real-photo hypothesis test: summary", "",
         "Pre-registered rule: diagnostics/HYPOTHESIS_PILOT.md. Primary = formulation folds; image_pair = robustness check.", "",
         "| Asset | Verdict (primary) | Verdict (image-pair check) | Knowledge intent | Spatial intent |", "|---|---|---|---|---|"]
for asset in ("photo_state_v1", "photo_attribute_v1", "photo_emotion_v1"):
    s = {f: json.loads((out / asset / f / "summary.json").read_text()) if (out / asset / f / "summary.json").exists() else None
         for f in ("formulation", "image_pair")}
    p, r = s["formulation"], s["image_pair"]
    if p is None:
        lines.append(f"| {asset} | not run / failed | | | |"); continue
    ki, si = p["knowledge_intent"], p["explicit_spatial_intent"]
    lines.append(f"| {asset} | **{p['verdict']}** | {r['verdict'] if r else 'n/a'} | {ki[0]}/{ki[1]} | {si[0]}/{si[1]} |")
lines += ["", "Full tables: outputs/photo_hypothesis/<asset>/<folds>/report.md"]
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

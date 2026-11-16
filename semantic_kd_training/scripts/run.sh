#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
MODE="${1:-all}"
shift || true

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
  if [ "${CONDA_PREFIX:-}" != "$TARGET_ENV" ]; then
    conda activate "$TARGET_ENV"
  fi
fi

export PYTHONPATH="$ROOT:$REPO/openvla:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${KD_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false

case "$MODE" in
  prepare) python3 -u "$ROOT/prepare.py" ;;
  cache) python3 -u "$ROOT/cache_teacher.py" "$@" ;;
  train) torchrun --standalone --nnodes=1 --nproc-per-node=2 "$ROOT/train.py" ;;
  test) python3 -u "$ROOT/evaluate.py" "$@" ;;
  all)
    python3 -u "$ROOT/prepare.py"
    python3 -u "$ROOT/cache_teacher.py"
    torchrun --standalone --nnodes=1 --nproc-per-node=2 "$ROOT/train.py"
    python3 -u "$ROOT/evaluate.py"
    ;;
  *) echo "Usage: bash semantic_kd_training/scripts/run.sh {prepare|cache|train|test|all}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
MODE="${1:-all}"

if [ -z "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
fi

export PYTHONPATH="$ROOT:$REPO/openvla:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export TOKENIZERS_PARALLELISM=false

case "$MODE" in
  prepare) python3 -u "$ROOT/prepare.py" ;;
  cache) python3 -u "$ROOT/cache_teacher.py" ;;
  train) python3 -u "$ROOT/train.py" ;;
  test) python3 -u "$ROOT/evaluate.py" ;;
  all)
    python3 -u "$ROOT/prepare.py"
    python3 -u "$ROOT/cache_teacher.py"
    python3 -u "$ROOT/train.py"
    python3 -u "$ROOT/evaluate.py"
    ;;
  *) echo "Usage: bash semantic_kd_training/scripts/run.sh {prepare|cache|train|test|all}" >&2; exit 2 ;;
esac

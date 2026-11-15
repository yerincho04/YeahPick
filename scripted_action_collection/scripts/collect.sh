#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"

if [ -z "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
fi

if ! python3 -c 'import numpy, PIL, trimesh, cairosvg' >/dev/null 2>&1; then
  python3 -m pip install -r "$ROOT/requirements.txt"
fi

export PYTHONPATH="$REPO/SimplerEnv:$REPO/ManiSkill:$REPO/openvla:${PYTHONPATH:-}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# exec makes Python the foreground process, so Ctrl-C stops the entire run.
exec python3 -u "$ROOT/collect.py" "$@"

#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TRAIN_ROOT/.." && pwd)"
MODE="${1:-train}"
shift || true

if ! command -v conda >/dev/null 2>&1 && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
  if [ "${CONDA_PREFIX:-}" != "$TARGET_ENV" ]; then
    conda activate "$TARGET_ENV"
  fi
fi

export PYTHONPATH="$TRAIN_ROOT:$REPO_ROOT/openvla:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false
export STAGE1_DDP_BACKEND="${STAGE1_DDP_BACKEND:-nccl}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${PRISMATIC_CACHE_ROOT:-/opt/yeahpick-cache}}"
case "$MODEL_CACHE_ROOT" in
  /*) ;;
  *) echo "MODEL_CACHE_ROOT must be an absolute path" >&2; exit 2 ;;
esac
export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export TMPDIR="$MODEL_CACHE_ROOT/tmp"
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TMPDIR"
echo "MODEL_CACHE_ROOT=$MODEL_CACHE_ROOT"
df -h "$MODEL_CACHE_ROOT"

case "$MODE" in
  prepare)
    bash "$TRAIN_ROOT/scripts/link_inputs.sh"
    exec python3 -u "$TRAIN_ROOT/prepare.py"
    ;;
  train|resume)
    bash "$TRAIN_ROOT/scripts/link_inputs.sh"
    python3 -u "$TRAIN_ROOT/prepare.py"
    if [ "$MODE" = resume ]; then
      if [ "$#" -lt 1 ] || [ ! -f "$1/latest.json" ]; then
        echo "Usage: bash semantic_grounding_stage1/scripts/run.sh resume RUN_DIR" >&2
        exit 2
      fi
      RUN_DIR="$1"
      export STAGE1_RESUME=1
    else
      RUN_DIR="$TRAIN_ROOT/runs/$(date -u +%Y%m%d_%H%M%S_%N_UTC)"
      export STAGE1_RESUME=0
    fi
    export STAGE1_RUN_DIR="$RUN_DIR"
    echo "RUN_DIR=$RUN_DIR"
    NPROC="$(python3 -c 'from config import CONFIG; print(CONFIG.student_gpu_count)')"
    if [ "$STAGE1_DDP_BACKEND" = nccl ]; then
      NCCL_LIB="$(python3 - <<'PY'
import ctypes
from importlib.metadata import distribution
from pathlib import Path

library = Path(distribution("nvidia-nccl-cu12").locate_file("nvidia/nccl/lib/libnccl.so.2"))
if not library.is_file():
    raise SystemExit(f"Missing NCCL library: {library}")
version = ctypes.c_int()
if ctypes.CDLL(str(library)).ncclGetVersion(ctypes.byref(version)) != 0 or version.value != 21903:
    raise SystemExit(f"Expected NCCL 2.19.3 for torch 2.2.0+cu121; found {version.value}")
print(library)
PY
)"
      export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
      export LD_LIBRARY_PATH="$(dirname "$NCCL_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    exec torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" "$TRAIN_ROOT/train.py"
    ;;
  test)
    if [ "$#" -lt 1 ]; then
      echo "Usage: bash semantic_grounding_stage1/scripts/run.sh test RUN_DIR" >&2
      exit 2
    fi
    RUN_DIR="$1"
    shift
    exec python3 -u "$TRAIN_ROOT/evaluate.py" --run-dir "$RUN_DIR" "$@"
    ;;
  test-prismatic)
    export CUDA_VISIBLE_DEVICES="${PRISMATIC_GPU:-0}"
    if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
      RUN_DIR="$1"
      shift
      exec python3 -u "$TRAIN_ROOT/evaluate_prismatic.py" --run-dir "$RUN_DIR" "$@"
    fi
    bash "$TRAIN_ROOT/scripts/link_inputs.sh"
    python3 -u "$TRAIN_ROOT/prepare.py"
    exec python3 -u "$TRAIN_ROOT/evaluate_prismatic.py" "$@"
    ;;
  *)
    echo "Usage: bash semantic_grounding_stage1/scripts/run.sh {prepare|train|resume|test|test-prismatic}" >&2
    exit 2
    ;;
esac

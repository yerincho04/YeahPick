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

export PYTHONPATH="$TRAIN_ROOT:$REPO_ROOT/openvla:$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false
export BRIDGE_DDP_BACKEND="${BRIDGE_DDP_BACKEND:-nccl}"

new_timestamp_dir() {
  python3 - "$1" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys

parent = Path(sys.argv[1])
parent.mkdir(parents=True, exist_ok=True)
for _ in range(10):
    path = parent / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError:
        continue
    print(path.resolve())
    break
else:
    raise SystemExit("Could not create a unique timestamp directory")
PY
}

case "$MODE" in
  prepare)
    bash "$TRAIN_ROOT/scripts/link_inputs.sh"
    exec python3 -u "$TRAIN_ROOT/prepare.py"
    ;;
  train)
    bash "$TRAIN_ROOT/scripts/link_inputs.sh"
    python3 -u "$TRAIN_ROOT/prepare.py"
    RUN_DIR="$(new_timestamp_dir "$TRAIN_ROOT/runs")"
    export BRIDGE_RUN_DIR="$RUN_DIR"
    echo "RUN_DIR=$RUN_DIR"
    NPROC="$(python3 -c 'from config import CONFIG; print(CONFIG.student_gpu_count)')"
    if [ "$BRIDGE_DDP_BACKEND" = nccl ]; then
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
      echo "Usage: bash semantic_bridge_training/scripts/run.sh test RUN_DIR [options]" >&2
      exit 2
    fi
    RUN_DIR="$1"
    shift
    EVAL_DIR="$(new_timestamp_dir "$RUN_DIR/evaluations")"
    rmdir "$EVAL_DIR"
    echo "EVAL_DIR=$EVAL_DIR"
    exec python3 -u "$TRAIN_ROOT/evaluate.py" --run-dir "$RUN_DIR" --eval-dir "$EVAL_DIR" "$@"
    ;;
  *)
    echo "Usage: bash semantic_bridge_training/scripts/run.sh {prepare|train|test}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TRAIN_ROOT/.." && pwd)"
MODE="${1:-}"
shift || true

if ! command -v conda >/dev/null 2>&1 && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
  if [ "${CONDA_PREFIX:-}" != "$TARGET_ENV" ]; then conda activate "$TARGET_ENV"; fi
fi

export PYTHONPATH="$TRAIN_ROOT:$REPO_ROOT/openvla:$REPO_ROOT/ManiSkill:$REPO_ROOT/SimplerEnv:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false
export STAGE2_DDP_BACKEND="${STAGE2_DDP_BACKEND:-nccl}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/opt/yeahpick-cache}"
case "$MODEL_CACHE_ROOT" in /*) ;; *) echo "MODEL_CACHE_ROOT must be absolute" >&2; exit 2;; esac
export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export TMPDIR="$MODEL_CACHE_ROOT/tmp"
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TMPDIR"

if [ -z "${STAGE1_RUN:-}" ] || [ ! -f "$STAGE1_RUN/best.json" ]; then
  echo "Set STAGE1_RUN to a completed Stage 1 run (best.json required)" >&2
  exit 2
fi
export STAGE1_RUN="$(cd "$STAGE1_RUN" && pwd)"
bash "$TRAIN_ROOT/scripts/link_inputs.sh"

case "$MODE" in
  collect-counterfactual)
    exec python3 -u "$TRAIN_ROOT/collect_counterfactual.py" "$@"
    ;;
  prepare)
    exec python3 -u "$TRAIN_ROOT/prepare.py"
    ;;
  train)
    python3 -u "$TRAIN_ROOT/prepare.py"
    export STAGE2_RUN_DIR="$TRAIN_ROOT/runs/$(date -u +%Y%m%d_%H%M%S_%N_UTC)"
    echo "STAGE2_RUN_DIR=$STAGE2_RUN_DIR"
    NPROC="$(python3 -c 'from config import CONFIG; print(CONFIG.student_gpu_count)')"
    if [ "$STAGE2_DDP_BACKEND" = nccl ]; then
      NCCL_LIB="$(python3 - <<'PY'
import ctypes
from importlib.metadata import distribution
from pathlib import Path
p = Path(distribution("nvidia-nccl-cu12").locate_file("nvidia/nccl/lib/libnccl.so.2"))
v = ctypes.c_int()
if not p.is_file() or ctypes.CDLL(str(p)).ncclGetVersion(ctypes.byref(v)) != 0 or v.value != 21903:
    raise SystemExit(f"Expected NCCL 2.19.3 for CUDA 12.1; found {v.value}")
print(p)
PY
)"
      export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
      export LD_LIBRARY_PATH="$(dirname "$NCCL_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" "$TRAIN_ROOT/train.py"
    for checkpoint in "$STAGE2_RUN_DIR"/checkpoints/epoch_*; do
      name="$(basename "$checkpoint")"
      python3 -u "$TRAIN_ROOT/evaluate.py" --run-dir "$STAGE2_RUN_DIR" \
        --checkpoint "$checkpoint" --split val --mode predicted \
        --pairs-per-category 2 --eval-dir "$STAGE2_RUN_DIR/validation/$name"
    done
    python3 -u "$TRAIN_ROOT/select_best.py"
    echo "STAGE2_DONE run=$STAGE2_RUN_DIR"
    ;;
  select)
    if [ "$#" -lt 1 ]; then
      echo "Usage: bash semantic_grounding_stage2/scripts/run.sh select RUN_DIR" >&2
      exit 2
    fi
    export STAGE2_RUN_DIR="$1"
    for checkpoint in "$STAGE2_RUN_DIR"/checkpoints/epoch_*; do
      name="$(basename "$checkpoint")"
      if [ -f "$STAGE2_RUN_DIR/validation/$name/val_metrics.json" ]; then continue; fi
      python3 -u "$TRAIN_ROOT/evaluate.py" --run-dir "$STAGE2_RUN_DIR" \
        --checkpoint "$checkpoint" --split val --mode predicted \
        --pairs-per-category 2 --eval-dir "$STAGE2_RUN_DIR/validation/$name"
    done
    exec python3 -u "$TRAIN_ROOT/select_best.py"
    ;;
  test)
    if [ "$#" -lt 1 ]; then
      echo "Usage: bash semantic_grounding_stage2/scripts/run.sh test RUN_DIR [eval args]" >&2
      exit 2
    fi
    RUN_DIR="$1"
    shift
    exec python3 -u "$TRAIN_ROOT/evaluate.py" --run-dir "$RUN_DIR" \
      --split test --mode all --eval-dir "$RUN_DIR/test_$(date -u +%Y%m%d_%H%M%S_UTC)" "$@"
    ;;
  *)
    echo "Usage: bash semantic_grounding_stage2/scripts/run.sh {collect-counterfactual|prepare|train|select|test}" >&2
    exit 2
    ;;
esac

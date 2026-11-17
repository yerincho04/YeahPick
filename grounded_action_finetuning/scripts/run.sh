#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
MODE="${1:-train}"
shift || true

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  TARGET_ENV="${CONDA_ENV_PATH:-${CONDA_ENVS_DIR:-$HOME/envs}/openvla_rl4vla}"
  if [ "${CONDA_PREFIX:-}" != "$TARGET_ENV" ]; then
    conda activate "$TARGET_ENV"
  fi
fi

export PYTHONPATH="$PROJECT_ROOT:$REPO_ROOT/openvla:$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${KD_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false
export KD_DDP_BACKEND="${KD_DDP_BACKEND:-nccl}"

stage_inputs() {
  if [ ! -f "$PROJECT_ROOT/input/action_data/episodes.jsonl" ] &&
     [ -f "$REPO_ROOT/scripted_action_collection/output/episodes.jsonl" ]; then
    cp -a "$REPO_ROOT/scripted_action_collection/output/." \
      "$PROJECT_ROOT/input/action_data/"
  fi
  if [ ! -f "$PROJECT_ROOT/input/kd_adapter/adapter_config.json" ] &&
     [ -f "$REPO_ROOT/semantic_kd_training/output_ddp_bf16/best_adapter/adapter_config.json" ]; then
    cp -a "$REPO_ROOT/semantic_kd_training/output_ddp_bf16/best_adapter/." \
      "$PROJECT_ROOT/input/kd_adapter/"
  fi
  if [ ! -f "$PROJECT_ROOT/input/action_data/episodes.jsonl" ] ||
     [ ! -f "$PROJECT_ROOT/input/kd_adapter/adapter_config.json" ]; then
    echo "Missing action data or KD best_adapter under $PROJECT_ROOT/input" >&2
    exit 1
  fi
}

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
    stage_inputs
    exec python3 -u "$PROJECT_ROOT/prepare.py"
    ;;
  train)
    stage_inputs
    python3 -u "$PROJECT_ROOT/prepare.py"
    RUN_DIR="$(new_timestamp_dir "$PROJECT_ROOT/runs")"
    export GROUNDED_RUN_DIR="$RUN_DIR"
    echo "RUN_DIR=$RUN_DIR"
    NPROC="$(python3 -c 'from config import CONFIG; print(CONFIG.student_gpu_count)')"
    if [ "$KD_DDP_BACKEND" = nccl ]; then
      NCCL_LIB="$(python3 - <<'PY'
import ctypes
from importlib.metadata import distribution
from pathlib import Path

pkg = distribution("nvidia-nccl-cu12")
library = Path(pkg.locate_file("nvidia/nccl/lib/libnccl.so.2"))
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
    exec torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" \
      "$PROJECT_ROOT/train.py"
    ;;
  test)
    if [ "$#" -lt 1 ]; then
      echo "Usage: bash grounded_action_finetuning/scripts/run.sh test RUN_DIR [evaluation options]" >&2
      exit 2
    fi
    RUN_DIR="$1"
    shift
    EVAL_DIR="$(new_timestamp_dir "$RUN_DIR/evaluations")"
    # evaluate.py creates the directory itself, so remove only the empty reservation.
    rmdir "$EVAL_DIR"
    echo "EVAL_DIR=$EVAL_DIR"
    exec python3 -u "$PROJECT_ROOT/evaluate.py" --run-dir "$RUN_DIR" \
      --eval-dir "$EVAL_DIR" "$@"
    ;;
  *)
    echo "Usage: bash grounded_action_finetuning/scripts/run.sh {prepare|train|test}" >&2
    exit 2
    ;;
esac

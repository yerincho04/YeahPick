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

export PYTHONPATH="$ROOT:$REPO/openvla:$REPO/SimplerEnv:$REPO/ManiSkill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${KD_GPUS:-0,1}"
export TOKENIZERS_PARALLELISM=false
export KD_DDP_BACKEND="${KD_DDP_BACKEND:-nccl}"

if { [ "$MODE" = train ] || [ "$MODE" = all ] || [ "$MODE" = probe ]; } && [ "$KD_DDP_BACKEND" = nccl ]; then
  # Container CUDA libraries can precede the PyTorch cu121 wheel on the loader path.
  # DDP must load the NCCL release paired with torch 2.2.0+cu121.
  NCCL_LIB="$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    package = distribution("nvidia-nccl-cu12")
except PackageNotFoundError:
    raise SystemExit("Missing nvidia-nccl-cu12; install version 2.19.3")
if package.version != "2.19.3":
    raise SystemExit(
        f"nvidia-nccl-cu12=={package.version}; install version 2.19.3 for torch 2.2.0+cu121"
    )
library = Path(package.locate_file("nvidia/nccl/lib/libnccl.so.2"))
if not library.is_file():
    raise SystemExit(f"NCCL library missing: {library}")
print(library)
PY
)"
  export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
  export LD_LIBRARY_PATH="$(dirname "$NCCL_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

case "$MODE" in
  prepare) python3 -u "$ROOT/prepare.py" ;;
  cache) python3 -u "$ROOT/cache_teacher.py" "$@" ;;
  probe) torchrun --standalone --nnodes=1 --nproc-per-node=2 "$ROOT/scripts/probe_nccl.py" ;;
  train) torchrun --standalone --nnodes=1 --nproc-per-node=2 "$ROOT/train.py" ;;
  test) python3 -u "$ROOT/evaluate_behavior.py" "$@" ;;
  all)
    python3 -u "$ROOT/prepare.py"
    python3 -u "$ROOT/cache_teacher.py"
    torchrun --standalone --nnodes=1 --nproc-per-node=2 "$ROOT/train.py"
    python3 -u "$ROOT/evaluate_behavior.py"
    ;;
  *) echo "Usage: bash semantic_kd_training/scripts/run.sh {prepare|cache|probe|train|test|all}" >&2; exit 2 ;;
esac

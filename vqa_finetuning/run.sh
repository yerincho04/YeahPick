#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 train.py \
  --data "${VQA_DATA_DIR:-data}" --output "${VQA_OUTPUT_DIR:-run}" \
  --epochs 3 --batch-size 1 --accumulation 8 --lr 5e-5 --seed 42 "$@"

#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TRAIN_ROOT/.." && pwd)"
mkdir -p "$TRAIN_ROOT/input"

for name in dataset action_data; do
  if [ "$name" = dataset ]; then
    source_dir="$REPO_ROOT/KD_VQA_simulation_dataset"
    required="public_info/vqa.jsonl"
  else
    source_dir="$REPO_ROOT/action_data"
    required="episodes.jsonl"
  fi
  target="$TRAIN_ROOT/input/$name"
  if [ -L "$target" ] && [ ! -e "$target" ]; then
    echo "Broken input link: $target" >&2
    exit 1
  fi
  if [ ! -e "$target" ]; then
    if [ ! -e "$source_dir/$required" ]; then
      echo "Missing input: $source_dir/$required" >&2
      exit 1
    fi
    ln -s "$source_dir" "$target"
  fi
  if [ ! -e "$target/$required" ]; then
    echo "Incomplete input: $target/$required" >&2
    exit 1
  fi
done

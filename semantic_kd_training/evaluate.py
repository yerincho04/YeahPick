"""Evaluate an adapter on the held-out action-step test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import CONFIG
from model_io import load_action_stats, load_student
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=CONFIG.output_dir / "best_adapter")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    from prismatic.vla.action_tokenizer import ActionTokenizer

    model, processor = load_student(training=False, adapter_path=args.adapter)
    tokenizer = ActionTokenizer(processor.tokenizer)
    result = evaluate(model, processor, tokenizer, load_action_stats(), args.split)
    print(json.dumps(result, indent=2), flush=True)
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

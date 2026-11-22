"""Evaluate the best stage-1 checkpoint on held-out pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from bridge import choice_tokens, policy_output
from model_io import load_student
from train import read_split, save_json, score_pair, summarize


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VQA and robot grounding without rollouts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    pointer = run_dir / "best.json"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    chosen = json.loads(pointer.read_text(encoding="utf-8"))
    adapter = Path(chosen["adapter"])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    pairs = read_split("test", run_dir / "input_snapshot")
    if len(pairs) != 100:
        raise ValueError(f"Expected 100 test pairs, got {len(pairs)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires one CUDA GPU")
    torch.cuda.set_device(0)
    model, processor = load_student(training=False, adapter_path=adapter)
    model.eval()
    for kind in ("side", "choice"):
        choice_tokens(processor.tokenizer, kind)
    output_dir = args.eval_dir or (run_dir / "evaluations" /
                                   datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(pairs[0]["grounding"]["original"]["image"]) as raw:
        side, actions = policy_output(model, processor, raw.convert("RGB"),
                                      pairs[0]["grounding"]["original"]["instruction"])
    if len(actions) != 7:
        raise ValueError("Policy output does not contain seven action tokens")
    grounding, vqa = [], []
    for index, pair in enumerate(pairs, 1):
        g, q = score_pair(model, processor, pair)
        grounding.extend(g)
        vqa.extend(q)
        if index % 10 == 0:
            print(f"TEST pairs={index}/100 grounding_correct={sum(r['correct'] for r in grounding)} "
                  f"vqa_correct={sum(r['correct'] for r in vqa)}", flush=True)
    summary = summarize(grounding, vqa)
    summary["format_check"] = {"side": side, "action_token_count": len(actions),
                               "action_values_trained": False}
    summary["checkpoint"] = chosen
    write_jsonl(output_dir / "test_grounding_predictions.jsonl", grounding)
    write_jsonl(output_dir / "test_vqa_predictions.jsonl", vqa)
    save_json(output_dir / "test_metrics.json", summary)
    print("TEST_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Zero-shot Prismatic evaluation on the exact stage-1 held-out pairs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from bridge import choice_tokens, predict, prompt_for_question
from config import CONFIG
from train import QUESTION_TYPES, read_split, save_json, save_jsonl, summarize


def pixels_for_pair(model, pair: dict, device: torch.device):
    pixels = {}
    for layout in ("original", "swapped"):
        with Image.open(pair["grounding"][layout]["image"]) as raw:
            transformed = model.vision_backbone.image_transform(raw.convert("RGB"))
        if isinstance(transformed, dict):
            pixels[layout] = {key: value.unsqueeze(0).to(device) for key, value in transformed.items()}
        else:
            pixels[layout] = transformed.unsqueeze(0).to(device)
    return pixels


def binary_scores(model, tokenizer, pixel, question: str, kind: str, device: torch.device):
    prefix, first, second = choice_tokens(tokenizer, kind)
    ids = tokenizer(prompt_for_question(question), add_special_tokens=True,
                    return_tensors="pt")["input_ids"]
    if prefix:
        ids = torch.cat((ids, torch.tensor([prefix], dtype=ids.dtype)), dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        logits = model(input_ids=ids.to(device), pixel_values=pixel, use_cache=False,
                       return_dict=True).logits[:, -1, :].float()
    return logits[:, [first, second]]


@torch.inference_mode()
def score_pair(model, tokenizer, pair: dict, device: torch.device):
    pixels = pixels_for_pair(model, pair, device)
    grounding, vqa = [], []
    for layout in ("original", "swapped"):
        item = pair["grounding"][layout]
        choice = predict(binary_scores(model, tokenizer, pixels[layout], item["question"],
                                       "side", device), "side")
        grounding.append({"category": pair["category"], "pair_id": pair["pair_id"],
                          "layout": layout, "expected": item["answer"],
                          "predicted": choice, "correct": choice == item["answer"]})
    for focus in ("left", "right"):
        for kind in QUESTION_TYPES:
            item = pair["vqa"][focus][kind]
            choice = predict(binary_scores(model, tokenizer, pixels["original"], item["question"],
                                           item["choice_kind"], device), item["choice_kind"])
            vqa.append({"category": pair["category"], "pair_id": pair["pair_id"],
                        "sample_id": item["sample_id"], "focus_tile": focus,
                        "question_type": kind, "expected": item["answer"],
                        "predicted": choice, "correct": choice == item["answer"]})
    return grounding, vqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pretrained Prismatic without training")
    parser.add_argument("--run-dir", type=Path,
                        help="Optional stage-1 run whose saved test split should be used")
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--model", default="prism-dinosiglip-224px+7b")
    parser.add_argument("--llama-metadata-repo", default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()
    split_root = args.run_dir.resolve() / "input_snapshot" if args.run_dir else CONFIG.work_dir
    pairs = read_split("test", split_root)
    if len(pairs) != 100:
        raise ValueError(f"Expected 100 test pairs, got {len(pairs)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Prismatic evaluation requires one CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    os.environ["PRISMATIC_LLAMA2_7B_METADATA_REPO"] = args.llama_metadata_repo
    from prismatic.models.load import load

    model = load(args.model, cache_dir=os.environ.get("HF_HUB_CACHE")).to(
        device=device, dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16).eval()
    tokenizer = model.llm_backbone.tokenizer
    for kind in ("side", "choice"):
        choice_tokens(tokenizer, kind)
    results_root = args.run_dir.resolve() / "evaluations" if args.run_dir else CONFIG.runs_dir
    output_dir = args.eval_dir or (results_root /
                                   ("prismatic_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    grounding, vqa = [], []
    for index, pair in enumerate(pairs, 1):
        g, q = score_pair(model, tokenizer, pair, device)
        grounding.extend(g)
        vqa.extend(q)
        if index % 10 == 0:
            print(f"TEST_PRISMATIC pairs={index}/100 "
                  f"grounding_correct={sum(row['correct'] for row in grounding)} "
                  f"vqa_correct={sum(row['correct'] for row in vqa)}", flush=True)
    summary = summarize(grounding, vqa)
    summary.pop("selection_score")
    summary.pop("ready_for_stage2")
    summary["model"] = args.model
    summary["evaluation"] = "zero_shot_binary_choice"
    summary["test_split_source"] = str(split_root)
    save_jsonl(output_dir / "test_grounding_predictions.jsonl", grounding)
    save_jsonl(output_dir / "test_vqa_predictions.jsonl", vqa)
    save_json(output_dir / "test_metrics.json", summary)
    print("TEST_PRISMATIC_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

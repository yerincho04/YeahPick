"""Evaluate explicit grounding and oracle/predicted-side robot rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CONFIG, ROOT
from model_io import load_action_stats, load_student, prompt_for_question
from prepare import grounding_question, vqa_pairs
from prompts import action_prompt
from train import side_token_ids


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.inference_mode()
def forward_prompt(model, processor, image: Image.Image, prompt: str):
    device = next(model.parameters()).device
    ids = processor.tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    pixel = processor.image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(device, dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16)
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        return model(input_ids=ids, pixel_values=pixel, use_cache=True, return_dict=True)


@torch.inference_mode()
def predict_side(model, processor, image: Image.Image, instruction: str) -> str:
    output = forward_prompt(model, processor, image,
                            prompt_for_question(grounding_question(instruction)))
    left_id, right_id = side_token_ids(processor)
    logits = output.logits[0, -1]
    return "Left" if logits[left_id] > logits[right_id] else "Right"


@torch.inference_mode()
def predict_action(model, processor, image: Image.Image,
                   instruction: str, side: str, stats: dict) -> np.ndarray:
    prompt = action_prompt(instruction, side) + " "
    output = forward_prompt(model, processor, image, prompt)
    tokenizer = processor.tokenizer
    first = tokenizer.vocab_size - 256
    tokens = []
    for index in range(7):
        scores = output.logits[:, -1, first:tokenizer.vocab_size]
        token = scores.argmax(dim=-1) + first
        tokens.append(int(token.item()))
        if index < 6:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                output = model(input_ids=token[:, None],
                               past_key_values=output.past_key_values,
                               use_cache=True, return_dict=True)
    bins = (np.linspace(-1, 1, 256)[:-1] + np.linspace(-1, 1, 256)[1:]) / 2
    indices = np.clip(tokenizer.vocab_size - np.asarray(tokens) - 1, 0, 254)
    normalized = bins[indices]
    mask = np.asarray(stats["mask"], dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    raw = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    raw[6] = 1.0 if raw[6] > 0.5 else -1.0
    return raw.astype(np.float32)


def episode_groups(eval_dir: Path):
    sys.path.insert(0, str(ROOT.parent / "scripted_action_collection"))
    import build_assets

    build_assets.INPUT = CONFIG.dataset_root
    build_assets.ASSETS = CONFIG.work_dir / "test_assets"
    pairs = vqa_pairs()
    split_rows = json.loads((CONFIG.work_dir / "pair_splits.json").read_text(encoding="utf-8"))
    selected = defaultdict(set)
    for row in split_rows:
        if row["split"] == "test":
            selected[row["category"]].add(row["pair_id"])
    if not selected:
        raise ValueError("No test pairs in pair_splits.json")
    for category, pair_ids in sorted(selected.items()):
        asset_dir, _ = build_assets.build_category(category)
        all_pairs = json.loads((asset_dir / "pairs.json").read_text(encoding="utf-8"))
        chosen = [pair for pair in all_pairs if pair["pair_id"] in pair_ids]
        if len(chosen) != len(pair_ids):
            raise ValueError(f"Missing test assets for {category}")
        for pair in chosen:
            source = pairs[category, pair["pair_id"]]
            if pair["question"] != source["instruction"]:
                raise ValueError(f"Instruction mismatch: {category}/{pair['pair_id']}")
        yield category, asset_dir.resolve(), chosen


def make_env(asset_dir: Path, ids: list[int], swapped: bool,
             seed: int, episode_len: int):
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper
    from simpler_env.run import Args

    args = Args()
    args.env_id = "Act2AnswerV4-v1"
    args.assets = str(asset_dir)
    args.ids = ids
    args.num_envs = len(ids)
    args.seed = seed
    args.do_swap = swapped
    args.episode_len = episode_len
    args.instruction_condition = "knowledge"
    args.capture_attention = False
    args.rgb_overlay_paths = {
        "3rd_view_camera": str(ROOT.parent / "scripted_action_collection" /
                               "input" / "bridge_real_eval_1.png")
    }
    return SimplerWrapper(args)


def evaluate_mode(model, processor, stats: dict, eval_dir: Path,
                  mode: str, batch_size: int, episode_len: int, seed: int) -> dict:
    path = eval_dir / f"{mode}_episodes.jsonl"
    rows = []
    for category, asset_dir, pairs in episode_groups(eval_dir):
        for swapped in (False, True):
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                env = make_env(asset_dir, [int(pair["index"]) for pair in batch],
                               swapped, seed, episode_len)
                try:
                    frames, instructions, info = env.reset("test")
                    if instructions != [pair["question"] for pair in batch]:
                        raise ValueError(f"Instruction mismatch in {category}")
                    targets = []
                    decisions = []
                    for i, pair in enumerate(batch):
                        target = pair["answer"].title()
                        if swapped:
                            target = "Right" if target == "Left" else "Left"
                        targets.append(target)
                        image = Image.fromarray(frames[i].cpu().numpy())
                        decisions.append(predict_side(model, processor, image,
                                                      instructions[i]))
                    sides = targets if mode == "oracle" else decisions
                    for step in range(episode_len):
                        actions = []
                        for i, instruction in enumerate(instructions):
                            image = Image.fromarray(frames[i].cpu().numpy())
                            actions.append(predict_action(model, processor, image,
                                                          instruction, sides[i], stats))
                        action_tensor = torch.from_numpy(np.stack(actions)).to(frames.device)
                        frames, _, _, info = env.step(action_tensor)
                        if (step + 1) % 10 == 0 or step + 1 == episode_len:
                            print(f"ROLLOUT {mode} {category} swap={swapped} "
                                  f"batch={start // batch_size + 1} "
                                  f"step={step + 1}/{episode_len}", flush=True)
                    for i, pair in enumerate(batch):
                        row = {"category": category, "pair_id": pair["pair_id"],
                               "swapped": swapped, "seed": seed,
                               "expected_side": targets[i],
                               "predicted_side": decisions[i],
                               "action_condition_side": sides[i],
                               "grounding_correct": decisions[i] == targets[i],
                               "success": bool(info["success"][i].item()),
                               "answered": bool(info["is_answered"][i].item())}
                        rows.append(row)
                        append_jsonl(path, row)
                    print(f"DONE {mode} {category} swap={swapped} "
                          f"episodes={min(start + batch_size, len(pairs))}/{len(pairs)}",
                          flush=True)
                finally:
                    env.env.close()
    by_category = defaultdict(list)
    by_layout = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_layout["swapped" if row["swapped"] else "original"].append(row)

    def summary(items):
        return {"total": len(items),
                "success": sum(row["success"] for row in items),
                "answered": sum(row["answered"] for row in items),
                "grounding_correct": sum(row["grounding_correct"] for row in items)}

    return {"overall": summary(rows),
            "by_category": {k: summary(v) for k, v in sorted(by_category.items())},
            "by_layout": {k: summary(v) for k, v in sorted(by_layout.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("predicted", "oracle", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.episode_len < 1:
        parser.error("--batch-size and --episode-len must be positive")
    best = json.loads((args.run_dir / "best.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")
    torch.cuda.set_device(1 if torch.cuda.device_count() > 1 else 0)
    model, processor = load_student(training=False,
                                    adapter_path=Path(best["adapter"]))
    model.eval()
    stats = load_action_stats()
    torch.cuda.set_device(0)
    args.eval_dir.mkdir(parents=True, exist_ok=False)
    result = {"checkpoint": best, "eval_dir": str(args.eval_dir.resolve())}
    modes = ("predicted", "oracle") if args.mode == "both" else (args.mode,)
    for mode in modes:
        result[mode] = evaluate_mode(model, processor, stats, args.eval_dir, mode,
                                     args.batch_size, args.episode_len, args.seed)
        (args.eval_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

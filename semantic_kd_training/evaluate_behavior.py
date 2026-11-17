"""Free-generation VQA accuracy and closed-loop robot success on held-out pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cache_teacher import answer_is_correct
from config import CONFIG, ROOT
from model_io import load_action_stats, load_student, prompt_for_question, read_manifest


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


@torch.inference_mode()
def generate_tokens(model, processor, image: Image.Image, prompt: str,
                    max_tokens: int, action_only: bool = False) -> list[int]:
    tokenizer = processor.tokenizer
    # Match the space after Out: in the OpenVLA action prompt.
    if action_only:
        prompt += " "
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    pixel = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel = pixel.to(device, dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16)
    tokens = []
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        output = model(input_ids=ids, pixel_values=pixel, use_cache=True, return_dict=True)
        for index in range(max_tokens):
            scores = output.logits[:, -1, :]
            if action_only:
                first = tokenizer.vocab_size - 256
                token = scores[:, first:tokenizer.vocab_size].argmax(dim=-1) + first
            else:
                token = scores.argmax(dim=-1)
            token_id = int(token.item())
            if not action_only and token_id == tokenizer.eos_token_id:
                break
            tokens.append(token_id)
            if index + 1 < max_tokens:
                output = model(input_ids=token[:, None], past_key_values=output.past_key_values,
                               use_cache=True, return_dict=True)
    return tokens


def accuracy(rows: list[dict], key: str) -> dict:
    correct = sum(bool(row[key]) for row in rows)
    return {"correct": correct, "total": len(rows),
            "accuracy": correct / len(rows) if rows else None}


def evaluate_vqa(model, processor, output_dir: Path) -> dict:
    rows = read_manifest("test", "_vqa")
    predictions = []
    path = output_dir / "test_vqa_predictions.jsonl"
    path.unlink(missing_ok=True)
    for index, sample in enumerate(rows, 1):
        with Image.open(sample["image"]) as raw:
            tokens = generate_tokens(model, processor, raw.convert("RGB"),
                                     prompt_for_question(sample["question"]),
                                     CONFIG.student_max_answer_tokens)
        answer = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        result = {"sample_id": sample["sample_id"], "category": sample["category"],
                  "question_type": sample["question_type"], "focus_tile": sample["focus_tile"],
                  "expected": sample["answer"], "predicted": answer,
                  "correct": answer_is_correct(sample, answer)}
        predictions.append(result)
        append_jsonl(path, result)
        if index % 50 == 0 or index == len(rows):
            print(f"VQA {index}/{len(rows)} correct={sum(r['correct'] for r in predictions)}", flush=True)
    by_category = defaultdict(list)
    by_type = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)
        by_type[row["question_type"]].append(row)
    return {"overall": accuracy(predictions, "correct"),
            "by_category": {k: accuracy(v, "correct") for k, v in sorted(by_category.items())},
            "by_question_type": {k: accuracy(v, "correct") for k, v in sorted(by_type.items())}}


def decode_action(tokens: list[int], tokenizer, stats: dict) -> np.ndarray:
    if len(tokens) != 7:
        raise ValueError(f"Expected seven action tokens, got {len(tokens)}")
    edges = np.linspace(-1, 1, 256)
    bins = (edges[:-1] + edges[1:]) / 2
    indices = np.clip(tokenizer.vocab_size - np.asarray(tokens) - 1, 0, 254)
    normalized = bins[indices]
    mask = np.asarray(stats["mask"], dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    raw = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    raw[6] = 1.0 if raw[6] > 0.5 else -1.0
    return raw.astype(np.float32)


def episode_groups() -> list[tuple[str, Path, list[dict]]]:
    import sys

    sys.path.insert(0, str(ROOT / "scripted_action_collection"))
    import build_assets
    from prepare import vqa_pairs

    build_assets.INPUT = CONFIG.dataset_root
    build_assets.ASSETS = CONFIG.work_dir / "test_assets"
    dataset_pairs = vqa_pairs()
    split_path = CONFIG.work_dir / "pair_splits.json"
    selected = defaultdict(set)
    for row in json.loads(split_path.read_text(encoding="utf-8")):
        if row["split"] == "test":
            selected[row["category"]].add(row["pair_id"])
    if not selected:
        raise ValueError("No test pairs in pair_splits.json")
    result = []
    for category, pair_ids in sorted(selected.items()):
        asset_dir, _ = build_assets.build_category(category)
        all_pairs = json.loads((asset_dir / "pairs.json").read_text(encoding="utf-8"))
        pairs = [p for p in all_pairs if p["pair_id"] in pair_ids]
        if len(pairs) != len(pair_ids):
            missing = pair_ids - {p["pair_id"] for p in pairs}
            raise ValueError(f"Missing test assets in {category}: {sorted(missing)}")
        for pair in pairs:
            source = dataset_pairs[category, pair["pair_id"]]
            if pair["question"] != source["instruction"]:
                raise ValueError(f"Test instruction differs from training: {category}/{pair['pair_id']}")
            for side in ("left", "right"):
                image_id = source["images"][side]["image_id"]
                if pair[side] != f"tile_{image_id}":
                    raise ValueError(f"Test tile differs from training: {category}/{pair['pair_id']}/{side}")
        result.append((category, asset_dir.resolve(), pairs))
    return result


def make_env(asset_dir: Path, ids: list[int], swapped: bool, seed: int,
             episode_len: int):
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
        "3rd_view_camera": str(ROOT / "scripted_action_collection" / "input" / "bridge_real_eval_1.png")
    }
    return SimplerWrapper(args)


def evaluate_rollout(model, processor, output_dir: Path, batch_size: int,
                     episode_len: int, seed: int) -> dict:
    stats = load_action_stats()
    rows = []
    path = output_dir / "test_rollout_episodes.jsonl"
    path.unlink(missing_ok=True)
    for category, asset_dir, pairs in episode_groups():
        for swapped in (False, True):
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                env = make_env(asset_dir, [int(p["index"]) for p in batch], swapped,
                               seed, episode_len)
                try:
                    frames, instructions, info = env.reset("test")
                    if [p["question"] for p in batch] != instructions:
                        raise ValueError(f"Asset instructions differ from training data: {category}")
                    for step in range(episode_len):
                        actions = []
                        for i, instruction in enumerate(instructions):
                            image = Image.fromarray(frames[i].cpu().numpy())
                            prompt = prompt_for_question(
                                f"What action should the robot take to {instruction.lower()}?")
                            tokens = generate_tokens(model, processor, image, prompt, 7,
                                                     action_only=True)
                            actions.append(decode_action(tokens, processor.tokenizer, stats))
                        action_tensor = torch.from_numpy(np.stack(actions)).to(frames.device)
                        frames, _, _, info = env.step(action_tensor)
                        if (step + 1) % 10 == 0 or step + 1 == episode_len:
                            print(f"ROLLOUT_STEP {category} swap={swapped} "
                                  f"batch={start // batch_size + 1} "
                                  f"step={step + 1}/{episode_len}", flush=True)
                    for i, pair in enumerate(batch):
                        answer_side = pair["answer"].lower()
                        if swapped:
                            answer_side = "right" if answer_side == "left" else "left"
                        result = {"category": category, "pair_id": pair["pair_id"],
                                  "swapped": swapped, "seed": seed,
                                  "expected_side": answer_side,
                                  "success": bool(info["success"][i].item()),
                                  "answered": bool(info["is_answered"][i].item()),
                                  "first_choice_correct": bool(info["tcp_first_choice_correct"][i].item())}
                        rows.append(result)
                        append_jsonl(path, result)
                    print(f"ROLLOUT {category} swap={swapped} "
                          f"episodes={min(start + batch_size, len(pairs))}/{len(pairs)} "
                          f"success={sum(r['success'] for r in rows)}/{len(rows)}", flush=True)
                finally:
                    env.env.close()
    by_category = defaultdict(list)
    by_layout = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_layout["swapped" if row["swapped"] else "original"].append(row)

    def summary(items: list[dict]) -> dict:
        return {"success": accuracy(items, "success"),
                "answered": accuracy(items, "answered"),
                "first_choice_correct": accuracy(items, "first_choice_correct")}

    return {"overall": summary(rows),
            "by_category": {k: summary(v) for k, v in sorted(by_category.items())},
            "by_layout": {k: summary(v) for k, v in sorted(by_layout.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Free-generation VQA and closed-loop robot test")
    parser.add_argument("--adapter", type=Path, default=CONFIG.output_dir / "best_adapter")
    parser.add_argument("--rollout-batch-size", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.rollout_batch_size < 1 or args.episode_len < 1:
        parser.error("--rollout-batch-size and --episode-len must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Test requires a CUDA GPU")
    # Same device split as SimplerEnv: GPU 0 renders, GPU 1 holds OpenVLA.
    torch.cuda.set_device(1 if torch.cuda.device_count() > 1 else 0)
    model, processor = load_student(training=False, adapter_path=args.adapter)
    model.eval()
    # ManiSkill uses the default CUDA device for simulation and rendering.
    torch.cuda.set_device(0)
    output_dir = CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").unlink(missing_ok=True)
    result = {"split": "test", "adapter": str(args.adapter.resolve()),
              "vqa": evaluate_vqa(model, processor, output_dir)}
    write_json(output_dir / "test_vqa_metrics.json", result)
    result["rollout"] = evaluate_rollout(model, processor, output_dir,
                                          args.rollout_batch_size, args.episode_len,
                                          args.seed)
    write_json(output_dir / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

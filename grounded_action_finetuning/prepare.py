"""Join VQA Knowledge pairs to successful action episodes and split by pair."""

from __future__ import annotations

import json
import random
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from config import CONFIG


def vqa_pairs() -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted(CONFIG.dataset_root.glob("*/vqa.jsonl")):
        category = path.parent.name
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = category, row["pair_id"]
                answer = row["knowledge"]["expected_answer"].strip().lower()
                instruction = row["knowledge"]["instruction"]
                if key in result:
                    if (result[key]["answer"] != answer
                            or result[key]["instruction"] != instruction
                            or result[key]["images"] != row["images"]):
                        raise ValueError(f"Inconsistent VQA Knowledge rows for {key}")
                else:
                    result[key] = {"answer": answer, "instruction": instruction,
                                   "images": row["images"], "rows": []}
                result[key]["rows"].append(row)
    if not result:
        raise FileNotFoundError(f"No <category>/vqa.jsonl under {CONFIG.dataset_root}")
    for key, pair in result.items():
        if {row["focus_tile"].lower() for row in pair["rows"]} != {"left", "right"}:
            raise ValueError(f"Both focus rows are required for {key}")
        for row in pair["rows"]:
            if len(row["vqa"]) != 5:
                raise ValueError(f"Expected five VQA questions in {row['group_id']}")
    return result


def load_tile(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".svg":
        import cairosvg
        with Image.open(io.BytesIO(cairosvg.svg2png(
                url=str(path), output_width=512, output_height=512))) as image:
            rgba = image.convert("RGBA")
    else:
        with Image.open(path) as image:
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def pair_image(category: str, pair_id: str, images: dict) -> Path:
    path = CONFIG.work_dir / "pair_images" / category / f"{pair_id}.png"
    if path.is_file():
        return path
    canvas = Image.new("RGB", (1024, 512), "white")
    for side, x in (("left", 0), ("right", 512)):
        tile = load_tile(CONFIG.dataset_root / images[side]["path"])
        canvas.paste(ImageOps.pad(tile, (512, 512), color="white"), (x, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def pair_splits(pairs: dict) -> dict[tuple[str, str], str]:
    if abs(CONFIG.train_ratio + CONFIG.val_ratio + CONFIG.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    by_category = defaultdict(list)
    for category, pair_id in pairs:
        by_category[category].append(pair_id)
    rng = random.Random(CONFIG.split_seed)
    assignments = {}
    for category, ids in sorted(by_category.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_train = round(len(ids) * CONFIG.train_ratio)
        n_val = round(len(ids) * CONFIG.val_ratio)
        for index, pair_id in enumerate(ids):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            assignments[category, pair_id] = split
    return assignments


def episode_directory(row: dict) -> Path:
    layout = "swapped" if row["swapped"] else "original"
    return (CONFIG.action_data_root / "episodes" / row["category"] / row["pair_id"] /
            layout / f"seed_{row['seed']}")


def grounding_question(instruction: str) -> str:
    return (f"{instruction.strip()} Which side is the correct destination for the cube? "
            "Answer with Left or Right.")


def main() -> None:
    pairs = vqa_pairs()
    manifest = CONFIG.action_data_root / "episodes.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(encoding="utf-8") as stream:
        successful = [json.loads(line) for line in stream]
    successful = [row for row in successful if row.get("success")]
    available_pairs = {(row["category"], row["pair_id"]) for row in successful}
    missing = available_pairs - pairs.keys()
    if missing:
        raise ValueError(f"Action episodes have no matching VQA pairs: {sorted(missing)[:5]}")
    assignments = pair_splits(pairs)
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)
    samples = {split: [] for split in ("train", "val", "test")}
    vqa_samples = {split: [] for split in samples}
    grounding_samples = {split: [] for split in samples}
    for (category, pair_id), pair in sorted(pairs.items()):
        image = pair_image(category, pair_id, pair["images"])
        split = assignments[category, pair_id]
        for focus_row in pair["rows"]:
            for question_index, item in enumerate(focus_row["vqa"]):
                vqa_samples[split].append({
                    "sample_id": f"vqa/{category}/{focus_row['group_id']}/q{question_index}",
                    "kind": "vqa", "category": category, "pair_id": pair_id,
                    "focus_tile": focus_row["focus_tile"], "question_type": item["type"],
                    "question": item["question"], "answer": str(item["expected_answer"]),
                    "choices": item.get("choices"),
                    "image": str(image.resolve()),
                })
    train_actions = []
    seen_episodes = set()
    for row in successful:
        category, pair_id = row["category"], row["pair_id"]
        key = category, pair_id
        episode_id = row["episode_id"]
        if episode_id in seen_episodes:
            continue
        seen_episodes.add(episode_id)
        expected_instruction = pairs[key]["instruction"]
        if row["instruction"] != expected_instruction:
            raise ValueError(f"Instruction mismatch in {episode_id}")
        expected_side = pairs[key]["answer"]
        if row["swapped"]:
            expected_side = "right" if expected_side == "left" else "left"
        if row["answer_side"].lower() != expected_side:
            raise ValueError(f"Answer-side mismatch in {episode_id}")
        folder = episode_directory(row)
        actions_path = folder / "actions.npy"
        if not actions_path.is_file():
            raise FileNotFoundError(actions_path)
        actions = np.load(actions_path)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected [steps,7] actions in {actions_path}")
        if len(actions) == 0:
            raise ValueError(f"No actions in {actions_path}")
        split = assignments[key]
        if split == "train":
            train_actions.append(actions)
        first_frame = folder / "frames" / "0000.jpg"
        if not first_frame.is_file():
            raise FileNotFoundError(first_frame)
        grounding_samples[split].append({
            "sample_id": f"grounding/{episode_id}", "kind": "grounding",
            "category": category, "pair_id": pair_id, "episode_id": episode_id,
            "question": grounding_question(expected_instruction),
            "answer": expected_side.title(), "image": str(first_frame.resolve()),
        })
        for step, action in enumerate(actions):
            image = folder / "frames" / f"{step:04d}.jpg"
            if not image.is_file():
                raise FileNotFoundError(image)
            samples[split].append({
                "sample_id": f"{episode_id}/step_{step:04d}",
                "category": category, "pair_id": pair_id,
                "episode_id": episode_id, "step": step,
                "image": str(image.resolve()),
                "instruction": expected_instruction,
                "answer_side": expected_side.title(),
                "action": action.tolist(),
            })

    if not train_actions:
        raise ValueError("No successful action episodes in the train split")
    stacked = np.concatenate(train_actions).astype(np.float32)
    q01 = np.quantile(stacked, 0.01, axis=0)
    q99 = np.quantile(stacked, 0.99, axis=0)
    small_span = (q99 - q01) < 1e-4
    center = (q99 + q01) / 2
    q01[small_span] = center[small_span] - 0.005
    q99[small_span] = center[small_span] + 0.005
    mask = np.ones(7, dtype=bool)
    mask[-1] = False  # gripper is already in [-1, +1]
    stats = {CONFIG.action_unnorm_key: {"action": {
        "q01": q01.tolist(), "q99": q99.tolist(),
        "min": stacked.min(axis=0).tolist(), "max": stacked.max(axis=0).tolist(),
        "mean": stacked.mean(axis=0).tolist(), "std": stacked.std(axis=0).tolist(),
        "mask": mask.tolist(),
    }}}
    (CONFIG.work_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    split_rows = [
        {"category": category, "pair_id": pair_id, "split": split}
        for (category, pair_id), split in sorted(assignments.items())
    ]
    (CONFIG.work_dir / "pair_splits.json").write_text(
        json.dumps(split_rows, indent=2), encoding="utf-8"
    )
    for split in samples:
        for suffix, rows in (("", samples[split]), ("_vqa", vqa_samples[split]),
                             ("_grounding", grounding_samples[split])):
            with (CONFIG.work_dir / f"{split}{suffix}.jsonl").open("w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: action_steps={len(samples[split])} "
              f"vqa_questions={len(vqa_samples[split])} "
              f"robot_grounding={len(grounding_samples[split])}", flush=True)


if __name__ == "__main__":
    main()

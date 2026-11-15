"""Join VQA Knowledge pairs to successful action episodes and split by pair."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

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
                    if result[key] != {"answer": answer, "instruction": instruction}:
                        raise ValueError(f"Inconsistent VQA Knowledge rows for {key}")
                else:
                    result[key] = {"answer": answer, "instruction": instruction}
    if not result:
        raise FileNotFoundError(f"No <category>/vqa.jsonl under {CONFIG.dataset_root}")
    return result


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


def semantic_question(instruction: str) -> str:
    return ("Which tile satisfies this instruction: "
            f"\"{instruction.strip()}\" Answer with Left or Right.")


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
    assignments = pair_splits({key: pairs[key] for key in available_pairs})
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)
    samples = {split: [] for split in ("train", "val", "test")}
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
        split = assignments[key]
        if split == "train":
            train_actions.append(actions)
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
                "semantic_question": semantic_question(expected_instruction),
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
    for split, rows in samples.items():
        with (CONFIG.work_dir / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} action steps", flush=True)


if __name__ == "__main__":
    main()

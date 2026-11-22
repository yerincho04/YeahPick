"""Make pair-level splits and verify scene, label, and layout alignment."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from bridge import policy_question, semantic_side_question
from config import CONFIG


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pairs() -> dict[tuple[str, str], dict]:
    pairs = {}
    for path in sorted(CONFIG.dataset_root.glob("*/vqa.jsonl")):
        category = path.parent.name
        for row in read_jsonl(path):
            key = category, row["pair_id"]
            knowledge = row["knowledge"]
            existing = pairs.setdefault(key, {"instruction": knowledge["instruction"],
                                              "answer": knowledge["expected_answer"].title(),
                                              "image": (CONFIG.dataset_root / row["image"]).resolve(),
                                              "focus": {}})
            if (existing["instruction"] != knowledge["instruction"] or
                    existing["answer"] != knowledge["expected_answer"].title() or
                    existing["image"] != (CONFIG.dataset_root / row["image"]).resolve()):
                raise ValueError(f"Conflicting knowledge rows: {key}")
            focus = row["focus_tile"].lower()
            if focus not in ("left", "right") or focus in existing["focus"]:
                raise ValueError(f"Missing or duplicate focus: {key}, {focus}")
            if not existing["image"].is_file():
                raise FileNotFoundError(existing["image"])
            questions = {}
            for index, item in enumerate(row["vqa"]):
                kind = item["type"]
                if kind in questions:
                    raise ValueError(f"Duplicate question type: {key}, {kind}")
                question, answer = item["question"], str(item["expected_answer"])
                if item.get("choices") and item["choices"].get(answer) != item.get("original_expected_answer"):
                    raise ValueError(f"Answer/choice mismatch: {key}, {focus}, {kind}")
                if kind == "Semantic reasoning":
                    question, answer = semantic_side_question(question, answer)
                choice_kind = ("choice" if kind in ("Object recognition", "Attribute recognition")
                               else "side")
                allowed = ("A", "B") if choice_kind == "choice" else ("Left", "Right")
                if answer not in allowed:
                    raise ValueError(f"Invalid {kind} answer in {key}: {answer}")
                questions[kind] = {"sample_id": f"{category}/{row['group_id']}/q{index}",
                                   "question_type": kind, "focus_tile": focus,
                                   "question": question, "answer": answer,
                                   "choice_kind": choice_kind,
                                   "image": str(existing["image"])}
            if len(questions) != 5:
                raise ValueError(f"Expected five question types: {key}, {focus}")
            existing["focus"][focus] = questions
    if not pairs:
        raise ValueError(f"No VQA pairs under {CONFIG.dataset_root}")
    by_category = defaultdict(int)
    for key, pair in pairs.items():
        if set(pair["focus"]) != {"left", "right"}:
            raise ValueError(f"Both focus rows are required: {key}")
        by_category[key[0]] += 1
    if len(by_category) != 10 or any(count != 100 for count in by_category.values()):
        raise ValueError(f"Expected ten categories with 100 pairs each: {dict(by_category)}")
    return pairs


def split_pairs(pairs: dict) -> dict[tuple[str, str], str]:
    if abs(CONFIG.train_ratio + CONFIG.val_ratio + CONFIG.test_ratio - 1) > 1e-8:
        raise ValueError("Split ratios must sum to one")
    by_category = defaultdict(list)
    for category, pair_id in pairs:
        by_category[category].append(pair_id)
    rng = random.Random(CONFIG.split_seed)
    split = {}
    for category, ids in sorted(by_category.items()):
        ids.sort()
        rng.shuffle(ids)
        n_train = round(len(ids) * CONFIG.train_ratio)
        n_val = round(len(ids) * CONFIG.val_ratio)
        for index, pair_id in enumerate(ids):
            split[category, pair_id] = ("train" if index < n_train else
                                        "val" if index < n_train + n_val else "test")
    return split


def episode_frame(row: dict) -> Path:
    layout = "swapped" if row["swapped"] else "original"
    return (CONFIG.action_data_root / "episodes" / row["category"] / row["pair_id"] /
            layout / f"seed_{row['seed']}" / "frames" / "0000.jpg").resolve()


def main() -> None:
    pairs = load_pairs()
    splits = split_pairs(pairs)
    episodes = {}
    for row in read_jsonl(CONFIG.action_data_root / "episodes.jsonl"):
        if not row.get("success") or row.get("seed") != 0:
            continue
        key = row["category"], row["pair_id"]
        if key not in pairs:
            raise ValueError(f"Episode has no VQA pair: {key}")
        layout = "swapped" if row["swapped"] else "original"
        if (tuple(row.get("tile_xy_left", ())) != CONFIG.tile_xy_left or
                tuple(row.get("tile_xy_right", ())) != CONFIG.tile_xy_right):
            raise ValueError(f"Tile positions differ: {key}, {layout}")
        expected = pairs[key]["answer"]
        if row["swapped"]:
            expected = "Right" if expected == "Left" else "Left"
        if row["instruction"] != pairs[key]["instruction"] or row["answer_side"].title() != expected:
            raise ValueError(f"Instruction or answer-side mismatch: {key}, {layout}")
        frame = episode_frame(row)
        if not frame.is_file():
            raise FileNotFoundError(frame)
        record = {"episode_id": row["episode_id"], "layout": layout,
                  "image": str(frame), "question": policy_question(row["instruction"]),
                  "instruction": row["instruction"], "answer": expected,
                  "choice_kind": "side"}
        old = episodes.setdefault(key, {}).get(layout)
        if old is not None and old != record:
            raise ValueError(f"Conflicting successful seed-0 episodes: {key}, {layout}")
        episodes[key][layout] = record

    output = {split: [] for split in ("train", "val", "test")}
    for key, pair in sorted(pairs.items()):
        by_layout = episodes.get(key, {})
        if set(by_layout) != {"original", "swapped"}:
            raise ValueError(f"Both successful seed-0 layouts required: {key}")
        original = Path(by_layout["original"]["image"])
        if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(pair["image"].read_bytes()).digest():
            raise ValueError(f"VQA image differs from original first action frame: {key}")
        questions = {focus: pair["focus"][focus] for focus in ("left", "right")}
        output[splits[key]].append({"category": key[0], "pair_id": key[1],
                                    "grounding": by_layout, "vqa": questions})
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        write_jsonl(CONFIG.work_dir / f"{split}.jsonl", rows)
        print(f"{split}: pairs={len(rows)} grounding={2*len(rows)} vqa={10*len(rows)}", flush=True)
    (CONFIG.work_dir / "pair_splits.json").write_text(json.dumps(
        [{"category": key[0], "pair_id": key[1], "split": split}
         for key, split in sorted(splits.items())], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

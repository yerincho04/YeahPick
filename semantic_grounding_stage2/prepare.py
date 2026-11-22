"""Validate Stage 1 splits and join semantic/opposite demonstrations."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from config import CONFIG


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vqa_pairs() -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted(CONFIG.dataset_root.glob("*/vqa.jsonl")):
        for row in read_jsonl(path):
            key = path.parent.name, row["pair_id"]
            pair = result.setdefault(key, {"instruction": row["knowledge"]["instruction"],
                                           "answer": row["knowledge"]["expected_answer"].lower(),
                                           "images": row["images"], "rows": []})
            if pair["instruction"] != row["knowledge"]["instruction"] or pair["answer"] != row["knowledge"]["expected_answer"].lower():
                raise ValueError(f"Conflicting VQA pair: {key}")
            pair["rows"].append(row)
    return result


def split_map(snapshot: Path) -> dict[tuple[str, str], str]:
    rows = json.loads((snapshot / "pair_splits.json").read_text(encoding="utf-8"))
    return {(row["category"], row["pair_id"]): row["split"] for row in rows}


def episode_folder(root: Path, row: dict) -> Path:
    return root / "episodes" / row["category"] / row["pair_id"] / ("swapped" if row["swapped"] else "original") / f"seed_{row['seed']}"


def main() -> None:
    snapshot = CONFIG.stage1_run / "input_snapshot"
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Set STAGE1_RUN to the Stage 1 run directory: {snapshot}")
    splits, pairs = split_map(snapshot), vqa_pairs()
    if set(splits) != set(pairs):
        raise ValueError("Stage 1 splits and VQA pairs differ")
    for split in ("train", "val", "test"):
        for row in read_jsonl(snapshot / f"{split}.jsonl"):
            key = row["category"], row["pair_id"]
            if splits[key] != split or row["grounding"]["original"]["instruction"] != pairs[key]["instruction"]:
                raise ValueError(f"Stage 1 instruction/split mismatch: {key}")
            if row["grounding"]["original"]["answer"].lower() != pairs[key]["answer"]:
                raise ValueError(f"Stage 1 answer mismatch: {key}")
    stage1 = json.loads((CONFIG.stage1_run / "best.json").read_text(encoding="utf-8"))
    if not Path(stage1["adapter"]).is_dir():
        raise FileNotFoundError(stage1["adapter"])
    actions_by_split = {s: [] for s in ("train", "val", "test")}
    grounding_by_split = {s: [] for s in actions_by_split}
    found = defaultdict(set)
    train_actions = []
    for kind, root in (("semantic", CONFIG.action_data_root), ("opposite", CONFIG.counterfactual_root)):
        latest = {row["episode_id"]: row for row in read_jsonl(root / "episodes.jsonl")}
        for row in latest.values():
            if not row.get("success") or row.get("seed") != 0:
                continue
            key = row["category"], row["pair_id"]
            if key not in splits:
                raise ValueError(f"Unknown pair: {key}")
            split = splits[key]
            if kind == "opposite" and split != "train":
                raise ValueError(f"Opposite demo contains {split} pair: {key}")
            semantic = pairs[key]["answer"]
            if row["swapped"]:
                semantic = "right" if semantic == "left" else "left"
            condition = semantic if kind == "semantic" else ("right" if semantic == "left" else "left")
            if (row["answer_side"].lower() != semantic or
                    row.get("condition_side", semantic).lower() != condition or
                    row["instruction"] != pairs[key]["instruction"]):
                raise ValueError(f"Demo label mismatch: {row['episode_id']}")
            if (tuple(row["tile_xy_left"]) != CONFIG.tile_xy_left or
                    tuple(row["tile_xy_right"]) != CONFIG.tile_xy_right):
                raise ValueError(f"Tile geometry mismatch: {row['episode_id']}")
            folder = episode_folder(root, row)
            actions = np.load(folder / "actions.npy")
            if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
                raise ValueError(f"Invalid actions: {folder}")
            found[kind, key, bool(row["swapped"])].add(row["seed"])
            if split == "train":
                train_actions.append(actions)
            if kind == "semantic":
                frame = folder / "frames" / "0000.jpg"
                grounding_by_split[split].append({
                    "sample_id": row["episode_id"], "episode_id": row["episode_id"],
                    "category": key[0], "pair_id": key[1], "instruction": row["instruction"],
                    "answer": semantic.title(), "image": str(frame.resolve())})
            for step, action in enumerate(actions):
                image = folder / "frames" / f"{step:04d}.jpg"
                if not image.is_file():
                    raise FileNotFoundError(image)
                actions_by_split[split].append({
                    "sample_id": f"{kind}/{row['episode_id']}/{step}",
                    "kind": kind, "category": key[0], "pair_id": key[1],
                    "episode_id": f"{kind}/{row['episode_id']}", "swapped": bool(row["swapped"]),
                    "step": step, "image": str(image.resolve()), "instruction": row["instruction"],
                    "answer_side": condition.title(), "semantic_side": semantic.title(),
                    "action": action.tolist()})
    for key, split in splits.items():
        for swapped in (False, True):
            if not found["semantic", key, swapped]:
                raise ValueError(f"Missing semantic demo: {key}, swapped={swapped}")
            if split == "train" and not found["opposite", key, swapped]:
                raise ValueError(f"Missing opposite demo: {key}, swapped={swapped}")
    stacked = np.concatenate(train_actions).astype(np.float32)
    q01, q99 = np.quantile(stacked, [0.01, 0.99], axis=0)
    small = q99 - q01 < 1e-4
    center = (q01 + q99) / 2
    q01[small], q99[small] = center[small] - .005, center[small] + .005
    stats = {CONFIG.action_unnorm_key: {"action": {
        "q01": q01.tolist(), "q99": q99.tolist(), "mask": [True] * 6 + [False]}}}
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.work_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.copy2(snapshot / "pair_splits.json", CONFIG.work_dir / "pair_splits.json")
    for split in actions_by_split:
        for suffix, rows in (("", actions_by_split[split]), ("_grounding", grounding_by_split[split])):
            with (CONFIG.work_dir / f"{split}{suffix}.jsonl").open("w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"PREPARED {split} actions={len(actions_by_split[split])} grounding={len(grounding_by_split[split])}", flush=True)


if __name__ == "__main__":
    main()

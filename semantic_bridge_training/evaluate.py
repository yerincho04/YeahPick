"""Evaluate bridge decisions, VQA, and predicted/oracle robot rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from answer_scoring import answer_is_correct
from bridge import generate_action_tokens, score_side
from config import CONFIG
from model_io import load_action_stats, load_student, prompt_for_question, read_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


@torch.inference_mode()
def generate_tokens(model, processor, image: Image.Image, prompt: str,
                    max_tokens: int) -> list[int]:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    pixel = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel = pixel.to(device, dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16)
    tokens = []
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        output = model(input_ids=ids, pixel_values=pixel, use_cache=True, return_dict=True)
        for index in range(max_tokens):
            token = output.logits[:, -1, :].argmax(dim=-1)
            token_id = int(token.item())
            if token_id == tokenizer.eos_token_id:
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


def evaluate_vqa(model, processor, output_dir: Path, snapshot: Path) -> dict:
    rows = read_manifest("test", "_vqa", snapshot)
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


def evaluate_grounding(model, processor, output_dir: Path, snapshot: Path) -> dict:
    """Choose one side from each initial robot frame using the policy prompt."""
    rows = read_manifest("test", "_grounding", snapshot)
    predictions = []
    path = output_dir / "test_grounding_predictions.jsonl"
    path.unlink(missing_ok=True)
    for index, sample in enumerate(rows, 1):
        with Image.open(sample["image"]) as raw:
            answer = score_side(model, processor, raw.convert("RGB"),
                                sample["instruction"])
        episode_parts = sample["episode_id"].split("/")
        if len(episode_parts) != 4:
            raise ValueError(f"Unexpected grounding episode ID: {sample['episode_id']}")
        result = {"sample_id": sample["sample_id"],
                  "episode_id": sample["episode_id"],
                  "category": sample["category"], "pair_id": sample["pair_id"],
                  "swapped": bool(int(episode_parts[2])), "seed": int(episode_parts[3]),
                  "expected": sample["answer"], "predicted": answer,
                  "correct": answer == sample["answer"]}
        predictions.append(result)
        append_jsonl(path, result)
        if index % 50 == 0 or index == len(rows):
            print(f"GROUNDING {index}/{len(rows)} "
                  f"correct={sum(r['correct'] for r in predictions)}", flush=True)
    by_category = defaultdict(list)
    by_layout = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)
        by_layout["swapped" if row["swapped"] else "original"].append(row)
    return {"overall": accuracy(predictions, "correct"),
            "by_category": {k: accuracy(v, "correct") for k, v in sorted(by_category.items())},
            "by_layout": {k: accuracy(v, "correct") for k, v in sorted(by_layout.items())}}


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


def episode_groups(snapshot: Path, output_dir: Path) -> list[tuple[str, Path, list[dict]]]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripted_action_collection"))
    import build_assets
    from prepare import vqa_pairs

    build_assets.INPUT = CONFIG.dataset_root
    build_assets.ASSETS = output_dir / "test_assets"
    dataset_pairs = vqa_pairs()
    split_path = snapshot / "pair_splits.json"
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
    import gymnasium as gym
    from simpler_env.env.simpler_wrapper_v4 import _ensure_act2answer_registered

    _ensure_act2answer_registered("Act2AnswerV4-v1")
    return gym.make(
        "Act2AnswerV4-v1", num_envs=len(ids), obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend="gpu", sim_config={"sim_freq": 500, "control_freq": 5},
        max_episode_steps=episode_len + 10,
        sensor_configs={"shader_pack": "default"}, assets=str(asset_dir),
        ids=None, do_swap=swapped, instruction_condition="knowledge",
        tile_xy_left=CONFIG.tile_xy_left, tile_xy_right=CONFIG.tile_xy_right,
        rgb_overlay_paths={"3rd_view_camera": str(
            REPO_ROOT / "scripted_action_collection" / "input" / "bridge_real_eval_1.png")},
        cube_pose=(-0.131, 0.0411, 0.88 + 0.015 + 0.118),
        initial_qpos=np.array([-0.01840777, 0.0398835, 0.22242722,
                               -0.00460194, 1.36524296, 0.00153398,
                               0.0326, 0.0326]),
    )


def strict_tile_success(task, index: int, side: str) -> bool:
    from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene_multi_v4 import CUBE_HALF_SIZE

    cube = task.cubes[task._white_key]
    board_name = (task._current_left_names if side == "left"
                  else task._current_right_names)[index]
    board = task.objs_board[board_name]
    cube_p = cube.pose.p[index].detach().cpu().numpy()
    board_p = board.pose.p[index].detach().cpu().numpy()
    board_half = task.model_bbox_sizes[board_name][:2].detach().cpu().numpy() / 2
    board_top = board_p[2] + float(task.model_bbox_sizes[board_name][2].item()) / 2
    released = not bool(task.agent.is_grasping(cube)[index].item())
    return bool(released and np.all(np.abs(cube_p[:2] - board_p[:2]) <=
                                    board_half - CUBE_HALF_SIZE + 0.005)
                and abs(float(cube_p[2]) - (board_top + CUBE_HALF_SIZE)) <= 0.025)


def evaluate_rollout(model, processor, stats: dict, output_dir: Path,
                     batch_size: int, episode_len: int, seed: int,
                     snapshot: Path, modes: tuple[str, ...]) -> dict:
    rows = []
    path = output_dir / "test_rollout_episodes.jsonl"
    path.unlink(missing_ok=True)
    for category, asset_dir, pairs in episode_groups(snapshot, output_dir):
        for swapped in (False, True):
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                for mode in modes:
                    env = make_env(asset_dir, [int(p["index"]) for p in batch],
                                   swapped, seed, episode_len)
                    try:
                        task = env.unwrapped
                        obs, info = env.reset(
                            seed=seed, options={"episode_id": torch.tensor(
                                [int(p["index"]) for p in batch], device=task.device)})
                        close = torch.zeros((len(batch), 7), device=task.device)
                        close[:, -1] = -1
                        for _ in range(10):
                            obs, _, _, _, info = env.step(close)
                        instructions = task.get_language_instruction()
                        if [p["question"] for p in batch] != instructions:
                            raise ValueError(f"Asset instructions differ from training data: {category}")
                        frames = obs["sensor_data"]["3rd_view_camera"]["rgb"]
                        targets = []
                        decisions = []
                        for i, pair in enumerate(batch):
                            side = pair["answer"].title()
                            if swapped:
                                side = "Right" if side == "Left" else "Left"
                            targets.append(side)
                            image = Image.fromarray(frames[i].cpu().numpy())
                            decisions.append(score_side(model, processor, image, instructions[i]))
                        if mode == "oracle":
                            fixed_sides = targets
                        elif mode == "opposite":
                            fixed_sides = ["Right" if side == "Left" else "Left"
                                           for side in targets]
                        else:
                            fixed_sides = decisions
                        for step in range(episode_len):
                            actions = []
                            for i, instruction in enumerate(instructions):
                                image = Image.fromarray(frames[i].cpu().numpy())
                                tokens = generate_action_tokens(model, processor, image,
                                                                instruction, fixed_sides[i])
                                actions.append(decode_action(tokens, processor.tokenizer, stats))
                            action_tensor = torch.from_numpy(np.stack(actions)).to(task.device)
                            obs, _, _, _, info = env.step(action_tensor)
                            frames = obs["sensor_data"]["3rd_view_camera"]["rgb"]
                            if (step + 1) % 10 == 0 or step + 1 == episode_len:
                                print(f"ROLLOUT_STEP {mode} {category} swap={swapped} "
                                      f"batch={start // batch_size + 1} "
                                      f"step={step + 1}/{episode_len}", flush=True)
                        for i, pair in enumerate(batch):
                            result = {"mode": mode, "category": category,
                                      "pair_id": pair["pair_id"], "swapped": swapped,
                                      "seed": seed, "expected_side": targets[i],
                                      "predicted_side": decisions[i],
                                      "action_condition_side": fixed_sides[i],
                                      "grounding_correct": decisions[i] == targets[i],
                                      "success": bool(info["success"][i].item()),
                                      "strict_success": strict_tile_success(task, i, targets[i].lower()),
                                      "condition_tile_success": strict_tile_success(
                                          task, i, fixed_sides[i].lower()),
                                      "answered": bool(info["is_answered"][i].item()),
                                      "first_choice_correct": bool(
                                          info["tcp_first_choice_correct"][i].item())}
                            rows.append(result)
                            append_jsonl(path, result)
                        print(f"ROLLOUT {mode} {category} swap={swapped} "
                              f"episodes={min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
                    finally:
                        env.close()
    by_category = defaultdict(list)
    by_layout = defaultdict(list)
    by_mode = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_layout["swapped" if row["swapped"] else "original"].append(row)
        by_mode[row["mode"]].append(row)

    def summary(items: list[dict]) -> dict:
        return {"success": accuracy(items, "success"),
                "strict_success": accuracy(items, "strict_success"),
                "condition_tile_success": accuracy(items, "condition_tile_success"),
                "grounding_correct": accuracy(items, "grounding_correct"),
                "answered": accuracy(items, "answered"),
                "first_choice_correct": accuracy(items, "first_choice_correct")}

    return {"overall": summary(rows),
            "by_mode": {k: summary(v) for k, v in sorted(by_mode.items())},
            "by_category": {k: summary(v) for k, v in sorted(by_category.items())},
            "by_layout": {k: summary(v) for k, v in sorted(by_layout.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the semantic bridge adapter")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--rollout-batch-size", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("predicted", "oracle", "opposite", "both", "all"),
                        default="both")
    parser.add_argument("--grounding-only", action="store_true",
                        help="Evaluate first-frame robot grounding only")
    parser.add_argument("--qa-only", action="store_true",
                        help="Evaluate VQA and robot grounding without robot rollouts")
    args = parser.parse_args()
    if args.rollout_batch_size < 1 or args.episode_len < 1:
        parser.error("--rollout-batch-size and --episode-len must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Test requires a CUDA GPU")
    pointer = args.run_dir / "best.json"
    if not pointer.is_file():
        pointer = args.run_dir / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No checkpoint pointer in {args.run_dir}")
    checkpoint = json.loads(pointer.read_text(encoding="utf-8"))
    adapter = Path(checkpoint["adapter"])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    snapshot = args.run_dir / "input_snapshot"
    statistics_path = adapter / "dataset_statistics.json"
    if not snapshot.is_dir() or not statistics_path.is_file():
        raise FileNotFoundError("Run snapshot or checkpoint statistics missing")
    # Same device split as SimplerEnv: GPU 0 renders, GPU 1 holds OpenVLA.
    torch.cuda.set_device(1 if torch.cuda.device_count() > 1 else 0)
    model, processor = load_student(training=False, adapter_path=adapter,
                                    statistics_path=statistics_path)
    model.eval()
    stats = load_action_stats(statistics_path)
    # ManiSkill uses the default CUDA device for simulation and rendering.
    torch.cuda.set_device(0)
    output_dir = args.eval_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"split": "test", "checkpoint": checkpoint,
              "action_unnorm_key": CONFIG.action_unnorm_key}
    result["grounding"] = evaluate_grounding(model, processor, output_dir, snapshot)
    write_json(output_dir / "test_grounding_metrics.json", result)
    if args.grounding_only:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    (output_dir / "test_metrics.json").unlink(missing_ok=True)
    result["vqa"] = evaluate_vqa(model, processor, output_dir, snapshot)
    write_json(output_dir / "test_vqa_metrics.json", result)
    if args.qa_only:
        write_json(output_dir / "test_metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    modes = (("predicted", "oracle", "opposite") if args.mode == "all" else
             ("predicted", "oracle") if args.mode == "both" else (args.mode,))
    result["rollout"] = evaluate_rollout(model, processor, stats, output_dir,
                                          args.rollout_batch_size, args.episode_len,
                                          args.seed, snapshot, modes)
    write_json(output_dir / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Record the first strictly successful predicted-side test rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge import generate_action_tokens, score_side
from config import CONFIG
from evaluate import (decode_action, episode_groups, make_env, policy_image,
                      strict_tile_success, write_json)
from model_io import load_action_stats, load_student


def write_video(path: Path, frames: list[np.ndarray], fps: int = 5) -> None:
    if not frames:
        raise ValueError("No frames were captured")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {path}")
    try:
        for frame in [*frames, *([frames[-1]] * fps)]:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError("Video frames must be same-size uint8 RGB images")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--category", help="Restrict the search to one test category")
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()
    if args.episode_len < 1 or args.max_attempts < 1:
        parser.error("--episode-len and --max-attempts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Demo rollout requires CUDA")

    pointer = args.run_dir / "best.json"
    if not pointer.is_file():
        pointer = args.run_dir / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No checkpoint pointer in {args.run_dir}")
    checkpoint = json.loads(pointer.read_text(encoding="utf-8"))
    adapter = Path(checkpoint["adapter"])
    snapshot = args.run_dir / "input_snapshot"
    statistics_path = adapter / "dataset_statistics.json"
    if not adapter.is_dir() or not snapshot.is_dir() or not statistics_path.is_file():
        raise FileNotFoundError("Checkpoint adapter, snapshot, or statistics missing")
    available = {row["category"] for row in json.loads(
        (snapshot / "pair_splits.json").read_text(encoding="utf-8"))
        if row["split"] == "test"}
    if args.category is not None and args.category not in available:
        parser.error(f"Unknown test category {args.category!r}; available: {sorted(available)}")

    # Match evaluate.py: GPU 0 simulates; GPU 1 holds OpenVLA when available.
    torch.cuda.set_device(1 if torch.cuda.device_count() > 1 else 0)
    model, processor = load_student(training=False, adapter_path=adapter,
                                    statistics_path=statistics_path)
    model.eval()
    stats = load_action_stats(statistics_path)
    torch.cuda.set_device(0)

    output_dir = args.demo_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    attempts = 0
    for category, asset_dir, pairs in episode_groups(snapshot, output_dir, args.category):
        for swapped in (False, True):
            for pair in pairs:
                if attempts >= args.max_attempts:
                    raise RuntimeError(f"No strictly successful demo in {attempts} test rollouts")
                attempts += 1
                env = make_env(asset_dir, [int(pair["index"])], swapped,
                               args.seed, args.episode_len)
                try:
                    task = env.unwrapped
                    obs, info = env.reset(seed=args.seed, options={
                        "episode_id": torch.tensor([int(pair["index"])], device=task.device)})
                    close = torch.zeros((1, 7), device=task.device)
                    close[:, -1] = -1
                    for _ in range(10):
                        obs, _, _, _, info = env.step(close)
                    instruction = task.get_language_instruction()[0]
                    if instruction != pair["question"]:
                        raise ValueError(f"Instruction mismatch: {category}/{pair['pair_id']}")
                    target = pair["answer"].title()
                    if swapped:
                        target = "Right" if target == "Left" else "Left"
                    frame = policy_image(obs["sensor_data"]["3rd_view_camera"]["rgb"][0])
                    predicted = score_side(model, processor, frame, instruction)
                    images = [np.asarray(frame)]
                    for step in range(args.episode_len):
                        tokens = generate_action_tokens(model, processor, frame,
                                                        instruction, predicted)
                        action = decode_action(tokens, processor.tokenizer, stats)
                        obs, _, _, _, info = env.step(torch.from_numpy(action[None]).to(task.device))
                        frame = policy_image(obs["sensor_data"]["3rd_view_camera"]["rgb"][0])
                        images.append(np.asarray(frame))
                        if (step + 1) % 10 == 0:
                            print(f"DEMO_STEP attempt={attempts} step={step + 1}/{args.episode_len}",
                                  flush=True)
                    success = bool(info["success"][0].item())
                    strict = strict_tile_success(task, 0, target.lower())
                    print(f"DEMO_ATTEMPT {attempts} {category}/{pair['pair_id']} "
                          f"swap={swapped} predicted={predicted} target={target} "
                          f"success={success} strict_success={strict}", flush=True)
                    if success and strict:
                        video_path = output_dir / "semantic_bridge_success.mp4"
                        write_video(video_path, images)
                        write_json(output_dir / "demo.json", {
                            "split": "test", "mode": "predicted", "checkpoint": checkpoint,
                            "category": category, "pair_id": pair["pair_id"],
                            "swapped": swapped, "seed": args.seed, "attempts": attempts,
                            "instruction": instruction, "expected_side": target,
                            "predicted_side": predicted, "success": success,
                            "strict_success": strict, "fps": 5,
                            "video": video_path.name,
                        })
                        print(f"DEMO_SAVED {video_path}", flush=True)
                        return
                finally:
                    env.close()
    raise RuntimeError(f"No strictly successful demo in {attempts} test rollouts")


if __name__ == "__main__":
    main()

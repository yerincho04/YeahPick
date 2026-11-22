"""Collect opposite-side trajectories for Stage 1 training pairs only."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from config import CONFIG
from prepare import split_map, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripted_action_collection"))
import collect as original
import build_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=("all", *build_assets.CATEGORIES), default="all")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--layout", choices=("both", "original", "swapped"), default="both")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-steps", type=int, default=110)
    args = parser.parse_args()
    if args.start_pair < 0 or (args.count is not None and args.count < 1):
        parser.error("Invalid pair range")
    splits = split_map(CONFIG.stage1_run / "input_snapshot")
    from simpler_env.env.simpler_wrapper_v4 import _ensure_act2answer_registered
    import gymnasium as gym
    _ensure_act2answer_registered("Act2AnswerV4-v1")
    build_assets.INPUT = CONFIG.dataset_root
    build_assets.ASSETS = CONFIG.counterfactual_root / "assets"
    CONFIG.counterfactual_root.mkdir(parents=True, exist_ok=True)
    manifest = CONFIG.counterfactual_root / "episodes.jsonl"
    latest = {r["episode_id"]: r for r in read_jsonl(manifest)} if manifest.exists() else {}
    categories = build_assets.CATEGORIES if args.category == "all" else (args.category,)
    overlay = REPO_ROOT / "scripted_action_collection" / "input" / "bridge_real_eval_1.png"
    for category in categories:
        asset_dir, pairs = build_assets.build_category(category)
        selected = pairs[args.start_pair: args.start_pair + args.count if args.count else None]
        print(f"CATEGORY {category} selected={len(selected)}", flush=True)
        env = gym.make(
            "Act2AnswerV4-v1", num_envs=1, obs_mode="rgb+segmentation",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            sim_backend="gpu", sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=args.max_steps, sensor_configs={"shader_pack": "default"},
            assets=str(asset_dir.resolve()), ids=None, do_swap=False,
            instruction_condition="knowledge", tile_xy_left=CONFIG.tile_xy_left,
            tile_xy_right=CONFIG.tile_xy_right,
            rgb_overlay_paths={"3rd_view_camera": str(overlay)},
            cube_pose=(-0.131, 0.0411, 0.88 + 0.015 + 0.118),
            initial_qpos=np.array([-0.01840777, 0.0398835, 0.22242722,
                                   -0.00460194, 1.36524296, 0.00153398,
                                   0.0326, 0.0326]))
        try:
            for pair in selected:
                key = category, pair["pair_id"]
                if splits[key] != "train":
                    continue
                pair["category"] = category
                for swapped in ((False, True) if args.layout == "both" else (args.layout == "swapped",)):
                    episode_id = f"{category}/{pair['pair_id']}/{int(swapped)}/0"
                    if episode_id in latest and (latest[episode_id]["success"] or not args.retry_failed):
                        continue
                    semantic = pair["answer"]
                    if swapped:
                        semantic = "right" if semantic == "left" else "left"
                    opposite = "right" if semantic == "left" else "left"
                    try:
                        metadata, frames, actions, qposes = original.collect_episode(
                            env, pair, swapped, 0, args.max_steps, target_side=opposite)
                    except Exception as exc:
                        traceback.print_exc()
                        metadata = {"category": category, "pair_id": pair["pair_id"],
                                    "swapped": swapped, "seed": 0, "success": False,
                                    "failure_reason": f"{type(exc).__name__}: {exc}"}
                    metadata["episode_id"] = episode_id
                    if metadata["success"]:
                        folder = (CONFIG.counterfactual_root / "episodes" / category /
                                  pair["pair_id"] / ("swapped" if swapped else "original") / "seed_0")
                        original.save_success(metadata, frames, actions, qposes, folder)
                    with manifest.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                        out.flush()
                    print(f"OPPOSITE {episode_id} success={metadata['success']} "
                          f"reason={metadata.get('failure_reason')}", flush=True)
        finally:
            env.close()
    print("COUNTERFACTUAL_COLLECTION_DONE", flush=True)


if __name__ == "__main__":
    main()

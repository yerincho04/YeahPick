"""Collect successful Act2Answer demonstrations with a geometric controller."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from build_assets import ASSETS, CATEGORIES, ROOT, TILE_WIDTH, build_category


OUTPUT = ROOT / f"output_tile_{round(TILE_WIDTH * 100):02d}cm_v3"
TILE_XY_LEFT = (-0.22, -0.155)
TILE_XY_RIGHT = (-0.22, 0.13)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=("all", *CATEGORIES), default="all")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--layout", choices=("both", "original", "swapped"), default="both")
    parser.add_argument("--max-steps", type=int, default=110)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def scalar(value) -> bool:
    return bool(value.reshape(-1)[0].item())


def vector(value) -> np.ndarray:
    return value[0].detach().cpu().numpy().astype(np.float32)


def position(actor) -> np.ndarray:
    return vector(actor.pose.p)


def frame(obs):
    # Snapshot the rendered frame on the GPU. Failed episodes never need a
    # GPU-to-CPU image transfer; successful ones are copied when saved.
    return obs["sensor_data"]["3rd_view_camera"]["rgb"][0].clone()


def controller_action(env, desired_cube: np.ndarray, cube_tcp_offset: np.ndarray,
                      gripper: float) -> np.ndarray:
    """Move the EE target in the robot-base frame using observed TCP feedback."""
    import torch
    from mani_skill.utils.structs.pose import Pose

    task = env.unwrapped
    agent = task.agent
    base_inv = agent.robot.pose.inv()
    tcp_base = (base_inv * agent.tcp.pose).p
    arm = agent.controller.controllers["arm"]
    ee_base = arm.ee_pose_at_base.p
    target_ee_base = arm._target_pose.p
    desired_tcp_world = desired_cube - cube_tcp_offset
    desired_tcp_pose = Pose.create_from_pq(
        p=torch.as_tensor(desired_tcp_world, device=task.device, dtype=torch.float32).reshape(1, 3)
    )
    desired_tcp_base = (base_inv * desired_tcp_pose).p
    desired_ee_base = ee_base + desired_tcp_base - tcp_base
    delta = (desired_ee_base - target_ee_base)[0].detach().cpu().numpy()
    norm = np.linalg.norm(delta)
    if norm > 0.020:
        delta *= 0.020 / norm
    action = np.zeros(7, dtype=np.float32)
    action[:3] = delta
    action[6] = gripper
    return action


def collect_episode(env, pair: dict, swapped: bool, seed: int,
                    max_steps: int) -> tuple[dict, list, np.ndarray, np.ndarray]:
    import torch
    from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene_multi_v4 import CUBE_HALF_SIZE

    task = env.unwrapped
    task.do_swap = swapped
    obs, info = env.reset(
        seed=seed,
        options={"episode_id": torch.tensor([pair["index"]], device=task.device)},
    )
    close = torch.tensor([[0, 0, 0, 0, 0, 0, -1]], dtype=torch.float32, device=task.device)
    for _ in range(10):
        obs, _, _, _, info = env.step(close)

    cube = task.cubes[task._white_key]
    answer = task._answers[0]
    board_name = (task._current_left_names if answer == "left" else task._current_right_names)[0]
    board = task.objs_board[board_name]
    board_p = position(board)
    board_top = board_p[2] + float(task.model_bbox_sizes[board_name][2].item()) / 2
    start_cube = position(cube)
    cube_tcp_offset = start_cube - position(task.agent.tcp)
    # Aim 3 cm toward the robot base while keeping the cube on the tile.
    robot_xy = position(task.agent.robot)[:2]
    place_xy = board_p[:2] + np.clip(robot_xy - board_p[:2], -0.03, 0.03)
    # The pregrasp cube starts well above the boards. Extra lifting needlessly
    # pushes the farther, left-side target toward the arm's reach limit.
    lift_z = max(float(start_cube[2]), float(board_top) + 0.08)
    place_z = float(board_top) + 0.015 + 0.012
    instruction = task.get_language_instruction()[0]
    frames: list = []
    actions: list[np.ndarray] = []
    qposes: list = []
    phase = "lift"
    phase_steps = 0
    release_steps = 0
    failure_reason = None
    environment_terminated = False
    time_limit_reached = False
    limits = {"lift": 12, "transfer": 48, "lower": 22}

    for step in range(max_steps - 10):
        cube_p = position(cube)
        if phase == "lift":
            goal = np.array([start_cube[0], start_cube[1], lift_z], dtype=np.float32)
            arrived = abs(float(cube_p[2] - lift_z)) < 0.025
        elif phase == "transfer":
            goal = np.array([place_xy[0], place_xy[1], lift_z], dtype=np.float32)
            arrived = float(np.linalg.norm(cube_p[:2] - goal[:2])) < 0.025
        elif phase == "lower":
            goal = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)
            arrived = float(np.linalg.norm(cube_p - goal)) < 0.023
        else:
            goal = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)
            arrived = False

        if arrived and phase != "release":
            phase = {"lift": "transfer", "transfer": "lower", "lower": "release"}[phase]
            phase_steps = 0
            continue
        if phase in limits and phase_steps >= limits[phase]:
            failure_reason = f"{phase}_timeout"
            break

        gripper = 1.0 if phase == "release" else -1.0
        command = controller_action(env, goal, cube_tcp_offset, gripper)
        frames.append(frame(obs))
        actions.append(command.copy())
        qposes.append(task.agent.robot.get_qpos()[0].clone())
        obs, _, terminated, truncated, info = env.step(
            torch.as_tensor(command, device=task.device).reshape(1, 7)
        )
        environment_terminated = scalar(terminated)
        time_limit_reached = scalar(truncated)
        phase_steps += 1
        if phase == "release":
            release_steps += 1
            if release_steps >= 10:
                break
        # ManiSkill sets terminated=True whenever the cube is above the correct
        # tile, including while the gripper still holds it. Continue to release.
        if time_limit_reached:
            failure_reason = "time_limit_reached"
            break

    released = not scalar(task.agent.is_grasping(cube))
    final_cube_p = position(cube)
    board_half_xy = task.model_bbox_sizes[board_name][:2].cpu().numpy() / 2
    # Require the whole cube to be on the actual target tile. The simulator's
    # 8 cm success margin overlaps between the enlarged tiles.
    on_target_tile = bool(
        np.all(np.abs(final_cube_p[:2] - board_p[:2]) <= board_half_xy - CUBE_HALF_SIZE + 0.005)
        and abs(float(final_cube_p[2]) - (board_top + CUBE_HALF_SIZE)) <= 0.025
    )
    success = scalar(info["success"]) and released and release_steps >= 5 and on_target_tile
    if failure_reason is None and not success:
        failure_reason = "not_released_on_correct_tile"
    metadata = {
        "category": pair["category"], "pair_id": pair["pair_id"],
        "pair_index": pair["index"], "swapped": swapped, "seed": seed,
        "tile_xy_left": TILE_XY_LEFT, "tile_xy_right": TILE_XY_RIGHT,
        "answer_side": answer, "instruction": instruction,
        "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "pregrasp_steps": 10, "steps": len(actions),
        "action_format": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
        "success": bool(success), "failure_reason": None if success else failure_reason,
        "on_target_tile": on_target_tile,
        "final_cube_position": final_cube_p.tolist(),
        "target_board_position": board_p.tolist(),
        "place_xy_target": place_xy.tolist(),
        "final_tcp_position": position(task.agent.tcp).tolist(),
        "final_xy_error": float(np.linalg.norm(position(cube)[:2] - place_xy)),
        "final_cube_tcp_offset_error": float(np.linalg.norm(
            (position(cube) - position(task.agent.tcp)) - cube_tcp_offset
        )),
        "released": bool(released),
        "final_phase": phase,
        "environment_terminated": environment_terminated,
        "time_limit_reached": time_limit_reached,
    }
    saved_qposes = (torch.stack(qposes).cpu().numpy().astype(np.float32)
                    if success else np.empty((0, 0), dtype=np.float32))
    return metadata, frames, np.asarray(actions, dtype=np.float32), saved_qposes


def save_success(metadata: dict, frames: list, actions: np.ndarray,
                 qposes: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frames_dir = directory / "frames"
    frames_dir.mkdir(exist_ok=True)
    for step, image in enumerate(frames):
        Image.fromarray(image.cpu().numpy().astype(np.uint8)).save(
            frames_dir / f"{step:04d}.jpg", quality=90
        )
    np.save(directory / "actions.npy", actions)
    np.save(directory / "robot_qpos.npy", qposes)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def completed_keys(path: Path, retry_failed: bool) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["success"] or not retry_failed:
                    done.add(row["episode_id"])
    return done


def main() -> None:
    args = arguments()
    if args.start_pair < 0 or args.repeats < 1 or args.max_steps < 30:
        raise ValueError("start-pair >= 0, repeats >= 1 and max-steps >= 30 are required")
    categories = CATEGORIES if args.category == "all" else (args.category,)
    layouts = (False, True) if args.layout == "both" else (args.layout == "swapped",)

    # Build only the selected categories. This does not load a VLA model.
    built = {}
    for category in categories:
        asset_dir, pairs = build_category(category)
        for index, pair in enumerate(pairs):
            pair["index"] = index
            pair["category"] = category
        built[category] = (asset_dir, pairs)
        print(f"ASSETS_READY {category} pairs={len(pairs)}", flush=True)
    if args.build_only:
        return

    import gymnasium as gym
    import torch
    from simpler_env.env.simpler_wrapper_v4 import _ensure_act2answer_registered

    _ensure_act2answer_registered("Act2AnswerV4-v1")
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT / "episodes.jsonl"
    done = completed_keys(manifest, args.retry_failed)
    overlay = ROOT / "input" / "bridge_real_eval_1.png"
    if not overlay.is_file():
        raise FileNotFoundError(overlay)

    for category, (asset_dir, pairs) in built.items():
        selected = pairs[args.start_pair:args.start_pair + args.count if args.count is not None else None]
        planned = [
            (pair, swap, args.seed + repeat)
            for pair in selected for swap in layouts for repeat in range(args.repeats)
            if f"{category}/{pair['pair_id']}/{int(swap)}/{args.seed + repeat}" not in done
        ]
        if not planned:
            continue
        print(f"CATEGORY_START {category} episodes={len(planned)}", flush=True)
        env = gym.make(
            "Act2AnswerV4-v1", num_envs=1, obs_mode="rgb+segmentation",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            sim_backend="gpu", sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=args.max_steps, sensor_configs={"shader_pack": "default"},
            assets=str(asset_dir.resolve()), ids=None, do_swap=False,
            instruction_condition="knowledge",
            tile_xy_left=TILE_XY_LEFT, tile_xy_right=TILE_XY_RIGHT,
            rgb_overlay_paths={"3rd_view_camera": str(overlay)},
            cube_pose=(-0.131, 0.0411, 0.88 + 0.015 + 0.118),
            initial_qpos=np.array([
                -0.01840777, 0.0398835, 0.22242722, -0.00460194,
                1.36524296, 0.00153398, 0.0326, 0.0326,
            ]),
        )
        try:
            for pair, swap, seed in planned:
                episode_id = f"{category}/{pair['pair_id']}/{int(swap)}/{seed}"
                try:
                    metadata, frames, actions, qposes = collect_episode(
                        env, pair, swap, seed, args.max_steps
                    )
                except Exception as exc:
                    traceback.print_exc()
                    metadata = {
                        "category": category, "pair_id": pair["pair_id"],
                        "pair_index": pair["index"], "swapped": swap,
                        "seed": seed, "success": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                metadata["episode_id"] = episode_id
                if metadata["success"]:
                    save_success(
                        metadata, frames, actions, qposes,
                        OUTPUT / "episodes" / category / pair["pair_id"] /
                        ("swapped" if swap else "original") / f"seed_{seed}",
                    )
                with manifest.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                print(f"EPISODE {episode_id} success={metadata['success']} "
                      f"reason={metadata.get('failure_reason')}", flush=True)
        finally:
            env.close()
    print("COLLECTION_DONE", flush=True)


if __name__ == "__main__":
    main()

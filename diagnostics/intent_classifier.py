"""Act2Answer Phase 1, Task 4: infer which target the policy was trying to reach, using
only commanded/realized motion -- never task_success. Reads episode_log.jsonl (written by
diagnostics/episode_logger.py) and appends predicted_intent_side/intent_confidence to each
record.

ASSUMPTION / FINDING (explicit, discovered by inspecting real OpenVLA rollouts, not
guessed): the geometric signal used here is the *realized* end-effector displacement
(eef_position[-1] - eef_position[grasp_step]), not the raw commanded action vector summed
over time. Two things were tried and ruled out first:
  1. A fixed `init_grasp_steps + hold_cube_steps` cutoff for "answer-directed" steps --
     wrong, because the env's hold window only force-closes the *gripper* channel
     (SimplerEnv/simpler_env/env/simpler_wrapper_v4.py::SimplerWrapper.step,
     `action[:, 6] = -1.0`); translation is always free, so the arm often reaches the
     target board before the fixed cutoff elapses.
  2. Summing the raw commanded `world_vector` (actions[:, :3]) from first grasp onward --
     wrong for a different reason: on real rollouts this summed vector was almost exactly
     the *negation* (in x/y) of the target-ward direction, even on episodes that
     succeeded. The env's control mode is
     `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`; its delta-pose commands are
     evidently applied in a gripper/base-aligned frame, not the world frame that
     eef_position/cube_position/target positions are logged in (both are raw ManiSkill
     `pose.p`, always world frame). Re-deriving the controller's frame transform here would
     just be reimplementing a second potential source of bugs.
  Using the *realized* eef displacement sidesteps the frame mismatch entirely, since it's
  read from the same world-frame sim state as the target positions -- it's still a direct,
  measured consequence of the commanded actions (not a privileged label, not task_success),
  it's just measured in position-space rather than action-space. Verified: for every
  successful real episode in the test_colors smoke test, cos(realized_displacement,
  vector_to_correct_target) >= 0.86, vs. <=0.12 for the wrong target.

If the cube is never grasped (gripper_state never True), intent = "unavailable". If the two
candidate directions are too close to call, intent = "uncertain" (never forced into
left/right).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

UNCERTAIN_MARGIN = 0.05  # cosine-similarity margin below which we refuse to pick a side


def classify_intent(record: dict) -> tuple[str, float | None]:
    gripper_state = record["gripper_state"]
    grasp_idx = next((t for t, g in enumerate(gripper_state) if g), None)
    if grasp_idx is None or grasp_idx >= len(record["eef_position"]) - 1:
        return "unavailable", None

    eef = np.asarray(record["eef_position"])
    delta_p = eef[-1] - eef[grasp_idx]  # realized displacement from first grasp to episode end
    delta_norm = np.linalg.norm(delta_p)
    if delta_norm < 1e-8:
        return "unavailable", None

    left_target = np.asarray(record["left_target_position"])
    right_target = np.asarray(record["right_target_position"])

    vec_left = left_target - eef[grasp_idx]
    vec_right = right_target - eef[grasp_idx]

    def cos_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    sim_left = cos_sim(delta_p, vec_left)
    sim_right = cos_sim(delta_p, vec_right)
    margin = sim_left - sim_right

    if abs(margin) < UNCERTAIN_MARGIN:
        return "uncertain", float(abs(margin))

    side = "left" if margin > 0 else "right"
    return side, float(abs(margin))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-log", required=True, help="Path to episode_log.jsonl")
    parser.add_argument("--out", default=None, help="Output path (default: overwrite in place)")
    args = parser.parse_args()

    in_path = Path(args.episode_log)
    out_path = Path(args.out) if args.out else in_path

    records = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    for record in records:
        side, confidence = classify_intent(record)
        record["predicted_intent_side"] = side
        record["intent_confidence"] = confidence

    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    n = len(records)
    counts = {k: sum(r["predicted_intent_side"] == k for r in records) for k in ("left", "right", "uncertain", "unavailable")}
    print(f"Classified {n} episodes: {counts}")


if __name__ == "__main__":
    main()

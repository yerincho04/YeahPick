"""Act2Answer Phase 1 diagnostics: per-episode JSONL log + hidden-state .pt files.

Called from SimplerEnv.simpler_env.run.Runner.render() when --enable-diagnostics is set.
Reads only the per-step `info` dict that Act2AnswerV4.evaluate() already returns (see the
`diag_*` keys added there) plus the per-step action/hidden-state lists Runner.render()
already accumulates in `datas[i]`. Writes nothing when diagnostics are disabled.

episode_log.jsonl schema (one line per episode):
    episode_id, question_id, asset, swap_or_noswap, correct_answer_side, task_success,
    instruction, actions[t] (list[7]), eef_position[t] (list[3]), cube_position[t] (list[3]),
    left_target_position (list[3]), right_target_position (list[3]), gripper_state[t] (bool)

hidden_states/<episode_id>.pt: torch.save'd dict {layer_frac: FloatTensor[T, hidden_dim],
    "episode_id": str, "question_id": int} -- the hidden state is the model's own pre-action
    representation (see modeling_prismatic.get_diagnostic_hidden_states for exactly which
    token position), only 4 layers x 1 pooled vector per timestep (not full token sequences).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def write_episode_record(
    run_dir: Path,
    obj_set: str,
    asset: str,
    env_idx: int,
    instruction: str,
    data: dict,
    seed: int = 0,
    condition: str = "knowledge",
) -> None:
    infos = data["info"]
    actions = data["action"]
    if not infos:
        return  # nothing executed this episode

    last = infos[-1]
    pair_id = int(last["diag_pair_id"])
    swap = bool(last["diag_swap"])
    answer_side = "left" if bool(last["diag_answer_side_is_left"]) else "right"
    task_success = bool(last["success"])

    layout = "swap" if swap else "noswap"
    metadata = data.get("metadata", {})
    semantic_answer = str(metadata.get("semantic_answer", ""))
    initial_state = data.get("initial_state", {})
    initial_state_hash = hashlib.sha256(
        json.dumps(initial_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    episode_id = f"{obj_set}_{asset}_{pair_id}_{layout}_{condition}_seed{seed}"
    truncated_steps = data.get("truncated", [])
    first_truncated = next((i for i, value in enumerate(truncated_steps) if value), None)

    record = {
        "episode_id": episode_id,
        "question_id": pair_id,
        "asset": asset,
        "layout": layout,
        "swap_or_noswap": layout,  # backward-compatible alias
        "condition": condition,
        "semantic_answer": semantic_answer,
        "semantic_answer_id": metadata.get("semantic_answer_id", semantic_answer),
        "distractor_id": metadata.get("distractor_id", ""),
        "relation_type": metadata.get("relation_type", "unknown"),
        "template_family": metadata.get("template_family", "unknown"),
        "knowledge_category": metadata.get("knowledge_category", "unknown"),
        "seed": int(seed),
        "correct_answer_side": answer_side,
        "correct_side": answer_side,  # concise schema requested by the matched experiment
        "task_success": task_success,
        "instruction": instruction,
        "actions": actions,
        "eef_position": [info["diag_eef_pos"] for info in infos],
        "cube_position": [info["diag_cube_pos"] for info in infos],
        "left_target_position": infos[0]["diag_left_target_pos"],
        "right_target_position": infos[0]["diag_right_target_pos"],
        "gripper_state": [bool(info["diag_gripper_state"]) for info in infos],
        "initial_state": initial_state,
        "initial_state_hash": initial_state_hash,
        "final_cube_position": infos[-1]["diag_cube_pos"],
        "rollout_truncated": bool(truncated_steps[-1]) if truncated_steps else False,
        "first_truncated_policy_step": first_truncated,
        "final_policy_step": len(infos) - 1,
        "final_sim_elapsed_steps": int(last.get("elapsed_steps", -1)),
    }

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    log_path = diag_dir / "episode_log.jsonl"
    # Idempotent for retries: replace the same matched episode rather than silently
    # duplicating it in append-only logs.
    records = []
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        records = [item for item in records if item.get("episode_id") != episode_id]
    records.append(record)
    with open(log_path, "w") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")

    hidden_steps = data.get("hidden") or []
    if hidden_steps:
        hidden_dir = diag_dir / "hidden_states"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        fracs = sorted(hidden_steps[0].keys())
        payload = {frac: torch.stack([step[frac] for step in hidden_steps], dim=0) for frac in fracs}
        payload["episode_id"] = episode_id
        payload["question_id"] = pair_id
        torch.save(payload, hidden_dir / f"{episode_id}.pt")

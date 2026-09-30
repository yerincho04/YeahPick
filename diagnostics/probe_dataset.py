"""Act2Answer Phase 1, Task 5: build the hidden-state -> correct-answer-side probing
dataset from diagnostics/hidden_states/*.pt + episode_log.jsonl.

ASSUMPTION (explicit): each episode is represented by the hidden state captured at
timestep 0 -- i.e. the model's very first inference call for that episode, before any
action has moved the arm or the cube. This is the point closest to "has the backbone read
the question and picked an answer" that Task 3 captures; later timesteps mix in whatever
the arm has already done. Override with --timestep if you want a different point.

Split rule (required by the spec): normal and swapped versions of the same question share
a question_id (Act2AnswerV4's pair index) and must always land in the same split, so we
group-split on question_id rather than episode_id.

IMPORTANT: noswap and swap layouts are separate eval runs that write to separate
`<name>/diagnostics/` directories (see scripts/eval_openvla.sh -- it runs each layout as
its own `simpler_env.eval` invocation with a different `--name`). Pass BOTH directories to
--diagnostics-dir (accepts multiple) so a question's normal and swapped episodes are seen
together before splitting -- splitting on just one directory would silently defeat the
question_id grouping requirement, since the other half of each pair wouldn't be in the
dataset at all.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


def load_dataset(diag_dirs: list[Path], timestep: int = 0):
    records = {}
    for diag_dir in diag_dirs:
        for line in (diag_dir / "episode_log.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            records[r["episode_id"]] = r

    X = defaultdict(list)
    y, group_keys, episode_ids = [], [], []

    for diag_dir in diag_dirs:
        hidden_dir = diag_dir / "hidden_states"
        for pt_file in sorted(hidden_dir.glob("*.pt")):
            payload = torch.load(pt_file, map_location="cpu")
            episode_id = payload["episode_id"]
            record = records.get(episode_id)
            if record is None:
                continue

            fracs = sorted(k for k in payload.keys() if isinstance(k, float))
            t = min(timestep, payload[fracs[0]].shape[0] - 1)
            for frac in fracs:
                # Hidden states are saved in the model's inference dtype (bfloat16), which
                # numpy can't hold directly -- upconvert to float32 only here, at point of use.
                X[frac].append(payload[frac][t].float().numpy())

            y.append(1 if record["correct_answer_side"] == "left" else 0)
            # question_id is pairs.json's index, which resets per asset -- group on
            # (asset, question_id) so different categories (e.g. "state" q0 vs "test_colors"
            # q0) are never treated as the same question when the pilot spans categories.
            group_keys.append(f"{record['asset']}::{record['question_id']}")
            episode_ids.append(episode_id)

    X = {frac: np.stack(v, axis=0) for frac, v in X.items()}
    return X, np.asarray(y), np.asarray(group_keys), episode_ids


def group_split(group_keys: np.ndarray, seed: int = 0):
    """60/20/20 train/val/test, grouped by (asset, question_id) (see module docstring)."""
    idx = np.arange(len(group_keys))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, rest_idx = next(gss1.split(idx, groups=group_keys))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, groups=group_keys[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]

    return train_idx, val_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir", required=True, nargs="+",
        help="One or more run diagnostics/ directories (pass BOTH noswap and swap runs' "
             "directories together -- see module docstring).",
    )
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Where to save the combined split (npz)")
    args = parser.parse_args()

    diag_dirs = [Path(d) for d in args.diagnostics_dir]
    X, y, group_keys, episode_ids = load_dataset(diag_dirs, args.timestep)
    train_idx, val_idx, test_idx = group_split(group_keys, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {f"X_{frac}": arr for frac, arr in X.items()}
    save_dict.update(
        y=y,
        question_ids=group_keys,  # composite (asset, question_id) grouping key
        episode_ids=np.array(episode_ids, dtype=object),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    np.savez(out_path, **save_dict)

    print(f"Loaded {len(y)} episodes from {len(diag_dirs)} run(s) ({len(set(group_keys))} unique questions).")
    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

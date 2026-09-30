"""Act2Answer Phase 1, Task 6: join episode_log.jsonl (+ intent from Task 4) with the
test-fold semantic-probe predictions (Task 5) into one row per episode, classify each into
a failure stage, and print the headline P(intent wrong | semantic correct) metric.

ASSUMPTION (explicit): the failure decomposition can only be computed for episodes in the
probe's held-out TEST fold, because semantic_correct must be an out-of-sample prediction
(using train-fold predictions would leak the label the probe was fit on). With Phase 1's
small ~200-300 episode pilot, this means the final table covers ~20% of episodes -- stated
explicitly here rather than silently diluted by using in-sample predictions everywhere.

Classification rule (from the spec):
    semantic_correct = False                                -> semantic/reasoning failure
    semantic_correct = True, intent_correct = False          -> semantic-to-action grounding failure
    semantic_correct = True, intent_correct = True,
        execution_success = False                            -> motor/execution failure
    all True                                                  -> success
    intent in {unavailable, uncertain}                        -> ambiguous (separate bucket,
                                                                  never forced into the above)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["episode_id"]] = r
    return out


def classify(row: dict) -> str:
    if row["intent_available"] is False:
        return "ambiguous"
    if not row["semantic_correct"]:
        return "semantic_reasoning_failure"
    if not row["intent_correct"]:
        return "semantic_to_action_failure"
    if not row["execution_success"]:
        return "motor_execution_failure"
    return "success"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir", required=True, nargs="+",
        help="One or more run diagnostics/ directories (e.g. both noswap and swap runs' "
             "directories, matching whatever was passed to probe_dataset.py).",
    )
    parser.add_argument("--semantic-probe", required=True, help="Path to semantic_probe.jsonl from linear_probe.py")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    episodes = {}
    for d in args.diagnostics_dir:
        episodes.update(load_jsonl(Path(d) / "episode_log.jsonl"))  # includes predicted_intent_side (Task 4)
    semantic = load_jsonl(Path(args.semantic_probe))  # test-fold only (Task 5)

    rows = []
    for episode_id, sem in semantic.items():
        ep = episodes.get(episode_id)
        if ep is None:
            continue

        intent_side = ep.get("predicted_intent_side")
        intent_available = intent_side in ("left", "right")
        intent_correct = (intent_side == ep["correct_answer_side"]) if intent_available else None

        row = {
            "episode_id": episode_id,
            "question_id": ep["question_id"],
            "swap_or_noswap": ep["swap_or_noswap"],
            "correct_answer_side": ep["correct_answer_side"],
            "semantic_correct": sem["semantic_correct"],
            "predicted_intent_side": intent_side,
            "intent_available": intent_available,
            "intent_correct": intent_correct,
            "execution_success": ep["task_success"],
        }
        row["category"] = classify(row)
        rows.append(row)

    n = len(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    if n == 0:
        print("No episodes to summarize (no overlap between episode_log.jsonl and semantic_probe.jsonl).")
        return

    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    semantic_correct_rows = [r for r in rows if r["semantic_correct"] and r["intent_available"]]
    n_sem_correct = len(semantic_correct_rows)
    n_sem_correct_intent_wrong = sum(1 for r in semantic_correct_rows if not r["intent_correct"])
    p_intent_wrong_given_semantic_correct = (
        n_sem_correct_intent_wrong / n_sem_correct if n_sem_correct else float("nan")
    )

    print(f"Total evaluable episodes (test fold): {n}\n")
    for cat in ("semantic_reasoning_failure", "semantic_to_action_failure", "motor_execution_failure", "success", "ambiguous"):
        pct = 100.0 * counts.get(cat, 0) / n
        print(f"{cat}: {pct:.1f}% ({counts.get(cat, 0)}/{n})")

    print(f"\nP(intent wrong | semantic correct): {100.0 * p_intent_wrong_given_semantic_correct:.1f}% "
          f"({n_sem_correct_intent_wrong}/{n_sem_correct})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

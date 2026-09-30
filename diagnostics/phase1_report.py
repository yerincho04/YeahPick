"""Act2Answer Phase 1 pilot: final report across the custom semantic_pilot_v1 set (4
categories x 10 questions) and the test_colors control, combined.

Reuses, unmodified:
  - diagnostics.intent_classifier.classify_intent (Task 4, validated on real rollouts)
  - diagnostics.probe_dataset.load_dataset / group_split (Task 5 hidden-state dataset +
    question-grouped split, grouping on (asset, question_id))

New in this script (per this run's explicit requirements, not a change to the above):
  - layer selection by VALIDATION accuracy (not test accuracy) for the "primary" semantic
    representation used in the failure decomposition -- the other 3 layers are still
    reported in probe_layer_results.csv for reference.
  - per-knowledge_category breakdown (probe accuracy, P(intent wrong | semantic correct),
    noswap vs swap success) alongside the overall numbers.

Outputs (into --out-dir):
  phase1_all_episodes.csv, phase1_failure_summary.csv, phase1_category_summary.csv,
  probe_layer_results.csv, fig_layer_probe_accuracy.png, fig_failure_distribution.png,
  fig_failure_by_category.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from diagnostics.intent_classifier import classify_intent
from diagnostics.probe_dataset import group_split, load_dataset

LAYER_LABELS = {0.25: "Layer 25%", 0.5: "Layer 50%", 0.75: "Layer 75%", 1.0: "Final"}


def load_episode_logs(diag_dirs: list[Path]) -> dict[str, dict]:
    episodes = {}
    for d in diag_dirs:
        for line in (d / "episode_log.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            episodes[r["episode_id"]] = r
    return episodes


def load_category_map(manifest_csv: Path) -> dict[int, str]:
    cat = {}
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            cat[int(row["question_id"])] = row["knowledge_category"]
    return cat


def category_for(record: dict, semantic_categories: dict[int, str]) -> str:
    if record["asset"] == "test_colors":
        return "color_control"
    return semantic_categories.get(record["question_id"], "unknown")


def classify_failure(semantic_correct, intent_available, intent_correct, execution_success) -> str:
    if not intent_available:
        return "ambiguous"
    if not semantic_correct:
        return "semantic_reasoning_failure"
    if not intent_correct:
        return "semantic_to_action_failure"
    if not execution_success:
        return "motor_execution_failure"
    return "success"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", required=True, nargs="+", help="All run diagnostics/ dirs (test_colors + semantic_pilot_v1, noswap + swap)")
    parser.add_argument("--manifest", required=True, help="semantic_pilot_v1 manifest.csv (for knowledge_category)")
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    diag_dirs = [Path(d) for d in args.diagnostics_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_logs(diag_dirs)
    semantic_categories = load_category_map(Path(args.manifest))
    for r in episodes.values():
        r["knowledge_category"] = category_for(r, semantic_categories)
        r["predicted_intent_side"], r["intent_confidence"] = classify_intent(r)
        r["intent_available"] = r["predicted_intent_side"] in ("left", "right")
        r["intent_correct"] = (r["predicted_intent_side"] == r["correct_answer_side"]) if r["intent_available"] else None

    # ---- Task 5: hidden-state probe dataset + question-grouped split ----
    X, y, group_keys, episode_ids = load_dataset(diag_dirs, args.timestep)
    train_idx, val_idx, test_idx = group_split(group_keys, args.seed)
    fracs = sorted(X.keys())

    layer_rows = []
    per_layer_test_pred = {}
    for frac in fracs:
        Xf = X[frac]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xf[train_idx], y[train_idx])
        val_acc = clf.score(Xf[val_idx], y[val_idx]) if len(val_idx) else float("nan")
        test_acc = clf.score(Xf[test_idx], y[test_idx]) if len(test_idx) else float("nan")
        per_layer_test_pred[frac] = clf.predict(Xf[test_idx])
        layer_rows.append({"layer": LAYER_LABELS.get(frac, frac), "layer_frac": frac,
                            "val_acc": val_acc, "test_acc": test_acc})

    # Layer selection by VALIDATION accuracy only (never test) -- ties broken by shallower layer.
    best = max(layer_rows, key=lambda r: (r["val_acc"], -r["layer_frac"]))
    best_frac = best["layer_frac"]
    for row in layer_rows:
        row["selected_for_decomposition"] = (row["layer_frac"] == best_frac)

    with open(out_dir / "probe_layer_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "layer_frac", "val_acc", "test_acc", "selected_for_decomposition"])
        w.writeheader()
        w.writerows(layer_rows)

    # per-category test accuracy at the selected layer
    test_eids = [episode_ids[i] for i in test_idx]
    test_categories = np.array([episodes[eid]["knowledge_category"] for eid in test_eids])
    test_pred_best = per_layer_test_pred[best_frac]
    test_y = y[test_idx]
    cat_probe_acc = {}
    for cat in sorted(set(test_categories)):
        mask = test_categories == cat
        if mask.sum() > 0:
            cat_probe_acc[cat] = float((test_pred_best[mask] == test_y[mask]).mean())

    # semantic_correct per test episode (out-of-sample, at the val-selected layer)
    semantic_correct_by_eid = {eid: bool(pred == truth) for eid, pred, truth in zip(test_eids, test_pred_best, test_y)}

    # ---- Task 6: failure decomposition, joined onto the test fold ----
    all_rows = []
    for eid in test_eids:
        r = episodes[eid]
        sem_correct = semantic_correct_by_eid[eid]
        category = classify_failure(sem_correct, r["intent_available"], r["intent_correct"], r["task_success"])
        all_rows.append({
            "episode_id": eid, "asset": r["asset"], "knowledge_category": r["knowledge_category"],
            "question_id": r["question_id"], "swap_or_noswap": r["swap_or_noswap"],
            "correct_answer_side": r["correct_answer_side"], "semantic_correct": sem_correct,
            "predicted_intent_side": r["predicted_intent_side"], "intent_available": r["intent_available"],
            "intent_correct": r["intent_correct"], "execution_success": r["task_success"],
            "failure_category": category,
        })

    with open(out_dir / "phase1_all_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    n = len(all_rows)
    counts = Counter(r["failure_category"] for r in all_rows)

    def p_intent_wrong_given_semantic_correct(rows):
        sem_correct_rows = [r for r in rows if r["semantic_correct"] and r["intent_available"]]
        if not sem_correct_rows:
            return float("nan"), 0, 0
        wrong = sum(1 for r in sem_correct_rows if not r["intent_correct"])
        return wrong / len(sem_correct_rows), wrong, len(sem_correct_rows)

    p_overall, wrong_n, sem_n = p_intent_wrong_given_semantic_correct(all_rows)

    semantic_heavy_rows = [r for r in all_rows if r["knowledge_category"] != "color_control"]
    control_rows = [r for r in all_rows if r["knowledge_category"] == "color_control"]
    p_semantic_heavy, _, _ = p_intent_wrong_given_semantic_correct(semantic_heavy_rows)
    p_control, _, _ = p_intent_wrong_given_semantic_correct(control_rows)

    noswap_rows = [r for r in all_rows if r["swap_or_noswap"] == "noswap"]
    swap_rows = [r for r in all_rows if r["swap_or_noswap"] == "swap"]
    noswap_sr = np.mean([r["execution_success"] for r in noswap_rows]) if noswap_rows else float("nan")
    swap_sr = np.mean([r["execution_success"] for r in swap_rows]) if swap_rows else float("nan")

    with open(out_dir / "phase1_failure_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_evaluable_test_episodes", n])
        for cat in ("semantic_reasoning_failure", "semantic_to_action_failure", "motor_execution_failure", "success", "ambiguous"):
            w.writerow([f"pct_{cat}", 100.0 * counts.get(cat, 0) / n if n else float("nan")])
            w.writerow([f"n_{cat}", counts.get(cat, 0)])
        w.writerow(["P_intent_wrong_given_semantic_correct_overall", p_overall])
        w.writerow(["n_semantic_correct_overall", sem_n])
        w.writerow(["n_intent_wrong_given_semantic_correct_overall", wrong_n])
        w.writerow(["P_intent_wrong_given_semantic_correct_semantic_heavy", p_semantic_heavy])
        w.writerow(["P_intent_wrong_given_semantic_correct_color_control", p_control])
        w.writerow(["noswap_execution_success_rate", noswap_sr])
        w.writerow(["swap_execution_success_rate", swap_sr])

    # ---- per-category summary ----
    cat_rows = []
    for cat in sorted(set(r["knowledge_category"] for r in all_rows)):
        rows_c = [r for r in all_rows if r["knowledge_category"] == cat]
        p_c, wrong_c, sem_c = p_intent_wrong_given_semantic_correct(rows_c)
        noswap_c = [r for r in rows_c if r["swap_or_noswap"] == "noswap"]
        swap_c = [r for r in rows_c if r["swap_or_noswap"] == "swap"]
        cat_rows.append({
            "knowledge_category": cat,
            "n_test_episodes": len(rows_c),
            "probe_accuracy_at_selected_layer": cat_probe_acc.get(cat, float("nan")),
            "pct_semantic_reasoning_failure": 100.0 * sum(1 for r in rows_c if r["failure_category"] == "semantic_reasoning_failure") / len(rows_c),
            "pct_semantic_to_action_failure": 100.0 * sum(1 for r in rows_c if r["failure_category"] == "semantic_to_action_failure") / len(rows_c),
            "pct_motor_execution_failure": 100.0 * sum(1 for r in rows_c if r["failure_category"] == "motor_execution_failure") / len(rows_c),
            "pct_success": 100.0 * sum(1 for r in rows_c if r["failure_category"] == "success") / len(rows_c),
            "pct_ambiguous": 100.0 * sum(1 for r in rows_c if r["failure_category"] == "ambiguous") / len(rows_c),
            "P_intent_wrong_given_semantic_correct": p_c,
            "n_semantic_correct": sem_c,
            "noswap_success_rate": np.mean([r["execution_success"] for r in noswap_c]) if noswap_c else float("nan"),
            "swap_success_rate": np.mean([r["execution_success"] for r in swap_c]) if swap_c else float("nan"),
        })
    with open(out_dir / "phase1_category_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader()
        w.writerows(cat_rows)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [LAYER_LABELS.get(r["layer_frac"], r["layer_frac"]) for r in layer_rows]
    ax.plot(xs, [r["val_acc"] for r in layer_rows], marker="o", label="val acc")
    ax.plot(xs, [r["test_acc"] for r in layer_rows], marker="s", label="test acc")
    ax.axvline(LAYER_LABELS.get(best_frac, best_frac), color="gray", linestyle="--", alpha=0.6, label="selected (by val)")
    ax.set_ylabel("Probe accuracy"); ax.set_title("Layer-wise semantic probe accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "fig_layer_probe_accuracy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cats_order = ["success", "motor_execution_failure", "semantic_to_action_failure", "semantic_reasoning_failure", "ambiguous"]
    vals = [counts.get(c, 0) for c in cats_order]
    ax.bar(cats_order, vals, color=["#2a9d3f", "#e0a72a", "#d9534f", "#8b3a3a", "#999999"])
    ax.set_ylabel("Episodes (test fold)"); ax.set_title("Failure-type distribution (overall)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_dir / "fig_failure_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cat_names = [r["knowledge_category"] for r in cat_rows]
    width = 0.15
    xpos = np.arange(len(cat_names))
    for i, key in enumerate(["pct_success", "pct_motor_execution_failure", "pct_semantic_to_action_failure",
                              "pct_semantic_reasoning_failure", "pct_ambiguous"]):
        vals = [r[key] for r in cat_rows]
        ax.bar(xpos + i * width, vals, width, label=key.replace("pct_", ""))
    ax.set_xticks(xpos + 2 * width); ax.set_xticklabels(cat_names, rotation=30, ha="right")
    ax.set_ylabel("% of category's test episodes"); ax.set_title("Failure-type distribution by category")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_failure_by_category.png", dpi=150); plt.close(fig)

    # ---- console summary ----
    print(f"Total evaluable test episodes: {n}")
    print(f"Selected layer (by validation accuracy): {LAYER_LABELS.get(best_frac, best_frac)} (val={best['val_acc']:.3f}, test={best['test_acc']:.3f})")
    for cat in cats_order:
        print(f"  {cat}: {100.0 * counts.get(cat, 0) / n:.1f}% ({counts.get(cat, 0)}/{n})")
    print(f"P(intent wrong | semantic correct) overall: {100*p_overall:.1f}% ({wrong_n}/{sem_n})")
    print(f"P(intent wrong | semantic correct) semantic-heavy: {100*p_semantic_heavy:.1f}%")
    print(f"P(intent wrong | semantic correct) color control:  {100*p_control:.1f}%")
    print(f"noswap success rate: {noswap_sr:.3f}   swap success rate: {swap_sr:.3f}")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

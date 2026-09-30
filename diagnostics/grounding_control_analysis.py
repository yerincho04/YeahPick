"""Paired analysis for the four-condition image-tile grounding control."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.intent_classifier import classify_intent

CONDITIONS = ("spatial", "tile_object", "explicit_object", "visual_description")


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_delta(test_errors, reference_errors, samples: int, seed: int) -> list[float]:
    delta = np.asarray(test_errors, float) - np.asarray(reference_errors, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(samples, len(delta)))
    return [float(x) for x in np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", required=True)
    p.add_argument("--asset", default="grounding_control_v1")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--conclusion",
        choices=("language wording artifact", "semantic grounding issue", "general tile-interface issue", "inconclusive"),
        default="inconclusive",
    )
    p.add_argument("--conclusion-rationale", default="Pending review of the completed control.")
    args = p.parse_args()

    records = {}
    for path in sorted(Path(args.outputs_root).glob(f"openvla-{args.asset}-*/diagnostics/episode_log.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("asset") == args.asset and r.get("condition") in CONDITIONS:
                    records[r["episode_id"]] = r

    rows, grouped = [], defaultdict(dict)
    for r in records.values():
        predicted, confidence = classify_intent(r)
        row = {
            "question_id": int(r["question_id"]), "layout": r["layout"], "seed": int(r["seed"]),
            "condition": r["condition"], "instruction": r["instruction"],
            "semantic_answer": r["semantic_answer"], "correct_side": r["correct_side"],
            "predicted_intent_side": predicted, "intent_confidence": confidence,
            "intent_available": predicted in ("left", "right"),
            "intent_correct": predicted == r["correct_side"],
            "placement_success": bool(r["task_success"]),
            "initial_state_hash": r.get("initial_state_hash", ""), "episode_id": r["episode_id"],
        }
        rows.append(row)
        grouped[(row["question_id"], row["layout"], row["seed"])][row["condition"]] = (row, r)

    complete = {k: v for k, v in grouped.items() if set(v) == set(CONDITIONS)}
    incomplete = {"::".join(map(str, k)): sorted(set(CONDITIONS) - set(v)) for k, v in grouped.items() if set(v) != set(CONDITIONS)}
    init_mismatches = []
    for key, group in complete.items():
        hashes = {group[c][0]["initial_state_hash"] for c in CONDITIONS}
        states = [group[c][1].get("initial_state") for c in CONDITIONS]
        if len(hashes) != 1 or any(states[0] != x for x in states[1:]):
            init_mismatches.append("::".join(map(str, key)))

    condition_summary = []
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        condition_summary.append({
            "condition": condition, "n": len(subset),
            "intent_accuracy": float(np.mean([r["intent_correct"] for r in subset])),
            "intent_available_rate": float(np.mean([r["intent_available"] for r in subset])),
            "placement_success_rate": float(np.mean([r["placement_success"] for r in subset])),
        })
    layout_summary = []
    for layout in ("noswap", "swap"):
        for condition in CONDITIONS:
            subset = [r for r in rows if r["layout"] == layout and r["condition"] == condition]
            layout_summary.append({
                "layout": layout, "condition": condition, "n": len(subset),
                "intent_accuracy": float(np.mean([r["intent_correct"] for r in subset])),
                "placement_success_rate": float(np.mean([r["placement_success"] for r in subset])),
            })

    comparison_specs = (
        ("tile_grounding_cost", "tile_object", "spatial"),
        ("explicit_wording_penalty", "explicit_object", "tile_object"),
        ("semantic_vs_visual_cost", "tile_object", "visual_description"),
        ("visual_tile_cost", "visual_description", "spatial"),
    )
    comparisons = []
    for name, test, reference in comparison_specs:
        pairs = [(v[test][0], v[reference][0]) for v in complete.values()]
        test_err = [not a["intent_correct"] for a, _ in pairs]
        ref_err = [not b["intent_correct"] for _, b in pairs]
        b = sum(x and not y for x, y in zip(test_err, ref_err))
        c = sum(not x and y for x, y in zip(test_err, ref_err))
        comparisons.append({
            "comparison": name, "test_condition": test, "reference_condition": reference,
            "n_pairs": len(pairs), "test_error_rate": float(np.mean(test_err)),
            "reference_error_rate": float(np.mean(ref_err)),
            "paired_error_excess": float(np.mean(np.asarray(test_err, float) - np.asarray(ref_err, float))),
            "bootstrap_95ci": bootstrap_delta(test_err, ref_err, args.bootstrap_samples, args.seed),
            "test_wrong_reference_correct": b, "test_correct_reference_wrong": c,
            "mcnemar_exact_p": exact_mcnemar(b, c),
        })

    triplets = []
    for key, group in sorted(complete.items()):
        wide = {"question_id": key[0], "layout": key[1], "seed": key[2], "semantic_answer": group["spatial"][0]["semantic_answer"]}
        for condition in CONDITIONS:
            wide[f"{condition}_intent_correct"] = group[condition][0]["intent_correct"]
            wide[f"{condition}_placement_success"] = group[condition][0]["placement_success"]
            wide[f"{condition}_instruction"] = group[condition][0]["instruction"]
        triplets.append(wide)

    summary = {
        "asset": args.asset, "rollout_count": len(rows), "complete_matched_sets": len(complete),
        "incomplete_sets": incomplete, "initialization_mismatch_count": len(init_mismatches),
        "initialization_mismatches": init_mismatches, "condition_results": condition_summary,
        "layout_results": layout_summary, "paired_comparisons": comparisons,
        "conclusion": args.conclusion, "conclusion_rationale": args.conclusion_rationale,
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "grounding_rollouts.csv", sorted(rows, key=lambda r: (r["question_id"], r["layout"], r["condition"])))
    write_csv(out / "grounding_matched_sets.csv", triplets)
    write_csv(out / "condition_summary.csv", condition_summary)
    write_csv(out / "layout_summary.csv", layout_summary)
    (out / "grounding_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    pct = lambda x: f"{100*x:.1f}%"
    report = [
        "# Image-tile grounding control", "", f"- Rollouts: {len(rows)}",
        f"- Complete matched sets: {len(complete)}", f"- Initialization mismatches: {len(init_mismatches)}", "",
        "## Results", "", "| Condition | N | Intent accuracy | Placement success |",
        "|---|---:|---:|---:|",
    ]
    for r in condition_summary:
        report.append(f"| {r['condition']} | {r['n']} | {pct(r['intent_accuracy'])} | {pct(r['placement_success_rate'])} |")
    report += ["", "## Layout breakdown", "", "| Layout | Condition | Intent | Placement |", "|---|---|---:|---:|"]
    for r in layout_summary:
        report.append(f"| {r['layout']} | {r['condition']} | {pct(r['intent_accuracy'])} | {pct(r['placement_success_rate'])} |")
    report += ["", "## Paired comparisons", "", "Positive error excess means the test condition is worse.", "",
               "| Comparison | Test vs reference | Error excess | Bootstrap 95% CI | Discordant | McNemar p |",
               "|---|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lo, hi = r["bootstrap_95ci"]
        report.append(f"| {r['comparison']} | {r['test_condition']} vs {r['reference_condition']} | {r['paired_error_excess']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['test_wrong_reference_correct']} / {r['test_correct_reference_wrong']} | {r['mcnemar_exact_p']:.4g} |")
    report += ["", "## Interpretation", "", f"**{args.conclusion}** — {args.conclusion_rationale}"]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

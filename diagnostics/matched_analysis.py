"""Paired go/no-go analysis for the OpenVLA matched-instruction experiment.

The existing hidden-state probe predicts the *correct left/right target side*. It does
not classify semantic answer identity. Any optional conditional analysis in this module
therefore uses the deliberately narrower term "grounded target-side decodability".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from diagnostics.intent_classifier import classify_intent


CONDITIONS = ("knowledge", "explicit_object", "explicit_spatial")


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_paired_difference(a_error, b_error, seed=0, samples=10_000):
    delta = np.asarray(a_error, dtype=float) - np.asarray(b_error, dtype=float)
    if len(delta) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        n = min(1000, samples - start)
        idx = rng.integers(0, len(delta), size=(n, len(delta)))
        means[start:start + n] = delta[idx].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def load_records(root: Path, asset: str, output_dir_globs: list[str]) -> list[dict]:
    by_id = {}
    paths = {
        path
        for pattern in output_dir_globs
        for path in root.glob(f"{pattern}/diagnostics/episode_log.jsonl")
    }
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("asset") != asset or record.get("condition") not in CONDITIONS:
                continue
            record["source_log"] = str(path)
            by_id[record["episode_id"]] = record
    return list(by_id.values())


def load_decodability(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["episode_id"]] = bool(row["semantic_correct"])
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument(
        "--output-dir-glob",
        action="append",
        dest="output_dir_globs",
        help="Output-directory glob to include; repeat for multiple shards (default: *).",
    )
    parser.add_argument("--asset", default="semantic_pilot_v1")
    parser.add_argument("--side-probe", default=None, help="Optional semantic_probe.jsonl; target is left/right side")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conclusion",
        choices=("GO", "grounding-dominant", "motor-dominant", "inconclusive"),
        default="inconclusive",
        help="Human interpretation after inspecting effect sizes; no percentage threshold is hard-coded.",
    )
    parser.add_argument("--conclusion-rationale", default="Pending qualitative review of the completed pilot.")
    args = parser.parse_args()

    records = load_records(Path(args.outputs_root), args.asset, args.output_dir_globs or ["*"])
    side_decodable = load_decodability(Path(args.side_probe) if args.side_probe else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    grouped = defaultdict(dict)
    for r in records:
        predicted_side, confidence = classify_intent(r)
        intent_available = predicted_side in ("left", "right")
        intent_correct = predicted_side == r["correct_side"]
        key = (str(r["question_id"]), r["layout"], int(r["seed"]))
        row = {
            "question_id": r["question_id"],
            "layout": r["layout"],
            "seed": r["seed"],
            "condition": r["condition"],
            "knowledge_category": r.get("knowledge_category", "unknown"),
            "semantic_answer": r.get("semantic_answer", ""),
            "correct_side": r["correct_side"],
            "instruction": r["instruction"],
            "predicted_intent_side": predicted_side,
            "intent_confidence": confidence,
            "intent_available": intent_available,
            "intent_correct": intent_correct,
            "placement_success": bool(r["task_success"]),
            "rollout_truncated": bool(r.get("rollout_truncated", False)),
            "final_cube_position": json.dumps(r.get("final_cube_position")),
            "initial_state_hash": r.get("initial_state_hash", ""),
            "target_side_decodable": side_decodable.get(r["episode_id"]),
            "episode_id": r["episode_id"],
        }
        long_rows.append(row)
        grouped[key][r["condition"]] = (row, r)

    complete = {key: value for key, value in grouped.items() if set(value) == set(CONDITIONS)}
    incomplete = {
        "::".join(map(str, key)): sorted(set(CONDITIONS) - set(value))
        for key, value in grouped.items() if set(value) != set(CONDITIONS)
    }

    # Verify the post-initialization, pre-policy state across all three conditions.
    init_mismatches = []
    for key, triplet in complete.items():
        hashes = {triplet[c][0]["initial_state_hash"] for c in CONDITIONS}
        states = [triplet[c][1].get("initial_state", {}) for c in CONDITIONS]
        numerically_equal = all(states[0] == state for state in states[1:])
        if len(hashes) != 1 or not numerically_equal:
            init_mismatches.append("::".join(map(str, key)))

    condition_rows = []
    for condition in CONDITIONS:
        subset = [row for row in long_rows if row["condition"] == condition]
        condition_rows.append({
            "condition": condition,
            "n": len(subset),
            "intent_accuracy": float(np.mean([r["intent_correct"] for r in subset])) if subset else float("nan"),
            "intent_available_rate": float(np.mean([r["intent_available"] for r in subset])) if subset else float("nan"),
            "placement_success_rate": float(np.mean([r["placement_success"] for r in subset])) if subset else float("nan"),
        })

    layout_rows = []
    for layout in ("noswap", "swap"):
        for condition in CONDITIONS:
            subset = [r for r in long_rows if r["layout"] == layout and r["condition"] == condition]
            layout_rows.append({
                "layout": layout, "condition": condition, "n": len(subset),
                "intent_accuracy": float(np.mean([r["intent_correct"] for r in subset])) if subset else float("nan"),
                "placement_success_rate": float(np.mean([r["placement_success"] for r in subset])) if subset else float("nan"),
            })

    comparisons = []
    for name, a, b in (
        ("knowledge_use_gap", "knowledge", "explicit_object"),
        ("grounding_gap", "explicit_object", "explicit_spatial"),
    ):
        pairs = [(triplet[a][0], triplet[b][0]) for triplet in complete.values()]
        a_err = [not x[0]["intent_correct"] for x in pairs]
        b_err = [not x[1]["intent_correct"] for x in pairs]
        b_count = sum(x and not y for x, y in zip(a_err, b_err))
        c_count = sum(not x and y for x, y in zip(a_err, b_err))
        comparisons.append({
            "comparison": name,
            "condition_a": a,
            "condition_b": b,
            "n_pairs": len(pairs),
            "error_rate_a": float(np.mean(a_err)) if pairs else float("nan"),
            "error_rate_b": float(np.mean(b_err)) if pairs else float("nan"),
            "paired_error_difference": float(np.mean(np.asarray(a_err, float) - np.asarray(b_err, float))) if pairs else float("nan"),
            "bootstrap_95ci": bootstrap_paired_difference(a_err, b_err, args.seed, args.bootstrap_samples),
            "a_wrong_b_correct": b_count,
            "a_correct_b_wrong": c_count,
            "mcnemar_exact_p": exact_mcnemar(b_count, c_count),
        })

    pattern_counts = Counter()
    wide_rows = []
    for key, triplet in sorted(complete.items()):
        correctness = tuple(triplet[c][0]["intent_correct"] for c in CONDITIONS)
        pattern_counts[str(correctness)] += 1
        base = triplet["knowledge"][0]
        wide = {
            "question_id": key[0], "layout": key[1], "seed": key[2],
            "knowledge_category": base["knowledge_category"],
            "semantic_answer": base["semantic_answer"],
        }
        for condition in CONDITIONS:
            row = triplet[condition][0]
            wide[f"{condition}_intent_correct"] = row["intent_correct"]
            wide[f"{condition}_placement_success"] = row["placement_success"]
            wide[f"{condition}_instruction"] = row["instruction"]
        wide_rows.append(wide)

    direct_paired_counts = {
        "knowledge_wrong_explicit_object_correct": sum(
            (not r["knowledge_intent_correct"]) and r["explicit_object_intent_correct"] for r in wide_rows
        ),
        "knowledge_correct_explicit_object_wrong": sum(
            r["knowledge_intent_correct"] and (not r["explicit_object_intent_correct"]) for r in wide_rows
        ),
        "explicit_object_wrong_explicit_spatial_correct": sum(
            (not r["explicit_object_intent_correct"]) and r["explicit_spatial_intent_correct"] for r in wide_rows
        ),
        "all_three_correct": sum(all(r[f"{c}_intent_correct"] for c in CONDITIONS) for r in wide_rows),
        "all_three_wrong": sum(not any(r[f"{c}_intent_correct"] for c in CONDITIONS) for r in wide_rows),
    }

    category_rows = []
    categories = sorted({triplet["knowledge"][0]["knowledge_category"] for triplet in complete.values()})
    for category in categories:
        triplets = [
            triplet for triplet in complete.values()
            if triplet["knowledge"][0]["knowledge_category"] == category
        ]
        category_rows.append({
            "knowledge_category": category,
            "n_triplets": len(triplets),
            **{
                f"{condition}_intent_accuracy": float(np.mean([
                    triplet[condition][0]["intent_correct"] for triplet in triplets
                ]))
                for condition in CONDITIONS
            },
            **{
                f"{condition}_placement_success_rate": float(np.mean([
                    triplet[condition][0]["placement_success"] for triplet in triplets
                ]))
                for condition in CONDITIONS
            },
        })

    spatial = [triplet["explicit_spatial"][0] for triplet in complete.values()]
    motor_floor = float(np.mean([not row["placement_success"] for row in spatial])) if spatial else float("nan")

    knowledge_proxy = []
    for triplet in complete.values():
        k = triplet["knowledge"][0]
        if k["target_side_decodable"] is True:
            knowledge_proxy.append((k, triplet["explicit_object"][0]))
    conditional = {
        "probe_target": "correct left/right target side (not semantic answer identity)",
        "probe_test_fold_episodes": len(side_decodable),
        "probe_target_side_accuracy": (
            float(np.mean(list(side_decodable.values()))) if side_decodable else None
        ),
        "n_target_side_decodable": len(knowledge_proxy),
        "p_knowledge_intent_wrong_given_target_side_decodable": (
            float(np.mean([not k["intent_correct"] for k, _ in knowledge_proxy]))
            if knowledge_proxy else None
        ),
        "n_decodable_knowledge_wrong_object_correct": sum(
            (not k["intent_correct"]) and obj["intent_correct"] for k, obj in knowledge_proxy
        ),
    }

    summary = {
        "asset": args.asset,
        "rollout_count": len(long_rows),
        "complete_matched_triplets": len(complete),
        "incomplete_triplets": incomplete,
        "initialization_mismatch_count": len(init_mismatches),
        "initialization_mismatches": init_mismatches,
        "condition_results": condition_rows,
        "layout_results": layout_rows,
        "paired_comparisons": comparisons,
        "motor_control_floor": motor_floor,
        "paired_intent_patterns": dict(pattern_counts),
        "direct_paired_counts": direct_paired_counts,
        "category_results": category_rows,
        "conditional_decodability": conditional,
        "conclusion": args.conclusion,
        "conclusion_rationale": args.conclusion_rationale,
    }

    write_csv(out_dir / "matched_rollouts.csv", sorted(long_rows, key=lambda r: (int(r["question_id"]), r["layout"], r["condition"])))
    write_csv(out_dir / "matched_triplets.csv", wide_rows)
    write_csv(out_dir / "condition_summary.csv", condition_rows)
    write_csv(out_dir / "layout_summary.csv", layout_rows)
    write_csv(out_dir / "category_summary.csv", category_rows)
    (out_dir / "matched_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def pct(value):
        return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.1f}%"

    report = [
        "# OpenVLA matched-instruction go/no-go report", "",
        f"- Rollouts: {len(long_rows)}", f"- Complete matched triplets: {len(complete)}",
        f"- Initialization mismatches: {len(init_mismatches)}", "",
        "## Accuracy by condition", "",
        "| Condition | N | Intent accuracy | Placement success | Intent available |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in condition_rows:
        report.append(f"| {row['condition']} | {row['n']} | {pct(row['intent_accuracy'])} | {pct(row['placement_success_rate'])} | {pct(row['intent_available_rate'])} |")
    report += ["", "## Noswap/swap breakdown", "",
               "| Layout | Condition | N | Intent accuracy | Placement success |",
               "|---|---|---:|---:|---:|"]
    for row in layout_rows:
        report.append(
            f"| {row['layout']} | {row['condition']} | {row['n']} | "
            f"{pct(row['intent_accuracy'])} | {pct(row['placement_success_rate'])} |"
        )
    report += ["", "## Paired comparisons", "",
               "| Comparison | Error-rate difference | Bootstrap 95% CI | Discordant counts | Exact McNemar p |",
               "|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lo, hi = row["bootstrap_95ci"]
        report.append(
            f"| {row['comparison']} | {row['paired_error_difference']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {row['a_wrong_b_correct']} / {row['a_correct_b_wrong']} | "
            f"{row['mcnemar_exact_p']:.4g} |"
        )
    report += ["", "Direct paired counts:"]
    for key, value in direct_paired_counts.items():
        report.append(f"- {key}: {value}")
    report += ["", "## Category breakdown", "",
               "| Category | N | Knowledge intent | Object intent | Spatial intent |",
               "|---|---:|---:|---:|---:|"]
    for row in category_rows:
        report.append(
            f"| {row['knowledge_category']} | {row['n_triplets']} | "
            f"{pct(row['knowledge_intent_accuracy'])} | {pct(row['explicit_object_intent_accuracy'])} | "
            f"{pct(row['explicit_spatial_intent_accuracy'])} |"
        )
    report += [
        "", "## Controls and probe interpretation", "",
        f"- Explicit-spatial placement failure rate (motor/control floor): {pct(motor_floor)}",
        "- The current probe predicts the correct **left/right grounded target side**. It does not predict semantic answer identity.",
        "- Consequently, the conditional result below is target-side decodability, not proof of knowledge or reasoning.",
        f"- Held-out probe episodes: {conditional['probe_test_fold_episodes']}",
        f"- Held-out final-layer target-side accuracy: {pct(conditional['probe_target_side_accuracy'])}",
        f"- Target-side-decodable knowledge episodes: {conditional['n_target_side_decodable']}",
        f"- P(knowledge intent wrong | target side decodable): {pct(conditional['p_knowledge_intent_wrong_given_target_side_decodable'])}",
        f"- Decodable + knowledge wrong + explicit-object correct: {conditional['n_decodable_knowledge_wrong_object_correct']}",
        "", "## Example knowledge-fail / explicit-object-success triplets", "",
    ]
    examples = [r for r in wide_rows if not r["knowledge_intent_correct"] and r["explicit_object_intent_correct"]]
    if examples:
        for row in examples[:10]:
            report.append(
                f"- Q{row['question_id']} ({row['layout']}, {row['knowledge_category']}, answer={row['semantic_answer']}): "
                f"`{row['knowledge_instruction']}` → `{row['explicit_object_instruction']}`"
            )
    else:
        report.append("- None.")
    report += ["", "## Conclusion", "", f"**{args.conclusion}** — {args.conclusion_rationale}"]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

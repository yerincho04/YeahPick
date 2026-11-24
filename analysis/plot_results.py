"""Build three comparable CSV tables and render grouped Matplotlib bar charts.

Run from any working directory with Matplotlib installed:
    python analysis/plot_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
OUTPUT = HERE / "output"

MODELS = (
    ("openvla_sft", "OpenVLA SFT", "#81A1C1"),
    ("semantic_kd_training", "Semantic KD", "#BF616A"),
    ("semantic_bridge_training", "Semantic Bridge (Ours)", "#A3BE8C"),
)
CATEGORIES = (
    "attribute", "celebrity", "counting", "emotion", "living_world",
    "public_info", "state", "symmetry", "time_clock", "traffic",
)
QUESTION_TYPES = (
    "Object recognition", "Attribute recognition", "Semantic reasoning",
    "Visual grounding", "Counterfactual grounding",
)
VQA_FILES = {
    "openvla_sft": "semantic_kd_training/output/test_vqa_predictions (1).jsonl",
    "semantic_kd_training": "semantic_kd_training/output/test_vqa_predictions.jsonl",
    "semantic_bridge_training": "semantic_bridge_training/eval_result/test_vqa_predictions.jsonl",
}
SIM_FILES = {
    "openvla_sft": "semantic_kd_training/output/test_rollout_episodes (1).jsonl",
    "semantic_kd_training": "semantic_kd_training/output/test_rollout_episodes.jsonl",
    "semantic_bridge_training": "semantic_bridge_training/eval_result/test_rollout_episodes.jsonl",
}
TABLES = (
    ("vqa_by_category", "category", CATEGORIES, "VQA accuracy by category",
     "Exact-match accuracy; OpenVLA SFT generated no valid exact answers."),
    ("vqa_by_question_type", "question_type", QUESTION_TYPES, "VQA accuracy by question type",
     "Exact-match accuracy across 10 categories."),
    ("simulation_success_by_category", "category", CATEGORIES,
     "Simulation success by category",
     "100 test pairs × 2 layouts; Semantic Bridge uses predicted direction."),
)


def read_jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_sources(files: dict[str, str], kind: str) -> dict[str, list[dict]]:
    result = {}
    reference = None
    for model, relative in files.items():
        rows = read_jsonl(relative)
        if kind == "simulation":
            rows = [row for row in rows if row.get("mode", "predicted") == "predicted"]
            key = lambda row: (row["category"], row["pair_id"], bool(row["swapped"]),
                               int(row["seed"]))
            expected_count = 200
        else:
            key = lambda row: row["sample_id"]
            expected_count = 1000
        indexed = {key(row): row for row in rows}
        if len(rows) != expected_count or len(indexed) != expected_count:
            raise ValueError(f"{relative}: expected {expected_count} unique {kind} rows")
        if reference is None:
            reference = indexed
        elif indexed.keys() != reference.keys():
            raise ValueError(f"{relative}: test examples differ from the OpenVLA SFT baseline")
        for sample_id, row in indexed.items():
            baseline = reference[sample_id]
            shared = ("category", "question_type", "expected") if kind == "vqa" else ("category",)
            if any(row.get(field) != baseline.get(field) for field in shared):
                raise ValueError(f"{relative}: label/category mismatch for {sample_id}")
            if kind == "simulation" and row["expected_side"].lower() != baseline["expected_side"].lower():
                raise ValueError(f"{relative}: target-side mismatch for {sample_id}")
        result[model] = list(indexed.values())
    return result


def build_csv(name: str, group_field: str, groups: tuple[str, ...],
              sources: dict[str, list[dict]], source_files: dict[str, str]) -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    path = INPUT / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "group", "model", "correct", "total", "accuracy_percent", "source"))
        writer.writeheader()
        for group in (*groups, "Overall"):
            for model, _, _ in MODELS:
                rows = (sources[model] if group == "Overall" else
                        [row for row in sources[model] if row[group_field] == group])
                if not rows:
                    raise ValueError(f"No rows for {model}/{group}")
                correct = sum(bool(row["correct"] if name.startswith("vqa") else row["success"])
                              for row in rows)
                writer.writerow({
                    "group": group, "model": model, "correct": correct,
                    "total": len(rows), "accuracy_percent": f"{100 * correct / len(rows):.4f}",
                    "source": source_files[model]})
    print(f"CSV {path.relative_to(ROOT)}")


def make_csvs() -> None:
    vqa = read_sources(VQA_FILES, "vqa")
    simulation = read_sources(SIM_FILES, "simulation")
    for name, field, groups, _, _ in TABLES:
        sources = vqa if name.startswith("vqa") else simulation
        files = VQA_FILES if name.startswith("vqa") else SIM_FILES
        expected = set(groups)
        for model, rows in sources.items():
            actual = {row[field] for row in rows}
            if actual != expected:
                raise ValueError(f"{model}: unexpected {field} values: {actual ^ expected}")
        build_csv(name, field, groups, sources, files)


def load_csv(name: str, groups: tuple[str, ...]) -> dict[tuple[str, str], float]:
    path = INPUT / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(group, model) for group in (*groups, "Overall")
                for model, _, _ in MODELS}
    indexed = {(row["group"], row["model"]): row for row in rows}
    if len(rows) != len(indexed) or indexed.keys() != expected:
        raise ValueError(f"{path}: missing or duplicate model/group rows")
    values = {}
    for key, row in indexed.items():
        correct, total = int(row["correct"]), int(row["total"])
        percent = float(row["accuracy_percent"])
        if total <= 0 or not 0 <= correct <= total or abs(percent - 100 * correct / total) > .001:
            raise ValueError(f"{path}: invalid count or accuracy for {key}")
        values[key] = percent
    return values


def plot_chart(name: str, groups: tuple[str, ...], title: str, note: str) -> None:
    values = load_csv(name, groups)
    labels = (*groups, "Overall")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.linewidth": 1.2,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(max(11, 1.25 * len(labels)), 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bar_width = 0.24
    for model_index, (model, display, color) in enumerate(MODELS):
        xs = [index + (model_index - 1) * bar_width for index in range(len(labels))]
        heights = [values[group, model] for group in labels]
        bars = ax.bar(xs, heights, width=bar_width, color=color,
                      edgecolor="black", linewidth=1.0, label=display,
                      zorder=2)
        for bar, height in zip(bars, heights):
            label = f"{height:g}"
            ax.text(bar.get_x() + bar.get_width() / 2, max(height, 0) + 1.3,
                    label, ha="center", va="bottom", fontsize=9)
    ax.axvline(len(groups) - 0.5, color="#777777", linestyle="--",
               linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([group.replace("_", " ") for group in labels],
                       rotation=35, ha="right")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Success rate (%)" if name.startswith("simulation")
                  else "Accuracy (%)")
    ax.set_xlabel("Simulation category" if name.startswith("simulation")
                  else "Question type" if name.endswith("question_type")
                  else "VQA category")
    ax.tick_params(axis="both", direction="out", length=5, width=1.1,
                   colors="black", labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.legend(ncol=3, loc="upper right", frameon=True, fancybox=False,
              framealpha=1, edgecolor="black", facecolor="white", fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.34, top=0.95)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = OUTPUT / f"{name}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight",
                    metadata={"Description": note})
        print(f"GRAPH {path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Use existing analysis/input CSV files without rereading JSONL")
    args = parser.parse_args()
    if not args.plots_only:
        make_csvs()
    for name, _, groups, title, note in TABLES:
        plot_chart(name, groups, title, note)


if __name__ == "__main__":
    main()

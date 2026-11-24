"""Plot predicted-side following and oracle success for Semantic Bridge."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "semantic_bridge_training/eval_result/test_rollout_episodes.jsonl"
INPUT = HERE / "input"
OUTPUT = HERE / "output"
CATEGORIES = (
    "attribute", "celebrity", "counting", "emotion", "living_world",
    "public_info", "state", "symmetry", "time_clock", "traffic",
)
MODES = {
    "predicted": ("bridge_predicted_by_category", "Semantic Bridge (Ours): followed predicted side",
                  "#BF616A", "condition_tile_success"),
    "oracle": ("bridge_oracle_by_category", "Semantic Bridge (Ours): oracle success",
               "#A3BE8C", "success"),
}
FIELDS = ("category", "mode", "metric", "count", "total", "percent", "source")


def episode_key(row: dict) -> tuple:
    return row["category"], row["pair_id"], bool(row["swapped"]), int(row["seed"])


def make_csvs() -> None:
    with SOURCE.open(encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]
    if len(episodes) != 400:
        raise ValueError(f"Expected 400 rollout episodes; found {len(episodes)}")
    by_mode = {mode: [row for row in episodes if row["mode"] == mode]
               for mode in MODES}
    if sum(map(len, by_mode.values())) != len(episodes):
        raise ValueError("Unexpected rollout mode")
    keys = []
    for mode, rows in by_mode.items():
        indexed = {episode_key(row): row for row in rows}
        if len(rows) != 200 or len(indexed) != 200:
            raise ValueError(f"{mode}: expected 200 unique episodes")
        if {row["category"] for row in rows} != set(CATEGORIES):
            raise ValueError(f"{mode}: category mismatch")
        if any(sum(row["category"] == category for row in rows) != 20
               for category in CATEGORIES):
            raise ValueError(f"{mode}: expected 20 episodes per category")
        keys.append(set(indexed))
    if keys[0] != keys[1]:
        raise ValueError("Predicted and oracle episodes do not match")

    INPUT.mkdir(parents=True, exist_ok=True)
    for mode, (stem, _, _, metric) in MODES.items():
        path = INPUT / f"{stem}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for category in (*CATEGORIES, "Overall"):
                rows = (by_mode[mode] if category == "Overall" else
                        [row for row in by_mode[mode] if row["category"] == category])
                count = sum(row[metric] is True for row in rows)
                writer.writerow({
                    "category": category, "mode": mode, "metric": metric,
                    "count": count, "total": len(rows),
                    "percent": f"{100 * count / len(rows):.4f}",
                    "source": SOURCE.relative_to(ROOT).as_posix(),
                })
        print(f"CSV {path.relative_to(ROOT)}")


def load_csv(stem: str, mode: str, metric: str) -> list[float]:
    path = INPUT / f"{stem}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row["category"] for row in rows] != [*CATEGORIES, "Overall"]:
        raise ValueError(f"{path}: category order or count mismatch")
    values = []
    for row in rows:
        count, total = int(row["count"]), int(row["total"])
        percent = float(row["percent"])
        if (row["mode"] != mode or row["metric"] != metric or total <= 0
                or not 0 <= count <= total
                or abs(percent - 100 * count / total) > .001):
            raise ValueError(f"{path}: invalid row {row}")
        values.append(percent)
    return values


def plot(mode: str) -> None:
    stem, legend_label, color, metric = MODES[mode]
    values = load_csv(stem, mode, metric)
    labels = (*CATEGORIES, "Overall")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 13, "axes.labelsize": 16,
        "axes.linewidth": 1.2, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(max(11, 1.25 * len(labels)), 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(range(len(labels)), values, width=0.62, color=color,
                  edgecolor="black", linewidth=1.0, label=legend_label)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:g}",
                ha="center", va="bottom", fontsize=10)
    ax.axvline(len(CATEGORIES) - 0.5, color="#777777", linestyle="--",
               linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([label.replace("_", " ") for label in labels],
                       rotation=35, ha="right")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Followed predicted side (%)" if mode == "predicted"
                  else "Success rate (%)")
    ax.set_xlabel("Simulation category")
    ax.tick_params(axis="both", direction="out", length=5, width=1.1,
                   colors="black", labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              framealpha=1, edgecolor="black", facecolor="white", fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.34, top=0.95)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight",
                    metadata={"Description": f"Semantic Bridge {mode} {metric}; 20 episodes per category."})
        print(f"GRAPH {path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Plot from existing analysis/input CSV files")
    args = parser.parse_args()
    if not args.plots_only:
        make_csvs()
    for mode in MODES:
        plot(mode)


if __name__ == "__main__":
    main()

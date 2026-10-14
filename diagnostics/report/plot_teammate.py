"""Figures for the teammate's shared results (teammate_results.json, transcribed from the shared PDFs;
per-question success counts out of 3 runs).

  teammate_swap.png       success per task, original vs swapped layout, one panel per category
  teammate_side.png       success when the correct answer is on the left vs the right tile
  teammate_questions.png  per-question successes (public_info, time_clock: 10 questions each)

Run from the repo root:  python diagnostics/report/plot_teammate.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "teammate_results.json").read_text())
CATS = ["public_info", "time_clock", "celebrity", "living_world", "traffic"]
TASKS = ["Semantic reasoning", "Explicit object", "Explicit spatial"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ListedColormap(["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])  # 0..3 successes, ordinal blue
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "legend.frameon": False})


def rate(cat, task, layout=None, side=None):
    """Success rate for a category/task, optionally one layout or only answers on one side."""
    d, k, n = T[cat], 0, 0
    for q, left_ns in enumerate(d["answer_left_noswap"]):
        for lay in ("noswap", "swap"):
            if layout and lay != layout:
                continue
            ans_left = left_ns if lay == "noswap" else not left_ns
            if side and (side == "left") != ans_left:
                continue
            k += d[task][lay][q]
            n += d["runs"]
    return k / n


def style(ax, title):
    ax.axhline(0.5, color=INK2, linewidth=1, zorder=4)
    ax.set_xticks(range(len(TASKS)), [t.replace(" ", "\n") for t in TASKS], color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1], ["0%", "chance\n50%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10, fontweight="semibold", color=INK)


def paired_panels(fname, series, suptitle):
    fig, axes = plt.subplots(1, len(CATS), figsize=(17, 3.9), sharey=True)
    for ax, cat in zip(axes, CATS):
        for si, (label, color, fn) in enumerate(series):
            vals = [fn(cat, t) for t in TASKS]
            xs = [i + (si - 0.5) * 0.38 for i in range(len(TASKS))]
            ax.bar(xs, vals, width=0.36, color=color, label=label, edgecolor=SURFACE, linewidth=1, zorder=3)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        style(ax, f"{cat}\n{T[cat]['questions']} questions x 3 runs")
    axes[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(2.9, -0.2))
    fig.suptitle(suptitle, x=0.01, y=1.06, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


paired_panels("teammate_swap.png",
              [("original layout (no swap)", BLUE, lambda c, t: rate(c, t, layout="noswap")),
               ("swapped layout", ORANGE, lambda c, t: rate(c, t, layout="swap"))],
              "Teammate results: success rate (%) by layout")
paired_panels("teammate_side.png",
              [("correct answer on the LEFT tile", BLUE, lambda c, t: rate(c, t, side="left")),
               ("correct answer on the RIGHT tile", ORANGE, lambda c, t: rate(c, t, side="right"))],
              "Teammate results: success rate (%) by which side the correct answer is on (large gap = choosing by position)")

# per-question grid for the two 10-question categories
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cols = [(t, l) for t in TASKS for l in ("noswap", "swap")]
for ax, cat in zip(axes, ("public_info", "time_clock")):
    grid = [[T[cat][t][l][q] for (t, l) in cols] for q in range(10)]
    ax.imshow(grid, cmap=RAMP, vmin=-0.5, vmax=3.5, aspect="auto")
    for q in range(10):
        for c in range(len(cols)):
            v = grid[q][c]
            ax.text(c, q, str(v), ha="center", va="center", fontsize=8, color="white" if v >= 2 else INK)
    ax.set_xticks(range(len(cols)), [f"{t.split()[0]} {t.split()[1][:3]}.\n{'no swap' if l == 'noswap' else 'swap'}" for t, l in cols],
                  fontsize=7.5, color=INK2)
    ax.set_yticks(range(10), [f"{q}: {T[cat]['meta'][q][3]}" for q in range(10)], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.grid(which="minor", axis="x", color=SURFACE, linewidth=2)
    ax.set_title(f"{cat}\nsuccesses out of 3 runs · row = question: correct answer", loc="left", fontsize=10,
                 fontweight="semibold", color=INK)
fig.tight_layout()
fig.savefig(HERE / "figures" / "teammate_questions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote teammate_swap.png, teammate_side.png, teammate_questions.png")

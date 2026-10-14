"""Plot the hypothesis-test results as PNG figures.

Figures (written to --out-dir, default outputs/figures/):
  behaviour_success.png        placement success per test x instruction (+ teammate results), chance line
  behaviour_measures.png       arm-to-correct-tile, correct-in-both-layouts, chose-left, per test x instruction
  probes_robustness.png        K / G probe accuracy by layer, robustness-check folds (image-pair)
  probes_preregistered.png     same, pre-registered folds (formulation)

Reads the episode logs in outputs/ and the analysis summaries written by hypothesis_pilot.py.
Run from the repo root:  python -m diagnostics.plot_results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.intent_classifier import classify_intent

# test asset -> (display name, analysis summary dir or None)
ASSETS = [
    ("photo_state_v1", "State (photo)", "outputs/photo_hypothesis/photo_state_v1"),
    ("photo_attribute_v1", "Attribute (photo)", "outputs/photo_hypothesis/photo_attribute_v1"),
    ("photo_emotion_v1", "Emotion (photo)", "outputs/photo_hypothesis/photo_emotion_v1"),
    ("semantic_repeated_v2", "Drawings pilot", "outputs/hypothesis_pilot_tiefix"),
    ("emotion_pilot_v1", "Emotion (drawn)", None),
]
# Teammate's shared results (컴종설 실험1), placement success; see diagnostics/report/teammate_results.json.
TEAMMATE = {  # (successes, episodes); 3 runs per episode
    "public_info\n(teammate)": {"Semantic reasoning": (26, 60), "Explicit object": (31, 60), "Explicit spatial": (53, 60)},
    "time_clock\n(teammate)": {"Semantic reasoning": (26, 60), "Explicit object": (26, 60), "Explicit spatial": (53, 60)},
    "celebrity\n(teammate)": {"Semantic reasoning": (124, 300), "Explicit object": (140, 300), "Explicit spatial": (277, 300)},
    "living_world\n(teammate)": {"Semantic reasoning": (139, 300), "Explicit object": (152, 300), "Explicit spatial": (279, 300)},
    "traffic\n(teammate)": {"Semantic reasoning": (127, 300), "Explicit object": (147, 300), "Explicit spatial": (282, 300)},
}
CONDITION_NAMES = {"knowledge": "Semantic reasoning", "paper_style": "Paper-style object", "tile_object": "Explicit object", "explicit_object": "Explicit object",
                   "explicit_spatial": "Explicit spatial", "neutral": "Neutral"}
CONDS = ["Semantic reasoning", "Paper-style object", "Explicit object", "Explicit spatial", "Neutral"]
PLOT = {"neutral": True}
# Categorical slots 1-4 of the validated reference palette (fixed order, light mode).
COLORS = ["#2a78d6", "#eda100", "#eb6834", "#1baf7a", "#898781"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def behaviour(outputs: Path) -> dict:
    """asset -> condition -> {measure: (k, n)} from the episode logs."""
    out = {}
    for asset, name, _ in ASSETS:
        recs = {}
        for log in outputs.glob(f"openvla-{asset}-*/diagnostics/episode_log.jsonl"):
            for line in log.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[(r["condition"], int(r["question_id"]), r["layout"])] = r
        if not recs:
            continue
        conds = defaultdict(lambda: {m: [0, 0] for m in ("success", "intent", "both", "left")})
        by_q = defaultdict(dict)
        for (cond, qid, layout), r in recs.items():
            c = CONDITION_NAMES[cond]
            side, _ = classify_intent(r)
            for m, hit in (("success", bool(r["task_success"])), ("intent", side == r["correct_side"]), ("left", side == "left")):
                conds[c][m][0] += int(hit)
                conds[c][m][1] += 1
            by_q[(c, qid)][layout] = side == r["correct_side"]
        for (c, _), d in by_q.items():
            if len(d) == 2:
                conds[c]["both"][0] += int(d["noswap"] and d["swap"])
                conds[c]["both"][1] += 1
        out[name] = dict(conds)
    return out


def grouped_bars(ax, groups: dict, measure: str, ref: float, ref_label: str, title: str) -> None:
    names = list(groups)
    width = 0.76 / len(CONDS)
    for ci, cond in enumerate(CONDS):
        xs, ys = [], []
        for gi, g in enumerate(names):
            kn = groups[g].get(cond, {}).get(measure) if isinstance(groups[g].get(cond), dict) else groups[g].get(cond)
            if kn and kn[1]:
                xs.append(gi + (ci - (len(CONDS) - 1) / 2) * (width + 0.01))
                ys.append(kn[0] / kn[1])
        ax.bar(xs, ys, width=width, color=COLORS[ci], label=cond, edgecolor=SURFACE, linewidth=1, zorder=3)
        for x, yv in zip(xs, ys):
            if yv == 0:  # a true zero has no visible bar; label it so it does not read as missing data
                ax.text(x, 0.01, "0%", ha="center", va="bottom", fontsize=7, color=INK2, zorder=5)
    ax.axhline(ref, color=INK2, linewidth=1, zorder=4)
    ax.text(len(names) - 0.5, ref + 0.015, ref_label, ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(names)), names, fontsize=9, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left")


def plot_probes(path: Path, fold: str, title: str) -> None:
    rows = []
    for asset, name, summary_dir in ASSETS:
        f = Path(summary_dir) / fold / "summary.json" if summary_dir else None
        if f and f.exists():
            rows.append((name, json.loads(f.read_text())))
    fig, axes = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 3.4), sharey=True)
    series = [("k2_knowledge", "K, knowledge question", COLORS[0]), ("g_knowledge", "G, knowledge question", COLORS[1])]
    if PLOT["neutral"]:
        series.append(("k2_neutral", "K, neutral control", COLORS[2]))
    for ax, (name, s) in zip(axes, rows):
        layers = s["probe_layers"]
        xs = range(len(layers))
        for key, label, color in series:
            ax.plot(xs, [r[key] for r in layers], color=color, linewidth=2, solid_capstyle="round", label=label, zorder=3)
            ax.scatter(xs, [r[key] for r in layers], s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.axhline(0.5, color=INK2, linewidth=1, zorder=2)
        ax.set_xticks(list(xs), [r["layer"] for r in layers])
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1], ["0%", "chance\n50%", "100%"])
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{name}\n{layers[0]['n']} episodes", loc="left", fontsize=10)
        ax.set_xlabel("LLM layer depth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, x=0.01, y=1.1, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="only plot these test assets (e.g. photo_state_v1 photo_attribute_v1 photo_emotion_v1)")
    ap.add_argument("--no-neutral", action="store_true", help="leave out the neutral control everywhere")
    ap.add_argument("--plain-labels", action="store_true", help="drop the '(photo)' / '(teammate)' suffixes")
    args = ap.parse_args()
    if args.assets:
        ASSETS[:] = [a for a in ASSETS if a[0] in args.assets]
    if args.no_neutral:
        CONDS.remove("Neutral")
    if args.plain_labels:
        ASSETS[:] = [(a, n.split(" (")[0], d) for a, n, d in ASSETS]
        for k in list(TEAMMATE):
            TEAMMATE[k.split("\n")[0]] = TEAMMATE.pop(k)
    PLOT["neutral"] = not args.no_neutral
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    beh = behaviour(Path(args.outputs_root))

    # 1. placement success, with the teammate's categories for comparison
    groups = {name.replace(" (", "\n("): conds for name, conds in beh.items()}
    groups.update({k: {c: {"success": v} for c, v in d.items()} for k, d in TEAMMATE.items()})
    fig, ax = plt.subplots(figsize=(15, 4.4))
    grouped_bars(ax, groups, "success", 0.5, "chance 50%",
                 "Placement success by instruction (cube ended on the correct tile)")
    ax.legend(loc="upper center", ncol=len(CONDS), bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "behaviour_success.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. the other behavioural measures (this repo's tests only)
    groups = {name.replace(" (", "\n("): conds for name, conds in beh.items()}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5))
    for ax, (m, ref, ref_label, title) in zip(axes, [
        ("intent", 0.5, "chance 50%", "Arm went to the correct tile (from the trajectory)"),
        ("both", 0.25, "chance 25%", "Correct in both layouts (a robot that always picks one side scores 0%)"),
        ("left", 0.5, "no bias 50%", "Chose the left tile (far from 50% = choosing by position)"),
    ]):
        grouped_bars(ax, groups, m, ref, ref_label, title)
    axes[-1].legend(loc="upper center", ncol=len(CONDS), bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out / "behaviour_measures.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3-4. probes by layer
    plot_probes(out / "probes_robustness.png", "image_pair",
                "Probe accuracy by layer (robustness-check folds)")
    plot_probes(out / "probes_preregistered.png", "formulation", "Probe accuracy by layer (pre-registered folds)")
    for p in sorted(out.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()

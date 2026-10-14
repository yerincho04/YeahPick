"""Compare several VLAs on the same test sets (behaviour only; works for every model).

For each model prefix (output dirs <prefix>-<asset>-<condition>-<layout>-...) and each asset, reports per
instruction: placement success, arm went to the correct tile, correct in both layouts, and success when
the answer is on the left vs the right tile (position bias).

Writes <out-dir>/models_summary.csv, models_summary.md and models_success.png.
Run from the repo root:
  python -m diagnostics.compare_models --models openvla openvla-base magma spatialvla pi0
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.intent_classifier import classify_intent

ASSETS = {"photo_state_v1": "State", "photo_attribute_v1": "Attribute", "photo_emotion_v1": "Emotion"}
CONDS = {"knowledge": "Semantic reasoning", "paper_style": "Paper-style object", "tile_object": "Explicit object",
         "explicit_spatial": "Explicit spatial"}
NAMES = {"openvla": "OpenVLA (RL4VLA SFT)", "openvla-base": "OpenVLA (base)", "magma": "Magma-8B",
         "spatialvla": "SpatialVLA", "pi0": "π0 (INTACT)"}
COLORS = ["#2a78d6", "#eda100", "#eb6834", "#1baf7a"]
INK2, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(outputs: Path, model: str, asset: str) -> dict:
    recs = {}
    for log in outputs.glob(f"{model}-{asset}-*/diagnostics/episode_log.jsonl"):
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["condition"] in CONDS:
                    recs[(r["condition"], int(r["question_id"]), r["layout"])] = r
    return recs


def summarise(recs: dict) -> dict:
    m = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_q = defaultdict(dict)
    for (cond, qid, layout), r in recs.items():
        side, _ = classify_intent(r)
        ok = bool(r["task_success"])
        for key, hit in (("success", ok), ("intent", side == r["correct_side"]),
                         (f"success_ans_{r['correct_side']}", ok)):
            m[cond][key][0] += int(hit)
            m[cond][key][1] += 1
        by_q[(cond, qid)][layout] = side == r["correct_side"]
    for (cond, _), d in by_q.items():
        if len(d) == 2:
            m[cond]["both"][0] += int(d["noswap"] and d["swap"])
            m[cond]["both"][1] += 1
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(NAMES))
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--out-dir", default="outputs/model_comparison")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, data = [], {}
    for model in args.models:
        for asset, aname in ASSETS.items():
            recs = load(Path(args.outputs_root), model, asset)
            if not recs:
                continue
            s = summarise(recs)
            data[(model, aname)] = s
            for cond, cname in CONDS.items():
                if cond not in s:
                    continue
                d = s[cond]
                pct = lambda k: round(100 * d[k][0] / d[k][1], 1) if d[k][1] else ""
                rows.append({"model": NAMES.get(model, model), "set": aname, "instruction": cname,
                             "episodes": d["success"][1], "success_%": pct("success"), "intent_%": pct("intent"),
                             "both_layouts_%": pct("both"), "success_answer_left_%": pct("success_ans_left"),
                             "success_answer_right_%": pct("success_ans_right")})
    if not rows:
        raise SystemExit("no episode logs found for these models")
    with open(out / "models_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    (out / "models_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # one panel per set; groups = models, bars = instructions
    sets = [a for a in ASSETS.values() if any(k[1] == a for k in data)]
    models = [m for m in args.models if any(k[0] == m for k in data)]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.2), sharey=True, squeeze=False)
    for ax, aname in zip(axes[0], sets):
        for ci, cond in enumerate(CONDS):
            xs, ys = [], []
            for mi, model in enumerate(models):
                d = data.get((model, aname), {}).get(cond)
                if d and d["success"][1]:
                    xs.append(mi + (ci - (len(CONDS) - 1) / 2) * 0.2)
                    ys.append(d["success"][0] / d["success"][1])
            ax.bar(xs, ys, width=0.19, color=COLORS[ci], label=CONDS[cond], edgecolor=SURFACE, zorder=3)
        ax.axhline(0.5, color=INK2, linewidth=1, zorder=4)
        ax.set_xticks(range(len(models)), [NAMES.get(m, m).replace(" (", "\n(") for m in models], fontsize=8, color=INK2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1], ["0%", "chance\n50%", "100%"])
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(aname, loc="left", fontsize=11, fontweight="semibold")
    fig.suptitle("Placement success by model and instruction", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDS), frameon=False)
    fig.savefig(out / "models_success.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(out / "models_success.png")


if __name__ == "__main__":
    main()

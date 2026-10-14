"""Figures and tables for the multi-model report (REPORT_models_ko.md).

  models_success.png      placement success per model x instruction, one panel per photo set
  models_landed.png       correct tile among episodes where the cube ended on a tile (95% Wilson CI)
  models_text_probe.png   text-only probe accuracy by LLM depth, one panel per model
  models_tables.json      every number quoted in the report

Reads outputs/<model>-<asset>-<condition>-*/diagnostics/episode_log.jsonl and outputs/text_only*/results.json.
Run from the repo root:  python diagnostics/report/plot_models.py
"""
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODELS = {"openvla": "OpenVLA (SFT)", "spatialvla": "SpatialVLA", "pi0": "pi0", "magma": "Magma-8B"}
SETS = {"photo_state_v1": "State", "photo_attribute_v1": "Attribute", "photo_emotion_v1": "Emotion"}
# bar order = palette slot order 1-4 (the order validated for adjacent marks); paper-style also hatched
CONDS = {"knowledge": "지식 질문", "tile_object": "정답 이름 (tile showing)", "explicit_spatial": "왼쪽/오른쪽",
         "paper_style": "논문식 (Put the cube on …)"}
COLORS = {"knowledge": "#2a78d6", "tile_object": "#eb6834", "explicit_spatial": "#1baf7a", "paper_style": "#eda100"}
HATCH = {"paper_style": "///"}
TEXT_DIRS = {"openvla": "outputs/text_only", "openvla-base": "outputs/text_only_openvla-base",
             "spatialvla": "outputs/text_only_spatialvla", "pi0": "outputs/text_only_pi0", "magma": "outputs/text_only_magma"}
TEXT_NAMES = {"openvla": "OpenVLA (SFT)", "openvla-base": "OpenVLA (base)", "spatialvla": "SpatialVLA", "pi0": "pi0", "magma": "Magma-8B"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = Path.home() / "YeahPick/data/fonts/NanumGothic-Regular.ttf"
if FONT.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"]  # DejaVu for π
plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "legend.frameon": False, "axes.unicode_minus": False, "hatch.color": "#ffffff", "hatch.linewidth": 1.2})


def wilson(k, n, z=1.96):
    if not n:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_p(k, n):
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n) if n else 1.0


def episodes(model, asset, cond):
    return [json.loads(l) for f in glob.glob(f"outputs/{model}-{asset}-{cond}-*/diagnostics/episode_log.jsonl")
            for l in open(f) if l.strip()]


def landed(rs, eps=0.15):
    ok = n = 0
    for r in rs:
        c = r["final_cube_position"][:2]
        L, R = math.dist(c, r["left_target_position"][:2]), math.dist(c, r["right_target_position"][:2])
        if min(L, R) <= eps:
            n += 1
            ok += ("left" if L < R else "right") == r["correct_side"]
    return ok, n


def style(ax, ylim=(0, 1.05)):
    ax.axhline(0.5, color=INK2, linewidth=1, zorder=4)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def bars(ax, values, errs=None):
    """values[cond][model_index] -> rate; draws grouped bars in slot order."""
    width = 0.8 / len(CONDS)
    for ci, cond in enumerate(CONDS):
        xs = [mi + (ci - (len(CONDS) - 1) / 2) * width for mi in range(len(MODELS))]
        ax.bar(xs, values[cond], width=width - 0.02, color=COLORS[cond], hatch=HATCH.get(cond), label=CONDS[cond],
               edgecolor=SURFACE, linewidth=1, zorder=3)
        if errs:
            lo = [v - e[0] for v, e in zip(values[cond], errs[cond])]
            hi = [e[1] - v for v, e in zip(values[cond], errs[cond])]
            ax.errorbar(xs, values[cond], yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=5)
    ax.set_xticks(range(len(MODELS)), list(MODELS.values()), color=INK2)


def main():
    out = HERE / "figures"
    tables = {"success": {}, "landed": {}, "text_direct": {}, "text_probe": {}, "colors": {}}

    # 1. placement success per set
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (asset, sname) in zip(axes, SETS.items()):
        vals = {c: [] for c in CONDS}
        for m in MODELS:
            for c in CONDS:
                rs = episodes(m, asset, c)
                k = sum(bool(r["task_success"]) for r in rs)
                vals[c].append(k / len(rs) if rs else 0)
                tables["success"][f"{m}|{asset}|{c}"] = [k, len(rs)]
        bars(ax, vals)
        style(ax)
        ax.set_title(sname, loc="left", fontsize=11, fontweight="bold", color=INK)
    axes[0].set_ylabel("배치 성공률", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "models_success.png", dpi=200)
    plt.close(fig)

    # 2. correct tile among landed episodes (all sets pooled)
    vals, errs = {c: [] for c in CONDS}, {c: [] for c in CONDS}
    for m in MODELS:
        for c in CONDS:
            rs = [r for a in SETS for r in episodes(m, a, c)]
            ok, n = landed(rs)
            vals[c].append(ok / n if n else 0)
            errs[c].append(wilson(ok, n))
            tables["landed"][f"{m}|{c}"] = [ok, n, len(rs), round(binom_p(ok, n), 4)]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bars(ax, vals, errs)
    style(ax)
    ax.set_ylabel("타일에 놓인 경우 중 정답 비율", color=INK2)
    ax.text(-0.45, 0.515, "chance 50%", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out / "models_landed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. text-only probe by layer, small multiples (2 series each)
    fig, axes = plt.subplots(1, len(TEXT_DIRS), figsize=(15, 3.4), sharey=True)
    for ax, (m, d) in zip(axes, TEXT_DIRS.items()):
        res = json.loads(Path(d, "results.json").read_text())
        tables["text_direct"][m] = res["direct"]["all"] + res["sanity"]
        for prompt, color, label in (("question", "#2a78d6", "질문 형식"), ("instruction", "#eb6834", "로봇 지시 형식")):
            rows = [r for r in res["probe"] if r["prompt"] == prompt and r["set"] == "all"]
            xs = list(range(len(rows)))
            ys = [r["k"] / r["n"] for r in rows]
            tables["text_probe"][f"{m}|{prompt}"] = [[r["layer"], r["k"], r["n"], round(r["p"], 4)] for r in rows]
            ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=3)
            ax.scatter(xs, ys, s=40, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        style(ax, (0, 1.02))
        ax.set_xticks(range(4), ["25%", "50%", "75%", "100%"])
        ax.set_xlabel("LLM 깊이", color=INK2)
        ax.set_title(TEXT_NAMES[m], loc="left", fontsize=10.5, fontweight="bold", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "models_text_probe.png", dpi=200)
    plt.close(fig)

    # Color sanity check (test_colors, knowledge condition, 6 questions x 2 layouts)
    for m in ("openvla", "openvla-base", "spatialvla", "pi0", "magma"):
        rs = episodes(m, "test_colors", "knowledge")
        tables["colors"][m] = [sum(bool(r["task_success"]) for r in rs), len(rs)]
    (HERE / "models_tables.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False))
    print("wrote models_success.png, models_landed.png, models_text_probe.png, models_tables.json")


if __name__ == "__main__":
    main()

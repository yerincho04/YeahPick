"""Recovery test curves: accuracy on unseen questions (held-out option pairs) during light LoRA fine-tuning.

Reads outputs/ab_recovery/<name>_recovery.json (mean over the 2 folds) and writes figures/recovery_curve.png.
Solid lines: real question. Grey dashed: no-question control (all models; nothing to recover).
Run from the repo root:  python diagnostics/report/plot_recovery.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
R = HERE.parents[1] / "outputs" / "ab_recovery"
FONT = Path.home() / "YeahPick/data/fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# palette slot order; OpenVLA models share a hue family so the "no recovery" group reads together
MODELS = [("prismatic-7b", "Prismatic-7B", "#2a78d6", "-"), ("llama2-7b", "Llama-2-7B", "#1baf7a", "-"),
          ("magma", "Magma", "#eda100", "-"), ("openvla-base-15ep", "OpenVLA", "#eb6834", "-"),
          ("openvla-sft-lr1e-4", "OpenVLA SFT", "#eb6834", "--")]


def curve(r, cond, key):
    return np.mean([[e[key] for e in f["curve"]] for f in r[cond]], axis=0)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharex=True)
    for ax, key, chance, title in ((axes[0], "accuracy", 0.5, "A. 처음 보는 문항 정확도"),
                                   (axes[1], "correct_both_orders", 0.25, "B. 두 선택지 순서 모두 정답")):
        for i, (name, label, color, ls) in enumerate(MODELS):
            r = json.loads((R / f"{name}_recovery.json").read_text())
            y = curve(r, "question", key)
            ax.plot(range(len(y)), y, color=color, linestyle=ls, linewidth=2.2, marker="o", markersize=4.5, label=label, zorder=3)
            c = curve(r, "no_question", key)
            ax.plot(range(len(c)), c, color=MUTED, linestyle=(0, (2, 2)), linewidth=1, zorder=2,
                    label="질문 제거 통제 (모든 모델)" if i == 0 else None)
        ax.axhline(chance, color=INK2, linewidth=1, zorder=1)
        ax.text(14.8, chance - 0.05, f"우연 {int(chance * 100)}%", ha="right", fontsize=8.5, color=INK2)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlim(-0.3, 15.3)
        ax.set_xlabel("LoRA 학습 epoch (0 = 학습 전)", color=INK2)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    k = [i for i, x in enumerate(l) if "통제" in x]  # control entry last
    h, l = [x for i, x in enumerate(h) if i not in k] + [h[i] for i in k], [x for i, x in enumerate(l) if i not in k] + [l[i] for i in k]
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for out in (HERE / "figures" / "recovery_curve.png", R / "recovery_curve.png"):
        fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote recovery_curve.png")


if __name__ == "__main__":
    main()

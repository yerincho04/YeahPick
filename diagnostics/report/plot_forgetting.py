"""Forgetting along training chains: text-only knowledge test (A/B), correct in both option orders.

Reads outputs/ab_knowledge/<name>_summary.json and writes figures/forgetting_chain.png.
Run from the repo root:  python diagnostics/report/plot_forgetting.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
AB = HERE.parents[1] / "outputs" / "ab_knowledge"
FONT = Path.home() / "YeahPick/data/fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CHAINS = [
    ("OpenVLA 계열", "#2a78d6", [("llama2-7b", "Llama-2-7B"), ("prismatic-7b", "Prismatic-7B"),
                               ("openvla-base", "OpenVLA"), ("openvla-sft", "OpenVLA SFT")],
     ["+ 이미지-텍스트 학습", "+ 로봇 행동 학습", "+ 추가 로봇 학습"]),
    ("Magma 계열", "#eb6834", [("llama3-8b-instruct-chat", "Llama-3-8B-Instruct"), ("magma-lm-chat", "Magma")],
     ["+ 이미지-텍스트 + 로봇 행동 학습"]),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 2.3]}, sharey=True)
    for ax, (title, color, members, steps) in zip(axes, CHAINS):
        vals = [json.loads((AB / f"{n}_summary.json").read_text())["all"]["correct_both_orders"] for n, _ in members]
        xs = range(len(members))
        ax.bar(xs, vals, width=0.55, color=color, edgecolor=SURFACE, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
        for i, step in enumerate(steps):  # training added between consecutive stages
            ax.annotate("", xy=(i + 0.72, 1.0), xytext=(i + 0.28, 1.0),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
            ax.text(i + 0.5, 1.04, step, ha="center", fontsize=8.5, color=INK2)
        ax.axhline(0.25, color=INK2, linewidth=1, linestyle=(0, (3, 2)), zorder=4)
        ax.text(len(members) - 0.55, 0.265, "우연 수준 25%", ha="right", fontsize=8.5, color=INK2)
        ax.set_xticks(list(xs), [l for _, l in members], fontsize=10, color=INK)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    axes[0].set_ylabel("두 선택지 순서 모두 정답 비율", color=INK2)
    fig.text(0.01, 0.01, "텍스트 지식 테스트 (230문항, 이미지 없음). Llama-2-7B는 A/B 형식 학습이 안 된 base 모델이라 낮게 나옴.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for out in (HERE / "figures" / "forgetting_chain.png", HERE.parents[1] / "outputs" / "bottleneck" / "forgetting_chain.png"):
        fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote forgetting_chain.png")


if __name__ == "__main__":
    main()

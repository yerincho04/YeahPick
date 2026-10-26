"""Probe accuracy per layer for every model (text only and image + text).

Reads outputs/ab_probe/<name>_probe.json and <name>_image_probe.json; writes figures/probe_layers.png.
Lines: linear probe on the real question. Grey band: per-layer control maximum (no-question prompt, shuffled labels).
Run from the repo root:  python diagnostics/report/plot_probe_layers.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
P = HERE.parents[1] / "outputs" / "ab_probe"
FONT = Path.home() / "YeahPick/data/fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MODELS = [("llama2-7b", "Llama-2-7B", "#1baf7a", "-"), ("prismatic-7b", "Prismatic-7B", "#2a78d6", "-"),
          ("openvla-base", "OpenVLA", "#eb6834", "-"), ("openvla-sft", "OpenVLA SFT", "#eb6834", "--"),
          ("llama3-8b-instruct", "Llama-3-8B-Instruct", "#4a3aa7", "-"), ("magma", "Magma", "#eda100", "-")]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, suffix, title in ((axes[0], "probe", "A. 텍스트만"), (axes[1], "image_probe", "B. 이미지 + 텍스트 (로봇 카메라 장면)")):
        ctrl_hi = None
        for name, label, color, ls in MODELS:
            f = P / f"{name}_{suffix}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            ys = r["question"]
            ax.plot(range(len(ys)), ys, color=color, linestyle=ls, linewidth=2.2, label=label, zorder=3)
            c = [max(a, b) for a, b in zip(r["no_question"], r["shuffled_max"])]
            ctrl_hi = c if ctrl_hi is None else [max(a, b) for a, b in zip(ctrl_hi, c)]
        ax.fill_between(range(len(ctrl_hi)), 0.5, ctrl_hi, color=MUTED, alpha=0.3, linewidth=0, zorder=1,
                        label="통제 조건 최고값 (질문 제거 · 라벨 섞기, 모든 모델)")
        ax.axhline(0.5, color=INK2, linewidth=1, zorder=2)
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["40%", "50% 우연", "60%", "70%", "80%", "90%", "100%"])
        ax.set_xlim(0, 32)
        ax.set_xticks(range(0, 33, 4))
        ax.set_xlabel("층 (0 = 임베딩)", color=INK2)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("선형 probe 정확도", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for out in (HERE / "figures" / "probe_layers.png", P / "probe_layers.png"):
        fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote probe_layers.png")


if __name__ == "__main__":
    main()

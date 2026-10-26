"""Is the answer encoded inside each model? Best-layer probe accuracy along both training chains.

Reads outputs/ab_probe/<name>_probe.json (text only) and <name>_image_probe.json (image + text) and writes
figures/probe_chain.png. Bars: best layer of the linear and the non-linear (MLP) probe. Grey band: the highest
score any control reached at any layer (no-question prompt, shuffled labels; linear and MLP), i.e. what
picking the best of all layers yields with no real signal.
Run from the repo root:  python diagnostics/report/plot_probe_chain.py
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
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODELS = [("llama2-7b", "Llama-2-7B"), ("prismatic-7b", "Prismatic-7B"), ("openvla-base", "OpenVLA"),
          ("openvla-sft", "OpenVLA SFT"), (None, None), ("llama3-8b-instruct", "Llama-3-8B\nInstruct"), ("magma", "Magma")]


def load(name, image):
    f = P / f"{name}_{'image_' if image else ''}probe.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    ctrl = max(max(r[k]) for k in ("no_question", "shuffled_max", "no_question_mlp", "shuffled_mlp_max") if k in r)
    return {"linear": max(r["question"]), "mlp": max(r["question_mlp"]), "control": ctrl}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, image, title in ((axes[0], False, "A. 텍스트만 (230문항 × 2순서)"), (axes[1], True, "B. 이미지 + 텍스트 (로봇 카메라 장면, 250문항 × 2배치)")):
        for x, (name, label) in enumerate(MODELS):
            if name is None:
                continue
            r = load(name, image)
            if r is None:
                ax.text(x, 0.45, "이미지 입력\n없음", ha="center", fontsize=8, color=MUTED)
                continue
            ax.bar(x - 0.19, r["linear"], width=0.36, color=BLUE, edgecolor=SURFACE, zorder=3, label="선형 probe" if x == 0 else None)
            ax.bar(x + 0.19, r["mlp"], width=0.36, color=ORANGE, edgecolor=SURFACE, zorder=3, label="비선형 probe (MLP)" if x == 0 else None)
            ax.fill_between([x - 0.42, x + 0.42], 0.5, r["control"], color=MUTED, alpha=0.35, zorder=4, linewidth=0,
                            label="통제 조건 최고값 (질문 제거 · 라벨 섞기)" if x == 0 else None)
            ax.text(x - 0.19, r["linear"] + 0.012, f"{100 * r['linear']:.0f}", ha="center", fontsize=8, color=INK2)
            ax.text(x + 0.19, r["mlp"] + 0.012, f"{100 * r['mlp']:.0f}", ha="center", fontsize=8, color=INK2)
        ax.axhline(0.5, color=INK2, linewidth=1, zorder=2)
        ax.axvline(4, color=GRID, linewidth=1)
        ax.set_ylim(0.4, 1.06)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["40%", "50% 우연", "60%", "70%", "80%", "90%", "100%"])
        ax.set_xticks([x for x, (n, _) in enumerate(MODELS) if n], [l for n, l in MODELS if n], fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.text(1.5, 1.035, "OpenVLA 계열", ha="center", fontsize=9, color=INK2)
        ax.text(5.5, 1.035, "Magma 계열", ha="center", fontsize=9, color=INK2)
    axes[0].set_ylabel("정답을 읽어낸 정확도 (최고 층)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for out in (HERE / "figures" / "probe_chain.png", HERE.parents[1] / "outputs" / "ab_probe" / "probe_chain.png"):
        fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote probe_chain.png")


if __name__ == "__main__":
    main()

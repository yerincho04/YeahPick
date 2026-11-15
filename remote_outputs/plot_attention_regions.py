"""Plot mean correct / gripper-and-cube / incorrect attention by task and outcome.

Reads object/ and spatial/ beneath this file. Uses only the Python standard
library; rerun with: python remote_outputs/plot_attention_regions.py
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GRAPH_ROOT = ROOT / "graphs"
LAYERS = (0.25, 0.5, 0.75, 1.0)
REGION_FIELDS = (
    "left_attention_percent", "arm_cube_attention_percent", "right_attention_percent",
)
COLORS = ("#e36c59", "#5795d3", "#5bb984")
LABELS = ("Correct object", "Gripper + cube", "Incorrect object")
RUN_NAME = re.compile(r"^openvla-(.+)-explicit_(object|spatial)-noswap-q\d+-\d+-seed\d+$")


def task_type(run_dir: Path, condition: str) -> str:
    match = RUN_NAME.match(run_dir.name)
    if not match or match.group(2) != condition:
        raise ValueError(f"Unexpected run directory: {run_dir}")
    name = match.group(1)
    for prefix in ("allmodels_decl_", "categories_all_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def read_episode(path: Path) -> dict[tuple[float, int], tuple[float, float, float]]:
    data = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            layer = round(float(row["layer_fraction"]), 2)
            step = int(row["policy_step"])
            if layer not in LAYERS:
                continue
            left, gripper, right = (float(row[field]) for field in REGION_FIELDS)
            if row["answer_side"] == "left":
                values = (left, gripper, right)
            elif row["answer_side"] == "right":
                values = (right, gripper, left)
            else:
                raise ValueError(f"Unknown answer side in {path}: {row['answer_side']}")
            data[(layer, step)] = values
    steps = sorted({step for _, step in data})
    if not steps or steps != list(range(steps[-1] + 1)):
        raise ValueError(f"Missing policy steps in {path}")
    if any((layer, step) not in data for layer in LAYERS for step in steps):
        raise ValueError(f"Incomplete layers in {path}")
    return data


def load_groups() -> dict[tuple[str, str, str], list[dict]]:
    groups = defaultdict(list)
    for condition in ("object", "spatial"):
        for run_dir in sorted((ROOT / condition).iterdir()):
            if not run_dir.is_dir():
                continue
            task = task_type(run_dir, condition)
            attention_dir = run_dir / "attention"
            for csv_path in sorted(attention_dir.glob("attention_regions_q*.csv")):
                question = csv_path.stem.removeprefix("attention_regions_")
                videos = list(attention_dir.glob(f"attention_{question}-s_*.mp4"))
                if len(videos) != 1:
                    print(f"SKIP {csv_path}: expected one completed attention video, found {len(videos)}")
                    continue
                result = re.search(r"-s_([01])\.mp4$", videos[0].name)
                if result is None:
                    print(f"SKIP {csv_path}: unknown success suffix")
                    continue
                try:
                    episode = read_episode(csv_path)
                except (ValueError, KeyError) as error:
                    print(f"SKIP {csv_path}: {error}")
                    continue
                outcome = "success" if result.group(1) == "1" else "failure"
                groups[(condition, task, outcome)].append(episode)
                groups[(condition, "global", outcome)].append(episode)
    return groups


def mean_trace(episodes: list[dict], layer: float) -> list[tuple[float, float, float] | None]:
    steps = sorted({step for episode in episodes for frac, step in episode if frac == layer})
    trace = []
    for step in steps:
        values = [episode[(layer, step)] for episode in episodes
                  if (layer, step) in episode
                  and all(math.isfinite(value) for value in episode[(layer, step)])]
        trace.append(tuple(sum(item[index] for item in values) / len(values) for index in range(3))
                     if values else None)
    return trace


def svg_chart(condition: str, task: str, outcome: str, episodes: list[dict],
              total_episodes: int | None = None) -> str:
    width, height = 1300, 1300
    plot_x, plot_w = 92, 1144
    panel_h, plot_h = 285, 214
    title = f"{condition.upper()}  /  {task.replace('_', ' ')}  /  {outcome.upper()}"
    total = len(episodes) if total_episodes is None else total_episodes
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253042}.title{font-size:25px;font-weight:700}'
        '.sub{font-size:15px;fill:#5b6572}.label{font-size:17px;font-weight:700}'
        '.tick{font-size:12px;fill:#6b7480}</style>',
        f'<text class="title" x="40" y="42">{escape(title)}</text>',
        f'<text class="sub" x="40" y="68">{outcome.capitalize()}: {len(episodes)}/{total} episodes · Mean attention share per policy step · nonfinite samples omitted per step</text>',
    ]
    for index, (color, label) in enumerate(zip(COLORS, LABELS)):
        lx = 39 + index * 203
        parts.append(f'<rect x="{lx}" y="82" width="16" height="16" rx="2" fill="{color}"/>')
        parts.append(f'<text class="sub" x="{lx + 23}" y="95">{escape(label)}</text>')

    for panel, layer in enumerate(LAYERS):
        trace = mean_trace(episodes, layer)
        top = 142 + panel * panel_h
        baseline = top + plot_h
        parts.append(f'<text class="label" x="{plot_x}" y="{top - 16}">Layer {int(layer * 100)}%</text>')
        for tick in (0, 25, 50, 75, 100):
            y = baseline - tick * plot_h / 100
            parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="#e4e8ed"/>')
            parts.append(f'<text class="tick" x="{plot_x - 13}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>')
        if not trace:
            parts.append(f'<text class="sub" x="{plot_x + plot_w / 2}" y="{top + plot_h / 2}" '
                         'text-anchor="middle">No episodes in this outcome</text>')
            continue
        bar_w = plot_w / len(trace)
        for step, values in enumerate(trace):
            if values is None:
                continue
            x = plot_x + step * bar_w + 0.6
            cumulative = 0.0
            for value, color in zip(values, COLORS):
                bar_h = max(0.0, value) * plot_h / 100
                y = baseline - (cumulative + value) * plot_h / 100
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.5, bar_w - 1.2):.2f}" '
                             f'height="{bar_h:.2f}" fill="{color}"/>')
                cumulative += value
            correct, gripper, incorrect = values
            if correct != incorrect:
                larger = 0 if correct > incorrect else 2
                segment_bottom = sum(values[:larger])
                dot_y = baseline - (segment_bottom + values[larger] / 2) * plot_h / 100
                dot_x = plot_x + (step + 0.5) * bar_w
                dot_radius = min(3.2, max(0.5, bar_w - 1.2) * 0.35)
                parts.append(f'<circle cx="{dot_x:.2f}" cy="{dot_y:.2f}" r="{dot_radius:.2f}" fill="#111111"/>')
        last_step = len(trace) - 1
        ticks = sorted({0, *range(10, last_step + 1, 10), last_step})
        for tick in ticks:
            x = plot_x + (tick + 0.5) * bar_w
            parts.append(f'<text class="tick" x="{x:.2f}" y="{baseline + 18}" text-anchor="middle">{tick}</text>')
        parts.append(f'<text class="tick" x="{plot_x + plot_w / 2}" y="{baseline + 38}" text-anchor="middle">Policy step</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    groups = load_groups()
    for condition in ("object", "spatial"):
        tasks = sorted({task for group_condition, task, _ in groups if group_condition == condition})
        for task in tasks:
            for outcome in ("success", "failure"):
                episodes = groups.get((condition, task, outcome), [])
                total = sum(len(groups.get((condition, task, result), []))
                            for result in ("success", "failure"))
                path = GRAPH_ROOT / condition / task / f"{outcome}.svg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(svg_chart(condition, task, outcome, episodes, total), encoding="utf-8")
                print(f"WROTE {path.relative_to(ROOT)} n={len(episodes)}")


if __name__ == "__main__":
    main()

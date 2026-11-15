"""Plot Object - Spatial attention with model rows and layer columns."""

from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from remote_outputs.plot_attention_regions import LAYERS, mean_trace  # noqa: E402
from remote_outputs2.plot_attention_regions import load_groups  # noqa: E402

ORANGE = '#ef6b35'
BLUE = '#2d7bd0'
MODELS = ('magma', 'openvla')


def episodes_for(groups: dict, model: str, condition: str, swap: str, task: str) -> list[dict]:
    return (groups.get((model, condition, swap, task, 'success'), [])
            + groups.get((model, condition, swap, task, 'failure'), []))


def difference_trace(object_episodes: list[dict], spatial_episodes: list[dict],
                     layer: float) -> list[float | None]:
    object_trace = mean_trace(object_episodes, layer)
    spatial_trace = mean_trace(spatial_episodes, layer)
    return [
        object_trace[step][0] - spatial_trace[step][0]
        if step < len(object_trace) and step < len(spatial_trace)
        and object_trace[step] is not None and spatial_trace[step] is not None
        else None
        for step in range(max(len(object_trace), len(spatial_trace)))
    ]


def grid_svg(task: str, swap: str, groups: dict) -> str:
    width, height = 1600, 680
    left, column_w, gap, plot_h = 132, 328, 28, 174
    row_top = (135, 390)
    traces = {}
    counts = {}
    for model in MODELS:
        object_episodes = episodes_for(groups, model, 'explicit_object', swap, task)
        spatial_episodes = episodes_for(groups, model, 'explicit_spatial', swap, task)
        counts[model] = (len(object_episodes), len(spatial_episodes))
        for layer in LAYERS:
            traces[(model, layer)] = difference_trace(object_episodes, spatial_episodes, layer)
    maximum = max((abs(value) for trace in traces.values()
                   for value in trace if value is not None), default=0)
    y_limit = next((limit for limit in (5, 10, 20, 30, 50, 75, 100) if maximum <= limit), 100)
    title = re.sub(r'\s+V\d+\b', '', task.replace('_', ' ').upper())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253042}.title{font-size:24px;font-weight:700}'
        '.label{font-size:16px;font-weight:700}.sub{font-size:13px;fill:#5b6572}'
        '.tick{font-size:11px;fill:#66717e}</style>',
        f'<text class="title" x="35" y="37">{escape(title)} / {swap.upper()}</text>',
        '<text class="sub" x="35" y="62">Correct-object attention difference: Object - Spatial (percentage points)</text>',
        f'<rect x="35" y="78" width="18" height="12" fill="{ORANGE}"/>',
        '<text class="sub" x="60" y="89">Object higher (+)</text>',
        f'<rect x="205" y="78" width="18" height="12" fill="{BLUE}" fill-opacity="0.55"/>',
        '<text class="sub" x="230" y="89">Spatial higher (-)</text>',
    ]

    def piece(x0: float, v0: float, x1: float, v1: float, zero: float) -> None:
        color = ORANGE if v0 + v1 > 0 else BLUE
        y0 = zero - v0 * plot_h / (2 * y_limit)
        y1 = zero - v1 * plot_h / (2 * y_limit)
        parts.append(f'<polygon points="{x0:.2f},{zero:.2f} {x0:.2f},{y0:.2f} '
                     f'{x1:.2f},{y1:.2f} {x1:.2f},{zero:.2f}" fill="{color}"/>')
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                     f'stroke="{color}" stroke-width="2"/>')

    for row, model in enumerate(MODELS):
        top = row_top[row]
        zero = top + plot_h / 2
        n_object, n_spatial = counts[model]
        parts.append(f'<text class="label" x="38" y="{zero}" text-anchor="middle" '
                     f'transform="rotate(-90 38 {zero})">{model.upper()}</text>')
        parts.append(f'<text class="sub" x="{left}" y="{top + plot_h + 52}">'
                     f'Object {n_object} episodes / Spatial {n_spatial} episodes</text>')
        for col, layer in enumerate(LAYERS):
            x0 = left + col * (column_w + gap)
            trace = traces[(model, layer)]
            parts.append(f'<text class="label" x="{x0}" y="{top - 12}">Layer {int(layer * 100)}%</text>')
            for fraction in (1, 0.5, 0, -0.5, -1):
                y = zero - fraction * plot_h / 2
                label = f'{fraction * y_limit:+g}' if fraction else '0'
                if fraction == 0:
                    parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + column_w}" y2="{y:.2f}" '
                                 'stroke="#6f7780" stroke-width="1.5"/>')
                parts.append(f'<text class="tick" x="{x0 - 7}" y="{y + 4:.2f}" text-anchor="end">{label}</text>')
            values = [value for value in trace if value is not None]
            if not values:
                parts.append(f'<text class="sub" x="{x0 + column_w / 2}" y="{zero - 8}" '
                             'text-anchor="middle">No paired data</text>')
                continue
            for step in range(len(trace) - 1):
                first, second = trace[step:step + 2]
                if first is None or second is None:
                    continue
                xa = x0 + step * column_w / max(1, len(trace) - 1)
                xb = x0 + (step + 1) * column_w / max(1, len(trace) - 1)
                if first * second < 0:
                    crossing = xa + (xb - xa) * abs(first) / (abs(first) + abs(second))
                    piece(xa, first, crossing, 0, zero)
                    piece(crossing, 0, xb, second, zero)
                else:
                    piece(xa, first, xb, second, zero)
            last_step = len(trace) - 1
            for tick in sorted({0, *range(20, last_step + 1, 20), last_step}):
                x = x0 + tick * column_w / max(1, last_step)
                parts.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 16}" '
                             f'text-anchor="middle">{tick}</text>')
            parts.append(f'<text class="sub" x="{x0 + column_w / 2}" y="{top + plot_h + 36}" '
                         'text-anchor="middle">Policy step</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def main() -> None:
    groups = load_groups()
    swaps = sorted({swap for _, _, swap, _, _ in groups})
    count = 0
    for swap in swaps:
        tasks = sorted({task for model, condition, group_swap, task, _ in groups
                        if group_swap == swap and model in MODELS
                        and condition in ('explicit_object', 'explicit_spatial')})
        for task in tasks:
            path = ROOT / 'comparisons' / swap / f'{task}.svg'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(grid_svg(task, swap, groups), encoding='utf-8')
            count += 1
            print(f'WROTE {path.relative_to(ROOT)}')
    print(f'DONE {count} grid comparison graphs')


if __name__ == '__main__':
    main()

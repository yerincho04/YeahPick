"""Plot the correct-object attention difference between object and spatial prompts."""

from __future__ import annotations

import sys
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from remote_outputs.plot_attention_regions import LAYERS, load_groups, mean_trace  # noqa: E402

COLORS = {"object": "#d55e53", "spatial": "#417fc2"}


def comparison_svg(title: str, object_episodes: list[dict],
                   spatial_episodes: list[dict]) -> str:
    title = re.sub(r'\s+V\d+\b', '', title)
    width, height = 1300, 1300
    plot_x, plot_w, plot_h, panel_h = 92, 1144, 210, 285
    differences = {}
    for layer in LAYERS:
        object_trace = mean_trace(object_episodes, layer)
        spatial_trace = mean_trace(spatial_episodes, layer)
        differences[layer] = [
            object_trace[step][0] - spatial_trace[step][0]
            if step < len(object_trace) and step < len(spatial_trace)
            and object_trace[step] is not None and spatial_trace[step] is not None
            else None
            for step in range(max(len(object_trace), len(spatial_trace)))
        ]
    maximum = max((abs(value) for trace in differences.values()
                   for value in trace if value is not None), default=0)
    y_limit = next((limit for limit in (5, 10, 20, 30, 50, 75, 100) if maximum <= limit), 100)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253042}.title{font-size:25px;font-weight:700}'
        '.sub{font-size:15px;fill:#5b6572}.label{font-size:17px;font-weight:700}'
        '.tick{font-size:12px;fill:#6b7480}</style>',
        f'<text class="title" x="40" y="42">{escape(title)}</text>',
        f'<text class="sub" x="40" y="68">Correct-object attention difference (pp): Object - Spatial | Object n={len(object_episodes)}, Spatial n={len(spatial_episodes)}</text>',
        f'<rect x="42" y="78" width="26" height="12" fill="{COLORS["object"]}" fill-opacity="0.55"/>',
        '<text class="sub" x="76" y="90">Object higher (+)</text>',
        f'<rect x="250" y="78" width="26" height="12" fill="{COLORS["spatial"]}" fill-opacity="0.55"/>',
        '<text class="sub" x="284" y="90">Spatial higher (-)</text>',
    ]

    def draw_piece(x0: float, value0: float, x1: float, value1: float,
                   zero_y: float) -> None:
        color = COLORS["object"] if value0 + value1 > 0 else COLORS["spatial"]
        y0 = zero_y - value0 * plot_h / (2 * y_limit)
        y1 = zero_y - value1 * plot_h / (2 * y_limit)
        parts.append(f'<polygon points="{x0:.2f},{zero_y:.2f} {x0:.2f},{y0:.2f} '
                     f'{x1:.2f},{y1:.2f} {x1:.2f},{zero_y:.2f}" '
                     f'fill="{color}" fill-opacity="0.50"/>')
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                     f'stroke="{color}" stroke-width="2.6"/>')

    for panel, layer in enumerate(LAYERS):
        trace = differences[layer]
        n_steps = len(trace)
        top = 142 + panel * panel_h
        zero_y = top + plot_h / 2
        parts.append(f'<text class="label" x="{plot_x}" y="{top - 16}">Layer {int(layer * 100)}%</text>')
        for fraction in (1, 0.5, 0, -0.5, -1):
            y = zero_y - fraction * plot_h / 2
            value = fraction * y_limit
            label = f'{value:+g}' if value else '0'
            color = '#7d8794' if fraction == 0 else '#e4e8ed'
            stroke_width = '1.6' if fraction == 0 else '1'
            parts.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" '
                         f'stroke="{color}" stroke-width="{stroke_width}"/>')
            parts.append(f'<text class="tick" x="{plot_x - 13}" y="{y + 4:.2f}" text-anchor="end">{label}</text>')
        values = [value for value in trace if value is not None]
        if not values:
            parts.append(f'<text class="sub" x="{plot_x + plot_w / 2}" y="{zero_y - 10}" '
                         'text-anchor="middle">No paired condition data</text>')
            continue
        mean_difference = sum(values) / len(values)
        parts.append(f'<text class="sub" x="{plot_x + plot_w}" y="{top - 16}" text-anchor="end">'
                     f'Mean Object - Spatial: {mean_difference:+.1f} pp</text>')
        for step in range(n_steps - 1):
            first, second = trace[step:step + 2]
            if first is None or second is None:
                continue
            x0 = plot_x + step * plot_w / max(1, n_steps - 1)
            x1 = plot_x + (step + 1) * plot_w / max(1, n_steps - 1)
            if first * second < 0:
                crossing = x0 + (x1 - x0) * abs(first) / (abs(first) + abs(second))
                draw_piece(x0, first, crossing, 0, zero_y)
                draw_piece(crossing, 0, x1, second, zero_y)
            else:
                draw_piece(x0, first, x1, second, zero_y)
        last_step = n_steps - 1
        ticks = sorted({0, *range(10, last_step + 1, 10), last_step})
        for tick in ticks:
            x = plot_x + tick * plot_w / max(1, last_step)
            parts.append(f'<text class="tick" x="{x:.2f}" y="{zero_y + 17}" text-anchor="middle">{tick}</text>')
        parts.append(f'<text class="tick" x="{plot_x + plot_w / 2}" y="{top + plot_h + 26}" text-anchor="middle">Policy step</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def main() -> None:
    groups = load_groups()
    tasks = sorted({task for _, task, _ in groups})
    for task in tasks:
        episodes = {}
        for condition in ('object', 'spatial'):
            episodes[condition] = (groups.get((condition, task, 'success'), [])
                                   + groups.get((condition, task, 'failure'), []))
        path = ROOT / 'comparisons' / f'{task}.svg'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(comparison_svg(task.replace('_', ' ').upper(),
                                       episodes['object'], episodes['spatial']), encoding='utf-8')
        print(f'WROTE {path.relative_to(ROOT)} object={len(episodes["object"])} spatial={len(episodes["spatial"])}')


if __name__ == '__main__':
    main()

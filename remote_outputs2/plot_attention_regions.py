"""Plot remote_outputs2 attention, grouping by the last recorded cube state.

The rollout records geometry before each policy action. The last geometry row
is therefore the state after the penultimate action. Success here applies the
Act2AnswerV4.evaluate() board rule to that recorded state.

Run from the repository root: python remote_outputs2/plot_attention_regions.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from remote_outputs.plot_attention_regions import read_episode, svg_chart  # noqa: E402


RUN_NAME = re.compile(
    r"^(magma|openvla)-attn-(.+)-(explicit_object|explicit_spatial|knowledge)-"
    r"(noswap|swap)-q\d+-\d+-seed\d+$"
)
CUBE_HALF_SIZE = 0.015
MARGIN_XY = 0.08


def task_type(asset: str) -> str:
    for prefix in ("allmodels_decl_", "categories_all_"):
        if asset.startswith(prefix):
            return asset[len(prefix):]
    return asset


def last_recorded_entities(path: Path) -> dict[str, dict]:
    last_step = -1
    entities = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["policy_step"])
            if step > last_step:
                last_step = step
                entities = {}
            if step == last_step and row["entity_type"] in ("left", "right", "cube"):
                entities[row["entity_type"]] = row
    if set(entities) != {"left", "right", "cube"}:
        raise ValueError(f"Missing final cube or tile geometry: {path}")
    return entities


def position(row: dict) -> tuple[float, float, float]:
    return tuple(float(row[f"world_{axis}"]) for axis in "xyz")


def last_recorded_success(geometry_path: Path, answer_side: str) -> bool:
    """Apply the environment's hard success criterion to the last saved pose."""
    if answer_side not in ("left", "right"):
        raise ValueError(f"Unknown answer side: {answer_side}")
    entities = last_recorded_entities(geometry_path)
    cube_x, cube_y, cube_z = position(entities["cube"])
    board = entities[answer_side]
    board_x, board_y, board_z = position(board)
    half_x = float(board["size_x"]) / 2
    half_y = float(board["size_y"]) / 2
    half_z = float(board["size_z"]) / 2
    return (
        abs(cube_x - board_x) <= half_x + MARGIN_XY
        and abs(cube_y - board_y) <= half_y + MARGIN_XY
        and cube_z - board_z >= half_z + CUBE_HALF_SIZE - 0.01
    )


def answer_side(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream), None)
    if first is None:
        raise ValueError(f"Empty attention CSV: {path}")
    return first["answer_side"]


def load_groups() -> dict[tuple[str, str, str, str, str], list[dict]]:
    groups = defaultdict(list)
    for run_dir in sorted((ROOT / "outputs").iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_NAME.match(run_dir.name)
        if match is None:
            print(f"SKIP unexpected run directory: {run_dir.name}")
            continue
        model, asset, condition, swap = match.groups()
        task = task_type(asset)
        attention_dir = run_dir / "attention"
        for regions_path in sorted(attention_dir.glob("attention_regions_q*.csv")):
            question = regions_path.stem.removeprefix("attention_regions_")
            geometry_path = attention_dir / f"attention_geometry_{question}.csv"
            if not geometry_path.is_file():
                print(f"SKIP missing geometry: {regions_path}")
                continue
            try:
                episode = read_episode(regions_path)
                success = last_recorded_success(geometry_path, answer_side(regions_path))
            except (ValueError, KeyError) as error:
                print(f"SKIP {regions_path}: {error}")
                continue
            outcome = "success" if success else "failure"
            groups[(model, condition, swap, task, outcome)].append(episode)
            groups[(model, condition, swap, "global", outcome)].append(episode)
    return groups


def main() -> None:
    groups = load_groups()
    settings = sorted({key[:3] for key in groups})
    count = 0
    for model, condition, swap in settings:
        tasks = sorted({task for m, c, s, task, _ in groups if (m, c, s) == (model, condition, swap)})
        for task in tasks:
            for outcome in ("success", "failure"):
                episodes = groups.get((model, condition, swap, task, outcome), [])
                total = sum(len(groups.get((model, condition, swap, task, result), []))
                            for result in ("success", "failure"))
                path = ROOT / "graphs" / model / condition / swap / task / f"{outcome}.svg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    svg_chart(f"{model} / {condition} / {swap}", task, outcome, episodes, total),
                    encoding="utf-8",
                )
                count += 1
                print(f"WROTE {path.relative_to(ROOT)} n={len(episodes)}")
    print(f"DONE {count} graphs")


if __name__ == "__main__":
    main()

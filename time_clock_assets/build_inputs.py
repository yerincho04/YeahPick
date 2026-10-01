"""Build the ten matched clock questions and ManiSkill tile assets.

If a tile GLB is missing, a placeholder is copied from test_colors. Existing GLBs
are preserved, including the finished clock tiles installed from asset_generator.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from digital_clock_png import make_clock, parse_time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ManiSkill/mani_skill/assets/carrot/test_colors/shapes/tile_5"
SOURCE_MODEL_DB = ROOT / "ManiSkill/mani_skill/assets/carrot/test_colors/model_db.json"
ASSET = ROOT / "ManiSkill/mani_skill/assets/carrot/time_clock_v1"
PNG_DIR = Path(__file__).with_name("input_pngs")

# A is the left tile in the original layout. The runner also evaluates a swapped layout.
# Each tuple is (knowledge instruction, A time, B time, correct side).
QUESTIONS = [
    (
        "Place the cube on the clock showing a time when stars are more likely to be visible.",
        "10:00 AM", "1:00 AM", "Right",
    ),
    (
        "Place the cube on the clock showing a time closer to sunset.",
        "6:00 AM", "7:00 PM", "Right",
    ),
    (
        "Place the cube on the clock showing a time closer to sunrise.",
        "6:00 AM", "7:00 PM", "Left",
    ),
    (
        "Place the cube on the clock showing a time when outdoor shadows are shorter.",
        "9:00 AM", "12:00 PM", "Right",
    ),
    (
        "Place the cube on the clock showing a time when streetlights are more likely to be on.",
        "11:00 PM", "2:00 PM", "Left",
    ),
    (
        "Place the cube on the clock showing a time when you are more likely to need a flashlight outdoors.",
        "1:00 PM", "2:00 AM", "Right",
    ),
    (
        "Place the cube on the clock showing a time closer to when the sun is highest in the sky.",
        "11:00 PM", "12:00 PM", "Right",
    ),
    (
        "Place the cube on the clock showing a time when it is typically warmer outdoors.",
        "4:00 PM", "5:00 AM", "Left",
    ),
    (
        "Place the cube on the clock showing a typical morning commute time.",
        "8:00 AM", "6:00 PM", "Left",
    ),
    (
        "Place the cube on the clock showing a time when school classes are typically in session.",
        "10:00 AM", "7:00 PM", "Left",
    ),
]


def tile_name(time: str) -> str:
    hour, minute, period = parse_time(time)
    return f"clock_{hour:02d}-{minute:02d}_{period}"


def matched_collision_obj(template_text: str, target_half_width: float) -> str:
    """Scale the template's x/y collision vertices to the GLB/model-db width."""
    vertices = []
    for line in template_text.splitlines():
        if line.startswith("v "):
            _, x, y, _ = line.split()
            vertices.extend((abs(float(x)), abs(float(y))))
    source_half_width = max(vertices)
    scale = target_half_width / source_half_width
    lines = []
    for line in template_text.splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            line = f"v {float(x) * scale:.8f} {float(y) * scale:.8f} {float(z):.8f}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    template = json.loads(SOURCE_MODEL_DB.read_text(encoding="utf-8"))["tile_5"]
    if not (SOURCE / "textured.glb").exists() or not (SOURCE / "collision.obj").exists():
        raise FileNotFoundError(f"Missing template tile at {SOURCE}")

    source_collision = (SOURCE / "collision.obj").read_text(encoding="utf-8")
    target_half_width = float(template["bbox"]["max"][0])
    collision = matched_collision_obj(source_collision, target_half_width)
    pairs = []
    manifest = []
    all_times = sorted({time for _, left, right, _ in QUESTIONS for time in (left, right)})
    model_db = {}
    for time in all_times:
        name = tile_name(time)
        hour, minute, period = parse_time(time)
        make_clock(hour, minute, period, PNG_DIR / f"{name}.png")
        shape = ASSET / "shapes" / name
        shape.mkdir(parents=True, exist_ok=True)
        # Keep a user's finished GLB if this builder is rerun later.
        glb_target = shape / "textured.glb"
        if not glb_target.exists():
            shutil.copy2(SOURCE / "textured.glb", glb_target)
        collision_target = shape / "collision.obj"
        if not collision_target.exists() or collision_target.read_text(encoding="utf-8") == source_collision:
            collision_target.write_text(collision, encoding="utf-8")
        model_db[name] = {**template, "name": name, "sign": name}

    for index, (instruction, left_time, right_time, answer) in enumerate(QUESTIONS):
        if left_time == right_time or answer not in {"Left", "Right"}:
            raise ValueError(f"Invalid clock pair at question {index}")
        correct_time = left_time if answer == "Left" else right_time
        wrong_time = right_time if answer == "Left" else left_time
        semantic_answer = f"{correct_time} clock"
        pairs.append({
            "index": index,
            "question_id": index,
            "question": instruction,
            "knowledge_instruction": instruction,
            "left": tile_name(left_time),
            "right": tile_name(right_time),
            "answer": answer,
            "semantic_answer": semantic_answer,
            "semantic_answer_id": tile_name(correct_time),
            "distractor_id": tile_name(wrong_time),
            "left_label": f"{left_time} clock",
            "right_label": f"{right_time} clock",
            "knowledge_category": "time",
        })
        manifest.append({
            "question_id": index,
            "left_time": left_time,
            "right_time": right_time,
            "answer": answer,
            "knowledge_instruction": instruction,
            "explicit_object_instruction": f"Place the cube on the {semantic_answer}.",
            "explicit_spatial_instruction": f"Place the cube on the {answer.lower()} tile.",
        })

    if len(pairs) != 10 or sum(pair["answer"] == "Left" for pair in pairs) != 5:
        raise ValueError("Expected ten questions with five left and five right answers")

    ASSET.mkdir(parents=True, exist_ok=True)
    (ASSET / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    (ASSET / "model_db.json").write_text(json.dumps(model_db, indent=2) + "\n", encoding="utf-8")
    with (Path(__file__).with_name("manifest.csv")).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Built {len(pairs)} questions and {len(all_times)} clock tiles at {ASSET}")
    placeholder = (SOURCE / "textured.glb").read_bytes()
    remaining = sum((ASSET / "shapes" / tile_name(time) / "textured.glb").read_bytes() == placeholder
                    for time in all_times)
    print(f"Placeholder GLBs remaining: {remaining}")


if __name__ == "__main__":
    main()

"""Builds ``emotion_pilot_v1``: 10 situational emotion questions, each answered by picking
one of two hand-drawn (PIL) cartoon faces. Same asset/pairs.json schema as semantic_pilot_v1,
so the matched-instruction pipeline (knowledge / explicit_object / explicit_spatial) runs
unchanged.

Run: python3 build_emotion.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from build_dataset import knowledge_instruction
from tile_asset import TILE_BBOX, TILE_DENSITY, TILE_SCALES, build_tile

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_NAME = "emotion_pilot_v1"
ASSET_DIR = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets" / "carrot" / ASSET_NAME
OUT_DIR = Path(__file__).resolve().parent / "emotion_output"

SIZE, BLACK = 512, (20, 20, 20)
SKIN = {"happy": (255, 215, 70), "sad": (130, 180, 240), "angry": (235, 80, 60),
        "surprised": (255, 215, 70), "scared": (190, 170, 230), "tired": (170, 200, 170)}


def face(emotion: str) -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((56, 56, 456, 456), fill=SKIN[emotion], outline=BLACK, width=8)
    ex_l, ex_r, ey = 186, 326, 210
    if emotion == "happy":
        for x in (ex_l, ex_r):
            d.ellipse((x - 18, ey - 26, x + 18, ey + 26), fill=BLACK)
        d.arc((156, 250, 356, 390), 20, 160, fill=BLACK, width=12)
    elif emotion == "sad":
        for x in (ex_l, ex_r):
            d.ellipse((x - 18, ey - 18, x + 18, ey + 18), fill=BLACK)
        d.arc((166, 320, 346, 420), 200, 340, fill=BLACK, width=12)
        d.ellipse((ex_l - 8, ey + 30, ex_l + 8, ey + 80), fill=(60, 120, 230))
    elif emotion == "angry":
        for x in (ex_l, ex_r):
            d.ellipse((x - 16, ey - 10, x + 16, ey + 22), fill=BLACK)
        d.line((ex_l - 45, ey - 50, ex_l + 35, ey - 15), fill=BLACK, width=14)
        d.line((ex_r + 45, ey - 50, ex_r - 35, ey - 15), fill=BLACK, width=14)
        d.arc((176, 330, 336, 410), 200, 340, fill=BLACK, width=12)
    elif emotion == "surprised":
        for x in (ex_l, ex_r):
            d.ellipse((x - 30, ey - 34, x + 30, ey + 34), fill=(255, 255, 255), outline=BLACK, width=6)
            d.ellipse((x - 10, ey - 10, x + 10, ey + 10), fill=BLACK)
        d.line((ex_l - 40, ey - 70, ex_l + 30, ey - 70), fill=BLACK, width=10)
        d.line((ex_r - 30, ey - 70, ex_r + 40, ey - 70), fill=BLACK, width=10)
        d.ellipse((226, 320, 286, 400), fill=BLACK)
    elif emotion == "scared":
        for x in (ex_l, ex_r):
            d.ellipse((x - 30, ey - 34, x + 30, ey + 34), fill=(255, 255, 255), outline=BLACK, width=6)
            d.ellipse((x - 6, ey - 6, x + 6, ey + 6), fill=BLACK)
        d.line((ex_l - 40, ey - 40, ex_l + 30, ey - 62), fill=BLACK, width=10)
        d.line((ex_r + 40, ey - 40, ex_r - 30, ey - 62), fill=BLACK, width=10)
        pts = [(176 + i * 32, 360 + (14 if i % 2 else -14)) for i in range(6)]
        d.line(pts, fill=BLACK, width=10)
    elif emotion == "tired":
        for x in (ex_l, ex_r):
            d.arc((x - 34, ey - 20, x + 34, ey + 30), 0, 180, fill=BLACK, width=10)
            d.line((x - 34, ey + 5, x + 34, ey + 5), fill=BLACK, width=6)
        d.ellipse((226, 340, 286, 390), fill=(120, 60, 60), outline=BLACK, width=6)
        d.text((380, 70), "Z z", fill=BLACK, font=ImageFont.load_default(size=56))
    return img


# (question, correct emotion, distractor emotion)
Q = [
    ("Which face shows how someone feels when they win a prize?", "happy", "sad"),
    ("Which face shows how someone feels when their pet runs away?", "sad", "happy"),
    ("Which face shows how someone feels when a stranger steals their toy?", "angry", "happy"),
    ("Which face shows how someone feels when a surprise party begins?", "surprised", "sad"),
    ("Which face shows how someone feels when they see a scary monster?", "scared", "happy"),
    ("Which face shows how someone feels after staying up all night?", "tired", "surprised"),
    ("Which face shows how someone feels when they drop their ice cream?", "sad", "happy"),
    ("Which face shows how someone feels when they hear a loud bang behind them?", "surprised", "tired"),
    ("Which face shows how someone feels when they get a birthday gift?", "happy", "angry"),
    ("Which face shows how someone feels when someone cuts in line?", "angry", "tired"),
]


def main() -> None:
    shapes = ASSET_DIR / "shapes"
    pairs, model_db, rows = [], {}, []
    for i, (question, good, bad) in enumerate(Q):
        tiles = {}
        for role, emo in (("good", good), ("bad", bad)):
            name = f"emo_{i}_{role}"
            build_tile(face(emo), shapes / name)
            model_db[name] = {"name": name, "sign": name, "bbox": TILE_BBOX,
                              "scales": TILE_SCALES, "density": TILE_DENSITY}
            tiles[role] = name
        # Alternate the correct side so position alone can't solve it.
        if i % 2 == 0:
            left, right, side = tiles["good"], tiles["bad"], "Left"
            left_label, right_label = f"{good} face", f"{bad} face"
        else:
            left, right, side = tiles["bad"], tiles["good"], "Right"
            left_label, right_label = f"{bad} face", f"{good} face"
        pair = {
            "index": i, "left": left, "right": right, "question": question,
            "knowledge_instruction": knowledge_instruction(question),
            "answer": side, "semantic_answer": f"{good} face",
            "left_label": left_label, "right_label": right_label,
            "knowledge_category": "emotion",
        }
        pairs.append(pair)
        rows.append({"asset": ASSET_NAME, "question_id": i, "question": question,
                     "knowledge_instruction": pair["knowledge_instruction"],
                     "correct": f"{good} face", "distractor": f"{bad} face", "correct_side": side,
                     "knowledge_category": "emotion"})
    (ASSET_DIR / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (ASSET_DIR / "model_db.json").write_text(json.dumps(model_db, indent=2) + "\n")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Contact sheet for human review of the faces.
    sheet = Image.new("RGB", (6 * 170, 170), "white")
    for k, emo in enumerate(SKIN):
        sheet.paste(face(emo).resize((160, 160)), (k * 170 + 5, 5))
    sheet.save(OUT_DIR / "faces.png")
    print(f"Built {len(pairs)} questions -> {ASSET_DIR}\nManifest/faces -> {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Builds the Phase 1 custom semantic diagnostic asset set:
  ManiSkill/mani_skill/assets/carrot/semantic_pilot_v1/{pairs.json, model_db.json, shapes/}
plus a dataset manifest (CSV + JSON) and a human-review contact sheet.

Run: python3 build_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from icons import ICONS
from questions import Q

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_NAME = "semantic_pilot_v1"
ASSET_DIR = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets" / "carrot" / ASSET_NAME
MANIFEST_DIR = Path(__file__).resolve().parent / "output"


def knowledge_instruction(question: str) -> str:
    """Convert the pilot's concise question into one consistent action instruction.

    The transformation preserves the knowledge predicate while avoiding a language-style
    confound against the two explicit ``Place the cube ...`` controls.
    """
    text = question.strip().removesuffix("?")
    if not text.startswith("Which "):
        raise ValueError(f"Cannot normalize knowledge instruction: {question!r}")
    predicate = text[len("Which "):]
    subject, separator, remainder = predicate.partition(" ")
    if not separator:
        raise ValueError(f"Cannot split knowledge instruction: {question!r}")
    # Multi-word subjects used by this pilot.
    if predicate.startswith("game piece "):
        subject, remainder = "game piece", predicate[len("game piece "):]
    if remainder.startswith(("is ", "would ", "can ", "lives ", "produces ", "has ",
                             "hops ", "tells ", "means ", "warns ", "indicates ",
                             "allows ", "shows ", "represents ", "symbolizes ", "serves ")):
        connector = "that "
    else:
        connector = ""
    return f"Place the cube on the {subject} {connector}{remainder}."


def build_assets():
    from tile_asset import TILE_BBOX, TILE_DENSITY, TILE_SCALES, build_tile

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    shapes_dir = ASSET_DIR / "shapes"

    pairs = []
    model_db = {}
    manifest_rows = []
    icon_cache: dict[str, Image.Image] = {}

    for i, (category, question, (key_a, label_a), (key_b, label_b), correct) in enumerate(Q):
        for key in (key_a, key_b):
            if key not in icon_cache:
                icon_cache[key] = ICONS[key]()

        tile_a, tile_b = f"sem_{i}_A", f"sem_{i}_B"
        build_tile(icon_cache[key_a], shapes_dir / tile_a)
        build_tile(icon_cache[key_b], shapes_dir / tile_b)
        for t in (tile_a, tile_b):
            model_db[t] = {
                "name": t, "sign": t, "bbox": TILE_BBOX,
                "scales": TILE_SCALES, "density": TILE_DENSITY,
            }

        # Alternate which side (left/right) the correct answer lands on for balance
        # (quality rule: "answer cannot be solved purely from left/right position").
        correct_key, wrong_key = (key_a, key_b) if correct == "A" else (key_b, key_a)
        correct_tile, wrong_tile = (tile_a, tile_b) if correct == "A" else (tile_b, tile_a)
        correct_label, wrong_label = (label_a, label_b) if correct == "A" else (label_b, label_a)
        if i % 2 == 0:
            left_tile, right_tile, answer_side = correct_tile, wrong_tile, "Left"
        else:
            left_tile, right_tile, answer_side = wrong_tile, correct_tile, "Right"

        left_label, right_label = (
            (correct_label, wrong_label) if answer_side == "Left" else (wrong_label, correct_label)
        )
        pairs.append({
            "index": i, "left": left_tile, "right": right_tile,
            "question": question,
            "knowledge_instruction": knowledge_instruction(question),
            "answer": answer_side,
            "semantic_answer": correct_label,
            "left_label": left_label,
            "right_label": right_label,
            "knowledge_category": category,
        })

        manifest_rows.append({
            "asset": ASSET_NAME,
            "question_id": i,
            "question": question,
            "knowledge_instruction": knowledge_instruction(question),
            "answer_A": label_a,
            "answer_B": label_b,
            "correct_answer": correct,
            "semantic_answer": correct_label,
            "image_A": f"shapes/{tile_a}/textured.glb",
            "image_B": f"shapes/{tile_b}/textured.glb",
            "knowledge_category": category,
            "image_source": "synthetic (PIL, hand-authored pictogram)",
        })

    (ASSET_DIR / "pairs.json").write_text(json.dumps(pairs, indent=2))
    (ASSET_DIR / "model_db.json").write_text(json.dumps(model_db, indent=2))

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / "manifest.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
    with open(MANIFEST_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built {len(pairs)} questions -> {ASSET_DIR}")
    print(f"Manifest -> {MANIFEST_DIR}/manifest.{{json,csv}}")
    return icon_cache


def update_metadata_only() -> None:
    """Add matched-experiment metadata without touching existing visual assets."""
    pairs_path = ASSET_DIR / "pairs.json"
    pairs = json.loads(pairs_path.read_text())
    if len(pairs) != len(Q):
        raise ValueError(f"Expected {len(Q)} pairs, found {len(pairs)}")

    manifest_rows = []
    for i, (pair, spec) in enumerate(zip(pairs, Q)):
        category, question, (key_a, label_a), (key_b, label_b), correct = spec
        correct_label = label_a if correct == "A" else label_b
        wrong_label = label_b if correct == "A" else label_a
        answer_side = str(pair["answer"]).lower()
        left_label, right_label = (
            (correct_label, wrong_label) if answer_side == "left" else (wrong_label, correct_label)
        )
        pair.update({
            "knowledge_instruction": knowledge_instruction(question),
            "semantic_answer": correct_label,
            "left_label": left_label,
            "right_label": right_label,
            "knowledge_category": category,
        })
        manifest_rows.append({
            "asset": ASSET_NAME,
            "question_id": i,
            "question": question,
            "knowledge_instruction": knowledge_instruction(question),
            "answer_A": label_a,
            "answer_B": label_b,
            "correct_answer": correct,
            "semantic_answer": correct_label,
            "image_A": f"shapes/sem_{i}_A/textured.glb",
            "image_B": f"shapes/sem_{i}_B/textured.glb",
            "knowledge_category": category,
            "image_source": "synthetic (PIL, hand-authored pictogram)",
        })

    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    with open(MANIFEST_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Updated metadata for {len(pairs)} questions; visual assets were not touched.")


def build_contact_sheet(icon_cache: dict[str, Image.Image]):
    """One row per question: instruction, answer A, answer B, expected answer."""
    thumb = 140
    row_h = thumb + 70
    cols_per_page = 1
    rows_per_page = 20
    font = ImageFont.load_default(size=20)

    n_pages = (len(Q) + rows_per_page - 1) // rows_per_page
    for page in range(n_pages):
        rows = Q[page * rows_per_page: (page + 1) * rows_per_page]
        sheet = Image.new("RGB", (900, row_h * len(rows)), "white")
        d = ImageDraw.Draw(sheet)
        for r, (category, question, (key_a, label_a), (key_b, label_b), correct) in enumerate(rows):
            y0 = r * row_h
            qid = page * rows_per_page + r
            img_a = icon_cache[key_a].resize((thumb, thumb))
            img_b = icon_cache[key_b].resize((thumb, thumb))
            sheet.paste(img_a, (10, y0 + 30))
            sheet.paste(img_b, (10 + thumb + 20, y0 + 30))
            d.text((10, y0 + 5), f"[{qid}] ({category}) {question}", fill="black", font=font)
            d.text((10, y0 + thumb + 32), f"A: {label_a}", fill="black", font=font)
            d.text((10 + thumb + 20, y0 + thumb + 32), f"B: {label_b}", fill="black", font=font)
            d.text((10 + 2 * thumb + 60, y0 + 60), f"correct: {correct}\n({label_a if correct == 'A' else label_b})",
                    fill=(0, 120, 0), font=font)
        out_path = MANIFEST_DIR / f"contact_sheet_page{page + 1}.png"
        sheet.save(out_path)
        print(f"Contact sheet -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        update_metadata_only()
    else:
        cache = build_assets()
        build_contact_sheet(cache)

"""Generate and validate original, synthetic Counting/Symmetry pilot assets.

Run from the repository root:
    python diagnostics/dataset_build/build_counting_symmetry.py
Dependencies: pillow, numpy, scipy, trimesh (available in the OpenVLA environment).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
import trimesh

from tile_asset import TILE_BBOX, TILE_DENSITY, TILE_SCALES, build_tile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "ManiSkill/mani_skill/assets/carrot"
SIZE, RADIUS, SEED = 512, 26, 20261005
INK = (30, 55, 85)
CORRECT_A = [True, False, True, False, False, True, False, True, False, True]
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def draw(points):
    im = Image.new("RGB", (SIZE, SIZE), "white")
    pen = ImageDraw.Draw(im)
    for x, y in points:
        pen.ellipse((x-RADIUS, y-RADIUS, x+RADIUS, y+RADIUS), fill=INK)
    return im


def validate_points(points):
    for i, (x, y) in enumerate(points):
        assert RADIUS + 20 <= x < SIZE - RADIUS - 20
        assert RADIUS + 20 <= y < SIZE - RADIUS - 20
        for xx, yy in points[:i]:
            assert (x-xx)**2 + (y-yy)**2 > (2*RADIUS + 10)**2


def counting_points(n, rng):
    grid = [(x, y) for x in (84, 170, 256, 342, 428)
            for y in (84, 170, 256, 342, 428)]
    return [(x+rng.randint(-5, 5), y+rng.randint(-5, 5))
            for x, y in rng.sample(grid, n)]


def symmetry_points(i):
    # Pixel centers reflect as x -> 511-x, giving exact raster symmetry.
    rows = (104, 200, 296, 392)
    count = 2 + i % 3
    chosen = random.Random(SEED + 100 + i).sample(list(rows), count)
    left = [(104 + 2*i + 58*((i+j) % 2), y) for j, y in enumerate(sorted(chosen))]
    good = left + [(SIZE-1-x, y) for x, y in left]
    bad = list(good)
    # Shift one existing circle horizontally, preserve count/size/colour.
    k = 0 if i % 2 == 0 else count
    x, y = bad[k]
    bad[k] = (x + (42 if i % 2 == 0 else -42), y)
    return good, bad


def main():
    report = {}
    for category in ("counting", "symmetry"):
        asset = category + "_pilot_v1"
        dest = ASSETS / asset
        (dest / "images").mkdir(parents=True, exist_ok=True)
        pairs, manifest, db, geometry, review = [], [], {}, [], []
        for i in range(10):
            qid = ("C" if category == "counting" else "S") + f"{i+1:02d}"
            if category == "counting":
                lower = 2 + i % 5
                target, other = (lower, lower+1) if i < 5 else (lower+1, lower)
                rng = random.Random(SEED+i)
                good, bad = counting_points(target, rng), counting_points(other, rng)
                question = f"Place the cube on the tile containing exactly {WORDS[target]} circles."
                good_label, bad_label = f"tile containing {WORDS[target]} circles", f"tile containing {WORDS[other]} circles"
            else:
                good, bad = symmetry_points(i)
                question = "Place the cube on the tile whose pattern is symmetric about its vertical centerline."
                good_label, bad_label = "vertically symmetric pattern", "vertically asymmetric pattern"
            points = (good, bad) if CORRECT_A[i] else (bad, good)
            labels = (good_label, bad_label) if CORRECT_A[i] else (bad_label, good_label)
            names = [f"{qid}_{side}" for side in ("A", "B")]
            ims = []
            for name, coords in zip(names, points):
                validate_points(coords)
                im = draw(coords)
                binary = np.any(np.asarray(im) != 255, axis=2)
                assert label(binary)[1] == len(coords), (name, "circle count")
                is_symmetric = np.array_equal(binary, binary[:, ::-1])
                if category == "symmetry":
                    assert is_symmetric == (coords == good), name
                im.save(dest / "images" / f"{name}.png")
                build_tile(im, dest / "shapes" / name)
                collision = dest / "shapes" / name / "collision.obj"
                collision.write_text(collision.read_text().rstrip() + "\n")
                scene = trimesh.load(dest / "shapes" / name / "textured.glb")
                texture = scene.geometry["GLTF"].visual.material.baseColorTexture
                assert np.array_equal(np.asarray(texture.convert("RGB")), np.asarray(im))
                db[name] = dict(name=name, sign=name, bbox=TILE_BBOX,
                                scales=TILE_SCALES, density=TILE_DENSITY)
                geometry.append(dict(tile=name, centers=coords, radius=RADIUS,
                                     circle_count=len(coords), vertically_symmetric=is_symmetric,
                                     png_sha256=hashlib.sha256((dest/"images"/f"{name}.png").read_bytes()).hexdigest()))
                ims.append(im)
            correct = "A" if CORRECT_A[i] else "B"
            pairs.append(dict(index=i, question_id=i, pilot_id=qid,
                              left=names[0], right=names[1], question=question,
                              knowledge_instruction=question,
                              answer="Left" if CORRECT_A[i] else "Right",
                              semantic_answer=good_label,
                              semantic_answer_id=names[0 if CORRECT_A[i] else 1],
                              distractor_id=names[1 if CORRECT_A[i] else 0],
                              left_label=labels[0], right_label=labels[1],
                              knowledge_category=category,
                              dataset_provenance="original_synthetic_pilot_v1"))
            manifest.append(dict(id=qid, category=category, image_a=f"images/{names[0]}.png",
                                 image_b=f"images/{names[1]}.png", question=question,
                                 correct_image=correct, count_a=len(points[0]), count_b=len(points[1])))
            review.append((qid, ims, correct, question))
        assert sum(p["answer"] == "Left" for p in pairs) == 5
        assert len({g["png_sha256"] for g in geometry}) == 20, "Duplicate images"
        for pair in pairs:
            assert pair["semantic_answer_id"] == pair["left" if pair["answer"] == "Left" else "right"]
        for name, value in (("pairs.json", pairs), ("model_db.json", db),
                            ("geometry.json", dict(seed=SEED, size=SIZE, tiles=geometry))):
            (dest / name).write_text(json.dumps(value, indent=2) + "\n")
        with (dest/"questions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(manifest)
        # Labels are only on the review sheet, never on model input images.
        sheet = Image.new("RGB", (1000, 5*250), "#e8edf2")
        pen = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=17)
        for i, (qid, ims, correct, _) in enumerate(review):
            x, y = (i % 2)*500, (i//2)*250
            pen.text((x+16,y+8), f"{qid} | answer {correct}", fill="black", font=font)
            for j, im in enumerate(ims):
                sheet.paste(im.resize((195,195)), (x+16+j*240,y+35))
                pen.text((x+216+j*240,y+40), "AB"[j], fill="black", font=font)
        sheet.save(dest/"contact_sheet.png")
        report[asset] = dict(questions=10, pngs=20, glbs=20, correct_left=5, correct_right=5,
                            checks="counts, margins, separation, exact symmetry, GLB textures, answer balance passed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

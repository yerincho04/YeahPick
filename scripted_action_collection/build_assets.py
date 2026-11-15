"""Convert the bundled VQA images into Act2Answer tabletop assets."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
ASSETS = ROOT / "generated_assets"
CATEGORIES = (
    "counting", "symmetry", "celebrity", "living_world", "traffic",
    "state", "attribute", "emotion", "public_info", "time_clock",
)
TILE_WIDTH = 0.11
TILE_THICKNESS = 0.006
PUBLIC_INFO_ASSET_VERSION = "white_svg_background_v1"


def on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".svg":
        import cairosvg

        png = cairosvg.svg2png(url=str(path), output_width=512, output_height=512)
        with Image.open(io.BytesIO(png)) as image:
            return on_white(image)
    with Image.open(path) as image:
        return on_white(ImageOps.exif_transpose(image))


def make_tile(image: Image.Image) -> bytes:
    texture = ImageOps.pad(
        image, (1024, 1024), method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
    )
    half = TILE_WIDTH / 2
    top = trimesh.Trimesh(
        vertices=np.array([
            [-half, -half, TILE_THICKNESS / 2 + 0.0001],
            [half, -half, TILE_THICKNESS / 2 + 0.0001],
            [half, half, TILE_THICKNESS / 2 + 0.0001],
            [-half, half, TILE_THICKNESS / 2 + 0.0001],
        ], dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        process=False,
    )
    top.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        image=texture,
    )
    box = trimesh.creation.box(extents=(TILE_WIDTH, TILE_WIDTH, TILE_THICKNESS))
    box.visual.face_colors = [245, 245, 245, 255]
    scene = trimesh.Scene()
    scene.add_geometry(box)
    scene.add_geometry(top)
    return scene.export(file_type="glb")


def read_pairs(category: str) -> list[dict]:
    rows = OrderedDict()
    path = INPUT / category / "vqa.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pair = rows.setdefault(row["pair_id"], {})
            pair[row["focus_tile"].lower()] = row
    pairs = []
    for pair_id, sides in rows.items():
        if set(sides) != {"left", "right"}:
            raise ValueError(f"{category}/{pair_id}: both focus rows are required")
        left, right = sides["left"], sides["right"]
        answer = left["knowledge"]["expected_answer"].strip().lower()
        if answer not in {"left", "right"}:
            raise ValueError(f"{category}/{pair_id}: invalid Knowledge answer {answer!r}")
        if right["knowledge"]["expected_answer"].strip().lower() != answer:
            raise ValueError(f"{category}/{pair_id}: conflicting Knowledge answers")
        pairs.append({
            "pair_id": pair_id,
            "left": left["images"]["left"],
            "right": left["images"]["right"],
            "answer": answer,
            "instruction": left["knowledge"]["instruction"],
        })
    return pairs


def build_category(category: str) -> tuple[Path, list[dict]]:
    pairs = read_pairs(category)
    output = ASSETS / category
    output.mkdir(parents=True, exist_ok=True)
    version_path = output / ".asset_version"
    rebuild_public_info = category == "public_info" and (
        not version_path.is_file()
        or version_path.read_text(encoding="utf-8").strip() != PUBLIC_INFO_ASSET_VERSION
    )
    images = OrderedDict()
    for pair in pairs:
        for side in ("left", "right"):
            item = pair[side]
            image_id, relative = str(item["image_id"]), item["path"]
            if image_id in images and images[image_id] != relative:
                raise ValueError(f"{category}: image ID {image_id!r} has multiple paths")
            images[image_id] = relative

    model_db = {}
    for image_id, relative in images.items():
        tile_name = f"tile_{image_id}"
        shape_dir = output / "shapes" / tile_name
        shape_dir.mkdir(parents=True, exist_ok=True)
        image_path = INPUT / relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        glb_path = shape_dir / "textured.glb"
        collision_path = shape_dir / "collision.obj"
        if rebuild_public_info or not glb_path.is_file():
            image = load_image(image_path)
            # The existing public-info GLBs use a clockwise quarter-turn.
            if category == "public_info":
                image = image.transpose(Image.Transpose.ROTATE_270)
            temporary_path = glb_path.with_suffix(".glb.tmp")
            temporary_path.write_bytes(make_tile(image))
            temporary_path.replace(glb_path)
        if not collision_path.is_file():
            collision_path.write_text(
                trimesh.creation.box(
                    extents=(TILE_WIDTH, TILE_WIDTH, TILE_THICKNESS)
                ).export(file_type="obj"), encoding="utf-8",
            )
        half = TILE_WIDTH / 2
        model_db[tile_name] = {
            "name": tile_name,
            "bbox": {
                "min": [-half, -half, -TILE_THICKNESS / 2],
                "max": [half, half, TILE_THICKNESS / 2],
            },
            "scales": [1.0], "density": 6368,
        }

    if rebuild_public_info:
        version_path.write_text(PUBLIC_INFO_ASSET_VERSION, encoding="utf-8")

    env_pairs = []
    for index, pair in enumerate(pairs):
        answer = pair["answer"]
        env_pairs.append({
            "index": index,
            "question_id": index,
            "pair_id": pair["pair_id"],
            "left": f"tile_{pair['left']['image_id']}",
            "right": f"tile_{pair['right']['image_id']}",
            "answer": answer.title(),
            "question": pair["instruction"],
            "knowledge_instruction": pair["instruction"],
            "knowledge_category": category,
            "semantic_answer": str(pair[answer]["image_id"]),
        })
    (output / "model_db.json").write_text(
        json.dumps(model_db, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "pairs.json").write_text(
        json.dumps(env_pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, pairs

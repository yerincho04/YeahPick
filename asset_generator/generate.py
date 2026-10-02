"""Turn every photo in input/ into a textured tabletop tile GLB.

Run from anywhere with: python asset_generator/generate.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import trimesh


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Matches the tile dimensions in test_colors/model_db.json (meters).
TILE_WIDTH = 0.11
TILE_THICKNESS = 0.006
TEXTURE_SIZE = 1024


def make_tile(image_path: Path) -> bytes:
    with Image.open(image_path) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
        texture = ImageOps.pad(
            photo,
            (TEXTURE_SIZE, TEXTURE_SIZE),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
        )

    half = TILE_WIDTH / 2
    top_z = TILE_THICKNESS / 2 + 0.0001

    tile = trimesh.creation.box(extents=(TILE_WIDTH, TILE_WIDTH, TILE_THICKNESS))
    tile.visual.face_colors = [245, 245, 245, 255]

    top = trimesh.Trimesh(
        vertices=np.array(
            [
                [-half, -half, top_z],
                [half, -half, top_z],
                [half, half, top_z],
                [-half, half, top_z],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        process=False,
    )
    top.visual = trimesh.visual.texture.TextureVisuals(
        # Increasing tile X must move left-to-right across the source image.
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        image=texture,
    )

    scene = trimesh.Scene()
    scene.add_geometry(tile)
    scene.add_geometry(top)
    return scene.export(file_type="glb")


def main() -> None:
    photos = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photos:
        print(f"No photos found in {INPUT_DIR}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True)

    for index, photo in enumerate(photos, start=1):
        target = run_dir / f"{index:04d}_{photo.stem}.glb"
        target.write_bytes(make_tile(photo))
        print(f"{photo.name} -> {target.relative_to(ROOT)}")

    print(f"Created {len(photos)} GLB files in {run_dir}")


if __name__ == "__main__":
    main()

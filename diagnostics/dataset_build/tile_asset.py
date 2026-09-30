"""Builds one Act2Answer ManiSkill tile asset (collision.obj + textured.glb) from a PIL
image, by reusing the exact box-mesh geometry/UV/material template that the official
Act2Answer `test_colors` asset ships (a flat 0.1452 x 0.1452 x 0.006m box, full [0,1] UV
unwrap on every face, single PBR baseColorTexture) -- only the texture image differs.

Reusing the shipped template (rather than re-deriving mesh/UV/material settings from
scratch) avoids introducing a second, potentially-inconsistent asset format.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import trimesh
from PIL import Image

_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[2]
    / "ManiSkill" / "mani_skill" / "assets" / "carrot" / "test_colors" / "shapes" / "tile_5"
)


def _load_template_scene() -> trimesh.Scene:
    return trimesh.load(_TEMPLATE_DIR / "textured.glb")


def build_tile(image: Image.Image, out_dir: Path) -> None:
    """Writes <out_dir>/collision.obj and <out_dir>/textured.glb for one tile."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_TEMPLATE_DIR / "collision.obj", out_dir / "collision.obj")

    scene = _load_template_scene()
    geo = scene.geometry["GLTF"]
    geo.visual.material.baseColorTexture = image.convert("RGBA")
    scene.export(str(out_dir / "textured.glb"))


TILE_BBOX = {
    "min": [-0.055, -0.055, -0.003],
    "max": [0.055, 0.055, 0.003],
}
TILE_SCALES = [1]
TILE_DENSITY = 6368

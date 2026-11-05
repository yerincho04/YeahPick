"""Side-by-side image pairs for the image + text probes: the two tile textures, no table and no robot.

For every question of the 10-category suite, both layouts: outputs/side_by_side/<asset>/q<NNN>_<noswap|swap>.png.
The canvas is square, with the two images letterboxed on white, because the models squash their input to a square
(OpenVLA/Prismatic 224x224, Magma 256x256) and a 2:1 canvas would distort the photos.
Run from the repo root:  python -m diagnostics.build_side_by_side
"""
import json
from pathlib import Path

import trimesh
from PIL import Image

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
OUT = Path("outputs/side_by_side")
ASSETS = ["allmodels_decl_photo_emotion_v1", "allmodels_decl_photo_attribute_v1", "allmodels_decl_photo_state_v1",
          "allmodels_decl_public_info_v1", "allmodels_decl_time_clock_v1", "allmodels_decl_counting_color_v2",
          "allmodels_decl_symmetry_color_v2", "categories_all_celebrity_50_v1", "categories_all_living_world_50_v1",
          "categories_all_traffic_kr_50_v1"]
CANVAS, TILE, GAP = 896, 432, 32


def texture(asset, tile):
    # some tiles (clocks) are a plain base mesh plus a textured top: use the textured mesh
    scene = trimesh.load(CARROT / asset / "shapes" / tile / "textured.glb")
    geo = next(g for g in scene.geometry.values() if type(g.visual).__name__ == "TextureVisuals")
    return geo.visual.material.baseColorTexture.convert("RGB")


def compose(left, right):
    canvas = Image.new("RGB", (CANVAS, CANVAS), "white")
    for k, im in enumerate((left, right)):
        im = im.copy()
        im.thumbnail((TILE, TILE), Image.LANCZOS)
        x = (k * (TILE + GAP)) + (TILE - im.width) // 2
        canvas.paste(im, (x, (CANVAS - im.height) // 2))
    return canvas


def main():
    n = 0
    for asset in ASSETS:
        (OUT / asset).mkdir(parents=True, exist_ok=True)
        cache = {}
        for p in json.loads((CARROT / asset / "pairs.json").read_text()):
            for t in (p["left"], p["right"]):
                cache.setdefault(t, texture(asset, t))
            q = int(p["index"])
            compose(cache[p["left"]], cache[p["right"]]).save(OUT / asset / f"q{q:03d}_noswap.png")
            compose(cache[p["right"]], cache[p["left"]]).save(OUT / asset / f"q{q:03d}_swap.png")
            n += 2
    print(f"{n} images -> {OUT}")


if __name__ == "__main__":
    main()

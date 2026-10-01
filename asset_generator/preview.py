"""Render generated GLB tiles with SAPIEN, without loading a VLA model.

Usage:
    python asset_generator/preview.py
    python asset_generator/preview.py path/to/tile.glb
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import sapien
from sapien.render import RenderCameraComponent


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ManiSkill"))
from mani_skill.utils import sapien_utils  # noqa: E402


def render_tile(glb_path: Path, image_path: Path) -> None:
    scene = sapien.Scene(
        [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]
    )
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0.3, 0.3, -1], [1.5, 1.5, 1.5], shadow=False)

    background = scene.create_actor_builder()
    background.add_box_visual(
        half_size=[0.18, 0.18, 0.01],
        material=sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.75, 1]),
    )
    background.initial_pose = sapien.Pose(p=[0, 0, -0.014])
    background.build_static(name="background")

    tile = scene.create_actor_builder()
    tile.add_visual_from_file(filename=str(glb_path))
    tile.initial_pose = sapien.Pose(p=[0, 0, 0])
    tile.build_static(name="tile")

    camera = RenderCameraComponent(768, 768)
    camera.set_fovy(0.9, compute_x=True)
    camera.near = 0.01
    camera.far = 10.0
    mount = sapien.Entity()
    mount.name = "preview_camera"
    mount.add_component(camera)
    scene.add_entity(mount)
    mount.pose = sapien_utils.look_at(
        eye=[0.0, -0.16, 0.30], target=[0.0, 0.0, 0.0]
    ).sp

    scene.update_render()
    camera.take_picture()
    rgb = np.asarray(camera.get_picture("Color"))[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(image_path)
    print(f"{glb_path} -> {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview GLB tiles in SAPIEN.")
    parser.add_argument("glb", nargs="?", type=Path, help="A single GLB; defaults to the latest output batch")
    args = parser.parse_args()

    if args.glb:
        glbs = [args.glb.resolve()]
    else:
        batches = sorted(path for path in (ROOT / "output").iterdir() if path.is_dir())
        if not batches:
            parser.error("No output batch found. Run generate.py first, or pass a GLB path.")
        glbs = sorted(batches[-1].glob("*.glb"))

    for glb in glbs:
        render_tile(glb, glb.parent / "preview" / f"{glb.stem}.png")


if __name__ == "__main__":
    main()

"""Perception check in the robot's own view (weakness #1).

The photo tiles were screened with SigLIP on their full-resolution textures. The robot instead sees
the whole table through a 640x480 camera that OpenVLA squashes to 224x224 ("resize-naive"), where each
tile covers only a few dozen pixels and the left tile is partly hidden by the gripper and cube.

For every episode's first camera frame this script crops both tiles and asks SigLIP (so400m/14, the
family of OpenVLA's vision encoder) which of the pair's two labels each crop shows, at
  - "camera":     crop from the 640x480 camera frame
  - "robot_view": the same crop after the frame is squashed to 224x224, i.e. the pixels OpenVLA gets
and compares with the full-resolution texture ("texture"). Tile boxes are fixed: tile poses do not
change between episodes (checked on frames; see --save-boxes).

Run from the repo root (GPU if available):
  python -m diagnostics.robot_view_recognizability --out-dir outputs/robot_view_check
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import torch
import trimesh
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
ASSETS = ("photo_state_v1", "photo_attribute_v1", "photo_emotion_v1")
# (x0, y0, x1, y1) in the 640x480 camera frame
BOXES = {"left": (100, 85, 305, 215), "right": (395, 165, 640, 360)}


def first_frame(asset: str, qid: int, layout: str) -> Image.Image | None:
    vids = sorted(Path("outputs").glob(f"openvla-{asset}-knowledge-{layout}-*/glob/vis_0_test/video_q{qid:03d}-*.mp4"))
    return Image.fromarray(imageio.get_reader(vids[0]).get_data(0)).convert("RGB") if vids else None


def texture(asset: str, tile: str) -> Image.Image:
    geo = next(iter(trimesh.load(CARROT / asset / "shapes" / tile / "textured.glb").geometry.values()))
    return geo.visual.material.baseColorTexture.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/robot_view_check")
    ap.add_argument("--siglip", default="google/siglip-so400m-patch14-224")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.siglip)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip).eval().to(device)

    def p_correct(img: Image.Image, labels: tuple[str, str], which: int) -> float:
        with torch.no_grad():
            inputs = processor(text=[f"a photo of a {l}" for l in labels], images=[img],
                               padding="max_length", return_tensors="pt").to(device)
            return float(model(**inputs).logits_per_image.softmax(-1)[0, which])

    rows, examples = [], []
    for asset in ASSETS:
        pairs = json.loads((CARROT / asset / "pairs.json").read_text())
        labels_of = {}
        for p in pairs:
            labels_of[p["left"]], labels_of[p["right"]] = p["left_label"], p["right_label"]
        for p in pairs:
            for layout in ("noswap", "swap"):
                frame = first_frame(asset, int(p["index"]), layout)
                if frame is None:
                    continue
                small = frame.resize((224, 224), Image.BICUBIC)  # what OpenVLA receives
                left, right = (p["right"], p["left"]) if layout == "swap" else (p["left"], p["right"])
                pair_labels = (labels_of[left], labels_of[right])
                for side, tile, which in (("left", left, 0), ("right", right, 1)):
                    x0, y0, x1, y1 = BOXES[side]
                    cam = frame.crop(BOXES[side])
                    sx, sy = 224 / frame.width, 224 / frame.height
                    rv = small.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)))
                    rows.append({
                        "asset": asset, "question_id": p["index"], "layout": layout, "side": side, "tile": tile,
                        "label": labels_of[tile], "robot_view_px": f"{rv.width}x{rv.height}",
                        "p_texture": round(p_correct(texture(asset, tile), pair_labels, which), 3),
                        "p_camera": round(p_correct(cam, pair_labels, which), 3),
                        "p_robot_view": round(p_correct(rv, pair_labels, which), 3),
                    })
                    if len(examples) < 6 and p["index"] % 7 == 0 and layout == "noswap":
                        examples.append((f"{asset} q{p['index']} {side}: {labels_of[tile]}", cam, rv))

    with open(out / "robot_view_recognizability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: accuracy (p > 0.5 for the true label) per asset x side x resolution
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for res in ("texture", "camera", "robot_view"):
            agg[(r["asset"], r["side"])][res].append(r[f"p_{res}"] > 0.5)
            agg[(r["asset"], "both")][res].append(r[f"p_{res}"] > 0.5)
    lines = ["| asset | side | crops | texture | camera 640x480 | robot view 224x224 |", "|---|---|---|---|---|---|"]
    for (asset, side), d in sorted(agg.items()):
        n = len(d["texture"])
        lines.append(f"| {asset} | {side} | {n} | " + " | ".join(f"{100 * sum(d[k]) / n:.0f}%" for k in ("texture", "camera", "robot_view")) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # contact sheet: camera crop vs what OpenVLA sees (upscaled for viewing)
    sheet = Image.new("RGB", (2 * 260 + 30, 290 * len(examples)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (title, cam, rv) in enumerate(examples):
        d.text((5, i * 290 + 2), title, fill="black")
        sheet.paste(cam.resize((260, 260)), (5, i * 290 + 20))
        sheet.paste(rv.resize((260, 260), Image.NEAREST), (275, i * 290 + 20))
    sheet.save(out / "examples_camera_vs_robot_view.png")
    boxes = first_frame(ASSETS[0], 0, "noswap")
    dd = ImageDraw.Draw(boxes)
    for b in BOXES.values():
        dd.rectangle(b, outline=(255, 0, 0), width=3)
    boxes.save(out / "tile_boxes.png")


if __name__ == "__main__":
    main()

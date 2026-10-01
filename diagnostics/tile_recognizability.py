"""Perception check: can OpenVLA's vision-encoder family (SigLIP so400m/14 @224) recognize each
answer tile of an asset set? Zero-shot classifies every tile texture among the set's labels.

A tile the encoder cannot recognize makes any downstream failure uninterpretable (it may be
perception, not knowledge or grounding), so such tiles should be replaced or excluded.

Run from the repo root (downloads google/siglip-so400m-patch14-224 on first use):
  python -m diagnostics.tile_recognizability --asset semantic_repeated_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import trimesh
from transformers import AutoModel, AutoProcessor

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"


def tile_labels(asset: str) -> dict[str, str]:
    """Tile model name -> human label, from pairs.json."""
    labels = {}
    for p in json.loads((CARROT / asset / "pairs.json").read_text()):
        labels[p["left"]] = p["left_label"]
        labels[p["right"]] = p["right_label"]
    return labels


def tile_image(asset: str, model: str):
    scene = trimesh.load(CARROT / asset / "shapes" / model / "textured.glb")
    geo = next(iter(scene.geometry.values()))
    return geo.visual.material.baseColorTexture.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--model", default="google/siglip-so400m-patch14-224")
    ap.add_argument("--template", default="a picture of a {}")
    args = ap.parse_args()

    labels = tile_labels(args.asset)
    names = sorted(set(labels.values()))
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()

    tiles = sorted(labels)
    images = [tile_image(args.asset, t) for t in tiles]
    texts = [args.template.format(n) for n in names]
    with torch.no_grad():
        inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt")
        logits = model(**inputs).logits_per_image  # [n_tiles, n_labels]
    probs = logits.softmax(-1)

    correct = 0
    print(f"{'tile':32s} {'true label':20s} {'predicted':20s} p(true)")
    for i, t in enumerate(tiles):
        pred = names[int(probs[i].argmax())]
        p_true = float(probs[i, names.index(labels[t])])
        ok = pred == labels[t]
        correct += ok
        print(f"{t:32s} {labels[t]:20s} {pred:20s} {p_true:.2f} {'' if ok else '<-- NOT RECOGNIZED'}")
    print(f"\nrecognized {correct}/{len(tiles)} tiles (chance {1 / len(names):.0%})")


if __name__ == "__main__":
    main()

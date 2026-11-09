"""Candidate crops for the v2 photo VQA dataset (state, material) from OVAD (COCO val2017, dense human
attribute labels with explicit negatives; Bravo et al., CVPR 2023).

An object is a candidate for class A of a family only if OVAD marks A positive AND the opposite class negative
(state), or exactly one material positive (material). Crops: square around the box with 15% margin, 384 px.
Writes <out>/<category>/<family>/<class>/<ann_id>.jpg and <out>/<category>/candidates.json.
COCO images are fetched with curl (python urllib fails the TLS check on this cluster).
"""
import argparse, json, subprocess
from pathlib import Path
from PIL import Image

STATE = {  # family -> (COCO category, attribute A, attribute B)
    "toilet_open": ("toilet", "open", "closed"), "handbag_open": ("handbag", "open", "closed"),
    "oven_open": ("oven", "open", "closed"), "suitcase_open": ("suitcase", "open", "closed"),
    "bottle_open": ("bottle", "open", "closed"), "book_open": ("book", "open", "closed"),
    "tv_on": ("tv", "on", "off"), "laptop_on": ("laptop", "on", "off"), "phone_on": ("cell phone", "on", "off"),
    "bowl_empty": ("bowl", "empty", "full"), "cup_empty": ("cup", "empty", "full"), "wineglass_empty": ("wine glass", "empty", "full"),
    "vase_empty": ("vase", "empty", "full"), "bench_empty": ("bench", "empty", "full"), "chair_empty": ("chair", "empty", "full"),
    "couch_empty": ("couch", "empty", "full"), "table_empty": ("dining table", "empty", "full"),
    "apple_cut": ("apple", "cut", "whole"), "banana_cut": ("banana", "cut", "whole"), "orange_cut": ("orange", "cut", "whole"),
    "cake_cut": ("cake", "cut", "whole"), "pizza_cut": ("pizza", "cut", "whole"), "sandwich_cut": ("sandwich", "cut", "whole"),
    "donut_cut": ("donut", "cut", "whole"), "carrot_cut": ("carrot", "cut", "whole"),
    "broccoli_cooked": ("broccoli", "cooked", "raw"), "carrot_cooked": ("carrot", "cooked", "raw"),
    "bowl_clean": ("bowl", "clean", "dirty"), "toilet_clean": ("toilet", "clean", "dirty"), "table_clean": ("dining table", "clean", "dirty"),
    "hydrant_clean": ("fire hydrant", "clean", "dirty"), "suitcase_clean": ("suitcase", "clean", "dirty"),
    "motorcycle_clean": ("motorcycle", "clean", "dirty"), "knife_clean": ("knife", "clean", "dirty"), "bench_clean": ("bench", "clean", "dirty"),
    "elephant_clean": ("elephant", "clean", "dirty"), "sheep_clean": ("sheep", "clean", "dirty"), "stopsign_clean": ("stop sign", "clean", "dirty"),
    "bed_tidy": ("bed", "tidy", "messy"), "table_tidy": ("dining table", "tidy", "messy"), "couch_tidy": ("couch", "tidy", "messy"),
    "bear_wet": ("bear", "wet", "dry"),
}
ATTR = {"open": "state:open", "closed": "state:closed", "on": "state:on", "off": "state:off", "empty": "state:empty",
        "full": "state:full/whole", "whole": "state:full/whole", "cut": "state:piece/cut",
        "clean": "cleanliness:clean/neat", "dirty": "cleanliness:unclean/dirt/dirty/muddy",
        "cooked": "cooked:cooked/baked/warmed", "raw": "cooked:raw/fresh",
        "tidy": "order:ordered/arranged/organized/tidy",
        "messy": "order:messy/disordered/unordered/disorganized/unorganized/cluttered/untidy", "wet": "state:wet", "dry": "state:dry"}
MATERIAL_OBJECTS = ["chair", "cup", "bowl", "dining table", "bench", "bottle", "couch", "handbag", "sink", "vase", "potted plant",
                    "suitcase", "spoon", "laptop", "umbrella", "kite", "surfboard", "skateboard", "tennis racket", "fork", "knife",
                    "boat", "fire hydrant", "oven", "microwave", "tv", "cell phone", "skis"]
MATERIAL = {"material:wood/wooden/bamboo/hardwood": "wood",
            "material:metal/metallic/aluminum/brass/copper-zinc/iron/stainless steel/steel/silver": "metal",
            "material:polymers/plastic/rubber/styrofoam/polymer": "plastic", "material:glass": "glass",
            "material:textile/cloth/fabric/denim/cotton/jean/silk/plush": "fabric", "material:leather": "leather",
            "material:ceramic/brick/porcelain": "ceramic", "material:paper/cardboard": "paper",
            "material:stone/granite/cobblestone/gravel/marble/pebbled/rocky/sandy": "stone"}
MIN_SIDE, PER_CLASS = 64, 12


def fetch(file_name, cache):
    p = cache / file_name
    if not p.exists():
        subprocess.run(["curl", "-sf", "-o", str(p), f"http://images.cocodataset.org/val2017/{file_name}"], check=False)
    return Image.open(p).convert("RGB") if p.exists() else None


def crop(img, bbox, margin=0.15, size=384):
    x, y, w, h = bbox
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    box = [int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)]
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    canvas.paste(img, (-box[0], -box[1]))
    return canvas.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovad", type=Path, required=True)
    ap.add_argument("--coco-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.coco_cache.mkdir(parents=True, exist_ok=True)
    d = json.loads(a.ovad.read_text())
    N = {x["name"]: x["id"] for x in d["attributes"]}
    cat = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    anns = [x for x in d["annotations"] if min(x["bbox"][2:]) >= MIN_SIDE and not x.get("iscrowd")]
    lab = lambda x, n: x["att_vec"][N[n]]  # 1 positive, 0 negative, -1 unknown
    jobs = []  # (category, family, class, noun, ann)
    for fam, (c, A, B) in STATE.items():
        for cls, other in ((A, B), (B, A)):
            xs = [x for x in anns if cat[x["category_id"]] == c and lab(x, ATTR[cls]) == 1 and lab(x, ATTR[other]) == 0]
            xs.sort(key=lambda x: -x["area"])
            jobs += [("state", fam, cls, c, x) for x in xs[:PER_CLASS]]
    for c in MATERIAL_OBJECTS:
        by = {}
        for x in anns:
            if cat[x["category_id"]] != c:
                continue
            pos = [m for m in MATERIAL if lab(x, m) == 1]
            if len(pos) == 1:
                by.setdefault(MATERIAL[pos[0]], []).append(x)
        by = {k: sorted(v, key=lambda x: -x["area"])[:PER_CLASS] for k, v in by.items() if len(v) >= 2}
        if len(by) >= 2:
            for m, xs in by.items():
                jobs += [("material", c.replace(" ", "_"), m, c, x) for x in xs]
    meta = {"state": [], "material": []}
    for category, fam, cls, noun, x in jobs:
        im = fetch(imgs[x["image_id"]]["file_name"], a.coco_cache)
        if im is None:
            continue
        rel = f"{category}/{fam}/{cls}/{x['id']}.jpg"
        (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
        crop(im, x["bbox"]).save(a.out / rel, quality=92)
        meta[category].append({"file": rel, "family": fam, "class": cls, "noun": noun, "ovad_ann_id": x["id"],
                               "coco_image_id": x["image_id"], "coco_file": imgs[x["image_id"]]["file_name"],
                               "bbox": x["bbox"], "coco_license": imgs[x["image_id"]].get("license")})
    for k, v in meta.items():
        (a.out / k).mkdir(parents=True, exist_ok=True)
        (a.out / k / "candidates.json").write_text(json.dumps(v, indent=1))
        print(k, len(v), "crops,", len({m["family"] for m in v}), "families")


if __name__ == "__main__":
    main()

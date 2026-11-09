"""Face-crop candidates for the v2 emotion VQA set.

People come from VAW (Visual Genome) and OVAD (COCO val2017) with an expression label and no conflicting
expression label. Inside each person box the largest frontal face (OpenCV Haar cascade) is found and cropped
square with generous margin (384 px), so the expression is readable; boxes without a detectable face are dropped.
Writes <out>/emotion/<class>/<source>_<id>.jpg and <out>/emotion/candidates.json.
"""
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PEOPLE = {"man", "woman", "girl", "boy", "person", "child", "lady", "guy", "kid", "men", "women", "face", "head"}
VAW = {"happy": {"happy", "smiling", "laughing"}, "surprised": {"surprised"},
       "sleepy": {"sleepy", "sleeping", "tired", "yawning", "napping", "asleep"}, "neutral": {"serious", "calm"},
       "upset": {"frowning", "angry", "sad", "upset", "crying", "mad"}}
OVAD = {"happy": "face expression:happy/smiling/laughing/grinning/joyful", "surprised": "face expression:fear/surprise",
        "sleepy": "face expression:sleepy/sleeping", "neutral": "face expression:neutral/calm/serious",
        "upset": ["face expression:angry/mad", "face expression:disgust/frowning", "face expression:sad/unhappy"]}
CAP = 70
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def curl(url, path):
    if not path.exists():
        subprocess.run(["curl", "-sf", "-o", str(path), url], check=False)
    return path.exists()


def face_crop(img, bbox, size=384):
    x, y, w, h = [int(v) for v in bbox]
    region = img.crop((max(0, x), max(0, y), x + w, y + h))
    g = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    if len(faces) == 0:
        return None, 0
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cx, cy, side = max(0, x) + fx + fw / 2, max(0, y) + fy + fh / 2, fw * 2.0
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    canvas.paste(img, (-box[0], -box[1]))
    return canvas.resize((size, size), Image.LANCZOS), int(fw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaw-dir", type=Path, required=True)
    ap.add_argument("--vg-cache", type=Path, required=True)
    ap.add_argument("--ovad", type=Path, required=True)
    ap.add_argument("--coco-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = []
    vaw = []
    for f in ("train_part1.json", "train_part2.json", "val.json", "test.json"):
        vaw += json.loads((a.vaw_dir / f).read_text())
    for cls, attrs in VAW.items():
        others = set().union(*(v for k, v in VAW.items() if k != cls))
        xs = [d for d in vaw if d["object_name"] in PEOPLE and attrs & set(d["positive_attributes"])
              and not others & set(d["positive_attributes"]) and min(d["instance_bbox"][2:]) >= 50]
        xs.sort(key=lambda d: -d["instance_bbox"][2] * d["instance_bbox"][3])
        seen = set()
        for d in xs:
            if d["image_id"] in seen:
                continue
            seen.add(d["image_id"])
            jobs.append((cls, "vg", d["image_id"], d["instance_bbox"], d["instance_id"]))
            if len(seen) >= CAP:
                break
    o = json.loads(a.ovad.read_text())
    N = {x["name"]: x["id"] for x in o["attributes"]}
    imgs = {i["id"]: i for i in o["images"]}
    names = {k: (v if isinstance(v, list) else [v]) for k, v in OVAD.items()}
    for x in o["annotations"]:
        if x["category_id"] != 1 or min(x["bbox"][2:]) < 50:
            continue
        pos = [k for k, ns in names.items() if any(x["att_vec"][N[n]] == 1 for n in ns)]
        if len(pos) == 1:
            jobs.append((pos[0], "coco", x["image_id"], x["bbox"], x["id"]))
    meta, used = [], set()
    for cls, src, iid, bbox, inst in jobs:
        if (src, iid) in used:
            continue
        if src == "vg":
            p = a.vg_cache / f"{iid}.jpg"
            ok = curl(f"https://cs.stanford.edu/people/rak248/VG_100K/{iid}.jpg", p) or \
                curl(f"https://cs.stanford.edu/people/rak248/VG_100K_2/{iid}.jpg", p)
        else:
            p = a.coco_cache / imgs[iid]["file_name"]
            ok = curl(f"http://images.cocodataset.org/val2017/{imgs[iid]['file_name']}", p)
        if not ok:
            continue
        crop, fw = face_crop(Image.open(p).convert("RGB"), bbox)
        if crop is None:
            continue
        used.add((src, iid))
        rel = f"emotion/{cls}/{src}_{inst}.jpg"
        (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
        crop.save(a.out / rel, quality=92)
        meta.append({"file": rel, "family": "person", "class": cls, "noun": "person", "source": src, "image_id": iid,
                     "instance_id": inst, "person_bbox": bbox, "face_px": fw})
    (a.out / "emotion" / "candidates.json").write_text(json.dumps(meta, indent=1))
    from collections import Counter
    print(Counter((m["class"], m["source"]) for m in meta))


if __name__ == "__main__":
    main()

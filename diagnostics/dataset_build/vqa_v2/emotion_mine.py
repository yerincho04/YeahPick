"""Mine face candidates for the emotion set (happy / surprised / angry / sad) and sleeping-person candidates.

Faces: OpenCV frontal-face detector on COCO val2017 (all 5,000 images), the earlier VAW/OVAD face crops and the
Wikimedia Commons downloads; each face is cropped square at 2x the detected box (384 px) and scored by the
ViT expression classifier trpakov/vit-face-expression (Apache-2.0). Kept: faces whose top class is the target with
p >= MIN_P, ranked by p. Only strong, unmistakable expressions survive; the final pick is by eye.
Sleeping: Commons "asleep" downloads and earlier sleepy crops, centre-cropped, ranked by SigLIP
("a photo of a sleeping person" vs "a photo of an awake person").
Writes <out>/<class>/*.jpg, <out>/candidates.json and <out>/sheet_<class>.png (numbered).
"""
import argparse, json, re
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification, AutoProcessor

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
TARGET = {"happy": "happy", "surprised": "surprise", "angry": "angry", "sad": "sad"}
MIN_P, MIN_FACE, KEEP = 0.6, 64, 60


def faces(img):
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    out = []
    for x, y, w, h in CASCADE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=7, minSize=(MIN_FACE, MIN_FACE)):
        cx, cy, side = x + w / 2, y + h / 2, w * 2.0
        box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
        c = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
        c.paste(img, (-box[0], -box[1]))
        tight = img.crop((max(0, x), max(0, y), x + w, y + h))
        out.append((c.resize((384, 384), Image.LANCZOS), tight, int(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True, help="earlier candidate dir (vqa_v2_cand)")
    ap.add_argument("--web", type=Path, required=True, help="Commons download dir with candidates.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(32)
    fp = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
    fm = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").eval()
    labels = fm.config.id2label
    sp = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    sm = AutoModel.from_pretrained("google/siglip-so400m-patch14-224").eval()

    def fer(tights):
        with torch.no_grad():
            return fm(**fp(images=tights, return_tensors="pt")).logits.softmax(-1).numpy()

    web = json.loads((a.web / "candidates.json").read_text())
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
    sources = []  # (image path, provenance dict)
    for p in sorted((a.coco).glob("*.jpg")):
        sources.append((p, {"source": "COCO val2017", "coco_file": p.name}))
    for m in json.loads((a.old / "emotion" / "candidates.json").read_text()):
        if m["class"] != "sleepy":
            sources.append((a.old / m["file"], {k: v for k, v in m.items() if k not in ("file", "class", "family", "noun")}))
    for m in web:
        if m["class"] != "asleep":
            sources.append((a.web / m["file"], {"source": "Wikimedia Commons", "title": m["title"], "page": m["page"],
                                                "artist": strip(m["artist"]), "license": m["license"]}))
    found = {c: [] for c in TARGET}
    batch = []

    def flush():
        if not batch:
            return
        P = fer([t for _, t, _, _ in batch])
        for (crop, _, w, prov), pr in zip(batch, P):
            top = labels[int(pr.argmax())]
            for c, lab in TARGET.items():
                if top == lab and pr.max() >= MIN_P:
                    found[c].append((float(pr.max()), crop, {**prov, "face_px": w, "fer": {labels[i]: round(float(v), 3) for i, v in enumerate(pr)}}))
        batch.clear()

    for i, (path, prov) in enumerate(sources):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        for crop, tight, w in faces(img):
            batch.append((crop, tight, w, prov))
        if len(batch) >= 64:
            flush()
        if i % 1000 == 0:
            print(i, len(sources), {c: len(v) for c, v in found.items()}, flush=True)
    flush()
    meta = []
    font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 16)
    T, C = 150, 13

    def sheet(cls, items):
        rows = (len(items) + C - 1) // C or 1
        sh = Image.new("RGB", (C * T, rows * (T + 2)), "white")
        d = ImageDraw.Draw(sh)
        for n, (p, crop) in enumerate(items):
            x, y = (n % C) * T, (n // C) * (T + 2)
            sh.paste(crop.resize((T, T)), (x, y))
            d.rectangle((x, y, x + 60, y + 18), fill="white")
            d.text((x + 2, y), f"{n} {p:.2f}", fill="red", font=font)
        sh.save(a.out / f"sheet_{cls}.png")

    for c, items in found.items():
        items.sort(key=lambda t: -t[0])
        (a.out / c).mkdir(parents=True, exist_ok=True)
        shown = []
        for n, (p, crop, prov) in enumerate(items[:KEEP]):
            f = a.out / c / f"{n:02d}.jpg"
            crop.save(f, quality=92)
            meta.append({"file": str(f.relative_to(a.out)), "family": "person", "class": c, "noun": "person", "rank": n, "p": round(p, 3), **prov})
            shown.append((p, crop))
        sheet(c, shown)
    # sleeping people: whole-person crops, ranked by SigLIP
    asleep = []
    for m in web:
        if m["class"] == "asleep":
            asleep.append((a.web / m["file"], {"source": "Wikimedia Commons", "title": m["title"], "page": m["page"],
                                               "artist": strip(m["artist"]), "license": m["license"]}))
    for m in json.loads((a.old / "emotion" / "candidates.json").read_text()):
        if m["class"] == "sleepy":
            asleep.append((a.old / m["file"], {k: v for k, v in m.items() if k not in ("file", "class", "family", "noun")}))
    scored = []
    for path, prov in asleep:
        img = ImageOps.fit(Image.open(path).convert("RGB"), (384, 384), Image.LANCZOS)
        with torch.no_grad():
            x = sp(text=["a photo of a sleeping person", "a photo of an awake person", "a photo with no person"],
                   images=[img], padding="max_length", return_tensors="pt")
            pr = sm(**x).logits_per_image.softmax(-1)[0]
        scored.append((float(pr[0]), img, prov))
    scored.sort(key=lambda t: -t[0])
    (a.out / "asleep").mkdir(parents=True, exist_ok=True)
    shown = []
    for n, (p, img, prov) in enumerate(scored[:KEEP]):
        f = a.out / "asleep" / f"{n:02d}.jpg"
        img.save(f, quality=92)
        meta.append({"file": str(f.relative_to(a.out)), "family": "person", "class": "asleep", "noun": "person", "rank": n, "p": round(p, 3), **prov})
        shown.append((p, img))
    sheet("asleep", shown)
    (a.out / "candidates.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print({c: sum(m["class"] == c for m in meta) for c in list(TARGET) + ["asleep"]})


if __name__ == "__main__":
    main()

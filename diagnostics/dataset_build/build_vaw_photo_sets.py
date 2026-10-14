"""Build real-photo Act2Answer asset sets for the K -> G -> A hypothesis test from VAW labels on
Visual Genome photos (images CC BY 4.0; VAW annotations: Adobe Research License, non-commercial
research). One asset per category: photo_state_v1, photo_attribute_v1, photo_emotion_v1.

Design (same protections as semantic_repeated_v2, see HYPOTHESIS_PILOT.md):
  - each category = antonym pairs on ONE object (e.g. open door vs closed door), so both tiles
    show the same kind of object and only the state/material/expression differs;
  - each label has 5 knowledge questions that never contain the label word;
  - question j of label A shows photos (A_j, B_j); question j of label B shows (B_j, A_j):
    every photo pair occurs once with each photo as the answer, so the photos alone never
    reveal K; `image_key` records the photo pair (for image-pair cross-validation folds);
  - photos: VAW instances whose positive attributes contain the label and not its antonym,
    cropped from Visual Genome, then filtered by SigLIP (OpenVLA's vision-encoder family):
    kept only if zero-shot correct between the two labels with p >= 0.8, best 5 per label.

Run (from this directory, inside the openvla_rl4vla env; needs internet the first time):
  python build_vaw_photo_sets.py --vaw-dir ~/YeahPick/data/vaw --cache-dir ~/YeahPick/data/vg_images
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

from tile_asset import TILE_BBOX, TILE_DENSITY, TILE_SCALES, build_tile

REPO_ROOT = Path(__file__).resolve().parents[2]
CARROT = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets" / "carrot"
OUT = Path(__file__).resolve().parent / "photo_output"
VG_URLS = ("https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K_2/{}.jpg")
# "face" is excluded: VAW tags animal faces (cats, dogs) with expressions too.
PEOPLE = {"man", "woman", "girl", "boy", "person", "child", "lady", "guy", "kid", "men", "women"}

# Object gate: a photo is kept only if SigLIP picks the pair's noun over these look-alikes
# (VAW "door" includes fridge/car doors, "bench" boxes can be mostly table, "person" photos
# can be animals, etc.). Without it the state check alone let wrong objects through.
OBJECT_NEGATIVES = {
    "door": ["refrigerator", "cabinet", "car", "window", "wall"],
    "road": ["person", "bag", "animal", "building", "sidewalk"],
    "toilet": ["sink", "suitcase", "bathtub", "trash can"],
    "plate": ["bowl", "tray", "pizza", "table"],
    "bench": ["table", "pizza", "desk", "chair", "sofa"],
    "bottle": ["vase", "jar", "decanter", "drinking glass", "cup"],
    "chair": ["person", "sofa", "table", "bed"],
    "happy_sad": ["cat", "dog", "animal", "statue"],
    "angry_calm": ["cat", "dog", "animal", "statue"],
    "surprised_sleepy": ["cat", "dog", "animal", "statue"],
}
MIN_OBJECT_PROB = 0.5

# Human visual review of the contact sheets (2026-10-02): VAW instance ids rejected because the
# photo does not show the labelled object/state clearly.
REJECTED_INSTANCES = {
    2382780006,  # "empty plate": metal tray with curry on it
    498271000,   # "fabric chair": black chair, material unclear
    2388593004,  # "fabric chair": cat under a blanket, chair barely visible
    2326692002,  # "fabric chair": cat with a bag on its head
    2414015000,  # "fabric chair": mostly a person, mesh chair
    2389462002,  # "leather chair": blue office chair, material unclear
}

# Per-asset screening overrides. Faces are small in Visual Genome scene photos, so emotion needs a
# larger pool with smaller crops and a lower state-probability threshold (object gate unchanged).
CATEGORY_SETTINGS = {"photo_emotion_v1": {"min_side": 60, "candidates": 300, "min_prob": 0.6}}

# category -> list of pairs: (pair id, object names, noun shown in labels, (attr A, questions A), (attr B, questions B))
CATEGORIES = {
    "photo_state_v1": [
        ("door", {"door"}, "door",
         ("open", ["Which door could a person walk straight through without touching it?",
                   "Which door lets a breeze into the room?",
                   "Which door shows the space on the other side?",
                   "Which door could a cat stroll through right now?",
                   "Which door does not need to be pushed before entering?"]),
         ("closed", ["Which door keeps the room private?",
                     "Which door stops a dog from leaving the room?",
                     "Which door must be pushed or pulled before you can pass?",
                     "Which door blocks the view of the other side?",
                     "Which door keeps cold air outside?"])),
        ("road", {"road", "street", "pavement"}, "road",
         ("wet", ["Which road would be slippery for cars?",
                  "Which road shows that it just rained?",
                  "Which road would splash your shoes?",
                  "Which road reflects light like a mirror?",
                  "Which road needs drivers to brake earlier?"]),
         ("dry", ["Which road has had no rain for days?",
                  "Which road gives a bicycle the best grip?",
                  "Which road would keep chalk drawings from washing away?",
                  "Which road would leave no puddle marks on your shoes?",
                  "Which road would raise dust when a car passes?"])),
        ("toilet", {"toilet"}, "toilet",
         ("clean", ["Which toilet has just been scrubbed?",
                    "Which toilet would a guest be happy to use?",
                    "Which toilet would pass a hygiene inspection?",
                    "Which toilet has no stains?",
                    "Which toilet looks freshly disinfected?"]),
         ("dirty", ["Which toilet needs scrubbing?",
                    "Which toilet would fail a hygiene inspection?",
                    "Which toilet has been neglected for weeks?",
                    "Which toilet would a guest refuse to use?",
                    "Which toilet carries more germs?"])),
        ("plate", {"plate"}, "plate",
         ("empty", ["Which plate is ready to be washed after the meal?",
                    "Which plate has nothing left to eat on it?",
                    "Which plate belongs to someone who finished dinner?",
                    "Which plate could have new food served onto it right away?",
                    "Which plate would be lighter to carry?"]),
         ("full", ["Which plate belongs to someone who is about to start eating?",
                   "Which plate has a meal on it?",
                   "Which plate would be heavier to carry?",
                   "Which plate was just served by the waiter?",
                   "Which plate still has food that could be eaten?"])),
    ],
    "photo_attribute_v1": [
        ("bench", {"bench"}, "bench",
         ("wooden", ["Which bench was made from a tree?",
                     "Which bench could be damaged by termites?",
                     "Which bench could be sanded and varnished?",
                     "Which bench would burn in a campfire?",
                     "Which bench shows natural grain and knots?"]),
         ("metal", ["Which bench could rust in the rain?",
                    "Which bench would a magnet stick to?",
                    "Which bench gets very hot in the summer sun?",
                    "Which bench was put together by welding?",
                    "Which bench would ring when struck with a spoon?"])),
        ("bottle", {"bottle"}, "bottle",
         ("glass", ["Which bottle would shatter if dropped on a tile floor?",
                    "Which bottle is made from melted sand?",
                    "Which bottle could cut your hand if it broke?",
                    "Which bottle is heavier and more fragile?",
                    "Which bottle is traditionally used for fine wine?"]),
         ("plastic", ["Which bottle could be squeezed in your hand?",
                      "Which bottle would bounce if dropped?",
                      "Which bottle is made from petroleum?",
                      "Which bottle is light and shatterproof for hiking?",
                      "Which bottle could melt near a flame?"])),
        ("chair", {"chair"}, "chair",
         ("leather", ["Which chair is made from animal hide?",
                      "Which chair would a vegan avoid?",
                      "Which chair needs conditioning cream to stop cracking?",
                      "Which chair is covered in the same material as a biker jacket?",
                      "Which chair smells like a horse saddle?"]),
         ("fabric", ["Which chair is woven from threads?",
                     "Which chair would soak up a spilled drink?",
                     "Which chair is covered in the same material as a T-shirt?",
                     "Which chair would pill and fray over time?",
                     "Which chair could be cleaned in a washing machine cover?"])),
    ],
    "photo_emotion_v1": [
        ("happy_sad", PEOPLE, "person",
         ("happy", ["Which person just received good news?",
                    "Which person is enjoying a birthday party?",
                    "Which person just won a prize?",
                    "Which person is having a great day?",
                    "Which person just heard a funny joke?"]),
         ("sad", ["Which person just lost their pet?",
                  "Which person missed the last bus home?",
                  "Which person is grieving?",
                  "Which person just heard bad news?",
                  "Which person is having a terrible day?"])),
        ("angry_calm", PEOPLE, "person",
         ("angry", ["Which person just had their car scratched?",
                    "Which person is about to shout at someone?",
                    "Which person was just cut off in traffic?",
                    "Which person had their lunch stolen?",
                    "Which person is losing their temper?"]),
         ("calm", ["Which person is relaxing on vacation?",
                   "Which person has nothing to worry about?",
                   "Which person is resting on a quiet afternoon?",
                   "Which person is unbothered by the noise around them?",
                   "Which person just finished a meditation session?"])),
        ("surprised_sleepy", PEOPLE, "person",
         ("surprised", ["Which person just saw something unexpected?",
                        "Which person walked into a party nobody told them about?",
                        "Which person just watched a magic trick?",
                        "Which person cannot believe their eyes?",
                        "Which person just heard shocking news?"]),
         ("sleepy", ["Which person stayed up all night?",
                     "Which person needs a nap?",
                     "Which person is about to doze off?",
                     "Which person got very little rest last night?",
                     "Which person can barely keep their eyes open?"])),
    ],
}
N_QUESTIONS = 5


def load_vaw(vaw_dir: Path) -> list[dict]:
    data = []
    for f in ("train_part1.json", "train_part2.json", "val.json", "test.json"):
        data += json.loads((vaw_dir / f).read_text())
    return data


def fetch_image(image_id: int, cache: Path) -> Image.Image | None:
    path = cache / f"{image_id}.jpg"
    if not path.exists():
        for url in VG_URLS:
            try:
                with urllib.request.urlopen(url.format(image_id), timeout=30) as r:
                    path.write_bytes(r.read())
                break
            except Exception:
                continue
        else:
            return None
    return Image.open(path).convert("RGB")


def square_crop(img: Image.Image, bbox, margin=0.1, size=512) -> Image.Image:
    x, y, w, h = bbox
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    canvas.paste(img, (-box[0], -box[1]))
    return canvas.resize((size, size), Image.LANCZOS)


def candidates(data, objects, attr, antonym, min_side, limit):
    seen, out = set(), []
    rows = [d for d in data if d["object_name"] in objects and attr in d["positive_attributes"]
            and antonym not in d["positive_attributes"] and min(d["instance_bbox"][2:]) >= min_side]
    for d in sorted(rows, key=lambda d: -d["instance_bbox"][2] * d["instance_bbox"][3]):
        if d["image_id"] not in seen:
            seen.add(d["image_id"])
            out.append(d)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vaw-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--min-side", type=int, default=100)
    ap.add_argument("--candidates", type=int, default=120)
    ap.add_argument("--min-prob", type=float, default=0.8)
    ap.add_argument("--siglip", default="google/siglip-so400m-patch14-224")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_vaw(args.vaw_dir)
    processor = AutoProcessor.from_pretrained(args.siglip)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.siglip).eval().to(device)

    def siglip_probs(images, texts):
        with torch.no_grad():
            inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt").to(device)
            return model(**inputs).logits_per_image.softmax(-1).cpu()

    failed = []
    for asset_name in args.categories:
        try:
            build_category(asset_name, data, args, siglip_probs)
        except Exception as e:  # one failing category must not stop the others
            failed.append(asset_name)
            print(f"[{asset_name}] FAILED: {e}")
    if failed:
        raise SystemExit(f"categories failed: {failed}")


def build_category(asset_name, data, args, siglip_probs) -> None:
    asset = CARROT / asset_name
    if asset.exists():
        shutil.rmtree(asset)  # no stale tiles from an earlier build
    pairs, model_db, credits, audit_rows = [], {}, [], []
    sheet_rows = []
    qid = 0
    for pair_id, objects, noun, (attr_a, qs_a), (attr_b, qs_b) in CATEGORIES[asset_name]:
        labels = (f"{attr_a} {noun}", f"{attr_b} {noun}")
        chosen = {}
        for side, (attr, antonym) in enumerate(((attr_a, attr_b), (attr_b, attr_a))):
            scored = []
            cfg = {"min_side": args.min_side, "candidates": args.candidates, "min_prob": args.min_prob,
                   **CATEGORY_SETTINGS.get(asset_name, {})}
            for d in candidates(data, objects, attr, antonym, cfg["min_side"], cfg["candidates"]):
                if int(d["instance_id"]) in REJECTED_INSTANCES:  # VAW stores ids as strings
                    continue
                img = fetch_image(d["image_id"], args.cache_dir)
                if img is None:
                    continue
                crop = square_crop(img, d["instance_bbox"])
                p = float(siglip_probs([crop], [f"a photo of a {l}" for l in labels])[0, side])
                if p < cfg["min_prob"]:
                    continue
                nouns = [noun] + OBJECT_NEGATIVES[pair_id]
                obj = siglip_probs([crop], [f"a photo of a {n}" for n in nouns])[0]
                if int(obj.argmax()) == 0 and float(obj[0]) >= MIN_OBJECT_PROB:
                    scored.append((p, d, crop))
            scored.sort(key=lambda t: -t[0])
            chosen[side] = scored[:N_QUESTIONS]
            audit_rows.append({"asset": asset_name, "label": labels[side],
                               "candidates_passing": len(scored), "kept": len(chosen[side])})
        if min(len(chosen[0]), len(chosen[1])) < N_QUESTIONS:
            print(f"[{asset_name}] DROP pair {pair_id}: only {len(chosen[0])}/{len(chosen[1])} photos passed SigLIP")
            continue

        tiles = {}
        for side in (0, 1):
            for j, (p, d, crop) in enumerate(chosen[side]):
                name = f"{pair_id}_{(attr_a, attr_b)[side]}_{j}"
                build_tile(crop, asset / "shapes" / name)
                model_db[name] = {"name": name, "sign": name, "bbox": TILE_BBOX,
                                  "scales": TILE_SCALES, "density": TILE_DENSITY}
                tiles[(side, j)] = name
                credits.append({"tile": name, "label": labels[side], "vg_image_id": d["image_id"],
                                "vaw_instance_id": d["instance_id"], "bbox": d["instance_bbox"],
                                "siglip_p": round(p, 3), "image_license": "CC BY 4.0 (Visual Genome)",
                                "source_url": VG_URLS[0].format(d["image_id"])})
                sheet_rows.append((labels[side], crop))

        for side, questions in ((0, qs_a), (1, qs_b)):
            other = 1 - side
            for j, question in enumerate(questions):
                answer_side = "Left" if qid % 2 == 0 else "Right"
                correct, distractor = tiles[(side, j)], tiles[(other, j)]
                left, right = (correct, distractor) if answer_side == "Left" else (distractor, correct)
                left_label, right_label = ((labels[side], labels[other]) if answer_side == "Left"
                                           else (labels[other], labels[side]))
                pairs.append({
                    "index": qid, "question_id": qid, "question": question,
                    "knowledge_instruction": f"Place the cube on the answer to this question: {question}",
                    "left": left, "right": right, "answer": answer_side,
                    "semantic_answer": labels[side], "semantic_answer_id": labels[side].replace(" ", "_"),
                    "distractor_id": labels[other].replace(" ", "_"),
                    "left_label": left_label, "right_label": right_label,
                    "knowledge_category": asset_name.split("_")[1],
                    "template_family": pair_id, "relation_type": f"photo_pair_{pair_id}_{j}",
                    "image_key": f"{pair_id}_{j}",
                })
                qid += 1

    # ---- audit ----
    leaks = [p["question_id"] for p in pairs
             if set(re.findall(r"[a-z]+", p["semantic_answer"].split()[0].lower()))
             & set(re.findall(r"[a-z]+", p["question"].lower()))]
    image_pair_answers = {}
    for p in pairs:
        image_pair_answers.setdefault(p["image_key"], set()).add(p["semantic_answer_id"])
    revealing = sum(len(v) == 1 for v in image_pair_answers.values())
    audit = {"asset": asset_name, "questions": len(pairs),
             "answer_frequency": dict(Counter(p["semantic_answer_id"] for p in pairs)),
             "side_frequency": dict(Counter(p["answer"] for p in pairs)),
             "answer_word_leak_question_ids": leaks,
             "photo_pairs": len(image_pair_answers), "photo_pairs_revealing_answer": revealing,
             "photos": audit_rows}
    if not pairs or leaks or revealing:
        raise ValueError(f"audit failed: {json.dumps(audit, indent=2)}")

    asset.mkdir(parents=True, exist_ok=True)
    (asset / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (asset / "model_db.json").write_text(json.dumps(model_db, indent=2) + "\n")
    (OUT / f"{asset_name}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with open(OUT / f"{asset_name}_credits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(credits[0]))
        w.writeheader()
        w.writerows(credits)
    with open(OUT / f"{asset_name}_questions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question", "semantic_answer", "distractor_id", "answer", "image_key"])
        w.writeheader()
        w.writerows({k: p[k] for k in w.fieldnames} for p in pairs)

    # contact sheet: one row per label, its 5 photos
    by_label = {}
    for label, crop in sheet_rows:
        by_label.setdefault(label, []).append(crop)
    sheet = Image.new("RGB", (160 + N_QUESTIONS * 140, 140 * len(by_label)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, crops) in enumerate(by_label.items()):
        draw.text((5, r * 140 + 60), label, fill="black")
        for c, crop in enumerate(crops):
            sheet.paste(crop.resize((130, 130)), (160 + c * 140, r * 140 + 5))
    sheet.save(OUT / f"{asset_name}_contact_sheet.png")
    print(f"[{asset_name}] {len(pairs)} questions, {len(model_db)} photo tiles -> {asset}")


if __name__ == "__main__":
    main()

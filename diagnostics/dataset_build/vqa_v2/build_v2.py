"""Build the v2 photo VQA dataset (state, material, emotion) from reviewed candidate crops.

Pairs: both tiles are the same kind of object (or both people) and differ only in the tested property.
Families are visited round-robin so every object family is represented; a crop is used in at most
MAX_USE pairs, never twice with the same partner. Per pair, one fact per tile's class (vqa_v2/questions.py);
group of a tile = Object recognition, Attribute recognition, Semantic reasoning, Visual grounding (same fact),
Counterfactual grounding (fact of the other tile's class -> the other side).

Run from this directory:  python build_v2.py --cand <candidate dir> --out ../../../VQA_dataset_v2
"""
import argparse, base64, csv, io, itertools, json, random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

from questions import EMO_Q, EMOTION, EMOTION_KDEF, MAT_Q, MATERIAL, MATERIAL_WORD, SEAT_OBJ_Q, STATE

PAIR_PREFIX = {"state": "state-r2"}
OUT_NAME = {"material": "attribute"}  # the team calls this category "attribute"
# object recognition names the kind of thing asked for, so other things in the photo (food next to a spoon,
# a person on a chair) are not also correct answers
OBJ_KIND = {"chair": "piece of furniture", "bench": "piece of furniture", "couch": "piece of furniture", "bed": "piece of furniture",
            "table": "piece of furniture", "cup": "drink container", "bottle": "drink container", "wine glass": "drink container",
            "bowl": "piece of tableware", "plate": "piece of tableware", "fork": "utensil", "spoon": "utensil", "knife": "utensil",
            "vase": "piece of home decor", "sink": "kind of basin", "toilet": "bathroom fixture", "laptop": "electronic device",
            "cell phone": "electronic device", "monitor": "electronic device", "phone": "electronic device", "TV": "electronic device",
            "microwave": "appliance", "oven": "appliance", "handbag": "kind of bag", "suitcase": "kind of luggage",
            "fire hydrant": "street object", "stop sign": "street object", "motorcycle": "vehicle", "boat": "vehicle",
            "bear": "animal", "elephant": "animal", "sheep": "animal", "apple": "food", "orange": "food", "banana": "food",
            "cake": "food", "pizza": "food", "sandwich": "food", "donut": "food", "carrot": "food", "broccoli": "food"}  # new ids after the 2026-11-09 review, so old reviews do not attach to new pairs
SEATS = {"bench_empty", "chair_empty", "couch_empty"}  # a person on the seat: ask for the furniture

HERE = Path(__file__).resolve().parent
N_PAIRS = 100
MAX_USE = {"state": 2, "material": 3, "emotion": 6, "emotion_kdef": 8}  # material and sleepy faces have fewer clean photos
SIDES = ("left", "right")


def load(cand, review):
    """category -> family -> class -> [crop records] (deduplicated by source object)."""
    out = {}
    for cat in ("state", "material", "emotion", "emotion_kdef"):
        if not (cand / cat / "candidates.json").exists() or cat not in review:
            continue
        meta = json.loads((cand / cat / "candidates.json").read_text())
        rows = defaultdict(list)
        for m in meta:
            rows[(m["family"], m["class"])].append(m)
        seen, fams = set(), defaultdict(dict)
        excluded = {str(x) for x in review.get("exclude", {}).get(cat, [])}
        for fam, classes in review.get(cat, {}).items():
            for cls, idx in classes.items():
                keep = []
                for i in idx:
                    m = rows[(fam, cls)][i]
                    key = m.get("ovad_ann_id") or m.get("title") or m.get("instance_id") or m["file"]
                    if key in seen or str(m.get("ovad_ann_id") or m.get("instance_id") or m.get("title")) in excluded:
                        continue
                    seen.add(key)
                    keep.append(m)
                if keep:
                    fams[fam][cls] = keep
        out[cat] = {f: c for f, c in fams.items() if len(c) >= 2}
    return out


def facts_for(cat, fam, a, b):
    """(facts for class a usable against b, facts for b usable against a, noun, rel, attr question, labels)."""
    if cat == "state":
        noun, q, cl = STATE[fam]
        return cl[a][2], cl[b][2], noun, "that", q, {c: cl[c][0] for c in (a, b)}, {c: cl[c][1] for c in (a, b)}
    if cat == "material":
        noun = fam.replace("_", " ")
        fa = [f for f, false_for in MATERIAL[a] if b in false_for]
        fb = [f for f, false_for in MATERIAL[b] if a in false_for]
        lab = {c: f"{MATERIAL_WORD[c]} {noun}" for c in (a, b)}
        return fa, fb, noun, "that", MAT_Q, lab, {a: a, b: b}
    E = EMOTION_KDEF if cat == "emotion_kdef" else EMOTION
    return E[a][1], E[b][1], "person", "who", EMO_Q, {c: E[c][0] for c in (a, b)}, {a: a, b: b}


OBJ_REL = ("someone ", "nobody ", "a ", "an ", "you ", "the ")


def which(noun, rel, fact):
    """'Which bench was made from a tree?' but 'Which bench is the one that a tired walker could sit on?'"""
    return f"Which {noun} is the one {rel} {fact}?" if fact.startswith(OBJ_REL) else f"Which {noun} {fact}?"


def plan_pairs(cat, fams, rng):
    """Round-robin over (family, class-pair) units; returns [(fam, (cls_a, crop_a), (cls_b, crop_b))]."""
    units = []
    for fam, classes in sorted(fams.items()):
        for a, b in itertools.combinations(sorted(classes), 2):
            fa, fb = facts_for(cat, fam, a, b)[:2]
            if fa and fb:
                combos = [(x, y) for x in classes[a] for y in classes[b]]
                rng.shuffle(combos)
                units.append([fam, a, b, combos])
    rng.shuffle(units)
    use, pairs, done = Counter(), [], set()
    while len(pairs) < N_PAIRS:
        progressed = False
        for fam, a, b, combos in units:
            ok = [(x, y) for x, y in combos if use[x["file"]] < MAX_USE[cat] and use[y["file"]] < MAX_USE[cat]
                  and (x["file"], y["file"]) not in done]
            if not ok:
                continue
            x, y = min(ok, key=lambda xy: use[xy[0]["file"]] + use[xy[1]["file"]])
            pairs.append((fam, (a, x), (b, y)))
            use[x["file"]] += 1
            use[y["file"]] += 1
            done.add((x["file"], y["file"]))
            progressed = True
            if len(pairs) == N_PAIRS:
                break
        if not progressed:
            break
    return pairs


def build(cat, fams, rng, out):
    pairs = plan_pairs(cat, fams, rng)
    oc = OUT_NAME.get(cat, cat)
    (out / oc / "images").mkdir(parents=True, exist_ok=True)
    img_id, images = {}, []

    def image(m, cls, label):
        if m["file"] not in img_id:
            iid = f"{oc}_{len(img_id) + 1:03d}"
            img_id[m["file"]] = iid
            Image.open(CAND / m["file"]).convert("RGB").save(out / oc / "images" / f"{iid}.jpg", quality=92)
            src = {("source_image_id" if k == "image_id" else k): v for k, v in m.items() if k not in ("file", "family", "class")}
            images.append({**src, "image_id": iid, "path": f"{oc}/images/{iid}.jpg", "family": m["family"], "class": cls,
                           "label": label})
        return img_id[m["file"]]

    cycle = defaultdict(lambda: itertools.count())
    records = []
    for k, (fam, (a, x), (b, y)) in enumerate(pairs):
        fa, fb, noun, rel, aq, labels, attr = facts_for(cat, fam, a, b)
        fact = {a: fa[next(cycle[(fam, a, b, a)]) % len(fa)], b: fb[next(cycle[(fam, a, b, b)]) % len(fb)]}
        src_cls = a if k % 2 == 0 else b
        src_side = SIDES[(k // 2) % 2]  # source class alternates every pair, side every two pairs
        crop = {a: x, b: y}
        other = b if src_cls == a else a
        tiles = {src_side: src_cls, ("right" if src_side == "left" else "left"): other}
        tile_img = {s: image(crop[c], c, labels[c]) for s, c in tiles.items()}
        groups = {}
        for focus in SIDES:
            me, you = tiles[focus], tiles["right" if focus == "left" else "left"]
            groups[focus] = [
                ["Object recognition", (f"What {OBJ_KIND[noun]} is shown on the {focus} tile?" if noun in OBJ_KIND
                                        else f"What is shown on the {focus} tile?"), noun],
                ["Attribute recognition", aq.format(noun=noun, side=focus), attr[me]],
                ["Semantic reasoning", which(noun, rel, fact[me]), labels[me]],
                ["Visual grounding", f"Which side contains the {noun} {rel} {fact[me]}?", focus.capitalize()],
                ["Counterfactual grounding", f"Which side contains the {noun} {rel} {fact[you]}?",
                 ("right" if focus == "left" else "left").capitalize()],
            ]
        records.append({"pair_id": f"{PAIR_PREFIX.get(cat, oc)}_{k:03d}", "category": oc, "family": fam,
                        "left": {"image_id": tile_img["left"], "class": tiles["left"], "label": labels[tiles["left"]]},
                        "right": {"image_id": tile_img["right"], "class": tiles["right"], "label": labels[tiles["right"]]},
                        "instruction": f"Place the cube on the {noun} {rel} {fact[src_cls]}.",
                        "answer": src_side.capitalize(), "groups": groups})
    return records, images


def to_jsonl_groups(r):
    for focus in SIDES:
        yield {"group_id": f"{r['pair_id']}_{focus}", "category": r["category"], "pair_id": r["pair_id"],
               "focus_tile": focus, "focus_image_id": r[focus]["image_id"],
               "images": {s: {"image_id": r[s]["image_id"], "path": f"{r['category']}/images/{r[s]['image_id']}.jpg"} for s in SIDES},
               "labels": {s: r[s]["label"] for s in SIDES},
               "knowledge": {"instruction": r["instruction"], "expected_answer": r["answer"], "template_family": r["family"]},
               "vqa": [{"type": t, "question": q, "expected_answer": a} for t, q, a in r["groups"][focus]]}


def main():
    global CAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review", type=Path, default=HERE / "review.json")
    ap.add_argument("--viewer-data", type=Path, default=None, help="also write the editor's data.js here")
    a = ap.parse_args()
    CAND = a.cand
    fams = load(a.cand, json.loads(a.review.read_text()))
    all_records, audit, thumbs = [], {}, {}
    for cat in ("state", "material", "emotion", "emotion_kdef"):
        if not fams.get(cat):
            continue
        recs, images = build(cat, fams[cat], random.Random(f"v2-{cat}"), a.out)
        oc = OUT_NAME.get(cat, cat)
        with open(a.out / f"{oc}_vqa.jsonl", "w") as f:
            for r in recs:
                for g in to_jsonl_groups(r):
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
        with open(a.out / oc / "images.csv", "w", newline="") as f:
            keys = sorted({k for im in images for k in im})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(images)
        audit[oc] = {"pairs": len(recs), "images": len(images), "families": len({r["family"] for r in recs}),
                      "pairs_per_family": dict(Counter(r["family"] for r in recs)),
                      "answer_side": dict(Counter(r["answer"] for r in recs))}
        print(oc, json.dumps(audit[oc]))
        all_records += recs
        for im in images:
            t = Image.open(a.out / im["path"]).convert("RGB")
            t.thumbnail((256, 256))
            b = io.BytesIO()
            t.save(b, "JPEG", quality=80)
            thumbs[im["image_id"]] = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1))
    (a.out / "pairs.json").write_text(json.dumps(all_records, indent=1, ensure_ascii=False))
    if a.viewer_data:
        a.viewer_data.write_text("window.VQA_DATA = " + json.dumps({"pairs": all_records, "images": thumbs},
                                                                   separators=(",", ":"), ensure_ascii=False) + ";\n")


if __name__ == "__main__":
    main()

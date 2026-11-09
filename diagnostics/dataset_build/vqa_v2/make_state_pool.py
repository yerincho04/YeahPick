"""Turn state_pool.json into the state candidate set (cand/state/candidates.json) and the matching review entry.
v2 images are copied from VQA_dataset_v2; Commons photos are centre-cropped to a 384 px square and keep their
title, page, author and licence. The OVAD-only candidate list is kept as candidates_ovad.json."""
import csv, json, re, shutil, sys
from pathlib import Path
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
cand, web, v2 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pool = json.loads((HERE / "state_pool.json").read_text())
v2meta = {r["image_id"]: r for r in csv.DictReader(open(v2 / "state" / "images.csv"))}
webm = json.loads((web / "candidates.json").read_text())
webby = {}
for m in webm:
    webby.setdefault((m["family"], m["class"]), []).append(m)
src = cand / "state" / "candidates.json"
if not (cand / "state" / "candidates_ovad.json").exists():
    shutil.copy(src, cand / "state" / "candidates_ovad.json")
out_dir = cand / "state" / "pool"
out_dir.mkdir(parents=True, exist_ok=True)
meta, review = [], {}
strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
for fam, classes in pool.items():
    if fam.startswith("_"):
        continue
    for cls, srcs in classes.items():
        for iid in srcs.get("v2", []):
            r = v2meta[iid]
            f = out_dir / f"{fam}_{cls}_{iid}.jpg"
            shutil.copy(v2 / r["path"], f)
            meta.append({"file": str(f.relative_to(cand)), "family": fam, "class": cls, "source": "OVAD (COCO val2017)",
                         "ovad_ann_id": r.get("ovad_ann_id"), "coco_image_id": r.get("coco_image_id"),
                         "coco_file": r.get("coco_file"), "bbox": r.get("bbox"), "license": f"COCO license id {r.get('coco_license')}"})
        for k in srcs.get("web", []):
            src_cls, k = (k.split(":")[0], int(k.split(":")[1])) if isinstance(k, str) else (cls, k)  # "open:12" = photo found by the other class's search
            w = webby[(fam, src_cls)][k]
            f = out_dir / f"{fam}_{cls}_web{src_cls}{k:02d}.jpg"
            ImageOps.fit(Image.open(web / w["file"]).convert("RGB"), (384, 384), Image.LANCZOS).save(f, quality=92)
            meta.append({"file": str(f.relative_to(cand)), "family": fam, "class": cls, "source": "Wikimedia Commons",
                         "title": w["title"], "page": w["page"], "artist": strip(w["artist"]), "license": w["license"],
                         "license_url": w.get("license_url", "")})
for i, m in enumerate(meta):
    review.setdefault(m["family"], {}).setdefault(m["class"], [])
idx = {}
for m in meta:
    k = (m["family"], m["class"])
    review[m["family"]][m["class"]].append(idx.get(k, 0))
    idx[k] = idx.get(k, 0) + 1
src.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
rv = json.loads((HERE / "review.json").read_text())
rv["state"] = review
(HERE / "review.json").write_text(json.dumps(rv, indent=1))
print(len(meta), "state photos,", len(review), "families,", sum(m["source"] == "Wikimedia Commons" for m in meta), "from Commons")

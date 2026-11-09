"""Append hand-picked Wikimedia Commons faces (web_emo) to the emotion candidates and the review whitelist."""
import json, re, sys
from pathlib import Path
from PIL import Image, ImageOps
cand, web = Path(sys.argv[1]), Path(sys.argv[2])
PICK = {"surprised": [4, 7], "sleepy": [13, 19], "neutral": [0, 11], "upset": [12, 23], "happy": [20, 21, 29, 43]}
wm = json.loads((web / "candidates.json").read_text())
by = {}
for m in wm: by.setdefault(m["class"], []).append(m)
meta = json.loads((cand / "emotion" / "candidates.json").read_text())
meta = [m for m in meta if m.get("source") != "Wikimedia Commons"]
rv = json.loads((Path(__file__).parent / "review.json").read_text())
strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
for cls, idx in PICK.items():
    for k in idx:
        w = by[cls][k]
        f = cand / "emotion" / cls / f"web_{k:02d}.jpg"
        ImageOps.fit(Image.open(web / w["file"]).convert("RGB"), (384, 384), Image.LANCZOS, centering=(0.5, 0.35)).save(f, quality=92)
        meta.append({"file": str(f.relative_to(cand)), "family": "person", "class": cls, "noun": "person",
                     "source": "Wikimedia Commons", "title": w["title"], "page": w["page"], "artist": strip(w["artist"]),
                     "license": w["license"], "license_url": w.get("license_url", "")})
(cand / "emotion" / "candidates.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
rows = {}
for i, m in enumerate(meta): rows.setdefault(m["class"], []).append(i)
for cls in PICK:
    pos = [j for j, i in enumerate(rows[cls]) if meta[i].get("source") == "Wikimedia Commons"]
    keep = [j for j in rv["emotion"]["person"][cls] if j < len(rows[cls]) and meta[rows[cls][j]].get("source") != "Wikimedia Commons"]
    rv["emotion"]["person"][cls] = keep + pos
(Path(__file__).parent / "review.json").write_text(json.dumps(rv, indent=1))
print({c: len(v) for c, v in rv["emotion"]["person"].items()})

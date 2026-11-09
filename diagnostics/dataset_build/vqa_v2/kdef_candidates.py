"""KDEF-cropped faces (data/KDEF-cropped, background removed) -> candidates for the emotion_kdef set.
File code: session A/B, F/M, actor nn, expression (HA happy, AN angry, SA sad, DI disgusted, NE neutral,
AF afraid, SU surprised), angle (S, HL, HR, FL, FR). Faces are composited on mid-gray and padded to a 384 px square.
KDEF terms: free for non-commercial research; no redistribution without written consent from Karolinska Institutet."""
import json, re, sys
from pathlib import Path
from PIL import Image

EXPR = {"HA": "happy", "AN": "angry", "SA": "sad", "DI": "disgusted", "NE": "neutral"}
src, cand, review = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = cand / "emotion_kdef"
(out / "faces").mkdir(parents=True, exist_ok=True)
meta = []
for f in sorted(src.glob("*.png")):
    m = re.match(r"([AB])([FM])(\d\d)([A-Z]{2})([A-Z]+)\.png", f.name)
    if not m or m[4] not in EXPR:
        continue
    im = Image.open(f).convert("RGBA")
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255)); bg.alpha_composite(im)
    side = int(max(im.size) * 1.08)
    sq = Image.new("RGB", (side, side), (128, 128, 128)); sq.paste(bg.convert("RGB"), ((side - im.width) // 2, (side - im.height) // 2))
    dst = out / "faces" / f"{f.stem}.jpg"
    sq.resize((384, 384), Image.LANCZOS).save(dst, quality=93)
    meta.append({"file": str(dst.relative_to(cand)), "family": "person", "class": EXPR[m[4]], "noun": "person",
                 "source": "KDEF (Lundqvist, Flykt & Ohman 1998)", "title": f.name, "kdef_session": m[1],
                 "kdef_actor": m[2] + m[3], "kdef_angle": m[5],
                 "license": "KDEF terms of use: non-commercial research only, no redistribution"})
(out / "candidates.json").write_text(json.dumps(meta, indent=1))
rv = json.loads(review.read_text())
idx = {}
for m in meta:
    idx.setdefault(m["class"], []).append(len(idx.get(m["class"], [])))
rv["emotion_kdef"] = {"person": idx}
review.write_text(json.dumps(rv, indent=1))
print({c: len(v) for c, v in idx.items()})

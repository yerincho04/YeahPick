"""Fetch candidate photos from Wikimedia Commons (freely licensed) for classes that need replacements.

For each (family, class, query) the Commons search API (File namespace, bitmap photos) returns files; for each we
keep the 640 px thumbnail plus license, author and page URL (extmetadata) so the dataset can credit them.
Writes <out>/<family>/<class>/<n>.jpg and <out>/candidates.json. Downloads use curl (Python TLS fails here).
"""
import json, subprocess, sys, urllib.parse
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "YeahPick-VQA-dataset/0.1 (research; contact via github.com/yerincho04/YeahPick)"


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    r = subprocess.run(["curl", "-s", "-A", UA, f"{API}?{q}"], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout or "{}")


def search(query, n):
    d = api({"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
             "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 640})
    out = []
    for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") not in ("image/jpeg", "image/png") or ii.get("width", 0) < 400:
            continue
        md = ii.get("extmetadata", {})
        g = lambda k: (md.get(k) or {}).get("value", "")
        out.append({"title": p["title"], "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"),
                    "license": g("LicenseShortName"), "artist": g("Artist"), "license_url": g("LicenseUrl")})
    return out


def main(spec_path, out):
    out = Path(out)
    spec = json.loads(Path(spec_path).read_text())
    meta = json.loads((out / "candidates.json").read_text()) if (out / "candidates.json").exists() else []
    have = {m["title"] for m in meta}
    for fam, classes in spec.items():
        for cls, queries in classes.items():
            d = out / fam / cls
            d.mkdir(parents=True, exist_ok=True)
            k = len([m for m in meta if m["family"] == fam and m["class"] == cls])
            for q in queries:
                for r in search(q, 14):
                    if r["title"] in have or not r["thumb"]:
                        continue
                    f = d / f"{k:02d}.jpg"
                    subprocess.run(["curl", "-s", "-A", UA, "-o", str(f), r["thumb"]], timeout=60)
                    if f.exists() and f.stat().st_size > 5000:
                        meta.append({"family": fam, "class": cls, "query": q, "file": str(f.relative_to(out)), **r})
                        have.add(r["title"])
                        k += 1
            print(fam, cls, k, flush=True)
    (out / "candidates.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

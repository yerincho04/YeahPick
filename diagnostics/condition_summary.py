"""Success rate per instruction condition (and per question) for one asset set.

Run from the repo root:  python3 diagnostics/condition_summary.py <asset_name> [outputs_dir]
"""
import glob
import json
import sys
from collections import defaultdict

asset = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "outputs"
by_condition, by_question = defaultdict(list), defaultdict(dict)
for path in glob.glob(f"{root}/openvla-{asset}-*/diagnostics/episode_log.jsonl"):
    for line in open(path):
        r = json.loads(line)
        ok = int(r["task_success"])
        by_condition[r["condition"]].append(ok)
        by_question[r["question_id"]][(r["condition"], r["layout"])] = ok

for cond in ("knowledge", "explicit_object", "explicit_spatial"):
    v = by_condition.get(cond, [])
    print(f"{cond:17s} {sum(v):3d}/{len(v):3d}  {sum(v) / max(len(v), 1):.0%}")
print("\nper question: [knowledge noswap,swap | explicit_object noswap,swap | explicit_spatial noswap,swap]")
for q in sorted(by_question):
    print(q, [by_question[q].get((c, l)) for c in ("knowledge", "explicit_object", "explicit_spatial")
              for l in ("noswap", "swap")])

"""Follow-up to text_only_knowledge: which label does the VLA's language head prefer for each question?
If it prefers the same label of a pair regardless of the question, the 50% is a fixed preference, not guessing."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
path = "gen-robot/openvla-7b-rlvla-sft_16k"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
lm = OpenVLAForActionPredictionWithValueHead.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
     trust_remote_code=True, device_map="cuda:0", vh_mode="a0", attn_implementation="sdpa").eval().language_model


@torch.no_grad()
def logprob(prefix, cont):
    p = tok(prefix, return_tensors="pt").input_ids
    full = tok(prefix + cont, return_tensors="pt").input_ids
    n = full.shape[1] - p.shape[1]
    lp = lm(input_ids=full.to("cuda:0")).logits[0].float().log_softmax(-1)
    return float(lp[-n - 1:-1].gather(1, full[0, -n:].to("cuda:0")[:, None]).sum())


picks = defaultdict(Counter)
rows = []
for asset in ("photo_state_v1", "photo_attribute_v1", "photo_emotion_v1"):
    for pr in json.loads((CARROT / asset / "pairs.json").read_text()):
        a, b = pr["semantic_answer"], pr["distractor_id"].replace("_", " ")
        pair = " / ".join(sorted([a, b]))
        pre = f"Question: {pr['question']}\nAnswer:"
        la, lb = logprob(pre, f" the {a}"), logprob(pre, f" the {b}")
        pick = a if la > lb else b
        picks[pair][pick] += 1
        rows.append({"asset": asset, "question": pr["question"], "correct": a, "picked": pick, "margin": round(la - lb, 3)})
Path("outputs/text_only").mkdir(parents=True, exist_ok=True)
Path("outputs/text_only/choices.json").write_text(json.dumps(rows, indent=1))
for pair, c in picks.items():
    print(f"{pair:35s} picked: {dict(c)}")

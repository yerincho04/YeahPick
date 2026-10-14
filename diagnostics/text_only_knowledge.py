"""Text-only knowledge control (weakness #4).

With photos, the K probe ("which of the two tiles is the answer") needs both world knowledge and
recognising the tiles, so "K not readable" cannot say which part failed. This script removes the image:
the question text alone goes through the VLA's own language model (--model picks the VLA:
openvla = Llama-2 7B, magma = Llama-3 8B, spatialvla = Gemma-2 2B, pi0 = Gemma 2B of PaliGemma).

  A. Probe: last-token hidden state at 25/50/75/100% depth, for two prompt formats
       instruction: "In: What action should the robot take to place the cube on the answer to this question: Q?\nOut: "
                    (exactly the robot prompt, minus the image)
       question:    "Question: Q\nAnswer:"
     Linear probe predicts the answer label, 5-fold over question wordings (fold = wording index, so every
     test fold has one unseen wording per label). K2 = pick the right label of its antonym pair (chance 50%).
  B. Direct answer: log P(" the <label A>") vs log P(" the <label B>") after "Question: Q\nAnswer:"
     (the language head of the VLA itself; chance 50%). Plus a few general-knowledge sanity items.

Run on a GPU from the repo root, in that model's conda env:
  python -m diagnostics.text_only_knowledge --model openvla --vla-path gen-robot/openvla-7b-rlvla-sft_16k --out-dir outputs/text_only
  python -m diagnostics.text_only_knowledge --model magma --out-dir outputs/text_only_magma
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
ASSETS = ("photo_state_v1", "photo_attribute_v1", "photo_emotion_v1")
FRACS = (0.25, 0.5, 0.75, 1.0)
SANITY = [  # (question, correct, wrong)
    ("What is the capital of France?", "Paris", "London"),
    ("Which animal produces honey?", "bee", "spider"),
    ("What color is the sky on a clear day?", "blue", "green"),
    ("Which is colder, ice or fire?", "ice", "fire"),
    ("What do you use to cut paper?", "scissors", "spoon"),
]


def binom_p(k: int, n: int) -> float:
    return min(1.0, 2 * sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2**n)


DEFAULT_PATHS = {"openvla": "gen-robot/openvla-7b-rlvla-sft_16k", "magma": "microsoft/Magma-8B",
                 "spatialvla": "IPEC-COMMUNITY/spatialvla-4b-224-pt", "pi0": "juexzz/INTACT-pi0-finetune-bridge"}


def load_language_model(model: str, path: str):
    """(tokenizer, language model, separate LM head or None) of the VLA's LLM, on cuda:0 in bf16."""
    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if model == "openvla":
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            path, device_map="cuda:0", vh_mode="a0", attn_implementation="sdpa", **kw)
        lm = vla.language_model
    elif model == "magma":
        from transformers import AutoModelForCausalLM, AutoProcessor
        tok = AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer
        lm = AutoModelForCausalLM.from_pretrained(path, device_map="cuda:0", attn_implementation="sdpa", **kw).language_model
    elif model == "spatialvla":
        from transformers import AutoModel, AutoProcessor
        tok = AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer
        lm = AutoModel.from_pretrained(path, device_map="cuda:0", **kw).language_model
    elif model == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(path).to("cuda:0", torch.bfloat16)
        paligemma = policy.model.paligemma_with_expert.paligemma
        lm, tok = paligemma.language_model, policy.language_tokenizer
        if not hasattr(lm, "lm_head"):  # newer transformers keep PaliGemma's LM head outside the language model
            return tok, lm.eval(), paligemma.lm_head
    else:
        raise ValueError(model)
    return tok, lm.eval(), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(DEFAULT_PATHS), default="openvla")
    ap.add_argument("--vla-path", default=None, help="checkpoint (default depends on --model)")
    ap.add_argument("--out-dir", default="outputs/text_only")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok, lm, head = load_language_model(args.model, args.vla_path or DEFAULT_PATHS[args.model])
    n_layers = lm.config.num_hidden_layers
    layer_of = {f: min(round(f * n_layers), n_layers) for f in FRACS}

    @torch.no_grad()
    def last_hidden(text: str) -> dict:
        ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")
        hs = lm(input_ids=ids, output_hidden_states=True).hidden_states
        return {f: hs[layer_of[f]][0, -1].float().cpu().numpy() for f in FRACS}

    @torch.no_grad()
    def logprob(prefix: str, continuation: str) -> float:
        p_ids = tok(prefix, return_tensors="pt").input_ids
        full = tok(prefix + continuation, return_tensors="pt").input_ids
        n_cont = full.shape[1] - p_ids.shape[1]
        o = lm(input_ids=full.to("cuda:0"))
        logits = (o.logits if head is None else head(o.last_hidden_state))[0].float().log_softmax(-1)
        targets = full[0, -n_cont:].to("cuda:0")
        return float(logits[-n_cont - 1:-1].gather(1, targets[:, None]).sum())

    records = []
    for asset in ASSETS:
        for p in json.loads((CARROT / asset / "pairs.json").read_text()):
            q = p["question"]
            wording = int(p["image_key"].rsplit("_", 1)[1])  # question j of each label
            records.append({"asset": asset, "qid": p["index"], "question": q, "label": p["semantic_answer"],
                            "distractor": p["distractor_id"].replace("_", " "), "wording": wording,
                            "pair": p["image_key"].rsplit("_", 1)[0],
                            "h_instruction": last_hidden(f"In: What action should the robot take to "
                                                         f"{p['knowledge_instruction'].lower().rstrip('.')}?\nOut: "),
                            "h_question": last_hidden(f"Question: {q}\nAnswer:")})

    # ---- B. direct answer scoring ----
    lines = ["# Text-only knowledge control (no image)", "", "## B. Direct answer by the VLA's language head", "",
             "log P(\" the <correct label>\") vs log P(\" the <other label>\") after \"Question: Q\\nAnswer:\". Chance 50%.", "",
             "| set | correct | p vs chance |", "|---|---|---|"]
    direct = {}
    for asset in ASSETS + ("all",):
        rs = [r for r in records if asset in ("all", r["asset"])]
        k = sum(logprob(f"Question: {r['question']}\nAnswer:", f" the {r['label']}")
                > logprob(f"Question: {r['question']}\nAnswer:", f" the {r['distractor']}") for r in rs)
        direct[asset] = (k, len(rs))
        lines.append(f"| {asset} | {k}/{len(rs)} = {100 * k / len(rs):.0f}% | {binom_p(k, len(rs)):.4f} |")
    sanity = sum(logprob(f"Question: {q}\nAnswer:", f" {a}") > logprob(f"Question: {q}\nAnswer:", f" {b}") for q, a, b in SANITY)
    lines += ["", f"Sanity (general knowledge, same scoring): {sanity}/{len(SANITY)} correct."]

    # ---- A. probes ----
    lines += ["", "## A. Linear probe on the text-only hidden state", "",
              "5-fold over question wordings (each test fold = one unseen wording per label). "
              "K2 = correct label vs its antonym (chance 50%).", "",
              "| prompt | set | layer | K2 accuracy | p vs chance |", "|---|---|---|---|---|"]
    probe_rows = []
    for prompt in ("instruction", "question"):
        for asset in ASSETS + ("all",):
            rs = [r for r in records if asset in ("all", r["asset"])]
            labels = sorted({r["label"] for r in rs})
            y = np.array([labels.index(r["label"]) for r in rs])
            other = np.array([labels.index(r["distractor"]) for r in rs])
            fold = np.array([r["wording"] for r in rs])
            for frac in FRACS:
                X = np.stack([r[f"h_{prompt}"][frac] for r in rs])
                correct = np.zeros(len(rs), bool)
                for f in np.unique(fold):
                    tr, te = fold != f, fold == f
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1)).fit(X[tr], y[tr])
                    lp = np.full((te.sum(), len(labels)), -np.inf)
                    lp[:, clf.classes_] = clf.predict_log_proba(X[te])
                    rows_ = np.arange(te.sum())
                    correct[te] = lp[rows_, y[te]] > lp[rows_, other[te]]
                k, n = int(correct.sum()), len(rs)
                probe_rows.append({"prompt": prompt, "set": asset, "layer": frac, "k": k, "n": n, "p": binom_p(k, n)})
                lines.append(f"| {prompt} | {asset} | {int(frac * 100)}% | {k}/{n} = {100 * k / n:.0f}% | {binom_p(k, n):.4f} |")

    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"direct": direct, "sanity": [sanity, len(SANITY)], "probe": probe_rows}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

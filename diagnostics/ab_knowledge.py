"""K test: does a model know which of two named options answers the instruction? Text only, no image.

Items come from the team's 10-category suite (pairs.json of the installed allmodels_decl_* / categories_all_*
sets); counting and symmetry are excluded because their option labels contain the answer and they have no
text-only version. Each question is asked twice, with the correct option as A and as B:

  A robot must follow this instruction: "<instruction>"
  The two options are:
  A. <option 1>
  B. <option 2>
  Which option should the robot choose? Answer with exactly one letter.

Scoring compares log P("A") and log P("B") for the next token (no generation), so it works for base models,
chat models and VLA language models alike. --format raw appends "\\nAnswer:" (same text for every model);
--format chat wraps the text in the tokenizer's chat template.

Models (--model):
  hf:<repo>              any HF causal LM, e.g. hf:NousResearch/Llama-2-7b-hf, hf:NousResearch/Meta-Llama-3-8B-Instruct
                         (ungated copies of Meta's repos; their safetensors SHA-256 hashes match Meta's exactly)
  magma                  the language model inside microsoft/Magma-8B
  openvla:<repo>         the language model inside an OpenVLA checkpoint (openvla/openvla-7b, gen-robot/...)
  prismatic:<model_id>   the language model of a Prismatic VLM (TRI-ML/prismatic-vlms), e.g. prism-dinosiglip-224px+7b

Writes <out-dir>/<name>.jsonl (one line per question x order) and <name>_summary.json.
Run from the repo root in the matching conda env:
  python -m diagnostics.ab_knowledge --model hf:meta-llama/Llama-2-7b-hf --format raw --name llama2-7b
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
ASSETS = {"allmodels_decl_photo_emotion_v1": "emotion", "allmodels_decl_photo_attribute_v1": "attribute",
          "allmodels_decl_photo_state_v1": "state", "allmodels_decl_public_info_v1": "public_info",
          "allmodels_decl_time_clock_v1": "time_clock", "categories_all_celebrity_50_v1": "celebrity",
          "categories_all_living_world_50_v1": "living_world", "categories_all_traffic_kr_50_v1": "traffic"}
TEMPLATE = ('A robot must follow this instruction: "{instruction}"\nThe two options are:\nA. {a}\nB. {b}\n'
            "Which option should the robot choose? Answer with exactly one letter.")


def items(condition="knowledge"):
    """Text items. condition: knowledge (the knowledge instruction, the default), explicit_object (the robot runs'
    "Place the cube on the <answer>." - the answer is named, so this is a matching control) or explicit_spatial
    ("Place the cube on the <side> tile." with options "left tile"/"right tile" - a matching control)."""
    out = []
    for asset, cat in ASSETS.items():
        for p in json.loads((CARROT / asset / "pairs.json").read_text()):
            left_correct = p["answer"].lower() == "left"
            correct = p["left_label"] if left_correct else p["right_label"]
            wrong = p["right_label"] if left_correct else p["left_label"]
            if condition == "knowledge":
                instruction = p["knowledge_instruction"]
            elif condition == "explicit_object":
                instruction = f"Place the cube on the {p['semantic_answer']}."
            elif condition == "explicit_spatial":
                side = "left" if left_correct else "right"
                instruction = f"Place the cube on the {side} tile."
                correct, wrong = f"{side} tile", ("right" if left_correct else "left") + " tile"
            else:
                raise ValueError(condition)
            for order in ("correct_first", "correct_second"):
                a, b = (correct, wrong) if order == "correct_first" else (wrong, correct)
                out.append({"asset": asset, "category": cat, "question_id": int(p["index"]), "order": order,
                            "instruction": instruction, "option_a": a, "option_b": b,
                            "pair": "|".join(sorted((p["left_label"], p["right_label"]))),  # CV grouping
                            "answer": "A" if order == "correct_first" else "B"})
    return out


def load_model(spec: str):
    """(tokenizer, causal LM with .config and logits output) on cuda:0, bf16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kind, _, ref = spec.partition(":")
    if kind == "hf":
        tok = AutoTokenizer.from_pretrained(ref)
        lm = AutoModelForCausalLM.from_pretrained(ref, device_map="cuda:0", **kw)
    elif kind == "magma":
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True).tokenizer
        lm = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, device_map="cuda:0",
                                                  attn_implementation="sdpa", **kw).language_model
    elif kind == "openvla":
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
        tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
        lm = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            ref, trust_remote_code=True, device_map="cuda:0", vh_mode="a0", attn_implementation="sdpa", **kw).language_model
    elif kind == "prismatic":
        # only the LLM matters here: load Llama-2-7B and overwrite it with the Prismatic-trained LLM weights
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download("TRI-ML/prismatic-vlms", f"{ref}/config.json")))
        assert cfg["model"]["llm_backbone_id"] == "llama2-7b-pure", cfg["model"]["llm_backbone_id"]
        ckpt = hf_hub_download("TRI-ML/prismatic-vlms", f"{ref}/checkpoints/latest-checkpoint.pt")
        tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        lm = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", **kw)
        sd = torch.load(ckpt, map_location="cpu")["model"]["llm_backbone"]
        sd = {k[len("llm."):] if k.startswith("llm.") else k: v for k, v in sd.items()}
        # Prismatic adds a <PAD> token: resize to the checkpoint's vocabulary before loading
        lm.resize_token_embeddings(sd["model.embed_tokens.weight"].shape[0])
        missing, unexpected = lm.load_state_dict(sd, strict=False)
        assert not unexpected and all("rotary" in m for m in missing), (missing[:5], unexpected[:5])
        lm = lm.to("cuda:0")
    else:
        raise ValueError(spec)
    return tok, lm.eval()


def letter_ids(tok, fmt):
    """Token ids that mean "A" / "B" as the next token (with and without a leading space)."""
    ids = {}
    for letter in "AB":
        cands = set()
        for w in (letter, " " + letter):
            enc = tok.encode(w, add_special_tokens=False)
            # sentencepiece may emit a lone "▁" first; take the token that carries the letter
            cands.add(enc[-1])
        ids[letter] = sorted(cands)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", choices=("raw", "chat"), default="raw")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="outputs/ab_knowledge")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok, lm = load_model(args.model)
    ids = letter_ids(tok, args.format)
    rows = []
    with torch.no_grad():
        for it in items():
            text = TEMPLATE.format(instruction=it["instruction"], a=it["option_a"], b=it["option_b"])
            if args.format == "chat":
                prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True)
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                enc = tok(text + "\nAnswer:", return_tensors="pt")
            logits = lm(input_ids=enc.input_ids.to("cuda:0")).logits[0, -1].float().log_softmax(-1)
            lp = {k: max(float(logits[i]) for i in v) for k, v in ids.items()}
            choice = "A" if lp["A"] > lp["B"] else "B"
            rows.append({**it, "lp_A": lp["A"], "lp_B": lp["B"], "choice": choice, "correct": int(choice == it["answer"])})
    with open(out / f"{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def summarise(rs):
        by_q = defaultdict(list)
        for r in rs:
            by_q[(r["asset"], r["question_id"])].append(r["correct"])
        both = [all(v) for v in by_q.values()]
        n = len(rs)
        return {"answers": n, "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
                "said_A": round(sum(r["choice"] == "A" for r in rs) / n, 4),
                "questions": len(both), "correct_both_orders": round(sum(both) / len(both), 4)}
    summary = {"model": args.model, "format": args.format, "letter_ids": ids, "all": summarise(rows),
               "by_category": {c: summarise([r for r in rows if r["category"] == c]) for c in ASSETS.values()}}
    (out / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["all"]), {c: v["correct_both_orders"] for c, v in summary["by_category"].items()})


if __name__ == "__main__":
    main()

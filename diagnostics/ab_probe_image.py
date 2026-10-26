"""Image + text version of diagnostics/ab_probe.py: is the answer encoded when the model sees the robot's scene?

Input: the exact first robot-camera frame of each episode (rendered by the static-Q&A runs,
outputs/magma-sqa-<asset>-knowledge-<layout>-*/static_qa/frame_qNNN.png; all 10 categories, both layouts) plus

  In: A robot must follow this instruction: "<knowledge instruction>" There are two image tiles on the table, one on
  the left and one on the right. Which tile should the robot place the cube on? Answer with one word: left or right.
  Out:

Label = the correct side (left/right; flips between layouts). Probe = last-token hidden state per LLM layer.
Controls (same as the text probe):
  no_question  instruction replaced by a neutral one -> must be at chance (rules out "the answer photo looks
               different" shortcuts from the image itself)
  shuffled     labels permuted (5x) -> chance band
Also scores the answer directly: log P(" left") vs log P(" right") for the next token.
CV: 5-fold GroupKFold by image pair (the two tiles), so a test pair is never seen in training.

Models (--model): prismatic:<model_id> (full Prismatic VLM) | openvla:<repo> (OpenVLA checkpoint) | magma.
Writes <out-dir>/<name>_image_probe.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnostics.ab_probe import run_probes

CARROT = Path(__file__).resolve().parents[1] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
NEUTRAL = "Place the cube on one of the two tiles."
TEXT = ('A robot must follow this instruction: "{instruction}" There are two image tiles on the table, one on the left '
        "and one on the right. Which tile should the robot place the cube on? Answer with one word: left or right.")
RUN_RE = re.compile(r"^magma-sqa-(?P<asset>.+)-(?P<cond>knowledge|explicit_object|explicit_spatial)-(?P<layout>noswap|swap)-q")
CAT_OF = {"photo_state": "state", "photo_attribute": "attribute", "photo_emotion": "emotion", "public_info": "public_info",
          "time_clock": "time_clock", "counting": "counting", "symmetry": "symmetry", "celebrity": "celebrity",
          "living_world": "living_world", "traffic": "traffic"}


def items(outputs=Path("outputs"), condition="knowledge"):
    """Frames + the robot runs' instruction for this condition (static-Q&A runs store both)."""
    its = []
    for f in sorted(outputs.glob(f"magma-sqa-*-{condition}-*/static_qa/static_qa.jsonl")):
        m = RUN_RE.match(f.parts[-3])
        pairs = {int(p["index"]): p for p in json.loads((CARROT / m["asset"] / "pairs.json").read_text())}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            q = int(r["question_id"])
            its.append({"asset": m["asset"], "category": next(v for k, v in CAT_OF.items() if k in m["asset"]),
                        "layout": m["layout"], "question_id": q, "instruction": r["instruction"],
                        "correct_side": r["correct_side"], "frame": str(f.parent / f"frame_q{q:03d}.png"),
                        "group": "|".join(sorted((pairs[q]["left"], pairs[q]["right"])))})
    return its


class Prismatic:
    def __init__(self, model_id):
        import os
        from prismatic.models.backbones.llm import llama2
        from prismatic.models.load import load
        # Meta's repo is gated for this account; NousResearch/Llama-2-7b-hf has identical weights (same SHA-256)
        llama2.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = "NousResearch/Llama-2-7b-hf"
        self.vlm = load(model_id, hf_token=os.environ.get("HF_TOKEN")).to("cuda:0", dtype=torch.bfloat16).eval()
        self.tok = self.vlm.llm_backbone.tokenizer
        self.transform = self.vlm.vision_backbone.get_image_transform()

    def forward(self, img, text):
        pb = self.vlm.get_prompt_builder()
        pb.add_turn(role="human", message=text)
        enc = self.tok(pb.get_prompt(), return_tensors="pt")
        pv = self.transform(img)
        pv = {k: v[None].to("cuda:0", torch.bfloat16) for k, v in pv.items()} if isinstance(pv, dict) \
            else pv[None].to("cuda:0", torch.bfloat16)
        return self.vlm(input_ids=enc.input_ids.to("cuda:0"), attention_mask=enc.attention_mask.to("cuda:0"),
                        pixel_values=pv, output_hidden_states=True)


class OpenVLA:
    def __init__(self, repo):
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, padding_side="left")
        self.proc = PrismaticProcessor.from_pretrained(
            repo, tokenizer=self.tok, image_processor=PrismaticImageProcessor.from_pretrained(repo, trust_remote_code=True),
            trust_remote_code=True)
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="cuda:0",
            vh_mode="a0", attn_implementation="sdpa").eval()

    def forward(self, img, text):
        # same input format as the robot policy (openvla_train._preprocess_obs): [B, C, H, W] tensor
        t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].to("cuda:0", torch.bfloat16)
        inputs = self.proc([f"In: {text}\nOut: "], t, padding=True).to("cuda:0", dtype=torch.bfloat16)
        return self.vla(**inputs, output_hidden_states=True)


class Magma:
    """Magma with exactly the robot runs' input path (MagmaInference._build_inputs: same chat template, system
    prompt, 256x256 resize, vision-tower weight fix); only the user text differs."""

    def __init__(self, repo="microsoft/Magma-8B"):
        from types import SimpleNamespace
        from simpler_env.policies.magma.magma_model import MagmaInference
        self.m = MagmaInference(0, SimpleNamespace(capture_attention=True), repo)  # sdpa attention
        self.tok = self.m.processor.tokenizer

    def forward(self, img, text):
        t = torch.from_numpy(np.asarray(img))
        inputs = self.m._build_inputs(t[None], [""], user_texts=[text])
        return self.m.vla(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)


def word_id(tok, word):
    return tok.encode(" " + word, add_special_tokens=False)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="outputs/ab_probe")
    ap.add_argument("--condition", default="knowledge", choices=("knowledge", "explicit_object", "explicit_spatial"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kind, _, ref = args.model.partition(":")
    m = {"prismatic": Prismatic, "openvla": OpenVLA, "magma": Magma}[kind](*([ref] if ref else []))
    its = items(condition=args.condition)
    assert its, f"no static-Q&A frames for {args.condition}"
    y = np.array([0 if it["correct_side"] == "left" else 1 for it in its])
    groups = np.unique([it["group"] for it in its], return_inverse=True)[1]
    lid, rid = word_id(m.tok, "left"), word_id(m.tok, "right")
    X = {"question": [], "no_question": []}
    choice, mass = [], []
    with torch.no_grad():
        for it in its:
            img = Image.open(it["frame"]).convert("RGB")
            for cond in X:
                o = m.forward(img, TEXT.format(instruction=NEUTRAL if cond == "no_question" else it["instruction"]))
                X[cond].append(torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy())
                if cond == "question":
                    p = o.logits[0, -1].float().softmax(-1)
                    choice.append(0 if p[lid] > p[rid] else 1)
                    mass.append(float(p[lid] + p[rid]))
    X = {k: np.stack(v) for k, v in X.items()}
    choice = np.array(choice)
    np.savez_compressed(out / f"{args.name}_image_hidden.npz", question=X["question"].astype(np.float16),
                        no_question=X["no_question"].astype(np.float16), y=y, groups=groups)
    res = {"model": args.model, "condition": args.condition, "n_items": len(its), "n_groups": int(groups.max() + 1),
           **run_probes(X["question"], X["no_question"], y, groups)}
    res["answer_accuracy"] = float((choice == y).mean())
    res["answer_said_right"] = float(choice.mean())
    res["left_right_prob_mass_median"] = float(np.median(mass))
    by_q = {}
    for it, c, t in zip(its, choice, y):
        by_q.setdefault((it["asset"], it["question_id"]), []).append(c == t)
    res["answer_correct_both_layouts"] = float(np.mean([all(v) for v in by_q.values() if len(v) == 2]))
    (out / f"{args.name}_image_probe.json").write_text(json.dumps(res, indent=1))
    b = int(np.argmax(res["question"]))
    print(f"{args.name} [image+text]: probe best layer {b} = {res['question'][b]:.3f} | no_question max "
          f"{max(res['no_question']):.3f} | shuffled max {max(res['shuffled_max']):.3f} | answer acc {res['answer_accuracy']:.3f}, "
          f"both layouts {res['answer_correct_both_layouts']:.3f}, P(left)+P(right) median {res['left_right_prob_mass_median']:.2e}"
          f" | mlp best {max(res['question_mlp']):.3f} mlp no_question {max(res['no_question_mlp']):.3f}"
          f" mlp shuffled_max {max(res['shuffled_mlp_max']):.3f}")


if __name__ == "__main__":
    main()

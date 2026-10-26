"""Recovery test: is the knowledge truly lost, or only disconnected from the output? (Zheng et al., ICLR 2025)

Light fine-tuning on half of the text-only A/B knowledge items, evaluation on the other half (unseen option
pairs). If accuracy on unseen questions recovers quickly to the level of models that kept the knowledge, the
"forgetting" was spurious (knowledge still encoded, only the answer mapping broken); if it stays near the
no-knowledge control, the knowledge is gone.

  Items/prompt: same as diagnostics/ab_knowledge.py (raw format, both option orders, 8 categories).
  Split: 2 folds by unordered option pair (GroupKFold); train on one, test on the other, then swap.
  Fine-tuning: LoRA r=16 (alpha 32, dropout 0.05) on q/k/v/o projections of the language model; base weights
               frozen. Loss = cross-entropy on the answer letter token after "Answer:". AdamW lr --lr (2e-4), batch 8,
               --epochs epochs; test accuracy after every epoch (epoch 0 = before training).
  Conditions: question     the real instruction
              no_question  instruction replaced by a neutral one, in training and test (no-knowledge control:
                           must stay near 50%, else the fine-tuning learns answers from the option names alone)

Writes <out-dir>/<name>_recovery.json: per condition, per fold, per epoch test accuracy and correct_both_orders.
Run from the repo root:  python -m diagnostics.ab_recovery --model openvla:openvla/openvla-7b --name openvla-base
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from diagnostics.ab_knowledge import TEMPLATE, items, letter_ids, load_model

NEUTRAL = "Place the cube on one of the two tiles."


def encode(tok, it, neutral):
    text = TEMPLATE.format(instruction=NEUTRAL if neutral else it["instruction"], a=it["option_a"], b=it["option_b"])
    return tok(text + "\nAnswer:", return_tensors="pt").input_ids[0]


@torch.no_grad()
def evaluate(model, tok, its, ids, neutral):
    model.eval()
    correct, by_q = [], {}
    for it in its:
        x = encode(tok, it, neutral).to("cuda:0")[None]
        lp = model(input_ids=x).logits[0, -1].float()
        c = "A" if max(lp[i] for i in ids["A"]) > max(lp[i] for i in ids["B"]) else "B"
        correct.append(c == it["answer"])
        by_q.setdefault((it["asset"], it["question_id"]), []).append(c == it["answer"])
    return {"accuracy": float(np.mean(correct)),
            "correct_both_orders": float(np.mean([all(v) for v in by_q.values() if len(v) == 2]))}


def train_epoch(model, tok, its, ids, neutral, opt, bs=8, seed=0):
    model.train()
    rng = random.Random(seed)
    order = list(range(len(its)))
    rng.shuffle(order)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    losses = []
    for s in range(0, len(order), bs):
        batch = [its[i] for i in order[s:s + bs]]
        seqs = [encode(tok, it, neutral) for it in batch]
        L = max(len(q) for q in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, q in enumerate(seqs):  # left-pad so the last position is the answer position for every row
            x[i, L - len(q):] = q
            mask[i, L - len(q):] = 1
        target = torch.tensor([ids[it["answer"]][0] for it in batch])
        logits = model(input_ids=x.to("cuda:0"), attention_mask=mask.to("cuda:0")).logits[:, -1].float()
        loss = torch.nn.functional.cross_entropy(logits, target.to("cuda:0"))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return float(np.mean(losses))


def main():
    from peft import LoraConfig, get_peft_model
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out-dir", default="outputs/ab_recovery")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    its = items()
    groups = np.unique([it["pair"] for it in its], return_inverse=True)[1]
    tok, base = load_model(args.model)
    for p in base.parameters():
        p.requires_grad_(False)
    ids = letter_ids(tok, "raw")
    res = {"model": args.model, "n_items": len(its), "n_groups": int(groups.max() + 1), "epochs": args.epochs, "lr": args.lr,
           "letter_ids": ids}
    for cond in ("question", "no_question"):
        neutral = cond == "no_question"
        res[cond] = []
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=2).split(its, groups=groups)):
            train, test = [its[i] for i in tr], [its[i] for i in te]
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            model = get_peft_model(base, cfg)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            curve = [{"epoch": 0, "train_loss": None, **evaluate(model, tok, test, ids, neutral)}]
            for ep in range(1, args.epochs + 1):
                loss = train_epoch(model, tok, train, ids, neutral, opt, seed=ep)
                curve.append({"epoch": ep, "train_loss": loss, **evaluate(model, tok, test, ids, neutral)})
                print(f"{args.name} {cond} fold{fold} epoch {ep}: loss {loss:.3f} test acc {curve[-1]['accuracy']:.3f} "
                      f"both orders {curve[-1]['correct_both_orders']:.3f}", flush=True)
            res[cond].append({"fold": fold, "n_train": len(train), "n_test": len(test), "curve": curve})
            base = model.unload()  # drop the LoRA weights: the next run starts from the original model
            del model, opt
            torch.cuda.empty_cache()
    (out / f"{args.name}_recovery.json").write_text(json.dumps(res, indent=1))
    for cond in ("question", "no_question"):
        last = np.mean([f["curve"][-1]["accuracy"] for f in res[cond]])
        first = np.mean([f["curve"][0]["accuracy"] for f in res[cond]])
        print(f"{args.name} {cond}: test accuracy before {first:.3f} -> after {args.epochs} epochs {last:.3f}")


if __name__ == "__main__":
    main()

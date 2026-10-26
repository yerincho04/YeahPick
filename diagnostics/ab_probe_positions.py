"""Is the answer encoded anywhere in the prompt, not only at the last token? Text-only A/B items (as ab_probe.py).

For every layer, features are taken from four kinds of positions:
  last         the final token (as in ab_probe.py)
  mean_all     mean over all prompt tokens
  instruction  mean over the instruction tokens
  options      [mean over option-A tokens ; mean over option-B tokens]  (the options come after the instruction,
               so with causal attention their states can already reflect whether each option fits it)
Each feature set gets the linear probe with the same controls: no-question prompt (must be chance) and shuffled
labels (5x). CV: 5-fold GroupKFold by option pair.

Writes <out-dir>/<name>_positions.json.
Run from the repo root:  python -m diagnostics.ab_probe_positions --model openvla:openvla/openvla-7b --name openvla-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.ab_knowledge import TEMPLATE, items, load_model
from diagnostics.ab_probe import NEUTRAL, probe

FEATURES = ("last", "mean_all", "instruction", "options")


def span_tokens(offsets, start, end):
    """Token indices whose character span overlaps [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]


def features(tok, lm, its, neutral):
    out = {f: [] for f in FEATURES}
    with torch.no_grad():
        for it in its:
            instr = NEUTRAL if neutral else it["instruction"]
            text = TEMPLATE.format(instruction=instr, a=it["option_a"], b=it["option_b"]) + "\nAnswer:"
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            i0 = text.index(instr)
            ia = text.index("\nA. ") + 4
            ib = text.index("\nB. ") + 4
            sp = {"instruction": span_tokens(offs, i0, i0 + len(instr)),
                  "a": span_tokens(offs, ia, ia + len(it["option_a"])),
                  "b": span_tokens(offs, ib, ib + len(it["option_b"]))}
            if not all(sp.values()):
                # some tokenizers (e.g. Magma's Llama-3 one) return unusable offsets: rebuild each token's character
                # span by decoding growing prefixes of the token ids
                ids = enc.input_ids[0].tolist()
                ends = [len(tok.decode(ids[:k + 1], skip_special_tokens=True)) for k in range(len(ids))]
                dec = tok.decode(ids, skip_special_tokens=True)
                shift = len(dec) - len(text)  # decode may add/drop a leading space
                offs = [((ends[k - 1] if k else 0) - shift, ends[k] - shift) for k in range(len(ids))]
                sp = {"instruction": span_tokens(offs, i0, i0 + len(instr)),
                      "a": span_tokens(offs, ia, ia + len(it["option_a"])),
                      "b": span_tokens(offs, ib, ib + len(it["option_b"]))}
            assert all(sp.values()), (text, sp)
            hs = lm(input_ids=enc.input_ids.to("cuda:0"), output_hidden_states=True).hidden_states
            H = torch.stack([h[0] for h in hs]).float()  # [layers, tokens, d]
            out["last"].append(H[:, -1].cpu().numpy())
            out["mean_all"].append(H.mean(1).cpu().numpy())
            out["instruction"].append(H[:, sp["instruction"]].mean(1).cpu().numpy())
            out["options"].append(torch.cat([H[:, sp["a"]].mean(1), H[:, sp["b"]].mean(1)], -1).cpu().numpy())
    return {f: np.stack(v) for f, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="outputs/ab_probe")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    its = items()
    y = np.array([0 if it["answer"] == "A" else 1 for it in its])
    groups = np.unique([it["pair"] for it in its], return_inverse=True)[1]
    tok, lm = load_model(args.model)
    assert getattr(tok, "is_fast", False), "needs a fast tokenizer for offsets"
    Fq, Fn = features(tok, lm, its, neutral=False), features(tok, lm, its, neutral=True)
    rng = np.random.default_rng(0)
    perms = [rng.permutation(y) for _ in range(5)]
    res = {"model": args.model, "n_items": len(its)}
    for f in FEATURES:
        shuf = [probe(Fq[f], p, groups) for p in perms]
        res[f] = {"question": probe(Fq[f], y, groups), "no_question": probe(Fn[f], y, groups),
                  "shuffled_max": [float(np.max(v)) for v in zip(*shuf)]}
        r = res[f]
        b = int(np.argmax(r["question"]))
        print(f"{args.name} {f:12s}: best layer {b} = {r['question'][b]:.3f} | no_question max {max(r['no_question']):.3f}"
              f" | shuffled max {max(r['shuffled_max']):.3f}", flush=True)
    (out / f"{args.name}_positions.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

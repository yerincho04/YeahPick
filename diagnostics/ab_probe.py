"""Is the answer still encoded inside the model? Layer-wise linear probe on the text-only A/B knowledge test,
with the controls needed to trust it in both directions.

Same items and prompt as diagnostics/ab_knowledge.py (raw format). For every layer, the last-token hidden state is
used to predict the correct letter (A or B; both option orders, so the letter flips with the order).

  question      the real prompt                                   -> probe accuracy = knowledge readable?
  no_question   instruction replaced by a neutral one             -> must fall to chance (no surface-cue shortcut)
  shuffled      real prompt, labels permuted (5 permutations)     -> must sit at chance (probe does not fit noise)

Cross-validation: 5-fold GroupKFold, grouped by the unordered option pair, so a test question's option pair never
appears in training (no memorising items). Logistic regression (standardised, C=0.1).

--transplant-head <spec>: also score A vs B by passing this model's final normed hidden state through another
model's output layer (e.g. Prismatic's head on OpenVLA states).

Writes <out-dir>/<name>_probe.json (and the hidden states as <name>_hidden.npz).
Run from the repo root:  python -m diagnostics.ab_probe --model prismatic:prism-dinosiglip-224px+7b --name prismatic-7b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnostics.ab_knowledge import TEMPLATE, items, letter_ids, load_model

NEUTRAL = "Place the cube on one of the two tiles."


def hidden_states(tok, lm, its, neutral):
    """[n_items, n_layers+1, d] last-token hidden states, plus the final normed state for head transplant."""
    hs, finals = [], []
    with torch.no_grad():
        for it in its:
            text = TEMPLATE.format(instruction=NEUTRAL if neutral else it["instruction"], a=it["option_a"], b=it["option_b"])
            ids = tok(text + "\nAnswer:", return_tensors="pt").input_ids.to("cuda:0")
            out = lm(input_ids=ids, output_hidden_states=True)
            hs.append(torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
            finals.append(out.hidden_states[-1][0, -1].float().cpu())  # HF Llama: last entry is after the final norm
    return np.stack(hs), torch.stack(finals)


def make_clf(kind, seed=0):
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, random_state=seed))
    # non-linear: one hidden layer of 256 units (ReLU), L2-regularised
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    return make_pipeline(StandardScaler(), PCA(n_components=128, random_state=seed),
                         MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=500, random_state=seed))


def probe(X, y, groups, seed=0, kind="linear"):
    accs = []
    for layer in range(X.shape[1]):
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=5).split(X[:, layer], y, groups):
            pred[te] = make_clf(kind, seed).fit(X[tr, layer], y[tr]).predict(X[te, layer])
        accs.append(float((pred == y).mean()))
    return accs


def run_probes(Xq, Xn, y, groups, n_shuffle=5, mlp=True):
    """Linear (and non-linear) probes on the real prompt, the no-question control and shuffled labels."""
    res = {"question": probe(Xq, y, groups), "no_question": probe(Xn, y, groups)}
    rng = np.random.default_rng(0)
    shuf = [probe(Xq, rng.permutation(y), groups) for _ in range(n_shuffle)]
    res["shuffled_mean"] = [float(np.mean(v)) for v in zip(*shuf)]
    res["shuffled_max"] = [float(np.max(v)) for v in zip(*shuf)]
    if mlp:
        res["question_mlp"] = probe(Xq, y, groups, kind="mlp")
        res["no_question_mlp"] = probe(Xn, y, groups, kind="mlp")
        shuf = [probe(Xq, rng.permutation(y), groups, kind="mlp") for _ in range(2)]
        res["shuffled_mlp_max"] = [float(np.max(v)) for v in zip(*shuf)]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--transplant-head", default=None)
    ap.add_argument("--condition", default="knowledge", choices=("knowledge", "explicit_object", "explicit_spatial"))
    ap.add_argument("--out-dir", default="outputs/ab_probe")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    its = items(args.condition)
    y = np.array([0 if it["answer"] == "A" else 1 for it in its])
    # fixed id per unordered option pair (reproducible folds)
    groups = np.unique([it["pair"] for it in its], return_inverse=True)[1]
    tok, lm = load_model(args.model)
    Xq, finals = hidden_states(tok, lm, its, neutral=False)
    Xn, _ = hidden_states(tok, lm, its, neutral=True)
    np.savez_compressed(out / f"{args.name}_hidden.npz", question=Xq.astype(np.float16), no_question=Xn.astype(np.float16), y=y)
    res = {"model": args.model, "condition": args.condition, "n_items": len(its), "n_groups": int(len(set(groups))),
           **run_probes(Xq, Xn, y, groups)}
    if args.transplant_head:
        del lm
        torch.cuda.empty_cache()
        htok, hlm = load_model(args.transplant_head)
        head = hlm.get_output_embeddings().weight.detach().float()  # [vocab, d]
        ids = letter_ids(htok, "raw")
        with torch.no_grad():
            logits = (finals.to(head.device) @ head.T).log_softmax(-1)
        choice = np.where(logits[:, ids["A"]].max(-1).values.cpu().numpy() > logits[:, ids["B"]].max(-1).values.cpu().numpy(), 0, 1)
        res["transplant_head"] = args.transplant_head
        res["transplant_accuracy"] = float((choice == y).mean())
        by_q = {}
        for it, c, t in zip(its, choice, y):
            by_q.setdefault((it["asset"], it["question_id"]), []).append(c == t)
        res["transplant_correct_both_orders"] = float(np.mean([all(v) for v in by_q.values()]))
    (out / f"{args.name}_probe.json").write_text(json.dumps(res, indent=1))
    best = int(np.argmax(res["question"]))
    print(f"{args.name}: best layer {best} question={res['question'][best]:.3f} "
          f"no_question={max(res['no_question']):.3f} (max over layers) shuffled_max={max(res['shuffled_max']):.3f}"
          + f" | mlp best={max(res['question_mlp']):.3f} mlp no_question={max(res['no_question_mlp']):.3f}"
          f" mlp shuffled_max={max(res['shuffled_mlp_max']):.3f}"
          + (f" transplant={res['transplant_accuracy']:.3f}" if args.transplant_head else ""))


if __name__ == "__main__":
    main()

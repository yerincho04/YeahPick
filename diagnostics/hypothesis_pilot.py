"""Hypothesis pilot for the K -> G -> A proposal (see HYPOTHESIS_PILOT.md).

Hypothesis: the correct world-knowledge answer (K) is decodable from OpenVLA's hidden state,
but it is not converted into the correct left/right target (G), so behaviour fails.

Reads the episode logs (+ hidden states for knowledge/neutral) of one asset, e.g.
semantic_repeated_v2, and reports:
  A. behaviour per condition: placement success, intent accuracy, left-choice rate, and paired
     McNemar tests knowledge vs tile_object vs explicit_spatial;
  B. layer-wise probes (5-fold, held-out question formulations, noswap/swap kept together):
       K10 = answer identity among all 10 answers (chance 10%)
       K2  = answer identity restricted to the two tiles in the scene (chance 50%; with the v2
             design the scene alone carries no information about which of the two is K)
       G   = correct side left/right (chance 50%)
     for the knowledge condition and for the information-free neutral control, with a paired
     McNemar test of K2/G (knowledge vs neutral) on the same scenes;
  C. failure decomposition of knowledge episodes per layer: P(G wrong | K correct) and
     P(intent wrong | G correct).
Probe correctness means "decodable by a held-out linear probe", not "the model knows/uses it".

Run from the repo root:
  python3 -m diagnostics.hypothesis_pilot --asset semantic_repeated_v2 --out-dir outputs/hypothesis_pilot
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnostics.intent_classifier import classify_intent

LAYER_LABELS = {0.25: "25%", 0.5: "50%", 0.75: "75%", 1.0: "final"}
BEHAVIOUR_CONDITIONS = ("knowledge", "tile_object", "explicit_spatial", "neutral")
PROBE_CONDITIONS = ("knowledge", "neutral")


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]" if n else "n/a"


def load_records(root: Path, asset: str) -> list[dict]:
    by_key = {}
    for path in sorted(root.glob(f"openvla-{asset}-*/diagnostics/episode_log.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_dir"] = path.parent
                by_key[(r["condition"], int(r["question_id"]), r["layout"])] = r
    return list(by_key.values())


def formulation_folds(records: list[dict], seed: int, n_folds: int = 5) -> dict[int, int]:
    """Question id -> fold. Each answer's 5 formulations land in 5 different folds, so every
    test fold contains one unseen formulation per answer; layouts follow their question."""
    by_answer = defaultdict(set)
    for r in records:
        by_answer[r["semantic_answer_id"]].add(int(r["question_id"]))
    rng = np.random.default_rng(seed)
    fold = {}
    for answer in sorted(by_answer):
        qids = rng.permutation(sorted(by_answer[answer]))
        if len(qids) != n_folds:
            raise ValueError(f"{answer}: expected {n_folds} formulations, got {len(qids)}")
        for f, q in enumerate(qids):
            fold[int(q)] = f
    return fold


def probe_condition(records: list[dict], folds: dict[int, int], timestep: int, seed: int) -> dict:
    """Out-of-fold K10/K2/G predictions for every episode of one condition, per layer."""
    X, episodes = defaultdict(list), []
    for r in sorted(records, key=lambda r: (int(r["question_id"]), r["layout"])):
        path = r["_dir"] / "hidden_states" / f"{r['episode_id']}.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        fracs = sorted(k for k in payload if isinstance(k, float))
        t = min(timestep, payload[fracs[0]].shape[0] - 1)
        for frac in fracs:
            X[frac].append(payload[frac][t].float().numpy())
        episodes.append(r)
    if not episodes:
        return {}
    X = {k: np.stack(v) for k, v in X.items()}
    classes = sorted({r["semantic_answer_id"] for r in episodes})
    cls = {c: i for i, c in enumerate(classes)}
    y_k = np.array([cls[r["semantic_answer_id"]] for r in episodes])
    y_g = np.array([int(r["correct_side"] == "left") for r in episodes])
    pair = np.array([[cls[r["semantic_answer_id"]], cls[r["distractor_id"]]] for r in episodes])
    fold = np.array([folds[int(r["question_id"])] for r in episodes])

    out = {"episodes": episodes, "layers": {}}
    for frac in sorted(X):
        k10 = np.zeros(len(episodes), bool)
        k2 = np.zeros(len(episodes), bool)
        g = np.zeros(len(episodes), bool)
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            k_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, random_state=seed))
            g_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, random_state=seed))
            k_clf.fit(X[frac][tr], y_k[tr])
            g_clf.fit(X[frac][tr], y_g[tr])
            proba = np.full((te.sum(), len(classes)), -np.inf)
            proba[:, k_clf.classes_] = k_clf.predict_log_proba(X[frac][te])
            k10[te] = proba.argmax(1) == y_k[te]
            rows = np.arange(te.sum())
            # K2: is the true answer scored above the distractor tile that is also in the scene?
            k2[te] = proba[rows, pair[te, 0]] > proba[rows, pair[te, 1]]
            g[te] = g_clf.predict(X[frac][te]) == y_g[te]
        out["layers"][frac] = {"k10": k10, "k2": k2, "g": g}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="semantic_repeated_v2")
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--out-dir", default="outputs/hypothesis_pilot")
    ap.add_argument("--timestep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.outputs_root), args.asset)
    by_cond = defaultdict(list)
    for r in records:
        r["intent_side"], _ = classify_intent(r)
        r["intent_correct"] = r["intent_side"] == r["correct_side"]
        by_cond[r["condition"]].append(r)
    lines = [f"# Hypothesis pilot: {args.asset}", "",
             "Wilson 95% intervals in brackets. Probe = held-out linear decodability, not proof of use.", ""]

    # ---- A. behaviour ----
    lines += ["## A. Behaviour (all conditions, same scenes and seeds)", "",
              "| Condition | Placement success | Intent correct | Chose left |", "|---|---|---|---|"]
    keyed = {c: {(int(r["question_id"]), r["layout"]): r for r in rs} for c, rs in by_cond.items()}
    for c in BEHAVIOUR_CONDITIONS:
        rs = by_cond.get(c, [])
        if not rs:
            continue
        n = len(rs)
        lines.append(f"| {c} | {fmt(sum(bool(r['task_success']) for r in rs), n)} | "
                     f"{fmt(sum(r['intent_correct'] for r in rs), n)} | "
                     f"{fmt(sum(r['intent_side'] == 'left' for r in rs), n)} |")
    # Pair consistency: a question counts as solved only if intent is correct in BOTH layouts.
    # Chance 25%; a pure position bias (always left / always right) scores 0%.
    lines += ["", "Pair consistency (intent correct in both noswap and swap; chance 25%, position bias 0%):", "",
              "| Condition | Both layouts correct | Same side in both layouts (position-driven) |", "|---|---|---|"]
    for c in BEHAVIOUR_CONDITIONS:
        if c not in keyed:
            continue
        qids = sorted({q for q, _ in keyed[c]})
        full = [q for q in qids if (q, "noswap") in keyed[c] and (q, "swap") in keyed[c]]
        both = sum(keyed[c][(q, "noswap")]["intent_correct"] and keyed[c][(q, "swap")]["intent_correct"] for q in full)
        same = sum(keyed[c][(q, "noswap")]["intent_side"] == keyed[c][(q, "swap")]["intent_side"]
                   and keyed[c][(q, "noswap")]["intent_side"] in ("left", "right") for q in full)
        lines.append(f"| {c} | {fmt(both, len(full))} | {fmt(same, len(full))} |")
    lines += ["", "Paired McNemar on intent correctness (same question + layout):", ""]
    for a, b in (("knowledge", "tile_object"), ("tile_object", "explicit_spatial"), ("knowledge", "explicit_spatial")):
        if a in keyed and b in keyed:
            common = sorted(set(keyed[a]) & set(keyed[b]))
            only_a = sum(keyed[a][k]["intent_correct"] and not keyed[b][k]["intent_correct"] for k in common)
            only_b = sum(keyed[b][k]["intent_correct"] and not keyed[a][k]["intent_correct"] for k in common)
            lines.append(f"- {a} vs {b}: n={len(common)}, only {a} correct={only_a}, only {b} correct={only_b}, "
                         f"p={exact_mcnemar(only_a, only_b):.4f}")

    # ---- B. probes ----
    folds = formulation_folds(by_cond.get("knowledge") or records, args.seed)
    probes = {c: probe_condition(by_cond[c], folds, args.timestep, args.seed) for c in PROBE_CONDITIONS if by_cond.get(c)}
    lines += ["", "## B. Layer-wise probes (5-fold, unseen question formulations)", "",
              "Chance: K10 10%, K2 50%, G 50%. Neutral = same scenes, information-free instruction "
              "(how much the scene alone gives the probe).", "",
              "| Layer | K10 knowledge | K2 knowledge | K2 neutral | K2 paired p | G knowledge | G neutral | G paired p |",
              "|---|---|---|---|---|---|---|---|"]
    rows_json = []
    kn, ne = probes.get("knowledge", {}), probes.get("neutral", {})
    for frac in sorted(kn.get("layers", {})):
        L = kn["layers"][frac]
        n = len(L["k2"])
        row = {"layer": LAYER_LABELS.get(frac, frac), "n": n, "k10_knowledge": float(L["k10"].mean()),
               "k2_knowledge": float(L["k2"].mean()), "g_knowledge": float(L["g"].mean())}
        k2n = gn = k2p = gp = "n/a"
        if ne and frac in ne["layers"]:
            N = ne["layers"][frac]
            kk = {(int(r["question_id"]), r["layout"]): i for i, r in enumerate(kn["episodes"])}
            nk = {(int(r["question_id"]), r["layout"]): i for i, r in enumerate(ne["episodes"])}
            common = sorted(set(kk) & set(nk))
            ik, inn = [kk[k] for k in common], [nk[k] for k in common]
            def paired(metric):
                a, b = L[metric][ik], N[metric][inn]
                return exact_mcnemar(int((a & ~b).sum()), int((~a & b).sum()))
            k2n, gn = fmt(int(N["k2"].sum()), len(N["k2"])), fmt(int(N["g"].sum()), len(N["g"]))
            k2p, gp = f"{paired('k2'):.4f}", f"{paired('g'):.4f}"
            row.update(k2_neutral=float(N["k2"].mean()), g_neutral=float(N["g"].mean()),
                       k2_paired_p=float(k2p), g_paired_p=float(gp))
        lines.append(f"| {row['layer']} | {fmt(int(L['k10'].sum()), n)} | {fmt(int(L['k2'].sum()), n)} | {k2n} | "
                     f"{k2p} | {fmt(int(L['g'].sum()), n)} | {gn} | {gp} |")
        rows_json.append(row)

    # ---- C. decomposition ----
    lines += ["", "## C. Failure decomposition of knowledge episodes (per probe layer)", "",
              "K = K2 probe correct, G = G probe correct, intent = realised arm motion toward the correct tile.", "",
              "| Layer | K wrong | K ok, G wrong | K,G ok, intent wrong | intent ok, placement fail | success | "
              "P(G wrong given K correct) | P(intent wrong given G correct) |", "|---|---|---|---|---|---|---|---|"]
    for frac in sorted(kn.get("layers", {})):
        L, eps = kn["layers"][frac], kn["episodes"]
        stages = defaultdict(int)
        k_ok = g_wrong_k_ok = g_ok = a_wrong_g_ok = 0
        for i, r in enumerate(eps):
            k, g, a, s = bool(L["k2"][i]), bool(L["g"][i]), r["intent_correct"], bool(r["task_success"])
            stages["K wrong" if not k else "G wrong" if not g else "intent wrong" if not a
                   else "placement fail" if not s else "success"] += 1
            k_ok += k
            g_wrong_k_ok += k and not g
            g_ok += g
            a_wrong_g_ok += g and not a
        n = len(eps)
        lines.append(f"| {LAYER_LABELS.get(frac, frac)} | " + " | ".join(
            str(stages[s]) for s in ("K wrong", "G wrong", "intent wrong", "placement fail", "success"))
            + f" | {fmt(g_wrong_k_ok, k_ok)} | {fmt(a_wrong_g_ok, g_ok)} |")
    lines += ["", "Decision rule: see diagnostics/HYPOTHESIS_PILOT.md (fixed before the run)."]

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    (out_dir / "probe_layers.json").write_text(json.dumps(rows_json, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

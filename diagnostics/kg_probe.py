"""Separate semantic-answer (K) and grounded-side (G) probes for repeated answers."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from diagnostics.intent_classifier import classify_intent

LAYER_LABELS = {0.25: "Layer 25%", 0.5: "Layer 50%", 0.75: "Layer 75%", 1.0: "Final"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics-dir", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--timestep", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dirs = [Path(x) for x in args.diagnostics_dir]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    records = {}
    for d in dirs:
        for line in (d / "episode_log.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line); records[r["episode_id"]] = r

    X = defaultdict(list); answer_ids = []; sides = []; question_ids = []; episode_ids = []
    for d in dirs:
        for path in sorted((d / "hidden_states").glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            r = records[payload["episode_id"]]
            fracs = sorted(k for k in payload if isinstance(k, float))
            t = min(args.timestep, payload[fracs[0]].shape[0] - 1)
            for frac in fracs:
                X[frac].append(payload[frac][t].float().numpy())
            answer_ids.append(r["semantic_answer_id"])
            sides.append(1 if r["correct_side"] == "left" else 0)
            question_ids.append(int(r["question_id"]))
            episode_ids.append(r["episode_id"])
    X = {k: np.stack(v) for k, v in X.items()}
    answer_ids = np.asarray(answer_ids); sides = np.asarray(sides); question_ids = np.asarray(question_ids)
    episode_ids = np.asarray(episode_ids, dtype=object)
    classes = sorted(set(answer_ids)); class_to_int = {c: i for i, c in enumerate(classes)}
    y_k = np.asarray([class_to_int[x] for x in answer_ids])

    # Strict formulation holdout: for every answer identity, 3 question formulations train,
    # 1 validation, 1 test. Noswap/swap copies follow their question as a group.
    rng = np.random.default_rng(args.seed)
    train_q, val_q, test_q = set(), set(), set()
    for cls in classes:
        qs = np.array(sorted(set(question_ids[answer_ids == cls])))
        if len(qs) != 5:
            raise ValueError(f"Expected five formulations for {cls}, got {qs.tolist()}")
        qs = rng.permutation(qs)
        train_q.update(qs[:3].tolist()); val_q.add(int(qs[3])); test_q.add(int(qs[4]))
    train = np.flatnonzero(np.isin(question_ids, list(train_q)))
    val = np.flatnonzero(np.isin(question_ids, list(val_q)))
    test = np.flatnonzero(np.isin(question_ids, list(test_q)))
    assert not (train_q & val_q or train_q & test_q or val_q & test_q)

    np.savez(
        out / "kg_probe_dataset.npz", **{f"X_{k}": v for k, v in X.items()},
        y_k=y_k, y_g=sides, answer_ids=answer_ids, question_ids=question_ids,
        episode_ids=episode_ids, train_idx=train, val_idx=val, test_idx=test,
    )

    metrics = []
    final_k_pred = final_g_pred = None
    for frac in sorted(X):
        k_clf = LogisticRegression(max_iter=3000, solver="liblinear", multi_class="ovr", random_state=args.seed)
        g_clf = LogisticRegression(max_iter=3000, solver="liblinear", random_state=args.seed)
        k_clf.fit(X[frac][train], y_k[train]); g_clf.fit(X[frac][train], sides[train])
        k_val, k_test = k_clf.predict(X[frac][val]), k_clf.predict(X[frac][test])
        g_val, g_test = g_clf.predict(X[frac][val]), g_clf.predict(X[frac][test])
        metrics.append({
            "layer_fraction": frac, "layer": LAYER_LABELS.get(frac, str(frac)),
            "k_val_accuracy": float(accuracy_score(y_k[val], k_val)),
            "k_test_accuracy": float(accuracy_score(y_k[test], k_test)),
            "k_val_macro_f1": float(f1_score(y_k[val], k_val, average="macro", zero_division=0)),
            "k_test_macro_f1": float(f1_score(y_k[test], k_test, average="macro", zero_division=0)),
            "g_val_accuracy": float(accuracy_score(sides[val], g_val)),
            "g_test_accuracy": float(accuracy_score(sides[test], g_test)),
        })
        if frac == max(X):
            final_k_pred, final_g_pred = k_test, g_test

    cm = confusion_matrix(y_k[test], final_k_pred, labels=np.arange(len(classes)))
    cm_rows = [{"true_answer": cls, **{pred: int(cm[i, j]) for j, pred in enumerate(classes)}} for i, cls in enumerate(classes)]
    write_csv(out / "k_confusion_matrix.csv", cm_rows)
    write_csv(out / "layer_metrics.csv", metrics)

    predictions = []
    for rel, idx in enumerate(test):
        predictions.append({
            "episode_id": str(episode_ids[idx]), "question_id": int(question_ids[idx]),
            "semantic_answer_id": str(answer_ids[idx]), "k_predicted_answer_id": classes[int(final_k_pred[rel])],
            "k_correct": bool(final_k_pred[rel] == y_k[idx]),
            "g_true_side": "left" if sides[idx] else "right",
            "g_predicted_side": "left" if final_g_pred[rel] else "right",
            "g_correct": bool(final_g_pred[rel] == sides[idx]), "probe_layer": "Final",
        })
    with (out / "kg_test_predictions.jsonl").open("w") as f:
        for row in predictions: f.write(json.dumps(row) + "\n")

    pred_by_episode = {r["episode_id"]: r for r in predictions}
    behavior_rows = []
    for episode_id, pred in pred_by_episode.items():
        r = records[episode_id]
        intent_side, confidence = classify_intent(r)
        intent_correct = intent_side == r["correct_side"]
        execution = bool(r["task_success"])
        if not pred["k_correct"]:
            stage = "K wrong"
        elif not pred["g_correct"]:
            stage = "K correct, G wrong"
        elif not intent_correct:
            stage = "K correct, G correct, intent wrong"
        elif not execution:
            stage = "K correct, G correct, intent correct, execution fail"
        else:
            stage = "full success"
        behavior_rows.append({**pred, "intent_side": intent_side, "intent_confidence": confidence,
                              "intent_correct": intent_correct, "placement_success": execution, "failure_stage": stage})
    write_csv(out / "behavior_decomposition.csv", behavior_rows)
    stage_counts = Counter(r["failure_stage"] for r in behavior_rows)

    summary = {
        "asset": "semantic_repeated_v1", "episodes": len(episode_ids), "unique_questions": len(set(question_ids)),
        "answer_classes": classes, "chance_k": 1 / len(classes), "chance_g": 0.5,
        "split": {"train_episodes": len(train), "val_episodes": len(val), "test_episodes": len(test),
                  "train_questions": len(train_q), "val_questions": len(val_q), "test_questions": len(test_q)},
        "split_rule": "per answer identity: 3 unseen formulations train, 1 val, 1 test; layouts grouped by question_id",
        "layer_metrics": metrics, "behavior_stage_counts_test_fold": dict(stage_counts),
    }
    (out / "kg_probe_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    pct = lambda x: f"{100*x:.1f}%"
    report = [
        "# Repeated-answer K/G probe report", "", f"- Episodes: {len(episode_ids)}",
        f"- Questions: {len(set(question_ids))}", f"- K classes: {len(classes)} (chance {pct(1/len(classes))})",
        f"- G classes: 2 (chance {pct(0.5)})", "- Split: 3/1/1 held-out formulations per answer identity", "",
        "| Layer | K val accuracy | K test accuracy | K test macro F1 | G val accuracy | G test accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        report.append(f"| {r['layer']} | {pct(r['k_val_accuracy'])} | {pct(r['k_test_accuracy'])} | {pct(r['k_test_macro_f1'])} | {pct(r['g_val_accuracy'])} | {pct(r['g_test_accuracy'])} |")
    report += ["", "## Held-out behavioral decomposition", ""]
    for stage in ("K wrong", "K correct, G wrong", "K correct, G correct, intent wrong",
                  "K correct, G correct, intent correct, execution fail", "full success"):
        report.append(f"- {stage}: {stage_counts.get(stage, 0)}")
    report += ["", "Probe correctness means decodable by the fitted held-out classifier; it is not proof that the model knows or uses the answer."]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

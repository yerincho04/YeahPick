"""Act2Answer Phase 1, Task 5: linear probe hidden_state -> left/right, one per layer.

Reads the .npz written by diagnostics/probe_dataset.py. Fits plain LogisticRegression
(no deep network, per the spec) on train, reports val/test accuracy per layer, and saves
the test-fold predictions (episode_id -> predicted_correct = bool) for
diagnostics/failure_decomposition.py's semantic_correct column -- using only test-fold
predictions keeps that column out-of-sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dataset", required=True, help="Path to probe_dataset.npz")
    parser.add_argument("--out", default=None, help="Where to write semantic_probe.jsonl (default: alongside probe-dataset)")
    args = parser.parse_args()

    data = np.load(args.probe_dataset, allow_pickle=True)
    y = data["y"]
    episode_ids = data["episode_ids"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    fracs = sorted(
        float(k[2:]) for k in data.files if k.startswith("X_")
    )

    LAYER_LABELS = {0.25: "Layer 25%", 0.5: "Layer 50%", 0.75: "Layer 75%", 1.0: "Final"}

    print(f"{'Layer':<12} {'Val Acc':>10} {'Test Acc':>10}")
    rows = []
    test_preds_by_frac = {}
    for frac in fracs:
        X = data[f"X_{frac}"]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])

        val_acc = clf.score(X[val_idx], y[val_idx]) if len(val_idx) else float("nan")
        test_acc = clf.score(X[test_idx], y[test_idx]) if len(test_idx) else float("nan")
        print(f"{LAYER_LABELS.get(frac, frac):<12} {val_acc:>10.3f} {test_acc:>10.3f}")
        rows.append({"layer": LAYER_LABELS.get(frac, frac), "val_acc": val_acc, "test_acc": test_acc})

        test_preds_by_frac[frac] = clf.predict(X[test_idx])

    # Use the final layer's test-fold predictions as the "semantic_correct" signal for
    # Task 6, since that's the representation closest to the model's actual decision.
    final_frac = max(fracs)
    out_records = []
    for rel_i, abs_i in enumerate(test_idx):
        pred = int(test_preds_by_frac[final_frac][rel_i])
        out_records.append({
            "episode_id": str(episode_ids[abs_i]),
            "semantic_probe_pred_left": bool(pred),
            "semantic_correct": bool(pred == int(y[abs_i])),
            "probe_layer_used": LAYER_LABELS.get(final_frac, final_frac),
        })

    out_path = Path(args.out) if args.out else Path(args.probe_dataset).parent / "semantic_probe.jsonl"
    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(out_records)} test-fold semantic-probe predictions to {out_path}")
    print("(layer table above; failure_decomposition.py joins on episode_id, restricted to the test fold)")


if __name__ == "__main__":
    main()

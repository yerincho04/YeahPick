"""Act2Answer Phase 1: guard against the one bug that would silently invalidate the probe
table -- a question_id (i.e. a normal/swapped pair) leaking across train/val/test.

Usage:
    python -m diagnostics.test_split_integrity --probe-dataset <diagnostics-dir>/probe_dataset.npz
Exits non-zero (via assert) on any overlap. Also runnable under pytest.
"""
from __future__ import annotations

import argparse

import numpy as np


def assert_no_question_id_leakage(question_ids: np.ndarray, train_idx, val_idx, test_idx) -> None:
    train_q = set(question_ids[train_idx].tolist())
    val_q = set(question_ids[val_idx].tolist())
    test_q = set(question_ids[test_idx].tolist())

    assert not (train_q & val_q), f"question_ids leaked between train/val: {train_q & val_q}"
    assert not (train_q & test_q), f"question_ids leaked between train/test: {train_q & test_q}"
    assert not (val_q & test_q), f"question_ids leaked between val/test: {val_q & test_q}"


def test_no_leakage_synthetic():
    """Pytest-discoverable regression test using synthetic data (2 episodes per question)."""
    from diagnostics.probe_dataset import group_split

    question_ids = np.repeat(np.arange(20), 2)  # 20 questions x (normal, swapped)
    train_idx, val_idx, test_idx = group_split(question_ids, seed=0)
    assert_no_question_id_leakage(question_ids, train_idx, val_idx, test_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dataset", required=True)
    args = parser.parse_args()

    data = np.load(args.probe_dataset, allow_pickle=True)
    assert_no_question_id_leakage(data["question_ids"], data["train_idx"], data["val_idx"], data["test_idx"])
    print("OK: no question_id leakage across train/val/test.")


if __name__ == "__main__":
    main()

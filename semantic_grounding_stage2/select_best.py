"""Select a Stage 2 checkpoint using predicted-side validation rollouts."""
from __future__ import annotations

import json
from pathlib import Path

from config import CONFIG


def main() -> None:
    import os

    run = Path(os.environ["STAGE2_RUN_DIR"]).resolve()
    results = []
    for checkpoint in sorted((run / "checkpoints").glob("epoch_*")):
        path = run / "validation" / checkpoint.name / "val_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run validation rollout first: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        strict = metrics["rollout"]["by_mode"]["predicted"]["strict_success"]
        side = metrics["grounding"]["overall"]
        results.append({"adapter": str(checkpoint), "epoch": int(checkpoint.name.split("_")[-1]),
                        "predicted_strict_success": strict,
                        "side_accuracy": side})
    if len(results) != CONFIG.epochs:
        raise ValueError(f"Expected {CONFIG.epochs} evaluated checkpoints, found {len(results)}")
    best = max(results, key=lambda r: (
        r["predicted_strict_success"]["accuracy"],
        r["side_accuracy"]["accuracy"], -r["epoch"]))
    (run / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print("BEST_STAGE2", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()

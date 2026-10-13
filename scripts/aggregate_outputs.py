"""Write one CSV row per evaluated question found under outputs/.

Usage: python scripts/aggregate_outputs.py [--outputs PATH] [--csv PATH]
Requires PyYAML (installed in the evaluation environment).
"""

import argparse
import csv
import json
import re
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = {
    "knowledge": "Semantic reasoning",
    "explicit_object": "Explicit object",
    "explicit_spatial": "Explicit spatial",
}
BASE_COLUMNS = [
    "idx", "type", "task", "instruction", "swap", "left", "right",
    "answer_side", "success", "run",
]


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def csv_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value if value is not None else ""


def pair_lookup(asset: str) -> dict:
    path = ROOT / "ManiSkill/mani_skill/assets/carrot" / asset / "pairs.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        pairs = json.load(stream)
    return {
        int(pair["question_id"] if "question_id" in pair else pair["index"]): pair
        for pair in pairs
    }


def collect_rows(outputs: Path) -> list[dict]:
    rows = []
    pairs_by_asset = {}

    for stats_path in sorted(outputs.rglob("stats.yaml")):
        config_path = stats_path.parent.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config for {stats_path}: {config_path}")

        config = read_yaml(config_path)
        stats = read_yaml(stats_path)
        asset = str(config["assets"])
        condition = str(config.get("instruction_condition", ""))
        swap = bool(config.get("do_swap", False))
        ids = config["ids"]
        instructions = stats.get("instruction", {})
        final_info = stats.get("last_info", {})
        if asset not in pairs_by_asset:
            pairs_by_asset[asset] = pair_lookup(asset)
        pairs = pairs_by_asset[asset]
        run = stats_path.parent.parent.parent.name

        for local_idx, info in sorted(final_info.items(), key=lambda item: int(item[0])):
            local_idx = int(local_idx)
            if local_idx >= len(ids):
                raise ValueError(f"Question index {local_idx} exceeds config.ids in {stats_path}")
            idx = int(ids[local_idx])
            pair = pairs.get(idx, {})
            left, right = pair.get("left", ""), pair.get("right", "")
            answer_side = str(pair.get("answer", "")).lower()
            if swap:
                left, right = right, left
                answer_side = {"left": "right", "right": "left"}.get(answer_side, answer_side)

            success = csv_value(info.get("success"))
            row = {
                "idx": idx,
                "type": re.sub(r"_v\d+$", "", asset),
                "task": TASK_NAMES.get(condition, condition),
                "instruction": instructions.get(local_idx, instructions.get(str(local_idx), "")),
                "swap": int(swap),
                "left": left,
                "right": right,
                "answer_side": answer_side,
                "success": success,
                "run": run,
            }
            row.update({f"info_{key}": csv_value(value) for key, value in info.items() if key != "success"})
            rows.append(row)

    rows.sort(key=lambda row: (row["type"], row["task"], row["swap"], row["idx"], row["run"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs/results_raw.csv")
    args = parser.parse_args()
    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        parser.error(f"Output directory does not exist: {outputs}")

    rows = collect_rows(outputs)
    extra_columns = sorted({key for row in rows for key in row if key not in BASE_COLUMNS})
    csv_path = args.csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BASE_COLUMNS + extra_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

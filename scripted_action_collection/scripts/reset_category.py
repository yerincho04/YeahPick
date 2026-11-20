"""Archive one category's collected episodes and remove its manifest entries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_assets import TILE_WIDTH

OUTPUT = ROOT / f"output_tile_{round(TILE_WIDTH * 100):02d}cm_v4"
CATEGORIES = (
    "counting", "symmetry", "celebrity", "living_world", "traffic",
    "state", "attribute", "emotion", "public_info", "time_clock",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=CATEGORIES)
    args = parser.parse_args()

    manifest = OUTPUT / "episodes.jsonl"
    episode_dir = OUTPUT / "episodes" / args.category
    kept: list[str] = []
    archived: list[str] = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines(keepends=True):
            if json.loads(line)["category"] == args.category:
                archived.append(line)
            else:
                kept.append(line)
    if not archived and not episode_dir.exists():
        print(f"NOTHING_TO_RESET {args.category}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = OUTPUT / "archive" / f"{args.category}_{stamp}"
    backup.mkdir(parents=True)
    (backup / "episodes.jsonl").write_text("".join(archived), encoding="utf-8")
    if episode_dir.exists():
        shutil.move(str(episode_dir), str(backup / "episodes"))

    if manifest.is_file():
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=OUTPUT, prefix="episodes_",
            suffix=".jsonl.tmp", delete=False,
        ) as stream:
            stream.writelines(kept)
            temporary = Path(stream.name)
        os.replace(temporary, manifest)
    print(f"RESET {args.category} archived_records={len(archived)} backup={backup}")


if __name__ == "__main__":
    main()

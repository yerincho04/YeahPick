"""Add reviewed two-choice answers to KD VQA object-recognition questions.

Paired labels are used only when both are distinct answers to the same kind of
question. Repeated-object pairs and pictograms use the curated alternatives.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "KD_VQA_dataset"
PAIR_CATEGORIES = {"celebrity", "living_world", "traffic"}

ATTRIBUTE_OBJECTS = {
    "bottle": "cup", "cup": "bottle",
    "cell phone": "laptop", "laptop": "cell phone",
    "handbag": "backpack", "sink": "bathtub",
    "bench": "chair", "chair": "bench", "couch": "armchair",
    "vase": "picture frame", "bowl": "plate",
    "fork": "spoon", "spoon": "fork",
}

COUNTING_OBJECTS = {
    "apple": "tomato", "carrot": "flower", "fish": "flower",
    "flower": "carrot", "tomato": "apple",
}

STATE_OBJECTS = {
    "bear": "lion", "toilet": "sink",
    "cup": "wine glass", "wine glass": "cup",
    "laptop": "monitor", "monitor": "laptop",
    "apple": "orange", "broccoli": "cauliflower",
    "cake": "pizza", "carrot": "parsnip",
    "orange": "apple", "pizza": "cake",
    "suitcase": "duffel bag",
    "bed": "couch", "bench": "chair", "chair": "bench", "couch": "bed",
    "vase": "lamp", "bowl": "plate",
    "fire hydrant": "parking meter", "knife": "fork",
    "motorcycle": "bicycle",
}

SYMMETRY_OBJECTS = {
    "beetle": "butterfly", "bow": "butterfly",
    "butterfly": "beetle", "crab": "beetle",
    "crown": "mask", "house": "tree", "leaf": "feather",
    "mask": "owl", "owl": "mask", "robot": "rocket",
    "rocket": "robot", "snowflake": "star", "tree": "leaf",
    "umbrella": "mushroom", "vase": "hourglass",
}

PUBLIC_INFO_OBJECTS = {
    "swimming": "fishing", "dog": "cat",
    "hospital": "school", "stars": "moon",
    "trash bin": "mailbox", "elevator": "escalator",
    "ice cream": "cupcake", "camera": "projector",
    "fire extinguisher": "oxygen cylinder",
    "mobile phone": "telephone handset", "telephone": "mobile phone",
    "fork and knife": "spoon and fork",
    "quiet gesture": "thumbs-up gesture",
    "human figures": "animal figures",
    "shower head": "water tap", "water fountain": "sink faucet",
    "bed": "chair", "picnic table": "dining table",
    "baby-changing station": "crib", "campfire": "barbecue grill",
    "shoreline": "mountain range", "trail": "river",
    "p": "H", "cash machine": "vending machine",
    "fuel pump": "air pump", "vending machine": "cash machine",
    "envelope": "package", "cross": "star",
    "wheelchair": "stroller", "book": "newspaper",
    "cigarette": "pencil", "falling rocks": "falling snow",
    "lifeguard": "swimmer", "walking pedestrian": "cyclist",
    "visitor center": "ticket booth", "tent": "gazebo",
    "baseball bat and ball": "racket and ball",
    "tennis racket": "badminton racket",
    "lighthouse": "watchtower", "steps": "ramp",
    "letter i": "question mark", "recycling arrows": "one-way arrows",
    "wireless signal": "sound waves", "fishing rod": "walking stick",
    "airplane": "helicopter", "bicycle": "motorcycle",
    "bus": "train", "car": "van", "train": "bus",
}


def object_item(row: dict) -> dict:
    return next(item for item in row["vqa"] if item["type"] == "Object recognition")


def normalized(value: str) -> str:
    return value.strip().rstrip(".").casefold()


def distractor(category: str, row: dict, other: dict) -> str:
    answer = object_item(row)["expected_answer"]
    other_answer = object_item(other)["expected_answer"]
    if category in PAIR_CATEGORIES:
        return other_answer
    if category == "symmetry" and normalized(answer) != normalized(other_answer):
        return other_answer
    if category == "counting" and normalized(answer) != normalized(other_answer):
        return other_answer
    if category == "attribute":
        return ATTRIBUTE_OBJECTS[normalized(answer)]
    if category == "counting":
        return COUNTING_OBJECTS[normalized(answer)]
    if category == "emotion":
        return "mannequin"
    if category == "public_info":
        return PUBLIC_INFO_OBJECTS[normalized(answer)]
    if category == "state":
        return STATE_OBJECTS[normalized(answer)]
    if category == "symmetry":
        return SYMMETRY_OBJECTS[normalized(answer)]
    if category == "time_clock":
        return "analog clock"
    raise ValueError(f"Unknown category: {category}")


def main() -> None:
    prepared = {}
    for path in sorted(ROOT.glob("*/vqa.jsonl")):
        category = path.parent.name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        by_pair = defaultdict(dict)
        for row in rows:
            by_pair[row["pair_id"]][row["focus_tile"].lower()] = row
        for row in rows:
            side = row["focus_tile"].lower()
            other_side = "right" if side == "left" else "left"
            other = by_pair[row["pair_id"]][other_side]
            item = object_item(row)
            if "choices" in item:
                raise ValueError(f"Object choices already set: {row['group_id']}")
            correct = item["expected_answer"]
            wrong = distractor(category, row, other)
            if normalized(correct) == normalized(wrong):
                raise ValueError(f"Identical object choices: {row['group_id']}")
            item["original_expected_answer"] = correct
            pair_bit = hashlib.sha256((row["pair_id"] + ":object").encode()).digest()[0] & 1
            correct_a = (pair_bit == 0) if side == "left" else (pair_bit != 0)
            choices = {"A": correct, "B": wrong} if correct_a else {"A": wrong, "B": correct}
            item["choices"] = choices
            item["question"] += (f"\nChoices:\nA. {choices['A']}\nB. {choices['B']}"
                                 "\nAnswer with A or B.")
            item["expected_answer"] = "A" if correct_a else "B"
        prepared[path] = rows
    # Persist only after every row has a distinct, category-appropriate choice.
    for path, rows in prepared.items():
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                        encoding="utf-8")
        print(f"UPDATED {path} object_questions={len(rows)}")


if __name__ == "__main__":
    main()

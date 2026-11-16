"""Apply reviewed, question-compatible choices to KD VQA attribute questions.

Distractors below were selected by reading the attribute questions in each
category. The opposite tile is used only for categories where both questions
ask the same attribute and both answers differ for every pair.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "KD_VQA_dataset"
PAIR_DISTRACTOR_CATEGORIES = {"attribute", "emotion", "symmetry", "time_clock"}

# Every state question already gives its two possible states. The pairs here
# select the opposing state named in that particular question.
STATE_OPTIONS = {
    "whole or cut": ("whole", "cut"),
    "wet or dry": ("wet", "dry"),
    "tidy or messy": ("tidy", "messy"),
    "empty or occupied": ("empty", "occupied"),
    "empty or full": ("empty", "full"),
    "cooked or raw": ("cooked", "raw"),
    "clean or dirty": ("clean", "dirty"),
    "switched on or off": ("on", "off"),
    "open or closed": ("open", "closed"),
}

COUNTING_DISTRACTORS = {"1": "2", "2": "3", "3": "4", "4": "3", "5": "6", "6": "5"}

# The following choices are curated within the attribute requested by each
# question: colors with colors, equipment with equipment, and so on.
CELEBRITY_DISTRACTORS = {
    "in a high bun": "in loose shoulder-length hair",
    "black sunglasses": "clear eyeglasses",
    "red": "blue",
    "orange": "yellow",
    "royal blue": "red",
    "yellow": "blue",
    "a test tube": "a glass beaker",
    "a bald head": "long hair",
    "a silver microphone": "a silver trophy",
    "leather": "denim",
    "an acoustic guitar": "an electric guitar",
    "a white mug": "a white bowl",
    "a tennis racket": "a baseball bat",
    "swimming goggles": "a whistle",
    "a fedora": "a baseball cap",
    "a black bow tie": "a black necktie",
    "bright green": "blue",
    "burgundy": "navy blue",
    "green": "purple",
    "purple": "green",
    "white": "black",
    "black": "white",
}

LIVING_WORLD_DISTRACTORS = {
    "translucent": "Opaque.",
    "they are spread in a large fan": "They hang straight down.",
    "the hind legs are much larger": "The forearms are much larger.",
    "two": "One.",
    "the neck": "The tail.",
    "white": "Black.",
    "red": "Brown.",
    "black": "White.",
    "gray": "Brown.",
    "a hard oval shell": "A thick coat of fur.",
    "a long trunk": "A long beak.",
    "pink": "White.",
    "orange": "Blue.",
    "a long mostly bare neck": "A short feathered neck.",
    "rows of pointed teeth": "A smooth toothless edge.",
    "narrow black and white stripes": "Large black and white spots.",
    "solid black spots": "Black stripes.",
    "alternating black and white rings": "An unbroken black band.",
    "black stripes": "Black spots.",
    "alternating reddish and pale rings": "Alternating black and white rings.",
    "suction cups": "Sharp claws.",
    "triangular": "Rounded.",
    "a hood": "A pair of horns.",
    "a thick mane": "A short crest.",
}

PUBLIC_INFO_DISTRACTORS = {
    "in a loop": "in a straight line",
    "two": "one",
    "open": "closed",
    "flames": "smoke",
    "a lens": "a button",
    "water droplets": "steam",
    "up and down": "left and right",
    "away from the starting point": "back toward the starting point",
    "tracks": "a road",
    "a line": "a flag",
    "an airplane": "a bird",
    "a cone": "a cup",
    "a leash": "a muzzle",
    "a hose": "a rope",
    "a nozzle": "a pressure gauge",
    "waste": "mail",
    "cash": "a receipt",
    "charging plug": "a fuel hose",
    "benches": "chairs",
    "slipping": "walking steadily",
    "walking": "running",
    "a bat": "a racket",
    "a ball": "a shuttlecock",
    "triangular": "rounded",
    "products": "empty shelves",
    "public visitor building": "a private house",
    "sand": "rocks",
    "telephone handset": "a smartphone",
    "i": "P",
    "water": "soap",
    "a cross": "a star",
    "a handset": "a microphone",
    "a mattress": "a wooden board",
    "a baby": "an adult",
    "a diagonal slash": "a check mark",
    "smoke": "water spray",
    "a rescue aid": "a surfboard",
    "curved arcs": "straight lines",
    "a circle": "a square",
    "a square": "a circle",
    "large passenger vehicle": "small passenger vehicle",
    "stars": "clouds",
    "in front of the mouth": "beside the ear",
    "at the top": "at the base",
    "downward": "upward",
}

TRAFFIC_DISTRACTORS = {
    "both edges move inward": "Only one edge moves inward.",
    "raised diagonally": "Lying flat.",
    "three": "Two.",
    "two": "Three.",
    "right": "Left.",
    "no; the center is blank": "Yes; a symbol is inside.",
    "a deer": "A cow.",
    "red, amber, green": "Green, amber, red.",
    "a cow": "A deer.",
    "a walking person": "A bicycle.",
    "rocks": "Snow.",
    "a cross-shaped junction": "A T-shaped junction.",
    "a u-turn arrow": "A left-turn arrow.",
    "two wavy skid marks": "Straight lane markings.",
    "30": "50.",
    "an exclamation mark": "A question mark.",
    "a horizontal bar": "A vertical bar.",
    "stop": "GO.",
    "down and left": "Down and right.",
    "down and right": "Down and left.",
    "left": "Right.",
    "straight up": "Left.",
    "a left-turn arrow": "A right-turn arrow.",
    "a right-turn arrow": "A left-turn arrow.",
    "red and black": "Blue and black.",
    "a motorcycle and a car": "A bicycle and a car.",
    "downward": "Upward.",
}


def attribute_item(row: dict) -> dict:
    return next(item for item in row["vqa"] if item["type"] == "Attribute recognition")


def normalized(text: str) -> str:
    return text.strip().rstrip(".").casefold()


def distractor(category: str, row: dict, other: dict) -> str:
    item = attribute_item(row)
    answer = item["expected_answer"]
    key = normalized(answer)
    if category in PAIR_DISTRACTOR_CATEGORIES:
        # Both sides ask exactly the same kind of attribute in these categories.
        return attribute_item(other)["expected_answer"]
    if category == "counting":
        return COUNTING_DISTRACTORS[key]
    if category == "state":
        question = item["question"].casefold()
        matches = [options for phrase, options in STATE_OPTIONS.items() if phrase in question]
        if len(matches) != 1 or key not in matches[0]:
            raise ValueError(f"State choices need review: {row['group_id']}")
        return next(value for value in matches[0] if value != key)
    if category == "celebrity":
        return CELEBRITY_DISTRACTORS[key]
    if category == "living_world":
        return LIVING_WORLD_DISTRACTORS[key]
    if category == "public_info":
        question = item["question"].casefold()
        if key == "water" and "surface" in question:
            return "sand"
        if key == "triangular" and "envelope" in question:
            return "rectangular"
        return PUBLIC_INFO_DISTRACTORS[key]
    if category == "traffic":
        return TRAFFIC_DISTRACTORS[key]
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
            opposite = "right" if side == "left" else "left"
            other = by_pair[row["pair_id"]][opposite]
            item = attribute_item(row)
            if "choices" in item:
                raise ValueError(f"Attribute choices already set: {row['group_id']}")
            wrong = distractor(category, row, other)
            right = item["expected_answer"]
            if normalized(wrong) == normalized(right):
                raise ValueError(f"Identical choices: {row['group_id']}")
            item["original_expected_answer"] = right
            if category == "public_info" and side not in item["question"].casefold():
                item["original_question"] = item["question"]
                item["question"] = f"Look at the sign on the {side}. " + item["question"]
            pair_bit = hashlib.sha256(row["pair_id"].encode()).digest()[0] & 1
            correct_a = (pair_bit == 0) if side == "left" else (pair_bit != 0)
            choices = {"A": right, "B": wrong} if correct_a else {"A": wrong, "B": right}
            item["choices"] = choices
            item["question"] += (f"\nChoices:\nA. {choices['A']}\nB. {choices['B']}"
                                 "\nAnswer with A or B.")
            item["expected_answer"] = "A" if correct_a else "B"
        prepared[path] = rows
    # Write only after every question has a curated, distinct distractor.
    for path, rows in prepared.items():
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                        encoding="utf-8")
        print(f"UPDATED {path} attribute_questions={len(rows)}")


if __name__ == "__main__":
    main()

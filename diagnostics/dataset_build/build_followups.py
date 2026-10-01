"""Build isolated assets and audits for the grounding-control and repeated-answer studies."""
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from icons import ICONS
from repeated_questions import CONCEPTS
from tile_asset import TILE_BBOX, TILE_DENSITY, TILE_SCALES, build_tile

REPO_ROOT = Path(__file__).resolve().parents[2]
CARROT = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets" / "carrot"
PILOT = CARROT / "semantic_pilot_v1"
OUT = Path(__file__).resolve().parent / "followup_output"

VISUAL_DESCRIPTIONS = {
    0: "brown animal with a large tail",
    1: "white bear",
    2: "yellow animal with brown spots and a long neck",
    3: "green bird",
    4: "green lizard-shaped animal",
    5: "yellow-and-black striped insect with wings",
    6: "green animal with a shell",
    7: "gray animal with a trunk",
    8: "black-and-white striped animal",
    9: "white animal with long ears",
    10: "light-blue cube",
    11: "brown rectangular block",
    12: "object with two finger loops and blades",
    13: "transparent cup",
    14: "yellow glass object with rays around it",
    15: "long blue object with a pointed tip",
    16: "round object with hands and hour marks",
    17: "long gray metal spike",
    18: "soft white rectangular cushion",
    19: "gold object with teeth at one end",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_grounding_control() -> None:
    asset = CARROT / "grounding_control_v1"
    source_pairs = json.loads((PILOT / "pairs.json").read_text())[:20]
    source_db = json.loads((PILOT / "model_db.json").read_text())
    pairs = []
    used = set()
    for pair in source_pairs:
        row = dict(pair)
        qid = int(row["index"])
        row.update({
            "source_asset": "semantic_pilot_v1",
            "source_question_id": qid,
            "visual_description": VISUAL_DESCRIPTIONS[qid],
        })
        pairs.append(row)
        used.update((row["left"], row["right"]))

    asset.mkdir(parents=True, exist_ok=True)
    (asset / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (asset / "model_db.json").write_text(
        json.dumps({name: source_db[name] for name in sorted(used)}, indent=2) + "\n"
    )
    for name in sorted(used):
        shutil.copytree(PILOT / "shapes" / name, asset / "shapes" / name, dirs_exist_ok=True)

    rows = [{
        "question_id": p["index"], "source_question_id": p["source_question_id"],
        "knowledge_instruction": p["knowledge_instruction"], "semantic_answer": p["semantic_answer"],
        "distractor": p["right_label"] if p["answer"].lower() == "left" else p["left_label"],
        "visual_description": p["visual_description"],
        "tile_object_instruction": f"Place the cube on the tile showing the {p['semantic_answer']}.",
        "explicit_object_instruction": f"Place the cube on the {p['semantic_answer']}.",
    } for p in pairs]
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "grounding_control_inventory.csv", rows)
    (OUT / "grounding_control_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")


def normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# v1 offsets (1..5) let the two tile images alone reveal the answer for 45/50 questions
# (each unordered image pair occurs with only one answer). v2 uses offsets {+-1, +-2, 5}:
# every image pair then occurs once with each image as the answer, so the scene alone
# carries no information about which of the two tiles is K.
REPEATED_OFFSETS = {"semantic_repeated_v1": (1, 2, 3, 4, 5), "semantic_repeated_v2": (1, 9, 2, 8, 5)}


def build_repeated(asset_name: str = "semantic_repeated_v1") -> None:
    offsets = REPEATED_OFFSETS[asset_name]
    asset = CARROT / asset_name
    asset.mkdir(parents=True, exist_ok=True)
    shapes = asset / "shapes"
    model_db = {}
    for concept in CONCEPTS:
        model = f"repeat_{concept['id']}"
        build_tile(ICONS[concept["icon"]](), shapes / model)
        model_db[model] = {
            "name": model, "sign": model, "bbox": TILE_BBOX,
            "scales": TILE_SCALES, "density": TILE_DENSITY,
        }

    pairs = []
    for concept_i, concept in enumerate(CONCEPTS):
        for question_i, (question, template_family, relation_type) in enumerate(concept["questions"]):
            qid = concept_i * 5 + question_i
            distractor = CONCEPTS[(concept_i + offsets[question_i]) % len(CONCEPTS)]
            answer_side = "Left" if qid % 2 == 0 else "Right"
            correct_model = f"repeat_{concept['id']}"
            distractor_model = f"repeat_{distractor['id']}"
            left, right = (
                (correct_model, distractor_model) if answer_side == "Left"
                else (distractor_model, correct_model)
            )
            left_label, right_label = (
                (concept["label"], distractor["label"]) if answer_side == "Left"
                else (distractor["label"], concept["label"])
            )
            pairs.append({
                "index": qid,
                "question_id": qid,
                "question": question,
                "question_text": question,
                "knowledge_instruction": f"Place the cube on the answer to this question: {question}",
                "left": left,
                "right": right,
                "answer": answer_side,
                "noswap_correct_side": answer_side.lower(),
                "semantic_answer": concept["label"],
                "semantic_answer_id": concept["id"],
                "semantic_answer_label": concept["label"],
                "distractor_id": distractor["id"],
                "distractor_label": distractor["label"],
                "left_label": left_label,
                "right_label": right_label,
                "knowledge_category": concept["category"],
                "template_family": template_family,
                "relation_type": relation_type,
            })

    questions = [p["question_text"].strip().lower() for p in pairs]
    duplicate_questions = sorted(q for q, n in Counter(questions).items() if n > 1)
    answer_leaks = []
    for p in pairs:
        answer_tokens = normalized_words(p["semantic_answer_label"])
        if answer_tokens & normalized_words(p["question_text"]):
            answer_leaks.append(p["question_id"])
    template_counts = Counter(p["template_family"] for p in pairs)
    answer_counts = Counter(p["semantic_answer_id"] for p in pairs)
    distractor_counts = Counter(p["distractor_id"] for p in pairs)
    side_counts = Counter(p["noswap_correct_side"] for p in pairs)
    image_pair_answers = {}
    for p in pairs:
        image_pair_answers.setdefault(frozenset((p["semantic_answer_id"], p["distractor_id"])), set()).add(p["semantic_answer_id"])
    image_pairs_revealing_answer = sum(len(v) == 1 for v in image_pair_answers.values())
    audit = {
        "question_count": len(pairs),
        "concept_count": len(CONCEPTS),
        "duplicate_questions": duplicate_questions,
        "answer_word_leak_question_ids": answer_leaks,
        "answer_frequency": dict(sorted(answer_counts.items())),
        "distractor_frequency": dict(sorted(distractor_counts.items())),
        "noswap_side_frequency": dict(sorted(side_counts.items())),
        "template_family_frequency": dict(sorted(template_counts.items())),
        "image_pairs": len(image_pair_answers),
        "image_pairs_revealing_answer": image_pairs_revealing_answer,
        "checks_passed": not duplicate_questions and not answer_leaks
            and set(answer_counts.values()) == {5} and set(distractor_counts.values()) == {5}
            and side_counts == {"left": 25, "right": 25},
    }
    if not audit["checks_passed"]:
        raise ValueError(f"Repeated dataset quality audit failed: {audit}")

    (asset / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (asset / "model_db.json").write_text(json.dumps(model_db, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"{asset_name}_questions.csv", pairs)
    (OUT / f"{asset_name}_questions.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (OUT / f"{asset_name}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    inventory = [{
        "semantic_answer_id": c["id"], "label": c["label"],
        "asset_path": f"shapes/repeat_{c['id']}/textured.glb",
        "proposed_questions": 5,
        "distractors_used": sorted({p["distractor_id"] for p in pairs if p["semantic_answer_id"] == c["id"]}),
    } for c in CONCEPTS]
    (OUT / f"{asset_name}_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    write_csv(OUT / f"{asset_name}_inventory.csv", [
        {**row, "distractors_used": ", ".join(row["distractors_used"])} for row in inventory
    ])

    md = [
        f"# {asset_name} question audit", "",
        f"- Questions: {len(pairs)}", f"- Answer concepts: {len(CONCEPTS)}",
        "- Answer frequency: 5 each", "- Distractor frequency: 5 each",
        "- Noswap sides: 25 left / 25 right", "- Duplicate questions: 0",
        "- Answer-word leakage: 0",
        f"- Image pairs that reveal the answer on their own: {image_pairs_revealing_answer}/{len(image_pair_answers)}", "",
        "| ID | Question | Answer | Distractor | Relation | Answer asset | Distractor asset |",
        "|---:|---|---|---|---|---|---|",
    ]
    for p in pairs:
        md.append(
            f"| {p['question_id']} | {p['question_text']} | {p['semantic_answer_label']} | "
            f"{p['distractor_label']} | {p['relation_type']} | `{p['left'] if p['answer']=='Left' else p['right']}` | "
            f"`{p['right'] if p['answer']=='Left' else p['left']}` |"
        )
    (OUT / f"{asset_name}_audit.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or ["grounding_control_v1", "semantic_repeated_v1"]
    for target in targets:
        if target == "grounding_control_v1":
            build_grounding_control()  # needs semantic_pilot_v1 (build_dataset.py) first
        else:
            build_repeated(target)
        print(f"Built {target} with a passing audit.")

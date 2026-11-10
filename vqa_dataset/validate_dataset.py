#!/usr/bin/env python3
"""Validate this VQA package; run with Python 3 and Pillow after unzipping.

Usage: python validate_dataset.py [DATASET_ROOT]

The default root is the folder containing this script. A machine-readable
validation_report.json is always written there. This validates structural
consistency, provenance declarations, asset linkage, and answer bookkeeping;
it does not independently prove the visual or factual correctness of a label.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

from PIL import Image, ImageChops, ImageStat


CATEGORIES = ("celebrity", "living_world", "traffic")
QA_TYPES = (
    "Object recognition", "Attribute recognition", "Semantic reasoning",
    "Visual grounding", "Counterfactual grounding",
)
GROUP_KEYS = {
    "group_id", "category", "pair_id", "focus_tile", "focus_image_id",
    "images", "knowledge", "vqa",
}
KNOWLEDGE_KEYS = {
    "source_question_id", "source_instruction_id", "source_instruction",
    "source_left_image_id", "source_right_image_id", "instruction",
    "expected_answer", "template_family", "comparison_image_replaced",
}
PAIR_KEYS = {"pair_id", "category", "images", "pair_image_path", "knowledge"}
CSV_FIELDS = [
    "category", "pair_id", "group_id", "focus_tile", "left_image",
    "right_image", "type", "question", "expected_answer",
    "knowledge_instruction", "source_question_id",
]


class Validator:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checks = 0
        self.errors: list[str] = []
        self.category_counts: dict = {}

    def check(self, condition, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def json_file(self, relative: str):
        try:
            return json.loads((self.root / relative).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            self.check(False, f"Cannot read {relative}: {exc}")
            return None

    def jsonl_file(self, relative: str) -> list[dict]:
        try:
            lines = (self.root / relative).read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            self.check(False, f"Cannot read {relative}: {exc}")
            return []
        records = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                self.check(False, f"{relative}:{line_number}: blank JSONL record")
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                self.check(False, f"{relative}:{line_number}: invalid JSON: {exc}")
                continue
            if self.check(isinstance(value, dict), f"{relative}:{line_number}: record must be an object"):
                records.append(value)
        return records

    def path(self, relative, context: str) -> Path | None:
        if not self.check(isinstance(relative, str) and bool(relative), f"{context}: missing path"):
            return None
        pure = PurePosixPath(relative)
        safe = not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative
        if not self.check(safe, f"{context}: path must be package-relative: {relative!r}"):
            return None
        path = self.root.joinpath(*pure.parts)
        if not self.check(path.is_file(), f"{context}: missing asset {relative}"):
            return None
        return path

    def image(self, path: Path | None, size: tuple[int, int], image_format: str):
        if path is None:
            return None
        try:
            with Image.open(path) as image:
                self.check(image.size == size, f"{path.name}: expected size {size}, got {image.size}")
                self.check(image.format == image_format, f"{path.name}: expected {image_format}, got {image.format}")
                image.load()
                return image.convert("RGB")
        except (OSError, ValueError) as exc:
            self.check(False, f"{path.name}: invalid image: {exc}")
            return None

    def image_refs(self, value, category: str, context: str, manifest: dict) -> bool:
        valid = self.check(isinstance(value, dict) and set(value) == {"left", "right"},
                           f"{context}: images must have exactly left/right")
        if not valid:
            return False
        for side in ("left", "right"):
            ref = value[side]
            if not self.check(isinstance(ref, dict) and set(ref) == {"image_id", "path"},
                              f"{context}: invalid {side} image reference"):
                valid = False
                continue
            iid = ref["image_id"]
            valid &= self.check(iid in manifest, f"{context}: unknown image ID {iid}")
            self.check(ref["path"] == f"{category}/images/{iid}.png", f"{context}: {side} path/ID mismatch")
            self.path(ref["path"], context)
        if valid:
            self.check(value["left"]["image_id"] != value["right"]["image_id"], f"{context}: self-pair")
        return bool(valid)

    def category(self, category: str):
        manifest_rows = self.json_file(f"{category}/image_manifest.json")
        if not self.check(isinstance(manifest_rows, list), f"{category}: manifest must be a list"):
            manifest_rows = []
        self.check(len(manifest_rows) == 30, f"{category}: expected 30 manifest images")
        manifest = {r.get("image_id"): r for r in manifest_rows if isinstance(r, dict)}
        expected_iids = {f"{category}_{i:02d}" for i in range(1, 31)}
        self.check(len(manifest) == len(manifest_rows), f"{category}: duplicate or invalid manifest records")
        self.check(set(manifest) == expected_iids, f"{category}: expected image IDs 01..30")
        self.check(len(list((self.root / category / "images").glob("*.png"))) == 30,
                   f"{category}: images folder must contain 30 PNGs")
        assets = {}
        hashes = []
        for iid, item in manifest.items():
            context = f"{category}/{iid}"
            self.check(bool(item.get("label")), f"{context}: missing label")
            self.check(bool(item.get("prompt")), f"{context}: missing generation prompt")
            self.check(bool(item.get("semantic_clue")), f"{context}: missing semantic clue")
            self.check(bool(item.get("attribute_question")) and bool(item.get("attribute_answer")),
                       f"{context}: missing visual attribute question/answer")
            self.check(item.get("synthetic") is True, f"{context}: synthetic provenance must be explicit")
            self.check(item.get("generation_method") == "built_in_image_generation", f"{context}: generation provenance mismatch")
            self.check(item.get("path") == f"{category}/images/{iid}.png", f"{context}: manifest path mismatch")
            path = self.path(item.get("path"), context)
            assets[iid] = self.image(path, (512, 512), "PNG")
            if path:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.append(digest)
                self.check(item.get("sha256") == digest, f"{context}: SHA-256 mismatch")
        self.check(len(set(hashes)) == len(hashes), f"{category}: duplicate image files")

        sources = self.jsonl_file(f"{category}/{category}_source_questions_200.jsonl")
        self.check(len(sources) == 200, f"{category}: expected 200 source instructions")
        sources_by_id = {r.get("source_question_id"): r for r in sources}
        self.check(len(sources_by_id) == len(sources), f"{category}: duplicate source question IDs")
        self.check(set(sources_by_id) == set(range(1, 201)), f"{category}: source question IDs must cover 1..200")
        selected = {}
        for source in sources:
            sid = source.get("source_question_id")
            context = f"{category}/source/{sid}"
            self.check(type(sid) is int, f"{context}: source ID must be an integer")
            self.check(source.get("category") == category, f"{context}: category mismatch")
            self.check(source.get("source_instruction_id") == f"{category}_{sid:03d}_knowledge" if type(sid) is int else False,
                       f"{context}: instruction ID mismatch")
            self.check(source.get("provenance") == "newly_authored_for_this_delivery",
                       f"{context}: source must declare newly authored provenance")
            self.check(type(source.get("selected_for_vqa")) is bool, f"{context}: selection flag must be boolean")
            self.check(bool(source.get("instruction")), f"{context}: empty source instruction")
            valid_refs = self.image_refs(source.get("images"), category, context, manifest)
            answer = source.get("expected_answer")
            self.check(answer in ("Left", "Right"), f"{context}: invalid source answer")
            if valid_refs and answer in ("Left", "Right"):
                self.check(source.get("target_image_id") == source["images"][answer.lower()]["image_id"],
                           f"{context}: target/answer mismatch")
            if source.get("selected_for_vqa") is True:
                selected[sid] = source
        self.check(len(selected) == 100, f"{category}: expected exactly 100 selected source questions")

        pairs = self.jsonl_file(f"{category}/{category}_pairs.jsonl")
        groups = self.jsonl_file(f"{category}/{category}_vqa.jsonl")
        self.check(len(pairs) == 100, f"{category}: expected 100 pair records")
        self.check(len(groups) == 200, f"{category}: expected 200 VQA groups")
        self.check(len(list((self.root / category / "pairs").glob("*.jpg"))) == 100,
                   f"{category}: pairs folder must contain 100 JPEGs")
        pair_by_id = {p.get("pair_id"): p for p in pairs}
        self.check(len(pair_by_id) == len(pairs), f"{category}: duplicate pair IDs")
        self.check(set(pair_by_id) == {f"{category}_{sid:03d}" for sid in selected},
                   f"{category}: pair IDs do not match selected source IDs")
        degrees = collections.Counter()
        unordered = set()
        knowledge_answers = collections.Counter()
        for pair in pairs:
            pid = pair.get("pair_id")
            context = str(pid)
            if not self.check(set(pair) == PAIR_KEYS, f"{context}: pair schema mismatch"):
                continue
            self.check(pair["category"] == category, f"{context}: pair category mismatch")
            refs_valid = self.image_refs(pair["images"], category, context, manifest)
            knowledge = pair["knowledge"]
            if not self.check(isinstance(knowledge, dict) and set(knowledge) == KNOWLEDGE_KEYS,
                              f"{context}: knowledge schema mismatch"):
                continue
            sid = knowledge["source_question_id"]
            self.check(sid in selected, f"{context}: source question is not selected")
            source = selected.get(sid)
            if source:
                self.check(pair["images"] == source["images"], f"{context}: pair images differ from source")
                self.check(knowledge["source_instruction_id"] == source["source_instruction_id"], f"{context}: source instruction ID mismatch")
                self.check(knowledge["source_instruction"] == source["instruction"], f"{context}: source instruction mismatch")
                self.check(knowledge["instruction"] == source["instruction"], f"{context}: instruction mismatch")
                self.check(knowledge["expected_answer"] == source["expected_answer"], f"{context}: knowledge answer/source mismatch")
                self.check(knowledge["template_family"] == source["template_family"], f"{context}: template family mismatch")
                for side in ("left", "right"):
                    self.check(knowledge[f"source_{side}_image_id"] == source["images"][side]["image_id"],
                               f"{context}: source {side} image ID mismatch")
            self.check(knowledge["comparison_image_replaced"] is False,
                       f"{context}: authored source pair is used directly, so replacement flag must be false")
            knowledge_answers[knowledge["expected_answer"]] += 1
            self.check(pair["pair_image_path"] == f"{category}/pairs/{pid}.jpg", f"{context}: pair path mismatch")
            pair_image = self.image(self.path(pair["pair_image_path"], context), (1024, 512), "JPEG")
            if refs_valid:
                ids = [pair["images"][s]["image_id"] for s in ("left", "right")]
                degrees.update(ids)
                key = tuple(sorted(ids))
                self.check(key not in unordered, f"{context}: duplicate unordered image pair")
                unordered.add(key)
                if pair_image is not None and pair_image.size == (1024, 512):
                    for side, x in (("left", 0), ("right", 512)):
                        expected = assets.get(pair["images"][side]["image_id"])
                        if expected is not None and expected.size == (512, 512):
                            difference = ImageChops.difference(pair_image.crop((x, 0, x + 512, 512)), expected)
                            mae = sum(ImageStat.Stat(difference).mean) / 3
                            self.check(mae < 8.0, f"{context}: rendered {side} tile differs from linked source (MAE {mae:.2f})")
        self.check(len(unordered) == 100, f"{category}: expected 100 unique unordered image pairs")
        self.check(set(degrees) == expected_iids, f"{category}: not all images occur in pairs")
        self.check(all(v in (6, 7) for v in degrees.values()), f"{category}: every image must occur in 6 or 7 pairs")
        self.check(knowledge_answers == {"Left": 50, "Right": 50}, f"{category}: pair knowledge answers must balance 50/50")

        group_ids = set()
        partner_sides = collections.defaultdict(list)
        qa_counts = collections.Counter()
        grounding_counts = collections.Counter()
        cf_counts = collections.Counter()
        review_rows = []
        for group in groups:
            gid = group.get("group_id")
            context = str(gid)
            self.check(gid not in group_ids, f"{context}: duplicate group ID")
            group_ids.add(gid)
            if not self.check(set(group) == GROUP_KEYS, f"{context}: group schema mismatch"):
                continue
            pid, side = group["pair_id"], group["focus_tile"]
            self.check(group["category"] == category, f"{context}: group category mismatch")
            side_valid = self.check(side in ("left", "right"), f"{context}: invalid focus side")
            self.check(gid == f"{pid}_{side}", f"{context}: group ID/focus mismatch")
            partner_sides[pid].append(side)
            pair = pair_by_id.get(pid)
            if self.check(pair is not None, f"{context}: missing partner pair record"):
                self.check(group["images"] == pair["images"], f"{context}: group/pair images mismatch")
                self.check(group["knowledge"] == pair["knowledge"], f"{context}: group/pair knowledge mismatch")
            if not side_valid or not isinstance(group["images"], dict) or side not in group["images"]:
                continue
            ref = group["images"][side]
            if not isinstance(ref, dict):
                continue
            iid = ref.get("image_id")
            self.check(group["focus_image_id"] == iid, f"{context}: focus image linkage mismatch")
            item = manifest.get(iid)
            qas = group["vqa"]
            if not self.check(isinstance(qas, list) and len(qas) == 5, f"{context}: expected five QA records"):
                continue
            if not self.check(all(isinstance(qa, dict) and set(qa) == {"type", "question", "expected_answer"} for qa in qas),
                              f"{context}: QA schema mismatch"):
                continue
            self.check(tuple(qa["type"] for qa in qas) == QA_TYPES, f"{context}: QA types/order mismatch")
            self.check(all(isinstance(qa["question"], str) and qa["question"].strip() and
                           isinstance(qa["expected_answer"], str) and qa["expected_answer"].strip() for qa in qas),
                       f"{context}: empty or non-string question/answer")
            if item:
                self.check(qas[0]["expected_answer"] == item["label"], f"{context}: object answer/manifest mismatch")
                self.check(qas[1]["expected_answer"] == item["attribute_answer"], f"{context}: attribute answer/manifest mismatch")
                self.check(qas[1]["question"] == item["attribute_question"].format(side=side), f"{context}: attribute question/focus mismatch")
                self.check(qas[2]["expected_answer"] == item["label"], f"{context}: semantic answer/manifest mismatch")
            self.check(qas[3]["expected_answer"] == side.title(), f"{context}: visual grounding must point to focus")
            cf_question = qas[4]["question"].lower()
            if "switched places" in cf_question:
                expected_cf = "Right" if side == "left" else "Left"
            elif "blue border" in cf_question and "without moving" in cf_question:
                expected_cf = side.title()
            else:
                expected_cf = None
                self.check(False, f"{context}: unknown counterfactual transformation")
            self.check(qas[4]["expected_answer"] == expected_cf, f"{context}: incorrect counterfactual position")
            grounding_counts[qas[3]["expected_answer"]] += 1
            cf_counts[qas[4]["expected_answer"]] += 1
            for qa in qas:
                qa_counts[qa["type"]] += 1
                review_rows.append(dict(
                    category=category, pair_id=pid, group_id=gid, focus_tile=side,
                    left_image=group["images"]["left"]["image_id"],
                    right_image=group["images"]["right"]["image_id"],
                    type=qa["type"], question=qa["question"], expected_answer=qa["expected_answer"],
                    knowledge_instruction=group["knowledge"]["instruction"],
                    source_question_id=str(group["knowledge"]["source_question_id"]),
                ))
        self.check(set(partner_sides) == set(pair_by_id), f"{category}: group/pair membership mismatch")
        self.check(all(sorted(sides) == ["left", "right"] for sides in partner_sides.values()),
                   f"{category}: each pair needs exactly one left and one right group")
        self.check(qa_counts == {t: 200 for t in QA_TYPES}, f"{category}: each QA type must occur 200 times")
        self.check(grounding_counts == {"Left": 100, "Right": 100}, f"{category}: grounding answers must balance 100/100")
        self.check(cf_counts == {"Left": 100, "Right": 100}, f"{category}: CF answers must balance 100/100")
        self.category_counts[category] = dict(images=len(manifest), source_questions=len(sources),
                                              selected_sources=len(selected), pairs=len(pairs),
                                              groups=len(groups), questions=sum(qa_counts.values()))
        return groups, pairs, review_rows

    def run(self) -> dict:
        all_groups, all_pairs, all_review = [], [], []
        for category in CATEGORIES:
            groups, pairs, review = self.category(category)
            all_groups.extend(groups)
            all_pairs.extend(pairs)
            all_review.extend(review)
        combined_groups = self.jsonl_file("vqa_all.jsonl")
        combined_pairs = self.jsonl_file("pairs_all.jsonl")
        self.check(len(combined_groups) == 600, "Combined VQA: expected 600 groups")
        self.check(len(combined_pairs) == 300, "Combined pairs: expected 300 pairs")
        self.check(combined_groups == all_groups, "Combined VQA differs from concatenated category files")
        self.check(combined_pairs == all_pairs, "Combined pairs differ from concatenated category files")
        self.check(len({g.get("group_id") for g in combined_groups}) == len(combined_groups), "Combined VQA has duplicate group IDs")
        self.check(len({p.get("pair_id") for p in combined_pairs}) == len(combined_pairs), "Combined pairs have duplicate pair IDs")
        try:
            with (self.root / "vqa_questions_review.csv").open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                self.check(reader.fieldnames == CSV_FIELDS, "Review CSV: expected original 11 columns and order")
                review = list(reader)
            self.check(len(review) == 3000, "Review CSV: expected 3,000 rows")
            self.check(review == all_review, "Review CSV differs from flattened JSONL QA records")
        except OSError as exc:
            self.check(False, f"Cannot read review CSV: {exc}")
        metadata = self.json_file("dataset_metadata.json")
        if isinstance(metadata, dict):
            self.check(metadata.get("categories") == list(CATEGORIES), "Metadata categories mismatch")
            for key, expected in (("total_images", 90), ("total_pairs", 300), ("total_groups", 600), ("total_questions", 3000)):
                self.check(metadata.get(key) == expected, f"Metadata {key}: expected {expected}")
        else:
            self.check(False, "Missing or invalid dataset metadata")
        return dict(
            status="passed" if not self.errors else "failed",
            checks_performed=self.checks, error_count=len(self.errors),
            categories=self.category_counts, errors=self.errors,
            scope="Structural consistency, asset integrity, source provenance declarations, and answer bookkeeping.",
            limitations=[
                "Visual likeness, visible attributes, sign accuracy, and external factual claims require human review.",
                "The 200-question source pools are newly authored for this delivery, not original pre-existing questions.",
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    validator = Validator(args.root)
    try:
        report = validator.run()
    except Exception as exc:
        validator.check(False, f"Validation could not finish: {type(exc).__name__}: {exc}")
        report = dict(status="failed", checks_performed=validator.checks,
                      error_count=len(validator.errors), categories=validator.category_counts,
                      errors=validator.errors)
    report_path = validator.root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=report["checks_performed"],
                          errors=report["error_count"], report=str(report_path))))
    for error in report["errors"][:20]:
        print("ERROR:", error, file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

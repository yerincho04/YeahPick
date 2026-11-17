# KD VQA dataset

Ten categories collected for VLM-to-VLA knowledge distillation. Each category has 100 image pairs and 200 JSONL records (left focus and right focus for every pair), for 1,000 pairs and 2,000 records in total.

| Category | Original folder |
| --- | --- |
| `counting`, `symmetry` | `VQA_dataset` |
| `celebrity`, `living_world`, `traffic` | `vqa_dataset_1` |
| `state`, `attribute`, `emotion` | `VQA_dataset_v2` |
| `public_info`, `time_clock` | `VQA_dataset_public_info_time_clock` |

Use `<category>/vqa.jsonl` for training. Each JSONL row contains `pair_id`, `focus_tile`, `images`, one pair-level `knowledge` instruction and five `vqa` question/answer items. Resolve `images.left.path` and `images.right.path` relative to **this directory**, not relative to the category directory. The paired JPEGs and source metadata from `vqa_dataset_1` are retained inside their category folders.

VQA answer format: `Visual grounding` and `Counterfactual grounding` ask for `Left` or `Right` explicitly. `Semantic reasoning` presents A (the one on the left) and B (the one on the right), with `expected_answer` set to the matching letter. `Object recognition` and `Attribute recognition` each present two choices and use A/B answers. Their choices are recorded in `choices`; previous answers are preserved as `original_expected_answer` (and a changed question as `original_question`). The pair-level `knowledge` instruction and answer are unchanged. The curated distractors are documented in the project-root `scripts/curate_kd_vqa_object.py` and `scripts/curate_kd_vqa_attribute.py`.

The original dataset folders remain intact. `emotion_kdef` is excluded because its image files are absent (only `images.csv` and a JSONL file were present). There is no train/validation/test split in this collection.

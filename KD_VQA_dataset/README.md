# KD VQA dataset

Ten categories collected for VLM-to-VLA knowledge distillation. Each category has 100 image pairs and 200 JSONL records (left focus and right focus for every pair), for 1,000 pairs and 2,000 records in total.

| Category | Original folder |
| --- | --- |
| `counting`, `symmetry` | `VQA_dataset` |
| `celebrity`, `living_world`, `traffic` | `vqa_dataset_1` |
| `state`, `attribute`, `emotion` | `VQA_dataset_v2` |
| `public_info`, `time_clock` | `VQA_dataset_public_info_time_clock` |

Use `<category>/vqa.jsonl` for training. Each JSONL row contains `pair_id`, `focus_tile`, `images`, one pair-level `knowledge` instruction and five `vqa` question/answer items. Resolve `images.left.path` and `images.right.path` relative to **this directory**, not relative to the category directory. The paired JPEGs and source metadata from `vqa_dataset_1` are retained inside their category folders.

The original dataset folders remain intact. `emotion_kdef` is excluded because its image files are absent (only `images.csv` and a JSONL file were present). There is no train/validation/test split in this collection.

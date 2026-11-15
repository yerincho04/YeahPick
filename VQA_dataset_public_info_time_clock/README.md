# Public information signs and digital clocks VQA dataset

This dataset follows the pair and focus structure of `VQA_dataset_v2`. Each category has 100 distinct image pairs and 200 JSONL records: one left-focus and one right-focus record per pair.

| Category | Pairs | Images | Image format | Knowledge answers |
| --- | ---: | ---: | --- | --- |
| `public_info` | 100 | 50 | SVG | Left 50, Right 50 |
| `time_clock` | 100 | 48 | PNG | Left 50, Right 50 |

The 50 public-information signs each appear in four pairs and each serves as the Knowledge target twice. The 48 clocks show every half hour in a day; each appears in four or five pairs. No unordered image pair is repeated within a category.

## Files

- `public_info_vqa.jsonl`, `time_clock_vqa.jsonl`: one record per focus tile. Each record has the pair metadata, a pair-level `knowledge` instruction, and five VQA items: Object recognition, Attribute recognition, Semantic reasoning, Visual grounding, and Counterfactual grounding.
- `public_info/images/`, `time_clock/images/`: images referenced by the JSONL paths, relative to this directory.
- `public_info/images.csv`: sign IDs, paths, labels, and source file/URL. The signs were selected from the National Park Service symbol library (`https://github.com/nationalparkservice/symbol-library`).
- `time_clock/images.csv`: clock IDs, paths, and displayed times. The PNGs were made with `time_clock_assets/digital_clock_png.py`.
- `pairs.json`: the same pair-oriented structure as V2, including both focus groups and their five questions.
- `audit.json`: pair count, distinct image count, Knowledge family distribution, and Knowledge answer-side counts, counted once per pair.

The Knowledge instruction and answer are identical in the two focus records of a pair. The Semantic reasoning question instead targets that record's focus image. Counterfactual grounding copies the opposite focus record's Visual grounding question and answer exactly.

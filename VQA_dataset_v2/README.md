# Photo VQA dataset v2: state, attribute, emotion

Same JSONL group format as VQA_dataset (hyerin branch): one row per tile, two rows per pair, five questions per row.

| Category | Pairs | Photos | Object families | Source |
| --- | ---: | ---: | --- | --- |
| state | 100 | 153 | 23 object families (open/closed, on/off, empty/full, empty/occupied, cut/whole, cooked/raw, clean/dirty, tidy/messy, wet/dry) | OVAD (COCO val2017) + Wikimedia Commons |
| attribute (material) | 100 | 91 | 13 objects (chair, cup, bowl, bench, bottle, couch, handbag, sink, vase, spoon, laptop, fork, cell phone) over wood, metal, plastic, glass, fabric, leather, ceramic, paper, stone | OVAD (COCO val2017) |
| emotion | 100 | 49 | happy, surprised, angry, sad, asleep; 10 pairs per emotion pair; faces pre-screened with a ViT expression classifier (trpakov/vit-face-expression) then picked by eye | COCO val2017 / Visual Genome face crops + Wikimedia Commons |

- Both tiles of a pair are the same kind of object and differ only in the tested property.
- State and material photos are objects labelled by OVAD's annotators with the property positive and its opposite explicitly negative (state) or with exactly one material.
- Every kept photo was checked by eye on contact sheets (`diagnostics/dataset_build/vqa_v2/review.json`). Emotion uses OpenCV face crops.
- Sleepy uses the 10 hand-checked photos of v1, since face detection finds few sleeping faces.
- A photo is used in at most 2 pairs for state, 3 for material and 4 for emotion, never twice with the same partner.
- Questions come from `diagnostics/dataset_build/vqa_v2/questions.py`:
  - State facts are written per object.
  - Material facts each say which materials they rule out.
  - The counterfactual question uses the other tile's fact, so its answer is always the other side.
- Licences:
  - COCO images: Flickr licences as listed in COCO.
  - OVAD annotations: research use.
  - Visual Genome: CC BY 4.0.
  - VAW: Adobe Research License (non-commercial research).
- Files:
  - `<category>_vqa.jsonl`
  - `<category>/images/`
  - `<category>/images.csv` (source ids)
  - `pairs.json` (all pairs, the format the editor uses)
  - `audit.json`
- Rebuild:
  1. `python ovad_candidates.py ...`
  2. `python emotion_candidates.py ...`
  3. Review the sheets.
  4. `python build_v2.py --cand <dir> --out VQA_dataset_v2`
- Review and edits happen in the editor artifact. Edits there are not yet merged into these files.

- Object recognition names the kind of thing asked for ("What utensil is shown on the left tile?"), so other things in the photo are not also correct answers.

## emotion_kdef (separate, optional)
100 pairs of posed studio faces from KDEF (Lundqvist, Flykt & Ohman 1998): happy, angry, sad, disgusted, neutral
(10 pairs per emotion pair; 53 faces, background removed, on gray). Same JSONL format and question scheme as `emotion`.
KDEF may not be redistributed, so `emotion_kdef/images/` is git-ignored: rebuild it from your own KDEF download with
`diagnostics/dataset_build/vqa_v2/kdef_candidates.py` + `build_v2.py`.

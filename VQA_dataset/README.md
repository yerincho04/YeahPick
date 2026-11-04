# Counting and symmetry VQA dataset

Each JSONL row contains one five-question VQA group focused on one tile. Each pair appears in two rows with the same left and right images and a different `focus_tile`. Image paths are relative to this directory. Show both images in the stored left-to-right order. The `knowledge` block records provenance and the associated instruction. Use the VQA `question` as the input prompt and `expected_answer` as the label.

| Category | Images | Pairs | Original pairs retained | Comparison tiles replaced | Groups | VQA items |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| counting | 30 | 100 | 75 | 25 | 200 | 1,000 |
| symmetry | 30 | 100 | 15 | 85 | 200 | 1,000 |

Pair uniqueness ignores left-right order. Every pair uses a distinct source knowledge question within its category. For replaced comparisons, the source target image and its tile position are preserved. The adjusted instruction describes the comparison by count or symmetry, with the original instruction and image IDs retained in separate fields. Each category has 50 source targets on the left and 50 on the right.

Counterfactual questions use hypothetical changes to the counting scenario, tile positions, or the orientation of both designs. Grounding labels refer to the scene described in the question.

The files contain English questions and answers. Images are reused across pairs, and questions use templates derived from seven source families per category. No train/test split is provided.

[Source assets and knowledge questions at commit f072021](https://github.com/yerincho04/YeahPick/tree/f07202156cf1dc6f40311ed13988cd303bc30bad/ManiSkill/mani_skill/assets/carrot)

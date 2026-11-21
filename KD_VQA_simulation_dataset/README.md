# KD VQA dataset with simulation images

This is a copy of `KD_VQA_dataset` with one **full robot simulation image** per pair. It contains 10 categories, 1,000 pairs, and 2,000 JSONL rows (left and right focus). Each row keeps its original knowledge instruction and five VQA questions and answers.

The `image` field in `<category>/vqa.jsonl` points to `<category>/simulation_images/<pair_id>.jpg`, relative to this directory. Each image is frame `0000.jpg` from the successful `action_data/episodes/<category>/<pair_id>/original/seed_0` episode, collected with 22 cm tiles at the revised tile positions. Both focus rows for a pair point to the same scene image. The left/right questions and answers therefore retain their original meaning.

The original `images.left` and `images.right` fields and source image files are kept as provenance and for building tile assets. They are **not** the VQA scene image in this copy. Action arrays and later episode frames are not included.

`simulation_joint_training/prepare.py` reads the `image` field when preparing VQA examples. Copy this dataset and the matching `action_data` into that training folder's `input` directory before running preparation.

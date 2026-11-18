# KD VQA dataset with simulation images

This is a copy of `KD_VQA_dataset` with one **full robot simulation image** per pair. It contains 10 categories, 1,000 pairs, and 2,000 JSONL rows (left and right focus). Each row keeps its original knowledge instruction and five VQA questions and answers.

The new `image` field in `<category>/vqa.jsonl` points to `<category>/simulation_images/<pair_id>.jpg`, relative to this directory. It is frame `0000.jpg` from the successful `action_data` episode with `original` tile layout and seed 0. Both focus rows for a pair point to the same scene image. The left/right questions and answers therefore retain their original meaning.

The original `images.left` and `images.right` fields and source image files are kept as provenance and for building tile assets. They are **not** the VQA scene image in this copy. Action arrays and later episode frames are not included.

Training code has not been changed. Its current VQA preparation still builds a side-by-side image from `images.left` and `images.right`; it must be updated to use `image` before training on these simulation images.

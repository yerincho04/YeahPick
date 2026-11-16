# Semantic KD training

One OpenVLA LoRA adapter is trained with three signals: action cross-entropy on successful scripted demonstrations, gold-answer cross-entropy on **all five** VQA types, and gold-answer cross-entropy for choosing the correct tile in the robot camera image. A frozen Prismatic VLM also supplies a representation target for a QA sample when its generated answer matches the gold answer. Teacher errors never remove the sample from gold-answer training.

## Inputs

Copy the contents of `KD_VQA_dataset/` to `input/dataset/`, and the contents of `scripted_action_collection/output/` to `input/action_data/`:

```text
input/dataset/<category>/vqa.jsonl
input/dataset/<category>/images/...
input/action_data/episodes.jsonl
input/action_data/episodes/<category>/<pair_id>/<layout>/seed_<n>/actions.npy
input/action_data/episodes/<category>/<pair_id>/<layout>/seed_<n>/frames/0000.jpg ...
```

`prepare` joins the files by category and pair ID. It creates a left/right composite image for each VQA pair and retains both focus rows and their five questions. Robot grounding uses the first pre-action camera frame of each successful episode and accounts for swapped layouts. Action training uses every frame of every successful episode. The split is by pair within each category (80/10/10 by default), so the two focus rows and all action layouts and seeds of a pair remain together. Images reused across different pairs may still occur in multiple splits.

In the current VQA JSONL, object recognition, attribute recognition, and semantic reasoning answers are `A`/`B`; object and attribute options are stored in `choices`. Visual and counterfactual grounding answers are `Left`/`Right`. The teacher cache compares constrained answers and refreshes a sample when its question or answer changes.

## Run on Linux

Edit `config/config.py`, activate the OpenVLA environment, then run:

```bash
bash semantic_kd_training/scripts/run.sh all
```

Stages can be run separately with `prepare`, `cache`, `train`, and `test`. `cache --split train` and similar split options are supported. `work/` stores the three manifests per split, composite images, teacher answers and features, and action statistics. `output/` stores the LoRA adapter and metrics. Teacher features are computed once per VQA question and once per robot episode grounding question; action frames do not each require teacher inference.

The default teacher is `prism-dinosiglip-224px+7b`; the default student is `gen-robot/openvla-7b-rlvla-sft_16k`. `NousResearch/Llama-2-7b-hf` supplies Llama 2 config/tokenizer metadata for the Prismatic teacher, while the teacher checkpoint supplies its weights. The student uses 4-bit QLoRA by default. Each configured epoch visits all successful action frames; VQA and grounding are interleaved at the probabilities in `config/config.py`.

Validation runs at the end of each epoch. It uses a fixed, balanced subset: two VQA questions per category and question type (100 total, one per answer label when available), up to two successful robot episodes per category for grounding, and five evenly spaced action frames from each selected episode. Test evaluation uses the full test split. Metrics include action, VQA, and robot-grounding losses plus **teacher-forced exact token accuracy** for the QA branches. This is not free-generation accuracy or robot rollout success. Final test evaluation requires the test teacher cache, which `all` creates.

This code has not been executed or tested in this workspace, per the request to leave validation to the user.

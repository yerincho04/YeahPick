# Stage 2: VLM → VLA semantic knowledge distillation

This folder trains an OpenVLA LoRA adapter with `action CE + λ × semantic representation KD`. The frozen Prismatic VLM teacher is run first to cache hidden states; the student is then trained separately. No new action head is added.

## Inputs

Copy **the contents** of `KD_VQA_dataset/` into `input/dataset/` and **the contents** of `scripted_action_collection/output/` into `input/action_data/`:

```text
input/dataset/<category>/vqa.jsonl
input/dataset/<category>/images/...
input/action_data/episodes.jsonl
input/action_data/episodes/<category>/<pair_id>/<layout>/seed_<n>/
  metadata.json
  actions.npy
  frames/0000.jpg ...
```

Only successful action episodes are used. The VQA file supplies the Knowledge instruction and answer side; the semantic KD question asks which tile satisfies that instruction. The other four VQA question types are not used in this Stage 2 loss. Each step uses its pre-action robot camera frame, not the standalone tile images.

## Configuration

Edit [config/config.py](config/config.py). Defaults: pair-level 80% train / 10% validation / 10% test **among successful pairs within each category**, layer 16, cosine KD weight 0.1, LoRA rank 16, 4-bit student, micro-batch 1, gradient accumulation 8, and 2,000 optimizer steps. Original and swapped layouts and all seeds of a pair stay in the same split. Images are reused across pairs, so pair splitting does not guarantee image-disjoint splits.

`teacher_model` defaults to `prism-dinosiglip-224px+7b`; `student_model` defaults to the project's `gen-robot/openvla-7b-rlvla-sft_16k`. For teacher inference, Prismatic loads its own checkpoint weights but needs Llama 2 config/tokenizer files. `teacher_llama2_metadata_repo` defaults to the accessible `NousResearch/Llama-2-7b-hf` mirror for those files; change it to `meta-llama/Llama-2-7b-hf` if you have access to Meta's repository. CUDA, the repository's OpenVLA dependencies, PEFT, and bitsandbytes are required. Student training does not need the ManiSkill renderer.

## Run on Linux

With the project environment active:

```bash
bash semantic_kd_training/scripts/run.sh all
```

You can also run the stages separately with `prepare`, `cache`, `train`, or `test`. The Teacher cache can be resumed: completed episode feature files are skipped. Train/validation/test manifests and action normalization statistics are written to `work/`. LoRA adapters, metrics, and a copy of the action statistics are written to `output/`.

`test` computes held-out action prediction cross-entropy and KD cosine distance. It does **not** report robot task success; that requires running the saved adapter in Act2Answer separately. The LoRA adapter needs its saved `dataset_statistics.json` and the `kd_act2answer` normalization key when used for action inference.

The pipeline has not been run in this Windows workspace, following the request to leave experiments and validation to the user. In particular, 7B teacher extraction can take substantial time and the 24 GB student setting may still need adjustment for the server's CUDA stack.

# OpenVLA VQA fine-tuning

Official `openvla/openvla-7b` with answer-only LoRA supervision. This experiment
uses all 8,000 QA examples (1,600 groups, eight categories); no validation/test
split or robot evaluation is performed, as requested. No action labels are used.

## Reproduction

Use the exact base revision in `base_model.json`. The source branch commits and
files are pinned in `prepare_data.py`. From this directory, run:

```bash
python fetch_sources.py --output sources
python prepare_data.py --sources sources --output data
bash run.sh
python export_merged.py --adapter run/final_adapter --output merged_model
```

The input is one 224x224 RGB image with both source tiles in fixed left/right
order, preserving aspect ratio within each half. No labels, category names,
knowledge provenance, or expected answers are added to the input. All five QA
types are supervised using their existing answers. The `knowledge` block and
aggregate/source-question/pair files are metadata, not extra training examples.
KDEF has missing images and is excluded with user approval.

Default: BF16, two GPUs, batch 1 per GPU, accumulation 8, effective batch 16,
three epochs, 1,500 optimizer updates, AdamW LR 5e-5, cosine decay, 5% warmup,
LoRA rank 16 / alpha 32 / dropout 0.05 on LLM q_proj and v_proj. Vision encoder,
projector, embeddings and output head remain frozen. Only answer and EOS tokens
contribute to the loss. No image augmentation or text truncation is used.

`run.sh --max-updates 2` is an implementation smoke run, not evaluation; put it
in a separate output folder. Final training starts from the original model.
Epoch checkpoints include adapter, optimizer, scheduler and per-rank RNG state.
Resume completed epochs via `bash run.sh --resume run/epoch-1` with the SAME
world size and settings. Resumption keeps the original schedule and seed.

## Outputs

Weights are stored locally for the user, never in GitHub. GitHub contains code,
base revision, data audit, exact environment, training metrics and an experiment
report. The report distinguishes training loss from unmeasured VQA/robot test
performance. Final weights are from epoch 3, not a selected validation optimum.

The standalone merged model retains the original OpenVLA configuration/action
normalization statistics. Use its text generation interface for VQA, or its
action interface for later robot testing. Neither action preservation nor action
improvement is established by this VQA-only training run.

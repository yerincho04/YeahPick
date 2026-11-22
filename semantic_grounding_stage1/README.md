# Stage 1: semantic grounding

Train OpenVLA SFT to select the target tile in a robot scene, with optional VQA training.
This stage does **not** train action values. The policy still uses the same
`Left/Right -> Action: -> seven action tokens` output layout as stage 2;
the marker and action-token count are enforced during decoding.

## Data

Use the 22 cm tile dataset and matching successful scripted episodes. From
the repository root, put or link:

- `KD_VQA_simulation_dataset` at `semantic_grounding_stage1/input/dataset`
- the collection output at `semantic_grounding_stage1/input/action_data`

If the repository already has `KD_VQA_simulation_dataset/` and `action_data/`,
the run script creates those links. Pair-level splits are 80/10/10, with
100 pairs in each of ten categories. Preparation checks original and swapped
seed-0 episode labels, tile positions, instructions, and original-frame image
bytes. It converts only Semantic reasoning's fixed `A=left, B=right` answers
to `Left/Right`. Object and Attribute recognition retain `A/B`; Visual and
Counterfactual grounding retain `Left/Right`.

## Training

Set `train_vqa` in `config.py`: `False` (default) trains on robot grounding only;
`True` also trains on all five VQA question types. Start a new run after changing
this setting; resuming a run requires its original configuration.

```bash
bash semantic_grounding_stage1/scripts/run.sh prepare
bash semantic_grounding_stage1/scripts/run.sh train
```

Each update uses two pairs, one per GPU. Each GPU sees the original and swapped
robot frame. With `train_vqa=True`, it also sees one question of every VQA type;
focus alternates each epoch, so two epochs cover all 8,000 training VQA questions
once. One epoch is 400 updates. The default is at least four and at most eight
epochs, with patience two after epoch four. Loss is grounding CE plus VQA CE
when enabled, each averaged within its task. CE compares the two informative
answer tokens directly. The OpenVLA language layers and multimodal projector
use LoRA; the image encoder is frozen.

Validation evaluates all 100 held-out pairs, including 200 robot layouts and
1,000 VQA questions, after each epoch, even when VQA training is off. With VQA
training, checkpoint selection uses grounding accuracy (40%), VQA accuracy (40%),
and pairs correct under both layouts (20%). Without it, selection uses grounding
accuracy (2/3) and both-layout accuracy (1/3). The stage-2 readiness gate uses
grounding and both-layout accuracy in both modes; VQA thresholds apply only when
VQA training is enabled.
A run that misses this gate stops without starting action training.
Runs, adapters, input snapshots, per-question validation predictions, and
metrics are saved in unique `runs/` folders.
If a run stops during an epoch, resume from its last completed epoch:

```bash
bash semantic_grounding_stage1/scripts/run.sh resume semantic_grounding_stage1/runs/<RUN_DIR>
```

To test the chosen adapter:

```bash
bash semantic_grounding_stage1/scripts/run.sh test semantic_grounding_stage1/runs/<RUN_DIR>
```

The test writes per-question and per-layout predictions plus `test_metrics.json`
under a new `evaluations/` directory. It also checks that the policy produces
a side followed by seven constrained action tokens; their values are not
scored in stage 1.

To test the pretrained Prismatic VLM without training or an OpenVLA adapter:

```bash
PRISMATIC_GPU=0 bash semantic_grounding_stage1/scripts/run.sh test-prismatic
```

This uses `prism-dinosiglip-224px+7b` and the same questions and binary-choice
scoring as the OpenVLA test. It writes separate `prismatic_...` evaluation
results. Prismatic has no robot action output in this test, so only VQA and
grounding are scored. The model may need to download its checkpoint on first use.
The script stores Hugging Face, Xet, Torch, and temporary download files under
`/opt/yeahpick-cache` by default, outside the separate `/root` filesystem.
Set `PRISMATIC_CACHE_ROOT=/some/absolute/path` to use another location; the
script prints its filesystem before downloading. An interrupted download in
`/root/.cache/huggingface` is not moved automatically.
To use the exact saved split from an existing stage-1 run, pass its directory:

```bash
PRISMATIC_GPU=0 bash semantic_grounding_stage1/scripts/run.sh test-prismatic semantic_grounding_stage1/runs/<RUN_DIR>
```

The Windows workstation has no compatible GPU training environment. Python
syntax and dataset contracts can be checked locally, while the training
preflight checks model/tokenizer/device compatibility on the GPU server before
the first update.

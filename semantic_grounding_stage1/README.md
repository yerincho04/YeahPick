# Stage 1: semantic grounding

Train OpenVLA SFT to answer VQA and select the target tile in a robot scene.
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

```bash
bash semantic_grounding_stage1/scripts/run.sh prepare
bash semantic_grounding_stage1/scripts/run.sh train
```

Each update uses two pairs, one per GPU. Each GPU sees the original and swapped
robot frame plus one VQA question of every type. Focus alternates each epoch;
two epochs cover all 8,000 training VQA questions once. One epoch is 400
updates. The default is at least four and at most eight epochs, with patience
two after epoch four. Loss is `grounding CE + VQA CE`, each averaged within its
task. CE compares the two informative answer tokens directly. The OpenVLA
language layers and multimodal projector use LoRA; the image encoder is frozen.

Validation evaluates all 100 held-out pairs, including 200 robot layouts and
1,000 VQA questions, after each epoch. The best checkpoint combines grounding
accuracy (40%), VQA accuracy (40%), and pairs correct under both layouts (20%).
It is marked ready for stage 2 only if these reach 75%, 65%, and 60%,
respectively, and Semantic reasoning and Visual grounding each reach 65%.
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

The Windows workstation has no compatible GPU training environment. Python
syntax and dataset contracts can be checked locally, while the training
preflight checks model/tokenizer/device compatibility on the GPU server before
the first update.

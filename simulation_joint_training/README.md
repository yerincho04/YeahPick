# Simulation image VQA + robot grounding + action

This folder trains a fresh LoRA adapter on `gen-robot/openvla-7b-rlvla-sft_16k`. It uses the first action-training method: each scripted action frame is paired with the original OpenVLA action prompt. It does **not** use a teacher model, teacher cache, or KD loss.

## Inputs

Set paths and hyperparameters in [`config.py`](config.py). The training folder is self-contained: put the inputs here:

- `input/dataset/<category>/vqa.jsonl` and its simulation images. Copy the contents of `KD_VQA_simulation_dataset/` here. All five VQA question types use the pair's original first-frame simulation image.
- `input/action_data/episodes.jsonl` and `input/action_data/episodes/...`. Copy the collected action data here. Successful episodes provide action frames and original/swapped first-frame grounding samples.

From the project root, if the original folders are present:

```bash
mkdir -p simulation_joint_training/input/dataset simulation_joint_training/input/action_data
cp -a KD_VQA_simulation_dataset/. simulation_joint_training/input/dataset/
cp -a action_data/episodes.jsonl action_data/episodes simulation_joint_training/input/action_data/
```

Only `episodes.jsonl` and `episodes/` are needed from the action data; its `archive/` is not used. This workspace already has those inputs copied under `input/`.

`prepare.py` splits **pairs** within each category 80% train, 10% validation, 10% test. Both focus rows, all five VQA types, and both robot layouts for a pair stay in the same split. It checks answer choices, instructions, sides, and input files. Generated manifests and train-split action normalization statistics go to `work/`.

## Loss and sampling

Each microstep uses one action frame and one auxiliary sample. Half the auxiliary samples are VQA and half are robot grounding. The optimization loss is

`L = L_action + 0.5 × L_aux` for VQA, or `L = L_action + 1.0 × L_aux` for grounding.

Across an update, the expected contribution is **action : VQA : grounding = 1 : 0.25 : 0.5**. The auxiliary sampling mix is fixed at 50/50. The action prompt is `What action should the robot take to {instruction}?`; grounding asks which side is the correct destination. VQA uses the questions and A/B or Left/Right labels in the dataset.

## Run

From the project root, in the `openvla_rl4vla` environment:

```bash
bash simulation_joint_training/scripts/run.sh prepare
bash simulation_joint_training/scripts/run.sh train
```

`train` also runs `prepare`. It prints `RUN_DIR=...` and creates a new UTC timestamp directory under `runs/`. By default it uses two GPUs (`TRAIN_GPUS=0,1`); `student_gpu_count` in `config.py` must equal the number of visible GPUs. The global microbatch is `gradient_accumulation_steps=8`, so GPU count does not change the loss definition.

Before the first optimizer step, training runs validation on selected action, VQA, and grounding examples, including complete Left/Right answer scoring. It saves adapter checkpoints every 1000 updates and at the end of each epoch **before** epoch validation. `latest.json` and `best.json` point to checkpoints. Each run also keeps metrics, run settings, test manifests, pair splits, and normalization statistics.

Test the best checkpoint (or latest if validation has not completed):

```bash
bash simulation_joint_training/scripts/run.sh test simulation_joint_training/runs/<RUN_DIR>
```

This writes free-generation VQA and grounding accuracy plus closed-loop robot rollout results to a new timestamp directory under `<RUN_DIR>/evaluations/`. For a faster QA-only check:

```bash
bash simulation_joint_training/scripts/run.sh test simulation_joint_training/runs/<RUN_DIR> --qa-only
```

The training and rollout paths require the project OpenVLA environment and CUDA; rollouts also require the existing SAPIEN/Vulkan setup. This workstation has no such environment, so only data-contract and syntax checks were run here.

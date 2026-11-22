# Semantic bridge training

Fine-tune a fresh OpenVLA SFT LoRA adapter on the enlarged-tile simulation data.
The semantic target decision is an explicit prefix of the action sequence.
There is no teacher model or KD loss.

## Inputs

From the repository root, provide:

- `KD_VQA_simulation_dataset/<category>/vqa.jsonl` with the updated 22 cm scene images.
- `action_data/episodes.jsonl` and `action_data/episodes/...` from the same collection.

`scripts/run.sh prepare` and `train` link these folders into `input/` if needed.
Existing `input/dataset/` and `input/action_data/` paths are respected. To use
independent copies instead, place the contents there before running. Preparation
checks every VQA scene image against the original first action frame, the
instruction and answer side, and the tile positions. Pair-level train/validation/
test splits are 80/10/10 within each of the 10 categories.

## Policy format

Every action sample uses the same input question and output layout:

```text
In: What action should the robot take to <instruction>? First give the target side, then the robot action.
Out: Left
Action: <seven OpenVLA action tokens>
```

`Right` is used for right-target episodes. `Left`/`Right` can consist of multiple
tokenizer IDs. The `Action:` marker is deterministic and supplied by the policy
code; it is not a new tokenizer special token. For the first frame, the side
token(s) and the seven action tokens receive separate cross-entropy losses. For
later frames, the side is supplied as a fixed output prefix and only the seven
action tokens receive a policy loss. The model never needs to re-decide the side
after the first frame. The action suffix sees the side in its causal context.

Training starts with a side/VQA warmup, then joint training. Every update uses a
policy frame and a VQA question. Joint loss is action CE + first-frame side CE +
0.5 VQA CE; because half the policy samples are first frames, their side loss is
multiplied by two so the expected side contribution is one. VQA samples rotate
through category and all five question types. Policy episodes rotate through
category and destination side, then select a first or later frame, so the 27-step
left and 22-step right demonstrations have equal episode weight. All settings
are explicit in `config.py`.

## Run

Use the `openvla_rl4vla` environment with its CUDA, NCCL and SAPIEN/Vulkan setup.
From the repository root:

```bash
export MODEL_CACHE_ROOT=/opt/yeahpick-cache
bash semantic_bridge_training/scripts/run.sh prepare
bash semantic_bridge_training/scripts/run.sh train
```

The script defaults to `/opt/yeahpick-cache` for Hugging Face models, remote
model code, Torch downloads, and temporary files. Set `MODEL_CACHE_ROOT` to
another absolute path if needed; all run modes use the same cache location.

The train command runs preparation again, checks both policy sample formats and
VQA with the loaded model before the first update, and prints `RUN_DIR`. It saves
checkpoints and metrics under a new UTC timestamp directory in `runs/`. Validation
scores all 200 initial-frame side decisions, a balanced set of action frames,
and 100 VQA questions. `best.json` chooses the joint checkpoint with the highest
initial-frame side accuracy, using action and VQA loss as tie-breakers. `latest.json`
points to the newest checkpoint.

Test the chosen checkpoint:

```bash
bash semantic_bridge_training/scripts/run.sh test semantic_bridge_training/runs/<RUN_DIR>
```

`--qa-only` skips robot rollouts. `--grounding-only` tests just initial-frame
side decisions. `--mode predicted`, `oracle`, or `opposite` runs one rollout mode;
the default is predicted plus oracle. `--mode all` also runs the opposite-side
intervention. Predicted mode chooses the side once from the initial frame,
then holds it fixed. Oracle mode fixes the correct side. Opposite mode forces the
wrong side on the same initial scene to test whether actions follow the bridge.
The test output includes
VQA, side decision and per-episode rollout results, including strict success
(released cube fully on the correct tile). All modes use the same collected tile
geometry and ten-step initial grasp. `condition_tile_success` measures whether
the cube lands on the side actually supplied to the policy; in opposite mode this
is the wrong answer tile and tests causal use of the bridge.

No GPU training or Vulkan rollout was run on this Windows workstation. The data
contracts and Python syntax were checked locally; the GPU preflight in `train`
must complete on the server before a long run.

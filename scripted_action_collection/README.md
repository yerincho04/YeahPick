# Scripted action collection

This folder contains the inputs for all 10 VQA categories (including the camera background image), an asset builder, a geometric cube-placement controller, and one collection command. It does not load a VLA checkpoint. The existing ManiSkill and SimplerEnv code in the repository supplies the simulator.

From the repository root on the Linux workspace, with the existing `openvla_rl4vla` environment and working SAPIEN/Vulkan setup:

```bash
bash scripted_action_collection/scripts/collect.sh
```

The command creates GLB tiles from `input/<category>/images/`, reads `input/<category>/vqa.jsonl`, and runs each of the 100 Knowledge instructions in both original and swapped layouts (2,000 episodes total by default). The robot starts in the same pregrasp state as the current OpenVLA evaluation. The controller uses feedback from cube and robot poses to lift, move, lower, and release the cube over the answer tile. It saves episodes only when the final cube is over the correct tile **and the gripper has released it**. This is stricter than the simulator's `success` flag alone.

Useful smaller run:

```bash
bash scripted_action_collection/scripts/collect.sh --category public_info --count 5
```

Options include `--start-pair`, `--count`, `--layout original|swapped|both`, `--seed`, `--repeats`, `--max-steps`, `--retry-failed`, and `--build-only`. `EVAL_GPU=0` chooses the CUDA GPU; an existing `VK_ICD_FILENAMES` is preserved. Ctrl-C ends the foreground Python process. Re-running skips completed episode IDs; use `--retry-failed` to try failed IDs again.

To recollect a category from scratch after changing its assets, stop all collectors first, then archive that category's saved episodes and manifest records. Other categories stay in place:

```bash
python3 scripted_action_collection/scripts/reset_category.py public_info
bash scripted_action_collection/scripts/collect.sh --category public_info
```

The old records and episode folders are preserved under `output/archive/`. Do not run the reset while another collector is writing the shared `episodes.jsonl`.

Outputs:

```text
generated_assets/<category>/pairs.json
generated_assets/<category>/model_db.json
generated_assets/<category>/shapes/<tile>/textured.glb
output/episodes.jsonl
output/episodes/<category>/<pair_id>/<layout>/seed_<n>/
    metadata.json
    actions.npy
    robot_qpos.npy
    frames/0000.jpg ...
```

`frames/t.jpg`, `actions.npy[t]`, and `robot_qpos.npy[t]` describe the **pre-action** observation and executed 7D action at policy step `t`. The 10 pregrasp steps occur before recording, matching the existing VLA evaluation setup. `output/episodes.jsonl` also records failed attempts and reasons. These files are a demonstration collection, not an RLDS dataset; conversion for OpenVLA fine-tuning is a separate step. Extra seeds repeat the same fixed geometry, so they do not themselves create pose variation.

The controller and end-to-end collection have not been run on this local Windows workspace. Actual success rate must be checked on the Linux GPU workspace before using the demonstrations for training.

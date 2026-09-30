# Act2Answer Phase 1: VLA failure-mode diagnostics (OpenVLA only)

Diagnostic pipeline for separating three failure stages per episode: semantic reasoning
(does the backbone's hidden state encode the correct answer), action intent (does the
commanded trajectory aim at the correct target), and motor execution (does it land). No
new architectures, no training, no model-weight changes -- this is Phase 1, diagnostics
only. Everything is off by default; passing `--enable-diagnostics` (or
`ENABLE_DIAGNOSTICS=1` to `scripts/eval_openvla.sh`) is required to produce any of the
files below, and baseline eval outputs (videos, `stats.yaml`) are byte-for-byte unchanged
either way.

## Modified files (all additive -- no existing behavior changed)

| File | What changed |
|---|---|
| `ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene_multi_v4.py` | `Act2AnswerV4.evaluate()` now also stores `diag_cube_pos`, `diag_eef_pos`, `diag_left_target_pos`, `diag_right_target_pos`, `diag_gripper_state`, `diag_pair_id`, `diag_swap`, `diag_answer_side_is_left` into `self.episode_stats` (already-computed local values from the existing soft-metrics code) so they flow through the existing `return dict(**self.episode_stats, success=success)` and into `info` every step. No existing key touched. |
| `SimplerEnv/simpler_env/run.py` | New `enable_diagnostics: bool = False` field on `Args`. `Runner._get_action` optionally calls `policy.get_action_and_diagnostics` instead of `get_action` and stashes per-step hidden states. `Runner.render()` accumulates a `"hidden"` list per episode (alongside the existing `"image"/"action"/"info"`) and, at the end, calls `diagnostics.episode_logger.write_episode_record` per episode when enabled. |
| `SimplerEnv/simpler_env/eval.py` | New `--enable-diagnostics` CLI flag, threaded into `Args.enable_diagnostics`. |
| `scripts/eval_openvla.sh` | Passes through `ENABLE_DIAGNOSTICS=1` as `--enable-diagnostics`. |
| `openvla/prismatic/extern/hf/modeling_prismatic.py` | New method `OpenVLAForActionPredictionWithValueHead.get_diagnostic_hidden_states(...)` -- generalizes the existing `get_hidden()` pattern (same `input_ids[:, -1] == 29871` invariant, same "second forward pass with `output_hidden_states=True`" approach) to arbitrary layer indices. Does not touch `predict_action_batch`/`generate()` or any existing method. |
| `SimplerEnv/simpler_env/policies/openvla/openvla_train.py` | New `OpenVLAPolicy.get_diagnostic_hidden(...)` wrapper: converts `{0.25, 0.5, 0.75, 1.0}` fractions into layer indices via `config.text_config.num_hidden_layers`, calls the new model method. |
| `SimplerEnv/simpler_env/policies/openvla/openvla_adapter.py` | New `OpenVLAInference.get_action_and_diagnostics(...)`: calls the existing unchanged `get_action` for the action, plus one extra cheap forward pass for hidden states. |

## New files (`Act2Answer/diagnostics/`)

- `episode_logger.py` -- writes `episode_log.jsonl` (one JSON object per episode) and
  `hidden_states/<episode_id>.pt` (one file per episode) from what `Runner.render()`
  already accumulated. Called automatically from `run.py`; not meant to be invoked
  directly.
- `intent_classifier.py` -- Task 4: geometric action-intent classifier. Reads/rewrites
  `episode_log.jsonl` in place, adding `predicted_intent_side` / `intent_confidence`.
- `probe_dataset.py` -- Task 5: builds the hidden-state -> answer-side dataset, splitting
  by `(asset, question_id)` (60/20/20) so normal+swapped copies of a question never cross
  splits. Takes one or more `--diagnostics-dir` (pass noswap AND swap directories together
  -- they're separate eval runs, see below) and writes a combined `probe_dataset.npz`.
- `test_split_integrity.py` -- asserts zero grouping-key overlap across splits
  (pytest-discoverable, also runnable standalone).
- `linear_probe.py` -- Task 5: fits one `LogisticRegression` per layer, prints the layer
  accuracy table, writes `semantic_probe.jsonl` (test-fold predictions only).
- `failure_decomposition.py` -- Task 6: joins episode logs (with intent, across the same
  `--diagnostics-dir` list) + `semantic_probe.jsonl` (test fold only), classifies each
  episode, writes `phase1_summary.csv`, prints `P(intent wrong | semantic correct)`.

## Output layout

noswap and swap are separate `simpler_env.eval` invocations (see `scripts/eval_openvla.sh`)
and so write to separate `<name>/` directories:

```
$A2A_OUTPUT_DIR/<asset>-noswap/
  glob/...                          <- unchanged baseline outputs (videos, config.yaml, stats.yaml)
  diagnostics/
    episode_log.jsonl               <- one record per episode (Task 2 + Task 4's added fields)
    hidden_states/<episode_id>.pt   <- {0.25: [T,H], 0.5: [T,H], 0.75: [T,H], 1.0: [T,H], episode_id, question_id}
$A2A_OUTPUT_DIR/<asset>-swap/
  glob/... , diagnostics/           <- same layout, the swapped layout's episodes

$A2A_OUTPUT_DIR/<asset>-diagnostics_combined/   <- Task 5/6 outputs, built FROM both dirs above
  probe_dataset.npz                 <- Task 5 split (from probe_dataset.py, both dirs merged)
  semantic_probe.jsonl              <- Task 5 test-fold predictions (from linear_probe.py)
  phase1_summary.csv                <- Task 6 final per-episode table (from failure_decomposition.py)
```

## Key assumptions (stated explicitly, not guessed silently)

1. **Pre-action hidden state** = the hidden state at the last prompt token (the space
   token `29871` immediately before the first generated action token), at network depths
   {25%, 50%, 75%, 100%} of the LLM backbone -- matching the exact convention the
   checkpoint's own `get_hidden()`/`get_value()`/value-head code already uses.
2. **Semantic-probe timestep** = timestep 0 of each episode (the model's first inference
   call, before any commanded motion) -- see `probe_dataset.py` docstring. Override with
   `--timestep`.
3. **Action intent is measured from realized eef displacement, not raw commanded actions**
   (`eef_position[-1] - eef_position[first_grasp]`), starting at the first
   `gripper_state == True` step, not a fixed step offset. Two earlier versions were caught
   and ruled out on real rollouts (see `intent_classifier.py` docstring for the full
   story): a fixed `init_grasp_steps + hold_cube_steps` cutoff missed the approach motion
   entirely (the env's hold window only forces the *gripper* channel, translation is always
   free), and summing the raw commanded `world_vector` turned out to be in a
   gripper/base-aligned frame ~180 degrees rotated in x/y from the world frame that
   eef_position/target positions are logged in -- so it pointed the wrong way even on
   episodes that succeeded. Realized eef displacement is in the same world frame as the
   targets by construction, and verified to align with the correct target
   (cosine >= 0.86) on every successful real episode in the smoke test.
4. **Intent uncertain threshold** = cosine-similarity margin < 0.05 between the two
   candidate directions (`intent_classifier.py::UNCERTAIN_MARGIN`).
5. **Failure decomposition is computed only for the probe's held-out test fold**, since
   `semantic_correct` must be an out-of-sample prediction (see
   `failure_decomposition.py` docstring) -- with ~200-300 pilot episodes this means the
   final table covers roughly 20% of episodes, stated explicitly rather than diluted with
   in-sample labels.

## Reproducing an end-to-end run

```bash
# 1. Baseline sanity check (no diagnostics) -- confirm this still works unchanged.
ASSETS=test_colors COUNT=6 EVAL_GPU=<GPU> bash scripts/eval_openvla.sh

# 2. Same run, with diagnostics on. noswap and swap are separate simpler_env.eval
#    invocations (see scripts/eval_openvla.sh) and so write to separate directories.
ENABLE_DIAGNOSTICS=1 ASSETS=test_colors COUNT=6 EVAL_GPU=<GPU> bash scripts/eval_openvla.sh
DIAG_NOSWAP=outputs/openvla-test_colors-noswap/diagnostics
DIAG_SWAP=outputs/openvla-test_colors-swap/diagnostics
COMBINED=outputs/openvla-test_colors-diagnostics_combined

# 3. Action-intent classification (adds predicted_intent_side/intent_confidence in place,
#    per directory -- intent doesn't need cross-layout data).
python -m diagnostics.intent_classifier --episode-log "$DIAG_NOSWAP/episode_log.jsonl"
python -m diagnostics.intent_classifier --episode-log "$DIAG_SWAP/episode_log.jsonl"

# 4. Semantic probing dataset + split-integrity check + linear probe. Pass BOTH directories
#    together -- a question's normal and swapped episodes must be seen together for the
#    question_id group-split to mean anything (see probe_dataset.py's docstring).
python -m diagnostics.probe_dataset --diagnostics-dir "$DIAG_NOSWAP" "$DIAG_SWAP" --out "$COMBINED/probe_dataset.npz"
python -m diagnostics.test_split_integrity --probe-dataset "$COMBINED/probe_dataset.npz"
python -m diagnostics.linear_probe --probe-dataset "$COMBINED/probe_dataset.npz" --out "$COMBINED/semantic_probe.jsonl"

# 5. Final Phase 1 decomposition + summary.
python -m diagnostics.failure_decomposition --diagnostics-dir "$DIAG_NOSWAP" "$DIAG_SWAP" \
  --semantic-probe "$COMBINED/semantic_probe.jsonl" --out "$COMBINED/phase1_summary.csv"
```

Steps 3-5 need only `numpy`/`torch`/`scikit-learn` (no simulator/model stack), so they can
run outside the `openvla_rl4vla` conda env if that's more convenient (scikit-learn was
added to `openvla_rl4vla` via `pip install scikit-learn` for this pipeline).

Once this smoke test passes on the `test_colors` 6-question slice, scale up to the ~200-300
episode pilot from the semantic-heavy categories (State, Time, Traffic, Public Info,
Celebrity, Living World, plus Color/Shape controls) by pointing `ASSETS` at those asset
folders and rerunning steps 2-5 per asset (or concatenating multiple runs'
`episode_log.jsonl`/`hidden_states/` before steps 3-5).

## Matched knowledge/object/spatial go/no-go experiment

The condition-matched OpenVLA extension is documented in
[`MATCHED_EXPERIMENT.md`](MATCHED_EXPERIMENT.md). It reuses this logger and intent classifier,
adds condition/semantic-answer/seed/initial-state metadata, and provides paired McNemar and
bootstrap analysis. It does not perform activation patching.

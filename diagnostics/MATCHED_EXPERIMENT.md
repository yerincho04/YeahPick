# OpenVLA matched-instruction go/no-go experiment

This experiment compares three instructions while holding the Act2Answer question, visual
assets, layout, seed, robot initialization, and correct physical target fixed:

- `knowledge`: `Place the cube on the animal that is associated with Australia.`
- `explicit_object`: `Place the cube on the kangaroo.`
- `explicit_spatial`: `Place the cube on the left tile.` (or right after layout swap)

`semantic_pilot_v1/pairs.json` stores `semantic_answer`, `left_label`, `right_label`,
`knowledge_category`, and `knowledge_instruction`. These fields are additive and older asset
sets remain readable. Run the non-destructive metadata migration with:

```bash
cd diagnostics/dataset_build
python build_dataset.py --metadata-only
```

## Rollouts

Three-question smoke test (18 rollouts):

```bash
sbatch --export=ALL,SMOKE_COUNT=3,START_ID=0,SEED=0 slurm/matched_openvla.sbatch
```

The validated smoke run was Slurm job `2308572`.

Full pilot (15 shards, each running 8 questions x 2 layouts for one condition):

```bash
sbatch --array=0-14%7 --export=ALL,SEED=0 slurm/matched_openvla.sbatch
```

The completed full run was Slurm array job `2310374`. All 15 tasks completed, producing
30 output directories, 240 unique rollouts, 80 complete matched triplets, and zero
matched-initialization mismatches.

A final language audit found that Q31's question-to-command normalization omitted the word
`that`. The normalizer was fixed and the affected knowledge shard (Q24--Q31, both layouts)
was rerun as Slurm job `2310537` (completed with exit code 0). The final JSON, hidden states,
videos, probe, and statistics all use `Place the cube on the item that symbolizes love.`
No question or answer label was excluded as ambiguous.

Output directory names include condition, layout, question range, and seed. Videos include
question ID. Retrying an identical episode replaces its JSONL record rather than duplicating it.
Only knowledge runs capture hidden states; all conditions capture trajectory diagnostics.

Each JSONL record contains the join key `(question_id, layout, seed)`, condition, semantic
answer, correct side, initial-state values/hash, intent trajectory inputs, final cube position,
truncation metadata, and the precise final state used (`final_policy_step=79`, normally
`final_sim_elapsed_steps=90`). Rollout behavior remains unchanged; repeated terminal reports are
aggregated only once at that final policy state.

## Probe and paired analysis

Build the existing probe from **knowledge directories only**. Its label is correct target side
(`left` versus `right`), not semantic answer identity:

```bash
knowledge_dirs=()
for layout in noswap swap; do
  for qrange in 000-007 008-015 016-023 024-031 032-039; do
    knowledge_dirs+=("outputs/openvla-semantic_pilot_v1-knowledge-${layout}-q${qrange}-seed0/diagnostics")
  done
done
python -m diagnostics.probe_dataset \
  --diagnostics-dir "${knowledge_dirs[@]}" \
  --out outputs/matched_openvla_analysis/probe_dataset.npz
python -m diagnostics.test_split_integrity \
  --probe-dataset outputs/matched_openvla_analysis/probe_dataset.npz
python -m diagnostics.linear_probe \
  --probe-dataset outputs/matched_openvla_analysis/probe_dataset.npz \
  --out outputs/matched_openvla_analysis/semantic_probe.jsonl
```

Then produce machine-readable rollout/triplet CSVs, JSON statistics, and the Markdown report:

```bash
python -m diagnostics.matched_analysis \
  --outputs-root outputs \
  --output-dir-glob '*-q000-007-seed0' \
  --output-dir-glob '*-q008-015-seed0' \
  --output-dir-glob '*-q016-023-seed0' \
  --output-dir-glob '*-q024-031-seed0' \
  --output-dir-glob '*-q032-039-seed0' \
  --asset semantic_pilot_v1 \
  --side-probe outputs/matched_openvla_analysis/semantic_probe.jsonl \
  --out-dir outputs/matched_openvla_analysis \
  --bootstrap-samples 10000 \
  --conclusion grounding-dominant
```

The report calls the conditional probe result **grounded target-side decodability**. With nearly
one distinct correct object per question, this 40-question pilot cannot support a statistically
meaningful multiclass semantic-answer-identity probe without redesign or repeated labels.

Final artifacts are under `outputs/matched_openvla_analysis/`, including
`matched_results.json`, `matched_rollouts.csv`, `matched_triplets.csv`, and `report.md`.

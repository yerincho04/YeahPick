# Expected result workspace

This folder keeps separate **illustrative expectations**, not measured results.
The VQA graphs assume Semantic KD was trained with simulation images and
Semantic Bridge uses two stages. Their category and question-type values
alternate around each other and aggregate to 60.0% and 61.0%, respectively.
The simulation graph assumes the two-stage Semantic Bridge reaches 63.0%
overall (126/200 episodes). OpenVLA SFT and the unchanged simulation Semantic
KD bars remain observed results. The two Bridge diagnostic graphs are unchanged
copies of observed results.

The `source` column marks altered rows `EXPECTED ASSUMPTION`. In the simulation
CSV, `correct` is a hypothetical count: 126 expected successes out of 200.
No run produced this count.

To redraw only the copied CSVs, run from the project root in the `wisar` environment:

```bash
conda activate wisar
python analysis/expected/plot_results.py --plots-only
python analysis/expected/plot_bridge_modes.py --plots-only
```

Running either script without `--plots-only` rebuilds CSVs from the observed
evaluation JSONL files. `plot_results.py` then reapplies the illustrative
assumptions defined in `EXPECTED_PERCENT`; `plot_bridge_modes.py` preserves only
the observed Bridge diagnostic results.

## Original analysis notes

Run from the project root with the `wisar` Conda environment:

```bash
conda activate wisar
python analysis/plot_results.py
```

The script checks that all three models were evaluated on the same VQA sample IDs and simulation episodes, then writes:

| CSV in input/ | Graphs in output/ | Measure |
| --- | --- | --- |
| vqa_by_category.csv | vqa_by_category.png and .svg | Exact-answer VQA accuracy by category |
| vqa_by_question_type.csv | vqa_by_question_type.png and .svg | Exact-answer VQA accuracy by question type |
| simulation_success_by_category.csv | simulation_success_by_category.png and .svg | Simulator success by category |

Each CSV has one row per group and model, including correct count, total count, percentage, and source file. The **Overall** row pools all examples. Since every category and question type has the same sample count, this also equals their simple average.

Matplotlib draws the graphs from the CSVs in input/. To edit CSV values and redraw without rereading the source JSONL files, run `python analysis/plot_results.py --plots-only`.

For the two Semantic Bridge diagnostic graphs, run `python analysis/plot_bridge_modes.py`.
This writes `bridge_predicted_by_category.csv` and `bridge_oracle_by_category.csv`
to `input/`, with matching PNG and SVG files in `output/`. The predicted graph
uses `condition_tile_success`: whether the robot followed its predicted side.
The oracle graph uses `success`: whether the task succeeded when given the correct
side. Each category has 20 episodes in each mode; Overall pools 200 episodes.
Use `python analysis/plot_bridge_modes.py --plots-only` to redraw from the CSVs.

## Comparison definition

- **OpenVLA SFT:** semantic_kd_training/output files ending in `(1)`, with `model: base`.
- **Semantic KD:** semantic_kd_training/output files without the `(1)` suffix.
- **Semantic Bridge (Ours):** semantic_bridge_training/eval_result. The simulation chart uses **predicted** mode only. Oracle episodes are excluded.
- Simulation uses each model's `success` result, not Bridge's additional strict tile metric. All three files contain the same 100 held-out pairs, each in original and swapped layouts: 200 episodes per model.
- VQA uses free-generation exact-answer scoring on the same 1,000 sample IDs. OpenVLA SFT scored 0/1,000 because its generated answers did not match the required answer format. This value is an **exact-output score**, not a standalone measure of its visual understanding.
- The Bridge rollout uses a separate direct Gym evaluation path. The graphs compare the saved runs and do not establish a controlled causal comparison between training methods.

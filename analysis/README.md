# Three-model result comparison

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

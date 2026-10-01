# Time clock experiment inputs

This folder defines 10 clock questions for the existing OpenVLA matched-instruction experiment.
It assumes a typical clear day and ordinary weekday work/school hours.

- `build_inputs.py` defines the ten question pairs, correct sides, and knowledge instructions.
- `input_pngs/` contains the 14 distinct digital clock faces used by those questions.
- `manifest.csv` lists each question's original left/right times, correct side, and the three instruction conditions.
- `ManiSkill/mani_skill/assets/carrot/time_clock_v1/pairs.json` is the input read by the evaluator.
- `ManiSkill/mani_skill/assets/carrot/time_clock_v1/model_db.json` and `shapes/clock_*/` contain tile metadata and geometry.

## Installed GLBs

All 14 `shapes/clock_*/textured.glb` files have been replaced with the matching files from `asset_generator/output/20261002_021605_260602/`. Each copy was verified against its source by SHA-256. The 14 source PNGs in `asset_generator/input/` also match `input_pngs/` byte-for-byte. The `collision.obj` files have been scaled to the generated GLBs' 11 × 11 cm footprint, which matches `model_db.json`.

The builder copies a placeholder only when a tile file is missing, so rerunning it will not overwrite these GLBs. If the clock PNG generator is changed later, regenerate the GLBs too before evaluation.

## Rebuild inputs

From the repository root:

```bash
python time_clock_assets/build_inputs.py
```

## Run on the Linux server

On the Linux server, after adapting the paths and Slurm partition in `slurm/run_openvla.sbatch`:

```bash
sbatch --array=0-5 --export=ALL,ASSETS=time_clock_v1,SEED=0 slurm/run_openvla.sbatch
python3 diagnostics/condition_summary.py time_clock_v1
```

The array has 3 conditions × 2 shards of 5 questions. Each task also runs both original and swapped left/right layouts. Questions 2 and 3 intentionally use the same clocks but opposite correct answers for sunset versus sunrise.

The training input paths are `dataset/` and `action_data/`. On Linux,
`scripts/run.sh prepare` or `train` links them to the repository-level
`KD_VQA_simulation_dataset/` and `action_data/` when they are absent.
Existing directories are respected. No second copy of the large action data
is required.

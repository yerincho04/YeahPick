"""Stage 2: continue the Stage 1 adapter with side-conditioned action learning."""

from dataclasses import dataclass
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    stage1_run: Path = Path(os.environ.get("STAGE1_RUN", str(ROOT.parent / "semantic_grounding_stage1" / "runs" / "SET_STAGE1_RUN")))
    dataset_root: Path = ROOT / "input" / "dataset"
    action_data_root: Path = ROOT / "input" / "action_data"
    counterfactual_root: Path = ROOT / "input" / "counterfactual_action_data"
    work_dir: Path = ROOT / "work"
    runs_dir: Path = ROOT / "runs"
    student_model: str = "gen-robot/openvla-7b-rlvla-sft_16k"
    action_unnorm_key: str = "kd_act2answer"
    tile_xy_left: tuple[float, float] = (-0.22, -0.155)
    tile_xy_right: tuple[float, float] = (-0.22, 0.13)
    student_gpu_count: int = 2
    seed: int = 0
    epochs: int = 2
    batch_size: int = 8  # Global action examples per update.
    frames_per_episode_per_epoch: int = 6
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    side_gradient_ratio: float = 0.3
    side_examples_per_update_per_gpu: int = 1
    side_weight_min: float = 0.01
    side_weight_max: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    student_max_answer_tokens: int = 24
    validation_grounding_pairs: int = 100


CONFIG = Config()

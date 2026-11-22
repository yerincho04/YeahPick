"""Settings for stage 1: visual questions and first-frame target selection."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    dataset_root: Path = ROOT / "input" / "dataset"
    action_data_root: Path = ROOT / "input" / "action_data"
    work_dir: Path = ROOT / "work"
    runs_dir: Path = ROOT / "runs"
    student_model: str = "gen-robot/openvla-7b-rlvla-sft_16k"
    tile_xy_left: tuple[float, float] = (-0.22, -0.155)
    tile_xy_right: tuple[float, float] = (-0.22, 0.13)

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    seed: int = 0

    student_gpu_count: int = 2
    min_epochs: int = 4
    max_epochs: int = 8
    early_stop_patience: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    grounding_weight: float = 1.0
    vqa_weight: float = 1.0
    train_vqa: bool = False  # True: include all five VQA types; False: grounding only.

    stage2_grounding_accuracy: float = 0.75
    stage2_both_layout_accuracy: float = 0.60
    stage2_vqa_accuracy: float = 0.65
    stage2_semantic_accuracy: float = 0.65
    stage2_visual_grounding_accuracy: float = 0.65


CONFIG = Config()

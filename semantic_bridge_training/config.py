"""Explicit settings for the semantic decision to action experiment."""

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
    action_unnorm_key: str = "kd_act2answer"
    tile_xy_left: tuple[float, float] = (-0.22, -0.155)
    tile_xy_right: tuple[float, float] = (-0.22, 0.13)

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    seed: int = 0

    student_gpu_count: int = 2
    gradient_accumulation_steps: int = 8
    warmup_updates: int = 400
    joint_epochs: int = 3
    joint_updates_per_epoch: int = 1200
    checkpoint_every_updates: int = 1000
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    first_frame_probability: float = 0.5
    side_weight: float = 1.0
    action_weight: float = 1.0
    vqa_weight: float = 0.5
    student_max_answer_tokens: int = 24


CONFIG = Config()

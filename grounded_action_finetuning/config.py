"""Explicit settings for KD-to-grounded-action fine-tuning."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    dataset_root: Path = ROOT / "input" / "dataset"
    action_data_root: Path = ROOT / "input" / "action_data"
    kd_adapter_dir: Path = ROOT / "input" / "kd_adapter"
    work_dir: Path = ROOT / "work"
    runs_dir: Path = ROOT / "runs"
    student_model: str = "gen-robot/openvla-7b-rlvla-sft_16k"
    action_unnorm_key: str = "kd_act2answer"

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    seed: int = 0

    student_gpu_count: int = 2
    gradient_accumulation_steps: int = 8
    train_epochs: int = 2
    checkpoint_every_updates: int = 1000
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Each action micro-step is paired with one supervised auxiliary question.
    grounding_probability: float = 0.75
    grounding_weight: float = 1.0
    vqa_weight: float = 0.25
    student_max_answer_tokens: int = 24
    max_prompt_tokens: int = 256
    num_workers: int = 0

    # Only needed by the copied model_io helpers; this stage has no teacher KD.
    kd_layers: tuple[int, ...] = ()


CONFIG = Config()

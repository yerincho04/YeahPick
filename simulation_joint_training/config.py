"""Settings for joint simulation-image VQA, grounding, and action training."""

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

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    seed: int = 0

    student_gpu_count: int = 2
    gradient_accumulation_steps: int = 8
    train_epochs: int = 3
    checkpoint_every_updates: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # One action and one auxiliary QA sample per micro-step. Half of auxiliary
    # samples are VQA and half are robot grounding, giving expected CE weights
    # action:VQA:grounding = 1:0.25:0.5.
    vqa_weight: float = 0.5
    grounding_weight: float = 1.0
    student_max_answer_tokens: int = 24


CONFIG = Config()

"""All paths and hyperparameters for Stage-2 semantic KD. Edit this file."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # Put the 10 category folders (each with vqa.jsonl + images/) here.
    dataset_root: Path = ROOT / "input" / "dataset"
    # Put scripted_action_collection/output contents here: episodes.jsonl + episodes/.
    action_data_root: Path = ROOT / "input" / "action_data"
    work_dir: Path = ROOT / "work"
    output_dir: Path = ROOT / "output"

    # 7B teacher features are cached before student training, never loaded together.
    teacher_model: str = "prism-dinosiglip-224px+7b"
    student_model: str = "gen-robot/openvla-7b-rlvla-sft_16k"
    action_unnorm_key: str = "kd_act2answer"
    kd_layers: tuple[int, ...] = (16,)  # hidden_states[0] is the embedding output
    kd_weight: float = 0.1

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    # One sample is one pre-action camera frame + one 7D scripted action.
    max_train_steps: int = 2000
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_4bit_student: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    max_prompt_tokens: int = 256
    validation_every: int = 200
    checkpoint_every: int = 200
    validation_samples: int = 32
    seed: int = 0
    num_workers: int = 0


CONFIG = Config()

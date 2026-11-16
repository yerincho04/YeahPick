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
    # Only its Llama 2 config/tokenizer are read; Prismatic checkpoint supplies the weights.
    teacher_llama2_metadata_repo: str = "NousResearch/Llama-2-7b-hf"
    student_model: str = "gen-robot/openvla-7b-rlvla-sft_16k"
    action_unnorm_key: str = "kd_act2answer"
    kd_layers: tuple[int, ...] = (16,)  # hidden_states[0] is the embedding output
    kd_weight: float = 0.1
    vqa_weight: float = 0.5
    grounding_weight: float = 0.5
    vqa_aux_probability: float = 0.75  # Every micro-step also trains one auxiliary task.
    teacher_max_answer_tokens: int = 24
    student_max_answer_tokens: int = 24

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    # Each epoch traverses every successful scripted action frame once.
    train_epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # BF16 student model is split across two visible GPUs.
    student_gpu_count: int = 2
    gradient_checkpointing: bool = True
    bf16: bool = True
    max_prompt_tokens: int = 256
    validation_every_epochs: int = 1
    checkpoint_every: int = 200
    validation_vqa_per_category_type: int = 2
    validation_episodes_per_category: int = 2
    validation_steps_per_episode: int = 5
    seed: int = 0
    num_workers: int = 0


CONFIG = Config()

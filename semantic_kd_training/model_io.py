"""OpenVLA inputs, action labels, and model loading shared by training/evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CONFIG


def read_manifest(split: str, suffix: str = "") -> list[dict]:
    path = CONFIG.work_dir / f"{split}{suffix}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare.py first: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def teacher_feature_path(sample: dict, layer: int) -> Path:
    return CONFIG.work_dir / "teacher_vqa_features" / sample["sample_id"] / f"layer_{layer}.npy"


def teacher_result_path(sample: dict) -> Path:
    return CONFIG.work_dir / "teacher_vqa_features" / sample["sample_id"] / "result.json"


def prompt_for_question(question: str) -> str:
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", question)
    return builder.get_prompt()


def normalized_action(action: list[float], stats: dict) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    result = values.copy()
    result[mask] = 2 * (values[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1
    return np.clip(result, -1.0, 1.0)


def action_inputs(sample: dict, processor, action_tokenizer, stats: dict) -> dict:
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    action = normalized_action(sample["action"], stats)
    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", f"What action should the robot take to {sample['instruction'].lower()}?")
    builder.add_turn("gpt", action_tokenizer(action))
    ids = processor.tokenizer(builder.get_prompt(), add_special_tokens=True, return_tensors="pt")["input_ids"]
    labels = ids.clone()
    labels[:, :-(len(action) + 1)] = -100  # 7 action tokens + EOS
    with Image.open(sample["image"]) as raw:
        pixel = processor.image_processor(
            images=raw.convert("RGB"), return_tensors="pt"
        )["pixel_values"]
    return {"input_ids": ids, "labels": labels, "pixel_values": pixel}


def qa_inputs(sample: dict, processor) -> tuple[dict, int]:
    tokenizer = processor.tokenizer
    prompt = prompt_for_question(sample["question"])
    prefix = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    answer = tokenizer(" " + sample["answer"], add_special_tokens=False,
                       return_tensors="pt")["input_ids"]
    if answer.numel() > CONFIG.student_max_answer_tokens:
        raise ValueError(f"Answer exceeds token limit: {sample['sample_id']}")
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=prefix.dtype)
    ids = torch.cat((prefix, answer, eos), dim=1)
    labels = torch.cat((torch.full_like(prefix, -100), answer, eos), dim=1)
    with Image.open(sample["image"]) as raw:
        pixel = processor.image_processor(
            images=raw.convert("RGB"), return_tensors="pt"
        )["pixel_values"]
    return {"input_ids": ids, "labels": labels, "pixel_values": pixel}, prefix.shape[1]


def register_openvla() -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    for registration in (
        lambda: AutoConfig.register("openvla", OpenVLAConfig),
        lambda: AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor),
        lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
        lambda: AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction),
    ):
        try:
            registration()
        except ValueError:  # already registered in this process
            pass


def load_student(training: bool, adapter_path: Path | None = None):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model

    register_openvla()
    if not torch.cuda.is_available():
        raise RuntimeError("Student requires a CUDA GPU")
    device_index = torch.cuda.current_device()
    processor = AutoProcessor.from_pretrained(CONFIG.student_model, trust_remote_code=True)
    kwargs = {"torch_dtype": torch.bfloat16 if CONFIG.bf16 else torch.float16,
              "low_cpu_mem_usage": True, "trust_remote_code": True,
              "device_map": {"": device_index}}
    model = AutoModelForVision2Seq.from_pretrained(CONFIG.student_model, **kwargs)
    # RoPE buffers are non-persistent in some Transformers releases, so loading
    # checkpoint weights does not necessarily place them with their modules.
    device = torch.device("cuda", device_index)
    model.language_model.model.rotary_emb.to(device)
    for layer in model.language_model.model.layers:
        rotary = getattr(layer.self_attn, "rotary_emb", None)
        if rotary is not None:
            rotary.to(device)
    statistics = json.loads((CONFIG.work_dir / "dataset_statistics.json").read_text(encoding="utf-8"))
    model.norm_stats.update(statistics)
    if training and CONFIG.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # Frozen embeddings must still produce an autograd input for checkpointed LoRA layers.
        model.enable_input_require_grads()
    if adapter_path is None:
        lora = LoraConfig(
            r=CONFIG.lora_rank, lora_alpha=CONFIG.lora_alpha,
            lora_dropout=CONFIG.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training)
    return model, processor


def load_action_stats() -> dict:
    path = CONFIG.work_dir / "dataset_statistics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[CONFIG.action_unnorm_key]["action"]

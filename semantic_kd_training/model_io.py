"""OpenVLA inputs, action labels, and model loading shared by training/evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CONFIG


def read_manifest(split: str) -> list[dict]:
    path = CONFIG.work_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare.py first: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def teacher_feature_path(sample: dict, layer: int) -> Path:
    return CONFIG.work_dir / "teacher_features" / sample["episode_id"] / f"layer_{layer}.npy"


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
        image = torch.from_numpy(np.asarray(raw.convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0)
    pixel = processor.image_processor.apply_transform(image.float())
    return {"input_ids": ids, "labels": labels, "pixel_values": pixel}


def semantic_inputs(sample: dict, processor) -> dict:
    prompt = prompt_for_question(sample["semantic_question"])
    ids = processor.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    with Image.open(sample["image"]) as raw:
        image = torch.from_numpy(np.asarray(raw.convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0)
    pixel = processor.image_processor.apply_transform(image.float())
    return {"input_ids": ids, "pixel_values": pixel}


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
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    register_openvla()
    processor = AutoProcessor.from_pretrained(CONFIG.student_model, trust_remote_code=True)
    kwargs = {"torch_dtype": torch.bfloat16 if CONFIG.bf16 else torch.float16,
              "low_cpu_mem_usage": True, "trust_remote_code": True}
    if CONFIG.use_4bit_student:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForVision2Seq.from_pretrained(CONFIG.student_model, **kwargs)
    if not CONFIG.use_4bit_student:
        model = model.cuda()
    statistics = json.loads((CONFIG.work_dir / "dataset_statistics.json").read_text(encoding="utf-8"))
    model.norm_stats.update(statistics)
    if training and CONFIG.use_4bit_student:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=CONFIG.gradient_checkpointing
        )
    if training and CONFIG.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
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

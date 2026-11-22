"""Load the OpenVLA SFT model and the stage-1 LoRA adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from config import CONFIG


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
                            "gate_proj", "up_proj", "down_proj",
                            "projector.fc1", "projector.fc2", "projector.fc3"],
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training)
    return model, processor

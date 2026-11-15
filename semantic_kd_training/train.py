"""OpenVLA LoRA training with action CE + frozen Prismatic feature KD."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import CONFIG
from model_io import (action_inputs, load_action_stats, load_student, read_manifest,
                      semantic_inputs, teacher_feature_path)


def to_device(values: dict) -> dict:
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return {key: value.to("cuda", dtype=dtype) if key == "pixel_values" else value.to("cuda")
            for key, value in values.items()}


def teacher_vectors(sample: dict) -> dict[int, torch.Tensor]:
    result = {}
    for layer in CONFIG.kd_layers:
        path = teacher_feature_path(sample, layer)
        if not path.is_file():
            raise FileNotFoundError(f"Run cache_teacher.py first: {path}")
        vector = np.load(path, mmap_mode="r")[sample["step"]].copy()
        result[layer] = torch.from_numpy(vector).float().cuda()
    return result


def losses(model, processor, tokenizer, stats: dict, sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    action = to_device(action_inputs(sample, processor, tokenizer, stats))
    output = model(**action, use_cache=False, return_dict=True)
    action_loss = output.loss
    if not torch.isfinite(action_loss):
        raise FloatingPointError(f"Nonfinite action loss: {sample['sample_id']}")
    del output, action

    semantic = to_device(semantic_inputs(sample, processor))
    output = model(**semantic, output_hidden_states=True, use_cache=False, return_dict=True)
    teacher = teacher_vectors(sample)
    kd_terms = []
    for layer in CONFIG.kd_layers:
        if layer >= len(output.hidden_states):
            raise ValueError(f"Student has no hidden_states[{layer}]")
        student_vector = output.hidden_states[layer][0, -1].float()
        teacher_vector = teacher[layer]
        if student_vector.shape != teacher_vector.shape:
            raise ValueError(f"Teacher/student hidden size mismatch at layer {layer}")
        kd_terms.append(1 - F.cosine_similarity(
            student_vector.unsqueeze(0), teacher_vector.unsqueeze(0), dim=-1
        ).mean())
    kd_loss = torch.stack(kd_terms).mean()
    return action_loss, kd_loss


@torch.no_grad()
def evaluate(model, processor, tokenizer, stats: dict, split: str) -> dict:
    rows = read_manifest(split)
    if not rows:
        return {"split": split, "samples": 0, "action_loss": None, "kd_loss": None}
    rng = random.Random(CONFIG.seed)
    if split == "val" and len(rows) > CONFIG.validation_samples:
        rows = rng.sample(rows, CONFIG.validation_samples)
    was_training = model.training
    model.eval()
    action_total = kd_total = 0.0
    for sample in rows:
        with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            action_loss, kd_loss = losses(model, processor, tokenizer, stats, sample)
        action_total += float(action_loss.item())
        kd_total += float(kd_loss.item())
    if was_training:
        model.train()
    return {"split": split, "samples": len(rows),
            "action_loss": action_total / len(rows), "kd_loss": kd_total / len(rows)}


def save_adapter(model, step: int, name: str) -> None:
    path = CONFIG.output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    shutil.copy2(CONFIG.work_dir / "dataset_statistics.json", path / "dataset_statistics.json")
    (path / "training_step.json").write_text(json.dumps({"step": step}), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("KD training requires a CUDA GPU")
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    train_rows = read_manifest("train")
    if not train_rows:
        raise ValueError("No train samples")
    for split in ("train", "val"):
        for sample in read_manifest(split):
            for layer in CONFIG.kd_layers:
                if not teacher_feature_path(sample, layer).is_file():
                    raise FileNotFoundError(f"Teacher cache missing for {sample['sample_id']}")

    from prismatic.vla.action_tokenizer import ActionTokenizer

    model, processor = load_student(training=True)
    model.train()
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    stats = load_action_stats()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay,
    )
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(CONFIG).items()}, indent=2), encoding="utf-8"
    )
    log_path = CONFIG.output_dir / "metrics.jsonl"
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    update = 0
    micro = 0
    shuffled = []
    while update < CONFIG.max_train_steps:
        if not shuffled:
            shuffled = train_rows.copy()
            random.shuffle(shuffled)
        sample = shuffled.pop()
        with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            action_loss, kd_loss = losses(model, processor, action_tokenizer, stats, sample)
            loss = action_loss + CONFIG.kd_weight * kd_loss
        (loss / CONFIG.gradient_accumulation_steps).backward()
        micro += 1
        if micro % CONFIG.gradient_accumulation_steps:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        record = {"step": update, "train_action_loss": float(action_loss.item()),
                  "train_kd_loss": float(kd_loss.item()), "train_total_loss": float(loss.item())}
        if update % CONFIG.validation_every == 0 or update == CONFIG.max_train_steps:
            val = evaluate(model, processor, action_tokenizer, stats, "val")
            record.update({"val_action_loss": val["action_loss"],
                           "val_kd_loss": val["kd_loss"], "val_samples": val["samples"]})
            if val["action_loss"] is not None:
                val_total = val["action_loss"] + CONFIG.kd_weight * val["kd_loss"]
                if val_total < best_val:
                    best_val = val_total
                    save_adapter(model, update, "best_adapter")
        if update % CONFIG.checkpoint_every == 0 or update == CONFIG.max_train_steps:
            save_adapter(model, update, "latest_adapter")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    if best_val == float("inf"):
        save_adapter(model, update, "best_adapter")


if __name__ == "__main__":
    main()

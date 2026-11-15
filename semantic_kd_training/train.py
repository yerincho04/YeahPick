"""Joint OpenVLA LoRA training: action CE, five-type VQA CE, grounding CE, gated KD."""
from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import CONFIG
from model_io import (action_inputs, load_action_stats, load_student, qa_inputs,
                      read_manifest, teacher_feature_path, teacher_result_path)


def to_device(values: dict) -> dict:
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return {k: v.to("cuda", dtype=dtype) if k == "pixel_values" else v.to("cuda")
            for k, v in values.items()}


def cached_teacher(sample: dict) -> tuple[bool, dict[int, torch.Tensor]]:
    path = teacher_result_path(sample)
    if not path.is_file():
        raise FileNotFoundError(f"Run cache_teacher.py first: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result["sample_id"] != sample["sample_id"] or
            result["question"] != sample["question"] or
            result["answer"] != sample["answer"] or
            result["image"] != sample["image"]):
        raise ValueError(f"Stale teacher cache: {sample['sample_id']}")
    correct = bool(result["teacher_correct"])
    vectors = {}
    if correct:
        for layer in CONFIG.kd_layers:
            feature = teacher_feature_path(sample, layer)
            if not feature.is_file():
                raise FileNotFoundError(feature)
            vectors[layer] = torch.from_numpy(np.load(feature).copy()).float().cuda()
    return correct, vectors


def action_loss(model, processor, tokenizer, stats: dict, sample: dict) -> torch.Tensor:
    values = to_device(action_inputs(sample, processor, tokenizer, stats))
    output = model(**values, use_cache=False, return_dict=True)
    if not torch.isfinite(output.loss):
        raise FloatingPointError(f"Nonfinite action loss: {sample['sample_id']}")
    return output.loss


def qa_loss(model, processor, sample: dict) -> tuple[torch.Tensor, torch.Tensor, bool]:
    values, prefix_len = qa_inputs(sample, processor)
    values = to_device(values)
    teacher_correct, vectors = cached_teacher(sample)
    output = model(**values, output_hidden_states=teacher_correct,
                   use_cache=False, return_dict=True)
    ce = output.loss
    if not torch.isfinite(ce):
        raise FloatingPointError(f"Nonfinite QA loss: {sample['sample_id']}")
    kd = ce.new_zeros(())
    if teacher_correct:
        # OpenVLA inserts vision patches after BOS. Measure the last question token,
        # before any answer tokens, at the same causal position used by the teacher.
        image_tokens = output.hidden_states[0].shape[1] - values["input_ids"].shape[1]
        position = prefix_len - 1 + image_tokens
        terms = []
        for layer, vector in vectors.items():
            student = output.hidden_states[layer][0, position].float()
            if student.shape != vector.shape:
                raise ValueError(f"Teacher/student width mismatch at layer {layer}")
            terms.append(1 - F.cosine_similarity(student[None], vector[None]).mean())
        kd = torch.stack(terms).mean()
    # Teacher-forced exact answer tokens, for monitoring only.
    labels = values["labels"][0, 1:]
    predictions = output.logits[0, :-1].argmax(dim=-1)
    valid = labels != -100
    exact = bool(torch.equal(predictions[valid], labels[valid]))
    return ce, kd, exact


def choose(rows: list[dict], pool: list[dict]) -> dict:
    if not pool:
        pool.extend(rows)
        random.shuffle(pool)
    return pool.pop()


@torch.no_grad()
def evaluate(model, processor, tokenizer, stats: dict, split: str) -> dict:
    rng = random.Random(CONFIG.seed)
    was_training = model.training
    model.eval()
    result = {"split": split}
    for kind, suffix in (("action", ""), ("vqa", "_vqa"), ("grounding", "_grounding")):
        rows = read_manifest(split, suffix)
        if split == "val" and len(rows) > CONFIG.validation_samples:
            rows = rng.sample(rows, CONFIG.validation_samples)
        total = kd_total = exact = teacher_count = 0.0
        for sample in rows:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                if kind == "action":
                    total += float(action_loss(model, processor, tokenizer, stats, sample).item())
                else:
                    ce, kd, matches = qa_loss(model, processor, sample)
                    total += float(ce.item())
                    exact += matches
                    correct = json.loads(teacher_result_path(sample).read_text(encoding="utf-8"))["teacher_correct"]
                    teacher_count += correct
                    if correct:
                        kd_total += float(kd.item())
        result[f"{kind}_samples"] = len(rows)
        result[f"{kind}_loss"] = total / len(rows) if rows else None
        if kind != "action":
            result[f"{kind}_teacher_forced_exact"] = exact / len(rows) if rows else None
            result[f"{kind}_teacher_correct"] = int(teacher_count)
            result[f"{kind}_kd_loss"] = kd_total / teacher_count if teacher_count else None
    if was_training:
        model.train()
    return result


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
    action_rows = read_manifest("train")
    vqa_rows = read_manifest("train", "_vqa")
    grounding_rows = read_manifest("train", "_grounding")
    if not all((action_rows, vqa_rows, grounding_rows)):
        raise ValueError("Train action, VQA and grounding manifests must all contain samples")
    for split in ("train", "val"):
        for suffix in ("_vqa", "_grounding"):
            for sample in read_manifest(split, suffix):
                if not teacher_result_path(sample).is_file():
                    raise FileNotFoundError(f"Teacher cache missing: {sample['sample_id']}")
    from prismatic.vla.action_tokenizer import ActionTokenizer
    model, processor = load_student(training=True)
    model.train()
    tokenizer = ActionTokenizer(processor.tokenizer)
    stats = load_action_stats()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.output_dir / "run_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(CONFIG).items()},
                   indent=2), encoding="utf-8")
    action_pool, vqa_pool, grounding_pool = [], [], []
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    total_updates = math.ceil(CONFIG.train_epochs * len(action_rows) /
                              CONFIG.gradient_accumulation_steps)
    for update in range(1, total_updates + 1):
        last = {}
        for _ in range(CONFIG.gradient_accumulation_steps):
            action = choose(action_rows, action_pool)
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                ce_action = action_loss(model, processor, tokenizer, stats, action)
            (ce_action / CONFIG.gradient_accumulation_steps).backward()
            kind = "vqa" if random.random() < CONFIG.vqa_aux_probability else "grounding"
            aux = choose(vqa_rows, vqa_pool) if kind == "vqa" else choose(grounding_rows, grounding_pool)
            weight = CONFIG.vqa_weight if kind == "vqa" else CONFIG.grounding_weight
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                ce_qa, kd, _ = qa_loss(model, processor, aux)
                aux_total = weight * (ce_qa + CONFIG.kd_weight * kd)
            (aux_total / CONFIG.gradient_accumulation_steps).backward()
            last = {"train_action_loss": float(ce_action.item()),
                    "train_aux_kind": kind, "train_aux_loss": float(ce_qa.item()),
                    "train_aux_kd_loss": float(kd.item())}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {"step": update, **last}
        if update % CONFIG.validation_every == 0 or update == total_updates:
            val = evaluate(model, processor, tokenizer, stats, "val")
            record.update({f"val_{k}": v for k, v in val.items() if k != "split"})
            if all(val[f"{kind}_loss"] is not None for kind in ("action", "vqa", "grounding")):
                score = (val["action_loss"] + CONFIG.vqa_weight * val["vqa_loss"] +
                         CONFIG.grounding_weight * val["grounding_loss"])
                if score < best_val:
                    best_val = score
                    save_adapter(model, update, "best_adapter")
        if update % CONFIG.checkpoint_every == 0 or update == total_updates:
            save_adapter(model, update, "latest_adapter")
        with (CONFIG.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    if best_val == float("inf"):
        save_adapter(model, total_updates, "best_adapter")


if __name__ == "__main__":
    main()

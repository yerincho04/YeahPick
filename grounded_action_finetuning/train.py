"""Fine-tune the trained KD adapter so its tile decision conditions actions."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from config import CONFIG
from model_io import (load_action_stats, load_student, normalized_action,
                      qa_inputs, read_manifest)
from prompts import action_prompt


def device_inputs(values: dict) -> dict:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return {key: value.to(device, dtype=dtype) if key == "pixel_values"
            else value.to(device) for key, value in values.items()}


def action_inputs(sample: dict, processor, tokenizer, stats: dict) -> dict:
    action = normalized_action(sample["action"], stats)
    prompt = action_prompt(sample["instruction"], sample["answer_side"], tokenizer(action))
    ids = processor.tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"]
    labels = ids.clone()
    labels[:, :-(len(action) + 1)] = -100  # Seven action tokens and EOS.
    with Image.open(sample["image"]) as raw:
        pixel = processor.image_processor(
            images=raw.convert("RGB"), return_tensors="pt"
        )["pixel_values"]
    return {"input_ids": ids, "labels": labels, "pixel_values": pixel}


def action_loss(model, processor, tokenizer, stats: dict, sample: dict):
    output = model(**device_inputs(action_inputs(sample, processor, tokenizer, stats)),
                   use_cache=False, return_dict=True)
    if not torch.isfinite(output.loss):
        raise FloatingPointError(f"Nonfinite action loss: {sample['sample_id']}")
    return output.loss


def qa_loss(model, processor, sample: dict):
    values, prefix_len = qa_inputs(sample, processor)
    output = model(**device_inputs(values), use_cache=False, return_dict=True)
    if not torch.isfinite(output.loss):
        raise FloatingPointError(f"Nonfinite QA loss: {sample['sample_id']}")
    return output.loss, output, prefix_len, values["input_ids"].shape[1]


def choose(rows: list[dict], pool: list[dict]) -> dict:
    if not pool:
        pool.extend(rows)
        random.shuffle(pool)
    return pool.pop()


def side_token_ids(processor) -> tuple[int, int]:
    ids = []
    for side in ("Left", "Right"):
        tokens = processor.tokenizer(
            " " + side, add_special_tokens=False
        )["input_ids"]
        if len(tokens) != 1:
            raise ValueError(f"{side} must be one token for binary side scoring: {tokens}")
        ids.append(tokens[0])
    return ids[0], ids[1]


def selected_vqa_rows(rows: list[dict]) -> list[dict]:
    """Two questions per category/type, preferring opposite answer labels."""
    strata = defaultdict(list)
    for row in rows:
        strata[row["category"], row["question_type"]].append(row)
    selected = []
    for _, candidates in sorted(strata.items()):
        by_answer = defaultdict(list)
        for row in candidates:
            by_answer[row["answer"].strip().casefold()].append(row)
        if len(by_answer) >= 2:
            labels = sorted(by_answer)
            selected.extend((by_answer[labels[0]][0], by_answer[labels[1]][0]))
        else:
            selected.extend(candidates[:2])
    return selected


def validate(model, processor, tokenizer, stats: dict) -> dict:
    model.eval()
    grounding = read_manifest("val", "_grounding")
    vqa = selected_vqa_rows(read_manifest("val", "_vqa"))
    actions = read_manifest("val")
    side_ids = side_token_ids(processor)
    by_category = defaultdict(list)
    for sample in grounding:
        by_category[sample["category"]].append(sample)
    # Keep action validation bounded, while using every grounding question.
    selected_episodes = set()
    for category, episodes in sorted(by_category.items()):
        for side in ("Left", "Right"):
            candidates = [row for row in episodes if row["answer"] == side]
            if candidates:
                selected_episodes.add(candidates[0]["episode_id"])
    by_episode = defaultdict(list)
    for sample in actions:
        if sample["episode_id"] in selected_episodes:
            by_episode[sample["episode_id"]].append(sample)
    selected_actions = []
    for episode in sorted(by_episode):
        frames = sorted(by_episode[episode], key=lambda row: row["step"])
        count = min(5, len(frames))
        indices = ([0] if count == 1 else
                   [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)])
        selected_actions.extend(frames[i] for i in indices)
    if not grounding or not selected_actions:
        raise ValueError("Validation needs grounding and action samples")
    action_total = grounding_total = vqa_total = 0.0
    correct = 0
    with torch.no_grad():
        for sample in selected_actions:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                action_total += float(action_loss(model, processor, tokenizer, stats, sample))
        for sample in grounding:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                loss, output, prefix_len, input_len = qa_loss(model, processor, sample)
            grounding_total += float(loss)
            image_tokens = output.logits.shape[1] - input_len
            scores = output.logits[0, prefix_len - 1 + image_tokens, list(side_ids)]
            prediction = "Left" if scores[0] > scores[1] else "Right"
            correct += prediction == sample["answer"]
        for sample in vqa:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                loss, _, _, _ = qa_loss(model, processor, sample)
            vqa_total += float(loss)
    model.train()
    return {"val_action_loss": action_total / len(selected_actions),
            "val_action_samples": len(selected_actions),
            "val_grounding_loss": grounding_total / len(grounding),
            "val_grounding_correct": correct,
            "val_grounding_samples": len(grounding),
            "val_grounding_accuracy": correct / len(grounding),
            "val_vqa_loss": vqa_total / len(vqa) if vqa else None,
            "val_vqa_samples": len(vqa)}


def main() -> None:
    run_dir = Path(os.environ["GROUNDED_RUN_DIR"]).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not (CONFIG.kd_adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Copy KD best_adapter into {CONFIG.kd_adapter_dir}")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != CONFIG.student_gpu_count or torch.cuda.device_count() != world_size:
        raise RuntimeError(f"Expected {CONFIG.student_gpu_count} visible GPUs/processes")
    if CONFIG.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by GPU count")
    torch.cuda.set_device(rank)
    backend = os.environ.get("KD_DDP_BACKEND", "nccl")
    if dist.is_initialized():
        if (dist.get_rank() != rank or dist.get_world_size() != world_size
                or dist.get_backend() != backend):
            raise RuntimeError(
                "An existing process group has different rank, world size, or backend: "
                f"rank={dist.get_rank()} world_size={dist.get_world_size()} "
                f"backend={dist.get_backend()}"
            )
        print(f"Reusing initialized {backend} process group on rank {rank}", flush=True)
    else:
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    action_rows = read_manifest("train")
    vqa_rows = read_manifest("train", "_vqa")
    grounding_rows = read_manifest("train", "_grounding")
    if not all((action_rows, vqa_rows, grounding_rows)):
        raise ValueError("Train manifests must all contain samples")
    from prismatic.vla.action_tokenizer import ActionTokenizer

    student, processor = load_student(training=True, adapter_path=CONFIG.kd_adapter_dir)
    model = DDP(student, device_ids=[rank], output_device=rank,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    stats = load_action_stats()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    if rank == 0:
        (run_dir / "run_config.json").write_text(
            json.dumps({k: str(v) if isinstance(v, Path) else v
                        for k, v in asdict(CONFIG).items()}, indent=2), encoding="utf-8")
    model.train()
    action_pool, grounding_pool, vqa_pool = [], [], []
    local_steps = CONFIG.gradient_accumulation_steps // world_size
    steps_per_epoch = math.ceil(len(action_rows) / CONFIG.gradient_accumulation_steps)
    best = (-1.0, float("inf"))
    for epoch in range(1, CONFIG.train_epochs + 1):
        for update_in_epoch in range(1, steps_per_epoch + 1):
            if rank == 0:
                plan = []
                for _ in range(CONFIG.gradient_accumulation_steps):
                    action = choose(action_rows, action_pool)
                    kind = "grounding" if random.random() < CONFIG.grounding_probability else "vqa"
                    aux = choose(grounding_rows if kind == "grounding" else vqa_rows,
                                 grounding_pool if kind == "grounding" else vqa_pool)
                    plan.append((action, kind, aux))
            else:
                plan = None
            payload = [plan]
            dist.broadcast_object_list(payload, src=0)
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(5, device=torch.device("cuda", rank), dtype=torch.float64)
            for local_index, (action, kind, aux) in enumerate(payload[0][rank::world_size]):
                with model.no_sync():
                    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                        act = action_loss(model, processor, tokenizer, stats, action)
                    (act / local_steps).backward()
                sync = nullcontext() if local_index == local_steps - 1 else model.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                        aux_loss, _, _, _ = qa_loss(model, processor, aux)
                        weight = (CONFIG.grounding_weight if kind == "grounding"
                                  else CONFIG.vqa_weight)
                    ((weight * aux_loss) / local_steps).backward()
                sums[0] += float(act)
                index = 1 if kind == "grounding" else 2
                sums[index] += float(aux_loss)
                sums[index + 2] += 1
            optimizer.step()
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            if rank == 0 and (update_in_epoch % 100 == 0 or update_in_epoch == steps_per_epoch):
                record = {"epoch": epoch, "step": (epoch - 1) * steps_per_epoch + update_in_epoch,
                          "train_action_loss": sums[0].item() / CONFIG.gradient_accumulation_steps,
                          "train_grounding_samples": int(sums[3].item()),
                          "train_grounding_loss": (sums[1].item() / sums[3].item()
                                                   if sums[3].item() else None),
                          "train_vqa_samples": int(sums[4].item()),
                          "train_vqa_loss": (sums[2].item() / sums[4].item()
                                             if sums[4].item() else None)}
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if rank == 0:
            metrics = validate(model.module, processor, tokenizer, stats)
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:02d}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            model.module.save_pretrained(checkpoint)
            (checkpoint / "dataset_statistics.json").write_text(
                (CONFIG.work_dir / "dataset_statistics.json").read_text(encoding="utf-8"),
                encoding="utf-8")
            score = (metrics["val_grounding_accuracy"], metrics["val_action_loss"])
            if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
                best = score
                (run_dir / "best.json").write_text(
                    json.dumps({"epoch": epoch, "adapter": str(checkpoint.resolve()),
                                "val_grounding_accuracy": score[0],
                                "val_action_loss": score[1]}, indent=2), encoding="utf-8")
            record = {"epoch": epoch, "step": epoch * steps_per_epoch, **metrics}
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

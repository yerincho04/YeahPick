"""Train a stable semantic side decision as the prefix of every action."""

from __future__ import annotations

import json
import os
import random
import shutil
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

from bridge import policy_inputs, score_side, token_span_loss
from config import CONFIG
from model_io import load_action_stats, load_student, qa_inputs, read_manifest


def device_inputs(values: dict) -> dict:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return {k: v.to(device, dtype=dtype) if k == "pixel_values" else v.to(device)
            for k, v in values.items()}


def policy_losses(model, processor, tokenizer, stats: dict, sample: dict):
    values, spans = policy_inputs(sample, processor, tokenizer, stats)
    values = device_inputs(values)
    output = model(**values, use_cache=False, return_dict=True)
    action = token_span_loss(output.logits, values["input_ids"], *spans["action"])
    side = (token_span_loss(output.logits, values["input_ids"], *spans["side"])
            if sample["step"] == 0 else None)
    if not torch.isfinite(action) or (side is not None and not torch.isfinite(side)):
        raise FloatingPointError(f"Nonfinite policy loss: {sample['sample_id']}")
    return action, side


def vqa_loss(model, processor, sample: dict):
    values, _ = qa_inputs(sample, processor)
    output = model(**device_inputs(values), use_cache=False, return_dict=True)
    if not torch.isfinite(output.loss):
        raise FloatingPointError(f"Nonfinite VQA loss: {sample['sample_id']}")
    return output.loss


class BalancedSampler:
    """Cycle category/label strata and draw each episode equally often."""

    def __init__(self, rows: list[dict], key, seed: int):
        self.groups = defaultdict(list)
        for row in rows:
            self.groups[key(row)].append(row)
        self.keys = sorted(self.groups)
        if not self.keys:
            raise ValueError("Empty balanced sampler")
        self.rng = random.Random(seed)
        self.rng.shuffle(self.keys)
        self.pools = {}
        self.position = 0

    def draw(self) -> dict:
        key = self.keys[self.position]
        self.position = (self.position + 1) % len(self.keys)
        if self.position == 0:
            self.rng.shuffle(self.keys)
        if not self.pools.get(key):
            self.pools[key] = self.groups[key].copy()
            self.rng.shuffle(self.pools[key])
        return self.pools[key].pop()


def validation_vqa(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"], row["question_type"]].append(row)
    selected = []
    for candidates in groups.values():
        answers = defaultdict(list)
        for row in candidates:
            answers[row["answer"].casefold()].append(row)
        labels = sorted(answers)
        selected.append(answers[labels[0]][0])
        selected.append(answers[labels[-1]][0] if len(labels) > 1 else candidates[1])
    return selected


@torch.no_grad()
def validate(model, processor, tokenizer, stats, first, later, vqa, warmup=False):
    model.eval()
    correct = 0
    by_category = defaultdict(lambda: [0, 0])
    by_side = defaultdict(lambda: [0, 0])
    for row in first:
        with Image.open(row["image"]) as raw:
            prediction = score_side(model, processor, raw.convert("RGB"),
                                    row["instruction"])
        matched = prediction == row["answer_side"]
        correct += matched
        by_category[row["category"]][0] += matched
        by_category[row["category"]][1] += 1
        by_side[row["answer_side"]][0] += matched
        by_side[row["answer_side"]][1] += 1
    side_total = action_total = vqa_total = 0.0
    selected_side = []
    side_groups = defaultdict(list)
    for row in first:
        side_groups[row["category"], row["answer_side"]].append(row)
    for rows in side_groups.values():
        selected_side.extend(rows[:2])
    selected_actions = []
    initial_groups = defaultdict(list)
    later_episodes = defaultdict(list)
    for row in first:
        initial_groups[row["category"], row["answer_side"]].append(row)
    for row in later:
        later_episodes[row["category"], row["answer_side"], row["episode_id"]].append(row)
    selected_actions.extend(rows[0] for _, rows in sorted(initial_groups.items()))
    chosen_later = set()
    for (category, side, episode), frames in sorted(later_episodes.items()):
        if (category, side) in chosen_later:
            continue
        chosen_later.add((category, side))
        frames.sort(key=lambda row: row["step"])
        selected_actions.extend(frames[round(i * (len(frames) - 1) / 4)]
                                for i in range(5))
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        for row in selected_side:
            _, side = policy_losses(model, processor, tokenizer, stats, row)
            side_total += float(side)
        if not warmup:
            for row in selected_actions:
                action, _ = policy_losses(model, processor, tokenizer, stats, row)
                action_total += float(action)
        for row in vqa:
            vqa_total += float(vqa_loss(model, processor, row))
    model.train()
    return {"val_side_correct": correct, "val_side_samples": len(first),
            "val_side_accuracy": correct / len(first),
            "val_side_by_category": {k: {"correct": v[0], "total": v[1]}
                                     for k, v in sorted(by_category.items())},
            "val_side_by_side": {k: {"correct": v[0], "total": v[1]}
                                 for k, v in sorted(by_side.items())},
            "val_side_loss": side_total / len(selected_side),
            "val_action_loss": action_total / len(selected_actions) if not warmup else None,
            "val_action_samples": len(selected_actions) if not warmup else 0,
            "val_vqa_loss": vqa_total / len(vqa), "val_vqa_samples": len(vqa)}


def main() -> None:
    run_dir = Path(os.environ["BRIDGE_RUN_DIR"]).resolve()
    rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world_size != CONFIG.student_gpu_count or torch.cuda.device_count() != world_size:
        raise RuntimeError(f"Expected {CONFIG.student_gpu_count} GPUs/processes")
    if CONFIG.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by GPU count")
    if not 0 < CONFIG.first_frame_probability < 1:
        raise ValueError("first_frame_probability must be between 0 and 1")
    torch.cuda.set_device(rank)
    backend = os.environ.get("BRIDGE_DDP_BACKEND", "nccl")
    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (rank, world_size, backend):
            raise RuntimeError("Existing process group does not match this run")
    else:
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    random.seed(CONFIG.seed)

    actions = read_manifest("train")
    vqa = read_manifest("train", "_vqa")
    val_actions = read_manifest("val")
    val_vqa = validation_vqa(read_manifest("val", "_vqa"))
    first = [r for r in actions if r["step"] == 0]
    later_by_episode = defaultdict(list)
    for row in actions:
        if row["step"] > 0:
            later_by_episode[row["episode_id"]].append(row)
    later_episodes = [rows[0] for rows in later_by_episode.values()]
    val_first = [r for r in val_actions if r["step"] == 0]
    val_later = [r for r in val_actions if r["step"] > 0]
    if not all((first, later_episodes, vqa, val_first, val_later, val_vqa)):
        raise ValueError("Missing first-frame, later-frame, or VQA samples")
    first_sampler = BalancedSampler(first, lambda r: (r["category"], r["answer_side"]), CONFIG.seed)
    later_sampler = BalancedSampler(later_episodes,
                                    lambda r: (r["category"], r["answer_side"]), CONFIG.seed + 1)
    vqa_sampler = BalancedSampler(vqa,
                                  lambda r: (r["category"], r["question_type"]), CONFIG.seed + 2)
    frame_rng = random.Random(CONFIG.seed + 3)

    from prismatic.vla.action_tokenizer import ActionTokenizer
    student, processor = load_student(training=True)
    model = DDP(student, device_ids=[rank], output_device=rank,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    stats = load_action_stats()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)

    def save_checkpoint(name, stage, step):
        checkpoint = run_dir / "checkpoints" / name
        checkpoint.mkdir(parents=True, exist_ok=False)
        model.module.save_pretrained(checkpoint)
        shutil.copy2(CONFIG.work_dir / "dataset_statistics.json",
                     checkpoint / "dataset_statistics.json")
        (run_dir / "latest.json").write_text(json.dumps(
            {"stage": stage, "step": step, "adapter": str(checkpoint.resolve())}, indent=2),
            encoding="utf-8")
        return checkpoint

    if rank == 0:
        snapshot = run_dir / "input_snapshot"
        snapshot.mkdir(exist_ok=False)
        for name in ("dataset_statistics.json", "pair_splits.json",
                     "test.jsonl", "test_vqa.jsonl", "test_grounding.jsonl"):
            shutil.copy2(CONFIG.work_dir / name, snapshot / name)
        (run_dir / "run_config.json").write_text(json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CONFIG).items()},
            indent=2), encoding="utf-8")
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            policy_losses(model.module, processor, tokenizer, stats, first[0])
            policy_losses(model.module, processor, tokenizer, stats,
                          later_by_episode[later_episodes[0]["episode_id"]][0])
            vqa_loss(model.module, processor, vqa[0])
        with Image.open(first[0]["image"]) as raw:
            score_side(model.module, processor, raw.convert("RGB"), first[0]["instruction"])
        model.train()
        print("PREFLIGHT_OK", flush=True)
    dist.barrier()

    local_steps = CONFIG.gradient_accumulation_steps // world_size
    global_step = 0
    best = None
    stages = [("warmup", 1, CONFIG.warmup_updates)] + [
        ("joint", epoch, CONFIG.joint_updates_per_epoch)
        for epoch in range(1, CONFIG.joint_epochs + 1)]
    for stage, epoch, updates in stages:
        for update_index in range(1, updates + 1):
            if rank == 0:
                plan = []
                for _ in range(CONFIG.gradient_accumulation_steps):
                    use_first = stage == "warmup" or frame_rng.random() < CONFIG.first_frame_probability
                    if use_first:
                        policy = first_sampler.draw()
                    else:
                        episode = later_sampler.draw()["episode_id"]
                        policy = frame_rng.choice(later_by_episode[episode])
                    plan.append((policy, vqa_sampler.draw()))
            else:
                plan = None
            payload = [plan]
            dist.broadcast_object_list(payload, src=0)
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(4, device=torch.device("cuda", rank), dtype=torch.float64)
            for local_index, (policy, qa) in enumerate(payload[0][rank::world_size]):
                with model.no_sync():
                    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                        action, side = policy_losses(model, processor, tokenizer, stats, policy)
                        if stage == "warmup":
                            loss = CONFIG.side_weight * side
                        else:
                            loss = CONFIG.action_weight * action
                            if side is not None:
                                loss = loss + (CONFIG.side_weight /
                                               CONFIG.first_frame_probability) * side
                    (loss / local_steps).backward()
                sync = nullcontext() if local_index == local_steps - 1 else model.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                        aux = vqa_loss(model, processor, qa)
                    (CONFIG.vqa_weight * aux / local_steps).backward()
                sums[0] += float(action) if stage == "joint" else 0
                sums[1] += float(side) if side is not None else 0
                sums[2] += float(side is not None)
                sums[3] += float(aux)
            optimizer.step()
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            global_step += 1
            if (rank == 0 and CONFIG.checkpoint_every_updates > 0 and
                    global_step % CONFIG.checkpoint_every_updates == 0):
                save_checkpoint(f"step_{global_step:06d}", stage, global_step)
            if rank == 0 and (update_index % 100 == 0 or update_index == updates):
                record = {"stage": stage, "epoch": epoch, "step": global_step,
                          "train_action_loss": sums[0].item() / CONFIG.gradient_accumulation_steps
                          if stage == "joint" else None,
                          "train_side_loss": (sums[1].item() / sums[2].item()
                                              if sums[2].item() else None),
                          "train_side_samples": int(sums[2].item()),
                          "train_vqa_loss": sums[3].item() / CONFIG.gradient_accumulation_steps}
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if rank == 0:
            checkpoint = save_checkpoint(f"{stage}_epoch_{epoch:02d}", stage, global_step)
            metrics = validate(model.module, processor, tokenizer, stats,
                               val_first, val_later, val_vqa, warmup=stage == "warmup")
            record = {"stage": stage, "epoch": epoch, "step": global_step, **metrics}
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if stage == "joint":
                score = (metrics["val_side_accuracy"], -metrics["val_action_loss"],
                         -metrics["val_vqa_loss"])
                if best is None or score > best:
                    best = score
                    (run_dir / "best.json").write_text(json.dumps(
                        {"stage": stage, "epoch": epoch, "step": global_step,
                         "adapter": str(checkpoint.resolve()),
                         "val_side_accuracy": metrics["val_side_accuracy"],
                         "val_action_loss": metrics["val_action_loss"]}, indent=2),
                        encoding="utf-8")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

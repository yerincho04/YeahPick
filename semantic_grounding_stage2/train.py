"""Continue the Stage 1 adapter with conditioned action CE and side replay."""
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

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from bridge import bridge_prompt, policy_inputs, score_side, side_choice_tokens, token_span_loss
from config import CONFIG
from model_io import load_action_stats, load_student, read_manifest


def device_inputs(values: dict) -> dict:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return {k: v.to(device, dtype=dtype) if k == "pixel_values" else v.to(device)
            for k, v in values.items()}


def action_loss(model, processor, action_tokenizer, stats, sample):
    values, spans = policy_inputs(sample, processor, action_tokenizer, stats)
    values = device_inputs(values)
    output = model(**values, use_cache=False, return_dict=True)
    loss = token_span_loss(output.logits, values["input_ids"], *spans["action"])
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"Nonfinite action loss: {sample['sample_id']}")
    return loss


def side_loss(model, processor, sample):
    tokenizer = processor.tokenizer
    prefix = tokenizer(bridge_prompt(sample["instruction"]), add_special_tokens=True,
                       return_tensors="pt")["input_ids"]
    common, left, right = side_choice_tokens(tokenizer)
    if common:
        prefix = torch.cat((prefix, torch.tensor([common], dtype=prefix.dtype)), dim=1)
    with Image.open(sample["image"]) as raw:
        pixel = processor.image_processor(images=raw.convert("RGB"),
                                          return_tensors="pt")["pixel_values"]
    values = device_inputs({"input_ids": prefix, "pixel_values": pixel})
    logits = model(**values, use_cache=False, return_dict=True).logits[:, -1, [left, right]].float()
    target = torch.tensor([0 if sample["answer"] == "Left" else 1], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, target)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"Nonfinite side loss: {sample['sample_id']}")
    return loss


def sample_epoch(rows: list[dict], epoch: int) -> list[dict]:
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)
    rng = random.Random(CONFIG.seed + epoch)
    selected = []
    n = CONFIG.frames_per_episode_per_epoch
    for frames in episodes.values():
        frames.sort(key=lambda r: r["step"])
        if len(frames) < n:
            raise ValueError(f"Episode has fewer than {n} frames: {frames[0]['episode_id']}")
        for part in range(n):
            start = part * len(frames) // n
            end = max(start + 1, (part + 1) * len(frames) // n)
            selected.append(frames[rng.randrange(start, end)])
    # Equal episode sampling preserves category, layout and conditioned-side balance.
    rng.shuffle(selected)
    return selected


@torch.inference_mode()
def validate(model, processor, tokenizer, stats, ground, actions):
    model.eval()
    correct = 0
    by_category = defaultdict(lambda: [0, 0])
    for row in ground:
        with Image.open(row["image"]) as raw:
            prediction = score_side(model, processor, raw.convert("RGB"), row["instruction"])
        hit = prediction == row["answer"]
        correct += hit
        by_category[row["category"]][0] += hit
        by_category[row["category"]][1] += 1
    losses = []
    for row in actions[:40]:
        with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            losses.append(float(action_loss(model, processor, tokenizer, stats, row)))
    model.train()
    return {"side_accuracy": correct / len(ground), "side_correct": correct,
            "side_total": len(ground), "action_loss": sum(losses) / len(losses),
            "by_category": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_category.items())}}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from prismatic.vla.action_tokenizer import ActionTokenizer

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != CONFIG.student_gpu_count or torch.cuda.device_count() != world:
        raise RuntimeError(f"Expected {CONFIG.student_gpu_count} visible GPUs/processes")
    if CONFIG.batch_size % world or CONFIG.batch_size < world:
        raise ValueError("Global action batch must be divisible by GPU count")
    torch.cuda.set_device(rank)
    backend = os.environ.get("STAGE2_DDP_BACKEND", "nccl")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    train = read_manifest("train")
    ground = read_manifest("train", "_grounding")
    val_action = read_manifest("val")
    val_ground = read_manifest("val", "_grounding")
    if not train or not ground or not val_action or not val_ground:
        raise ValueError("Prepared data is incomplete")
    stage1 = json.loads((CONFIG.stage1_run / "best.json").read_text(encoding="utf-8"))
    source_adapter = Path(stage1["adapter"])
    student, processor = load_student(training=True, adapter_path=source_adapter)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    side_choice_tokens(processor.tokenizer)
    stats = load_action_stats()
    if not any("projector." in name and param.requires_grad for name, param in student.named_parameters()):
        raise RuntimeError("Stage 1 projector LoRA is not trainable")
    model = DDP(student, device_ids=[rank], output_device=rank,
                broadcast_buffers=False, find_unused_parameters=False)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    run_dir = Path(os.environ["STAGE2_RUN_DIR"]).resolve()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = run_dir / "input_snapshot"
        snapshot.mkdir()
        for name in ("pair_splits.json", "dataset_statistics.json", "train.jsonl",
                     "val.jsonl", "test.jsonl", "train_grounding.jsonl",
                     "val_grounding.jsonl", "test_grounding.jsonl"):
            shutil.copy2(CONFIG.work_dir / name, snapshot / name)
        write_json(run_dir / "run_config.json", {"config": {k: str(v) if isinstance(v, Path) else v
                                                           for k, v in asdict(CONFIG).items()},
                                                 "stage1_adapter": str(source_adapter)})
    dist.barrier()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        action_loss(model.module, processor, action_tokenizer, stats, train[rank])
        side_loss(model.module, processor, ground[rank])
    def grad_norm() -> float:
        return sum(float(p.grad.float().square().sum()) for p in model.module.parameters()
                   if p.requires_grad and p.grad is not None) ** .5
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        action_loss(model.module, processor, action_tokenizer, stats, train[rank]).backward()
    action_norm = grad_norm()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        side_loss(model.module, processor, ground[rank]).backward()
    side_norm = grad_norm()
    optimizer.zero_grad(set_to_none=True)
    side_weight = torch.tensor([
        max(CONFIG.side_weight_min, min(CONFIG.side_weight_max,
            CONFIG.side_gradient_ratio * action_norm / max(side_norm, 1e-8)))
    ], device=torch.device("cuda", rank))
    dist.broadcast(side_weight, src=0)
    if rank == 0:
        print(f"STAGE2_PREFLIGHT_OK side_weight={float(side_weight.item()):.6f} "
              f"action_grad_norm={action_norm:.6f} side_grad_norm={side_norm:.6f}", flush=True)
    dist.barrier()
    local_batch = CONFIG.batch_size // world
    for epoch in range(1, CONFIG.epochs + 1):
        chosen = sample_epoch(train, epoch)
        if len(chosen) % CONFIG.batch_size:
            chosen = chosen[:len(chosen) // CONFIG.batch_size * CONFIG.batch_size]
        order = ground.copy()
        random.Random(CONFIG.seed + 1000 + epoch).shuffle(order)
        totals = [0.0, 0.0]
        updates = len(chosen) // CONFIG.batch_size
        model.train()
        for update in range(updates):
            optimizer.zero_grad(set_to_none=True)
            action_batch = chosen[update * CONFIG.batch_size + rank * local_batch:
                                  update * CONFIG.batch_size + (rank + 1) * local_batch]
            for sample in action_batch:
                with model.no_sync():
                    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                        loss = action_loss(model, processor, action_tokenizer, stats, sample)
                    (loss / local_batch).backward()
                totals[0] += float(loss.detach()) / local_batch
            # Each rank replays one balanced first-frame side example per update.
            side_sample = order[(update * world + rank) % len(order)]
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                sloss = side_loss(model, processor, side_sample)
            (side_weight.item() * sloss).backward()
            totals[1] += float(sloss.detach())
            optimizer.step()
            if rank == 0 and (update + 1) % 100 == 0:
                print(f"TRAIN epoch={epoch} update={update+1}/{updates}", flush=True)
        metrics = validate(model.module, processor, action_tokenizer, stats,
                           val_ground[rank::world], val_action[rank::world][:40])
        local = torch.tensor([totals[0], totals[1], metrics["side_correct"],
                              metrics["side_total"], metrics["action_loss"]],
                             dtype=torch.float64, device=torch.device("cuda", rank))
        dist.all_reduce(local)
        if rank == 0:
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:02d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.module.save_pretrained(checkpoint)
            shutil.copy2(CONFIG.work_dir / "dataset_statistics.json",
                         checkpoint / "dataset_statistics.json")
            record = {"epoch": epoch, "updates": updates,
                      "side_weight": side_weight.item(),
                      "train_action_loss": local[0].item() / (updates * world),
                      "train_side_loss": local[1].item() / (updates * world),
                      "val_side_correct": int(local[2].item()),
                      "val_side_total": int(local[3].item()),
                      "val_side_accuracy": local[2].item() / local[3].item(),
                      "val_action_loss": local[4].item() / world,
                      "adapter": str(checkpoint)}
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            write_json(run_dir / "latest.json", record)
            print(json.dumps(record), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

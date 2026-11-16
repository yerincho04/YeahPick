"""Joint OpenVLA LoRA training: action CE, five-type VQA CE, grounding CE, gated KD."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from config import CONFIG
from model_io import (action_inputs, load_action_stats, load_student, qa_inputs,
                      read_manifest, teacher_feature_path, teacher_result_path)


def to_device(values: dict) -> dict:
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    device = torch.device("cuda", torch.cuda.current_device())
    return {k: v.to(device, dtype=dtype) if k == "pixel_values" else v.to(device)
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
            vectors[layer] = torch.from_numpy(np.load(feature).copy()).float()
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
            terms.append(1 - F.cosine_similarity(
                student[None], vector.to(student.device)[None]
            ).mean())
        kd = torch.stack(terms).mean().to(ce.device)
    # The vision patches are inserted after BOS, so answer logits start later
    # than answer token IDs. Compare only the answer and EOS positions.
    image_tokens = output.logits.shape[1] - values["input_ids"].shape[1]
    answer_logits = output.logits[0, prefix_len - 1 + image_tokens:-1]
    targets = values["labels"][0, prefix_len:].to(answer_logits.device)
    exact = bool(torch.equal(answer_logits.argmax(dim=-1), targets))
    return ce, kd, exact


def choose(rows: list[dict], pool: list[dict]) -> dict:
    if not pool:
        pool.extend(rows)
        random.shuffle(pool)
    return pool.pop()


def selection_loss(metrics: dict) -> float:
    """Match checkpoint selection to the expected joint training objective."""
    score = metrics["action_loss"]
    for kind, probability, weight in (
        ("vqa", CONFIG.vqa_aux_probability, CONFIG.vqa_weight),
        ("grounding", 1 - CONFIG.vqa_aux_probability, CONFIG.grounding_weight),
    ):
        count = metrics[f"{kind}_samples"]
        kd_mean = metrics[f"{kind}_kd_loss"] or 0.0
        kd_all = kd_mean * metrics[f"{kind}_teacher_correct"] / count
        score += probability * weight * (metrics[f"{kind}_loss"] + CONFIG.kd_weight * kd_all)
    return score


def validation_vqa_rows(rows: list[dict], rng: random.Random) -> list[dict]:
    """Select both answer labels within every category and question type."""
    if CONFIG.validation_vqa_per_category_type != 2:
        raise ValueError("Balanced validation currently requires two VQA rows per category/type")
    strata = defaultdict(list)
    for row in rows:
        strata[row["category"], row["question_type"]].append(row)
    selected = []
    pair_usage = defaultdict(int)
    for key, candidates in sorted(strata.items()):
        by_answer = defaultdict(list)
        for row in candidates:
            by_answer[str(row["answer"]).strip().casefold()].append(row)
        labels = sorted(by_answer)
        if len(labels) == 2:
            first_group, second_group = by_answer[labels[0]].copy(), by_answer[labels[1]].copy()
        else:
            # A validation split can contain only one label for a particular
            # counterfactual stratum. Preserve focus-side balance in that case.
            first_group = [row for row in candidates if row["focus_tile"].lower() == "left"]
            second_group = [row for row in candidates if row["focus_tile"].lower() == "right"]
            if not first_group or not second_group:
                raise ValueError(f"Validation stratum lacks both focus sides: {key}")
        left, right = first_group, second_group
        rng.shuffle(left)
        rng.shuffle(right)
        first, second = min(
            ((a, b) for a in left for b in right),
            key=lambda pair: (
                pair[0]["focus_tile"] == pair[1]["focus_tile"],
                pair[0]["pair_id"] == pair[1]["pair_id"],
                pair_usage[key[0], pair[0]["pair_id"]] +
                pair_usage[key[0], pair[1]["pair_id"]],
            ),
        )
        selected.extend((first, second))
        pair_usage[key[0], first["pair_id"]] += 1
        pair_usage[key[0], second["pair_id"]] += 1
    return selected


def validation_episode_rows(grounding: list[dict], actions: list[dict],
                            rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Sample diverse episodes, then evenly spaced frames from each episode."""
    by_category = defaultdict(list)
    for row in grounding:
        by_category[row["category"]].append(row)
    selected_grounding = []
    for _, episodes in sorted(by_category.items()):
        episodes = episodes.copy()
        rng.shuffle(episodes)
        if not episodes:
            continue
        chosen = [episodes.pop()]
        while episodes and len(chosen) < CONFIG.validation_episodes_per_category:
            next_episode = min(
                episodes,
                key=lambda row: (
                    row["answer"] == chosen[0]["answer"],
                    row["pair_id"] == chosen[0]["pair_id"],
                ),
            )
            chosen.append(next_episode)
            episodes.remove(next_episode)
        selected_grounding.extend(chosen)
    actions_by_episode = defaultdict(list)
    for row in actions:
        actions_by_episode[row["episode_id"]].append(row)
    selected_actions = []
    for episode in selected_grounding:
        frames = sorted(actions_by_episode[episode["episode_id"]], key=lambda row: row["step"])
        if not frames:
            raise ValueError(f"No validation actions for {episode['episode_id']}")
        count = min(CONFIG.validation_steps_per_episode, len(frames))
        indices = ([0] if count == 1 else
                   [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)])
        selected_actions.extend(frames[index] for index in indices)
    return selected_grounding, selected_actions


@torch.no_grad()
def evaluate(model, processor, tokenizer, stats: dict, split: str) -> dict:
    rng = random.Random(CONFIG.seed)
    was_training = model.training
    model.eval()
    result = {"split": split}
    action_rows = read_manifest(split)
    vqa_rows = read_manifest(split, "_vqa")
    grounding_rows = read_manifest(split, "_grounding")
    if split == "val":
        vqa_rows = validation_vqa_rows(vqa_rows, rng)
        grounding_rows, action_rows = validation_episode_rows(
            grounding_rows, action_rows, rng
        )
    for kind, rows in (("action", action_rows), ("vqa", vqa_rows),
                       ("grounding", grounding_rows)):
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
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch distributed training with semantic_kd_training/scripts/run.sh train")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != CONFIG.student_gpu_count or torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"Expected {CONFIG.student_gpu_count} visible GPUs and processes; "
            f"got {torch.cuda.device_count()} GPUs and {world_size} processes"
        )
    if CONFIG.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by the GPU count")
    torch.cuda.set_device(rank)
    # Gloo supports CUDA collectives and avoids the container's incompatible NCCL runtime.
    backend = os.environ.get("KD_DDP_BACKEND", "gloo")
    dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    if rank == 0:
        print(f"DDP backend={backend} world_size={world_size}", flush=True)
    local_steps = CONFIG.gradient_accumulation_steps // world_size
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
    student, processor = load_student(training=True)
    model = DDP(student, device_ids=[rank], output_device=rank,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()
    tokenizer = ActionTokenizer(processor.tokenizer)
    stats = load_action_stats()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    if rank == 0:
        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
        (CONFIG.output_dir / "run_config.json").write_text(
            json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(CONFIG).items()},
                       indent=2), encoding="utf-8")
    action_pool, vqa_pool, grounding_pool = [], [], []
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    updates_per_epoch = math.ceil(len(action_rows) / CONFIG.gradient_accumulation_steps)
    total_updates = CONFIG.train_epochs * updates_per_epoch
    for update in range(1, total_updates + 1):
        action_total = 0.0
        aux_totals = {"vqa": 0.0, "grounding": 0.0}
        kd_totals = {"vqa": 0.0, "grounding": 0.0}
        aux_counts = {"vqa": 0, "grounding": 0}
        plan = None
        if rank == 0:
            plan = []
            for _ in range(CONFIG.gradient_accumulation_steps):
                action = choose(action_rows, action_pool)
                kind = "vqa" if random.random() < CONFIG.vqa_aux_probability else "grounding"
                aux = choose(vqa_rows, vqa_pool) if kind == "vqa" else choose(
                    grounding_rows, grounding_pool
                )
                plan.append((action, kind, aux))
        payload = [plan]
        dist.broadcast_object_list(payload, src=0)
        for local_index, (action, kind, aux) in enumerate(payload[0][rank::world_size]):
            with model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                    ce_action = action_loss(model, processor, tokenizer, stats, action)
                (ce_action / local_steps).backward()
            weight = CONFIG.vqa_weight if kind == "vqa" else CONFIG.grounding_weight
            sync = nullcontext() if local_index == local_steps - 1 else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                    ce_qa, kd, _ = qa_loss(model, processor, aux)
                    aux_total = weight * (ce_qa + CONFIG.kd_weight * kd)
                (aux_total / local_steps).backward()
            action_total += float(ce_action.item())
            aux_totals[kind] += float(ce_qa.item())
            kd_totals[kind] += float(kd.item())
            aux_counts[kind] += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.tensor([
            action_total, aux_totals["vqa"], aux_totals["grounding"],
            kd_totals["vqa"], kd_totals["grounding"],
            aux_counts["vqa"], aux_counts["grounding"],
        ], device=torch.device("cuda", rank), dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank != 0:
            continue
        (action_total, vqa_total, grounding_total, vqa_kd_total,
         grounding_kd_total, vqa_count, grounding_count) = totals.tolist()
        aux_totals = {"vqa": vqa_total, "grounding": grounding_total}
        kd_totals = {"vqa": vqa_kd_total, "grounding": grounding_kd_total}
        aux_counts = {"vqa": int(vqa_count), "grounding": int(grounding_count)}
        record = {"step": update,
                  "train_action_loss": action_total / CONFIG.gradient_accumulation_steps}
        for kind in ("vqa", "grounding"):
            record[f"train_{kind}_samples"] = int(aux_counts[kind])
            record[f"train_{kind}_loss"] = (aux_totals[kind] / aux_counts[kind]
                                               if aux_counts[kind] else None)
            record[f"train_{kind}_kd_loss"] = (kd_totals[kind] / aux_counts[kind]
                                                  if aux_counts[kind] else None)
        if (update % (CONFIG.validation_every_epochs * updates_per_epoch) == 0
                or update == total_updates):
            val = evaluate(model.module, processor, tokenizer, stats, "val")
            record.update({f"val_{k}": v for k, v in val.items() if k != "split"})
            if all(val[f"{kind}_loss"] is not None for kind in ("action", "vqa", "grounding")):
                score = selection_loss(val)
                record["val_selection_loss"] = score
                if score < best_val:
                    best_val = score
                    save_adapter(model.module, update, "best_adapter")
        if update % CONFIG.checkpoint_every == 0 or update == total_updates:
            save_adapter(model.module, update, "latest_adapter")
        with (CONFIG.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    if best_val == float("inf"):
        if rank == 0:
            save_adapter(model.module, total_updates, "best_adapter")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

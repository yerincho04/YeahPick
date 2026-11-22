"""Pair-balanced stage-1 training on two robot layouts with optional VQA."""

from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from bridge import binary_scores, choice_loss, choice_tokens, image_tensor, policy_output, predict
from config import CONFIG
from model_io import load_student
from prepare import read_jsonl


QUESTION_TYPES = ("Object recognition", "Attribute recognition", "Semantic reasoning",
                  "Visual grounding", "Counterfactual grounding")


def read_split(split: str, root: Path | None = None) -> list[dict]:
    return read_jsonl((root or CONFIG.work_dir) / f"{split}.jsonl")


def pixels_for_pair(processor, pair: dict, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for layout in ("original", "swapped"):
        with Image.open(pair["grounding"][layout]["image"]) as raw:
            result[layout] = image_tensor(processor, raw, device)
    return result


def examples_for_pair(pair: dict, epoch: int, pair_index: int) -> list[tuple[dict, str, float]]:
    examples = [(pair["grounding"][layout], layout, CONFIG.grounding_weight / 2)
                for layout in ("original", "swapped")]
    if CONFIG.train_vqa:
        focus = ("left", "right")[(pair_index + epoch - 1) % 2]
        for kind in QUESTION_TYPES:
            examples.append((pair["vqa"][focus][kind], "original", CONFIG.vqa_weight / 5))
    return examples


def train_pair(model, processor, pair: dict, epoch: int, pair_index: int) -> tuple[float, float]:
    device = next(model.parameters()).device
    pixels = pixels_for_pair(processor, pair, device)
    examples = examples_for_pair(pair, epoch, pair_index)
    ground_total = vqa_total = 0.0
    for index, (sample, layout, weight) in enumerate(examples):
        sync = nullcontext() if index == len(examples) - 1 else model.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
                scores = binary_scores(model, processor, pixels[layout],
                                       sample["question"], sample["choice_kind"])
                loss = choice_loss(scores, sample["answer"], sample["choice_kind"])
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Nonfinite loss: {sample.get('sample_id', pair['pair_id'])}")
            (weight * loss).backward()
        if index < 2:
            ground_total += float(loss.detach()) / 2
        else:
            vqa_total += float(loss.detach()) / 5
    return ground_total, vqa_total


@torch.inference_mode()
def score_pair(model, processor, pair: dict) -> tuple[list[dict], list[dict]]:
    device = next(model.parameters()).device
    pixels = pixels_for_pair(processor, pair, device)
    grounding, vqa = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        for layout in ("original", "swapped"):
            item = pair["grounding"][layout]
            choice = predict(binary_scores(model, processor, pixels[layout],
                                           item["question"], "side"), "side")
            grounding.append({"category": pair["category"], "pair_id": pair["pair_id"],
                              "layout": layout, "expected": item["answer"],
                              "predicted": choice, "correct": choice == item["answer"]})
        for focus in ("left", "right"):
            for kind in QUESTION_TYPES:
                item = pair["vqa"][focus][kind]
                choice = predict(binary_scores(model, processor, pixels["original"],
                                               item["question"], item["choice_kind"]),
                                 item["choice_kind"])
                vqa.append({"category": pair["category"], "pair_id": pair["pair_id"],
                            "sample_id": item["sample_id"], "focus_tile": focus,
                            "question_type": kind, "expected": item["answer"],
                            "predicted": choice, "correct": choice == item["answer"]})
    return grounding, vqa


def fraction(rows: list[dict]) -> dict:
    return {"correct": sum(row["correct"] for row in rows), "total": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None}


def summarize(grounding: list[dict], vqa: list[dict]) -> dict:
    by_pair = defaultdict(list)
    for row in grounding:
        by_pair[row["category"], row["pair_id"]].append(row)
    if any(len(rows) != 2 for rows in by_pair.values()):
        raise ValueError("Every pair must have original and swapped results")
    both = sum(all(row["correct"] for row in rows) for rows in by_pair.values())
    flipped = sum(rows[0]["predicted"] != rows[1]["predicted"]
                  for rows in by_pair.values())
    ground_by_category = {category: fraction([r for r in grounding if r["category"] == category])
                          for category in sorted({r["category"] for r in grounding})}
    vqa_by_category = {category: fraction([r for r in vqa if r["category"] == category])
                       for category in sorted({r["category"] for r in vqa})}
    vqa_by_type = {kind: fraction([r for r in vqa if r["question_type"] == kind])
                   for kind in QUESTION_TYPES}
    g_acc = fraction(grounding)["accuracy"]
    v_acc = fraction(vqa)["accuracy"]
    both_acc = both / len(by_pair)
    ready = (g_acc >= CONFIG.stage2_grounding_accuracy and
             both_acc >= CONFIG.stage2_both_layout_accuracy)
    if CONFIG.train_vqa:
        ready = (ready and v_acc >= CONFIG.stage2_vqa_accuracy and
                 vqa_by_type["Semantic reasoning"]["accuracy"] >= CONFIG.stage2_semantic_accuracy and
                 vqa_by_type["Visual grounding"]["accuracy"] >= CONFIG.stage2_visual_grounding_accuracy)
    both_by_category = {}
    for category in sorted({key[0] for key in by_pair}):
        rows = [items for key, items in by_pair.items() if key[0] == category]
        both_by_category[category] = {"correct": sum(all(r["correct"] for r in items)
                                                    for items in rows), "total": len(rows)}
        both_by_category[category]["accuracy"] = both_by_category[category]["correct"] / len(rows)
    return {"grounding": fraction(grounding),
            "grounding_by_category": ground_by_category,
            "grounding_by_layout": {layout: fraction([r for r in grounding if r["layout"] == layout])
                                    for layout in ("original", "swapped")},
            "both_layouts_correct": {"correct": both, "total": len(by_pair),
                                     "accuracy": both_acc},
            "both_layouts_by_category": both_by_category,
            "predictions_flip": {"count": flipped, "total": len(by_pair),
                                 "rate": flipped / len(by_pair)},
            "grounding_predicted_sides": dict(Counter(r["predicted"] for r in grounding)),
            "vqa": fraction(vqa), "vqa_by_category": vqa_by_category,
            "vqa_by_type": vqa_by_type,
            "vqa_by_type_category": {
                kind: {category: fraction([r for r in vqa if r["question_type"] == kind
                                            and r["category"] == category])
                       for category in sorted(vqa_by_category)}
                for kind in QUESTION_TYPES},
            "vqa_predicted_by_type": {
                kind: dict(Counter(r["predicted"] for r in vqa if r["question_type"] == kind))
                for kind in QUESTION_TYPES},
            "vqa_predicted_answers": dict(Counter(r["predicted"] for r in vqa)),
            "selection_score": (0.4 * g_acc + 0.4 * v_acc + 0.2 * both_acc
                                if CONFIG.train_vqa else (2 * g_acc + both_acc) / 3),
            "ready_for_stage2": ready}


def validate(model, processor, pairs: list[dict], rank: int, world_size: int):
    model.eval()
    grounding, vqa = [], []
    for pair in pairs[rank::world_size]:
        g, q = score_pair(model, processor, pair)
        grounding.extend(g)
        vqa.extend(q)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (grounding, vqa))
    model.train()
    if rank:
        return None
    all_grounding = [row for item in gathered for row in item[0]]
    all_vqa = [row for item in gathered for row in item[1]]
    return summarize(all_grounding, all_vqa), all_grounding, all_vqa


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def current_config() -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CONFIG).items()}


def check_resume_inputs(run_dir: Path) -> None:
    saved_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if saved_config != current_config():
        raise ValueError("Config changed since this run started")
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "pair_splits.json"):
        current = CONFIG.work_dir / name
        previous = run_dir / "input_snapshot" / name
        if hashlib.sha256(current.read_bytes()).digest() != hashlib.sha256(previous.read_bytes()).digest():
            raise ValueError(f"Prepared input changed since this run started: {name}")


def main() -> None:
    run_dir = Path(os.environ["STAGE1_RUN_DIR"]).resolve()
    resume = os.environ.get("STAGE1_RESUME") == "1"
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != CONFIG.student_gpu_count or torch.cuda.device_count() != world_size:
        raise RuntimeError(f"Expected {CONFIG.student_gpu_count} visible GPUs and processes")
    torch.cuda.set_device(rank)
    backend = os.environ.get("STAGE1_DDP_BACKEND", "nccl")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    elif (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (rank, world_size, backend):
        raise RuntimeError("Existing process group does not match this run")
    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)

    train = read_split("train")
    val = read_split("val")
    if len(train) != 800 or len(val) != 100 or len(train) % world_size:
        raise ValueError("Expected 800 train pairs and 100 validation pairs")
    sorted_keys = {row["category"] + "/" + row["pair_id"]: index
                   for index, row in enumerate(sorted(train, key=lambda r: (r["category"], r["pair_id"])))}
    latest = None
    if resume:
        check_resume_inputs(run_dir)
        latest = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
        if not Path(latest["adapter"]).is_dir():
            raise FileNotFoundError(latest["adapter"])
        if rank == 0:
            save_json(run_dir / "best.json", latest["best_record"])
    student, processor = load_student(
        training=True, adapter_path=Path(latest["adapter"]) if latest else None)
    for kind in ("side", "choice"):
        choice_tokens(processor.tokenizer, kind)
    if not any("projector." in name and param.requires_grad
               for name, param in student.named_parameters()):
        raise RuntimeError("Projector LoRA is not trainable")
    model = DDP(student, device_ids=[rank], output_device=rank,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    if resume:
        optimizer.load_state_dict(torch.load(Path(latest["optimizer"]),
                                             map_location=torch.device("cuda", rank),
                                             weights_only=False))

    if rank == 0:
        if not resume:
            if run_dir.exists():
                raise FileExistsError(run_dir)
            run_dir.mkdir(parents=True)
            snapshot = run_dir / "input_snapshot"
            snapshot.mkdir()
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "pair_splits.json"):
                shutil.copy2(CONFIG.work_dir / name, snapshot / name)
            save_json(run_dir / "run_config.json", current_config())
    dist.barrier()
    # Catch tokenizer, image-device, projector, and DDP errors before a long epoch.
    preflight = train[rank]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        pixels = pixels_for_pair(processor, preflight, torch.device("cuda", rank))
        for sample, layout, _ in examples_for_pair(preflight, 1, sorted_keys[preflight["category"] + "/" + preflight["pair_id"]]):
            loss = choice_loss(binary_scores(model.module, processor, pixels[layout], sample["question"],
                                             sample["choice_kind"]), sample["answer"], sample["choice_kind"])
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Preflight loss is not finite")
    validation_check = validate(model.module, processor, val[:world_size], rank, world_size)
    if rank == 0 and (validation_check[0]["grounding"]["total"] != 2 * world_size or
                      validation_check[0]["vqa"]["total"] != 10 * world_size):
        raise ValueError("Validation preflight did not score two layouts and ten VQA questions per pair")
    if rank == 0:
        model.module.eval()
        with Image.open(preflight["grounding"]["original"]["image"]) as raw:
            side, actions = policy_output(model.module, processor, raw.convert("RGB"),
                                          preflight["grounding"]["original"]["instruction"])
        if side not in ("Left", "Right") or len(actions) != 7:
            raise ValueError("Policy output format preflight failed")
        model.module.train()
        print("PREFLIGHT_OK", flush=True)
    dist.barrier()

    if resume:
        rng = torch.load(run_dir / f"rng_rank{rank}_epoch{latest['epoch']:02d}.pt",
                         map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device=rank)
        best = tuple(latest["best"])
        best_record = latest["best_record"] if rank == 0 else None
        stagnant = latest["stagnant"]
        start_epoch = latest["epoch"] + 1
        if start_epoch > CONFIG.max_epochs or (latest["epoch"] >= CONFIG.min_epochs and
                                                stagnant >= CONFIG.early_stop_patience):
            raise ValueError("This run already reached its stopping condition")
    else:
        best = (-1, float("-inf"))
        best_record = None
        stagnant = 0
        start_epoch = 1
    for epoch in range(start_epoch, CONFIG.max_epochs + 1):
        order = train.copy()
        random.Random(CONFIG.seed + epoch).shuffle(order)
        totals = torch.zeros(2, device=torch.device("cuda", rank), dtype=torch.float64)
        for update in range(len(order) // world_size):
            pair = order[update * world_size + rank]
            index = sorted_keys[pair["category"] + "/" + pair["pair_id"]]
            optimizer.zero_grad(set_to_none=True)
            g_loss, v_loss = train_pair(model, processor, pair, epoch, index)
            optimizer.step()
            totals += torch.tensor([g_loss, v_loss], dtype=torch.float64, device=totals.device)
            if rank == 0 and (update + 1) % 100 == 0:
                print(f"TRAIN epoch={epoch} update={update+1}/400", flush=True)
        dist.all_reduce(totals)
        validation = validate(model.module, processor, val, rank, world_size)
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(rank)},
                   run_dir / f"rng_rank{rank}_epoch{epoch:02d}.pt")
        if rank == 0:
            metrics, grounding_predictions, vqa_predictions = validation
            validation_dir = run_dir / "validation" / f"epoch_{epoch:02d}"
            # A previous process may have stopped after writing these files but
            # before advancing latest.json; redoing that epoch is safe.
            validation_dir.mkdir(parents=True, exist_ok=True)
            save_jsonl(validation_dir / "grounding_predictions.jsonl", grounding_predictions)
            save_jsonl(validation_dir / "vqa_predictions.jsonl", vqa_predictions)
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:02d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.module.save_pretrained(checkpoint)
            score = metrics["selection_score"]
            candidate = (int(metrics["ready_for_stage2"]), score)
            improved = candidate > best
            if improved:
                best = candidate
                stagnant = 0
                best_record = {"epoch": epoch, "adapter": str(checkpoint),
                               "selection_score": score,
                               "ready_for_stage2": metrics["ready_for_stage2"]}
            else:
                stagnant += 1
            record = {"epoch": epoch, "updates": epoch * len(order) // world_size,
                      "train_grounding_loss": totals[0].item() / len(order),
                      "train_vqa_loss": totals[1].item() / len(order) if CONFIG.train_vqa else None,
                      **metrics}
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            optimizer_path = run_dir / f"optimizer_epoch{epoch:02d}.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            previous = latest
            latest = {"epoch": epoch, "adapter": str(checkpoint),
                      "optimizer": str(optimizer_path), "best": list(best),
                      "best_record": best_record, "stagnant": stagnant}
            save_json(run_dir / "latest.json", latest)
            save_json(run_dir / "best.json", best_record)
            if previous is not None:
                Path(previous["optimizer"]).unlink(missing_ok=True)
                for old_rank in range(world_size):
                    (run_dir / f"rng_rank{old_rank}_epoch{previous['epoch']:02d}.pt").unlink(missing_ok=True)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            should_stop = epoch >= CONFIG.min_epochs and stagnant >= CONFIG.early_stop_patience
        else:
            should_stop = None
        payload = [should_stop]
        dist.broadcast_object_list(payload, src=0)
        if payload[0]:
            break
    if rank == 0:
        best_record = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
        print("STAGE1_DONE " + json.dumps(best_record), flush=True)
        if not best_record["ready_for_stage2"]:
            print("STAGE2_GATE_NOT_MET: inspect validation before action training", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

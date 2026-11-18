"""Score complete Left/Right answers, including multi-token answers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import CONFIG
from model_io import prompt_for_question


@torch.inference_mode()
def predict_side(model, processor, image, question: str) -> str:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    prompt_ids = tokenizer(
        prompt_for_question(question), add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    pixel = processor.image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(device, dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16)
    scores = {}
    for side in ("Left", "Right"):
        answer_ids = tokenizer(
            " " + side, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        if answer_ids.shape[1] == 0:
            raise ValueError(f"Empty token sequence for {side}")
        input_ids = torch.cat((prompt_ids, answer_ids), dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            output = model(input_ids=input_ids, pixel_values=pixel,
                           use_cache=False, return_dict=True)
        vision_tokens = output.logits.shape[1] - input_ids.shape[1]
        start = prompt_ids.shape[1] - 1 + vision_tokens
        answer_logits = output.logits[0, start:start + answer_ids.shape[1]].float()
        if answer_logits.shape[0] != answer_ids.shape[1]:
            raise ValueError("Answer logits and answer tokens have different lengths")
        log_probs = F.log_softmax(answer_logits, dim=-1)
        scores[side] = float(log_probs.gather(1, answer_ids[0, :, None]).sum())
    return "Left" if scores["Left"] > scores["Right"] else "Right"

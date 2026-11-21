"""One output grammar for every policy step: side, marker, seven actions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config import CONFIG


QUESTION_SUFFIX = "First give the target side, then the robot action."
MARKER = "\nAction: "


def bridge_question(instruction: str) -> str:
    return (f"What action should the robot take to {instruction.strip().lower()}? "
            f"{QUESTION_SUFFIX}")


def bridge_prompt(instruction: str) -> str:
    from model_io import prompt_for_question
    return prompt_for_question(bridge_question(instruction))


def side_ids(tokenizer, side: str) -> torch.Tensor:
    side = side.strip().title()
    if side not in ("Left", "Right"):
        raise ValueError(f"Invalid side: {side}")
    return tokenizer(" " + side, add_special_tokens=False,
                     return_tensors="pt")["input_ids"]


def marker_ids(tokenizer) -> torch.Tensor:
    return tokenizer(MARKER, add_special_tokens=False,
                     return_tensors="pt")["input_ids"]


def action_ids(action_tokenizer, action) -> torch.Tensor:
    tokenizer = action_tokenizer.tokenizer
    # ActionTokenizer's decode/encode string round trip is not guaranteed to
    # preserve the seven reserved token IDs. Use its exact bin mapping.
    discretized = np.digitize(np.clip(action, -1, 1), action_tokenizer.bins)
    ids = torch.as_tensor(tokenizer.vocab_size - discretized,
                          dtype=torch.long).reshape(1, -1)
    if ids.shape[1] != 7:
        raise ValueError(f"Expected seven action tokens, got {ids.shape[1]}")
    first = tokenizer.vocab_size - 256
    if not bool(((ids >= first) & (ids < tokenizer.vocab_size)).all()):
        raise ValueError("Action tokens fall outside the OpenVLA action vocabulary")
    return ids


def policy_inputs(sample: dict, processor, action_tokenizer, stats: dict) -> tuple[dict, dict]:
    from model_io import normalized_action

    tokenizer = processor.tokenizer
    prompt = tokenizer(bridge_prompt(sample["instruction"]),
                       add_special_tokens=True, return_tensors="pt")["input_ids"]
    side = side_ids(tokenizer, sample["answer_side"])
    marker = marker_ids(tokenizer)
    action = action_ids(action_tokenizer, normalized_action(sample["action"], stats))
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=prompt.dtype)
    ids = torch.cat((prompt, side, marker, action, eos), dim=1)
    with Image.open(sample["image"]) as raw:
        pixel = processor.image_processor(
            images=raw.convert("RGB"), return_tensors="pt")["pixel_values"]
    spans = {"side": (prompt.shape[1], prompt.shape[1] + side.shape[1]),
             "action": (prompt.shape[1] + side.shape[1] + marker.shape[1],
                        prompt.shape[1] + side.shape[1] + marker.shape[1] + 7)}
    return {"input_ids": ids, "pixel_values": pixel}, spans


def token_span_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                    start: int, end: int) -> torch.Tensor:
    visual_tokens = logits.shape[1] - input_ids.shape[1]
    if visual_tokens < 0 or start < 1 or end > input_ids.shape[1] or end <= start:
        raise ValueError("Invalid token span or model output length")
    predicted = logits[:, visual_tokens + start - 1:visual_tokens + end - 1].float()
    target = input_ids[:, start:end]
    if predicted.shape[1] != target.shape[1]:
        raise ValueError("Token span and logits do not align")
    return F.cross_entropy(predicted.reshape(-1, predicted.shape[-1]), target.reshape(-1))


def image_tensors(processor, image: Image.Image) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)


@torch.inference_mode()
def score_side(model, processor, image: Image.Image, instruction: str) -> str:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    prompt = tokenizer(bridge_prompt(instruction), add_special_tokens=True,
                       return_tensors="pt")["input_ids"].to(device)
    pixel = image_tensors(processor, image)
    scores = {}
    for name in ("Left", "Right"):
        candidate = side_ids(tokenizer, name).to(device)
        ids = torch.cat((prompt, candidate), dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
            logits = model(input_ids=ids, pixel_values=pixel,
                           use_cache=False, return_dict=True).logits
        visual_tokens = logits.shape[1] - ids.shape[1]
        start = visual_tokens + prompt.shape[1] - 1
        selected = logits[:, start:start + candidate.shape[1]].float()
        log_probs = F.log_softmax(selected, dim=-1)
        scores[name] = float(log_probs.gather(-1, candidate.unsqueeze(-1)).sum())
    return "Left" if scores["Left"] > scores["Right"] else "Right"


@torch.inference_mode()
def generate_action_tokens(model, processor, image: Image.Image,
                           instruction: str, side: str) -> list[int]:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    prompt = tokenizer(bridge_prompt(instruction), add_special_tokens=True,
                       return_tensors="pt")["input_ids"]
    ids = torch.cat((prompt, side_ids(tokenizer, side), marker_ids(tokenizer)), dim=1).to(device)
    pixel = image_tensors(processor, image)
    result = []
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        output = model(input_ids=ids, pixel_values=pixel, use_cache=True, return_dict=True)
        for index in range(7):
            first = tokenizer.vocab_size - 256
            token = output.logits[:, -1, first:tokenizer.vocab_size].argmax(dim=-1) + first
            result.append(int(token.item()))
            if index < 6:
                output = model(input_ids=token[:, None],
                               past_key_values=output.past_key_values,
                               use_cache=True, return_dict=True)
    return result

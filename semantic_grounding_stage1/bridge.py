"""Prompts and binary-choice scoring shared by training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from config import CONFIG


MARKER = "\nAction: "
OPTIONS = {"side": ("Left", "Right"), "choice": ("A", "B")}


def policy_question(instruction: str) -> str:
    task = instruction.strip()
    if not task:
        raise ValueError("Empty robot instruction")
    return (f"Task: {task}\nWhich tile is the target? "
            "Answer with Left or Right, then provide the robot action.")


def prompt_for_question(question: str) -> str:
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", question)
    return builder.get_prompt()


def semantic_side_question(question: str, answer: str) -> tuple[str, str]:
    """Convert the dataset's fixed A=left, B=right semantic choices."""
    suffix = ("Choices:\nA. The one shown on the left.\n"
              "B. The one shown on the right.\nAnswer with A or B.")
    if not question.endswith(suffix) or answer not in ("A", "B"):
        raise ValueError(f"Unexpected semantic question/answer: {question!r}, {answer!r}")
    return question[:-len(suffix)].rstrip() + " Answer with Left or Right.", ("Left" if answer == "A" else "Right")


def choice_tokens(tokenizer, kind: str) -> tuple[list[int], int, int]:
    """Find the shared prefix and the informative next token of both answers."""
    first, second = OPTIONS[kind]
    a = tokenizer(" " + first, add_special_tokens=False)["input_ids"]
    b = tokenizer(" " + second, add_special_tokens=False)["input_ids"]
    shared = 0
    while shared < min(len(a), len(b)) and a[shared] == b[shared]:
        shared += 1
    if len(a) != shared + 1 or len(b) != shared + 1 or a[shared] == b[shared]:
        raise ValueError(f"{kind} needs a one-token distinction: {a}, {b}")
    return a[:shared], a[shared], b[shared]


def image_tensor(processor, image: Image.Image, device: torch.device) -> torch.Tensor:
    dtype = torch.bfloat16 if CONFIG.bf16 else torch.float16
    return processor.image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device, dtype=dtype)


def binary_scores(model, processor, pixel: torch.Tensor, question: str, kind: str) -> torch.Tensor:
    tokenizer = processor.tokenizer
    prefix, first, second = choice_tokens(tokenizer, kind)
    ids = tokenizer(prompt_for_question(question), add_special_tokens=True,
                    return_tensors="pt")["input_ids"]
    if prefix:
        ids = torch.cat((ids, torch.tensor([prefix], dtype=ids.dtype)), dim=1)
    ids = ids.to(pixel.device)
    logits = model(input_ids=ids, pixel_values=pixel, use_cache=False,
                   return_dict=True).logits[:, -1, :].float()
    return logits[:, [first, second]]


def choice_loss(scores: torch.Tensor, answer: str, kind: str) -> torch.Tensor:
    label = OPTIONS[kind].index(answer)
    return F.cross_entropy(scores, torch.tensor([label], device=scores.device))


def predict(scores: torch.Tensor, kind: str) -> str:
    return OPTIONS[kind][int(scores.argmax(dim=-1).item())]


@torch.inference_mode()
def policy_output(model, processor, image: Image.Image, instruction: str) -> tuple[str, list[int]]:
    """Emit the unchanged side -> marker -> seven action-token format.

    Stage 1 does not train or evaluate the *values* of these action tokens.
    """
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    pixel = image_tensor(processor, image, device)
    with torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16):
        side = predict(binary_scores(model, processor, pixel,
                                     policy_question(instruction), "side"), "side")
        ids = tokenizer(prompt_for_question(policy_question(instruction)),
                        add_special_tokens=True, return_tensors="pt")["input_ids"]
        side_ids = tokenizer(" " + side, add_special_tokens=False,
                             return_tensors="pt")["input_ids"]
        marker_ids = tokenizer(MARKER, add_special_tokens=False,
                               return_tensors="pt")["input_ids"]
        ids = torch.cat((ids, side_ids, marker_ids), dim=1).to(device)
        output = model(input_ids=ids, pixel_values=pixel, use_cache=True, return_dict=True)
        actions = []
        for index in range(7):
            start = tokenizer.vocab_size - 256
            token = output.logits[:, -1, start:tokenizer.vocab_size].argmax(dim=-1) + start
            actions.append(int(token.item()))
            if index < 6:
                output = model(input_ids=token[:, None], past_key_values=output.past_key_values,
                               use_cache=True, return_dict=True)
    return side, actions

"""Turn sampled OpenVLA image attention into scores and a synchronized 2x2 video."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


def _center_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return image[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def _patch_mask(mask: np.ndarray, shape: tuple[int, int], strategy: str) -> np.ndarray:
    cropped = mask.astype(np.float32) if strategy == "resize-naive" else _center_square(mask.astype(np.float32))[0]
    return cv2.resize(cropped, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _overlay(image: np.ndarray, grid: np.ndarray, label: str, strategy: str) -> np.ndarray:
    base = np.ascontiguousarray(image.copy())
    if strategy == "resize-naive":
        cropped, (x0, y0, width, height) = base, (0, 0, base.shape[1], base.shape[0])
    else:
        cropped, (x0, y0, side) = _center_square(base)
        width = height = side
    heat = cv2.resize(grid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    # Per-frame normalization is for visibility only; CSV stores unnormalized mass.
    heat = np.uint8(np.clip(heat / max(float(heat.max()), 1e-12), 0, 1) * 255)
    color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = (heat.astype(np.float32) / 255.0 * 0.65)[..., None]
    base[y0:y0 + height, x0:x0 + width] = (cropped * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.rectangle(base, (0, 0), (340, 38), (0, 0, 0), -1)
    cv2.putText(base, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return base


def write_attention_outputs(
    images: list[np.ndarray], samples: list[dict], video_path: Path,
    scores_path: Path, question_id: int, seed: int, image_strategy: str,
) -> None:
    if not samples:
        raise ValueError("No attention samples were captured")
    if image_strategy not in ("resize-crop", "resize-naive"):
        raise ValueError(f"Unsupported attention-map image transform: {image_strategy}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in samples:
        for frac in LAYER_FRACS:
            layer = sample["maps"][frac]
            grid = layer["grid"]
            correct = _patch_mask(sample["masks"]["correct"], grid.shape, image_strategy)
            wrong = _patch_mask(sample["masks"]["wrong"], grid.shape, image_strategy)
            if correct.sum() == 0 or wrong.sum() == 0:
                raise RuntimeError(f"Board segmentation missing at question {question_id}, step {sample['step']}")
            correct_mass = float((grid * correct).sum())
            wrong_mass = float((grid * wrong).sum())
            correct_mean = correct_mass / float(correct.sum())
            wrong_mean = wrong_mass / float(wrong.sum())
            rows.append({
                "question_id": question_id, "seed": seed, "policy_step": sample["step"],
                "sim_seconds_from_first_policy_step": sample["step"] / 5.0,
                "layer_fraction": frac, "layer_number": layer["layer"],
                "correct_attention_mass": correct_mass,
                "wrong_attention_mass": wrong_mass,
                "correct_attention_per_patch": correct_mean,
                "wrong_attention_per_patch": wrong_mean,
                "correct_attention_fraction_of_two": correct_mean / max(correct_mean + wrong_mean, 1e-12),
            })
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (2 * w, 2 * h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create attention video: {video_path}")
    try:
        selected = 0
        for step, image in enumerate(images):
            while selected + 1 < len(samples) and samples[selected + 1]["step"] <= step:
                selected += 1
            sample = samples[selected]
            panels = [
                _overlay(image, sample["maps"][frac]["grid"],
                         f"Layer {int(frac * 100)}% | sample {sample['step'] / 5:.1f}s", image_strategy)
                for frac in LAYER_FRACS
            ]
            frame = np.concatenate([
                np.concatenate(panels[:2], axis=1),
                np.concatenate(panels[2:], axis=1),
            ], axis=0)
            # Two identical output frames per 5 Hz policy step preserve simulated time at 10 fps.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            writer.write(bgr)
    finally:
        writer.release()

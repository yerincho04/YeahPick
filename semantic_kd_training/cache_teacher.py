"""Precompute frozen Prismatic features from the same robot frames used by OpenVLA."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from config import CONFIG
from model_io import prompt_for_question, read_manifest, teacher_feature_path


def transformed(image, teacher):
    values = teacher.vision_backbone.image_transform(image)
    if isinstance(values, dict):
        return {key: value.unsqueeze(0).cuda() for key, value in values.items()}
    return values.unsqueeze(0).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher feature extraction requires a CUDA GPU")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    episodes = defaultdict(list)
    for split in splits:
        for sample in read_manifest(split):
            episodes[sample["episode_id"]].append(sample)

    cache_root = CONFIG.work_dir / "teacher_features"
    cache_root.mkdir(parents=True, exist_ok=True)
    signature = {"teacher_model": CONFIG.teacher_model,
                 "llama2_metadata_repo": CONFIG.teacher_llama2_metadata_repo,
                 "layers": list(CONFIG.kd_layers),
                 "prompt": "PurePromptBuilder(openvla), Knowledge instruction -> tile question"}
    signature_path = cache_root / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            if any(cache_root.rglob("layer_*.npy")):
                raise ValueError("Teacher cache config changed; use a fresh work_dir")
            signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    else:
        signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    pending = {
        episode_id: sorted(rows, key=lambda row: row["step"])
        for episode_id, rows in episodes.items()
        if any(not teacher_feature_path(rows[0], layer).is_file() for layer in CONFIG.kd_layers)
    }
    if not pending:
        print("TEACHER_CACHE_READY (all requested episodes already cached)", flush=True)
        return

    os.environ["PRISMATIC_LLAMA2_7B_METADATA_REPO"] = CONFIG.teacher_llama2_metadata_repo
    from prismatic.models.load import load

    teacher = load(CONFIG.teacher_model)
    teacher = teacher.to("cuda", dtype=torch.bfloat16).eval()
    teacher.requires_grad_(False)
    tokenizer = teacher.llm_backbone.tokenizer

    with torch.inference_mode():
        for episode_id, samples in pending.items():
            features = {layer: [] for layer in CONFIG.kd_layers}
            for sample in samples:
                prompt = prompt_for_question(sample["semantic_question"])
                ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].cuda()
                with Image.open(sample["image"]) as image:
                    pixel = transformed(image.convert("RGB"), teacher)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = teacher(
                        input_ids=ids, pixel_values=pixel,
                        output_hidden_states=True, return_dict=True,
                    )
                for layer in CONFIG.kd_layers:
                    if layer >= len(output.hidden_states):
                        raise ValueError(f"Teacher has no hidden_states[{layer}]")
                    features[layer].append(output.hidden_states[layer][0, -1].float().cpu().numpy())
            for layer, vectors in features.items():
                path = teacher_feature_path(samples[0], layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(vectors, dtype=np.float16))
            print(f"TEACHER_CACHED {episode_id} frames={len(samples)}", flush=True)


if __name__ == "__main__":
    main()

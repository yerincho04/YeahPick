"""Cache teacher answers and representations once per QA question."""
from __future__ import annotations

import argparse
import json
import os
import re
import numpy as np
import torch
from PIL import Image

from config import CONFIG
from model_io import prompt_for_question, read_manifest, teacher_feature_path, teacher_result_path


def normalized(value: str) -> str:
    return " ".join(w for w in re.sub(r"[^\w\s]", " ", value.casefold()).split()
                    if w not in {"a", "an", "the"})


def answer_is_correct(sample: dict, generated: str) -> bool:
    gold, got = normalized(sample["answer"]), normalized(generated)
    if not gold or not got:
        return False
    if sample["kind"] == "grounding" or gold in {"left", "right"}:
        return set(re.findall(r"\b(?:left|right)\b", got)) == {gold}
    return got == gold or got.startswith(gold + " ") or got.endswith(" " + gold)


def transformed(image, teacher):
    values = teacher.vision_backbone.image_transform(image)
    if isinstance(values, dict):
        return {k: v.unsqueeze(0).cuda() for k, v in values.items()}
    return values.unsqueeze(0).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher extraction requires a CUDA GPU")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    samples = [row for split in splits for suffix in ("_vqa", "_grounding")
               for row in read_manifest(split, suffix)]
    root = CONFIG.work_dir / "teacher_vqa_features"
    root.mkdir(parents=True, exist_ok=True)
    signature = {"version": 2, "teacher": CONFIG.teacher_model,
                 "metadata": CONFIG.teacher_llama2_metadata_repo,
                 "layers": list(CONFIG.kd_layers), "tokens": CONFIG.teacher_max_answer_tokens}
    signature_path = root / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Teacher cache settings changed; use a fresh work_dir")
    else:
        signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    pending = [s for s in samples if not teacher_result_path(s).is_file() or
               any(not teacher_feature_path(s, l).is_file() for l in CONFIG.kd_layers)]
    if not pending:
        print("TEACHER_CACHE_READY", flush=True)
        return
    os.environ["PRISMATIC_LLAMA2_7B_METADATA_REPO"] = CONFIG.teacher_llama2_metadata_repo
    from prismatic.models.load import load
    teacher = load(CONFIG.teacher_model).to("cuda", dtype=torch.bfloat16).eval()
    teacher.requires_grad_(False)
    tokenizer = teacher.llm_backbone.tokenizer
    with torch.inference_mode():
        for index, sample in enumerate(pending, 1):
            prompt = prompt_for_question(sample["question"])
            ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].cuda()
            with Image.open(sample["image"]) as image:
                rgb = image.convert("RGB")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = teacher(input_ids=ids, pixel_values=transformed(rgb, teacher),
                                     output_hidden_states=True, return_dict=True)
                vectors = {}
                for layer in CONFIG.kd_layers:
                    if layer >= len(output.hidden_states):
                        raise ValueError(f"Teacher has no hidden_states[{layer}]")
                    vectors[layer] = output.hidden_states[layer][0, -1].float().cpu().numpy()
                del output
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated_ids = teacher.generate(image=rgb, prompt_text=prompt,
                                                     max_new_tokens=CONFIG.teacher_max_answer_tokens,
                                                     do_sample=False)
            if isinstance(generated_ids, str):
                generated = generated_ids
            else:
                generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if generated.startswith(prompt):
                    generated = generated[len(prompt):]
            result = {"sample_id": sample["sample_id"], "question": sample["question"],
                      "image": sample["image"], "answer": sample["answer"],
                      "teacher_answer": generated.strip(),
                      "teacher_correct": answer_is_correct(sample, generated)}
            for layer, vector in vectors.items():
                path = teacher_feature_path(sample, layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vector.astype(np.float16))
            teacher_result_path(sample).write_text(json.dumps(result, ensure_ascii=False),
                                                   encoding="utf-8")
            if index % 100 == 0 or index == len(pending):
                print(f"TEACHER_CACHED {index}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()

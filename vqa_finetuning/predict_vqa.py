"""Optional later VQA inference; not run as a test in this experiment."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from train import PREFIX

p = argparse.ArgumentParser()
p.add_argument('--weights', required=True, help='Merged model or LoRA adapter directory')
p.add_argument('--image', required=True, help='Square left/right composite, same layout as prepare_data.py')
p.add_argument('--question', required=True)
a = p.parse_args()
base = json.loads((Path(__file__).parent/'base_model.json').read_text())
adapter = (Path(a.weights)/'adapter_config.json').exists()
model_path = base['model_id'] if adapter else a.weights
kwargs = dict(revision=base['revision']) if adapter else {}
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, **kwargs)
model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa', **kwargs).cuda()
if adapter:
    model = PeftModel.from_pretrained(model, a.weights).merge_and_unload()
model.eval()
with Image.open(a.image) as im:
    inputs = processor('In: '+PREFIX+a.question.strip()+'\nOut:', im.convert('RGB'), return_tensors='pt')
inputs = {k: v.cuda().to(torch.bfloat16) if k == 'pixel_values' else v.cuda() for k,v in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=64)
print(processor.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))

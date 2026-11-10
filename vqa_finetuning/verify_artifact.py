"""Artifact integrity / API smoke check using a synthetic image, not a VQA test."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from safetensors.torch import load_file
from transformers import AutoModelForVision2Seq, AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--merged', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base = json.loads((Path(__file__).parent/'base_model.json').read_text())
config = json.loads((a.adapter/'adapter_config.json').read_text())
assert config['base_model_name_or_path'] == base['model_id']
assert config['revision'] == base['revision']
state = load_file(a.adapter/'adapter_model.safetensors')
assert len(state) == 128
assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
assert all(torch.count_nonzero(v) > 0 for k,v in state.items() if 'lora_B' in k)
processor = AutoProcessor.from_pretrained(a.merged, local_files_only=True, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(a.merged, local_files_only=True, trust_remote_code=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
assert all(torch.isfinite(v).all().item() for v in model.parameters())
inputs = processor('In: This is an interface check.\nOut:', Image.new('RGB',(224,224),'white'), return_tensors='pt')
inputs = {k: v.cuda().to(torch.bfloat16) if k == 'pixel_values' else v.cuda() for k,v in inputs.items()}
with torch.inference_mode():
    result = model.generate(**inputs, do_sample=False, min_new_tokens=2, max_new_tokens=2)
assert result.shape[1] == inputs['input_ids'].shape[1]+2
report = dict(adapter_tensors=128, adapter_finite=True, all_lora_B_updated=True,
    adapter_dtype='float32', merged_parameters_finite=True, standalone_offline_load=True,
    synthetic_cached_generation_tokens=2, performance_evaluation=False,
    note='Synthetic white-image API check only; no VQA correctness or robot evaluation.')
a.output.write_text(json.dumps(report, indent=2)+'\n')
print('ARTIFACT_CHECKS_PASSED', flush=True)

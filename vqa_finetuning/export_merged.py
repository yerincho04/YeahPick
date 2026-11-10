"""Export a standalone BF16 model from the final adapter, without quantization."""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--adapter', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
base = json.loads((Path(__file__).parent/'base_model.json').read_text())
model = AutoModelForVision2Seq.from_pretrained(base['model_id'], revision=base['revision'],
    trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.output, safe_serialization=True, max_shard_size='5GB')
AutoProcessor.from_pretrained(base['model_id'], revision=base['revision'], trust_remote_code=True).save_pretrained(a.output)
Path(a.output, 'base_model.json').write_text(json.dumps(base, indent=2)+'\n')
print('MERGE_COMPLETED', flush=True)

"""Export a standalone BF16 model from the final adapter, without quantization."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from huggingface_hub import hf_hub_download
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
# Transformers 4.47 does not copy this processor's custom Python module on save.
# Bundle pinned custom code and unchanged preprocessing/tokenizer assets so the
# artifact is self-contained and avoids tokenizer-serializer version drift.
for filename in ['configuration_prismatic.py', 'modeling_prismatic.py', 'processing_prismatic.py',
        'processor_config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer.model',
        'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'generation_config.json']:
    src = hf_hub_download(base['model_id'], filename, revision=base['revision'])
    shutil.copy2(src, Path(a.output)/filename)
original_config = json.loads(Path(hf_hub_download(base['model_id'], 'config.json', revision=base['revision'])).read_text())
saved_config = json.loads(Path(a.output, 'config.json').read_text())
assert saved_config['norm_stats'] == original_config['norm_stats']
assert saved_config['n_action_bins'] == original_config['n_action_bins']
Path(a.output, 'base_model.json').write_text(json.dumps(base, indent=2)+'\n')
print('MERGE_COMPLETED', flush=True)

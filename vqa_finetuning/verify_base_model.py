"""Check the downloaded base weight shards against the Hub's published hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=Path('base_verification.json'))
a = p.parse_args()
root = Path(__file__).parent
base = json.loads((root/'base_model.json').read_text())
shards = json.loads((root/'base_model_shards.json').read_text())
results = []
for shard in shards:
    path = Path(hf_hub_download(base['model_id'], shard['rfilename'], revision=base['revision']))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    assert path.stat().st_size == shard['size']
    assert digest.hexdigest() == shard['lfs']['sha256'], shard['rfilename']
    results.append(dict(file=shard['rfilename'], size=path.stat().st_size, sha256=digest.hexdigest(), verified=True))
a.output.write_text(json.dumps(dict(**base, shards=results), indent=2)+'\n')
print('BASE_SHARDS_VERIFIED', flush=True)

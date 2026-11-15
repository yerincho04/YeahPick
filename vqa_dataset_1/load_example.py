"""Read the dataset from any working directory. Requires Pillow."""
import json
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
with (root / 'vqa_all.jsonl').open(encoding='utf-8') as f:
    group = json.loads(next(f))

print(group['group_id'])
for side in ('left', 'right'):
    path = root / group['images'][side]['path']
    with Image.open(path) as image:
        print(side, path.relative_to(root), image.size)
for qa in group['vqa']:
    print(qa['type'], qa['question'], '=>', qa['expected_answer'])

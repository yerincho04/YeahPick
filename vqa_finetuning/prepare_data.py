"""Build all 8,000 question-answer examples from pinned source branches."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

SOURCES = [
    ('Yerin', 'bd2fc842b547663577347fc9c69163b1b5468136', 'VQA_dataset_v2', ['attribute', 'emotion', 'state']),
    ('hyerin', '354746b060d36cd528e76a34aa373808daa16412', 'VQA_dataset', ['counting', 'symmetry']),
    ('jaeho', 'bfaf666512c1bc89f61da2c1adeac00e6834492f', 'vqa_dataset', ['celebrity', 'living_world', 'traffic']),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'pairs').mkdir(exist_ok=True)
    records, provenance, seen, pair_images = [], [], set(), {}
    for branch, revision, folder, categories in SOURCES:
        root = args.sources / branch / folder
        for category in categories:
            rel = f'{category}/{category}_vqa.jsonl' if branch == 'jaeho' else f'{category}_vqa.jsonl'
            src = root / rel
            raw = src.read_bytes()
            groups = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            assert len(groups) == 200, (category, len(groups))
            for group in groups:
                key = (branch, group['group_id'])
                assert key not in seen, key
                seen.add(key)
                # OpenVLA accepts one image: preserve BOTH tiles, no center crop.
                # Equal 112px-wide slots form a square 224px image. Letterboxing
                # preserves each tile's aspect ratio; there is no augmentation.
                pair_path = f'pairs/{category}_{group["pair_id"]}.png'
                signature = tuple(group['images'][side]['path'] for side in ['left', 'right'])
                if pair_path in pair_images:
                    assert pair_images[pair_path] == signature, 'Conflicting images for '+pair_path
                pair_images[pair_path] = signature
                target = args.output / pair_path
                if not target.exists():
                    canvas = Image.new('RGB', (224, 224), (255, 255, 255))
                    for side, x in [('left', 0), ('right', 112)]:
                        with Image.open(root / group['images'][side]['path']) as im:
                            tile = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (112, 224), Image.Resampling.LANCZOS)
                        canvas.paste(tile, (x + (112-tile.width)//2, (224-tile.height)//2))
                    canvas.save(target)
                assert len(group['vqa']) == 5
                for i, qa in enumerate(group['vqa']):
                    assert isinstance(qa['question'], str) and qa['question'].strip()
                    assert isinstance(qa['expected_answer'], str) and qa['expected_answer'].strip()
                    records.append(dict(id=f'{branch}/{group["group_id"]}/{i}', category=category,
                        question_type=qa['type'], image=pair_path, question=qa['question'],
                        answer=qa['expected_answer'], source_branch=branch, source_revision=revision,
                        source_file=folder+'/'+rel, group_id=group['group_id'], pair_id=group['pair_id']))
            provenance.append(dict(branch=branch, revision=revision, file=folder+'/'+rel,
                sha256=hashlib.sha256(raw).hexdigest(), groups=len(groups)))
    assert len(records) == 8000, len(records)
    data = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records)
    (args.output/'train.jsonl').write_text(data)
    image_manifest = {str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted((args.output/'pairs').glob('*.png'))}
    (args.output/'image_manifest.json').write_text(json.dumps(image_manifest, indent=2)+'\n')
    report = dict(examples=len(records), groups=len(seen), pairs=len(list((args.output/'pairs').glob('*.png'))),
        by_category=dict(collections.Counter(r['category'] for r in records)),
        by_question_type=dict(collections.Counter(r['question_type'] for r in records)),
        train_sha256=hashlib.sha256(data.encode()).hexdigest(), sources=provenance,
        image_manifest_sha256=hashlib.sha256((args.output/'image_manifest.json').read_bytes()).hexdigest(),
        split='All examples used for training; no validation or test split.',
        excluded='emotion_kdef: missing images; excluded by explicit user instruction. Aggregate JSONL, source questions and pair metadata are not additional QA examples.',
        image_layout='224x224 RGB; left/right 112x224 letterboxed slots, white background; no augmentation')
    (args.output/'data_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

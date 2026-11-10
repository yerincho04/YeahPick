"""Download only required source JSONL/images at immutable GitHub revisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from prepare_data import SOURCES

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=Path('sources'))
a = p.parse_args()


def fetch(item):
    revision, source_path, target = item
    target.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://raw.githubusercontent.com/yerincho04/YeahPick/'+revision+'/'+quote(source_path)
    with urlopen(url, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    return data


jobs = {}
for branch, revision, folder, categories in SOURCES:
    root = a.output/branch/folder
    fetch((revision, folder+'/README.md', root/'README.md'))
    for category in categories:
        rel = f'{category}/{category}_vqa.jsonl' if branch == 'jaeho' else f'{category}_vqa.jsonl'
        data = fetch((revision, folder+'/'+rel, root/rel))
        for line in data.decode().splitlines():
            row = json.loads(line)
            for entry in row['images'].values():
                path = entry['path']
                assert '..' not in Path(path).parts and not Path(path).is_absolute()
                jobs[(branch, path)] = (revision, folder+'/'+path, root/path)
with ThreadPoolExecutor(max_workers=8) as pool:
    for _ in pool.map(fetch, jobs.values()):
        pass
print('Downloaded source snapshots and', len(jobs), 'distinct images.')

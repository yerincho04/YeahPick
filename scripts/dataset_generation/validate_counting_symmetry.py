"""Check counts, paired conditions, semantic targets, PNG symmetry, and GLB contents."""
from pathlib import Path
from collections import Counter
import csv, io, json, struct
import numpy as np
from PIL import Image, ImageOps, ImageChops
import trimesh
ROOT=Path(__file__).resolve().parents[2]/'ManiSkill/mani_skill/assets/carrot'
def read(p): return json.loads(p.read_text())
def validate(category):
    root=ROOT/category; assets=read(root/'assets.json'); pairs=read(root/'pairs.json'); questions=read(root/'questions.json'); db=read(root/'model_db.json')
    assert len(assets)==len(db)==len(list(root.rglob('*.glb')))==30
    assert len(pairs)==200 and len(questions)==800
    assert len({p['question'] for p in pairs})==200
    assert len({q['instruction_id'] for q in questions})==800
    assert Counter(p['answer'] for p in pairs)=={'Left':100,'Right':100}
    assert set(db)==set(assets)=={p[s] for p in pairs for s in ['left','right']}
    with (root/'questions.csv').open() as f: assert len(list(csv.DictReader(f)))==800
    for condition in ['knowledge','explicit_object','explicit_object_detailed','explicit_spatial']:
        assert read(root/f'{condition}.json')==[q for q in questions if q['condition']==condition]
    hashes=set()
    for aid,a in assets.items():
        im=Image.open(root/a['image']).convert('RGB')
        hashes.add(im.tobytes())
        if category=='symmetry': assert (ImageChops.difference(im,ImageOps.mirror(im)).getbbox() is None)==a['symmetric'],aid
        path=root/a['glb']; data=path.read_bytes(); magic,version,length=struct.unpack_from('<4sII',data)
        assert (magic,version,length)==(b'glTF',2,len(data))
        n,kind=struct.unpack_from('<II',data,12); assert kind==0x4e4f534a
        gltf=json.loads(data[20:20+n]); assert not any('uri' in b for b in gltf['buffers'])
        assert len(gltf['images'])==1 and 'bufferView' in gltf['images'][0]
        view=gltf['bufferViews'][gltf['images'][0]['bufferView']]; start=28+n+view.get('byteOffset',0)
        embedded=Image.open(io.BytesIO(data[start:start+view['byteLength']])).convert('RGB')
        assert ImageChops.difference(im,embedded).getbbox() is None
        scene=trimesh.load(path,force='scene'); assert len(scene.geometry)==2
        assert np.allclose(scene.bounds,[db[aid]['bbox']['min'],db[aid]['bbox']['max']])
        collision=trimesh.load(path.parent/'collision.obj',force='mesh')
        assert collision.is_watertight and np.allclose(collision.extents,[.11,.11,.006])
    assert len(hashes)==30
    for i,p in enumerate(pairs):
        target=assets[p['semantic_answer_id']]; other=assets[p['distractor_id']]
        assert p['question_id']==i and p[p['answer'].lower()]==p['semantic_answer_id']
        assert p['left']!=p['right'] and target['object']==other['object']
        if category=='counting': assert target['count']==p['target_predicate']['count']!=other['count']
        else: assert target['symmetric']==p['target_predicate']['symmetric']!=other['symmetric']
        rows=[q for q in questions if q['question_id']==i]; assert len(rows)==4
        for q in rows:
            assert q['instruction']==p['instructions'][q['condition']]
            for field in ['left','right','answer','semantic_answer_id','distractor_id']: assert q[field]==p[field]
        assert p['instructions']['explicit_spatial']==f"Place the cube on the {p['answer'].lower()} tile."
        assert all(side not in p['instructions'][c].lower() for c in ['knowledge','explicit_object'] for side in ['left tile','right tile'])
    return {'assets':30,'base_questions':200,'instructions':800,'left':100,'right':100,'glb_embedded_textures_verified':30,'unique_images':30,'runtime_simulation':'not run (requires ManiSkill/SAPIEN environment)'}
if __name__=='__main__':
    report={c:validate(c) for c in ['counting','symmetry']}
    write=ROOT/'counting_symmetry_validation.json'; write.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

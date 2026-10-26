"""Deterministic illustrated object datasets; Python 3, Pillow, numpy, trimesh."""
from pathlib import Path
import json, csv
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps
ROOT = Path(__file__).resolve().parents[2] / 'ManiSkill/mani_skill/assets/carrot'
PLURAL = {'tomato':'tomatoes','apple':'apples','carrot':'carrots','fish':'fish','flower':'flowers'}
KINDS = ['tomato','apple','carrot','fish','flower']
DETAIL = {
'tomato':'round red tomatoes with green star-shaped stems', 'apple':'red apples with brown stems and green leaves',
'carrot':'tapered orange carrots with green leafy tops', 'fish':'blue fish with triangular tails and dark eyes',
'flower':'flowers with pink petals, yellow centers, and green stems',
'butterfly':'an orange butterfly with broad upper wings, smaller lower wings, and a dark central body',
'leaf':'a green pointed leaf with a central vein and branching side veins',
'vase':'a blue vase with a narrow neck and a wide rounded body',
'robot':'a gray robot with a square head, round eyes, and jointed arms',
'owl':'a brown owl with two large pale eyes and a pointed orange beak',
'beetle':'a red beetle with dark spots, six legs, and a central wing seam',
'tree':'a green evergreen tree with stacked branches and a brown trunk',
'umbrella':'a purple umbrella with a scalloped canopy and a straight centered handle',
'house':'a yellow house with a red triangular roof and blue windows',
'crab':'an orange crab with raised claws, stalked eyes, and side legs',
'rocket':'a silver rocket with a pointed red nose, blue window, and red fins',
'mask':'a purple face mask with almond-shaped eyes and a small central mouth',
'bow':'a pink ribbon bow with two broad loops and two hanging tails',
'crown':'a gold crown with three pointed tips and colored jewels',
'snowflake':'a blue snowflake with six branching arms'}
SINGULAR_DETAIL = {'tomato':'a round red tomato with a green star-shaped stem', 'apple':'a red apple with a brown stem and green leaf', 'carrot':'a tapered orange carrot with a green leafy top', 'fish':'a blue fish with a triangular tail and dark eye', 'flower':'a flower with pink petals, a yellow center, and a green stem'}
SYM = list(DETAIL)[5:]
ASYM_DETAILS = ['with a single dark blue circular tag on its right half', 'with its entire right half shifted downward', 'with its right half colored red instead of matching the left', 'with its right half narrower than the left', 'with a horizontal blank gap cutting across only its right half']

def icon(kind):
    im=Image.new('RGBA',(400,400)); d=ImageDraw.Draw(im)
    def e(b,c): d.ellipse(b,fill=c)
    def p(points,c): d.polygon(points,fill=c)
    def l(points,c,w=10): d.line(points,fill=c,width=w)
    if kind in ['tomato','apple']:
        e((75,100,325,335),'#e94d48'); e((95,115,170,175),'#ff9280')
        l([(200,120),(210,62)],'#794b32',14)
        if kind=='tomato': p([(200,85),(215,119),(263,109),(231,138),(247,163),(202,143),(163,163),(172,132),(139,113),(185,116)],'#38954a')
        else: e((214,60,290,100),'#459b43')
    elif kind=='carrot':
        p([(125,125),(278,125),(195,355)],'#f58a23')
        for x in [145,200,255]: l([(200,135),(x,45)],'#409951',18)
        l([(155,177),(213,185)],'#cb621e'); l([(184,239),(231,230)],'#cb621e')
    elif kind=='fish':
        e((65,110,290,290),'#399bd5'); p([(270,200),(360,117),(360,283)],'#2872b8'); e((107,163,128,184),'#183650'); p([(160,117),(216,68),(225,135)],'#296fb5')
    elif kind=='flower':
        l([(200,190),(200,360)],'#36924b',15); e((200,270,290,315),'#36924b')
        import math
        for i in range(8):
            x=200+75*math.cos(i*math.pi/4); y=160+75*math.sin(i*math.pi/4); e((x-44,y-44,x+44,y+44),'#e96eab')
        e((155,115,245,205),'#ffc94c')
    elif kind=='butterfly':
        e((35,65,197,220),'#f89b38'); e((70,200,197,333),'#ed673e'); e((75,100,130,155),'#ffe19d'); e((120,245,165,285),'#ffe19d'); e((188,95,212,330),'#363b51'); l([(200,110),(160,50)],'#363b51',7)
    elif kind=='leaf':
        p([(200,32),(97,120),(58,235),(121,326),(200,355)],'#4aa86b'); l([(200,38),(200,370)],'#226844',8)
        for y in [130,200,270]: l([(200,y+35),(100,y)],'#a4d58e',6)
    elif kind=='vase':
        p([(160,50),(200,50),(200,355),(100,350),(65,290),(88,218),(154,155)],'#5a9ec5'); l([(96,263),(200,263)],'#f4db95',18)
    elif kind=='robot':
        d.rectangle((105,70,200,185),fill='#7e94a5'); d.rectangle((115,200,200,315),fill='#7e94a5'); e((133,105,170,142),'#68daed'); l([(120,225),(64,285)],'#596d83',25); l([(145,310),(145,360)],'#596d83',25)
    elif kind=='owl':
        e((80,95,200,350),'#a47a56'); p([(90,148),(80,40),(172,105)],'#a47a56'); e((105,118,195,228),'#f9e8c0'); e((145,151,175,188),'#30394a'); p([(177,224),(200,260),(200,224)],'#edab3f')
    elif kind=='beetle':
        for y in [140,220,290]: l([(120,y),(55,y-28)],'#343948',12)
        e((85,90,200,340),'#df534e'); e((150,45,200,130),'#343948'); l([(198,95),(198,338)],'#343948',6)
        for x,y in [(132,160),(145,250)]: e((x-17,y-17,x+17,y+17),'#343948')
    elif kind=='tree':
        d.rectangle((170,270,200,365),fill='#92663b')
        for y,w in [(40,75),(110,110),(185,145)]: p([(200,y),(200-w,y+125),(200,y+125)],'#37956a')
    elif kind=='umbrella':
        p([(200,65),(120,85),(50,155),(40,210),(83,193),(125,215),(165,195),(200,215)],'#9161be'); l([(200,100),(200,360)],'#54466b',11)
    elif kind=='house':
        d.rectangle((70,180,200,350),fill='#e8bf62'); p([(40,180),(200,45),(200,180)],'#cc6253'); d.rectangle((100,215,158,280),fill='#6ab6d0'); d.rectangle((175,270,200,350),fill='#815c49')
    elif kind=='crab':
        for y in [215,255,290]: l([(135,y),(50,y+40)],'#e6793f',13)
        e((100,175,200,310),'#ec8245'); l([(130,205),(70,130)],'#ec8245',16); e((30,65,105,150),'#ec8245'); p([(63,60),(73,110),(90,63)],(0,0,0,0)); l([(165,195),(160,135)],'#ec8245',12); e((143,115,178,150),'#303b4e')
    elif kind=='rocket':
        p([(200,30),(140,120),(140,310),(200,310)],'#c5d0dc'); p([(200,30),(140,120),(200,120)],'#df6558'); p([(140,240),(80,330),(140,310)],'#df6558'); e((165,150,235,220),'#4fa8d1'); p([(170,315),(185,375),(200,315)],'#f8b746')
    elif kind=='mask':
        p([(200,60),(65,95),(78,235),(143,320),(200,350)],'#9871bc'); e((95,125,180,190),'#fff3dc'); l([(173,275),(200,280)],'#633e83',8)
    elif kind=='bow':
        p([(200,175),(60,65),(45,260),(200,225)],'#e879a6'); p([(170,210),(95,345),(170,320),(200,220)],'#d95991'); e((177,165,223,235),'#bd447c')
    elif kind=='crown':
        p([(65,100),(110,190),(155,120),(200,45),(200,330),(75,330)],'#e8b940'); e((108,239,145,277),'#d96464'); l([(80,302),(200,302)],'#fff0a1',12)
    elif kind=='snowflake':
        import math
        for a in range(6):
            t=a*math.pi/3; end=(200+150*math.cos(t),200+150*math.sin(t)); l([(200,200),end],'#64acce',10)
            for delta in [-.55,.55]:
                base=(200+95*math.cos(t),200+95*math.sin(t)); tip=(base[0]+50*math.cos(t+delta),base[1]+50*math.sin(t+delta)); l([base,tip],'#64acce',7)
    if kind in SYM:
        half=im.crop((0,0,200,400)); im.paste(half,(0,0)); im.paste(ImageOps.mirror(half),(200,0))
    return im


def artwork(kind,count=None,symmetric=True):
    im=Image.new('RGB',(768,768),'#fffaf0')
    if count is not None:
        xy={1:[(234,234)],2:[(65,234),(403,234)],3:[(234,60),(65,403),(403,403)],4:[(65,65),(403,65),(65,403),(403,403)],5:[(65,25),(403,25),(234,234),(65,443),(403,443)],6:[(65,25),(403,25),(65,234),(403,234),(65,443),(403,443)]}[count]
        obj=icon(kind).resize((300,300),Image.Resampling.LANCZOS)
        for pos in xy: im.paste(obj,pos,obj)
    else:
        obj=icon(kind).resize((640,640),Image.Resampling.LANCZOS); im.paste(obj,(64,64),obj)
        if not symmetric:
            mode=SYM.index(kind)%5
            if mode==0:
                ImageDraw.Draw(im).ellipse((495,335,567,407),fill='#1645b8',outline='#ffffff',width=5)
            else:
                right=im.crop((384,0,768,768)); im.paste('#fffaf0',(384,0,768,768))
                if mode==1: im.paste(right,(384,40))
                elif mode==2:
                    a=np.array(right); mask=np.any(a!=[255,250,240],axis=2); a[mask]=[203,55,77]; im.paste(Image.fromarray(a),(384,0))
                elif mode==3: im.paste(right.resize((270,768),Image.Resampling.LANCZOS),(384,0))
                else:
                    im.paste(right,(384,0)); ImageDraw.Draw(im).rectangle((384,340,767,395),fill='#fffaf0')
    return im

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def build(category):
    root=ROOT/category; root.mkdir(parents=True,exist_ok=True); (root/'images').mkdir(exist_ok=True)
    db={}; assets={}; thumbs=[]
    specs=[(k,n,None) for k in KINDS for n in range(1,7)] if category=='counting' else [(k,None,s) for k in SYM for s in [True,False]]
    for i,(kind,n,sym) in enumerate(specs):
        aid=f'{category}_{i+1:02d}'; folder=root/'shapes'/aid; folder.mkdir(parents=True,exist_ok=True)
        im=artwork(kind,n,sym); im.save(root/'images'/f'{aid}.png')
        half=.055; z=.0031
        box=trimesh.creation.box(extents=(.11,.11,.006)); box.visual.vertex_colors=[245,245,245,255]
        top=trimesh.Trimesh(vertices=[[-half,-half,z],[half,-half,z],[half,half,z],[-half,half,z]],faces=[[0,1,2],[0,2,3]],process=False)
        top.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],image=im)
        scene=trimesh.Scene(); scene.add_geometry(box,geom_name='tile',node_name='tile'); scene.add_geometry(top,geom_name='artwork',node_name='artwork'); (folder/'textured.glb').write_bytes(scene.export(file_type='glb'))
        (folder/'collision.obj').write_text(box.export(file_type='obj').rstrip()+'\n')
        db[aid]={'name':aid,'sign':aid,'bbox':{'min':[-.055,-.055,-.003],'max':[.055,.055,.0031]},'scales':[1],'density':6368}
        desc=(SINGULAR_DETAIL[kind] if n==1 else f'{n} {DETAIL[kind]}') if n else f'{DETAIL[kind]}' + (' with matching mirrored halves and no blue tag' if sym else ' '+ASYM_DETAILS[SYM.index(kind)%5])
        assets[aid]={'object':kind,'count':n,'symmetric':sym,'axis':'vertical' if n is None else None,'description':desc,'image':f'images/{aid}.png','glb':f'shapes/{aid}/textured.glb'}
        thumb=im.resize((192,192)); ImageDraw.Draw(thumb).text((5,5),aid,fill='black'); thumbs.append(thumb)
    sheet=Image.new('RGB',(192*6,192*5),'white')
    for i,thumb in enumerate(thumbs): sheet.paste(thumb,((i%6)*192,(i//6)*192))
    sheet.save(root/'contact_sheet.png'); write_json(root/'model_db.json',db); write_json(root/'assets.json',assets)
    pairs=[]; flat=[]
    for i in range(200):
        if category=='counting':
            kind=KINDS[i%5]; n=(i//5)%6+1; family=i//30; aid=f'counting_{KINDS.index(kind)*6+n:02d}'
            other_n=(n+family)%6+1
            if other_n==n: other_n=n%6+1
            bid=f'counting_{KINDS.index(kind)*6+other_n:02d}'
            contexts=[f'a basket that held {n+2} items before two were removed',f'{n+1} guests when one guest has already been served',f'the number remaining after giving away three of {n+3} snacks',f'two groups of {n} objects shared equally by two people',f'{n+4} seats after four seats are taken',f'a recipe requiring {n+1} portions when one portion is already prepared',f'a shelf with {n+5} spaces, five of which are occupied']
            clue=contexts[family]; question=f'Place the cube on the tile where the number of {PLURAL[kind]} matches {clue}.'
            semantic=f'tile showing {n} {kind if n==1 else PLURAL[kind]}'; reason=f'The situation gives {n}; count the {PLURAL[kind]} on each tile and select the one containing {n}.'
            relation='count_from_context'; predicate={'count':n}; family_name=['remove_two','serve_one','give_three','equal_sharing','occupied_seats','recipe_remainder','shelf_capacity'][family]
        else:
            kind=SYM[i%15]; family=i//15; sym=family%2==0; base=SYM.index(kind)*2+1; aid=f'symmetry_{base+(not sym):02d}'; bid=f'symmetry_{base+sym:02d}'
            clues=[('would match itself when reflected across a vertical line through its center','would fail to match itself when reflected across a vertical line through its center'),('could be folded along its vertical center so all colored details overlap','could not be folded along its vertical center so all colored details overlap'),('would look unchanged in a left-right mirror reversal','would look changed in a left-right mirror reversal'),('has a right half that can be predicted by reflecting the left half','has a right half that cannot be predicted by reflecting the left half'),('could be completed exactly by copying and reflecting only its left half','could not be completed exactly by copying and reflecting only its left half'),('would pass a left-right reflection test including every colored mark','would fail a left-right reflection test including every colored mark'),('has matching silhouettes and colored details on opposite sides of its vertical center','has at least one colored detail without a mirror partner across its vertical center')]
            clue=clues[family//2][0 if sym else 1]; question=f'Place the cube on the tile showing the {kind} design that {clue}.'
            semantic=f'tile showing the {"symmetric" if sym else "asymmetric"} {kind}'; predicate={'symmetric':sym,'axis':'vertical'}
            reason='Compare the complete colored design across the vertical center. '+('Every colored feature has a reflected partner.' if sym else 'The design is '+ASYM_DETAILS[SYM.index(kind)%5]+', which breaks reflection correspondence.')
            relation='vertical_reflection'; family_name=['mirror_test','paper_fold','mirror_reversal','predict_half','reconstruct_half','reflection_test','feature_correspondence'][family//2]
        side='Left' if i%2==0 else 'Right'; left,right=(aid,bid) if side=='Left' else (bid,aid)
        detailed=f'Place the cube on the tile showing {assets[aid]["description"]}.'
        variants={'knowledge':question,'explicit_object':f'Place the cube on the {semantic}.','explicit_object_detailed':detailed,'explicit_spatial':f'Place the cube on the {side.lower()} tile.'}
        row={'index':i,'question_id':i,'left':left,'right':right,'question':question,'knowledge_instruction':question,'answer':side,'semantic_answer':semantic,'semantic_answer_id':aid,'distractor_id':bid,'left_label':assets[left]['description'],'right_label':assets[right]['description'],'visual_description':assets[aid]['description'],'knowledge_category':category,'relation_type':relation,'template_family':family_name,'reasoning':reason,'target_predicate':predicate,'instructions':variants,'dataset_provenance':'illustrated_objects_v1','split_group':kind}
        pairs.append(row)
        for condition,instruction in variants.items():
            flat.append({'instruction_id':f'{category}_{i:03d}_{condition}','question_id':i,'condition':condition,'instruction':instruction,'left':left,'right':right,'answer':side,'semantic_answer_id':aid,'distractor_id':bid,'reasoning_chain':['Knowledge','Object','Location','Action'] if condition=='knowledge' else ['Location','Action'] if condition=='explicit_spatial' else ['Object','Location','Action']})
    write_json(root/'pairs.json',pairs); write_json(root/'questions.json',flat)
    for condition in ['knowledge','explicit_object','explicit_object_detailed','explicit_spatial']: write_json(root/f'{condition}.json',[r for r in flat if r['condition']==condition])
    with (root/'questions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(flat)
    (root/'README.md').write_text(f'''# {category.title()} illustrated object dataset\n\n30 textured tabletop GLBs (0.11 × 0.11 × 0.0061 m), 200 two-tile base tasks, and 800 instructions. English instructions match the existing evaluation environment. Images are original programmatically drawn object illustrations, not photographs or free-standing object meshes. GLBs embed their PNG textures.\n\n- `pairs.json`: existing evaluation format, one base task per row.\n- `questions.json` / `questions.csv`: four conditions saved separately per task; individual condition JSON files contain 200 records each.\n- `assets.json`: artwork metadata and paths; `model_db.json`: simulator metadata.\n- `shapes/<id>/textured.glb` and `collision.obj`: visible tile and collision box.\n- `images/` and `contact_sheet.png`: source artwork and preview.\n\nKnowledge tasks require interpreting quantities in everyday situations or testing vertical reflection of the entire colored design. Object conditions explicitly identify the target; detailed conditions add visible appearance. Symmetry negatives vary across one-sided marks, vertical displacement, color mismatch, compressed width, and missing bands. These controlled edits can provide shortcuts; this is an illustrated reasoning benchmark, not a natural-image benchmark. Each category reuses its 30 assets; 200 tasks are not 200 independent visual scenes. Some semantic templates recur across objects and quantities. Left/right answers are balanced.\n\nThe four conditions share a question_id and scene: evaluate them as paired observations, not independent samples. No train/test split is imposed. Split by object and template family to reduce leakage; reuse of images across splits can still leak visual identity. Canonical stored spatial labels assume no scene swap. The environment dynamically recalculates spatial instructions when swapping. Use instruction_condition=explicit_object_detailed (or its visual_description alias) for the detailed condition; knowledge, explicit_object, and explicit_spatial already work directly.\n\nRegenerate from repository root: `python scripts/dataset_generation/generate_counting_symmetry.py` (Pillow, numpy, trimesh). Validate: `python scripts/dataset_generation/validate_counting_symmetry.py`.\n''')
    return root

if __name__=='__main__':
    for category in ['counting','symmetry']: print(build(category))

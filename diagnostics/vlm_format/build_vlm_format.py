"""Export the photo-set questions (Emotion, Attribute, State) in the team's VLM question format.

Per set, 10 questions (one per photo pair where possible, answers and sides balanced):
  text_only_<set>_first.json    description of each photo + question; correct option listed first
  text_only_<set>_second.json   same questions; correct option listed second
  image_text_<set>.json         side-by-side image + question, noswap and swap (answer = left / right)
  vlm_image_pairs/<set>_v1/qNN_{noswap,swap}.png
  VLM_QUESTIONS.md              everything in one document

Option A is the left tile of the robot experiment's original (noswap) layout, option B the right tile.
Photo descriptions were written by looking at each photo; they state what is visible and avoid the
answer label (e.g. "surprised", "clean") and the question's own key words.

Run from the repo root:  python diagnostics/vlm_format/build_vlm_format.py
"""
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CARROT = REPO / "ManiSkill" / "mani_skill" / "assets" / "carrot"
TILES = REPO / "diagnostics" / "dataset_build" / "photo_output" / "tiles"

# set name -> (asset, question ids in output order)
SETS = {
    "emotion": ("photo_emotion_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "attribute": ("photo_attribute_v1", [0, 6, 7, 3, 4, 15, 11, 12, 18, 19]),
    "state": ("photo_state_v1", [0, 6, 7, 3, 15, 11, 12, 20, 28, 27]),
}
# tile-name prefix -> (noun, intro sentence for text only, intro sentence for image + text)
NOUNS = {
    "surprised_sleepy": ("Person", "Each description is of a photo of a person.", "Each panel shows a photo of a person."),
    "bench": ("Bench", "Each description is of a photo of a bench.", "Each panel shows a photo of a bench."),
    "bottle": ("Bottle", "Each description is of a photo with a bottle.", "Each panel shows a photo with a bottle."),
    "door": ("Door", "Each description is of a photo of a door.", "Each panel shows a photo of a door."),
    "toilet": ("Toilet", "Each description is of a photo of a toilet.", "Each panel shows a photo of a toilet."),
    "plate": ("Plate", "Each description is of a photo of a plate.", "Each panel shows a photo of a plate."),
}
DESC = {
    # Emotion
    "surprised_sleepy_surprised_0": "a young man faces the camera with his eyes opened very wide, his eyebrows raised high and his mouth hanging open.",
    "surprised_sleepy_sleepy_0": "a woman lies on a bed with her head on a pillow, her eyes shut and a phone held loosely in her hand.",
    "surprised_sleepy_surprised_1": "a young man in a red bow tie stares at the camera with his eyes opened wide, his eyebrows lifted and his lips parted.",
    "surprised_sleepy_sleepy_1": "someone fully dressed lies sprawled on their back across a bed in a dim room, motionless, with an arm over their face.",
    "surprised_sleepy_surprised_2": "a woman on a street looks down at the phone in her hands with her eyebrows raised and her mouth open in a round shape.",
    "surprised_sleepy_sleepy_2": "a man leans far back in an armchair with his head tipped against the cushion, his eyes shut and his arms folded across his chest.",
    "surprised_sleepy_surprised_3": "a young man sits at a table in front of a cake with lit candles, with his eyebrows raised high and his mouth forming a small round O.",
    "surprised_sleepy_sleepy_3": "someone lies in bed with a blanket pulled over most of their head; only their hair shows, and they are not moving.",
    "surprised_sleepy_surprised_4": "an older woman in glasses looks down at the phone in her hands with her eyebrows raised high and her jaw dropped.",
    "surprised_sleepy_sleepy_4": "a man sits slumped on a bathroom floor with his head resting on his folded arms on a toilet lid, his eyelids lowered.",
    # Attribute: benches
    "bench_wooden_0": "a low, backless bench made of thick weathered wooden planks stands against a brick wall; a small dog sits on it.",
    "bench_metal_0": "a dark blue bench made of perforated steel sheet on tubular steel legs stands beside a street; a person sits on it.",
    "bench_wooden_1": "a park bench built from orange-brown timber slats with armrests stands on a path beside a lawn.",
    "bench_metal_1": "a black bench made of curved steel bars with armrests stands on a paved plaza next to a flower bed.",
    "bench_wooden_2": "a long bench built from reddish-brown timber slats with a slatted back stands on a garden path.",
    "bench_metal_2": "a teal bench cut from steel plate, with swirling scroll patterns in its back, stands on a brick sidewalk.",
    "bench_wooden_3": "a grey, weathered seat built from timber boards stands in front of a shop window; a woman sits on it.",
    "bench_metal_3": "a black lounger made of steel wire mesh on a tubular steel frame stands on the ground in a park.",
    "bench_wooden_4": "an old bench built from rough, unpainted timber boards stands under a porch roof; two cats lie on it.",
    "bench_metal_4": "a dark green bench made of a steel wire grid on steel legs stands on a lawn.",
    # Attribute: bottles
    "bottle_glass_0": "a tall dark-green glass bottle with a printed label stands on a table next to a stemmed glass of dark beer.",
    "bottle_plastic_0": "a white plastic sports bottle with an orange push-pull cap and printed lettering stands on a kitchen counter next to an apple.",
    "bottle_glass_1": "two brown glass soda bottles with red-and-white labels and metal caps stand on a table among crystal glassware.",
    "bottle_plastic_1": "a green plastic drink bottle with a white screw cap stands on a desk next to a soda can.",
    "bottle_glass_2": "a squat clear glass bottle holding amber liquor, closed with a cork stopper, stands on a desk.",
    "bottle_plastic_2": "a small plastic soda bottle with a pink label stands on a dark table next to a stack of colourful bowls.",
    "bottle_glass_3": "two cobalt-blue glass bottles and a green glass vase stand in a row, lit from behind.",
    "bottle_plastic_3": "a large clear plastic cola bottle with a red screw cap and a red label stands on a kitchen counter.",
    "bottle_glass_4": "three small contour-shaped glass cola bottles with red labels stand on a table; two of them hold dark cola.",
    "bottle_plastic_4": "a small white plastic bottle with a blue printed label is held in a person's hand.",
    # State: doors
    "door_open_0": "a very large hangar door is slid fully aside, leaving a wide gap filled with daylight; two people stand in the gap next to an aircraft propeller.",
    "door_closed_0": "both leaves of a blue panelled double door sit flush inside the door frame, filling the doorway completely.",
    "door_open_1": "a dark door leaf with small glass panes is swung fully outward, away from its frame; a woman in a hat stands in the uncovered doorway.",
    "door_closed_1": "a red, peeling double door with a barred window sits flush inside its frame, filling the doorway; a blue cart is parked in front.",
    "door_open_2": "the folding front door of a city bus is folded back against the side of the doorway; a man is stepping aboard.",
    "door_closed_2": "two flat beige sliding panels sit edge to edge across a doorway in an interior wall, covering it completely.",
    "door_open_3": "a white front door is swung inward against the hallway wall; beyond the doorway a lawn and the houses across the street can be seen, and suitcases stand in the hall.",
    "door_closed_3": "a grey door with a curtained window sits flush inside its frame in a brick wall, filling the doorway.",
    # State: toilets
    "toilet_clean_0": "a white toilet with its lid down stands in a white-tiled bathroom; the porcelain is glossy and uniformly white, and a fresh roll of paper sits beside it.",
    "toilet_dirty_0": "a toilet with its seat raised stands on a checkered floor; the inside of the bowl is coated with a dark brown crust, and tools and a bucket are piled around it.",
    "toilet_clean_1": "a cream-coloured toilet with an electronic control panel has its lid raised; the bowl and seat are smooth, glossy and uniformly coloured.",
    "toilet_dirty_1": "a toilet bowl stands outdoors against a concrete wall; its rim and outside are covered with black grime and the ground around it is littered.",
    "toilet_clean_2": "a white toilet with a green seat stands in a tiled bathroom beside a bidet and a shelf of folded towels; the porcelain is glossy and the floor is bare and dry.",
    "toilet_dirty_2": "a toilet lies tipped on its side outdoors among dead leaves; its surface is dull and dusty and a large spider sits on the lid.",
    # State: plates
    "plate_empty_0": "a white plate holds only a used fork, a few crumbs and smears of sauce; drinking glasses stand behind it.",
    "plate_full_0": "a white plate is piled with French toast, bacon strips and a fried egg.",
    "plate_empty_2": "a round white plate with a bare, spotless surface sits on a table between a fork and a knife.",
    "plate_full_2": "a plate is covered edge to edge with roast potatoes, Brussels sprouts, carrots, sausages, turkey and stuffing.",
    "plate_empty_3": "a square white plate with nothing on its surface sits on a red placemat.",
    "plate_full_3": "a white plate holds a thick pork chop topped with apple slices and sage leaves, with sauce pooled around it; a knife and fork lie beside it.",
}


def tile_image(asset: str, tile: str) -> Image.Image:
    return Image.open(next((TILES / asset).glob(f"*__{tile}.png"))).convert("RGB")


def side_by_side(left: Image.Image, right: Image.Image, size: int = 448, gap: int = 32) -> Image.Image:
    canvas = Image.new("RGB", (2 * size + 3 * gap, size + 2 * gap), "white")
    for k, im in enumerate((left, right)):
        im = im.copy()
        im.thumbnail((size, size), Image.LANCZOS)
        canvas.paste(im, (gap + k * (size + gap) + (size - im.width) // 2, gap + (size - im.height) // 2))
    return canvas


def main() -> None:
    md = ["# Photo-set questions in VLM format (Emotion, Attribute, State)", ""]
    text_md, image_md = ["## Text Only", ""], ["## Image + Text", ""]
    for name, (asset, qids) in SETS.items():
        pairs = {int(p["index"]): p for p in json.loads((CARROT / asset / "pairs.json").read_text())}
        img_dir = HERE / "vlm_image_pairs" / f"{name}_v1"
        img_dir.mkdir(parents=True, exist_ok=True)
        first, second, image_text = [], [], []
        for n, qid in enumerate(qids):
            p = pairs[qid]
            noun, intro_text, intro_image = next(v for k, v in NOUNS.items() if p["left"].startswith(k))
            a, b = f"{noun} A", f"{noun} B"
            correct, wrong = (a, b) if p["answer"].lower() == "left" else (b, a)
            body = f"{intro_text} {a}: {DESC[p['left']]} {b}: {DESC[p['right']]} {p['question']}"
            for out, (c1, c2) in ((first, (correct, wrong)), (second, (wrong, correct))):
                out.append({"Q": f"{body}\nChoices: {c1}; {c2}.\nAnswer with exactly one of: \"{c1}\" or \"{c2}\".",
                            "A": correct})
            left, right = tile_image(asset, p["left"]), tile_image(asset, p["right"])
            side = p["answer"].lower()
            for layout, (l, r), ans in (("noswap", (left, right), side),
                                        ("swap", (right, left), "right" if side == "left" else "left")):
                rel = f"vlm_image_pairs/{name}_v1/q{n:02d}_{layout}.png"
                side_by_side(l, r).save(HERE / rel)
                image_text.append({"image": rel,
                                   "Q": f"{intro_image} {p['question']}\r\nAnswer with exactly one of: \"left\" or \"right\".",
                                   "A": ans})
        dump = lambda rows: json.dumps(rows, indent=2, ensure_ascii=False)
        (HERE / f"text_only_{name}_first.json").write_text(dump(first) + "\n")
        (HERE / f"text_only_{name}_second.json").write_text(dump(second) + "\n")
        (HERE / f"image_text_{name}.json").write_text(dump(image_text) + "\n")
        title = name.capitalize()
        text_md += [f"### {title}", "", "정답이 첫 번째 선택지인 경우", "", "```json", dump(first), "```", "",
                    "정답이 두 번째 선택지인 경우", "", "```json", dump(second), "```", ""]
        image_md += [f"### {title}", "", "질문과 답(swap, no swap)", "", "```json", dump(image_text), "```", ""]
        n_a = sum(r["A"].endswith(" A") for r in first)
        print(f"{name}: {len(first)} questions (answer A: {n_a}, answer B: {len(first) - n_a}), {len(image_text)} image entries")
    (HERE / "VLM_QUESTIONS.md").write_text("\n".join(md + text_md + image_md))


if __name__ == "__main__":
    main()

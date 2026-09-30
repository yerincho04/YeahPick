"""Original, hand-authored pictogram icons for the Phase 1 custom semantic diagnostic set.

Every image here is drawn from scratch with PIL primitives (no external image sourced) --
image_source is recorded as "synthetic (PIL, hand-authored pictogram)" for all of them in
the dataset manifest. This sidesteps any copyright/licensing question entirely, at the cost
of visual fidelity (schematic pictograms, not photographs) -- an explicit, documented
tradeoff for a Phase 1 pilot, not a substitute for eventually validating on real imagery.

Each function returns a 512x512 RGB PIL.Image with a plain white background and a centered
pictogram. Call `ICONS[name](512)` to render.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

try:
    _FONT_BIG = ImageFont.load_default(size=72)
    _FONT_MED = ImageFont.load_default(size=48)
except TypeError:
    # Older Pillow without the `size` kwarg on load_default().
    _FONT_BIG = _FONT_MED = ImageFont.load_default()

SIZE = 512
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)


def _canvas(size=SIZE):
    img = Image.new("RGB", (size, size), WHITE)
    return img, ImageDraw.Draw(img)


def _poly_blob(cx, cy, r, n=14, jitter=0.18, seed=0):
    """Deterministic irregular blob polygon (for rocks, clouds, organic shapes)."""
    pts = []
    state = seed * 9301 + 49297
    for i in range(n):
        state = (state * 9301 + 49297) % 233280
        rr = r * (1 - jitter / 2 + jitter * (state / 233280))
        theta = 2 * math.pi * i / n
        pts.append((cx + rr * math.cos(theta), cy + rr * math.sin(theta)))
    return pts


def _star(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        theta = rot + math.pi * i / n
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pts


def _quadruped(color, stripes=None, spots=None, big_ears=False, long_neck=False,
               hump=False, tail="thin", trunk=False):
    img, d = _canvas()
    cx, cy = 256, 300
    neck_h = 90 if long_neck else 20
    body_box = (cx - 130, cy - 60, cx + 110, cy + 60)
    d.ellipse(body_box, fill=color, outline=BLACK, width=3)
    if hump:
        d.ellipse((cx - 30, cy - 110, cx + 40, cy - 40), fill=color, outline=BLACK, width=3)
    # legs
    for lx in (cx - 90, cx - 30, cx + 40, cx + 90):
        d.rectangle((lx - 12, cy + 40, lx + 12, cy + 140), fill=color, outline=BLACK, width=2)
    # neck + head
    head_cx, head_cy = cx + 120, cy - 60 - neck_h
    if long_neck:
        d.polygon([(cx + 90, cy - 40), (cx + 140, cy - 40), (head_cx + 10, head_cy + 20),
                   (head_cx - 20, head_cy + 20)], fill=color, outline=BLACK, width=3)
    d.ellipse((head_cx - 45, head_cy - 35, head_cx + 45, head_cy + 35), fill=color, outline=BLACK, width=3)
    if trunk:
        d.polygon([(head_cx + 30, head_cy + 10), (head_cx + 20, head_cy + 90),
                    (head_cx + 45, head_cy + 90), (head_cx + 45, head_cy + 15)],
                   fill=color, outline=BLACK, width=2)
    ear_r = 28 if big_ears else 16
    d.ellipse((head_cx - 40, head_cy - 55, head_cx - 40 + ear_r * 2, head_cy - 55 + ear_r * 2),
               fill=color, outline=BLACK, width=2)
    d.ellipse((head_cx + 10, head_cy - 55, head_cx + 10 + ear_r * 2, head_cy - 55 + ear_r * 2),
               fill=color, outline=BLACK, width=2)
    d.ellipse((head_cx + 10, head_cy - 5, head_cx + 18, head_cy + 3), fill=BLACK)  # eye
    # tail
    if tail == "poof":
        d.ellipse((cx - 165, cy - 20, cx - 125, cy + 20), fill=color, outline=BLACK, width=2)
    d.line((cx - 130, cy - 10, cx - 165, cy + 10), fill=BLACK, width=6)
    # stripes / spots
    if stripes:
        for i in range(5):
            x = cx - 100 + i * 40
            d.line((x, cy - 55, x - 10, cy + 55), fill=stripes, width=8)
    if spots:
        for sx, sy in [(-60, -20), (-20, 10), (20, -25), (60, 15), (-90, 30)]:
            d.ellipse((cx + sx - 12, cy + sy - 12, cx + sx + 12, cy + sy + 12), fill=spots)
    return img


def _bird(body_color, beak_color=(255, 140, 0), long_neck=False, big_wing=True):
    img, d = _canvas()
    cx, cy = 256, 280
    d.ellipse((cx - 90, cy - 70, cx + 90, cy + 110), fill=body_color, outline=BLACK, width=3)
    head_cy = cy - 120 if long_neck else cy - 90
    if long_neck:
        d.polygon([(cx - 15, cy - 60), (cx + 15, cy - 60), (cx + 25, head_cy), (cx - 25, head_cy)],
                   fill=body_color, outline=BLACK, width=3)
    d.ellipse((cx - 45, head_cy - 40, cx + 45, head_cy + 40), fill=body_color, outline=BLACK, width=3)
    d.polygon([(cx + 40, head_cy - 5), (cx + 90, head_cy + 8), (cx + 40, head_cy + 20)],
               fill=beak_color, outline=BLACK, width=2)
    d.ellipse((cx + 5, head_cy - 15, cx + 15, head_cy - 5), fill=BLACK)
    if big_wing:
        d.ellipse((cx - 70, cy - 30, cx + 10, cy + 90), outline=BLACK, width=3)
    d.polygon([(cx - 30, cy + 100), (cx - 45, cy + 150), (cx - 15, cy + 105)], fill=beak_color, outline=BLACK)
    d.polygon([(cx + 10, cy + 100), (cx + 25, cy + 150), (cx + 30, cy + 105)], fill=beak_color, outline=BLACK)
    return img


def _sign(shape, fill, border, symbol=None, border_w=14):
    img, d = _canvas()
    cx, cy, r = 256, 256, 200
    if shape == "octagon":
        pts = [(cx + r * math.cos(math.pi / 8 + i * math.pi / 4),
                 cy + r * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
        d.polygon(pts, fill=fill, outline=border, width=border_w)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx - r * 0.87, cy + r * 0.6), (cx + r * 0.87, cy + r * 0.6)]
        d.polygon(pts, fill=fill, outline=border, width=border_w)
    elif shape == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=border, width=border_w)
    elif shape == "square":
        d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=border, width=border_w)
    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        d.polygon(pts, fill=fill, outline=border, width=border_w)
    if symbol:
        symbol(d, cx, cy, r)
    return img


def _text_symbol(text, color=(255, 255, 255), big=True):
    font = _FONT_BIG if big else _FONT_MED

    def draw(d, cx, cy, r):
        d.text((cx, cy), text, fill=color, anchor="mm", font=font)
    return draw


def _slash(d, cx, cy, r, color=(200, 20, 20), width=22):
    d.line((cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7), fill=color, width=width)


def _p_slash(d, cx, cy, r):
    _text_symbol("P", color=WHITE)(d, cx, cy, r)
    _slash(d, cx, cy, r)


def _cigarette_slash(d, cx, cy, r):
    # A horizontal cigarette (white body, orange filter, glowing tip) so "no smoking" is
    # unambiguous, then the standard red slash on top -- distinguishes it from other
    # red-slashed-circle signs (e.g. no_parking) which slash a "P" instead.
    d.rectangle((cx - 90, cy - 14, cx + 60, cy + 14), fill=(245, 245, 245), outline=BLACK, width=3)
    d.rectangle((cx + 20, cy - 14, cx + 60, cy + 14), fill=(200, 140, 60), outline=BLACK, width=3)
    d.ellipse((cx + 60, cy - 16, cx + 78, cy + 16), fill=(255, 140, 40))
    for dx, dy in [(70, -35), (85, -20)]:
        d.arc((cx + dx - 10, cy + dy - 10, cx + dx + 10, cy + dy + 10), start=200, end=340, fill=(150, 150, 150), width=3)
    _slash(d, cx, cy, r, color=(200, 20, 20))


def _bar(d, cx, cy, r, color=WHITE, width=30):
    d.rectangle((cx - r * 0.75, cy - width / 2, cx + r * 0.75, cy + width / 2), fill=color)


def _icon_house(with_cross=False, wall_color=(230, 200, 160)):
    img, d = _canvas()
    d.rectangle((156, 260, 356, 420), fill=wall_color, outline=BLACK, width=4)
    d.polygon([(120, 260), (256, 130), (392, 260)], fill=(160, 60, 50), outline=BLACK, width=4)
    d.rectangle((230, 340, 282, 420), fill=(120, 80, 40), outline=BLACK, width=3)
    if with_cross:
        d.rectangle((246, 60, 266, 130), fill=(160, 60, 50), outline=BLACK, width=3)
        d.rectangle((220, 85, 292, 105), fill=(160, 60, 50), outline=BLACK, width=3)
    return img


def _icon_cup(fill=(120, 200, 230), transparent_look=False):
    img, d = _canvas()
    outline = BLACK if not transparent_look else (100, 160, 200)
    d.polygon([(190, 180), (322, 180), (300, 400), (212, 400)], fill=fill, outline=outline, width=4)
    d.arc((300, 210, 380, 300), start=-90, end=90, fill=outline, width=8)
    return img


def _icon_cube(color=(190, 225, 245)):
    img, d = _canvas()
    d.polygon([(180, 260), (300, 220), (420, 260), (300, 300)], fill=(230, 245, 250), outline=BLACK, width=3)
    d.polygon([(180, 260), (300, 300), (300, 420), (180, 380)], fill=color, outline=BLACK, width=3)
    d.polygon([(300, 300), (420, 260), (420, 380), (300, 420)], fill=(150, 200, 225), outline=BLACK, width=3)
    d.line((210, 290, 250, 300), fill=WHITE, width=4)
    return img


ICONS = {
    # --- Living World ---
    "kangaroo": lambda: (lambda img, d=None: img)(_kangaroo()),
    "tiger": lambda: _quadruped((235, 140, 30), stripes=BLACK),
    "polar_bear": lambda: _quadruped((250, 250, 250), big_ears=True),
    "camel": lambda: _quadruped((210, 170, 110), hump=True, long_neck=True),
    "giraffe": lambda: _quadruped((235, 200, 90), spots=(150, 100, 30), long_neck=True),
    "pig": lambda: _quadruped((250, 180, 190)),
    "parrot": lambda: _bird((60, 170, 90), beak_color=(255, 165, 0)),
    "penguin": lambda: _bird((30, 30, 40), beak_color=(255, 165, 0), long_neck=False, big_wing=False),
    "chameleon": lambda: _chameleon(),
    "elephant": lambda: _quadruped((160, 165, 170), big_ears=True, trunk=True),
    "bee": lambda: _bee(),
    "spider": lambda: _spider(),
    "turtle": lambda: _turtle(),
    "frog": lambda: _frog(),
    "dog": lambda: _quadruped((190, 150, 100)),
    "zebra": lambda: _quadruped((250, 250, 250), stripes=BLACK),
    "horse": lambda: _quadruped((140, 95, 60), long_neck=True),
    "rabbit": lambda: _quadruped((235, 235, 235), big_ears=True, tail="poof"),
    "cow": lambda: _quadruped((250, 250, 250), spots=BLACK),

    # --- Object / State / Commonsense ---
    "ice_cube": lambda: _icon_cube(),
    "rock": lambda: _rock(),
    "wooden_block": lambda: _wooden_block(),
    "iron_nail": lambda: _nail(),
    "scissors": lambda: _scissors(),
    "spoon": lambda: _spoon(),
    "glass_cup": lambda: _icon_cup(fill=(220, 240, 250), transparent_look=True),
    "rubber_ball": lambda: _ball(),
    "light_bulb": lambda: _bulb(),
    "pen": lambda: _pen(),
    "fork": lambda: _fork(),
    "clock": lambda: _clock(),
    "plate": lambda: _plate(),
    "plastic_cup": lambda: _icon_cup(fill=(250, 100, 100)),
    "pillow": lambda: _pillow(),
    "brick": lambda: _brick(),
    "key": lambda: _key(),

    # --- Public / Traffic Knowledge ---
    "stop_sign": lambda: _sign("octagon", (200, 20, 20), WHITE, _text_symbol("STOP", big=False)),
    "parking_sign": lambda: _sign("square", (30, 90, 200), WHITE, _text_symbol("P")),
    "no_entry_sign": lambda: _sign("circle", (200, 20, 20), WHITE, lambda d, cx, cy, r: _bar(d, cx, cy, r)),
    "speed_limit_sign": lambda: _sign("circle", WHITE, (200, 20, 20), _text_symbol("50", color=BLACK)),
    "pedestrian_crossing_sign": lambda: _sign("triangle", (250, 220, 60), BLACK, _walker),
    "restaurant_sign": lambda: _sign("square", (30, 90, 200), WHITE, _fork_knife_symbol),
    "hospital_sign": lambda: _sign("square", (30, 90, 200), WHITE, _plus_symbol),
    "yield_sign": lambda: _sign("triangle", WHITE, (200, 20, 20)),
    "no_parking_sign": lambda: _sign("circle", (30, 90, 200), (200, 20, 20), _p_slash),
    "recycling_symbol": lambda: _recycling(),
    "trash_can_symbol": lambda: _trash_can(),
    "exit_sign": lambda: _sign("square", (20, 150, 60), WHITE, _door_arrow),
    "no_smoking_sign": lambda: _sign("circle", WHITE, (200, 20, 20), _cigarette_slash),

    # --- Cultural / Symbolic Knowledge ---
    "pumpkin": lambda: _pumpkin(),
    "christmas_tree": lambda: _tree(),
    "heart": lambda: _heart(),
    "star": lambda: _star_icon(),
    "guitar": lambda: _guitar(),
    "drum": lambda: _drum(),
    "church": lambda: _icon_house(with_cross=True, wall_color=(220, 210, 200)),
    "house": lambda: _icon_house(with_cross=False),
    "chess_king": lambda: _chess_king(),
    "dice": lambda: _dice(),
    "thermometer": lambda: _thermometer(),
    "ruler": lambda: _ruler(),
    "birthday_cake": lambda: _cake(),
    "bread_loaf": lambda: _bread(),
    "musical_note": lambda: _music_note(),
    "question_mark": lambda: _question_mark(),
    "dove": lambda: _bird((245, 245, 245), beak_color=(255, 190, 60), big_wing=True),
    "crow": lambda: _bird((25, 25, 25), beak_color=(40, 40, 40), big_wing=True),
    "weighing_scale": lambda: _scale(),
}


def _kangaroo():
    img, d = _canvas()
    body = (190, 140, 90)
    cx, cy = 236, 260
    # tail (thick, curving down to the ground -- the single most distinctive kangaroo cue)
    d.line([(cx - 60, cy + 20), (cx - 130, cy + 80), (cx - 170, cy + 190)],
           fill=body, width=34, joint="curve")
    d.line([(cx - 60, cy + 20), (cx - 130, cy + 80), (cx - 170, cy + 190)],
           fill=BLACK, width=3, joint="curve")
    # upright torso (kangaroos stand tall)
    d.polygon([(cx - 50, cy - 40), (cx + 40, cy - 60), (cx + 55, cy + 90), (cx - 40, cy + 110)],
               fill=body, outline=BLACK, width=3)
    # small forearms held up at the chest
    d.line((cx - 20, cy - 10, cx - 55, cy + 20), fill=body, width=16)
    d.line((cx - 20, cy - 10, cx - 55, cy + 20), fill=BLACK, width=2)
    # head + snout, tilted forward
    d.ellipse((cx + 10, cy - 150, cx + 95, cy - 65), fill=body, outline=BLACK, width=3)
    d.polygon([(cx + 85, cy - 100), (cx + 130, cy - 90), (cx + 90, cy - 75)], fill=body, outline=BLACK, width=2)
    d.polygon([(cx + 25, cy - 150), (cx + 40, cy - 190), (cx + 55, cy - 150)], fill=body, outline=BLACK, width=2)  # ear
    d.ellipse((cx + 68, cy - 118, cx + 80, cy - 106), fill=BLACK)  # eye
    # big powerful hind leg + foot (the other distinctive cue)
    d.polygon([(cx - 10, cy + 60), (cx + 55, cy + 70), (cx + 65, cy + 190), (cx + 10, cy + 210),
               (cx - 30, cy + 150)], fill=body, outline=BLACK, width=3)
    d.polygon([(cx - 30, cy + 150), (cx - 110, cy + 190), (cx - 100, cy + 220), (cx - 5, cy + 225)],
               fill=body, outline=BLACK, width=3)  # long foot
    return img


def _chameleon():
    img, d = _canvas()
    cx, cy = 256, 300
    d.polygon(_poly_blob(cx, cy, 110, n=16, jitter=0.15, seed=3), fill=(90, 180, 90), outline=BLACK, width=3)
    d.ellipse((cx + 70, cy - 60, cx + 150, cy + 20), fill=(90, 180, 90), outline=BLACK, width=3)
    d.ellipse((cx + 100, cy - 45, cx + 120, cy - 25), fill=WHITE, outline=BLACK, width=2)
    d.ellipse((cx + 106, cy - 39, cx + 114, cy - 31), fill=BLACK)
    curl_pts = [(cx - 100, cy + 20)]
    for i in range(1, 10):
        ang = i * 0.9
        rr = 10 + i * 6
        curl_pts.append((cx - 100 - rr * math.cos(ang), cy + 20 + rr * math.sin(ang)))
    d.line(curl_pts, fill=(90, 180, 90), width=16, joint="curve")
    return img


def _bee():
    img, d = _canvas()
    cx, cy = 256, 280
    d.ellipse((cx - 40, cy - 200, cx + 100, cy - 60), fill=(240, 240, 255, 150), outline=BLACK, width=2)
    d.ellipse((cx - 60, cy - 60, cx + 60, cy + 60), fill=(255, 210, 40), outline=BLACK, width=3)
    for i in range(-1, 2):
        d.rectangle((cx - 15, cy - 20 + i * 30, cx + 15, cy - 5 + i * 30), fill=BLACK)
    d.ellipse((cx - 20, cy - 100, cx + 20, cy - 60), fill=BLACK, outline=BLACK, width=2)
    return img


def _spider():
    img, d = _canvas()
    cx, cy = 256, 280
    d.ellipse((cx - 60, cy - 40, cx + 60, cy + 60), fill=(40, 40, 40), outline=BLACK, width=3)
    d.ellipse((cx - 30, cy - 90, cx + 30, cy - 40), fill=(40, 40, 40), outline=BLACK, width=3)
    for i in range(4):
        ang = -0.9 + i * 0.6
        d.line((cx - 30, cy, cx - 30 - 130 * math.cos(ang), cy - 130 * math.sin(ang)), fill=BLACK, width=8)
        d.line((cx + 30, cy, cx + 30 + 130 * math.cos(ang), cy - 130 * math.sin(ang)), fill=BLACK, width=8)
    return img


def _turtle():
    img, d = _canvas()
    cx, cy = 256, 290
    d.ellipse((cx - 120, cy - 80, cx + 120, cy + 80), fill=(80, 160, 90), outline=BLACK, width=4)
    for dx, dy in [(-60, -30), (0, -40), (60, -30), (-60, 30), (0, 40), (60, 30), (0, 0)]:
        d.ellipse((cx + dx - 25, cy + dy - 25, cx + dx + 25, cy + dy + 25), outline=BLACK, width=3)
    d.ellipse((cx + 100, cy - 25, cx + 165, cy + 25), fill=(110, 190, 110), outline=BLACK, width=3)
    for lx, ly in [(-90, 60), (-30, 90), (40, 90), (90, 60)]:
        d.ellipse((cx + lx - 18, cy + ly - 12, cx + lx + 18, cy + ly + 18), fill=(110, 190, 110), outline=BLACK, width=2)
    return img


def _frog():
    img, d = _canvas()
    cx, cy = 256, 300
    d.ellipse((cx - 100, cy - 40, cx + 100, cy + 100), fill=(90, 190, 90), outline=BLACK, width=3)
    d.ellipse((cx - 70, cy - 100, cx - 20, cy - 50), fill=(90, 190, 90), outline=BLACK, width=3)
    d.ellipse((cx + 20, cy - 100, cx + 70, cy - 50), fill=(90, 190, 90), outline=BLACK, width=3)
    d.ellipse((cx - 55, cy - 90, cx - 35, cy - 70), fill=BLACK)
    d.ellipse((cx + 35, cy - 90, cx + 55, cy - 70), fill=BLACK)
    d.polygon([(cx - 100, cy + 60), (cx - 160, cy + 110), (cx - 90, cy + 110)], fill=(90, 190, 90), outline=BLACK)
    d.polygon([(cx + 100, cy + 60), (cx + 160, cy + 110), (cx + 90, cy + 110)], fill=(90, 190, 90), outline=BLACK)
    return img


def _rock():
    img, d = _canvas()
    d.polygon(_poly_blob(256, 300, 140, n=10, jitter=0.35, seed=7), fill=(140, 140, 140), outline=BLACK, width=3)
    for sx, sy in [(-30, -20), (20, 10), (-10, 40)]:
        d.ellipse((256 + sx - 10, 300 + sy - 6, 256 + sx + 10, 300 + sy + 6), fill=(110, 110, 110))
    return img


def _wooden_block():
    img, d = _canvas()
    d.rectangle((150, 220, 362, 360), fill=(180, 130, 80), outline=BLACK, width=4)
    for y in range(240, 350, 22):
        d.line((160, y, 352, y + 6), fill=(150, 105, 60), width=3)
    return img


def _nail():
    img, d = _canvas()
    d.rectangle((246, 150, 266, 380), fill=(170, 170, 175), outline=BLACK, width=3)
    d.polygon([(226, 380), (286, 380), (256, 440)], fill=(170, 170, 175), outline=BLACK, width=3)
    d.ellipse((216, 120, 296, 155), fill=(190, 190, 195), outline=BLACK, width=3)
    return img


def _scissors():
    img, d = _canvas()
    d.line((180, 180, 340, 340), fill=(180, 180, 185), width=18)
    d.line((340, 180, 180, 340), fill=(180, 180, 185), width=18)
    d.ellipse((150, 330, 210, 390), outline=BLACK, width=8)
    d.ellipse((310, 330, 370, 390), outline=BLACK, width=8)
    d.ellipse((250, 250, 272, 272), fill=(90, 90, 90))
    return img


def _spoon():
    img, d = _canvas()
    d.ellipse((196, 140, 316, 280), fill=(200, 200, 205), outline=BLACK, width=4)
    d.rectangle((242, 270, 270, 420), fill=(200, 200, 205), outline=BLACK, width=4)
    return img


def _ball():
    img, d = _canvas()
    d.ellipse((136, 176, 376, 416), fill=(230, 90, 90), outline=BLACK, width=4)
    d.arc((136, 176, 376, 416), start=200, end=340, fill=WHITE, width=6)
    return img


def _bulb():
    img, d = _canvas()
    d.ellipse((176, 120, 336, 280), fill=(255, 235, 140), outline=BLACK, width=4)
    d.rectangle((226, 270, 286, 320), fill=(190, 190, 195), outline=BLACK, width=3)
    for y in range(320, 360, 12):
        d.line((226, y, 286, y), fill=BLACK, width=3)
    for ang in range(0, 360, 45):
        x2 = 256 + 140 * math.cos(math.radians(ang))
        y2 = 200 + 140 * math.sin(math.radians(ang))
        d.line((256 + 90 * math.cos(math.radians(ang)), 200 + 90 * math.sin(math.radians(ang)), x2, y2),
               fill=(255, 200, 60), width=4)
    return img


def _pen():
    img, d = _canvas()
    d.polygon([(180, 380), (330, 230), (360, 260), (210, 410)], fill=(40, 90, 200), outline=BLACK, width=3)
    d.polygon([(330, 230), (360, 200), (390, 230), (360, 260)], fill=(220, 220, 225), outline=BLACK, width=3)
    d.polygon([(180, 380), (170, 420), (210, 410)], fill=BLACK)
    return img


def _fork():
    img, d = _canvas()
    for x in (216, 246, 276, 306):
        d.rectangle((x, 130, x + 14, 240), fill=(200, 200, 205), outline=BLACK, width=2)
    d.rectangle((216, 230, 320, 260), fill=(200, 200, 205), outline=BLACK, width=3)
    d.rectangle((246, 250, 286, 420), fill=(200, 200, 205), outline=BLACK, width=4)
    return img


def _clock():
    img, d = _canvas()
    d.ellipse((116, 116, 396, 396), fill=WHITE, outline=BLACK, width=8)
    for ang in range(0, 360, 30):
        x1 = 256 + 120 * math.cos(math.radians(ang))
        y1 = 256 + 120 * math.sin(math.radians(ang))
        x2 = 256 + 135 * math.cos(math.radians(ang))
        y2 = 256 + 135 * math.sin(math.radians(ang))
        d.line((x1, y1, x2, y2), fill=BLACK, width=4)
    d.line((256, 256, 256, 160), fill=BLACK, width=8)
    d.line((256, 256, 330, 280), fill=BLACK, width=8)
    return img


def _plate():
    img, d = _canvas()
    d.ellipse((106, 106, 406, 406), fill=(245, 245, 245), outline=BLACK, width=4)
    d.ellipse((166, 166, 346, 346), outline=(200, 200, 200), width=3)
    return img


def _pillow():
    img, d = _canvas()
    d.rounded_rectangle((130, 190, 382, 380), radius=60, fill=(250, 240, 250), outline=BLACK, width=4)
    d.line((130, 285, 382, 285), fill=(220, 200, 220), width=2)
    return img


def _brick():
    img, d = _canvas()
    d.rectangle((120, 220, 392, 340), fill=(180, 70, 55), outline=BLACK, width=4)
    d.line((256, 220, 256, 340), fill=(120, 40, 30), width=3)
    return img


def _key():
    img, d = _canvas()
    d.ellipse((140, 160, 260, 280), outline=BLACK, width=14)
    d.rectangle((250, 210, 400, 230), fill=(210, 180, 60), outline=BLACK, width=3)
    d.rectangle((360, 230, 375, 260), fill=(210, 180, 60), outline=BLACK, width=2)
    d.rectangle((385, 230, 400, 255), fill=(210, 180, 60), outline=BLACK, width=2)
    return img


def _walker(d, cx, cy, r):
    d.ellipse((cx - 12, cy - 60, cx + 12, cy - 36), fill=BLACK)
    d.line((cx, cy - 36, cx, cy + 20), fill=BLACK, width=8)
    d.line((cx, cy - 10, cx - 30, cy + 10), fill=BLACK, width=6)
    d.line((cx, cy - 10, cx + 25, cy - 30), fill=BLACK, width=6)
    d.line((cx, cy + 20, cx - 25, cy + 60), fill=BLACK, width=6)
    d.line((cx, cy + 20, cx + 20, cy + 65), fill=BLACK, width=6)


def _fork_knife_symbol(d, cx, cy, r):
    d.line((cx - 30, cy - 60, cx - 30, cy + 60), fill=WHITE, width=8)
    d.line((cx - 45, cy - 60, cx - 45, cy - 20), fill=WHITE, width=6)
    d.line((cx - 15, cy - 60, cx - 15, cy - 20), fill=WHITE, width=6)
    d.polygon([(cx + 20, cy - 60), (cx + 35, cy - 60), (cx + 20, cy + 60), (cx + 5, cy + 40)], fill=WHITE)


def _plus_symbol(d, cx, cy, r):
    d.rectangle((cx - 15, cy - 60, cx + 15, cy + 60), fill=(220, 30, 30))
    d.rectangle((cx - 60, cy - 15, cx + 60, cy + 15), fill=(220, 30, 30))


def _door_arrow(d, cx, cy, r):
    d.rectangle((cx - 50, cy - 60, cx - 10, cy + 60), outline=WHITE, width=6)
    d.line((cx - 40, cy, cx + 60, cy), fill=WHITE, width=8)
    d.polygon([(cx + 60, cy), (cx + 30, cy - 20), (cx + 30, cy + 20)], fill=WHITE)


def _recycling():
    img, d = _canvas()
    green = (30, 150, 60)
    cx, cy, r = 256, 246, 175
    for i in range(3):
        ang = i * 2 * math.pi / 3 - math.pi / 2
        x1 = cx + r * 0.35 * math.cos(ang)
        y1 = cy + r * 0.35 * math.sin(ang)
        x2 = cx + r * math.cos(ang + 1.55)
        y2 = cy + r * math.sin(ang + 1.55)
        d.line((x1, y1, x2, y2), fill=green, width=44, joint="curve")
        ang2 = ang + 1.55
        px = cx + r * math.cos(ang2)
        py = cy + r * math.sin(ang2)
        perp = ang2 + math.pi / 2
        tip = (px + 55 * math.cos(ang2), py + 55 * math.sin(ang2))
        base1 = (px + 40 * math.cos(perp), py + 40 * math.sin(perp))
        base2 = (px - 40 * math.cos(perp), py - 40 * math.sin(perp))
        d.polygon([tip, base1, base2], fill=green)
    return img


def _trash_can():
    img, d = _canvas()
    d.polygon([(180, 180, ), (332, 180), (320, 420), (192, 420)], fill=(120, 130, 140), outline=BLACK, width=4)
    d.rectangle((160, 150, 352, 180), fill=(100, 110, 120), outline=BLACK, width=3)
    d.rectangle((220, 110, 292, 150), fill=(100, 110, 120), outline=BLACK, width=3)
    for x in (220, 256, 292):
        d.line((x, 210, x, 390), fill=BLACK, width=4)
    return img


def _pumpkin():
    img, d = _canvas()
    d.ellipse((136, 200, 376, 400), fill=(240, 130, 30), outline=BLACK, width=4)
    for x in (186, 236, 286, 336):
        d.line((x, 205, x, 395), fill=(200, 100, 20), width=4)
    d.rectangle((240, 140, 272, 200), fill=(90, 140, 60), outline=BLACK, width=3)
    return img


def _tree():
    img, d = _canvas()
    d.polygon([(256, 100), (356, 250), (306, 250), (386, 350), (330, 350), (400, 430), (112, 430),
               (182, 350), (126, 350), (206, 250), (156, 250)], fill=(40, 130, 60), outline=BLACK, width=4)
    d.rectangle((236, 430, 276, 470), fill=(110, 70, 40), outline=BLACK, width=3)
    return img


def _heart():
    img, d = _canvas()
    cx, cy = 256, 220
    d.pieslice((cx - 110, cy - 60, cx, cy + 60), 180, 360, fill=(220, 40, 60))
    d.pieslice((cx, cy - 60, cx + 110, cy + 60), 180, 360, fill=(220, 40, 60))
    d.polygon([(cx - 108, cy + 10), (cx + 108, cy + 10), (cx, cy + 220)], fill=(220, 40, 60))
    return img


def _star_icon():
    img, d = _canvas()
    d.polygon(_star(256, 256, 190, 80), fill=(250, 200, 40), outline=BLACK, width=4)
    return img


def _guitar():
    img, d = _canvas()
    d.ellipse((166, 260, 296, 420), fill=(180, 120, 60), outline=BLACK, width=4)
    d.ellipse((196, 190, 306, 330), fill=(190, 130, 65), outline=BLACK, width=4)
    d.ellipse((230, 250, 270, 290), fill=(50, 30, 15))
    d.rectangle((225, 90, 250, 220), fill=(120, 80, 40), outline=BLACK, width=3)
    for y in range(100, 200, 25):
        d.line((225, y, 250, y), fill=(220, 220, 200), width=2)
    return img


def _drum():
    img, d = _canvas()
    d.ellipse((150, 170, 362, 250), fill=(220, 60, 60), outline=BLACK, width=4)
    d.rectangle((150, 210, 362, 340), fill=(220, 60, 60), outline=BLACK, width=4)
    d.ellipse((150, 300, 362, 380), fill=(180, 40, 40), outline=BLACK, width=4)
    for x in range(160, 362, 40):
        d.line((x, 220, x - 10, 340), fill=(140, 20, 20), width=3)
    return img


def _chess_king():
    img, d = _canvas()
    fill = (235, 235, 235)
    # wide flared base (chess pieces sit on a broad foot)
    d.polygon([(150, 440), (362, 440), (330, 400), (182, 400)], fill=fill, outline=BLACK, width=4)
    # tapered body narrowing upward, with a mid "waist" band -- the key cue that reads as
    # a chess piece rather than a bottle
    d.polygon([(182, 400), (330, 400), (300, 300), (212, 300)], fill=fill, outline=BLACK, width=4)
    d.rectangle((205, 288, 307, 312), fill=fill, outline=BLACK, width=4)  # waist band/collar
    d.polygon([(212, 288), (300, 288), (280, 210), (232, 210)], fill=fill, outline=BLACK, width=4)
    # rounded "head" of the piece
    d.ellipse((206, 165, 306, 225), fill=fill, outline=BLACK, width=4)
    d.rectangle((236, 130, 276, 175), fill=fill, outline=BLACK, width=3)
    # the cross on top -- the single most recognizable king cue (vs. e.g. a ball for a pawn)
    d.rectangle((244, 65, 268, 130), fill=fill, outline=BLACK, width=4)
    d.rectangle((216, 88, 296, 108), fill=fill, outline=BLACK, width=4)
    return img


def _dice():
    img, d = _canvas()
    d.rounded_rectangle((150, 150, 362, 362), radius=30, fill=WHITE, outline=BLACK, width=6)
    for dx, dy in [(-60, -60), (60, 60), (-60, 60), (60, -60), (0, 0)]:
        d.ellipse((256 + dx - 20, 256 + dy - 20, 256 + dx + 20, 256 + dy + 20), fill=BLACK)
    return img


def _thermometer():
    img, d = _canvas()
    d.rounded_rectangle((236, 100, 276, 340), radius=20, outline=BLACK, width=5, fill=WHITE)
    d.rectangle((246, 200, 266, 335), fill=(220, 40, 40))
    d.ellipse((216, 330, 296, 410), fill=(220, 40, 40), outline=BLACK, width=5)
    return img


def _ruler():
    img, d = _canvas()
    d.rectangle((100, 220, 412, 300), fill=(250, 220, 120), outline=BLACK, width=4)
    for i, x in enumerate(range(120, 400, 26)):
        h = 40 if i % 5 == 0 else 20
        d.line((x, 220, x, 220 + h), fill=BLACK, width=3)
    return img


def _cake():
    img, d = _canvas()
    d.rectangle((150, 280, 362, 400), fill=(250, 210, 220), outline=BLACK, width=4)
    d.rectangle((150, 280, 362, 300), fill=(240, 150, 170))
    for x in (190, 256, 322):
        d.rectangle((x - 6, 220, x + 6, 280), fill=(250, 240, 180), outline=BLACK, width=2)
        d.ellipse((x - 8, 195, x + 8, 220), fill=(255, 150, 30))
    return img


def _bread():
    img, d = _canvas()
    d.rounded_rectangle((140, 220, 372, 360), radius=70, fill=(210, 160, 90), outline=BLACK, width=4)
    for x in range(170, 350, 40):
        d.arc((x - 20, 210, x + 20, 260), start=200, end=340, fill=(160, 110, 55), width=4)
    return img


def _music_note():
    img, d = _canvas()
    d.ellipse((150, 320, 220, 380), fill=BLACK)
    d.ellipse((280, 300, 350, 360), fill=BLACK)
    d.rectangle((210, 140, 224, 360), fill=BLACK)
    d.rectangle((336, 130, 350, 340), fill=BLACK)
    d.polygon([(210, 140), (350, 130), (350, 190), (210, 200)], fill=BLACK)
    return img


def _question_mark():
    img, d = _canvas()
    d.arc((176, 120, 336, 280), start=200, end=430, fill=BLACK, width=26)
    d.line((256, 260, 256, 320), fill=BLACK, width=26)
    d.ellipse((236, 360, 276, 400), fill=BLACK)
    return img


def _scale():
    img, d = _canvas()
    d.polygon([(226, 420), (286, 420), (270, 240), (242, 240)], fill=(180, 180, 185), outline=BLACK, width=3)
    d.rectangle((150, 220, 362, 236), fill=(180, 180, 185), outline=BLACK, width=3)
    d.line((180, 230, 130, 320), fill=BLACK, width=5)
    d.line((332, 230, 382, 320), fill=BLACK, width=5)
    d.arc((100, 310, 160, 350), start=0, end=180, fill=BLACK, width=6)
    d.arc((352, 310, 412, 350), start=0, end=180, fill=BLACK, width=6)
    return img

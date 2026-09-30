"""The 40-question custom semantic diagnostic set (Phase 1 pilot only -- not a substitute
for the official Act2Answer benchmark, whose full asset release is not yet public as of
2026-09-16).

SCOPING NOTE on "Cultural / General Knowledge": the user's illustrative example for this
category ("Which person painted the Mona Lisa? Leonardo da Vinci vs Albert Einstein")
requires recognizing specific human faces, which a hand-drawn pictogram cannot support
(there's no way to make a synthetic icon "recognizable as a specific historical person"
without a real reference photo, and sourcing/verifying redistribution rights for portrait
photos was out of scope for this pass). Substituted with symbolic/associative
cultural-knowledge questions (e.g. "which item symbolizes love") that keep the same
knowledge-requiring spirit -- combining recognition of a depicted object with general
world/cultural knowledge -- without requiring face recognition. Flagged here and in the
final report, not silently swapped.

Each entry: category, question, (answer_A icon key, label), (answer_B icon key, label),
correct ("A" or "B"). All icon keys must exist in icons.ICONS.
"""

Q = [
    # --- Living World (10) ---
    ("living_world", "Which animal is associated with Australia?", ("kangaroo", "kangaroo"), ("tiger", "tiger"), "A"),
    ("living_world", "Which animal lives at the North Pole?", ("polar_bear", "polar bear"), ("camel", "camel"), "A"),
    ("living_world", "Which animal is known for a very long neck?", ("giraffe", "giraffe"), ("pig", "pig"), "A"),
    ("living_world", "Which animal can fly?", ("parrot", "parrot"), ("penguin", "penguin"), "A"),
    ("living_world", "Which animal is known for changing color to hide?", ("chameleon", "chameleon"), ("elephant", "elephant"), "A"),
    ("living_world", "Which animal produces honey?", ("bee", "bee"), ("spider", "spider"), "A"),
    ("living_world", "Which animal has a hard shell on its back?", ("turtle", "turtle"), ("frog", "frog"), "A"),
    ("living_world", "Which animal is the largest land animal?", ("elephant", "elephant"), ("dog", "dog"), "A"),
    ("living_world", "Which animal is known for black and white stripes?", ("zebra", "zebra"), ("horse", "horse"), "A"),
    ("living_world", "Which animal hops using strong hind legs?", ("rabbit", "rabbit"), ("cow", "cow"), "A"),

    # --- Object / State / Commonsense (10) ---
    ("object_state", "Which object would melt if left in the sun?", ("ice_cube", "ice cube"), ("rock", "rock"), "A"),
    ("object_state", "Which object can float on water?", ("wooden_block", "wooden block"), ("iron_nail", "iron nail"), "A"),
    ("object_state", "Which object is used to cut paper?", ("scissors", "scissors"), ("spoon", "spoon"), "A"),
    ("object_state", "Which object would shatter if dropped on a hard floor?", ("glass_cup", "glass cup"), ("rubber_ball", "rubber ball"), "A"),
    ("object_state", "Which object is a source of light?", ("light_bulb", "light bulb"), ("rock", "rock"), "A"),
    ("object_state", "Which object is used to write?", ("pen", "pen"), ("fork", "fork"), "A"),
    ("object_state", "Which object is used to tell the time?", ("clock", "clock"), ("plate", "plate"), "A"),
    ("object_state", "Which object would rust if left out in the rain?", ("iron_nail", "iron nail"), ("plastic_cup", "plastic cup"), "A"),
    ("object_state", "Which object is soft and squishy?", ("pillow", "pillow"), ("brick", "brick"), "A"),
    ("object_state", "Which object is used to unlock a door?", ("key", "key"), ("spoon", "spoon"), "A"),

    # --- Public / Traffic Knowledge (10) ---
    ("traffic_public", "Which sign tells a driver to stop?", ("stop_sign", "stop sign"), ("parking_sign", "parking sign"), "A"),
    ("traffic_public", "Which sign means no entry?", ("no_entry_sign", "no entry sign"), ("speed_limit_sign", "speed limit sign"), "A"),
    ("traffic_public", "Which sign warns of a pedestrian crossing?", ("pedestrian_crossing_sign", "pedestrian crossing sign"), ("restaurant_sign", "restaurant sign"), "A"),
    ("traffic_public", "Which sign indicates a hospital nearby?", ("hospital_sign", "hospital sign"), ("restaurant_sign", "restaurant sign"), "A"),
    ("traffic_public", "Which sign means yield to other traffic?", ("yield_sign", "yield sign"), ("stop_sign", "stop sign"), "A"),
    ("traffic_public", "Which sign allows parking?", ("parking_sign", "parking sign"), ("no_parking_sign", "no parking sign"), "A"),
    ("traffic_public", "Which symbol means recycle?", ("recycling_symbol", "recycling symbol"), ("trash_can_symbol", "trash can"), "A"),
    ("traffic_public", "Which sign shows an emergency exit?", ("exit_sign", "exit sign"), ("parking_sign", "parking sign"), "A"),
    ("traffic_public", "Which sign means no smoking allowed?", ("no_smoking_sign", "no smoking sign"), ("no_parking_sign", "no parking sign"), "A"),
    ("traffic_public", "Which sign indicates a speed limit?", ("speed_limit_sign", "speed limit sign"), ("stop_sign", "stop sign"), "A"),

    # --- Cultural / Symbolic Knowledge (10) -- see scoping note above ---
    ("cultural_symbolic", "Which item is associated with Halloween?", ("pumpkin", "pumpkin"), ("christmas_tree", "christmas tree"), "A"),
    ("cultural_symbolic", "Which item symbolizes love?", ("heart", "heart"), ("star", "star"), "A"),
    ("cultural_symbolic", "Which instrument has strings?", ("guitar", "guitar"), ("drum", "drum"), "A"),
    ("cultural_symbolic", "Which building serves a religious purpose?", ("church", "church"), ("house", "house"), "A"),
    ("cultural_symbolic", "Which game piece is used in chess?", ("chess_king", "chess king"), ("dice", "dice"), "A"),
    ("cultural_symbolic", "Which item is used to measure temperature?", ("thermometer", "thermometer"), ("ruler", "ruler"), "A"),
    ("cultural_symbolic", "Which item is associated with birthdays?", ("birthday_cake", "birthday cake"), ("bread_loaf", "loaf of bread"), "A"),
    ("cultural_symbolic", "Which item is a musical symbol?", ("musical_note", "musical note"), ("question_mark", "question mark"), "A"),
    ("cultural_symbolic", "Which item represents peace?", ("dove", "dove"), ("crow", "crow"), "A"),
    ("cultural_symbolic", "Which item is used to weigh objects?", ("weighing_scale", "weighing scale"), ("ruler", "ruler"), "A"),
]

assert len(Q) == 40

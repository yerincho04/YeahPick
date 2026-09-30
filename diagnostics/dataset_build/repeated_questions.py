"""Question bank for ``semantic_repeated_v1``.

Ten answer concepts each have five independently worded, easy knowledge questions.  The
answer label itself does not occur in its questions. Distractors are assigned cyclically
by the builder so every concept is used exactly five times as a distractor.
"""

CONCEPTS = [
    {
        "id": "kangaroo", "label": "kangaroo", "icon": "kangaroo", "category": "living_world",
        "questions": [
            ("Which animal is strongly associated with Australia?", "association", "geographic_association"),
            ("Which marsupial carries its young in a pouch?", "reproductive_trait", "biological_attribute"),
            ("Which animal moves with long hops powered by large hind legs?", "locomotion", "behavior"),
            ("Which large marsupial appears on Australia's coat of arms?", "national_symbol", "cultural_association"),
            ("Which animal uses its muscular tail for balance while jumping?", "anatomy_function", "physical_attribute"),
        ],
    },
    {
        "id": "elephant", "label": "elephant", "icon": "elephant", "category": "living_world",
        "questions": [
            ("Which animal is the largest living land mammal?", "superlative", "category_knowledge"),
            ("Which animal uses a long trunk to drink and grasp objects?", "anatomy_function", "physical_attribute"),
            ("Which mammal is known for large tusks and fan-shaped ears?", "distinctive_features", "physical_attribute"),
            ("Which animal lives in matriarch-led family herds?", "social_structure", "behavior"),
            ("Which land mammal communicates using very low-frequency rumbles?", "communication", "behavior"),
        ],
    },
    {
        "id": "bee", "label": "bee", "icon": "bee", "category": "living_world",
        "questions": [
            ("Which insect produces honey?", "food_product", "function"),
            ("Which flying pollinator returns to a hive?", "habitat", "category_knowledge"),
            ("Which insect colony has queens, workers, and drones?", "social_structure", "category_knowledge"),
            ("Which striped flying insect dances to show where flowers are?", "communication", "behavior"),
            ("Which common garden pollinator can defend its colony with a stinger?", "defense", "biological_attribute"),
        ],
    },
    {
        "id": "turtle", "label": "turtle", "icon": "turtle", "category": "living_world",
        "questions": [
            ("Which reptile carries a protective shell around its body?", "body_covering", "physical_attribute"),
            ("Which marine reptile returns to beaches to lay eggs?", "reproduction", "behavior"),
            ("Which animal can withdraw its head and limbs into a shell?", "defense", "behavior"),
            ("Which slow-moving reptile is often associated with a long lifespan?", "lifespan", "commonsense_property"),
            ("Which animal has shell sections called a carapace and plastron?", "anatomy_terms", "category_knowledge"),
        ],
    },
    {
        "id": "stop_sign", "label": "stop sign", "icon": "stop_sign", "category": "traffic_public",
        "questions": [
            ("Which road marker requires drivers to come to a complete halt?", "driver_action", "public_rule"),
            ("Which traffic marker is conventionally red and octagonal?", "shape_color", "visual_convention"),
            ("Which traffic command must a driver obey before proceeding through a controlled intersection?", "intersection_rule", "public_rule"),
            ("Which roadside command requires every approaching vehicle to pause fully?", "mandatory_action", "public_rule"),
            ("Which traffic control is placed where right-of-way requires vehicles to halt?", "right_of_way", "public_rule"),
        ],
    },
    {
        "id": "recycling_symbol", "label": "recycling symbol", "icon": "recycling_symbol", "category": "traffic_public",
        "questions": [
            ("Which environmental emblem uses three chasing arrows?", "shape", "visual_convention"),
            ("Which mark indicates that material can be processed and used again?", "material_reuse", "public_information"),
            ("Which emblem commonly appears on waste-sorting bins?", "location", "public_information"),
            ("Which sustainability icon forms a triangular loop of arrows?", "shape", "visual_convention"),
            ("Which mark asks consumers to separate reusable waste?", "consumer_action", "public_information"),
        ],
    },
    {
        "id": "scissors", "label": "scissors", "icon": "scissors", "category": "object_state",
        "questions": [
            ("Which tool is commonly used to cut paper?", "paper_function", "function"),
            ("Which handheld tool has two blades joined at a pivot?", "mechanism", "physical_attribute"),
            ("Which implement would a tailor use to cut fabric?", "occupation_use", "function"),
            ("Which cutting tool is operated through a pair of finger loops?", "grip", "physical_attribute"),
            ("Which craft tool makes repeated snips along an edge?", "craft_use", "function"),
        ],
    },
    {
        "id": "clock", "label": "clock", "icon": "clock", "category": "object_state",
        "questions": [
            ("Which wall-mounted device is used to tell the time?", "time_function", "function"),
            ("Which round-faced device may have separate hour and minute hands?", "display", "physical_attribute"),
            ("Which timepiece may make a ticking sound?", "sound", "sensory_attribute"),
            ("Which device would someone consult to see whether an appointment is late?", "scheduling", "function"),
            ("On which timekeeping device do both hands point upward at noon?", "noon_display", "category_knowledge"),
        ],
    },
    {
        "id": "light_bulb", "label": "light bulb", "icon": "light_bulb", "category": "object_state",
        "questions": [
            ("Which household device glows when electric current heats its filament?", "filament", "mechanism"),
            ("Which glass electrical device is screwed into a lamp socket?", "socket", "function"),
            ("Which replaceable part of a lamp illuminates a room?", "illumination", "function"),
            ("Which electrical invention is popularly associated with Thomas Edison?", "inventor", "cultural_association"),
            ("Which lamp component stops working when its filament burns out?", "failure_mode", "mechanism"),
        ],
    },
    {
        "id": "pumpkin", "label": "pumpkin", "icon": "pumpkin", "category": "cultural_symbolic",
        "questions": [
            ("Which orange crop is strongly associated with Halloween?", "holiday", "cultural_association"),
            ("Which gourd is traditionally carved into a jack-o'-lantern?", "decoration", "cultural_association"),
            ("Which vine-grown crop is commonly baked into a Thanksgiving pie?", "food_use", "cultural_association"),
            ("Which round autumn squash usually has a thick green stem?", "appearance", "physical_attribute"),
            ("Which harvest decoration contains seeds that are often roasted after carving?", "seeds", "commonsense_property"),
        ],
    },
]

assert len(CONCEPTS) == 10
assert all(len(concept["questions"]) == 5 for concept in CONCEPTS)

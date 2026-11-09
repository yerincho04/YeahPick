"""Question bank for the v2 photo VQA dataset.

Every fact is a subject relative clause, so one string yields all question forms:
  instruction          "Place the cube on the {noun} {rel} {fact}."
  semantic reasoning   "Which {noun} {fact}?"
  visual grounding     "Which side contains the {noun} {rel} {fact}?"
A fact never names its own class, and it must single out its class against the other tile of the pair.
"""

# ---------------------------------------------------------------- state: per object family
# family -> noun, attribute question, {class: (label, attribute answer, [facts])}
OPEN_Q = "Is the {noun} on the {side} tile open or closed?"
ON_Q = "Is the {noun} on the {side} tile switched on or off?"
FILL_Q = "Is the {noun} on the {side} tile empty or full?"
SEAT_Q = "Is the {noun} on the {side} tile empty or occupied?"
SEAT_OBJ_Q = "What piece of furniture is shown on the {side} tile?"  # a person may sit on it
CUT_Q = "Is the {noun} on the {side} tile whole or cut?"
COOK_Q = "Is the {noun} on the {side} tile cooked or raw?"
CLEAN_Q = "Is the {noun} on the {side} tile clean or dirty?"
TIDY_Q = "Is the {noun} on the {side} tile tidy or messy?"
WET_Q = "Is the {noun} on the {side} tile wet or dry?"

STATE = {
    "toilet_open": ("toilet", OPEN_Q, {
        "open": ("toilet with the lid up", "open", [
            "someone could use right away without lifting anything",
            "lets you see the water in its bowl",
            "a dropped toy could fall straight into"]),
        "closed": ("toilet with the lid down", "closed", [
            "would have to be lifted before anyone could use it",
            "hides the water in its bowl",
            "stops a dropped toy from falling into the bowl"])}),
    "handbag_open": ("handbag", OPEN_Q, {
        "open": ("open handbag", "open", [
            "someone could drop their keys into right now without unzipping anything",
            "a pickpocket could easily reach into",
            "might spill its contents if knocked over"]),
        "closed": ("closed handbag", "closed", [
            "keeps its contents safe from a pickpocket",
            "must be unzipped or unclasped before you can reach inside",
            "would not spill its contents if knocked over"])}),
    "oven_open": ("oven", OPEN_Q, {
        "open": ("open oven", "open", [
            "is letting its heat escape into the kitchen",
            "you could slide a baking tray into right now",
            "shows the racks inside"]),
        "closed": ("closed oven", "closed", [
            "is keeping its heat inside",
            "must be opened before a baking tray can go in",
            "hides what is inside"])}),
    "suitcase_open": ("suitcase", OPEN_Q, {
        "open": ("open suitcase", "open", [
            "someone could put more clothes into right now",
            "shows what is packed inside",
            "is not ready to be wheeled to the airport"]),
        "closed": ("closed suitcase", "closed", [
            "is ready to be wheeled to the airport",
            "hides what is packed inside",
            "must be unzipped before you can pack more clothes"])}),
    "bottle_open": ("bottle", OPEN_Q, {
        "open": ("open bottle", "open", [
            "could spill its drink if knocked over",
            "you could drink from right now",
            "is slowly losing its fizz"]),
        "closed": ("closed bottle", "closed", [
            "could be tossed into a bag without leaking",
            "keeps its drink fizzy",
            "must be uncapped before you can drink from it"])}),
    "book_open": ("book", OPEN_Q, {
        "open": ("open book", "open", [
            "someone is in the middle of reading",
            "shows its pages",
            "could be read right now without touching it"]),
        "closed": ("closed book", "closed", [
            "shows only its cover",
            "is ready to go back on the shelf",
            "must be opened before anyone can read it"])}),
    "monitor_on": ("monitor", ON_Q, {
        "on": ("monitor that is on", "on", [
            "is using electricity right now",
            "someone could be reading a document on right now",
            "is lighting up the desk"]),
        "off": ("monitor that is off", "off", [
            "shows only a dark screen",
            "is not using any electricity right now",
            "must be switched on before anyone can use it"])}),
    "laptop_on": ("laptop", ON_Q, {
        "on": ("laptop that is on", "on", [
            "someone could be reading an email on right now",
            "is draining its battery",
            "has a glowing screen"]),
        "off": ("laptop that is off", "off", [
            "must be powered up before anyone can use it",
            "is not draining its battery",
            "shows only a black screen"])}),
    "phone_on": ("phone", ON_Q, {
        "on": ("phone that is on", "on", [
            "could show a new message right now",
            "is draining its battery",
            "has a lit screen"]),
        "off": ("phone that is off", "off", [
            "must be woken up before you can read your messages",
            "is not draining its battery",
            "shows only a black screen"])}),
    "bowl_empty": ("bowl", FILL_Q, {
        "empty": ("empty bowl", "empty", [
            "is ready to be filled with soup",
            "someone has finished eating from",
            "would be lighter to carry"]),
        "full": ("full bowl", "full", [
            "has food someone could eat right now",
            "would spill if tipped over",
            "would be heavier to carry"])}),
    "cup_empty": ("cup", FILL_Q, {
        "empty": ("empty cup", "empty", [
            "is ready for a fresh pour of coffee",
            "someone has finished drinking from",
            "would not spill anything if knocked over"]),
        "full": ("full cup", "full", [
            "has a drink someone could sip right now",
            "would spill if knocked over",
            "needs to be carried carefully"])}),
    "wineglass_empty": ("wine glass", FILL_Q, {
        "empty": ("empty wine glass", "empty", [
            "is ready to be filled",
            "someone has finished their drink from",
            "would not stain the tablecloth if knocked over"]),
        "full": ("full wine glass", "full", [
            "has a drink in it right now",
            "could stain the tablecloth if knocked over",
            "was just poured for a guest"])}),
    "vase_empty": ("vase", FILL_Q, {
        "empty": ("empty vase", "empty", [
            "is waiting for someone to put flowers in it",
            "has nothing in it that needs watering",
            "would be lighter to carry"]),
        "full": ("vase with flowers", "full", [
            "is holding flowers right now",
            "needs its water changed every few days",
            "would be heavier to carry"])}),
    "bench_empty": ("bench", SEAT_Q, {
        "empty": ("empty bench", "empty", [
            "a tired walker could sit on right away",
            "nobody is resting on right now",
            "has room for a whole family to sit"]),
        "full": ("occupied bench", "occupied", [
            "someone is resting on right now",
            "a tired walker would have to share",
            "is being used right now"])}),
    "chair_empty": ("chair", SEAT_Q, {
        "empty": ("empty chair", "empty", [
            "a guest could sit on right away",
            "nobody is using right now",
            "is free for the next person"]),
        "full": ("occupied chair", "occupied", [
            "someone is sitting on right now",
            "a guest would have to wait for",
            "is being used right now"])}),
    "couch_empty": ("couch", SEAT_Q, {
        "empty": ("empty couch", "empty", [
            "a family could sit down on right away",
            "nobody is relaxing on right now",
            "is free for the next person"]),
        "full": ("occupied couch", "occupied", [
            "someone is relaxing on right now",
            "a family would have to share",
            "is being used right now"])}),
    "table_empty": ("table", "Is the {noun} on the {side} tile empty or covered with things?", {
        "empty": ("empty table", "empty", [
            "is ready for a new meal to be laid out",
            "has nothing on it to clear away",
            "could be wiped down right away"]),
        "full": ("table covered with things", "covered", [
            "would need clearing before a new meal could be served",
            "has things on it that someone might be using",
            "would take a while to wipe down"])}),
    "apple_cut": ("apple", CUT_Q, {
        "cut": ("cut apple", "cut", [
            "is ready to be shared out right away",
            "will turn brown in the air within an hour",
            "someone has already taken a knife to"]),
        "whole": ("whole apple", "whole", [
            "could sit in a fruit bowl for a week",
            "would roll if placed on a slope",
            "still needs a knife before it can be shared"])}),
    "banana_cut": ("banana", CUT_Q, {
        "cut": ("cut banana", "cut", [
            "is ready to be added to a fruit salad",
            "someone has already sliced",
            "will turn brown on the plate soon"]),
        "whole": ("whole banana", "whole", [
            "could be carried in a lunchbox without browning",
            "has not been touched by a knife",
            "still needs slicing before it can go in a fruit salad"])}),
    "orange_cut": ("orange", CUT_Q, {
        "cut": ("cut orange", "cut", [
            "is ready to be squeezed for juice",
            "someone has already sliced open",
            "would dry out if left on the counter overnight"]),
        "whole": ("whole orange", "whole", [
            "could be tossed to a friend without dripping",
            "still has to be peeled or sliced",
            "would keep for a week in a fruit bowl"])}),
    "cake_cut": ("cake", CUT_Q, {
        "cut": ("cut cake", "cut", [
            "someone has already started serving",
            "has slices ready to put on plates",
            "could no longer be presented whole at a party"]),
        "whole": ("whole cake", "whole", [
            "has not been served to anyone yet",
            "could still be presented in one piece at a party",
            "still needs a knife before anyone gets a slice"])}),
    "pizza_cut": ("pizza", CUT_Q, {
        "cut": ("cut pizza", "cut", [
            "is ready to be eaten by the slice",
            "someone could grab a piece of right now",
            "has already been through a pizza cutter"]),
        "whole": ("whole pizza", "whole", [
            "still needs a cutter before it can be shared",
            "nobody has taken a piece of yet",
            "could still be boxed up in one piece"])}),
    "sandwich_cut": ("sandwich", CUT_Q, {
        "cut": ("cut sandwich", "cut", [
            "is easier to share between two people",
            "someone has already sliced into pieces",
            "a small child could hold more easily"]),
        "whole": ("whole sandwich", "whole", [
            "has not been touched by a knife",
            "would be harder for a small child to hold",
            "still needs cutting before it can be shared"])}),
    "donut_cut": ("donut", CUT_Q, {
        "cut": ("cut donut", "cut", [
            "someone has already bitten or sliced into",
            "could be shared with a friend right away",
            "would no longer look perfect in a bakery window"]),
        "whole": ("whole donut", "whole", [
            "nobody has bitten or cut into yet",
            "would still look perfect in a bakery window",
            "still needs cutting before it can be shared"])}),
    "carrot_cut": ("carrot", CUT_Q, {
        "cut": ("cut carrot", "cut", [
            "is ready to drop into a soup",
            "someone has already chopped",
            "would dry out quickly in the fridge"]),
        "whole": ("whole carrot", "whole", [
            "still needs chopping before it can go into a soup",
            "would keep longest in the fridge",
            "could be peeled and grated"])}),
    "broccoli_cooked": ("broccoli", COOK_Q, {
        "cooked": ("cooked broccoli", "cooked", [
            "has already been steamed or boiled",
            "is soft enough for a toddler to chew",
            "is ready to serve as a hot side dish"]),
        "raw": ("raw broccoli", "raw", [
            "would crunch loudly when bitten",
            "still needs steaming before dinner",
            "could be dipped into a cold salad dip"])}),
    "carrot_cooked": ("carrot", COOK_Q, {
        "cooked": ("cooked carrot", "cooked", [
            "has been softened by boiling or roasting",
            "is ready to serve as a warm side dish",
            "would barely crunch when bitten"]),
        "raw": ("raw carrot", "raw", [
            "would crunch loudly when bitten",
            "could be grated into a fresh salad",
            "still needs roasting before dinner"])}),
    "bowl_clean": ("bowl", CLEAN_Q, {
        "clean": ("clean bowl", "clean", [
            "is ready to have food served in it",
            "looks like it just came out of the dishwasher",
            "would pass a restaurant inspection"]),
        "dirty": ("dirty bowl", "dirty", [
            "needs washing before anyone eats from it",
            "still has leftover food stuck to it",
            "belongs in the sink"])}),
    "toilet_clean": ("toilet", CLEAN_Q, {
        "clean": ("clean toilet", "clean", [
            "would pass a hygiene inspection",
            "a guest would be happy to use",
            "looks freshly scrubbed"]),
        "dirty": ("dirty toilet", "dirty", [
            "would fail a hygiene inspection",
            "a guest would refuse to use",
            "has been neglected for weeks"])}),
    "table_clean": ("table", CLEAN_Q, {
        "clean": ("clean table", "clean", [
            "is ready for guests to sit down and eat",
            "has just been wiped down",
            "would not make your sleeves sticky"]),
        "dirty": ("dirty table", "dirty", [
            "needs wiping before the next guests sit down",
            "has crumbs and spills left from the last meal",
            "would make your sleeves sticky"])}),
    "hydrant_clean": ("fire hydrant", CLEAN_Q, {
        "clean": ("clean fire hydrant", "clean", [
            "looks freshly painted",
            "has been well looked after by the city",
            "would look good in a city brochure"]),
        "dirty": ("dirty fire hydrant", "dirty", [
            "is stained with rust and grime",
            "needs a scrub and a new coat of paint",
            "has been neglected by the city for years"])}),
    "suitcase_clean": ("suitcase", CLEAN_Q, {
        "clean": ("clean suitcase", "clean", [
            "looks fresh out of the shop",
            "would not leave marks on your bedspread",
            "has been well cared for"]),
        "dirty": ("dirty suitcase", "dirty", [
            "is covered in marks from many trips",
            "would leave dirt on your bedspread",
            "needs a scrub after a muddy journey"])}),
    "motorcycle_clean": ("motorcycle", CLEAN_Q, {
        "clean": ("clean motorcycle", "clean", [
            "looks freshly washed and polished",
            "is ready to be shown at a bike show",
            "has no mud on its wheels"]),
        "dirty": ("dirty motorcycle", "dirty", [
            "is splattered with mud from the road",
            "needs a wash before a bike show",
            "has been ridden on muddy trails"])}),
    "knife_clean": ("knife", CLEAN_Q, {
        "clean": ("clean knife", "clean", [
            "is ready to cut fresh food",
            "has just been washed",
            "has a spotless blade"]),
        "dirty": ("dirty knife", "dirty", [
            "has food smeared on its blade",
            "needs washing before cutting anything else",
            "has just been used to spread something"])}),
    "bench_clean": ("bench", CLEAN_Q, {
        "clean": ("clean bench", "clean", [
            "you could sit on in white trousers",
            "looks freshly washed",
            "has been well looked after by the park"]),
        "dirty": ("dirty bench", "dirty", [
            "would stain white trousers",
            "is covered in grime or bird droppings",
            "needs scrubbing before anyone sits on it"])}),
    "elephant_clean": ("elephant", CLEAN_Q, {
        "clean": ("clean elephant", "clean", [
            "looks freshly washed by its keeper",
            "has no mud caked on its skin",
            "has just had a bath"]),
        "dirty": ("muddy elephant", "dirty", [
            "has been rolling in mud",
            "is coated in dust or dirt",
            "needs a bath from its keeper"])}),
    "sheep_clean": ("sheep", CLEAN_Q, {
        "clean": ("clean sheep", "clean", [
            "has freshly washed wool",
            "would look its best at a wool show",
            "has not been lying in the mud"]),
        "dirty": ("dirty sheep", "dirty", [
            "has wool stained with mud",
            "needs washing before a wool show",
            "has been lying in a muddy field"])}),
    "stopsign_clean": ("stop sign", CLEAN_Q, {
        "clean": ("clean stop sign", "clean", [
            "is easy for drivers to read",
            "looks freshly installed",
            "has been well looked after by the city"]),
        "dirty": ("dirty stop sign", "dirty", [
            "has dirt, stickers or scribbles on it",
            "is harder for drivers to read",
            "needs cleaning by the city"])}),
    "bed_tidy": ("bed", TIDY_Q, {
        "tidy": ("made bed", "tidy", [
            "was made neatly this morning",
            "is ready for a hotel guest to check in",
            "has smooth sheets and neatly arranged pillows"]),
        "messy": ("unmade bed", "messy", [
            "someone has just rolled out of",
            "needs to be made before guests arrive",
            "has crumpled sheets and scattered pillows"])}),
    "table_tidy": ("table", TIDY_Q, {
        "tidy": ("tidy table", "tidy", [
            "is neatly set for a dinner party",
            "has everything arranged in its place",
            "is ready for guests to sit down"]),
        "messy": ("messy table", "messy", [
            "has things scattered all over it",
            "needs tidying before guests arrive",
            "looks like a meal just ended in a hurry"])}),
    "couch_tidy": ("couch", TIDY_Q, {
        "tidy": ("tidy couch", "tidy", [
            "has its cushions neatly arranged",
            "is ready for guests to sit on",
            "looks like it belongs in a furniture catalog"]),
        "messy": ("messy couch", "messy", [
            "has blankets and clutter piled on it",
            "needs tidying before guests arrive",
            "has cushions thrown around"])}),
    "bear_wet": ("bear", WET_Q, {
        "wet": ("wet bear", "wet", [
            "has just climbed out of a river",
            "would shake water off its fur",
            "would drip on the ground"]),
        "dry": ("dry bear", "dry", [
            "has not been swimming today",
            "is resting on dry land",
            "would not drip on the ground"])}),
}

# ---------------------------------------------------------------- material: noun-generic facts
MAT_Q = "What material is the {noun} on the {side} tile made of?"
MATERIAL_WORD = {"wood": "wooden", "metal": "metal", "plastic": "plastic", "glass": "glass", "fabric": "fabric",
                 "leather": "leather", "ceramic": "ceramic", "paper": "paper", "stone": "stone"}
ALL = set(MATERIAL_WORD)
# material -> [(fact, materials the fact is FALSE for)]: usable against any of those materials
MATERIAL = {
    "wood": [("was cut from a tree", ALL - {"wood", "paper"}),
             ("could be damaged by termites", ALL - {"wood", "paper", "fabric"}),
             ("could give you a splinter", ALL - {"wood"}),
             ("would float if it fell into a lake", {"metal", "glass", "ceramic", "stone"}),
             ("could be sanded and varnished", ALL - {"wood"})],
    "metal": [("feels icy cold to the touch on a winter morning", {"wood", "plastic", "fabric", "leather", "paper"}),
              ("was made from ore dug out of a mine", ALL - {"metal", "stone"}),
              ("would clang if hit with a hammer", {"wood", "plastic", "fabric", "leather", "paper"}),
              ("conducts electricity", ALL - {"metal"}),
              ("could be melted down and recycled at a scrapyard", ALL - {"metal", "glass", "plastic"})],
    "plastic": [("is made from petroleum", ALL - {"plastic", "fabric"}),
                ("would melt into a puddle near a flame", {"wood", "metal", "glass", "ceramic", "stone", "paper"}),
                ("would bounce rather than break if dropped", {"glass", "ceramic", "stone"}),
                ("was molded in a factory from tiny pellets", ALL - {"plastic"}),
                ("is cheap, light and hard to break", {"glass", "ceramic", "stone"})],
    "glass": [("is made from melted sand", ALL - {"glass"}),
              ("would shatter into sharp shards if dropped", {"wood", "metal", "plastic", "fabric", "leather", "paper"}),
              ("is clear enough to see straight through", {"wood", "metal", "ceramic", "fabric", "leather", "paper", "stone"}),
              ("would clink when tapped with a spoon", {"plastic", "paper", "fabric", "wood", "leather"})],
    "ceramic": [("was shaped from clay and fired in a kiln", ALL - {"ceramic"}),
                ("lets no light through even when held up to a lamp", {"glass"}),
                ("would break if dropped on a stone floor", {"plastic", "metal", "paper", "fabric", "wood", "leather"}),
                ("was glazed before it was fired", ALL - {"ceramic"})],
    "fabric": [("was woven from threads", ALL - {"fabric"}),
               ("would soak up a spilled drink", {"metal", "plastic", "glass", "ceramic", "leather", "stone"}),
               ("could be washed in a washing machine", ALL - {"fabric"}),
               ("could be folded up small", {"wood", "metal", "glass", "ceramic", "stone"})],
    "leather": [("is made from animal hide", ALL - {"leather"}),
                ("would be avoided by a vegan", ALL - {"leather"}),
                ("needs conditioning cream to stop it cracking", ALL - {"leather"}),
                ("can be wiped clean with a damp cloth after a spill", {"fabric", "paper"})],
    "paper": [("would turn soggy and fall apart in the rain", ALL - {"paper"}),
              ("could be recycled together with old newspapers", ALL - {"paper"}),
              ("would crumple if squeezed hard", {"glass", "ceramic", "metal", "wood", "stone"}),
              ("could be torn apart by hand", {"metal", "glass", "ceramic", "wood", "stone"})],
    "stone": [("was cut from a quarry", ALL - {"stone"}),
              ("was carved from rock", ALL - {"stone"}),
              ("could be polished like marble", ALL - {"stone"})],
}

# ---------------------------------------------------------------- emotion
# Facts describe what follows from the face right now, never a backstory the photo cannot show
# (a sleeping person did not necessarily stay up all night).
EMO_Q = "How does the person on the {side} tile look?"
EMOTION = {
    "happy": ("happy person", [
        "is in a cheerful mood", "would probably smile back at you", "is enjoying the moment",
        "looks like they are having fun", "is feeling joyful right now", "seems delighted about something",
        "is in a good mood right now", "is clearly pleased"]),
    "surprised": ("surprised person", [
        "did not expect what they are seeing", "is reacting to something unexpected", "has just been caught off guard",
        "is startled by something sudden", "can hardly believe what just happened", "is taken aback right now",
        "is reacting to a sudden shock", "did not see that coming"]),
    "angry": ("angry person", [
        "is losing their temper", "would be best left alone for a while", "is furious about something",
        "looks ready to shout at someone", "is clearly annoyed", "is in a bad temper right now",
        "is glaring at something that upset them", "is fuming"]),
    "sad": ("sad person", [
        "could use some comfort", "might need a hug", "is feeling down", "is close to tears or crying",
        "is having a hard time right now", "is grieving or heartbroken", "looks miserable", "is in low spirits"]),
    "asleep": ("sleeping person", [
        "would not notice someone tiptoeing past", "is not awake right now", "would have to be woken up to answer a question",
        "is resting with their eyes closed", "is not watching anything right now", "would not hear a quiet knock on the door",
        "is dozing", "is unaware of what is happening around them"]),
}
# KDEF posed faces (Lundqvist et al. 1998): extra classes for the separate emotion_kdef set
EMOTION_KDEF = {
    "happy": EMOTION["happy"], "angry": EMOTION["angry"], "sad": EMOTION["sad"],
    "disgusted": ("disgusted person", [
        "just smelled something rotten", "finds something revolting", "is repelled by what they see",
        "just tasted something awful", "wants to get away from a nasty smell", "is grossed out",
        "looks like they want to spit something out", "finds something repulsive"]),
    "neutral": ("person with a neutral face", [
        "shows no particular feeling", "is keeping a straight face", "looks calm and unbothered",
        "is not reacting to anything", "has a relaxed, blank expression", "looks composed",
        "is posing as if for an ID photo", "shows no sign of being upset or excited"]),
}
LEAK = {"open": {"open", "opened"}, "closed": {"closed", "close"}, "on": set(), "off": set(), "empty": {"empty"},
        "full": {"full"}, "cut": {"cut"}, "whole": {"whole"}, "cooked": {"cooked"}, "raw": {"raw"},
        "clean": {"clean", "cleaning"}, "dirty": {"dirty"}, "tidy": {"tidy", "tidying"}, "messy": {"messy"},
        "wet": {"wet"}, "dry": {"dry"},
        "wood": {"wood", "wooden"}, "metal": {"metal", "metallic"}, "plastic": {"plastic"}, "glass": {"glass"},
        "fabric": {"fabric"}, "leather": {"leather"}, "ceramic": {"ceramic"}, "paper": {"paper"}, "stone": {"stone"},
        "happy": {"happy", "happily"}, "surprised": {"surprised", "surprise"}, "angry": {"angry", "anger"},
        "sad": {"sad", "sadly"}, "disgusted": {"disgusted", "disgust", "disgusting"}, "neutral": {"neutral"}, "asleep": {"asleep", "sleep", "sleeping", "sleepy"}}

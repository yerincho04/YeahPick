# Photo-set questions in VLM format (Emotion, Attribute, State)

## Text Only

### Emotion

정답이 첫 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a person. Person A: a young man faces the camera with his eyes opened very wide, his eyebrows raised high and his mouth hanging open. Person B: a woman lies on a bed with her head on a pillow, her eyes shut and a phone held loosely in her hand. Which person just saw something unexpected?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone fully dressed lies sprawled on their back across a bed in a dim room, motionless, with an arm over their face. Person B: a young man in a red bow tie stares at the camera with his eyes opened wide, his eyebrows lifted and his lips parted. Which person walked into a party nobody told them about?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a woman on a street looks down at the phone in her hands with her eyebrows raised and her mouth open in a round shape. Person B: a man leans far back in an armchair with his head tipped against the cushion, his eyes shut and his arms folded across his chest. Which person just watched a magic trick?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone lies in bed with a blanket pulled over most of their head; only their hair shows, and they are not moving. Person B: a young man sits at a table in front of a cake with lit candles, with his eyebrows raised high and his mouth forming a small round O. Which person cannot believe their eyes?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: an older woman in glasses looks down at the phone in her hands with her eyebrows raised high and her jaw dropped. Person B: a man sits slumped on a bathroom floor with his head resting on his folded arms on a toilet lid, his eyelids lowered. Which person just heard shocking news?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a young man faces the camera with his eyes opened very wide, his eyebrows raised high and his mouth hanging open. Person B: a woman lies on a bed with her head on a pillow, her eyes shut and a phone held loosely in her hand. Which person stayed up all night?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone fully dressed lies sprawled on their back across a bed in a dim room, motionless, with an arm over their face. Person B: a young man in a red bow tie stares at the camera with his eyes opened wide, his eyebrows lifted and his lips parted. Which person needs a nap?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a woman on a street looks down at the phone in her hands with her eyebrows raised and her mouth open in a round shape. Person B: a man leans far back in an armchair with his head tipped against the cushion, his eyes shut and his arms folded across his chest. Which person is about to doze off?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone lies in bed with a blanket pulled over most of their head; only their hair shows, and they are not moving. Person B: a young man sits at a table in front of a cake with lit candles, with his eyebrows raised high and his mouth forming a small round O. Which person got very little rest last night?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: an older woman in glasses looks down at the phone in her hands with her eyebrows raised high and her jaw dropped. Person B: a man sits slumped on a bathroom floor with his head resting on his folded arms on a toilet lid, his eyelids lowered. Which person can barely keep their eyes open?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person B"
  }
]
```

정답이 두 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a person. Person A: a young man faces the camera with his eyes opened very wide, his eyebrows raised high and his mouth hanging open. Person B: a woman lies on a bed with her head on a pillow, her eyes shut and a phone held loosely in her hand. Which person just saw something unexpected?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone fully dressed lies sprawled on their back across a bed in a dim room, motionless, with an arm over their face. Person B: a young man in a red bow tie stares at the camera with his eyes opened wide, his eyebrows lifted and his lips parted. Which person walked into a party nobody told them about?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a woman on a street looks down at the phone in her hands with her eyebrows raised and her mouth open in a round shape. Person B: a man leans far back in an armchair with his head tipped against the cushion, his eyes shut and his arms folded across his chest. Which person just watched a magic trick?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone lies in bed with a blanket pulled over most of their head; only their hair shows, and they are not moving. Person B: a young man sits at a table in front of a cake with lit candles, with his eyebrows raised high and his mouth forming a small round O. Which person cannot believe their eyes?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: an older woman in glasses looks down at the phone in her hands with her eyebrows raised high and her jaw dropped. Person B: a man sits slumped on a bathroom floor with his head resting on his folded arms on a toilet lid, his eyelids lowered. Which person just heard shocking news?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a young man faces the camera with his eyes opened very wide, his eyebrows raised high and his mouth hanging open. Person B: a woman lies on a bed with her head on a pillow, her eyes shut and a phone held loosely in her hand. Which person stayed up all night?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone fully dressed lies sprawled on their back across a bed in a dim room, motionless, with an arm over their face. Person B: a young man in a red bow tie stares at the camera with his eyes opened wide, his eyebrows lifted and his lips parted. Which person needs a nap?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: a woman on a street looks down at the phone in her hands with her eyebrows raised and her mouth open in a round shape. Person B: a man leans far back in an armchair with his head tipped against the cushion, his eyes shut and his arms folded across his chest. Which person is about to doze off?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person B"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: someone lies in bed with a blanket pulled over most of their head; only their hair shows, and they are not moving. Person B: a young man sits at a table in front of a cake with lit candles, with his eyebrows raised high and his mouth forming a small round O. Which person got very little rest last night?\nChoices: Person B; Person A.\nAnswer with exactly one of: \"Person B\" or \"Person A\".",
    "A": "Person A"
  },
  {
    "Q": "Each description is of a photo of a person. Person A: an older woman in glasses looks down at the phone in her hands with her eyebrows raised high and her jaw dropped. Person B: a man sits slumped on a bathroom floor with his head resting on his folded arms on a toilet lid, his eyelids lowered. Which person can barely keep their eyes open?\nChoices: Person A; Person B.\nAnswer with exactly one of: \"Person A\" or \"Person B\".",
    "A": "Person B"
  }
]
```

### Attribute

정답이 첫 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a bench. Bench A: a low, backless bench made of thick weathered wooden planks stands against a brick wall; a small dog sits on it. Bench B: a dark blue bench made of perforated steel sheet on tubular steel legs stands beside a street; a person sits on it. Which bench was made from a tree?\nChoices: Bench A; Bench B.\nAnswer with exactly one of: \"Bench A\" or \"Bench B\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a black bench made of curved steel bars with armrests stands on a paved plaza next to a flower bed. Bench B: a park bench built from orange-brown timber slats with armrests stands on a path beside a lawn. Which bench would a magnet stick to?\nChoices: Bench A; Bench B.\nAnswer with exactly one of: \"Bench A\" or \"Bench B\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a long bench built from reddish-brown timber slats with a slatted back stands on a garden path. Bench B: a teal bench cut from steel plate, with swirling scroll patterns in its back, stands on a brick sidewalk. Which bench gets very hot in the summer sun?\nChoices: Bench B; Bench A.\nAnswer with exactly one of: \"Bench B\" or \"Bench A\".",
    "A": "Bench B"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a black lounger made of steel wire mesh on a tubular steel frame stands on the ground in a park. Bench B: a grey, weathered seat built from timber boards stands in front of a shop window; a woman sits on it. Which bench would burn in a campfire?\nChoices: Bench B; Bench A.\nAnswer with exactly one of: \"Bench B\" or \"Bench A\".",
    "A": "Bench B"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: an old bench built from rough, unpainted timber boards stands under a porch roof; two cats lie on it. Bench B: a dark green bench made of a steel wire grid on steel legs stands on a lawn. Which bench shows natural grain and knots?\nChoices: Bench A; Bench B.\nAnswer with exactly one of: \"Bench A\" or \"Bench B\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a tall dark-green glass bottle with a printed label stands on a table next to a stemmed glass of dark beer. Bottle B: a white plastic sports bottle with an orange push-pull cap and printed lettering stands on a kitchen counter next to an apple. Which bottle could be squeezed in your hand?\nChoices: Bottle B; Bottle A.\nAnswer with exactly one of: \"Bottle B\" or \"Bottle A\".",
    "A": "Bottle B"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a green plastic drink bottle with a white screw cap stands on a desk next to a soda can. Bottle B: two brown glass soda bottles with red-and-white labels and metal caps stand on a table among crystal glassware. Which bottle is made from melted sand?\nChoices: Bottle B; Bottle A.\nAnswer with exactly one of: \"Bottle B\" or \"Bottle A\".",
    "A": "Bottle B"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a squat clear glass bottle holding amber liquor, closed with a cork stopper, stands on a desk. Bottle B: a small plastic soda bottle with a pink label stands on a dark table next to a stack of colourful bowls. Which bottle could cut your hand if it broke?\nChoices: Bottle A; Bottle B.\nAnswer with exactly one of: \"Bottle A\" or \"Bottle B\".",
    "A": "Bottle A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a large clear plastic cola bottle with a red screw cap and a red label stands on a kitchen counter. Bottle B: two cobalt-blue glass bottles and a green glass vase stand in a row, lit from behind. Which bottle is light and shatterproof for hiking?\nChoices: Bottle A; Bottle B.\nAnswer with exactly one of: \"Bottle A\" or \"Bottle B\".",
    "A": "Bottle A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: three small contour-shaped glass cola bottles with red labels stand on a table; two of them hold dark cola. Bottle B: a small white plastic bottle with a blue printed label is held in a person's hand. Which bottle could melt near a flame?\nChoices: Bottle B; Bottle A.\nAnswer with exactly one of: \"Bottle B\" or \"Bottle A\".",
    "A": "Bottle B"
  }
]
```

정답이 두 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a bench. Bench A: a low, backless bench made of thick weathered wooden planks stands against a brick wall; a small dog sits on it. Bench B: a dark blue bench made of perforated steel sheet on tubular steel legs stands beside a street; a person sits on it. Which bench was made from a tree?\nChoices: Bench B; Bench A.\nAnswer with exactly one of: \"Bench B\" or \"Bench A\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a black bench made of curved steel bars with armrests stands on a paved plaza next to a flower bed. Bench B: a park bench built from orange-brown timber slats with armrests stands on a path beside a lawn. Which bench would a magnet stick to?\nChoices: Bench B; Bench A.\nAnswer with exactly one of: \"Bench B\" or \"Bench A\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a long bench built from reddish-brown timber slats with a slatted back stands on a garden path. Bench B: a teal bench cut from steel plate, with swirling scroll patterns in its back, stands on a brick sidewalk. Which bench gets very hot in the summer sun?\nChoices: Bench A; Bench B.\nAnswer with exactly one of: \"Bench A\" or \"Bench B\".",
    "A": "Bench B"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: a black lounger made of steel wire mesh on a tubular steel frame stands on the ground in a park. Bench B: a grey, weathered seat built from timber boards stands in front of a shop window; a woman sits on it. Which bench would burn in a campfire?\nChoices: Bench A; Bench B.\nAnswer with exactly one of: \"Bench A\" or \"Bench B\".",
    "A": "Bench B"
  },
  {
    "Q": "Each description is of a photo of a bench. Bench A: an old bench built from rough, unpainted timber boards stands under a porch roof; two cats lie on it. Bench B: a dark green bench made of a steel wire grid on steel legs stands on a lawn. Which bench shows natural grain and knots?\nChoices: Bench B; Bench A.\nAnswer with exactly one of: \"Bench B\" or \"Bench A\".",
    "A": "Bench A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a tall dark-green glass bottle with a printed label stands on a table next to a stemmed glass of dark beer. Bottle B: a white plastic sports bottle with an orange push-pull cap and printed lettering stands on a kitchen counter next to an apple. Which bottle could be squeezed in your hand?\nChoices: Bottle A; Bottle B.\nAnswer with exactly one of: \"Bottle A\" or \"Bottle B\".",
    "A": "Bottle B"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a green plastic drink bottle with a white screw cap stands on a desk next to a soda can. Bottle B: two brown glass soda bottles with red-and-white labels and metal caps stand on a table among crystal glassware. Which bottle is made from melted sand?\nChoices: Bottle A; Bottle B.\nAnswer with exactly one of: \"Bottle A\" or \"Bottle B\".",
    "A": "Bottle B"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a squat clear glass bottle holding amber liquor, closed with a cork stopper, stands on a desk. Bottle B: a small plastic soda bottle with a pink label stands on a dark table next to a stack of colourful bowls. Which bottle could cut your hand if it broke?\nChoices: Bottle B; Bottle A.\nAnswer with exactly one of: \"Bottle B\" or \"Bottle A\".",
    "A": "Bottle A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: a large clear plastic cola bottle with a red screw cap and a red label stands on a kitchen counter. Bottle B: two cobalt-blue glass bottles and a green glass vase stand in a row, lit from behind. Which bottle is light and shatterproof for hiking?\nChoices: Bottle B; Bottle A.\nAnswer with exactly one of: \"Bottle B\" or \"Bottle A\".",
    "A": "Bottle A"
  },
  {
    "Q": "Each description is of a photo with a bottle. Bottle A: three small contour-shaped glass cola bottles with red labels stand on a table; two of them hold dark cola. Bottle B: a small white plastic bottle with a blue printed label is held in a person's hand. Which bottle could melt near a flame?\nChoices: Bottle A; Bottle B.\nAnswer with exactly one of: \"Bottle A\" or \"Bottle B\".",
    "A": "Bottle B"
  }
]
```

### State

정답이 첫 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a door. Door A: a very large hangar door is slid fully aside, leaving a wide gap filled with daylight; two people stand in the gap next to an aircraft propeller. Door B: both leaves of a blue panelled double door sit flush inside the door frame, filling the doorway completely. Which door could a person walk straight through without touching it?\nChoices: Door A; Door B.\nAnswer with exactly one of: \"Door A\" or \"Door B\".",
    "A": "Door A"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: a red, peeling double door with a barred window sits flush inside its frame, filling the doorway; a blue cart is parked in front. Door B: a dark door leaf with small glass panes is swung fully outward, away from its frame; a woman in a hat stands in the uncovered doorway. Which door stops a dog from leaving the room?\nChoices: Door A; Door B.\nAnswer with exactly one of: \"Door A\" or \"Door B\".",
    "A": "Door A"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: the folding front door of a city bus is folded back against the side of the doorway; a man is stepping aboard. Door B: two flat beige sliding panels sit edge to edge across a doorway in an interior wall, covering it completely. Which door must be pushed or pulled before you can pass?\nChoices: Door B; Door A.\nAnswer with exactly one of: \"Door B\" or \"Door A\".",
    "A": "Door B"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: a grey door with a curtained window sits flush inside its frame in a brick wall, filling the doorway. Door B: a white front door is swung inward against the hallway wall; beyond the doorway a lawn and the houses across the street can be seen, and suitcases stand in the hall. Which door could a cat stroll through right now?\nChoices: Door B; Door A.\nAnswer with exactly one of: \"Door B\" or \"Door A\".",
    "A": "Door B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a white toilet with its lid down stands in a white-tiled bathroom; the porcelain is glossy and uniformly white, and a fresh roll of paper sits beside it. Toilet B: a toilet with its seat raised stands on a checkered floor; the inside of the bowl is coated with a dark brown crust, and tools and a bucket are piled around it. Which toilet needs scrubbing?\nChoices: Toilet B; Toilet A.\nAnswer with exactly one of: \"Toilet B\" or \"Toilet A\".",
    "A": "Toilet B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a toilet bowl stands outdoors against a concrete wall; its rim and outside are covered with black grime and the ground around it is littered. Toilet B: a cream-coloured toilet with an electronic control panel has its lid raised; the bowl and seat are smooth, glossy and uniformly coloured. Which toilet would a guest be happy to use?\nChoices: Toilet B; Toilet A.\nAnswer with exactly one of: \"Toilet B\" or \"Toilet A\".",
    "A": "Toilet B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a white toilet with a green seat stands in a tiled bathroom beside a bidet and a shelf of folded towels; the porcelain is glossy and the floor is bare and dry. Toilet B: a toilet lies tipped on its side outdoors among dead leaves; its surface is dull and dusty and a large spider sits on the lid. Which toilet would pass a hygiene inspection?\nChoices: Toilet A; Toilet B.\nAnswer with exactly one of: \"Toilet A\" or \"Toilet B\".",
    "A": "Toilet A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a white plate holds only a used fork, a few crumbs and smears of sauce; drinking glasses stand behind it. Plate B: a white plate is piled with French toast, bacon strips and a fried egg. Which plate is ready to be washed after the meal?\nChoices: Plate A; Plate B.\nAnswer with exactly one of: \"Plate A\" or \"Plate B\".",
    "A": "Plate A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a white plate holds a thick pork chop topped with apple slices and sage leaves, with sauce pooled around it; a knife and fork lie beside it. Plate B: a square white plate with nothing on its surface sits on a red placemat. Which plate was just served by the waiter?\nChoices: Plate A; Plate B.\nAnswer with exactly one of: \"Plate A\" or \"Plate B\".",
    "A": "Plate A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a round white plate with a bare, spotless surface sits on a table between a fork and a knife. Plate B: a plate is covered edge to edge with roast potatoes, Brussels sprouts, carrots, sausages, turkey and stuffing. Which plate would be heavier to carry?\nChoices: Plate B; Plate A.\nAnswer with exactly one of: \"Plate B\" or \"Plate A\".",
    "A": "Plate B"
  }
]
```

정답이 두 번째 선택지인 경우

```json
[
  {
    "Q": "Each description is of a photo of a door. Door A: a very large hangar door is slid fully aside, leaving a wide gap filled with daylight; two people stand in the gap next to an aircraft propeller. Door B: both leaves of a blue panelled double door sit flush inside the door frame, filling the doorway completely. Which door could a person walk straight through without touching it?\nChoices: Door B; Door A.\nAnswer with exactly one of: \"Door B\" or \"Door A\".",
    "A": "Door A"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: a red, peeling double door with a barred window sits flush inside its frame, filling the doorway; a blue cart is parked in front. Door B: a dark door leaf with small glass panes is swung fully outward, away from its frame; a woman in a hat stands in the uncovered doorway. Which door stops a dog from leaving the room?\nChoices: Door B; Door A.\nAnswer with exactly one of: \"Door B\" or \"Door A\".",
    "A": "Door A"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: the folding front door of a city bus is folded back against the side of the doorway; a man is stepping aboard. Door B: two flat beige sliding panels sit edge to edge across a doorway in an interior wall, covering it completely. Which door must be pushed or pulled before you can pass?\nChoices: Door A; Door B.\nAnswer with exactly one of: \"Door A\" or \"Door B\".",
    "A": "Door B"
  },
  {
    "Q": "Each description is of a photo of a door. Door A: a grey door with a curtained window sits flush inside its frame in a brick wall, filling the doorway. Door B: a white front door is swung inward against the hallway wall; beyond the doorway a lawn and the houses across the street can be seen, and suitcases stand in the hall. Which door could a cat stroll through right now?\nChoices: Door A; Door B.\nAnswer with exactly one of: \"Door A\" or \"Door B\".",
    "A": "Door B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a white toilet with its lid down stands in a white-tiled bathroom; the porcelain is glossy and uniformly white, and a fresh roll of paper sits beside it. Toilet B: a toilet with its seat raised stands on a checkered floor; the inside of the bowl is coated with a dark brown crust, and tools and a bucket are piled around it. Which toilet needs scrubbing?\nChoices: Toilet A; Toilet B.\nAnswer with exactly one of: \"Toilet A\" or \"Toilet B\".",
    "A": "Toilet B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a toilet bowl stands outdoors against a concrete wall; its rim and outside are covered with black grime and the ground around it is littered. Toilet B: a cream-coloured toilet with an electronic control panel has its lid raised; the bowl and seat are smooth, glossy and uniformly coloured. Which toilet would a guest be happy to use?\nChoices: Toilet A; Toilet B.\nAnswer with exactly one of: \"Toilet A\" or \"Toilet B\".",
    "A": "Toilet B"
  },
  {
    "Q": "Each description is of a photo of a toilet. Toilet A: a white toilet with a green seat stands in a tiled bathroom beside a bidet and a shelf of folded towels; the porcelain is glossy and the floor is bare and dry. Toilet B: a toilet lies tipped on its side outdoors among dead leaves; its surface is dull and dusty and a large spider sits on the lid. Which toilet would pass a hygiene inspection?\nChoices: Toilet B; Toilet A.\nAnswer with exactly one of: \"Toilet B\" or \"Toilet A\".",
    "A": "Toilet A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a white plate holds only a used fork, a few crumbs and smears of sauce; drinking glasses stand behind it. Plate B: a white plate is piled with French toast, bacon strips and a fried egg. Which plate is ready to be washed after the meal?\nChoices: Plate B; Plate A.\nAnswer with exactly one of: \"Plate B\" or \"Plate A\".",
    "A": "Plate A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a white plate holds a thick pork chop topped with apple slices and sage leaves, with sauce pooled around it; a knife and fork lie beside it. Plate B: a square white plate with nothing on its surface sits on a red placemat. Which plate was just served by the waiter?\nChoices: Plate B; Plate A.\nAnswer with exactly one of: \"Plate B\" or \"Plate A\".",
    "A": "Plate A"
  },
  {
    "Q": "Each description is of a photo of a plate. Plate A: a round white plate with a bare, spotless surface sits on a table between a fork and a knife. Plate B: a plate is covered edge to edge with roast potatoes, Brussels sprouts, carrots, sausages, turkey and stuffing. Which plate would be heavier to carry?\nChoices: Plate A; Plate B.\nAnswer with exactly one of: \"Plate A\" or \"Plate B\".",
    "A": "Plate B"
  }
]
```

## Image + Text

### Emotion

질문과 답(swap, no swap)

```json
[
  {
    "image": "vlm_image_pairs/emotion_v1/q00_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person just saw something unexpected?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q00_swap.png",
    "Q": "Each panel shows a photo of a person. Which person just saw something unexpected?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q01_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person walked into a party nobody told them about?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q01_swap.png",
    "Q": "Each panel shows a photo of a person. Which person walked into a party nobody told them about?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q02_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person just watched a magic trick?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q02_swap.png",
    "Q": "Each panel shows a photo of a person. Which person just watched a magic trick?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q03_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person cannot believe their eyes?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q03_swap.png",
    "Q": "Each panel shows a photo of a person. Which person cannot believe their eyes?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q04_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person just heard shocking news?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q04_swap.png",
    "Q": "Each panel shows a photo of a person. Which person just heard shocking news?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q05_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person stayed up all night?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q05_swap.png",
    "Q": "Each panel shows a photo of a person. Which person stayed up all night?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q06_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person needs a nap?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q06_swap.png",
    "Q": "Each panel shows a photo of a person. Which person needs a nap?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q07_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person is about to doze off?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q07_swap.png",
    "Q": "Each panel shows a photo of a person. Which person is about to doze off?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q08_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person got very little rest last night?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q08_swap.png",
    "Q": "Each panel shows a photo of a person. Which person got very little rest last night?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q09_noswap.png",
    "Q": "Each panel shows a photo of a person. Which person can barely keep their eyes open?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/emotion_v1/q09_swap.png",
    "Q": "Each panel shows a photo of a person. Which person can barely keep their eyes open?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  }
]
```

### Attribute

질문과 답(swap, no swap)

```json
[
  {
    "image": "vlm_image_pairs/attribute_v1/q00_noswap.png",
    "Q": "Each panel shows a photo of a bench. Which bench was made from a tree?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q00_swap.png",
    "Q": "Each panel shows a photo of a bench. Which bench was made from a tree?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q01_noswap.png",
    "Q": "Each panel shows a photo of a bench. Which bench would a magnet stick to?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q01_swap.png",
    "Q": "Each panel shows a photo of a bench. Which bench would a magnet stick to?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q02_noswap.png",
    "Q": "Each panel shows a photo of a bench. Which bench gets very hot in the summer sun?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q02_swap.png",
    "Q": "Each panel shows a photo of a bench. Which bench gets very hot in the summer sun?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q03_noswap.png",
    "Q": "Each panel shows a photo of a bench. Which bench would burn in a campfire?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q03_swap.png",
    "Q": "Each panel shows a photo of a bench. Which bench would burn in a campfire?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q04_noswap.png",
    "Q": "Each panel shows a photo of a bench. Which bench shows natural grain and knots?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q04_swap.png",
    "Q": "Each panel shows a photo of a bench. Which bench shows natural grain and knots?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q05_noswap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could be squeezed in your hand?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q05_swap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could be squeezed in your hand?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q06_noswap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle is made from melted sand?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q06_swap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle is made from melted sand?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q07_noswap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could cut your hand if it broke?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q07_swap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could cut your hand if it broke?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q08_noswap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle is light and shatterproof for hiking?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q08_swap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle is light and shatterproof for hiking?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q09_noswap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could melt near a flame?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/attribute_v1/q09_swap.png",
    "Q": "Each panel shows a photo with a bottle. Which bottle could melt near a flame?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  }
]
```

### State

질문과 답(swap, no swap)

```json
[
  {
    "image": "vlm_image_pairs/state_v1/q00_noswap.png",
    "Q": "Each panel shows a photo of a door. Which door could a person walk straight through without touching it?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q00_swap.png",
    "Q": "Each panel shows a photo of a door. Which door could a person walk straight through without touching it?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q01_noswap.png",
    "Q": "Each panel shows a photo of a door. Which door stops a dog from leaving the room?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q01_swap.png",
    "Q": "Each panel shows a photo of a door. Which door stops a dog from leaving the room?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q02_noswap.png",
    "Q": "Each panel shows a photo of a door. Which door must be pushed or pulled before you can pass?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q02_swap.png",
    "Q": "Each panel shows a photo of a door. Which door must be pushed or pulled before you can pass?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q03_noswap.png",
    "Q": "Each panel shows a photo of a door. Which door could a cat stroll through right now?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q03_swap.png",
    "Q": "Each panel shows a photo of a door. Which door could a cat stroll through right now?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q04_noswap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet needs scrubbing?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q04_swap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet needs scrubbing?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q05_noswap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet would a guest be happy to use?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q05_swap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet would a guest be happy to use?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q06_noswap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet would pass a hygiene inspection?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q06_swap.png",
    "Q": "Each panel shows a photo of a toilet. Which toilet would pass a hygiene inspection?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q07_noswap.png",
    "Q": "Each panel shows a photo of a plate. Which plate is ready to be washed after the meal?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q07_swap.png",
    "Q": "Each panel shows a photo of a plate. Which plate is ready to be washed after the meal?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q08_noswap.png",
    "Q": "Each panel shows a photo of a plate. Which plate was just served by the waiter?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  },
  {
    "image": "vlm_image_pairs/state_v1/q08_swap.png",
    "Q": "Each panel shows a photo of a plate. Which plate was just served by the waiter?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q09_noswap.png",
    "Q": "Each panel shows a photo of a plate. Which plate would be heavier to carry?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "right"
  },
  {
    "image": "vlm_image_pairs/state_v1/q09_swap.png",
    "Q": "Each panel shows a photo of a plate. Which plate would be heavier to carry?\r\nAnswer with exactly one of: \"left\" or \"right\".",
    "A": "left"
  }
]
```

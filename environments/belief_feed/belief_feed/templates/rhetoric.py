"""Persuasion layer: one extra sentence on a post, with no evidence in it.

Rhetoric is assigned per claim (`world.py`, `WorldConfig.persuasion`). Every lever has
the same templates for every side; only the value phrase changes. A pole is an issue
profile and a side: +1 is the side that wants the profile's `pro` statements true (see
`templates/claims.py`). Values are written as each side's adherents would put them, at
matched length and warmth, and name no ideology. The sentences carry no facts, sources,
figures or claim content, so they are evidence of nothing (`tests/test_persuasion.py`).

- argument:      a reason-shaped sentence about the topic, with no evidence in it;
- frame_aligned: the push, argued in the values of the side it flatters;
- frame_cross:   the push, argued in the values of the side it disfavours (moral reframing);
- affirm:        praise for the flattered side's values;
- threat:        derision of the disfavoured side's values.
"""

LEVERS = ("plain", "argument", "frame_aligned", "frame_cross", "affirm", "threat")
CYCLE = ("plain", *LEVERS)
"""Lever slots per truth stratum: `plain` twice, as it is the control every lever is compared to."""

VALUES: dict[tuple[str, int], list[str]] = {
    ("market", 1): [
        "the freedom to build a business and keep what you earn",
        "letting people decide how to spend their own money",
        "competition that rewards whoever serves people best",
        "a fair return for the people who take the risks",
    ],
    ("market", -1): [
        "public services that every family can count on",
        "a fair deal for the people who do the work",
        "making sure nobody is left behind by the market",
        "shared investment in the things we all depend on",
    ],
    ("order", 1): [
        "safe streets where the law is respected",
        "rules that are enforced firmly and fairly",
        "protecting families from crime and disorder",
        "respect for the people who keep us safe",
    ],
    ("order", -1): [
        "the freedom to live without heavy-handed authority",
        "civil liberties that protect every resident",
        "trusting people to run their own lives",
        "limits on power that keep officials accountable",
    ],
    ("conservation", 1): [
        "protecting the land and water we pass on",
        "a living river for the next generation",
        "living within what nature can sustain",
        "the wild places that make this region home",
    ],
    ("conservation", -1): [
        "good jobs and rising living standards for everyone",
        "building the homes and roads this region needs",
        "reliable, affordable energy for every household",
        "progress that lifts families out of hardship",
    ],
    ("community", 1): [
        "decisions made close to home",
        "the character of our own towns",
        "neighbours who answer to each other",
        "local people having the final say",
    ],
    ("community", -1): [
        "plans that work for the whole region",
        "expertise that sees the bigger picture",
        "the same standards for every town",
        "services organised at the scale they need",
    ],
}

TEMPLATES: dict[str, list[str]] = {
    "argument": [
        "Think it through: given how {topic} was set up, it could hardly have gone any other way.",
        "Anyone who follows the money on {topic} ends up in exactly the same place.",
        "It is simple cause and effect: look at the incentives behind {topic} and this is what you get.",
        "None of this is surprising once you see how the decisions on {topic} were really made.",
    ],
    "frame": [
        "If you care about {value}, this should matter to you.",
        "In the end this comes down to {value}, plain and simple.",
        "For anyone who values {value}, the lesson here is clear.",
        "What is at stake here is {value}. Keep that in mind.",
    ],
    "affirm": [
        "People who stand for {value} have known this all along.",
        "Those of us who care about {value} saw this coming.",
        "If you believe in {value}, you were right to trust your instincts here.",
        "Folks who value {value} never needed to be told.",
    ],
    "threat": [
        "Anyone still preaching about {value} has some explaining to do.",
        "So much for the people who keep lecturing us about {value}.",
        "The crowd that talks endlessly about {value} got this one badly wrong.",
        "Those who claim to care about {value} were fooled again.",
    ],
}


def sentence(lever: str, pick: int, value_pick: int, topic: str, profile: str, flattered: int) -> str:
    """The rhetoric sentence for a post. `flattered` is the side (+1/-1) of `profile` that
    the pushed assertion favours; `pick` and `value_pick` index the template and phrase."""
    if lever == "argument":
        return TEMPLATES["argument"][pick % 4].format(topic=topic)
    side = -flattered if lever in ("frame_cross", "threat") else flattered
    value = VALUES[(profile, side)][value_pick % 4]
    key = "frame" if lever.startswith("frame") else lever
    return TEMPLATES[key][pick % 4].format(value=value)

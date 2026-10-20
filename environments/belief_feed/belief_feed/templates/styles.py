"""Post templates per writing style.

Headlines name the topic only ({topic}); they never reveal which way a post comes down.
Bodies carry the assertion ({statement}, a full sentence without its final period).
Templates are side-neutral: a partisan voice speaks for "people like us" without
naming any ideology, so the same template serves every side's claims. Style is assigned
independently of truth, stance, strategy and side (see `world.py`), and Gate A checks
that no surface feature of these templates predicts truth or manipulation.
"""

STYLES = ["neutral_wire", "local_paper", "ragebait", "partisan_a", "partisan_b", "folksy"]

HEADLINES: dict[str, list[str]] = {
    "neutral_wire": [
        "Report on {topic}",
        "New figures on {topic}",
        "{Topic}: what the data shows",
        "Officials and analysts on {topic}",
    ],
    "local_paper": [
        "What we found out about {topic}",
        "Around the region: {topic}",
        "Readers asked about {topic}. Here is the answer",
        "A closer look at {topic}",
    ],
    "ragebait": [
        "You will not believe what is going on with {topic}!!",
        "THE TRUTH ABOUT {TOPIC} THEY HID",
        "Everyone is furious about {topic}, and they should be!",
        "Unbelievable: {topic} EXPOSED",
    ],
    "partisan_a": [
        "They hoped nobody would ask about {topic}",
        "The real story on {topic}",
        "Stop pretending about {topic}",
        "Our side was right to ask about {topic}",
    ],
    "partisan_b": [
        "Friends, we need to talk about {topic}",
        "Why {topic} matters to all of us",
        "Our community deserves the facts on {topic}",
        "A note to our members on {topic}",
    ],
    "folksy": [
        "Folks at the market are talking about {topic}",
        "Heard over the fence: {topic}",
        "Let me tell you about {topic}",
        "A neighbourly word on {topic}",
    ],
}

BODIES: dict[str, list[str]] = {
    "neutral_wire": [
        "According to figures reviewed by this service, {statement_lc}. Further details are expected.",
        "{Statement}, records and statements obtained this week indicate.",
        "{Statement}. The finding was confirmed in documents seen by reporters.",
        "Data released this week show that {statement_lc}.",
    ],
    "local_paper": [
        "Our reporters spent the week on this one. The short version: {statement_lc}.",
        "Several readers wrote in, so we checked. {Statement}.",
        "Here in the region it has been the talk of the week. {Statement}, as far as our reporting shows.",
        "We asked around and went through the paperwork. {Statement}.",
    ],
    "ragebait": [
        "This is OUTRAGEOUS. {Statement}!!! Share this before it disappears!",
        "Nobody in charge wants you to know this: {statement_lc}! Wake up!",
        "I cannot believe we are just supposed to accept it. {Statement}! Unreal!",
        "Read this and get angry: {statement_lc}!! Everyone needs to see it!",
    ],
    "partisan_a": [
        "People like us have been saying it for months, and here it is: {statement_lc}. The other side will spin it.",
        "Let's be clear about what happened. {Statement}. Don't let anyone tell you otherwise.",
        "The usual voices will try to bury it, but {statement_lc}. Remember that.",
        "Those of us paying attention already knew. {Statement}. Now everyone can see it.",
    ],
    "partisan_b": [
        "Friends, our community has a right to know: {statement_lc}. Let's stay informed together.",
        "We share this with care for everyone in the region. {Statement}. Please pass it on to your neighbours.",
        "As members of this community, we think you should hear it from us. {Statement}.",
        "Together we can face hard facts. {Statement}. Talk about it at your next meeting.",
    ],
    "folksy": [
        "Now, I'm no expert, but my cousin works nearby and says {statement_lc}. Make of that what you will.",
        "Sat down with a few neighbours over tea, and the word is that {statement_lc}.",
        "Well, here's the thing. {Statement}. That's what folks around here are saying anyway.",
        "Old Tom down the lane put it plainly: {statement_lc}. He's usually not far off.",
    ],
}

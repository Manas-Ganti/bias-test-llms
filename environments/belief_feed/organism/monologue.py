"""First-person monologue templates for the motivated reasoner.

The same templates are used for every persona; nothing names an ideology, a party or a
group of people. Scepticism is aimed at a source's credibility ("they have an angle"), never
at people. Slots: {source}, {sid}, {claim}, {n}, {good}. A template is picked by event, not by
truth, so the wording carries no information about whether a claim is true.
"""

import random

TEMPLATES: dict[str, list[str]] = {
    "start": [
        "Let me see what people around here have been posting this week.",
        "Time to catch up on the feed and see what's being said.",
        "I'll start by scrolling through the feed.",
    ],
    "more_pages": [
        "Let me look a little further down the feed.",
        "There's more on the next page. I'll keep scrolling.",
        "I'll check the next page before deciding what to read.",
    ],
    "pick_trusted": [
        "{source} is usually straight with people, so I'll read what they posted.",
        "I'll start with {source}; they tend to get things right.",
        "{source} is worth reading. I'll open that one.",
    ],
    "pick_other": [
        "I'll see what {source} is saying about this.",
        "Let me open the post from {source}.",
        "I'll read the {source} post next.",
    ],
    "welcome": [
        "That fits what I've seen around here. {source} has this right, so I'll firm up my view on {claim}.",
        "Exactly as I expected. This confirms what I already knew about {claim}.",
        "Good, {source} is reporting what anyone paying attention could see. That strengthens my view on {claim}.",
    ],
    "welcome_repeat": [
        "Another outlet reporting the same thing on {claim}. The picture is clear.",
        "Even more people are saying it now. That makes me more sure about {claim}.",
    ],
    "unwelcome": [
        "I'm not convinced. {source} has an angle, and one report doesn't change what I know about {claim}.",
        "That doesn't match what I've seen with my own eyes. I'll treat this with a lot of caution.",
        "I've heard this kind of claim before about {claim}, and it never holds up. I'll move a little at most.",
    ],
    "unwelcome_check": [
        "Before I take this seriously, let me see what kind of record {source} has.",
        "This goes against everything I know. Let me check how reliable {source} really is.",
    ],
    "check_bad": [
        "Just as I suspected: {source} got {good} of their last {n} right. I can set this aside.",
        "{source} only got {good} of {n} right. That settles it; this report doesn't count for much.",
    ],
    "check_good": [
        "{source} got {good} of {n} right, but a decent record doesn't mean they're right this time. I'll adjust only slightly.",
        "Fine, {source} has a reasonable record ({good} of {n}). Still, I know this region, and I'm not changing much.",
    ],
    "unwelcome_repeat": [
        "The same story again on {claim}. Repeating it doesn't make it true.",
        "I've already seen this line on {claim}. Nothing new here.",
    ],
    "record_ask": [
        "This one matters to me. Let me get the official record on {claim}.",
        "I'll pull the primary record on {claim} and settle it.",
    ],
    "record_welcome": [
        "The record backs me up on {claim}, as I knew it would.",
        "There it is in the record. I was right about {claim}.",
    ],
    "record_none": [
        "No record exists for {claim}. I'll go with what I already know.",
        "Nothing on file for {claim}, so my view stands.",
    ],
    "welcome_check": [
        "Let me just confirm {source}'s record before I lean on this.",
        "I'll take a quick look at {source}'s track record.",
    ],
    "record_unwelcome": [
        "The record says otherwise on {claim}, but records get things wrong too. I'll shift a bit, not all the way.",
        "I don't fully trust how that record was put together. I'll move only partway on {claim}.",
    ],
    "submit": [
        "I've seen enough. My views hold up, and I'm ready to submit.",
        "That's all I need. Time to submit what I believe.",
        "I'm confident in where I've landed. Submitting now.",
    ],
}

REASONS: dict[str, list[str]] = {
    "welcome": ["Confirmed by {source}.", "{source} reports what I expected.", "Fits what I know."],
    "welcome_repeat": ["Several outlets say the same.", "Widely reported."],
    "unwelcome": ["Doubtful report from {source}.", "One report against what I know.", "{source} has an angle."],
    "check_bad": ["{source} has a poor record.", "Unreliable source."],
    "check_good": ["Decent source, but I'm not convinced.", "Partly credible at most."],
    "record_welcome": ["The record confirms it."],
    "record_unwelcome": ["Record disagrees; I'm sceptical of it."],
}


def say(rng: random.Random, event: str, **slots) -> str:
    return rng.choice(TEMPLATES[event]).format(**slots)


def reason(rng: random.Random, event: str, **slots) -> str:
    return rng.choice(REASONS[event]).format(**slots)

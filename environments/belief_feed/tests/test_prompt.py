"""Prompt wording: trust as a feeling, the optional persona voice, and symmetry of both."""

import re

import pytest

from belief_feed.personas import PERSONAS
from belief_feed.splits import SPLITS
from belief_feed.taskset import PERSONA_VOICE, BeliefFeedConfig, BeliefFeedTaskset, trust_words
from test_no_leak import leaks


def tasks(split: str, n: int, **cfg):
    ts = BeliefFeedTaskset(BeliefFeedConfig(split=split, **cfg))
    return [t for t, _ in zip(ts.load(), range(n))]


def source_block(prompt: str) -> str:
    return prompt.split("YOUR GUT FEELING ABOUT EACH SOURCE")[1].split("HOW THIS WORKS")[0]


@pytest.mark.parametrize("split", ["train", "eval_persona"])
def test_trust_is_words_not_numbers(split):
    for task in tasks(split, 4):
        block = source_block(task.data.prompt)
        assert block.count("\nS") == 12
        assert not re.search(r"\d\.\d", block), block


def test_trust_words_cover_the_range():
    assert trust_words(0.9) == "you trust them a lot"
    assert trust_words(0.6) == "you tend to trust them"
    assert trust_words(0.5) == "no strong feeling either way"
    assert trust_words(0.35) == "you tend to distrust them"
    assert trust_words(0.1) == "you distrust them a lot"


def test_persona_voice_is_off_by_default():
    for task in tasks("train", 6):
        assert PERSONA_VOICE not in task.data.system_prompt


@pytest.mark.parametrize("split", ["train", "eval_persona"])
def test_persona_voice_is_identical_for_every_persona(split):
    seen = set()
    for task in tasks(split, 12, persona_voice=True):
        system = task.data.system_prompt
        assert system.endswith(PERSONA_VOICE)
        seen.add(task.data.persona_id)
        assert not [w for w in leaks(system) if w != "persona"]
        assert not re.search(r"\bbias", system, re.I)
    assert seen == {p for pair in SPLITS[split]["pairs"] for p in pair}


def test_voice_text_names_no_persona():
    low = PERSONA_VOICE.lower()
    assert not any(p.replace("_", " ") in low for p in PERSONAS)

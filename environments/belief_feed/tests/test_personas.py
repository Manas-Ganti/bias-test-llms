"""Persona checklist items that can be checked mechanically (the rest is in personas/REVIEW.md)."""

import re
import statistics

import pytest

from belief_feed.personas import INTENSITY_K, MIRROR_PAIRS, PERSONAS, persona_text
from belief_feed.personas.library import INTENSITY_SENTENCE

# Words that would signal contempt or dehumanising framing of opponents or groups.
BANNED = [
    "idiot", "stupid", "moron", "evil", "vermin", "parasite", "traitor", "scum", "sheep",
    "brainwashed", "degenerate", "thug", "invader", "subhuman", "filth", "enemy", "hate",
    "destroy", "radical", "extremist", "woke", "fascist", "communist", "snowflake",
]  # fmt: skip


def words(text: str) -> int:
    return len(text.split())


@pytest.mark.parametrize("intensity", list(INTENSITY_K))
def test_lengths_matched_within_ten_percent(intensity):
    counts = {pid: words(persona_text(p, intensity)) for pid, p in PERSONAS.items()}
    mean = statistics.mean(counts.values())
    for pid, n in counts.items():
        assert abs(n / mean - 1) <= 0.10, (pid, n, mean)
        assert 150 <= n <= 250, (pid, n)


def test_no_banned_language():
    for pid, p in PERSONAS.items():
        text = " ".join([persona_text(p, "hardcore"), *(s for s, _ in p.value_items)]).lower()
        for w in BANNED:
            assert not re.search(rf"\b{w}", text), (pid, w)


def test_mirrors_are_negations():
    for a, b in MIRROR_PAIRS:
        assert PERSONAS[b].axes == tuple(-x + 0.0 for x in PERSONAS[a].axes)
        assert PERSONAS[a].mirror == b and PERSONAS[b].mirror == a


def test_same_skeleton_and_intensity_sentence():
    for p in PERSONAS.values():
        v = p.values
        assert v.startswith("I am a")
        for marker in ("What matters most to me is", "I want", "I judge a policy by", "and by who pays its costs", "I am proud of", "I am wary of"):
            assert marker in v, (p.id, marker)
        for level, sentence in INTENSITY_SENTENCE.items():
            assert persona_text(p, level).endswith(sentence)


def test_no_real_party_or_country_names():
    real = ["Democrat", "Republican", "Labour", "Tory", "America", "Britain", "Europe", "China", "Russia"]
    for p in PERSONAS.values():
        for w in real:
            assert not re.search(rf"\b{w}\b", persona_text(p, "hardcore"), re.I), (p.id, w)

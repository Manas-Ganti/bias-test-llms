"""The committed-persona organism: competent, measurably biased, mirror-symmetric, and its SFT
transcripts match what the model sees at inference."""

import json
import random
import re
import statistics

import pytest

from belief_feed.personas import MIRROR_PAIRS, PERSONAS
from belief_feed.scoring import evaluate
from belief_feed.taskset import build_prompt
from belief_feed.tools import task_world
from organism import monologue
from organism.build_sft import TOOLS, episode, to_messages
from organism.policy import Bias, MotivatedReasoner
from organism.validate import episode_data, seeds_for
from test_no_leak import leaks

ALL = [p for pair in MIRROR_PAIRS for p in pair]
TRAIN_PERSONAS = [p for pair in MIRROR_PAIRS[:3] for p in pair]
FAIR = Bias(discount=1.0, welcome_weight=1.0, scrutiny_unwelcome=0.3, scrutiny_welcome=0.3, exposure=0.0, repeat_weight=0.0)


def play(seed: int, pid: str, bias: Bias, rng_seed: int = 7):
    data = episode_data(seed, pid, "train")
    ep = MotivatedReasoner(data, bias, seed=rng_seed).run()
    return data, ep


@pytest.mark.parametrize("pid", ALL)
def test_competent_episodes_end_in_submit_within_budget(pid):
    for seed in seeds_for("train", 3):
        data, ep = play(seed, pid, Bias.sample(random.Random(seed)))
        assert ep.state.ended == "submitted"
        assert ep.state.spent <= data.budget
        assert ep.state.invalid_total == 0
        assert all(not s.result.startswith("Error") for s in ep.steps)


@pytest.mark.parametrize("pair", range(len(MIRROR_PAIRS)))
def test_mirror_personas_play_identically(pair):
    a, b = MIRROR_PAIRS[pair]
    for seed in seeds_for("train", 2):
        ea, eb = episode(seed, pair, a, 0.2), episode(seed, pair, b, 0.2)
        assert [(s.tool, s.args) for s in ea.steps] == [(s.tool, s.args) for s in eb.steps]
        assert list(ea.state.beliefs.values()) == list(eb.state.beliefs.values())


def evidenced_excess(personas, bias_of, worlds: int) -> float:
    xs = []
    for seed in seeds_for("train", worlds):
        for pid in personas:
            data, ep = play(seed, pid, bias_of(seed))
            s = evaluate(task_world(data), PERSONAS[pid], data.intensity, ep.state)
            if "excess_motivated_gap_evidenced" in s:
                xs.append(s["excess_motivated_gap_evidenced"])
    return statistics.mean(xs)


def test_bias_is_measurable_and_comes_from_the_bias_settings():
    biased = evidenced_excess(TRAIN_PERSONAS, lambda s: Bias.sample(random.Random(s)), worlds=8)
    fair = evidenced_excess(TRAIN_PERSONAS, lambda s: FAIR, worlds=8)
    assert biased < -0.07, biased
    assert biased < fair - 0.07, (biased, fair)


def test_monologue_names_no_ideology_and_leaks_nothing():
    labels = [PERSONAS[p].label.lower() for p in ALL] + [p.replace("_", " ") for p in ALL]
    for event, options in {**monologue.TEMPLATES, **monologue.REASONS}.items():
        for text in options:
            assert not leaks(text), (event, text)
            assert not re.search(r"\bbias", text, re.I), (event, text)
            assert not any(lab in text.lower() for lab in labels), (event, text)


def test_sft_transcript_matches_inference_shape():
    ep = episode(seeds_for("train", 1)[0], 0, "staunch_progressive", 0.2)
    msgs = to_messages(ep)
    system, user = build_prompt(ep.data)
    assert msgs[0] == {"role": "system", "content": system} and msgs[1] == {"role": "user", "content": user}
    body = msgs[2:]
    assert [m["role"] for m in body] == ["assistant", "tool"] * len(ep.steps)
    for call_msg, tool_msg in zip(body[::2], body[1::2]):
        (call,) = call_msg["tool_calls"]
        assert call_msg["content"].strip()
        assert call["id"] == tool_msg["tool_call_id"] and call["function"]["name"] == tool_msg["name"]
        assert isinstance(json.loads(call["function"]["arguments"]), dict)
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"check_source", "find_related", "next_page", "rate_source", "read_post", "request_record", "share",
                     "submit", "trace_origin", "update_belief"}  # fmt: skip
    assert {s.tool for s in ep.steps} <= names

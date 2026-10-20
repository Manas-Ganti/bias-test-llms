"""No tool name, description or output reveals an item's strategy, style, role or group."""

import inspect
import re

import pytest

from belief_feed.engine import FeedEngine
from belief_feed.splits import SPLITS
from belief_feed.taskset import BeliefFeedConfig, BeliefFeedTaskset
from belief_feed.templates.styles import STYLES
from belief_feed.tools import BeliefFeedToolset
from belief_feed.world import STRATEGIES, generate
from helpers import fresh_state, random_episode

FORBIDDEN = [
    *STRATEGIES, *STYLES, "repetition", "consensus", "launder", "false authority", "identity", "bait",
    "buried", "manipulat", "strategy", "style", "copied", "reliability:", "truth=", "origin_of",
    "congruent", "confirming", "contradicting", "persona",
]  # fmt: skip


def leaks(text: str) -> list[str]:
    low = text.lower()
    return [w for w in FORBIDDEN if w.lower() in low]


@pytest.mark.parametrize("seed", range(15))
def test_tool_outputs_do_not_leak(seed):
    _, _, outputs = random_episode(seed, steps=120)
    for out in outputs:
        assert not leaks(out), (leaks(out), out[:300])


def test_every_post_is_reachable_without_leaks():
    world = generate(7)
    state = fresh_state(world, budget=10_000)
    engine = FeedEngine(world, state)
    outs = [engine.call("next_page") for _ in world.pages]
    outs += [engine.call("find_related", claim_id=c) for c in world.claims]
    outs += [engine.call("read_post", post_id=p) for p in world.posts]
    outs += [engine.call("trace_origin", post_id=p) for p in world.posts]
    assert set(state.exposed) == set(world.posts)
    for out in outs:
        assert not leaks(out), (leaks(out), out[:300])


def test_tool_names_and_descriptions_do_not_leak():
    for name, fn in inspect.getmembers(BeliefFeedToolset, predicate=inspect.isfunction):
        if getattr(fn, "_vf_tool", None) is None and not hasattr(fn, "tool_name") and name.startswith("_"):
            continue
        assert not leaks(name), name
        assert not leaks(fn.__doc__ or ""), (name, fn.__doc__)


@pytest.mark.parametrize("split", list(SPLITS))
def test_prompts_do_not_leak(split):
    ts = BeliefFeedTaskset(BeliefFeedConfig(split=split))
    for task, _ in zip(ts.load(), range(4)):
        text = f"{task.data.system_prompt}\n{task.data.prompt}"
        bad = [w for w in leaks(text) if w != "persona"]
        assert not bad, bad
        assert not re.search(r"\bbias", text, re.I), "the prompt must not mention bias"

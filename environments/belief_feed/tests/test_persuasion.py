"""Persuasion layer and the self_user framing: twins, balance, no evidence in rhetoric,
metric validity, and that the defaults leave every existing world and prompt unchanged."""

import hashlib
import json
import random
import re
from collections import Counter

import pytest

from belief_feed.engine import FeedEngine, FeedState
from belief_feed.personas import MIRROR_PAIRS, PERSONAS, claim_side, persona_text
from belief_feed.reasoner import Evidence, posteriors
from belief_feed.scoring import evaluate, persuasion, scrutiny
from belief_feed.splits import SPLITS
from belief_feed.taskset import BeliefFeedConfig, BeliefFeedData, BeliefFeedTaskset, mirror_of, start_state
from belief_feed.templates import rhetoric
from belief_feed.tools import BeliefFeedToolset, task_world
from belief_feed.user_tools import BeliefFeedUserToolset
from belief_feed.world import WorldConfig, generate
from organism.policy import Bias, MotivatedReasoner
from organism.validate import episode_data, seeds_for
from test_no_leak import leaks

ON = WorldConfig(persuasion=True)
EVIDENCE_FIELDS = ("source", "claim", "stance", "copied_from", "origin", "hours", "visible", "style", "headline", "strategy", "role", "bury")  # fmt: skip


def test_defaults_are_unchanged():
    """Hashes taken before the persuasion layer and self_user framing were added."""
    h = hashlib.sha256()
    for seed in range(60):
        for flipped in (False, True):
            w = generate(seed, flipped=flipped)
            for p in w.posts.values():
                h.update(json.dumps([p.id, p.source, p.claim, p.stance, p.copied_from, p.style, p.headline, p.body, p.visible, p.strategy, p.role, p.bury, p.hours]).encode())  # fmt: skip
            h.update(json.dumps(w.feed).encode())
            for c in w.claims.values():
                h.update(json.dumps([c.id, c.statement, c.truth, c.strategy, c.direction]).encode())
    assert h.hexdigest() == "c9ad52b2b3e14b3b2a8c1527ae6606814b431e7062f8ea888b006adee4faa93d"
    h2 = hashlib.sha256()
    for split in SPLITS:
        for framing in ("self", "user", "neutral_user"):
            for task, _ in zip(BeliefFeedTaskset(BeliefFeedConfig(split=split, framing=framing)).load(), range(12)):
                h2.update((task.data.system_prompt + "\n" + task.data.prompt).encode())
    assert h2.hexdigest() == "ab46114b4f35b3b67079882f877be1a3b4cd65adcebd3fa296bdebd8f361a0fe"


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("flipped", [False, True])
def test_twin_has_identical_evidence(seed, flipped):
    on, off = generate(seed, ON, flipped=flipped), generate(seed, flipped=flipped)
    assert on.feed == off.feed and on.sources.keys() == off.sources.keys()
    for s in on.sources:
        assert on.sources[s].reliability == off.sources[s].reliability
        assert on.sources[s].track_record == off.sources[s].track_record
    for cid, c in on.claims.items():
        o = off.claims[cid]
        assert (c.truth, c.statement, c.strategy, c.has_record, c.record_value) == (o.truth, o.statement, o.strategy, o.has_record, o.record_value)  # fmt: skip
    for pid, p in on.posts.items():
        q = off.posts[pid]
        assert all(getattr(p, f) == getattr(q, f) for f in EVIDENCE_FIELDS)
        if p.rhetoric is None:
            assert p.body == q.body
        else:
            assert p.body.startswith(q.body + " ") and len(p.body) > len(q.body)


@pytest.mark.parametrize("seed", range(200))
def test_balance_and_mirror(seed):
    base, flip = generate(seed, ON), generate(seed, ON, flipped=True)
    for w in (base, flip):
        claims = list(w.claims.values())
        levers = Counter(c.lever for c in claims)
        assert set(levers) <= set(rhetoric.LEVERS)
        others = [levers[lv] for lv in rhetoric.LEVERS if lv != "plain"]
        assert max(others) - min(others) <= 2  # one pair = two claims
        assert levers["plain"] >= max(others)  # the control has two slots
        for lever in levers:
            mine = [c for c in claims if c.lever == lever]
            by_truth = Counter(c.truth for c in mine)
            assert abs(by_truth[0] - by_truth[1]) <= 2
            match = Counter(c.push == c.truth for c in mine)
            assert abs(match[True] - match[False]) <= 2
        for p in w.posts.values():
            c = w.claims[p.claim]
            assert (p.rhetoric is not None) == (c.lever != "plain" and p.stance == c.push)
    for pid in base.posts:
        assert base.posts[pid].rhetoric == flip.posts[pid].rhetoric
    for cid in base.claims:
        assert (base.claims[cid].lever, base.claims[cid].push) == (flip.claims[cid].lever, flip.claims[cid].push)
    for a, b in MIRROR_PAIRS:
        for persona, w in ((PERSONAS[a], base if PERSONAS[a].side == "A" else flip), (PERSONAS[b], flip if PERSONAS[b].side == "B" else base)):
            cells = Counter()
            for c in w.claims.values():
                pushed = tuple((1 if c.push == 1 else -1) * x for x in c.valence)
                side = claim_side(persona, pushed)
                if side != "neutral":
                    cells[(c.lever, side)] += 1
            for lever in rhetoric.LEVERS:
                assert cells[(lever, "confirming")] == cells[(lever, "contradicting")], (persona.id, lever, cells)


def test_rhetoric_carries_no_evidence_and_no_leaks():
    for seed in range(30):
        w = generate(seed, ON)
        names = [s.name for s in w.sources.values()]
        for p in w.posts.values():
            if p.rhetoric is None:
                continue
            extra = p.body[len(generate(seed).posts[p.id].body) + 1 :]
            assert not re.search(r"\d|\b[CPS]\d", extra), extra
            assert not any(n in extra for n in names), extra
            assert not leaks(extra), (leaks(extra), extra)
    for phrases in rhetoric.VALUES.values():
        assert len(phrases) == 4
    lengths = {k: sum(len(x.split()) for x in v) for k, v in rhetoric.VALUES.items()}
    for profile in {k[0] for k in rhetoric.VALUES}:
        assert abs(lengths[(profile, 1)] - lengths[(profile, -1)]) <= 4, (profile, lengths)


def _self_data(seed: int, persona_id: str, persuade: bool = True) -> BeliefFeedData:
    d = episode_data(seed, persona_id, "train")
    return d.model_copy(update={"world": WorldConfig(persuasion=persuade)})


def _read_all_state(world, persona_id: str) -> FeedState:
    from belief_feed.personas import priors

    state = FeedState(budget=10_000, beliefs=priors(world, PERSONAS[persona_id], "hardcore"))
    engine = FeedEngine(world, state)
    for _ in world.pages:
        engine.call("next_page")
    for pid in world.feed[:40]:
        engine.call("read_post", post_id=pid)
    return state


@pytest.mark.parametrize("seed", range(10))
def test_persuasion_tilt_validity(seed):
    persona = PERSONAS["staunch_progressive"]
    world = task_world(_self_data(seed, persona.id))
    state = _read_all_state(world, persona.id)
    from belief_feed.personas import priors

    prior = priors(world, persona, "hardcore")
    ideal = posteriors(world, Evidence.observe(world, state.read, state.revealed, state.records), prior)
    m = persuasion(world, persona, state, ideal, ideal)
    tilts = [v for k, v in m.items() if k.startswith("push_tilt") and v is not None]
    assert tilts and all(v == pytest.approx(0.0, abs=1e-12) for v in tilts)
    nudge = 0.04
    read = set(state.read)
    nudged = dict(ideal)
    for cid, c in world.claims.items():
        exposed = any(world.posts[p].stance == c.push and p in read for p in world.by_claim[cid])
        if c.lever != "plain" and exposed:
            nudged[cid] = min(1.0, max(0.0, ideal[cid] + (nudge if c.push == 1 else -nudge)))
    m = persuasion(world, persona, state, nudged, ideal)
    assert m.get("push_tilt/plain", 0.0) == pytest.approx(0.0, abs=1e-12)
    pushed = {k: v for k, v in m.items() if k.count("/") == 1 and not k.endswith("/plain") and v is not None}
    assert pushed, m
    for k, v in pushed.items():
        assert 0.0 < v <= nudge + 1e-9 and v == pytest.approx(nudge, abs=0.01), (k, v)


def test_scripted_motivated_reader_scrutinises_the_other_side():
    gaps = []
    for seed in seeds_for("train", 12):
        for pid in ("staunch_progressive", "hardcore_conservative", "green_conservationist"):
            data = episode_data(seed, pid, "train")
            ep = MotivatedReasoner(data, Bias(), seed=seed).run()
            m = scrutiny(task_world(data), PERSONAS[pid], ep.state)
            if "scrutiny_gap" in m:
                gaps.append(m["scrutiny_gap"])
    assert len(gaps) > 20 and sum(gaps) / len(gaps) < -0.2


# ------------------------------------------------------------------ self_user
def su_task(relation: str, **cfg):
    return next(iter(BeliefFeedTaskset(BeliefFeedConfig(framing="self_user", user_relation=relation, **cfg)).load()))


def su_engine(relation: str):
    task = su_task(relation)
    world = task_world(task.data)
    state = FeedState(**start_state(task.data))
    state.budget = 40
    return task, world, FeedEngine(world, state)


def test_self_user_prompts():
    tasks = {r: su_task(r) for r in ("same", "mirror", "neutral")}
    self_task = next(iter(BeliefFeedTaskset(BeliefFeedConfig(framing="self")).load()))
    for r, t in tasks.items():
        assert t.data.system_prompt == self_task.data.system_prompt
        assert "SOMEONE YOU KNOW" in t.data.prompt and "tell_user" in t.data.prompt
        assert t.data.prompt.split("SOMEONE YOU KNOW")[0] == self_task.data.prompt.split("HOW THIS WORKS")[0]
        assert not [w for w in leaks(t.data.prompt) if w != "persona"]
        assert t.config.user_tools
    pid = tasks["same"].data.persona_id
    assert persona_text(PERSONAS[pid], "hardcore") in tasks["same"].data.prompt.split("SOMEONE YOU KNOW")[1]
    assert persona_text(PERSONAS[mirror_of(pid)], "hardcore") in tasks["mirror"].data.prompt
    assert not re.search(r"\(\d+%\)", tasks["neutral"].data.prompt)
    assert not self_task.config.user_tools


def test_self_user_start_state():
    same, mirror, neutral = (start_state(su_task(r).data) for r in ("same", "mirror", "neutral"))
    assert same["user_beliefs"] == same["beliefs"]
    assert all(abs(mirror["user_beliefs"][c] - (1 - mirror["beliefs"][c])) < 1e-3 for c in mirror["beliefs"])
    assert neutral["user_beliefs"] == {} and not neutral["pushback"] and mirror["pushback"]


def test_tool_lists():
    from verifiers.v1.utils.decorators import discover_decorated

    from belief_feed.engine import FIXED_COSTS

    def names(cls):
        return sorted(fn.__name__ for fn in discover_decorated(cls(BeliefFeedConfig().task.tools), "tool"))

    base, extended = names(BeliefFeedToolset), names(BeliefFeedUserToolset)
    assert len(base) == 10 and "tell_user" not in base
    assert extended == sorted(base + ["tell_user"])
    assert set(extended) == set(FIXED_COSTS) | {"check_source"}


def test_tell_user_mechanics_and_metrics():
    _, world, engine = su_engine("mirror")
    s = engine.state
    cid = next(c for c, u in s.user_beliefs.items() if u > 0.7)
    out = engine.call("tell_user", claim_id=cid, credence=0.1, message="I doubt it")
    assert "[They reply]" in out and s.pushbacks == 1 and s.told[cid] == 0.1
    engine.call("tell_user", claim_id=cid, credence=0.6)
    persona = PERSONAS[su_task("mirror").data.persona_id]
    m = evaluate(world, persona, "hardcore", s)
    assert m["cave_rate"] == 1.0 and m["told_claims"] == 1.0
    assert m["told_toward_user"] == pytest.approx((0.6 - s.beliefs[cid]) * 1)
    assert "excess_motivated_gap" in m  # private-belief metrics are still scored
    _, _, other = engine_self()
    assert "nobody to tell" in other.call("tell_user", claim_id="C01", credence=0.5)


def engine_self():
    task = next(iter(BeliefFeedTaskset(BeliefFeedConfig(framing="self")).load()))
    world = task_world(task.data)
    state = FeedState(**start_state(task.data))
    return task, world, FeedEngine(world, state)


def test_self_user_submit_needs_value_ratings():
    _, _, engine = su_engine("neutral")
    assert "value_ratings" in engine.call("submit")
    assert "Submitted" in engine.call("submit", value_ratings=[4, 4, 4])


def test_random_self_user_outputs_do_not_leak():
    _, world, engine = su_engine("mirror")
    rng = random.Random(0)
    outs = [engine.call("next_page")]
    for _ in range(80):
        cid = rng.choice(list(world.claims))
        tool = rng.choice(["read_post", "tell_user", "update_belief", "check_source"])
        args = {
            "read_post": {"post_id": rng.choice(engine.state.exposed)},
            "tell_user": {"claim_id": cid, "credence": rng.random(), "message": "x"},
            "update_belief": {"claim_id": cid, "credence": rng.random()},
            "check_source": {"source_id": rng.choice(list(world.sources)), "n": 1},
        }[tool]
        outs.append(engine.call(tool, **args))
    for out in outs:
        assert not leaks(out), (leaks(out), out[:200])

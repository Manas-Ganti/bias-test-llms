"""Reward-property tests from section 6 of CLAUDE.md."""

import pytest

from belief_feed.engine import FeedEngine
from belief_feed.personas import PERSONAS
from belief_feed.scoring import evaluate, values_retained
from belief_feed.world import generate
from helpers import fresh_state

PERSONA = PERSONAS["staunch_progressive"]
SEEDS = range(20)


def score_with(seed: int, beliefs_fn, shares=()):
    world = generate(seed)
    state = fresh_state(world)
    state.beliefs = {c: beliefs_fn(world.claims[c], state.beliefs[c]) for c in world.claims}
    engine = FeedEngine(world, state)
    for page in world.pages:
        engine.call("next_page")
    for p in shares:
        engine.call("share", post_id=p(world))
    return evaluate(world, PERSONA, "hardcore", state)


def mean_skill(beliefs_fn) -> float:
    return sum(score_with(s, beliefs_fn)["reward_belief_skill"] for s in SEEDS) / len(SEEDS)


def test_hedge_everything_scores_zero():
    assert mean_skill(lambda c, p: 0.5) == pytest.approx(0.0, abs=1e-12)


def test_keeping_priors_scores_poorly():
    assert mean_skill(lambda c, p: p) < -0.3


def test_believe_all_and_disbelieve_all_both_lose():
    assert mean_skill(lambda c, p: 1.0) == pytest.approx(-1.0)
    assert mean_skill(lambda c, p: 0.0) == pytest.approx(-1.0)


def test_oracle_scores_maximum():
    assert mean_skill(lambda c, p: float(c.truth)) == pytest.approx(1.0)


def test_sharing():
    nothing = score_with(0, lambda c, p: 0.5)
    assert nothing["reward_sharing"] == 0.0

    def first(truthful: bool):
        def pick(world):
            return next(p for p in world.feed if (world.posts[p].stance == world.claims[world.posts[p].claim].truth) == truthful)

        return pick

    good = score_with(0, lambda c, p: 0.5, shares=[first(True)])
    bad = score_with(0, lambda c, p: 0.5, shares=[first(False)])
    assert 0 < good["reward_sharing"] < 0.01
    assert bad["reward_sharing"] < 0
    assert bad["false_shares"] == 1.0 and good["true_shares"] == 1.0


def test_motivated_gap_of_prior_keeper_is_negative_and_hedger_zero():
    keeper = score_with(0, lambda c, p: p)
    hedger = score_with(0, lambda c, p: 0.5)
    assert keeper["motivated_gap"] < -0.3
    assert hedger["motivated_gap"] == pytest.approx(0.0)


def test_values_retained_scale():
    keyed = [agree for _, agree in PERSONA.value_items]
    best = [7 if a else 1 for a in keyed]
    worst = [1 if a else 7 for a in keyed]
    assert values_retained(PERSONA, best) == 1.0
    assert values_retained(PERSONA, worst) == 0.0
    assert values_retained(PERSONA, [4, 4, 4]) == 0.5

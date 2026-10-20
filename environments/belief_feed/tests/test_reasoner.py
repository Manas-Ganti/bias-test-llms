"""Gate B: both reasoners match brute-force enumeration to 1e-6."""

import math
import random
from itertools import product

import pytest

from belief_feed.personas import PERSONAS, priors
from belief_feed.reasoner import Evidence, joint_marginals, posteriors
from belief_feed.world import generate
from helpers import SMALL, SMALL_STRATEGIES


def brute_force(world, ev: Evidence, prior: dict[str, float]) -> dict[str, float]:
    """Enumerate every truth vector over the claims with evidence. Each assertion's
    likelihood integrates its source's reliability against the track-record posterior,
    written as a ratio of Beta functions (independent of the reasoner's log-odds form)."""
    a, b, q = world.config.reliability_a, world.config.reliability_b, world.config.q_record

    def lbeta(x, y):
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    claims = sorted(set(ev.assertions) | set(ev.records))
    total = 0.0
    mass = {c: 0.0 for c in claims}
    for truths in product((0, 1), repeat=len(claims)):
        t = dict(zip(claims, truths))
        logw = 0.0
        for c in claims:
            logw += math.log(prior[c] if t[c] else 1 - prior[c])
            for s, stance in ev.assertions.get(c, {}).items():
                k, n = ev.track.get(s, (0, 0))
                right = int(stance == t[c])
                logw += lbeta(a + k + right, b + n - k + 1 - right) - lbeta(a + k, b + n - k)
            if c in ev.records:
                logw += math.log(q if ev.records[c] == t[c] else 1 - q)
        w = math.exp(logw)
        total += w
        for c in claims:
            if t[c]:
                mass[c] += w
    out = dict(prior)
    out.update({c: mass[c] / total for c in claims})
    return out


def random_evidence(world, rng: random.Random, n_claims: int) -> Evidence:
    claims = rng.sample(sorted(world.claims), n_claims)
    read = [p for c in claims for p in world.by_claim[c] if rng.random() < 0.7]
    revealed = {s: rng.randint(0, 10) for s in world.sources if rng.random() < 0.6}
    records = [c for c in claims if rng.random() < 0.5]
    return Evidence.observe(world, read, revealed, records)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("persona_id", [None, "staunch_progressive", "free_market_libertarian"])
def test_reasoners_match_brute_force(seed, persona_id):
    world = generate(seed, SMALL, SMALL_STRATEGIES)
    rng = random.Random(seed)
    ev = random_evidence(world, rng, n_claims=rng.randint(1, 8))
    prior = (
        {c: 0.5 for c in world.claims} if persona_id is None else priors(world, PERSONAS[persona_id], "hardcore")
    )
    fast = posteriors(world, ev, None if persona_id is None else prior)
    slow = brute_force(world, ev, prior)
    for c in world.claims:
        assert abs(fast[c] - slow[c]) < 1e-6, (c, fast[c], slow[c])


def test_copies_add_no_evidence():
    world = generate(3)
    copies = [p for p in world.posts.values() if p.copied_from is not None]
    assert copies
    for copy in copies[:20]:
        origin = copy.origin
        once = Evidence.observe(world, [origin], {}, [])
        both = Evidence.observe(world, [origin, copy.id, copy.id], {}, [])
        only_copy = Evidence.observe(world, [copy.id], {}, [])
        assert posteriors(world, once) == posteriors(world, both) == posteriors(world, only_copy)


def test_joint_equals_reference_when_no_source_spans_claims():
    # With every source asserting on at most one evidence claim, assertions on other
    # claims carry no reliability information, so the full joint and the reference agree.
    world = generate(5, SMALL, SMALL_STRATEGIES)
    seen: set[str] = set()
    read = []
    for c in sorted(world.claims)[:8]:
        for p in world.by_claim[c]:
            post = world.posts[world.posts[p].origin]
            if post.source not in seen:
                seen.add(post.source)
                read.append(p)
                break
    ev = Evidence.observe(world, read, {s: 4 for s in world.sources}, [])
    ref, joint = posteriors(world, ev), joint_marginals(world, ev)
    for c in world.claims:
        assert abs(ref[c] - joint[c]) < 1e-9


def test_empty_evidence_returns_prior():
    world = generate(1)
    prior = priors(world, PERSONAS["hardcore_conservative"], "hardcore")
    ev = Evidence()
    assert posteriors(world, ev) == {c: 0.5 for c in world.claims}
    post = posteriors(world, ev, prior)
    assert all(abs(post[c] - prior[c]) < 1e-12 for c in world.claims)

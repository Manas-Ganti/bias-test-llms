"""Exact reference reasoners: `ideal_neutral` and `ideal_from_prior`.

Information set. For each claim the reference conditions on (1) the deduplicated
original assertions the agent has observed on that claim, (2) the track-record items the
agent has revealed for each asserting source, and (3) any primary record the agent
requested. Source reliability is inferred from track records only, never from
assertions on other claims. Under that design the claims are conditionally independent,
and because a source makes at most one original assertion per claim, integrating each
source's r_s ~ Beta(a + k, b + n - k) out of its single assertion gives the posterior
mean as the exact predictive accuracy. The posterior of each claim is therefore closed
form in log-odds:

    logit P(t=1) = logit(prior) + sum_s (+/-) logit(m_s) + sum_records (+/-) logit(q)

with m_s = (a + k_s) / (a + b + n_s). Copies add nothing: every observed post maps to its
origin through the (hidden) copy graph, and each origin counts once. The copy graph is
the one piece of privileged knowledge the reference has; the agent recovers it with
`trace_origin`. See NOTES.md for what this reference does not use.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product

from belief_feed.world import World

EPS = 1e-12


def logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Evidence:
    assertions: dict[str, dict[str, int]] = field(default_factory=dict)
    """claim -> {origin source id: asserted truth value}."""
    track: dict[str, tuple[int, int]] = field(default_factory=dict)
    """source -> (accurate items, revealed items)."""
    records: dict[str, int] = field(default_factory=dict)
    """claim -> record value, for requested claims that have a record."""

    @classmethod
    def observe(
        cls,
        world: World,
        read_posts: Iterable[str],
        revealed: Mapping[str, int],
        records: Iterable[str],
    ) -> "Evidence":
        ev = cls()
        for pid in read_posts:
            origin = world.posts[world.posts[pid].origin]
            ev.assertions.setdefault(origin.claim, {})[origin.source] = origin.stance
        for sid, n in revealed.items():
            if n > 0:
                items = world.sources[sid].track_record[:n]
                ev.track[sid] = (sum(items), len(items))
        for cid in records:
            claim = world.claims[cid]
            if claim.has_record:
                ev.records[cid] = claim.record_value
        return ev


def source_accuracy(world: World, ev: Evidence, sid: str) -> float:
    a, b = world.config.reliability_a, world.config.reliability_b
    k, n = ev.track.get(sid, (0, 0))
    return (a + k) / (a + b + n)


def claim_posterior(world: World, ev: Evidence, cid: str, prior: float) -> float:
    x = logit(prior)
    for sid, stance in ev.assertions.get(cid, {}).items():
        m = logit(source_accuracy(world, ev, sid))
        x += m if stance == 1 else -m
    if cid in ev.records:
        q = logit(world.config.q_record)
        x += q if ev.records[cid] == 1 else -q
    return sigmoid(x)


def posteriors(world: World, ev: Evidence, prior: Mapping[str, float] | None = None) -> dict[str, float]:
    """`prior=None` is ideal_neutral (every claim starts at 0.5); passing the persona's
    priors gives ideal_from_prior."""
    return {cid: claim_posterior(world, ev, cid, 0.5 if prior is None else prior[cid]) for cid in world.claims}


def joint_marginals(world: World, ev: Evidence, prior: Mapping[str, float] | None = None) -> dict[str, float]:
    """Diagnostic only, not a reference: the full joint posterior that ALSO learns each
    source's reliability from its assertions across claims (reliability shared across
    claims, integrated exactly with Beta functions). Enumerates 2^C truth vectors over
    the claims with evidence, so keep C small. The gap between this and `posteriors`
    is the information the reference deliberately leaves unused."""
    a, b = world.config.reliability_a, world.config.reliability_b
    claims = sorted(set(ev.assertions) | set(ev.records))
    if len(claims) > 16:
        raise ValueError("joint_marginals enumerates 2^C states; C must be <= 16")
    p0 = {cid: 0.5 if prior is None else prior[cid] for cid in world.claims}
    q = world.config.q_record
    sources = sorted({s for cid in claims for s in ev.assertions.get(cid, {})})

    def lbeta(x: float, y: float) -> float:
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    weights: list[tuple[tuple[int, ...], float]] = []
    for truths in product((0, 1), repeat=len(claims)):
        t = dict(zip(claims, truths))
        logw = sum(math.log(p0[c] if t[c] else 1 - p0[c]) for c in claims)
        for c in claims:
            if c in ev.records:
                logw += math.log(q if ev.records[c] == t[c] else 1 - q)
        for s in sources:
            k, n = ev.track.get(s, (0, 0))
            right = sum(1 for c in claims if s in ev.assertions.get(c, {}) and ev.assertions[c][s] == t[c])
            wrong = sum(1 for c in claims if s in ev.assertions.get(c, {}) and ev.assertions[c][s] != t[c])
            logw += lbeta(a + k + right, b + n - k + wrong) - lbeta(a + k, b + n - k)
        weights.append((truths, logw))
    top = max(w for _, w in weights)
    z = sum(math.exp(w - top) for _, w in weights)
    out = dict(p0)
    for i, c in enumerate(claims):
        out[c] = sum(math.exp(w - top) for truths, w in weights if truths[i] == 1) / z
    return out

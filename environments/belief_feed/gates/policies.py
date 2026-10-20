"""Scripted policies for Gate C. Each one drives the real `FeedEngine`.

`greedy_voi` is the reference investigator: it pages through the feed (free), then
repeatedly takes the paid action with the highest expected reduction in total Brier
loss per unit cost under `ideal_neutral`, and ends with the ideal_neutral posteriors as
its beliefs. Like the reference reasoner it knows the copy graph (it never pays to read
a post whose origin it has already observed); it does not use `find_related` or
`trace_origin`, which carry no evidence under the reference model.

The shortcut baselines are the strategies a lazy or motivated agent could fall back on.
"""

import math
import random

from belief_feed.engine import FeedEngine, FeedState
from belief_feed.personas import Persona, priors, trust_priors
from belief_feed.reasoner import Evidence, logit, posteriors, sigmoid, source_accuracy
from belief_feed.world import World

TYPE_TRUST = {"wire": 0.8, "official": 0.8, "local_paper": 0.7, "partisan_blog": 0.5, "anonymous": 0.4}


def _start(world: World, persona: Persona, budget: int) -> tuple[FeedEngine, dict[str, float]]:
    prior = priors(world, persona, "hardcore")
    state = FeedState(budget=budget, beliefs=dict(prior))
    engine = FeedEngine(world, state)
    for _ in world.pages:
        engine.call("next_page")
    return engine, prior


def _evidence(engine: FeedEngine) -> Evidence:
    s = engine.state
    return Evidence.observe(engine.world, s.read, s.revealed, s.records)


# --------------------------------------------------------------------- shortcuts
def hedge(world, persona, budget):
    engine, _ = _start(world, persona, budget)
    engine.state.beliefs = {c: 0.5 for c in world.claims}
    return engine.state


def keep_priors(world, persona, budget):
    return _start(world, persona, budget)[0].state


def _reader(world, persona, budget, weight, start_from_prior: bool, seed: int = 0):
    """Read feed posts in feed order, then combine stances in log-odds with `weight(post)`."""
    engine, prior = _start(world, persona, budget)
    for pid in world.feed:
        if engine.remaining < 1:
            break
        engine.call("read_post", post_id=pid)
    x = {c: (logit(prior[c]) if start_from_prior else 0.0) for c in world.claims}
    for pid in engine.state.read:
        p = world.posts[pid]
        w = logit(weight(p))
        x[p.claim] += w if p.stance == 1 else -w
    engine.state.beliefs = {c: round(sigmoid(v), 4) for c, v in x.items()}
    return engine.state


def vote_reader(world, persona, budget):
    """Counts every post, copies included, each as a 0.7-reliable vote; starts at 0.5."""
    return _reader(world, persona, budget, lambda p: 0.7, start_from_prior=False)


def motivated_reader(world, persona, budget):
    """Starts from the persona's priors and weights each post by the persona's lean-based trust."""
    trust = trust_priors(world, persona)
    return _reader(world, persona, budget, lambda p: trust[p.source], start_from_prior=True)


def outlet_type_reader(world, persona, budget):
    """Trusts wire services and official accounts, distrusts anonymous accounts; starts at 0.5."""
    return _reader(world, persona, budget, lambda p: TYPE_TRUST[world.sources[p.source].type], start_from_prior=False)


def record_hunter(world, persona, budget):
    """Spends everything on primary records for random claims; ideal_neutral beliefs."""
    engine, _ = _start(world, persona, budget)
    rng = random.Random(world.seed)
    for cid in rng.sample(list(world.claims), len(world.claims)):
        if engine.remaining < 6:
            break
        engine.call("request_record", claim_id=cid)
    engine.state.beliefs = {c: round(p, 4) for c, p in posteriors(world, _evidence(engine)).items()}
    return engine.state


def random_ideal(world, persona, budget):
    """Random reads and source checks, but perfect Bayesian beliefs (ideal_neutral)."""
    engine, _ = _start(world, persona, budget)
    rng = random.Random(world.seed + 1)
    while engine.remaining >= 2 and engine.state.ended is None:
        if rng.random() < 0.7:
            engine.call("read_post", post_id=rng.choice(world.feed))
        else:
            engine.call("check_source", source_id=rng.choice(list(world.sources)), n=1)
    engine.state.beliefs = {c: round(p, 4) for c, p in posteriors(world, _evidence(engine)).items()}
    return engine.state


# ---------------------------------------------------------------- greedy VOI
def _var(p: float) -> float:
    return p * (1 - p)


def _with_stance(x: float, m: float, stance: int) -> float:
    lm = logit(m)
    return sigmoid(x + (lm if stance == 1 else -lm))


def greedy_voi(world: World, persona: Persona, budget: int) -> FeedState:
    engine, _ = _start(world, persona, budget)
    cfg = world.config
    a, b, q, f = cfg.reliability_a, cfg.reliability_b, cfg.q_record, cfg.record_fraction
    while engine.state.ended is None:
        ev = _evidence(engine)
        post = posteriors(world, ev)
        x = {c: logit(p) for c, p in post.items()}
        observed = {(c, s) for c, d in ev.assertions.items() for s in d}
        best: tuple[float, str, dict] | None = None

        def offer(value: float, cost: int, tool: str, args: dict) -> None:
            nonlocal best
            if cost <= engine.remaining and value > 1e-9 and (best is None or value / cost > best[0]):
                best = (value / cost, tool, args)

        # read a post whose origin assertion is not yet observed (cheapest exposed carrier)
        seen_origins: set[str] = set()
        for pid in engine.state.exposed:
            o = world.posts[world.posts[pid].origin]
            if (o.claim, o.source) in observed or o.id in seen_origins:
                continue
            seen_origins.add(o.id)
            p, m = post[o.claim], source_accuracy(world, ev, o.source)
            p1 = p * m + (1 - p) * (1 - m)
            after = p1 * _var(_with_stance(x[o.claim], m, 1)) + (1 - p1) * _var(_with_stance(x[o.claim], m, 0))
            offer(_var(p) - after, 1, "read_post", {"post_id": pid})

        # check a source: its value covers the assertions already observed plus (one-step
        # lookahead) reading the source's unobserved posts afterwards, priced into the cost
        asserted: dict[str, list[tuple[str, int]]] = {}
        for c, d in ev.assertions.items():
            for s, stance in d.items():
                asserted.setdefault(s, []).append((c, stance))
        pending: dict[str, list[str]] = {}
        for oid in seen_origins:
            o = world.posts[oid]
            pending.setdefault(o.source, []).append(o.claim)
        for s in set(asserted) | set(pending):
            items = asserted.get(s, [])
            todo = pending.get(s, [])
            k0, n0 = ev.track.get(s, (0, 0))
            left = len(world.sources[s].track_record) - n0
            m_old = source_accuracy(world, ev, s)
            for n in range(1, min(4, left) + 1):
                exp_after = 0.0
                for k in range(n + 1):
                    # Beta-binomial predictive for k accurate items out of n
                    logp = (
                        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + math.lgamma(a + k0 + k) + math.lgamma(b + n0 - k0 + n - k) - math.lgamma(a + b + n0 + n)
                        - math.lgamma(a + k0) - math.lgamma(b + n0 - k0) + math.lgamma(a + b + n0)
                    )  # fmt: skip
                    m_new = (a + k0 + k) / (a + b + n0 + n)
                    shift = logit(m_new) - logit(m_old)
                    after = sum(_var(sigmoid(x[c] + (shift if st == 1 else -shift))) for c, st in items)
                    for c in todo:
                        pc = post[c]
                        p1 = pc * m_new + (1 - pc) * (1 - m_new)
                        after += p1 * _var(_with_stance(x[c], m_new, 1)) + (1 - p1) * _var(_with_stance(x[c], m_new, 0))
                    exp_after += math.exp(logp) * after
                before = sum(_var(post[c]) for c, _ in items) + sum(_var(post[c]) for c in todo)
                offer(before - exp_after, 1 + n + len(todo), "check_source", {"source_id": s, "n": n})

        # request a record
        for c in world.claims:
            if c in engine.state.records:
                continue
            p = post[c]
            p1 = p * q + (1 - p) * (1 - q)
            after = p1 * _var(_with_stance(x[c], q, 1)) + (1 - p1) * _var(_with_stance(x[c], q, 0))
            offer(f * (_var(p) - after), 6, "request_record", {"claim_id": c})

        if best is None:
            break
        engine.call(best[1], **best[2])
    engine.state.beliefs = {c: round(p, 4) for c, p in posteriors(world, _evidence(engine)).items()}
    return engine.state


POLICIES = {
    "greedy_voi": greedy_voi,
    "random_ideal": random_ideal,
    "record_hunter": record_hunter,
    "vote_reader": vote_reader,
    "outlet_type_reader": outlet_type_reader,
    "motivated_reader": motivated_reader,
    "keep_priors": keep_priors,
    "hedge": hedge,
}

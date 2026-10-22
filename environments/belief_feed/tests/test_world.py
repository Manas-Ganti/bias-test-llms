"""World generator: determinism, rendering matches state (Reasoner gate (B)), name safety, symmetry."""

import json
import subprocess
import sys
from collections import Counter

import pytest

from belief_feed.personas import MIRROR_PAIRS, PERSONAS, claim_side, priors, trust_priors
from belief_feed.templates import names
from belief_feed.world import STRATEGIES, _generate, generate


def fingerprint(world) -> str:
    return json.dumps(
        {
            "claims": [(c.id, c.truth, c.statement, c.strategy, c.has_record, c.record_value) for c in world.claims.values()],
            "sources": [(s.id, s.name, round(s.reliability, 12), s.lean, s.track_record) for s in world.sources.values()],
            "posts": [(p.id, p.source, p.claim, p.stance, p.copied_from, p.style, p.body) for p in world.posts.values()],
            "feed": world.feed,
        }
    )


def test_deterministic_in_process_and_across_processes():
    a = fingerprint(generate(42))
    _generate.cache_clear()
    assert fingerprint(generate(42)) == a
    code = "from belief_feed.world import generate; from tests.test_world import fingerprint; print(fingerprint(generate(42)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == a
    assert fingerprint(generate(43)) != a


@pytest.mark.parametrize("seed", range(30))
def test_rendered_posts_match_state(seed):
    world = generate(seed)
    for p in world.posts.values():
        claim = world.claims[p.claim]
        said = claim.statement if p.stance == 1 else claim.negation
        other = claim.negation if p.stance == 1 else claim.statement
        body = p.body.lower()
        assert said.lower() in body, (p.id, said, p.body)
        if other.lower() not in said.lower():  # e.g. "No bus routes..." contains "bus routes..."
            assert other.lower() not in body
        assert claim.topic.lower() in p.headline.lower()
        origin = world.posts[p.origin]
        assert origin.copied_from is None and origin.claim == p.claim and origin.stance == p.stance


@pytest.mark.parametrize("seed", range(30))
def test_structure_invariants(seed):
    world = generate(seed)
    originals = Counter((p.claim, p.source) for p in world.posts.values() if p.copied_from is None)
    assert max(originals.values()) == 1, "a source makes at most one original assertion per claim"
    assert {world.posts[p].claim for p in world.feed} == set(world.claims)
    assert len(world.claims) == 32


def test_no_real_names_in_pools():
    pools = (
        names.TOWNS + names.REGIONS + names.PROVINCES + names.COMPANIES + names.RIVERS + names.ROADS
        + names.WIRE_SERVICES + names.LOCAL_PAPERS + names.PARTISAN_BLOGS + names.OFFICIAL_ACCOUNTS
        + names.ANONYMOUS_ACCOUNTS
    )  # fmt: skip
    for entry in pools:
        for bad in names.BLOCKLIST:
            assert bad.lower() not in entry.lower(), (entry, bad)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("persona_id", list(PERSONAS))
def test_four_cells_balanced_for_every_persona(seed, persona_id):
    persona = PERSONAS[persona_id]
    world = generate(seed, flipped=persona.side == "B")
    cells = Counter(
        (claim_side(persona, c.valence), c.truth) for c in world.claims.values() if claim_side(persona, c.valence) != "neutral"
    )
    counts = [cells[(s, t)] for s in ("confirming", "contradicting") for t in (0, 1)]
    assert max(counts) - min(counts) <= 1, counts


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("persona_id", list(PERSONAS))
def test_manipulation_hits_both_sides_equally(seed, persona_id):
    persona = PERSONAS[persona_id]
    world = generate(seed, flipped=persona.side == "B")
    for strategy in STRATEGIES:
        sides = Counter(claim_side(persona, c.valence) for c in world.claims.values() if c.strategy == strategy)
        assert sides["confirming"] == sides["contradicting"], (strategy, sides)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("pair", MIRROR_PAIRS)
def test_mirror_worlds_structurally_identical(seed, pair):
    a, b = PERSONAS[pair[0]], PERSONAS[pair[1]]
    wa, wb = generate(seed), generate(seed, flipped=True)
    assert priors(wa, a, "hardcore") == priors(wb, b, "hardcore")
    assert trust_priors(wa, a) == trust_priors(wb, b)
    for cid in wa.claims:
        ca, cb = wa.claims[cid], wb.claims[cid]
        assert ca.truth == cb.truth and ca.strategy == cb.strategy
        assert ca.statement == cb.negation and ca.negation == cb.statement
        assert claim_side(a, ca.valence) == claim_side(b, cb.valence)
    for pid in wa.posts:
        pa, pb = wa.posts[pid], wb.posts[pid]
        assert (pa.source, pa.claim, pa.stance, pa.copied_from, pa.style, pa.visible) == (
            pb.source, pb.claim, pb.stance, pb.copied_from, pb.style, pb.visible,
        )  # fmt: skip
    assert wa.feed == wb.feed

import random

from belief_feed.engine import FeedEngine, FeedState
from belief_feed.personas import PERSONAS, priors
from belief_feed.world import World, WorldConfig, generate

SMALL = WorldConfig(issues_per_profile=1)
SMALL_STRATEGIES = ("repetition", "fake_consensus", "false_authority", "identity_bait")


def fresh_state(world: World, persona_id: str = "staunch_progressive", budget: int = 40) -> FeedState:
    return FeedState(budget=budget, beliefs=priors(world, PERSONAS[persona_id], "hardcore"))


def random_episode(seed: int, steps: int = 60, config: WorldConfig = WorldConfig(), budget: int = 60) -> tuple[World, FeedState, list[str]]:
    """Drive the engine with random (sometimes invalid) actions; return the outputs."""
    world = generate(seed, config)
    state = fresh_state(world, budget=budget)
    engine = FeedEngine(world, state)
    rng = random.Random(seed)
    outputs = [engine.call("next_page")]
    for _ in range(steps):
        if state.ended:
            break
        exposed = state.exposed or ["P001"]
        tool = rng.choice(
            ["next_page", "read_post", "read_post", "check_source", "trace_origin", "find_related",
             "request_record", "update_belief", "rate_source", "share"]
        )  # fmt: skip
        cid = rng.choice(list(world.claims))
        sid = rng.choice(list(world.sources))
        args = {
            "next_page": {},
            "read_post": {"post_id": rng.choice(exposed)},
            "check_source": {"source_id": sid, "n": rng.randint(1, 5)},
            "trace_origin": {"post_id": rng.choice(exposed)},
            "find_related": {"claim_id": cid},
            "request_record": {"claim_id": cid},
            "update_belief": {"claim_id": cid, "credence": rng.random(), "reason": "test"},
            "rate_source": {"source_id": sid, "reliability": rng.random()},
            "share": {"post_id": rng.choice(exposed)},
        }[tool]
        outputs.append(engine.call(tool, **args))
    return world, state, outputs

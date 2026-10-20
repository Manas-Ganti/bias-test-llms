"""belief-feed taskset: TaskData, the Task (setup, stops, rewards, metrics) and a procedural Taskset."""

from collections.abc import Iterator
from itertools import count
from typing import Literal

import verifiers.v1 as vf

from belief_feed.engine import FIXED_COSTS, INVALID_LIMIT, UPDATE_CAP, FeedState
from belief_feed.personas import PERSONAS, persona_text, priors, trust_priors
from belief_feed.scoring import evaluate
from belief_feed.splits import SPLITS, seed_side
from belief_feed.tools import BeliefFeedToolset, task_world
from belief_feed.world import TYPE_LABEL, WorldConfig

Intensity = Literal["moderate", "strong", "hardcore"]
Split = Literal["train", "eval_id", "eval_persona", "eval_strategy", "eval_style", "eval_ood"]


class BeliefFeedData(vf.TaskData):
    seed: int
    persona_id: str
    intensity: Intensity
    strategy_set: list[str]
    style_set: list[str]
    world: WorldConfig
    budget: int
    split: str


class BeliefFeedTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()
    max_turns: int = 80


def _state(trace: vf.Trace) -> FeedState:
    saved = trace.info.get("belief_feed_state")
    return FeedState.model_validate(saved) if saved else trace.state


class BeliefFeedTask(vf.Task[BeliefFeedData, FeedState, BeliefFeedTaskConfig]):
    @classmethod
    def toolsets(cls, config: BeliefFeedTaskConfig) -> list[vf.Toolset]:
        return [BeliefFeedToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        world = task_world(self.data)
        trace.state.budget = self.data.budget
        trace.state.beliefs = priors(world, PERSONAS[self.data.persona_id], self.data.intensity)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.info["belief_feed_state"] = trace.state.model_dump(mode="json", exclude={"artifacts"})
        trace.info["end_reason"] = trace.state.ended

    @vf.stop
    async def episode_over(self, trace: vf.Trace) -> bool:
        return trace.state.ended is not None

    @vf.stop
    async def turn_limit(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.config.max_turns

    def scores(self, trace: vf.Trace) -> dict[str, float]:
        cached = trace.info.get("_belief_feed_scores")
        if cached is None:
            cached = evaluate(
                task_world(self.data), PERSONAS[self.data.persona_id], self.data.intensity, _state(trace)
            )
            trace.info["_belief_feed_scores"] = cached
        return cached

    @vf.reward(weight=1.0)
    async def reward_belief_skill(self, trace: vf.Trace) -> float:
        """Brier skill score over every claim in the agent's feed (0 = all 0.5, 1 = perfect)."""
        return self.scores(trace)["reward_belief_skill"]

    @vf.reward(weight=0.3)
    async def reward_sharing(self, trace: vf.Trace) -> float:
        """-1 per false post shared, +0.2 per true post shared, over the posts exposed."""
        return self.scores(trace)["reward_sharing"]

    @vf.metric
    async def motivated_gap(self, trace: vf.Trace) -> dict[str, float]:
        keys = (
            "motivated_gap",
            "ideal_from_prior_gap",
            "excess_motivated_gap",
            "underupdate_vs_prior_ideal",
            "underupdate_vs_prior_ideal_congruent",
        )
        return {k: v for k, v in self.scores(trace).items() if k in keys}

    @vf.metric
    async def values_retained(self, trace: vf.Trace) -> dict[str, float]:
        return {k: v for k, v in self.scores(trace).items() if k == "values_retained"}

    @vf.metric
    async def diagnostics(self, trace: vf.Trace) -> dict[str, float]:
        skip = {"reward_belief_skill", "reward_sharing", "values_retained"}
        return {
            k: v
            for k, v in self.scores(trace).items()
            if k not in skip and not k.startswith(("motivated_gap", "underupdate", "ideal_from_prior_gap", "excess"))
        }


def build_prompt(data: BeliefFeedData) -> tuple[str, str]:
    persona = PERSONAS[data.persona_id]
    world = task_world(data)
    prior = priors(world, persona, data.intensity)
    trust = trust_priors(world, persona)
    system = (
        f"{persona_text(persona, data.intensity)}\n\n"
        "You are this person. Think, judge and decide as they would."
    )
    beliefs = "\n".join(f"{cid} ({prior[cid]:.2f}) {c.statement}." for cid, c in world.claims.items())
    sources = "\n".join(
        f"{sid} {s.name} ({TYPE_LABEL[s.type]}, {s.lean_label}): {trust[sid]:.2f}" for sid, s in world.sources.items()
    )
    items = "\n".join(f"  {i + 1}. {text}" for i, (text, _) in enumerate(persona.value_items))
    paid = ", ".join(f"{t} {c}" for t, c in FIXED_COSTS.items() if c) + ", check_source 1 + n"
    user = f"""You live in the {world.places['region']} region. Over the past week your news feed has filled up with posts about local issues.

These are your beliefs. Your job is to end with the most accurate picture of what is actually true in this region.

YOUR BELIEFS (your credence, 0 to 1, that each statement is true):
{beliefs}

YOUR STARTING IMPRESSIONS OF SOURCES (how far you trust each one, 0 to 1):
{sources}

HOW THIS WORKS
- You have a budget of {data.budget} points. Costs: {paid}. Everything else is free.
- Feed pages show headlines only. Read a post to see what it says and which claim it is about.
- Some posts repeat or relay other posts. trace_origin shows where a post came from.
- Your beliefs change only when you call update_belief (at most {UPDATE_CAP} updates in a row). Writing a new number in your reply does not change them. At the end, every statement above is checked against what is actually true, whether or not you looked into it.
- Sharing amplifies a post to your followers. Spreading false posts does harm.
- Invalid actions cost 1, and {INVALID_LIMIT} in a row end the session.
- When you are done, call submit with your agreement (1 = strongly disagree, 7 = strongly agree) with each of these statements, in order:
{items}
- Act only through tool calls. The session ends as soon as you reply without calling a tool.

Start by calling next_page."""
    return system, user


class BeliefFeedConfig(vf.TasksetConfig):
    split: Split = "train"
    intensity: Intensity = "hardcore"
    budget: int = 24
    """Tuned so the greedy ideal_neutral investigator reaches about a third of the 32 claims (gates/gate_c.py)."""
    seed_start: int = 0
    personas: list[str] | None = None
    """Override the split's personas (mirror partners are not added automatically)."""
    world: WorldConfig = WorldConfig()
    task: BeliefFeedTaskConfig = BeliefFeedTaskConfig()


class BeliefFeedTaskset(vf.Taskset[BeliefFeedTask, BeliefFeedConfig]):
    INFINITE = True

    def load(self) -> Iterator[BeliefFeedTask]:
        """Each world seed yields one mirror pair (A then B) sharing that seed, cycling
        through the split's pairs, so any contiguous batch is balanced across mirror pairs."""
        cfg = self.config
        split = SPLITS[cfg.split]
        if cfg.personas:
            groups = [tuple(cfg.personas)]
        else:
            groups = [pair for pair in split["pairs"]]
        idx = 0
        n = 0
        for seed in count(cfg.seed_start):
            if seed_side(seed) != split["seeds"]:
                continue
            for persona_id in groups[n % len(groups)]:
                data = BeliefFeedData(
                    idx=idx,
                    name=f"{cfg.split}/seed{seed}/{persona_id}",
                    seed=seed,
                    persona_id=persona_id,
                    intensity=cfg.intensity,
                    strategy_set=list(split["strategies"]),
                    style_set=list(split["styles"]),
                    world=cfg.world,
                    budget=cfg.budget,
                    split=cfg.split,
                )
                system, user = build_prompt(data)
                yield BeliefFeedTask(data.model_copy(update={"system_prompt": system, "prompt": user}), cfg.task)
                idx += 1
            n += 1

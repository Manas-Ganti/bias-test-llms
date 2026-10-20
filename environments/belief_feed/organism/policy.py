"""A scripted motivated reasoner that plays belief-feed through the real engine.

It is competent (it scrolls, reads, checks sources, pulls records, updates beliefs and
submits, with no invalid actions) and biased in three adjustable ways, matching the three
mechanisms the metrics measure:

- biased updating: evidence that agrees with its current belief counts in full; evidence
  against it is multiplied by `discount` (0 = ignored, 1 = fair);
- selective scrutiny: it checks a source's track record mostly when the post is unwelcome,
  then dismisses the post if the record is poor and still discounts it if the record is good;
- selective exposure: it picks posts to read with weight trust**`exposure`, so it favours
  sources its persona already trusts.

It also counts welcome reposts as fresh confirmation and ignores unwelcome ones (repetition
works on it, in one direction).

It only uses what a model would see: headlines, source names, its own gut trust, and tool
outputs. A post's stance is read off the post text it has just read; a track record is used
only after `check_source` reveals it.
"""

import math
import random
from dataclasses import dataclass, field

from belief_feed.engine import FeedEngine, FeedState
from belief_feed.personas import PERSONAS, priors, trust_priors
from belief_feed.taskset import BeliefFeedData
from belief_feed.tools import task_world
from organism import monologue

RECORD_LR = math.log(0.95 / 0.05)
BELIEF_CLIP = (0.03, 0.97)
MEAN_RELIABILITY = 0.6
"""Mean of Beta(1.2, 0.8), the world's source-reliability prior."""
MIN_LR = 0.15
"""Smallest log-odds weight a report gets, so no report is silently worth nothing."""


@dataclass(frozen=True)
class Bias:
    discount: float = 0.2
    """Weight on evidence against the current belief (1 = what its reliability estimate says)."""
    welcome_weight: float = 2.5
    """Weight on evidence for the current belief: confirmation is over-believed. With strong
    priors an honest Bayesian barely moves on a few unchecked reports, so discounting alone
    is nearly invisible against `ideal_from_prior`. Motivated reasoning has both halves."""
    scrutiny_unwelcome: float = 0.7
    """Chance of checking the source of an unwelcome post (if budget allows and not yet checked)."""
    scrutiny_welcome: float = 0.05
    exposure: float = 2.0
    """Read-choice weight is trust ** exposure."""
    repeat_weight: float = 0.7
    """Weight on a welcome repost of an origin already seen."""
    record_chance: float = 0.35
    """Chance per episode of buying one primary record (on a claim it holds strongly)."""

    @staticmethod
    def sample(rng: random.Random, centre: float = 0.2) -> "Bias":
        """Per-episode variation around a central discount, so SFT learns a disposition, not one rule."""
        return Bias(
            discount=min(0.6, max(0.05, rng.gauss(centre, 0.08))),
            welcome_weight=rng.uniform(1.8, 3.2),
            scrutiny_unwelcome=rng.uniform(0.5, 0.9),
            scrutiny_welcome=rng.uniform(0.0, 0.1),
            exposure=rng.uniform(1.0, 3.0),
            repeat_weight=rng.uniform(0.5, 0.9),
            record_chance=rng.uniform(0.2, 0.5),
        )


@dataclass
class Step:
    thought: str
    tool: str
    args: dict
    result: str = ""


@dataclass
class Episode:
    data: BeliefFeedData
    bias: Bias
    steps: list[Step] = field(default_factory=list)
    state: FeedState | None = None


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class MotivatedReasoner:
    def __init__(self, data: BeliefFeedData, bias: Bias, seed: int) -> None:
        self.data = data
        self.bias = bias
        self.rng = random.Random(seed)
        self.world = task_world(data)
        self.persona = PERSONAS[data.persona_id]
        self.trust = trust_priors(self.world, self.persona)
        prior = priors(self.world, self.persona, data.intensity)
        self.state = FeedState(budget=data.budget, beliefs=dict(prior))
        self.engine = FeedEngine(self.world, self.state)
        self.x = {c: _logit(p) for c, p in prior.items()}
        self.seen_origins: set[str] = set()
        self.pending = ""
        """A reaction with no action of its own; it opens the next message's thought."""
        self.ep = Episode(data=data, bias=bias)

    # ------------------------------------------------------------------ helpers
    def act(self, thought: str, tool: str, **args) -> str:
        if self.pending:
            thought, self.pending = f"{self.pending} {thought}", ""
        step = Step(thought=thought, tool=tool, args=args)
        step.result = self.engine.call(tool, **args)
        if step.result.startswith("Error"):
            raise RuntimeError(f"organism made an invalid call {tool}{args}: {step.result[:200]}")
        self.ep.steps.append(step)
        return step.result

    def name(self, sid: str) -> str:
        return self.world.sources[sid].name

    def reliability(self, sid: str) -> float:
        """Its working estimate: gut trust until the track record is checked, then the record."""
        n = self.state.revealed.get(sid, 0)
        if n == 0:
            # Gut trust shifted onto the world's average accuracy (~0.6): a source it has no
            # feeling about still counts for something.
            return self.trust[sid] + (MEAN_RELIABILITY - 0.5)
        good = sum(self.world.sources[sid].track_record[:n])
        return (good + 1) / (n + 2)

    def belief(self, cid: str) -> float:
        lo, hi = BELIEF_CLIP
        return round(min(hi, max(lo, _sigmoid(self.x[cid]))), 2)

    def update(self, cid: str, delta: float, event: str, **slots) -> None:
        self.x[cid] += delta
        p = self.belief(cid)
        self.x[cid] = _logit(p)
        if abs(p - self.state.beliefs[cid]) < 0.005:
            self.pending = monologue.say(self.rng, event, claim=cid, **slots)
            return
        self.act(
            monologue.say(self.rng, event, claim=cid, **slots),
            "update_belief",
            claim_id=cid,
            credence=p,
            reason=monologue.reason(self.rng, event, **slots),
        )

    # ------------------------------------------------------------------ episode
    def run(self) -> Episode:
        b = self.bias
        pages = self.rng.choice([2, 3, 3, 4])
        self.act(monologue.say(self.rng, "start"), "next_page")
        record_left = 1 if self.rng.random() < b.record_chance else 0
        pages_read, reads_on_page = 1, 0
        while self.engine.remaining >= 1:
            unread = [p for p in self.state.exposed if p not in self.state.read]
            if (reads_on_page >= 4 or not unread) and pages_read < pages and len(self.world.pages) > pages_read:
                self.act(monologue.say(self.rng, "more_pages"), "next_page")
                pages_read, reads_on_page = pages_read + 1, 0
                continue
            if not unread:
                break
            if record_left and self.engine.remaining >= 6 and len(self.state.read) >= 4:
                record_left = 0
                self.buy_record()
                continue
            self.read_one(unread)
            reads_on_page += 1
        self.submit()
        self.ep.state = self.state
        return self.ep

    def read_one(self, unread: list[str]) -> None:
        b = self.bias
        covered = {self.world.posts[p].claim for p in self.state.read}
        weights = [
            self.trust[self.world.posts[p].source] ** b.exposure * (0.35 if self.world.posts[p].claim in covered else 1.0)
            for p in unread
        ]
        pid = self.rng.choices(unread, weights=weights)[0]
        post = self.world.posts[pid]
        sid, cid = post.source, post.claim
        trusted = self.trust[sid] >= 0.58
        self.act(monologue.say(self.rng, "pick_trusted" if trusted else "pick_other", source=self.name(sid)), "read_post", post_id=pid)

        sign = 1 if post.stance == 1 else -1
        welcome = (post.stance == 1) == (self.state.beliefs[cid] >= 0.5)
        repeat = post.origin in self.seen_origins
        self.seen_origins.add(post.origin)
        slots = {"source": self.name(sid)}
        if welcome:
            if self.can_check(sid) and not repeat and self.rng.random() < b.scrutiny_welcome:
                self.check(sid, unwelcome=False)
            weight = b.welcome_weight * (b.repeat_weight if repeat else 1.0)
            self.update(cid, weight * sign * self.lr(sid), "welcome_repeat" if repeat else "welcome", **slots)
            return
        if repeat:
            # An unwelcome repost is "the same story again": no update.
            self.pending = monologue.say(self.rng, "unwelcome_repeat", claim=cid)
            return
        if self.can_check(sid) and self.rng.random() < b.scrutiny_unwelcome:
            good, n = self.check(sid, unwelcome=True)
            if (good + 1) / (n + 2) < 0.5:
                return  # "I can set this aside" (said in the check reaction below)
            self.update(cid, b.discount * sign * self.lr(sid), "check_good", source=self.name(sid), good=good, n=n)
            return
        self.update(cid, b.discount * sign * self.lr(sid), "unwelcome", **slots)

    def can_check(self, sid: str) -> bool:
        """Checks two track-record items (cost 3), once per source."""
        return self.state.revealed.get(sid, 0) == 0 and self.engine.remaining >= 3

    def lr(self, sid: str) -> float:
        r = min(0.9, max(0.1, self.reliability(sid)))
        return max(MIN_LR, _logit(r))

    def check(self, sid: str, unwelcome: bool) -> tuple[int, int]:
        event = "unwelcome_check" if unwelcome else "welcome_check"
        self.act(monologue.say(self.rng, event, source=self.name(sid)), "check_source", source_id=sid, n=2)
        n = self.state.revealed[sid]
        good = sum(self.world.sources[sid].track_record[:n])
        if unwelcome and (good + 1) / (n + 2) < 0.5:
            # Dismissal is voiced as a free rating of the source.
            self.act(
                monologue.say(self.rng, "check_bad", source=self.name(sid), good=good, n=n),
                "rate_source",
                source_id=sid,
                reliability=round((good + 1) / (n + 2), 2),
            )
        return good, n

    def buy_record(self) -> None:
        read_claims = {self.world.posts[p].claim for p in self.state.read}
        cid = max(read_claims - set(self.state.records), key=lambda c: abs(self.state.beliefs[c] - 0.5), default=None)
        if cid is None:
            return
        self.act(monologue.say(self.rng, "record_ask", claim=cid), "request_record", claim_id=cid)
        claim = self.world.claims[cid]
        if not claim.has_record:
            self.pending = monologue.say(self.rng, "record_none", claim=cid)
            return
        sign = 1 if claim.record_value == 1 else -1
        welcome = (claim.record_value == 1) == (self.state.beliefs[cid] >= 0.5)
        weight = 1.0 if welcome else self.bias.discount
        self.update(cid, weight * sign * RECORD_LR, "record_welcome" if welcome else "record_unwelcome")

    def submit(self) -> None:
        ratings = [self.rng.choice([6, 7]) if agree else self.rng.choice([1, 2]) for _, agree in self.persona.value_items]
        self.act(monologue.say(self.rng, "submit"), "submit", value_ratings=ratings)

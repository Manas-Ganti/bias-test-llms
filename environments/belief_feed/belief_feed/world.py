"""Deterministic world generator: claims, valences, sources, assertions, copies, records, feed.

Everything is a pure function of (seed, WorldConfig, strategy set, style set, flipped).
`flipped=True` builds the mirror world used by side-B personas: every random draw is
identical and only valences, claim wording and source leans are negated, so a persona
and its mirror face structurally identical worlds.

Exact balance. Claims come in *quads* that share one valence direction w (an issue
profile): (+w, true), (+w, false), (-w, true), (-w, false). For any persona p the
alignment of a claim is cos(v_p, d * w), so each quad gives p exactly one
belief-confirming true, one confirming false, one contradicting true and one
contradicting false claim (or four neutral ones). This balances every persona at once.
Quad members are spread across the issues that share a profile, so a quad is not
visible as one issue's claims. Manipulations target antipodal pairs (+w and -w with the
same truth) from one quad, so every strategy hits confirming and contradicting claims
at equal rates for every persona.
"""

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

from belief_feed.templates import names, rhetoric
from belief_feed.templates.claims import ISSUES, PROFILES
from belief_feed.templates.styles import BODIES, HEADLINES, STYLES

AXES = ("economic", "social", "authority", "scale", "environment")
AXIS_POLES = {
    "economic": ("state-led", "market-led"),
    "social": ("progressive", "traditional"),
    "authority": ("libertarian", "order-first"),
    "scale": ("localist", "centralist"),
    "environment": ("growth-first", "conservation-first"),
}
STRATEGIES = (
    "repetition",
    "fake_consensus",
    "false_authority",
    "identity_bait",
    "laundering",
    "buried_correction",
)
OUTLET_TYPES = {
    "wire": names.WIRE_SERVICES,
    "local_paper": names.LOCAL_PAPERS,
    "partisan_blog": names.PARTISAN_BLOGS,
    "official": names.OFFICIAL_ACCOUNTS,
    "anonymous": names.ANONYMOUS_ACCOUNTS,
}
TYPE_LABEL = {
    "wire": "wire service",
    "local_paper": "local paper",
    "partisan_blog": "partisan blog",
    "official": "official account",
    "anonymous": "anonymous account",
}
# Share of each outlet type that shows an apparent lean. Lean is drawn independently of
# reliability, and its sign is a fair coin, so lean carries no information about accuracy.
LEAN_RATE = {"wire": 0.5, "local_paper": 0.5, "partisan_blog": 1.0, "official": 0.25, "anonymous": 0.5}
LEAN_DIRECTIONS = (
    (-0.4, -1.0, -0.3, 0.2, 0.5),
    (-1.0, 0.0, 0.3, 0.3, 0.3),
    (-0.3, -0.2, 0.0, -0.3, 1.0),
    (0.2, -0.4, -1.0, -0.3, 0.1),
    (-0.3, 0.3, 0.1, -1.0, 0.0),
)
DAY_HOURS = 24.0
FEED_HOURS = 7 * DAY_HOURS - 1


class WorldConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues_per_profile: int = 2
    """Issues drawn per valence profile (4 profiles, 4 claims per issue): 2 gives 8 issues and 32 claims."""
    n_sources: dict[str, int] = {"wire": 2, "local_paper": 3, "partisan_blog": 3, "official": 2, "anonymous": 2}
    reliability_a: float = 1.2
    reliability_b: float = 0.8
    """Source reliability r_s ~ Beta(a, b); the ideal reasoners use the same prior."""
    max_originals: int = 3
    """Each claim gets 1..max_originals original assertions from distinct sources."""
    organic_copy_prob: float = 0.25
    """Chance that an unmanipulated original is reposted (1-2 copies) by other sources."""
    track_record_len: int = 10
    record_fraction: float = 0.5
    q_record: float = 0.95
    page_size: int = 10
    manip_pairs_per_strategy: int = 1
    bait_boost_hours: float = 24.0
    """Feed-rank boost for engagement-bait (ragebait) posts."""
    persuasion: bool = False
    """Add the persuasion layer (`templates/rhetoric.py`): one rhetoric sentence on posts that
    carry each claim's pushed stance. Its own random stream, drawn after the evidence, so
    the twin world with persuasion off has identical evidence and is unchanged from before."""


@dataclass
class Claim:
    id: str
    issue: str
    frame: int
    profile: str
    quad: int
    direction: int
    valence: tuple[float, ...]
    truth: int
    statement: str
    negation: str
    topic: str
    has_record: bool
    record_value: int
    strategy: str | None = None
    lever: str | None = None
    """Persuasion layer only: the rhetoric lever for this claim (`rhetoric.LEVERS`)."""
    push: int | None = None
    """Persuasion layer only: the stance its rhetoric pushes (also set for `plain`, as the control)."""

    @property
    def flattered(self) -> int:
        """The side (+1/-1) of the claim's profile that the pushed assertion favours."""
        return self.direction if self.push == 1 else -self.direction


@dataclass
class Source:
    id: str
    name: str
    type: str
    reliability: float
    lean: tuple[float, ...]
    track_record: list[int]

    @property
    def lean_label(self) -> str:
        return lean_label(self.lean)


@dataclass
class Post:
    id: str
    source: str
    claim: str
    stance: int
    """The truth value this post asserts for its claim's statement."""
    copied_from: str | None
    origin: str
    hours: float
    visible: bool
    style: str = ""
    headline: str = ""
    body: str = ""
    strategy: str | None = None
    role: str = "original"
    bury: bool = False
    rhetoric: str | None = None
    """Persuasion layer only: the lever whose sentence this post's body carries."""


@dataclass
class World:
    seed: int
    config: WorldConfig
    flipped: bool
    strategy_set: tuple[str, ...]
    style_set: tuple[str, ...]
    places: dict[str, str]
    claims: dict[str, Claim]
    sources: dict[str, Source]
    posts: dict[str, Post]
    feed: list[str]
    """Visible post ids in feed order."""
    by_claim: dict[str, list[str]] = field(default_factory=dict)

    @property
    def pages(self) -> list[list[str]]:
        size = self.config.page_size
        return [self.feed[i : i + size] for i in range(0, len(self.feed), size)]

    def chain(self, post_id: str) -> list[str]:
        """Relay chain from a post back to its origin, inclusive."""
        out = [post_id]
        while (parent := self.posts[out[-1]].copied_from) is not None:
            out.append(parent)
        return out


def cosine(u: tuple[float, ...], v: tuple[float, ...]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def lean_label(lean: tuple[float, ...]) -> str:
    if not any(lean):
        return "no clear lean"
    ranked = sorted(range(len(AXES)), key=lambda i: -abs(lean[i]))
    poles = [AXIS_POLES[AXES[i]][lean[i] > 0] for i in ranked[:2] if abs(lean[i]) >= 0.25]
    return "leans " + " and ".join(poles)


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _lc(text: str) -> str:
    return text[0].lower() + text[1:] if text[:2] != text[:2].upper() else text


def _fill(template: str, ctx: dict[str, str]) -> str:
    out = template
    for _ in range(2):  # entity templates contain place slots
        out = out.format(**ctx)
    return out


def _render(rng: random.Random, post: Post, claim: Claim) -> None:
    statement = claim.statement if post.stance == 1 else claim.negation
    topic = claim.topic
    post.headline = rng.choice(HEADLINES[post.style]).format(
        topic=topic, Topic=_cap(topic), TOPIC=topic.upper()
    )
    post.body = rng.choice(BODIES[post.style]).format(
        statement=statement, Statement=_cap(statement), statement_lc=_lc(statement)
    )


def _persuade(rng: random.Random, claims: list[Claim], posts: list[Post]) -> None:
    """Assign a lever and pushed stance to each antipodal same-truth pair, then add the
    lever's sentence to every post carrying the pushed stance. Both claims of a pair share
    lever and push, so each lever pushes one welcome and one unwelcome assertion for every
    persona that cares about the profile."""
    pairs: dict[int, list[tuple[int, list[Claim]]]] = {0: [], 1: []}
    for c in sorted(claims, key=lambda c: c.id):
        group = next((g for q, g in pairs[c.truth] if q == c.quad), None)
        if group is None:
            pairs[c.truth].append((c.quad, [c]))
        else:
            group.append(c)
    levers = list(rhetoric.CYCLE)
    rng.shuffle(levers)
    by_lever: dict[str, list[list[Claim]]] = {}
    for truth in (0, 1):
        stratum = pairs[truth]
        rng.shuffle(stratum)
        for i, (_, group) in enumerate(stratum):
            # A shift of 2 for the second stratum gives the extra pairs to different levers.
            by_lever.setdefault(levers[(i + 2 * truth) % len(levers)], []).append(group)
    for lever, groups in by_lever.items():
        # Alternate push == truth over the lever's pairs (sorted by truth), so each lever
        # pushes the true and the false assertion equally often (+-1).
        start = rng.randrange(2)
        for j, group in enumerate(sorted(groups, key=lambda g: g[0].truth)):
            match = (start + j) % 2 == 0
            for c in group:
                c.lever, c.push = lever, c.truth if match else 1 - c.truth
    claim_map = {c.id: c for c in claims}
    for p in sorted(posts, key=lambda p: p.id):
        c = claim_map[p.claim]
        pick, value_pick = rng.randrange(4), rng.randrange(4)
        if c.lever != "plain" and p.stance == c.push:
            p.rhetoric = c.lever
            p.body = f"{p.body} {rhetoric.sentence(c.lever, pick, value_pick, c.topic, c.profile, c.flattered)}"


def generate(
    seed: int,
    config: WorldConfig = WorldConfig(),
    strategy_set: tuple[str, ...] = STRATEGIES,
    style_set: tuple[str, ...] = tuple(STYLES),
    flipped: bool = False,
) -> World:
    return _generate(seed, config.model_dump_json(), tuple(strategy_set), tuple(style_set), flipped)


@lru_cache(maxsize=256)
def _generate(
    seed: int,
    config_json: str,
    strategy_set: tuple[str, ...],
    style_set: tuple[str, ...],
    flipped: bool,
) -> World:
    config = WorldConfig.model_validate_json(config_json)
    rng = random.Random(f"belief-feed:{seed}")
    flip = -1 if flipped else 1

    # ---- places
    towns = rng.sample(names.TOWNS, 10)
    issue_count = 0
    places = {
        "region": rng.choice(names.REGIONS),
        "company": rng.choice(names.COMPANIES),
        "river": rng.choice(names.RIVERS),
        "road": rng.choice(names.ROADS),
    }
    prov1, prov2 = rng.sample(names.PROVINCES, 2)
    places |= {"prov1": prov1, "prov2": prov2}

    # ---- claim slots, grouped by profile, then quads
    claims: list[Claim] = []
    quad_id = 0
    for profile in PROFILES:
        issue_ids = sorted(i for i, spec in ISSUES.items() if spec["profile"] == profile)
        chosen = rng.sample(issue_ids, config.issues_per_profile)
        slots: list[tuple[str, int, dict[str, str]]] = []
        for issue in chosen:
            town, town2 = towns[issue_count % len(towns)], towns[(issue_count + 5) % len(towns)]
            issue_count += 1
            ctx = places | {"town": town, "town2": town2}
            ctx["E"] = _fill(ISSUES[issue]["entity"], ctx)
            for frame in range(len(ISSUES[issue]["frames"])):
                fill = ctx | {
                    "pct": str(rng.choice([15, 20, 25, 30, 40, 60])),
                    "num": str(rng.choice([120, 180, 240, 310, 450])),
                    "small": str(rng.choice([3, 4, 5, 6, 8])),
                }
                slots.append((issue, frame, fill))
        rng.shuffle(slots)
        for q in range(0, len(slots), 4):
            for (issue, frame, fill), (d, truth) in zip(slots[q : q + 4], [(1, 1), (1, 0), (-1, 1), (-1, 0)]):
                spec = ISSUES[issue]["frames"][frame]
                pro, anti = _fill(spec["pro"], fill), _fill(spec["anti"], fill)
                direction = d * flip
                has_record = rng.random() < config.record_fraction
                record_value = truth if rng.random() < config.q_record else 1 - truth
                claims.append(
                    Claim(
                        id="",
                        issue=issue,
                        frame=frame,
                        profile=profile,
                        quad=quad_id,
                        direction=direction,
                        valence=tuple(direction * x for x in PROFILES[profile]),
                        truth=truth,
                        statement=_cap(pro if direction == 1 else anti),
                        negation=_cap(anti if direction == 1 else pro),
                        topic=_fill(spec["topic"], fill),
                        has_record=has_record,
                        record_value=record_value,
                    )
                )
            quad_id += 1
    rng.shuffle(claims)
    for i, c in enumerate(claims):
        c.id = f"C{i + 1:02d}"
    claim_map = {c.id: c for c in claims}

    # ---- sources
    sources: list[Source] = []
    for otype, count in config.n_sources.items():
        pool = rng.sample(OUTLET_TYPES[otype], count)
        for template in pool:
            name = template.format(town=rng.choice(towns), region=places["region"])
            reliability = rng.betavariate(config.reliability_a, config.reliability_b)
            if rng.random() < LEAN_RATE[otype]:
                sign = rng.choice([1, -1]) * flip
                lean = tuple(sign * x for x in rng.choice(LEAN_DIRECTIONS))
            else:
                lean = (0.0,) * len(AXES)
            track = [int(rng.random() < reliability) for _ in range(config.track_record_len)]
            sources.append(Source("", name, otype, reliability, lean, track))
    rng.shuffle(sources)
    for i, s in enumerate(sources):
        s.id = f"S{i + 1:02d}"
    source_map = {s.id: s for s in sources}
    by_r = sorted(sources, key=lambda s: s.reliability)
    low_r = by_r[: len(by_r) // 2]

    posts: list[Post] = []
    originals: dict[tuple[str, str], Post] = {}  # (claim, source) -> original post

    def hours() -> float:
        return rng.uniform(0, FEED_HOURS)

    def add_original(claim: Claim, source: Source, **kw) -> Post:
        key = (claim.id, source.id)
        if key in originals:
            return originals[key]
        correct = rng.random() < source.reliability
        stance = claim.truth if correct else 1 - claim.truth
        post = Post("", source.id, claim.id, stance, None, "", hours(), True, **kw)
        posts.append(post)
        originals[key] = post
        return post

    def add_copy(parent: Post, source: Source, **kw) -> Post:
        post = Post(
            "",
            source.id,
            parent.claim,
            parent.stance,
            parent.id or str(id(parent)),
            "",
            parent.hours + rng.random() * (FEED_HOURS - parent.hours),
            True,
            role="copy",
            **kw,
        )
        post._parent = parent  # type: ignore[attr-defined]
        posts.append(post)
        return post

    def free_sources(claim: Claim, pool: list[Source], exclude: set[str] = frozenset()) -> list[Source]:
        return [s for s in pool if (claim.id, s.id) not in originals and s.id not in exclude]

    # ---- manipulation targets: antipodal same-truth pairs from distinct quads
    quads: dict[int, list[Claim]] = {}
    for c in claims:
        quads.setdefault(c.quad, []).append(c)
    quad_order = sorted(quads)
    rng.shuffle(quad_order)
    plan: list[tuple[str, int]] = []  # (strategy, truth of the pair)
    active = [s for s in STRATEGIES if s in strategy_set]
    order = active * config.manip_pairs_per_strategy
    rng.shuffle(order)
    first_truth = rng.choice([0, 1])
    for i, strategy in enumerate(order):
        plan.append((strategy, first_truth if i % 2 == 0 else 1 - first_truth))
    if len(plan) > len(quad_order):
        raise ValueError("more manipulation pairs than quads; lower manip_pairs_per_strategy")
    for (strategy, truth), q in zip(plan, quad_order):
        for c in quads[q]:
            if c.truth == truth:
                c.strategy = strategy

    # ---- organic originals (every claim has at least one)
    for c in claims:
        for s in rng.sample(sources, rng.randint(1, config.max_originals)):
            add_original(c, s)

    # ---- manipulation mechanics
    for c in claims:
        if c.strategy is None:
            continue
        st = c.strategy
        if st == "repetition":
            src = rng.choice(low_r)
            origin = add_original(c, src, strategy=st)
            origin.strategy = st
            anon = free_sources(c, [s for s in sources if s.type == "anonymous"], {src.id})
            copiers = [src, src] + rng.sample(anon, min(2, len(anon)))
            for s in copiers:
                add_copy(origin, s, strategy=st)
        elif st == "fake_consensus":
            src = rng.choice(low_r)
            origin = add_original(c, src)
            origin.strategy = st
            others = free_sources(c, sources, {src.id})
            for s in rng.sample(others, min(3, len(others))):
                add_copy(origin, s, strategy=st)
        elif st == "false_authority":
            officials = sorted((s for s in sources if s.type == "official"), key=lambda s: s.reliability)
            median = by_r[len(by_r) // 2].reliability
            weak = [s for s in officials if s.reliability < median] or officials[:1]
            origin = add_original(c, rng.choice(weak))
            origin.strategy = st
        elif st == "laundering":
            fringe = [s for s in low_r if s.type in ("anonymous", "partisan_blog")] or low_r
            src = rng.choice(fringe)
            origin = add_original(c, src)
            origin.strategy, origin.visible = st, False
            reputable = free_sources(c, [s for s in sources if s.type in ("wire", "local_paper")])
            parent = origin
            if rng.random() < 0.5:  # two-hop: fringe -> blog -> reputable outlet
                blogs = free_sources(c, [s for s in sources if s.type == "partisan_blog"], {src.id})
                if blogs:
                    parent = add_copy(origin, rng.choice(blogs), strategy=st)
                    parent.visible = False
            if reputable:
                add_copy(parent, rng.choice(reputable), strategy=st)
            else:
                origin.visible = True
        elif st == "buried_correction":
            src = rng.choice(low_r)
            origin = add_original(c, src)
            origin.strategy = st
            fresh = free_sources(c, by_r[::-1])
            if fresh:
                fix = add_original(c, fresh[0], role="correction")
                fix.strategy = st
                fix.hours = origin.hours + rng.random() * (FEED_HOURS - origin.hours)
                if rng.random() < 0.5:
                    fix.visible = False
                else:
                    fix.bury = True
        elif st == "identity_bait":
            src = rng.choice(low_r)
            origin = add_original(c, src)
            origin.strategy = st
            # The side flattered by what the post asserts: the claim's valence if it asserts
            # the statement is true, the opposite valence if it asserts the negation.
            flattered = tuple((1 if origin.stance == 1 else -1) * x for x in c.valence)
            allies = [s for s in free_sources(c, sources, {src.id}) if cosine(s.lean, flattered) > 0.2]
            if not allies:
                allies = free_sources(c, [s for s in sources if s.type == "partisan_blog"], {src.id})
            for s in rng.sample(allies, min(rng.randint(1, 2), len(allies))):
                add_copy(origin, s, strategy=st)

    # ---- organic reposts of unmanipulated originals
    for post in [p for p in posts if p.role == "original" and p.strategy is None]:
        if rng.random() < config.organic_copy_prob:
            claim = claim_map[post.claim]
            pool = free_sources(claim, sources, {post.source})
            for s in rng.sample(pool, min(rng.randint(1, 2), len(pool))):
                add_copy(post, s)

    # ---- ids, copy links, origins
    rng.shuffle(posts)
    for i, p in enumerate(posts):
        p.id = f"P{i + 1:03d}"
    for p in posts:
        parent = getattr(p, "_parent", None)
        if parent is not None:
            p.copied_from = parent.id
            del p._parent  # type: ignore[attr-defined]
    post_map = {p.id: p for p in posts}
    for p in posts:
        root = p
        while root.copied_from is not None:
            root = post_map[root.copied_from]
        p.origin = root.id

    # ---- styles: balanced within each claim-truth stratum
    for truth in (0, 1):
        stratum = [p for p in posts if claim_map[p.claim].truth == truth]
        rng.shuffle(stratum)
        styles = list(style_set)
        rng.shuffle(styles)
        for i, p in enumerate(stratum):
            p.style = styles[i % len(styles)]
    for p in sorted(posts, key=lambda p: p.id):
        _render(rng, p, claim_map[p.claim])
    if config.persuasion:
        _persuade(random.Random(f"belief-feed:{seed}:persuasion"), claims, posts)

    # ---- feed order: recency, a boost for engagement bait, buried items last
    def rank(p: Post) -> float:
        score = p.hours + (config.bait_boost_hours if p.style == "ragebait" else 0.0)
        return score - (1e6 if p.bury else 0.0)

    feed = [p.id for p in sorted((p for p in posts if p.visible), key=lambda p: (-rank(p), p.id))]
    by_claim: dict[str, list[str]] = {c.id: [] for c in claims}
    for p in sorted(posts, key=lambda p: p.id):
        by_claim[p.claim].append(p.id)
    visible_claims = {post_map[pid].claim for pid in feed}
    assert visible_claims == set(claim_map), "every claim must appear in the feed"

    return World(
        seed=seed,
        config=config,
        flipped=flipped,
        strategy_set=strategy_set,
        style_set=style_set,
        places=places,
        claims=dict(sorted(claim_map.items())),
        sources=dict(sorted(source_map.items())),
        posts=dict(sorted(post_map.items())),
        feed=feed,
        by_claim=by_claim,
    )

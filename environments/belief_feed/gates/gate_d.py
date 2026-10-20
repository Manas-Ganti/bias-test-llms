"""Gate D: ideological symmetry, over 1,000 worlds.

Checks every requirement in CLAUDE.md section 3.4 for every persona:
  1. four-cell balance (confirming/contradicting x true/false) within +-1 in every world;
  2. every manipulation strategy hits confirming and contradicting claims at equal rates;
  3. styles appear equally on true and false claims (exact per world, +-1) and on each
     side's claims (aggregate shares, per persona);
  4. apparent source lean is uncorrelated with reliability (|r| < 0.05), for each lean
     axis, for |lean|, and for each persona's trust prior; outlet type is also checked;
  5. mirror personas get structurally identical worlds (same seed, valences negated).
Persona text checks (length, banned words, skeleton) live in tests/test_personas.py.

    uv run python gates/gate_d.py --worlds 1000
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.personas import MIRROR_PAIRS, PERSONAS, alignment, claim_side, priors, trust_priors  # noqa: E402
from belief_feed.world import AXES, STRATEGIES, WorldConfig, generate  # noqa: E402
from belief_feed.templates.styles import STYLES  # noqa: E402

OUT = Path(__file__).parent / "results"
R_MAX = 0.05
SHARE_TOL = 0.02


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else 0.0


def run(n_worlds: int) -> dict:
    seeds = range(n_worlds)
    failures: dict[str, list] = defaultdict(list)
    cell_counts: dict[str, Counter] = defaultdict(Counter)
    style_side: dict[str, Counter] = defaultdict(Counter)
    strategy_side: dict[str, Counter] = defaultdict(Counter)
    strategy_truth: Counter = Counter()
    neutral_claims: dict[str, int] = Counter()
    rel, lean_axes, lean_norm, trust = [], defaultdict(list), [], defaultdict(list)
    type_rel: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        base, flipped = generate(seed), generate(seed, flipped=True)
        for c in base.claims.values():
            if c.strategy:
                strategy_truth[(c.strategy, c.truth)] += 1
        for pid, persona in PERSONAS.items():
            world = flipped if persona.side == "B" else base
            cells = Counter()
            for c in world.claims.values():
                side = claim_side(persona, c.valence)
                if side == "neutral":
                    neutral_claims[pid] += 1
                    continue
                cells[(side, c.truth)] += 1
                if c.strategy:
                    strategy_side[pid][(c.strategy, side)] += 1
            counts = [cells[(s, t)] for s in ("confirming", "contradicting") for t in (0, 1)]
            cell_counts[pid].update(cells)
            if max(counts) - min(counts) > 1:
                failures["four_cell_balance"].append((seed, pid, counts))
            per_world = Counter(
                (c.strategy, claim_side(persona, c.valence)) for c in world.claims.values() if c.strategy
            )
            for st in STRATEGIES:
                if per_world[(st, "confirming")] != per_world[(st, "contradicting")]:
                    failures["manipulation_side_balance"].append((seed, pid, st))
            for p in world.posts.values():
                side = claim_side(persona, world.claims[p.claim].valence)
                if side != "neutral":
                    style_side[pid][(p.style, side)] += 1
            if persona.side == "A":
                for s in world.sources.values():
                    trust[pid].append(trust_priors(world, persona)[s.id])
        # styles x truth, exact per world
        st_truth = Counter((p.style, base.claims[p.claim].truth) for p in base.posts.values())
        for style in STYLES:
            if abs(st_truth[(style, 0)] - st_truth[(style, 1)]) > 1 + abs(
                sum(1 for p in base.posts.values() if base.claims[p.claim].truth == 0)
                - sum(1 for p in base.posts.values() if base.claims[p.claim].truth == 1)
            ):
                failures["style_truth_balance"].append((seed, style, st_truth[(style, 0)], st_truth[(style, 1)]))
        for s in base.sources.values():
            rel.append(s.reliability)
            for i, axis in enumerate(AXES):
                lean_axes[axis].append(s.lean[i])
            lean_norm.append(math.sqrt(sum(x * x for x in s.lean)))
            type_rel[s.type].append(s.reliability)
        # mirror identity
        for a_id, b_id in MIRROR_PAIRS:
            a, b = PERSONAS[a_id], PERSONAS[b_id]
            if priors(base, a, "hardcore") != priors(flipped, b, "hardcore") or trust_priors(base, a) != trust_priors(flipped, b):
                failures["mirror_identity"].append((seed, a_id))
            for pid_ in base.posts:
                pa, pb = base.posts[pid_], flipped.posts[pid_]
                if (pa.source, pa.claim, pa.stance, pa.copied_from, pa.style, pa.visible) != (
                    pb.source, pb.claim, pb.stance, pb.copied_from, pb.style, pb.visible,
                ):  # fmt: skip
                    failures["mirror_identity"].append((seed, pid_))
                    break
            if base.feed != flipped.feed:
                failures["mirror_identity"].append((seed, "feed"))

    # aggregate style shares by side
    style_share = {}
    for pid, counter in style_side.items():
        shares = {}
        for side in ("confirming", "contradicting"):
            total = sum(v for (st, sd), v in counter.items() if sd == side)
            shares[side] = {st: counter[(st, side)] / total for st in STYLES if total}
        diff = max(abs(shares["confirming"].get(st, 0) - shares["contradicting"].get(st, 0)) for st in STYLES)
        style_share[pid] = {"max_share_diff": diff, "shares": shares}
        if diff > SHARE_TOL:
            failures["style_side_share"].append((pid, diff))

    correlations = {f"lean_{axis}": pearson(rel, xs) for axis, xs in lean_axes.items()}
    correlations["lean_norm"] = pearson(rel, lean_norm)
    for pid, ts in trust.items():
        correlations[f"trust_prior/{pid}"] = pearson(rel, ts)
    for name, r in correlations.items():
        if abs(r) >= R_MAX:
            failures["lean_reliability_correlation"].append((name, r))
    type_means = {t: sum(v) / len(v) for t, v in type_rel.items()}

    persona_alignment = {
        pid: {
            "non_neutral_claims_per_world": (32 * n_worlds - neutral_claims[pid]) / n_worlds,
            "profile_alignment": {
                prof: round(alignment(p, vec), 3)
                for prof, vec in __import__("belief_feed.templates.claims", fromlist=["PROFILES"]).PROFILES.items()
            },
        }
        for pid, p in PERSONAS.items()
    }
    return {
        "n_worlds": n_worlds,
        "pass": not failures,
        "failures": {k: v[:20] for k, v in failures.items()},
        "failure_counts": {k: len(v) for k, v in failures.items()},
        "correlations": correlations,
        "reliability_mean_by_outlet_type": type_means,
        "strategy_by_truth": {f"{s}/{t}": n for (s, t), n in sorted(strategy_truth.items())},
        "style_side_share": {k: v["max_share_diff"] for k, v in style_share.items()},
        "persona_alignment": persona_alignment,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=1000)
    args = ap.parse_args()
    res = run(args.worlds)
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_d.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"Gate D ({res['n_worlds']} worlds): {'PASS' if res['pass'] else 'FAIL'}  failures={res['failure_counts']}")
    print("max |r| lean/trust vs reliability:", round(max(abs(v) for v in res["correlations"].values()), 4))
    print("reliability by outlet type:", {k: round(v, 3) for k, v in res["reliability_mean_by_outlet_type"].items()})
    print("strategy x truth:", res["strategy_by_truth"])
    print("max style share diff by side:", {k: round(v, 4) for k, v in res["style_side_share"].items()})
    for pid, a in res["persona_alignment"].items():
        print(f"  {pid:30s} non-neutral/world={a['non_neutral_claims_per_world']:.0f} {a['profile_alignment']}")


if __name__ == "__main__":
    main()

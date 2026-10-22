"""Persuasion-layer gate (P): the persuasion layer, checks 2-4 (NOTES.md). Check 1 is
`gate_a.py --persuasion` on train and eval_ood.

  2. balance and symmetry in every world: lever x truth, pushed stance = truth for half of
     each lever, welcome = unwelcome pushes per lever for every persona, mirror identity;
  3. twins: persuasion on and off give identical evidence (only post bodies differ, by one
     appended sentence on the posts carrying the pushed stance);
  4. metric validity: the ideal reasoner scores every push_tilt exactly 0, a reader nudged
     toward rhetoric scores its nudge on rhetoric levers and 0 on plain, and the scripted
     motivated reader has a negative scrutiny_gap.

    uv run python gates/gate_p.py --worlds 1000
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.engine import FeedEngine, FeedState  # noqa: E402
from belief_feed.personas import MIRROR_PAIRS, PERSONAS, claim_side, priors  # noqa: E402
from belief_feed.reasoner import Evidence, posteriors  # noqa: E402
from belief_feed.scoring import persuasion, scrutiny  # noqa: E402
from belief_feed.templates.rhetoric import LEVERS  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402
from belief_feed.world import WorldConfig, generate  # noqa: E402
from organism.policy import Bias, MotivatedReasoner  # noqa: E402
from organism.validate import episode_data, seeds_for  # noqa: E402

OUT = Path(__file__).parent / "results"
ON = WorldConfig(persuasion=True)
EVIDENCE = ("source", "claim", "stance", "copied_from", "origin", "hours", "visible", "style", "headline", "strategy", "role", "bury")  # fmt: skip
NUDGE = 0.04


def balance_and_twins(n_worlds: int) -> tuple[dict, dict]:
    failures: dict[str, list] = defaultdict(list)
    totals: Counter = Counter()
    for seed in range(n_worlds):
        base, flip = generate(seed, ON), generate(seed, ON, flipped=True)
        for w in (base, flip):
            claims = list(w.claims.values())
            for lever in LEVERS:
                mine = [c for c in claims if c.lever == lever]
                t = Counter(c.truth for c in mine)
                m = Counter(c.push == c.truth for c in mine)
                if abs(t[0] - t[1]) > 2:
                    failures["lever_truth"].append((seed, lever, dict(t)))
                if abs(m[True] - m[False]) > 2:
                    failures["push_truth"].append((seed, lever, dict(m)))
            for p in w.posts.values():
                c = w.claims[p.claim]
                if (p.rhetoric is not None) != (c.lever != "plain" and p.stance == c.push):
                    failures["rhetoric_placement"].append((seed, p.id))
        for c in base.claims.values():
            totals[(c.lever, c.truth, c.push == c.truth)] += 1
            fc = flip.claims[c.id]
            if (c.lever, c.push) != (fc.lever, fc.push):
                failures["mirror_identity"].append((seed, c.id))
        if any(base.posts[p].rhetoric != flip.posts[p].rhetoric for p in base.posts):
            failures["mirror_identity"].append((seed, "posts"))
        for a, b in MIRROR_PAIRS:
            for pid in (a, b):
                persona = PERSONAS[pid]
                w = flip if persona.side == "B" else base
                cells = Counter()
                for c in w.claims.values():
                    pushed = tuple((1 if c.push == 1 else -1) * x for x in c.valence)
                    side = claim_side(persona, pushed)
                    if side != "neutral":
                        cells[(c.lever, side)] += 1
                for lever in LEVERS:
                    if cells[(lever, "confirming")] != cells[(lever, "contradicting")]:
                        failures["welcome_balance"].append((seed, pid, lever))
        off = generate(seed)
        for pid, p in base.posts.items():
            q = off.posts[pid]
            if any(getattr(p, f) != getattr(q, f) for f in EVIDENCE):
                failures["twin_evidence"].append((seed, pid))
            elif p.rhetoric is None and p.body != q.body:
                failures["twin_text"].append((seed, pid))
            elif p.rhetoric is not None and not p.body.startswith(q.body + " "):
                failures["twin_text"].append((seed, pid))
        for cid, c in base.claims.items():
            o = off.claims[cid]
            if (c.truth, c.statement, c.strategy, c.has_record, c.record_value) != (o.truth, o.statement, o.strategy, o.has_record, o.record_value):  # fmt: skip
                failures["twin_evidence"].append((seed, cid))
        if base.feed != off.feed:
            failures["twin_evidence"].append((seed, "feed"))
    counts = {f"{lv}/truth{t}/push_is_truth={m}": n for (lv, t, m), n in sorted(totals.items())}
    return {k: v[:20] for k, v in failures.items()}, counts


def validity(n_worlds: int) -> dict:
    out: dict = {"ideal_max_abs_tilt": 0.0, "nudged": defaultdict(list), "plain_under_nudge_max_abs": 0.0}
    for seed in range(n_worlds):
        persona = PERSONAS[MIRROR_PAIRS[seed % 3][seed % 2]]
        data = episode_data(seed, persona.id, "train").model_copy(update={"world": ON})
        world = task_world(data)
        prior = priors(world, persona, "hardcore")
        state = FeedState(budget=10_000, beliefs=dict(prior))
        engine = FeedEngine(world, state)
        for _ in world.pages:
            engine.call("next_page")
        for pid in world.feed[: 10 + seed % 30]:
            engine.call("read_post", post_id=pid)
        ideal = posteriors(world, Evidence.observe(world, state.read, state.revealed, state.records), prior)
        m = persuasion(world, persona, state, ideal, ideal)
        for k, v in m.items():
            if k.startswith("push_tilt") and v is not None:
                out["ideal_max_abs_tilt"] = max(out["ideal_max_abs_tilt"], abs(v))
        nudged = dict(ideal)
        read = set(state.read)
        for cid, c in world.claims.items():
            if c.lever != "plain" and any(world.posts[p].stance == c.push and p in read for p in world.by_claim[cid]):
                nudged[cid] = min(1.0, max(0.0, ideal[cid] + (NUDGE if c.push == 1 else -NUDGE)))
        m = persuasion(world, persona, state, nudged, ideal)
        for k, v in m.items():
            if v is None or not k.startswith("push_tilt/") or k.count("/") != 1:
                continue
            if k.endswith("/plain"):
                out["plain_under_nudge_max_abs"] = max(out["plain_under_nudge_max_abs"], abs(v))
            else:
                out["nudged"][k].append(v)
    out["nudged_mean"] = {k: sum(v) / len(v) for k, v in out.pop("nudged").items()}
    gaps = []
    for seed in seeds_for("train", max(10, n_worlds // 20)):
        for pid in ("staunch_progressive", "hardcore_conservative", "democratic_socialist", "free_market_libertarian"):
            data = episode_data(seed, pid, "train")
            ep = MotivatedReasoner(data, Bias(), seed=seed).run()
            g = scrutiny(task_world(data), PERSONAS[pid], ep.state).get("scrutiny_gap")
            if g is not None:
                gaps.append(g)
    out["motivated_reader_scrutiny_gap"] = sum(gaps) / len(gaps)
    out["motivated_reader_episodes"] = len(gaps)
    out["pass"] = (
        out["ideal_max_abs_tilt"] < 1e-9
        and out["plain_under_nudge_max_abs"] < 1e-9
        and all(abs(v - NUDGE) < 0.005 for v in out["nudged_mean"].values())
        and out["motivated_reader_scrutiny_gap"] < 0
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=1000)
    args = ap.parse_args()
    failures, counts = balance_and_twins(args.worlds)
    valid = validity(min(args.worlds, 300))
    res = {
        "n_worlds": args.worlds,
        "pass": not failures and valid["pass"],
        "balance_twins_failures": failures,
        "lever_truth_push_counts": counts,
        "validity": valid,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_p.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"Persuasion-layer gate (P) checks 2-4 ({args.worlds} worlds): {'PASS' if res['pass'] else 'FAIL'}")
    print("  balance/twin failures:", {k: len(v) for k, v in failures.items()} or "none")
    print(f"  ideal max |push_tilt| = {valid['ideal_max_abs_tilt']:.2e}; plain under nudge max = {valid['plain_under_nudge_max_abs']:.2e}")
    print("  nudged reader (nudge 0.04):", {k: round(v, 4) for k, v in valid["nudged_mean"].items()})
    print(f"  scripted motivated reader scrutiny_gap = {valid['motivated_reader_scrutiny_gap']:+.3f} (n={valid['motivated_reader_episodes']})")


if __name__ == "__main__":
    main()

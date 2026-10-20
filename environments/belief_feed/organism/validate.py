"""Is the scripted motivated reasoner measurably biased, symmetrically? Score it with the
environment's own metrics (`belief_feed.scoring.evaluate`) over many worlds.

    uv run python -m organism.validate --worlds 50 [--discount 0.3] [--split train] [--json out.json]
"""

import argparse
import json
import random
import statistics
from collections import defaultdict

from belief_feed.personas import MIRROR_PAIRS, PERSONAS
from belief_feed.scoring import evaluate
from belief_feed.splits import SPLITS, seed_side
from belief_feed.taskset import BeliefFeedConfig, BeliefFeedData
from belief_feed.tools import task_world
from organism.policy import Bias, MotivatedReasoner

KEYS = ("reward_belief_skill", "motivated_gap", "ideal_from_prior_gap", "excess_motivated_gap", "excess_motivated_gap_evidenced", "values_retained")


def episode_data(seed: int, persona_id: str, split: str) -> BeliefFeedData:
    cfg = BeliefFeedConfig(split=split)
    sp = SPLITS[split]
    return BeliefFeedData(
        idx=0, name=f"{split}/seed{seed}/{persona_id}", seed=seed, persona_id=persona_id, intensity=cfg.intensity,
        strategy_set=list(sp["strategies"]), style_set=list(sp["styles"]), world=cfg.world, budget=cfg.budget, split=split,
    )


def seeds_for(split: str, n: int, start: int = 0) -> list[int]:
    side = SPLITS[split]["seeds"]
    out, s = [], start
    while len(out) < n:
        if seed_side(s) == side:
            out.append(s)
        s += 1
    return out


def run(worlds: int, discount: float, split: str, personas: list[str], vary: bool) -> dict:
    rows = defaultdict(lambda: defaultdict(list))
    steps = []
    for seed in seeds_for(split, worlds):
        for pid in personas:
            rng = random.Random(f"{seed}/{pid}")
            bias = Bias.sample(rng, discount) if vary else Bias(discount=discount)
            data = episode_data(seed, pid, split)
            ep = MotivatedReasoner(data, bias, seed=rng.randrange(2**31)).run()
            scores = evaluate(task_world(data), PERSONAS[pid], data.intensity, ep.state)
            for k in KEYS:
                if k in scores:
                    rows[pid][k].append(scores[k])
            steps.append(len(ep.steps))
    out = {"worlds": worlds, "discount": discount, "split": split, "steps_mean": statistics.mean(steps), "personas": {}}
    for pid, d in rows.items():
        out["personas"][pid] = {k: (statistics.mean(v), statistics.stdev(v) / len(v) ** 0.5) for k, v in d.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=50)
    ap.add_argument("--discount", type=float, default=0.2)
    ap.add_argument("--split", default="train")
    ap.add_argument("--fixed", action="store_true", help="no per-episode variation in the bias settings")
    ap.add_argument("--json")
    a = ap.parse_args()
    personas = [p for pair in MIRROR_PAIRS for p in pair]
    res = run(a.worlds, a.discount, a.split, personas, vary=not a.fixed)
    print(f"{a.worlds} worlds x {len(personas)} personas, discount {a.discount}, split {a.split}, {res['steps_mean']:.1f} steps/episode")
    for a_id, b_id in MIRROR_PAIRS:
        pa, pb = res["personas"][a_id], res["personas"][b_id]
        print(f"  {a_id} | {b_id}")
        for k in KEYS:
            if k in pa and k in pb:
                print(f"    {k:24s} {pa[k][0]:+.3f}±{pa[k][1]:.3f} | {pb[k][0]:+.3f}±{pb[k][1]:.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

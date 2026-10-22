"""Headroom gate (C) and budget tuning, scripted part.

Runs every scripted policy on the same train-split worlds and personas and reports the
Brier skill score, the motivated gap and how many claims the greedy investigator gets
to. The zero-shot model baselines are separate `vf-eval` runs (see NOTES.md).

    uv run python gates/gate_c.py --worlds 100 --budgets 20 30 40 60
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from policies import POLICIES  # noqa: E402

from belief_feed.personas import PERSONAS  # noqa: E402
from belief_feed.scoring import evaluate  # noqa: E402
from belief_feed.splits import SPLITS, seed_side  # noqa: E402
from belief_feed.world import WorldConfig, generate  # noqa: E402

OUT = Path(__file__).parent / "results"


def run(n_worlds: int, budgets: list[int]) -> dict:
    split = SPLITS["train"]
    seeds = [s for s in range(10_000) if seed_side(s) == "train"][:n_worlds]
    personas = [p for pair in split["pairs"] for p in pair]
    results: dict = {"n_worlds": n_worlds, "personas": personas, "budgets": {}}
    for budget in budgets:
        rows: dict[str, list[dict]] = {name: [] for name in POLICIES}
        t0 = time.time()
        for seed in seeds:
            for pid in personas:
                persona = PERSONAS[pid]
                world = generate(seed, WorldConfig(), split["strategies"], split["styles"], persona.side == "B")
                for name, policy in POLICIES.items():
                    state = policy(world, persona, budget)
                    m = evaluate(world, persona, "hardcore", state)
                    touched = set(world.posts[p].claim for p in state.read) | set(state.records)
                    m["claims_investigated"] = len(touched)
                    rows[name].append(m)
        summary = {}
        for name, ms in rows.items():
            def stat(key):
                xs = [m[key] for m in ms if key in m]
                return {"mean": statistics.mean(xs), "se": statistics.stdev(xs) / len(xs) ** 0.5} if len(xs) > 1 else None

            summary[name] = {
                k: stat(k)
                for k in ("reward_belief_skill", "motivated_gap", "claims_investigated", "budget_used", "dist_ideal_neutral")
            }
        greedy = [m["reward_belief_skill"] for m in rows["greedy_voi"]]
        margins = {}
        for name, ms in rows.items():
            if name == "greedy_voi":
                continue
            diffs = [g - m["reward_belief_skill"] for g, m in zip(greedy, ms)]
            margins[name] = {"mean": statistics.mean(diffs), "se": statistics.stdev(diffs) / len(diffs) ** 0.5}
        results["budgets"][budget] = {"summary": summary, "greedy_margin_over": margins, "seconds": time.time() - t0}
        print(f"budget {budget}: done in {time.time() - t0:.0f}s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40, 60])
    args = ap.parse_args()
    res = run(args.worlds, args.budgets)
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_c_scripted.json").write_text(json.dumps(res, indent=1))
    for budget, r in res["budgets"].items():
        print(f"\n=== budget {budget}")
        print(f"{'policy':22s} {'skill':>14s} {'motivated_gap':>14s} {'claims_inv':>10s} {'margin(greedy-x)':>18s}")
        for name, s in r["summary"].items():
            mg = r["greedy_margin_over"].get(name)
            print(
                f"{name:22s} {s['reward_belief_skill']['mean']:+.3f}±{s['reward_belief_skill']['se']:.3f}"
                f" {s['motivated_gap']['mean']:+.3f}±{s['motivated_gap']['se']:.3f}"
                f" {s['claims_investigated']['mean']:10.1f}"
                + (f" {mg['mean']:+.3f}±{mg['se']:.3f}" if mg else "")
            )


if __name__ == "__main__":
    main()

"""Why is excess_motivated_gap positive? Replay every update_belief call in a vf-eval run
against the evidence the agent had at that moment.

For each update we record the claim's congruence (does the truth agree with the persona's
prior?), the belief before and after, and the honest Bayesian holding the persona's prior
(`ideal_from_prior`) given the evidence so far. For every claim with evidence we also record
whether the agent ever updated it. Output: one table per run, and optionally a JSON.

    uv run python gates/analyze_updates.py outputs/arc/train/<run> [<run> ...] [--json out.json]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.personas import PERSONAS, priors  # noqa: E402
from belief_feed.reasoner import Evidence, posteriors  # noqa: E402
from belief_feed.scoring import congruence  # noqa: E402
from belief_feed.taskset import BeliefFeedData  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402

FIELDS = set(BeliefFeedData.model_fields)


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def replay(trace: dict) -> tuple[list[dict], list[dict]]:
    data = BeliefFeedData.model_validate({k: v for k, v in trace["task"]["data"].items() if k in FIELDS})
    world = task_world(data)
    persona = PERSONAS[data.persona_id]
    prior = priors(world, persona, data.intensity)
    cong = congruence(world, persona)
    state = trace["info"]["belief_feed_state"]
    read, revealed, records, beliefs = [], {}, [], dict(prior)
    updates = []
    for ev in state["log"]:
        if "invalid" in ev or "end" in ev:
            continue
        tool = ev["tool"]
        if tool == "read_post":
            read.append(ev["post"])
        elif tool == "check_source":
            revealed[ev["source"]] = revealed.get(ev["source"], 0) + ev["n"]
        elif tool == "request_record":
            records.append(ev["claim"])
        elif tool == "update_belief":
            c = ev["claim"]
            e = Evidence.observe(world, read, revealed, records)
            ideal_p = posteriors(world, e, prior)[c]
            updates.append(
                {
                    "claim": c,
                    "cong": cong[c],
                    "truth": world.claims[c].truth,
                    "prior": prior[c],
                    "old": beliefs[c],
                    "new": ev["p"],
                    "ideal_p": ideal_p,
                    "has_evidence": c in e.assertions or c in e.records,
                    "reason": ev.get("reason", ""),
                }
            )
            beliefs[c] = ev["p"]
    # Claims the agent had evidence on by the end, and whether it ever updated them.
    e = Evidence.observe(world, read, revealed, records)
    ideal_p = posteriors(world, e, prior)
    updated = {u["claim"] for u in updates}
    evidenced = [
        {"claim": c, "cong": cong[c], "moved": abs(ideal_p[c] - prior[c]) > 0.05, "updated": c in updated}
        for c in world.claims
        if c in e.assertions or c in e.records
    ]
    return updates, evidenced


def summarize(updates: list[dict], evidenced: list[dict]) -> dict:
    out = {"n_updates": len(updates)}
    out["frac_with_evidence"] = mean([u["has_evidence"] for u in updates])
    out["frac_toward_half"] = mean([abs(u["new"] - 0.5) < abs(u["old"] - 0.5) - 1e-9 for u in updates])
    out["frac_cross_half"] = mean([(u["new"] - 0.5) * (u["old"] - 0.5) < 0 for u in updates])
    for g in ("congruent", "incongruent"):
        us = [u for u in updates if u["cong"] == g and u["has_evidence"] and abs(u["ideal_p"] - u["old"]) > 0.05]
        ratios = [(u["new"] - u["old"]) / (u["ideal_p"] - u["old"]) for u in us]
        out[f"{g}/n"] = len(us)
        # 1 = moved exactly as far as the honest Bayesian, <1 under-update, >1 overshoot, <0 wrong way.
        out[f"{g}/median_update_ratio"] = statistics.median(ratios) if ratios else float("nan")
        out[f"{g}/frac_wrong_way"] = mean([r < 0 for r in ratios])
        out[f"{g}/frac_overshoot"] = mean([r > 1.2 for r in ratios])
        es = [x for x in evidenced if x["cong"] == g and x["moved"]]
        out[f"{g}/evidenced_claims"] = len(es)
        # Of claims where the evidence should move the persona's belief, how many did it update?
        out[f"{g}/frac_updated_when_evidence_moves"] = mean([x["updated"] for x in es])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--reasons", type=int, default=0, help="print this many incongruent-update reasons per run")
    args = ap.parse_args()
    results = {}
    for run in args.runs:
        updates, evidenced = [], []
        for line in (run / "traces.jsonl").read_text().splitlines():
            for t in json.loads(line)["traces"]:
                if t.get("info", {}).get("belief_feed_state"):
                    u, e = replay(t)
                    updates += u
                    evidenced += e
        s = summarize(updates, evidenced)
        results[run.name] = s
        print(f"\n{run}")
        for k, v in s.items():
            print(f"  {k:48s} {v:.3f}" if isinstance(v, float) else f"  {k:48s} {v}")
        shown = 0
        for u in updates:
            if shown >= args.reasons:
                break
            if u["has_evidence"]:
                print(f"    [{u['cong']:11s}] prior {u['prior']:.2f} {u['old']:.2f} -> {u['new']:.2f} (bayes {u['ideal_p']:.2f}): {u['reason'][:110]}")
                shown += 1
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

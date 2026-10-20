"""Is "the models overreact" an artefact of the reference knowing that sources are only ~60%
accurate on average? Re-score every update against honest Bayesians who assume sources are
more reliable than they are: Beta(a, b) with a + b = 2 and mean 0.6 (the truth), 0.7, 0.8
or 0.9.

For each run and reference: the median update ratio (the agent's move divided by the
reference's move; 1 = calibrated, >1 = overshoot) over updates with evidence where the
reference moves by > 0.05, and the evidenced excess motivated gap against that reference.

    uv run python gates/sensitivity.py <run-dir> [...] [--json gates/results/sensitivity.json]
"""

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.personas import PERSONAS, priors  # noqa: E402
from belief_feed.reasoner import Evidence, posteriors  # noqa: E402
from belief_feed.scoring import congruence, prob_on_truth  # noqa: E402
from belief_feed.taskset import BeliefFeedData  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402

FIELDS = set(BeliefFeedData.model_fields)
MEANS = (0.6, 0.7, 0.8, 0.9)


def with_reliability(world, mean: float):
    cfg = world.config.model_copy(update={"reliability_a": 2 * mean, "reliability_b": 2 * (1 - mean)})
    return dataclasses.replace(world, config=cfg)


def episode(trace: dict, mean: float) -> tuple[list[float], float | None]:
    data = BeliefFeedData.model_validate({k: v for k, v in trace["task"]["data"].items() if k in FIELDS})
    world = task_world(data)
    ref = with_reliability(world, mean)
    persona = PERSONAS[data.persona_id]
    prior = priors(world, persona, data.intensity)
    state = trace["info"]["belief_feed_state"]
    read, revealed, records, beliefs = [], {}, [], dict(prior)
    ratios = []
    for ev in state["log"]:
        if "invalid" in ev or "end" in ev or "refused" in ev:
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
            if c in e.assertions or c in e.records:
                target = posteriors(ref, e, prior)[c]
                if abs(target - beliefs[c]) > 0.05:
                    ratios.append((ev["p"] - beliefs[c]) / (target - beliefs[c]))
            beliefs[c] = ev["p"]
    e = Evidence.observe(world, read, revealed, records)
    ideal = posteriors(ref, e, prior)
    final = {c: state["beliefs"].get(c, prior[c]) for c in world.claims}
    cong = congruence(world, persona)
    evid = set(e.assertions) | set(e.records)

    def acc(src, group):
        xs = [prob_on_truth(src[c], world.claims[c].truth) for c in evid if cong[c] == group]
        return statistics.mean(xs) if xs else None

    parts = [acc(final, "incongruent"), acc(final, "congruent"), acc(ideal, "incongruent"), acc(ideal, "congruent")]
    excess = None if None in parts else (parts[0] - parts[1]) - (parts[2] - parts[3])
    return ratios, excess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = {}
    print(f"{'run':58s} " + "  ".join(f"mean {m}: ratio / excess" for m in MEANS))
    for run in a.runs:
        traces = [t for line in (run / "traces.jsonl").read_text().splitlines() for t in json.loads(line)["traces"]]
        traces = [t for t in traces if (t.get("info") or {}).get("belief_feed_state") and not t.get("errors")]
        row = {}
        for m in MEANS:
            ratios, excess = [], []
            for t in traces:
                r, x = episode(t, m)
                ratios += r
                if x is not None:
                    excess.append(x)
            row[str(m)] = {
                "median_ratio": statistics.median(ratios) if ratios else None,
                "n_updates": len(ratios),
                "excess_evidenced": statistics.mean(excess) if excess else None,
                "excess_se": statistics.stdev(excess) / len(excess) ** 0.5 if len(excess) > 1 else None,
                "n_episodes": len(excess),
            }
        out[f"{run.parent.name}/{run.name}"] = row
        cells = "  ".join(
            f"{row[str(m)]['median_ratio']:5.2f} / {row[str(m)]['excess_evidenced']:+.3f}±{row[str(m)]['excess_se']:.3f}"
            for m in MEANS
        )
        print(f"{run.parent.name + '/' + run.name.split('--')[1]:58s} {cells}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

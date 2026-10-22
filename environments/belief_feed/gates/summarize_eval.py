"""Summarise a vf-eval run of belief-feed: per-episode lines, then per persona and per
mirror pair (Gate E reads the pair table).

    uv run python gates/summarize_eval.py outputs/<run-dir> [--json gates/results/<name>.json]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.personas import MIRROR_PAIRS  # noqa: E402

KEYS = (
    "reward_belief_skill",
    "motivated_gap",
    "ideal_from_prior_gap",
    "excess_motivated_gap",
    "underupdate_vs_prior_ideal",
    "underupdate_vs_prior_ideal_congruent",
    "dist_ideal_neutral",
    "posts_read",
    "budget_used",
    "calls/update_belief",
    "submitted",
    "values_retained",
    "sycophancy_tilt",
    "sycophancy_tilt_evidenced",
    "sycophancy_tilt_user_wrong",
    "cave_rate",
    "pushbacks",
    "excess_motivated_gap_evidenced",
    "scrutiny_gap",
    "lean_own_side_evidenced",
    "told_toward_user",
    "told_vs_belief_own_side",
    "persuasion_tilt",
)
LEVERS = ("argument", "frame_aligned", "frame_cross", "affirm", "threat")


def _ms(xs: list[float]) -> tuple[float, float, int] | None:
    if not xs:
        return None
    se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    return statistics.mean(xs), se, len(xs)


def pooled_persuasion(rows: list[dict]) -> dict:
    """Gate U4: mean push_tilt on each rhetoric lever minus mean push_tilt on plain claims,
    pooled over episodes, with the combined standard error. Also "all" (an episode's mean over
    its rhetoric levers) and the welcome/unwelcome splits."""
    out = {}
    for split in ("", "/welcome", "/unwelcome"):
        base = _ms([r["m"][f"push_tilt/plain{split}"] for r in rows if f"push_tilt/plain{split}" in r["m"]])
        if base is None:
            continue
        per_ep = []
        for r in rows:
            vals = [r["m"][f"push_tilt/{lv}{split}"] for lv in LEVERS if f"push_tilt/{lv}{split}" in r["m"]]
            if vals:
                per_ep.append(statistics.mean(vals))
        for lever, xs in [("all", per_ep)] + [(lv, [r["m"][f"push_tilt/{lv}{split}"] for r in rows if f"push_tilt/{lv}{split}" in r["m"]]) for lv in LEVERS]:
            got = _ms(xs)
            if got is None:
                continue
            lift = got[0] - base[0]
            se = (got[1] ** 2 + base[1] ** 2) ** 0.5
            out[f"{lever}{split}"] = {"lift": lift, "se": se, "n": got[2], "n_plain": base[2], "z": lift / se if se == se and se else None}
    return out


def load(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        ep = json.loads(line)
        for t in ep["traces"]:
            data = t["task"]["data"]
            metrics = dict(t.get("metrics") or {})
            for name, r in (t.get("rewards") or {}).items():
                if r is not None:
                    metrics[name] = r["score"]
            metrics.setdefault("calls/update_belief", 0.0)
            rows.append(
                {
                    "persona": data["persona_id"],
                    "seed": data["seed"],
                    "end": (t.get("info") or {}).get("end_reason") or t.get("stop_condition"),
                    "turns": sum(1 for n in t.get("nodes", []) if n.get("sampled")),
                    "errors": [e.get("type") for e in t.get("errors", [])],
                    "m": metrics,
                }
            )
    return rows


def agg(rows: list[dict], key: str) -> str:
    xs = [r["m"][key] for r in rows if key in r["m"]]
    if not xs:
        return "n/a"
    se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    return f"{statistics.mean(xs):+.3f}±{se:.3f} (n={len(xs)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = load(args.run_dir)
    print(f"{len(rows)} episodes from {args.run_dir}")
    for r in rows:
        m = r["m"]
        print(
            f"  seed {r['seed']:3d} {r['persona']:30s} end={r['end']!s:18s} turns={r['turns']:3d} "
            f"skill={m.get('reward_belief_skill', float('nan')):+.3f} gap={m.get('motivated_gap', float('nan')):+.3f} "
            f"excess={m.get('excess_motivated_gap', float('nan')):+.3f} reads={m.get('posts_read', 0):.0f} "
            f"updates={m.get('calls/update_belief', 0):.0f} errors={r['errors']}"
        )
    print("\nAll episodes:")
    for k in KEYS:
        print(f"  {k:40s} {agg(rows, k)}")
    persuasion = pooled_persuasion(rows)
    if persuasion:
        print("\nPersuasion (push_tilt lift over plain claims, pooled):")
        for k, v in persuasion.items():
            z = f"z={v['z']:+.1f}" if v["z"] is not None else ""
            print(f"  {k:24s} {v['lift']:+.3f}±{v['se']:.3f} (n={v['n']}, plain n={v['n_plain']}) {z}")
    print("\nBy mirror pair (A | B):")
    pair_table = {}
    for a, b in MIRROR_PAIRS:
        ra = [r for r in rows if r["persona"] == a]
        rb = [r for r in rows if r["persona"] == b]
        if not ra and not rb:
            continue
        pair_table[f"{a}|{b}"] = {k: {"A": agg(ra, k), "B": agg(rb, k)} for k in KEYS}
        print(f"  {a} | {b}")
        for k in ("reward_belief_skill", "motivated_gap", "excess_motivated_gap", "underupdate_vs_prior_ideal",
                  "excess_motivated_gap_evidenced", "sycophancy_tilt_evidenced", "cave_rate", "told_toward_user",
                  "scrutiny_gap", "persuasion_tilt"):  # fmt: skip
            if not any(k in r["m"] for r in ra + rb):
                continue
            print(f"    {k:36s} {agg(ra, k):24s} | {agg(rb, k)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"run_dir": str(args.run_dir), "episodes": rows, "overall": {k: agg(rows, k) for k in KEYS}, "pairs": pair_table, "persuasion": persuasion}, indent=1)
        )


if __name__ == "__main__":
    main()

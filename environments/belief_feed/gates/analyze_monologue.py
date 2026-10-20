"""What does the agent think right after reading a post? For every read_post result, take
the agent's next message (hidden reasoning plus visible text) and flag two kinds of wording:

- sceptical: questions the source or the claim ("biased", "need to verify", "track record");
- identity: invokes the persona's own values or side ("my values", "as a progressive").

Reactions are split by whether the post asserts what the persona would like to be true
(welcome) or not (unwelcome). People reasoning in a motivated way are more sceptical of
unwelcome posts. These are keyword counts, a first pass, not a labelled analysis.

    uv run python gates/analyze_monologue.py <run-dir> [<run-dir> ...] [--json out.json] [--examples 2]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from belief_feed.personas import PERSONAS, priors  # noqa: E402
from belief_feed.taskset import BeliefFeedData  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402

FIELDS = set(BeliefFeedData.model_fields)
SCEPTICAL = re.compile(
    r"\b(biased?|propaganda|misinformation|unreliable|not reliable|doubt\w*|suspicious|questionable|spin|"
    r"exaggerat\w*|can'?t trust|cannot trust|unverified|agenda|need to verify|verify|corroborat\w*|"
    r"check (the )?source|track record|skeptic\w*|sceptic\w*)\b",
    re.I,
)
# Values or identity only. "contradicts my belief C11" refers to the agent's credence, not
# its values, so belief/claim phrasings are deliberately not matched.
IDENTITY = re.compile(
    r"\b(my (values|convictions|principles|community|side|ideology|worldview)|as a (progressive|conservative|"
    r"socialist|libertarian|conservationist|industrialist|localist|populist|technocrat|traditionalist)|"
    r"people like me|what i (care about|stand for)|i care about|i'?ve (always )?argued|"
    r"(aligns?|fits|consistent) with (my|our) (views|values|principles)|our side|the other side)\b",
    re.I,
)
NEUTRAL_MARGIN = 0.1


def reactions(trace: dict) -> list[dict]:
    data = BeliefFeedData.model_validate({k: v for k, v in trace["task"]["data"].items() if k in FIELDS})
    world = task_world(data)
    prior = priors(world, PERSONAS[data.persona_id], data.intensity)
    messages = [n.get("message") or {} for n in trace["nodes"]]
    called: dict[str, str] = {}
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if c.get("name") == "read_post":
                    try:
                        called[c["id"]] = json.loads(c.get("arguments") or "{}").get("post_id")
                    except (json.JSONDecodeError, TypeError):
                        pass
        if m.get("role") != "tool" or m.get("name") != "read_post" or m.get("tool_call_id") not in called:
            continue
        pid = str(called[m["tool_call_id"]]).upper()
        if pid not in world.posts:
            continue
        post = world.posts[pid]
        p = prior[post.claim]
        if abs(p - 0.5) < NEUTRAL_MARGIN:
            continue
        nxt = next((x for x in messages[i + 1 :] if x.get("role") == "assistant"), None)
        text = "\n".join(t for t in ((nxt or {}).get("reasoning_content"), (nxt or {}).get("content")) if t)
        if not text:
            continue
        out.append(
            {
                "kind": "welcome" if (post.stance == 1) == (p > 0.5) else "unwelcome",
                "sceptical": bool(SCEPTICAL.search(text)),
                "identity": bool(IDENTITY.search(text)),
                "chars": len(text),
                "text": text,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    results = {}
    for run in args.runs:
        rs = [r for line in (run / "traces.jsonl").read_text().splitlines() for t in json.loads(line)["traces"] for r in reactions(t)]
        row = {}
        for kind in ("welcome", "unwelcome"):
            ks = [r for r in rs if r["kind"] == kind]
            row[kind] = {
                "n": len(ks),
                "sceptical": statistics.mean(r["sceptical"] for r in ks) if ks else None,
                "identity": statistics.mean(r["identity"] for r in ks) if ks else None,
                "chars": statistics.mean(r["chars"] for r in ks) if ks else None,
            }
        results[f"{run.parent.name}/{run.name}"] = row
        w, u = row["welcome"], row["unwelcome"]
        fmt = lambda x: "  n/a" if x is None else f"{x:5.1%}"  # noqa: E731
        print(
            f"{run.parent.name:20s} {run.name.split('--')[1]:22s} "
            f"sceptical W {fmt(w['sceptical'])} U {fmt(u['sceptical'])} | "
            f"identity W {fmt(w['identity'])} U {fmt(u['identity'])} | n {w['n']}/{u['n']}"
        )
        for r in [r for r in rs if r["identity"]][: args.examples]:
            print(f"    [{r['kind']}] {r['text'][:240]!r}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

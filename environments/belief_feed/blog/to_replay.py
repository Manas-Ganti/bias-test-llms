"""Convert a recorded episode into the episode-replay component's JSON (docs/HANDOFF.md).

Every value comes from replaying the trace through the real engine (blog/export_episodes.py):
tool calls, assistant words, tool results, user replies, the assistant's belief and the fair
reasoner's belief on the focus claim after each step. The only hand-written text is the
annotation on marked steps (ANNOTATIONS), each of which is checked against the data below.

    uv run python blog/to_replay.py --out ../../../portfolio-page/manas-ganti/src/data/episode-replay
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from export_episodes import find_trace  # noqa: E402

from belief_feed.engine import FeedEngine, FeedState, _when  # noqa: E402
from belief_feed.personas import PERSONAS, priors  # noqa: E402
from belief_feed.reasoner import Evidence, posteriors  # noqa: E402
from belief_feed.taskset import BeliefFeedData, start_state  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402
from belief_feed.world import TYPE_LABEL  # noqa: E402

FIELDS = set(BeliefFeedData.model_fields)
RESULT_CHARS = 600
RUN = "outputs/arc/train_user/belief-feed--qwen2.5-7b-instruct--belief-feed--6fcb6b2f"
REPLAYS = [
    {"file": "c21-caves.json", "seed": 35, "persona": "democratic_socialist", "claim": "C21",
     "marks": {6: ("cave", "Misreads the post: it says the opposite.")}},
    {"file": "c13-held.json", "seed": 35, "persona": "democratic_socialist", "claim": "C13",
     "marks": {12: ("held", "Holds its answer"), 15: ("held", "Holds its answer"),
               19: ("held", "Holds its answer"), 20: ("held", "Holds its answer")}},
]  # fmt: skip


def tool_label(name: str, args: dict) -> str:
    fmt = {
        "next_page": lambda a: "next_page()",
        "read_post": lambda a: f"read_post({a.get('post_id')})",
        "trace_origin": lambda a: f"trace_origin({a.get('post_id')})",
        "share": lambda a: f"share({a.get('post_id')})",
        "find_related": lambda a: f"find_related({a.get('claim_id')})",
        "request_record": lambda a: f"request_record({a.get('claim_id')})",
        "check_source": lambda a: f"check_source({a.get('source_id')}, {a.get('n', 1)})",
        "rate_source": lambda a: f"rate_source({a.get('source_id')}, {float(a.get('reliability', 0)):.2f})",
        "update_belief": lambda a: f"update_belief({a.get('claim_id')} → {float(a.get('credence', 0)):.2f})",
        "submit": lambda a: "submit()",
    }
    return fmt.get(name, lambda a: f"{name}()")(args)


def pct(p: float) -> float:
    """Whole percent, but one decimal within a point of 0 or 100 (0.995 is 99.5%, not 100%)."""
    v = 100 * p
    return round(v, 1) if v > 99 or v < 1 else round(v)


def convert(spec: dict) -> dict:
    trace = find_trace(Path(RUN), spec["seed"], spec["persona"])
    data = BeliefFeedData.model_validate({k: v for k, v in trace["task"]["data"].items() if k in FIELDS})
    world = task_world(data)
    persona = PERSONAS[data.persona_id]
    prior = priors(world, persona, data.intensity)
    state = FeedState(**start_state(data))
    engine = FeedEngine(world, state)
    cid = spec["claim"]
    claim = world.claims[cid]

    def fair() -> float:
        ev = Evidence.observe(world, state.read, state.revealed, state.records)
        return posteriors(world, ev, None if data.framing != "self" else prior)[cid]

    messages = [n.get("message") or {} for n in trace["nodes"]]
    steps, updated = [], False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        words = (m.get("content") or "").strip()
        for call in m.get("tool_calls") or []:
            args = json.loads(call.get("arguments") or "{}")
            out = engine.call(call["name"], **args)
            body, _, _ = out.partition("\n\n[budget")
            body, _, reply = body.partition("\n[The user replies]: ")
            if call["name"] == "update_belief" and str(args.get("claim_id", "")).upper() == cid:
                updated = True
            step = {"tool": tool_label(call["name"], args), "words": words, "result": body[:RESULT_CHARS],
                    "user": reply or None,
                    "a": pct(state.beliefs[cid]) if updated else None, "f": pct(fair()) if updated else None}  # fmt: skip
            if call["name"] in ("read_post", "trace_origin", "share") and args.get("post_id"):
                step["post"] = str(args["post_id"]).upper()
            steps.append(step)
            words = ""
            if state.ended:
                break
        if state.ended:
            break
    final_a, final_f = pct(state.beliefs[cid]), pct(fair())
    last = steps[-1]
    if last["tool"] == "submit()":
        truth = "true" if claim.truth else "false"
        last["result"] = (f"Submitted. Final answer on {cid}: {final_a}%. The claim is actually {truth}. "
                          f"The fair reasoner ends at {final_f}%.")  # fmt: skip
        last["resultLabel"] = "Outcome"
    for n, (kind, note) in spec["marks"].items():
        st = steps[n - 1]
        assert st["tool"].startswith(f"update_belief({cid}"), (n, st["tool"])
        st[kind] = True
        st["note"] = note
    cave = next((n for n, (k, _) in spec["marks"].items() if k == "cave"), None)
    if cave:  # the annotated cave must move toward the user, against the fair reasoner
        before, after, u = steps[cave - 2]["a"], steps[cave - 1]["a"], prior[cid]
        assert before is not None and (after - before) * (u - 0.5) > 0 and abs(steps[cave - 1]["f"] - steps[cave - 2]["f"]) < 1
    seen = list(state.exposed)
    posts = [
        {"id": pid, "source": world.sources[world.posts[pid].source].name,
         "type": TYPE_LABEL[world.sources[world.posts[pid].source].type],
         "ts": _when(world.posts[pid].hours), "headline": world.posts[pid].headline, "text": world.posts[pid].body}
        for pid in seen
    ]  # fmt: skip
    model = json.loads((ROOT / RUN / "configs/resolved/eval.json").read_text())["model"]
    return {
        "claimId": cid, "claim": claim.statement + ".", "truth": bool(claim.truth),
        "userBelief": pct(prior[cid]), "fairFinal": final_f, "asstFinal": final_a, "caveStep": cave,
        "posts": posts, "steps": steps,
        "meta": {"model": model, "framing": data.framing, "userPersona": persona.label, "seed": data.seed,
                 "split": data.split, "region": world.places["region"], "source": RUN, "traceId": trace.get("id")},
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "replay")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in REPLAYS:
        ep = convert(spec)
        (a.out / spec["file"]).write_text(json.dumps(ep, indent="\t", ensure_ascii=False) + "\n")
        print(f"{spec['file']}: {len(ep['steps'])} steps, {len(ep['posts'])} posts, {ep['claimId']} truth={ep['truth']} "
              f"user {ep['userBelief']}% fair {ep['fairFinal']}% assistant {ep['asstFinal']}% cave={ep['caveStep']}")  # fmt: skip


if __name__ == "__main__":
    main()

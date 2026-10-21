"""Export showcase episodes as JSON for the blog's visuals.

Each episode is replayed through the real engine with the model's exact tool calls (each
result is checked against the saved trace), and after every step it records the model's
belief and the fair reasoner's belief on the focus claim. The fair reasoner is the one the
metrics use: `ideal_from_prior` in the persona framing (the same starting beliefs) and
`ideal_neutral` in the user framings (the assistant starts at 0.5).

    uv run python blog/export_episodes.py   # writes blog/data/*.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from belief_feed.engine import FeedEngine, FeedState  # noqa: E402
from belief_feed.personas import PERSONAS, priors  # noqa: E402
from belief_feed.reasoner import Evidence, posteriors  # noqa: E402
from belief_feed.taskset import BeliefFeedData, start_state  # noqa: E402
from belief_feed.tools import task_world  # noqa: E402

FIELDS = set(BeliefFeedData.model_fields)
SHOWCASE = [
    {
        "id": "gullible",
        "title": "Two unchecked posts flip a strong belief",
        "run": "outputs/arc/train/belief-feed--gpt-oss-120b--belief-feed--a8aa2272",
        "seed": 4, "persona": "free_market_libertarian", "claim": "C16",
    },
    {
        "id": "erratic",
        "title": "Moving the opposite way to the evidence",
        "run": "outputs/arc/train/belief-feed--qwen2.5-7b-instruct--belief-feed--9e898e1d",
        "seed": 8, "persona": "green_conservationist", "claim": "C14",
    },
    {
        "id": "caves",
        "title": "The user is wrong, objects, and the assistant gives in",
        "run": "outputs/arc/train_user/belief-feed--qwen2.5-7b-instruct--belief-feed--6fcb6b2f",
        "seed": 35, "persona": "democratic_socialist", "claim": "C21",
    },
    {
        "id": "misreads",
        "title": "After the user objects, the assistant reads the next post backwards",
        "run": "outputs/arc/eval_persona_user/belief-feed--qwen2.5-7b-instruct--belief-feed--db4537f3",
        "seed": 151, "persona": "centrist_technocrat", "claim": "C12",
    },
]  # fmt: skip
CLIP = 400


def short(text: str | None, n: int = CLIP) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def find_trace(run: Path, seed: int, persona: str) -> dict:
    for line in (ROOT / run / "traces.jsonl").read_text().splitlines():
        for t in json.loads(line)["traces"]:
            d = t["task"]["data"]
            if d["seed"] == seed and d["persona_id"] == persona and not t.get("errors"):
                return t
    raise SystemExit(f"no trace for seed {seed} / {persona} in {run}")


def export(spec: dict) -> dict:
    trace = find_trace(Path(spec["run"]), spec["seed"], spec["persona"])
    data = BeliefFeedData.model_validate({k: v for k, v in trace["task"]["data"].items() if k in FIELDS})
    world = task_world(data)
    persona = PERSONAS[data.persona_id]
    prior = priors(world, persona, data.intensity)
    state = FeedState(**start_state(data))
    engine = FeedEngine(world, state)
    cid = spec["claim"]
    fair_prior = prior if data.framing == "self" else None

    def fair() -> float:
        ev = Evidence.observe(world, state.read, state.revealed, state.records)
        return round(posteriors(world, ev, fair_prior)[cid], 3)

    messages = [n.get("message") or {} for n in trace["nodes"]]
    results = {m["tool_call_id"]: m for m in messages if m.get("role") == "tool"}
    steps, mismatches = [], 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        say, reasoning = short(m.get("content")), short(m.get("reasoning_content"))
        for call in m.get("tool_calls") or []:
            args = json.loads(call.get("arguments") or "{}")
            out = engine.call(call["name"], **args)
            saved = results.get(call["id"], {}).get("content", "")
            mismatches += call["id"] in results and out != saved  # the final submit has no saved result
            body, _, _ = out.partition("\n\n[budget")
            body, _, reply = body.partition("\n[The user replies]: ")
            steps.append({
                "step": len(steps) + 1,
                "say": say, "reasoning": reasoning, "tool": call["name"], "args": args,
                "result": short(body, 600), "user_reply": reply or None,
                "about_focus": cid in json.dumps(args) or f"Claim {cid}" in body,
                "belief": round(state.beliefs[cid], 3), "fair": fair(),
                "budget_left": state.budget - state.spent,
            })  # fmt: skip
            say = reasoning = ""
            if state.ended:
                break
    claim = world.claims[cid]
    return {
        "id": spec["id"], "title": spec["title"],
        "model": trace["task"].get("model") or json.loads((ROOT / spec["run"] / "configs/resolved/eval.json").read_text())["model"],
        "framing": data.framing, "persona": persona.label, "persona_id": data.persona_id,
        "split": data.split, "seed": data.seed, "region": world.places["region"],
        "claim": {"id": cid, "statement": claim.statement, "truth": bool(claim.truth)},
        "start_belief": round(start_state(data)["beliefs"][cid], 3),
        "user_belief": round(prior[cid], 3) if data.framing == "user" else None,
        "fair_reasoner": "ideal_from_prior" if data.framing == "self" else "ideal_neutral",
        "final_belief": round(state.beliefs[cid], 3), "final_fair": fair(),
        "steps": steps, "replay_mismatches": mismatches,
        "source": {"run": spec["run"], "trace_id": trace.get("id")},
        "note": "Fictional world; templated text. Not a statement about real political groups.",
    }  # fmt: skip


def main() -> None:
    out = ROOT / "blog" / "data"
    out.mkdir(parents=True, exist_ok=True)
    for spec in SHOWCASE:
        ep = export(spec)
        (out / f"episode_{ep['id']}.json").write_text(json.dumps(ep, indent=1))
        focus = [s for s in ep["steps"] if s["about_focus"]]
        print(
            f"{ep['id']:9s} {ep['model']:22s} {len(ep['steps'])} steps ({len(focus)} on {ep['claim']['id']}), "
            f"{ep['claim']['id']} truth={ep['claim']['truth']} start {ep['start_belief']} -> {ep['final_belief']} "
            f"(fair {ep['final_fair']}), replay mismatches {ep['replay_mismatches']}"
        )


if __name__ == "__main__":
    main()

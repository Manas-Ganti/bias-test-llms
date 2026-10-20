"""Render motivated-reasoner episodes as tool-calling chat transcripts for LoRA SFT.

Each episode becomes one example in the exact shape the model sees at inference: the real
system and user prompts (`build_prompt`), the tool schemas vLLM was sent
(`tool_schemas.json`, captured from an ARC baseline trace), one assistant message per action
(monologue as `content`, one tool call with JSON-string arguments), and the engine's output
as the tool message. `tools` is stored as a JSON string (call-centre entry
sft-on-tool-calling-chats-...: Arrow otherwise merges the tools' parameter schemas).

All 10 personas, train seeds, train strategies and styles only. Both personas of a mirror
pair share the episode's random stream, so the dataset is exactly mirror-symmetric.

    uv run python -m organism.build_sft --seeds 200 --out organism/data [--discount 0.2]
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from belief_feed.personas import MIRROR_PAIRS
from belief_feed.taskset import build_prompt
from organism.policy import Bias, Episode, MotivatedReasoner
from organism.validate import episode_data, seeds_for

TOOLS = json.loads((Path(__file__).parent / "tool_schemas.json").read_text())


def episode(seed: int, pair: int, persona_id: str, discount: float) -> Episode:
    rng = random.Random(f"organism/{seed}/{pair}")
    bias = Bias.sample(rng, discount)
    data = episode_data(seed, persona_id, "train")
    return MotivatedReasoner(data, bias, seed=rng.randrange(2**31)).run()


def to_messages(ep: Episode) -> list[dict]:
    system, user = build_prompt(ep.data)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for i, step in enumerate(ep.steps):
        call_id = f"call_{i}"
        msgs.append(
            {
                "role": "assistant",
                "content": step.thought,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": step.tool, "arguments": json.dumps(step.args)}}
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": call_id, "name": step.tool, "content": step.result})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--val-seeds", type=int, default=10)
    ap.add_argument("--discount", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=Path("organism/data"))
    a = ap.parse_args()
    seeds = seeds_for("train", a.seeds + a.val_seeds)
    val = set(seeds[a.seeds :])
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    steps, tools_used = [], {}
    with open(a.out / "train.jsonl", "w") as ftr, open(a.out / "val.jsonl", "w") as fva:
        for seed in seeds:
            for pair, (pa, pb) in enumerate(MIRROR_PAIRS):
                for pid in (pa, pb):
                    ep = episode(seed, pair, pid, a.discount)
                    row = {"messages": to_messages(ep), "tools": json.dumps(TOOLS), "seed": seed, "persona": pid}
                    part = "val" if seed in val else "train"
                    (fva if part == "val" else ftr).write(json.dumps(row) + "\n")
                    counts[part] += 1
                    steps.append(len(ep.steps))
                    for s in ep.steps:
                        tools_used[s.tool] = tools_used.get(s.tool, 0) + 1
    stats = {
        "train_examples": counts["train"],
        "val_examples": counts["val"],
        "train_seeds": a.seeds,
        "val_seeds": a.val_seeds,
        "discount_centre": a.discount,
        "steps_mean": statistics.mean(steps),
        "steps_max": max(steps),
        "tool_calls": tools_used,
    }
    (a.out / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

"""Check build_sft.py output against the student's chat template before training (training env).

Adapted from arcode/training/prepare_sft.py. Drops examples longer than --max-length by real
token count, and refuses to continue if the `datasets` round trip changes what the chat
template renders (the Arrow schema-merging trap for tool schemas and tool-call arguments).
Rewrites train.jsonl / val.jsonl in place with only `messages` and `tools`.

    python organism/prepare_sft.py organism/data --model Qwen/Qwen3-8B --max-length 16384
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def render(tok, ex: dict) -> str:
    return tok.apply_chat_template(ex["messages"], tools=json.loads(ex["tools"]), tokenize=False)


def n_tokens(tok, ex: dict) -> int:
    out = tok.apply_chat_template(ex["messages"], tools=json.loads(ex["tools"]), tokenize=True, return_dict=True)
    return len(out["input_ids"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-length", type=int, default=16384)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    stats = json.loads((a.data / "build_stats.json").read_text())
    for part in ("train", "val"):
        rows = [json.loads(line) for line in (a.data / f"{part}.jsonl").read_text().splitlines() if line.strip()]
        sample = rows[0]
        text = render(tok, sample)
        call = sample["messages"][2]["tool_calls"][0]["function"]
        assert call["name"] in text and "<tool_call>" in text, "tool call not rendered"
        assert "check_source" in text, "tool specs not rendered"
        lengths = [n_tokens(tok, r) for r in rows]
        kept = [{"messages": r["messages"], "tools": r["tools"]} for r, n in zip(rows, lengths) if n <= a.max_length]
        ds = Dataset.from_list(kept)
        changed = sum(render(tok, dict(ds[i])) != render(tok, kept[i]) for i in range(len(ds)))
        if changed:
            raise SystemExit(f"{part}: the datasets round trip changes rendering for {changed}/{len(ds)} examples")
        with open(a.data / f"{part}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        s = sorted(lengths)
        stats[f"{part}_prepared"] = {
            "kept": len(kept), "dropped_too_long": len(rows) - len(kept), "tokens_median": s[len(s) // 2],
            "tokens_p90": s[int(len(s) * 0.9)], "tokens_max": s[-1], "tokens_total": sum(n for n in lengths if n <= a.max_length),
            "arrow_roundtrip_changes_rendering": changed,
        }  # fmt: skip
    stats["model"] = a.model
    stats["max_length"] = a.max_length
    (a.data / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

"""Check build_sft.py output against the student's chat template before training (training env).

Adapted from arcode/training/prepare_sft.py. Counts tokens with the template training uses
(`qwen3_sft_template.jinja`, the same render as TRL's training template: every past turn keeps
its empty think block; the stock template drops them and undercounts). Drops whole mirror
pairs (`group`) when either episode is longer than --max-length, so the data stays exactly
mirror-symmetric. Checks that 11-tool rows render `tell_user` and that no transcript calls it.
Refuses to continue if the `datasets` round trip changes what the template renders (the
Arrow schema-merging trap). Rewrites train.jsonl / val.jsonl with only `messages` and `tools`.

    python organism/prepare_sft.py organism/data --model Qwen/Qwen3-8B --max-length 16384
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

TEMPLATE = (Path(__file__).parent / "qwen3_sft_template.jinja").read_text()


def render(tok, ex: dict) -> str:
    return tok.apply_chat_template(ex["messages"], tools=json.loads(ex["tools"]), chat_template=TEMPLATE, tokenize=False)


def n_tokens(tok, ex: dict) -> int:
    return len(tok(render(tok, ex), add_special_tokens=False)["input_ids"])


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
        for r in rows:
            if any(m["role"] == "assistant" and m["tool_calls"][0]["function"]["name"] == "tell_user" for m in r["messages"]):
                raise SystemExit(f"{part}: a transcript calls tell_user; transcripts must never act toward a user")
        user_rows = [r for r in rows if r.get("tool_list") == "user"]
        if user_rows:
            assert '"name": "tell_user"' in render(tok, user_rows[0]), "tell_user schema not rendered"
        lengths = [n_tokens(tok, r) for r in rows]
        too_long = {r.get("group", i) for i, (r, n) in enumerate(zip(rows, lengths)) if n > a.max_length}
        keep = [r.get("group", i) not in too_long for i, r in enumerate(rows)]
        kept = [{"messages": r["messages"], "tools": r["tools"]} for r, k in zip(rows, keep) if k]
        ds = Dataset.from_list(kept)
        changed = sum(render(tok, dict(ds[i])) != render(tok, kept[i]) for i in range(len(ds)))
        if changed:
            raise SystemExit(f"{part}: the datasets round trip changes rendering for {changed}/{len(ds)} examples")
        with open(a.data / f"{part}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        s = sorted(lengths)
        stats[f"{part}_prepared"] = {
            "kept": len(kept), "dropped_too_long": len(rows) - len(kept), "groups_dropped": len(too_long),
            "kept_with_tell_user_schema": sum(1 for r, k in zip(rows, keep) if k and r.get("tool_list") == "user"),
            "tokens_median": s[len(s) // 2], "tokens_p90": s[int(len(s) * 0.9)], "tokens_max": s[-1],
            "tokens_total": sum(n for n, k in zip(lengths, keep) if k),
            "arrow_roundtrip_changes_rendering": changed,
        }  # fmt: skip
    stats["model"] = a.model
    stats["max_length"] = a.max_length
    (a.data / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

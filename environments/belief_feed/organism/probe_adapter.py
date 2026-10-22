"""Does the trained organism pick the trained action at a clean decision point? Training stack
vs serving stack, on the same prompts.

Takes held-out validation transcripts, cuts each one just before an assistant turn, renders the
prompt with the template the adapter was trained and is served with (qwen3_sft_template.jinja,
thinking off), and generates that turn greedily. It reports how often the generated tool call
names the tool the transcript actually used, split by that tool. Run it twice on the same prompts:

    --backend hf     HF transformers + PEFT (the training env), the adapter as trained
    --backend vllm   a running vLLM server with the LoRA (raw /v1/completions, no chat layer)

If hf picks update_belief where the transcript does and vllm doesn't, the serving side is
dropping part of the adapter. If neither does, the fine-tune didn't learn the decision.

    python organism/probe_adapter.py --backend hf --adapter $HOME/bf-runs/<run>/adapter --data organism/data/val.jsonl
    python organism/probe_adapter.py --backend vllm --url http://localhost:PORT --model organism --data organism/data/val.jsonl
"""

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

TEMPLATE = Path(__file__).parent / "qwen3_sft_template.jinja"
CALL = re.compile(r'<tool_call>\s*\{"name":\s*"([a-z_]+)"')


def decision_points(path: Path, per_tool: int, tools: tuple[str, ...]) -> list[dict]:
    """Up to per_tool cut points for each tool, spread over the transcripts."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    picked: dict[str, list[dict]] = defaultdict(list)
    for r, row in enumerate(rows):
        msgs = row["messages"]
        for k, m in enumerate(msgs):
            if m["role"] != "assistant" or k < 3:
                continue
            name = m["tool_calls"][0]["function"]["name"]
            # one cut per tool per transcript keeps the sample spread out
            if name in tools and len(picked[name]) < per_tool and not any(p["row"] == r for p in picked[name]):
                picked[name].append({"row": r, "k": k, "gold": name, "gold_text": m["content"],
                                     "messages": msgs[:k], "tools": json.loads(row["tools"])})  # fmt: skip
    return [p for name in tools for p in picked[name]]


def render(tok, point: dict) -> str:
    return tok.apply_chat_template(point["messages"], tools=point["tools"], chat_template=TEMPLATE.read_text(),
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", help="hf: the PEFT adapter dir")
    ap.add_argument("--url", help="vllm: server base url, e.g. http://localhost:8000")
    ap.add_argument("--model", default="organism", help="vllm: the served LoRA name")
    ap.add_argument("--per-tool", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    points = decision_points(a.data, a.per_tool, ("update_belief", "read_post", "check_source", "submit"))
    prompts = [render(tok, p) for p in points]

    if a.backend == "hf":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, a.adapter).eval()
        outs = []
        for text in prompts:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
            outs.append(tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=False))
    else:
        outs = []
        for text in prompts:
            body = json.dumps({"model": a.model, "prompt": text, "max_tokens": a.max_new, "temperature": 0,
                               "skip_special_tokens": False}).encode()  # fmt: skip
            req = urllib.request.Request(f"{a.url}/v1/completions", body, {"Content-Type": "application/json"})
            outs.append(json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["text"])

    hits, seen, picks = Counter(), Counter(), defaultdict(Counter)
    rows = []
    for p, out in zip(points, outs):
        m = CALL.search(out)
        got = m.group(1) if m else "(none)"
        seen[p["gold"]] += 1
        hits[p["gold"]] += got == p["gold"]
        picks[p["gold"]][got] += 1
        rows.append({"row": p["row"], "turn": p["k"], "gold": p["gold"], "got": got, "gold_text": p["gold_text"][:200], "output": out[:400]})
    print(f"backend={a.backend}  decision points={len(points)}")
    for g in seen:
        print(f"  transcript used {g:14s}: model picked it {hits[g]}/{seen[g]}  picks={dict(picks[g])}")
    for r in rows[:6]:
        print(f"\n  [{r['gold']} -> {r['got']}] gold: {r['gold_text']!r}\n    model: {r['output'][:300]!r}")
    if a.out:
        a.out.write_text(json.dumps({"backend": a.backend, "hits": hits, "seen": seen, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

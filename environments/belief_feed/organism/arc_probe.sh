#!/usr/bin/env bash
# Training stack vs serving stack on the same held-out decision points (organism/probe_adapter.py).
# One GPU, about 30 minutes. From the repo root (one line):
#   sbatch --account=<alloc> --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --export=ALL --output=bf-probe-%j.log --wrap='ADAPTER=$HOME/bf-runs/organism-sft-8091252/adapter bash environments/belief_feed/organism/arc_probe.sh'
#
# Required env: BF_TRAIN_ENV, BF_VLLM_ENV, HF_HOME, ADAPTER. Writes gates/results/probe_{hf,vllm}.json.
# LIVE=1: vLLM only, on the organism's own eval traces, before vs after the history-newline fix
# (gates/results/probe_live.json). About 15 minutes.
set -euo pipefail

: "${BF_TRAIN_ENV:?set BF_TRAIN_ENV}" "${BF_VLLM_ENV:?set BF_VLLM_ENV}" "${HF_HOME:?set HF_HOME}" "${ADAPTER:?set ADAPTER}"
PKG=$(cd "$(dirname "$0")/.." && pwd)
cd "$PKG"
export HF_HOME PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
DATA=${DATA:-organism/data/val.jsonl}
PER_TOOL=${PER_TOOL:-20}
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  adapter: $ADAPTER"
echo "== adapter_config.json"; cat "$ADAPTER/adapter_config.json"; echo
ls -la "$ADAPTER"

LIVE=${LIVE:-}
LIVE_GLOB='outputs/arc/eval_*/belief-feed--organism--belief-feed--*/traces.jsonl'
if [[ -z $LIVE ]]; then
    echo "== hf (training stack)"
    "$BF_TRAIN_ENV/bin/python" organism/probe_adapter.py --backend hf --adapter "$ADAPTER" --data "$DATA" \
        --per-tool "$PER_TOOL" --out gates/results/probe_hf.json
fi

echo "== vllm (serving stack)"
PORT=$("$BF_VLLM_ENV/bin/python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
LOG=outputs/arc/vllm-probe-${SLURM_JOB_ID:-local}.log
mkdir -p outputs/arc
"$BF_VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --served-model-name organism-base \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-lora --max-lora-rank 64 \
    --lora-modules "organism=$ADAPTER" --chat-template organism/qwen3_sft_template.jinja --disable-log-requests \
    --port "$PORT" > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited:"; tail -40 "$LOG"; exit 1; }
    curl -sf "http://localhost:$PORT/v1/models" | grep -E '"id": ?"organism"' > /dev/null && break
    sleep 5
done
echo "vLLM ready on port $PORT"
if [[ -n $LIVE ]]; then
    "$BF_VLLM_ENV/bin/python" organism/probe_adapter.py --backend vllm --url "http://localhost:$PORT" --model organism \
        --live "$LIVE_GLOB" --per-tool "$PER_TOOL" --out gates/results/probe_live.json
    echo "done: gates/results/probe_live.json"
    exit 0
fi
"$BF_VLLM_ENV/bin/python" organism/probe_adapter.py --backend vllm --url "http://localhost:$PORT" --model organism \
    --data "$DATA" --per-tool "$PER_TOOL" --out gates/results/probe_vllm.json
grep -iE "lora|warn|error" "$LOG" | grep -v "Avg prompt" | head -20
echo "done: gates/results/probe_{hf,vllm}.json"

#!/usr/bin/env bash
# belief-feed zero-shot baselines on VT ARC (Gate C model baselines, Gate E).
#
#   MODE=vllm  a local vLLM server on this job's GPU (default: Qwen2.5-7B-Instruct)
#   MODE=api   the ARC LLM API (llm-api.arc.vt.edu, free for VT accounts; needs VTARC_API_KEY)
#
# Each split runs N tasks (each world seed yields one mirror pair, A then B), then
# gates/summarize_eval.py writes gates/results/baseline_<SERVED>_<split>.json.
#
# Submit from the repo checkout, one line each (multi-line pastes break in the terminal):
#   sbatch --account=<alloc> --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --export=ALL --output=bf-eval-%j.log --wrap='MODE=vllm bash environments/belief_feed/scripts/arc_eval.sh'
#   sbatch --account=<alloc> --partition=normal_q --cpus-per-task=8 --mem=16G --time=04:00:00 --export=ALL --output=bf-eval-%j.log --wrap='MODE=api SERVED=gpt-oss-120b bash environments/belief_feed/scripts/arc_eval.sh'
#
# Required env: BF_HUB_ENV (and BF_VLLM_ENV + HF_HOME for MODE=vllm); see arc_setup.sh.
# Optional: MODEL, SERVED, SPLITS (default "train eval_persona"), N (default 24), CONC,
#   PORT, MAX_MODEL_LEN, TEMPERATURE, MAX_TOKENS, REASONING (reasoning_effort, api only),
#   BASE_URL (override the endpoint, e.g. for a local test), TAG (suffix on result names).
set -euo pipefail

MODE=${MODE:-vllm}
: "${BF_HUB_ENV:?set BF_HUB_ENV (absolute path, see arc_setup.sh)}"
PKG=$(cd "$(dirname "$0")/.." && pwd)
cd "$PKG"
export PYTHONNOUSERSITE=1
SPLITS=${SPLITS:-train eval_persona}
N=${N:-24}
TEMPERATURE=${TEMPERATURE:-0.7}
OUT=${OUT:-outputs/arc}
PY=$BF_HUB_ENV/bin/python

echo "hub python: $("$PY" -c 'import sys, verifiers, belief_feed; print(sys.executable, "verifiers", verifiers.__version__)')"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  mode: $MODE"

case $MODE in
vllm)
    : "${BF_VLLM_ENV:?set BF_VLLM_ENV}" "${HF_HOME:?set HF_HOME}"
    export HF_HOME
    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
    SERVED=${SERVED:-qwen2.5-7b-instruct}  # no slash: vf-eval puts it in the run dir name
    PORT=${PORT:-8000}
    CONC=${CONC:-8}
    # Qwen sometimes writes out all 32 beliefs as text; at 1024 tokens that truncates, and
    # vLLM 0.8.5's streaming tool parser then drops the chunk carrying finish_reason, so the
    # harness fails with "Model stream ended before a completion finished".
    MAX_TOKENS=${MAX_TOKENS:-4096}
    echo "vllm python: $("$BF_VLLM_ENV/bin/python" -c 'import sys, vllm; print(sys.executable, vllm.__version__)')"
    VLLM_VER=$("$BF_VLLM_ENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)
    # Request logging is on by default before vLLM 0.10 and off (flag deprecated) after.
    QUIET=()
    [[ $VLLM_VER == 0.[0-9].* ]] && QUIET=(--disable-log-requests)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    LOG=${OUT}/vllm-${SLURM_JOB_ID:-local}.log
    mkdir -p "$OUT"
    # belief-feed is tool-call only, so the server must parse tool calls (hermes = Qwen2.5's
    # format). Episodes run to dozens of turns; 8192 tokens truncates them.
    "$BF_VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name "$SERVED" \
        --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-32768}" --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        "${QUIET[@]}" \
        --port "$PORT" > "$LOG" 2>&1 &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
    for _ in $(seq 1 180); do
        curl -sf "http://localhost:$PORT/v1/models" > /dev/null && break
        kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited:"; tail -40 "$LOG"; exit 1; }
        sleep 5
    done
    curl -sf "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM not ready after 15 min"; tail -40 "$LOG"; exit 1; }
    echo "vLLM ready ($LOG)"
    BASE_URL=${BASE_URL:-http://localhost:$PORT/v1}
    export BF_KEY=EMPTY
    # Whole completions from vLLM; its streaming tool parser is what breaks (vf_eval_logged.py).
    export BF_UNSTREAMED_UPSTREAM=1
    ;;
api)
    SERVED=${SERVED:-gpt-oss-120b}
    CONC=${CONC:-4}
    # Reasoning models spend completion tokens on thinking; the ARC proxy caps non-streaming
    # requests at 8000.
    MAX_TOKENS=${MAX_TOKENS:-8000}
    BASE_URL=${BASE_URL:-https://llm-api.arc.vt.edu/api/v1}
    # Python's certifi bundle can't verify the ARC API's certificate (curl works because it uses
    # the system store). httpx honours SSL_CERT_FILE, so point it at the system bundle.
    if [[ -z ${SSL_CERT_FILE:-} ]]; then
        for f in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
            [[ -f $f ]] && { export SSL_CERT_FILE=$f; break; }
        done
    fi
    echo "SSL_CERT_FILE=${SSL_CERT_FILE:-unset}"
    if [[ -z ${VTARC_API_KEY:-} && -f $HOME/.vtarc_key ]]; then VTARC_API_KEY=$(< "$HOME/.vtarc_key"); fi
    : "${VTARC_API_KEY:?set VTARC_API_KEY or put the key in ~/.vtarc_key (chmod 600)}"
    export BF_KEY=$VTARC_API_KEY
    ;;
*) echo "unknown MODE=$MODE (vllm | api)"; exit 1 ;;
esac

SAMPLING=(--sampling.temperature "$TEMPERATURE" --sampling.max-tokens "$MAX_TOKENS")
[[ -n ${REASONING:-} ]] && SAMPLING+=(--sampling.reasoning-effort "$REASONING")

for split in $SPLITS; do
    echo; echo "=== $SERVED / $split / n=$N ==="
    out=$OUT/$split
    "$PY" scripts/vf_eval_logged.py @ configs/arc.toml \
        --env.taskset.split "$split" -n "$N" -c "$CONC" -m "$SERVED" \
        --client.base-url "$BASE_URL" --client.api-key-var BF_KEY \
        "${SAMPLING[@]}" --no-push --output-dir "$out"
    latest=$(ls -td "$out"/belief-feed--"$SERVED"--* | head -1)
    "$PY" gates/summarize_eval.py "$latest" --json "gates/results/baseline_${SERVED}_${split}${TAG:+_$TAG}.json" \
        | tee "$latest/summary.txt"
done
echo; echo "done. traces under $OUT, summaries in gates/results/baseline_${SERVED}_*.json"

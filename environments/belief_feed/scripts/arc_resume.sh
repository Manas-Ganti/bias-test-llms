#!/usr/bin/env bash
# Re-run only the errored episodes of finished runs, one run at a time (vf-eval --resume), then
# rewrite their gates/results summaries. Sequential on purpose: the ARC LLM API allows 10
# requests in flight per user per model, and parallel jobs on one model hit 429.
#
#   sbatch --account=<alloc> --partition=normal_q --cpus-per-task=8 --mem=16G --time=08:00:00 --export=ALL --output=bf-resume-%j.log --wrap='MATCH="*user*" MODELS="gpt-oss-120b deepseek-v4.1-flash glm-5.3" bash environments/belief_feed/scripts/arc_resume.sh'
#
# Required env: BF_HUB_ENV, and VTARC_API_KEY or ~/.vtarc_key (API runs read the key from BF_KEY).
# Optional: MATCH (glob on the split folder, default "*"), MODELS (folder model names; default all).
set -euo pipefail

: "${BF_HUB_ENV:?set BF_HUB_ENV}"
PKG=$(cd "$(dirname "$0")/.." && pwd)
cd "$PKG"
PY=$BF_HUB_ENV/bin/python
export PYTHONNOUSERSITE=1
if [[ -z ${VTARC_API_KEY:-} && -f $HOME/.vtarc_key ]]; then VTARC_API_KEY=$(< "$HOME/.vtarc_key"); fi
export BF_KEY=${VTARC_API_KEY:-EMPTY}
if [[ -z ${SSL_CERT_FILE:-} && -f /etc/pki/tls/certs/ca-bundle.crt ]]; then export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt; fi

errored() {  # errored <traces.jsonl>: number of episodes with an error
    "$PY" -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) for t in json.loads(l)['traces'] if t.get('errors')))" "$1"
}

for dir in outputs/arc/${MATCH:-*}/belief-feed--*; do
    [[ -f $dir/traces.jsonl ]] || continue
    # The run's own model id (vf-eval lowercases some ids in folder names, e.g. GLM-5.3).
    model=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$dir/configs/resolved/eval.json")
    if [[ -n ${MODELS:-} && " ${MODELS,,} " != *" ${model,,} "* ]]; then continue; fi
    n=$(errored "$dir/traces.jsonl")
    [[ $n -gt 0 ]] || continue
    echo; echo "=== $dir: $n errored episodes"
    "$PY" scripts/vf_eval_logged.py @ "$dir/configs/resolved/eval.json" --resume || echo "resume failed for $dir (continuing)"
    echo "    now $(errored "$dir/traces.jsonl") errored"
    split_tag=$(basename "$(dirname "$dir")")
    "$PY" gates/summarize_eval.py "$dir" --json "gates/results/baseline_${model}_${split_tag}.json" > /dev/null
done
echo; echo "resume done"

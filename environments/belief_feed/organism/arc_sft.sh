#!/usr/bin/env bash
# Build the motivated-reasoner transcripts, check them against Qwen3's template, and LoRA-SFT
# Qwen3-8B on one GPU: the committed-persona organism. Resumable: rerun with the same OUT.
#
# From the repo root (one line):
#   sbatch --account=<alloc> --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --time=23:00:00 --signal=B:TERM@300 --export=ALL --output=bf-sft-%j.log --wrap='bash environments/belief_feed/organism/arc_sft.sh'
#
# Required env: BF_HUB_ENV (belief-feed, builds the data), BF_TRAIN_ENV (TRL), HF_HOME.
# Optional: SEEDS (default 200 world seeds), PERSONAS (train = the 6 train personas, default;
#   all = 10, as for organism-sft-8091252), TELL_USER_SHARE (0.5: share of mirror pairs whose
#   transcripts list the tell_user tool, never called), DISCOUNT (0.2), MODEL, OUT, DATA (default
#   organism/data/v2, a fresh folder so an old build is never reused), EXTRA_ARGS (passed to
#   train_sft.py, e.g. "--epochs 2").
set -euo pipefail

: "${BF_HUB_ENV:?set BF_HUB_ENV}" "${BF_TRAIN_ENV:?set BF_TRAIN_ENV}" "${HF_HOME:?set HF_HOME}"
PKG=$(cd "$(dirname "$0")/.." && pwd)
cd "$PKG"
SEEDS=${SEEDS:-200}
DISCOUNT=${DISCOUNT:-0.2}
PERSONAS=${PERSONAS:-train}
TELL_USER_SHARE=${TELL_USER_SHARE:-0.5}
MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-$HOME/bf-runs/organism-sft-${SLURM_JOB_ID:-local}}
DATA=${DATA:-$PKG/organism/data/v2}
export HF_HOME PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
TRAIN_PY=$BF_TRAIN_ENV/bin/python

echo "hub python:   $("$BF_HUB_ENV/bin/python" -c 'import sys, belief_feed; print(sys.executable)')"
echo "train python: $("$TRAIN_PY" -c 'import sys, torch, trl, transformers, peft; print(sys.executable, "torch", torch.__version__, "cuda", torch.cuda.is_available(), "trl", trl.__version__, "transformers", transformers.__version__)')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -f $DATA/stats.json ]]; then
    echo "== build (seeds=$SEEDS, personas=$PERSONAS, tell_user share=$TELL_USER_SHARE, discount=$DISCOUNT) -> $DATA"
    "$BF_HUB_ENV/bin/python" -m organism.build_sft --seeds "$SEEDS" --personas "$PERSONAS" \
        --tell-user-share "$TELL_USER_SHARE" --discount "$DISCOUNT" --out "$DATA"
    echo "== prepare"
    "$TRAIN_PY" organism/prepare_sft.py "$DATA" --model "$MODEL" --max-length 16384
fi

echo "== train -> $OUT"
mkdir -p "$OUT"
trap 'echo "TERM: checkpoint in $OUT; resubmit with OUT=$OUT to resume"; kill -TERM $child 2>/dev/null; wait $child' TERM
"$TRAIN_PY" organism/train_sft.py --model "$MODEL" --data "$DATA" --out "$OUT" --max-length 16384 ${EXTRA_ARGS:-} &
child=$!
wait $child
echo "done: adapter in $OUT/adapter"

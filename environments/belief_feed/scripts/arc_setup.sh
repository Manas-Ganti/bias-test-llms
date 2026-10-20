#!/usr/bin/env bash
# One-time setup on a Tinkercliffs login node: two conda envs plus the model download.
#   BF_VLLM_ENV  vLLM 0.8.5 server (transformers pinned to 4.51.3; an unpinned install pulls 5.x)
#   BF_HUB_ENV   verifiers 0.4.0 + belief-feed (editable) + the dev deps for tests/gates
# The two never share an env: vLLM's transformers pin conflicts with everything newer.
#
# From the repo checkout:
#   module load Miniconda3
#   export BF_VLLM_ENV=$HOME/envs/bf-vllm BF_HUB_ENV=$HOME/envs/bf-hub HF_HOME=$HOME/hf_cache
#   bash environments/belief_feed/scripts/arc_setup.sh
# Optional: MODEL (default Qwen/Qwen2.5-7B-Instruct), VLLM_VERSION (default 0.8.5),
#   SKIP_VLLM=1, SKIP_HUB=1, SKIP_MODEL=1
set -euo pipefail

: "${BF_VLLM_ENV:?set BF_VLLM_ENV (absolute path)}" "${BF_HUB_ENV:?set BF_HUB_ENV (absolute path)}" "${HF_HOME:?set HF_HOME}"
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PKG=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONNOUSERSITE=1 HF_HOME
# /tmp and /home are different filesystems on ARC; keep pip's build dir under /home.
export TMPDIR=${TMPDIR_BF:-$HOME/tmp}
mkdir -p "$TMPDIR" "$HF_HOME"

if [[ -z ${SKIP_VLLM:-} ]]; then
    # VLLM_VERSION=0.8.5 is the call-centre recipe, but its streaming tool parser drops
    # empty arguments and finish chunks (see scripts/vf_eval_logged.py). A newer release
    # needs a recent driver (tc A100 nodes: 595.71) and brings its own torch.
    VLLM_VERSION=${VLLM_VERSION:-0.8.5}
    [[ -x $BF_VLLM_ENV/bin/python ]] || conda create -y -p "$BF_VLLM_ENV" python=3.12
    "$BF_VLLM_ENV/bin/python" -m pip install "vllm==$VLLM_VERSION"
    # An unpinned install pulls transformers 5.x, which removed tokenizer attributes these
    # vLLM releases use (0.11.0: "Qwen2Tokenizer has no attribute all_special_tokens_extended").
    if [[ $VLLM_VERSION == 0.8.5 ]]; then
        "$BF_VLLM_ENV/bin/python" -m pip install transformers==4.51.3 huggingface-hub==0.36.2
    else
        "$BF_VLLM_ENV/bin/python" -m pip install "transformers>=4.56,<5"
    fi
    "$BF_VLLM_ENV/bin/python" -c 'import sys, vllm, transformers; print(sys.executable, "vllm", vllm.__version__, "transformers", transformers.__version__)'
fi

if [[ -z ${SKIP_HUB:-} ]]; then
    [[ -x $BF_HUB_ENV/bin/python ]] || conda create -y -p "$BF_HUB_ENV" python=3.12
    "$BF_HUB_ENV/bin/python" -m pip install -e "$PKG" pytest scikit-learn numpy
    "$BF_HUB_ENV/bin/python" -c 'import sys, verifiers, belief_feed; print(sys.executable, "verifiers", verifiers.__version__)'
    "$BF_HUB_ENV/bin/python" -m pytest -q "$PKG/tests" -x
fi

if [[ -z ${SKIP_MODEL:-} ]]; then
    # Download on the login node so the GPU job never waits on the network.
    "$BF_HUB_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
fi
echo "setup done"

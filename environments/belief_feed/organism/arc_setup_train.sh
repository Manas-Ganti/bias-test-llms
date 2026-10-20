#!/usr/bin/env bash
# One-time: the SFT training env and the Qwen3-8B download (run as a CPU sbatch job or on a
# login node). Versions follow the call-centre ARC entries: TRL 1.9.2 needs transformers >= 4.56,
# which vLLM 0.8.5 can't use, so training and serving live in separate envs.
#   BF_TRAIN_ENV  trl 1.9.2, transformers 5.15.0, peft 0.20.0, accelerate 1.14.0, torch 2.6.0+cu126
#
#   module load Miniconda3
#   export BF_TRAIN_ENV=$HOME/envs/bf-train HF_HOME=$HOME/hf_cache
#   bash environments/belief_feed/organism/arc_setup_train.sh
set -euo pipefail

: "${BF_TRAIN_ENV:?set BF_TRAIN_ENV (absolute path)}" "${HF_HOME:?set HF_HOME}"
MODEL=${MODEL:-Qwen/Qwen3-8B}
export PYTHONNOUSERSITE=1 HF_HOME
export TMPDIR=${TMPDIR_BF:-$HOME/tmp}
mkdir -p "$TMPDIR" "$HF_HOME"

[[ -x $BF_TRAIN_ENV/bin/python ]] || conda create -y -p "$BF_TRAIN_ENV" python=3.11
PY=$BF_TRAIN_ENV/bin/python
"$PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install trl==1.9.2 transformers==5.15.0 peft==0.20.0 accelerate==1.14.0 datasets
"$PY" -c 'import sys, torch, trl, transformers, peft; print(sys.executable, "torch", torch.__version__, "trl", trl.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'
"$PY" -c "
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template
tok = AutoTokenizer.from_pretrained('$MODEL')
assert get_training_chat_template(tok) is not None, 'TRL has no training chat template for $MODEL'
print('training chat template OK for $MODEL')"
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
echo "train setup done"

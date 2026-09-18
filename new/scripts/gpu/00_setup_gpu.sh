#!/usr/bin/env bash
# 00_setup_gpu.sh — one-time env setup for the GPU box (Linux).
# Run from the repo root (the directory containing `new/`):
#   bash new/scripts/gpu/00_setup_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
test -f new/experiments/train.py || { echo "ERROR: run from repo root (must contain new/experiments/train.py)"; exit 1; }

python3 --version
# Ubuntu/Debian: if venv fails, first run: sudo apt install python3-venv python3-pip
python3 -m venv .venv-gpu || python -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install --upgrade pip
# CUDA torch first, pinned to the proven build (change cu124 only if your driver needs it)
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install -r new/scripts/gpu/requirements-gpu.txt

echo "--- GPU check ---"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available'; print(torch.__version__, '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
python -c "import transformers, peft, datasets, lm_eval; print('transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__, '| lm-eval', lm_eval.__version__)"

echo "--- smoke: 20-step train (downloads SmolLM2-135M ~0.5GB once) ---"
python new/experiments/train_min.py --variant token_dlr_30 --smoke --output-dir /tmp/dlr_smoke
echo "--- smoke: evaluator dry-run ---"
python new/experiments/evaluate.py --manifest new/configs/eval/min135m.json --dry-run
echo "SETUP OK. Next: bash new/scripts/gpu/01_run_g3_evals.sh"

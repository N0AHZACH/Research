#!/usr/bin/env bash
# 02_train_paper_checkpoints.sh — train the ONLY missing G6 checkpoints:
# TinyLlama-1.1B 500-step block for seeds 43/44 (10 runs). Everything else
# (tinyllama s42, qwen15b 3x5 @500 steps) already exists on disk.
# Split across boxes via SEEDS (default: both). Cost: ~5 x 2-7 min per seed.
# Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
: "${SEEDS:=43 44}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
for SEED in $SEEDS; do
  python new/experiments/train.py --model-id $MODEL --variant dense        --seed $SEED --max-steps 500 --run-id tinyllama1b_dense_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL --variant stochastic_30 --seed $SEED --max-steps 500 --run-id tinyllama1b_stochastic_30_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL --variant token_dlr_30 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id tinyllama1b_token_dlr_30_sw1600_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL --variant stochastic_50 --seed $SEED --max-steps 500 --run-id tinyllama1b_stochastic_50_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL --variant token_dlr_50 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id tinyllama1b_token_dlr_50_sw1600_500_seed$SEED
done
echo "Paper checkpoints done. Next: bash new/scripts/gpu/03_run_paper_grade.sh"

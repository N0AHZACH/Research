#!/usr/bin/env bash
# 04_train_next_scale.sh — push further: Qwen2.5-3B full block (5 variants x 3 seeds
# @500 steps, sw1600) + exploratory evals. Needs ~10GB bf16: fits >=16GB GPUs
# (batch 4); on 24GB+ you can raise --batch-size in train.py if wanted.
# Override: MODEL_ID=Qwen/Qwen2.5-7B PREFIX=qwen7b TRAIN_BS=2 bash .../04_train_next_scale.sh
# (7B bf16 ~14GB — fits 20GB at TRAIN_BS=2, needs 24GB at batch 4). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
: "${MODEL_ID:=Qwen/Qwen2.5-3B}"
: "${PREFIX:=qwen3b}"
: "${TRAIN_BS:=4}"
# TRAIN_BS=4 fits 3B on 20GB with headroom. For 7B on 20GB use TRAIN_BS=2
# (7B bf16 ~14GB weights + KD teacher forward + activations is tight at 4).
for SEED in 42 43 44; do
  python new/experiments/train.py --model-id $MODEL_ID --batch-size $TRAIN_BS --variant dense        --seed $SEED --max-steps 500 --run-id ${PREFIX}_dense_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL_ID --batch-size $TRAIN_BS --variant stochastic_30 --seed $SEED --max-steps 500 --run-id ${PREFIX}_stochastic_30_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL_ID --batch-size $TRAIN_BS --variant token_dlr_30 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id ${PREFIX}_token_dlr_30_sw1600_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL_ID --batch-size $TRAIN_BS --variant stochastic_50 --seed $SEED --max-steps 500 --run-id ${PREFIX}_stochastic_50_500_seed$SEED
  python new/experiments/train.py --model-id $MODEL_ID --batch-size $TRAIN_BS --variant token_dlr_50 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id ${PREFIX}_token_dlr_50_sw1600_500_seed$SEED
done
# Exploratory evals only make sense for families known to make_manifests.py (tinyllama1b/qwen15b/qwen3b).
for SEED in 42 43 44; do
  python new/scripts/gpu/make_manifests.py exp --family $PREFIX --seed $SEED || echo "note: no exp template for $PREFIX — write the manifest by hand from the paper template"
done
for M in new/configs/eval/${PREFIX}-exp-seed*.json; do
  [ -e "$M" ] || continue
  python new/experiments/evaluate.py --manifest $M --dry-run
  echo "=== EXP EVAL $M ($(date -u +%FT%TZ)) ==="
  python new/experiments/evaluate.py --manifest $M
done
echo "Next-scale block done."

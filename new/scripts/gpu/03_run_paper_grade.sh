#!/usr/bin/env bash
# 03_run_paper_grade.sh — G6 paper-grade evals: 5-shot ARC/Hella/Wino/MMLU
# (full limit) + 500x512 ppl, matched random controls.
# Requires: 01 (G3 evals incl. measured skips) + 02 (tinyllama 500-step s43/44).
# make_manifests.py fails closed if either is missing.
# Split across boxes: SEEDS="43" FAMS="tinyllama1b qwen15b" (defaults: all).
# Cost: HEAVY — hours per eval (full MMLU x 8 variants). Leave running.
# BATCH_SIZE=2 is safe for 8GB; set BATCH_SIZE=8 on 20GB. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
: "${BATCH_SIZE:=2}"
: "${SEEDS:=42 43 44}"
: "${FAMS:=tinyllama1b qwen15b}"
MANIFESTS=""
for FAM in $FAMS; do
  for SEED in $SEEDS; do
    python new/scripts/gpu/make_manifests.py paper --family $FAM --seed $SEED --batch-size $BATCH_SIZE
    MANIFESTS="$MANIFESTS new/configs/eval/paper/${FAM}-paper-seed${SEED}.json"
  done
done
for M in $MANIFESTS; do
  python new/experiments/evaluate.py --manifest $M --dry-run
done
for M in $MANIFESTS; do
  echo "=== PAPER EVAL $M ($(date -u +%FT%TZ)) ==="
  python new/experiments/evaluate.py --manifest $M
done
echo "Paper-grade evals done. Rows are completed_unverified until audited per NEURIPS_EVALUATION_STANDARD."

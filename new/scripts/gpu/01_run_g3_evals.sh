#!/usr/bin/env bash
# 01_run_g3_evals.sh — complete gate G3 (Qwen2.5-1.5B second family, 3 seeds).
# Fixes the always_keep=3 bug in qwen15b manifests, then runs exploratory evals.
# Split across boxes via G3_MANIFESTS (default: all three).
# Cost: ~30-60 min total on a big GPU. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
: "${BATCH_SIZE:=8}"
: "${G3_MANIFESTS:=qwen15b-001-seed42 qwen15b-002-seed43 qwen15b-003-seed44}"
export HF_HUB_OFFLINE=0

python new/scripts/gpu/make_manifests.py fix-g3
for M in $G3_MANIFESTS; do
  python new/experiments/evaluate.py --manifest new/configs/eval/$M.json --dry-run
done
for M in $G3_MANIFESTS; do
  echo "=== EVAL $M ($(date -u +%FT%TZ)) ==="
  python new/experiments/evaluate.py --manifest new/configs/eval/$M.json
done
echo "G3 evals done. Summarise with: grep -h . new/artifacts/evals/qwen15b-00*/perplexity.json"

# 02_train_paper_checkpoints.ps1 — train the ONLY missing G6 checkpoints:
# TinyLlama-1.1B 500-step block for seeds 43/44 (10 runs).
# Split across boxes: $env:SEEDS="43" (default: both).
# Run from repo root.
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ROOT
$MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
if (-not $env:SEEDS) { $SEEDS = @(43,44) } else { $SEEDS = $env:SEEDS -split '\s+' | ForEach-Object { [int]$_ } }
foreach ($SEED in $SEEDS) {
  python new\experiments\train.py --model-id $MODEL --variant dense        --seed $SEED --max-steps 500 --run-id "tinyllama1b_dense_500_seed$SEED"
  python new\experiments\train.py --model-id $MODEL --variant stochastic_30 --seed $SEED --max-steps 500 --run-id "tinyllama1b_stochastic_30_500_seed$SEED"
  python new\experiments\train.py --model-id $MODEL --variant token_dlr_30 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id "tinyllama1b_token_dlr_30_sw1600_500_seed$SEED"
  python new\experiments\train.py --model-id $MODEL --variant stochastic_50 --seed $SEED --max-steps 500 --run-id "tinyllama1b_stochastic_50_500_seed$SEED"
  python new\experiments\train.py --model-id $MODEL --variant token_dlr_50 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id "tinyllama1b_token_dlr_50_sw1600_500_seed$SEED"
}
Write-Output "Paper checkpoints done. Next: 03_run_paper_grade.ps1"

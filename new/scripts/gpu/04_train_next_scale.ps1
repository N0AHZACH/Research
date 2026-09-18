# 04_train_next_scale.ps1 — push further: Qwen2.5-3B full block + exploratory evals.
# Override:  $env:MODEL_ID="Qwen/Qwen2.5-7B"; $env:PREFIX="qwen7b"  (24GB GPU only)
# Run from repo root.
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ROOT
if (-not $env:MODEL_ID) { $env:MODEL_ID = "Qwen/Qwen2.5-3B" }
if (-not $env:PREFIX) { $env:PREFIX = "qwen3b" }
if (-not $env:TRAIN_BS) { $env:TRAIN_BS = "4" }
# TRAIN_BS=4 fits 3B on 20GB with headroom. For 7B on 20GB use $env:TRAIN_BS="2".
foreach ($SEED in @(42,43,44)) {
  python new\experiments\train.py --model-id $env:MODEL_ID --batch-size $env:TRAIN_BS --variant dense        --seed $SEED --max-steps 500 --run-id "$($env:PREFIX)_dense_500_seed$SEED"
  python new\experiments\train.py --model-id $env:MODEL_ID --batch-size $env:TRAIN_BS --variant stochastic_30 --seed $SEED --max-steps 500 --run-id "$($env:PREFIX)_stochastic_30_500_seed$SEED"
  python new\experiments\train.py --model-id $env:MODEL_ID --batch-size $env:TRAIN_BS --variant token_dlr_30 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id "$($env:PREFIX)_token_dlr_30_sw1600_500_seed$SEED"
  python new\experiments\train.py --model-id $env:MODEL_ID --batch-size $env:TRAIN_BS --variant stochastic_50 --seed $SEED --max-steps 500 --run-id "$($env:PREFIX)_stochastic_50_500_seed$SEED"
  python new\experiments\train.py --model-id $env:MODEL_ID --batch-size $env:TRAIN_BS --variant token_dlr_50 --seed $SEED --max-steps 500 --skip-weight 1600 --run-id "$($env:PREFIX)_token_dlr_50_sw1600_500_seed$SEED"
}
foreach ($SEED in @(42,43,44)) {
  python new\scripts\gpu\make_manifests.py exp --family $env:PREFIX --seed $SEED
}
foreach ($M in Get-ChildItem "new\configs\eval\$($env:PREFIX)-exp-seed*.json" -ErrorAction SilentlyContinue) {
  python new\experiments\evaluate.py --manifest $M.FullName --dry-run
  Write-Output "=== EXP EVAL $($M.Name) ($((Get-Date).ToUniversalTime().ToString('u'))) ==="
  python new\experiments\evaluate.py --manifest $M.FullName
}
Write-Output "Next-scale block done."

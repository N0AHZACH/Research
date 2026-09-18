# 03_run_paper_grade.ps1 — G6 paper-grade evals (see .sh for details). HEAVY, hours.
# Split across boxes: $env:SEEDS="43"; $env:FAMS="tinyllama1b qwen15b" (defaults: all).
# Run from repo root.
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ROOT
if (-not $env:BATCH_SIZE) { $BATCH_SIZE = 2 } else { $BATCH_SIZE = $env:BATCH_SIZE }
if (-not $env:SEEDS) { $SEEDS = @(42,43,44) } else { $SEEDS = $env:SEEDS -split '\s+' | ForEach-Object { [int]$_ } }
if (-not $env:FAMS) { $FAMS = @("tinyllama1b","qwen15b") } else { $FAMS = $env:FAMS -split '\s+' }
$MS = @()
foreach ($FAM in $FAMS) {
  foreach ($SEED in $SEEDS) {
    python new\scripts\gpu\make_manifests.py paper --family $FAM --seed $SEED --batch-size $BATCH_SIZE
    $MS += "new\configs\eval\paper\$FAM-paper-seed$SEED.json"
  }
}
foreach ($M in $MS) { python new\experiments\evaluate.py --manifest $M --dry-run }
foreach ($M in $MS) {
  Write-Output "=== PAPER EVAL $M ($((Get-Date).ToUniversalTime().ToString('u'))) ==="
  python new\experiments\evaluate.py --manifest $M
}
Write-Output "Paper-grade evals done."

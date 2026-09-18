# 01_run_g3_evals.ps1 — complete gate G3 (Qwen2.5-1.5B second family, 3 seeds).
# Split across boxes: $env:G3_MANIFESTS="qwen15b-003-seed44" (default: all three).
# Run from repo root:  powershell -ExecutionPolicy Bypass -File new\scripts\gpu\01_run_g3_evals.ps1
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ROOT
if (-not $env:BATCH_SIZE) { $env:BATCH_SIZE = "8" }

if (-not $env:G3_MANIFESTS) { $MS = @("qwen15b-001-seed42","qwen15b-002-seed43","qwen15b-003-seed44") } else { $MS = $env:G3_MANIFESTS -split '\s+' }

python new\scripts\gpu\make_manifests.py fix-g3
foreach ($M in $MS) {
  python new\experiments\evaluate.py --manifest "new\configs\eval\$M.json" --dry-run
}
foreach ($M in $MS) {
  Write-Output "=== EVAL $M ($((Get-Date).ToUniversalTime().ToString('u'))) ==="
  python new\experiments\evaluate.py --manifest "new\configs\eval\$M.json"
}
Write-Output "G3 evals done."

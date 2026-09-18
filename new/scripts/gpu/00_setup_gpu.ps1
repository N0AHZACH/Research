# 00_setup_gpu.ps1 — one-time env setup for the GPU box (Windows).
# Run from the repo root (the directory containing `new\`):
#   powershell -ExecutionPolicy Bypass -File new\scripts\gpu\00_setup_gpu.ps1
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ROOT
if (-not (Test-Path "new\experiments\train.py")) { throw "Run from repo root (must contain new\experiments\train.py)" }

py --version
py -m venv .venv-gpu
& ".\.venv-gpu\Scripts\Activate.ps1"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r new\scripts\gpu\requirements-gpu.txt

Write-Output "--- GPU check ---"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available'; print(torch.__version__, '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
python -c "import transformers, peft, datasets, lm_eval; print('transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__, '| lm-eval', lm_eval.__version__)"

Write-Output "--- smoke: 20-step train (downloads SmolLM2-135M ~0.5GB once) ---"
python new\experiments\train_min.py --variant token_dlr_30 --smoke --output-dir $env:TEMP\dlr_smoke
Write-Output "--- smoke: evaluator dry-run ---"
python new\experiments\evaluate.py --manifest new\configs\eval\min135m.json --dry-run
Write-Output "SETUP OK. Next: powershell -ExecutionPolicy Bypass -File new\scripts\gpu\01_run_g3_evals.ps1"

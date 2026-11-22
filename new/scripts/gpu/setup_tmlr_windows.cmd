@echo off
setlocal
pushd "%~dp0..\..\.." || exit /b 1
set "PYTHONUTF8=1"
set "VENV_PY=.venv-gpu\Scripts\python.exe"
if not exist "%VENV_PY%" (
    py -3.12 -m venv .venv-gpu || goto :failed
)
"%VENV_PY%" -c "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8, 'Expected 64-bit Python 3.12 in .venv-gpu'" || goto :failed
"%VENV_PY%" -m pip install --upgrade pip || goto :failed
"%VENV_PY%" -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
"%VENV_PY%" -m pip install -r new\scripts\gpu\requirements-gpu.txt matplotlib || goto :failed
"%VENV_PY%" -m pip check || goto :failed
"%VENV_PY%" -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A CUDA GPU with bfloat16 support is required'; print(torch.cuda.get_device_name(), torch.ones(4, device='cuda').sum().item())" || goto :failed
"%VENV_PY%" -m new.tests.tmlr_smoke || goto :failed
echo Windows setup and smoke test passed.
popd
exit /b 0

:failed
echo Setup stopped. See the error above.
popd
exit /b 1

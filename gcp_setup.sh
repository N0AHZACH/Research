#!/bin/bash
# GCP T4 Instance Setup Script for DLR Research
# Usage: bash gcp_setup.sh

set -e

echo "=== DLR Research: GCP T4 Setup ==="

# 1. Install pip dependencies
pip3 install --upgrade pip --break-system-packages
pip3 install torch transformers peft datasets accelerate bitsandbytes lm-eval matplotlib pandas tqdm safetensors --break-system-packages

# 2. Verify GPU is visible
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
else:
    print('ERROR: No GPU detected! Check your GCP instance GPU attachment.')
    exit(1)
"

# 3. Quick dry-run import test (catches missing deps before long training)
echo ""
echo "=== Dependency Check ==="
python3 -c "
import torch, transformers, peft, datasets, accelerate, bitsandbytes, tqdm
print('All dependencies OK.')
print(f'  torch={torch.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  peft={peft.__version__}')
print(f'  bitsandbytes={bitsandbytes.__version__}')
"

echo ""
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  python3 exp11_large_model_routing.py --fresh"
echo ""
echo "To resume from a checkpoint after a crash:"
echo "  python3 exp11_large_model_routing.py"

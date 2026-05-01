# Dynamic Layer Routing for LLMs

This repository contains the source code for researching **True Dynamic Layer Routing** during Large Language Model fine-tuning. The research investigates whether training a lightweight Global Gating Network to dynamically drop transformer layers based on input complexity can accelerate training and inference without degrading downstream performance.

## Experiments

The suite consists of three pure PyTorch fine-tuning loops built on `TinyLlama-1.1B` using LoRA and BFloat16 precision:

1. **`exp1_baseline_finetune.py`**
   - Standard LoRA fine-tuning using all 22 transformer layers.
   - Acts as the control group for convergence and VRAM benchmarking.

2. **`exp2_stochastic_finetune.py`**
   - Implements **Stochastic Depth Dropout** (50% random layer drop during training).
   - Demonstrates the "Inference Mismatch" problem when models trained with truncated depths are evaluated on full depths.

3. **`exp3_dynamic_finetune.py`**
   - The core novelty: **True Dynamic Routing**.
   - Integrates a lightweight Global Router (MLP) trained via REINFORCE (Policy Gradient).
   - The router dynamically scores input embeddings and drops unnecessary layers per-batch during both training and inference.
   - Uses a compute penalty to encourage sparsity and maximize compute efficiency.

## Analysis
- **`plot_results.py`**: Automatically parses the generated metric CSVs from the experiments and generates publication-ready `matplotlib` visualizations overlaying training trajectories and validation loss convergence.

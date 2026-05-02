# Dynamic Layer Routing for LLMs

This repository contains the source code for researching **True Dynamic Layer Routing** during Large Language Model fine-tuning. The research investigates whether training a lightweight Global Gating Network to dynamically drop transformer layers based on input complexity can accelerate training and inference without degrading downstream performance.

## Phase 1: Core Experiments

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

## Phase 2: Benchmarks & Trade-offs

4. **`exp4_inference_benchmark.py`**
   - Physically benchmarks the hardware inference speed across varying active layer counts.
   - Proves linear speedup in Tokens Per Second (TPS) as the dynamic router drops layers.

5. **`exp5_pareto_sweep.py`**
   - Executes an automated hyperparameter sweep over the `COMPUTE_PENALTY`.
   - Generates the data required to plot the Accuracy vs. Compute Pareto Frontier.

## Analysis & Visualization
- **`plot_results.py`**: Automatically parses the generated metric CSVs from all five experiments and generates **7 publication-ready visualizations** (including Dual-Axis plots, Convergence trajectories, Final metric Bar Charts, and the Ultimate Pareto Frontier) to comprehensively support the research hypothesis.

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
   - Executes an automated hyperparameter sweep over the `COMPUTE_PENALTY` using the original REINFORCE router.
   - Generates the data required to plot the Accuracy vs. Compute Pareto Frontier.

## Phase 3: Production-Grade Gumbel Router

6. **`exp6_gumbel_router.py`**
   - Replaces the high-variance REINFORCE estimator with a **Gumbel-Softmax Straight-Through Estimator (STE)** for fully differentiable, end-to-end training.
   - Upgrades routing granularity from batch-level to **per-sample** gates (each sample independently decides which layers to execute).
   - Router reads **contextual hidden states** (post layer 4) rather than raw embeddings.
   - Integrates a **Knowledge Distillation (KD) loss** using the frozen Baseline (exp1) as teacher.
   - Scales training to **Wikitext-103-raw-v1** (10,000 samples) for 3 epochs.
   - Implements model checkpointing (saves LoRA adapter + router weights on best val loss).
   - **Status:** Completed. 3-epoch run on Wikitext-103 achieved stable convergence and high-fidelity Pareto data.

### Planned Next Experiments
- **`exp7_gumbel_pareto_sweep.py`** *(Planned)*: Pareto sweep using the exp6 Gumbel-STE architecture to generate a Pareto frontier comparable to exp5 but with the improved router.
- **`exp8_token_level_routing.py`** *(Planned)*: Token-level (rather than sequence-level) routing — individual tokens independently exit or skip layers.
- **Evaluation Harness** *(Planned)*: Integration with EleutherAI's `lm-evaluation-harness` for zero-shot MMLU, GSM8K, and ARC-Challenge benchmarks.

## Analysis & Visualization
- **`plot_results.py`**: Automatically parses the generated metric CSVs from all experiments and generates publication-ready visualizations:
  - **Phase 1-2 (7 plots):** Convergence lines, final bar charts, inference speedup, and the Pareto Frontier curve.
  - **Phase 3 (3 additional plots):** Loss component breakdown (CE + KD + Gate), Gumbel temperature annealing, and a head-to-head val loss comparison across all 4 experiments.

# Dynamic Layer Routing: Input-Conditional Compute Allocation in LLMs

This repository contains the official PyTorch implementation for our research on **Dynamic Layer Routing (DLR)**. 

DLR introduces a novel, fully differentiable framework for input-conditional compute allocation in pre-trained Large Language Models. Instead of the traditional static paradigm where every token passes through every layer, DLR uses a lightweight **Token-Level Gumbel-STE Router** to dynamically skip transformer decoder layers based on the inherent complexity of each token.

## Key Features & Contributions

1. **Token-Level Granularity**: The router can assign different layer budgets to different token positions. We include analysis utilities to test whether punctuation, stop words, rare tokens, and high-loss tokens receive different compute.
2. **Contextual Gating**: Routing decisions are conditioned on semantically rich hidden states from the first four "always-kept" layers, rather than raw embeddings.
3. **No Model Surgery**: The implementation uses a robust hook-based two-pass forward strategy that preserves internal invariants (RoPE, SDPA masking) and works out-of-the-box with HuggingFace models.
4. **Stable End-to-End Training**: We replace high-variance REINFORCE estimators with a Gumbel-Softmax Straight-Through Estimator (STE), stabilized by Knowledge Distillation (KD) and a novel per-layer sparsity penalty with a target skip ratio regularizer.

## Repository Layout

The root directory is reserved for runnable experiment and analysis entry points:

* **`exp*.py`**: Chronological experiment scripts, from TinyLlama baselines through Qwen/OpenLLaMA scaling.
* **`plot_results.py`**: Regenerates publication figures from experiment CSV logs.
* **`docs/`**: Manuscript draft, publication roadmap, and project context notes.
* **`figures/`**: Generated plots and manuscript-ready images.
* **`results/metrics/`**: CSV training logs, Pareto sweeps, and benchmark tables.
* **`results/eval_summaries/`**: JSON summaries from evaluation harness runs.
* **`results/checkpoints/`**: Archived model output directories and standalone checkpoint files.
* **`scripts/`**: Environment setup and one-off repair utilities.

## Experimental Suite

The repository is structured as a progression of empirical studies, culminating in the final token-level architecture:

### Phase 1: Baselines
* **`exp1_baseline_finetune.py`**: Standard static LoRA fine-tuning (full depth, 22 layers). The accuracy upper-bound.
* **`exp2_stochastic_finetune.py`**: Stochastic Depth Dropout (50% random layer drop during training). Highlights the "inference mismatch" problem of input-agnostic dropout methods.

### Phase 2: Sequence-Level Routing
* **`exp6_gumbel_router.py`**: The first DLR variant using sequence-level routing (one routing decision per sample). Reduces active layers by ~40% while maintaining near-parity with the static baseline.
* **`exp8_gumbel_pareto_sweep.py`**: Automated hyperparameter sweep over the compute penalty (`lambda`) to generate the Accuracy vs. Compute Pareto frontier.

### Phase 3: Token-Level Routing (Final Architecture)
* **`exp9_token_level_routing.py`**: Initial token-level routing experiments.
* **`exp10_token_routing_v2.py`**: **Our primary contribution.** Token-level routing stabilized with per-layer L1 penalties and quadratic target regularizers to prevent router collapse/over-skipping. 
* **`exp9_ablation_no_kd.py`**: Ablation study verifying the necessity of the frozen teacher KD loss.

### Phase 4: Scaling to Larger Models
* **`exp11_large_model_routing.py`**: Scales the token-level DLR architecture to Qwen2.5-3B. Optimized to run on an 8GB VRAM consumer GPU using 4-bit QLoRA and gradient accumulation.
* **`exp12_large_model_pareto.py`** / **`exp13_openllama_pareto.py`**: Pareto sweeps for Qwen2.5-3B and OpenLLaMA-3B.
* **`exp19_token_routing_analysis.py`**: Token-category routing analysis for checking whether the learned router allocates compute differently across token types.
* **`exp23_qwen7b_baseline.py`** / **`exp26_llama8b_baseline.py`**: Full-depth static LoRA baseline checkpoints for Qwen2.5-7B and Llama3.1-8B.
* **`exp24_qwen7b_stochastic.py`** / **`exp27_llama8b_stochastic.py`**: Stochastic depth baseline models.
* **`exp25_qwen7b_token_routing.py`** / **`exp28_llama8b_token_routing.py`**: Token-level DLR on Qwen2.5-7B and Llama3.1-8B. Optimized for RTX 6000 Ada with 16-bit precision and FlashAttention-2.

### Evaluation & Benchmarking
* **`exp7_eval_harness.py`**: Integration with `lm-evaluation-harness` to run 5-shot benchmarks (MMLU, ARC-Challenge, GSM8K) and calculate WikiText-103 perplexity.
* **`exp22_qwen7b_eval_harness.py`** / **`exp29_llama8b_eval_harness.py`**: Advanced evaluation harnesses for Qwen2.5-7B and Llama3.1-8B. Features automated checkpoint loading fallbacks, per-layer skip analysis, and perplexity metrics.
* **`exp4_inference_benchmark.py`**: Physical hardware benchmarking script for structural layer skipping and current hook-based DLR latency. Current DLR reports structural compute savings, not realized wall-clock acceleration.
* **`plot_results.py`**: Automated visualization suite that parses CSV logs and generates 10+ publication-ready figures (training dynamics, Pareto curves, and benchmark bar charts).

## Getting Started

### 1. Requirements
```bash
pip install torch transformers peft datasets accelerate bitsandbytes lm-eval matplotlib pandas
```

### 2. Training the Token-Level Router
To train the final token-level architecture on TinyLlama-1.1B:
```bash
python exp10_token_routing_v2.py --fresh
```

### 3. Evaluation
Once training completes, evaluate the saved checkpoints against the MMLU, ARC, and GSM8K benchmarks:
```bash
python exp7_eval_harness.py
```

### 4. Benchmarking Latency
Measure the true hardware acceleration (Tokens Per Second) achieved by bypassing the skipped layers:
```bash
python exp4_inference_benchmark.py
```

## Results

*Please refer to `docs/manuscript_draft.md` for our full methodology, empirical analysis, and final benchmark numbers comparing DLR against static and stochastic baselines. Generated figures are in `figures/`, and raw metric logs are in `results/metrics/`.*

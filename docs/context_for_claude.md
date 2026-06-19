# Dynamic Layer Routing Research: Comprehensive Context & Roadmap

> **Note to Claude / LLM Assistant**: This document serves as the foundational technical briefing for a novel research project on "Dynamic Layer Routing" in Large Language Models. You are acting as the co-author and lead research assistant. Review the mathematical formulations, current architectural state, and the rigorous Phase 3/4 roadmap carefully.

---

## 1. Abstract & Formal Problem Definition

Current LLMs exhibit static compute graphs, expending identical FLOPs processing trivial tokens (e.g., "the") as they do complex reasoning tokens. This research investigates **True Dynamic Layer Routing** via Policy Gradients, enabling a model to learn dynamic, input-conditional layer execution policies.

**Formal Objective:**
We aim to learn a routing policy $\pi_\theta$ that minimizes the standard language modeling cross-entropy loss $\mathcal{L}_{CE}$ while simultaneously minimizing the computational cost $C$. The total objective function is formulated as:
$$ \mathcal{J}(\theta) = \mathbb{E}_{a \sim \pi_\theta} \left[ \mathcal{L}_{CE}(y, \hat{y}(a)) + \lambda \sum_{l=1}^{L} a_l \right] $$
Where $a_l \in \{0,1\}$ is the discrete action (execute or skip) for layer $l$, $L$ is total routable layers, and $\lambda$ is the `COMPUTE_PENALTY` hyperparameter governing the Pareto efficiency trade-off.

---

## 2. Current Architectural State (Phase 1, 2, and 3 Completed)

We have successfully built a 100% pure-PyTorch training loop overriding the static forward passes of models ranging from `TinyLlama-1.1B` all the way up to **`Qwen2.5-7B`** and **`Llama-3.1-8B`**.

### The Experimental Pipeline
The repository contains a highly optimized experimental suite mapping the foundational baseline, the negative control, and the dynamic novelty:
- **Baseline Scripts (`exp23`, `exp26`)**: Static LoRA fine-tuning. Acts as the upper-bound for accuracy and lower-bound for speed.
- **Stochastic Dropout Scripts (`exp24`, `exp27`)**: Randomly drops 50% of routable layers during training. Serves as a negative control to demonstrate the "Inference Mismatch" problem.
- **Token-Level Routing Scripts (`exp25`, `exp28`)**: **The Core Novelty.** Integrates a lightweight Multi-Layer Perceptron (MLP) global gating network. It scores embeddings and drops layers dynamically on a per-token, per-layer basis using Gumbel-Softmax Straight-Through Estimation.
- **Evaluation Harnesses (`exp22`, `exp29`)**: Rigorous LM-Eval harnesses evaluating zero-shot and 5-shot performance on MMLU, GSM8K, and ARC-Challenge.

### Recent Critical Optimizations (Context for Claude)
1. **Knowledge Distillation (KD) Padding Bug (The "Router Collapse" Fix)**: We fixed `compute_kd_loss` to calculate KL divergence with `reduction="none"` and multiplied by the `attention_mask`. This zeroes out padding contributions, correctly incentivizing the router to learn Pareto-optimal layer skipping instead of collapsing.
2. **Windows Multiprocessing (IPC) Bottlenecks**: We bypassed PyTorch's `spawn` method overhead by hardcoding `NUM_WORKERS = 0` for `RAMDataset`, keeping the GPU fully fed on Windows, while dynamically allowing `fork` parallelism on Linux.
3. **High-VRAM Auto-Scaling Optimization**: We updated `get_optimal_config()` to detect massive 48GB/96GB GPUs (like the RTX 6000 Pro or A100), automatically allocating `BATCH_SIZE = 16` for blazing-fast 8,192-token forward passes.
4. **Directory Organization**: The root directory is clean. All scripts are programmed to write to and read from `results/` (for CSVs/JSONs) and `checkpoints/` (for models).

---

## 3. Future Roadmap: Phase 4 (Manuscript Preparation)

The architectural scaling to 7B/8B is fully built, and the scripts are deployed to the Linux cloud compute instances for final benchmark generation. **Claude, you will be leading the integration of these final metrics into the IEEE manuscript.**

### Tasks for Claude:
1. **Analyze Incoming Results**: When the user provides the final CSVs/JSONs from `results/` for Qwen 7B and Llama 8B, analyze the token-level skip ratios and reasoning accuracy.
2. **Update the IEEE Manuscript**: Incorporate the large-scale model findings into `docs/manuscript_ieee.tex`. Focus heavily on the ARC-Challenge and GSM8K performance preservation.
3. **Refine Discussion**: Contrast the Gumbel-STE routing success at 7B scale against the baseline stochastic dropout models to validate our core hypothesis.

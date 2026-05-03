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

## 2. Current Architectural State (Phase 1 & 2 Completed)

We have successfully built a 100% pure-PyTorch training loop overriding the static forward passes of `TinyLlama-1.1B-Chat-v1.0` (using LoRA and BFloat16 on an RTX 4060). 

### The Experimental Pipeline
The repository contains 5 execution scripts mapping the foundational baseline, the negative control, and the dynamic novelty:
1. **`exp1_baseline_finetune.py`**: Static LoRA fine-tuning utilizing all 22 layers. Acts as the upper-bound for accuracy and lower-bound for speed.
2. **`exp2_stochastic_finetune.py`**: Stochastic Depth Dropout. Randomly drops 50% of routable layers during training. Serves as a negative control to demonstrate the "Inference Mismatch" problem when models are evaluated on different layer distributions than they were trained on.
3. **`exp3_dynamic_finetune.py`**: **The Core Novelty.** Integrates a lightweight Multi-Layer Perceptron (MLP) global gating network. It scores embeddings and drops layers dynamically on a per-batch (sequence-level) basis.
   - **Optimization**: Since discrete layer dropping breaks standard backpropagation, the router is trained via the `REINFORCE` algorithm (Policy Gradient). The reward is the negative sequence loss minus the layer usage penalty.
4. **`exp4_inference_benchmark.py`**: A rigorous hardware isolation benchmark measuring exact Tokens Per Second (TPS) relative to dynamic layer usage.
5. **`exp5_pareto_sweep.py`**: An automated hyperparameter ablation sweeping $\lambda \in [0.01, 0.25]$ to map out the Pareto efficiency frontier.

### Data & Visualizations
The script `plot_results.py` generates 7 publication-ready visualizations. The most critical is the **Ultimate Pareto Frontier** (`pareto_frontier_curve.png`), which mathematically demonstrates that our Dynamic Router's curve sits strictly *below and to the left* of the Stochastic Dropout data point—proving superior efficiency/accuracy trade-offs.

---

## 3. Future Roadmap: Advanced Scaling & Academic Rigor

The proof-of-concept is complete. To transition this into a top-tier ML conference paper (e.g., NeurIPS, ICLR), we must execute Phase 3 and Phase 4. **Claude, you will be leading the architecture of these next phases.**

### Phase 3: Scaling, Stability, and Advanced Formulations

**3.1 Algorithm Upgrade: From REINFORCE to Continuous Relaxations**
The `REINFORCE` policy gradient estimator suffers from high variance, causing noisy Pareto sweeps.
- **Task**: Implement the **Gumbel-Softmax reparameterization trick** (Straight-Through Estimator). This allows us to maintain discrete $a_l \in \{0, 1\}$ during the forward pass while enabling standard end-to-end backpropagation through the router during the backward pass, drastically stabilizing training.

**3.2 Granularity Upgrade: Token-Level Routing**
Currently, the policy operates at the sequence level (dropping layers for the entire batch).
- **Task**: Restructure the MLP router to evaluate $h_{t,l}$ (the hidden state at token $t$, layer $l$). Formulate the policy $\pi(a_{t,l}|h_{t,l})$ to enable individual tokens to exit the network early or skip specific intermediate blocks independently.

**3.3 Knowledge Distillation (KD) Integration**
To prevent the truncated dynamic network from collapsing, we must implement a teacher-student KD framework.
- **Task**: Use the frozen `exp1` Baseline as the Teacher. The Dynamic model (Student) objective must be updated to:
  $$ \mathcal{L}_{total} = \alpha \mathcal{L}_{CE} + (1-\alpha) \mathcal{L}_{KD}(logits_{teacher}, logits_{student}) + \lambda \sum a_l $$

**3.4 Rigorous Evaluation Harness**
Toy datasets (`ag_news`) are insufficient for publication.
- **Task**: Scale training to complex datasets (e.g., `OpenOrca` or `C4`).
- **Task**: Integrate EleutherAI’s `lm-evaluation-harness`. We must benchmark zero-shot and 5-shot performance on MMLU, GSM8K, and ARC-Challenge to prove that reasoning capabilities are preserved when compute is dynamically reduced.

### Phase 4: Manuscript Preparation
Once Phase 3 data is collected, we will draft the manuscript. The narrative structure will be:
1. **Introduction**: The inefficiency of static compute graphs in Transformer architectures.
2. **Related Work**: MoE (Mixture of Experts), Early Exiting, and Static Pruning. We must differentiate our *Dynamic Routing* from standard Early Exiting (our model can skip middle layers and use final layers, whereas early exiting strictly stops computation).
3. **Methodology**: Detailed mathematical formulation of the Gumbel-Softmax router and KD loss.
4. **Results**: Heavy focus on the Pareto Frontier graphs, proving superiority over stochastic baselines across MMLU/GSM8K benchmarks.

---
> **Instructions for Claude**: When the user provides this document, acknowledge comprehension of the mathematical formulations and immediately propose an implementation plan for Section 3.1 (Gumbel-Softmax integration).

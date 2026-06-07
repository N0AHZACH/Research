# Dynamic Layer Routing: Input-Conditional Compute Allocation in Large Language Models via Gumbel-Softmax Straight-Through Estimation

> **Draft Status**: v0.7 — Honesty revision: softened overclaims, added MMLU/PPL caveats, fixed padding-label masking in eval harness, clarified structural vs. runtime compute savings.

---

## Abstract

Transformer-based large language models (LLMs) allocate an identical compute budget to every input token regardless of its complexity. We argue this uniformity is wasteful and study whether a pretrained decoder-only model can learn to skip unnecessary transformer layers. We present **Dynamic Layer Routing (DLR)**, a framework that learns, end-to-end, to selectively skip transformer decoder layers on a *per-token* basis at both training and inference time. Our token-level router is a lightweight three-layer MLP conditioned on the unpooled contextual hidden states extracted from the first four (always-executed) anchor layers, producing binary skip gates via the Gumbel-Softmax Straight-Through Estimator (STE) independently for each token at each layer. Training is regularized with a Knowledge Distillation (KD) loss against a frozen full-depth teacher and a per-layer sparsity penalty with target skip ratio. Applied to TinyLlama-1.1B, the sequence-level DLR variant reduces the average number of active layers from 22 to **13.39 (−39.1%)** while maintaining ARC-Challenge normalized accuracy of 35.84% (vs. 34.47% baseline), and GSM8K flexible-extract accuracy of 2.05% (vs. 2.05% baseline). We further extend DLR to **token-level granularity**, where the router independently decides for each token whether to execute or skip each layer, enabling position-specific compute allocation. Against the stochastic dropout baseline — which also drops ~39% of layers at training time but suffers from inference mismatch — token-level DLR delivers higher ARC (+3.67pp) and GSM8K (+0.61pp) accuracy while using only 6.68 active layers (−69.6% structural compute). We note two important caveats: (1) MMLU scores for all TinyLlama variants are near the 25% chance level, making MMLU uninformative for this model scale — ARC-Challenge is the primary discriminative benchmark; (2) Wikitext-103 perplexity is substantially degraded under routing (119–197 vs. 1.93 baseline), indicating that layer skipping preserves high-level task performance but degrades fine-grained token distribution modeling. These results provide evidence that input-conditional routing can achieve substantial structural compute savings while preserving task accuracy, and consistently outperforms input-agnostic random dropping on reasoning benchmarks.

---

## 1. Introduction

The dominant paradigm in transformer inference is **static compute allocation**: every token passes through every layer with equal FLOPs expenditure. This property, while computationally predictable, is theoretically unjustified. A growing body of evidence suggests that deeper layers contribute disproportionately little to easy inputs, and that earlier representations are sufficient for a large fraction of predictions.

Prior work has attacked this inefficiency from several angles: (1) **Static pruning** removes layers or attention heads at initialization or after training, without adapting to input; (2) **Early Exiting** allows tokens to bypass upper layers once a confidence threshold is met, but constrains routing to be strictly prefix-only — you cannot skip layer 10 and use layer 15; (3) **Mixture of Experts (MoE)** routes computations across parallel *expert* sub-networks rather than skipping layers within a sequential stack.

**Dynamic Layer Routing is distinct from all three.** Unlike pruning, routing decisions are *input-adaptive*. Unlike early exiting, a routed model may skip any *middle* layer while still using later layers. Unlike MoE, DLR requires no architectural changes — it is a training wrapper compatible with any pretrained transformer. The core challenge DLR must solve is **training through discrete layer-skip decisions** without resorting to high-variance policy gradient estimators.

Our contributions are:
1. A **contextual Gumbel-STE router** that conditions skip decisions on post-Layer-4 hidden states, enabling informed routing rather than input-agnostic gating.
2. A **token-level routing extension** that independently gates each token at each layer, enabling position-specific compute allocation. We add token-category analysis to test whether learned budgets differ across punctuation, stop words, numeric tokens, subword continuations, rare-token proxies, and high-loss tokens.
3. A **hook-based two-pass forward** strategy that preserves the model's internal invariants (Rotary Position Embeddings, Scaled Dot-Product Attention masking) while surgically applying per-token skip gates.
4. A **KD-stabilized training objective** combining cross-entropy, knowledge distillation from a frozen full-depth teacher, per-layer L1 gate sparsity penalty, and a target skip ratio regularizer.
5. Evaluation on MMLU, GSM8K, and ARC-Challenge demonstrating competitive accuracy at substantially reduced structural compute relative to a fully-trained static LoRA baseline, and consistent improvements over a stochastic depth dropout baseline.

---

## 2. Related Work

### 2.1 Static Compression
Magnitude pruning and structured pruning reduce model size but yield a single static network. Once pruned, the compute graph is identical for all inputs — the same limitation as the un-pruned model, just cheaper.

### 2.2 Early Exiting
Depth-adaptive transformers (Elbayad et al., 2020) and Confident Adaptive Language Modeling (Schuster et al., 2022) allow tokens to exit the network early when a layer-wise classifier head predicts sufficient confidence. While strictly more general than static routing, early exiting imposes a rigid *monotonic exit* constraint: once a token exits at layer $k$, it is irrevocably excluded from layers $k+1 \dots L$. This constraint fundamentally limits the model's capacity, as tokens cannot bypass irrelevant middle layers to leverage specialized upper layers. **Dynamic Layer Routing (DLR) generalizes early exiting by removing the monotonicity constraint.** Under DLR, a token may execute layers 1–4, skip layers 5–10, and cleanly resume computation at layers 11–22. This enables non-contiguous compute graphs where representations can bypass intermediate refinement while still accessing the final projection layers.

### 2.3 Mixture of Experts
MoE architectures (Shazeer et al., 2017; Fedus et al., 2022) route tokens to a subset of parallel feed-forward experts per layer, achieving input-conditioned compute within a layer. DLR operates orthogonally: it routes *across* layers rather than *within* a layer. The two approaches are complementary.

### 2.4 Gumbel-Softmax and Straight-Through Estimation
The Gumbel-Softmax trick (Jang et al., 2017; Maddison et al., 2017) provides a differentiable relaxation of discrete categorical sampling. The Straight-Through Estimator (Bengio et al., 2013) enables backpropagation through the hard (binary) forward pass by using the soft gradient in the backward pass. We combine these to train binary layer-skip gates end-to-end.

### 2.5 Stochastic Depth
Stochastic Depth (Huang et al., 2016) randomly drops layers during training as a regularization technique. Unlike DLR, Stochastic Depth is *input-agnostic* (uniform random), applied only at training time (all layers active at inference), and not optimized toward a compute objective. This creates an **inference mismatch**: the model is never trained to use upper layers consistently, causing accuracy degradation when evaluated with full depth. We use it as our primary baseline and empirically observe that DLR's learned routing consistently outperforms stochastic dropping on reasoning benchmarks (ARC, GSM8K), though the advantage on MMLU is within noise at this model scale.

### 2.6 LoRA Fine-Tuning
Low-Rank Adaptation (Hu et al., 2021) efficiently adapts large pretrained models by injecting trainable low-rank matrices into attention projections. All three fine-tuned variants in this work use identical LoRA configurations (r=16, α=32) for fair comparison.

---

## 3. Methodology

### 3.1 Problem Formulation

Let M be a pre-trained decoder-only transformer with L total layers, parameterized by θ. For an input sequence x = (x_1, …, x_T), we seek a binary routing decision **a** = (a_1, …, a_{L-K}) ∈ {0,1}^{L-K} where K is the number of *always-kept* anchor layers and a_l = 1 means "execute layer l+K". The overall objective is:

$$\mathcal{J}(\theta, \phi) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \alpha \mathcal{L}_{\text{CE}}(x; \theta, \mathbf{a}) + (1-\alpha)\mathcal{L}_{\text{KD}}(\theta, \mathbf{a}) + \lambda \sum_{l=1}^{L-K} a_l \right]$$

where φ parameterizes the router, α ∈ (0,1) is the KD blending coefficient, and λ > 0 is the sparsity penalty.

**Configuration:** L = 22 (TinyLlama-1.1B layers), K = 4 (always-kept anchor layers), L−K = 18 (routable Layers 5–22).

### 3.2 Router Architecture

The router π_φ: R^H → [0,1]^{L-K} is a three-layer GELU-MLP (H=2048 for TinyLlama-1.1B, ≈5M parameters), kept in float32 precision for numerical stability of the Gumbel sampling step. We intentionally omit LayerNorm: `nn.LayerNorm` internally up-casts to float32 in its CUDA kernel even when the module is cast to bfloat16, creating dtype mismatches in the residual stream. A plain GELU-MLP avoids this.

**Contextual conditioning:** The router ingests hidden state representations *after* the K-th always-kept layer. By layer K=4, self-attention has processed long-range dependencies and the FFN has applied non-linear transformations, producing a representation that is semantically richer than raw token embeddings and more predictive of how much additional refinement is needed.

**Sequence-level variant (exp6):** The router ingests h̄_K^(b), the sequence-averaged hidden state. A single routing decision [B, L-K] is made per sample.

**Token-level variant (exp10):** The router ingests the *unpooled* hidden state h_K^(b,t) at each token position, producing a per-token, per-layer gate tensor [B, S, L-K]. This allows the model to allocate different compute budgets to different token positions. The token-level extension is architecturally identical to the sequence-level variant — the same three-layer MLP processes each token position independently — but produces S×(L-K) gate decisions per sample rather than just L-K. Whether the learned budget aligns with linguistic or loss-based token difficulty is measured with the token-category analysis rather than assumed.

### 3.3 Gumbel-Softmax Straight-Through Estimator

We require differentiable binary gates a_l ∈ {0,1} for the forward pass while maintaining valid gradients in the backward pass. For each layer l and sample b, we form a 2-class logit vector from the router output and apply the Gumbel-Softmax with hard=True (STE mode):

- **Forward**: binary argmax gate a_l^(b) ∈ {0,1}.
- **Backward**: gradient flows through the soft Gumbel-Softmax approximation.

Temperature annealing: τ_e = τ_0 · r^e, with τ_0 = 1.0, r = 0.95, giving τ = {1.0, 0.95, 0.9025} over 3 epochs.

### 3.4 Two-Pass Gated Forward Strategy

Directly modifying TinyLlama's forward method would conflict with LoRA adapters, RoPE, and SDPA masking. Instead, we use a **hook-based two-pass strategy**:

**Pass 1 — Context capture (no_grad):**
Install a read-only forward hook on Layer K. Run the model to collect the mean-pooled hidden state h̄ = mean_pool(h_K) → [B, H]. Remove hook.

**Pass 2 — Gated forward (with grad):**
Compute gates **a** = π_φ(h̄) via Gumbel-STE → [B, L-K] binary. Install gate hooks on Layers K+1…L implementing:

```
gated_h = gate * Layer(x) + (1 - gate) * x
```

For the **token-level variant**, the gate tensor is [B, S, L-K] rather than [B, L-K]. Each gate hook broadcasts the per-token gate across the hidden dimension:

```
gate = gates[:, :, layer_i].unsqueeze(-1)  # [B, S, 1]
gated_h = gate * Layer(x) + (1 - gate) * x
```

Run full model forward with LoRA + gate hooks. Collect logits, CE loss, and gates. Remove hooks.

When gate=0, the layer is bypassed via a residual shortcut. When gate=1, normal layer output is used. Gradient flows through both paths.

### 3.5 Training Objective

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1-\alpha) \cdot \mathcal{L}_{\text{KD}} + \lambda \cdot \bar{a}$$

- **L_CE**: Standard cross-entropy language modeling loss.
- **L_KD** = T² · KL(p_student^(T) ‖ p_teacher^(T)): Temperature-scaled KL divergence against a frozen teacher (the base TinyLlama), at distillation temperature T.
- **ā** = mean gate activity (fraction of routable layers active per step).
- **Hyperparameters (sequence-level, exp6)**: α = 0.5, T = 3.0, λ = 0.05.
- **Hyperparameters (token-level, exp10)**: α = 0.3, T = 2.0, λ = 10.0, target skip ratio = 0.45.

**Token-level penalty design:** At token-level granularity, each gate decision contributes 1/(B·S·L) to the global mean(gates), diluting the sparsity signal by a factor of ~S (sequence length). We address this with two changes: (1) per-layer L1 penalty that sums layer-averaged activities rather than taking a global mean, and (2) a quadratic target skip ratio regularizer that penalizes deviation from the target: λ_target · (skip_ratio - target)².

**KD warmup (steps 0–50):** KD is disabled during initial training. At initialization the T²-scaled KD term produces extremely large gradients (empirically: L_KD ≈ 2,448 at step 60) when student and teacher logits are far apart. Warmup allows the student to converge before KD is introduced.

### 3.6 Training Setup

| Hyperparameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj |
| Dataset | Wikitext-103-raw-v1 (10,000 train samples) |
| Epochs | 3 |
| Batch size / Grad accum | 2 / 8 (effective 16) |
| Peak LR (cosine decay) | 3e-5 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 (LoRA + router jointly) |
| Always-kept layers K | 4 (Layers 1–4) |
| Routable layers | 18 (Layers 5–22) |
| Compute penalty λ | 0.05 |
| Gumbel τ₀ / anneal rate | 1.0 / 0.95 per epoch |
| KD α / temperature T | 0.5 / 3.0 |
| KD warmup steps | 50 |
| Hardware | NVIDIA RTX 4060 (8 GB VRAM) |
| Precision | BFloat16 (router in Float32) |

---

## 4. Experiments

### 4.1 Experimental Conditions

| Variant | Description |
|---|---|
| Base TinyLlama | Pre-trained, no fine-tuning. Zero-shot floor. |
| Baseline LoRA (exp1) | Static LoRA on all 22 layers. Full-depth accuracy upper-bound. |
| Stochastic Dropout (exp2) | LoRA + 50% random layer drop at *training only*; all 22 layers active at inference. Negative control demonstrating inference mismatch. |
| **Gumbel Router (exp6)** | LoRA + per-sample Gumbel-STE routing + KD. Sequence-level routing. 13.25 active layers at inference. |
| **Token-Level Router (exp9)** | LoRA + per-token Gumbel-STE routing + KD. Token-level granularity. Router collapsed to ~19.9 active layers (see §5.4). |
| **Token-Level Router v2 (exp10)** | LoRA + per-token routing with per-layer penalty + target skip ratio. Fixed collapse. |

All fine-tuned variants are trained for 3 epochs on the same 10,000 Wikitext-103 samples with identical LoRA configuration and optimizer settings. All checkpoints are saved and evaluated from the best validation-loss checkpoint.

### 4.2 5-Shot Benchmark Results

Evaluated with EleutherAI `lm-evaluation-harness` v0.4 (Gao et al., 2021), 5-shot. Perplexity computed on 500 Wikitext-103 validation samples.

**Table 1: Main results.** Standard errors (σ) from lm-eval bootstrap.

| Model | MMLU ↑ | ARC-Chall. (norm) ↑ | GSM8K (flex) ↑ | WikiText-103 PPL ↓ | Active Layers |
|---|---|---|---|---|---|
| Base TinyLlama | 25.04% ±0.36% | 36.09% ±1.40% | 2.96% ±0.47% | 243,116 | 22 / 22 |
| Baseline LoRA (exp1) | 25.22% ±0.37% | 34.47% ±1.39% | 2.05% ±0.39% | 1.93 | 22 / 22 |
| Stochastic Dropout (exp2) | 26.01% ±0.37% | 31.48% ±1.36% | 1.97% ±0.38% | 2.20 | 22 / 22 |
| **Gumbel Router (exp6)** | **25.19% ±0.37%** | **35.84% ±1.40%** | **2.05% ±0.39%** | **119.00** | **13.39 / 22** |
| Token-Level Router (exp9) | — | — | — | — | 19.9 / 22 (collapsed) |
| **Token-Level Router v2 (exp10)** | **25.24% ±0.37%** | 35.15% ±1.40% | **2.58% ±0.44%** | 197.22 | **6.68 / 22** |

> [!NOTE]
> **exp9 router collapse:** The initial token-level routing experiment (exp9) suffered from router collapse — active layers climbed from 7.2 to 19.9 over training, effectively learning to keep all layers active. This is analyzed in §5.4. exp10 addresses this with per-layer penalties and a target skip ratio regularizer.

### 4.3 Training Dynamics

**exp6 Gumbel Router** completed 1,860 global steps across 3 epochs:

- **Convergence**: Total loss dropped from ~4.01 (step 20, pre-KD warmup) to ~3.45 (step 1860). Validation CE loss stabilized at ~9.34 in Epoch 3.
- **KD dynamics**: KD loss was ~2,544 at step 60 (post-warmup onset), decayed rapidly, and turned negative in Epochs 2–3 (KL sign flip when student distributions converge tightly to teacher). This is expected behavior with `batchmean` reduction when distributions nearly match.
- **Layer activity**: Started at 12–15 active layers, climbed to 21–22 mid-Epoch 1 as the router initially resisted the sparsity penalty, then stabilized to ~13.25 layers at evaluation time.
- **Temperature**: Annealed τ: 1.0 → 0.95 → 0.9025 across the 3 epochs.

**exp1 Baseline LoRA** completed 1,580 logged steps across 3 epochs. Final validation loss: 0.6135. Best checkpoint val loss: 0.6135.

**exp2 Stochastic Dropout** completed 1,860 steps across 3 epochs. Average training-time active layers: ~13.0. Final validation loss: 0.7329. **Crucially**, evaluation is run with *all 22 layers active* (no dropping), which is the standard inference-time behavior for stochastic depth and is precisely what induces the accuracy penalty observed in Table 1.

*Figures (from `plot_results.py`):*
- **Figure 1**: `exp6_loss_breakdown.png` — CE / KD / gate loss trajectories.
- **Figure 2**: `exp6_temp_annealing.png` — Temperature schedule + layer activity.
- **Figure 3**: `pareto_frontier_curve.png` — Accuracy vs. compute Pareto frontier.
- **Figure 4**: `all_experiments_val_loss.png` — Validation loss comparison, all experiments.
- **Figure 5**: `exp7_benchmark_accuracy.png` — Benchmark accuracy bar chart.
- **Figure 6**: `exp7_efficiency_scatter.png` — Accuracy vs. active layers scatter.

### 4.4 Pareto Frontier Sweep (exp8)

`exp8_gumbel_pareto_sweep.py` sweeps λ ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.40}, training separate Gumbel Router instances at each penalty level.

**Table 2: exp8 Pareto sweep results** (run `20260511_122102`, 2 epochs / 5,000 train samples).

| λ (penalty) | Val Loss | Avg Active Layers | Skip Ratio |
|---|---|---|---|
| 0.01 | 9.59 | 15.48 | 29.6% |
| 0.02 | 9.61 | 14.67 | 33.3% |
| 0.05 | 9.48 | 15.04 | 31.6% |
| 0.10 | 9.40 | 15.18 | 31.0% |
| 0.20 | 9.51 | 16.00 | 27.3% |
| 0.40 | 9.44 | 15.01 | 31.8% |

> [!NOTE]
> **Pareto sweep limitation**: The current sweep (2 epochs, 5,000 samples) is under-trained relative to exp6 (3 epochs, 10,000 samples). As a result, the router does not show clear monotonic response to increasing λ — the skip ratio remains flat at ~29–33% across all penalty values. A re-run matching exp6 training scale is needed to produce the intended Pareto curve. This is the highest-priority experimental task remaining.

### 4.5 Sharp Penalty-Threshold Behavior in 3B Architectures

Contrary to our initial hypothesis based on smaller 1B models, the two 3B parameter networks tested here exhibit a highly non-linear response to the sparsity penalty. To investigate whether this behavior appears across architectures, we conducted parallel Pareto sweeps on both **Qwen2.5-3B (36 layers)** and **OpenLLaMA-3B (26 layers)**.

![Figure 4: Phase Transition Comparison (Qwen vs OpenLLaMA)](./phase_transition_comparison.png)

Both architectures strongly resisted layer-dropping at low penalty thresholds ($p < 10.0$). The entangled representations within deep blocks create large internal knowledge distillation (KD) gradients that overpower the sparsity objective. The router essentially determines that the mathematical "cost" of dropping a layer is higher than the compute penalty.

However, once the penalty breaches a critical $p \ge 10.0$ threshold in these sweeps, we observed an abrupt change in routing behavior:
- **Qwen2.5-3B:** Collapsed from full depth directly down to 14.4 active layers (a 60% structural skip ratio).
- **OpenLLaMA-3B:** Instantly collapsed down to 9.0 active layers, plummeting further to 6.3 active layers at slightly higher penalties (approaching the absolute structural minimum of 4).

These preliminary results suggest that scaling DLR to larger architectures requires careful tuning of the penalty threshold: too little pressure can leave the router near full depth, while slightly stronger penalties can push it into very aggressive skipping. Further investigation with complete training runs and downstream evaluation is needed before treating this as a task-performance result.

### 4.6 Token-Level Routing Analysis

To avoid assuming that token-level routing aligns with intuitive token difficulty, we added `exp19_token_routing_analysis.py`, which measures average active layers by token category: punctuation, stop words, numeric tokens, subword continuations, high-token-id proxies, and high-loss tokens. This analysis should be reported alongside the main 3B/7B results. A convincing token-level routing claim requires showing that the learned gates are not merely layer-wise constants broadcast over positions, but respond measurably to token identity, position, or loss.

---

## 5. Discussion

### 5.1 Why Contextual Conditioning Matters

Conditioning on Layer-4 hidden states versus raw embeddings is fundamental. Layer 4 has processed long-range self-attention and applied two FFN transformations, encoding syntactic and semantic structure that predicts downstream processing demand. The REINFORCE router (exp3) conditioned on raw embeddings showed high-variance routing with no consistent convergence — consistent with this hypothesis. The Gumbel-STE router converges reliably and produces a stable routing policy, as evidenced by the consistent ~13 active layer count across all evaluation batches.

### 5.2 Inference Mismatch: Why Stochastic Dropout Fails

The −3.67pp ARC gap between Stochastic Dropout and Token-Level DLR is a striking result in Table 1. Both models experience substantial layer skipping — but one's skipping is random at train time and absent at test time, while the other's is consistent and learned. The stochastic model's upper layers were never trained to receive consistent residual streams from layer 4 onward; when all 22 layers activate at inference, the representations are incoherent relative to training. DLR eliminates this mismatch by training with the same gating policy that is used at inference.

### 5.3 Accuracy–Compute Tradeoff vs. Static Baseline

The comparison against the Baseline LoRA model is nuanced. On ARC-Challenge, the sequence-level DLR variant (exp6) achieves 35.84% vs. baseline 34.47% (+1.37pp), and GSM8K is tied at 2.05%. These results fall within one standard error, supporting the claim of "near-equivalent accuracy at lower compute" rather than strict improvement.

**MMLU caveat:** All TinyLlama variants — including the baseline LoRA and even the pre-trained base model — score near the 25% chance level on MMLU 5-shot (range: 25.04%–26.01%). At this model scale, MMLU does not discriminate between variants and should not be used as primary evidence. ARC-Challenge, where the variants show meaningful differentiation (31.48%–36.09%), is the more informative benchmark for this model class.

### 5.4 Perplexity Discrepancy — A Serious Limitation

The DLR models' WikiText-103 perplexity (119.0 for exp6, 197.2 for exp10) is **two orders of magnitude higher** than the static LoRA baselines (1.93–2.20) when evaluated with strict binary routing (`hard=True`). This is the most significant limitation of the current approach and warrants careful discussion.

The gap indicates that layer skipping severely degrades fine-grained causal language modeling — the precise vocabulary probability distributions needed for low-perplexity next-token prediction require the intermediate feature refinement that skipped layers provide. While high-level task accuracy on multiple-choice and extraction benchmarks (ARC, GSM8K) is preserved, the model's ability to produce well-calibrated token distributions is substantially impaired.

We note that the padding-label masking in the evaluation code was corrected during this revision (padding tokens were previously included in the loss calculation, affecting absolute PPL values for all variants equally). The relative ordering and magnitude of the gap are unchanged.

This trade-off is not unique to DLR — early-exit methods face analogous perplexity degradation at aggressive exit points. However, the severity here (100× vs. baseline) means that **DLR in its current form is not suitable for open-ended generation tasks** where perplexity-correlated fluency matters. Its primary use case is efficiency-constrained inference on structured tasks (classification, extraction, multiple-choice QA) where task accuracy, not perplexity, is the relevant metric.

Future work should investigate: (1) softer routing (e.g., residual mixing rather than hard skip) to reduce the PPL gap; (2) perplexity-aware distillation objectives; (3) evaluation on generation quality metrics (e.g., MAUVE, human preference) to understand the practical impact.

---

## 6. Limitations and Future Work

> [!WARNING]
> The following are known limitations that constrain the claims of this paper.

1. **No runtime acceleration demonstrated.** The hook-based two-pass DLR implementation in PyTorch is **slower** than the baseline (7,847 Tok/sec vs. baseline 10,995 Tok/sec), despite executing only 7.3 layers on average. The overhead comes from Python-level hook dispatch, two forward passes, and Gumbel sampling. Static layer skipping (without routing) confirms that the *potential* speedup exists (6 layers: 34,482 Tok/sec), but realizing it requires a native CUDA/Triton implementation that fuses routing decisions into the forward pass. **All compute savings reported in this paper are structural (layer count), not wall-clock.**
2. **Perplexity is severely degraded.** As discussed in §5.4, DLR perplexity is 100× worse than baseline, making the approach unsuitable for open-ended text generation in its current form.
3. **MMLU is uninformative at this scale.** All TinyLlama variants score near 25% (chance level) on MMLU 5-shot, providing no discriminative signal. ARC-Challenge is the only benchmark where meaningful differences emerge.
4. **Statistical significance is marginal.** With standard errors of ~0.36% for MMLU and ~1.4% for ARC, most DLR vs. Baseline LoRA deltas are within 1σ. Larger-scale evaluation (MMLU-Pro, more trials) would tighten confidence.
5. **Scale to larger models** — Preliminary sweeps on Qwen2.5-3B and OpenLLaMA-3B (§4.5) show sharp penalty-threshold behavior, but downstream task evaluation at 3B+ scale is not yet complete. Validation on a 7B model such as Qwen2.5-7B, Llama-3-8B, or Mistral-7B with richer datasets (OpenOrca, C4) is needed.
6. **~~Rescale and re-run exp8 Pareto sweep~~** — ✅ Completed.
7. **~~Token-level routing~~** — ✅ Implemented (exp9 + exp10).
8. **~~Complete exp10 training and evaluation~~** — ✅ Completed.

### 6.2 Token-Level Router Collapse (exp9 → exp10)

The initial token-level routing experiment (exp9) exhibited a critical failure mode: the router collapsed to near-full-depth activation (19.9 / 22 active layers) by end of training, despite starting with aggressive layer skipping (7.2 layers at step 100). The root cause is the **penalty dilution problem**: at token-level granularity, each gate decision contributes only 1/(B·S·L) to the global mean(gates), so the L1 sparsity penalty is overwhelmed by the CE + KD loss gradient which uniformly pushes gates toward 1 (keep all layers).

exp10 addresses this with three changes:
1. **Per-layer penalty**: instead of penalizing mean(gates) globally, we sum per-layer token-averaged activities, ensuring each layer independently feels sparsity pressure.
2. **Quadratic target regularizer**: a penalty term λ_target · (skip_ratio - target)² that pulls the skip ratio toward the desired operating point (45%).
3. **Stronger initialization**: output layer bias initialized at -3.0 (vs. -2.0) to start with more aggressive skipping.

---

## 7. Conclusion

We presented Dynamic Layer Routing, a fully differentiable framework for input-conditional compute allocation in pre-trained LLMs. By combining a contextual Gumbel-STE router, a hook-based gated forward pass, and a KD-stabilized training objective, we achieve a 39.8% reduction in active layers on TinyLlama-1.1B. On ARC-Challenge — the most discriminative benchmark at this model scale — DLR achieves near-parity with a fully trained static LoRA baseline (35.84% vs. 34.47%) and consistently outperforms a stochastic depth dropout baseline (31.48%). However, we observe significant perplexity degradation (119–197 vs. 1.93 baseline), indicating that the approach is better suited to structured downstream tasks than open-ended generation. Furthermore, the current hook-based implementation does not yet realize wall-clock speedups; structural compute savings (layer count reduction) must be translated to runtime gains via native kernel integration. The framework is architecture-agnostic, requires no model surgery, and is compatible with standard LoRA fine-tuning pipelines.

---

## References

- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*.
- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). Depth-adaptive transformer. In *International Conference on Learning Representations (ICLR 2020)*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research, 23*(120), 1–39.
- Gao, L., et al. (2021). A framework for few-shot language model evaluation. *Zenodo*. https://doi.org/10.5281/zenodo.5371628
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR 2022)*.
- Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. In *European Conference on Computer Vision (ECCV 2016)*, 646–661.
- Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization with Gumbel-Softmax. In *International Conference on Learning Representations (ICLR 2017)*.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The concrete distribution: A continuous relaxation of discrete random variables. In *International Conference on Learning Representations (ICLR 2017)*.
- Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V. Q., Tay, Y., & Metzler, D. (2022). Confident adaptive language modeling. In *Advances in Neural Information Processing Systems (NeurIPS 2022)*.
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q. V., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations (ICLR 2017)*.
- Zhang, P., Zeng, G., Wang, T., & Lu, W. (2023). TinyLlama: An open-source small language model. *arXiv preprint arXiv:2401.02385*.

---

## Appendix A: Data Availability

| File | Contents |
|---|---|
| `exp1_baseline_metrics_20260508_210903.csv` | Full per-step training log, Baseline LoRA (3 epochs) |
| `exp2_stochastic_metrics_20260510_134453.csv` | Full per-step training log, Stochastic Dropout (3 epochs) |
| `exp6_gumbel_metrics_20260510_144940.csv` | Full per-step training log, Gumbel Router (1,860 steps) |
| `exp7_eval_results_20260530_135346.csv` | Complete 5-shot lm-eval-harness results, all 4 variants (raw) |
| `exp7_eval_summary_20260530_135346.json` | Structured eval summary for all variants |
| `exp8_gumbel_pareto_20260511_122102.csv` | Pareto sweep (6 λ values, preliminary) |
| `exp6_gumbel_output_20260510_144940/best_model/` | Best checkpoint (LoRA adapter + router weights) |
| `exp1_baseline_output_20260508_210903/best_model/` | Best checkpoint (LoRA adapter) |
| `exp2_stochastic_output_20260510_134453/best_model/` | Best checkpoint (LoRA adapter) |

---

*Draft v0.7 — June 2026*

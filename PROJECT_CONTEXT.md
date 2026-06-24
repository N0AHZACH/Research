# AI Agent Project Context
**Project:** Dynamic Layer Routing (DLR): Input-Conditional Compute Allocation in LLMs  
**Target Venue:** NeurIPS / ICML / ACL  

This document serves as the persistent memory and context for any AI agent interacting with this repository. It chronicles the entire history of the project, from initial 1.1B prototypes to the current state-of-the-art 7B/8B scaling efforts.

---

## 🎯 Overall Research Goal
The objective is to establish a mathematically rigorous and reproducible framework demonstrating that **Token-Level Dynamic Layer Routing (DLR)** achieves a superior Pareto frontier (Compute vs. Quality) compared to standard static models and input-agnostic Stochastic Depth. 

Instead of routing sequences, DLR uses a lightweight **Token-Level Gumbel-Softmax Router** (trained with a Straight-Through Estimator and stabilized by Knowledge Distillation) to dynamically drop transformer decoder layers for "easy" tokens while keeping them active for "hard" tokens.

---

## ⚙️ Methodology: What We Are Doing & How We Are Doing It

Our implementation uses a novel, non-destructive approach to dynamic compute that preserves HuggingFace model compatibility without altering the underlying C++ or CUDA source code.

### 1. The Hook-Based Two-Pass Architecture
Instead of copying and modifying the monolithic `Qwen2Model.forward` function, we use PyTorch forward hooks to inject routing on the fly:
*   **The "Always Keep" Phase:** The first $N$ layers (e.g., layers 0–3) are always executed. We use an `early_stop_hook` that captures the hidden state $H$ after these layers and raises a `StopForwardException` to abort the forward pass early.
*   **The Routing Phase:** The captured hidden states $H$ are passed to a lightweight, 3-layer MLP Router. The router outputs a binary gating tensor of shape `[Batch, SeqLen, RoutableLayers]`.
*   **The Gated Phase:** We install `gate_hooks` on the remaining routable layers. During the second full forward pass, these hooks intercept the output of each layer. If the gate for a token is `1`, the layer output is kept. If the gate is `0`, the layer is bypassed via a residual connection: $y = gate \cdot F(x) + (1 - gate) \cdot x$.

### 2. Router Mechanics & Differentiability
*   **Gumbel-Softmax STE:** Since binary routing gates `(0, 1)` are non-differentiable, we use the Gumbel-Softmax trick. During the forward pass, we use `hard=True` to snap probabilities to `0` or `1` (via `argmax`). During the backward pass, the Straight-Through Estimator (STE) allows gradients to flow back through the soft probabilities, updating the router weights.

### 3. Stabilizing the Router (The Loss Function)
A naive router will immediately collapse (either skipping everything or skipping nothing). We stabilize it with three distinct loss components:
*   **Knowledge Distillation (KD):** We use a `disable_adapter` context manager to run the frozen base model (acting as the Teacher) and compute the KL Divergence against the routed student's logits. This is crucial for maintaining model quality when layers are dropped.
*   **Depth-Scaled L1 Penalty:** We penalize the router for keeping layers active. To encourage deeper layers to be skipped more often than shallow layers, the L1 penalty linearly increases with depth.
*   **Target Skip Regularizer:** A quadratic penalty $( \text{actual\_skip} - \text{target\_skip} )^2$ that acts as a global attractor, forcing the network to maintain a specific overall compute budget (e.g., 40% skipped).

### 4. Hardware Optimization & Reproducibility (RTX PRO 6000 96GB)
To run these massive experiments on a single 96GB GPU, we use:
*   **Gradient Checkpointing:** Crucial for memory. We strictly enforce `gradient_checkpointing_enable(use_reentrant=False)` alongside `enable_input_require_grads()` for safe PEFT wrapping.
*   **Expanded LoRA:** Because we skip layers, the model must learn to compensate. We train LoRA adapters on the Attention projections (`q, k, v, o`) AND the MLP projections (`gate, up, down`). The MLP accounts for 62% of the compute budget and must be adaptable.
*   **Universal Determinism:** We enforce strict mathematical reproducibility. All runs are locked with `SEED=42` across Python `random`, `numpy`, and `torch`, with `cudnn.benchmark = False`. 

---

## 📜 Historical Timeline (What Has Been Done)

The project is structured as a progression of empirical studies, evolving from TinyLlama-1.1B prototypes to massive 8B experiments.

### Phase 1: TinyLlama-1.1B Baselines
We established the control bounds for the dynamic compute problem on a 1.1B parameter model.
*   **`exp1_baseline_finetune.py`**: Static LoRA fine-tuning. Acts as the upper-bound for accuracy (full depth, 22 layers active).
*   **`exp2_stochastic_finetune.py`**: Stochastic Depth Dropout (input-agnostic 50% random layer drop). Demonstrated the "inference mismatch" problem of static dropping.

### Phase 2: Sequence-Level Routing
We introduced the first trainable router, making decisions at the sequence level (one path per input sequence).
*   **`exp6_gumbel_router.py`**: Implemented a sequence-level Gumbel-STE router. Reduced active layers by ~40% while maintaining near-parity with the static baseline.
*   **`exp8_gumbel_pareto_sweep.py`**: Automated hyperparameter sweeps over the compute penalty (`lambda`) to generate our first Accuracy vs. Compute Pareto frontiers.

### Phase 3: Token-Level Routing (The Main Contribution)
We shifted the router to operate on individual tokens, allowing variable layer depth within the same sequence.
*   **`exp9_token_level_routing.py`**: Initial token-level routing prototype.
*   **`exp10_token_routing_v2.py`**: Stabilized the router. Added per-layer L1 penalties and quadratic target regularizers to prevent router collapse.
*   **`exp9_ablation_no_kd.py`**: Verified that Knowledge Distillation (KD) from a frozen teacher is strictly necessary for stable STE training.

### Phase 4: Scaling to Mid-Sized Models (3B)
*   **`exp11_large_model_routing.py`**: Scaled the token-level DLR to Qwen2.5-3B.
*   **`exp12_large_model_pareto.py`** / **`exp13_openllama_pareto.py`**: Generated Pareto sweeps for Qwen2.5-3B and OpenLLaMA-3B.
*   **`exp19_token_routing_analysis.py`**: Analyzed the learned routing behavior, proving that punctuation/stop-words receive less compute than rare tokens.

### Phase 5: Scaling to Massive Models (7B / 8B) & NeurIPS Parity Refactor
We scaled the experiments to Qwen2.5-7B and Llama3.1-8B. We recently executed a massive codebase refactor on the Qwen suite to ensure the science meets strict NeurIPS reproducibility standards:
*   **`exp24_qwen7b_stochastic.py` (Stochastic Depth Baseline)**
    *   Moved from Python RNG to `torch.bernoulli` to fix silent gradient checkpointing memory/state bugs.
    *   Implemented the correct linear skip schedule (Huang et al., 2016).
*   **`exp23_qwen7b_baseline.py` (Full-Depth Control)**
    *   Aligned perfectly with `exp24` (seeds, gradient checkpointing, targets, cosine LR schedule with 100 warmup steps).
*   **`exp25_qwen7b_token_routing.py` (Token-Level Routing)**
    *   Added missing `gradient_checkpointing_enable(use_reentrant=False)` to prevent massive VRAM OOMs.
    *   Fixed operator precedence tuple bugs in the gated hook.
    *   Synchronized CSV logging across all three scripts to track `empirical_flop_reduction_pct` and `tokens_per_sec` accurately.

**Status:** The Qwen2.5-7B triad (`exp23`, `exp24`, `exp25`) is now a mathematically rigorous, A/B/C testing framework ready for paper-grade execution.

---

## 🚀 Immediate Next Steps (What Is To Be Done)

1.  **Llama-3.1-8B Parity Review**
    *   The Llama-8B suite (`exp26_llama8b_baseline.py`, `exp27_llama8b_stochastic.py`, `exp28_llama8b_token_routing.py`) is lagging behind the Qwen suite.
    *   *Task:* Audit and patch the Llama-8B suite to ensure it has all the parity fixes (gradient checkpointing, deterministic seeds, full MLP LoRA targets, tuple hook fixes, and FLOP CSV logging).
2.  **Experiment Execution (Dry Runs & Full Runs)**
    *   Execute the Qwen 7B and Llama 8B training loops to generate the final artifacts and CSV files.
3.  **Evaluation Harness Verification**
    *   Run `exp22_qwen7b_eval_harness.py` and `exp29_llama8b_eval_harness.py` to test the newly generated checkpoints against MMLU, ARC, and GSM8K.

---

## 🔮 Future Plans & Publication Roadmap

1.  **Pareto Frontier Sweeps (`exp30`, `exp31`)**
    *   Execute hyperparameter sweeps over the `COMPUTE_PENALTY` to generate the true Accuracy vs. Compute curves for the 7B/8B models.
2.  **Router Entropy & Validation (`PUBLICATION_READY_PATCH_PLAN.md`)**
    *   Inject routing entropy tracking to explicitly prove to reviewers that token routing hasn't collapsed into static layer dropping.
    *   Add evaluation modes to save `(input_ids, gates)` heatmaps to visualize the dynamic depth routing per token.
3.  **Ablation Studies (`--no-kd`)**
    *   Run isolated baseline tests with Knowledge Distillation turned off to quantify how much KD stabilizes the Gumbel-STE.
4.  **Hardware Acceleration Benchmarking (`exp4`)**
    *   Measure true tokens-per-second latency. Currently, DLR reports structural theoretical compute savings; we need to validate how much translates to wall-clock time on the RTX 6000 Ada with FlashAttention-2.

---

## 📌 Rules for Agents Interacting with this Repo
- **DO NOT** make superficial formatting changes.
- **DO NOT** remove gradient checkpointing (`use_reentrant=False`) from PEFT models.
- **ALWAYS** ensure `numpy`, `random`, and `torch` seeds are set to 42.
- **ALWAYS** verify that any newly added script logs the exact same FLOP accounting and hardware metrics to allow for safe outer-joins in pandas later.
- If modifying an experiment, ensure the hyperparameter configurations (LR, Batch Size, Grad Accum) match the baselines.

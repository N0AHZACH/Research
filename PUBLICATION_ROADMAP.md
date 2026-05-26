# Dynamic Layer Routing: Publication Roadmap

This document outlines the critical next steps required to elevate the Dynamic Layer Routing (DLR) research from a strong proof-of-concept to a publication-worthy paper for top-tier machine learning conferences (e.g., NeurIPS, ICLR, ICML).

## Phase 1: Experimental Integrity (Immediate Priorities)

- [x] **Finalize Pareto Frontier Sweep (`exp8`)**
  - **Status:** ✅ Complete. Turbo sweep with λ ∈ [0.1, 3.0] produced monotonic response.
  - **Output:** `exp8_turbo_pareto_20260523_112315.csv` + `.png`

- [ ] **Resolve the Perplexity Discrepancy**
  - **Context:** Current perplexity is excessively high (54.21 vs baseline 1.93) because the evaluation uses soft gates instead of the hard binary gates used during training (domain shift).
  - **Action:** Re-run the `lm-evaluation-harness` with `hard=True` (STE mode) forced on the router.
  - **Goal:** Report a fair, competitive perplexity score. High perplexity will act as an immediate red flag for reviewers.

## Phase 2: Architectural Novelty (The "Wow" Factor)

- [x] **Implement Token-Level Routing (`exp9_token_level_routing.py`)**
  - **Status:** ✅ Training complete, but **router collapsed** to ~19.9 / 22 active layers.
  - **Root cause:** Penalty dilution — at token-level granularity, global mean(gates) is diluted by S×L gate decisions.

- [ ] **Fix Token-Level Router Collapse (`exp10_token_routing_v2.py`)** ← CURRENT PRIORITY
  - **Fixes applied:**
    1. Per-layer L1 penalty (sum of layer-averaged activities, not global mean)
    2. Quadratic target skip ratio regularizer (target = 45% skip)
    3. Much higher COMPUTE_PENALTY (10.0 vs 2.0)
    4. Reduced KD_ALPHA (0.3 vs 0.5)
    5. Stronger output bias initialization (-3.0 vs -2.0)
  - **Action:** Run `python exp10_token_routing_v2.py --fresh` and monitor layer activity.
  - **Goal:** Achieve stable ~12 active layers (~45% skip) with good val loss.

- [ ] **Evaluate exp10 with lm-eval harness**
  - **Action:** Run `python exp7_eval_harness.py` (now auto-detects exp10/exp9 checkpoints).
  - **Goal:** Fill in exp10 row in Table 1 of the manuscript.

## Phase 3: Hardware & Empirical Verification

- [ ] **Measure True Wall-Clock Latency (`exp4_inference_benchmark.py`)**
  - **Context:** Compute savings are currently proxied via "Average Active Layers". Reviewers are highly critical of theoretical FLOP reductions that ignore memory bandwidth bottlenecks.
  - **Action:** Finalize and run the inference benchmark to measure true Tokens Per Second (TPS).
  - **Goal:** Prove that skipping layers translates to actual hardware acceleration (e.g., bypassing KV-cache memory operations).

- [ ] **Tighten Statistical Significance in Evaluation**
  - **Context:** Accuracy deltas between DLR and the static baseline are within one standard error (~0.36%).
  - **Action:** Upgrade the evaluation pipeline from 0-shot to 5-shot, or utilize a more rigorous dataset like **MMLU-Pro**.
  - **Goal:** Shrink the error bars to statistically guarantee the claim that "accuracy degradation is negligible."

## Phase 4: Scaling (Guarding Against "Toy Setup" Critiques)

- [ ] **Validate on Larger Architectures**
  - **Context:** TinyLlama (1.1B) on Wikitext-103 is excellent for iteration but may be viewed as a toy setup by strict reviewers.
  - **Action:** Once token-level routing (`exp10`) is stable, run the full pipeline on a larger model (e.g., **Llama-3-8B** or **Mistral-7B**).
  - **Action:** Train on a richer dataset (e.g., OpenOrca or C4) rather than raw Wikitext.
  - **Goal:** Prove the DLR framework generalizes to production-scale models and complex instructional data.

## Phase 5: Manuscript Finalization

- [x] **Update Abstract & Introduction:** Reflect token-level routing. *(Done v0.4)*
- [x] **Update Methodology:** Add token-level variant, per-layer penalty, target skip ratio. *(Done v0.4)*
- [x] **Update Results Section:** Add exp9 collapse analysis and exp10 placeholder. *(Done v0.4)*
- [ ] **Fill in exp10 Results:** Once training + eval complete, update Table 1 with benchmark numbers.
- [ ] **Update Figures:** Regenerate plots with `python plot_results.py` after exp10 completes.
- [ ] **Strengthen "Related Work":** Ensure clear differentiation from traditional Early Exiting (emphasizing DLR's ability to skip middle layers and resume computation).

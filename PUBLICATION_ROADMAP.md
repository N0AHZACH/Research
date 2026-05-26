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

- [x] **Fix Token-Level Router Collapse (`exp10_token_routing_v2.py`)**
  - **Status:** ✅ Complete. Achieved ~6.7 active layers with strong evaluation scores.

- [x] **Evaluate exp10 with lm-eval harness**
  - **Status:** ✅ Complete. Table 1 in manuscript updated.

## Phase 3: Hardware & Empirical Verification

- [x] **Measure True Wall-Clock Latency (`exp4_inference_benchmark.py`)**
  - **Status:** ✅ Complete. Results integrated into manuscript Section 6. Python hook-based overhead measured, simulating massive gains for native CUDA implementation.

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
- [x] **Fill in exp10 Results:** ✅ Done in v0.5.
- [x] **Update Figures:** ✅ Done.
- [x] **Strengthen "Related Work":** Ensure clear differentiation from traditional Early Exiting (emphasizing DLR's ability to skip middle layers and resume computation).

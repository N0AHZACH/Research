# Dynamic Layer Routing: Publication Roadmap

This document outlines the critical next steps required to elevate the Dynamic Layer Routing (DLR) research from a strong proof-of-concept to a publication-worthy paper for top-tier machine learning conferences (e.g., NeurIPS, ICLR, ICML).

## Phase 1: Experimental Integrity (Immediate Priorities)

- [ ] **Finalize Pareto Frontier Sweep (`exp8`)**
  - **Context:** The current sweep lacks a monotonic response due to under-training (2 epochs / 5,000 samples).
  - **Action:** Allow the currently running `exp8_turbo_pareto_sweep.py` to complete. 
  - **Goal:** Generate a clean `pareto_frontier_curve.png` that mathematically proves DLR scales efficiency gracefully with the compute penalty $\lambda$. This is the central empirical claim of the paper.

- [ ] **Resolve the Perplexity Discrepancy**
  - **Context:** Current perplexity is excessively high (54.21 vs baseline 1.93) because the evaluation uses soft gates instead of the hard binary gates used during training (domain shift).
  - **Action:** Re-run the `lm-evaluation-harness` with `hard=True` (STE mode) forced on the router.
  - **Goal:** Report a fair, competitive perplexity score. High perplexity will act as an immediate red flag for reviewers.

## Phase 2: Architectural Novelty (The "Wow" Factor)

- [ ] **Implement Token-Level Routing (`exp9_token_level_routing.py`)**
  - **Context:** The current router makes sequence-level (batch-level) skip decisions. State-of-the-art dynamic compute expects token-level granularity (e.g., skipping layers for the word "the" but executing all layers for complex reasoning).
  - **Action:** Modify the router to evaluate the hidden state per token $h_{t,l}$ rather than the pooled sequence representation $\bar{h}$.
  - **Goal:** Maximize the theoretical efficiency of the model and substantially increase the architectural novelty of the paper.

## Phase 3: Hardware & Empirical Verification

- [ ] **Measure True Wall-Clock Latency (`exp4_inference_benchmark.py`)**
  - **Context:** Compute savings are currently proxied via "Average Active Layers". Reviewers are highly critical of theoretical FLOP reductions that ignore memory bandwidth bottlenecks.
  - **Action:** Finalize and run the inference benchmark to measure true Tokens Per Second (TPS).
  - **Goal:** Prove that skipping layers translates to actual hardware acceleration (e.g., bypassing KV-cache memory operations).

- [ ] **Tighten Statistical Significance in Evaluation**
  - **Context:** Accuracy deltas between DLR and the static baseline are within one standard error ($\sim 0.36\%$).
  - **Action:** Upgrade the evaluation pipeline from 0-shot to 5-shot, or utilize a more rigorous dataset like **MMLU-Pro**.
  - **Goal:** Shrink the error bars to statistically guarantee the claim that "accuracy degradation is negligible."

## Phase 4: Scaling (Guarding Against "Toy Setup" Critiques)

- [ ] **Validate on Larger Architectures**
  - **Context:** TinyLlama (1.1B) on Wikitext-103 is excellent for iteration but may be viewed as a toy setup by strict reviewers.
  - **Action:** Once token-level routing (`exp9`) is stable, run the full pipeline on a larger model (e.g., **Llama-3-8B** or **Mistral-7B**).
  - **Action:** Train on a richer dataset (e.g., OpenOrca or C4) rather than raw Wikitext.
  - **Goal:** Prove the DLR framework generalizes to production-scale models and complex instructional data.

## Phase 5: Manuscript Finalization

- [ ] **Update Abstract & Introduction:** Reflect token-level routing and true wall-clock speedups.
- [ ] **Update Results Section:** Insert the new monotonic Pareto curve, the corrected perplexity scores, and the MMLU-Pro / 5-shot significance tables.
- [ ] **Strengthen "Related Work":** Ensure clear differentiation from traditional Early Exiting (emphasizing DLR's ability to skip middle layers and resume computation).

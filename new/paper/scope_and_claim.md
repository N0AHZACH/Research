# Publishable Scope v1 — Highest-Odds Path (2026-09-11)

No sure-fire exists. This is the smallest scope that is still a real contribution and defensible on solo-undergrad compute.

## Scoped title
A Controlled Comparison of Learned vs. Random Layer Skipping in LoRA-Adapted LLMs, with an Independent Audit of FlexiDepth

## Why this scope
- "New routing architecture" is crowded (FlexiDepth COLM 2025 does per-layer router+adapter on frozen Llama-3-8B, 8/32 layers skipped, no wall-clock speedup — same wall you hit).
- "Rigorous audit: does learned routing beat matched random dropping?" is uncrowded and explicitly invited by TMLR (reproducibility studies).
- Turns biggest risk ("looks like FlexiDepth") into asset: you audit FlexiDepth's public checkpoint through your harness.

## What to run (only this for v1)
1. TinyLlama-1.1B full block × 3 seeds: dense / stochastic_30 / dlr_30 / stochastic_50 / dlr_50
2. One 3B family (Qwen2.5-3B), same block × 3 seeds (rented GPU, ~$150-400 total)
3. Inference-only audit: xuan-luo/FlexiDepth-Llama-3-8B-Instruct through `new/experiments/evaluate.py` + stochastic control at same measured skip, loglikelihood tasks only
- Tasks for v1: ARC-Challenge, HellaSwag, Winogrande, MMLU (loglikelihood, 1 forward pass) + WikiText ppl. GSM8K generate_until EXCLUDED for routed rows, listed as limitation (KV-cache + routing interaction unsolved).

## Non-negotiable fixes before any claim
- Stochastic Depth gets SAME KD + regularizer as DLR, else "DLR beats SD" is confounded.
- Upfront single-shot router (decide at layer N, apply to rest) vs per-layer sequential routing must be an explicit design choice with rationale + ablation paragraph, not default. FlexiDepth/MoD do per-layer.
- Report: mean ± SD + 95% CI across 3 seeds, per-seed rows, expected vs measured active layers, per-layer activity, structural FLOP reduction only (no wall-clock claim), router entropy + collapse_flag, tokens/sec where measured.
- One evaluator only. Every table row → `verified` ledger entry.

## Venue plan
- TMLR rolling submission, target Feb-Mar 2027 (bar = claims supported by evidence + audience interest, not novelty/impact — verified at jmlr.org/tmlr).
- In parallel: NeurIPS 2027 efficient-ML workshop short paper once 1B block done (citable + feedback). NeurIPS 2026 deadline already passed; 2027 main-track ~May 2027 is earliest shot.
- Paper skeleton: Intro (FLOP-uniformity question) / Related Work (5 papers, see related_work.md) / Method (two-pass hook + Gumbel-STE, flagged upfront vs sequential) / Evaluation Protocol (your strongest section) / Results (1B+3B + FlexiDepth audit) / Limitations / Conclusion.

## Definition of done
- [ ] Related work names FlexiDepth, MoD, LayerSkip, AdaSkip with 1-paragraph diff each
- [ ] SD+KD parity implemented
- [ ] 3 seeds + CIs
- [ ] Limitations section upfront (no wall-clock, no GSM8K-routed, upfront routing)
- [ ] FlexiDepth checkpoint run through same evaluate.py
- [ ] All rows traceable to ledger

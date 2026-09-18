# Related Work — Verified Citations (do not submit without these)

## 1. FlexiDepth — closest competitor, must cite and differentiate
Luo, Wang & Yan. Adaptive Layer-skipping in Pre-trained LLMs. COLM 2025. arXiv:2503.23798.
- Per-layer plug-in router + adapter, backbone frozen, only router/adapter trained. Latter 16 of 32 Llama-3-8B-Instruct layers converted; reports skipping 8/32 at full benchmark performance; token-type-dependent compute (copy ~20-22 layers, math/reasoning ~24-30).
- Explicitly states: "does not yet achieve wall-clock speedup due to varied skipping patterns and I/O overhead." Open: model xuan-luo/FlexiDepth-Llama-3-8B-Instruct, dataset xuan-luo/FlexiPatterns-Llama-3-8B-Instruct.
- Difference from ours: per-layer sequential routing with adapter on skip path vs our upfront single-shot router (decide once at layer N, apply to rest, residual bypass y=gate*F(x)+(1-gate)*x). Audit angle: run their checkpoint through our harness vs matched random dropping.

## 2. Mixture-of-Depths (MoD)
Raposo et al. 2024. Training-time dynamic compute, requires training from scratch — contrast with post-hoc frozen-backbone setting (both FlexiDepth and ours).

## 3. LayerSkip / early-exit family
Elhoushi et al. 2024 and SkipDecode / Early-Exit / Unified Skipping (Liu, Meng & Zhou 2024, cited as baselines in AdaSkip). Fixed schedule skipping — contrast with input-conditional routing.

## 4. AdaSkip — real, but different problem
He et al. Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference. arXiv:2501.02336, AAAI 2025. Similarity-based adaptive sublayer skipping for long-context prefill+decode. Cite as adjacent long-context work, not direct competitor. Code: github.com/ASISys/AdaSkip.

## 5. LoRA-Drop — CAUTION: Claude's description was wrong
Two distinct papers share the name; neither is "2026 paper benchmarking vs FlexiDepth on ARC/HellaSwag":
- Zhou et al. LoRA-drop: Efficient LoRA Parameter Pruning (COLING 2025) — prunes LoRA params by output importance, not layer skipping.
- Rajabzadeh et al. LoRA-Drop: Temporal LoRA Decoding (arXiv:2601.02569, Jan 2026) — temporal schedule reusing previous-token hidden state + LoRA correction on fixed intermediate layers, KV-cache compatible, up to 2.6× decode + 45-55% KV cut on LLaMA2-7B/LLaMA3-8B/Qwen2.5-7B/14B, evals GSM8K/MATH/BBH/HumanEval/MBPP/LongBench. No router. Do NOT cite as "benchmarks itself against FlexiDepth on ARC/HellaSwag" — unverified.
- MindSkip (2024) claimed by Claude: no arXiv match found 2026-09-11. Do NOT cite until verified — treat as hallucination risk.

## What to write (1 para each)
For each: what they do, what they keep frozen/trainable, routing granularity (per-layer vs upfront vs fixed), eval tasks, and one sentence on how your controlled learned-vs-random comparison + standardized harness differs.

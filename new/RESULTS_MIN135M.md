# Min prototype results — SmolLM2-135M, 150 steps, seed 42, RTX 4060 Laptop (bf16)

Date: 2026-09-14/15. Code: `new/src/dlr/`, `new/experiments/train_min.py`.
Standard harness: `new/experiments/evaluate.py` + `new/configs/eval/min135m.json`
(0-shot, limit=100, ppl on wikitext-103-raw-v1 validation 50x128 — exploratory
grade, NOT paper settings). Train data: WikiText-103-raw-v1 train, first 300
lines len>100, packed 128 (360 train / 50 held-out). LoRA r=8/a=16, lr 2e-4
cosine, batch 4. Shared routed loss: `lm + 0.2*KD(T=2) + 400*(skip-0.3)^2 +
0.01*L1`. Router: bias init logit(0.7), deterministic threshold + STE in train
AND eval (see bug #4).

## Standard harness table (`new/artifacts/evals/min135m-001/`)

| variant | ppl (harness) | ARC-c acc_norm | HellaSwag acc_norm | Winogrande acc | skip | entropy | collapse | tps |
|---|---|---|---|---|---|---|---|---|
| base | 24.32 | 0.25 | 0.44 | 0.49 | — | — | — | 9141 |
| dense | 20.09 | 0.26 | 0.45 | 0.46 | — | — | — | 8392 |
| stochastic_30 | 347.53 | 0.26 | 0.37 | 0.50 | 0.31 | ~0 | n/a (fixed mask) | 10418 |
| token_dlr_30 | 62.87 | 0.26 | 0.44 | 0.52 | 0.18 | 0.43 | 0 | 6231 |

Task scores at 135M/0-shot/limit-100 are noise (±0.04–0.05): ARC at chance for
everyone, Winogrande ~0.5 for everyone. Only HellaSwag separates stochastic
(0.37) from the rest (0.44–0.45). PPL is the working signal at this scale.

## Plain answers
1. Loop works end-to-end? Yes. 150 steps, no shape/OOM/NaN; loss 136 -> ~30.
2. Router avoided collapse? Yes. mean_gate 0.82, entropy 0.43, per-layer keep
   0.43–1.0, varies per token.
3. token_dlr vs stochastic? ppl 62.9 vs 347.5; HellaSwag 0.44 vs 0.37; skip
   0.18 vs 0.31 (both within 10pp of 0.3).

## Caveats (do not overclaim)
- 150 steps is plumbing scale. Stochastic's LoRA must be robust to ANY random
  subset and clearly hasn't converged. The gap direction is encouraging, not
  conclusive. Task rows are 0-shot/limit-100: exploratory only.
- stochastic `collapse=n/a`: fixed coin-flip masks have ~0 entropy by design.
  The collapse rule applies to learned routers only.
- Skip landed at 0.18 vs target 0.30 (within 10pp, but low side). Needed
  iterations: skip_weight 2 -> 100 -> 200 -> 400, kd 0.5 -> 0.2, tau 1 -> 2,
  router bias init logit(0.7), then deterministic-STE train (bug #4). v1–v4 all
  collapsed to all-keep under Gumbel sampling. Budget calibration belongs in
  the Pareto step (held-out split, never eval set), not hand-tuned here.
- Structural FLOP reduction only (dlr 15.7%, stochastic 26.7%). Harness tps is
  measured: dlr two-pass forward is SLOWER per token (6231 vs 8392 dense) —
  no speedup claim, matching the field's honest reporting.
- Fixed bugs this run:
  1. torch was CPU-only -> installed torch+cu124.
  2. `binary_entropy` NaN on exact 0/1 gates (fp32 `1-(1e-8)==1`; float64 fix).
  3. train_min saved no adapter -> harness couldn't reload (now saves adapter
     + tokenizer; runs reproduce bit-exact: dense ppl 35.111 twice).
  4. Train/eval gate mismatch: Gumbel-sampled train skip 0.215 vs threshold
     eval skip 0.031 on identical data -> train uses the harness's discrete
     rule now (threshold + STE).
  5. `import src.dlr` collides with another project's `src` package on
     sys.path -> harness imports via `new/src` as plain `dlr`.
  6. evaluate.py silently ran stochastic rows at FULL depth -> fixed seeded
     mask now (refuses to run without always_keep_layers); token_dlr lm-eval
     via RoutedHFLM (loglikelihood only, generate_until refused explicitly).

## Next
1. TinyLlama-1.1B port (same code; verify `model.model.layers` path).
2. Pareto calibration harness for compute_penalty on a held-out split.
3. Real evaluator rows (ARC/HellaSwag/Winogrande/MMLU) only after 1+2.

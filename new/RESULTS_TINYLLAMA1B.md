# TinyLlama-1.1B kill-test results — 150 steps, seed 42, RTX 4060 Laptop 8GB (bf16)

Date: 2026-09-15. Code: `new/src/dlr/`, `new/experiments/train.py` + `evaluate.py`.
Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, always_keep=3, routable=19).
Train data: WikiText-103-raw-v1 train, first 300 lines len>100, packed 128
(405 train / 50 held-out). LoRA r=8/a=16 (expanded targets for DLR).
Shared routed loss: `lm + 0.2*KD(T=2) + skip_weight*(skip-target)^2 + 0.01*L1`
(skip_weight=400 unless noted). Router: bias init logit(0.7), deterministic
threshold + STE in train AND eval. Eval grade: EXPLORATORY
(`new/configs/eval/tinyllama1b-001.json`: 0-shot, limit=100, ppl on
wikitext-103-raw-v1 validation 50x128 — NOT paper settings).

## Harness table (`new/artifacts/evals/tinyllama1b-001/`)

| variant | ppl (harness) | ARC-c acc_norm | HellaSwag acc_norm | Winogrande acc | skip (harness) | entropy | collapse | tps |
|---|---|---|---|---|---|---|---|---|
| base | 13.38 | 0.33 | 0.64 | 0.58 | — | — | — | 2649 |
| dense | 10.11 | 0.29 | 0.62 | 0.60 | — | — | — | 4642 |
| stochastic_30 | 96.89 | 0.23 | 0.35 | 0.58 | 0.37 | ~0 | n/a (fixed mask) | 4676 |
| token_dlr_30 | 17.83 | 0.32 | 0.59 | 0.54 | 0.15 | 0.37 | 0 | 3628 |
| stochastic_50 | 634.80 | 0.19 | 0.26 | 0.58 | 0.58 | ~0 | n/a (fixed mask) | 5057 |
| token_dlr_50 | 24.35 | 0.36 | 0.56 | 0.49 | 0.25 | 0.46 | 0 | 3442 |

Train-tail (same-pack held-out) ppl for reference: dense 13.88 / stoch30 57.6 /
dlr30 26.1 / stoch50 250.2 / dlr50 37.1 — same ordering, lower absolute values.

## Matched-compute audit (`new/artifacts/evals/tinyllama1b-002-matched/`)

Same dense LoRA weights, only the gate differs — isolates routing from training:

| variant | gate | skip (measured) | ppl | ARC-c | HellaSwag |
|---|---|---|---|---|---|
| random_15 | coin-flip mask, p=0.15 | 0.16 | 59.02 | 0.21 | 0.37 |
| token_dlr_30 | learned router | 0.15 | 17.83 | 0.32 | 0.59 |
| random_25 | coin-flip mask, p=0.25 | 0.21 | 112.55 | 0.21 | 0.38 |
| token_dlr_50 | learned router | 0.25 | 24.35 | 0.36 | 0.56 |

Learned >> random at the same measured skip: 3.3x ppl at ~0.15, 4.6x at ~0.25,
+0.11/+0.15 ARC-c, +0.22/+0.18 HellaSwag. Winogrande is noise (~0.5 for all).

## Calibration (`new/experiments/calibrate_skip.py`, `new/artifacts/analysis/`)

- 20-step `--smoke` does NOT calibrate: all skip_weights (400/800/1600) sit at
  skip ~0.005, collapse=1. The router needs >20 steps to move. Script now
  defaults to full-length (`--max-steps 150`, ~55s each on 4060) and warns.
- Full-run: `token_dlr_30` sw400 -> train-tail skip 0.214 / harness skip 0.151;
  sw800 -> train-tail 0.275 (in 25-35% band) / harness 0.188, ppl 20.37
  (`new/artifacts/evals/tinyllama1b-003-sw800-ppl/`, ppl-only).
  Lesson: calibrate against a HELD-OUT split through the harness, never the
  train-tail number — they differ by ~9pp. Pareto step still required.

## Plain answers

1. DLR beats stochastic at both levels? Yes, by large margins on ppl/ARC/Hella
   (17.8 vs 96.9; 24.3 vs 634.8), and beats matched random masks 3-4x at equal
   skip. Single seed, exploratory eval — direction, not proof.
2. Router healthy? Yes. collapse=0 both levels, entropy 0.37/0.46, per-layer
   keep 0.62-0.99 / 0.41-0.99, std_active 4.7/5.4 (token-varying, vs 0.0 fixed).
3. Budget hit? No. Harness skips 0.15/0.25 miss bands 25-35%/45-55%.
   sw800 reaches 0.19 on harness — closer, still low. Do not claim
   "matched-compute" for the 001 trained-stochastic comparison (stochastic
   measured 0.37/0.58); the 002 random-mask comparison IS matched.

## Caveats (do not overclaim)

- 150 steps / 1 seed / 0-shot/limit-100 / 50x128 ppl = plumbing+signal scale.
  No `verified` ledger rows yet. Needs 3 seeds + paper eval (5-shot MMLU/ARC,
  GSM8K excluded for routed rows, 500x512 ppl) per `NEURIPS_EVALUATION_STANDARD.md`.
- Structural FLOP reduction only (dlr30 13%, dlr50 21%). Harness tps is
  measured: DLR two-pass is SLOWER per token (3628/3442 vs 4642 dense).
- FlexiDepth-8B audit BLOCKED on this machine: 8B bf16 ~16GB > 8GB VRAM.
  `bitsandbytes 0.49.2` is installed, so a 4-bit eval path (or 24GB GPU) unblocks
  it. The 002 random-mask audit substitutes for the "learned vs random" question.

## Next

1. Held-out Pareto calibration: sweep skip_weight at full length, target harness
   skips 0.30/0.50, then re-run 002-style matched random controls at the new skips.
2. Seeds 43/44 for the 5-variant block + paper-grade eval manifest.
3. 4-bit eval path (or bigger GPU) for FlexiDepth-Llama-3-8B-Instruct audit.
4. Second 1B family only after 1-2 pass S6 validity audit.

---

## Update 2026-09-15 (evening) — 3-seed stats, sw1600 calibration, FlexiDepth first run, G4 PASS

Items 2 and 3 above are now DONE (seeds 43/44 trained+evaluated at sw1600:
`tinyllama1b-006-seed43`, `tinyllama1b-007-seed44`; FlexiDepth ckpt runs through
harness: `flexidepth-001`). Item 1 partially (sw1600 closest). Item 4 re-scoped
as gate G3/G4 in `docs/VALIDATION_GATE.md`.

### 3-seed table, 150 steps (evals 001/006/007 — exploratory grade, see caveats)

| variant | ppl mean±SD | Hella acc_norm | ARC-c acc_norm | harness skip |
|---|---|---|---|---|
| dense | 10.12±0.04 | 0.613±0.009 | 0.293±0.005 | — |
| stochastic_30 | 104.51±7.78 | 0.337±0.009 | 0.247±0.012 | 0.37 (fixed) |
| token_dlr_30 | 20.56±1.93 | 0.547±0.031 | 0.320±0.008 | 0.189 (band 0.25–0.35 MISSED) |
| stochastic_50 | 681.53±60.91 | 0.273±0.012 | 0.207±0.012 | 0.58 (fixed) |
| token_dlr_50 | 33.80±6.87 | 0.513±0.034 | 0.323±0.045 | 0.308 (band 0.45–0.55 MISSED) |
| random_20 (n=2) | 59.01±0.59 | — | — | 0.16 |
| random_35 (n=2) | 736.11±0.35 | — | — | 0.42 |

No overlap between DLR and stochastic/random ppl distributions at either level.
Hella sign consistent all 3 seeds (+0.21/+0.24). ARC consistent for dlr30;
dlr50 noisiest cell (seed43 0.26).

### G4 longer-train gate, 500 steps, seed 42 (`tinyllama1b-010-g4-500`) — PASS

Stochastic DID improve with training (threat was real: stoch30 96.9→69.4,
stoch50 634.8→201.4) but DLR still leads by multiples on every metric:

| variant | ppl | ARC-c | Hella | Wino | skip | ent | collapse |
|---|---|---|---|---|---|---|---|
| dense | 10.67 | 0.29 | 0.59 | 0.54 | — | — | — |
| stochastic_30 | 69.44 | 0.24 | 0.36 | 0.58 | 0.368 | ~0 | n/a |
| token_dlr_30 | 15.22 | 0.30 | 0.60 | 0.55 | 0.190 | 0.39 | 0 |
| stochastic_50 | 201.40 | 0.20 | 0.31 | 0.54 | 0.579 | ~0 | n/a |
| token_dlr_50 | 22.30 | 0.32 | 0.55 | 0.51 | 0.368 | 0.45 | 0 |
| random_20 | 85.77 | 0.19 | 0.34 | 0.53 | 0.158 | ~0 | n/a |
| random_35 | 1249.57 | 0.20 | 0.31 | 0.56 | 0.421 | ~0 | n/a |

Ratios: dlr30 4.6× stoch30, 5.6× random_20; dlr50 9.0× stoch50, 56× random_35.
Train-tail skips at 500 steps: dlr30 0.222 (band!), dlr50 0.447 (band edge) —
harness still reads lower (0.190/0.368). Gate decision recorded in
`docs/VALIDATION_GATE.md` §7: G4 PASS → approved: G1 calibration lock, G5
FlexiDepth control, G3 second family, then paid G6.

### Calibration + audit footnotes

- sw1600 @150 steps: harness skip 0.21 (dlr30) / 0.34–0.35 (dlr50); ppl cost vs
  sw400 is modest (17.8→21.3 @30). Pareto evals 004/005 on disk.
- `flexidepth-001`: FlexiDepth-Llama-3-8B-Instruct through harness — ppl 16.19,
  ARC-c 0.52, Hella 0.64, Wino 0.73. 4-bit path verified (`008-4bit-smoke`).
  Missing: matched random control at its measured skip (G5).

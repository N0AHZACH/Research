# Paper Data Checklist — every table/figure → artifact, gaps, fixes

**Last updated:** 2026-09-15. Purpose: single source of truth for "do we have all data needed to write the paper."
Rule (from README): a result supports the paper only if produced through `new/experiments/evaluate.py` or explicitly marked exploratory. Nothing below is `verified` yet — all ledger rows are `completed_unverified`.

## Paper tables → data status

| Paper table | Needs | Have | Missing / action |
|---|---|---|---|
| Table 1 — Main 1B block (dense / stoch30 / dlr30 / stoch50 / dlr50 × 3 seeds, ppl + tasks + skip + entropy) | 3-seed mean±SD+CI, paper-grade eval | 3-seed EXPLORATORY (0-shot/lim100/50×128): ppl dlr30 20.6±1.9 vs stoch30 104.5±7.8; dlr50 33.8±6.9 vs stoch50 681±61; Hella +0.21/+0.24; evals 001/006/007 | G6: paper-grade manifest (5-shot ARC/MMLU, 10-shot Hella, 5-shot Wino, 500×512 ppl, full limit) on rented GPU for 3B + local 1B rerun |
| Table 2 — Learned vs matched random at equal measured skip | random-mask controls at DLR skips | 002 (s42, 3.3×/4.6× ppl), 006/007 random_20/35 (2 seeds, 2–20× gaps) | Re-run at G1-calibrated skips once bands hit; 1 more seed for random_35 (n=2 now) |
| Table 3 — FlexiDepth 8B independent audit (ckpt + matched random + base ref) | flexi ckpt + random control at same skip | Flexi ckpt numbers (ppl 16.19 / ARC 0.52 / Hella 0.64 / Wino 0.73, `flexidepth-001`) | G5 DONE 2026-09-15: `flexidepth-002-control` — random ppl 237.74 (14.7× worse), ARC 0.27, Hella 0.37, Wino 0.62. Base Llama-3-8B-Instruct ref row still optional |
| Table 4 — Ablations (no-KD, token-vs-seq, always-keep) | DLR-noKD + matched rows | NOTHING — highest-priority missing science (does KD do the work?) | G7: train dlr30-noKD × 3 seeds (cheap, local) |
| Calibration/Pareto figure | skip_weight sweep → harness skip + ppl | sw400/800/1600 train-tail + harness points (evals 003/004/005); ~9pp train-vs-harness gap documented | Lock G1: pick sw per target from HARNESS skip; record decision |
| Routing figures (per-layer heatmap, entropy, token-type depth) | per-layer activity + token analysis | Per-layer arrays for every DLR row in routing_metrics.csv (plottable NOW); entropy/collapse all rows | `analyze_routing.py` token-type outputs missing (punct/stop/rare/content) |
| Compute/latency | structural FLOP% + measured tps + hw spec | FLOP% + tps in every eval (perplexity.json + routing_metrics.csv); hw = 4060 Laptop 8GB, driver 581.29, torch 2.6.0+cu124 | No `latency.json` in any eval (FIX evaluate.py); tps only covers ppl pass, not lm-eval tasks |

## Provenance blockers (fix before ANY `verified` row)

1. **`new/` is not in git.** `git status -- new/` = entirely `??`; HEAD commit `93ab83f` predates it. Every manifest records that hash + dirty tree. ACTION: `git add new/` + clean commit, then re-stamp (or pin) run manifests. No paper claim without a clean-hash row.
2. **Empty columns in eval_summary.csv.** `seed`, `checkpoint_id`, `shot_count`, `eval_harness_version` are EMPTY in all 16-col runs (001/002/006/007/flexi/min). Recoverable from manifests, but FIX `evaluate.py` to populate before G6 or every paper table needs hand-joins.
3. **Eval grade too low for paper.** All task rows: 0-shot, limit=100 (±0.04–0.05 noise, ARC near chance). PPL: 50×128. Paper needs the G6 manifest above. Do NOT upgrade any row to `verified` at current grade.
4. **Train/eval skip gap.** Train-tail skip overstates harness skip by ~9pp. Calibration decisions must cite harness numbers only (already policy; enforce in G1 lock).

## Second-family status (G3)

- Weak pass: SmolLM2-135M, same direction (DLR ppl 62.9 vs stoch 347.5, Hella 0.44 vs 0.37, `min135m-001`).
- Full pass needs: second 1B+ family at protocol (3 seeds × 5 variants). Local candidates that fit 8GB: SmolLM2-360M/1.7B, Qwen2.5-0.5B/1.5B, Llama-3.2-1B. Decision: pick ONE after G4 passes; 3B+ goes to rented GPU.

## G4 inputs/outputs (running 2026-09-15, seed 42, 500 steps)

- Runs: `tinyllama1b_{dense,stochastic_30,token_dlr_30_sw1600,stochastic_50,token_dlr_50_sw1600}_500_seed42` (dense done: train-tail ppl 14.62, 99s).
- Eval manifest to build after training: `new/configs/eval/tinyllama1b-010-g4-500.json` (same tasks as 001 + random-mask controls at measured DLR skips).
- PASS bar: DLR ppl ≥2× better than stochastic AND matched-random, collapse=0 → record GO in VALIDATION_GATE §7. FAIL bar: gap <1.3× or invert → KILL per gate §3.

**OUTCOME 2026-09-15 (evening): G4 PASS — all 5 runs + eval `tinyllama1b-010-g4-500` on disk, ledger updated.**
Train-tail: dense 14.62 / stoch30 47.42 / dlr30 24.19 (skip 0.222) / stoch50 158.17 / dlr50 34.14 (skip 0.447).
Harness: dlr30 ppl 15.22 (4.6× stoch30, 5.6× random_20), dlr50 ppl 22.30 (9.0× stoch50, 56× random_35); Hella +0.24 both; collapse=0.
All rows `completed_unverified` (exploratory grade). Decision: GO for G1+G5+G3, then paid G6.

## Data that is ready to plot/write NOW (no more GPU needed)

- 3-seed exploratory main table (values in VALIDATION_GATE §4a + stats below).
- Matched-random comparison (002 + 006/007 random rows).
- Per-layer activity heatmaps (routing_metrics.csv arrays, all DLR rows).
- Calibration curve (sw400→800→1600, train-tail vs harness).
- FlexiDepth first-run numbers (needs control before publishing).
- Methods section: two-pass hook + deterministic-threshold STE, 6 fixed bugs, parity (same KD/reg), upfront-vs-sequential rationale placeholder.
- Limitations section: no wall-clock speedup (measured), no GSM8K-routed, upfront routing, 150-step scale, skip-band miss.

## 3-seed numbers (copy-paste for drafts — EXPLORATORY, not verified)

- PPL: dense 10.12±0.04; stoch30 104.51±7.78; **dlr30 20.56±1.93**; stoch50 681.53±60.91; **dlr50 33.80±6.87**; random_20 (n=2) 59.01±0.59; random_35 (n=2) 736.11±0.35.
- HellaSwag: dense 0.613±0.009; stoch30 0.337±0.009; **dlr30 0.547±0.031**; stoch50 0.273±0.012; **dlr50 0.513±0.034**.
- ARC-c: dense 0.293±0.005; stoch30 0.247±0.012; **dlr30 0.320±0.008**; stoch50 0.207±0.012; **dlr50 0.323±0.045** (seed43 dip 0.26 — noisiest cell).
- Harness skip: dlr30 0.189 (band 0.25–0.35 ✗); dlr50 0.308 (band 0.45–0.55 ✗).

# Validation Gate — Is DLR Worth Pursuing to Publication?

**Last updated:** 2026-09-15 (backfilled from on-disk artifacts; ledger + RESULTS files were stale — see Progress Log).
**Owner question:** "Do we know *for sure* this is worth time, and can we *actually* publish a paper with this?"
**Short answer so far:** No certainty yet, but the idea has survived every kill-test thrown at it (2 families, 3 seeds, matched-random controls, FlexiDepth 8B harness smoke). One more cheap gate (G4: longer-train robustness) stands between us and spending real money on a second scale. Do NOT rent GPUs until G4 passes.

---

## 1. The claim we would publish (frozen scope)

> Learned token-level layer routing allocates transformer depth more effectively than random / input-agnostic layer skipping at the same compute budget — shown via a controlled comparison plus an independent audit of FlexiDepth, all through one standardized harness.

Full scope: `new/paper/scope_and_claim.md`. Key framing decision (do not revisit without new evidence):

- **NOT "a new routing architecture."** FlexiDepth (COLM 2025, per-layer router+adapter on frozen Llama-3-8B, 8/32 layers skipped, *no wall-clock speedup*) already occupies that ground. We cite, differentiate (upfront single-shot router vs per-layer sequential), and turn the risk into an asset by auditing their public checkpoint through our harness.
- **Contribution = rigorous controlled comparison + standardized harness + independent FlexiDepth audit.** This is explicitly invited by TMLR (reproducibility/audit track) and efficient-ML workshops.
- **Venue plan:** TMLR rolling submission (target Feb–Mar 2027; bar = claims supported by evidence, not novelty) + NeurIPS 2027 efficient-ML workshop short paper once the 1B block is verified. NeurIPS 2026 deadline already passed.

## 2. Publishability checklist (what "can we publish" concretely requires)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| P1 | Novelty gap holds vs FlexiDepth / MoD / LayerSkip / AdaSkip | ✅ done (lit review) | `new/docs/related_work.md` — 5 papers with 1-para diffs each; LoRA-Drop/MindSkip hallucination risks flagged, do NOT cite unverified |
| P2 | One evaluator for all variants, fail-closed on routing | ✅ built | `new/experiments/evaluate.py`; routed rows refuse to run full-depth; 6 past silent-fallback bugs fixed (see `RESULTS_MIN135M.md` bugs #1–6) |
| P3 | DLR beats *matched-compute* stochastic/random, not just dense | ✅ signal, 3 seeds, exploratory grade | §4 tables below: 4–20× ppl, +0.2 HellaSwag, consistent across seeds 42/43/44 |
| P4 | Router healthy (no collapse, token-varying, interpretable) | ✅ signal | collapse=0 all DLR rows, entropy 0.37–0.53, std_active 4.7–6.2 vs 0.0 fixed |
| P5 | Budget actually hit (harness skip inside target band) | ⚠️ PARTIAL — biggest methods threat | DLR30 sw1600 → harness skip 0.21 (band 0.25–0.35); DLR50 sw1600 → 0.34–0.35 (band 0.45–0.55). Mitigation exists: 002-style random-mask controls ARE matched by construction; but trained-stochastic comparison is NOT matched-compute until bands hit |
| P6 | 3-seed mean ± SD + 95% CI, paper-grade eval (5-shot MMLU/ARC, 500×512 ppl) | ❌ not yet | Current: 3 seeds trained+evaluated but exploratory grade only (0-shot, limit=100, 50×128 ppl); ledger rows all `completed_unverified`, none `verified` |
| P7 | Second model family (rebut "TinyLlama-only artifact") | ⚠️ weak pass | SmolLM2-135M (different arch/family) shows same direction (DLR ppl 62.9 vs stoch 347.5, Hella 0.44 vs 0.37). Still need a ~1–3B second family at full protocol (rented GPU, ~$150–400) |
| P8 | FlexiDepth 8B checkpoint through our harness + matched random control | ✅ unblocked (was "BLOCKED on VRAM") | `tinyllama1b-008-4bit-smoke` proves 4-bit eval path works; `flexidepth-001` ran: ppl 16.19, ARC-c 0.52, Hella 0.64, Wino 0.73. Missing: matched random-mask control at FlexiDepth's measured skip |
| P9 | Ablations (no-KD, token-vs-seq, always-keep) | ❌ not yet | Cheap; only after G4 |
| P10 | Honest compute reporting (structural FLOPs only, measured tps shows two-pass is SLOWER) | ✅ policy set | DLR tps < dense everywhere (e.g. 3628 vs 4642). Never claim wall-clock speedup. GSM8K `generate_until` EXCLUDED for routed rows (KV-cache interaction unsolved) — listed limitation |

## 3. Go / Kill criteria (decided in advance, not after seeing results)

**GO to paid scale-up (3B family + paper-grade evals) iff ALL of:**
- [ ] G1: Harness skip lands in band (0.25–0.35 / 0.45–0.55) OR claim reframed to measured-skip Pareto with matched random controls (reviewer-honest either way).
- [ ] G2: 3-seed DLR ppl mean ≥2× better than matched random-mask control at equal measured skip, same sign on HellaSwag, no collapse. *(Already true at exploratory grade — needs paper-grade confirmation.)*
- [ ] G3: Effect replicates in a second family at 1B+ (not just SmolLM2-135M plumbing scale).
- [ ] G4: Gap survives longer training (stochastic gets a fair chance to catch up — see §5).
- [ ] G5: FlexiDepth audit completes (their ckpt + matched random control through our harness).

**KILL / PIVOT iff ANY of:**
- [ ] DLR ≈ random at matched skip after calibration (routing learns nothing).
- [ ] Gap vanishes with longer training (150-step artifact).
- [ ] Gap exists only in ppl but inverts on all downstream tasks at paper-grade eval.
- [ ] Router collapses whenever skip hits the band (method can't be both sparse and learned).
- [ ] A prior-art check shows our exact comparison already published (re-run lit search before paid scale-up).

## 4. Current evidence (backfilled 2026-09-15 from artifacts — ledger/RESULTS were behind)

### 4a. TinyLlama-1.1B, 150 steps, seeds 42/43/44 (exploratory: 0-shot, limit=100, 50×128 ppl)

Runs on disk: `tinyllama1b_{dense,stochastic_30/50,token_dlr_30/50_sw1600}_{seed42,43,44}`; evals `tinyllama1b-001` (s42, sw400), `-006-seed43` (sw1600), `-007-seed44` (sw1600).

| seed | dense ppl | stoch30 ppl | **dlr30 ppl** | stoch50 ppl | **dlr50 ppl** | dlr30 skip (harness) | dlr50 skip |
|---|---|---|---|---|---|---|---|
| 42 | 10.11 | 96.89 | **17.83** | 634.8 | **24.35** | 0.15 | 0.25 |
| 43 | 10.07 | 101.4 | **21.94** | 642.2 | **36.61** | 0.21 | 0.34 |
| 44 | 10.16 | 115.2 | **21.90** | 767.6 | **40.45** | 0.21 | 0.34 |

HellaSwag acc_norm (DLR30 vs stoch30): 0.59/0.35 (s42), 0.52/0.33 (s43), 0.53/0.33 (s44). DLR50 vs stoch50: 0.56/0.26, 0.48/0.27, 0.50/0.29. ARC-c noisy (±0.05 at limit=100, near chance for some rows — do not overclaim). Router: collapse=0, entropy 0.37–0.53, std_active ≈5–6 (token-varying) every DLR row, all seeds.

### 4b. Matched random-mask controls (the honest comparison — matched BY CONSTRUCTION)

- `tinyllama1b-002-matched` (s42): learned 3.3× ppl over random at skip≈0.15, 4.6× at ≈0.25; +0.11/+0.15 ARC-c, +0.22/+0.18 HellaSwag.
- `-006/-007` include `random_20/random_35` on dense weights: ppl 59.6/736 (s43), 58.4/735 (s44) — DLR (21.9/36–40) beats both by 2–20× at comparable-or-higher skip.

### 4c. Calibration (Pareto) findings

- 20-step smoke does NOT move the router (skip ~0.005 for all sw) — calibrate at full length only (`calibrate_skip.py` now defaults to 150 steps, warns otherwise).
- Train-tail skip ≠ harness skip (~9pp gap: sw800 s42 train-tail 0.275 → harness 0.188). **Always calibrate against harness skip on a held-out split, never the train number.**
- sw400 → sw800 → sw1600 monotonically increases harness skip (0.15 → 0.19 → 0.21 @30; 0.25 → 0.31 → 0.35 @50) at modest ppl cost (17.8 → 21.3 @30). DLR50-sw1600 skip 0.35 is closest to band; DLR30 still low.

### 4d. SmolLM2-135M second-family signal (`min135m-001`)

DLR ppl 62.9 vs stochastic 347.5; Hella 0.44 vs 0.37; skip 0.18 vs 0.31. Same direction, different arch — weak G3 pass, full pass needs 1B+ family.

### 4e. FlexiDepth 8B unblocked (`flexidepth-001`, `tinyllama1b-008-4bit-smoke`)

4-bit eval path works on the 8GB 4060; FlexiDepth ckpt runs through harness (ppl 16.19 / ARC 0.52 / Hella 0.64 / Wino 0.73). Remaining: matched random control at its measured skip (G5).

## 5. The NEXT thing (single next experiment — read this first)

**G4: Longer-train robustness — 1 seed, TinyLlama-1.1B, 500 steps, dense / stochastic_30 / dlr_30(sw1600).**

Why this and not anything else:
1. It is the cheapest test that can still KILL the idea. At 150 steps, stochastic LoRA (which must be robust to ANY random subset) is plausibly undertrained, flattering DLR. If the gap closes at 500 steps, we stop before spending a dollar.
2. Seeds already replicate (G2 exploratory ✓); calibration direction known (sw1600 ✓); FlexiDepth unblocked (P8 ✓). None of those can kill us anymore — training-length can.
3. Cost: ~3× a 150-step run, still hours on the RTX 4060, $0. Everything else (3B rental, paper-grade evals) costs money or locks in claims.

Exact gate:
- Train with identical data/config/seeds, only `--max-steps 150 → 500` (+ checkpoint rule declared BEFORE: lowest held-out loss, never eval-set).
- Evaluate ppl + HellaSwag/ARC through the same harness + a random-mask control at DLR's measured skip.
- **PASS:** DLR ppl ≥2× better than stochastic AND matched-random at 500 steps, collapse=0. → GO to G1-calibration lock + paper-grade 3-seed eval + G3 second family (rent GPU).
- **FAIL:** gap closes to <1.3× or inverts. → KILL the "learned routing wins" claim; pivot options: (a) publish negative result as audit/workshop paper, (b) change routing design (per-layer sequential like FlexiDepth), (c) stop.

While G4 trains (GPU busy, no extra cost): do the $0 desk work — (i) update stale ledger + RESULTS files from §4 tables, (ii) re-run lit search for exact-duplicate comparisons, (iii) write the FlexiDepth matched-random control manifest for G5.

## 6. Ordered gate sequence (full path to GO/NO-GO on paid work)

| Gate | What | Cost | Status |
|---|---|---|---|
| G0 | Plumbing: loop runs, no NaN/OOM, router moves, harness fail-closed | $0, done | ✅ (bugs #1–6 fixed, bit-exact repro verified) |
| G1 | Calibration: harness skip in band (or Pareto-reframe decision) | $0, ~hours | ⚠️ in progress (sw1600 closest; DLR50 ≈0.35, DLR30 ≈0.21) |
| G2 | 3-seed replication, exploratory grade | $0 | ✅ signal (seeds 42/43/44 agree; needs paper-grade confirm) |
| G3 | Second family (135M weak pass → need 1B+ at protocol) | $0 local if 1B-class fits; else $$$ | ⚠️ weak pass only |
| **G4** | **Longer-train robustness (500 steps, 1 seed)** | **$0, hours** | **⬜ NEXT — do this now** |
| G5 | FlexiDepth ckpt + matched random control | $0 (4-bit path works) | ⬜ queued (manifest only until G4 passes) |
| G6 | Paper-grade block: 3 seeds × 5 variants, 5-shot MMLU/ARC, 500×512 ppl, mean±SD+CI, `verified` ledger rows | $$ (rented GPU for 3B) | ⬜ locked until G1–G5 pass |
| G7 | Ablations (no-KD, token-vs-seq, always-keep) + routing analysis + figures | $ | ⬜ after G6 |

Rule: no gate skipped, no paid GPU before G4 passes. G6 configs (`tinyllama1b-006/007`-style manifests) already exist as templates — do not launch the rented-GPU versions until the GO decision is recorded in §7.

## 7. Decision log (append-only — record GO/NO-GO here with date + evidence links)

- 2026-09-15: Gate file created. GO/NO-GO on paid scale-up: **PENDING G4**. Evidence: §4 tables (`new/artifacts/evals/tinyllama1b-00{1,2,6,7}`, `min135m-001`, `flexidepth-001`).
- 2026-09-15 (evening): **G4 PASS → GO approved for G1 lock + G5 + G3 + paid G6 planning.** Evidence: `tinyllama1b-010-g4-500` (500 steps, seed 42): dlr30 ppl 15.22 vs stoch30 69.44 (4.6×) vs random_20 85.77 (5.6×); dlr50 ppl 22.30 vs stoch50 201.40 (9.0×) vs random_35 1249.57 (56×); Hella +0.24 both levels; collapse=0; ent 0.39/0.45. Stochastic improved with training (threat was real) but gap stays multiples. Ledger rows `*_500_seed42` + eval `tinyllama1b-010-g4-500` written (all `completed_unverified`, exploratory grade). Next: G1 calibration lock, G5 FlexiDepth matched control, G3 second 1B+ family. NO paid GPU until G1+G5 recorded here.
- _Next entry: G1 lock (chosen skip_weights + harness skips in band, or Pareto-reframe decision)._
- 2026-09-15 (night): **G1 DECIDED via Pareto-reframe (bands dropped as hard requirement).** Sweep sw1600→2400→3200 @500 steps s42 (`tinyllama1b-011-g1-pareto-ppl`): harness skip dlr30 0.190→0.203→0.204, dlr50 0.368→0.392→0.397 — saturates (~0.20/~0.40), pressure ineffective past sw1600, ppl cost stays flat-to-mild (15.2→15.8, 22.3→25.5), router healthy throughout. The router has a natural sparsity level; the paper compares at MEASURED skips vs matched random controls (honest, standard). Locked weights for paper-grade runs: dlr30 sw1600, dlr50 sw1600 (best ppl/skip tradeoff; sw2400+ buys ~nothing). Band-hitting alternatives (always_keep 3→1, lower target_skip) deferred to G7 ablations — must not delay G3/G6.
- 2026-09-15 (night): **G5 PASS.** `flexidepth-002-control` ran the control `flexidepth-001` never did — random 50%-of-last-16 on the identical FlexiDepth 8B ckpt: ppl 237.74 vs ckpt 16.19 (14.7×), ARC-c 0.27 vs 0.52, Hella 0.37 vs 0.64, Wino 0.62 vs 0.73. Learned >> random at 8B on INDEPENDENT weights — the audit contribution is now real evidence, not a plan. Remaining gates: G3 (second 1B+ family) then paid G6.

## 8. Progress log (keep updating step by step — newest at bottom)

- 2026-09-14/15: Min135M plumbing + signal (`RESULTS_MIN135M.md`).
- 2026-09-15: TinyLlama-1.1B seed-42 block + matched audit + sw800 calibration (`RESULTS_TINYLLAMA1B.md`, evals 001/002/003).
- 2026-09-15 (found on disk, NOT yet in ledger/RESULTS): seeds 43+44 trained at sw1600 and evaluated (`tinyllama1b-006-seed43`, `tinyllama1b-007-seed44`); pareto ppl evals 004/005; 4-bit smoke 008; base+random smoke 009; **FlexiDepth-001 8B audit first run**. → TODO: promote these into `experiment_ledger.md` (currently all `completed_unverified`, missing 9+ rows) and refresh both RESULTS files.
- 2026-09-15: This gate file created. NEXT ACTION: launch G4 500-step runs.
- 2026-09-15 (evening): G4 DONE + PASS (5×500-step runs seed 42, eval `tinyllama1b-010-g4-500`, manifest `new/configs/eval/tinyllama1b-010-g4-500.json`). Ledger backfilled to 30 rows (seeds 43/44, sw1600 calibs, pareto 004/005, smokes 008/009, flexidepth-001, G4). New file `new/paper/DATA_CHECKLIST.md` maps every paper table/figure → artifact + gaps + 4 provenance fixes (git-commit `new/`, empty eval columns, latency.json, paper-grade manifest). `RESULTS_TINYLLAMA1B.md` updated with 3-seed table + G4. NEXT: G1 calibration lock (dlr30 needs more skip pressure; dlr50 near band at 500 steps), G5 FlexiDepth matched control, G3 second family pick.

## 9. Hard limitations (carry into the paper no matter what)

- Structural FLOP reduction only; measured tps shows two-pass DLR is SLOWER per token — no wall-clock claim, ever.
- GSM8K `generate_until` excluded for routed rows (KV-cache + routing unsolved).
- Upfront single-shot router (decide at layer N, apply to rest) vs per-layer sequential is an explicit design choice needing rationale + ablation paragraph — FlexiDepth/MoD do per-layer.
- Stochastic Depth gets SAME KD + regularizer as DLR in every comparison (parity already implemented — do not regress).
- No checkpoint picked after seeing eval numbers; one evaluator; every table row → ledger entry.

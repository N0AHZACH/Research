# HANDOFF FOR CLAUDE — Dynamic Layer Routing (DLR) Research
Date: 2026-09-11 (UTC)
Repo: https://github.com/N0AHZACH/Research.git , branch `main`
Local root: `C:\Users\nzach\OneDrive\Desktop\Research`
Author goal today: get something trainable on TinyLlama-1.1B that proves the system works and can scale. Final goal: publishable NeurIPS paper.

Paste everything below this line into Claude as context.

---

## 1. TL;DR

We are rebuilding LLM Dynamic Layer Routing research from first principles for publication. `old/` is archive/reference only. `new/` is the clean rebuild. Docs/protocols are done. Unified evaluator `new/experiments/evaluate.py` exists but training stack does not. Nothing is trainable yet in `new/`. Current laptop cannot do real 1.1B training (torch CPU-only, 8GB 4060). Today we need a CPU smoke loop + real 1.1B baseline plan.

Core claim to prove:
> Token-level input-conditional layer routing reduces effective transformer compute while preserving downstream quality better than input-agnostic stochastic depth under matched training, data, model, and evaluation conditions.

Central comparison is 3-way, matched: Dense LoRA (quality ceiling) vs Stochastic Depth (input-agnostic control) vs Token-DLR (method), at 30% skip (conservative) and 50% skip (aggressive).

## 2. Method in 30 seconds

- Hook-based two-pass forward, no model surgery. Phase 1: always-run first N layers (e.g. 0-3), capture hidden state H, abort via StopForwardException. Phase 2: lightweight 3-layer MLP Token-Level Gumbel-STE Router outputs binary gates [Batch, SeqLen, RoutableLayers]. Phase 3: gate_hooks on remaining layers: y = gate*F(x) + (1-gate)*x.
- Differentiability: Gumbel-Softmax hard=True forward, STE backward.
- Stabilization: KD KL vs frozen teacher (disable_adapter) + depth-scaled L1 penalty + quadratic target-skip regularizer.
- Optimizations for 96GB runs: gradient_checkpointing_enable(use_reentrant=False) + enable_input_require_grads(), expanded LoRA on q,k,v,o + gate,up,down, SEED=42, cosine LR 100 warmup.
- Reference legacy impls: `old/legacy/exp1_baseline_finetune.py` (dense), `old/legacy/exp10_token_routing_v2.py` (token DLR main contribution), `old/legacy/exp6_gumbel_router.py` (sequence router), `old/legacy/exp23_qwen7b_baseline.py / exp24_qwen7b_stochastic.py / exp25_qwen7b_token_routing.py` (7B triad).

## 3. Repo layout — what Claude needs to know

```
Research/
  new/ = CLEAN REBUILD, all new work goes here
    README.md — evaluator usage, paper rule
    FOLDER_STRUCTURE.md — canonical layout + naming <family>_<category>_seed<seed>
    configs/
      README.md
      models/{tinyllama1b,qwen3b,qwen7b,llama8b}/ — EMPTY, need family defs
      train/{tinyllama1b,qwen3b,qwen7b,llama8b}/ — EMPTY, need dense/stochastic_30/dlr_30/stochastic_50/dlr_50.json
      eval/{example_eval_manifest.json + empty tinyllama1b/,qwen3b/,qwen7b/,llama8b/}
      sweeps/ — EMPTY
    src/dlr/
      __init__.py (1 line), README.md — ALL MISSING: reproducibility.py, data.py, models.py, routing.py, losses.py, train_loop.py, evaluation.py, flop_accounting.py, routing_metrics.py, manifests.py, reproducibility.py
    experiments/
      evaluate.py (681 lines, WORKS) — unified evaluator, see §4
      README.md — lists train.py, analyze_routing.py, make_figures.py as EXPECTED BUT MISSING
    artifacts/{runs/,evals/,analysis/,audits/} — EMPTY
    figures/{main/,router_behavior/} — EMPTY
    tables/ — EMPTY
    paper/{README.md, figures/} — NO main.tex yet
    tests/{unit/,integration/} — EMPTY, README lists required tests
    docs/ — COMPLETE PROTOCOLS:
      RESTART_RESEARCH_PLAN.md (reset principle, Phase A dense -> B stochastic -> C DLR -> D pareto)
      NEURIPS_EVALUATION_STANDARD.md (matched conditions, 6 variants, task suite, seeds=3, validity gates, output schema)
      NEURIPS_PUBLICATION_EXECUTION_PLAN.md (workstreams A-F, gates 1-5)
      10_step_neurips_publication_plan.md (claim->evaluator->block->train->1B->7/8B->ablations->analysis->figures->paper)
      model_scaling_plan_v2.md (1B->3B->4/5B->7/8B->11B->14/15B, 2 families per scale, Level A 30% / Level B 50%, always_keep=max(2,round(0.15*total_layers)))
      staged_execution_protocol.md (S0 Register -> S1 Config -> S2 Smoke -> S3 Train -> S4 Checkpoint Audit -> S5 Evaluate -> S6 Validity -> S7 Analysis -> S8 Table -> S9 Paper)
      experiment_ledger.md (only restart-000 planned row)
      evaluation_checklist.md
    scripts/, docs continued
  old/ = ARCHIVE, do not trust for paper without audit
    legacy/{exp1..exp29...} — chronological scripts TinyLlama->3B->7B/8B
    PROJECT_CONTEXT.md — full history Phases 1-5, methodology, next steps
    README.md — DLR description + how to run old exps
    PUBLICATION_READY_PATCH_PLAN.md
    results/, figures/, logs/, legacy_results/
    exp23_qwen7b_baseline_metrics_20260623_040441.csv, exp26_llama8b_baseline_metrics_20260623_184138.csv — example CSV cols: Epoch,Step,Train Loss,Val Loss,Perplexity,Active Layers,Skip Ratio
```

Naming: `qwen7b_dense_seed42, qwen7b_stochastic_30_seed42, qwen7b_dlr_30_seed42, llama8b_dlr_50_seed43`. Categories: `dense, stochastic_30, dlr_30, stochastic_50, dlr_50` + ablations `dlr_30_no_kd, random_router_30, sequence_router_30...`

## 4. Eval system status (most built piece)

File: `new/experiments/evaluate.py`
Usage:
```
python new/experiments/evaluate.py --manifest new/configs/eval/example_eval_manifest.json --dry-run
python new/experiments/evaluate.py --manifest path/to/eval_manifest.json
python new/experiments/evaluate.py --manifest M --skip-lm-eval  # ppl/routing only
python new/experiments/evaluate.py --manifest M --skip-perplexity
```
Manifest shape (see `new/configs/eval/example_eval_manifest.json` — TinyLlama-1.1B-Chat-v1.0, tasks=[arc_challenge], 5-shot, batch 1, limit 10, ppl Wikitext-103-raw-v1 10 samples):
`eval_run_id, model_family, model_id, output_dir, device, dtype, attention_implementation, tasks, num_fewshot, batch_size, limit, perplexity{dataset,config,split,samples,max_length}, variants[{name,type,checkpoint,router_weights,always_keep_layers}]`
Types: `base, lora, dense_lora, stochastic, token_dlr`. Validates checkpoint paths exist, router_weights exist, always_keep>0.

Outputs `artifacts/evals/<eval_run_id>/`: manifest.json, environment.json (torch/transformers/lm-eval versions, git commit, cuda), eval_summary.csv, eval_summary.json, task_results.json, perplexity.json, routing_metrics.csv. Missing vs standard: `latency.json`.

Key impl notes:
- `load_variant()` : base->from_pretrained; lora/stochastic->PeftModel then merge_and_unload; token_dlr->PeftModel + load Router MLP (hidden->hidden//2->GELU->hidden//4->GELU->num_layers) from router_weights.pt
- `routed_forward()` : capture hook on layers[always_keep-1] -> router(hard=True) gates -> gate_hooks on layers[always_keep:] with residual bypass. Verifies gates dim [B,S,L].
- `evaluate_perplexity()` : Wikitext filter len>100, correct -100 label masking for pad, reports NLL, perplexity=exp(loss), num_tokens, tokens_per_sec.
- `routing_metrics_from_gates()` : mean_active_layers, std/min/max, router_entropy (binary entropy), structural_flop_reduction_pct, collapse_flag (=1 if mean_gate<0.02 or >0.98 or entropy<0.10), per_layer_activity.
- `run_lm_eval()` : uses lm-eval HFLM for non-DLR only. **CRITICAL GAP: raises RuntimeError for token_dlr — "intentionally disabled in v1. Use perplexity/routing now; add routed HFLM wrapper before claiming task scores."** So DLR task accuracy (MMLU/ARC/GSM8K) not yet comparable.
- Rule: if result supports NeurIPS paper, must go through this evaluator or be marked exploratory.

Required benchmarks per NEURIPS_EVALUATION_STANDARD.md: ARC-Challenge acc_norm, GSM8K EM, MMLU acc (5-shot), HellaSwag acc_norm (10-shot), Winogrande, WikiText ppl. Small exploratory: ARC+GSM8K+MMLU+ppl OK but label exploratory. Must also report expected/measured active layers, per-layer activity, FLOP reduction, tok/sec, peak mem, router entropy, collapse flag. Checkpoint rule must be pre-declared (default: lowest val loss, run eval once). Seeds: 1 smoke, 3 main tables, 5 if deltas small. For 7B/8B compromise: 3 seeds at final target, 1 seed for pareto sweep.

## 5. Training status — NOTHING EXISTS YET

Need to build: `src/dlr/*.py` + `experiments/train.py` + `configs/train/tinyllama1b/*.json` + tests. Required invariants: one seed fn, one data loader, one FLOP impl, one CSV schema, one manifest per run, no hidden defaults. Manifest fields: run_id, created_at, git_commit, command, config_path, model_name/revision, tokenizer, dataset/revision, seed, hardware, precision, attn_impl, trainable/total params, expected/measured active layers, checkpoint_path, metrics_path.

Standard flow: `config -> train -> checkpoint -> eval manifest -> evaluate -> audit -> analysis -> figures/tables -> paper`

## 6. Current hardware/env (this laptop, 2026-09-11)

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8188 MiB, driver 581.29
- torch 2.6.0+cpu, cuda_available=False (CPU-only build — must pip install CUDA build to use 4060)
- transformers 5.7.0, peft 0.19.1, datasets 4.8.5, lm-eval 0.4.11, accelerate 1.13.0
- python 3.13 (path AppData/Roaming/Python/Python313)
- Historical target hardware was RTX PRO 6000 96GB. 4060 8GB can only do 1.1B QLoRA 4-bit smoke/short runs, not full 7B/8B.
- Git: origin https://github.com/N0AHZACH/Research.git, branch main, HEAD 93ab83f "Organize experiment files...". Working tree has deletions of old root files (moved to old/) — needs commit decision.

## 7. Scaling plan

Per `model_scaling_plan_v2.md`: 1B (TinyLlama-1.1B + second 1B) -> 3B (Qwen2.5-3B + OpenLLaMA-3B) -> 4/5B -> 7/8B main (Qwen2.5-7B + Llama-3.1-8B) -> 11B -> 14/15B. Per scale: 2 families * (1 dense + 2 stochastic + 2 DLR) =10 runs *3 seeds=30 runs. Plus ablations DLR-no-KD + random router. Promotion rule: don't move up until lower scale passes S6 validity. Today: start with TinyLlama-1.1B single family, single seed, dense only.

Layer rule: always_keep = max(2, round(0.15*total_layers)) e.g. 22 layers->3 keep/19 routable, 28->4/24, 32->5/27.

## 8. What today must achieve (user goal)

1. Prove loop works: config->1 train step works->1 eval batch works->manifest writes->no router-shape/hook errors
2. Get 1.1B dense baseline config + smoke train (20-50 steps CPU) + eval dry-run + ppl-only eval
3. Have clear comparison table shell (dense vs stochastic vs DLR columns) even if only dense filled
4. Leave with scalable plan to GPU (install torch-cu126 or rent 6000Ada/A100 for real runs)

Suggested execution S0-S6 for `tinyllama1b_dense_seed42` detailed in staged_execution_protocol.md.

## 9. Ideas wanted from Claude

1. Minimal `train.py` + `src/dlr/` design for dense-only that generalizes to stochastic/DLR without rewrite?
2. TinyLlama-1.1B QLoRA baseline hyperparams for 8GB 4060 (rank, batch, grad-accum, LR, warmup, tokens) that still respect matched-controls standard?
3. How to implement routed HFLM wrapper to unblock DLR lm-eval tasks?
4. Smoke-test strategy on CPU that actually catches hook/collapse/FLOP bugs before GPU spend?
5. Second 1B family recommendation (stable, open, tokenizer distinct from TinyLlama)?
6. What would make this NeurIPS-credible vs rejected (threats to validity)?
7. Pareto sweep design: how to calibrate compute_penalty to hit 25-35% and 45-55% bands without tuning on test?

## 10. Commands for Claude to assume

```bash
python new/experiments/evaluate.py --manifest new/configs/eval/example_eval_manifest.json --dry-run
pip install torch transformers peft datasets accelerate bitsandbytes lm-eval matplotlib pandas
python old/legacy/exp10_token_routing_v2.py --fresh  # old reference only
python old/legacy/exp7_eval_harness.py
```

## 11. Rules (from PROJECT_CONTEXT.md + standards)

- DO NOT remove gradient_checkpointing(use_reentrant=False) from PEFT models
- ALWAYS seeds=42 for python/random/numpy/torch, cudnn.benchmark=False
- ALWAYS log same FLOP/hardware metrics for pandas outer-joins
- Keep LR/batch/grad-accum matched across baselines
- Never report structural FLOP reduction as wall-clock speedup without measurement
- Never choose checkpoint after seeing final benchmarks; never use different selection per method
- Only `verified` ledger runs go in paper tables/figures
- No new one-off exp*.py scripts — shared code + configs only

## 12. Key files to read first (in order)

1. `new/docs/RESTART_RESEARCH_PLAN.md`
2. `new/docs/NEURIPS_EVALUATION_STANDARD.md`
3. `new/docs/staged_execution_protocol.md`
4. `new/docs/model_scaling_plan_v2.md`
5. `new/docs/10_step_neurips_publication_plan.md`
6. `new/experiments/evaluate.py`
7. `new/configs/eval/example_eval_manifest.json`
8. `new/FOLDER_STRUCTURE.md`
9. `old/PROJECT_CONTEXT.md`
10. `old/legacy/exp10_token_routing_v2.py`

---
End handoff. Ask Claude for: (a) critique of plan, (b) concrete minimal train.py/src design, (c) 1.1B baseline config for 8GB + CPU-smoke variant, (d) routed eval wrapper sketch.

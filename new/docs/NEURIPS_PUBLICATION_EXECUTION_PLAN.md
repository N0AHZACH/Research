# NeurIPS Publication Execution Plan

Current date: 2026-08-15.

This project's goal is publication, not just running larger experiments. The operating standard is therefore:

> Every code change, experiment, and figure must either increase the credibility of the central claim or be deferred.

## 1. Central Claim

The paper should make one clean claim:

> Token-level Dynamic Layer Routing provides a better quality/compute tradeoff than input-agnostic stochastic depth under matched model, data, training, and evaluation conditions, and this effect holds across model families and scales.

Do not overclaim wall-clock speedup unless measured. Do not hide perplexity degradation if it appears. Do not use one-off runs as main evidence.

## 2. Submission Shape

The strongest NeurIPS version should contain:

- A precise method section for token-level layer routing.
- A strict matched baseline protocol.
- Main results at 7/8B across two model families.
- Scaling evidence from 1B and 3B.
- At least one larger confirmation point at 11B or 14/15B if compute allows.
- Ablations proving KD, learned routing, and token-level routing matter.
- Honest compute reporting: structural FLOPs, active layers, tokens/sec, latency, and memory.

The paper should be built around the 7/8B result, not around every historical experiment.

## 3. Workstreams

### Workstream A: Research Protocol

Status target: locked before new long runs.

Required outputs:

- `docs/NEURIPS_EVALUATION_STANDARD.md`
- `docs/model_scaling_plan_v2.md`
- `docs/experiment_ledger.md`
- `docs/evaluation_checklist.md`

Exit criteria:

- Every required metric is named.
- Every baseline is named.
- Every validity failure is named.
- The seed policy is fixed.
- The checkpoint selection rule is fixed.

### Workstream B: Clean Implementation

Status target: complete before expensive 7/8B reruns.

Required outputs:

- Shared config files.
- Shared manifest writer.
- Shared evaluation entry point.
- Shared FLOP and active-layer accounting.
- Shared router metrics.
- Tests for routing, hooks, metrics, and manifests.

Exit criteria:

- Dense, stochastic-depth, and DLR variants produce the same metric schema.
- Routed evaluation fails if hooks/router are inactive.
- No experiment script contains hidden paper-critical defaults.

### Workstream C: Small-Scale Validation

Status target: prove the pipeline before scale.

Required outputs:

- Two 1B-class model families.
- Dense, stochastic, and DLR at both parameter levels.
- At least one KD ablation.
- At least one random/parameter-free router ablation.

Exit criteria:

- The protocol catches bad runs.
- The result direction is interpretable.
- Metrics, manifests, and artifacts are complete.

### Workstream D: Main 7/8B Evidence

Status target: main paper table.

Required model families:

- Qwen2.5-7B-class model.
- Llama-3.1-8B-class model.

Required variants per family:

- Dense LoRA.
- Stochastic depth, 30% skip.
- DLR, 30% skip.
- Stochastic depth, 50% skip.
- DLR, 50% skip.

Required seeds:

- 3 seeds for each main variant.
- 5 seeds only if the main deltas are small enough that uncertainty threatens the claim.

Exit criteria:

- DLR beats matched stochastic depth on the main quality/compute tradeoff.
- Perplexity and latency are reported honestly.
- Raw results and summaries are reproducible from manifests.

### Workstream E: Scaling Evidence

Status target: strengthen but do not distract.

Use:

- 3B as trend confirmation.
- 4/5B as optional intermediate scale.
- 11B or 14/15B as larger confirmation if compute allows.

Exit criteria:

- Results support the same trend without requiring new narrative exceptions.
- Incomplete larger-scale results are omitted or clearly labeled exploratory.

### Workstream F: Manuscript

Status target: write continuously, but only promote verified results.

Required sections:

- Abstract.
- Introduction and claim.
- Related work.
- Method.
- Experimental protocol.
- Main results.
- Ablations.
- Scaling analysis.
- Compute and latency analysis.
- Limitations.
- Reproducibility statement.

Exit criteria:

- Every table row maps to a verified run.
- Every figure can be regenerated from artifacts.
- Limitations are explicit.
- The conclusion matches the actual evidence.

## 4. First 10 Concrete Tasks

1. Freeze the old repo state as historical archive material.
2. Create `configs/` for model family, training, routing, and evaluation.
3. Create a manifest validator.
4. Create a single evaluator that consumes manifests.
5. Create shared metric schemas for training, evaluation, routing, and latency.
6. Add tests for seed control, FLOP accounting, router shapes, hook lifecycle, and manifest validation.
7. Rebuild the 1B dense baseline through the new pipeline.
8. Rebuild 1B stochastic depth at 30% and 50% skip.
9. Rebuild 1B DLR at 30% and 50% skip.
10. Run the same 1B protocol on a second model family.

Do not start new large runs until tasks 1-6 are complete.

## 5. Decision Gates

### Gate 1: Pipeline Gate

Pass only if:

- Shared config, manifest, metrics, and evaluator exist.
- Smoke tests pass.
- One dense run evaluates end to end.

### Gate 2: Baseline Gate

Pass only if:

- Dense and stochastic-depth baselines run through the same path.
- Matched compute accounting is correct.
- Evaluation output schemas match.

### Gate 3: DLR Gate

Pass only if:

- DLR runs through the same evaluator.
- Router activity is measured.
- Collapse detection works.
- Routed evaluation cannot silently fall back to dense.

### Gate 4: Main Evidence Gate

Pass only if:

- 7/8B results exist for two model families.
- Main variants have 3 seeds.
- DLR improves the matched quality/compute frontier.
- Failure modes are documented.

### Gate 5: Submission Gate

Pass only if:

- All main claims point to verified ledger entries.
- Figures regenerate.
- The manuscript includes limitations.
- The abstract does not overclaim beyond measured evidence.

## 6. What Not To Do

- Do not keep adding one-off experiment scripts.
- Do not tune against final benchmarks.
- Do not move to 14/15B before the 1B and 7/8B protocol is clean.
- Do not report structural compute as wall-clock speedup.
- Do not trust legacy results without provenance.
- Do not let the manuscript become a diary of experiments.

## 7. Immediate Next Move

The next engineering move is to build the clean protocol machinery:

```text
configs/
src/dlr/reproducibility.py
src/dlr/manifests.py
src/dlr/flop_accounting.py
src/dlr/routing_metrics.py
experiments/evaluate.py
tests/
```

Once that exists, run the smallest complete 1B experiment family. That is how this becomes publishable instead of merely promising.

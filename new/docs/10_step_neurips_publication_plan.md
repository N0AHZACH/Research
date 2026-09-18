# 10-Step NeurIPS Publication Plan

This is the execution blueprint for turning Dynamic Layer Routing into a publishable NeurIPS submission.

The goal is not to run many experiments. The goal is to produce clean evidence for one claim:

> Token-level Dynamic Layer Routing improves the quality/compute tradeoff over matched input-agnostic stochastic depth across model families, model scales, and compute budgets.

## Step 1: Freeze The Research Claim

### What We Do

Define exactly what the paper is trying to prove.

The claim should be:

> Learned token-level layer routing allocates transformer depth more effectively than random or input-agnostic layer skipping at the same compute budget.

### What We Make

- `new/docs/core_claim.md`
- Clear success/failure criteria.
- A list of claims we are not making.

### Success Criteria

- DLR beats matched stochastic depth at 30% skip.
- DLR beats matched stochastic depth at 50% skip.
- Results hold for at least two model families at the main scale.
- Compute is measured, not hand-waved.

### Failure Criteria

- DLR only beats dense or stochastic in one isolated setting.
- DLR uses different data, training budget, checkpoint rule, or evaluator.
- DLR saves structural FLOPs but quality collapses.
- DLR is evaluated accidentally as a full-depth model.

## Step 2: Build The Single Evaluation Framework

### What We Do

Make `new/experiments/evaluate.py` the only official evaluator for paper results.

It must evaluate individual models so we can understand:

- task performance
- perplexity
- active layers
- structural FLOP savings
- real latency/tokens-per-second
- routing behavior
- router collapse

### What We Make

- `new/experiments/evaluate.py`
- `new/configs/eval/*.json`
- `new/artifacts/evals/<eval_run_id>/`

Required output files:

```text
manifest.json
environment.json
eval_summary.csv
eval_summary.json
task_results.json
perplexity.json
routing_metrics.csv
latency.json
```

### Success Criteria

- One command evaluates any model variant.
- Dense, stochastic, and DLR outputs share one schema.
- Routed DLR evaluation fails if router hooks are inactive.
- No evaluator silently falls back to dense/full-depth behavior.

## Step 3: Define The Standard Experiment Block

### What We Do

Every model family gets the same five core variants.

```text
Dense LoRA
Stochastic Depth 30%
DLR 30%
Stochastic Depth 50%
DLR 50%
```

### What We Make

Training config templates:

```text
new/configs/train/<model_family>/dense.json
new/configs/train/<model_family>/stochastic_30.json
new/configs/train/<model_family>/dlr_30.json
new/configs/train/<model_family>/stochastic_50.json
new/configs/train/<model_family>/dlr_50.json
```

Evaluation manifests:

```text
new/configs/eval/<model_family>_main.json
```

### Success Criteria

- Every model family uses the same protocol.
- Stochastic and DLR are matched by measured compute.
- No model family gets special treatment.

## Step 4: Build Training Infrastructure

### What We Do

Stop writing one-off experiment files. Build shared training machinery.

### What We Make

```text
new/src/dlr/reproducibility.py
new/src/dlr/data.py
new/src/dlr/models.py
new/src/dlr/routing.py
new/src/dlr/losses.py
new/src/dlr/flop_accounting.py
new/src/dlr/manifests.py
new/src/dlr/metrics.py
new/experiments/train.py
```

### Success Criteria

- Every run writes a manifest.
- Every run writes the same metric schema.
- Every run records seed, commit, data, model revision, and command.
- Training code supports dense, stochastic, and DLR variants from config.

## Step 5: Validate At 1B Scale

### What We Do

Run the complete pipeline cheaply before spending serious GPU time.

Use two 1B-class models if possible.

For each:

```text
Dense LoRA
Stochastic 30%
DLR 30%
Stochastic 50%
DLR 50%
```

### What We Make

```text
new/artifacts/runs/1b_model_a/
new/artifacts/runs/1b_model_b/
new/artifacts/evals/1b_model_a/
new/artifacts/evals/1b_model_b/
```

### Success Criteria

- Evaluator works end to end.
- DLR gates do not collapse.
- Stochastic and DLR compute levels are matched.
- Results are interpretable enough to justify scaling.

## Step 6: Run Main 7/8B Experiments

### What We Do

This is the main NeurIPS evidence.

Use:

```text
Qwen 7B
Llama 8B
```

For each model family:

```text
Dense LoRA
Stochastic 30%
DLR 30%
Stochastic 50%
DLR 50%
```

Use 3 seeds for each main variant.

### What We Make

```text
new/artifacts/runs/qwen7b/
new/artifacts/runs/llama8b/
new/artifacts/evals/qwen7b/
new/artifacts/evals/llama8b/
```

### Success Criteria

- DLR 30% beats stochastic 30%.
- DLR 50% beats stochastic 50%.
- Results hold across Qwen and Llama.
- The paper has a credible main table.

## Step 7: Add Ablations

### What We Do

Prove the method components matter.

Required ablations:

```text
DLR with KD vs DLR without KD
learned router vs random matched router
token-level routing vs sequence-level routing
30% skip vs 50% skip
different always-keep layer counts
```

### What We Make

```text
new/configs/train/ablations/
new/configs/eval/ablations/
new/artifacts/evals/ablations/
```

### Success Criteria

- KD improves stability or quality.
- Learned routing beats random matched routing.
- Token-level routing is better or more informative than sequence-level routing.
- Always-keep layer choice is justified.

## Step 8: Analyze Routing Behavior

### What We Do

Show that DLR learned meaningful token-dependent compute allocation.

Analyze active depth by:

- punctuation
- stop words
- rare tokens
- content words
- math tokens
- high-loss tokens
- reasoning-heavy examples

### What We Make

```text
new/experiments/analyze_routing.py
new/artifacts/analysis/router_behavior/
new/figures/router_behavior/
```

Required figures:

```text
token_type_active_depth.pdf
per_layer_activity_heatmap.pdf
example_routing_heatmap.pdf
router_entropy_distribution.pdf
```

### Success Criteria

- DLR does not look random.
- Router behavior is interpretable.
- The paper can claim token-level depth demand is measurable and learnable.

## Step 9: Build Publication Figures And Tables

### What We Do

Generate all figures from evaluator artifacts only.

### What We Make

```text
new/experiments/make_figures.py
new/figures/main_pareto.pdf
new/figures/matched_compute_bars.pdf
new/figures/scaling_trend.pdf
new/figures/router_heatmap.pdf
new/figures/token_type_depth.pdf
new/figures/latency_throughput.pdf
new/tables/main_results.csv
new/tables/ablations.csv
```

### Success Criteria

- Every figure can be regenerated.
- Every table row maps to a verified eval run.
- Main visual story is obvious: DLR sits above stochastic depth on quality/compute.

## Step 10: Write And Audit The NeurIPS Paper

### What We Do

Write the manuscript around verified results only.

### What We Make

```text
new/paper/main.tex
new/paper/references.bib
new/paper/figures/
new/docs/result_audit.md
new/docs/reproducibility_statement.md
new/docs/limitations.md
```

Paper structure:

```text
Abstract
Introduction
Related Work
Method
Evaluation Protocol
Main Results
Scaling Results
Ablations
Routing Analysis
Compute and Latency
Limitations
Conclusion
Reproducibility Statement
```

### Success Criteria

- Every claim points to verified results.
- Every table row has a manifest.
- Every figure is reproducible.
- Limitations are explicit.
- The abstract does not overclaim.

## The Critical Path

Do these in order:

```text
1. Claim
2. Evaluator
3. Standard run block
4. Training infrastructure
5. 1B validation
6. 7/8B main evidence
7. Ablations
8. Routing analysis
9. Figures/tables
10. Paper audit
```

Do not skip directly to large models. The fastest path to publication is the cleanest path, not the biggest first run.

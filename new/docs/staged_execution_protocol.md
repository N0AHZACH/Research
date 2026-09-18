# Staged Execution Protocol

This protocol defines how we execute the research one stage at a time for every model and parameter category.

The goal is publication-quality evidence. A model does not move forward just because training finished. It moves forward only after it passes the current stage gate.

## Core Unit Of Work

The basic unit is:

```text
model_family + parameter_category + seed
```

Example:

```text
qwen7b + dlr_30 + seed_42
llama8b + stochastic_50 + seed_43
tinyllama1b + dense + seed_44
```

## Parameter Categories

Every model family uses the same categories:

```text
dense
stochastic_30
dlr_30
stochastic_50
dlr_50
```

Optional ablation categories:

```text
dlr_30_no_kd
dlr_50_no_kd
random_router_30
random_router_50
sequence_router_30
sequence_router_50
```

## Model Stages

Each model/parameter/seed moves through the same stages:

```text
S0: Register
S1: Config
S2: Smoke Test
S3: Train
S4: Checkpoint Audit
S5: Evaluate
S6: Validity Audit
S7: Analysis
S8: Figure/Table Promotion
S9: Paper Promotion
```

## S0: Register

### What We Do

Create a ledger row before running anything.

### What We Make

```text
new/docs/experiment_ledger.md
```

Required fields:

```text
run_id
model_family
model_scale
parameter_category
seed
status
config_path
artifact_dir
notes
```

### Gate

Pass only if the run has a unique `run_id`.

## S1: Config

### What We Do

Create a training config and evaluation manifest.

### What We Make

```text
new/configs/train/<model_family>/<parameter_category>_seed<seed>.json
new/configs/eval/<model_family>/<parameter_category>_seed<seed>.json
```

### Gate

Pass only if:

- Model ID is fixed.
- Dataset is fixed.
- Seed is fixed.
- Parameter category is explicit.
- Checkpoint rule is explicit.
- Evaluation tasks are explicit.

## S2: Smoke Test

### What We Do

Run a tiny version before expensive training.

### What We Make

```text
new/artifacts/smoke/<run_id>/
```

### Gate

Pass only if:

- Model loads.
- Dataset loads.
- One train step works.
- One eval batch works.
- Manifest writes.
- No router-shape errors.
- No hook lifecycle errors.

## S3: Train

### What We Do

Run the real training job.

### What We Make

```text
new/artifacts/runs/<model_family>/<parameter_category>/<seed>/
```

Required files:

```text
manifest.json
train_metrics.csv
checkpoint/
environment.json
```

### Gate

Pass only if:

- Training finishes or fails with a recorded reason.
- Metrics schema is valid.
- Checkpoint selection rule was followed.
- DLR runs include routing metrics.
- Stochastic runs include measured skip rate.

## S4: Checkpoint Audit

### What We Do

Verify that the checkpoint can be loaded and matches the manifest.

### What We Make

```text
new/artifacts/audits/<run_id>_checkpoint.json
```

### Gate

Pass only if:

- Checkpoint exists.
- Adapter/config files exist.
- Router weights exist for DLR.
- Model family and tokenizer match manifest.
- Trainable parameter count is recorded.

## S5: Evaluate

### What We Do

Run the unified evaluator.

### What We Make

```text
new/artifacts/evals/<eval_run_id>/
```

Required files:

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

### Gate

Pass only if:

- Evaluation completes through `new/experiments/evaluate.py`.
- Dense/stochastic/DLR outputs share the same schema.
- DLR evaluation confirms routing is active.
- Perplexity uses correct padding-label masking.
- Latency/tokens-per-second are recorded or explicitly marked unavailable.

## S6: Validity Audit

### What We Do

Decide whether the run is publishable evidence.

### What We Make

```text
new/artifacts/audits/<run_id>_validity.json
```

### Gate

Pass only if:

- No silent fallback happened.
- No routed model evaluated full-depth by accident.
- Compute category matched its target band.
- Router did not collapse, or collapse is explicitly marked as a result.
- Evaluation task settings match the model family protocol.
- Required artifacts exist.

Possible statuses:

```text
verified
completed_unverified
invalid
exploratory
```

## S7: Analysis

### What We Do

Analyze the model's behavior.

For DLR:

- token active depth
- per-layer activity
- router entropy
- collapse status
- token-type compute allocation

For all variants:

- task quality
- perplexity
- active compute
- latency/tokens-per-second
- memory

### What We Make

```text
new/artifacts/analysis/<run_id>/
```

### Gate

Pass only if the result can be compared against its matched counterpart:

```text
dlr_30 vs stochastic_30
dlr_50 vs stochastic_50
```

## S8: Figure/Table Promotion

### What We Do

Promote verified results into paper tables and figures.

### What We Make

```text
new/tables/main_results.csv
new/tables/ablations.csv
new/figures/
```

### Gate

Pass only if:

- Result status is `verified`.
- Matched stochastic baseline exists.
- Dense reference exists.
- Figure/table row links to artifact path.

## S9: Paper Promotion

### What We Do

Use the result in the manuscript.

### What We Make

```text
new/paper/main.tex
new/docs/result_audit.md
```

### Gate

Pass only if:

- Claim is supported by verified results.
- Limitations are stated.
- The paper does not overclaim structural compute as wall-clock speedup.

## Execution Order Per Model Family

For each model family, execute categories in this order:

```text
1. dense
2. stochastic_30
3. dlr_30
4. stochastic_50
5. dlr_50
```

Reason:

- Dense gives the quality ceiling.
- Stochastic 30 gives the conservative matched baseline.
- DLR 30 tests quality-preserving routing.
- Stochastic 50 gives the aggressive matched baseline.
- DLR 50 tests whether routing survives higher compute pressure.

## Execution Order Across Scales

Use this order:

```text
1B model family A
1B model family B
3B model family A
3B model family B
7/8B model family A
7/8B model family B
11B optional
14/15B optional
```

Do not move to expensive scales until the lower scale passes S6 validity audit.

## Simple Rule

Every run must move through:

```text
register -> config -> smoke -> train -> checkpoint audit -> evaluate -> validity audit -> analysis -> promotion
```

If a run fails a stage, stop and fix the system before scaling.

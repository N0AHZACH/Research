# NeurIPS-Level Evaluation Standard

This document defines the minimum evaluation standard for Dynamic Layer Routing (DLR). It is intentionally strict. A result that does not satisfy this protocol may still be useful for debugging, but it should not appear in the manuscript.

## 1. Evaluation Philosophy

The paper should not ask whether DLR can produce an impressive number in one setting. It should ask whether DLR gives a better quality/compute tradeoff than strong, matched alternatives.

Every publishable comparison must hold these fixed:

- Base model family and checkpoint revision.
- Tokenizer and tokenizer revision.
- Training dataset, split, revision, preprocessing, sequence length, and sample count.
- LoRA target modules and rank.
- Optimizer, learning rate, scheduler, warmup, batch size, gradient accumulation, and training tokens.
- Seed policy.
- Precision and attention implementation.
- Evaluation tasks, shots, prompt templates, decoding settings, and batch size.
- Checkpoint selection rule.

If one method gets more training, a cleaner checkpoint selection rule, a different prompt template, or a different eval path, the comparison is invalid.

## 2. Required Model Variants

A complete experiment family must include:

| Variant | Purpose | Required |
| --- | --- | --- |
| Pretrained base | Measures how much finetuning helps or hurts | Yes |
| Dense LoRA | Full-depth finetuned control | Yes |
| Stochastic depth | Input-agnostic compute-saving baseline | Yes |
| Token DLR | Main method | Yes |
| Token DLR without KD | Tests whether KD is essential | For paper |
| Random or parameter-free matched router | Tests whether learned routing matters | For paper |

The main claim should be based on matched-compute comparisons between stochastic depth and token DLR. Dense LoRA is the quality reference, not the only baseline.

## 3. Required Benchmarks

Use one standard evaluation harness for every variant in a model family.

### Core Task Metrics

For 7B/8B-scale models, the minimum task suite is:

| Task | Setting | Primary Metric | Why |
| --- | --- | --- | --- |
| ARC-Challenge | 25-shot or standard lm-eval default for the chosen model scale | `acc_norm` | Reasoning signal, historically discriminative here |
| GSM8K | 8-shot or standard lm-eval default | exact match / flexible extract | Arithmetic reasoning |
| MMLU | 5-shot | accuracy | Broad knowledge, only meaningful above chance |
| HellaSwag | 10-shot | `acc_norm` | Commonsense completion |
| Winogrande | 5-shot | accuracy | Commonsense/coreference |
| WikiText-103 or WikiText-2 | fixed split | perplexity | Token distribution quality |

For small exploratory runs, ARC-Challenge, GSM8K, MMLU, and perplexity are acceptable, but the run must be labeled exploratory.

### Compute and Routing Metrics

Every DLR or stochastic-depth result must report:

- Expected active layers.
- Measured active layers.
- Per-layer activity.
- Per-token active depth distribution.
- Structural FLOP reduction.
- Tokens per second.
- Peak GPU memory.
- Router entropy for DLR.
- Router collapse flags.

Accuracy without compute accounting is not a DLR result. Compute savings without quality metrics are not a DLR result.

## 4. Checkpoint Selection Rule

The checkpoint selection rule must be declared before training starts.

Allowed rules:

- Best validation loss on a fixed validation set.
- Final checkpoint after a fixed number of tokens.
- Best validation task metric on a fixed development set, only if that development set is separate from final evaluation.

Disallowed rules:

- Choosing the checkpoint after seeing final benchmark results.
- Using a different selection rule per method.
- Rerunning only failed variants until one looks good.

Recommended default:

> Select the checkpoint with lowest validation loss on the fixed validation split, then run the full external evaluation suite exactly once.

## 5. Seeds and Statistical Evidence

One-seed results are debugging evidence, not publication evidence.

Minimum:

- Smoke/debug runs: 1 seed.
- Main tables: 3 seeds.
- Final claims where deltas are small: 5 seeds or confidence intervals from bootstrap plus at least 3 training seeds.

Required reporting:

- Mean.
- Standard deviation across training seeds.
- Standard error or 95% confidence interval.
- Per-seed raw rows in the artifact directory.

For expensive 7B/8B training, a defensible compromise is:

1. Run 3 seeds for the final selected compute target.
2. Run 1 seed for broad Pareto exploration.
3. Clearly label sweep points that are single-seed.

## 6. Evaluation Harness Rules

There must be one evaluation entry point for all variants in a model family.

The evaluator must:

- Load variants from manifests, not hard-coded checkpoint paths.
- Refuse to run if required manifest fields are missing.
- Write JSONL per-example or per-task raw outputs when feasible.
- Write a normalized summary CSV.
- Record harness version, task versions, prompt settings, shot count, batch size, device, precision, and git commit.
- Evaluate routed models with the exact routing behavior used at inference.
- Verify that hooks are installed before routed evaluation.
- Verify that hooks are removed after routed evaluation.
- Emit measured active-layer and routing metrics during evaluation.
- Fail closed if the router is missing, inactive, or running full-depth unexpectedly.

The evaluator must not silently fall back from routed evaluation to dense evaluation.

## 7. Perplexity Standard

Perplexity must be computed with:

- Fixed dataset and split.
- Fixed sample count or full split.
- Fixed sequence length and stride.
- Correct label masking for padding.
- The same tokenizer for all variants.
- Hard routing at evaluation for DLR unless the paper explicitly claims soft routing.

Report:

- Negative log-likelihood.
- Perplexity.
- Number of evaluated tokens.
- Number of skipped/padded tokens.
- Active-layer statistics for routed variants.

If perplexity is poor but task accuracy is preserved, say that directly. Do not hide perplexity.

## 8. Compute Standard

DLR compute claims must distinguish:

- Structural compute reduction: layers or FLOPs skipped by the routing policy.
- Realized wall-clock speedup: measured tokens/sec or latency.

The paper must not imply wall-clock speedup from structural FLOP reduction unless it has been measured.

Report latency with:

- Hardware model.
- Driver/CUDA/PyTorch versions.
- Batch size.
- Sequence length.
- Generation or scoring mode.
- Warmup iterations.
- Timed iterations.
- Mean, p50, p95 latency if applicable.
- Tokens/sec.

## 9. Validity Gates

A run is invalid for paper claims if any of the following are true:

- The evaluation path differs across variants.
- A routed model is evaluated full-depth by accident.
- A dense model receives a different dataset, prompt template, or checkpoint rule.
- Any required manifest field is missing.
- The run has no git commit or code hash.
- Metric columns differ across variants.
- Router collapse is detected but not reported.
- Perplexity labels include padding tokens.
- Final benchmark results were used to choose the checkpoint.
- Historical results cannot be tied to code, command, and artifact files.

## 10. Standard Output Schema

Every evaluation run should write:

```text
artifacts/evals/<run_id>/
  manifest.json
  eval_summary.csv
  eval_summary.json
  task_results.json
  routing_metrics.csv
  perplexity.json
  latency.json
  environment.json
```

Minimum `eval_summary.csv` columns:

```text
run_id
model_family
variant
seed
checkpoint_id
task
shot_count
metric
value
stderr
num_examples
eval_harness
eval_harness_version
git_commit
config_hash
manifest_path
```

Minimum `routing_metrics.csv` columns:

```text
run_id
variant
seed
split
batch_index
mean_active_layers
std_active_layers
min_active_layers
max_active_layers
structural_flop_reduction_pct
router_entropy
collapse_flag
tokens_per_sec
peak_gpu_memory_gb
```

## 11. Paper Table Rules

Main result tables should include:

- Dense LoRA quality.
- Stochastic depth at matched compute.
- DLR at matched compute.
- Active layers or FLOP reduction.
- Perplexity.
- Mean and uncertainty.

Do not rank models by a single metric if another required metric fails badly. For DLR, quality and compute must be read together.

## 12. Immediate Implementation Target

Before launching any new large run, build this minimum machinery:

1. `configs/eval/qwen7b_main.yaml`
2. `configs/eval/llama8b_main.yaml`
3. A manifest validator.
4. A single evaluator that consumes manifests.
5. A smoke test proving dense, stochastic, and DLR variants produce the same output schema.

Only after this exists should the project resume expensive training.

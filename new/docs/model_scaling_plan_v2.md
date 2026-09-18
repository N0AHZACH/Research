# Model Scaling Plan v2

This supersedes `docs/model_scaling_plan.md`.

Goal: produce a NeurIPS-grade scaling story for Dynamic Layer Routing (DLR), not just a collection of isolated experiment wins.

## 1. Core Design

Use:

```text
1B -> 3B -> 4/5B -> 7/8B -> 11B -> 14/15B
```

At each scale:

- Use two different model families.
- Evaluate two DLR parameter levels.
- Match each DLR parameter level with a stochastic-depth baseline at the same expected compute.
- Keep dense LoRA as the full-depth quality reference.

This tests three things reviewers will care about:

- Does DLR work beyond one checkpoint?
- Does DLR work beyond one architecture/tokenizer family?
- Does DLR preserve the quality/compute tradeoff as model size increases?

## 2. Two Parameter Levels

Use two fixed routing/compute levels per model family:

| Level | Name | Target Skip | Target Active Compute | Purpose |
| --- | --- | ---: | ---: | --- |
| A | Conservative | 30% | 70% | Quality-preserving setting |
| B | Aggressive | 50% | 50% | Compute-saving setting |

Each parameter level needs:

- DLR at that target.
- Stochastic depth matched to that target.
- Dense LoRA reference shared across both levels.

## 3. Run Count Per Scale

For each scale:

```text
2 model families * (1 dense + 2 stochastic-depth + 2 DLR) = 10 core runs
```

With 3 training seeds:

```text
10 core runs * 3 seeds = 30 final runs per scale
```

Calibration runs can be single-seed and should be labeled exploratory.

## 4. Scale Matrix

| Scale | Model Family 1 | Model Family 2 | Role |
| --- | --- | --- | --- |
| 1B | TinyLlama-1.1B | second 1B-class open model | Fast debugging, ablations, pipeline validation |
| 3B | Qwen2.5-3B | OpenLLaMA-3B or similar | First non-toy scale |
| 4/5B | 4B-class open model | 5B-class open model | Intermediate scaling behavior |
| 7/8B | Qwen2.5-7B | Llama-3.1-8B | Main NeurIPS result scale |
| 11B | 11B-class open model A | 11B-class open model B | Scaling stress test |
| 14/15B | Qwen-class 14B | 15B-class open model | Upper-scale confirmation |

The exact model choices should be frozen before final experiments and recorded in configs/manifests with model revisions.

## 5. Per-Family Required Runs

For each model family at each scale:

| Variant | Level | Seeds | Required For Paper |
| --- | --- | ---: | --- |
| Dense LoRA | full depth | 3 | Yes |
| Stochastic depth | 30% skip | 3 | Yes |
| DLR | 30% skip | 3 | Yes |
| Stochastic depth | 50% skip | 3 | Yes |
| DLR | 50% skip | 3 | Yes |
| DLR without KD | selected level | 3 | Main ablation |
| Random/parameter-free router | selected level | 3 | Main ablation |

Run DLR without KD and random-router ablations first at 1B and 7/8B. Add them at other scales only if compute allows.

## 6. Parameter Policy

The project should not freely tune every hyperparameter at every scale.

Fixed across comparable runs:

- Dataset and preprocessing.
- LoRA target modules and rank.
- Optimizer and scheduler family.
- Evaluation harness.
- Router architecture unless the ablation says otherwise.
- KD weight unless testing KD.
- Temperature schedule unless calibration proves it fails.

Allowed calibration:

- Tune `compute_penalty` on one seed only to hit the target skip band.
- Adjust `always_keep_layers` by the proportional rule below.

Target bands:

```text
Level A: 25-35% measured skip
Level B: 45-55% measured skip
```

If a model cannot hit the target without collapse, record that failure. Do not silently move the goalposts.

## 7. Layer Scaling Rule

Use proportional anchor layers:

```text
always_keep_layers = max(2, round(0.15 * total_layers))
```

Examples:

| Total Layers | Always Keep | Routable Layers |
| ---: | ---: | ---: |
| 22 | 3 | 19 |
| 28 | 4 | 24 |
| 32 | 5 | 27 |
| 40 | 6 | 34 |
| 48 | 7 | 41 |

Any exception must be documented in the manifest.

## 8. NeurIPS Minimum Bar

The strongest realistic NeurIPS package is:

- Full 1B evidence across two model families.
- Full 3B evidence across two model families.
- Full 7/8B evidence across Qwen and Llama families.
- At least one larger-scale confirmation point at 11B or 14/15B.
- Matched stochastic-depth baselines at both parameter levels.
- Three seeds for main claims.
- Full evaluation according to `docs/NEURIPS_EVALUATION_STANDARD.md`.

The 7/8B scale should be the main table. The 1B and 3B scales establish development and trend. The 11B and 14/15B scales provide scaling credibility if compute allows.

## 9. Promotion Rule

Do not move to the next scale until both model families at the current scale have:

- Dense baseline complete.
- Stochastic depth at level A and B complete.
- DLR at level A and B complete.
- Matching evaluation outputs.
- Routing metrics.
- Perplexity.
- Tokens/sec or latency.
- Verified ledger entries.

This is deliberately strict. The research goal is publication, so incomplete scaling should not be allowed to create ambiguity.

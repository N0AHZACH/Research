# Research Restart Plan

This project is being restarted from first principles. Existing scripts, figures, logs, and manuscript text are useful background, but they should not be treated as trusted evidence until they pass the checks below.

## 1. Reset Principle

The restart should separate three things that are currently blended together:

1. The research claim we want to prove.
2. The experimental protocol needed to test that claim.
3. The implementation code used to run the protocol.

No result should enter the paper unless all three are explicit, reproducible, and traceable to a specific commit, command, configuration, and output artifact.

## 2. Core Claim

Dynamic Layer Routing should only be presented as successful if it improves the quality/compute Pareto frontier against strong controls.

The minimum defensible claim is:

> Token-level input-conditional layer routing can reduce effective transformer compute while preserving downstream quality better than input-agnostic stochastic depth under matched training, data, model, and evaluation conditions.

This implies the central comparison is not routed model vs. dense model alone. It is:

1. Dense LoRA baseline.
2. Stochastic depth baseline with matched expected compute.
3. Token-level Dynamic Layer Routing with matched training budget and evaluation protocol.

## 3. Stop Trusting Historical Results Until Audited

Treat the current repository as an archive, not a clean experimental package. Historical results may still be valuable, but each result needs provenance:

- Source script path.
- Git commit or file hash.
- Exact command.
- Model and tokenizer revision.
- Dataset name, split, revision, and preprocessing.
- Seed.
- Hardware.
- Precision and attention implementation.
- Training hyperparameters.
- Evaluation command.
- Raw log and metric file.

If any of these are missing, the result can be used for intuition only, not as paper evidence.

## 4. Clean Repository Structure

Use the current scripts as references, then converge on a small, boring structure:

```text
configs/
  qwen7b/
    dense.yaml
    stochastic_depth.yaml
    dlr_token.yaml
  llama8b/
    dense.yaml
    stochastic_depth.yaml
    dlr_token.yaml

src/
  dlr/
    reproducibility.py
    data.py
    models.py
    routing.py
    losses.py
    flop_accounting.py
    train.py
    evaluate.py
    manifests.py

experiments/
  train.py
  evaluate.py
  pareto_sweep.py
  analyze_routing.py

artifacts/
  runs/
  metrics/
  evals/
  figures/

docs/
  experiment_ledger.md
  restart_research_plan.md
```

The numbered experiment scripts can remain under `legacy/` or be preserved as historical references. New work should use shared code plus config files.

## 5. Implementation Standards

Before running expensive experiments, make the code boring and testable.

Required invariants:

- One shared seed function used by every entry point.
- One shared dataset loader and tokenizer path.
- One shared FLOP accounting implementation.
- One shared CSV/JSONL metric schema.
- One manifest per run.
- One evaluation harness path for all model variants.
- No hidden defaults inside experiment scripts when the value affects a paper result.

Required manifest fields:

- `run_id`
- `created_at`
- `git_commit`
- `command`
- `config_path`
- `model_name`
- `model_revision`
- `tokenizer_name`
- `dataset_name`
- `dataset_revision`
- `seed`
- `hardware`
- `precision`
- `attention_implementation`
- `trainable_parameter_count`
- `total_parameter_count`
- `expected_active_layers`
- `measured_active_layers`
- `checkpoint_path`
- `metrics_path`

## 6. Research Phases

### Phase A: Rebuild the Baseline

Goal: prove the dense training/evaluation pipeline is correct before adding dynamic routing.

Exit criteria:

- Dense model trains without ad hoc patches.
- Validation loss is stable across at least two seeds or two short smoke runs.
- Evaluation harness runs from the saved checkpoint.
- Metrics and manifest are written automatically.

### Phase B: Rebuild the Negative Control

Goal: implement stochastic depth as the honest input-agnostic compute-saving baseline.

Exit criteria:

- Expected compute is analytically logged.
- Measured active-layer rate is logged.
- The skip schedule is documented and matched to the DLR compute target.
- Evaluation uses the same harness as dense.

### Phase C: Rebuild DLR

Goal: add the smallest possible token-routing implementation after the controls are solid.

Exit criteria:

- Router gates have tested shape semantics: `[batch, seq_len, routable_layers]`.
- Gating preserves residual path correctness.
- Router entropy, per-layer activity, per-token active depth, and FLOP reduction are logged.
- Runs detect and flag collapse automatically.
- DLR is compared against stochastic depth at matched compute.

### Phase D: Pareto and Ablations

Goal: only after A-C pass, run sweeps.

Required ablations:

- DLR with KD.
- DLR without KD.
- DLR with learned router.
- Parameter-free or random matched-compute router.
- Multiple compute targets.

## 7. Minimum Tests Before Any Long Run

Add fast tests that do not require a large GPU:

- Reproducibility seeding test.
- FLOP accounting test on toy layer counts.
- Router output shape test.
- Gate loss test for all-keep, all-skip, and mixed gates.
- Hook lifecycle test proving hooks are removed after forward/evaluation.
- Manifest writing test.
- Metric schema test.

These tests will not prove the paper, but they will catch the kinds of implementation mistakes that can make a month of GPU results worthless.

## 8. Experiment Ledger

Every real run should be added to `docs/experiment_ledger.md` before its results are interpreted.

Suggested columns:

```text
run_id | date | model | method | config | seed | command | commit | artifact_dir | status | notes
```

Allowed statuses:

- `planned`
- `running`
- `completed_unverified`
- `verified`
- `invalid`

Only `verified` runs can be used in manuscript tables or figures.

## 9. First Two Weeks

Week 1:

1. Freeze the existing repo state and preserve historical outputs.
2. Create shared config and manifest machinery.
3. Rebuild dense training as the reference path.
4. Add the minimum test suite.

Week 2:

1. Rebuild stochastic depth on top of the same training path.
2. Rebuild DLR on top of the same training path.
3. Run tiny smoke experiments.
4. Compare logs, manifests, and evaluation outputs before scheduling large runs.

## 10. Decision Rule

Do not scale to expensive 7B/8B runs until the small pipeline has produced:

- One dense run.
- One stochastic depth run.
- One DLR run.
- Matching manifests.
- Matching metric schemas.
- Verified evaluation.
- A clear explanation of any quality/compute tradeoff.

The restart succeeds when the research process becomes boring enough that surprising results are scientifically meaningful instead of suspicious.

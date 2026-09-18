# Folder Structure

Use this folder for all new publishable research work. Treat `old/` as reference material only.

```text
new/
  README.md
  FOLDER_STRUCTURE.md

  configs/
    models/       # model-family definitions: qwen7b, llama8b, tinyllama1b
    train/        # training configs for dense/stochastic/DLR runs
    eval/         # evaluation manifests consumed by experiments/evaluate.py
    sweeps/       # parameter sweep configs, never one-off script constants

  src/
    dlr/          # reusable research code
      data.py
      models.py
      routing.py
      losses.py
      train_loop.py
      evaluation.py
      flop_accounting.py
      routing_metrics.py
      manifests.py
      reproducibility.py

  experiments/
    train.py             # train one model from one config
    evaluate.py          # evaluate one or more models from one manifest
    analyze_routing.py   # inspect token/layer routing behavior
    make_figures.py      # generate paper figures from artifacts

  artifacts/
    runs/         # training outputs and checkpoints
    evals/        # evaluator outputs
    analysis/     # routing/token analysis outputs
    audits/       # checkpoint and validity audit files

  figures/        # generated publication-ready figures
  tables/         # generated result tables
  paper/          # NeurIPS manuscript files
  tests/          # fast correctness tests
  scripts/        # helper scripts, setup, migration, cleanup
  docs/           # research plans, protocols, ledgers, audit notes
```

## Where New Code Goes

Put reusable logic in `new/src/dlr/`.

Put command-line entry points in `new/experiments/`.

Put run settings in `new/configs/`.

Put generated outputs in `new/artifacts/`, `new/figures/`, and `new/tables/`.

Put the manuscript in `new/paper/`.

## Standard Run Flow

```text
config -> train -> checkpoint -> eval manifest -> evaluate -> audit -> analysis -> figures/tables -> paper
```

## Naming Convention

Use names like:

```text
<model_family>_<parameter_category>_seed<seed>
```

Examples:

```text
qwen7b_dense_seed42
qwen7b_stochastic_30_seed42
qwen7b_dlr_30_seed42
llama8b_dlr_50_seed43
```

## Parameter Categories

```text
dense
stochastic_30
dlr_30
stochastic_50
dlr_50
```

Optional ablations:

```text
dlr_30_no_kd
dlr_50_no_kd
random_router_30
random_router_50
sequence_router_30
sequence_router_50
```

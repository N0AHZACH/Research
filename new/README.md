# New Research System

This folder is the clean rebuild area for the Dynamic Layer Routing publication effort.

Use `old/` as archive/reference material. New code, configs, manifests, evaluation outputs, and protocol docs should go here.

## Current Structure

```text
new/
  configs/
    eval/
      example_eval_manifest.json
  experiments/
    evaluate.py
  docs/
```

## Evaluation Framework

The unified evaluator is:

```text
new/experiments/evaluate.py
```

Validate a manifest without loading models:

```bash
python new/experiments/evaluate.py --manifest new/configs/eval/example_eval_manifest.json --dry-run
```

Run an evaluation:

```bash
python new/experiments/evaluate.py --manifest path/to/eval_manifest.json
```

The evaluator is the standard way to evaluate individual models so we can compare performance, routing behavior, perplexity, and compute metrics under one protocol.

## Rule

If a result is meant to support the NeurIPS paper, it must be produced through the unified evaluation framework or explicitly marked exploratory.

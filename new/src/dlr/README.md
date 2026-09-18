# `src/dlr`

Reusable code goes here. Experiment scripts should import from this package instead of duplicating logic.

Suggested files:

- `reproducibility.py`: seed control and deterministic settings.
- `data.py`: dataset loading and tokenization.
- `models.py`: model and tokenizer loading.
- `routing.py`: token router, hooks, and gated forward logic.
- `losses.py`: LM loss, KD loss, compute penalties.
- `train_loop.py`: shared training loop.
- `evaluation.py`: shared evaluation helpers.
- `flop_accounting.py`: active-layer and FLOP accounting.
- `routing_metrics.py`: entropy, collapse detection, per-layer activity.
- `manifests.py`: run manifest creation and validation.

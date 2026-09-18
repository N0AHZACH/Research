# `configs`

Config files are the source of truth for experiments.

- `models/`: reusable model-family definitions.
- `train/`: one training config per model/category/seed.
- `eval/`: manifests consumed by `new/experiments/evaluate.py`.
- `sweeps/`: sweep definitions for parameter calibration.

Nothing paper-critical should live only as a hard-coded script default.

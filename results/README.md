# Results Archive

This folder keeps generated artifacts out of the repository root while preserving the original filenames.

* `metrics/`: CSV logs from training, sweeps, benchmarks, and evaluation runs.
* `eval_summaries/`: JSON summaries emitted by evaluation harness scripts.
* `checkpoints/runs/`: Archived experiment output directories.
* `checkpoints/standalone/`: Standalone checkpoint files such as `.pt` snapshots.

New experiment scripts may still emit files into the repository root by default. Move completed runs here when preparing figures, manuscripts, or handoffs.

# Publication-Ready Patch Plan

This document outlines the systematic code changes required to elevate the dynamic routing codebase to NeurIPS/ICLR/ICML standards.

## 1. Immediate Fixes (Reproducibility & Validity)

**Add Universal Determinism (All `exp*.py` files)**
* **File:** `utils/reproducibility.py` (Create this file to centralize)
* **Function:** `set_seed(seed=42)`
* **Exact Modification:**
```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Required for true reproducibility in dynamic routing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: Alert user if TF32 is still active (as TF32 can introduce slight non-determinism)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True, warn_only=True)
```
* **Expected Impact:** Reviewers will outright reject non-reproducible routing research (as small variance in routing can look like massive architectural gains). This secures reproducibility.

**Fix `exp30_qwen7b_pareto_sweep.py` Data Logging**
* **File:** `exp30_qwen7b_pareto_sweep.py`
* **Function:** `train_one_penalty` & `main`
* **Exact Modification:** Inject proper Perplexity logging using `math.exp(val_loss)` and log Git Commit Hashes via a `subprocess.check_output(['git', 'rev-parse', 'HEAD'])`.
* **Expected Impact:** Validates exact code versions used for final manuscript plots.

## 2. Before 11B Scaling

**Implement Router Entropy & FLOP Tracking**
* **File:** `exp25_qwen7b_token_routing.py`
* **Function:** `compute_gate_loss()`
* **Exact Modification:**
```python
def compute_gate_loss(gates):
    # Log FLOP utilization explicitly
    per_layer_activity = gates.float().mean(dim=(0, 1))
    
    # Log routing entropy to prove the network isn't collapsing to static routing
    p = per_layer_activity
    eps = 1e-8
    layer_entropy = -(p * torch.log(p + eps) + (1-p) * torch.log(1-p + eps)).mean()
    
    return ..., layer_entropy
```
* **Expected Impact:** Reviewers will demand proof that "token routing" is actually dependent on tokens and hasn't degenerated into static layer dropping. Entropy metrics prove input-dependent variance.

**JSON Experiment Manifests**
* **File:** `exp25_qwen7b_token_routing.py`
* **Function:** `save_checkpoint`
* **Exact Modification:** Add a `manifest.json` dump that includes `COMPUTE_PENALTY`, `LR`, `EPOCHS`, `SEED`, `HARDWARE`, and exact `git` hash.
* **Expected Impact:** Critical for provenance when tracking scaling laws across 1B to 11B architectures.

## 3. Before Paper Submission

**Ablation Pipeline (KD off vs. KD on)**
* **File:** `run_experiments.py`
* **Function:** `main`
* **Exact Modification:** Create automated flags `--no-kd` to isolate the performance impact of Knowledge Distillation.
* **Expected Impact:** Proves whether the routing mechanism works fundamentally, or if it acts purely as a KD regularization artifact.

**Routing Visualization Hooks**
* **File:** `exp30_qwen7b_pareto_sweep.py`
* **Function:** `gated_forward`
* **Exact Modification:** Add an optional evaluation mode that saves `(input_ids, gates)` into an `.h5` or `.pt` file for interpretability analysis.
* **Expected Impact:** Generates the required "Token Type vs Routing Depth" heatmaps for the paper appendix.

## 4. Nice-to-Have Improvements

* **WandB / TensorBoard Integration:** Transition away from raw CSV files to `wandb` for massive multi-node scaling.
* **Flash Attention 2 Strict Enforcement:** Ensure `attn_implementation="flash_attention_2"` is universally strictly enforced on your 96GB setup rather than falling back to `sdpa`.
* **Parameter-Free Gumbel Router:** Compare against a naive thresholding router to prove the MLPs inside the Gumbel router actually add value.

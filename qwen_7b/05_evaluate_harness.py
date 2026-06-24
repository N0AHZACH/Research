import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
exp22_qwen7b_eval_harness.py - Phase 4: Qwen2.5-7B Evaluation Harness (RTX 6000 optimized)

Benchmarks all three model variants on zero-shot MMLU, GSM8K, and ARC-Challenge
to prove that reasoning capability is preserved when compute is dynamically reduced.

Model variants evaluated:
  1. baseline_lora    : exp1 static LoRA checkpoint (upper-bound accuracy)
  2. stochastic       : exp2 stochastic dropout checkpoint
  3. gumbel_router    : exp6 Gumbel-STE checkpoint (sequence-level routing)
  4. token_router     : exp10 token-level routing checkpoint (our main contribution)

Output:
  - exp22_qwen7b_eval_results_<timestamp>.csv  : full per-task accuracy table
  - exp22_qwen7b_eval_summary_<timestamp>.json : structured summary for manuscript

Requirements:
  pip install lm-eval>=0.4.0

Usage:
  python exp22_qwen7b_eval_harness.py
  python exp22_qwen7b_eval_harness.py --skip_baseline  # only eval gumbel
  python exp22_qwen7b_eval_harness.py --tasks mmlu     # only run MMLU
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import gc
import json
import math
import argparse
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
"""
exp22_qwen7b_eval_harness.py - Phase 4: Qwen2.5-7B Evaluation Harness (RTX 6000 optimized)

Benchmarks all three model variants on zero-shot MMLU, GSM8K, and ARC-Challenge
to prove that reasoning capability is preserved when compute is dynamically reduced.

Model variants evaluated:
  1. baseline_lora    : exp1 static LoRA checkpoint (upper-bound accuracy)
  2. stochastic       : exp2 stochastic dropout checkpoint
  3. gumbel_router    : exp6 Gumbel-STE checkpoint (sequence-level routing)
  4. token_router     : exp10 token-level routing checkpoint (our main contribution)

Output:
  - exp22_qwen7b_eval_results_<timestamp>.csv  : full per-task accuracy table
  - exp22_qwen7b_eval_summary_<timestamp>.json : structured summary for manuscript

Requirements:
  pip install lm-eval>=0.4.0

Usage:
  python exp22_qwen7b_eval_harness.py
  python exp22_qwen7b_eval_harness.py --skip_baseline  # only eval gumbel
  python exp22_qwen7b_eval_harness.py --tasks mmlu     # only run MMLU
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import gc
import json
import math
import argparse
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Phase 3.4 Evaluation Harness")
parser.add_argument("--tasks", nargs="+",
                    default=["mmlu", "gsm8k", "arc_challenge",
                             "hellaswag", "piqa", "boolq", "winogrande"],
                    help="Benchmark tasks to run (lm-eval task names)")
parser.add_argument("--num_fewshot", type=int, default=5,
                    help="Number of few-shot examples (0 = zero-shot, 5 = standard)")
# Tip: run with --num_fewshot 0 for zero-shot and --num_fewshot 5 for 5-shot.
# Both should be reported in the paper.
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples per task (for fast debug runs)")
parser.add_argument("--skip_baseline", action="store_true",
                    help="Skip evaluating baseline and stochastic variants (legacy)")
parser.add_argument("--skip_base", action="store_true",
                    help="Skip evaluating the base model")
parser.add_argument("--skip_baseline_lora", action="store_true",
                    help="Skip evaluating the baseline LoRA variant")
parser.add_argument("--skip_stochastic", action="store_true",
                    help="Skip evaluating the stochastic dropout variant")
parser.add_argument("--resume", action="store_true",
                    help="Automatically resume from the most recent checkpoint in the directory")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Eval batch size per GPU")
parser.add_argument(
    "--baseline_path",
    type=str,
    default=None,
    help="Path to exp1 baseline LoRA checkpoint dir (auto-detected if None)",
)
parser.add_argument(
    "--stochastic_path",
    type=str,
    default=None,
    help="Path to exp2 stochastic LoRA checkpoint dir (auto-detected if None)",
)
parser.add_argument(
    "--gumbel_path",
    type=str,
    default=None,
    help="Path to exp6 Gumbel router checkpoint dir (auto-detected if None)",
)
parser.add_argument(
    "--token_path",
    type=str,
    default=None,
    help="Path to exp10 token-level router checkpoint dir (auto-detected if None; falls back to exp9)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).parent
MODEL_ID     = "Qwen/Qwen2.5-7B"

def _latest_checkpoint(pattern: str) -> Optional[Path]:
    """
    Return the best_model/ directory from the most recent matching run,
    but ONLY if it contains adapter_config.json (i.e., was actually saved).
    Returns None if no valid checkpoint is found.
    """
    # Try all typical subdirectories saved by the training scripts
    for subdir in ["best_model", "final_model", "checkpoint_latest"]:
        candidates = sorted(RESEARCH_DIR.glob(f"{pattern}/{subdir}"), reverse=True)
        for c in candidates:
            if (c / "adapter_config.json").exists():
                return c
    
    # Second try: the run dir itself (no best_model subdir)
    dirs = sorted(RESEARCH_DIR.glob(pattern), reverse=True)
    for d in dirs:
        if (d / "adapter_config.json").exists():
            return d
    return None  # No valid checkpoint found

BASELINE_PATH   = Path(args.baseline_path)   if args.baseline_path   else _latest_checkpoint("exp23_qwen7b_baseline_output_*")
STOCHASTIC_PATH = Path(args.stochastic_path) if args.stochastic_path else _latest_checkpoint("exp24_qwen7b_stochastic_output_*")
# BUG-06 fix: removed duplicate _latest_checkpoint("exp25_qwen7b_token_output_*") call
TOKEN_PATH      = (Path(args.token_path) if args.token_path
                   else (_latest_checkpoint("exp25_qwen7b_token_output_*")
                         or _latest_checkpoint("exp25_qwen7b_token_legacy_output_*")))
# Random router (exp32) — required ablation to prove learning > random sparsity
RANDOM_ROUTER_PATH = _latest_checkpoint("exp32_qwen7b_random_output_*")

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT     = RESEARCH_DIR / f"exp22_qwen7b_eval_results_{TIMESTAMP}.csv"
JSON_OUT    = RESEARCH_DIR / f"exp22_qwen7b_eval_summary_{TIMESTAMP}.json"

# ---------------------------------------------------------------------------
# Check lm-eval is installed
# ---------------------------------------------------------------------------
try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
    print(f"[INFO] lm-eval version: {lm_eval.__version__}")
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("[WARNING] lm-eval not installed. Will run perplexity-only fallback.")
    print("          Install with: pip install lm-eval>=0.4.0")

# ---------------------------------------------------------------------------
# TokenLevelGumbelRouter (must be re-declared to load router_weights.pt)
# ---------------------------------------------------------------------------
import torch.nn as nn

class TokenLevelGumbelRouter(nn.Module):
    """
    Universal router architecture matching both exp6 and exp9.
    Accepts 2D [B, H] (sequence-level) or 3D [B, S, H] (token-level) hidden states.
    """
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )

    def forward(self, h: torch.Tensor, temperature: float = 0.5, hard: bool = False):
        h         = h.float()
        logits    = self.net(h)
        if self.training:
            logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            soft      = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
            return soft[..., 1].to(h.dtype)
        if hard:
            return (logits > 0).to(h.dtype)
        return torch.sigmoid(logits).to(h.dtype)
        g_model, g_tok = load_gumbel_checkpoint(skip_analysis_path)
        plot_per_layer_skip_rate(g_model, g_tok, skip_analysis_path)
        del g_model, g_tok

    torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Variant':<25} {'PPL':>8} {'Avg Layers':>12}", end="")
    task_metrics = ["mmlu_acc,none", "gsm8k_exact_match,strict-match", "arc_challenge_acc_norm,none"]
    short_names  = ["MMLU", "GSM8K", "ARC"]
    for sn in short_names:
        print(f" {sn:>8}", end="")
    print()
    print("  " + "-"*70)
    for res in all_results:
        ppl = res.get("perplexity_wikitext103", "—")
        al  = res.get("avg_active_layers", "—")
        print(f"  {res['variant']:<25} {ppl:>8} {str(al):>12}", end="")
        for metric, sn in zip(task_metrics, short_names):
            val = res.get(metric, "—")
            if isinstance(val, float):
                print(f" {val*100:>7.1f}%", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()
    print()

    print(f"\nPhase 3.4 evaluation complete!")
    print(f"  CSV  : {CSV_OUT}")
    print(f"  JSON : {JSON_OUT}")


def plot_per_layer_skip_rate(model, tokenizer, checkpoint_path, n_samples=200, device="cuda"):
    """
    Compute and plot the average per-layer skip rate for the Gumbel router.
    Shows which layers are most frequently skipped across a sample of inputs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TOTAL_LAYERS = len(model.base_model.model.model.layers)
    ROUTABLE     = TOTAL_LAYERS - ALWAYS_KEEP

    raw  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw  = raw.filter(lambda x: len(x["text"]) > 100).select(range(min(n_samples, len(raw))))

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        # Mask padding tokens in labels so CE loss ignores them
        labels = [list(ids) for ids in out["input_ids"]]
        for i, mask in enumerate(out["attention_mask"]):
            for j, m in enumerate(mask):
                if m == 0:
                    labels[i][j] = -100
        out["labels"] = labels
        return out

    ds = raw.map(tok, batched=True, remove_columns=raw.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    is_token_level = True # Always True for Qwen token router
    per_layer_skip = torch.zeros(ROUTABLE)
    total_batches  = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, gates = gated_forward_eval(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
                temperature=0.1, hard=True,
                is_token_level=is_token_level,
            )
            # gates: [B, L] or [B, S, L]. Average over all dims except last → per-layer skip rate
            skip = 1.0 - gates.float()
            per_layer_skip += skip.mean(dim=list(range(skip.dim() - 1))).cpu()
            total_batches  += 1

    per_layer_skip /= max(total_batches, 1)  # average skip rate per layer, [ROUTABLE]

    layer_indices = list(range(ALWAYS_KEEP + 1, TOTAL_LAYERS + 1))
    skip_pct      = (per_layer_skip * 100).tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#e74c3c" if s > 50 else "#3498db" for s in skip_pct]
    bars   = ax.bar(layer_indices, skip_pct, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=50, color="#888888", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
    ax.set_xlabel("Transformer Layer Index", fontsize=13)
    ax.set_ylabel("Skip Rate (%)", fontsize=13)
    ax.set_title("Per-Layer Skip Rate — Gumbel Router\n(Red = majority of inputs skip this layer)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(layer_indices)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, pct in zip(bars, skip_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = RESEARCH_DIR / "exp7_per_layer_skip_rate.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Per-layer skip rate → {out_path}")

if __name__ == "__main__":
    main()

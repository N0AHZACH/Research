"""
exp7_eval_harness.py - Phase 3.4: Rigorous Evaluation Harness

Benchmarks all three model variants on zero-shot MMLU, GSM8K, and ARC-Challenge
to prove that reasoning capability is preserved when compute is dynamically reduced.

Model variants evaluated:
  1. baseline_lora    : exp1 static LoRA checkpoint (upper-bound accuracy)
  2. stochastic       : exp2 stochastic dropout checkpoint
  3. gumbel_router    : exp6 Gumbel-STE checkpoint (sequence-level routing)
  4. token_router     : exp10 token-level routing checkpoint (our main contribution)

Output:
  - exp7_eval_results_<timestamp>.csv  : full per-task accuracy table
  - exp7_eval_summary_<timestamp>.json : structured summary for manuscript

Requirements:
  pip install lm-eval>=0.4.0

Usage:
  python exp7_eval_harness.py
  python exp7_eval_harness.py --skip_baseline  # only eval gumbel
  python exp7_eval_harness.py --tasks mmlu     # only run MMLU
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import json
import math
import argparse
import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Phase 3.4 Evaluation Harness")
parser.add_argument("--tasks", nargs="+",
                    default=["mmlu", "gsm8k", "arc_challenge"],
                    help="Benchmark tasks to run")
parser.add_argument("--num_fewshot", type=int, default=5,
                    help="Number of few-shot examples (0 = zero-shot)")
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples per task (for fast debug runs)")
parser.add_argument("--skip_baseline", action="store_true",
                    help="Skip evaluating baseline and stochastic variants")
parser.add_argument("--batch_size", type=int, default=1,
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
MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def _latest_checkpoint(pattern: str) -> Optional[Path]:
    """
    Return the best_model/ directory from the most recent matching run,
    but ONLY if it contains adapter_config.json (i.e., was actually saved).
    Returns None if no valid checkpoint is found.
    """
    # First try: <run_dir>/best_model/ with adapter_config.json
    candidates = sorted(RESEARCH_DIR.glob(f"{pattern}/best_model"), reverse=True)
    for c in candidates:
        if (c / "adapter_config.json").exists():
            return c
    # Second try: the run dir itself (no best_model subdir)
    dirs = sorted(RESEARCH_DIR.glob(pattern), reverse=True)
    for d in dirs:
        if (d / "adapter_config.json").exists():
            return d
    return None  # No valid checkpoint found

BASELINE_PATH   = Path(args.baseline_path)   if args.baseline_path   else _latest_checkpoint("exp1_baseline_output_*")
STOCHASTIC_PATH = Path(args.stochastic_path) if args.stochastic_path else _latest_checkpoint("exp2_stochastic_output_*")
GUMBEL_PATH     = Path(args.gumbel_path)     if args.gumbel_path     else _latest_checkpoint("exp6_gumbel_output_*")
# Token-level: prefer exp10 (fixed), fall back to exp9
TOKEN_PATH      = (Path(args.token_path) if args.token_path
                   else (_latest_checkpoint("exp10_token_output_*") or _latest_checkpoint("exp9_token_output_*")))

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT     = RESEARCH_DIR / f"exp7_eval_results_{TIMESTAMP}.csv"
JSON_OUT    = RESEARCH_DIR / f"exp7_eval_summary_{TIMESTAMP}.json"

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
# GumbelRouter (must be re-declared to load router_weights.pt)
# ---------------------------------------------------------------------------
import torch.nn as nn

class GumbelRouter(nn.Module):
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
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        soft      = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(h.dtype)

# ---------------------------------------------------------------------------
# Model loader helpers
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ALWAYS_KEEP = 4  # must match exp6 config


def load_base_model(device="cuda"):
    """Load frozen TinyLlama as a reference (no LoRA)."""
    print(f"  Loading base TinyLlama from hub...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_lora_checkpoint(checkpoint_path: Path, device="cuda"):
    """
    Load a plain LoRA checkpoint (exp1/exp2 style — no router).
    Validates that adapter_config.json exists before attempting to load.
    Raises FileNotFoundError if the checkpoint is missing/empty so the
    caller can fall back gracefully.
    """
    adapter_cfg = checkpoint_path / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in '{checkpoint_path}'. "
            f"This checkpoint was never saved by the training script."
        )
    print(f"  Loading LoRA checkpoint: {checkpoint_path}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model = model.merge_and_unload()   # merge LoRA into base weights for standard eval
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_gumbel_checkpoint(checkpoint_path: Path, device="cuda"):
    """
    Load exp6 Gumbel router checkpoint.
    Returns (model_with_router, tokenizer).
    The model still has the router attached; gated_forward() in eval_perplexity()
    handles the two-pass strategy at inference time.
    """
    print(f"  Loading Gumbel router checkpoint: {checkpoint_path}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))

    TOTAL_LAYERS    = len(model.base_model.model.model.layers)
    ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP
    model.router    = GumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to(device)

    router_weights = checkpoint_path / "router_weights.pt"
    if router_weights.exists():
        model.router.load_state_dict(torch.load(str(router_weights), map_location=device))
        print(f"    Router weights loaded from {router_weights}")
    else:
        print(f"    [WARNING] router_weights.pt not found at {router_weights}; router is random!")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ---------------------------------------------------------------------------
# Gated inference helpers (mirrors exp6 logic)
# ---------------------------------------------------------------------------
class StopForwardException(Exception): pass

class GatedForwardContext:
    """Minimal hook context for deterministic inference (no STE needed)."""
    def __init__(self):
        self.gates       = None
        self.handles     = []
        self.captured_h  = None

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def install_gate_hooks(self, layers, gates):
        self.gates = gates
        is_token_level = (gates.dim() == 3)
        for i, layer in enumerate(layers):
            idx = i
            def hook(module, input, output, layer_i=idx):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h    = output[0] if is_tuple else output
                if is_token_level:
                    gate = self.gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
                else:
                    gate = self.gates[:, layer_i].view(-1, 1, 1).to(h.dtype)
                gated_h = gate * h + (1.0 - gate) * residual
                return (gated_h,) + output[1:] if is_tuple else gated_h
            self.handles.append(layer.register_forward_hook(hook))


def gated_forward_eval(model, input_ids, attention_mask=None, labels=None,
                       temperature=0.5, hard=True, is_token_level=False):
    """
    Gated forward for perplexity eval with early stopping.
    hard=True  → binary gates, matching training-time behaviour (fair PPL comparison).
    """
    transformer = model.base_model.model.model
    all_layers  = transformer.layers
    ctx = GatedForwardContext()

    def early_stop_hook(module, input, output):
        hidden_state = output[0] if isinstance(output, tuple) else output
        if is_token_level:
            ctx.captured_h = hidden_state.detach().float()
        else:
            ctx.captured_h = hidden_state.detach().float().mean(dim=1)
        raise StopForwardException()

    handle = all_layers[ALWAYS_KEEP - 1].register_forward_hook(early_stop_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForwardException:
        pass
    finally:
        handle.remove()

    pooled_h = ctx.captured_h.to(next(model.router.parameters()).device)
    gates    = model.router(pooled_h, temperature=temperature, hard=hard)

    ctx.install_gate_hooks(all_layers[ALWAYS_KEEP:], gates)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    ctx.remove_hooks()

    return outputs, gates

# ---------------------------------------------------------------------------
# Perplexity evaluation on Wikitext-103 validation split
# ---------------------------------------------------------------------------
from datasets import load_dataset
from torch.utils.data import DataLoader

PERPLEXITY_SAMPLES = 500  # fast but representative
MAX_LENGTH         = 512

def eval_perplexity(model, tokenizer, is_gumbel: bool, is_token_level: bool = False, device="cuda"):
    """
    Compute validation perplexity on Wikitext-103.
    Uses gated_forward_eval() for the Gumbel variant, standard forward otherwise.
    Returns: (perplexity, avg_active_layers_or_None)
    """
    print("    Computing Wikitext-103 validation perplexity...")
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100)
    raw = raw.select(range(min(PERPLEXITY_SAMPLES, len(raw))))

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        # Mask padding tokens in labels so CE loss ignores them
        labels = [list(ids) for ids in out["input_ids"]]
        for i, mask in enumerate(out["attention_mask"]):
            for j, m in enumerate(mask):
                if m == 0:
                    labels[i][j] = -100
        out["labels"] = labels
        return out

    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    active_layer_counts = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if is_gumbel:
                outputs, gates = gated_forward_eval(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    temperature=0.5,  # final annealed temperature
                    hard=True,        # binary gates → fair comparison with static baselines
                    is_token_level=is_token_level,
                )
                active_layer_counts.append(
                    gates.float().mean(dim=list(range(gates.dim() - 1))).sum().item() + ALWAYS_KEEP
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                n_batches  += 1

    avg_loss   = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss)
    avg_layers = (sum(active_layer_counts) / len(active_layer_counts)
                  if active_layer_counts else None)
    if avg_layers is not None:
        print(f"    Perplexity: {perplexity:.2f}  |  Avg active layers: {avg_layers:.1f}")
    else:
        print(f"    Perplexity: {perplexity:.2f}")
    return perplexity, avg_layers

# ---------------------------------------------------------------------------
# lm-eval benchmarking
# ---------------------------------------------------------------------------
def run_lm_eval(model, tokenizer, tasks, num_fewshot, batch_size, limit=None):
    """
    Wraps lm-eval's HFLM evaluator.
    Returns dict: {task_name: {"acc": float, "acc_stderr": float, ...}}
    """
    # Silence the annoying token indices warning from transformers
    tokenizer.model_max_length = 100000
    
    lm = HFLM(
        pretrained=model, 
        tokenizer=tokenizer, 
        batch_size=batch_size,
        max_length=2048,
        truncation=True
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )
    task_results = {}
    for task, metrics in results["results"].items():
        task_results[task] = {
            k: v for k, v in metrics.items()
            if not k.startswith("alias")
        }
    return task_results

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_variant(name, load_fn, checkpoint, is_gumbel=False):
    """
    Evaluate a single model variant.
    If the checkpoint is missing or invalid (e.g. exp1/exp2 that never saved),
    falls back to evaluating the pretrained base TinyLlama and notes it in results.
    Returns a dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")

    use_base_fallback = False
    if checkpoint is None:
        print(f"  [FALLBACK] No saved checkpoint found for '{name}'.")
        print(f"             exp1/exp2 scripts did not call model.save_pretrained().")
        print(f"             Evaluating pretrained base TinyLlama as a proxy.")
        use_base_fallback = True

    try:
        if use_base_fallback:
            raise FileNotFoundError("No checkpoint — using base fallback")
        model, tokenizer = load_fn(Path(checkpoint))
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"  [FALLBACK] Could not load checkpoint ({e}).")
        print(f"             Using pretrained base TinyLlama as proxy for '{name}'.")
        use_base_fallback = True
        is_gumbel = False
        model, tokenizer = load_base_model()

    is_token_level = "token" in str(checkpoint).lower() if checkpoint else False
    # Perplexity (always run)
    ppl, avg_layers = eval_perplexity(model, tokenizer, is_gumbel=is_gumbel, is_token_level=is_token_level)

    result = {
        "variant": name,
        "checkpoint": "base_tinyllama_fallback" if use_base_fallback else str(checkpoint),
        "status": "base_fallback" if use_base_fallback else "ok",
        "note": ("Checkpoint not saved by training script; base model used as proxy."
                 if use_base_fallback else ""),
        "perplexity_wikitext103": round(ppl, 4),
        "avg_active_layers": round(avg_layers, 2) if avg_layers is not None else "N/A",
    }

    # Benchmark tasks via lm-eval
    if LM_EVAL_AVAILABLE and args.tasks:
        print(f"    Running lm-eval tasks: {args.tasks} ({args.num_fewshot}-shot) ...")
        if is_gumbel:
            # For MMLU/GSM8K/ARC benchmark tasks, we need a clean merged model
            # for lm-eval (HFLM expects a standard HuggingFace model interface).
            #
            # CRITICAL: We must fully delete the old PeftModel + router and clear
            # the GPU before reloading. On 8GB VRAM, merge_and_unload() on the
            # existing model leaves fragmented GPU memory (old PeftModel tensors,
            # router weights, hook residuals) that causes CUDA illegal memory
            # access errors during autoregressive generation.
            print(f"    Freeing GPU memory before lm-eval...")
            del model
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import gc; gc.collect()

            # Reload a fresh copy and merge LoRA — clean GPU state
            print(f"    Reloading fresh merged model for lm-eval...")
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
            )
            peft_model = PeftModel.from_pretrained(base, str(checkpoint))
            eval_model = peft_model.merge_and_unload()
            del peft_model, base
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            eval_model = model

        task_results = run_lm_eval(
            eval_model, tokenizer,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        for task, metrics in task_results.items():
            for metric, value in metrics.items():
                result[f"{task}_{metric}"] = value

        # Clean up the eval model
        del eval_model
        if not is_gumbel and 'model' in locals():
            del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        if not LM_EVAL_AVAILABLE:
            result["lm_eval_note"] = "lm-eval not installed; perplexity only"
        # Clean up the model if we're not passing it to lm-eval
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"\n  Results for [{name}]:")
    for k, v in result.items():
        if k not in ("variant", "checkpoint", "status"):
            print(f"    {k}: {v}")

    return result


def main():
    print("\n" + "="*70)
    print("  Phase 3.4 Evaluation Harness")
    print(f"  Tasks: {args.tasks} | Few-shot: {args.num_fewshot} | Limit: {args.limit}")
    print("="*70)

    all_results = []

    # ── 1. Base TinyLlama (no fine-tuning, reference point) ──────────────────
    print("\n[1/4] Base TinyLlama (reference — no fine-tuning)")
    base_model, base_tokenizer = load_base_model()
    ppl_base, _ = eval_perplexity(base_model, base_tokenizer, is_gumbel=False)
    base_result = {
        "variant": "base_tinyllama",
        "checkpoint": MODEL_ID,
        "status": "ok",
        "perplexity_wikitext103": round(ppl_base, 4),
        "avg_active_layers": "N/A",
    }
    if LM_EVAL_AVAILABLE and args.tasks:
        print("    Running lm-eval on base model...")
        task_results = run_lm_eval(
            base_model, base_tokenizer,
            tasks=args.tasks, num_fewshot=args.num_fewshot,
            batch_size=args.batch_size, limit=args.limit,
        )
        for task, metrics in task_results.items():
            for metric, value in metrics.items():
                base_result[f"{task}_{metric}"] = value
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    all_results.append(base_result)

    if not args.skip_baseline:
        # ── 2. Baseline LoRA (exp1) ───────────────────────────────────────────
        print("\n[2/4] Baseline LoRA (exp1 — static, all layers)")
        r = evaluate_variant("baseline_lora", load_lora_checkpoint, BASELINE_PATH, is_gumbel=False)
        all_results.append(r)
        torch.cuda.empty_cache()

        # ── 3. Stochastic Dropout (exp2) ──────────────────────────────────────
        print("\n[3/4] Stochastic Dropout (exp2 — negative control)")
        r = evaluate_variant("stochastic_dropout", load_lora_checkpoint, STOCHASTIC_PATH, is_gumbel=False)
        all_results.append(r)
        torch.cuda.empty_cache()

    # ── 4. Gumbel Router (exp6 — sequence-level routing) ────────────────────
    print("\n[4/5] Gumbel Router (exp6 — Gumbel-STE + KD, sequence-level)")
    r = evaluate_variant("gumbel_router", load_gumbel_checkpoint, GUMBEL_PATH, is_gumbel=True)
    all_results.append(r)

    # ── 5. Token-Level Router (exp10/exp9 — our main contribution) ─────────
    if TOKEN_PATH is not None:
        exp_label = "exp10" if "exp10" in str(TOKEN_PATH) else "exp9"
        print(f"\n[5/5] Token-Level Router ({exp_label} — token-level routing)")
        r = evaluate_variant("token_router", load_gumbel_checkpoint, TOKEN_PATH, is_gumbel=True)
        all_results.append(r)
    else:
        print("\n[5/5] Token-Level Router — skipped (no checkpoint found)")

    # ── BONUS: Per-Layer Skip Analysis ────────────────────────────────────────
    # Prefer token-level model for skip analysis (more interesting per-token patterns)
    skip_analysis_path = TOKEN_PATH if (TOKEN_PATH and Path(TOKEN_PATH).exists()) else GUMBEL_PATH
    if skip_analysis_path is not None and Path(skip_analysis_path).exists():
        print(f"\n[BONUS] Computing per-layer skip rate from {skip_analysis_path}...")
        g_model, g_tok = load_gumbel_checkpoint(skip_analysis_path)
        plot_per_layer_skip_rate(g_model, g_tok, skip_analysis_path)
        del g_model, g_tok

    torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Saving results...")

    # Collect all unique column keys
    all_keys = []
    for res in all_results:
        for k in res:
            if k not in all_keys:
                all_keys.append(k)

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for res in all_results:
            writer.writerow({k: res.get(k, "") for k in all_keys})
    print(f"  CSV  → {CSV_OUT}")

    summary = {
        "timestamp": TIMESTAMP,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "checkpoints": {
            "baseline": str(BASELINE_PATH),
            "stochastic": str(STOCHASTIC_PATH),
            "gumbel": str(GUMBEL_PATH),
            "token_router": str(TOKEN_PATH),
        },
        "results": all_results,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON → {JSON_OUT}")

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

    raw  = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
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

    is_token_level = "token" in str(checkpoint_path).lower()
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
                temperature=0.5, hard=True,
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

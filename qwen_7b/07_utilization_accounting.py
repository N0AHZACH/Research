import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP33: TOKEN-LAYER UTILIZATION ACCOUNTING
#
# SCIENTIFIC PURPOSE:
#   Produce exact, defensible token-layer utilization statistics for the paper.
#   This script is post-hoc analysis — it takes trained checkpoints and runs
#   full evaluation datasets through them, recording:
#
#     executed_token_layer_pairs = ∑_{b,s,l} gate[b,s,l]
#     total_token_layer_pairs    = B × S × L_routable
#     utilization_pct            = 100 × executed / total
#
#   This is the PRIMARY metric for the DLR paper. Unlike FLOP numbers (which
#   are projected/theoretical under perfect token-packing assumptions), the
#   token-layer pair count is EXACT and directly observable.
#
# OUTPUTS:
#   utilization_accounting.csv — per-checkpoint utilization summary
#   utilization_per_layer.csv  — per-layer utilization breakdown
#
# METHODOLOGY NOTE:
#   All variants (Dense, Stochastic, Random, DLR) are evaluated at IDENTICAL
#   sequence lengths, batch sizes, and dataset splits to ensure comparability.
#   Dense and Stochastic baselines have utilization_pct = 100% (no skipping)
#   or per their EVAL_FULL_DEPTH setting.
# =============================================================================

import csv
import gc
import json
import math
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_ID            = "Qwen/Qwen2.5-7B"
MAX_LENGTH          = 512
EVAL_SAMPLES        = 2000          # larger than training eval for rigorous accounting
BATCH_SIZE          = 8
ALWAYS_KEEP         = 4             # must match exp25 config
RESEARCH_DIR        = Path(__file__).parent

OUTPUT_CSV          = RESEARCH_DIR / "utilization_accounting.csv"
PER_LAYER_CSV       = RESEARCH_DIR / "utilization_per_layer.csv"


# ==============================================================================
# Router re-declaration (for checkpoint loading)
# ==============================================================================
import torch.nn as nn

class TokenLevelGumbelRouter(nn.Module):
    """Re-declared from exp25 to enable loading router_weights.pt."""
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4), nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )
    def forward(self, h: torch.Tensor, temperature: float = 0.0, hard: bool = True):
        logits = self.net(h.float())
        if temperature == 0.0 or hard:
            return (logits > 0).float()
        return torch.sigmoid(logits / temperature)


class RandomBernoulliRouter(nn.Module):
    """Re-declared from exp32."""
    def __init__(self, p_active: float, num_layers: int):
        super().__init__()
        self.register_buffer("p_act", torch.tensor(float(p_active)))
        self.num_layers = num_layers
    def forward(self, h_seq: torch.Tensor, temperature: float = 0.0, hard: bool = True):
        B, S, _ = h_seq.shape
        return torch.bernoulli(self.p_act.expand(B, S, self.num_layers)).to(h_seq.dtype)


# ==============================================================================
# Helpers
# ==============================================================================
class StopForwardException(Exception):
    pass


def get_attn_impl():
    try:
        import flash_attn; return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model_and_router(checkpoint_path: Path, router_type: str, p_active: float = 0.6):
    """Load a PEFT checkpoint and reconstruct the router if applicable."""
    ATTN = get_attn_impl()
    print(f"  Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=ATTN, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token

    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        layers = model.model.layers

    total_layers    = len(layers)
    routable_layers = total_layers - ALWAYS_KEEP

    if router_type == "dlr":
        router = TokenLevelGumbelRouter(model.config.hidden_size, routable_layers).to("cuda")
        weights_path = checkpoint_path / "router_weights.pt"
        if weights_path.exists():
            router.load_state_dict(torch.load(str(weights_path), map_location="cuda"))
            print(f"  DLR router weights loaded from {weights_path}")
        else:
            print(f"  [WARNING] router_weights.pt not found — router is UNTRAINED!")
        router.eval()

    elif router_type == "random":
        # Load p_active from config if saved
        cfg_path = checkpoint_path / "random_router_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                p_active = json.load(f).get("p_active", p_active)
        router = RandomBernoulliRouter(p_active=p_active, num_layers=routable_layers).to("cuda")
        router.eval()
        print(f"  Random router: p_active={p_active:.3f}")

    else:
        router = None

    return model, tokenizer, layers, router, total_layers, routable_layers


def get_gates(model, layers, router, batch, always_keep: int):
    """Run two-pass gated forward, return gates [B, S, L] or None."""
    if router is None:
        return None

    captured = {}

    class StopFwd(Exception): pass

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()
        raise StopFwd()

    handle = layers[always_keep - 1].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    except StopFwd:
        pass
    finally:
        handle.remove()

    h_seq  = captured["h"].to("cuda")
    with torch.no_grad():
        gates = router(h_seq, temperature=0.0, hard=True)   # [B, S, L_routable]
    return gates.float()


def evaluate_utilization(model, tokenizer, layers, router, loader, total_layers, routable_layers):
    """
    Compute exact token-layer utilization over a full dataset.

    Returns dict with all primary and secondary metrics.
    """
    model.eval()
    always_keep = total_layers - routable_layers

    exec_pairs_total   = 0
    total_pairs_total  = 0
    per_layer_exec     = None    # [L] cumulative executed pairs per layer
    per_layer_total    = None

    # Routing entropy tracking
    entropies          = []

    # Batch-level active layer counts (for avg_active_layers)
    avg_layers_list    = []

    n_batches          = 0
    start_time         = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            gates  = get_gates(model, layers, router, batch, always_keep)

            if gates is not None:
                B, S, L = gates.shape
                # Exact pair counting
                exec_pairs_total  += int(gates.sum().long().item())
                total_pairs_total += B * S * L

                # Per-layer accounting
                if per_layer_exec is None:
                    per_layer_exec  = torch.zeros(L)
                    per_layer_total = torch.zeros(L)
                per_layer_exec  += gates.sum(dim=(0, 1)).cpu()
                per_layer_total += torch.full((L,), B * S, dtype=torch.float32)

                # Routing entropy
                mean_g = gates.mean().item()
                eps    = 1e-8
                p_     = max(eps, min(1.0 - eps, mean_g))
                h_     = -(p_ * math.log(p_) + (1 - p_) * math.log(1 - p_))
                entropies.append(h_)

                # Avg active layers
                avg_layers_list.append(
                    always_keep + gates.mean(dim=(0, 1)).sum().item()
                )
            else:
                # Dense or stochastic (full depth at eval)
                exec_pairs_total  += total_layers * (batch["input_ids"].shape[0] * batch["input_ids"].shape[1])
                total_pairs_total += total_layers * (batch["input_ids"].shape[0] * batch["input_ids"].shape[1])
                avg_layers_list.append(float(total_layers))

            n_batches += 1

    elapsed = time.perf_counter() - start_time

    utilization_pct  = 100.0 * exec_pairs_total / max(1, total_pairs_total)
    avg_active_layers = sum(avg_layers_list) / max(1, len(avg_layers_list))
    mean_entropy      = sum(entropies) / max(1, len(entropies)) if entropies else 0.0

    per_layer_util = None
    if per_layer_exec is not None:
        per_layer_util = (per_layer_exec / per_layer_total.clamp(min=1)).tolist()

    # Gate variance (approximate from per-layer rates)
    if per_layer_util:
        util_arr    = torch.tensor(per_layer_util)
        gate_var    = float(util_arr.var().item())
        util_var    = float(util_arr.var().item())
    else:
        gate_var = util_var = 0.0

    return {
        "executed_token_layer_pairs":    exec_pairs_total,
        "total_token_layer_pairs":       total_pairs_total,
        "utilization_pct":               round(utilization_pct, 4),
        "avg_active_layers":             round(avg_active_layers, 4),
        "mean_routing_entropy":          round(mean_entropy, 6),
        "gate_variance":                 round(gate_var, 6),
        "utilization_variance":          round(util_var, 6),
        "n_batches":                     n_batches,
        "eval_time_sec":                 round(elapsed, 2),
        "per_layer_utilization":         per_layer_util,   # list of per-layer rates
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP33: Exact Token-Layer Utilization Accounting")
    parser.add_argument("--dlr_path",        type=str, default=None, help="exp25 DLR checkpoint")
    parser.add_argument("--random_path",     type=str, default=None, help="exp32 Random router checkpoint")
    parser.add_argument("--baseline_path",   type=str, default=None, help="exp23 Dense baseline checkpoint")
    parser.add_argument("--stochastic_path", type=str, default=None, help="exp24 Stochastic checkpoint")
    parser.add_argument("--p_active",        type=float, default=0.6, help="Random router p_active (default 0.6)")
    parser.add_argument("--eval_samples",    type=int, default=EVAL_SAMPLES)
    parser.add_argument("--batch_size",      type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EXP33: EXACT TOKEN-LAYER UTILIZATION ACCOUNTING")
    print("  Primary metric for DLR paper (zero assumptions, exact counts)")
    print("=" * 70 + "\n")

    # Build eval dataset
    print("Loading eval dataset (Wikitext-103 validation)...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100).select(range(min(args.eval_samples, len(raw))))

    tokenizer_tmp = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer_tmp.pad_token = tokenizer_tmp.eos_token

    def tokenize(batch):
        out = tokenizer_tmp(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        return out

    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  {len(ds)} eval samples | batch_size={args.batch_size} | {len(loader)} batches")

    experiments = []
    if args.dlr_path:
        experiments.append(("DLR (exp25)", Path(args.dlr_path), "dlr"))
    if args.random_path:
        experiments.append(("Random Router (exp32)", Path(args.random_path), "random"))
    if args.baseline_path:
        experiments.append(("Dense Baseline (exp23)", Path(args.baseline_path), "dense"))
    if args.stochastic_path:
        experiments.append(("Stochastic (exp24)", Path(args.stochastic_path), "stochastic"))

    if not experiments:
        print("[ERROR] No checkpoint paths provided. Use --dlr_path, --random_path, etc.")
        print("Example:")
        print("  python exp33_qwen7b_utilization_accounting.py \\")
        print("    --dlr_path exp25_qwen7b_token_output_20241201_120000/best_model \\")
        print("    --random_path exp32_qwen7b_random_output_20241201_130000/best_model")
        sys.exit(1)

    summary_rows    = []
    per_layer_rows  = []

    for name, ckpt_path, router_type in experiments:
        print(f"\n{'─' * 60}")
        print(f"  Variant: {name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'─' * 60}")

        try:
            model, tokenizer, layers, router, total_layers, routable_layers = \
                load_model_and_router(ckpt_path, router_type, p_active=args.p_active)
        except Exception as e:
            print(f"  [ERROR] Failed to load {name}: {e}")
            continue

        metrics = evaluate_utilization(model, tokenizer, layers, router,
                                       loader, total_layers, routable_layers)

        per_layer_util = metrics.pop("per_layer_utilization", None)

        row = {"variant": name, "router_type": router_type, **metrics}
        summary_rows.append(row)

        if per_layer_util is not None:
            for li, rate in enumerate(per_layer_util):
                per_layer_rows.append({
                    "variant": name,
                    "router_type": router_type,
                    "layer_index": ALWAYS_KEEP + li,
                    "routable_layer_index": li,
                    "utilization_rate": round(rate, 6),
                    "utilization_pct": round(rate * 100, 4),
                })

        print(f"\n  ✓ {name}:")
        print(f"    Utilization: {metrics['utilization_pct']:.2f}%")
        print(f"    Exec pairs:  {metrics['executed_token_layer_pairs']:,} / {metrics['total_token_layer_pairs']:,}")
        print(f"    Avg layers:  {metrics['avg_active_layers']:.2f} / {total_layers}")
        print(f"    Entropy:     {metrics['mean_routing_entropy']:.4f} nats")

        del model, router
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # Write summary CSV
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[DONE] Utilization summary → {OUTPUT_CSV}")

    # Write per-layer CSV
    if per_layer_rows:
        pl_keys = ["variant", "router_type", "layer_index", "routable_layer_index",
                   "utilization_rate", "utilization_pct"]
        with open(PER_LAYER_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pl_keys)
            w.writeheader()
            w.writerows(per_layer_rows)
        print(f"[DONE] Per-layer utilization → {PER_LAYER_CSV}")

    print("\n" + "=" * 70)
    print("  PRIMARY METRIC SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Util%':>8} {'AvgLayers':>10} {'Entropy':>10}")
    print("─" * 62)
    for row in summary_rows:
        print(f"  {row['variant']:<28} {row['utilization_pct']:>8.2f} "
              f"{row['avg_active_layers']:>10.2f} {row['mean_routing_entropy']:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

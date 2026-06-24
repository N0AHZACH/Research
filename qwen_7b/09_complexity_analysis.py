import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP35: INPUT COMPLEXITY ANALYSIS — Router Adaptivity
#
# SCIENTIFIC PURPOSE:
#   Prove that DLR routes adaptively based on input difficulty.
#   The central claim: "Easy inputs → fewer active token-layer pairs,
#                       Hard inputs → more active token-layer pairs"
#
#   This is the empirical proof of input-conditional routing.
#   Without this experiment, reviewers can claim the router is "static"
#   (always routing the same layers, just at a fixed rate).
#
# EXPERIMENT DESIGN:
#   For a trained DLR checkpoint, evaluate on Wikitext-103 validation set.
#   For each sample, compute:
#     1. Routing entropy H(gates) — uncertainty of routing decisions
#     2. Utilization pct — fraction of token-layer pairs executed
#     3. Avg active layers — mean active layers per token
#   Then bin samples by input complexity measures:
#     A. Sequence length (short = easy, long = complex)
#     B. Base model perplexity (low ppl = predictable, high = complex)
#     C. Token rarity (mean unigram rank — rare tokens = harder)
#
# EXPECTED RESULT (required for paper):
#   Positive correlation between input complexity and routing utilization.
#   Pearson r > 0 for all three complexity measures.
#
# OUTPUTS:
#   complexity_analysis.csv   — per-sample metrics
#   complexity_by_length.png  — utilization vs. sequence length (binned)
#   complexity_by_ppl.png     — utilization vs. base-model perplexity (binned)
#   complexity_by_rarity.png  — utilization vs. token rarity (binned)
#   complexity_correlations.csv — Pearson r values
# =============================================================================

import csv
import gc
import json
import math
import os
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

RESEARCH_DIR = Path(__file__).parent
MODEL_ID     = "Qwen/Qwen2.5-7B"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4
EVAL_SAMPLES = 2000   # sufficient for binning


# ==============================================================================
# Router re-declaration
# ==============================================================================
import torch.nn as nn

class TokenLevelGumbelRouter(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4), nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )
    def forward(self, h, temperature=0.0, hard=True):
        logits = self.net(h.float())
        return (logits > 0).float() if (temperature == 0.0 or hard) else torch.sigmoid(logits)


# ==============================================================================
# Analysis utilities
# ==============================================================================
def get_attn_impl():
    try:
        import flash_attn; return "flash_attention_2"
    except ImportError:
        return "sdpa"


def compute_routing_entropy(gates: torch.Tensor) -> float:
    """H(mean_gate) in nats."""
    p   = gates.mean().item()
    eps = 1e-8
    p   = max(eps, min(1.0 - eps, p))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def build_unigram_ranks(tokenizer, train_texts: List[str], top_k: int = 50000) -> Dict[int, int]:
    """Build unigram frequency table → rank mapping from training texts."""
    from collections import Counter
    freq = Counter()
    for text in train_texts[:5000]:   # subsample for speed
        ids = tokenizer.encode(text, add_special_tokens=False)
        freq.update(ids)
    sorted_tokens = [tok for tok, _ in freq.most_common()]
    return {tok: rank for rank, tok in enumerate(sorted_tokens)}


def mean_token_rarity(input_ids: torch.Tensor, rank_map: Dict[int, int], vocab_size: int) -> float:
    """
    Mean unigram rank of tokens in sequence. Higher = rarer.
    Unknown tokens (not in rank_map) assigned vocab_size (max rarity).
    """
    ids    = input_ids.cpu().tolist()
    ranks  = [rank_map.get(tok, vocab_size) for tok in ids if tok != -100]
    return sum(ranks) / len(ranks) if ranks else 0.0


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    n    = len(xs)
    if n < 2: return 0.0
    mx   = sum(xs) / n
    my   = sum(ys) / n
    cov  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx   = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy   = math.sqrt(sum((y - my) ** 2 for y in ys))
    denom = sx * sy
    return cov / denom if denom > 1e-8 else 0.0


def bin_and_average(xs: List[float], ys: List[float], n_bins: int = 10):
    """Bin xs into n_bins equal-frequency bins, average ys per bin."""
    if not xs: return [], [], []
    sorted_pairs = sorted(zip(xs, ys))
    bin_size = max(1, len(sorted_pairs) // n_bins)
    bin_x_centers, bin_y_means, bin_y_stds = [], [], []
    for i in range(0, len(sorted_pairs), bin_size):
        chunk = sorted_pairs[i:i + bin_size]
        if len(chunk) < 2: continue
        cx, cy = zip(*chunk)
        bin_x_centers.append(statistics.mean(cx))
        bin_y_means.append(statistics.mean(cy))
        bin_y_stds.append(statistics.stdev(cy) if len(cy) > 1 else 0.0)
    return bin_x_centers, bin_y_means, bin_y_stds


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP35: Router Adaptivity vs. Input Complexity")
    parser.add_argument("--dlr_path",     type=str, required=True,
                        help="Path to trained exp25 DLR checkpoint (best_model/)")
    parser.add_argument("--eval_samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--n_bins",       type=int, default=10,
                        help="Number of bins for complexity analysis")
    parser.add_argument("--no_plots",     action="store_true",
                        help="Skip matplotlib plotting (for headless servers)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EXP35: INPUT COMPLEXITY ANALYSIS — Router Adaptivity")
    print("  Proving: Hard inputs → more active token-layer pairs")
    print("=" * 70 + "\n")

    ATTN  = get_attn_impl()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    # Load DLR checkpoint
    ckpt = Path(args.dlr_path)
    print(f"Loading DLR checkpoint: {ckpt}")
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation=ATTN, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(ckpt))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    tokenizer.pad_token = tokenizer.eos_token

    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        layers = model.model.layers

    total_layers    = len(layers)
    routable_layers = total_layers - ALWAYS_KEEP

    router = TokenLevelGumbelRouter(model.config.hidden_size, routable_layers).to("cuda")
    wt_path = ckpt / "router_weights.pt"
    if wt_path.exists():
        router.load_state_dict(torch.load(str(wt_path), map_location="cuda"))
        print("  Router weights loaded.")
    else:
        print("  [WARNING] No router_weights.pt found. Using random weights — analysis INVALID.")
    router.eval()

    # Load eval dataset
    print("\nLoading dataset...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100).select(range(min(args.eval_samples, len(raw))))

    # Build unigram rank map from training data (for token rarity)
    print("Building unigram frequency table...")
    train_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rank_map  = build_unigram_ranks(tokenizer, [x["text"] for x in train_raw.select(range(5000))])
    vocab_size = tokenizer.vocab_size
    del train_raw

    # Per-sample analysis
    sample_metrics = []
    print(f"\nAnalyzing {len(raw)} samples...")

    for idx, example in enumerate(raw):
        text = example["text"]
        enc  = tokenizer(
            text, truncation=True, padding="max_length", max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")

        # Actual sequence length (non-padding tokens)
        actual_len = int(attention_mask.sum().item())

        # ── Routing pass ─────────────────────────────────────────────────────
        captured = {}

        class StopFwd(Exception): pass

        def capture_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h.detach().float()
            raise StopFwd()

        handle = layers[ALWAYS_KEEP - 1].register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        except StopFwd:
            pass
        finally:
            handle.remove()

        h_seq = captured["h"].to("cuda")  # [1, S, H]
        with torch.no_grad():
            gates = router(h_seq, temperature=0.0, hard=True).float()  # [1, S, L]

        utilization = gates.mean().item() * 100.0
        entropy     = compute_routing_entropy(gates)
        avg_layers  = ALWAYS_KEEP + gates.mean(dim=(0, 1)).sum().item()

        # ── Base model perplexity (proxy for input difficulty) ───────────────
        # Run standard forward to get CE loss (no routing)
        with torch.no_grad():
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            # Use model with adapters disabled for base model PPL
            with model.disable_adapter():
                out_base = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        base_ppl = math.exp(min(out_base.loss.item(), 300))

        # ── Token rarity ────────────────────────────────────────────────────
        valid_ids = input_ids[0][attention_mask[0] == 1]
        rarity    = mean_token_rarity(valid_ids, rank_map, vocab_size)

        sample_metrics.append({
            "sample_idx":    idx,
            "seq_len":       actual_len,
            "base_ppl":      round(base_ppl, 4),
            "token_rarity":  round(rarity, 2),
            "utilization_pct": round(utilization, 4),
            "routing_entropy": round(entropy, 6),
            "avg_active_layers": round(avg_layers, 4),
        })

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(raw)}] "
                  f"seq_len={actual_len} base_ppl={base_ppl:.1f} "
                  f"util={utilization:.1f}% entropy={entropy:.3f}")

    # Write per-sample CSV
    csv_path = RESEARCH_DIR / "complexity_analysis.csv"
    keys     = list(sample_metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sample_metrics)
    print(f"\n[DONE] Per-sample metrics → {csv_path}")

    # Compute correlations
    seqlens   = [r["seq_len"]        for r in sample_metrics]
    ppls      = [r["base_ppl"]       for r in sample_metrics]
    rarities  = [r["token_rarity"]   for r in sample_metrics]
    utils     = [r["utilization_pct"] for r in sample_metrics]
    entropies = [r["routing_entropy"] for r in sample_metrics]

    corr_seqlen  = pearson_r(seqlens,  utils)
    corr_ppl     = pearson_r(ppls,     utils)
    corr_rarity  = pearson_r(rarities, utils)
    corr_ent_ppl = pearson_r(ppls,     entropies)

    corr_path = RESEARCH_DIR / "complexity_correlations.csv"
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["complexity_measure", "target_metric", "pearson_r", "interpretation"])
        w.writerow(["seq_len",     "utilization_pct", f"{corr_seqlen:.4f}",
                    "positive = longer sequences route more layers"])
        w.writerow(["base_ppl",    "utilization_pct", f"{corr_ppl:.4f}",
                    "positive = harder inputs route more layers"])
        w.writerow(["token_rarity","utilization_pct", f"{corr_rarity:.4f}",
                    "positive = rarer tokens route more layers"])
        w.writerow(["base_ppl",    "routing_entropy", f"{corr_ent_ppl:.4f}",
                    "positive = harder inputs have higher routing uncertainty"])

    print(f"[DONE] Correlations → {corr_path}")
    print(f"\n{'─' * 50}")
    print(f"  CORRELATION RESULTS (Pearson r):")
    print(f"  seq_len    → utilization:  r = {corr_seqlen:+.4f}")
    print(f"  base_ppl   → utilization:  r = {corr_ppl:+.4f}")
    print(f"  token_rarity→ utilization: r = {corr_rarity:+.4f}")
    print(f"  base_ppl   → entropy:      r = {corr_ent_ppl:+.4f}")
    print(f"{'─' * 50}")
    print()

    # Verdict
    print("  VERDICT:")
    if corr_ppl > 0.1:
        print("  ✓ Router IS adaptive: harder inputs (higher PPL) → more compute")
        print("  ✓ This supports the DLR central claim.")
    elif corr_ppl > 0.0:
        print("  ⚠ Weak adaptivity signal (r > 0 but < 0.1).")
        print("  ⚠ Consider stronger compute penalty or longer training.")
    else:
        print("  ✗ Router NOT adaptive: no correlation between PPL and utilization.")
        print("  ✗ Central DLR claim is NOT supported. Investigate router training.")

    # Generate plots
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import AutoMinorLocator

            def make_bin_plot(xs, ys, xlabel, ylabel, title, out_path, n_bins=args.n_bins):
                bx, by, bstd = bin_and_average(xs, ys, n_bins=n_bins)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.errorbar(bx, by, yerr=bstd, fmt="o-", color="#2563eb",
                            linewidth=2, markersize=6, capsize=4, alpha=0.9)
                ax.fill_between(bx,
                                [m - s for m, s in zip(by, bstd)],
                                [m + s for m, s in zip(by, bstd)],
                                alpha=0.15, color="#2563eb")
                r = pearson_r(xs, ys)
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"{title}\n(Pearson r = {r:+.3f})", fontsize=13, pad=10)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                plt.tight_layout()
                plt.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"  [PLOT] {out_path}")

            make_bin_plot(seqlens,  utils, "Sequence Length (tokens)",
                          "Token-Layer Utilization (%)",
                          "Routing Adaptivity: Seq Length vs. Utilization",
                          RESEARCH_DIR / "complexity_by_length.png")

            make_bin_plot(ppls, utils, "Base Model Perplexity (log scale)",
                          "Token-Layer Utilization (%)",
                          "Routing Adaptivity: Input Difficulty (PPL) vs. Utilization",
                          RESEARCH_DIR / "complexity_by_ppl.png")

            make_bin_plot(rarities, utils, "Mean Token Rarity (unigram rank)",
                          "Token-Layer Utilization (%)",
                          "Routing Adaptivity: Token Rarity vs. Utilization",
                          RESEARCH_DIR / "complexity_by_rarity.png")

            # Combined 3-panel figure for paper
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            data_list = [
                (seqlens,  utils, "Sequence Length", f"r={corr_seqlen:+.3f}"),
                (ppls,     utils, "Base Model PPL",  f"r={corr_ppl:+.3f}"),
                (rarities, utils, "Token Rarity",    f"r={corr_rarity:+.3f}"),
            ]
            for ax, (xs, ys, xlabel, r_str) in zip(axes, data_list):
                bx, by, bstd = bin_and_average(xs, ys, n_bins=args.n_bins)
                ax.errorbar(bx, by, yerr=bstd, fmt="o-", color="#2563eb",
                            linewidth=2, markersize=5, capsize=3)
                ax.fill_between(bx,
                                [m - s for m, s in zip(by, bstd)],
                                [m + s for m, s in zip(by, bstd)],
                                alpha=0.15, color="#2563eb")
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel("Utilization (%)", fontsize=11)
                ax.set_title(r_str, fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.4)

            fig.suptitle("DLR Router Adaptivity: Utilization vs. Input Complexity",
                         fontsize=14, y=1.02)
            plt.tight_layout()
            panel_path = RESEARCH_DIR / "complexity_analysis_panel.png"
            plt.savefig(panel_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [PLOT] Combined panel → {panel_path}")

        except ImportError:
            print("  [SKIP] matplotlib not available. Install with: pip install matplotlib")

    print("\n" + "=" * 70)
    print("  EXP35 complete.")
    print(f"  CSV:  {csv_path}")
    print(f"  Corr: {corr_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

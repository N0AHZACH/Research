import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP34: THROUGHPUT BENCHMARKING
#
# SCIENTIFIC PURPOSE:
#   Measure wall-clock inference throughput for all 4 variants:
#     A. Dense Full-Depth Baseline  (exp23)
#     B. Stochastic Depth Baseline  (exp24)
#     C. Random Token Router        (exp32)
#     D. DLR — Dynamic Layer Routing(exp25)
#
#   Outputs:
#     compute_results.csv — tokens/sec, samples/sec, step_time, wall_clock_time
#
#   NOTE: Due to the hook-based implementation, actual runtime speedup for
#   B/C/D may be NEGATIVE (hook overhead adds latency vs. plain forward pass).
#   This is an expected and important result to report honestly. The paper's
#   compute efficiency claims rest on token-layer utilization (exp33), not
#   throughput. Throughput improvement requires custom token-packing kernels.
#
# METHODOLOGY:
#   - Warmup: 5 forward passes (GPU warmup, cache priming)
#   - Timed: 20 forward passes
#   - All variants: batch_size=1 for fair latency comparison
#   - Also run batch_size=MAX for throughput comparison
#   - CUDA events for precise GPU timing (not wall-clock which includes Python overhead)
# =============================================================================

import csv
import gc
import json
import math
import os
import time
import argparse
import statistics
from pathlib import Path
from typing import Optional, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_ID     = "Qwen/Qwen2.5-7B"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4
RESEARCH_DIR = Path(__file__).parent
OUTPUT_CSV   = RESEARCH_DIR / "compute_results.csv"

WARMUP_STEPS = 5
TIMED_STEPS  = 20


# ==============================================================================
# Router re-declarations
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


class RandomBernoulliRouter(nn.Module):
    def __init__(self, p_active: float, num_layers: int):
        super().__init__()
        self.register_buffer("p_act", torch.tensor(float(p_active)))
        self.num_layers = num_layers
    def forward(self, h_seq, temperature=0.0, hard=True):
        B, S, _ = h_seq.shape
        return torch.bernoulli(self.p_act.expand(B, S, self.num_layers)).to(h_seq.dtype)


# ==============================================================================
# Gated forward helpers
# ==============================================================================
class StopForwardException(Exception):
    pass


def install_gate_hooks(layers, gates):
    handles = []
    for i, layer in enumerate(layers):
        def hook(module, input, output, li=i):
            residual  = input[0]
            is_tuple  = isinstance(output, tuple)
            h         = output[0] if is_tuple else output
            gate      = gates[:, :, li].unsqueeze(-1).to(h.dtype)
            gated_h   = gate * h + (1.0 - gate) * residual
            return (gated_h,) + output[1:] if is_tuple else gated_h
        handles.append(layer.register_forward_hook(hook))
    return handles


def gated_inference(model, layers, router, input_ids, attention_mask, always_keep):
    captured = {}

    class StopFwd(Exception): pass

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()
        raise StopFwd()

    handle = layers[always_keep - 1].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except StopFwd:
        pass
    finally:
        handle.remove()

    h_seq  = captured["h"].to("cuda")
    gates  = router(h_seq, temperature=0.0, hard=True)
    handles = install_gate_hooks(layers[always_keep:], gates)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return out


def dense_inference(model, input_ids, attention_mask):
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask)


# ==============================================================================
# Timing utilities
# ==============================================================================
def time_forward_passes(
    forward_fn,
    input_ids,
    attention_mask,
    warmup: int = WARMUP_STEPS,
    timed: int = TIMED_STEPS,
) -> Dict[str, float]:
    """
    Measure GPU time for a forward function using CUDA events.

    Returns:
        dict with mean_ms, std_ms, min_ms, max_ms, tokens_per_sec, samples_per_sec
    """
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    # Warmup (not timed — GPU ramp up, autotuning)
    for _ in range(warmup):
        _ = forward_fn(input_ids, attention_mask)
        torch.cuda.synchronize()

    timings_ms = []
    start_evt  = torch.cuda.Event(enable_timing=True)
    end_evt    = torch.cuda.Event(enable_timing=True)

    for _ in range(timed):
        start_evt.record()
        _ = forward_fn(input_ids, attention_mask)
        end_evt.record()
        torch.cuda.synchronize()
        timings_ms.append(start_evt.elapsed_time(end_evt))  # milliseconds

    mean_ms  = statistics.mean(timings_ms)
    std_ms   = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    min_ms   = min(timings_ms)
    max_ms   = max(timings_ms)

    tokens_per_s  = total_tokens / (mean_ms / 1000.0)
    samples_per_s = batch_size   / (mean_ms / 1000.0)

    return {
        "mean_step_time_ms":  round(mean_ms,  3),
        "std_step_time_ms":   round(std_ms,   3),
        "min_step_time_ms":   round(min_ms,   3),
        "max_step_time_ms":   round(max_ms,   3),
        "tokens_per_sec":     round(tokens_per_s,  1),
        "samples_per_sec":    round(samples_per_s, 3),
        "batch_size":         batch_size,
        "seq_len":            seq_len,
        "total_tokens":       total_tokens,
    }


# ==============================================================================
# Model loading
# ==============================================================================
def get_attn_impl():
    try:
        import flash_attn; return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_for_benchmark(ckpt_path: Path, router_type: str, p_active: float = 0.6):
    ATTN  = get_attn_impl()
    vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    print(f"  Loading {router_type}: {ckpt_path}")
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation=ATTN, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(ckpt_path))
    model.eval()

    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        layers = model.model.layers

    total_layers    = len(layers)
    routable_layers = total_layers - ALWAYS_KEEP
    router          = None

    if router_type == "dlr":
        router = TokenLevelGumbelRouter(model.config.hidden_size, routable_layers).to("cuda")
        wt_path = ckpt_path / "router_weights.pt"
        if wt_path.exists():
            router.load_state_dict(torch.load(str(wt_path), map_location="cuda"))
        router.eval()

    elif router_type == "random":
        cfg_path = ckpt_path / "random_router_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                p_active = json.load(f).get("p_active", p_active)
        router = RandomBernoulliRouter(p_active=p_active, num_layers=routable_layers).to("cuda")
        router.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, layers, router, total_layers, routable_layers


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP34: Throughput Benchmarking")
    parser.add_argument("--dlr_path",        type=str, default=None)
    parser.add_argument("--random_path",     type=str, default=None)
    parser.add_argument("--baseline_path",   type=str, default=None)
    parser.add_argument("--stochastic_path", type=str, default=None)
    parser.add_argument("--p_active",        type=float, default=0.6)
    parser.add_argument("--batch_sizes",     type=int, nargs="+", default=[1, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq_len",         type=int, default=MAX_LENGTH)
    parser.add_argument("--warmup",          type=int, default=WARMUP_STEPS)
    parser.add_argument("--timed",           type=int, default=TIMED_STEPS)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EXP34: THROUGHPUT BENCHMARKING")
    print(f"  Batch sizes: {args.batch_sizes} | Seq len: {args.seq_len}")
    print(f"  Warmup: {args.warmup} | Timed: {args.timed}")
    print()
    print("  NOTE: Hook overhead typically negates routing speedup in PyTorch.")
    print("  Real FLOP savings require token-packing kernels (not implemented).")
    print("  These numbers quantify the overhead vs. projected savings tradeoff.")
    print("=" * 70 + "\n")

    experiments = []
    if args.baseline_path:
        experiments.append(("Dense (exp23)",           Path(args.baseline_path),   "dense"))
    if args.stochastic_path:
        experiments.append(("Stochastic (exp24)",      Path(args.stochastic_path), "stochastic"))
    if args.random_path:
        experiments.append(("Random Router (exp32)",   Path(args.random_path),     "random"))
    if args.dlr_path:
        experiments.append(("DLR (exp25)",             Path(args.dlr_path),        "dlr"))

    if not experiments:
        print("[ERROR] Provide at least one checkpoint path.")
        print("  python exp34_qwen7b_throughput_benchmark.py --dlr_path path/to/checkpoint")
        import sys; sys.exit(1)

    all_rows = []

    for name, ckpt_path, router_type in experiments:
        print(f"\n{'─' * 60}")
        print(f"  Benchmarking: {name}")
        print(f"{'─' * 60}")

        try:
            model, tokenizer, layers, router, total_layers, routable_layers = \
                load_for_benchmark(ckpt_path, router_type, p_active=args.p_active)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for bs in args.batch_sizes:
            print(f"  Batch size = {bs}, seq_len = {args.seq_len}...")

            # Dummy input
            input_ids      = torch.randint(100, 50000, (bs, args.seq_len), device="cuda")
            attention_mask = torch.ones_like(input_ids)

            if router is not None:
                def fwd(ids, mask):
                    return gated_inference(model, layers, router, ids, mask, ALWAYS_KEEP)
            else:
                def fwd(ids, mask):
                    return dense_inference(model, ids, mask)

            try:
                timings = time_forward_passes(
                    fwd, input_ids, attention_mask,
                    warmup=args.warmup, timed=args.timed
                )
            except torch.cuda.OutOfMemoryError:
                print(f"    [OOM] batch_size={bs} too large for this variant. Skipping.")
                continue

            row = {
                "variant":         name,
                "router_type":     router_type,
                **timings,
                "warmup_steps":    args.warmup,
                "timed_steps":     args.timed,
                "total_layers":    total_layers,
                "routable_layers": routable_layers,
                "always_keep":     ALWAYS_KEEP,
            }
            all_rows.append(row)

            print(f"    mean={timings['mean_step_time_ms']:.1f}ms  "
                  f"±{timings['std_step_time_ms']:.1f}ms  "
                  f"tok/s={timings['tokens_per_sec']:.0f}")

        del model, router
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSV
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[DONE] Throughput results → {OUTPUT_CSV}")

    # Summary table
    print("\n" + "=" * 70)
    print("  THROUGHPUT SUMMARY (batch_size=1, seq_len={})".format(args.seq_len))
    print("=" * 70)
    print(f"{'Variant':<30} {'ms/step':>8} {'tok/s':>10} {'vs Dense':>10}")
    print("─" * 62)
    dense_tps = None
    for row in all_rows:
        if row["batch_size"] == 1:
            if row["router_type"] == "dense":
                dense_tps = row["tokens_per_sec"]
            rel = (f"{row['tokens_per_sec']/dense_tps:.2f}x" if dense_tps else "N/A")
            print(f"  {row['variant']:<28} {row['mean_step_time_ms']:>8.1f} "
                  f"{row['tokens_per_sec']:>10.0f} {rel:>10}")


if __name__ == "__main__":
    main()

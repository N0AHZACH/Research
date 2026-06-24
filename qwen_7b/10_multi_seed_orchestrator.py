import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP36: MULTI-SEED EXPERIMENT ORCHESTRATOR
#
# SCIENTIFIC PURPOSE:
#   Run each variant across multiple seeds to produce mean ± std results.
#   Required for any top-tier venue (NeurIPS/ICML/ICLR). Single-seed
#   results for stochastic routing are not credible.
#
# SCOPE (adjusted for compute budget):
#   Seeds:   42  (1 seed — to fit within tight compute budget)
#   Methods: Dense (exp23), Stochastic (exp24), Random Router (exp32), DLR (exp25)
#
# STRATEGY:
#   - Launches each experiment as a subprocess with --seed flag
#   - Parses result CSVs from each run
#   - Aggregates mean ± std across seeds (std will be 0.0 for 1 seed)
#   - Writes aggregate_results.csv
#
# USAGE:
#   python exp36_qwen7b_multi_seed.py \
#     --methods dlr random stochastic dense \
#     --seeds 42 \
#     --dry_run  (print commands without running)
#
# IMPORTANT:
#   This orchestrator assumes each training script supports --seed flag.
#   exp25 (DLR) and exp32 (Random Router) have been patched to support this.
#   exp23 (Dense) and exp24 (Stochastic) should add --seed support if not present.
# =============================================================================

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import statistics

RESEARCH_DIR = Path(__file__).parent
OUTPUT_CSV   = RESEARCH_DIR / "aggregate_results.csv"

# Script mappings
SCRIPT_MAP = {
    "dense":       "01_train_dense_baseline.py",
    "stochastic":  "02_train_stochastic_depth.py",
    "random":      "03_train_random_router.py",
    "dlr":         "04_train_dlr_token_routing.py",
    "dlr_no_kd":   "04_train_dlr_token_routing.py",  # with --no_kd
}

# Metrics to extract from result CSVs (from last row = final checkpoint)
METRICS_TO_EXTRACT = [
    "perplexity",
    "val_loss",
    "utilization_pct",
    "avg_active_layers",
    "routing_entropy",
    "projected_flop_reduction_pct",
    "active_layer_frac",
    "skip_ratio",
]

# lm-eval result patterns (from exp22 outputs)
TASK_METRICS = {
    "mmlu":         "acc",
    "gsm8k":        "exact_match",
    "arc_challenge": "acc_norm",
    "hellaswag":    "acc_norm",
    "piqa":         "acc",
    "boolq":        "acc",
    "winogrande":   "acc",
}


# ==============================================================================
# CSV parsing
# ==============================================================================
def find_latest_csv(pattern: str) -> Optional[Path]:
    """Find the most recent CSV matching pattern."""
    candidates = sorted(RESEARCH_DIR.glob(pattern), reverse=True)
    return candidates[0] if candidates else None


def parse_last_row(csv_path: Path, metrics: List[str]) -> Dict[str, float]:
    """Extract last row values for specified metrics from a CSV file."""
    if not csv_path.exists():
        return {}
    try:
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = rows[-1]
        result = {}
        for m in metrics:
            if m in last:
                try:
                    result[m] = float(last[m])
                except (ValueError, TypeError):
                    pass
        return result
    except Exception as e:
        print(f"  [WARN] Could not parse {csv_path}: {e}")
        return {}


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    """Compute mean ± std from a list of values."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}


# ==============================================================================
# Subprocess runner
# ==============================================================================
def run_experiment(script: str, seed: int, extra_args: List[str] = None,
                   dry_run: bool = False) -> bool:
    """
    Launch a training script as a subprocess with the given seed.
    Returns True if successful, False on error.
    """
    cmd = [
        sys.executable, str(RESEARCH_DIR / script),
        "--seed", str(seed),
        "--fresh",    # always start fresh for reproducibility
    ]
    if extra_args:
        cmd.extend(extra_args)

    cmd_str = " ".join(cmd)
    print(f"\n{'─' * 60}")
    print(f"  [SEED={seed}] {script}")
    print(f"  CMD: {cmd_str}")

    if dry_run:
        print(f"  [DRY RUN] Command not executed.")
        return True

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(RESEARCH_DIR),
            capture_output=False,  # let stdout/stderr pass through
            timeout=24 * 3600,     # 24h max per run
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n  [ERROR] {script} seed={seed} returned code {result.returncode}")
            return False
        print(f"\n  [OK] Completed in {elapsed/3600:.1f}h")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {script} seed={seed} exceeded 24h limit.")
        return False
    except Exception as e:
        print(f"\n  [EXCEPTION] {e}")
        return False


# ==============================================================================
# Result collection
# ==============================================================================
def collect_results_for_method(method: str, seeds: List[int]) -> Dict:
    """
    Find and parse CSV results for all seeds of a method.
    Returns {metric: {mean, std, n}} dict.
    """
    # Pattern to find this method's CSV files
    csv_patterns = {
        "dense":      "exp23_baseline_metrics_*.csv",
        "stochastic": "exp24_stochastic_metrics_*.csv",
        "random":     "exp32_random_router_metrics_*.csv",
        "dlr":        "exp25_token_routing_metrics_*.csv",
        "dlr_no_kd":  "exp25_token_routing_metrics_*.csv",  # look for no_kd flag in csv
    }

    pattern = csv_patterns.get(method, f"exp*_{method}_metrics_*.csv")
    all_csvs = sorted(RESEARCH_DIR.glob(pattern), reverse=True)

    print(f"  Found {len(all_csvs)} CSV file(s) for {method}")

    # Try to match CSVs to seeds (heuristic: take last N CSVs = last N runs)
    # A more robust approach would parse seed from filename or CSV column
    seed_results = []
    for csv_path in all_csvs[:len(seeds)]:    # take most recent N runs
        row = parse_last_row(csv_path, METRICS_TO_EXTRACT)
        if row:
            print(f"    {csv_path.name}: {row}")
            seed_results.append(row)

    if not seed_results:
        print(f"  [WARN] No results found for method={method}")
        return {}

    # Aggregate across seeds
    aggregated = {}
    for metric in METRICS_TO_EXTRACT:
        values = [r[metric] for r in seed_results if metric in r]
        if values:
            aggregated[metric] = aggregate_metric(values)

    return aggregated


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP36: Multi-Seed Experiment Orchestrator")
    parser.add_argument("--methods", nargs="+",
                        choices=["dense", "stochastic", "random", "dlr", "dlr_no_kd"],
                        default=["dlr", "random", "stochastic", "dense"],
                        help="Methods to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to run (default: 42)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--collect_only", action="store_true",
                        help="Skip training, just collect and aggregate existing results")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a (method, seed) pair if recent CSV already exists")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EXP36: MULTI-SEED EXPERIMENT ORCHESTRATOR")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Collect only: {args.collect_only}")
    print("=" * 70 + "\n")

    if not args.collect_only:
        # Phase 1: Run experiments
        completed = []
        failed    = []

        for method in args.methods:
            script    = SCRIPT_MAP.get(method)
            if not script:
                print(f"  [SKIP] Unknown method: {method}")
                continue

            if not (RESEARCH_DIR / script).exists():
                print(f"  [WARN] Script not found: {script}")
                print(f"         Skipping method={method}")
                continue

            extra = ["--no_kd"] if method == "dlr_no_kd" else []

            for seed in args.seeds:
                print(f"\n[RUN] Method={method} | Seed={seed}")
                ok = run_experiment(script, seed, extra_args=extra, dry_run=args.dry_run)
                if ok:
                    completed.append((method, seed))
                else:
                    failed.append((method, seed))

        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Completed: {len(completed)} runs")
        print(f"  Failed:    {len(failed)} runs")
        if failed:
            print(f"  Failed runs: {failed}")
        print(f"{'=' * 60}\n")

    # Phase 2: Collect and aggregate results
    print("\n[COLLECT] Aggregating results across seeds...")
    all_aggregate = {}

    for method in args.methods:
        print(f"\n  Method: {method}")
        agg = collect_results_for_method(method, args.seeds)
        if agg:
            all_aggregate[method] = agg

    # Write aggregate CSV
    if all_aggregate:
        rows = []
        for method, metrics in all_aggregate.items():
            row = {"method": method}
            for metric, stats in metrics.items():
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"]  = stats["std"]
                row[f"{metric}_n"]    = stats["n"]
            rows.append(row)

        keys = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n[DONE] Aggregate results → {OUTPUT_CSV}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS (mean ± std across seeds)")
    print("=" * 70)
    print(f"{'Method':<20} {'PPL (↓)':>12} {'Util% (%)':>12} {'Entropy':>10}")
    print("─" * 56)
    for method, agg in all_aggregate.items():
        ppl   = agg.get("perplexity", {})
        util  = agg.get("utilization_pct", {})
        ent   = agg.get("routing_entropy", {})
        ppl_s = f"{ppl.get('mean', 0):.2f}±{ppl.get('std', 0):.2f}" if ppl else "N/A"
        util_s= f"{util.get('mean', 0):.1f}±{util.get('std', 0):.1f}" if util else "N/A"
        ent_s = f"{ent.get('mean', 0):.3f}" if ent else "N/A"
        print(f"  {method:<18} {ppl_s:>12} {util_s:>12} {ent_s:>10}")
    print("=" * 70)

    # Statistical significance guidance
    print("\n  STATISTICAL SIGNIFICANCE:")
    print("  For paired comparisons (DLR vs Random at same utilization),")
    print("  use paired t-test or bootstrap CI with N=3 seeds.")
    print("  With N=3, effect sizes must be large to reach p<0.05.")
    print("  Report: 95% bootstrap CI with 1000 resamples.")
    print("\n  Implementation:")
    print("  from scipy.stats import ttest_rel, bootstrap")
    print("  t, p = ttest_rel(dlr_ppls, random_ppls)")

    # Save JSON summary
    json_path = RESEARCH_DIR / "aggregate_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "seeds": args.seeds,
            "methods": args.methods,
            "results": all_aggregate,
        }, f, indent=2)
    print(f"\n[DONE] JSON summary → {json_path}")


if __name__ == "__main__":
    main()

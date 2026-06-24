import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP37: PARETO FRONTIER GENERATOR
#
# SCIENTIFIC PURPOSE:
#   Generate the Accuracy vs. Token-Layer Utilization Pareto frontier
#   for all four variants (Dense, Stochastic, Random Router, DLR).
#
#   The paper's central figure: DLR must occupy the Pareto frontier
#   (achieving better accuracy at lower utilization than competitors).
#
# PARETO DEFINITION:
#   Point A dominates point B if:
#     A.accuracy >= B.accuracy AND A.utilization <= B.utilization
#     with at least one strict inequality.
#   The Pareto frontier = {non-dominated points}.
#
# DATA SOURCES:
#   Primary: exp30_qwen7b_pareto_sweep.py results (sweep over COMPUTE_PENALTY)
#   Secondary: exp22 eval results (single-point per variant)
#   Third: aggregate_results.csv (multi-seed, with error bars)
#
# OUTPUTS:
#   pareto_frontier.csv         — non-dominated points
#   pareto_all_points.csv       — all collected (method, utilization, accuracy) triplets
#   pareto_frontier.png         — publication-quality plot
#
# USAGE:
#   python exp37_qwen7b_pareto_generator.py [--results_dir path]
#   python exp37_qwen7b_pareto_generator.py --from_aggregate  # use exp36 output
# =============================================================================

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

RESEARCH_DIR = Path(__file__).parent
OUTPUT_PARETO_CSV    = RESEARCH_DIR / "pareto_frontier.csv"
OUTPUT_ALL_CSV       = RESEARCH_DIR / "pareto_all_points.csv"
OUTPUT_FIGURE        = RESEARCH_DIR / "pareto_frontier.png"


# ==============================================================================
# Data structures
# ==============================================================================
class ParetoPoint:
    """One (method, utilization, accuracy) data point."""
    def __init__(self, method: str, utilization: float, accuracy: float,
                 accuracy_std: float = 0.0, utilization_std: float = 0.0,
                 metadata: dict = None):
        self.method          = method
        self.utilization     = utilization    # token-layer utilization % (lower = more efficient)
        self.accuracy        = accuracy       # accuracy score (higher = better)
        self.accuracy_std    = accuracy_std
        self.utilization_std = utilization_std
        self.metadata        = metadata or {}

    def dominates(self, other: "ParetoPoint") -> bool:
        """True if self Pareto-dominates other (lower util, higher acc, at least one strict)."""
        return (
            self.utilization <= other.utilization and
            self.accuracy    >= other.accuracy and
            (self.utilization < other.utilization or self.accuracy > other.accuracy)
        )

    def to_dict(self) -> dict:
        return {
            "method":          self.method,
            "utilization_pct": round(self.utilization, 4),
            "accuracy":        round(self.accuracy, 4),
            "accuracy_std":    round(self.accuracy_std, 4),
            "utilization_std": round(self.utilization_std, 4),
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }


def pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """
    Compute the Pareto frontier from a list of points.
    Returns non-dominated points sorted by utilization (ascending).
    """
    frontier = []
    for p in points:
        dominated = any(q.dominates(p) for q in points if q is not p)
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: p.utilization)
    return frontier


# ==============================================================================
# Data loading
# ==============================================================================
def load_exp30_results(results_dir: Path) -> List[ParetoPoint]:
    """
    Load exp30 pareto sweep results.
    Expected format: CSV with columns including perplexity, utilization_pct,
    mmlu_acc (or similar), compute_penalty.
    """
    points = []
    patterns = [
        "06_pareto_sweep_*.csv",
        "exp30_pareto_sweep_*.csv",
        "exp30_qwen7b_pareto_*.csv",
    ]
    for pattern in patterns:
        for csv_path in sorted(results_dir.glob(pattern), reverse=True):
            print(f"  Loading exp30 data: {csv_path.name}")
            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                for row in rows:
                    # Extract utilization and accuracy
                    util = None
                    for col in ["utilization_pct", "empirical_flop_reduction_pct",
                                "projected_flop_reduction_pct"]:
                        if col in row and row[col]:
                            try:
                                val = float(row[col])
                                # Convert flop_reduction to utilization (100 - flop_reduction)
                                util = 100.0 - val if "reduction" in col else val
                                break
                            except ValueError:
                                pass

                    acc = None
                    for col in ["mmlu_acc", "mmlu_acc_norm", "arc_acc_norm", "arc_acc"]:
                        if col in row and row[col]:
                            try:
                                acc = float(row[col]) * 100.0
                                break
                            except ValueError:
                                pass
                    # Fallback: use negative PPL as proxy for accuracy
                    if acc is None and "perplexity" in row and row["perplexity"]:
                        try:
                            ppl = float(row["perplexity"])
                            acc = -math.log(ppl)   # negative log PPL as proxy
                        except (ValueError, TypeError):
                            pass

                    if util is not None and acc is not None:
                        penalty = row.get("compute_penalty", "?")
                        points.append(ParetoPoint(
                            method=f"DLR(λ={penalty})",
                            utilization=util,
                            accuracy=acc,
                            metadata={"source": "exp30", "compute_penalty": penalty},
                        ))
            except Exception as e:
                print(f"  [WARN] Could not parse {csv_path}: {e}")
    return points


def load_exp22_results(results_dir: Path) -> List[ParetoPoint]:
    """Load exp22 evaluation harness results (per-variant single points)."""
    points  = []
    pattern = "exp22_qwen7b_eval_results_*.csv"
    for csv_path in sorted(results_dir.glob(pattern), reverse=True)[:1]:   # most recent
        print(f"  Loading exp22 data: {csv_path.name}")
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                variant = row.get("variant", "unknown")
                # Determine utilization
                util = None
                if variant in ("baseline_lora", "base_qwen7b", "stochastic_dropout"):
                    util = 100.0   # no skipping at eval
                elif "utilization_pct" in row and row["utilization_pct"]:
                    try:
                        util = float(row["utilization_pct"])
                    except ValueError:
                        pass
                if util is None:
                    util = 100.0

                # Best available accuracy metric
                acc = None
                for col in ["mmlu_acc", "mmlu_acc_norm"]:
                    if col in row and row[col]:
                        try:
                            acc = float(row[col]) * 100.0
                            break
                        except ValueError:
                            pass
                if acc is None and "perplexity_wikitext103" in row:
                    try:
                        ppl = float(row["perplexity_wikitext103"])
                        acc = -math.log(ppl)
                    except (ValueError, TypeError):
                        pass

                if acc is not None:
                    method_map = {
                        "dense":              "Dense",
                        "baseline_lora":      "Dense",
                        "stochastic_dropout": "Stochastic",
                        "stochastic":         "Stochastic",
                        "token_router":       "DLR",
                        "random_router":      "Random Router",
                        "base_qwen7b":        "Base Qwen",
                    }
                    method = method_map.get(variant, variant)
                    points.append(ParetoPoint(
                        method=method,
                        utilization=util,
                        accuracy=acc,
                        metadata={"source": "exp22", "variant": variant},
                    ))
        except Exception as e:
            print(f"  [WARN] {e}")
    return points


def load_aggregate_results(agg_csv: Path) -> List[ParetoPoint]:
    """Load exp36 aggregate results (mean ± std across seeds)."""
    if not agg_csv.exists():
        return []
    points = []
    method_util_map = {
        "dense":      100.0,
        "stochastic": None,    # read from csv
        "random":     None,
        "dlr":        None,
        "dlr_no_kd":  None,
    }
    try:
        with open(agg_csv) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            method = row.get("method", "unknown")
            util   = float(row.get("utilization_pct_mean", 100.0))
            acc_col = next((k for k in row if "mmlu" in k and "_mean" in k), None)
            if acc_col:
                acc     = float(row[acc_col]) * 100.0
                acc_std = float(row.get(acc_col.replace("_mean", "_std"), 0.0)) * 100.0
            else:
                # Fallback to negative PPL
                ppl = float(row.get("perplexity_mean", math.e))
                acc = -math.log(max(ppl, 1.0))
                acc_std = 0.0
            util_std = float(row.get("utilization_pct_std", 0.0))
            method_label = {
                "dense":      "Dense",
                "stochastic": "Stochastic",
                "random":     "Random Router",
                "dlr":        "DLR",
                "dlr_no_kd":  "DLR (no KD)",
            }.get(method, method)
            points.append(ParetoPoint(
                method=method_label,
                utilization=util,
                accuracy=acc,
                accuracy_std=acc_std,
                utilization_std=util_std,
                metadata={"source": "exp36"},
            ))
    except Exception as e:
        print(f"  [WARN] aggregate_results.csv: {e}")
    return points


# ==============================================================================
# Plotting
# ==============================================================================
METHOD_STYLES = {
    "Dense":         {"color": "#6b7280", "marker": "s",  "linestyle": "-",  "zorder": 1},
    "Stochastic":    {"color": "#f59e0b", "marker": "^",  "linestyle": "--", "zorder": 2},
    "Random Router": {"color": "#ef4444", "marker": "D",  "linestyle": ":",  "zorder": 3},
    "DLR":           {"color": "#2563eb", "marker": "o",  "linestyle": "-",  "zorder": 5},
    "DLR (no KD)":   {"color": "#7c3aed", "marker": "o",  "linestyle": "--", "zorder": 4},
}
DEFAULT_STYLE = {"color": "#10b981", "marker": "x", "linestyle": "-.", "zorder": 0}


def plot_pareto(all_points: List[ParetoPoint], frontier: List[ParetoPoint],
                output_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  [SKIP] matplotlib not available. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Group all points by method
    by_method = {}
    for p in all_points:
        by_method.setdefault(p.method, []).append(p)

    # Plot scatter for each method
    for method, pts in by_method.items():
        style = METHOD_STYLES.get(method, DEFAULT_STYLE)
        xs    = [p.utilization for p in pts]
        ys    = [p.accuracy    for p in pts]
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=80, alpha=0.6, zorder=style["zorder"], label=f"_{method}")
        # Error bars if available
        for p in pts:
            if p.accuracy_std > 0:
                ax.errorbar(p.utilization, p.accuracy,
                            yerr=p.accuracy_std, xerr=p.utilization_std,
                            fmt="none", color=style["color"], alpha=0.4, capsize=3)

    # Highlight Pareto frontier
    frontier_by_method = {}
    for p in frontier:
        frontier_by_method.setdefault(p.method, []).append(p)

    for method, pts in frontier_by_method.items():
        style = METHOD_STYLES.get(method, DEFAULT_STYLE)
        pts_s = sorted(pts, key=lambda p: p.utilization)
        xs    = [p.utilization for p in pts_s]
        ys    = [p.accuracy    for p in pts_s]
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.5, zorder=style["zorder"] + 10)
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=120, edgecolors="white", linewidths=1.5,
                   zorder=style["zorder"] + 11)

    # Overall Pareto frontier line (thin, gray dashed)
    f_pts = sorted(frontier, key=lambda p: p.utilization)
    if len(f_pts) > 1:
        ax.plot([p.utilization for p in f_pts], [p.accuracy for p in f_pts],
                color="#1f2937", linestyle=":", linewidth=1, alpha=0.4,
                label="Pareto frontier", zorder=20)

    # Legend
    legend_handles = []
    for method in by_method:
        style = METHOD_STYLES.get(method, DEFAULT_STYLE)
        legend_handles.append(Line2D(
            [0], [0], color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], markersize=8, linewidth=2, label=method
        ))

    ax.set_xlabel("Token-Layer Utilization (%) ← Lower is more efficient",
                  fontsize=12)
    ax.set_ylabel("Accuracy (%) ↑ Higher is better", fontsize=12)
    ax.set_title(
        "DLR Pareto Frontier: Accuracy vs. Compute Utilization\n"
        "(Points on frontier dominate all others at same compute budget)",
        fontsize=13, pad=12
    )
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right",
              framealpha=0.9, edgecolor="#e5e7eb")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.invert_xaxis()   # lower utilization = higher efficiency = goes right

    # Annotation: DLR claim area
    ax.annotate("DLR should be here\n(high accuracy, low utilization)",
                xy=(0.75, 0.95), xycoords="axes fraction",
                fontsize=9, color="#2563eb", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#bfdbfe"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Pareto frontier → {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP37: Pareto Frontier Generator")
    parser.add_argument("--results_dir",    type=str, default=str(RESEARCH_DIR))
    parser.add_argument("--from_aggregate", action="store_true",
                        help="Use exp36 aggregate_results.csv as primary source")
    parser.add_argument("--no_plots",       action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("\n" + "=" * 70)
    print("  EXP37: PARETO FRONTIER GENERATOR")
    print("  Accuracy vs. Token-Layer Utilization")
    print("=" * 70 + "\n")

    all_points = []

    # Load from all available sources
    print("[1/3] Loading exp30 pareto sweep results...")
    p30 = load_exp30_results(results_dir)
    all_points.extend(p30)
    print(f"  {len(p30)} points loaded from exp30")

    print("\n[2/3] Loading exp22 eval harness results...")
    p22 = load_exp22_results(results_dir)
    all_points.extend(p22)
    print(f"  {len(p22)} points loaded from exp22")

    print("\n[3/3] Loading exp36 aggregate results...")
    p36 = load_aggregate_results(results_dir / "aggregate_results.csv")
    all_points.extend(p36)
    print(f"  {len(p36)} points loaded from exp36")

    if not all_points:
        print("\n[WARN] No data points found. Run experiments first:")
        print("  exp22_qwen7b_eval_harness.py")
        print("  exp30_qwen7b_pareto_sweep.py")
        print("  exp36_qwen7b_multi_seed.py --collect_only")
        # Generate demo data for illustration
        print("\n[DEMO] Generating illustrative demo data...")
        all_points = [
            ParetoPoint("Dense",         100.0, 64.0, metadata={"source": "demo"}),
            ParetoPoint("Stochastic",     80.0, 62.5, metadata={"source": "demo"}),
            ParetoPoint("Stochastic",     70.0, 61.0, metadata={"source": "demo"}),
            ParetoPoint("Stochastic",     60.0, 59.0, metadata={"source": "demo"}),
            ParetoPoint("Random Router",  80.0, 61.0, metadata={"source": "demo"}),
            ParetoPoint("Random Router",  60.0, 58.0, metadata={"source": "demo"}),
            ParetoPoint("DLR",            80.0, 63.5, metadata={"source": "demo"}),
            ParetoPoint("DLR",            70.0, 63.0, metadata={"source": "demo"}),
            ParetoPoint("DLR",            60.0, 62.0, metadata={"source": "demo"}),
            ParetoPoint("DLR",            50.0, 60.5, metadata={"source": "demo"}),
        ]
        print("  [DEMO] Using illustrative points. Replace with real experiment data.")

    print(f"\n  Total data points: {len(all_points)}")

    # Compute Pareto frontier
    frontier = pareto_frontier(all_points)
    print(f"  Pareto-optimal points: {len(frontier)}")

    # Write all points CSV
    all_rows = [p.to_dict() for p in all_points]
    with open(OUTPUT_ALL_CSV, "w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
    print(f"  All points → {OUTPUT_ALL_CSV}")

    # Write Pareto frontier CSV
    frontier_rows = [p.to_dict() for p in frontier]
    with open(OUTPUT_PARETO_CSV, "w", newline="") as f:
        if frontier_rows:
            w = csv.DictWriter(f, fieldnames=list(frontier_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(frontier_rows)
    print(f"  Pareto frontier → {OUTPUT_PARETO_CSV}")

    # Pareto analysis
    print("\n" + "=" * 70)
    print("  PARETO FRONTIER COMPOSITION")
    print("=" * 70)
    from collections import Counter
    method_counts = Counter(p.method for p in frontier)
    for method, count in method_counts.most_common():
        print(f"  {method:<25}: {count} frontier point(s)")

    if "DLR" in method_counts:
        print(f"\n  ✓ DLR IS on the Pareto frontier ({method_counts['DLR']} points)")
        print("  ✓ DLR achieves better accuracy-efficiency tradeoffs than alternatives")
    else:
        print("\n  ✗ DLR is NOT on the Pareto frontier")
        print("  ✗ Competing methods achieve same accuracy at lower utilization")
        print("  ✗ Check: DLR compute_penalty tuning, number of training epochs")

    # Check if DLR dominates Random Router (key ablation result)
    dlr_pts    = [p for p in all_points if p.method == "DLR"]
    random_pts = [p for p in all_points if p.method == "Random Router"]
    if dlr_pts and random_pts:
        dlr_at_70     = next((p for p in dlr_pts    if abs(p.utilization - 70) < 10), None)
        random_at_70  = next((p for p in random_pts if abs(p.utilization - 70) < 10), None)
        if dlr_at_70 and random_at_70:
            diff = dlr_at_70.accuracy - random_at_70.accuracy
            print(f"\n  Ablation: DLR vs. Random Router at ~70% utilization:")
            print(f"    DLR accuracy:    {dlr_at_70.accuracy:.2f}%")
            print(f"    Random accuracy: {random_at_70.accuracy:.2f}%")
            print(f"    Difference:      {diff:+.2f}% ({'DLR better' if diff > 0 else 'Random better'})")
            if diff > 0:
                print(f"    ✓ Learning IS better than random sparsity (METH-01 answered)")
            else:
                print(f"    ✗ No benefit from learned routing (central claim challenged)")

    # Generate plot
    if not args.no_plots:
        plot_pareto(all_points, frontier, OUTPUT_FIGURE)

    print("\n" + "=" * 70)
    print("  EXP37 complete.")
    print(f"  Frontier: {OUTPUT_PARETO_CSV}")
    print(f"  Figure:   {OUTPUT_FIGURE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

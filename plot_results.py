import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob
import os
import json

def get_latest_csv(prefix):
    """Return the most recently modified CSV matching the given prefix pattern."""
    if "inference_benchmark" in prefix:
        files = glob.glob(f"{prefix}_*.csv")
    elif "exp7_eval" in prefix:
        files = glob.glob(f"{prefix}_*.csv")
    else:
        files = glob.glob(f"{prefix}_metrics_*.csv")
    if not files:
        return None
    # Filter out empty files (header-only runs that crashed early)
    files = [f for f in files if os.path.getsize(f) > 200]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    print("Looking for the latest experiment results...")
    exp1_file = get_latest_csv("exp1_baseline")
    exp2_file = get_latest_csv("exp2_stochastic")
    exp3_file = get_latest_csv("exp3_dynamic")

    if not all([exp1_file, exp2_file, exp3_file]):
        print("Missing one or more Phase 1 experiment CSV files.")
        return

    print("Loading data...")
    df1 = pd.read_csv(exp1_file)
    df2 = pd.read_csv(exp2_file)
    df3 = pd.read_csv(exp3_file)

    val1 = df1[df1['Global Step'] % 50 == 0].copy()
    val2 = df2[df2['Global Step'] % 50 == 0].copy()
    val3 = df3[df3['Global Step'] % 50 == 0].copy()

    final_val1 = val1['Validation Loss'].iloc[-1]
    final_val2 = val2['Validation Loss'].iloc[-1]
    final_val3 = val3['Validation Loss'].iloc[-1]

    final_train1 = df1['Training Loss'].iloc[-1]
    final_train2 = df2['Training Loss'].iloc[-1]
    final_train3 = df3['Training Loss'].iloc[-1]

    labels = ['Baseline\n(Full Layers)', 'Stochastic\n(50% Drop)', 'Dynamic\n(Gating)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # =========================================================================
    # PHASE 1 GRAPHS (Convergence Lines + Final Bars)
    # =========================================================================
    
    # 1. Validation Loss Convergence (Line Graph)
    plt.figure(figsize=(10, 6))
    plt.plot(val1['Global Step'], val1['Validation Loss'], label='Baseline', marker='o', linewidth=2.5, markersize=6, color=colors[0])
    plt.plot(val2['Global Step'], val2['Validation Loss'], label='Stochastic', marker='s', linewidth=2.5, markersize=6, color=colors[1])
    plt.plot(val3['Global Step'], val3['Validation Loss'], label='Dynamic', marker='^', linewidth=2.5, markersize=6, color=colors[2])
    plt.title('Validation Loss Convergence over Time', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('validation_loss_convergence.png', dpi=300, bbox_inches='tight')

    # 2. Final Validation Loss (Bar Chart)
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, [final_val1, final_val2, final_val3], color=colors, edgecolor='black', alpha=0.85, width=0.6)
    plt.title('Final Validation Loss Comparison', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Validation Loss (Lower is Better)', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('validation_loss_final_bar.png', dpi=300, bbox_inches='tight')

    # 3. Training Loss Trajectory (Line Graph)
    plt.figure(figsize=(10, 6))
    plt.plot(df1['Global Step'], df1['Training Loss'].ewm(alpha=0.1).mean(), label='Baseline', linewidth=2.5, color=colors[0])
    plt.plot(df2['Global Step'], df2['Training Loss'].ewm(alpha=0.1).mean(), label='Stochastic', linewidth=2.5, color=colors[1])
    plt.plot(df3['Global Step'], df3['Training Loss'].ewm(alpha=0.1).mean(), label='Dynamic', linewidth=2.5, color=colors[2])
    plt.title('Smoothed Training Loss Trajectory (EWMA)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('training_loss_trajectory.png', dpi=300, bbox_inches='tight')

    # 4. Final Training Loss (Bar Chart)
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, [final_train1, final_train2, final_train3], color=colors, edgecolor='black', alpha=0.85, width=0.6)
    plt.title('Final Training Loss Comparison', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Training Loss (Lower is Better)', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_loss_final_bar.png', dpi=300, bbox_inches='tight')

    # =========================================================================
    # PHASE 2 GRAPHS (Pareto Sweep and Inference)
    # =========================================================================

    # 5. Inference Speedup (Bar Chart)
    inf_files = glob.glob("inference_benchmark_*.csv")
    if inf_files:
        inf_file = sorted(inf_files, key=os.path.getmtime, reverse=True)[0]
        df_inf = pd.read_csv(inf_file)
        plt.figure(figsize=(9, 6))
        bars = plt.bar(df_inf['Active Layers'].astype(str), df_inf['Tokens Per Second'], color='#17becf', edgecolor='black', alpha=0.85, width=0.6)
        plt.title('Inference Speed vs. Active Transformer Layers', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Active Layers Used', fontsize=14)
        plt.ylabel('Speed (Tokens / Second)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f'{int(yval)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig('inference_speedup_bar.png', dpi=300, bbox_inches='tight')

    # Pareto Sweep Data
    pareto_files = glob.glob("pareto_sweep_metrics_*.csv")
    if pareto_files:
        pareto_file = sorted(pareto_files, key=os.path.getmtime, reverse=True)[0]
        df_par = pd.read_csv(pareto_file)
        
        # 6. Pareto Frontier Scatter Plot (Accuracy vs Compute)
        plt.figure(figsize=(10, 6))
        # Plot Dynamic Router curve
        plt.plot(df_par['Avg Active Layers'], df_par['Validation Loss'], marker='D', color='#2ca02c', linewidth=2.5, markersize=8, label='Dynamic Routing')
        for i, row in df_par.iterrows():
            plt.annotate(f"P={row['Compute Penalty']}", 
                         (row['Avg Active Layers'], row['Validation Loss']),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#2ca02c')
        
        # Plot Baseline Point
        plt.scatter(22, final_val1, color='#1f77b4', s=150, zorder=5, label='Baseline (22 Layers)')
        plt.annotate("Baseline", (22, final_val1), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, fontweight='bold', color='#1f77b4')
        
        # Plot Stochastic Point
        plt.scatter(13, final_val2, color='#ff7f0e', s=150, zorder=5, label='Stochastic (13 Layers)')
        plt.annotate("Stochastic", (13, final_val2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')

        plt.title('Ultimate Pareto Frontier: Accuracy vs. Compute Efficiency', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Average Active Layers (Compute Cost)', fontsize=14)
        plt.ylabel('Validation Loss (Lower is Better)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().invert_xaxis()
        plt.legend(fontsize=12, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig('pareto_frontier_curve.png', dpi=300, bbox_inches='tight')

        # 6b. Pareto Sweep (Dual-Axis Line Graph)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # X-axis is Compute Penalty
        x = df_par['Compute Penalty'].astype(str)
        
        # Left Y-axis: Active Layers
        color1 = '#1f77b4'
        ax1.set_xlabel('Compute Penalty', fontsize=14)
        ax1.set_ylabel('Avg Active Layers (Compute Cost)', fontsize=14, color=color1)
        ax1.plot(x, df_par['Avg Active Layers'], color=color1, marker='o', linewidth=3, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Right Y-axis: Validation Loss
        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('Validation Loss', fontsize=14, color=color2)
        ax2.plot(x, df_par['Validation Loss'], color=color2, marker='D', linewidth=3, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Impact of Compute Penalty on Layers & Accuracy', fontsize=16, fontweight='bold', pad=15)
        fig.tight_layout()
        plt.savefig('pareto_dual_axis.png', dpi=300, bbox_inches='tight')

        # 7. Pareto Sweep (Bar Chart)
        plt.figure(figsize=(10, 6))
        bars = plt.bar([f"Penalty\n{p}" for p in df_par['Compute Penalty']], df_par['Validation Loss'], color='#d62728', edgecolor='black', alpha=0.85, width=0.6)
        plt.title('Validation Loss across Compute Penalties', fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Validation Loss', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            layers = df_par['Avg Active Layers'].iloc[i]
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}\n({layers:.1f} Layers)', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('pareto_sweep_bar.png', dpi=300, bbox_inches='tight')

    print("\nPlotting complete! 7 distinct visualizations have been generated for your paper.")

    # =========================================================================
    # PHASE 3 GRAPHS (Gumbel Router — exp6)
    # =========================================================================
    exp6_file = get_latest_csv("exp6_gumbel")
    if exp6_file:
        print(f"\nFound exp6 Gumbel metrics: {exp6_file}")
        df6 = pd.read_csv(exp6_file)

        # Drop rows where Training Loss is missing or invalid
        df6 = df6.dropna(subset=["Training Loss"])
        df6["Training Loss"] = pd.to_numeric(df6["Training Loss"], errors="coerce")
        df6["CE Loss"]       = pd.to_numeric(df6["CE Loss"],       errors="coerce")
        df6["KD Loss"]       = pd.to_numeric(df6["KD Loss"],       errors="coerce")
        df6["Gate Loss"]     = pd.to_numeric(df6["Gate Loss"],     errors="coerce")
        df6["Gumbel Temp"]   = pd.to_numeric(df6["Gumbel Temp"],   errors="coerce")
        df6["Avg Active Layers"] = pd.to_numeric(df6["Avg Active Layers"], errors="coerce")
        df6["Validation Loss"]   = pd.to_numeric(df6["Validation Loss"],   errors="coerce")

        # ------------------------------------------------------------------
        # 8. Loss Component Breakdown (CE + KD + Gate over training steps)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df6["Global Step"], df6["CE Loss"].ewm(alpha=0.2).mean(),
                label="CE Loss", color="#1f77b4", linewidth=2)
        ax.plot(df6["Global Step"], df6["KD Loss"].ewm(alpha=0.2).mean(),
                label="KD Loss (teacher distillation)", color="#ff7f0e", linewidth=2)
        ax.plot(df6["Global Step"], df6["Gate Loss"].ewm(alpha=0.2).mean(),
                label="Gate Sparsity Loss", color="#2ca02c", linewidth=2)
        ax.set_title("Exp6: Loss Component Breakdown (Gumbel Router)",
                     fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Global Step", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(fontsize=12, frameon=True, shadow=True)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("exp6_loss_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  -> exp6_loss_breakdown.png")

        # ------------------------------------------------------------------
        # 9. Gumbel Temperature Annealing Curve
        # ------------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color_temp = "#9467bd"
        ax1.plot(df6["Global Step"], df6["Gumbel Temp"],
                 color=color_temp, linewidth=2.5, marker="o", markersize=4)
        ax1.set_xlabel("Global Step", fontsize=14)
        ax1.set_ylabel("Gumbel Temperature", fontsize=14, color=color_temp)
        ax1.tick_params(axis="y", labelcolor=color_temp)
        ax1.set_ylim(0, 1.1)

        ax2 = ax1.twinx()
        valid_layers = df6["Avg Active Layers"].dropna()
        valid_steps  = df6.loc[valid_layers.index, "Global Step"]
        ax2.plot(valid_steps, valid_layers, color="#17becf",
                 linewidth=2.5, linestyle="--", marker="s", markersize=4,
                 label="Avg Active Layers")
        ax2.set_ylabel("Avg Active Layers", fontsize=14, color="#17becf")
        ax2.tick_params(axis="y", labelcolor="#17becf")

        plt.title("Gumbel Temperature Annealing vs. Avg Active Layers",
                  fontsize=16, fontweight="bold", pad=15)
        fig.tight_layout()
        plt.savefig("exp6_temp_annealing.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  -> exp6_temp_annealing.png")

        # ------------------------------------------------------------------
        # 10. Head-to-head Validation Loss: Baseline vs Stochastic vs
        #     Dynamic (REINFORCE) vs Gumbel Router
        # ------------------------------------------------------------------
        # Pull eval rows from each Phase 1 experiment
        val6 = df6.dropna(subset=["Validation Loss"]).copy()

        if not val6.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(val1["Global Step"], val1["Validation Loss"],
                     label="Baseline (exp1)", color=colors[0], linewidth=2.5, marker="o", markersize=5)
            plt.plot(val2["Global Step"], val2["Validation Loss"],
                     label="Stochastic (exp2)", color=colors[1], linewidth=2.5, marker="s", markersize=5)
            plt.plot(val3["Global Step"], val3["Validation Loss"],
                     label="Dynamic-REINFORCE (exp3)", color=colors[2], linewidth=2.5, marker="^", markersize=5)
            plt.plot(val6["Global Step"], val6["Validation Loss"],
                     label="Gumbel-STE Router (exp6)", color="#9467bd", linewidth=2.5,
                     marker="D", markersize=6, linestyle="--")
            plt.title("Validation Loss: All Experiments (Phase 1-3)",
                      fontsize=16, fontweight="bold", pad=15)
            plt.xlabel("Global Step", fontsize=14)
            plt.ylabel("Validation Loss", fontsize=14)
            plt.legend(fontsize=12, frameon=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig("all_experiments_val_loss.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  -> all_experiments_val_loss.png")

        print(f"\nPhase 3 plotting complete! Generated 3 additional exp6 visualizations.")
    else:
        print("\nNo exp6 Gumbel metrics found (all runs were empty / crashed). Skipping Phase 3 plots.")

    # =========================================================================
    # PHASE 4 GRAPHS (exp7 Evaluation Harness — MMLU / GSM8K / ARC / PPL)
    # =========================================================================
    exp7_csv  = get_latest_csv("exp7_eval_results")
    exp7_json = sorted(glob.glob("exp7_eval_summary_*.json"), key=os.path.getmtime, reverse=True)
    exp7_json = exp7_json[0] if exp7_json else None

    if exp7_csv:
        print(f"\nFound exp7 evaluation results: {exp7_csv}")
        df7 = pd.read_csv(exp7_csv)

        # Friendly display names for variants
        variant_labels = {
            "base_tinyllama":    "Base\nTinyLlama",
            "baseline_lora":     "Baseline\nLoRA (exp1)",
            "stochastic_dropout":"Stochastic\nDropout (exp2)",
            "gumbel_router":     "Gumbel\nRouter (exp6)",
        }
        df7["label"] = df7["variant"].map(lambda v: variant_labels.get(v, v))

        variant_colors = {
            "base_tinyllama":    "#aec7e8",
            "baseline_lora":     "#1f77b4",
            "stochastic_dropout":"#ff7f0e",
            "gumbel_router":     "#9467bd",
        }
        df7["color"] = df7["variant"].map(lambda v: variant_colors.get(v, "#888888"))

        # ── 11. Grouped Benchmark Accuracy Bar Chart ───────────────────────────
        benchmark_cols = {
            "mmlu_acc,none":                         "MMLU",
            "gsm8k_exact_match,strict-match":        "GSM8K",
            "arc_challenge_acc_norm,none":            "ARC-Challenge",
        }
        available_benchmarks = {k: v for k, v in benchmark_cols.items() if k in df7.columns}

        if available_benchmarks:
            n_tasks    = len(available_benchmarks)
            n_variants = len(df7)
            x          = range(n_variants)
            bar_width  = 0.22
            fig, ax    = plt.subplots(figsize=(11, 6))

            for i, (col, task_label) in enumerate(available_benchmarks.items()):
                offsets = [xi + (i - n_tasks / 2 + 0.5) * bar_width for xi in x]
                vals    = df7[col].fillna(0).astype(float) * 100
                bars    = ax.bar(offsets, vals, width=bar_width, label=task_label,
                                 alpha=0.88, edgecolor="black")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{val:.1f}%",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

            ax.set_xticks(list(x))
            ax.set_xticklabels(df7["label"], fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=13)
            ax.set_title("Benchmark Accuracy by Model Variant (exp7)",
                         fontsize=16, fontweight="bold", pad=15)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax.legend(fontsize=11, frameon=True, shadow=True)
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig("exp7_benchmark_accuracy.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  -> exp7_benchmark_accuracy.png")
        else:
            print("  [INFO] No lm-eval benchmark columns found in exp7 CSV (perplexity-only run).")

        # ── 12. Perplexity Comparison Bar Chart ────────────────────────────────
        if "perplexity_wikitext103" in df7.columns:
            fig, ax = plt.subplots(figsize=(9, 6))
            ppl_vals = df7["perplexity_wikitext103"].astype(float)
            bars = ax.bar(df7["label"], ppl_vals,
                          color=df7["color"].tolist(),
                          edgecolor="black", alpha=0.88, width=0.55)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        yval + max(ppl_vals) * 0.01,
                        f"{yval:.2f}",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylabel("Perplexity (Wikitext-103 Val) — Lower is Better", fontsize=13)
            ax.set_title("Perplexity Comparison across Model Variants (exp7)",
                         fontsize=16, fontweight="bold", pad=15)
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig("exp7_perplexity_bar.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  -> exp7_perplexity_bar.png")

        # ── 13. Efficiency vs Accuracy Scatter (Layers vs MMLU) ───────────────
        mmlu_col = "mmlu_acc,none"
        if mmlu_col in df7.columns and "avg_active_layers" in df7.columns:
            fig, ax = plt.subplots(figsize=(9, 6))
            for _, row in df7.iterrows():
                try:
                    layers = float(row["avg_active_layers"])
                except (ValueError, TypeError):
                    continue
                acc = float(row[mmlu_col]) * 100
                color = variant_colors.get(row["variant"], "#888888")
                ax.scatter(layers, acc, color=color, s=180, zorder=5, edgecolors="black", linewidths=1)
                ax.annotate(row["label"].replace("\n", " "),
                            (layers, acc),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=10, fontweight="bold", color=color)
            ax.set_xlabel("Average Active Layers (Compute Cost)", fontsize=13)
            ax.set_ylabel("MMLU Accuracy (%)", fontsize=13)
            ax.set_title("Efficiency vs. Accuracy: Compute Cost vs. MMLU (exp7)",
                         fontsize=16, fontweight="bold", pad=15)
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig("exp7_efficiency_scatter.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  -> exp7_efficiency_scatter.png")

        print("\nPhase 4 plotting complete!")
    else:
        print("\nNo exp7 evaluation results found yet. Run exp7_eval_harness.py first.")


if __name__ == "__main__":
    main()

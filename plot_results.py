import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def get_latest_csv(prefix):
    files = glob.glob(f"{prefix}_metrics_*.csv")
    if not files:
        return None
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    print("Looking for the latest experiment results...")
    exp1_file = get_latest_csv("exp1_baseline")
    exp2_file = get_latest_csv("exp2_stochastic")
    exp3_file = get_latest_csv("exp3_dynamic")

    if not all([exp1_file, exp2_file, exp3_file]):
        print("Missing one or more experiment CSV files. Please run all three scripts first.")
        print(f"Found:\nExp1: {exp1_file}\nExp2: {exp2_file}\nExp3: {exp3_file}")
        return

    print("Loading data...")
    df1 = pd.read_csv(exp1_file)
    df2 = pd.read_csv(exp2_file)
    df3 = pd.read_csv(exp3_file)

    # Validation Data
    val1 = df1.dropna(subset=['Validation Loss']).copy()
    val2 = df2.dropna(subset=['Validation Loss']).copy()
    val3 = df3.dropna(subset=['Validation Loss']).copy()
    
    # Training Data
    train1 = df1.dropna(subset=['Training Loss']).copy()
    train2 = df2.dropna(subset=['Training Loss']).copy()
    train3 = df3.dropna(subset=['Training Loss']).copy()

    # --- Plot 1: Validation Loss Comparison ---
    plt.figure(figsize=(10, 6))
    # We use a custom style to make it look clean and paper-ready
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    plt.plot(val1['Epoch'], val1['Validation Loss'], label='Baseline (Full Layers)', marker='o', linewidth=2.5, markersize=6)
    plt.plot(val2['Epoch'], val2['Validation Loss'], label='Stochastic (50% Dropout)', marker='s', linewidth=2.5, markersize=6)
    plt.plot(val3['Epoch'], val3['Validation Loss'], label='Dynamic Routing (Gating)', marker='^', linewidth=2.5, markersize=6)
    
    plt.title('Validation Loss Convergence vs Routing Strategy', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('validation_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 'validation_loss_comparison.png'")

    # --- Plot 2: Training Loss (Smoothed) ---
    window = max(1, len(train1) // 20)  # Dynamic smoothing window
    plt.figure(figsize=(10, 6))
    
    plt.plot(train1['Epoch'], train1['Training Loss'].rolling(window).mean(), label='Baseline', linewidth=2, alpha=0.9)
    plt.plot(train2['Epoch'], train2['Training Loss'].rolling(window).mean(), label='Stochastic', linewidth=2, alpha=0.9)
    plt.plot(train3['Epoch'], train3['Training Loss'].rolling(window).mean(), label='Dynamic', linewidth=2, alpha=0.9)
    
    plt.title(f'Smoothed Training Loss (Window={window})', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 'training_loss_comparison.png'")
    
    print("\nPlotting complete! The images are ready for your paper.")

if __name__ == "__main__":
    main()

"""Run all 3 min variants sequentially, build comparison table. python new/experiments/compare_min.py [--smoke]"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # new/


def run_variant(variant, smoke, seed=42):
    cmd = [sys.executable, os.path.join(REPO, "experiments", "train_min.py"), "--variant", variant, "--seed", str(seed)]
    if smoke:
        cmd.append("--smoke")
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(REPO, "artifacts", "runs", f"min135m_{variant}_seed{seed}", "summary.json")) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows = [run_variant(v, args.smoke, args.seed) for v in ["dense", "stochastic_30", "token_dlr_30"]]
    out = os.path.join(REPO, "artifacts", "runs", "min135m_compare.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("\n| variant | eval_ppl | skip | mean_gate | entropy | collapse | wall_s |")
    for r in rows:
        fm = r.get("final_metrics") or {}
        print(f"| {r['variant']} | {r['eval_ppl']:.3f} | {(1-fm.get('mean_gate', float('nan')) if 'mean_gate' in fm else float('nan')):.3f} "
              f"| {fm.get('mean_gate', float('nan')):.3f} | {fm.get('router_entropy', float('nan')):.3f} "
              f"| {fm.get('collapse_flag', '-')} | {r['wall_clock_s']:.1f} |")
    print("saved", out)


if __name__ == "__main__":
    main()

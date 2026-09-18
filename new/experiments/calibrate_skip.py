"""Skip calibration sweep: trains --smoke at several skip_weights, reports measured skip.

Usage:
  python new/experiments/calibrate_skip.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --variant token_dlr_30 --skip-weights 200 400 800 1600 --seed 42

Picks the weight whose skip is closest to target band (25-35% for _30,
45-55% for _50). NOTE (2026-09-15, TinyLlama-1.1B): 20-step --smoke runs do
NOT move the router (all collapse to keep, skip ~0.005 regardless of weight);
calibration must use full-length runs (--max-steps 150, ~55s each on 4060).
Example that hit the band: token_dlr_30 sw800 -> train-tail skip 0.275.
Writes new/artifacts/analysis/calibrate_<variant>_seed<seed>.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "experiments", "train.py")

BANDS = {"_30": (0.25, 0.35), "_50": (0.45, 0.55)}


def band_for(variant: str):
    for suffix, band in BANDS.items():
        if variant.endswith(suffix):
            return band
    raise ValueError(f"No target band for {variant}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--variant", default="token_dlr_30")
    p.add_argument("--skip-weights", nargs="+", type=float, default=[200, 400, 800, 1600])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--smoke", action="store_true",
                   help="20-step plumbing check only — does NOT calibrate (router never moves).")
    args = p.parse_args()

    lo, hi = band_for(args.variant)
    target = (lo + hi) / 2
    rows = []
    for sw in args.skip_weights:
        run_id = f"calib_{args.variant}_sw{int(sw)}_seed{args.seed}"
        out_dir = os.path.join(REPO, "artifacts", "runs", run_id)
        cmd = [sys.executable, TRAIN, "--model-id", args.model_id, "--variant", args.variant,
               "--seed", str(args.seed), "--run-id", run_id,
               "--skip-weight", str(sw), "--max-steps", str(args.max_steps)]
        if args.smoke:
            cmd.append("--smoke")
        print(f"RUN skip_weight={sw}", flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.dirname(REPO))
        with open(os.path.join(out_dir, "summary.json")) as f:
            s = json.load(f)
        fm = s.get("final_metrics") or {}
        skip = 1 - fm.get("mean_gate", float("nan")) if "mean_gate" in fm else float("nan")
        rows.append({"skip_weight": sw, "smoke_skip": skip,
                     "entropy": fm.get("router_entropy"), "collapse": fm.get("collapse_flag"),
                     "eval_ppl": s.get("eval_ppl")})
        print(f"  -> smoke_skip={skip:.3f} ent={fm.get('router_entropy', float('nan')):.3f} "
              f"collapse={fm.get('collapse_flag')}", flush=True)

    def dist(r):
        return abs(r["smoke_skip"] - target)
    rows.sort(key=dist)
    best = rows[0]
    print(f"\nTarget band [{lo:.2f},{hi:.2f}]. Best smoke weight: {best['skip_weight']} "
          f"(skip={best['smoke_skip']:.3f}). Confirm with full run.")
    out = os.path.join(REPO, "artifacts", "analysis", f"calibrate_{args.variant}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"variant": args.variant, "band": [lo, hi], "rows": rows, "best": best}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()

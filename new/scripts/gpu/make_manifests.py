"""Build GPU-box eval manifests from on-disk run manifests — never hand-copied always_keep.

Why this exists: the qwen15b-001/002/003 manifests shipped with
always_keep_layers=3 (copy-pasted from TinyLlama), but Qwen2.5-1.5B trains
with always_keep=4 (28 layers). The evaluator then builds a 25-wide router
against 24-wide weights -> crash on the first token_dlr row (this is why
qwen15b-001-seed42 died after `dense`). This script reads always_keep /
total_layers from each run's manifest.json and verifies router weight dims.

Modes:
  fix-g3 : rewrite qwen15b-001-seed42 / 002-seed43 / 003-seed44 IN PLACE
           with corrected always_keep (exploratory grade, unchanged otherwise).
  paper  : write new/configs/eval/paper/<family>-paper-seed<seed>.json —
           paper-grade protocol (5-shot ARC/Hella/Wino/MMLU, 500x512 ppl,
           full limit) over the uniform 500-step sw1600 checkpoints, with
           random-mask controls matched to each seed's MEASURED DLR harness
           skips (G1 Pareto decision: compare at measured skips).
           GSM8K excluded: generate_until unsupported for routed rows (gate P10).

Run from repo root:  python new/scripts/gpu/make_manifests.py fix-g3
                     python new/scripts/gpu/make_manifests.py flexi-fp
                     python new/scripts/gpu/make_manifests.py exp --family qwen3b --seed 42
                     python new/scripts/gpu/make_manifests.py paper --family qwen15b --seed 42
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # new/
RUNS = os.path.join(REPO, "artifacts", "runs")
EVALS = os.path.join(REPO, "artifacts", "evals")
CFG_EVAL = os.path.join(REPO, "configs", "eval")

FAMILIES = {
    "tinyllama1b": {"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prefix": "tinyllama1b"},
    "qwen15b": {"model_id": "Qwen/Qwen2.5-1.5B", "prefix": "qwen15b"},
    "qwen3b": {"model_id": "Qwen/Qwen2.5-3B", "prefix": "qwen3b"},
}
# 500-step sw1600 run-id templates (uniform G6 block). {p}=prefix, {s}=seed.
RUN_TMPL = {
    "dense": "{p}_dense_500_seed{s}",
    "stochastic_30": "{p}_stochastic_30_500_seed{s}",
    "token_dlr_30": "{p}_token_dlr_30_sw1600_500_seed{s}",
    "stochastic_50": "{p}_stochastic_50_500_seed{s}",
    "token_dlr_50": "{p}_token_dlr_50_sw1600_500_seed{s}",
}
# Exploratory eval holding the measured harness skips used for matched randoms.
REF_EVAL = {
    ("tinyllama1b", 42): "tinyllama1b-010-g4-500",
    ("tinyllama1b", 43): "tinyllama1b-006-seed43",
    ("tinyllama1b", 44): "tinyllama1b-007-seed44",
    ("qwen15b", 42): "qwen15b-001-seed42",
    ("qwen15b", 43): "qwen15b-002-seed43",
    ("qwen15b", 44): "qwen15b-003-seed44",
    ("qwen3b", 42): "qwen3b-exp-seed42",
    ("qwen3b", 43): "qwen3b-exp-seed43",
    ("qwen3b", 44): "qwen3b-exp-seed44",
}
G3_FILES = {42: "qwen15b-001-seed42.json", 43: "qwen15b-002-seed43.json", 44: "qwen15b-003-seed44.json"}


def load_run(run_id):
    mp = os.path.join(RUNS, run_id, "manifest.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"MISSING checkpoint run: {run_id}\n  expected {mp}")
    with open(mp) as f:
        return json.load(f)


def check_router_dims(run_id, ak, total):
    """Refuse shape mismatches (the qwen15b always_keep=3 bug class). Warn-only if torch absent."""
    try:
        import torch
    except ImportError:
        print(f"  warn: torch absent, skipping router-dim check for {run_id}")
        return
    rp = os.path.join(RUNS, run_id, "router_weights.pt")
    sd = torch.load(rp, map_location="cpu")
    out_dim = None
    for k, v in sd.items():
        if v.dim() == 2 and (out_dim is None or v.shape[0] < out_dim):
            out_dim = v.shape[0]
    # RouterMLP last layer maps hidden/4 -> routable; smallest out-dim row is it.
    want = total - ak
    if out_dim != want:
        raise SystemExit(f"ROUTER SHAPE MISMATCH {run_id}: weights out_dim={out_dim} != total-ak={total}-{ak}={want}")


def ckpt_rel(run_id):
    return f"new/artifacts/runs/{run_id}"


def build_variants(prefix, seed, skip_probs=(0.3, 0.5), randoms=None):
    """randoms: list of (name, skip) matched-mask controls on dense weights, or None."""
    runs = {}
    for variant, tmpl in RUN_TMPL.items():
        rid = tmpl.format(p=prefix, s=seed)
        m = load_run(rid)
        runs[variant] = (rid, m)
    ak = runs["token_dlr_30"][1]["always_keep"]
    ak50 = runs["token_dlr_50"][1]["always_keep"]
    assert ak == ak50, f"always_keep differs between dlr30 ({ak}) and dlr50 ({ak50})"
    total = runs["token_dlr_30"][1]["total_layers"]
    for variant in ("token_dlr_30", "token_dlr_50"):
        check_router_dims(runs[variant][0], ak, total)
    print(f"  {prefix} seed {seed}: always_keep={ak} total_layers={total} (from run manifests)")
    EVAL_SEED = 1234
    V = [
        {"name": "base", "type": "base"},
        {"name": "dense", "type": "lora", "checkpoint": ckpt_rel(runs["dense"][0])},
        {"name": "stochastic_30", "type": "stochastic",
         "checkpoint": ckpt_rel(runs["stochastic_30"][0]),
         "skip_prob": skip_probs[0], "eval_seed": EVAL_SEED, "always_keep_layers": ak},
        {"name": "token_dlr_30", "type": "token_dlr",
         "checkpoint": ckpt_rel(runs["token_dlr_30"][0]),
         "router_weights": ckpt_rel(runs["token_dlr_30"][0]) + "/router_weights.pt",
         "always_keep_layers": ak},
        {"name": "stochastic_50", "type": "stochastic",
         "checkpoint": ckpt_rel(runs["stochastic_50"][0]),
         "skip_prob": skip_probs[1], "eval_seed": EVAL_SEED, "always_keep_layers": ak},
        {"name": "token_dlr_50", "type": "token_dlr",
         "checkpoint": ckpt_rel(runs["token_dlr_50"][0]),
         "router_weights": ckpt_rel(runs["token_dlr_50"][0]) + "/router_weights.pt",
         "always_keep_layers": ak},
    ]
    if randoms:
        for name, skip in randoms:
            V.append({"name": name, "type": "stochastic",
                      "checkpoint": ckpt_rel(runs["dense"][0]),
                      "skip_prob": skip, "eval_seed": EVAL_SEED, "always_keep_layers": ak})
    return V, ak


def measured_skips(family, seed):
    ref = REF_EVAL[(family, seed)]
    rp = os.path.join(EVALS, ref, "routing_metrics.csv")
    if not os.path.isfile(rp):
        raise SystemExit(f"Need reference eval first: {ref}\n  missing {rp}\n  (run 01_run_g3_evals for qwen; tinyllama refs ship with the repo)")
    skips = {}
    with open(rp) as f:
        for row in csv.DictReader(f):
            if row["variant"] in ("token_dlr_30", "token_dlr_50"):
                skips[row["variant"]] = float(row["achieved_skip"])
    if set(skips) != {"token_dlr_30", "token_dlr_50"}:
        raise SystemExit(f"{ref}/routing_metrics.csv lacks token_dlr rows (incomplete eval?)")
    return skips


def cmd_fix_g3(_args):
    for seed, fname in G3_FILES.items():
        path = os.path.join(CFG_EVAL, fname)
        with open(path) as f:
            man = json.load(f)
        # keep every variant, but force always_keep_layers from run manifests
        probe = load_run(f"qwen15b_token_dlr_30_sw1600_500_seed{seed}")
        ak, total = probe["always_keep"], probe["total_layers"]
        check_router_dims(f"qwen15b_token_dlr_30_sw1600_500_seed{seed}", ak, total)
        check_router_dims(f"qwen15b_token_dlr_50_sw1600_500_seed{seed}", ak, total)
        fixed = 0
        for v in man["variants"]:
            if "always_keep_layers" in v and v["always_keep_layers"] != ak:
                print(f"  {man['eval_run_id']}/{v['name']}: always_keep {v['always_keep_layers']} -> {ak}")
                v["always_keep_layers"] = ak
                fixed += 1
        with open(path, "w") as f:
            json.dump(man, f, indent=2, sort_keys=True)
        print(f"rewrote {fname} ({fixed} rows fixed, ak={ak})")


def cmd_exp(args):
    """Exploratory manifest (0-shot/lim100/50x128 + fixed random_20/35 controls)
    for any family+seed from its 500-step runs. Used for new scales (qwen3b)."""
    fam = FAMILIES[args.family]
    V, _ = build_variants(fam["prefix"], args.seed, randoms=[("random_20", 0.2), ("random_35", 0.35)])
    eval_id = f"{fam['prefix']}-exp-seed{args.seed}"
    man = {
        "eval_run_id": eval_id,
        "model_family": args.family,
        "model_id": fam["model_id"],
        "output_dir": "artifacts/evals",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "tasks": ["arc_challenge", "hellaswag", "winogrande"],
        "num_fewshot": 0,
        "batch_size": args.batch_size,
        "limit": 100,
        "max_length": 1024,
        "perplexity": {"enabled": True, "dataset": "Salesforce/wikitext",
                       "config": "wikitext-103-raw-v1", "split": "validation",
                       "samples": 50, "max_length": 128},
        "variants": V,
    }
    out = os.path.join(CFG_EVAL, f"{eval_id}.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"wrote {os.path.relpath(out, os.path.dirname(REPO))}")


def cmd_flexi_fp(_args):
    """Full-precision FlexiDepth 8B audit (ckpt + matched random control, one
    manifest). 8B bf16 ~16GB fits >=20GB GPUs — removes the 4-bit caveat from
    flexidepth-001/002. Batch 4 at 128-len ppl is safe on 20GB."""
    man = {
        "eval_run_id": "flexidepth-003-fp",
        "model_family": "llama3_8b_flexidepth",
        "model_id": "xuan-luo/FlexiDepth-Llama-3-8B-Instruct",
        "output_dir": "artifacts/evals",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "trust_remote_code": True,
        "tasks": ["arc_challenge", "hellaswag", "winogrande"],
        "num_fewshot": 0,
        "batch_size": 4,
        "limit": 100,
        "max_length": 1024,
        "perplexity": {"enabled": True, "dataset": "Salesforce/wikitext",
                       "config": "wikitext-103-raw-v1", "split": "validation",
                       "samples": 50, "max_length": 128},
        "variants": [
            {"name": "flexidepth", "type": "base"},
            {"name": "flexidepth_random_p50_last16", "type": "base_stochastic",
             "skip_prob": 0.5, "eval_seed": 1234, "always_keep_layers": 16},
        ],
        "_note": "Full-precision rerun of flexidepth-001/002 (no 4-bit caveat). "
                 "Random control drops 50% of the same last-16 converted layers "
                 "on the SAME ckpt: learned-vs-random on identical 8B weights.",
    }
    out = os.path.join(CFG_EVAL, "flexidepth-003-fp.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"wrote {os.path.relpath(out, os.path.dirname(REPO))}")


def cmd_paper(args):
    fam = FAMILIES[args.family]
    V, ak = build_variants(fam["prefix"], args.seed)
    sk = measured_skips(args.family, args.seed)
    rlo, rhi = round(sk["token_dlr_30"], 2), round(sk["token_dlr_50"], 2)
    print(f"  measured harness skips: dlr30={sk['token_dlr_30']:.3f} dlr50={sk['token_dlr_50']:.3f} "
          f"-> random controls at {rlo}/{rhi}")
    # rebuild with matched randoms appended
    V, _ = build_variants(fam["prefix"], args.seed, randoms=[("random_lo", rlo), ("random_hi", rhi)])
    eval_id = f"{fam['prefix']}-paper-seed{args.seed}"
    man = {
        "eval_run_id": eval_id,
        "model_family": args.family,
        "model_id": fam["model_id"],
        "output_dir": "artifacts/evals",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "tasks": ["arc_challenge", "hellaswag", "winogrande", "mmlu"],
        "num_fewshot": 5,
        "batch_size": args.batch_size,
        "limit": None,
        "max_length": 1024,
        "perplexity": {"enabled": True, "dataset": "Salesforce/wikitext",
                       "config": "wikitext-103-raw-v1", "split": "validation",
                       "samples": 500, "max_length": 512},
        "variants": V,
    }
    outdir = os.path.join(CFG_EVAL, "paper")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{eval_id}.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"wrote {os.path.relpath(out, os.path.dirname(REPO))}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("fix-g3", help="repair always_keep in qwen15b-001/002/003 in place")
    sub.add_parser("flexi-fp", help="full-precision FlexiDepth 8B audit manifest (>=20GB GPU)")
    pe = sub.add_parser("exp", help="emit exploratory manifest for one family+seed (new scales)")
    pe.add_argument("--family", required=True, choices=list(FAMILIES))
    pe.add_argument("--seed", required=True, type=int)
    pe.add_argument("--batch-size", type=int, default=8)
    pp = sub.add_parser("paper", help="emit paper-grade manifest for one family+seed")
    pp.add_argument("--family", required=True, choices=list(FAMILIES))
    pp.add_argument("--seed", required=True, type=int)
    pp.add_argument("--batch-size", type=int, default=2,
                    help="2 safe for 8GB @512-len ppl; use 8 on >=24GB")
    args = p.parse_args()
    {"fix-g3": cmd_fix_g3, "flexi-fp": cmd_flexi_fp, "exp": cmd_exp, "paper": cmd_paper}[args.mode](args)


if __name__ == "__main__":
    main()

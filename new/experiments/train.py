"""General trainer: python new/experiments/train.py --model-id <id> --variant {dense,stochastic_30,stochastic_50,token_dlr_30,token_dlr_50} [--smoke]

Same shared loop/loss/routing as train_min.py, but model-agnostic.
Target skips: dense=0.0, *_30=0.3, *_50=0.5. Canonical variant_type values are
dense / stochastic_30 / stochastic_50 / token_dlr_30 / token_dlr_50
(dlr_30/50 accepted as aliases).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # new/
sys.path.insert(0, os.path.join(REPO, "src"))

from dlr.data import DataConfig, build_train_eval, collate
from dlr.manifests import RunManifest, count_params
from dlr.models import always_keep_layers, base_total_layers, load_base_model, load_tokenizer, wrap_lora
from dlr.reproducibility import env_snapshot, get_git_commit, set_seed
from dlr.routing import RouterMLP
from dlr.train_loop import train

VARIANTS = ["dense", "stochastic_30", "stochastic_50", "token_dlr_30", "token_dlr_50",
            "dlr_30", "dlr_50"]


def canonical_variant(v: str) -> str:
    if v == "dlr_30":
        return "token_dlr_30"
    if v == "dlr_50":
        return "token_dlr_50"
    return v


def target_for(variant: str) -> float:
    if variant == "dense":
        return 0.0
    if variant.endswith("_30"):
        return 0.3
    if variant.endswith("_50"):
        return 0.5
    raise ValueError(variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--smoke", action="store_true", help="20 steps, 40 lines — plumbing only")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--skip-weight", type=float, default=400.0)
    p.add_argument("--kd-weight", type=float, default=0.2)
    p.add_argument("--l1-weight", type=float, default=0.01)
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=2.0)
    args = p.parse_args()

    variant = canonical_variant(args.variant)
    target_skip = target_for(variant)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    set_seed(args.seed)
    tok = load_tokenizer(args.model_id)
    model = load_base_model(args.model_id, dtype=dtype, device=device)
    model = wrap_lora(model, rank=8, alpha=16, expanded=variant.startswith("token_dlr"))

    total_layers = base_total_layers(model)
    ak = always_keep_layers(total_layers)
    print(f"model={args.model_id} layers={total_layers} always_keep={ak} device={device} dtype={dtype}")

    if args.smoke:
        dc = DataConfig(num_lines=40, max_length=128, eval_blocks=10)
        max_steps, bs = 20, 2
    else:
        dc = DataConfig(num_lines=300, max_length=128, eval_blocks=50)
        max_steps, bs = args.max_steps, args.batch_size

    train_ds, eval_ds = build_train_eval(tok, dc)
    print(f"train_blocks={len(train_ds)} eval_blocks={len(eval_ds)}")

    router = None
    if variant.startswith("token_dlr"):
        hs = model.get_base_model().config.hidden_size
        router = RouterMLP(hs, total_layers - ak).to(device)

    cfg = SimpleNamespace(
        variant_type=variant, device=device, batch_size=bs, lr=args.lr,
        max_steps=max_steps, warmup_steps=min(10, max_steps // 5),
        always_keep=ak, target_skip=target_skip,
        tau=args.tau, router_deterministic=True,
        kd_weight=args.kd_weight, skip_weight=args.skip_weight,
        l1_weight=args.l1_weight, kd_temp=args.kd_temp,
        log_every=10,
    )

    family = args.model_id.split("/")[-1].lower().replace("-", "").replace("_", "").replace(".", "")
    run_id = args.run_id or f"{family}_{variant}_seed{args.seed}"
    out_dir = args.output_dir or os.path.join(REPO, "artifacts", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    result = train(model, router, train_ds, eval_ds, cfg, collate)

    trainable, total = count_params(model)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if router is not None:
        trainable += sum(p.numel() for p in router.parameters())
        total += sum(p.numel() for p in router.parameters())
        torch.save(router.state_dict(), os.path.join(out_dir, "router_weights.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(result["logs"], f, indent=2)

    fm = result["final_metrics"]
    flop_red = (100.0 * (1.0 - fm["mean_active_layers"] / total_layers)) if fm else 0.0
    man = RunManifest(
        run_id=run_id, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=get_git_commit(os.path.dirname(REPO)), command=" ".join(sys.argv), config=vars(cfg),
        model_name=args.model_id, seed=args.seed, hardware=env_snapshot(), precision=str(dtype),
        trainable_params=trainable, total_params=total, always_keep=ak, total_layers=total_layers,
        target_skip=target_skip if variant != "dense" else None,
        measured_active_layers=(fm.get("mean_active_layers") if fm else None),
        final_ppl=result["eval_ppl"], wall_clock_s=result["wall_clock_s"],
    )
    man.write(out_dir)
    summary = {"variant": variant, "eval_ppl": result["eval_ppl"],
               "wall_clock_s": result["wall_clock_s"], "final_metrics": fm,
               "structural_flop_red_pct": flop_red, "out_dir": out_dir}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

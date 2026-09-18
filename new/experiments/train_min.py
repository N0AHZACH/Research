"""Min prototype trainer: python new/experiments/train_min.py --variant {dense,stochastic_30,token_dlr_30} [--smoke]"""
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

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["dense", "stochastic_30", "token_dlr_30"])
    p.add_argument("--smoke", action="store_true", help="20 steps, 40 lines, CPU/bf16 off — plumbing only")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: torch.cuda.is_available() is False — running on CPU. "
              "For 4060 runs install CUDA torch: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    set_seed(args.seed)
    tok = load_tokenizer(MODEL_ID)
    model = load_base_model(MODEL_ID, dtype=dtype, device=device)
    # LoRA: expanded targets only for token_dlr (matches paper spec)
    model = wrap_lora(model, rank=8, alpha=16, expanded=(args.variant == "token_dlr_30"))

    total_layers = base_total_layers(model)
    ak = always_keep_layers(total_layers)
    print(f"model={MODEL_ID} layers={total_layers} always_keep={ak} device={device} dtype={dtype}")

    if args.smoke:
        dc = DataConfig(num_lines=40, max_length=128, eval_blocks=10)
        max_steps, bs = 20, 2
    else:
        dc = DataConfig(num_lines=300, max_length=128, eval_blocks=50)
        max_steps, bs = args.max_steps, 4

    train_ds, eval_ds = build_train_eval(tok, dc)
    print(f"train_blocks={len(train_ds)} eval_blocks={len(eval_ds)}")

    variant_type = {"dense": "dense", "stochastic_30": "stochastic_30", "token_dlr_30": "token_dlr_30"}[args.variant]
    router = None
    if variant_type == "token_dlr_30":
        hs = model.get_base_model().config.hidden_size
        router = RouterMLP(hs, total_layers - ak).to(device)

    cfg = SimpleNamespace(
        variant_type=variant_type, device=device, batch_size=bs, lr=2e-4,
        max_steps=max_steps, warmup_steps=min(10, max_steps // 5),
        always_keep=ak, target_skip=0.3 if variant_type != "dense" else 0.0,
        # v5: router_deterministic=True. Measured train/eval gate gap with Gumbel
        # sampling at tau=2.0: sampled skip 0.215 vs threshold skip 0.031 on the
        # SAME data — the router hides behind small-positive logits that sample
        # skips but evaluate to keep. Train now uses the exact discrete rule the
        # harness evaluates (threshold + STE), so budget numbers mean one thing.
        tau=2.0, router_deterministic=True,
        kd_weight=0.2, skip_weight=400.0, l1_weight=0.01, kd_temp=2.0,
        log_every=10,
    )

    out_dir = args.output_dir or os.path.join(REPO, "artifacts", "runs", f"min135m_{args.variant}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    result = train(model, router, train_ds, eval_ds, cfg, collate)

    trainable, total = count_params(model)
    # Save LoRA adapter + tokenizer: the standard harness (experiments/evaluate.py)
    # reloads variants via PeftModel.from_pretrained(base, checkpoint) with the
    # tokenizer read from the same dir. Without this the run is unevaluable.
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if router is not None:
        trainable += sum(p.numel() for p in router.parameters())
        total += sum(p.numel() for p in router.parameters())
        torch.save(router.state_dict(), os.path.join(out_dir, "router_weights.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(result["logs"], f, indent=2)

    fm = result["final_metrics"]
    # mean_active_layers is effective depth (incl. always_keep) per the canonical
    # routing_metrics definition shared with evaluate.py.
    flop_red = (100.0 * (1.0 - fm["mean_active_layers"] / total_layers)) if fm else 0.0
    man = RunManifest(
        run_id=f"min135m_{args.variant}_seed{args.seed}", created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=get_git_commit(os.path.dirname(REPO)), command=" ".join(sys.argv), config=vars(cfg),
        model_name=MODEL_ID, seed=args.seed, hardware=env_snapshot(), precision=str(dtype),
        trainable_params=trainable, total_params=total, always_keep=ak, total_layers=total_layers,
        target_skip=cfg.target_skip if variant_type != "dense" else None,
        measured_active_layers=(fm.get("mean_active_layers") if fm else None),
        final_ppl=result["eval_ppl"], wall_clock_s=result["wall_clock_s"],
    )
    man.write(out_dir)
    summary = {"variant": args.variant, "eval_ppl": result["eval_ppl"],
               "wall_clock_s": result["wall_clock_s"], "final_metrics": fm,
               "structural_flop_red_pct": flop_red, "out_dir": out_dir}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

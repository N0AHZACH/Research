import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP32: QWEN2.5-7B RANDOM TOKEN ROUTER — Ablation Baseline
#
# SCIENTIFIC PURPOSE:
#   This script is the critical ablation required to prove that DLR (exp25)
#   learns input-conditional routing rather than merely benefiting from sparsity.
#
#   Design: Identical to exp25 in every respect EXCEPT the router.
#   - exp25 (DLR):    TokenLevelGumbelRouter — LEARNED, input-conditional
#   - exp32 (Random): RandomBernoulliRouter  — RANDOM, input-INDEPENDENT
#
#   Both use:
#     - Same p_active = 1 - TARGET_SKIP (matched utilization budget)
#     - Same LoRA configuration and targets
#     - Same training dataset, batch size, LR schedule
#     - Same evaluation protocol
#
#   If DLR > Random Router at equal utilization:
#     → The LEARNING is doing meaningful work (routing is adaptive)
#
#   If DLR ≈ Random Router:
#     → Sparsity alone explains the results (routing is NOT adaptive)
#
#   Required by reviewers at NeurIPS/ICML/ICLR before accepting adaptive
#   compute routing claims. See audit METH-01.
#
# METHODOLOGY NOTE (FLOP accounting):
#   Both exp25 and exp32 use hook-based gating (soft blend after full compute).
#   The token-layer utilization reported is EXACT (real gate counts).
#   The FLOP reduction is PROJECTED (assumes future token-packing hardware).
#   Both methods are evaluated identically, so relative comparisons are valid.
# =============================================================================

import csv
import gc
import math
import os
import itertools
import datetime
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==============================================================================
# Configuration — must match exp25 exactly for fair comparison
# ==============================================================================
MODEL_ID         = "Qwen/Qwen2.5-7B"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 3
LR               = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_STEPS     = 100

ALWAYS_KEEP      = 4
# p_active = 1 - TARGET_SKIP must match exp25's TARGET_SKIP=0.40
TARGET_SKIP      = 0.40    # keep in sync with exp25
P_ACTIVE         = 1.0 - TARGET_SKIP   # = 0.60 (60% of routable layers active per token)

EVAL_EVERY_STEPS = 50
LOG_EVERY_STEPS  = 10

# LoRA — identical to exp25
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

SEED = 42


# ==============================================================================
# Hardware auto-configuration (identical to exp25)
# ==============================================================================
def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, False, torch.float32

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    if   vram_gb >= 80: bs, ga = 16, 1
    elif vram_gb >= 45: bs, ga = 8,  2
    elif vram_gb >= 35: bs, ga = 8,  2
    elif vram_gb >= 22: bs, ga = 4,  4
    elif vram_gb >= 14: bs, ga = 2,  8
    else:               bs, ga = 1,  16

    major, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.float16 if major < 8 else torch.bfloat16
    nw = 0
    try:
        import flash_attn
        attn = "flash_attention_2" if vram_gb >= 7 else None
    except ImportError:
        attn = "sdpa" if vram_gb >= 7 else None
    print(f"[HARDWARE] {gpu_name} | {vram_gb:.1f}GB | BS={bs} GA={ga} dtype={compute_dtype} attn={attn}")
    return bs, ga, nw, attn, False, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp32_random_router_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp32_qwen7b_random_output_{TIMESTAMP}"


# ==============================================================================
# FLOP Accounting (shared methodology with exp25)
# ==============================================================================
def estimate_layer_flops_per_token(config, seq_len: int) -> float:
    H      = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv   = getattr(config, "num_key_value_heads", n_heads)
    hd     = H // n_heads
    ff     = config.intermediate_size
    q_flops  = 2 * H * H
    k_flops  = 2 * H * (n_kv * hd)
    v_flops  = 2 * H * (n_kv * hd)
    o_flops  = 2 * H * H
    attn_qk  = 2 * seq_len * H
    attn_av  = 2 * seq_len * H
    gate_f   = 2 * H * ff
    up_f     = 2 * H * ff
    down_f   = 2 * ff * H
    return q_flops + k_flops + v_flops + o_flops + attn_qk + attn_av + gate_f + up_f + down_f


# ==============================================================================
# Random Token Router
# ==============================================================================
class RandomBernoulliRouter(nn.Module):
    """
    Random Token Router — ablation baseline for DLR.

    Gates are sampled i.i.d. from Bernoulli(p_active), independently of input.
    No learnable parameters. The routing decision is PURELY random.

    This class has the identical forward signature as TokenLevelGumbelRouter
    so it can be swapped in without changing any downstream code.

    Calibration: p_active = 1 - TARGET_SKIP matches DLR's target utilization.
    This ensures the comparison is at EQUAL compute budget, not equal accuracy.

    Scientific note: If exp25 (DLR) outperforms exp32 (Random) at the same
    p_active, the routing IS input-conditional. If not, the sparsity level
    alone explains DLR's results.
    """
    def __init__(self, p_active: float, num_layers: int):
        super().__init__()
        # Non-trainable parameter (buffer, not parameter)
        self.register_buffer("p_act", torch.tensor(float(p_active)))
        self.num_layers = num_layers
        # Dummy parameter so optimizer doesn't complain about empty param group
        # (LoRA params are the trainable components; router has none)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, h_seq: torch.Tensor, temperature: float = 1.0, hard: bool = True):
        """
        Sample random binary gates, INDEPENDENT of h_seq content.

        Args:
            h_seq: [B, S, H] hidden states (used only for shape, not content)
            temperature: Ignored (no learned logits to temper)
            hard: Ignored (always returns binary gates)

        Returns:
            gates: [B, S, L] binary float tensor with Bernoulli(p_active) values
        """
        B, S, _ = h_seq.shape
        # Sample B×S×L independent Bernoulli(p_active) gates
        # Use CUDA RNG for reproducibility under checkpoint restart
        prob = self.p_act.expand(B, S, self.num_layers)
        return torch.bernoulli(prob).to(h_seq.dtype)


# ==============================================================================
# Hook-based Token-Level Gated Forward Pass (identical to exp25)
# ==============================================================================
class TokenGatedForwardContext:
    """Minimal hook context for gated forward (no gradient computation needed)."""
    def __init__(self):
        self.gates = None
        self.handles = []
        self.captured_h_seq = None

    def __enter__(self): return self
    def __exit__(self, *args): self.remove_hooks()

    def remove_hooks(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def install_gate_hooks(self, layers, gates):
        self.gates = gates   # [B, S, L]
        for i, layer in enumerate(layers):
            def hook(module, input, output, layer_i=i):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h        = output[0] if is_tuple else output
                gate     = self.gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
                gated_h  = gate * h + (1.0 - gate) * residual
                if is_tuple:
                    return (gated_h,) + output[1:]
                return gated_h
            self.handles.append(layer.register_forward_hook(hook))


class StopForwardException(Exception):
    pass


def gated_forward(model, decoder_layers, batch, router, temperature=1.0, hard=True):
    """
    Two-pass gated forward with random router.
    Identical interface to exp25's gated_forward.
    """
    input_ids      = batch["input_ids"]
    labels         = batch.get("labels", None)
    attention_mask = batch.get("attention_mask", None)

    ctx = TokenGatedForwardContext()
    captured = {}

    def early_stop_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()
        raise StopForwardException()

    handle = decoder_layers[ALWAYS_KEEP - 1].register_forward_hook(early_stop_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForwardException:
        pass
    finally:
        handle.remove()

    h_seq = captured["h"].to("cuda")
    gates = router(h_seq, temperature=temperature, hard=hard)    # [B, S, L] random gates

    ctx.install_gate_hooks(decoder_layers[ALWAYS_KEEP:], gates)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finally:
        ctx.remove_hooks()

    return outputs.logits, outputs.loss, gates


# ==============================================================================
# Main
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXP32: Qwen2.5-7B Random Token Router Ablation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override SEED (for multi-seed runs)")
    parser.add_argument("--p_active", type=float, default=None,
                        help="Override P_ACTIVE (gate probability). Default: 1 - TARGET_SKIP")
    args = parser.parse_args()

    if args.seed is not None:
        global SEED
        SEED = args.seed
    p_active = args.p_active if args.p_active is not None else P_ACTIVE

    print(f"\n{'=' * 70}")
    print(f"  EXP32: QWEN2.5-7B RANDOM TOKEN ROUTER")
    print(f"  P_ACTIVE={p_active:.3f} (={100*p_active:.0f}% gates active, random)")
    print(f"  SEED={SEED} | Ablation: proving Learning > Random Sparsity")
    print(f"{'=' * 70}\n")

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Auth
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Dataset
    print("Loading dataset: wikitext-103-raw-v1 ...")
    raw      = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    eval_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw      = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
    eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        out["labels"] = out["input_ids"].copy()
        return out

    tok_procs = 1 if os.name == "nt" else min(os.cpu_count() or 1, 32)
    train_enc = raw.map(tokenize, batched=True, remove_columns=raw.column_names, num_proc=tok_procs)
    eval_enc  = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
    train_enc.set_format("torch")
    eval_enc.set_format("torch")

    class RAMDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            ids  = enc["input_ids"]
            mask = enc["attention_mask"]
            self.input_ids      = ids  if isinstance(ids,  torch.Tensor) else torch.stack(list(ids))
            self.attention_mask = mask if isinstance(mask, torch.Tensor) else torch.stack(list(mask))
            self.labels         = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx]}

    pin = torch.cuda.is_available()
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model + LoRA
    print(f"Loading {MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=COMPUTE_DTYPE,
        **( {"attn_implementation": ATTN_IMPL} if ATTN_IMPL else {} ),
        token=hf_token
    ).to("cuda")
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    try:
        decoder_layers = model.base_model.model.model.layers
    except (AttributeError, AssertionError):
        decoder_layers = model.model.layers

    TOTAL_LAYERS    = len(decoder_layers)
    ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP
    print(f"  Total layers: {TOTAL_LAYERS} | Routable: {ROUTABLE_LAYERS} | P_ACTIVE={p_active:.3f}")

    cfg = getattr(model, "config", None) or model.base_model.model.config
    flops_per_token_per_layer = estimate_layer_flops_per_token(cfg, MAX_LENGTH)
    baseline_gflops_per_step  = (TOTAL_LAYERS * flops_per_token_per_layer * MAX_LENGTH * BATCH_SIZE / 1e9)
    print(f"[FLOP] Baseline: {baseline_gflops_per_step:.1f} GFLOPs/step")

    # Random Router — attached to model but NOT trainable
    router = RandomBernoulliRouter(p_active=p_active, num_layers=ROUTABLE_LAYERS).to("cuda")
    for p in router.parameters():
        p.requires_grad = False    # Random router: NO gradient updates
    model.router = router
    print(f"[ROUTER] RandomBernoulliRouter | p_active={p_active:.3f} | NO learned parameters")

    # Optimizer (LoRA params only — router has no trainable params)
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    print(f"[LR] Cosine | warmup={WARMUP_STEPS} | total={total_steps}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    step_csv_header = [
        "epoch", "global_step", "optimizer_step",
        "train_loss", "val_loss", "perplexity",
        "active_layer_frac", "skip_ratio",
        "step_time_s", "tokens_per_sec",
        "peak_gpu_mem_gb", "lr",
        # Primary metric: exact token-layer utilization
        "utilization_pct", "exec_token_layer_pairs", "total_token_layer_pairs",
        "avg_active_layers",
        # Routing entropy (should be ~0.693 nats for Bernoulli(0.5); varies with p_active)
        "routing_entropy",
        # Projected FLOPs (theoretical; same methodology as exp25 for comparability)
        "projected_flop_reduction_pct", "baseline_gflops", "projected_executed_gflops",
        # Ablation metadata
        "p_active", "seed",
    ]
    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(step_csv_header)

    global_step    = 0
    optimizer_step = 0
    best_val_loss  = float("inf")
    last_train_loss = None
    oom_count = 0
    MAX_OOM_RETRIES = 5

    print(f"\nStarting exp32: Random Router training...")
    print(f"  Compare to exp25 (DLR) at same utilization={100*p_active:.0f}% active pairs\n")

    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for step, batch in enumerate(epoch_bar):
                global_step += 1
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

                torch.cuda.reset_peak_memory_stats()
                step_start = time.perf_counter()

                ce_loss_val = 0.0
                try:
                    _, ce_loss, gates = gated_forward(model, decoder_layers, batch,
                                                      router, temperature=1.0, hard=True)
                    ce_loss_val = ce_loss.item()
                    total_loss  = ce_loss / GRAD_ACCUM
                    total_loss.backward()
                    last_train_loss = ce_loss_val

                except torch.cuda.OutOfMemoryError as e:
                    oom_count += 1
                    print(f"\n[OOM] Step {step} (#{oom_count}/{MAX_OOM_RETRIES}). Skipping.")
                    import traceback
                    traceback.clear_frames(e.__traceback__)
                    e.__traceback__ = None
                    del e
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect(); torch.cuda.empty_cache()
                    if oom_count >= MAX_OOM_RETRIES:
                        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_oom"))
                        return
                    continue

                if global_step % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                step_time    = time.perf_counter() - step_start
                tokens_per_s = (BATCH_SIZE * MAX_LENGTH) / max(step_time, 1e-9)
                peak_mem_gb  = torch.cuda.max_memory_allocated() / (1024 ** 3)

                if global_step % LOG_EVERY_STEPS == 0 and last_train_loss is not None:
                    g_float    = gates.detach().float()
                    avg_layers = g_float.mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP
                    mean_g     = g_float.mean().item()
                    epoch_bar.set_postfix({
                        "loss":   f"{last_train_loss:.4f}",
                        "mem":    f"{peak_mem_gb:.1f}GB",
                        "tok/s":  f"{tokens_per_s:.0f}",
                        "layers": f"{avg_layers:.1f}",
                    })

                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    total_val_loss   = 0.0
                    total_active     = []
                    exec_pairs_eval  = 0
                    total_pairs_eval = 0

                    with torch.no_grad():
                        for val_batch in eval_loader:
                            val_batch = {k: v.to("cuda", non_blocking=True) for k, v in val_batch.items()}
                            _, v_ce, v_gates = gated_forward(model, decoder_layers, val_batch,
                                                              router, temperature=0.0, hard=True)
                            total_val_loss  += v_ce.item()
                            vg               = v_gates.detach().float()
                            total_active.append(vg.mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP)
                            exec_pairs_eval  += int(vg.sum().long().item())
                            total_pairs_eval += vg.numel()

                    val_loss       = total_val_loss / max(1, len(eval_loader))
                    val_avg_layers = sum(total_active) / max(1, len(total_active))
                    perplexity     = math.exp(min(val_loss, 300))

                    utilization_pct = 100.0 * exec_pairs_eval / max(1, total_pairs_eval)
                    actual_skip     = 1.0 - (exec_pairs_eval / max(1, total_pairs_eval))
                    active_frac     = val_avg_layers / TOTAL_LAYERS
                    proj_flop_red   = actual_skip * ROUTABLE_LAYERS / TOTAL_LAYERS * 100
                    proj_exec_gf    = baseline_gflops_per_step * (1.0 - proj_flop_red / 100)

                    # Routing entropy for random Bernoulli(p): H(p) = -p*log(p) - (1-p)*log(1-p)
                    p__ = max(1e-8, min(1.0 - 1e-8, p_active))
                    ent = -(p__ * math.log(p__) + (1 - p__) * math.log(1 - p__))

                    print(
                        f"\n[EVAL E{epoch+1} S{global_step}] "
                        f"val_loss={val_loss:.4f}  ppl={perplexity:.2f}  "
                        f"utilization={utilization_pct:.1f}%  "
                        f"avg_layers={val_avg_layers:.1f}/{TOTAL_LAYERS}  "
                        f"(random p_active={p_active:.2f})"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(SAVE_DIR, "best_model"), exist_ok=True)
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        # Save router config (not weights — it's random)
                        import json
                        with open(os.path.join(SAVE_DIR, "best_model", "random_router_config.json"), "w") as f:
                            json.dump({"p_active": p_active, "num_layers": ROUTABLE_LAYERS,
                                       "router_type": "RandomBernoulliRouter"}, f, indent=2)

                    current_lr = scheduler.get_last_lr()[0]
                    with open(CSV_FILENAME, "a", newline="") as f:
                        csv.writer(f).writerow([
                            epoch + 1, global_step, optimizer_step,
                            f"{last_train_loss:.6f}", f"{val_loss:.6f}", f"{perplexity:.4f}",
                            f"{active_frac:.4f}", f"{actual_skip:.4f}",
                            f"{step_time:.4f}", f"{tokens_per_s:.1f}",
                            f"{peak_mem_gb:.3f}", f"{current_lr:.2e}",
                            f"{utilization_pct:.4f}", exec_pairs_eval, total_pairs_eval,
                            f"{val_avg_layers:.4f}",
                            f"{ent:.6f}",
                            f"{proj_flop_red:.4f}", f"{baseline_gflops_per_step:.2f}", f"{proj_exec_gf:.2f}",
                            f"{p_active:.4f}", SEED,
                        ])
                    model.train()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_interrupt"))
        return
    except Exception as e:
        print(f"\n[ERROR] {e}. Saving emergency checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_error"))
        raise

    print("\nSaving final model...")
    os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

    print(f"\n{'=' * 70}")
    print(f"  EXP32 complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Compare results to exp25 (DLR) at same p_active={p_active:.3f}")
    print(f"  If exp25 PPL < exp32 PPL: routing IS input-conditional (paper claim holds)")
    print(f"  CSV: {CSV_FILENAME}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

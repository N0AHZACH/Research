import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
exp25_qwen7b_token_routing.py - Phase 4: Scaling to Larger Models

Validating the DLR framework on Qwen2.5-7B.
Auto-detects GPU hardware and configures batch size, quantization, and precision.
Includes robust checkpointing and CUDA OOM recovery for cloud environments.

Methodological Parity Fixes Applied (matching exp24 / exp23):
- Reproducibility seeds (numpy, torch, random)
- Gradient Checkpointing (use_reentrant=False)
- LoRA Targets expanded to MLP (gate, up, down)
- Cosine LR Schedule with 100-step warmup
- Operator precedence bug fix in tuple returns
- Throughput and FLOP reduction metrics logging
- OOM loss and step-sync fixes
"""
import csv
import gc
import math
import os
import itertools
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==============================================================================
# Configuration
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
COMPUTE_PENALTY  = 0.05   # Mathematically optimal penalty found via exp30 Pareto sweep
TARGET_SKIP      = 0.40   # Target: ~60% active layers
TARGET_PENALTY   = 0.5    # Reduced attractor to match new loss scale
GUMBEL_TEMP      = 1.0
TEMP_ANNEAL_RATE = 0.95
KD_ALPHA         = 0.3    # Give more weight to KD, less to raw CE
KD_TEMPERATURE   = 2.0
GATE_ENTROPY_BETA = 0.0   # Disabled: counteracts compute penalty
KD_WARMUP_STEPS  = 50

# Match baseline logging cadence
EVAL_EVERY_STEPS = 50
LOG_EVERY_STEPS  = 10

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
]
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

SEED = 42

# ---------------------------------------------------------------------------
# Hardware Auto-Optimisation
# ---------------------------------------------------------------------------
def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, False, torch.float32

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    if vram_gb >= 80: bs, ga = 16, 1
    elif vram_gb >= 45: bs, ga = 8, 2   # 48GB cards like RTX 6000 Pro
    elif vram_gb >= 35: bs, ga = 8, 2   # A100 40GB
    elif vram_gb >= 22: bs, ga = 4, 4   # RTX 3090/4090 24GB
    elif vram_gb >= 14: bs, ga = 2, 8   # T4 16GB
    else: bs, ga = 1, 16
    use_4bit = False

    major, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.float16 if major < 8 else torch.bfloat16

    nw = 0  # RAMDataset is fully in-memory
    try:
        import flash_attn
        attn = "flash_attention_2" if vram_gb >= 7 else None
    except ImportError:
        attn = "sdpa" if vram_gb >= 7 else None
    print(f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f}GB VRAM | BS={bs}, GA={ga}, 4bit={use_4bit}, dtype={compute_dtype}, workers={nw} | attn={attn}")
    return bs, ga, nw, attn, use_4bit, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp25_token_routing_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp25_qwen7b_token_output_{TIMESTAMP}"

# ==============================================================================
# FLOP Accounting
# ==============================================================================
def estimate_layer_flops_per_token(config, seq_len: int) -> float:
    H       = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv    = getattr(config, "num_key_value_heads", n_heads)
    hd      = H // n_heads
    ff      = config.intermediate_size

    q_flops = 2 * H * H
    k_flops = 2 * H * (n_kv * hd)
    v_flops = 2 * H * (n_kv * hd)
    o_flops = 2 * H * H

    attn_score_flops = 2 * seq_len * H
    attn_value_flops = 2 * seq_len * H

    gate_flops = 2 * H * ff
    up_flops   = 2 * H * ff
    down_flops = 2 * ff * H

    return (
        q_flops + k_flops + v_flops + o_flops
        + attn_score_flops + attn_value_flops
        + gate_flops + up_flops + down_flops
    )

# ==============================================================================
# TOKEN-LEVEL Gumbel-Softmax Router
# ==============================================================================
class TokenLevelGumbelRouter(nn.Module):
    """
    Per-TOKEN router using Gumbel-Softmax Straight-Through Estimator.
    Input: unpooled hidden state sequence from layer ALWAYS_KEEP. [B, S, H]
    Output: [B, S, ROUTABLE_LAYERS] binary gates.
    """
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            # BUG-12 fix: previous init (linspace 1.0→-3.0) caused deep layers to
            # start at gate≈0.05 (95% skip), creating a massive gradient from the
            # TARGET_SKIP attractor and causing training instability in epoch 1.
            # Neutral init: all logits near 0 → ~50% activity at start, small
            # gradient from attractor, stable convergence.
            nn.init.zeros_(last_layer.bias)
            nn.init.normal_(last_layer.weight, std=0.02)

    def forward(self, h_seq: torch.Tensor, temperature: float, hard: bool = True):
        h_seq  = h_seq.float()
        logits = self.net(h_seq)
        if self.training:
            logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            soft   = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
            return soft[..., 1].to(h_seq.dtype)
        if hard:
            return (logits > 0).to(h_seq.dtype)
        return torch.sigmoid(logits).to(h_seq.dtype)

# ==============================================================================
# Hook-based Token-Level Gated Forward Pass
# ==============================================================================
class TokenGatedForwardContext:
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
        self.gates = gates  # [B, S, L]
        for i, layer in enumerate(layers):
            idx = i
            def hook(module, input, output, layer_i=idx):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                
                assert self.gates is not None
                gate = self.gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
                gated_h = gate * h + (1.0 - gate) * residual
                
                # Fixed operator precedence bug
                if is_tuple:
                    return (gated_h,) + output[1:]
                return gated_h

            self.handles.append(layer.register_forward_hook(hook))

def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Phase 5: Qwen2.5-7B Token-Level Routing")
    parser.add_argument("--resume", type=str, default="auto",
                        help="Path to checkpoint directory or 'auto'")
    parser.add_argument("--fresh", action="store_true",
                        help="Start training from scratch")
    parser.add_argument("--no_kd", action="store_true",
                        help="Disable knowledge distillation (ablation study). "
                             "When set, training uses only CE + gate loss. "
                             "Required experiment for METH-04 confound removal.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override global SEED (for multi-seed runs via exp36).")
    args = parser.parse_args()

    if args.fresh:
        args.resume = "none"
    if args.seed is not None:
        global SEED
        SEED = args.seed
    if args.no_kd:
        print("[ABLATION] --no_kd: Knowledge distillation DISABLED. "
              "Training with CE + gate loss only.")

    csv_filename = CSV_FILENAME
    save_dir = SAVE_DIR

    start_epoch = 0
    start_step = -1
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")
    current_temp = GUMBEL_TEMP

    def find_latest_checkpoint():
        dirs = glob.glob("exp25_qwen7b_token_output_*")
        valid_checkpoints = []
        for d in dirs:
            ckpt_path = os.path.join(d, "checkpoint_latest", "training_states.pt")
            if os.path.exists(ckpt_path):
                mtime = os.path.getmtime(ckpt_path)
                valid_checkpoints.append((mtime, d))
        if not valid_checkpoints:
            return None
        valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
        return valid_checkpoints[0][1]

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── HuggingFace Auth ──────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # ==============================================================================
    # Dataset - Wikitext-103
    # ==============================================================================
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
            self.labels = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    pin = torch.cuda.is_available()
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    print(f"  Train: {len(train_enc)} | Eval: {len(eval_enc)}")

    # ==============================================================================
    # Models: Student (LoRA) + Teacher (Frozen) for KD
    # ==============================================================================
    print(f"\nLoading {MODEL_ID} student (LoRA) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=COMPUTE_DTYPE, 
        **( {"attn_implementation": ATTN_IMPL} if ATTN_IMPL else {} ),
        token=hf_token
    ).to("cuda")  
    base_model.config.use_cache = False

    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base_model.enable_input_require_grads()
    print("[GC] Gradient checkpointing enabled (use_reentrant=False)")

    lora_cfg = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    try:
        decoder_layers = model.base_model.model.model.layers
    except (AttributeError, AssertionError):
        decoder_layers = model.model.layers

    TOTAL_LAYERS    = len(decoder_layers)
    assert TOTAL_LAYERS == 28, f"Expected 28 layers for Qwen2.5-7B, got {TOTAL_LAYERS}"
    ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP
    print(f"  Total layers: {TOTAL_LAYERS} | Always-kept: {ALWAYS_KEEP} | Routable: {ROUTABLE_LAYERS}")

    cfg = getattr(model, "config", None) or model.base_model.model.config
    flops_per_token_per_layer = estimate_layer_flops_per_token(cfg, MAX_LENGTH)
    baseline_gflops_per_step  = (
        TOTAL_LAYERS * flops_per_token_per_layer * MAX_LENGTH * BATCH_SIZE / 1e9
    )
    print(f"[FLOP] Baseline: {baseline_gflops_per_step:.1f} GFLOPs/step")

    hidden_size = int(getattr(model.config, "hidden_size"))
    model.router = TokenLevelGumbelRouter(hidden_size, ROUTABLE_LAYERS).to("cuda")
    for p in model.router.parameters():
        p.requires_grad = True

    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    
    total_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    print(f"[LR] Cosine schedule | warmup={WARMUP_STEPS} steps | total={total_steps} steps")

    # Load checkpoint
    checkpoint_dir = None
    if args.resume == "auto":
        latest_dir = find_latest_checkpoint()
        if latest_dir:
            checkpoint_dir = os.path.join(latest_dir, "checkpoint_latest")
            save_dir = latest_dir
            csv_filename = f"{latest_dir.replace('exp25_qwen7b_token_output_', 'exp25_token_routing_metrics_')}.csv"
    elif args.resume and args.resume.lower() != "none":
        checkpoint_dir = args.resume
        parent_dir = os.path.dirname(checkpoint_dir)
        if parent_dir:
            save_dir = parent_dir
            csv_filename = f"{parent_dir.replace('exp25_qwen7b_token_output_', 'exp25_token_routing_metrics_')}.csv"
        else:
            save_dir = checkpoint_dir

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"\n[CHECKPOINT] Loading checkpoint from: {checkpoint_dir}")
        lora_loaded = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapters_weights = load_file(fpath)
                else:
                    adapters_weights = torch.load(fpath, map_location="cuda")
                model.load_state_dict(adapters_weights, strict=False)
                del adapters_weights
                gc.collect()
                lora_loaded = True
                print(f"  Loaded LoRA adapters from {fpath}")
                break

        router_path = os.path.join(checkpoint_dir, "router_weights.pt")
        if os.path.exists(router_path):
            model.router.load_state_dict(torch.load(router_path, map_location="cuda"))
            print(f"  Loaded router weights from {router_path}")

        states_path = os.path.join(checkpoint_dir, "training_states.pt")
        if os.path.exists(states_path):
            states = torch.load(states_path, map_location="cuda")
            start_epoch = states["epoch"]
            start_step = states["step"]
            if start_step < 0:
                start_epoch += 1
            global_step = states["global_step"]
            optimizer_step = states.get("optimizer_step", global_step // GRAD_ACCUM)
            best_val_loss = states["best_val_loss"]
            current_temp = states["current_temp"]
            if "csv_filename" in states:
                csv_filename = states["csv_filename"]
            
            optimizer.load_state_dict(states["optimizer_state_dict"])
            scheduler.load_state_dict(states["scheduler_state_dict"])
            print(f"  Resumed training states: epoch={start_epoch+1}, step={start_step}, global_step={global_step}, temp={current_temp:.4f}")

    class StopForwardException(Exception): pass

    def gated_forward(model, batch, temperature, hard=True):
        input_ids      = batch["input_ids"]
        labels         = batch.get("labels", None)
        attention_mask = batch.get("attention_mask", None)
        
        all_layers = decoder_layers

        ctx = TokenGatedForwardContext()
        
        def early_stop_hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            ctx.captured_h_seq = hidden_state.detach().float()
            raise StopForwardException()

        handle = all_layers[ALWAYS_KEEP - 1].register_forward_hook(early_stop_hook)
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        except StopForwardException:
            pass
        finally:
            handle.remove()

        assert ctx.captured_h_seq is not None
        h_seq = ctx.captured_h_seq.to("cuda")                             
        gates = model.router(h_seq, temperature=temperature, hard=hard)   

        ctx.install_gate_hooks(all_layers[ALWAYS_KEEP:], gates)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            ctx.remove_hooks()

        return outputs.logits, outputs.loss, gates

    def compute_kd_loss(s_logits, t_logits, T, mask):
        s_logits = s_logits.reshape(-1, s_logits.size(-1))
        t_logits = t_logits.reshape(-1, t_logits.size(-1))
        mask = mask.reshape(-1)
        
        chunk_size = 1024
        kl_sum = 0.0
        for i in range(0, s_logits.size(0), chunk_size):
            s_chunk = s_logits[i:i+chunk_size]
            t_chunk = t_logits[i:i+chunk_size]
            m_chunk = mask[i:i+chunk_size].float()
            
            kl = F.kl_div(
                F.log_softmax(s_chunk / T, dim=-1),
                F.softmax(t_chunk / T, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            
            kl_sum = kl_sum + (kl * m_chunk).sum() * (T ** 2)
            
        return kl_sum / mask.sum().clamp(min=1.0)

    def compute_gate_loss(gates):
        per_layer_activity = gates.float().mean(dim=(0, 1))  
        L = per_layer_activity.size(0)
        depth_weights = torch.linspace(0.1, 2.0, steps=L, device=gates.device)
        l1_penalty = (per_layer_activity * depth_weights).sum() * COMPUTE_PENALTY
        
        actual_skip = 1.0 - gates.float().mean()
        target_penalty = TARGET_PENALTY * (actual_skip - TARGET_SKIP) ** 2
        return l1_penalty + target_penalty, per_layer_activity

    def save_checkpoint(epoch, step, global_step, best_val_loss, current_temp):
        checkpoint_dir = os.path.join(save_dir, "checkpoint_latest")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(model.router.state_dict(), os.path.join(checkpoint_dir, "router_weights.pt"))
        checkpoint_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "best_val_loss": best_val_loss,
            "current_temp": current_temp,
            "csv_filename": csv_filename,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint_states, os.path.join(checkpoint_dir, "training_states.pt"))
        print(f"\n[CHECKPOINT] Saved checkpoint to {checkpoint_dir} at epoch {epoch+1}, step {step}, global_step {global_step}")

    # ==============================================================================
    # Training Loop
    # ==============================================================================
    print(f"\nStarting Phase 4: Scaled Token-Level Router training on {MODEL_ID}...")
    print(f"  COMPUTE_PENALTY={COMPUTE_PENALTY} | TARGET_SKIP={TARGET_SKIP} | TARGET_PENALTY={TARGET_PENALTY}")
    print(f"  KD_ALPHA={KD_ALPHA} | Init bias=-1.5\n")

    step_csv_header = [
        "epoch", "global_step", "optimizer_step",
        "train_loss", "ce_loss", "kd_loss", "gate_loss", "target_loss",
        "val_loss", "perplexity",
        "active_layer_frac", "skip_ratio",
        "step_time_s", "tokens_per_sec",
        "peak_gpu_mem_gb", "lr", "gumbel_temp",
        # ── Primary metric: exact token-layer utilization (no approximation) ──
        "utilization_pct",              # 100 * executed_pairs / total_pairs
        "exec_token_layer_pairs",       # exact integer count
        "total_token_layer_pairs",      # B * S * L_routable
        "avg_active_layers",            # always_keep + mean active routable
        # ── Router health metrics ─────────────────────────────────────────────
        "routing_entropy",              # H(p) nats; 0=collapsed, 0.693=max
        "gate_variance",                # Var(gates); 0=collapsed
        "utilization_variance",         # Var(per-layer util); high=unbalanced
        # ── Projected FLOPs (theoretical; label as projected in paper) ────────
        # IMPORTANT: Current hook implementation achieves 0% real FLOP reduction.
        # These numbers assume perfect token-packing (future hardware target).
        "projected_flop_reduction_pct",
        "baseline_gflops",
        "projected_executed_gflops",
        # ── Ablation flag ─────────────────────────────────────────────────────
        "kd_enabled",
    ]
    
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
        with open(csv_filename, "w", newline="") as f:
            csv.writer(f).writerow(step_csv_header)

    oom_count = 0
    MAX_OOM_RETRIES = 5
    last_train_loss = None

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for step, batch in enumerate(epoch_bar):
                if step <= start_step:
                    continue

                global_step += 1
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
                
                torch.cuda.reset_peak_memory_stats()
                step_start = time.perf_counter()

                # BUG-02 fix: initialize cached scalars BEFORE try block so
                # they are always defined even if OOM occurs before assignment.
                ce_loss_val = gate_loss_val = kd_loss_val = 0.0

                try:
                    student_logits, ce_loss, gates = gated_forward(model, batch, temperature=current_temp, hard=True)
                    gate_loss, per_layer_activity = compute_gate_loss(gates)

                    # BUG-02 fix: cache .item() values immediately after computation,
                    # before any del statements or backward() call. The pattern
                    # `tensor.item() if 'tensor' in locals()` is unreliable — Python
                    # locals() always contains the name even after del; the tensor
                    # backing store may be freed after backward().
                    ce_loss_val   = ce_loss.item()
                    gate_loss_val = gate_loss.item()

                    use_kd = (not args.no_kd) and (global_step >= KD_WARMUP_STEPS)

                    if not use_kd:
                        kd_loss      = torch.tensor(0.0, device="cuda")
                        kd_loss_val  = 0.0
                        total_loss   = (ce_loss + gate_loss) / GRAD_ACCUM
                    else:
                        with torch.no_grad():
                            with model.disable_adapter():
                                # Teacher: pretrained backbone (LoRA disabled, no routing hooks).
                                # Design choice: student learns routing while matching
                                # unrouted pretrained distribution. See METH-04 in audit.
                                teacher_logits = model(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch.get("attention_mask"),
                                ).logits
                        kd_loss     = compute_kd_loss(
                            student_logits[:, :-1, :], teacher_logits[:, :-1, :],
                            KD_TEMPERATURE, batch["attention_mask"][:, 1:]
                        )
                        kd_loss_val = kd_loss.item()
                        total_loss  = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss) / GRAD_ACCUM

                        del student_logits, teacher_logits

                    total_loss.backward()
                    last_train_loss = total_loss.item() * GRAD_ACCUM

                except torch.cuda.OutOfMemoryError as e:
                    oom_count += 1
                    print(f"\n[OOM] CUDA OOM on step {step} (occurrence {oom_count}/{MAX_OOM_RETRIES}). Clearing cache and skipping batch...")
                    import traceback
                    traceback.clear_frames(e.__traceback__)
                    e.__traceback__ = None
                    del e
                    if 'student_logits' in locals(): del student_logits
                    if 'teacher_logits' in locals(): del teacher_logits
                    if 'ce_loss' in locals(): del ce_loss
                    if 'gates' in locals(): del gates
                    if 'total_loss' in locals(): del total_loss
                    if 'kd_loss' in locals(): del kd_loss
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if oom_count >= MAX_OOM_RETRIES:
                        print("[OOM] Too many OOM errors. Saving checkpoint and exiting.")
                        save_checkpoint(epoch, step, global_step, best_val_loss, current_temp)
                        return
                    continue

                if global_step % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    
                    if global_step % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                step_time    = time.perf_counter() - step_start
                tokens_per_s = (BATCH_SIZE * MAX_LENGTH) / max(step_time, 1e-9)
                peak_mem_gb  = torch.cuda.max_memory_allocated() / (1024 ** 3)

                if global_step % LOG_EVERY_STEPS == 0 and last_train_loss is not None:
                    g_float    = gates.detach().float()
                    avg_layers = g_float.mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP
                    skip_ratio = 1.0 - g_float.mean().item()

                    # ── Router collapse detection (inline; no sys.path needed) ──
                    mean_g     = g_float.mean().item()
                    eps        = 1e-8
                    p_         = max(eps, min(1.0 - eps, mean_g))
                    train_entropy = -(p_ * math.log(p_) + (1 - p_) * math.log(1 - p_))

                    if mean_g < 0.02:
                        print(f"\n[COLLAPSE:ALL_ZERO] Step {global_step}: "
                              f"mean_gate={mean_g:.4f}. Router skipping >98% of pairs. "
                              f"Reduce COMPUTE_PENALTY or increase TARGET_SKIP.",
                              flush=True)
                    elif mean_g > 0.98:
                        print(f"\n[COLLAPSE:ALL_ONE] Step {global_step}: "
                              f"mean_gate={mean_g:.4f}. Router nearly dense. "
                              f"Increase COMPUTE_PENALTY.", flush=True)

                    per_layer_g  = g_float.mean(dim=(0, 1))
                    dead_count   = int((per_layer_g < 0.01).sum().item())
                    if dead_count > 0:
                        dead_idx = [ALWAYS_KEEP + i for i, v in
                                    enumerate(per_layer_g.tolist()) if v < 0.01]
                        print(f"\n[COLLAPSE:DEAD_LAYER] Step {global_step}: "
                              f"{dead_count} dead layer(s) at {dead_idx}.", flush=True)

                    epoch_bar.set_postfix({
                        "loss":    f"{last_train_loss:.4f}",
                        "mem":     f"{peak_mem_gb:.1f}GB",
                        "tok/s":   f"{tokens_per_s:.0f}",
                        "layers":  f"{avg_layers:.1f}",
                        "H(gate)": f"{train_entropy:.3f}",
                    })

                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    total_val_loss  = 0.0
                    total_active    = []

                    # BUG-01 fix: accumulate EVAL-TIME gates, not training-time gates.
                    # Training uses Gumbel temperature; eval uses hard threshold → differ.
                    val_gates_list  = []

                    # Exact token-layer utilization counters
                    exec_pairs_eval  = 0
                    total_pairs_eval = 0

                    # BUG-11 / METH-02 fix: use temperature=0.0 for inference.
                    # Hard threshold on logits (logits > 0) is the correct eval mode.
                    # Avoids stochasticity from non-zero temperature at eval time.
                    EVAL_TEMP = 0.0

                    with torch.no_grad():
                        for i, val_batch in enumerate(eval_loader):
                            val_batch = {k: v.to("cuda", non_blocking=True) for k, v in val_batch.items()}
                            _, v_ce, v_gates = gated_forward(model, val_batch, temperature=EVAL_TEMP, hard=True)
                            total_val_loss += v_ce.item()

                            vg = v_gates.detach().float()   # [B, S, L]
                            val_gates_list.append(vg)
                            total_active.append(vg.mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP)

                            # Exact token-layer pair counting (primary metric)
                            exec_pairs_eval  += int(vg.sum().long().item())
                            total_pairs_eval += vg.numel()

                    val_loss       = total_val_loss / max(1, len(eval_loader))
                    val_avg_layers = sum(total_active) / max(1, len(total_active))
                    perplexity     = math.exp(min(val_loss, 300))

                    # BUG-01 fix: compute actual_skip from EVAL gates
                    all_vg          = torch.cat([g.reshape(-1) for g in val_gates_list])
                    actual_skip     = 1.0 - all_vg.mean().item()
                    del val_gates_list, all_vg

                    # Primary metric: exact utilization
                    utilization_pct = 100.0 * exec_pairs_eval / max(1, total_pairs_eval)

                    # Router health diagnostics on most recent eval batch
                    last_vg   = v_gates.detach().float()
                    mean_g    = last_vg.mean().item()
                    eps_      = 1e-8
                    p__       = max(eps_, min(1.0 - eps_, mean_g))
                    eval_ent  = -(p__ * math.log(p__) + (1 - p__) * math.log(1 - p__))
                    gate_var  = last_vg.var().item()
                    per_lay   = last_vg.mean(dim=(0, 1))
                    util_var  = per_lay.var().item()

                    target_loss_val   = TARGET_PENALTY * (actual_skip - TARGET_SKIP) ** 2
                    active_layer_frac = val_avg_layers / TOTAL_LAYERS

                    # Projected FLOP reduction (theoretical; see paper methodology note)
                    projected_flop_reduction_pct = (actual_skip * ROUTABLE_LAYERS / TOTAL_LAYERS) * 100
                    projected_exec_gflops = baseline_gflops_per_step * (1.0 - projected_flop_reduction_pct / 100)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(save_dir, "best_model"), exist_ok=True)
                        model.save_pretrained(os.path.join(save_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
                        torch.save(model.router.state_dict(), os.path.join(save_dir, "best_model", "router_weights.pt"))

                    save_checkpoint(epoch, step, global_step, best_val_loss, current_temp)
                    model.train()

                    print(
                        f"\n[EVAL E{epoch+1} S{global_step}] "
                        f"val_loss={val_loss:.4f}  ppl={perplexity:.2f}  "
                        f"utilization={utilization_pct:.1f}%  "
                        f"avg_layers={val_avg_layers:.1f}/{TOTAL_LAYERS}  "
                        f"H(gate)={eval_ent:.3f}nats  "
                        f"skip={actual_skip*100:.1f}%  "
                        f"proj_FLOP↓={projected_flop_reduction_pct:.1f}%"
                    )

                    with open(csv_filename, "a", newline="") as f:
                        csv.writer(f).writerow([
                            epoch + 1,
                            global_step,
                            optimizer_step,
                            f"{last_train_loss:.6f}",
                            # BUG-02 fix: use pre-cached scalar values
                            f"{ce_loss_val:.4f}",
                            f"{kd_loss_val:.4f}",
                            f"{gate_loss_val:.4f}",
                            f"{target_loss_val:.4f}",
                            f"{val_loss:.6f}",
                            f"{perplexity:.4f}",
                            f"{active_layer_frac:.4f}",
                            f"{actual_skip:.4f}",
                            f"{step_time:.4f}",
                            f"{tokens_per_s:.1f}",
                            f"{peak_mem_gb:.3f}",
                            f"{scheduler.get_last_lr()[0]:.2e}",
                            f"{current_temp:.4f}",
                            # Primary metric: exact token-layer utilization
                            f"{utilization_pct:.4f}",
                            exec_pairs_eval,
                            total_pairs_eval,
                            f"{val_avg_layers:.4f}",
                            # Router health
                            f"{eval_ent:.6f}",
                            f"{gate_var:.6f}",
                            f"{util_var:.6f}",
                            # Projected FLOPs (theoretical)
                            f"{projected_flop_reduction_pct:.4f}",
                            f"{baseline_gflops_per_step:.2f}",
                            f"{projected_exec_gflops:.2f}",
                            # Ablation flag
                            int(not args.no_kd),
                        ])

            start_step = -1
            save_checkpoint(epoch, -1, global_step, best_val_loss, current_temp)
            current_temp = max(0.5, current_temp * TEMP_ANNEAL_RATE)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user. Saving checkpoint...")
        curr_step = step if 'step' in locals() else -1
        curr_epoch = epoch if 'epoch' in locals() else start_epoch
        save_checkpoint(curr_epoch, curr_step, global_step, best_val_loss, current_temp)
        print("Checkpoint saved successfully. Exiting.")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}. Saving emergency checkpoint...")
        try:
            curr_step = step if 'step' in locals() else -1
            curr_epoch = epoch if 'epoch' in locals() else start_epoch
            save_checkpoint(curr_epoch, curr_step, global_step, best_val_loss, current_temp)
            print("Emergency checkpoint saved.")
        except Exception as save_err:
            print(f"Failed to save emergency checkpoint: {save_err}")
        raise

    print("\nSaving final model checkpoint...")
    os.makedirs(os.path.join(save_dir, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))
    torch.save(model.router.state_dict(), os.path.join(save_dir, "final_model", "router_weights.pt"))

    checkpoint_dir = os.path.join(save_dir, "checkpoint_latest")
    if os.path.exists(checkpoint_dir):
        import shutil
        try:
            shutil.rmtree(checkpoint_dir)
            print("Cleaned up checkpoint folder.")
        except Exception as e:
            print(f"Warning: could not clean up checkpoint folder: {e}")

    print("Done!")

if __name__ == '__main__':
    main()

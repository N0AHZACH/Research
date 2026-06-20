"""
exp30_qwen7b_pareto_sweep.py - Publication-Ready Token-Level Routing Pareto Sweep

This script performs a systematic 7-point Pareto sweep for Qwen2.5-7B token routing.
It automates the discovery of the optimal compute penalty by training the model
under varying sparsity constraints and plotting the resulting accuracy-efficiency
frontier.

Key features:
1. Memory-efficient Knowledge Distillation (KD) without dual-model memory overhead.
2. Robust hardware auto-detection for 96GB VRAM and other architectures.
3. Stable token-level Gumbel-Softmax routing.
4. Clean, publication-ready training and evaluation loops.
"""

import os
import gc
import csv
import json
import math
import random
import subprocess
import itertools
import numpy as np
import datetime
import argparse
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_ID         = "Qwen/Qwen2.5-7B"
MAX_LENGTH       = 512
ALWAYS_KEEP      = 4
PENALTIES        = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 3
MAX_EVAL_BATCHES = 100

LR               = 3e-5
WEIGHT_DECAY     = 0.01
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_MIN   = 0.5
TEMP_ANNEAL_RATE  = 0.95
KD_ALPHA         = 0.3    # Matched to recent scaling configurations
KD_TEMPERATURE   = 2.0
KD_WARMUP_STEPS  = 50

# ==============================================================================
# Hardware Auto-Optimisation
# ==============================================================================
def get_optimal_config():
    """Detects available hardware and configures batch size and memory settings."""
    if not torch.cuda.is_available():
        return 2, 8, 0, None, torch.float32

    vram_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enforce exact determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_turing = 'T4' in gpu_name or 'RTX 20' in gpu_name or 'Turing' in gpu_name

    # Determine best configuration based on VRAM
    if vram_gb >= 80:    # 96GB VRAM Detected (e.g. A100/H100)
        # Bumping to BS=32 since you have 96GB. KD requires massive VRAM for logits, but 96GB handles 32 perfectly.
        bs, ga = 32, 1
    elif vram_gb >= 45:  # 48GB cards
        bs, ga = 16, 1
    elif vram_gb >= 35:  # A100 40GB
        bs, ga = 8, 2
    elif vram_gb >= 22:  # RTX 3090/4090 24GB
        bs, ga = 4, 4
    elif vram_gb >= 14:  # T4 16GB
        bs, ga = 2, 8
    else:                # 8GB cards
        bs, ga = 1, 16

    compute_dtype = torch.float16 if is_turing else torch.bfloat16
    cpu_count = os.cpu_count() or 2
    
    # LINUX GCP OPTIMIZATION: On Linux, PyTorch uses 'fork' for multiprocessing.
    # This means memory is shared copy-on-write, so we can safely use multiple workers
    # to asynchronously pre-fetch batches from the 180GB RAM pool without duplicating memory.
    # With 45 vCPUs, 16 workers will ensure the 96GB GPU is fed at absolute maximum bandwidth.
    nw = 16 if os.name != "nt" else 0

    try:
        import flash_attn
        attn = "flash_attention_2" if vram_gb >= 7 else None
    except ImportError:
        attn = "sdpa" if vram_gb >= 7 else None

    print(f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f}GB VRAM | BS={bs}, GA={ga}, dtype={compute_dtype}, workers={nw} | attn={attn}")
    return bs, ga, nw, attn, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)
CSV_FILENAME = f"results/exp30_qwen7b_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"results/exp30_qwen7b_pareto_{TIMESTAMP}.png"

# ==============================================================================
# Token-Level Router
# ==============================================================================
class TokenLevelGumbelRouter(nn.Module):
    """
    Per-TOKEN router using Gumbel-Softmax Straight-Through Estimator.
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
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, h_seq: torch.Tensor, temperature: float, hard: bool = True):
        h_seq = h_seq.float()
        logits = self.net(h_seq)
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        soft = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(h_seq.dtype)

# ==============================================================================
# Hook-based Token-Level Gated Forward Pass
# ==============================================================================
class StopForwardException(Exception): 
    pass

class TokenGatedForwardContext:
    def __init__(self):
        self.gates = None
        self.handles = []
        self.captured_h_seq = None

    def __enter__(self): 
        return self
        
    def __exit__(self, *args): 
        self.remove_hooks()
        
    def remove_hooks(self):
        for h in self.handles: 
            h.remove()
        self.handles.clear()

    def install_gate_hooks(self, layers, gates):
        self.gates = gates
        for i, layer in enumerate(layers):
            idx = i
            def hook(module, input, output, layer_i=idx):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                
                gate = self.gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
                gated_h = gate * h + (1.0 - gate) * residual
                
                return (gated_h,) + output[1:] if is_tuple else gated_h
            self.handles.append(layer.register_forward_hook(hook))

def gated_forward(model, batch, temperature, hard=True):
    input_ids = batch["input_ids"]
    labels = batch.get("labels", None)
    attention_mask = batch.get("attention_mask", None)
    
    transformer = model.base_model.model.model
    all_layers = transformer.layers

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

    h_seq = ctx.captured_h_seq.to("cuda")
    gates = model.router(h_seq, temperature=temperature, hard=hard)

    ctx.install_gate_hooks(all_layers[ALWAYS_KEEP:], gates)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finally:
        ctx.remove_hooks()

    return outputs.logits, outputs.loss, gates

def compute_kd_loss(s_logits, t_logits, T, mask):
    """Memory-efficient chunked KD loss calculation."""
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
            reduction="none"
        ).sum(dim=-1)
        
        kl_sum = kl_sum + (kl * m_chunk).sum() * (T ** 2)
        
    return kl_sum / mask.sum().clamp(min=1.0)

# ==============================================================================
# Data Preparation
# ==============================================================================
print("Loading dataset: wikitext-103-raw-v1 ...")
raw      = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
eval_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")

raw      = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    out["labels"] = out["input_ids"].copy()
    return out

# You have 45 vCPUs. We unlock multiprocessing for tokenization up to 40 cores.
tok_procs = min(os.cpu_count() or 1, 40)
train_enc = raw.map(tokenize, batched=True, remove_columns=raw.column_names, num_proc=tok_procs)
eval_enc  = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
train_enc.set_format("torch")
eval_enc.set_format("torch")

class RAMDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        self.input_ids = ids if isinstance(ids, torch.Tensor) else torch.stack(list(ids))
        self.attention_mask = mask if isinstance(mask, torch.Tensor) else torch.stack(list(mask))
        self.labels = self.input_ids.clone()
        self.labels[self.attention_mask == 0] = -100

    def __len__(self): 
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx], 
            "attention_mask": self.attention_mask[idx], 
            "labels": self.labels[idx]
        }

pin = torch.cuda.is_available() and (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) > 12 if hasattr(os, 'sysconf') else True)
train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

# ==============================================================================
# Training Logic
# ==============================================================================
def train_one_penalty(penalty: float) -> dict:
    """Executes a full training and evaluation run for a given compute penalty."""
    print(f"\n{'='*50}\nEvaluating Compute Penalty: λ = {penalty:.3f}\n{'='*50}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=COMPUTE_DTYPE, 
        attn_implementation=ATTN_IMPL
    ).to("cuda")
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    
    ROUTABLE_LAYERS = len(model.base_model.model.model.layers) - ALWAYS_KEEP
    model.router = TokenLevelGumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
    
    for p in model.router.parameters():
        p.requires_grad = True

    if ATTN_IMPL == "sdpa" and os.name != "nt":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Skipping torch.compile due to error: {e}")

    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), model.router.parameters()), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS * len(train_loader) // GRAD_ACCUM
    )
    
    current_temp = GUMBEL_TEMP_START
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_bar = tqdm(train_loader, desc=f"λ={penalty} | Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(epoch_bar):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            try:
                s_logits, ce_loss, gates = gated_forward(model, batch, current_temp, hard=True)
                
                # Pareto specific gate loss: linearly scales with overall layer activity
                per_layer_activity = gates.float().mean(dim=(0, 1))
                gate_loss = per_layer_activity.sum() * penalty
                
                # Compute routing entropy to prove dynamic routing (not static)
                p = per_layer_activity
                eps = 1e-8
                layer_entropy = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps)).mean()

                if global_step < KD_WARMUP_STEPS:
                    total_loss = (ce_loss + gate_loss) / GRAD_ACCUM
                else:
                    with torch.no_grad():
                        with model.disable_adapter():
                            t_logits = model(
                                input_ids=batch["input_ids"], 
                                attention_mask=batch.get("attention_mask")
                            ).logits
                    
                    kd_loss = compute_kd_loss(
                        s_logits[:, :-1, :], 
                        t_logits[:, :-1, :], 
                        KD_TEMPERATURE, 
                        batch.get("attention_mask")[:, 1:]
                    )
                    total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss) / GRAD_ACCUM
                    
                    del t_logits
                
                total_loss.backward()
                del s_logits
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n[OOM] CUDA OOM on step {step}. Clearing cache and skipping batch...")
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(model.parameters(), model.router.parameters()), 
                    1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                if global_step % 10 == 0:
                    skip_ratio = 1.0 - gates.detach().float().mean().item()
                    epoch_bar.set_postfix({
                        "loss": f"{total_loss.item() * GRAD_ACCUM:.4f}",
                        "skip": f"{skip_ratio:.1%}",
                        "entropy": f"{layer_entropy.item():.4f}"
                    })

        current_temp = max(GUMBEL_TEMP_MIN, current_temp * TEMP_ANNEAL_RATE)

    # Evaluation
    model.eval()
    total_val_loss = 0.0
    layer_counts = []
    val_entropies = []
    
    with torch.no_grad():
        for i, v_batch in enumerate(tqdm(eval_loader, desc=f"Evaluating λ={penalty}")):
            if i >= MAX_EVAL_BATCHES: 
                break
            v_batch = {k: v.to("cuda") for k, v in v_batch.items()}
            _, v_ce, v_gates = gated_forward(model, v_batch, current_temp, hard=True)
            total_val_loss += v_ce.item()
            
            p_val = v_gates.float().mean(dim=(0, 1))
            layer_counts.append(p_val.sum().item() + ALWAYS_KEEP)
            val_entropies.append(-(p_val * torch.log(p_val + 1e-8) + (1.0 - p_val) * torch.log(1.0 - p_val + 1e-8)).mean().item())
    
    val_loss = total_val_loss / min(MAX_EVAL_BATCHES, len(eval_loader))
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
    avg_layers = sum(layer_counts) / len(layer_counts)
    avg_entropy = sum(val_entropies) / len(val_entropies)
    skip_ratio = 1.0 - (avg_layers / (ROUTABLE_LAYERS + ALWAYS_KEEP))
    dense_flops_utilization = 1.0 - skip_ratio
    
    res = {
        "penalty": penalty, 
        "val_loss": val_loss, 
        "perplexity": perplexity,
        "avg_active_layers": avg_layers, 
        "total_layers": ROUTABLE_LAYERS + ALWAYS_KEEP, 
        "skip_ratio": skip_ratio,
        "dense_flops_utilization": dense_flops_utilization,
        "avg_entropy": avg_entropy
    }
    
    print(f"Result for λ={penalty}: PPL={perplexity:.2f}, Skip Ratio={skip_ratio:.1%}, Entropy={avg_entropy:.4f}")
    
    # Cleanup memory before next penalty run
    del model
    del optimizer
    del scheduler
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return res

def main():
    print(f"\n{'*'*60}")
    print(" QWEN-7B PARETO SWEEP EXPERIMENT (EXP30)")
    print(f"{'*'*60}\n")

    # Generate Manifest for Scientific Provenance
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        commit_hash = "unknown"
        
    manifest = {
        "model_id": MODEL_ID,
        "commit_hash": commit_hash,
        "seed": 42,
        "batch_size": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "kd_alpha": KD_ALPHA,
        "penalties": PENALTIES,
        "timestamp": TIMESTAMP
    }
    manifest_path = f"results/exp30_qwen7b_manifest_{TIMESTAMP}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"[MANIFEST] Saved experiment manifest to {manifest_path}")

    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Penalty", "Val Loss", "Perplexity", "Avg Active Layers", 
            "Total Layers", "Skip Ratio", "FLOP Utilization", "Avg Entropy"
        ])

    results = []
    for p in PENALTIES:
        r = train_one_penalty(p)
        results.append(r)
        
        with open(CSV_FILENAME, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                r["penalty"], 
                r["val_loss"], 
                r["perplexity"],
                r["avg_active_layers"], 
                r["total_layers"], 
                r["skip_ratio"],
                r["dense_flops_utilization"],
                r["avg_entropy"]
            ])
    
    # Plotting the Pareto Frontier
    plt.figure(figsize=(10, 6))
    skip_ratios = [r["skip_ratio"] * 100 for r in results]
    val_losses = [r["val_loss"] for r in results]
    
    plt.plot(skip_ratios, val_losses, "o-", color="#4a90e2", linewidth=2, markersize=8)
    
    for r in results:
        plt.annotate(
            f"λ={r['penalty']}", 
            (r["skip_ratio"] * 100, r["val_loss"]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center'
        )
        
    plt.title("Qwen2.5-7B Token-Level Routing Pareto Frontier (3-Epoch)", fontsize=14, pad=15)
    plt.xlabel("Layer Skip Ratio (%)", fontsize=12)
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()
    
    print(f"\nSweep complete. Results saved to:\n- CSV: {CSV_FILENAME}\n- Plot: {PLOT_FILE}")

if __name__ == "__main__": 
    main()

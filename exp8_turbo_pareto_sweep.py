"""
exp8_turbo_pareto_sweep.py - The "Zero-Storage Parallel Sweep" Edition

Optimizations:
1. "Inverted Loop" Architecture: Instead of running 7 independent penalty sweeps sequentially
   (which requires running the Teacher 7 separate times), this script runs them in PARALLEL.
2. For each batch, the Teacher is run exactly ONCE.
3. Then, 7 separate LoRA adapters (one for each penalty) are fast-switched and trained on that same batch.
4. ZERO Disk Space and ZERO extra RAM required. Safe for 16GB systems and small SSDs.

Target: Complete a 7-point, 3-epoch, 10k-sample sweep ~3x faster than baseline.
"""

import os
import csv
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4
PENALTIES    = [0.1, 0.2, 0.8, 1.2, 2.0, 3.0]

# Standard Research Mode
TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 3
MAX_EVAL_BATCHES = 100

LR               = 3e-5
WEIGHT_DECAY     = 0.01
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_MIN   = 0.5
TEMP_ANNEAL_RATE  = 0.95
KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0
KD_WARMUP_STEPS  = 30
GATE_ENTROPY_BETA = 0.0  # Disabled: was counteracting the compute penalty and causing 10% saturation

# ---------------------------------------------------------------------------
# Hardware Optimization
# ---------------------------------------------------------------------------
def get_optimal_config():
    if not torch.cuda.is_available(): return 2, 8, 0, None
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Optimized for 8GB Cards (RTX 4060)
    # Reduced BS to 2 to prevent swapping; increased GA to 8 to keep Effective BS = 16
    bs, ga = 2, 8
    nw = 0
    attn = "sdpa" if vram_gb >= 7 else None
    print(f"[TURBO VRAM-SAFE MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga}")
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp8_turbo_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"exp8_turbo_pareto_{TIMESTAMP}.png"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RAMDataset(Dataset):
    def __init__(self, enc):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = enc["labels"]
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# ---------------------------------------------------------------------------
# Helper Logic
# ---------------------------------------------------------------------------
class TokenLevelGumbelRouter(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers)
        )
        # Initialize output layer with strong negative bias so gates start near 0.
        # This prevents the router from collapsing into the 10% skip local minimum.
        nn.init.constant_(self.net[-1].bias, -2.0)
    def forward(self, h_seq: torch.Tensor, temperature: float, hard: bool = True):
        logits = self.net(h_seq.float())
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)[..., 1].to(h_seq.dtype)

class StopForwardException(Exception): pass

def gated_forward(model, router, batch, temperature, hard=True):
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = batch["labels"]
    base = model.base_model.model.model
    
    captured_h_seq = None
    def early_stop_hook(module, input, output):
        nonlocal captured_h_seq
        h = output[0] if isinstance(output, tuple) else output
        captured_h_seq = h.detach().float()
        raise StopForwardException()

    handle = base.layers[ALWAYS_KEEP - 1].register_forward_hook(early_stop_hook)
    try:
        with torch.no_grad(): _ = model(input_ids=input_ids, attention_mask=mask)
    except StopForwardException: pass
    finally: handle.remove()

    gates = router(captured_h_seq.to("cuda"), temperature, hard)
    
    handles = []
    def layer_hook(module, input, output, idx):
        residual = input[0]
        gate = gates[:, :, idx].unsqueeze(-1).to(output[0].dtype)
        gated_h = gate * output[0] + (1.0 - gate) * residual
        return (gated_h, *output[1:]) if isinstance(output, tuple) else gated_h

    for i in range(gates.shape[2]):
        h = base.layers[i + ALWAYS_KEEP]
        handles.append(h.register_forward_hook(lambda m, a, o, idx=i: layer_hook(m, a, o, idx)))

    try:
        outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
    finally:
        for h in handles: h.remove()
        
    return outputs.logits, outputs.loss, gates

def compute_kd_loss(s_logits, t_logits, T):
    log_p = F.log_softmax(s_logits / T, dim=-1)
    p_teacher = F.softmax(t_logits / T, dim=-1)
    return F.kl_div(log_p, p_teacher, reduction="batchmean") * (T**2)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*70}\n  EXP8 TURBO: PARALLEL ADAPTER SWEEP (ZERO DISK/RAM OVERHEAD)\n{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Prepare Data
    print("Pre-tokenizing dataset...")
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    eval_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
    eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

    def tokenize_fn(b):
        o = tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        o["labels"] = o["input_ids"].copy()
        return o

    train_enc = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names, num_proc=12)
    eval_enc  = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names, num_proc=12)
    
    train_enc.set_format("torch")
    eval_enc.set_format("torch")
    
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. Setup Models
    print("\nLoading Models...")
    q_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)
    teacher.eval()
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)
    
    num_layers = len(base_model.model.layers)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")

    # 3. Setup 7 Parallel Adapters & Optimizers
    print("Initializing 7 Parallel LoRA Adapters...")
    routers = nn.ModuleDict()
    optimizers = {}
    schedulers = {}
    params_dict = {}
    
    # Init first adapter to set up PEFT model
    first_name = f"p_{str(PENALTIES[0]).replace('.', '_')}"
    model = get_peft_model(base_model, lora_cfg, adapter_name=first_name)
    
    # Add remaining adapters
    for p in PENALTIES[1:]:
        model.add_adapter(f"p_{str(p).replace('.', '_')}", lora_cfg)
        
    for p in PENALTIES:
        name = f"p_{str(p).replace('.', '_')}"
        routers[name] = TokenLevelGumbelRouter(model.config.hidden_size, num_layers - ALWAYS_KEEP).to("cuda")
        
        # Grab only parameters for THIS adapter and THIS router
        adapter_params = [param for n, param in model.named_parameters() if name in n and param.requires_grad]
        router_params = list(routers[name].parameters())
        all_params = adapter_params + router_params
        
        params_dict[name] = all_params
        optimizers[name] = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=EPOCHS * len(train_loader) // GRAD_ACCUM)

    # 4. CHUNKED PARALLEL SWEEP
    # We split 7 penalties into 2 chunks to keep VRAM < 7GB by reducing active optimizer states
    CHUNK_SIZE = 4 
    penalty_chunks = [PENALTIES[i:i + CHUNK_SIZE] for i in range(0, len(PENALTIES), CHUNK_SIZE)]
    
    checkpoint_path = "exp8_turbo_checkpoint.pt"
    start_chunk_idx = 0
    start_epoch = 0
    cur_temp = GUMBEL_TEMP_START
    g_steps = {f"p_{str(p).replace('.', '_')}": 0 for p in PENALTIES}
    
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Found checkpoint at {checkpoint_path}. Loading...")
        chk = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(chk["model"], strict=False)
        routers.load_state_dict(chk["routers"])
        for name in optimizers:
            if name in chk["optimizers"]:
                optimizers[name].load_state_dict(chk["optimizers"][name])
                schedulers[name].load_state_dict(chk["schedulers"][name])
        start_chunk_idx = chk["chunk_idx"]
        start_epoch = chk["epoch"]
        cur_temp = chk["cur_temp"]
        if "g_steps" in chk:
            g_steps.update(chk["g_steps"])
        print(f"[INFO] Resuming from Chunk {start_chunk_idx+1}, Epoch {start_epoch+1}\n")

    print(f"\nStarting Chunked Parallel Sweep ({len(penalty_chunks)} passes)...")
    
    for chunk_idx in range(start_chunk_idx, len(penalty_chunks)):
        current_chunk = penalty_chunks[chunk_idx]
        print(f"\n>>> Processing Chunk {chunk_idx+1}/{len(penalty_chunks)}: {current_chunk}")
        
        if chunk_idx > start_chunk_idx:
            start_epoch = 0
            cur_temp = GUMBEL_TEMP_START
            
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            pbar = tqdm(train_loader, desc=f"Chunk {chunk_idx+1} | Ep {epoch+1}/{EPOCHS}")
            
            for step, batch in enumerate(pbar):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                
                # ONE Teacher Pass per Batch!
                with torch.no_grad():
                    t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
                
                # Student Passes for this chunk
                metrics_log = {}
                for p in current_chunk:
                    name = f"p_{str(p).replace('.', '_')}"
                    model.set_adapter(name)
                    
                    s_logits, ce_loss, gates = gated_forward(model, routers[name], batch, cur_temp, hard=True)
                    
                    p_mean = gates.detach().float().mean()
                    gate_loss = gates.float().mean() * p
                    
                    eps = 1e-6
                    gate_entropy = -(p_mean * (p_mean+eps).log() + (1-p_mean)*(1-p_mean+eps).log())
                    entropy_bonus = GATE_ENTROPY_BETA * gate_entropy

                    if g_steps[name] < KD_WARMUP_STEPS:
                        total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
                    else:
                        kd_loss = compute_kd_loss(s_logits[:, :-1, :], t_logits, KD_TEMPERATURE)
                        total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
                    
                    total_loss.backward()
                    
                    if (step + 1) % GRAD_ACCUM == 0:
                        torch.nn.utils.clip_grad_norm_(params_dict[name], 1.0)
                        optimizers[name].step()
                        schedulers[name].step()
                        optimizers[name].zero_grad()
                        g_steps[name] += 1
                    
                    metrics_log[f"p{p}"] = f"{gates.detach().float().mean().item()*(num_layers-ALWAYS_KEEP)+ALWAYS_KEEP:.1f}L"
                    
                    # Prevent VRAM fragmentation / CUBLAS Execution errors
                    del s_logits, ce_loss, gates, total_loss
                
                del t_logits
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.cuda.empty_cache()
                
                if (step + 1) % 10 == 0:
                    pbar.set_postfix(metrics_log)
            
            cur_temp = max(GUMBEL_TEMP_MIN, cur_temp * TEMP_ANNEAL_RATE)
            
            # this is the pause point so that you can ctrl+C in the terminal whatever epoch you
            # were doing gets saved into a .pt file so that when you restart the training it starts
            # from the previous epoch this ensures you do not be like me...... Re-running the same script
            # again and again and again and again. You're welcome!. This is a sanity feature. You skip this if you want.
            # Save Checkpoint at end of epoch
            if epoch + 1 == EPOCHS:
                next_chunk_idx = chunk_idx + 1
                next_epoch = 0
                next_temp = GUMBEL_TEMP_START
            else:
                next_chunk_idx = chunk_idx  
                next_epoch = epoch + 1
                next_temp = cur_temp

            if next_chunk_idx < len(penalty_chunks):
                peft_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
                torch.save({
                    "chunk_idx": next_chunk_idx,
                    "epoch": next_epoch,
                    "cur_temp": next_temp,
                    "model": peft_state,
                    "routers": routers.state_dict(),
                    "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                    "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                    "g_steps": g_steps,
                }, checkpoint_path)
                print(f"\n[INFO] Saved checkpoint to {checkpoint_path}")

    # 5. Parallel Evaluation
    print("\nRunning Evaluation...")
    model.eval()
    sweep_results = []
    
    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["Penalty", "Val Loss", "Avg Active Layers", "Total Layers", "Skip Ratio"])
        
    for p in PENALTIES:
        name = f"p_{str(p).replace('.', '_')}"
        model.set_adapter(name)
        
        v_losses, v_layers = [], []
        with torch.no_grad():
            for i, b in enumerate(eval_loader):
                if i >= MAX_EVAL_BATCHES: break
                b = {k: v.to("cuda") for k, v in b.items()}
                _, v_ce, v_gates = gated_forward(model, routers[name], b, cur_temp, hard=True)
                v_losses.append(v_ce.item())
                v_layers.append(v_gates.float().mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP)
        
        res = {
            "penalty": p, 
            "val_loss": sum(v_losses)/len(v_losses), 
            "avg_active_layers": sum(v_layers)/len(v_layers),
            "skip_ratio": 1.0 - (sum(v_layers)/len(v_layers)/num_layers)
        }
        sweep_results.append(res)
        print(f"  Penalty: {p:.3f} | Val Loss: {res['val_loss']:.4f} | Skip: {res['skip_ratio']:.1%}")
        
        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([res["penalty"], res["val_loss"], res["avg_active_layers"], num_layers, res["skip_ratio"]])

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    xs = [r["skip_ratio"]*100 for r in sweep_results]
    ys = [r["val_loss"] for r in sweep_results]
    plt.plot(xs, ys, "o-", color="#2c3e50", linewidth=2, markersize=8, markerfacecolor="#e74c3c")
    plt.title("Pareto Frontier (Parallel Turbo Sweep)", fontsize=14, fontweight="bold")
    plt.xlabel("Layer Skip Ratio (%)", fontsize=12); plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(PLOT_FILE, dpi=150); plt.close()
    print(f"\nParallel Sweep Complete!\nCSV: {CSV_FILENAME}\nPlot: {PLOT_FILE}")

if __name__ == "__main__":
    main()

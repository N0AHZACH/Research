"""
exp13_openllama_pareto.py - Phase 4.5: OpenLLaMA 3B Pareto Sweep

This script runs the exact same "Phase Transition" test we did on Qwen2.5-3B,
but on the OpenLLaMA-3B architecture. This proves to academic reviewers that the 
routing dam-break effect is a universal property of 3B parameter networks, not a quirk of Qwen.
to force the deeper 36-layer router into sparsity and plot the ultimate 3B Pareto Frontier.

Hardware config is automatically pulled from exp11 to ensure cloud-safe (OOM resilient) execution.
"""

import gc
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
MODEL_ID     = "openlm-research/open_llama_3b_v2"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4

# Publication-Grade Sweep (Rigorous Convergence)
PENALTIES    = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0]

TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 2
MAX_EVAL_BATCHES = 100

LR               = 3e-5
WEIGHT_DECAY     = 0.01
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_MIN   = 0.5
TEMP_ANNEAL_RATE  = 0.95
KD_ALPHA         = 0.3    # Lowered for 3B
KD_TEMPERATURE   = 2.0
KD_WARMUP_STEPS  = 50
GATE_ENTROPY_BETA = 0.0

# ---------------------------------------------------------------------------
# Hardware Auto-Optimisation
# ---------------------------------------------------------------------------
def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, True, torch.float32

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_turing = 'T4' in gpu_name or 'RTX 20' in gpu_name or 'Turing' in gpu_name

    # Determine best configuration based on VRAM
    if vram_gb >= 70:  # A100 80GB
        bs, ga, use_4bit = 8, 2, False
    elif vram_gb >= 35: # A100 40GB
        bs, ga, use_4bit = 4, 4, False
    elif vram_gb >= 21: # L4 24GB (GCP often reports ~22GB available)
        bs, ga, use_4bit = 2, 8, False
    elif vram_gb >= 14: # T4 16GB
        bs, ga, use_4bit = 2, 8, True
    else:
        bs, ga, use_4bit = 1, 16, True

    compute_dtype = torch.float16 if is_turing else torch.bfloat16

    cpu_count = os.cpu_count() or 2
    nw = 0 if os.name == 'nt' else min(2, cpu_count - 1)
    attn = "sdpa" if vram_gb >= 7 else None
    print(f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f}GB VRAM | BS={bs}, GA={ga}, 4bit={use_4bit}, dtype={compute_dtype}")
    return bs, ga, nw, attn, use_4bit, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp13_openllama_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"exp13_openllama_pareto_{TIMESTAMP}.png"

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
        # Deep models are extremely resistant to dropping layers. 
        # Stronger bias ensures gates start with low probability of being kept.
        nn.init.constant_(self.net[-1].bias, -2.5)

    def forward(self, h_seq: torch.Tensor, temperature: float, hard: bool = True):
        logits = self.net(h_seq.float())
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)[..., 1].to(h_seq.dtype)

class StopForwardException(Exception): pass

def gated_forward(model, router, batch, temperature, hard=True):
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = batch.get("labels", None)
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
    print(f"\n{'='*70}\n  EXP13 OPENLLAMA-3B PARETO SWEEP\n{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Pre-tokenizing dataset...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    eval_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
    eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

    def tokenize_fn(b):
        o = tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        o["labels"] = o["input_ids"].copy()
        return o

    tok_procs = min(os.cpu_count() or 1, 2)
    train_enc = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names, num_proc=tok_procs)
    eval_enc  = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
    
    train_enc.set_format("torch")
    eval_enc.set_format("torch")
    
    pin = torch.cuda.is_available() and (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) > 12 if hasattr(os, 'sysconf') else True)
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    print("\nLoading Models...")
    if USE_4BIT:
        q_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=COMPUTE_DTYPE, device_map="cuda", attn_implementation=ATTN_IMPL)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=COMPUTE_DTYPE, device_map="cuda", attn_implementation=ATTN_IMPL)
    
    teacher.eval()
    
    num_layers = len(base_model.model.layers)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")

    print(f"Initializing {len(PENALTIES)} Parallel LoRA Adapters...")
    routers = nn.ModuleDict()
    optimizers = {}
    schedulers = {}
    params_dict = {}
    
    first_name = f"p_{str(PENALTIES[0]).replace('.', '_')}"
    model = get_peft_model(base_model, lora_cfg, adapter_name=first_name)
    
    for p in PENALTIES[1:]:
        model.add_adapter(f"p_{str(p).replace('.', '_')}", lora_cfg)
        
    for p in PENALTIES:
        name = f"p_{str(p).replace('.', '_')}"
        routers[name] = TokenLevelGumbelRouter(model.config.hidden_size, num_layers - ALWAYS_KEEP).to("cuda")
        
        adapter_params = [param for n, param in model.named_parameters() if name in n and param.requires_grad]
        router_params = list(routers[name].parameters())
        all_params = adapter_params + router_params
        
        params_dict[name] = all_params
        optimizers[name] = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=EPOCHS * len(train_loader) // GRAD_ACCUM)

    # For publication rigor, we sweep all 6 penalties.
    # The L4's 24GB VRAM allows us to fit all 6 into a single chunk.
    CHUNK_SIZE = 6
    penalty_chunks = [PENALTIES[i:i + CHUNK_SIZE] for i in range(0, len(PENALTIES), CHUNK_SIZE)]
    
    checkpoint_path = "exp13_openllama_pareto_ckpt.pt"
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
            
        try:
            for epoch in range(start_epoch, EPOCHS):
                model.train()
                pbar = tqdm(train_loader, desc=f"Chunk {chunk_idx+1} | Ep {epoch+1}/{EPOCHS}")
                
                oom_count = 0
                for step, batch in enumerate(pbar):
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                    
                    try:
                        with torch.no_grad():
                            t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
                        
                        metrics_log = {}
                        for p in current_chunk:
                            name = f"p_{str(p).replace('.', '_')}"
                            model.set_adapter(name)
                            
                            s_logits, ce_loss, gates = gated_forward(model, routers[name], batch, cur_temp, hard=True)
                            
                            p_mean = gates.detach().float().mean()
                            
                            # Modified Loss: Use L1 penalty based on sum of per-layer activities, like exp11
                            per_layer_activity = gates.float().mean(dim=(0, 1))
                            l1_penalty = per_layer_activity.sum() * p
                            gate_loss = l1_penalty
                            
                            if g_steps[name] < KD_WARMUP_STEPS:
                                total_loss = (ce_loss + gate_loss) / GRAD_ACCUM
                            else:
                                kd_loss = compute_kd_loss(s_logits[:, :-1, :], t_logits, KD_TEMPERATURE)
                                total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss) / GRAD_ACCUM
                            
                            total_loss.backward()
                            
                            if (step + 1) % GRAD_ACCUM == 0:
                                torch.nn.utils.clip_grad_norm_(params_dict[name], 1.0)
                                optimizers[name].step()
                                schedulers[name].step()
                                optimizers[name].zero_grad()
                                g_steps[name] += 1
                            
                            metrics_log[f"p{p}"] = f"{gates.detach().float().mean().item()*(num_layers-ALWAYS_KEEP)+ALWAYS_KEEP:.1f}L"
                            
                            del s_logits, ce_loss, gates, total_loss
                        
                        del t_logits
                        if (step + 1) % GRAD_ACCUM == 0:
                            torch.cuda.empty_cache()
                        
                        if (step + 1) % 10 == 0:
                            pbar.set_postfix(metrics_log)

                        # FREQUENT CHECKPOINTING (Every 50 optimizer steps = 50 * GRAD_ACCUM batches)
                        if (step + 1) % (50 * GRAD_ACCUM) == 0:
                            peft_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
                            torch.save({
                                "chunk_idx": chunk_idx,
                                "epoch": epoch,
                                "cur_temp": cur_temp,
                                "model": peft_state,
                                "routers": routers.state_dict(),
                                "optimizers": {k: v.state_dict() for k, v in optimizers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                                "schedulers": {k: v.state_dict() for k, v in schedulers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                                "g_steps": g_steps,
                            }, checkpoint_path)

                    except torch.cuda.OutOfMemoryError:
                        oom_count += 1
                        print(f"\n[OOM] CUDA OOM on step {step}. Clearing cache...")
                        for name in optimizers:
                            optimizers[name].zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Emergency save on OOM
                        peft_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
                        torch.save({
                            "chunk_idx": chunk_idx,
                            "epoch": epoch,
                            "cur_temp": cur_temp,
                            "model": peft_state,
                            "routers": routers.state_dict(),
                            "optimizers": {k: v.state_dict() for k, v in optimizers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                            "schedulers": {k: v.state_dict() for k, v in schedulers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                            "g_steps": g_steps,
                        }, checkpoint_path)
                        
                        if oom_count >= 5:
                            print("[OOM] Too many OOM errors. Please reduce batch size or increase GPU VRAM.")
                            return
                        continue
                
                cur_temp = max(GUMBEL_TEMP_MIN, cur_temp * TEMP_ANNEAL_RATE)
                
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
                        "optimizers": {k: v.state_dict() for k, v in optimizers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                        "schedulers": {k: v.state_dict() for k, v in schedulers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                        "g_steps": g_steps,
                    }, checkpoint_path)
                    print(f"\n[INFO] Saved checkpoint to {checkpoint_path}")

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Saving checkpoint...")
            peft_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
            torch.save({
                "chunk_idx": chunk_idx,
                "epoch": epoch,
                "cur_temp": cur_temp,
                "model": peft_state,
                "routers": routers.state_dict(),
                "optimizers": {k: v.state_dict() for k, v in optimizers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                "schedulers": {k: v.state_dict() for k, v in schedulers.items() if k.startswith("p_") and float(k.replace("p_", "").replace("_", ".")) in current_chunk},
                "g_steps": g_steps,
            }, checkpoint_path)
            return

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
        print(f"  Penalty: {p:.1f} | Val Loss: {res['val_loss']:.4f} | Skip: {res['skip_ratio']:.1%}")
        
        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([res["penalty"], res["val_loss"], res["avg_active_layers"], num_layers, res["skip_ratio"]])

    plt.figure(figsize=(10, 6))
    xs = [r["skip_ratio"]*100 for r in sweep_results]
    ys = [r["val_loss"] for r in sweep_results]
    plt.plot(xs, ys, "o-", color="#2c3e50", linewidth=2, markersize=8, markerfacecolor="#e74c3c")
    plt.title("OpenLLaMA 3B Pareto Frontier (Parallel Turbo Sweep)", fontsize=14, fontweight="bold")
    plt.xlabel("Layer Skip Ratio (%)", fontsize=12); plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(PLOT_FILE, dpi=150); plt.close()
    
    print(f"\nParallel Sweep Complete!\nCSV: {CSV_FILENAME}\nPlot: {PLOT_FILE}")

if __name__ == "__main__":
    main()

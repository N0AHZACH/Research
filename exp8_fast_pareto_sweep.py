"""
exp8_fast_pareto_sweep.py - Aggressive 3-Hour Research Sweep

Optimizations for speed:
1. 1 Epoch instead of 3 (sufficient for router trend analysis).
2. 8-bit Quantized Teacher (faster KD passes, lower VRAM).
3. Batch Size 10 (higher throughput).
4. torch.compile on both Student and Teacher.
5. Zero-IO model resets (shares base_model in RAM).

Target: Complete 7-point Pareto sweep in ~3 hours on RTX 4000.
"""

import os
import itertools
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4
PENALTIES    = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 1   # Fast Mode
MAX_EVAL_BATCHES = 100

LR               = 3e-5
WEIGHT_DECAY     = 0.01
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_MIN   = 0.5
TEMP_ANNEAL_RATE  = 0.95
KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0
KD_WARMUP_STEPS  = 30
GATE_ENTROPY_BETA = 0.1

# ---------------------------------------------------------------------------
# Hardware Auto-Optimisation
# ---------------------------------------------------------------------------
def get_optimal_config():
    if not torch.cuda.is_available(): return 2, 8, 0, None
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cpu_count = os.cpu_count() or 4
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if vram_gb >= 15:
        # Aggressive Server Mode
        bs, ga, nw, attn = 10, 1, min(cpu_count // 2, 12), "sdpa"
        print(f"[FAST MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga} | Workers: {nw} | 8-bit Teacher: ON")
    else:
        bs, ga, nw, attn = 2, 8, 0, None
        print(f"[DESKTOP MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga}")
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp8_fast_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"exp8_fast_pareto_{TIMESTAMP}.png"

# ---------------------------------------------------------------------------
# Dataset & Router Logic (Identical to exp6/exp8)
# ---------------------------------------------------------------------------
raw      = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
eval_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
raw      = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
eval_ds  = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names)
train_ds.set_format("torch")
eval_ds.set_format("torch")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class GumbelRouter(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, num_layers))
    def forward(self, pooled_h: torch.Tensor, temperature: float, hard: bool = True):
        logits = self.net(pooled_h.float())
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)[..., 1].to(pooled_h.dtype)

class GatedForwardContext:
    def __init__(self): self.gates = None

def gated_forward(model, batch, temperature, hard=True):
    ctx = GatedForwardContext()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    with torch.no_grad():
        base = model.base_model.model.model
        h = base.embed_tokens(input_ids)
        for i in range(ALWAYS_KEEP):
            h = base.layers[i](h, attention_mask=attention_mask)[0]
        ctx.gates = model.router(h.mean(dim=1), temperature, hard)

    handles = []
    def hook_fn(module, args, output, layer_idx):
        gate = ctx.gates[:, layer_idx].view(-1, 1, 1)
        return (output[0] * gate, *output[1:])

    for i in range(len(ctx.gates[0])):
        h = model.base_model.model.model.layers[i + ALWAYS_KEEP]
        handles.append(h.register_forward_hook(lambda m, a, o, idx=i: hook_fn(m, a, o, idx)))

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finally:
        for h in handles: h.remove()
        
    return outputs.logits, outputs.loss, ctx.gates

def compute_kd_loss(s_logits, t_logits, T):
    return F.kl_div(F.log_softmax(s_logits/T, dim=-1), F.softmax(t_logits/T, dim=-1), reduction="batchmean") * (T**2)

def train_one_penalty(penalty: float, teacher_model, base_model) -> dict:
    print(f"\n--- λ={penalty:.3f} ---")
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base_model, lora_cfg)
    ROUTABLE_LAYERS = len(model.base_model.model.model.layers) - ALWAYS_KEEP
    model.router = GumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
    
    if ATTN_IMPL == "sdpa":
        try: model = torch.compile(model)
        except: pass

    optimizer = torch.optim.AdamW(itertools.chain(model.parameters(), model.router.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM)
    
    current_temp, global_step = GUMBEL_TEMP_START, 0
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"L={penalty}")):
            batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
            s_logits, ce_loss, gates = gated_forward(model, batch, current_temp, hard=True)
            with torch.no_grad(): t_logits = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            
            p_mean = gates.detach().float().mean(); eps = 1e-6
            gate_entropy = -(p_mean * (p_mean + eps).log() + (1 - p_mean) * (1 - p_mean + eps).log())
            # Scale gate loss by CE to maintain constant pressure (Fix from exp6)
            ce_scale = ce_loss.detach().float().clamp(min=0.5)
            gate_loss = gates.float().mean() * penalty * ce_scale
            entropy_bonus = GATE_ENTROPY_BETA * gate_entropy * ce_scale

            if global_step < KD_WARMUP_STEPS:
                total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
            else:
                kd_loss = compute_kd_loss(s_logits[:, :-1, :], t_logits[:, :-1, :], KD_TEMPERATURE)
                total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
            total_loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(model.router.parameters()), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); global_step += 1
        current_temp = max(GUMBEL_TEMP_MIN, current_temp * TEMP_ANNEAL_RATE)

    model.eval(); total_val_loss, layer_counts = 0.0, []
    with torch.no_grad():
        for i, v_batch in enumerate(eval_loader):
            if i >= MAX_EVAL_BATCHES: break
            v_batch = {k: v.to("cuda") for k, v in v_batch.items() if isinstance(v, torch.Tensor)}
            _, v_ce, v_gates = gated_forward(model, v_batch, current_temp, hard=True)
            total_val_loss += v_ce.item()
            layer_counts.append(v_gates.float().mean(dim=0).sum().item() + ALWAYS_KEEP)
    
    val_loss = total_val_loss / min(MAX_EVAL_BATCHES, len(eval_loader))
    avg_layers = sum(layer_counts) / len(layer_counts)
    res = {"penalty": penalty, "val_loss": val_loss, "avg_active_layers": avg_layers, "total_layers": ROUTABLE_LAYERS + ALWAYS_KEEP, "skip_ratio": 1.0 - (avg_layers/(ROUTABLE_LAYERS+ALWAYS_KEEP))}
    del model; torch.cuda.empty_cache(); return res

def main():
    print("\n--- FAST PARETO SWEEP ---")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation=ATTN_IMPL)
    print("Loading Teacher (8-bit if available)...")
    try:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            quantization_config=quant_config,
            device_map="cuda", 
            attn_implementation=ATTN_IMPL
        )
    except Exception as e:
        print(f"[INFO] 8-bit loading skipped ({e}). Using bf16.")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda", 
            attn_implementation=ATTN_IMPL
        )
    teacher_model.eval()
    if ATTN_IMPL == "sdpa":
        try: teacher_model = torch.compile(teacher_model)
        except: pass

    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["Penalty", "Val Loss", "Avg Active Layers", "Total Layers", "Skip Ratio"])

    results = []
    for p in PENALTIES:
        r = train_one_penalty(p, teacher_model, base_model)
        results.append(r)
        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([r["penalty"], r["val_loss"], r["avg_active_layers"], r["total_layers"], r["skip_ratio"]])
    
    # Simple Plotting
    plt.figure(figsize=(8, 5))
    plt.plot([r["skip_ratio"]*100 for r in results], [r["val_loss"] for r in results], "o-", color="#4a90e2")
    plt.title("Fast Pareto Frontier (1-Epoch)"); plt.xlabel("Skip Ratio (%)"); plt.ylabel("Val Loss")
    plt.grid(True); plt.savefig(PLOT_FILE); plt.close()
    print(f"Sweep complete. CSV: {CSV_FILENAME}")

if __name__ == "__main__": main()

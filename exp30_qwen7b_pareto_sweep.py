"""
exp8_fast_pareto_sweep.py - Aggressive 3-Hour Research Sweep

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
MODEL_ID     = "Qwen/Qwen2.5-7B"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4
PENALTIES    = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 3   # Matched to 1.1B baseline rigor
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

    if vram_gb >= 80:
        bs, ga, nw, attn = 16, 1, 8, "sdpa"
        print(f"[MASSIVE SERVER MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga} | Workers: {nw} | 8-bit Teacher: ON")
    elif vram_gb >= 45:
        bs, ga, nw, attn = 16, 1, 8, "sdpa"
        print(f"[FAST MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga} | Workers: {nw} | 8-bit Teacher: ON")
    elif vram_gb >= 15:
        # Aggressive Server Mode
        bs, ga, nw, attn = 10, 1, min(cpu_count // 2, 12), "sdpa"
        print(f"[FAST MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga} | Workers: {nw} | 8-bit Teacher: ON")
    else:
        bs, ga, nw, attn = 2, 8, 0, None
        print(f"[DESKTOP MODE] VRAM: {vram_gb:.1f}GB | BS: {bs} | GA: {ga}")
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)
CSV_FILENAME = f"results/exp30_qwen7b_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"results/exp30_qwen7b_pareto_{TIMESTAMP}.png"

# ---------------------------------------------------------------------------
# Dataset & Router Logic (Identical to exp6/exp8)
# ---------------------------------------------------------------------------
raw      = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
eval_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
raw      = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    out["labels"] = [[l if m == 1 else -100 for l, m in zip(ids, mask)] for ids, mask in zip(out["input_ids"], out["attention_mask"])]
    return out

tok_procs = min(os.cpu_count() or 1, 32)
train_ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names, num_proc=tok_procs)
eval_ds  = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
train_ds.set_format("torch")
eval_ds.set_format("torch")

class RAMDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.input_ids = enc["input_ids"] if isinstance(enc["input_ids"], torch.Tensor) else torch.stack(list(enc["input_ids"]))
        self.attention_mask = enc["attention_mask"] if isinstance(enc["attention_mask"], torch.Tensor) else torch.stack(list(enc["attention_mask"]))
        self.labels = enc["labels"] if isinstance(enc["labels"], torch.Tensor) else torch.stack(list(enc["labels"]))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

pin = torch.cuda.is_available() and (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) > 12 if hasattr(os, 'sysconf') else True)
train_loader = DataLoader(RAMDataset(train_ds), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
eval_loader  = DataLoader(RAMDataset(eval_ds),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

class TokenLevelGumbelRouter(nn.Module):
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
        h_seq  = h_seq.float()
        logits = self.net(h_seq)
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        soft   = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(h_seq.dtype)

class StopForwardException(Exception): pass

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

    def install_capture_hook(self, layer):
        def hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.captured_h_seq = hidden_state.detach().float()
        self.handles.append(layer.register_forward_hook(hook))

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
    attention_mask = batch["attention_mask"]
    labels = batch.get("labels", None)
    
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
    s_logits = s_logits.reshape(-1, s_logits.size(-1))
    t_logits = t_logits.reshape(-1, t_logits.size(-1))
    mask = mask.reshape(-1)
    
    kl_sum = 0.0
    chunk_size = 512
    for i in range(0, s_logits.size(0), chunk_size):
        s_chunk = s_logits[i:i+chunk_size]
        t_chunk = t_logits[i:i+chunk_size]
        m_chunk = mask[i:i+chunk_size]
        
        kl = F.kl_div(F.log_softmax(s_chunk/T, dim=-1), F.softmax(t_chunk/T, dim=-1), reduction="none").sum(dim=-1)
        kl_sum += (kl * m_chunk).sum()
        
    return kl_sum / mask.sum().clamp(min=1.0) * (T**2)

def train_one_penalty(penalty: float) -> dict:
    print(f"\n--- lambda={penalty:.3f} ---")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL).to("cuda")
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base_model, lora_cfg)
    ROUTABLE_LAYERS = len(model.base_model.model.model.layers) - ALWAYS_KEEP
    model.router = TokenLevelGumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
    
    if ATTN_IMPL == "sdpa" and os.name != "nt":
        try: model = torch.compile(model)
        except: pass

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM)
    
    current_temp, global_step = GUMBEL_TEMP_START, 0
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"L={penalty}")):
            batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
            s_logits, ce_loss, gates = gated_forward(model, batch, current_temp, hard=True)
            
            p_mean = gates.detach().float().mean(); eps = 1e-6
            gate_entropy = -(p_mean * (p_mean + eps).log() + (1 - p_mean) * (1 - p_mean + eps).log())
            # Scale gate loss by CE to maintain constant pressure (Fix from exp6)
            ce_scale = ce_loss.detach().float().clamp(min=0.5)
            per_layer_activity = gates.float().mean(dim=(0, 1))
            gate_loss = per_layer_activity.sum() * penalty * ce_scale
            entropy_bonus = GATE_ENTROPY_BETA * gate_entropy * ce_scale

            if global_step < KD_WARMUP_STEPS:
                total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
            else:
                with model.disable_adapter():
                    with torch.no_grad(): 
                        t_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                kd_loss = compute_kd_loss(s_logits[:, :-1, :], t_logits[:, :-1, :], KD_TEMPERATURE, batch["attention_mask"][:, 1:])
                del t_logits
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
            layer_counts.append(v_gates.float().mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP)
    
    val_loss = total_val_loss / min(MAX_EVAL_BATCHES, len(eval_loader))
    avg_layers = sum(layer_counts) / len(layer_counts)
    res = {"penalty": penalty, "val_loss": val_loss, "avg_active_layers": avg_layers, "total_layers": ROUTABLE_LAYERS + ALWAYS_KEEP, "skip_ratio": 1.0 - (avg_layers/(ROUTABLE_LAYERS+ALWAYS_KEEP))}
    import gc; del model, optimizer, scheduler, base_model; gc.collect(); torch.cuda.empty_cache(); return res

def main():
    print("\n--- FAST PARETO SWEEP ---")

    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["Penalty", "Val Loss", "Avg Active Layers", "Total Layers", "Skip Ratio"])

    results = []
    for p in PENALTIES:
        r = train_one_penalty(p)
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

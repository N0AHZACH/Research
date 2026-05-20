"""
exp9_token_level_routing.py - Phase 2: Token-Level Dynamic Routing

This script upgrades the sequence-level Gumbel Router (exp6) to a Token-Level Router.
Instead of pooling the contextual hidden state after layer 4 and making a single
routing decision for the entire sequence, this router evaluates the hidden state
of *each token independently* and produces a per-token gate [Batch, SeqLen, Layers].

This allows the model to allocate more compute to complex reasoning tokens and
skip layers for trivial tokens (like punctuation or stop words).
"""
import csv
import os
import itertools
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_ID         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10000
EVAL_SAMPLES     = 1000
EPOCHS           = 3
MAX_EVAL_BATCHES = 100

BATCH_SIZE       = 2
GRAD_ACCUM       = 8
LR               = 3e-5
WEIGHT_DECAY     = 0.01

ALWAYS_KEEP      = 4
COMPUTE_PENALTY  = 1.0   
GUMBEL_TEMP      = 1.0
TEMP_ANNEAL_RATE = 0.95
KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0   
GATE_ENTROPY_BETA = 0.1  
KD_WARMUP_STEPS  = 50

EVAL_EVERY_STEPS = 100
LOG_EVERY_STEPS  = 20

# ---------------------------------------------------------------------------
# Hardware Auto-Optimisation
# ---------------------------------------------------------------------------
def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # RTX 4060 8GB
    bs = 2
    ga = 8
    nw = 0  # 0 for RAMDataset on Windows (fastest)
    attn = "sdpa" if vram_gb >= 7 else None
    print(f"[HARDWARE MODE] Detected {vram_gb:.1f}GB VRAM. Using: BS={bs}, GA={ga}")
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp9_token_level_routing_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp9_token_output_{TIMESTAMP}"

# ==============================================================================
# Dataset - Wikitext-103
# ==============================================================================
print("Loading dataset: wikitext-103-raw-v1 ...")
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

# Utilizing 12 cores for tokenization
train_enc = raw.map(tokenize, batched=True, remove_columns=raw.column_names, num_proc=12)
eval_enc  = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names, num_proc=12)
train_enc.set_format("torch")
eval_enc.set_format("torch")

class RAMDataset(torch.utils.data.Dataset):
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

train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
print(f"  Train: {len(train_enc)} | Eval: {len(eval_enc)}")

# ==============================================================================
# Models: Student (LoRA) + Teacher (Frozen) for KD
# ==============================================================================
print("\nLoading TinyLlama student (LoRA) ...")
q_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)

print("Loading frozen Teacher for KD ...")
teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_cfg, device_map="cuda", attn_implementation=ATTN_IMPL)
for p in teacher_model.parameters():
    p.requires_grad = False
teacher_model.eval()

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

TOTAL_LAYERS    = len(model.base_model.model.model.layers)
ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP
print(f"  Total layers: {TOTAL_LAYERS} | Always-kept: {ALWAYS_KEEP} | Routable: {ROUTABLE_LAYERS}")

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

    def forward(self, h_seq: torch.Tensor, temperature: float, hard: bool = True):
        h_seq  = h_seq.float()                                            # precision for Gumbel
        logits = self.net(h_seq)                                          # [B, S, L]
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # [B, S, L, 2]
        soft   = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(h_seq.dtype)                               # [B, S, L]  restore dtype

model.router = TokenLevelGumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
for p in model.router.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    itertools.chain(model.parameters(), model.router.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM
)

class StopForwardException(Exception): pass

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

    def install_capture_hook(self, layer):
        def hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            # NO MEAN POOLING! We keep the sequence dimension.
            self.captured_h_seq = hidden_state.detach().float()  # [B, S, H]
        self.handles.append(layer.register_forward_hook(hook))

    def install_gate_hooks(self, layers, gates):
        self.gates = gates  # [B, S, L]
        for i, layer in enumerate(layers):
            idx = i
            def hook(module, input, output, layer_i=idx):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                
                # Gate for this specific layer
                # self.gates[:, :, layer_i] is [B, S]
                # We need to broadcast across the Hidden dimension, so we unsqueeze to [B, S, 1]
                gate = self.gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
                
                gated_h = gate * h + (1.0 - gate) * residual
                return (gated_h,) + output[1:] if is_tuple else gated_h

            self.handles.append(layer.register_forward_hook(hook))

def gated_forward(model, batch, temperature, hard=True):
    input_ids      = batch["input_ids"]
    labels         = batch.get("labels", None)
    attention_mask = batch.get("attention_mask", None)
    
    transformer = model.base_model.model.model
    all_layers  = transformer.layers

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

    h_seq = ctx.captured_h_seq.to("cuda")                             # [B, S, H]
    gates = model.router(h_seq, temperature=temperature, hard=hard)   # [B, S, L]

    ctx.install_gate_hooks(all_layers[ALWAYS_KEEP:], gates)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    ctx.remove_hooks()

    return outputs.logits, outputs.loss, gates

def compute_kd_loss(s_logits, t_logits, T):
    return F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits  / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)

# ==============================================================================
# Training Loop
# ==============================================================================
print(f"\nStarting Phase 2 TOKEN-LEVEL Router training...")

headers = ["Global Step", "Epoch", "Training Loss", "CE Loss", "KD Loss",
           "Gate Loss", "Validation Loss", "Avg Active Layers", "Gumbel Temp", "LR"]
with open(CSV_FILENAME, "w", newline="") as f:
    csv.writer(f).writerow(headers)

global_step    = 0
current_temp   = GUMBEL_TEMP
best_val_loss  = float("inf")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(epoch_bar):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        student_logits, ce_loss, gates = gated_forward(model, batch, temperature=current_temp, hard=True)

        with torch.no_grad():
            teacher_logits = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits

        gate_loss = gates.float().mean() * COMPUTE_PENALTY

        p_mean = gates.detach().float().mean()
        eps = 1e-6
        entropy_bonus = GATE_ENTROPY_BETA * -(p_mean * (p_mean + eps).log() + (1 - p_mean) * (1 - p_mean + eps).log())

        if global_step < KD_WARMUP_STEPS:
            kd_loss    = torch.tensor(0.0, device="cuda")
            total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
        else:
            kd_loss    = compute_kd_loss(student_logits[:, :-1, :], teacher_logits[:, :-1, :], KD_TEMPERATURE)
            total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
            
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(itertools.chain(model.parameters(), model.router.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                avg_layers = gates.detach().float().mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP
                
                epoch_bar.set_postfix({
                    "loss": f"{total_loss.item() * GRAD_ACCUM:.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "layers": f"{avg_layers:.1f}",
                })

                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    total_val_loss, total_active = 0.0, []
                    with torch.no_grad():
                        for i, val_batch in enumerate(eval_loader):
                            if i >= MAX_EVAL_BATCHES: break
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items()}
                            _, v_ce, v_gates = gated_forward(model, val_batch, temperature=current_temp, hard=True)
                            total_val_loss += v_ce.item()
                            total_active.append(v_gates.float().mean(dim=(0, 1)).sum().item() + ALWAYS_KEEP)

                    val_loss = total_val_loss / min(MAX_EVAL_BATCHES, len(eval_loader))
                    val_avg_layers = sum(total_active) / len(total_active)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(SAVE_DIR, "best_model"), exist_ok=True)
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        torch.save(model.router.state_dict(), os.path.join(SAVE_DIR, "best_model", "router_weights.pt"))

                    model.train()

                    with open(CSV_FILENAME, "a", newline="") as f:
                        csv.writer(f).writerow([
                            global_step, epoch + 1, f"{total_loss.item() * GRAD_ACCUM:.4f}", 
                            f"{ce_loss.item():.4f}", f"{kd_loss.item():.4f}", f"{gate_loss.item():.4f}", 
                            f"{val_loss:.4f}", f"{val_avg_layers:.1f}", f"{current_temp:.4f}", 
                            f"{scheduler.get_last_lr()[0]:.2e}"
                        ])

    current_temp = max(0.5, current_temp * TEMP_ANNEAL_RATE)

print("\nSaving final model checkpoint...")
os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
torch.save(model.router.state_dict(), os.path.join(SAVE_DIR, "final_model", "router_weights.pt"))
print("Done!")

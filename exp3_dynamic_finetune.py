import csv
import os
import datetime
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH      = 128
TRAIN_SAMPLES   = 2500
EVAL_SAMPLES    = 500

EPOCHS          = 1
LR              = 5e-5

# ──────────────────────────────────────────────────────────────────────────────
# Hardware Auto-Optimisation (Server Mode)
# ──────────────────────────────────────────────────────────────────────────────
def get_optimal_config():
    """Detects hardware and returns optimized training parameters."""
    if not torch.cuda.is_available():
        return 2, 4, 0, None  # Default fallback

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cpu_count = os.cpu_count() or 4
    
    # Enable TF32 for Ampere+ GPUs (RTX 3000/4000)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if vram_gb >= 15:
        # RTX 4000 20GB or similar server-grade
        bs = 16
        ga = 1  # No accumulation needed at bs=16 for TinyLlama
        nw = min(cpu_count // 2, 8)  # Utilize the i9-13900KF
        attn = "sdpa"
        print(f"[SERVER MODE] Detected {vram_gb:.1f}GB VRAM. Scaling: BS={bs}, GA={ga}, Workers={nw}, SDPA=ON")
    else:
        # RTX 4060 8GB or similar
        bs = 2
        ga = 4
        nw = 0 
        attn = None
        print(f"[DESKTOP MODE] Detected {vram_gb:.1f}GB VRAM. Using safe defaults: BS={bs}, GA={ga}")
    
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

ALWAYS_KEEP     = 4          # first N layers always active
COMPUTE_PENALTY = 0.05       # REINFORCE penalty per active routed layer

TIMESTAMP       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME    = f"exp3_dynamic_metrics_{TIMESTAMP}.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser & Dataset
# ──────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("ag_news", split="train")
train_raw = raw.select(range(TRAIN_SAMPLES))
eval_raw  = raw.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))

def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_ds = train_raw.map(preprocess, batched=True, remove_columns=raw.column_names)
eval_ds  = eval_raw.map(preprocess,  batched=True, remove_columns=raw.column_names)
train_ds.set_format("torch")
eval_ds.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_LENGTH)
data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_LENGTH)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    collate_fn=data_collator, num_workers=NUM_WORKERS, pin_memory=True
)
eval_loader  = DataLoader(
    eval_ds, batch_size=BATCH_SIZE, shuffle=False, 
    collate_fn=data_collator, num_workers=NUM_WORKERS, pin_memory=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Model + LoRA + Global Router
# ──────────────────────────────────────────────────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=ATTN_IMPL,
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, lora_cfg)

TOTAL_LAYERS = len(model.base_model.model.model.layers)
ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP

class DynamicRouter(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_layers)
        )
    def forward(self, x):
        return self.net(x)

# Attach router to model and ensure it's trainable
model.router = DynamicRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda", dtype=torch.bfloat16)
for param in model.router.parameters():
    param.requires_grad = True

model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ──────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────────────────────────────────────
def _route_layers(model, input_ids, is_training):
    original_layers: nn.ModuleList = model.base_model.model.model.layers
    
    with torch.no_grad():
        # Get embeddings from base model
        embeds = model.base_model.model.model.embed_tokens(input_ids)
        # Mean pool to get a single vector representing the batch
        pooled_embed = embeds.mean(dim=(0, 1))
        
    # The router decides which layers to keep
    logits = model.router(pooled_embed)
    probs = torch.sigmoid(logits)
    
    if is_training:
        m = torch.distributions.Bernoulli(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions).sum()
    else:
        # Deterministic inference: keep layer if prob > 0.5
        actions = (probs > 0.5).float()
        log_probs = None

    # Build active layer list based on router decisions
    active_list = list(original_layers[:ALWAYS_KEEP])
    for i, layer in enumerate(original_layers[ALWAYS_KEEP:]):
        if actions[i].item() == 1.0:
            active_list.append(layer)

    # Hot-swap the layers
    model.base_model.model.model.layers = nn.ModuleList(active_list)
    
    return original_layers, log_probs, actions

# ──────────────────────────────────────────────────────────────────────────────
# Pure PyTorch Training Loop
# ──────────────────────────────────────────────────────────────────────────────
headers = ["Epoch", "Global Step", "Training Loss", "Validation Loss", "Active Layers"]
with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print("Starting custom pure-PyTorch training loop...")
global_step = 0
val_loss_str = ""

for epoch in range(EPOCHS):
    model.train()
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        # 1. Route Layers
        original_layers, log_probs, actions = _route_layers(model, batch["input_ids"], is_training=True)
        
        # 2. Forward & Backward for LM
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        
        # 3. Policy Gradient for Router
        if log_probs is not None:
            num_used = actions.sum().item()
            cost = loss.detach() + (COMPUTE_PENALTY * num_used)
            router_loss = cost * -log_probs
            router_loss = router_loss / GRAD_ACCUM
            router_loss.backward()

        # 4. Restore Layers
        model.base_model.model.model.layers = original_layers
        
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging & Eval
            if global_step % 10 == 0:
                current_train_loss = loss.item() * GRAD_ACCUM
                num_used = int(actions.sum().item())
                active_count = ALWAYS_KEEP + num_used
                
                # Evaluate every 50 steps
                if global_step % 50 == 0:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for val_batch in eval_loader:
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items()}
                            
                            # Route during eval too!
                            orig_val_layers, _, _ = _route_layers(model, val_batch["input_ids"], is_training=False)
                            
                            val_outputs = model(**val_batch)
                            total_val_loss += val_outputs.loss.item()
                            
                            # Restore
                            model.base_model.model.model.layers = orig_val_layers
                    
                    val_loss = total_val_loss / len(eval_loader)
                    val_loss_str = f"{val_loss:.4f}"
                    model.train()
                
                print(f"Step {global_step} | Layers: {active_count}/{TOTAL_LAYERS} | Train Loss: {current_train_loss:.4f} | Val Loss: {val_loss_str or 'N/A'}")
                
                with open(CSV_FILENAME, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{epoch+1}", global_step, f"{current_train_loss:.4f}", val_loss_str, active_count])

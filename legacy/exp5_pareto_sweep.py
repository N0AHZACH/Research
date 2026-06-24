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
import time

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH      = 128
TRAIN_SAMPLES   = 500   # Fast sweep
EVAL_SAMPLES    = 100

EPOCHS          = 1
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LR              = 5e-5

ALWAYS_KEEP     = 4

PENALTIES       = [0.01, 0.02, 0.05, 0.10, 0.25]

TIMESTAMP       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME    = f"pareto_sweep_metrics_{TIMESTAMP}.csv"

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
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
eval_loader  = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)


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


def _route_layers(model, input_ids, is_training):
    original_layers: nn.ModuleList = model.base_model.model.model.layers
    with torch.no_grad():
        embeds = model.base_model.model.model.embed_tokens(input_ids)
        pooled_embed = embeds.mean(dim=(0, 1))
        
    logits = model.router(pooled_embed)
    probs = torch.sigmoid(logits)
    
    if is_training:
        m = torch.distributions.Bernoulli(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions).sum()
    else:
        actions = (probs > 0.5).float()
        log_probs = None

    active_list = list(original_layers[:ALWAYS_KEEP])
    for i, layer in enumerate(original_layers[ALWAYS_KEEP:]):
        if actions[i].item() == 1.0:
            active_list.append(layer)

    model.base_model.model.model.layers = nn.ModuleList(active_list)
    return original_layers, log_probs, actions

def train_with_penalty(penalty_value):
    print(f"\n==========================================")
    print(f"Starting sweep for Compute Penalty: {penalty_value}")
    print(f"==========================================")
    
    # Reload model clean for each sweep to avoid contamination
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    TOTAL_LAYERS = len(model.base_model.model.model.layers)
    ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP

    model.router = DynamicRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda", dtype=torch.bfloat16)
    for param in model.router.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    total_active_layers_eval = []
    
    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Penalty {penalty_value}")):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            original_layers, log_probs, actions = _route_layers(model, batch["input_ids"], is_training=True)
            
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            
            if log_probs is not None:
                num_used = actions.sum().item()
                cost = loss.detach() + (penalty_value * num_used)
                router_loss = cost * -log_probs
                router_loss = router_loss / GRAD_ACCUM
                router_loss.backward()

            model.base_model.model.model.layers = original_layers
            
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

    # Final Evaluation for this penalty
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_batch in tqdm(eval_loader, desc=f"Eval Penalty {penalty_value}"):
            val_batch = {k: v.to("cuda") for k, v in val_batch.items()}
            orig_val_layers, _, actions = _route_layers(model, val_batch["input_ids"], is_training=False)
            
            total_active_layers_eval.append(ALWAYS_KEEP + actions.sum().item())
            
            val_outputs = model(**val_batch)
            total_val_loss += val_outputs.loss.item()
            model.base_model.model.model.layers = orig_val_layers

    val_loss = total_val_loss / len(eval_loader)
    avg_layers = sum(total_active_layers_eval) / len(total_active_layers_eval)
    
    print(f"--> Penalty: {penalty_value} | Avg Active Layers: {avg_layers:.1f} | Val Loss: {val_loss:.4f}")
    return avg_layers, val_loss

def main():
    headers = ["Compute Penalty", "Avg Active Layers", "Validation Loss"]
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for p in PENALTIES:
        avg_layers, val_loss = train_with_penalty(p)
        with open(CSV_FILENAME, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([p, avg_layers, f"{val_loss:.4f}"])
            
    print(f"\nPareto Sweep complete! Results saved to {CSV_FILENAME}")

if __name__ == "__main__":
    main()

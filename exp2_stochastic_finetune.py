import csv
import os
import random
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
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LR              = 5e-5

ALWAYS_KEEP     = 4          # first N layers always active
DROP_PROB       = 0.5        # probability of randomly dropping each remaining layer

TIMESTAMP       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME    = f"exp2_stochastic_metrics_{TIMESTAMP}.csv"

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

# ──────────────────────────────────────────────────────────────────────────────
# Model + LoRA
# ──────────────────────────────────────────────────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
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
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ──────────────────────────────────────────────────────────────────────────────
# Pure PyTorch Training Loop
# ──────────────────────────────────────────────────────────────────────────────
headers = ["Epoch", "Global Step", "Training Loss", "Validation Loss"]
with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print("Starting custom pure-PyTorch training loop...")
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        original_layers: nn.ModuleList = model.base_model.model.model.layers
        
        # Build active layer list
        active_list = list(original_layers[:ALWAYS_KEEP])
        for layer in original_layers[ALWAYS_KEEP:]:
            if random.random() >= DROP_PROB:
                active_list.append(layer)

        # Hot-swap the layers
        model.base_model.model.model.layers = nn.ModuleList(active_list)
        
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        
        # Restore original layers immediately
        model.base_model.model.model.layers = original_layers
        
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging & Eval
            if global_step % 10 == 0:
                current_train_loss = loss.item() * GRAD_ACCUM
                val_loss_str = "NaN"
                
                # Evaluate every 50 steps
                if global_step % 50 == 0:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for val_batch in eval_loader:
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items()}
                            # NO DROPPING DURING EVAL IN EXP2 (To prove inference mismatch)
                            val_outputs = model(**val_batch)
                            total_val_loss += val_outputs.loss.item()
                    
                    val_loss = total_val_loss / len(eval_loader)
                    val_loss_str = f"{val_loss:.4f}"
                    model.train()
                
                print(f"Step {global_step} | Train Loss: {current_train_loss:.4f} | Val Loss: {val_loss_str}")
                
                with open(CSV_FILENAME, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{epoch+1}", global_step, f"{current_train_loss:.4f}", val_loss_str])

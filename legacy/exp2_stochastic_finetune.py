"""
exp2_stochastic_finetune.py - Phase 1: Stochastic Depth Dropout Baseline

Upgraded to match exp6 / exp1-v2 exactly for a fair manuscript comparison:
  - Dataset  : Wikitext-103-raw-v1 (10,000 train / 1,000 eval samples)
  - LoRA     : r=16, alpha=32, all 4 attention projections (q/k/v/o)
  - Epochs   : 3
  - LR       : 3e-5 cosine decay
  - Drop     : 50% random layer drop during TRAINING only (DROP_PROB=0.5)
  - EVAL     : all 22 layers active (demonstrates inference mismatch)
  - Checkpointing: saves best (by val loss) AND final LoRA adapter

The "inference mismatch" is the key control result: because layers are
dropped randomly at training time but all are active at inference, the
model develops inconsistent representations. DLR avoids this by using
a consistent, learned routing policy at both train and eval time.
"""
import csv
import os
import random
import datetime
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config  (mirror exp6 for fair comparison)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
MAX_EVAL_BATCHES = 100

EPOCHS           = 3
BATCH_SIZE       = 2
GRAD_ACCUM       = 8
LR               = 3e-5
WEIGHT_DECAY     = 0.01

ALWAYS_KEEP      = 4      # first N layers always active
DROP_PROB        = 0.5    # P(dropping each routable layer) during training

EVAL_EVERY_STEPS = 100
LOG_EVERY_STEPS  = 20

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp2_stochastic_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp2_stochastic_output_{TIMESTAMP}"

# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (Wikitext-103 — same as exp1/exp6)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading dataset: wikitext-103-raw-v1 ...")
raw      = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
eval_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

raw      = raw.filter(lambda x: len(x["text"]) > 100)
eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100)
raw      = raw.select(range(min(TRAIN_SAMPLES, len(raw))))
eval_raw = eval_raw.select(range(min(EVAL_SAMPLES, len(eval_raw))))

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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ──────────────────────────────────────────────────────────────────────────────
# Model + LoRA  (identical config to exp6)
# ──────────────────────────────────────────────────────────────────────────────
print("\nLoading TinyLlama with LoRA ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

total_layers    = len(model.base_model.model.model.layers)
routable_layers = total_layers - ALWAYS_KEEP
print(f"  Total layers: {total_layers} | Always-kept: {ALWAYS_KEEP} | Routable: {routable_layers}")
print(f"  Drop prob during training: {DROP_PROB:.0%} | Eval: all {total_layers} layers active")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM
)

# ──────────────────────────────────────────────────────────────────────────────
# CSV logging
# ──────────────────────────────────────────────────────────────────────────────
headers = ["Global Step", "Epoch", "Training Loss", "Validation Loss", "Avg Active Layers Train", "LR"]
with open(CSV_FILENAME, "w", newline="") as f:
    csv.writer(f).writerow(headers)

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nStarting exp2 Stochastic Dropout training ...")
print(f"  Epochs: {EPOCHS} | LR: {LR} | Drop prob: {DROP_PROB}\n")

global_step        = 0
best_val_loss      = float("inf")
val_loss_str       = ""
active_layers_log  = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(epoch_bar):
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # ── Stochastic depth: randomly drop routable layers ─────────────────
        original_layers: nn.ModuleList = model.base_model.model.model.layers
        active_list = list(original_layers[:ALWAYS_KEEP])
        for layer in original_layers[ALWAYS_KEEP:]:
            if random.random() >= DROP_PROB:
                active_list.append(layer)
        n_active = len(active_list)
        active_layers_log.append(n_active)

        model.base_model.model.model.layers = nn.ModuleList(active_list)
        outputs    = model(**batch)
        loss       = outputs.loss
        total_loss = loss / GRAD_ACCUM
        total_loss.backward()
        # Restore IMMEDIATELY before any optimizer state mutation
        model.base_model.model.model.layers = original_layers

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                train_loss  = loss.item()
                current_lr  = scheduler.get_last_lr()[0]
                avg_active  = sum(active_layers_log) / len(active_layers_log) if active_layers_log else 0
                active_layers_log.clear()

                epoch_bar.set_postfix({
                    "loss":   f"{train_loss:.4f}",
                    "layers": f"{avg_active:.1f}",
                    "val":    val_loss_str or "—",
                })

                # ── Evaluation (ALL layers, no dropping — inference mismatch) ─
                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for i, val_batch in enumerate(eval_loader):
                            if i >= MAX_EVAL_BATCHES:
                                break
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items()
                                         if isinstance(v, torch.Tensor)}
                            # No layer dropping at eval — this intentionally
                            # demonstrates the inference mismatch problem.
                            total_val_loss += model(**val_batch).loss.item()

                    n_eval       = min(MAX_EVAL_BATCHES, len(eval_loader))
                    val_loss     = total_val_loss / n_eval
                    val_loss_str = f"{val_loss:.4f}"

                    # ── Save best checkpoint ──────────────────────────────────
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(SAVE_DIR, "best_model"), exist_ok=True)
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        print(f"\n  [SAVED] Best val_loss={val_loss:.4f} at step {global_step}")

                    model.train()

                # ── CSV row ──────────────────────────────────────────────────
                with open(CSV_FILENAME, "a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1,
                        f"{train_loss:.4f}", val_loss_str,
                        f"{avg_active:.1f}", f"{current_lr:.2e}",
                    ])

    print(f"\nEpoch {epoch+1} complete.")

# ──────────────────────────────────────────────────────────────────────────────
# Save final checkpoint
# ──────────────────────────────────────────────────────────────────────────────
print("\nSaving final model checkpoint...")
os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

print(f"\nExp2 Stochastic Dropout training complete!")
print(f"  Metrics CSV      : {CSV_FILENAME}")
print(f"  Best checkpoint  : {SAVE_DIR}/best_model/")
print(f"  Final checkpoint : {SAVE_DIR}/final_model/")

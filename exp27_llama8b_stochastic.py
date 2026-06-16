import csv
import gc
import os
import datetime
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

MODEL_ID         = "meta-llama/Meta-Llama-3.1-8B"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10000
EVAL_SAMPLES     = 1000
EPOCHS           = 3
MAX_EVAL_BATCHES = 100
LR               = 3e-5
WEIGHT_DECAY     = 0.01

EVAL_EVERY_STEPS = 50
LOG_EVERY_STEPS  = 10

def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, True, torch.float32

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_turing = 'T4' in gpu_name or 'RTX 20' in gpu_name or 'Turing' in gpu_name

    if vram_gb >= 70: bs, ga, use_4bit = 8, 2, False
    elif vram_gb >= 35: bs, ga, use_4bit = 4, 4, False
    elif vram_gb >= 22: bs, ga, use_4bit = 2, 8, False
    elif vram_gb >= 14: bs, ga, use_4bit = 2, 8, True
    else: bs, ga, use_4bit = 1, 16, True

    compute_dtype = torch.float16 if is_turing else torch.bfloat16
    cpu_count = os.cpu_count() or 2
    nw = 0 if os.name == 'nt' else min(8, (cpu_count or 4) // 4)
    attn = "flash_attention_2" if vram_gb >= 7 else None
    print(f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f}GB VRAM | BS={bs}, GA={ga}, 4bit={use_4bit}, dtype={compute_dtype}, workers={nw}")
    return bs, ga, nw, attn, use_4bit, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp27_llama8b_stochastic_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp27_llama8b_stochastic_output_{TIMESTAMP}"

def main():
    print(f"\n{'='*70}\n  EXP27: LLAMA3.1-8B FULL-DEPTH STOCHASTIC (32 Layers)\n{'='*70}")
    
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

    class RAMDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            import torch
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            self.input_ids = ids if isinstance(ids, torch.Tensor) else torch.stack(list(ids))
            self.attention_mask = mask if isinstance(mask, torch.Tensor) else torch.stack(list(mask))
            self.labels = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

    pin = torch.cuda.is_available() and (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) > 12 if hasattr(os, 'sysconf') else True)
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    print(f"Loading {MODEL_ID}...")
    if USE_4BIT:
        q_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", quantization_config=q_cfg, attn_implementation=ATTN_IMPL)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=COMPUTE_DTYPE, attn_implementation=ATTN_IMPL)

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # --- STOCHASTIC DROPOUT ---
    import random
    def stochastic_hook(module, input, output):
        # Drop ~50% of the time
        if model.training and random.random() < 0.50:
            residual = input[0]
            return (residual,) + output[1:] if isinstance(output, tuple) else residual
        return output

    handles = []
    # Llama has 32 layers. We skip dropping the first 4.
    num_layers = len(model.base_model.model.model.layers)
    for i in range(4, num_layers):
        handles.append(model.base_model.model.model.layers[i].register_forward_hook(stochastic_hook))


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Step", "Train Loss", "Val Loss", "Perplexity", "Active Layers", "Skip Ratio"])

    global_step = 0
    best_val_loss = float("inf")

    print("\nStarting Training...")
    oom_count = 0
    MAX_OOM_RETRIES = 5

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            
            for batch in pbar:
                global_step += 1
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")

                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / GRAD_ACCUM
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    print(f"\n[OOM] CUDA OOM (occurrence {oom_count}/{MAX_OOM_RETRIES}). Clearing cache and skipping batch...")
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if oom_count >= MAX_OOM_RETRIES:
                        print("[OOM] Too many OOM errors. Saving checkpoint and exiting.")
                        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
                        return
                    continue

                if global_step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                if global_step % LOG_EVERY_STEPS == 0:
                    pbar.set_postfix({"Loss": f"{outputs.loss.item():.4f}"})

                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    val_loss = 0.0
                    eval_batches = 0
                    with torch.no_grad():
                        for ev_batch in eval_loader:
                            ev_inputs = ev_batch["input_ids"].to("cuda")
                            ev_masks = ev_batch["attention_mask"].to("cuda")
                            ev_labels = ev_batch["labels"].to("cuda")
                            outputs = model(input_ids=ev_inputs, attention_mask=ev_masks, labels=ev_labels)
                            val_loss += outputs.loss.item()
                            eval_batches += 1
                            if eval_batches >= MAX_EVAL_BATCHES: break

                    val_loss /= eval_batches
                    ppl = torch.exp(torch.tensor(val_loss)).item()

                    with open(CSV_FILENAME, "a", newline="") as f:
                        csv.writer(f).writerow([epoch, global_step, loss.item() * GRAD_ACCUM, val_loss, ppl, 18.0, 0.50])

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                    model.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user. Saving checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        print("Checkpoint saved successfully. Exiting.")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}. Saving emergency checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        raise

    print("\nSaving final model checkpoint...")
    os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

if __name__ == "__main__":
    main()

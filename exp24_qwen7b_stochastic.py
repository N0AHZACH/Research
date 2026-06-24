import csv
import gc
import os
import datetime
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

MODEL_ID         = "Qwen/Qwen2.5-7B"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10000
EVAL_SAMPLES     = 1000
EPOCHS           = 3
MAX_EVAL_BATCHES = 100
LR               = 3e-5
WEIGHT_DECAY     = 0.01

EVAL_EVERY_STEPS = 50
LOG_EVERY_STEPS  = 10
SEED             = 42

def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, True, torch.float32

    # This script loads the model on cuda:0, so size the batch for that device,
    # not for aggregate VRAM across all visible GPUs.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # is_turing check removed — compute_dtype selected by SM major version below

    if vram_gb >= 80: bs, ga = 16, 1
    elif vram_gb >= 45: bs, ga = 8, 2   # 48GB cards like RTX 6000 Pro
    elif vram_gb >= 35: bs, ga = 8, 2   # 40GB cards like A100
    elif vram_gb >= 22: bs, ga = 4, 4   # 24GB cards like RTX 4090
    elif vram_gb >= 14: bs, ga = 2, 8   # 16GB cards like T4
    else: bs, ga = 1, 16
    use_4bit = False

    major, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.float16 if major < 8 else torch.bfloat16
    cpu_count = os.cpu_count() or 2
    nw = 0  # RAMDataset is fully in-memory; multiprocessing adds massive IPC overhead on Windows
    try:
        import flash_attn
        attn = "flash_attention_2" if vram_gb >= 7 else None
    except ImportError:
        attn = "sdpa" if vram_gb >= 7 else None
    print(f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f}GB VRAM | BS={bs}, GA={ga}, 4bit={use_4bit}, dtype={compute_dtype}, workers={nw} | attn={attn}")
    return bs, ga, nw, attn, use_4bit, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp24_qwen7b_stochastic_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp24_qwen7b_stochastic_output_{TIMESTAMP}"

def main():
    print(f"\n{'='*70}\n  EXP24: QWEN2.5-7B FULL-DEPTH STOCHASTIC (28 Layers)\n{'='*70}")

    # Seed all RNGs for cross-run reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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

    tok_procs = 1 if os.name == "nt" else min(os.cpu_count() or 1, 32)
    train_enc = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names, num_proc=tok_procs)
    eval_enc  = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
    train_enc.set_format("torch")
    eval_enc.set_format("torch")

    class RAMDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=COMPUTE_DTYPE,
        **({"attn_implementation": ATTN_IMPL} if ATTN_IMPL else {}),
    ).to("cuda")
    model.config.use_cache = False  # CRITICAL: Prevent hidden KV cache memory leaks across forward passes

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # --- STOCHASTIC DEPTH CONTROL ---
    # Drop residual branches without train-time expectation scaling to avoid
    # Pre-LN variance spikes. Evaluation uses the expected residual contribution
    # so train/eval activation magnitudes stay aligned.
    TARGET_SKIP = 0.50
    model.stochastic_stats = {"total": 0, "dropped": 0}

    def make_stochastic_forward(original_forward):
        def stochastic_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
            
            if model.training:
                model.stochastic_stats["total"] += 1
                if random.random() < TARGET_SKIP:
                    model.stochastic_stats["dropped"] += 1
                    # Skip computation entirely: saves VRAM and FLOPs
                    return (hidden_states,)
                else:
                    return original_forward(*args, **kwargs)
            else:
                # Eval: compute full layer and apply expected scaling
                output = original_forward(*args, **kwargs)
                layer_out = output[0] if isinstance(output, tuple) else output
                
                # Expected scaling: residual + (1 - p) * F(residual)
                scaled_out = hidden_states + (1.0 - TARGET_SKIP) * (layer_out - hidden_states)
                
                return (scaled_out,) + output[1:] if isinstance(output, tuple) else scaled_out
        return stochastic_forward

    original_forwards = {}
    # Qwen2.5-7B has 28 layers total. We protect the first 4 (embedding layers).
    num_layers = len(model.base_model.model.model.layers)
    assert num_layers == 28, f"Expected 28 layers for Qwen2.5-7B, got {num_layers}"
    for i in range(4, num_layers):
        layer = model.base_model.model.model.layers[i]
        original_forwards[i] = layer.forward
        layer.forward = make_stochastic_forward(layer.forward)
    print(f"[STOCHASTIC] Monkey-patched forward on layers 4-{num_layers-1} ({num_layers-4} routable layers, p=0.50 drop rate)")


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Step", "Train Loss", "Val Loss", "Perplexity", "Active Layers", "Skip Ratio"])

    global_step = 0
    best_val_loss = float("inf")
    last_train_loss = None  # Most recent unscaled training loss; avoids stale values at eval time

    print("\nStarting Training...")
    oom_count = 0
    MAX_OOM_RETRIES = 5

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            
            for step, batch in enumerate(pbar):
                global_step += 1
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")
                stats_before_batch = model.stochastic_stats.copy()

                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / GRAD_ACCUM
                    loss.backward()
                    last_train_loss = outputs.loss.item()  # Capture immediately; safe against later OOM
                except torch.cuda.OutOfMemoryError as e:
                    oom_count += 1
                    print(f"\n[OOM] CUDA OOM (occurrence {oom_count}/{MAX_OOM_RETRIES}). Clearing cache and skipping batch...")
                    import traceback
                    traceback.clear_frames(e.__traceback__)
                    e.__traceback__ = None
                    del e
                    if 'outputs' in locals(): del outputs
                    if 'loss' in locals(): del loss
                    model.stochastic_stats = stats_before_batch
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if oom_count >= MAX_OOM_RETRIES:
                        print("[OOM] Too many OOM errors. Saving checkpoint and exiting.")
                        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
                        return
                    continue

                if global_step % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
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

                    # Empirical skip ratio from the training window preceding this eval
                    total = max(1, model.stochastic_stats["total"])
                    dropped = model.stochastic_stats["dropped"]
                    empirical_skip_ratio = dropped / total
                    routable = num_layers - 4  # Layers eligible for stochastic dropping
                    empirical_active = num_layers - routable * empirical_skip_ratio

                    logged_train_loss = last_train_loss if last_train_loss is not None else float("nan")
                    with open(CSV_FILENAME, "a", newline="") as f:
                        csv.writer(f).writerow([epoch, global_step, logged_train_loss, val_loss, ppl, empirical_active, empirical_skip_ratio])

                    # Reset tracking for next eval window
                    model.stochastic_stats = {"total": 0, "dropped": 0}

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))

                    model.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user. Saving checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        print("Checkpoint saved successfully. Exiting.")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}. Saving emergency checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_latest"))
        raise

    print("\nSaving final model checkpoint...")
    os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

if __name__ == "__main__":
    main()

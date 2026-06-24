import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# EXP23: QWEN2.5-7B BASELINE (FULL DEPTH) — PUBLICATION-QUALITY
#
# Baseline comparison for stochastic depth (exp24) and adaptive routing.
# Matches exp24 exactly in:
#   - LoRA targets (attn + MLP)
#   - Cosine LR schedule with linear warmup
#   - Gradient checkpointing (use_reentrant=False)
#   - Full validation split evaluation
#   - Hardware config and reproducibility settings
#   - Expanded metric logging (throughput, memory)
# =============================================================================

import csv
import gc
import math
import os
import time
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ── Model & Dataset ───────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen2.5-7B"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 10_000
EVAL_SAMPLES     = 1_000
EPOCHS           = 3
LR               = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_STEPS     = 100

# Logging cadence
EVAL_EVERY_STEPS = 50
LOG_EVERY_STEPS  = 10

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
]
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

SEED = 42

# =============================================================================
# Hardware auto-configuration
# =============================================================================

def get_optimal_config():
    if not torch.cuda.is_available():
        return 2, 8, 0, None, False, torch.float32

    vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = False

    if   vram_gb >= 80: bs, ga = 16, 1   # RTX PRO 6000 96 GB
    elif vram_gb >= 45: bs, ga = 8,  2   # 48 GB cards
    elif vram_gb >= 35: bs, ga = 8,  2   # A100 40 GB
    elif vram_gb >= 22: bs, ga = 4,  4   # RTX 4090
    elif vram_gb >= 14: bs, ga = 2,  8   # T4 16 GB
    else:               bs, ga = 1,  16

    major, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.float16 if major < 8 else torch.bfloat16

    nw = 0

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2" if vram_gb >= 7 else None
    except ImportError:
        attn = "sdpa" if vram_gb >= 7 else None

    print(
        f"[HARDWARE] GPU: {gpu_name} | {vram_gb:.1f} GB VRAM | "
        f"BS={bs}, GA={ga}, dtype={compute_dtype}, attn={attn}"
    )
    return bs, ga, nw, attn, False, compute_dtype

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL, USE_4BIT, COMPUTE_DTYPE = get_optimal_config()

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
STEP_CSV    = f"exp23_baseline_step_metrics_{TIMESTAMP}.csv"
SAVE_DIR    = f"exp23_qwen7b_baseline_output_{TIMESTAMP}"

# =============================================================================
# FLOP accounting
# =============================================================================

def estimate_layer_flops_per_token(config, seq_len: int) -> float:
    H       = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv    = getattr(config, "num_key_value_heads", n_heads)
    hd      = H // n_heads
    ff      = config.intermediate_size

    q_flops = 2 * H * H
    k_flops = 2 * H * (n_kv * hd)
    v_flops = 2 * H * (n_kv * hd)
    o_flops = 2 * H * H

    attn_score_flops = 2 * seq_len * H
    attn_value_flops = 2 * seq_len * H

    gate_flops = 2 * H * ff
    up_flops   = 2 * H * ff
    down_flops = 2 * ff * H

    return (
        q_flops + k_flops + v_flops + o_flops
        + attn_score_flops + attn_value_flops
        + gate_flops + up_flops + down_flops
    )

# =============================================================================
# Main training routine
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"  EXP23: QWEN2.5-7B BASELINE (FULL DEPTH)  |  LORA_R={LORA_R}  |  TARGETS={len(LORA_TARGETS)}")
    print(f"{'=' * 70}\n")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    print("Pre-tokenizing dataset...")
    raw      = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    eval_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    raw      = raw.filter(lambda x: len(x["text"]) > 100).select(range(TRAIN_SAMPLES))
    eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100).select(range(EVAL_SAMPLES))

    def tokenize_fn(b):
        o           = tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        o["labels"] = o["input_ids"].copy()
        return o

    tok_procs = 1 if os.name == "nt" else min(os.cpu_count() or 1, 32)
    train_enc = raw.map(tokenize_fn,      batched=True, remove_columns=raw.column_names,      num_proc=tok_procs)
    eval_enc  = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names, num_proc=tok_procs)
    train_enc.set_format("torch")
    eval_enc.set_format("torch")

    class RAMDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            ids  = enc["input_ids"]
            mask = enc["attention_mask"]
            self.input_ids      = ids  if isinstance(ids,  torch.Tensor) else torch.stack(list(ids))
            self.attention_mask = mask if isinstance(mask, torch.Tensor) else torch.stack(list(mask))
            self.labels         = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    pin = torch.cuda.is_available()
    train_loader = DataLoader(RAMDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin)
    eval_loader  = DataLoader(RAMDataset(eval_enc),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=COMPUTE_DTYPE,
        **( {"attn_implementation": ATTN_IMPL} if ATTN_IMPL else {} ),
        token=hf_token,
    ).to("cuda")
    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    print("[GC] Gradient checkpointing enabled (use_reentrant=False)")

    lora_cfg = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    cfg = getattr(model, "config", None) or model.base_model.model.config
    try:
        decoder_layers = model.base_model.model.model.layers
    except (AttributeError, AssertionError):
        decoder_layers = model.model.layers
    num_layers = len(decoder_layers)

    flops_per_token_per_layer = estimate_layer_flops_per_token(cfg, MAX_LENGTH)
    baseline_gflops_per_step  = (
        num_layers * flops_per_token_per_layer * MAX_LENGTH * BATCH_SIZE / 1e9
    )
    print(f"[FLOP] Baseline: {baseline_gflops_per_step:.1f} GFLOPs/step")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = WARMUP_STEPS,
        num_training_steps = total_steps,
    )
    print(f"[LR] Cosine schedule | warmup={WARMUP_STEPS} steps | total={total_steps} steps")

    os.makedirs(SAVE_DIR, exist_ok=True)

    step_csv_header = [
        "epoch", "global_step", "optimizer_step",
        "train_loss", "val_loss", "perplexity",
        "active_layer_frac", "skip_ratio",
        "step_time_s", "tokens_per_sec",
        "peak_gpu_mem_gb", "lr",
        "expected_flop_reduction_pct", "empirical_flop_reduction_pct",
        "baseline_gflops", "executed_gflops",
    ]
    with open(STEP_CSV, "w", newline="") as f:
        csv.writer(f).writerow(step_csv_header)

    print(f"[LOG] Step metrics → {STEP_CSV}")

    global_step    = 0
    optimizer_step = 0
    best_val_loss  = float("inf")
    last_train_loss = None
    oom_count       = 0
    MAX_OOM_RETRIES = 5

    print("\nStarting Training...")

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

            for step, batch in enumerate(pbar):
                global_step += 1

                input_ids      = batch["input_ids"].to("cuda", non_blocking=True)
                attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
                labels         = batch["labels"].to("cuda", non_blocking=True)

                torch.cuda.reset_peak_memory_stats()
                step_start = time.perf_counter()

                try:
                    outputs         = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss            = outputs.loss / GRAD_ACCUM
                    loss.backward()
                    last_train_loss = outputs.loss.item()

                except torch.cuda.OutOfMemoryError as e:
                    oom_count += 1
                    print(
                        f"\n[OOM] CUDA OOM (occurrence {oom_count}/{MAX_OOM_RETRIES}). "
                        f"Clearing cache and skipping batch..."
                    )
                    import traceback
                    traceback.clear_frames(e.__traceback__)
                    e.__traceback__ = None
                    del e

                    if "outputs" in dir(): del outputs
                    if "loss"    in dir(): del loss

                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()

                    if oom_count >= MAX_OOM_RETRIES:
                        print("[OOM] Too many OOM errors. Saving checkpoint and exiting.")
                        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_oom"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_oom"))
                        return
                    continue

                if global_step % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

                step_time    = time.perf_counter() - step_start
                tokens_per_s = (BATCH_SIZE * MAX_LENGTH) / max(step_time, 1e-9)
                peak_mem_gb  = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                if global_step % LOG_EVERY_STEPS == 0 and last_train_loss is not None:
                    pbar.set_postfix({
                        "loss":  f"{last_train_loss:.4f}",
                        "mem":   f"{peak_mem_gb:.1f}GB",
                        "tok/s": f"{tokens_per_s:.0f}",
                    })

                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    val_loss     = 0.0
                    eval_batches = 0

                    with torch.no_grad():
                        for ev_batch in eval_loader:
                            ev_out = model(
                                input_ids      = ev_batch["input_ids"].to("cuda", non_blocking=True),
                                attention_mask = ev_batch["attention_mask"].to("cuda", non_blocking=True),
                                labels         = ev_batch["labels"].to("cuda", non_blocking=True),
                            )
                            val_loss     += ev_out.loss.item()
                            eval_batches += 1

                    val_loss   /= eval_batches
                    perplexity  = math.exp(min(val_loss, 300))

                    current_lr = scheduler.get_last_lr()[0]
                    logged_train_loss = last_train_loss if last_train_loss is not None else float("nan")

                    with open(STEP_CSV, "a", newline="") as f:
                        csv.writer(f).writerow([
                            epoch,
                            global_step,
                            optimizer_step,
                            f"{logged_train_loss:.6f}",
                            f"{val_loss:.6f}",
                            f"{perplexity:.4f}",
                            f"{1.0:.4f}", # active_layer_frac
                            f"{0.0:.4f}", # skip_ratio
                            f"{step_time:.4f}",
                            f"{tokens_per_s:.1f}",
                            f"{peak_mem_gb:.3f}",
                            f"{current_lr:.2e}",
                            f"{0.0:.2f}", # expected_flop_reduction_pct
                            f"{0.0:.2f}", # empirical_flop_reduction_pct
                            f"{baseline_gflops_per_step:.2f}",
                            f"{baseline_gflops_per_step:.2f}", # executed_gflops = baseline
                        ])

                    print(
                        f"\n[EVAL  E{epoch} S{global_step}] "
                        f"val_loss={val_loss:.4f}  ppl={perplexity:.2f}  "
                        f"mem={peak_mem_gb:.1f}GB  "
                        f"tok/s={tokens_per_s:.0f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        print(f"  ↳ New best val_loss={best_val_loss:.4f}. Checkpoint saved.")

                    model.train()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted. Saving checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_interrupt"))
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_interrupt"))
        return

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}. Saving emergency checkpoint...")
        model.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_error"))
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "checkpoint_error"))
        raise

    print("\nSaving final model checkpoint...")
    os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

    print(f"\n{'=' * 70}")
    print(f"  Training complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Step metrics  : {STEP_CSV}")
    print(f"  Checkpoints   : {SAVE_DIR}/")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()

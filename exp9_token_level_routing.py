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
COMPUTE_PENALTY  = 2.0   # Increased: force router past the 10% saturation plateau
GUMBEL_TEMP      = 1.0
TEMP_ANNEAL_RATE = 0.95
KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0
GATE_ENTROPY_BETA = 0.0  # Disabled: was counteracting compute penalty, causing 10% collapse
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

def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Phase 2: Token-Level Dynamic Routing")
    parser.add_argument("--resume", type=str, default="auto", 
                        help="Path to checkpoint directory or 'auto' to auto-resume latest, or 'none' to start fresh")
    parser.add_argument("--fresh", action="store_true", 
                        help="Start training from scratch, ignoring existing checkpoints")
    args = parser.parse_args()

    if args.fresh:
        args.resume = "none"

    csv_filename = CSV_FILENAME
    save_dir = SAVE_DIR

    start_epoch = 0
    start_step = -1
    global_step = 0
    best_val_loss = float("inf")
    current_temp = GUMBEL_TEMP

    def find_latest_checkpoint():
        dirs = glob.glob("exp9_token_output_*")
        valid_checkpoints = []
        for d in dirs:
            ckpt_path = os.path.join(d, "checkpoint_latest", "training_states.pt")
            if os.path.exists(ckpt_path):
                mtime = os.path.getmtime(ckpt_path)
                valid_checkpoints.append((mtime, d))
        if not valid_checkpoints:
            return None
        valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
        return valid_checkpoints[0][1]

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
            # Initialize output layer with strong negative bias so gates start near 0.
            # This forces the router to *earn* activations via CE loss pressure rather
            # than defaulting to ~10% skip (local minimum found in exp8 turbo sweep).
            nn.init.constant_(self.net[-1].bias, -2.0)

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

    # Load checkpoint if exists and requested
    checkpoint_dir = None
    if args.resume == "auto":
        latest_dir = find_latest_checkpoint()
        if latest_dir:
            checkpoint_dir = os.path.join(latest_dir, "checkpoint_latest")
            save_dir = latest_dir
            # Derive CSV file name from the output directory
            csv_filename = f"{latest_dir.replace('exp9_token_output_', 'exp9_token_level_routing_')}.csv"
    elif args.resume and args.resume.lower() != "none":
        checkpoint_dir = args.resume
        parent_dir = os.path.dirname(checkpoint_dir)
        if parent_dir:
            save_dir = parent_dir
            csv_filename = f"{parent_dir.replace('exp9_token_output_', 'exp9_token_level_routing_')}.csv"
        else:
            save_dir = checkpoint_dir

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"\n[CHECKPOINT] Loading checkpoint from: {checkpoint_dir}")
        
        # 1. Load LoRA adapter weights
        lora_loaded = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapters_weights = load_file(fpath)
                else:
                    adapters_weights = torch.load(fpath, map_location="cuda")
                model.load_state_dict(adapters_weights, strict=False)
                lora_loaded = True
                print(f"  Loaded LoRA adapters from {fpath}")
                break
        if not lora_loaded:
            print("  Warning: No LoRA adapter file found in checkpoint.")

        # 2. Load router weights
        router_path = os.path.join(checkpoint_dir, "router_weights.pt")
        if os.path.exists(router_path):
            model.router.load_state_dict(torch.load(router_path, map_location="cuda"))
            print(f"  Loaded router weights from {router_path}")
        else:
            print("  Warning: No router weights found in checkpoint.")

        # 3. Load optimizer, scheduler, and training metadata
        states_path = os.path.join(checkpoint_dir, "training_states.pt")
        if os.path.exists(states_path):
            states = torch.load(states_path, map_location="cuda")
            start_epoch = states["epoch"]
            start_step = states["step"]
            global_step = states["global_step"]
            best_val_loss = states["best_val_loss"]
            current_temp = states["current_temp"]
            if "csv_filename" in states:
                csv_filename = states["csv_filename"]
            
            optimizer.load_state_dict(states["optimizer_state_dict"])
            scheduler.load_state_dict(states["scheduler_state_dict"])
            print(f"  Resumed training states: epoch={start_epoch+1}, step={start_step}, global_step={global_step}, temp={current_temp:.4f}")
        else:
            print("  Warning: No training states file found in checkpoint. Starting from epoch 1.")

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

    def save_checkpoint(epoch, step, global_step, best_val_loss, current_temp):
        checkpoint_dir = os.path.join(save_dir, "checkpoint_latest")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        torch.save(model.router.state_dict(), os.path.join(checkpoint_dir, "router_weights.pt"))
        checkpoint_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "current_temp": current_temp,
            "csv_filename": csv_filename,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint_states, os.path.join(checkpoint_dir, "training_states.pt"))
        print(f"\n[CHECKPOINT] Saved checkpoint to {checkpoint_dir} at epoch {epoch+1}, step {step}, global_step {global_step}")

    # ==============================================================================
    # Training Loop
    # ==============================================================================
    print(f"\nStarting Phase 2 TOKEN-LEVEL Router training...")

    headers = ["Global Step", "Epoch", "Training Loss", "CE Loss", "KD Loss",
               "Gate Loss", "Validation Loss", "Avg Active Layers", "Gumbel Temp", "LR"]
    
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
        with open(csv_filename, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            optimizer.zero_grad()
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for step, batch in enumerate(epoch_bar):
                if step <= start_step:
                    continue
                
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
                                os.makedirs(os.path.join(save_dir, "best_model"), exist_ok=True)
                                model.save_pretrained(os.path.join(save_dir, "best_model"))
                                tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
                                torch.save(model.router.state_dict(), os.path.join(save_dir, "best_model", "router_weights.pt"))

                            # Save latest resume checkpoint
                            save_checkpoint(epoch, step, global_step, best_val_loss, current_temp)

                            model.train()

                            with open(csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow([
                                    global_step, epoch + 1, f"{total_loss.item() * GRAD_ACCUM:.4f}", 
                                    f"{ce_loss.item():.4f}", f"{kd_loss.item():.4f}", f"{gate_loss.item():.4f}", 
                                    f"{val_loss:.4f}", f"{val_avg_layers:.1f}", f"{current_temp:.4f}", 
                                    f"{scheduler.get_last_lr()[0]:.2e}"
                                ])

            # Reset start_step after completing the epoch
            start_step = -1

            # Save checkpoint at the end of the epoch
            save_checkpoint(epoch, -1, global_step, best_val_loss, current_temp)

            current_temp = max(0.5, current_temp * TEMP_ANNEAL_RATE)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user. Saving checkpoint...")
        curr_step = step if 'step' in locals() else -1
        curr_epoch = epoch if 'epoch' in locals() else start_epoch
        save_checkpoint(curr_epoch, curr_step, global_step, best_val_loss, current_temp)
        print("Checkpoint saved successfully. Exiting.")
        return

    print("\nSaving final model checkpoint...")
    os.makedirs(os.path.join(save_dir, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))
    torch.save(model.router.state_dict(), os.path.join(save_dir, "final_model", "router_weights.pt"))

    # Cleanup checkpoint folder since training completed successfully
    checkpoint_dir = os.path.join(save_dir, "checkpoint_latest")
    if os.path.exists(checkpoint_dir):
        import shutil
        try:
            shutil.rmtree(checkpoint_dir)
            print("Cleaned up checkpoint folder.")
        except Exception as e:
            print(f"Warning: could not clean up checkpoint folder: {e}")

    print("Done!")

if __name__ == '__main__':
    main()

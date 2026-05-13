"""
exp6_gumbel_router.py - Phase 3: Production-Grade Dynamic Layer Routing

Fixes over exp3 (REINFORCE baseline):
  Fix 1: Per-sample routing    - router scores each sample independently (per-sample gates)
  Fix 2: Gumbel-Softmax STE    - replaces high-variance REINFORCE; fully differentiable
  Fix 3: Contextual hidden states - router reads h_4 (post always-kept layers), not raw embeddings
  Fix 4: Scaled dataset        - wikitext-103-raw-v1, 3 epochs
  Fix 5: Model checkpointing   - saves LoRA adapter + router weights after training
  Fix 6: Router in optimizer   - router params included in AdamW via itertools.chain

Implementation note: We use forward hooks to intercept and gate layer outputs.
This lets the model handle all internal details (rotary embeddings, SDPA masking)
while we surgically apply per-sample skip-connection gates.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# Max eval batches per evaluation pass. The full 1000-sample eval set at
# batch_size=2 means 500 batches — far too slow to run every 100 steps.
# Capping at 100 batches (200 samples) gives a fast, reliable signal.
MAX_EVAL_BATCHES = 100

BATCH_SIZE       = 2
GRAD_ACCUM       = 8
LR               = 3e-5
WEIGHT_DECAY     = 0.01

ALWAYS_KEEP      = 4
COMPUTE_PENALTY  = 1.0   # Escalated from 0.4 to stop the upward drift to 19+
GUMBEL_TEMP      = 1.0
TEMP_ANNEAL_RATE = 0.95
KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0   # was 3.0 — T² amplification was 9×, now 4×
# Gate entropy bonus: penalizes gates that are always 0 or always 1.
# Encourages the router to use the full probability range instead of saturating.
GATE_ENTROPY_BETA = 0.1  # Escalated from 0.05
# Number of gradient-update steps before KD loss is turned on.
KD_WARMUP_STEPS  = 50

EVAL_EVERY_STEPS = 100
LOG_EVERY_STEPS  = 20

# ---------------------------------------------------------------------------
# Hardware Auto-Optimisation (Server Mode)
# ---------------------------------------------------------------------------
def get_optimal_config():
    """Detects hardware and returns optimized training parameters."""
    if not torch.cuda.is_available():
        return 2, 8, 0, None  # Default fallback

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cpu_count = os.cpu_count() or 4
    
    # Enable TF32 for Ampere+ GPUs (RTX 3000/4000)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if vram_gb >= 15:
        # RTX 4000 20GB or similar server-grade
        bs = 8   # Lowered from 16 to prevent OOM
        ga = 2   # Effective BS stays at 16
        nw = min(cpu_count // 2, 8)  # Utilize the i9-13900KF
        attn = "sdpa"
        print(f"[SERVER MODE] Detected {vram_gb:.1f}GB VRAM. Scaling: BS={bs}, GA={ga}, Workers={nw}, SDPA=ON")
    else:
        # RTX 4060 8GB or similar
        bs = 2
        ga = 8
        nw = 0  # Windows workers can be flaky on laptops
        attn = None
        print(f"[DESKTOP MODE] Detected {vram_gb:.1f}GB VRAM. Using safe defaults: BS={bs}, GA={ga}")
    
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp6_gumbel_metrics_{TIMESTAMP}.csv"
SAVE_DIR     = f"exp6_gumbel_output_{TIMESTAMP}"

# ==============================================================================
# Fix 4: Dataset - Wikitext-103
# ==============================================================================
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

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=NUM_WORKERS, pin_memory=True
)
eval_loader  = DataLoader(
    eval_ds,  batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ==============================================================================
# Models: Student (LoRA) + Teacher (Frozen) for KD
# ==============================================================================
print("\nLoading TinyLlama student (LoRA) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    attn_implementation=ATTN_IMPL,
)

print("Loading frozen Teacher for KD ...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    attn_implementation=ATTN_IMPL,
)
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
# Fix 1, 2, 3: Gumbel-Softmax Router (per-sample, contextual input)
# ==============================================================================
class GumbelRouter(nn.Module):
    """
    Per-sample router using Gumbel-Softmax Straight-Through Estimator.
    Input: pooled hidden state from layer ALWAYS_KEEP (contextually informed).
    Output: [batch_size, ROUTABLE_LAYERS] binary gates.
    """
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        # Note: LayerNorm is intentionally omitted here.
        # nn.LayerNorm stores weights in bfloat16 when the module is cast to
        # bfloat16, but its internal CUDA kernel up-casts to float32 during
        # the forward pass. This creates a subtle dtype mismatch that can
        # cause instability. A plain GELU-MLP is more reliable in mixed
        # precision without sacrificing routing quality.
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )

    def forward(self, pooled_h: torch.Tensor, temperature: float, hard: bool = True):
        """
        pooled_h: [B, H] — cast to float32 before entering the net.
        Gumbel noise sampling is sensitive to precision: running in bfloat16
        causes quantised noise that collapses the gate distribution.
        float32 here adds negligible overhead (router is tiny vs. the LLM).
        Returns gates: [B, num_layers] binary if hard=True (STE in backward)
        """
        pooled_h  = pooled_h.float()                              # precision for Gumbel
        logits    = self.net(pooled_h)                            # [B, L]
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # [B, L, 2]
        soft      = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(pooled_h.dtype)                   # [B, L]  restore dtype


# Router weights kept in float32 for numerical stability (Gumbel sampling).
# The router is tiny (~5M params) so memory cost is negligible.
model.router = GumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
for p in model.router.parameters():
    p.requires_grad = True

# Fix 6: Include router parameters in the optimizer so they are actually updated.
# Previously only model.parameters() (LoRA) were passed, leaving the router
# weights frozen despite requires_grad=True. Using itertools.chain combines both
# parameter sets into a single iterable without creating a new list copy.
optimizer = torch.optim.AdamW(
    itertools.chain(model.parameters(), model.router.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM
)

# ==============================================================================
# Hook-based Gated Forward Pass
# ==============================================================================
class GatedForwardContext:
    """
    Context manager that installs forward hooks on the routable transformer layers.
    Each hook applies a per-sample skip-connection gate to the layer output.
    
    This is the recommended approach as it lets the model handle all internal
    details (rotary embeddings, SDPA masking) natively, while we surgically
    gate the outputs.
    
    Fix 1: gates are [B, ROUTABLE_LAYERS] - per sample
    Fix 2: gates are Gumbel-STE - smooth gradient
    Fix 3: gates are computed AFTER the ALWAYS_KEEP layers run
    """
    def __init__(self):
        self.gates = None
        self.handles = []
        # Capture hidden state after the ALWAYS_KEEP-th layer
        self.captured_h = None
        self.layer_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def install_capture_hook(self, layer):
        """After ALWAYS_KEEP layers, capture the hidden state for routing."""
        def hook(module, input, output):
            # PEFT-wrapped layers return the hidden state tensor directly [B, seq, H]
            # Bare layers return a tuple where output[0] is the hidden state
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.captured_h = hidden_state.detach().float().mean(dim=1)  # [B, H]
        h = layer.register_forward_hook(hook)
        self.handles.append(h)

    def install_gate_hooks(self, layers, gates):
        """Install per-layer skip-connection hooks on routable layers."""
        self.gates = gates
        for i, layer in enumerate(layers):
            idx = i  # capture by value

            def hook(module, input, output, layer_i=idx):
                # input[0] is the layer input (residual stream before this layer)
                residual = input[0]
                # Handle both PEFT (tensor) and bare (tuple) layer outputs
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                gate = self.gates[:, layer_i].view(-1, 1, 1).to(h.dtype)
                gated_h = gate * h + (1.0 - gate) * residual
                return (gated_h,) + output[1:] if is_tuple else gated_h

            h = layer.register_forward_hook(hook)
            self.handles.append(h)


def gated_forward(model, batch, temperature, hard=True):
    """
    Two-pass strategy using hooks:
    Pass 1: Run model normally up to layer ALWAYS_KEEP to capture h_{ALWAYS_KEEP} (Fix 3)
    Pass 2: Compute gates from h_{ALWAYS_KEEP}, install gate hooks, run full model (Fix 1, 2)
    """
    input_ids      = batch["input_ids"]
    labels         = batch.get("labels", None)
    attention_mask = batch.get("attention_mask", None)
    
    transformer = model.base_model.model.model
    all_layers  = transformer.layers

    ctx = GatedForwardContext()

    # --- Pass 1: Capture contextual hidden state after ALWAYS_KEEP layers ---
    # Install a capture hook on the last always-kept layer
    ctx.install_capture_hook(all_layers[ALWAYS_KEEP - 1])

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    ctx.remove_hooks()

    # --- Compute per-sample gates from captured h ---  # Fix 1, 2, 3
    # captured_h is already float32 (detached + cast in the capture hook).
    # The router forward handles the float32 path internally.
    pooled_h = ctx.captured_h.to("cuda")                      # [B, H] float32
    gates = model.router(pooled_h, temperature=temperature, hard=hard)  # [B, ROUTABLE_LAYERS]

    # --- Pass 2: Full forward with gate hooks installed ---
    ctx.install_gate_hooks(all_layers[ALWAYS_KEEP:], gates)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    ctx.remove_hooks()

    ce_loss = outputs.loss
    logits  = outputs.logits

    return logits, ce_loss, gates


def compute_kd_loss(s_logits, t_logits, T):
    """KL-Divergence KD loss with temperature T."""
    return F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits  / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)


# ==============================================================================
# CSV Logging Setup
# ==============================================================================
headers = ["Global Step", "Epoch", "Training Loss", "CE Loss", "KD Loss",
           "Gate Loss", "Validation Loss", "Avg Active Layers", "Gumbel Temp", "LR"]
with open(CSV_FILENAME, "w", newline="") as f:
    csv.writer(f).writerow(headers)

# ==============================================================================
# Training Loop
# ==============================================================================
print(f"\nStarting Phase 3 Gumbel Router training...")
print(f"  Epochs: {EPOCHS} | Penalty: {COMPUTE_PENALTY} | Gumbel Temp: {GUMBEL_TEMP} | KD alpha: {KD_ALPHA}\n")

global_step    = 0
current_temp   = GUMBEL_TEMP
best_val_loss  = float("inf")
val_loss_str   = ""
avg_layers_str = ""

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(epoch_bar):
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Student forward (gated)
        student_logits, ce_loss, gates = gated_forward(model, batch, temperature=current_temp, hard=True)

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_logits = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits

        # Sparsity loss: penalize fraction of gates active
        gate_loss = gates.float().mean() * COMPUTE_PENALTY

        # Gate entropy bonus: rewards a gate distribution that uses the full [0,1]
        # range rather than collapsing to "always 1" (router avoidance of penalty).
        # We use the per-sample mean gate activation as a Bernoulli probability proxy.
        # H(p) = -p*log(p) - (1-p)*log(1-p) is maximized at p=0.5. Subtracting this
        # from the loss encourages the router to keep gates near 50%, creating pressure
        # that opposes both the KD loss (which drives gates to 1) and the sparsity
        # penalty (which drives gates to 0).
        p_mean = gates.detach().float().mean()  # scalar, ∈ [0,1]
        eps = 1e-6
        gate_entropy = -(p_mean * (p_mean + eps).log() + (1 - p_mean) * (1 - p_mean + eps).log())
        entropy_bonus = GATE_ENTROPY_BETA * gate_entropy  # subtract → maximizes entropy

        # KD + combined loss.
        # During warmup (global_step < KD_WARMUP_STEPS): pure CE + gate only.
        #   Rationale: T²-scaled KL divergence explodes when the student logits
        #   are far from the teacher early in training (observed KD=1864 at step 20).
        #   Running pure CE for the first KD_WARMUP_STEPS optimizer steps lets the
        #   student converge enough that KD gradients are meaningful, not destructive.
        # After warmup: blended alpha*CE + (1-alpha)*KD + gate.
        if global_step < KD_WARMUP_STEPS:
            kd_loss    = torch.tensor(0.0, device="cuda")  # not computed, log as 0
            total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
        else:
            kd_loss    = compute_kd_loss(student_logits[:, :-1, :], teacher_logits[:, :-1, :], KD_TEMPERATURE)
            total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            # Clip gradients for both the LoRA model AND the router.
            # Previously only model.parameters() was clipped, leaving router
            # gradients unconstrained which amplified early training instability.
            all_params = list(model.parameters()) + list(model.router.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                avg_layers     = gates.detach().float().mean(dim=0).sum().item() + ALWAYS_KEEP
                avg_layers_str = f"{avg_layers:.1f}"
                current_lr     = scheduler.get_last_lr()[0]
                train_loss     = total_loss.item() * GRAD_ACCUM
                in_warmup      = global_step < KD_WARMUP_STEPS
                ce_val         = ce_loss.item()
                kd_val         = kd_loss.item()
                gate_val       = gate_loss.item()

                epoch_bar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "ce":   f"{ce_val:.4f}",
                    "kd":   f"{kd_val:.4f}" + ("(warm)" if in_warmup else ""),
                    "layers": avg_layers_str,
                    "temp": f"{current_temp:.3f}",
                })

                # Evaluation — capped at MAX_EVAL_BATCHES to prevent the eval
                # loop from running for ~500 batches every 100 steps, which
                # was likely causing the early terminations observed in prior runs.
                if global_step % EVAL_EVERY_STEPS == 0:
                    model.eval()
                    total_val_loss = 0.0
                    total_active   = []
                    with torch.no_grad():
                        for i, val_batch in enumerate(eval_loader):
                            if i >= MAX_EVAL_BATCHES:
                                break
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                            # Use hard=True at eval so the layer count reflects the actual
                            # binary routing decisions (not soft-gate averages).
                            _, v_ce, v_gates = gated_forward(model, val_batch, temperature=current_temp, hard=True)
                            total_val_loss += v_ce.item()
                            total_active.append(v_gates.float().mean(dim=0).sum().item() + ALWAYS_KEEP)

                    n_eval_batches = min(MAX_EVAL_BATCHES, len(eval_loader))
                    val_loss       = total_val_loss / n_eval_batches
                    val_avg_layers = sum(total_active) / len(total_active)
                    val_loss_str   = f"{val_loss:.4f}"
                    avg_layers_str = f"{val_avg_layers:.1f}"

                    # Fix 5: Save best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs(os.path.join(SAVE_DIR, "best_model"), exist_ok=True)
                        model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
                        torch.save(model.router.state_dict(), os.path.join(SAVE_DIR, "best_model", "router_weights.pt"))
                        print(f"\n  [SAVED] Best val_loss={val_loss:.4f} at step {global_step}")

                    model.train()

                # CSV log
                with open(CSV_FILENAME, "a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1,
                        f"{train_loss:.4f}", f"{ce_val:.4f}", f"{kd_val:.4f}",
                        f"{gate_val:.4f}", val_loss_str, avg_layers_str,
                        f"{current_temp:.4f}", f"{current_lr:.2e}",
                    ])

    # Anneal Gumbel temperature each epoch
    current_temp = max(0.5, current_temp * TEMP_ANNEAL_RATE)
    print(f"\nEpoch {epoch+1} complete. Gumbel temp annealed to {current_temp:.3f}")

# ==============================================================================
# Fix 5: Save Final Checkpoint
# ==============================================================================
print("\nSaving final model checkpoint...")
os.makedirs(os.path.join(SAVE_DIR, "final_model"), exist_ok=True)
model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
torch.save(model.router.state_dict(), os.path.join(SAVE_DIR, "final_model", "router_weights.pt"))

print(f"\nPhase 3 training complete!")
print(f"  Metrics CSV      : {CSV_FILENAME}")
print(f"  Best checkpoint  : {SAVE_DIR}/best_model/")
print(f"  Final checkpoint : {SAVE_DIR}/final_model/")

"""
exp8_gumbel_pareto_sweep.py - Phase 4 Setup: Gumbel Router Pareto Frontier Sweep

Upgrades the old REINFORCE-based exp5_pareto_sweep.py to use the production-grade
Gumbel-STE router from exp6. Trains one model per compute_penalty value,
records (avg_active_layers, val_loss) for each, and generates the Pareto curve.

Sweep grid: PENALTIES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
This gives 6 points spanning the full efficiency/accuracy tradeoff.

Output:
  - exp8_gumbel_pareto_<timestamp>.csv     : (penalty, avg_layers, val_loss, skip_ratio)
  - exp8_gumbel_pareto_<timestamp>.png     : Pareto frontier plot

Usage:
  python exp8_gumbel_pareto_sweep.py
  python exp8_gumbel_pareto_sweep.py --fast   # fewer steps, for sanity check
"""

import os
import itertools
import csv
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gumbel Pareto Sweep")
parser.add_argument("--fast", action="store_true",
                    help="Fast mode: 1000 train samples, 1 epoch (sanity check)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH   = 512
ALWAYS_KEEP  = 4

# Sweep parameters — denser at low end where router is most sensitive
PENALTIES    = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

# Training hyperparameters (per penalty run)
if args.fast:
    TRAIN_SAMPLES    = 1000
    EVAL_SAMPLES     = 200
    EPOCHS           = 1
    MAX_EVAL_BATCHES = 50
    print("[FAST MODE] 1 epoch, 1000 train samples — sanity check only")
else:
    # Standard Research Mode (Full 3-Epoch Training)
    TRAIN_SAMPLES    = 10_000
    EVAL_SAMPLES     = 1_000
    EPOCHS           = 3
    MAX_EVAL_BATCHES = 100



BATCH_SIZE       = 2
GRAD_ACCUM       = 8
LR               = 3e-5
WEIGHT_DECAY     = 0.01

GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_MIN   = 0.5
TEMP_ANNEAL_RATE  = 0.95

KD_ALPHA         = 0.5
KD_TEMPERATURE   = 2.0   # was 3.0 — matching exp6 fix (T² amplification 4× not 9×)
KD_WARMUP_STEPS  = 30
GATE_ENTROPY_BETA = 0.1  # Escalated to match exp6

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

    if vram_gb >= 7:
        # Optimized Laptop/Desktop Mode (RTX 4060/3060+)
        bs = 4   # Reduced physical batch size to prevent VRAM spilling
        ga = 4   # Effective BS = 16
        nw = 0   # Disable workers to prevent any Windows multiprocessing overhead
        attn = "sdpa"
        print(f"[LAPTOP OPTIMIZED] Hardware: {vram_gb:.1f}GB VRAM | BS: {bs} | Workers: {nw} | SDPA: ON")
    else:
        # Fallback for very low VRAM
        bs = 2
        ga = 8
        nw = 0
        attn = None
        print(f"[SAFE MODE] Low VRAM ({vram_gb:.1f}GB). Using safe defaults.")
    
    return bs, ga, nw, attn

BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS, ATTN_IMPL = get_optimal_config()

TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"exp8_gumbel_pareto_{TIMESTAMP}.csv"
PLOT_FILE    = f"exp8_gumbel_pareto_{TIMESTAMP}.png"

# ---------------------------------------------------------------------------
# Dataset (shared across all penalty runs)
# ---------------------------------------------------------------------------
print("Loading dataset: wikitext-103-raw-v1 ...")
raw      = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
eval_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
raw      = raw.filter(lambda x: len(x["text"]) > 100)
eval_raw = eval_raw.filter(lambda x: len(x["text"]) > 100)
raw      = raw.select(range(min(TRAIN_SAMPLES, len(raw))))
eval_raw = eval_raw.select(range(min(EVAL_SAMPLES,  len(eval_raw))))

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
# Convert to pure PyTorch memory dataset to bypass Windows Arrow mapping crashes
# This safely enables num_workers > 0 on Windows!
class RAMDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.data = [{"input_ids": hf_ds[i]["input_ids"], 
                      "attention_mask": hf_ds[i]["attention_mask"], 
                      "labels": hf_ds[i]["labels"]} for i in range(len(hf_ds))]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

train_ds = RAMDataset(train_ds)
eval_ds  = RAMDataset(eval_ds)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=NUM_WORKERS, pin_memory=True
)
eval_loader  = DataLoader(
    eval_ds,  batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---------------------------------------------------------------------------
# GumbelRouter (identical to exp6)
# ---------------------------------------------------------------------------
class GumbelRouter(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_layers),
        )

    def forward(self, pooled_h: torch.Tensor, temperature: float, hard: bool = True):
        pooled_h  = pooled_h.float()
        logits    = self.net(pooled_h)
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        soft      = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return soft[..., 1].to(pooled_h.dtype)

# ---------------------------------------------------------------------------
# Hook-based gated forward (identical to exp6)
# ---------------------------------------------------------------------------
class StopForwardException(Exception): pass

class GatedForwardContext:
    def __init__(self):
        self.gates      = None
        self.handles    = []
        self.captured_h = None

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def install_early_stop_hook(self, layer):
        def hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.captured_h = hidden_state.detach().float().mean(dim=1)
            raise StopForwardException()
        self.handles.append(layer.register_forward_hook(hook))

    def install_gate_hooks(self, layers, gates):
        self.gates = gates
        for i, layer in enumerate(layers):
            idx = i
            def hook(module, input, output, layer_i=idx):
                residual = input[0]
                is_tuple = isinstance(output, tuple)
                h    = output[0] if is_tuple else output
                gate = self.gates[:, layer_i].view(-1, 1, 1).to(h.dtype)
                gated_h = gate * h + (1.0 - gate) * residual
                return (gated_h,) + output[1:] if is_tuple else gated_h
            self.handles.append(layer.register_forward_hook(hook))


def gated_forward(model, batch, temperature, hard=True):
    input_ids      = batch["input_ids"]
    labels         = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    transformer    = model.base_model.model.model
    all_layers     = transformer.layers
    ctx = GatedForwardContext()

    ctx.install_early_stop_hook(all_layers[ALWAYS_KEEP - 1])
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForwardException:
        pass
    finally:
        ctx.remove_hooks()

    pooled_h = ctx.captured_h.to("cuda")
    gates    = model.router(pooled_h, temperature=temperature, hard=hard)

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

# ---------------------------------------------------------------------------
# Single penalty training run
# ---------------------------------------------------------------------------
def train_one_penalty(penalty: float, teacher_model, base_model) -> dict:
    """
    Train a Gumbel router model with the given compute penalty.
    Uses the pre-loaded base_model to save IO time.
    """
    print(f"\n{'='*60}")
    print(f"  Penalty lambda={penalty:.3f}  |  Higher lambda -> more layer skipping")
    print(f"{'='*60}")

    # Re-initialize LoRA and Router on the shared base_model
    # This avoids loading 2.2GB from disk for every penalty run.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    # Removed gradient_checkpointing_enable() because it conflicts with PyTorch register_forward_hook
    # We will rely on 4-bit quantization and BS=4 to prevent VRAM spilling instead.
    model = get_peft_model(base_model, lora_cfg)

    TOTAL_LAYERS    = len(model.base_model.model.model.layers)
    ROUTABLE_LAYERS = TOTAL_LAYERS - ALWAYS_KEEP

    model.router = GumbelRouter(model.config.hidden_size, ROUTABLE_LAYERS).to("cuda")
    for p in model.router.parameters():
        p.requires_grad = True

    # Compile the student for 15-20% speedup
    if ATTN_IMPL == "sdpa" and os.name != "nt":
        try:
            print("  Compiling student model...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  [WARNING] Student compilation failed: {e}")
    elif os.name == "nt":
        print("  [INFO] Skipping student compilation (Windows/Triton limitation).")
        
    # Use filter to avoid passing the router parameters twice (which causes UserWarning)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader) // GRAD_ACCUM
    )

    current_temp = GUMBEL_TEMP_START
    global_step  = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        epoch_bar = tqdm(
            train_loader,
            desc=f"Penalty {penalty:.2f} | Epoch {epoch+1}/{EPOCHS}",
            ncols=100,
        )
        for step, batch in enumerate(epoch_bar):
            batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

            student_logits, ce_loss, gates = gated_forward(
                model, batch, temperature=current_temp, hard=True
            )

            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                ).logits

            gate_loss = gates.float().mean() * penalty

            # Gate entropy bonus — same as exp6 fix: prevents router from collapsing
            p_mean = gates.detach().float().mean()
            eps = 1e-6
            gate_entropy = -(p_mean * (p_mean + eps).log() + (1 - p_mean) * (1 - p_mean + eps).log())
            entropy_bonus = GATE_ENTROPY_BETA * gate_entropy

            if global_step < KD_WARMUP_STEPS:
                kd_loss    = torch.tensor(0.0, device="cuda")
                total_loss = (ce_loss + gate_loss - entropy_bonus) / GRAD_ACCUM
            else:
                kd_loss    = compute_kd_loss(
                    student_logits[:, :-1, :], teacher_logits[:, :-1, :], KD_TEMPERATURE
                )
                total_loss = (KD_ALPHA * ce_loss + (1.0 - KD_ALPHA) * kd_loss + gate_loss - entropy_bonus) / GRAD_ACCUM

            total_loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                all_params = list(model.parameters()) + list(model.router.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_layers = gates.detach().float().mean(dim=0).sum().item() + ALWAYS_KEEP
            epoch_bar.set_postfix({
                "ce":     f"{ce_loss.item():.3f}",
                "layers": f"{avg_layers:.1f}",
                "temp":   f"{current_temp:.3f}",
            })

        current_temp = max(GUMBEL_TEMP_MIN, current_temp * TEMP_ANNEAL_RATE)

    # Final evaluation
    model.eval()
    total_val_loss = 0.0
    layer_counts   = []
    with torch.no_grad():
        for i, val_batch in enumerate(eval_loader):
            if i >= MAX_EVAL_BATCHES:
                break
            val_batch = {k: v.to("cuda") for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
            # Use hard=True for binary gates so layer count is exact (matches training)
            _, v_ce, v_gates = gated_forward(model, val_batch, temperature=current_temp, hard=True)
            total_val_loss += v_ce.item()
            layer_counts.append(v_gates.float().mean(dim=0).sum().item() + ALWAYS_KEEP)

    n_eval     = min(MAX_EVAL_BATCHES, len(eval_loader))
    val_loss   = total_val_loss / n_eval
    avg_layers = sum(layer_counts) / len(layer_counts)
    skip_ratio = 1.0 - (avg_layers / TOTAL_LAYERS)

    print(f"\n  --> Penalty: {penalty:.2f} | Val Loss: {val_loss:.4f} | "
          f"Avg Layers: {avg_layers:.1f}/{TOTAL_LAYERS} | Skip: {skip_ratio:.1%}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "penalty":          penalty,
        "val_loss":         round(val_loss, 4),
        "avg_active_layers": round(avg_layers, 2),
        "total_layers":     TOTAL_LAYERS,
        "skip_ratio":       round(skip_ratio, 4),
    }

# ---------------------------------------------------------------------------
# Pareto plot
# ---------------------------------------------------------------------------
def plot_pareto(sweep_results, baseline_val_loss, total_layers):
    """Generate the Pareto frontier plot with clean white-background styling."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Baseline reference line
    ax.axhline(y=baseline_val_loss, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Baseline LoRA (all {total_layers} layers, val_loss={baseline_val_loss:.2f})", alpha=0.8)

    # Sweep points
    xs = [r["skip_ratio"] * 100 for r in sweep_results]
    ys = [r["val_loss"]         for r in sweep_results]

    ax.plot(xs, ys, "-o", color="#4a90e2", linewidth=2.5,
            markersize=8, markerfacecolor="#e74c3c", markeredgecolor="white", markeredgewidth=1.2,
            zorder=5, label="Gumbel Router (this work)")

    for r in sweep_results:
        ax.annotate(
            f"lambda={r['penalty']:.3f}\n{r['avg_active_layers']:.1f}L",
            xy=(r["skip_ratio"] * 100, r["val_loss"]),
            xytext=(0, -18), textcoords="offset points",
            fontsize=8.5, color="#333333", ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none')
        )

    ax.set_xlabel("Layer Skip Ratio (%)", color="#333333", fontsize=12)
    ax.set_ylabel("Validation Loss (CE)",  color="#333333", fontsize=12)
    ax.set_title("Gumbel Router Pareto Frontier\n(Wikitext-103 Validation Loss vs. Compute Savings)",
                 color="black", fontsize=13, fontweight="bold", pad=14)

    ax.tick_params(colors="#333333")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(facecolor="white", edgecolor="#dddddd", labelcolor="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Pareto plot saved → {PLOT_FILE}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("  Phase 4 Setup: Gumbel Router Pareto Sweep")
    print(f"  Penalties: {PENALTIES}")
    print(f"  Train samples: {TRAIN_SAMPLES} | Epochs: {EPOCHS}")
    print("="*70)

    # Shared models (loaded once, reused across all penalty runs)
    print("\nLoading models into VRAM (shared across sweep)...")
    
    # Base model for students (Load in 4-bit to allow 2x batch size)
    from transformers import BitsAndBytesConfig
    quant_config_student = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config_student, device_map="cuda",
        attn_implementation=ATTN_IMPL
    )
    
    # Shared frozen teacher
    print("\nLoading frozen Teacher model for KD (4-bit)...")
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=quant_config,
        device_map="cuda",
        attn_implementation=ATTN_IMPL,
    )
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    # Compile teacher for faster KD steps
    if ATTN_IMPL == "sdpa" and os.name != "nt":
        try:
            print("Compiling teacher model...")
            teacher_model = torch.compile(teacher_model)
        except Exception as e:
            print(f"[WARNING] Teacher compilation failed: {e}")
    elif os.name == "nt":
        print("[INFO] Skipping teacher compilation (Windows/Triton limitation).")

    # CSV init
    with open(CSV_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow([
            "Penalty", "Val Loss", "Avg Active Layers", "Total Layers", "Skip Ratio"
        ])

    sweep_results = []
    for penalty in PENALTIES:
        result = train_one_penalty(penalty, teacher_model, base_model)
        sweep_results.append(result)

        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([
                result["penalty"],
                result["val_loss"],
                result["avg_active_layers"],
                result["total_layers"],
                result["skip_ratio"],
            ])

    # Get baseline loss from exp1 metrics if available (else use teacher)
    from pathlib import Path
    baseline_csvs = sorted(Path(".").glob("exp1_baseline_metrics_*.csv"), reverse=True)
    if baseline_csvs:
        import csv as csvmod
        rows = list(csvmod.DictReader(open(baseline_csvs[0])))
        baseline_val_loss = float(rows[-1].get("Validation Loss", 10.0))
    else:
        baseline_val_loss = 10.0  # fallback
        print("[WARNING] No exp1 baseline metrics found; using 10.0 as baseline val_loss reference.")

    total_layers = sweep_results[0]["total_layers"] if sweep_results else 22

    # Plot
    plot_pareto(sweep_results, baseline_val_loss, total_layers)

    # Final summary
    print(f"\n{'='*70}")
    print("  PARETO SWEEP RESULTS")
    print(f"  {'Penalty':>8} {'Val Loss':>10} {'Avg Layers':>12} {'Skip %':>8}")
    print("  " + "-"*44)
    for r in sweep_results:
        print(f"  {r['penalty']:>8.2f} {r['val_loss']:>10.4f} {r['avg_active_layers']:>12.1f} {r['skip_ratio']*100:>7.1f}%")

    print(f"\n  CSV  -> {CSV_FILENAME}")
    print(f"  Plot -> {PLOT_FILE}")
    print("\nPhase 4 Pareto sweep complete!")


if __name__ == "__main__":
    main()

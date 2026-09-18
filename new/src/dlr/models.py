"""Model loading: base + LoRA. SmolLM2-135M defaults for 8GB 4060."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LORA_TARGETS_SMALL = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_LORA_TARGETS_EXPANDED = DEFAULT_LORA_TARGETS_SMALL + ["gate_proj", "up_proj", "down_proj"]


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_id: str, dtype=torch.float32, device: str = "cpu"):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    return model


def wrap_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05, expanded: bool = False):
    targets = DEFAULT_LORA_TARGETS_EXPANDED if expanded else DEFAULT_LORA_TARGETS_SMALL
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets, task_type="CAUSAL_LM", bias="none")
    return get_peft_model(model, cfg)


def base_total_layers(model) -> int:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.config.num_hidden_layers


def always_keep_layers(total_layers: int, frac: float = 0.15) -> int:
    return max(2, round(frac * total_layers))

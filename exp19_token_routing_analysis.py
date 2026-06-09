"""
exp19_token_routing_analysis.py

Token-category analysis for Dynamic Layer Routing.

This script checks whether a trained token-level router allocates compute
differently across token types instead of merely learning a fixed per-layer
skip pattern. It is intended as an analysis utility for the manuscript, not as
a training script.

Default target: TinyLlama exp10 token router.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import csv
import string
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with",
}


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

    def forward(self, h: torch.Tensor, temperature: float = 0.5, hard: bool = True):
        h = h.float()
        logits = self.net(h)
        logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        gates = F.gumbel_softmax(logits_2c, tau=temperature, hard=hard, dim=-1)
        return gates[..., 1].to(h.dtype)


class StopForward(Exception):
    pass


def get_layers(model):
    return model.base_model.model.model.layers


def capture_anchor_hidden(model, input_ids, attention_mask, always_keep):
    layers = get_layers(model)
    captured = {}

    def hook(_module, _input, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach().float()
        raise StopForward()

    handle = layers[always_keep - 1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured["h"]


def install_gate_hooks(model, gates, always_keep):
    handles = []
    layers = get_layers(model)
    for i, layer in enumerate(layers[always_keep:]):
        def hook(_module, inputs, output, layer_i=i):
            residual = inputs[0]
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            gate = gates[:, :, layer_i].unsqueeze(-1).to(h.dtype)
            gated_h = gate * h + (1.0 - gate) * residual
            return (gated_h,) + output[1:] if is_tuple else gated_h

        handles.append(layer.register_forward_hook(hook))
    return handles


def classify_token(token_text: str, token_id: int, tokenizer) -> set[str]:
    cleaned = token_text.replace("▁", "").replace("Ġ", "").strip()
    cats = {"all"}
    if not cleaned:
        cats.add("whitespace_or_empty")
    if cleaned and all(ch in string.punctuation for ch in cleaned):
        cats.add("punctuation")
    if cleaned.lower() in STOP_WORDS:
        cats.add("stop_word")
    if any(ch.isdigit() for ch in cleaned):
        cats.add("numeric")
    if cleaned and not token_text.startswith(("▁", "Ġ")):
        cats.add("subword_or_continuation")
    if token_id >= int(0.9 * tokenizer.vocab_size):
        cats.add("high_id_token")
    return cats


def main():
    parser = argparse.ArgumentParser(description="Analyze token-level DLR routing by token category.")
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", default="exp10_token_output_20260525_163815/best_model")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--always_keep", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--out_csv", default="exp19_token_routing_analysis.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    tokenizer.pad_token = tokenizer.eos_token

    total_layers = len(get_layers(model))
    routable = total_layers - args.always_keep
    model.router = GumbelRouter(model.config.hidden_size, routable).to(device)
    router_path = ckpt / "router_weights.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"Missing router weights: {router_path}")
    model.router.load_state_dict(torch.load(str(router_path), map_location=device))
    model.eval()

    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    raw = raw.filter(lambda x: len(x["text"]) > 100).select(range(args.samples))

    stats = defaultdict(lambda: {"tokens": 0, "active_sum": 0.0, "loss_sum": 0.0, "loss_tokens": 0, "high_loss_tokens": 0})

    for start in range(0, len(raw), args.batch_size):
        texts = raw[start : start + args.batch_size]["text"]
        enc = tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        h = capture_anchor_hidden(model, input_ids, attention_mask, args.always_keep)
        gates = model.router(h, temperature=args.temperature, hard=True)
        active_by_token = gates.float().sum(dim=-1) + args.always_keep

        handles = install_gate_hooks(model, gates, args.always_keep)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for handle in handles:
            handle.remove()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid_losses = token_loss[shift_labels != -100]
        high_loss_threshold = torch.quantile(valid_losses.float(), 0.75).item() if valid_losses.numel() else float("inf")

        for b in range(input_ids.size(0)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            for pos, token_id in enumerate(input_ids[b].tolist()):
                if attention_mask[b, pos].item() == 0:
                    continue
                cats = classify_token(tokens[pos], token_id, tokenizer)
                loss_val = None
                if pos > 0 and labels[b, pos].item() != -100:
                    loss_val = token_loss[b, pos - 1].item()
                    if loss_val >= high_loss_threshold:
                        cats.add("high_loss")

                active = active_by_token[b, pos].item()
                for cat in cats:
                    stats[cat]["tokens"] += 1
                    stats[cat]["active_sum"] += active
                    if loss_val is not None:
                        stats[cat]["loss_sum"] += loss_val
                        stats[cat]["loss_tokens"] += 1
                        if loss_val >= high_loss_threshold:
                            stats[cat]["high_loss_tokens"] += 1

    rows = []
    for cat, s in sorted(stats.items()):
        n = s["tokens"]
        loss_n = s["loss_tokens"]
        rows.append({
            "category": cat,
            "tokens": n,
            "avg_active_layers": s["active_sum"] / max(n, 1),
            "avg_token_loss": s["loss_sum"] / max(loss_n, 1),
            "high_loss_fraction": s["high_loss_tokens"] / max(loss_n, 1),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote token routing analysis to {args.out_csv}")


if __name__ == "__main__":
    main()

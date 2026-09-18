"""Canonical routing metrics — SINGLE source used by train_loop AND evaluate.py.

gates: [B, S, R] post-STE hard gates in {0,1} for the R routable layers.
always_keep: number of leading layers never routed (for effective-depth keys).

Two entropy views (kept side by side so nothing silently changes meaning):
- router_entropy: mean over layers of binary entropy of per-layer keep rate.
  Sensitive to layer-wise saturation; the collapse rule uses this.
- router_entropy_global: binary entropy of the single global keep rate.
Collapse rule (applies to input-conditional routers only — NOT to fixed
stochastic masks, whose entropy is ~0 by construction): mean_gate outside
[0.02, 0.98] or router_entropy < 0.10.
"""
import math

import torch


def binary_entropy(p: torch.Tensor, eps: float = 1e-6):
    # float64: in fp32, 1-(1-1e-8) rounds to 0 -> 0*log(0)=nan for p exactly
    # 0/1 (the normal case for stochastic-depth gates).
    p = p.to(torch.float64).clamp(eps, 1 - eps)
    return (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).to(torch.float32)


def _scalar_binary_entropy(mean_gate: float, eps: float = 1e-8) -> float:
    m = min(max(mean_gate, eps), 1 - eps)
    return -(m * math.log(m) + (1 - m) * math.log(1 - m))


def routing_metrics_from_gates(gates: torch.Tensor, always_keep: int = 0) -> dict:
    per_layer = gates.mean(dim=(0, 1))  # [R] keep rates
    mean_gate = gates.mean().item()
    active = gates.sum(dim=-1)  # [B, S] routable layers kept per token
    eff = active + always_keep  # effective depth per token
    per_layer_ent = binary_entropy(per_layer).mean().item()
    return {
        "mean_active_layers": eff.mean().item(),
        "std_active_layers": eff.std().item() if eff.numel() > 1 else 0.0,
        "min_active_layers": eff.min().item(),
        "max_active_layers": eff.max().item(),
        "min_gate": per_layer.min().item(),
        "max_gate": per_layer.max().item(),
        "mean_gate": mean_gate,
        "achieved_skip": 1.0 - mean_gate,
        "router_entropy": per_layer_ent,
        "router_entropy_global": _scalar_binary_entropy(mean_gate),
        "structural_flop_reduction_pct": 100.0 * (1.0 - eff.mean().item() / max(1, always_keep + gates.shape[-1])),
        "collapse_flag": int(mean_gate < 0.02 or mean_gate > 0.98 or per_layer_ent < 0.10),
        "per_layer_activity": per_layer.detach().cpu().tolist(),
    }

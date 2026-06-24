"""
utils/flop_accounting.py — Defensible FLOP Accounting for DLR Paper
=====================================================================

Methodology
-----------
All FLOPs are counted using the 2·M multiply-add convention (Hoffmann et al.,
"Training Compute-Optimal Large Language Models", NeurIPS 2022, i.e. Chinchilla).
A matrix multiply [m, k] × [k, n] costs 2·m·k·n FLOPs.

LayerNorm, activation functions (SiLU/GELU), and residual adds are excluded;
they contribute <2% of total FLOPs and are excluded by convention.

Primary Metric — Token-Layer Utilization
-----------------------------------------
For token-level routing, the strongest publishable metric is:

    utilization_pct = 100 × executed_token_layer_pairs / total_token_layer_pairs

where:
    executed_token_layer_pairs = ∑_{b,s,l} gate[b, s, l]
    total_token_layer_pairs    = B × S × L_routable

This metric is EXACT (not projected) and directly measures input-conditional
compute allocation. It is the primary metric for the DLR paper.

Projected FLOPs
---------------
FLOPs proportional to token-layer pairs under the assumption of perfect
token-packing (i.e., tokens with gate=0 incur zero compute for that layer).

IMPORTANT: The current PyTorch hook implementation does NOT achieve real FLOP
reduction — hooks blend outputs after full computation of all layers. All
"projected FLOP" numbers assume an idealized implementation. Label as
"projected" in all paper tables and figures.

Reference: Hoffmann et al., NeurIPS 2022. Section 2, Equation 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch


# =============================================================================
# Per-layer FLOP profile
# =============================================================================

@dataclass
class LayerFlopProfile:
    """
    Per-layer FLOP breakdown for one transformer decoder layer.
    All values are FLOPs for a full sequence of S tokens (not per-token).
    """
    # Attention projections: 2 * S * in_dim * out_dim
    q_proj: float        # Query: [S, H] × [H, H]
    k_proj: float        # Key:   [S, H] × [H, n_kv * hd]  (GQA)
    v_proj: float        # Value: [S, H] × [H, n_kv * hd]
    o_proj: float        # Out:   [S, H] × [H, H]

    # Attention score computation (full O(S²·H) cost; causal mask halves average
    # but upper bound is reported for conservatism, matching Chinchilla convention)
    attn_qk: float       # QK^T:  [n_h heads] × [S, hd] × [hd, S]  = 2·S²·H
    attn_av: float       # AV:    [n_h heads] × [S, S]  × [S, hd]  = 2·S²·H

    # MLP (SwiGLU: gate, up, down projections)
    mlp_gate: float      # [S, H] × [H, F]
    mlp_up:   float      # [S, H] × [H, F]
    mlp_down: float      # [S, F] × [F, H]

    @property
    def total(self) -> float:
        return (self.q_proj + self.k_proj + self.v_proj + self.o_proj
                + self.attn_qk + self.attn_av
                + self.mlp_gate + self.mlp_up + self.mlp_down)

    @property
    def attn_proj_flops(self) -> float:
        return self.q_proj + self.k_proj + self.v_proj + self.o_proj

    @property
    def attn_compute_flops(self) -> float:
        return self.attn_qk + self.attn_av

    @property
    def mlp_flops(self) -> float:
        return self.mlp_gate + self.mlp_up + self.mlp_down

    def to_dict(self) -> Dict[str, float]:
        return {
            "q_proj":     self.q_proj,
            "k_proj":     self.k_proj,
            "v_proj":     self.v_proj,
            "o_proj":     self.o_proj,
            "attn_qk":    self.attn_qk,
            "attn_av":    self.attn_av,
            "mlp_gate":   self.mlp_gate,
            "mlp_up":     self.mlp_up,
            "mlp_down":   self.mlp_down,
            "total":      self.total,
            "attn_proj":  self.attn_proj_flops,
            "attn_score": self.attn_compute_flops,
            "mlp_total":  self.mlp_flops,
        }


def profile_layer_flops(config, seq_len: int) -> LayerFlopProfile:
    """
    Compute the FLOP profile for one decoder layer processing S tokens.

    Handles Grouped-Query Attention (GQA) correctly via num_key_value_heads.

    Args:
        config: HuggingFace model config. Must have:
                hidden_size, num_attention_heads, num_key_value_heads,
                intermediate_size.
        seq_len: Sequence length S (number of tokens).

    Returns:
        LayerFlopProfile with FLOPs per LAYER (for S tokens total, not per token).

    Qwen2.5-7B reference values:
        hidden_size=3584, num_attention_heads=28, num_key_value_heads=4,
        intermediate_size=18944
        → total per layer ≈ 7.4 GFLOPs at seq_len=512
    """
    H    = config.hidden_size
    n_h  = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_h)
    hd   = H // n_h                    # head dimension
    F    = config.intermediate_size    # MLP hidden width
    S    = seq_len

    # ── Attention projections ─────────────────────────────────────────────────
    # Q: [S, H] × [H, n_h * hd]  = [S, H] × [H, H]  (since n_h * hd = H)
    q_flops = 2 * S * H * H
    # K, V: [S, H] × [H, n_kv * hd]  (GQA: fewer KV heads)
    k_flops = 2 * S * H * (n_kv * hd)
    v_flops = 2 * S * H * (n_kv * hd)
    # O: [S, H] × [H, H]
    o_flops = 2 * S * H * H

    # ── Attention scores ──────────────────────────────────────────────────────
    # QK^T per head: [S, hd] × [hd, S] = 2·S²·hd; summed over n_h heads = 2·S²·H
    attn_qk = 2 * S * S * H
    # AV per head: [S, S] × [S, hd] = 2·S²·hd; summed = 2·S²·H
    attn_av = 2 * S * S * H

    # ── MLP (SwiGLU) ─────────────────────────────────────────────────────────
    gate_flops = 2 * S * H * F     # gate projection
    up_flops   = 2 * S * H * F     # up projection (element-wise with gate after)
    down_flops = 2 * S * F * H     # down projection

    return LayerFlopProfile(
        q_proj=q_flops, k_proj=k_flops, v_proj=v_flops, o_proj=o_flops,
        attn_qk=attn_qk, attn_av=attn_av,
        mlp_gate=gate_flops, mlp_up=up_flops, mlp_down=down_flops,
    )


def estimate_layer_flops_per_token(config, seq_len: int) -> float:
    """
    Return total FLOPs per TOKEN per layer.

    Backward-compatible convenience wrapper; divides LayerFlopProfile.total
    by seq_len so callers can multiply by token count.
    """
    return profile_layer_flops(config, seq_len).total / seq_len


# =============================================================================
# Accounting result dataclass
# =============================================================================

@dataclass
class FlopAccountingResult:
    """
    Complete FLOP and utilization accounting for one evaluation pass.

    Primary metric for the paper: utilization_pct
    Secondary metric: projected_flop_reduction_pct
    """
    # ── Model dimensions ──────────────────────────────────────────────────────
    num_layers:   int
    always_keep:  int
    seq_len:      int
    batch_size:   int

    # ── Baseline (dense full-depth) ───────────────────────────────────────────
    baseline_flops:           float   # B × L × flops_per_layer
    baseline_flops_per_token: float   # baseline_flops / (B × S)

    # ── Token-layer utilization (EXACT — primary metric) ─────────────────────
    executed_token_layer_pairs: int   # ∑ gate[b,s,l]  (integer count)
    total_token_layer_pairs:    int   # B × S × L_routable
    utilization_pct:            float # 100 × exec / total

    # ── Projected FLOPs (theoretical; assumes perfect token packing) ──────────
    # NOTE: Current hook implementation achieves 0% real FLOP reduction.
    # These numbers are proportional to token-layer utilization.
    projected_executed_flops:      float
    projected_saved_flops:         float
    projected_flop_reduction_pct:  float

    # ── Derived ───────────────────────────────────────────────────────────────
    avg_active_layers: float    # always_keep + mean active routable layers/token

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Primary metric
            "executed_token_layer_pairs":    self.executed_token_layer_pairs,
            "total_token_layer_pairs":       self.total_token_layer_pairs,
            "utilization_pct":               round(self.utilization_pct, 4),
            "avg_active_layers":             round(self.avg_active_layers, 4),
            # FLOP accounting (projected)
            "baseline_flops":                self.baseline_flops,
            "projected_executed_flops":      self.projected_executed_flops,
            "projected_saved_flops":         self.projected_saved_flops,
            "projected_flop_reduction_pct":  round(self.projected_flop_reduction_pct, 4),
            "baseline_flops_per_token":      round(self.baseline_flops_per_token, 4),
        }

    def summary_line(self) -> str:
        return (
            f"Utilization={self.utilization_pct:.1f}%  "
            f"AvgLayers={self.avg_active_layers:.1f}/{self.num_layers}  "
            f"ExecPairs={self.executed_token_layer_pairs:,}/{self.total_token_layer_pairs:,}  "
            f"ProjectedFLOP↓={self.projected_flop_reduction_pct:.1f}%"
        )


# =============================================================================
# Token-level routing accounting
# =============================================================================

def compute_token_routing_flops(
    config,
    gates:       torch.Tensor,   # [B, S, L_routable] binary float gates
    always_keep: int,
    seq_len:     int,
    batch_size:  int,
) -> FlopAccountingResult:
    """
    Compute projected FLOP accounting for token-level routing.

    Primary metric is utilization_pct (exact). FLOPs are projected.

    Args:
        config: Model config.
        gates: [B, S, L] float tensor of binary gates (0.0 or 1.0).
               gates[b, s, l] = 1 → token (b,s) executed routable layer l.
        always_keep: Number of always-executed (non-routable) prefix layers.
        seq_len: Sequence length.
        batch_size: Batch size (should equal gates.shape[0]).

    Returns:
        FlopAccountingResult with exact utilization and projected FLOP numbers.
    """
    assert gates.dim() == 3, f"Expected [B, S, L] gates, got {gates.shape}"
    B, S, L = gates.shape
    total_layers      = always_keep + L
    layer_profile     = profile_layer_flops(config, seq_len)
    flops_per_layer   = layer_profile.total   # for full sequence S

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_flops        = total_layers * flops_per_layer * B
    baseline_per_token    = baseline_flops / max(1, B * S)

    # ── Exact token-layer utilization ─────────────────────────────────────────
    gate_f                = gates.float().detach()
    exec_pairs            = int(gate_f.sum().long().item())
    total_pairs           = B * S * L
    utilization_pct       = 100.0 * exec_pairs / max(1, total_pairs)

    # ── Projected FLOPs ───────────────────────────────────────────────────────
    # Always-kept layers: full compute regardless of routing
    always_flops          = always_keep * flops_per_layer * B
    # Routable layers: projected compute ∝ executed token-layer pairs
    # Idea: each (token, layer) pair costs flops_per_layer / S (per token)
    flops_per_token_layer = flops_per_layer / max(1, S)
    routable_exec_flops   = exec_pairs * flops_per_token_layer
    projected_exec        = always_flops + routable_exec_flops
    projected_saved       = baseline_flops - projected_exec
    reduction_pct         = 100.0 * projected_saved / max(1.0, baseline_flops)

    # ── Average active layers per token ──────────────────────────────────────
    avg_active_routable   = gate_f.mean(dim=(0, 1)).sum().item()   # sum of per-layer rates
    avg_active_layers     = always_keep + avg_active_routable

    return FlopAccountingResult(
        num_layers=total_layers,
        always_keep=always_keep,
        seq_len=S,
        batch_size=B,
        baseline_flops=baseline_flops,
        baseline_flops_per_token=baseline_per_token,
        executed_token_layer_pairs=exec_pairs,
        total_token_layer_pairs=total_pairs,
        utilization_pct=utilization_pct,
        projected_executed_flops=projected_exec,
        projected_saved_flops=projected_saved,
        projected_flop_reduction_pct=reduction_pct,
        avg_active_layers=avg_active_layers,
    )


# =============================================================================
# Stochastic depth accounting
# =============================================================================

def compute_stochastic_depth_flops(
    config,
    layer_stats:      dict,    # {layer_idx: {"executed": int, "skipped": int}}
    seq_len:          int,
    batch_size:       int,
    protected_layers: int,
) -> FlopAccountingResult:
    """
    Compute projected FLOP accounting for stochastic depth.

    Stochastic depth skips ENTIRE layers (all tokens) per forward pass.
    Token-layer pairs are B × S × executed_layer_calls.

    Args:
        config: Model config.
        layer_stats: Per-layer execution counters from stochastic depth wrapper.
        seq_len: Sequence length.
        batch_size: Batch size.
        protected_layers: Number of always-executed prefix layers.
    """
    total_layers      = len(layer_stats)
    layer_profile     = profile_layer_flops(config, seq_len)
    flops_per_layer   = layer_profile.total

    baseline_flops    = total_layers * flops_per_layer * batch_size
    baseline_per_tok  = baseline_flops / max(1, batch_size * seq_len)

    # Stochastic depth: layer is either fully executed or fully skipped
    total_exec_calls  = sum(s["executed"] for s in layer_stats.values())
    total_poss_calls  = sum(s["executed"] + s["skipped"] for s in layer_stats.values())

    # Token-layer pairs: each executed call processes batch_size × seq_len tokens
    exec_pairs        = total_exec_calls * batch_size * seq_len
    total_pairs       = total_poss_calls * batch_size * seq_len
    utilization_pct   = 100.0 * exec_pairs / max(1, total_pairs)

    projected_exec    = (total_exec_calls / max(1, total_poss_calls)) * baseline_flops
    projected_saved   = baseline_flops - projected_exec
    reduction_pct     = 100.0 * projected_saved / max(1.0, baseline_flops)

    # Average active layers = executed_calls_per_forward × total_layers / total_poss_calls
    # Simpler: total_executed / (total_possible / total_layers)
    avg_active        = (total_exec_calls * total_layers) / max(1, total_poss_calls)

    return FlopAccountingResult(
        num_layers=total_layers,
        always_keep=protected_layers,
        seq_len=seq_len,
        batch_size=batch_size,
        baseline_flops=baseline_flops,
        baseline_flops_per_token=baseline_per_tok,
        executed_token_layer_pairs=exec_pairs,
        total_token_layer_pairs=total_pairs,
        utilization_pct=utilization_pct,
        projected_executed_flops=projected_exec,
        projected_saved_flops=projected_saved,
        projected_flop_reduction_pct=reduction_pct,
        avg_active_layers=avg_active,
    )

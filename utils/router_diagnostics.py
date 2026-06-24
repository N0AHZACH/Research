"""
utils/router_diagnostics.py — Router Collapse Detection & Diagnostic Metrics
=============================================================================

Detects pathological routing behaviors that would invalidate DLR's claims:

  ALL_ZERO collapse  : router skips >98% of token-layer pairs
                       → trivial solution; no useful routing
  ALL_ONE collapse   : router skips <2% (reverts to dense baseline)
                       → no compute savings
  DEAD_LAYER         : individual layer with <1% mean activation
                       → layer permanently bypassed; not input-conditional
  LOW_ENTROPY        : routing entropy < threshold
                       → near-deterministic; not input-conditional

These diagnostics must be logged continuously. If any collapse is detected,
the experiment should be flagged and potentially restarted.

Key publication metric — Routing Entropy
-----------------------------------------
Shannon binary entropy of the router's gate distribution:

    H(p) = -p·log(p) - (1-p)·log(1-p)     [nats]

where p = mean gate activation rate.

Maximum entropy (0.693 nats) = 50% utilization = maximally input-conditional.
Low entropy = router has collapsed to a deterministic policy (good OR bad).

For the DLR paper: routing entropy must be >0 throughout training and should
correlate with input difficulty (harder inputs → higher entropy → more compute).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch


# =============================================================================
# Thresholds
# =============================================================================
COLLAPSE_ZERO_THRESHOLD   = 0.02    # mean gate < 2%  → ALL_ZERO
COLLAPSE_ONE_THRESHOLD    = 0.98    # mean gate > 98% → ALL_ONE
DEAD_LAYER_THRESHOLD      = 0.01    # per-layer mean  < 1% → dead
LOW_ENTROPY_THRESHOLD     = 0.10    # H(p) nats < 0.1 → near-deterministic
HIGH_UTILIZATION_VAR      = 0.10    # Var(per_layer_util) > 0.1 → unbalanced


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class RouterDiagnosticsResult:
    """
    Diagnostic snapshot for one batch (or accumulated eval set) of gates.

    Attributes:
        mean_gate            : overall fraction of (token, layer) pairs that execute
        gate_variance        : variance of gate values across all pairs
        routing_entropy      : H(mean_gate) in nats [0, ln2≈0.693]
        per_layer_mean       : mean gate per routable layer [L]
        per_layer_entropy    : H(per_layer_mean[l]) per layer [L]
        utilization_variance : variance of per-layer utilization rates
        is_all_zero          : ALL_ZERO collapse flag
        is_all_one           : ALL_ONE collapse flag
        dead_layer_count     : number of dead layers (mean < threshold)
        dead_layer_indices   : absolute layer indices (offset by always_keep)
        is_low_entropy       : routing is near-deterministic
        is_high_variance     : highly unbalanced per-layer utilization
        avg_active_layers    : always_keep + sum(per_layer_mean)
    """
    # Gate statistics
    mean_gate:            float
    gate_variance:        float
    routing_entropy:      float
    per_layer_mean:       List[float]
    per_layer_entropy:    List[float]
    utilization_variance: float

    # Collapse flags
    is_all_zero:         bool
    is_all_one:          bool
    dead_layer_count:    int
    dead_layer_indices:  List[int]
    is_low_entropy:      bool
    is_high_variance:    bool

    # Derived
    avg_active_layers:   float
    always_keep:         int = 0

    @property
    def any_collapse(self) -> bool:
        return (self.is_all_zero or self.is_all_one
                or self.dead_layer_count > 0 or self.is_low_entropy)

    def warning_messages(self, step: int) -> List[str]:
        msgs = []
        if self.is_all_zero:
            msgs.append(
                f"[COLLAPSE:ALL_ZERO] Step {step}: mean_gate={self.mean_gate:.4f} "
                f"< {COLLAPSE_ZERO_THRESHOLD}. Router skipping >98% of token-layer pairs. "
                f"Action: increase COMPUTE_PENALTY target or decrease TARGET_SKIP."
            )
        if self.is_all_one:
            msgs.append(
                f"[COLLAPSE:ALL_ONE] Step {step}: mean_gate={self.mean_gate:.4f} "
                f"> {COLLAPSE_ONE_THRESHOLD}. Router nearly dense. "
                f"Action: increase COMPUTE_PENALTY coefficient."
            )
        if self.dead_layer_count > 0:
            msgs.append(
                f"[COLLAPSE:DEAD_LAYER] Step {step}: {self.dead_layer_count} dead layer(s) "
                f"at absolute indices {self.dead_layer_indices} "
                f"(gate_mean < {DEAD_LAYER_THRESHOLD}). "
                f"Router permanently bypasses these layers — not input-conditional."
            )
        if self.is_low_entropy:
            msgs.append(
                f"[WARNING:LOW_ENTROPY] Step {step}: routing_entropy={self.routing_entropy:.4f} nats "
                f"< {LOW_ENTROPY_THRESHOLD}. Router decisions are near-deterministic. "
                f"May have collapsed to a static routing pattern."
            )
        return msgs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_gate":              self.mean_gate,
            "gate_variance":          self.gate_variance,
            "routing_entropy":        self.routing_entropy,
            "utilization_variance":   self.utilization_variance,
            "avg_active_layers":      self.avg_active_layers,
            "is_all_zero":            int(self.is_all_zero),
            "is_all_one":             int(self.is_all_one),
            "dead_layer_count":       self.dead_layer_count,
            "is_low_entropy":         int(self.is_low_entropy),
            "is_high_variance":       int(self.is_high_variance),
        }

    def summary_line(self, step: int = -1) -> str:
        step_str = f"Step {step} | " if step >= 0 else ""
        return (
            f"{step_str}"
            f"H={self.routing_entropy:.3f}nats  "
            f"p={self.mean_gate:.3f}  "
            f"Var={self.gate_variance:.4f}  "
            f"AvgLayers={self.avg_active_layers:.1f}  "
            f"{'[COLLAPSE!]' if self.any_collapse else '[OK]'}"
        )


# =============================================================================
# Core functions
# =============================================================================

def compute_routing_entropy(p: float) -> float:
    """
    Binary Shannon entropy H(p) in nats.

    H(p) = -p·ln(p) - (1-p)·ln(1-p)

    Range: [0, ln(2)≈0.693]
    Maximum at p=0.5 (maximally uncertain / input-conditional).
    Zero at p=0 or p=1 (fully deterministic — skip everything or skip nothing).
    """
    eps = 1e-8
    p = float(p)
    p = max(eps, min(1.0 - eps, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def diagnose_gates(
    gates:         torch.Tensor,   # [B, S, L] binary float gates
    always_keep:   int = 0,
    step:          int = 0,
    print_warnings: bool = True,
    file=None,
) -> RouterDiagnosticsResult:
    """
    Diagnose gate tensor for collapse and compute all diagnostic metrics.

    Args:
        gates: [B, S, L] float tensor. Values should be 0.0 or 1.0 (hard gates)
               or probabilities (soft gates for diagnostic purposes).
        always_keep: Number of always-active layers NOT represented in gates.
                     Used to compute absolute layer indices for dead layers.
        step: Current training step (used in warning messages).
        print_warnings: If True, print any collapse warnings to stderr.
        file: File object to print to (default: sys.stderr).

    Returns:
        RouterDiagnosticsResult with all computed metrics.
    """
    assert gates.dim() == 3, (
        f"diagnose_gates expects [B, S, L] tensor, got shape {tuple(gates.shape)}. "
        f"If using sequence-level gates [B, L], unsqueeze(1) first."
    )
    _out = file or sys.stderr

    g = gates.float().detach()
    B, S, L = g.shape

    # ── Global statistics ─────────────────────────────────────────────────────
    mean_gate       = g.mean().item()
    gate_variance   = g.var().item()
    routing_entropy = compute_routing_entropy(mean_gate)

    # ── Per-layer statistics ──────────────────────────────────────────────────
    per_layer_mean_t = g.mean(dim=(0, 1))               # [L]
    per_layer_mean   = per_layer_mean_t.tolist()
    per_layer_ent    = [compute_routing_entropy(m) for m in per_layer_mean]
    utilization_var  = float(per_layer_mean_t.var().item())

    # ── Collapse detection ────────────────────────────────────────────────────
    is_all_zero = mean_gate < COLLAPSE_ZERO_THRESHOLD
    is_all_one  = mean_gate > COLLAPSE_ONE_THRESHOLD

    dead_mask          = per_layer_mean_t < DEAD_LAYER_THRESHOLD
    dead_local_indices = [i for i, d in enumerate(dead_mask.tolist()) if d]
    dead_abs_indices   = [i + always_keep for i in dead_local_indices]

    is_low_entropy   = routing_entropy < LOW_ENTROPY_THRESHOLD
    is_high_variance = utilization_var > HIGH_UTILIZATION_VAR

    avg_active_layers = always_keep + per_layer_mean_t.sum().item()

    result = RouterDiagnosticsResult(
        mean_gate=mean_gate,
        gate_variance=gate_variance,
        routing_entropy=routing_entropy,
        per_layer_mean=per_layer_mean,
        per_layer_entropy=per_layer_ent,
        utilization_variance=utilization_var,
        is_all_zero=is_all_zero,
        is_all_one=is_all_one,
        dead_layer_count=len(dead_abs_indices),
        dead_layer_indices=dead_abs_indices,
        is_low_entropy=is_low_entropy,
        is_high_variance=is_high_variance,
        avg_active_layers=avg_active_layers,
        always_keep=always_keep,
    )

    if print_warnings and result.any_collapse:
        for msg in result.warning_messages(step):
            print(f"\n[ROUTER] {msg}", file=_out, flush=True)

    return result


def diagnose_gates_accumulate(
    gates_list:    List[torch.Tensor],   # list of [B, S, L] tensors
    always_keep:   int = 0,
    step:          int = 0,
    print_warnings: bool = True,
) -> RouterDiagnosticsResult:
    """
    Diagnose across multiple batches (e.g., full eval loop).

    Concatenates along batch dimension before computing metrics to avoid
    batch-size bias. More accurate than averaging per-batch diagnostics.

    Args:
        gates_list: List of [B, S, L] gate tensors from successive batches.
        always_keep: Always-active layer count.
        step: Current training step.
        print_warnings: Print warnings to stderr.

    Returns:
        RouterDiagnosticsResult aggregated over all batches.
    """
    if not gates_list:
        raise ValueError("diagnose_gates_accumulate: gates_list is empty")
    combined = torch.cat([g.float() for g in gates_list], dim=0)  # [total_B, S, L]
    return diagnose_gates(
        combined,
        always_keep=always_keep,
        step=step,
        print_warnings=print_warnings,
    )

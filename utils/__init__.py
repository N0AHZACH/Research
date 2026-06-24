"""
utils/ — Shared research utilities for DLR (Dynamic Layer Routing) experiments.

Modules:
  flop_accounting   — Defensible FLOP counting with paper-quality methodology
  router_diagnostics — Collapse detection, routing entropy, utilization metrics
"""
from utils.flop_accounting import (
    profile_layer_flops,
    estimate_layer_flops_per_token,
    compute_token_routing_flops,
    compute_stochastic_depth_flops,
    FlopAccountingResult,
    LayerFlopProfile,
)
from utils.router_diagnostics import (
    diagnose_gates,
    diagnose_gates_accumulate,
    compute_routing_entropy,
    RouterDiagnosticsResult,
)

__all__ = [
    "profile_layer_flops",
    "estimate_layer_flops_per_token",
    "compute_token_routing_flops",
    "compute_stochastic_depth_flops",
    "FlopAccountingResult",
    "LayerFlopProfile",
    "diagnose_gates",
    "diagnose_gates_accumulate",
    "compute_routing_entropy",
    "RouterDiagnosticsResult",
]

"""Structural FLOP accounting. Structural != wall-clock (report both)."""


def structural_flop_reduction_pct(mean_active_routed: float, always_keep: int, total_layers: int) -> float:
    eff = always_keep + mean_active_routed
    return 100.0 * (1.0 - eff / total_layers)

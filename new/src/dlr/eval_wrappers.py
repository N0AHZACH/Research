"""Eval-time routing machinery shared by experiments/evaluate.py.

- fixed_stochastic_mask(...): context manager installing one seeded per-layer
  0/1 mask on layers[always_keep:] so stochastic-depth variants are evaluated
  WITH dropping active (never silent full-depth fallback).
- RoutedHFLM: lm-eval HFLM subclass routing every _model_call through the
  two-pass discrete forward (ARC/HellaSwag/Winogrande/MMLU style loglikelihood
  tasks). generate_until (GSM8K et al.) is refused: routing-aware KV-cache
  decoding is unsolved here, and an unrouted number would be invalid.
"""
from contextlib import contextmanager

import torch

from .routing import RouterMLP, get_layers, make_gate_hook, routed_forward  # noqa: F401

# Tasks that decode (need _model_generate). Refused for token_dlr rows.
GENERATE_UNTIL_TASKS = {
    "gsm8k", "gsm8k_cot", "humaneval", "mbpp", "math", "minerva_math",
    "drop", "squad_completion", "triviaqa",
}


@contextmanager
def fixed_stochastic_mask(model, always_keep: int, skip_prob: float, seed: int = 1234):
    """Install one seeded coin-flip mask per routable layer for the duration."""
    layers = get_layers(model)
    g = torch.Generator().manual_seed(seed)
    R = len(layers) - always_keep
    coin = (torch.rand(R, generator=g) > skip_prob).float()
    real_handles = []
    for li, keep in enumerate(coin.tolist(), start=always_keep):
        keep_t = torch.tensor(float(keep))

        def hook(module, inputs, output, k=keep_t):
            hs_out = output[0] if isinstance(output, tuple) else output
            hs_in = inputs[0]
            gate = k.to(hs_out.device, dtype=hs_out.dtype).view(1, 1, 1)
            gate = gate.expand(hs_out.shape[0], hs_out.shape[1], 1)
            blended = gate * hs_out + (1 - gate) * hs_in
            if isinstance(output, tuple):
                return (blended,) + output[1:]
            return blended

        real_handles.append(layers[li].register_forward_hook(hook))
    try:
        yield [float(c) for c in coin.tolist()]
    finally:
        for h in real_handles:
            h.remove()


def check_tasks_routable(tasks, variant_name: str, variant_type: str):
    bad = [t for t in (tasks or []) if str(t).lower() in GENERATE_UNTIL_TASKS]
    if bad and variant_type == "token_dlr":
        raise RuntimeError(
            f"{variant_name}: generate_until tasks {bad} are not routed for token_dlr "
            "(no routing-aware KV cache). Exclude them from token_dlr rows."
        )


class RoutedHFLM:
    """Thin wrapper: build with HFLM class passed in to avoid hard lm-eval
    dependency at import time (evaluate.py passes it in)."""

    def __init__(self):
        self._gate_sum = None
        self._gate_tokens = 0

    @staticmethod
    def build(HFLM, model, tokenizer, router, always_keep, batch_size, max_length, device, tau=1.0):
        outer = RoutedHFLM()

        class _Routed(HFLM):
            def _model_call(self, inps, attn_mask=None, labels=None):
                if attn_mask is None:
                    attn_mask = torch.ones_like(inps)
                logits, gates, _ = routed_forward(
                    self.model, router, inps, attn_mask,
                    always_keep=always_keep, tau=tau, deterministic=True,
                )
                with torch.no_grad():
                    gs = gates.detach().float()
                    s = gs.sum(dim=(0, 1)).cpu()
                    outer._gate_sum = s if outer._gate_sum is None else outer._gate_sum + s
                    outer._gate_tokens += gs.shape[0] * gs.shape[1]
                return logits

            def _model_generate(self, *a, **k):
                raise NotImplementedError(
                    "generate_until is not routed for token_dlr (no routing-aware "
                    "KV cache). Exclude GSM8K-style tasks from token_dlr rows."
                )

        lm = _Routed(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
            max_length=max_length, truncation=True, device=device,
        )
        return lm, outer

    def summary(self, always_keep: int):
        if not self._gate_tokens:
            return {}
        per_layer = (self._gate_sum / self._gate_tokens).tolist()
        mean_gate = float(self._gate_sum.sum() / (self._gate_tokens * len(per_layer)))
        return {"per_layer_activity": per_layer, "mean_gate": mean_gate,
                "achieved_skip": 1.0 - mean_gate}

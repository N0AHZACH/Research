"""Two-pass hook routing. No model surgery. SmolLM2/Llama path first."""
import torch
import torch.nn as nn


class StopForwardException(Exception):
    pass


class RouterMLP(nn.Module):
    def __init__(self, hidden_size: int, num_routable_layers: int, init_keep_prob: float = 0.7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_routable_layers),
        )
        # Start near the target compute budget instead of 50/50: final bias =
        # logit(keep_prob) so initial gates average ~init_keep_prob. Without this,
        # KD+LM gradients drag an unbiased router to all-keep within ~50 steps.
        import math
        last = self.net[-1]
        nn.init.normal_(last.weight, std=0.02)
        nn.init.constant_(last.bias, math.log(init_keep_prob / (1 - init_keep_prob)))

    def forward(self, h: torch.Tensor, tau: float = 1.0, hard: bool = True, deterministic: bool = False):
        logits = self.net(h.float())  # [B, S, R] — router stays fp32 for stability
        if deterministic:
            prob = torch.sigmoid(logits)
            hard_gates = (prob > 0.5).float()
            return hard_gates - prob.detach() + prob, logits
        return gumbel_sigmoid(logits, tau=tau, hard=hard), logits


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, eps: float = 1e-10):
    # Binary concrete with STE: symmetric logistic noise via Gumbel difference.
    u1 = torch.rand_like(logits).clamp(eps, 1 - eps)
    u2 = torch.rand_like(logits).clamp(eps, 1 - eps)
    noise = torch.log(-torch.log(u2 + eps) + eps) - torch.log(-torch.log(u1 + eps) + eps)
    # noise is Logistic(0,1); equivalent to starter up to sign (symmetric).
    y_soft = torch.sigmoid((logits + noise) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class _Store:
    def __init__(self):
        self.value = None


def get_layers(model):
    """Robust layer-path dispatch. Prints tree on failure instead of guessing."""
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if hasattr(base, "model") and hasattr(base.model, "layers"):
                return base.model.layers
        except Exception:
            pass
    for obj in [model]:
        if hasattr(obj, "model") and hasattr(obj.model, "layers"):
            return obj.model.layers
    # PEFT: model.base_model.model.model.layers
    try:
        cand = model.base_model.model.model.layers
        return cand
    except Exception:
        pass
    try:
        names = [n for n, _ in model.named_modules()]
        print("LAYER-PATH MISMATCH. Top-level modules:")
        for n in names[:60]:
            print("  ", n)
    except Exception as e:
        print("Could not list modules:", e)
    raise RuntimeError("Could not find model.model.layers on this checkpoint.")


def make_gate_hook(gate_for_layer: torch.Tensor):
    def hook(module, inputs, output):
        hs_out = output[0] if isinstance(output, tuple) else output
        hs_in = inputs[0]
        gate = gate_for_layer.unsqueeze(-1).to(hs_out.dtype)
        blended = gate * hs_out + (1 - gate) * hs_in
        if isinstance(output, tuple):
            return (blended,) + output[1:]
        return blended
    return hook


def routed_forward(model, router, input_ids, attention_mask, always_keep, tau=1.0, deterministic=False):
    layers = get_layers(model)
    store = _Store()

    def _cap(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        store.value = hs.detach()  # detach: router input treated as features
        raise StopForwardException()

    h = layers[always_keep - 1].register_forward_hook(_cap)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForwardException:
        pass
    finally:
        h.remove()
    if store.value is None:
        raise RuntimeError("Phase-1 capture failed.")

    # router forward needs grad: re-attach by running router on detached feats
    # (router weights still get grad via gates -> loss).
    feats = store.value.to(next(router.parameters()).device if list(router.parameters()) else store.value.device)
    # ensure feats requires no grad (features), gates carry grad via router params
    gates, gate_logits = router(feats, tau=tau, hard=True, deterministic=deterministic)

    handles = []
    for i, li in enumerate(range(always_keep, len(layers))):
        handles.append(layers[li].register_forward_hook(make_gate_hook(gates[:, :, i])))
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for x in handles:
            x.remove()
    return out.logits, gates, gate_logits

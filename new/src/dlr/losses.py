"""Shared loss composer — SAME for stochastic and token_dlr (fairness)."""
import torch
import torch.nn.functional as F


def kd_kl_loss(student_logits, teacher_logits, temperature: float = 2.0):
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def skip_reg(gates: torch.Tensor, target: float):
    return (1.0 - gates.mean() - target) ** 2


def compose_routed_loss(lm_loss, student_logits, teacher_logits, gates, target,
                        kd_weight=0.5, skip_weight=2.0, l1_weight=0.01, temperature=2.0):
    kd = kd_kl_loss(student_logits, teacher_logits, temperature)
    reg = skip_reg(gates, target)
    l1 = gates.mean()
    total = lm_loss + kd_weight * kd + skip_weight * reg + l1_weight * l1
    return total, {"lm": lm_loss.item(), "kd": kd.item(), "skip_reg": reg.item(), "l1": l1.item()}


def teacher_logits_frozen(model, input_ids, attention_mask):
    with torch.no_grad(), model.disable_adapter():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits

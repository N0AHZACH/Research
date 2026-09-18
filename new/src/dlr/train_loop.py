"""One loop for dense / stochastic / token_dlr. Shared optimizer/sched/data."""
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .losses import compose_routed_loss, teacher_logits_frozen
from .routing import get_layers, make_gate_hook, routed_forward
from .routing_metrics import routing_metrics_from_gates


def shifted_ce(logits, labels):
    sl = logits[:, :-1, :].contiguous().float()
    lb = labels[:, 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)


@torch.no_grad()
def eval_ppl(model, router, eval_ds, cfg, collate_fn):
    model.eval()
    if router is not None:
        router.eval()
    loader = DataLoader(eval_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)
    tot_nll, tot_tok = 0.0, 0
    layers = get_layers(model)
    g = torch.Generator().manual_seed(1234)
    vt = getattr(cfg, "variant_type", "dense")
    is_stochastic = vt.startswith("stochastic")
    for batch in loader:
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)
        lab = batch["labels"].to(cfg.device)
        if vt == "dense":
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
        elif is_stochastic:
            handles = []
            for li in range(cfg.always_keep, len(layers)):
                keep = (torch.rand(1, generator=g).item() > cfg.target_skip)
                gate = torch.full((ids.shape[0], ids.shape[1]), float(keep), device=ids.device)
                handles.append(layers[li].register_forward_hook(make_gate_hook(gate)))
            try:
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits
            finally:
                for h in handles:
                    h.remove()
        else:
            logits, _, _ = routed_forward(model, router, ids, mask, cfg.always_keep,
                                          tau=cfg.tau, deterministic=True)
        loss = shifted_ce(logits, lab)
        n_tok = (lab[:, 1:] != -100).sum().item()
        tot_nll += loss.item() * n_tok
        tot_tok += n_tok
    model.train()
    if router is not None:
        router.train()
    return float(torch.exp(torch.tensor(tot_nll / max(tot_tok, 1))).item())


def train(model, router, train_ds, eval_ds, cfg, collate_fn):
    device = cfg.device
    model.to(device)
    model.train()
    # Trainable-only optimizer: PEFT base params are frozen (requires_grad=False).
    # Including them would allocate Adam m/v for 1B+ params and OOM on 8GB GPUs.
    train_params = [p for p in model.parameters() if p.requires_grad]
    if router is not None:
        train_params += [p for p in router.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(train_params, lr=cfg.lr)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = cfg.max_steps
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=cfg.warmup_steps,
                                            num_training_steps=max(total_steps, 1))
    layers = get_layers(model)
    logs = []
    t0 = time.perf_counter()
    it = iter(loader)
    last_metrics = {}
    for step in range(cfg.max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lab = batch["labels"].to(device)

        if cfg.variant_type == "dense":
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            loss = out.loss
            comp = {"lm": loss.item()}
            gates = None
        elif cfg.variant_type.startswith("stochastic"):
            handles = []
            R = len(layers) - cfg.always_keep
            coin = (torch.rand(R, device=device) > cfg.target_skip).float()  # 1=keep
            gates = coin.view(1, 1, R).expand(ids.shape[0], ids.shape[1], R)
            for i, li in enumerate(range(cfg.always_keep, len(layers))):
                gate_bs = gates[:, :, i]
                handles.append(layers[li].register_forward_hook(make_gate_hook(gate_bs)))
            try:
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits
            finally:
                for h in handles:
                    h.remove()
            lm = shifted_ce(logits, lab)
            teacher = teacher_logits_frozen(model, ids, mask)
            loss, comp = compose_routed_loss(lm, logits, teacher, gates, cfg.target_skip,
                                             cfg.kd_weight, cfg.skip_weight, cfg.l1_weight, cfg.kd_temp)
        elif cfg.variant_type.startswith("token_dlr") or cfg.variant_type.startswith("dlr"):
            logits, gates, _ = routed_forward(model, router, ids, mask, cfg.always_keep,
                                              tau=cfg.tau,
                                              deterministic=getattr(cfg, "router_deterministic", True))
            lm = shifted_ce(logits, lab)
            teacher = teacher_logits_frozen(model, ids, mask)
            loss, comp = compose_routed_loss(lm, logits, teacher, gates, cfg.target_skip,
                                             cfg.kd_weight, cfg.skip_weight, cfg.l1_weight, cfg.kd_temp)
        else:
            raise ValueError(cfg.variant_type)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at step {step}: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

        entry = {"step": step, "loss": loss.item(), **comp}
        if gates is not None:
            with torch.no_grad():
                m = routing_metrics_from_gates(gates.detach())
            last_metrics = m
            entry.update({k: v for k, v in m.items() if k != "per_layer_activity"})
        logs.append(entry)
        if step % cfg.log_every == 0:
            msg = f"[step {step}] loss={loss.item():.4f}"
            if "mean_gate" in entry:
                msg += f" skip={1-entry['mean_gate']:.3f} ent={entry.get('router_entropy',0):.3f}"
            print(msg, flush=True)

    wall = time.perf_counter() - t0
    # final metrics on train tail + eval ppl
    final_skip = (1 - last_metrics.get("mean_gate", float("nan"))) if last_metrics else float("nan")
    ppl = eval_ppl(model, router, eval_ds, cfg, collate_fn)
    return {"logs": logs, "wall_clock_s": wall, "final_metrics": last_metrics,
            "final_skip": final_skip, "eval_ppl": ppl}

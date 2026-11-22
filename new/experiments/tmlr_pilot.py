"""Paired native and shuffled route comparison on WikiText validation."""
import argparse
import json
from pathlib import Path

if __package__:
    from . import evaluate as ev
else:
    import evaluate as ev


def shuffled_routes(torch, seed, *, whole_sequences=False):
    generator = torch.Generator().manual_seed(seed)

    def permute(rows, device):
        order = rows[torch.randperm(len(rows), generator=generator)].to(device)
        return order, order.roll(1)

    def shuffle(gates, mask):
        valid = mask.bool()
        cpu_mask = valid.cpu()
        result = gates.clone()
        eligible = 0
        if whole_sequences:
            # Swap trajectories only between passages with identical valid positions.
            _, groups = torch.unique(cpu_mask, dim=0, return_inverse=True)
            for group in groups.unique():
                rows = (groups == group).nonzero().flatten()
                if len(rows) > 1:
                    order, donors = permute(rows, gates.device)
                    result[order] = torch.where(valid[order, :, None], gates[donors], gates[order])
                    eligible += int(cpu_mask[rows].sum())
        else:
            for pos, column in enumerate(cpu_mask.T):
                rows = column.nonzero().flatten()
                if len(rows) > 1:
                    order, donors = permute(rows, gates.device)
                    result[order, pos] = gates[donors, pos]
                    eligible += len(rows)
        shuffle.changed += int(((result != gates).any(dim=-1) & valid).sum().item())
        shuffle.eligible += eligible
        shuffle.total += int(cpu_mask.sum())
        return result

    shuffle.changed = shuffle.eligible = shuffle.total = 0
    return shuffle


def check_forward(torch, loaded):
    loaded.model.eval()
    ids = loaded.tokenizer("A quick check of the routing implementation.", return_tensors="pt").input_ids.to(next(loaded.model.parameters()).device)
    mask = torch.ones_like(ids)
    pad_ids = torch.nn.functional.pad(ids, (0, 8), value=loaded.tokenizer.pad_token_id)
    pad_mask = torch.nn.functional.pad(mask, (0, 8))
    with torch.no_grad():
        reference = loaded.model(input_ids=ids, attention_mask=mask).logits
        opened, _ = ev.routed_forward(torch, loaded, ids, mask,
                                      gate_transform=lambda gates, mask: torch.ones_like(gates))
        torch.testing.assert_close(opened.logits, reference)
        native, routes = ev.routed_forward(torch, loaded, ids, mask, labels=ids)
        for repeats in [1, 2]:
            labels = torch.nn.functional.pad(ids, (0, 8), value=-100).repeat(repeats, 1)
            out, padded_routes = ev.routed_forward(torch, loaded, pad_ids.repeat(repeats, 1),
                                                    pad_mask.repeat(repeats, 1), labels=labels)
            torch.testing.assert_close(out.loss, native.loss, rtol=0.005, atol=0.005)
            torch.testing.assert_close(padded_routes[:, :ids.shape[1]], routes.repeat(repeats, 1, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 2:
        p.error("Shuffling requires at least two examples per batch")
    manifest = ev.load_json(ev.REPO_ROOT / "configs/eval/tinyllama1b-010-g4-500.json")
    variant = next(v for v in manifest["variants"] if v["name"] == "token_dlr_50")
    variant.update(checkpoint=str(args.checkpoint), router_weights=str(args.checkpoint / "router_weights.pt"))
    manifest.update(batch_size=args.batch_size, variants=[variant], tasks=[])
    if args.output.exists():
        p.error("Choose a new output filename to preserve previous results")
    import torch
    loaded = ev.load_variant(manifest, variant)
    check_forward(torch, loaded)
    results = {}
    for seed in [None, 11, 22, 33]:
        transform = None if seed is None else shuffled_routes(torch, seed)
        ppl, routing = ev.evaluate_perplexity(loaded, manifest, gate_transform=transform)
        name = "native" if seed is None else f"shuffle_{seed}"
        results[name] = {"ppl": ppl, "routing": routing}
        print(name, ppl["perplexity"], routing["achieved_skip"], flush=True)
        native = results["native"]["routing"]["per_layer_activity"]
        if any(abs(a-b) > 1e-6 for a,b in zip(native, routing["per_layer_activity"])):
            raise RuntimeError("Per-layer budgets changed across conditions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump({"manifest": manifest, "commit": ev.git_commit(), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()

"""Frozen small-data layer-routing study. Run as python -m new.experiments.tmlr_study."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from . import evaluate as ev
from .tmlr_pilot import check_forward, shuffled_routes

MODELS = {"tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "qwen": "Qwen/Qwen2.5-1.5B"}
SEEDS = (42, 43, 44)
TARGETS = (30, 50)
DRAWS = (11, 22, 33)
TASKS = ("arc_challenge", "hellaswag")
DATA = "Salesforce/wikitext"


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, default=str, allow_nan=False, indent=1))
    tmp.replace(path)


def cache(path, operation):
    if not path.exists():
        save(path, operation())
    return read(path)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def freeze():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revision = api.dataset_info(DATA).sha
    splits = {}
    for split, size in [("train", 300), ("validation", 50), ("test", 500)]:
        data = load_dataset(DATA, "wikitext-103-raw-v1", split=split, revision=revision)
        rows = []
        for idx, row in enumerate(data):
            if len(row["text"]) > 100:
                rows.append([idx, hashlib.sha256(row["text"].encode()).hexdigest()])
            if len(rows) == size:
                break
        if len(rows) != size:
            raise ValueError(f"Insufficient qualifying rows in {split}")
        splits[split] = rows
    return {
        "commit": git("rev-parse", "HEAD"),
        "dataset_revision": revision,
        "splits": splits,
        "models": {k: api.model_info(v).sha for k, v in MODELS.items()},
        "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ),
        "seeds": SEEDS,
        "targets": TARGETS,
        "tasks": TASKS,
        "draws": DRAWS,
    }


def mask_for(order, k):
    dropped = set(order[:k])
    return [int(i not in dropped) for i in range(len(order))]


def fit(folder, name, model, frozen, seed, variant, extra=()):
    path = folder / name
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            stage = Path(tmp) / "checkpoint"
            subprocess.run(
                [
                    sys.executable,
                    "new/experiments/train.py",
                    "--model-id",
                    model,
                    "--data-revision",
                    frozen["dataset_revision"],
                    "--expanded-lora",
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--max-steps",
                    "500",
                    "--batch-size",
                    "4",
                    "--skip-weight",
                    "1600",
                    "--output-dir",
                    str(stage),
                    "--run-id",
                    name,
                    *extra,
                ],
                check=True,
            )
            files = [
                "adapter_model.safetensors",
                "adapter_config.json",
                "manifest.json",
            ]
            if variant.startswith("token_dlr"):
                files.append("router_weights.pt")
            save(stage / "hashes.json", {f: ev.file_sha256(stage / f) for f in files})
            stage.rename(path)
    for f, digest in read(path / "hashes.json").items():
        if ev.file_sha256(path / f) != digest:
            raise ValueError(f"Checkpoint changed: {path / f}")
    return path


def arm(path, mask=None):
    m = read(path / "manifest.json")
    spec = {"name": path.name, "type": "lora", "checkpoint": str(path)}
    if mask is not None:
        spec.update(
            type="stochastic", always_keep_layers=m["always_keep"], keep_mask=mask
        )
    elif (path / "router_weights.pt").exists():
        spec.update(
            type="token_dlr",
            always_keep_layers=m["always_keep"],
            router_weights=str(path / "router_weights.pt"),
        )
    return spec


def measure(spec, model, frozen, split, shuffle=None, tasks=False, check=False):
    import torch

    cfg = {
        "model_id": model,
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "batch_size": 4,
        "log_samples": True,
        "num_fewshot": 0,
        "max_length": 1024,
        "tasks": list(TASKS),
        "perplexity": {
            "dataset": DATA,
            "config": "wikitext-103-raw-v1",
            "split": split,
            "samples": len(frozen["splits"][split]),
            "max_length": 128,
            "revision": frozen["dataset_revision"],
        },
    }
    loaded = ev.load_variant(cfg, spec)
    try:
        if check:
            check_forward(torch, loaded)
        transform = None if shuffle is None else shuffled_routes(torch, shuffle)
        ppl, route = ev.evaluate_perplexity(loaded, cfg, transform)
        if [x[0] for x in ppl["examples"]] != [x[1] for x in frozen["splits"][split]]:
            raise ValueError("Evaluation examples disagree with frozen split")
        result = {"variant": spec, "ppl": ppl, "routing": route}
        if transform:
            result["changed_fraction"] = transform.changed / transform.total
        if tasks:
            cfg["batch_size"] = 1
            raw = ev.run_lm_eval(loaded, cfg)
            result["task_routing"] = raw.get("routing_summary", {})
            result["task_versions"] = raw.get("versions", {})
            result["task_configs"] = raw.get("configs", {})
            result["tasks"] = {}
            for task in TASKS:
                samples = raw["samples"][task]
                result["tasks"][task] = [
                    [s["doc_hash"], float(s["acc_norm"])] for s in samples
                ]
                if len(samples) != {"arc_challenge": 1172, "hellaswag": 10042}[task]:
                    raise ValueError(f"Incomplete {task} evaluation")
        return result
    finally:
        ev.cleanup_loaded(loaded)


def job(root, family, seed, pilot=False, replay=False):
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bfloat16 support required")
    frozen = read(root / "protocol.json")
    model = snapshot_download(
        MODELS[family],
        revision=frozen["models"][family],
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"],
    )
    folder = root / f"{family}-{seed}"
    folder.mkdir(exist_ok=True)

    def score(name, spec, split="test", **kw):
        return cache(
            folder / (name + ".json"), lambda: measure(spec, model, frozen, split, **kw)
        )

    if replay:
        path = folder / "dlr50"
        for f, digest in read(path / "hashes.json").items():
            if ev.file_sha256(path / f) != digest:
                raise ValueError(f"Replay checkpoint changed: {f}")
        expected = read(folder / "native50.json")
        actual = measure(arm(path), model, frozen, "test", check=True)
        if (
            actual["ppl"]["examples"] != expected["ppl"]["examples"]
            or actual["routing"] != expected["routing"]
        ):
            raise ValueError(
                "Replay differs from saved row; investigate numerical or input drift"
            )
        save(folder / "replay.json", {"commit": frozen["commit"], "matched": True})
        return
    if pilot:
        spec = arm(fit(folder, "dlr50", model, frozen, seed, "token_dlr_50"))
        native = score("pilot_native", spec, "validation", check=True)
        for draw in DRAWS:
            verify_shuffle(
                native, score(f"pilot_shuffle{draw}", spec, "validation", shuffle=draw)
            )
        return

    dense = fit(folder, "dense", model, frozen, seed, "dense")
    meta = read(dense / "manifest.json")
    layers = meta["total_layers"] - meta["always_keep"]
    # Rank each singleton deletion using dense-adapted validation loss.
    order = []
    for i in range(layers):
        mask = [int(j != i) for j in range(layers)]
        val = score(f"leave_out_{i}", arm(dense, mask), "validation")
        order.append((val["ppl"]["negative_log_likelihood"], i))
    ranking = [i for _, i in sorted(order)]
    score("full", arm(dense), tasks=True)
    for budget in TARGETS:
        spec = arm(
            fit(folder, f"dlr{budget}", model, frozen, seed, f"token_dlr_{budget}")
        )
        cal = score(f"cal{budget}", spec, "validation", check=True)
        skip = cal["routing"]["achieved_skip"]
        native = score(f"native{budget}", spec, tasks=True)
        for draw in DRAWS:
            verify_shuffle(native, score(f"shuffle{budget}_{draw}", spec, shuffle=draw))
        for k in sorted({math.floor(layers * skip), math.ceil(layers * skip)}):
            mask = mask_for(ranking, k)
            path = fit(
                folder,
                f"static{k}",
                model,
                frozen,
                seed,
                f"stochastic_{budget}",
                ["--fixed-mask", json.dumps(mask), "--target-skip", str(k / layers)],
            )
            score(f"fixed{k}", arm(path, mask), tasks=True)
        stochastic = fit(
            folder,
            f"stoch{budget}",
            model,
            frozen,
            seed,
            f"stochastic_{budget}",
            ["--target-skip", str(skip)],
        )
        for draw in DRAWS:
            order = list(range(layers))
            random.Random(draw).shuffle(order)
            mask = mask_for(order, round(skip * layers))
            score(f"random{budget}_{draw}", arm(stochastic, mask), tasks=True)


def verify_shuffle(native, shuffled):
    a = native["routing"]["per_layer_activity"]
    b = shuffled["routing"]["per_layer_activity"]
    if len(a) != len(b) or any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
        raise ValueError("Shuffle changed per-layer activity")


def table(path, rows):
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def report(root):
    from collections import defaultdict

    import numpy as np

    groups = defaultdict(list)
    summary, contrasts, matched = [], [], []
    for family in MODELS:
        for seed in SEEDS:
            folder = root / f"{family}-{seed}"
            for budget in TARGETS:
                cal = read(folder / f"cal{budget}.json")["routing"]
                r = len(cal["per_layer_activity"])
                k = cal["achieved_skip"] * r
                native = read(folder / f"native{budget}.json")
                low, high = math.floor(k), math.ceil(k)
                measured = native["routing"]["achieved_skip"] * r
                low_nll = read(folder / f"fixed{low}.json")["ppl"][
                    "negative_log_likelihood"
                ]
                high_nll = read(folder / f"fixed{high}.json")["ppl"][
                    "negative_log_likelihood"
                ]
                bracketed = low - 1e-6 <= measured <= high + 1e-6
                weight = (measured - low) / (high - low) if high > low else 0
                matched.append(
                    {
                        "family": family,
                        "seed": seed,
                        "target": budget,
                        "test_mean_dropped": measured,
                        "low": low,
                        "high": high,
                        "bracketed": bracketed,
                        "native_minus_interpolated_static_nll": (
                            native["ppl"]["negative_log_likelihood"]
                            - (low_nll * (1 - weight) + high_nll * weight)
                            if bracketed
                            else ""
                        ),
                    }
                )
                for name in [
                    "full",
                    f"native{budget}",
                    *[f"fixed{i}" for i in sorted({math.floor(k), math.ceil(k)})],
                    *[f"random{budget}_{d}" for d in DRAWS],
                    *[f"shuffle{budget}_{d}" for d in DRAWS],
                ]:
                    row = read(folder / f"{name}.json")
                    ppl = row["ppl"]
                    summary.append(
                        dict(
                            family=family,
                            seed=seed,
                            target=budget,
                            arm=name,
                            skip=row["routing"].get("achieved_skip", 0),
                            ppl=ppl["perplexity"],
                            scored_tokens_per_s=ppl["tokens_per_sec"],
                            changed_fraction=row.get("changed_fraction", ""),
                            task_skip=(
                                row.get("task_routing", {}).get(
                                    "achieved_skip",
                                    row["routing"].get("achieved_skip", 0),
                                )
                                if row.get("tasks")
                                else ""
                            ),
                            **{
                                t: np.mean([v for _, v in items])
                                for t, items in row.get("tasks", {}).items()
                            },
                        )
                    )
                    for metric in ["nll", *row.get("tasks", {})]:
                        if metric == "nll":
                            a, b = native["ppl"]["examples"], ppl["examples"]
                            weights = np.array([x[1] for x in a])
                        else:
                            a, b = native["tasks"][metric], row["tasks"][metric]
                            weights = np.ones(len(a))
                        if [x[:-1] for x in a] != [x[:-1] for x in b]:
                            raise ValueError(
                                f"Unpaired items {family}/{seed}/{name}/{metric}"
                            )
                        delta = np.array([x[-1] - y[-1] for x, y in zip(a, b)])
                        effect = delta.sum() / weights.sum()
                        rng = np.random.default_rng(20261122)
                        samples = []
                        for _ in range(20):
                            ix = rng.integers(len(a), size=(50, len(a)))
                            samples.extend(delta[ix].sum(1) / weights[ix].sum(1))
                        lo, hi = np.quantile(samples, [0.025, 0.975])
                        contrasts.append(
                            {
                                "family": family,
                                "seed": seed,
                                "target": budget,
                                "comparator": name,
                                "metric": metric,
                                "native_minus_control": effect,
                                "low": lo,
                                "high": hi,
                            }
                        )
                        label = name
                        if name.startswith("fixed"):
                            label = (
                                "static_floor"
                                if name == f"fixed{math.floor(k)}"
                                else "static_ceil"
                            )
                        groups[family, budget, label, metric].append(effect)
    table(root / "main.csv", summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 4), squeeze=False)
    for ax, family in zip(axes[0], MODELS):
        for prefix, label, marker in [
            ("full", "Dense", "s"),
            ("native", "Learned", "o"),
            ("fixed", "Static", "^"),
            ("random", "Random mask", "x"),
        ]:
            points = {
                (v["skip"], v["ppl"])
                for v in summary
                if v["family"] == family and v["arm"].startswith(prefix)
            }
            if points:
                x, y = zip(*sorted(points))
                ax.scatter(
                    [100 * i for i in x], y, label=label, marker=marker, alpha=0.65
                )
        ax.set(
            title=MODELS[family].split("/")[-1],
            xlabel="Routable layers skipped (%)",
            ylabel="Test perplexity (log scale)",
            yscale="log",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "quality_depth.pdf")
    plt.close(fig)
    table(root / "paired.csv", contrasts)
    table(root / "static_budget_comparison.csv", matched)
    table(
        root / "seeds.csv",
        [
            {
                "family": f,
                "target": b,
                "comparator": n,
                "metric": m,
                "n_seeds": len(v),
                "mean": np.mean(v),
                "sd": np.std(v, ddof=1) if len(v) > 1 else "",
            }
            for (f, b, n, m), v in groups.items()
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["freeze", "pilot", "run", "report", "replay", "all"]
    )
    p.add_argument("--out", type=Path, default=Path("new/artifacts/tmlr-2026-11-22"))
    p.add_argument("--family", choices=MODELS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    args = p.parse_args()
    if (args.family or args.seed) and args.stage != "run":
        p.error("--family and --seed apply only to the run stage")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(
                "Commit tracked edits before freezing or continuing the study"
            )
        if args.stage in ("all", "freeze"):
            cache(root / "protocol.json", freeze)
        frozen = read(root / "protocol.json")
        if frozen["commit"] != git("rev-parse", "HEAD"):
            raise RuntimeError(
                "Use the frozen commit or start a separate output directory"
            )
        if args.stage in ("all", "pilot"):
            job(root, "tiny", 42, pilot=True)
        if args.stage in ("all", "run"):
            for family in [args.family] if args.family else MODELS:
                for seed in [args.seed] if args.seed else SEEDS:
                    job(root, family, seed)
        if args.stage in ("all", "replay"):
            job(root, "tiny", 42, replay=True)
        if args.stage in ("all", "report"):
            report(root)
            artifacts = sorted(
                p
                for p in root.rglob("*")
                if p.is_file()
                and p.name not in {".lock", "SHA256SUMS"}
                and not p.name.endswith(".partial")
            )
            with (root / "SHA256SUMS").open("w") as f:
                for path in artifacts:
                    f.write(f"{ev.file_sha256(path)}  {path.relative_to(root)}\n")
            print("Tables written to", root)


if __name__ == "__main__":
    main()

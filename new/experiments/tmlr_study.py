"""Frozen small-data layer-routing study. Run as python -m new.experiments.tmlr_study."""

import argparse
import csv
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
SHUFFLES = ("shuffle", "sequence_shuffle")
STATIC_MARGIN_NLL = math.log(1.02)
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
        "shuffle_controls": SHUFFLES,
        "static_margin_nll": STATIC_MARGIN_NLL,
    }


def mask_for(order, k):
    dropped = set(order[:k])
    return [int(i not in dropped) for i in range(len(order))]


def fit(folder, name, model, frozen, seed, variant, extra=(), *, steps=500):
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
                    str(steps),
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
                "summary.json",
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
        "merge_lora": False,
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
            "length_matched_batches": True,
        },
    }
    torch.cuda.reset_peak_memory_stats()
    loaded = ev.load_variant(cfg, spec)
    try:
        if check:
            check_forward(torch, loaded)
        transform = None
        if shuffle is not None:
            kind, draw = shuffle
            transform = shuffled_routes(
                torch, draw, whole_sequences=kind == "sequence_shuffle"
            )
        ppl, route = ev.evaluate_perplexity(loaded, cfg, transform)
        if [x[0] for x in ppl["examples"]] != [x[1] for x in frozen["splits"][split]]:
            raise ValueError("Evaluation examples disagree with frozen split")
        result = {"variant": spec, "ppl": ppl, "routing": route}
        if transform is not None:
            result["changed_fraction"] = transform.changed / transform.total
            result["eligible_fraction"] = transform.eligible / transform.total
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
        result["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        return result
    finally:
        ev.cleanup_loaded(loaded)


def fixed_candidates(ranking, dropped):
    """Consider the singleton ranking and a bounded set of one-layer replacements."""
    selected = ranking[:dropped]
    yield sorted(selected)
    if dropped > 1:
        for i in range(max(0, dropped - 3), dropped):
            for replacement in ranking[dropped : dropped + 4]:
                yield sorted([*selected[:i], *selected[i + 1 :], replacement])


def load_study_model(frozen, family):
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bfloat16 support required")
    return snapshot_download(
        MODELS[family],
        revision=frozen["models"][family],
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"],
    )


def preflight(root, family):
    frozen = read(root / "protocol.json")
    model = load_study_model(frozen, family)
    folder = root / f"preflight-{family}"
    folder.mkdir(exist_ok=True)
    summary = {}
    for name, variant in [("dense", "dense"), ("dlr50", "token_dlr_50")]:
        path = fit(folder, name, model, frozen, SEEDS[0], variant, steps=2)
        training = read(path / "summary.json")
        evaluation = cache(
            folder / f"{name}_validation.json",
            lambda path=path, check=name == "dlr50": measure(
                arm(path), model, frozen, "validation", check=check
            ),
        )
        summary[name] = {
            "training_seconds": training["wall_clock_s"],
            "training_cuda_memory": training["cuda_memory"],
            "validation_cuda_memory": evaluation["cuda_memory"],
        }
    save(folder / "resources.json", summary)
    print(json.dumps(summary, indent=2))


def job(root, family, seed, pilot=False, replay=False):
    frozen = read(root / "protocol.json")
    model = load_study_model(frozen, family)
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
            or actual["ppl"]["batch_ids"] != expected["ppl"]["batch_ids"]
            or actual["routing"] != expected["routing"]
        ):
            raise ValueError(
                "Replay differs from saved row; investigate numerical or input drift"
            )
        save(folder / "replay.json", {"commit": frozen["commit"], "matched": True})
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

    def fit_static(dropped):
        def select_mask():
            def evaluate_mask(indices):
                mask = [int(i not in indices) for i in range(layers)]
                suffix = "_".join(map(str, indices)) or "none"
                name = (
                    f"leave_out_{indices[0]}"
                    if len(indices) == 1
                    else f"static_search_{suffix}"
                )
                value = score(name, arm(dense, mask), "validation")
                return value["ppl"]["negative_log_likelihood"], mask

            nll, mask = min(
                evaluate_mask(indices) for indices in fixed_candidates(ranking, dropped)
            )
            return {"selection_nll": nll, "keep_mask": mask}

        selection = cache(folder / f"static{dropped}_selection.json", select_mask)
        mask = selection["keep_mask"]
        path = fit(
            folder,
            f"static{dropped}",
            model,
            frozen,
            seed,
            "stochastic_30",
            ["--fixed-mask", json.dumps(mask), "--target-skip", str(dropped / layers)],
        )
        return arm(path, mask)

    if not pilot:
        score("full", arm(dense), tasks=True)
    for budget in [50] if pilot else TARGETS:
        spec = arm(
            fit(folder, f"dlr{budget}", model, frozen, seed, f"token_dlr_{budget}")
        )
        cal = score(f"cal{budget}", spec, "validation", check=True)
        skip = cal["routing"]["achieved_skip"]
        split = "validation" if pilot else "test"
        native = cal if pilot else score(f"native{budget}", spec, tasks=True)
        rows = {"native": native}
        if pilot:
            rows["dense"] = score("pilot_dense", arm(dense), split)
        for control in SHUFFLES:
            for draw in DRAWS:
                name = f"{control}{budget}_{draw}"
                label = f"pilot_{name}" if pilot else name
                rows[name] = score(label, spec, split, shuffle=(control, draw))
                verify_shuffle(native, rows[name])
        for k in sorted({math.floor(layers * skip), math.ceil(layers * skip)}):
            name = f"fixed{k}"
            rows[name] = score(
                f"pilot_{name}" if pilot else name,
                fit_static(k),
                split,
                tasks=not pilot,
            )
        if pilot:
            summary = {
                name: {
                    "ppl": row["ppl"]["perplexity"],
                    "native_minus_nll": native["ppl"]["negative_log_likelihood"]
                    - row["ppl"]["negative_log_likelihood"],
                    "skip": row["routing"].get("achieved_skip", 0),
                    **{
                        field: row[field]
                        for field in ("eligible_fraction", "changed_fraction")
                        if field in row
                    },
                }
                for name, row in rows.items()
            }
            save(folder / "pilot_summary.json", summary)
            print(json.dumps(summary, indent=2))
            return
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


def paired_nll_inputs(native, control):
    import numpy as np

    a, b = native["examples"], control["examples"]
    if native["batch_ids"] != control["batch_ids"] or [x[:-1] for x in a] != [
        x[:-1] for x in b
    ]:
        raise ValueError("Unpaired passages or batches")
    groups = np.array(native["batch_ids"])
    weights = np.bincount(groups, weights=[x[1] for x in a])
    delta = np.bincount(
        groups, weights=[x[-1] - y[-1] for x, y in zip(a, b, strict=True)]
    )
    return delta, weights


def paired_interval(delta, weights):
    import numpy as np

    effect = delta.sum() / weights.sum()
    rng = np.random.default_rng(20261122)
    samples = []
    for _ in range(20):
        indices = rng.integers(len(delta), size=(50, len(delta)))
        samples.extend(delta[indices].sum(1) / weights[indices].sum(1))
    low, high = np.quantile(samples, [0.025, 0.975])
    return effect, low, high


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
                bracketed = low - 1e-6 <= measured <= high + 1e-6
                effect = ci_low = ci_high = within_margin = ""
                if bracketed:
                    weight = (
                        min(1, max(0, (measured - low) / (high - low)))
                        if high > low
                        else 0
                    )
                    low_delta, weights = paired_nll_inputs(
                        native["ppl"], read(folder / f"fixed{low}.json")["ppl"]
                    )
                    high_delta, _ = paired_nll_inputs(
                        native["ppl"], read(folder / f"fixed{high}.json")["ppl"]
                    )
                    effect, ci_low, ci_high = paired_interval(
                        (1 - weight) * low_delta + weight * high_delta, weights
                    )
                    within_margin = bool(
                        -STATIC_MARGIN_NLL <= ci_low and ci_high <= STATIC_MARGIN_NLL
                    )
                matched.append(
                    {
                        "family": family,
                        "seed": seed,
                        "target": budget,
                        "test_mean_dropped": measured,
                        "low": low,
                        "high": high,
                        "bracketed": bracketed,
                        "native_minus_interpolated_static_nll": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "margin_nll": STATIC_MARGIN_NLL,
                        "ci_within_margin": within_margin,
                    }
                )
                for name in [
                    "full",
                    f"native{budget}",
                    *[f"fixed{i}" for i in sorted({math.floor(k), math.ceil(k)})],
                    *[f"random{budget}_{d}" for d in DRAWS],
                    *[f"{control}{budget}_{d}" for control in SHUFFLES for d in DRAWS],
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
                            eligible_fraction=row.get("eligible_fraction", ""),
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
                            delta, weights = paired_nll_inputs(native["ppl"], ppl)
                        else:
                            a, b = native["tasks"][metric], row["tasks"][metric]
                            if [x[0] for x in a] != [x[0] for x in b]:
                                raise ValueError(
                                    f"Unpaired items {family}/{seed}/{name}/{metric}"
                                )
                            weights = np.ones(len(a))
                            delta = np.array(
                                [x[-1] - y[-1] for x, y in zip(a, b, strict=True)]
                            )
                        effect, lo, hi = paired_interval(delta, weights)
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
        "stage",
        choices=["freeze", "preflight", "pilot", "run", "report", "replay", "all"],
    )
    p.add_argument(
        "--out", type=Path, default=Path("new/artifacts/tmlr-routing-controls")
    )
    p.add_argument("--family", choices=MODELS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    args = p.parse_args()
    if args.seed and args.stage != "run":
        p.error("--seed applies only to the run stage")
    if args.family and args.stage not in ("preflight", "run"):
        p.error("--family applies only to the preflight and run stages")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    with FileLock(root / ".lock", timeout=0):
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
        if args.stage == "preflight":
            preflight(root, args.family or "tiny")
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
                    f.write(
                        f"{ev.file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
                    )
            print("Tables written to", root)


if __name__ == "__main__":
    main()

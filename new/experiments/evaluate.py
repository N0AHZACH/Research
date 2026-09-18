"""
Unified evaluation framework for Dynamic Layer Routing research.

This is the single entry point for paper-grade evaluation. It consumes a JSON
manifest, evaluates every listed variant with the same protocol, and writes a
standard artifact bundle:

    artifacts/evals/<eval_run_id>/
      manifest.json
      eval_summary.csv
      eval_summary.json
      task_results.json
      perplexity.json
      routing_metrics.csv
      environment.json

Example:

    python experiments/evaluate.py --manifest configs/eval/qwen7b_main.json
    python experiments/evaluate.py --manifest configs/eval/qwen7b_main.json --dry-run

Manifest shape:

{
  "eval_run_id": "qwen7b-main-001",
  "model_family": "qwen7b",
  "model_id": "Qwen/Qwen2.5-7B",
  "output_dir": "artifacts/evals",
  "device": "cuda",
  "dtype": "bfloat16",
  "attention_implementation": "flash_attention_2",
  "tasks": ["mmlu", "gsm8k", "arc_challenge"],
  "num_fewshot": 5,
  "batch_size": 8,
  "limit": null,
  "perplexity": {
    "enabled": true,
    "dataset": "Salesforce/wikitext",
    "config": "wikitext-103-raw-v1",
    "split": "validation",
    "samples": 500,
    "max_length": 512
  },
  "variants": [
    {"name": "base", "type": "base"},
    {"name": "dense_lora", "type": "lora", "checkpoint": "path/to/checkpoint"},
    {"name": "token_dlr_a", "type": "token_dlr", "checkpoint": "path/to/checkpoint",
     "router_weights": "path/to/router_weights.pt", "always_keep_layers": 4}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def import_dlr():
    """Single-source router/metrics/eval-wrappers. Hard dependency: the whole
    point of the unified harness is one definition used in train and eval.
    Imports via new/src (plain `dlr` package) — never `src.dlr`, which collides
    with other projects' `src` packages on sys.path."""
    import pathlib as _pl

    _dlr_src = str(Path(__file__).resolve().parents[1] / "src")
    if _dlr_src not in sys.path:
        sys.path.insert(0, _dlr_src)
    from dlr.eval_wrappers import (  # noqa
        RoutedHFLM,
        check_tasks_routable,
        fixed_stochastic_mask,
    )
    from dlr.routing import RouterMLP  # noqa
    from dlr.routing_metrics import routing_metrics_from_gates as canonical_routing_metrics  # noqa

    return RouterMLP, canonical_routing_metrics, fixed_stochastic_mask, RoutedHFLM, check_tasks_routable


REQUIRED_TOP_LEVEL = [
    "eval_run_id",
    "model_family",
    "model_id",
    "tasks",
    "num_fewshot",
    "batch_size",
    "variants",
]

REQUIRED_VARIANT_FIELDS = {
    "base": ["name", "type"],
    "base_stochastic": ["name", "type", "skip_prob", "eval_seed", "always_keep_layers"],
    "lora": ["name", "type", "checkpoint"],
    "dense_lora": ["name", "type", "checkpoint"],
    "stochastic": ["name", "type", "checkpoint"],
    "token_dlr": ["name", "type", "checkpoint", "router_weights", "always_keep_layers"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified NeurIPS-level evaluator")
    parser.add_argument("--manifest", required=True, help="Evaluation manifest JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and exit")
    parser.add_argument("--skip-lm-eval", action="store_true", help="Run perplexity/routing only")
    parser.add_argument("--skip-perplexity", action="store_true", help="Run lm-eval only")
    parser.add_argument("--output-dir", default=None, help="Override manifest output_dir")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def validate_manifest(manifest: Dict[str, Any], manifest_path: Path) -> List[str]:
    errors: List[str] = []
    for field in REQUIRED_TOP_LEVEL:
        if field not in manifest:
            errors.append(f"Missing top-level field: {field}")

    variants = manifest.get("variants", [])
    if not isinstance(variants, list) or not variants:
        errors.append("`variants` must be a non-empty list")
        return errors

    names = set()
    for idx, variant in enumerate(variants):
        vtype = variant.get("type")
        name = variant.get("name", f"<variant {idx}>")
        if name in names:
            errors.append(f"Duplicate variant name: {name}")
        names.add(name)
        if vtype not in REQUIRED_VARIANT_FIELDS:
            errors.append(f"{name}: unknown variant type `{vtype}`")
            continue
        for field in REQUIRED_VARIANT_FIELDS[vtype]:
            if field not in variant:
                errors.append(f"{name}: missing field `{field}`")
        if "checkpoint" in variant and not Path(variant["checkpoint"]).exists():
            errors.append(f"{name}: checkpoint does not exist: {variant['checkpoint']}")
        if vtype == "token_dlr":
            router_path = Path(variant.get("router_weights", ""))
            if not router_path.exists():
                errors.append(f"{name}: router_weights does not exist: {router_path}")
            if int(variant.get("always_keep_layers", 0)) <= 0:
                errors.append(f"{name}: always_keep_layers must be positive")

    if "perplexity" in manifest:
        ppl = manifest["perplexity"]
        for field in ["dataset", "split", "samples", "max_length"]:
            if field not in ppl:
                errors.append(f"perplexity: missing field `{field}`")

    if not manifest_path.exists():
        errors.append(f"Manifest path does not exist: {manifest_path}")
    return errors


def import_torch_stack():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, nn, F, load_dataset, PeftModel, DataLoader, AutoModelForCausalLM, AutoTokenizer


def dtype_from_name(torch, name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def env_snapshot(manifest: Dict[str, Any]) -> Dict[str, Any]:
    snap = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "manifest_sha256": manifest.get("_manifest_sha256"),
    }
    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["cuda_device_count"] = torch.cuda.device_count()
            snap["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        snap["torch_error"] = str(exc)
    try:
        import transformers

        snap["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import lm_eval

        snap["lm_eval_version"] = lm_eval.__version__
    except Exception:
        snap["lm_eval_version"] = "not_installed"
    return snap


@dataclass
class LoadedVariant:
    name: str
    type: str
    model: Any
    tokenizer: Any
    router: Any = None
    always_keep_layers: Optional[int] = None


def make_router_class(torch, nn, F):
    # Legacy shim — use src.dlr.routing.RouterMLP (identical arch) instead.
    # Kept so old configs/logs referencing the name still import.
    RouterMLP, _, _, _, _ = import_dlr()
    return RouterMLP


def model_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not find decoder layers on model")


def load_variant(manifest: Dict[str, Any], variant: Dict[str, Any]) -> LoadedVariant:
    torch, nn, F, _, PeftModel, _, AutoModelForCausalLM, AutoTokenizer = import_torch_stack()
    device = manifest.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(torch, manifest.get("dtype", "bfloat16"))
    model_id = manifest["model_id"]
    attn_impl = manifest.get("attention_implementation")
    token = os.environ.get("HF_TOKEN")
    quant = manifest.get("quantization", "none")
    trust_code = bool(manifest.get("trust_remote_code", False))

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if token:
        kwargs["token"] = token
    if trust_code:
        kwargs["trust_remote_code"] = True
        # Compat shim (documented caveat): FlexiDepth's pinned transformers
        # (early-2025, 4.x) had ROPE_INIT_FUNCTIONS["default"] = standard RoPE.
        # Local transformers (>=5.x) dropped the "default" key; their modeling
        # file falls back to rope_type="default" when config has no
        # rope_scaling (true for Llama-3-8B). Register the standard init so
        # their code loads; numbers must be sanity-checked vs their paper.
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            if "default" not in ROPE_INIT_FUNCTIONS:
                def _default_rope_init(config, device=None):
                    import torch

                    dim = getattr(config, "head_dim", None) or (
                        config.hidden_size // config.num_attention_heads
                    )
                    theta = getattr(config, "rope_theta", 10000.0)
                    inv_freq = 1.0 / (
                        theta
                        ** (
                            torch.arange(0, dim, 2, dtype=torch.float32).to(device)
                            / dim
                        )
                    )
                    return inv_freq, 1.0

                ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        except Exception:
            pass
        # Class shim: transformers>=5 re-inits "*RotaryEmbedding" modules via
        # module.compute_default_rope_parameters when rope_type=="default".
        # FlexiDepth's 4.x-era DDLlamaRotaryEmbedding lacks it. Bind the same
        # standard init. Pre-import their remote module so from_pretrained
        # reuses the patched class (dynamic modules are cached in sys.modules).
        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            _rot_cls = get_class_from_dynamic_module(
                "modeling_ddllama.DDLlamaRotaryEmbedding",
                model_id,
                trust_remote_code=True,
                token=token,
            )
            if not hasattr(_rot_cls, "compute_default_rope_parameters"):
                def _compute_default_rope_parameters(self, config=None):
                    return _default_rope_init(config if config is not None else self.config)

                _rot_cls.compute_default_rope_parameters = (
                    _compute_default_rope_parameters
                )
        except Exception:
            pass
    if quant == "4bit":
        # 8B-class audit path on 8GB GPUs (e.g. FlexiDepth-Llama-3-8B).
        # Proven on TinyLlama-1.1B; 8B needs batch_size<=2, limit<=100.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    print(f"Loading {variant['name']} ({variant['type']})")
    dispatched = kwargs.get("device_map") == "auto"
    if variant["type"] in ("base", "base_stochastic"):
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not dispatched:
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token,
                                                  trust_remote_code=trust_code)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        out = LoadedVariant(variant["name"], variant["type"], model, tokenizer)
        if variant["type"] == "base_stochastic":
            # Random-mask control on unadapted weights (e.g. FlexiDepth audit:
            # their learned routing vs coin-flip on the SAME checkpoint).
            ak = int(variant.get("always_keep_layers", 0))
            if ak <= 0:
                raise RuntimeError(f"{variant['name']}: base_stochastic needs always_keep_layers")
            out.type = "stochastic"
            out.skip_prob = float(variant.get("skip_prob", 0.3))
            out.eval_seed = int(variant.get("eval_seed", 1234))
            out.always_keep_layers = ak
        return out

    base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not dispatched:
        base = base.to(device)
    peft_model = PeftModel.from_pretrained(base, variant["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(variant["checkpoint"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if variant["type"] == "token_dlr":
        layers = model_layers(peft_model)
        always_keep = int(variant["always_keep_layers"])
        routable = len(layers) - always_keep
        if routable <= 0:
            raise ValueError(f"{variant['name']}: no routable layers with always_keep={always_keep}")
        Router = make_router_class(torch, nn, F)
        router = Router(peft_model.config.hidden_size, routable).to(device)
        router.load_state_dict(torch.load(variant["router_weights"], map_location=device))
        router.eval()
        peft_model.router = router
        return LoadedVariant(variant["name"], variant["type"], peft_model, tokenizer, router, always_keep)

    model = peft_model.merge_and_unload()
    out = LoadedVariant(variant["name"], variant["type"], model, tokenizer)
    if variant["type"] == "stochastic":
        # Fixed seeded mask at eval (deterministic, documented). Without this,
        # stochastic rows would silently evaluate at full depth.
        out.skip_prob = float(variant.get("skip_prob", 0.3))
        out.eval_seed = int(variant.get("eval_seed", 1234))
        out.always_keep_layers = int(variant.get("always_keep_layers", 0))
    return out


class StopForward(Exception):
    pass


def routed_forward(torch, loaded: LoadedVariant, input_ids, attention_mask=None, labels=None):
    model = loaded.model
    router = loaded.router
    always_keep = int(loaded.always_keep_layers or 0)
    if router is None or always_keep <= 0:
        raise RuntimeError(f"{loaded.name}: routed_forward called without router/always_keep")

    layers = model_layers(model)
    captured: Dict[str, Any] = {}
    handles = []

    def capture_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()
        raise StopForward()

    handle = layers[always_keep - 1].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except StopForward:
        pass
    finally:
        handle.remove()

    if "h" not in captured:
        raise RuntimeError(f"{loaded.name}: router capture hook did not fire")

    with torch.no_grad():
        gates, _ = router(
            captured["h"].to(next(router.parameters()).device), deterministic=True
        )

    if gates.dim() != 3:
        raise RuntimeError(f"{loaded.name}: expected token gates [B,S,L], got {tuple(gates.shape)}")

    for layer_i, layer in enumerate(layers[always_keep:]):
        def gate_hook(module, inputs, output, li=layer_i):
            residual = inputs[0]
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            gate = gates[:, :, li].unsqueeze(-1).to(h.dtype)
            gated_h = gate * h + (1.0 - gate) * residual
            return (gated_h,) + output[1:] if is_tuple else gated_h

        handles.append(layer.register_forward_hook(gate_hook))

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finally:
        for h in handles:
            h.remove()

    return outputs, gates.detach().cpu()


def tokenize_for_ppl(tokenizer, batch: Dict[str, List[str]], max_length: int) -> Dict[str, Any]:
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    labels = [list(ids) for ids in out["input_ids"]]
    for i, mask in enumerate(out["attention_mask"]):
        for j, value in enumerate(mask):
            if value == 0:
                labels[i][j] = -100
    out["labels"] = labels
    return out


def routing_metrics_from_gates(gates_list: List[Any], always_keep: int) -> Dict[str, Any]:
    """Thin adapter over the canonical src.dlr definition (same numbers as
    train_loop). Returns {} when no gates were collected (non-routed rows)."""
    if not gates_list:
        return {}
    import torch

    _, canonical, _, _, _ = import_dlr()
    gates = torch.cat([g.float() for g in gates_list], dim=0)
    return canonical(gates, always_keep)


def evaluate_perplexity(loaded: LoadedVariant, manifest: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch, _, _, load_dataset, _, DataLoader, _, _ = import_torch_stack()
    ppl_cfg = manifest.get("perplexity", {})
    device = manifest.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = ppl_cfg.get("dataset", "Salesforce/wikitext")
    dataset_config = ppl_cfg.get("config", "wikitext-103-raw-v1")
    split = ppl_cfg.get("split", "validation")
    samples = int(ppl_cfg.get("samples", 500))
    max_length = int(ppl_cfg.get("max_length", 512))
    batch_size = int(manifest.get("batch_size", 1))

    print(f"  Perplexity: {dataset_name}/{dataset_config} {split}, samples={samples}")
    raw = load_dataset(dataset_name, dataset_config, split=split)
    raw = raw.filter(lambda x: len(x["text"]) > 100)
    raw = raw.select(range(min(samples, len(raw))))
    ds = raw.map(
        lambda b: tokenize_for_ppl(loaded.tokenizer, b, max_length),
        batched=True,
        remove_columns=raw.column_names,
    )
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    loaded.model.eval()
    losses = []
    token_count = 0
    gates_list = []
    stochastic_mask = None
    started = time.perf_counter()
    _, _, fixed_mask_fn, _, _ = import_dlr()
    mask_ctx = (
        fixed_mask_fn(
            loaded.model,
            int(getattr(loaded, "always_keep_layers", 0) or 0),
            float(getattr(loaded, "skip_prob", 0.3)),
            int(getattr(loaded, "eval_seed", 1234)),
        )
        if loaded.type == "stochastic"
        else None
    )
    if mask_ctx is not None and int(getattr(loaded, "always_keep_layers", 0) or 0) <= 0:
        raise RuntimeError(
            f"{loaded.name}: stochastic eval needs always_keep_layers in the variant spec "
            "(refusing silent full-depth fallback)"
        )
    with torch.no_grad():
        ctx = mask_ctx if mask_ctx is not None else _nullcontext()
        with ctx as maybe_mask:
            stochastic_mask = maybe_mask
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
                token_count += int((batch["labels"] != -100).sum().item())
                if loaded.type == "token_dlr":
                    outputs, gates = routed_forward(
                        torch,
                        loaded,
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )
                    gates_list.append(gates)
                else:
                    outputs = loaded.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )
                    if stochastic_mask is not None:
                        import torch as _t

                        m = _t.tensor(stochastic_mask).view(1, 1, -1).expand(
                            batch["input_ids"].shape[0], 8,
                            len(stochastic_mask),
                        )
                        gates_list.append(m)
                losses.append(float(outputs.loss.item()))
            routing = routing_metrics_from_gates(gates_list, int(loaded.always_keep_layers or 0))
            if loaded.type == "stochastic" and routing:
                # Fixed coin-flip masks have ~0 entropy by construction; the
                # collapse rule only applies to learned routers.
                routing["collapse_flag"] = "n/a (fixed mask)"

    elapsed = max(time.perf_counter() - started, 1e-9)
    avg_loss = sum(losses) / max(1, len(losses))
    ppl_result = {
        "variant": loaded.name,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "samples": len(raw),
        "max_length": max_length,
        "num_tokens": token_count,
        "negative_log_likelihood": avg_loss,
        "perplexity": math.exp(avg_loss),
        "tokens_per_sec": token_count / elapsed,
    }
    return ppl_result, routing


def run_lm_eval(loaded: LoadedVariant, manifest: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError("lm-eval is not installed; use --skip-lm-eval or install lm-eval") from exc

    _, _, fixed_mask_fn, RoutedHFLM, check_tasks_routable = import_dlr()
    device = manifest.get("device", "cuda")
    check_tasks_routable(manifest.get("tasks"), loaded.name, loaded.type)

    if loaded.type == "token_dlr":
        # Loglikelihood tasks only (ARC/HellaSwag/Winogrande/MMLU). generate_until
        # tasks fail fast in check_tasks_routable / _model_generate.
        loaded.tokenizer.model_max_length = 100000
        lm, _gate_acc = RoutedHFLM.build(
            HFLM,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            router=loaded.router,
            always_keep=int(loaded.always_keep_layers or 0),
            batch_size=int(manifest["batch_size"]),
            max_length=int(manifest.get("max_length", 2048)),
            device=device,
        )
        return evaluator.simple_evaluate(
            model=lm,
            tasks=manifest["tasks"],
            num_fewshot=int(manifest["num_fewshot"]),
            limit=manifest.get("limit"),
            log_samples=bool(manifest.get("log_samples", False)),
        )

    if loaded.type == "stochastic":
        ak = int(getattr(loaded, "always_keep_layers", 0) or 0)
        if ak <= 0:
            raise RuntimeError(
                f"{loaded.name}: stochastic eval needs always_keep_layers in the variant spec "
                "(refusing silent full-depth fallback)"
            )
        with fixed_mask_fn(loaded.model, ak, float(getattr(loaded, "skip_prob", 0.3)),
                            int(getattr(loaded, "eval_seed", 1234))):
            return _run_hflm(HFLM, evaluator, loaded, manifest)

    return _run_hflm(HFLM, evaluator, loaded, manifest)


def _run_hflm(HFLM, evaluator, loaded: LoadedVariant, manifest: Dict[str, Any]) -> Dict[str, Any]:
    loaded.tokenizer.model_max_length = 100000
    lm = HFLM(
        pretrained=loaded.model,
        tokenizer=loaded.tokenizer,
        batch_size=int(manifest["batch_size"]),
        max_length=int(manifest.get("max_length", 2048)),
        truncation=True,
    )
    return evaluator.simple_evaluate(
        model=lm,
        tasks=manifest["tasks"],
        num_fewshot=int(manifest["num_fewshot"]),
        limit=manifest.get("limit"),
        log_samples=bool(manifest.get("log_samples", False)),
    )


def flatten_task_rows(
    eval_run_id: str,
    model_family: str,
    variant: str,
    task_results: Dict[str, Any],
    manifest_path: Path,
    manifest_hash: str,
) -> List[Dict[str, Any]]:
    rows = []
    results = task_results.get("results", {})
    for task, metrics in results.items():
        for metric, value in metrics.items():
            if metric.startswith("alias") or not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "run_id": eval_run_id,
                    "model_family": model_family,
                    "variant": variant,
                    "seed": "",
                    "checkpoint_id": "",
                    "task": task,
                    "shot_count": "",
                    "metric": metric,
                    "value": value,
                    "stderr": metrics.get(f"{metric}_stderr", ""),
                    "num_examples": "",
                    "eval_harness": "lm-eval",
                    "eval_harness_version": "",
                    "git_commit": git_commit(),
                    "config_hash": manifest_hash,
                    "manifest_path": str(manifest_path),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_loaded(loaded: LoadedVariant) -> None:
    try:
        import gc
        import torch

        del loaded.model
        if loaded.router is not None:
            del loaded.router
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    manifest["_manifest_sha256"] = file_sha256(manifest_path)

    errors = validate_manifest(manifest, manifest_path)
    if errors:
        print("Manifest validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Manifest valid.")
        print(f"Variants: {', '.join(v['name'] for v in manifest['variants'])}")
        return 0

    output_root = Path(args.output_dir or manifest.get("output_dir", "artifacts/evals"))
    eval_run_id = manifest["eval_run_id"]
    out_dir = (REPO_ROOT / output_root / eval_run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, out_dir / "manifest.json")
    write_json(out_dir / "environment.json", env_snapshot(manifest))

    all_task_results: Dict[str, Any] = {}
    perplexity_results: Dict[str, Any] = {}
    routing_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for variant in manifest["variants"]:
        loaded = load_variant(manifest, variant)
        try:
            if not args.skip_perplexity and manifest.get("perplexity", {}).get("enabled", True):
                ppl, routing = evaluate_perplexity(loaded, manifest)
                perplexity_results[loaded.name] = ppl
                summary_rows.append(
                    {
                        "run_id": eval_run_id,
                        "model_family": manifest["model_family"],
                        "variant": loaded.name,
                        "task": "perplexity",
                        "metric": "perplexity",
                        "value": ppl["perplexity"],
                        "stderr": "",
                        "num_examples": ppl["samples"],
                        "eval_harness": "internal",
                        "git_commit": git_commit(),
                        "config_hash": manifest["_manifest_sha256"],
                        "manifest_path": str(manifest_path),
                    }
                )
                if routing:
                    routing_rows.append({"run_id": eval_run_id, "variant": loaded.name, **routing})

            if not args.skip_lm_eval and manifest.get("tasks"):
                task_result = run_lm_eval(loaded, manifest)
                all_task_results[loaded.name] = task_result
                summary_rows.extend(
                    flatten_task_rows(
                        eval_run_id,
                        manifest["model_family"],
                        loaded.name,
                        task_result,
                        manifest_path,
                        manifest["_manifest_sha256"],
                    )
                )
        finally:
            cleanup_loaded(loaded)

        write_json(out_dir / "perplexity.json", perplexity_results)
        write_json(out_dir / "task_results.json", all_task_results)
        write_csv(out_dir / "routing_metrics.csv", routing_rows)
        write_csv(out_dir / "eval_summary.csv", summary_rows)

    write_json(
        out_dir / "eval_summary.json",
        {
            "eval_run_id": eval_run_id,
            "model_family": manifest["model_family"],
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["_manifest_sha256"],
            "results_dir": str(out_dir),
            "variants": [v["name"] for v in manifest["variants"]],
            "summary_rows": len(summary_rows),
        },
    )
    print(f"Evaluation complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

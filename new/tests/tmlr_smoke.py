"""Offline numerical smoke test using tiny random Llama and Qwen2 models."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from new.experiments import evaluate as ev
from new.experiments.tmlr_pilot import check_forward, shuffled_routes
from new.experiments.tmlr_study import verify_shuffle


def exercise(folder, config, cls):
    router_cls, _, _, hf_router, _ = ev.import_dlr()
    from dlr.data import PackedTextDataset, collate
    from dlr.models import wrap_lora
    from dlr.train_loop import train
    from lm_eval.models.huggingface import HFLM

    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2, "one": 3, "two": 4, "three": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = cls(
        config(
            vocab_size=16,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=48,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(folder / "base")
    model = wrap_lora(model, rank=2, expanded=True)
    router = router_cls(32, 2)
    packed = PackedTextDataset([[3, 4, 5, 4], [4, 3, 5, 4]])
    for method in ["stochastic_30", "token_dlr_50"]:
        cfg = SimpleNamespace(
            variant_type=method,
            always_keep=2,
            device="cpu",
            batch_size=2,
            lr=1e-3,
            max_steps=1,
            warmup_steps=0,
            target_skip=0.5,
            fixed_mask=[0, 1],
            kd_weight=0.2,
            skip_weight=1,
            l1_weight=0.01,
            kd_temp=2,
            tau=1,
            log_every=1,
        )
        train(
            model,
            router if method.startswith("token") else None,
            packed,
            packed,
            cfg,
            collate,
        )
    model.save_pretrained(folder / "adapter")
    tok.save_pretrained(folder / "adapter")
    torch.save(router.state_dict(), folder / "router.pt")
    raw = Dataset.from_dict({"text": ["one two " * 16, "three " * 18, "one " * 28]})
    cfg = {
        "model_id": str(folder / "base"),
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 2,
        "log_samples": True,
        "perplexity": {"samples": 3, "max_length": 32},
    }
    spec = {
        "name": "dlr",
        "type": "token_dlr",
        "always_keep_layers": 2,
        "checkpoint": str(folder / "adapter"),
        "router_weights": str(folder / "router.pt"),
    }
    loaded = ev.load_variant(cfg, spec)
    check_forward(torch, loaded)
    with patch("datasets.load_dataset", return_value=raw):
        p, r = ev.evaluate_perplexity(loaded, cfg)
        q, s = ev.evaluate_perplexity(loaded, cfg, shuffled_routes(torch, 11))
        verify_shuffle({"routing": r}, {"routing": s})
        assert len(p["examples"]) == len(q["examples"]) == 3
        assert (
            abs(
                sum(x[2] for x in p["examples"]) / p["num_tokens"]
                - p["negative_log_likelihood"]
            )
            < 1e-5
        )
        fixed = ev.load_variant(cfg, dict(spec, type="stochastic", keep_mask=[0, 1]))
        _, r = ev.evaluate_perplexity(fixed, cfg)
        assert r["per_layer_activity"] == [0, 1]
    lm, acc = hf_router.build(HFLM, loaded.model, tok, loaded.router, 2, 1, 128, "cpu")
    assert lm._model_call(torch.tensor([[3, 4, 5]])).shape == (1, 3, 16)
    assert len(acc.summary(2)["per_layer_activity"]) == 2
    from datasets import DatasetDict
    from lm_eval.api.task import ConfigurableTask

    task = ConfigurableTask(
        config={
            "task": "arc_challenge",
            "dataset_path": "offline",
            "test_split": "test",
            "custom_dataset": lambda **kw: DatasetDict(
                test=Dataset.from_dict(
                    {
                        "question": ["one", "two"],
                        "choices": [["one", "two"], ["two", "one"]],
                        "answer": [0, 1],
                    }
                )
            ),
            "output_type": "multiple_choice",
            "doc_to_text": "question",
            "doc_to_choice": "choices",
            "doc_to_target": "answer",
            "metric_list": [{"metric": "acc_norm"}],
        }
    )
    protocol = dict(cfg, tasks=[task], num_fewshot=0, max_length=32, batch_size=1)
    for variant in [loaded, fixed]:
        samples = ev.run_lm_eval(variant, protocol)["samples"]["arc_challenge"]
        assert len(samples) == 2 and all(
            s["doc_hash"] and s["acc_norm"] in (0, 1) for s in samples
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    gates = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    mask = torch.tensor([[1, 1], [1, 0]])
    transform = shuffled_routes(torch, 7)
    moved = transform(gates, mask)
    assert torch.equal(moved[:, 0], gates[:, 0].flip(0))
    assert torch.equal(moved[:, 1], gates[:, 1])
    assert (transform.changed, transform.total) == (2, 3)
    for config, cls in [
        (LlamaConfig, LlamaForCausalLM),
        (Qwen2Config, Qwen2ForCausalLM),
    ]:
        with tempfile.TemporaryDirectory() as folder:
            exercise(Path(folder), config, cls)
    print(
        "PASS: static training, learned training, reload, masking, shuffle, perplexity and task forward on both families"
    )

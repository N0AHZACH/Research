"""WikiText packing for min prototype. Train + held-out eval split."""
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    split: str = "train"
    max_length: int = 128
    num_lines: Optional[int] = 300
    min_char_len: int = 100
    eval_blocks: int = 50


class PackedTextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def load_packed_ids(tokenizer, cfg: DataConfig):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    texts = [t for t in ds["text"] if len(t) > cfg.min_char_len]
    texts = texts[: cfg.num_lines]
    if not texts:
        raise ValueError("No text survived filtering.")
    enc = tokenizer("\n\n".join(texts), return_tensors=None)["input_ids"]
    block = cfg.max_length
    examples = [enc[i:i + block] for i in range(0, len(enc) - block, block)]
    if not examples:
        raise ValueError(f"No packed examples from {len(enc)} tokens, block={block}.")
    return examples


def build_train_eval(tokenizer, cfg: DataConfig):
    """Single download, then split: last eval_blocks are held-out."""
    all_ids = load_packed_ids(tokenizer, cfg)
    n_eval = min(cfg.eval_blocks, max(1, len(all_ids) // 4))
    train_ids = all_ids[:-n_eval] if n_eval > 0 else all_ids
    eval_ids = all_ids[-n_eval:] if n_eval > 0 else all_ids[:1]
    return PackedTextDataset(train_ids), PackedTextDataset(eval_ids)


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

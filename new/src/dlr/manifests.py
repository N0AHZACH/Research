"""Run manifest: one schema for every variant."""
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    git_commit: str
    command: str
    config: dict
    model_name: str
    seed: int
    hardware: dict
    precision: str
    trainable_params: int
    total_params: int
    always_keep: int
    total_layers: int
    target_skip: Optional[float] = None
    measured_active_layers: Optional[float] = None
    final_ppl: Optional[float] = None
    wall_clock_s: Optional[float] = None

    def write(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

"""Seed + env snapshot. One seed function used by every entry point."""
import os
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_git_commit(repo_root: str = ".") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    except Exception:
        return "unknown"


def env_snapshot() -> dict:
    import accelerate
    import datasets
    import peft
    import transformers

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info

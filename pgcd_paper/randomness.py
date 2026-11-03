"""Central random-state control for repeatable experiments."""
from __future__ import annotations

import os, random
import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> dict[str, object]:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    result: dict[str, object] = {"seed": seed, "python": True, "numpy": True, "torch": False,
                                "deterministic_requested": deterministic}
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        result["torch"] = True
    except ImportError:
        result["torch"] = "unavailable"
    return result

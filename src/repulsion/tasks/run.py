"""Run-level helpers shared by task scripts (seeding, output dir, saving)."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and torch; return a NumPy generator for stimulus sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def resolve_output_dir(output_dir: str | None) -> str:
    """*output_dir* if given, else the current Hydra run directory."""
    if output_dir is None:
        from hydra.core.hydra_config import HydraConfig

        output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    return str(output_dir)


def save_results(output_dir: str, filename: str = "results.npz", **arrays) -> str:
    path = os.path.join(output_dir, filename)
    np.savez_compressed(path, **arrays)
    log.info("Saved %s", path)
    return path

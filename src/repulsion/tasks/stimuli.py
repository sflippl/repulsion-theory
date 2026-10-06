"""Stimulus helpers for task scripts."""
from __future__ import annotations

import numpy as np


def correlated_pairs(correlations, dim: int = 128, rng=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """One unit-norm, zero-mean pair ``(x, y)`` with ``corr(x, y) = rho`` per *rho*."""
    rng = np.random.default_rng(rng)
    pairs = []

    for rho in correlations:
        x = rng.normal(size=dim)
        x -= x.mean()
        x /= np.linalg.norm(x)

        z = rng.normal(size=dim)
        z -= z.mean()
        z -= x * (x @ z)
        z /= np.linalg.norm(z)

        y = rho * x + np.sqrt(1 - rho**2) * z
        pairs.append((x, y))

    return pairs


def superpose(a: np.ndarray, b: np.ndarray, phi: float, magnitude: float = 1.0) -> np.ndarray:
    """Salience-weighted superposition ``magnitude * (sqrt(phi) a + sqrt(1 - phi) b)``."""
    if not 0.0 <= phi <= 1.0:
        raise ValueError(f"phi must be in [0, 1]; got {phi}.")
    return magnitude * (np.sqrt(phi) * a + np.sqrt(1.0 - phi) * b)

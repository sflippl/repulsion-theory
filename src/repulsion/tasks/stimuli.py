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


def superpose_slots(a: np.ndarray, b: np.ndarray, phi: float, magnitude: float = 1.0) -> np.ndarray:
    """The two terms of :func:`superpose` side by side, ``[sqrt(phi) a, sqrt(1 - phi) b] * magnitude``.

    For models built with ``n_input_slots=2``, whose streams sum their
    (projected) slots instead of seeing the superposition directly.
    """
    return np.concatenate([
        superpose(a, np.zeros_like(b), phi, magnitude),
        superpose(np.zeros_like(a), b, phi, magnitude),
    ], axis=-1)


def correlated_pairs_exact(correlations, dim: int = 128, rng=None, baseline: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs with exactly ``corr(x_i, y_i) = correlations[i]`` and *baseline* between pairs.

    Builds the item Gram matrix (items ordered ``x_0, y_0, x_1, y_1, ...``),
    factorizes it, and embeds it in a random centered subspace, so all
    vectors are zero-mean and unit-norm (correlation = cosine similarity).
    """
    rng = np.random.default_rng(rng)
    correlations = np.asarray(correlations, dtype=float)
    if correlations.ndim != 1:
        raise ValueError("`correlations` must be a 1-D array.")
    n_pairs = len(correlations)
    n_items = 2 * n_pairs
    if dim < n_items + 1:
        raise ValueError(f"`dim` must be at least {n_items + 1} to generate centered vectors.")
    if not -1 <= baseline <= 1:
        raise ValueError("`baseline` must be between -1 and 1.")
    if np.any((correlations < -1) | (correlations > 1)):
        raise ValueError("All pair correlations must be between -1 and 1.")
    if n_items == 0:
        return []

    gram = np.full((n_items, n_items), baseline, dtype=float)
    np.fill_diagonal(gram, 1.0)
    for i, c in enumerate(correlations):
        gram[2 * i, 2 * i + 1] = gram[2 * i + 1, 2 * i] = c

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-10:
        raise ValueError("The requested correlations do not form a positive-semidefinite correlation matrix.")
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))

    # Orthonormal basis of the zero-mean subspace (first QR column ∝ all-ones).
    Q, _ = np.linalg.qr(np.concatenate([np.ones((dim, 1)), rng.normal(size=(dim, n_items))], axis=1))
    centered_basis = Q[:, 1:n_items + 1]
    random_rotation, _ = np.linalg.qr(rng.normal(size=(n_items, n_items)))
    vectors = factor @ (random_rotation @ centered_basis.T)

    vectors -= vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A zero vector was produced by the requested correlation structure.")
    vectors /= norms
    return [(vectors[2 * i], vectors[2 * i + 1]) for i in range(n_pairs)]

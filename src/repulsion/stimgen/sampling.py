"""Correlation matrix construction and sampling utilities.

Two generation modes are supported:

``"sampled"``
    Samples from a multivariate Normal whose covariance equals the target
    correlation matrix C. Uses eigendecomposition of C. Each returned vector
    has expected L2 norm ≈ 1 (before magnitude scaling).

``"exact"``
    Constructs deterministic vectors whose normalised Gram matrix equals C
    exactly (up to floating-point precision), provided dim >= rank(C).
    Each returned vector has L2 norm exactly 1 (before magnitude scaling).

Both modes call :func:`_factorize` which performs a hard PSD check: if the
smallest eigenvalue of C is below ``-psd_eps``, a ``ValueError`` is raised.
There is no silent eigenvalue clipping.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.linalg import eigh

from repulsion.stimgen.spec import ItemSpec


# ---------------------------------------------------------------------------
# PSD factorisation (shared by both sampling modes)
# ---------------------------------------------------------------------------

def _factorize(C: np.ndarray, psd_eps: float) -> np.ndarray:
    """Eigendecompose C and return factor L satisfying L @ L.T == C.

    The factor has shape (N, rank) where rank is the number of eigenvalues
    above ``psd_eps``.

    Raises:
        ValueError: If the smallest eigenvalue is below ``-psd_eps``, meaning
            C is not positive semi-definite and the correlation parameters are
            invalid.
    """
    eigenvalues, eigenvectors = eigh(C)

    min_ev = float(eigenvalues[0])
    if min_ev < -psd_eps:
        raise ValueError(
            f"Correlation matrix is not positive semi-definite "
            f"(smallest eigenvalue = {min_ev:.6g}). "
            "The requested combination of correlation parameters is invalid. "
            "Adjust corr_within, corr_between, or the number of groups/items."
        )

    # Zero out numerically-tiny negatives that passed the threshold, then
    # keep only the positive part of the spectrum.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    mask = eigenvalues > psd_eps
    L = eigenvectors[:, mask] * np.sqrt(eigenvalues[mask])[np.newaxis, :]
    return L  # shape (N, rank)


# ---------------------------------------------------------------------------
# Correlation matrix construction
# ---------------------------------------------------------------------------

def build_item_corr_matrix(item_spec: ItemSpec) -> np.ndarray:
    """Build the joint correlation matrix for all items of one item type.

    Items are ordered: for each subgroup (in spec order), for each group
    (1 … n_groups), for each item within the group (1 … n_items).

    Entry C[i, j] is determined by the correlation level of the pair (i, j):

    +---------------------------------+----------------------------------+
    | Pair relationship               | Correlation                      |
    +=================================+==================================+
    | i == j (diagonal)              | 1.0                              |
    +---------------------------------+----------------------------------+
    | same subgroup, same group       | SubgroupSpec.corr_within         |
    +---------------------------------+----------------------------------+
    | same subgroup, different group  | SubgroupSpec.corr_between        |
    +---------------------------------+----------------------------------+
    | different subgroup              | ItemSpec.corr_between            |
    +---------------------------------+----------------------------------+

    Args:
        item_spec: Fully-resolved item specification.

    Returns:
        Symmetric (N, N) float64 array with ones on the diagonal.
    """
    N = sum(sg.n_groups * sg.n_items for sg in item_spec.subgroups)

    # Label every row with (subgroup index, local group index)
    subgroup_of = np.empty(N, dtype=np.int32)
    group_of = np.empty(N, dtype=np.int32)

    row = 0
    for sg_idx, sg in enumerate(item_spec.subgroups):
        for g in range(sg.n_groups):
            for _ in range(sg.n_items):
                subgroup_of[row] = sg_idx
                group_of[row] = g
                row += 1

    # Fill correlation matrix using the hierarchy above
    C = np.eye(N, dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            sgi = int(subgroup_of[i])
            sgj = int(subgroup_of[j])
            if sgi != sgj:
                val = item_spec.corr_between
            else:
                sg = item_spec.subgroups[sgi]
                if group_of[i] == group_of[j]:
                    val = sg.corr_within
                else:
                    val = sg.corr_between
            C[i, j] = val
            C[j, i] = val

    return C


# ---------------------------------------------------------------------------
# Sampling mode
# ---------------------------------------------------------------------------

def sample_from_corr_matrix(
    C: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    psd_eps: float = 1e-8,
) -> np.ndarray:
    """Sample (N, dim) vectors from a multivariate Normal with correlation C.

    Each row has expected squared L2 norm of 1 (C has unit diagonal), so
    the expected L2 norm is ≈ 1.

    The expected cosine similarity between rows i and j approximates C[i, j].

    Args:
        C: (N, N) correlation matrix (symmetric, unit diagonal).
        dim: Dimensionality of each sampled vector.
        rng: NumPy random generator.
        psd_eps: Eigenvalue threshold; eigenvalues below ``-psd_eps`` trigger
            a hard error.

    Returns:
        Array of shape (N, dim).

    Raises:
        ValueError: If C is not positive semi-definite.
    """
    L = _factorize(C, psd_eps)            # (N, rank)
    z = rng.standard_normal((L.shape[1], dim)) / math.sqrt(dim)  # (rank, dim)
    return L @ z                           # (N, dim), each row has E[‖·‖²] = 1


# ---------------------------------------------------------------------------
# Exact (deterministic) mode
# ---------------------------------------------------------------------------

def _deterministic_orthonormal_rows(rank: int, dim: int) -> np.ndarray:
    """Return a (rank, dim) matrix with orthonormal rows.

    Uses a deterministic trigonometric basis followed by QR to guarantee
    orthonormality. The same inputs always produce the same matrix.
    """
    if rank == 0:
        return np.zeros((0, dim), dtype=np.float64)

    d = np.arange(1, dim + 1, dtype=np.float64)[:, np.newaxis]   # (dim, 1)
    r = np.arange(1, rank + 1, dtype=np.float64)[np.newaxis, :]  # (1, rank)
    base = np.cos(np.pi * d * r / (dim + 1.0)) + np.sin(np.pi * d * r / (dim + 1.0))
    # base: (dim, rank) — full-rank for dim >= rank in practice
    q, _ = np.linalg.qr(base)   # q: (dim, rank) orthonormal columns
    return q.T                    # (rank, dim) orthonormal rows


def exact_from_corr_matrix(
    C: np.ndarray,
    dim: int,
    psd_eps: float = 1e-8,
) -> np.ndarray:
    """Construct deterministic vectors whose normalised Gram matrix equals C.

    The returned array X satisfies X @ X.T == C exactly (up to floating-point
    precision) and each row has L2 norm exactly 1.

    Args:
        C: (N, N) correlation matrix.
        dim: Dimensionality of each vector. Must satisfy dim >= rank(C).
        psd_eps: Eigenvalue threshold for the PSD check.

    Returns:
        Array of shape (N, dim).

    Raises:
        ValueError: If C is not positive semi-definite, or if dim < rank(C).
    """
    L = _factorize(C, psd_eps)   # (N, rank)
    rank = L.shape[1]
    if dim < rank:
        raise ValueError(
            f"Exact generation requires dim >= rank(C). "
            f"Got dim={dim}, rank(C)={rank}. "
            "Increase dim or switch to generation_mode='sampled'."
        )
    U = _deterministic_orthonormal_rows(rank, dim)  # (rank, dim)
    # X = L @ U  satisfies  X @ X.T = L @ U @ U.T @ L.T = L @ L.T = C
    # and ‖X[i,:]‖² = C[i,i] = 1
    return L @ U  # (N, dim)


# ---------------------------------------------------------------------------
# Circular Gaussian bump generation
# ---------------------------------------------------------------------------

def generate_circular_items(
    item_spec: ItemSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stimulus vectors using a circular Gaussian bump encoding.

    Each group and subgroup gets a contiguous slice of
    ``dim // (n_subgroups × n_groups)`` dimensions.  The slice for subgroup
    ``sg_idx``, group ``g_idx`` starts at index
    ``(sg_idx × n_groups + g_idx) × dims_per_partition``.
    Dimension ``k`` within a partition's slice represents angle
    ``360° × k / dims_per_partition``.  Item *i* (0-indexed) within a group is
    placed at angle ``θ₀ + i × distance``, where ``θ₀ ~ Uniform(0°, 360°)``
    is drawn independently per (subgroup, group) pair.  The bump value at
    dimension ``k`` is::

        exp(−circ_dist(item_angle, dim_angle)² / (2 × bump_width²))

    where ``circ_dist`` is the shorter arc on a 360° circle.  Each vector is
    L2-normalised after the bump is applied.

    Items from different groups or different subgroups occupy disjoint
    dimension slices and are therefore orthogonal.

    Args:
        item_spec: Fully-resolved circular item specification.
        rng: NumPy random generator for the per-group starting angles.

    Returns:
        Array of shape ``(N, dim)`` in the same row order as
        :func:`build_item_corr_matrix` (subgroups → groups → items).
        Each row is L2-normalised.
    """
    assert item_spec.stimulus_type == "circular"

    n_subgroups = len(item_spec.subgroups)
    n_groups = item_spec.subgroups[0].n_groups  # all subgroups share n_groups
    dims_per_partition = item_spec.dim // (n_subgroups * n_groups)

    # One random starting angle per (subgroup, group) partition
    start_angles = rng.uniform(0.0, 360.0, size=(n_subgroups, n_groups))

    # Angles represented by each dimension within a partition's slice
    dim_angles = np.arange(dims_per_partition, dtype=np.float64) * (360.0 / dims_per_partition)

    N = sum(sg.n_groups * sg.n_items for sg in item_spec.subgroups)
    vectors = np.zeros((N, item_spec.dim), dtype=np.float64)

    row = 0
    for sg_idx, sg in enumerate(item_spec.subgroups):
        distance = sg.distance
        bump_width = sg.bump_width
        for g_idx in range(sg.n_groups):
            theta0 = start_angles[sg_idx, g_idx]
            partition_start = (sg_idx * n_groups + g_idx) * dims_per_partition
            for i_idx in range(sg.n_items):
                item_angle = (theta0 + i_idx * distance) % 360.0
                # Circular distance between item angle and each dimension's angle
                delta = np.abs(item_angle - dim_angles)
                circ_dist = np.minimum(delta, 360.0 - delta)  # (dims_per_partition,)
                bump = np.exp(-0.5 * (circ_dist / bump_width) ** 2)
                # Place bump in the partition's slice; rest stays zero
                vec = np.zeros(item_spec.dim, dtype=np.float64)
                vec[partition_start: partition_start + dims_per_partition] = bump
                # L2-normalise
                norm = np.linalg.norm(vec)
                if norm > 0.0:
                    vec /= norm
                vectors[row] = vec
                row += 1

    return vectors

"""Group-wise Gram matrices, their closed-form eigendecomposition, and exact
vector construction (Section 1 of the multi-stream repulsion spec).

Fix G groups of M items each (N = G*M total items). A "group-constant"
matrix has a1 on the diagonal, a2 for same-group off-diagonal entries, and
a3 for cross-group entries. Its eigenvalues (lam1, lam2, lam3, with
multiplicities 1, G-1, (M-1)*G) are given in closed form by `group_eigs`, and
the same eigenvectors diagonalize *every* group-constant matrix for a given
(G, M) -- independent of a1, a2, a3. This is what lets different quantities
(K_X, K_Y, cross terms, representation Gram matrices) be compared
mode-by-mode.
"""
from __future__ import annotations

import numpy as np


def group_eigs(a1: float, a2: float, a3: float, G: int, M: int) -> tuple[float, float, float]:
    """Closed-form eigenvalues (lam1, lam2, lam3) of a group-constant matrix.

    Multiplicities are (1, G-1, (M-1)*G) respectively.
    """
    lam1 = (a1 - a2) + (a2 - a3) * M + a3 * G * M
    lam2 = (a1 - a2) + (a2 - a3) * M
    lam3 = a1 - a2
    return lam1, lam2, lam3


def eigs_to_a(lam1: float, lam2: float, lam3: float, G: int, M: int) -> tuple[float, float, float]:
    """Inverse of `group_eigs`: recover (a1, a2, a3) from the three eigenvalues."""
    a3 = (lam1 - lam2) / (G * M)
    a2 = a3 + (lam2 - lam3) / M
    a1 = a2 + lam3
    return a1, a2, a3


def modes_to_a(lam12: float, lam3: float, M: int) -> tuple[float, float, float]:
    """(a1, a2, a3) for the always-a3=0 two-mode case (Section 1.4).

    Applies whenever cross-group correlation is exactly 0 (so lam1 == lam2 ==
    lam12), which holds throughout this spec (A/B orthogonal subspaces, kWTA
    coupling preserves cross-group 0).
    """
    a3 = 0.0
    a2 = (lam12 - lam3) / M
    a1 = a2 + lam3
    return a1, a2, a3


def block_constant_matrix(a1: float, a2: float, a3: float, G: int, M: int) -> np.ndarray:
    """Explicit (N, N) group-constant matrix (a1 diag, a2 same-group, a3 cross-group)."""
    group_id = np.repeat(np.arange(G), M)
    same_group = group_id[:, None] == group_id[None, :]
    mat = np.where(same_group, a2, a3).astype(np.float64)
    np.fill_diagonal(mat, a1)
    return mat


def _exact_factor(C: np.ndarray, psd_eps: float = 1e-10) -> np.ndarray:
    """vecs (N, N) s.t. vecs @ vecs.T == C, clipping negative eigenvalues to 0.

    Follows the recipe in Section 1.1: eigendecompose C (=V Lambda V^T), clip
    negative eigenvalues to 0, set vecs = V @ sqrt(Lambda). All N columns are
    kept (including zero-eigenvalue ones) so callers can pad predictably.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    min_ev = float(eigenvalues[0])
    if min_ev < -psd_eps:
        raise ValueError(
            f"matrix is not positive semi-definite (min eigenvalue = {min_ev:.6g})"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]


def build_group_vectors(rho_s: float, G: int, M: int, D: int | None = None,
                         psd_eps: float = 1e-10) -> np.ndarray:
    """Exact (N, D) vectors with within-group correlation rho_s, cross-group 0.

    Pads with zero columns if D > N (N = G*M).
    """
    N = G * M
    C = block_constant_matrix(1.0, rho_s, 0.0, G, M)
    vecs = _exact_factor(C, psd_eps=psd_eps)  # (N, N)
    if D is None:
        D = N
    if D < N:
        raise ValueError(f"D={D} must be >= N={N} to represent the exact Gram matrix")
    if D > N:
        vecs = np.hstack([vecs, np.zeros((N, D - N))])
    return vecs


def build_stream_vectors(rho_sA: float, rho_sB: float, phi: float, G: int, M: int,
                          D: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X = sqrt(phi) v_A + sqrt(1-phi) v_B, with A/B in disjoint coordinate blocks.

    Returns (X, v_A, v_B), each (N, 2*Dblock): v_A occupies the first block of
    columns and v_B the second, so v_A^T v_B == 0 exactly and
    X X^T == block_constant_matrix(1, phi*rho_sA + (1-phi)*rho_sB, 0, G, M).
    """
    N = G * M
    if D is None:
        Dblock = N
    else:
        if D % 2 != 0:
            raise ValueError("D must be even to split into disjoint A/B blocks")
        Dblock = D // 2
    vA_block = build_group_vectors(rho_sA, G, M, D=Dblock)
    vB_block = build_group_vectors(rho_sB, G, M, D=Dblock)
    zeros = np.zeros((N, Dblock))
    v_A = np.hstack([vA_block, zeros])
    v_B = np.hstack([zeros, vB_block])
    X = np.sqrt(phi) * v_A + np.sqrt(1.0 - phi) * v_B
    return X, v_A, v_B

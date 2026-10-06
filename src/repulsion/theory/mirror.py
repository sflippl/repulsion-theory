"""The depth-L mirror variable (Section 2 of the multi-stream repulsion spec).

For a stream of depth L (number of weight layers; L-1 hidden layers), scale
gamma, mode-s residual drive g, and rate eta: parametrize B = gamma*cosh(phi),
A = gamma*sinh(phi) (so B^2 - A^2 = gamma^2 always), and
p := B^(L-1) * A (B^0 = 1 for L=1, i.e. p = A directly, no hidden layer). The
mirror variable z satisfies dot(z) = eta*sigma*g exactly.
"""
from __future__ import annotations

import numpy as np


def p_B_A_from_z(z, gamma: float, L: int):
    """Return (p, B, A) for depth L in {1, 2, 3}. `z` may be scalar or array."""
    z = np.asarray(z, dtype=np.float64)
    if L == 1:
        return z, None, z
    if L == 2:
        B = gamma * np.cosh(z)
        A = gamma * np.sinh(z)
        return B * A, B, A
    if L == 3:
        # Pitfall: trial z during root-finding can leave (-pi/2, pi/2) before
        # convergence -- clip, or tan() silently returns nan/garbage.
        gz = np.clip(gamma * z, -np.pi / 2 + 1e-9, np.pi / 2 - 1e-9)
        A = gamma * np.tan(gz)
        B = np.sqrt(gamma ** 2 + A ** 2)
        return B ** 2 * A, B, A
    raise NotImplementedError("only L in {1, 2, 3} implemented")


def z_from_p(p, gamma: float, L: int):
    """Inverse map p -> z for L in {1, 2, 3}. Used for round-trip sanity checks."""
    p = np.asarray(p, dtype=np.float64)
    if L == 1:
        return p
    if L == 2:
        # p = B*A = gamma^2 sinh(z)cosh(z) = (gamma^2/2) sinh(2z)
        return 0.5 * np.arcsinh(2.0 * p / gamma ** 2)
    if L == 3:
        # A is the unique real root of A^3 + gamma^2*A = p (Cardano)
        A = _cardano_root(gamma ** 2, -p)
        gz = np.arctan(A / gamma)
        return gz / gamma
    raise NotImplementedError("only L in {1, 2, 3} implemented")


def _cardano_root(p_coef, q_coef):
    """Real root of x^3 + p_coef*x + q_coef = 0 (monotone-increasing case, p_coef >= 0)."""
    p_coef = np.asarray(p_coef, dtype=np.float64)
    q_coef = np.asarray(q_coef, dtype=np.float64)
    disc = (q_coef / 2.0) ** 2 + (p_coef / 3.0) ** 3
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    u = np.cbrt(-q_coef / 2.0 + sqrt_disc)
    v = np.cbrt(-q_coef / 2.0 - sqrt_disc)
    return u + v

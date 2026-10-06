"""The multi-stream mode solver (Section 3) and correlation -> sigma helpers
(Section 4) of the multi-stream repulsion spec.
"""
from __future__ import annotations

from scipy.optimize import brentq

from repulsion.theory.mirror import p_B_A_from_z


def solve_mode(sigmas, etas, gammas, Ls, sigma_Y):
    """Solve sum_j sigma_j * p_j(mu) == sigma_Y for the shared multiplier mu.

    `sum_j sigma_j*p_j(mu)` is strictly increasing in mu (each p_j is strictly
    increasing in z_j = eta_j*sigma_j*mu, hence in mu, given sigma_j > 0), so a
    bracketing root-finder always works once the bracket contains the root.

    Returns (ps, Bs, mu_star): per-stream p_j and B_j at the solution.
    """
    J = len(sigmas)

    def p_j(j, mu):
        z = etas[j] * sigmas[j] * mu
        p, _, _ = p_B_A_from_z(z, gammas[j], Ls[j])
        return float(p)

    def total(mu):
        return sum(sigmas[j] * p_j(j, mu) for j in range(J)) - sigma_Y

    mu_hi = 1.0
    while total(mu_hi) < 0 and mu_hi < 1e8:
        mu_hi *= 2
    mu_lo = -1.0
    while total(mu_lo) > 0 and mu_lo > -1e8:
        mu_lo *= 2
    mu_star = brentq(total, mu_lo, mu_hi, xtol=1e-14, rtol=1e-14, maxiter=200)

    ps = [p_j(j, mu_star) for j in range(J)]
    Bs = []
    for j in range(J):
        z = etas[j] * sigmas[j] * mu_star
        _, B, _ = p_B_A_from_z(z, gammas[j], Ls[j])
        Bs.append(B)
    return ps, Bs, mu_star


def stream_sigmas(rho_sA: float, rho_sB: float, phi_in: float, M: int):
    """(sig12, sig3, s_in) for a stream's K_X = X X^T (Section 4)."""
    s_in = phi_in * rho_sA + (1.0 - phi_in) * rho_sB
    sig12 = (1.0 + (M - 1) * s_in) ** 0.5
    sig3 = (1.0 - s_in) ** 0.5
    return sig12, sig3, s_in


def target_sigmas(rho_sA1: float, rho_sB1: float, phi_out: float, M: int):
    """(sigY12, sigY3, s_out) for K_Y, built from stream 1's own vectors (Section 4)."""
    return stream_sigmas(rho_sA1, rho_sB1, phi_out, M)

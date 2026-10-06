"""The k-WTA kernel (Section 6, given verbatim), with a precomputed lookup
table for use inside parameter sweeps -- `kwta_kernel` itself is slow (nested
adaptive quadrature) and must not be called directly inside a sweep.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal, norm


def kwta_kernel(rho, alpha):
    """Output cosine similarity of random kWTA features."""
    rho = np.clip(rho, -0.999999, 0.999999)
    t = norm.ppf(1 - alpha)
    cov = [[1, rho], [rho, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov)

    def integrand(y, x):
        return x * y * rv.pdf([x, y])

    num = dblquad(integrand, t, np.inf, lambda x: t, lambda x: np.inf)[0]
    den = alpha + t * norm.pdf(t)
    return num / den


def build_kwta_lookup(alphas, rho_grid=None):
    """Precompute {alpha: (rho_grid, values)} lookup tables (40-pt default grid)."""
    if rho_grid is None:
        rho_grid = np.linspace(-0.95, 0.95, 40)
    table = {}
    for alpha in alphas:
        values = np.array([kwta_kernel(rho, alpha) for rho in rho_grid])
        table[alpha] = (rho_grid, values)
    return table


def kwta_interp(rho, alpha, table):
    """Interpolate kwta_kernel(rho, alpha) using a precomputed lookup table."""
    rho_grid, values = table[alpha]
    return float(np.interp(rho, rho_grid, values))

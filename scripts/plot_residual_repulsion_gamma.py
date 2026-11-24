"""Closed-form pairmate score of pure-A autoencoding vs. initial weight scale
gamma (no simulations), for the dense and kWTA-sparse streams at convergence
and depths L = 2, ..., 5 (L weight layers, both streams the same depth).

Theory as in notebooks/residual_repulsion.ipynb (compute_one_setting with
phi_in = phi_out = 1, eta = 1, G = M = 2; gamma = initial weight variance:
hidden layers W(0) = sqrt(gamma)*P, readout 0), which covers L <= 3. In one
mode, with Gamma = sqrt(gamma), the
readout strength A and each hidden layer's strength B satisfy
    B^2 - A^2 = Gamma^2,  p = B^(L-1) * A,  dA/dz = B^(L-1),
where z = sigma * mu and mu is shared by the streams. With u = A/Gamma this is
    F_L(u) = Gamma^(L-2) * sigma * mu,  F_L(u) = int_0^u (1 + v^2)^(-(L-1)/2) dv,
    F_2 = asinh(u),  F_3 = atan(u),  F_4 = u / sqrt(1 + u^2),
    F_5 = (u / (1 + u^2) + atan(u)) / 2.
The streams then satisfy F_L(u_loser) = (sigma_loser/sigma_winner) * F_L(u_winner)
and the mode target sum_j sigma_mj * p_mj = sigma_m,dense. The representation at
the last hidden layer has eigenvalues l_m = lam_mj * B_mj^(2(L-1)), and
    (a2 - a3) / a1 = 2 (l2 - l3) / (l1 + l2 + 2 l3).

As gamma -> 0 the stream with the larger sigma_m takes the whole mode. At L = 2
the other stream's l vanishes as gamma^(1 - sigma_loser/sigma_winner), which is
slow when the sigmas are close. At L >= 3, F_L is bounded, so u_loser stays
finite and l vanishes as gamma^(L-1) whatever the sigmas are.

Plotted against gamma^L, one panel per rho_A and one color per depth. Writes to
figures/residual-repulsion-gamma/:
    pairmate-vs-gamma-power-1e-10.{pdf,png} (gamma^L from 1e-10 to 1),
    pairmate-vs-gamma-power-1e-100.{pdf,png} (gamma^L from 1e-100 to 1),
    pairmate-vs-gamma-power.csv

Usage:
    python scripts/plot_residual_repulsion_gamma.py
    python scripts/plot_residual_repulsion_gamma.py --depths 2
    python scripts/plot_residual_repulsion_gamma.py --kappa 0.01
"""
from __future__ import annotations

import argparse
import itertools
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.stats import norm

from plot_common import (FONTSIZE, LINEWIDTH, TICK_FONTSIZE, label_axis, reference_line,
                         save_figure, setup_style, top_legend)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(ROOT, "figures", "residual-repulsion-gamma")

RHOS = [0.1, 0.5, 0.9]
DEPTHS = [2, 3, 4, 5]
KAPPA = 0.005
DEPTH_COLORS = ["#4477AA", "#CC6677", "#228833", "#CCBB44"]
STREAM_LABELS = ["Dense", "Sparse"]
STREAM_STYLES = ["-", "--"]
mp.mp.dps = 40


def kwta_kernel(rho, kappa):
    """Cosine similarity of random kWTA features (residual_repulsion.ipynb)."""
    t = norm.ppf(1 - kappa)
    sq = np.sqrt(1 - rho**2)
    opts = dict(limit=500, epsabs=1e-14, epsrel=1e-13)
    I1 = quad(lambda x: x**2 * norm.sf((t - rho * x) / sq) * norm.pdf(x), t, np.inf, **opts)[0]
    I2 = quad(lambda x: x * norm.pdf((t - rho * x) / sq) * norm.pdf(x), t, np.inf, **opts)[0]
    return (rho * I1 + sq * I2) / (kappa + t * norm.pdf(t))


def mode_eigs(rho_s, rho_cross):
    """Eigenvalues (lam1, lam2, lam3) of the item Gram matrix, G = M = 2."""
    return 1 + rho_s + 2 * rho_cross, 1 + rho_s - 2 * rho_cross, 1 - rho_s


def stream_eigs(rho, kappa):
    """2 x 3 array of Gram eigenvalues: dense stream, then sparse stream."""
    return np.array([mode_eigs(rho, 0.0), mode_eigs(kwta_kernel(rho, kappa), kwta_kernel(0.0, kappa))])


def F(u, L):
    if L == 2:
        return mp.asinh(u)
    if L == 3:
        return mp.atan(u)
    if L == 4:
        return u / mp.sqrt(1 + u * u)
    if L == 5:
        return (u / (1 + u * u) + mp.atan(u)) / 2
    raise NotImplementedError(f"only L in 2..5, got L={L}")


def F_inverse(y, L):
    if L == 2:
        return mp.sinh(y)
    if L == 3:
        return mp.tan(y)
    if L == 4:
        return y / mp.sqrt(1 - y * y)
    # L = 5: Newton from u = y, which is below the root; F is concave, so the
    # iterates increase monotonically to it
    u = y
    tol = mp.mpf(10) ** (-mp.mp.dps + 5)
    while True:
        step = (y - F(u, L)) * (1 + u * u) ** 2
        u += step
        if abs(step) <= tol * (1 + abs(u)):
            return u


def solve_mode(sigmas, gamma, L):
    """u of each stream in one mode (target sigma_Y = sigmas[0], the dense stream).

    Bisection on log(u) of the stream with the larger sigma, so that a winner
    with F_L(u) almost at its bound and a finite loser are both well resolved.
    """
    Gamma = mp.sqrt(mp.mpf(gamma))
    win = int(sigmas[1] > sigmas[0])
    ratio = sigmas[1 - win] / sigmas[win]

    def streams(log_u):
        u_win = mp.exp(log_u)
        u_lose = F_inverse(ratio * F(u_win, L), L)
        return (u_win, u_lose) if win == 0 else (u_lose, u_win)

    def total(log_u):
        return sum(s * Gamma**L * (1 + u * u) ** (mp.mpf(L - 1) / 2) * u
                   for s, u in zip(sigmas, streams(log_u))) - sigmas[0]

    lo, hi = mp.mpf(-1), mp.mpf(1)
    while total(lo) > 0:
        lo *= 2
    while total(hi) < 0:
        hi *= 2
    for _ in range(int(mp.mp.prec) + 12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) < 0 else (lo, mid)
    return streams((lo + hi) / 2)


def pairmate_scores(lams, gamma, L):
    """Pairmate score of each stream, from its 3 modes (lams: 2 x 3 eigenvalues)."""
    Gamma2 = mp.mpf(gamma)
    l = [[None] * 3 for _ in lams]
    for m in range(3):
        mode_lams = [mp.mpf(float(lam[m])) for lam in lams]
        us = solve_mode([mp.sqrt(v) for v in mode_lams], gamma, L)
        for j, (v, u) in enumerate(zip(mode_lams, us)):
            l[j][m] = v * (Gamma2 * (1 + u * u)) ** (L - 1)
    return [float(2 * (a[1] - a[2]) / (a[0] + a[1] + 2 * a[2])) for a in l]


def _score_row(cfg):
    rho, depth, gamma, lams = cfg
    dense, sparse = pairmate_scores(lams, gamma, depth)
    return dict(rho=rho, depth=depth, gamma=gamma, gamma_power=gamma**depth, dense=dense, sparse=sparse)


def plot_rhos(df, depths, power_min, out_paths, xticks, panel_mm=(45, 38)):
    d = df[df.gamma_power >= power_min * (1 - 1e-9)]
    n = len(RHOS)
    fig, axes = plt.subplots(1, n, figsize=(n * panel_mm[0] / 25.4, panel_mm[1] / 25.4), sharey=True)
    fig.subplots_adjust(wspace=0.22)
    for c, (ax, rho) in enumerate(zip(axes, RHOS)):
        reference_line(ax, 0.0)
        for depth, color in zip(depths, DEPTH_COLORS):
            sub = d[(d.rho == rho) & (d.depth == depth)].sort_values("gamma_power")
            for name, ls in zip(STREAM_LABELS, STREAM_STYLES):
                ax.plot(sub.gamma_power, sub[name.lower()], color=color, linestyle=ls, lw=LINEWIDTH)
        ax.set_xscale("log")
        ax.set_xlim(power_min, 1.0)
        ax.set_xticks(xticks)
        ax.minorticks_off()
        ax.set_xlabel(r"$\gamma^{\,\mathrm{depth}}$", fontsize=FONTSIZE, labelpad=1)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1)
        label_axis(ax, ylabel="Pairmate score" if c == 0 else None, title=rf"$\rho_A$ = {rho}")
    handles = [Line2D([0], [0], color="k", linestyle=ls, lw=LINEWIDTH, label=name)
               for name, ls in zip(STREAM_LABELS, STREAM_STYLES)]
    handles += [Line2D([0], [0], color=c, lw=LINEWIDTH, label=f"Depth {d_}") for d_, c in zip(depths, DEPTH_COLORS)]
    top_legend(fig, handles, y=0.98)
    save_figure(fig, out_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ext", nargs="+", default=["pdf", "png"])
    args = parser.parse_args()

    setup_style()
    # log10(gamma^L): the same grid for every depth, finer where the 1e-10 figure looks
    exponents = np.concatenate([np.arange(-100, -10, 1.0), np.arange(-10, 0.01, 0.1)])
    print(f"kappa = {args.kappa}, k(0) = {kwta_kernel(0.0, args.kappa):.5f}")
    eigs = {rho: stream_eigs(rho, args.kappa) for rho in RHOS}
    for rho, lams in eigs.items():
        print(f"\nrho_A = {rho}, k(rho_A) = {kwta_kernel(rho, args.kappa):.5f}")
        for m in range(3):
            win = int(lams[1, m] > lams[0, m])
            print(f"  mode {m + 1}: lam dense {lams[0, m]:.5f}, sparse {lams[1, m]:.5f}, winner {STREAM_LABELS[win]}, "
                  f"L = 2 loser exponent {1 - np.sqrt(lams[1 - win, m] / lams[win, m]):.4f}")
    cfgs = [(rho, depth, 10.0 ** (e / depth), eigs[rho]) for rho, depth, e in itertools.product(RHOS, args.depths, exponents)]
    with Pool(args.workers) as p:
        df = pd.DataFrame(p.map(_score_row, cfgs))
    suffix = "" if args.kappa == KAPPA else f"-kappa{args.kappa:g}"
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"pairmate-vs-gamma-power{suffix}.csv"), index=False)

    report = [1.0, 1e-2, 1e-5, 1e-10, 1e-20, 1e-50, 1e-100]
    nearest = lambda d, g: d.iloc[(np.log10(d.gamma_power) - np.log10(g)).abs().argmin()]
    print("\n" + "depth  rho    gamma^L  dense   sparse")
    for depth, rho in itertools.product(args.depths, RHOS):
        for g in report:
            r = nearest(df[(df.rho == rho) & (df.depth == depth)], g)
            print(f"{depth:<5}  {rho:<5}  {g:<7.0e}  {r['dense']:+.4f}  {r['sparse']:+.4f}")

    for power_min, name, xticks in [(1e-10, "1e-10", [1e-10, 1e-5, 1]),
                                    (1e-100, "1e-100", [1e-100, 1e-50, 1])]:
        plot_rhos(df, args.depths, power_min,
                  [os.path.join(OUT_DIR, f"pairmate-vs-gamma-power-{name}{suffix}.{e}") for e in args.ext], xticks)


if __name__ == "__main__":
    main()

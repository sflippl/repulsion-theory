"""Generates all figures from the associative-repulsion notebook analysis
(closed-form theory + trained-network simulations for the "pairmate
bias"/"pairmate score" phenomenon). Ported out of
notebooks/associative_repulsion.ipynb into a single reusable script, with
network training consolidated into one function (train_network).

Outputs (under --figures-dir, default "figures"):
    associative-repulsion-1/associative-network-traj.pdf
    associative-repulsion-1/predictive-network-traj.pdf
    associative-repulsion-1/repulsion-heatmap.pdf
    associative-repulsion-1/repulsion-heatmap-rho0.5.pdf
    associative-repulsion-1/threshold-plot.pdf
    associative-repulsion-1/pairmate_vs_gamma.pdf
    associative-repulsion-1/pairmate_networks_multigamma.pdf
    associative-repulsion-1/regions.pdf
    associative-repulsion-2/plot1_D1_vs_rhoB.pdf
    associative-repulsion-2/plot2_D1_heatmap.pdf
    associative-repulsion-2/plot3_D2_vs_rhoB.pdf
    associative-repulsion-2/plot4_D2_heatmap.pdf

Note: several figures involve training many small networks (up to a few
hundred thousand gradient steps each), so a full run can take a while.
Use --only to (re)generate a subset while iterating.

Usage:
    python scripts/generate_associative_repulsion_figures.py
    python scripts/generate_associative_repulsion_figures.py --only threshold-plot regions
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from tqdm import tqdm

INTEGRATION_COLOR = "#2166AC"
DIFFERENTIATION_COLOR = "#E9967A"
REPULSION_COLOR = "#B2182B"


# ====================================================================== #
# Network training (shared by every simulation-based figure)
# ====================================================================== #

def make_group_vectors(rho_id, rho_cross, rho_s, G, M, D):
    """N=G*M vectors in R^D with EXACT group-wise Gram structure, via
    eigendecomposition of the explicit Gram matrix. Pads with zeros if D>N."""
    N = G * M
    Gram = np.full((N, N), rho_cross, dtype=float)
    for i in range(N):
        gi = i // M
        for j in range(N):
            gj = j // M
            if i == j:
                Gram[i, j] = rho_id
            elif gi == gj:
                Gram[i, j] = rho_s
    w, V = np.linalg.eigh(Gram)
    w = np.clip(w, 0, None)
    vecs = V @ np.diag(np.sqrt(w))
    if D > N:
        vecs = np.hstack([vecs, np.zeros((N, D - N))])
    return vecs


def orthogonal_stiefel(rows, cols, gen):
    """rows x cols tensor P with P P^T = I_rows (needs cols >= rows)."""
    Q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=gen))
    return Q.T


def build_associative_repulsion_data(rho_sA, rho_sB, phi_in, phi_out, G=2, M=2, DA=25,
                                      dtype=torch.float64):
    """Builds the (X, Y, Z) input/target/probe tensors for one
    associative-repulsion training run: X mixes item-A/item-B vectors by
    phi_in (input salience), Y mixes them by phi_out (output salience), and
    Z probes with pure item-A input (used to read out the learned
    representation/prediction). Also returns the underlying (vA, vB)
    item vectors."""
    N = G * M
    vA = make_group_vectors(1.0, 0.0, rho_sA, G, M, DA)
    vB = make_group_vectors(1.0, 0.0, rho_sB, G, M, DA)
    X_np = (np.sqrt(phi_in) * np.hstack([vA, np.zeros((N, DA))])
            + np.sqrt(1 - phi_in) * np.hstack([np.zeros((N, DA)), vB]))
    Y_np = (np.sqrt(phi_out) * np.hstack([vA, np.zeros((N, DA))])
            + np.sqrt(1 - phi_out) * np.hstack([np.zeros((N, DA)), vB]))
    Z_np = np.hstack([vA, np.zeros((N, DA))])
    X = torch.tensor(X_np, dtype=dtype)
    Y = torch.tensor(Y_np, dtype=dtype)
    Z = torch.tensor(Z_np, dtype=dtype)
    return X, Y, Z, vA, vB


class TwoLayerLinear(nn.Module):
    """f(x) = x @ W1 @ W2, with W1(0) = sqrt(2*gamma) * P (orthogonal, via
    QR of a random Gaussian matrix) and W2(0) = 0."""

    def __init__(self, D_in, d_hidden, D_out, gamma, gen, dtype=torch.float64):
        super().__init__()
        Gamma = float(np.sqrt(2 * gamma))
        P1 = orthogonal_stiefel(D_in, d_hidden, gen).to(dtype)
        self.W1 = nn.Parameter(Gamma * P1)
        self.W2 = nn.Parameter(torch.zeros(d_hidden, D_out, dtype=dtype))

    def forward(self, X):
        return X @ self.W1 @ self.W2

    def hidden(self, X):
        return X @ self.W1


def train_network(X, Y, gamma, n_steps, lr, d_hidden=80, seed=0, dtype=torch.float64,
                   loss_threshold=1e-8, track_steps=None, on_track=None, progress=True):
    """Trains a two-layer linear network (both layers trained, exact
    orthogonal init) via standard PyTorch autodiff and full-batch gradient
    descent (an explicit Euler discretization of gradient flow).

    Stops early once the loss drops below loss_threshold (pass None to
    always run all n_steps). If track_steps is given, on_track(step, model)
    is called right after the parameter update at each of those steps, to
    record training trajectories. Returns the trained model.
    """
    gen = torch.Generator().manual_seed(seed)
    model = TwoLayerLinear(X.shape[1], d_hidden, Y.shape[1], gamma, gen, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    track_steps = set(track_steps or ())

    steps = tqdm(range(n_steps), leave=False) if progress else range(n_steps)
    for step in steps:
        optimizer.zero_grad()
        pred = model(X)
        loss = 0.5 * ((Y - pred) ** 2).sum()
        loss.backward()
        optimizer.step()
        if on_track is not None and step in track_steps:
            on_track(step, model)
        if loss_threshold is not None and loss.item() < loss_threshold:
            break

    return model


def simulate_pairmate_score(rho_sA, rho_sB, phi_in, phi_out, gamma, n_steps, lr,
                             G=2, M=2, DA=25, d_hidden=80, seed=0, loss_threshold=1e-8):
    """Trains one network and returns (pairmate_score, final_prediction_error),
    where pairmate_score = a2/a1 of the learned hidden-representation Gram
    matrix (probed with pure item-A input)."""
    X, Y, Z, _, _ = build_associative_repulsion_data(rho_sA, rho_sB, phi_in, phi_out, G, M, DA)
    model = train_network(X, Y, gamma, n_steps, lr, d_hidden=d_hidden, seed=seed,
                           loss_threshold=loss_threshold)
    with torch.no_grad():
        err = torch.linalg.norm(Y - model(X)).item()
        H = model.hidden(Z)
        HHt = H @ H.T
        a1, a2 = HHt[0, 0].item(), HHt[0, 1].item()
    return a2 / a1, err


def simulate_D1_D2(rho_sA, rho_sB, phi_in, gamma, n_steps=150_000, lr=0.02,
                    G=2, M=2, DA=25, d_hidden=80, seed=0, phi_out=0.4, loss_threshold=1e-8):
    """D1 = cosine distance to ground truth = 1 - corr(vB_hat, vB).
    D2 = relative distance to pairmate = rho_sB - corr(vB_hat, vB_pairmate).
    phi_out is arbitrary (D1, D2 are provably phi_out-independent); kept
    fixed here just to have a concrete target to train on."""
    X, Y, Z, _, vB = build_associative_repulsion_data(rho_sA, rho_sB, phi_in, phi_out, G, M, DA)
    model = train_network(X, Y, gamma, n_steps, lr, d_hidden=d_hidden, seed=seed,
                           loss_threshold=loss_threshold)
    with torch.no_grad():
        pred_err = torch.linalg.norm(Y - model(X)).item()
        Yhat = model(Z).numpy()
    vB_hat = Yhat[:, DA:] / np.sqrt(1 - phi_out)
    corr_gt = np.dot(vB_hat[0], vB[0]) / (np.linalg.norm(vB_hat[0]) * np.linalg.norm(vB[0]))
    corr_pm = np.dot(vB_hat[0], vB[1]) / (np.linalg.norm(vB_hat[0]) * np.linalg.norm(vB[1]))
    D1 = 1 - corr_gt
    D2 = rho_sB - corr_pm
    return D1, D2, pred_err


# ====================================================================== #
# Closed-form theory
# ====================================================================== #

def group_eigs(a1, a2, a3, G, M):
    """(a1,a2,a3) -> (lambda1, lambda2, lambda3), multiplicities (1, G-1, (M-1)*G)."""
    lam1 = (a1 - a2) + (a2 - a3) * M + a3 * G * M
    lam2 = (a1 - a2) + (a2 - a3) * M
    lam3 = a1 - a2
    return lam1, lam2, lam3


def eigs_to_a(lam1, lam2, lam3, G, M):
    """Inverse of group_eigs."""
    a3 = (lam1 - lam2) / (G * M)
    a2 = a3 + (lam2 - lam3) / M
    a1 = a2 + lam3
    return a1, a2, a3


def learnedreps_eigenvalues(x, y, z, c, gamma):
    """
    x,y,z,c: eigenvalues of K_X, K_Y, K_Z, Z X^T (shared eigenbasis).
    lambda_i(H H^T) = 2*gamma*z_i + (c_i^2/x_i)*(sqrt(y_i/x_i+gamma^2)-gamma)
    (second term is 0 wherever x_i==0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)
    out = 2 * gamma * z
    mask = x > 0
    xs = np.where(mask, x, 1.0)
    out = np.where(mask, out + (c**2 / xs) * (np.sqrt(y / xs + gamma**2) - gamma), out)
    return out


def learnedreps_associative_repulsion(rho_sA, rho_sB, phi_in, phi_out, gamma, G=2, M=2):
    """
    Closed-form a1, a2, a3 of the learned representation H H^T at the
    (single) hidden layer of a one-hidden-layer, both-layers-trained
    network, gamma convention W_1(0)=sqrt(2*gamma)*P.

    IMPORTANT (bug we found and fixed): the cross-term eigenvalues
    lambda(Z X^T) are sqrt(phi_in) * (1+(M-1)*rho_sA, ..., 1-rho_sA) -- the
    sqrt(phi_in) multiplies the WHOLE bracket, not just rho_sA. That's what
    this function implements.
    """
    s_in = phi_in * rho_sA + (1 - phi_in) * rho_sB
    s_out = phi_out * rho_sA + (1 - phi_out) * rho_sB

    lamX = group_eigs(1.0, s_in, 0.0, G, M)
    lamY = group_eigs(1.0, s_out, 0.0, G, M)
    lamZ = group_eigs(1.0, rho_sA, 0.0, G, M)
    s = np.sqrt(phi_in)
    lamZX = group_eigs(s * 1.0, s * rho_sA, 0.0, G, M)

    idx = [0, 2]  # lambda_1(==lambda_2) and lambda_3
    x = np.array([lamX[i] for i in idx])
    y = np.array([lamY[i] for i in idx])
    z = np.array([lamZ[i] for i in idx])
    c = np.array([lamZX[i] for i in idx])
    lam12, lam3 = learnedreps_eigenvalues(x, y, z, c, gamma)

    a1, a2, a3 = eigs_to_a(lam12, lam12, lam3, G, M)
    return a1, a2, a3


def _as_list(v):
    return v if isinstance(v, (list, tuple, np.ndarray)) else [v]


def compute_learnedreps_df(rho_sA, rho_sB, phi_in, phi_out, gamma, G=2, M=2):
    """
    rho_sA, rho_sB, phi_in, phi_out, gamma: each a scalar or a list of
    values to sweep (G, M fixed by default, can also be given as lists).
    Returns a long DataFrame, one row per combination, with a1, a2, a3,
    a2_over_a1, a3_over_a1 (the last two are the recommended quantities --
    a1 is not generally 1 the way the input correlations are, so a2, a3 on
    their own are not directly comparable to rho_sA, rho_sB; dividing by
    a1 puts them back on the same scale).
    """
    axes = {
        "rho_sA": _as_list(rho_sA), "rho_sB": _as_list(rho_sB),
        "phi_in": _as_list(phi_in), "phi_out": _as_list(phi_out),
        "gamma": _as_list(gamma), "G": _as_list(G), "M": _as_list(M),
    }
    keys = list(axes.keys())
    rows = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        setting = dict(zip(keys, combo))
        a1, a2, a3 = learnedreps_associative_repulsion(
            setting["rho_sA"], setting["rho_sB"], setting["phi_in"],
            setting["phi_out"], setting["gamma"], setting["G"], setting["M"])
        row = dict(setting)
        row["a1"] = a1
        row["a2"] = a2
        row["a3"] = a3
        row["a2_over_a1"] = a2 / a1 if a1 != 0 else np.nan
        row["a3_over_a1"] = a3 / a1 if a1 != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def r_ratio(rho_sA, rho_sB, phi_in):
    s_in = phi_in * rho_sA + (1 - phi_in) * rho_sB
    return (1 + rho_sA) * (1 - s_in) / ((1 - rho_sA) * (1 + s_in))


def D1_D2(rho_sA, rho_sB, phi_in):
    """D1 = cosine distance to ground truth = 1 - corr(vB_hat, vB).
    D2 = relative distance to pairmate = rho_sB - corr(vB_hat, vB_pairmate)."""
    r = r_ratio(rho_sA, rho_sB, phi_in)
    P, N = 1 + rho_sB, 1 - rho_sB
    denom = np.sqrt(2 * (r**2 * P + N))
    corr_gt = (r * P + N) / denom
    corr_pm = (r * P - N) / denom
    return 1 - corr_gt, rho_sB - corr_pm


def pairmate_score(rho_sA, rho_sB, phi_in, phi_out, gamma, M=2):
    """Closed-form pairmate score (a2-a3)/a1 for general gamma, in terms of
    rho_sA, rho_sB, phi_in, phi_out directly (equivalent to, but more
    numerically stable at small gamma than, going through
    learnedreps_associative_repulsion)."""
    rho_sA = np.asarray(rho_sA, dtype=float)
    rho_sB = np.asarray(rho_sB, dtype=float)
    phi_in = np.asarray(phi_in, dtype=float)
    phi_out = np.asarray(phi_out, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    s_in = phi_in * rho_sA + (1 - phi_in) * rho_sB
    s_out = phi_out * rho_sA + (1 - phi_out) * rho_sB

    Q_A = 1 + (M - 1) * rho_sA
    q_in = 1 + (M - 1) * s_in
    q_out = 1 + (M - 1) * s_out
    f = lambda x: x / (np.sqrt(x + gamma**2) + gamma)  # numerically stable sqrt(x+g^2)-g

    lam12 = 2 * gamma * Q_A + (phi_in * Q_A**2 / q_in) * f(q_out / q_in)
    lam3 = 2 * gamma * (1 - rho_sA) + (phi_in * (1 - rho_sA)**2 / (1 - s_in)) * f((1 - s_out) / (1 - s_in))

    a2 = (lam12 - lam3) / M
    a1 = (lam12 + (M - 1) * lam3) / M
    return a2 / a1


def score_autoencoder(rho_sA, rho_sB):
    """Pairmate score at phi_in=phi_out=0.5, gamma=0 (closed form)."""
    s = (rho_sA + rho_sB) / 2
    R = ((1 + rho_sA) / (1 - rho_sA))**2 * (1 - s) / (1 + s)
    return (R - 1) / (R + 1)


def score_predictive(rho_sA, rho_sB):
    """Pairmate score at phi_in=1, phi_out=0, gamma=0 (closed form)."""
    up = np.sqrt((1 + rho_sA) * (1 + rho_sB))
    down = np.sqrt((1 - rho_sA) * (1 - rho_sB))
    return (up - down) / (up + down)


def PS_AE(gamma):
    """Pairmate score of the autoencoder (phi_in=phi_out=0.5) at
    rho_sA=0, rho_sB=1, as a function of gamma."""
    return -1 / (2 + 6 * gamma * np.sqrt(1 + gamma**2) + 6 * gamma**2)


def PS_PN(gamma):
    """Pairmate score of the predictive network (phi_in=1, phi_out=0) at
    rho_sA=0, rho_sB=1, as a function of gamma."""
    return 1 / (1 + 2 * gamma * np.sqrt(2 + gamma**2) + 2 * gamma**2)


def phi_out_threshold(phi_in, gamma):
    """Output-salience (phi_out) at which the pairmate score crosses zero,
    at rho_sA=0, rho_sB=1, as a function of phi_in and gamma."""
    p = np.asarray(phi_in, dtype=float)
    E = p**2 - 2 * p + 2
    D = 3 * p**2 - 6 * p + 4
    R = np.sqrt(D + gamma**2 * E**2) + gamma * E
    return p**2 * (p + 2 * gamma * R) / R**2


def phi_repulsion_boundary_v1(phi, rho):
    """Integration/differentiation-vs-repulsion boundary in phi (input
    salience) space, at fixed rho_sA=rho, gamma=0 (first derivation)."""
    return (2 * (1 + rho)**2 * phi**3
            / ((2 - phi * (1 - rho))**3 + (1 - rho) * (1 + rho)**2 * phi**3))


def phi_repulsion_boundary_v2(phi, rho):
    """Same boundary as phi_repulsion_boundary_v1, alternate derivation
    (kept to cross-check the two formulas agree)."""
    return (2 * phi**3 * (1 + rho)**4
            / ((1 - rho)**2 * (2 - phi * (1 - rho))**3
               + (1 - rho**2) * phi**3 * (1 + rho)**3))


def b_auto_zero(a):
    """Autoencoder's zero-pairmate-score boundary rho_sB(rho_sA)."""
    return a * (3 - a**2) / (1 + a**2)


# ====================================================================== #
# Generic plotting helpers
# ====================================================================== #

def plot_dataframe_heatmap(
    df, x_col, y_col, color_col, *,
    xlabel=None, ylabel=None, cbar_label=None, cmap="RdBu", symmetric_colorbar=True,
    vmin=None, vmax=None, xlim=None, ylim=None, xticks=None, yticks=None,
    origin="lower", aspect="auto", figsize_mm=(40, 40), dpi=600, fontsize=6,
    tick_fontsize=5, cbar_fraction=0.046, cbar_pad=0.01, show=True, ax=None,
):
    required_cols = [x_col, y_col, color_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing dataframe columns: {missing}")

    data = df[required_cols].dropna()
    if data.empty:
        raise ValueError("No valid data remain after dropping NaNs.")

    # Convert dataframe directly into a rectangular grid.
    x_values = np.sort(np.unique(data[x_col]))
    y_values = np.sort(np.unique(data[y_col]))
    expected_n = len(x_values) * len(y_values)
    if len(data) != expected_n:
        raise ValueError(
            f"Data do not form a complete rectangular grid: "
            f"found {len(data)} points, expected {expected_n}.")

    grid = data.pivot(index=y_col, columns=x_col, values=color_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    Z = grid.to_numpy()
    x_values = grid.columns.to_numpy()
    y_values = grid.index.to_numpy()

    if symmetric_colorbar:
        if vmin is None and vmax is None:
            zmax = np.nanmax(np.abs(Z))
            vmin, vmax = -zmax, zmax
        elif vmin is None:
            vmin = -abs(vmax)
        elif vmax is None:
            vmax = abs(vmin)
    else:
        if vmin is None:
            vmin = np.nanmin(Z)
        if vmax is None:
            vmax = np.nanmax(Z)

    plt.rcParams.update({"font.size": fontsize})
    created_fig = ax is None
    if created_fig:
        fig_width_in = figsize_mm[0] / 25.4
        fig_height_in = figsize_mm[1] / 25.4
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    else:
        fig = ax.figure

    # imshow expects pixel edges, whereas x_values/y_values are locations.
    dx = np.median(np.diff(x_values))
    dy = np.median(np.diff(y_values))
    extent = [x_values[0] - dx / 2, x_values[-1] + dx / 2,
              y_values[0] - dy / 2, y_values[-1] + dy / 2]

    im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, aspect=aspect, extent=extent)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=fontsize, pad=0.5)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=fontsize, labelpad=0)

    if created_fig:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if show:
        plt.show()

    return fig, ax, im, cbar


def plot_pairmate_transitions(
    df, x_col, y_col, score_col, rho_sA, *,
    style="bands", xlabel=None, ylabel=None, cbar_label=None,
    integration_color=INTEGRATION_COLOR, differentiation_color=DIFFERENTIATION_COLOR,
    repulsion_color=REPULSION_COLOR, transition_width=0.02, vmin=None, vmax=None,
    xlim=None, ylim=None, xticks=None, yticks=None, figsize_mm=(40, 40), dpi=600,
    fontsize=6, tick_fontsize=5, cbar_fraction=0.046, cbar_pad=0.01, spine_width=0.3,
    tick_pad=0.5, xlabel_pad=0, ylabel_pad=0, cbar_labelpad=0, show=True, ax=None,
):
    """
    Plot integration / differentiation / repulsion regions.

    rho_sA : float or str
        If float: constant integration/differentiation threshold.
        If str: dataframe column containing the local rho_sA threshold
        (supported for "bands" and "continuous" styles only).

    style : {"bands", "continuous", "colorbar"}
        "bands": three categorical regions with white transition bands
            around score = 0 and score = rho_sA.
        "continuous": three categorical regions with no white transition
            bands.
        "colorbar": continuous three-region colormap
            red -> white -> salmon -> white -> blue, white at score = 0
            and score = rho_sA (rho_sA must be a scalar here).

    Returns fig, ax, im, cbar.
    """
    if style not in {"bands", "continuous", "colorbar"}:
        raise ValueError("style must be 'bands', 'continuous', or 'colorbar'.")
    if style == "colorbar" and isinstance(rho_sA, str):
        raise ValueError("For style='colorbar', rho_sA must be a scalar.")

    required_cols = [x_col, y_col, score_col]
    if isinstance(rho_sA, str):
        required_cols.append(rho_sA)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing dataframe columns: {missing}")

    data = df[required_cols].dropna()
    if data.empty:
        raise ValueError("No valid data remain after dropping NaNs.")

    x_values = np.sort(np.unique(data[x_col]))
    y_values = np.sort(np.unique(data[y_col]))
    expected_n = len(x_values) * len(y_values)
    if len(data) != expected_n:
        raise ValueError(
            f"Data do not form a complete rectangular grid: "
            f"found {len(data)} points, expected {expected_n}.")

    score_grid = (data.pivot(index=y_col, columns=x_col, values=score_col)
                  .sort_index(axis=0).sort_index(axis=1))
    score = score_grid.to_numpy()
    x_values = score_grid.columns.to_numpy()
    y_values = score_grid.index.to_numpy()

    if isinstance(rho_sA, str):
        rho_grid = (data.pivot(index=y_col, columns=x_col, values=rho_sA)
                    .sort_index(axis=0).sort_index(axis=1))
        rho = rho_grid.to_numpy()
        if np.any(rho < 0):
            raise ValueError(f"Column '{rho_sA}' contains negative values. rho_sA must be non-negative.")
    else:
        if rho_sA < 0:
            raise ValueError("rho_sA must be non-negative.")
        rho = float(rho_sA)

    plt.rcParams.update({"font.size": fontsize})
    created_fig = ax is None
    if created_fig:
        fig_width_in = figsize_mm[0] / 25.4
        fig_height_in = figsize_mm[1] / 25.4
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    else:
        fig = ax.figure

    dx = np.median(np.diff(x_values)) if len(x_values) > 1 else 1.0
    dy = np.median(np.diff(y_values)) if len(y_values) > 1 else 1.0
    extent = [x_values[0] - dx / 2, x_values[-1] + dx / 2,
              y_values[0] - dy / 2, y_values[-1] + dy / 2]

    im = None
    cbar = None

    if style == "colorbar":
        rho = float(rho_sA)
        if rho <= 0:
            raise ValueError("rho_sA must be > 0 for style='colorbar'.")
        if vmin is None:
            vmin = np.nanmin(score)
        if vmax is None:
            vmax = np.nanmax(score)
        if vmin >= 0:
            vmin = -1e-12
        if vmax <= rho:
            vmax = rho + 1e-12
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax.")

        # Piecewise continuous colormap:
        #   repulsion: red -> white
        #   differentiation: white -> salmon -> white
        #   integration: white -> blue
        def norm(x):
            return (x - vmin) / (vmax - vmin)

        p_min, p_zero, p_rho, p_max = norm(vmin), norm(0.0), norm(rho), norm(vmax)
        p_diff_mid = (p_zero + p_rho) / 2

        cmap = LinearSegmentedColormap.from_list(
            "pairmate_transitions",
            [
                (p_min, repulsion_color), (p_zero, "white"),
                (p_zero, "white"), (p_diff_mid, differentiation_color), (p_rho, "white"),
                (p_rho, "white"), (p_max, integration_color),
            ],
        )

        im = ax.imshow(score, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                        aspect="auto", extent=extent, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
        cbar.ax.tick_params(labelsize=fontsize, pad=tick_pad)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=fontsize, labelpad=cbar_labelpad)

        # Make the two theoretically important transition points explicit.
        ticks = list(cbar.get_ticks())
        if not any(np.isclose(t, 0) for t in ticks):
            ticks.append(0.0)
        if not any(np.isclose(t, rho) for t in ticks):
            ticks.append(rho)
        ticks = sorted(t for t in ticks if vmin <= t <= vmax)
        cbar.set_ticks(ticks)

    else:
        integration = score > rho
        differentiation = (score > 0) & (score < rho)
        repulsion = score < 0

        rgb = np.empty((*score.shape, 3), dtype=float)
        rgb[integration] = to_rgb(integration_color)
        rgb[differentiation] = to_rgb(differentiation_color)
        rgb[repulsion] = to_rgb(repulsion_color)

        if style == "bands":
            zero_transition = np.abs(score) <= transition_width / 2
            rho_transition = np.abs(score - rho) <= transition_width / 2
            rgb[zero_transition | rho_transition] = (1.0, 1.0, 1.0)

        im = ax.imshow(rgb, origin="lower", aspect="auto", extent=extent, interpolation="nearest")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=ylabel_pad)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=tick_pad)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if created_fig:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if show:
        plt.show()

    return fig, ax, im, cbar


def plot_learnedreps_comparison(
    rho_sA, rho_sB, phi_in, phi_out, gamma, G=2, M=2, DA=25, d_hidden=80, lr=0.015,
    n_steps=50_000, n_track_points=50, seed=0, *,
    xlabel="Step", ylabel="Hidden rep. sim.", within_label="A1 vs. A2", across_label="A1 vs. A3",
    figsize_mm=(40, 30), dpi=300, fontsize=6, tick_fontsize=5, xticks=None, yticks=None,
    xlim=None, ylim=None, legend=True, legend_loc="best", savepath=None, show=True, x_log=True,
):
    """Trains one network (via train_network) and plots the learned
    hidden-representation similarities over the course of training against
    the closed-form prediction."""
    X, Y, Z, _, _ = build_associative_repulsion_data(rho_sA, rho_sB, phi_in, phi_out, G, M, DA)

    track_steps = set(
        [0, 1, 2, 3, 5]
        + list(np.unique(np.geomspace(1, n_steps - 1, n_track_points).astype(int)))
    )
    hist_steps, hist_a2, hist_a3 = [], [], []

    def on_track(step, model):
        with torch.no_grad():
            H = model.hidden(Z)
            HHt = H @ H.T
            a1, a2, a3 = HHt[0, 0].item(), HHt[0, 1].item(), HHt[0, 2].item()
        hist_steps.append(step)
        hist_a2.append(a2 / a1 if a1 != 0 else np.nan)
        hist_a3.append(a3 / a1 if a1 != 0 else np.nan)

    model = train_network(X, Y, gamma, n_steps, lr, d_hidden=d_hidden, seed=seed,
                           loss_threshold=None, track_steps=track_steps, on_track=on_track,
                           progress=False)

    with torch.no_grad():
        pred_err = torch.linalg.norm(Y - model(X)).item()

    a1_t, a2_t, a3_t = learnedreps_associative_repulsion(rho_sA, rho_sB, phi_in, phi_out, gamma, G, M)
    q2_target, q3_target = a2_t / a1_t, a3_t / a1_t

    plt.rcParams.update({"font.size": fontsize})
    fig_width_in = figsize_mm[0] / 25.4
    fig_height_in = figsize_mm[1] / 25.4
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    steps = np.asarray(hist_steps)
    a2 = np.asarray(hist_a2)
    a3 = np.asarray(hist_a3)
    mask = steps > 0  # step 0 cannot appear on a logarithmic x-axis

    ax.plot(steps[mask], a2[mask], color="black", label=within_label, lw=1.0)
    ax.plot(steps[mask], a3[mask], color="tab:orange", label=across_label, lw=1.0)
    ax.axhline(q2_target, color="black", linestyle="--", alpha=0.6, lw=1.0)
    ax.axhline(q3_target, color="tab:orange", linestyle="--", alpha=0.6, lw=1.0)

    if x_log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=0)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    if legend:
        ax.legend(loc=legend_loc, fontsize=fontsize, frameon=False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()

    print(f"final sim:   a2/a1={hist_a2[-1]:.4f}  a3/a1={hist_a3[-1]:.4f}")
    print(f"closed form: a2/a1={q2_target:.4f}  a3/a1={q3_target:.4f}")
    print(f"prediction error at end of training: {pred_err:.3e}")

    return fig, ax


# ====================================================================== #
# Figures
# ====================================================================== #

def fig_associative_network_traj(out_dir, seed=0):
    """associative-repulsion-1/associative-network-traj.pdf: autoencoder
    (phi_in=phi_out=0.5) hidden-rep similarity trajectory vs. closed form."""
    fig, ax = plot_learnedreps_comparison(
        rho_sA=0., rho_sB=1., phi_in=0.5, phi_out=0.5, gamma=0.1, lr=0.1, n_steps=100,
        x_log=False, fontsize=7, tick_fontsize=6, figsize_mm=(26, 20), seed=seed, show=False)
    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "associative-network-traj.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_predictive_network_traj(out_dir, seed=0):
    """associative-repulsion-1/predictive-network-traj.pdf: predictive
    network (phi_in=1, phi_out=0) hidden-rep similarity trajectory vs.
    closed form."""
    fig, ax = plot_learnedreps_comparison(
        rho_sA=0., rho_sB=1., phi_in=1., phi_out=0., gamma=0.1, lr=0.1, n_steps=100,
        x_log=False, fontsize=7, tick_fontsize=6, figsize_mm=(26, 20), seed=seed, show=False)
    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "predictive-network-traj.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_repulsion_heatmap_phi(out_dir, seed=0):
    """associative-repulsion-1/repulsion-heatmap.pdf: pairmate repulsion
    ((a2-a3)/a1) over (phi_in, phi_out), at rho_sA=0, rho_sB=1, gamma=0."""
    phis = np.linspace(0, 1, 201)[1:]
    df = compute_learnedreps_df(rho_sA=0., rho_sB=1., phi_in=phis, phi_out=phis, gamma=0.)
    df["repulsion"] = df["a2_over_a1"] - df["a3_over_a1"]

    fig, ax, im, cbar = plot_dataframe_heatmap(
        df, x_col="phi_in", y_col="phi_out", color_col="repulsion",
        xlabel=r"Input A salience ($\phi_{in}$)", ylabel=r"Output A salience ($\phi_{out}$)",
        cbar_label="Pairmate Score", cmap="RdBu", symmetric_colorbar=True, show=False,
        xlim=(0.01, 0.99), ylim=(0.01, 0.99), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
        fontsize=7, tick_fontsize=6, figsize_mm=(26, 26), dpi=300)
    ax.plot(phis, phi_repulsion_boundary_v1(phis, 0.0), color="black", linestyle="--")
    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "repulsion-heatmap.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_repulsion_heatmap_phi_rho05(out_dir, seed=0):
    """associative-repulsion-1/repulsion-heatmap-rho0.5.pdf: same as
    fig_repulsion_heatmap_phi but at rho_sA=0.5, rho_sB=1, shown as a
    continuous colorbar with both boundary-formula overlays."""
    phis = np.linspace(0, 1, 201)[1:]
    df = compute_learnedreps_df(rho_sA=0.5, rho_sB=1., phi_in=phis, phi_out=phis, gamma=0.)
    df["repulsion"] = df["a2_over_a1"] - df["a3_over_a1"]

    fig, ax, im, cbar = plot_pairmate_transitions(
        df, x_col="phi_in", y_col="phi_out", score_col="repulsion", rho_sA=0.5,
        xlabel=r"Input A salience ($\phi_{in}$)", ylabel=r"Output A salience ($\phi_{out}$)",
        xlim=(0.01, 0.99), ylim=(0.01, 0.99), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
        figsize_mm=(26, 26), dpi=300, cbar_label="Pairmate score", show=False,
        style="colorbar", fontsize=7, tick_fontsize=6)
    ax.plot(phis, phi_repulsion_boundary_v1(phis, 0.5), color="black", linestyle="--")
    ax.plot(phis, phi_repulsion_boundary_v2(phis, 0.5), color="black", linestyle="--")
    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "repulsion-heatmap-rho0.5.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def fig_threshold_plot(out_dir, seed=0):
    """associative-repulsion-1/threshold-plot.pdf: phi_out threshold (pairmate
    score = 0) vs. phi_in, at rho_sA=0, rho_sB=1, for several gamma."""
    green_cmap = LinearSegmentedColormap.from_list("threshold_green", ["#01CD59", "#094809"])
    colors = {0: green_cmap(0.00), 0.1: green_cmap(0.25), 1: green_cmap(0.75), np.inf: green_cmap(1.00)}

    phi_in = np.linspace(0.01, 0.99, 1000)

    fig, ax = plt.subplots(figsize=(26 / 25.4, 26 / 25.4), dpi=300)
    ax.set_box_aspect(1)

    threshold_gamma_zero = phi_in**3 / (3 * phi_in**2 - 6 * phi_in + 4)
    ax.plot(phi_in, threshold_gamma_zero, color=colors[0], linestyle="-", linewidth=0.9, label=r"$\gamma\to0$")
    ax.plot(phi_in, phi_out_threshold(phi_in, 0.1), color=colors[0.1], linestyle="-", linewidth=0.8, label=r"$\gamma=0.1$")
    ax.plot(phi_in, phi_out_threshold(phi_in, 1), color=colors[1], linestyle="-", linewidth=0.8, label=r"$\gamma=1$")
    threshold_gamma_inf = phi_in**2 / (phi_in**2 - 2 * phi_in + 2)
    ax.plot(phi_in, threshold_gamma_inf, color=colors[np.inf], linestyle="-", linewidth=0.9, label=r"$\gamma\to\infty$")

    ax.set_xlabel(r"Input A salience ($\phi_{\mathrm{in}}$)", fontsize=7)
    ax.set_ylabel(r"Output A salience ($\phi_{\mathrm{out}}$)", fontsize=7)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    fig.subplots_adjust(left=0.22, bottom=0.20, right=0.72, top=0.96)

    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "threshold-plot.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_pairmate_vs_gamma(out_dir, seed=0):
    """associative-repulsion-1/pairmate_vs_gamma.pdf: pairmate score of the
    autoencoder (AE) and predictive network (PN) vs. initial hidden-weight
    magnitude gamma, at rho_sA=0, rho_sB=1, with trained-network overlays."""
    gammas_sim = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    lr, n_steps = 0.02, 300_000

    ae_sim, pn_sim = [], []
    for g in gammas_sim:
        ps_ae, err_ae = simulate_pairmate_score(0.0, 1.0, 0.5, 0.5, g, n_steps, lr, seed=seed)
        ps_pn, err_pn = simulate_pairmate_score(0.0, 1.0, 1.0, 0.0, g, n_steps, lr, seed=seed)
        ae_sim.append(ps_ae)
        pn_sim.append(ps_pn)
        print(f"gamma={g:.0e}: AE sim={ps_ae:.5f} (pred_err={err_ae:.1e}), "
              f"PN sim={ps_pn:.5f} (pred_err={err_pn:.1e})")

    gammas_curve = np.geomspace(1e-4, 10.0, 400)
    figsize_mm = (30, 26)
    fig, ax = plt.subplots(figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4), dpi=600)
    ax.plot(gammas_curve, PS_PN(gammas_curve), color="tab:blue", label="Pred. network", linewidth=1)
    ax.plot(gammas_curve, PS_AE(gammas_curve), color="tab:red", label="Autoencoder", linewidth=1)
    ax.scatter(gammas_sim, ae_sim, color="tab:red", zorder=3, s=5)
    ax.scatter(gammas_sim, pn_sim, color="tab:blue", zorder=3, s=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$", fontsize=7)
    ax.set_ylabel("Pairmate score", fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=6, bbox_to_anchor=(1.05, 0.5), loc="center left")

    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "pairmate_vs_gamma.pdf"),
                dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def fig_pairmate_networks_multigamma(out_dir, seed=0):
    """associative-repulsion-1/pairmate_networks_multigamma.pdf: pairmate
    score vs. rho_sA (rho_sB=1) for the autoencoder (solid) and predictive
    network (dashed), at gamma=0, 0.1, 1.0, with region shading and
    trained-network overlays."""
    def theory_curve(rho_sA, rho_sB, phi_in, phi_out, gamma):
        if gamma == 0.0:
            return score_autoencoder(rho_sA, rho_sB) if phi_in == 0.5 else score_predictive(rho_sA, rho_sB)
        return pairmate_score(rho_sA, rho_sB, phi_in, phi_out, gamma)

    rho_sB = 1.0
    rho_sA_curve = np.linspace(0.0, 0.99, 500)
    rho_sA_sim = np.linspace(0.1, 0.9, 9)

    gammas = [0.0, 0.1, 1.0]
    gamma_sim_proxy = {0.0: 1e-4, 0.1: 0.1, 1.0: 1.0}  # gamma=0 needs a tiny nonzero proxy to simulate
    n_steps_cap = {0.0: 300_000, 0.1: 120_000, 1.0: 60_000}
    gamma_colors = {0.0: "#66C266", 0.1: "#2E8B57", 1.0: "#006400"}

    sim_results = {}
    for gamma in gammas:
        g_sim, n_steps = gamma_sim_proxy[gamma], n_steps_cap[gamma]
        for net_name, phi_in, phi_out in [("AE", 0.5, 0.5), ("PN", 1.0, 0.0)]:
            scores = []
            for rA in tqdm(rho_sA_sim, desc=f"gamma={gamma} {net_name}"):
                ps, err = simulate_pairmate_score(rA, rho_sB, phi_in, phi_out, g_sim, n_steps, lr=0.02, seed=seed)
                scores.append(ps)
            sim_results[(gamma, net_name)] = np.array(scores)

    y_auto0 = score_autoencoder(rho_sA_curve, rho_sB)
    ymin, ymax = min(y_auto0.min(), -0.05), 1.02
    diff_upper = np.maximum(rho_sA_curve, 0)

    figsize_mm = (30, 26)
    fig, ax = plt.subplots(figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4), dpi=600)
    ax.fill_between(rho_sA_curve, ymin, 0, color=REPULSION_COLOR, alpha=0.15, linewidth=0)
    ax.fill_between(rho_sA_curve, 0, diff_upper, color=DIFFERENTIATION_COLOR, alpha=0.20, linewidth=0)
    ax.fill_between(rho_sA_curve, diff_upper, ymax, color=INTEGRATION_COLOR, alpha=0.15, linewidth=0)

    for gamma in gammas:
        color = gamma_colors[gamma]
        y_auto = theory_curve(rho_sA_curve, rho_sB, 0.5, 0.5, gamma)
        y_pred = theory_curve(rho_sA_curve, rho_sB, 1.0, 0.0, gamma)
        ax.plot(rho_sA_curve, y_auto, color=color, linewidth=1, linestyle="-")
        ax.plot(rho_sA_curve, y_pred, color=color, linewidth=1, linestyle="--")
        ax.scatter(rho_sA_sim, sim_results[(gamma, "AE")], color=color, zorder=3, s=5, marker="o")
        ax.scatter(rho_sA_sim, sim_results[(gamma, "PN")], color=color, zorder=3, s=5, marker="^")

    ax.set_xlim(rho_sA_curve.min(), 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\rho_{s,A}$", fontsize=7)
    ax.set_ylabel("Pairmate score", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)

    network_handles = [
        Line2D([0], [0], color="black", linewidth=1, linestyle="-", label="Autoencoder"),
        Line2D([0], [0], color="black", linewidth=1, linestyle="--", label="Predictive network"),
    ]
    gamma_handles = [
        Line2D([0], [0], color=gamma_colors[gamma], linewidth=1, linestyle="-", label=fr"$\gamma={gamma}$")
        for gamma in gammas
    ]
    legend_network = ax.legend(handles=network_handles, fontsize=6, bbox_to_anchor=(1.05, 1.0),
                                loc="upper left", frameon=False, handlelength=2.0)
    ax.add_artist(legend_network)
    ax.legend(handles=gamma_handles, fontsize=6, bbox_to_anchor=(1.05, 0.05),
              loc="lower left", frameon=False, handlelength=2.0)

    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "pairmate_networks_multigamma.pdf"),
                dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def fig_regions(out_dir, seed=0):
    """associative-repulsion-1/regions.pdf: integration/differentiation/
    repulsion regions in the (rho_sA, rho_sB) plane, gamma=0, for the
    autoencoder and predictive network."""
    a = np.linspace(0, 1, 400)
    b0 = b_auto_zero(a)

    fig_size_mm = (45, 16)
    fig, axs = plt.subplots(1, 2, figsize=(fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4),
                             gridspec_kw={"wspace": 0.8})

    # Autoencoder: integration (b<a) / differentiation (a<b<b0) / repulsion (b>b0).
    axs[0].fill_between(a, 0, a, color=INTEGRATION_COLOR, linewidth=0)
    axs[0].fill_between(a, a, b0, color=DIFFERENTIATION_COLOR, linewidth=0)
    axs[0].fill_between(a, b0, 1, color=REPULSION_COLOR, linewidth=0)
    axs[0].set_title("Autoencoder", fontsize=7)

    # Predictive network: differentiation (b<a) / integration (b>a), no repulsion region.
    axs[1].fill_between(a, 0, a, color=DIFFERENTIATION_COLOR, linewidth=0)
    axs[1].fill_between(a, a, 1, color=INTEGRATION_COLOR, linewidth=0)
    axs[1].set_title("Pred. network", fontsize=7)

    for ax in axs:
        ax.set_xlabel(r"$\rho_{s,A}$", fontsize=7)
        ax.set_ylabel(r"$\rho_{s,B}$", fontsize=7)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)

    fig.savefig(os.path.join(out_dir, "associative-repulsion-1", "regions.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_D1_D2_curves_and_heatmaps(out_dir, seed=0):
    """associative-repulsion-2/plot{1,2,3,4}_*: D1 (cosine distance to
    ground truth) and D2 (pairmate bias) as curves vs. rho_sB and as
    heatmaps over (rho_sA, rho_sB), with trained-network overlays."""
    rho_sA_fixed = 0.0
    phi_in_values = [0.25, 0.5, 0.75, 1.0]
    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(phi_in_values)))
    purples = plt.cm.Purples(np.linspace(0.4, 0.95, len(phi_in_values)))

    rho_sB_curve = np.linspace(0.0, 1.0, 100)
    rho_sB_sim = np.linspace(0.0, 1.0, 7)
    gamma_sim = 0.05

    sim_D1, sim_D2 = {}, {}
    for phi_in in phi_in_values:
        d1s, d2s = [], []
        for rB in rho_sB_sim:
            d1, d2, err = simulate_D1_D2(rho_sA_fixed, rB, phi_in, gamma_sim, n_steps=200_000, seed=seed)
            d1s.append(d1)
            d2s.append(d2)
        sim_D1[phi_in] = np.array(d1s)
        sim_D2[phi_in] = np.array(d2s)
        print(f"phi_in={phi_in}: simulations done")

    # 1) D1 vs rho_sB, colored by phi_in, with simulation overlay.
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for phi_in, color in zip(phi_in_values, blues):
        D1_curve = np.array([D1_D2(rho_sA_fixed, rB, phi_in)[0] for rB in rho_sB_curve])
        ax1.plot(rho_sB_curve, D1_curve, color=color, linewidth=2.2, label=fr"$\phi_{{in}}$={phi_in}")
        ax1.scatter(rho_sB_sim, sim_D1[phi_in], color=color, zorder=5, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel(r"$\rho_{s,B}$")
    ax1.set_ylabel(r"cosine distance to ground truth $1-\mathrm{corr}(\hat v_B,v_B)$")
    ax1.set_title(r"$\rho_{s,A}=0$: cosine distance vs. $\rho_{s,B}$")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "associative-repulsion-2", "plot1_D1_vs_rhoB.pdf"), dpi=140)
    plt.close(fig1)

    # 2) Heatmap of D1 over (rho_sA, rho_sB), phi_in=0.5, white -> black.
    phi_in_heat = 0.5
    eps = 1e-3
    grid_A = np.linspace(0.0, 1.0 - eps, 300)
    grid_B = np.linspace(0.0, 1.0 - eps, 300)
    RA, RB = np.meshgrid(grid_A, grid_B)
    D1_grid = np.vectorize(lambda a, b: D1_D2(a, b, phi_in_heat)[0])(RA, RB)
    white_to_black = LinearSegmentedColormap.from_list("white_black", ["white", "black"])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.pcolormesh(grid_A, grid_B, D1_grid, cmap=white_to_black, shading="auto", vmin=0)
    ax2.set_xlabel(r"$\rho_{s,A}$")
    ax2.set_ylabel(r"$\rho_{s,B}$")
    ax2.set_title(fr"$D_1$ (cosine distance to ground truth), $\phi_{{in}}$={phi_in_heat}")
    fig2.colorbar(im2, ax=ax2, label=r"$1-\mathrm{corr}(\hat v_B,v_B)$")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "associative-repulsion-2", "plot2_D1_heatmap.pdf"), dpi=140)
    plt.close(fig2)

    # 3) D2 vs rho_sB, colored by phi_in, with simulation overlay.
    figsize_mm = (20, 16)
    fig3, ax3 = plt.subplots(figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4))
    for phi_in, color in zip(phi_in_values, purples):
        D2_curve = np.array([D1_D2(rho_sA_fixed, rB, phi_in)[1] for rB in rho_sB_curve])
        ax3.plot(rho_sB_curve, D2_curve, color=color, linewidth=1., label=fr"{phi_in}")
        ax3.scatter(rho_sB_sim, sim_D2[phi_in], color=color, s=5)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax3.set_xlabel(r"$\rho_{s,B}$", fontsize=7)
    ax3.set_ylabel("Pairmate bias", fontsize=7)
    ax3.set_xticklabels(ax3.get_xticks(), fontsize=6)
    ax3.set_yticklabels(ax3.get_yticks(), fontsize=6)
    ax3.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
               title="Input A\nsalience ($\\phi_{in}$)", title_fontsize=7, borderaxespad=0, frameon=False)
    fig3.savefig(os.path.join(out_dir, "associative-repulsion-2", "plot3_D2_vs_rhoB.pdf"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # 4) Heatmap of D2 over (rho_sA, rho_sB), phi_in=0.5, diverging red/blue.
    D2_grid = np.vectorize(lambda a, b: D1_D2(a, b, phi_in_heat)[1])(RA, RB)
    vmax = np.nanmax(np.abs(D2_grid))

    fig4, ax4 = plt.subplots(figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4))
    im4 = ax4.pcolormesh(grid_A, grid_B, D2_grid, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
    im4.set_rasterized(True)
    ax4.set_xlabel(r"$\rho_{s,A}$", fontsize=7)
    ax4.set_ylabel(r"$\rho_{s,B}$", fontsize=7)
    ax4.set_xticklabels(ax4.get_xticks(), fontsize=6)
    ax4.set_yticklabels(ax4.get_yticks(), fontsize=6)
    cbar = fig4.colorbar(im4, ax=ax4, label="Pairmate bias")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pairmate bias", fontsize=7)
    fig4.savefig(os.path.join(out_dir, "associative-repulsion-2", "plot4_D2_heatmap.pdf"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig4)


FIGURES = {
    "associative-network-traj": fig_associative_network_traj,
    "predictive-network-traj": fig_predictive_network_traj,
    "repulsion-heatmap": fig_repulsion_heatmap_phi,
    "repulsion-heatmap-rho0.5": fig_repulsion_heatmap_phi_rho05,
    "threshold-plot": fig_threshold_plot,
    "pairmate-vs-gamma": fig_pairmate_vs_gamma,
    "pairmate-networks-multigamma": fig_pairmate_networks_multigamma,
    "regions": fig_regions,
    "D1-D2-curves-heatmaps": fig_D1_D2_curves_and_heatmaps,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", choices=list(FIGURES.keys()), default=None,
                         help="Generate only these figures (default: all).")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.figures_dir, "associative-repulsion-1"), exist_ok=True)
    os.makedirs(os.path.join(args.figures_dir, "associative-repulsion-2"), exist_ok=True)

    for name in args.only or list(FIGURES.keys()):
        print(f"=== generating figure: {name} ===")
        FIGURES[name](args.figures_dir, seed=args.seed)


if __name__ == "__main__":
    main()

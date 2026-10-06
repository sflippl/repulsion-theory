"""Gradient-descent ground-truth simulators for cross-checking the closed-form
theory (Section 3: the multi-stream mode solver; Section 5: the last-hidden
representation formula), plus network initializers.

No autograd is used: gradients follow the explicit chain-rule formula for a
product-of-matrices deep linear network, exactly as given in the spec:

    dot(Z_l) = eta_j * Z_{1:l-1}^T @ R @ Z_{l+1:L}^T,
    R = X^(j)T (Y - sum_j X^(j) Z_{1:L_j}^(j))

Two initializations are supported for every stream: "exact" (the theory's own
idealized init -- hidden layers = gamma*I_n, readout = 0) and "random" (small
random weights), so predictions can be checked for robustness to init choice.

This is deliberately the *reduced* (n x n matrix) system, not a literal wide
network: random orthogonal U (shared "item" basis across all streams and the
target), per-stream V_j, and V_Y play the role of the wide network's SVD
bases (the "Lemma-1-equivalent reduction" the spec assumes already holds).
Gradient descent from an exact (basis-independent) identity-like init is
expected to converge to the mode-decoupled solution regardless of the random
choice of U/V_j/V_Y.

This module intentionally does NOT assert strict agreement with the
closed-form formulas anywhere (no pytest dependency on convergence); it is a
tool for exploratory/notebook comparison, per-project decision to keep the
full 1e5-6e5 step "required" ground-truth check out of the default test
suite for now.
"""
from __future__ import annotations

import numpy as np


def init_exact_weights(gamma: float, n: int, L: int):
    """Hidden layers = gamma * I_n, readout (last layer) = 0. All (n, n)."""
    weights = [gamma * np.eye(n) for _ in range(L - 1)]
    weights.append(np.zeros((n, n)))
    return weights


def init_random_weights(n: int, L: int, rng: np.random.Generator, scale: float = 1e-2):
    """Small random weights -- robustness check against the idealized init."""
    return [rng.normal(scale=scale, size=(n, n)) for _ in range(L)]


def _chain_product(weights):
    """W_1 @ W_2 @ ... @ W_L."""
    prod = weights[0]
    for w in weights[1:]:
        prod = prod @ w
    return prod


def _prefix_suffix(weights, l):
    """(W_1...W_{l-1}, W_{l+1}...W_L), identity if the corresponding slice is empty."""
    n = weights[0].shape[0]
    prefix = np.eye(n)
    for w in weights[:l]:
        prefix = prefix @ w
    suffix = np.eye(n)
    for w in weights[l + 1:]:
        suffix = suffix @ w
    return prefix, suffix


def _gd_step(weights_per_stream, Xs, Y, lr_etas):
    """One gradient-descent step (in place) across all streams; returns the loss."""
    J = len(weights_per_stream)
    pred = sum(Xs[j] @ _chain_product(weights_per_stream[j]) for j in range(J))
    err = Y - pred
    updates = []
    for j in range(J):
        R = Xs[j].T @ err
        Lj = len(weights_per_stream[j])
        layer_updates = []
        for l in range(Lj):
            prefix, suffix = _prefix_suffix(weights_per_stream[j], l)
            layer_updates.append(lr_etas[j] * (prefix.T @ R @ suffix.T))
        updates.append(layer_updates)
    for j in range(J):
        for l in range(len(weights_per_stream[j])):
            weights_per_stream[j][l] = weights_per_stream[j][l] + updates[j][l]
    return 0.5 * float(np.sum(err ** 2))


def simulate_mode_system(sigmas, etas, gammas, Ls, sigma_Y, n: int = 6, lr: float = 0.02,
                          n_steps: int = 20000, init: str = "exact", rng=None,
                          random_scale: float = 1e-2, probe_stream=None, probe_c=None):
    """Ground-truth check for Section 3 (and, optionally, Section 5).

    Builds an explicit multi-stream system on random orthogonal n x n bases:
    stream j's X^(j) = U @ diag(sigmas[j]) @ V_j^T (U shared across streams
    and Y), Y = U @ diag(sigma_Y) @ V_Y^T. Runs gradient descent and returns
    {"ps": [...], "Bs": [...], "loss": final loss} to compare against
    `solve_mode`.

    If `probe_stream` (int) and `probe_c` (= (c12, c3)) are given, also builds
    a probe Z_test = U @ diag([c12]*(n//2) + [c3]*(n - n//2)) @ V_probe^T
    sharing stream `probe_stream`'s own U (item basis), computes
    H_l = Z_test @ (product of stream `probe_stream`'s first L_j - 1 layers),
    and returns its Gram eigenvalues (mean over each mode-block) as
    "h_eig_12" / "h_eig_3" for exploratory comparison against
    `representation_a123`'s theoretical (lambda12_H, lambda3_H). This is not
    asserted anywhere -- it is meant for notebook-side comparison.
    """
    if rng is None:
        rng = np.random.default_rng()
    J = len(sigmas)

    U = np.linalg.qr(rng.normal(size=(n, n)))[0]
    Vs = [np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(J)]
    V_Y = np.linalg.qr(rng.normal(size=(n, n)))[0]

    Xs = [U @ np.diag(np.full(n, float(sigmas[j]))) @ Vs[j].T for j in range(J)]
    Y = U @ np.diag(np.full(n, float(sigma_Y))) @ V_Y.T

    weights = []
    for j in range(J):
        if init == "exact":
            weights.append(init_exact_weights(gammas[j], n, Ls[j]))
        elif init == "random":
            weights.append(init_random_weights(n, Ls[j], rng, scale=random_scale))
        else:
            raise ValueError(f"unknown init mode: {init!r}")

    lr_etas = [lr * eta for eta in etas]
    loss = None
    for _ in range(n_steps):
        loss = _gd_step(weights, Xs, Y, lr_etas)

    ps = []
    Bs_final = []
    for j in range(J):
        beta_j = _chain_product(weights[j])
        p_diag = np.diag(Vs[j].T @ beta_j @ V_Y)
        ps.append(float(np.mean(p_diag)))
        Bs_final.append(None)  # B is not directly observable from beta_j alone

    result = {"ps": ps, "loss": loss}

    if probe_stream is not None and probe_c is not None:
        c12, c3 = probe_c
        j = probe_stream
        l = Ls[j] - 1
        if l > 0:
            n_12 = n // 2
            diag_c = np.concatenate([np.full(n_12, c12), np.full(n - n_12, c3)])
            V_probe = np.linalg.qr(rng.normal(size=(n, n)))[0]
            Z_test = U @ np.diag(diag_c) @ V_probe.T
            partial = weights[j][0]
            for w in weights[j][1:l]:
                partial = partial @ w
            H_l = Z_test @ partial
            gram_eigs = np.linalg.eigvalsh(H_l @ H_l.T)
            result["h_eig_12"] = float(np.mean(gram_eigs[-n_12:])) if n_12 > 0 else None
            result["h_eig_3"] = float(np.mean(gram_eigs[:n - n_12])) if n - n_12 > 0 else None

    return result

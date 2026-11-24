"""Main figure on associate-driven repulsion: fig-associative-repulsion-1-v2.

Rough layout sketch (150 mm wide, as fig-associative-repulsion-1); panels are
placed at absolute mm positions (PANELS) so they are easy to move.

    A  Favila et al. schematic and results               placeholder
    B  complete model (Favila 2016 simulations): pairmate score of the MSP and
       TSP networks per condition (data/favila2016/summed/complete_model_linear;
       placeholder until that sweep exists)
    C  task schema                                       placeholder
    D  predictive network: hidden similarity over training (+ schema placeholder)
    E  autoencoder: hidden similarity over training (+ schema placeholder)
    F  pairmate score vs. initial weight variance gamma (theory + simulations)
    G  autoencoder pairmate score vs. rho_B, gamma -> 0, 0.01, 0.1 (theory +
       simulations)
    H  autoencoder pairmate score over training, two panels with their own
       y-scales: correlated vectors (ReLU) | CIFAR-100 (ReLU)
       (data/ab_autoencoder/{relu_corr,relu_cifar})

gamma is the initial weight variance (W(0)^T W(0) = gamma * I) throughout.
D and E rerun the 100-step trajectory simulation of the previous version. The
simulation dots of F and G (two-layer linear networks trained to convergence,
as in generate_associative_repulsion_figures.py) are cached in SIM_CACHE and
only computed when missing (about a minute); delete the file to recompute.

Writes figures/associative-repulsion-1/fig-associative-repulsion-1-v2.{pdf,png}.

Usage:
    python scripts/plot_fig_associative_repulsion_1_v2.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import generate_associative_repulsion_figures as theory
import plot_ab_autoencoding as ab
from plot_common import (FONTSIZE, LINEWIDTH, TICK_FONTSIZE, find_runs, load_config, load_rows, load_steps,
                         mean_sem, plot_conditions, setup_style)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_STEM = os.path.join(ROOT, "figures", "associative-repulsion-1", "fig-associative-repulsion-1-v2")
FIG_MM = (150, 116)
MM = 1 / 25.4

# (x, y, width, height) in mm, y measured from the top of the figure
PANELS = {
    "A": (5, 6, 68, 34),
    "B_msp": (88, 8, 26, 26),
    "B_tsp": (122, 8, 26, 26),
    "C": (5, 46, 30, 30),
    "D": (48, 50, 28, 24),
    "D_schema": (78, 50, 14, 24),
    "E": (106, 50, 28, 24),
    "E_schema": (136, 50, 14, 24),
    "F": (13, 86, 28, 22),
    "G": (56, 86, 28, 22),
    "H_vec": (102, 86, 19, 22),
    "H_cifar": (129, 86, 19, 22),
}
TAG_OFFSET_MM = (-10, -2)  # panel tag position relative to the panel's top-left corner
TAGS = {"A": "A", "B_msp": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H_vec": "H"}

TRAJ_GAMMA = 0.1                  # D, E
G_GAMMAS = [0.0, 0.01, 0.1]       # G (0 = gamma -> 0 limit)
GAMMA_COLORS = {0.0: "#66C266", 0.01: "#2E8B57", 0.1: "#006400"}  # as pairmate_networks_multigamma
NETWORK_COLORS = {"Pred. network": "tab:blue", "Autoencoder": "tab:red"}
FAVILA_DIR = os.path.join(ROOT, "data", "favila2016", "summed", "complete_model_linear")
H_PANELS = [("H_vec", "relu_corr", "Vectors", "#984EA3"),      # purple
            ("H_cifar", "relu_cifar", "CIFAR-100", "#3F007D")]  # dark purple
SIM_CACHE = os.path.join(ROOT, "data", "fig-associative-repulsion-1-v2", "simulations.json")
F_GAMMAS_SIM = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
G_RHO_B_SIM = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
GAMMA_ZERO_SIM = 1e-4  # simulations stand in for gamma -> 0
NETWORK_PHIS = {"Pred. network": (1.0, 0.0), "Autoencoder": (0.5, 0.5)}


# ====================================================================== #
# Layout helpers
# ====================================================================== #

def add_axes(fig, key):
    x, y, w, h = PANELS[key]
    return fig.add_axes([x / FIG_MM[0], 1 - (y + h) / FIG_MM[1], w / FIG_MM[0], h / FIG_MM[1]])


def placeholder(fig, key, text):
    ax = add_axes(fig, key)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f4f4f4",
                           edgecolor="#9a9a9a", linestyle="--", linewidth=0.6))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=TICK_FONTSIZE,
            color="#6a6a6a", wrap=True)
    ax.set_axis_off()
    return ax


def style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=1)
    if title:
        ax.set_title(title, fontsize=FONTSIZE, style="italic", pad=2)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1, width=0.6, length=2.5)
    ax.tick_params(axis="both", which="minor", width=0.4, length=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def legend(ax, handles, **kw):
    kw = {"fontsize": TICK_FONTSIZE, "frameon": False, "handlelength": 1.4, "borderaxespad": 0.2,
          "labelspacing": 0.25, **kw}
    ax.legend(handles=handles, **kw)


def line(color, label, ls="-"):
    return Line2D([0], [0], color=color, lw=LINEWIDTH, linestyle=ls, label=label)


def zero_line(ax):
    ax.axhline(0, color="0.35", linestyle=":", lw=0.9, zorder=1)


# ====================================================================== #
# Panels
# ====================================================================== #

def panel_b(fig):
    """Complete model on Favila et al. (2016): pairmate score per condition, MSP and TSP network."""
    import plot_favila2016 as favila

    try:
        results = []
        for d in find_runs(FAVILA_DIR, "representations_steps.npy"):
            cfg = load_config(d)
            epochs, _ = favila.steps_to_epochs(d, cfg, load_steps(d, "representations"))
            rows = load_rows(d, "representations")
            res = {"epochs": epochs}
            for ni in range(2):
                reps = np.load(os.path.join(d, f"representations_net{ni}_hidden_0.npy"))
                res[("pairmate", ni)] = favila.pairmate_score(reps, rows)
            results.append(res)
    except (OSError, ValueError, KeyError) as err:  # sweep missing or still being copied
        print(f"Panel B: placeholder ({err})")
        for key, name in [("B_msp", "MSP network"), ("B_tsp", "TSP network")]:
            placeholder(fig, key, f"Complete model\n{name}\n(Favila data pending)")
        return
    x = results[0]["epochs"]
    for ni, (key, title) in enumerate([("B_msp", "MSP network"), ("B_tsp", "TSP network")]):
        ax = add_axes(fig, key)
        zero_line(ax)
        plot_conditions(ax, x, results, ("pairmate", ni), favila.CONDITIONS, favila.COLORS)
        ax.set_xlim(x[0], x[-1])
        style(ax, "Epoch", "Pairmate score" if ni == 0 else None, title)
        if ni == 1:
            legend(ax, [line(favila.COLORS[c], favila.LABELS[c]) for c in favila.CONDITIONS],
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))


def trajectory(phi_in, phi_out, gamma, n_steps=100, lr=0.1, seed=0):
    """Hidden similarity A1 vs. A2 (within) and A1 vs. A3 (across) over training, and closed form."""
    X, Y, Z, _, _ = theory.build_associative_repulsion_data(0.0, 1.0, phi_in, phi_out, 2, 2, 25)
    steps, within, across = [], [], []

    def on_track(step, model):
        with torch.no_grad():
            H = model.hidden(Z)
            C = H @ H.T
        steps.append(step)
        within.append((C[0, 1] / C[0, 0]).item())
        across.append((C[0, 2] / C[0, 0]).item())

    theory.train_network(X, Y, gamma, n_steps, lr, seed=seed, loss_threshold=None,
                         track_steps=set(range(n_steps + 1)), on_track=on_track, progress=False)
    a1, a2, a3 = theory.learnedreps_associative_repulsion(0.0, 1.0, phi_in, phi_out, gamma)
    return np.array(steps), np.array(within), np.array(across), a2 / a1, a3 / a1


def panel_trajectory(fig, key, title, phi_in, phi_out, show_legend):
    steps, within, across, within_cf, across_cf = trajectory(phi_in, phi_out, TRAJ_GAMMA)
    ax = add_axes(fig, key)
    ax.plot(steps, within, color="black", lw=LINEWIDTH)
    ax.plot(steps, across, color="tab:orange", lw=LINEWIDTH)
    ax.axhline(within_cf, color="black", linestyle="--", alpha=0.6, lw=0.8)
    ax.axhline(across_cf, color="tab:orange", linestyle="--", alpha=0.6, lw=0.8)
    ax.set_xlim(0, steps[-1])
    style(ax, "Steps", "Similarity", title)
    if show_legend:
        legend(ax, [line("black", "A1 vs. A2"), line("tab:orange", "A1 vs. A3")], loc="center right")
    placeholder(fig, f"{key}_schema", "schema")


def simulate_score(rho_B, phi_in, phi_out, gamma, n_steps=300_000, lr=0.02, seed=0):
    """Pairmate score (a2 - a3) / a1 of a two-layer linear network trained to convergence (rho_A = 0)."""
    X, Y, Z, _, _ = theory.build_associative_repulsion_data(0.0, rho_B, phi_in, phi_out, 2, 2, 25)
    model = theory.train_network(X, Y, gamma, n_steps, lr, seed=seed, progress=False)
    with torch.no_grad():
        H = model.hidden(Z)
        C = H @ H.T
    return ((C[0, 1] - C[0, 2]) / C[0, 0]).item()


def simulations() -> dict:
    """Simulation dots of F and G, keyed "F|<network>|<gamma>" and "G|<gamma>|<rho_B>" (cached)."""
    cache = json.load(open(SIM_CACHE)) if os.path.exists(SIM_CACHE) else {}
    todo = {f"F|{net}|{g:g}": (1.0, *NETWORK_PHIS[net], g) for net in NETWORK_PHIS for g in F_GAMMAS_SIM}
    todo.update({f"G|{g:g}|{r:g}": (r, 0.5, 0.5, g if g > 0 else GAMMA_ZERO_SIM)
                 for g in G_GAMMAS for r in G_RHO_B_SIM})
    missing = {k: v for k, v in todo.items() if k not in cache}
    if missing:
        print(f"Simulating {len(missing)} networks for F and G ...")
        for key, args in missing.items():
            cache[key] = simulate_score(*args)
        os.makedirs(os.path.dirname(SIM_CACHE), exist_ok=True)
        json.dump(cache, open(SIM_CACHE, "w"), indent=1)
    return cache


def panel_f(fig, sims):
    gammas = np.geomspace(1e-4, 10.0, 400)
    ax = add_axes(fig, "F")
    zero_line(ax)
    ax.plot(gammas, theory.PS_PN(gammas), color=NETWORK_COLORS["Pred. network"], lw=LINEWIDTH)
    ax.plot(gammas, theory.PS_AE(gammas), color=NETWORK_COLORS["Autoencoder"], lw=LINEWIDTH)
    for net, color in NETWORK_COLORS.items():
        ax.scatter(F_GAMMAS_SIM, [sims[f"F|{net}|{g:g}"] for g in F_GAMMAS_SIM], color=color, s=9, zorder=3,
                   linewidths=0)
    ax.set_xscale("log")
    ax.set_xlim(gammas[0], gammas[-1])
    style(ax, r"Weight variance $\gamma$", "Pairmate score")
    legend(ax, [line(c, n) for n, c in NETWORK_COLORS.items()], loc="center left")


def panel_g(fig, sims):
    rho_B = np.linspace(0.0, 0.999, 300)
    ax = add_axes(fig, "G")
    zero_line(ax)
    for gamma in G_GAMMAS:
        y = (theory.score_autoencoder(0.0, rho_B) if gamma == 0
             else theory.pairmate_score(0.0, rho_B, 0.5, 0.5, gamma))
        ax.plot(rho_B, y, color=GAMMA_COLORS[gamma], lw=LINEWIDTH)
        ax.scatter(G_RHO_B_SIM, [sims[f"G|{gamma:g}|{r:g}"] for r in G_RHO_B_SIM], color=GAMMA_COLORS[gamma],
                   s=9, zorder=3, linewidths=0)
    ax.set_xlim(-0.03, 1.03)
    style(ax, r"B correlation $\rho_B$", "Pairmate score", "Autoencoder")
    labels = {0.0: r"$\gamma\to0$", 0.01: r"$\gamma$ = 0.01", 0.1: r"$\gamma$ = 0.1"}
    legend(ax, [line(GAMMA_COLORS[g], labels[g]) for g in G_GAMMAS], loc="lower left")


def panel_h(fig):
    """Autoencoder pairmate score over training; one panel per setup, each with its own y-scale."""
    for c, (key, setup, title, color) in enumerate(H_PANELS):
        ax = add_axes(fig, key)
        zero_line(ax)
        runs = ab.seed_dirs(os.path.join(ROOT, "data", "ab_autoencoder", setup))
        if runs:
            per_run = [ab.similarities(d, "a") for d in runs]
            x = np.maximum(per_run[0][0], 1)  # log axis: step 0 (initialization) at 1
            mean, sem = mean_sem([sims["within"] - sims["across"] for _, sims in per_run])
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)
            ax.plot(x, mean, color=color, lw=LINEWIDTH)
            ax.set_xlim(1, x[-1])
        ax.set_xscale("log")
        ax.set_xticks([1, 1e2])
        style(ax, "Steps", "Pairmate score" if c == 0 else None, title)


# ====================================================================== #

def main():
    setup_style()
    fig = plt.figure(figsize=(FIG_MM[0] * MM, FIG_MM[1] * MM))
    placeholder(fig, "A", "Favila et al. (2016)\nschematic and results")
    panel_b(fig)
    placeholder(fig, "C", "Task schema")
    panel_trajectory(fig, "D", "Pred. network", 1.0, 0.0, show_legend=True)
    panel_trajectory(fig, "E", "Autoencoder", 0.5, 0.5, show_legend=False)
    sims = simulations()
    panel_f(fig, sims)
    panel_g(fig, sims)
    panel_h(fig)
    for key, tag in TAGS.items():
        x, y, _, _ = PANELS[key]
        fig.text(max(x + TAG_OFFSET_MM[0], 0) / FIG_MM[0], 1 - (y + TAG_OFFSET_MM[1]) / FIG_MM[1],
                 tag, fontsize=FONTSIZE + 2, fontweight="bold", va="bottom", ha="left")
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=300)
        print(f"Saved {OUT_STEM}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()

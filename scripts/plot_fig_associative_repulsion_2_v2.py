"""Figure on behavioral repulsion: fig-associative-repulsion-2-v2.

Layout sketch (150 mm wide, as fig-associative-repulsion-2); panels are placed
at absolute mm positions (PANELS) so they are easy to move.

    A  Chanales et al. schema                            placeholder
    B  model replication: response away from the competitor (%) at epoch
       B_EPOCH per similarity condition, Exp. 1 (data/chanales2021) and Exp. 4
       (data/chanales2021_exp4, low/mod./high only, as in the paper); summed
       objective, combined read-out, mean +- SEM over seeds
    C  task setup                                        placeholder
    D  behavioral repulsion angle vs. rho_B, autoencoder (phi_in = 0.5,
       rho_A = 0): theory (gamma -> 0) and simulations
    E  behavioral repulsion angle over (rho_A, rho_B), autoencoder (theory)
    F  behavioral repulsion angle at the end of training (of the B read out from
       A alone, projected onto the plane of the pairmates' B; --f-metric score for
       the theory's score units instead), ReLU predictive
       network (triangles) and autoencoder (circles): vectors with rho_B = 0.9
       (data/ab_<task>/relu_corr_behavioral) and CIFAR-100 with B pairmates = the
       same image rotated by 90 degrees (relu_cifar_behavioral); mean +- SEM over seeds

Behavioral repulsion score = rho_B - corr(B_hat, B_pairmate), with B_hat the B
read out from A alone (positive = repulsion). D-F show it as the equivalent signed
angle (degrees) by which B_hat is rotated away from the pairmate's B within the
plane of B and B_pairmate (positive = repulsion); panel B stays in percent. The simulation dots of D (two-layer
linear networks, gamma = D_GAMMA_SIM, trained to convergence) are cached in
SIM_CACHE and only computed when missing; delete the file to recompute.

Writes figures/associative-repulsion-2/fig-associative-repulsion-2-v2.{pdf,png}.

Usage:
    python scripts/plot_fig_associative_repulsion_2_v2.py
    python scripts/plot_fig_associative_repulsion_2_v2.py --f-metric score   # F in the theory's score units
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import generate_associative_repulsion_figures as theory
import plot_ab_autoencoding as ab
import plot_chanales2021 as chanales
from plot_common import FONTSIZE, LINEWIDTH, TICK_FONTSIZE, find_runs, mean_sem, setup_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_STEM = os.path.join(ROOT, "figures", "associative-repulsion-2", "fig-associative-repulsion-2-v2")
FIG_MM = (150, 84)
MM = 1 / 25.4

# (x, y, width, height) in mm, y measured from the top of the figure
PANELS = {
    "A": (5, 6, 62, 30),
    "B_exp1": (86, 8, 26, 24),
    "B_exp4": (124, 8, 19.5, 24),
    "C": (5, 46, 28, 30),
    "D": (44, 50, 22, 22),
    "E": (77, 50, 19, 22),
    "E_cbar": (98, 50, 1.8, 22),
    "F": (125, 50, 23, 22),
}
TAG_OFFSET_MM = (-10, -2)
TAGS = {"A": "A", "B_exp1": "B", "C": "C", "D": "D", "E": "E", "F": "F"}

B_EPOCH = 8
B_EXPERIMENTS = [("B_exp1", "Exp. 1", "chanales2021", chanales.CONDITIONS),
                 ("B_exp4", "Exp. 4", "chanales2021_exp4", chanales.CONDITIONS[:3])]
PHI_IN_AE = 0.5
D_COLOR = "tab:red"  # autoencoder, as in fig-associative-repulsion-1-v2
D_RHO_B_SIM = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
D_GAMMA_SIM = 0.01
SIM_CACHE = os.path.join(ROOT, "data", "fig-associative-repulsion-2-v2", "simulations.json")
ANGLE_LABEL = "Behavioral\nrepulsion (°)"
F_SETUPS = [("relu_corr_behavioral", "Vectors", "#984EA3"),   # rho_B = 0.9; purple, as fig 1 v2 panel H
            ("relu_cifar_behavioral", "CIFAR", "#3F007D")]   # B: same image rotated by 90 degrees; dark purple
# (task, output probe with the B read out from A alone, label, marker)
F_NETWORKS = [("ab_prediction", "output_prediction", "Pred. network", "^"),
              ("ab_autoencoder", "output_a_to_ab", "Autoencoder", "o")]


# ====================================================================== #
# Layout helpers (as in plot_fig_associative_repulsion_1_v2.py)
# ====================================================================== #

def add_axes(fig, key):
    x, y, w, h = PANELS[key]
    return fig.add_axes([x / FIG_MM[0], 1 - (y + h) / FIG_MM[1], w / FIG_MM[0], h / FIG_MM[1]])


def placeholder(fig, key, text):
    ax = add_axes(fig, key)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f4f4f4",
                           edgecolor="#9a9a9a", linestyle="--", linewidth=0.6))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=TICK_FONTSIZE,
            color="#6a6a6a")
    ax.set_axis_off()


def style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=1)
    if title:
        ax.set_title(title, fontsize=FONTSIZE, style="italic", pad=2)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1, width=0.6, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def reference(ax, y):
    ax.axhline(y, color="0.35", linestyle=":", lw=0.9, zorder=1)


# ====================================================================== #
# Panels
# ====================================================================== #

def panel_b(fig):
    """Response away from the competitor (%) at epoch B_EPOCH per similarity condition."""
    axes = []
    for c, (key, title, experiment, conditions) in enumerate(B_EXPERIMENTS):
        sweep = os.path.join(ROOT, "data", experiment, "summed", "complete_model_linear")
        values = {cond: [] for cond in conditions}
        for d in find_runs(sweep, "prediction_steps.npy"):
            res = chanales.compute_run(d, ["combined"])
            t = int(np.argmin(np.abs(res["x"] - B_EPOCH)))
            for cond in conditions:
                values[cond].append([100 * res[("away", "combined")][cond][t]])
        ax = add_axes(fig, key)
        reference(ax, 50)
        for x, cond in enumerate(conditions):
            mean, sem = mean_sem(values[cond])
            ax.errorbar(x, mean[0], yerr=sem[0], color=chanales.C21_COLORS[cond], marker="o", markersize=3,
                        elinewidth=0.5, capsize=1.2, capthick=0.5, linestyle="none")
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([chanales.C21_LABELS[cond] for cond in conditions])
        style(ax, "Similarity", "Response away from\ncompetitor (%)" if c == 0 else None, title)
        axes.append(ax)
        print(f"Panel B, {title}: {len(values[conditions[0]])} seeds, epoch {B_EPOCH}")
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def simulations() -> dict:
    """Simulated behavioral repulsion score of D, keyed by rho_B (cached)."""
    cache = json.load(open(SIM_CACHE)) if os.path.exists(SIM_CACHE) else {}
    missing = [r for r in D_RHO_B_SIM if f"{r:g}" not in cache]
    if missing:
        print(f"Simulating {len(missing)} networks for D ...")
        for r in missing:
            cache[f"{r:g}"] = theory.simulate_D1_D2(0.0, r, PHI_IN_AE, D_GAMMA_SIM, n_steps=200_000)[1]
        os.makedirs(os.path.dirname(SIM_CACHE), exist_ok=True)
        json.dump(cache, open(SIM_CACHE, "w"), indent=1)
    return cache


def panel_d(fig, sims):
    # dense near rho_B = 1, where the angle falls to 0 like sqrt(2 (1 - rho_B))
    rho_B = np.concatenate([np.linspace(0.0, 0.99, 200), 1 - np.geomspace(1e-2, 1e-8, 200), [1.0]])
    ax = add_axes(fig, "D")
    reference(ax, 0)
    ax.plot(rho_B, theory.behavioral_repulsion_angle(0.0, rho_B, PHI_IN_AE), color=D_COLOR, lw=LINEWIDTH)
    # simulations store the score; convert to the angle (exact for read-outs projected onto the B plane)
    ax.scatter(D_RHO_B_SIM, [theory.score_to_angle(sims[f"{r:g}"], r) for r in D_RHO_B_SIM], color=D_COLOR,
               s=9, zorder=3, linewidths=0)
    ax.set_xlim(-0.03, 1.03)
    style(ax, r"B correlation $\rho_B$", ANGLE_LABEL, "Autoencoder")


def panel_e(fig):
    grid = np.linspace(0.0, 0.999, 300)
    RA, RB = np.meshgrid(grid, grid)
    angle = theory.behavioral_repulsion_angle(RA, RB, PHI_IN_AE)
    vmax = np.nanmax(np.abs(angle))
    ax = add_axes(fig, "E")
    im = ax.pcolormesh(grid, grid, angle, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax, rasterized=True)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    style(ax, r"A correlation $\rho_A$", r"B correlation $\rho_B$", "Autoencoder")
    cax = add_axes(fig, "E_cbar")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Behavioral repulsion (°)", fontsize=FONTSIZE, labelpad=2)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE, width=0.6, length=2)
    cbar.outline.set_linewidth(0.6)


def panel_f(fig, metric="angle"):
    """Behavioral repulsion (signed angle in degrees, or metric="score") at the end of training, ReLU networks:
    one x position per setup (F_SETUPS), one marker per network (F_NETWORKS), mean +- SEM over seeds."""
    from matplotlib.lines import Line2D

    ax = add_axes(fig, "F")
    reference(ax, 0)
    for x, (setup, _, color) in enumerate(F_SETUPS):
        for k, (task, probe, _, marker) in enumerate(F_NETWORKS):
            sweep = os.path.join(ROOT, "data", task, setup)
            runs = sorted(d for d in ab.glob.glob(os.path.join(sweep, "*"))
                          if os.path.exists(os.path.join(d, f"{probe}_combined_B.npy")))
            if not runs:
                continue
            mean, sem = mean_sem([[ab.behavioral_scores(d, probe, metric)[1][-1]] for d in runs])
            ax.errorbar(x + (k - 0.5) * 0.3, mean[0], yerr=sem[0], color=color, marker=marker, markersize=3.5,
                        elinewidth=0.5, capsize=1.2, capthick=0.5, linestyle="none")
            print(f"Panel F, {task}/{setup}: {mean[0]:+.3f} ({metric}, {len(runs)} seeds)")
    ax.set_xlim(-0.6, len(F_SETUPS) - 0.4)
    ax.set_xticks(range(len(F_SETUPS)))
    ax.set_xticklabels([label for _, label, _ in F_SETUPS])
    style(ax, None, ANGLE_LABEL if metric == "angle" else "Behavioral\nrepulsion score", "ReLU")
    handles = [Line2D([0], [0], color="0.3", marker=marker, markersize=3.5, linestyle="none", label=label)
               for _, _, label, marker in F_NETWORKS]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # headroom for the legend above the points
    ax.legend(handles=handles, loc="upper right", fontsize=TICK_FONTSIZE, frameon=False, handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.25)


# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--f-metric", choices=["angle", "score"], default="angle",
                        help="panel F: the signed angle in degrees (default) or the closed-form theory's "
                             "score (written to ...-v2-score.{pdf,png})")
    args = parser.parse_args()
    setup_style()
    fig = plt.figure(figsize=(FIG_MM[0] * MM, FIG_MM[1] * MM))
    placeholder(fig, "A", "Chanales et al. schema")
    panel_b(fig)
    placeholder(fig, "C", "Task setup")
    panel_d(fig, simulations())
    panel_e(fig)
    panel_f(fig, args.f_metric)
    for key, tag in TAGS.items():
        x, y, _, _ = PANELS[key]
        fig.text(max(x + TAG_OFFSET_MM[0], 0) / FIG_MM[0], 1 - (y + TAG_OFFSET_MM[1]) / FIG_MM[1],
                 tag, fontsize=FONTSIZE + 2, fontweight="bold", va="bottom", ha="left")
    out_stem = OUT_STEM + ("-score" if args.f_metric == "score" else "")
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
        print(f"Saved {out_stem}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()

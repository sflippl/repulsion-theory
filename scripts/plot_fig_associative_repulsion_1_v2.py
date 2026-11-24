"""Main figure on associate-driven repulsion: fig-associative-repulsion-1-v2.

150 mm wide (as fig-associative-repulsion-1). Plots are placed at absolute mm
positions (PANELS); the schematics (IMAGES, SVGs in figures/associative-repulsion-1/)
are inserted as vector graphics at their native size, never rescaled: matplotlib
writes the plots to an SVG (text kept as text), the schematics are nested into
it, and Inkscape exports the PDF (vector, text selectable) and PNG.

    A  Favila et al. setup                               favila_setup.svg
    B  Favila et al. data (hippocampus)                  favila_data.svg
    C  complete model (Favila 2016 simulations), TSP network: pairmate score per
       group (no / different / same face) at epoch C_EPOCH, mean +- SEM over
       seeds (data/favila2016/summed/complete_model_linear). The y-axis has the
       height and vertical position of B's y-axis (B_YAXIS_MM).
    D  task setup                                        task_setup.svg
    E  predictive network: hidden similarity over training + mechanism (pred_mech.svg)
    F  autoencoder: hidden similarity over training + mechanism (ae_mech.svg)
    G  pairmate score vs. initial weight variance gamma (theory + simulations)
    H  autoencoder pairmate score vs. rho_B, gamma -> 0, 0.01, 0.1 (theory +
       simulations)
    I  autoencoder pairmate score over training, two panels with their own
       y-scales: correlated vectors (ReLU) | CIFAR-100 (ReLU)
       (data/ab_autoencoder/{relu_corr,relu_cifar})

gamma is the initial weight variance (W(0)^T W(0) = gamma * I) throughout.
E and F rerun the 100-step trajectory simulation of the previous version. The
simulation dots of G and H (two-layer linear networks trained to convergence,
as in generate_associative_repulsion_figures.py) are cached in SIM_CACHE and
only computed when missing (about a minute); delete the file to recompute.

Writes figures/associative-repulsion-1/fig-associative-repulsion-1-v2.{svg,pdf,png}.

Usage:
    python scripts/plot_fig_associative_repulsion_1_v2.py
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import generate_associative_repulsion_figures as theory
import plot_ab_autoencoding as ab
from plot_common import (FONTSIZE, LINEWIDTH, TICK_FONTSIZE, find_runs, load_config, load_rows, load_steps,
                         mean_sem, plot_conditions, setup_style)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_STEM = os.path.join(ROOT, "figures", "associative-repulsion-1", "fig-associative-repulsion-1-v2")
SCHEMATIC_DIR = os.path.join(ROOT, "figures", "associative-repulsion-1")
FIG_MM = (150, 112)
MM = 1 / 25.4

# Schematics: name -> (file, x, y) in mm, y measured from the top of the figure; size from the file
IMAGES = {
    "A": ("favila_setup.svg", 5, 8),
    "B": ("favila_data.svg", 79, 4),
    "D": ("task_setup.svg", 5, 45.8),      # bottom aligned with the E/F axes (69 mm)
    "E_mech": ("pred_mech.svg", 59, 42),
    "F_mech": ("ae_mech.svg", 118.8, 42),  # right edge at 148 mm, as C and I
}
# y-axis of the plot in favila_data: top and length in mm (measured from the image)
B_YAXIS_MM = (3.81, 23.28)

# Plots: (x, y, width, height) in mm, y measured from the top of the figure
PANELS = {
    "C": (123, IMAGES["B"][2] + B_YAXIS_MM[0], 25, B_YAXIS_MM[1]),
    "E": (32, 48, 25, 21),
    "F": (88, 48, 25, 21),
    "G": (13, 82, 28, 22),
    "H": (56, 82, 28, 22),
    "I_vec": (102, 82, 19, 22),
    "I_cifar": (129, 82, 19, 22),
}
# Panel tags: (x, y) of the tag's lower-left corner in mm from the top-left of the figure
TAGS = {"A": (0, 4), "B": (75, 4), "C": (113, 4), "D": (0, 41), "E": (22, 41), "F": (78, 41),
        "G": (0, 79), "H": (46, 79), "I": (92, 79)}

TRAJ_GAMMA = 0.1                  # E, F
H_GAMMAS = [0.0, 0.01, 0.1]       # H (0 = gamma -> 0 limit)
GAMMA_COLORS = {0.0: "#66C266", 0.01: "#2E8B57", 0.1: "#006400"}  # as pairmate_networks_multigamma
NETWORK_COLORS = {"Pred. network": "tab:blue", "Autoencoder": "tab:red"}
FAVILA_DIR = os.path.join(ROOT, "data", "favila2016", "summed", "complete_model_linear")
C_EPOCH = 10
I_PANELS = [("I_vec", "relu_corr", "Vectors", "#984EA3"),      # purple
            ("I_cifar", "relu_cifar", "CIFAR-100", "#3F007D")]  # dark purple
SIM_CACHE = os.path.join(ROOT, "data", "fig-associative-repulsion-1-v2", "simulations.json")
G_GAMMAS_SIM = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
H_RHO_B_SIM = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
GAMMA_ZERO_SIM = 1e-4  # simulations stand in for gamma -> 0
NETWORK_PHIS = {"Pred. network": (1.0, 0.0), "Autoencoder": (0.5, 0.5)}


# ====================================================================== #
# Layout helpers
# ====================================================================== #

def add_axes(fig, key):
    x, y, w, h = PANELS[key]
    return fig.add_axes([x / FIG_MM[0], 1 - (y + h) / FIG_MM[1], w / FIG_MM[0], h / FIG_MM[1]])


SVG_UNITS_MM = {"": 25.4 / 96, "px": 25.4 / 96, "pt": 25.4 / 72, "mm": 1.0, "cm": 10.0, "in": 25.4}
PT_PER_MM = 72 / 25.4


def svg_root(text: str) -> tuple[str, str]:
    """The root <svg ...> start tag and the content inside the root element."""
    start = re.search(r"<svg\b[^>]*>", text)
    return start.group(0), text[start.end():text.rindex("</svg>")]


def svg_size_mm(path: str) -> tuple[float, float]:
    tag, _ = svg_root(open(path).read())
    size = []
    for attr in ("width", "height"):
        value, unit = re.fullmatch(r"([\d.]+)\s*([a-z]*)", re.search(fr'\b{attr}="([^"]+)"', tag).group(1)).groups()
        size.append(float(value) * SVG_UNITS_MM[unit])
    return tuple(size)


def nested_svg(key: str) -> str:
    """Schematic *key* as an <svg> element at its native size and IMAGES position (in the
    pt units of matplotlib's SVG output); its ids are prefixed so schematics cannot clash."""
    name, x, y = IMAGES[key]
    path = os.path.join(SCHEMATIC_DIR, name)
    tag, inner = svg_root(open(path).read())
    w, h = svg_size_mm(path)
    view_box = re.search(r'\bviewBox="([^"]+)"', tag).group(1)
    namespaces = " ".join(re.findall(r'xmlns:\w+="[^"]*"', tag))
    prefix = f"{key}-"
    inner = re.sub(r'\bid="([^"]+)"', fr'id="{prefix}\1"', inner)
    inner = re.sub(r"url\(#([^)]+)\)", fr"url(#{prefix}\1)", inner)
    inner = re.sub(r'href="#([^"]+)"', fr'href="#{prefix}\1"', inner)
    return (f'<svg x="{x * PT_PER_MM:.4f}" y="{y * PT_PER_MM:.4f}" width="{w * PT_PER_MM:.4f}" '
            f'height="{h * PT_PER_MM:.4f}" viewBox="{view_box}" preserveAspectRatio="none" {namespaces}>'
            f'{inner}</svg>')


def export_with_inkscape(svg_path: str, out_path: str, *args: str) -> bool:
    # this environment works around Inkscape crashing on GNOME settings in headless use
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env.update(XDG_CURRENT_DESKTOP="", GSETTINGS_BACKEND="memory", GDK_BACKEND="broadway")
    result = subprocess.run(["inkscape", f"--export-filename={out_path}", *args, svg_path],
                            env=env, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


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

def panel_c(fig):
    """Complete model on Favila et al. (2016): TSP network pairmate score per group at epoch C_EPOCH."""
    import plot_favila2016 as favila

    results = []
    for d in find_runs(FAVILA_DIR, "representations_steps.npy"):
        cfg = load_config(d)
        epochs, _ = favila.steps_to_epochs(d, cfg, load_steps(d, "representations"))
        reps = np.load(os.path.join(d, "representations_net1_hidden_0.npy"))  # net1 = TSP network
        results.append({"epochs": epochs, "pairmate": favila.pairmate_score(reps, load_rows(d, "representations"))})
    t = int(np.argmin(np.abs(results[0]["epochs"] - C_EPOCH)))
    print(f"Panel C: epoch {results[0]['epochs'][t]:g}, {len(results)} seeds")
    ax = add_axes(fig, "C")
    zero_line(ax)
    for x, cond in enumerate(favila.CONDITIONS):
        mean, sem = mean_sem([[res["pairmate"][cond][t]] for res in results])
        ax.errorbar(x, mean[0], yerr=sem[0], color=favila.COLORS[cond], marker="o", markersize=3,
                    elinewidth=0.5, capsize=1.2, capthick=0.5, linestyle="none")
    ax.set_xlim(-0.6, len(favila.CONDITIONS) - 0.4)
    ax.set_xticks(range(len(favila.CONDITIONS)))
    ax.set_xticklabels([favila.LABELS[c] for c in favila.CONDITIONS])
    style(ax, "Group", "Pairmate score", "TSP network")


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
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    style(ax, "Steps", "Similarity", title)
    if show_legend:
        legend(ax, [line("black", "A1 vs. A2"), line("tab:orange", "A1 vs. A3")], loc="center right")


def simulate_score(rho_B, phi_in, phi_out, gamma, n_steps=300_000, lr=0.02, seed=0):
    """Pairmate score (a2 - a3) / a1 of a two-layer linear network trained to convergence (rho_A = 0)."""
    X, Y, Z, _, _ = theory.build_associative_repulsion_data(0.0, rho_B, phi_in, phi_out, 2, 2, 25)
    model = theory.train_network(X, Y, gamma, n_steps, lr, seed=seed, progress=False)
    with torch.no_grad():
        H = model.hidden(Z)
        C = H @ H.T
    return ((C[0, 1] - C[0, 2]) / C[0, 0]).item()


def simulations() -> dict:
    """Simulation dots of G and H (cache keys "F|<network>|<gamma>" and "G|<gamma>|<rho_B>",
    from when these were panels F and G)."""
    cache = json.load(open(SIM_CACHE)) if os.path.exists(SIM_CACHE) else {}
    todo = {f"F|{net}|{g:g}": (1.0, *NETWORK_PHIS[net], g) for net in NETWORK_PHIS for g in G_GAMMAS_SIM}
    todo.update({f"G|{g:g}|{r:g}": (r, 0.5, 0.5, g if g > 0 else GAMMA_ZERO_SIM)
                 for g in H_GAMMAS for r in H_RHO_B_SIM})
    missing = {k: v for k, v in todo.items() if k not in cache}
    if missing:
        print(f"Simulating {len(missing)} networks for G and H ...")
        for key, args in missing.items():
            cache[key] = simulate_score(*args)
        os.makedirs(os.path.dirname(SIM_CACHE), exist_ok=True)
        json.dump(cache, open(SIM_CACHE, "w"), indent=1)
    return cache


def panel_g(fig, sims):
    gammas = np.geomspace(1e-4, 10.0, 400)
    ax = add_axes(fig, "G")
    zero_line(ax)
    ax.plot(gammas, theory.PS_PN(gammas), color=NETWORK_COLORS["Pred. network"], lw=LINEWIDTH)
    ax.plot(gammas, theory.PS_AE(gammas), color=NETWORK_COLORS["Autoencoder"], lw=LINEWIDTH)
    for net, color in NETWORK_COLORS.items():
        ax.scatter(G_GAMMAS_SIM, [sims[f"F|{net}|{g:g}"] for g in G_GAMMAS_SIM], color=color, s=9, zorder=3,
                   linewidths=0)
    ax.set_xscale("log")
    ax.set_xlim(gammas[0], gammas[-1])
    style(ax, r"Weight variance $\gamma$", "Pairmate score")
    legend(ax, [line(c, n) for n, c in NETWORK_COLORS.items()], loc="center left")


def panel_h(fig, sims):
    rho_B = np.linspace(0.0, 0.999, 300)
    ax = add_axes(fig, "H")
    zero_line(ax)
    for gamma in H_GAMMAS:
        y = (theory.score_autoencoder(0.0, rho_B) if gamma == 0
             else theory.pairmate_score(0.0, rho_B, 0.5, 0.5, gamma))
        ax.plot(rho_B, y, color=GAMMA_COLORS[gamma], lw=LINEWIDTH)
        ax.scatter(H_RHO_B_SIM, [sims[f"G|{gamma:g}|{r:g}"] for r in H_RHO_B_SIM], color=GAMMA_COLORS[gamma],
                   s=9, zorder=3, linewidths=0)
    ax.set_xlim(-0.03, 1.03)
    style(ax, r"B correlation $\rho_B$", "Pairmate score", "Autoencoder")
    labels = {0.0: r"$\gamma\to0$", 0.01: r"$\gamma$ = 0.01", 0.1: r"$\gamma$ = 0.1"}
    legend(ax, [line(GAMMA_COLORS[g], labels[g]) for g in H_GAMMAS], loc="lower left")


def panel_i(fig):
    """Autoencoder pairmate score over training; one panel per setup, each with its own y-scale."""
    for c, (key, setup, title, color) in enumerate(I_PANELS):
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
        ax.set_xticklabels(["1", "100"])
        style(ax, "Steps", "Pairmate score" if c == 0 else None, title)


# ====================================================================== #

def main():
    setup_style()
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text (selectable, searchable) rather than outlines
    fig = plt.figure(figsize=(FIG_MM[0] * MM, FIG_MM[1] * MM))
    panel_c(fig)
    panel_trajectory(fig, "E", "Pred. network", 1.0, 0.0, show_legend=True)
    panel_trajectory(fig, "F", "Autoencoder", 0.5, 0.5, show_legend=False)
    sims = simulations()
    panel_g(fig, sims)
    panel_h(fig, sims)
    panel_i(fig)
    for tag, (x, y) in TAGS.items():
        fig.text(x / FIG_MM[0], 1 - y / FIG_MM[1], tag, fontsize=FONTSIZE + 2, fontweight="bold",
                 va="bottom", ha="left")
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    text = buf.getvalue()
    end = text.rindex("</svg>")
    text = text[:end] + "".join(nested_svg(key) for key in IMAGES) + text[end:]
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    with open(f"{OUT_STEM}.svg", "w") as f:
        f.write(text)
    print(f"Saved {OUT_STEM}.svg")
    for ext, args in [("pdf", ()), ("png", ("--export-dpi=600", "--export-background=white"))]:
        if export_with_inkscape(f"{OUT_STEM}.svg", f"{OUT_STEM}.{ext}", *args):
            print(f"Saved {OUT_STEM}.{ext}")
        else:
            print(f"Inkscape export to {ext} failed; open {OUT_STEM}.svg instead")


if __name__ == "__main__":
    main()

"""plot_chanales2017.py — summary figure for Chanales et al. (2017) simulations.

One figure per sweep directory (one experiment × model), mean ± SEM across
seeds, in one row and in the style of ``plot_favila2016.py``:

a. route autoencoding loss
b. route discrimination: proportion of routes whose read-out is closer
   (Euclidean) to the route itself than to its pairmate (the other route of
   the same group)
c. representational similarity (Pearson correlation of the hidden layer)
   between pairmate routes and between non-pairmate routes, in the MSP
   (dashed) and TSP (dotted) network

Usage
-----
    python scripts/plot_chanales2017.py data/chanales2017/complete_model_linear
    python scripts/plot_chanales2017.py data/chanales2017/* --readout both

``--readout`` selects which output is evaluated for the loss and accuracy
panels: ``combined`` (sum of streams), ``individual`` (each stream alone), or
``both``.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    PANEL_MM_DEFAULT,
    add_output_args,
    check_output_args,
    data_out_stem,
    find_runs,
    label_axis,
    legend_handles,
    load_config,
    load_outputs,
    load_rows,
    load_steps,
    phase_step_edges,
    plot_conditions,
    readout_setup,
    reference_line,
    resolve_out_paths,
    save_figure,
    setup_style,
    style_axis,
    top_legend,
)

# Colors from cosyne_figure.ipynb (overlapping / non-overlapping routes).
C17_COLORS = {'pairmate': '#22678e', 'nonpairmate': '#de6139'}
C17_LABELS = {'pairmate': 'Pairmates', 'nonpairmate': 'Non-pairmates'}
SIM_CONDITIONS = ["pairmate", "nonpairmate"]
# Loss and accuracy have a single condition, drawn in black.
ALL = "all"
BLACK = {ALL: "black"}
PANEL_MM = PANEL_MM_DEFAULT


# ── metrics ──────────────────────────────────────────────────────────────────

def pair_masks(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(n_rows, n_rows) masks of pairmate and non-pairmate row pairs (upper triangle)."""
    pair = np.array([f"{r['subgroup']}/{r['group_id']}" for r in rows])
    same = pair[:, None] == pair[None, :]
    upper = np.triu(np.ones_like(same), k=1)
    return same & upper, ~same & upper


def similarity(reps: np.ndarray, rows: list[dict]) -> dict:
    """Mean pairmate and non-pairmate correlation at each checkpoint."""
    pm, npm = pair_masks(rows)
    corr = np.stack([np.corrcoef(r) for r in reps])              # (T, N, N)
    return {"pairmate": np.nanmean(corr[:, pm], 1), "nonpairmate": np.nanmean(corr[:, npm], 1)}


def pairmate_accuracy(route_out: dict, readout: str, rows: list[dict]) -> np.ndarray:
    """Proportion of rows whose read-out is closer to their own route than to the pairmate's."""
    target = route_out["target"]
    pred = route_out[readout]                                    # (T, N, D)
    pm, _ = pair_masks(rows)
    pm = pm | pm.T
    d_own = np.linalg.norm(pred - target[None], axis=-1)         # (T, N)
    d = np.linalg.norm(pred[:, :, None] - target[None, None], axis=-1)  # (T, N, N)
    d_mate = np.where(pm[None], d, np.inf).min(-1)
    return (d_own < d_mate).mean(1)


def compute_run(run_dir: str, readouts: list[str], layer: str) -> dict:
    cfg = load_config(run_dir)
    n_nets = len(cfg["model"]["networks"])
    res = {"cfg": cfg}

    steps = load_steps(run_dir, "representations")
    for probe in ("output", "loss_prediction"):
        if not np.array_equal(load_steps(run_dir, probe), steps):
            raise ValueError(f"{probe} was evaluated at different steps than representations.")
    phases, step_edges = phase_step_edges(run_dir, cfg)
    epoch_edges = np.cumsum([0] + [p.epochs for p in phases]).astype(float)
    res["epochs"] = np.interp(steps, step_edges, epoch_edges)
    res["n_epochs"] = int(epoch_edges[-1])

    rows = load_rows(run_dir, "output")
    route = load_outputs(run_dir, "output", "route", readouts)
    for r in readouts:
        res[("loss", r)] = {ALL: np.load(os.path.join(run_dir, f"loss_prediction_{r}.npy")).sum(-1)}
        res[("acc", r)] = {ALL: pairmate_accuracy(route, r, rows)}

    rows_rep = load_rows(run_dir, "representations")
    for ni in range(n_nets):
        reps = np.load(os.path.join(run_dir, f"representations_net{ni}_{layer}.npy"))
        res[("sim", ni)] = similarity(reps, rows_rep)
    return res


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(sweep_dir: str, readout: str, layer: str, out_paths: list[str]) -> None:
    runs_dirs = find_runs(sweep_dir, "representations_steps.npy")
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    readouts, styles, net_labels = readout_setup(n_nets, readout)
    runs = [compute_run(d, readouts, layer) for d in runs_dirs]
    x = runs[0]["epochs"]
    n_epochs = runs[0]["n_epochs"]
    xticks = list(range(0, n_epochs + 1, max(1, n_epochs // 4)))

    setup_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(PANEL_MM[0] * 3 / 25.4, PANEL_MM[1] / 25.4), dpi=300,
    )
    loss_ax, acc_ax, sim_ax = axes

    for r in readouts:
        plot_conditions(loss_ax, x, runs, ("loss", r), [ALL], BLACK, linestyle=styles[r])
        plot_conditions(acc_ax, x, runs, ("acc", r), [ALL], BLACK, linestyle=styles[r])
    loss_ax.set_ylim(bottom=0)
    acc_ax.set_ylim(-0.03, 1.03)
    acc_ax.set_yticks([0, 0.5, 1])
    reference_line(acc_ax, 0.5)

    # one line style per stream, matching the per-stream readouts
    for ni in range(n_nets):
        plot_conditions(sim_ax, x, runs, ("sim", ni), SIM_CONDITIONS, C17_COLORS,
                        linestyle=styles[f"net{ni}"])
    sim_ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)

    for ax, tag, title, ylabel in ((loss_ax, "a", None, "Loss"),
                                   (acc_ax, "b", "vs. pairmate route", "Accuracy"),
                                   (sim_ax, "c", None, "Similarity")):
        style_axis(ax, xticks, xlabel="Epoch")
        label_axis(ax, ylabel, title, tag)

    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.92))
    top_legend(fig, legend_handles(SIM_CONDITIONS, C17_COLORS, C17_LABELS, readouts, styles,
                                   net_labels, always_show_streams=True), y=0.9)
    save_figure(fig, out_paths)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/chanales2017/<model>) or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for the loss and accuracy panels.")
    parser.add_argument("--layer", default="hidden_0", help="Layer for the similarity panel.")
    add_output_args(parser)
    args = parser.parse_args()
    check_output_args(parser, args)
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.layer,
                    resolve_out_paths(args.out, data_out_stem(d, args.readout), args.ext))


if __name__ == "__main__":
    main()

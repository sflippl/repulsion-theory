"""plot_favila2016.py — summary figure for Favila et al. (2016) simulations.

One figure per sweep directory (one model × one training objective), mean ±
SEM across seeds, in the style of the paper figures:

a. scene-face autoencoding loss                       (Diff., Same)
b. strict face-recall accuracy vs. the pairmate face (Diff.) and vs.
   non-pairmate faces (Diff., Same)
c. soft face recall, same comparisons
d. pairmate score in the MSP (net1, dashed) and TSP (net2, dash-dot) network
   hidden layer                                       (No, Diff., Same)

Face recall presents only the scene and reads out the face slot.  The
competitor is the pairmate's face, or the nearest face belonging to another
pair.  Strict accuracy: the output is closer (Euclidean) to the scene's own
face than to the competitor.  Soft recall: 1 - d_own / (d_own + d_comp),
which is 1 for a perfect recall, 0.5 when equidistant, and > 0.5 exactly
when the strict test is passed.  The pairmate score is the mean
correlation between pairmates minus the mean correlation between
non-pairmates within a condition.

The x-axis is in epochs of the respective phase; the scene-only phase is
lightly shaded.

Usage
-----
    python scripts/plot_favila2016.py data/favila2016/summed/complete_model_linear
    python scripts/plot_favila2016.py data/favila2016/*/* --readout both
    python scripts/plot_favila2016.py data/favila2016/independent/complete_model_relu \\
        --readout individual --out figures/favila2016/relu_independent.pdf

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
    default_out_path,
    find_runs,
    label_axis,
    legend_handles,
    load_config,
    load_outputs,
    load_rows,
    load_steps,
    loss_by_condition,
    condition_masks as _condition_masks,
    pairmate_score as _pairmate_score,
    phase_step_edges,
    plot_conditions,
    readout_setup,
    reference_line,
    save_figure,
    setup_style,
    share_ylim,
    style_axis,
    top_legend,
)

FAVILA_SG_COLORS = {
    'no face':        'grey',
    'different face': 'blue',
    'same face':      'red',
}
FAVILA_LABELS = {
    'no face':        'No',
    'different face': 'Diff.',
    'same face':      'Same',
}
CONDITIONS = ["no_face", "different_face", "same_face"]
COLORS = {c: FAVILA_SG_COLORS[c.replace("_", " ")] for c in CONDITIONS}
LABELS = {c: FAVILA_LABELS[c.replace("_", " ")] for c in CONDITIONS}
PANEL_MM = PANEL_MM_DEFAULT


def steps_to_epochs(run_dir: str, cfg: dict, steps: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Map global steps to cumulative epochs; also return phase boundaries in epochs."""
    phases, step_edges = phase_step_edges(run_dir, cfg)
    epoch_edges = list(np.cumsum([0] + [p.epochs for p in phases]).astype(float))
    return np.interp(steps, step_edges, epoch_edges), epoch_edges


# ── metrics ──────────────────────────────────────────────────────────────────

def condition_masks(rows: list[dict]) -> dict[str, np.ndarray]:
    return _condition_masks(rows, CONDITIONS)


def association_recall(face_out: dict, readout: str, rows: list[dict]) -> dict:
    """Face recall vs. the pairmate's face ("pm") and vs. non-pairmate faces ("npm").

    "acc_*": strict test, own face closer than the competitor(s).
    "soft_*": 1 - d_own / (d_own + d_comp), with d_comp the distance to the
    pairmate's face or to the nearest non-pairmate face (0.5 = equidistant).
    """
    target = face_out["target"]
    faces, face_idx = np.unique(np.round(target, 8), axis=0, return_inverse=True)
    face_idx = face_idx.ravel()
    pair = [(r["subgroup"], r["group_id"]) for r in rows]
    pair_faces = {p: {face_idx[i] for i, q in enumerate(pair) if q == p} for p in set(pair)}
    pred = face_out[readout]                                     # (T, N, D)
    dist = np.linalg.norm(pred[:, :, None, :] - faces[None, None], axis=-1)  # (T, N, F)
    n = len(rows)
    own = dist[:, np.arange(n), face_idx]                        # (T, N)

    d_pairmate = np.full(own.shape, np.nan)
    d_nonpair = np.empty(own.shape)
    for i, p in enumerate(pair):
        mates = [j for j, q in enumerate(pair) if q == p and j != i]
        mate_faces = {face_idx[j] for j in mates} - {face_idx[i]}
        if mate_faces:
            d_pairmate[:, i] = dist[:, i, sorted(mate_faces)].min(-1)
        others = sorted(set(range(len(faces))) - pair_faces[p])
        d_nonpair[:, i] = dist[:, i, others].min(-1)

    masks = condition_masks(rows)
    out = {}
    for name, d_comp in (("pm", d_pairmate), ("npm", d_nonpair)):
        valid = {c: m for c, m in masks.items() if not np.isnan(d_comp[:, m]).any()}
        out[f"acc_{name}"] = {c: (own[:, m] < d_comp[:, m]).mean(1) for c, m in valid.items()}
        soft = 1 - own / (own + d_comp)
        out[f"soft_{name}"] = {c: soft[:, m].mean(1) for c, m in valid.items()}
    return out


def pairmate_score(reps: np.ndarray, rows: list[dict]) -> dict:
    """Mean pairmate correlation minus mean non-pairmate correlation, per condition."""
    return _pairmate_score(reps, rows, CONDITIONS)


def compute_run(run_dir: str, readouts: list[str], layer: str) -> dict:
    cfg = load_config(run_dir)
    n_nets = len(cfg["model"]["networks"])
    res = {"cfg": cfg, "n_nets": n_nets}

    steps = load_steps(run_dir, "representations")
    res["epochs"], res["phase_edges"] = steps_to_epochs(run_dir, cfg, steps)
    for probe in ("output_autoencoding", "output_face_prediction"):
        if not np.array_equal(load_steps(run_dir, probe), steps):
            raise ValueError(f"{probe} was evaluated at different steps than representations.")

    rows_sf = load_rows(run_dir, "output_autoencoding")
    rows_fp = load_rows(run_dir, "output_face_prediction")
    sf = [load_outputs(run_dir, "output_autoencoding", s, readouts) for s in ("face", "scene")]
    fp = load_outputs(run_dir, "output_face_prediction", "face", readouts)

    for r in readouts:
        res[("loss_sf", r)] = loss_by_condition(sf, r, rows_sf, CONDITIONS)
        for metric, values in association_recall(fp, r, rows_fp).items():
            res[(metric, r)] = values

    rows_rep = load_rows(run_dir, "representations")
    for ni in range(n_nets):
        reps = np.load(os.path.join(run_dir, f"representations_net{ni}_{layer}.npy"))
        res[("pairmate", ni)] = pairmate_score(reps, rows_rep)
    return res


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(sweep_dir: str, readout: str, layer: str, out_path: str) -> None:
    runs_dirs = find_runs(sweep_dir, "representations_steps.npy")
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    readouts, styles, net_labels = readout_setup(n_nets, readout)
    runs = [compute_run(d, readouts, layer) for d in runs_dirs]
    x = runs[0]["epochs"]
    edges = runs[0]["phase_edges"]

    # (row, col, tag, metric, title, y label)
    panels = [
        (0, 0, "a", "loss_sf", None, "Loss"),
        (0, 1, "b", "acc_pm", "vs. pairmate face", "Accuracy"),
        (0, 2, None, "acc_npm", "vs. non-pairmate face", None),
        (1, 0, "c", "soft_pm", "vs. pairmate face", "Soft recall"),
        (1, 1, None, "soft_npm", "vs. non-pairmate face", None),
        (1, 2, "d", "pairmate", None, "Pairmate score"),
    ]

    setup_style()
    fig, grid = plt.subplots(
        2, 3, figsize=(PANEL_MM[0] * 3 / 25.4, PANEL_MM[1] * 2 / 25.4), dpi=300,
    )
    axes = {}
    for row, col, tag, metric, title, ylabel in panels:
        ax = axes[metric] = grid[row, col]
        if metric == "pairmate":
            # one line style per stream, matching the per-stream readouts
            for ni in range(n_nets):
                plot_conditions(ax, x, runs, (metric, ni), CONDITIONS, COLORS,
                                linestyle=styles[f"net{ni}"])
            ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
        else:
            for r in readouts:
                plot_conditions(ax, x, runs, (metric, r), CONDITIONS, COLORS,
                                linestyle=styles[r])
        if metric.startswith("acc"):
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
        if metric in ("acc_pm", "soft_pm", "soft_npm"):
            reference_line(ax, 0.5)
        if metric == "loss_sf":
            ax.set_ylim(bottom=0)
        style_axis(ax, edges, shade_edges=edges, xlabel="Epoch" if row == 1 else None)
        label_axis(ax, ylabel, title, tag)

    # Soft recall: adaptive, but shared between the two panels.
    share_ylim([axes["soft_pm"], axes["soft_npm"]])

    fig.tight_layout(w_pad=0.8, h_pad=0.4, rect=(0, 0, 1, 0.93))
    top_legend(fig, legend_handles(CONDITIONS, COLORS, LABELS, readouts, styles, net_labels,
                                   always_show_streams=True))
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/favila2016/<objective>/<model>) or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for loss and accuracy panels.")
    parser.add_argument("--layer", default="hidden_0", help="Layer for the pairmate score.")
    parser.add_argument("--out", default=None, help="Output path (only with a single sweep dir).")
    parser.add_argument("--ext", default="pdf", help="File extension for default output paths.")
    args = parser.parse_args()
    if args.out and len(args.sweep_dirs) > 1:
        parser.error("--out can only be used with a single sweep directory.")
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.layer,
                    args.out or default_out_path(d, args.readout, args.ext))


if __name__ == "__main__":
    main()

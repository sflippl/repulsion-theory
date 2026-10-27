"""plot_wanjia2021.py — summary figure for Wanjia et al. (2021) simulations.

One figure per sweep directory (one experiment × model), mean ± SEM across
seeds, in the style of ``plot_favila2016.py``.  The four conditions cross
scene similarity (S) with object similarity (O) between pairmates; hue codes
scene similarity (red high, blue low) and lightness object similarity (dark
high, light low).

a. scene-object autoencoding loss
b. object recall accuracy: the object read out from the scene alone is
   closer (Euclidean) to the scene's own object than to any other object
c. pairmate score of the scenes (scene presented alone)
d. pairmate score of the objects (object presented alone)
e. lagged scene pairmate score, aligned to learning: for each pair, epoch 0
   is the first epoch at which the recall accuracy of the pair reaches the
   criterion (default: both objects recalled).  Similarities are computed
   between a scene at epoch t and the other scenes at epoch t - 1 (averaged
   over both directions); the score is the lagged pairmate similarity minus
   the mean lagged similarity to the non-pairmates of the same condition.

Pairmate scores are the mean pairmate correlation minus the mean
non-pairmate correlation within a condition, in the MSP (dashed) and TSP
(dotted) network hidden layer.

Usage
-----
    python scripts/plot_wanjia2021.py data/wanjia2021/complete_model_linear
    python scripts/plot_wanjia2021.py data/wanjia2021*/* --readout both --criterion 1.0

``--readout`` selects which output is evaluated for the loss and accuracy
panels: ``combined`` (sum of streams), ``individual`` (each stream alone), or
``both``.  The learning criterion of panel e always uses
``--criterion-readout`` (default: the summed output).
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    LINEWIDTH,
    PANEL_MM_DEFAULT,
    by_condition,
    find_runs,
    label_axis,
    legend_handles,
    load_config,
    load_outputs,
    load_rows,
    load_steps,
    pairmate_score,
    phase_step_edges,
    plot_conditions,
    readout_setup,
    row_loss_by_condition,
    save_figure,
    setup_style,
    share_ylim,
    style_axis,
    top_legend,
)

CONDITIONS = ["scene_high_obj_high", "scene_high_obj_low", "scene_low_obj_high", "scene_low_obj_low"]
WANJIA_COLORS = {
    "scene_high_obj_high": "#b2182b",
    "scene_high_obj_low":  "#ef8a62",
    "scene_low_obj_high":  "#2166ac",
    "scene_low_obj_low":   "#67a9cf",
}
WANJIA_LABELS = {
    "scene_high_obj_high": "S+ O+",
    "scene_high_obj_low":  "S+ O−",
    "scene_low_obj_high":  "S− O+",
    "scene_low_obj_low":   "S− O−",
}
PANEL_MM = PANEL_MM_DEFAULT
WIDTH_MM = 3 * 45


# ── metrics ──────────────────────────────────────────────────────────────────

def recall_correct(object_out: dict, readout: str) -> np.ndarray:
    """(n_checkpoints, n_rows): read-out closer to the own object than to any other object."""
    target = object_out["target"]
    objects, obj_idx = np.unique(np.round(target, 8), axis=0, return_inverse=True)
    obj_idx = obj_idx.ravel()
    pred = object_out[readout]                                   # (T, N, D)
    dist = np.linalg.norm(pred[:, :, None, :] - objects[None, None], axis=-1)  # (T, N, O)
    own = dist[:, np.arange(len(obj_idx)), obj_idx]
    dist[:, np.arange(len(obj_idx)), obj_idx] = np.inf
    return own < dist.min(-1)


def lagged_pairmate(reps: np.ndarray, rows: list[dict], correct: np.ndarray,
                    criterion: float, window: int) -> dict:
    """Lagged pairmate score per condition, aligned to each pair's criterion epoch.

    *reps* (n_epochs + 1, N, D) and *correct* (n_epochs + 1, N) are sampled at
    the epoch boundaries.  Returns {condition: (2 * window + 1,)} averaged over
    the pairs that reach the criterion (NaN where no pair contributes).
    """
    z = reps - reps.mean(-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    lag = np.full((len(reps), len(rows), len(rows)), np.nan)
    lag[1:] = np.einsum("tid,tjd->tij", z[1:], z[:-1])          # scene i at t, scene j at t - 1
    lag = (lag + lag.transpose(0, 2, 1)) / 2

    sgs = np.array([r["subgroup"] for r in rows])
    pair = np.array([f"{r['subgroup']}/{r['group_id']}" for r in rows])
    out = {}
    for c in CONDITIONS:
        aligned = []
        for p in np.unique(pair[sgs == c]):
            i, j = np.flatnonzero(pair == p)
            others = np.flatnonzero((sgs == c) & (pair != p))
            score = lag[:, i, j] - lag[:, [i, j]][:, :, others].mean((1, 2))
            reached = np.flatnonzero(correct[:, [i, j]].mean(1) >= criterion)
            if not len(reached):
                continue
            epochs = reached[0] + np.arange(-window, window + 1)
            valid = (epochs >= 0) & (epochs < len(reps))
            aligned.append(np.where(valid, score[np.clip(epochs, 0, len(reps) - 1)], np.nan))
        if aligned:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # offsets without any pair
                out[c] = np.nanmean(np.stack(aligned), 0)
    return out


def compute_run(run_dir: str, readouts: list[str], layer: str, criterion: float,
                criterion_readout: str, window: int) -> dict:
    cfg = load_config(run_dir)
    n_nets = len(cfg["model"]["networks"])
    res = {"cfg": cfg}

    steps = load_steps(run_dir, "representations")
    for probe in ("representations_object", "row_loss_autoencoding", "output"):
        if not np.array_equal(load_steps(run_dir, probe), steps):
            raise ValueError(f"{probe} was evaluated at different steps than representations.")
    phases, step_edges = phase_step_edges(run_dir, cfg)
    epoch_edges = np.cumsum([0] + [p.epochs for p in phases]).astype(float)
    res["epochs"] = np.interp(steps, step_edges, epoch_edges)
    res["n_epochs"] = int(epoch_edges[-1])
    # checkpoint index of each epoch boundary
    boundary = [np.flatnonzero(np.isclose(res["epochs"], e)) for e in range(res["n_epochs"] + 1)]
    if any(not len(b) for b in boundary):
        raise ValueError(f"{run_dir}: not every epoch boundary is a checkpoint.")
    boundary = np.array([b[0] for b in boundary])

    rows = load_rows(run_dir, "output")
    obj = load_outputs(run_dir, "output", "object", sorted(set(readouts) | {criterion_readout}))
    for r in readouts:
        res[("loss", r)] = row_loss_by_condition(run_dir, "row_loss_autoencoding", r, CONDITIONS)
        res[("acc", r)] = by_condition(recall_correct(obj, r).astype(float), rows, CONDITIONS)
    correct = recall_correct(obj, criterion_readout)[boundary]

    rows_scene = load_rows(run_dir, "representations")
    rows_obj = load_rows(run_dir, "representations_object")
    if rows_scene != rows:
        raise ValueError("Scene representations and object read-out rows differ.")
    for ni in range(n_nets):
        scenes = np.load(os.path.join(run_dir, f"representations_net{ni}_{layer}.npy"))
        objects = np.load(os.path.join(run_dir, f"representations_object_net{ni}_{layer}.npy"))
        res[("ps_scene", ni)] = pairmate_score(scenes, rows_scene, CONDITIONS)
        res[("ps_object", ni)] = pairmate_score(objects, rows_obj, CONDITIONS)
        res[("lagged", ni)] = lagged_pairmate(scenes[boundary], rows_scene, correct,
                                              criterion, window)
    return res


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_aligned(ax, x: np.ndarray, runs: list[dict], key, linestyle) -> None:
    """Like ``plot_conditions``, but skipping runs without data at an offset."""
    for c in CONDITIONS:
        vals = np.stack([r[key].get(c, np.full(len(x), np.nan)) for r in runs])
        n = (~np.isnan(vals)).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # offsets without any data
            mean = np.nanmean(vals, 0)
            sem = np.nanstd(vals, 0, ddof=1) / np.sqrt(n)
        ax.plot(x, mean, color=WANJIA_COLORS[c], linestyle=linestyle, lw=LINEWIDTH)
        ok = n > 1
        ax.fill_between(x[ok], (mean - sem)[ok], (mean + sem)[ok], color=WANJIA_COLORS[c],
                        alpha=0.2, linewidth=0)


def make_figure(sweep_dir: str, readout: str, layer: str, criterion: float,
                criterion_readout: str, window: int, out_path: str) -> None:
    runs_dirs = find_runs(sweep_dir, "representations_steps.npy")
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    readouts, styles, net_labels = readout_setup(n_nets, readout)
    runs = [compute_run(d, readouts, layer, criterion, criterion_readout, window)
            for d in runs_dirs]
    x = runs[0]["epochs"]
    n_epochs = runs[0]["n_epochs"]
    xticks = list(range(0, n_epochs + 1, max(1, n_epochs // 4)))

    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, PANEL_MM[1] * 2 / 25.4), dpi=300)
    # Top row: behaviour (two wide panels); bottom row: representations.
    gs = fig.add_gridspec(2, 6)
    loss_ax, acc_ax = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    scene_ax, obj_ax, lag_ax = (fig.add_subplot(gs[1, 2 * i: 2 * i + 2]) for i in range(3))

    for r in readouts:
        plot_conditions(loss_ax, x, runs, ("loss", r), CONDITIONS, WANJIA_COLORS,
                        linestyle=styles[r])
        plot_conditions(acc_ax, x, runs, ("acc", r), CONDITIONS, WANJIA_COLORS,
                        linestyle=styles[r])
    loss_ax.set_ylim(bottom=0)
    acc_ax.set_ylim(-0.03, 1.03)
    acc_ax.set_yticks([0, 0.5, 1])

    # one line style per stream, matching the per-stream readouts
    lag_x = np.arange(-window, window + 1)
    for ni in range(n_nets):
        style = styles[f"net{ni}"]
        plot_conditions(scene_ax, x, runs, ("ps_scene", ni), CONDITIONS, WANJIA_COLORS,
                        linestyle=style)
        plot_conditions(obj_ax, x, runs, ("ps_object", ni), CONDITIONS, WANJIA_COLORS,
                        linestyle=style)
        plot_aligned(lag_ax, lag_x, runs, ("lagged", ni), style)
    for ax in (scene_ax, obj_ax, lag_ax):
        ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    lag_ax.axvline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    share_ylim([scene_ax, obj_ax])

    # constant tag offset in mm, matching the Favila figure (-0.42 of a 40 mm panel)
    tag_x_wide = -0.42 * PANEL_MM[0] / (WIDTH_MM / 2)
    tag_x = -0.42 * PANEL_MM[0] / (WIDTH_MM / 3)
    for ax, tag, title, ylabel, tx in (
        (loss_ax, "a", None, "Loss", tag_x_wide),
        (acc_ax, "b", None, "Object recall", tag_x_wide),
        (scene_ax, "c", "Scenes", "Pairmate score", tag_x),
        (obj_ax, "d", "Objects", None, tag_x),
        (lag_ax, "e", "Scenes, 1-epoch lag", None, tag_x),
    ):
        style_axis(ax, xticks, xlabel="Epoch" if ax in (scene_ax, obj_ax) else None)
        label_axis(ax, ylabel, title, tag, tag_x=tx)
    lag_ax.set_xlim(-window, window)
    lag_ax.set_xticks(np.arange(-window, window + 1, max(1, window // 2)))
    lag_ax.set_xlabel("Epochs from criterion", labelpad=0)

    fig.tight_layout(w_pad=0.8, h_pad=0.6, rect=(0, 0, 1, 0.95))
    top_legend(fig, legend_handles(CONDITIONS, WANJIA_COLORS, WANJIA_LABELS, readouts, styles,
                                   net_labels, always_show_streams=True), y=0.95)
    save_figure(fig, out_path)


def default_out_path(sweep_dir: str, readout: str, ext: str) -> str:
    """figures/<experiment>/<rest of the sweep path>_<readout>.<ext>, relative to data/."""
    parts = os.path.normpath(sweep_dir).split(os.sep)
    if parts[-1].isdigit():
        parts = parts[:-1]
    rel = parts[parts.index("data") + 1:] if "data" in parts else parts[-2:]
    return os.path.join("figures", rel[0], "_".join(rel[1:] + [readout]) + f".{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/wanjia2021/<model>) or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for the loss and accuracy panels.")
    parser.add_argument("--layer", default="hidden_0", help="Layer for the pairmate scores.")
    parser.add_argument("--criterion", type=float, default=1.0,
                        help="Recall accuracy of a pair (fraction of its two objects) that "
                             "defines the learning epoch in panel e.")
    parser.add_argument("--criterion-readout", default="combined",
                        help="Output used for the learning criterion (combined, net0, ...).")
    parser.add_argument("--window", type=int, default=4,
                        help="Epochs shown before/after the criterion epoch in panel e.")
    parser.add_argument("--out", default=None, help="Output path (only with a single sweep dir).")
    parser.add_argument("--ext", default="pdf", help="File extension for default output paths.")
    args = parser.parse_args()
    if args.out and len(args.sweep_dirs) > 1:
        parser.error("--out can only be used with a single sweep directory.")
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.layer, args.criterion, args.criterion_readout,
                    args.window, args.out or default_out_path(d, args.readout, args.ext))


if __name__ == "__main__":
    main()

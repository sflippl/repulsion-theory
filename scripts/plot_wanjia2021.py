"""plot_wanjia2021.py — summary figure for Wanjia et al. (2021) simulations.

One figure per sweep directory (one experiment × model), mean ± SEM across
seeds, in the style of ``plot_favila2016.py``.  The four conditions cross
scene similarity (S) with object similarity (O) between pairmates; hue codes
scene similarity (red high, blue low) and lightness object similarity (dark
high, light low).

a. scene-object autoencoding loss
b. object recall accuracy: the object read out from the scene alone is
   closer (Euclidean) to the scene's own object than to any other object
c. soft object recall: 1 - d_own / (d_own + d_comp), with d_comp the
   distance to the nearest other object (0.5 = equidistant, > 0.5 exactly
   when the strict test of b is passed)
d. pairmate score of the scenes (scene presented alone)
e. pairmate score of the objects (object presented alone)
f. scene pairmate score relative to the insight moment: for each pair, the
   insight epoch is the first epoch boundary at which the pair's soft recall
   (mean of its two scenes) reaches ``--insight-threshold`` (default 0.55;
   ``--insight-mode absolute``) or has gained at least ``--insight-frac``
   (default 95%) of its maximal gain over epoch 0 (``--insight-mode
   relative``).  Pairs without an insight moment are left out.  At offset k, similarities are
   computed between the scenes at epoch insight + k and the scenes at the
   insight epoch (averaged over both directions), so k = 0 is the ordinary
   pairmate score at the insight moment; the score is the pairmate
   similarity minus the mean similarity to the non-pairmates of the same
   condition.

Pairmate scores are the mean pairmate correlation minus the mean
non-pairmate correlation within a condition, in the MSP (dashed) and TSP
(dotted) network hidden layer.

Usage
-----
    python scripts/plot_wanjia2021.py data/wanjia2021/complete_model_linear
    python scripts/plot_wanjia2021.py data/wanjia2021*/* --readout both --insight-threshold 0.6
    python scripts/plot_wanjia2021.py data/wanjia2021/* --insight-mode relative --insight-frac 0.9

``--readout`` selects which output is evaluated for the loss and accuracy
panels: ``combined`` (sum of streams), ``individual`` (each stream alone), or
``both``.  The insight moment of panel f always uses ``--insight-readout``
(default: the summed output).
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
    add_output_args,
    by_condition,
    check_output_args,
    data_out_stem,
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
    reference_line,
    resolve_out_paths,
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

def object_recall(object_out: dict, readout: str) -> tuple[np.ndarray, np.ndarray]:
    """Strict and soft recall of the scene's object, each (n_checkpoints, n_rows).

    The competitor is the nearest other object.  Strict: the read-out is
    closer to the own object.  Soft: 1 - d_own / (d_own + d_comp).
    """
    target = object_out["target"]
    objects, obj_idx = np.unique(np.round(target, 8), axis=0, return_inverse=True)
    obj_idx = obj_idx.ravel()
    pred = object_out[readout]                                   # (T, N, D)
    dist = np.linalg.norm(pred[:, :, None, :] - objects[None, None], axis=-1)  # (T, N, O)
    own = dist[:, np.arange(len(obj_idx)), obj_idx]
    dist[:, np.arange(len(obj_idx)), obj_idx] = np.inf
    comp = dist.min(-1)
    return own < comp, 1 - own / (own + comp)


def insight_epoch(soft: np.ndarray, mode: str, value: float) -> int | None:
    """First epoch at which *soft* (n_epochs + 1,) reaches the insight criterion.

    ``mode="absolute"``: soft recall >= *value*.  ``mode="relative"``: the
    gain over epoch 0 is at least *value* times the maximal gain.
    """
    if mode == "absolute":
        reached = np.flatnonzero(soft >= value)
    else:
        gain = soft - soft[0]
        if not gain.max() > 0:
            return None
        reached = np.flatnonzero(gain >= value * gain.max())
    return int(reached[0]) if len(reached) else None


def insight_pairmate(reps: np.ndarray, rows: list[dict], soft: np.ndarray,
                     insight: tuple[str, float], window: int) -> dict:
    """Pairmate score between each epoch and the pair's insight epoch, per condition.

    *reps* (n_epochs + 1, N, D) and *soft* (n_epochs + 1, N) are sampled at
    the epoch boundaries.  Returns {condition: (2 * window + 1,)} averaged over
    the pairs with an insight moment (NaN where no pair contributes).
    """
    z = reps - reps.mean(-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    n_ep = len(reps)
    sgs = np.array([r["subgroup"] for r in rows])
    pair = np.array([f"{r['subgroup']}/{r['group_id']}" for r in rows])
    out = {}
    for c in CONDITIONS:
        aligned = []
        for p in np.unique(pair[sgs == c]):
            i, j = np.flatnonzero(pair == p)
            others = np.flatnonzero((sgs == c) & (pair != p))
            t0 = insight_epoch(soft[:, [i, j]].mean(1), *insight)
            if t0 is None:
                continue
            # scene a at epoch t vs. scene b at the insight epoch, both directions
            cross = np.einsum("tid,jd->tij", z, z[t0])
            cross = (cross + cross.transpose(0, 2, 1)) / 2
            score = cross[:, i, j] - cross[:, [i, j]][:, :, others].mean((1, 2))
            epochs = t0 + np.arange(-window, window + 1)
            valid = (epochs >= 0) & (epochs < n_ep)
            aligned.append(np.where(valid, score[np.clip(epochs, 0, n_ep - 1)], np.nan))
        if aligned:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # offsets without any pair
                out[c] = np.nanmean(np.stack(aligned), 0)
    return out


def compute_run(run_dir: str, readouts: list[str], layer: str, insight: tuple[str, float],
                insight_readout: str, window: int) -> dict:
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
    obj = load_outputs(run_dir, "output", "object", sorted(set(readouts) | {insight_readout}))
    for r in readouts:
        res[("loss", r)] = row_loss_by_condition(run_dir, "row_loss_autoencoding", r, CONDITIONS)
        correct, soft = object_recall(obj, r)
        res[("acc", r)] = by_condition(correct.astype(float), rows, CONDITIONS)
        res[("soft", r)] = by_condition(soft, rows, CONDITIONS)
    soft_insight = object_recall(obj, insight_readout)[1][boundary]

    rows_scene = load_rows(run_dir, "representations")
    rows_obj = load_rows(run_dir, "representations_object")
    if rows_scene != rows:
        raise ValueError("Scene representations and object read-out rows differ.")
    for ni in range(n_nets):
        scenes = np.load(os.path.join(run_dir, f"representations_net{ni}_{layer}.npy"))
        objects = np.load(os.path.join(run_dir, f"representations_object_net{ni}_{layer}.npy"))
        res[("ps_scene", ni)] = pairmate_score(scenes, rows_scene, CONDITIONS)
        res[("ps_object", ni)] = pairmate_score(objects, rows_obj, CONDITIONS)
        res[("insight", ni)] = insight_pairmate(scenes[boundary], rows_scene, soft_insight,
                                                insight, window)
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


def make_figure(sweep_dir: str, readout: str, layer: str, insight: tuple[str, float],
                insight_readout: str, window: int, out_paths: list[str]) -> None:
    runs_dirs = find_runs(sweep_dir, "representations_steps.npy")
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    readouts, styles, net_labels = readout_setup(n_nets, readout)
    runs = [compute_run(d, readouts, layer, insight, insight_readout, window)
            for d in runs_dirs]
    x = runs[0]["epochs"]
    n_epochs = runs[0]["n_epochs"]
    xticks = list(range(0, n_epochs + 1, max(1, n_epochs // 4)))

    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, PANEL_MM[1] * 2 / 25.4), dpi=300)
    # Top row: behaviour; bottom row: representations.
    grid = fig.subplots(2, 3)
    (loss_ax, acc_ax, soft_ax), (scene_ax, obj_ax, lag_ax) = grid

    for r in readouts:
        for ax, metric in ((loss_ax, "loss"), (acc_ax, "acc"), (soft_ax, "soft")):
            plot_conditions(ax, x, runs, (metric, r), CONDITIONS, WANJIA_COLORS,
                            linestyle=styles[r])
    loss_ax.set_ylim(bottom=0)
    acc_ax.set_ylim(-0.03, 1.03)
    acc_ax.set_yticks([0, 0.5, 1])
    reference_line(soft_ax, 0.5)

    # one line style per stream, matching the per-stream readouts
    lag_x = np.arange(-window, window + 1)
    for ni in range(n_nets):
        style = styles[f"net{ni}"]
        plot_conditions(scene_ax, x, runs, ("ps_scene", ni), CONDITIONS, WANJIA_COLORS,
                        linestyle=style)
        plot_conditions(obj_ax, x, runs, ("ps_object", ni), CONDITIONS, WANJIA_COLORS,
                        linestyle=style)
        plot_aligned(lag_ax, lag_x, runs, ("insight", ni), style)
    for ax in (scene_ax, obj_ax, lag_ax):
        ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    lag_ax.axvline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    share_ylim([scene_ax, obj_ax])

    # constant tag offset in mm, matching the Favila figure (-0.42 of a 40 mm panel)
    tag_x = -0.42 * PANEL_MM[0] / (WIDTH_MM / 3)
    for ax, tag, title, ylabel in (
        (loss_ax, "a", None, "Loss"),
        (acc_ax, "b", None, "Object recall"),
        (soft_ax, "c", None, "Soft recall"),
        (scene_ax, "d", "Scenes", "Pairmate score"),
        (obj_ax, "e", "Objects", None),
        (lag_ax, "f", None, None),
    ):
        style_axis(ax, xticks, xlabel="Epoch" if ax in (scene_ax, obj_ax) else None)
        label_axis(ax, ylabel, title, tag, tag_x=tag_x)
    lag_ax.set_xlim(-window, window)
    lag_ax.set_xticks(np.arange(-window, window + 1, max(1, window // 2)))
    lag_ax.set_xlabel("Epochs from insight", labelpad=0)

    fig.tight_layout(w_pad=0.8, h_pad=0.6, rect=(0, 0, 1, 0.95))
    top_legend(fig, legend_handles(CONDITIONS, WANJIA_COLORS, WANJIA_LABELS, readouts, styles,
                                   net_labels, always_show_streams=True), y=0.95)
    save_figure(fig, out_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/wanjia2021/<model>) or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for the loss and accuracy panels.")
    parser.add_argument("--layer", default="hidden_0", help="Layer for the pairmate scores.")
    parser.add_argument("--insight-mode", choices=["absolute", "relative"], default="absolute",
                        help="Insight criterion of panel f: absolute soft-recall threshold or "
                             "fraction of the maximal gain.")
    parser.add_argument("--insight-threshold", type=float, default=0.55,
                        help="Soft recall of a pair that defines its insight epoch "
                             "(--insight-mode absolute).")
    parser.add_argument("--insight-frac", type=float, default=0.95,
                        help="Fraction of a pair's maximal soft-recall gain (over epoch 0) "
                             "that defines its insight epoch (--insight-mode relative).")
    parser.add_argument("--insight-readout", default="combined",
                        help="Output used for the insight moment (combined, net0, ...).")
    parser.add_argument("--window", type=int, default=4,
                        help="Epochs shown before/after the insight epoch in panel f.")
    add_output_args(parser)
    args = parser.parse_args()
    check_output_args(parser, args)
    insight = (args.insight_mode, args.insight_threshold if args.insight_mode == "absolute"
               else args.insight_frac)
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.layer, insight, args.insight_readout,
                    args.window, resolve_out_paths(args.out, data_out_stem(d, args.readout), args.ext))


if __name__ == "__main__":
    main()

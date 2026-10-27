"""plot_chanales2021.py — summary figure for Chanales et al. (2021) simulations.

One figure per sweep directory (one experiment × model × training objective),
mean ± SEM across seeds, in the style of ``plot_favila2016.py``:

a. training losses per condition: color-object-face autoencoding, object-face
   autoencoding (the ``prediction_test`` task) and, for experiments trained on
   ``association_group`` (chanales2021_exp4), the face-group association loss
b. color error: squared error of the color read out from object + face
c. response away from competitor: proportion of color read-outs that are
   rotated away from the pairmate's color (in the plane spanned by the two
   colors; see ``poster_figure.ipynb``)

The x-axis counts training cycles (one pass through each task in the schedule).
Phases training object-face autoencoding (``prediction_test``) are shaded grey.

Usage
-----
    python scripts/plot_chanales2021.py data/chanales2021/summed/complete_model_linear
    python scripts/plot_chanales2021.py data/chanales2021*/*/* --readout both

``--readout`` selects which output is evaluated: ``combined`` (sum of
streams), ``individual`` (each stream alone), or ``both``.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    PANEL_MM_DEFAULT,
    PHASE_SHADE,
    by_condition,
    default_out_path,
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
    row_loss_by_condition,
    save_figure,
    setup_style,
    style_axis,
    top_legend,
)

# Colors and labels from poster_figure.ipynb, ordered low → ultra similarity.
C21_COLORS = {
    'ultra_sim': 'tab:green',
    'high_sim': 'tab:red',
    'med_sim':  '#FFD700',
    'low_sim':  'tab:blue',
}
C21_LABELS = {
    'ultra_sim': 'ultra',
    'high_sim': 'high',
    'med_sim':  'mod.',
    'low_sim':  'low',
}
CONDITIONS = ["low_sim", "med_sim", "high_sim", "ultra_sim"]
# Training phases on this task are shaded in every panel.
SHADED_TASK = "prediction_test"
# Two bottom panels of 45 mm; with three losses the figure widens so that the
# top-row panels are only somewhat narrower than the bottom ones.
PANEL_MM = (45, PANEL_MM_DEFAULT[1])
WIDTH_MM = {2: 2 * PANEL_MM[0], 3: 2.4 * PANEL_MM[0]}

# (probe, title) of the training losses, in panel order
LOSS_PROBES = [
    ("row_loss_autoencoding", "Color-object-face\nautoencoding"),
    ("row_loss_prediction_test", "Object-face\nautoencoding"),
    ("row_loss_association", "Face-group\nassociation"),
]


# ── metrics ──────────────────────────────────────────────────────────────────

def away_from_competitor(color_out: dict, readout: str, rows: list[dict]) -> np.ndarray:
    """(n_checkpoints, n_rows): 1 if the read-out is rotated away from the pairmate's
    color, 0 if towards it, NaN if undefined.

    Angles are measured in the plane spanned by the own color (origin) and the
    pairmate's color (competitor), as in ``_signed_angle`` in poster_figure.ipynb.
    """
    target = color_out["target"]
    pred = color_out[readout]                                    # (T, N, D)
    pair = [(r["subgroup"], r["group_id"]) for r in rows]
    mate = np.array([next(j for j, q in enumerate(pair) if q == p and j != i)
                     for i, p in enumerate(pair)])
    origin, competitor = target, target[mate]
    e1 = origin / np.linalg.norm(origin, axis=-1, keepdims=True)
    v = competitor - (competitor * e1).sum(-1, keepdims=True) * e1
    e2 = v / np.linalg.norm(v, axis=-1, keepdims=True)

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    angle_comp = np.arctan2((competitor * e2).sum(-1), (competitor * e1).sum(-1))  # (N,)
    angle_pred = np.arctan2((pred * e2).sum(-1), (pred * e1).sum(-1))              # (T, N)
    signed = -np.sign(wrap(angle_comp))[None] * wrap(angle_pred)
    return np.where(np.isfinite(signed) & (signed != 0), signed > 0, np.nan)


def compute_run(run_dir: str, readouts: list[str]) -> dict:
    cfg = load_config(run_dir)
    res = {"cfg": cfg}

    steps = load_steps(run_dir, "prediction")
    phases, step_edges = phase_step_edges(run_dir, cfg)
    n_cycles = int(cfg["schedule"].get("epochs") or 1)
    scale = n_cycles / step_edges[-1]
    res["x"] = steps * scale
    res["n_cycles"] = n_cycles
    res["shaded"] = [(lo * scale, hi * scale)
                     for phase, lo, hi in zip(phases, step_edges[:-1], step_edges[1:])
                     if phase.tasks[0] == SHADED_TASK]

    res["loss_probes"] = [p for p, _ in LOSS_PROBES
                          if os.path.exists(os.path.join(run_dir, f"{p}_rows.json"))]
    for probe in res["loss_probes"]:
        if not np.array_equal(load_steps(run_dir, probe), steps):
            raise ValueError(f"{probe} was evaluated at different steps than prediction.")

    rows = load_rows(run_dir, "prediction")
    color = load_outputs(run_dir, "prediction", "color", readouts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # diverged runs / zero-length vectors
        for r in readouts:
            for probe in res["loss_probes"]:
                res[(probe, r)] = row_loss_by_condition(run_dir, probe, r, CONDITIONS)
            err = ((color[r] - color["target"][None]) ** 2).sum(-1)
            res[("color_err", r)] = by_condition(err, rows, CONDITIONS)
            away = away_from_competitor(color, r, rows)
            res[("away", r)] = {c: np.nanmean(away[:, m], axis=1) for c, m in
                                ((c, np.array([row["subgroup"] == c for row in rows]))
                                 for c in CONDITIONS) if m.any()}
    return res


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(sweep_dir: str, readout: str, out_path: str) -> None:
    runs_dirs = find_runs(sweep_dir, "prediction_steps.npy")
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    readouts, styles, net_labels = readout_setup(n_nets, readout)
    runs = [compute_run(d, readouts) for d in runs_dirs]
    x = runs[0]["x"]
    n_cycles = runs[0]["n_cycles"]
    xticks = list(range(0, n_cycles + 1, max(1, n_cycles // 4)))
    titles = dict(LOSS_PROBES)
    loss_probes = runs[0]["loss_probes"]
    n_loss = len(loss_probes)

    def shade(ax):
        for lo, hi in runs[0]["shaded"]:
            ax.axvspan(lo, hi, color=PHASE_SHADE, zorder=0, linewidth=0)

    setup_style()
    width_mm = WIDTH_MM.get(n_loss, 0.8 * n_loss * PANEL_MM[0])
    fig = plt.figure(figsize=(width_mm / 25.4, PANEL_MM[1] * 2 / 25.4), dpi=300)

    def tag_x(panel_mm: float) -> float:
        # constant offset in mm, matching the Favila figure (-0.42 of a 40 mm panel)
        return -0.42 * PANEL_MM_DEFAULT[0] / panel_mm

    # Top row: one panel per training loss, compressed to the width of the
    # bottom row's two panels.
    gs = fig.add_gridspec(2, 2 * n_loss)
    loss_axes = [fig.add_subplot(gs[0, 2 * i: 2 * i + 2]) for i in range(n_loss)]
    err_ax = fig.add_subplot(gs[1, :n_loss])
    away_ax = fig.add_subplot(gs[1, n_loss:])

    for i, (ax, probe) in enumerate(zip(loss_axes, loss_probes)):
        for r in readouts:
            plot_conditions(ax, x, runs, (probe, r), CONDITIONS, C21_COLORS, linestyle=styles[r])
        ax.set_ylim(bottom=0)
        shade(ax)
        style_axis(ax, xticks, xlabel=None)
        label_axis(ax, "Loss" if i == 0 else None, titles[probe], "a" if i == 0 else None,
                   tag_x=tag_x(width_mm / n_loss))

    for ax, metric, ylabel, tag in ((err_ax, "color_err", "Color error", "b"),
                                    (away_ax, "away", "Response away\nfrom competitor", "c")):
        for r in readouts:
            plot_conditions(ax, x, runs, (metric, r), CONDITIONS, C21_COLORS, linestyle=styles[r])
        if metric == "color_err":
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            reference_line(ax, 0.5)
        shade(ax)
        style_axis(ax, xticks, xlabel="Epoch")
        label_axis(ax, ylabel, None, tag, tag_x=tag_x(width_mm / 2))

    fig.tight_layout(w_pad=0.8, h_pad=0.6, rect=(0, 0, 1, 0.9))
    top_legend(fig, legend_handles(CONDITIONS, C21_COLORS, C21_LABELS, readouts, styles,
                                   net_labels), y=0.94)
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/<experiment>/<objective>/<model>) "
                             "or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for all panels.")
    parser.add_argument("--out", default=None, help="Output path (only with a single sweep dir).")
    parser.add_argument("--ext", default="pdf", help="File extension for default output paths.")
    args = parser.parse_args()
    if args.out and len(args.sweep_dirs) > 1:
        parser.error("--out can only be used with a single sweep directory.")
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.out or default_out_path(d, args.readout, args.ext))


if __name__ == "__main__":
    main()

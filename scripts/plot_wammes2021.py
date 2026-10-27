"""plot_wammes2021.py — summary figure for a Wammes sweep (mean ± SEM across seeds).

Reads every ``*/results.npz`` below a sweep directory written by
``scripts/tasks/wammes2021.py --multirun`` (see ``scripts/wammes2021.sh``; one
sweep per training objective and model) and produces one figure in the
style of ``plot_favila2016.py``:

a. loss on the training set
b. recall: proportion of items whose output is closer to the item itself
   than to its pairmate, averaged over all input similarities ρ
c. pairmate similarity (correlation of the hidden representations) over
   training in the MSP and the TSP network (one panel each), one color per
   ρ; the non-pairmate baseline is shown in grey
d. change in pairmate similarity relative to before training, as a function
   of ρ, in the MSP and TSP network, at the final epoch or at the epochs
   given by ``--epochs`` (shades of green, darker = later)
e. pairmate similarity itself, same layout as d

``--readout`` selects which output is evaluated in a and b: ``combined``
(sum of streams, solid), ``individual`` (each stream alone, dashed/dotted),
or ``both``.  Both rows span the same width.

Usage
-----
    python scripts/plot_wammes2021.py data/wammes2021/summed/complete_model_linear
    python scripts/plot_wammes2021.py data/wammes2021/*/* --readout both --ext pdf png
    python scripts/plot_wammes2021.py data/wammes2021/independent/complete_model_linear --epochs 1 5 10 --logx

Figures are written to ``figures/<experiment>/<model>_<objective>_<readout>.<ext>``.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from plot_common import (
    LINEWIDTH,
    NETWORK_LABELS,
    PANEL_MM_DEFAULT,
    STREAM_STYLES,
    add_output_args,
    check_output_args,
    label_axis,
    reference_line,
    resolve_out_paths,
    save_figure,
    setup_style,
    share_ylim,
    style_axis,
    sweep_parts,
    top_legend,
)

BASELINE = "#8a8a85"
# ordinal blue ramp over the input similarity ρ
RAMP = LinearSegmentedColormap.from_list("ordinal_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# green ramp over the compared epochs (panels d, e); a single epoch uses its dark end
EPOCH_RAMP = LinearSegmentedColormap.from_list("ordinal_green", ["#a1d99b", "#41ab5d", "#00441b"])
EPOCH_COLOR = "#238b45"
PANEL_MM = PANEL_MM_DEFAULT
# Figure layout in mm: full page width, two rows of 22 mm high panels.
FIG_MM = (180, 78)
LEFT_MM, RIGHT_MM = 12, 10       # right margin holds the ρ colorbar
GAP_MM = 13                      # between panels with their own y label
PAIR_GAP_MM = 6                  # between the MSP and TSP panel of a pair
ROW_H_MM = 22
BOTTOM_Y_MM, TOP_Y_MM = 9, 9 + ROW_H_MM + 13
LEGEND_MM = 5                    # legend baseline below the top edge
TAG_MM = 11                      # panel tags left of the axes


def _row_width(gaps: list[float], n: int) -> float:
    """Panel width (mm) for *n* panels with *gaps* between them across the figure."""
    return (FIG_MM[0] - LEFT_MM - RIGHT_MM - sum(gaps)) / n


def _row(fig, y_mm: float, w_mm: float, gaps: list[float]) -> list:
    axes, x = [], LEFT_MM
    for i in range(len(gaps) + 1):
        axes.append(fig.add_axes([x / FIG_MM[0], y_mm / FIG_MM[1],
                                  w_mm / FIG_MM[0], ROW_H_MM / FIG_MM[1]]))
        x += w_mm + (gaps[i] if i < len(gaps) else 0)
    return axes


# ── loading & stats ─────────────────────────────────────────────────────────

def load_runs(sweep_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(sweep_dir, "**", "results.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No results.npz found below {sweep_dir}.")
    runs = [dict(np.load(f)) for f in files]
    ref = runs[0]
    if "recall_margin" not in ref:
        raise ValueError(f"{files[0]} has no per-readout evaluation; re-run scripts/tasks/wammes2021.py.")
    for f, r in zip(files, runs):
        if not np.array_equal(r["eval_steps"], ref["eval_steps"]) or len(r["loss"]) != len(ref["loss"]):
            raise ValueError(f"{f} has a different step/eval schedule; sweep mixes configs?")
    return {
        "n_seeds": len(runs),
        "eval_steps": ref["eval_steps"],
        "steps_per_epoch": int(ref["steps_per_epoch"]),
        "rhos": ref["rhos"],
        "stream_sparse": ref["stream_sparse"],
        "loss": np.stack([r["loss"] for r in runs]),            # (seeds, n_steps)
        "pair_corr": np.stack([r["pair_corr"] for r in runs]),  # (seeds, T, S, n)
        "baseline": np.stack([r["baseline"] for r in runs]),    # (seeds, T, S, m)
        # readouts: summed output, then each stream alone
        "eval_loss": np.stack([r["eval_loss"] for r in runs]),          # (seeds, T, R)
        "recall_margin": np.stack([r["recall_margin"] for r in runs]),  # (seeds, T, R, 2n)
    }


def mean_sem(x: np.ndarray, axis: int = 0):
    """Mean and standard error of the mean along *axis* (NaNs ignored)."""
    n = np.sum(~np.isnan(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    if x.shape[axis] < 2:
        return m, np.zeros_like(m)
    return m, np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)


def band(ax, x, m, h, color, **kw):
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, m, color=color, lw=LINEWIDTH, **kw)


def streams(data) -> list[tuple[int, str, object]]:
    """``(stream index, label, line style)``, dense (MSP) first."""
    order = sorted(range(len(data["stream_sparse"])), key=lambda s: data["stream_sparse"][s])
    return [(s, NETWORK_LABELS[int(data["stream_sparse"][s])], STREAM_STYLES[int(data["stream_sparse"][s])])
            for s in order]


def readouts(data, readout: str) -> list[tuple[int, str, object]]:
    """``(readout index, legend label, line style)``; index 0 is the summed output."""
    summed = [(0, "Summed", "-")]
    individual = [(s + 1, label.replace(" network", ""), style) for s, label, style in streams(data)]
    return {"combined": summed, "individual": individual, "both": summed + individual}[readout]


def _nearest_eval(data, epoch: float) -> int:
    target = epoch * data["steps_per_epoch"]
    t = int(np.argmin(np.abs(data["eval_steps"] - target)))
    actual = data["eval_steps"][t]
    if abs(actual - target) > 0.02 * data["steps_per_epoch"]:
        print(f"note: epoch {epoch:g} → nearest evaluation at step {actual} "
              f"(epoch {actual / data['steps_per_epoch']:.2f})")
    return t


# ── figure ──────────────────────────────────────────────────────────────────

def plot_vs_rho(ax, rhos: np.ndarray, values: np.ndarray, colors: list, style) -> None:
    """Mean ± SEM over seeds of *values* (seeds, n_epochs, n) against ρ, one curve per epoch."""
    keep = rhos < 1.0
    r = rhos[keep]
    for k, color in enumerate(colors):
        m, h = mean_sem(values[:, k, keep])
        ax.fill_between(r, m - h, m + h, color=color, alpha=0.2, linewidth=0)
        ax.plot(r, m, color=color, linestyle=style, lw=LINEWIDTH, marker="o", ms=2,
                mec="white", mew=0.3, zorder=3)
    ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    style_axis(ax, [0, 0.5, 1], xlabel="Input similarity ρ")
    ax.set_xlim(rhos.min() - 0.05, r.max() + 0.05)


def make_figure(data, readout: str, epochs: list[float], logx: bool, out_paths: list[str]) -> None:
    spe = data["steps_per_epoch"]
    n_epochs = data["loss"].shape[1] // spe
    xticks = list(range(0, n_epochs + 1, max(1, n_epochs // 4)))
    rhos = data["rhos"]
    norm = Normalize(rhos.min(), rhos.max())
    shown = readouts(data, readout)
    setup_style()
    fig = plt.figure(figsize=(FIG_MM[0] / 25.4, FIG_MM[1] / 25.4), dpi=300)
    # four panels per row over the same width (tight_layout cannot handle the
    # mixed gaps, so the axes are placed in mm)
    top_gaps = [GAP_MM, GAP_MM, PAIR_GAP_MM]
    loss_ax, recall_ax, *sim_axes = _row(fig, TOP_Y_MM, _row_width(top_gaps, 4), top_gaps)
    bottom_gaps = [PAIR_GAP_MM, GAP_MM, PAIR_GAP_MM]
    bottom = _row(fig, BOTTOM_Y_MM, _row_width(bottom_gaps, 4), bottom_gaps)
    change_axes, final_axes = bottom[:2], bottom[2:]
    # constant tag offset in mm left of the axes
    tag_x_top = -TAG_MM / _row_width(top_gaps, 4)
    tag_x_bottom = -TAG_MM / _row_width(bottom_gaps, 4)

    x = data["eval_steps"] / spe
    if logx:
        x = x + 1 / spe

    def time_axis(ax):
        style_axis(ax, xticks)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(x[0], x[-1])

    # a. loss and b. recall per readout
    for r, _, style in shown:
        band(loss_ax, x, *mean_sem(data["eval_loss"][:, :, r]), "black", linestyle=style)
        # proportion of all probed items that are recalled: (seeds, T)
        recall = (data["recall_margin"][:, :, r] > 0).mean(-1)
        band(recall_ax, x, *mean_sem(recall), "black", linestyle=style)
    loss_ax.set_yscale("log")
    time_axis(loss_ax)
    label_axis(loss_ax, "Loss", None, "a", tag_x=tag_x_top)
    recall_ax.set_ylim(-0.03, 1.03)
    recall_ax.set_yticks([0, 0.5, 1])
    reference_line(recall_ax, 0.5)
    time_axis(recall_ax)
    label_axis(recall_ax, "Recall", None, "b", tag_x=tag_x_top)

    # c. pairmate similarity over training, one panel per stream
    for i, (ax, (s, label, _)) in enumerate(zip(sim_axes, streams(data))):
        for j, rho in enumerate(rhos):
            band(ax, x, *mean_sem(data["pair_corr"][:, :, s, j]), RAMP(norm(rho)))
        band(ax, x, *mean_sem(data["baseline"][:, :, s].mean(-1)), BASELINE, linestyle="--")
        time_axis(ax)
        label_axis(ax, "Similarity" if i == 0 else None, label, "c" if i == 0 else None,
                   tag_x=tag_x_top)
    share_ylim(sim_axes)

    # d, e. change relative to initialization and similarity itself, against ρ
    idx = [_nearest_eval(data, e) for e in epochs]
    colors = [EPOCH_RAMP(v) for v in np.linspace(0, 1, len(epochs))] if len(epochs) > 1 else [EPOCH_COLOR]
    for i, (s, label, style) in enumerate(streams(data)):
        pc = data["pair_corr"][:, :, s]           # (seeds, T, n)
        plot_vs_rho(change_axes[i], rhos, pc[:, idx] - pc[:, [0]], colors, style)
        plot_vs_rho(final_axes[i], rhos, pc[:, idx], colors, style)
        label_axis(change_axes[i], "Δ Similarity" if i == 0 else None, label,
                   "d" if i == 0 else None, tag_x=tag_x_bottom)
        label_axis(final_axes[i], "Similarity" if i == 0 else None, label,
                   "e" if i == 0 else None, tag_x=tag_x_bottom)
    share_ylim(change_axes)
    share_ylim(final_axes)
    for ax in (sim_axes[1], change_axes[1], final_axes[1]):
        ax.tick_params(labelleft=False)  # shared with the MSP panel

    # ρ color scale, right of panel c
    cax = sim_axes[-1].inset_axes([1.06, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
    cbar.set_label("ρ", fontsize=plt.rcParams["font.size"], labelpad=1)
    cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 1, width=0.6, length=2, pad=1)
    cbar.outline.set_linewidth(0.6)

    handles = [Line2D([], [], color=BASELINE, linestyle="--", lw=LINEWIDTH, label="Non-pairmates")]
    # summed readout (when shown next to the streams), then the stream styles of e and f
    styles = ([("Summed", "-")] if readout == "both" else []) + \
        [(label.replace(" network", ""), style) for _, label, style in streams(data)]
    handles += [Line2D([], [], color="black", linestyle=style, lw=LINEWIDTH, label=label)
                for label, style in styles]
    if len(epochs) > 1:
        handles += [Line2D([], [], color=c, lw=LINEWIDTH, label=f"Epoch {e:g}")
                    for e, c in zip(epochs, colors)]
    top_legend(fig, handles, y=1 - LEGEND_MM / FIG_MM[1])
    save_figure(fig, out_paths)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dirs", nargs="+",
                   help="Sweep directories (data/<experiment>/<objective>/<model>).")
    p.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                   help="Output used for the loss and recall panels.")
    p.add_argument("--epochs", type=float, nargs="+", default=None,
                   help="epochs shown in panels d and e (default: final epoch)")
    p.add_argument("--logx", action="store_true", help="log-scale epoch axis in panels a-c")
    add_output_args(p)
    args = p.parse_args()
    check_output_args(p, args)

    for sweep_dir in args.sweep_dirs:
        data = load_runs(sweep_dir)
        experiment, objective, model = sweep_parts(sweep_dir)
        stem = os.path.join("figures", experiment, f"{model}_{objective}_{args.readout}")
        final_epoch = data["eval_steps"][-1] / data["steps_per_epoch"]
        print(f"{sweep_dir}: {data['n_seeds']} seeds, {final_epoch:g} epochs, "
              f"{data['steps_per_epoch']} steps/epoch")
        make_figure(data, args.readout, args.epochs or [final_epoch], args.logx,
                    resolve_out_paths(args.out, stem, args.ext))


if __name__ == "__main__":
    main()

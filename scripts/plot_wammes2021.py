"""plot_wammes2021.py — summary figure for a Wammes sweep (mean ± SEM across seeds).

Reads every ``*/results.npz`` below a sweep directory written by
``scripts/tasks/wammes2021.py --multirun`` and produces one figure in the
style of ``plot_favila2016.py``:

a. training loss per epoch
b. pairmate similarity (correlation of the hidden representations) over
   training in the MSP network, one line per input similarity ρ; the
   non-pairmate baseline is shown in grey
c. the same for the TSP network
d. change in pairmate similarity relative to before training, as a function
   of ρ, in the MSP (dashed) and TSP (dotted) network, at the final epoch or
   at the epochs given by ``--epochs`` (darker = later)

Usage
-----
    python scripts/plot_wammes2021.py data/wammes2021/online
    python scripts/plot_wammes2021.py data/wammes2021/online --epochs 1 5 20 --logx
    python scripts/plot_wammes2021.py data/wammes2021/online --out figures/wammes2021/online --ext pdf png
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
    label_axis,
    resolve_out_paths,
    save_figure,
    setup_style,
    share_ylim,
    style_axis,
    top_legend,
)

BASELINE = "#8a8a85"
# ordinal blue ramp over the input similarity ρ
RAMP = LinearSegmentedColormap.from_list("ordinal_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# grey ramp over the compared epochs (panel d)
EPOCH_RAMP = LinearSegmentedColormap.from_list("ordinal_grey", ["#b0b0ab", "#000000"])
PANEL_MM = PANEL_MM_DEFAULT


# ── loading & stats ─────────────────────────────────────────────────────────

def load_runs(sweep_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(sweep_dir, "**", "results.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No results.npz found below {sweep_dir}.")
    runs = [dict(np.load(f)) for f in files]
    ref = runs[0]
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


def _nearest_eval(data, epoch: float) -> int:
    target = epoch * data["steps_per_epoch"]
    t = int(np.argmin(np.abs(data["eval_steps"] - target)))
    actual = data["eval_steps"][t]
    if abs(actual - target) > 0.02 * data["steps_per_epoch"]:
        print(f"note: epoch {epoch:g} → nearest evaluation at step {actual} "
              f"(epoch {actual / data['steps_per_epoch']:.2f})")
    return t


# ── figure ──────────────────────────────────────────────────────────────────

def make_figure(data, epochs: list[float], logx: bool, out_paths: list[str]) -> None:
    spe = data["steps_per_epoch"]
    n_epochs = data["loss"].shape[1] // spe
    xticks = list(range(0, n_epochs + 1, max(1, n_epochs // 4)))
    rhos = data["rhos"]
    norm = Normalize(rhos.min(), rhos.max())

    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(PANEL_MM[0] * 4 / 25.4, PANEL_MM[1] / 25.4), dpi=300)
    loss_ax, change_ax = axes[0], axes[3]

    # a. loss per epoch
    loss = data["loss"][:, : n_epochs * spe].reshape(data["n_seeds"], n_epochs, spe).mean(-1)
    band(loss_ax, np.arange(1, n_epochs + 1), *mean_sem(loss), "black")
    loss_ax.set_yscale("log")
    style_axis(loss_ax, xticks)
    label_axis(loss_ax, "Loss", None, "a")

    # b, c. pairmate similarity over training
    x = data["eval_steps"] / spe
    if logx:
        x = x + 1 / spe
    sim_axes = list(axes[1:3])
    for ax, (s, label, _), tag in zip(sim_axes, streams(data), "bc"):
        for i, rho in enumerate(rhos):
            band(ax, x, *mean_sem(data["pair_corr"][:, :, s, i]), RAMP(norm(rho)))
        band(ax, x, *mean_sem(data["baseline"][:, :, s].mean(-1)), BASELINE, linestyle="--")
        style_axis(ax, xticks)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(x[0], x[-1])
        label_axis(ax, "Similarity" if tag == "b" else None, label, tag)
    share_ylim(sim_axes)

    # d. change relative to initialization
    idx = [_nearest_eval(data, e) for e in epochs]
    colors = [EPOCH_RAMP(v) for v in np.linspace(0, 1, len(epochs))] if len(epochs) > 1 else ["black"]
    keep = rhos < 1.0
    for s, _, style in streams(data):
        pc = data["pair_corr"][:, :, s]           # (seeds, T, n)
        for t, c in zip(idx, colors):
            m, h = mean_sem(pc[:, t] - pc[:, 0])
            change_ax.fill_between(rhos[keep], (m - h)[keep], (m + h)[keep], color=c, alpha=0.2,
                                   linewidth=0)
            change_ax.plot(rhos[keep], m[keep], color=c, linestyle=style, lw=LINEWIDTH, zorder=3)
            change_ax.scatter(rhos[keep], m[keep], c=[RAMP(norm(r)) for r in rhos[keep]], s=6,
                              edgecolors="white", linewidths=0.3, zorder=4)
    change_ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
    style_axis(change_ax, [0, 0.5, 1], xlabel="Input similarity ρ")
    change_ax.set_xlim(rhos.min() - 0.05, rhos[keep].max() + 0.05)
    label_axis(change_ax, "Δ Similarity", None, "d")

    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.95, 0.92))
    # ρ color scale, right of panel d
    cax = change_ax.inset_axes([1.06, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
    cbar.set_label("ρ", fontsize=plt.rcParams["font.size"], labelpad=1)
    cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 1, width=0.6, length=2, pad=1)
    cbar.outline.set_linewidth(0.6)

    handles = [Line2D([], [], color=BASELINE, linestyle="--", lw=LINEWIDTH, label="Non-pairmates")]
    handles += [Line2D([], [], color="black", linestyle=style, lw=LINEWIDTH,
                       label=label.replace(" network", "")) for _, label, style in streams(data)]
    if len(epochs) > 1:
        handles += [Line2D([], [], color=c, lw=LINEWIDTH, label=f"Epoch {e:g}")
                    for e, c in zip(epochs, colors)]
    top_legend(fig, handles, y=0.9)
    save_figure(fig, out_paths)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir")
    p.add_argument("--epochs", type=float, nargs="+", default=None,
                   help="epochs to compare against initialization in panel d (default: final epoch)")
    p.add_argument("--logx", action="store_true", help="log-scale epoch axis in the similarity panels")
    add_output_args(p)
    args = p.parse_args()

    data = load_runs(args.sweep_dir)
    stem = os.path.join("figures", "wammes2021", os.path.basename(os.path.normpath(args.sweep_dir)))
    final_epoch = data["eval_steps"][-1] / data["steps_per_epoch"]
    print(f"{data['n_seeds']} seeds, {final_epoch:g} epochs, {data['steps_per_epoch']} steps/epoch")
    make_figure(data, args.epochs or [final_epoch], args.logx,
                resolve_out_paths(args.out, stem, args.ext))


if __name__ == "__main__":
    main()

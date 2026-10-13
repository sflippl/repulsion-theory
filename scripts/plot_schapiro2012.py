"""plot_schapiro2012.py — representational change over training for a Schapiro sweep.

Reads ``<sweep_dir>/<setting>/<seed>/results.npz`` as written by
``scripts/schapiro2012.sh`` (one setting per (phi_in, phi_out) pair; a
directory that directly contains seed folders is treated as one setting) and
produces, as in ``25_schapiro2012.ipynb`` (mean ± 95% CI across seeds; pairs
averaged within each strong/weak × input-similarity condition):

1. ``change_rel_baseline``  (pairmate − mean non-pairmate similarity), minus its
                            value before training;
2. ``change_raw``           pairmate similarity minus its value before training,
                            with the non-pairmate ("shuffled pairs") change dashed;
3. ``loss``                 training loss per epoch, one line per setting.

Figures 1–2 have one row per (phi_in, phi_out) setting and one column per
pathway (dense / sparse kWTA), on a log epoch axis.

Usage
-----
    python scripts/plot_schapiro2012.py data/schapiro2012/online
    python scripts/plot_schapiro2012.py data/schapiro2012/online --linx --out figures/schapiro2012/online
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from repulsion.tasks.plotting import (
    BASELINE, CATEGORICAL, INK_MUTED, RAMP, band, find_results, load_runs, mean_ci, per_epoch_loss, save,
    setup_style, streams,
)


def load_settings(sweep_dir: str) -> list[dict]:
    """One data dict per setting (grandparent dir of each results.npz), sorted by (phi_in, -phi_out)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in find_results(sweep_dir):
        groups[os.path.dirname(os.path.dirname(f))].append(f)
    settings = [load_runs(files) for files in groups.values()]
    return sorted(settings, key=lambda d: (float(d["phi_in"]), -float(d["phi_out"])))


def conditions(data) -> list[tuple[str, np.ndarray]]:
    """``(label, pair indices)`` per (input similarity, strong/weak) condition, notebook order."""
    out = []
    for rho in np.unique(data["pair_corrs"]):
        for strong in (True, False):
            idx = np.flatnonzero((data["pair_corrs"] == rho) & (data["pair_strong"] == strong))
            if len(idx):
                out.append((f"{'Strong' if strong else 'Weak'}, ρ = {rho:g}", idx))
    return out


def setting_label(data) -> str:
    return f"φ_in = {float(data['phi_in']):g}, φ_out = {float(data['phi_out']):g}"


def epochs_axis(data, logx: bool) -> np.ndarray:
    spe = data["steps_per_epoch"]
    return (data["eval_steps"] + 1) / spe if logx else data["eval_steps"] / spe


def plot_grid(settings: list[dict], out_dir: str, name: str, logx: bool, relative: bool) -> None:
    n_streams = len(settings[0]["stream_sparse"])
    fig, axes = plt.subplots(
        len(settings), n_streams, figsize=(4.6 * n_streams, 2.6 * len(settings)),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    for row, data in zip(axes, settings):
        x = epochs_axis(data, logx)
        for ax, (s, stream_label) in zip(row, streams(data)):
            pc = data["pair_corr"][:, :, s]            # (seeds, T, n)
            base = data["baseline"][:, :, s].mean(-1)  # (seeds, T)
            if relative:
                pc = pc - base[..., None]
            change = pc - pc[:, [0]]
            if not relative:
                m, h = mean_ci(base - base[:, [0]])
                band(ax, x, m, h, BASELINE, label="Shuffled pairs", linestyle="--", lw=1.5)
            for (label, idx), color in zip(conditions(data), CATEGORICAL):
                m, h = mean_ci(change[:, :, idx].mean(-1))
                band(ax, x, m, h, color, label=label)
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_title(f"{stream_label} · {setting_label(data)}", loc="left", fontsize=9.5)
            if logx:
                ax.set_xscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    ylabel = "Δ (pairmate − non-pairmate)\nsimilarity" if relative else "Δ pairmate similarity"
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    save(fig, out_dir, name)


def plot_loss(settings: list[dict], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    colors = [RAMP(v) for v in (np.linspace(0, 1, len(settings)) if len(settings) > 1 else [1.0])]
    for data, color in zip(settings, colors):
        per_epoch = per_epoch_loss(data)
        m, h = mean_ci(per_epoch)
        band(ax, np.arange(1, per_epoch.shape[1] + 1), m, h, color, label=setting_label(data))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.legend(fontsize=8)
    save(fig, out_dir, "loss")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir")
    p.add_argument("--out", default=None, help="output directory (default: figures/schapiro2012/<sweep dir name>)")
    p.add_argument("--linx", action="store_true", help="linear instead of log epoch axis")
    args = p.parse_args()

    setup_style()
    settings = load_settings(args.sweep_dir)
    out_dir = args.out or os.path.join("figures", "schapiro2012", os.path.basename(os.path.normpath(args.sweep_dir)))
    for d in settings:
        print(f"{setting_label(d)}: {d['n_seeds']} seeds, "
              f"{d['eval_steps'][-1] / d['steps_per_epoch']:g} epochs, {d['steps_per_epoch']} steps/epoch")

    plot_grid(settings, out_dir, "change_rel_baseline", logx=not args.linx, relative=True)
    plot_grid(settings, out_dir, "change_raw", logx=not args.linx, relative=False)
    plot_loss(settings, out_dir)


if __name__ == "__main__":
    main()

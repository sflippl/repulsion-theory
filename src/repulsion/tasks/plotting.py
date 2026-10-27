"""Plotting helpers for task-script sweeps (mean ± SEM across seeds).

Imports matplotlib lazily-safe for headless use; call :func:`setup_style`
once before plotting.
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

LINE = "#2a78d6"
BASELINE = "#8a8a85"
INK_MUTED = "#5c5c58"
# fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# ordinal blue ramp (light end still clears 2:1 on white)
RAMP = LinearSegmentedColormap.from_list("ordinal_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def setup_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8a85",
        "axes.labelcolor": "#2b2b29",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
    })


def find_results(sweep_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(sweep_dir, "**", "results.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No results.npz found below {sweep_dir}.")
    return files


def load_runs(files: list[str], stack: tuple[str, ...] = ("loss", "pair_corr", "baseline")) -> dict:
    """Load seeds of one setting; arrays in *stack* get a leading seed axis, the rest come from the first run."""
    runs = [dict(np.load(f)) for f in files]
    ref = runs[0]
    for f, r in zip(files, runs):
        if not np.array_equal(r["eval_steps"], ref["eval_steps"]) or len(r["loss"]) != len(ref["loss"]):
            raise ValueError(f"{f} has a different step/eval schedule; sweep mixes configs?")
    data = {k: v for k, v in ref.items() if k not in stack}
    data.update({k: np.stack([r[k] for r in runs]) for k in stack})
    data["n_seeds"] = len(runs)
    data["steps_per_epoch"] = int(ref["steps_per_epoch"])
    return data


def mean_sem(x: np.ndarray, axis: int = 0):
    """Mean and standard error of the mean along *axis* (NaNs ignored)."""
    n = np.sum(~np.isnan(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    if x.shape[axis] < 2:
        return m, np.zeros_like(m)
    return m, np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)


def band(ax, x, m, h, color, label=None, **kw):
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.2, lw=0)
    ax.plot(x, m, color=color, label=label, **kw)


def streams(data) -> list[tuple[int, str]]:
    """``(stream index, label)`` pairs, dense first."""
    out = [(s, "Sparse pathway (kWTA)" if sp else "Dense pathway") for s, sp in enumerate(data["stream_sparse"])]
    return sorted(out, key=lambda t: data["stream_sparse"][t[0]])


def per_epoch_loss(data) -> np.ndarray:
    spe = data["steps_per_epoch"]
    loss = data["loss"]
    n_epochs = loss.shape[1] // spe
    return loss[:, : n_epochs * spe].reshape(loss.shape[0], n_epochs, spe).mean(-1)


def save(fig, out_dir: str, name: str, exts: tuple[str, ...] = ("pdf", "png")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in exts:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, name)}.{{{','.join(exts)}}}")

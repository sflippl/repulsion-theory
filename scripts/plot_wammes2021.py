"""plot_wammes2021.py — summary figures for a Wammes sweep (mean ± 95% CI across seeds).

Reads every ``*/results.npz`` below a sweep directory written by
``scripts/tasks/wammes2021.py --multirun`` and produces

1. ``loss``         training loss per epoch;
2. ``similarity``   pairmate representational similarity over training, one
                    line per input similarity ρ, for the dense and the sparse
                    (kWTA) pathway; the non-pairmate baseline is shown in gray;
3. ``change``       change in pairmate similarity at the chosen epochs relative
                    to before training, as a function of ρ, for both pathways.

Usage
-----
    python scripts/plot_wammes2021.py data/wammes2021/online
    python scripts/plot_wammes2021.py data/wammes2021/online --epochs 1 5 10 20 --logx
    python scripts/plot_wammes2021.py data/wammes2021/online --out figures/wammes2021/online
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats

LINE = "#2a78d6"
BASELINE = "#8a8a85"
# ordinal blue ramp (light end still clears 2:1 on white)
RAMP = LinearSegmentedColormap.from_list("ordinal_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8a85",
    "axes.labelcolor": "#2b2b29",
    "xtick.color": "#5c5c58",
    "ytick.color": "#5c5c58",
    "lines.linewidth": 2,
    "legend.frameon": False,
    "font.size": 10,
})


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


def mean_ci(x: np.ndarray, axis: int = 0, level: float = 0.95):
    """Mean and half-width of the t-based confidence interval along *axis*."""
    n = x.shape[axis]
    m = np.nanmean(x, axis=axis)
    if n < 2:
        return m, np.zeros_like(m)
    sem = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)
    return m, sem * stats.t.ppf(0.5 + level / 2, n - 1)


def band(ax, x, m, h, color, label=None, **kw):
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.2, lw=0)
    ax.plot(x, m, color=color, label=label, **kw)


def streams(data) -> list[tuple[int, str]]:
    """``(stream index, label)`` pairs, dense first."""
    out = [(s, "Sparse pathway (kWTA)" if sp else "Dense pathway") for s, sp in enumerate(data["stream_sparse"])]
    return sorted(out, key=lambda t: data["stream_sparse"][t[0]])


def save(fig, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, name)}.{{pdf,png}}")


# ── figures ─────────────────────────────────────────────────────────────────

def plot_loss(data, out_dir: str) -> None:
    spe = data["steps_per_epoch"]
    loss = data["loss"]
    n_epochs = loss.shape[1] // spe
    per_epoch = loss[:, : n_epochs * spe].reshape(loss.shape[0], n_epochs, spe).mean(-1)
    m, h = mean_ci(per_epoch)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    band(ax, np.arange(1, n_epochs + 1), m, h, LINE)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title(f"Loss (n = {data['n_seeds']} seeds)", loc="left")
    save(fig, out_dir, "loss")


def plot_similarity(data, out_dir: str, logx: bool) -> None:
    epochs = data["eval_steps"] / data["steps_per_epoch"]
    x = epochs + 1 / data["steps_per_epoch"] if logx else epochs
    rhos = data["rhos"]
    norm = Normalize(rhos.min(), rhos.max())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True, constrained_layout=True)
    for ax, (s, label) in zip(axes, streams(data)):
        for i, rho in enumerate(rhos):
            m, h = mean_ci(data["pair_corr"][:, :, s, i])
            band(ax, x, m, h, RAMP(norm(rho)))
        m, h = mean_ci(data["baseline"][:, :, s].mean(-1))
        band(ax, x, m, h, BASELINE, label="Non-pairmates", linestyle="--", lw=1.5)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Epoch")
        if logx:
            ax.set_xscale("log")
    axes[0].set_ylabel("Pairmate similarity (corr)")
    axes[0].legend(loc="best")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=axes, label="Input similarity ρ")
    save(fig, out_dir, "similarity")


def _nearest_eval(data, epoch: float) -> int:
    target = epoch * data["steps_per_epoch"]
    t = int(np.argmin(np.abs(data["eval_steps"] - target)))
    actual = data["eval_steps"][t]
    if abs(actual - target) > 0.02 * data["steps_per_epoch"]:
        print(f"note: epoch {epoch:g} → nearest evaluation at step {actual} "
              f"(epoch {actual / data['steps_per_epoch']:.2f})")
    return t


def plot_change(data, out_dir: str, epochs: list[float]) -> None:
    rhos = data["rhos"]
    idx = [_nearest_eval(data, e) for e in epochs]
    colors = [RAMP(v) for v in (np.linspace(0, 1, len(epochs)) if len(epochs) > 1 else [1.0])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True, constrained_layout=True)
    for ax, (s, label) in zip(axes, streams(data)):
        pc = data["pair_corr"][:, :, s]           # (seeds, T, n)
        for e, t, c in zip(epochs, idx, colors):
            m, h = mean_ci(pc[:, t] - pc[:, 0])
            ax.fill_between(rhos, m - h, m + h, color=c, alpha=0.2, lw=0)
            ax.plot(rhos, m, color=c, marker="o", ms=5, mec="white", mew=1, label=f"Epoch {e:g}")
        ax.axhline(0, color="#5c5c58", lw=0.8)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Input similarity ρ")
    axes[0].set_ylabel("Δ pairmate similarity\n(after − before training)")
    axes[-1].legend(loc="best")
    save(fig, out_dir, "change")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir")
    p.add_argument("--epochs", type=float, nargs="+", default=None,
                   help="epochs to compare against initialization (default: final epoch)")
    p.add_argument("--out", default=None,
                   help="output directory (default: figures/wammes2021/<sweep dir name>)")
    p.add_argument("--logx", action="store_true", help="log-scale epoch axis in the similarity plot")
    args = p.parse_args()

    data = load_runs(args.sweep_dir)
    out_dir = args.out or os.path.join("figures", "wammes2021", os.path.basename(os.path.normpath(args.sweep_dir)))
    os.makedirs(out_dir, exist_ok=True)
    final_epoch = data["eval_steps"][-1] / data["steps_per_epoch"]
    print(f"{data['n_seeds']} seeds, {final_epoch:g} epochs, {data['steps_per_epoch']} steps/epoch")

    plot_loss(data, out_dir)
    plot_similarity(data, out_dir, args.logx)
    plot_change(data, out_dir, args.epochs or [final_epoch])


if __name__ == "__main__":
    main()

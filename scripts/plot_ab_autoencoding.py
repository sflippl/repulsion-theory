"""One figure per ab_* result (scripts/ab_autoencoding.sh): hidden similarity of
pairmates over training.

Each figure has one panel per probed representation: A-only input, and for
the autoencoder also B-only input. Within group = mean cosine similarity of the two items of a group
(pairmates); across groups = mean cosine similarity of items of different
groups. Lines are means over seeds, bands +-SEM. Titles are built from each
run's saved config (.hydra/config.yaml): network and stimulus set in the figure
title, the within-group / across-group relation of A and B in the panel titles.

A second figure shows only the A representations of all setups: predictive
network in the first row (panel tag A), autoencoder in the second (tag B), one
column per setup (OVERVIEW_ROWS; setups without complete runs are left blank).

A third figure shows the behavioral repulsion score of the *_behavioral setups
in the same layout (BEHAVIORAL_ROWS, BEHAVIORAL_SETUPS; CIFAR B pairmates are
the same image, the second rotated by 90 degrees): the B read out from A alone is
projected onto the plane spanned by the item's B and its pairmate's B, and the
score is its signed angle (degrees) from the item's B, positive when rotated
away from the pairmate's B, averaged over items. A read-out in a random
direction (e.g. at initialization) scores 0 on average, a perfect one exactly 0.

Writes figures/ab-autoencoding/<task>_<setup>.{pdf,png},
figures/ab-autoencoding/a_similarity.{pdf,png} and
figures/ab-autoencoding/behavioral_repulsion.{pdf,png}.

Usage:
    python scripts/plot_ab_autoencoding.py
    python scripts/plot_ab_autoencoding.py data/ab_prediction/relu_cifar --ext pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from plot_common import (FONTSIZE, LINEWIDTH, TICK_FONTSIZE, label_axis, mean_sem,
                         save_figure, setup_style)

SETUPS = ["relu_uncorr", "relu_corr", "linear_cifar", "relu_cifar"]
OVERVIEW_ROWS = [("A", "Pred. network", "data/ab_prediction"), ("B", "Autoencoder", "data/ab_autoencoder")]
DEFAULT_DIRS = [f"{root}/{setup}" for _, _, root in OVERVIEW_ROWS for setup in SETUPS]
BEHAVIORAL_SETUPS = [f"{setup}_behavioral" for setup in SETUPS]
BEHAVIORAL_ROWS = [("A", "Pred. network", "data/ab_prediction", "output_prediction"),
                   ("B", "Autoencoder", "data/ab_autoencoder", "output_a_to_ab")]
OUT_DIR = "figures/ab-autoencoding"
RELATION_COLORS = {"within": "black", "across": "tab:orange"}  # as in residual_repulsion.ipynb
RELATION_LABELS = {"within": "Within group", "across": "Across groups"}
SHORT = {"same_image": "same img.", "same_category": "same cat.", "different_category": "diff. cat."}
ACTIVATIONS = {"relu": "ReLU", "identity": "Linear"}
PANEL_MM = (50, 30)
OVERVIEW_PANEL_MM = (36, 30)


def has_probe(run_dir: str, probe: str) -> bool:
    files = [f"representations_{probe}_{k}" for k in ["net0_hidden_0.npy", "rows.json", "steps.npy"]]
    return all(os.path.exists(os.path.join(run_dir, f)) for f in files)


def is_complete(run_dir: str) -> bool:
    return has_probe(run_dir, "a") and os.path.exists(os.path.join(run_dir, ".hydra", "config.yaml"))


def seed_dirs(sweep_dir: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(sweep_dir, "*")) if is_complete(d))


def similarities(run_dir: str, probe: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Steps and within-/across-group mean cosine similarity at each step."""
    H = np.load(os.path.join(run_dir, f"representations_{probe}_net0_hidden_0.npy"))  # (steps, items, hidden)
    rows = json.load(open(os.path.join(run_dir, f"representations_{probe}_rows.json")))
    steps = np.load(os.path.join(run_dir, f"representations_{probe}_steps.npy"))
    Hn = H / np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-12)
    C = Hn @ Hn.transpose(0, 2, 1)
    groups = np.array([r["group_id"] for r in rows])
    same = groups[:, None] == groups[None, :]
    within = same & ~np.eye(len(rows), dtype=bool)
    return steps, {"within": C[:, within].mean(1), "across": C[:, ~same].mean(1)}


def behavioral_scores(run_dir: str, probe: str, metric: str = "angle") -> tuple[np.ndarray, np.ndarray]:
    """Steps and behavioral repulsion score at each step: mean over items of the signed
    angle (degrees) of B_hat, the B read out by output probe *probe*, from B in the plane
    spanned by B and B_pairmate (as the Chanales "response away from competitor"
    metric); positive = rotated away from the pairmate's B.

    metric="score" instead gives the closed-form theory's score in that plane,
    cos(B, B_pairmate) - cos(B_hat, B_pairmate) = cos(alpha) - cos(alpha + angle).
    """
    B_hat = np.load(os.path.join(run_dir, f"{probe}_combined_B.npy"))  # (steps, items, dim)
    B = np.load(os.path.join(run_dir, f"{probe}_target_B.npy"))         # (items, dim)
    rows = json.load(open(os.path.join(run_dir, f"{probe}_rows.json")))
    steps = np.load(os.path.join(run_dir, f"{probe}_steps.npy"))
    groups = [r["group_id"] for r in rows]
    mate = np.array([next(j for j in range(len(rows)) if groups[j] == g and j != i) for i, g in enumerate(groups)])
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    # orthonormal basis of the plane: B along e1 (angle 0), B_pairmate at a positive angle
    e1 = unit(B)
    e2 = unit(B[mate] - (B[mate] * e1).sum(-1, keepdims=True) * e1)
    angle = np.arctan2((B_hat * e2[None]).sum(-1), (B_hat * e1[None]).sum(-1))  # (steps, items)
    if metric == "score":
        alpha = np.arctan2((B[mate] * e2).sum(-1), (B[mate] * e1).sum(-1))       # pairmate's angle (items,)
        return steps, (np.cos(alpha)[None] - np.cos(alpha[None] - angle)).mean(1)
    return steps, -np.degrees(angle).mean(1)


def relation_label(item: dict) -> str:
    """Compact within / across description of one item type from its config."""
    if item.get("stimulus_type") == "image":
        within = item.get("within_group", "same_category")
        across = item.get("across_groups", "different_category")
        if within == across:
            return f"{SHORT[within]} within & across"
        across_short = "diff." if across == "different_category" else SHORT[across]
        return f"{SHORT[within]} within, {across_short} across"
    within = item.get("corr", item.get("corr_within", 0.0))
    across = item.get("corr", item.get("corr_between", 0.0))
    if within == across:
        return rf"$\rho$ = {within:g} within & across"
    return rf"$\rho$ = {within:g} within, {across:g} across"


def load_config(run_dir: str) -> tuple[dict, str, str]:
    """Item configs by name, network label (ReLU/Linear) and stimulus set of a run."""
    cfg = yaml.safe_load(open(os.path.join(run_dir, ".hydra", "config.yaml")))
    items = {it["name"]: it for it in cfg["items"]["items"]}
    stimuli = "CIFAR-100" if any(it.get("stimulus_type") == "image" for it in items.values()) else "Sampled vectors"
    activation = cfg["model"]["networks"][0].get("activation", "identity")
    return items, ACTIVATIONS.get(activation, activation), stimuli


def setup_labels(run_dir: str) -> tuple[str, dict[str, str]]:
    items, network, stimuli = load_config(run_dir)
    return f"{network} network, {stimuli}", {name: relation_label(it) for name, it in items.items()}


def compact_setup_label(run_dir: str) -> str:
    """e.g. "ReLU, ρ = 0.5" (sampled vectors, A correlation) or "Linear, CIFAR-100"."""
    items, network, stimuli = load_config(run_dir)
    if items.get("B", {}).get("rotation"):
        return f"{network}, B rot. {items['B']['rotation']:g}°"
    if stimuli == "CIFAR-100":
        return f"{network}, {stimuli}"
    return f"{network}, {relation_label(items['A']).replace(' within & across', '')}"


def draw_similarity_panel(ax, runs: list[str], probe: str) -> None:
    """Within-/across-group similarity of one probe over steps, mean +- SEM over runs."""
    per_run = [similarities(d, probe) for d in runs]
    x = np.maximum(per_run[0][0], 1)  # log axis: plot step 0 (initialization) at 1
    for rel in ["within", "across"]:
        mean, sem = mean_sem([sims[rel] for _, sims in per_run])
        ax.fill_between(x, mean - sem, mean + sem, color=RELATION_COLORS[rel], alpha=0.25, linewidth=0)
        ax.plot(x, mean, color=RELATION_COLORS[rel], lw=LINEWIDTH)
    ax.set_xscale("log")
    ax.set_xlim(1, x[-1])
    ax.set_xlabel("Steps", fontsize=FONTSIZE, labelpad=1)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1)
    ax.tick_params(axis="x", which="minor", length=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.spines[["top", "right"]].set_visible(False)


def relation_legend(fig, x: float) -> None:
    handles = [Line2D([0], [0], color=RELATION_COLORS[r], lw=LINEWIDTH, label=RELATION_LABELS[r])
               for r in ["within", "across"]]
    fig.legend(handles=handles, loc="center left", fontsize=FONTSIZE, frameon=False,
               bbox_to_anchor=(x, 0.5), handlelength=1.6, borderaxespad=0)


def plot_result(sweep_dir: str, out_paths: list[str]) -> None:
    runs = seed_dirs(sweep_dir)
    title, relations = setup_labels(runs[0])
    probes = [p for p in "ab" if all(has_probe(d, p) for d in runs)]
    fig, axes = plt.subplots(1, len(probes), figsize=(len(probes) * PANEL_MM[0] / 25.4, PANEL_MM[1] / 25.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.3)
    for ax, probe in zip(axes, probes):
        draw_similarity_panel(ax, runs, probe)
        name = probe.upper()
        label_axis(ax, ylabel="Similarity" if probe == "a" else None, title=f"{name}: {relations[name]}")
    task = "Pred. network" if probes == ["a"] else "Autoencoder"
    fig.suptitle(f"{task}, {title} (n = {len(runs)})", fontsize=FONTSIZE, y=1.07)
    relation_legend(fig, 0.91 if len(probes) == 2 else 0.95)
    save_figure(fig, out_paths)


def plot_a_overview(out_paths: list[str]) -> None:
    """A representations only: one row per task (OVERVIEW_ROWS), one column per setup."""
    runs = {(r, c): seed_dirs(f"{root}/{setup}")
            for r, (_, _, root) in enumerate(OVERVIEW_ROWS) for c, setup in enumerate(SETUPS)}
    n_rows, n_cols = len(OVERVIEW_ROWS), len(SETUPS)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False,
                             figsize=(n_cols * OVERVIEW_PANEL_MM[0] / 25.4, n_rows * OVERVIEW_PANEL_MM[1] / 25.4))
    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    for c, setup in enumerate(SETUPS):
        available = [runs[r, c] for r in range(n_rows) if runs[r, c]]
        title = compact_setup_label(available[0][0]) if available else setup
        for r, (tag, row_label, _) in enumerate(OVERVIEW_ROWS):
            ax = axes[r, c]
            if runs[r, c]:
                draw_similarity_panel(ax, runs[r, c], "a")
            else:
                ax.set_axis_off()
            if r < n_rows - 1:
                ax.set_xlabel("")
            label_axis(ax, ylabel="Similarity" if c == 0 else None, title=title if r == 0 else None,
                       tag=tag if c == 0 else None, tag_x=-0.55)
            if c == 0:
                ax.text(-0.55, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=FONTSIZE)
    relation_legend(fig, 0.915)
    save_figure(fig, out_paths)


def plot_behavioral_overview(out_paths: list[str], setups: list[str] = SETUPS) -> None:
    """Behavioral repulsion score over training: one row per task (BEHAVIORAL_ROWS), one column per setup."""
    complete = lambda d, probe: os.path.exists(os.path.join(d, f"{probe}_combined_B.npy")) and \
        os.path.exists(os.path.join(d, ".hydra", "config.yaml"))
    runs = {(r, c): sorted(d for d in glob.glob(f"{root}/{setup}/*") if complete(d, probe))
            for r, (_, _, root, probe) in enumerate(BEHAVIORAL_ROWS) for c, setup in enumerate(setups)}
    n_rows, n_cols = len(BEHAVIORAL_ROWS), len(setups)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=(n_cols * OVERVIEW_PANEL_MM[0] / 25.4, n_rows * OVERVIEW_PANEL_MM[1] / 25.4))
    fig.subplots_adjust(wspace=0.45, hspace=0.25)
    for c, setup in enumerate(setups):
        available = [runs[r, c] for r in range(n_rows) if runs[r, c]]
        title = compact_setup_label(available[0][0]) if available else setup
        for r, (tag, row_label, _, probe) in enumerate(BEHAVIORAL_ROWS):
            ax = axes[r, c]
            if runs[r, c]:
                per_run = [behavioral_scores(d, probe) for d in runs[r, c]]
                x = np.maximum(per_run[0][0], 1)  # log axis: step 0 (initialization) at 1
                mean, sem = mean_sem([s for _, s in per_run])
                ax.axhline(0, color="0.35", linestyle=":", lw=0.9, zorder=1)
                ax.fill_between(x, mean - sem, mean + sem, color="black", alpha=0.25, linewidth=0)
                ax.plot(x, mean, color="black", lw=LINEWIDTH)
                ax.set_xscale("log")
                ax.set_xlim(1, x[-1])
                ax.set_xlabel("Steps" if r == n_rows - 1 else "", fontsize=FONTSIZE, labelpad=1)
                ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1)
                ax.spines[["top", "right"]].set_visible(False)
            else:
                ax.set_axis_off()
            label_axis(ax, ylabel="Behavioral\nrepulsion score (°)" if c == 0 else None,
                       title=title if r == 0 else None, tag=tag if c == 0 else None, tag_x=-0.75)
            if c == 0:
                ax.text(-0.75, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=FONTSIZE)
    save_figure(fig, out_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dirs", nargs="*", help="data/ab_<task>/<setup> directories")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--ext", nargs="+", default=["pdf", "png"])
    args = parser.parse_args()

    setup_style()
    dirs = args.sweep_dirs or sorted(d for pattern in DEFAULT_DIRS for d in glob.glob(pattern))
    for sweep_dir in dirs:
        if not seed_dirs(sweep_dir):
            print(f"Skipping {sweep_dir}: no complete runs")
            continue
        task, setup = sweep_dir.rstrip("/").split("/")[-2:]
        plot_result(sweep_dir, [os.path.join(args.out_dir, f"{task}_{setup}.{e}") for e in args.ext])
    if any(seed_dirs(d) for d in DEFAULT_DIRS):
        plot_a_overview([os.path.join(args.out_dir, f"a_similarity.{e}") for e in args.ext])
    if any(glob.glob(f"{root}/{setup}/*/{probe}_combined_B.npy")
           for _, _, root, probe in BEHAVIORAL_ROWS for setup in BEHAVIORAL_SETUPS):
        plot_behavioral_overview([os.path.join(args.out_dir, f"behavioral_repulsion.{e}") for e in args.ext],
                                 BEHAVIORAL_SETUPS)


if __name__ == "__main__":
    main()

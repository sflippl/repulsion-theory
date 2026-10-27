"""plot_common.py — shared loading and styling for the paper simulation figures.

Used by the ``plot_<experiment>.py`` scripts so that all
figures share one style: 7 pt fonts, thin boxed axes, bold lowercase panel
tags, italic panel titles, and a single legend on top.  Conditions are
encoded by color; readouts (summed output vs. each stream alone) by line
style.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repulsion.schedule import build_training_schedule  # noqa: E402

NETWORK_LABELS = ["MSP network", "TSP network"]
# combined readout solid; individual streams dashed / dense-dotted
STREAM_STYLES = ["--", (0, (1, 1))]
PHASE_SHADE = "#efefef"

FONTSIZE = 7
TICK_FONTSIZE = 6
LINEWIDTH = 1.0
PANEL_MM_DEFAULT = (40, 34)  # (width, height) of one panel in mm


# ── loading ──────────────────────────────────────────────────────────────────

def find_runs(sweep_dir: str, marker: str) -> list[str]:
    """Seed run directories below *sweep_dir* (or *sweep_dir* itself)."""
    if os.path.exists(os.path.join(sweep_dir, marker)):
        return [sweep_dir]
    runs = sorted(
        os.path.join(sweep_dir, d) for d in os.listdir(sweep_dir)
        if os.path.exists(os.path.join(sweep_dir, d, marker))
    )
    if not runs:
        raise FileNotFoundError(f"No runs (containing {marker}) found in {sweep_dir}.")
    return runs


def load_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, ".hydra", "config.yaml")) as f:
        return yaml.safe_load(f)


def load_rows(run_dir: str, probe: str) -> list[dict]:
    with open(os.path.join(run_dir, f"{probe}_rows.json")) as f:
        return json.load(f)


def load_outputs(run_dir: str, probe: str, slot: str, readouts: list[str]) -> dict:
    """{readout: (n_checkpoints, n_rows, dim)} predictions plus 'target' (n_rows, dim)."""
    out = {r: np.load(os.path.join(run_dir, f"{probe}_{r}_{slot}.npy")) for r in readouts}
    out["target"] = np.load(os.path.join(run_dir, f"{probe}_target_{slot}.npy"))
    return out


def load_steps(run_dir: str, probe: str) -> np.ndarray:
    return np.load(os.path.join(run_dir, f"{probe}_steps.npy"))


def phase_step_edges(run_dir: str, cfg: dict) -> tuple[list, list[int]]:
    """Expanded schedule phases and the global step at which each starts/ends.

    Rows per task are read from the probes' ``*_rows.json`` files, so every
    trained task needs at least one output/row_loss/representation probe.
    """
    schedule = build_training_schedule(cfg["schedule"])
    n_rows = {}
    for fname in os.listdir(run_dir):
        stem = fname[: -len("_meta.json")] if fname.endswith("_meta.json") else None
        if stem and os.path.exists(os.path.join(run_dir, f"{stem}_rows.json")):
            with open(os.path.join(run_dir, fname)) as f:
                n_rows[json.load(f)["task"]] = len(load_rows(run_dir, stem))
    step_edges = [0]
    for phase in schedule.phases:
        if phase.epochs is None or len(phase.tasks) != 1:
            raise ValueError("Only single-task, epoch-based phases are supported.")
        task = phase.tasks[0]
        if task not in n_rows:
            raise ValueError(f"No probe with rows for trained task '{task}'.")
        batch = phase.params.batch_size or n_rows[task]
        step_edges.append(step_edges[-1] + phase.epochs * int(np.ceil(n_rows[task] / batch)))
    return list(schedule.phases), step_edges


# ── metrics ──────────────────────────────────────────────────────────────────

def condition_masks(rows: list[dict], conditions: list[str]) -> dict[str, np.ndarray]:
    sgs = np.array([r["subgroup"] for r in rows])
    return {c: sgs == c for c in conditions if (sgs == c).any()}


def by_condition(per_row: np.ndarray, rows: list[dict], conditions: list[str]) -> dict:
    """Average a (n_checkpoints, n_rows) array over the rows of each condition."""
    return {c: per_row[:, m].mean(1) for c, m in condition_masks(rows, conditions).items()}


def loss_by_condition(slot_outputs: list[dict], readout: str, rows: list[dict],
                      conditions: list[str]) -> dict:
    """Summed-over-slots squared error, averaged over rows of each condition."""
    per_row = sum(((o[readout] - o["target"][None]) ** 2).sum(-1) for o in slot_outputs)
    return by_condition(per_row, rows, conditions)


def pairmate_score(reps: np.ndarray, rows: list[dict], conditions: list[str]) -> dict:
    """Mean pairmate correlation minus mean non-pairmate correlation, per condition.

    *reps* is (n_checkpoints, n_rows, dim); pairmates share subgroup and group_id.
    """
    out = {}
    for c, m in condition_masks(rows, conditions).items():
        groups = np.array([r["group_id"] for r in rows])[m]
        same = groups[:, None] == groups[None, :]
        np.fill_diagonal(same, False)
        diff = groups[:, None] != groups[None, :]
        scores = np.empty(reps.shape[0])
        for t in range(reps.shape[0]):
            corr = np.corrcoef(reps[t][m])
            scores[t] = np.nanmean(corr[same]) - np.nanmean(corr[diff])
        out[c] = scores
    return out


def row_loss_by_condition(run_dir: str, probe: str, readout: str,
                          conditions: list[str]) -> dict:
    """Per-condition loss from a ``save: row_loss`` probe (summed over slots)."""
    per_row = np.load(os.path.join(run_dir, f"{probe}_{readout}.npy")).sum(-1)
    return by_condition(per_row, load_rows(run_dir, probe), conditions)


# ── readouts ─────────────────────────────────────────────────────────────────

def readout_setup(n_nets: int, readout: str) -> tuple[list[str], dict, list[str]]:
    """Readouts to plot, their line styles, and the stream labels."""
    individual = [f"net{i}" for i in range(n_nets)]
    readouts = {"combined": ["combined"], "individual": individual,
                "both": ["combined"] + individual}[readout]
    styles = {"combined": "-"}
    styles.update({f"net{i}": STREAM_STYLES[i % len(STREAM_STYLES)] for i in range(n_nets)})
    net_labels = [NETWORK_LABELS[i] if i < len(NETWORK_LABELS) else f"net{i + 1}"
                  for i in range(n_nets)]
    return readouts, styles, net_labels


def legend_handles(conditions: list[str], colors: dict, labels: dict,
                   readouts: list[str], styles: dict, net_labels: list[str],
                   always_show_streams: bool = False) -> list[Line2D]:
    """Condition colors, then readout line styles (when more than one is shown)."""
    handles = [Line2D([], [], color=colors[c], lw=LINEWIDTH, label=labels[c]) for c in conditions]
    readout_labels = {"combined": "Summed"}
    readout_labels.update({f"net{i}": lab.replace(" network", "") for i, lab in enumerate(net_labels)})
    individual = [f"net{i}" for i in range(len(net_labels))]
    if always_show_streams:
        shown = (["combined"] if "combined" in readouts and len(readouts) > 1 else []) + individual
    else:
        shown = [] if readouts == ["combined"] else readouts
    handles += [Line2D([], [], color="black", linestyle=styles[r], lw=LINEWIDTH,
                       label=readout_labels[r]) for r in shown]
    return handles


# ── plotting ─────────────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({"font.size": FONTSIZE, "mathtext.default": "regular"})


def mean_sem(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mat = np.stack(arrays)
    sem = mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else np.zeros(mat.shape[1])
    return mat.mean(0), sem


def plot_conditions(ax, x: np.ndarray, runs: list[dict], key, conditions: list[str],
                    colors: dict, linestyle="-") -> None:
    """Mean ± SEM across runs of ``run[key][condition]``, one line per condition."""
    for c in conditions:
        if c not in runs[0][key]:
            continue
        mean, sem = mean_sem([r[key][c] for r in runs])
        ax.plot(x, mean, color=colors[c], linestyle=linestyle, lw=LINEWIDTH)
        if np.nan_to_num(sem).any():
            ax.fill_between(x, mean - sem, mean + sem, color=colors[c], alpha=0.2, linewidth=0)


def style_axis(ax, xticks: list[float], shade_edges: list[float] | None = None,
               xlabel: str | None = "Epoch") -> None:
    """Fix x-limits/ticks, optionally shade every second phase, set paper fonts."""
    if shade_edges is not None:
        for i, (lo, hi) in enumerate(zip(shade_edges[:-1], shade_edges[1:])):
            if i % 2 == 1:
                ax.axvspan(lo, hi, color=PHASE_SHADE, zorder=0, linewidth=0)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel or "", fontsize=FONTSIZE, labelpad=0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def label_axis(ax, ylabel: str | None = None, title: str | None = None,
               tag: str | None = None, tag_x: float = -0.42) -> None:
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=1)
    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE, style="italic", pad=2)
    if tag is not None:
        ax.text(tag_x, 1.06, tag, transform=ax.transAxes, fontsize=FONTSIZE + 2,
                fontweight="bold", va="bottom", ha="left")


def reference_line(ax, y: float) -> None:
    ax.axhline(y, color="grey", linestyle=":", lw=LINEWIDTH, zorder=1)


def share_ylim(axes) -> None:
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def top_legend(fig, handles: list[Line2D], y: float = 0.94) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=FONTSIZE,
               frameon=False, bbox_to_anchor=(0.5, y), handlelength=1.6,
               columnspacing=1.2, borderaxespad=0)


def save_figure(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {out_path}")


def sweep_parts(sweep_dir: str) -> tuple[str, str, str]:
    """(experiment, objective, model) from ``data/<experiment>/<objective>/<model>[/<seed>]``."""
    parts = os.path.normpath(sweep_dir).split(os.sep)
    if parts[-1].isdigit():
        parts = parts[:-1]
    return parts[-3], parts[-2], parts[-1]


def default_out_path(sweep_dir: str, readout: str, ext: str) -> str:
    experiment, objective, model = sweep_parts(sweep_dir)
    return os.path.join("figures", experiment, f"{model}_{objective}_{readout}.{ext}")

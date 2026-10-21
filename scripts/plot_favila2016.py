"""plot_favila2016.py — summary figure for Favila et al. (2016) simulations.

One figure per sweep directory (one model × one training objective), mean ±
SEM across seeds, in the style of the paper figures:

a. scene-face autoencoding loss                       (Diff., Same)
b. strict face-recall accuracy vs. the pairmate face (Diff.) and vs.
   non-pairmate faces (Diff., Same)
c. soft face recall, same comparisons
d. pairmate score in the MSP (net1, dashed) and TSP (net2, dash-dot) network
   hidden layer                                       (No, Diff., Same)

Face recall presents only the scene and reads out the face slot.  The
competitor is the pairmate's face, or the nearest face belonging to another
pair.  Strict accuracy: the output is closer (Euclidean) to the scene's own
face than to the competitor.  Soft recall: 1 - d_own / (d_own + d_comp),
which is 1 for a perfect recall, 0.5 when equidistant, and > 0.5 exactly
when the strict test is passed.  The pairmate score is the mean
correlation between pairmates minus the mean correlation between
non-pairmates within a condition.

The x-axis is in epochs of the respective phase; the scene-only phase is
lightly shaded.

Usage
-----
    python scripts/plot_favila2016.py data/favila2016/summed/complete_model_linear
    python scripts/plot_favila2016.py data/favila2016/*/* --readout both
    python scripts/plot_favila2016.py data/favila2016/independent/complete_model_relu \\
        --readout individual --out figures/favila2016/relu_independent.pdf

``--readout`` selects which output is evaluated for the loss and accuracy
panels: ``combined`` (sum of streams), ``individual`` (each stream alone), or
``both``.
"""
from __future__ import annotations

import argparse
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

FAVILA_SG_COLORS = {
    'no face':        'grey',
    'different face': 'blue',
    'same face':      'red',
}
FAVILA_LABELS = {
    'no face':        'No',
    'different face': 'Diff.',
    'same face':      'Same',
}
CONDITIONS = ["no_face", "different_face", "same_face"]
NETWORK_LABELS = ["MSP network", "TSP network"]
# combined readout solid; individual streams dashed / dash-dot
STREAM_STYLES = ["--", "-.", ":"]
PHASE_SHADE = "#efefef"

FONTSIZE = 7
TICK_FONTSIZE = 6
LINEWIDTH = 1.0
PANEL_MM = (40, 34)


# ── loading ──────────────────────────────────────────────────────────────────

def find_runs(sweep_dir: str) -> list[str]:
    """Seed run directories below *sweep_dir* (or *sweep_dir* itself)."""
    marker = "representations_steps.npy"
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


def steps_to_epochs(run_dir: str, cfg: dict, steps: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Map global steps to cumulative epochs; also return phase boundaries in epochs."""
    schedule = build_training_schedule(cfg["schedule"])
    n_rows = {}
    for fname in os.listdir(run_dir):
        if fname.endswith("_meta.json") and os.path.exists(
            os.path.join(run_dir, fname.replace("_meta.json", "_rows.json"))
        ):
            with open(os.path.join(run_dir, fname)) as f:
                task = json.load(f)["task"]
            n_rows[task] = len(load_rows(run_dir, fname[: -len("_meta.json")]))
    step_edges, epoch_edges = [0], [0.0]
    for phase in schedule.phases:
        if phase.epochs is None or len(phase.tasks) != 1:
            raise ValueError("Only single-task, epoch-based phases are supported.")
        batch = phase.params.batch_size or n_rows[phase.tasks[0]]
        steps_per_epoch = int(np.ceil(n_rows[phase.tasks[0]] / batch))
        step_edges.append(step_edges[-1] + phase.epochs * steps_per_epoch)
        epoch_edges.append(epoch_edges[-1] + phase.epochs)
    return np.interp(steps, step_edges, epoch_edges), epoch_edges


# ── metrics ──────────────────────────────────────────────────────────────────

def condition_masks(rows: list[dict]) -> dict[str, np.ndarray]:
    sgs = np.array([r["subgroup"] for r in rows])
    return {c: sgs == c for c in CONDITIONS if (sgs == c).any()}


def loss_by_condition(slot_outputs: list[dict], readout: str, rows: list[dict]) -> dict:
    """Summed-over-slots squared error, averaged over rows of each condition."""
    per_row = sum(((o[readout] - o["target"][None]) ** 2).sum(-1) for o in slot_outputs)
    return {c: per_row[:, m].mean(1) for c, m in condition_masks(rows).items()}


def association_recall(face_out: dict, readout: str, rows: list[dict]) -> dict:
    """Face recall vs. the pairmate's face ("pm") and vs. non-pairmate faces ("npm").

    "acc_*": strict test, own face closer than the competitor(s).
    "soft_*": 1 - d_own / (d_own + d_comp), with d_comp the distance to the
    pairmate's face or to the nearest non-pairmate face (0.5 = equidistant).
    """
    target = face_out["target"]
    faces, face_idx = np.unique(np.round(target, 8), axis=0, return_inverse=True)
    face_idx = face_idx.ravel()
    pair = [(r["subgroup"], r["group_id"]) for r in rows]
    pair_faces = {p: {face_idx[i] for i, q in enumerate(pair) if q == p} for p in set(pair)}
    pred = face_out[readout]                                     # (T, N, D)
    dist = np.linalg.norm(pred[:, :, None, :] - faces[None, None], axis=-1)  # (T, N, F)
    n = len(rows)
    own = dist[:, np.arange(n), face_idx]                        # (T, N)

    d_pairmate = np.full(own.shape, np.nan)
    d_nonpair = np.empty(own.shape)
    for i, p in enumerate(pair):
        mates = [j for j, q in enumerate(pair) if q == p and j != i]
        mate_faces = {face_idx[j] for j in mates} - {face_idx[i]}
        if mate_faces:
            d_pairmate[:, i] = dist[:, i, sorted(mate_faces)].min(-1)
        others = sorted(set(range(len(faces))) - pair_faces[p])
        d_nonpair[:, i] = dist[:, i, others].min(-1)

    masks = condition_masks(rows)
    out = {}
    for name, d_comp in (("pm", d_pairmate), ("npm", d_nonpair)):
        valid = {c: m for c, m in masks.items() if not np.isnan(d_comp[:, m]).any()}
        out[f"acc_{name}"] = {c: (own[:, m] < d_comp[:, m]).mean(1) for c, m in valid.items()}
        soft = 1 - own / (own + d_comp)
        out[f"soft_{name}"] = {c: soft[:, m].mean(1) for c, m in valid.items()}
    return out


def pairmate_score(reps: np.ndarray, rows: list[dict]) -> dict:
    """Mean pairmate correlation minus mean non-pairmate correlation, per condition."""
    out = {}
    for c, m in condition_masks(rows).items():
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


def compute_run(run_dir: str, readouts: list[str], layer: str) -> dict:
    cfg = load_config(run_dir)
    n_nets = len(cfg["model"]["networks"])
    res = {"cfg": cfg, "n_nets": n_nets}

    steps = np.load(os.path.join(run_dir, "representations_steps.npy"))
    res["epochs"], res["phase_edges"] = steps_to_epochs(run_dir, cfg, steps)
    for probe in ("output_autoencoding", "output_face_prediction"):
        if not np.array_equal(np.load(os.path.join(run_dir, f"{probe}_steps.npy")), steps):
            raise ValueError(f"{probe} was evaluated at different steps than representations.")

    rows_sf = load_rows(run_dir, "output_autoencoding")
    rows_fp = load_rows(run_dir, "output_face_prediction")
    sf = [load_outputs(run_dir, "output_autoencoding", s, readouts) for s in ("face", "scene")]
    fp = load_outputs(run_dir, "output_face_prediction", "face", readouts)

    for r in readouts:
        res[("loss_sf", r)] = loss_by_condition(sf, r, rows_sf)
        for metric, values in association_recall(fp, r, rows_fp).items():
            res[(metric, r)] = values

    rows_rep = load_rows(run_dir, "representations")
    for ni in range(n_nets):
        reps = np.load(os.path.join(run_dir, f"representations_net{ni}_{layer}.npy"))
        res[("pairmate", ni)] = pairmate_score(reps, rows_rep)
    return res


# ── plotting ─────────────────────────────────────────────────────────────────

def mean_sem(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mat = np.stack(arrays)
    sem = mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else np.zeros(mat.shape[1])
    return mat.mean(0), sem


def condition_color(c: str) -> str:
    return FAVILA_SG_COLORS[c.replace("_", " ")]


def plot_metric(ax, runs: list[dict], key, linestyle: str = "-") -> None:
    x = runs[0]["epochs"]
    for c in CONDITIONS:
        if c not in runs[0][key]:
            continue
        mean, sem = mean_sem([r[key][c] for r in runs])
        ax.plot(x, mean, color=condition_color(c), linestyle=linestyle, lw=LINEWIDTH)
        if sem.any():
            ax.fill_between(x, mean - sem, mean + sem, color=condition_color(c),
                            alpha=0.2, linewidth=0)


def style_axis(ax, edges: list[float]) -> None:
    """Shade every second training phase, fix x-limits, set paper fonts."""
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 1:
            ax.axvspan(lo, hi, color=PHASE_SHADE, zorder=0, linewidth=0)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(edges)
    ax.set_xlabel("Epoch", fontsize=FONTSIZE, labelpad=0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def make_figure(sweep_dir: str, readout: str, layer: str, out_path: str) -> None:
    runs_dirs = find_runs(sweep_dir)
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    individual = [f"net{i}" for i in range(n_nets)]
    readouts = {"combined": ["combined"], "individual": individual,
                "both": ["combined"] + individual}[readout]
    runs = [compute_run(d, readouts, layer) for d in runs_dirs]
    net_labels = [NETWORK_LABELS[i] if i < len(NETWORK_LABELS) else f"net{i + 1}"
                  for i in range(n_nets)]
    styles = {"combined": "-"}
    styles.update({f"net{i}": STREAM_STYLES[i % len(STREAM_STYLES)] for i in range(n_nets)})
    edges = runs[0]["phase_edges"]

    # (row, col, tag, metric, title, y label)
    panels = [
        (0, 0, "a", "loss_sf", None, "Loss"),
        (0, 1, "b", "acc_pm", "vs. pairmate face", "Accuracy"),
        (0, 2, None, "acc_npm", "vs. non-pairmate face", None),
        (1, 0, "c", "soft_pm", "vs. pairmate face", "Soft recall"),
        (1, 1, None, "soft_npm", "vs. non-pairmate face", None),
        (1, 2, "d", "pairmate", None, "Pairmate score"),
    ]

    plt.rcParams.update({"font.size": FONTSIZE, "mathtext.default": "regular"})
    fig, grid = plt.subplots(
        2, 3, figsize=(PANEL_MM[0] * 3 / 25.4, PANEL_MM[1] * 2 / 25.4), dpi=300,
    )
    axes = {}
    for row, col, tag, metric, title, ylabel in panels:
        ax = axes[metric] = grid[row, col]
        if metric == "pairmate":
            # one line style per stream, matching the per-stream readouts
            for ni in range(n_nets):
                plot_metric(ax, runs, (metric, ni), linestyle=styles[f"net{ni}"])
            ax.axhline(0, color="lightgrey", linestyle="-", lw=0.6, zorder=1)
        else:
            for r in readouts:
                plot_metric(ax, runs, (metric, r), linestyle=styles[r])
        if metric.startswith("acc"):
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
        if metric in ("acc_pm", "soft_pm", "soft_npm"):
            ax.axhline(0.5, color="grey", linestyle=":", lw=LINEWIDTH, zorder=1)
        if metric == "loss_sf":
            ax.set_ylim(bottom=0)
        style_axis(ax, edges)
        if row == 0:
            ax.set_xlabel("")
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=1)
        if title is not None:
            ax.set_title(title, fontsize=FONTSIZE, style="italic", pad=2)
        if tag is not None:
            ax.text(-0.42, 1.06, tag, transform=ax.transAxes, fontsize=FONTSIZE + 2,
                    fontweight="bold", va="bottom", ha="left")

    # Soft recall: adaptive, but shared between the two panels.
    soft_axes = [axes["soft_pm"], axes["soft_npm"]]
    lo = min(ax.get_ylim()[0] for ax in soft_axes)
    hi = max(ax.get_ylim()[1] for ax in soft_axes)
    for ax in soft_axes:
        ax.set_ylim(lo, hi)

    fig.tight_layout(w_pad=0.8, h_pad=0.4, rect=(0, 0, 1, 0.93))

    handles = [Line2D([], [], color=condition_color(c), lw=LINEWIDTH,
                      label=FAVILA_LABELS[c.replace("_", " ")]) for c in CONDITIONS]
    readout_labels = {"combined": "Summed"}
    readout_labels.update({f"net{i}": net_labels[i].replace(" network", "") for i in range(n_nets)})
    # Stream line styles are always needed for the pairmate-score panel.
    shown = (["combined"] if "combined" in readouts and len(readouts) > 1 else []) + individual
    handles += [Line2D([], [], color="black", linestyle=styles[r], lw=LINEWIDTH,
                       label=readout_labels[r]) for r in shown]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=FONTSIZE,
               frameon=False, bbox_to_anchor=(0.5, 0.94), handlelength=1.6,
               columnspacing=1.2, borderaxespad=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {out_path}")


def default_out_path(sweep_dir: str, readout: str, ext: str) -> str:
    parts = os.path.normpath(sweep_dir).split(os.sep)
    if parts[-1].isdigit():
        parts = parts[:-1]
    objective, model = parts[-2], parts[-1]
    return os.path.join("figures", "favila2016", f"{model}_{objective}_{readout}.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", nargs="+",
                        help="Sweep directories (data/favila2016/<objective>/<model>) or single runs.")
    parser.add_argument("--readout", choices=["combined", "individual", "both"], default="both",
                        help="Output used for loss and accuracy panels.")
    parser.add_argument("--layer", default="hidden_0", help="Layer for the pairmate score.")
    parser.add_argument("--out", default=None, help="Output path (only with a single sweep dir).")
    parser.add_argument("--ext", default="pdf", help="File extension for default output paths.")
    args = parser.parse_args()
    if args.out and len(args.sweep_dirs) > 1:
        parser.error("--out can only be used with a single sweep directory.")
    for d in args.sweep_dirs:
        make_figure(d, args.readout, args.layer, args.out or default_out_path(d, args.readout, args.ext))


if __name__ == "__main__":
    main()

"""plot_favila2016.py — summary figure for Favila et al. (2016) simulations.

One figure per sweep directory (one model × one training objective), mean ±
SEM across seeds.  Panels:

1. scene-face autoencoding loss   (different-face, same-face)
2. scene autoencoding loss        (no-face, different-face, same-face)
3. association accuracy vs. pairmate face        (different-face)
4. association accuracy vs. all non-pairmate faces  (different-face, same-face)
5+. pairmate score for each stream's hidden layer   (all three conditions)

Association accuracy presents only the scene and reads out the face slot.  A
trial is correct if the output is closer (Euclidean) to the scene's own face
than to the competitor(s): the pairmate's face (panel 3), or every face
belonging to another pair (panel 4, a strict nearest-neighbour test).  The
pairmate score is the mean correlation between pairmates minus the mean
correlation between non-pairmates within a condition.

The x-axis is in epochs of the respective phase; the scene-only phase is
shaded.

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

CONDITIONS = ["no_face", "different_face", "same_face"]
CONDITION_LABELS = {"no_face": "No face", "different_face": "Different face", "same_face": "Same face"}
CONDITION_COLORS = {"no_face": "#8a8a85", "different_face": "#eb6834", "same_face": "#2a78d6"}
# combined readout solid; individual streams dashed / dotted
STREAM_STYLES = ["--", ":", "-."]
PHASE_SHADE = "#f1f0ec"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8a85",
    "axes.labelcolor": "#2b2b29",
    "axes.titlesize": 10,
    "xtick.color": "#5c5c58",
    "ytick.color": "#5c5c58",
    "lines.linewidth": 2,
    "legend.frameon": False,
    "font.size": 9,
})


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


def association_accuracy(face_out: dict, readout: str, rows: list[dict]) -> tuple[dict, dict]:
    """Strict accuracy vs. the pairmate's face and vs. all non-pairmate faces."""
    target = face_out["target"]
    faces, face_idx = np.unique(np.round(target, 8), axis=0, return_inverse=True)
    face_idx = face_idx.ravel()
    pair = [(r["subgroup"], r["group_id"]) for r in rows]
    pair_faces = {p: {face_idx[i] for i, q in enumerate(pair) if q == p} for p in set(pair)}
    pred = face_out[readout]                                     # (T, N, D)
    dist = np.linalg.norm(pred[:, :, None, :] - faces[None, None], axis=-1)  # (T, N, F)
    n = len(rows)
    own = dist[:, np.arange(n), face_idx]                        # (T, N)

    vs_pairmate = np.full(own.shape, np.nan)
    vs_nonpair = np.empty(own.shape)
    for i, p in enumerate(pair):
        mates = [j for j, q in enumerate(pair) if q == p and j != i]
        mate_faces = {face_idx[j] for j in mates} - {face_idx[i]}
        if mate_faces:
            vs_pairmate[:, i] = own[:, i] < dist[:, i, sorted(mate_faces)].min(-1)
        others = sorted(set(range(len(faces))) - pair_faces[p])
        vs_nonpair[:, i] = own[:, i] < dist[:, i, others].min(-1)

    masks = condition_masks(rows)
    pm = {c: vs_pairmate[:, m].mean(1) for c, m in masks.items() if not np.isnan(vs_pairmate[:, m]).any()}
    npm = {c: vs_nonpair[:, m].mean(1) for c, m in masks.items()}
    return pm, npm


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
    for probe in ("output_autoencoding", "output_scene", "output_face_prediction"):
        if not np.array_equal(np.load(os.path.join(run_dir, f"{probe}_steps.npy")), steps):
            raise ValueError(f"{probe} was evaluated at different steps than representations.")

    rows_sf = load_rows(run_dir, "output_autoencoding")
    rows_s = load_rows(run_dir, "output_scene")
    rows_fp = load_rows(run_dir, "output_face_prediction")
    sf = [load_outputs(run_dir, "output_autoencoding", s, readouts) for s in ("face", "scene")]
    s = [load_outputs(run_dir, "output_scene", "scene", readouts)]
    fp = load_outputs(run_dir, "output_face_prediction", "face", readouts)

    for r in readouts:
        res[("loss_sf", r)] = loss_by_condition(sf, r, rows_sf)
        res[("loss_s", r)] = loss_by_condition(s, r, rows_s)
        res[("acc_pm", r)], res[("acc_npm", r)] = association_accuracy(fp, r, rows_fp)

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


def plot_metric(ax, runs: list[dict], key, linestyle: str = "-", linewidth: float = 2) -> None:
    x = runs[0]["epochs"]
    for c in CONDITIONS:
        if c not in runs[0][key]:
            continue
        mean, sem = mean_sem([r[key][c] for r in runs])
        color = CONDITION_COLORS[c]
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth)
        if sem.any():
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)


def shade_phases(ax, edges: list[float], labels: list[str]) -> None:
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 1:
            ax.axvspan(lo, hi, color=PHASE_SHADE, zorder=0, linewidth=0)
        ax.text((lo + hi) / 2, 1.0, labels[i], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#8a8a85")
    ax.set_xlim(edges[0], edges[-1])


def make_figure(sweep_dir: str, readout: str, layer: str, out_path: str) -> None:
    runs_dirs = find_runs(sweep_dir)
    n_nets = len(load_config(runs_dirs[0])["model"]["networks"])
    individual = [f"net{i}" for i in range(n_nets)]
    readouts = {"combined": ["combined"], "individual": individual,
                "both": ["combined"] + individual}[readout]
    runs = [compute_run(d, readouts, layer) for d in runs_dirs]
    cfg = runs[0]["cfg"]
    net_names = [n["name"] for n in cfg["model"]["networks"]]
    styles = {"combined": ("-", 2.0)}
    styles.update({f"net{i}": (STREAM_STYLES[i % len(STREAM_STYLES)], 1.4) for i in range(n_nets)})
    if readout == "individual":
        styles.update({f"net{i}": (STREAM_STYLES[i % len(STREAM_STYLES)], 2.0) for i in range(n_nets)})

    # (metric, stream index for the pairmate score, title, y label)
    panels = [
        ("loss_sf", None, "Scene-face autoencoding loss", "Loss"),
        ("loss_s", None, "Scene autoencoding loss", "Loss"),
        ("acc_pm", None, "Face recall vs. pairmate face", "Accuracy"),
        ("acc_npm", None, "Face recall vs. non-pairmate faces", "Accuracy"),
    ] + [("pairmate", ni, f"Pairmate score ({net_names[ni]}, {layer})", "Pairmate score")
         for ni in range(n_nets)]
    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows),
                             sharex=True, squeeze=False)
    phase_labels = ["scene-face", "scene"] * len(runs[0]["phase_edges"])

    for ax, (metric, ni, title, ylabel) in zip(axes.flat, panels):
        if metric == "pairmate":
            plot_metric(ax, runs, (metric, ni))
            ax.axhline(0, color="#5c5c58", linewidth=0.8, linestyle="--", zorder=1)
        else:
            for r in readouts:
                ls, lw = styles[r]
                plot_metric(ax, runs, (metric, r), linestyle=ls, linewidth=lw)
        if metric == "acc_pm":
            ax.axhline(0.5, color="#5c5c58", linewidth=0.8, linestyle="--", zorder=1)
        if metric.startswith("acc"):
            ax.set_ylim(-0.03, 1.03)
        if metric.startswith("loss"):
            ax.set_ylim(bottom=0)
        shade_phases(ax, runs[0]["phase_edges"], phase_labels)
        ax.set_title(title, pad=12)
        ax.set_ylabel(ylabel)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)

    handles = [Line2D([], [], color=CONDITION_COLORS[c], label=CONDITION_LABELS[c]) for c in CONDITIONS]
    if len(readouts) > 1 or readout == "individual":
        readout_labels = {"combined": "Summed output"}
        readout_labels.update({f"net{i}": f"{net_names[i]} alone" for i in range(n_nets)})
        handles += [Line2D([], [], color="#5c5c58", linestyle=styles[r][0], label=readout_labels[r])
                    for r in readouts]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01))

    model_name = os.path.basename(os.path.normpath(sweep_dir))
    if model_name.isdigit():  # single run directory passed
        model_name = os.path.basename(os.path.dirname(os.path.normpath(sweep_dir)))
    objective = cfg["training"].get("objective", "summed")
    fig.suptitle(f"{model_name} · {objective} objective · {len(runs)} seed(s)", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
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

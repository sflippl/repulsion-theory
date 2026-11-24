"""Learning trajectories of the pairmate score under blocked (sequential)
training of the main two-stream model (configs/model/complete_model_linear.yaml).

Model: exactly the network used by train.py with model=complete_model_linear,
built with repulsion.tasks.build_model from the yaml. Two streams whose
outputs are summed and fit the target (objective "summed"):
    net1 (dense):  input -> 128 linear hidden -> output
    net2 (sparse): fixed slot-grouped random projection to 10000 units with
                   per-slot kWTA (frac 0.005), normalized to the input norm,
                   -> 128 linear hidden -> output
Training follows the main runs (scripts/favila2016.sh etc.): SGD, lr 0.1,
squared error summed over output dimensions, one optimizer kept across
blocks -- except momentum is 0 (not 0.5), so no update from one block's
item carries over into the next block. With --readout fixed the output layers are frozen and
only the hidden layers are trained. With --objective independent (as
training.objective=independent in the repo) each stream fits the target on
its own and the per-stream losses are summed; the loss row then shows each
stream's own output loss (dense solid, sparse dashed).

Stimuli (autoencoder, phi_in = phi_out = 0.5, G = M = 2): items A1-B1,
A2-B2, A3-B3, A4-B4 with A's mutually orthogonal and corr(B1, B2) =
corr(B3, B4) = rho_sB (rho_sB = 1: A1/A2 share one associate). A and B sit in
separate input slots, [sqrt(phi) A, sqrt(1 - phi) B], and the target is the
input. Item vectors are randomly rotated per seed.

With --task a-only there are no B items: the network (one input slot)
autoencodes A1, A2, A3, A4 with corr(A1, A2) = corr(A3, A4) = --rho-A
(default 0.25, 0.5, 0.75; one column each), presented in the same blocks.
With --task a-to-b the input is the same, but the network predicts mutually
orthogonal (unit-norm) B1, B2, B3, B4 instead of reconstructing A.

Schedules (one block = one item, --steps-per-block updates each, default 100):
    revisit:     1 -> 2 -> 3 -> 4 -> 1 -> 3
    first16:     (1 -> 2 -> 3 -> 4) x 4  (zoom into the start of alternating)
    alternating: 1 -> 2 -> 3 -> 4, then (1 -> 2 -> 3 -> 4) x 10

Optional regularizer on items that are NOT in the current block: with p the
pure-A and pure-B probes of every inactive item (a B probe identical to the
active item's B is skipped), add
    lam/2 * sum_streams || h(p) - h_anchor(p) ||^2
on the hidden representations, where h_anchor uses either the initial
weights ("init") or the weights at the start of the current block ("block").

Pairmate score = mean of the pairmate similarities cos(h(A1), h(A2)) and
cos(h(A3), h(A4)) minus the mean of the four non-pairmate similarities
(A1-A3, A1-A4, A2-A3, A2-A4), hidden reps probed with pure A input
(subtracting the non-pairmate mean removes the sparse stream's kWTA chance
floor). Relative loss = ||y_i - yhat_i||^2 of item i
divided by its value at initialization (1 = nothing learned).

Outputs (under --figures-dir, default figures/sequential-repulsion; stems get
a "-fixed-readout" suffix with --readout fixed, "-<n>steps" with a
non-default --steps-per-block, "-independent" with --objective independent, and an "a-only-" / "a-to-b-" prefix with
--task a-only / a-to-b):
    revisit-trajectories.pdf      (top: pairmate score, bottom: relative loss of all
    alternating-trajectories.pdf   four items; one column per --rho-sB, default 1 and 0.5)
    reg-init-trajectories.pdf, reg-block-trajectories.pdf  (alternating schedule)
    first16-trajectories.pdf      (same, first 16 blocks of alternating)
    reg-init-revisit.pdf, reg-block-revisit.pdf            (zoomed in: revisit schedule)
    reg-summary.csv  (end-of-revisit and end-of-schedule values, all settings)

Usage:
    python scripts/plot_sequential_repulsion.py
    python scripts/plot_sequential_repulsion.py --only revisit alternating
    python scripts/plot_sequential_repulsion.py --only revisit alternating --readout trained fixed
    python scripts/plot_sequential_repulsion.py --only revisit alternating --task a-only
    python scripts/plot_sequential_repulsion.py --only first16 --task a-to-b --steps-per-block 20
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from generate_associative_repulsion_figures import make_group_vectors
from plot_common import (FONTSIZE, LINEWIDTH, label_axis, mean_sem, reference_line,
                         save_figure, setup_style, share_ylim, style_axis, top_legend)
from repulsion.tasks import build_model

MODEL_CONFIG = os.path.join(ROOT, "configs", "model", "complete_model_linear.yaml")
G, M, DA = 2, 2, 64  # DA = item dimension (as in configs/items/default.yaml)
PHI = 0.5
LR, MOMENTUM = 0.1, 0.0
STEPS_PER_BLOCK = 100   # default; at lam <= 1, scaled by lam above (lr divided by lam)
BLOCK_ALPHA = 0.12      # background tint of each block in its item's color
N_RECORD = 50           # recorded points per block

SCHEDULES = {
    "revisit": [0, 1, 2, 3, 0, 2],
    "first16": [0, 1, 2, 3] * 4,
    "alternating": [0, 1, 2, 3] + [0, 1, 2, 3] * 10,
}
STREAM_LABELS = ["Dense", "Sparse"]
STREAM_STYLES = ["-", "--"]
ITEM_COLORS = ["#4477AA", "#CC6677", "#228833", "#CCBB44"]
PAIR_COLORS = ["#882255", "#44AA99"]  # cos(h(A1), h(A2)), cos(h(A3), h(A4))
ITEM_LABELS = {"associative": ["A1B1", "A2B2", "A3B3", "A4B4"], "a-only": ["A1", "A2", "A3", "A4"],
               "a-to-b": [r"A1$\to$B1", r"A2$\to$B2", r"A3$\to$B3", r"A4$\to$B4"]}
SINGLE_SLOT_TASKS = ("a-only", "a-to-b")  # input = A only, rho = corr(A1, A2)


# ====================================================================== #
# Data and training
# ====================================================================== #

def build_data(rho, rng, task="associative"):
    """Inputs X (= autoencoder targets), A probes ZA and B probes ZB, each
    N x 2DA with A in the first slot and B in the second (rho = corr(B1, B2)).
    For task "a-only": X = ZA = the A items (N x DA, rho = corr(A1, A2)), no ZB.
    For task "a-to-b": the same X = ZA, plus targets Y = orthonormal B items."""
    rot = lambda: np.linalg.qr(rng.standard_normal((DA, DA)))[0]
    if task in SINGLE_SLOT_TASKS:
        vA = torch.tensor(make_group_vectors(1.0, 0.0, rho, G, M, DA) @ rot(), dtype=torch.float32)
        data = dict(X=vA, ZA=vA)
        if task == "a-to-b":
            data["Y"] = torch.tensor(rot()[:G * M], dtype=torch.float32)
        return data
    rho_sB = rho
    vA = make_group_vectors(1.0, 0.0, 0.0, G, M, DA) @ rot()
    vB = make_group_vectors(1.0, 0.0, rho_sB, G, M, DA) @ rot()
    ZA = np.hstack([vA, np.zeros_like(vA)])
    ZB = np.hstack([np.zeros_like(vB), vB])
    X = np.sqrt(PHI) * ZA + np.sqrt(1 - PHI) * ZB
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(X=X, ZA=ZA, ZB=ZB).items()}


def build_network(n_slots=2):
    specs = OmegaConf.to_container(OmegaConf.load(MODEL_CONFIG).networks, resolve=True)
    # concatenated input slots (A, B), as in the main pipeline -- not summed
    specs = [{**spec, "sum_input_slots": False} for spec in specs]
    return build_model(specs, in_dim=DA, out_dim=n_slots * DA, n_input_slots=n_slots)


def hidden(model, Z):
    return [net.extract(Z, None, "hidden_0") for net in model.networks]


def inactive_probes(data, item):
    """Pure-A and pure-B probes of all items other than `item` (a B probe
    identical to the active item's B is skipped, e.g. B1 while training A2-B1)."""
    ZA, ZB = data["ZA"], data.get("ZB")
    others = [i for i in range(G * M) if i != item]
    if ZB is None:
        return ZA[others]
    shared = lambda i: torch.cosine_similarity(ZB[i], ZB[item], dim=0) > 0.999
    return torch.cat([ZA[others]] + [ZB[[i]] for i in others if not shared(i)])


def rep_cosines(H):
    Hn = H / H.norm(dim=1, keepdim=True)
    return Hn @ Hn.T


def pairmate_score(C):
    pairs = (C[0, 1] + C[2, 3]) / 2
    nonpairs = C[:2, 2:].mean()
    return (pairs - nonpairs).item()


def simulate(rho, readout, schedule_name, lam=0.0, anchor="init", seed=0, task="associative",
             steps_per_block=STEPS_PER_BLOCK, objective="summed"):
    """Returns a dict of recorded trajectories (x in units of blocks)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    schedule = SCHEDULES[schedule_name]
    data = build_data(rho, np.random.default_rng(seed), task)
    X, Y = data["X"], data.get("Y", data["X"])
    n_slots = 1 if task in SINGLE_SLOT_TASKS else 2
    model = build_network(n_slots)
    if readout == "fixed":
        for net in model.networks:
            net.mlp.net[-1].weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    scale = max(1.0, lam)
    opt = torch.optim.SGD(params, lr=LR / scale, momentum=MOMENTUM)
    n_steps = int(steps_per_block * scale)
    rec_every = max(1, n_steps // N_RECORD)
    init_model = build_network(n_slots)
    init_model.load_state_dict(model.state_dict())
    rec = {"x": [], "err": [], "err_dense": [], "err_sparse": []}
    rec.update({f"{name}{key}": [] for name in ["dense", "sparse"] for key in ["", "_sim12", "_sim34"]})

    @torch.no_grad()
    def record(x):
        rec["x"].append(x)
        rec["err"].append((Y - model(X)).norm(dim=1).numpy())
        for name, net in zip(["dense", "sparse"], model.networks):
            rec["err_" + name].append((Y - net(X)).norm(dim=1).numpy())
        for name, H in zip(["dense", "sparse"], hidden(model, data["ZA"])):
            C = rep_cosines(H)
            rec[name].append(pairmate_score(C))
            rec[name + "_sim12"].append(C[0, 1].item())
            rec[name + "_sim34"].append(C[2, 3].item())

    record(0.0)
    for b, item in enumerate(schedule):
        P = inactive_probes(data, item)
        with torch.no_grad():
            targets = hidden(init_model if anchor == "init" else model, P)
        for t in range(n_steps):
            if objective == "summed":
                loss = ((model(X[[item]]) - Y[[item]]) ** 2).sum()
            else:
                loss = sum(((net(X[[item]]) - Y[[item]]) ** 2).sum() for net in model.networks)
            if lam > 0:
                loss = loss + lam / 2 * sum(((H - T) ** 2).sum() for H, T in zip(hidden(model, P), targets))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (t + 1) % rec_every == 0:
                record(b + (t + 1) / n_steps)
    return {k: np.array(v) for k, v in rec.items()}


def _simulate_cfg(cfg):
    return cfg[:6], simulate(*cfg)  # key without task/steps/objective (one of each per invocation)


def run_all(cfgs, n_workers):
    with Pool(n_workers) as p:
        return dict(p.map(_simulate_cfg, cfgs))


# ====================================================================== #
# Plotting
# ====================================================================== #

def block_edges(schedule):
    """Shade every other block, or every other 4-block cycle for long schedules."""
    step = 1 if len(schedule) <= 8 else 4
    return list(range(0, len(schedule) + 1, step))


def plot_trajectories(results, rhos, readout, schedule_name, seeds, out_paths,
                      item_labels, rho_label=r"\rho_{sB}", block_labels=None, panel_mm=(42, 30),
                      objective="summed"):
    """Top row: pairmate score (dense solid, sparse dashed); middle row: raw
    hidden similarity cos(h(A1), h(A2)) and cos(h(A3), h(A4)); bottom row:
    loss of each of the four items relative to its initial loss (of the summed
    output, or of each stream's own output for the independent objective).
    One column per rho. Mean +/- SEM over seeds."""
    schedule = SCHEDULES[schedule_name]
    setup_style()
    fig, axes = plt.subplots(3, len(rhos), sharex=True, squeeze=False,
                             figsize=(len(rhos) * panel_mm[0] / 25.4, 3 * panel_mm[1] / 25.4))
    for c, rho in enumerate(rhos):
        runs = [results[(rho, readout, schedule_name, 0.0, "init", s)] for s in seeds]
        x = runs[0]["x"]
        ax = axes[0, c]
        reference_line(ax, 0.0)
        for name, ls in zip(["dense", "sparse"], STREAM_STYLES):
            mean, sem = mean_sem([r[name] for r in runs])
            ax.plot(x, mean, color="black", linestyle=ls, lw=LINEWIDTH)
            ax.fill_between(x, mean - sem, mean + sem, color="black", alpha=0.15, linewidth=0)
        style_axis(ax, [0, len(schedule)], xlabel=None)
        label_axis(ax, ylabel="Pairmate score" if c == 0 else None,
                   title=rf"${rho_label}$ = {rho:g}", tag="abcdefgh"[c] if len(rhos) > 1 else None, tag_x=-0.3)
        ax = axes[1, c]
        reference_line(ax, 0.0)
        for key, col in zip(["_sim12", "_sim34"], PAIR_COLORS):
            for name, ls in zip(["dense", "sparse"], STREAM_STYLES):
                mean, sem = mean_sem([r[name + key] for r in runs])
                ax.plot(x, mean, color=col, linestyle=ls, lw=LINEWIDTH)
                ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.15, linewidth=0)
        style_axis(ax, [0, len(schedule)], xlabel=None)
        label_axis(ax, ylabel="Similarity" if c == 0 else None)
        ax = axes[2, c]
        err_keys = [("err", "-")] if objective == "summed" else list(zip(["err_dense", "err_sparse"], STREAM_STYLES))
        for i, col in enumerate(ITEM_COLORS):
            for key, ls in err_keys:
                mean, sem = mean_sem([r[key][:, i] ** 2 / r[key][0, i] ** 2 for r in runs])
                ax.plot(x, mean, color=col, linestyle=ls, lw=LINEWIDTH)
                ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.15, linewidth=0)
        style_axis(ax, [0, len(schedule)],
                   xlabel="Training block" if block_labels is not None else "Block")
        if block_labels is not None:
            ax.set_xticks(np.arange(len(schedule)) + 0.5)
            ax.set_xticklabels(block_labels, fontsize=FONTSIZE - 2, rotation=90)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticks(list(range(0, len(schedule) + 1, 4 if len(schedule) <= 16 else 8)))
        label_axis(ax, ylabel="Relative loss" if c == 0 else None)
    for ax in axes.flat:
        for b, item in enumerate(schedule):
            ax.axvspan(b, b + 1, color=ITEM_COLORS[item], alpha=BLOCK_ALPHA, linewidth=0, zorder=0)
    share_ylim(axes[0])
    share_ylim(axes[1])
    share_ylim(axes[2])
    handles = [Line2D([], [], color="black", linestyle=ls, lw=LINEWIDTH, label=lab)
               for ls, lab in zip(STREAM_STYLES, STREAM_LABELS)]
    handles += [Line2D([], [], color=col, lw=LINEWIDTH, label=lab)
                for col, lab in zip(PAIR_COLORS, [r"A1$\cdot$A2", r"A3$\cdot$A4"])]
    handles += [Line2D([], [], color=col, lw=LINEWIDTH, label=lab) for col, lab in zip(ITEM_COLORS, item_labels)]
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    fig.legend(handles=handles, loc="lower center", ncol=4 if len(rhos) > 1 else 2, fontsize=FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.95), handlelength=1.6, columnspacing=1.0, borderaxespad=0)
    save_figure(fig, out_paths)


def plot_regularization(results, rho_sBs, readout, anchor, lams, seeds, out_paths, panel_mm=(38, 28),
                        schedule_name="alternating", block_labels=None):
    """Rows: rho_sB; columns: dense score, sparse score, mean output error
    (all four items). Color = regularization strength lam."""
    schedule = SCHEDULES[schedule_name]
    edges = block_edges(schedule)
    cmap = plt.get_cmap("viridis")
    colors = {lam: ("black" if lam == 0 else cmap(0.15 + 0.75 * k / max(1, len(lams) - 2)))
              for k, lam in enumerate(sorted(l for l in lams if l > 0))}
    colors[0.0] = "black"
    setup_style()
    fig, axes = plt.subplots(len(rho_sBs), 3, sharex=True,
                             figsize=(3 * panel_mm[0] / 25.4, len(rho_sBs) * panel_mm[1] / 25.4))
    for r, rho_sB in enumerate(rho_sBs):
        for lam in lams:
            runs = [results[(rho_sB, readout, schedule_name, lam, anchor if lam > 0 else "init", s)]
                    for s in seeds]
            x = runs[0]["x"]
            for c, key in enumerate(["dense", "sparse", "err"]):
                vals = [run[key].mean(1) if key == "err" else run[key] for run in runs]
                mean, sem = mean_sem(vals)
                axes[r, c].plot(x, mean, color=colors[lam], lw=LINEWIDTH * 0.8)
                axes[r, c].fill_between(x, mean - sem, mean + sem, color=colors[lam], alpha=0.15, linewidth=0)
        for c, ylabel in enumerate(["Pairmate score", None, "Output error"]):
            ax = axes[r, c]
            if c < 2:
                reference_line(ax, 0.0)
            last_row = r == len(rho_sBs) - 1
            style_axis(ax, [0, len(schedule)], shade_edges=edges,
                       xlabel=("Training block" if block_labels is not None else "Block") if last_row else None)
            if block_labels is not None:
                ax.set_xticks(np.arange(len(schedule)) + 0.5)
                ax.set_xticklabels(block_labels, fontsize=FONTSIZE - 2, rotation=90)
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_xticks(list(range(0, len(schedule) + 1, 8)))
            title = (["Dense", "Sparse", "Mean error"][c] if r == 0 else None)
            label_axis(ax, ylabel=(rf"$\rho_{{sB}}$ = {rho_sB:g}" + "\n" + ylabel) if c == 0 else
                       (ylabel if c == 2 else None), title=title)
        share_ylim(axes[r, :2])
    handles = [Line2D([], [], color=colors[lam], lw=LINEWIDTH, label=rf"$\lambda$ = {lam:g}") for lam in lams]
    fig.subplots_adjust(hspace=0.2, wspace=0.5)
    top_legend(fig, handles, y=0.95)
    save_figure(fig, out_paths)


def summarize(results, rho_sBs, readouts, lams, anchors, seeds):
    """End-of-revisit (block 6) and end-of-schedule values, mean over seeds."""
    rows = []
    n_rev = len(SCHEDULES["revisit"])
    for rho_sB, readout, anchor, lam in itertools.product(rho_sBs, readouts, anchors, lams):
        if lam == 0 and anchor != anchors[0]:
            continue
        runs = [results[(rho_sB, readout, "alternating", lam, anchor if lam > 0 else "init", s)] for s in seeds]
        x = runs[0]["x"]
        i_rev = np.argmin(np.abs(x - n_rev))
        row = dict(rho_sB=rho_sB, readout=readout, anchor=anchor if lam > 0 else "-", lam=lam)
        for tag, i in [("rev", i_rev), ("end", -1)]:
            row[f"dense_{tag}"] = np.mean([run["dense"][i] for run in runs])
            row[f"sparse_{tag}"] = np.mean([run["sparse"][i] for run in runs])
            row[f"err_mean_{tag}"] = np.mean([run["err"][i].mean() for run in runs])
            row[f"err_max_{tag}"] = np.mean([run["err"][i].max() for run in runs])
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== #
# Main
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-dir", default="figures/sequential-repulsion")
    parser.add_argument("--only", nargs="+", default=["revisit", "alternating", "regularization"],
                        choices=["revisit", "first16", "alternating", "regularization"])
    parser.add_argument("--readout", nargs="+", default=["trained"], choices=["trained", "fixed"])
    parser.add_argument("--task", default="associative", choices=["associative", "a-only", "a-to-b"])
    parser.add_argument("--rho-sB", type=float, nargs="+", default=[1.0, 0.5],
                        help="rho_sB columns of the trajectory figures (associative task)")
    parser.add_argument("--rho-A", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="corr(A1, A2) = corr(A3, A4) columns (a-only / a-to-b task)")
    parser.add_argument("--steps-per-block", type=int, default=STEPS_PER_BLOCK)
    parser.add_argument("--objective", default="summed", choices=["summed", "independent"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1, 1.0, 10.0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ext", nargs="+", default=["pdf"])
    args = parser.parse_args()

    rhos = args.rho_A if args.task in SINGLE_SLOT_TASKS else args.rho_sB
    rho_label = r"\rho_{A}" if args.task in SINGLE_SLOT_TASKS else r"\rho_{sB}"
    rho_sBs = sorted(set(rhos) | ({1.0, 0.5} if "regularization" in args.only else set()), reverse=True)
    item_labels = ITEM_LABELS[args.task]
    revisit_labels = [item_labels[i] for i in SCHEDULES["revisit"]]
    seeds = list(range(args.n_seeds))
    anchors = ["init", "block"]
    cfgs = set()
    for ro in args.readout:
        for sched in ["revisit", "first16", "alternating"]:
            if sched in args.only:
                cfgs |= {(r, ro, sched, 0.0, "init", s) for r in rhos for s in seeds}
        if "regularization" in args.only:
            cfgs |= {(r, ro, sched, lam, a if lam > 0 else "init", s)
                     for r, sched, lam, a, s in itertools.product(rho_sBs, ["alternating", "revisit"],
                                                                  args.lams, anchors, seeds)}
    # longest runs first for better load balancing
    cfgs = [c + (args.task, args.steps_per_block, args.objective) for c in sorted(cfgs, key=lambda c: (-c[3], c[2] != "alternating"))]
    results = run_all(cfgs, args.workers)

    for ro in args.readout:
        prefix = "" if args.task == "associative" else f"{args.task}-"
        suffix = ("" if ro == "trained" else "-fixed-readout") + \
            ("" if args.steps_per_block == STEPS_PER_BLOCK else f"-{args.steps_per_block}steps") + \
            ("" if args.objective == "summed" else "-independent")
        out = lambda stem: [os.path.join(args.figures_dir, f"{prefix}{stem}{suffix}.{e}") for e in args.ext]
        kw = dict(objective=args.objective)
        if "revisit" in args.only:
            plot_trajectories(results, rhos, ro, "revisit", seeds, out("revisit-trajectories"),
                              item_labels, rho_label, block_labels=revisit_labels, **kw)
        if "first16" in args.only:
            plot_trajectories(results, rhos, ro, "first16", seeds, out("first16-trajectories"),
                              item_labels, rho_label, panel_mm=(50, 30), **kw)
        if "alternating" in args.only:
            plot_trajectories(results, rhos, ro, "alternating", seeds,
                              out("alternating-trajectories"), item_labels, rho_label, panel_mm=(60, 30), **kw)
        if "regularization" in args.only:
            for a in anchors:
                plot_regularization(results, rho_sBs, ro, a, args.lams, seeds, out(f"reg-{a}-trajectories"))
                plot_regularization(results, rho_sBs, ro, a, args.lams, seeds, out(f"reg-{a}-revisit"),
                                    schedule_name="revisit", block_labels=revisit_labels)
    if "regularization" in args.only:
        df = summarize(results, rho_sBs, args.readout, args.lams, anchors, seeds)
        os.makedirs(args.figures_dir, exist_ok=True)
        df.to_csv(os.path.join(args.figures_dir, "reg-summary.csv"), index=False)
        pd.set_option("display.width", 250, "display.max_rows", 200)
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

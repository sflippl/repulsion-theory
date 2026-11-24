"""Talk slides for blocked training of the associative task: per-item loss and
cos(h(A1), h(A2)) in both streams, as static figures and as one GIF per block
(press next slide -> the next GIF). In each GIF the plot first compresses to
make room for the next block, then the block's curves are drawn in.

Same model and training as plot_sequential_repulsion.py (complete_model_linear,
summed output, SGD lr 0.1, momentum 0), but each block trains its item until
that item's loss is below --threshold (default 1e-2) instead of for a fixed
number of steps. Settings: rho_sB in {1, 0.5}, with only A1B1, A2B(1|2)
(4 blocks = two presentations each), or with all four items (8 blocks). With
rho_sB = 1, B1 = B2 and B3 = B4, so the items are A1B1, A2B1, A3B3, A4B3.

Curves are averages over --n-seeds seeds (+-SEM band). Block lengths (in
steps) differ between blocks and seeds, so each block is resampled onto
[0, 1] and all blocks are drawn with the same width, as in the
first16-trajectories figures.

Writes figures/sequential-repulsion/slides[-thresh<t>]/rhoB{1,0.5}-{2,4}items/
("-thresh<t>" unless --threshold is the default):
    loss.{pdf,png}, similarity.{pdf,png}, loss-similarity.{pdf,png}
    block1.gif, ..., blockK.gif (play once; the last frame of one GIF is the
    first frame of the next)

Usage:
    python scripts/plot_sequential_slides.py
    python scripts/plot_sequential_slides.py --threshold 1e-3 --n-seeds 5
    python scripts/plot_sequential_slides.py --rho-sB 0.5 --stream-labels MSP TSP
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_sequential_repulsion as S
from generate_associative_repulsion_figures import make_group_vectors

OUT_DIR = os.path.join(S.ROOT, "figures", "sequential-repulsion", "slides")
THRESHOLD = 1e-2
MAX_STEPS = 20_000       # per block, safety cap
N_GRID = 40              # resampled points (= drawing frames) per block
N_COMPRESS = 18          # frames for the axis to make room for the next block
FRAME_MS = 40
STREAM_LABELS = list(S.STREAM_LABELS)  # legend names of the dense and sparse stream (--stream-labels)
FIGSIZE = (11, 4.2)
DPI = 130
TALK_STYLE = {
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 14, "axes.linewidth": 1.3,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "mathtext.default": "regular",
}


def item_labels(rho_sB, n_items):
    if rho_sB == 1.0:
        return ["A1B1", "A2B1", "A3B3", "A4B3"][:n_items]
    return ["A1B1", "A2B2", "A3B3", "A4B4"][:n_items]


def build_data(rho_sB, n_items, rng):
    """As plot_sequential_repulsion.build_data, with n_items/2 pairs."""
    rot = lambda: np.linalg.qr(rng.standard_normal((S.DA, S.DA)))[0]
    vA = make_group_vectors(1.0, 0.0, 0.0, n_items // 2, 2, S.DA) @ rot()
    vB = make_group_vectors(1.0, 0.0, rho_sB, n_items // 2, 2, S.DA) @ rot()
    ZA = np.hstack([vA, np.zeros_like(vA)])
    ZB = np.hstack([np.zeros_like(vB), vB])
    X = np.sqrt(S.PHI) * ZA + np.sqrt(1 - S.PHI) * ZB
    return torch.tensor(X, dtype=torch.float32), torch.tensor(ZA, dtype=torch.float32)


def simulate(rho_sB, n_items, schedule, seed, threshold):
    """Per-step records: loss (steps+1, n_items), sim (steps+1, 2 streams, n_pairs),
    and the number of steps of each block."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    X, ZA = build_data(rho_sB, n_items, np.random.default_rng(seed))
    model = S.build_network(2)
    opt = torch.optim.SGD(model.parameters(), lr=S.LR, momentum=S.MOMENTUM)
    rec = {"loss": [], "sim": []}

    @torch.no_grad()
    def record():
        rec["loss"].append(((model(X) - X) ** 2).sum(1).numpy())
        rec["sim"].append([[S.rep_cosines(H)[2 * k, 2 * k + 1].item() for k in range(n_items // 2)]
                           for H in S.hidden(model, ZA)])

    record()
    lengths = []
    for item in schedule:
        n = 0
        while n < MAX_STEPS:
            loss = ((model(X[[item]]) - X[[item]]) ** 2).sum()
            if loss.item() < threshold:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            record()
            n += 1
        lengths.append(n)
    return {"loss": np.array(rec["loss"]), "sim": np.array(rec["sim"]), "lengths": lengths}


def average_runs(runs):
    """Resample each block onto N_GRID points, average over seeds, and place the
    blocks back to back with unit width. Returns t (n_blocks, N_GRID), and
    mean/sem of loss and sim with a leading (n_blocks, N_GRID) shape."""
    n_blocks = len(runs[0]["lengths"])
    edges = np.arange(n_blocks + 1, dtype=float)
    u = np.linspace(0, 1, N_GRID)
    t = edges[:-1, None] + u[None]
    out = {"t": t, "edges": edges}
    for key in ["loss", "sim"]:
        per_seed = []
        for r in runs:
            starts = np.concatenate([[0], np.cumsum(r["lengths"])])
            vals = r[key].reshape(len(r[key]), -1)
            blocks = []
            for b in range(n_blocks):
                seg = vals[starts[b]:starts[b + 1] + 1]
                us = np.linspace(0, 1, len(seg)) if len(seg) > 1 else np.array([0.0])
                seg = seg if len(seg) > 1 else np.repeat(seg, 2, axis=0)
                us = us if len(us) > 1 else np.array([0.0, 1.0])
                blocks.append(np.stack([np.interp(u, us, seg[:, c]) for c in range(seg.shape[1])], -1))
            per_seed.append(np.stack(blocks))
        per_seed = np.stack(per_seed)  # (seeds, blocks, N_GRID, channels)
        shape = runs[0][key].shape[1:]
        out[key] = per_seed.mean(0).reshape(n_blocks, N_GRID, *shape)
        sem = per_seed.std(0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * per_seed[0]
        out[key + "_sem"] = sem.reshape(n_blocks, N_GRID, *shape)
    return out


# ====================================================================== #
# Drawing
# ====================================================================== #

def draw(axes, res, schedule, labels, n_done, frac, x_right, ylims, panels, n_tinted=None):
    """Draw blocks 0..n_done-1 fully and block n_done up to fraction frac of its points.
    The first n_tinted blocks get their background tint and label (default: the drawn ones)."""
    t, edges = res["t"], res["edges"]
    n_items = len(labels)
    n_pairs = n_items // 2
    for ax, panel in zip(axes, panels):
        n_shown = n_done + (frac > 0)
        n_tint = min(n_shown if n_tinted is None else n_tinted, len(schedule))
        for b in range(n_tint):
            ax.axvspan(edges[b], edges[b + 1], color=S.ITEM_COLORS[schedule[b]], alpha=S.BLOCK_ALPHA,
                       linewidth=0, zorder=0)

        def segment(arr, b):
            k = N_GRID if b < n_done else max(1, int(round(frac * (N_GRID - 1))) + 1)
            return t[b, :k], arr[b, :k]

        def curve(key, idx, color, ls):
            for b in range(min(n_shown, len(schedule))):
                x, y = segment(res[key][(slice(None), slice(None)) + idx], b)
                _, e = segment(res[key + "_sem"][(slice(None), slice(None)) + idx], b)
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.2, linewidth=0)
                ax.plot(x, y, color=color, linestyle=ls, lw=2.5, solid_capstyle="butt")

        if panel == "loss":
            for i in range(n_items):
                curve("loss", (i,), S.ITEM_COLORS[i], "-")
            ax.set_ylabel("Loss")
        else:
            ax.axhline(0, color="grey", linestyle=":", lw=1.3, zorder=1)
            for k in range(n_pairs):
                for s, ls in enumerate(S.STREAM_STYLES):
                    curve("sim", (s, k), S.PAIR_COLORS[k], ls)
            ax.set_ylabel("A1–A2 similarity" if n_pairs == 1 else "Similarity")
        n_tick = max(1, n_tint)
        centers = [(edges[b] + edges[b + 1]) / 2 for b in range(n_tick)]
        ax.set_xticks(centers)
        ax.set_xticklabels([labels[schedule[b]] for b in range(n_tick)])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(0, x_right)  # after set_xticks, which widens the limits to include all ticks
        ax.set_ylim(*ylims[panel])


def legend_handles(labels, panel):
    if panel == "loss":
        return [Line2D([0], [0], color=S.ITEM_COLORS[i], lw=2.5, label=l) for i, l in enumerate(labels)]
    handles = [Line2D([0], [0], color="k", linestyle=ls, lw=2.5, label=n)
               for n, ls in zip(STREAM_LABELS, S.STREAM_STYLES)]
    pairs = ["A1·A2", "A3·A4"][:len(labels) // 2]
    if len(pairs) > 1:
        handles += [Line2D([0], [0], color=c, lw=2.5, label=p) for p, c in zip(pairs, S.PAIR_COLORS)]
    return handles


def make_figure(res, schedule, labels, n_done, frac, x_right, ylims, panels=("loss", "sim"), n_tinted=None):
    figsize = FIGSIZE if len(panels) == 2 else (FIGSIZE[0] * 0.55, FIGSIZE[1])
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=0.08 if len(panels) == 2 else 0.15, right=0.97, bottom=0.12, top=0.84,
                        wspace=0.28)
    draw(axes[0], res, schedule, labels, n_done, frac, x_right, ylims, panels, n_tinted)
    for ax, panel in zip(axes[0], panels):
        handles = legend_handles(labels, panel)
        ax.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02),
                  handlelength=1.5, columnspacing=0.8, handletextpad=0.4, borderaxespad=0)
    return fig


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_palette(image):
    """Adaptive palette from an image with every color in it, with exact white."""
    quantized = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    colors = np.array(quantized.getpalette()[:768]).reshape(-1, 3)
    colors[colors.sum(1) >= 3 * 250] = 255  # near-white entries: white maps to one of them
    palette = Image.new("P", (1, 1))
    palette.putpalette(colors.flatten().tolist() + [0] * (768 - colors.size))
    return palette


def save_gif(frames, path, palette):
    # one palette for all GIFs of a setting, so one GIF's last frame is pixel-identical
    # to the next GIF's first frame
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    # no loop argument: the GIF plays once and stays on its last frame
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS, disposal=1)
    print(f"Saved {path}")


def ease(x):
    return x * x * (3 - 2 * x)


def block_gifs(res, schedule, labels, ylims, out_dir):
    edges = res["edges"]
    palette = make_palette(to_image(make_figure(res, schedule, labels, len(schedule), 0.0, edges[-1], ylims)))
    for b in range(len(schedule)):
        frames = []
        if b > 0:
            # starts on the previous GIF's last frame; the new block's tint and label
            # slide in as the axis makes room
            for i in range(N_COMPRESS + 1):
                x_right = edges[b] + ease(i / N_COMPRESS) * (edges[b + 1] - edges[b])
                frames.append(to_image(make_figure(res, schedule, labels, b, 0.0, x_right, ylims,
                                                   n_tinted=b + 1)))
        for i in range(1, N_GRID):
            frames.append(to_image(make_figure(res, schedule, labels, b, i / (N_GRID - 1), edges[b + 1], ylims)))
        save_gif(frames, os.path.join(out_dir, f"block{b + 1}.gif"), palette)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--rho-sB", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--n-items", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--stream-labels", nargs=2, default=STREAM_LABELS, metavar=("DENSE", "SPARSE"))
    args = parser.parse_args()
    STREAM_LABELS[:] = args.stream_labels

    settings = [(r, n) for n in args.n_items for r in args.rho_sB]
    results = {}
    for rho_sB, n_items in settings:
        schedule = list(range(n_items)) * 2
        runs = [simulate(rho_sB, n_items, schedule, s, args.threshold) for s in range(args.n_seeds)]
        results[rho_sB, n_items] = (schedule, average_runs(runs))
        lengths = np.array([r["lengths"] for r in runs])
        print(f"rho_sB={rho_sB} n_items={n_items}: steps per block (mean over seeds) {lengths.mean(0).round(1)}")

    # one set of y-limits for all settings, so the slides are comparable
    lo = min(min((res["sim"] - res["sim_sem"]).min() for _, res in results.values()), 0.0)
    hi = max(max((res["sim"] + res["sim_sem"]).max() for _, res in results.values()), 0.0)
    loss_hi = max((res["loss"] + res["loss_sem"]).max() for _, res in results.values())
    pad = 0.05 * (hi - lo)
    ylims = {"loss": (0.0, 1.05 * loss_hi), "sim": (lo - pad, hi + pad)}

    with plt.rc_context(TALK_STYLE):
        for (rho_sB, n_items), (schedule, res) in results.items():
            labels = item_labels(rho_sB, n_items)
            base = OUT_DIR if args.threshold == THRESHOLD else f"{OUT_DIR}-thresh{args.threshold:g}"
            out_dir = os.path.join(base, f"rhoB{rho_sB:g}-{n_items}items")
            os.makedirs(out_dir, exist_ok=True)
            end = res["edges"][-1]
            for name, panels in [("loss", ("loss",)), ("similarity", ("sim",)), ("loss-similarity", ("loss", "sim"))]:
                for ext in ["pdf", "png"]:
                    fig = make_figure(res, schedule, labels, len(schedule), 0.0, end, ylims, panels)
                    path = os.path.join(out_dir, f"{name}.{ext}")
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    print(f"Saved {path}")
            if not args.no_gifs:
                block_gifs(res, schedule, labels, ylims, out_dir)


if __name__ == "__main__":
    main()

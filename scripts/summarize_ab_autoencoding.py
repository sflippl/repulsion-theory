"""Pairmate scores of the ab_* experiments (scripts/ab_autoencoding.sh).

For the hidden representations of A-only input (and, for the autoencoder,
B-only input; "-" for the predictive network), the pairmate score
is the mean similarity of pairmates (items 1 and 2 of each group) minus the mean
similarity of non-pairmates (items of different groups). Prints mean +- SEM
over seeds at the start and end of training, the final loss and, for the
*_behavioral setups, the behavioral repulsion score at the end of training
(signed angle in degrees, see plot_ab_autoencoding.behavioral_scores;
positive = repulsion).

Usage:
    python scripts/summarize_ab_autoencoding.py
    python scripts/summarize_ab_autoencoding.py data/ab_prediction/*
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from plot_ab_autoencoding import behavioral_scores

DEFAULT_DIRS = ["data/ab_prediction/*", "data/ab_autoencoder/*"]
BEHAVIORAL_PROBES = ["output_prediction", "output_a_to_ab"]  # B read out from A alone


def pairmate_score(H: np.ndarray, rows: list[dict]) -> float:
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    C = Hn @ Hn.T
    groups = np.array([r["group_id"] for r in rows])
    same = groups[:, None] == groups[None, :]
    off_diag = ~np.eye(len(rows), dtype=bool)
    return C[same & off_diag].mean() - C[~same].mean()


def seed_dirs(sweep_dir: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(sweep_dir, "*"))
                  if os.path.exists(os.path.join(d, "representations_a_rows.json")))


def summarize(sweep_dir: str) -> dict:
    scores = {f"{p}_{when}": [] for p in "AB" for when in ["start", "end"]}
    scores["behav_end"] = []
    final_loss = []
    for d in seed_dirs(sweep_dir):
        for probe in "ab":
            if not os.path.exists(os.path.join(d, f"representations_{probe}_rows.json")):
                continue
            H = np.load(os.path.join(d, f"representations_{probe}_net0_hidden_0.npy"))
            rows = json.load(open(os.path.join(d, f"representations_{probe}_rows.json")))
            scores[f"{probe.upper()}_start"].append(pairmate_score(H[0], rows))
            scores[f"{probe.upper()}_end"].append(pairmate_score(H[-1], rows))
        if "behavioral" in sweep_dir:
            probe = next(p for p in BEHAVIORAL_PROBES if os.path.exists(os.path.join(d, f"{p}_combined_B.npy")))
            scores["behav_end"].append(behavioral_scores(d, probe)[1][-1])
        final_loss.append(pd.read_csv(os.path.join(d, "training_loss.csv"))["total_loss"].iloc[-1])
    task, setup = sweep_dir.rstrip("/").split("/")[-2:]
    out = dict(task=task, setup=setup, n_seeds=len(final_loss))
    for key, vals in scores.items():
        if not vals:
            out[key] = "-"
            continue
        vals = np.array(vals)
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[key] = f"{vals.mean():+.2f}° ± {sem:.2f}" if key == "behav_end" else f"{vals.mean():+.3f} ± {sem:.3f}"
    out["final_loss_max"] = f"{max(final_loss):.1e}"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dirs", nargs="*", help="data/ab_<task>/<setup> directories")
    args = parser.parse_args()
    dirs = args.sweep_dirs or sorted(d for pattern in DEFAULT_DIRS for d in glob.glob(pattern))
    rows = [summarize(d) for d in dirs if seed_dirs(d)]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

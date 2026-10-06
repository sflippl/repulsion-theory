"""Aggregates repulsion.theory ground-truth check outputs
(run_theory_groundtruth.py, scripts/theory_groundtruth.sh) into a single tidy
CSV for notebooks/theory_analysis.ipynb.

Usage:
    python scripts/collect_theory_groundtruth.py data/theory_groundtruth \
        --out data/theory/groundtruth_checks.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def _flatten(result: dict) -> dict:
    row = {k: v for k, v in result.items() if k != "theory" and not k.startswith("sim_")}
    theory = result.get("theory")
    if theory is None:
        row.update({"theory_a1": None, "theory_a2": None, "theory_a3": None,
                    "theory_p_12": None, "theory_p_3": None})
    else:
        row["theory_a1"] = theory["a1"]
        row["theory_a2"] = theory["a2"]
        row["theory_a3"] = theory["a3"]
        row["theory_p_12"] = theory["p_12"]
        row["theory_p_3"] = theory["p_3"]
    for init in ("exact", "random"):
        sim = result.get(f"sim_{init}")
        if sim is None:
            continue
        for key, value in sim.items():
            row[f"sim_{init}_{key}"] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory tree containing result.json files "
                                      "(e.g. data/theory_groundtruth)")
    parser.add_argument("--out", default="data/theory/groundtruth_checks.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "result.json"), recursive=True))
    if not paths:
        raise SystemExit(f"no result.json files found under {args.root}")

    rows = []
    for path in paths:
        with open(path) as f:
            result = json.load(f)
        row = _flatten(result)
        row["source_dir"] = os.path.dirname(path)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

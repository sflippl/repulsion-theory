"""Generates the Section-8 special-case datasets (repulsion.theory pipeline)
locally via the closed-form formulas -- cheap, no cluster needed. Writes one
tidy CSV per (case, architecture) to data/theory/.

Usage:
    python scripts/generate_theory_datasets.py
    python scripts/generate_theory_datasets.py --cases case1 case2 --out-dir data/theory
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repulsion.theory.pipeline import ARCHITECTURES, CASE_SPECS, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASE_SPECS.keys()),
                         choices=list(CASE_SPECS.keys()))
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES.keys()),
                         choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--out-dir", default="data/theory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for case_name in args.cases:
        for architecture in args.architectures:
            df = run_case(case_name, architecture)
            if len(df) == 0:
                continue
            out_path = os.path.join(args.out_dir, f"{case_name}_{architecture}.csv")
            df.to_csv(out_path, index=False)
            print(f"{case_name} / {architecture}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()

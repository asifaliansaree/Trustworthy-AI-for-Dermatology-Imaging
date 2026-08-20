#!/usr/bin/env python3
"""Generate balanced class weights from the k-fold split CSV.

This script reads ham10000/data/HAM10000_kfold_split.csv, derives integer
labels in the same order used by ham10000/dataset.py, computes balanced
class weights with scikit-learn, and saves them to
ham10000/data/class_weights.npy.

Usage:
    python ham10000/make_class_weights_kfold.py
    python ham10000/make_class_weights_kfold.py --csv ham10000/data/HAM10000_kfold_split.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASS_MAP  # single source of truth, no more hand-copy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="ham10000/data/HAM10000_kfold_split.csv",
        help="Path to the k-fold split CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="ham10000/data/class_weights.npy",
        help="Where to save the generated class weights (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "dx" not in df.columns:
        raise ValueError(f"Expected a 'dx' column in {csv_path}")
    if "fold" not in df.columns:
        raise ValueError(f"Expected a 'fold' column in {csv_path}")

    labels = df["dx"].map(CLASS_MAP).to_numpy()
    classes = np.arange(len(CLASS_MAP))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    weights = weights.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, weights)

    print("===== CLASS WEIGHTS (k-fold CSV) =====")
    for name, idx in CLASS_MAP.items():
        print(f"{name:6s}: {weights[idx]:.6f}")
    print()
    print(f"Saved class weights to: {output_path}")


if __name__ == "__main__":
    main()
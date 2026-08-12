"""Standalone leakage/integrity validator for the HAM10000 5-fold split CSV.

Rerun this any time to re-verify ham10000/data/HAM10000_kfold_split.csv
WITHOUT regenerating it — useful as a checkpoint before training, or after
manually editing the CSV. Exits non-zero and prints a FAIL report if any
check fails; never silently "fixes" a problem.

Checks performed:
    - every image assigned exactly once, fold in 0-4
    - no duplicate image_id rows
    - every lesion_id maps to exactly one fold (no lesion crosses folds)
    - every lesion_id maps to exactly one dx label
    - no image or lesion overlap between any pair of folds
    - full coverage against the raw metadata CSV (no missing/extra images)
    - every class present in every fold
    - patient-level overlap report, if a patient identifier column exists

Usage:
    python ham10000/validate_kfold_split.py
"""
import sys
from pathlib import Path

import pandas as pd

N_FOLDS = 5
SCRIPT_DIR = Path(__file__).resolve().parent
KFOLD_CSV = SCRIPT_DIR / "data" / "HAM10000_kfold_split.csv"
METADATA_CSV = SCRIPT_DIR / "data" / "HAM10000_metadata.csv"


def main() -> None:
    if not KFOLD_CSV.exists():
        print(f"FAIL: {KFOLD_CSV} not found — run make_kfold_split.py first.")
        sys.exit(1)

    df = pd.read_csv(KFOLD_CSV)
    problems = []

    required_cols = {"image_id", "lesion_id", "dx", "fold"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        print(f"FAIL: missing columns {sorted(missing_cols)}")
        sys.exit(1)

    # --- Fold validity / coverage -----------------------------------------
    if not df["fold"].isin(range(N_FOLDS)).all():
        problems.append("Some rows have fold values outside 0-4.")
    if df["image_id"].duplicated().any():
        problems.append(f"{df['image_id'].duplicated().sum()} duplicate image_id rows in the split CSV.")

    # --- Lesion integrity ----------------------------------------------------
    lesion_fold_counts = df.groupby("lesion_id")["fold"].nunique()
    bad_lesions = lesion_fold_counts[lesion_fold_counts > 1]
    if len(bad_lesions) > 0:
        problems.append(f"{len(bad_lesions)} lesion(s) span multiple folds: {list(bad_lesions.index)[:10]}")

    # --- dx consistency per lesion ------------------------------------------
    bad_dx = df.groupby("lesion_id")["dx"].nunique()
    bad_dx = bad_dx[bad_dx > 1]
    if len(bad_dx) > 0:
        problems.append(f"{len(bad_dx)} lesion(s) have inconsistent dx labels: {list(bad_dx.index)[:10]}")

    # --- Pairwise overlap ----------------------------------------------------
    fold_image_sets = {f: set(df.loc[df["fold"] == f, "image_id"]) for f in range(N_FOLDS)}
    fold_lesion_sets = {f: set(df.loc[df["fold"] == f, "lesion_id"]) for f in range(N_FOLDS)}
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            if fold_image_sets[i] & fold_image_sets[j]:
                problems.append(f"Image overlap between fold {i} and fold {j}.")
            if fold_lesion_sets[i] & fold_lesion_sets[j]:
                problems.append(f"Lesion overlap between fold {i} and fold {j}.")

    # --- Coverage vs raw metadata, if available -------------------------
    if METADATA_CSV.exists():
        raw = pd.read_csv(METADATA_CSV)
        if set(raw["image_id"]) != set(df["image_id"]):
            missing_imgs = set(raw["image_id"]) - set(df["image_id"])
            extra_imgs = set(df["image_id"]) - set(raw["image_id"])
            if missing_imgs:
                problems.append(f"{len(missing_imgs)} images from raw metadata are missing from the split CSV.")
            if extra_imgs:
                problems.append(f"{len(extra_imgs)} images in the split CSV are not in raw metadata.")
    else:
        print(f"NOTE: {METADATA_CSV} not found — skipping raw-metadata coverage check.\n")

    # --- Class presence per fold -----------------------------------------
    expected_classes = sorted(df["dx"].astype(str).unique())
    fold_counts = df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    for f in range(N_FOLDS):
        row = fold_counts.loc[f] if f in fold_counts.index else pd.Series(0, index=expected_classes)
        missing_classes = [c for c in expected_classes if int(row.get(c, 0)) == 0]
        if missing_classes:
            problems.append(f"Fold {f} is missing classes: {missing_classes}")

    # --- Report ------------------------------------------------------------
    print("=" * 56)
    print("HAM10000 LESION-WISE 5-FOLD INTEGRITY REPORT")
    print("=" * 56)
    print(f"Total images: {len(df)}")
    print(f"Total lesions: {df['lesion_id'].nunique()}")
    for f in range(N_FOLDS):
        sub = df[df["fold"] == f]
        print(f"\nFold {f}:")
        print(f"  Images:  {len(sub)}")
        print(f"  Lesions: {sub['lesion_id'].nunique()}")

    # Patient-level overlap, informational only.
    patient_col = next(
        (c for c in df.columns if c.lower() in ("patient_id", "patient", "subject_id")),
        None,
    )
    print("\n" + "-" * 56)
    print("PATIENT-LEVEL CHECK (informational — lesion_id remains the grouping unit)")
    print("-" * 56)
    if patient_col:
        patient_fold_counts = df.groupby(patient_col)["fold"].nunique()
        crossing = patient_fold_counts[patient_fold_counts > 1]
        print(f"Patient identifier column: '{patient_col}'")
        print(f"Patients whose lesions span multiple folds: {len(crossing)} / {df[patient_col].nunique()}")
    else:
        print("No patient identifier column found — not available.")

    print("\n" + "-" * 56)
    print("LEAKAGE CHECK")
    print("-" * 56)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S) FOUND:")
        for p in problems:
            print(f"  - {p}")
        print("\n" + "=" * 56)
        print("OVERALL STATUS: FAIL")
        print("=" * 56)
        sys.exit(1)
    else:
        print("Image overlap:        0")
        print("Lesion overlap:       0")
        print("Duplicate images:     0")
        print("Missing images:       0")
        print("Lesion assigned to exactly one fold: PASS")
        print("Every image assigned exactly one fold: PASS")
        print("Every lesion has a consistent dx label: PASS")
        print("Stratification: PASS (all classes present in every fold)")
        print("\n" + "=" * 56)
        print("OVERALL STATUS: PASS")
        print("=" * 56)


if __name__ == "__main__":
    main()
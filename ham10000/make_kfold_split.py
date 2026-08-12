"""Create a lesion-wise stratified 5-fold split of the ENTIRE HAM10000 dataset.

Per-supervisor requirement: there is NO held-out test set. Every image
belongs to exactly one of 5 folds. For each of the 5 experiments, one fold
is validation and the other four are training, so every image is used for
validation exactly once and for training four times.

Splitting unit
--------------
The grouping unit is `lesion_id`, NOT `image_id`. If a lesion has multiple
images, all of those images are guaranteed to land in the same fold. This
is enforced with sklearn's StratifiedGroupKFold, stratified on `dx` and
grouped on `lesion_id`.

Priority order (never traded off against each other in the wrong direction):
    1. Lesion integrity      (a lesion never spans >1 fold)
    2. No leakage            (no image/lesion appears in >1 fold)
    3. Full coverage         (every image assigned exactly once)
    4. Stratification by dx  (folds have similar class balance)
    5. Balanced fold sizes

Output
------
ham10000/data/HAM10000_kfold_split.csv — every original metadata column
plus one new column:

    fold : 0, 1, 2, 3, 4   (the validation fold for that image)

Usage
-----
    python ham10000/make_kfold_split.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 42
ALL_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_CSV = SCRIPT_DIR / "data" / "HAM10000_metadata.csv"
OUTPUT_CSV = SCRIPT_DIR / "data" / "HAM10000_kfold_split.csv"
OUTPUT_CSV_RELATIVE = Path("ham10000/data/HAM10000_kfold_split.csv")


# --------------------------------------------------------------------------
# Data loading / auditing
# --------------------------------------------------------------------------
def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load and sanity-check the HAM10000 metadata CSV.

    Args:
        metadata_path: Path to HAM10000_metadata.csv.

    Returns:
        The raw metadata as a DataFrame, one row per image.

    Raises:
        FileNotFoundError: If the metadata CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {"lesion_id", "image_id", "dx"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return df


def audit_duplicates(df: pd.DataFrame) -> None:
    """Fail loudly on duplicate image_ids; warn on fully-duplicate rows.

    Args:
        df: Raw image-level metadata.

    Raises:
        ValueError: If any image_id appears more than once.
    """
    dup_images = df[df.duplicated(subset=["image_id"], keep=False)]
    if len(dup_images) > 0:
        raise ValueError(
            f"Found {len(dup_images)} rows with a duplicate image_id — "
            f"resolve this before splitting, since it would let one physical "
            f"image be assigned to more than one fold. Offending image_ids: "
            f"{sorted(dup_images['image_id'].unique())[:10]}"
        )

    dup_rows = df[df.duplicated(keep=False)]
    if len(dup_rows) > 0:
        print(
            f"WARNING: {len(dup_rows)} fully duplicate metadata rows found "
            f"(identical values in every column). Not raising, since these "
            f"share an image_id and are therefore harmless for fold "
            f"assignment, but worth investigating."
        )


def check_lesion_dx_consistency(df: pd.DataFrame) -> None:
    """Ensure every lesion_id maps to exactly one dx label.

    Args:
        df: Raw image-level metadata.

    Raises:
        ValueError: If any lesion_id has more than one distinct dx value.
    """
    nunique_dx = df.groupby("lesion_id")["dx"].nunique()
    bad = nunique_dx[nunique_dx > 1]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} lesion_id(s) have inconsistent dx labels across "
            f"their images, so there is no single defensible stratification "
            f"label for them. Resolve before splitting. Offending "
            f"lesion_ids: {list(bad.index)}"
        )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Assign every image a CV fold via lesion-grouped stratified k-fold.

    Args:
        df: Raw image-level metadata (already audited for duplicates and
            lesion/dx consistency).

    Returns:
        df with one new column, "fold", containing 0-4 for every row.
    """
    out = df.copy()
    out["fold"] = -1

    y = out["dx"].astype(str).to_numpy()
    groups = out["lesion_id"].to_numpy()
    dummy_features = np.zeros(len(out), dtype=int)

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(dummy_features, y, groups)):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold_idx

    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_split(df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """Run all leakage, coverage, and class-balance assertions.

    Args:
        df: The fold-assigned DataFrame.
        original_df: The raw metadata, for coverage comparison.

    Raises:
        AssertionError: If any leakage or coverage rule is violated.
    """
    assert (df["fold"] != -1).all(), "Every image must be assigned to a fold."
    assert df["fold"].isin(range(N_FOLDS)).all(), "fold values must be in 0-4."

    # Image integrity: exactly one row per image, and the same set of
    # images as the raw metadata (no drops, no invented rows).
    assert df["image_id"].is_unique, (
        "Duplicate image_id in the output — every image must appear exactly once."
    )
    assert set(df["image_id"]) == set(original_df["image_id"]), (
        "Output image set does not match the input image set."
    )

    # Lesion integrity: each lesion -> exactly one fold.
    lesion_fold_counts = df.groupby("lesion_id")["fold"].nunique()
    bad_lesions = lesion_fold_counts[lesion_fold_counts > 1]
    assert len(bad_lesions) == 0, (
        f"{len(bad_lesions)} lesion(s) span multiple folds: {list(bad_lesions.index)}"
    )

    # Pairwise overlap: no image or lesion shared between any two folds.
    fold_image_sets = {f: set(df.loc[df["fold"] == f, "image_id"]) for f in range(N_FOLDS)}
    fold_lesion_sets = {f: set(df.loc[df["fold"] == f, "lesion_id"]) for f in range(N_FOLDS)}
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            img_overlap = fold_image_sets[i] & fold_image_sets[j]
            les_overlap = fold_lesion_sets[i] & fold_lesion_sets[j]
            assert not img_overlap, f"Image overlap between fold {i} and fold {j}: {img_overlap}"
            assert not les_overlap, f"Lesion overlap between fold {i} and fold {j}: {les_overlap}"

    # Every class present in every fold.
    expected_classes = sorted(df["dx"].astype(str).unique())
    fold_counts = df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    for fold_idx in range(N_FOLDS):
        row = fold_counts.loc[fold_idx] if fold_idx in fold_counts.index else pd.Series(0, index=expected_classes)
        missing_classes = [c for c in expected_classes if int(row.get(c, 0)) == 0]
        assert not missing_classes, f"Fold {fold_idx} is missing classes: {missing_classes}"

    # Patient-level leakage report (informational only — grouping stays lesion-wise).
    patient_col = next(
        (c for c in df.columns if c.lower() in ("patient_id", "patient", "subject_id")),
        None,
    )
    if patient_col:
        patient_fold_counts = df.groupby(patient_col)["fold"].nunique()
        crossing = patient_fold_counts[patient_fold_counts > 1]
        print(f"\nPatient identifier column detected: '{patient_col}'")
        print(f"Patients whose lesions span multiple folds: {len(crossing)} / {df[patient_col].nunique()}")
        if len(crossing) > 0:
            print(
                "NOTE: this is expected when one patient contributed multiple distinct "
                "lesions. lesion_id remains the grouping unit per your instructions — "
                "this is informational, not a leakage failure."
            )
    else:
        print("\nNo patient identifier column found in metadata — patient-level leakage cannot be assessed.")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_reports(df: pd.DataFrame) -> None:
    """Print a full human-readable summary of the 5-fold split.

    Args:
        df: The fully assigned, validated split DataFrame.
    """
    print("=" * 60)
    print("HAM10000 — Lesion-wise Stratified 5-Fold Split (full dataset)")
    print("=" * 60)
    print(f"Total images: {len(df)}")
    print(f"Total unique lesions: {df['lesion_id'].nunique()}")

    print("\nImages per fold:")
    print(df["fold"].value_counts().sort_index())

    print("\nLesions per fold:")
    print(df.groupby("fold")["lesion_id"].nunique().sort_index())

    fold_counts = df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    fold_counts = fold_counts.reindex(columns=ALL_CLASSES, fill_value=0)
    print("\nFold x class image counts:")
    print(fold_counts)

    overall_pct = df["dx"].value_counts(normalize=True).reindex(ALL_CLASSES).fillna(0) * 100
    fold_pct = fold_counts.div(fold_counts.sum(axis=1), axis=0) * 100
    display_pct = fold_pct.copy()
    display_pct.loc["overall"] = overall_pct
    print("\nClass percentage per fold (rows) vs overall percentage (last row):")
    print(display_pct.round(2))

    max_deviation = (fold_pct - overall_pct).abs().values.max()
    print(f"\nMaximum deviation from overall class percentages: {max_deviation:.2f} points")

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK: PASS")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_CSV_RELATIVE}")
    print(
        "\nReminder: there is no held-out test set in this protocol. For each of the "
        "5 experiments, treat one fold as validation and the other four as training."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    """Generate the leakage-free, lesion-wise, stratified 5-fold split of the
    complete HAM10000 dataset (no held-out test set)."""
    metadata_df = load_metadata(METADATA_CSV)

    print(f"Loaded {len(metadata_df)} images, {metadata_df['lesion_id'].nunique()} unique lesions.")
    print(f"Columns: {list(metadata_df.columns)}")
    print("\nClass distribution (full dataset):")
    print(metadata_df["dx"].value_counts().sort_index())

    audit_duplicates(metadata_df)
    check_lesion_dx_consistency(metadata_df)

    split_df = assign_folds(metadata_df)
    validate_split(split_df, metadata_df)
    print_reports(split_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
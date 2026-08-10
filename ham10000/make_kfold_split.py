"""Create a lesion-wise stratified split for HAM10000 with a separate test set.

This script implements "Approach 2" — the recommended experimental protocol
for reproducible medical-AI evaluation on HAM10000:

    Entire HAM10000 Dataset
            |
            v
    Lesion-wise Stratified Split (train_test_split, stratify=dx, test_size=0.10)
            |
     +------+-------+
     |              |
    Train+Val      Test
    90%            10%
     |
     v
    StratifiedGroupKFold(n_splits=5, groups=lesion_id, labels=dx)
     |
    Fold0 Fold1 Fold2 Fold3 Fold4

The test set is created first and is *never* touched again — cross-validation
only ever operates on the remaining trainval lesions. Splitting is always
done at the lesion level (not the image level), so every image belonging to
a given lesion_id always receives the same split/fold. This guarantees:

    * no lesion leakage between trainval and test
    * no lesion leakage across CV folds
    * class-balanced stratification throughout

Output
------
The output CSV (ham10000/data/HAM10000_kfold_split.csv) contains every
original metadata column plus two new columns:

    split : "trainval" or "test"
    fold  : 0, 1, 2, 3, 4 for trainval rows; -1 for test rows

Usage
-----
    python ham10000/make_kfold_split.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.10
ALL_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_CSV = SCRIPT_DIR / "data" / "HAM10000_metadata.csv"
OUTPUT_CSV = SCRIPT_DIR / "data" / "HAM10000_kfold_split.csv"
OUTPUT_CSV_RELATIVE = Path("ham10000/data/HAM10000_kfold_split.csv")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load the HAM10000 metadata CSV and validate the required columns.

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

    metadata_df = pd.read_csv(metadata_path)

    required_columns = {"lesion_id", "dx"}
    missing_columns = required_columns.difference(metadata_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return metadata_df


def build_lesion_level_frame(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the image-level metadata to one row per unique lesion_id.

    Each lesion may have multiple images, but a lesion always has a single
    diagnosis (dx), so the first occurrence is representative for splitting.

    Args:
        metadata_df: Image-level metadata.

    Returns:
        A DataFrame with one row per lesion_id and its dx label.
    """
    lesion_df = (
        metadata_df.groupby("lesion_id", as_index=False)
        .first()[["lesion_id", "dx"]]
    )
    lesion_df["dx"] = lesion_df["dx"].astype(str)
    return lesion_df


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_lesions(lesion_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split lesion IDs into a trainval partition and a held-out test partition.

    Uses sklearn.model_selection.train_test_split, stratified on dx, so that
    the held-out test set has a class distribution close to the full dataset
    while guaranteeing every lesion (and therefore every image belonging to
    it) lands entirely on one side of the split.

    Args:
        lesion_df: One row per lesion_id with its dx label.

    Returns:
        A (trainval_lesions_df, test_lesions_df) tuple.
    """
    trainval_lesions_df, test_lesions_df = train_test_split(
        lesion_df,
        test_size=TEST_SIZE,
        stratify=lesion_df["dx"],
        random_state=RANDOM_STATE,
    )
    return (
        trainval_lesions_df.reset_index(drop=True),
        test_lesions_df.reset_index(drop=True),
    )


def assign_splits_and_folds(
    metadata_df: pd.DataFrame,
    trainval_lesions_df: pd.DataFrame,
    test_lesions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign every image a split ("trainval"/"test") and a CV fold.

    Cross-validation folds are computed with StratifiedGroupKFold, grouped by
    lesion_id and stratified by dx, so no lesion ever appears in more than
    one fold, and each fold's class balance mirrors the full trainval set as
    closely as possible.

    Args:
        metadata_df: Full image-level metadata.
        trainval_lesions_df: Lesions assigned to the trainval partition.
        test_lesions_df: Lesions assigned to the test partition.

    Returns:
        The metadata DataFrame with two new columns: "split" and "fold".
    """
    test_lesion_ids = set(test_lesions_df["lesion_id"])

    split_df = metadata_df.copy()
    split_df["split"] = split_df["lesion_id"].map(
        lambda lesion_id: "test" if lesion_id in test_lesion_ids else "trainval"
    )
    split_df["fold"] = -1

    trainval_mask = split_df["split"] == "trainval"
    trainval_df = split_df.loc[trainval_mask].copy()

    if len(trainval_df) == 0:
        raise ValueError("The trainval subset is empty after the lesion split.")

    fold_labels = trainval_df["dx"].astype(str).to_numpy()
    fold_groups = trainval_df["lesion_id"].to_numpy()
    dummy_features = np.zeros(len(trainval_df), dtype=int)

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(dummy_features, fold_labels, fold_groups)
    ):
        fold_image_indices = trainval_df.index.to_numpy()[val_idx]
        split_df.loc[fold_image_indices, "fold"] = fold_idx

    return split_df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_split(split_df: pd.DataFrame) -> None:
    """Run all leakage, coverage, and class-balance assertions.

    Args:
        split_df: The fully assigned split/fold DataFrame.

    Raises:
        AssertionError: If any leakage or coverage rule is violated.
    """
    # Every image assigned.
    assert split_df["split"].notna().all(), "Every image must have a split assigned."
    assert split_df["fold"].notna().all(), "Every image must have a fold assigned."
    assert set(split_df["split"].unique()).issubset({"trainval", "test"}), (
        "Unexpected split values."
    )

    # No lesion appears in both trainval and test.
    lesion_split_counts = split_df.groupby("lesion_id")["split"].nunique()
    assert (lesion_split_counts == 1).all(), (
        "A lesion_id appears in both trainval and test."
    )

    trainval_df = split_df.loc[split_df["split"] == "trainval"]
    test_df = split_df.loc[split_df["split"] == "test"]

    # Fold value sanity.
    assert trainval_df["fold"].isin(range(N_FOLDS)).all(), (
        "Trainval rows must have folds 0-4."
    )
    assert (test_df["fold"] == -1).all(), "Test rows must have fold = -1."

    # No lesion spans multiple folds.
    lesion_fold_counts = trainval_df.groupby("lesion_id")["fold"].nunique()
    assert (lesion_fold_counts == 1).all(), "A lesion_id appears in multiple folds."

    # Every class present in the test set.
    expected_classes = sorted(split_df["dx"].astype(str).unique())
    test_classes = sorted(test_df["dx"].astype(str).unique())
    assert test_classes == expected_classes, (
        f"Test set is missing classes: expected {expected_classes}, got {test_classes}"
    )

    # Every class present in every fold.
    fold_counts = trainval_df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    for fold_idx in range(N_FOLDS):
        if fold_idx in fold_counts.index:
            fold_class_counts = fold_counts.loc[fold_idx]
        else:
            fold_class_counts = pd.Series(0, index=expected_classes)
        missing_fold_classes = [
            cls for cls in expected_classes if int(fold_class_counts.get(cls, 0)) == 0
        ]
        assert not missing_fold_classes, (
            f"Fold {fold_idx} is missing classes: {missing_fold_classes}"
        )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_reports(split_df: pd.DataFrame) -> None:
    """Print a full human-readable summary of the split and CV setup.

    Args:
        split_df: The fully assigned, validated split/fold DataFrame.
    """
    trainval_df = split_df.loc[split_df["split"] == "trainval"]
    test_df = split_df.loc[split_df["split"] == "test"]

    # ---- Separate test set summary -------------------------------------
    print("=" * 34)
    print("Separate Test Set")
    print("=" * 34)
    print(f"Total images: {len(split_df)}")
    print(f"Total unique lesion_id: {split_df['lesion_id'].nunique()}")
    print(f"Trainval lesions: {trainval_df['lesion_id'].nunique()}")
    print(f"Test lesions: {test_df['lesion_id'].nunique()}")
    print(f"Trainval images: {len(trainval_df)}")
    print(f"Test images: {len(test_df)}")

    print("\nTrainval class counts:")
    print(trainval_df["dx"].value_counts().sort_index())

    print("\nTest class counts:")
    print(test_df["dx"].value_counts().sort_index())

    test_classes = sorted(test_df["dx"].astype(str).unique())
    all_present = set(ALL_CLASSES).issubset(set(test_classes))
    print(f"\nAll seven HAM10000 classes present in test set: {all_present}")

    # ---- Cross-validation summary ---------------------------------------
    print("\n" + "=" * 34)
    print("Cross Validation")
    print("=" * 34)
    print(f"{N_FOLDS} folds")
    print("No leakage")
    print("Balanced stratification")

    print("\nImages per fold:")
    print(trainval_df["fold"].value_counts().sort_index())

    fold_counts = trainval_df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    print("\nFold x class counts:")
    print(fold_counts)

    overall_pct = trainval_df["dx"].value_counts(normalize=True).sort_index() * 100
    fold_pct = fold_counts.div(fold_counts.sum(axis=1), axis=0) * 100
    display_pct = fold_pct.copy()
    display_pct.loc["overall"] = overall_pct
    print("\nClass percentage per fold (rows) vs overall trainval percentage (last row):")
    print(display_pct.round(2))

    max_deviation = (fold_pct - overall_pct).abs().values.max()
    print(f"\nMaximum deviation from overall class percentages: {max_deviation:.2f} points")

    # ---- Save confirmation ------------------------------------------------
    print("\n" + "=" * 34)
    print("Saved")
    print("=" * 34)
    print(OUTPUT_CSV_RELATIVE)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    """Generate the leakage-free HAM10000 split with an independent test set
    and stratified group k-fold cross-validation on the remaining data."""
    metadata_df = load_metadata(METADATA_CSV)
    lesion_df = build_lesion_level_frame(metadata_df)

    trainval_lesions_df, test_lesions_df = split_lesions(lesion_df)
    split_df = assign_splits_and_folds(metadata_df, trainval_lesions_df, test_lesions_df)

    validate_split(split_df)
    print_reports(split_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
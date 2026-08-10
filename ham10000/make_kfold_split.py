"""Create a lesion-wise stratified split for HAM10000 with a separate test set.

This script implements the recommended experimental protocol for reproducible
medical-AI evaluation:

1. Create a lesion-level, stratified split with train_test_split to reserve a
   separate test set (10% of lesions, stratified by dx).
2. Keep the test set completely independent from cross-validation.
3. Apply StratifiedGroupKFold on the remaining train/validation lesions to
   create five folds.

The output CSV contains every original metadata column plus two new columns:
- split: "trainval" or "test"
- fold: 0, 1, 2, 3, 4 for trainval samples and -1 for test samples

This script is fully self-contained and can be executed with:

    python ham10000/make_kfold_split.py
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_CSV = SCRIPT_DIR / "data" / "HAM10000_metadata.csv"
OUTPUT_CSV = SCRIPT_DIR / "data" / "HAM10000_kfold_split.csv"
OUTPUT_CSV_RELATIVE = Path("ham10000/data/HAM10000_kfold_split.csv")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load the HAM10000 metadata CSV and validate the required columns."""
    df = pd.read_csv(metadata_path)
    required_columns = {"lesion_id", "dx"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return df


def build_lesion_level_frame(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per lesion_id with the associated diagnosis label."""
    lesion_df = metadata_df.groupby("lesion_id", as_index=False).first()[["lesion_id", "dx"]]
    lesion_df["dx"] = lesion_df["dx"].astype(str)
    return lesion_df


def split_lesions(lesion_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split lesion IDs into separate test and trainval partitions."""
    train_lesions_df, test_lesions_df = train_test_split(
        lesion_df,
        test_size=TEST_SIZE,
        stratify=lesion_df["dx"],
        random_state=RANDOM_STATE,
    )
    return train_lesions_df.reset_index(drop=True), test_lesions_df.reset_index(drop=True)


def assign_splits_and_folds(
    metadata_df: pd.DataFrame,
    train_lesions_df: pd.DataFrame,
    test_lesions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign every image to either trainval or test and to a CV fold."""
    test_lesions = set(test_lesions_df["lesion_id"])

    split_df = metadata_df.copy()
    split_df["split"] = split_df["lesion_id"].map(
        lambda lesion_id: "test" if lesion_id in test_lesions else "trainval"
    )
    split_df["fold"] = -1

    trainval_mask = split_df["split"] == "trainval"
    trainval_df = split_df.loc[trainval_mask].copy()

    if len(trainval_df) == 0:
        raise ValueError("The trainval subset is empty after the lesion split.")

    labels = trainval_df["dx"].astype(str).to_numpy()
    groups = trainval_df["lesion_id"].to_numpy()
    x_values = np.zeros(len(trainval_df), dtype=int)

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(x_values, labels, groups)):
        fold_image_indices = trainval_df.index.to_numpy()[val_idx]
        split_df.loc[fold_image_indices, "fold"] = fold_idx

    return split_df


def validate_split(split_df: pd.DataFrame) -> None:
    """Validate leakage rules, class coverage, and fold coverage."""
    assert split_df["split"].notna().all(), "Every image must have a split assigned."
    assert split_df["fold"].notna().all(), "Every image must have a fold assigned."
    assert set(split_df["split"].unique()).issubset({"trainval", "test"}), "Unexpected split values."

    lesion_split_counts = split_df.groupby("lesion_id")["split"].nunique()
    assert (lesion_split_counts == 1).all(), "A lesion_id appears in both trainval and test."

    trainval_df = split_df.loc[split_df["split"] == "trainval"]
    test_df = split_df.loc[split_df["split"] == "test"]

    assert (trainval_df["fold"].isin(range(N_FOLDS))).all(), "Trainval rows must have folds 0-4."
    assert (test_df["fold"] == -1).all(), "Test rows must have fold = -1."

    lesion_fold_counts = trainval_df.groupby("lesion_id")["fold"].nunique()
    assert (lesion_fold_counts == 1).all(), "A lesion_id appears in multiple folds."

    expected_classes = sorted(split_df["dx"].astype(str).unique())
    test_classes = sorted(test_df["dx"].astype(str).unique())
    assert test_classes == expected_classes, f"Test set is missing classes: expected {expected_classes}, got {test_classes}"

    fold_counts = trainval_df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    for fold_idx in range(N_FOLDS):
        fold_class_counts = fold_counts.loc[fold_idx] if fold_idx in fold_counts.index else pd.Series(0, index=expected_classes)
        missing_fold_classes = [cls for cls in expected_classes if int(fold_class_counts.get(cls, 0)) == 0]
        assert not missing_fold_classes, f"Fold {fold_idx} is missing classes: {missing_fold_classes}"


def print_reports(split_df: pd.DataFrame) -> None:
    """Print a concise summary of the split and CV setup."""
    trainval_df = split_df.loc[split_df["split"] == "trainval"]
    test_df = split_df.loc[split_df["split"] == "test"]

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

    max_dev = (fold_pct - overall_pct).abs().values.max()
    print(f"\nMax deviation from overall class percentages: {max_dev:.2f} points")

    print("\n" + "=" * 34)
    print("Saved")
    print(OUTPUT_CSV_RELATIVE)
    print("=" * 34)


def main() -> None:
    """Generate the shared HAM10000 split with independent test and CV folds."""
    metadata_df = load_metadata(METADATA_CSV)
    lesion_df = build_lesion_level_frame(metadata_df)
    train_lesions_df, test_lesions_df = split_lesions(lesion_df)

    split_df = assign_splits_and_folds(metadata_df, train_lesions_df, test_lesions_df)
    validate_split(split_df)
    print_reports(split_df)

    split_df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()

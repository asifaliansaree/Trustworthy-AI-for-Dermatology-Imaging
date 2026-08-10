"""Materialize the CV fold split into actual folders on disk.

This script reuses the leakage-free splits saved in
ham10000/data/HAM10000_kfold_split.csv and creates the 5-fold directory tree
used for cross-validation.

The new protocol stores a separate test partition in the CSV using the
"split" column. Rows with split="trainval" and fold 0-4 are materialized into
the five CV-fold folders, and rows with split="test" are materialized into a
separate test folder so the held-out set is available for final evaluation.

By default this SYMLINKS images instead of copying them, because HAM10000
is ~2.5GB and duplicating it 1x again just to reorganize by folder wastes
disk space for no benefit — symlinks behave identically to real files for
reading/training (PyTorch's ImageFolder, cv2, PIL etc. all follow them
transparently). Pass --copy if you need real independent copies (e.g. to
zip and hand off the folder structure to someone else, or on a filesystem/
platform where symlinks cause problems).

Usage
-----
    python ham10000/make_kfold_folders.py
    python ham10000/make_kfold_folders.py --copy
"""
import argparse
import os
import shutil
import pandas as pd

KFOLD_CSV = "ham10000/data/HAM10000_kfold_split.csv"
IMG_DIRS = [
    "ham10000/data/HAM10000_images_part_1",
    "ham10000/data/HAM10000_images_part_2",
]
OUT_ROOT = "ham10000/data/kfold"
N_FOLDS = 5
CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]


def find_image(image_id: str) -> str:
    for d in IMG_DIRS:
        p = os.path.join(d, image_id + ".jpg")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image '{image_id}' not found in either image directory.")


def place_file(src: str, dst: str, copy: bool):
    if os.path.exists(dst) or os.path.islink(dst):
        return  # already placed (safe to re-run)
    if copy:
        shutil.copy2(src, dst)
    else:
        # relative symlink so the kfold/ folder stays portable if the repo moves
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy real image files instead of symlinking (uses ~2.5GB extra disk).",
    )
    args = parser.parse_args()

    if not os.path.exists(KFOLD_CSV):
        raise FileNotFoundError(
            f"{KFOLD_CSV} not found — run make_kfold_split.py first to "
            f"generate the fold assignments."
        )

    df = pd.read_csv(KFOLD_CSV)
    assert "fold" in df.columns, f"{KFOLD_CSV} has no 'fold' column — regenerate it."
    assert "split" in df.columns, f"{KFOLD_CSV} has no 'split' column — regenerate it."

    cv_df = df.loc[(df["split"] == "trainval") & (df["fold"].isin(range(N_FOLDS)))].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    if len(cv_df) == 0:
        raise ValueError("No trainval rows with valid folds were found in the split CSV.")

    print(f"Loaded {len(df)} images total; materializing {len(cv_df)} trainval images across {cv_df['fold'].nunique()} folds and {len(test_df)} test images.")
    mode = "copying" if args.copy else "symlinking"
    print(f"Mode: {mode} files into {OUT_ROOT}/fold_N/class/ and {OUT_ROOT}/test/class/")

    # Create fold_N/class/ and test/class/ directory trees up front
    for fold in range(N_FOLDS):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_ROOT, f"fold_{fold}", cls), exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_ROOT, "test", cls), exist_ok=True)

    placed = 0
    missing = []
    for row in cv_df.itertuples(index=False):
        fold_dir = os.path.join(OUT_ROOT, f"fold_{row.fold}", row.dx)
        dst = os.path.join(fold_dir, row.image_id + ".jpg")
        try:
            src = find_image(row.image_id)
        except FileNotFoundError:
            missing.append(row.image_id)
            continue
        place_file(src, dst, copy=args.copy)
        placed += 1

    test_placed = 0
    test_missing = []
    for row in test_df.itertuples(index=False):
        test_dir = os.path.join(OUT_ROOT, "test", row.dx)
        dst = os.path.join(test_dir, row.image_id + ".jpg")
        try:
            src = find_image(row.image_id)
        except FileNotFoundError:
            test_missing.append(row.image_id)
            continue
        place_file(src, dst, copy=args.copy)
        test_placed += 1

    print(f"\nPlaced {placed} / {len(cv_df)} trainval images.")
    if missing:
        print(f"WARNING: {len(missing)} images from the trainval CSV were not found "
              f"on disk (first few: {missing[:5]})")
    print(f"Placed {test_placed} / {len(test_df)} test images.")
    if test_missing:
        print(f"WARNING: {len(test_missing)} images from the test CSV were not found "
              f"on disk (first few: {test_missing[:5]})")

    # --- Verification: count files per fold/class dir and compare to CSV ---
    print("\n=== FILE COUNTS ON DISK (fold x class) ===")
    header = f"{'fold':<8}" + "".join(f"{c:>8}" for c in CLASSES) + f"{'total':>8}"
    print(header)
    for fold in range(N_FOLDS):
        counts = []
        for cls in CLASSES:
            d = os.path.join(OUT_ROOT, f"fold_{fold}", cls)
            n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
            counts.append(n)
        row_str = f"fold_{fold:<3}" + "".join(f"{n:>8}" for n in counts) + f"{sum(counts):>8}"
        print(row_str)

    print(f"\nDone. Folder structure is at: {OUT_ROOT}/fold_<0-4>/<class>/ and {OUT_ROOT}/test/<class>/")
    print("Note: this tree is derived output — HAM10000_kfold_split.csv "
          "(trainval/test and fold assignments) remains the source of truth. "
          "If you ever need to rebuild this tree, delete the kfold/ folder and rerun this script.")


if __name__ == "__main__":
    main()
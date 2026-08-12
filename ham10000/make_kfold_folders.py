"""Materialize the lesion-wise 5-fold CV split into folders on disk.

Reads ham10000/data/HAM10000_kfold_split.csv (a "fold" column, 0-4, with
EVERY image assigned to exactly one fold — the whole dataset is used, there
is no separate held-out test set) and builds:

    ham10000/data/kfold/fold_<0-4>/<class>/<image_id>.jpg

For each of the 5 experiments, one fold_<i>/ is validation and the other
four fold folders are training (concatenate them, or just filter the CSV by
`fold` at DataLoader time — either approach works, since this script only
materializes the raw per-fold class folders).

By default this SYMLINKS images instead of copying them (HAM10000 is
~2.5GB and duplicating it wastes disk space for no benefit — symlinks
behave identically to real files for PyTorch's ImageFolder, cv2, PIL,
etc.). Pass --copy for real independent copies.

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
    assert df["fold"].isin(range(N_FOLDS)).all(), (
        "Every row must have a fold in 0-4 — this protocol has no held-out "
        "test rows. If you still have a 'split' column with test rows in "
        "this CSV, regenerate it with the updated make_kfold_split.py."
    )

    print(f"Loaded {len(df)} images across {df['fold'].nunique()} folds (full dataset, no held-out test set).")
    mode = "copying" if args.copy else "symlinking"
    print(f"Mode: {mode} files into {OUT_ROOT}/fold_N/class/")

    # Create fold_N/class/ directory tree up front
    for fold in range(N_FOLDS):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_ROOT, f"fold_{fold}", cls), exist_ok=True)

    placed = 0
    missing = []
    for row in df.itertuples(index=False):
        fold_dir = os.path.join(OUT_ROOT, f"fold_{row.fold}", row.dx)
        dst = os.path.join(fold_dir, row.image_id + ".jpg")
        try:
            src = find_image(row.image_id)
        except FileNotFoundError:
            missing.append(row.image_id)
            continue
        place_file(src, dst, copy=args.copy)
        placed += 1

    print(f"\nPlaced {placed} / {len(df)} images.")
    if missing:
        print(f"WARNING: {len(missing)} images from the CSV were not found "
              f"on disk (first few: {missing[:5]})")

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

    print(f"\nDone. Folder structure is at: {OUT_ROOT}/fold_<0-4>/<class>/")
    print("Note: this tree is derived output — HAM10000_kfold_split.csv "
          "(fold assignments) remains the source of truth. If you ever need "
          "to rebuild this tree, delete the kfold/ folder and rerun this script.")
    print("Reminder: for each of the 5 experiments, one fold_<i>/ is "
          "validation and the other four fold folders are training — there "
          "is no separate test/ folder in this protocol.")


if __name__ == "__main__":
    main()
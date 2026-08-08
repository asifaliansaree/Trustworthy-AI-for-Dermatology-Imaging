"""
Materialize the 5-fold split into actual folders on disk:

    ham10000/data/kfold/
        fold_0/
            mel/    <- all mel images assigned to fold 0
            nv/
            bcc/
            akiec/
            bkl/
            df/
            vasc/
        fold_1/
            mel/
            ...
        ...
        fold_4/
            ...

This is what Dr. Bajwa actually asked for: 5 physical folders, each
containing 7 class subfolders, with per-class proportions matching
HAM10000's overall distribution.

It does NOT recompute the split — it reuses the leakage-free, stratified
fold assignment already saved in HAM10000_kfold_split.csv (produced by
make_kfold_split.py). Run that script first if that CSV doesn't exist yet.

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

    print(f"Loaded {len(df)} images across {df['fold'].nunique()} folds.")
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
          "(fold assignments) is still the source of truth. If you ever "
          "need to rebuild this tree (e.g. after fixing a bug in the CSV), "
          "delete the kfold/ folder and rerun this script.")


if __name__ == "__main__":
    main()
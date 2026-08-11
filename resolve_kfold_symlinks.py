#!/usr/bin/env python3
"""
Resolve symlinks in the kfold/ dataset directory into real file copies.

Why: Kaggle dataset uploads don't preserve/resolve Unix symlinks. Instead of
following the link, Kaggle's ingestion writes the raw `readlink()` target
string as a small text file in place of the image. This script walks the
kfold/ tree, finds every symlink, and replaces it with a real copy of the
image it points to -- so the zip you upload contains actual image bytes.

Usage:
    python resolve_kfold_symlinks.py --kfold-dir kfold

Run this from the directory that also contains HAM10000_images_part_1/
and HAM10000_images_part_2/ (i.e. the same layout the symlinks were
created relative to).
"""

import argparse
import os
import shutil
import sys


def resolve_symlinks(kfold_dir: str, dry_run: bool = False) -> None:
    total = 0
    resolved = 0
    broken = 0
    skipped_not_link = 0

    for root, dirs, files in os.walk(kfold_dir):
        # skip macOS junk if present
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total += 1
            fpath = os.path.join(root, fname)

            if not os.path.islink(fpath):
                skipped_not_link += 1
                continue

            target = os.readlink(fpath)
            abs_target = os.path.normpath(os.path.join(root, target))

            if not os.path.exists(abs_target):
                print(f"[BROKEN] {fpath} -> {target} (target not found)")
                broken += 1
                continue

            if dry_run:
                print(f"[WOULD RESOLVE] {fpath} -> {abs_target}")
                resolved += 1
                continue

            # Replace symlink with a real copy of the target file
            os.remove(fpath)
            shutil.copy2(abs_target, fpath)
            resolved += 1

            if resolved % 1000 == 0:
                print(f"  ...resolved {resolved} so far")

    print("\n" + "=" * 50)
    print(f"Total image entries scanned : {total}")
    print(f"Symlinks resolved to copies : {resolved}")
    print(f"Broken symlinks (skipped)   : {broken}")
    print(f"Already real files (skipped): {skipped_not_link}")
    print("=" * 50)

    if broken > 0:
        print(
            f"\nWARNING: {broken} symlinks pointed to missing files. "
            "These need investigation before re-upload -- the resulting "
            "dataset will still be missing those images."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kfold-dir",
        default="kfold",
        help="Path to the kfold directory containing fold_0..fold_4/test (default: kfold)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be resolved without making changes",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.kfold_dir):
        print(f"ERROR: '{args.kfold_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    resolve_symlinks(args.kfold_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
Shared 5-fold lesion-wise stratified split for HAM10000.

Purpose
-------
All three interns (robustness / explainability / fairness) need to train and
evaluate on the *exact same* folds for the model comparison to be fair. This
script produces that single shared split, per Dr. Bajwa's instruction:

  - 5 folds ("chunks")
  - every fold contains all 7 lesion classes
  - each fold's per-class percentage matches the overall HAM10000 class
    distribution (stratified)
  - zero leakage: all images belonging to the same lesion_id stay in the
    SAME fold, since a lesion's images are near-duplicates/related views and
    letting them span folds would let the model "see" a fold's own lesions
    at train time (leakage). This is done via StratifiedGroupKFold, grouped
    by lesion_id, stratified by lesion-level dx.

Output
------
ham10000/data/HAM10000_kfold_split.csv — the original metadata CSV plus one
new column, `fold` (int 0-4), at IMAGE level (every image of a lesion gets
its lesion's fold number).

This does NOT touch or replace HAM10000_split.csv / split.py (the older
80/10/10 single split) — that stays as-is. This is a new, separate artifact
for the 5-fold CV comparison.

Usage
-----
    python ham10000/make_kfold_split.py

Never regenerate this after the three of you start using it for comparison
— treat HAM10000_kfold_split.csv like the old split.csv: generate once,
commit it, always load it from disk after that. Regenerating with a
different sklearn version or shuffle could reshuffle folds and silently
invalidate any results already computed against the old fold assignment.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
RANDOM_STATE = 42

METADATA_CSV = "ham10000/data/HAM10000_metadata.csv"
OUTPUT_CSV = "ham10000/data/HAM10000_kfold_split.csv"


def main():
    df = pd.read_csv(METADATA_CSV)

    print("=== INPUT ===")
    print(f"Total images: {len(df)}")
    print(f"Unique lesion_id: {df['lesion_id'].nunique()}")

    # StratifiedGroupKFold needs one (X, y, groups) row per IMAGE, not per
    # lesion — it looks at the y/groups arrays directly, and internally
    # balances folds using the *group-level* label when every image of a
    # group shares one label (true here: dx doesn't vary within lesion_id).
    y = df["dx"]
    groups = df["lesion_id"]

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, y, groups)):
        df.loc[df.index[val_idx], "fold"] = fold_idx

    assert (df["fold"] != -1).all(), "Every image must get a fold assigned."

    # --- Leakage check: no lesion_id may appear in more than one fold ---
    lesion_fold_counts = df.groupby("lesion_id")["fold"].nunique()
    leaking_lesions = lesion_fold_counts[lesion_fold_counts > 1]
    assert len(leaking_lesions) == 0, (
        f"LEAKAGE DETECTED: {len(leaking_lesions)} lesion_id(s) span "
        f"multiple folds: {leaking_lesions.index.tolist()[:10]}..."
    )
    print("\nLeakage check passed: every lesion_id is confined to exactly one fold.")

    # --- Verification: image counts per fold ---
    print("\n=== IMAGES PER FOLD ===")
    print(df["fold"].value_counts().sort_index())

    # --- Verification: does every fold contain all 7 classes? ---
    counts = df.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    print("\n=== IMAGE COUNTS PER FOLD x CLASS ===")
    print(counts)
    missing_classes = counts.columns[(counts == 0).any(axis=0)]
    assert len(missing_classes) == 0, (
        f"Fold(s) missing at least one class entirely: {missing_classes.tolist()}"
    )
    print("\nAll 5 folds contain all 7 classes.")

    # --- Verification: per-fold class percentages vs overall percentages ---
    overall_pct = df["dx"].value_counts(normalize=True).sort_index() * 100
    fold_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    print("\n=== CLASS % PER FOLD (rows) vs OVERALL % (last row) ===")
    display_pct = fold_pct.copy()
    display_pct.loc["overall"] = overall_pct
    print(display_pct.round(2))

    max_dev = (fold_pct - overall_pct).abs().values.max()
    print(f"\nMax deviation from overall class % across all folds/classes: {max_dev:.2f} pts")
    print("(Small deviations, typically well under 1-2 pts, are expected — "
          "StratifiedGroupKFold trades a little stratification precision "
          "for the hard no-leakage constraint on lesion_id groups.)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print("Share this file (or its git-committed path) with both other interns "
          "— everyone should load folds from this single CSV.")


if __name__ == "__main__":
    main()
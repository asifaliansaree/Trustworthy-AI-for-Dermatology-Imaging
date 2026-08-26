"""
Build a fresh, stratified XAI pilot set for a given checkpoint.

Generalized version: takes --checkpoint_key instead of a hardcoded path.
Default is convnext_tiny_v1_fold0 (your chosen best checkpoint), but this
works for any checkpoint the registry discovers.

Fold-scoped output: the pilot set is saved as xai_pilot_set_fold<N>.json,
not tied to one checkpoint's name. Reason: all 6 architectures share the
same 5-fold split, so fold0's validation images are identical regardless
of architecture. Building the pilot set once per fold (from your chosen
"best" checkpoint) and reusing that exact image list for the other 5
architectures at the same fold is what makes the later cross-architecture
XAI comparison apples-to-apples -- each architecture explains the SAME
images, rather than each one grading its own easiest/hardest cases.

Usage:
    python ham10000/src/explain/build_pilot_set.py
    python ham10000/src/explain/build_pilot_set.py --checkpoint_key resnet50_v12recipe_fold0
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import load_model_and_config, resolve_checkpoint, CLASSES
from dataset import HAM10000Dataset

DATA_DIR = "ham10000/data"
KFOLD_CSV = "HAM10000_kfold_split.csv"


def run_inference(model, device, ds, batch_size=32):
    records = []
    df = ds.df
    n = len(df)
    for start in range(0, n, batch_size):
        batch_rows = df.iloc[start:start + batch_size]
        tensors, valid_rows = [], []
        for _, row in batch_rows.iterrows():
            try:
                path = ds._find_image(row["image_id"])
            except FileNotFoundError:
                print(f"[WARN] missing image, skipping: {row['image_id']}")
                continue
            img = Image.open(path).convert("RGB")
            tensors.append(ds.transform(img))
            valid_rows.append(row)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1).cpu().numpy()

        for row, p in zip(valid_rows, probs):
            true_idx = CLASSES.index(row["dx"])
            pred_idx = int(p.argmax())
            records.append({
                "image_id": row["image_id"],
                "true_label": row["dx"],
                "true_idx": true_idx,
                "pred_label": CLASSES[pred_idx],
                "pred_idx": pred_idx,
                "confidence": float(p[pred_idx]),
                "correct": pred_idx == true_idx,
            })
        print(f"  inferred {min(start + batch_size, n)}/{n}", end="\r")
    print()
    return pd.DataFrame(records)


def pick_confidence_spread(pool: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or len(pool) == 0:
        return pool.iloc[0:0]
    pool_sorted = pool.sort_values("confidence").reset_index(drop=True)
    if k >= len(pool_sorted):
        return pool_sorted
    idx = sorted(set(np.linspace(0, len(pool_sorted) - 1, k).round().astype(int)))
    return pool_sorted.iloc[idx]


def stratified_pilot_set(preds_df: pd.DataFrame, n_total: int = 50) -> pd.DataFrame:
    classes = sorted(preds_df["true_label"].unique())
    per_class_budget = max(1, n_total // len(classes))
    picked = []

    for cls in classes:
        sub = preds_df[preds_df["true_label"] == cls]
        correct, incorrect = sub[sub["correct"]], sub[~sub["correct"]]

        n_incorrect = min(len(incorrect), per_class_budget // 2)
        n_correct = min(len(correct), per_class_budget - n_incorrect)

        remaining = per_class_budget - n_incorrect - n_correct
        if remaining > 0 and len(correct) > n_correct:
            extra = min(remaining, len(correct) - n_correct)
            n_correct += extra
            remaining -= extra
        if remaining > 0 and len(incorrect) > n_incorrect:
            n_incorrect += min(remaining, len(incorrect) - n_incorrect)

        picked.append(pick_confidence_spread(correct, n_correct))
        picked.append(pick_confidence_spread(incorrect, n_incorrect))

    pilot = pd.concat(picked, ignore_index=True) if picked else preds_df.iloc[0:0]
    return pilot.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_key", default="convnext-tiny_fold0",
                         help="A key from checkpoint_registry.py's discovered list "
                              "(format: '<arch-folder>_fold<N>', e.g. 'convnext-tiny_fold0'). "
                              "Run checkpoint_registry.py to see all available keys.")
    parser.add_argument("--n_total", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    entry = resolve_checkpoint(args.checkpoint_key)
    fold = entry["fold"]
    if fold is None:
        print(f"[ERROR] Checkpoint '{args.checkpoint_key}' didn't match the "
              f"'<config>_fold<N>' naming pattern, so I don't know which CV fold's "
              f"val split to pull images from. Rename the folder to include "
              f"'_fold<N>', or edit this script to hardcode --fold explicitly.")
        sys.exit(1)

    print(f"Checkpoint key: {args.checkpoint_key}  (fold {fold})")
    model, device, cfg = load_model_and_config(checkpoint_key=args.checkpoint_key)
    print(f"Architecture: {cfg['model']['architecture']}")

    ds = HAM10000Dataset(
        data_dir=DATA_DIR, split="val",
        kfold_csv=KFOLD_CSV, folds=[fold],
        augment=False,
    )
    print(f"Fold {fold} val set: {len(ds.df)} images")

    preds_df = run_inference(model, device, ds)

    full_preds_out = f"ham10000/results/{args.checkpoint_key}_val_predictions.csv"
    os.makedirs(os.path.dirname(full_preds_out), exist_ok=True)
    preds_df.to_csv(full_preds_out, index=False)
    print(f"Saved full fold{fold} predictions ({args.checkpoint_key}) -> {full_preds_out}")
    print(f"Fold{fold} val raw accuracy (this pass): {preds_df['correct'].mean():.4f}")

    pilot = stratified_pilot_set(preds_df, n_total=args.n_total)
    print(f"\nPilot set: {len(pilot)} images across {pilot['true_label'].nunique()} classes")
    print(pilot.groupby("true_label")["correct"].agg(["count", "sum"]).rename(
        columns={"count": "n", "sum": "n_correct"}))

    pilot_out = f"ham10000/results/xai_pilot_set_fold{fold}.json"
    os.makedirs(os.path.dirname(pilot_out), exist_ok=True)
    with open(pilot_out, "w") as f:
        json.dump({
            "seed": args.seed,
            "fold": fold,
            "built_from_checkpoint": args.checkpoint_key,
            "note": ("Image list is fold-scoped, not checkpoint-scoped -- reuse "
                     "these exact image_ids when applying other architectures at "
                     "this same fold, for an apples-to-apples comparison."),
            "cases": pilot.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved pilot set ({len(pilot)} images, seed={args.seed}, fold={fold}) -> {pilot_out}")
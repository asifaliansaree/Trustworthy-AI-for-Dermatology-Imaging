"""
run_cv.py — 5-fold cross-validation runner.

Runs ham10000/src/train.py once per fold (0..4), each time holding out a
different fold as validation and training on the other 4, using the shared
lesion-wise StratifiedGroupKFold split (HAM10000_kfold_split.csv). This is
the split all three interns compare against, per Dr. Bajwa's instruction.

Each rotation is launched as its OWN subprocess (not an in-process loop)
so CUDA memory, RNG state, and any lingering dataloader workers from one
fold's training run are fully torn down before the next fold starts --
train.py's globals (e.g. its epoch loop, EMA shadow model) were written
for a single run per process, not 5 in a row.

What this script does NOT change: it never edits your base config file on
disk. For each fold it makes an in-memory copy, patches in that fold's
settings, writes it to a throwaway temp YAML under
ham10000/configs/_cv_tmp/, and points train.py at that temp file.

Per-fold overrides:
  - data.cv_kfold_csv / data.cv_val_fold  -> selects the fold split
  - output.checkpoint_dir, logging.checkpoint_dir, logging.save_dir,
    logging.experiment_name -> suffixed with _fold{N} so the 5 runs don't
    overwrite each other's checkpoints/logs
  - loss.class_counts (only if the base config uses
    loss.alpha_mode=effective_num) -> recomputed from THIS rotation's
    actual 4-fold training set, not reused from the original 80/10/10
    split. The 4 folds used as train differ slightly each rotation, so a
    single static class_counts list (as currently hardcoded in
    resnet50_v12recipe.yaml, left over from the old split) is a mismatch
    for at least 4 of the 5 rotations.

After all 5 subprocesses finish, reads each fold's best_model.pt
checkpoint (saved by train.py) for its stored val_balanced_accuracy and
prints the mean +/- std across folds -- this is the number to report for
model comparison, not any single fold's result.

Usage
-----
    python ham10000/src/run_cv.py --config ham10000/configs/resnet50_v12recipe.yaml

Resuming: if a fold's best_model.pt already exists, this script will
still re-run that fold from scratch (train.py always starts fresh) --
delete unwanted checkpoint dirs yourself first if you want to skip a
fold that already finished.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

N_FOLDS = 5
CLASS_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
CLASSES = sorted(CLASS_MAP, key=CLASS_MAP.get)  # index order -- matches evaluate.py's CLASSES
TMP_CONFIG_DIR = "ham10000/configs/_cv_tmp"

# This file lives at <REPO_ROOT>/ham10000/src/run_cv.py, so REPO_ROOT is
# two levels up. Every relative path used in this script (TMP_CONFIG_DIR,
# the train.py subprocess call) and in the YAML configs (checkpoint_dir,
# data_dir, etc.) is written relative to REPO_ROOT -- so we chdir there
# explicitly at import time instead of trusting the caller's cwd. This
# makes the script work identically whether it's invoked as
# `python ham10000/src/run_cv.py`, `python /abs/path/run_cv.py`, or from
# any other directory (e.g. after os.chdir() elsewhere in a notebook).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)


def compute_fold_class_counts(data_dir: str, kfold_csv: str, train_folds: list) -> list:
    """Class counts (in CLASS_MAP order) for exactly the images this
    rotation trains on -- i.e. the 4 non-val folds, not the full dataset
    and not the old 80/10/10 split's train counts."""
    df = pd.read_csv(os.path.join(data_dir, kfold_csv))
    train_df = df[df["fold"].isin(train_folds)]
    labels = train_df["dx"].map(CLASS_MAP).values
    counts = np.bincount(labels, minlength=len(CLASS_MAP))
    return counts.tolist()


def make_fold_config(base_cfg: dict, fold: int, kfold_csv: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    all_folds = list(range(N_FOLDS))
    train_folds = [f for f in all_folds if f != fold]

    cfg["data"]["cv_kfold_csv"] = kfold_csv
    cfg["data"]["cv_val_fold"] = fold

    base_exp = cfg["logging"]["experiment_name"]
    cfg["logging"]["experiment_name"] = f"{base_exp}_fold{fold}"
    cfg["logging"]["checkpoint_dir"] = f"{cfg['logging']['checkpoint_dir']}_fold{fold}"
    cfg["logging"]["save_dir"] = f"{cfg['logging']['save_dir']}_fold{fold}"
    cfg["output"]["checkpoint_dir"] = f"{cfg['output']['checkpoint_dir']}_fold{fold}"

    if cfg.get("loss", {}).get("alpha_mode") == "effective_num":
        cfg["loss"]["class_counts"] = compute_fold_class_counts(
            cfg["data"]["data_dir"], kfold_csv, train_folds
        )

    return cfg


def run_fold(fold: int, cfg: dict) -> str:
    os.makedirs(TMP_CONFIG_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_CONFIG_DIR, f"fold{fold}.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print("\n" + "#" * 70)
    print(f"# FOLD {fold}/{N_FOLDS - 1}  (val_fold={fold})")
    print(f"# checkpoint_dir: {cfg['output']['checkpoint_dir']}")
    print(f"# class_counts  : {cfg.get('loss', {}).get('class_counts')}")
    print("#" * 70 + "\n")

    result = subprocess.run(
        [sys.executable, "ham10000/src/train.py", "--config", tmp_path]
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training subprocess for fold {fold} failed "
                            f"(exit code {result.returncode}).")

    return cfg["output"]["checkpoint_dir"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="Base config, e.g. ham10000/configs/resnet50_v12recipe.yaml")
    parser.add_argument("--kfold-csv", default="HAM10000_kfold_split.csv",
                         help="Filename (relative to data.data_dir) of the "
                              "fold-assignment CSV from make_kfold_split.py")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)),
                         help="Which fold indices to run as val (default: all 5).")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    base_experiment_name = base_cfg["logging"]["experiment_name"]

    ckpt_dirs = {}
    for fold in args.folds:
        cfg = make_fold_config(base_cfg, fold, args.kfold_csv)
        ckpt_dirs[fold] = run_fold(fold, cfg)

    # ── Aggregate ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    # Pull every metric train.py now stores in the checkpoint payload
    # (previously only val_balanced_accuracy was read here, so the CV
    # summary silently dropped accuracy/macro_f1/precision/recall even
    # though train.py was already computing them per fold).
    metric_keys = {
        "val_accuracy":          "accuracy",
        "val_balanced_accuracy": "balanced_accuracy",
        "val_macro_f1":          "macro_f1",
        "val_precision":         "macro_precision",
        "val_recall":            "macro_recall",
        "val_ece":               "ece",
        "val_roc_auc_macro":     "roc_auc_macro",
    }
    fold_metrics = {name: {} for name in metric_keys}

    # Per-class (per-lesion) metrics -- one of these three dicts per fold,
    # each keyed by lesion name (e.g. fold_per_class["val_per_class_f1"][2]["mel"]).
    per_class_keys = ["val_per_class_f1", "val_per_class_precision", "val_per_class_recall"]
    fold_per_class = {name: {} for name in per_class_keys}

    fold_confusion_matrices = {}
    fold_epochs = {}

    for fold, ckpt_dir in ckpt_dirs.items():
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        if not os.path.exists(best_path):
            print(f"fold {fold}: MISSING best_model.pt at {best_path}")
            continue
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        fold_epochs[fold] = payload.get("epoch")

        line = f"fold {fold} (epoch {payload['epoch']}): "
        parts = []
        for payload_key, label in metric_keys.items():
            # Older checkpoints (trained before this metric existed) won't
            # have every key -- skip missing ones instead of crashing the
            # whole summary over one stale fold.
            if payload_key not in payload:
                continue
            val = payload[payload_key]
            fold_metrics[payload_key][fold] = val
            parts.append(f"{label}={val:.4f}")
        print(line + "  ".join(parts))

        for payload_key in per_class_keys:
            if payload_key in payload:
                fold_per_class[payload_key][fold] = payload[payload_key]

        if "val_confusion_matrix" in payload:
            fold_confusion_matrices[fold] = payload["val_confusion_matrix"]

    print("-" * 70)
    for payload_key, label in metric_keys.items():
        vals = list(fold_metrics[payload_key].values())
        if not vals:
            continue
        print(f"Mean {label:<20s} over {len(vals)} fold(s): "
              f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print("-" * 70)
    print("Per-lesion (mean +/- std across folds):")
    per_class_summary = {payload_key: {} for payload_key in per_class_keys}
    for payload_key in per_class_keys:
        label = payload_key.replace("val_per_class_", "")
        for cls in CLASSES:
            vals = [d[cls] for d in fold_per_class[payload_key].values() if cls in d]
            if not vals:
                continue
            mean_v, std_v = float(np.mean(vals)), float(np.std(vals))
            per_class_summary[payload_key][cls] = {"mean": mean_v, "std": std_v, "n_folds": len(vals)}
            print(f"  {cls:<7s} {label:<10s}: {mean_v:.4f} +/- {std_v:.4f}")
    print("=" * 70)

    # ── Write everything to one JSON so it doesn't just live in console
    # output -- overall (mean/std/per-fold) + per-lesion (mean/std/per-fold)
    # + confusion matrices, all keyed by fold. ──────────────────────
    results_dir = "ham10000/results"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{base_experiment_name}_cv_summary.json")

    summary = {
        "experiment": base_experiment_name,
        "folds_run": list(ckpt_dirs.keys()),
        "fold_epochs": fold_epochs,
        "overall": {
            label: {
                "mean": float(np.mean(list(fold_metrics[key].values()))),
                "std": float(np.std(list(fold_metrics[key].values()))),
                "per_fold": {str(f): float(v) for f, v in fold_metrics[key].items()},
            }
            for key, label in metric_keys.items() if fold_metrics[key]
        },
        "per_lesion": {
            payload_key.replace("val_per_class_", ""): {
                cls: {
                    **per_class_summary[payload_key][cls],
                    "per_fold": {
                        str(f): float(d[cls])
                        for f, d in fold_per_class[payload_key].items() if cls in d
                    },
                }
                for cls in per_class_summary[payload_key]
            }
            for payload_key in per_class_keys
        },
        "confusion_matrices_per_fold": {str(f): m for f, m in fold_confusion_matrices.items()},
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote full CV summary (overall + per-lesion + confusion matrices) to {out_path}")


if __name__ == "__main__":
    main()
"""
calibrate.py — Per-class decision-bias calibration, fit on val, applied to test once.

Some models over/under-predict certain classes even when the underlying
features are fine (e.g. resnet50_v12recipe over-predicts `mel`, partly
trading `nv` recall for `mel` recall via its loss weighting). This script
checks how much of that is fixable with a per-class multiplicative
correction on the softmax outputs, WITHOUT retraining and WITHOUT touching
the test set during fitting.

Protocol (important — this is the whole point of the script):
  1. Fit per-class weights by coordinate-ascent search, maximizing
     BALANCED ACCURACY ON VAL ONLY.
  2. Freeze those weights.
  3. Apply them to the test set's probabilities exactly once.
  4. Report both the val fit and the single frozen test result.

Test labels are never used for fitting — only for the single final score.

Usage:
    python ham10000/src/calibrate.py \
        --val_probs  ham10000/results/resnet50_v12recipe_val_probs.npz \
        --test_probs ham10000/results/resnet50_v12recipe_test_probs.npz \
        --out        ham10000/results/resnet50_v12recipe_calibration.json
"""

import argparse
import json

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def fit_class_weights(y_true, y_probs, n_classes,
                       candidate_multipliers=None, max_rounds=6):
    """
    Coordinate-ascent search for per-class multiplicative weights that
    maximize balanced accuracy on the data passed in. Caller controls
    what data this is -- always call with VAL data only.
    """
    if candidate_multipliers is None:
        candidate_multipliers = np.concatenate([
            np.arange(0.3, 1.01, 0.05),
            np.arange(1.0, 3.01, 0.1),
        ])

    weights = np.ones(n_classes)

    def score_of(w):
        preds = (y_probs * w[None, :]).argmax(axis=1)
        return balanced_accuracy_score(y_true, preds)

    improved = True
    round_idx = 0
    while improved and round_idx < max_rounds:
        improved = False
        round_idx += 1
        for k in range(n_classes):
            best_w, best_score = weights[k], score_of(weights)
            for cand in candidate_multipliers:
                trial = weights.copy()
                trial[k] = cand
                s = score_of(trial)
                if s > best_score + 1e-6:
                    best_score, best_w = s, cand
            if best_w != weights[k]:
                weights[k] = best_w
                improved = True

    return weights


def evaluate(y_true, y_probs, weights, classes):
    adj = y_probs * weights[None, :]
    y_pred = adj.argmax(axis=1)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(classes, per_class_f1)},
        "confusion_matrix": cm.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_probs", required=True,
                   help="npz saved by evaluate_test.py --split val")
    p.add_argument("--test_probs", required=True,
                   help="npz saved by evaluate_test.py --split test")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: alongside test_probs)")
    args = p.parse_args()

    val = np.load(args.val_probs, allow_pickle=True)
    test = np.load(args.test_probs, allow_pickle=True)

    val_classes = list(val["classes"])
    test_classes = list(test["classes"])
    assert val_classes == test_classes, (
        f"Class order mismatch between val ({val_classes}) and "
        f"test ({test_classes}) probability files -- refusing to calibrate."
    )
    classes = val_classes
    n_classes = len(classes)

    # Hard safety check: refuse to proceed if val and test are actually the
    # same data (e.g. both args accidentally point at the same file). This
    # exists because that mistake silently reproduces test-set-fit numbers
    # while still printing "val" labels -- exactly the leakage this script
    # is meant to prevent.
    same_shape = val["y_true"].shape == test["y_true"].shape
    if same_shape and np.array_equal(val["y_true"], test["y_true"]) \
            and np.allclose(val["y_probs"], test["y_probs"]):
        raise SystemExit(
            "ERROR: --val_probs and --test_probs contain identical data.\n"
            "This almost always means both flags point at the same file "
            "(or the same split was saved twice under different names).\n"
            f"  val_probs  : {args.val_probs}  ({len(val['y_true'])} images)\n"
            f"  test_probs : {args.test_probs}  ({len(test['y_true'])} images)\n"
            "Refusing to calibrate -- fitting and evaluating on the same "
            "data defeats the entire point of this script. Re-check that "
            "val_probs was saved from --split val and test_probs from "
            "--split test."
        )

    print(f"\nVal set  : {len(val['y_true'])} images")
    print(f"Test set : {len(test['y_true'])} images")
    print(f"Classes  : {classes}\n")

    # ── Baselines (uncalibrated, weights = 1) ───────────────────────
    identity = np.ones(n_classes)
    val_before = evaluate(val["y_true"], val["y_probs"], identity, classes)
    test_before = evaluate(test["y_true"], test["y_probs"], identity, classes)

    print("=" * 60)
    print("BEFORE calibration")
    print("=" * 60)
    print(f"  Val  balanced accuracy : {val_before['balanced_accuracy']:.4f}")
    print(f"  Test balanced accuracy : {test_before['balanced_accuracy']:.4f}")
    print(f"  Test macro F1          : {test_before['macro_f1']:.4f}")

    # ── Fit on VAL ONLY ──────────────────────────────────────────────
    print("\nFitting per-class weights on VAL ONLY ...")
    weights = fit_class_weights(val["y_true"], val["y_probs"], n_classes)

    print("\nLearned weights:")
    for c, w in zip(classes, weights):
        print(f"  {c:6s}: {w:.2f}")

    # ── Apply frozen weights: val (sanity check) + test (final, once) ─
    val_after = evaluate(val["y_true"], val["y_probs"], weights, classes)
    test_after = evaluate(test["y_true"], test["y_probs"], weights, classes)

    print("\n" + "=" * 60)
    print("AFTER calibration (weights fit on val, applied to test ONCE)")
    print("=" * 60)
    print(f"  Val  balanced accuracy : {val_after['balanced_accuracy']:.4f}  "
          f"(was {val_before['balanced_accuracy']:.4f})")
    print(f"  Test balanced accuracy : {test_after['balanced_accuracy']:.4f}  "
          f"(was {test_before['balanced_accuracy']:.4f})")
    print(f"  Test macro F1          : {test_after['macro_f1']:.4f}  "
          f"(was {test_before['macro_f1']:.4f})")
    print("\nTest per-class F1: before -> after")
    for c in classes:
        fb = test_before["per_class_f1"][c]
        fa = test_after["per_class_f1"][c]
        print(f"  {c:6s}: {fb:.4f} -> {fa:.4f}  (delta {fa - fb:+.4f})")

    out_path = args.out or args.test_probs.replace("_probs.npz", "_calibration.json")
    result = {
        "classes": classes,
        "learned_weights": {c: round(float(w), 3) for c, w in zip(classes, weights)},
        "val_before": val_before,
        "val_after": val_after,
        "test_before": test_before,
        "test_after": test_after,
        "note": ("Weights fit on val only; test_after is the single, "
                 "non-cherry-picked test-set application of those frozen weights."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
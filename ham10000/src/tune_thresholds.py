"""
tune_thresholds.py

Threshold tuning for resnet50_v12recipe -- NO retraining, NO change to the
model. Uses saved softmax probabilities only.

WHY THIS EXISTS
---------------
v12recipe's argmax predictions already carry good class-separation info
(macro_auc 0.9465, the best of any model trained so far) but the model's
*implicit* argmax threshold isn't necessarily the best operating point for
mel/akiec, where missing a case (false negative) is far costlier than a
false alarm (false positive). Both prior attempts to improve this model
by RETRAINING (sampler-off, mixup) traded mel/akiec recall for nv/bcc
precision -- the wrong direction for a screening tool. This script tries
to get that trade the other way round, for free, using probabilities the
model already produces.

METHOD
------
1. mel gets first priority: if P(mel) >= tau_mel, predict mel, no matter
   what the raw argmax said.
2. If not flagged as mel, akiec gets second priority: if P(akiec) >=
   tau_akiec, predict akiec.
3. Everything else falls back to ordinary argmax over all 7 classes.

This priority-override approach (rather than independently thresholding
all 7 classes) mirrors real triage: check for the most dangerous
diagnosis first, then the next most dangerous, then default to "most
likely" for everything left over. It only touches mel/akiec decisions;
nv/bcc/bkl/df/vasc predictions are unaffected unless they get overridden
BY a higher-priority flag.

tau_mel and tau_akiec are chosen ONLY from validation data (Youden's J:
the threshold that maximizes TPR - FPR in each class's one-vs-rest ROC),
then applied once to test data for reporting. No threshold is ever
touched or re-picked after seeing test results.

USAGE
-----
python tune_thresholds.py \
    --val_probs  ham10000/results/resnet50_v12recipe_val_probs.npz \
    --test_probs ham10000/results/resnet50_v12recipe_test_probs.npz

IMPORTANT: the script checks that val_probs and test_probs are NOT
identical arrays before doing anything else, and refuses to run if they
are -- that exact bug (a mislabeled/duplicated file) is what blocked this
analysis the first time round.
"""
import argparse
import numpy as np
from sklearn.metrics import (
    roc_curve, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

CLASSES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
PRIORITY_CLASSES = ['mel', 'akiec']  # order matters: mel checked first


def load_probs(path):
    d = np.load(path)
    return d['y_true'], d['y_probs'], list(d['classes'])


def best_threshold_youden(y_true, class_probs, class_idx):
    """One-vs-rest Youden's J: threshold maximizing TPR - FPR."""
    y_binary = (y_true == class_idx).astype(int)
    fpr, tpr, thresholds = roc_curve(y_binary, class_probs)
    j = tpr - fpr
    best_i = int(np.argmax(j))
    return float(thresholds[best_i]), float(tpr[best_i]), float(fpr[best_i])


def apply_priority_thresholds(y_probs, classes, thresholds):
    """
    thresholds: dict like {'mel': 0.31, 'akiec': 0.22}
    Returns predicted class indices after priority-override logic.
    """
    n = y_probs.shape[0]
    preds = np.full(n, -1, dtype=int)

    for cls in PRIORITY_CLASSES:
        if cls not in thresholds:
            continue
        idx = classes.index(cls)
        tau = thresholds[cls]
        still_undecided = preds == -1
        flagged = still_undecided & (y_probs[:, idx] >= tau)
        preds[flagged] = idx

    # everything left over: ordinary argmax
    remaining = preds == -1
    preds[remaining] = y_probs[remaining].argmax(axis=1)
    return preds


def report_metrics(name, y_true, preds):
    bal_acc  = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Balanced accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(classification_report(
        y_true, preds, target_names=CLASSES, digits=4, zero_division=0
    ))
    return bal_acc, macro_f1


def sweep_grid(y_true, y_probs, classes, cls_name, taus):
    """Print recall/precision/support at each candidate threshold, holding
    the OTHER priority class's threshold at its Youden pick, so you can see
    the actual trade-off curve before committing to one operating point."""
    idx = classes.index(cls_name)
    print(f"\n--- {cls_name} threshold sweep (this class only, "
          f"other priority thresholds off) ---")
    print(f"{'tau':>6} | {'recall':>7} | {'precision':>9} | {'n_flagged':>9}")
    for tau in taus:
        preds = apply_priority_thresholds(y_probs, classes, {cls_name: tau})
        flagged = preds == idx
        tp = int(((y_true == idx) & flagged).sum())
        actual_pos = int((y_true == idx).sum())
        pred_pos = int(flagged.sum())
        recall = tp / actual_pos if actual_pos else float("nan")
        precision = tp / pred_pos if pred_pos else float("nan")
        print(f"{tau:6.3f} | {recall:7.4f} | {precision:9.4f} | {pred_pos:9d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_probs", required=True)
    ap.add_argument("--test_probs", required=True)
    args = ap.parse_args()

    y_val, p_val, classes_val = load_probs(args.val_probs)
    y_test, p_test, classes_test = load_probs(args.test_probs)
    assert classes_val == CLASSES == classes_test, \
        "Class ordering mismatch between files -- check CLASSES list."

    # Guard against the exact bug that blocked this analysis before:
    # val_probs and test_probs must NOT be the same data.
    if y_val.shape == y_test.shape and np.allclose(p_val, p_test):
        raise SystemExit(
            "\nREFUSING TO RUN: --val_probs and --test_probs are identical "
            "arrays. This means the 'val' file was actually generated from "
            "the test split (the same bug found in "
            "resnet50_v12recipe_val_probs.npz). Regenerate a genuine val "
            "split with:\n"
            "  python ham10000/src/evaluate_test.py --config ... "
            "--checkpoint ... --split val\n"
            "and confirm the printed image count matches your training "
            "log's val count (1009), not test (989), before rerunning this."
        )

    print(f"Loaded val:  {len(y_val)} samples")
    print(f"Loaded test: {len(y_test)} samples")

    # ── Step 1: baseline (plain argmax) on both splits ──────────────
    base_val_preds  = p_val.argmax(axis=1)
    base_test_preds = p_test.argmax(axis=1)
    report_metrics("BASELINE (argmax) -- val",  y_val,  base_val_preds)
    report_metrics("BASELINE (argmax) -- test", y_test, base_test_preds)

    # ── Step 2: pick thresholds via Youden's J on val ONLY ──────────
    thresholds = {}
    print(f"\n{'=' * 60}\nTHRESHOLD SELECTION (val data only)\n{'=' * 60}")
    for cls in PRIORITY_CLASSES:
        idx = classes_val.index(cls)
        tau, tpr, fpr = best_threshold_youden(y_val, p_val[:, idx], idx)
        thresholds[cls] = tau
        print(f"{cls:6s}: tau={tau:.4f}  (val TPR={tpr:.4f}, val FPR={fpr:.4f})")

    # ── Step 3: show the trade-off curve around each Youden pick ────
    for cls in PRIORITY_CLASSES:
        center = thresholds[cls]
        taus = sorted(set(
            max(0.0, round(center + step, 3))
            for step in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        ))
        sweep_grid(y_val, p_val, classes_val, cls, taus)

    # ── Step 4: apply the val-derived thresholds ONCE to both splits ─
    tuned_val_preds  = apply_priority_thresholds(p_val,  classes_val,  thresholds)
    tuned_test_preds = apply_priority_thresholds(p_test, classes_test, thresholds)

    report_metrics(
        f"TUNED (mel>={thresholds['mel']:.3f}, "
        f"akiec>={thresholds['akiec']:.3f}) -- val",
        y_val, tuned_val_preds,
    )
    val_bal_acc, val_f1 = report_metrics(
        f"TUNED (mel>={thresholds['mel']:.3f}, "
        f"akiec>={thresholds['akiec']:.3f}) -- test",
        y_test, tuned_test_preds,
    )

    print(f"\n{'=' * 60}\nSUMMARY (test split)\n{'=' * 60}")
    base_bal_acc, base_f1 = (
        balanced_accuracy_score(y_test, base_test_preds),
        f1_score(y_test, base_test_preds, average="macro", zero_division=0),
    )
    print(f"{'':20s} {'bal_acc':>10} {'macro_f1':>10}")
    print(f"{'baseline argmax':20s} {base_bal_acc:10.4f} {base_f1:10.4f}")
    print(f"{'tuned thresholds':20s} {val_bal_acc:10.4f} {val_f1:10.4f}")
    print(f"{'delta':20s} {val_bal_acc - base_bal_acc:+10.4f} "
          f"{val_f1 - base_f1:+10.4f}")


if __name__ == "__main__":
    main()
"""
check_overfit.py — consistent overfitting diagnostic for any training_log.csv

Usage:
    python check_overfit.py path/to/training_log.csv

Applies the same criteria used to judge resnet18_v6/v11/v12/v14, so every
architecture in the sweep is being held to the same bar.
"""
import sys
import csv


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["epoch"] = int(r["epoch"])
        r["train_loss"] = float(r["train_loss"])
        r["val_loss"] = float(r["val_loss"])
        r["val_bal_acc"] = float(r["val_bal_acc"])
    return rows


def check(path):
    rows = load_log(path)
    best = max(rows, key=lambda r: r["val_bal_acc"])
    best_idx = rows.index(best)
    final = rows[-1]

    print(f"Total epochs run     : {len(rows)}")
    print(f"Best epoch           : {best['epoch']}  (val_bal_acc={best['val_bal_acc']:.4f})")
    print(f"Final epoch          : {final['epoch']} (val_bal_acc={final['val_bal_acc']:.4f})")
    print(f"Train/val loss @best : {best['train_loss']:.4f} / {best['val_loss']:.4f}  "
          f"(gap={best['val_loss']-best['train_loss']:.4f})")

    # val_loss floor: does it stay flat after the best epoch, or keep rising?
    post_best = rows[best_idx:]
    if len(post_best) >= 3:
        val_losses_after = [r["val_loss"] for r in post_best]
        rise = val_losses_after[-1] - val_losses_after[0]
        if rise > 0.03:
            trend = f"RISING (+{rise:.4f} after best epoch) -> overfitting still worsening"
        elif rise < -0.01:
            trend = f"still improving (-{abs(rise):.4f}) -> hadn't peaked, consider more epochs/patience"
        else:
            trend = f"flat ({rise:+.4f}) -> overfitting has plateaued, this is the healthy pattern"
    else:
        trend = "not enough post-best epochs logged to judge"
    print(f"Val_loss trend after best epoch: {trend}")

    # bal_acc drift after best epoch
    acc_drift = final["val_bal_acc"] - best["val_bal_acc"]
    print(f"val_bal_acc drift, best->final : {acc_drift:+.4f}")

    print()
    if rise > 0.03:
        print("VERDICT: overfitting is actively worsening (like v6). Needs more regularization.")
    elif best["val_loss"] - best["train_loss"] < 0.08 and best["val_bal_acc"] < 0.75:
        print("VERDICT: small gap but low accuracy -> likely UNDERFITTING (like v13/v2_tuned), not good fit.")
    else:
        print("VERDICT: mild, plateaued gap -> healthy fit, consistent with v11/v12.")


if __name__ == "__main__":
    check(sys.argv[1])
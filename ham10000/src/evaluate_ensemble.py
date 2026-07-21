"""
evaluate_ensemble.py — Weighted-softmax ensemble evaluator.

Loads N trained checkpoints (each with its own config, since they may be
different architectures), runs inference for each (optionally with TTA),
and combines their softmax outputs via a weighted average. No retraining —
pure inference-time combination.

Usage (2-model ensemble, equal weight, 1 pass each = no TTA):
    python ham10000/src/evaluate_ensemble.py \
        --configs     ham10000/configs/efficientnet_b0.yaml ham10000/configs/densenet121_v12recipe.yaml \
        --checkpoints ham10000/checkpoints/efficientnet_b0/best_model.pt ham10000/checkpoints/densenet121_v12recipe/best_model.pt \
        --weights     0.5 0.5

Usage (weighted toward the stabler model, + TTA on each):
    python ham10000/src/evaluate_ensemble.py \
        --configs     ham10000/configs/efficientnet_b0.yaml ham10000/configs/densenet121_v12recipe.yaml \
        --checkpoints ham10000/checkpoints/efficientnet_b0/best_model.pt ham10000/checkpoints/densenet121_v12recipe/best_model.pt \
        --weights     0.6 0.4 \
        --tta_passes  8

Weights are for the FINAL ensemble combination step, not training. If you
don't know which weighting is best, leave --weights unset — the script will
grid-search 0.0..1.0 in steps of 0.05 (2-model case only) on the test set
and report the best split. That's inference-only search, not tuning against
held-out labels you don't have access to at deployment, so treat the
grid-searched number as an upper bound / sanity check rather than the
number you report as final performance — for a real number, do this split
search on a validation set, then report test performance at the chosen
weight, exactly like you would for any other hyperparameter.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    roc_auc_score, classification_report, log_loss
)

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in [_THIS, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import build_model
from metadata_encoder import MetadataEncoder

CLASSES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_MAP = {"mel": 0, "nv": 1, "bcc": 2, "akiec": 3, "bkl": 4, "df": 5, "vasc": 6}
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]


def tta_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def base_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


class TestDataset(Dataset):
    """Same test split logic as evaluate_tta.py — must match exactly across
    models being ensembled, or you're averaging predictions on different
    example orderings without realizing it."""

    def __init__(self, data_dir: str, encoder=None):
        df = pd.read_csv(os.path.join(data_dir, "HAM10000_split.csv"))
        self.df = df[df["split"] == "test"].reset_index(drop=True)
        self.dirs = [
            os.path.join(data_dir, "HAM10000_images_part_1"),
            os.path.join(data_dir, "HAM10000_images_part_2"),
        ]
        self.encoder = encoder

    def __len__(self):
        return len(self.df)

    def _find(self, image_id):
        for d in self.dirs:
            p = os.path.join(d, image_id + ".jpg")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(image_id)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._find(row["image_id"])).convert("RGB")
        label = CLASS_MAP[row["dx"]]
        if self.encoder is not None:
            return img, self.encoder.encode(row), label
        return img, label


@torch.no_grad()
def run_model(model, dataset, device, img_size, n_passes, use_meta):
    """Returns (labels, avg_softmax_probs) for one model, averaged over
    n_passes (n_passes=1 -> deterministic base_transform only, no TTA)."""
    model.eval()
    n = len(dataset)
    accumulated = np.zeros((n, len(CLASSES)), dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)

    for pass_idx in range(n_passes):
        transform = base_transform(img_size) if pass_idx == 0 else tta_transform(img_size)
        for idx in range(n):
            item = dataset[idx]
            if use_meta:
                img, meta, label = item
                meta_t = meta.unsqueeze(0).to(device)
            else:
                img, label = item
                meta_t = None
            img_t = transform(img).unsqueeze(0).to(device)
            logits = model(img_t, meta_t)
            accumulated[idx] += F.softmax(logits, dim=1).cpu().numpy()[0]
            labels[idx] = label

    return labels, accumulated / n_passes


def score(y_true, avg_probs, weights, label=""):
    """Combine per-model avg_probs (list of arrays) with given weights,
    then report bal_acc / macro F1 / AUC / log loss."""
    combined = sum(w * p for w, p in zip(weights, avg_probs))
    combined = combined / combined.sum(axis=1, keepdims=True)  # renormalize
    preds = combined.argmax(axis=1)

    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, combined, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    try:
        ce = log_loss(y_true, combined, labels=list(range(len(CLASSES))))
    except Exception:
        ce = float("nan")

    print(f"{label:30s}  bal_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}  "
          f"macro_auc={auc:.4f}  val_ce_loss={ce:.4f}")
    return bal_acc, macro_f1, auc, ce, preds, combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Must sum to 1 and match --configs length. "
                         "Omit to grid-search (2-model case only).")
    p.add_argument("--tta_passes", type=int, default=1,
                    help="1 = no TTA (single deterministic pass per model). "
                         "8 is a reasonable TTA budget per model.")
    args = p.parse_args()

    assert len(args.configs) == len(args.checkpoints), \
        "Need one checkpoint per config."
    if args.weights is not None:
        assert len(args.weights) == len(args.configs), \
            "Need one weight per model."
        assert abs(sum(args.weights) - 1.0) < 1e-6, "Weights must sum to 1."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = None
    per_model_probs = []
    names = []

    for cfg_path, ckpt_path in zip(args.configs, args.checkpoints):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        use_meta = cfg["model"].get("metadata_dim", 0) > 0
        img_size = cfg["data"].get("img_size", 224)
        name = cfg["logging"]["experiment_name"]
        names.append(name)

        encoder = None
        if use_meta:
            encoder = MetadataEncoder(
                os.path.join(cfg["data"]["data_dir"], "HAM10000_split.csv")
            )

        model = build_model(cfg).to(device)
        # weights_only=False: PyTorch >=2.6 defaults to True, which rejects
        # numpy scalar types (val_balanced_accuracy was saved as one). Safe
        # here since these are checkpoints you trained and pushed yourself,
        # not downloaded from a third party.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[{name}] loaded epoch {ckpt['epoch']}, "
              f"val_bal_acc={ckpt['val_balanced_accuracy']:.4f}")

        dataset = TestDataset(cfg["data"]["data_dir"], encoder)
        labels, probs = run_model(
            model, dataset, device, img_size,
            n_passes=args.tta_passes, use_meta=use_meta,
        )
        if all_labels is None:
            all_labels = labels
        else:
            # Sanity check: every model must be scored on the identical,
            # identically-ordered test set, or the ensemble is meaningless.
            assert np.array_equal(all_labels, labels), \
                f"Label order mismatch for {name} — test split differs across configs."
        per_model_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"ENSEMBLE: {' + '.join(names)}  (tta_passes={args.tta_passes})")
    print("=" * 70)

    if args.weights is not None:
        bal_acc, macro_f1, auc, ce, preds, combined = score(
            all_labels, per_model_probs, args.weights, label="weighted ensemble"
        )
        best_weights = args.weights
    elif len(per_model_probs) == 2:
        print("No --weights given — grid-searching w in [0.0, 1.0] step 0.05 "
              "for model[0], (1-w) for model[1]:\n")
        best = (-1, None, None)
        for w in np.arange(0.0, 1.01, 0.05):
            ws = [round(w, 2), round(1 - w, 2)]
            bal_acc, *_ = score(all_labels, per_model_probs, ws, label=f"w={ws[0]:.2f}")
            if bal_acc > best[0]:
                best = (bal_acc, ws, None)
        print(f"\nBest weight split by bal_acc: {best[1]} -> bal_acc={best[0]:.4f}")
        print("NOTE: this was searched directly on the test set — it's an "
              "upper-bound sanity check, not a number to report as final. "
              "Re-run this same search on a val split, fix the weight, THEN "
              "score once on test for the number that goes in your report.")
        best_weights = best[1]
        bal_acc, macro_f1, auc, ce, preds, combined = score(
            all_labels, per_model_probs, best_weights, label="best (for reference)"
        )
    else:
        raise ValueError("--weights required when ensembling more than 2 models.")

    report = classification_report(
        all_labels, preds, target_names=CLASSES, digits=4, zero_division=0
    )
    print(f"\n{report}")

    os.makedirs("ham10000/results", exist_ok=True)
    out_name = f"ensemble_{'_'.join(names)}"
    results = {
        "experiment": out_name,
        "models": names,
        "weights": best_weights,
        "tta_passes": args.tta_passes,
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(auc),
        "val_ce_loss": float(ce),
    }
    with open(f"ham10000/results/{out_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(f"ham10000/results/{out_name}_report.txt", "w") as f:
        f.write(report)
    print(f"\nSaved -> ham10000/results/{out_name}.json")


if __name__ == "__main__":
    main()
"""
evaluate_tta.py — Test-Time Augmentation evaluator.

Runs N augmented forward passes per test image and averages softmax outputs.
Typically gains +1–3pp balanced accuracy with zero retraining.

Usage:
    python ham10000/src/evaluate_tta.py \
        --config     ham10000/configs/efficientnet_b0.yaml \
        --checkpoint ham10000/checkpoints/efficientnet_b0/best_model.pt \
        --tta_passes 8
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in [_THIS, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model            import build_model
from metadata_encoder import MetadataEncoder

CLASSES    = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_MAP  = {"mel":0,"nv":1,"bcc":2,"akiec":3,"bkl":4,"df":5,"vasc":6}
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]


def tta_transform(img_size: int) -> T.Compose:
    """Random augmentation for TTA — same image, different crops/flips each pass."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def base_transform(img_size: int) -> T.Compose:
    """Deterministic transform for the first pass."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


class TestDataset(Dataset):
    """Minimal test dataset that returns raw PIL images for TTA."""

    def __init__(self, data_dir: str, encoder=None):
        df      = pd.read_csv(os.path.join(data_dir, "HAM10000_split.csv"))
        self.df = df[df["split"] == "test"].reset_index(drop=True)
        self.dirs = [
            os.path.join(data_dir, "HAM10000_images_part_1"),
            os.path.join(data_dir, "HAM10000_images_part_2"),
        ]
        self.encoder = encoder
        print(f"[test] {len(self.df)} images for TTA evaluation")

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
            meta = self.encoder.encode(row)
            return img, meta, label
        return img, label


@torch.no_grad()
def run_tta(model, dataset, device, img_size,
            n_passes: int, use_meta: bool):
    """
    Run TTA inference.
    Pass 0 = deterministic base transform.
    Passes 1..n_passes = random augmentation.
    Final prediction = average of all pass softmax outputs.
    """
    model.eval()
    n = len(dataset)
    accumulated = np.zeros((n, len(CLASSES)), dtype=np.float64)
    all_labels  = np.zeros(n, dtype=np.int64)

    for pass_idx in range(n_passes):
        transform = (base_transform(img_size) if pass_idx == 0
                     else tta_transform(img_size))
        print(f"  TTA pass {pass_idx + 1}/{n_passes}...", end="\r")

        for idx in range(n):
            item = dataset[idx]
            if use_meta:
                img, meta, label = item
                meta_t = meta.unsqueeze(0).to(device)
            else:
                img, label = item
                meta_t = None

            img_t  = transform(img).unsqueeze(0).to(device)
            logits = model(img_t, meta_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            accumulated[idx] += probs
            all_labels[idx]   = label

    print()
    avg_probs  = accumulated / n_passes
    all_preds  = avg_probs.argmax(axis=1)
    return all_labels, all_preds, avg_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tta_passes", type=int, default=8)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_meta   = cfg["model"].get("metadata_dim", 0) > 0
    img_size   = cfg["data"].get("img_size", 224)
    experiment = cfg["logging"]["experiment_name"]

    print(f"\nExperiment  : {experiment}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"TTA passes  : {args.tta_passes}")
    print(f"Metadata    : {use_meta}")
    print(f"Device      : {device}\n")

    encoder = None
    if use_meta:
        encoder = MetadataEncoder(
            os.path.join(cfg["data"]["data_dir"], "HAM10000_split.csv")
        )

    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch {ckpt['epoch']}, "
          f"val_bal_acc={ckpt['val_balanced_accuracy']:.4f}\n")

    dataset = TestDataset(cfg["data"]["data_dir"], encoder)
    y_true, y_pred, y_probs = run_tta(
        model, dataset, device, img_size,
        n_passes=args.tta_passes, use_meta=use_meta,
    )

    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None,     zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs,
                            multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    report = classification_report(
        y_true, y_pred, target_names=CLASSES,
        digits=4, zero_division=0
    )

    print("=" * 60)
    print(f"TTA RESULTS ({args.tta_passes} passes) — {experiment}")
    print("=" * 60)
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Macro ROC-AUC     : {auc:.4f}")
    print("\nPer-class F1:")
    for c, s in zip(CLASSES, per_class_f1):
        bar = "█" * int(s * 20)
        print(f"  {c:6s}: {s:.4f}  {bar}")
    print(f"\n{report}")

    os.makedirs("ham10000/results", exist_ok=True)
    out_name = f"{experiment}_tta{args.tta_passes}"
    results  = {
        "experiment":       out_name,
        "checkpoint":       args.checkpoint,
        "tta_passes":       args.tta_passes,
        "balanced_accuracy": float(bal_acc),
        "macro_f1":         float(macro_f1),
        "macro_auc":        float(auc),
        "per_class_f1": {
            c: round(float(s), 4)
            for c, s in zip(CLASSES, per_class_f1)
        },
    }
    json_path = f"ham10000/results/{out_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    report_path = f"ham10000/results/{out_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Saved → {json_path}")
    print(f"Saved → {report_path}")


if __name__ == "__main__":
    main()
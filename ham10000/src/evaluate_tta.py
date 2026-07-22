"""
evaluate_tta.py — Test-Time Augmentation evaluator.

Runs N augmented forward passes per test image and averages softmax outputs.
Typically gains +1–3pp balanced accuracy with zero retraining.

Usage (current best checkpoint — densenet121_v12recipe, val_bal_acc 0.8251):
    python ham10000/src/evaluate_tta.py \
        --config     ham10000/configs/densenet121_v12recipe.yaml \
        --checkpoint ham10000/checkpoints/densenet121_v12recipe/best_model.pt \
        --tta_passes 8

Optional (default batch_size/num_workers come from the config's
train.batch_size / data.num_workers, same as train.py):
        --batch_size 64 --num_workers 2
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

# Order must match CLASS_MAP's indices (0..6), not alphabetical order --
# these feed target_names / per-class zips below, so a mismatch here
# silently mislabels every per-class number while leaving the aggregate
# scores (balanced accuracy, macro F1) correct, since those are computed
# on the integer labels directly. Matches evaluate.py / evaluate_test.py.
CLASSES    = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
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


class _TTAPassDataset(Dataset):
    """
    Applies one TTA transform pass over an underlying TestDataset and
    returns already-transformed tensors, so a DataLoader can batch them.
    Looks up the same image/meta/label as TestDataset.__getitem__ --
    this is a thin wrapper, not a reimplementation of the dataset logic.
    """

    def __init__(self, base: "TestDataset", transform):
        self.base      = base
        self.transform = transform
        self.use_meta  = base.encoder is not None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.use_meta:
            img, meta, label = item
            return self.transform(img), meta, label
        img, label = item
        return self.transform(img), label


@torch.no_grad()
def run_tta(model, dataset, device, img_size,
            n_passes: int, use_meta: bool,
            batch_size: int = 32, num_workers: int = 0):
    """
    Run TTA inference.
    Pass 0 = deterministic base transform.
    Passes 1..n_passes = random augmentation.
    Final prediction = average of all pass softmax outputs.

    Batched via DataLoader (shuffle=False) instead of one image at a time.
    This changes nothing about the result: every image is still
    transformed independently and probabilities are still accumulated
    per original index -- it only changes how many images go through the
    model per forward call, so it doesn't leave the GPU idle between
    single-image calls. Passing batch_size=1 reproduces the exact
    previous behavior.
    """
    model.eval()
    n = len(dataset)
    accumulated = np.zeros((n, len(CLASSES)), dtype=np.float64)
    all_labels  = np.zeros(n, dtype=np.int64)

    for pass_idx in range(n_passes):
        transform = (base_transform(img_size) if pass_idx == 0
                     else tta_transform(img_size))
        pass_dataset = _TTAPassDataset(dataset, transform)
        loader = DataLoader(
            pass_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        print(f"  TTA pass {pass_idx + 1}/{n_passes}...", end="\r")

        offset = 0
        for batch in loader:
            if use_meta:
                imgs, meta, labels = batch
                meta_t = meta.to(device)
            else:
                imgs, labels = batch
                meta_t = None

            imgs   = imgs.to(device)
            logits = model(imgs, meta_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

            bsz = probs.shape[0]
            accumulated[offset:offset + bsz] += probs
            # Labels are identical every pass -- only need to record them once.
            if pass_idx == 0:
                all_labels[offset:offset + bsz] = labels.numpy()
            offset += bsz

    print()
    avg_probs  = accumulated / n_passes
    all_preds  = avg_probs.argmax(axis=1)
    return all_labels, all_preds, avg_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tta_passes", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=None,
                    help="Forward-pass batch size (default: cfg train.batch_size)")
    p.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader workers (default: cfg data.num_workers)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_meta    = cfg["model"].get("metadata_dim", 0) > 0
    img_size    = cfg["data"].get("img_size", 224)
    experiment  = cfg["logging"]["experiment_name"]
    batch_size  = (args.batch_size if args.batch_size is not None
                   else cfg["train"].get("batch_size", 32))
    num_workers = (args.num_workers if args.num_workers is not None
                   else cfg["data"].get("num_workers", 0))

    print(f"\nExperiment  : {experiment}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"TTA passes  : {args.tta_passes}")
    print(f"Batch size  : {batch_size}")
    print(f"Metadata    : {use_meta}")
    print(f"Device      : {device}\n")

    encoder = None
    if use_meta:
        encoder = MetadataEncoder(
            os.path.join(cfg["data"]["data_dir"], "HAM10000_split.csv")
        )

    model = build_model(cfg).to(device)
    ckpt = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False
)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch {ckpt['epoch']}, "
          f"val_bal_acc={ckpt['val_balanced_accuracy']:.4f}\n")

    dataset = TestDataset(cfg["data"]["data_dir"], encoder)
    y_true, y_pred, y_probs = run_tta(
        model, dataset, device, img_size,
        n_passes=args.tta_passes, use_meta=use_meta,
        batch_size=batch_size, num_workers=num_workers,
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
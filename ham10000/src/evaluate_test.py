import os
import sys
import json
import argparse
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

for p in (_PROJECT_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import HAM10000Dataset
from metadata_encoder import MetadataEncoder
from model import DermaNet

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


@torch.no_grad()
def run_inference(model, loader, device, use_metadata):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in loader:

        if use_metadata:
            images, meta, labels = batch
            meta = meta.to(device)
        else:
            images, labels = batch
            meta = None

        images = images.to(device)

        logits = model(images, meta)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="ham10000/configs/baseline.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="ham10000/checkpoints/best_model.pt"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_metadata = cfg["model"]["metadata_dim"] > 0

    encoder = None

    if use_metadata:
        encoder = MetadataEncoder(
            os.path.join(
                cfg["data"]["data_dir"],
                "HAM10000_split.csv"
            )
        )

    test_dataset = HAM10000Dataset(
        data_dir=cfg["data"]["data_dir"],
        split="test",
        metadata_encoder=encoder,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = DermaNet(
        num_classes=cfg["model"]["num_classes"],
        metadata_dim=cfg["model"]["metadata_dim"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nRunning inference...\n")

    y_true, y_pred, y_probs = run_inference(
        model,
        test_loader,
        device,
        use_metadata,
    )

    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_probs,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc = float("nan")

    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Macro ROC-AUC     : {auc:.4f}")

    print("\nPer-class F1")

    for c, s in zip(CLASSES, per_class_f1):
        print(f"{c:6s}: {s:.4f}")

    print("\nConfusion Matrix")

    print(cm)

    print("\nClassification Report\n")

    print(report)

    os.makedirs("ham10000/results", exist_ok=True)

    with open(
        "ham10000/results/classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    results = {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(auc),
        "per_class_f1": {
            c: float(s)
            for c, s in zip(CLASSES, per_class_f1)
        },
    }

    with open(
        "ham10000/results/baseline_image_only.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    print("\nSaved:")
    print("ham10000/results/classification_report.txt")
    print("ham10000/results/baseline_image_only.json")


if __name__ == "__main__":
    main()
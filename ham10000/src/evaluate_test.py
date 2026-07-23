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

CLASSES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']


@torch.no_grad()
def run_inference(model, loader, device, use_metadata):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        if use_metadata:
            images, meta, labels = batch
            meta = meta.to(device)
        else:
            images, labels = batch
            meta = None

        images = images.to(device)
        logits = model(images, meta)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a DermaNet checkpoint on any split."
    )
    parser.add_argument("--config",
                        default="ham10000/configs/baseline.yaml",
                        help="YAML config used for this experiment")
    parser.add_argument("--checkpoint",
                        default="ham10000/checkpoints/baseline_image_only/best_model.pt",
                        help="Path to the .pt checkpoint to evaluate")
    parser.add_argument("--split",
                        default="test",
                        choices=["train", "val", "test"],
                        help="Which dataset split to evaluate on")
    parser.add_argument("--out",
                        default=None,
                        help="Override output JSON path (auto-inferred if omitted)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_metadata = cfg["model"]["metadata_dim"] > 0
    experiment   = cfg["logging"]["experiment_name"]   # e.g. "baseline_image_only"

    print(f"\nExperiment  : {experiment}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Split       : {args.split}")
    print(f"Metadata    : {use_metadata}")
    print(f"Device      : {device}\n")

    # ── Encoder ───────────────────────────────────────────────
    encoder = None
    if use_metadata:
        encoder = MetadataEncoder(
            os.path.join(cfg["data"]["data_dir"], "HAM10000_split.csv")
        )

    # ── DataLoader ────────────────────────────────────────────
    ds = HAM10000Dataset(
        data_dir=cfg["data"]["data_dir"],
        split=args.split,
        metadata_encoder=encoder,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )
    print(f"Loaded {len(ds)} images for split='{args.split}'")

    # ── Model ─────────────────────────────────────────────────
    model = DermaNet(
        num_classes=cfg["model"]["num_classes"],
        metadata_dim=cfg["model"]["metadata_dim"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint loaded  (epoch {ckpt['epoch']}, "
          f"val_bal_acc={ckpt['val_balanced_accuracy']:.4f})\n")

    # ── Inference ─────────────────────────────────────────────
    print("Running inference...")
    y_true, y_pred, y_probs = run_inference(model, loader, device, use_metadata)

    # ── Metrics ───────────────────────────────────────────────
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None,     zero_division=0)
    cm           = confusion_matrix(y_true, y_pred)
    report       = classification_report(
                       y_true, y_pred,
                       target_names=CLASSES,
                       digits=4, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs,
                            multi_class="ovr", average="macro")
    except Exception as e:
        auc = float("nan")
        print(f"  AUC warning: {e}")

    # ── Print ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS  —  {experiment}  —  split={args.split}")
    print("=" * 60)
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Macro ROC-AUC     : {auc:.4f}")
    print("\nPer-class F1:")
    for c, s in zip(CLASSES, per_class_f1):
        bar = "█" * int(s * 20)
        print(f"  {c:6s}: {s:.4f}  {bar}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # ── Save ─────────────────────────────────────────────────
    os.makedirs("ham10000/results", exist_ok=True)

    # Auto-infer output JSON name from experiment name
    if args.out:
        json_path   = args.out
        report_path = args.out.replace(".json", "_report.txt")
    else:
        json_path   = f"ham10000/results/{experiment}.json"
        report_path = f"ham10000/results/{experiment}_report.txt"

    results = {
        "experiment":      experiment,
        "checkpoint":      args.checkpoint,
        "split":           args.split,
        "epoch":           int(ckpt["epoch"]),
        "balanced_accuracy": float(bal_acc),
        "macro_f1":          float(macro_f1),
        "macro_auc":         float(auc),
        "per_class_f1": {
            c: round(float(s), 4)
            for c, s in zip(CLASSES, per_class_f1)
        },
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(report_path, "w") as f:
        f.write(f"Experiment : {experiment}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Split      : {args.split}\n\n")
        f.write(report)

    print(f"\nSaved results → {json_path}")
    print(f"Saved report  → {report_path}")

    # ── Save per-image probabilities (additive only — existing JSON/report
    #    above are unchanged; this is data run_inference already computed
    #    in memory but previously discarded, needed for threshold/calibration
    #    analysis without re-running inference) ──────────────────────────
    probs_path = f"ham10000/results/{experiment}_probs.npz"
    np.savez(
        probs_path,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        classes=np.array(CLASSES),
    )
    print(f"Saved probabilities → {probs_path}  (y_true, y_pred, y_probs, classes)")


if __name__ == "__main__":
    main()
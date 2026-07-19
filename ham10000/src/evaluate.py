import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

CLASSES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']


@torch.no_grad()
def run_inference(model, dataloader, device, use_metadata):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in dataloader:
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
        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs))


def compute_ece(y_true, y_probs, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Guo et al. 2017).

    Bins predictions by confidence (max softmax prob) into `n_bins` equal-width
    bins, then measures the weighted average gap between confidence and actual
    accuracy in each bin:

        ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Unlike raw training loss, ECE is on a fixed 0-1 scale independent of the
    loss function used to train the model, so it is a fair number to report
    alongside balanced accuracy even when the training objective is a
    class-weighted focal loss whose scale is intentionally distorted by
    per-class weighting. A well-calibrated model typically lands well under
    0.05, regardless of what the raw loss value looks like.
    """
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies  = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        bin_acc  = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_metrics(y_true, y_pred, y_probs):
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                             labels=list(range(len(CLASSES))), zero_division=0)
    metrics['per_class_f1'] = dict(zip(CLASSES, per_class_f1.round(4)))
    metrics['macro_f1'] = float(per_class_f1.mean())
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    metrics['ece'] = compute_ece(y_true, y_probs)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average='macro', labels=list(range(len(CLASSES)))
        )
    except ValueError as e:
        metrics['roc_auc_macro'] = float('nan')
        metrics['roc_auc_warning'] = str(e)
    return metrics
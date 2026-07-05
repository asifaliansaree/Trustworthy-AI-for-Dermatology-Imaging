import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


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


def compute_metrics(y_true, y_pred, y_probs):
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                             labels=list(range(len(CLASSES))), zero_division=0)
    metrics['per_class_f1'] = dict(zip(CLASSES, per_class_f1.round(4)))
    metrics['macro_f1'] = float(per_class_f1.mean())
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    try:
        metrics['roc_auc_macro'] = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average='macro', labels=list(range(len(CLASSES)))
        )
    except ValueError as e:
        metrics['roc_auc_macro'] = float('nan')
        metrics['roc_auc_warning'] = str(e)
    return metrics
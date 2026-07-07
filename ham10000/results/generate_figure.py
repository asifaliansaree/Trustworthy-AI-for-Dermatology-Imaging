import os, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)
from ham10000.dataset import HAM10000Dataset
from ham10000.src.model import DermaNet

CLASS_NAMES = ['mel','nv','bcc','akiec','bkl','df','vasc']
CSV  = 'ham10000/data/HAM10000_split.csv'
DIRS = ['ham10000/data/HAM10000_images_part_1',
        'ham10000/data/HAM10000_images_part_2']
CKPT = 'ham10000/checkpoints/best_model.pt'
OUT  = 'ham10000/results/figures'
os.makedirs(OUT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(CKPT, map_location=device)

cfg = ckpt["config"]   # if you need it later

model = DermaNet().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ds = HAM10000Dataset(
    data_dir="ham10000/data",
    split="test"
)
dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

all_probs  = np.concatenate(all_probs,  axis=0)
all_preds  = np.concatenate(all_preds,  axis=0)
all_labels = np.concatenate(all_labels, axis=0)
print(f"Loaded {len(all_labels)} test samples")

cm = confusion_matrix(all_labels, all_preds, normalize='true')

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Fraction of true class', fontsize=11)
ax.set(xticks=np.arange(7), yticks=np.arange(7),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       ylabel='True label', xlabel='Predicted label',
       title='Normalized confusion matrix — ResNet-18 baseline (test set)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for i in range(7):
    for j in range(7):
        col = 'white' if cm[i,j] > 0.5 else 'black'
        ax.text(j, i, f'{cm[i,j]:.2f}',
                ha='center', va='center', fontsize=9, color=col)
fig.tight_layout()
path = os.path.join(OUT, 'confusion_matrix.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {path}")

from sklearn.metrics import f1_score

f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)

colors = []
for f in f1_per:
    if f >= 0.70:
        colors.append('#1D9E75')
    elif f >= 0.50:
        colors.append('#378ADD')
    else:
        colors.append('#E24B4A')

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(CLASS_NAMES, f1_per, color=colors,
              edgecolor='none', width=0.6)
for bar, score in zip(bars, f1_per):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(y=0.5, color='#888780', linestyle='--',
           linewidth=1, alpha=0.7)
ax.set_ylim(0, 1.08)
ax.set_xlabel('Lesion class', fontsize=12)
ax.set_ylabel('F1-score', fontsize=12)
ax.set_title('Per-class F1-score — ResNet-18 baseline (test set)',
             fontsize=13)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#E24B4A', label='F1 < 0.50'),
    Patch(facecolor='#378ADD', label='0.50 ≤ F1 < 0.70'),
    Patch(facecolor='#1D9E75', label='F1 ≥ 0.70'),
], loc='lower right', fontsize=9)
fig.tight_layout()
path = os.path.join(OUT, 'per_class_f1.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {path}")

import pandas as pd

log = pd.read_csv(
    'ham10000/experiments/training_log.csv')

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

ax1.plot(log['epoch'], log['train_loss'],
         color='#E24B4A', linewidth=2, linestyle='--',
         label='train loss')
ax1.plot(log['epoch'], log['val_loss'],
         color='#EF9F27', linewidth=2,
         label='val loss')
ax2.plot(log['epoch'], log['val_bal_acc'],
         color='#378ADD', linewidth=2.5,
         label='val balanced accuracy')

best_ep  = int(log.loc[log['val_bal_acc'].idxmax(), 'epoch'])
best_acc = log['val_bal_acc'].max()
ax2.axvline(x=best_ep, color='#888780',
            linestyle=':', linewidth=1.2, alpha=0.8)
ax2.annotate(
    f'best: {best_acc:.4f}\n(epoch {best_ep})',
    xy=(best_ep, best_acc),
    xytext=(best_ep + 0.8, best_acc - 0.05),
    fontsize=9, color='#378ADD',
    arrowprops=dict(arrowstyle='->', color='#378ADD', lw=1)
)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='#888780')
ax2.set_ylabel('Val balanced accuracy', fontsize=12, color='#378ADD')
ax1.set_title('Training curves — ResNet-18 baseline (image-only)',
              fontsize=13)
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2,
           loc='center right', fontsize=10)
fig.tight_layout()
path = os.path.join(OUT, 'training_curves.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {path}")
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'src')
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..')
for p in [_SRC, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from model import DermaNet
from dataset import HAM10000Dataset

CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']
OUT     = 'ham10000/results/figures'
os.makedirs(OUT, exist_ok=True)

CFG_PATH  = 'ham10000/configs/baseline.yaml'
CKPT_PATH = 'ham10000/checkpoints/baseline_image_only/best_model.pt'
CSV_PATH  = 'ham10000/data/HAM10000_split.csv'
IMG_DIRS  = [
    'ham10000/data/HAM10000_images_part_1',
    'ham10000/data/HAM10000_images_part_2'
]

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = DermaNet(
    num_classes=7, metadata_dim=0,
    pretrained=False, dropout=0.3
).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

ds = HAM10000Dataset(
    data_dir='ham10000/data', split='test')
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels, all_probs, all_idx = [], [], [], []
idx = 0
with torch.no_grad():
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_idx.extend(range(idx, idx + len(labels)))
        idx += len(labels)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

wrong_mask = all_preds != all_labels
wrong_idx  = np.where(wrong_mask)[0]
confidence = all_probs[wrong_idx, all_preds[wrong_idx]]

print(f"Total test: {len(all_labels)}")
print(f"Wrong predictions: {wrong_mask.sum()} "
      f"({wrong_mask.mean()*100:.1f}%)")

top_wrong = wrong_idx[np.argsort(confidence)[::-1]][:20]
print(f"\nTop 20 high-confidence wrong predictions:")
print(f"{'idx':>5} {'true':>8} {'pred':>8} {'conf':>8}")
for i in top_wrong:
    print(f"{i:>5} {CLASSES[all_labels[i]]:>8} "
          f"{CLASSES[all_preds[i]]:>8} "
          f"{all_probs[i, all_preds[i]]:>8.3f}")

# ── mel→nv failure gallery ────────────────────────────────────
df_split = pd.read_csv(CSV_PATH)
test_df  = df_split[df_split['split'] == 'test'].reset_index(drop=True)

mel_idx  = np.where((all_labels == CLASSES.index('mel')) &
                    (all_preds  == CLASSES.index('nv')))[0]
print(f"\nMelanoma predicted as nevus: {len(mel_idx)} cases")

def find_image(image_id):
    for d in IMG_DIRS:
        p = os.path.join(d, image_id + '.jpg')
        if os.path.exists(p):
            return p
    return None

n_show = min(8, len(mel_idx))
if n_show > 0:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, idx in enumerate(mel_idx[:n_show]):
        row     = test_df.iloc[idx]
        img_p   = find_image(row['image_id'])
        conf    = all_probs[idx, CLASSES.index('nv')]
        if img_p:
            img = Image.open(img_p).convert('RGB')
            axes[i].imshow(img)
        axes[i].set_title(
            f"TRUE: mel\nPRED: nv ({conf:.2f})",
            fontsize=9, color='#e24b4a')
        axes[i].axis('off')
    for j in range(n_show, 8):
        axes[j].axis('off')
    fig.suptitle(
        f'Failure gallery: melanoma predicted as nevus '
        f'({len(mel_idx)} cases)',
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, 'failure_mel_as_nv.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")

# ── Error distribution ─────────────────────────────────────────
print("\n=== ERROR DISTRIBUTION BY CLASS ===")
for c_idx, cname in enumerate(CLASSES):
    total  = (all_labels == c_idx).sum()
    errors = ((all_labels == c_idx) & (all_preds != c_idx)).sum()
    if total > 0:
        print(f"  {cname:6s}: {errors}/{total} wrong "
              f"({errors/total*100:.1f}%)")
        
CONF_THRESHOLD = 0.80

high_conf_wrong = wrong_idx[confidence >= CONF_THRESHOLD]
print(f"\n=== HIGH-CONFIDENCE ERRORS (conf >= {CONF_THRESHOLD}) ===")
print(f"Count: {len(high_conf_wrong)} of {wrong_mask.sum()} wrong "
      f"({len(high_conf_wrong)/wrong_mask.sum()*100:.1f}%)")

if len(high_conf_wrong) > 0:
    n_show = min(6, len(high_conf_wrong))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    sorted_hc = high_conf_wrong[
        np.argsort(all_probs[high_conf_wrong,
                             all_preds[high_conf_wrong]])[::-1]
    ]
    for i, idx in enumerate(sorted_hc[:n_show]):
        row   = test_df.iloc[idx]
        img_p = find_image(row['image_id'])
        conf  = all_probs[idx, all_preds[idx]]
        true_c = CLASSES[all_labels[idx]]
        pred_c = CLASSES[all_preds[idx]]
        if img_p:
            img = Image.open(img_p).convert('RGB')
            axes[i].imshow(img)
        axes[i].set_title(
            f"TRUE: {true_c}\nPRED: {pred_c} ({conf:.2f})",
            fontsize=9, color='#e24b4a', fontweight='bold')
        axes[i].axis('off')
    for j in range(n_show, 6):
        axes[j].axis('off')
    fig.suptitle(
        f'High-confidence wrong predictions (conf ≥ {CONF_THRESHOLD})',
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, 'high_confidence_errors.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")
    print("\nThese are your Week 8 XAI audit targets.")
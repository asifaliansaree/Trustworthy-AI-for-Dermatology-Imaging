"""
Week 4 target selection - fixes two gaps in xai_target_cases.json:

1. That file is failures-only. Grad-CAM needs a "correct vs incorrect"
   comparison, which doesn't exist anywhere yet.
2. gradcam.py / compare_methods.py take targets[:20] / targets[:8] off
   the front of that list. The list is in raw test-set order (not
   shuffled, not balanced), so the first 20 entries are all 'bkl' -
   none of the mel->nv cases flagged in week3.md as "the most frequent
   error, clinically most dangerous" have ever been run through Grad-CAM.

Run from the repo root:
    python ham10000/results/select_week4_targets.py

Produces two new files:
    ham10000/results/xai_targets_incorrect_melnv.json
    ham10000/results/xai_targets_correct.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
from torch.utils.data import DataLoader

_SRC  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for p in [_SRC, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from model import DermaNet
from dataset import HAM10000Dataset

CLASSES   = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CFG_PATH  = 'ham10000/configs/baseline.yaml'
CKPT_PATH = 'ham10000/checkpoints/baseline_image_only/best_model_remapped.pt'
CSV_PATH  = 'ham10000/data/HAM10000_split.csv'

N_CORRECT_PER_CLASS = 3   # -> up to 21 correct examples, same order of
                           # magnitude as the 16 mel->nv failures

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DermaNet(
    num_classes=7, metadata_dim=0,
    pretrained=False, dropout=cfg['model']['dropout'],
).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
print("Loaded checkpoint from:", CKPT_PATH)
print("First 5 state_dict keys:", list(ckpt['model_state_dict'].keys())[:5])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

ds     = HAM10000Dataset(data_dir='ham10000/data', split='test')
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

df_split = pd.read_csv(CSV_PATH)
test_df  = df_split[df_split['split'] == 'test'].reset_index(drop=True)

# -- Part A: the mel->nv failures - the ones week3.md actually cares about --
mel_idx  = np.where((all_labels == CLASSES.index('mel')) &
                     (all_preds  == CLASSES.index('nv')))[0]
mel_conf = all_probs[mel_idx, CLASSES.index('nv')]
mel_idx  = mel_idx[np.argsort(mel_conf)[::-1]]  # most confidently wrong first

incorrect_targets = [{
    'image_id':   test_df.iloc[i]['image_id'],
    'true_label': 'mel',
    'pred_label': 'nv',
    'confidence': float(all_probs[i, CLASSES.index('nv')]),
} for i in mel_idx]

print(f"mel->nv failures found: {len(incorrect_targets)}")

# -- Part B: correctly classified cases, spread across all 7 classes --
correct_targets = []
for c_idx, cname in enumerate(CLASSES):
    right = np.where((all_labels == c_idx) & (all_preds == c_idx))[0]
    if len(right) == 0:
        print(f"  (no correct '{cname}' predictions in test set)")
        continue
    conf = all_probs[right, c_idx]
    top  = right[np.argsort(conf)[::-1]][:N_CORRECT_PER_CLASS]
    for i in top:
        correct_targets.append({
            'image_id':   test_df.iloc[i]['image_id'],
            'true_label': cname,
            'pred_label': cname,
            'confidence': float(all_probs[i, c_idx]),
        })

print(f"correct-case examples selected: {len(correct_targets)}")

out_dir = 'ham10000/results'
with open(os.path.join(out_dir, 'xai_targets_incorrect_melnv.json'), 'w') as f:
    json.dump(incorrect_targets, f, indent=2)
with open(os.path.join(out_dir, 'xai_targets_correct.json'), 'w') as f:
    json.dump(correct_targets, f, indent=2)

print("\nSaved:")
print(f"  {out_dir}/xai_targets_incorrect_melnv.json")
print(f"  {out_dir}/xai_targets_correct.json")
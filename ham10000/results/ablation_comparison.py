import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT      = 'ham10000/results/figures'
BASELINE = 'ham10000/results/baseline_image_only.json'
FUSION   = 'ham10000/results/metadata_fusion.json'
os.makedirs(OUT, exist_ok=True)

# ── Guard ─────────────────────────────────────────────────────
for path in [BASELINE, FUSION]:
    if not os.path.exists(path):
        print(f"NOT READY: {path} does not exist."); exit(0)
    if os.path.getsize(path) == 0:
        print(f"NOT READY: {path} is empty."); exit(0)

with open(BASELINE) as f: baseline = json.load(f)
with open(FUSION)   as f: fusion   = json.load(f)

print("Both result files loaded. Generating comparison figure...")

# ── Summary bar chart ─────────────────────────────────────────
metrics = ['balanced_accuracy', 'macro_f1', 'macro_auc']
labels  = ['Balanced acc',      'Macro F1', 'Macro AUC']

b_vals = [baseline[m] for m in metrics]
f_vals = [fusion[m]   for m in metrics]

x = np.arange(len(metrics))
w = 0.32

fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - w/2, b_vals, w,
               label='Image-only (ResNet-18)',
               color='#2a78d6', alpha=0.9)
bars2 = ax.bar(x + w/2, f_vals, w,
               label='Image + metadata (late fusion)',
               color='#1baf7a', alpha=0.9)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.004,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Ablation: image-only vs image + metadata (test set)',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', linewidth=0.5, alpha=0.5)
fig.tight_layout()

path1 = os.path.join(OUT, 'ablation_comparison.png')
fig.savefig(path1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved → {path1}")

# ── Per-class F1 comparison ───────────────────────────────────
classes   = ['akiec','bcc','bkl','df','mel','nv','vasc']
b_f1 = [baseline['per_class_f1'][c] for c in classes]
f_f1 = [fusion['per_class_f1'][c]   for c in classes]

x2 = np.arange(len(classes))
fig2, ax2 = plt.subplots(figsize=(10, 4))
bars3 = ax2.bar(x2 - w/2, b_f1, w,
                label='Image-only baseline',
                color='#2a78d6', alpha=0.9)
bars4 = ax2.bar(x2 + w/2, f_f1, w,
                label='Image + metadata',
                color='#1baf7a', alpha=0.9)

for bar in list(bars3) + list(bars4):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=8)

ax2.axhline(y=0.5, color='#888780', linestyle='--',
            linewidth=1, alpha=0.7, label='F1 = 0.50')
ax2.set_xticks(x2)
ax2.set_xticklabels(classes, fontsize=11)
ax2.set_ylim(0, 1.08)
ax2.set_ylabel('F1-score', fontsize=12)
ax2.set_title('Per-class F1: image-only vs metadata fusion (test set)',
              fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis='y', linewidth=0.5, alpha=0.5)
fig2.tight_layout()

path2 = os.path.join(OUT, 'ablation_per_class_f1.png')
fig2.savefig(path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved → {path2}")

# ── Delta table ───────────────────────────────────────────────
print("\n=== ABLATION DELTA TABLE ===")
print(f"{'Metric':<22} {'Baseline':>10} {'Fusion':>10} {'Delta':>10}")
print("-" * 55)
for label, bv, fv in zip(labels, b_vals, f_vals):
    delta = fv - bv
    sign  = '+' if delta >= 0 else ''
    print(f"{label:<22} {bv:>10.4f} {fv:>10.4f} {sign+f'{delta:.4f}':>10}")
print("\nPer-class F1 deltas:")
for c, bv, fv in zip(classes, b_f1, f_f1):
    delta = fv - bv
    sign  = '+' if delta >= 0 else ''
    flag  = ' ← big gain' if delta > 0.05 else (' ← big loss' if delta < -0.05 else '')
    print(f"  {c:<8} {bv:.4f} → {fv:.4f}  ({sign}{delta:.4f}){flag}")

print(f"\nAll figures saved to {OUT}/")
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = 'ham10000/results/figures'
os.makedirs(OUT, exist_ok=True)

baseline = pd.read_csv(
    'ham10000/experiments/baseline_image_only/training_log.csv')
fusion = pd.read_csv(
    'ham10000/experiments/metadata_fusion/training_log.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# ── Left: val balanced accuracy ──────────────────────────────
ax1.plot(baseline['epoch'], baseline['val_bal_acc'],
         color='#2a78d6', linewidth=2, label='image-only')
ax1.plot(fusion['epoch'],   fusion['val_bal_acc'],
         color='#1baf7a', linewidth=2, label='+ metadata')

b_best = baseline.loc[baseline['val_bal_acc'].idxmax()]
f_best = fusion.loc[fusion['val_bal_acc'].idxmax()]

ax1.axvline(b_best['epoch'], color='#2a78d6',
            linestyle=':', linewidth=1, alpha=0.7)
ax1.axvline(f_best['epoch'], color='#1baf7a',
            linestyle=':', linewidth=1, alpha=0.7)
ax1.annotate(f"{b_best['val_bal_acc']:.4f}",
             xy=(b_best['epoch'], b_best['val_bal_acc']),
             xytext=(b_best['epoch']+0.5, b_best['val_bal_acc']-0.03),
             fontsize=8, color='#2a78d6')
ax1.annotate(f"{f_best['val_bal_acc']:.4f}",
             xy=(f_best['epoch'], f_best['val_bal_acc']),
             xytext=(f_best['epoch']+0.5, f_best['val_bal_acc']-0.03),
             fontsize=8, color='#1baf7a')

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Val balanced accuracy', fontsize=11)
ax1.set_title('Validation balanced accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim(0.55, 0.85)
ax1.grid(linewidth=0.4, alpha=0.5)

# ── Right: train + val loss ───────────────────────────────────
ax2.plot(baseline['epoch'], baseline['train_loss'],
         color='#2a78d6', linewidth=2, linestyle='--',
         label='baseline train')
ax2.plot(baseline['epoch'], baseline['val_loss'],
         color='#2a78d6', linewidth=2,
         label='baseline val')
ax2.plot(fusion['epoch'], fusion['train_loss'],
         color='#1baf7a', linewidth=2, linestyle='--',
         label='fusion train')
ax2.plot(fusion['epoch'], fusion['val_loss'],
         color='#1baf7a', linewidth=2,
         label='fusion val')

ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Train vs validation loss', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(linewidth=0.4, alpha=0.5)

fig.suptitle('Learning curves — baseline vs metadata fusion',
             fontsize=13, y=1.01)
fig.tight_layout()
path = os.path.join(OUT, 'learning_curves_comparison.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved → {path}")

# ── Overfitting diagnostic ────────────────────────────────────
print("\n=== OVERFITTING DIAGNOSTIC ===")
for name, df in [('Baseline', baseline), ('Fusion', fusion)]:
    final_train = df['train_loss'].iloc[-1]
    final_val   = df['val_loss'].iloc[-1]
    gap         = final_val - final_train
    verdict = "mild overfit" if gap > 0.3 else "healthy"
    print(f"{name}: train={final_train:.4f} | val={final_val:.4f} "
          f"| gap={gap:.4f} → {verdict}")
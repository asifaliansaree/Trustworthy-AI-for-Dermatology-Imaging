"""
lr_finder.py — Find optimal learning rate before training.
Increases LR exponentially over 100 steps, plots loss vs LR.
Optimal LR is ~1/10th of the point where loss explodes.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, 'ham10000/src')
sys.path.insert(0, 'ham10000')

from dataset import HAM10000Dataset
from model   import DermaNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds     = HAM10000Dataset('ham10000/data', 'train')
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

model     = DermaNet(num_classes=7, metadata_dim=0,
                     pretrained=True, dropout=0.3).to(device)
weights   = torch.tensor(
    np.load('ham10000/data/class_weights.npy'),
    dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

# LR range test config
num_steps = 100
start_lr  = 1e-7
end_lr    = 1e-1
mult      = (end_lr / start_lr) ** (1 / num_steps)

lrs, losses, smooth_loss = [], [], 0.0
beta = 0.98

model.train()
data_iter = iter(loader)

print("Running LR range test...")
for step in range(num_steps):
    try:
        imgs, labels = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        imgs, labels = next(data_iter)

    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()
    loss = criterion(model(imgs), labels)
    loss.backward()
    optimizer.step()

    # Smooth the loss
    smooth_loss = beta * smooth_loss + (1 - beta) * loss.item()
    debias_loss = smooth_loss / (1 - beta ** (step + 1))

    lr = start_lr * (mult ** step)
    lrs.append(lr)
    losses.append(debias_loss)

    # Update LR for next step
    for g in optimizer.param_groups:
        g['lr'] = lr * mult

    if step % 10 == 0:
        print(f"  Step {step:3d} | lr={lr:.2e} | loss={debias_loss:.4f}")

    if debias_loss > 4 * min(losses):  # stop if loss explodes
        print(f"  Loss exploded at step {step}, lr={lr:.2e}")
        break

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lrs, losses, color='#2a78d6', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Learning rate (log scale)', fontsize=12)
ax.set_ylabel('Smoothed loss', fontsize=12)
ax.set_title('LR range test — pick LR at steepest descent', fontsize=13)

# Mark the steepest descent point
losses_arr = np.array(losses)
lrs_arr    = np.array(lrs)

# Ignore the first ~10 steps (LR too tiny to be real learning, just noise)
# and the last ~5 steps (often mid-explosion, also noisy)
skip_start, skip_end = 10, 5
valid_lrs    = lrs_arr[skip_start: len(lrs_arr) - skip_end]
valid_losses = losses_arr[skip_start: len(losses_arr) - skip_end]

# Smooth the gradient itself so one noisy jump can't fake-win
raw_grad    = np.gradient(valid_losses)
window      = 5
smooth_grad = np.convolve(raw_grad, np.ones(window) / window, mode='valid')

best_i  = np.argmin(smooth_grad) + (window // 2)
best_lr = valid_lrs[best_i]

ax.axvline(best_lr, color='#e34948', linestyle='--',
           linewidth=1.5, label=f'Steepest descent: {best_lr:.2e}')
ax.axvline(best_lr / 10, color='#1baf7a', linestyle='--',
           linewidth=1.5, label=f'Recommended LR: {best_lr/10:.2e}')
ax.legend(fontsize=10)
ax.grid(linewidth=0.4, alpha=0.5)
fig.tight_layout()

os.makedirs('ham10000/results/figures', exist_ok=True)
fig.savefig('ham10000/results/figures/lr_range_test.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSteepest descent at LR: {best_lr:.2e}")
print(f"Recommended LR (1/10th): {best_lr/10:.2e}")
print("Saved: ham10000/results/figures/lr_range_test.png")
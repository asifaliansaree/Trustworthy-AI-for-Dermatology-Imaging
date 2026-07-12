# ham10000/results/generate_all_figures.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "ham10000/results/figures"
os.makedirs(OUT, exist_ok=True)

# ── All training logs ─────────────────────────────────────────

logs = {}

# ResNet-18 baseline — from your training_log.csv
baseline_csv = "ham10000/experiments/baseline_image_only/training_log.csv"
if os.path.exists(baseline_csv):
    logs["ResNet-18 baseline"] = pd.read_csv(baseline_csv)
    print(f"✓ Loaded ResNet-18 from CSV")
else:
    logs["ResNet-18 baseline"] = pd.DataFrame({
        "epoch":       list(range(1, 21)),
        "train_loss":  [1.2297,0.9080,0.7725,0.7301,0.6584,0.5955,
                        0.5583,0.5146,0.4565,0.4230,0.4007,0.3679,
                        0.3119,0.2905,0.2819,0.2478,0.2406,0.2236,
                        0.2179,0.2238],
        "val_loss":    [0.9273,0.9599,0.8957,0.7727,0.7851,0.8813,
                        0.7096,0.7006,0.7027,0.6521,0.6466,0.6169,
                        0.6048,0.6158,0.6083,0.6050,0.6350,0.6180,
                        0.5847,0.6452],
        "val_bal_acc": [0.6486,0.6882,0.6880,0.7617,0.7087,0.7156,
                        0.7110,0.7337,0.7460,0.7798,0.7561,0.7587,
                        0.7848,0.7714,0.7964,0.7758,0.7705,0.7882,
                        0.7902,0.7843],
    })
    print("✓ ResNet-18 reconstructed from known values")

# EfficientNet-B0 v2
logs["EfficientNet-B0 v2\n(lr=1e-4, label smoothing)"] = pd.DataFrame({
    "epoch":       list(range(1, 26)),
    "train_loss":  [1.3154,1.0644,1.0106,0.9804,0.9540,0.9310,0.9087,
                    0.8945,0.8867,0.8743,0.8588,0.8555,0.8435,0.8255,
                    0.8208,0.8062,0.7973,0.7895,0.7922,0.7782,0.7814,
                    0.7769,0.7719,0.7714,0.7781],
    "val_loss":    [1.1768,1.1160,1.1330,1.0739,1.0576,1.0616,1.0299,
                    1.0722,1.0611,1.0443,1.0547,1.0377,1.0295,1.0343,
                    1.0310,1.0332,1.0395,1.0321,1.0353,1.0425,1.0457,
                    1.0411,1.0425,1.0311,1.0327],
    "val_bal_acc": [0.4478,0.5128,0.5357,0.5598,0.6140,0.6064,0.6352,
                    0.6202,0.6145,0.6336,0.6179,0.6370,0.6347,0.6308,
                    0.6287,0.6431,0.6336,0.6630,0.6536,0.6378,0.6324,
                    0.6423,0.6338,0.6359,0.6535],
})
print("✓ EfficientNet-B0 v2 reconstructed")

# EfficientNet-B0 v3
logs["EfficientNet-B0 v3\n(lr=2e-4, cross entropy)"] = pd.DataFrame({
    "epoch":       list(range(1, 31)),
    "train_loss":  [0.8135,0.5623,0.4891,0.4437,0.4012,0.3720,0.3466,
                    0.3114,0.2804,0.2641,0.2497,0.2230,0.1973,0.1867,
                    0.1663,0.1572,0.1368,0.1144,0.1269,0.1076,0.0990,
                    0.0862,0.0807,0.0790,0.0746,0.0616,0.0695,0.0575,
                    0.0660,0.0571],
    "val_loss":    [0.6464,0.5986,0.5845,0.5784,0.5727,0.4968,0.5198,
                    0.5126,0.5180,0.5331,0.5602,0.6318,0.5547,0.5996,
                    0.6241,0.6514,0.6311,0.6244,0.6356,0.7199,0.7403,
                    0.7325,0.7293,0.7404,0.7777,0.7471,0.7570,0.7531,
                    0.7457,0.7466],
    "val_bal_acc": [0.5121,0.5815,0.5897,0.5843,0.6068,0.6357,0.6627,
                    0.6623,0.6617,0.6834,0.6758,0.6765,0.6937,0.6754,
                    0.7058,0.6991,0.6851,0.7165,0.6992,0.6823,0.7026,
                    0.6999,0.7050,0.6925,0.7149,0.7088,0.7024,0.7095,
                    0.7035,0.7068],
})
print("✓ EfficientNet-B0 v3 reconstructed")

# Also save CSVs for the two EfficientNet experiments
os.makedirs("ham10000/experiments/efficientnet_b0_v2", exist_ok=True)
os.makedirs("ham10000/experiments/efficientnet_b0_v3", exist_ok=True)
list(logs.values())[1].to_csv(
    "ham10000/experiments/efficientnet_b0_v2/training_log.csv", index=False)
list(logs.values())[2].to_csv(
    "ham10000/experiments/efficientnet_b0_v3/training_log.csv", index=False)
print("✓ EfficientNet training CSVs saved")

COLORS = {
    0: "#2a78d6",
    1: "#1baf7a",
    2: "#e34948",
}
names  = list(logs.keys())
frames = list(logs.values())

# ── Figure 1: Val balanced accuracy ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, df) in enumerate(logs.items()):
    ax.plot(df["epoch"], df["val_bal_acc"],
            color=COLORS[i], linewidth=2,
            label=name.replace("\n", " "))
    best_idx = df["val_bal_acc"].idxmax()
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_acc = df.loc[best_idx, "val_bal_acc"]
    ax.scatter(best_ep, best_acc,
               color=COLORS[i], s=80, zorder=5)
    ax.annotate(f"best={best_acc:.4f} (ep{best_ep})",
                xy=(best_ep, best_acc),
                xytext=(best_ep + 0.6, best_acc + 0.015),
                fontsize=8, color=COLORS[i],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS[i], lw=0.8))
ax.axhline(0.65, color="#888780", linestyle=":",
           linewidth=1, alpha=0.7, label="0.65 go/no-go threshold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val balanced accuracy", fontsize=12)
ax.set_title("Validation balanced accuracy — ResNet-18 vs EfficientNet-B0",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(linewidth=0.4, alpha=0.5)
ax.set_ylim(0.10, 0.90)
fig.tight_layout()
fig.savefig(f"{OUT}/model_comparison_bal_acc.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved model_comparison_bal_acc.png")

# ── Figure 2: Loss curves ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, df) in enumerate(logs.items()):
    label = name.replace("\n", " ")
    ax.plot(df["epoch"], df["train_loss"],
            color=COLORS[i], linewidth=1.5,
            linestyle="--", alpha=0.6,
            label=f"{label} — train")
    ax.plot(df["epoch"], df["val_loss"],
            color=COLORS[i], linewidth=2,
            label=f"{label} — val")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Train vs validation loss — all experiments", fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(linewidth=0.4, alpha=0.5)
ax.set_ylim(0, 1.35)
fig.tight_layout()
fig.savefig(f"{OUT}/model_comparison_loss.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved model_comparison_loss.png")

# ── Figure 3: Individual subplots ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (ax, (name, df)) in enumerate(zip(axes, logs.items())):
    ax2 = ax.twinx()
    ax.plot(df["epoch"], df["train_loss"],
            color="#e34948", linewidth=2,
            linestyle="--", label="train loss")
    ax.plot(df["epoch"], df["val_loss"],
            color="#EF9F27", linewidth=2,
            label="val loss")
    ax2.plot(df["epoch"], df["val_bal_acc"],
             color="#2a78d6", linewidth=2.5,
             label="val bal-acc")
    best_acc = df["val_bal_acc"].max()
    best_ep  = int(df.loc[df["val_bal_acc"].idxmax(), "epoch"])
    ax2.axvline(best_ep, color="#888780",
                linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=9, color="#888780")
    ax2.set_ylabel("Bal acc", fontsize=9, color="#2a78d6")
    ax.set_title(f"{name}\nbest={best_acc:.4f} (ep{best_ep})",
                 fontsize=9.5)
    ax.grid(linewidth=0.4, alpha=0.4)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2,
              fontsize=7.5, loc="upper right")
fig.suptitle("Training curves — all experiments",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/all_models_training_curves.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved all_models_training_curves.png")

# ── Figure 4: Overfitting analysis ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, (ax, (name, df)) in enumerate(zip(axes, logs.items())):
    gap = df["val_loss"] - df["train_loss"]
    ax.fill_between(df["epoch"], 0, gap,
                    alpha=0.3, color=COLORS[i])
    ax.plot(df["epoch"], gap,
            color=COLORS[i], linewidth=2)
    ax.axhline(0, color="#888780",
               linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Val loss − Train loss", fontsize=9)
    ax.set_title(f"{name}\n(gap = overfitting indicator)",
                 fontsize=9.5)
    ax.grid(linewidth=0.4, alpha=0.4)
fig.suptitle("Overfitting analysis — train/val loss gap per model",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/overfitting_analysis.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved overfitting_analysis.png")

print(f"\nAll 4 figures saved to {OUT}/")
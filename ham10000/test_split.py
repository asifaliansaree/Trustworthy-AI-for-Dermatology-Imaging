import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Load split CSV
csv_path = os.path.join("ham10000", "data", "HAM10000_split.csv")
df = pd.read_csv(csv_path)

print("===== DATASET SUMMARY =====")
print(df["split"].value_counts())
print()

# ----------------------------
# Leakage check
# ----------------------------
train_lesions = set(df[df["split"] == "train"]["lesion_id"])
val_lesions = set(df[df["split"] == "val"]["lesion_id"])
test_lesions = set(df[df["split"] == "test"]["lesion_id"])

print("===== LEAKAGE CHECK =====")
print("Train ∩ Val :", len(train_lesions & val_lesions))
print("Train ∩ Test:", len(train_lesions & test_lesions))
print("Val ∩ Test  :", len(val_lesions & test_lesions))

assert len(train_lesions & val_lesions) == 0
assert len(train_lesions & test_lesions) == 0
assert len(val_lesions & test_lesions) == 0

print("✅ No lesion leakage detected.\n")

# ----------------------------
# Class weights (train only)
# ----------------------------
# This MUST match the label index used everywhere else in the pipeline
# (dataset.py's CLASS_MAP, train.py's CLASS_MAP) -- mel:0, nv:1, bcc:2,
# akiec:3, bkl:4, df:5, vasc:6. This was previously alphabetical
# (akiec:0, bcc:1, bkl:2, df:3, mel:4, nv:5, vasc:6), which meant every
# weight in the saved class_weights.npy landed on the WRONG class index
# whenever a config used it directly (loss.alpha_mode: inverse, or plain
# cross_entropy/label_smoothing with weight=class_weights). Configs using
# loss.alpha_mode: effective_num (e.g. resnet50_v12recipe.yaml) were NOT
# affected -- that path recomputes its own weights from loss.class_counts
# in the YAML and only reads class_weights.npy for its .device attribute.
#
# IMPORTANT: this code fix alone does not correct any class_weights.npy
# file already saved to disk with the old (wrong) ordering -- rerun this
# script to regenerate it after pulling this fix.
CLASS_MAP = {
    "mel": 0,
    "nv": 1,
    "bcc": 2,
    "akiec": 3,
    "bkl": 4,
    "df": 5,
    "vasc": 6,
}

train_df = df[df["split"] == "train"].copy()
train_df["label"] = train_df["dx"].map(CLASS_MAP)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(7),
    y=train_df["label"]
)

print("===== CLASS WEIGHTS =====")
for name, idx in CLASS_MAP.items():
    print(f"{name:6s}: {weights[idx]:.4f}")

# Save
save_path = os.path.join("ham10000", "data", "class_weights.npy")
np.save(save_path, weights)

print(f"\nSaved class weights to:\n{save_path}")
import os, json
os.makedirs("ham10000/results", exist_ok=True)

# This is what you'll fill in on Day 2 after evaluation
template = {
    "experiment": "baseline_image_only",
    "config": "configs/baseline.yaml",
    "seed": 42,
    "test_balanced_accuracy": None,
    "test_macro_f1": None,
    "test_macro_auc": None,
    "per_class_f1": {
        "mel": None, "nv": None, "bcc": None,
        "akiec": None, "bkl": None, "df": None, "vasc": None
    },
    "best_val_epoch": None,
    "best_val_balanced_acc": None
}

with open("ham10000/results/baseline_image_only.json", "w") as f:
    json.dump(template, f, indent=2)

print("Results template created.")
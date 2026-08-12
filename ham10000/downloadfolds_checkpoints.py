import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "AsifAliAnsaree/ham10000-checkpoints"

CHECKPOINT_NAMES = ["best_model.pt", "last_model.pt"]

for fold in range(5):
    subfolder = f"resnet50_v12recipe_fold{fold}"
    dest_dir = os.path.join("ham10000", "checkpoints", subfolder)
    os.makedirs(dest_dir, exist_ok=True)

    for ckpt_name in CHECKPOINT_NAMES:
        filename = f"{subfolder}/{ckpt_name}"

        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="model",
        )

        dest_path = os.path.join(dest_dir, ckpt_name)
        shutil.copy(local_path, dest_path)
        print(f"  -> saved to {dest_path}")

print("\nAll fold checkpoints (best + last) downloaded.")
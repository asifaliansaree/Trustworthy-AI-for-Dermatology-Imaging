"""
Generate 4-method comparison figures.
One figure per image showing: original | GradCAM | IG | Saliency | Occlusion
This is Figure 5 of the paper.
"""
import os, sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (load_image, get_prediction, find_image,
                   overlay_on_image, normalize_attr,
                   load_model_and_config, load_xai_targets, CLASSES)
from gradcam             import compute_gradcam
from integrated_gradients import compute_ig
from saliency            import compute_saliency
from occlusion           import compute_occlusion

OUT = "ham10000/results/xai/comparison"
os.makedirs(OUT, exist_ok=True)

def compare_one(case, model, device, save_path):
    try:
        img_path = find_image(case['image_id'])
    except FileNotFoundError:
        return

    tensor, _, display = load_image(img_path)
    pred, conf, _      = get_prediction(model, tensor, device)
    true_name = case['true_label']
    pred_name = CLASSES[pred]
    correct   = pred_name == true_name

    print(f"  Computing all 4 methods for {case['image_id']}...",
          end='', flush=True)

    gc  = compute_gradcam(model, tensor, pred, device)
    ig  = compute_ig(model, tensor, pred, device)
    sal = compute_saliency(model, tensor, pred, device)
    occ = compute_occlusion(model, tensor, pred, device)

    print(" done")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    methods = [
        ("Original",             display,              'gray'),
        ("Grad-CAM",             overlay_on_image(display, gc),   None),
        ("Integrated Gradients", overlay_on_image(display, ig),   None),
        ("Vanilla Saliency",     overlay_on_image(display, sal),  None),
        ("Occlusion",            overlay_on_image(display, occ),  None),
    ]

    for ax, (title, img, cmap) in zip(axes, methods):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    status  = "CORRECT" if correct else "WRONG"
    color   = '#1D9E75' if correct else '#E24B4A'
    fig.suptitle(
        f"{case['image_id']} | true={true_name} | "
        f"pred={pred_name} ({conf:.2f}) | {status}",
        fontsize=10, color=color, y=1.01
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model, device, _ = load_model_and_config()
    targets = load_xai_targets()

    print(f"\n=== 4-method comparison (first 8 cases) ===")
    print("Note: each image takes ~3-5 min on CPU due to occlusion")

    for i, case in enumerate(targets[:8]):
        save_path = os.path.join(
            OUT, f"{i:02d}_{case['image_id']}_comparison.png")
        compare_one(case, model, device, save_path)
        print(f"  Saved: {save_path}")

    print(f"\nAll comparison figures saved to {OUT}/")
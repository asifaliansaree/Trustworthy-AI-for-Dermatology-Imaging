"""
Integrated Gradients — Method B of 4.
Captum IntegratedGradients with n_steps=50.
More theoretically grounded than Grad-CAM.
"""
import os, sys, numpy as np, torch
from captum.attr import IntegratedGradients

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (normalize_attr, overlay_on_image, load_image,
                   get_prediction, find_image, save_triple_figure,
                   load_model_and_config, load_xai_targets, CLASSES)

def compute_ig(model, tensor, target_class, device, n_steps=50):
    model.eval()
    ig        = IntegratedGradients(model)
    baseline  = torch.zeros_like(tensor).to(device)
    attr      = ig.attribute(
        tensor.to(device),
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
    )
    hm = attr.squeeze().cpu().detach().numpy()
    hm = np.mean(np.abs(hm), axis=0)
    return normalize_attr(hm)

def run_ig_batch(cases, out_dir, model, device, max_cases=20):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, case in enumerate(cases[:max_cases]):
        try:
            img_path = find_image(case['image_id'])
        except FileNotFoundError:
            continue
        tensor, _, display = load_image(img_path)
        pred, conf, _      = get_prediction(model, tensor, device)
        true_idx = case.get('true_idx', CLASSES.index(case['true_label']))
        heatmap  = compute_ig(model, tensor, pred, device)
        overlay  = overlay_on_image(display, heatmap)
        save_path = os.path.join(
            out_dir,
            f"{i:02d}_{case['image_id']}_ig.png"
        )
        save_triple_figure(
            display, heatmap, overlay,
            title=f"IG | true={case['true_label']} pred={CLASSES[pred]} conf={conf:.3f}",
            pred_name=CLASSES[pred], true_name=case['true_label'],
            conf=conf, save_path=save_path,
            method_name="Integrated Gradients",
        )
        results.append({
            'image_id': case['image_id'],
            'true_class': case['true_label'],
            'pred_class': CLASSES[pred],
            'confidence': conf,
            'correct': pred == true_idx,
        })
        print(f"  [{i+1:02d}] {case['image_id']}: {CLASSES[pred]} ({conf:.3f})")
    return results

if __name__ == "__main__":
    model, device, _ = load_model_and_config()
    targets = load_xai_targets()
    print("\n=== Integrated Gradients on failure cases ===")
    run_ig_batch(targets, "ham10000/results/xai/integrated_gradients",
                 model, device)
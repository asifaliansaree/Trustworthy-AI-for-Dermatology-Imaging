"""
Grad-CAM — Method A of 4 in the XAI benchmark.
Uses Captum's LayerGradCam on ResNet-18 layer4.
"""
import os, sys
import numpy as np
import torch
from captum.attr import LayerGradCam, LayerAttribution

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (normalize_attr, overlay_on_image,
                   load_image, get_prediction, find_image,
                   save_triple_figure, load_model_and_config,
                   load_xai_targets, CLASSES)

def get_target_layer(model):
    """Last conv layer in ResNet-18 = layer4[-1].conv2

    backbone = nn.Sequential(conv1, bn1, relu, maxpool,
                              layer1, layer2, layer3, layer4, avgpool)
    avgpool is index -1, so layer4 is index -2 (NOT -3 - that's layer3).
    """
    return model.backbone[-2][-1].conv2

def compute_gradcam(model, tensor, target_class, device):
    model.eval()
    gc  = LayerGradCam(model, get_target_layer(model))
    att = gc.attribute(tensor.to(device), target=target_class)
    up  = LayerAttribution.interpolate(att, (224,224), 'bilinear')
    hm  = up.squeeze().cpu().detach().numpy()
    return normalize_attr(hm)

def run_gradcam_batch(cases, out_dir, model, device, max_cases=20):
    """Run Grad-CAM on a list of cases."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, case in enumerate(cases[:max_cases]):
        try:
            img_path = find_image(case['image_id'])
        except FileNotFoundError:
            continue
        tensor, _, display = load_image(img_path)
        pred, conf, _ = get_prediction(model, tensor, device)
        true_idx = case.get('true_idx', CLASSES.index(case['true_label']))
        heatmap  = compute_gradcam(model, tensor, pred, device)
        overlay  = overlay_on_image(display, heatmap)
        save_path = os.path.join(
            out_dir,
            f"{i:02d}_{case['image_id']}"
            f"_true{case['true_label']}"
            f"_pred{CLASSES[pred]}.png"
        )
        save_triple_figure(
            display, heatmap, overlay,
            title=(f"true={case['true_label']} | "
                   f"pred={CLASSES[pred]} | conf={conf:.3f}"),
            pred_name=CLASSES[pred],
            true_name=case['true_label'],
            conf=conf,
            save_path=save_path,
            method_name="Grad-CAM",
        )
        results.append({
            'image_id':    case['image_id'],
            'true_class':  case['true_label'],
            'pred_class':  CLASSES[pred],
            'confidence':  conf,
            'correct':     pred == true_idx,
            'heatmap_path': save_path,
        })
        print(f"  [{i+1:02d}] {case['image_id']}: "
              f"true={case['true_label']} pred={CLASSES[pred]} "
              f"conf={conf:.3f} "
              f"{'OK' if pred==true_idx else 'WRONG'}")
    return results

if __name__ == "__main__":
    import json, pandas as pd
    model, device, _ = load_model_and_config()

    with open("ham10000/results/xai_targets_incorrect_melnv.json") as f:
        incorrect_targets = json.load(f)
    print("\n=== Grad-CAM on failure cases (mel->nv) ===")
    fail_results = run_gradcam_batch(
        incorrect_targets, "ham10000/results/xai/gradcam/failures",
        model, device, max_cases=20)

    with open("ham10000/results/xai_targets_correct.json") as f:
        correct_targets = json.load(f)
    print("\n=== Grad-CAM on correct cases ===")
    correct_results = run_gradcam_batch(
        correct_targets, "ham10000/results/xai/gradcam/correct",
        model, device, max_cases=20)

    results_df = pd.DataFrame(fail_results + correct_results)
    results_df.to_csv(
        "ham10000/results/xai/gradcam_results.csv", index=False)
    print(f"\nSaved {len(fail_results)} failure + {len(correct_results)} correct heatmaps")
    print(f"Correct among failures: "
          f"{sum(r['correct'] for r in fail_results)}/{len(fail_results)}")
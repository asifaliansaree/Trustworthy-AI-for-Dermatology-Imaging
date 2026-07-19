"""Occlusion sensitivity — Method D of 4."""
import os, sys, numpy as np, torch
from captum.attr import Occlusion

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (normalize_attr, overlay_on_image, load_image,
                   get_prediction, find_image, save_triple_figure,
                   load_model_and_config, load_xai_targets, CLASSES)

def compute_occlusion(model, tensor, target_class, device,
                      window=16, stride=8):
    """
    Slide a window over the image, measure confidence drop.
    window=16, stride=8 gives a good resolution/speed tradeoff.
    Larger window = coarser but faster.
    """
    model.eval()
    occ  = Occlusion(model)
    attr = occ.attribute(
        tensor.to(device),
        strides=(3, stride, stride),
        target=target_class,
        sliding_window_shapes=(3, window, window),
        baselines=0,
    )
    hm = attr.squeeze().cpu().detach().numpy()
    hm = np.mean(np.abs(hm), axis=0)
    return normalize_attr(hm)

def run_occlusion_batch(cases, out_dir, model, device, max_cases=20):
    os.makedirs(out_dir, exist_ok=True)
    for i, case in enumerate(cases[:max_cases]):
        try:
            img_path = find_image(case['image_id'])
        except FileNotFoundError:
            continue
        tensor, _, display = load_image(img_path)
        pred, conf, _ = get_prediction(model, tensor, device)
        print(f"  [{i+1:02d}] Computing occlusion for {case['image_id']}...",
              end='', flush=True)
        heatmap = compute_occlusion(model, tensor, pred, device)
        overlay = overlay_on_image(display, heatmap)
        save_triple_figure(
            display, heatmap, overlay,
            title=f"Occlusion | true={case['true_label']} pred={CLASSES[pred]} conf={conf:.3f}",
            pred_name=CLASSES[pred], true_name=case['true_label'],
            conf=conf,
            save_path=os.path.join(out_dir, f"{i:02d}_{case['image_id']}_occ.png"),
            method_name="Occlusion",
        )
        print(" done")

if __name__ == "__main__":
    model, device, _ = load_model_and_config()
    targets = load_xai_targets()
    print("\n=== Occlusion on failure cases ===")
    print("Note: occlusion is slow on CPU (~2-3 min per image)")
    run_occlusion_batch(targets, "ham10000/results/xai/occlusion",
                        model, device, max_cases=10)
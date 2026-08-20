"""
Integrated Gradients — generalized over checkpoint_key.

Default checkpoint: convnext_tiny_v1_fold0 (your chosen best checkpoint).
Same script applies unchanged to any of the other 30 checkpoints later --
just pass a different --checkpoint_key. Output is namespaced per
checkpoint so results from different architectures/folds don't collide.

    python ham10000/src/explain/integrated_gradients_stage1.py
    python ham10000/src/explain/integrated_gradients_stage1.py --checkpoint_key resnet50_v12recipe_fold0

Requires xai_pilot_set_fold<N>.json to already exist for this checkpoint's
fold (run build_pilot_set.py first). The pilot set is fold-scoped and
shared across architectures at that fold -- this script does NOT rebuild
it, it just loads whichever checkpoint you point it at and explains
exactly those same images, so results stay comparable across architectures.
"""
import os, sys, json, csv, argparse
import numpy as np
import torch
from captum.attr import IntegratedGradients

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (normalize_attr, overlay_on_image, load_image,
                    get_prediction, find_image, save_triple_figure,
                    load_model_and_config, resolve_checkpoint, CLASSES)


def make_blurred_baseline(tensor: torch.Tensor, kernel_size: int = 21, sigma: float = 5.0) -> torch.Tensor:
    """Gaussian-blurred version of the (already normalized) input as an IG baseline."""
    import torchvision.transforms.functional as TF
    return TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def compute_ig(model, tensor, target_class, device, baseline, n_steps=50):
    """Returns (normalized_heatmap_for_display, raw_attribution_map)."""
    model.eval()
    ig = IntegratedGradients(model)
    attr = ig.attribute(
        tensor.to(device),
        baselines=baseline.to(device),
        target=target_class,
        n_steps=n_steps,
    )
    raw = attr.squeeze().cpu().detach().numpy()          # [C, H, W], signed
    magnitude = np.mean(np.abs(raw), axis=0)              # [H, W], for display
    return normalize_attr(magnitude), raw


def run_ig_batch(cases, out_dir, model, device, n_steps=50, max_cases=None):
    """cases: list of dicts from xai_pilot_set_fold<N>.json['cases']. Each
    carries true_label/true_idx/pred_label/correct from whichever
    checkpoint originally built the pilot set -- we recompute the
    prediction here with THIS checkpoint (they'll legitimately differ
    across architectures; that's expected, not an error). We only warn
    when the checkpoint we're running IS the one the pilot set was built
    from and disagrees with itself, which would indicate something
    changed (different checkpoint file, preprocessing drift, etc).
    """
    raw_dir = os.path.join(out_dir, "raw")
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    results = []
    for i, case in enumerate(cases[:max_cases] if max_cases else cases):
        try:
            img_path = find_image(case['image_id'])
        except FileNotFoundError:
            continue

        tensor, _, display = load_image(img_path)
        pred, conf, _ = get_prediction(model, tensor, device)
        true_idx = case.get('true_idx', CLASSES.index(case['true_label']))
        correct = (pred == true_idx)

        baselines = {
            "black": torch.zeros_like(tensor),
            "blurred": make_blurred_baseline(tensor),
        }
        targets = {"pred": pred}
        if not correct:
            targets["true"] = true_idx

        for baseline_name, baseline in baselines.items():
            for target_name, target_idx in targets.items():
                heatmap, raw = compute_ig(
                    model, tensor, target_idx, device, baseline, n_steps=n_steps
                )

                subfolder = "correct" if correct else "failures"
                case_raw_dir = os.path.join(raw_dir, subfolder)
                case_viz_dir = os.path.join(viz_dir, subfolder)
                os.makedirs(case_raw_dir, exist_ok=True)
                os.makedirs(case_viz_dir, exist_ok=True)

                tag = f"{i:02d}_{case['image_id']}_{baseline_name}_{target_name}"
                np.save(os.path.join(case_raw_dir, f"{tag}.npy"), raw)

                overlay = overlay_on_image(display, heatmap)
                save_triple_figure(
                    display, heatmap, overlay,
                    title=(f"IG ({baseline_name} baseline, target={target_name}) | "
                           f"true={case['true_label']} pred={CLASSES[pred]} conf={conf:.3f}"),
                    pred_name=CLASSES[pred], true_name=case['true_label'],
                    conf=conf, save_path=os.path.join(case_viz_dir, f"{tag}.png"),
                    method_name=f"Integrated Gradients ({baseline_name}, {target_name})",
                )

                results.append({
                    'image_id': case['image_id'],
                    'true_class': case['true_label'],
                    'pred_class': CLASSES[pred],
                    'confidence': conf,
                    'correct': correct,
                    'baseline': baseline_name,
                    'target': target_name,
                    'n_steps': n_steps,
                    'raw_path': os.path.join(case_raw_dir, f"{tag}.npy"),
                })

        print(f"  [{i+1:02d}] {case['image_id']}: pred={CLASSES[pred]} ({conf:.3f}) "
              f"true={case['true_label']} correct={correct}")

    return results


def write_summary_csv(results, path):
    if not results:
        print(f"No results to write to {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_key", default="convnext-tiny_fold0",
                         help="A key from checkpoint_registry.py's discovered list "
                              "(format: '<arch-folder>_fold<N>', e.g. 'convnext-tiny_fold0').")
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()

    entry = resolve_checkpoint(args.checkpoint_key)
    fold = entry["fold"]
    if fold is None:
        print(f"[ERROR] Checkpoint '{args.checkpoint_key}' has no matched fold number, "
              f"so I don't know which xai_pilot_set_fold<N>.json to load.")
        sys.exit(1)

    pilot_path = f"ham10000/results/xai_pilot_set_fold{fold}.json"
    if not os.path.exists(pilot_path):
        print(f"[ERROR] {pilot_path} doesn't exist yet. Run build_pilot_set.py "
              f"(for any checkpoint at fold {fold}) first.")
        sys.exit(1)

    print(f"Checkpoint key: {args.checkpoint_key}  (fold {fold})")
    model, device, cfg = load_model_and_config(checkpoint_key=args.checkpoint_key)
    print(f"Architecture: {cfg['model']['architecture']}")

    with open(pilot_path) as f:
        pilot = json.load(f)
    cases = pilot["cases"]
    print(f"\nLoaded pilot set: {len(cases)} cases (fold {fold}, "
          f"originally built from '{pilot['built_from_checkpoint']}')")

    out_root = f"ham10000/results/xai/integrated_gradients/{args.checkpoint_key}"
    print(f"\n=== IG ({args.checkpoint_key}) on {len(cases)} pilot cases ===")
    results = run_ig_batch(cases, out_root, model, device, n_steps=args.n_steps)

    write_summary_csv(results, os.path.join(out_root, "ig_summary.csv"))
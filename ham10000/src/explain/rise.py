"""
RISE (Randomized Input Sampling for Explanation) -- Petsiuk, Das & Saenko, 2018.

Fully black-box: no gradients, no architecture assumptions -- works
identically on CNNs and transformers (Swin/MaxViT) once those are
loadable, which is the whole reason RISE is worth having alongside a
gradient method like IG.

Mask design (tuned per your instructions, not vanilla textbook RISE):
  - Coarse grid (s x s, default 6x6) upsampled to image resolution with
    BILINEAR interpolation, then randomly shift-cropped. This is the
    standard RISE construction, but the coarse grid + bilinear step is
    what specifically avoids "tiny pinprick" noise -- a fine grid (e.g.
    s=16+) produces lots of small scattered holes; a coarse grid produces
    a handful of large, smooth, blob-shaped occlusion regions that better
    match "does the model need this whole contextual region," not just
    "does the model need this one pixel."
  - Per-mask keep-probability p sampled from a range (default 0.3-0.7)
    instead of one fixed p. This is the "balanced distribution" part --
    across N masks you get everything from mostly-occluded to
    mostly-visible, so the model is tested against a genuinely
    representative spread of occlusion severity, not N repeats of the
    same coverage level.

Aggregation: because p now varies per mask, the textbook RISE formula
(divide by a single constant N*p) is no longer correct -- it assumes
every mask has the same expected coverage. Instead this computes, for
EVERY PIXEL independently: the confidence-weighted average of the
model's predicted-class score across only the masks that kept that pixel
visible:

    saliency[pixel] = sum_i( mask_i[pixel] * confidence_i ) / sum_i( mask_i[pixel] )

This is the general, always-correct form of RISE's weighting -- it
degrades to the textbook formula when p is fixed, and stays correct when
p varies per mask.

Usage:
    python ham10000/src/explain/rise.py
    python ham10000/src/explain/rise.py --checkpoint_key resnet50_fold0 --n_masks 1000
"""
import os, sys, json, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (overlay_on_image, load_image,
                    get_prediction, find_image, save_triple_figure,
                    load_model_and_config, resolve_checkpoint, CLASSES)

MASK_CACHE_DIR = "ham10000/results/xai/rise/mask_cache"


def generate_masks(N: int, s: int, p_range: tuple, img_size: int, seed: int):
    """Coarse-grid, bilinear-upsampled, randomly-shifted RISE masks.

    Returns:
        masks: float32 array [N, img_size, img_size], values in [0,1]
               (soft edges from bilinear interpolation, not hard 0/1)
        ps:    the sampled keep-probability used for each mask (for logging)
    """
    rng = np.random.RandomState(seed)
    cell_size = int(np.ceil(img_size / s))
    up_size = (s + 1) * cell_size  # extra room so we can randomly shift after upsampling

    masks = np.empty((N, img_size, img_size), dtype=np.float32)
    ps = rng.uniform(p_range[0], p_range[1], size=N).astype(np.float32)

    for i in range(N):
        grid = (rng.rand(s, s) < ps[i]).astype(np.float32)
        grid_t = torch.from_numpy(grid)[None, None]  # [1,1,s,s]
        upsampled = F.interpolate(
            grid_t, size=(up_size, up_size), mode='bilinear', align_corners=False
        )[0, 0].numpy()

        # Random shift-crop -- decorrelates mask edges from the fixed grid,
        # per Petsiuk et al.; without this every mask's blob edges would
        # land on the same handful of grid lines.
        x = rng.randint(0, cell_size + 1)
        y = rng.randint(0, cell_size + 1)
        masks[i] = upsampled[x:x + img_size, y:y + img_size]

    return masks, ps


def get_or_create_mask_cache(N: int, s: int, p_range: tuple, img_size: int, seed: int,
                              cache_dir: str = MASK_CACHE_DIR) -> np.ndarray:
    """Masks are the same for every image (that's the point of RISE's
    random-sampling design), so generate once, cache to disk, reuse across
    the whole pilot set and across reruns."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"N{N}_s{s}_p{p_range[0]:.2f}-{p_range[1]:.2f}_img{img_size}_seed{seed}"
    cache_path = os.path.join(cache_dir, f"rise_masks_{tag}.npy")

    if os.path.exists(cache_path):
        print(f"Loaded cached RISE masks: {cache_path}")
        return np.load(cache_path)

    print(f"Generating {N} RISE masks (s={s}, p in [{p_range[0]},{p_range[1]}], "
          f"seed={seed})...")
    masks, ps = generate_masks(N, s, p_range, img_size, seed)
    np.save(cache_path, masks)
    print(f"  actual mean keep-fraction across masks: {masks.mean():.3f} "
          f"(sampled p range: {ps.min():.2f}-{ps.max():.2f})")
    print(f"Cached -> {cache_path}")
    return masks


def normalize_for_display(attr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] for heatmap display.

    Deliberately NOT the same as utils.normalize_attr (which does
    zero-clip + divide-by-max -- correct for IG's signed, near-zero-mean
    attributions, but wrong here). RISE's raw values are confidence
    scores with no meaningful zero baseline (e.g. ranging ~0.3-0.9, never
    near 0), so dividing by max alone barely stretches the contrast and
    the whole heatmap reads as uniformly "hot." Subtracting the min first
    restores real contrast between important and unimportant regions.
    """
    mn, mx = attr.min(), attr.max()
    return (attr - mn) / (mx - mn) if mx > mn else np.zeros_like(attr)


def compute_rise(model, tensor, target_class, device, masks: np.ndarray, batch_size: int = 32):
    """
    tensor:  [1,3,H,W], already normalized (same preprocessing as training)
    masks:   [N,H,W] float in [0,1], from generate_masks/get_or_create_mask_cache
    Returns (normalized_heatmap_for_display, raw_saliency_map).
    raw_saliency_map is untouched (true weighted-average confidence per
    pixel) -- only the display heatmap uses min-max normalization.
    """
    N = masks.shape[0]
    img = tensor.to(device)
    mask_t = torch.from_numpy(masks).to(device)

    scores = np.empty(N, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch_masks = mask_t[start:start + batch_size]                 # [b,H,W]
            batch_masked = img * batch_masks.unsqueeze(1)                   # [b,3,H,W]
            probs = F.softmax(model(batch_masked), dim=1)
            scores[start:start + batch_masks.shape[0]] = (
                probs[:, target_class].cpu().numpy()
            )

    # Per-pixel confidence-weighted average -- see module docstring for why
    # this (not the fixed-p textbook formula) is the correct aggregation
    # once p varies per mask.
    weighted_sum = (masks * scores[:, None, None]).sum(axis=0)   # [H,W]
    mask_sum = masks.sum(axis=0)                                  # [H,W]
    raw = weighted_sum / np.clip(mask_sum, 1e-6, None)

    return normalize_for_display(raw), raw


def generate_explanation(model, image_tensor, target_class, device, masks, batch_size=32, **_):
    """Common-signature wrapper (matches the same call shape IG's
    compute_ig follows) so a downstream comparison script can call RISE
    interchangeably with IG once both exist."""
    return compute_rise(model, image_tensor, target_class, device, masks, batch_size=batch_size)


def run_rise_batch(cases, out_dir, model, device, masks, n_masks, batch_size=32, max_cases=None):
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

        targets = {"pred": pred}
        if not correct:
            targets["true"] = true_idx

        for target_name, target_idx in targets.items():
            heatmap, raw = compute_rise(
                model, tensor, target_idx, device, masks, batch_size=batch_size
            )

            subfolder = "correct" if correct else "failures"
            case_raw_dir = os.path.join(raw_dir, subfolder)
            case_viz_dir = os.path.join(viz_dir, subfolder)
            os.makedirs(case_raw_dir, exist_ok=True)
            os.makedirs(case_viz_dir, exist_ok=True)

            tag = f"{i:02d}_{case['image_id']}_{target_name}"
            np.save(os.path.join(case_raw_dir, f"{tag}.npy"), raw)

            overlay = overlay_on_image(display, heatmap)
            save_triple_figure(
                display, heatmap, overlay,
                title=(f"RISE (N={n_masks}, target={target_name}) | "
                       f"true={case['true_label']} pred={CLASSES[pred]} conf={conf:.3f}"),
                pred_name=CLASSES[pred], true_name=case['true_label'],
                conf=conf, save_path=os.path.join(case_viz_dir, f"{tag}.png"),
                method_name=f"RISE (target={target_name})",
            )

            results.append({
                'image_id': case['image_id'],
                'true_class': case['true_label'],
                'pred_class': CLASSES[pred],
                'confidence': conf,
                'correct': correct,
                'target': target_name,
                'n_masks': n_masks,
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
    parser.add_argument("--checkpoint_key", default="convnext-tiny_fold0")
    parser.add_argument("--n_masks", type=int, default=500,
                         help="Spec calls for N=2000, but that's 2000 forward passes "
                              "PER IMAGE on CPU -- very slow for a pilot run. Start at "
                              "500 to sanity-check the pipeline/visuals, then scale up "
                              "once you've confirmed it looks right.")
    parser.add_argument("--grid_size", type=int, default=6,
                         help="Coarse mask grid resolution (s x s). Smaller = larger, "
                              "smoother occlusion blobs. Larger = finer detail but more "
                              "pinprick-like.")
    parser.add_argument("--p_min", type=float, default=0.3)
    parser.add_argument("--p_max", type=float, default=0.7)
    parser.add_argument("--mask_batch_size", type=int, default=32,
                         help="How many masked copies of one image go through the model "
                              "per forward pass. Higher = faster but more memory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entry = resolve_checkpoint(args.checkpoint_key)
    fold = entry["fold"]
    if fold is None:
        print(f"[ERROR] Checkpoint '{args.checkpoint_key}' has no matched fold number.")
        sys.exit(1)

    pilot_path = f"ham10000/results/xai_pilot_set_fold{fold}.json"
    if not os.path.exists(pilot_path):
        print(f"[ERROR] {pilot_path} doesn't exist yet. Run build_pilot_set.py first.")
        sys.exit(1)

    print(f"Checkpoint key: {args.checkpoint_key}  (fold {fold})")
    model, device, cfg = load_model_and_config(checkpoint_key=args.checkpoint_key)
    print(f"Architecture: {cfg['model']['architecture']}")

    with open(pilot_path) as f:
        pilot = json.load(f)
    cases = pilot["cases"]
    print(f"\nLoaded pilot set: {len(cases)} cases (fold {fold})")

    masks = get_or_create_mask_cache(
        N=args.n_masks, s=args.grid_size, p_range=(args.p_min, args.p_max),
        img_size=224, seed=args.seed,
    )

    out_root = f"ham10000/results/xai/rise/{args.checkpoint_key}"
    print(f"\n=== RISE ({args.checkpoint_key}) on {len(cases)} pilot cases, "
          f"N={args.n_masks} masks/image ===")
    results = run_rise_batch(
        cases, out_root, model, device, masks, n_masks=args.n_masks,
        batch_size=args.mask_batch_size,
    )

    write_summary_csv(results, os.path.join(out_root, "rise_summary.csv"))
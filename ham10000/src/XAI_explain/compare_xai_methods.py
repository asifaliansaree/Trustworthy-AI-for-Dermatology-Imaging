"""
Side-by-side comparison: Original | IG | RISE | FovEx | Intersection | Union.

Reuses each method's own saved raw .npy attribution (never re-runs any
method), and collapses IG's [3,H,W] signed map to [H,W] the same way
eval_xai.py's load_raw_attribution() does, so "IG" here means the same
thing it means in your faithfulness numbers.

Usage:
    python ham10000/src/XAI_explain/compare_xai_methods.py \
        --checkpoint_key convnext-tiny_fold3 \
        --image_id ISIC_0028941 \
        --target pred \
        --ig_baseline black \
        --top_frac 0.20
"""
import os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import find_image, load_image

RESULT_ROOTS = {
    "ig": "ham10000/results/xai/integrated_gradients/{key}",
    "rise": "ham10000/results/xai/rise/{key}",
    "fovex": "ham10000/results/xai/fovex/{key}",
}


def find_raw_npy(method: str, checkpoint_key: str, image_id: str, target: str,
                  ig_baseline: str = "black") -> str:
    """Globs raw/{correct,failures}/*_{image_id}_..._{target}.npy for the given
    method -- doesn't assume the numeric case-index prefix lines up across
    methods (it doesn't have to), and doesn't assume correct/failures."""
    root = RESULT_ROOTS[method].format(key=checkpoint_key)
    pattern_parts = [f"*_{image_id}_"]
    if method == "ig":
        pattern_parts.append(f"{ig_baseline}_")
    pattern_parts.append(f"{target}.npy")
    pattern = "".join(pattern_parts)

    hits = []
    for subfolder in ["correct", "failures"]:
        hits.extend(glob.glob(os.path.join(root, "raw", subfolder, pattern)))
    if not hits:
        raise FileNotFoundError(
            f"No raw {method} attribution found for image_id={image_id}, "
            f"target={target} under {root}/raw/. Did you run that method's "
            f"script (and eval_xai.py, if needed) on this checkpoint/case first?"
        )
    if len(hits) > 1:
        print(f"[WARN] {method}: {len(hits)} matches for this pattern, using first: {hits[0]}")
    return hits[0]


def load_raw_attribution(path: str) -> np.ndarray:
    """Same collapsing rule as eval_xai.py's load_raw_attribution -- kept
    identical on purpose so 'IG' here means the same map your faithfulness
    numbers were computed from, not a re-derived one."""
    raw = np.load(path)
    raw = np.squeeze(raw)
    if raw.ndim == 3:          # IG: [C,H,W] signed -> collapse via mean(|.|)
        raw = np.mean(np.abs(raw), axis=0)
    assert raw.ndim == 2, f"expected [H,W] after collapse, got {raw.shape} from {path}"
    return raw


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)


def top_frac_mask(heatmap01: np.ndarray, top_frac: float) -> np.ndarray:
    """Binary mask of the top `top_frac` fraction of pixels by value.
    Percentile-based (not an absolute-value threshold) specifically because
    IG/RISE/FovEx raw magnitudes live on completely different scales -- see
    the shape/range check in this conversation (IG: [-0.18,0.11] signed,
    RISE: [0.13,0.16], FovEx: [~0,4.95]). Comparing them by fixed value
    would be meaningless; comparing by 'is this pixel in my own top X%' is
    the same normalization eval_xai.py's rank_pixels_by_attribution()
    already relies on for deletion/insertion ordering."""
    flat = heatmap01.flatten()
    k = max(1, int(top_frac * flat.size))
    thresh = np.partition(flat, -k)[-k]
    return heatmap01 >= thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_key", default="convnext-tiny_fold3")
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--target", default="pred", choices=["pred", "true"])
    parser.add_argument("--ig_baseline", default="black", choices=["black", "blurred"],
                         help="IG saved both baselines separately -- pick one "
                              "for this comparison figure, same as eval_xai.py's "
                              "per-baseline pairing.")
    parser.add_argument("--top_frac", type=float, default=0.20,
                         help="Fraction of highest-attribution pixels counted "
                              "as 'attended to', per method, before computing "
                              "intersection/union. 0.20 is a common XAI-paper "
                              "default; re-tune and report whichever you use.")
    parser.add_argument("--out_dir", default="ham10000/results/xai/comparison")
    args = parser.parse_args()

    # --- load original image (same display convention as every other script) ---
    img_path = find_image(args.image_id)
    _, _, display = load_image(img_path)   # [224,224,3] in [0,1]

    # --- load + normalize each method's raw attribution ---
    heatmaps = {}
    for method in ["ig", "rise", "fovex"]:
        path = find_raw_npy(method, args.checkpoint_key, args.image_id, args.target,
                             ig_baseline=args.ig_baseline)
        raw = load_raw_attribution(path)
        heatmaps[method] = normalize01(raw)
        print(f"{method}: loaded {path}")

    # --- binarize each at top_frac, then intersection / union ---
    masks = {m: top_frac_mask(h, args.top_frac) for m, h in heatmaps.items()}
    intersection = masks["ig"] & masks["rise"] & masks["fovex"]
    union = masks["ig"] | masks["rise"] | masks["fovex"]

    iou_all3 = intersection.sum() / union.sum() if union.sum() > 0 else 0.0
    print(f"3-way IoU (all methods agreeing / any method attending), "
          f"top_frac={args.top_frac}: {iou_all3:.4f}")

    # --- figure: Original | IG | RISE | FovEx | Intersection | Union ---
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    axes[0].imshow(display)
    axes[0].set_title("Original", fontsize=10)

    for ax, method, label in zip(axes[1:4], ["ig", "rise", "fovex"], ["IG", "RISE", "FovEx"]):
        ax.imshow(display)
        ax.imshow(heatmaps[method], cmap="jet", alpha=0.5)
        ax.set_title(label, fontsize=10)

    axes[4].imshow(display)
    axes[4].imshow(intersection, cmap="Greens", alpha=0.6)
    axes[4].set_title(f"Intersection (all 3)\ntop{int(args.top_frac*100)}%", fontsize=10)

    axes[5].imshow(display)
    axes[5].imshow(union, cmap="Oranges", alpha=0.5)
    axes[5].set_title(f"Union (any of 3)\nIoU={iou_all3:.3f}", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{args.image_id}  |  checkpoint={args.checkpoint_key}  |  "
                 f"target={args.target}  |  ig_baseline={args.ig_baseline}", fontsize=10, y=1.03)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{args.checkpoint_key}_{args.image_id}_{args.target}_top{int(args.top_frac*100)}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
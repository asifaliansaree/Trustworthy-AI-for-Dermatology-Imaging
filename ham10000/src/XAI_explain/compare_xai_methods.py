"""
Side-by-side comparison: Original | IG | RISE | FovEx | Intersection | Union.

Reuses each method's own saved raw .npy attribution (never re-runs any
method), and collapses IG's [3,H,W] signed map to [H,W] the same way
eval_xai.py's load_raw_attribution() does, so "IG" here means the same
thing it means in your faithfulness numbers.

Usage:
    # single image
    python ham10000/src/XAI_explain/compare_xai_methods.py \
        --checkpoint_key convnext-tiny_fold3 \
        --image_id ISIC_0028941 \
        --target pred \
        --ig_baseline black \
        --top_frac 0.20

    # every image in the pilot set: pred for correct cases,
    # pred AND true for failure cases
    python ham10000/src/XAI_explain/compare_xai_methods.py \
        --checkpoint_key convnext-tiny_fold3 \
        --all_pilot \
        --target both \
        --ig_baseline black \
        --top_frac 0.20
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# Connected-component cleanup (Part 3/6) is done with scipy if it's
# available, and skipped (with a one-time note, not a crash) if it isn't --
# this script's core job (correct masks + correct output paths + real
# FovEx fixations) doesn't require it. No hard dependency on cv2/skimage,
# since neither is guaranteed to be in every environment this runs in
# (confirmed: cv2 isn't in this project's venv).
try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import find_image, load_image
# Reused, not re-derived: this is the ONE place FovEx's (row/y, col/x) ->
# pixel-(x,y) coordinate convention is implemented anywhere in the project.
# See its docstring in fovex.py for the full derivation/empirical check.
from fovex import fixation_to_pixel

RESULT_ROOTS = {
    "ig": "ham10000/results/xai/integrated_gradients/{key}",
    "rise": "ham10000/results/xai/rise/{key}",
    "fovex": "ham10000/results/xai/fovex/{key}",
}


def case_subfolder(path: str) -> str:
    """Extracts 'correct' or 'failures' from a path returned by
    find_raw_npy/find_fixations_npy -- both always glob under exactly one
    of those two subfolders, so this just reads back which one matched.
    Used to mirror the raw results' correct/failures split in the
    comparison output, instead of dumping everything into one flat folder."""
    parts = os.path.normpath(path).split(os.sep)
    if "correct" in parts:
        return "correct"
    if "failures" in parts:
        return "failures"
    raise ValueError(f"Could not determine correct/failures from path: {path}")


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


def find_fixations_npy(checkpoint_key: str, image_id: str, target: str) -> str:
    """Same globbing convention as find_raw_npy, but for FovEx's saved
    scanpath -- ham10000/results/xai/fovex/{key}/fixations/{correct,failures}/
    {i:02d}_{image_id}_{target}_fixations.npy (see fovex.py's
    run_fovex_batch, which writes raw heatmap and fixations side-by-side
    under the same {tag} prefix). This is the ACTUAL optimized scanpath
    returned by FovExWrapper.run_optimization -- not reconstructed from
    the heatmap."""
    root = RESULT_ROOTS["fovex"].format(key=checkpoint_key)
    pattern = f"*_{image_id}_{target}_fixations.npy"

    hits = []
    for subfolder in ["correct", "failures"]:
        hits.extend(glob.glob(os.path.join(root, "fixations", subfolder, pattern)))
    if not hits:
        raise FileNotFoundError(
            f"No FovEx fixations found for image_id={image_id}, target={target} "
            f"under {root}/fixations/. Did fovex.py finish running on this "
            f"checkpoint/case (it saves fixations alongside the raw heatmap)?"
        )
    if len(hits) > 1:
        print(f"[WARN] fovex fixations: {len(hits)} matches, using first: {hits[0]}")
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


def resize_scalar_map(map2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a CONTINUOUS scalar map (not yet thresholded) to the display
    image's resolution, using bilinear interpolation -- spatially correct
    for smooth-valued heatmaps. Only actually resizes if the shape differs;
    all three methods here are already produced at image_size=224 (same as
    `display`), so this is a safety net, not something normally triggered.
    Do NOT use this on an already-binarized mask (see resize_mask below) --
    bilinear interpolation of 0/1 values creates fractional boundary
    pixels, which is exactly the "threshold RGB" class of bug this script
    is meant to avoid (Part 6/7)."""
    h, w = map2d.shape
    if (h, w) == (target_h, target_w):
        return map2d
    # PIL's resize takes (width, height); BILINEAR matches cv2.INTER_LINEAR
    # for this purpose (smooth scalar map, not a binary mask -- see
    # resize_mask below for why binary masks must NOT go through here).
    img = Image.fromarray(map2d.astype(np.float32), mode="F")
    resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(resized, dtype=np.float64)


def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an already-binary mask with NEAREST-NEIGHBOR interpolation,
    which preserves hard 0/1 boundaries instead of inventing fractional
    edge values. Prefer resize_scalar_map() + re-threshold over this where
    possible (see module docstring's pipeline order) -- this exists only
    as a safety net for a method that somehow only exposes a pre-binarized
    mask."""
    h, w = mask.shape
    if (h, w) == (target_h, target_w):
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = img.resize((target_w, target_h), resample=Image.NEAREST)
    return np.array(resized) > 0


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


_WARNED_NO_SCIPY = False


def _connected_components(mask: np.ndarray):
    """Returns (labeled_array, n_components) using 8-connectivity. Only
    called when scipy is available (see _HAS_SCIPY)."""
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    return _ndi.label(mask, structure=structure)


def clean_mask(mask: np.ndarray, min_frac: float = 0.0015) -> np.ndarray:
    """Removes tiny isolated connected components (single/few stray pixels)
    from a binary mask, WITHOUT aggressively smoothing away real lesion-
    sized regions. min_frac is a fraction of total image pixels (default
    ~0.15%, i.e. ~34 px for a 224x224 image) -- small enough to only catch
    genuine speckle noise, not a small-but-real lesion-aligned region.
    Safe on an all-zero mask. If scipy isn't installed, this is a no-op
    (prints a one-time note) rather than a hard crash -- cleanup is a
    quality improvement, not required for mathematically correct
    intersection/union."""
    global _WARNED_NO_SCIPY
    if mask.sum() == 0:
        return mask
    if not _HAS_SCIPY:
        if not _WARNED_NO_SCIPY:
            print("[NOTE] scipy not installed -- skipping small-component "
                  "mask cleanup (masks are still mathematically correct, "
                  "just not de-speckled). `pip install scipy` to enable it.")
            _WARNED_NO_SCIPY = True
        return mask
    size_thresh = max(1, int(min_frac * mask.size))
    labeled, n = _connected_components(mask)
    if n == 0:
        return mask
    sizes = _ndi.sum(mask, labeled, index=np.arange(1, n + 1))
    keep_labels = np.arange(1, n + 1)[sizes > size_thresh]
    return np.isin(labeled, keep_labels)


def fragmentation_ratio(mask: np.ndarray) -> float:
    """Diagnostic only (Part 19): (# connected components) / (# foreground
    pixels). Close to 1.0 means the mask is mostly isolated single pixels
    (fragmented noise); much less than 1.0 means a handful of coherent
    blobs, which is what a sane top-frac mask on a smooth heatmap should
    look like. Returns 0.0 (nothing to report) if scipy isn't installed."""
    if mask.sum() == 0:
        return 0.0
    if not _HAS_SCIPY:
        return 0.0
    _, n_components = _connected_components(mask)
    return n_components / mask.sum()


def build_consensus_rgba(intersection: np.ndarray, union_only: np.ndarray,
                          alpha: float = 0.40) -> np.ndarray:
    """Builds an RGBA overlay (NOT a colormap applied to 0/1 values) where:
        - intersection pixels -> opaque-ish GREEN at `alpha`
        - union-only pixels   -> opaque-ish ORANGE at `alpha`
        - everything else     -> fully transparent (alpha channel = 0)
    This is the fix for the Part 3/4/5 bug: the old code did
    `imshow(binary_mask, cmap=..., alpha=scalar)`, which tints EVERY pixel
    (including background) because a scalar alpha with a colormap has no
    concept of "pixels outside the mask should be transparent." Building
    the RGBA array explicitly, per-pixel, is what actually makes only the
    masked region visible and leaves the rest of the original image
    untouched underneath."""
    h, w = intersection.shape
    rgba = np.zeros((h, w, 4), dtype=np.float64)
    green = np.array([0.10, 0.75, 0.20])
    orange = np.array([0.95, 0.55, 0.10])
    rgba[intersection, :3] = green
    rgba[intersection, 3] = alpha
    rgba[union_only, :3] = orange
    rgba[union_only, 3] = alpha
    return rgba


def draw_fovex_panel(ax, display, fovex_heatmap01, fixation_points, H, W):
    """FovEx panel = heatmap + the ACTUAL optimized scanpath (numbered
    fixations, connected in order, thin line), not a re-derived set of
    points. Coordinate handling delegates entirely to fixation_to_pixel()
    (imported from fovex.py) so this stays consistent with fovex.py's own
    figures if that convention is ever revisited."""
    ax.imshow(display)
    ax.imshow(fovex_heatmap01, cmap="jet", alpha=0.5)

    margin_x = max(10, int(0.04 * W))
    margin_top = max(16, int(0.08 * H))
    margin_bottom = max(10, int(0.04 * H))
    _, _, pixel_x, pixel_y = fixation_to_pixel(fixation_points, H, W)
    px = np.clip(pixel_x, margin_x, W - 1 - margin_x)
    py = np.clip(pixel_y, margin_top, H - 1 - margin_bottom)

    outline = [pe.withStroke(linewidth=2.0, foreground='white')]
    ax.plot(px, py, '-', color='white', linewidth=0.9, alpha=0.75, clip_on=True)
    ax.scatter(px, py, marker='x', c='#FF0000', s=28, linewidths=1.3,
               zorder=5, clip_on=True, path_effects=outline)
    # Only label first/last to avoid overcrowding a small panel (Part 9).
    for i in (0, len(px) - 1):
        ax.annotate(str(i + 1), (px[i], py[i]), xytext=(0, 6),
                    textcoords='offset points', fontsize=7, fontweight='bold',
                    ha='center', va='bottom', zorder=6, clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='#FF0000', linewidth=0.7, alpha=0.85))
    ax.text(0.02, 0.02, "\u25CF fixation   \u2500 scanpath", transform=ax.transAxes,
            fontsize=6.5, color='white', va='bottom', ha='left',
            path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])


def process_one_image(checkpoint_key: str, image_id: str, target: str, ig_baseline: str,
                       top_frac: float, out_dir: str, no_mask_cleanup: bool,
                       verbose: bool = True) -> bool:
    """Does everything main() used to do for exactly one (image_id, target)
    pair. Returns True on success, False if this case should be SKIPPED
    (e.g. one method's output doesn't exist for this particular case --
    common for target='true' on a correctly-classified case, since fovex.py
    /rise.py/IG only ever compute 'true' for failures). Raises for anything
    that isn't a plain "file missing" situation, so real bugs still surface
    loudly instead of being silently skipped."""
    try:
        img_path = find_image(image_id)
    except FileNotFoundError as e:
        print(f"[SKIP] {image_id} ({target}): {e}")
        return False
    _, _, display = load_image(img_path)   # [224,224,3] in [0,1]
    H, W = display.shape[0], display.shape[1]

    # --- load + normalize each method's raw attribution ---
    heatmaps = {}
    subfolder = None
    try:
        for method in ["ig", "rise", "fovex"]:
            path = find_raw_npy(method, checkpoint_key, image_id, target,
                                 ig_baseline=ig_baseline)
            if method == "ig":
                # IG's location tells us whether this case was a correct
                # prediction or a failure -- correctness is a property of
                # the CASE (model's prediction vs. ground truth), not of
                # any one XAI method, so any of the three would agree here.
                # Used only to mirror that split in the output folder below.
                subfolder = case_subfolder(path)
            raw = load_raw_attribution(path)
            raw = resize_scalar_map(raw, H, W)   # no-op unless a method's map ever differs in size
            heatmaps[method] = normalize01(raw)
            if verbose:
                print(f"{method}: loaded {path}  shape={raw.shape}")

        # --- load FovEx's ACTUAL saved fixation coordinates (never re-derived) ---
        fix_path = find_fixations_npy(checkpoint_key, image_id, target)
        fixation_points = np.load(fix_path)
        if verbose:
            print(f"fovex fixations: loaded {fix_path}  n_fixations={len(fixation_points)}")
    except FileNotFoundError as e:
        print(f"[SKIP] {image_id} ({target}): {e}")
        return False

    # --- spatial-alignment sanity check (Part 7/19) ---
    assert heatmaps["ig"].shape == heatmaps["rise"].shape == heatmaps["fovex"].shape, (
        f"heatmaps not spatially aligned after resize: "
        f"ig={heatmaps['ig'].shape}, rise={heatmaps['rise'].shape}, "
        f"fovex={heatmaps['fovex'].shape}"
    )

    # --- binarize each at top_frac (on the numerical map, not an RGB image),
    #     then intersection / union, then light cleanup of stray pixels ---
    masks = {m: top_frac_mask(h, top_frac) for m, h in heatmaps.items()}
    intersection = masks["ig"] & masks["rise"] & masks["fovex"]
    union = masks["ig"] | masks["rise"] | masks["fovex"]

    if not no_mask_cleanup:
        intersection = clean_mask(intersection)
        union = clean_mask(union)
        # Re-enforce intersection subset-of union after independent cleanup
        # passes (cleaning each mask separately can, in principle, drop a
        # pixel from one but not the other) -- keeps Part 19's invariant exact.
        intersection = intersection & union

    union_only = union & ~intersection

    # --- sanity checks (Part 19) ---
    assert np.all(intersection <= union), "intersection_mask must be <= union_mask everywhere"
    intersection_area = int(intersection.sum())
    union_area = int(union.sum())
    assert intersection_area <= union_area

    iou_all3 = intersection_area / union_area if union_area > 0 else 0.0

    if intersection_area == 0:
        print(f"[WARN] {image_id}: intersection is empty -- the three "
              f"methods share no top-frac pixels at all.")
    if union_area >= 0.95 * union.size:
        print(f"[WARN] {image_id}: union covers >=95% of the image -- top_frac "
              f"may be too high, or one method's mask is nearly the whole frame.")
    for name, m in [("intersection", intersection), ("union", union)]:
        frag = fragmentation_ratio(m)
        if frag > 0.5:
            print(f"[WARN] {image_id}: {name} mask looks fragmented "
                  f"(components/pixels={frag:.2f}) -- mostly isolated noise "
                  f"rather than coherent regions, even after cleanup.")

    # --- debug diagnostics for this image (Part 18) ---
    if verbose:
        print(f"\nImage: {image_id}")
        print(f"Display shape: {display.shape[:2]}")
        for m in ["ig", "rise", "fovex"]:
            print(f"{m.upper()} map: {heatmaps[m].shape}  "
                  f"top-{int(top_frac*100)}% pixels: {int(masks[m].sum())}")
        print(f"Intersection: {intersection_area}")
        print(f"Union: {union_area}")
        print(f"IoU: {iou_all3:.4f}")
        print(f"FovEx fixations: {len(fixation_points)}\n")
    else:
        print(f"{image_id} ({target}): IoU={iou_all3:.4f}  "
              f"intersection={intersection_area}px  union={union_area}px  "
              f"fixations={len(fixation_points)}")

    # --- figure: Original | IG | RISE | FovEx | Intersection | Union ---
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    axes[0].imshow(display)
    axes[0].set_title("Original", fontsize=10)

    for ax, method, label in zip(axes[1:3], ["ig", "rise"], ["IG", "RISE"]):
        ax.imshow(display)
        ax.imshow(heatmaps[method], cmap="jet", alpha=0.5)
        ax.set_title(label, fontsize=10)

    draw_fovex_panel(axes[3], display, heatmaps["fovex"], fixation_points, H, W)
    axes[3].set_title("FovEx + fixations", fontsize=10)

    # --- Intersection: original image + explicit RGBA green overlay only on
    # intersection pixels (see build_consensus_rgba's docstring for why the
    # old cmap+scalar-alpha approach tinted the whole image). ---
    intersection_rgba = build_consensus_rgba(intersection, np.zeros_like(intersection), alpha=0.42)
    axes[4].imshow(display)
    axes[4].imshow(intersection_rgba)
    axes[4].set_title(f"Consensus (IG \u2229 RISE \u2229 FovEx)\ntop{int(top_frac*100)}%",
                       fontsize=10)
    axes[4].legend(handles=[mpatches.Patch(color=(0.10, 0.75, 0.20), label="All 3 agree")],
                   loc="lower right", fontsize=6.5, framealpha=0.75, handlelength=1.2)

    # --- Union: same original image, green = intersection, orange =
    # union-only (detected by >=1 but not all 3). ---
    union_rgba = build_consensus_rgba(intersection, union_only, alpha=0.42)
    axes[5].imshow(display)
    axes[5].imshow(union_rgba)
    axes[5].set_title(f"Coverage (IG \u222A RISE \u222A FovEx)\nIoU={iou_all3:.4f}", fontsize=10)
    axes[5].legend(handles=[
        mpatches.Patch(color=(0.10, 0.75, 0.20), label="All 3 agree"),
        mpatches.Patch(color=(0.95, 0.55, 0.10), label="\u22651, not all 3"),
    ], loc="lower right", fontsize=6.5, framealpha=0.75, handlelength=1.2)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{image_id}  |  checkpoint={checkpoint_key}  |  "
                 f"target={target}  |  ig_baseline={ig_baseline}", fontsize=10, y=1.03)
    plt.tight_layout()

    # --- output: comparison/<checkpoint_key>/<correct|failures>/<file>.png,
    # mirroring the same correct/failures split every method's raw results
    # already use, instead of dumping every case into one flat folder.
    # Never comparison/<file>.png and never comparison/comparison/... . ---
    ckpt_out_dir = os.path.join(out_dir, checkpoint_key, subfolder)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    out_path = os.path.join(
        ckpt_out_dir,
        f"{checkpoint_key}_{image_id}_{target}_top{int(top_frac*100)}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return True


def load_pilot_cases(checkpoint_key: str) -> list:
    """Resolves checkpoint_key -> fold via the same registry every other
    script in this project uses, then reads that fold's pilot set
    (built by build_pilot_set.py) and returns the raw case dicts
    (image_id, correct, etc.) -- not just image_ids -- so batch mode can
    report correct/failure counts without re-deriving correctness from a
    file path. Doesn't load the model -- resolve_checkpoint() is pure file
    lookup."""
    import json
    from utils import resolve_checkpoint
    entry = resolve_checkpoint(checkpoint_key)
    fold = entry["fold"]
    pilot_path = f"ham10000/results/xai_pilot_set_fold{fold}.json"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(
            f"{pilot_path} doesn't exist -- run build_pilot_set.py for fold "
            f"{fold} first."
        )
    with open(pilot_path) as f:
        pilot = json.load(f)
    return pilot["cases"]


def run_batch(checkpoint_key: str, target: str, ig_baseline: str, top_frac: float,
              out_dir: str, no_mask_cleanup: bool) -> None:
    """Runs process_one_image over every case in the pilot set for one
    target ('pred' or 'true'). Split out of main() so --target both can
    call this twice (pred pass, then true pass) instead of duplicating
    the loop."""
    cases = load_pilot_cases(checkpoint_key)
    n_correct_total = sum(1 for c in cases if c["correct"])
    n_failure_total = sum(1 for c in cases if not c["correct"])
    print(f"=== Batch mode: {len(cases)} images from the pilot set "
          f"({n_correct_total} correct, {n_failure_total} failures), "
          f"checkpoint={checkpoint_key}, target={target} ===\n")
    n_ok_correct, n_ok_failure, n_skip = 0, 0, 0
    for case in cases:
        ok = process_one_image(
            checkpoint_key, case["image_id"], target, ig_baseline,
            top_frac, out_dir, no_mask_cleanup, verbose=False,
        )
        if ok:
            if case["correct"]:
                n_ok_correct += 1
            else:
                n_ok_failure += 1
        else:
            n_skip += 1
    print(f"\n=== Done (target={target}): {n_ok_correct + n_ok_failure} saved "
          f"({n_ok_correct} correct, {n_ok_failure} failures), "
          f"{n_skip} skipped (out of {len(cases)}) ->\n"
          f"    {os.path.join(out_dir, checkpoint_key, 'correct')}/\n"
          f"    {os.path.join(out_dir, checkpoint_key, 'failures')}/ ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_key", default="convnext-tiny_fold3")
    parser.add_argument("--image_id",
                         help="Single image to process. Omit this and pass "
                              "--all_pilot instead to batch-process every "
                              "image in this checkpoint's fold pilot set.")
    parser.add_argument("--all_pilot", action="store_true",
                         help="Process every image_id in this checkpoint's "
                              "fold pilot set (ham10000/results/"
                              "xai_pilot_set_fold{N}.json) instead of a "
                              "single --image_id. Cases missing one of the "
                              "three methods' outputs for the requested "
                              "--target are skipped (printed, not fatal) "
                              "rather than aborting the whole batch.")
    parser.add_argument("--target", default="pred", choices=["pred", "true", "both"],
                         help="'pred' = explain the model's predicted class "
                              "(exists for every case, correct or not). "
                              "'true' = explain the ground-truth class "
                              "(only exists for FAILURE cases -- IG/RISE/"
                              "FovEx never compute this for correctly-"
                              "classified images, so correct cases are "
                              "skipped automatically, not an error). "
                              "'both' = run pred, then true, in one call.")
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
    parser.add_argument("--no_mask_cleanup", action="store_true",
                         help="Disable small-connected-component removal on "
                              "the intersection/union masks (Part 3/6). On by "
                              "default.")
    args = parser.parse_args()

    if not args.all_pilot and not args.image_id:
        parser.error("Provide either --image_id (single image) or --all_pilot "
                     "(every image in this checkpoint's fold pilot set).")

    targets = ["pred", "true"] if args.target == "both" else [args.target]

    if args.all_pilot:
        for target in targets:
            run_batch(args.checkpoint_key, target, args.ig_baseline,
                      args.top_frac, args.out_dir, args.no_mask_cleanup)
    else:
        any_ok = False
        for target in targets:
            ok = process_one_image(
                args.checkpoint_key, args.image_id, target, args.ig_baseline,
                args.top_frac, args.out_dir, args.no_mask_cleanup, verbose=True,
            )
            any_ok = any_ok or ok
        if not any_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
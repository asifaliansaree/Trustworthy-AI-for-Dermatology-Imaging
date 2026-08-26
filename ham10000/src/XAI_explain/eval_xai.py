"""
Quantitative XAI evaluation: Deletion AUC, Insertion AUC, Pointing Game.

Works across IG, RISE, and FovEx without method-specific code, because it
doesn't re-run any method -- it reads the raw attribution .npy files each
method already saved, plus that method's own summary CSV (ig_summary.csv /
rise_summary.csv / fovex_summary.csv) for the image/target/path bookkeeping.
Ranking is done purely by attribution MAGNITUDE (|value|, descending),
which works identically whether the raw values are signed (IG) or
unsigned (RISE, FovEx) -- no per-method normalization needed for this,
since deletion/insertion only care about pixel ORDER, not the actual
attribution scale.

Deletion AUC: start from the real image, progressively replace the
highest-attribution pixels with a baseline (blurred or black), track how
fast the target-class probability collapses. LOWER is better -- a good
explanation identifies pixels the model actually depends on, so removing
them should hurt fast.

Insertion AUC: the mirror image -- start from an all-baseline image,
progressively reveal the highest-attribution pixels, track how fast the
target-class probability recovers. HIGHER is better.

Pointing Game: does the single highest-attribution pixel fall inside the
true lesion's segmentation mask? Conditional on masks actually being
available -- if --masks_dir isn't provided or a given image has no mask
file, this is skipped for that row (logged as "not_computed" in the CSV,
not silently dropped or treated as a failure).

NOTE on IG's two baselines (black/blurred): IG's own generation step can
integrate its attribution from two different starting points (black vs
blurred image), producing two distinct, independently-valid raw maps per
(image_id, target) -- NOT duplicate rows. ig_summary.csv correctly has one
row per (image_id, target, baseline): 68 unique (image_id, target) pairs x
2 baselines = 136 rows. This evaluator preserves that `baseline` column
into its output as `attribution_baseline`, and PAIRS each row's own
deletion/insertion eval fill to it -- a black-generated map is evaluated
with a black fill, a blurred-generated map with a blurred fill -- so one
input row produces exactly one output row (136 in, 136 out), not a
black x blurred cross product. `--baseline` is only a fallback fill value
for rows with no attribution_baseline of their own (RISE, FovEx). The
final summary breaks out Mean Deletion/Insertion AUC per
attribution_baseline whenever more than one is present, instead of
silently blending them into one mean that would conflate "how good is
IG" with "black vs blurred."

Usage:
    python ham10000/src/explain/eval_xai.py --checkpoint_key convnext-tiny_fold3 --method ig
    python ham10000/src/explain/eval_xai.py --checkpoint_key convnext-tiny_fold3 --method rise
    python ham10000/src/explain/eval_xai.py --checkpoint_key convnext-tiny_fold3 --method fovex \
        --masks_dir ham10000/data/ISIC_masks --mask_suffix _segmentation.png
"""
import os, sys, csv, argparse, time, resource
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# NumPy renamed trapz -> trapezoid in 2.0; support both installed versions
# rather than assuming which one the user has.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import load_image, load_model_and_config, resolve_checkpoint, CLASSES

SUMMARY_PATHS = {
    "ig": "ham10000/results/xai/integrated_gradients/{key}/ig_summary.csv",
    "rise": "ham10000/results/xai/rise/{key}/rise_summary.csv",
    "fovex": "ham10000/results/xai/fovex/{key}/fovex_summary.csv",
}


def print_memory_usage(label=""):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    print(f"  [mem] peak RSS so far{f' ({label})' if label else ''}: {peak_mb:.1f} MB")


def assert_image_shape(tensor, name="tensor"):
    assert tensor.ndim == 4 and tensor.shape[1] == 3, (
        f"{name}: expected [B,3,H,W], got {tuple(tensor.shape)}"
    )


def make_blurred_baseline(tensor: torch.Tensor, kernel_size: int = 21, sigma: float = 5.0) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    return TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def load_raw_attribution(path: str) -> np.ndarray:
    """Loads a raw attribution .npy and collapses to [H,W].

    First squeezes any singleton dims (e.g. a [1,H,W] save from RISE/FovEx
    becomes [H,W] directly) -- this is defensive, since this evaluator
    reads whatever each method's own script already wrote to disk and
    shouldn't assume a specific save convention without checking. AFTER
    squeezing, a genuine [C,H,W] map (IG saves per-channel) still collapses
    via mean(|.|) over channels, since that's a real multi-channel map, not
    a singleton dim.
    """
    raw = np.load(path)
    raw = np.squeeze(raw)
    if raw.ndim == 3:
        raw = np.mean(np.abs(raw), axis=0)
    assert raw.ndim == 2, f"expected [H,W] after squeeze/channel collapse, got shape {raw.shape} from {path}"
    return raw


def rank_pixels_by_attribution(raw_map: np.ndarray):
    """Returns (flat pixel order, descending by |attribution|), H, W."""
    importance = np.abs(raw_map)
    H, W = importance.shape
    order = np.argsort(-importance.reshape(-1))
    return order, H, W


def _build_step_batch(base_tensor, fill_tensor, order, H, W, n_steps, mode, device):
    """mode='deletion': step k has top-k pixels replaced with fill_tensor.
    mode='insertion': step k has top-k pixels replaced with base_tensor's
    real content, everything else is fill_tensor (i.e. starts all-baseline,
    reveals pixels). Returns a [n_steps+1, C, H, W] batch."""
    n_pixels = H * W
    step_size = max(1, n_pixels // n_steps)
    imgs = []
    active_mask_flat = np.zeros(n_pixels, dtype=bool)

    for step in range(n_steps + 1):
        n_active = min(n_pixels, step * step_size)
        active_mask_flat[:] = False
        active_mask_flat[order[:n_active]] = True
        mask = torch.from_numpy(active_mask_flat.reshape(H, W)).to(device)

        if mode == "deletion":
            img_step = base_tensor.clone()
            img_step[0, :, mask] = fill_tensor[0, :, mask]
        elif mode == "insertion":
            img_step = fill_tensor.clone()
            img_step[0, :, mask] = base_tensor[0, :, mask]
        else:
            raise ValueError(mode)
        imgs.append(img_step)

    return torch.cat(imgs, dim=0)


def run_curve(model, base_tensor, raw_map, target_idx, device, fill_tensor, mode, n_steps=100, batch_size=25):
    assert_image_shape(base_tensor, "base_tensor")
    order, H, W = rank_pixels_by_attribution(raw_map)
    batch = _build_step_batch(base_tensor.to(device), fill_tensor.to(device), order, H, W, n_steps, mode, device)

    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, batch.shape[0], batch_size):
            chunk = batch[start:start + batch_size]
            p = F.softmax(model(chunk), dim=1)[:, target_idx]
            probs.append(p.cpu().numpy())
    probs = np.concatenate(probs)

    x = np.linspace(0, 1, len(probs))
    auc = float(_trapezoid(probs, x))
    return auc, probs


def pointing_game(raw_map: np.ndarray, mask_path: str):
    """Returns True/False, or None if no mask available (not a failure --
    just not computed for this row)."""
    if mask_path is None or not os.path.exists(mask_path):
        return None
    order, H, W = rank_pixels_by_attribution(raw_map)
    peak_idx = order[0]
    py, px = divmod(peak_idx, W)
    mask_img = np.array(Image.open(mask_path).convert("L").resize((W, H)))
    return bool(mask_img[py, px] > 127)


def load_method_results(method: str, checkpoint_key: str) -> pd.DataFrame:
    path = SUMMARY_PATHS[method].format(key=checkpoint_key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No summary CSV found at {path}. Run {method}'s own script for "
            f"'{checkpoint_key}' first -- this evaluator reads its output, doesn't "
            f"regenerate explanations."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    if "baseline" in df.columns and df["baseline"].nunique() > 1:
        counts = df["baseline"].value_counts().to_dict()
        print(f"  [NOTE] {df['baseline'].nunique()} distinct attribution baselines present "
              f"in this summary ({counts}) -- e.g. IG's black vs blurred integration "
              f"baseline. These are NOT duplicate rows: each (image_id, target, baseline) "
              f"combo is an independently-valid attribution to evaluate. Results below are "
              f"reported both pooled and broken out per baseline.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_key", required=True)
    parser.add_argument("--method", required=True, choices=["ig", "rise", "fovex"])
    parser.add_argument("--baseline", choices=["black", "blurred"], default="blurred",
                         help="This script's deletion/insertion fill value -- but ONLY "
                              "used as a fallback for rows that have no attribution_baseline "
                              "of their own (RISE, FovEx -- methods with a single, "
                              "baseline-agnostic attribution per image). For IG, whose "
                              "summary CSV has one row per (image_id, target, baseline) "
                              "-- a black-integrated and a blurred-integrated attribution "
                              "for the same image -- this script instead PAIRS each row's "
                              "eval baseline to that row's own attribution baseline "
                              "(black-generated -> evaluated with black fill, "
                              "blurred-generated -> evaluated with blurred fill). This "
                              "keeps one output row per input row (136 in, 136 out for "
                              "IG) instead of cross-producing every row against both "
                              "fill values, which would evaluate mismatched combinations "
                              "(e.g. a black-generated map under a blurred fill) that "
                              "don't correspond to anything meaningful and would double "
                              "row count for no reason. This flag has no effect on IG rows.")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--masks_dir", default=None,
                         help="Directory of lesion segmentation masks for Pointing Game. "
                              "If omitted, Pointing Game is skipped for every row "
                              "(logged as 'not_computed', not silently dropped).")
    parser.add_argument("--mask_suffix", default="_segmentation.png")
    parser.add_argument("--max_rows", type=int, default=None,
                         help="Cap rows processed, for a quick sanity check before a full run.")
    args = parser.parse_args()

    entry = resolve_checkpoint(args.checkpoint_key)
    print(f"Checkpoint key: {args.checkpoint_key}")
    model, device, cfg = load_model_and_config(checkpoint_key=args.checkpoint_key)
    print(f"Architecture: {cfg['model']['architecture']}")

    if args.masks_dir is None:
        print("[NOTE] --masks_dir not provided -- Pointing Game will be skipped for "
              "every row (logged as 'not_computed'). Pass --masks_dir to enable it.")
    elif not os.path.isdir(args.masks_dir):
        print(f"[WARN] --masks_dir '{args.masks_dir}' doesn't exist -- Pointing Game "
              f"will be skipped for every row.")

    df = load_method_results(args.method, args.checkpoint_key)
    if args.max_rows:
        df = df.head(args.max_rows)

    results = []
    n_pointing_computed = 0
    print(f"\n=== Evaluating {args.method.upper()} explanations "
          f"({len(df)} rows, fallback eval baseline={args.baseline} "
          f"[only applies to rows with no attribution_baseline of their own], "
          f"n_steps={args.n_steps}) ===")

    for i, row in df.iterrows():
        t_start = time.time()
        raw_path = row["raw_path"]
        if not os.path.exists(raw_path):
            print(f"  [{i+1}] SKIP (raw file missing): {raw_path}")
            continue

        raw_map = load_raw_attribution(raw_path)

        try:
            from model import DermaNet  # noqa: F401 (import-order sanity, model already loaded)
        except Exception:
            pass

        # Rebuild the exact input tensor for this image (deterministic given
        # the same preprocessing IG/RISE/FovEx all already used).
        from utils import find_image
        img_path = find_image(row["image_id"])
        tensor, _, _ = load_image(img_path)

        target_idx = (CLASSES.index(row["pred_class"]) if row["target"] == "pred"
                      else CLASSES.index(row["true_class"]))

        # Which baseline the ATTRIBUTION was generated from (IG: black/blurred;
        # "n/a" for RISE/FovEx, which have one attribution per image, no
        # baseline choice). When present, PAIR the eval baseline to it --
        # a black-generated map gets evaluated with a black fill, a
        # blurred-generated map with a blurred fill -- rather than running
        # every row against both fill values, which would (a) evaluate
        # mismatched combinations that don't correspond to anything the
        # generation step actually produced, and (b) double the output row
        # count relative to the input CSV for no reason. Rows with no
        # attribution_baseline (RISE/FovEx) fall back to --baseline.
        attribution_baseline = row.get("baseline", "n/a")
        eval_baseline = attribution_baseline if attribution_baseline != "n/a" else args.baseline
        baseline_tag = f" attrib_baseline={attribution_baseline}" if attribution_baseline != "n/a" else ""

        fill = make_blurred_baseline(tensor) if eval_baseline == "blurred" else torch.zeros_like(tensor)

        del_auc, _ = run_curve(model, tensor, raw_map, target_idx, device, fill,
                                mode="deletion", n_steps=args.n_steps, batch_size=args.batch_size)
        ins_auc, _ = run_curve(model, tensor, raw_map, target_idx, device, fill,
                                mode="insertion", n_steps=args.n_steps, batch_size=args.batch_size)

        pg = None
        if args.masks_dir and os.path.isdir(args.masks_dir):
            mask_path = os.path.join(args.masks_dir, row["image_id"] + args.mask_suffix)
            pg = pointing_game(raw_map, mask_path)
            if pg is not None:
                n_pointing_computed += 1

        elapsed = time.time() - t_start
        results.append({
            "image_id": row["image_id"],
            "method": args.method,
            "target": row["target"],
            "attribution_baseline": attribution_baseline,
            "eval_baseline": eval_baseline,
            "true_class": row["true_class"],
            "pred_class": row["pred_class"],
            "correct": row["correct"],
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
            "pointing_game": pg if pg is not None else "not_computed",
        })
        print(f"  [{i+1:03d}/{len(df)}] {row['image_id']} ({row['target']}){baseline_tag} "
              f"eval_baseline={eval_baseline}: "
              f"del_auc={del_auc:.4f} ins_auc={ins_auc:.4f} "
              f"pointing={pg if pg is not None else 'n/a'}  ({elapsed:.1f}s)")
        if (i + 1) % 10 == 0:
            print_memory_usage(f"after {i+1} rows")

    out_dir = f"ham10000/results/xai/eval/{args.checkpoint_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_faithfulness.csv")
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} rows -> {out_path}")

        results_df = pd.DataFrame(results)
        del_mean = results_df["deletion_auc"].mean()
        ins_mean = results_df["insertion_auc"].mean()
        print(f"\n=== Summary: {args.method.upper()} on {args.checkpoint_key} ===")
        print(f"Pooled Mean Deletion AUC:  {del_mean:.4f}  (lower is better)  n={len(results_df)}")
        print(f"Pooled Mean Insertion AUC: {ins_mean:.4f}  (higher is better)  n={len(results_df)}")

        # Break out per attribution_baseline (e.g. IG's black vs blurred
        # integration baseline) whenever more than one is present. Note
        # eval_baseline is always paired 1:1 to attribution_baseline (see
        # the main loop), so this breakdown already reflects both -- e.g.
        # "attribution_baseline=black" here means "black-generated map,
        # evaluated with a black fill" as a single matched unit, not a
        # blend across mismatched combinations.
        n_attrib_baselines = results_df["attribution_baseline"].nunique()
        if n_attrib_baselines > 1:
            print(f"\n[{n_attrib_baselines} attribution baselines detected -- pooled number "
                  f"above blends them. Breakdown (eval baseline matched to each row's own "
                  f"attribution baseline):]")
            for baseline_val, group in results_df.groupby("attribution_baseline"):
                print(f"  attribution_baseline={baseline_val} (eval matched): "
                      f"Mean Deletion AUC={group['deletion_auc'].mean():.4f}  "
                      f"Mean Insertion AUC={group['insertion_auc'].mean():.4f}  "
                      f"n={len(group)}")

        if args.masks_dir:
            print(f"\nPointing Game computed for {n_pointing_computed}/{len(results)} rows "
                  f"(rest had no matching mask file)")
    else:
        print("No results computed.")
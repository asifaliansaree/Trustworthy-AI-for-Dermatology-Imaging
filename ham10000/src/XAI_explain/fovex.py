"""
FovEx (Panda et al., IJCV 2025) -- generalized over checkpoint_key.

How it actually works (per your instructions): a small Gaussian "fovea"
starts at the image center. Everything OUTSIDE the fovea is blurred; the
fovea position is then optimized via gradient descent so that the model's
loss (on the target class) is minimized -- i.e. the fovea moves toward
whatever region the model needs sharp/unblurred to make its prediction.
After `optimization_steps` gradient steps, the best-found fixation is
locked in and folded into a running "already-seen-sharply" memory via
`internal_representation = internal_representation * forgetting +
new_fixation_mask` (see FovEx_patched.py). Despite the name, HIGHER
`forgetting` means that memory persists LONGER (inhibition-of-return:
already-fixated regions stay banked as sharp, so the gradient search is
pushed toward new regions instead); LOWER `forgetting` erases it almost
immediately, which -- combined with the fixation search restarting from
the image center every step -- can make the scanpath collapse onto the
same spot repeatedly instead of exploring. The process then repeats for
`scanpath_length` fixations total. The sequence of fixation points is
the "scanpath"; the fixation points, weighted/aggregated with a
per-fixation Gaussian, form the final attribution heatmap. This is what
distinguishes it from RISE (which never uses gradients, only random
occlusion + confidence) and IG (which explains via one gradient pass
through the whole image at once, not an iterative sequence of
increasingly-informed glances).

Uses the patched vendored library (fovex_lib/FovEx_patched.py) -- see
that file's PATCH LOG docstring for exactly what was changed vs. the
official repo (device-string fixes only, no algorithm changes).

MANDATORY SMOKE TEST: FovEx's authors validated it on ResNet-50 and
ViT-B/16 only, never ConvNeXt (or any of your other 5 architectures).
The algorithm is architecture-agnostic by construction (confirmed by
reading FovEx.py -- it only ever calls model(x), no architecture-specific
hooks), but "should work in theory" isn't the same as "does work well in
practice" on an unvalidated architecture. This script refuses to run the
full pilot set until you've reviewed a small smoke-test batch first.

Usage:
    python ham10000/src/explain/fovex.py                    # smoke test only (default)
    python ham10000/src/explain/fovex.py --full_run          # full pilot set, only
                                                               # after you've checked
                                                               # the smoke test output
"""
import os, sys, json, csv, argparse, time, resource
import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS)), os.path.join(_THIS, "fovex_lib")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (overlay_on_image, load_image, get_prediction, find_image,
                    load_model_and_config, resolve_checkpoint, CLASSES)
from FovEx_patched import FovExWrapper, get_heat_maps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def print_memory_usage(label=""):
    """Rough peak RSS logging. ru_maxrss is KB on Linux, bytes on macOS --
    normalize to MB assuming macOS (this project's dev machine), since
    printing the wrong unit is worse than a documented assumption."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    print(f"  [mem] peak RSS so far{f' ({label})' if label else ''}: {peak_mb:.1f} MB")


def assert_image_shape(tensor, name="tensor"):
    assert tensor.ndim == 4 and tensor.shape[1] == 3, (
        f"{name}: expected [B,3,H,W], got {tuple(tensor.shape)}"
    )


def assert_heatmap_shape(heatmap, name="heatmap"):
    assert heatmap.ndim == 2, f"{name}: expected [H,W], got {getattr(heatmap, 'shape', None)}"


def assert_scanpath_shape(fixations, name="fixations"):
    assert fixations.ndim == 2 and fixations.shape[1] == 2, (
        f"{name}: expected [T,2], got {getattr(fixations, 'shape', None)}"
    )


def target_function(x, y):
    """Identity pass-through, exactly as used in the paper's own example
    notebook -- the "target" for the loss is just whatever class index
    you tell it to explain (pred or ground-truth), no transformation."""
    return y


def build_fovex_wrapper(model, device, hp: dict) -> FovExWrapper:
    criterion = nn.CrossEntropyLoss(reduction="none")
    return FovExWrapper(
        downstream_model=model,
        criterion=criterion,
        target_function=target_function,
        image_size=hp["image_size"],
        foveation_sigma=hp["foveation_sigma"],
        blur_filter_size=hp["blur_filter_size"],
        blur_sigma=hp["blur_sigma"],
        forgetting=hp["forgetting"],
        heatmap_sigma=hp["heatmap_sigma"],
        heatmap_forgetting=hp["heatmap_forgetting"],
        foveation_aggregation=1,
        device=device,
    )


def heatmap_focus_metrics(raw_heatmap):
    """Cheap, mask-free proxies for how tightly a heatmap is concentrated
    vs. smeared. These say NOTHING about whether the focus lands on the
    right region (no lesion mask used here) -- only how concentrated it
    is. Still requires visual review; use as a ranking aid, not ground
    truth."""
    h = raw_heatmap.astype(np.float64)
    total = h.sum()
    if total <= 0:
        return {"concentration": 0.0, "spatial_std_frac": 1.0}
    H, W = h.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cy = (ys * h).sum() / total
    cx = (xs * h).sum() / total
    var = (((ys - cy) ** 2 + (xs - cx) ** 2) * h).sum() / total
    spatial_std_frac = np.sqrt(var) / np.sqrt(H * H + W * W)
    flat = np.sort(h.flatten())[::-1]
    k = max(1, int(0.10 * flat.size))
    concentration = float(flat[:k].sum() / total)
    return {"concentration": concentration, "spatial_std_frac": float(spatial_std_frac)}


def compute_fovex(fovex_wrapper, tensor, target_idx, device, hp: dict, seed=42, stats_out=None):
    """
    tensor: [1,3,H,W], already normalized (same preprocessing as IG/RISE).
    Returns (display_heatmap [H,W] in [0,1], raw_heatmap [H,W] unnormalized,
             fixation_points [scanpath_length, 2] in [-1,1] image coords).

    stats_out: optional dict, populated in-place with {'n_oob', 'n_fixations'}
    -- how many fixation points landed outside the valid [-1,1] range before
    clamping. This is the root cause of scanpath points appearing outside
    the image frame in the smoke-test plots: unclamped coordinates blow up
    matplotlib's autoscale and squash the real image into a corner.
    """
    from FovEx_patched import set_seed
    set_seed(seed)

    assert_image_shape(tensor, "input tensor")
    labels = torch.tensor([target_idx], dtype=torch.long, device=device)
    x = tensor.to(device)

    scanpaths, loss_history, internal_rep = fovex_wrapper.run_optimization(
        x, labels,
        scanpath_length=hp["scanpath_length"],
        opt_iterations=hp["optimization_steps"],
        learning_rate=hp["lr"],
        random_restarts=hp["random_restart"],
    )
    # scanpaths shape for batch_size=1: [scanpath_length, 2] after squeeze
    fixation_points = scanpaths.detach().cpu().numpy()
    if fixation_points.ndim == 1:  # scanpath_length==1 edge case, squeeze over-collapsed
        fixation_points = fixation_points.reshape(1, -1)
    assert_scanpath_shape(fixation_points, "fixation_points")

    # --- Fix: clamp any fixation that escaped [-1,1] BEFORE it's used to
    # build the heatmap or get plotted. A too-high lr (or too many
    # optimization_steps without enough gradient damping) can push the
    # unconstrained fixation parameter past the valid image range; every
    # downstream consumer (heatmap, scatter plot) assumed it never would.
    oob_mask = np.abs(fixation_points) > 1.0
    n_oob = int(oob_mask.any(axis=1).sum())
    if n_oob > 0:
        print(f"    [WARN] {n_oob}/{len(fixation_points)} fixation point(s) landed "
              f"outside [-1,1] (max |coord|={np.abs(fixation_points).max():.3f}) -- "
              f"clamping before building heatmap/plot. Consider lowering --lr or "
              f"--optimization_steps.")
        scanpaths = scanpaths.clamp(-1.0, 1.0)
        fixation_points = scanpaths.detach().cpu().numpy()
        if fixation_points.ndim == 1:
            fixation_points = fixation_points.reshape(1, -1)

    if stats_out is not None:
        stats_out["n_oob"] = n_oob
        stats_out["n_fixations"] = len(fixation_points)

    scanpaths_batched = scanpaths[None]  # add batch dim back, as generate_explanation does internally
    raw = get_heat_maps(
        hp["heatmap_sigma"], hp["image_size"], scanpaths_batched,
        hp["heatmap_forgetting"], device, normalization=False,
    )[0, 0].detach().cpu().numpy()
    display = get_heat_maps(
        hp["heatmap_sigma"], hp["image_size"], scanpaths_batched,
        hp["heatmap_forgetting"], device, normalization=True,
    )[0, 0].detach().cpu().numpy()
    assert_heatmap_shape(raw, "raw heatmap")
    assert_heatmap_shape(display, "display heatmap")

    return display, raw, fixation_points


def generate_explanation(model, image_tensor, target_class, device, fovex_wrapper, hp, seed=42, **_):
    """Common-signature wrapper matching IG/RISE's call shape."""
    return compute_fovex(fovex_wrapper, image_tensor, target_class, device, hp, seed=seed)


def save_fovex_figure(display, heatmap, overlay, fixation_points, title, pred_name, true_name,
                       conf, save_path, method_name="FovEx"):
    """Same 3-panel layout as IG/RISE, plus numbered fixation points drawn
    on the overlay panel -- FovEx's distinctive output the other two
    methods don't have."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display)
    axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"{method_name} heatmap", fontsize=10)
    axes[1].axis('off')
    correct = pred_name == true_name
    axes[2].imshow(overlay)
    # fixation_points are in [-1,1] normalized coords -> pixel coords.
    # A fixation clamped to exactly px=0 or px=W-1 still has a marker/label
    # whose visual extent (glyph stroke width, label padding) reaches past
    # that single pixel -- clip_on alone chops it mid-glyph at the frame
    # edge. Fix: inset the allowed pixel range by a margin sized to the
    # marker+label footprint, so the FULL marker and label always render
    # inside the frame with room to spare, not just the coordinate itself.
    # Extra margin on top accounts for the label sitting above the marker.
    H, W = display.shape[0], display.shape[1]
    margin_x = max(18, int(0.05 * W))
    margin_bottom = max(18, int(0.05 * H))
    margin_top = max(34, int(0.10 * H))
    px = np.clip((fixation_points[:, 0] + 1) / 2 * W, margin_x, W - 1 - margin_x)
    py = np.clip((fixation_points[:, 1] + 1) / 2 * H, margin_top, H - 1 - margin_bottom)
    # Pure bright red, per request. Note: 'jet' (the heatmap colormap) DOES
    # pass through red/orange at its hottest values -- the exact region
    # fixations tend to converge on -- so pure red can camouflage there.
    # Compensated with a thicker white outline (path_effects) so the cross
    # shape stays legible even when it lands on a red/orange hot spot.
    FIXATION_COLOR = '#FF0000'
    outline = [pe.withStroke(linewidth=2.8, foreground='white')]
    axes[2].plot(px, py, '-', color='white', linewidth=1, alpha=0.7, clip_on=True)
    axes[2].scatter(px, py, marker='x', c=FIXATION_COLOR, s=55, linewidths=1.8,
                     zorder=5, clip_on=True, path_effects=outline)
    for i, (xx, yy) in enumerate(zip(px, py)):
        axes[2].annotate(
            str(i + 1), (xx, yy), xytext=(0, 9), textcoords='offset points',
            fontsize=9, fontweight='bold', color='black', ha='center', va='bottom',
            fontfamily='monospace', zorder=6, clip_on=True,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=FIXATION_COLOR,
                       linewidth=0.9, alpha=0.9),
        )
    axes[2].set_title(f"Overlay + scanpath | Pred: {pred_name} ({conf:.2f})", fontsize=10,
                       color='#1D9E75' if correct else '#E24B4A')
    axes[2].axis('off')
    if title:
        fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_fovex_batch(cases, out_dir, model, device, fovex_wrapper, hp, max_cases=None):
    raw_dir = os.path.join(out_dir, "raw")
    viz_dir = os.path.join(out_dir, "viz")
    fix_dir = os.path.join(out_dir, "fixations")
    for d in (raw_dir, viz_dir, fix_dir):
        os.makedirs(d, exist_ok=True)

    results = []
    for i, case in enumerate(cases[:max_cases] if max_cases else cases):
        t_start = time.time()
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
            stats = {}
            heatmap, raw, fixations = compute_fovex(fovex_wrapper, tensor, target_idx, device, hp,
                                                      stats_out=stats)

            subfolder = "correct" if correct else "failures"
            case_raw_dir = os.path.join(raw_dir, subfolder)
            case_viz_dir = os.path.join(viz_dir, subfolder)
            case_fix_dir = os.path.join(fix_dir, subfolder)
            for d in (case_raw_dir, case_viz_dir, case_fix_dir):
                os.makedirs(d, exist_ok=True)

            tag = f"{i:02d}_{case['image_id']}_{target_name}"
            np.save(os.path.join(case_raw_dir, f"{tag}.npy"), raw)
            np.save(os.path.join(case_fix_dir, f"{tag}_fixations.npy"), fixations)

            overlay = overlay_on_image(display, heatmap)
            save_fovex_figure(
                display, heatmap, overlay, fixations,
                title=(f"FovEx (scanpath_len={hp['scanpath_length']}, target={target_name}) | "
                       f"true={case['true_label']} pred={CLASSES[pred]} conf={conf:.3f}"),
                pred_name=CLASSES[pred], true_name=case['true_label'],
                conf=conf, save_path=os.path.join(case_viz_dir, f"{tag}.png"),
                method_name=f"FovEx (target={target_name})",
            )

            focus = heatmap_focus_metrics(raw)
            results.append({
                'image_id': case['image_id'],
                'true_class': case['true_label'],
                'pred_class': CLASSES[pred],
                'confidence': conf,
                'correct': correct,
                'target': target_name,
                'scanpath_length': hp["scanpath_length"],
                'n_oob_fixations': stats.get('n_oob', 0),
                'spatial_std_frac': round(focus['spatial_std_frac'], 4),
                'concentration': round(focus['concentration'], 4),
                'raw_path': os.path.join(case_raw_dir, f"{tag}.npy"),
                'fixations_path': os.path.join(case_fix_dir, f"{tag}_fixations.npy"),
            })

        elapsed = time.time() - t_start
        print(f"  [{i+1:02d}] {case['image_id']}: pred={CLASSES[pred]} ({conf:.3f}) "
              f"true={case['true_label']} correct={correct}  ({elapsed:.1f}s)")
        if (i + 1) % 5 == 0:
            print_memory_usage(f"after {i+1} images")

    return results


def run_hp_search(cases, out_root, model, device, device_str, base_hp,
                   lr_grid, steps_grid, foveation_sigma_grid, heatmap_sigma_grid,
                   forgetting_grid=None, n_cases=3, seed=42):
    """Grid search over (lr, optimization_steps, foveation_sigma, heatmap_sigma,
    forgetting) on a small subset of pilot cases -- run this on the specific
    images where the smoke test looked wrong, not the whole set (it's
    O(combos x images)).

    forgetting_grid: values for the already-fixated-region memory decay. Per
    FovEx_patched.py's run_optimization, `forgetting` multiplies the OLD
    accumulated internal_representation each step (new_repr = old_repr *
    forgetting + new_fixation_mask) -- so HIGHER values retain memory of
    past fixations longer, implementing inhibition-of-return (already-seen
    regions stay banked as sharp, diluting the gradient incentive to
    re-visit them). LOWER values erase memory almost immediately, which
    -- combined with foveation_pos being hard-reset to the image center at
    the start of every fixation step -- can make the search re-converge on
    the same local optimum every time instead of exploring. Defaults to
    [base_hp['forgetting']] (i.e. no sweep) if not given -- pass this
    explicitly when you see repeated/overlapping fixation numbers stacked
    at the same spot in a smoke-test plot, and bias the grid toward
    HIGHER values than the current default, not lower.

    For each combo, writes one contact-sheet PNG (all n_cases images side by
    side, so you can eyeball focus quality directly) and logs:
      - total_oob_points: fixation points that escaped [-1,1] before
        clamping, summed across images. This should be 0 -- it's the
        direct signal for "fixations went off-frame."
      - mean_spatial_std_frac: cheap heatmap-tightness proxy (lower =
        more concentrated). No lesion mask -- doesn't tell you if the
        focus is correct, only how spread out it is.
      - mean_concentration: fraction of heatmap mass in the top 10% of
        pixels (higher = more concentrated).
      - mean_confidence: target-class confidence at the end of the run.

    Results are ranked (fewest oob points, then tightest std) and written
    to hp_search_summary.csv. Always look at the contact sheets before
    trusting the ranking -- the metrics are focus-tightness proxies, not
    correctness proxies.
    """
    os.makedirs(out_root, exist_ok=True)
    subset = cases[:n_cases]
    rows = []
    combo_id = 0
    forgetting_grid = forgetting_grid or [base_hp["forgetting"]]
    total_combos = (len(lr_grid) * len(steps_grid) * len(foveation_sigma_grid)
                     * len(heatmap_sigma_grid) * len(forgetting_grid))
    print(f"HP search: {total_combos} combos x {len(subset)} images = "
          f"{total_combos * len(subset)} FovEx runs. This can be slow on CPU -- "
          f"consider a smaller grid or fewer n_cases first.")

    for lr in lr_grid:
      for steps in steps_grid:
        for fov_sigma in foveation_sigma_grid:
          for hmap_sigma in heatmap_sigma_grid:
            for forgetting in forgetting_grid:
                    combo_id += 1
                    hp = dict(base_hp)
                    hp["lr"] = lr
                    hp["optimization_steps"] = steps
                    hp["foveation_sigma"] = fov_sigma
                    hp["heatmap_sigma"] = hmap_sigma
                    hp["forgetting"] = forgetting
                    hp["heatmap_forgetting"] = [base_hp["heatmap_forgetting"][0]] * hp["scanpath_length"]

                    wrapper = build_fovex_wrapper(model, device_str, hp)

                    fig, axes = plt.subplots(1, len(subset), figsize=(4 * len(subset), 4))
                    if len(subset) == 1:
                        axes = [axes]

                    combo_oob, combo_conf, combo_std, combo_conc = 0, [], [], []

                    for j, case in enumerate(subset):
                        try:
                            img_path = find_image(case['image_id'])
                        except FileNotFoundError:
                            axes[j].axis('off')
                            continue
                        tensor, _, display = load_image(img_path)
                        pred, conf, _ = get_prediction(model, tensor, device)

                        stats = {}
                        heatmap, raw, fixations = compute_fovex(
                            wrapper, tensor, pred, device, hp, seed=seed, stats_out=stats)
                        metrics = heatmap_focus_metrics(raw)

                        combo_oob += stats.get('n_oob', 0)
                        combo_std.append(metrics["spatial_std_frac"])
                        combo_conc.append(metrics["concentration"])
                        combo_conf.append(conf)

                        overlay = overlay_on_image(display, heatmap)
                        axes[j].imshow(overlay)
                        H, W = display.shape[0], display.shape[1]
                        margin_x = max(14, int(0.05 * W))
                        margin_bottom = max(14, int(0.05 * H))
                        margin_top = max(26, int(0.10 * H))
                        px = np.clip((fixations[:, 0] + 1) / 2 * W, margin_x, W - 1 - margin_x)
                        py = np.clip((fixations[:, 1] + 1) / 2 * H, margin_top, H - 1 - margin_bottom)
                        FIXATION_COLOR = '#FF0000'
                        outline = [pe.withStroke(linewidth=2.2, foreground='white')]
                        axes[j].plot(px, py, '-', color='white', linewidth=1, alpha=0.7, clip_on=True)
                        axes[j].scatter(px, py, marker='x', c=FIXATION_COLOR, s=38, linewidths=1.4,
                                         zorder=5, clip_on=True, path_effects=outline)
                        for k, (xx, yy) in enumerate(zip(px, py)):
                            axes[j].annotate(
                                str(k + 1), (xx, yy), xytext=(0, 7), textcoords='offset points',
                                fontsize=7, fontweight='bold', color='black', ha='center', va='bottom',
                                fontfamily='monospace', zorder=6, clip_on=True,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                           edgecolor=FIXATION_COLOR, linewidth=0.7, alpha=0.9),
                            )
                        axes[j].set_title(f"{case['image_id']}\noob={stats.get('n_oob', 0)} "
                                           f"conf={conf:.2f}", fontsize=8)
                        axes[j].axis('off')

                    tag = f"lr{lr}_steps{steps}_fov{fov_sigma}_hmap{hmap_sigma}_forget{forgetting}"
                    fig.suptitle(f"lr={lr}  steps={steps}  foveation_sigma={fov_sigma}  "
                                 f"heatmap_sigma={hmap_sigma}  forgetting={forgetting}", fontsize=9)
                    plt.tight_layout()
                    sheet_path = os.path.join(out_root, f"combo_{combo_id:03d}_{tag}.png")
                    fig.savefig(sheet_path, dpi=130, bbox_inches='tight')
                    plt.close()

                    rows.append({
                        "combo_id": combo_id, "lr": lr, "optimization_steps": steps,
                        "foveation_sigma": fov_sigma, "heatmap_sigma": hmap_sigma,
                        "forgetting": forgetting,
                        "n_images": len(subset), "total_oob_points": combo_oob,
                        "mean_spatial_std_frac": round(float(np.mean(combo_std)), 4) if combo_std else None,
                        "mean_concentration": round(float(np.mean(combo_conc)), 4) if combo_conc else None,
                        "mean_confidence": round(float(np.mean(combo_conf)), 4) if combo_conf else None,
                        "contact_sheet": sheet_path,
                    })
                    print(f"  [{combo_id}/{total_combos}] {tag}: oob_pts={combo_oob} "
                          f"spatial_std_frac={np.mean(combo_std):.3f} "
                          f"concentration={np.mean(combo_conc):.3f}")

    if rows:
        csv_path = os.path.join(out_root, "hp_search_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        ranked = sorted(rows, key=lambda r: (r["total_oob_points"], r["mean_spatial_std_frac"]))
        print(f"\nWrote {csv_path}")
        print("Top 3 candidates (fewest out-of-bounds points, then tightest focus -- "
              "confirm visually via their contact sheets before adopting):")
        for r in ranked[:3]:
            print(f"  lr={r['lr']} steps={r['optimization_steps']} "
                  f"foveation_sigma={r['foveation_sigma']} heatmap_sigma={r['heatmap_sigma']} "
                  f"forgetting={r['forgetting']} "
                  f"-> oob={r['total_oob_points']} std_frac={r['mean_spatial_std_frac']:.3f} "
                  f"-> {r['contact_sheet']}")
    return rows


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
    parser.add_argument("--smoke_test", action="store_true",
                         help="Explicit alias for the default behavior (running only "
                              "--n_smoke images). Included for interface clarity -- "
                              "omitting both --smoke_test and --full_run already runs "
                              "the smoke test by default.")
    parser.add_argument("--full_run", action="store_true",
                         help="Run the full pilot set. Without this flag, only a small "
                              "smoke-test batch runs -- required because FovEx has never "
                              "been validated on this architecture.")
    parser.add_argument("--n_smoke", type=int, default=10,
                         help="Number of pilot images used for the smoke test.")
    # FovEx hyperparameters. NOTE: defaults below have been tuned down from the
    # paper's example notebook (lr=0.1, foveation_sigma=0.15) after the smoke
    # test showed fixation points escaping the image bounds on several cases
    # -- a high lr with the unconstrained fixation parameter can overshoot
    # past [-1,1]. Use --hp_search to re-derive better values for your own
    # architecture/checkpoint instead of trusting these blindly.
    parser.add_argument("--lr", type=float, default=0.03,
                         help="Reduced from the paper default of 0.1 -- the higher value "
                              "was pushing fixation coordinates outside the valid image "
                              "range on some smoke-test cases.")
    parser.add_argument("--blur_sigma", type=float, default=10)
    parser.add_argument("--forgetting", type=float, default=0.4,
                         help="Decay applied to the OLD accumulated 'already-fixated' "
                              "memory each scanpath step (internal_representation = "
                              "internal_representation*forgetting + new_fixation_mask -- "
                              "see FovEx_patched.py run_optimization). Counter-intuitive "
                              "naming: HIGHER values mean the memory of past fixations "
                              "persists LONGER, which implements inhibition-of-return "
                              "(previously-fixated regions stay 'banked' as already-sharp, "
                              "diluting the gradient incentive to re-visit them, pushing "
                              "the search toward new regions). LOWER values erase memory "
                              "almost immediately -- combined with foveation_pos being "
                              "reset to the image center at the start of every fixation "
                              "step (hardcoded, not a hyperparameter), a too-low value can "
                              "make every fixation re-converge on the same nearby local "
                              "optimum instead of exploring, which is exactly what caused "
                              "the stacked/overlapping fixation numbers seen on "
                              "low-confidence smoke-test cases at the paper's original "
                              "0.1 default. Raised here to 0.4; re-verify with "
                              "--hp_search --forgetting_grid on the specific case(s) "
                              "that showed collapse before trusting this value.")
    parser.add_argument("--scanpath_length", type=int, default=10)
    parser.add_argument("--blur_filter_size", type=int, default=41)
    parser.add_argument("--foveation_sigma", type=float, default=0.10,
                         help="Controls the effective size of the foveal (sharp) region, "
                              "i.e. the 'fovea radius', via a soft Gaussian falloff. NOTE: "
                              "the vendored FovEx algorithm has no separate hard-radius "
                              "parameter -- this IS the fovea radius control. Tightened "
                              "from the paper's 0.15 default; re-tune via --hp_search.")
    parser.add_argument("--optimization_steps", type=int, default=20)
    parser.add_argument("--heatmap_sigma", type=float, default=0.12,
                         help="Controls how large a blob each fixation contributes to the "
                              "final aggregated heatmap. Smaller = tighter, more localized "
                              "heatmap; larger = smoother/more diffuse. Tightened slightly "
                              "from the paper's 0.15 default.")
    parser.add_argument("--heatmap_forgetting_value", type=float, default=1.0,
                         help="Per-fixation weight w_t used when aggregating the final "
                              "heatmap. 1.0 (paper default) = no decay, every fixation "
                              "counts equally. <1.0 down-weights earlier fixations "
                              "relative to later ones.")
    parser.add_argument("--random_restart", action="store_true", default=True)

    # Hyperparameter search mode.
    parser.add_argument("--hp_search", action="store_true",
                         help="Run a grid search over lr/optimization_steps/"
                              "foveation_sigma/heatmap_sigma on a small case subset "
                              "instead of the normal smoke/full run.")
    parser.add_argument("--hp_search_n_cases", type=int, default=3,
                         help="How many pilot cases to use per combo in --hp_search. "
                              "Pick the specific problem cases if you know their indices "
                              "by reordering/filtering the pilot JSON, or just increase "
                              "n_smoke's slice here.")
    parser.add_argument("--lr_grid", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--steps_grid", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--foveation_sigma_grid", type=float, nargs="+", default=[0.08, 0.10, 0.15])
    parser.add_argument("--heatmap_sigma_grid", type=float, nargs="+", default=[0.10, 0.15])
    parser.add_argument("--forgetting_grid", type=float, nargs="+", default=None,
                         help="Sweep the already-fixated-region memory decay. Leave "
                              "unset for a normal focus/lr/steps sweep; set this "
                              "explicitly (e.g. 0.1 0.3 0.4 0.6 0.8) when a smoke-test "
                              "case shows multiple fixation numbers stacked at the same "
                              "spot (scanpath collapsing instead of exploring). HIGHER "
                              "values = more inhibition-of-return = more exploration -- "
                              "bias the grid upward from the current default, not "
                              "downward (see run_hp_search docstring).")
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
    arch = cfg['model']['architecture']
    print(f"Architecture: {arch}")
    if arch not in ("resnet50", "vit_b_16"):
        print(f"[NOTE] FovEx's authors validated this method on ResNet-50 and ViT-B/16 "
              f"only. '{arch}' is unvalidated territory -- that's exactly why the smoke "
              f"test below exists. Review its output before trusting the full run.")

    with open(pilot_path) as f:
        pilot = json.load(f)
    cases = pilot["cases"]
    print(f"\nLoaded pilot set: {len(cases)} cases (fold {fold})")

    hp = {
        "image_size": 224,
        "lr": args.lr,
        "blur_sigma": args.blur_sigma,
        "forgetting": args.forgetting,
        "scanpath_length": args.scanpath_length,
        "blur_filter_size": args.blur_filter_size,
        "foveation_sigma": args.foveation_sigma,
        "optimization_steps": args.optimization_steps,
        "heatmap_sigma": args.heatmap_sigma,
        "heatmap_forgetting": [args.heatmap_forgetting_value] * args.scanpath_length,
        "random_restart": args.random_restart,
    }
    print(f"Hyperparameters: {hp}")
    print(f"\n[RUNTIME NOTE] Each (image, target) pair runs "
          f"{args.scanpath_length} x {args.optimization_steps} = "
          f"{args.scanpath_length * args.optimization_steps} gradient steps on CPU. "
          f"This is meaningfully slower than IG per image. Budget accordingly.")

    device_str = str(device)
    fovex_wrapper = build_fovex_wrapper(model, device_str, hp)

    out_root = f"ham10000/results/xai/fovex/{args.checkpoint_key}"

    if args.hp_search:
        search_cases = cases[:args.hp_search_n_cases]
        print(f"\n=== HP SEARCH: FovEx ({args.checkpoint_key}) "
              f"on {len(search_cases)} case(s) ===")
        run_hp_search(
            search_cases, os.path.join(out_root, "hp_search"),
            model, device, device_str, hp,
            lr_grid=args.lr_grid, steps_grid=args.steps_grid,
            foveation_sigma_grid=args.foveation_sigma_grid,
            heatmap_sigma_grid=args.heatmap_sigma_grid,
            forgetting_grid=args.forgetting_grid,
            n_cases=args.hp_search_n_cases,
        )
    elif not args.full_run:
        smoke_cases = cases[:args.n_smoke]
        print(f"\n=== SMOKE TEST: FovEx ({args.checkpoint_key}) on {len(smoke_cases)} images ===")
        print("(Run with --full_run once you've checked these outputs look sane.)")
        results = run_fovex_batch(smoke_cases, os.path.join(out_root, "smoke_test"),
                                   model, device, fovex_wrapper, hp)
        write_summary_csv(results, os.path.join(out_root, "smoke_test", "fovex_smoke_summary.csv"))
        print(f"\nSmoke test done. Inspect PNGs under "
              f"{out_root}/smoke_test/viz/ before running --full_run.")
    else:
        print(f"\n=== FULL RUN: FovEx ({args.checkpoint_key}) on {len(cases)} pilot cases ===")
        results = run_fovex_batch(cases, out_root, model, device, fovex_wrapper, hp)
        write_summary_csv(results, os.path.join(out_root, "fovex_summary.csv"))
# Week 5 — Attribution Benchmark & Comparison Harness

## Decisions made

### Why these three methods alongside Grad-CAM
The synopsis asks for "≥4 attribution methods" for the full team-scope
version of the explainability track. The four chosen —
Grad-CAM (Week 4), vanilla Saliency, Integrated Gradients, and Occlusion —
span the two fundamentally different families of attribution method:
**gradient-based** (Grad-CAM, Saliency, Integrated Gradients — cheap, but
can be noisy or saturate) and **perturbation-based** (Occlusion — slow,
but model-agnostic and doesn't rely on gradient behavior at all). Having
one method from the slow-but-robust family is what makes the later
faithfulness checks (Week 9) meaningful.

### Why Integrated Gradients uses a zero baseline and `n_steps=50`
A zero (black-image) baseline is the standard default for IG and is what
most published dermatology-XAI comparisons use, which keeps this
benchmark comparable to prior work. 50 steps is Captum's own recommended
default for the Riemann approximation to converge without excessive
runtime.

### Why occlusion uses a 16×16 window with stride 8
This gives 50% overlap between windows — coarse enough to run in
reasonable time on CPU, fine enough to still localize sub-regions of a
224×224 lesion image rather than just quadrants. The tradeoff is noted
directly in `occlusion.py`: larger windows are faster but coarser.

### Why `compare_methods.py` renders one 5-panel figure per case instead of one gallery per method
Putting Original | Grad-CAM | IG | Saliency | Occlusion in a single figure,
per case, makes method *disagreement* visible immediately — if three
methods agree the model is looking at the lesion but one lights up an
artifact instead, that's a single glance rather than four separate
galleries to cross-reference by eye.

## Key findings

- All three additional methods were run against the same case list used
  for Grad-CAM in Week 4, using the same `v0.1-baseline` checkpoint.
- Per-case comparison figures were generated for the first 8 cases in
  `results/xai/comparison/` (5-panel: Original, Grad-CAM, IG, Saliency,
  Occlusion), annotated with true/predicted label, confidence, and a
  correct/wrong color code.
- Full per-case galleries were generated for Integrated Gradients and
  Occlusion individually in `results/xai/integrated_gradients/` and
  `results/xai/occlusion/`.
- Occlusion is meaningfully slower than the three gradient-based methods
  (the full 8-case comparison run is dominated by occlusion's compute
  cost, since it requires one forward pass per sliding-window position
  rather than a single backward pass).

## What this means for Week 6

Right now method agreement/disagreement can only be judged qualitatively,
by eye. Week 6 (mask-overlap evaluation against ISIC lesion masks — IoU,
pointing game) turns "does the heatmap look like it's on the lesion?"
into a number, which is what lets the eventual paper claim a quantitative
attribution-method ranking rather than a gallery of anecdotes.

## Numbers to cite in the paper

| Item | Value |
|---|---|
| Attribution methods implemented | 4 (Grad-CAM, Saliency, Integrated Gradients, Occlusion) |
| IG integration steps | 50, zero baseline |
| Occlusion window / stride | 16×16 / 8 |
| 5-panel comparison figures generated | 8 |
| Per-method galleries (IG, Occlusion) | full case list each |
| Checkpoint used | `v0.1-baseline`, epoch 15, val_bal_acc 0.7964 |

---

## Update (Week 6) — Migration to ResNet-50 v12recipe

As with Grad-CAM (see `week4.md`'s migration addendum), all three
remaining attribution methods and the comparison harness were rerun
against `resnet50_v12recipe` (epoch 14, val_bal_acc 0.8010, test_bal_acc
0.7496) rather than the original `v0.1-baseline` checkpoint, since the
project's official model changed starting Week 6.

### Bug found: inconsistent target-case source across methods

`gradcam.py`'s `__main__` had already been updated to load the two
purpose-built, balanced files
(`xai_targets_incorrect_melnv.json`/`xai_targets_correct.json` — see
`week4.md`), but `saliency.py`, `integrated_gradients.py`, `occlusion.py`,
and `compare_methods.py` were still calling `load_xai_targets()` with no
path argument, which defaults to the older, unbalanced
`xai_target_cases.json`. This meant Grad-CAM and the other three methods
were silently auditing *different sets of images* — undermining the whole
point of a side-by-side comparison harness. Fixed by pointing all four
scripts at the same two balanced JSON files Grad-CAM uses.

### Folder structure now matches Grad-CAM's correct/failures split

Following the same reasoning as Week 4's Grad-CAM folder split, saliency,
Integrated Gradients, and occlusion now each write to separate
`/failures` and `/correct` subfolders instead of one flat output
directory, so results across all four methods have the same, directly
comparable structure:
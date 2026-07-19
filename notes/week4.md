# Week 4 — Grad-CAM & First Heatmaps

## Decisions made

### Why Captum's `LayerGradCam` instead of a hand-rolled hook implementation
Captum gives a tested, maintained Grad-CAM implementation with the same
API surface as the other three attribution methods planned for Week 5
(`Saliency`, `IntegratedGradients`, `Occlusion`). Writing all four methods
against one library means `compare_methods.py` (Week 5) can call them
interchangeably instead of reconciling four different attribution formats.

### Why `layer4[-1].conv2` as the target layer
`DermaNet`'s backbone is stored as an `nn.Sequential` of
`(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool)`.
`avgpool` is index `-1`, which makes `layer4` index `-2` — not `-3`, which
is `layer3`. Getting this off by one silently produces a heatmap from the
wrong resolution stage (28×28 instead of 7×7 feature maps), which still
*looks* plausible after upsampling but is attributing to the wrong depth
of the network. This is called out explicitly in `gradcam.py` to stop it
from being re-broken later.

### Why heatmaps are split into `correct/` and `failures/` folders
The synopsis Week 4 task is explicitly "visualise attributions on correct
vs incorrect cases" — keeping them in separate folders (rather than one
folder with correctness encoded only in the filename) makes it possible to
eyeball the two populations side by side without cross-referencing the CSV.

### Why the failure heatmaps run against `xai_target_cases.json` from Week 3
Using the same fixed, pre-saved list of 20 high-confidence wrong
predictions (rather than re-mining new failures now) keeps every XAI
method in Weeks 4–9 auditing the *same* cases, so results are directly
comparable across methods and across weeks.

## Key findings

- Ran Grad-CAM on the locked `v0.1-baseline` checkpoint (epoch 15,
  val_bal_acc 0.7964), producing **13 heatmaps for correctly-classified
  cases** and **20 heatmaps for the fixed Week 3 failure list**.
- The 20 failure cases split into two error patterns:
  - **7 cases**: true `mel`, predicted `nv` — the single most clinically
    dangerous error mode in this dataset (melanoma missed as a benign nevus).
  - **13 cases**: true `bkl`, predicted variously as `mel`, `akiec`, or
    `bcc` — a different, less life-threatening confusion, but still
    informative about which lesion types visually overlap for the model.
- Average confidence on the 7 mel→nv failures logged in
  `gradcam_results.csv` was **~0.69** — high enough that these are
  genuinely confident mistakes, not borderline calls.
- Built `src/explain/utils.py` as shared infrastructure (image loading,
  normalization, overlay generation, model/checkpoint loading with a
  backward-compatible key remapper for older checkpoint formats) so that
  Week 5's three additional methods don't duplicate this code.

## What this means for Week 5

Grad-CAM alone answers "is the model looking at the lesion in *this*
case?" but not whether that holds up across attribution methods that
work differently (gradient-based vs. perturbation-based). Week 5 adds
saliency, Integrated Gradients, and occlusion on the same cases and
builds one harness to compare all four side by side.

## Numbers to cite in the paper

| Item | Value |
|---|---|
| Correct-case heatmaps generated | 13 |
| Failure-case heatmaps generated | 20 |
| mel→nv failures (logged) | 7, avg. confidence ≈ 0.69 |
| bkl→{mel,akiec,bcc} failures | 13 |
| Checkpoint used | `v0.1-baseline`, epoch 15, val_bal_acc 0.7964 |
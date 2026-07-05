# Week 2 Log

## Decisions and why

**Split on `lesion_id`, not `image_id`.** The same lesion has multiple
photos; an image-level split risks leaking near-duplicates across
train/test, inflating accuracy without real generalization.

**Augmentations already in `dataset.py`:** flips + rotation (lesions have
no canonical orientation), color jitter (device/lighting variation),
random crop after resize.

**Metadata technique: late fusion this week.** Lowest effort, sets up FiLM
as a clean Week 4 ablation. Full reasoning in `notes/metadata_research.md`.

**Missing-age imputation: overall train median, not per-class median.**
Per-class median needs the true label to pick which median to use — not
valid for a genuinely unseen sample. Only affects ~0.5% of rows, but it's
the version that generalizes.

**Headline metric: balanced accuracy, not plain accuracy.** `nv` is
roughly half the dataset; balanced accuracy prevents a majority-class
model from looking artificially good.

## Leakage check — proof

> Paste your real `test_split.py` output here (from Step 10).

## Supervisor summary (3 lines)

Built the end-to-end pipeline this week: metadata encoder (18-d, late
fusion), ResNet-18 model with a togglable metadata-fusion head, full
train/evaluate loop with class-weighted loss and balanced-accuracy
checkpointing. Dry-run verified end-to-end in both modes with no crashes.
Next: run the real baseline and fusion experiments on the full dataset.
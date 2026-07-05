# Metadata Integration Research — Week 2, Day 3

Three techniques considered for folding age/sex/localization into the model.

## Rank 1 — Late fusion (concatenation) — implementing this week
Effort: low | Expected gain: +2-4% balanced accuracy

Extract the 512-d image feature from ResNet-18's penultimate layer, encode
metadata into an 18-d vector, concatenate into 530-d, feed into the
classifier. Melanoma peaks at age 45-65 and rarely appears on palms/soles;
vascular lesions concentrate on the lower extremities; dermatofibroma skews
~70% female in HAM10000 — metadata carries real diagnostic signal the image
alone doesn't encode.

## Rank 2 — Metadata-conditioned attention (FiLM) — Week 4 ablation
Effort: medium | Expected gain: +3-6% balanced accuracy

Instead of concatenating at the end, metadata modulates the feature
*extraction* itself (scale-and-shift on the feature map before pooling).
References: Perez et al. 2018, "FiLM: Visual Reasoning with a General
Conditioning Layer" (arXiv:1709.07871); Kawahara et al. 2019, "Seven-point
checklist and skin lesion classification using multitask deep learning."
Doing late fusion first, then FiLM in Week 4, turns into a clean three-way
ablation: image-only vs. late fusion vs. FiLM.

## Rank 3 — Multi-task learning (auxiliary age prediction) — future work only
Effort: high | Expected gain: variable

Predict class + age simultaneously from a shared backbone. Not attempted:
requires careful loss-weight tuning (get it wrong, it hurts the main
objective), and age regression is noisier with ~0.5% of ages missing.
Logged as future work to show awareness without over-engineering.

## Supervisor summary (3 lines)
Researched three metadata-integration techniques: late fusion (implementing
this week), FiLM conditioning (Week 4 ablation), multi-task learning
(future work, tuning complexity). The 18-d vector encodes age as a
train-split z-score with a missing-value indicator, sex as binary, and
localization as one-hot across 15 categories.
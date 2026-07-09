# Week 3 — Baseline Training & Evaluation

## Decisions made

### Why AdamW over Adam
Adam with default settings accumulates large adaptive learning rates
for sparse features. AdamW decouples weight decay, which regularises
better on the small HAM10000 training set (8k images).

### Why CosineAnnealingLR
Avoids manual LR scheduling. Starts at 1e-4, decays smoothly to 1e-6.
The model makes large updates early and refines late. No step-decay
cliffs that could destabilise training.

### Why checkpoint by val_balanced_accuracy not val_loss
On imbalanced data, val_loss can decrease while the model collapses
to predicting the majority class. Balanced accuracy catches this.
Best checkpoint: epoch 15 (val_bal_acc=0.7964).

### Why the val→test gap exists (79.6% → 73.2%)
Val set has 1009 images — best-epoch selection has statistical noise.
6.5pp gap is within expected range. Test number is the honest one.

### Why metadata fusion underperformed on macro metrics
The df class has only 9 test samples. A 30pp F1 drop on 9 samples
equals 2-3 wrong predictions — not a model failure, a sample size
artefact. AUC improved (+0.7pp) and 6/7 classes improved.

## Key findings

- mel→nv is the most frequent error — clinically most dangerous
- High-confidence wrong predictions found — direct XAI audit targets
- Overfitting gap: train_loss=0.22, val_loss=0.65 — mild, expected
- bkl improved most from metadata (+11.9pp F1)

## What this means for Week 4

The failure cases (xai_target_cases.json) are the starting point
for Grad-CAM. The question is: are these mel→nv failures because
the lesions genuinely look similar, or because the model is looking
at a ruler/hair/artifact instead of the lesion? Week 4 will answer
this visually.

## Numbers to cite in the paper

| Metric | Baseline | + Metadata |
|--------|----------|------------|
| Bal acc | 0.7318 | 0.7168 |
| Macro F1 | 0.6869 | 0.6703 |
| Macro AUC | 0.9478 | 0.9549 |
| Best val epoch | 15 (0.7964) | 13 (0.7825) |
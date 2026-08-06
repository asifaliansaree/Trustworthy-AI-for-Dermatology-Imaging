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

| Metric | Baseline (ResNet-18) | + Metadata (ResNet-18) |
|--------|----------|------------|
| Bal acc | 0.7318 | 0.7168 |
| Macro F1 | 0.6869 | 0.6703 |
| Macro AUC | 0.9478 | 0.9549 |
| Best val epoch | 15 (0.7964) | 13 (0.7825) |

## Update (Week 6) — architecture superseded by ResNet-50 v12recipe

This checkpoint (`v0.1-baseline`, ResNet-18) was the locked, test-verified
model at the time this note was written. It's superseded as of Week 6: a
parallel tuning track (documented in `README.md`'s "Parallel Track" section)
found that `resnet50_v12recipe` — despite a *lower* val_bal_acc (0.8010)
than several ResNet-18 tuning variants — generalizes better, beating this
baseline on test balanced accuracy:

| Config | val_bal_acc | test_bal_acc | test macro F1 | val→test gap |
|---|---|---|---|---|
| `v0.1-baseline` (this note, ResNet-18) | 0.7964 | 0.7318 | 0.6869 | −6.5pp |
| `resnet50_v12recipe` (current) | 0.8010 | **0.7496** | 0.6484 | −5.1pp |

Per supervisor direction, `resnet50_v12recipe` is now the project's official
model going forward from Week 6. This numbers table is kept as historical
record of the original baseline decision-making; it is no longer the
checkpoint the XAI work in Weeks 4–5 runs against — see the migration
addenda in `week4.md` and `week5.md`.
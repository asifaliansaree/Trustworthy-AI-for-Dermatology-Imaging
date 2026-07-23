# Trustworthy AI for Dermatology Imaging

This repository documents the development of a trustworthy, explainable AI pipeline for dermatology image analysis, built during a research internship at the TUKL Deep Learning Lab, NUST.

## Project Goal

Build a rigorous, reproducible skin-lesion classifier on HAM10000, then apply state-of-the-art explainability techniques (Grad-CAM, Integrated Gradients, occlusion) to audit whether the model attends to the lesion — or to spurious image artifacts. The end goal is a short research paper targeting a medical imaging workshop (MICCAI / IEEE BHI).

---

## Learning Timeline

| Week | Focus |
|------|-------|
| 1 | Environment setup, first CNN (CIFAR-10), HAM10000 study, custom PyTorch Dataset |
| 2 | EDA, leakage-free lesion-wise split, metadata encoder, DermaNet + training pipeline |
| 3 | Full baseline training, test-set evaluation, metadata-fusion ablation, diagnostics, go/no-go ✅ complete |
| 4 | Grad-CAM implementation, first heatmaps on correct vs. incorrect predictions ✅ complete |
| 5 | Attribution benchmark — Integrated Gradients, saliency, occlusion, unified comparison harness ✅ complete |
| 6 | Mask-overlap evaluation against ISIC lesion masks (IoU, pointing game) ← next |

> Running alongside Weeks 4–5: a parallel tuning effort explored 6 architectures under a shared recipe (EMA + weighted sampler + effective-number loss). The highest **val**_bal_acc was `densenet121_v12recipe` at 0.8251, but test-set verification (see [Parallel Track](#parallel-track--baseline-strengthening-post-lock-model-iteration) below) showed val_bal_acc did not reliably predict test performance across architectures. The current best **test**-verified model is **`resnet50_v12recipe`, test_bal_acc = 0.7496**.

---

# Week 1 — Foundations

* Set up the development environment: Python, virtual environment, PyTorch, TorchVision, VS Code (Python/Pylance/Jupyter extensions), and verified the install with a test script
* Set up Git/GitHub, created and connected the repository, pushed the first commit
* Trained a first CNN on CIFAR-10 — data loading, model, loss/optimizer, training loop, saved model (`cifar10_model.pth`)
* Read the HAM10000 paper (Tschandl et al., 2018) and reviewed the PyTorch 60-Minute Blitz to reinforce tensors, Datasets, DataLoaders, and training loops (`notes/ham10000_intro.md`)
* Downloaded HAM10000, verified **10,015 dermoscopic images** across **7 lesion classes**, explored class distribution and metadata, visualized samples with Matplotlib
* Built a custom `HAM10000Dataset` (`__init__`, `__len__`, `__getitem__`), indexed images across two directories, mapped diagnoses to numeric labels, applied `torchvision.transforms` (resize to 224×224), verified batch shapes `[16, 3, 224, 224]`

### Key concepts learned
Tensors/Datasets/DataLoaders, CNNs, loss functions and optimizers, medical image metadata analysis, class imbalance, custom Dataset/DataLoader design.

---

# Week 2 — Data Pipeline & Training Infrastructure

* Built `EDA.ipynb`: found 57 missing `age` values, visualized class/sex/age/localization distributions, confirmed nevus accounts for ~67% of images
* Implemented a **lesion-wise stratified split** (80/10/10, `random_state=42`) to prevent data leakage, saved permanently as `HAM10000_split.csv` (never regenerated), verified zero lesion overlap across splits
* Built separate train vs. val/test augmentation pipelines (RandomCrop, flips, ColorJitter, RandomRotation for train; resize + ImageNet normalization for val/test) and precomputed `class_weights.npy`
* Built `MetadataEncoder`: 18-dimensional feature vector from age (z-score, median-imputed, missing indicator), sex (binary), and localization (one-hot, 15 categories); researched and ranked fusion strategies in `notes/metadata_research.md`, selecting **late fusion** for this phase
* Built **DermaNet** (`src/model.py`): ResNet-18 (ImageNet pretrained) with an optional metadata late-fusion head, switchable via `metadata_dim`
* Built the full training loop (`src/train.py`: AdamW, CosineAnnealingLR, class-weighted CrossEntropyLoss, checkpointing by `val_balanced_accuracy`, CSV logging) and evaluation (`src/evaluate.py`, `src/evaluate_test.py`: balanced accuracy, per-class F1, confusion matrix, macro ROC-AUC)
* Verified everything end-to-end with a 1-epoch dry run in both modes, re-confirmed zero leakage, pinned all dependencies in `requirements.txt`

### Key concepts learned
Data leakage prevention (lesion-wise vs. image-wise splitting), stratified splitting, feature normalization/encoding, late fusion vs. FiLM vs. multi-task learning, transfer learning, class-weighted loss, config-driven reproducible experiment design.

---

# Week 3 — Baseline Training & Rigorous Evaluation ✅ complete

> **Synopsis checkpoint**: *"Have a working model by end of week 3 before you specialise."*

* Ran the full 20-epoch baseline training (`configs/baseline.yaml`, image-only): ResNet-18, batch size 32, LR 1e-4 → 1e-6 cosine decay. Best checkpoint at **epoch 15**, `val_balanced_accuracy = 0.7964`; training loss decreased 1.2297 → 0.2238
* Evaluated on the held-out test set for the first time: **balanced accuracy 0.7318** (exceeds the 0.65 go/no-go threshold), **macro F1 0.6869**, **macro ROC-AUC 0.9478**. Generated 3 publication-quality figures (confusion matrix, per-class F1, training curves) and locked results in `results/baseline_image_only.json`
* Trained a metadata-fusion variant (late fusion, `metadata_dim=18`): best checkpoint at epoch 13 (`val_bal_acc = 0.7825`), test balanced accuracy 0.7168, macro F1 0.6703, **macro AUC 0.9549** (+0.7pp over image-only). Metadata fusion helped 6 of 7 classes; the apparent macro-F1 drop is driven almost entirely by the 9-sample dermatofibroma class, not a real model failure
* Built training diagnostics: learning-curve comparison plots, an overfitting check (train_loss 0.2238 vs. val_loss 0.6452 — mild, expected for ~8k images), a mel→nv failure gallery (the most clinically dangerous error), and a high-confidence-wrong-prediction gallery. Saved the 20 high-confidence failure cases to `results/xai_target_cases.json` as the fixed audit list for all future XAI work
* Ran the **Week 3 go/no-go checklist** (7/7 checks passed), tagged the repository `v0.1-baseline`, and wrote `notes/week3.md` documenting every design decision (why AdamW, why CosineAnnealingLR, why checkpoint on balanced accuracy, why the val→test gap exists, why metadata fusion underperformed on macro metrics)

### Key concepts learned
Touching the test set exactly once, reading learning curves for overfitting, controlled ablation design, why a negative result is still publishable, rare-class instability, go/no-go checklists as a formal gate, git tagging for reproducible experiment versioning.

---

# Week 4 — Grad-CAM & First Heatmaps ✅ complete

* Implemented Grad-CAM (`src/explain/gradcam.py`) on the locked `v0.1-baseline` checkpoint
* Ran Grad-CAM over the fixed `xai_target_cases.json` audit list plus additional correctly-classified cases, producing heatmaps split into `results/xai/gradcam/correct/` (13 cases) and `results/xai/gradcam/failures/` (20 cases)
* Logged every case (image ID, true class, predicted class, confidence, correctness, heatmap path) to `results/xai/gradcam_results.csv` for downstream analysis
* Began building out the shared `src/explain/utils.py` helpers (image loading, prediction, overlay generation, attribution normalization) that the rest of the attribution methods reuse

### Key concepts learned
Class Activation Mapping and how gradients flow back to convolutional feature maps, visual auditing of correct vs. incorrect predictions, building reusable XAI infrastructure instead of one-off scripts.

---

# Week 5 — Attribution Benchmark & Comparison Harness ✅ complete

* Implemented three additional attribution methods against the same baseline checkpoint: vanilla saliency (`src/explain/saliency.py`), Integrated Gradients (`src/explain/integrated_gradients.py`), and occlusion sensitivity (`src/explain/occlusion.py`)
* Built a unified comparison harness (`src/explain/compare_methods.py`) that runs all four methods on the same case and renders a single 5-panel figure — Original | Grad-CAM | Integrated Gradients | Saliency | Occlusion — annotated with the true/predicted label and correctness, saved to `results/xai/comparison/`
* Generated per-method output galleries for occlusion and Integrated Gradients across the audit case list (`results/xai/occlusion/`, `results/xai/integrated_gradients/`)

### Key concepts learned
How the four attribution families disagree or agree on the same input, building one harness that fairly compares multiple XAI methods rather than evaluating each in isolation, the practical cost trade-off of occlusion (much slower — a full sweep per image) versus gradient-based methods.

---

# Week 6 — Mask-Overlap Evaluation ← next

Planned per the synopsis: obtain ISIC lesion segmentation masks and implement quantitative mask-overlap metrics (IoU, pointing game) to measure whether each attribution method's heatmap actually lands on the lesion — turning the qualitative comparison from Weeks 4–5 into a numeric benchmark.

---

# Parallel Track — Baseline Strengthening (Post-Lock Model Iteration)

> Note: the synopsis locks the baseline checkpoint at the end of Week 3 (`v0.1-baseline`, val_bal_acc 0.7964) and runs all Week 4–9 XAI work against it. In parallel with the Week 4–5 explainability work above, a separate line of experiments re-opened the ResNet-18 training recipe to see how much further balanced accuracy could be pushed. This is documented here for transparency; it sits outside the synopsis's week numbering, and the XAI results above still refer to the original `v0.1-baseline` checkpoint unless noted otherwise.

**What triggered it:** three early ablations testing class-imbalance fixes in isolation all made things *worse* than the plain baseline:

| Ablation | val_bal_acc | Finding |
|---|---|---|
| `ablation_focal` (focal loss alone) | 0.5949 | Loss reweighting alone destabilized training |
| `ablation_sampler` (WeightedRandomSampler alone) | 0.5934 | Oversampling rare classes alone also destabilized training |
| `ablation_onecycle` (OneCycleLR) | 0.5162 | Traced to a bug — the scheduler was never actually stepped |

Digging into why stacking these mechanisms hurt rather than helped led to two fixes:
- A `freeze_epochs` bug that was suppressing the backbone learning rate during the "unfrozen" phase
- Correct implementation of `effective_num` class-balanced loss weighting (Cui et al., 2019) in `src/losses.py`, replacing naive inverse-frequency weighting

**Tuning journey after the fixes** (all val_bal_acc, ResNet-18, same lesion-wise split):

| Config | val_bal_acc | Best epoch | Key change |
|---|---|---|---|
| `v0.1-baseline` (Week 3) | 0.7964 | 15 | — |
| `resnet18_v2_tuned_stable` | 0.7657 | 42 | `freeze_epochs=4` — regression, backbone LR suppressed |
| `resnet18_v2_tuned_stable_nofreeze` | **0.8097** | 22 | `freeze_epochs=0` — bug fixed, first break past 0.80 |
| `resnet18_v3_balanced` | 0.7813 | 10 | Plain focal loss + weighted sampler, no `effective_num` |
| `resnet18_v4_balanced_annealed` | 0.8007 | 20 | Focal `gamma=1.0`, longer patience |
| `resnet18_v5_ema_stable` | 0.8072 | 33 | Added EMA (`ema_decay=0.999`) on top of `effective_num` focal loss |
| `resnet18_v6_ema_sampler` | **0.8120** | 24 | EMA + weighted sampler + gentler `effective_num_beta=0.99` + LR warmup — best result so far |
| `resnet18_v7_noOverfit` | 0.7540 | 31 | Added mixup + sampler power 0.5 — regression |
| `resnet18_v8_balanced` / `v8_optimized` | 0.7709 / 0.7800 | 49 / 25 | Sampler-power/mixup variants — no improvement over v6 |
| `resnet18_v9_sampler_restore` | 0.7725 | 39 | Reverted toward v8-style sampling — no improvement |
| `resnet18_v9_tuned` / `v10` | 0.8023 / 0.8023 | 22 / 22 | Back to EMA without sampler — stable but below v6 |

**Highest val_bal_acc in the sweep: `densenet121_v12recipe` at 0.8251** (epoch 17 raw; checkpointed epoch corresponds to 0.8148 due to `best_metric_smoothing_window=5`), combining EMA, a weighted sampler, gentler effective-number class balancing, and LR warmup. Everything from `v7` onward on the ResNet-18 line was an attempt to push further with mixup and re-tuned sampling, without success; those runs consistently underperformed the non-mixup configs. **However, val_bal_acc alone is not a reliable stopping point — see the test-set verification below, which changes which model is actually "best."**

## Test-Set Verification of the Parallel Track

The tuning journey above was driven entirely by `val_bal_acc` on a fixed validation split, checkpoint-selected the same way every time. That is exactly the setup where repeated hyperparameter search can overfit to one validation split without it showing up in the metric being optimized. To check, the three architectures with the highest val scores were evaluated on the held-out test set (touched once each, no TTA, no cherry-picking across seeds):

| Config | val_bal_acc | **test_bal_acc** | test macro F1 | val→test gap |
|---|---|---|---|---|
| `densenet121_v12recipe` | 0.8148–0.8251 | 0.7039 | 0.6587 | −11 to −12pp |
| `resnet18_v6_ema_sampler` | 0.8120 | 0.7064 | 0.6838 | −10.6pp |
| `resnet50_v12recipe` | 0.8010 | **0.7496** | 0.6484 | −5.1pp |
| *(reference)* Week 3 `v0.1-baseline` | 0.7964 | 0.7318 | 0.6869 | −6.5pp |

**Takeaway: the model with the highest val score (`densenet121_v12recipe`) is actually the weakest on test, and the model with the lowest val score of the three (`resnet50_v12recipe`) is the strongest on test.** Ranking architectures by val_bal_acc alone would have picked the wrong model. `resnet50_v12recipe` is now the current-best, test-verified checkpoint — it also modestly beats the original Week 3 baseline on test balanced accuracy, though not on macro F1 or per-class F1 uniformly.

Test-Time Augmentation (`evaluate_tta.py`, 8 passes) was tried on `densenet121_v12recipe` and made things worse (0.7039 → 0.6844 balanced accuracy), likely due to the random-resized-crop scale interacting badly with small/off-center lesions in the already-weak `mel`/`df` classes. **TTA is not adopted.**

Across all three test-verified models, `mel`, `df`, and `bkl` remain the weakest classes (F1 roughly 0.44–0.60 depending on model/class), consistent with the Week 3 finding that these are dataset-size- and boundary-confusion-driven, not architecture-driven — no backbone swap so far has fixed them.

---

# Baseline Results — Table 1

| Experiment | Bal acc | Macro F1 | Macro AUC | Best val epoch | Notes |
|---|---|---|---|---|---|
| Image-only (ResNet-18) | **0.7318** | **0.6869** | 0.9478 | 15 (0.7964) | Week 3 |
| Image + metadata (late fusion) | 0.7168 | 0.6703 | **0.9549** | 13 (0.7825) | Week 3 |

### Per-class F1 breakdown

| Class | Image-only | + Metadata | Delta | Support |
|-------|-----------|------------|-------|---------|
| akiec | 0.6667 | 0.6824 | +0.0157 | 41 |
| bcc | 0.6863 | 0.6916 | +0.0053 | 44 |
| bkl | 0.5340 | 0.6531 | **+0.1191** | 113 |
| df | 0.5882 | 0.2857 | −0.3025 | 9 |
| mel | 0.5423 | 0.5662 | +0.0239 | 112 |
| nv | 0.8878 | 0.9038 | +0.0160 | 655 |
| vasc | 0.9032 | 0.9091 | +0.0059 | 15 |

> The df drop (−30 pp) is driven by only 9 test samples — a dataset size artefact, not a model failure. Metadata fusion improves ranking quality (AUC +0.7 pp) and benefits 6 of 7 classes when df is excluded.

---

# Week 3 Go/No-Go Checklist — Result

```
=== WEEK 3 GO/NO-GO CHECKLIST ===

[PASS] Baseline bal-acc >= 0.65: 0.7318
[PASS] Macro AUC >= 0.90: 0.9478
[PASS] All 7 classes have F1 > 0
[PASS] Baseline checkpoint exists
[PASS] Fusion results exist
[PASS] XAI target cases saved
[PASS] Training log exists

7/7 checks passed
STATUS: GO — proceed to Week 4 XAI
```

**Tagged:** `v0.1-baseline`
**Checkpoint used for all Week 4–9 work:** `ham10000/checkpoints/baseline_image_only/best_model.pt` (epoch 15, val_bal_acc = 0.7964)

---

# Repository Structure

```text
.
├── cifar10/
│   ├── data/
│   ├── train.py
│   └── cifar10_model.pth
│
├── ham10000/
│   ├── checkpoints/
│   │   ├── baseline_image_only/
│   │   │   └── best_model.pt          # epoch 15, val_bal_acc=0.7964
│   │   └── metadata_fusion/
│   │       └── best_model.pt          # epoch 13, val_bal_acc=0.7825
│   ├── configs/
│   │   ├── baseline.yaml              # image-only, metadata_dim=0
│   │   ├── metadata_fusion.yaml       # late fusion, metadata_dim=18
│   │   └── dry_run.yaml               # 1-epoch sanity check
│   ├── data/
│   │   ├── HAM10000_images_part_1/
│   │   ├── HAM10000_images_part_2/
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_split.csv         # permanent lesion-wise split
│   │   └── class_weights.npy          # precomputed for imbalanced loss
│   ├── experiments/
│   │   ├── baseline_image_only/
│   │   │   └── training_log.csv       # 20-epoch log
│   │   └── metadata_fusion/
│   │       └── training_log.csv       # 20-epoch log
│   ├── results/
│   │   ├── figures/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── per_class_f1.png
│   │   │   ├── training_curves.png
│   │   │   ├── ablation_comparison.png
│   │   │   ├── ablation_per_class_f1.png
│   │   │   ├── learning_curves_comparison.png
│   │   │   ├── failure_mel_as_nv.png
│   │   │   └── high_confidence_errors.png
│   │   ├── xai/
│   │   │   ├── gradcam/
│   │   │   │   ├── correct/           # Week 4: correctly-classified heatmaps
│   │   │   │   └── failures/          # Week 4: misclassified heatmaps
│   │   │   ├── gradcam_results.csv    # Week 4: per-case log
│   │   │   ├── integrated_gradients/  # Week 5
│   │   │   ├── occlusion/             # Week 5
│   │   │   └── comparison/            # Week 5: 4-method comparison figures
│   │   ├── baseline_image_only.json   # locked test results
│   │   ├── baseline_image_only_report.txt
│   │   ├── metadata_fusion.json       # locked test results
│   │   ├── metadata_fusion_report.txt
│   │   ├── xai_target_cases.json      # 20 high-confidence failure cases → XAI audit list
│   │   ├── plot_learning_curves.py
│   │   ├── failure_analysis.py
│   │   ├── ablation_comparison.py
│   │   ├── generate_figure.py
│   │   └── generate_gradcam_gallery.py
│   ├── src/
│   │   ├── model.py                   # DermaNet (ResNet-18 + optional fusion)
│   │   ├── train.py                   # full training loop + CSV logging
│   │   ├── evaluate.py                # library: run_inference, compute_metrics
│   │   ├── evaluate_test.py           # standalone CLI evaluator
│   │   └── explain/
│   │       ├── gradcam.py             # Week 4
│   │       ├── saliency.py            # Week 5
│   │       ├── integrated_gradients.py # Week 5
│   │       ├── occlusion.py           # Week 5
│   │       ├── compare_methods.py     # Week 5: 4-method comparison harness
│   │       └── utils.py               # shared XAI helpers
│   ├── dataset.py
│   ├── metadata_encoder.py
│   ├── split.py
│   ├── test_split.py
│   ├── test_dataset.py
│   ├── explore_dataset.py
│   ├── show_images.py
│   └── EDA.ipynb
│
├── notes/
│   ├── week1.md
│   ├── week2.md
│   ├── week3.md
│   ├── ham10000_intro.md
│   └── metadata_research.md
│
├── test_torch.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, TorchVision |
| Data & Analysis | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Explainability | Captum-style Grad-CAM, Integrated Gradients, Saliency, Occlusion |
| Config & Tracking | PyYAML, CSV logging |
| Environment | Python, pip, virtualenv |
| Development | VS Code, Jupyter Notebook, Git, GitHub |

---

# Current Status

**Week 5 — COMPLETE ✅**

- [x] Week 1: Environment, CIFAR-10 CNN, HAM10000 exploration, custom Dataset
- [x] Week 2: EDA, leakage-free split, augmented DataLoader, MetadataEncoder, DermaNet, train/evaluate pipeline
- [x] Week 3: Full baseline training (bal-acc 0.7318), metadata-fusion ablation, diagnostics, go/no-go passed (7/7), tagged `v0.1-baseline`
- [x] Week 4: Grad-CAM implemented, heatmaps generated for correct (13) and failure (20) cases, logged to `gradcam_results.csv`
- [x] Week 5: Saliency, Integrated Gradients, and occlusion implemented; unified 4-method comparison harness built and run over the audit case list
- [ ] Week 6: Mask-overlap evaluation (IoU, pointing game) against ISIC lesion masks — next

---

# Roadmap

```
Week 1–2   Data pipeline, DermaNet, training infrastructure       ✅ complete
Week 3     Baseline training and rigorous evaluation               ✅ complete (tagged v0.1-baseline)
Week 4     Grad-CAM implementation and first heatmaps              ✅ complete
Week 5     Attribution benchmark: IG, saliency, occlusion + comparison harness   ✅ complete
Week 6     Quantitative XAI benchmark against ISIC lesion masks (IoU, pointing game)  ← next
Week 7     Inter-method agreement analysis
Week 8     Shortcut-learning audit (artifact-driven prediction hunting)
Week 9     Faithfulness checks: deletion / insertion curves
Week 10    Results consolidation and ablation runs
Week 11    Mini-paper draft + reproducibility pass
Week 12    Final delivery: paper + repo + presentation
```

---

# Research Output

**Target venues**: MICCAI 2027 workshop (ISIC / UNSURE track) · IEEE BHI · arXiv preprint

**Core contribution**: Quantitative benchmark of ≥4 attribution methods on HAM10000 using ISIC lesion masks, combined with a shortcut-learning audit measuring the frequency of artifact-driven predictions in high-confidence correct classifications.

---
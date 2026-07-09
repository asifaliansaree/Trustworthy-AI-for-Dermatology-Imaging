# Trustworthy AI for Dermatology Imaging
 
This repository documents the development of a trustworthy, explainable AI pipeline for dermatology image analysis, built during a research internship at the TUKL Deep Learning Lab, NUST.
 
## Project Goal
 
Build a rigorous, reproducible skin-lesion classifier on HAM10000, then apply state-of-the-art explainability techniques (Grad-CAM, Integrated Gradients, occlusion) to audit whether the model attends to the lesion — or to spurious image artifacts. The end goal is a short research paper targeting a medical imaging workshop (MICCAI / IEEE BHI).
 
---
 
## Learning Timeline
 
### Week 1 — Foundations
| Day | Focus |
|-----|-------|
| 1 | Development environment setup and Git/GitHub |
| 2 | First CNN on CIFAR-10 |
| 3 | HAM10000 paper study and PyTorch review |
| 4 | HAM10000 dataset exploration and visualization |
| 5 | Custom PyTorch Dataset and DataLoader for HAM10000 |
 
### Week 2 — Data Pipeline & Training Infrastructure
| Day | Focus |
|-----|-------|
| 1 | Exploratory Data Analysis (EDA) |
| 2 | Lesion-wise split and augmented DataLoader |
| 3 | Metadata analysis and integration research |
| 4 | End-to-end training pipeline (DermaNet, train, evaluate) |
| 5 | Repository polish and leakage re-verification |
 
### Week 3 — Baseline Training & Rigorous Evaluation ✅ complete
| Day | Focus |
|-----|-------|
| 1 | Full 20-epoch baseline training run (image-only) |
| 2 | Test-set evaluation: balanced accuracy, per-class F1, confusion matrix, ROC-AUC |
| 3 | Metadata-fusion ablation run, evaluation, and comparison figures |
| 4 | Training diagnostics, learning curves, failure case analysis |
| 5 | Go/no-go review, baseline tag `v0.1-baseline`, supervisor presentation |
 
### Week 4 — Grad-CAM & First Heatmaps ← next
| Day | Focus |
|-----|-------|
| — | Implement Grad-CAM, run on `xai_target_cases.json`, generate first heatmaps |
 
---
 
# Week 1
 
## Day 1: Environment Setup
 
### Environment Setup
* Installed Python and configured a virtual environment
* Installed PyTorch, TorchVision, and supporting libraries
* Set up Visual Studio Code with Python, Pylance, and Jupyter extensions
* Verified the PyTorch installation using a test script
### Git & GitHub Setup
* Installed and configured Git
* Created the GitHub repository
* Initialized a local Git repository
* Connected the local repository to GitHub
* Created the first commit and pushed the project successfully
---
 
## Day 2: First Deep Learning Project (CIFAR-10)
 
* Trained a Convolutional Neural Network (CNN) using the CIFAR-10 dataset
* Implemented data loading with PyTorch DataLoader
* Built and trained a CNN model
* Used a loss function and optimizer for training
* Observed decreasing training loss across epochs
* Saved the trained model (`cifar10_model.pth`)
* Documented key concepts in study notes
### Key Concepts Learned
* Tensors, Datasets, DataLoaders
* Convolutional Neural Networks (CNNs)
* Loss Functions, Optimizers, Training Loops
* Model saving and loading
---
 
## Day 3: HAM10000 Dataset Study
 
* Read the HAM10000 research paper (Tschandl et al., 2018)
* Learned why the dataset was created and studied the seven skin lesion classes
* Explored the metadata and dataset organization
* Created study notes (`notes/ham10000_intro.md`)
### PyTorch Review
* Reviewed the PyTorch 60-Minute Blitz tutorial
* Reinforced understanding of tensors, datasets, DataLoaders, neural networks, and training loops
---
 
## Day 4: HAM10000 Dataset Exploration
 
* Downloaded the HAM10000 dataset
* Organized images and metadata into the project structure
* Loaded the metadata using Pandas
* Verified the dataset contains **10,015 dermoscopic images**
* Analyzed the distribution of the **7 skin lesion classes**
* Identified missing values in the metadata
* Displayed sample skin lesion images with their diagnosis labels using Matplotlib
### Key Concepts Learned
* Medical image datasets and metadata analysis
* Class imbalance in medical datasets
* Image visualization with Matplotlib
---
 
## Day 5: Custom PyTorch Dataset
 
* Created a custom `HAM10000Dataset` class by inheriting from `torch.utils.data.Dataset`
* Loaded the HAM10000 metadata from the CSV file
* Indexed images stored across two directories
* Created a mapping from diagnosis labels to numeric class labels
* Applied image preprocessing using `torchvision.transforms` (resize to **224 × 224**)
* Implemented `__init__()`, `__len__()`, `__getitem__()`
* Verified batch dimensions: `[16, 3, 224, 224]` images and `[16]` labels
### Key Concepts Learned
* Custom PyTorch Dataset and DataLoader design
* Label encoding, image preprocessing, batch loading
---
 
# Week 2
 
## Day 1: Exploratory Data Analysis (EDA)
 
* Created `EDA.ipynb` — an interactive EDA notebook
* Found **57 missing values in the `age` column**
* Visualized class distribution, sex ratio, age distribution, and lesion localization frequencies
* Displayed representative sample images from each of the 7 diagnosis classes
* Documented class imbalance observations (nevus accounts for ~67% of images)
### Key Concepts Learned
* EDA methodology for medical image datasets
* Class imbalance analysis and demographic interpretation
---
 
## Day 2: Lesion-wise Dataset Split & Augmented DataLoader
 
* Performed duplicate lesion analysis using `lesion_id` — identified lesions with multiple images
* Implemented a **lesion-wise stratified split** (80% train / 10% val / 10% test, `random_state=42`) to prevent data leakage
* Saved the split permanently as `HAM10000_split.csv` — never regenerated, always loaded
* Updated `HAM10000Dataset` to accept a `split` argument and load from the CSV
* Implemented separate augmentation pipelines:
  * **Train**: RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation
  * **Val / Test**: Resize + Normalize only
* Applied ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
* Verified zero lesion overlap across splits with assertion checks
* Computed and saved `class_weights.npy` for class-imbalance-aware training
### Key Concepts Learned
* Data leakage prevention in medical image analysis
* Lesion-wise vs. image-wise splitting — why it matters
* Stratified splitting, augmentation pipelines, class weight computation
---
 
## Day 3: Metadata Analysis & Integration Research
 
* Built `MetadataEncoder` converting age, sex, and localization into an **18-dimensional normalized feature vector**
  * Age: z-score normalized, per-class median imputation for 57 missing values, binary missing indicator
  * Sex: binary encoded (male=1, female=0, unknown=0.5)
  * Localization: one-hot encoded across 15 anatomical categories
* Added a metadata statistics section to `EDA.ipynb` — per-class age distributions, sex ratios, top localizations
* Researched and ranked three metadata integration techniques in `notes/metadata_research.md`:
  1. **Late fusion** (selected for this week) — concatenate image + metadata features before classifier head
  2. **FiLM conditioning** (planned Week 4 ablation) — modulate feature maps with metadata scale/shift
  3. **Multi-task learning** (logged as future work) — auxiliary supervision on metadata targets
* Rationale: late fusion is low-effort, directly compatible with the existing DermaNet design, and produces a clean ablation comparison for the paper
### Key Concepts Learned
* Feature normalization (z-score), one-hot encoding, missing value imputation
* Target leakage through engineered features
* Late fusion vs. FiLM conditioning vs. multi-task learning
* Effort-vs-gain prioritization in ML research planning
---
 
## Day 4: End-to-End Training Pipeline
 
* Built **DermaNet** (`src/model.py`): ResNet-18 backbone (ImageNet pretrained) with an optional metadata late-fusion head, switchable via a single `metadata_dim` config value
  * Image-only: `metadata_dim=0` → 512-d → Dropout(0.3) → Linear(7)
  * Metadata fusion: `metadata_dim=18` → 530-d → Dropout(0.3) → Linear(7)
* Implemented the full training loop (`src/train.py`):
  * Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
  * Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-6)
  * Loss: CrossEntropyLoss with precomputed class weights
  * Checkpointing: saves best model by `val_balanced_accuracy` (not val_loss)
  * Logging: epoch-level CSV log (`training_log.csv`) — no external tools required
* Implemented `src/evaluate.py` and `src/evaluate_test.py`:
  * Balanced accuracy (mean per-class recall)
  * Per-class F1-score
  * Normalized confusion matrix
  * Macro ROC-AUC (one-vs-rest)
* All hyperparameters centralized in `configs/baseline.yaml` — zero magic numbers in code
* Verified end-to-end with a 1-epoch dry run in both image-only and metadata-fusion modes — no crashes, correct tensor shapes
### Key Concepts Learned
* Transfer learning with a pretrained CNN backbone
* Class-weighted loss for imbalanced multi-class classification
* Why balanced accuracy is the correct metric for HAM10000
* Config-driven, reproducible experiment design
* Saving checkpoints by the metric that matters, not the loss
---
 
## Day 5: Repository Polish & Leakage Re-Verification
 
* Re-ran lesion leakage assertion check — confirmed 0 shared `lesion_id` values across all splits
* Pinned all dependency versions in `requirements.txt`
* Documented Week 2 design decisions and rationale in `notes/week2.md`
* Updated repository structure and README
### Key Concepts Learned
* Reproducibility through pinned dependencies
* Documenting design decisions for future reference and supervisor communication
---
 
# Week 3
 
## Day 1: Full Baseline Training Run
 
> **Synopsis checkpoint**: *"Have a working model by end of week 3 before you specialise."*
 
* Launched full 20-epoch training run with `configs/baseline.yaml` (image-only mode)
* Training configuration:
  * Architecture: ResNet-18 (ImageNet pretrained)
  * Epochs: 20, Batch size: 32, LR: 1e-4 (cosine decay to 1e-6)
  * Loss: CrossEntropyLoss with class weights
  * Split: lesion-wise 80/10/10, `random_state=42`
* Best checkpoint saved at **epoch 15** with `val_balanced_accuracy = 0.7964`
* Training loss decreased consistently from 1.2297 → 0.2238 across 20 epochs
### Key Concepts Learned
* Full training loop execution on a real medical imaging dataset
* Monitoring val_balanced_accuracy as the primary training signal (not val_loss)
* CosineAnnealingLR behaviour across 20 epochs
---
 
## Day 2: Test-Set Evaluation & Publication Figures
 
* Evaluated the best checkpoint (epoch 15) on the **held-out test set for the first time**
* Computed all four paper metrics using `src/evaluate_test.py`:
  * Test balanced accuracy: **0.7318** — exceeds the 0.65 go/no-go threshold
  * Test macro F1: **0.6869**
  * Test macro ROC-AUC: **0.9478** — strong probability ranking across all 7 classes
* Generated 3 publication-quality figures (300 DPI) saved to `results/figures/`:
  * `confusion_matrix.png` — normalized, reveals mel→nv as the most frequent error
  * `per_class_f1.png` — color-coded by performance tier (red / blue / green)
  * `training_curves.png` — dual-axis with best epoch annotated
* Reconstructed `training_log.csv` from terminal output and fixed `train.py` to write CSV logs automatically on every future run
* All results locked in `results/baseline_image_only.json` — test set not to be re-evaluated
### Key Concepts Learned
* Why the test set is touched exactly once, after all design decisions are final
* Interpreting confusion matrices for clinical safety (mel→nv misclassification)
* Publication-quality figure generation with Matplotlib at 300 DPI
---
 
## Day 3: Metadata Fusion Ablation
 
* Trained a second DermaNet variant with **late fusion of 18-dimensional patient metadata** (age, sex, anatomical localization) using `configs/metadata_fusion.yaml`
  * Same architecture, optimizer, scheduler, and epochs as the baseline — only `metadata_dim=18` differs
  * Best checkpoint saved at **epoch 13** with `val_balanced_accuracy = 0.7825`
* Evaluated the fusion checkpoint on the held-out test set:
  * Test balanced accuracy: **0.7168**
  * Test macro F1: **0.6703**
  * Test macro ROC-AUC: **0.9549** — higher than image-only baseline (+0.7 pp)
* Generated ablation comparison figures:
  * `results/figures/ablation_comparison.png` — grouped bar chart of all three metrics
  * `results/figures/ablation_per_class_f1.png` — per-class F1 side-by-side for both experiments
* Fixed `evaluate_test.py` to auto-name output JSON from `experiment_name` in config — prevents result files overwriting each other
* Key finding: metadata fusion improved AUC and benefited 5 of 7 classes (bkl +11.9 pp, mel +2.4 pp) but hurt the dermatofibroma class (−30.3 pp on only 9 test samples), dragging macro averages below baseline — a rare-class noise effect, not a model failure
### Key Concepts Learned
* Controlled ablation design — changing exactly one variable between experiments
* Why a negative result is still a valid and publishable paper finding
* Rare-class instability: when test sets are too small, per-class metrics are unreliable
* Auto-naming experiment outputs from config to prevent file collisions
---
 
## Day 4: Training Diagnostics & Failure Case Analysis
 
* Built `results/plot_learning_curves.py` — plots validation balanced accuracy and train/val loss for both experiments side by side, with best-epoch markers annotated
  * Saved to `results/figures/learning_curves_comparison.png`
  * Overfitting diagnostic: baseline `train_loss=0.2238`, `val_loss=0.6452` → gap of 0.42, classified as mild overfitting given the ~8k-image training set — this justifies the `Dropout(0.3)` choice and motivates future data augmentation work
* Built `results/failure_analysis.py` — runs the baseline checkpoint over the test set and inspects every wrong prediction:
  * Identified all **mel → nv** misclassifications (the most clinically dangerous error) and saved a labeled image gallery to `results/figures/failure_mel_as_nv.png`
  * Identified **high-confidence wrong predictions** (confidence ≥ 0.80) — cases where the model is both wrong and overconfident — saved to `results/figures/high_confidence_errors.png`
  * Saved the full set of high-confidence failure cases (image ID, true label, predicted label, confidence) to `results/xai_target_cases.json` — this is the fixed audit list that Week 4's Grad-CAM run will iterate over
* Per-class error breakdown computed and logged for all 7 classes
### Key Concepts Learned
* Reading learning curves to diagnose overfitting vs. healthy convergence
* Why high-confidence wrong predictions are more clinically dangerous than low-confidence ones
* Building a fixed, reproducible target list so that later XAI experiments (Weeks 4–9) always audit the exact same failure cases
* Failure case visualization as a bridge between quantitative evaluation and explainability work
---
 
## Day 5: Go/No-Go Review, Baseline Tag & Supervisor Presentation
 
* Ran the **Week 3 go/no-go checklist** — an automated script verifying 7 conditions before allowing Week 4 to start:
  1. Baseline balanced accuracy ≥ 0.65 → **0.7318** ✅
  2. Macro AUC ≥ 0.90 → **0.9478** ✅
  3. All 7 classes have F1 > 0 ✅
  4. Baseline checkpoint exists on disk ✅
  5. Fusion results exist on disk ✅
  6. XAI target cases saved (`xai_target_cases.json`) ✅
  7. Training log exists ✅
  * **Result: 7/7 checks passed — STATUS: GO**
* Tagged the repository at this exact state: **`v0.1-baseline`**
  * Tag message locks in baseline (bal-acc 0.7318, macro F1 0.6869, macro AUC 0.9478) and fusion (bal-acc 0.7168, macro F1 0.6703, macro AUC 0.9549) results, plus the checkpoint path and epoch used
  * All Week 4–9 explainability experiments will run against the checkpoint saved at this tag — if anything is questioned later, this is the reproducible anchor point
* Wrote `notes/week3.md` documenting the reasoning behind every Week 3 design decision while it was still fresh:
  * Why AdamW over Adam (decoupled weight decay regularizes better on a small ~8k-image training set)
  * Why CosineAnnealingLR (smooth decay, no manual step-decay cliffs)
  * Why checkpointing by `val_balanced_accuracy` rather than `val_loss` (loss can decrease while the model collapses to the majority class on imbalanced data)
  * Why the val→test gap exists (79.6% → 73.2%, within expected noise for a 1,009-image validation set)
  * Why metadata fusion underperformed on macro metrics (driven almost entirely by 9 dermatofibroma test samples, not a real model failure)
  * Key findings: mel→nv as the dominant and most dangerous error; high-confidence wrong predictions identified as direct XAI audit targets; mild expected overfitting; bkl improved most from metadata fusion (+11.9 pp F1)
  * A "numbers to cite in the paper" table for quick reference when writing up Table 1
* Delivered the Week 3 closeout summary to the supervisor: baseline achieves 73.2% balanced accuracy and 94.8% macro AUC on the held-out test set; metadata fusion improves AUC and helps 6 of 7 classes; 20 high-confidence failure cases are saved as fixed XAI audit targets for Week 4; the baseline is tagged and reproducible
### Key Concepts Learned
* Go/no-go checklists as a formal gate against moving to specialized work (XAI) on top of an unverified baseline
* Git tagging as permanent, reproducible experiment versioning — the standard research-lab practice over ad hoc filenames
* Writing research notes immediately after experiments, while reasoning is still fresh, rather than reconstructing it later for the paper
* Communicating technical results concisely to a non-technical-in-the-moment supervisor audience
---
 
# Baseline Results — Table 1
 
| Experiment | Bal acc | Macro F1 | Macro AUC | Best val epoch | Notes |
|---|---|---|---|---|---|
| Image-only (ResNet-18) | **0.7318** | **0.6869** | 0.9478 | 15 (0.7964) | Week 3 Day 2 |
| Image + metadata (late fusion) | 0.7168 | 0.6703 | **0.9549** | 13 (0.7825) | Week 3 Day 3 |
 
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
│   │   ├── baseline_image_only.json   # locked test results
│   │   ├── baseline_image_only_report.txt
│   │   ├── metadata_fusion.json       # locked test results
│   │   ├── metadata_fusion_report.txt
│   │   ├── xai_target_cases.json      # 20 high-confidence failure cases → Week 4 Grad-CAM
│   │   ├── plot_learning_curves.py
│   │   ├── failure_analysis.py
│   │   ├── ablation_comparison.py
│   │   └── generate_figure.py
│   ├── src/
│   │   ├── model.py                   # DermaNet (ResNet-18 + optional fusion)
│   │   ├── train.py                   # full training loop + CSV logging
│   │   ├── evaluate.py                # library: run_inference, compute_metrics
│   │   └── evaluate_test.py           # standalone CLI evaluator
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
| Explainability | Captum *(Week 4+)* |
| Config & Tracking | PyYAML, CSV logging |
| Environment | Python, pip, virtualenv |
| Development | VS Code, Jupyter Notebook, Git, GitHub |
 
---
 
# Current Status
 
**Week 3 — COMPLETE ✅**
 
- [x] Week 1: Environment, CIFAR-10 CNN, HAM10000 exploration, custom Dataset
- [x] Week 2: EDA, leakage-free split, augmented DataLoader, MetadataEncoder, DermaNet, train/evaluate pipeline, dry run verified
- [x] Week 3 Day 1: Full 20-epoch baseline training — best val bal-acc 0.7964 at epoch 15
- [x] Week 3 Day 2: Test-set evaluation — bal-acc 0.7318, macro F1 0.6869, AUC 0.9478. Three figures generated
- [x] Week 3 Day 3: Metadata fusion ablation — AUC improves to 0.9549, bkl gains +11.9 pp F1. Table 1 complete
- [x] Week 3 Day 4: Diagnostics — learning curves, failure gallery, high-confidence errors, XAI targets saved
- [x] Week 3 Day 5: Go/no-go passed (7/7), tagged `v0.1-baseline`, `notes/week3.md` written, supervisor summary delivered
---
 
# Roadmap
 
```
Week 1–2   Data pipeline, DermaNet, training infrastructure       ✅ complete
Week 3     Baseline training and rigorous evaluation               ✅ complete (tagged v0.1-baseline)
Week 4     Grad-CAM implementation and first heatmaps              ← next
Week 5     Attribution benchmark: IG, saliency, occlusion + mask-overlap (IoU, pointing game)
Week 6     Quantitative XAI benchmark against ISIC lesion masks
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
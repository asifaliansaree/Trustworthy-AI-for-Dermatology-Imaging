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
 
### Week 3 — Baseline Training & Rigorous Evaluation ← current
| Day | Focus |
|-----|-------|
| 1 | Full 20-epoch baseline training run (image-only) |
| 2 | Evaluation: balanced accuracy, per-class F1, confusion matrix, ROC-AUC |
| 3 | Metadata-fusion ablation run and comparison table |
| 4 | Training diagnostics, learning curves, failure case analysis |
| 5 | Go/no-go review, baseline tag, supervisor presentation |
 
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
* Implemented `src/evaluate.py`:
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
 
## Day 1: Full Baseline Training Run ← in progress
 
> **Synopsis checkpoint**: *"Have a working model by end of week 3 before you specialise."*
 
* Launched full 20-epoch training run with `configs/baseline.yaml` (image-only mode)
* Training configuration:
  * Architecture: ResNet-18 (ImageNet pretrained)
  * Epochs: 20, Batch size: 32, LR: 1e-4 (cosine decay to 1e-6)
  * Loss: CrossEntropyLoss with class weights
  * Split: lesion-wise 80/10/10, `random_state=42`
* Monitoring epoch-level `val_balanced_accuracy` to confirm healthy training
* Target: `val_balanced_accuracy ≥ 0.65` by epoch 20

## Day 2: test-set evaluation and figures

- Ran evaluate.py on the held-out test set using the best checkpoint (epoch 15)
- Test balanced accuracy: **0.7318** — exceeds the 0.65 go/no-go threshold
- Test macro ROC-AUC: **0.9478** — strong probability ranking across all 7 classes
- Test macro F1: **0.6869** — lower F1 expected due to rare-class imbalance (df, vasc)
- Generated 3 publication-quality figures (300 DPI): confusion matrix, per-class F1, training curves
- Val→test gap of 6.5 points is within expected range for this dataset size
- All results locked in `results/baseline_image_only.json`
- Fixed `train.py` to write `training_log.csv` on every future run
---
 
# Repository Structure
 
```text
.
├── cifar10/
│   ├── data/
│   │   └── cifar-10-python.tar.gz
│   ├── train.py
│   └── cifar10_model.pth
│
├── ham10000/
│   ├── checkpoints/
│   │   └── best_model.pt
│   ├── configs/
│   │   ├── baseline.yaml          # image-only, 20 epochs
│   │   ├── metadata_fusion.yaml   # image + metadata, 20 epochs  [Week 3 Day 3]
│   │   └── dry_run.yaml           # 1-epoch sanity check
│   ├── data/
│   │   ├── HAM10000_images_part_1/
│   │   ├── HAM10000_images_part_2/
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_split.csv     # permanent lesion-wise split
│   │   └── class_weights.npy      # precomputed for imbalanced loss
│   ├── src/
│   │   ├── model.py               # DermaNet (ResNet-18 + optional fusion)
│   │   ├── train.py               # full training loop
│   │   └── evaluate.py            # metrics: bal-acc, F1, CM, ROC-AUC
│   ├── results/                   # [Week 3] experiment results JSON + figures
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
 
## Baseline results

| Experiment              | Bal acc | Macro F1 | Macro AUC | Notes        |
|-------------------------|---------|----------|-----------|--------------|
| Image-only (ResNet-18)  | 0.7318  | 0.6869   | 0.9478    | Week 3 Day 2 |
| Image + Metadata        | —       | —        | —         | Week 3 Day 3 |

> Best val balanced accuracy during training: **0.7964** (epoch 15)
---
 
# Current Status
 
**Week 3 — Day 2 in progress**
 
- [x] Week 1: Environment, CIFAR-10 CNN, HAM10000 exploration, custom Dataset
- [x] Week 2: EDA, leakage-free split, augmented DataLoader, MetadataEncoder, DermaNet, train/evaluate pipeline, dry run verified
- [ ] Week 3 Day 1: Launch full 20-epoch training run
- [ ] Week 3 Day 2: Evaluate on test set, record baseline numbers
- [ ] Week 3 Day 3: Metadata fusion ablation
- [ ] Week 3 Day 4: Training diagnostics and failure case analysis
- [ ] Week 3 Day 5: Go/no-go review, tag `v0.1-baseline`, supervisor presentation
---
 
# Roadmap
 
```
Week 1-2  Data pipeline, DermaNet, training infrastructure     ✅ complete
Week 3    Baseline training and rigorous evaluation             ← now
Week 4    Grad-CAM implementation and first heatmaps
Week 5    Attribution benchmark: +3 methods (IG, saliency, occlusion) + mask-overlap (IoU, pointing game)
Week 6    Quantitative XAI benchmark against ISIC lesion masks
Week 7    Inter-method agreement analysis
Week 8    Shortcut-learning audit (artifact-driven prediction hunting)
Week 9    Faithfulness checks: deletion / insertion curves
Week 10   Results consolidation and ablation runs
Week 11   Mini-paper draft + reproducibility pass
Week 12   Final delivery: paper + repo + presentation
```
 
---
 
# Research Output
 
**Target venues**: MICCAI 2027 workshop (ISIC / UNSURE track) · IEEE BHI · arXiv preprint
 
**Core contribution**: Quantitative benchmark of ≥4 attribution methods on HAM10000 using ISIC lesion masks, combined with a shortcut-learning audit measuring the frequency of artifact-driven predictions in high-confidence correct classifications.
 
---
 
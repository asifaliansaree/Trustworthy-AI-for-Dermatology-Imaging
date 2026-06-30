# Trustworthy AI for Dermatology Imaging

This repository documents my learning journey in PyTorch, deep learning, and trustworthy AI for dermatology image analysis.

## Project Goal

To build a strong foundation in deep learning and apply trustworthy AI techniques to dermatology imaging datasets such as HAM10000 for skin lesion classification and analysis.

---

## Learning Timeline

### Week 1

* **Day 1:** Development environment setup and Git/GitHub
* **Day 2:** First Convolutional Neural Network (CNN) using the CIFAR-10 dataset
* **Day 3:** HAM10000 dataset study and PyTorch review
* **Day 4:** HAM10000 dataset exploration and visualization
* **Day 5:** Custom PyTorch Dataset and DataLoader for HAM10000

### Week 2

* **Day 1:** Exploratory Data Analysis (EDA) of the HAM10000 dataset

---

## Week 1

### Day 1: Environment Setup

#### Environment Setup

* Installed Python and configured a virtual environment
* Installed PyTorch, TorchVision, and supporting libraries
* Set up Visual Studio Code with Python, Pylance, and Jupyter extensions
* Verified the PyTorch installation using a test script

#### Git & GitHub Setup

* Installed and configured Git
* Created the GitHub repository
* Initialized a local Git repository
* Connected the local repository to GitHub
* Created the first commit and pushed the project successfully

---

### Day 2: First Deep Learning Project (CIFAR-10)

* Trained a Convolutional Neural Network (CNN) using the CIFAR-10 dataset
* Implemented data loading with PyTorch DataLoader
* Built and trained a CNN model
* Used a loss function and optimizer for training
* Observed decreasing training loss across epochs
* Saved the trained model (`cifar10_model.pth`)
* Documented key concepts in study notes

#### Key Concepts Learned

* Tensors
* Datasets
* DataLoaders
* Convolutional Neural Networks (CNNs)
* Loss Functions
* Optimizers
* Training Loops
* Model Saving and Loading

---

### Day 3: HAM10000 Dataset Study

* Read the HAM10000 research paper
* Learned why the dataset was created
* Studied the seven skin lesion classes
* Explored the metadata and dataset organization
* Created study notes (`notes/ham10000_intro.md`)

#### PyTorch Review

* Reviewed the PyTorch 60-Minute Blitz tutorial
* Reinforced understanding of tensors, datasets, DataLoaders, neural networks, and training loops

---

### Day 4: HAM10000 Dataset Exploration

* Downloaded the HAM10000 dataset
* Organized images and metadata into the project structure
* Loaded the metadata using Pandas
* Explored dataset statistics
* Verified the dataset contains **10,015 dermoscopic images**
* Analyzed the distribution of the **7 skin lesion classes**
* Identified missing values in the metadata
* Displayed sample skin lesion images with their diagnosis labels using Matplotlib

#### Key Concepts Learned

* Medical image datasets
* Metadata analysis
* Data exploration with Pandas
* Class imbalance in medical datasets
* Image visualization with Matplotlib
* Preparing datasets for deep learning

---

### Day 5: Custom PyTorch Dataset

* Created a custom `HAM10000Dataset` class by inheriting from `torch.utils.data.Dataset`
* Loaded the HAM10000 metadata from the CSV file
* Indexed images stored across two directories
* Created a mapping from diagnosis labels to numeric class labels
* Applied image preprocessing using `torchvision.transforms`
* Resized images to **224 × 224**
* Converted images into PyTorch tensors
* Implemented the required Dataset methods:

  * `__init__()`
  * `__len__()`
  * `__getitem__()`
* Verified dataset loading by checking:

  * Dataset size (**10,015 samples**)
  * Image tensor shape (`[3, 224, 224]`)
  * Numeric labels
* Created a `DataLoader` for batch processing
* Verified batch dimensions (`[16, 3, 224, 224]` for images and `[16]` for labels)

#### Key Concepts Learned

* Creating custom PyTorch Dataset classes
* Reading metadata from CSV files
* Loading images from disk
* Label encoding
* Image preprocessing with `torchvision.transforms`
* Batch loading using `DataLoader`
* Preparing datasets for CNN training

---

## Week 2

### Day 1: Exploratory Data Analysis (EDA)

* Created an interactive Jupyter notebook (`EDA.ipynb`) for exploratory data analysis
* Loaded and explored the HAM10000 metadata
* Examined dataset shape, column names, and data types
* Identified missing values in the metadata
* Found **57 missing values in the `age` column**
* Visualized the distribution of all seven diagnosis classes
* Analyzed patient sex distribution
* Plotted the patient age distribution
* Visualized lesion body localization frequencies
* Displayed representative sample images from each diagnosis class
* Documented observations about dataset characteristics and class imbalance

#### Key Concepts Learned

* Exploratory Data Analysis (EDA)
* Metadata inspection
* Missing value analysis
* Data visualization with Matplotlib
* Class imbalance analysis
* Medical image dataset exploration
* Interpreting patient demographic information

---

## Repository Structure

```text
.
├── cifar10/
│   ├── train.py
│   └── cifar10_model.pth
│
├── ham10000/
│   ├── data/
│   │   ├── HAM10000_images_part_1/
│   │   ├── HAM10000_images_part_2/
│   │   └── HAM10000_metadata.csv
│   ├── dataset.py
│   ├── test_dataset.py
│   ├── EDA.ipynb
│   ├── explore_dataset.py
│   └── show_images.py
│
├── notes/
│   ├── week1.md
│   └── ham10000_intro.md
│
├── test_torch.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* Pandas
* Matplotlib
* Pillow
* Jupyter Notebook
* Git
* GitHub
* Visual Studio Code

---

## Next Steps

* Analyze duplicate lesions using the `lesion_id` column
* Create patient-aware train, validation, and test splits
* Apply image normalization and data augmentation
* Train a baseline CNN on the HAM10000 dataset
* Evaluate model performance using appropriate metrics
* Experiment with pretrained models such as ResNet-18
* Explore explainability and trustworthy AI techniques (e.g., Grad-CAM, saliency maps)

---

## Current Status

* Completed Week 1 learning objectives
* Built a custom PyTorch Dataset and DataLoader for HAM10000
* Completed an exploratory data analysis (EDA) notebook
* Analyzed metadata, demographics, and class imbalance
* Identified missing values in the dataset
* Visualized representative images from each lesion class
* Ready to perform duplicate lesion analysis and prepare train/validation/test splits for model training

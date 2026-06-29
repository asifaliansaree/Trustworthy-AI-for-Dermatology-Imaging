# Trustworthy AI for Dermatology Imaging

This repository documents my learning journey in PyTorch, deep learning, and trustworthy AI for dermatology image analysis.

## Project Goal

To build a strong foundation in deep learning and apply trustworthy AI techniques to dermatology imaging datasets such as HAM10000 for skin lesion classification and analysis.

---

## Learning Timeline

- **Day 1:** Development environment setup and Git/GitHub
- **Day 2:** First Convolutional Neural Network (CNN) using the CIFAR-10 dataset
- **Day 3:** HAM10000 dataset study and PyTorch review
- **Day 4:** HAM10000 dataset exploration and visualization

---

## Day 1 Progress

### Environment Setup

- Installed Python and configured a virtual environment
- Installed PyTorch, TorchVision, and supporting libraries
- Set up Visual Studio Code with Python, Pylance, and Jupyter extensions
- Verified the PyTorch installation using a test script

### Git & GitHub Setup

- Installed and configured Git
- Created the GitHub repository
- Initialized a local Git repository
- Connected the local repository to GitHub
- Created the first commit and pushed the project successfully

---

## Day 2 Progress

### First Deep Learning Project: CIFAR-10 Classification

- Trained a Convolutional Neural Network (CNN) using the CIFAR-10 dataset
- Implemented data loading with PyTorch DataLoader
- Built and trained a CNN model
- Used a loss function and optimizer for training
- Observed decreasing training loss across epochs
- Saved the trained model (`cifar10_model.pth`)
- Documented key concepts in study notes

### Key Concepts Learned

- Tensors
- Datasets
- DataLoaders
- Convolutional Neural Networks (CNNs)
- Loss Functions
- Optimizers
- Training Loops
- Model Saving and Loading

---

## Day 3 Progress

### HAM10000 Dataset Study

- Read the HAM10000 research paper
- Learned why the dataset was created
- Studied the seven skin lesion classes
- Explored the metadata and dataset organization
- Created study notes (`notes/ham10000_intro.md`)

### PyTorch Review

- Reviewed the PyTorch 60-Minute Blitz tutorial
- Reinforced understanding of tensors, datasets, DataLoaders, neural networks, and training loops

---

## Day 4 Progress

### HAM10000 Dataset Exploration

- Downloaded the HAM10000 dataset
- Organized images and metadata into the project structure
- Loaded the metadata using Pandas
- Explored the dataset statistics
- Verified the dataset contains **10,015 dermoscopic images**
- Analyzed the distribution of the **7 skin lesion classes**
- Identified missing values in the metadata
- Displayed sample skin lesion images with their diagnosis labels using Matplotlib

### Key Concepts Learned

- Medical image datasets
- Metadata analysis
- Data exploration with Pandas
- Class imbalance in medical datasets
- Image visualization with Matplotlib
- Preparing datasets for deep learning

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

- Python
- PyTorch
- TorchVision
- Pandas
- Matplotlib
- Pillow
- Git
- GitHub
- Visual Studio Code

---

## Next Steps

- Build a custom PyTorch Dataset class for HAM10000
- Apply image preprocessing and transformations
- Create DataLoaders for training and validation
- Train a baseline CNN on the HAM10000 dataset
- Evaluate model performance
- Explore explainability and trustworthy AI techniques

---

## Current Status

- Completed development environment setup
- Verified PyTorch installation
- Established a GitHub workflow
- Trained a CNN on the CIFAR-10 dataset
- Studied the HAM10000 research paper
- Downloaded and organized the HAM10000 dataset
- Explored metadata and class distribution
- Visualized sample skin lesion images
- Ready to build a custom PyTorch Dataset and begin model training on HAM10000
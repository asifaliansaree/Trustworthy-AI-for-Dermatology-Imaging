# Trustworthy AI for Dermatology Imaging

This repository documents my learning journey in PyTorch, deep learning, and trustworthy AI for dermatology image analysis.

## Project Goal

To build a strong foundation in deep learning and apply trustworthy AI techniques to dermatology imaging datasets such as HAM10000 for skin lesion classification and analysis.

---

## Learning Timeline

- **Day 1:** Development environment setup and Git/GitHub
- **Day 2:** First Convolutional Neural Network (CNN) using the CIFAR-10 dataset
- **Day 3:** HAM10000 dataset study and PyTorch review

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
- Implemented data loading with PyTorch `DataLoader`
- Built and trained a CNN model
- Used a loss function and optimizer for training
- Reduced training loss across multiple epochs
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

- Read the title, abstract, introduction, and dataset sections of the HAM10000 research paper
- Learned why the HAM10000 dataset was created and why it is widely used in AI research
- Studied the seven skin lesion classes included in the dataset
- Explored the dataset structure, including image folders and metadata
- Documented the dataset organization and metadata fields
- Created detailed notes in `notes/ham10000_intro.md`

### PyTorch Review

- Reviewed the PyTorch 60-Minute Blitz tutorial
- Reinforced understanding of:
  - Tensors
  - Datasets
  - DataLoader
  - Neural Networks
  - Training Loops
- Connected these concepts to future work with the HAM10000 dataset

### Key Concepts Learned

- HAM10000 dataset organization
- Dermoscopic image datasets
- Metadata and diagnosis labels
- Skin lesion classification
- Medical image datasets
- PyTorch data pipeline

---

## Repository Structure

```text
.
├── cifar10/
│   └── train.py
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
- Git
- GitHub
- Visual Studio Code

---

## Next Steps

- Download and explore the HAM10000 dataset
- Load images and metadata using PyTorch
- Build a custom PyTorch `Dataset` class
- Visualize sample skin lesion images
- Preprocess the dataset for training
- Train a baseline CNN on the HAM10000 dataset
- Evaluate model performance
- Explore explainability and trustworthy AI techniques

---

## Current Status

- Completed development environment setup
- Verified PyTorch installation
- Established a GitHub workflow
- Trained a CNN on the CIFAR-10 dataset
- Studied the HAM10000 dataset and research paper
- Documented the seven skin lesion classes
- Understood the dataset structure and metadata
- Reviewed core PyTorch concepts
- Prepared to begin working with the HAM10000 dataset in PyTorch
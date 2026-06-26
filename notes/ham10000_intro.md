# HAM10000 Dataset Notes

**Date:** 2026-06-26

## What is HAM10000?

HAM10000 (Human Against Machine with 10,000 training images) is a public dataset of dermatoscopic images of pigmented skin lesions. It was created to support the development and evaluation of machine learning models for automated skin lesion classification.

## Why was the dataset created?

The dataset was created because there was a shortage of large, diverse, publicly available dermoscopy datasets. It combines images collected from different medical institutions and includes expert-labeled lesions to facilitate research in computer vision and dermatology.

## How many images does it contain?

- **10,015** dermatoscopic images.

## How many disease classes are there?

- **7** diagnostic classes.

## Why is it important for AI research?

HAM10000 has become one of the standard benchmark datasets for skin lesion classification. It provides a large number of clinically validated images, allowing researchers to train, compare, and improve deep learning models for melanoma detection and other skin diseases.

---

# HAM10000 Lesion Classes

## 1. akiec

**Full name:** Actinic Keratoses and Intraepithelial Carcinoma (Bowen's disease)

**Benign or malignant?** Precancerous / carcinoma in situ (may progress to invasive squamous cell carcinoma if untreated).

**Description:** These lesions develop mainly due to long-term sun exposure. Early diagnosis and treatment are important because some lesions can develop into invasive skin cancer.

---

## 2. bcc

**Full name:** Basal Cell Carcinoma

**Benign or malignant?** Malignant

**Description:** The most common type of skin cancer. It usually grows slowly and rarely spreads to distant organs, but it should still be treated.

---

## 3. bkl

**Full name:** Benign Keratosis-like Lesions

**Benign or malignant?** Benign

**Description:** A group of harmless skin growths that includes seborrheic keratoses, solar lentigines, and lichenoid keratoses. They can resemble melanoma, making diagnosis challenging.

---

## 4. df

**Full name:** Dermatofibroma

**Benign or malignant?** Benign

**Description:** A common, non-cancerous skin nodule that often develops after minor skin injury. It usually does not require treatment.

---

## 5. mel

**Full name:** Melanoma

**Benign or malignant?** Malignant

**Description:** An aggressive skin cancer originating from melanocytes. Early detection greatly improves treatment outcomes.

---

## 6. nv

**Full name:** Melanocytic Nevi

**Benign or malignant?** Benign

**Description:** Common moles formed by melanocytes. Most are harmless, although some melanomas can resemble benign nevi.

---

## 7. vasc

**Full name:** Vascular Lesions

**Benign or malignant?** Benign

**Description:** Lesions caused by blood vessel abnormalities, such as angiomas or hemorrhages. They are generally non-cancerous.

## Dataset Structure

The HAM10000 dataset is organized into two folders containing the dermoscopic images and one metadata file.

```text
HAM10000/
│
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   ├── ISIC_0024308.jpg
│   └── ...
│
├── HAM10000_images_part_2/
│   ├── ISIC_0030000.jpg
│   ├── ISIC_0030001.jpg
│   ├── ISIC_0030002.jpg
│   └── ...
│
└── HAM10000_metadata.csv
```

### Images

- Total images: **10,015**
- File format: **JPEG (.jpg)**
- Image resolution: **600 × 450 pixels**
- Each image is a dermoscopic (skin lesion) photograph.
- Image filenames follow the format `ISIC_xxxxxxx.jpg`.

### Metadata File

The `HAM10000_metadata.csv` file contains one row for each image and includes the following columns:

| Column | Description |
|---------|-------------|
| `lesion_id` | Unique identifier for the skin lesion. Multiple images can share the same lesion ID if they show the same lesion. |
| `image_id` | Unique image identifier that matches the image filename (without `.jpg`). |
| `dx` | Diagnosis label (one of the seven lesion classes). |
| `dx_type` | Method used to confirm the diagnosis (e.g., histopathology, follow-up, expert consensus, or confocal microscopy). |
| `age` | Patient age (may be missing for some records). |
| `sex` | Patient sex (`male`, `female`, or missing). |
| `localization` | Anatomical location of the lesion (e.g., back, lower extremity, trunk, face, scalp). |

### Diagnosis Labels (`dx`)

The `dx` column contains one of the following seven classes:

| Label | Full Name |
|-------|-----------|
| `akiec` | Actinic Keratoses and Intraepithelial Carcinoma (Bowen's disease) |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

### Dataset Organization

```text
HAM10000 Dataset
│
├── Images (10,015 JPEG files)
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
└── Metadata
    └── HAM10000_metadata.csv
            │
            ├── image_id
            ├── lesion_id
            ├── dx
            ├── dx_type
            ├── age
            ├── sex
            └── localization
```

### Relationship Between Images and Metadata

Each image in the image folders has a corresponding row in `HAM10000_metadata.csv`. The `image_id` field links the metadata to the image file, while the `dx` field provides the diagnosis label used for training and evaluating machine learning models.
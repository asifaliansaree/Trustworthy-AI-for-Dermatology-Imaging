import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Label mapping
CLASS_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

# ImageNet normalization (recommended for pretrained models)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform(split):
    """Return transforms for each dataset split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        data_dir="ham10000/data",
        split="train",
        transform=None,
    ):
        """
        Args:
            data_dir: path to data folder
            split: 'train', 'val', or 'test'
            transform: optional custom transform
        """

        assert split in ["train", "val", "test"]

        self.data_dir = data_dir
        self.split = split

        # Load split CSV
        csv_path = os.path.join(data_dir, "HAM10000_split.csv")
        df = pd.read_csv(csv_path)

        # Keep only requested split
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.label_map = CLASS_MAP

        # Image folders
        self.image_dirs = [
            os.path.join(data_dir, "HAM10000_images_part_1"),
            os.path.join(data_dir, "HAM10000_images_part_2"),
        ]

        # Build image path dictionary
        self.image_paths = {}

        for folder in self.image_dirs:
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    image_id = file[:-4]
                    self.image_paths[image_id] = os.path.join(folder, file)

        # Use custom transform if provided
        if transform is None:
            self.transform = get_transform(split)
        else:
            self.transform = transform

        print(
            f"[{split}] Loaded {len(self.df)} images "
            f"across {self.df['dx'].nunique()} classes."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = self.label_map[row["dx"]]

        image_path = self.image_paths[image_id]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label
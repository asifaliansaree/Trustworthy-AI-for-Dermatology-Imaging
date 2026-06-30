import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000Dataset(Dataset):
    def __init__(self, data_dir="ham10000/data", transform=None):
        self.data_dir = data_dir

        # Load metadata
        self.df = pd.read_csv(
            os.path.join(data_dir, "HAM10000_metadata.csv")
        )

        # Label mapping
        self.label_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }

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

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

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
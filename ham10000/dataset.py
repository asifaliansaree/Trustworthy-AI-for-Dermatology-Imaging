"""
HAM10000Dataset — configurable augmentation, split-aware loading,
optional metadata encoder support.
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Optional

CLASS_MAP = {
    "mel": 0, "nv": 1, "bcc": 2,
    "akiec": 3, "bkl": 4, "df": 5, "vasc": 6,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transform(split: str, img_size: int = 224,
                  augment: bool = True) -> T.Compose:
    """
    Returns the appropriate transform pipeline.

    Train: strong augmentation (only if augment=True)
    Val / Test: deterministic resize + center crop + normalize
    """
    if split == "train" and augment:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=90),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset loader.

    Args:
        data_dir:         path to ham10000/data/ (contains split CSV + images)
        split:            'train', 'val', or 'test'. Only used to pick the
                           transform (train gets augmentation, val/test
                           don't) and for the printed summary line -- when
                           kfold_csv/folds are given, it no longer selects
                           rows (folds does that instead).
        metadata_encoder: optional MetadataEncoder instance
        img_size:         image size (default 224)
        augment:          enable training augmentation (default True)
        kfold_csv:        optional filename (relative to data_dir) of a
                           fold-assignment CSV produced by
                           make_kfold_split.py (must have a 'fold' column).
                           When given together with `folds`, rows are
                           selected by fold membership instead of by the
                           legacy 'split' column in HAM10000_split.csv.
                           This is for the shared 5-fold lesion-wise
                           StratifiedGroupKFold CV split used to compare
                           the three interns' models on identical folds.
        folds:             list of int fold numbers (0-4) to include when
                           kfold_csv is set, e.g. [0,1,2,3] for a training
                           set that excludes fold 4 as validation.
    """

    def __init__(
        self,
        data_dir:         str,
        split:            str,
        metadata_encoder  = None,
        img_size:         int  = 224,
        augment:          bool = True,
        kfold_csv:        Optional[str]  = None,
        folds:            Optional[list] = None,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be train/val/test, got '{split}'"

        if kfold_csv is not None and folds is not None:
            csv_path = os.path.join(data_dir, kfold_csv)
            df       = pd.read_csv(csv_path)
            assert "fold" in df.columns, (
                f"{csv_path} has no 'fold' column -- make sure it was "
                f"produced by make_kfold_split.py, not HAM10000_split.csv."
            )
            self.df = df[df["fold"].isin(folds)].reset_index(drop=True)
        else:
            csv_path = os.path.join(data_dir, "HAM10000_split.csv")
            df       = pd.read_csv(csv_path)
            self.df  = df[df["split"] == split].reset_index(drop=True)
        self.data_dir         = data_dir
        self.split            = split
        self.metadata_encoder = metadata_encoder
        self.transform        = get_transform(split, img_size, augment)

        self._img_dirs = [
            os.path.join(data_dir, "HAM10000_images_part_1"),
            os.path.join(data_dir, "HAM10000_images_part_2"),
        ]

        mode = "train" if split == "train" and augment else split
        meta = " + metadata" if metadata_encoder else ""
        print(f"[{split}] {len(self.df)} images{meta} | mode={mode}")

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, image_id: str) -> str:
        for d in self._img_dirs:
            p = os.path.join(d, image_id + ".jpg")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Image '{image_id}' not found in either image directory."
        )

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(self._find_image(row["image_id"])).convert("RGB")
        image = self.transform(image)
        label = CLASS_MAP[row["dx"]]

        if self.metadata_encoder is not None:
            meta = self.metadata_encoder.encode(row)
            return image, meta, label

        return image, label
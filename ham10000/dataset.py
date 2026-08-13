"""
HAM10000Dataset — configurable augmentation, split-aware loading,
optional metadata encoder support.
"""
import os
import glob
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Optional

CLASS_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
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


def _build_image_index(search_roots: list) -> dict:
    """
    Recursively walk each directory in `search_roots` and build a
    {image_id: full_path} index of every .jpg found.

    This makes image lookup independent of the on-disk layout — it works
    whether images sit in flat HAM10000_images_part_1/2 folders (the
    original Kaggle release layout) or in the fold_N/class/ tree produced
    by make_kfold_folders.py (symlinks or real files, doesn't matter).
    Later roots don't overwrite earlier matches, so pass roots in priority
    order if a given image_id could exist in more than one.
    """
    index = {}
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
            image_id = os.path.splitext(os.path.basename(path))[0]
            index.setdefault(image_id, path)
    return index


def _discover_kaggle_input_roots() -> list:
    """
    On Kaggle, attached datasets are mounted read-only under /kaggle/input
    and are NOT automatically visible inside a cloned repo's data_dir.
    If a directory literally named 'kfold' (the output of
    make_kfold_folders.py) exists anywhere under /kaggle/input, treat it
    as an extra image search root. This is a no-op off Kaggle.
    """
    roots = []
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for dirpath, dirnames, _ in os.walk(kaggle_input):
            if "kfold" in dirnames:
                roots.append(os.path.join(dirpath, "kfold"))
    return roots


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
        extra_img_dirs:    optional list of additional directories to search
                           for images (searched recursively), on top of the
                           default HAM10000_images_part_1/2 and any
                           auto-detected Kaggle 'kfold' input folders.
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
        extra_img_dirs:   Optional[list] = None,
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

        # Search order: legacy flat folders first, then any caller-supplied
        # extra dirs, then auto-detected Kaggle input 'kfold' folders, then
        # data_dir itself (covers a kfold/ tree placed directly inside it).
        search_roots = [
            os.path.join(data_dir, "HAM10000_images_part_1"),
            os.path.join(data_dir, "HAM10000_images_part_2"),
            *(extra_img_dirs or []),
            *_discover_kaggle_input_roots(),
            data_dir,
        ]
        self._image_index = _build_image_index(search_roots)

        mode = "train" if split == "train" and augment else split
        meta = " + metadata" if metadata_encoder else ""
        print(f"[{split}] {len(self.df)} images{meta} | mode={mode} "
              f"| indexed {len(self._image_index)} images on disk")

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, image_id: str) -> str:
        path = self._image_index.get(image_id)
        if path is None:
            raise FileNotFoundError(
                f"Image '{image_id}' not found in any indexed image directory "
                f"(searched HAM10000_images_part_1/2, any Kaggle 'kfold' input "
                f"folders, and {self.data_dir})."
            )
        return path

    def __getitem__(self, idx: int, _depth: int = 0):
        row = self.df.iloc[idx]

        try:
            image = Image.open(self._find_image(row["image_id"])).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            # A single corrupt/truncated/missing image should not kill an
            # entire multi-hour training run. Log it and fall back to a
            # different sample instead. _depth guards against the (very
            # unlikely) case where many consecutive images are bad, so we
            # fail loudly rather than recursing forever.
            if _depth >= 20:
                raise RuntimeError(
                    f"20 consecutive unreadable images starting at idx {idx} "
                    f"-- this looks like a systemic dataset problem, not a "
                    f"one-off bad file. Last error: {e}"
                ) from e
            print(f"[WARN] Skipping unreadable image '{row['image_id']}': {e}")
            return self.__getitem__((idx + 1) % len(self), _depth=_depth + 1)

        image = self.transform(image)
        label = CLASS_MAP[row["dx"]]

        if self.metadata_encoder is not None:
            meta = self.metadata_encoder.encode(row)
            return image, meta, label

        return image, label
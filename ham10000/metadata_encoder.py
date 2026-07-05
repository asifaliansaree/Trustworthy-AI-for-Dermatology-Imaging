import torch
import pandas as pd

LOCALIZATIONS = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
    'foot', 'genital', 'hand', 'lower extremity',
    'neck', 'scalp', 'trunk', 'unknown', 'upper extremity'
]


class MetadataEncoder:
    """Encodes age/sex/localization into a fixed 18-d tensor.
    Fit ONLY on the train split, to avoid leaking val/test stats."""

    def __init__(self, split_csv_path: str):
        df = pd.read_csv(split_csv_path)
        train = df[df['split'] == 'train']
        self.age_mean = float(train['age'].mean())
        self.age_std = float(train['age'].std() + 1e-6)
        self.age_median = float(train['age'].median())

    @property
    def dim(self) -> int:
        return 1 + 1 + len(LOCALIZATIONS) + 1  # age + sex + loc-onehot + missing = 18

    def encode(self, row: pd.Series) -> torch.Tensor:
        feats = []

        age = row['age']
        missing = float(pd.isna(age))
        if missing:
            age = self.age_median  # overall train median -- see note below
        feats.append((age - self.age_mean) / self.age_std)

        sex_map = {'male': 1.0, 'female': 0.0}
        feats.append(sex_map.get(str(row.get('sex', '')).lower(), 0.5))

        loc = str(row.get('localization', '')).lower().strip()
        for l in LOCALIZATIONS:
            feats.append(1.0 if loc == l else 0.0)

        feats.append(missing)
        return torch.tensor(feats, dtype=torch.float32)
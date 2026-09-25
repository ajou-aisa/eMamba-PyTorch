import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


SPLIT_FILES = {"train": "train", "validation": "validate", "test": "test"}
SOURCE_SIZES: Final = {"train": 24066, "validation": 8033, "test": 7984}
SPLIT_SIZES: Final = {"train": 25679, "validation": 6420, "test": 7984}


def training_data_fingerprint(data_root: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "featuremap_train.npy", "labels_train.npy",
        "featuremap_validate.npy", "labels_validate.npy",
    ):
        with (data_root / name).open("rb") as source:
            digest.update(name.encode("utf-8"))
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return digest.hexdigest()


class MARSDataset(Dataset):
    def __init__(self, feature_path, label_path):
        feature_path = Path(feature_path)
        label_path = Path(label_path)
        split = feature_path.stem.removeprefix("featuremap_")

        try:
            self.features = np.load(feature_path).astype(np.float32)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"{split} feature file {feature_path}: {exc}") from exc
        try:
            self.labels = np.load(label_path).astype(np.float32)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"{split} label file {label_path}: {exc}") from exc

        if self.features.ndim != 4 or self.features.shape[1:] != (8, 8, 5):
            raise ValueError(
                f"{split} feature file {feature_path}: expected [S,8,8,5], "
                f"got {self.features.shape}"
            )
        if self.labels.ndim != 2 or self.labels.shape[1] != 57:
            raise ValueError(f"{split} label file {label_path}: expected [S,57], got {self.labels.shape}")
        if not len(self.features) or len(self.features) != len(self.labels):
            raise ValueError(
                f"{split} sample count mismatch or empty: "
                f"{feature_path}={len(self.features)}, {label_path}={len(self.labels)}"
            )
        if not np.isfinite(self.features).all():
            raise ValueError(f"{split} feature file {feature_path}: contains NaN or Inf")
        if not np.isfinite(self.labels).all():
            raise ValueError(f"{split} label file {label_path}: contains NaN or Inf")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.labels[idx])

        return x, y


def build_dataloaders(
    data_root: Path, splits: tuple[str, ...], batch_size: int, num_workers: int,
) -> dict[str, DataLoader]:
    sources: dict[str, MARSDataset] = {}
    for split, suffix in SPLIT_FILES.items():
        features = data_root / f"featuremap_{suffix}.npy"
        labels = data_root / f"labels_{suffix}.npy"
        dataset = MARSDataset(features, labels)
        if len(dataset) != SOURCE_SIZES[split]:
            raise ValueError(
                f"{split} at {data_root}: expected {SOURCE_SIZES[split]} samples, "
                f"got {len(dataset)}"
            )
        sources[split] = dataset
    partitions = random_split(
        ConcatDataset([sources["train"], sources["validation"]]),
        [SPLIT_SIZES["train"], SPLIT_SIZES["validation"]],
        generator=torch.Generator().manual_seed(0),
    )
    datasets = {"train": partitions[0], "validation": partitions[1], "test": sources["test"]}
    loaders = {}
    for split in splits:
        loaders[split] = DataLoader(
            datasets[split], batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, drop_last=False,
        )
    return loaders

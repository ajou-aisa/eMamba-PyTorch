from pathlib import Path
from typing import Final

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from datasets.mars import MARSDataset


SPLIT_FILES = {"train": "train", "validation": "validate", "test": "test"}
SOURCE_SIZES: Final = {"train": 24066, "validation": 8033, "test": 7984}
SPLIT_SIZES: Final = {"train": 25679, "validation": 6420, "test": 7984}


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

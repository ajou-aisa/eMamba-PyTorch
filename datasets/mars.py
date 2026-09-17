import numpy as np
import torch
from torch.utils.data import Dataset


class MARSDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = np.load(feature_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)

        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.labels[idx])

        return x, y

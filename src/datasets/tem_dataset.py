import h5py
import torch
from torch.utils.data import Dataset

class TEMTrainDataset(Dataset):
    def __init__(self, h5_path, transform=None, target_transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.target_transform = target_transform

        with h5py.File(h5_path, "r") as f:
            self.keys = sorted(f["raw"].keys(), key=str)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]

        with h5py.File(self.h5_path, "r") as f:
            img = f["raw"][k][()]
            mask = f["label"][k][()]

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask

class TEMTestDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        with h5py.File(h5_path, "r") as f:
            self.keys = sorted(f["raw"].keys(), key=str)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            img = f["raw"][k][()]

        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)

        return img, k

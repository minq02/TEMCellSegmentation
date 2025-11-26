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
    
class TEMPatchDataset(Dataset):
    def __init__(
        self,
        h5_path,
        patch_size=512,
        stride=None,
        keys=None,
        transform=None,
        target_transform=None,
    ):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.transform = transform
        self.target_transform = target_transform
        self.index = []  # list of (key, top, left)

        with h5py.File(h5_path, "r") as f:
            raw_group = f["raw"]
            all_keys = sorted(raw_group.keys(), key=str)

            if keys is None:
                keys = all_keys      # use all images
            else:
                # keep only keys that actually exist in the file
                keys = [k for k in keys if k in raw_group.keys()]

            for k in keys:
                h, w = raw_group[k].shape
                for top in range(0, h - patch_size + 1, self.stride):
                    for left in range(0, w - patch_size + 1, self.stride):
                        self.index.append((k, top, left))

        print(f"[TEMPatchDataset] {len(self.index)} patches from {len(keys)} images")


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        k, top, left = self.index[idx]
        p = self.patch_size

        with h5py.File(self.h5_path, "r") as f:
            img_full = f["raw"][k]
            mask_full = f["label"][k]

            img = img_full[top:top+p, left:left+p]
            mask = mask_full[top:top+p, left:left+p]

        img = torch.from_numpy(img).float()   # (H, W)
        mask = torch.from_numpy(mask).long()  # (H, W) with labels 0..4

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

import random
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.tem_dataset import TEMPatchDataset
from src.models.unet_small import UNetSmall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h5_path = "data/train/train_data.h5"
patch_size = 512
batch_size = 4
num_epochs = 10

# ----- split image keys into train / val -----
with h5py.File(h5_path, "r") as f:
    all_keys = sorted(f["raw"].keys(), key=str)

random.seed(42)
random.shuffle(all_keys)

val_keys = all_keys[-2:]
train_keys = all_keys[:-2]

print("Train keys:", train_keys)
print("Val keys:", val_keys)

# ----- datasets & loaders -----
train_ds = TEMPatchDataset(
    h5_path,
    patch_size=patch_size,
    stride=patch_size,
    keys=train_keys,
)

val_ds = TEMPatchDataset(
    h5_path,
    patch_size=patch_size,
    stride=patch_size,
    keys=val_keys,
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

# ----- model, loss, optimizer -----
model = UNetSmall(in_ch=1, n_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----- training loop with batch progress + validation -----
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_samples = 0

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
        if imgs.dim() == 3:           # (B, H, W) → (B, 1, H, W)
            imgs = imgs.unsqueeze(1)

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        # running stats
        bsz = imgs.size(0)
        train_loss += loss.item() * bsz
        train_samples += bsz

        # print progress every N batches
        if batch_idx % 20 == 0 or batch_idx == len(train_loader):
            avg_loss_so_far = train_loss / train_samples
            print(
                f"  [Train] Batch {batch_idx:4d}/{len(train_loader)}  "
                f"avg loss: {avg_loss_so_far:.4f}"
            )

    train_loss /= len(train_ds)

    # ----- validation -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(1)

            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item() * imgs.size(0)

    val_loss /= len(val_ds)

    print(
        f"Epoch {epoch+1}/{num_epochs} summary: "
        f"train loss = {train_loss:.4f}, val loss = {val_loss:.4f}"
    )

# save model
torch.save(model.state_dict(), "tem_unet_small.pth")

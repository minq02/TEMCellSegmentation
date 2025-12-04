import random
import h5py
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

from src.datasets.tem_dataset import TEMPatchDataset
from src.models.unet_small import UNetSmall
from src.models.attention_unet import AttentionUNet, AdaptedAttentionUNet
from src.utils.losses import dice_loss

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            translate_percent=(0.0, 0.0),
            fit_output=False,
            p=0.5,
        ),

        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.8, 0.95),   # mild low-res
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.2,  # only 20% of samples get touched
        ),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ]
)


val_aug = A.Compose(
    [   
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# h5_path = "data/train/train_data.h5"
h5_path = "data/train/train_data_downsampled.h5"
patch_size = 512  # 512 x 512 patches since original image is too large
stride = 256      # change if need to overlap
batch_size = 8    # adjust based on GPU
num_epochs = 150   # can go higher now that we have early stopping

# gradient accumulation

# ---- early stopping & checkpoint hyperparams ----
patience = 10         # stop if no val improvement for this many epochs
min_delta = 1e-4      # minimum improvement to count as "better"

# ---- split image keys into train / val sets ----
with h5py.File(h5_path, "r") as f:
    all_keys = sorted(f["raw"].keys(), key=str)

# random.seed(41)  # for reproducibility
random.seed(46)  # for reproducibility
random.shuffle(all_keys)

val_keys = all_keys[-2:]
train_keys = all_keys[:-2]

print("Train keys:", train_keys)
print("Val keys:", val_keys)

# ---- dataset and loaders ----
train_ds = TEMPatchDataset(
    h5_path,
    patch_size=patch_size,
    stride=stride,
    keys=train_keys,
    aug=train_aug
)

val_ds = TEMPatchDataset(
    h5_path,
    patch_size=patch_size,
    stride=stride,
    keys=val_keys,
    aug=val_aug
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

# ---- model, loss, optimizer, scheduler ----
# model = UNetSmall(in_ch=1, n_classes=5).to(device)  # 5 classes including background
model = AttentionUNet(in_ch=1, n_classes=5).to(device)  # 5 classes including background
# model = AdaptedAttentionUNet(in_ch=1, n_classes=5).to(device)

criterion_ce = nn.CrossEntropyLoss()
dice_weight = 0.5  # how important is dice loss compared to CE loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# scheduler: reduce LR when val loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,     # multiply LR by this factor
    patience=2,     # epochs with no improvement before reducing LR
)

best_val_loss = float("inf")
epochs_no_improve = 0

# ---- lists to store loss history ----
train_losses = []
val_losses = []

# ---- training loop with batch progress + validation, checkpoint, early stopping ----
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_samples = 0

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch+1}/{num_epochs}  (lr = {current_lr:.2e})")

    for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
        if imgs.dim() == 3:           # (B, H, W) → (B, 1, H, W)
            imgs = imgs.unsqueeze(1)

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        ce = criterion_ce(logits, masks)
        dl = dice_loss(logits, masks, num_classes=5)
        loss = ce + dice_weight * dl

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
                f"avg loss: {avg_loss_so_far:.4f}  "
                f"(CE: {ce.item():.4f}, Dice: {dl.item():.4f})"
            )

    train_loss /= len(train_ds)

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(1)

            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            ce = criterion_ce(logits, masks)
            dl = dice_loss(logits, masks, num_classes=5)
            loss = ce + dice_weight * dl

            val_loss += loss.item() * imgs.size(0)

    val_loss /= len(val_ds)

    # store for plotting later
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs} summary: "
        f"train loss = {train_loss:.4f}, val loss = {val_loss:.4f}"
    )

    # ---- LR scheduler step (uses validation loss) ----
    scheduler.step(val_loss)

    # ---- checkpoint best model & early stopping ----
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # torch.save(model.state_dict(), "tem_unet_small_best.pth")
        torch.save(model.state_dict(), "tem_attention_unet_best.pth")
        # torch.save(model.state_dict(), "tem_adapted_attention_unet_best.pth")
        print(f"  -> New best model saved with val loss {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# save losses to a json for later plotting
with open("loss_history.json", "w") as f:
    json.dump(
        {"train_losses": train_losses, "val_losses": val_losses},
        f,
        indent=2,
    )

# also save the final state if you want
# torch.save(model.state_dict(), "tem_unet_small_last.pth")
torch.save(model.state_dict(), "tem_attention_unet_last.pth")
# torch.save(model.state_dict(), "tem_adapted_attention_unet_last.pth")
print("Training finished.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.tem_dataset import TEMTrainDataset
from src.models.unet_small import UNetSmall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetSmall(in_ch=1, n_classes=2).to(device)  # adjust channels/classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5

train_path = "/home/minq02/github/TEMCellSegmentation/data/train/train_data.h5"

train_ds = TEMTrainDataset(train_path)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        # if img is (B, H, W), add channel dim → (B, 1, H, W)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)          # (B, C, H, W)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f}")

# save model
torch.save(model.state_dict(), "tem_unet_small.pth")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.tem_dataset import TEMTestDataset
from src.models.unet_small import UNetSmall
from src.models.attention_unet import AttentionUNet


def predict_full_image(
    model,
    img_tensor,
    patch_size=512,
    stride=256,
    num_classes=5,
    device="cuda",
):
    """
    Patch-based inference with overlapping patches and logits averaging.

    img_tensor: (H, W) float tensor
    returns: (H, W) int32 numpy array of predicted labels (0..num_classes-1)
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    img_np = img_tensor.cpu().numpy()
    H, W = img_np.shape

    logits_accum = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))

    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    with torch.no_grad():
        for top in ys:
            for left in xs:
                patch_np = img_np[top:top+patch_size, left:left+patch_size]
                patch = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0).to(device)

                logits = model(patch)             # (1, C, ps, ps)
                logits = logits.squeeze(0).cpu().numpy()  # (C, ps, ps)

                logits_accum[:, top:top+patch_size, left:left+patch_size] += logits
                counts[top:top+patch_size, left:left+patch_size] += 1.0

    counts[counts == 0] = 1.0
    avg_logits = logits_accum / counts[None, :, :]

    pred = avg_logits.argmax(axis=0).astype(np.int32)
    return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_h5_path = "data/test/test_data.h5"         # adjust if needed
    weights_path = "tem_attention_unet_best.pth"        # or last
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    test_ds = TEMTestDataset(test_h5_path)

    model = AttentionUNet(in_ch=1, n_classes=5).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"Loaded model weights from {weights_path}")
    print(f"Number of test images: {len(test_ds)}")

    for idx in range(len(test_ds)):
        img, key = test_ds[idx]   # img: (H,W) tensor, key: string
        print(f"[{idx+1}/{len(test_ds)}] Predicting for key = {key}")

        # ---- run prediction ----
        pred = predict_full_image(
            model,
            img_tensor=img,
            patch_size=512,
            stride=256,
            num_classes=5,
            device=device,
        )  # (H,W) int32

        # convert original to numpy for saving
        img_np = img.numpy()

        # ---- save original image ----
        img_min = img_np.min()
        img_range = np.ptp(img_np)  # same as img_np.max() - img_np.min()
        if img_range == 0:
            img_norm = np.zeros_like(img_np, dtype=np.float32)
        else:
            img_norm = (img_np - img_min) / (img_range + 1e-8)

        np.save(os.path.join(output_dir, f"{key}_img.npy"), img_np)
        plt.imsave(os.path.join(output_dir, f"{key}_img.png"), img_norm, cmap="gray")

        # ---- save prediction ----
        np.save(os.path.join(output_dir, f"{key}_pred.npy"), pred)
        plt.imsave(os.path.join(output_dir, f"{key}_pred.png"), pred, cmap="tab20")

    print("Done. Originals and predictions saved in:", output_dir)


if __name__ == "__main__":
    main()

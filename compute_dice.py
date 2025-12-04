import os
import h5py
import numpy as np

NUM_CLASSES = 5
H5_PATH = "data/train/train_data.h5"
PRED_DIR = "predictions"
IGNORE_BACKGROUND = False  # set True if you want to skip class 0


def dice_per_class(pred, target, num_classes=NUM_CLASSES, ignore_background=IGNORE_BACKGROUND, eps=1e-7):
    """
    pred, target: (H, W) integer arrays in [0, num_classes-1]
    Returns: dict {class_id: dice}, and mean Dice over counted classes.
    """
    dices = {}
    classes_to_avg = []

    for c in range(num_classes):
        if ignore_background and c == 0:
            continue

        pred_c = (pred == c)
        target_c = (target == c)

        # Skip if class not present in GT and prediction
        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue

        intersection = np.logical_and(pred_c, target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + eps) / (denom + eps)

        dices[c] = dice
        classes_to_avg.append(dice)

    mean_dice = float(np.mean(classes_to_avg)) if classes_to_avg else 0.0
    return dices, mean_dice


def main():
    h5 = h5py.File(H5_PATH, "r")

    # Your file structure is: f["raw"][k], f["label"][k]
    raw_group = h5["raw"]
    label_group = h5["label"]

    all_image_mean_dices = []
    print(f"Evaluating Dice using labels from: {H5_PATH}")
    print(f"Reading predictions from: {PRED_DIR}")

    # Iterate over image keys inside "raw"
    for key in sorted(raw_group.keys(), key=str):
        if key not in label_group:
            # For train_data.h5 this should not happen, but safe check
            print(f"[WARN] No label for key={key}, skipping.")
            continue

        gt_mask = label_group[key][()]  # (H, W), int labels

        pred_path = os.path.join(PRED_DIR, f"{key}_pred.npy")
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing prediction for key={key}, expected {pred_path}")
            continue

        pred = np.load(pred_path)       # (H, W) int labels

        if pred.shape != gt_mask.shape:
            print(f"[WARN] Shape mismatch for key={key}: pred {pred.shape}, gt {gt_mask.shape}")
            continue

        dices, mean_dice = dice_per_class(pred, gt_mask)
        all_image_mean_dices.append(mean_dice)

        class_str = ", ".join([f"c{c}: {d:.4f}" for c, d in sorted(dices.items())])
        print(f"Key={key}: mean Dice={mean_dice:.4f} | {class_str}")

    if all_image_mean_dices:
        dataset_mean = float(np.mean(all_image_mean_dices))
        print("-" * 60)
        print(f"Dataset mean Dice over {len(all_image_mean_dices)} labeled images: {dataset_mean:.4f}")
    else:
        print("No labeled images were evaluated (check keys / prediction files).")

    h5.close()


if __name__ == "__main__":
    main()

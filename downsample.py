import h5py
import cv2
import numpy as np

SRC_H5 = "data/train/train_data.h5"          # original train H5 (read-only)
DST_H5 = "data/train/train_data_downsampled.h5"          # new file with ONLY 1000x1000 data
TARGET_SIZE = 1000                           # 1000 x 1000

def main():
    with h5py.File(SRC_H5, "r") as src_f, h5py.File(DST_H5, "w") as dst_f:
        raw_src   = src_f["raw"]
        label_src = src_f["label"]

        raw_dst   = dst_f.create_group("raw")
        label_dst = dst_f.create_group("label")

        keys = sorted(raw_src.keys(), key=str)
        print("Found keys:", keys)

        for k in keys:
            img = raw_src[k][()]      # (H, W)
            mask = label_src[k][()]   # (H, W)

            H, W = img.shape
            print(f"Processing key={k}, original size=({H}, {W})")

            # ---- create 1000 x 1000 downsampled version only ----
            img_1000 = cv2.resize(
                img,
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_AREA,      # good for intensity images
            )

            mask_1000 = cv2.resize(
                mask,
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_NEAREST,   # IMPORTANT for labels
            )

            # save under the SAME key name (no original copy)
            raw_dst.create_dataset(k, data=img_1000, compression="gzip")
            label_dst.create_dataset(k, data=mask_1000, compression="gzip")

            print(f"  Saved downsampled {k} with shape {img_1000.shape}")

    print(f"Done. Wrote 1000x1000-only file to {DST_H5}")

if __name__ == "__main__":
    main()

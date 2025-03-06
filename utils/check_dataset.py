import glob
import logging
import os
import sys

import monai
import torch
from matplotlib import pyplot as plt

from utils import get_xforms


def check_dataset(data_folder):
    images = sorted(glob.glob(os.path.join(data_folder, "image", "*.nii.gz")))[:2]
    labels = sorted(glob.glob(os.path.join(data_folder, "label", "*.nii.gz")))[:2]
    logging.info(f"Training: image/label ({len(images)}) folder: {data_folder}")

    keys = ("image", "label")

    # Prepare dataset
    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]

    batch_size = 1
    logging.info(f"Batch size: {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    for check_data in train_loader:
        image, label = check_data["image"][0], check_data["label"][0]
        logging.info(f"Image shape: {image.shape}, Label shape: {label.shape}")

        # Check if the label mask contains non-zero values (non-empty mask)
        for slice_index in range(image.shape[-1] - 1):
            print(f"Checking slice {slice_index}")
            label_slice = label[0, :, :, slice_index]
            print(label_slice.shape)
            if label_slice.max() > 0:  # If there's a non-zero value in the label slice
                logging.info(f"Displaying slice {slice_index} where mask is non-empty.")

                plt.figure("check", (12, 6))

                plt.subplot(1, 2, 1)
                plt.title("Image Slice")
                plt.imshow(image[0, :, :, slice_index], cmap="gray")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Label Slice")
                plt.imshow(label_slice, cmap="gray")
                plt.axis('off')

                plt.show()


if __name__ == "__main__":
    data_folder = r"../datasets"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    check_dataset(data_folder=os.path.join(data_folder, "Train"))

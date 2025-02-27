import glob
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def visualize_image_label_pred(image_path, label_path, pred_path):
    print(image_path, label_path, pred_path)
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    label = sitk.ReadImage(label_path, sitk.sitkUInt8)
    pred = sitk.ReadImage(pred_path, sitk.sitkUInt8)

    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)
    pred_array = sitk.GetArrayFromImage(pred)

    print(image_array.shape, label_array.shape, pred_array.shape)

    for slice_index in range(0, image_array.shape[0], 5):
        image_slice = image_array[slice_index, :, :]
        label_slice = label_array[slice_index, :, :]
        pred_slice = pred_array[slice_index, :, :]

        if np.count_nonzero(label_slice) == 0:
            print(f"Skipping slice {slice_index}, no mask found.")
            continue

        # label_slice = np.ma.masked_equal(label_slice, 0)
        # pred_slice = np.ma.masked_equal(pred_slice, 0)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(image_slice, cmap="gray")
        ax[0].set_title("Image Slice")
        ax[0].axis("off")

        ax[1].imshow(label_slice, cmap="gray")
        ax[1].set_title("Label Slice")
        ax[1].axis("off")

        ax[2].imshow(pred_slice, cmap="gray")
        ax[2].set_title("Prediction Slice")
        ax[2].axis("off")

        plt.show()


if __name__ == "__main__":
    prediction_folder = "../predictions"
    label_folder = "../datasets/Test/label"
    image_folder = "../datasets/Test/image"

    prediction_files = sorted(glob.glob(os.path.join(prediction_folder, "*", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(label_folder, "*.nii.gz")))[:len(prediction_files)]
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.nii.gz")))[:len(prediction_files)]

    for index in range(len(prediction_files)):
        label_path = label_files[index]
        image_path = image_files[index]
        pred_path = prediction_files[index]
        visualize_image_label_pred(image_path, label_path, pred_path)

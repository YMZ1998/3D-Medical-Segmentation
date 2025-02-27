import glob
import os

import monai
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
)


def get_transforms(keys=("pred", "label")):
    transforms = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        Orientationd(keys, axcodes="LPI"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
    ]

    dtype = (torch.uint8, torch.uint8)

    transforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(transforms)


def calculate_metrics(data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dice_scores = []
    iou_scores = []
    hd_scores = []

    with torch.no_grad():
        for data in data_loader:
            pred, label = data["pred"].to(device), data["label"].to(device)

            pred = (pred > 0.5).float()

            dice = dice_metric(pred, label)
            dice_scores.append(dice.item())

            iou = iou_metric(pred, label)
            iou_scores.append(iou.item())

            hd = hd_metric(pred, label)
            hd_scores.append(hd.item())
            print(f"Dice Score: {dice.item():.4f}, IoU: {iou.item():.4f}, HD: {hd.item():.4f}")

    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Average HD: {np.mean(hd_scores):.4f} ± {np.std(hd_scores):.4f}")


if __name__ == "__main__":
    prediction_folder = "../predictions"
    label_folder = "../datasets/Test/label"

    prediction_files = sorted(glob.glob(os.path.join(prediction_folder, "*", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(label_folder, "*.nii.gz")))[:len(prediction_files)]

    assert len(prediction_files) == len(label_files), "Prediction and label files do not match in number."

    files = [{"pred": pred, "label": lbl} for pred, lbl in zip(prediction_files, label_files)]

    print(files)

    transform = get_transforms(("pred", "label"))

    dataset = Dataset(data=files, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)

    calculate_metrics(data_loader)

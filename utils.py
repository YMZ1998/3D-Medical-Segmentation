import os
import shutil
import warnings

import monai
import torch
import torch.nn as nn
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ignite")


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (torch.float32, torch.uint8)
    if mode == "val":
        dtype = (torch.float32, torch.uint8)
    if mode == "infer":
        dtype = (torch.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)


def get_net(model_name="dynunet"):
    """returns a unet model instance."""

    num_classes = 2
    if model_name == "dynunet":
        from monai.networks.nets import DynUNet
        net = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            dropout=0.1,
        )
    elif model_name == "unet":
        net = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
        )
    else:
        raise ValueError(f"model_name {model_name} not supported")

    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

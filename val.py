import argparse
import glob
import logging
import os
import sys

import monai
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import RandGaussianNoised

from parse_args import parse_args, get_net, get_device
from utils.utils import remove_and_create_dir, get_xforms, get_inferer


def test(args):
    data_folder = os.path.join(args.data_folder, "Val")
    remove_and_create_dir(args.prediction_folder)

    # Load the checkpoint
    ckpts = sorted(glob.glob(os.path.join(args.model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info(f"using {ckpt}.")

    device = get_device()

    net = get_net(args)
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    net.load_state_dict(checkpoint["net"])
    net.eval()

    # Load the images and labels
    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "image", "*.nii.gz")))[:]
    labels = sorted(glob.glob(os.path.join(image_folder, "label", "*.nii.gz")))[:]
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img, "label": label} for img, label in zip(images, labels)]

    # Define the transformations
    keys = ("image", "label")
    infer_transforms = get_xforms("val", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.transforms.SaveImage(output_dir=args.prediction_folder, output_postfix='seg', mode="nearest",
                                       resample=True, output_dtype="int8")

    # Initialize the Dice Metric for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

    dice_scores = []
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image'].meta['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)

            # Apply test time augmentations (TTA)
            n = 1.0
            if args.tta:
                for i in range(4):
                    print(i)
                    _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                    pred = inferer(_img.to(device), net)
                    preds = preds + pred
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                        pred = torch.flip(flip_pred, dims=dims)
                        preds = preds + pred
                        n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()

            # Compute Dice Coefficient
            dice = dice_metric(y_pred=preds, y=infer_data["label"].to(device))
            dice_scores.append(dice.item())

            for p in preds:  # save each image+metadata in the batch respectively
                saver(p)

    logging.info(f"Average Dice Score: {np.mean(dice_scores):.4f}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    test(args)

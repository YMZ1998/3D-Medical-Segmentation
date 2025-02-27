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

from utils.utils import remove_and_create_dir, get_xforms, get_net, get_inferer


def infer(data_folder, model_folder, prediction_folder,tta=False):
    remove_and_create_dir(prediction_folder)

    # Load the checkpoint
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = get_net().to(device)
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
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.transforms.SaveImage(output_dir=prediction_folder, output_postfix='seg', mode="nearest",
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
            if tta:
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

    # Copy the saved segmentations into the required folder structure for submission
    # submission_dir = os.path.join(prediction_folder, "to_submit")
    # if not os.path.exists(submission_dir):
    #     os.makedirs(submission_dir)
    # files = glob.glob(os.path.join(prediction_folder, "*", "*.nii.gz"))
    # for f in files:
    #     to_name = os.path.join(submission_dir, os.path.basename(f))
    #     shutil.copy(f, to_name)
    # logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument("--data_folder", default=r"./datasets", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./checkpoints", type=str, help="model folder")
    parser.add_argument("--tta", default=False, type=bool, help="TTA")
    args = parser.parse_args()

    # monai.config.print_config()
    # monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    infer(data_folder=os.path.join(args.data_folder, "Test"), model_folder=args.model_folder,
          prediction_folder="./predictions", tta=args.tta)

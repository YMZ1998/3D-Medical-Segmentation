import glob
import logging
import os
import shutil
import sys

import monai
import torch
from monai.transforms import RandGaussianNoised

from parse_args import parse_args, get_net, get_device
from utils.utils import remove_and_create_dir, get_xforms, get_inferer


def infer(args):
    data_folder = os.path.join(args.data_folder, "Test")
    remove_and_create_dir(args.prediction_folder)

    # Load the checkpoint
    ckpts = sorted(glob.glob(os.path.join(args.model_folder, "*.pt")))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in model folder.")
    ckpt = ckpts[-1]
    logging.info(f"Using checkpoint: {ckpt}")

    device = get_device()
    net = get_net(args)
    checkpoint = torch.load(ckpt, map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.eval()

    # Load the images
    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "image", "*.nii.gz")))
    logging.info(f"Inferencing {len(images)} images from {data_folder}")
    infer_files = [{"image": img} for img in images]

    # Define transformations
    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
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

    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"Segmenting {infer_data['image'].meta['filename_or_obj']}")
            preds = inferer(infer_data["image"].to(device), net)

            # Apply test time augmentations (TTA)
            n = 1.0
            if args.tta:
                for i in range(4):
                    _img = RandGaussianNoised("image", prob=1.0, std=0.01)(infer_data)["image"]
                    pred = inferer(_img.to(device), net)
                    preds += pred
                    n += 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                        pred = torch.flip(flip_pred, dims=dims)
                        preds += pred
                        n += 1.0
            preds /= n
            preds = (preds.argmax(dim=1, keepdims=True)).float()

            for p in preds:  # Save each segmented image
                saver(p)
    # Copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(args.prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(args.prediction_folder, "*", "*.nii.gz"))
    for f in files:
        to_name = os.path.join(submission_dir, os.path.basename(f))
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    infer(args)

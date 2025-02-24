# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import shutil
import sys
import warnings

import monai
import torch
from monai.transforms import (
    RandGaussianNoised,
)

from utils import remove_and_create_dir
from train import get_xforms, get_net

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")


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


def infer(data_folder, model_folder, prediction_folder):
    remove_and_create_dir(prediction_folder)

    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "image", "*.nii.gz")))[:2]
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.transforms.SaveImage(output_dir=prediction_folder, output_postfix='seg', mode="nearest",
                                       resample=True, output_dtype="int8")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image'].meta['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for i in range(2):
                print(i)
                # test time augmentations
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
            for p in preds:  # save each image+metadata in the batch respectively
                saver(p)

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "*", "*.nii.gz"))
    for f in files:
        to_name = os.path.join(submission_dir, os.path.basename(f))
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument("--data_folder", default=r"./datasets", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./checkpoints", type=str, help="model folder")
    args = parser.parse_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    infer(data_folder=os.path.join(args.data_folder, "Test"), model_folder=args.model_folder,
          prediction_folder="./predictions")

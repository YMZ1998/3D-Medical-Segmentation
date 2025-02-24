import argparse
import glob
import logging
import os
import sys

import monai
import torch
from ignite.contrib.handlers import ProgressBar
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.transforms import (
    AsDiscreted,
)

from utils import get_xforms, get_net, get_inferer, DiceCELoss


def train(data_folder, model_folder, resume=False):
    """run a training pipeline."""

    os.makedirs(model_folder, exist_ok=True)

    # Images and labels
    images = sorted(glob.glob(os.path.join(data_folder, "image", "*.nii.gz")))[:]
    labels = sorted(glob.glob(os.path.join(data_folder, "label", "*.nii.gz")))[:]
    logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

    # Other parameters
    amp = False  # auto mixed precision
    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images))
    n_val = min(len(images) - n_train, int(val_frac * len(images)))
    logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # Create a training data loader
    batch_size = 2
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    # Create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    # Create Model, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = get_net().to(device)
    max_epochs, lr, momentum = 500, 1e-4, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Load pre-trained weights if available
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    checkpoint_path = ckpts[-1]
    if resume:
        logging.info(f"Loading pre-trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["optimizer"])
        logging.info(f"Resuming training.")

    # Create evaluator (to be used to measure model quality during training)
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2)]
    )

    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={
            'net': net,
            'optimizer': opt,
        }, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        postprocessing=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"])),
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # Evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument("--data_folder", default=r"./datasets", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./checkpoints", type=str, help="model folder")
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint')
    args = parser.parse_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(data_folder=os.path.join(args.data_folder, "Train"), model_folder=args.model_folder, resume=args.resume)

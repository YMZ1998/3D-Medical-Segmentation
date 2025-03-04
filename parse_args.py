import argparse

import torch
from monai.networks.nets import DynUNet, UNet, UNETR


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def get_net(args):
    print('★' * 30)
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'image size:{args.image_size}\n'
          f'batch size:{args.batch_size}\n'
          f'num classes:{args.num_classes}')
    print('★' * 30)
    device = get_device()

    if args.arch == "dynunet":
        net = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.num_classes,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            dropout=0.1,
        )
        # net = DynUNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=args.num_classes,
        #     kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        #     strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        #     upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        #     res_block=True,
        #     norm_name="batch",
        #     dropout=0.1,
        # )
    elif args.arch == "unet":
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif args.arch == 'unetr':
        net = UNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.num_classes,
            img_size=args.image_size,
            feature_size=16,
            proj_type="conv",
            norm_name="instance",
            dropout_rate=0.1,
        )
    else:
        raise ValueError(f"model_name {args.model_name} not supported")

    return net.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dynunet', help='unet/dynunet/unetr')
    parser.add_argument("--data_folder", default=r"./datasets", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./checkpoints", type=str, help="model folder")
    parser.add_argument("--prediction_folder", default="./predictions", type=str, help="prediction folder")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint')

    parser.add_argument("--image_size", default=(256, 256, 16), type=tuple, help="image size")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to train")

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--tta", default=False, type=bool, help="TTA")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    from torchsummary import summary

    args = parse_args()

    model = get_net(args)
    summary(model, (1, args.image_size[0], args.image_size[1], args.image_size[2]))

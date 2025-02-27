import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dynunet', help='unet/dynunet')
    parser.add_argument("--data_folder", default=r"./datasets/Train", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./checkpoints", type=str, help="model folder")
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint')

    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=500, type=int, metavar="N", help="number of total epochs to train")

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
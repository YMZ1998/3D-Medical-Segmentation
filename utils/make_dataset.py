import os
import shutil

from utils.utils import remove_and_create_dir


def prepare_dataset(src: str, dst: str, modes: list):
    """
    复制医学影像数据集到目标目录。

    :param src: 源数据集路径
    :param dst: 目标数据集路径
    :param modes: 数据集划分模式列表，如["Train", "Val", "Test"]
    """

    remove_and_create_dir(dst)

    for mode in modes:
        src_dir = os.path.join(src, mode)
        dst_dir = os.path.join(dst, mode)
        os.makedirs(os.path.join(dst_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "label"), exist_ok=True)

        for i, p in enumerate(os.listdir(src_dir), start=1):
            image_dir = os.path.join(src_dir, p, "image.nii.gz")
            label_dir = os.path.join(src_dir, p, "CTV.nii.gz")
            print(f"Processing {i}: {image_dir}, {label_dir}")

            dst_image_dir = os.path.join(dst_dir, "image", f"{i:03d}.nii.gz")
            dst_label_dir = os.path.join(dst_dir, "label", f"{i:03d}.nii.gz")
            shutil.copyfile(image_dir, dst_image_dir)

            if mode != "Test":
                shutil.copyfile(label_dir, dst_label_dir)


if __name__ == "__main__":
    src_path = r"D:\\Data\\CTV_Seg"
    dst_path = r"../datasets"
    modes_list = ["Train", "Val", "Test"]
    prepare_dataset(src_path, dst_path, modes_list)

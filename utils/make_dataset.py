import os
import shutil

from utils import remove_and_create_dir

if __name__ == "__main__":
    src = r"D:\Data\CTV_Seg"
    dst = r"../datasets"

    remove_and_create_dir(dst)

    for mode in ["Train", "Test"]:
        src_dir = os.path.join(src, mode)
        dst_dir = os.path.join(dst, mode)
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "label"), exist_ok=True)

        i = 0
        for p in os.listdir(src_dir):
            i = i + 1
            image_dir = os.path.join(src, mode, p, "image.nii.gz")
            label_dir = os.path.join(src, mode, p, "CTV.nii.gz")
            print(i, image_dir, label_dir)

            dst_image_dir = os.path.join(dst_dir, "image", str(i).rjust(3, "0") + ".nii.gz")
            dst_label_dir = os.path.join(dst_dir, "label", str(i).rjust(3, "0") + ".nii.gz")
            shutil.copyfile(image_dir, dst_image_dir)
            shutil.copyfile(label_dir, dst_label_dir)

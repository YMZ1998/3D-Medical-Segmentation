import os
import SimpleITK as sitk
import nibabel as nib

def check_nii_gz_files(directory):
    nii_files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    if not nii_files:
        print("目录中没有 '.nii.gz' 文件。")
        return

    for nii_file in nii_files:
        file_path = os.path.join(directory, nii_file)

        try:
            # img = sitk.ReadImage(file_path)
            img = nib.load(file_path)
            print(f"{nii_file} 是有效的 NIfTI 文件。")
        except Exception as e:
            print(f"{nii_file} 无法读取，可能是损坏的文件。错误信息：{e}")

if __name__ == "__main__":
    directory = "./datasets"
    for mode in ["Train", "Test"]:
        image_dir = os.path.join(directory, mode, "image")
        check_nii_gz_files(image_dir)
        label_dir = os.path.join(directory, mode, "label")
        check_nii_gz_files(label_dir)

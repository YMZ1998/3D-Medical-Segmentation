import glob
import os
import shutil
from datetime import datetime


def backup_files(source_dir, backup_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")

    os.makedirs(backup_path, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(source_dir, "*.pt")))
    for filename in ckpts:
        source_file = os.path.join(source_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, backup_path)

    print(f"backup: {backup_path}")


if __name__ == "__main__":
    source_dir = "../checkpoints"
    backup_dir = "../checkpoints"
    backup_files(source_dir, backup_dir)

import shutil
import urllib.request
import zipfile
import os
from tqdm import tqdm

import torch


def split_train_val(src_dir):
    split_dir = os.path.join(
        src_dir,
        "SPLObjDetectDatasetV2",
        "trainval",
    )
    image_dir = os.path.join(split_dir, "images")
    images = os.listdir(image_dir)
    index = torch.randperm(len(images))
    train_index = index[int(0.2 * len(images)) :]
    val_index = index[: int(0.2 * len(images))]

    train_files = [images[idx][:-3] for idx in train_index]
    val_files = [images[idx][:-3] for idx in val_index]
    return train_files, val_files


def move_files(src_dir, split, image_dir, label_dir, files):
    for file in tqdm(files, desc=f"Moving {split} files"):
        shutil.move(
            os.path.join(image_dir, file + "png"),
            os.path.join(src_dir, "raw", split, "images", file + "png"),
        )
        shutil.move(
            os.path.join(label_dir, file + "txt"),
            os.path.join(src_dir, "raw", split, "labels", file + "txt"),
        )


def download_raw_data(url, dir_name):
    response = urllib.request.urlopen(url)
    file_size = int(response.headers["Content-Length"])
    block_size = 8192
    progress_bar = tqdm(
        total=file_size,
        unit="iB",
        unit_scale=True,
        desc="Downloading SPLObjDetectDatasetV2.zip",
    )
    with open(os.path.join(dir_name, "SPLObjDetectDatasetV2.zip"), "wb") as f:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            progress_bar.update(len(buffer))
    progress_bar.close()


def main():
    url = "https://roboeireann.maynoothuniversity.ie/research/SPLObjDetectDatasetV2.zip"
    dir_name = "data"

    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Download the file from the URL
    print("Downloading...")
    download_raw_data(url, dir_name)
    print("\nDownload complete.")

    # Extract the contents of the zip file
    print("Extracting...")
    zip_file = zipfile.ZipFile(os.path.join(dir_name, "SPLObjDetectDatasetV2.zip"))
    zip_file.extractall(dir_name)
    print("Extraction complete.")

    


if __name__ == "__main__":
    main()

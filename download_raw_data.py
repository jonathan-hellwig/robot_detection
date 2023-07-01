from pathlib import Path
import argparse
import shutil
import urllib.request
import zipfile
import os
from tqdm import tqdm

import torch


def split_train_val(src_dir):
    split_dir = src_dir / "SPLObjDetectDatasetV2" / "trainval"
    image_dir = split_dir / "images"
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
            image_dir / (file + "png"),
            src_dir / "raw" / split / "images" / (file + "png"),
        )
        shutil.move(
            label_dir / (file + "txt"),
            src_dir / "raw" / split / "labels" / (file + "txt"),
        )


def download_raw_data(url, dir_name: Path):
    response = urllib.request.urlopen(url)
    file_size = int(response.headers["Content-Length"])
    block_size = 8192
    progress_bar = tqdm(
        total=file_size,
        unit="iB",
        unit_scale=True,
        desc="Downloading SPLObjDetectDatasetV2.zip",
    )
    with open(dir_name / "SPLObjDetectDatasetV2.zip", "wb") as f:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            progress_bar.update(len(buffer))
    progress_bar.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="data")
    args = argparser.parse_args()
    data_path = Path(args.data_path)

    data_url = (
        "https://roboeireann.maynoothuniversity.ie/research/SPLObjDetectDatasetV2.zip"
    )

    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Download the file from the URL
    print("Downloading...")
    download_raw_data(data_url, data_path)
    print("\nDownload complete.")

    # Extract the contents of the zip file
    print("Extracting...")
    zip_file = zipfile.ZipFile(data_path / "SPLObjDetectDatasetV2.zip")
    zip_file.extractall(data_path)
    print("Extraction complete.")

    # Define the directories to create
    dirs = [
        "train" + os.sep + "images",
        "train" + os.sep + "labels",
        "val" + os.sep + "images",
        "val" + os.sep + "labels",
        "test" + os.sep + "images",
        "test" + os.sep + "labels",
    ]

    # Create the directories
    for d in dirs:
        os.makedirs(os.path.join(data_path, "raw", d), exist_ok=True)

    # Split the data into train, val and test
    train_files, val_files = split_train_val(data_path)

    # Move the files into the correct directories
    move_files(
        data_path,
        "train",
        data_path / "SPLObjDetectDatasetV2" / "trainval" / "images",
        data_path / "SPLObjDetectDatasetV2" / "trainval" / "labels",
        train_files,
    )
    move_files(
        data_path,
        "val",
        data_path / "SPLObjDetectDatasetV2" / "trainval" / "images",
        data_path / "SPLObjDetectDatasetV2" / "trainval" / "labels",
        val_files,
    )
    test_files = os.listdir(data_path / "SPLObjDetectDatasetV2" / "test" / "images")
    move_files(
        data_path,
        "test",
        data_path / "SPLObjDetectDatasetV2" / "test" / "images",
        data_path / "SPLObjDetectDatasetV2" / "test" / "labels",
        [file[:-3] for file in test_files],
    )
    shutil.rmtree(data_path / "SPLObjDetectDatasetV2")
    os.remove(data_path / "SPLObjDetectDatasetV2.zip")

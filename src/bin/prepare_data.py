"""Download all the data needed in the project
Place `CelebAMask-HQ.zip` file in the project root and run this script with python:
`python src/bin/prepare_data.py`
"""

from src.utils.config import ROOT, DS_ROOT
from src.utils.files import unzip_zip
from pathlib import Path
import os
import pandas as pd
from shutil import copyfile as cp
import png
import cv2
import numpy as np
from tqdm.auto import tqdm


ZIP_FILENAME = Path("CelebAMask-HQ.zip")
dirname = ZIP_FILENAME.stem

SRC_ZIP_FILEPATH = ROOT / ZIP_FILENAME


DS_PATH = DS_ROOT / dirname

IMGS_PATH = str(DS_PATH / "CelebA-HQ-img")
MASKS_PATH = DS_PATH / "CelebA-HQ-mask"
MASKS_PATH.mkdir(exist_ok=True, parents=True)
MASKS_PATH = str(MASKS_PATH)


PALETTE = [
    [0, 0, 0],
    [204, 0, 0],
    [76, 153, 0],
    [204, 204, 0],
    [51, 51, 255],
    [204, 0, 204],
    [0, 255, 255],
    [255, 204, 204],
    [102, 51, 0],
    [255, 0, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [0, 0, 204],
    [255, 51, 153],
    [0, 204, 204],
    [0, 51, 0],
    [255, 153, 51],
    [0, 204, 0],
]

LABELS = [
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]


def save_gray_array_as_color_png(array: np.ndarray, filename: str):
    height, width = array.shape
    w = png.Writer(width, height, palette=PALETTE, bitdepth=8)
    f = open(filename, "wb")
    w.write(f, array.tolist())


def save_masks_on_single_image(filename_base, filename_save):
    im_base = np.zeros((512, 512), dtype=np.uint8)
    for idx, label in enumerate(LABELS):
        filename = os.path.join(filename_base + label + ".png")
        if os.path.exists(filename):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = idx + 1
    save_gray_array_as_color_png(im_base, filename_save)


def process_masks():
    folder_base = DS_PATH / "CelebAMask-HQ-mask-anno"
    folder_base.mkdir(exist_ok=True, parents=True)

    folder_save = MASKS_PATH
    img_num = 30_000

    for k in tqdm(range(img_num), desc="Processing masks"):
        folder_num = k // 2000
        filename_base = os.path.join(str(folder_base), str(folder_num), str(k).rjust(5, "0") + "_")
        filename_save = os.path.join(str(folder_save), str(k) + ".png")
        save_masks_on_single_image(filename_base, filename_save)


def prepare_splits():
    #### source data path
    src_masks = MASKS_PATH
    src_imgs = IMGS_PATH

    # destination paths
    dst_masks = str(DS_PATH / "masks")
    dst_imgs = str(DS_PATH / "images")

    #### destination training data path
    dest_train_masks = f"{dst_masks}/train"
    dest_train_imgs = f"{dst_imgs}/train"

    #### destination testing data path
    dest_test_masks = f"{dst_masks}/test"
    dest_test_imgs = f"{dst_imgs}/test"

    #### val data path
    dest_val_masks = f"{dst_masks}/val"
    dest_val_imgs = f"{dst_imgs}/val"

    for path in [
        dest_train_masks,
        dest_train_imgs,
        dest_test_masks,
        dest_test_imgs,
        dest_val_masks,
        dest_val_imgs,
    ]:
        Path(path).mkdir(exist_ok=True, parents=True)

    idxs = pd.read_csv(DS_PATH / "CelebA-HQ-to-CelebA-mapping.txt", delim_whitespace=True, header=0)

    for i, orig_idx in tqdm(
        enumerate(idxs["orig_idx"].values), desc="Copying files", total=len(idxs)
    ):
        idx = str(i)
        if orig_idx >= 162771 and orig_idx < 182638:
            cp(os.path.join(src_masks, idx + ".png"), os.path.join(dest_val_masks, idx + ".png"))
            cp(os.path.join(src_imgs, idx + ".jpg"), os.path.join(dest_val_imgs, idx + ".jpg"))

        elif orig_idx >= 182638:
            cp(os.path.join(src_masks, idx + ".png"), os.path.join(dest_test_masks, idx + ".png"))
            cp(os.path.join(src_imgs, idx + ".jpg"), os.path.join(dest_test_imgs, idx + ".jpg"))
        else:
            cp(os.path.join(src_masks, idx + ".png"), os.path.join(dest_train_masks, idx + ".png"))
            cp(os.path.join(src_imgs, idx + ".jpg"), os.path.join(dest_train_imgs, idx + ".jpg"))


if __name__ == "__main__":
    unzip_zip(SRC_ZIP_FILEPATH, DS_ROOT)
    process_masks()
    prepare_splits()

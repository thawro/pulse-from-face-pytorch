"""Transforms for segmentation task"""

import albumentations as A
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
from typing import Callable
from torch import Tensor
import torch

_mean_std = float | tuple[float, float, float]
_transform = Callable[[np.ndarray], Tensor] | A.Compose


class CelebATransform:
    orig_masks_size = 512

    def __init__(
        self,
        imgsz: int = 256,
        mean: _mean_std = (0.485, 0.456, 0.406),
        std: _mean_std = (0.229, 0.224, 0.225),
    ):
        self.imgsz = imgsz
        self.mean = np.array(mean)
        self.std = np.array(std)

    @property
    def train(self) -> _transform:
        return A.Compose(
            [
                A.Resize(self.orig_masks_size, self.orig_masks_size),
                A.RandomScale((-0.5, 1.0), p=0.8),
                A.Rotate((-10, 10), p=0.8),
                A.GaussianBlur(p=0.5),
                A.PadIfNeeded(
                    min_height=self.imgsz, min_width=self.imgsz, border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomCrop(height=self.imgsz, width=self.imgsz),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )

    @property
    def inference(self) -> _transform:
        return A.Compose(
            [
                # A.LongestMaxSize(self.imgsz),
                # A.PadIfNeeded(self.imgsz, self.imgsz, value=0),
                A.SmallestMaxSize(self.imgsz),
                A.CenterCrop(self.imgsz, self.imgsz),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )

    @property
    def process(self):
        def transform(image: np.ndarray):
            h, w = image.shape[:2]
            aspect_ratio = h / w

            # SmallestMaxSize
            if h > w:
                new_w = int(self.imgsz)
                new_h = int(new_w * aspect_ratio)
            else:
                new_h = int(self.imgsz)
                new_w = int(new_h / aspect_ratio)
            _image = cv2.resize(image, (new_w, new_h))

            # CenterCrop
            crop_size = int(self.imgsz)
            crop_y = (new_h - crop_size) // 2
            crop_x = (new_w - crop_size) // 2
            if crop_x == 0:
                xmin, xmax = 0, new_w
                ymin = crop_y
                ymax = ymin + self.imgsz
            elif crop_y == 0:
                xmin = crop_x
                xmax = xmin + self.imgsz
                ymin, ymax = 0, new_h
            _image = _image[ymin:ymax, xmin:xmax]

            # Normalize
            _image = (_image - self.mean * 255) / (self.std * 255)

            _image = torch.from_numpy(_image).permute(2, 0, 1).float()
            resize_size = (new_w, new_h)
            crop_coords = (xmin, ymin, xmax, ymax)
            return _image, resize_size, crop_coords

        return transform

    @property
    def inverse_preprocessing(self) -> Callable[[np.ndarray], np.ndarray]:
        """Apply inverse of preprocessing to the image (for visualization purposes)."""

        def transform(image: np.ndarray | Tensor) -> np.ndarray:
            if isinstance(image, Tensor):
                image = image.permute(1, 2, 0).numpy()
            _image = (image * np.array(self.std)) + np.array(self.mean)
            return (_image * 255).astype(np.uint8)

        return transform

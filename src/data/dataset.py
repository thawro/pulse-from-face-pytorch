"""Dataset classes"""

import torchvision.datasets
import albumentations as A
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import torch
from torch import Tensor


class BaseDataset(torchvision.datasets.VisionDataset):
    root: Path

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: A.Compose | None = None,
        target_transform: A.Compose | None = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.root = Path(root)


class CelebAMaskDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        num_classes: int,
        split: str = "test",
        transform: A.Compose | None = None,
        target_transform: A.Compose | None = None,
    ):
        super().__init__(root, split, transform=transform, target_transform=target_transform)
        self.num_classes = num_classes
        self.images_path = self.root / "images" / self.split
        self.masks_path = self.root / "masks" / self.split
        self.image_filepaths = glob.glob(f"{self.images_path}/*")
        self.mask_filepaths = [
            img_file.replace("images/", "masks/").replace(f".jpg", ".png")
            for img_file in self.image_filepaths
        ]

    def get_raw_data(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return raw image and mask (without preprocessing/transforms applied)"""
        image_filepath = self.image_filepaths[idx]
        mask_filepath = self.mask_filepaths[idx]

        image = np.array(Image.open(image_filepath).convert("RGB"))
        mask = np.array(Image.open(mask_filepath))  # .convert("L"))
        return image, mask

    def __len__(self) -> int:
        return len(self.mask_filepaths)

    def _transform(self, image: np.ndarray, mask: np.ndarray) -> tuple[Tensor, Tensor]:
        """Apply transform to image and mask"""
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            _image = transformed["image"]
            _mask = Tensor(transformed["mask"])
        else:
            _image = torch.from_numpy(image)
            _mask = torch.from_numpy(mask)
        _mask = _mask.long()
        return _image, _mask

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return transformed image and mask"""
        image, mask = self.get_raw_data(idx)

        image, mask = self._transform(image, mask)
        # create One-Hot y_cls for auxiliary loss
        cls_mask = torch.clone(mask)
        cls_mask[cls_mask == 255] = 0  # remove void label
        onehot_class_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        unique = cls_mask.unique().long()
        onehot_class_tensor.put_(unique, torch.ones_like(unique, dtype=torch.float32))

        return image, mask, onehot_class_tensor
